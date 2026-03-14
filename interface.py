import streamlit as st
import tempfile
import os
import shutil 
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# Partie 1 : Configuration visuelle de la page
st.set_page_config(
    page_title="ChatDoc Pro | IA Privée", 
    page_icon="🛡️", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Partie 2 : Initialisation des variables
# On garde en mémoire la base de données et l'historique des messages
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Partie 3 : Barre latérale (Menu de gauche)
with st.sidebar:
    st.title("⚙️ Centre de contrôle")
    st.caption("Gérez vos documents et vos données.")
    st.divider()
    
    st.header("📂 1. Vos Documents")
    # On permet de charger plusieurs types de fichiers en même temps
    fichiers_upload = st.file_uploader(
        "Glissez vos fichiers ici", 
        type=["pdf", "txt", "docx", "csv"], 
        accept_multiple_files=True
    )

    # Si on a des fichiers et que la base n'est pas encore créée
    if fichiers_upload and st.session_state.vectordb is None:
        if st.button("🚀 Lancer l'analyse", use_container_width=True, type="primary"):
            
            with st.status("📥 Traitement de vos documents...", expanded=True) as status:
                st.write("Lecture et extraction des données...")
                toutes_les_pages = []
                
                # On analyse chaque fichier et on utilise le bon outil de lecture selon l'extension
                # On analyse chaque fichier et on utilise le bon outil de lecture
                for fichier in fichiers_upload:
                    extension = os.path.splitext(fichier.name)[1].lower()
                    nom_original = fichier.name # <-- On garde le vrai nom de côté !
                    
                    with tempfile.NamedTemporaryFile(delete=False, suffix=extension) as fichier_temp:
                        fichier_temp.write(fichier.getvalue())
                        chemin_temp = fichier_temp.name
                    
                    if extension == ".pdf":
                        loader = PyPDFLoader(chemin_temp)
                    elif extension == ".docx":
                        loader = Docx2txtLoader(chemin_temp)
                    elif extension == ".txt":
                        loader = TextLoader(chemin_temp, encoding="utf-8")
                    elif extension == ".csv":
                        loader = CSVLoader(chemin_temp, encoding="utf-8")
                    else:
                        continue 
                        
                    # On charge les pages du document
                    documents_charges = loader.load()
                    
                    # L'ASTUCE EST ICI : On remplace le nom moche par le vrai nom dans la mémoire de l'IA !
                    for doc in documents_charges:
                        doc.metadata['source'] = nom_original
                        
                    toutes_les_pages.extend(documents_charges)

                st.write("Découpage analytique du texte...")
                decoupeur = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                morceaux = decoupeur.split_documents(toutes_les_pages)

                st.write("Création de la mémoire vectorielle...")
                modele_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # On sauvegarde la base pour ne pas tout recalculer si l'appli s'endort
                st.session_state.vectordb = Chroma.from_documents(
                    documents=morceaux, 
                    embedding=modele_embedding,
                    persist_directory="./chroma_sauvegarde" 
                )

                status.update(label="✅ Documents mémorisés avec succès !", state="complete", expanded=False)

    st.divider()
    st.header("🧰 2. Options")
    
    # Bouton pour télécharger la conversation au format texte
    if len(st.session_state.messages) > 0:
        texte_export = "--- Rapport généré par ChatDoc Pro ---\n\n"
        for msg in st.session_state.messages:
            role = "👤 Utilisateur" if msg["role"] == "user" else "🤖 IA"
            texte_export += f"{role} :\n{msg['content']}\n\n"
        
        st.download_button(
            label="💾 Exporter le compte-rendu (.txt)",
            data=texte_export,
            file_name="rapport_ia.txt",
            mime="text/plain",
            use_container_width=True
        )

    # Nettoyer seulement la discussion (on garde les PDF en mémoire)
    if st.button("🧹 Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    # Tout remettre à zéro (Discussion + PDF)
    # Le bouton Reset total (Conversation + Documents)
    if st.button("🗑️ Vider la mémoire totale et recommencer", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vectordb = None
        
        # ON DÉTRUIT LE DOSSIER PHYSIQUE !
        if os.path.exists("./chroma_sauvegarde"):
            shutil.rmtree("./chroma_sauvegarde")
            
        st.rerun()

# Partie 4 : Zone principale (Le Chat)
st.title("🛡️ ChatDoc Pro")
st.markdown("##### *Discutez avec vos documents professionnels en toute confidentialité.*")

# Écran d'accueil quand c'est vide
if not st.session_state.messages:
    st.write("") 
    st.info("👋 **Bienvenue !** Pour commencer, veuillez charger vos documents (PDF, Word, TXT, CSV) sur votre gauche.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("🔒 **100% Privé**\nAucune donnée n'est envoyée sur internet. Tout tourne sur votre machine.")
    with col2:
        st.warning("📚 **Multi-Documents**\nCroisez les informations de plusieurs PDF, Word ou Excel simultanément.")
    with col3:
        st.info("🧠 **IA Avancée**\nPropulsé par LLaMA 3.3 (Groq) optimisé pour la précision.")

# On affiche les anciens messages avec de beaux emojis
for message in st.session_state.messages:
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Quand l'utilisateur tape une question
if question := st.chat_input("Posez votre question sur les documents..."):
    
    # Sécurité : on vérifie qu'il y a bien un document chargé
    if st.session_state.vectordb is None:
        st.error("⚠️ Veuillez d'abord analyser un document via le menu de gauche.")
    else:
        # On affiche la question de l'utilisateur
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Recherche dans les documents et rédaction..."):
                
                # 1. On cherche les infos dans nos documents
                resultats = st.session_state.vectordb.similarity_search(question, k=4)
                contexte_trouve = "\n\n".join([doc.page_content for doc in resultats])

                # 2. On configure l'IA
                llm = ChatGroq(
                    groq_api_key=st.secrets["GROQ_API_KEY"], 
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.3
                )
                
                # 3. ON RESTAURE LA MÉMOIRE CONVERSATIONNELLE (Ce que j'avais oublié !)
                # On récupère les 4 derniers messages pour que l'IA comprenne le contexte
                historique_recent = ""
                if len(st.session_state.messages) > 1:
                    historique_recent = "Rappel de nos derniers échanges :\n"
                    for msg in st.session_state.messages[-4:]:
                        role = "Utilisateur" if msg["role"] == "user" else "IA"
                        historique_recent += f"- {role} a dit : {msg['content']}\n"
                
                # 4. On extrait proprement le nom du fichier et la page exacte pour nos sources
                sources_precises = set()
                for doc in resultats:
                    source_chemin = doc.metadata.get('source', 'Inconnu')
                    nom_fichier = os.path.basename(source_chemin)
                    page = doc.metadata.get('page')
                    
                    if page is not None:
                        sources_precises.add(f"*{nom_fichier}* (Page {int(page) + 1})")
                    else:
                        sources_precises.add(f"*{nom_fichier}*")
                
                # 5. On envoie tout ça à l'IA avec notre prompt strict
                prompt = f"""
                Tu es ChatDoc Pro. Ton rôle est de répondre de manière SYNTHÉTIQUE et PRÉCISE.
                
                RÈGLES :
                1. RÉSUMÉ : Utilise 3 à 4 puces maximum pour aller à l'essentiel.
                2. STYLE : Sois direct, professionnel et humain.
                3. SOURCE : Utilise uniquement le contexte fourni.

                {historique_recent}

                CONTEXTE :
                {contexte_trouve}

                QUESTION : {question}

                RÉPONSE RÉSUMÉE :
                """
                
                reponse_brute = llm.invoke(prompt)
                texte_final = reponse_brute.content
                
                # On affiche la réponse de l'IA
                st.markdown(texte_final)
                
                # On affiche les sources cliquables juste en dessous
                if sources_precises:
                    st.caption(f"🎯 **Citations exactes :** {', '.join(sources_precises)}")
                
                # On sauvegarde la réponse dans l'historique
                st.session_state.messages.append({"role": "assistant", "content": texte_final})