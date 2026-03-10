import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

# ==========================================
# 1. CONFIGURATION VISUELLE DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="ChatDoc Pro | IA Privée", 
    page_icon="🛡️", 
    layout="wide", # Utilise toute la largeur de l'écran
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. INITIALISATION DE LA MÉMOIRE
# ==========================================
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ==========================================
# 3. BARRE LATÉRALE (Interface de contrôle)
# ==========================================
with st.sidebar:
    st.title("⚙️ Centre de contrôle")
    st.caption("Gérez vos documents et vos données.")
    st.divider()
    
    st.header("📂 1. Vos Documents")
    fichiers_upload = st.file_uploader("Glissez vos PDF ici", type="pdf", accept_multiple_files=True)

    if fichiers_upload and st.session_state.vectordb is None:
        if st.button("🚀 Lancer l'analyse", use_container_width=True, type="primary"):
            
            # Animation professionnelle détaillée
            with st.status("📥 Traitement de vos documents...", expanded=True) as status:
                st.write("Lecture des fichiers PDF...")
                toutes_les_pages = []
                for fichier in fichiers_upload:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as fichier_temp:
                        fichier_temp.write(fichier.getvalue())
                        chemin_temp = fichier_temp.name
                    loader = PyPDFLoader(chemin_temp)
                    toutes_les_pages.extend(loader.load()) 

                st.write("Découpage analytique du texte...")
                decoupeur = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                morceaux = decoupeur.split_documents(toutes_les_pages)

                st.write("Création de la mémoire vectorielle locale...")
                modele_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                st.session_state.vectordb = Chroma.from_documents(documents=morceaux, embedding=modele_embedding)

                status.update(label="✅ Documents mémorisés avec succès !", state="complete", expanded=False)

    st.divider()
    st.header("🧰 2. Options")
    
    # Bouton d'exportation
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

    # Bouton Reset
    if st.button("🗑️ Vider la mémoire et recommencer", use_container_width=True):
        st.session_state.messages = []
        st.session_state.vectordb = None
        st.rerun()

# ==========================================
# 4. ZONE PRINCIPALE (Le Chatbot)
# ==========================================
st.title("🛡️ ChatDoc Pro")
st.markdown("##### *Discutez avec vos documents professionnels en toute confidentialité.*")

# Écran d'accueil si la conversation est vide
if not st.session_state.messages:
    st.write("") # Espace
    st.info("👋 **Bienvenue !** Pour commencer, veuillez charger vos documents PDF dans le menu sur votre gauche.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.success("🔒 **100% Privé**\nAucune donnée n'est envoyée sur internet. Tout tourne sur votre machine.")
    with col2:
        st.warning("📚 **Multi-Documents**\nCroisez les informations de plusieurs PDF simultanément.")
    with col3:
        st.info("🧠 **IA Avancée**\nPropulsé par le modèle Gemma 3 (Google) optimisé pour la précision.")

# Affichage de l'historique
for message in st.session_state.messages:
    # On donne des avatars sympas
    avatar = "👤" if message["role"] == "user" else "🤖"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Boîte de dialogue
if question := st.chat_input("Posez votre question sur les documents..."):
    
    if st.session_state.vectordb is None:
        st.error("⚠️ Veuillez d'abord analyser un document via le menu de gauche.")
    else:
        with st.chat_message("user", avatar="👤"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Recherche dans les documents et rédaction..."):
                
                resultats = st.session_state.vectordb.similarity_search(question, k=3)
                contexte_trouve = "\n\n".join([doc.page_content for doc in resultats])

                # On utilise l'API ultra-rapide de Groq au lieu du PC local (OllamaLLM)
                # st.secrets permet de cacher la clé pour ne pas la mettre sur GitHub
                # NOUVEAUTÉ 1 : On passe sur LLaMA 3 (ultra-stable sur Groq)
                llm = ChatGroq(
                    groq_api_key=st.secrets["GROQ_API_KEY"], 
                    model_name="llama3-8b-8192",
                    temperature=0.3
                )
                
                historique_recent = ""
                if len(st.session_state.messages) > 1:
                    historique_recent = "Rappel de nos derniers échanges :\n"
                    for msg in st.session_state.messages[-4:]:
                        role = "Utilisateur" if msg["role"] == "user" else "IA"
                        historique_recent += f"- {role} a dit : {msg['content']}\n"

                prompt = f"""
                Tu es un assistant IA d'entreprise (ChatDoc Pro), amical, expert et très rigoureux.

                Instructions :
                1. EXACTITUDE : Basé UNIQUEMENT sur le contexte fourni. 
                2. CLARTÉ : Fais des phrases courtes, aère ton texte, utilise des listes à puces et mets les points clés en **gras**.
                3. HONNÊTETÉ : Si la réponse n'est pas dans le document, dis "Je suis désolé, cette information ne figure pas dans le document."

                {historique_recent}

                Contexte extrait des documents :
                {contexte_trouve}

                Nouvelle question : {question}

                Réponse :
                """
                
                # NOUVEAUTÉ 2 : On extrait le texte propre avec ".content"
                reponse_brute = llm.invoke(prompt)
                texte_final = reponse_brute.content
                
                st.markdown(texte_final)
                
                # Option bonus : Affichage des sources de manière discrète
                sources = set([doc.metadata.get('source', 'Inconnue').split('\\')[-1] for doc in resultats])
                st.caption(f"Sources utilisées : {', '.join(sources)}")
                
                st.session_state.messages.append({"role": "assistant", "content": texte_final})