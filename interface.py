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
                # On met à jour avec le modèle le plus récent et stable
                # --- ÉTAPE 5 : GÉNÉRATION AVEC PROMPT AMÉLIORÉ ---
                # --- ÉTAPE 5 : GÉNÉRATION CONCISE ET SOURCES PROPRES ---
                llm = ChatGroq(
                    groq_api_key=st.secrets["GROQ_API_KEY"], 
                    model_name="llama-3.3-70b-versatile",
                    temperature=0.3
                )
                
                # Récupération des vrais noms de fichiers chargés par l'utilisateur
                noms_originaux = [f.name for f in fichiers_upload]
                
                prompt = f"""
                Tu es ChatDoc Pro. Ton rôle est de répondre de manière SYNTHÉTIQUE et PRÉCISE.
                
                RÈGLES :
                1. RÉSUMÉ : Ne fais pas de longs paragraphes. Utilise 3 à 4 puces maximum pour aller à l'essentiel.
                2. STYLE : Sois direct, professionnel et humain.
                3. SOURCE : Utilise uniquement le contexte fourni.

                CONTEXTE :
                {contexte_trouve}

                QUESTION : {question}

                RÉPONSE RÉSUMÉE :
                """
                
                reponse_brute = llm.invoke(prompt)
                texte_final = reponse_brute.content
                
                st.markdown(texte_final)
                
                # Affichage des vrais noms de fichiers sans les "/tmp/"
                if noms_originaux:
                    st.caption(f"📚 Documents consultés : {', '.join(noms_originaux)}")
                
                st.session_state.messages.append({"role": "assistant", "content": texte_final})