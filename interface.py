import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Mon IA Privée", page_icon="🤖")
st.title("📄 Discute avec ton PDF")
st.markdown("Cette IA tourne 100% localement sur ton PC, sans envoyer tes données sur internet !")

# --- GESTION DE LA MÉMOIRE (Session State) ---
# Streamlit recharge la page à chaque action. On utilise "session_state" pour garder 
# en mémoire la base de données et l'historique de la conversation.
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- BARRE LATÉRALE (Sidebar) : Upload du fichier ---
with st.sidebar:
    st.header("1. Charge ton document")
    fichier_upload = st.file_uploader("Glisse ton PDF ici", type="pdf")

    # Si un fichier est chargé et qu'il n'a pas encore été traité
    if fichier_upload is not None and st.session_state.vectordb is None:
        with st.spinner("🧠 Analyse et mémorisation du document en cours..."):
            
            # Astuce : On sauvegarde temporairement le fichier uploadé pour que PyPDFLoader puisse le lire
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as fichier_temp:
                fichier_temp.write(fichier_upload.getvalue())
                chemin_temp = fichier_temp.name

            # Étape 1 & 2 : Lecture et Découpage
            loader = PyPDFLoader(chemin_temp)
            pages = loader.load()
            decoupeur = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            morceaux = decoupeur.split_documents(pages)

            # Étape 3 : Vectorisation (On garde Chroma en mémoire vive, sans créer de dossier)
            modele_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.vectordb = Chroma.from_documents(documents=morceaux, embedding=modele_embedding)

            st.success("✅ Document analysé ! Tu peux poser tes questions.")

# --- ZONE DE CHAT PRINCIPALE ---

# On affiche l'historique des messages précédents
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Boîte de dialogue pour poser une nouvelle question
if question := st.chat_input("Pose ta question ici..."):
    
    # On vérifie qu'un document a bien été chargé
    if st.session_state.vectordb is None:
        st.error("⚠️ S'il te plaît, charge d'abord un fichier PDF sur le côté gauche !")
    else:
        # On affiche la question de l'utilisateur
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # On génère la réponse de l'IA
        with st.chat_message("assistant"):
            with st.spinner("L'IA réfléchit..."):
                
                # Étape 4 : Recherche (Retrieval)
                resultats = st.session_state.vectordb.similarity_search(question, k=2)
                contexte_trouve = "\n\n".join([doc.page_content for doc in resultats])

                # Étape 5 : Génération avec Gemma
                llm = OllamaLLM(model="gemma3:4b")
                prompt = f"""
                Tu es un assistant francophone. Utilise UNIQUEMENT le contexte ci-dessous pour répondre.
                Contexte: {contexte_trouve}
                Question: {question}
                """
                
                reponse = llm.invoke(prompt)
                st.markdown(reponse)
                
                # On sauvegarde la réponse dans l'historique
                st.session_state.messages.append({"role": "assistant", "content": reponse})