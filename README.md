# 🛡️ ChatDoc Pro - Assistant RAG Intelligent

<a href="https://chatdoc-anirfiraouni.streamlit.app/" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge.svg" alt="Streamlit App">
</a>

## 📝 Description
ChatDoc Pro est une application web d'Intelligence Artificielle d'entreprise basée sur l'architecture **RAG (Retrieval-Augmented Generation)**. Elle permet d'interroger et de croiser les informations de multiples documents privés en langage naturel, avec une traçabilité complète des sources. Initialement conçue en local, l'application est désormais propulsée par le Cloud pour offrir des performances professionnelles.

## 🚀 Fonctionnalités Clés
* **Support Multi-Formats :** Ingestion simultanée de fichiers PDF, Word (.docx), Texte (.txt) et Excel/Données (.csv).
* **Citations Exactes :** L'IA source systématiquement ses réponses en indiquant le nom du document original et le **numéro de page exact**.
* **Recherche Sémantique Hybride :** Découpage intelligent (Chunking) et vectorisation en mémoire vive pour des requêtes instantanées.
* **Moteur LLM Haute Performance :** Génération de réponses synthétiques via le modèle **LLaMA 3.3 70b** (via l'API Groq), garantissant rapidité et précision.
* **Gestion de l'historique :** Exportation des comptes-rendus de chat et gestion indépendante de la mémoire (nettoyage de conversation sans perdre l'indexation des documents).

## 🛠️ Technologies Utilisées (Stack)
* **Langage :** Python
* **Interface Web :** Streamlit
* **Orchestration RAG :** LangChain
* **Modèle LLM :** API Groq (`llama-3.3-70b-versatile`)
* **Base de Données Vectorielle :** ChromaDB (In-Memory)
* **Modèle d'Embedding :** HuggingFace (`all-MiniLM-L6-v2`)

## ⚙️ Installation et Lancement Local

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/AnirFiraouni/Local-RAG-App](https://github.com/AnirFiraouni/Local-RAG-App)
   cd Local-RAG-App