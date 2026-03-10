# 🛡️ ChatDoc Pro - Assistant RAG Intelligent

<a href="https://chatdoc-anirfiraouni.streamlit.app/" target="_blank">
    <img src="https://static.streamlit.io/badges/streamlit_badge.svg" alt="Streamlit App">
</a>
## 📝 Description
Cette application web permet d'interroger un document PDF en langage naturel. Basée sur l'architecture **RAG (Retrieval-Augmented Generation)**, elle fonctionne entièrement en local sur la machine. Cela garantit une confidentialité absolue des données, car aucun document n'est envoyé sur des serveurs externes (Cloud ou API tierces).

## 🚀 Fonctionnalités
* **Interface Web interactive** développée avec Streamlit (façon chatbot).
* **Ingestion de documents PDF** avec découpage intelligent (Chunking) pour préserver le contexte.
* **Recherche sémantique** ultra-rapide via des embeddings locaux et une base vectorielle en mémoire.
* **Génération de réponses** propulsée par le Grand Modèle de Langage (LLM) local **Gemma 3** de Google.

## 🛠️ Technologies Utilisées (Stack)
* **Langage :** Python
* **Orchestration IA :** LangChain
* **Interface Utilisateur :** Streamlit
* **Modèle LLM :** Ollama (Modèle `gemma3:4b`)
* **Base de Données Vectorielle :** ChromaDB
* **Modèle d'Embedding :** HuggingFace (`all-MiniLM-L6-v2`)

## 📸 Démonstration
*Insérer une capture d'écran de l'interface Streamlit ici.*

## ⚙️ Installation et Lancement

1. **Cloner le dépôt :**
   ```bash
   git clone [https://github.com/VOTRE_PSEUDO/Local-RAG-App.git](https://github.com/VOTRE_PSEUDO/Local-RAG-App.git)
   cd Local-RAG-App