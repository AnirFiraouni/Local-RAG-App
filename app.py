from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from langchain_community.llms import Ollama

# 1. On indique le nom de notre fichier PDF
chemin_du_fichier = "Interpolation.pdf"

# 2. On initialise l'outil de LangChain dédié aux PDF
loader = PyPDFLoader(chemin_du_fichier)

# 3. On lui demande d'extraire toutes les pages
pages = loader.load()

# 4. On vérifie que ça a fonctionné en affichant un résumé
print(f"Succès ! Le document contient {len(pages)} page(s).")
print("\n--- Voici les 500 premiers caractères de la page 1 ---")
print(pages[0].page_content[0:500])


print("\n--- ÉTAPE 2 : Découpage du texte ---")

# 1. On configure notre "découpeur" de texte
decoupeur = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # Taille maximale d'un morceau (environ 150 à 200 mots)
    chunk_overlap=200, # Le chevauchement pour ne pas perdre le sens
    length_function=len
)

# 2. On applique le découpage sur nos pages extraites
morceaux = decoupeur.split_documents(pages)

# 3. On vérifie le résultat
print(f"Ton document original de {len(pages)} page(s) a été découpé en {len(morceaux)} petits morceaux.")
print("\n--- Voici à quoi ressemble le morceau n°2 ---")
# On affiche le deuxième morceau (index 1) pour voir le chevauchement
print(morceaux[1].page_content)


print("\n--- ÉTAPE 3 : Vectorisation et création de la base de données ---")

# 1. On charge un modèle d'embedding gratuit et performant (en français/multilingue)
# "all-MiniLM-L6-v2" est un grand classique, très léger et rapide.
modele_embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. On crée un dossier pour sauvegarder notre base de données sur l'ordinateur
dossier_chroma = "./chroma_db"

# 3. On crée la base de données vectorielle avec Chroma
print("Création des embeddings en cours... (ça peut prendre quelques secondes la première fois)")
base_de_donnees = Chroma.from_documents(
    documents=morceaux, # Nos morceaux de l'étape 2
    embedding=modele_embedding, # Notre traducteur mathématique
    persist_directory=dossier_chroma # L'endroit où sauvegarder
)

print(f"Super ! La base de données est créée et contient {base_de_donnees._collection.count()} éléments vectorisés.")

print("\n--- ÉTAPE 4 : Le test de recherche (Retrieval) ---")

# 1. Je teste le programme avec une question en relation avec le pdf !
question = "c'est quoi des polynomes orthogonaux?"

# 2. On interroge notre base de données Chroma
# k=2 signifie qu'on veut récupérer les 2 morceaux (chunks) les plus pertinents
resultats = base_de_donnees.similarity_search(question, k=2)

# 3. On affiche les résultats trouvés
print(f"\nQuestion posée : '{question}'\n")
print("Voici les passages les plus pertinents trouvés par le système :")
for i, resultat in enumerate(resultats):
    print(f"\n--- Extrait pertinent n°{i+1} ---")
    print(resultat.page_content)



print("\n--- ÉTAPE 5 : Génération de la réponse avec Ollama ---")

# 1. On charge le modèle que tu as déjà sur ton PC !
llm = Ollama(model="gemma3:4b")

# 2. On prépare les morceaux de texte trouvés à l'étape 4
contexte_trouve = "\n\n".join([doc.page_content for doc in resultats])

# 3. On rédige "l'Instruction" stricte (le Prompt) pour l'IA
prompt = f"""
Tu es un assistant francophone très utile et précis. 
Utilise UNIQUEMENT le contexte fourni ci-dessous pour répondre à la question de l'utilisateur.
Si la réponse ne se trouve pas dans le contexte, dis simplement que tu ne sais pas, n'invente rien.

Contexte extrait du document :
{contexte_trouve}

Question de l'utilisateur : {question}

Réponse :
"""

# 4. On envoie tout ça au cerveau !
print("🧠 L'IA lit les extraits et rédige sa réponse... (patiente un peu, ton PC réfléchit !)")
reponse_finale = llm.invoke(prompt)

# 5. On affiche le résultat magique
print("\n🤖 RÉPONSE DE L'IA :")
print(reponse_finale)