from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader

import os
import dotenv
import subprocess
from pathlib import Path

dotenv.load_dotenv()

command = 'cls' if os.name == 'nt' else 'clear'
subprocess.run(command, shell=True)

pdfs_folder = Path( Path.cwd(), 'Tema3', 'contratos' )
loader = PyPDFDirectoryLoader(pdfs_folder)
documents = loader.load()
print(f"Se cargaron {len(documents)} documentos desde el directorio")
 
# Crear retriever BM25 (búsqueda por palabras clave)
bm25_retriever = BM25Retriever.from_documents(documents)
bm25_retriever.k = 5
 
# Crear retriever vectorial (búsqueda semántica)
api_key_google = os.getenv('GOOGLE_API_KEY')
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=api_key_google)
vectorstore = FAISS.from_documents(documents, embedding)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
 
# Combinar ambos con pesos
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]  # 30% BM25, 70% vectorial
)

# ✅ PRUEBA: hacer una consulta real
consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"
resultados = ensemble_retriever.invoke(consulta)

print(f"Top resultados ({len(resultados)}):\n")
for i, doc in enumerate(resultados, 1):
    print(f"--- Resultado {i} ---")
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}\n")