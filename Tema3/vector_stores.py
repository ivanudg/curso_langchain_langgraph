from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pathlib import Path
import os
import dotenv
import subprocess

dotenv.load_dotenv()

command = 'cls' if os.name == 'nt' else 'clear'
subprocess.run(command, shell=True)

pdfs_folder = Path( Path.cwd(), 'Tema3', 'contratos' )
loader = PyPDFDirectoryLoader(pdfs_folder)
documentos = loader.load()
print(f"Se cargaron {len(documentos)} documentos desde el directorio")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs_split = text_splitter.split_documents(documentos)
print(f"Se crearon {len(docs_split)} chunks de texto a partir de los documentos.")

api_key_google = os.getenv('GOOGLE_API_KEY')
embedding = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=api_key_google
)

persist_directory_BD = str(Path( Path.cwd(), 'Tema3', 'chroma_db' ))
# Chroma.from_documents: Crear la colección y agregar los documentos en un solo paso
# Chroma: Conectarse a una colección existente (sin cargar docs)
vectorStore = Chroma.from_documents(
    documents=docs_split,
    embedding=embedding,
    persist_directory=persist_directory_BD
)

consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"
resultados = vectorStore.similarity_search(consulta, k=2) # (K) Son los chunks que queremos que nos devuelva
print(f"Top 3 documentos más similares a la consulta:\n")
for i, doc in enumerate(resultados, 1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")