from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader

from pathlib import Path
import os
import dotenv
import subprocess

dotenv.load_dotenv()

command = 'cls' if os.name == 'nt' else 'clear'
subprocess.run(command, shell=True)

api_key_google = os.getenv('GOOGLE_API_KEY')
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=api_key_google)

pdfs_folder = Path( Path.cwd(), 'Tema3', 'contratos' )
loader = PyPDFDirectoryLoader(pdfs_folder)
documents = loader.load()
print(f"Se cargaron {len(documents)} documentos desde el directorio")
 
# Splitter para documentos padre (chunks grandes)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
 
# Splitter para documentos hijo (chunks pequeños para embedding)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
 
# Vector store para los chunks pequeños
vectorstore = Chroma(
    collection_name="parent_docs",
    embedding_function=embedding
)
 
# Almacenamiento para documentos padre
store = InMemoryStore()
 
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)
 
# Agregar documentos
retriever.add_documents(documents)

print(f"retriever: {retriever}")