from langchain_chroma import Chroma
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from pathlib import Path
import os
import dotenv
import subprocess

dotenv.load_dotenv()

command = 'cls' if os.name == 'nt' else 'clear'
subprocess.run(command, shell=True)

api_key_google = os.getenv('GOOGLE_API_KEY')
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=api_key_google)
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0, api_key=api_key_google)

persist_directory_BD = str(Path( Path.cwd(), 'Tema3', 'chroma_db' ))
vector_store = Chroma( embedding_function=embedding, persist_directory=persist_directory_BD)
 
# Crear compresor que extrae solo contenido relevante
compressor = LLMChainExtractor.from_llm(llm)
 
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vector_store.as_retriever()
)
 
# Solo obtener las partes relevantes
compressed_results = compression_retriever.invoke(
    "¿Dame 3 puntos más importantes y que deba de poner puntual atención de estos contratos?"
)

for i, doc in enumerate(compressed_results, 1):
    print(f"Contenido: {doc.page_content}\n\n")