from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

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

# Chroma.from_documents: Crear la colección y agregar los documentos en un solo paso
# Chroma: Conectarse a una colección existente (sin cargar docs)
vectorStore = Chroma(embedding_function=embedding,persist_directory=persist_directory_BD)

base_retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
retriever = MultiQueryRetriever.from_llm(retriever=base_retriever, llm=llm)

consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"
resultados = retriever.invoke(consulta)

print(f"Top documentos más similares a la consulta:\n")
for i, doc in enumerate(resultados, 1):
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}")