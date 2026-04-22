from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
import dotenv
import subprocess
from pathlib import Path

dotenv.load_dotenv()

command = 'cls' if os.name == 'nt' else 'clear'
subprocess.run(command, shell=True)

api_key_google = os.getenv('GOOGLE_API_KEY')
persist_directory_BD = str(Path( Path.cwd(), 'Tema3', 'chroma_db' ))

# Configurar el retriever
embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", api_key=api_key_google)
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)
vector_store = Chroma( embedding_function=embedding, persist_directory=persist_directory_BD )


multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=vector_store.as_retriever(),
    llm=llm
    #verbose=True  # Ver las consultas generadas
)
 
# Una pregunta se convierte en múltiples perspectivas
results = multi_query_retriever.invoke("¿Dame los 5 puntos más importantes y que deba de poner puntual atención de estos contratos?")

for i, doc in enumerate(results, 1):
    print(f"Contenido: {doc.page_content}")