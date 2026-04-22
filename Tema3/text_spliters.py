import os
import dotenv
import subprocess
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 0. Limpirar consola
command = 'cls' if os.name == 'nt' else 'clear'
subprocess.run(command, shell=True)

# 1. Cargar documento PDF
pdf_path = Path( Path.cwd(), 'Tema3', 'insumos', 'quijote.pdf' )
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# 1.1 Dividir el texto en chunks más pequeños
text_spliter = RecursiveCharacterTextSplitter(
    chunk_size=10000,
    chunk_overlap=200 # Envía ciertos carácteres del chubk anteriór al llm para que tenga contexto
)

chunks = text_spliter.split_documents(pages)

# 3. Pasar el texto al LLM
dotenv.load_dotenv()
api_key_google = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature= 0.2, api_key=api_key_google)
summaries = []

for chunk in chunks:
    response = llm.invoke(f"Haz un resumen de los puntos más importantes del siguiente texto: {chunk}")
    summaries.append(response.content)

final_summary = llm.invoke(f"Combina y sintetiza estos resumenes en un resumen coherente y comlelto: {" ".join(summaries)}")
print(final_summary.content)