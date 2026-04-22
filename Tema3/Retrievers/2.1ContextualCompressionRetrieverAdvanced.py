from langchain_chroma import Chroma
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline, LLMChainExtractor
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_transformers import EmbeddingsRedundantFilter
from langchain_text_splitters import CharacterTextSplitter
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

# 1️⃣ Reconectar a la BD Chroma ya persistida (del ejercicio anterior)
persist_directory_BD = str(Path( Path.cwd(), 'Tema3', 'chroma_db' ))
vectorStore = Chroma( embedding_function=embedding, persist_directory=persist_directory_BD)

# 2️⃣ Retriever base
base_retriever = vectorStore.as_retriever(search_kwargs={"k": 5})

# 3️⃣ Pipeline: dividir → filtrar redundantes → comprimir por relevancia
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0, separator=".")
redundant_filter = EmbeddingsRedundantFilter(embeddings=embedding)
relevant_filter = LLMChainExtractor.from_llm(llm)
 
pipeline_compressor = DocumentCompressorPipeline(
    transformers=[splitter, redundant_filter, relevant_filter]
)

# 4️⃣ Retriever con compresión contextual
compression_retriever = ContextualCompressionRetriever(
    base_compressor=pipeline_compressor,
    base_retriever=base_retriever
)

# 5️⃣ Consulta de prueba
#consulta = "¿Cuál es el inmueble que forma parte del contrato en el que participa María Jiménez Campos?"
consulta = "¿Dónde se encuentra el local del contrato en el que participa María Jiménez Campos?"
resultados = compression_retriever.invoke(consulta)

print(f"Documentos comprimidos y relevantes ({len(resultados)}):\n")
for i, doc in enumerate(resultados, 1):
    print(f"--- Resultado {i} ---")
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}\n")