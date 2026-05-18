from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from pathlib import Path
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from typing import List, Dict, Any

from config.config import *
import dotenv
import subprocess
import os

dotenv.load_dotenv()
api_key_google = os.getenv('GOOGLE_API_KEY')
command = 'cls' if os.name == 'nt' else 'clear'
subprocess.run(command, shell=True)

class VectorRAGSystem:
    # Sistema RAG avanzado con ChromaDB y MultiQueryRetriever.

    def __init__(self, chroma_path: str = "chroma_db"):
        self.chroma_path = Path(chroma_path)
        self.embeddings = GoogleGenerativeAIEmbeddings( model= EMBEDDINGS_MODEL, api_key=api_key_google )
        self.llm = GoogleGenerativeAIEmbeddings( model='gemini-3.1-flash-lite', api_key=api_key_google, temperature=0 )
        self.vectorstore = None
        self.retriever = None

        # Configurar logging para MultiQueryRetriever
        logging.basicConfig()
        logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

        self._load_vectorstore()

    def _load_vectorstore(self):
        # Carga el vectorstore de ChromaDB.
        try:
            if not self.chroma_path.exists():
                print(f"⚠️ Vectorstore no encontrado en {self.chroma_path}")
                return
            
            self.vectorstore = Chroma(
                persist_directory=str(self.chroma_path),
                embbeeding_function=self.embeddings,
                collection_name="helpdesk_knowledge"
            )

            # Crear MultiQueryRetriever
            self.retriever = MultiQueryRetriever.from_llm(
                retriever=self.vectorstore.as_retriever(
                    search_type="similarity", 
                    search_kwargs={"k": 4}
                ),
                llm = self.llm,
                prompt=self._get_multi_query_prompt()
            )

            print("✅ VectorRAGSystem inicializado correctamente")
        
        except Exception as e:
            print(f"❌ Error cargando vectorstore: {str(e)}")
            self.vectorstore = None
            self.retriever = None

    def _get_multi_query_prompt(self):
        # Prompt personalizado para MultiQueryRetriever.
        return ChatPromptTemplate.from_template(
            """Eres un asistente de helpdesk experto. Tu tarea es generar múltiples 
versiones de la consulta del usuario para recuperar documentos relevantes de una 
base de conocimiento de soporte técnico.

Genera 3 versiones diferentes de la consulta original, considerando:
- Sinónimos técnicos
- Diferentes formas de expresar el mismo problema
- Variaciones en terminología de helpdesk

Consulta original: {question}

Versiones alternativas:"""
        )
    
    def buscar(self, consulta: str ) -> Dict[str, Any]:
        # Busca respuestas usando MultiQueryRetriever.
        if not self.retriever:
            return {
                "respuesta": "Sistema RAG no disponible. Verifique la configuración.",
                "confianza": 0.0,
                "fuentes": []
            }
        
        try:
            # Buscar documentos relevantes con MultiQueryRetriever
            documentos = self.retriever.invoke(consulta)

            if not documentos:
                return {
                    "respuesta": "No encontré información relevante en la base de conocimiento.",
                    "confianza": 0.1,
                    "fuentes": []
                }
            
            # Extraer información de los documentos
            contexto_partes = []
            fuentes = []

            for i, doc in enumerate(documentos[:3]): # Usar top 3 documentos
                contenido = doc.page_content.strip()
                if contenido:
                    contexto_partes.append(f"Documento {i+1}: {contenido}")