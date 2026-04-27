import os
import dotenv
from pathlib import Path

dotenv.load_dotenv()

# Ruta base del proyecto
BASE_DIR  = Path(__file__).parent.parent

# Configuración de modelos
EMBEDDING_MODEL = 'gemini-embedding-001'
QUERY_MODEL = 'gemini-3.1-flash-lite-preview'
GENERATION_MODEL = 'gemini-3.1-flash-lite-preview'
#GENERATION_MODEL = 'gemini-2.5-pro'

# Configuración del Vector Store
#CHROMA_DB_PATH = str( Path( PATH_BASE, 'Tema3', 'asistente_legal_RAG', 'BD' ) )
CHROMA_DB_PATH = str( Path(__file__).parent.parent / 'DB' )

# ApiKey Google
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Configuración del Retiever
SEARCH_TYPE = 'mmr'
MMR_DIVERSITY_LAMBDA =  0.7 #Balance entre relevancia (mas pegado a 1) y diversidad (mas pegado a 0)
MMR_FETCH_K = 20 # Documentos iniciales a evaluar antes de aplicar MMR
SEARCH_K = 2 # Documentos finales, que recibiremos en la respuesta

# Configuracion alternativa para retriever hibrido
ENABLE_HYBRID_SEARCH = True
SIMILARITY_THRESHOLD = 0.70