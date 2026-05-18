from pathlib import Path

CHROMADB_PATH = str( Path( __file__ ).parent.parent / 'chroma_db' )
DOCS_PATH =  str( Path( __file__ ).parent.parent / "docs" ) 
EMBEDDINGS_MODEL = "gemini-embedding-001"