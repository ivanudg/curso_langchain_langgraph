from langchain_classic.retrievers import TimeWeightedVectorStoreRetriever
from datetime import datetime, timedelta
 
tw_retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore,
    decay_rate=0.999,  # Qué tan rápido "olvida" documentos antiguos
    k=5
)
 
# Los documentos más antiguos tendrán menos peso
yesterday = datetime.now() - timedelta(days=1)
tw_retriever.add_documents([Document(page_content="Noticia vieja")], timestamps=[yesterday])
tw_retriever.add_documents([Document(page_content="Noticia nueva")])
