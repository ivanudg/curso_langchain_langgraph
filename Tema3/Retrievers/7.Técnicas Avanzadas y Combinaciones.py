from langchain_classic.retrievers.document_compressors import CohereRerank
 
# Primer pase: recuperar muchos documentos
base_retriever = vectorstore.as_retriever(search_kwargs={"k": 20})
 
# Segundo pase: reordenar con Cohere
reranker = CohereRerank(
    model="rerank-multilingual-v2.0",
    top_n=5
)
 
compression_retriever = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=base_retriever
)