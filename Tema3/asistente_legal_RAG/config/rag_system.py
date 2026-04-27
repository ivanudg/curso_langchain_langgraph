from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever
import streamlit as st

from config.config  import *
from prompts.prompts import *

import os
import subprocess
from pathlib import Path

# @st.cache_resource -> Almacena en cache
@st.cache_resource
def initialize_rag_system():
    # Vectore Store
    vectorstore = Chroma(
        embedding_function=GoogleGenerativeAIEmbeddings(model = EMBEDDING_MODEL, api_key = GOOGLE_API_KEY),
        persist_directory = CHROMA_DB_PATH
    )

    # Modelos
    llm_queries = ChatGoogleGenerativeAI(model = QUERY_MODEL, temperature = 0, api_key = GOOGLE_API_KEY)
    llm_generation = ChatGoogleGenerativeAI(model = GENERATION_MODEL, temperature = 0, api_key = GOOGLE_API_KEY)

    # Retriever MMR (Maximal Margin Relevacne)
    base_retriever = vectorstore.as_retriever(
        search_type = SEARCH_TYPE,
        search_kwargs = {
            "k": SEARCH_K,
            "lambda_mul": MMR_DIVERSITY_LAMBDA,
            "fetch_k": MMR_FETCH_K
        }
    )

    # Retriever adicional con similarity para comparar
    similatiry_retriever = vectorstore.as_retriever(
        search_type = "similarity",
        search_kwargs = { "k": SEARCH_K }
    )

    # Prompt personalizado para MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # MultiQueryRetriever con prompt personalizado
    mmr_multy_retiever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt
    )

    # Ensembled Retiever que combina MMR y Similarity
    if ENABLE_HYBRID_SEARCH:
        ensembled_retriever = EnsembleRetriever(
            retrievers= [ mmr_multy_retiever, similatiry_retriever ],
            weights= [ 0.7, 0.3 ], # Mayor peso al MMR
            similarity_threshold = SIMILARITY_THRESHOLD
        )

        final_retriever = ensembled_retriever
    else:
        final_retriever = mmr_multy_retiever

    # Prompt
    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    # Preprocesar los documentos relevantes obtenidos con el retiever
    # Función para formatear y preprocesar los documentos formateados
    def format_docs(docs):
        formatted = []

        for i, doc in enumerate(docs, 1):
            header = f"[Fragmento {i}]"
            
            if doc.metadata:
                if 'source' in doc.metadata:
                    source = doc.metadata['source'].split('\\')[-1] if '\\' in doc.metadata['source'] else doc.metadata['source']
                    header += f" - Fuente: {source}"
                if 'page' in doc.metadata:
                    header += f" - Página: {doc.metadata['page']}"
        
            content = doc.page_content.strip()
            formatted.append(f"{header}\n{content}")

        return "\n\n".join(formatted)

    # Chain:
    #   {} => Vairables que se le pasarán al siguiente paso
    rag_chain = (
        {
            "context": final_retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm_generation
        | StrOutputParser()
    )

    return rag_chain, mmr_multy_retiever

def query_rag(question):
    try:
        rag_chain, retriever = initialize_rag_system()

        # Obtener la respuesta
        response = rag_chain.invoke(question)

        # Obtener los documentos para mostrarlos 
        docs = retriever.invoke(question)

        # Formatear los documentos para mostrar
        docs_info = []
        for i, doc in enumerate(docs[:SEARCH_K], 1):
            doc_info = {
                "fragmento": i,
                "contenido": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": doc.metadata.get('source', 'No especificada').split('\\')[-1],
                "pagina": doc.metadata.get('page', 'No especificada')
            }
            docs_info.append(doc_info)
        
        return  response, docs_info
    
    except Exception as e:
        error_msg = f"Error al procesar la consulta {str(e)}"
        return error_msg, []
    
# Obtiene información sobre la configuración del retriever
def get_retriever_info():
    return {
        "tipo": f"{SEARCH_TYPE.upper()} + MultiQuery" + (" + Hybrid" if ENABLE_HYBRID_SEARCH else ""),
        "documentos": SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMBDA,
        "candidatos": MMR_FETCH_K,
        "umbral": SIMILARITY_THRESHOLD if ENABLE_HYBRID_SEARCH else "N/A"
    }