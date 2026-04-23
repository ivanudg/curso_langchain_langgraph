from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
 
# Definir metadatos de los documentos
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="El género de la película",
        type="string"
    ),
    AttributeInfo(
        name="year",
        description="El año de lanzamiento de la película",
        type="integer"
    ),
    AttributeInfo(
        name="rating",
        description="La calificación de la película (1-10)",
        type="float"
    ),
]
 
document_content_description = "Breve resumen de una película"
 
retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(temperature=0),
    vectorstore=vectorstore,
    document_content_description=document_content_description,
    metadata_field_info=metadata_field_info,
)
 
# Consulta que se convertirá en filtros estructurados
results = retriever.invoke("películas de ciencia ficción de después de 2010 con calificación alta")