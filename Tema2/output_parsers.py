from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import dotenv

# Inicializar modelo (LLM)
dotenv.load_dotenv()
api_key_google = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(
    model='gemini-3-flash-preview',
    temperature=0.3,
    api_key=api_key_google
)

class AnalisisTexto(BaseModel):
    resumen: str = Field(description="Resumen breve del texto.")
    sentimiento: str = Field(description="Sentimiento del texto (Positivo, negativo o neutro)")
    palabras_clave: list[str] = Field(description="Lista de palabras clave del texto")

# Crear el parser con nuestro modelo
parser = PydanticOutputParser(pydantic_object=AnalisisTexto)

# Ver las instrucciones que genera automáticamente
# print("Instrucciones generadas:")
# print(parser.get_format_instructions())

prompt = PromptTemplate(
    template="""Eres un experto analista de texto. Analiza el siguiente texto con mucho cuidado y proporciona un análisis detallado.
 
{format_instructions}
 
Texto a analizar:
{texto}
 
Análisis:""",
    input_variables=["texto"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

# Crear la cadena: prompt → LLM → parser
chain = prompt | llm | parser

# Texto de ejemplo
texto_prueba = """
La nueva película de ciencia ficción 'Estrella Galáctica' es absolutamente 
espectacular. Los efectos visuales son impresionantes y la trama mantiene 
la tensión durante toda la película. Los actores principales entregan 
actuaciones convincentes que realmente te hacen creer en este mundo futurista.
Sin duda una de las mejores películas del año.
"""

try:
    # Invocar la cadena
    resultado = chain.invoke({"texto": texto_prueba})

     # Acceder a los datos
    print("=== RESULTADO DEL ANÁLISIS ===")
    print(f"Resumen: {resultado.resumen}")
    print(f"Sentimiento: {resultado.sentimiento}")
    print(f"Palabras clave: {', '.join(resultado.palabras_clave)}")

    # Exportar como JSON
    print("\n=== JSON RESULTANTE ===")
    print(resultado.model_dump_json(indent=2))

    # Exportar como diccionario
    dict_resultado = resultado.model_dump()
    print(f"\nDic Resultado: {dict_resultado}")
    print(f"\nTipo de objeto: {type(resultado)}")
    print(f"Tipo de diccionario: {type(dict_resultado)}")

except Exception as e:
    print(f"❌ Error durante el procesamiento: {e}")

