from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_google_genai import ChatGoogleGenerativeAI
import json
import os
import dotenv
import subprocess

dotenv.load_dotenv()
api_key_google = os.getenv('GOOGLE_API_KEY')

# Configuración del modelo
llm = ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite-preview',
    temperature=0,
    api_key=api_key_google
)

# Limpia la consola
comando = 'cls' if os.name == 'nt' else 'clear'
subprocess.run(comando, shell=True)

# Extrae el texto de la respuesta del modelo
def extract_text(content):
    if isinstance(content, list):
        return " ".join([b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"])
    return content

# Limpia el texto eliminando espacios extras y limitando longitud
def preprocess_text(text):
    texto = text.strip()
    texto = " ".join(texto.split())
    return texto[:500]

# Convertir la función en un Runnable
preprocessor = RunnableLambda(preprocess_text)

# Genera un resumen conciso del texto
def generate_summary(text):
    #prompt = f"Resume en una sola oración: {text}"
    prompt = f"""
Resume el siguiente texto en un solo párrafo conciso, sin perder las ideas principales. 
El resumen debe ser claro, fluido y no exceder las 5 oraciones. 
No agregues opiniones ni información externa, solo lo que está en el texto.

Texto a resumir:
{text}
"""
    response = llm.invoke(prompt)
    return response.content

# Convertir en Runnable
summary_branch = RunnableLambda(generate_summary)

# Analiza el sentimiento y devuelve resultado estructurado
def analyze_sentiment(text):
    prompt = f"""Analiza el sentimiento del siguiente texto.
    Responde ÚNICAMENTE en formato JSON válido:
    {{"sentimiento": "positivo|negativo|neutro", "razon": "justificación breve"}}
    
    Texto: {text}
"""
    
    response = llm.invoke(prompt)
    try:
        print(f"JSON: {extract_text(response.content)}")
        return json.loads(extract_text(response.content))
    except json.JSONDecodeError:
        return {"sentimiento": "neutro", "razon": "Error en análisis"}

# Convertir en Runnable
sentiment_branch = RunnableLambda(analyze_sentiment)

 # Combina los resultados de ambas ramas en un formato unificado
def merge_results(data):
    return {
        "resumen": data["resumen"],
        "sentimiento": data["sentimiento_data"]["sentimiento"],
        "razon": data["sentimiento_data"]["razon"]
    }

# Convertir en Runnable
merge_branch = RunnableLambda(merge_results)

# Coodinar el análisis completo (resumen + sentimientos).
parallel_analizys = RunnableParallel({
    "resumen": summary_branch,
    "sentimiento_data": sentiment_branch
})

# Conectar todos los componentes usando LCEL.
# La cadena completa
chain = preprocess_text | parallel_analizys | merge_branch

# review = "Este producto es muy malo. No me ha gustado nada."
# resultado = chain.invoke(review)
# print(resultado)

# Prueba con diferentes textos
reviews_batch = [
    "Excelente producto, muy satisfecho con la compra",
    "Terrible calidad, no lo recomiendo para nada",
    "Está bien, cumple su función básica pero nada especial"
]

# Ejecutará múltiples elementos simultaneamente
resultado_branch = chain.batch(reviews_batch)

print(resultado_branch)