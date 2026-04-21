from langchain_google_genai import ChatGoogleGenerativeAI
from models.cv_model import AnalisisCV
from prompts.cv_prompts import crear_sistema_prompts
import os
import dotenv

dotenv.load_dotenv()
API_KEY_GOOGLE = os.getenv('GOOGLE_API_KEY')


def crear_evaluador_cv():
    modelo_base = ChatGoogleGenerativeAI(
        model='gemini-3.1-flash-lite-preview',
        temperature=0.2,
        api_key=API_KEY_GOOGLE
    )

    modelo_estructurado = modelo_base.with_structured_output(AnalisisCV)
    chat_prompt = crear_sistema_prompts()
    cadena_evaluacion = chat_prompt | modelo_estructurado

    return cadena_evaluacion


def evaluar_candidato(texto_cv: str, descripcion_puesto: str) -> AnalisisCV:
    try:
        cadena_evaluacion = crear_evaluador_cv()
        resultado = cadena_evaluacion.invoke({
            "texto_cv": texto_cv,
            "descripcion_puesto": descripcion_puesto
        })

        return resultado
    
    except Exception as e:
        return AnalisisCV(
            nombre_candidato="Error en procesamiento.",
            experiencia_anios=0,
            habilidades_clave=["Error al procesar CV"],
            educacion="No se puede determinar",
            experiencia_relevante="Error durante el análisis.",
            fortalezas=["Requiere revisión manual del CV"],
            areas_mejora=["Verificar formato y legibilidad del PDF"],
            porcentaje_ajuste=0
        )