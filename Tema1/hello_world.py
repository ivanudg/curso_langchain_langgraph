from langchain_google_genai import ChatGoogleGenerativeAI
import os
import subprocess
import dotenv

comando = 'cls' if os.name == 'nt' else 'clear'
subprocess.run(comando, shell=True)

dotenv.load_dotenv()
api_key_google = os.getenv('GOOGLE_API_KEY')
#os.environ["SERPAPI_API_KEY"] = get_serpapi_key

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    temperature=0.7,
    api_key=api_key_google
)

pregunta = "¿En que año llegó el ser humano a la luna por primera vez?"
print(f'Pregunta: {pregunta}')

respuesta = llm.invoke(pregunta)
print(f'Respuesta del modelo: {respuesta}')
