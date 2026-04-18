from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

import os
import subprocess
import dotenv

commando = 'cls' if os.name == 'nt' else 'clear'
subprocess.run(commando, shell=True)

dotenv.load_dotenv()
api_key_google = os.getenv('GOOGLE_API_KEY')

llm = ChatGoogleGenerativeAI(
    model='gemini-3.1-flash-lite-preview',
    temperature=0.7,
    api_key=api_key_google
)

# SE DEFINE LA PLANTILLA  con la clase PromptTemplate  en donde:
plantilla = PromptTemplate(
    input_variables=["nombre"],
    template="Saluda al usuario por su nombre.\nNombre del usuario: {nombre}\nAsistente:"
)

# IMPLEMENTACION DE CADENAS CON |
chain = plantilla | llm
resultado = chain.invoke({"nombre": "Iván Uribe"})
print(resultado.content)