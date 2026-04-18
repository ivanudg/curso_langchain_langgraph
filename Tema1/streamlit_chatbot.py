from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
import streamlit as st

import os
import dotenv

dotenv.load_dotenv()
api_key_google = os.getenv('GOOGLE_API_KEY')

def extract_text(content):
    if isinstance(content, list):
        return " ".join([b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"])
    return content

# Configurar la página de la aplicación
st.set_page_config(page_title="Chatbot Básico", page_icon="🤖")
st.title("🤖 Chatbot Básico con LangChain")
st.markdown("Este es un *chatbod de ejemplo* construido con LangChain + Streamlit. ¡Escribe tu mensaje abajo para comenzar!")

with st.sidebar:
    st.header("Configuración")
    temperature = st.slider("Temperatura", 0.0, 1.0, 0.5, 0.1)
    model_name = st.selectbox("Modelo", ['gemini-3-flash-preview', 'gemini-3.1-flash-lite-preview'])

    #chat_model = ChatGoogleGenerativeAI(model='gemini-3.1-flash-lite-preview', temperature=0.5, api_key=api_key_google)
    #chat_model = ChatGoogleGenerativeAI(model='gemini-3-flash-preview', temperature=0.5, api_key=api_key_google)
    chat_model = ChatGoogleGenerativeAI(
        model=model_name, 
        temperature=temperature, 
        api_key=api_key_google
    )

# Inicializar el historial de mensajes
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

# Crear el template de prompt con comportamiento especifico
prompt_template = PromptTemplate(
    input_variables=["mensaje", "historial"],
    template="""Eres un asistente útil y amigable llamado ChatBot Pro. 
 
Historial de conversación:
{historial}
 
Responde de manera clara y concisa a la siguiente pregunta: {mensaje}"""
)

# Crear cadena usando LCEL (LangChain Expression Language)
cadena = prompt_template | chat_model

# Mostrar mensajes previos en la interfaz
for msg in st.session_state.mensajes:
    if isinstance(msg, SystemMessage):
        # No muestro el mensjae en pantalla
        continue

    role = "assistant" if isinstance(msg, AIMessage) else "user"
    
    with st.chat_message(role):
        #st.markdown(msg.content)
        st.markdown(extract_text(msg.content))

# Botón para generar nueva conversación
if st.button("🗑️ Nueva conversación:"):
    st.session_state.mensajes = []
    st.rerun()

# Cuadro de entrada de texto de usuario
pregunta = st.chat_input("Escribe tu mensaje: ")

if pregunta:
    #Mostrar inmediatamente el mensaje del usuario en la interfaz
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Generar y mostrar respuesta del asistente
    try:
        #Mostrar la respuesta en la interface
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""

        # Streaming de la respuesta
        for chunk in cadena.stream({"mensaje": pregunta, "historial": st.session_state.mensajes}):
            full_response += extract_text(chunk.content)
            response_placeholder.markdown(full_response + "▌ ")

        response_placeholder.markdown(full_response)

        # Almacenamos el mensaje en la memoria de streamlit
        st.session_state.mensajes.append(HumanMessage(content=pregunta))
        st.session_state.mensajes.append(AIMessage(content=full_response))

    except Exception as e:
        st.error(f"Error al generar respuesta {str(e)}")
        st.info("Verifica que tu API de Google esté configurada correctamente")

    #st.session_state.mensajes.append(respuesta)