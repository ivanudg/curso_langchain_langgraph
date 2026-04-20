from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

plantilla_sistema = SystemMessagePromptTemplate.from_template("" \
    "Eres un {rol} especializado en {especialidad}. Responde de manera {tono}"
)

plnatilla_humano = HumanMessagePromptTemplate.from_template(
    "Mi pregunta sobre {tema} es: {pregunta}"
)

chat_prompt = ChatPromptTemplate([
    plantilla_sistema,
    plnatilla_humano
])

mensajes = chat_prompt.format_messages(
    rol='nutricionista',
    especialidad='dietas veganas',
    tono='profesional pero accesible',
    tema='proteinas vegetales',
    pregunta='¿Cuáles son las mejores fuentes de proteina vegana para un atleta profesional?'
)

for m in mensajes:
    print(m.content)