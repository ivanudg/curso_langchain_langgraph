from langchain_core.prompts import ChatPromptTemplate

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "Eres un traductor del españo al ingles muy preciso"),
    ("human", "{texto}")
])

mensajes = chat_prompt.format_messages(texto="Hola Mundo, ¿cómo estas?")
print(mensajes)
print("*" * 200)
for m in mensajes:
    print(f'{type(m)}: {m.content}')