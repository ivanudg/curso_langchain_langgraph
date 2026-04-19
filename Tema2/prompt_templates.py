from langchain_core.prompts import PromptTemplate

template = "Eres un experto en marketing. Sugiere un eslogan creativo para un producto {producto}"
prompt = PromptTemplate(
    template=template,
    input_variables=["producto"]
)

# Esto sirve para validar como se estará formando el promt con la(s) variables
prompt_completo = prompt.format(producto="Café Orgánico")
print(prompt_completo)

