from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader

import os
os.environ["USER_AGENT"] = "MiApp/1.0"

# loader = PyPDFLoader("C:\\Users\\jiuribe\\Documents\\Cursos\\curso_langchain_langgraph\\Tema3\\insumos\\MINSAIT26-JorgeIvanUribeMejia.pdf")

# pages = loader.load()

# for i, page in enumerate(pages, 1):
#     print(f"=== PÁGINA {i} ===")
#     print(f"Contenido: {page.page_content}")
#     print(f"Metadata: {page.metadata}")

# print("*" * 250)

loader_web = WebBaseLoader("https://techmind.ac/")
docs = loader_web.load()
print(docs)