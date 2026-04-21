from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from pathlib import Path

import os
os.environ["USER_AGENT"] = "MiApp/1.0"

file_pdf = Path( Path.cwd(), 'Tema3', 'insumos', 'MINSAIT26-JorgeIvanUribeMejia.pdf' )
#/Users/jorgeivanuribemejia/Documents/langChain_langGraph/Tema3/insumos/MINSAIT26-JorgeIvanUribeMejia.pdf
loader = PyPDFLoader(str(file_pdf))

pages = loader.load()

for i, page in enumerate(pages, 1):
    print(f"=== PÁGINA {i} ===")
    print(f"Contenido: {page.page_content}")
    print(f"Metadata: {page.metadata}")

print("*" * 250)

loader_web = WebBaseLoader("https://techmind.ac/")
docs = loader_web.load()
print(docs)