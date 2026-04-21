import os
import dotenv
from pathlib import Path
from langchain_google_community import GoogleDriveLoader

dotenv.load_dotenv()

credentials_path = Path( Path.cwd(), "Tema3", "credentials.json" )
token_path = Path( Path.cwd(), "Tema3", "token.json" )
folder_id = os.getenv("GOOGLE_FOLDER_ID")

loader = GoogleDriveLoader(
    folder_id=folder_id,
    credentials_path=credentials_path,
    token_path=token_path,
    recursive=True
)

documents = loader.load()

print(f"Metadatos: {documents[0].metadata}")
print(f"Contenido: {documents[0].page_content}")