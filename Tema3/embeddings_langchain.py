from langchain_google_genai import GoogleGenerativeAIEmbeddings
import  numpy as np
import dotenv
import os

dotenv.load_dotenv()
api_key_google = os.getenv('GOOGLE_API_KEY')

embeddings = GoogleGenerativeAIEmbeddings(
    #model="models/gemini-embedding-001"
    model='gemini-embedding-001',
    api_key=api_key_google
)

texto1 = "La capital de Francia es Paris."
#texto2 = "Paris es la ciudad capital de Francia."
texto2 = "Paris es un nombre común para mascotas"

vec1 = embeddings.embed_query(texto1)
vec2 = embeddings.embed_query(texto2)

print(f"Dimensión de los vectores: {len(vec1)}")

cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

print(f"Similitud coseno entre vec1 y vec2: {cos_sim:.3f}")