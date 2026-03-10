from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
model = ChatGoogleGenerativeAI(model="gemini-3.1-pro-preview", api_key=api_key)

result = model.invoke("What is the capital of India?")
print(result.content)
