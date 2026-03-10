from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(
    model="gpt-4o-mini", temperature=1.5, max_completion_tokens=10, api_key=api_key
)

result = model.invoke("Write a 5 line poem on cricket")

print(result.content)
