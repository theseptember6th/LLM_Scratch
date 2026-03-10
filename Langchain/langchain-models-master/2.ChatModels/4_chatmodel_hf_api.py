from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

api_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")

# Create LLM endpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceTB/SmolLM3-3B",
    task="text-generation",
    huggingfacehub_api_token=api_key,
)

# Wrap with chat interface
model = ChatHuggingFace(llm=llm)

# Ask question
result = model.invoke("What is the capital of India?")

print(result.content)
