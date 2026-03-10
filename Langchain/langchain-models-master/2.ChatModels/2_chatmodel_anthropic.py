from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("CLAUDE_API_KEY")

model = ChatAnthropic(model="claude-3-5-sonnet-20241022", api_key=api_key)

result = model.invoke("What is the capital of India")

print(result.content)
