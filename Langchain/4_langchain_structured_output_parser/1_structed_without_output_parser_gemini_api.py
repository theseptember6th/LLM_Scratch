from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=api_key,
)

# 1st prompt ->Detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)
prompt1 = template1.invoke({"topic": "blackhole"})
result = model.invoke(prompt1)

# 2nd prompt
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text./n {text}",
    input_variables=["text"],
)
prompt2 = template2.invoke({"text": result.content})
result1 = model.invoke(prompt2)

print(result1.content)
