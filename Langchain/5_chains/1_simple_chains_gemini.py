from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os


load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key)


from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}", input_variables=["topic"]
)

from langchain_core.output_parsers import StrOutputParser


parser = StrOutputParser()

# chain
# | => pipe operator
chain = prompt | model | parser
result = chain.invoke({"topic": "cricket"})
print(result)
