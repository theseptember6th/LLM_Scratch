from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os

# from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

# this also works but its locally

# os.environ["HF_HOME"] = "/Users/kristalshrestha/Documents/Code/LLM_Scratch/models"
# llm = HuggingFacePipeline.from_model_id(
#     model_id="mistralai/Mistral-7B-Instruct-v0.2",
#     task="text-generation",
#     pipeline_kwargs={
#         "max_new_tokens": 100,
#         "temperature": 0.5,
#     },
# )

# model = ChatHuggingFace(llm=llm)


# but i want to use gemini
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY")
)
messages = [
    SystemMessage(content="You are a helpful asssistant."),
    HumanMessage(content="tell me about the langchain"),
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)
