import os

# from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

# this works as i am using locally but i am using gemini api
# os.environ["HF_HOME"] = "/Users/kristalshrestha/Documents/Code/LLM_Scratch/models"
# llm = HuggingFacePipeline.from_model_id(
#     model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation",
#     pipeline_kwargs={
#         "max_new_tokens": 100,
#         "temperature": 0.5,
#     },
# )
# model = ChatHuggingFace(llm=llm)

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY")
)
chat_history = [SystemMessage(content="You are helpful AI assistant")]
while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content=user_input))
    if user_input.strip().lower() == "exit":
        break
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print(f"AI: {result.content}")


print(chat_history)
