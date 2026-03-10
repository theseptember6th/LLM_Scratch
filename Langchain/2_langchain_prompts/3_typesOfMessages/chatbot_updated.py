import os
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

os.environ["HF_HOME"] = "/Users/kristalshrestha/Documents/Code/LLM_Scratch/models"
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.5,
    },
)
model = ChatHuggingFace(llm=llm)

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
