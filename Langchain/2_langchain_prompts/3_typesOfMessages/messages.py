from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import os
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace

os.environ["HF_HOME"] = "/Users/kristalshrestha/Documents/Code/LLM_Scratch/models"
llm = HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.5,
    },
)

model = ChatHuggingFace(llm=llm)

messages = [
    SystemMessage(content="You are a helpful asssistant."),
    HumanMessage(content="tell me about the langchain"),
]

result = model.invoke(messages)
messages.append(AIMessage(content=result.content))

print(messages)
