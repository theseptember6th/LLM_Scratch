from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os


# Set HuggingFace cache directory
os.environ["HF_HOME"] = "/Users/kristalshrestha/Documents/Code/LLM_Scratch/models"


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.5,
    },
)


# Create a Chat Model instance
# Wrap llm with chat interface
model = ChatHuggingFace(llm=llm)

# Invoke with a prompt
result = model.invoke("What is the capital of India?")

# The result is an object; print the whole object first to see the metadata
# To get just the answer, access the 'content' attribute
print(result.content)
