from langchain_huggingface import (
    HuggingFaceEndpoint,
    ChatHuggingFace,
    HuggingFacePipeline,
)

from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

# hugging face free api endpoint is unreliable so trying locally
# load_dotenv()
# api_key = os.getenv("Hugging_face_api_token")

# llm = HuggingFaceEndpoint(
#     # repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     repo_id="google/gemma-2-2b-it",
#     task="text-generation",
#     huggingfacehub_api_token=api_key,
# )
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

# 1st prompt->Detailed report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)
# 2nd prompt->summary
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text ./n {text}",
    input_variables=["text"],
)

from langchain_core.output_parsers import StrOutputParser


parser = StrOutputParser()
chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic": "blackhole"})

print(result)
