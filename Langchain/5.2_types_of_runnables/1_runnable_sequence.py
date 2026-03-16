# define prompts
from langchain_core.prompts import PromptTemplate

prompt1 = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Explain the following joke - {text}", input_variables=["text"]
)

# define model
from langchain_core.prompts.dict import _insert_input_variables
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
    HuggingFaceEndpoint,
)

from dotenv import load_dotenv
import os

# hugging face free api endpoint is unreliable so trying locally
load_dotenv()
api_key = os.getenv("Hugging_face_api_token")

llm = HuggingFaceEndpoint(
    # repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # repo_id="google/gemma-2-2b-it",
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=api_key,
)
# os.environ["HF_HOME"] = "/Users/kristalshrestha/Documents/Code/LLM_Scratch/models"
# define the model
# llm = HuggingFacePipeline.from_model_id(
#     # this tinyllama is vert small for structured output tasks
#     # model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     model_id="google/gemma-2-2b-it",
#     task="text-generation",
#     pipeline_kwargs={
#         "max_new_tokens": 100,
#         "temperature": 0.5,
#     },
# )
model = ChatHuggingFace(llm=llm)

# define parser
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# define runnable chains
from langchain_core.runnables import RunnableSequence

chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

# invoking chain

result = chain.invoke({"topic": "AI"})
print(result)
