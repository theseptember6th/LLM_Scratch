from langchain_core.prompts import PromptTemplate
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

from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser()
template = PromptTemplate(
    template="Give me the name,age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={
        "format_instruction": parser.get_format_instructions()
    },  # this will give instruction to give in json
)

# prompt = template.format()

# # print(prompt)
# # """Give me the name,age and city of a fictional person
# #  Return a JSON object."""

# result = model.invoke(prompt)
# final_result = parser.parse(result.content)

# using chains
chain = template | model | parser
final_result = chain.invoke({})
print(final_result)
print(type(final_result))
print(final_result["name"])
