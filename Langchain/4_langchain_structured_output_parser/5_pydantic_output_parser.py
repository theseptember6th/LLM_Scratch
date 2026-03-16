# pip show langchain
## this won't work because on newer version.In this version StructuredOutputParser was removed completely. That is why none of the old imports work. Also no data validation

from ast import parse
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

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Annotated


# pydantic schema


class Person(BaseModel):
    name: Annotated[str, Field(..., description="Name of the person")]
    age: Annotated[int, Field(..., gt=18, description="Age of the person")]
    city: Annotated[
        str, Field(..., description="Name of the city the person belongs to")
    ]


parser = PydanticOutputParser(pydantic_object=Person)


template = PromptTemplate(
    template="Generate the name,age and city of a fictional {place} person \n {format_instruction}",
    input_variables=["place"],
    partial_variables={"format_instruction": parser.get_format_instructions()},
)

# without chain

# prompt = template.invoke({"place": "indian"})
# print(
#     prompt
# )  ##text='Generate the name,age and city of a fictional indian person \n The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nAs an example, for the schema {"properties": {"foo": {"title": "Foo", "description": "a list of strings", "type": "array", "items": {"type": "string"}}}, "required": ["foo"]}\nthe object {"foo": ["bar", "baz"]} is a well-formatted instance of the schema. The object {"properties": {"foo": ["bar", "baz"]}} is not well-formatted.\n\nHere is the output schema:\n```\n{"properties": {"name": {"description": "Name of the person", "title": "Name", "type": "string"}, "age": {"description": "Age of the person", "exclusiveMinimum": 18, "title": "Age", "type": "integer"}, "city": {"description": "Name of the city the person belongs to", "title": "City", "type": "string"}}, "required": ["name", "age", "city"]}\n```'
# result = model.invoke(prompt)
# final_result = parser.parse(result.content)
# print(final_result)


# using chains

chain = template | model | parser
result = chain.invoke({"place": "nepalese"})
print(result)
