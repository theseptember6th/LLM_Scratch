# 1 defining prompts
from langchain_core.prompts import PromptTemplate

# 3.1 defining output structure parser through pydantic
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Annotated, Literal


class Feedback(BaseModel):
    sentiment: Annotated[
        Literal["positive", "negative"],
        Field(..., description="Give the sentiment of the feedback"),
    ]


parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the following feedback text into positive or negative.\n {feedback} \n {format_instruction}",
    input_variables=["feedback"],
    partial_variables={"format_instruction": parser2.get_format_instructions()},
)
# 2.defining model
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


# 4.defining chain
classifier_chain = prompt1 | model | parser2

# # 5.invoking the chain

# result = classifier_chain.invoke({"feedback": "This is a wonderful smartphone."})
# print(result)
# print(result.sentiment)

# """Important Output:
# sentiment='positive'
# positive
# """

# now for runnable branch

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback \n {feedback}",
    input_variables=["feedback"],
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to this feedback \n {feedback}",
    input_variables=["feedback"],
)
# 3.defining parser
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# if-else statement of langchain universe

from langchain_core.runnables import RunnableBranch, RunnableLambda

branch_chain = RunnableBranch(
    # (condition to check, which chain to run)
    # (default chain is also needed)
    (lambda x: x.sentiment == "positive", prompt2 | model | parser),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(
        lambda x: "could not find sentiment",
    ),
)

# final chain

chain = classifier_chain | branch_chain
# result = chain.invoke({"feedback": "This is a terrible phone"})
result = chain.invoke({"feedback": "This is a wonderful phone"})
print(result)


# to visualize the graph
chain.get_graph().print_ascii()
