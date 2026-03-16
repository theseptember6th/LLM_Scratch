# define template
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Write a joke about {topic}", input_variables=["topic"]
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

# define runnable lambda chain
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSequence,
    RunnableParallel,
    RunnableLambda,
)

joke_gen_chain = RunnableSequence(prompt, model, parser)


def word_count(text):
    return len(text.split())


parallel_chain = RunnableParallel(
    {"joke": RunnablePassthrough(), "word_count": RunnableLambda(word_count)}
)

# or directly using lambda function

# parallel_chain = RunnableParallel(
#     {
#         "joke": RunnablePassthrough(),
#         "word_count": RunnableLambda(lambda x: len(x.split())),
#     }
# )


final_chain = RunnableSequence(joke_gen_chain, parallel_chain)
result = final_chain.invoke({"topic": "AI"})
print(result)

"""Output:
{'joke': ' Why don\'t computers take their hats off during the National Anthem?\n\nBecause they don\'t want to risk a "logic error" during the salute! 💻💪🇺🇸\n\n(Note: This joke is meant to be light-hearted and in no way implies that AI or computers have the ability to experience national pride or salute.)', 'word_count': 50}"""


# formatting ur final result as ur own way.

print("\n final_result\n\n\n")
final_result = f"""{result["joke"]} \n word count - {result["word_count"]}"""
print(final_result)


"""Output:

 final_result



 Why don't computers take their hats off during the National Anthem?

Because they don't want to risk a "logic error" during the salute! 💻💪🇺🇸

(Note: This joke is meant to be light-hearted and in no way implies that AI or computers have the ability to experience national pride or salute.) 
 word count - 50"""
