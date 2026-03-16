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

# define passthrough runnable chains
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSequence,
    RunnableParallel,
)

joke_gen_chain = RunnableSequence(prompt1, model, parser)
parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "explaination": RunnableSequence(prompt2, model, parser),
    }
)

final_chain = RunnableSequence(joke_gen_chain, parallel_chain)


# invoking chain

result = final_chain.invoke({"topic": "AI"})
print(result)

""""Output:
{'joke': " Why don't AI ever get lost?\n\nBecause they always find their way back to the mainframe! (or server, if you prefer a more modern twist)\n\nBut in all seriousness, here's a joke that puts AI in a more relatable situation:\n\nWhy don't AI go to parties?\n\nBecause they don't know how to small talk! They just keep asking the same questions over and over again, trying to learn and understand. But at least they never run out of topics!", 'explaination': " This joke is based on the common perception that artificial intelligence (AI) are programmed systems that follow predetermined instructions and lack the ability to experience things or engage in social interactions like humans. In the first joke, the punchline plays on the idea that since AI always follow their programming, they never get lost because they can always find their way back to their main source of instructions or data.\n\nIn the second joke, the punchline is that AI don't attend parties because they struggle with small talk, a common social activity that involves light and often trivial conversations. The implication is that AI don't have the ability to engage in these types of conversations and may be repetitive in their interactions. However, despite this limitation, the joke suggests that at least AI never run out of topics to discuss, as they can analyze and learn from their surroundings and interactions."}
"""
