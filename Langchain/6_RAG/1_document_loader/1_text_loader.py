# load the document
from langchain_community.document_loaders import TextLoader

loader = TextLoader(
    file_path=r"./1_cricket_peom_generated_from_chatgpt.txt", encoding="utf-8"
)  # encoding is specified because there are special characters besides normal alphabets

docs = loader.load()
print(docs)
"""Output:
[Document(metadata={'source': './1_cricket_peom_generated_from_chatgpt.txt'}, page_content='“The Song of the Cricket Field” 🏏\n\nUnder the sun on a grassy stage,\nWhere bat meets ball and crowds engage,\nA bowler runs with silent might,\nTo send the leather ball in flight.\n\nThe batter stands with steady gaze,\nDreaming of glory, runs, and praise,\nA swing, a crack — the ball takes wing,\nAcross the field the cheers now ring.\n\nFielders chase through dust and green,\nA diving stop, a catch unseen,\nThe scoreboard ticks, the tension grows,\nWith every ball the drama flows.\n\nFrom village grounds to stadium light,\nCricket unites in shared delight,\nA game of patience, skill, and art—\nPlayed not just with hands, but heart. ❤️🏏')]"""
print(type(docs))  # <class 'list'>
print(len(docs))  # 1
print(docs[0])
"""Output:
page_content='“The Song of the Cricket Field” 🏏

Under the sun on a grassy stage,
Where bat meets ball and crowds engage,
A bowler runs with silent might,
To send the leather ball in flight.

The batter stands with steady gaze,
Dreaming of glory, runs, and praise,
A swing, a crack — the ball takes wing,
Across the field the cheers now ring.

Fielders chase through dust and green,
A diving stop, a catch unseen,
The scoreboard ticks, the tension grows,
With every ball the drama flows.

From village grounds to stadium light,
Cricket unites in shared delight,
A game of patience, skill, and art—
Played not just with hands, but heart. ❤️🏏' metadata={'source': './1_cricket_peom_generated_from_chatgpt.txt'}"""
print(type(docs[0]))  # <class 'langchain_core.documents.base.Document'>

print(docs[0].page_content)
"""Output:
“The Song of the Cricket Field” 🏏

Under the sun on a grassy stage,
Where bat meets ball and crowds engage,
A bowler runs with silent might,
To send the leather ball in flight.

The batter stands with steady gaze,
Dreaming of glory, runs, and praise,
A swing, a crack — the ball takes wing,
Across the field the cheers now ring.

Fielders chase through dust and green,
A diving stop, a catch unseen,
The scoreboard ticks, the tension grows,
With every ball the drama flows.

From village grounds to stadium light,
Cricket unites in shared delight,
A game of patience, skill, and art—
Played not just with hands, but heart. ❤️🏏
"""
print(docs[0].metadata)
"""Output:{'source': './1_cricket_peom_generated_from_chatgpt.txt'}"""

# define prompts
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Write a summary for the following poem -\n {poem}",
    input_variables=["poem"],
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


# define output parser

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# define chain

chain = prompt | model | parser
result = chain.invoke({"poem": docs[0].page_content})
print(f"\n\n\n Summary :\n\n{result}")

"""OUTPUT:
Summary :

 "The Song of the Cricket Field" is a poetic tribute to the beloved sport of cricket. The scene is set under the sun on a grassy cricket field where the action unfolds between the bat and ball. The bowler, filled with silent determination, runs to send the ball flying towards the batsman. The batsman, filled with dreams of glory, steadily faces the challenge, anticipating the perfect swing and the satisfying crack of the ball as it takes flight. The fielders, dressed in green, chase after the ball through dust and grass, making diving stops and performing unseen catches. The excitement builds as the scoreboard ticks, and the tension grows with every ball. From village grounds to stadiums, cricket unites people with its game of patience, skill, and art, played not just with hands, but from the heart."""
