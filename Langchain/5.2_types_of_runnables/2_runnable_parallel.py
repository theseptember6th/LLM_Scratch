# defining prompts
from langchain_core.prompts import PromptTemplate


prompt1 = PromptTemplate(
    template="Generate a tweet about {topic}", input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Generate a Linkedin post about {topic} ", input_variables=["topic"]
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

# define string output parser

from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# defining parallel chain

from langchain_core.runnables import RunnableParallel, RunnableSequence

parallel_chain = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt1, model, parser),
        "linkedin": RunnableSequence(prompt2, model, parser),
    }
)
result = parallel_chain.invoke({"topic": "AI"})
print(result)
"""Output:
{'tweet': ' "Discovering the power of #ArtificialIntelligence every day! From automating mundane tasks to revolutionizing industries, AI is transforming our world in incredible ways. Let\'s embrace the future and continue pushing the boundaries of what\'s possible. #Tech #Innovation #AI #FutureTech" #MachineLearning #DeepLearning #Robotics #AIethics #NeuralNetworks #DataScience #TechNews #InnovationMonday #AIchat #AIassistant #AInews #techlife #innovation #digitaltransformation #futureisnow #intelligence #technology #innovationcommunity #innovationmindset #AIadvancements #AIeducation #AIresearch #AItools #AIworld #AIupdate #AIcommunity #AIfuture #AIstrategy #AIintegration #AIevolution #AIdiscovery #AIprogress #AIinnovations #AIpotential #AIinsights #AIimpact #AIinteresting #AIapplications #AInewsflash #AIdaily #AIinspiration #AIthoughts #AIthinking #AItrending #AIadvice #AIdialogue #AIminds #AItalks #AIdiscussion #AIdiscussions #AIdialogue #AIdebate #AInnovation #AIdiscovery #AIdevelopment #AIresearchers #AIscientists #AIengineers #AIdigerati #AIminds #AIfuture #AImindsAI #AImindsTech #AImindset #AIcommunity #AItogether #AItomorrow #AItrends #AInspiration #AIdreams #AIleadership #AIrevolution #AIminded #AInext #AIfutureisnow #AIvision #AIvisionary #AIorigin #AIevolving #AIjourney #AIadvance #AIlearning #AIprogression #AIsuccess #AIambition #AIdevelopment #AIpotential #AIadvancement #AIgrowth #AIprogress #AIdrive #AIachievement #AIimpactful #AIimpact #AIdiscovery #AIexploration #AIinnovation #AIevolving #AInextlevel #AIchallenges #AIfutureis', 'linkedin': " 💡 Exciting times in the world of technology! Artificial Intelligence (AI) is now a reality and is transforming industries in ways we never thought possible. From automating repetitive tasks to providing personalized recommendations and even diagnosing complex medical conditions, the potential applications of AI are endless.\n\nBut what sets AI apart from other technologies? It's not just about automation or efficiency. It's about creating intelligent machines that can learn from data, adapt to new situations, and even make decisions on their own.\n\nAs we continue to explore the possibilities of AI, it's important to remember that its implementation should be ethical and beneficial for all. Let's use AI to solve real-world problems, enhance human capabilities, and create a better future for everyone.\n\nStay tuned as I share more insights and updates about AI and its impact on our world. #ArtificialIntelligence #AI #TechTrends #DigitalTransformation #Innovation #FutureOfWork #EthicsInTech"}"""

print(result["tweet"])
print(result["linkedin"])
