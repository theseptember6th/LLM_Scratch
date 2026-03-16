# define prompt
from langchain_core.prompts import PromptTemplate


prompt1 = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Summarize the following text\n {text}", input_variables=["text"]
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


# define runnables,chains
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableSequence,
    RunnableBranch,
)

report_gen_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    # (condition 1,run this chain1/runnable)
    # (condition2,run this chain2/runnable)
    # default condition
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough(),  # default
)
# merge chains/runnables
final_chain = RunnableSequence(report_gen_chain, branch_chain)

result = final_chain.invoke({"topic": "Russia vs Ukraine"})
print(result)


"""Output:
 Title: Russia-Ukraine Conflict: A Complex and Multifaceted Dispute

Executive Summary:
The Russia-Ukraine conflict is a complex and multifaceted dispute that has its roots in historical, cultural, political, and economic differences between the two countries. This report provides an in-depth analysis of the conflict, its causes, developments, and the implications for international relations.

Background:
The Russia-Ukraine conflict began in earnest in 2014 when Ukraine's then-president, Viktor Yanukovych, announced that he would abandon a proposed association agreement with the European Union in favor of closer ties with Russia. This decision sparked widespread protests in Ukraine, culminating in the Euromaidan Revolution, which led to Yanukovych's ousting in February 2014. In response, Russia annexed Crimea, a region historically and culturally tied to Russia but administratively part of Ukraine. This annexation was followed by a rebellion in Eastern Ukraine, primarily in the Donetsk and Luhansk regions, which declared independence as the Donetsk People's Republic (DPR) and the Luhansk People's Republic (LPR).

Causes:
Historical: The Russia-Ukraine conflict is rooted in long-standing historical and cultural ties between the two countries. Russia and Ukraine share a common Slavic heritage and have been politically and culturally interlinked for centuries. The region of Ukraine has been a battleground for various powers throughout history, with Russia and Poland, among others, vying for influence.

Political: The political dimension of the conflict arises from Ukraine's desire for closer ties with Europe and Russia's opposition to this development. Ukraine's decision to seek closer ties with the EU was perceived as a threat to Russia's influence in the region and its strategic interests.

Economic: Economic factors also come into play, with Ukraine being a major transit country for Russian natural gas exports to Europe. Russia's control over this supply route is crucial for its energy security and economic interests. Ukraine, on the other hand, has sought to diversify its energy sources and reduce its dependence on Russian gas.

Military: The conflict has a military dimension, with the DPR and LPR engaging in armed confrontations with Ukrainian """
