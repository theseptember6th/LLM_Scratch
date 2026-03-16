# 1. defining prompts
from langchain_core.prompts import PromptTemplate

# first prompt for generating detail from use rgiven text
prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"],
)
# second prompt to generate quiz from the same usergiven text
prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=["text"],
)

# third prompt for merging them both
prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes->{notes} and quiz->{quiz}",
    input_variables=["notes", "quiz"],
)

# 2.defining two models
# first gemini model for detail and merge
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
googles_api_key = os.getenv("GOOGLE_API_KEY")
model1 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite", google_api_key=googles_api_key
)

# defining another model which will be local for quiz
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
    HuggingFaceEndpoint,
)


# hugging face free api endpoint is unreliable so trying locally
local_model_api_key = os.getenv("Hugging_face_api_token")

llm = HuggingFaceEndpoint(
    # repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # repo_id="google/gemma-2-2b-it",
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=local_model_api_key,
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
model2 = ChatHuggingFace(llm=llm)


# 3.defining parsers
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

# 4.defining chains

# 4.1 defining parallel chains
# for parallel chains ,u need runnable parallel library

from langchain_core.runnables import RunnableParallel


parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser,
    }
)

# 4.2 defining merge_chain
merge_chain = prompt3 | model1 | parser

# 4.3 defining the final flow of chains
chain = parallel_chain | merge_chain

# 5 invoking the chain
# 5.1 user_input
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

result = chain.invoke({"text": text})
print(result)

# to visualize graph
chain.get_graph().print_ascii()
