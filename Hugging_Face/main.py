import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-v0.1",
    device=1,
    max_length=256,
    truncation=True,
)
