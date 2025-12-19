# Generated from: main_colab.ipynb (with model caching)
# Converted at: 2025-12-18T14:51:15.335Z

from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import torch
import os

# ============================================
# OPTION 1: Mount Google Drive (Recommended)
# ============================================
# Uncomment these lines to mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')
# 
# # Set cache directory to Google Drive
# cache_dir = "/content/drive/MyDrive/huggingface_models"

# ============================================
# OPTION 2: Use Colab's local cache (faster but temporary)
# ============================================
# This persists within a session but may be cleared
cache_dir = "/root/.cache/huggingface"

# Create cache directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)

print(f"Using cache directory: {cache_dir}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================
# Load model with caching
# ============================================
print("\nLoading model (this will use cached version if available)...")

model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-v0.1",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    max_new_tokens=256,
    model_kwargs={"cache_dir": cache_dir}  # This enables caching!
)

print("Model loaded successfully!")

# ============================================
# Create LangChain pipeline
# ============================================
llm = HuggingFacePipeline(pipeline=model)

template = PromptTemplate.from_template(
    "Explain {topic} in detail for a {age} year old to understand."
)

chain = template | llm

# ============================================
# Interactive input
# ============================================
topic = input("Topic: ")
age = input("Age: ")

response = chain.invoke({
    "topic": topic,
    "age": age
})

print("\n" + "="*50)
print("RESPONSE:")
print("="*50)
print(response)
