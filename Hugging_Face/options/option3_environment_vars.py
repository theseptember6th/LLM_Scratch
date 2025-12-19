# ============================================
# OPTION 3: Environment Variables (Global Cache)
# ============================================
# Best for: Setting cache location globally for all HF operations
# Works automatically without specifying cache_dir each time
# Can use with Google Drive for permanent storage

from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import torch
import os

# ============================================
# Choose your cache location (uncomment one):
# ============================================

# Option A: Google Drive (permanent storage)
from google.colab import drive
drive.mount('/content/drive')
cache_location = "/content/drive/MyDrive/huggingface_models"

# Option B: Local cache (session only - faster)
# cache_location = "/root/.cache/huggingface"

# Set environment variables globally
os.environ['HF_HOME'] = cache_location
os.environ['TRANSFORMERS_CACHE'] = cache_location
os.environ['HF_DATASETS_CACHE'] = cache_location

os.makedirs(cache_location, exist_ok=True)

print(f"✓ Environment variables set")
print(f"✓ HF_HOME: {os.environ['HF_HOME']}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# Load model (automatically uses environment variables)
print("\n" + "="*60)
print("Loading model (using global environment cache)...")
print("="*60)

model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-v0.1",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    max_new_tokens=256,
    # No need to specify cache_dir - uses env variables automatically!
)

print("✓ Model loaded successfully!\n")

# Create LangChain pipeline
llm = HuggingFacePipeline(pipeline=model)

# Create prompt template
template = PromptTemplate.from_template(
    "Explain {topic} in detail for a {age} year old to understand."
)

# Create chain
chain = template | llm

# Interactive input
print("="*60)
print("READY TO GENERATE!")
print("="*60)
topic = input("Topic: ")
age = input("Age: ")

print("\nGenerating response...\n")

response = chain.invoke({
    "topic": topic,
    "age": age
})

print("\n" + "="*60)
print("RESPONSE:")
print("="*60)
print(response)
