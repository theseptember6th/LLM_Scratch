# ============================================
# OPTION 2: Default Hugging Face Cache
# ============================================
# Best for: Fast loading within the same session
# Cache persists during runtime but cleared on disconnect
# No Google Drive needed - fastest performance

from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import torch
import os

# Use default Hugging Face cache location
cache_dir = "/root/.cache/huggingface"
os.makedirs(cache_dir, exist_ok=True)

print(f"✓ Using cache directory: {cache_dir}")
print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# Load model with local caching
print("\n" + "="*60)
print("Loading model from cache (or downloading if first time)...")
print("="*60)
print("NOTE: Cache will persist during this session only")
print("="*60)

model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-v0.1",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    max_new_tokens=256,
    model_kwargs={"cache_dir": cache_dir}  # Saves to local cache
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
