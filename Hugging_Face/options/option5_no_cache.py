# ============================================
# OPTION 5: No Caching (Always Downloads)
# ============================================
# Best for: Understanding the problem - DON'T USE THIS!
# This is your original code without caching
# Model downloads every time runtime restarts

from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import torch

print("WARNING: This script does NOT cache the model!")
print("Model will be re-downloaded every time runtime restarts.")
print("Use one of the other options for persistent caching.\n")

print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# Load model WITHOUT caching (downloads every time)
print("\n" + "="*60)
print("Downloading model (no cache)...")
print("="*60)

model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-v0.1",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    max_new_tokens=256,
    # NO cache_dir specified - downloads to temporary location
)

print("✓ Model loaded (but not cached for next time)\n")

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
