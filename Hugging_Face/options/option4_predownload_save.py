# ============================================
# OPTION 4: Pre-Download and Save Locally (ADVANCED)
# ============================================
# Best for: Maximum control and fastest loading
# Two-step process: 1) Download once, 2) Load from local path
# Ideal for repeated use of the same model

from transformers.utils.logging import set_verbosity_error
set_verbosity_error()

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
import torch
import os

# ============================================
# STEP 1: Download and Save Model (Run Once)
# ============================================

# Choose storage location
# Option A: Google Drive (permanent)
from google.colab import drive
drive.mount('/content/drive')
save_directory = "/content/drive/MyDrive/models/mistral-7b"

# Option B: Local storage (session only, but faster)
# save_directory = "/content/models/mistral-7b"

os.makedirs(save_directory, exist_ok=True)

model_name = "mistralai/Mistral-7B-v0.1"

print("="*60)
print("STEP 1: Downloading Model (Only needed once)")
print("="*60)

# Check if model already exists
if os.path.exists(os.path.join(save_directory, "config.json")):
    print(f"✓ Model already exists at {save_directory}")
    print("✓ Skipping download...")
else:
    print(f"Downloading {model_name}...")
    print("This may take several minutes on first run...")
    
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Download model
    model_download = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        device_map="auto"
    )
    
    # Save to local directory
    print(f"\nSaving model to {save_directory}...")
    model_download.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    
    print("✓ Model saved successfully!")
    
    # Clean up
    del model_download
    del tokenizer
    torch.cuda.empty_cache()

# ============================================
# STEP 2: Load Model from Local Directory
# ============================================

print("\n" + "="*60)
print("STEP 2: Loading Model from Local Directory")
print("="*60)
print(f"Loading from: {save_directory}")

print(f"✓ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

# Load model from saved directory (very fast!)
model = pipeline(
    "text-generation",
    model=save_directory,  # Load from local path
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    max_new_tokens=256,
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
