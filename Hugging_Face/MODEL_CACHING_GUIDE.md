# Model Caching Guide for Google Colab + VSCode

## Problem
By default, Hugging Face models are downloaded every time you restart your Colab runtime, which wastes time and bandwidth.

## Solutions (Best to Worst)

### ✅ Option 1: Google Drive (BEST - Permanent Storage)

**Pros:** 
- Models persist forever
- Works across sessions
- One-time download

**Cons:** 
- Slower I/O than local storage
- Uses your Google Drive quota (~3-4GB for Mistral-7B)

```python
from google.colab import drive
drive.mount('/content/drive')

cache_dir = "/content/drive/MyDrive/huggingface_models"
os.makedirs(cache_dir, exist_ok=True)

model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-v0.1",
    device_map="auto",
    torch_dtype=torch.float16,
    load_in_8bit=True,
    max_new_tokens=256,
    model_kwargs={"cache_dir": cache_dir}  # ← KEY LINE
)
```

---

### ✅ Option 2: Default Hugging Face Cache (GOOD - Session Persistent)

**Pros:**
- Fastest loading
- Automatic management
- No setup needed

**Cons:**
- Cleared when runtime disconnects
- Not truly permanent

```python
# Hugging Face automatically caches to /root/.cache/huggingface
# Just add cache_dir to ensure it's used:

cache_dir = "/root/.cache/huggingface"

model = pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-v0.1",
    model_kwargs={"cache_dir": cache_dir},
    # ... other parameters
)
```

---

### ✅ Option 3: Pre-download Models (ADVANCED)

Download once, then load from disk:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# First time only - download and save
model_name = "mistralai/Mistral-7B-v0.1"
cache_dir = "/content/drive/MyDrive/huggingface_models"

# Download model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    torch_dtype=torch.float16,
    load_in_8bit=True,
    device_map="auto"
)

# Save locally (optional for even faster loading)
local_path = "/content/mistral-7b-local"
model.save_pretrained(local_path)
tokenizer.save_pretrained(local_path)

# Later, load from local path
from transformers import pipeline
model = pipeline(
    "text-generation",
    model=local_path,  # Load from local directory
    device_map="auto",
    torch_dtype=torch.float16,
)
```

---

## Quick Comparison

| Method | Speed | Persistence | Setup |
|--------|-------|-------------|-------|
| Google Drive | Slow | ✅ Permanent | Mount drive |
| Default cache | Fast | ⚠️ Session only | None |
| Pre-download | Fastest | ✅ Permanent | Manual save |

---

## Recommended Setup

For most users, use **Google Drive**:

1. Mount drive once per session
2. Set cache_dir to Drive folder
3. First run downloads model (takes time)
4. Subsequent runs are much faster

---

## Environment Variables (Alternative)

You can also set environment variables:

```python
import os
os.environ['HF_HOME'] = '/content/drive/MyDrive/huggingface_models'
os.environ['TRANSFORMERS_CACHE'] = '/content/drive/MyDrive/huggingface_models'

# Now all Hugging Face operations will use this cache
model = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1")
```

---

## Troubleshooting

**Issue:** "No space left on device"
- **Solution:** Use Google Drive, not local Colab storage

**Issue:** Loading is still slow from Drive
- **Solution:** After loading from Drive once, model stays in RAM. Only first load is slow.

**Issue:** Model not found in cache
- **Solution:** Ensure `cache_dir` is consistent across runs

---

## Storage Requirements

- Mistral-7B: ~3-4 GB
- Larger models (13B): ~7-8 GB
- Make sure you have enough Google Drive space
