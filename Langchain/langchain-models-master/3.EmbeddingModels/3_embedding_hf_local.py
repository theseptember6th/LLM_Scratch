from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# text = "Delhi is the capital of India"

# vector = embedding.embed_query(text)

documents = [
    "kathmandu is the capital of Nepal",
    "Pokhara is the capital of province 4",
    "Kristal is my Name",
]

vector = embedding.embed_documents(documents)
print(str(vector))
