from langchain_experimental.text_splitter import SemanticChunker

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""


# define hugging face endpoint embeddings
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("Hugging_face_api_token")
hf_embeddings_api = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",  # lightweight
    # model="Qwen/Qwen3-Embedding-8B",
    # task="feature-extraction",# heavyweight but uses a lot of quota token
    huggingfacehub_api_token=api_key,
)

# initialize the text splitter
text_splitter = SemanticChunker(
    embeddings=hf_embeddings_api,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)


docs = text_splitter.create_documents([sample])
print(len(docs))  # 2

print(docs)
"""[Document(metadata={}, page_content='\nFarmers were working hard in the fields, preparing the soil and planting seeds for the next season.'), Document(metadata={}, page_content='The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams. Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety. ')]"""
