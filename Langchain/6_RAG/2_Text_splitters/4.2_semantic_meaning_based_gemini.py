from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
import os

load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
gemini_embedding_model = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001", api_key=google_api_key
)


from langchain_experimental.text_splitter import SemanticChunker

sample = """
Farmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world. People all over the world watch the matches and cheer for their favourite teams.


Terrorism is a big danger to peace and safety. It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety.
"""

# initialize the text splitter
text_splitter = SemanticChunker(
    embeddings=gemini_embedding_model,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1,
)


docs = text_splitter.create_documents([sample])
print(len(docs))  # 3

print(docs)
"""Output:
[Document(metadata={}, page_content='\nFarmers were working hard in the fields, preparing the soil and planting seeds for the next season. The sun was bright, and the air smelled of earth and fresh grass. The Indian Premier League (IPL) is the biggest cricket league in the world.'), Document(metadata={}, page_content='People all over the world watch the matches and cheer for their favourite teams. Terrorism is a big danger to peace and safety.'), Document(metadata={}, page_content='It causes harm to people and creates fear in cities and villages. When such attacks happen, they leave behind pain and sadness. To fight terrorism, we need strong laws, alert security forces, and support from people who care about peace and safety. ')]"""
