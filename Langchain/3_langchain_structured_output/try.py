from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
import os
from dotenv import load_dotenv

load_dotenv()

# Ensure your .env has HUGGINGFACEHUB_API_TOKEN
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 1. Setup the Strategy/Endpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # Zephyr is great for Chat
    task="text-generation",
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token,
)

# 2. Wrap it in the Chat Interface
chat_model = ChatHuggingFace(llm=llm)

# 3. Invoke with a Message object (standard for ChatModels)
from langchain_core.messages import HumanMessage

response = chat_model.invoke(
    [HumanMessage(content="Explain quantum physics in one sentence.")]
)

print(response.content)
