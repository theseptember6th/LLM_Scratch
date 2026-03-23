from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
import os
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv


def predict_output(user_input: dict):
    load_dotenv()
    api_key = os.getenv("Hugging_face_api_token")
    # Create llm endpoint
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        task="text-generation",
        huggingfacehub_api_token=api_key,
    )
    # Wrap with chat interface
    model = ChatHuggingFace(llm=llm)

    template = load_prompt("./prompt/template.json")
    # using chain
    chain = template | model
    result = chain.invoke(
        {
            "paper_input": user_input["paper_input"],
            "style_input": user_input["style_input"],
            "length_input": user_input["length_input"],
        }
    )

    return {"output": result.content}
