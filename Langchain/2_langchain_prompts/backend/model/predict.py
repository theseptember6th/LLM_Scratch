from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from langchain_core.prompts import PromptTemplate, load_prompt


def predict_output(user_input: dict):
    os.environ["HF_HOME"] = "/Users/kristalshrestha/Documents/Code/LLM_Scratch/models"
    llm = HuggingFacePipeline.from_model_id(
        model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        pipeline_kwargs={
            "max_new_tokens": 100,
            "temperature": 0.5,
        },
    )
    template = load_prompt("prompt/template.json")
    # using chain
    chain = template | llm
    result = chain.invoke(
        {
            "paper_input": user_input["paper_input"],
            "style_input": user_input["style_input"],
            "length_input": user_input["length_input"],
        }
    )

    return {"output": result}
