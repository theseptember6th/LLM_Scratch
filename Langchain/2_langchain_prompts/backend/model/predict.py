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

    # template
    # template = PromptTemplate(
    #     template=""" Please summarize the research paper titled "{paper_input}" with the following
    # specifications:
    # Explanation Style: {style_input}
    # Explanation Length: {length_input}
    # 1. Mathematical Details:
    # - Include relevant mathematical equations if present in the paper.
    # - Explain the mathematical concepts using simple, intuitive code snippets
    # where applicable.
    # 2. Analogies:
    # - Use relatable analogies to simplify complex ideas.
    # If certain information is not available in the paper, respond with: "Insufficient
    # information available" instead of guessing.
    # Ensure the summary is clear, accurate, and aligned with the provided style and
    # length.
    # """,
    #     input_variables=["paper_input", "style_input", "length_input"],
    #     validate_template=True,
    # )
    template = load_prompt("prompt/template.json")
    # fill the placeholders
    # prompt = template.format(
    #     paper_input=user_input["paper_input"],
    #     style_input=user_input["style_input"],
    #     length_input=user_input["length_input"],
    # )
    # model = ChatHuggingFace(llm=llm)

    # result = model.invoke(prompt)

    # return {"output": result.content}

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
