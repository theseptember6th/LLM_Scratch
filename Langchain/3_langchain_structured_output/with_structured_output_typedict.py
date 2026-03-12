import os
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from typing import TypedDict

os.environ["HF_HOME"] = "/Users/kristalshrestha/Documents/Code/LLM_Scratch/models"
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "max_new_tokens": 100,
        "temperature": 0.5,
    },
)
model = ChatHuggingFace(llm=llm)


# schema
class Review(TypedDict):
    summary: str
    sentiment: str


structured_model = model.with_structured_output(Review)

result = structured_model.invoke(
    """I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.

The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100x actually works well for distant objects, but anything beyond 30x loses quality.

However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.

Pros:
Insanely powerful processor (great for gaming and productivity)
Stunning 200MP camera with incredible zoom capabilities
Long battery life with fast charging
S-Pen support is unique and useful
                                 
Review by Kristal Shrestha """
)

print(result)
print(type(result))  # dict
print(result["summary"])
print(result["sentiment"])
