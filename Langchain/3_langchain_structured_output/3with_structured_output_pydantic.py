import os
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

# Ensure your .env has HUGGINGFACEHUB_API_TOKEN
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# 1. Setup the Strategy/Endpoint
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",  # Zephyr is great for Chat
    task="text-generation",
    max_new_tokens=512,
    huggingfacehub_api_token=hf_token,
)
model = ChatHuggingFace(llm=llm)


# schema
class Review(BaseModel):
    key_themes: Annotated[
        list[str],
        Field(
            ...,
            description="write down all the themes discussed in the review in a list",
        ),
    ]
    summary: Annotated[str, Field(..., description="A brief summary of the review")]
    sentiment: Annotated[
        Literal["pos", "neg"],
        Field(
            ...,
            description="Return sentiment of the review either negative,positive or neutral",
        ),
    ]
    pros: Annotated[
        Optional[list[str]],
        Field(description="Write down all the pros inside a list", default=None),
    ]
    cons: Annotated[
        Optional[list[str]],
        Field(description="Write down all the cons inside a list", default=None),
    ]
    name: Annotated[
        Optional[str],
        Field(description="Write down the name of the reviewer", default=None),
    ]


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
print(result.summary)
print(result.sentiment)

print(result.name)
