# loading the content from the webpage
from langchain_community.document_loaders import WebBaseLoader
import re

url = """https://www.daraz.com.np/products/apple-macbook-air-13-inch-m4-16gb256gb-evostore-i374215735-s1649980045.html?c=&channelLpJumpArgs=&clickTrackInfo=query%253Amacbook%253Bnid%253A374215735%253Bsrc%253ALazadaMainSrp%253Brn%253A0539b7e2623bb0fa483228d639ad2974%253Bregion%253Anp%253Bsku%253A374215735_NP%253Bprice%253A152500%253Bclient%253Adesktop%253Bsupplier_id%253A8660%253Bsession_id%253A%253Bbiz_source%253Ahttps%253A%252F%252Fwww.daraz.com.np%252F%253Bslot%253A0%253Butlog_bucket_id%253A470687%253Basc_category_id%253A7902%253Bitem_id%253A374215735%253Bsku_id%253A1649980045%253Bshop_id%253A8922%253BtemplateInfo%253A&freeshipping=0&fs_ab=1&fuse_fs=&lang=en&location=Bagmati%20Province&price=1.525E%205&priceCompare=skuId%3A1649980045%3Bsource%3Alazada-search-voucher%3Bsn%3A0539b7e2623bb0fa483228d639ad2974%3BoriginPrice%3A15250000%3BdisplayPrice%3A15250000%3BsinglePromotionId%3A50000039753004%3BsingleToolCode%3ApromPrice%3BvoucherPricePlugin%3A0%3Btimestamp%3A1773665366728&ratingscore=5.0&request_id=0539b7e2623bb0fa483228d639ad2974&review=1&sale=4&search=1&source=search&spm=a2a0e.searchlist.list.0&stock=1"""
loader = WebBaseLoader(web_path=url)
docs = loader.load()
# print(len(docs))  # 1
"""one url is always 1 document object,needs a list of n urls for n document objects"""
content = docs[0].page_content

# Remove multiple consecutive newlines
content = re.sub(r"\n\s*\n", "\n", content)
# Remove leading/trailing whitespace from each line
lines = [line.strip() for line in content.splitlines()]
# Filter out empty lines
cleaned_lines = [line for line in lines if line]
cleaned_content = "\n".join(cleaned_lines)

# print(cleaned_content)

"""Output:Apple | MacBook Air (13-inch) M4 - 16GB/256GB - EvoStore | Daraz.com.np
Product Images
Apple%20%7C%20MacBook%20Air%20(13-inch)%20M4%20-%2016GB/256GB%20-%20EvoStore%20-%20Image%201
Apple%20%7C%20MacBook%20Air%20(13-inch)%20M4%20-%2016GB/256GB%20-%20EvoStore%20-%20Image%202
Apple%20%7C%20MacBook%20Air%20(13-inch)%20M4%20-%2016GB/256GB%20-%20EvoStore%20-%20Image%203
Apple%20%7C%20MacBook%20Air%20(13-inch)%20M4%20-%2016GB/256GB%20-%20EvoStore%20-%20Image%204
Apple%20%7C%20MacBook%20Air%20(13-inch)%20M4%20-%2016GB/256GB%20-%20EvoStore%20-%20Image%205
Apple%20%7C%20MacBook%20Air%20(13-inch)%20M4%20-%2016GB/256GB%20-%20EvoStore%20-%20Image%206
Apple%20%7C%20MacBook%20Air%20(13-inch)%20M4%20-%2016GB/256GB%20-%20EvoStore%20-%20Image%207
Apple%20%7C%20MacBook%20Air%20(13-inch)%20M4%20-%2016GB/256GB%20-%20EvoStore%20-%20Image%208
Apple%20%7C%20MacBook%20Air%20(13-inch)%20M4%20-%2016GB/256GB%20-%20EvoStore%20-%20Image%209
Save More on App
Download the App
Become a Seller
Help & Support
Help Center
Contact Customer Care
Shipping & Delivery
Payment
Order
Login
Sign Up
Manage My Account
My Orders
My Wishlist & Followed Stores
My Reviews
My Returns & Cancellations
Log out
भाषा परिवर्तन
EN / English
NE / Nepali
Categories
Apple | MacBook Air (13-inch) M4 - 16GB/256GB - EvoStoreNo RatingsBrand: AppleMore Laptops from AppleColor FamilyMidnightQuantity
Customer Care
Help Center
How to Buy
Returns & Refunds
Contact Us
Daraz
About Daraz
Careers
Daraz Blog
Terms & Conditions
Privacy Policy
Digital Payments
Daraz Customer University
Daraz Affiliate Program
Review & Win
Meet the winners
Daraz University
Sell on Daraz
Code of Conduct
Happy Shopping
Download App
Payment Methods
Verified by
Daraz International
Pakistan
Bangladesh
Sri Lanka
Myanmar
Nepal
Follow Us
© Daraz 2026"""

# define prompts
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text -\n {text}",
    input_variables=["question", "text"],
)

# defining model
from langchain_core.prompts.dict import _insert_input_variables
from langchain_huggingface import (
    ChatHuggingFace,
    HuggingFacePipeline,
    HuggingFaceEndpoint,
)

from dotenv import load_dotenv
import os

# hugging face free api endpoint is unreliable so trying locally
load_dotenv()
api_key = os.getenv("Hugging_face_api_token")

llm = HuggingFaceEndpoint(
    # repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    # repo_id="google/gemma-2-2b-it",
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    huggingfacehub_api_token=api_key,
)
# os.environ["HF_HOME"] = "/Users/kristalshrestha/Documents/Code/LLM_Scratch/models"
# define the model
# llm = HuggingFacePipeline.from_model_id(
#     # this tinyllama is vert small for structured output tasks
#     # model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     model_id="google/gemma-2-2b-it",
#     task="text-generation",
#     pipeline_kwargs={
#         "max_new_tokens": 100,
#         "temperature": 0.5,
#     },
# )
model = ChatHuggingFace(llm=llm)


# defining parser
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()


# define chain

chain = prompt | model | parser
result = chain.invoke(
    {
        "question": "What is the product that we are talking about?",
        "text": cleaned_content,
    }
)

print(result)
