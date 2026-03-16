from langchain_community.document_loaders import JSONLoader

# Extract both question and answer fields, combining them into document content
loader = JSONLoader(
    file_path="./6_cleaned_socialInclusion.json",
    jq_schema='.[] | "Question: " + .question + "\n\nAnswer: " + .answer',
    text_content=False,
)

docs = loader.load()

# print(len(docs))  # 1009
# print(docs[0])
"""Output:
page_content='Question: My teacher insults my caste in front of other students and creates hatred in the classroom. Can I take legal action?

Answer: **Yes. Acts that harm harmony between communities are prohibited.**

**Law says:**
- Muluki Criminal (Code) Act, 2074 (2017) Section 65: Prohibition of acts on the ground of caste or community that are prejudicial to harmonious relationships.

**Punishment:**
- Imprisonment for a term not exceeding one year and a fine not exceeding ten thousand rupees.

**What to do:**
1. Write down dates, words used, and names of student witnesses.
2. File a complaint at the nearest police office.' metadata={'source': '/Users/kristalshrestha/Documents/Code/LLM_Scratch/Langchain/6_RAG/1_document_loader/6_cleaned_socialInclusion.json', 'seq_num': 1}"""

print(docs[0].page_content)
"""Output:
Question: My teacher insults my caste in front of other students and creates hatred in the classroom. Can I take legal action?

Answer: **Yes. Acts that harm harmony between communities are prohibited.**

**Law says:**
- Muluki Criminal (Code) Act, 2074 (2017) Section 65: Prohibition of acts on the ground of caste or community that are prejudicial to harmonious relationships.

**Punishment:**
- Imprisonment for a term not exceeding one year and a fine not exceeding ten thousand rupees.

**What to do:**
1. Write down dates, words used, and names of student witnesses.
2. File a complaint at the nearest police office.
"""
print(type(docs[0].page_content))  # <class 'str'>
print(type(docs[0].metadata))  # <class 'dict'>
