# --- PDF Reader Application: Manual Component Connection ---
# (As shown in the video around 13:00)

# 1. Load the document
from langchain.document_loaders import TextLoader  # Example with a text file

loader = TextLoader("your_document.txt")  # Replace with your document path
documents = loader.load()

# 2. Split the document into smaller chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 3. Create embeddings and store in a vector database
from langchain.embeddings import OpenAIEmbeddings  # Example with OpenAI
from langchain.vectorstores import FAISS  # Example with FAISS

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# 4. Create a retriever from the vector store
retriever = vectorstore.as_retriever()

# 5. User query comes in
query = "What is the capital of India?"

# 6. Retrieve relevant documents using semantic search
retrieved_docs = retriever.get_relevant_documents(query)
retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])

# 7. Create a new prompt with the query and retrieved context
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

llm = OpenAI()
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="Based on the following context, answer the question.\nContext: {context}\nQuestion: {question}\nAnswer:",
)
formatted_prompt = prompt_template.format(context=retrieved_text, question=query)

# 8. Send the prompt to the LLM and get the answer
answer = llm.predict(formatted_prompt)

# 9. Display the answer
print(answer)
