from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


loader = DirectoryLoader(
    path=r"./3_books_directory_allpdfs", glob="*.pdf", loader_cls=PyPDFLoader
)
# load()-> directly loads everything at once on RAM,very memory consuming

# docs = loader.load()
# print(len(docs))  # 23+326 = 349 pages/document objects

# for document in docs:
#     print(document.metadata)


# lazy_load() -> one at a time ,in generator,no Memory consuming

docs = loader.lazy_load()


for document in docs:
    print(document.metadata)
