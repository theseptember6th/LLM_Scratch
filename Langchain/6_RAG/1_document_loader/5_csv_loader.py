from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path=r"./5_Social_Network_Ads.csv")
docs = loader.load()
# print(len(docs))  # 400 ->400 rows
"""Each row in csv is a document object"""

# print(docs[0])
"""Output:
page_content='User ID: 15624510
Gender: Male
Age: 19
EstimatedSalary: 19000
Purchased: 0' metadata={'source': './5_Social_Network_Ads.csv', 'row': 0}"""
# print(docs[0].page_content)
"""User ID: 15624510
Gender: Male
Age: 19
EstimatedSalary: 19000
Purchased: 0"""
print(type(docs[0].page_content))  # <class 'str'>
