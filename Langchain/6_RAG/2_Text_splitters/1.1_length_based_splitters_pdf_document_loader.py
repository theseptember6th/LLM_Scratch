from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


loader = PyPDFLoader(file_path=r"./1.1_dl_curriculum.pdf")

docs = loader.load()

splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10, separator="")
# for RAG based application,10% of chunksize for  chunk overlap is good.
result = splitter.split_documents(docs)

# print(result)
"""[------------------Document(metadata={'producer': 'Skia/PDF m131 Google Docs Renderer', 'creator': 'PyPDF', 'creationdate': '', 'title': 'Deep Learning Curriculum', 'source': './1.1_dl_curriculum.pdf', 'total_pages': 23, 'page': 21, 'page_label': '22'}, page_content='Creatingconversational agentsusingGPTmodels.● StoryGeneration○ Generatingcoherent narratives.\nF.Nam'), Document(metadata={'producer': 'Skia/PDF m131 Google Docs Renderer', 'creator': 'PyPDF', 'creationdate': '', 'title': 'Deep Learning Curriculum', 'source': './1.1_dl_curriculum.pdf', 'total_pages': 23, 'page': 21, 'page_label': '22'}, page_content='edEntityRecognition\n● SequenceLabeling○ Identifyingentitieslikenames, locations, dates.● Fine-Tuning'), Document(metadata={'producer': 'Skia/PDF m131 Google Docs Renderer', 'creator': 'PyPDF', 'creationdate': '', 'title': 'Deep Learning Curriculum', 'source': './1.1_dl_curriculum.pdf', 'total_pages': 23, 'page': 21, 'page_label': '22'}, page_content='○ Adaptingpre-trainedmodelsfor NERtasks.')]"""

# print(result[0])  # 1st chunk
"""Output:
page_content='CampusXDeepLearningCurriculum
A.ArtificialNeuralNetworkandhowtoimprovethem
1.BiologicalInspiration
●' metadata={'producer': 'Skia/PDF m131 Google Docs Renderer', 'creator': 'PyPDF', 'creationdate': '', 'title': 'Deep Learning Curriculum', 'source': './1.1_dl_curriculum.pdf', 'total_pages': 23, 'page': 0, 'page_label': '1'}"""

print(result[0].page_content)
"""Output:
CampusXDeepLearningCurriculum
A.ArtificialNeuralNetworkandhowtoimprovethem
1.BiologicalInspiration
"""
print(result[0].metadata)
"""Output:
{'producer': 'Skia/PDF m131 Google Docs Renderer', 'creator': 'PyPDF', 'creationdate': '', 'title': 'Deep Learning Curriculum', 'source': './1.1_dl_curriculum.pdf', 'total_pages': 23, 'page': 0, 'page_label': '1'}
"""
