from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader


loader = DirectoryLoader(
    path=r"./3_books_directory_allpdfs", glob="*.pdf", loader_cls=PyPDFLoader
)

docs = loader.load()
print(len(docs))  # 23+326 = 349 pages/document objects


# print(f"\n\n\n\n{docs[0].page_content}\n\n\n\n")
# """Output:
# CampusXDeepLearningCurriculum
# A.ArtificialNeuralNetworkandhowtoimprovethem
# 1.BiologicalInspiration
# ● Understandingtheneuronstructure● Synapsesandsignal transmission● Howbiological conceptstranslatetoartificial neurons
# 2.HistoryofNeuralNetworks
# ● Earlymodels(Perceptron)● BackpropagationandMLPs● The"AI Winter" andresurgenceof neural networks● Emergenceof deeplearning
# 3.PerceptronandMultilayerPerceptrons(MLP)
# ● Single-layer perceptronlimitations● XORproblemandtheneedfor hiddenlayers● MLParchitecture
# 4. LayersandTheirFunctions
# ● InputLayer○ Acceptinginput data● HiddenLayers○ Featureextraction● OutputLayer○ Producingfinal predictions
# 5.ActivationFunctions"""
# print(f"\n\n\n\n{docs[0].metadata}\n\n\n\n")
# """Output:
# {'producer': 'Skia/PDF m131 Google Docs Renderer', 'creator': 'PyPDF', 'creationdate': '', 'title': 'Deep Learning Curriculum', 'source': '3_books_directory_allpdfs/2_dl_curriculum.pdf', 'total_pages': 23, 'page': 0, 'page_label': '1'}
# """
# print(type(docs[0]))  # <class 'langchain_core.documents.base.Document'>

print(f"\n\n\n\n{docs[25].page_content}\n\n\n\n")
"""Output:
Building Machine Learning Systems with Python 
Second Edition
Copyright © 2015 Packt Publishing
All rights reserved. No part of this book may be reproduced, stored in a retrieval 
system, or transmitted in any form or by any means, without the prior written 
permission of the publisher, except in the case of brief quotations embedded in 
critical articles or reviews.
Every effort has been made in the preparation of this book to ensure the accuracy 
of the information presented. However, the information contained in this book is 
sold without warranty, either express or implied. Neither the authors, nor Packt 
Publishing, and its dealers and distributors will be held liable for any damages 
caused or alleged to be caused directly or indirectly by this book.
Packt Publishing has endeavored to provide trademark information about all of the 
companies and products mentioned in this book by the appropriate use of capitals. 
However, Packt Publishing cannot guarantee the accuracy of this information.
First published: July 2013
Second edition: March 2015
Production reference: 1230315
Published by Packt Publishing Ltd.
Livery Place
35 Livery Street
Birmingham B3 2PB, UK.
ISBN 978-1-78439-277-2
www.packtpub.com

"""

print(f"\n\n\n\n{docs[25].metadata}\n\n\n\n")
"""Output:{'producer': 'Adobe PDF Library 10.0.1', 'creator': 'Adobe InDesign CS6 (Windows)', 'creationdate': '2015-03-24T13:14:02+05:30', 'moddate': '2015-03-25T17:33:08+05:30', 'trapped': '/False', 'source': '3_books_directory_allpdfs/Building Machine Learning Systems with Python - Second Edition.pdf', 'total_pages': 326, 'page': 2, 'page_label': 'FM2'}"""
print(type(docs[25]))
