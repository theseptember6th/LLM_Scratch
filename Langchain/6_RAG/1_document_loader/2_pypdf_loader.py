# load the text-based pdf model

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path=r"./2_dl_curriculum.pdf")

docs = loader.load()

print(docs)

"""Output:
[Document(metadata={'producer': 'Skia/PDF m131 Google Docs Renderer', 'creator': 'PyPDF', 'creationdate': '', 'title': 'Deep Learning Curriculum', 'source': './2_dl_curriculum.pdf', 'total_pages': 23, 'page': 0, 'page_label': '1'}, page_content='CampusXDeepLearningCurriculum\nA.ArtificialNeuralNetworkandhowtoimprovethem\n1.BiologicalInspiration\n● Understandingtheneuronstructure● Synapsesandsignal transmission● Howbiological conceptstranslatetoartificial neurons\n2.HistoryofNeuralNetworks\n● Earlymodels(Perceptron)● BackpropagationandMLPs● The"AI Winter" andresurgenceof neural networks● Emergenceof deeplearning\n3.PerceptronandMultilayerPerceptrons(MLP)\n● Single-layer perceptronlimitations● XORproblemandtheneedfor hiddenlayers● MLParchitecture\n4. LayersandTheirFunctions\n● InputLayer○ Acceptinginput data● HiddenLayers○ Featureextraction● OutputLayer○ Producingfinal predictions\n5.ActivationFunctions'), Document(metadata={'producer': 'Skia/PDF m131 Google Docs Renderer', 'creator': 'PyPDF', 'creationdate': '', 'title': 'Deep Learning Curriculum', 'source': './2_dl_curriculum.pdf', 'total_pages': 23, 'page': 1, 'page_label': '2'}, page_content='● SigmoidFunction○ Characteristicsandlimitations● HyperbolicTangent(tanh)○ Comparisonwithsigmoid● ReLU(RectifiedLinearUnit)○ Advantagesinmitigatingvanishinggradients● LeakyReLUandParametricReLU○ AddressingthedyingReLUproblem● SoftmaxFunction○ Multi-classclassificationoutputs\n6.ForwardPropagation\n● Mathematical computationsat eachneuron● Passinginputsthroughthenetworktogenerateoutputs\n7.LossFunctions\n● MeanSquaredError(MSE)○ Usedinregressiontasks● Cross-EntropyLoss○ Usedinclassificationtasks● HingeLoss○ UsedwithSVMs● Selectingappropriatelossfunctionsbasedontasks\n8.Backpropagation\n● Derivationusingthechainrule● Computinggradientsfor eachlayer● Updatingweightsandbiases● Understandingcomputational graphs\n9.GradientDescentVariants\n● BatchGradientDescent○ Prosandcons'),"""

"""Each page is a document object,so in this pdf there is 25 pages so 25 documents """
print(len(docs))  # 23


print(f"\n\n\n\nPage_content:\n\n{docs[0].page_content}")
"""Output:
Page_content:

CampusXDeepLearningCurriculum
A.ArtificialNeuralNetworkandhowtoimprovethem
1.BiologicalInspiration
● Understandingtheneuronstructure● Synapsesandsignal transmission● Howbiological conceptstranslatetoartificial neurons
2.HistoryofNeuralNetworks
● Earlymodels(Perceptron)● BackpropagationandMLPs● The"AI Winter" andresurgenceof neural networks● Emergenceof deeplearning
3.PerceptronandMultilayerPerceptrons(MLP)
● Single-layer perceptronlimitations● XORproblemandtheneedfor hiddenlayers● MLParchitecture
4. LayersandTheirFunctions
● InputLayer○ Acceptinginput data● HiddenLayers○ Featureextraction● OutputLayer○ Producingfinal predictions
5.ActivationFunctions"""

print(f"\n\n\n\nMetaData:\n\n\n{docs[0].metadata}")

"""Output:
MetaData:


{'producer': 'Skia/PDF m131 Google Docs Renderer', 'creator': 'PyPDF', 'creationdate': '', 'title': 'Deep Learning Curriculum', 'source': './2_dl_curriculum.pdf', 'total_pages': 23, 'page': 0, 'page_label': '1'}"""
