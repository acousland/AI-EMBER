import pickle5 as pk
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import TextLoader, DirectoryLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os


def generateFireRisk():
    fire_risk_loader = CSVLoader(file_path="assets/data/fire_risk.csv")
    fire_risk_documents = fire_risk_loader.load()
    fire_risk_db = FAISS.from_documents(fire_risk_documents, OpenAIEmbeddings())
    f = open("assets/embeddings/fire_risk.pkl","wb")
    pk.dump(fire_risk_db,f)
    f.close()

def generatePreperation():
    preperation_loader = DirectoryLoader('assets/data/preperation_advice/', glob="*", loader_cls=TextLoader)
    preperation_documents = preperation_loader.load()
    preperation_db = FAISS.from_documents(preperation_documents, OpenAIEmbeddings())
    f = open("assets/embeddings/preperation.pkl","wb")
    pk.dump(preperation_db,f)
    f.close()

def loadFireEmbeddings():
    if os.path.isfile('assets/embeddings/fire_risk.pkl')==False:
        generateFireRisk()
    f = open('assets/embeddings/fire_risk.pkl', 'rb')
    fireRiskDB = pk.load(f)
    f.close()
    return(fireRiskDB)

def loadPreperationsEmbeddings():
    if os.path.isfile('assets/embeddings/preperation.pkl')==False:
        generatePreperation()
    f = open('assets/embeddings/preperation.pkl', 'rb')
    prepDB = pk.load(f)
    f.close()
    return(prepDB)