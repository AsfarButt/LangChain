

from langchain_community.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://clerk.com/docs")

docs = loader.load()

# print(docs)

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs) 

print(documents)

vectorstore = FAISS.from_documents(documents, embeddings)

print(vectorstore)

