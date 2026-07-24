import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Loading Files

from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import PyPDFLoader

csv_loader = CSVLoader(file_path=os.path.join(BASE_DIR, "sample.csv"))
csv_docs = csv_loader.load()

html_loader = UnstructuredHTMLLoader(file_path=os.path.join(BASE_DIR, "sample.html"))
html_docs = html_loader.load()

pdf_loader = PyPDFLoader(os.path.join(BASE_DIR, "sample.pdf"))
pdf_docs = pdf_loader.load()

all_docs = csv_docs + html_docs + pdf_docs

# Breaking into Chunks

from langchain_text_splitters.character import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10
)

chunks = splitter.split_documents(all_docs)
for c in chunks:
    print(c.page_content)

# Embeddings
# To create chunk | number pairs
# This helps the AI check similarities in text and give related output

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Storing embeddings | chunk pairs in Vector Database

from langchain_community.vectorstores import Chroma

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=os.path.join(BASE_DIR, "my_database")  # any location to store the database
)