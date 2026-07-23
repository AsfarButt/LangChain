# Loading Files

from langchain_community.document_loaders.csv_loader import CSVLoader

loader = CSVLoader(file_path="/sample.csv")
docs = loader.load()

from langchain_community.document_loaders import UnstructuredHTMLLoader 

loader = UnstructuredHTMLLoader(file_path="/sample/html")
docs = loader.load()

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("/sample.pdf")
docs = loader.load()


# Breaking into Chunks

from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = CharacterTextSplitter(       # RecursiveCharacterTextSplitter
    seperator = '\n\n',                     # ['\n',' ','']
    chunk_size = 100,
    chunk_overlap = 10
)

chunks = splitter.split_text(docs)

# Embeddings 
# To create chunk | number pairs 
# This helps the AI check similarities in text and give related output

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Storing embeddings | chunk pairs in Vector Database

from langchain_community.vectorstores import Chroma

vector_store = Chroma.from_documents(
    documents = docs,
    embedding = embeddings,
    persist_directory = "my_database" # any location to store the database
)

