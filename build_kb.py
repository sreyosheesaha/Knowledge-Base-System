from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

files = ["data/ml_notes.pdf", "data/python_book.pdf", "data/college_notes.pdf"]
docs = []

# Load PDFs
for file in files:
    loader = PyPDFLoader(file)
    docs.extend(loader.load())

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Create embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Store in vector database
db = Chroma.from_documents(chunks, embeddings, persist_directory="kb_db")
db.persist()

print("✅ Knowledge Base Created Successfully!")