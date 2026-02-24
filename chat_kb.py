from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load DB
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = Chroma(persist_directory="kb_db", embedding_function=embeddings)

print("Knowledge Base AI Ready. Type exit to stop.")

while True:
    query = input("\nAsk: ")
    if query.lower() == "exit":
        break
    
    docs = db.similarity_search(query)
    print("\nAnswer:\n", docs[0].page_content)