from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
import os

DATA_PATH = "data"
VECTOR_DB_DIR = "vector_db"

def load_documents():
    print("📄 Loading documents...")
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, filename))
            documents.extend(loader.load())
    print(f"📚 Loaded {len(documents)} documents")
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)

def create_vector_store(docs):
    print("🧠 Creating vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create vectorstore
    vectorstore = FAISS.from_documents(docs, embeddings)

    # Save it to disk
    os.makedirs(VECTOR_DB_DIR, exist_ok=True)
    vectorstore.save_local(VECTOR_DB_DIR)
    print("✅ Vector store saved!")

if __name__ == "__main__":
    raw_docs = load_documents()
    docs = split_documents(raw_docs)
    create_vector_store(docs)
