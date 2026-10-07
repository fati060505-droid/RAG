"""Run once (and again whenever data/document.txt changes) to build the FAISS index."""
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load the document
docs = TextLoader("data/document.txt", encoding="utf-8").load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# 3. Embed with a Hugging Face model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store in FAISS and save to disk
db = FAISS.from_documents(chunks, embeddings)
db.save_local("vectorstore")
print("Saved FAISS index to vectorstore/")
