import os
from langchain_community.vectorstores import FAISS
from ingestion.embedder import get_embeddings
from config import FAISS_PATH


def save_db(chunks):
    embeddings = get_embeddings()
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_PATH)


def load_db():
    if not os.path.exists(FAISS_PATH):
        return None

    embeddings = get_embeddings()
    return FAISS.load_local(
        FAISS_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
