import streamlit as st
import os

from ingestion.loader import load_file
from ingestion.splitter import split_docs
from vectordb.faiss_store import save_db
from rag.pipeline import RAGChat
from config import DATA_PATH

st.set_page_config(layout="wide")

# ---------------- Sidebar ----------------
st.sidebar.title("📄 Documind AI")
st.sidebar.write("Chat with your documents locally using Ollama + phi3")

uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    accept_multiple_files=True
)

if uploaded_files:
    all_docs = []

    os.makedirs(DATA_PATH, exist_ok=True)

    for file in uploaded_files:
        path = os.path.join(DATA_PATH, file.name)

        with open(path, "wb") as f:
            f.write(file.getbuffer())

        docs = load_file(path)
        all_docs.extend(docs)

    chunks = split_docs(all_docs)
    save_db(chunks)

    st.sidebar.success("Documents processed!")

# show files
if os.path.exists(DATA_PATH):
    st.sidebar.subheader("Uploaded Files")
    for f in os.listdir(DATA_PATH):
        st.sidebar.write("•", f)

# ---------------- Chat UI ----------------
st.title("💬 Document Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask something about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    chatbot = RAGChat()
    answer = chatbot.ask(query)


    with st.chat_message("assistant"):
        st.write(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
