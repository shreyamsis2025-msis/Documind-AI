import streamlit as st
import os
import urllib.parse

from ingestion.loader import load_file
from ingestion.splitter import split_docs
from vectordb.faiss_store import save_db
from rag.pipeline import RAGChat
from config import DATA_PATH

# -------- SETTINGS --------
STATIC_PATH = "static_docs"

st.set_page_config(layout="wide")
os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(STATIC_PATH, exist_ok=True)

# -------- Sidebar --------
st.sidebar.title("📄 Documind AI")
st.sidebar.write("Chat with your documents locally using Ollama")

uploaded_files = st.sidebar.file_uploader(
    "Upload documents",
    accept_multiple_files=True
)

if uploaded_files:
    all_docs = []

    with st.spinner("Processing documents..."):
        for file in uploaded_files:

            path = os.path.join(DATA_PATH, file.name)
            static_path = os.path.join(STATIC_PATH, file.name)

            # Skip if already processed
            if os.path.exists(path):
                st.sidebar.warning(f"{file.name} already uploaded")
                continue

            # Save for processing
            with open(path, "wb") as f:
                f.write(file.getbuffer())

            # Save for viewing
            with open(static_path, "wb") as f:
                f.write(file.getbuffer())

            docs = load_file(path)
            all_docs.extend(docs)

        if all_docs:
            chunks = split_docs(all_docs)
            save_db(chunks)
            st.sidebar.success("Documents processed!")

# -------- Show uploaded files --------
if os.path.exists(DATA_PATH):
    st.sidebar.subheader("Uploaded Files")
    for f in os.listdir(DATA_PATH):
        st.sidebar.write("•", f)

# -------- Cache chatbot --------
@st.cache_resource
def load_chatbot():
    return RAGChat()

chatbot = load_chatbot()

# -------- Chat UI --------
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

    with st.spinner("Thinking..."):
        result = chatbot.ask(query)

    with st.chat_message("assistant"):
        st.write(result["answer"])

        # -------- Sources --------
        if result["sources"]:
            st.markdown("### 📚 Sources")

            for s in result["sources"]:
                # Split filename & page
                if "(Page" in s:
                    file, page = s.split("(Page")
                    file = file.strip()
                    page = page.replace(")", "").strip()
                else:
                    file = s
                    page = None

                file_url = f"/static_docs/{urllib.parse.quote(file)}"

                if page:
                    st.markdown(
                        f"• [{file} (Page {page})]({file_url}#page={page})"
                    )
                else:
                    st.markdown(f"• [{file}]({file_url})")

            # -------- Show context --------
            with st.expander("Show context used"):
                for snip in result["snippets"]:
                    st.write("-", snip)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"]}
    )
