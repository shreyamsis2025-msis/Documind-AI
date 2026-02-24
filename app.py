import streamlit as st
import os
import urllib.parse

from ingestion.loader import load_file
from ingestion.splitter import split_docs
from vectordb.faiss_store import save_db
from rag.pipeline import RAGChat
from config import DATA_PATH
from streamlit_mic_recorder import mic_recorder
from voice.whisper_local import speech_to_text
from streamlit_pdf_viewer import pdf_viewer

if "open_pdf" not in st.session_state:
    st.session_state.open_pdf = None

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

col1, col2 = st.columns([6, 1])

with col1:
    text_query = st.chat_input("Ask something about your documents...")

with col2:
    audio = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key="voice")

query = None

# Voice input
if audio:
    try:
        query = speech_to_text(audio["bytes"])
        # st.success(f"You said: {query}")
    except Exception as e:
        st.error(f"Voice error: {e}")

# Text input
if text_query:
    query = text_query

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Thinking..."):
        result = chatbot.ask(query)

    with st.chat_message("assistant"):
        st.write(result["answer"])

    if st.session_state.open_pdf:
        file, page = st.session_state.open_pdf
        file_path = os.path.join(DATA_PATH, file)

        if os.path.exists(file_path):
            st.markdown("---")
            st.markdown(f"### 📄 Viewing: {file} (Page {page})")
            pdf_viewer(file_path, width=800, height=1000, page=page)
        else:
            st.error("File not found.")

    # -------- Sources --------
    if result["sources"]:
        st.markdown("### 📚 Sources")

        for s in result["sources"]:
            if "(Page" in s:
                file, page = s.split("(Page")
                file = file.strip()
                page = int(page.replace(")", "").strip())
            else:
                file = s
                page = 0

            if st.button(f"Open {file} (Page {page})"):
                st.session_state.open_pdf = (file, page)

        # -------- Show context --------
        with st.expander("Show context used"):
            for snip in result["snippets"]:
                st.write("-", snip)

    st.session_state.messages.append(
        {"role": "assistant", "content": result["answer"]}
    )