import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader


def load_file(path):
    ext = path.split(".")[-1].lower()

    if ext == "pdf":
        return PyPDFLoader(path).load()

    if ext == "docx":
        return Docx2txtLoader(path).load()

    if ext == "csv":
        df = pd.read_csv(path)
        return [Document(page_content=df.to_string())]

    if ext == "xlsx":
        df = pd.read_excel(path)
        return [Document(page_content=df.to_string())]

    return []
