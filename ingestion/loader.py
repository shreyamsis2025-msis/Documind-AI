import os
import pandas as pd
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader


def load_file(path):
    """
    Loads a document and attaches metadata for source highlighting.
    Supports: PDF, DOCX, CSV, XLSX
    """

    ext = path.split(".")[-1].lower()
    filename = os.path.basename(path)

    docs = []

    # ---------- PDF ----------
    if ext == "pdf":
        loader = PyPDFLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = filename
            # page already exists from PyPDFLoader, but keep safe
            if "page" not in d.metadata:
                d.metadata["page"] = 0

        return docs

    # ---------- DOCX ----------
    if ext == "docx":
        loader = Docx2txtLoader(path)
        docs = loader.load()

        for d in docs:
            d.metadata["source"] = filename
            d.metadata["page"] = ""   # no page for docx

        return docs

    # ---------- CSV ----------
    if ext == "csv":
        df = pd.read_csv(path)
        text = df.to_string()

        doc = Document(
            page_content=text,
            metadata={
                "source": filename,
                "page": ""
            }
        )
        return [doc]

    # ---------- Excel ----------
    if ext == "xlsx":
        df = pd.read_excel(path)
        text = df.to_string()

        doc = Document(
            page_content=text,
            metadata={
                "source": filename,
                "page": ""
            }
        )
        return [doc]

    # ---------- Unsupported ----------
    return []
