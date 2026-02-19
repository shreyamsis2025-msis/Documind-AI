from langchain_community.llms import Ollama
from vectordb.faiss_store import load_db
from config import OLLAMA_MODEL


class RAGChat:
    def __init__(self):
        self.db = load_db()
        self.llm = Ollama(model=OLLAMA_MODEL)

    def ask(self, query: str):

        if self.db is None:
            return {
                "answer": self.llm.invoke(query),
                "sources": []
            }

        retriever = self.db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        # ---------- Build context ----------
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Answer the question using the context below.
If answer not in context, say you don't know.

Context:
{context}

Question:
{query}
"""

        answer = self.llm.invoke(prompt)

        # ---------- Extract source info ----------
        sources = []
        for d in docs:
            meta = d.metadata
            file = meta.get("source", "Unknown file")
            page = meta.get("page", "")

            if page != "":
                sources.append(f"{file} (Page {page})")
            else:
                sources.append(file)

        return {
            "answer": answer,
            "sources": list(set(sources)),  # remove duplicates
            "snippets": [d.page_content[:200] for d in docs]
        }
