from langchain_community.llms import Ollama
from vectordb.faiss_store import load_db
from config import OLLAMA_MODEL


class RAGChat:
    def __init__(self):
        self.db = load_db()
        self.llm = Ollama(model=OLLAMA_MODEL)

    def ask(self, query: str):

        # ✅ If no documents → normal LLM chat
        if self.db is None:
            return self.llm.invoke(query)

        # ✅ If docs exist → RAG
        retriever = self.db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
Answer using the context if relevant. If not, answer normally.

Context:
{context}

Question:
{query}
"""

        return self.llm.invoke(prompt)
