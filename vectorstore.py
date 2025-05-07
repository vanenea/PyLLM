# vectorstore.py
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

class VectorStore:
    def __init__(self, api_key: str):
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.store = None

    def build(self, docs: list):
        texts = [d['text'] for d in docs]
        metadatas = [{'source': d['source']} for d in docs]
        self.store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)

    def query(self, q: str, k: int = 5):
        return self.store.similarity_search_with_score(q, k=k)