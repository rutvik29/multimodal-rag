"""Hybrid multimodal retriever."""
from typing import List
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os


class MultimodalRetriever:
    def __init__(self, collection_prefix: str = "docs", persist_dir: str = "./chroma_db", k: int = 5):
        self.k = k
        client = chromadb.PersistentClient(path=persist_dir)
        ef = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small")
        self.collections = {
            "text": client.get_or_create_collection(f"{collection_prefix}_text", embedding_function=ef),
            "image": client.get_or_create_collection(f"{collection_prefix}_images", embedding_function=ef),
            "table": client.get_or_create_collection(f"{collection_prefix}_tables", embedding_function=ef),
        }

    def retrieve(self, query: str, k: int = None) -> List[dict]:
        k = k or self.k
        all_results = []
        for col_name, col in self.collections.items():
            try:
                results = col.query(query_texts=[query], n_results=k)
                for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
                    all_results.append({"content": doc, "metadata": meta, "type": col_name, "score": 1 - dist})
            except Exception:
                pass
        return sorted(all_results, key=lambda x: x["score"], reverse=True)[:k * 2]
