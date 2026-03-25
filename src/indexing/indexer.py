"""Multimodal ChromaDB indexer."""
from typing import List, Dict, Any
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
import os


class MultimodalIndexer:
    def __init__(self, collection_prefix: str = "docs", persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        ef = OpenAIEmbeddingFunction(api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small")
        self.text_col = self.client.get_or_create_collection(f"{collection_prefix}_text", embedding_function=ef)
        self.image_col = self.client.get_or_create_collection(f"{collection_prefix}_images", embedding_function=ef)
        self.table_col = self.client.get_or_create_collection(f"{collection_prefix}_tables", embedding_function=ef)
        self.total_chunks = 0

    def index_chunks(self, chunks: List[Dict[str, Any]], source: str):
        for i, chunk in enumerate(chunks):
            col = {"text": self.text_col, "image": self.image_col, "table": self.table_col}.get(chunk["type"], self.text_col)
            doc_id = f"{source}_{chunk['type']}_{chunk.get('page',0)}_{i}"
            col.add(documents=[chunk["content"]], ids=[doc_id], metadatas=[chunk.get("metadata", {})])
            self.total_chunks += 1

    def index_image(self, description: str, source: str):
        self.image_col.add(documents=[description], ids=[source], metadatas=[{"source": source, "type": "image"}])
        self.total_chunks += 1
