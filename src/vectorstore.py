import os
import faiss # faiss database for vector storage and retrieval
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = 'all-MiniLM-L6-v2', index_file: str = 'faiss_index.bin', chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the FAISS vector store with the specified embedding model and index file."""
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
       print(f"[INFO] Building FAISS Vectore store {len(documents)} raw documents...")

       emb_pipeline = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
       chunks = emb_pipeline.chunk_documents(documents)
       embedding = emb_pipeline.embd_chunks(chunks)
       metadatas = [{"text": chunk.page_content} for chunk in chunks]
       self.add_embeddings(np.array(embedding).astype('float32'), metadatas)
       self.save()
       print(f"[INFO] FAISS Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any]=None):
        """Add embeddings and their corresponding metadata to the FAISS index."""
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas) 
        print(f"[INFO] Added {embeddings.shape[0]} embeddings to the FAISS index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] FAISS index and metadata saved to {self.persist_dir} successfully.")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss_index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        if os.path.exists(faiss_path) and os.path.exists(meta_path):
            self.index = faiss.read_index(faiss_path)
            with open(meta_path, "rb") as f:
                self.metadata = pickle.load(f)
            print(f"[INFO] FAISS index and metadata loaded from {self.persist_dir} successfully.")
        else:
            print(f"[WARNING] No existing FAISS index found in {self.persist_dir}. Starting with an empty index.")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """Search the FAISS index for the most similar embeddings to the query."""
        D, I = self.index.search(np.array(query_embedding).astype('float32'), top_k)
        results = []
        for Idx, Dist in zip(I[0], D[0]):
            meta = self.metadata[Idx] if Idx < len(self.metadata) else None
            results.append({"index": Idx, "metadata": meta, "distance": Dist})
        return results
    
    def query(self, query_text: str, top_k: int = 5):
        """Generate an embedding for the query and search the FAISS index."""
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)
        

    