from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the embedding manager with a specified sentence transformer model."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded Embedding Model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """Chunk the documents into smaller pieces using RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, 
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Chunked {len(documents)} documents into {len(chunks)} chunks.")
        return chunks
    
    def embd_chunks(self, chunks: List[Any]) -> np.ndarray:
        """Generate embeddings for the given chunks using the sentence transformer model."""
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO]Embedding shape: {embeddings.shape}")
        return embeddings