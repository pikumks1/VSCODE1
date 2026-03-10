##code to run streamlit from github repo
import sys
import subprocess

# Force an install check for debugging
try:
    import langchain_groq
except ImportError:
    print("langchain_groq not found! Attempting runtime install...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "langchain-groq"])
    import langchain_groq

import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()  # Load environment variables from .env file

class RAGSearch:
    def __init__(self, Persist_dir: str = "faiss_store", embedding_model: str = 'all-MiniLM-L6-v2',llm_model: str = 'groq/compound', chunk_size: int = 1000, chunk_overlap: int = 200):
        """Initialize the RAG search with the specified FAISS vector store directory."""
        
        self.vectorestore = FaissVectorStore(persist_dir=Persist_dir, embedding_model=embedding_model)
        
        #Load or Build vectorestore
        faiss_path = os.path.join(Persist_dir, "faiss_index")
        meta_path = os.path.join(Persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorestore.build_from_documents(docs)
        else:
            self.vectorestore.load()
        
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized with model: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        """Search the FAISS vector store for relevant chunks and summarize the results using Groq LLM."""
        results = self.vectorestore.query(query, top_k=top_k)
        if not results:
            return "No relevant information found."
        
        # Combine retrieved chunks into a single context for summarization
        texts = [r['metadata'].get('text', '') for r in results if r['metadata'].get('text')]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant information found in the retrieved chunks."
        
        # Generate a summary using the Groq LLM
        prompt = f"Summarize the following information:\n\n{context}"
        response = self.llm.invoke([prompt])
        return response.content
    