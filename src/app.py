
from src.data_loader import load_all_documents
#from src.embedding import EmbeddingPipeline
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch



# Example usage
if __name__ == "__main__":
    #doc = load_all_documents("data")

    store = FaissVectorStore("faiss_store")
    #store.build_from_documents(doc) ## Build the FAISS vector store from the loaded documents and save it to disk
        # store not needed to build every time, only when new documents are added. Once built, we can load the existing store and perform retrieval operations.
    store.load() ## Load the existing FAISS vector store from disk for retrieval operations
    #print(store.query("What monkeys are doing?", top_k=3)) ## Example query to retrieve relevant chunks from the FAISS vector store based on a user query. Adjust the query and top_k as needed.
    
    
    rag_search = RAGSearch()
    query = "Share clean and short story of A Visit to the Market"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print(f"Summary:\n{summary}")
    
    
    
    
    
    #chunks=EmbeddingPipeline().chunk_documents(doc)
    #chunksvector=EmbeddingPipeline().embd_chunks(chunks)

    #print(doc)
    #print(chunks)
    #print(chunksvector)