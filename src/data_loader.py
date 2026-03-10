from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader 
#from langchain_community.document_loaders.csv import UnstructuredCSVLoader
#from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """Load all documents from the specified directory, supporting multiple formats."""
    
    # Use Project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data Path: {data_path}")
    documents = []
    
    # pdf Files
    pdf_files = list(data_path.glob("**/*.pdf"))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(pdf) for pdf in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            docs = loader.load()
            documents.extend(docs)
            print(f"[DEBUG] Loaded {len(docs)} pages from {pdf_file}")
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")
    return documents

    # Text Files
    
    # CSV Files

    # Sql Files

    # Docx Files

    # Excel Files

    # JSON Files