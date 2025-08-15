from .build_index import index_file, metadata_file
from typing import List
import faiss

def load_faiss_index() -> tuple[faiss.Index,List[str]]:
    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("FaISS index or metadat file not found. Please the index first.")
    
    