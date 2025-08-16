import pickle
from .build_index import index_file, metadata_file
from typing import List, Tuple
import faiss

def load_faiss_index() -> Tuple[faiss.Index,List[str]]:
    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("FaISS index or metadat file not found. Please the index first.")
    
    index = faiss.read_index(str(index_file))
    with open(metadata_file, "rb") as f:
        metadata: List[str] = pickle.load(f)
    
    return index, metadata