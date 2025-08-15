from .embedder import get_embedding_array
import faiss
import numpy as np
from typing import List, Tuple, Dict, Union

def build_index(
        user_columns: Union[List[Dict[str, str]], None] = None ) -> Tuple[faiss.IndexFlatL2, Dict[int, Dict[str, str]]]:
    
    if not user_columns:
        raise ValueError("user_columns required to build index.")
    