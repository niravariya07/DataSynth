import numpy as np
from typing import List, Tuple, Union
from .embedder import get_embedding
from .load_index import load_faiss_index

def retrieve_chunks(user_input: Union[str, List[str]], top_k: int = 3) -> List[Tuple[str, float]]:
    if isinstance(user_input, list):
        query = " ".join(user_input)
    elif isinstance(user_input, str):
        query = user_input
    else:
        raise TypeError("user_input must be a string or list of strings.")

    
    index, id_to_text = load_faiss_index()

    if index is None or not id_to_text:
        raise ValueError("FAISS index or metadata not found. Please build the index first.")
    
    query_embedding = get_embedding(query).astype(np.float32)

    distances, indices = index.search(np.array([query_embedding]), top_k)

    results = []
    for indx, score in zip(indices[0], distances[0]):
        if indx != -1 and indx in id_to_text:
            results.append((id_to_text[indx], float(score)))
    
    return results