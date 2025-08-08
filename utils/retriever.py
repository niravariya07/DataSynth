import faiss
import numpy as np
from typing import Dict, List, Tuple
from embedder import get_embedding

def retrieve_context(
        query: str,
        index: faiss.IndexFlatL2,
        mapping: Dict[int, Dict[str, str]],
        top_k: int = 1
) -> List[Dict[str, str]]:
    query_vector = get_embedding(query).astype(np.float32).reshape(1, -1)

    distances, indices = index.search(query_vector, top_k)

    results = []
    for idx in indices[0]:
        if idx in mapping:
            results.append(mapping[idx])

    return results