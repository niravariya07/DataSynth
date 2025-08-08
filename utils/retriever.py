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


if __name__ == "__main__":
    from build_index import build_index

    # Test with sample RAG content
    rag_data = [
        {"name": "Age", "description": "Age of the person in years"},
        {"name": "Email", "description": "Email address of the person"},
        {"name": "Name", "description": "Full name of the person"},
    ]

    index, mapping = build_index(rag_data)
    query = "customer's email"
    matches = retrieve_context(query, index, mapping, top_k=2)
    print("Query:", query)
    print("Matches:", matches)
