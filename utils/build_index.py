import faiss
import numpy as np
from typing import List, Tuple, Dict

from embedder import get_embeddings

def build_index(rag_entries: List[Dict[str, str]]) -> Tuple[faiss.IndexFlatL2, Dict[int, Dict[int,Dict[str, str]]]]:
    if not rag_entries:
        raise ValueError("RAG entries list is empty.")
    
    texts = [f"{col['name']}: {col['description']}" for col in rag_entries]

    embeddings = get_embeddings(texts)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    mapping = {i: entry for i, entry in enumerate(rag_entries)}

    return index, mapping