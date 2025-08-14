from embedder import get_embeddings
import faiss
import numpy as np
from typing import List, Tuple, Dict, Union

# def build_index(rag_entries: List[Dict[str, str]]) -> Tuple[faiss.IndexFlatL2, Dict[int, Dict[int,Dict[str, str]]]]:
#     if not rag_entries:
#         raise ValueError("RAG entries list is empty.")
    
#     texts = [f"{col['name']}: {col['description']}" for col in rag_entries]

#     embeddings = get_embeddings(texts)
#     dim = embeddings.shape[1]

#     index = faiss.IndexFlatL2(dim)
#     index.add(np.array(embeddings, dtype=np.float32))

#     mapping = {i: entry for i, entry in enumerate(rag_entries)}

#     return index, mapping

def build_index(
        rag_entries: Union[List[Dict[str,str]], None] = None,
        user_columns: Union[List[Dict[str, str]], None] = None ) -> Tuple[faiss.IndexFlatL2, Dict[int, Dict[str, str]]]:
    
    if user_columns and len(user_columns) > 0:
        source_entries = user_columns
    elif rag_entries and len(rag_entries) > 0:
        source_entries = rag_entries
    else:
        raise ValueError("No RAG entries or user columns provided to build index.")
    
    texts = [f"{col['name']}: {col['description']}" for col in source_entries]

    embeddings = get_embeddings(texts)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    mapping = {i: entry for i, entry in enumerate(source_entries)}