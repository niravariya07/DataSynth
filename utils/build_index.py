from .embedder import get_embedding_array, get_embedding
from .chunks import chunk_text
import faiss
import pickle
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Union

index_file = Path("faiss_index.index")
metadata_file = Path("metadata.pkl")

def build_faiss_index(columns: List[str]) -> None:
    all_chunks: List[str] = []
    for col in columns:
        chunk = chunk_text(col)
        all_chunks.extend(chunk)

    embeddings = get_embedding(all_chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    