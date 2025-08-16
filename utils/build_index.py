from .embedder import get_embedding_array
from .chunks import chunk_text
import faiss
import pickle
from pathlib import Path
from typing import List, Dict

index_file = Path("/faiss/faiss_index.index")
metadata_file = Path("/faiss/metadata.pkl")

def build_faiss_index(columns_input: List[str]):
    all_chunks: List[str] = []
    for col in columns_input:
        all_chunks.extend(chunk_text(col))
        
    embeddings = get_embedding_array(all_chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_file))

    with open(metadata_file, "wb") as f:
        pickle.dump(all_chunks, f)

    return index, all_chunks