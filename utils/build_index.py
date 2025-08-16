from .embedder import get_embedding, get_embedding_array
from .chunks import chunk_text
from .user_input_parser import user_input_parser
import faiss
import pickle
from pathlib import Path
from typing import List, Dict

index_file = Path("faiss_index.index")
metadata_file = Path("metadata.pkl")

def build_faiss_index(columns_input: List[Dict[str, str]]):

    all_chunks: List[str] = []
    for col in columns_input:
        col_text = f"{col['name']}: {col['description']}"
        chunks = chunk_text(col_text)
        all_chunks.extend(chunks)

    embeddings = get_embedding_array(all_chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_file))

    with open(metadata_file, "wb") as f:
        pickle.dump(all_chunks, f)

    return index, all_chunks