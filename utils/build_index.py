from .embedder import get_embedding, get_embedding_array
from .chunks import chunk_text
from .user_input_parser import user_input_parser
import faiss
import pickle
from pathlib import Path
from typing import List

index_file = Path("faiss_index.index")
metadata_file = Path("metadata.pkl")

def build_faiss_index(user_input: str) -> None:

    parsed_data = user_input_parser(user_input)
    columns = parsed_data["columns"]

    all_chunks: List[str] = []
    for col in columns:
        chunk = chunk_text(col)
        all_chunks.extend(chunk_text(col))

    embeddings = get_embedding_array(all_chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(index_file))

    with open(metadata_file, "wb") as f:
        pickle.dump(all_chunks, f)