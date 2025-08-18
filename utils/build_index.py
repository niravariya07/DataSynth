import os
import numpy as np
from .embedder import get_embedding
from .chunks import chunk_text
import faiss
import pickle
from pathlib import Path

data_dir = 'data'
index_file = Path("/faiss/faiss_index.index")
metadata_file = Path("/faiss/metadata.pkl")

def build_faiss_index():
    docs = []
    id_to_text = {}
    idx = 0

    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            with open(os.path.join(data_dir, file), "r") as f:
                text = f.read()
                chunks = chunk_text(text)
                for chunk in chunks:
                    docs.append(chunk)
                    id_to_text[idx] = chunk
                    idx += 1

    embeddings = [get_embedding(doc) for doc in docs]
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, str(index_file))
    with open(metadata_file, "wb") as f:
        pickle.dump(id_to_text, f)

    return index, id_to_text