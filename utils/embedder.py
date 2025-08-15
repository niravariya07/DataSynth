from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('./models/all-MiniLM-L6-v2')

def get_embedding(text: str) -> np.ndarray:
    return model.encode(text, convert_to_numpy=True)

def get_embedding_array(texts: List[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True)
