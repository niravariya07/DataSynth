from typing import List
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def get_embedding(text: str) -> np.ndarray:
    return model.encode(text, convert_to_numpy=True)

def get_embeddings(texts: List[str]) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True)

if __name__ == "__main__":
    sample_texts = ["age of the person", "name of customer", "email address"]
    embeddings = get_embeddings(sample_texts)
    print(embeddings.shape)  # should be (3, 384)
    print(embeddings[0][:5])  # preview first 5 dimensions
