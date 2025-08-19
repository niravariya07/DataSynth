from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def get_embedding(text):
    return model.encode(text, convert_to_numpy=True)

def get_embedding_array(texts):
    return model.encode(texts, convert_to_numpy=True)