import os
from sentence_transformers import SentenceTransformer

# This will download it locally (default: ~/.cache/torch/sentence_transformers/)
model_name = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
save_path = "./models/all-MiniLM-L6-v2"

model = SentenceTransformer(model_name)
model.save(save_path)

print(f"Model saved locally at: {save_path}")