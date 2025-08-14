import os
from sentence_transformers import SentenceTransformer
model_name = "sentence-transformers/all-MiniLM-L6-v2"

model = SentenceTransformer(model_name)
save_path = "./models/all-MiniLM-L6-v2"
os.makedirs(save_path, exist_ok=True)
model.save(save_path)

print(f"Model saved locally at: {save_path}")