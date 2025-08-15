from typing import Union
from sentence_transformers import SentenceTransformer
import numpy as np
from .user_input_parser import user_input_parser

model = SentenceTransformer('./models/all-MiniLM-L6-v2')

def get_embedding(user_input: Union[str, dict]) -> np.ndarray:
    if isinstance(user_input, dict):
        parsed_text = user_input_parser(user_input)
    else:
        parsed_text = user_input

    return model.encode(parsed_text, convert_to_numpy=True)
