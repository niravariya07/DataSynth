from typing import Union
from .user_input_parser import user_input_parser

def chunk_text(user_input: Union[str, dict], chunk_size: int =500, overlap: int = 50) -> list[str]:
    
    if isinstance(user_input, dict):
        text = user_input_parser(user_input)
    else:
        text = user_input

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks