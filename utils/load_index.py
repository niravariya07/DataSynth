import pickle
from .build_index import index_file, metadata_file
from .user_input_parser import user_input_parser
from typing import List, Tuple, Union
import faiss

def load_faiss_index(user_input: Union[dict, None] = None) -> tuple[faiss.Index,List[str]]:
    if not index_file.exists() or not metadata_file.exists():
        raise FileNotFoundError("FaISS index or metadat file not found. Please the index first.")
    
    index = faiss.read_index(str(index_file))
    with open(metadata_file, "rb") as f:
        metadata: List[str] = pickle.load(f)

    if user_input:
        parsed_data = user_input_parser(user_input)
        if isinstance(parsed_data, list):
            metadata = parsed_data
        else:
            metadata = [parsed_data]
    
    return index, metadata