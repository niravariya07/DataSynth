from .embedder import get_embedding_array
from .chunks import chunk_text
import faiss
import pickle
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict, Union

def build_index()