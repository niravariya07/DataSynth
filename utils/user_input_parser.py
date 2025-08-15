from dataclasses import dataclass
from typing import List, Literal

ColumnType = Literal["string", "int", "float", "bool", "date"]

@dataclass
class ColumnSpec:
    name: str
    dtype: ColumnType

@dataclass
class UserInput:
    columns: List[ColumnSpec]
    row_count: int


def user_input_parser(raw_columns: List[str], raw_row_count: int) -> UserInput:

    if not isinstance(raw_columns, list) or not all(isinstance(c, str) for c in raw_columns):
        raise ValueError("No column names provided.")
    
    if not isinstance(raw_row_count, int) or raw_row_count <= 0:
        raise ValueError("C'mon enter rows in positive....")
    
    parsed_columns = []
    for col in raw_columns:
        parts = col.split(":")
        name = parts[0].strip()
        dtype = parts[1].strip().lower() if len(parts) > 1 else "string"

        