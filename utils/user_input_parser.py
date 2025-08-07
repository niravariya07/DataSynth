from typing import List, Tuple, Dict

def user_input_parser(columns: List[str], num_rows: int) -> Tuple[List[Dict], int]:
    if not columns:
        raise ValueError("No column names provided.")
    
    if num_rows <= 0:
        raise ValueError("C'mon enter rows in positive....")
    
    parsed_columns = []
    for col in columns:
        if ":" in col:
            name, type_hint = col.split(":", 1)
            name = name.strip()
            type_hint = type_hint.strip().lower()
        else:
            name = col.strip()
            type_hint = None

        parsed_columns.append({
            "name" : name,
            "raw": col,
            "type_hint": type_hint
        })

    return parsed_columns, num_rows