from io import StringIO
import os
import pandas as pd
from typing import List
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def data_generator_llm(
        columns: List[str],
        num_rows: int
) -> pd.DataFrame:
    
    if not columns or not all(isinstance(c, str) for c in columns):
        raise ValueError("Columns must be a non-empty list of strings.")
    if not isinstance(num_rows, int) or num_rows <= 0:
        raise ValueError("num_rows must be a positive integer.")
    
    col_descriptions = "\n".join(
        [f"{col['name']}: {col['type_hint'] or 'string'}" for col in parsed_columns]
    )

    prompt = f"""You are a data generator. Create a synthetic dataset with {num_rows} rows and the following columns:
    {col_descriptions}
    Return ONLY the dataset in CSV format without any extra text.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    csv_output = response.text.strip()
    df = pd.read_csv(StringIO(csv_output))

    return df