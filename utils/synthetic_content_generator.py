from io import StringIO
import os
import pandas as pd
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv
from .user_input_parser import user_input_parser

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def data_generator_llm(
        columns: List[str],
        num_rows: int
) -> pd.DataFrame:
    
    parsed_columns, num_rows = user_input_parser(columns, num_rows)
    
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