import os
import pandas as pd
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def data_generator_llm(
        columns: List[Dict[str, str]],
        num_rows: int
) -> pd.DataFrame:
    
    col_descriptions = "\n".join(
        [f"{col['name']}: {col['description']}" for col in columns]
    )

    prompt = f"""You are a data generator. Create a synthetic dataset with {num_rows} rows and the following columns:
    {col_descriptions}
    Return ONLY the dataset in CSV format without any extra text.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )

    csv_output = response.choices[0].message.content.strip()

    from io import StringIO
    df = pd.read_csv(StringIO(csv_output))

    return df

if __name__ == "__main__":
    # Example usage
    sample_columns = [
        {"name": "Name", "description": "Full name of the customer"},
        {"name": "Email", "description": "Email address in standard format"},
        {"name": "Age", "description": "Age in years (integer)"},
    ]

    df = data_generator_llm(sample_columns, 5)
    print(df)