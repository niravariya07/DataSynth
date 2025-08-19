from io import StringIO
import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def data_generator_llm(columns, num_rows):
    
    col_descriptions = ", ".join([
    f"{col.get('name', '')}: {col.get('description', 'string')}"
    for col in columns])

    prompt = f"""You are a data generator. Create a synthetic dataset with {num_rows} rows and the following columns:
    {col_descriptions}
    Return ONLY the dataset in CSV format without any extra text.
    """

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(prompt)

    csv_output = response.text.strip()
    df = pd.read_csv(StringIO(csv_output))

    return df
