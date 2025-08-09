import streamlit as st
import pandas as pd
from utils.synthetic_content_generator import data_generator_llm

st.set_page_config(page_title="DataSynth", layout="centered")

st.title("DataSynth")
st.subheader("Creates synthetic datasets in seconds")

num_columns = st.number_input("Number of columns", min_value=1, max_value=20, value=3)

columns = []
for i in range(num_columns):
    col_name = st.text_input(f"Column {i+1} Name", key=f"name_{i}")
    col_desc = st.text_input(f"Column {i+1} Description", key=f"desc_{i}")
    if col_name and col_desc:
        columns.append({"name": col_name, "description": col_desc})


num_rows = st.number_input("Number of rows", min_value=1, max_value=1000, value=10)