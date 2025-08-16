import streamlit as st
from utils.synthetic_content_generator import data_generator_llm
from utils.authenticate_apikey import authenticate
from utils.load_index import load_faiss_index
from utils.retriever import retrieve_chunks
from utils.user_input_parser import user_input_parser
import pandas as pd

st.set_page_config(page_title="DataSynth", layout="centered")

if not authenticate():
    st.stop()

st.title("DataSynth")
st.subheader("Creates synthetic datasets in seconds")

st.markdown("### Define your dataset schema")
num_columns = st.number_input("Number of columns", min_value=1, max_value=20, value=3)

columns = []
for i in range(num_columns):
    col_name = st.text_input(f"Column {i+1} Name", key=f"name_{i}")
    col_desc = st.text_input(f"Column {i+1} Description", key=f"desc_{i}")
    if col_name and col_desc:
        columns.append({"name": col_name, "description": col_desc})

num_rows = st.number_input("Number of rows", min_value=1, max_value=1000, value=10)

if st.button("Generate Dataset"):
    if not columns:
        st.error("Please provide all column names and descriptions.")
    else:
        with st.spinner("Generating synthetic data..."):
            try:
                index, mapping = build_index(columns)

                context = retrieve_context(index, mapping, query="Generate synthetic dataset"
                                           )
                df = data_generator_llm(columns, num_rows)
                st.success("Dataset generated successfully!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="synthetic_dataset.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(f"Error: {str(e)}")