import streamlit as st
from utils.synthetic_content_generator import data_generator_llm
from utils.authenticate_apikey import authenticate
from utils.load_index import load_faiss_index
from utils.retriever import retrieve_chunks
from utils.build_index import build_faiss_index
from utils.user_input_parser import user_input_parser
import pandas as pd

st.set_page_config(page_title="DataSynth", layout="centered")

if not authenticate():
    st.stop()

st.title("DataSynth")
st.subheader("Creates synthetic datasets in seconds")

try:
    index, metadata = load_faiss_index()
except FileNotFoundError:
    st.warning("Index not found. Building index now...")
    index, metadata = build_faiss_index()
    
st.markdown("### Define your dataset schema")
num_columns = st.number_input("Number of columns", min_value=1, max_value=20, value=3)

columns_input = []
for i in range(num_columns):
    col_name = st.text_input(f"Column {i+1} Name", key=f"name_{i}")
    col_desc = st.text_input(f"Column {i+1} Description", key=f"desc_{i}")
    if col_name and col_desc:
        columns_input.append({"name": col_name, "description": col_desc})

num_rows = st.number_input("Number of rows", min_value=1, max_value=1000, value=10)

if st.button("Generate Dataset"):
    if not columns_input:
        st.error("Please provide all column names and descriptions.")
    else:
        try:
            with st.spinner("Loading FAISS index..."):
                index, metadata = load_faiss_index()

            with st.spinner("Parsing user input..."):
                user_query = user_input_parser(columns_input)
            
            with st.spinner("Retrieving relevant context..."):
                context_chunks = retrieve_chunks(index, metadata, user_query)
            
            with st.spinner("Generating synthetic dataset..."):
                df = data_generator_llm(columns_input, num_rows, context_chunks)
                
                st.success("Dataset generated successfully!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="synthetic_dataset.csv",
                    mime="text/csv",
                )
        # except Exception as e:
        #     st.error(f"Index not found. Please build the index first.: ({str(e)})")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")