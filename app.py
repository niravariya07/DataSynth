import streamlit as st
import pandas as pd
from utils.synthetic_content_generator import data_generator_llm

st.set_page_config(page_title="DataSynth", layout="centered")

st.title("DataSynth")
st.subheader("Creates synthetic datasets in seconds")

num_columns = st.number_input("Number of columns", min_value=1, max_value=20, value=3)

