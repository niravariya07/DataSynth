import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
client = genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
