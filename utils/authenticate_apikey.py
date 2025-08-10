import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() 
owner_api_key = genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
max_free_users = 5

