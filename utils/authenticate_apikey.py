import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv() 
owner_api_key = genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
max_free_users = 5

def authenticate():
    if "auth_passed" not in st.session_state:
        st.session_state.auth_passed = False
    if "free_uses" not in st.session_state:
        st.session_state.free_users = 0
    if "api_key" not in st.session_state:
        st.session_state.api_key = None

    st.title("DataSynth Authentication")
    st.write("To generate datasets, you need a Gemini API key.")

    api_key_input = st.text_input(
        "Enter your Gemini API key (leave blank if you don't have one)",
        type="password"
    )

    if api_key_input:
        st.session_state.api_key = api_key_input
        st.session_state.auth_passed = True
        st.success("API key accepted! You can now use DataSynth.")
        st.rerun()

    else:
        st.info(f"You can use the owner's API key up to {max_free_users} times for free..")
        if st.button("Use Owner's API key"):
            if st.session_state.free_users < max_free_users:
                st.session_state.api_key = owner_api_key
                st.session_state.free_users += 1
                st.session_state.auth_passed = True
                st.success(f"Using owner's API key (Usage {st.session_state.free_users}/{max_free_users})")
                st.rerun()
            else:
                st.error("Free usage limit reached. Please enter your own API key.")

    return st.session_state.auth_passed