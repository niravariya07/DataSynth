# DataSynth

DataSynth is an AI-powered synthetic dataset generator that can dynamically create datasets based on **user-specified columns and row counts**.  
It leverages **Google’s Gemini API** for text generation, and is designed with an architecture that supports **Retrieval-Augmented Generation (RAG)** for richer, domain-specific dataset generation.






DataSynth is a **Streamlit-powered synthetic dataset generator** that uses the Gemini API to create datasets on demand.
1. Define your **column names** and **number of rows**.
2. Let the model craft a realistic, 
mock dataset for you.
3. Download it instantly as CSV.

## Features
- **Dynamic Dataset Generation** - Using Gemini AI.
- **Authentication System** - Use your own API key or the owner's API key (limited free uses).
- **Streamlit UI** - Simple, browser-based interface.
- **Download as CSV** - Save generated data in one click.

## Tech Stack
- Python 3.10+
- Streamlit-UI
- Google Generative AI (Gemini) - Dataset generation
- dotenv-Environment variable management
- Retrieval Augmented Generation