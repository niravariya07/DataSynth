# DataSynth

DataSynth is an AI-power*ed synthetic dataset generator that can dynamically create datasets based on **user-specified columns and row counts**.  
It leverages **Google’s Gemini API** for text generation, and is designed with an architecture that supports **Retrieval-Augmented Generation (RAG)** for richer, domain-specific dataset generation.

## Live App
**https://datasynth-dfbo7px5lwqqqcpszik6ti.streamlit.app/**

## Introduction
In AI and Data engineering,synthetic data is a powerful tool for:
- Rapid prototyping of ML models
- Avoiding privacy concerns by replacing real datasets
- Testing ETL pipelines and analytics dashboards

**With RAG integration**
DataSynth can go beyond plain data generation - by retrieving relevant context from stored documents, it can create **domain-specific synthetic datasets**.
- Generating medical trial datasets informed by real-world schema descriptions
- Producing financial reports based on retrieved industry formats

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| **streamit** | 1.36.0 | Web app interface |
| **pandas** | 2.2.2 | Data handling |
| **google-generativeai** | 0.3.0 | Access Gemini API |
| **python-dotenv** | 1.0.1 | Load environment variables |
| **io** | Built-in | String buffer for CSV handling |
| **numpy** | 1.26.0 | Array & numerical ops for dataset processing |
| **faiss-cpu** | 1.8.0 | Vector database for RAG |
| **langchain** | ^0.2.10 | RAG orchestration & prompt management |
| **langchain-community** | ^0.2.7 | Community integrations for LangChain |
| **sentence-transformers** | ^2.2.2 | Embedding generation for retrieval |


## Project Structure
    datasynth/
    │
    ├── data/ # Optional: Store generated datasets / RAG source docs
    │
    ├── utils/
    │ ├── embedder.py # Embedding generation (SentenceTransformers)
    │ ├── build_index.py # Creates FAISS index from RAG documents
    │ ├── retriever.py # Retrieves context from stored embeddings
    │ ├── generator.py # Dataset generation logic
    │ ├── user_input_parser.py # Parses and validates user inputs
    │ ├── synthetic_content_generator.py # Calls Gemini API to generate data
    │ ├── chunks.py # Splits docs into chunks for embedding
    │
    ├── auth.py # Handles API key authentication & usage limits
    ├── app.py # Main Streamlit application
    ├── requirements.txt # Python dependencies
    ├── README.md # Documentation
    └── .env # API keys & config


## How It Works

**Without RAG (Basic Mode)**  
1. **User Authentication**  
   - User enters their own Gemini API key.  
   - If no key is provided, the system offers a fallback key (max 5 uses).  

2. **Dataset Specification**  
   - User specifies column names & descriptions.  
   - User enters the desired number of rows.  

3. **AI Generation**  
   - Gemini generates the dataset in CSV format.  
   - Data is parsed into a Pandas DataFrame.  

4. **Download**  
   - User can instantly download the generated dataset.  
