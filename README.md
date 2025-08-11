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
