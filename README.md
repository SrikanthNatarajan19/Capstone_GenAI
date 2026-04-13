# Academic Question Answering using RAG + FLAN-T5

## Overview
This project implements a web-based academic question answering and summarization system using:
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- FLAN-T5-base

The system allows users to upload academic PDFs or paste text, retrieve relevant context using semantic search, and generate grounded answers.

## Features
- PDF upload and text extraction
- Manual text input
- Text chunking with overlap
- Embedding generation using sentence-transformers
- Vector search using FAISS
- Grounded answer generation using FLAN-T5-base
- Document summarization
- Inference time and memory tracking
- Simple QA evaluation using token-level F1

## Installation
```bash
pip install -r requirements.txt