# Document Retrieval & RAG using LangChain and Qdrant

An end-to-end **Retrieval-Augmented Generation (RAG)** system that ingests PDF documents, generates semantic embeddings, stores them in **Qdrant**, and retrieves relevant context to answer user queries using **LangChain**.

This project demonstrates the full RAG workflow: document loading, text chunking, vector indexing, semantic retrieval, and context-aware generation.

---

## 🚀 Features

- PDF document ingestion and preprocessing
- Recursive text chunking for efficient embeddings
- Vector indexing and similarity search using Qdrant
- Retrieval-Augmented Generation (RAG) pipeline
- Easily switch between OpenAI and open-source embedding models
- Modular and beginner-friendly code structure

---

## 🧱 Tech Stack

- **Python 3.9+**
- **LangChain**
- **Qdrant (Vector Database)**
- **OpenAI Embeddings / HuggingFace Embeddings**
- **PyPDF**
- **Docker (for Qdrant)**

---

## 📂 Project Structure

```bash
rag/
├── index.py # PDF ingestion & vector indexing
├── retrieve.py # Retrieval + RAG pipeline
├── requirements.txt
├── .env 

```