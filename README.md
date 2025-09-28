# Chemistry RAG Model

This project implements a Retrieval-Augmented Generation (RAG) system for chemistry notes. It leverages vector databases and language models to answer questions based on a provided chemistry PDF and user-uploaded content.

## Features
- Ingests and indexes chemistry notes from `chem.pdf`.
- Uses a vector database (ChromaDB) for efficient document retrieval.
- Integrates with OpenAI for answer generation.
- User management with per-user vector stores.
- Example scripts and a Jupyter notebook for experimentation.

## File Structure
- `chemistry_rag_app.py` — Main application logic.
- `chemrag.py`, `final.py`, `datasatx.py` — Supporting scripts for RAG and data processing.
- `openai.ps1` — OpenAI API integration (PowerShell script).
- `vectordb.py` — Vector database utilities.
- `rag.ipynb` — Jupyter notebook for interactive exploration.
- `chem.pdf` — Source chemistry notes.
- `chroma_db/` — Main vector database storage.
- `users_db/` — User data and per-user vector stores.
- `requirements.txt` — Python dependencies.

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Configure API keys:**
   Copy `.env.example` to `.env` and fill in your actual API keys.
3. **Run the main app:**
   ```bash
   python chemistry_rag_app.py
   ```

## Usage
- Ask chemistry questions and get answers based on the indexed notes.
- Add new users and manage their own document stores.
- Use the notebook for custom queries and experiments.

## Requirements
- Python 3.8+
- Packages listed in `requirements.txt`

## License
This project is for educational and research purposes only.
