# NyAI-Lens — Setup Guide

## Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (for scanned PDFs)
- A [Qdrant Cloud](https://cloud.qdrant.io) account (for similar-case search)

---

## 1. Clone and enter the repo

```bash
git clone <repo-url>
cd NyAI-Lens
```

---

## 2. Pull Ollama models

The backend requires these models to be available in Ollama before starting:

```bash
ollama pull bge-m3          # embeddings (1024-dim)
ollama pull qwen3.5:8b      # RAG + rhetorical classification LLM
ollama pull gpt-oss:20b     # general LLM
ollama pull translategemma:4b
```

Verify Ollama is running:

```bash
ollama list
```

---

## 3. Install system dependencies

**Tesseract** (needed for OCR on scanned PDFs):

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract
```

---

## 4. Set up the backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install the spaCy legal NER model:

```bash
pip install https://huggingface.co/legalnlp/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl
```

Copy and fill in the environment file:

```bash
cp .env.example .env
```

Open `.env` and set the following:

| Variable | Description |
|---|---|
| `EMBED_MODEL` | Ollama embedding model (default: `bge-m3`) |
| `RAG_LLM_MODEL` | LLM for RAG chat (default: `qwen3.5:8b`) |
| `LLM_MODEL` | LLM for metadata extraction (default: `gpt-oss:20b`) |
| `RHET_MODEL` | LLM for rhetorical classification (default: `qwen3.5:8b`) |
| `TRANSLATION_MODEL` | Translation model (default: `translategemma:4b`) |
| `QDRANT_URL` | Your Qdrant Cloud cluster URL |
| `QDRANT_API_KEY` | Your Qdrant Cloud API key |

Start the backend server:

```bash
cd ..                           # back to repo root
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## 5. Set up the frontend

Open a new terminal:

```bash
cd frontend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the Streamlit app:

```bash
streamlit run app.py
```

The UI will open at `http://localhost:8501`.

---

## 6. Ingest judgments into Qdrant (for similar-case search)

Place judgment PDFs in a directory, then run:

```bash
python -m backend.ingest --pdf-dir /path/to/pdfs
```

Options:

| Flag | Default | Description |
|---|---|---|
| `--pdf-dir` | required | Directory containing PDF files |
| `--workers` | 4 | Parallel ingestion workers |
| `--qdrant-path` | (cloud) | Local Qdrant path for dev/testing |

The ingestion script extracts text, embeds each judgment using three semantic vectors (ratio, facts, full text), extracts NER spans, and upserts into Qdrant.

---

## 7. Verify everything is working

1. Open `http://localhost:8501`
2. Upload a Supreme Court or High Court PDF
3. You should see metadata, paragraph classification, NER highlights, and chat

---

## Architecture overview

```
frontend/app.py   (Streamlit)
      │  HTTP + SSE
      ▼
backend/main.py   (FastAPI, port 8000)
      ├── /extract        — PDF → structured text + metadata
      ├── /classify       — rhetorical paragraph tagging (SSE stream)
      ├── /chat           — agentic RAG Q&A (SSE stream)
      ├── /similar        — top-5 similar cases from Qdrant
      ├── /ner            — legal entity extraction
      └── /metadata       — structured metadata extraction

backend/ingest.py — offline bulk ingestion into Qdrant Cloud
```

See [Flowcharts/Documentation.md](Flowcharts/Documentation.md) for a full endpoint reference.
