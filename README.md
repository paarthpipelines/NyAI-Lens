# NyAI-Lens — Indian Court Judgment Analyser

A FastAPI backend + Streamlit frontend for analyzing Indian Supreme Court judgments.

## Setup

### 1. Copy environment config

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` and fill in your **Qdrant API key** and the Ollama model names you want to use.

### 2. Install Ollama models

Pull each model before starting the server:

```bash
ollama pull bge-m3          # embedding model (must produce 1024-dim vectors)
ollama pull llama3.3:70b    # LLM for extraction & RAG
ollama pull deepseek-r1:7b  # rhetorical classification
ollama pull translategemma:4b
```

### 3. Install Python dependencies

```bash
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
```

Also install the legal NER spaCy model:

```bash
pip install https://huggingface.co/opennyaiorg/en_legal_ner_trf/resolve/main/en_legal_ner_trf-any-py3-none-any.whl
```

### 4. Populate the Qdrant corpus (required for Similar Judgments)

```bash
# Ingest a folder of PDF judgments into Qdrant Cloud (uses QDRANT_API_KEY from .env)
python -m backend.ingest --pdf-dir /path/to/judgment_pdfs

# With concurrency and a debug limit:
python -m backend.ingest --pdf-dir /path/to/judgment_pdfs --workers 4 --limit 10

# Dev / local Qdrant (no auth needed):
python -m backend.ingest --pdf-dir /path/to/judgment_pdfs --qdrant-path ./qdrant_local
```

## Running

Start the backend (from the project root):

```bash
uvicorn backend.main:app --reload
```

In a separate terminal, start the frontend:

```bash
streamlit run frontend/app.py
```

The frontend connects to `http://localhost:8000` by default. Change `BACKEND_URL` at the top of [frontend/app.py](frontend/app.py) if needed.

## API Endpoints

| Method | Path          | Description |
|--------|---------------|-------------|
| POST   | `/extract`    | Upload PDF → `{ doc_id, markdown, header, chunks, footer, metadata }` |
| POST   | `/classify`   | `{ doc_id, indices? }` — streams SSE, one JSON per paragraph |
| POST   | `/chat`       | `{ doc_id, question, history }` — streams SSE tokens |
| POST   | `/translate`  | `{ text, language }` — streams SSE tokens |
| GET    | `/ner`        | `?text=...` → `{ spans: [{text, label}] }` (text capped at 10 K chars) |
| POST   | `/similar`    | Upload PDF → top-5 similar judgments from Qdrant corpus |
| GET    | `/debug/footer` | `?doc_id=...` → raw footer extraction data (debugging) |

## Architecture

```
frontend/app.py      ← Streamlit UI (upload, view, chat, translate, similar)
backend/main.py      ← FastAPI server (all endpoints + paragraph parsing)
backend/ingest.py    ← Batch ingest script (PDFs → Qdrant)
backend/legal_normalizer.py  ← Canonical entity keys for Jaccard comparison
backend/text_utils.py        ← Shared text-cleaning helpers
```

> **Note:** `backend/chunker.py` is deprecated — all chunking logic lives in `main.py`.
