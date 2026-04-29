import asyncio
import json
import logging
import numpy as np
import os
import re
import tempfile
import threading
import time
import uuid
import warnings
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Any
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

import fitz  # pymupdf
import PIL.Image
import pymupdf4llm
import pytesseract
import spacy
from fastapi import FastAPI, File, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from pydantic import BaseModel, Field

import faiss

from backend.legal_normalizer import canonicalize_spans
from backend.text_utils import (
    _DIGITAL_SIG_RE,
    _FOOTER_SIG_RE,
    _PAGE_PATTERNS,
    _ROMAN_PAGE_RE,
    _is_page_line,
    strip_page_numbers,
)

async def _warmup(loop: asyncio.AbstractEventLoop) -> None:
    await asyncio.gather(
        loop.run_in_executor(None, get_cross_encoder),
        loop.run_in_executor(None, get_qdrant),
        loop.run_in_executor(None, get_similar_embedder),
        loop.run_in_executor(None, get_nlp),
        get_rhet_chain_async(),
    )

@asynccontextmanager
async def lifespan(app):
    loop = asyncio.get_running_loop()
    asyncio.create_task(_warmup(loop))
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model configs ──────────────────────────────────────────────────────────────
# Override any model via env var: RHET_MODEL, RAG_LLM_MODEL, EMBED_MODEL
_RHET_OLLAMA_MODEL   = os.getenv("RHET_MODEL",        "deepseek-v3.1:671b-cloud")
_RAG_LLM_MODEL       = os.getenv("RAG_LLM_MODEL",     "gpt-oss:120b-cloud")
_RAG_EMBED_MODEL     = os.getenv("EMBED_MODEL",        "bge-m3")
_SIMILAR_EMBED_MODEL = os.getenv("EMBED_MODEL",        "bge-m3")  # 1024-dim; used for /similar and ingest
_SIMILAR_EMBED_DIM   = 1024
_QDRANT_COLLECTION   = "indian_judgements"
_CHUNK_LLM_MODEL     = os.getenv("CHUNK_LLM_MODEL",   _RHET_OLLAMA_MODEL)

# ── Page-number line detection ─────────────────────────────────────────────────

# ── Footer metadata patterns ───────────────────────────────────────────────────
_FOOTER_JUDGE_NAME_RE = re.compile(
    r'^[\[(]([A-Za-z][^\]\)\n]{1,80})[\])]\s*$',
)
_JUDGMENT_DATE_RE = re.compile(
    r'\b(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST'
    r'|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)'
    r'\s+(?:\d{1,2},\s*)?\d{4}\b',
    re.IGNORECASE,
)
_JUDGMENT_LOCATION_RE = re.compile(
    r'\b(NEW\s+DELHI|MUMBAI|CHENNAI|KOLKATA|HYDERABAD|BENGALURU|BANGALORE)\b',
    re.IGNORECASE,
)

# ── Para-body line-joining helpers ─────────────────────────────────────────────
_SENT_END_RE = re.compile(r'[.;:—–!?]\*{0,4}\s*$')
_STRUCT_NEXT_RE = re.compile(
    r'^\|'                          # table rows — never merge into preceding line
    r'|^>|^[-*+]\s'
    r'|^\(([a-z]|[ivx]+|\d)\)'
    r'|^\d+\.\d'
    r'|^#{1,4}\s'
    r'|^\*\*[A-Z]'
)
_PAGE_OR_SIG = re.compile(
    r'^\s*(?:Page\s+\d+\s+of\s+\d+|\d+\s*/\s*\d+|\d{1,3})\s*$',
    re.IGNORECASE,
)

# ── Shared LCEL post-processing steps ─────────────────────────────────────────
# Strips <think>…</think> reasoning blocks emitted by some local models (e.g. DeepSeek-R1).
_STRIP_THINK = RunnableLambda(
    lambda t: re.sub(r'<think>.*?</think>', '', t, flags=re.DOTALL).strip()
)
# Strips markdown code fences (```json … ```) that models sometimes wrap output in.
_STRIP_FENCES = RunnableLambda(
    lambda t: re.sub(r'^```[a-z]*\s*|\s*```$', '', t.strip(), flags=re.MULTILINE).strip()
)

# ── Judge-line detection for implicit-para-1 ───────────────────────────────────
_JUDGE_LINE_RE = re.compile(
    r'^[A-Z][A-Z\s.,]+J\.$|\*\*[A-Z\s.,]+J\.\*\*',
    re.MULTILINE,
)

# ── Upper-metadata boundary ────────────────────────────────────────────────────
# Matches either:
#   JUDGEMENT/JUDGMENT  (then a judge-name line with J./CJI.)
#   ORDER               (standalone — no following judge name required)
_JUDGE_SUFFIX = r'(?:J\.|CJI\.?|,\s*J)'   # matches "J.", "CJI", "CJI.", ", J"

_UPPER_META_END_RE = re.compile(
    r'(?:'
    r'\bJUDG[EE]?MENT\b[^\n]*\n(?:[^\S\n]*\n)*[^\n]*' + _JUDGE_SUFFIX + r'[^\n]*'  # JUDGEMENT + judge
    r'|'
    r'^\*{0,2}\s*ORDER\s*\*{0,2}\s*$'                                                # standalone ORDER line
    r')',
    re.IGNORECASE | re.MULTILINE,
)

# Captures the authoring judge name line that follows a JUDGEMENT/JUDGMENT heading.
_AUTHORING_JUDGE_RE = re.compile(
    r'\bJUDG[EE]?MENT\b[^\n]*\n(?:[^\S\n]*\n)*([^\n]*' + _JUDGE_SUFFIX + r'[^\n]*)',
    re.IGNORECASE,
)

# ── NER label config ───────────────────────────────────────────────────────────
_LEGAL_LABELS: dict[str, str] = {
    "PROVISION":  "#F0B27A",
    "STATUTE":    "#F1948A",
    "PRECEDENT":  "#C39BD3",
}

# ── Rhetorical role config ─────────────────────────────────────────────────────
_LABEL_MAP: dict[str, str] = {
    "[PREAMBLE]":             "Preamble",
    "[FACT_ESSENTIAL]":       "Facts",
    "[FACT_PROCEDURAL]":      "Procedural",
    "[EVIDENCE_REF]":         "Evidence",
    "[ISSUE_LEGAL]":          "Issue",
    "[ARG_PETITIONER]":       "Arg · Petitioner",
    "[ARG_RESPONDENT]":       "Arg · Respondent",
    "[PRECEDENT_ANALYSIS]":   "Precedent",
    "[JUDICIAL_OBSERVATION]": "Observation",
    "[RATIO_DECIDENDI]":      "Ratio Decidendi",
    "[DISPOSITION]":          "Disposition",
    "[DIRECTIVES]":           "Directives",
    "[COSTS]":                "Costs",
}
_VALID_ROLES: list[str] = list(_LABEL_MAP.values())

_RHET_SYSTEM = (
    "You are a Senior Legal Document Architect specialising in Indian Supreme Court judgments.\n"
    "Classify the paragraph into EXACTLY ONE role.\n\n"
    f"ROLES (use exactly as written): {', '.join(_VALID_ROLES)}\n\n"
    "Context is provided as three snippets: [PREV], [CURR], [NEXT].\n"
    "Focus on [CURR]. Use [PREV] and [NEXT] only to resolve ambiguity.\n\n"
    "OUTPUT FORMAT (strict JSON, no markdown, no extra text):\n"
    '{{"para_no":"{para_no}","voice":"<Court|Petitioner|Respondent|Lower Court>",'
    '"role":"<exact role name>","confidence":<0.0-1.0>}}'
)
_RHET_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _RHET_SYSTEM),
    ("human",
     "Para {para_no}:\n"
     "[PREV] {prev}\n"
     "[CURR] {curr}\n"
     "[NEXT] {next}"),
])

# ── RAG config ─────────────────────────────────────────────────────────────────
_RAG_SYSTEM = (
    "You are a legal research assistant. Answer the user's question using ONLY "
    "the judgment excerpts provided below. Cite relevant paragraphs inline as "
    "[Para N] (e.g. [Para 12], [Para 47]). You may cite multiple paragraphs. "
    "If the answer is not present in the excerpts, say so clearly. "
    "Do not hallucinate facts.\n\n"
    "JUDGMENT EXCERPTS:\n{context}"
)
_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _RAG_SYSTEM),          # {context} filled at invoke time
    MessagesPlaceholder("history"),    # prior turns as LangChain message objects
    ("human", "{question}"),
])

_BATCH_SIZE          = 8
_CLASSIFY_WORKERS    = 4
_CLASSIFY_RETRIES    = 2
_CLASSIFY_CONCURRENCY = int(os.getenv("CLASSIFY_CONCURRENCY", "2"))
_classify_semaphore  = asyncio.Semaphore(_CLASSIFY_CONCURRENCY)

# ── Extraction (for /similar and ingest) ───────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a senior Indian legal analyst. Given the text of a court judgment, "
    "extract the following six fields:\n\n"
    "1. summary: A 3-5 sentence summary covering who the parties are, the core legal "
    "dispute, the procedural history, and the final decision.\n\n"
    "2. ratio_decidendi: A list of 2-6 sentences, each capturing one distinct legal "
    "principle or holding laid down by this judgment that carries precedential weight. "
    "Be precise; cite specific sections/articles where relevant.\n\n"
    "3. outcome: Exactly one of: allowed, dismissed, remanded, affirmed, partly allowed, "
    "quashed, set aside, unknown.\n\n"
    "4. provisions: A list of statutory provisions cited, each as "
    '{\"text\": \"Section 302 IPC\", \"role\": \"decisive\"}. '
    "Role must be one of: decisive, supporting, background, distinguished.\n\n"
    "5. precedents: A list of case citations relied upon, each as "
    '{\"citation\": \"AIR 1961 SC 1808\", \"short_name\": \"Kehar Singh v State\", \"role\": \"relied_upon\"}. '
    "Role must be one of: relied_upon, followed, distinguished, overruled, merely_cited.\n\n"
    "6. statutes: A list of bare statute names cited (e.g. \"Indian Penal Code\", \"CrPC\").\n\n"
    "Respond with ONLY valid JSON — no markdown fences, no explanation:\n"
    '{"summary": "...", "ratio_decidendi": ["..."], "outcome": "...", '
    '"provisions": [{"text": "...", "role": "..."}], '
    '"precedents": [{"citation": "...", "short_name": "...", "role": "..."}], '
    '"statutes": ["..."]}'
)
# SystemMessage (not tuple) prevents ChatPromptTemplate from treating the JSON
# examples inside SYSTEM_PROMPT as template variables.
_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYSTEM_PROMPT),
    ("human", "{text}"),
])


class CasedProvision(BaseModel):
    text: str
    role: str  # decisive | supporting | background | distinguished


class CasedPrecedent(BaseModel):
    citation: str
    short_name: str
    role: str  # relied_upon | followed | distinguished | overruled | merely_cited


class Extraction(BaseModel):
    summary: str = Field(default="")
    ratio_decidendi: list[str] = Field(default_factory=list)
    outcome: str = Field(
        default="unknown",
        description=(
            "One of: allowed, dismissed, remanded, affirmed, "
            "partly allowed, quashed, set aside, unknown"
        ),
    )
    provisions: list[CasedProvision] = Field(default_factory=list)
    precedents: list[CasedPrecedent] = Field(default_factory=list)
    statutes: list[str] = Field(default_factory=list)


def _parse_extraction(raw: str) -> Extraction:
    """Parse LLM JSON output → Extraction, with graceful fallback."""
    clean = re.sub(r'^```[a-z]*\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(clean)
        return Extraction(
            summary=str(data.get("summary", "")),
            ratio_decidendi=[str(r) for r in data.get("ratio_decidendi", [])],
            outcome=str(data.get("outcome", "unknown")),
            provisions=[
                CasedProvision(**p) for p in data.get("provisions", [])
                if isinstance(p, dict) and "text" in p and "role" in p
            ],
            precedents=[
                CasedPrecedent(**p) for p in data.get("precedents", [])
                if isinstance(p, dict) and "citation" in p and "short_name" in p and "role" in p
            ],
            statutes=[str(s) for s in data.get("statutes", [])],
        )
    except Exception:
        m_s = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
        m_o = re.search(r'"outcome"\s*:\s*"([^"]+)"', raw)
        return Extraction(
            summary=m_s.group(1) if m_s else "",
            ratio_decidendi=[],
            outcome=m_o.group(1) if m_o else "unknown",
        )


def _extract_plain_text(pdf_bytes: bytes, max_chars: int = 15_000) -> str:
    """Fast fitz plain-text extraction, truncated to max_chars."""
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        doc = fitz.open(tmp_path)
        parts = [doc[i].get_text() for i in range(len(doc))]
        doc.close()
        return "\n".join(parts)[:max_chars]
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _llm_extract(text: str) -> Extraction:
    return _parse_extraction(_get_extract_chain().invoke({"text": text}))


# ── Similarity helpers ─────────────────────────────────────────────────────────

_BM25_STOPWORDS = frozenset({
    "the", "of", "and", "in", "to", "a", "that", "is", "this", "it",
    "was", "for", "on", "are", "be", "has", "had", "with", "have",
    "as", "at", "by", "from", "or", "an", "not", "been", "their",
    "court", "also", "bench", "matter", "case", "learned", "counsel",
    "appellant", "respondent", "petitioner", "writ", "appeal", "petition",
})


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _shared_raw(query_canon: set, doc_canon: list[str], doc_raw: list[str]) -> list[str]:
    return [doc_raw[i] for i, c in enumerate(doc_canon) if c in query_canon]


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ── Singleton model loaders ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_nlp():
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return spacy.load("en_legal_ner_trf")
    except Exception:
        return None


_rhet_chain: Any = None

async def get_rhet_chain_async():
    global _rhet_chain
    if _rhet_chain is None:
        llm = ChatOllama(model=_RHET_OLLAMA_MODEL)
        _rhet_chain = _RHET_CHAT_PROMPT | llm | StrOutputParser()
    return _rhet_chain


_chunk_llm_local = threading.local()

def get_chunk_llm() -> ChatOllama:
    if not hasattr(_chunk_llm_local, "llm"):
        _chunk_llm_local.llm = ChatOllama(model=_CHUNK_LLM_MODEL, temperature=0)
    return _chunk_llm_local.llm


_extract_chain_local = threading.local()

def _get_extract_chain():
    if not hasattr(_extract_chain_local, "chain"):
        _extract_chain_local.chain = (
            _EXTRACT_PROMPT | ChatOllama(model=_RAG_LLM_MODEL) | StrOutputParser()
        )
    return _extract_chain_local.chain


_rag_chain_local = threading.local()

def _get_rag_chain():
    if not hasattr(_rag_chain_local, "chain"):
        _rag_chain_local.chain = (
            _RAG_PROMPT | ChatOllama(model=_RAG_LLM_MODEL) | StrOutputParser()
        )
    return _rag_chain_local.chain


@lru_cache(maxsize=1)
def get_cross_encoder():
    from sentence_transformers import CrossEncoder  # noqa: PLC0415
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# Maximum number of cached document stores before evicting oldest (prevents OOM)
_DOC_STORE_MAX = 20


@lru_cache(maxsize=1)
def get_qdrant():
    from qdrant_client import QdrantClient  # noqa: PLC0415
    path = os.getenv("QDRANT_PATH")
    if path:
        return QdrantClient(path=path)
    url = os.getenv("QDRANT_URL")
    if not url:
        raise RuntimeError(
            "QDRANT_URL is not set. Add it to backend/.env — see .env.example."
        )
    api_key = os.getenv("QDRANT_API_KEY")
    if not api_key:
        logging.warning(
            "QDRANT_API_KEY is not set — attempting unauthenticated connection. "
            "Set QDRANT_API_KEY in backend/.env for Qdrant Cloud."
        )
        return QdrantClient(url=url, timeout=60)
    return QdrantClient(url=url, api_key=api_key, timeout=60)


@lru_cache(maxsize=1)
def get_similar_embedder():
    return OllamaEmbeddings(model=_SIMILAR_EMBED_MODEL)


# ── Server-side document & embedding stores ────────────────────────────────────
_doc_store: dict[str, dict] = {}       # doc_id → extracted document data
_embedding_cache: dict[str, tuple] = {}     # doc_id → (faiss_index, para_keys)
_embed_events: dict[str, asyncio.Event] = {}  # doc_id → Event set when index is ready


def _evict_if_needed() -> None:
    """Evict the oldest entry from _doc_store and _embedding_cache when over limit.
    Called before inserting a new document to bound memory usage."""
    while len(_doc_store) >= _DOC_STORE_MAX:
        oldest_key = next(iter(_doc_store))
        _doc_store.pop(oldest_key, None)
        _embedding_cache.pop(oldest_key, None)
        _embed_events.pop(oldest_key, None)
        logging.info("Evicted doc_store entry %s (cache limit %d reached)", oldest_key, _DOC_STORE_MAX)


# ── Page-number helpers: imported from backend.text_utils ─────────────────────
# _is_page_line and strip_page_numbers are no longer defined here;
# they are re-exported via the import at the top of this file.



# ── Paragraph parsing ──────────────────────────────────────────────────────────

def para_number(chunk: str, fallback: str = "?") -> str:
    m = re.match(r"\*\*(\d+)\.\*\*|^_?(\d+)\._?\s", chunk)
    return (m.group(1) or m.group(2)) if m else fallback


def para_body(chunk: str) -> str:
    body = re.sub(r"^\*\*\d+\.\*\*\s*|^_?\d+\._?\s+", "", chunk, count=1).strip()
    body = _DIGITAL_SIG_RE.sub('', body)
    body = strip_page_numbers(body)

    lines = body.split('\n')
    merged = []
    i = 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            merged.append(ln)
            i += 1
            continue
        while (
            ln.strip()
            and not _SENT_END_RE.search(ln.rstrip())
            and i + 1 < len(lines)
            and lines[i + 1].strip()
            and not _STRUCT_NEXT_RE.match(lines[i + 1].strip())
            and not _PAGE_OR_SIG.match(lines[i + 1])
        ):
            i += 1
            ln = ln.rstrip() + ' ' + lines[i].strip()
        merged.append(ln)
        i += 1
    return '\n'.join(merged)


def _is_section_heading(line: str) -> bool:
    s = line.strip()
    cleaned = re.sub(r'^#{1,4}\s*', '', s)
    cleaned = re.sub(r'^\*{1,2}|\*{1,2}$', '', cleaned).strip()
    if not cleaned or len(cleaned) > 40:
        return False
    # Parenthesised text is a judge/party name, not a section heading
    if cleaned.startswith('(') and cleaned.endswith(')'):
        return False
    # Party names contain ampersands; section headings never do
    if '&' in cleaned:
        return False
    if cleaned.endswith('.'):
        return False
    if re.match(r'^\d+\.', cleaned):
        return False
    if _is_page_line(cleaned):
        return False
    letters = re.sub(r'[^A-Za-z]', '', cleaned)
    if len(letters) < 7:
        return False
    return cleaned == cleaned.upper()


def _enforce_sequential(chunks: list) -> list:
    merged = []
    expected = None
    for chunk in chunks:
        m = re.match(r"\*\*(\d+)\.\*\*|^_?(\d+)\._?\s", chunk)
        if m:
            num = int(m.group(1) or m.group(2))
            if expected is None or num == expected:
                merged.append(chunk)
                expected = num + 1
            else:
                if merged:
                    merged[-1] += "\n\n" + chunk
                else:
                    merged.append(chunk)
                expected = num + 1
        else:
            if merged:
                merged[-1] += "\n\n" + chunk
            else:
                merged.append(chunk)
    return merged


def _log_sequence_gaps(chunks: list) -> None:
    """Warn about non-consecutive paragraph numbers. Does not mutate chunks."""
    prev: int | None = None
    for c in chunks:
        try:
            n = int(c["para_no"])
            if prev is not None and n != prev + 1:
                logging.warning(
                    "Paragraph sequence gap detected: %d → %d (section: %r)",
                    prev, n, c.get("section", ""),
                )
            prev = n
        except ValueError:
            pass  # para_no is "?" or non-numeric — skip


def extract_footer(last_chunk: str) -> tuple:
    lines = last_chunk.split('\n')
    for i, line in enumerate(lines):
        s = re.sub(r'\*+', '', line).strip()   # strip bold markers before matching
        if _FOOTER_SIG_RE.match(s) or re.match(r'^[.\u2026]{15,}[^.]{0,80}?J\.$', s):
            return '\n'.join(lines[:i]).strip(), '\n'.join(lines[i:]).strip()
    return last_chunk, ""


def parse_footer(footer_raw: str) -> dict:
    """Extract judges, judgment date, and location from the footer signature block."""
    result: dict = {"judges": [], "judgment_date": "", "judgment_location": ""}
    if not footer_raw:
        return result

    lines = footer_raw.split('\n')

    # Pass 1: sig line → next non-blank line is the judge name.
    # Strip bold markers (**) before matching since pymupdf4llm wraps lines in **.
    i = 0
    while i < len(lines):
        line = re.sub(r'\*+', '', lines[i]).strip()
        if _FOOTER_SIG_RE.match(line):
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines):
                name_line = re.sub(r'\*+', '', lines[j]).strip()
                m = _FOOTER_JUDGE_NAME_RE.match(name_line)
                if m:
                    name = m.group(1).strip()
                    if name and name not in result["judges"]:
                        result["judges"].append(name)
                    i = j
        i += 1

    # Pass 2: fallback — any bracketed/parenthesised name after stripping bold.
    if not result["judges"]:
        _fallback = re.compile(r'^[\[(]([A-Za-z][^\]\)\n]{2,60})[\])]\s*$')
        for line in lines:
            clean = re.sub(r'\*+', '', line).strip()
            m = _fallback.match(clean)
            if m:
                name = m.group(1).strip()
                if name and name not in result["judges"]:
                    result["judges"].append(name)

    clean_footer = re.sub(r'\*+', '', footer_raw)
    m = _JUDGMENT_DATE_RE.search(clean_footer)
    if m:
        result["judgment_date"] = m.group(0).strip().rstrip('.')

    m = _JUDGMENT_LOCATION_RE.search(clean_footer)
    if m:
        result["judgment_location"] = m.group(0).strip().title()

    return result


def _split_upper_meta(header_raw: str) -> tuple[str, str]:
    """Split header_raw at the JUDGEMENT + judge-name boundary.

    Returns (upper_meta, post_judgement) where upper_meta ends with the
    authoring judge's name and post_judgement is the body text that precedes
    the first numbered paragraph (becomes implicit para 1 when present).
    """
    m = _UPPER_META_END_RE.search(header_raw)
    if m:
        return header_raw[:m.end()].strip(), header_raw[m.end():].strip()
    return header_raw, ""


def split_paragraphs(text: str):
    # Normalise spaced-letter headings that some PDF extractors produce
    text = re.sub(r'\bJ\s+U\s+D\s+G\s+(?:E\s+)?M\s+E\s+N\s+T\b', 'JUDGMENT', text, flags=re.IGNORECASE)
    text = re.sub(r'\bO\s+R\s+D\s+E\s+R\b', 'ORDER', text, flags=re.IGNORECASE)

    BOLD  = re.compile(r"(?=^\*\*\d+\.\*\*)", re.MULTILINE)
    PLAIN = re.compile(r"(?=^_?\d+\._?\s+\S)", re.MULTILINE)

    bold_first  = BOLD.search(text)
    plain_first = PLAIN.search(text)
    if bold_first is None and plain_first is None:
        return text.strip(), [], ""

    pattern = BOLD if bold_first else PLAIN
    first   = bold_first if bold_first else plain_first

    full_header = text[: first.start()].strip()
    upper_meta_raw, post_judgement = _split_upper_meta(full_header)
    header_raw = upper_meta_raw   # alias used by the rest of this function
    body_text  = text[first.start():]

    # Pass 1 — detect section headings (must be surrounded by blank lines)
    blocks = re.split(r'(\n\n+)', body_text)
    sections: list[tuple[str, str]] = []
    current_name: str = ""
    current_parts: list[str] = []
    for block in blocks:
        stripped = block.strip()
        if stripped and _is_section_heading(stripped):
            sections.append((current_name, ''.join(current_parts)))
            current_name = re.sub(r'^#{1,4}\s*|\*{1,2}', '', stripped).strip()
            current_parts = []
        else:
            current_parts.append(block)
    sections.append((current_name, ''.join(current_parts)))
    sections = [(n, t) for n, t in sections if t.strip()]

    # Pass 2 — split each section on numbered paragraphs, build chunk dicts
    _PARA_START = re.compile(r"^\*\*\d+\.\*\*|^_?\d+\._?\s")
    all_chunks: list[dict] = []
    for sec_name, sec_text in sections:
        raw_strings = [c.strip() for c in pattern.split(sec_text) if c.strip()]
        raw_strings = _enforce_sequential(raw_strings)

        # If the first raw string has no paragraph number, extract its body
        # content and prepend it into the first numbered paragraph's body.
        # This avoids emitting a para_no="?" chunk.
        leading_body = ""
        if raw_strings and not _PARA_START.match(raw_strings[0]):
            if len(raw_strings) > 1:
                leading_body = para_body(raw_strings[0])
                raw_strings = raw_strings[1:]
            else:
                raw_strings = []

        sec_dicts = []
        for j, rs in enumerate(raw_strings):
            body = para_body(rs)
            if j == 0 and leading_body:
                body = leading_body + "\n\n" + body
            sec_dicts.append({
                "para_no": para_number(rs),
                "section": sec_name,
                "body":    body,
                "is_section_start": False,
            })
        if sec_dicts:
            sec_dicts[0]["is_section_start"] = True
        all_chunks.extend(sec_dicts)
        _log_sequence_gaps(sec_dicts)

    # Footer extraction operates on the last chunk's body string
    footer_raw = ""
    if all_chunks:
        body, footer_raw = extract_footer(all_chunks[-1]["body"])
        all_chunks[-1]["body"] = body
        if not all_chunks[-1]["body"]:
            all_chunks.pop()
        footer_raw = strip_page_numbers(footer_raw)

    all_chunks = _detect_implicit_para_1(post_judgement, all_chunks)

    return upper_meta_raw, all_chunks, footer_raw


def _detect_implicit_para_1(post_judgement: str, chunks: list) -> list:
    """Synthesise para 1 from text that sits between the JUDGEMENT heading and the
    first numbered paragraph.  post_judgement is already the text after the
    upper-metadata boundary, so no further judge-line search is needed."""
    if not chunks or chunks[0]["para_no"] != "2":
        return chunks
    if not post_judgement:
        return chunks
    implicit_body = strip_page_numbers(_DIGITAL_SIG_RE.sub('', post_judgement))
    if not implicit_body:
        return chunks
    synthetic = {
        "para_no": "1",
        "section": chunks[0]["section"],
        "body":    implicit_body,
        "is_section_start": True,
    }
    chunks[0]["is_section_start"] = False
    return [synthetic] + chunks


# Pages with fewer characters than this are treated as scanned images and OCR'd.
_OCR_CHAR_THRESHOLD = 50


def extract_markdown(pdf_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    try:
        doc = fitz.open(tmp_path)

        # Detect which pages have a real text layer vs. scanned images.
        char_counts = [len(doc[i].get_text().strip()) for i in range(len(doc))]
        scanned = {i for i, n in enumerate(char_counts) if n < _OCR_CHAR_THRESHOLD}

        if not scanned:
            # Fast path: all pages have text — pymupdf4llm handles everything.
            doc.close()
            md = pymupdf4llm.to_markdown(tmp_path)
        else:
            # Mixed PDF: get per-page markdown from pymupdf4llm, then replace
            # scanned pages with Tesseract output.
            page_chunks = pymupdf4llm.to_markdown(tmp_path, page_chunks=True)
            parts = []
            for i, chunk in enumerate(page_chunks):
                if i in scanned:
                    pix = doc[i].get_pixmap(dpi=300)
                    img = PIL.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    parts.append(pytesseract.image_to_string(img, lang="eng"))
                else:
                    parts.append(chunk["text"])
            doc.close()
            md = "\n\n".join(parts)

        md = re.sub(r'<br\s*/?>', ' ', md, flags=re.IGNORECASE)
        return md
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def parse_metadata(raw: str) -> dict:
    meta = {
        "citation": "",
        "reportable_status": "",
        "court": "",
        "jurisdiction": "",
        "case_type": "",
        "case_number": "",
        "case_year": "",
        "petitioner_or_appellant": "",
        "respondent": "",
        "authoring_judge": "",
        "case_title": "",
        "is_batch_matter": False,
        "judge": "",
        "judges": [],
        "judgment_date": "",
        "judgment_location": "",
        "toc_md": "",
    }

    normalised = re.sub(r'\bW\s+I\s+T\s+H\b', 'WITH', raw, flags=re.IGNORECASE)
    if re.search(r'\bWITH\b', normalised, re.IGNORECASE):
        meta["is_batch_matter"] = True
    header = re.split(r'\bWITH\b', normalised, maxsplit=1, flags=re.IGNORECASE)[0]

    m = re.search(r'(\d{4}\s+INSC\s+\d+)', header)
    if m:
        meta["citation"] = m.group(1).strip()

    if re.search(r'NON[-\s]?REPORTABLE', raw, re.IGNORECASE):
        meta["reportable_status"] = "NON-REPORTABLE"
    elif re.search(r'REPORTABLE', raw, re.IGNORECASE):
        meta["reportable_status"] = "REPORTABLE"

    if re.search(r'IN\s+THE\s+SUPREME\s+COURT\s+OF\s+INDIA', header, re.IGNORECASE):
        meta["court"] = "Supreme Court of India"

    m = re.search(
        r'((?:CRIMINAL|CIVIL|ORIGINAL)\s+(?:ORIGINAL|APPELLATE|MISC(?:ELLANEOUS)?)'
        r'(?:\s+JURISDICTION)?)',
        header, re.IGNORECASE,
    )
    if m:
        raw_jur = re.sub(r'\s+', ' ', m.group(0)).strip()
        if not raw_jur.upper().endswith('JURISDICTION'):
            raw_jur += ' Jurisdiction'
        meta["jurisdiction"] = raw_jur.title()

    _CASE_PAT = re.compile(
        r'(WRIT\s+PETITION(?:\s*\([^)]+\))?'
        r'|CRIMINAL\s+APPEAL(?:\s*\([^)]+\))?'
        r'|CIVIL\s+APPEAL(?:\s*\([^)]+\))?'
        r'|SPECIAL\s+LEAVE\s+PETITION(?:\s*\([^)]+\))?'
        r'|SLP(?:\s*\([^)]+\))?'
        r'|TRANSFER\s+PETITION(?:\s*\([^)]+\))?'
        r'|CONTEMPT\s+PETITION(?:\s*\([^)]+\))?'
        r'|REVIEW\s+PETITION(?:\s*\([^)]+\))?'
        r'|CURATIVE\s+PETITION(?:\s*\([^)]+\))?'
        r'|ORIGINAL\s+PETITION(?:\s*\([^)]+\))?)'
        r'\s+(?:NO|NOS)\.?\s*([\d,\s&/-]+?)\s+OF\s+(\d{4})',
        re.IGNORECASE,
    )
    m = _CASE_PAT.search(header)
    if m:
        case_type_raw = re.sub(r'\s+', ' ', m.group(1)).strip()
        meta["case_type"] = case_type_raw.title()
        meta["case_number"] = re.sub(r'\s+', '', m.group(2).strip())  # full range e.g. "4872-4873"
        meta["case_year"] = m.group(3).strip()

    # Separator pattern covering ASCII dots and Unicode ellipsis (…, U+2026)
    _ELLIPSIS_SEP = r'\s*[.\u2026]+\s*'

    def _clean_party(raw_name: str) -> str:
        s = re.sub(r'\*+', '', raw_name)
        s = re.sub(r'[.\u2026]{2,}', '', s)    # strip dot/ellipsis runs
        s = re.sub(
            r'\s*[\(\[]?(?:Appellant|Petitioner|Respondent|Opposite Party)'
            r's?\s*[\)\]]?\s*$', '', s, flags=re.IGNORECASE,
        )
        return s.strip().strip('.')

    # Priority 1: bold-formatted names (pymupdf4llm wraps prominent text in **)
    m_pet = re.search(
        r'^\*\*([^*\n]+)\*\*[^\n]*\b(?:APPELLANT|PETITIONER)\b',
        header, re.IGNORECASE | re.MULTILINE,
    )
    m_res = re.search(
        r'^\*\*([^*\n]+)\*\*[^\n]*\bRESPONDENT\b',
        header, re.IGNORECASE | re.MULTILINE,
    )
    if m_pet:
        meta["petitioner_or_appellant"] = _clean_party(m_pet.group(1))
    if m_res:
        meta["respondent"] = _clean_party(m_res.group(1))

    # Priority 2: name ….. Role  (with or without trailing s / (s))
    if not meta["petitioner_or_appellant"]:
        m = re.search(
            r'^([A-Z][^\n\u2026]+?)' + _ELLIPSIS_SEP + r'(?:Appellant|Petitioner)s?\b',
            header, re.IGNORECASE | re.MULTILINE,
        )
        if m:
            meta["petitioner_or_appellant"] = _clean_party(m.group(1))
    if not meta["respondent"]:
        m = re.search(
            r'^([A-Z][^\n\u2026]+?)' + _ELLIPSIS_SEP + r'Respondents?\b',
            header, re.IGNORECASE | re.MULTILINE,
        )
        if m:
            meta["respondent"] = _clean_party(m.group(1))

    # Priority 3: VERSUS inline on the same line
    if not meta["petitioner_or_appellant"] and not meta["respondent"]:
        m_vs = re.search(
            r'^(.+?)\s+(?:VERSUS|vs?\.)\s+(.+?)$',
            header, re.IGNORECASE | re.MULTILINE,
        )
        if m_vs:
            meta["petitioner_or_appellant"] = _clean_party(m_vs.group(1))
            meta["respondent"] = _clean_party(m_vs.group(2))

    # Priority 4: VERSUS on its own line
    if not meta["petitioner_or_appellant"] and not meta["respondent"]:
        m_vs = re.search(
            r'^([^\n]+)\n\s*(?:VERSUS|vs?\.)\s*\n\s*([^\n]+)',
            header, re.IGNORECASE | re.MULTILINE,
        )
        if m_vs:
            meta["petitioner_or_appellant"] = _clean_party(m_vs.group(1))
            meta["respondent"] = _clean_party(m_vs.group(2))

    if meta["petitioner_or_appellant"] and meta["respondent"]:
        meta["case_title"] = f"{meta['petitioner_or_appellant']} vs {meta['respondent']}"

    m = re.search(r'#\s+\*\*([\w\s.,]+J\.)\*\*|^([\w\s.,]+J\.):?\s*$', raw, re.MULTILINE)
    if m:
        meta["judge"] = (m.group(1) or m.group(2)).strip()

    m = _AUTHORING_JUDGE_RE.search(raw)
    if m:
        meta["authoring_judge"] = re.sub(r'\*+', '', m.group(1)).strip()

    m = re.search(r'(##\s+\*\*INDEX\*\*.*)', raw, re.DOTALL)
    if m:
        meta["toc_md"] = m.group(1).strip()

    return meta


_META_LLM_SYSTEM = (
    "You are a legal metadata extractor for Indian court judgments. "
    "Extract the following fields from the document header text provided.\n\n"
    "Return ONLY valid JSON, no markdown fences, no explanation:\n"
    "{\n"
    '  "citation": "e.g. 2024 INSC 123 or empty string",\n'
    '  "reportable_status": "REPORTABLE or NON-REPORTABLE or empty string",\n'
    '  "court": "full court name or empty string",\n'
    '  "jurisdiction": "e.g. Criminal Appellate Jurisdiction or empty string",\n'
    '  "case_type": "e.g. Criminal Appeal or empty string",\n'
    '  "case_number": "e.g. 4872-4873 or empty string",\n'
    '  "case_year": "4-digit year or empty string",\n'
    '  "petitioner_or_appellant": "name only, no role label, or empty string",\n'
    '  "respondent": "name only, no role label, or empty string",\n'
    '  "authoring_judge": "name with J./CJI. suffix or empty string",\n'
    '  "case_title": "Petitioner vs Respondent or empty string",\n'
    '  "is_batch_matter": false,\n'
    '  "judge": "primary judge name or empty string",\n'
    '  "judges": [],\n'
    '  "judgment_date": "e.g. JANUARY 15, 2024 or empty string",\n'
    '  "judgment_location": "e.g. New Delhi or empty string",\n'
    '  "toc_md": ""\n'
    "}"
)

_META_EMPTY: dict = {
    "citation": "", "reportable_status": "", "court": "", "jurisdiction": "",
    "case_type": "", "case_number": "", "case_year": "",
    "petitioner_or_appellant": "", "respondent": "", "authoring_judge": "",
    "case_title": "", "is_batch_matter": False, "judge": "", "judges": [],
    "judgment_date": "", "judgment_location": "", "toc_md": "",
}
# SystemMessage (not tuple) prevents template parsing of the JSON schema inside.
_META_PROMPT = ChatPromptTemplate.from_messages([
    SystemMessage(content=_META_LLM_SYSTEM),
    ("human", "{text}"),
])


def parse_metadata_llm(raw: str) -> dict:
    chain = _META_PROMPT | get_chunk_llm() | StrOutputParser() | _STRIP_THINK | _STRIP_FENCES
    try:
        data = json.loads(chain.invoke({"text": raw[:3_000]}))
        result = {**_META_EMPTY}
        for key in _META_EMPTY:
            if key not in data:
                continue
            if key == "is_batch_matter":
                result[key] = bool(data[key])
            elif key == "judges":
                result[key] = [str(j) for j in data[key]] if isinstance(data[key], list) else []
            elif data[key] is None:
                pass  # keep _META_EMPTY default; _merge_metadata will fall back to regex
            else:
                result[key] = str(data[key])
        return result
    except Exception as exc:
        logging.warning("parse_metadata_llm failed: %s", exc)
        return {**_META_EMPTY}


def _merge_metadata(llm: dict, regex: dict) -> dict:
    """LLM is primary; regex fills any field the LLM left as None or empty."""
    result = {}
    for key in _META_EMPTY:
        llm_val = llm.get(key)
        regex_val = regex.get(key)
        if llm_val is None:
            result[key] = regex_val if regex_val is not None else _META_EMPTY[key]
        else:
            result[key] = llm_val or regex_val or _META_EMPTY[key]
    return result


# ── NER ────────────────────────────────────────────────────────────────────────

def _extract_legal_spans(text: str) -> list[tuple[str, str]]:
    nlp = get_nlp()
    if nlp is None:
        return []
    clean = re.sub(r'\*+', '', text).strip()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        doc = nlp(clean)
    seen: dict[str, str] = {}
    for ent in doc.ents:
        if ent.label_ in _LEGAL_LABELS and ent.text.strip() not in seen:
            seen[ent.text.strip()] = ent.label_
    return sorted(seen.items(), key=lambda t: len(t[0]), reverse=True)


# ── Classification ─────────────────────────────────────────────────────────────

def _resolve_role(raw_role: str) -> str:
    if not raw_role:
        return "Not classified"
    if raw_role in _VALID_ROLES:
        return raw_role
    if raw_role in _LABEL_MAP:
        return _LABEL_MAP[raw_role]
    bracketed = f"[{raw_role}]"
    if bracketed in _LABEL_MAP:
        return _LABEL_MAP[bracketed]
    rl = raw_role.lower()
    return next(
        (v for v in _VALID_ROLES if v.lower() in rl or rl in v.lower()),
        "Not classified",
    )


def _context_snippets(chunks: list, idx: int, window: int = 60) -> dict[str, str]:
    def _snippet(c: dict) -> str:
        words = re.sub(r'\*+', '', c["body"]).split()
        return " ".join(words[:window])

    prev = _snippet(chunks[idx - 1]) if idx > 0 else ""
    curr = _snippet(chunks[idx])
    nxt  = _snippet(chunks[idx + 1]) if idx + 1 < len(chunks) else ""
    return {"prev": prev, "curr": curr, "next": nxt}


def _retry_after_delay(exc: Exception) -> float:
    """Return the Retry-After seconds from a 429 exception string, or 0 if not a 429."""
    exc_str = str(exc)
    if "429" not in exc_str:
        return 0.0
    m = re.search(r"[Rr]etry-[Aa]fter['\"]?\s*:\s*['\"]?(\d+)", exc_str)
    return float(m.group(1)) + 2 if m else 30.0


async def _invoke_chain_async(pn: str, invoke_args: dict) -> dict:
    """Call the rhet chain with retry+backoff. Returns {para_no, role, confidence}."""
    chain = await get_rhet_chain_async()
    last_exc: Exception | None = None
    next_delay: float = 0.0
    for attempt in range(_CLASSIFY_RETRIES + 1):
        if next_delay:
            await asyncio.sleep(next_delay)
        try:
            raw = await chain.ainvoke(invoke_args)
            try:
                json_str = re.sub(r'^```[a-z]*\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE)
                obj = json.loads(json_str)
                raw_role = str(obj.get("role", obj.get("label", ""))).strip()
                return {"para_no": pn, "role": _resolve_role(raw_role),
                        "confidence": float(obj.get("confidence", 1.0))}
            except Exception:
                m = (re.search(r'"role"\s*:\s*"([^"]+)"', raw) or
                     re.search(r'"label"\s*:\s*"([^"]+)"', raw) or
                     re.search(r'\[([A-Z_]+)\]', raw))
                role = _resolve_role(
                    m.group(1) if m and '[' not in m.group(0) else
                    f"[{m.group(1)}]" if m else ""
                )
                return {"para_no": pn, "role": role, "confidence": 0.5}
        except Exception as exc:
            last_exc = exc
            ra = _retry_after_delay(exc)
            next_delay = ra if ra else float(2 ** (attempt + 1))
            logging.warning("_invoke_chain_async attempt %d/%d failed for para %s: %s",
                            attempt + 1, _CLASSIFY_RETRIES + 1, pn, exc)
    logging.error("_invoke_chain_async all retries exhausted for para %s: %s", pn, last_exc)
    return {"para_no": pn, "role": "Not classified", "confidence": 0.0}


async def classify_paragraph_async(chunks: list, idx: int) -> dict:
    pn = chunks[idx]["para_no"]
    ctx = _context_snippets(chunks, idx)
    invoke_args = {
        "para_no": pn,
        "prev": ctx["prev"] or "(none)",
        "curr": ctx["curr"],
        "next": ctx["next"] or "(none)",
    }
    return await _invoke_chain_async(pn, invoke_args)


# ── Embeddings & retrieval ─────────────────────────────────────────────────────

def _build_index(chunks: list) -> tuple:
    embedder  = OllamaEmbeddings(model=_RAG_EMBED_MODEL)
    para_keys = [c["para_no"] for c in chunks]
    # Ensure no chunk has an empty body — empty strings cause NaN embeddings from bge-m3
    bodies    = [c["body"].strip() or "empty paragraph" for c in chunks]
    raw_vecs  = embedder.embed_documents(bodies)
    vecs = np.array(raw_vecs, dtype=np.float32)

    # ── Guard: sanitize NaN/Inf values before normalising ─────────────────────
    # Ollama/bge-m3 can return NaN-containing vectors for edge-case inputs
    # (empty strings, overly long tokens). faiss.normalize_L2 on a NaN vector
    # produces more NaN and causes downstream JSON serialisation errors.
    nan_count = int(np.isnan(vecs).any(axis=1).sum())
    inf_count = int(np.isinf(vecs).any(axis=1).sum())
    if nan_count or inf_count:
        logging.warning(
            "_build_index: %d NaN and %d Inf vectors detected from embedder; "
            "replacing with zeros before normalisation.",
            nan_count, inf_count,
        )
    vecs = np.nan_to_num(vecs, nan=0.0, posinf=0.0, neginf=0.0)

    # Replace zero-norm vectors (would produce NaN after L2 normalisation)
    # with a uniform unit vector so those paragraphs remain retrievable.
    norms = np.linalg.norm(vecs, axis=1)
    zero_mask = norms == 0.0
    if zero_mask.any():
        logging.warning(
            "_build_index: %d zero-norm vectors after NaN cleanup; "
            "replacing with uniform unit vectors.",
            int(zero_mask.sum()),
        )
        dim = vecs.shape[1]
        vecs[zero_mask] = 1.0 / np.sqrt(dim)
    # ──────────────────────────────────────────────────────────────────────────

    faiss.normalize_L2(vecs)
    dim   = vecs.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch        = 64
    index.add(vecs)
    return index, para_keys


def _retrieve_chunks(
    question: str,
    chunks: list,
    index_bundle: tuple,
    k: int = 5,
) -> list[tuple[str, str]]:
    faiss_index, para_keys = index_bundle
    embedder = OllamaEmbeddings(model=_RAG_EMBED_MODEL)
    try:
        q_vec = np.array([embedder.embed_query(question)], dtype=np.float32)
    except Exception as exc:
        raise RuntimeError(f"Embedding query failed: {exc}") from exc

    # ── Guard: sanitize query vector before normalisation ──────────────────────
    q_vec = np.nan_to_num(q_vec, nan=0.0, posinf=0.0, neginf=0.0)
    q_norm = np.linalg.norm(q_vec)
    if q_norm == 0.0:
        logging.warning("_retrieve_chunks: query vector is zero-norm; results will be arbitrary.")
        q_vec[0] = 1.0 / np.sqrt(q_vec.shape[1])
    # ──────────────────────────────────────────────────────────────────────────

    faiss.normalize_L2(q_vec)
    _, idxs = faiss_index.search(q_vec, min(k, len(para_keys)))
    body_map = {c["para_no"]: c["body"] for c in chunks}
    return [
        (para_keys[i], body_map[para_keys[i]])
        for i in idxs[0]
        if i >= 0 and para_keys[i] in body_map
    ]


# ── Chat streaming ─────────────────────────────────────────────────────────────

def _history_to_messages(history: list[dict]) -> list:
    """Convert frontend chat history dicts to LangChain message objects."""
    msgs = []
    for h in history:
        if h["role"] == "user":
            msgs.append(HumanMessage(content=h["content"]))
        else:
            msgs.append(AIMessage(content=h["content"]))
    return msgs


async def stream_chat(question: str, chunks: list, history: list[dict], doc_id: str | None = None):
    # Immediately signal the client so it receives a response before any blocking work
    yield f"data: {json.dumps({'token': '', 'status': 'retrieving', 'done': False})}\n\n"

    loop = asyncio.get_running_loop()
    embedder = OllamaEmbeddings(model=_RAG_EMBED_MODEL)

    if doc_id:
        if doc_id not in _embed_events:
            # First caller: create event, kick off index build as a background task
            evt = asyncio.Event()
            _embed_events[doc_id] = evt

            async def _build_and_signal() -> None:
                try:
                    bundle = await loop.run_in_executor(None, _build_index, chunks)
                    _embedding_cache[doc_id] = bundle
                except Exception as _exc:
                    logging.warning("Index build failed for %s: %s", doc_id, _exc)
                finally:
                    evt.set()

            asyncio.create_task(_build_and_signal())
        else:
            evt = _embed_events[doc_id]

        # Overlap index-readiness wait with query embedding
        try:
            _, q_vec_raw = await asyncio.gather(
                asyncio.wait_for(evt.wait(), timeout=60),
                loop.run_in_executor(None, embedder.embed_query, question),
            )
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'token': '[Index build timed out]', 'done': True, 'sources': []})}\n\n"
            return
        except Exception as exc:
            yield f"data: {json.dumps({'token': f'[Retrieval error: {exc}]', 'done': True, 'sources': []})}\n\n"
            return

        index_bundle = _embedding_cache.get(doc_id)
        if index_bundle is None:
            yield f"data: {json.dumps({'token': '[Index unavailable]', 'done': True, 'sources': []})}\n\n"
            return
    else:
        # No doc_id: build index and embed query in parallel (no caching)
        try:
            index_bundle, q_vec_raw = await asyncio.gather(
                loop.run_in_executor(None, _build_index, chunks),
                loop.run_in_executor(None, embedder.embed_query, question),
            )
        except Exception as exc:
            yield f"data: {json.dumps({'token': f'[Index error: {exc}]', 'done': True, 'sources': []})}\n\n"
            return

    # FAISS search with pre-computed query vector (fast, runs synchronously)
    faiss_index, para_keys = index_bundle
    try:
        q_vec = np.array([q_vec_raw], dtype=np.float32)
        q_vec = np.nan_to_num(q_vec, nan=0.0, posinf=0.0, neginf=0.0)
        if np.linalg.norm(q_vec) == 0.0:
            logging.warning("stream_chat: query vector is zero-norm; results will be arbitrary.")
            q_vec[0] = 1.0 / np.sqrt(q_vec.shape[1])
        faiss.normalize_L2(q_vec)
        _, idxs = faiss_index.search(q_vec, min(5, len(para_keys)))
        body_map = {c["para_no"]: c["body"] for c in chunks}
        retrieved = [
            (para_keys[i], body_map[para_keys[i]])
            for i in idxs[0]
            if i >= 0 and para_keys[i] in body_map
        ]
    except Exception as exc:
        yield f"data: {json.dumps({'token': f'[Retrieval error: {exc}]', 'done': True, 'sources': []})}\n\n"
        return

    context = "\n\n".join(f"[Para {pn}] {body}" for pn, body in retrieved)
    sources  = [f"Para {pn}" for pn, _ in retrieved]
    chain    = _get_rag_chain()

    try:
        async for tok in chain.astream({
            "context":  context,
            "history":  _history_to_messages(history),
            "question": question,
        }):
            yield f"data: {json.dumps({'token': tok, 'done': False})}\n\n"
        yield f"data: {json.dumps({'token': '', 'done': True, 'sources': sources})}\n\n"
    except Exception as exc:
        yield f"data: {json.dumps({'token': f'[Chat error: {exc}]', 'done': True, 'sources': []})}\n\n"


# ── Request models ─────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    doc_id: str | None = None
    chunks: list[dict] = []
    indices: list[int] | None = None


class ChatRequest(BaseModel):
    doc_id: str | None = None
    chunks: list[dict] = []
    question: str
    history: list[dict]


def _resolve_chunks(doc_id: str | None, chunks: list) -> list:
    if doc_id and doc_id in _doc_store:
        return _doc_store[doc_id]["chunks"]
    return chunks


class SimilarResult(BaseModel):
    case_number: str
    case_type: str
    court: str
    year: int
    judge: str
    outcome: str
    summary: str
    final_score: float
    rerank_score: float
    rrf_score: float
    semantic_score: float  # cosine similarity from dense_full vector space [0,1]
    precedent_score: float
    provision_score: float
    statute_score: float
    ratio_decidendi: list[str] = Field(default_factory=list)
    shared_precedents_raw: list[str]
    shared_provisions_raw: list[str]
    shared_statutes_raw: list[str]
    provision_roles: dict[str, str] = Field(default_factory=dict)
    precedent_roles: dict[str, str] = Field(default_factory=dict)


class SimilarResponse(BaseModel):
    query_summary: str
    query_ratio_decidendi: list[str]
    query_provisions: list[dict]
    query_precedents: list[dict]
    query_statutes: list[str]
    results: list[SimilarResult]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.post("/extract")
async def extract_endpoint(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    raw_md = extract_markdown(pdf_bytes)
    header_raw, chunks, footer_raw = split_paragraphs(raw_md)

    loop = asyncio.get_running_loop()
    meta_regex = parse_metadata(header_raw)
    meta_llm_fut = loop.run_in_executor(None, parse_metadata_llm, header_raw)

    footer_meta = parse_footer(footer_raw)
    if not any([footer_meta["judges"], footer_meta["judgment_date"], footer_meta["judgment_location"]]):
        footer_meta = parse_footer(raw_md[-3000:])

    meta_regex["judges"] = footer_meta["judges"]
    meta_regex["judgment_date"] = footer_meta["judgment_date"]
    meta_regex["judgment_location"] = footer_meta["judgment_location"]
    if footer_meta["judges"] and not meta_regex["judge"]:
        meta_regex["judge"] = footer_meta["judges"][0]

    meta_llm = await meta_llm_fut
    if not meta_llm["judges"]:
        meta_llm["judges"] = footer_meta["judges"]
    if not meta_llm["judgment_date"]:
        meta_llm["judgment_date"] = footer_meta["judgment_date"]
    if not meta_llm["judgment_location"]:
        meta_llm["judgment_location"] = footer_meta["judgment_location"]

    # LLM is primary; regex fills any field the LLM left empty.
    meta = _merge_metadata(meta_llm, meta_regex)

    doc_id = uuid.uuid4().hex
    _evict_if_needed()
    _doc_store[doc_id] = {
        "chunks": chunks, "header": header_raw,
        "footer": footer_raw,
        "metadata": meta,
        "metadata_regex": meta_regex,
        "metadata_llm": meta_llm,
        "markdown": raw_md,
        "raw_md_tail": raw_md[-3000:],
    }

    def _eager_embed(did: str, ch: list, _loop: asyncio.AbstractEventLoop, _evt: asyncio.Event) -> None:
        if did in _embedding_cache:
            _loop.call_soon_threadsafe(_evt.set)
            return
        try:
            _embedding_cache[did] = _build_index(ch)
        except Exception as exc:
            logging.warning("Eager embed failed for %s: %s", did, exc)
        finally:
            _loop.call_soon_threadsafe(_evt.set)

    _loop = asyncio.get_running_loop()
    _embed_evt = _embed_events.setdefault(doc_id, asyncio.Event())
    threading.Thread(target=_eager_embed, args=(doc_id, chunks, _loop, _embed_evt), daemon=True).start()

    return {
        "doc_id": doc_id,
        "markdown": raw_md,
        "upper_metadata": header_raw,
        "header": header_raw,
        "chunks": chunks,
        "footer": footer_raw,
        "metadata": meta,
        "metadata_regex": meta_regex,
        "metadata_llm": meta_llm,
    }


@app.post("/classify")
async def classify_endpoint(req: ClassifyRequest):
    chunks = _resolve_chunks(req.doc_id, req.chunks)
    indices = req.indices if req.indices is not None else list(range(len(chunks)))

    async def generate():
        yield f"data: {json.dumps({'total': len(indices), 'done': False})}\n\n"
        queue: asyncio.Queue = asyncio.Queue()

        async def _worker(idx: int) -> None:
            try:
                async with _classify_semaphore:
                    result = await classify_paragraph_async(chunks, idx)
            except Exception as exc:
                pn = chunks[idx]["para_no"] if idx < len(chunks) else str(idx)
                result = {"para_no": pn, "role": "Not classified", "confidence": 0.0}
                logging.warning("classify worker failed for idx %s: %s", idx, exc)
            await queue.put(result)

        tasks = [asyncio.create_task(_worker(idx)) for idx in indices]
        for _ in indices:
            result = await queue.get()
            yield f"data: {json.dumps(result)}\n\n"
        await asyncio.gather(*tasks, return_exceptions=True)

    return StreamingResponse(generate(), media_type="text/event-stream")



@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    chunks = _resolve_chunks(req.doc_id, req.chunks)
    return StreamingResponse(
        stream_chat(req.question, chunks, req.history, doc_id=req.doc_id),
        media_type="text/event-stream",
    )


@app.post("/similar")
async def similar_endpoint(
    file: UploadFile = File(...),
    court:     str | None = Query(default=None),
    year_from: int | None = Query(default=None),
    year_to:   int | None = Query(default=None),
    act:       str | None = Query(default=None),
) -> SimilarResponse:
    from qdrant_client import models as qm  # noqa: PLC0415

    pdf_bytes = await file.read()
    loop = asyncio.get_running_loop()

    # 1 — plain-text extraction (fast, no markdown overhead)
    raw_text = await loop.run_in_executor(None, _extract_plain_text, pdf_bytes)

    # 2+3 — LLM extraction and NER run concurrently (independent inputs, both thread-pool-bound)
    extraction, spans = await asyncio.gather(
        loop.run_in_executor(None, _llm_extract, raw_text),
        loop.run_in_executor(None, _extract_legal_spans, raw_text),
    )
    canon  = canonicalize_spans(spans)
    q_prov = set(canon["provisions_canon"])
    q_stat = set(canon["statutes_canon"])
    q_prec = set(canon["precedents_canon"])
    from backend.legal_normalizer import canonicalize_entity  # noqa: PLC0415
    for cp in extraction.provisions:
        q_prov.add(canonicalize_entity(cp.text, "PROVISION"))
    for cs in extraction.statutes:
        q_stat.add(canonicalize_entity(cs, "STATUTE"))
    for cp in extraction.precedents:
        q_prec.add(canonicalize_entity(cp.citation, "PRECEDENT"))

    # 4 — Three embeddings
    embedder   = get_similar_embedder()
    ratio_text = " ".join(extraction.ratio_decidendi) or extraction.summary
    master_text = extraction.summary + " " + " ".join(extraction.ratio_decidendi)
    dense_ratio, dense_facts, dense_full = await asyncio.gather(
        loop.run_in_executor(None, embedder.embed_query, ratio_text),
        loop.run_in_executor(None, embedder.embed_query, extraction.summary),
        loop.run_in_executor(None, embedder.embed_query, master_text),
    )

    # 5 — Build Qdrant filter from optional params
    must: list = []
    if court and court != "Any":
        must.append(qm.FieldCondition(key="court", match=qm.MatchValue(value=court)))
    if year_from or year_to:
        must.append(qm.FieldCondition(
            key="year",
            range=qm.Range(gte=year_from, lte=year_to),
        ))
    if act:
        from backend.legal_normalizer import canonicalize_entity  # noqa: PLC0415
        act_canon = canonicalize_entity(act, "STATUTE")
        must.append(qm.FieldCondition(
            key="statutes_canon", match=qm.MatchValue(value=act_canon)
        ))
    qdrant_filter = qm.Filter(must=must) if must else None

    # 6 — BM25 keywords (for text-filtered dense prefetch)
    keywords = [
        w for w in re.findall(r'\b[A-Za-z]{4,}\b', master_text[:2000])
        if w.lower() not in _BM25_STOPWORDS
    ]
    bm25_query = " ".join(dict.fromkeys(keywords))[:500]

    prefetches = [
        qm.Prefetch(query=dense_ratio, using="dense_ratio", limit=20),
        qm.Prefetch(query=dense_facts, using="dense_facts", limit=20),
        qm.Prefetch(query=dense_full,  using="dense_full",  limit=20),
    ]
    if bm25_query:
        prefetches.append(qm.Prefetch(
            query=dense_full,
            using="dense_full",
            filter=qm.Filter(must=[
                qm.FieldCondition(
                    key="text_for_bm25",
                    match=qm.MatchText(text=bm25_query),
                )
            ]),
            limit=20,
        ))

    # 7 — Hybrid RRF query → top-20 candidates
    try:
        client = get_qdrant()
        rrf_hits = client.query_points(
            collection_name=_QDRANT_COLLECTION,
            prefetch=prefetches,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=20,
            with_payload=True,
            query_filter=qdrant_filter,
        ).points
    except Exception as exc:
        logging.warning("/similar: Qdrant query failed: %s", exc)
        return SimilarResponse(
            query_summary=extraction.summary,
            query_ratio_decidendi=extraction.ratio_decidendi,
            query_provisions=[{"text": p.text, "role": p.role} for p in extraction.provisions],
            query_precedents=[{"citation": p.citation, "short_name": p.short_name, "role": p.role} for p in extraction.precedents],
            query_statutes=extraction.statutes,
            results=[],
        )

    if not rrf_hits:
        return SimilarResponse(
            query_summary=extraction.summary,
            query_ratio_decidendi=extraction.ratio_decidendi,
            query_provisions=[{"text": p.text, "role": p.role} for p in extraction.provisions],
            query_precedents=[{"citation": p.citation, "short_name": p.short_name, "role": p.role} for p in extraction.precedents],
            query_statutes=extraction.statutes,
            results=[],
        )

    # 8 — Normalise RRF scores to [0,1] using local max (ordinal ranking signal).
    max_rrf = max(h.score for h in rrf_hits) or 1.0
    rrf_norm = {h.id: h.score / max_rrf for h in rrf_hits}

    # 8b+9 — Cosine re-query and cross-encoder rerank run concurrently (independent)
    # RRF is an ordinal ranking tool whose scores oscillate across queries because
    # the normalisation denominator changes with every result set.  Cosine from a
    # single vector space gives a stable [0,1] similarity the UI can display as
    # "Semantic".
    hit_ids = [h.id for h in rrf_hits]
    ce = get_cross_encoder()
    pairs = [(master_text, h.payload.get("text_for_bm25", "")) for h in rrf_hits]

    def _cosine_query():
        return client.query_points(
            collection_name=_QDRANT_COLLECTION,
            query=dense_full,
            using="dense_full",
            query_filter=qm.Filter(must=[qm.HasIdCondition(has_id=hit_ids)]),
            limit=len(hit_ids),
            with_payload=False,
        ).points

    cosine_result, ce_scores_raw = await asyncio.gather(
        loop.run_in_executor(None, _cosine_query),
        loop.run_in_executor(None, ce.predict, pairs),
        return_exceptions=True,
    )
    cosine_map = (
        {h.id: float(h.score) for h in cosine_result}
        if not isinstance(cosine_result, Exception)
        else {}
    )
    ce_scores = _sigmoid(np.array(ce_scores_raw)).tolist()

    # 10 — Weighted final score + assemble results
    results: list[SimilarResult] = []
    for hit, rerank_score in zip(rrf_hits, ce_scores):
        p = hit.payload or {}

        doc_prov  = p.get("provisions_canon", [])
        doc_stat  = p.get("statutes_canon",   [])
        doc_prec  = p.get("precedents_canon", [])

        prov_j = _jaccard(q_prov, set(doc_prov))
        stat_j = _jaccard(q_stat, set(doc_stat))
        prec_j = _jaccard(q_prec, set(doc_prec))

        semantic = cosine_map.get(hit.id, rrf_norm[hit.id])

        final = (
            0.35 * rerank_score
            + 0.25 * semantic
            + 0.20 * prec_j
            + 0.12 * prov_j
            + 0.08 * stat_j
        )

        results.append(SimilarResult(
            case_number=p.get("case_number", ""),
            case_type=p.get("case_type", ""),
            court=p.get("court", ""),
            year=int(p.get("year", 0) or 0),
            judge=p.get("judge", ""),
            outcome=p.get("outcome", "unknown"),
            summary=p.get("summary", ""),
            ratio_decidendi=p.get("ratio_decidendi", []),
            final_score=round(float(final), 4),
            rerank_score=round(float(rerank_score), 4),
            rrf_score=round(float(rrf_norm[hit.id]), 4),
            semantic_score=round(semantic, 4),
            precedent_score=round(prec_j, 4),
            provision_score=round(prov_j, 4),
            statute_score=round(stat_j, 4),
            shared_precedents_raw=_shared_raw(q_prec, doc_prec, p.get("precedents_raw", [])),
            shared_provisions_raw=_shared_raw(q_prov, doc_prov, p.get("provisions_raw", [])),
            shared_statutes_raw=_shared_raw(q_stat, doc_stat, p.get("statutes_raw", [])),
            provision_roles=p.get("provision_roles", {}),
            precedent_roles=p.get("precedent_roles", {}),
        ))

    results.sort(key=lambda r: r.final_score, reverse=True)
    return SimilarResponse(
        query_summary=extraction.summary,
        query_ratio_decidendi=extraction.ratio_decidendi,
        query_provisions=[{"text": p.text, "role": p.role} for p in extraction.provisions],
        query_precedents=[{"citation": p.citation, "short_name": p.short_name, "role": p.role} for p in extraction.precedents],
        query_statutes=extraction.statutes,
        results=results[:5],
    )


_NER_MAX_CHARS = 10_000  # guard against OOM from very large inputs


class NerRequest(BaseModel):
    text: str


@app.post("/ner")
async def ner_endpoint(req: NerRequest):
    """Extract legal NER spans. Input is capped at 10,000 characters."""
    text = req.text[:_NER_MAX_CHARS]
    spans = _extract_legal_spans(text)
    return {"spans": [{"text": t, "label": lbl} for t, lbl in spans]}


# Keep a deprecated GET alias for backward compatibility with the current frontend
@app.get("/ner")
async def ner_endpoint_get(text: str):
    """Deprecated: use POST /ner. Kept for backward compatibility."""
    truncated = text[:_NER_MAX_CHARS]
    spans = _extract_legal_spans(truncated)
    return {"spans": [{"text": t, "label": lbl} for t, lbl in spans]}


@app.get("/debug/footer")
async def debug_footer_endpoint(doc_id: str | None = None):
    """Return raw footer extraction data for debugging. Pass ?doc_id=... or omit for latest."""
    if doc_id and doc_id in _doc_store:
        entry = _doc_store[doc_id]
    elif _doc_store:
        entry = next(reversed(_doc_store.values()))
    else:
        return {"error": "no documents extracted yet"}
    return {
        "footer_raw":   entry.get("footer", ""),
        "raw_md_tail":  entry.get("raw_md_tail", ""),
        "metadata":     {k: entry["metadata"].get(k) for k in ("judges", "judgment_date", "judgment_location", "judge")},
    }