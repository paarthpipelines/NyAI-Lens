"""
Batch ingestion of Indian court judgment PDFs into Qdrant Cloud.

Usage:
    python -m backend.ingest --pdf-dir /path/to/pdfs
    python -m backend.ingest --pdf-dir /path/to/pdfs --workers 4
    python -m backend.ingest --pdf-dir /path/to/pdfs --qdrant-path ./local_db  # dev only
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import fitz
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from tqdm import tqdm

from backend.main import (
    CasedPrecedent,
    CasedProvision,
    Extraction,
    _extract_legal_spans,
    parse_metadata,
    extract_markdown,
    split_paragraphs,
    parse_metadata_llm,
    _merge_metadata,
)
from backend.legal_normalizer import canonicalize_entity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

COLLECTION      = "indian_judgements"
_EMBED_DIM      = 1024
# NOTE: ensure this model is pulled in Ollama: `ollama pull bge-m3`
_EMBED_MODEL    = os.getenv("EMBED_MODEL", "bge-m3")
# NOTE: ensure this model is pulled in Ollama, e.g. `ollama pull llama3.3:70b`
_LLM_MODEL      = os.getenv("LLM_MODEL", "deepseek-v3.1:671b-cloud")
_BATCH_SIZE     = 32
# Cloud URL can be overridden via QDRANT_URL env var
_CLOUD_URL      = os.getenv("QDRANT_URL", "https://2299273f-afca-4fbe-a283-8d8e34a4aa20.us-east4-0.gcp.cloud.qdrant.io:6333")
_UPSERT_RETRIES = 3
# Maximum characters sent to the LLM (guards against context-window overflow)
_MAX_LLM_CHARS  = 15_000

_EXTRACTION_SYSTEM = (
    "You are extracting structured metadata from Indian legal judgments.\n\n"
    "Focus on HIGH-QUALITY LEGAL SUMMARIZATION and CLEAN NORMALIZATION.\n"
    "Do NOT over-structure the output. Keep the schema lightweight and retrieval-friendly.\n\n"
    "Return ONLY valid JSON — no markdown fences, no explanation:\n\n"
    "{\n"
    '  "case_title": "<Petitioner v. Respondent — normalize Vs./vs/versus to v.>",\n'
    '  "citation": "<e.g. 2024 INSC 123 — full neutral citation if present, else empty>",\n'
    '  "court": "<full court name>",\n'
    '  "year": <4-digit integer year of decision>,\n'
    '  "summary": "<120-300 word legally dense headnote: core legal issue, procedural posture, '
    'dispute context, key legal principles, precedent relied upon, final holding>",\n'
    '  "ratio_decidendi": ["<concise doctrinal principle actually applied>", "..."],\n'
    '  "outcome": "<one of: allowed | dismissed | partly_allowed | remanded | affirmed | '
    'quashed | set aside | disposed | unknown>",\n'
    '  "provisions": [\n'
    '    {"text": "<Section 302 IPC>", "role": "<decisive|supporting|background|distinguished>"}\n'
    '  ],\n'
    '  "precedents": [\n'
    '    {"case_title": "<Party v. Party>", "citation": "<full reporter citation>",\n'
    '     "role": "<relied_upon|followed|distinguished|overruled|merely_cited>"}\n'
    '  ]\n'
    "}\n\n"
    "INSTRUCTIONS:\n"
    "case_title: Extract proper case title. Normalize Vs./vs/versus to v. "
    "Remove trailing footnote numbers and OCR junk. "
    'Example: "National Insurance Company Limited Vs. Boghara Polyfab Private Limited1" '
    '→ "National Insurance Co. Ltd. v. Boghara Polyfab Pvt. Ltd."\n'
    "summary: 120-300 words. Include core legal issue, procedural posture, dispute context, "
    "important legal principles, key precedent relied upon, and final holding. "
    "Resemble a professional legal headnote. Avoid generic or shallow summaries.\n"
    "ratio_decidendi: 2-6 concise doctrinal principles actually applied. Do not repeat the summary.\n"
    "provisions: every section/article/rule applied or discussed. "
    "ONE provision per list item — never group multiple sections into one string "
    "(wrong: 'Sections 302 and 201 IPC'; right: two separate items). "
    "Always include the Act name with each section.\n"
    "precedents: every case citation. Merge case_title and citation into one object. "
    "citation must be the full reporter string. Normalize case_title: remove footnote markers, "
    "trailing digits, OCR artifacts. Preserve Ltd., Pvt., Co.\n"
    "outcome: use underscore form for multi-word values (partly_allowed, set_aside)."
)


_VALID_OUTCOMES = frozenset({
    "allowed", "dismissed", "remanded", "affirmed",
    "partly allowed", "quashed", "set aside", "disposed", "unknown",
})


def _ws(s: str) -> str:
    """Collapse all whitespace (including PDF-embedded newlines) to single spaces."""
    return re.sub(r'\s+', ' ', s).strip()


_FOOTNOTE_TRAIL_RE = re.compile(r'\s*\d{1,3}\s*$')
_YEAR_FOOTNOTE_RE  = re.compile(r'(\b\d{4})\d{1,4}\b')

def _strip_footnote(s: str) -> str:
    """Remove trailing footnote digit(s) from case titles: 'Party v. Party1' → 'Party v. Party'."""
    return _FOOTNOTE_TRAIL_RE.sub('', s).strip()


def _clean_statute(s: str) -> str:
    """Fix OCR year+footnote artifacts: 'CrPC, 197310' → 'CrPC, 1973'."""
    return _YEAR_FOOTNOTE_RE.sub(r'\1', s).strip()


# Regex supplement for statutes spaCy's NER misses (domain-specific Acts/Codes).
# Requires a year (,YYYY) to avoid false positives from procedural phrases like
# "set aside the Order" or "reliance on Act". "Order"/"Ordinance" excluded —
# too common in CPC procedural references (Order XXXVIII Rule 5 etc.).
_STATUTE_RE = re.compile(
    r'\b([A-Z][A-Za-z ]{3,60}'
    r'(?:Act|Code|Rules|Regulations)'
    r'\s*,\s*\d{4})\b'
)
_STATUTE_AND_RE = re.compile(r'\band\b', re.IGNORECASE)
_STATUTE_MIN_LEN = 10


def _extract_statutes_regex(text: str) -> list[tuple[str, str]]:
    """Regex fallback: extract 'X Act/Code, YYYY' patterns spaCy may miss."""
    seen: set[str] = set()
    spans: list[tuple[str, str]] = []
    for m in _STATUTE_RE.finditer(text):
        raw = _ws(m.group(0))
        # "Act and X Rules, YYYY" — conjunction signals a fragment, not a statute name
        if _STATUTE_AND_RE.search(raw):
            continue
        raw = _clean_statute(raw)
        if len(raw) >= _STATUTE_MIN_LEN and raw not in seen:
            seen.add(raw)
            spans.append((raw, "STATUTE"))
    return spans


def _normalize_outcome(raw: str) -> str:
    """Map LLM-produced outcome strings to the canonical allowed set."""
    val = _ws(raw).lower().replace("_", " ")
    if val in _VALID_OUTCOMES:
        return val
    # Partial match — handles "Appeal dismissed", "partly_allowed", "set aside by court"
    for vo in sorted(_VALID_OUTCOMES, key=len, reverse=True):
        if vo in val:
            return vo
    return "unknown"


def _parse_ingest_response(raw: str) -> tuple[Extraction, dict]:
    """Parse LLM JSON into (Extraction, extras).

    extras carries top-level fields the new schema adds (case_title, citation,
    court, year) without requiring changes to the shared Extraction model in main.py.
    statutes are intentionally omitted — spaCy NER fills them via _merge_entities.
    """
    clean = re.sub(r'^```[a-z]*\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE).strip()
    _empty_extras: dict = {"case_title": "", "citation": "", "court": "", "year": 0}
    try:
        data = json.loads(clean)
        provisions = []
        for p in data.get("provisions", []):
            if isinstance(p, dict) and p.get("text"):
                text = _ws(str(p["text"]))
                if text:
                    provisions.append(CasedProvision(
                        text=text,
                        role=_ws(str(p.get("role", "background"))),
                    ))
        precedents = []
        for p in data.get("precedents", []):
            if isinstance(p, dict) and p.get("citation"):
                citation = _ws(str(p["citation"]))
                if citation:
                    precedents.append(CasedPrecedent(
                        citation=citation,
                        case_title=_strip_footnote(_ws(str(p.get("case_title", citation)))),
                        role=_ws(str(p.get("role", "merely_cited"))),
                    ))
        year_raw = data.get("year", 0)
        try:
            year_int = int(str(year_raw)) if year_raw else 0
        except (ValueError, TypeError):
            year_int = 0
        extras = {
            "case_title": _strip_footnote(_ws(str(data.get("case_title", "")))),
            "citation":   _ws(str(data.get("citation", ""))),
            "court":      _ws(str(data.get("court", ""))),
            "year":       year_int,
        }
        extraction = Extraction(
            summary=_ws(str(data.get("summary", ""))),
            ratio_decidendi=[_ws(str(r)) for r in data.get("ratio_decidendi", []) if str(r).strip()],
            outcome=_normalize_outcome(str(data.get("outcome", "unknown"))),
            provisions=provisions,
            precedents=precedents,
            statutes=[],  # spaCy fills statutes via _merge_entities
        )
        return extraction, extras
    except Exception as exc:
        log.warning("LLM JSON parse failed (response len=%d): %s — using regex fallback", len(raw), exc)
        m_s = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
        m_o = re.search(r'"outcome"\s*:\s*"([^"]+)"', raw)
        return Extraction(
            summary=_ws(m_s.group(1)) if m_s else "",
            ratio_decidendi=[],
            outcome=_normalize_outcome(m_o.group(1)) if m_o else "unknown",
        ), _empty_extras


# ── Collection + index setup ──────────────────────────────────────────────────

def setup_collection(client) -> None:
    from qdrant_client import models as qm  # noqa: PLC0415

    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        log.info("Collection %s already exists — skipping creation", COLLECTION)
        return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense_ratio": qm.VectorParams(size=_EMBED_DIM, distance=qm.Distance.COSINE),
            "dense_facts": qm.VectorParams(size=_EMBED_DIM, distance=qm.Distance.COSINE),
            "dense_full":  qm.VectorParams(size=_EMBED_DIM, distance=qm.Distance.COSINE),
        },
    )
    log.info("Created collection %s", COLLECTION)

    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="text_for_bm25",
        field_schema=qm.TextIndexParams(
            type="text",
            tokenizer=qm.TokenizerType.WORD,
            min_token_len=2,
            max_token_len=20,
            lowercase=True,
        ),
    )
    for field in ("court", "outcome", "text_hash",
                  "provisions_canon", "statutes_canon", "precedents_canon"):
        client.create_payload_index(
            collection_name=COLLECTION,
            field_name=field,
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )
    client.create_payload_index(
        collection_name=COLLECTION,
        field_name="year",
        field_schema=qm.PayloadSchemaType.INTEGER,
    )
    log.info("Created all payload indexes on %s", COLLECTION)


# ── Batch upsert with retry ────────────────────────────────────────────────────

def _upsert_with_retry(client, points: list) -> None:
    """Upsert a batch to Qdrant with exponential-backoff retry on transient errors."""
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse  # noqa: PLC0415

    for attempt in range(_UPSERT_RETRIES + 1):
        try:
            client.upsert(collection_name=COLLECTION, points=points, wait=True)
            return
        except (ResponseHandlingException, UnexpectedResponse) as exc:
            retryable = isinstance(exc, ResponseHandlingException) or (
                isinstance(exc, UnexpectedResponse) and exc.status_code in (429, 503)
            )
            if not retryable or attempt == _UPSERT_RETRIES:
                raise
            delay = 2 ** attempt  # 1s, 2s, 4s
            log.warning(
                "Qdrant upsert attempt %d/%d failed (%s) — retrying in %ds",
                attempt + 1, _UPSERT_RETRIES, exc, delay,
            )
            time.sleep(delay)


# ── Entity merge: LLM (primary) + spaCy (gap-fill) ───────────────────────────

def _merge_entities(
    extraction: Extraction,
    spacy_spans: list[tuple[str, str]],
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, str]]:
    """
    Merge LLM-extracted entities (carry role context spaCy cannot provide) with
    spaCy NER spans (fill entities the LLM missed due to truncation).
    Deduplication is on canonical key so surface variants collapse to one entry.

    Returns:
      (canon_dict, provision_roles, precedent_roles)
      canon_dict keys: provisions_raw/canon, statutes_raw/canon, precedents_raw/canon
      provision_roles: {raw_text -> role}
      precedent_roles: {citation -> role}
    """
    prov: dict[str, str] = {}        # canon -> raw
    prov_roles: dict[str, str] = {}  # raw -> role
    stat: dict[str, str] = {}        # canon -> raw
    prec: dict[str, str] = {}        # canon -> raw
    prec_roles: dict[str, str] = {}  # citation -> role

    # LLM entities are primary — they carry decisive/supporting/relied_upon roles.
    # _ws() already applied in _parse_ingest_response, but guard here too.
    for p in extraction.provisions:
        text = _ws(p.text)
        if not text:
            continue
        c = canonicalize_entity(text, "PROVISION")
        if c not in prov:
            prov[c] = text
            prov_roles[text] = p.role

    # statutes intentionally omitted here — spaCy NER fills the stat dict below.

    for p in extraction.precedents:
        citation = _ws(p.citation)
        if not citation:
            continue
        c = canonicalize_entity(citation, "PRECEDENT")
        if c not in prec:
            prec[c] = citation
            prec_roles[citation] = p.role

    # spaCy fills gaps (entities the LLM missed); default roles are least significant.
    # Newlines from PDF text are stripped before canonicalization.
    for raw, label in spacy_spans:
        raw = _ws(raw)
        if not raw:
            continue
        if label == "PROVISION":
            # Skip compound multi-section spans — "Sections 302 and 201 IPC" or
            # "Sections 395, 363, 365" canonicalize to useless slugs that never
            # match individual act-qualified canons during Jaccard scoring.
            if re.match(r'^sections?\s+\d', raw, re.I) and re.search(r'[,&]|\band\b', raw, re.I):
                continue
            c = canonicalize_entity(raw, "PROVISION")
            if c not in prov:
                prov[c] = raw
                prov_roles[raw] = "background"
        elif label == "STATUTE":
            raw = _clean_statute(raw)
            c = canonicalize_entity(raw, "STATUTE")
            if c not in stat:
                stat[c] = raw
        elif label == "PRECEDENT":
            raw = _strip_footnote(raw)
            c = canonicalize_entity(raw, "PRECEDENT")
            if c not in prec:
                prec[c] = raw
                prec_roles[raw] = "merely_cited"

    # Remove bare provision slugs subsumed by an act-qualified canonical.
    # Handles two bare-canon forms produced by _canon_provision when the Act is missing:
    #   "_401"       (very short bare slug)
    #   "section_27" (slugified "Section 27" with no Act)
    # Both are redundant when "crpc_401" / "ea_27" already exists.
    _BARE_PROV_RE = re.compile(r'^(?:sections?|articles?|rules?)_(.+)$')
    qualified = {
        c for c in prov
        if not c.startswith("_") and not _BARE_PROV_RE.match(c)
    }
    for c in list(prov):
        suffix = None
        if c.startswith("_"):
            suffix = c.lstrip("_")
        else:
            m = _BARE_PROV_RE.match(c)
            if m:
                suffix = m.group(1)
        if suffix and any(q.endswith(f"_{suffix}") for q in qualified):
            raw_text = prov[c]
            prov_roles.pop(raw_text, None)
            del prov[c]

    canon_dict: dict[str, list[str]] = {
        "provisions_raw":   list(prov.values()),
        "provisions_canon": list(prov.keys()),
        "statutes_raw":     list(stat.values()),
        "statutes_canon":   list(stat.keys()),
        "precedents_raw":   list(prec.values()),
        "precedents_canon": list(prec.keys()),
    }
    return canon_dict, prov_roles, prec_roles


# ── Per-PDF pipeline (synchronous — runs inside ThreadPoolExecutor) ───────────

def _extract_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    pages = [doc[i].get_text() for i in range(len(doc))]
    doc.close()
    return "\n".join(pages)


def _process_pdf_sync(
    pdf_path: Path,
    llm_model,
    embedder: OllamaEmbeddings,
    client,
) -> tuple[str, object]:
    """
    Returns ("ok", PointStruct) for a new document or ("skip", None) for a dedup hit.
    Raises on unrecoverable error — caller logs and continues.
    """
    from qdrant_client import models as qm  # noqa: PLC0415

    raw_text  = _extract_text(pdf_path)
    text_hash = hashlib.sha256(raw_text.encode()).hexdigest()  # hash full text for dedup
    truncated = raw_text[:_MAX_LLM_CHARS]                      # LLM only sees first 15 K chars

    # Stable deterministic ID from file path — re-runs upsert over the same point
    # rather than inserting duplicates with fresh random UUIDs (BUG 2 fix)
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, str(pdf_path.resolve())))

    # Dedup by content hash (catches identical text at different paths)
    existing, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=qm.Filter(
            must=[qm.FieldCondition(
                key="text_hash",
                match=qm.MatchValue(value=text_hash),
            )]
        ),
        limit=1,
    )
    if existing:
        return "skip", None

    resp = llm_model.invoke([
        SystemMessage(content=_EXTRACTION_SYSTEM),
        HumanMessage(content=f"Judgment text:\n\n{truncated}"),
    ])
    extraction, llm_extras = _parse_ingest_response(resp.content)

    spacy_spans  = _extract_legal_spans(truncated)
    regex_spans  = _extract_statutes_regex(truncated)
    # Combine: spaCy primary, regex fills statute gaps (dedup happens inside _merge_entities)
    all_spans    = spacy_spans + regex_spans
    canon, prov_roles, prec_roles = _merge_entities(extraction, all_spans)

    # ── Metadata: markdown → header slice → regex + LLM → merge ──────────────
    # parse_metadata's Priority-1 party-name patterns use markdown bold markers
    # (**Name**) that fitz plain-text never produces. Using extract_markdown +
    # split_paragraphs gives the same header_raw that /extract uses, so the
    # regex and LLM metadata pipelines match exactly.
    pdf_bytes  = pdf_path.read_bytes()
    raw_md     = extract_markdown(pdf_bytes)
    header_raw, _, _ = split_paragraphs(raw_md)
    meta_regex = parse_metadata(header_raw)
    meta_llm   = parse_metadata_llm(header_raw)
    meta       = _merge_metadata(meta_llm, meta_regex)

    # LLM extraction is fallback for metadata fields parse_metadata may miss.
    court = meta.get("court", "") or llm_extras.get("court", "")

    year_raw = meta.get("case_year", "") or ""
    try:
        m = re.search(r'\d{4}', str(year_raw))
        year = int(m.group(0)) if m else (llm_extras.get("year") or 0)
    except Exception:
        year = llm_extras.get("year") or 0

    # case_title: LLM extraction primary (cleaned, normalized by the prompt).
    # citation: header metadata primary (more reliable for neutral citations),
    #           LLM extraction as fallback.
    case_title = llm_extras.get("case_title", "")
    citation   = meta.get("citation", "") or llm_extras.get("citation", "")

    ratio_text  = " ".join(extraction.ratio_decidendi) or extraction.summary
    master_text = extraction.summary + " " + " ".join(extraction.ratio_decidendi)

    # Sequential embed calls (~50ms each) — nested ThreadPoolExecutor inside an
    # already-pooled thread causes starvation at workers >= 4 (BUG 4 fix)
    dense_ratio = embedder.embed_query(ratio_text)
    dense_facts = embedder.embed_query(extraction.summary)
    dense_full  = embedder.embed_query(master_text)

    point = qm.PointStruct(
        id=point_id,
        vector={
            "dense_ratio": dense_ratio,
            "dense_facts": dense_facts,
            "dense_full":  dense_full,
        },
        payload={
            "text_for_bm25":    master_text,
            "text_hash":        text_hash,
            "file_path":        str(pdf_path),
            "case_title":       case_title,
            "citation":         citation,
            "summary":          extraction.summary,
            "ratio_decidendi":  extraction.ratio_decidendi,
            "outcome":          extraction.outcome,
            "court":            court,
            "year":             year,
            "case_number":      meta.get("case_number", ""),
            "case_type":        meta.get("case_type", ""),
            "judge":            meta.get("judge", ""),
            "provisions_raw":   canon["provisions_raw"],
            "provisions_canon": canon["provisions_canon"],
            "statutes_raw":     canon["statutes_raw"],
            "statutes_canon":   canon["statutes_canon"],
            "precedents_raw":   canon["precedents_raw"],
            "precedents_canon": canon["precedents_canon"],
            "provision_roles":  prov_roles,
            "precedent_roles":  prec_roles,
        },
    )
    return "ok", point


# ── Async orchestration ────────────────────────────────────────────────────────

async def _run_all(
    pdf_dir: Path,
    qdrant_path: str | None,
    workers: int,
    limit: int | None = None,
) -> None:
    from qdrant_client import QdrantClient  # noqa: PLC0415

    if qdrant_path:
        client = QdrantClient(path=qdrant_path)
        log.info("Using local Qdrant storage at %s", qdrant_path)
    else:
        cloud_url = os.getenv("QDRANT_URL", _CLOUD_URL)
        api_key   = os.environ.get("QDRANT_API_KEY")
        is_local  = cloud_url.startswith(("http://localhost", "http://127."))
        if not api_key and not is_local:
            log.error(
                "QDRANT_API_KEY not set — cannot connect to cloud Qdrant. "
                "Check backend/.env contains QDRANT_API_KEY = \"...\"."
            )
            sys.exit(1)
        client = QdrantClient(url=cloud_url, api_key=api_key or None, timeout=60)
        log.info("Using Qdrant at %s", cloud_url)

    # format="json" instructs Ollama to constrain output to valid JSON.
    # _parse_ingest_response handles the parsing — no SYSTEM_PROMPT or
    # _parse_extraction imported from main.py.
    llm_model = ChatOllama(model=_LLM_MODEL, temperature=0, format="json")
    embedder  = OllamaEmbeddings(model=_EMBED_MODEL)

    probe  = embedder.embed_query("test")
    actual = len(probe)
    if actual != _EMBED_DIM:
        log.error(
            "Embed dim mismatch: got %d, expected %d — update _EMBED_DIM in ingest.py",
            actual, _EMBED_DIM,
        )
        sys.exit(1)
    log.info("Embed dim confirmed: %d", actual)

    setup_collection(client)

    # Flat directory — judgement PDFs are not nested (BUG 7 fix)
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        log.warning("No PDFs found in %s", pdf_dir)
        return
    if limit:
        pdf_files = pdf_files[:limit]
        log.info("Demo mode: processing first %d of available PDFs", limit)
    log.info("Found %d PDFs to process", len(pdf_files))

    sem        = asyncio.Semaphore(workers)
    executor   = ThreadPoolExecutor(max_workers=workers)
    loop       = asyncio.get_running_loop()   # get_event_loop() is deprecated in Python ≥ 3.10
    batch_lock = asyncio.Lock()               # guards batch list against concurrent flush races
    error_log  = Path(__file__).parent / "ingest_errors.log"
    counts     = {"ok": 0, "skip": 0, "error": 0}
    batch: list = []

    pbar = tqdm(total=len(pdf_files), desc="Ingesting", unit="pdf")

    async def _one(path: Path) -> None:
        async with sem:
            try:
                status, point = await loop.run_in_executor(
                    executor, _process_pdf_sync, path, llm_model, embedder, client
                )
                counts[status] += 1
                if point is not None:
                    # Use lock so check→append→flush is atomic across concurrent tasks
                    async with batch_lock:
                        batch.append(point)
                        if len(batch) >= _BATCH_SIZE:
                            points_copy = batch[:]
                            batch.clear()
                        else:
                            points_copy = []
                    if points_copy:
                        await loop.run_in_executor(
                            executor, _upsert_with_retry, client, points_copy
                        )
                        log.info("Flushed batch of %d points to Qdrant", len(points_copy))
            except Exception as exc:
                counts["error"] += 1
                log.error("Failed %s: %s", path.name, exc)
                with open(error_log, "a") as ef:
                    ef.write(f"{path}\t{exc}\n")
            finally:
                pbar.update(1)

    await asyncio.gather(*[_one(p) for p in pdf_files])
    pbar.close()

    # Flush remaining points after all PDFs are processed
    if batch:
        points_final = batch[:]
        batch.clear()
        try:
            await loop.run_in_executor(executor, _upsert_with_retry, client, points_final)
            log.info("Flushed final batch of %d points to Qdrant", len(points_final))
        except Exception as exc:
            log.error("Final batch upsert failed: %s", exc)
            with open(error_log, "a") as ef:
                ef.write(f"[FINAL BATCH ERROR] {exc}\n")

    executor.shutdown(wait=True)  # wait=True ensures all in-flight Qdrant writes complete

    log.info(
        "Ingestion complete — ingested: %d, skipped (dedup): %d, errors: %d",
        counts["ok"], counts["skip"], counts["error"],
    )
    if counts["error"]:
        log.info("Error details written to %s", error_log)


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-ingest Indian court judgment PDFs into Qdrant."
    )
    parser.add_argument(
        "--pdf-dir", required=True, type=Path,
        help="Directory containing PDF files (flat *.pdf — not searched recursively)",
    )
    parser.add_argument(
        "--qdrant-path", default=os.getenv("QDRANT_PATH"),
        help="Use embedded local Qdrant at this path (dev only; skips cloud auth)",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Max concurrent LLM+embed workers (default: 4)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N PDFs (demo/test mode)",
    )
    args = parser.parse_args()

    if not args.pdf_dir.is_dir():
        print(f"Error: {args.pdf_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    asyncio.run(_run_all(args.pdf_dir, args.qdrant_path, args.workers, args.limit))


if __name__ == "__main__":
    main()
