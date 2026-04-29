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
_LLM_MODEL      = os.getenv("LLM_MODEL", "llama3.1:8b")
_BATCH_SIZE     = 32
# Cloud URL can be overridden via QDRANT_URL env var
_CLOUD_URL      = os.getenv("QDRANT_URL", "https://2299273f-afca-4fbe-a283-8d8e34a4aa20.us-east4-0.gcp.cloud.qdrant.io:6333")
_UPSERT_RETRIES = 3
# Maximum characters sent to the LLM (guards against context-window overflow)
_MAX_LLM_CHARS  = 15_000

# Standalone extraction prompt — does NOT import SYSTEM_PROMPT from main.py.
# Uses with_structured_output(Extraction) so no manual JSON parsing is needed.
_EXTRACTION_SYSTEM = (
    "You are a senior Indian legal analyst. Extract structured information from the court "
    "judgment text and respond with ONLY valid JSON — no markdown fences, no explanation.\n\n"
    "Required JSON schema:\n"
    "{\n"
    '  "summary": "<3-5 sentences: parties, core legal dispute, procedural history, decision>",\n'
    '  "ratio_decidendi": ["<legal principle with precedential weight>", "..."],\n'
    '  "outcome": "<one of: allowed | dismissed | remanded | affirmed | partly allowed | quashed | set aside | unknown>",\n'
    '  "provisions": [\n'
    '    {"text": "<Section 302 IPC>", "role": "<decisive|supporting|background|distinguished>"}\n'
    '  ],\n'
    '  "precedents": [\n'
    '    {"citation": "<full citation>", "short_name": "<Party v Party>",\n'
    '     "role": "<relied_upon|followed|distinguished|overruled|merely_cited>"}\n'
    '  ],\n'
    '  "statutes": ["<Indian Penal Code, 1860>", "..."]\n'
    "}\n\n"
    "ratio_decidendi: 2-6 principles, each a complete sentence citing specific sections/articles.\n"
    "provisions: every section/article/rule applied or discussed.\n"
    "precedents: every case citation; citation must be the full reporter string.\n"
    "statutes: every Act or Code mentioned by name."
)


_VALID_OUTCOMES = frozenset({
    "allowed", "dismissed", "remanded", "affirmed",
    "partly allowed", "quashed", "set aside", "unknown",
})


def _ws(s: str) -> str:
    """Collapse all whitespace (including PDF-embedded newlines) to single spaces."""
    return re.sub(r'\s+', ' ', s).strip()


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


def _parse_ingest_response(raw: str) -> Extraction:
    """Parse LLM JSON response into Extraction. No import of _parse_extraction from main."""
    clean = re.sub(r'^```[a-z]*\s*|\s*```$', '', raw.strip(), flags=re.MULTILINE).strip()
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
                        short_name=_ws(str(p.get("short_name", citation))),
                        role=_ws(str(p.get("role", "merely_cited"))),
                    ))
        return Extraction(
            summary=_ws(str(data.get("summary", ""))),
            ratio_decidendi=[_ws(str(r)) for r in data.get("ratio_decidendi", []) if str(r).strip()],
            outcome=_normalize_outcome(str(data.get("outcome", "unknown"))),
            provisions=provisions,
            precedents=precedents,
            statutes=[_ws(str(s)) for s in data.get("statutes", []) if str(s).strip()],
        )
    except Exception as exc:
        log.warning("LLM JSON parse failed (response len=%d): %s — using regex fallback", len(raw), exc)
        m_s = re.search(r'"summary"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
        m_o = re.search(r'"outcome"\s*:\s*"([^"]+)"', raw)
        return Extraction(
            summary=_ws(m_s.group(1)) if m_s else "",
            ratio_decidendi=[],
            outcome=_normalize_outcome(m_o.group(1)) if m_o else "unknown",
        )


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

    for s in extraction.statutes:
        s = _ws(s)
        if not s:
            continue
        c = canonicalize_entity(s, "STATUTE")
        if c not in stat:
            stat[c] = s

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
            c = canonicalize_entity(raw, "PROVISION")
            if c not in prov:
                prov[c] = raw
                prov_roles[raw] = "background"
        elif label == "STATUTE":
            c = canonicalize_entity(raw, "STATUTE")
            if c not in stat:
                stat[c] = raw
        elif label == "PRECEDENT":
            c = canonicalize_entity(raw, "PRECEDENT")
            if c not in prec:
                prec[c] = raw
                prec_roles[raw] = "merely_cited"

    # Remove bare provision slugs subsumed by an act-qualified canonical.
    # e.g. "_401" (from bare "Section 401") is redundant when "crpc_401" already exists.
    qualified = {c for c in prov if not c.startswith("_")}
    for c in list(prov):
        if c.startswith("_"):
            num = c.lstrip("_")
            if any(q.endswith(f"_{num}") for q in qualified):
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
    extraction: Extraction = _parse_ingest_response(resp.content)

    spacy_spans = _extract_legal_spans(truncated)
    canon, prov_roles, prec_roles = _merge_entities(extraction, spacy_spans)
    meta = parse_metadata(truncated)

    ratio_text  = " ".join(extraction.ratio_decidendi) or extraction.summary
    master_text = extraction.summary + " " + " ".join(extraction.ratio_decidendi)

    # Sequential embed calls (~50ms each) — nested ThreadPoolExecutor inside an
    # already-pooled thread causes starvation at workers >= 4 (BUG 4 fix)
    dense_ratio = embedder.embed_query(ratio_text)
    dense_facts = embedder.embed_query(extraction.summary)
    dense_full  = embedder.embed_query(master_text)

    year_raw = meta.get("case_year", "") or ""
    try:
        m = re.search(r'\d{4}', str(year_raw))
        year = int(m.group(0)) if m else 0
    except Exception:
        year = 0

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
            "summary":          extraction.summary,
            "ratio_decidendi":  extraction.ratio_decidendi,   # list[str] directly (BUG 3 fix)
            "outcome":          extraction.outcome,
            "court":            meta.get("court", ""),
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
            "provision_roles":  prov_roles,   # {text -> role} (BUG 9 fix)
            "precedent_roles":  prec_roles,   # {citation -> role} (BUG 9 fix)
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
        api_key = os.environ.get("QDRANT_API_KEY")
        if not api_key:
            log.error(
                "QDRANT_API_KEY not set — cannot connect to cloud Qdrant. "
                "Check backend/.env contains QDRANT_API_KEY = \"...\"."
            )
            sys.exit(1)
        client = QdrantClient(url=_CLOUD_URL, api_key=api_key, timeout=60)
        log.info("Using Qdrant Cloud at %s", _CLOUD_URL)

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
