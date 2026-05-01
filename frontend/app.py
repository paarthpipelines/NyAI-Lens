import base64
import html as _html_mod
import json as _json
import os
import re
import sys
import streamlit as st
import streamlit.components.v1 as _st_components
import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.text_utils import (  # noqa: E402
    _DIGITAL_SIG_RE,
    _PAGE_PATTERNS,
    _ROMAN_PAGE_RE,
    _is_page_line,
    strip_page_numbers,
)

BACKEND_URL = "http://localhost:8000"


def _clean_md_tables(md: str) -> str:
    """Collapse runs of identical consecutive cells caused by merged PDF cells."""
    def _collapse(row_line: str) -> list[str]:
        cells = [c.strip() for c in row_line.split("|")[1:-1]]
        out: list[str] = []
        for c in cells:
            if not out or c != out[-1]:
                out.append(c)
        return out

    lines = md.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not (line.startswith("|") and line.endswith("|")):
            result.append(line)
            i += 1
            continue
        block: list[str] = []
        while i < len(lines) and lines[i].startswith("|") and lines[i].endswith("|"):
            block.append(lines[i])
            i += 1
        collapsed = [_collapse(r) for r in block]
        is_sep = [all(re.match(r"^[-: ]+$", c) for c in r) for r in collapsed]
        data_widths = [len(r) for r, sep in zip(collapsed, is_sep) if not sep]
        width = min(data_widths) if data_widths else (len(collapsed[0]) if collapsed else 0)
        for row, sep in zip(collapsed, is_sep):
            if sep:
                result.append("|" + "|".join(["---"] * width) + "|")
            else:
                result.append("|" + "|".join(row[:width]) + "|")
    return "\n".join(result)

st.set_page_config(page_title="Judgment Viewer", layout="centered")

# ── NER display config ─────────────────────────────────────────────────────────
_LEGAL_LABELS: dict[str, str] = {
    "PROVISION":  "#F0B27A",
    "STATUTE":    "#F1948A",
    "PRECEDENT":  "#C39BD3",
}
_LEGAL_LEGEND_HTML = (
    "<div style='display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;font-size:0.78rem'>"
    + "".join(
        f"<span style='background:{c};color:#1f2937;padding:2px 9px;"
        f"border-radius:4px;font-weight:600'>{label}</span>"
        for label, c in _LEGAL_LABELS.items()
    )
    + "</div>"
)

# ── Rhetorical role display config ─────────────────────────────────────────────
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

_RHET_COLORS = {
    "Preamble":         ("#E8F4FD", "#1a5276"),
    "Facts":            ("#FFF3CD", "#7d6608"),
    "Procedural":       ("#D1ECF1", "#0c5460"),
    "Evidence":         ("#FEF9C3", "#854d0e"),
    "Issue":            ("#F8D7DA", "#721c24"),
    "Arg \u00b7 Petitioner": ("#D4EDDA", "#155724"),
    "Arg \u00b7 Respondent": ("#FCE4EC", "#880e4f"),
    "Precedent":        ("#F3E5F5", "#6a1b9a"),
    "Observation":      ("#EDE7F6", "#4a235a"),
    "Ratio Decidendi":  ("#E8F5E9", "#1b5e20"),
    "Disposition":      ("#FFFDE7", "#f57f17"),
    "Directives":       ("#FFF0E6", "#c2410c"),
    "Costs":            ("#F1F5F9", "#475569"),
    "Not classified":   ("#F5F5F5", "#9CA3AF"),
}

_RHET_LEGEND_HTML = (
    "<div style='display:flex;flex-wrap:wrap;gap:6px;margin-bottom:10px'>"
    + "".join(
        f"<span style='background:{bg};color:{fg};padding:2px 9px;"
        f"border-radius:20px;font-size:.72rem;font-weight:600'>{label}</span>"
        for label, (bg, fg) in _RHET_COLORS.items()
    )
    + "</div>"
)

MAJOR_COURTS = [
    "Any",
    "Supreme Court of India",
    "High Court of Delhi",
    "High Court of Bombay",
    "High Court of Calcutta",
    "High Court of Madras",
    "High Court of Allahabad",
    "High Court of Karnataka",
    "High Court of Kerala",
    "High Court of Gujarat",
    "High Court of Rajasthan",
    "High Court of Punjab and Haryana",
    "High Court of Andhra Pradesh",
    "High Court of Telangana",
    "NCLAT",
    "NCLT",
    "Income Tax Appellate Tribunal",
]

_BATCH_SIZE = 8
_RERENDER_EVERY = 8


# ── Role helpers ───────────────────────────────────────────────────────────────

def _rhet_role(entry) -> str:
    if isinstance(entry, dict):
        return entry.get("role", "Not classified")
    return entry or "Not classified"


def _rhet_confidence(entry) -> float:
    if isinstance(entry, dict):
        return float(entry.get("confidence", 1.0))
    return 1.0


# ── Backend calls ──────────────────────────────────────────────────────────────

_NER_REQUEST_MAX = 10_000  # matches backend cap


def _get_legal_spans(text: str) -> list[tuple[str, str]]:
    try:
        resp = httpx.post(
            f"{BACKEND_URL}/ner",
            json={"text": text[:_NER_REQUEST_MAX]},
            timeout=60,
        )
        return [(s["text"], s["label"]) for s in resp.json()["spans"]]
    except Exception:
        return []


# ── UI helpers ─────────────────────────────────────────────────────────────────

def highlight(text: str, query: str) -> str:
    if not query:
        return text
    return re.sub(
        f'({re.escape(query)})',
        r'<mark>\1</mark>',
        text,
        flags=re.IGNORECASE,
    )


def _highlight_query_in_md(body: str, q: str) -> str:
    if not q:
        return body
    return re.sub(
        f'({re.escape(q)})',
        r'<mark style="background:#FFD54F;color:#000;border-radius:2px;padding:0 2px">\1</mark>',
        body,
        flags=re.IGNORECASE,
    )


def _apply_legal_highlights(body: str, spans: list[tuple[str, str]]) -> str:
    for span_text, label in sorted(spans, key=lambda t: len(t[0]), reverse=True):
        color = _LEGAL_LABELS.get(label, "#FFFDE7")
        body = re.sub(
            rf'({re.escape(span_text)})',
            rf'<mark style="background:{color};color:#1f2937;border-radius:3px;'
            rf'padding:0 3px;font-weight:600">\1</mark>',
            body,
        )
    return body



def _card_style_from_role(role: str, confidence: float) -> tuple[str, str]:
    bg, fg = _RHET_COLORS.get(role, ("#F5F5F5", "#616161"))
    border = "dashed" if confidence < 0.65 else "solid"
    card_style = (
        f"background:{bg};border-left:4px {border} {fg};"
        f"border-radius:6px;padding:10px 16px;margin-bottom:2px;color:#1f2937"
    )
    badge_html = (
        f"<span style='background:{fg};color:#fff;padding:1px 8px;"
        f"border-radius:20px;font-size:.66rem;font-weight:700;"
        f"letter-spacing:.4px;display:inline-block;margin-bottom:5px'>"
        f"{role}</span>"
    )
    return card_style, badge_html


def render_para_with_override(
    num: str,
    role: str,
    confidence: float,
    body_md: str,
    card_style: str,
    badge_html: str,
    needs_html: bool = False,
) -> None:
    bg, fg = _RHET_COLORS.get(role, ("#F5F5F5", "#616161"))
    border = "dashed" if confidence < 0.65 else "solid"
    # Inject CSS that applies the card background to the immediately following
    # Streamlit element-container (the native st.markdown body call below).
    # Using :has() keeps the body untouched so tables/lists render natively.
    st.markdown(
        f"<style>.element-container:has(#para-{num}) + .element-container{{"
        f"background:{bg};border-left:4px {border} {fg};"
        f"border-radius:0 0 6px 6px;padding:0 16px 10px;margin-bottom:2px;color:#1f2937"
        f"}}</style>"
        f"<div id='para-{num}' style='background:{bg};border-left:4px {border} {fg};"
        f"border-radius:6px 6px 0 0;padding:6px 16px 0;margin-bottom:0'>"
        f"{badge_html}</div>",
        unsafe_allow_html=True,
    )
    st.markdown(f"**{num}.** " + body_md, unsafe_allow_html=needs_html)


def render_footer(meta: dict) -> None:
    judges   = meta.get("judges") or []
    date     = meta.get("judgment_date", "")
    location = meta.get("judgment_location", "")

    if not judges and not date and not location:
        return

    judge_cols = ""
    for name in judges:
        judge_cols += (
            "<div style='text-align:center;flex:1;min-width:140px'>"
            "<div style='font-family:monospace;color:#6b7280;letter-spacing:1px;"
            "font-size:.9rem;padding-bottom:6px;border-bottom:1px solid #d1d5db'>"
            ".....................................J.</div>"
            f"<div style='font-size:.82rem;color:#374151;margin-top:6px;"
            f"font-style:italic'>({name})</div>"
            "</div>"
        )

    bottom = ""
    if location or date:
        loc_line  = f"<div>{location.upper()};</div>" if location else ""
        date_line = f"<div>{date.upper()}.</div>"     if date     else ""
        bottom = (
            "<div style='text-align:right;font-size:.8rem;color:#6b7280;"
            f"margin-top:20px;font-family:monospace;line-height:1.9'>"
            f"{loc_line}{date_line}</div>"
        )

    st.markdown(
        "<div style='margin-top:8px;padding:20px 0 8px;"
        "border-top:2px solid #e5e7eb'>"
        f"<div style='display:flex;justify-content:space-around;"
        f"gap:16px;flex-wrap:wrap'>{judge_cols}</div>"
        f"{bottom}</div>",
        unsafe_allow_html=True,
    )


def show_pdf(pdf_bytes: bytes) -> None:
    b64 = base64.b64encode(pdf_bytes).decode()
    st.components.v1.html(
        f"<iframe src='data:application/pdf;base64,{b64}' "
        f"width='100%' height='900' "
        f"style='border:none;border-radius:6px'></iframe>",
        height=910,
        scrolling=False,
    )


def inject_chat_fab() -> None:
    doc_id = st.session_state.get("doc_id", "") or ""

    with st.sidebar:
        st.markdown("---")
        st.markdown("#### ⚖ Judgment Assistant")

        chat_container = st.container(height=350, border=True)
        with chat_container:
            history = st.session_state.get("chat_history", [])
            if not history:
                st.caption("Ask anything about this document.")
            for msg in history:
                if msg["role"] == "user":
                    st.chat_message("user").write(msg["content"])
                else:
                    st.chat_message("assistant").markdown(msg["content"])

        question = st.chat_input("Ask about this judgment…", key="chat_input_fab")
        if question and doc_id:
            st.session_state.setdefault("chat_history", []).append(
                {"role": "user", "content": question}
            )
            sources = []
            streamed = ""
            with chat_container:
                st.chat_message("user").write(question)
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    try:
                        with httpx.stream(
                            "POST",
                            f"{BACKEND_URL}/chat",
                            json={
                                "doc_id": doc_id,
                                "question": question,
                                "history": st.session_state["chat_history"][:-1],
                                "chunks": [],
                            },
                            timeout=300,
                        ) as r:
                            for line in r.iter_lines():
                                if line.startswith("data: "):
                                    data = _json.loads(line[6:])
                                    if not data.get("done"):
                                        streamed += data.get("token", "")
                                        response_placeholder.markdown(streamed + "▌")
                                    elif "sources" in data:
                                        sources = data["sources"]
                        response_placeholder.markdown(streamed)
                    except httpx.ConnectError:
                        streamed = f"[Cannot reach backend at {BACKEND_URL}]"
                        response_placeholder.markdown(streamed)

            full_answer = streamed
            if sources:
                full_answer += "\n\n**Sources:** " + ", ".join(sources)

            st.session_state["chat_history"].append(
                {"role": "assistant", "content": full_answer}
            )
            st.rerun()

        if st.session_state.get("chat_history"):
            if st.button("Clear chat", key="clear_chat_btn", use_container_width=True):
                st.session_state["chat_history"] = []
                st.rerun()
# ── UI ─────────────────────────────────────────────────────────────────────────

st.title("Judgment Viewer")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is None:
    st.stop()

file_key = uploaded_file.name
if st.session_state.get("_pdf_key") != file_key:
    st.session_state["legal_spans_cache"] = {}
    st.session_state["rhet_cache"]        = {}
    st.session_state["chat_history"]      = []
    st.session_state["para_embeddings"]   = {}
    st.session_state["chat_open"]         = False
    st.session_state["_chat_pending"]     = None
    st.session_state["similar_results"]   = None
    st.session_state["_extracted"]        = None
    st.session_state["_pdf_key"]          = file_key

pdf_bytes = uploaded_file.getvalue()
st.session_state["pdf_bytes"] = pdf_bytes

if st.session_state.get("_extracted") is None:
    with st.spinner("Extracting..."):
        try:
            resp = httpx.post(
                f"{BACKEND_URL}/extract",
                files={"file": (uploaded_file.name, pdf_bytes, "application/pdf")},
                timeout=120,
            )
            resp.raise_for_status()
            st.session_state["_extracted"] = resp.json()
        except httpx.ConnectError:
            st.error(
                f"Cannot reach the backend at **{BACKEND_URL}**. "
                "Start it with:\n```\nuvicorn backend.main:app --reload\n```"
            )
            st.stop()
        except httpx.HTTPStatusError as _e:
            st.error(f"Backend error {_e.response.status_code}: {_e.response.text}")
            st.stop()

_extracted  = st.session_state["_extracted"]
doc_id      = _extracted["doc_id"]
raw_md      = _extracted["markdown"]
header_raw  = _extracted["header"]
chunks      = _extracted["chunks"]
footer_raw  = _extracted["footer"]
meta        = _extracted["metadata"]
meta_regex  = _extracted.get("metadata_regex", meta)
meta_llm    = _extracted.get("metadata_llm", {})
st.session_state["doc_id"] = doc_id

inject_chat_fab()


# ── Header metadata ────────────────────────────────────────────────────────────
def render_metadata(meta: dict) -> None:
    status_html = ""
    if meta["reportable_status"] == "REPORTABLE":
        status_html = (
            "<span style='background:#fee2e2;color:#b91c1c;"
            "padding:2px 10px;border-radius:20px;font-size:0.75rem;"
            "font-weight:600;letter-spacing:.4px'>REPORTABLE</span>"
        )
    elif meta["reportable_status"] == "NON-REPORTABLE":
        status_html = (
            "<span style='background:#f3f4f6;color:#6b7280;"
            "padding:2px 10px;border-radius:20px;font-size:0.75rem;"
            "font-weight:600;letter-spacing:.4px'>NON-REPORTABLE</span>"
        )
    if meta["is_batch_matter"]:
        status_html += (
            " <span style='background:#dbeafe;color:#1d4ed8;"
            "padding:2px 10px;border-radius:20px;font-size:0.75rem;"
            "font-weight:600;letter-spacing:.4px;margin-left:4px'>BATCH MATTER</span>"
        )

    court_html = ""
    if meta["court"]:
        court_html += f"<div style='font-size:.8rem;color:#555;font-weight:500;letter-spacing:.3px'>{meta['court'].upper()}</div>"
    if meta["jurisdiction"]:
        court_html += f"<div style='font-size:.78rem;color:#888;margin-top:2px'>{meta['jurisdiction'].upper()}</div>"

    citation_html = ""
    if meta["citation"]:
        citation_html = (
            f"<div style='font-size:.8rem;color:#555;font-family:monospace;"
            f"font-weight:600;letter-spacing:.5px'>{meta['citation']}</div>"
        )

    top_row = (
        f"<div style='display:flex;justify-content:space-between;"
        f"align-items:flex-start;margin-bottom:14px'>"
        f"  <div>{court_html}</div>"
        f"  <div style='display:flex;flex-direction:column;align-items:flex-end;gap:6px'>"
        f"    {status_html}{citation_html}"
        f"  </div>"
        f"</div>"
    )

    case_no = ""
    if meta["case_type"] and meta["case_number"] and meta["case_year"]:
        case_no = f"{meta['case_type'].upper()} NO. {meta['case_number']} OF {meta['case_year']}"
    elif meta["case_type"]:
        case_no = meta["case_type"].upper()
    case_no_html = ""
    if case_no:
        case_no_html = (
            f"<div style='font-size:1rem;font-weight:700;color:#111;"
            f"padding-bottom:12px;margin-bottom:12px;"
            f"border-bottom:1px solid #e5e7eb'>{case_no}</div>"
        )

    label_style = (
        "font-size:.72rem;color:#9ca3af;text-transform:uppercase;"
        "letter-spacing:.6px;padding:5px 0;vertical-align:top;white-space:nowrap;padding-right:12px"
    )
    name_style = "font-size:.9rem;font-weight:600;color:#111;padding:5px 0"

    parties_html = ""
    if meta["petitioner_or_appellant"]:
        role = "Appellant" if meta["case_type"] and "Appeal" in meta["case_type"] else "Petitioner"
        parties_html += (
            f"<tr>"
            f"<td style='{label_style}'>{role}</td>"
            f"<td style='{name_style}'>{meta['petitioner_or_appellant']}</td>"
            f"</tr>"
        )
    if meta["respondent"]:
        parties_html += (
            f"<tr>"
            f"<td style='{label_style}'>Respondent</td>"
            f"<td style='{name_style}'>{meta['respondent']}</td>"
            f"</tr>"
        )
    parties_table = (
        f"<table style='width:100%;border-collapse:collapse;margin-bottom:10px'>"
        f"{parties_html}</table>"
        if parties_html else ""
    )

    _lo_lbl = (
        "font-size:.7rem;color:#9ca3af;text-transform:uppercase;"
        "letter-spacing:.5px;padding:4px 0;vertical-align:top;"
        "white-space:nowrap;padding-right:14px"
    )
    _lo_val = "font-size:.82rem;color:#374151;padding:4px 0"

    footer_rows = []
    coram = ", ".join(meta.get("judges") or []) or meta.get("judge", "")
    if coram:
        footer_rows.append(("Coram", f"<span style='font-style:italic'>{coram}</span>"))
    if meta.get("judgment_date"):
        footer_rows.append(("Date", meta["judgment_date"]))
    if meta.get("judgment_location"):
        footer_rows.append(("Place", meta["judgment_location"]))

    lower_meta_html = ""
    if footer_rows:
        rows_html = "".join(
            f"<tr><td style='{_lo_lbl}'>{lbl}</td>"
            f"<td style='{_lo_val}'>{val}</td></tr>"
            for lbl, val in footer_rows
        )
        lower_meta_html = (
            "<div style='border-top:1px solid #e5e7eb;margin-top:10px;padding-top:8px'>"
            f"<table style='border-collapse:collapse;width:100%'>{rows_html}</table>"
            "</div>"
        )

    card_html = (
        "<div style='border:1px solid #e5e7eb;border-radius:10px;padding:20px;"
        "font-family:-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;"
        "background:#fafafa;margin-bottom:8px'>"
        + top_row + case_no_html + parties_table + lower_meta_html +
        "</div>"
    )
    st.markdown(card_html, unsafe_allow_html=True)

    if meta["toc_md"]:
        with st.expander("Table of Contents"):
            st.markdown(meta["toc_md"])


_META_COMPARE_FIELDS = [
    ("court",                   "Court"),
    ("jurisdiction",            "Jurisdiction"),
    ("case_type",               "Case Type"),
    ("case_number",             "Case Number"),
    ("case_year",               "Year"),
    ("petitioner_or_appellant", "Petitioner / Appellant"),
    ("respondent",              "Respondent"),
    ("authoring_judge",         "Authoring Judge"),
    ("judgment_date",           "Date"),
    ("judgment_location",       "Location"),
    ("citation",                "Citation"),
    ("reportable_status",       "Status"),
]


def render_metadata_comparison(meta: dict, meta_r: dict, meta_l: dict) -> None:
    render_metadata(meta)  # merged (LLM primary, regex fallback) shown as main card

    with st.expander("🔍 Metadata Extraction Comparison (Regex vs LLM)", expanded=False):
        hdr_l, hdr_r, hdr_m = st.columns([1.5, 2, 2])
        hdr_l.markdown("**Field**")
        hdr_r.markdown(
            "<span style='background:#dbeafe;color:#1e40af;padding:2px 10px;"
            "border-radius:20px;font-size:.75rem;font-weight:700'>REGEX</span>",
            unsafe_allow_html=True,
        )
        hdr_m.markdown(
            "<span style='background:#d1fae5;color:#065f46;padding:2px 10px;"
            "border-radius:20px;font-size:.75rem;font-weight:700'>LLM</span>",
            unsafe_allow_html=True,
        )
        st.divider()

        for field_key, field_label in _META_COMPARE_FIELDS:
            val_r = str(meta_r.get(field_key) or "")
            val_l = str(meta_l.get(field_key) or "")
            if not val_r and not val_l:
                continue

            col_l, col_r, col_m = st.columns([1.5, 2, 2])
            col_l.markdown(f"<span style='color:#6b7280;font-size:.82rem'>{field_label}</span>",
                           unsafe_allow_html=True)

            if val_r == val_l and val_r:
                col_r.markdown(f"**{val_r}**")
                col_m.markdown(f"**{val_l}** ✅")
            elif val_r and val_l and val_r != val_l:
                col_r.markdown(
                    f"<span style='background:#FEF3C7;color:#92400E;padding:2px 8px;"
                    f"border-radius:4px;font-size:.85rem'>{val_r}</span>",
                    unsafe_allow_html=True,
                )
                col_m.markdown(
                    f"<span style='background:#FEF3C7;color:#92400E;padding:2px 8px;"
                    f"border-radius:4px;font-size:.85rem'>{val_l}</span> ⚠️",
                    unsafe_allow_html=True,
                )
            elif val_r:
                col_r.markdown(val_r)
                col_m.caption("not found")
            else:
                col_r.caption("not found")
                col_m.markdown(val_l)

        st.divider()

        # Batch matter row
        b_r = meta_r.get("is_batch_matter", False)
        b_l = meta_l.get("is_batch_matter", False)
        if b_r or b_l:
            c1, c2, c3 = st.columns([1.5, 2, 2])
            c1.markdown("<span style='color:#6b7280;font-size:.82rem'>Batch Matter</span>",
                        unsafe_allow_html=True)
            c2.markdown("Yes" if b_r else "No")
            c3.markdown(("Yes" if b_l else "No") + (" ✅" if b_r == b_l else " ⚠️"))

        # Judges row
        j_r = meta_r.get("judges") or []
        j_l = meta_l.get("judges") or []
        if j_r or j_l:
            c1, c2, c3 = st.columns([1.5, 2, 2])
            c1.markdown("<span style='color:#6b7280;font-size:.82rem'>Judges (Coram)</span>",
                        unsafe_allow_html=True)
            c2.markdown(", ".join(j_r) if j_r else "—")
            agreed_judges = set(j_r) == set(j_l) and j_r
            c3.markdown((", ".join(j_l) if j_l else "—") + (" ✅" if agreed_judges else ""))


if meta_llm:
    render_metadata_comparison(meta, meta_regex, meta_llm)
else:
    render_metadata(meta)



# ── View tabs ──────────────────────────────────────────────────────────────────
tab_text, tab_pdf, tab_similar = st.tabs(["Extracted Text", "Original PDF", "Similar Judgments"])

with tab_pdf:
    show_pdf(pdf_bytes)

with tab_text:
    if not chunks:
        st.warning("No numbered paragraphs found — displaying raw text.")
        st.markdown(_clean_md_tables(raw_md))
        st.stop()

    st.caption(f"{len(chunks)} paragraphs")

# ── Sidebar: rhetorical role controls ─────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("#### Rhetorical Labels")
    show_rhet = st.toggle("Show labels", value=True, key="show_rhet_labels")
    if show_rhet:
        _rhet_cache_now: dict = st.session_state.get("rhet_cache", {})
        _all_roles = list(_RHET_COLORS.keys())
        _all_classified_now = bool(_rhet_cache_now) and all(
            c["para_no"] in _rhet_cache_now for c in (
                st.session_state.get("_chunks_ref", [])
            )
        )
        _sidebar_roles = (
            sorted(
                {_rhet_role(_rhet_cache_now[c["para_no"]])
                 for c in st.session_state.get("_chunks_ref", [])
                 if c["para_no"] in _rhet_cache_now},
                key=_all_roles.index,
            )
            if _all_classified_now else _all_roles
        )
        st.caption("Filter paragraphs by role")
        if "rhet_role_filter" not in st.session_state:
            st.session_state["rhet_role_filter"] = set(_all_roles)
        _col_all, _col_none = st.columns(2)
        if _col_all.button("All", use_container_width=True):
            st.session_state["rhet_role_filter"] = set(_all_roles)
        if _col_none.button("None", use_container_width=True):
            st.session_state["rhet_role_filter"] = set()
        for _role in _sidebar_roles:
            _bg, _fg = _RHET_COLORS.get(_role, ("#F5F5F5", "#616161"))
            _checked = _role in st.session_state["rhet_role_filter"]
            _new = st.checkbox(
                _role,
                value=_checked,
                key=f"rhet_cb_{_role}",
            )
            if _new != _checked:
                if _new:
                    st.session_state["rhet_role_filter"].add(_role)
                else:
                    st.session_state["rhet_role_filter"].discard(_role)

    st.divider()
    st.markdown("#### Legal Highlights")
    st.toggle(
        "Show provisions, statutes & precedents",
        value=True,
        key="show_legal_highlights",
        help="Highlights PROVISION, STATUTE and PRECEDENT spans detected by the NER model",
    )
    if st.session_state.get("show_legal_highlights", True):
        st.html(_LEGAL_LEGEND_HTML)





@st.fragment
def search_and_display(chunks: list, meta: dict) -> None:
    rhet_cache: dict = st.session_state.get("rhet_cache", {})
    all_classified = bool(rhet_cache) and all(c["para_no"] in rhet_cache for c in chunks)

    show_rhet = st.session_state.get("show_rhet_labels", True)
    show_legal = st.session_state.get("show_legal_highlights", True)
    selected_roles: set = st.session_state.get("rhet_role_filter", set(_RHET_COLORS.keys()))

    search_q = st.text_input("Search", placeholder="Keyword...")
    q = search_q.strip()

    display_chunks = [c for c in chunks if not q or q.lower() in c["body"].lower()]
    if show_rhet:
        display_chunks = [
            c for c in display_chunks
            if c["para_no"] not in rhet_cache
            or _rhet_role(rhet_cache[c["para_no"]]) in selected_roles
        ]

    if q:
        st.caption(f"{len(display_chunks)} result(s)")

    if show_rhet and rhet_cache:
        st.html(_RHET_LEGEND_HTML)

    for chunk in display_chunks:
        num  = chunk["para_no"]
        body = _clean_md_tables(chunk["body"])

        if chunk["is_section_start"] and chunk["section"]:
            st.markdown("---")
            st.markdown(f"### {chunk['section']}")

        _cache_entry = rhet_cache.get(num) if show_rhet else None
        role = _rhet_role(_cache_entry) if _cache_entry is not None else None
        confidence = _rhet_confidence(_cache_entry) if _cache_entry is not None else 1.0

        if show_rhet and role:
            card_style, badge_html = _card_style_from_role(role, confidence)
        elif show_rhet:
            badge_html = (
                "<span style='font-size:.66rem;color:#9CA3AF;font-style:italic;"
                "display:inline-block;margin-bottom:5px'>classifying\u2026</span>"
            )
            card_style = (
                "border-left:4px solid #9CA3AF;border-radius:6px;"
                "padding:10px 16px;margin-bottom:2px"
            )
        else:
            badge_html = ""
            card_style = ""

        _legal_spans: list[tuple[str, str]] = []
        if show_legal:
            _span_cache: dict = st.session_state.setdefault("legal_spans_cache", {})
            if num not in _span_cache:
                _span_cache[num] = _get_legal_spans(body)
            _legal_spans = _span_cache[num]

        display_body = body
        needs_html = bool(q or _legal_spans)
        if q:
            display_body = _highlight_query_in_md(display_body, q)
        if _legal_spans:
            display_body = _apply_legal_highlights(display_body, _legal_spans)

        if badge_html and role:
            render_para_with_override(num, role, confidence, display_body, card_style, badge_html, needs_html)
        elif badge_html:
            st.markdown(
                f"<style>.element-container:has(#para-{num}) + .element-container{{"
                f"border-left:4px solid #9CA3AF;border-radius:0 0 6px 6px;"
                f"padding:0 16px 10px;margin-bottom:2px"
                f"}}</style>"
                f"<div id='para-{num}' style='border-left:4px solid #9CA3AF;"
                f"border-radius:6px 6px 0 0;padding:6px 16px 0;margin-bottom:0'>"
                f"{badge_html}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**{num}.** " + display_body, unsafe_allow_html=needs_html)
        else:
            st.markdown(
                f"<div id='para-{num}'></div>",
                unsafe_allow_html=True,
            )
            st.markdown(f"**{num}.** " + display_body, unsafe_allow_html=needs_html)

        st.divider()

    render_footer(meta)

    # ── Batch rhetorical classification via backend SSE ────────────────────────
    rhet_cache2: dict = st.session_state.setdefault("rhet_cache", {})
    unclassified_indices = [
        i for i, c in enumerate(chunks) if c["para_no"] not in rhet_cache2
    ]
    if unclassified_indices:
        total = len(unclassified_indices)
        _prog = st.progress(0, text=f"Classifying {total} paragraphs\u2026")
        done = 0
        try:
            payload = {"doc_id": st.session_state.get("doc_id"), "indices": unclassified_indices}
            with httpx.stream(
                "POST",
                f"{BACKEND_URL}/classify",
                json=payload,
                timeout=600,
            ) as r:
                for line in r.iter_lines():
                    if line.startswith("data: "):
                        data = _json.loads(line[6:])
                        if "para_no" not in data:
                            continue
                        pn = data["para_no"]
                        rhet_cache2[pn] = {"role": data["role"], "confidence": data["confidence"]}
                        done += 1
                        _prog.progress(done / total, text=f"Classifying\u2026 {done}/{total}")
                        if done % _RERENDER_EVERY == 0:
                            try:
                                st.rerun(scope="fragment")  # fragment-only rerun; shows batch of colored cards
                            except Exception:
                                st.rerun()
            _prog.empty()
        except httpx.ConnectError:
            _prog.empty()
            st.error(f"Cannot reach backend at {BACKEND_URL}. Is `uvicorn backend.main:app --reload` running?")
        except Exception as _exc:
            _prog.empty()
            st.error(f"Classification error: {_exc}")
        try:
            st.rerun(scope="fragment")
        except Exception:
            st.rerun()


with tab_text:
    st.session_state["_chunks_ref"] = chunks
    search_and_display(chunks, meta)



# ── Similar Judgments tab ──────────────────────────────────────────────────────


def _pct_bar(label: str, pct: int, color: str) -> str:
    return (
        f"<div style='margin-bottom:6px'>"
        f"<div style='font-size:.7rem;color:#6b7280;margin-bottom:2px'>{label}</div>"
        f"<div style='display:flex;align-items:center;gap:8px'>"
        f"<div style='flex:1;height:7px;background:#e5e7eb;border-radius:4px;overflow:hidden'>"
        f"<div style='width:{pct}%;height:100%;background:{color};border-radius:4px'></div>"
        f"</div>"
        f"<span style='font-size:.72rem;color:#6b7280;min-width:28px'>{pct}%</span>"
        f"</div></div>"
    )


def _pills(items: list[str], category: str) -> str:
    if not items:
        return ""
    pill_html = "".join(
        f"<span onclick=\"navigator.clipboard.writeText('{_html_mod.escape(e)}')\""
        f" title='Click to copy' style='background:#e0f2fe;color:#0369a1;"
        f"padding:2px 9px;border-radius:12px;font-size:.72rem;cursor:pointer;"
        f"white-space:nowrap;display:inline-block;margin:2px 3px 2px 0'>"
        f"{_html_mod.escape(e)}</span>"
        for e in items[:5]
    )
    return (
        f"<div style='margin:4px 0'>"
        f"<span style='font-size:.7rem;color:#9ca3af;margin-right:4px'>{category}:</span>"
        f"{pill_html}</div>"
    )


def _similar_card_html(r: dict, idx: int) -> str:
    score_pct = min(int(r["final_score"] * 100), 100)
    badge_color = (
        "#16a34a" if score_pct >= 70 else
        "#d97706" if score_pct >= 50 else
        "#6b7280"
    )
    case_label = " ".join(filter(None, [r.get("case_type", ""), "NO.", r.get("case_number", "")])).strip(" NO.")
    if not case_label:
        case_label = f"Case {idx + 1}"
    meta_parts = [r.get("court", ""), str(r["year"]) if r.get("year") else "", r.get("judge", "")]
    meta_line  = " · ".join(p for p in meta_parts if p)
    summary    = r.get("summary", "")
    summary_trunc = (summary[:220] + "…") if len(summary) > 220 else summary
    ratio_items = r.get("ratio_decidendi", [])
    ratio_html = "".join(
        f"<li style='margin-bottom:4px'>{_html_mod.escape(pt)}</li>"
        for pt in ratio_items
    )
    ratio_block = (
        "<div style='border-top:1px solid #e5e7eb;padding-top:8px;margin-top:6px'>"
        "<div style='font-size:.75rem;font-weight:600;color:#6b7280;margin-bottom:4px;letter-spacing:.04em'>"
        "RATIO DECIDENDI</div>"
        f"<ul style='margin:0;padding-left:18px;font-size:.81rem;color:#1f2937;line-height:1.6'>{ratio_html}</ul>"
        "</div>"
    ) if ratio_items else ""

    bars = (
        _pct_bar("Semantic",  int(r["semantic_score"]   * 100), "#3b82f6") +
        _pct_bar("Precedent", int(r["precedent_score"] * 100), "#7c3aed") +
        _pct_bar("Provision", int(r["provision_score"] * 100), "#0891b2")
    )

    shared = (
        _pills(r.get("shared_precedents_raw", []), "Shared precedents") +
        _pills(r.get("shared_provisions_raw", []), "Shared provisions") +
        _pills(r.get("shared_statutes_raw",   []), "Shared statutes")
    )

    return f"""
<div style='border:1px solid #e5e7eb;border-radius:10px;padding:18px 20px;
            margin-bottom:14px;background:#fafafa;
            font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif'>
  <div style='display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:6px'>
    <div>
      <div style='font-size:.95rem;font-weight:700;color:#111'>{_html_mod.escape(case_label)}</div>
      <div style='font-size:.77rem;color:#6b7280;margin-top:2px'>{_html_mod.escape(meta_line)}</div>
    </div>
    <span style='background:{badge_color};color:#fff;padding:3px 12px;border-radius:20px;
                 font-size:.77rem;font-weight:700;white-space:nowrap;margin-left:10px'>
      {score_pct}% similar
    </span>
  </div>
  <div style='font-size:.78rem;color:#374151;margin-bottom:10px'>
    Outcome: <strong>{_html_mod.escape(r.get("outcome","").title())}</strong>
  </div>
  <div style='border-top:1px solid #e5e7eb;padding-top:10px;margin-bottom:10px'>
    {bars}
  </div>
  {'<div style="border-top:1px solid #e5e7eb;padding-top:8px;margin-bottom:10px">' + shared + '</div>' if shared.strip() else ''}
  <div style='border-top:1px solid #e5e7eb;padding-top:8px;font-size:.82rem;
              color:#374151;line-height:1.55;font-style:italic'>
    {_html_mod.escape(summary_trunc)}
  </div>
  {ratio_block}
</div>
"""


with tab_similar:
    with st.expander("Filters", expanded=False):
        _col1, _col2, _col3, _col4 = st.columns(4)
        _sim_court     = _col1.selectbox("Court",     MAJOR_COURTS, key="sim_court")
        _sim_year_from = _col2.number_input("Year from", min_value=1950, max_value=2025, value=2000, key="sim_year_from")
        _sim_year_to   = _col3.number_input("Year to",   min_value=1950, max_value=2025, value=2025, key="sim_year_to")
        _sim_act       = _col4.text_input("Act", placeholder="e.g. Indian Penal Code", key="sim_act")
        if st.button("Apply filters & re-search", key="sim_apply_filters"):
            st.session_state.pop("_sim_doc_id", None)
            st.session_state["similar_results"] = None
            st.session_state["query_extraction"] = None

    # Auto-trigger search when a new document is loaded
    if st.session_state.get("_sim_doc_id") != doc_id:
        st.session_state["_sim_doc_id"] = doc_id
        st.session_state["similar_results"] = None
        st.session_state["query_extraction"] = None

    if st.session_state.get("similar_results") is None and doc_id:
        _params: dict = {}
        if _sim_court and _sim_court != "Any":
            _params["court"] = _sim_court
        if _sim_year_from:
            _params["year_from"] = int(_sim_year_from)
        if _sim_year_to:
            _params["year_to"] = int(_sim_year_to)
        if _sim_act and _sim_act.strip():
            _params["act"] = _sim_act.strip()

        with st.spinner("Searching for similar judgments… (30–60 s)"):
            try:
                _sim_resp = httpx.post(
                    f"{BACKEND_URL}/similar",
                    files={"file": (uploaded_file.name, st.session_state["pdf_bytes"], "application/pdf")},
                    params=_params,
                    timeout=300,
                )
                _sim_resp.raise_for_status()
                _sim_data = _sim_resp.json()
                st.session_state["similar_results"] = _sim_data.get("results", [])
                st.session_state["query_extraction"] = {
                    "summary": _sim_data.get("query_summary", ""),
                    "ratio_decidendi": _sim_data.get("query_ratio_decidendi", []),
                    "provisions": _sim_data.get("query_provisions", []),
                    "precedents": _sim_data.get("query_precedents", []),
                    "statutes": _sim_data.get("query_statutes", []),
                }
            except httpx.ConnectError:
                st.error(f"Cannot reach backend at {BACKEND_URL}.")
                st.session_state["similar_results"] = []
            except httpx.HTTPStatusError as _e:
                st.error(f"Backend error {_e.response.status_code}: {_e.response.text}")
                st.session_state["similar_results"] = []

    _results = st.session_state.get("similar_results")
    _qex = st.session_state.get("query_extraction")

    if _qex:
        with st.expander("Your document — extracted summary & ratio decidendi", expanded=True):
            st.markdown(f"**Summary**\n\n{_qex['summary']}")
            if _qex["ratio_decidendi"]:
                st.markdown("**Ratio Decidendi**")
                for _pt in _qex["ratio_decidendi"]:
                    st.markdown(f"- {_pt}")
            if _qex.get("precedents"):
                st.markdown("**Precedents**")
                st.dataframe(
                    [{"Citation": p["citation"], "Short Name": p["short_name"], "Role": p["role"]} for p in _qex["precedents"]],
                    use_container_width=True, hide_index=True,
                )
            if _qex.get("provisions"):
                st.markdown("**Provisions**")
                st.dataframe(
                    [{"Provision": p["text"], "Role": p["role"]} for p in _qex["provisions"]],
                    use_container_width=True, hide_index=True,
                )
            if _qex.get("statutes"):
                st.markdown("**Statutes**")
                st.dataframe(
                    [{"Statute": s} for s in _qex["statutes"]],
                    use_container_width=True, hide_index=True,
                )

    if _results is not None and len(_results) == 0:
        st.warning("No similar judgments found. The Qdrant corpus may be empty — run `python -m backend.ingest` first.")
    elif _results:
        st.caption(f"Top {len(_results)} similar judgments")
        _cards_html = "".join(_similar_card_html(r, i) for i, r in enumerate(_results))
        _st_components.html(
            f"<!DOCTYPE html><html><head>"
            f"<meta charset='utf-8'>"
            f"<style>body{{margin:0;padding:4px}}</style>"
            f"</head><body>{_cards_html}</body></html>",
            height=len(_results) * 380,
            scrolling=True,
        )
