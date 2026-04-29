"""
Legal entity canonicalization for Indian court judgments.

Converts raw NER spans (PROVISION, STATUTE, PRECEDENT) to stable canonical
identifiers so Jaccard comparison is reliable regardless of surface variation:
  "Section 302 IPC"  vs  "S.302 of the Indian Penal Code"  →  both "ipc_302"
  "Maneka Gandhi v. UOI (1978) 1 SCC 248"  →  "maneka gandhi v union of india"
"""
from __future__ import annotations

import re

# ── Known act abbreviations (checked longest-first) ───────────────────────────
_ACT_ABBREV: dict[str, str] = {
    "indian penal code":                              "ipc",
    "ipc":                                            "ipc",
    "code of criminal procedure":                     "crpc",
    "criminal procedure code":                        "crpc",
    "crpc":                                           "crpc",
    "code of civil procedure":                        "cpc",
    "civil procedure code":                           "cpc",
    "cpc":                                            "cpc",
    "constitution of india":                          "const",
    "constitution":                                   "const",
    "negotiable instruments act":                     "ni",
    "ni act":                                         "ni",
    "income tax act":                                 "ita",
    "prevention of corruption act":                   "pca",
    "companies act":                                  "ca",
    "arbitration and conciliation act":               "aca",
    "transfer of property act":                       "tpa",
    "specific relief act":                            "sra",
    "hindu marriage act":                             "hma",
    "motor vehicles act":                             "mva",
    "consumer protection act":                        "cpa",
    "indian evidence act":                            "ea",
    "evidence act":                                   "ea",
    "limitation act":                                 "la",
    "protection of children from sexual offences act": "pocso",
    "pocso act":                                      "pocso",
    "pocso":                                          "pocso",
    "narcotic drugs and psychotropic substances act": "ndps",
    "ndps act":                                       "ndps",
    "ndps":                                           "ndps",
    "prevention of money laundering act":             "pmla",
    "pmla":                                           "pmla",
    "foreign exchange management act":                "fema",
    "fema":                                           "fema",
    "right to information act":                       "rti",
    "rti act":                                        "rti",
    "insolvency and bankruptcy code":                 "ibc",
    "ibc":                                            "ibc",
    "customs act":                                    "customs",
    "goods and services tax act":                     "gst",
    "gst act":                                        "gst",
}

# ── Regex helpers ──────────────────────────────────────────────────────────────
_SECTION_RE     = re.compile(r'(?:section|sec\.?|s\.)\s*(\d+[A-Za-z]?(?:\s*\(\w+\))*)', re.I)
_ARTICLE_RE     = re.compile(r'article\s*(\d+[A-Za-z]?(?:\s*\(\w+\))*)', re.I)
_RULE_RE        = re.compile(r'rule\s*(\d+[A-Za-z]?(?:\s*\(\w+\))*)', re.I)
_ORDER_RULE_RE  = re.compile(r'order\s+(\w+)\s+rule\s+(\d+)', re.I)
_YEAR_STRIP_RE  = re.compile(r',?\s*\d{4}\s*$')

# Strip citation suffixes: "(1978) 1 SCC 248", "AIR 1973 SC 1461", "2003 (3) SCC 1" …
_CITATION_RE = re.compile(
    r'\s*[\(\[]\s*\d{4}\s*[\)\]].*$'
    r'|\s+\d{4}\s+\(\d+\)\s+\w+.*$'
    r'|\s+AIR\s+\d{4}.*$'
    r'|\s+\d{4}\s+(?:SCC|SCR|AIR|All|MLJ|Bom|Cal|Mad|Del|Ker).*$',
    re.I,
)
_PUNCT_RE       = re.compile(r'[^\w\s]')
_MULTI_SPACE_RE = re.compile(r'\s+')


# ── Internal helpers ───────────────────────────────────────────────────────────

def _slugify(s: str) -> str:
    s = s.lower()
    s = _PUNCT_RE.sub(' ', s)
    s = _MULTI_SPACE_RE.sub('_', s.strip())
    return s


def _find_act_abbrev(text: str) -> str | None:
    lower = text.lower()
    for act in sorted(_ACT_ABBREV, key=len, reverse=True):
        if act in lower:
            return _ACT_ABBREV[act]
    return None


def _canon_provision(s: str) -> str:
    """'Section 302 IPC' → 'ipc_302';  'Article 21 Constitution' → 'const_21'."""
    m = _ORDER_RULE_RE.search(s)
    if m:
        abbrev = _find_act_abbrev(s) or "order"
        return f"{abbrev}_o{m.group(1).lower()}_r{m.group(2)}"

    num = None
    for pattern in (_ARTICLE_RE, _SECTION_RE, _RULE_RE):
        m = pattern.search(s)
        if m:
            num = re.sub(r'[^\w]', '', m.group(1)).lower()
            break

    abbrev = _find_act_abbrev(s)
    if num and abbrev:
        return f"{abbrev}_{num}"
    return _slugify(s)


def _canon_precedent(s: str) -> str:
    """'Maneka Gandhi v. UOI (1978) 1 SCC 248' → 'maneka gandhi v union of india'."""
    cleaned = _CITATION_RE.sub('', s)
    cleaned = cleaned.strip(' .,;:()')
    # Expand common abbreviations before slugifying
    cleaned = re.sub(r'\bU\.?O\.?I\.?\b', 'union of india', cleaned, flags=re.I)
    cleaned = re.sub(r'\bGovt\.?\b',      'government',     cleaned, flags=re.I)
    cleaned = re.sub(r'\bSt(?:ate?)?\b',  'state',          cleaned, flags=re.I)
    cleaned = _PUNCT_RE.sub(' ', cleaned.lower())
    cleaned = _MULTI_SPACE_RE.sub(' ', cleaned).strip()
    return cleaned


def _canon_statute(s: str) -> str:
    """'Indian Penal Code, 1860' → 'ipc';  'Companies Act, 2013' → 'ca'."""
    abbrev = _find_act_abbrev(s)
    if abbrev:
        return abbrev
    cleaned = _YEAR_STRIP_RE.sub('', s.strip())
    return _slugify(cleaned)


# ── Public API ─────────────────────────────────────────────────────────────────

def canonicalize_entity(s: str, label: str) -> str:
    """Return a stable canonical key for one NER span."""
    if label == "PROVISION":
        return _canon_provision(s)
    if label == "PRECEDENT":
        return _canon_precedent(s)
    if label == "STATUTE":
        return _canon_statute(s)
    return _slugify(s)


def canonicalize_spans(spans: list[tuple[str, str]]) -> dict[str, list[str]]:
    """
    Convert a list of (raw_text, label) NER spans into parallel raw/canonical
    lists per entity type.  Each type is deduplicated by canonical form.

    Returns:
        {
          "provisions_raw":   [...],  "provisions_canon":  [...],
          "statutes_raw":     [...],  "statutes_canon":    [...],
          "precedents_raw":   [...],  "precedents_canon":  [...],
        }
    Indices are parallel: provisions_raw[i] canonicalizes to provisions_canon[i].
    """
    out: dict[str, list[str]] = {
        "provisions_raw": [],  "provisions_canon": [],
        "statutes_raw":   [],  "statutes_canon":   [],
        "precedents_raw": [],  "precedents_canon": [],
    }
    seen: dict[str, set[str]] = {
        "PROVISION": set(), "STATUTE": set(), "PRECEDENT": set()
    }

    for raw, label in spans:
        if label not in seen:
            continue
        canon = canonicalize_entity(raw, label)
        if canon in seen[label]:
            continue
        seen[label].add(canon)
        key = label.lower() + "s"
        out[f"{key}_raw"].append(raw)
        out[f"{key}_canon"].append(canon)

    return out


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests: list[tuple[str, str, str]] = [
        # PROVISION
        ("Section 302 of the Indian Penal Code",          "PROVISION", "ipc_302"),
        ("Article 21 of the Constitution of India",        "PROVISION", "const_21"),
        ("Section 138 NI Act",                             "PROVISION", "ni_138"),
        ("Section 482 CrPC",                               "PROVISION", "crpc_482"),
        ("Section 34 IPC",                                 "PROVISION", "ipc_34"),
        ("S.302 IPC",                                      "PROVISION", "ipc_302"),
        # PRECEDENT
        ("Maneka Gandhi v. Union of India (1978) 1 SCC 248",    "PRECEDENT",
         "maneka gandhi v union of india"),
        ("State of Maharashtra v. Salman Khan AIR 2015 SC 100", "PRECEDENT",
         "state of maharashtra v salman khan"),
        # STATUTE
        ("Indian Penal Code, 1860",                        "STATUTE", "ipc"),
        ("Code of Criminal Procedure",                     "STATUTE", "crpc"),
        ("Protection of Children from Sexual Offences Act","STATUTE", "pocso"),
    ]

    passed = 0
    for raw, label, expected in tests:
        got = canonicalize_entity(raw, label)
        ok  = got == expected
        passed += ok
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {label}: {raw!r}")
        if not ok:
            print(f"         expected: {expected!r}")
            print(f"         got:      {got!r}")

    print(f"\n{passed}/{len(tests)} passed")
