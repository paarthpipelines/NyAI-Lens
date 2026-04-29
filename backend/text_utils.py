"""
Shared text-cleaning utilities for NyAI-Lens.

These helpers were previously copy-pasted across backend/main.py,
backend/chunker.py, and frontend/app.py — each copy had subtle
divergences.  This module is the single source of truth.

Import from here instead of duplicating:

    from backend.text_utils import strip_page_numbers, _is_page_line, _DIGITAL_SIG_RE
"""
from __future__ import annotations

import re

# ── Page-number line detection ─────────────────────────────────────────────────
_PAGE_PATTERNS = (
    re.compile(r'\d+'),
    re.compile(r'-\s*\d+\s*-'),
    re.compile(r'\|\s*\d+\s*\|'),
    re.compile(r'\d+\s+of\s+\d+'),
    re.compile(r'(?:page|pg\.?|p\.)\s*\d+(?:\s*[/]\s*\d+)?', re.I),
    re.compile(r'(?:page|pg\.?|p\.)\s*\d+\s+of\s+\d+', re.I),
    re.compile(r'(?:i{1,3}|iv|vi{0,3}|ix|xi{0,3}|xiv|xv|xvi{0,3}|xix|xx)'),
    re.compile(r'.+\s+(?:page|pg\.?|p\.)\s*\d+(?:\s+of\s+\d+)?', re.I),
    # Standalone court case-header lines: any prefix + "No.? (s)? <digits> of <YYYY>"
    # fullmatch ensures this never fires on citations embedded inside sentences.
    re.compile(r'.+\bNo\.?\s*(?:\(s\)\.?)?\s*[\d,\s/@-]+of\s+\d{4}', re.I),
)

_ROMAN_PAGE_RE = re.compile(
    r'^(?:i{1,3}|iv|vi{0,3}|ix|xi{0,3}|xiv|xv|xvi{0,3}|xix|xx)$',
    re.IGNORECASE,
)

_DIGITAL_SIG_RE = re.compile(
    r'Signature\s+Not\s+Verified\s+Digitally\s+signed\s+by\s+[^\n]{1,120}\n'
    r'.{0,300}?Reason:\s*',
    re.DOTALL | re.IGNORECASE,
)

_FOOTER_SIG_RE = re.compile(
    r'^\*?\*?[.\u2026]{5,}[\s.,\u2026]*J\.\*?\*?$',
    re.IGNORECASE,
)


def _is_page_line(line: str) -> bool:
    """Return True if *line* looks like a bare page number / page marker."""
    s = line.strip()
    had_bold = '**' in s or '__' in s
    s = re.sub(r'^\*+|\*+$', '', s).strip()
    s = re.sub(r'^_+|_+$', '', s).strip()
    if not s:
        return False
    # Bold roman numerals are chapter headings, not page numbers.
    if had_bold and _ROMAN_PAGE_RE.fullmatch(s):
        return False
    return any(p.fullmatch(s) for p in _PAGE_PATTERNS)


def strip_page_numbers(text: str) -> str:
    """Remove bare page-number lines and collapse excessive blank lines.

    When a page marker falls between two table rows (lines starting with '|'),
    the surrounding blank lines are also removed so the table stays contiguous.
    All other markdown is left completely untouched.
    """
    lines = text.split('\n')
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not _is_page_line(line):
            result.append(line)
            i += 1
            continue

        # Page marker detected — check whether it sits between two table rows.
        # Look back through result (past any blank lines) for a table row.
        j = len(result) - 1
        while j >= 0 and result[j].strip() == '':
            j -= 1
        prev_is_table = j >= 0 and result[j].strip().startswith('|')

        # Look forward through remaining lines (past any blank lines) for a table row.
        k = i + 1
        while k < len(lines) and lines[k].strip() == '':
            k += 1
        next_is_table = k < len(lines) and lines[k].strip().startswith('|')

        if prev_is_table and next_is_table:
            # Strip the blank lines already written after the last table row,
            # skip the page marker and any blank lines before the next table row,
            # so the two table segments become contiguous.
            while result and result[-1].strip() == '':
                result.pop()
            i = k  # resume from the next table row; the marker is discarded
        else:
            # Not between table rows — discard only the marker line itself.
            i += 1

    cleaned = '\n'.join(result)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()


def clean_markdown(md: str) -> str:
    """Strip digital-signature blocks and page-number lines from raw markdown."""
    md = _DIGITAL_SIG_RE.sub('', md)
    return strip_page_numbers(md)
