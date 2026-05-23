"""Deterministic chunkers. All chunking logic lives here (no LLM)."""
import re


def _slug(s):
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "doc"


def _chunk_record(doc, idx, text, start_in_body, prefix):
    start = doc.body_start_char + start_in_body
    return {
        "chunk_id": f"{_slug(doc.title)}_{prefix}{idx}",
        "doc_title": doc.title,
        "section": doc.section,
        "text": text,
        "start_char": start,
        "end_char": start + len(text),
    }


def paragraph_chunks(doc):
    """Primary strategy. Split on blank-line paragraphs; if none, split per line."""
    body = doc.body
    chunks = []

    if re.search(r"\n\s*\n", body):
        segments = []
        cursor = 0
        for m in re.finditer(r"\n\s*\n", body):
            segments.append((cursor, m.start()))
            cursor = m.end()
        segments.append((cursor, len(body)))
    else:
        segments = []
        cursor = 0
        for m in re.finditer(r"\n", body):
            segments.append((cursor, m.start()))
            cursor = m.end()
        segments.append((cursor, len(body)))

    idx = 1
    for s, e in segments:
        text = body[s:e]
        if not text.strip():
            continue
        chunks.append(_chunk_record(doc, idx, text, s, prefix="p"))
        idx += 1
    return chunks


def fixed_size_chunks(doc, size=300, overlap=50):
    """Comparison strategy. Sliding window over the body."""
    body = doc.body
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("invalid size/overlap")
    chunks = []
    step = size - overlap
    idx = 1
    i = 0
    while i < len(body):
        text = body[i:i + size]
        if text.strip():
            chunks.append(_chunk_record(doc, idx, text, i, prefix="f"))
            idx += 1
        if i + size >= len(body):
            break
        i += step
    return chunks
