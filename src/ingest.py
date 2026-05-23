"""Load KB documents from disk, parsing Title: and Section: headers."""
import os
import re
from dataclasses import dataclass


@dataclass
class Document:
    filename: str
    title: str
    section: str
    body: str
    body_start_char: int  # offset of body within the original file


_TITLE_RE = re.compile(r"^\s*Title\s*:\s*(.+?)\s*$", re.IGNORECASE)
_SECTION_RE = re.compile(r"^\s*Section\s*:\s*(.+?)\s*$", re.IGNORECASE)


def load_documents(kb_dir="kb"):
    docs = []
    for name in sorted(os.listdir(kb_dir)):
        if not name.lower().endswith(".txt"):
            continue
        path = os.path.join(kb_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
        docs.append(_parse(name, raw))
    if not docs:
        raise RuntimeError(f"No .txt documents found in {kb_dir}/")
    return docs


def _parse(filename, raw):
    title = None
    section = None
    lines = raw.splitlines(keepends=True)
    consumed = 0
    i = 0
    while i < len(lines) and (title is None or section is None):
        line = lines[i]
        m_t = _TITLE_RE.match(line)
        m_s = _SECTION_RE.match(line)
        if m_t and title is None:
            title = m_t.group(1).strip()
            consumed += len(line)
            i += 1
            continue
        if m_s and section is None:
            section = m_s.group(1).strip()
            consumed += len(line)
            i += 1
            continue
        if line.strip() == "":
            consumed += len(line)
            i += 1
            continue
        break

    while i < len(lines) and lines[i].strip() == "":
        consumed += len(lines[i])
        i += 1

    body = "".join(lines[i:])
    if title is None:
        title = os.path.splitext(filename)[0]
    if section is None:
        section = ""
    return Document(
        filename=filename,
        title=title,
        section=section,
        body=body,
        body_start_char=consumed,
    )
