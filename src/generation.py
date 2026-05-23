"""Citation-strict answer generation via OpenAI gpt-4o-mini."""
import json
import os
import re

from openai import OpenAI

from . import llm_log
from .vocab import ANSWER_LABELS

_DEFAULT_MODEL = "gpt-4o-mini"
_client = None


def _get_model():
    return os.environ.get("OPENAI_MODEL", _DEFAULT_MODEL)


def _get_client():
    global _client
    if _client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        _client = OpenAI()
    return _client


_SYSTEM = (
    "You answer questions using ONLY the provided chunks. "
    "Do not use outside knowledge. Be concise."
)

_INSTRUCTIONS = """\
Rules:
- If the chunks do not contain the answer, set label="insufficient_context".
- If the chunks contradict each other on the question, set label="conflicting_context".
- Otherwise set label="grounded_answer" and cite every factual claim as
  [doc_title §chunk_id] using the EXACT doc_title and chunk_id strings shown.
- "citations" must be a list of full "[doc_title §chunk_id]" strings.
- "used_chunk_ids" must list only the chunk_id values you cited.
- Respond ONLY as JSON with keys: label, answer, citations, used_chunk_ids.
"""


def _format_chunks(top_k):
    lines = []
    for c in top_k:
        text = c["chunk_text"].strip().replace("\n", " ")
        lines.append(
            f'[chunk_id={c["chunk_id"]} | doc_title={c["doc_title"]}] "{text}"'
        )
    return "\n".join(lines)


def _build_prompt(question, top_k):
    return (
        _INSTRUCTIONS
        + "\nChunks:\n"
        + _format_chunks(top_k)
        + f"\n\nQuestion: {question}\n"
    )


_CITATION_RE = re.compile(r"§\s*([^\[\]§]+?)\s*(?:\]|$)")


def _extract_chunk_id(citation):
    """Pull chunk_id out of a citation string, tolerant of bracket/format drift."""
    if not isinstance(citation, str):
        return None
    m = _CITATION_RE.search(citation)
    if m:
        return m.group(1).strip()
    return None


def generate_answer(query_id, question, top_k_chunks, stage="ANSWERS_GENERATED",
                    output_artifact="artifacts/answers.json"):
    client = _get_client()
    prompt = _build_prompt(question, top_k_chunks)

    completion = client.chat.completions.create(
        model=_get_model(),
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    raw = completion.choices[0].message.content or "{}"

    llm_log.log_call(
        stage=stage,
        query_id=query_id,
        prompt=prompt,
        model=_get_model(),
        input_artifacts=["artifacts/retrieval.json"],
        output_artifact=output_artifact,
    )

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    label = data.get("label")
    if label not in ANSWER_LABELS:
        label = "insufficient_context"

    answer_text = data.get("answer", "") or ""
    citations_in = data.get("citations", []) or []
    used_in = data.get("used_chunk_ids", []) or []

    retrieved_ids = {c["chunk_id"] for c in top_k_chunks}
    id_to_title = {c["chunk_id"]: c["doc_title"] for c in top_k_chunks}

    cited_ids = []
    for cit in citations_in:
        cid = _extract_chunk_id(cit)
        if cid and cid in retrieved_ids and cid not in cited_ids:
            cited_ids.append(cid)

    for cid in used_in:
        if isinstance(cid, str) and cid in retrieved_ids and cid not in cited_ids:
            cited_ids.append(cid)

    valid_citations = [f"[{id_to_title[cid]} §{cid}]" for cid in cited_ids]
    valid_chunk_ids = list(cited_ids)

    if label == "grounded_answer" and not valid_citations:
        label = "insufficient_context"
        answer_text = answer_text or "The retrieved context does not contain the answer."
        valid_citations = []
        valid_chunk_ids = []

    return {
        "query_id": query_id,
        "answer_label": label,
        "answer": answer_text,
        "citations": valid_citations,
        "used_chunk_ids": valid_chunk_ids,
    }
