"""Deterministic grounding validator: does each citation's chunk support the answer?"""
import re

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "and", "or", "but", "if", "then",
    "with", "by", "at", "as", "from", "this", "that", "these", "those",
    "it", "its", "i", "you", "we", "they", "he", "she", "him", "her",
    "do", "does", "did", "have", "has", "had", "can", "could", "may",
    "might", "will", "would", "should", "shall", "must", "not", "no",
    "yes", "than", "so", "such", "any", "all", "some", "more", "most",
    "less", "least", "what", "which", "who", "when", "where", "why",
    "how", "about", "into", "out", "up", "down", "over", "under",
    "again", "further", "once", "here", "there",
}


def _tokens(text):
    raw = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in raw if t not in _STOPWORDS and len(t) > 1}


_CITATION_RE = re.compile(r"\[([^\[\]§]+?)\s*§\s*([^\[\]]+?)\]")


def check_grounding(answers, retrieval_records):
    by_qid = {r["query_id"]: r for r in retrieval_records}
    results = []
    for ans in answers:
        qid = ans["query_id"]
        ret = by_qid.get(qid, {"top_k": []})
        chunk_by_id = {c["chunk_id"]: c for c in ret.get("top_k", [])}

        per_citation = []
        all_supported = True
        any_checked = False

        if ans["answer_label"] == "grounded_answer":
            answer_tokens = _tokens(ans.get("answer", ""))
            for cit in ans.get("citations", []):
                m = _CITATION_RE.search(cit)
                if not m:
                    per_citation.append({
                        "citation": cit, "exists_in_retrieval": False,
                        "supported": False, "overlap_count": 0,
                    })
                    all_supported = False
                    any_checked = True
                    continue
                chunk_id = m.group(2).strip()
                chunk = chunk_by_id.get(chunk_id)
                if chunk is None:
                    per_citation.append({
                        "citation": cit, "exists_in_retrieval": False,
                        "supported": False, "overlap_count": 0,
                    })
                    all_supported = False
                    any_checked = True
                    continue
                chunk_tokens = _tokens(chunk["chunk_text"])
                overlap = answer_tokens & chunk_tokens
                supported = len(overlap) >= 2
                per_citation.append({
                    "citation": cit,
                    "exists_in_retrieval": True,
                    "supported": supported,
                    "overlap_count": len(overlap),
                })
                any_checked = True
                if not supported:
                    all_supported = False
        else:
            all_supported = True  # nothing to ground

        results.append({
            "query_id": qid,
            "answer_label": ans["answer_label"],
            "citations_checked": per_citation,
            "all_citations_supported": all_supported if any_checked or ans["answer_label"] != "grounded_answer" else False,
        })
    return results
