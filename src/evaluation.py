"""Deterministic retrieval evaluation against queries.json ground truth."""


def evaluate(retrieval_records, queries):
    by_qid = {q["query_id"]: q for q in queries}
    per_query = []
    hits = partial_hits = misses = 0

    for rec in retrieval_records:
        qid = rec["query_id"]
        gt = by_qid.get(qid, {})
        expected = gt.get("expected_doc_titles", []) or []
        top3 = rec["top_k"][:3]
        top3_titles = [c["doc_title"] for c in top3]

        status = "miss"
        matched = False
        explanation = ""

        if expected:
            rank_found = None
            matched_title = None
            for i, title in enumerate(top3_titles, start=1):
                if title in expected:
                    rank_found = i
                    matched_title = title
                    break
            if rank_found == 1:
                status = "hit"
                matched = True
                explanation = f"Expected title '{matched_title}' found at rank 1"
            elif rank_found is not None:
                status = "partial_hit"
                matched = True
                explanation = f"Expected title '{matched_title}' found at rank {rank_found}"
            else:
                status = "miss"
                explanation = (
                    f"None of {expected} found in top-3 {top3_titles}"
                )
        else:
            explanation = "No expected_doc_titles in ground truth"

        if status == "hit":
            hits += 1
        elif status == "partial_hit":
            partial_hits += 1
        else:
            misses += 1

        per_query.append({
            "query_id": qid,
            "expected_doc_titles": expected,
            "retrieved_doc_titles_top3": top3_titles,
            "retrieval_status": status,
            "matched_expected_title": matched,
            "explanation": explanation,
        })

    total = len(per_query) or 1
    summary = {
        "top3_hit_rate": (hits + partial_hits) / total,
        "total_queries": len(per_query),
        "hits": hits,
        "partial_hits": partial_hits,
        "misses": misses,
    }
    return {"per_query": per_query, "summary": summary}
