"""Validates that all required artifacts exist, are well-formed, and consistent."""
import json
import os
import sys
from numbers import Number

from src.vocab import ANSWER_LABELS, RETRIEVAL_STATUSES

ARTIFACTS = "artifacts"
REQUIRED = [
    f"{ARTIFACTS}/chunks.json",
    f"{ARTIFACTS}/retrieval.json",
    f"{ARTIFACTS}/answers.json",
    f"{ARTIFACTS}/eval.json",
    f"{ARTIFACTS}/grounding_check.json",
    f"{ARTIFACTS}/chunking_comparison.json",
    "llm_calls.jsonl",
]


class Checks:
    def __init__(self):
        self.failures = []

    def expect(self, cond, msg):
        if not cond:
            self.failures.append(msg)

    def report(self):
        if self.failures:
            print(f"VALIDATION FAILED ({len(self.failures)} issue(s)):")
            for m in self.failures:
                print(f"  - {m}")
            return 1
        print("VALIDATION PASSED: all checks satisfied.")
        return 0


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    c = Checks()

    for p in REQUIRED:
        c.expect(os.path.exists(p), f"missing artifact: {p}")
    if c.failures:
        return c.report()

    try:
        chunks = _load_json(f"{ARTIFACTS}/chunks.json")
        retrieval = _load_json(f"{ARTIFACTS}/retrieval.json")
        answers = _load_json(f"{ARTIFACTS}/answers.json")
        eval_data = _load_json(f"{ARTIFACTS}/eval.json")
        _ = _load_json(f"{ARTIFACTS}/grounding_check.json")
        _ = _load_json(f"{ARTIFACTS}/chunking_comparison.json")
    except json.JSONDecodeError as e:
        c.expect(False, f"invalid JSON: {e}")
        return c.report()

    with open("llm_calls.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                c.expect(False, f"llm_calls.jsonl line {i}: invalid JSON ({e})")

    with open("queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    query_ids = {q["query_id"] for q in queries}

    retrieval_ids = {r["query_id"] for r in retrieval}
    answer_ids = {a["query_id"] for a in answers}
    c.expect(query_ids <= retrieval_ids,
             f"retrieval.json missing queries: {query_ids - retrieval_ids}")
    c.expect(query_ids <= answer_ids,
             f"answers.json missing queries: {query_ids - answer_ids}")

    for r in retrieval:
        c.expect(len(r["top_k"]) >= 3,
                 f"query {r['query_id']}: fewer than 3 retrieved chunks")
        for item in r["top_k"]:
            c.expect(isinstance(item.get("score"), Number),
                     f"query {r['query_id']} chunk {item.get('chunk_id')}: non-numeric score")

    retrieved_by_qid = {r["query_id"]: {c2["chunk_id"] for c2 in r["top_k"]} for r in retrieval}
    for a in answers:
        c.expect(a.get("answer_label") in ANSWER_LABELS,
                 f"query {a['query_id']}: invalid answer_label {a.get('answer_label')!r}")
        if a.get("answer_label") == "grounded_answer":
            c.expect(len(a.get("citations", [])) >= 1,
                     f"query {a['query_id']}: grounded_answer without citations")
        retrieved_ids = retrieved_by_qid.get(a["query_id"], set())
        import re
        for cit in a.get("citations", []):
            m = re.search(r"§\s*([^\[\]]+?)\]", cit)
            c.expect(m is not None, f"query {a['query_id']}: malformed citation {cit!r}")
            if m:
                c.expect(m.group(1).strip() in retrieved_ids,
                         f"query {a['query_id']}: citation {cit!r} references non-retrieved chunk")

    for ev in eval_data.get("per_query", []):
        c.expect(ev.get("retrieval_status") in RETRIEVAL_STATUSES,
                 f"query {ev.get('query_id')}: invalid retrieval_status {ev.get('retrieval_status')!r}")

    summary = eval_data.get("summary", {})
    for key in ("top3_hit_rate", "total_queries", "hits", "partial_hits", "misses"):
        c.expect(key in summary, f"eval.json summary missing key: {key}")

    c.expect(isinstance(chunks, list) and len(chunks) > 0, "chunks.json empty")

    return c.report()


if __name__ == "__main__":
    sys.exit(main())
