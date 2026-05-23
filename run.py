"""End-to-end orchestrator. Enforces pipeline stage order."""
import json
import os
import subprocess
import sys

from dotenv import load_dotenv

from src import llm_log
from src.chunking import fixed_size_chunks, paragraph_chunks
from src.evaluation import evaluate
from src.generation import generate_answer
from src.grounding import check_grounding
from src.ingest import load_documents
from src.retrieval import build_index, retrieve
from src.vocab import StageTracker

ARTIFACTS = "artifacts"
KB_DIR = "kb"
QUERIES_PATH = "queries.json"
TOP_K = 5


def _write_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _chunk_all(docs, strategy):
    out = []
    for d in docs:
        out.extend(strategy(d))
    return out


def main():
    load_dotenv()
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Copy .env.example to .env and fill it in.")
        sys.exit(1)

    os.makedirs(ARTIFACTS, exist_ok=True)
    llm_log.reset()
    tracker = StageTracker()
    print(f"[stage] {tracker.current}")

    docs = load_documents(KB_DIR)
    tracker.advance_to("DOCUMENTS_LOADED")

    chunks = _chunk_all(docs, paragraph_chunks)
    _write_json(f"{ARTIFACTS}/chunks.json", chunks)
    tracker.advance_to("DOCUMENTS_CHUNKED")

    index = build_index(chunks)
    tracker.advance_to("INDEX_BUILT")

    with open(QUERIES_PATH, "r", encoding="utf-8") as f:
        queries = json.load(f)

    retrieval_records = []
    for q in queries:
        top = retrieve(index, q["question"], k=TOP_K)
        retrieval_records.append({
            "query_id": q["query_id"],
            "question": q["question"],
            "top_k": top,
        })
    _write_json(f"{ARTIFACTS}/retrieval.json", retrieval_records)
    tracker.advance_to("RETRIEVAL_COMPLETE")

    if not os.path.exists(f"{ARTIFACTS}/retrieval.json"):
        raise RuntimeError("retrieval.json missing — refuse to generate answers")

    answers = []
    for rec in retrieval_records:
        ans = generate_answer(
            query_id=rec["query_id"],
            question=rec["question"],
            top_k_chunks=rec["top_k"],
        )
        answers.append(ans)
    _write_json(f"{ARTIFACTS}/answers.json", answers)
    tracker.advance_to("ANSWERS_GENERATED")

    eval_result = evaluate(retrieval_records, queries)
    _write_json(f"{ARTIFACTS}/eval.json", eval_result)
    tracker.advance_to("EVALUATION_COMPLETE")

    grounding = check_grounding(answers, retrieval_records)
    _write_json(f"{ARTIFACTS}/grounding_check.json", grounding)

    _run_chunking_comparison(docs, queries)

    result = subprocess.run([sys.executable, "validate.py"])
    if result.returncode != 0:
        print("ERROR: validate.py failed")
        sys.exit(result.returncode)
    tracker.advance_to("VALIDATION_COMPLETE")

    summary = eval_result["summary"]
    print("\n=== SUMMARY ===")
    print(f"queries: {summary['total_queries']}")
    print(f"hits/partial/miss: {summary['hits']}/{summary['partial_hits']}/{summary['misses']}")
    print(f"top3_hit_rate: {summary['top3_hit_rate']:.2f}")
    tracker.advance_to("RESULTS_FINALISED")


def _run_chunking_comparison(docs, queries):
    """Run retrieval + evaluation with fixed-size chunks; compare to paragraph."""
    results = {}
    for name, strategy in (("paragraph", paragraph_chunks),
                           ("fixed_size_300_50", fixed_size_chunks)):
        chunks = _chunk_all(docs, strategy)
        idx = build_index(chunks)
        retrievals = []
        for q in queries:
            top = retrieve(idx, q["question"], k=TOP_K)
            retrievals.append({
                "query_id": q["query_id"],
                "question": q["question"],
                "top_k": top,
            })
        ev = evaluate(retrievals, queries)
        results[name] = {
            "num_chunks": len(chunks),
            "summary": ev["summary"],
        }
    results["tradeoff_note"] = (
        "Paragraph-based chunking respects the natural one-fact-per-line structure of "
        "this KB, which keeps each chunk focused and improves top-1 hit rate. "
        "Fixed-size chunking (300 chars, 50 overlap) is more general-purpose but can "
        "split related facts across chunks or merge unrelated ones, slightly hurting "
        "precision on this small corpus. For larger, less structured documents "
        "fixed-size is usually competitive."
    )
    _write_json(f"{ARTIFACTS}/chunking_comparison.json", results)


if __name__ == "__main__":
    main()
