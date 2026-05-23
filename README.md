# Mini-RAG Pipeline — Deriv Technical Interview

**Author:** Zain Ul Wahab

A small, replayable retrieval-augmented-generation pipeline built for the Deriv
technical interview. Reads `kb/*.txt` and `queries.json`, builds a
sentence-transformers index, retrieves top-k chunks per query, asks
`gpt-4o-mini` for a citation-strict answer, then deterministically evaluates
retrieval quality and grounding.

## Setup

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
# edit .env and set OPENAI_API_KEY=sk-...
# OPENAI_MODEL defaults to gpt-4o-mini
```

First run downloads the `all-MiniLM-L6-v2` model (~90 MB).

## Run the pipeline

```powershell
python run.py
```

This writes:

- `artifacts/chunks.json`
- `artifacts/retrieval.json`
- `artifacts/answers.json`
- `artifacts/eval.json`
- `artifacts/grounding_check.json`
- `artifacts/chunking_comparison.json`
- `llm_calls.jsonl`

…and finishes with a summary printed to stdout.

## Validate

```powershell
python validate.py
```

Pure JSON checks — no API calls. Exits non-zero with a list of failures if anything
is missing, malformed, or violates the controlled vocabularies.

## Stretch: Flask API

```powershell
python api.py
```

Then in another terminal:

```powershell
python scripts/test_api.py
```

Or hit it directly:

```powershell
curl -X POST http://localhost:5000/answer ^
     -H "Content-Type: application/json" ^
     -d "{\"question\":\"Can I withdraw demo profit?\"}"
```

## Sample outputs

### `python run.py`

<img width="2032" height="478" alt="run_dot_py_output" src="https://github.com/user-attachments/assets/ef4778c5-abc6-415a-9fcf-caba923fe83c" />


### `python validate.py`

<img width="610" height="90" alt="run_validate_py_output" src="https://github.com/user-attachments/assets/677da7c1-9ea0-451a-bb2b-b2455db87479" />


### `python scripts/test_api.py`

<img width="883" height="866" alt="testing_api_output" src="https://github.com/user-attachments/assets/dd80ff9c-9acc-4a3c-ae8c-567bd52fbd3d" />


## Design notes

- **Chunking is in code.** `src/chunking.py` splits each document by paragraph
  (primary) or fixed-size sliding window (used for the comparison artifact). The
  LLM never decides chunk boundaries.
- **Retrieval is in code.** `src/retrieval.py` uses
  `sentence-transformers/all-MiniLM-L6-v2` to embed chunks and queries, normalizes
  the vectors, and ranks by cosine similarity using NumPy. No external vector DB
  — the corpus is small enough that an in-memory matrix is the right call.
- **Evaluation is deterministic.** `src/evaluation.py` checks expected document
  titles against the top-3 retrieved titles and assigns `hit` / `partial_hit` /
  `miss`. No LLM involvement.
- **Grounded answers only.** The LLM is told to return `insufficient_context` if
  the retrieved chunks don't cover the question, and `conflicting_context` if they
  contradict each other. Citations are post-validated in code: any citation
  pointing at a non-retrieved chunk is dropped, and a `grounded_answer` with no
  surviving citations is downgraded to `insufficient_context`. Citation parsing
  is format-tolerant — we key off the `§<chunk_id>` segment and re-emit citations
  in canonical `[doc_title §chunk_id]` form.
- **Filename independence.** The pipeline iterates `kb/*.txt` (sorted) and does
  not rely on `article_01.txt` etc.; rename or replace freely.
- **Stage enforcement.** `src/vocab.py::StageTracker` raises if stages are skipped
  or reordered; `run.py` walks every stage in sequence.

## LLM usage

`gpt-4o-mini` (or whatever `OPENAI_MODEL` is set to in `.env`) is used only for
answer generation — one call per query, plus one per `/answer` API request. Each
call is logged to `llm_calls.jsonl` with stage, `query_id`, timestamp, model,
`prompt_hash`, input artifacts, and output artifact. The pipeline cannot run
without `OPENAI_API_KEY`.

## Repo layout

```
kb/                 input documents
queries.json        input queries + ground truth
artifacts/          generated outputs
src/                pipeline modules
scripts/            dev tooling (API smoke test)
run.py              orchestrator
validate.py         artifact validator
api.py              Flask POST /answer (stretch)
```

## Remarks

Thank you for the opportunity. Hope to hear from you soon!
