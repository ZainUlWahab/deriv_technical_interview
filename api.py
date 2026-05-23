"""Flask app exposing POST /answer."""
import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from src.chunking import paragraph_chunks
from src.generation import generate_answer
from src.ingest import load_documents
from src.retrieval import build_index, retrieve

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not set (check .env)")

app = Flask(__name__)

_docs = load_documents("kb")
_chunks = []
for d in _docs:
    _chunks.extend(paragraph_chunks(d))
_index = build_index(_chunks)


@app.post("/answer")
def answer():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question")
    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "missing 'question' field"}), 400

    top = retrieve(_index, question, k=5)
    result = generate_answer(
        query_id="api",
        question=question,
        top_k_chunks=top,
        stage="API_ANSWER",
    )
    return jsonify({
        "answer_label": result["answer_label"],
        "answer": result["answer"],
        "citations": result["citations"],
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)
