"""Append-only logger for llm_calls.jsonl."""
import hashlib
import json
from datetime import datetime, timezone

LOG_PATH = "llm_calls.jsonl"


def log_call(stage, query_id, prompt, model, input_artifacts, output_artifact,
             provider="openai", path=LOG_PATH):
    record = {
        "stage": stage,
        "query_id": query_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "prompt_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
        "input_artifacts": list(input_artifacts),
        "output_artifact": output_artifact,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def reset(path=LOG_PATH):
    open(path, "w", encoding="utf-8").close()
