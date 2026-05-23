"""Smoke-test the local Flask API at POST /answer.

Usage:
    # In one terminal:
    python api.py
    # In another:
    python scripts/test_api.py
    python scripts/test_api.py "Your custom question here"
"""
import json
import sys
import urllib.error
import urllib.request

URL = "http://127.0.0.1:5000/answer"

DEFAULT_QUESTIONS = [
    "How long can a bank withdrawal take after approval?",
    "My reset link expired after an hour, is that expected?",
    "Can support tell me my current password?",
    "Will a utility bill from 8 months ago work for proof of address?",
    "Can I withdraw profit from a demo account?",
    "What is the capital of France?",  # should be insufficient_context
]


def ask(question):
    payload = json.dumps({"question": question}).encode("utf-8")
    req = urllib.request.Request(
        URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            return resp.status, json.loads(body)
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode("utf-8", errors="replace")}
    except urllib.error.URLError as e:
        return None, {"error": f"connection failed: {e.reason}. Is `python api.py` running?"}


def main():
    questions = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_QUESTIONS

    print(f"POST {URL}\n")
    failures = 0
    for i, q in enumerate(questions, start=1):
        print(f"[{i}] Q: {q}")
        status, body = ask(q)
        if status is None or status >= 400:
            failures += 1
            print(f"    FAIL (status={status}): {body}\n")
            continue
        print(f"    status: {status}")
        print(f"    label : {body.get('answer_label')}")
        print(f"    answer: {body.get('answer')}")
        cites = body.get("citations") or []
        print(f"    cites : {cites if cites else '(none)'}")
        print()

    if failures:
        print(f"{failures}/{len(questions)} request(s) failed.")
        sys.exit(1)
    print(f"All {len(questions)} request(s) succeeded.")


if __name__ == "__main__":
    main()
