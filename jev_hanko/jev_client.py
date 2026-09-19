"""Jev (TypeSafe System One) client over OpenRouter's Decisions endpoint.

Standard library only. Jev is NOT reachable via /chat/completions; it lives at
/api/alpha/decisions and takes the native TypeSafe body {model, state, questions}.
"""
import json
import os
import time
import urllib.error
import urllib.request

ENDPOINT = "https://openrouter.ai/api/alpha/decisions"
MODEL = "typesafe/jev-1.13"
USD_PER_INPUT_TOKEN = 0.042 / 1_000_000  # output tokens are free


class JevError(RuntimeError):
    pass


def _encode_criteria(criteria):
    """OpenRouter validates criteria values as strings, so rich criteria
    objects ({what, not_for, examples}) are JSON-encoded per value."""
    if criteria is None:
        return None
    if isinstance(criteria, dict):
        return {k: v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                for k, v in criteria.items()}
    if isinstance(criteria, list):
        return [v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
                for v in criteria]
    raise TypeError(f"criteria must be dict, list or None, got {type(criteria).__name__}")


def build_request(state, questions, model=MODEL):
    """Build the request body. `questions` maps your own id -> question object
    with keys: type ('choice' | 'score' | 'noul'), instructions, criteria (optional).
    Question ids are never shown to the model."""
    body_questions = {}
    for qid, q in questions.items():
        if q.get("type") not in ("choice", "score", "noul"):
            raise ValueError(f"question {qid!r}: unknown type {q.get('type')!r}")
        instructions = q["instructions"]
        if not isinstance(instructions, str):
            instructions = json.dumps(instructions, ensure_ascii=False)
        out = {"type": q["type"], "instructions": instructions}
        criteria = _encode_criteria(q.get("criteria"))
        if criteria is not None:
            out["criteria"] = criteria
        body_questions[qid] = out
    return {"model": model, "state": state, "questions": body_questions}


def cost_usd(usage):
    """Prefer the billed cost OpenRouter reports; fall back to the price list."""
    if isinstance(usage.get("cost"), (int, float)):
        return float(usage["cost"])
    return usage.get("input_tokens", 0) * USD_PER_INPUT_TOKEN


def decide(state, questions, model=MODEL, api_key=None, timeout=30):
    """One Jev call. Returns {'answers', 'usage', 'model', 'latency_s', 'cost_usd'}."""
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise JevError("OPENROUTER_API_KEY is not set")
    data = json.dumps(build_request(state, questions, model), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        ENDPOINT, data=data, method="POST",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise JevError(f"jev: {e.code} {e.reason}: {e.read().decode('utf-8', 'replace')[:500]}") from None
    latency = time.perf_counter() - t0
    if "answers" not in payload:
        raise JevError(f"jev: unexpected response: {json.dumps(payload, ensure_ascii=False)[:500]}")
    usage = payload.get("usage", {})
    return {"answers": payload["answers"], "usage": usage, "model": payload.get("model"),
            "latency_s": latency, "cost_usd": cost_usd(usage)}
