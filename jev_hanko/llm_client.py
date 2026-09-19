"""比較用: 普通の LLM を OpenRouter の chat/completions で呼ぶ(標準ライブラリのみ)。

Jev と同じ判定基準の文章を渡し、選択肢の1つと自信(0-1)だけを返させる。thinking は使わない。
"""
import json
import os
import re
import time
import urllib.error
import urllib.request

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


class LLMError(RuntimeError):
    pass


def build_messages(instructions, criteria, state):
    options = "\n".join(f"- {k}: {v}" for k, v in criteria.items())
    system = (f"{instructions}\n\n選択肢と基準:\n{options}\n\n"
              "出力は1行だけ。形式: <選択肢の名前>|<自信を0から1の小数で>\n説明や前置きは書かない。")
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(state, ensure_ascii=False)}]


def parse_answer(text, labels):
    """'委託料|0.9' のような出力を読む。ラベルが読めなければ (None, 0.0)。"""
    text = (text or "").strip()
    head, _, tail = text.partition("|")
    label = next((lab for lab in sorted(labels, key=len, reverse=True) if lab in head), None)
    if label is None:
        label = next((lab for lab in sorted(labels, key=len, reverse=True) if lab in text), None)
    m = re.search(r"[01](?:\.\d+)?", tail)
    conf = min(max(float(m.group()), 0.0), 1.0) if m else 0.0
    return label, conf


def chat(model, messages, max_tokens=32, reasoning=None, api_key=None, timeout=60, extra=None):
    api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise LLMError("OPENROUTER_API_KEY is not set")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "usage": {"include": True}}
    if reasoning is not None:
        body["reasoning"] = reasoning
    if extra:
        body.update(extra)
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body, ensure_ascii=False).encode("utf-8"), method="POST",
                                 headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMError(f"llm: {e.code} {e.reason}: {e.read().decode('utf-8', 'replace')[:300]}") from None
    latency = time.perf_counter() - t0
    if "choices" not in payload:
        raise LLMError(f"llm: unexpected response: {json.dumps(payload, ensure_ascii=False)[:300]}")
    usage = payload.get("usage", {})
    return {"text": payload["choices"][0]["message"].get("content") or "", "usage": usage,
            "latency_s": latency, "cost_usd": float(usage.get("cost") or 0.0), "model": payload.get("model")}
