"""Smoke test: ONE paid Jev call (~400 input tokens, well under $0.0001).

Usage:  OPENROUTER_API_KEY=... python3 scripts/smoke_jev.py
Confirms the Decisions endpoint shape, Japanese handling, and the response fields
for all three primitives (noul / choice / score).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_hanko.jev_client import decide  # noqa: E402

STATE = {
    "invoice_line": {"品目": "本みりん 1.8L", "数量": 2, "適用税率": "8%"},
    "摘要": "品代一式",
}

QUESTIONS = {
    "reduced_rate_ok": {
        "type": "noul",
        "instructions": "`invoice_line.品目` は日本の消費税の軽減税率(8%)の対象か",
        "criteria": {
            "true": "酒類を除く飲食料品。みりん風調味料(アルコール1%未満)は対象",
            "false": "酒税法上の酒類(アルコール1度以上。本みりん・料理酒を含む)、外食、ケータリング、医薬部外品",
        },
    },
    "account": {
        "type": "choice",
        "instructions": "`invoice_line.品目` を社内で購入した場合の勘定科目",
        "criteria": {
            "消耗品費": "事務用品・日用品などの少額物品",
            "会議費": "会議・打合せに伴う飲食",
            "交際費": "取引先への接待・贈答",
            "その他": "上のどれにも当てはまらない",
        },
    },
    "specificity": {
        "type": "score",
        "instructions": "`摘要` の記載から取引内容をどこまで特定できるか",
        "criteria": ["何の取引か特定できない(一式・諸経費など)", "分野は分かるが品目・役務は不明", "品目または役務が特定できる"],
    },
}

if __name__ == "__main__":
    r = decide(STATE, QUESTIONS)
    print(json.dumps(r["answers"], ensure_ascii=False, indent=2))
    print(f"model={r['model']} usage={r['usage']} latency={r['latency_s']:.3f}s cost=${r['cost_usd']:.7f}")
