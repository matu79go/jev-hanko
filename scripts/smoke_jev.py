"""Smoke test: ONE paid Jev call (~400 input tokens, well under $0.0001).

Usage:  OPENROUTER_API_KEY=... python3 scripts/smoke_jev.py
Confirms the Decisions endpoint shape and the response fields for all three primitives
(noul / choice / score).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jev_hanko.jev_client import decide  # noqa: E402

STATE = {
    "clause": (
        "Either party may terminate this Agreement at any time, for any reason or no reason, "
        "upon ninety (90) days' prior written notice to the other party."
    ),
    "contract_type": "Master Services Agreement",
}

QUESTIONS = {
    "termination_for_convenience": {
        "type": "noul",
        "instructions": "Does `clause` let a party walk away without cause?",
        "criteria": {
            "true": "Either or one party may terminate for convenience, with or without a notice period",
            "false": "Termination only for cause, for breach, on expiry, or not addressed at all",
        },
    },
    "who_may_terminate": {
        "type": "choice",
        "instructions": "Who is given the right to terminate in `clause`?",
        "criteria": {
            "either": "Both parties have the same right",
            "customer": "Only the customer / buyer side",
            "supplier": "Only the supplier / vendor side",
            "none": "No termination right is granted here",
        },
    },
    "notice_burden": {
        "type": "score",
        "instructions": "How much advance notice does `clause` demand of the terminating party?",
        "criteria": [
            "no notice required, effective immediately",
            "a short notice period, under 60 days",
            "a long notice period, 60 days or more",
        ],
    },
}

if __name__ == "__main__":
    r = decide(STATE, QUESTIONS)
    print(json.dumps(r["answers"], ensure_ascii=False, indent=2))
    print(f"model={r['model']} usage={r['usage']} latency={r['latency_s']:.3f}s cost=${r['cost_usd']:.7f}")
