"""CUAD: for each contract excerpt, run all 41 clause checks in one shot and score them against
the lawyers' annotations.

Usage: OPENROUTER_API_KEY=... python3 scripts/eval_cuad.py <CUADv1.json> --pos 250 --rand 250 --calib 100 \
         --llm google/gemini-2.5-flash-lite anthropic/claude-haiku-4.5 \
         --llm-full google/gemini-2.5-flash-lite --llm-small anthropic/claude-sonnet-5 --small-n 200
  --llm       : make the model answer with just the IDs of the clauses present (the format most
                favourable to a chat LLM)
  --llm-full  : make the model give a probability for all 41 clauses (same information as Jev)
  --llm-small : for expensive models; measure only the first small-n test excerpts (IDs-only format)
  --calib     : pick a single Jev threshold on excerpts from contracts held out from the test set
                (N positive-bearing excerpts + N random ones)
Responses are cached under cache/, so re-runs cost nothing.
"""
import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jev_hanko.cuad_task import categories, jev_questions, llm_system_prompt, parse_llm, sample_windows  # noqa: E402
from jev_hanko.jev_client import MODEL, JevError, build_request, decide  # noqa: E402
from jev_hanko.llm_client import LLMError, chat  # noqa: E402

# Per-model settings that disable or minimise thinking. Output token counts are printed in the
# results table so this can be verified.
MODEL_OPTS = {
    "openai/gpt-oss-20b:nitro": {"reasoning": {"effort": "low"}, "max_tokens_bonus": 400},
    "qwen/qwen3.7-flash": {"reasoning": {"enabled": False}},
    "deepseek/deepseek-v4-flash": {"reasoning": {"enabled": False}},
}


def _cached(path, fn):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            r = fn()
            path.write_text(json.dumps(r, ensure_ascii=False), encoding="utf-8")
            return r
        except (JevError, LLMError, OSError):
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def run_jev(w, questions):
    state = {"excerpt": w["text"]}
    key = hashlib.sha256(json.dumps(build_request(state, questions, MODEL), sort_keys=True).encode()).hexdigest()[:24]
    r = _cached(ROOT / "cache" / "jev" / f"{key}.json", lambda: decide(state, questions, timeout=60))
    probs = {int(qid[1:]): a["noul"] for qid, a in r["answers"].items()}
    return probs, r["latency_s"], r["cost_usd"], r["usage"].get("output_tokens", 0)


def run_llm(model, w, system, max_tokens, tag):
    opts = MODEL_OPTS.get(model, {})
    messages = [{"role": "system", "content": system}, {"role": "user", "content": w["text"]}]
    key = hashlib.sha256(json.dumps([model, messages]).encode()).hexdigest()[:24]
    r = _cached(ROOT / "cache" / "llm" / f"{model.replace('/', '__').replace(':', '_')}__{tag}" / f"{key}.json",
                lambda: chat(model, messages, max_tokens=max_tokens + opts.get("max_tokens_bonus", 0),
                             reasoning=opts.get("reasoning")))
    return parse_llm(r["text"]), r["latency_s"], r["cost_usd"], r["usage"].get("completion_tokens", 0)


def prf(wins, cats, outs, th):
    tp = fp = fn = 0
    for w, (probs, *_rest) in zip(wins, outs):
        for i, (cat, _d) in enumerate(cats, 1):
            gold = w["labels"].get(cat, False)
            if gold is None:
                continue
            pred = probs.get(i, 0.0) >= th
            tp += pred and gold
            fp += pred and not gold
            fn += (not pred) and gold
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-9)


def report(name, wins, cats, outs, th=0.5):
    prec, rec, f1 = prf(wins, cats, outs, th)
    lat = sorted(o[1] for o in outs)
    n = len(outs)
    cost = sum(o[2] for o in outs) / n * 1000
    print(f"{name:<50} recall {rec:5.1%} | precision {prec:5.1%} | F1 {f1:.3f} | "
          f"per page {lat[n//2]:.2f}s (p95 {lat[int(n*0.95)]:.2f}s) | per 1000 ${cost:.3f} | out {sum(o[3] for o in outs)/n:.0f}tok")
    return {"name": name, "f1": f1, "p50": lat[n // 2], "per1000": cost}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cuad")
    ap.add_argument("--pos", type=int, default=50)
    ap.add_argument("--rand", type=int, default=50)
    ap.add_argument("--calib", type=int, default=0)
    ap.add_argument("--llm", nargs="*", default=[])
    ap.add_argument("--llm-full", nargs="*", default=[])
    ap.add_argument("--llm-small", nargs="*", default=[])
    ap.add_argument("--small-n", type=int, default=200)
    a = ap.parse_args()
    cats = categories(a.cuad)
    pos, rnd, n_all, n_pos = sample_windows(a.cuad, a.pos, a.rand)
    wins = pos + rnd
    gold_pos = sum(v is True for w in wins for v in w["labels"].values())
    print(f"{len(wins)} excerpts taken from {n_all} test-contract excerpts ({n_pos} contain a clause). "
          f"{len(wins)*len(cats)} decisions, of which {gold_pos} were marked present by the lawyers\n")
    questions = jev_questions(cats)
    pool = ThreadPoolExecutor(8)

    th = 0.5
    if a.calib:
        cpos, crnd, *_ = sample_windows(a.cuad, a.calib, a.calib, seed=1, test=False)
        cw = cpos + crnd
        cout = list(pool.map(lambda w: run_jev(w, questions), cw))
        th = max((t / 100 for t in range(10, 100, 5)), key=lambda t: prf(cw, cats, cout, t)[2])
        print(f"Calibration: Jev threshold chosen on {len(cw)} held-out excerpts -> {th:.2f}"
              f" (F1 {prf(cw, cats, cout, th)[2]:.3f} on the calibration set)\n")

    jev_out = list(pool.map(lambda w: run_jev(w, questions), wins))
    summary = [report("Jev (all 41 in one call, threshold 0.5)", wins, cats, jev_out)]
    if a.calib:
        summary[0] = report(f"Jev (all 41 in one call, calibrated threshold {th:.2f})", wins, cats, jev_out, th)

    for models, compact, tag, mt in ((a.llm, True, "compact", 160), (a.llm_full, False, "full", 700)):
        system = llm_system_prompt(cats, compact)
        for model in models:
            try:
                out = list(pool.map(lambda w, m=model: run_llm(m, w, system, mt, tag), wins))
            except (LLMError, OSError) as e:
                print(f"{model}: failed -> {str(e)[:160]}")
                continue
            summary.append(report(f"{model} ({'IDs only' if compact else 'all 41 clauses'})", wins, cats, out))

    if a.llm_small:
        k = a.small_n // 2
        sw = pos[:k] + rnd[:k]
        sj = jev_out[:k] + jev_out[len(pos):len(pos) + k]
        print(f"\n--- Comparison against the expensive model (same {len(sw)} excerpts) ---")
        report(f"Jev (threshold {th:.2f})", sw, cats, sj, th)
        system = llm_system_prompt(cats, True)
        for model in a.llm_small:
            try:
                out = list(pool.map(lambda w, m=model: run_llm(m, w, system, 160, "compact"), sw))
            except (LLMError, OSError) as e:
                print(f"{model}: failed -> {str(e)[:160]}")
                continue
            summary.append(report(f"{model} (IDs only, {len(sw)} excerpts)", sw, cats, out))

    jev = summary[0]
    print("\n=== Ratios, with Jev as 1 ===")
    for s in summary[1:]:
        print(f"{s['name']:<56} cost x{s['per1000']/jev['per1000']:6.1f} | latency x{s['p50']/jev['p50']:4.1f} | F1 {s['f1']-jev['f1']:+.3f}")


if __name__ == "__main__":
    main()
