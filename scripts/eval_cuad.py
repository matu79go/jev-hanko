"""CUAD: 契約書の抜粋1件につき 41 種類の条項チェックをまとめて判定し、弁護士の注釈と突き合わせる。

Usage: OPENROUTER_API_KEY=... python3 scripts/eval_cuad.py <CUADv1.json> --pos 250 --rand 250 --calib 100 \
         --llm google/gemini-2.5-flash-lite anthropic/claude-haiku-4.5 \
         --llm-full google/gemini-2.5-flash-lite --llm-small anthropic/claude-sonnet-5 --small-n 200
  --llm       : 該当する条項の番号だけを短く答えさせる(LLM に最も有利な形式)
  --llm-full  : 41 項目すべての確率を答えさせる(Jev と同じ情報量)
  --llm-small : 高いモデル用。テストの先頭 small-n 件だけで測る(番号だけ形式)
  --calib     : テストとは別の契約書の抜粋で Jev のしきい値を1つ決める(正例あり N 件 + 無作為 N 件)
応答は cache/ に保存し、再実行は課金なし。
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

# thinking を使わせない/最小にするための指定(モデルごと)。結果の表に出力トークン数を出して検証できるようにする
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
    print(f"{name:<50} 見つけた率 {rec:5.1%} | 的中率 {prec:5.1%} | F1 {f1:.3f} | "
          f"1件 {lat[n//2]:.2f}s(95%点 {lat[int(n*0.95)]:.2f}s)| 1000件 ${cost:.3f} | 出力 {sum(o[3] for o in outs)/n:.0f}tok")
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
    print(f"テスト用契約書の抜粋 全{n_all}件(条項を含む {n_pos}件)から {len(wins)}件。"
          f"判定数 {len(wins)*len(cats)}、うち弁護士が『ある』とした箇所 {gold_pos}\n")
    questions = jev_questions(cats)
    pool = ThreadPoolExecutor(8)

    th = 0.5
    if a.calib:
        cpos, crnd, *_ = sample_windows(a.cuad, a.calib, a.calib, seed=1, test=False)
        cw = cpos + crnd
        cout = list(pool.map(lambda w: run_jev(w, questions), cw))
        th = max((t / 100 for t in range(10, 100, 5)), key=lambda t: prf(cw, cats, cout, t)[2])
        print(f"較正: テストと別の契約書の抜粋 {len(cw)} 件で Jev のしきい値を決定 → {th:.2f}"
              f"(較正データ上の F1 {prf(cw, cats, cout, th)[2]:.3f})\n")

    jev_out = list(pool.map(lambda w: run_jev(w, questions), wins))
    summary = [report("Jev(41問を1回で、しきい値 0.5)", wins, cats, jev_out)]
    if a.calib:
        summary[0] = report(f"Jev(41問を1回で、較正しきい値 {th:.2f})", wins, cats, jev_out, th)

    for models, compact, tag, mt in ((a.llm, True, "compact", 160), (a.llm_full, False, "full", 700)):
        system = llm_system_prompt(cats, compact)
        for model in models:
            try:
                out = list(pool.map(lambda w, m=model: run_llm(m, w, system, mt, tag), wins))
            except (LLMError, OSError) as e:
                print(f"{model}: 実行できず → {str(e)[:160]}")
                continue
            summary.append(report(f"{model}({'番号だけ' if compact else '41項目すべて'})", wins, cats, out))

    if a.llm_small:
        k = a.small_n // 2
        sw = pos[:k] + rnd[:k]
        sj = jev_out[:k] + jev_out[len(pos):len(pos) + k]
        print(f"\n--- 高いモデルとの比較(同じ {len(sw)} 件)---")
        report(f"Jev(しきい値 {th:.2f})", sw, cats, sj, th)
        system = llm_system_prompt(cats, True)
        for model in a.llm_small:
            try:
                out = list(pool.map(lambda w, m=model: run_llm(m, w, system, 160, "compact"), sw))
            except (LLMError, OSError) as e:
                print(f"{model}: 実行できず → {str(e)[:160]}")
                continue
            summary.append(report(f"{model}(番号だけ、{len(sw)}件)", sw, cats, out))

    jev = summary[0]
    print("\n=== Jev を 1 としたときの比 ===")
    for s in summary[1:]:
        print(f"{s['name']:<56} 値段 {s['per1000']/jev['per1000']:6.1f} 倍 | 待ち時間 {s['p50']/jev['p50']:4.1f} 倍 | F1 差 {s['f1']-jev['f1']:+.3f}")


if __name__ == "__main__":
    main()
