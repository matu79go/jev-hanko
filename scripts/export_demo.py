"""デモページ用のデータを、測定済みのキャッシュから書き出す(API は呼ばない)。

Usage: env -u OPENROUTER_API_KEY python3 scripts/export_demo.py <CUADv1.json> docs/demo_data.js
各モデルの「待ち時間」「確率」「費用」はすべて実測値。抜粋は CUAD(CC BY 4.0)の本文。
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jev_hanko.cuad_task import categories, jev_questions, llm_system_prompt, sample_windows  # noqa: E402
from scripts.eval_cuad import run_jev, run_llm  # noqa: E402

N_POS, N_RAND = 14, 6
JEV_THRESHOLD = 0.70  # テストと別の契約書で決めた値
LANES = [
    ("Jev", None),
    ("Qwen 3.7 Flash", "qwen/qwen3.7-flash"),
    ("Gemini 2.5 Flash-Lite", "google/gemini-2.5-flash-lite"),
    ("Claude Haiku 4.5", "anthropic/claude-haiku-4.5"),
    ("Claude Sonnet 5", "anthropic/claude-sonnet-5"),
]
JA = ["文書名", "当事者", "契約日", "発効日", "満了日", "更新期間", "更新拒絶の通知期間", "準拠法", "最恵待遇", "競業避止",
      "独占", "顧客の勧誘禁止", "競業制限の例外", "従業員の引抜禁止", "誹謗禁止", "任意解約", "先買権等", "支配権変更",
      "譲渡禁止", "収益分配", "価格制限", "最低購入義務", "数量制限", "知財の譲渡", "知財の共有", "ライセンス許諾",
      "譲渡不可ライセンス", "関連会社ライセンス(許諾者)", "関連会社ライセンス(被許諾者)", "無制限ライセンス",
      "取消不能・永久ライセンス", "ソースコード預託", "契約終了後の役務", "監査権", "責任無制限", "責任上限",
      "損害賠償額の予定", "保証期間", "保険", "不提訴の誓約", "第三者受益者"]


def main(cuad, out_path):
    cats = categories(cuad)
    assert len(cats) == len(JA) == 41
    pos, rnd, *_ = sample_windows(cuad, 250, 250)
    pool_pos, pool_rnd = pos[:100], rnd[:100]   # 全レーン(Sonnet 含む)で測定済みの 200 件
    questions, system = jev_questions(cats), llm_system_prompt(cats, True)

    def result(model, w):
        probs, latency, cost, _tok = run_jev(w, questions) if model is None else run_llm(model, w, system, 160, "compact")
        return {"p": [round(probs.get(i, 0.0), 2) for i in range(1, 42)], "t": round(latency, 3), "c": cost}

    res = {name: {w["id"]: result(model, w) for w in pool_pos + pool_rnd} for name, model in LANES}
    overall = {name: sum(r["t"] for r in res[name].values()) / len(res[name]) for name, _m in LANES}

    # デモに使う 20 ページの選び方(どのモデルにも公平な規則):
    # 乱数の種を 0 から順に試し、全レーンで「20 ページの平均待ち時間が 200 件全体の平均の ±15% 以内」
    # になる最初の種を採用する。待ち時間の偏った抜き方で競走の見え方が変わるのを防ぐため。
    for seed in range(1000):
        rng = random.Random(seed)
        wins = rng.sample(pool_pos, N_POS) + rng.sample(pool_rnd, N_RAND)
        rng.shuffle(wins)
        if all(abs(sum(res[name][w["id"]]["t"] for w in wins) / len(wins) / overall[name] - 1) <= 0.15 for name, _m in LANES):
            break
    else:
        raise SystemExit("代表的な 20 ページが見つからない")
    print(f"採用した乱数の種: {seed}")
    lanes = [{"name": name, "threshold": JEV_THRESHOLD if model is None else 0.5,
              "mean_all": round(overall[name], 3), "pages": [res[name][w["id"]] for w in wins]} for name, model in LANES]
    cat_index = {c: i for i, (c, _d) in enumerate(cats)}
    data = {
        "categories": [{"en": c, "ja": j} for (c, _d), j in zip(cats, JA)],
        "excerpts": [{"id": w["id"], "text": w["text"][:900],
                      "gold": [1 if w["labels"].get(c) is True else (None if w["labels"].get(c, False) is None else 0)
                               for c, _d in cats],
                      # 弁護士が印を付けた条文(条項の番号, 抜粋内の開始位置, 終了位置, 本文)
                      "marks": [[cat_index[c], s, e, w["text"][s:e][:220]] for c, s, e in w["marks"]]} for w in wins],
        "lanes": lanes,
        "note": "待ち時間・確率・費用はすべて実測値。各モデルに同じ抜粋を1件ずつ順に処理させた場合の再現。",
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text("window.DEMO = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    for lane in lanes:
        print(f"{lane['name']:<24} 合計 {sum(p['t'] for p in lane['pages']):6.2f}s  費用 ${sum(p['c'] for p in lane['pages']):.5f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
