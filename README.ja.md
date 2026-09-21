# jev-hanko

判断特化 AI「Jev」(TypeSafe AI)に、契約書の 41 項目チェックをまとめて判定させ、速さ・値段・精度を
速くて安い LLM と実測比較したリポジトリです。

English → [README.md](README.md)

**何をさせているか**: 契約書を 1 ページ読むたびに、41 種類の条項が「このページにあるか」を一度に判定し、項目ごとのカゴへ振り分けます。

![Jev が契約書を41個のカゴに振り分ける](media/demo_sort.gif)

**同じ仕事を、速くて安い LLM と競走させた結果**:

![契約書41項目チェック競走](media/demo.gif)

どちらの動画も演出ではなく、測定した待ち時間と判定結果をそのまま再生したものです。日本語の条文は筆者による参考訳です。

- ブラウザで動かす: [振り分け](https://matu79go.github.io/jev-hanko/sort.html) / [速度の競走](https://matu79go.github.io/jev-hanko/)(英語版は URL に `?lang=en`)
- 動画ファイル: `media/demo_sort.mp4`, `media/demo.mp4`(英語版は `*_en.mp4`)

## 結果(英文契約書 500 ページ × 41 項目 = 20,500 判定、正解は弁護士の注釈)

| モデル | 1 ページの判定時間(平均) | 1,000 ページあたり | 見つけた率 | 的中率 | F1 |
|---|---|---|---|---|---|
| **Jev(41 問を 1 回で)** | **0.40 秒** | $0.117 | 68.3% | 41.9% | 0.519 |
| Llama 3.1 8B(最速プロバイダ) | 0.57 秒 | $0.074 | 55.4% | 10.8% | 0.180 |
| Qwen 3.7 Flash | 0.84 秒 | **$0.060** | 65.7% | 59.3% | **0.623** |
| gpt-oss-20b(最速プロバイダ) | 0.92 秒 | $0.175 | 59.5% | 52.2% | 0.556 |
| Gemini 2.5 Flash-Lite(番号だけ) | 1.08 秒 | $0.206 | 73.9% | 21.7% | 0.335 |
| Claude Haiku 4.5 | 1.47 秒 | $2.143 | 79.1% | 37.4% | 0.508 |
| Gemini 2.5 Flash-Lite(41 項目すべて) | 1.50 秒 | $0.322 | 72.0% | 51.0% | 0.597 |
| Claude Sonnet 5(200 ページ) | 2.36 秒 | $6.068 | 77.0% | 52.1% | 0.622 |

- 判定時間は、リクエストを送ってから応答を受け取り終わるまでの秒数の、1 ページあたりの平均です
- 今回の測定では、速さは試した全構成で Jev が最速。値段は主流モデルより安いが、最安は Qwen 3.7 Flash(Jev の半額)。精度は中位
- これはこの題材・この聞き方・この比較相手での結果です。質問の書き方や題材によって変わり得ます
- 集計の生出力: [`results/cuad_500x41_2026-09-19.txt`](results/cuad_500x41_2026-09-19.txt)

## すぐ試す

### 0. 鍵なし・費用ゼロで見る

```bash
git clone https://github.com/matu79go/jev-hanko && cd jev-hanko
python3 -m pytest -q          # テスト(外部 API は呼びません)
# docs/sort.html と docs/index.html をブラウザで開くと、測定結果の再生が見られます
```

### 1. Jev を 1 回だけ呼んでみる(約 $0.00003)

[OpenRouter](https://openrouter.ai/) の API キーが必要です。キーは環境変数で渡し、ファイルには書きません。

```bash
export OPENROUTER_API_KEY=...   # 自分のキー
python3 scripts/smoke_jev.py
```

Jev は OpenRouter の `/api/alpha/decisions` 専用で、`/chat/completions` からは呼べません。

### 2. 測定を再現する

```bash
# CUAD v1(CC BY 4.0)を取得
curl -L -o data.zip https://github.com/TheAtticusProject/cuad/raw/main/data.zip
unzip data.zip CUADv1.json

# 小さく試す(100 ページ。Jev + Flash-Lite で約 $0.05)
python3 scripts/eval_cuad.py CUADv1.json --pos 50 --rand 50 --llm google/gemini-2.5-flash-lite

# 記事と同じ本測定(全モデルで約 $3)
python3 scripts/eval_cuad.py CUADv1.json --pos 250 --rand 250 --calib 100 \
  --llm google/gemini-2.5-flash-lite anthropic/claude-haiku-4.5 qwen/qwen3.7-flash \
        meta-llama/llama-3.1-8b-instruct:nitro openai/gpt-oss-20b:nitro \
  --llm-full google/gemini-2.5-flash-lite \
  --llm-small anthropic/claude-sonnet-5 --small-n 200
```

応答は `cache/` に保存され、同じ条件の再実行では課金されません。待ち時間は回線や時間帯で変わります。

### 3. デモと動画を作り直す(任意)

```bash
python3 scripts/export_demo.py CUADv1.json docs/demo_data.js   # 測定済みキャッシュからデモ用データを書き出す
pip install playwright && playwright install chromium          # 録画に必要(ほかに ffmpeg)
python3 scripts/record_demo.py sort        # → media/demo_sort.mp4 / .gif
python3 scripts/record_demo.py race en     # → media/demo_en.mp4 / .gif(英語版)
```

必要なもの: Python 3.10 以上。評価コードの依存は標準ライブラリのみです。

## 構成

| パス | 内容 |
|---|---|
| `jev_hanko/jev_client.py` | Jev クライアント(OpenRouter Decisions API) |
| `jev_hanko/llm_client.py` | 比較用 LLM クライアント |
| `jev_hanko/cuad_task.py` | CUAD からページ単位の問題と正解を機械的に作る |
| `scripts/eval_cuad.py` | 本測定(較正しきい値、LLM 比較) |
| `scripts/export_demo.py`, `scripts/record_demo.py` | デモ用データの書き出しと録画 |
| `docs/` | デモページ(GitHub Pages) |

## データの出典

- [CUAD v1](https://www.atticusprojectai.org/cuad)(The Atticus Project、CC BY 4.0)。`docs/demo_data.js` には CUAD の契約書の抜粋が含まれます

## License

MIT(コード)。データはそれぞれの出典の条件に従います。
