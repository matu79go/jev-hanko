"""The task: CUAD is a set of 510 English contracts in which lawyers marked up 41 clause types.
For each excerpt of a contract we ask, in one shot, all 41 questions of the form
"does this clause appear in this excerpt?".

Ground truth is derived mechanically from the CUAD annotations (which carry character offsets).
The question text is CUAD's own official description of each clause, used verbatim.
CUAD: https://www.atticusprojectai.org/cuad (CC BY 4.0)
"""
import hashlib
import json
import random
import re
from pathlib import Path

WINDOW = 2000      # excerpt length, in characters
MIN_WINDOW = 300


def load_contracts(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))["data"]
    contracts = []
    for doc in data:
        para = doc["paragraphs"][0]
        spans = []
        for qa in para["qas"]:
            cat = re.search(r'related to "(.+?)"', qa["question"]).group(1)
            for ans in qa["answers"]:
                spans.append((cat, ans["answer_start"], ans["answer_start"] + len(ans["text"])))
        contracts.append({"title": doc["title"], "text": para["context"], "spans": spans})
    return contracts


def categories(path):
    """Return the 41 (clause name, official CUAD description) pairs."""
    para = json.loads(Path(path).read_text(encoding="utf-8"))["data"][0]["paragraphs"][0]
    out = []
    for qa in para["qas"]:
        cat = re.search(r'related to "(.+?)"', qa["question"]).group(1)
        details = qa["question"].split("Details:", 1)[1].strip() if "Details:" in qa["question"] else cat
        out.append((cat, details))
    return out


def windows(contract):
    """Cut a contract into ~2000-character excerpts and label each excerpt clause -> truth.

    A label is True when at least 50% of the annotated span falls inside the excerpt. A span that
    only partially overlaps (0 < overlap < 50%) becomes None and is excluded from scoring."""
    text, out, start = contract["text"], [], 0
    while start < len(text):
        end = min(start + WINDOW, len(text))
        if end < len(text):
            cut = text.rfind(" ", start + WINDOW // 2, end)
            end = cut if cut > 0 else end
        if end - start >= MIN_WINDOW:
            labels, marks = {}, []
            for cat, s, e in contract["spans"]:
                overlap = max(0, min(e, end) - max(s, start))
                if overlap == 0 or e <= s:
                    continue
                ratio = overlap / (e - s)
                if ratio >= 0.5:
                    labels[cat] = True
                    marks.append((cat, max(s, start) - start, min(e, end) - start))  # span position, relative to the excerpt
                elif labels.get(cat) is not True:
                    labels[cat] = None
            out.append({"id": f"{contract['title'][:40]}@{start}", "text": text[start:end], "labels": labels,
                        "marks": marks})
        start = end
    return out


def is_test_contract(title):
    return hashlib.md5(title.encode("utf-8")).hexdigest()[0] in "0123"


def sample_windows(path, n_positive, n_random, seed=0, test=True):
    """Sample excerpts from the test contracts (or, with test=False, from the calibration split).

    Returns n_positive excerpts that contain at least one clause plus n_random excerpts drawn at
    random. Both lists are in random order, so taking a prefix gives a nested subset."""
    allw = [w for c in load_contracts(path) if is_test_contract(c["title"]) == test for w in windows(c)]
    rng = random.Random(seed)
    pos = [w for w in allw if any(v is True for v in w["labels"].values())]
    chosen = rng.sample(pos, n_positive)
    ids = {w["id"] for w in chosen}
    rest = [w for w in allw if w["id"] not in ids]
    return chosen, rng.sample(rest, n_random), len(allw), len(pos)


def jev_questions(cats):
    return {f"q{i:02d}": {"type": "noul",
                          "instructions": f'`excerpt` contains contract language related to "{cat}". {details}'}
            for i, (cat, details) in enumerate(cats, 1)}


def llm_system_prompt(cats, compact=True):
    lines = "\n".join(f'{i}. {cat}: {details}' for i, (cat, details) in enumerate(cats, 1))
    head = ("You review an excerpt of a commercial contract. For each of the 41 clause types below, decide whether "
            "the excerpt contains contract language related to it.\n\n" + lines + "\n\n")
    if compact:
        return head + ("Output one line only: the numbers of the clause types that ARE present, each with your "
                       "confidence between 0 and 1, like `8:0.9,19:0.7`. If none are present, output `none`. No explanation.")
    return head + ("Output one line only: for ALL 41 clause types, `number:probability` that it is present, "
                   "comma-separated, like `1:0.02,2:0.91,...,41:0.10`. No explanation.")


def parse_llm(text, n=41):
    probs = {i: 0.0 for i in range(1, n + 1)}
    for m in re.finditer(r"(\d{1,2})\s*:\s*([01](?:\.\d+)?)", text or ""):
        i = int(m.group(1))
        if 1 <= i <= n:
            probs[i] = min(max(float(m.group(2)), 0.0), 1.0)
    return probs
