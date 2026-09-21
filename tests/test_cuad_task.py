from jev_hanko.cuad_task import jev_questions, llm_system_prompt, parse_llm, windows


def _contract(n_chars, spans):
    return {"title": "T", "text": ("word " * (n_chars // 5))[:n_chars], "spans": spans}


def test_windows_cover_text_without_overlap():
    c = _contract(5000, [])
    ws = windows(c)
    assert "".join(w["text"] for w in ws) == c["text"][:sum(len(w["text"]) for w in ws)]
    assert all(300 <= len(w["text"]) <= 2000 for w in ws)


def test_span_inside_window_is_positive():
    ws = windows(_contract(5000, [("Non-Compete", 100, 200)]))
    assert ws[0]["labels"] == {"Non-Compete": True}
    assert all("Non-Compete" not in w["labels"] for w in ws[1:])


def test_span_cut_by_boundary_majority_side_positive_other_side_excluded():
    ws = windows(_contract(5000, []))
    boundary = len(ws[0]["text"])
    ws = windows(_contract(5000, [("Audit Rights", boundary - 90, boundary + 10)]))
    assert ws[0]["labels"]["Audit Rights"] is True      # 90% of the span lies in the first excerpt
    assert ws[1]["labels"]["Audit Rights"] is None      # the 10% overlap side is excluded from scoring


def test_jev_questions_are_41_style_nouls_with_official_details():
    q = jev_questions([("Governing Law", "Which state/country's law governs?")])
    assert q["q01"]["type"] == "noul" and "Governing Law" in q["q01"]["instructions"]
    assert "Which state/country's law governs?" in q["q01"]["instructions"]


def test_parse_llm_compact_and_none():
    p = parse_llm("8:0.9, 19:0.7")
    assert p[8] == 0.9 and p[19] == 0.7 and p[1] == 0.0
    assert all(v == 0.0 for v in parse_llm("none").values())


def test_prompt_variants():
    cats = [("Governing Law", "law")]
    assert "none" in llm_system_prompt(cats, compact=True)
    assert "ALL 41" in llm_system_prompt(cats, compact=False)
