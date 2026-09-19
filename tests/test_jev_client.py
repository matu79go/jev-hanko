import json

import pytest

from jev_hanko.jev_client import MODEL, _encode_criteria, build_request, cost_usd


def test_plain_criteria_pass_through():
    q = {"reduced": {"type": "noul", "instructions": "8%対象か",
                     "criteria": {"true": "飲食料品", "false": "酒類・外食"}}}
    body = build_request({"item": "本みりん"}, q)
    assert body["model"] == MODEL
    assert body["state"] == {"item": "本みりん"}
    assert body["questions"]["reduced"]["criteria"] == {"true": "飲食料品", "false": "酒類・外食"}


def test_rich_criteria_are_json_encoded_strings():
    crit = {"消耗品費": {"what": "10万円未満の物品", "examples": ["コピー用紙"]}, "会議費": "会議の飲食"}
    enc = _encode_criteria(crit)
    assert isinstance(enc["消耗品費"], str)
    assert json.loads(enc["消耗品費"])["examples"] == ["コピー用紙"]
    assert "コピー用紙" in enc["消耗品費"]  # ensure_ascii=False: Japanese stays readable
    assert enc["会議費"] == "会議の飲食"


def test_score_criteria_list_and_missing_criteria():
    body = build_request("s", {
        "sev": {"type": "score", "instructions": "具体性", "criteria": ["一式のみ", "品目あり"]},
        "dup": {"type": "noul", "instructions": "同一役務か"},
    })
    assert body["questions"]["sev"]["criteria"] == ["一式のみ", "品目あり"]
    assert "criteria" not in body["questions"]["dup"]


def test_dict_instructions_are_encoded():
    body = build_request("s", {"q": {"type": "noul", "instructions": {"question": "対象か", "focus": "品目のみ"}}})
    assert json.loads(body["questions"]["q"]["instructions"])["focus"] == "品目のみ"


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        build_request("s", {"q": {"type": "bool", "instructions": "x"}})


def test_cost_is_input_tokens_only():
    assert cost_usd({"input_tokens": 1_000_000, "output_tokens": 999}) == pytest.approx(0.042)


def test_billed_cost_wins_when_reported():
    assert cost_usd({"input_tokens": 686, "output_tokens": 90, "cost": 2.8812e-05}) == pytest.approx(2.8812e-05)
