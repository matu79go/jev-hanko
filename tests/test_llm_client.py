from jev_hanko.llm_client import build_messages, parse_answer

LABELS = ["需用費", "役務費", "委託料", "使用料及賃借料", "負担金補助及交付金"]


def test_parse_plain():
    assert parse_answer("委託料|0.92", LABELS) == ("委託料", 0.92)


def test_parse_with_noise_and_spaces():
    assert parse_answer(" 使用料及賃借料 | 0.7\n", LABELS) == ("使用料及賃借料", 0.7)


def test_parse_missing_confidence():
    assert parse_answer("役務費", LABELS) == ("役務費", 0.0)


def test_parse_unknown_label():
    assert parse_answer("わかりません|0.5", LABELS) == (None, 0.5)


def test_parse_confidence_clamped_and_integer():
    assert parse_answer("需用費|1", LABELS) == ("需用費", 1.0)


def test_messages_carry_same_criteria_and_state():
    msgs = build_messages("何費か", {"需用費": "物品の買入れ", "委託料": "業務委託"}, {"件名": "清掃業務委託"})
    assert "需用費: 物品の買入れ" in msgs[0]["content"] and "委託料: 業務委託" in msgs[0]["content"]
    assert "清掃業務委託" in msgs[1]["content"]
