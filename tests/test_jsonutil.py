import pytest

from agent.deliberation.jsonutil import extract_json, normalize_options


def test_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_and_prose():
    assert extract_json('Here you go:\n```json\n{"options": ["x"]}\n```\nthanks') == {"options": ["x"]}
    assert extract_json('Sure! {"a": "has } brace", "b": [1]} trailing') == {"a": "has } brace", "b": [1]}


def test_none():
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_normalize():
    assert normalize_options({"hooks": ["a", {"text": "b", "rationale": "r"}, {"nope": 1}]}) == [
        {"text": "a", "rationale": ""}, {"text": "b", "rationale": "r"},
    ]
    assert normalize_options([{"caption": " c "}])[0]["text"] == "c"
