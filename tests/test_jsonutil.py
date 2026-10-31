from xcortex.utils.jsonutil import extract_json


def test_plain_object():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_fenced_json():
    text = "```json\n{\"a\": {\"b\": 2}}\n```"
    assert extract_json(text) == {"a": {"b": 2}}


def test_json_with_surrounding_prose():
    text = 'Here is the report:\n{"x": "value with } brace", "y": [1, 2]}\nThank you.'
    assert extract_json(text) == {"x": "value with } brace", "y": [1, 2]}


def test_missing_object():
    assert extract_json("no json here") is None
    assert extract_json("") is None
    assert extract_json(None) is None


def test_truncated_json():
    assert extract_json('{"a": 1') is None
