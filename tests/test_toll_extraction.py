"""Tests for toll extraction validation/parsing logic (no live API calls)."""
from __future__ import annotations

from src.toll_extraction import _validate, _default_unknown_result, extract_toll_rates
from src.utils import extract_json


def test_extract_json_parses_plain_json():
    raw = '{"toll_rate_1": "$2.50", "toll_rate_2": "UNKNOWN", "toll_rate_3": "UNKNOWN", "raw_extracted_text": "$2.50", "extraction_confidence": 0.9}'
    data = extract_json(raw)
    assert data is not None
    assert data["toll_rate_1"] == "$2.50"


def test_extract_json_parses_fenced_json():
    raw = """```json
    {"toll_rate_1": "UNKNOWN", "toll_rate_2": "UNKNOWN", "toll_rate_3": "UNKNOWN", "raw_extracted_text": "", "extraction_confidence": 0.0}
    ```"""
    data = extract_json(raw)
    assert data is not None
    assert data["toll_rate_1"] == "UNKNOWN"


def test_extract_json_returns_none_for_garbage():
    assert extract_json("not json at all") is None
    assert extract_json("") is None


def test_validate_requires_all_fields():
    valid = {
        "toll_rate_1": "$2.50",
        "toll_rate_2": "UNKNOWN",
        "toll_rate_3": "UNKNOWN",
        "raw_extracted_text": "$2.50",
        "extraction_confidence": 0.9,
    }
    assert _validate(valid) is True

    missing_field = dict(valid)
    del missing_field["toll_rate_3"]
    assert _validate(missing_field) is False

    bad_confidence = dict(valid)
    bad_confidence["extraction_confidence"] = "high"
    assert _validate(bad_confidence) is False

    out_of_range = dict(valid)
    out_of_range["extraction_confidence"] = 1.5
    assert _validate(out_of_range) is False


def test_default_unknown_result_marks_invalid():
    result = _default_unknown_result("some error")
    assert result.valid is False
    assert result.toll_rate_1 == "UNKNOWN"
    assert result.toll_rate_2 == "UNKNOWN"
    assert result.toll_rate_3 == "UNKNOWN"
    assert result.extraction_confidence == 0.0
    assert result.error == "some error"


# --- End-to-end tests against a fake Anthropic client (no network calls) ---


class _FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    """Stub for client.messages that returns a scripted sequence of
    responses, one per call, so we can simulate retry behavior."""

    def __init__(self, texts: list[str]):
        self._texts = list(texts)
        self.calls = 0

    def create(self, **kwargs):
        idx = min(self.calls, len(self._texts) - 1)
        self.calls += 1
        return _FakeResponse(self._texts[idx])


class _FakeClient:
    def __init__(self, texts: list[str]):
        self.messages = _FakeMessages(texts)


def test_extract_toll_rates_valid_json_first_try(tmp_path):
    image_path = tmp_path / "toll.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    valid_json = (
        '{"toll_rate_1": "$2.50", "toll_rate_2": "$3.00", "toll_rate_3": "UNKNOWN", '
        '"raw_extracted_text": "$2.50 $3.00", "extraction_confidence": 0.92}'
    )
    client = _FakeClient([valid_json])

    result = extract_toll_rates(client, image_path, model="claude-sonnet-4-6")

    assert result.valid is True
    assert result.toll_rate_1 == "$2.50"
    assert result.toll_rate_2 == "$3.00"
    assert result.toll_rate_3 == "UNKNOWN"
    assert result.extraction_confidence == 0.92
    assert client.messages.calls == 1  # no retry needed


def test_extract_toll_rates_invalid_json_recovers_on_retry(tmp_path):
    image_path = tmp_path / "toll.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    valid_json = (
        '{"toll_rate_1": "$1.75", "toll_rate_2": "UNKNOWN", "toll_rate_3": "UNKNOWN", '
        '"raw_extracted_text": "$1.75", "extraction_confidence": 0.7}'
    )
    client = _FakeClient(["this is not json at all", valid_json])

    result = extract_toll_rates(client, image_path, model="claude-sonnet-4-6")

    assert result.valid is True
    assert result.toll_rate_1 == "$1.75"
    assert client.messages.calls == 2  # first attempt failed, retry succeeded


def test_extract_toll_rates_invalid_json_both_attempts_fail(tmp_path):
    image_path = tmp_path / "toll.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    client = _FakeClient(["garbage response", "still not json"])

    result = extract_toll_rates(client, image_path, model="claude-sonnet-4-6")

    assert result.valid is False
    assert result.toll_rate_1 == "UNKNOWN"
    assert result.toll_rate_2 == "UNKNOWN"
    assert result.toll_rate_3 == "UNKNOWN"
    assert result.extraction_confidence == 0.0
    assert client.messages.calls == 2  # first attempt + one retry, then give up


def test_extract_toll_rates_missing_image_returns_unknown(tmp_path):
    missing_path = tmp_path / "does_not_exist.jpg"
    client = _FakeClient(["should not be called"])

    result = extract_toll_rates(client, missing_path, model="claude-sonnet-4-6")

    assert result.valid is False
    assert result.toll_rate_1 == "UNKNOWN"
    assert client.messages.calls == 0

