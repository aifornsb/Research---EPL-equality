"""Tests for traffic extraction schema validation and pricing lookup."""
from __future__ import annotations

import json

from src.traffic_extraction import (
    _validate,
    _validate_vehicle,
    _to_observations,
    extract_traffic,
)
from src.vehicle_pricing import estimate_price


def _base_vehicle(**overrides):
    v = {
        "sequence_number": 1,
        "direction_facing": "toward camera",
        "lane_type": "Express",
        "lane_description": "middle express lane 1",
        "vehicle_make": "Toyota",
        "vehicle_model": "Camry",
        "vehicle_year_estimate": "2018",
        "vehicle_body_type": "sedan",
        "vehicle_color": "silver",
        "vehicle_confidence": 0.8,
    }
    v.update(overrides)
    return v


def test_validate_vehicle_accepts_valid_entry():
    assert _validate_vehicle(_base_vehicle()) is True


def test_validate_vehicle_rejects_bad_lane_type():
    assert _validate_vehicle(_base_vehicle(lane_type="Carpool")) is False


def test_validate_vehicle_rejects_missing_field():
    v = _base_vehicle()
    del v["vehicle_color"]
    assert _validate_vehicle(v) is False


def test_validate_full_payload():
    payload = {"vehicles": [_base_vehicle(), _base_vehicle(sequence_number=2, lane_type="General Purpose")]}
    assert _validate(payload) is True


def test_validate_rejects_non_list_vehicles():
    assert _validate({"vehicles": "not a list"}) is False


def test_to_observations_converts_types():
    payload = {"vehicles": [_base_vehicle()]}
    obs = _to_observations(payload)
    assert len(obs) == 1
    assert obs[0].sequence_number == 1
    assert obs[0].vehicle_confidence == 0.8
    assert obs[0].lane_type == "Express"


def test_estimate_price_known_vehicle():
    price = estimate_price("Toyota", "Camry", "2018")
    assert price.price_source == "manual_lookup"
    assert price.price_range_currency == "USD"
    assert price.price_range_low != "UNKNOWN"
    assert float(price.price_range_low) < float(price.price_range_high)


def test_estimate_price_unknown_make_returns_unknown():
    price = estimate_price("UNKNOWN", "UNKNOWN", "UNKNOWN")
    assert price.price_range_low == "UNKNOWN"
    assert price.price_range_high == "UNKNOWN"
    assert price.price_source == "UNKNOWN"
    assert price.price_confidence == 0.0


def test_estimate_price_unlisted_model_returns_unknown():
    price = estimate_price("Toyota", "SomeObscureModelXYZ", "2018")
    assert price.price_source == "UNKNOWN"


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


def test_extract_traffic_valid_json_first_try(tmp_path):
    image_path = tmp_path / "traffic.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    payload = {"vehicles": [_base_vehicle(), _base_vehicle(sequence_number=2, lane_type="General Purpose")]}
    client = _FakeClient([json.dumps(payload)])

    result = extract_traffic(client, image_path, model="claude-sonnet-4-6")

    assert result.valid is True
    assert len(result.vehicles) == 2
    assert result.vehicles[0].lane_type == "Express"
    assert result.vehicles[1].lane_type == "General Purpose"
    assert client.messages.calls == 1


def test_extract_traffic_empty_vehicles_is_valid(tmp_path):
    image_path = tmp_path / "traffic.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    client = _FakeClient([json.dumps({"vehicles": []})])
    result = extract_traffic(client, image_path, model="claude-sonnet-4-6")

    assert result.valid is True
    assert result.vehicles == []


def test_extract_traffic_invalid_json_recovers_on_retry(tmp_path):
    image_path = tmp_path / "traffic.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    payload = {"vehicles": [_base_vehicle()]}
    client = _FakeClient(["not valid json", json.dumps(payload)])

    result = extract_traffic(client, image_path, model="claude-sonnet-4-6")

    assert result.valid is True
    assert len(result.vehicles) == 1
    assert client.messages.calls == 2


def test_extract_traffic_invalid_lane_type_recovers_on_retry(tmp_path):
    image_path = tmp_path / "traffic.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    bad_payload = {"vehicles": [_base_vehicle(lane_type="Carpool")]}  # invalid enum value
    good_payload = {"vehicles": [_base_vehicle(lane_type="Unknown")]}
    client = _FakeClient([json.dumps(bad_payload), json.dumps(good_payload)])

    result = extract_traffic(client, image_path, model="claude-sonnet-4-6")

    assert result.valid is True
    assert result.vehicles[0].lane_type == "Unknown"
    assert client.messages.calls == 2


def test_extract_traffic_invalid_json_both_attempts_fail(tmp_path):
    image_path = tmp_path / "traffic.jpg"
    image_path.write_bytes(b"fake-jpeg-bytes")

    client = _FakeClient(["garbage", "still garbage"])
    result = extract_traffic(client, image_path, model="claude-sonnet-4-6")

    assert result.valid is False
    assert result.vehicles == []
    assert result.error is not None
    assert client.messages.calls == 2


def test_extract_traffic_missing_image_returns_invalid(tmp_path):
    missing_path = tmp_path / "does_not_exist.jpg"
    client = _FakeClient(["should not be called"])

    result = extract_traffic(client, missing_path, model="claude-sonnet-4-6")

    assert result.valid is False
    assert result.vehicles == []
    assert client.messages.calls == 0
