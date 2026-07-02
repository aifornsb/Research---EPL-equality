"""Tests for CSV append/dedup logic in src/csv_writer.py."""
from __future__ import annotations

import csv
from pathlib import Path

from src.csv_writer import append_toll_row, append_traffic_rows, TOLL_FIELDNAMES, TRAFFIC_FIELDNAMES


def _read_rows(path: Path) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_append_toll_row_creates_file_with_header(tmp_path):
    csv_path = tmp_path / "toll_rates.csv"
    row = {
        "snapshot_date": "2026-07-02",
        "snapshot_time": "10:00:00",
        "toll_rate_1": "$2.50",
        "toll_rate_2": "UNKNOWN",
        "toll_rate_3": "UNKNOWN",
        "image_path": "data/images/toll/2026-07-02/2026-07-02_10-00-00.jpg",
        "extraction_confidence": 0.9,
        "raw_extracted_text": "$2.50",
    }
    written = append_toll_row(csv_path, row)
    assert written is True
    assert csv_path.exists()
    rows = _read_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["toll_rate_1"] == "$2.50"
    assert list(rows[0].keys()) == TOLL_FIELDNAMES


def test_append_toll_row_deduplicates(tmp_path):
    csv_path = tmp_path / "toll_rates.csv"
    row = {
        "snapshot_date": "2026-07-02",
        "snapshot_time": "10:00:00",
        "toll_rate_1": "$2.50",
        "toll_rate_2": "UNKNOWN",
        "toll_rate_3": "UNKNOWN",
        "image_path": "img.jpg",
        "extraction_confidence": 0.9,
        "raw_extracted_text": "$2.50",
    }
    first = append_toll_row(csv_path, row)
    second = append_toll_row(csv_path, row)
    assert first is True
    assert second is False
    rows = _read_rows(csv_path)
    assert len(rows) == 1


def test_append_traffic_rows_creates_and_dedups(tmp_path):
    csv_path = tmp_path / "traffic_observations.csv"
    row = {
        "snapshot_date": "2026-07-02",
        "snapshot_time": "10:00:00",
        "vehicle_id": "20260702_100000_ih30_carrier_pkwy_1",
        "direction_facing": "toward camera",
        "lane_type": "Express",
        "lane_description": "middle express lane 1",
        "vehicle_make": "UNKNOWN",
        "vehicle_model": "UNKNOWN",
        "vehicle_year_estimate": "UNKNOWN",
        "vehicle_body_type": "sedan",
        "vehicle_color": "white",
        "price_range_low": "UNKNOWN",
        "price_range_high": "UNKNOWN",
        "price_range_currency": "USD",
        "price_source": "UNKNOWN",
        "vehicle_confidence": 0.7,
        "price_confidence": 0.0,
        "image_path": "img.jpg",
    }
    written_count = append_traffic_rows(csv_path, [row, row])  # duplicate in same call
    assert written_count == 1
    rows = _read_rows(csv_path)
    assert len(rows) == 1
    assert list(rows[0].keys()) == TRAFFIC_FIELDNAMES

    # A second call with the same row should not add another
    written_again = append_traffic_rows(csv_path, [row])
    assert written_again == 0
    assert len(_read_rows(csv_path)) == 1


def test_append_toll_row_dedup_key_ignores_other_field_changes(tmp_path):
    csv_path = tmp_path / "toll_rates.csv"
    row_v1 = {
        "snapshot_date": "2026-07-02",
        "snapshot_time": "10:00:00",
        "toll_rate_1": "$2.50",
        "toll_rate_2": "UNKNOWN",
        "toll_rate_3": "UNKNOWN",
        "image_path": "img.jpg",
        "extraction_confidence": 0.9,
        "raw_extracted_text": "$2.50",
    }
    # Same (date, time, image_path) key, but different confidence/text —
    # still counts as the same snapshot and must not be re-appended.
    row_v2 = dict(row_v1, extraction_confidence=0.4, raw_extracted_text="different")

    first = append_toll_row(csv_path, row_v1)
    second = append_toll_row(csv_path, row_v2)

    assert first is True
    assert second is False
    rows = _read_rows(csv_path)
    assert len(rows) == 1
    assert rows[0]["raw_extracted_text"] == "$2.50"  # original row preserved


def test_append_traffic_rows_empty_list_is_noop(tmp_path):
    csv_path = tmp_path / "traffic_observations.csv"
    written = append_traffic_rows(csv_path, [])
    assert written == 0
    assert not csv_path.exists()
