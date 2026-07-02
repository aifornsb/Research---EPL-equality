"""Tests for timestamp/path helpers in src/utils.py."""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from src.utils import (
    format_date,
    format_filename_timestamp,
    format_time,
    image_path_for,
    now_local,
    slugify,
)


def test_format_date_and_time():
    dt = datetime(2026, 7, 2, 14, 5, 9, tzinfo=ZoneInfo("America/Chicago"))
    assert format_date(dt) == "2026-07-02"
    assert format_time(dt) == "14:05:09"


def test_format_filename_timestamp():
    dt = datetime(2026, 7, 2, 14, 5, 9, tzinfo=ZoneInfo("America/Chicago"))
    assert format_filename_timestamp(dt) == "2026-07-02_14-05-09"


def test_now_local_uses_requested_timezone():
    dt = now_local("America/Chicago")
    assert dt.tzinfo is not None
    assert dt.utcoffset() is not None


def test_image_path_for_builds_expected_structure(tmp_path):
    dt = datetime(2026, 7, 2, 14, 5, 9, tzinfo=ZoneInfo("America/Chicago"))
    path = image_path_for(tmp_path, "ih30_carrier_pkwy", dt)

    assert path.parent == tmp_path / "ih30_carrier_pkwy" / "2026-07-02"
    assert path.name == "2026-07-02_14-05-09.jpg"
    # The date directory should have been created as a side effect.
    assert path.parent.exists()


def test_image_path_for_same_timestamp_is_deterministic(tmp_path):
    dt = datetime(2026, 7, 2, 14, 5, 9, tzinfo=ZoneInfo("America/Chicago"))
    path_a = image_path_for(tmp_path, "ih30_carrier_pkwy", dt)
    path_b = image_path_for(tmp_path, "ih30_carrier_pkwy", dt)
    # Same camera + same timestamp must resolve to the same file path,
    # which is what keeps CSV rows and image filenames in sync.
    assert path_a == path_b


def test_image_path_for_different_cameras_are_isolated(tmp_path):
    dt = datetime(2026, 7, 2, 14, 5, 9, tzinfo=ZoneInfo("America/Chicago"))
    toll_path = image_path_for(tmp_path, "ih30_loop12_wb_trdms_sta_1251_37", dt)
    traffic_path = image_path_for(tmp_path, "ih30_carrier_pkwy", dt)
    assert toll_path != traffic_path
    assert "ih30_loop12_wb_trdms_sta_1251_37" in str(toll_path)
    assert "ih30_carrier_pkwy" in str(traffic_path)


def test_slugify_basic():
    assert slugify("IH30 @ Carrier Pkwy") == "ih30_carrier_pkwy"
    assert slugify("IH30 @ Loop 12 WB TRDMS Sta 1251-37") == "ih30_loop_12_wb_trdms_sta_1251_37"
