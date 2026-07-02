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
    safe_folder_name,
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
    path = image_path_for(tmp_path, safe_folder_name("IH30 @ Carrier Pkwy"), dt)

    assert path.parent == tmp_path / "IH30 @ Carrier Pkwy" / "2026-07-02"
    assert path.name == "2026-07-02_14-05-09.jpg"
    # The date directory should have been created as a side effect.
    assert path.parent.exists()


def test_image_path_for_same_timestamp_is_deterministic(tmp_path):
    dt = datetime(2026, 7, 2, 14, 5, 9, tzinfo=ZoneInfo("America/Chicago"))
    path_a = image_path_for(tmp_path, safe_folder_name("IH30 @ Carrier Pkwy"), dt)
    path_b = image_path_for(tmp_path, safe_folder_name("IH30 @ Carrier Pkwy"), dt)
    # Same camera + same timestamp must resolve to the same file path,
    # which is what keeps CSV rows and image filenames in sync.
    assert path_a == path_b


def test_image_path_for_different_cameras_are_isolated(tmp_path):
    dt = datetime(2026, 7, 2, 14, 5, 9, tzinfo=ZoneInfo("America/Chicago"))
    toll_path = image_path_for(
        tmp_path, safe_folder_name("IH30 @ Loop 12 WB TRDMS Sta 1251-37"), dt
    )
    traffic_path = image_path_for(tmp_path, safe_folder_name("IH30 @ Carrier Pkwy"), dt)
    assert toll_path != traffic_path
    assert "IH30 @ Loop 12 WB TRDMS Sta 1251-37" in str(toll_path)
    assert "IH30 @ Carrier Pkwy" in str(traffic_path)


def test_safe_folder_name_preserves_readable_camera_names():
    # Spaces, '@', '-' and digits are all safe and should be kept as-is.
    assert safe_folder_name("IH30 @ Carrier Pkwy") == "IH30 @ Carrier Pkwy"
    assert (
        safe_folder_name("IH30 @ Loop 12 WB TRDMS Sta 1251-37")
        == "IH30 @ Loop 12 WB TRDMS Sta 1251-37"
    )


def test_safe_folder_name_replaces_unsafe_characters():
    assert safe_folder_name("US75 N/S: split*view?") == "US75 N_S_ split_view_"
    assert safe_folder_name('a"b<c>d|e') == "a_b_c_d_e"
    # Collapses whitespace runs, strips edges and trailing dots
    assert safe_folder_name("  spaced   name .") == "spaced name"
    # Never returns an empty string
    assert safe_folder_name("???") == "___"
    assert safe_folder_name(" . ") == "unnamed_camera"


def test_slugify_basic():
    assert slugify("IH30 @ Carrier Pkwy") == "ih30_carrier_pkwy"
    assert slugify("IH30 @ Loop 12 WB TRDMS Sta 1251-37") == "ih30_loop_12_wb_trdms_sta_1251_37"
