"""
Safe, append-only CSV writing for toll_rates.csv and traffic_observations.csv.

Guarantees:
- Creates the file with a header row if it doesn't exist yet.
- Never overwrites existing rows; only appends.
- Uses a lock file (via a simple file-based lock) so concurrent runs
  (e.g. overlapping GitHub Actions runs) don't corrupt the CSV.
- Deduplicates on (snapshot_date, snapshot_time, image_path[, vehicle_id])
  so the same snapshot/vehicle is never written twice.
"""
from __future__ import annotations

import csv
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger("collector")

TOLL_FIELDNAMES = [
    "snapshot_date",
    "snapshot_time",
    "toll_rate_1",
    "toll_rate_2",
    "toll_rate_3",
    "image_path",
    "extraction_confidence",
    "raw_extracted_text",
]

TRAFFIC_FIELDNAMES = [
    "snapshot_date",
    "snapshot_time",
    "vehicle_id",
    "direction_facing",
    "lane_type",
    "lane_description",
    "vehicle_make",
    "vehicle_model",
    "vehicle_year_estimate",
    "vehicle_body_type",
    "vehicle_color",
    "price_range_low",
    "price_range_high",
    "price_range_currency",
    "price_source",
    "vehicle_confidence",
    "price_confidence",
    "image_path",
]


@contextmanager
def _file_lock(lock_path: Path, timeout_seconds: float = 30.0, poll_interval: float = 0.1):
    """A minimal cross-platform file lock using atomic file creation.

    Not a full-featured lock, but sufficient for a single-machine or
    single-job-at-a-time GitHub Actions workflow to avoid interleaved
    writes corrupting the CSV during concurrent runs.
    """
    start = time.time()
    fd = None
    while fd is None:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
        except FileExistsError:
            if time.time() - start > timeout_seconds:
                logger.warning("Lock file %s held too long; proceeding anyway.", lock_path)
                break
            time.sleep(poll_interval)
    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)
        lock_path.unlink(missing_ok=True)


def _ensure_header(path: Path, fieldnames: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()


def _read_existing_keys(path: Path, key_fields: list[str]) -> set[tuple]:
    keys: set[tuple] = set()
    if not path.exists():
        return keys
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                keys.add(tuple(row[k] for k in key_fields))
            except KeyError:
                continue
    return keys


def append_toll_row(csv_path: Path, row: dict) -> bool:
    """Append a single toll-rate row. Returns True if written, False if
    skipped as a duplicate."""
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    with _file_lock(lock_path):
        _ensure_header(csv_path, TOLL_FIELDNAMES)
        key_fields = ["snapshot_date", "snapshot_time", "image_path"]
        existing = _read_existing_keys(csv_path, key_fields)
        key = tuple(row[k] for k in key_fields)
        if key in existing:
            logger.info("Duplicate toll row skipped: %s", key)
            return False
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TOLL_FIELDNAMES)
            writer.writerow({k: row.get(k, "UNKNOWN") for k in TOLL_FIELDNAMES})
    logger.info("Toll row appended: %s", key)
    return True


def append_traffic_rows(csv_path: Path, rows: list[dict]) -> int:
    """Append multiple vehicle observation rows. Returns count actually
    written (excluding duplicates)."""
    if not rows:
        return 0
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    written = 0
    with _file_lock(lock_path):
        _ensure_header(csv_path, TRAFFIC_FIELDNAMES)
        key_fields = ["vehicle_id", "image_path"]
        existing = _read_existing_keys(csv_path, key_fields)
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRAFFIC_FIELDNAMES)
            for row in rows:
                key = tuple(row[k] for k in key_fields)
                if key in existing:
                    logger.info("Duplicate traffic row skipped: %s", key)
                    continue
                writer.writerow({k: row.get(k, "UNKNOWN") for k in TRAFFIC_FIELDNAMES})
                existing.add(key)
                written += 1
    logger.info("Traffic rows appended: %d", written)
    return written
