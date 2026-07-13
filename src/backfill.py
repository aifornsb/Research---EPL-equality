"""
Backfill analysis for saved camera images that were captured but never
successfully analyzed (e.g. while API credits were exhausted).

What it does:
- TOLL camera: finds saved images whose CSV row is "blind" (all three
  rates UNKNOWN with extraction_confidence == 0) or missing entirely,
  re-runs toll extraction on each, and repairs/appends the row using the
  timestamp from the image filename.
- TRAFFIC camera: finds saved images that have no rows at all in the
  traffic CSV, re-runs vehicle extraction, and appends the vehicle rows.

Idempotent: images that already have valid data are skipped, so the
script can be re-run safely after an interruption and will only process
what's still missing.

Usage:
    python -m src.backfill --dry-run          # report what would be done
    python -m src.backfill                    # process everything missing
    python -m src.backfill --limit 20         # process at most 20 images per camera
    python -m src.backfill --camera toll      # only the toll camera
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import datetime
from pathlib import Path

import anthropic

from src.config import (
    ensure_directories,
    load_config,
    IMAGES_DIR,
    TOLL_CSV,
    TRAFFIC_CSV,
)
from src.csv_writer import (
    TOLL_FIELDNAMES,
    append_toll_row,
    append_traffic_rows,
    _file_lock,
)
from src.toll_extraction import extract_toll_rates
from src.traffic_extraction import extract_traffic
from src.utils import safe_folder_name, setup_logging
from src.vehicle_pricing import estimate_price

logger = logging.getLogger("collector")

FILENAME_FMT = "%Y-%m-%d_%H-%M-%S"


def parse_timestamp_from_filename(image_path: Path) -> datetime | None:
    try:
        return datetime.strptime(image_path.stem, FILENAME_FMT)
    except ValueError:
        return None


def camera_image_files(camera_name: str, legacy_slug: str) -> list[Path]:
    """All saved .jpg files for a camera, covering both the current
    display-name folder and the legacy slug-named folder."""
    files: list[Path] = []
    for folder in {safe_folder_name(camera_name), legacy_slug}:
        base = IMAGES_DIR / folder
        if base.exists():
            files.extend(sorted(base.rglob("*.jpg")))
    return files


# ---------------------------------------------------------------------------
# Toll backfill: repair blind rows in place / append missing rows
# ---------------------------------------------------------------------------

def _load_toll_rows() -> list[dict]:
    if not TOLL_CSV.exists():
        return []
    with TOLL_CSV.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _save_toll_rows(rows: list[dict]) -> None:
    lock = TOLL_CSV.with_suffix(TOLL_CSV.suffix + ".lock")
    with _file_lock(lock):
        tmp = TOLL_CSV.with_suffix(".tmp")
        with tmp.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TOLL_FIELDNAMES)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "UNKNOWN") for k in TOLL_FIELDNAMES})
        tmp.replace(TOLL_CSV)


def _is_blind(row: dict) -> bool:
    try:
        conf = float(row.get("extraction_confidence", 0) or 0)
    except (TypeError, ValueError):
        conf = 0.0
    all_unknown = all(
        str(row.get(k, "UNKNOWN")).strip().upper() == "UNKNOWN"
        for k in ("toll_rate_1", "toll_rate_2", "toll_rate_3")
    )
    return conf == 0.0 and all_unknown


def backfill_toll(client, config, limit: int | None, dry_run: bool) -> tuple[int, int]:
    rows = _load_toll_rows()
    by_basename = {Path(r["image_path"]).name: i for i, r in enumerate(rows) if r.get("image_path")}

    images = camera_image_files(config.toll_camera.name, config.toll_camera.slug)
    todo: list[tuple[Path, int | None]] = []  # (image, row_index or None)
    for img in images:
        idx = by_basename.get(img.name)
        if idx is None:
            todo.append((img, None))            # no row at all
        elif _is_blind(rows[idx]):
            todo.append((img, idx))             # blind row to repair
        # else: valid row exists -> skip

    logger.info("[toll] %d saved images | %d need processing", len(images), len(todo))
    if dry_run:
        return len(todo), 0

    if limit:
        todo = todo[:limit]

    fixed = 0
    for n, (img, idx) in enumerate(todo, 1):
        ts = parse_timestamp_from_filename(img)
        if ts is None:
            logger.warning("[toll] skipping unparseable filename: %s", img.name)
            continue
        logger.info("[toll] (%d/%d) analyzing %s", n, len(todo), img.name)
        result = extract_toll_rates(client, img, config.anthropic_model)
        if not result.valid:
            logger.warning("[toll] extraction failed for %s: %s", img.name, result.error)
            continue

        row = {
            "snapshot_date": ts.strftime("%Y-%m-%d"),
            "snapshot_time": ts.strftime("%H:%M:%S"),
            "toll_rate_1": result.toll_rate_1,
            "toll_rate_2": result.toll_rate_2,
            "toll_rate_3": result.toll_rate_3,
            "image_path": str(img),
            "extraction_confidence": result.extraction_confidence,
            "raw_extracted_text": result.raw_extracted_text,
        }
        if idx is not None:
            rows[idx] = row
            _save_toll_rows(rows)               # small file; safe to rewrite
        else:
            append_toll_row(TOLL_CSV, row)
        fixed += 1
    return len(todo), fixed


# ---------------------------------------------------------------------------
# Traffic backfill: append rows for images that have none
# ---------------------------------------------------------------------------

def _traffic_basenames_with_rows() -> set[str]:
    if not TRAFFIC_CSV.exists():
        return set()
    with TRAFFIC_CSV.open("r", newline="", encoding="utf-8") as f:
        return {Path(r["image_path"]).name for r in csv.DictReader(f) if r.get("image_path")}


def backfill_traffic(client, config, limit: int | None, dry_run: bool) -> tuple[int, int]:
    have = _traffic_basenames_with_rows()
    images = camera_image_files(config.traffic_camera.name, config.traffic_camera.slug)
    todo = [img for img in images if img.name not in have]

    logger.info("[traffic] %d saved images | %d need processing", len(images), len(todo))
    if dry_run:
        return len(todo), 0

    if limit:
        todo = todo[:limit]

    done = 0
    for n, img in enumerate(todo, 1):
        ts = parse_timestamp_from_filename(img)
        if ts is None:
            logger.warning("[traffic] skipping unparseable filename: %s", img.name)
            continue
        logger.info("[traffic] (%d/%d) analyzing %s", n, len(todo), img.name)
        result = extract_traffic(client, img, config.anthropic_model)
        if not result.valid:
            logger.warning("[traffic] extraction failed for %s: %s", img.name, result.error)
            continue

        tag = ts.strftime("%Y%m%d_%H%M%S")
        out_rows = []
        for v in result.vehicles:
            price = estimate_price(v.vehicle_make, v.vehicle_model, v.vehicle_year_estimate)
            if price.price_source == "UNKNOWN" and (v.estimated_price_low > 0 or v.estimated_price_high > 0):
                plo, phi = str(v.estimated_price_low), str(v.estimated_price_high)
                psrc, pconf = "claude_estimate", v.price_confidence
            else:
                plo, phi = price.price_range_low, price.price_range_high
                psrc, pconf = price.price_source, price.price_confidence
            out_rows.append({
                "snapshot_date": ts.strftime("%Y-%m-%d"),
                "snapshot_time": ts.strftime("%H:%M:%S"),
                "vehicle_id": f"{tag}_{config.traffic_camera.slug}_{v.sequence_number}",
                "direction_facing": v.direction_facing,
                "lane_type": v.lane_type,
                "lane_description": v.lane_description,
                "vehicle_make": v.vehicle_make,
                "vehicle_model": v.vehicle_model,
                "vehicle_year_estimate": v.vehicle_year_estimate,
                "vehicle_body_type": v.vehicle_body_type,
                "vehicle_color": v.vehicle_color,
                "price_range_low": plo,
                "price_range_high": phi,
                "price_range_currency": "USD",
                "price_source": psrc,
                "vehicle_confidence": v.vehicle_confidence,
                "price_confidence": pconf,
                "image_path": str(img),
            })
        append_traffic_rows(TRAFFIC_CSV, out_rows)
        done += 1
    return len(todo), done


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Backfill analysis for unprocessed saved images")
    p.add_argument("--dry-run", action="store_true", help="Report counts only; no API calls")
    p.add_argument("--limit", type=int, default=None, help="Max images per camera this run")
    p.add_argument("--camera", choices=["toll", "traffic", "both"], default="both")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    ensure_directories()
    setup_logging()

    config = load_config(require_api_key=not args.dry_run)
    client = None if args.dry_run else anthropic.Anthropic(api_key=config.anthropic_api_key)

    if args.camera in ("toll", "both"):
        need, fixed = backfill_toll(client, config, args.limit, args.dry_run)
        logger.info("[toll] backfill complete: %d needed, %d fixed this run", need, fixed)
    if args.camera in ("traffic", "both"):
        need, done = backfill_traffic(client, config, args.limit, args.dry_run)
        logger.info("[traffic] backfill complete: %d needed, %d processed this run", need, done)
    return 0


if __name__ == "__main__":
    sys.exit(main())
