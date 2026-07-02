"""
Entry point for the TxDOT Toll & Traffic Camera Analytics collector.

Usage:
    python -m src.main --mode single-run
    python -m src.main --mode single-run --skip-analysis
    python -m src.main --mode continuous --duration-days 7 --interval-minutes 5

single-run:      capture both cameras once, analyze, append to CSVs, exit.
                 Intended to be invoked on a schedule (e.g. GitHub Actions
                 cron every 5 minutes).
continuous:      loop locally/on a server, capturing every N minutes for D
                 days, then exit automatically.
--skip-analysis: capture-only mode. Captures and saves both camera
                 snapshots, logs the result, but never calls the Claude
                 API and never writes CSV rows. Use this to verify camera
                 capture works (correct cameras, real images, correct
                 file paths) without spending any API tokens. Works with
                 either --mode value and does not require
                 ANTHROPIC_API_KEY to be set.
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime, timedelta

import anthropic

from src.camera_capture import capture_both_cameras, SnapshotResult
from src.config import (
    AppConfig,
    ensure_directories,
    load_config,
    FAILED_IMAGES_DIR,
    TOLL_CSV,
    TRAFFIC_CSV,
)
from src.csv_writer import append_toll_row, append_traffic_rows
from src.toll_extraction import extract_toll_rates
from src.traffic_extraction import extract_traffic
from src.utils import format_date, format_time, setup_logging
from src.vehicle_pricing import estimate_price

logger = logging.getLogger("collector")


def _save_for_manual_review(snapshot: SnapshotResult, reason: str) -> None:
    """Copy a problematic image into data/failed_review for later inspection."""
    try:
        FAILED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        if snapshot.image_path.exists():
            dest = FAILED_IMAGES_DIR / snapshot.image_path.name
            shutil.copy2(snapshot.image_path, dest)
            logger.warning("Saved failed image for review: %s (%s)", dest, reason)
        else:
            logger.warning(
                "Could not save failed image (file missing): %s (%s)",
                snapshot.image_path,
                reason,
            )
    except Exception as exc:  # noqa: BLE001
        logger.error("Error while saving failed image for review: %s", exc, exc_info=True)


def _log_capture_only(snapshot: SnapshotResult) -> None:
    """Log the outcome of a capture-only (--skip-analysis) snapshot."""
    if snapshot.success:
        logger.info(
            "[capture-only] OK | camera=%s | image=%s | timestamp=%s",
            snapshot.camera_name,
            snapshot.image_path,
            snapshot.captured_at.isoformat(),
        )
    else:
        logger.error(
            "[capture-only] FAILED | camera=%s | error=%s",
            snapshot.camera_name,
            snapshot.error,
        )


def process_toll_snapshot(
    client: anthropic.Anthropic, model: str, snapshot: SnapshotResult
) -> None:
    if not snapshot.success:
        logger.error(
            "Skipping toll extraction; capture failed for %s: %s",
            snapshot.camera_name,
            snapshot.error,
        )
        return

    result = extract_toll_rates(client, snapshot.image_path, model)
    if not result.valid:
        _save_for_manual_review(snapshot, f"toll extraction invalid: {result.error}")

    row = {
        "snapshot_date": format_date(snapshot.captured_at),
        "snapshot_time": format_time(snapshot.captured_at),
        "toll_rate_1": result.toll_rate_1,
        "toll_rate_2": result.toll_rate_2,
        "toll_rate_3": result.toll_rate_3,
        "image_path": str(snapshot.image_path),
        "extraction_confidence": result.extraction_confidence,
        "raw_extracted_text": result.raw_extracted_text,
    }
    append_toll_row(TOLL_CSV, row)


def process_traffic_snapshot(
    client: anthropic.Anthropic, model: str, snapshot: SnapshotResult
) -> None:
    if not snapshot.success:
        logger.error(
            "Skipping traffic extraction; capture failed for %s: %s",
            snapshot.camera_name,
            snapshot.error,
        )
        return

    result = extract_traffic(client, snapshot.image_path, model)
    if not result.valid:
        _save_for_manual_review(snapshot, f"traffic extraction invalid: {result.error}")
        return  # nothing structured to write

    timestamp_tag = snapshot.captured_at.strftime("%Y%m%d_%H%M%S")
    rows = []
    for vehicle in result.vehicles:
        vehicle_id = f"{timestamp_tag}_{snapshot.camera_slug}_{vehicle.sequence_number}"

        # Pricing: prefer the curated manual lookup table when it has an
        # entry for this make/model; otherwise use Claude's own estimate
        # from the vision analysis (price_source="claude_estimate").
        price = estimate_price(
            vehicle.vehicle_make, vehicle.vehicle_model, vehicle.vehicle_year_estimate
        )
        if price.price_source == "UNKNOWN" and (
            vehicle.estimated_price_low > 0 or vehicle.estimated_price_high > 0
        ):
            price_low = str(vehicle.estimated_price_low)
            price_high = str(vehicle.estimated_price_high)
            price_source = "claude_estimate"
            price_confidence = vehicle.price_confidence
        else:
            price_low = price.price_range_low
            price_high = price.price_range_high
            price_source = price.price_source
            price_confidence = price.price_confidence

        rows.append(
            {
                "snapshot_date": format_date(snapshot.captured_at),
                "snapshot_time": format_time(snapshot.captured_at),
                "vehicle_id": vehicle_id,
                "direction_facing": vehicle.direction_facing,
                "lane_type": vehicle.lane_type,
                "lane_description": vehicle.lane_description,
                "vehicle_make": vehicle.vehicle_make,
                "vehicle_model": vehicle.vehicle_model,
                "vehicle_year_estimate": vehicle.vehicle_year_estimate,
                "vehicle_body_type": vehicle.vehicle_body_type,
                "vehicle_color": vehicle.vehicle_color,
                "price_range_low": price_low,
                "price_range_high": price_high,
                "price_range_currency": "USD",
                "price_source": price_source,
                "vehicle_confidence": vehicle.vehicle_confidence,
                "price_confidence": price_confidence,
                "image_path": str(snapshot.image_path),
            }
        )
    append_traffic_rows(TRAFFIC_CSV, rows)


def run_single_cycle(
    config: AppConfig,
    client: anthropic.Anthropic | None,
    skip_analysis: bool = False,
) -> None:
    """Capture both cameras once, optionally analyze, and append CSV rows.

    When `skip_analysis` is True, this only captures and logs the two
    snapshots — no Claude API calls are made and no CSV rows are written.
    `client` may be None in that case.
    """
    logger.info("=== Starting collection cycle (skip_analysis=%s) ===", skip_analysis)
    toll_snapshot, traffic_snapshot = capture_both_cameras(config)

    if skip_analysis:
        _log_capture_only(toll_snapshot)
        _log_capture_only(traffic_snapshot)
        logger.info("=== Collection cycle complete (capture-only) ===")
        return

    assert client is not None, "Anthropic client is required unless skip_analysis=True"

    try:
        process_toll_snapshot(client, config.anthropic_model, toll_snapshot)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unhandled error processing toll snapshot: %s", exc, exc_info=True)

    try:
        process_traffic_snapshot(client, config.anthropic_model, traffic_snapshot)
    except Exception as exc:  # noqa: BLE001
        logger.error("Unhandled error processing traffic snapshot: %s", exc, exc_info=True)

    logger.info("=== Collection cycle complete ===")


def run_continuous(
    config: AppConfig,
    client: anthropic.Anthropic | None,
    duration_days: int,
    interval_minutes: int,
    skip_analysis: bool = False,
) -> None:
    """Run collection cycles every `interval_minutes` for `duration_days`."""
    end_time = datetime.now() + timedelta(days=duration_days)
    interval_seconds = interval_minutes * 60
    logger.info(
        "Starting continuous collection: duration_days=%s interval_minutes=%s "
        "skip_analysis=%s until=%s",
        duration_days,
        interval_minutes,
        skip_analysis,
        end_time.isoformat(),
    )

    cycle_num = 0
    while datetime.now() < end_time:
        cycle_num += 1
        cycle_start = time.monotonic()
        logger.info("--- Continuous mode: cycle %d ---", cycle_num)
        try:
            run_single_cycle(config, client, skip_analysis=skip_analysis)
        except Exception as exc:  # noqa: BLE001
            # Never let a single cycle's failure kill the 7-day run.
            logger.error("Cycle %d failed unexpectedly: %s", cycle_num, exc, exc_info=True)

        elapsed = time.monotonic() - cycle_start
        sleep_for = max(0.0, interval_seconds - elapsed)
        if datetime.now() + timedelta(seconds=sleep_for) > end_time:
            logger.info("Reached end of collection window; stopping.")
            break
        logger.info("Sleeping %.1f seconds until next cycle.", sleep_for)
        time.sleep(sleep_for)

    logger.info("Continuous collection finished after %d cycles.", cycle_num)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TxDOT Toll & Traffic Camera Analytics collector")
    parser.add_argument(
        "--mode", choices=["single-run", "continuous"], required=True, help="Execution mode"
    )
    parser.add_argument(
        "--duration-days", type=int, default=None, help="Override duration for continuous mode"
    )
    parser.add_argument(
        "--interval-minutes", type=int, default=None, help="Override interval for continuous mode"
    )
    parser.add_argument(
        "--skip-analysis",
        action="store_true",
        help=(
            "Capture-only mode: capture and save both camera snapshots, log the "
            "result, but skip Claude vision analysis and CSV writes entirely. "
            "Does not require ANTHROPIC_API_KEY. Useful for verifying capture "
            "works without spending API tokens."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_directories()
    setup_logging()

    try:
        config = load_config(require_api_key=not args.skip_analysis)
    except (FileNotFoundError, RuntimeError) as exc:
        logger.error("Configuration error: %s", exc)
        return 1

    client = None
    if not args.skip_analysis:
        client = anthropic.Anthropic(api_key=config.anthropic_api_key)

    if args.mode == "single-run":
        run_single_cycle(config, client, skip_analysis=args.skip_analysis)
    else:
        duration_days = args.duration_days or config.duration_days
        interval_minutes = args.interval_minutes or config.interval_minutes
        run_continuous(
            config,
            client,
            duration_days,
            interval_minutes,
            skip_analysis=args.skip_analysis,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
