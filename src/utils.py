"""
Shared utility helpers: logging setup, timezone-aware timestamps,
slugging, and safe JSON parsing for Claude vision responses.
"""
from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from src.config import LOGS_DIR


def setup_logging(log_name: str = "collector") -> logging.Logger:
    """Configure a logger that writes to both stdout and logs/collector.log."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / "collector.log"

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        # Avoid duplicate handlers if setup_logging is called more than once
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def now_local(timezone: str) -> datetime:
    """Return the current time localized to the given IANA timezone."""
    return datetime.now(ZoneInfo(timezone))


def format_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def format_time(dt: datetime) -> str:
    return dt.strftime("%H:%M:%S")


def format_filename_timestamp(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H-%M-%S")


def slugify(text: str) -> str:
    """Fallback slug generator; prefer explicit slugs from config.yaml."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def safe_folder_name(text: str) -> str:
    """Convert a camera display name into a filesystem-safe folder name,
    keeping it human-readable (spaces and '@' are preserved).

    Only characters that are actually unsafe/reserved on common
    filesystems are replaced with '_':  / \\ : * ? " < > |  plus any
    control characters. Leading/trailing whitespace and dots are stripped
    (trailing dots are invalid on Windows).
    """
    cleaned = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", text)
    cleaned = " ".join(cleaned.split())  # collapse runs of whitespace
    return cleaned.strip(" .") or "unnamed_camera"


def extract_json(raw_text: str) -> dict | None:
    """Best-effort extraction of a JSON object from a Claude text response.

    Handles cases where the model wraps JSON in markdown code fences or
    adds stray whitespace/preamble despite instructions to return JSON only.
    Returns None if no valid JSON object could be parsed.
    """
    if not raw_text:
        return None

    text = raw_text.strip()

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first {...} block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            return None

    return None


def image_path_for(images_dir: Path, folder_name: str, dt: datetime) -> Path:
    """Build the standard timestamped image path for a camera snapshot.

    `folder_name` is the per-camera directory name — the camera's display
    name (sanitized via safe_folder_name), e.g. "IH30 @ Carrier Pkwy".
    """
    date_dir = images_dir / folder_name / format_date(dt)
    date_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{format_filename_timestamp(dt)}.jpg"
    return date_dir / filename
