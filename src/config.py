"""
Configuration loading for the TxDOT camera collector.

Loads settings from config/cameras.yaml, allows environment-variable
overrides, and loads secrets (ANTHROPIC_API_KEY) via python-dotenv.
Fails fast and loudly if required secrets are missing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "cameras.yaml"
DATA_DIR = PROJECT_ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
LOGS_DIR = PROJECT_ROOT / "logs"
TOLL_CSV = DATA_DIR / "toll_rates.csv"
TRAFFIC_CSV = DATA_DIR / "traffic_observations.csv"
FAILED_IMAGES_DIR = DATA_DIR / "failed_review"

# Default Claude model used for vision analysis. Can be overridden via the
# ANTHROPIC_MODEL environment variable (see .env.example) without touching
# any source files that call the API.
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-6"

# Minimum acceptable dimensions (pixels) for a captured camera element.
# Used to reject screenshots that are just a small label/icon rather than
# the actual camera feed. Overridable via config/cameras.yaml.
DEFAULT_MIN_MEDIA_WIDTH = 120
DEFAULT_MIN_MEDIA_HEIGHT = 90


@dataclass
class CameraConfig:
    name: str
    slug: str


@dataclass
class ScreenshotConfig:
    wait_after_load_ms: int = 3000
    wait_after_click_ms: int = 2000
    viewport_width: int = 1600
    viewport_height: int = 1000
    min_media_width: int = DEFAULT_MIN_MEDIA_WIDTH
    min_media_height: int = DEFAULT_MIN_MEDIA_HEIGHT
    # Quality settings. The capture code first tries to download the
    # camera's ORIGINAL image file at native resolution (lossless — no
    # scaling or re-encoding). These two settings only affect the
    # screenshot fallback path used when a direct download isn't possible.
    jpeg_quality: int = 95          # 1-100, JPEG quality for fallback screenshots
    device_scale_factor: int = 2    # render page at 2x DPI so fallback shots are sharper


@dataclass
class AppConfig:
    site_url: str
    timezone: str
    interval_minutes: int
    duration_days: int
    toll_camera: CameraConfig
    traffic_camera: CameraConfig
    screenshot: ScreenshotConfig
    anthropic_api_key: str
    anthropic_model: str
    raw: dict[str, Any] = field(default_factory=dict)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Path | None = None, require_api_key: bool = True) -> AppConfig:
    """Load configuration, applying environment variable overrides.

    Raises RuntimeError if ANTHROPIC_API_KEY is missing and
    `require_api_key` is True (the default). Pass `require_api_key=False`
    for capture-only / --skip-analysis runs that never call Claude, so the
    tool can be tested without an API key.
    """
    load_dotenv()  # loads .env if present; safe no-op otherwise

    path = config_path or CONFIG_PATH
    raw = _load_yaml(path)

    site_url = os.getenv("TXDOT_SITE_URL", raw.get("site_url", ""))
    timezone = os.getenv("TIMEZONE", raw.get("timezone", "America/Chicago"))
    interval_minutes = int(os.getenv("INTERVAL_MINUTES", raw.get("interval_minutes", 5)))
    duration_days = int(os.getenv("DURATION_DAYS", raw.get("duration_days", 7)))

    cameras = raw.get("cameras", {})
    toll_raw = cameras.get("toll", {})
    traffic_raw = cameras.get("traffic", {})

    toll_camera = CameraConfig(name=toll_raw["name"], slug=toll_raw["slug"])
    traffic_camera = CameraConfig(name=traffic_raw["name"], slug=traffic_raw["slug"])

    shot_raw = raw.get("screenshot", {})
    screenshot = ScreenshotConfig(
        wait_after_load_ms=int(shot_raw.get("wait_after_load_ms", 3000)),
        wait_after_click_ms=int(shot_raw.get("wait_after_click_ms", 2000)),
        viewport_width=int(shot_raw.get("viewport_width", 1600)),
        viewport_height=int(shot_raw.get("viewport_height", 1000)),
        min_media_width=int(shot_raw.get("min_media_width", DEFAULT_MIN_MEDIA_WIDTH)),
        min_media_height=int(shot_raw.get("min_media_height", DEFAULT_MIN_MEDIA_HEIGHT)),
        jpeg_quality=int(shot_raw.get("jpeg_quality", 95)),
        device_scale_factor=int(shot_raw.get("device_scale_factor", 2)),
    )

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key and require_api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and set "
            "your key, or export it as an environment variable / GitHub secret. "
            "(Use --skip-analysis if you only want to test camera capture "
            "without calling Claude.)"
        )

    anthropic_model = os.getenv(
        "ANTHROPIC_MODEL", raw.get("anthropic_model", DEFAULT_ANTHROPIC_MODEL)
    )

    return AppConfig(
        site_url=site_url,
        timezone=timezone,
        interval_minutes=interval_minutes,
        duration_days=duration_days,
        toll_camera=toll_camera,
        traffic_camera=traffic_camera,
        screenshot=screenshot,
        anthropic_api_key=api_key,
        anthropic_model=anthropic_model,
        raw=raw,
    )


def ensure_directories() -> None:
    """Create data/log directories used at runtime if they don't exist."""
    for d in (DATA_DIR, IMAGES_DIR, LOGS_DIR, FAILED_IMAGES_DIR):
        d.mkdir(parents=True, exist_ok=True)
