"""
Traffic/vehicle extraction from IH30 @ Carrier Pkwy snapshots using
Claude vision. Identifies each visible vehicle, its lane type (Express
vs General Purpose based on the concrete-barrier middle lanes), and
visually-supportable make/model/year/body/color. Never fabricates
vehicle details; unknown fields are explicitly marked UNKNOWN.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from tenacity import retry, stop_after_attempt, wait_fixed

from src.utils import extract_json

logger = logging.getLogger("collector")

VALID_LANE_TYPES = {"Express", "General Purpose", "Unknown"}

TRAFFIC_PROMPT = """You are analyzing a still image from a Texas DOT traffic \
camera at IH30 @ Carrier Pkwy in Dallas, TX.

Lane rule (must follow exactly):
- The two lanes in the middle of the roadway that are separated by concrete \
barriers are "Express" lanes.
- All other visible lanes are "General Purpose" lanes.
- If you cannot determine which lane a vehicle is in, use "Unknown".

For every distinct vehicle visible in the image, provide a structured \
observation. For each vehicle:
- sequence_number: an integer starting at 1, unique within this image, in \
left-to-right or front-to-back order as convenient.
- direction_facing: describe the apparent direction the vehicle is traveling/ \
facing, e.g. "toward camera", "away from camera", "left", "right", or \
"unknown".
- lane_type: exactly one of "Express", "General Purpose", or "Unknown", per \
the lane rule above.
- lane_description: a short free-text description of the specific lane \
(e.g. "middle express lane 1", "rightmost general purpose lane").
- vehicle_make, vehicle_model, vehicle_year_estimate: only provide a specific \
value if it is visually supportable from body shape, badging, or other clear \
visual cues. Otherwise use "UNKNOWN". Do not guess or hallucinate.
- vehicle_body_type: e.g. "sedan", "pickup truck", "SUV", "semi-truck", "van", \
"motorcycle", "bus", or "UNKNOWN" if unclear.
- vehicle_color: the dominant visible color, or "UNKNOWN" if not determinable.
- vehicle_confidence: a float 0.0-1.0 for your confidence in this vehicle's \
identification overall.

If no vehicles are visible, return an empty "vehicles" array.

Respond with STRICT JSON ONLY, no markdown fences, no commentary, matching \
exactly this schema:

{
  "vehicles": [
    {
      "sequence_number": 1,
      "direction_facing": "string",
      "lane_type": "Express | General Purpose | Unknown",
      "lane_description": "string",
      "vehicle_make": "string or UNKNOWN",
      "vehicle_model": "string or UNKNOWN",
      "vehicle_year_estimate": "string or UNKNOWN",
      "vehicle_body_type": "string or UNKNOWN",
      "vehicle_color": "string or UNKNOWN",
      "vehicle_confidence": 0.0
    }
  ]
}
"""

REQUIRED_VEHICLE_KEYS = {
    "sequence_number",
    "direction_facing",
    "lane_type",
    "lane_description",
    "vehicle_make",
    "vehicle_model",
    "vehicle_year_estimate",
    "vehicle_body_type",
    "vehicle_color",
    "vehicle_confidence",
}


@dataclass
class VehicleObservation:
    sequence_number: int
    direction_facing: str
    lane_type: str
    lane_description: str
    vehicle_make: str
    vehicle_model: str
    vehicle_year_estimate: str
    vehicle_body_type: str
    vehicle_color: str
    vehicle_confidence: float


@dataclass
class TrafficExtractionResult:
    vehicles: list[VehicleObservation] = field(default_factory=list)
    valid: bool = False
    error: str | None = None


def _validate_vehicle(v: dict) -> bool:
    if not isinstance(v, dict):
        return False
    if not REQUIRED_VEHICLE_KEYS.issubset(v.keys()):
        return False
    if v["lane_type"] not in VALID_LANE_TYPES:
        return False
    try:
        int(v["sequence_number"])
        conf = float(v["vehicle_confidence"])
    except (TypeError, ValueError):
        return False
    return 0.0 <= conf <= 1.0


def _validate(data: dict) -> bool:
    if not isinstance(data, dict) or "vehicles" not in data:
        return False
    if not isinstance(data["vehicles"], list):
        return False
    return all(_validate_vehicle(v) for v in data["vehicles"])


def _to_observations(data: dict) -> list[VehicleObservation]:
    obs = []
    for v in data["vehicles"]:
        obs.append(
            VehicleObservation(
                sequence_number=int(v["sequence_number"]),
                direction_facing=str(v["direction_facing"]),
                lane_type=str(v["lane_type"]),
                lane_description=str(v["lane_description"]),
                vehicle_make=str(v["vehicle_make"]),
                vehicle_model=str(v["vehicle_model"]),
                vehicle_year_estimate=str(v["vehicle_year_estimate"]),
                vehicle_body_type=str(v["vehicle_body_type"]),
                vehicle_color=str(v["vehicle_color"]),
                vehicle_confidence=float(v["vehicle_confidence"]),
            )
        )
    return obs


def _encode_image(image_path: Path) -> str:
    return base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3), reraise=True)
def _call_claude(
    client: anthropic.Anthropic, model: str, image_b64: str, media_type: str, prompt: str
) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    text_blocks = [b.text for b in response.content if b.type == "text"]
    raw_text = "\n".join(text_blocks)
    logger.info("Claude traffic-extraction raw response: %s", raw_text)
    return raw_text


def extract_traffic(
    client: anthropic.Anthropic, image_path: Path, model: str
) -> TrafficExtractionResult:
    """Analyze a traffic-camera snapshot and return structured vehicle data.

    `model` is the Claude model identifier to use (see config.py /
    ANTHROPIC_MODEL env var) — never hardcode a model name at the call site.

    Retries once with a stricter prompt on invalid JSON. If both attempts
    fail validation, returns an empty, invalid result so the caller can
    log the failure, save the image for manual review, and continue.
    """
    if not image_path.exists():
        return TrafficExtractionResult(vehicles=[], valid=False, error=f"Image not found: {image_path}")

    media_type = "image/jpeg"
    image_b64 = _encode_image(image_path)

    try:
        raw_text = _call_claude(client, model, image_b64, media_type, TRAFFIC_PROMPT)
    except Exception as exc:  # noqa: BLE001
        logger.error("Claude API call failed for traffic extraction: %s", exc, exc_info=True)
        return TrafficExtractionResult(vehicles=[], valid=False, error=f"Claude API error: {exc}")

    data = extract_json(raw_text)
    if data and _validate(data):
        return TrafficExtractionResult(vehicles=_to_observations(data), valid=True)

    logger.warning("Invalid/malformed traffic JSON from Claude; retrying with stricter prompt.")
    strict_prompt = (
        TRAFFIC_PROMPT
        + "\n\nIMPORTANT: Your previous response was not valid JSON or was "
        "missing required fields / used an invalid lane_type value. "
        "Respond with ONLY the JSON object, nothing else."
    )
    try:
        raw_text_retry = _call_claude(client, model, image_b64, media_type, strict_prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("Claude API retry failed for traffic extraction: %s", exc, exc_info=True)
        return TrafficExtractionResult(vehicles=[], valid=False, error=f"Claude API error on retry: {exc}")

    data_retry = extract_json(raw_text_retry)
    if data_retry and _validate(data_retry):
        return TrafficExtractionResult(vehicles=_to_observations(data_retry), valid=True)

    logger.error("Traffic extraction failed validation after retry. raw=%s", raw_text_retry)
    return TrafficExtractionResult(vehicles=[], valid=False, error="Invalid JSON from Claude after retry")
