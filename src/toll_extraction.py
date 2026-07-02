"""
Toll-rate extraction from IH30 @ Loop 12 WB TRDMS Sta 1251-37 snapshots
using Claude vision. Falls back to UNKNOWN for any value that cannot be
read with reasonable confidence. Never guesses/fabricates values.
"""
from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path

import anthropic
from tenacity import retry, stop_after_attempt, wait_fixed

from src.utils import extract_json

logger = logging.getLogger("collector")

TOLL_PROMPT = """You are analyzing a still image from a Texas DOT traffic camera \
that shows an electronic toll sign (TRDMS - Toll Rate Display Management System).

Look carefully at the toll rate sign(s) visible in the image and extract up to \
three displayed toll rate values (these are typically dollar amounts shown for \
different exit points or lanes, e.g. "$2.50").

Rules:
- Only report a value if it is clearly and legibly visible in the image.
- If fewer than three toll rates are visible or legible, use "UNKNOWN" for the \
missing ones. Do not guess or estimate.
- Do not fabricate values under any circumstances.
- raw_extracted_text should contain the literal text/characters you can read \
from the sign, exactly as displayed (include currency symbols).
- extraction_confidence is a single float from 0.0 to 1.0 representing your \
overall confidence in the extracted values (1.0 = fully certain, 0.0 = no \
readable data).

Respond with STRICT JSON ONLY, no markdown fences, no commentary, matching \
exactly this schema:

{
  "toll_rate_1": "string or UNKNOWN",
  "toll_rate_2": "string or UNKNOWN",
  "toll_rate_3": "string or UNKNOWN",
  "raw_extracted_text": "string",
  "extraction_confidence": 0.0
}
"""

REQUIRED_KEYS = {
    "toll_rate_1",
    "toll_rate_2",
    "toll_rate_3",
    "raw_extracted_text",
    "extraction_confidence",
}


@dataclass
class TollExtractionResult:
    toll_rate_1: str
    toll_rate_2: str
    toll_rate_3: str
    raw_extracted_text: str
    extraction_confidence: float
    valid: bool
    error: str | None = None


def _default_unknown_result(error: str) -> TollExtractionResult:
    return TollExtractionResult(
        toll_rate_1="UNKNOWN",
        toll_rate_2="UNKNOWN",
        toll_rate_3="UNKNOWN",
        raw_extracted_text="",
        extraction_confidence=0.0,
        valid=False,
        error=error,
    )


def _validate(data: dict) -> bool:
    if not isinstance(data, dict):
        return False
    if not REQUIRED_KEYS.issubset(data.keys()):
        return False
    try:
        conf = float(data["extraction_confidence"])
    except (TypeError, ValueError):
        return False
    return 0.0 <= conf <= 1.0


def _encode_image(image_path: Path) -> str:
    return base64.standard_b64encode(image_path.read_bytes()).decode("utf-8")


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3), reraise=True)
def _call_claude(
    client: anthropic.Anthropic, model: str, image_b64: str, media_type: str, prompt: str
) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=500,
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
    logger.info("Claude toll-extraction raw response: %s", raw_text)
    return raw_text


def extract_toll_rates(
    client: anthropic.Anthropic, image_path: Path, model: str
) -> TollExtractionResult:
    """Analyze a toll-camera snapshot and return structured toll rate data.

    `model` is the Claude model identifier to use (see config.py /
    ANTHROPIC_MODEL env var) — never hardcode a model name at the call site.

    Retries once with a stricter prompt if the first response is not
    valid JSON matching the required schema. If both attempts fail,
    returns an UNKNOWN result with `valid=False` so the caller can log
    the failure and preserve the image without crashing the pipeline.
    """
    if not image_path.exists():
        return _default_unknown_result(f"Image not found: {image_path}")

    media_type = "image/jpeg"
    image_b64 = _encode_image(image_path)

    try:
        raw_text = _call_claude(client, model, image_b64, media_type, TOLL_PROMPT)
    except Exception as exc:  # noqa: BLE001
        logger.error("Claude API call failed for toll extraction: %s", exc, exc_info=True)
        return _default_unknown_result(f"Claude API error: {exc}")

    data = extract_json(raw_text)
    if data and _validate(data):
        return TollExtractionResult(
            toll_rate_1=str(data["toll_rate_1"]),
            toll_rate_2=str(data["toll_rate_2"]),
            toll_rate_3=str(data["toll_rate_3"]),
            raw_extracted_text=str(data["raw_extracted_text"]),
            extraction_confidence=float(data["extraction_confidence"]),
            valid=True,
        )

    # Retry once with a stricter prompt
    logger.warning("Invalid/malformed JSON from Claude; retrying with stricter prompt.")
    strict_prompt = (
        TOLL_PROMPT
        + "\n\nIMPORTANT: Your previous response was not valid JSON or was "
        "missing required fields. Respond with ONLY the JSON object, nothing else."
    )
    try:
        raw_text_retry = _call_claude(client, model, image_b64, media_type, strict_prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("Claude API retry failed for toll extraction: %s", exc, exc_info=True)
        return _default_unknown_result(f"Claude API error on retry: {exc}")

    data_retry = extract_json(raw_text_retry)
    if data_retry and _validate(data_retry):
        return TollExtractionResult(
            toll_rate_1=str(data_retry["toll_rate_1"]),
            toll_rate_2=str(data_retry["toll_rate_2"]),
            toll_rate_3=str(data_retry["toll_rate_3"]),
            raw_extracted_text=str(data_retry["raw_extracted_text"]),
            extraction_confidence=float(data_retry["extraction_confidence"]),
            valid=True,
        )

    logger.error(
        "Toll extraction failed validation after retry. raw=%s", raw_text_retry
    )
    return _default_unknown_result("Invalid JSON from Claude after retry")
