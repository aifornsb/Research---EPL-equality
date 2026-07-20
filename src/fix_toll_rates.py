"""
Repair UNKNOWN cells in data/toll_rates.csv.

Many toll_rate_1/2/3 cells are written as "UNKNOWN" even though the sign's
content was actually captured. There are two distinct causes, handled in two
stages:

  Stage 1 - text reparse (no API, no images, fast, safe):
    The literal characters Claude read off the sign are already stored in the
    `raw_extracted_text` column. When a lane shows a closed indicator the sign
    literally reads "CLSD" (and OCR variants like "CL SD", "CL50", "CLOSED"),
    and when it shows a price it reads e.g. "$0.62". Stage 1 re-parses that
    stored text and fills any UNKNOWN toll_rate_* cell whose corresponding
    left-to-right token is present. Closed indicators normalise to "CLSD";
    dollar readings normalise to "$X.XX". A cell is only ever filled from a
    token that actually exists in the raw text for that lane position - values
    are never invented.

  Stage 2 - Claude re-vision (opt-in, needs ANTHROPIC_API_KEY + the images):
    A residue of cells stay UNKNOWN because the original extraction never
    reported that lane at all (e.g. the third "PAY BY MAIL" panel was omitted,
    or the whole sign was flagged too low-resolution). No amount of re-parsing
    stored text can recover those. With --revision, Stage 2 re-sends the saved
    snapshot image (from the row's `image_path`) to Claude under a stricter,
    per-lane prompt that asks it to report CLSD per lane when closed, and to
    use UNKNOWN only when a lane is truly unreadable. Cells are filled only
    from confident readings; anything unreadable stays UNKNOWN.

Both stages are append-safe and idempotent: only UNKNOWN cells are ever
touched, existing values are left byte-for-byte unchanged, and re-running the
script does not duplicate rows or overwrite already-resolved data.

Usage:
    python -m src.fix_toll_rates                       # Stage 1 only (default)
    python -m src.fix_toll_rates --revision            # Stage 1 + Stage 2
    python -m src.fix_toll_rates --revision --limit 200
    python -m src.fix_toll_rates --dry-run             # report only, no write
    python -m src.fix_toll_rates --csv data/toll_rates.csv --no-backup
"""
from __future__ import annotations

import argparse
import base64
import logging
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger("fix_toll_rates")

RATE_COLS = ["toll_rate_1", "toll_rate_2", "toll_rate_3"]
DEFAULT_MODEL = "claude-sonnet-4-6"

# A dollar amount ($D.DD / D.DD) NOT immediately followed by an uncertainty
# marker ('?', '_', or a letter/digit) - so "0.62" matches but "0.9?" does not.
_MONEY = re.compile(r"\$?\b\d{1,2}\.\d{1,2}\b(?![?\w_])")
# Closed-lane indicator and its common OCR spellings, as whole tokens.
_CLOSED = re.compile(
    r"\b(?:CLSD|CLOSED|CLST|CLOS|CL[.\s]?SD|CL5D|CL50|0L50|CLSO)\b",
    re.IGNORECASE,
)
_COMBINED = re.compile(f"(?:{_MONEY.pattern})|(?:{_CLOSED.pattern})", re.IGNORECASE)


def parse_readings(text: str) -> list[str]:
    """Return the ordered lane readings found in a raw_extracted_text string.

    Each element is either a normalised price ("$X.XX") or "CLSD". Tokens that
    are ambiguous/uncertain (e.g. "0.9?") are intentionally NOT returned, so we
    never fill a cell from a reading the vision step itself was unsure about.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    out: list[str] = []
    for m in _COMBINED.finditer(text):
        tok = m.group(0)
        if _CLOSED.fullmatch(tok):
            out.append("CLSD")
        else:
            out.append(f"${float(tok.replace('$', '')):.2f}")
    return out


def _is_unknown(value: str) -> bool:
    return str(value).strip().upper() == "UNKNOWN"


# --------------------------------------------------------------------------- #
# Stage 1: reparse the stored raw_extracted_text
# --------------------------------------------------------------------------- #
def stage1_text_reparse(df: pd.DataFrame) -> int:
    """Fill UNKNOWN rate cells from each row's raw_extracted_text. Returns the
    number of cells filled. Mutates df in place."""
    filled = 0
    for idx, row in df.iterrows():
        readings = parse_readings(row.get("raw_extracted_text", ""))
        if not readings:
            continue
        for i, col in enumerate(RATE_COLS):
            if _is_unknown(row[col]) and i < len(readings):
                df.at[idx, col] = readings[i]
                filled += 1
    return filled


# --------------------------------------------------------------------------- #
# Stage 2: re-run Claude vision on the saved images for stubborn rows
# --------------------------------------------------------------------------- #
REVISION_PROMPT = """You are re-reading a Texas DOT electronic toll sign (TRDMS) \
from a traffic-camera still. The sign has up to three stacked LED panels, one per \
lane/label (commonly "HOV 2+", "TxTag/TAG", "PAY BY MAIL"), top to bottom.

For each of the three panels, report EXACTLY what it displays:
- If the panel shows a dollar amount, report it verbatim including the currency \
symbol (e.g. "$0.62").
- If the panel shows a closed indicator, report the token "CLSD".
- If, and only if, a panel is genuinely unreadable (out of frame, fully occluded, \
too low-resolution to make out at all), report "UNKNOWN".

Do NOT guess a dollar amount you cannot clearly read, and do NOT leave a panel as \
UNKNOWN if it is plainly showing CLSD. Map panels top-to-bottom to toll_rate_1, \
toll_rate_2, toll_rate_3.

Respond with STRICT JSON ONLY, no markdown, matching exactly:
{
  "toll_rate_1": "string, or CLSD, or UNKNOWN",
  "toll_rate_2": "string, or CLSD, or UNKNOWN",
  "toll_rate_3": "string, or CLSD, or UNKNOWN",
  "raw_extracted_text": "string",
  "extraction_confidence": 0.0
}
"""


def _load_model_name(repo_root: Path) -> str:
    """Model precedence: ANTHROPIC_MODEL env > config/cameras.yaml > default."""
    env = os.environ.get("ANTHROPIC_MODEL")
    if env:
        return env
    cfg = repo_root / "config" / "cameras.yaml"
    if cfg.exists():
        try:
            import yaml  # optional; only needed for Stage 2 config read

            data = yaml.safe_load(cfg.read_text()) or {}
            if isinstance(data.get("anthropic_model"), str):
                return data["anthropic_model"]
        except Exception:  # noqa: BLE001
            pass
    return DEFAULT_MODEL


def _extract_json(text: str) -> dict | None:
    import json

    if not text:
        return None
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


def _normalise_cell(value: str) -> str:
    """Normalise a single Claude-reported panel value to $X.XX / CLSD / UNKNOWN."""
    v = str(value).strip()
    if not v or v.upper() == "UNKNOWN":
        return "UNKNOWN"
    if _CLOSED.fullmatch(v) or v.upper() in {"CLSD", "CLOSED"}:
        return "CLSD"
    m = _MONEY.search(v)
    if m:
        return f"${float(m.group(0).replace('$', '')):.2f}"
    return "UNKNOWN"  # unrecognised / not confidently a value


def stage2_revision(df: pd.DataFrame, repo_root: Path, limit: int | None) -> int:
    """Re-vision rows that still contain any UNKNOWN rate cell. Returns cells
    filled. Requires ANTHROPIC_API_KEY and readable image files."""
    try:
        import anthropic
        from tenacity import retry, stop_after_attempt, wait_fixed
    except ImportError as exc:  # noqa: BLE001
        logger.error("Stage 2 needs the 'anthropic' and 'tenacity' packages: %s", exc)
        return 0

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("Stage 2 requires ANTHROPIC_API_KEY; skipping re-vision.")
        return 0

    client = anthropic.Anthropic(api_key=api_key)
    model = _load_model_name(repo_root)
    logger.info("Stage 2 re-vision using model=%s", model)

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(3), reraise=True)
    def _call(image_b64: str) -> str:
        resp = client.messages.create(
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
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": REVISION_PROMPT},
                    ],
                }
            ],
        )
        return "\n".join(b.text for b in resp.content if b.type == "text")

    needs = df[df[RATE_COLS].apply(lambda r: any(_is_unknown(v) for v in r), axis=1)]
    logger.info("Stage 2: %d rows still contain an UNKNOWN cell.", len(needs))

    filled = 0
    processed = 0
    for idx, row in needs.iterrows():
        if limit is not None and processed >= limit:
            logger.info("Stage 2: hit --limit %d; stopping.", limit)
            break

        raw_path = str(row.get("image_path", "")).strip()
        if not raw_path:
            continue
        img = Path(raw_path)
        if not img.is_absolute():
            img = repo_root / img
        if not img.exists():
            logger.warning("Row %s: image not found, leaving UNKNOWN: %s", idx, img)
            continue

        processed += 1
        try:
            b64 = base64.standard_b64encode(img.read_bytes()).decode("utf-8")
            data = _extract_json(_call(b64))
        except Exception as exc:  # noqa: BLE001
            logger.error("Row %s: re-vision call failed (%s); leaving as-is.", idx, exc)
            continue

        if not isinstance(data, dict):
            logger.warning("Row %s: invalid JSON from re-vision; leaving as-is.", idx)
            continue

        for col in RATE_COLS:
            if _is_unknown(row[col]) and col in data:
                new_val = _normalise_cell(data[col])
                if new_val != "UNKNOWN":
                    df.at[idx, col] = new_val
                    filled += 1
        logger.info("Row %s re-visioned -> %s", idx, df.loc[idx, RATE_COLS].tolist())

    logger.info("Stage 2: processed %d image(s), filled %d cell(s).", processed, filled)
    return filled


# --------------------------------------------------------------------------- #
# Reporting + orchestration
# --------------------------------------------------------------------------- #
def _counts(df: pd.DataFrame) -> tuple[int, int, int]:
    unknown_cells = int(sum((df[c].map(_is_unknown)).sum() for c in RATE_COLS))
    all_unk_rows = int(df[RATE_COLS].apply(lambda r: all(_is_unknown(v) for v in r), axis=1).sum())
    any_unk_rows = int(df[RATE_COLS].apply(lambda r: any(_is_unknown(v) for v in r), axis=1).sum())
    return unknown_cells, all_unk_rows, any_unk_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Repair UNKNOWN toll-rate cells.")
    parser.add_argument("--csv", default="data/toll_rates.csv", help="Path to the CSV to repair.")
    parser.add_argument("--revision", action="store_true", help="Also re-vision stubborn rows via Claude (needs images + API key).")
    parser.add_argument("--limit", type=int, default=None, help="Max images to re-vision in Stage 2.")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing.")
    parser.add_argument("--no-backup", action="store_true", help="Do not write a .bak copy before saving.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    csv_path = Path(args.csv)
    if not csv_path.exists():
        logger.error("CSV not found: %s", csv_path)
        return 1
    repo_root = csv_path.resolve().parent.parent  # data/toll_rates.csv -> repo root

    # Read as strings; never coerce/lose the original text.
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    missing = [c for c in RATE_COLS + ["raw_extracted_text"] if c not in df.columns]
    if missing:
        logger.error("CSV missing required columns: %s", missing)
        return 1

    before = _counts(df)
    logger.info("Before: %d UNKNOWN cells | %d all-UNKNOWN rows | %d rows w/ any UNKNOWN", *before)

    f1 = stage1_text_reparse(df)
    logger.info("Stage 1 filled %d cell(s) from raw_extracted_text.", f1)

    f2 = 0
    if args.revision:
        f2 = stage2_revision(df, repo_root, args.limit)

    after = _counts(df)
    logger.info("After:  %d UNKNOWN cells | %d all-UNKNOWN rows | %d rows w/ any UNKNOWN", *after)
    logger.info("Total cells filled: %d (stage1=%d, stage2=%d)", f1 + f2, f1, f2)

    if args.dry_run:
        logger.info("--dry-run set; not writing any file.")
        return 0
    if (f1 + f2) == 0:
        logger.info("Nothing to change; leaving %s untouched.", csv_path)
        return 0

    if not args.no_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = csv_path.with_suffix(f".{stamp}.bak.csv")
        shutil.copy2(csv_path, backup)
        logger.info("Backup written: %s", backup)

    # Atomic write: temp file then replace, preserving column order.
    tmp = csv_path.with_suffix(".tmp.csv")
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)
    logger.info("Updated %s", csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
