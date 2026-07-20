"""
Second-pass validation: independently re-analyze every traffic image with
Claude vision and record the results for agreement analysis against
data/traffic_observations.csv (the first pass).

Idempotent and chunked: processed images are tracked in the output CSV, so the
script can be run repeatedly (locally or in GitHub Actions) until complete.

Usage:
    python -m src.validate_dataset --max-images 150            # one chunk
    python -m src.validate_dataset --sample 300 --seed 42      # random subsample
    python -m src.validate_dataset --daylight-only             # 07:00-20:00 frames only

Output: data/validation_pass2.csv  (one row per vehicle, same core schema
as traffic_observations.csv, plus pass2_ prefix-free columns and image_path
as the join key).
"""

import argparse
import base64
import csv
import json
import re
import sys
import time
from pathlib import Path

import anthropic

REPO_ROOT = Path(__file__).resolve().parents[1]
FIRST_PASS = REPO_ROOT / "data" / "traffic_observations.csv"
OUTPUT = REPO_ROOT / "data" / "validation_pass2.csv"
RUNNER_PREFIX = "/home/runner/work/Research---EPL-equality/Research---EPL-equality/"

MODEL = "claude-sonnet-4-6"

PROMPT = """Analyze this traffic camera image of IH-30 at Carrier Pkwy in Dallas.
The two lanes in the middle separated by concrete barriers are Express lanes.
All other visible lanes are General Purpose lanes.

For every clearly visible vehicle, report it. Return STRICT JSON ONLY, no
markdown fences, matching exactly:

{
  "vehicles": [
    {
      "sequence_number": 1,
      "lane_type": "Express | General Purpose | Unknown",
      "lane_description": "string",
      "direction_facing": "string",
      "vehicle_make": "string or UNKNOWN",
      "vehicle_model": "string or UNKNOWN",
      "vehicle_body_type": "string or UNKNOWN",
      "vehicle_color": "string or UNKNOWN",
      "vehicle_confidence": 0.0
    }
  ]
}

Rules: never guess make/model that is not visually supportable — use UNKNOWN.
If the image is too dark or vehicles too distant to assess, return fewer
vehicles rather than low-quality guesses."""

FIELDS = [
    "image_path", "sequence_number", "lane_type", "lane_description",
    "direction_facing", "vehicle_make", "vehicle_model",
    "vehicle_body_type", "vehicle_color", "vehicle_confidence",
]


def load_done() -> set:
    if not OUTPUT.exists():
        return set()
    with OUTPUT.open() as f:
        return {row["image_path"] for row in csv.DictReader(f)}


def image_paths_from_first_pass(daylight_only: bool) -> list:
    paths = []
    seen = set()
    with FIRST_PASS.open() as f:
        for row in csv.DictReader(f):
            p = row["image_path"]
            if p in seen:
                continue
            seen.add(p)
            if daylight_only:
                m = re.search(r"_(\d{2})-\d{2}-\d{2}\.jpg$", p)
                if m and not (7 <= int(m.group(1)) <= 20):
                    continue
            paths.append(p)
    return paths


def to_local(path: str) -> Path:
    return REPO_ROOT / path.replace(RUNNER_PREFIX, "")


def analyze(client: anthropic.Anthropic, img: Path, strict_retry: bool = False) -> list:
    data = base64.standard_b64encode(img.read_bytes()).decode()
    prompt = PROMPT if not strict_retry else (
        PROMPT + "\n\nIMPORTANT: your previous response was not valid JSON. "
        "Respond with the JSON object ONLY — no prose, no code fences.")
    msg = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/jpeg", "data": data}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
    return json.loads(text).get("vehicles", [])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-images", type=int, default=200,
                    help="images to process this run (chunking for CI time limits)")
    ap.add_argument("--sample", type=int, default=0,
                    help="if >0, restrict the whole job to a fixed random subsample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--daylight-only", action="store_true")
    args = ap.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    done = load_done()
    paths = image_paths_from_first_pass(args.daylight_only)

    if args.sample:
        import random
        rng = random.Random(args.seed)
        paths = sorted(rng.sample(paths, min(args.sample, len(paths))))

    todo = [p for p in paths if p not in done]
    print(f"{len(paths)} target images, {len(done)} done, {len(todo)} remaining")
    todo = todo[: args.max_images]

    new_file = not OUTPUT.exists()
    with OUTPUT.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()

        for k, p in enumerate(todo, 1):
            local = to_local(p)
            if not local.exists():
                print(f"[{k}/{len(todo)}] MISSING {local}")
                writer.writerow({"image_path": p, "sequence_number": "MISSING_IMAGE"})
                f.flush()
                continue
            try:
                try:
                    vehicles = analyze(client, local)
                except (json.JSONDecodeError, KeyError):
                    vehicles = analyze(client, local, strict_retry=True)
            except Exception as exc:
                print(f"[{k}/{len(todo)}] FAILED {p}: {exc}", file=sys.stderr)
                time.sleep(5)
                continue

            if not vehicles:
                writer.writerow({"image_path": p, "sequence_number": "NO_VEHICLES"})
            for v in vehicles:
                writer.writerow({
                    "image_path": p,
                    "sequence_number": v.get("sequence_number", ""),
                    "lane_type": v.get("lane_type", "Unknown"),
                    "lane_description": v.get("lane_description", ""),
                    "direction_facing": v.get("direction_facing", "unknown"),
                    "vehicle_make": v.get("vehicle_make", "UNKNOWN"),
                    "vehicle_model": v.get("vehicle_model", "UNKNOWN"),
                    "vehicle_body_type": v.get("vehicle_body_type", "UNKNOWN"),
                    "vehicle_color": v.get("vehicle_color", "UNKNOWN"),
                    "vehicle_confidence": v.get("vehicle_confidence", 0.0),
                })
            f.flush()
            print(f"[{k}/{len(todo)}] {p} -> {len(vehicles)} vehicles")
            time.sleep(1)  # gentle pacing

    print("chunk complete")


if __name__ == "__main__":
    main()
