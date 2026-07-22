"""
price_vehicles.py  -  Step 2: attach a market price range to each vehicle.

Reads the step-1 output (vehicles.csv), finds every unique (make, model, year)
that was confidently identified, estimates a current used-market USD price range
for each, and writes an enriched CSV with price columns added to every row.
Unidentified vehicles get blank price fields (never fabricated).

Pricing source
--------------
* ``--provider claude`` (default): estimates a used-market range with Claude via
  the Batch API, from its market knowledge for the given region. It is
  APPROXIMATE and knowledge-based (the model has a training cutoff); it is not a
  live market feed and not a per-VIN appraisal. Because it prices each unique
  make/model/year only once (de-duplicated), a whole dataset costs cents.
* ``--provider api``: calls an external valuation API you configure via
  ``VEHICLE_PRICE_API_URL`` and ``VEHICLE_PRICE_API_KEY``. Use this for live
  market data. Adapt ``_parse_api`` to your provider's JSON.

Dedup / cache / resume
----------------------
Unique specs are priced once and cached to ``<output_dir>/price_cache.json``.
Re-running reuses the cache, so it never re-pays for a spec it already priced.

Usage
-----
    export ANTHROPIC_API_KEY=...
    pip install anthropic httpx
    python price_vehicles.py --input data/output/vehicles/vehicles.csv \
        --output data/output/vehicles/vehicles_priced.csv --location "Dallas, Texas"
    python price_vehicles.py --input <csv> --estimate      # cost projection only
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("price_vehicles")

UNKNOWN = {"", "Unknown", "unknown", "UNKNOWN", None}
PRICE_COLUMNS = [
    "price_range_low", "price_range_typical", "price_range_high",
    "price_range_currency", "price_source", "price_confidence", "price_basis",
]
PRICES = {  # per-million tokens
    "claude-fable-5": (10.0, 50.0), "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0), "claude-haiku-4-5": (1.0, 5.0),
}

PRICE_PROMPT = """You are a used-vehicle pricing analyst estimating a typical used-market value.
Vehicle: {year} {make} {model}
Market / region: {location}
Assume typical mileage and average condition for the vehicle's age.

Give an APPROXIMATE current used-market price range in US dollars, from general market
knowledge. This is a rough estimate, not a per-VIN appraisal. If you cannot reasonably
estimate it, use nulls and confidence 0.

Respond with STRICT JSON ONLY, no prose:
{{"price_low": null, "price_typical": null, "price_high": null, "currency": "USD", "confidence": 0.0, "basis": "one short sentence"}}"""


def _extract_json(text: str):
    if not text:
        return None
    t = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(t[i : j + 1])
            except json.JSONDecodeError:
                return None
    return None


def spec_key(make: str, model: str, yl: str, yh: str) -> str:
    return f"{make}|{model}|{yl}|{yh}".lower()


def year_str(yl: str, yh: str) -> str:
    yl, yh = str(yl).strip(), str(yh).strip()
    if yl and yh and yl != yh:
        return f"{yl}-{yh}"
    if yl:
        return yl
    if yh:
        return yh
    return "unknown model year (assume roughly 5-8 years old)"


# --------------------------------------------------------------------------- #
def read_rows(path: Path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def collect_specs(rows: list[dict]) -> dict[str, dict]:
    """Unique identified specs -> {make, model, year_low, year_high}."""
    specs: dict[str, dict] = {}
    for r in rows:
        make, model = r.get("make"), r.get("model")
        if make in UNKNOWN or model in UNKNOWN:
            continue
        yl, yh = r.get("year_low", ""), r.get("year_high", "")
        k = spec_key(make, model, yl, yh)
        if k not in specs:
            specs[k] = {"make": make, "model": model, "year_low": yl, "year_high": yh}
    return specs


# ------------------------------------------------------------- claude batch --
def _client():
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    return anthropic.Anthropic(api_key=key)


def _price_request(cid: str, spec: dict, model: str, location: str) -> dict:
    text = PRICE_PROMPT.format(year=year_str(spec["year_low"], spec["year_high"]),
                               make=spec["make"], model=spec["model"], location=location)
    return {"custom_id": cid,
            "params": {"model": model, "max_tokens": 300,
                       "messages": [{"role": "user", "content": text}]}}


def _normalise(data: dict, min_conf: float) -> dict:
    def num(x):
        try:
            return round(float(x), -1)
        except (TypeError, ValueError):
            return None
    conf = float(data.get("confidence", 0) or 0)
    low, typ, high = num(data.get("price_low")), num(data.get("price_typical")), num(data.get("price_high"))
    if conf < min_conf or low is None or high is None:
        return {"low": None, "typical": None, "high": None, "currency": "USD",
                "source": "low_confidence", "confidence": round(conf, 3),
                "basis": str(data.get("basis", ""))[:160]}
    return {"low": low, "typical": typ, "high": high, "currency": data.get("currency", "USD"),
            "source": "claude_estimate", "confidence": round(conf, 3),
            "basis": str(data.get("basis", ""))[:160]}


def price_with_claude(specs: dict[str, dict], cache: dict, out_dir: Path, model: str,
                      location: str, min_conf: float, max_wait: int) -> None:
    todo = {k: v for k, v in specs.items() if k not in cache}
    if not todo:
        logger.info("all %d spec(s) already cached; nothing to price", len(specs))
        return
    logger.info("pricing %d new unique spec(s) via Batch API (%s)", len(todo), model)
    client = _client()
    keys = list(todo)
    cid_to_key = {f"s{i}": k for i, k in enumerate(keys)}
    requests = [_price_request(cid, todo[k], model, location) for cid, k in cid_to_key.items()]

    batch = client.messages.batches.create(requests=requests)
    manifest = out_dir / "price_batch.json"
    manifest.write_text(json.dumps({"batch_id": batch.id, "cid_to_key": cid_to_key}, indent=2))
    logger.info("submitted price batch %s (%d requests)", batch.id, len(requests))

    deadline = time.time() + max_wait
    while True:
        status = client.messages.batches.retrieve(batch.id).processing_status
        if status == "ended":
            break
        if time.time() > deadline:
            logger.warning("price batch still %s at max-wait; re-run to collect.", status)
            return
        logger.info("price batch %s: %s; waiting 20s", batch.id, status)
        time.sleep(20)

    for entry in client.messages.batches.results(batch.id):
        k = cid_to_key.get(entry.custom_id)
        if not k:
            continue
        res = entry.result
        if getattr(res, "type", None) == "succeeded":
            text = "".join(b.text for b in res.message.content if getattr(b, "type", None) == "text")
            data = _extract_json(text) or {}
            cache[k] = _normalise(data, min_conf)
        else:
            cache[k] = {"low": None, "typical": None, "high": None, "currency": "USD",
                        "source": f"batch_{getattr(res, 'type', 'error')}", "confidence": 0.0,
                        "basis": ""}
    (out_dir / "price_cache.json").write_text(json.dumps(cache, indent=2))
    logger.info("priced %d spec(s); cache updated", len(todo))


# --------------------------------------------------------------- external API --
def _parse_api(data: dict, min_conf: float) -> dict:
    low = data.get("low") or data.get("price_low")
    high = data.get("high") or data.get("price_high")
    typ = data.get("typical") or data.get("price_typical") or data.get("price_mean")
    if low is None or high is None:
        return {"low": None, "typical": None, "high": None, "currency": "USD",
                "source": "api_no_data", "confidence": 0.0, "basis": ""}
    return {"low": low, "typical": typ, "high": high, "currency": data.get("currency", "USD"),
            "source": data.get("source", "external_api"), "confidence": 1.0,
            "basis": f"{data.get('comparable_listing_count', '?')} comparable listings"}


def price_with_api(specs: dict[str, dict], cache: dict, out_dir: Path, location: str,
                   min_conf: float) -> None:
    import httpx

    url = os.environ.get("VEHICLE_PRICE_API_URL")
    key = os.environ.get("VEHICLE_PRICE_API_KEY")
    if not url or not key:
        raise RuntimeError("provider=api needs VEHICLE_PRICE_API_URL and VEHICLE_PRICE_API_KEY.")
    todo = {k: v for k, v in specs.items() if k not in cache}
    logger.info("pricing %d spec(s) via external API", len(todo))
    with httpx.Client(timeout=20.0) as client:
        for k, spec in todo.items():
            payload = {"make": spec["make"], "model": spec["model"],
                       "year_min": spec["year_low"], "year_max": spec["year_high"],
                       "location": location}
            try:
                resp = client.post(url, json=payload,
                                   headers={"Authorization": f"Bearer {key}"})
                resp.raise_for_status()
                cache[k] = _parse_api(resp.json(), min_conf)
            except Exception as exc:  # noqa: BLE001
                logger.warning("api lookup failed for %s: %s", k, exc)
                cache[k] = {"low": None, "typical": None, "high": None, "currency": "USD",
                            "source": "api_error", "confidence": 0.0, "basis": ""}
    (out_dir / "price_cache.json").write_text(json.dumps(cache, indent=2))


# --------------------------------------------------------------------- write --
def enrich_and_write(rows: list[dict], fieldnames: list[str], cache: dict,
                     out_path: Path) -> tuple[int, int]:
    out_fields = list(fieldnames) + [c for c in PRICE_COLUMNS if c not in fieldnames]
    priced = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            make, model = r.get("make"), r.get("model")
            row = dict(r)
            if make in UNKNOWN or model in UNKNOWN:
                est = {"source": "not_identified"}
            else:
                est = cache.get(spec_key(make, model, r.get("year_low", ""),
                                         r.get("year_high", "")), {"source": "no_estimate"})
            row["price_range_low"] = "" if est.get("low") is None else est["low"]
            row["price_range_typical"] = "" if est.get("typical") is None else est["typical"]
            row["price_range_high"] = "" if est.get("high") is None else est["high"]
            row["price_range_currency"] = est.get("currency", "USD") if est.get("low") is not None else ""
            row["price_source"] = est.get("source", "")
            row["price_confidence"] = est.get("confidence", "")
            row["price_basis"] = est.get("basis", "")
            if est.get("low") is not None:
                priced += 1
            w.writerow(row)
    return priced, len(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Step 2: market price range for identified vehicles.")
    ap.add_argument("--input", required=True, help="Step-1 vehicles.csv")
    ap.add_argument("--output", default=None, help="Enriched CSV (default: <input>_priced.csv)")
    ap.add_argument("--provider", choices=["claude", "api"], default="claude")
    ap.add_argument("--model", default="claude-opus-4-8",
                    help="Pricing model for provider=claude (knowledge task; Opus is a good default).")
    ap.add_argument("--location", default="Dallas, Texas")
    ap.add_argument("--min-price-conf", type=float, default=0.40)
    ap.add_argument("--max-wait", type=int, default=7200)
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error("input CSV not found: %s", in_path)
        return 2
    rows, fieldnames = read_rows(in_path)
    specs = collect_specs(rows)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_priced.csv")
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("%d row(s), %d unique identified spec(s) to price", len(rows), len(specs))

    if args.estimate:
        # ~250 input + ~120 output tokens per unique spec
        n = len(specs)
        tin, tout = n * 250, n * 120
        pi, po = PRICES.get(args.model, PRICES["claude-opus-4-8"])
        full = tin / 1e6 * pi + tout / 1e6 * po
        print(f"\nunique specs to price: {n}")
        print(f"est. cost on {args.model}: standard ${full:.2f}  |  batch (-50%) ${full/2:.2f}")
        print("(pricing is text-only and de-duplicated by spec, hence very cheap)")
        return 0

    cache_path = out_dir / "price_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    if specs:
        if args.provider == "claude":
            price_with_claude(specs, cache, out_dir, args.model, args.location,
                              args.min_price_conf, args.max_wait)
        else:
            price_with_api(specs, cache, out_dir, args.location, args.min_price_conf)

    priced, total = enrich_and_write(rows, fieldnames, cache, out_path)
    logger.info("Done. wrote %s | %d/%d rows have a price range", out_path, priced, total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
