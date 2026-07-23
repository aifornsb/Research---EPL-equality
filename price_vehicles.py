"""
price_vehicles.py  -  Step 2: attach a market price range to each vehicle.

Reads the step-1 output (vehicles.csv) and adds a used-market USD price range to
every row it can. To maximise coverage it prices at the BEST GRANULARITY each
vehicle supports, rather than all-or-nothing on make+model:

  Tier 1  make + model + year   -> precise range          (price_source claude_make_model)
  Tier 2  make + body type      -> make-level range        (price_source claude_make_body)
  Tier 3  body type only        -> segment range           (price_source claude_body_segment)

Body type is recorded by step 1 for nearly every vehicle (it is far easier to see
than make/model), so Tier 3 gives a useful value proxy for the majority of
vehicles. Coarser tiers get wider, lower-confidence ranges - honest about the
added uncertainty. Choose the coarsest tier allowed with --detail:
  --detail model  (Tier 1 only; most precise, lowest coverage)
  --detail make   (down to Tier 2)
  --detail body   (down to Tier 3; default, highest meaningful coverage)

Each unique spec (at its tier) is priced once via the Batch API and cached, so a
whole dataset costs cents. provider=claude uses Claude's market knowledge
(approximate, not a live feed, not a per-VIN appraisal); provider=api calls an
external valuation API you configure.

Usage
-----
    python price_vehicles.py --input data/output/vehicles/vehicles.csv \
        --output data/output/vehicles/vehicles_priced.csv --detail body
    python price_vehicles.py --input <csv> --estimate
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
PRICES = {
    "claude-fable-5": (10.0, 50.0), "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0), "claude-haiku-4-5": (1.0, 5.0),
}
TIER_SOURCE = {1: "claude_make_model", 2: "claude_make_body", 3: "claude_body_segment"}
DETAIL_FLOOR = {"model": 1, "make": 2, "body": 3}

PRICE_PROMPT = """You are a used-vehicle pricing analyst for the {location} market.
Estimate the current used-market price range in US dollars for:

  {description}

Assume typical mileage and average condition for the age. The LESS specific the
description, the WIDER your range should be to reflect that uncertainty. This is
an approximate estimate from general market knowledge, not a per-VIN appraisal.
If you truly cannot estimate it, use nulls and confidence 0.

Respond with STRICT JSON ONLY, no prose:
{{"price_low": null, "price_typical": null, "price_high": null, "currency": "USD", "confidence": 0.0, "basis": "one short sentence"}}"""


def _known(v) -> bool:
    return v not in UNKNOWN


def _year_phrase(yl: str, yh: str) -> str:
    yl, yh = str(yl or "").strip(), str(yh or "").strip()
    if yl and yh and yl != yh:
        return f"{yl}-{yh}"
    if yl:
        return yl
    return ""


def _body_words(body: str) -> str:
    return str(body or "vehicle").replace("_", " ")


def classify_tier(row: dict, floor_tier: int):
    """Return (tier, key, description) at the best granularity <= floor_tier, or
    (None, None, None) if the row cannot be priced at the allowed detail."""
    make, model, body = row.get("make"), row.get("model"), row.get("body_type")
    yr = _year_phrase(row.get("year_low"), row.get("year_high"))
    ystr = f"{yr} " if yr else ""

    if _known(make) and _known(model) and floor_tier >= 1:
        desc = f"a {ystr}{make} {model}".strip()
        return 1, f"1|{make}|{model}|{yr}".lower(), desc
    if _known(make) and _known(body) and floor_tier >= 2:
        yphr = ystr if yr else "recent-model-year "
        desc = f"a {yphr}{make} {_body_words(body)} (specific model unknown)"
        return 2, f"2|{make}|{_body_words(body)}|{yr}".lower(), desc
    if _known(body) and floor_tier >= 3:
        yphr = ystr if yr else "roughly 5-8 year old "
        desc = f"a typical {yphr}used {_body_words(body)} (make, model, and year uncertain)"
        return 3, f"3|{_body_words(body)}|{yr}".lower(), desc
    return None, None, None


def read_rows(path: Path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


def collect_specs(rows: list[dict], floor_tier: int) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    for r in rows:
        tier, key, desc = classify_tier(r, floor_tier)
        if key and key not in specs:
            specs[key] = {"tier": tier, "description": desc}
    return specs


# ------------------------------------------------------------------ claude ---
def _client():
    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    return anthropic.Anthropic(api_key=key)


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


def _normalise(data: dict, tier: int, min_conf: float) -> dict:
    def num(x):
        try:
            return round(float(x), -1)
        except (TypeError, ValueError):
            return None
    conf = float(data.get("confidence", 0) or 0)
    low, typ, high = num(data.get("price_low")), num(data.get("price_typical")), num(data.get("price_high"))
    basis = str(data.get("basis", ""))[:160]
    if conf < min_conf or low is None or high is None:
        return {"low": None, "typical": None, "high": None, "currency": "USD",
                "source": "low_confidence", "confidence": round(conf, 3), "basis": basis}
    return {"low": low, "typical": typ, "high": high, "currency": data.get("currency", "USD"),
            "source": TIER_SOURCE.get(tier, "claude_estimate"),
            "confidence": round(conf, 3), "basis": basis}


def price_with_claude(specs, cache, out_dir, model, location, min_conf, max_wait):
    todo = {k: v for k, v in specs.items() if k not in cache}
    if not todo:
        logger.info("all %d spec(s) cached; nothing to price", len(specs))
        return
    logger.info("pricing %d new spec(s) via Batch API (%s)", len(todo), model)
    client = _client()
    keys = list(todo)
    cid_to_key = {f"s{i}": k for i, k in enumerate(keys)}
    requests = [
        {"custom_id": cid,
         "params": {"model": model, "max_tokens": 300,
                    "messages": [{"role": "user",
                                  "content": PRICE_PROMPT.format(
                                      location=location, description=todo[k]["description"])}]}}
        for cid, k in cid_to_key.items()
    ]
    batch = client.messages.batches.create(requests=requests)
    (out_dir / "price_batch.json").write_text(
        json.dumps({"batch_id": batch.id, "cid_to_key": cid_to_key}, indent=2))
    logger.info("submitted price batch %s (%d requests)", batch.id, len(requests))

    deadline = time.time() + max_wait
    while True:
        status = client.messages.batches.retrieve(batch.id).processing_status
        if status == "ended":
            break
        if time.time() > deadline:
            logger.warning("price batch still %s at max-wait; re-run to collect.", status)
            return
        logger.info("price batch: %s; waiting 20s", status)
        time.sleep(20)

    for entry in client.messages.batches.results(batch.id):
        k = cid_to_key.get(entry.custom_id)
        if not k:
            continue
        res = entry.result
        if getattr(res, "type", None) == "succeeded":
            text = "".join(b.text for b in res.message.content if getattr(b, "type", None) == "text")
            cache[k] = _normalise(_extract_json(text) or {}, specs[k]["tier"], min_conf)
        else:
            cache[k] = {"low": None, "typical": None, "high": None, "currency": "USD",
                        "source": f"batch_{getattr(res, 'type', 'error')}", "confidence": 0.0,
                        "basis": ""}
    (out_dir / "price_cache.json").write_text(json.dumps(cache, indent=2))
    logger.info("priced %d spec(s)", len(todo))


# ------------------------------------------------------------------ write ----
def enrich_and_write(rows, fieldnames, cache, out_path, floor_tier):
    out_fields = list(fieldnames) + [c for c in PRICE_COLUMNS if c not in fieldnames]
    priced = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in rows:
            _, key, _ = classify_tier(r, floor_tier)
            est = cache.get(key, {"source": "not_identified"}) if key else {"source": "not_identified"}
            row = dict(r)
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
    ap = argparse.ArgumentParser(description="Step 2: tiered market price range for vehicles.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--provider", choices=["claude"], default="claude")
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--location", default="Dallas, Texas")
    ap.add_argument("--detail", choices=["model", "make", "body"], default="body",
                    help="Coarsest pricing tier allowed (body = most coverage).")
    ap.add_argument("--min-price-conf", type=float, default=0.35)
    ap.add_argument("--max-wait", type=int, default=7200)
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        logger.error("input CSV not found: %s", in_path)
        return 2
    floor = DETAIL_FLOOR[args.detail]
    rows, fieldnames = read_rows(in_path)
    specs = collect_specs(rows, floor)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_priced.csv")
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # coverage projection
    tiers = [classify_tier(r, floor)[0] for r in rows]
    coverable = sum(1 for t in tiers if t is not None)
    by_tier = {1: tiers.count(1), 2: tiers.count(2), 3: tiers.count(3)}
    logger.info("%d rows | priceable at --detail %s: %d (%.0f%%) | unique specs: %d | "
                "tier1=%d tier2=%d tier3=%d",
                len(rows), args.detail, coverable, 100 * coverable / max(1, len(rows)),
                len(specs), by_tier[1], by_tier[2], by_tier[3])

    if args.estimate:
        n = len(specs)
        tin, tout = n * 260, n * 120
        pi, po = PRICES.get(args.model, PRICES["claude-opus-4-8"])
        full = tin / 1e6 * pi + tout / 1e6 * po
        print(f"\nunique specs to price: {n}")
        print(f"est. cost on {args.model}: standard ${full:.2f}  |  batch (-50%) ${full/2:.2f}")
        print(f"projected coverage at --detail {args.detail}: "
              f"{100 * coverable / max(1, len(rows)):.0f}% of rows")
        return 0

    cache_path = out_dir / "price_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if specs:
        price_with_claude(specs, cache, out_dir, args.model, args.location,
                          args.min_price_conf, args.max_wait)

    priced, total = enrich_and_write(rows, fieldnames, cache, out_path, floor)
    logger.info("Done. wrote %s | %d/%d rows priced (%.0f%%)",
                out_path, priced, total, 100 * priced / max(1, total))
    return 0


if __name__ == "__main__":
    sys.exit(main())
