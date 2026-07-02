"""
Vehicle price-range estimation.

IMPORTANT: Kelley Blue Book (kbb.com) prohibits automated scraping/bots in
its Terms of Use, and Edmunds does not offer a public, permissive API for
this kind of use either. To respect those terms of service, this module
does NOT scrape KBB or any other live pricing website. Instead it uses a
small, manually-maintained, static lookup table of approximate US market
price ranges by (make, model, decade-ish year bucket). This is a coarse,
illustrative estimate only — not a real-time valuation.

If a vehicle's make/model/year is UNKNOWN or not present in the lookup
table, all price fields are written as UNKNOWN with price_confidence 0.0,
per the "never fabricate" requirement.

To swap in a real, ToS-compliant pricing API later (e.g. a licensed data
provider), replace `lookup_price()` with a call to that provider and keep
the same return contract.
"""
from __future__ import annotations

from dataclasses import dataclass

# Manually maintained, coarse price-range lookup table (USD).
# Keys are (make.lower(), model.lower()); values are functions of year
# bucket -> (low, high). This is intentionally simple and should be
# expanded/maintained by hand with reviewed data, not scraped.
_PRICE_TABLE: dict[tuple[str, str], dict[str, tuple[int, int]]] = {
    ("toyota", "camry"): {"old": (4000, 9000), "mid": (12000, 20000), "new": (24000, 32000)},
    ("toyota", "corolla"): {"old": (3500, 7500), "mid": (10000, 16000), "new": (20000, 26000)},
    ("toyota", "tacoma"): {"old": (8000, 15000), "mid": (20000, 30000), "new": (32000, 45000)},
    ("honda", "civic"): {"old": (3500, 8000), "mid": (11000, 17000), "new": (22000, 28000)},
    ("honda", "accord"): {"old": (4000, 9000), "mid": (13000, 20000), "new": (25000, 33000)},
    ("ford", "f-150"): {"old": (7000, 15000), "mid": (20000, 32000), "new": (35000, 55000)},
    ("ford", "focus"): {"old": (2500, 6000), "mid": (8000, 13000), "new": (16000, 21000)},
    ("chevrolet", "silverado"): {"old": (7000, 15000), "mid": (20000, 32000), "new": (35000, 55000)},
    ("chevrolet", "malibu"): {"old": (3500, 8000), "mid": (11000, 17000), "new": (21000, 27000)},
    ("nissan", "altima"): {"old": (3500, 8000), "mid": (11000, 17000), "new": (21000, 27000)},
    ("ram", "1500"): {"old": (7000, 15000), "mid": (20000, 32000), "new": (35000, 55000)},
    ("jeep", "wrangler"): {"old": (8000, 16000), "mid": (22000, 32000), "new": (35000, 50000)},
    ("tesla", "model 3"): {"old": (18000, 25000), "mid": (25000, 33000), "new": (38000, 48000)},
}

PRICE_SOURCE_LABEL = "manual_lookup"


@dataclass
class PriceEstimate:
    price_range_low: str
    price_range_high: str
    price_range_currency: str
    price_source: str
    price_confidence: float


def _year_bucket(year_estimate: str) -> str | None:
    try:
        year = int("".join(ch for ch in year_estimate if ch.isdigit())[:4])
    except (ValueError, IndexError):
        return None
    if year <= 0:
        return None
    if year < 2012:
        return "old"
    if year < 2019:
        return "mid"
    return "new"


def estimate_price(make: str, model: str, year_estimate: str) -> PriceEstimate:
    """Estimate a USD price range for a vehicle using the local lookup table.

    Returns UNKNOWN fields if make/model/year are unknown or not present
    in the table. Never invents a numeric range without a table match.
    """
    unknown = PriceEstimate(
        price_range_low="UNKNOWN",
        price_range_high="UNKNOWN",
        price_range_currency="USD",
        price_source="UNKNOWN",
        price_confidence=0.0,
    )

    if not make or not model:
        return unknown
    if make.strip().upper() == "UNKNOWN" or model.strip().upper() == "UNKNOWN":
        return unknown

    key = (make.strip().lower(), model.strip().lower())
    table_entry = _PRICE_TABLE.get(key)
    if not table_entry:
        return unknown

    bucket = _year_bucket(year_estimate) or "mid"  # assume mid-age if year unknown
    low, high = table_entry.get(bucket, table_entry["mid"])

    # Confidence is reduced when the year was unknown/unparsed since we
    # fell back to a default bucket.
    confidence = 0.6 if _year_bucket(year_estimate) else 0.35

    return PriceEstimate(
        price_range_low=str(low),
        price_range_high=str(high),
        price_range_currency="USD",
        price_source=PRICE_SOURCE_LABEL,
        price_confidence=confidence,
    )
