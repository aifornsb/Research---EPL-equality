"""
add_lanes.py  -  Backfill EL/GPL lane columns onto an existing vehicles CSV.

Your CSV already has the detection box (bbox_x1..bbox_y2), and lane assignment is
purely geometric, so lane can be added WITHOUT re-running identification (no API
cost). For each row it computes the wheels-on-road point (bottom-center of the
box) and tests it against a camera's express-lane polygon from lanes.json,
writing lane_type / lane_description / lane_ref_x / lane_ref_y.

Because the CSV stores only the image basename (no camera path), you specify which
camera's polygon to apply with --camera; it is applied to every row. Run it once
per camera if your CSV mixes cameras and you want them handled separately.

Usage
-----
    python add_lanes.py --input data/output/vehicles/vehicles_priced.csv \
        --output data/output/vehicles/vehicles_priced_lanes.csv \
        --lanes lanes.json --camera "IH30 @ Carrier Pkwy"
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import identify_vehicles as iv


def _int(v):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Add EL/GPL lane columns to an existing CSV.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--lanes", default="lanes.json")
    ap.add_argument("--camera", default="IH30 @ Carrier Pkwy",
                    help="Which camera's express_polygon to apply to every row.")
    args = ap.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"input not found: {in_path}", file=sys.stderr)
        return 2
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_lanes.csv")

    cfg = json.loads(Path(args.lanes).read_text()) if Path(args.lanes).exists() else {}
    spec = (cfg.get("cameras", {}) or {}).get(args.camera, {})
    poly_raw = spec.get("express_polygon")
    poly = [(float(x), float(y)) for x, y in poly_raw] if poly_raw and len(poly_raw) >= 3 else None
    buf = float(spec.get("unknown_buffer_px", 0))
    if poly is None:
        print(f"WARNING: no express_polygon for camera '{args.camera}' in {args.lanes}; "
              "all rows will be lane_type=Unknown.", file=sys.stderr)
    if not spec.get("calibrated", False):
        print("NOTE: lanes.json is marked calibrated=false; verify the polygon with "
              "calibrate_lanes.py so the EL/GPL labels are trustworthy.", file=sys.stderr)

    with open(in_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fn = list(reader.fieldnames or [])

    for col in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"):
        if col not in fn:
            print(f"ERROR: input CSV is missing '{col}'; cannot compute lanes.", file=sys.stderr)
            return 3

    # Insert lane columns to match the native schema: type/description after
    # coarse_class, ref points just before notes (append if either anchor absent).
    if "lane_type" not in fn:
        anchor = fn.index("coarse_class") + 1 if "coarse_class" in fn else len(fn)
        fn[anchor:anchor] = ["lane_type", "lane_description"]
    if "lane_ref_x" not in fn:
        if "notes" in fn:
            j = fn.index("notes")
            fn[j:j] = ["lane_ref_x", "lane_ref_y"]
        else:
            fn += ["lane_ref_x", "lane_ref_y"]

    counts = {"Express": 0, "General Purpose": 0, "Unknown": 0}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fn)
        w.writeheader()
        for r in rows:
            bbox = tuple(_int(r.get(c)) for c in ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
            if None in bbox:
                lt, ld, rx, ry = "Unknown", "missing bounding box", None, None
            else:
                lt, ld, rx, ry = iv.classify_lane(bbox, poly, buf)
            r["lane_type"] = lt
            r["lane_description"] = ld
            r["lane_ref_x"] = "" if rx is None else rx
            r["lane_ref_y"] = "" if ry is None else ry
            counts[lt] = counts.get(lt, 0) + 1
            w.writerow(r)

    total = len(rows)
    print(f"wrote {out_path}  ({total} rows)")
    for k in ("Express", "General Purpose", "Unknown"):
        print(f"  {k:16s} {counts.get(k, 0):>6}  ({100 * counts.get(k, 0) / max(1, total):.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
