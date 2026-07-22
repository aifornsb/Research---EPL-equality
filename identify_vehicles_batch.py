"""
identify_vehicles_batch.py  -  Step 1 via the Anthropic Message Batches API.

Same detect -> crop-and-upscale -> identify -> confidence-gate pipeline as
``identify_vehicles.py``, but the identification calls go through the Batch API:
asynchronous, and 50% cheaper per token. Default model is Haiku 4.5, the
cheapest tier, which combined with the batch discount makes a full-dataset run
inexpensive. It reuses the detection, prompt, gating, and CSV logic from
``identify_vehicles.py`` so behavior stays identical to the synchronous script.

How it works
------------
1. **submit**: run YOLO over every not-yet-done image, crop+upscale each vehicle,
   and submit the crops as one or more batches (chunked so no batch exceeds the
   API's size/count limits, and so each image's vehicles stay within a single
   batch). A manifest (batch ids + per-request metadata) is written to the output
   folder so the run is fully resumable.
2. **collect**: poll each submitted batch; when it has ended, download the
   results, apply the same confidence gating, and append rows to the CSV. Images
   whose batch has been collected are marked done.
3. **run** (default): submit, then poll up to ``--max-wait`` seconds and collect.
   If the wait is exceeded (large jobs can take a while), the manifest is left in
   place; re-run with ``--mode collect`` (or just ``run`` again) to finish. No
   work or spend is repeated.

Usage
-----
    export ANTHROPIC_API_KEY=...
    pip install ultralytics anthropic opencv-python-headless pillow

    python identify_vehicles_batch.py --input "data/images/IH30 @ Carrier Pkwy" \
        --recursive --output data/output/traffic          # submit + poll + collect
    python identify_vehicles_batch.py --input <dir> --recursive \
        --output data/output/traffic --mode collect        # finish a pending run
    python identify_vehicles_batch.py --input <dir> --recursive --estimate
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import identify_vehicles as iv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("identify_vehicles_batch")

# Stay comfortably under the API limits (256 MB / 100,000 requests per batch).
MAX_BATCH_BYTES = 180 * 1024 * 1024
MAX_BATCH_REQUESTS = 90_000
MANIFEST = "batches.json"


def _client():
    import os

    import anthropic

    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    return anthropic.Anthropic(api_key=key)


def _request_for(cid: str, b64: str, model: str) -> dict:
    """One Batch API request in the {custom_id, params} shape."""
    return {
        "custom_id": cid,
        "params": {
            "model": model,
            "max_tokens": 600,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64",
                         "media_type": "image/jpeg", "data": b64}},
                        {"type": "text", "text": iv.IDENTIFY_PROMPT},
                    ],
                }
            ],
        },
    }


def _load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {"model": None, "batches": [], "meta": {}}


def _save_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))


# --------------------------------------------------------------------------- #
# submit
# --------------------------------------------------------------------------- #
def submit(inputs, out_dir: Path, model: str, detector, crop_long_edge: int) -> dict:
    csv_path = out_dir / "vehicles.csv"
    manifest_path = out_dir / MANIFEST
    manifest = _load_manifest(manifest_path)
    manifest["model"] = model

    done_images = iv.load_done(csv_path)
    submitted_images = {m["image"] for m in manifest["meta"].values()}
    pending = [s for s in inputs
               if Path(s).name not in done_images and Path(s).name not in submitted_images]
    logger.info("submit: %d image(s) pending detection", len(pending))
    if not pending:
        return manifest

    client = _client()
    cid_counter = len(manifest["meta"])
    chunk: list[dict] = []
    chunk_bytes = 0

    def flush_chunk():
        nonlocal chunk, chunk_bytes
        if not chunk:
            return
        batch = client.messages.batches.create(requests=chunk)
        manifest["batches"].append({"id": batch.id, "collected": False,
                                    "custom_ids": [r["custom_id"] for r in chunk]})
        logger.info("submitted batch %s with %d request(s)", batch.id, len(chunk))
        _save_manifest(manifest_path, manifest)
        chunk, chunk_bytes = [], 0

    for src in pending:
        try:
            image = iv.load_image_bgr(src)
        except Exception as exc:  # noqa: BLE001
            logger.error("skip unreadable %s: %s", src, exc)
            continue
        dets = detector.detect(image)
        name = Path(src).name
        # Build this image's requests together so it never splits across batches.
        image_reqs, image_meta = [], {}
        for i, (x1, y1, x2, y2, dconf, coarse) in enumerate(dets, start=1):
            crop = iv.crop_and_upscale(image, (x1, y1, x2, y2), crop_long_edge)
            b64 = iv.encode_jpeg(crop)
            cid = f"r{cid_counter}"
            cid_counter += 1
            image_reqs.append(_request_for(cid, b64, model))
            image_meta[cid] = {"image": name, "path": src, "idx": i, "x1": x1, "y1": y1,
                               "x2": x2, "y2": y2, "cw": x2 - x1, "ch": y2 - y1,
                               "dconf": round(dconf, 4), "coarse": coarse}
        est = sum(len(r["params"]["messages"][0]["content"][0]["source"]["data"])
                  for r in image_reqs)
        if chunk and (chunk_bytes + est > MAX_BATCH_BYTES
                      or len(chunk) + len(image_reqs) > MAX_BATCH_REQUESTS):
            flush_chunk()
        chunk.extend(image_reqs)
        chunk_bytes += est
        manifest["meta"].update(image_meta)
        # persist meta as we go (so a crash mid-detection is recoverable)
        if name and len(image_reqs) == 0:
            # image with no detections: record a zero-vehicle marker so it is "done"
            manifest["meta"][f"empty::{name}"] = {"image": name, "idx": 0, "empty": True}
    flush_chunk()
    _save_manifest(manifest_path, manifest)
    return manifest


# --------------------------------------------------------------------------- #
# collect
# --------------------------------------------------------------------------- #
def _vehicle_from_meta(m: dict) -> iv.Vehicle:
    return iv.Vehicle(m["idx"], m["x1"], m["y1"], m["x2"], m["y2"], m["cw"], m["ch"],
                      m["dconf"], m["coarse"])


def collect(out_dir: Path, thr: iv.Thresholds, max_wait: int, lane_cfg=None, poll_every: int = 30) -> int:
    import csv

    lane_cfg = lane_cfg or {}
    csv_path = out_dir / "vehicles.csv"
    manifest_path = out_dir / MANIFEST
    manifest = _load_manifest(manifest_path)
    if not manifest["batches"]:
        logger.info("collect: nothing to collect.")
        return 0
    client = _client()
    meta = manifest["meta"]

    new = not csv_path.exists()
    f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=iv.CSV_COLUMNS)
    if new:
        writer.writeheader()
        f.flush()

    # write zero-vehicle images once (so empty frames count as processed)
    done_images = iv.load_done(csv_path)

    written_rows = 0
    deadline = time.time() + max_wait
    try:
        for b in manifest["batches"]:
            if b["collected"]:
                continue
            while True:
                status = client.messages.batches.retrieve(b["id"]).processing_status
                if status == "ended":
                    break
                if time.time() > deadline:
                    logger.info("max-wait reached; batch %s still %s. Re-run to collect.",
                                b["id"], status)
                    return written_rows
                logger.info("batch %s: %s; waiting %ds", b["id"], status, poll_every)
                time.sleep(poll_every)

            # collect image rows grouped so we can mark whole images done
            by_image: dict[str, list[iv.Vehicle]] = {}
            for entry in client.messages.batches.results(b["id"]):
                cid = entry.custom_id
                m = meta.get(cid)
                if not m:
                    continue
                veh = _vehicle_from_meta(m)
                # geometric lane assignment from the vehicle's position in the frame
                src_path = m.get("path", m.get("image", ""))
                poly, buf = iv.camera_polygon(src_path, lane_cfg)
                lt, ld, rx, ry = iv.classify_lane((m["x1"], m["y1"], m["x2"], m["y2"]), poly, buf)
                veh.lane_type, veh.lane_description = lt, ld
                veh.lane_ref_x, veh.lane_ref_y = rx, ry
                res = entry.result
                if getattr(res, "type", None) == "succeeded":
                    msg = res.message
                    if getattr(msg, "stop_reason", None) == "refusal":
                        veh.notes = "model refusal"
                    else:
                        text = "".join(bl.text for bl in msg.content
                                       if getattr(bl, "type", None) == "text")
                        data = iv._extract_json(text)
                        if data is not None:
                            iv.apply_identification(veh, data, thr)
                        else:
                            veh.notes = "invalid JSON"
                else:
                    veh.notes = f"batch result: {getattr(res, 'type', 'unknown')}"
                by_image.setdefault(m["image"], []).append(veh)

            for name, vehs in by_image.items():
                if name in done_images:
                    continue
                for veh in sorted(vehs, key=lambda v: v.vehicle_index):
                    writer.writerow(iv.row_of(name, veh))
                    written_rows += 1
            f.flush()
            b["collected"] = True
            _save_manifest(manifest_path, manifest)
            logger.info("collected batch %s (%d rows so far)", b["id"], written_rows)
    finally:
        f.close()
    logger.info("collect: wrote %d row(s) to %s", written_rows, csv_path)
    return written_rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Step 1 via Anthropic Batch API (Haiku 4.5).")
    ap.add_argument("--input", required=True, nargs="+", help="One or more files, directories, or URLs.")
    ap.add_argument("--output", default="identify_output")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--mode", choices=["run", "submit", "collect"], default="run")
    ap.add_argument("--detector", default="yolov8x.pt")
    ap.add_argument("--image-size", type=int, default=1536)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--det-iou", type=float, default=0.60)
    ap.add_argument("--crop-long-edge", type=int, default=768)
    ap.add_argument("--min-make-conf", type=float, default=0.70)
    ap.add_argument("--min-model-conf", type=float, default=0.70)
    ap.add_argument("--min-year-conf", type=float, default=0.60)
    ap.add_argument("--min-direction-conf", type=float, default=0.55)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-wait", type=int, default=18000, help="Seconds to poll before exiting.")
    ap.add_argument("--lanes", default="lanes.json",
                    help="Lane-geometry config (per-camera express-lane polygons).")
    ap.add_argument("--estimate", action="store_true")
    ap.add_argument("--avg-vehicles", type=float, default=6.5)
    args = ap.parse_args(argv)

    try:
        inputs = iv.gather_inputs(args.input, args.recursive)
    except FileNotFoundError as exc:
        logger.error("Input not found: %s", exc)
        return 2
    if args.limit:
        inputs = inputs[: args.limit]

    if args.estimate:
        # Batch API is 50% off; reuse the base estimator then halve.
        calls = len(inputs) * args.avg_vehicles
        tin = calls * (470 + 650) * 1.10
        tout = calls * 200 * 1.10
        pi, po = iv.PRICES.get(args.model, iv.PRICES["claude-haiku-4-5"])
        full = tin / 1e6 * pi + tout / 1e6 * po
        print(f"\nImages={len(inputs):,}  est. calls={calls:,.0f}")
        print(f"est. cost on {args.model}: standard ${full:,.0f}  |  batch (-50%) ${full/2:,.0f}")
        return 0

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = iv.Thresholds(args.min_make_conf, args.min_model_conf, args.min_year_conf,
                        args.min_direction_conf)

    if args.mode in ("run", "submit"):
        detector = iv.Detector(args.detector, args.image_size, args.det_conf, args.det_iou)
        submit(inputs, out_dir, args.model, detector, args.crop_long_edge)
    if args.mode in ("run", "collect"):
        lane_cfg = iv.load_lane_config(args.lanes)
        if lane_cfg.get("cameras"):
            logger.info("lane geometry loaded for %d camera(s)", len(lane_cfg["cameras"]))
        collect(out_dir, thr, args.max_wait, lane_cfg=lane_cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
