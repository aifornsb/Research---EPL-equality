"""
identify_vehicles.py  -  Step 1: per-vehicle identification for road-traffic images.

Finds every vehicle in a traffic-camera image and identifies make, model,
model-year range, and heading, with a per-field confidence. Precision first: it
reports ``Unknown`` rather than guess, and accepts a field only above a
confidence floor.

Accuracy strategy (why this is set up the way it is)
----------------------------------------------------
The limiting factor on traffic frames is pixels-on-the-vehicle, not the model.
So: (1) detect with YOLO at high resolution to localize even small/distant
vehicles, (2) crop each detection at native resolution then UPSCALE the crop so
the recognizer sees maximum detail, (3) identify each crop with a capable
vision-language model returning strict JSON with confidences, (4) gate every
field on confidence. The default model is Claude Fable 5, the most capable
model, for highest accuracy.

Built for a GitHub run
----------------------
* ``--workers`` runs the (network-bound) identification calls concurrently so a
  full camera folder finishes inside a CI time limit.
* Output CSV is append-only and the run is RESUMABLE: images already in the CSV
  are skipped, so a timed-out or restarted job continues without re-paying.
* ``--estimate`` prints a projected token/cost figure (no API calls) so you can
  confirm a run fits your credit before starting.

Usage
-----
    export ANTHROPIC_API_KEY=...
    pip install ultralytics anthropic opencv-python-headless pillow tenacity
    # cost projection only:
    python identify_vehicles.py --input "data/images/IH30 @ Carrier Pkwy" --recursive --estimate
    # real run (highest accuracy, Fable 5):
    python identify_vehicles.py --input "data/images/IH30 @ Carrier Pkwy" --recursive \
        --output data/output --workers 8
    # precision cross-check within budget (Opus + verify, ~same cost as Fable single-pass):
    python identify_vehicles.py --input <dir> --recursive --model claude-opus-4-8 --verify
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import logging
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("identify_vehicles")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
UNKNOWN = "Unknown"
VEHICLE_COCO_IDS = {2, 3, 5, 7}

# Timestamp embedded in TxDOT-style filenames, e.g. "2026-07-11_08-17-27.jpg"
# (also tolerates a camera-name prefix and "HH:MM:SS").
_TS_RE = re.compile(r"(\d{4}-\d{2}-\d{2})[_ T](\d{2})[-:](\d{2})[-:](\d{2})")


def parse_timestamp(name: str) -> tuple[str, str]:
    """Return (snapshot_date 'YYYY-MM-DD', snapshot_time 'HH:MM:SS') from a
    filename, or ('', '') if no timestamp is present."""
    m = _TS_RE.search(name)
    if not m:
        return "", ""
    d, hh, mm, ss = m.groups()
    try:
        datetime(*(int(x) for x in d.split("-")), int(hh), int(mm), int(ss))
    except ValueError:
        return "", ""
    return d, f"{hh}:{mm}:{ss}"


# --------------------------------------------------------------------------- #
# Lane assignment (geometric): a vehicle is Express if its wheels-on-road point
# falls inside the camera's fixed express-lane region, else General Purpose.
# The region is defined per camera in a lanes config (see lanes.json / calibrate).
# --------------------------------------------------------------------------- #
def load_lane_config(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("lanes config not found at %s; lane_type will be Unknown.", p)
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not parse lanes config (%s); lane_type will be Unknown.", exc)
        return {}


def camera_polygon(source_path: str, lane_cfg: dict):
    """Return (polygon, buffer_px) for the camera this image belongs to, matched
    by substring of a configured camera name against the file path. None if none."""
    cams = (lane_cfg or {}).get("cameras", {})
    for name, spec in cams.items():
        if name and name in source_path:
            poly = spec.get("express_polygon")
            if poly and len(poly) >= 3:
                return [(float(x), float(y)) for x, y in poly], float(spec.get("unknown_buffer_px", 0))
    return None, 0.0


def _point_in_polygon(x: float, y: float, poly: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test (no external dependency)."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_cross:
                inside = not inside
        j = i
    return inside


def classify_lane(bbox: tuple[int, int, int, int], poly, buffer_px: float = 0.0):
    """Return (lane_type, lane_description, ref_x, ref_y).

    Reference point is the bottom-center of the box (where the vehicle meets the
    road), which is the most reliable indicator of lane occupancy. If no polygon
    is configured for the camera, lane_type is Unknown.
    """
    x1, y1, x2, y2 = bbox
    rx, ry = int((x1 + x2) / 2), int(y2)
    if not poly:
        return "Unknown", "no lane geometry configured for this camera", rx, ry
    inside = _point_in_polygon(rx, ry, poly)
    if inside:
        return "Express", "barrier-separated center express lane", rx, ry
    return "General Purpose", "general-purpose lane", rx, ry
COCO_COARSE = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DIRECTIONS = {"toward camera", "away from camera", "left-to-right", "right-to-left", "unknown"}

PRICES = {
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}

IDENTIFY_PROMPT = """You are a forensic vehicle-recognition expert examining a single cropped image of ONE vehicle from a roadway traffic camera.

Identify the vehicle as precisely as the image supports, and no further. Accuracy matters more than completeness: if a field is not clearly supported by what is visible, return "Unknown" for it. Never guess a make, model, or year you cannot actually see evidence for.

Report:
- make: manufacturer (e.g. "Toyota"), or "Unknown"
- model: specific model (e.g. "Camry"), or "Unknown"
- year_low / year_high: the tightest model-year range you can justify from the visible generation/styling; same value twice if confident in a single year; null if unknown
- body_type: sedan | suv | pickup_truck | van | hatchback | coupe | bus | motorcycle | medium_truck | heavy_truck | unknown
- color: dominant exterior color, or "Unknown"
- direction_facing: orientation relative to the camera, exactly one of:
  "toward camera", "away from camera", "left-to-right", "right-to-left", "unknown"
- make_confidence, model_confidence, year_confidence, direction_confidence, overall_confidence: numbers in [0,1]
- top_models: up to 3 {"make_model": "Make Model", "confidence": 0.0}, most likely first

Respond with STRICT JSON ONLY, no prose, no markdown:
{"make":"...","model":"...","year_low":null,"year_high":null,"body_type":"...","color":"...","direction_facing":"...","make_confidence":0.0,"model_confidence":0.0,"year_confidence":0.0,"direction_confidence":0.0,"overall_confidence":0.0,"top_models":[]}"""


@dataclass
class Vehicle:
    vehicle_index: int
    bbox_x1: int
    bbox_y1: int
    bbox_x2: int
    bbox_y2: int
    crop_width: int
    crop_height: int
    detection_confidence: float
    coarse_class: str
    make: str = UNKNOWN
    model: str = UNKNOWN
    year_low: int | None = None
    year_high: int | None = None
    body_type: str = UNKNOWN
    color: str = UNKNOWN
    direction_facing: str = "unknown"
    lane_type: str = "Unknown"
    lane_description: str = ""
    lane_ref_x: int | None = None
    lane_ref_y: int | None = None
    make_confidence: float = 0.0
    model_confidence: float = 0.0
    year_confidence: float = 0.0
    direction_confidence: float = 0.0
    overall_confidence: float = 0.0
    top_models: list[dict] = field(default_factory=list)
    identification_status: str = "unidentified"
    notes: str = ""


@dataclass
class Thresholds:
    make: float = 0.70
    model: float = 0.70
    year: float = 0.60
    direction: float = 0.55


def load_image_bgr(source: str):
    import cv2
    import numpy as np

    if source.startswith(("http://", "https://")):
        import httpx

        with httpx.Client(timeout=30.0, follow_redirects=True) as c:
            r = c.get(source)
            r.raise_for_status()
            buf = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    else:
        img = cv2.imread(source, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {source}")
    return img


class Detector:
    def __init__(self, weights: str, image_size: int, conf: float, iou: float):
        self.weights, self.image_size, self.conf, self.iou = weights, image_size, conf, iou
        self._model = None

    def load(self):
        if self._model is None:
            from ultralytics import YOLO

            logger.info("Loading detector %s", self.weights)
            self._model = YOLO(self.weights)

    def detect(self, image_bgr):
        self.load()
        h, w = image_bgr.shape[:2]
        results = self._model.predict(
            source=image_bgr[:, :, ::-1], imgsz=self.image_size,
            conf=self.conf, iou=self.iou, verbose=False,
        )
        dets = []
        for res in results:
            for b in getattr(res, "boxes", []) or []:
                cid = int(b.cls.item())
                if cid not in VEHICLE_COCO_IDS:
                    continue
                x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(w, int(x2)), min(h, int(y2))
                if x2 > x1 and y2 > y1:
                    dets.append((x1, y1, x2, y2, float(b.conf.item()), COCO_COARSE[cid]))
        return dets


def crop_and_upscale(image_bgr, box, target_long_edge: int):
    import cv2

    x1, y1, x2, y2 = box
    crop = image_bgr[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    if 0 < max(h, w) < target_long_edge:
        s = target_long_edge / max(h, w)
        crop = cv2.resize(crop, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC)
    return crop


def encode_jpeg(image_bgr) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image_bgr[:, :, ::-1]).save(buf, format="JPEG", quality=95)
    return base64.standard_b64encode(buf.getvalue()).decode()


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


class Identifier:
    def __init__(self, model: str):
        self.model = model
        self._client = None
        self._lock = threading.Lock()

    def _client_obj(self):
        with self._lock:
            if self._client is None:
                import anthropic

                key = os.environ.get("ANTHROPIC_API_KEY")
                if not key:
                    raise RuntimeError("ANTHROPIC_API_KEY is not set.")
                self._client = anthropic.Anthropic(api_key=key)
        return self._client

    def identify(self, crop_bgr) -> dict | None:
        from tenacity import retry, stop_after_attempt, wait_exponential

        client = self._client_obj()
        b64 = encode_jpeg(crop_bgr)

        @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=45), reraise=True)
        def _send(extra: str):
            return client.messages.create(
                model=self.model, max_tokens=600,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64",
                     "media_type": "image/jpeg", "data": b64}},
                    {"type": "text", "text": IDENTIFY_PROMPT + extra},
                ]}],
            )

        for extra in ("", "\nReturn STRICT valid JSON only."):
            resp = _send(extra)
            if getattr(resp, "stop_reason", None) == "refusal":
                return None
            text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            data = _extract_json(text)
            if data is not None:
                return data
        return None


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def apply_identification(veh: Vehicle, data: dict, thr: Thresholds) -> None:
    mk_c = float(data.get("make_confidence", 0) or 0)
    md_c = float(data.get("model_confidence", 0) or 0)
    yr_c = float(data.get("year_confidence", 0) or 0)
    dir_c = float(data.get("direction_confidence", 0) or 0)
    veh.make_confidence, veh.model_confidence = round(mk_c, 3), round(md_c, 3)
    veh.year_confidence, veh.direction_confidence = round(yr_c, 3), round(dir_c, 3)
    veh.overall_confidence = round(float(data.get("overall_confidence", 0) or 0), 3)
    veh.body_type = str(data.get("body_type", UNKNOWN) or UNKNOWN)
    veh.color = str(data.get("color", UNKNOWN) or UNKNOWN)
    veh.top_models = data.get("top_models", []) or []

    veh.make = str(data.get("make", UNKNOWN) or UNKNOWN) if mk_c >= thr.make else UNKNOWN
    veh.model = (str(data.get("model", UNKNOWN) or UNKNOWN)
                 if (md_c >= thr.model and veh.make != UNKNOWN) else UNKNOWN)
    if yr_c >= thr.year and veh.model != UNKNOWN:
        veh.year_low, veh.year_high = _to_int(data.get("year_low")), _to_int(data.get("year_high"))
    d = str(data.get("direction_facing", "unknown") or "unknown").lower()
    veh.direction_facing = d if (d in DIRECTIONS and dir_c >= thr.direction) else "unknown"

    veh.identification_status = ("identified" if veh.model != UNKNOWN
                                 else "make_only" if veh.make != UNKNOWN else "class_only")


def reconcile(primary: dict, second: dict) -> dict:
    out = dict(primary)
    for f in ("make", "model", "body_type", "direction_facing"):
        if str(primary.get(f, "")).strip().lower() != str(second.get(f, "")).strip().lower():
            out[f] = UNKNOWN if f in ("make", "model", "body_type") else "unknown"
            ck = {"make": "make_confidence", "model": "model_confidence",
                  "direction_facing": "direction_confidence"}.get(f)
            if ck:
                out[ck] = 0.0
    if _to_int(primary.get("year_low")) != _to_int(second.get("year_low")):
        out["year_confidence"] = min(float(primary.get("year_confidence", 0) or 0),
                                     float(second.get("year_confidence", 0) or 0))
    return out


def process_image(source, detector, identifier, thr, crop_long_edge, verify, workers,
                  save_crops_dir, lane_cfg=None):
    image = load_image_bgr(source)
    dets = detector.detect(image)
    logger.info("%s: %d vehicle(s)", Path(source).name, len(dets))
    poly, buffer_px = camera_polygon(source, lane_cfg or {})

    def build(i_det):
        i, (x1, y1, x2, y2, dconf, coarse) = i_det
        crop = crop_and_upscale(image, (x1, y1, x2, y2), crop_long_edge)
        veh = Vehicle(i, x1, y1, x2, y2, x2 - x1, y2 - y1, round(dconf, 4), coarse)
        lt, ld, rx, ry = classify_lane((x1, y1, x2, y2), poly, buffer_px)
        veh.lane_type, veh.lane_description = lt, ld
        veh.lane_ref_x, veh.lane_ref_y = rx, ry
        try:
            data = identifier.identify(crop)
            if data is not None and verify:
                second = identifier.identify(crop)
                if second is not None:
                    data, veh.notes = reconcile(data, second), "two-pass verified"
            if data is not None:
                apply_identification(veh, data, thr)
            else:
                veh.notes = "no valid identification"
        except Exception as exc:  # noqa: BLE001
            veh.notes = f"error: {exc}"
        if save_crops_dir is not None:
            import cv2

            save_crops_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(save_crops_dir / f"{Path(source).stem}_veh{i:03d}.jpg"), crop)
        return veh

    indexed = list(enumerate(dets, start=1))
    if workers > 1 and len(indexed) > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            vehicles = list(ex.map(build, indexed))
    else:
        vehicles = [build(x) for x in indexed]
    return sorted(vehicles, key=lambda v: v.vehicle_index)


CSV_COLUMNS = [
    "image_filename", "snapshot_date", "snapshot_time", "vehicle_index", "coarse_class",
    "lane_type", "lane_description",
    "make", "model", "year_low",
    "year_high", "direction_facing", "body_type", "color", "identification_status",
    "make_confidence", "model_confidence", "year_confidence", "direction_confidence",
    "detection_confidence", "overall_confidence", "bbox_x1", "bbox_y1", "bbox_x2",
    "bbox_y2", "crop_width", "crop_height", "lane_ref_x", "lane_ref_y", "notes",
]


def row_of(image_name: str, v: Vehicle) -> dict:
    snapshot_date, snapshot_time = parse_timestamp(image_name)
    return {
        "image_filename": image_name,
        "snapshot_date": snapshot_date, "snapshot_time": snapshot_time,
        "vehicle_index": v.vehicle_index,
        "coarse_class": v.coarse_class,
        "lane_type": v.lane_type, "lane_description": v.lane_description,
        "make": v.make, "model": v.model,
        "year_low": "" if v.year_low is None else v.year_low,
        "year_high": "" if v.year_high is None else v.year_high,
        "direction_facing": v.direction_facing, "body_type": v.body_type, "color": v.color,
        "identification_status": v.identification_status,
        "make_confidence": v.make_confidence, "model_confidence": v.model_confidence,
        "year_confidence": v.year_confidence, "direction_confidence": v.direction_confidence,
        "detection_confidence": v.detection_confidence, "overall_confidence": v.overall_confidence,
        "bbox_x1": v.bbox_x1, "bbox_y1": v.bbox_y1, "bbox_x2": v.bbox_x2, "bbox_y2": v.bbox_y2,
        "crop_width": v.crop_width, "crop_height": v.crop_height,
        "lane_ref_x": "" if v.lane_ref_x is None else v.lane_ref_x,
        "lane_ref_y": "" if v.lane_ref_y is None else v.lane_ref_y,
        "notes": v.notes,
    }


def gather_inputs(specs, recursive: bool) -> list[str]:
    """Collect image paths from one or more files/dirs/URLs, then de-duplicate by
    basename. Each timestamp filename identifies one unique capture; when the same
    basename appears in more than one place (e.g. a failed_review copy of a camera
    image), the non-failed_review path is kept so the recorded location is the
    original. Returns a sorted list of unique paths."""
    if isinstance(specs, str):
        specs = [specs]
    raw: list[str] = []
    for spec in specs:
        if spec.startswith(("http://", "https://")):
            raw.append(spec)
            continue
        p = Path(spec)
        if p.is_file():
            raw.append(str(p))
        elif p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            raw += [str(x) for x in it if x.suffix.lower() in IMAGE_EXTS]
        else:
            raise FileNotFoundError(spec)

    chosen: dict[str, str] = {}
    dupes = 0
    for pth in raw:
        name = Path(pth).name
        is_failed = "failed_review" in Path(pth).parts
        if name not in chosen:
            chosen[name] = pth
        else:
            dupes += 1
            # prefer a non-failed_review copy if the stored one is from failed_review
            if not is_failed and "failed_review" in Path(chosen[name]).parts:
                chosen[name] = pth
    if dupes:
        logger.info("de-duplicated %d image(s) by basename (e.g. failed_review copies)", dupes)
    return sorted(chosen.values())


def load_done(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    done = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("image_filename"):
                done.add(row["image_filename"])
    return done


def estimate(n_images: int, avg_vehicles: float, model: str, verify: bool) -> None:
    calls = n_images * avg_vehicles * (2 if verify else 1)
    tin = calls * (470 + 650) * 1.10
    tout = calls * 200 * 1.10
    print(f"\nImages={n_images:,}  avg vehicles/image={avg_vehicles}  "
          f"est. calls={calls:,.0f}{'  (x2 verify)' if verify else ''}")
    print(f"est. input={tin/1e6:.1f}M tok, output={tout/1e6:.1f}M tok")
    pi, po = PRICES.get(model, PRICES["claude-fable-5"])
    print(f"est. cost on {model}: ${tin/1e6*pi + tout/1e6*po:,.0f}")
    print("(rough; actual depends on real vehicle counts and crop sizes)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Step 1: identify road-traffic vehicles.")
    ap.add_argument("--input", required=True, nargs="+", help="One or more files, directories, or URLs.")
    ap.add_argument("--output", default="identify_output")
    ap.add_argument("--model", default="claude-fable-5",
                    help="Vision model. Default Fable 5 (highest accuracy).")
    ap.add_argument("--detector", default="yolov8x.pt")
    ap.add_argument("--image-size", type=int, default=1536)
    ap.add_argument("--det-conf", type=float, default=0.25)
    ap.add_argument("--det-iou", type=float, default=0.60)
    ap.add_argument("--crop-long-edge", type=int, default=768)
    ap.add_argument("--min-make-conf", type=float, default=0.70)
    ap.add_argument("--min-model-conf", type=float, default=0.70)
    ap.add_argument("--min-year-conf", type=float, default=0.60)
    ap.add_argument("--min-direction-conf", type=float, default=0.55)
    ap.add_argument("--verify", action="store_true",
                    help="Two passes per vehicle; keep only agreeing fields (doubles cost).")
    ap.add_argument("--workers", type=int, default=8,
                    help="Concurrent identification calls (fit CI time limits / rate limits).")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-crops", action="store_true")
    ap.add_argument("--no-json", action="store_true")
    ap.add_argument("--lanes", default="lanes.json",
                    help="Lane-geometry config (per-camera express-lane polygons). "
                         "If missing, lane_type is Unknown.")
    ap.add_argument("--estimate", action="store_true", help="Print cost projection and exit.")
    ap.add_argument("--avg-vehicles", type=float, default=6.5, help="For --estimate only.")
    args = ap.parse_args(argv)

    try:
        inputs = gather_inputs(args.input, args.recursive)
    except FileNotFoundError as exc:
        logger.error("Input not found: %s", exc)
        return 2
    if args.limit:
        inputs = inputs[: args.limit]

    if args.estimate:
        estimate(len(inputs), args.avg_vehicles, args.model, args.verify)
        return 0

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "vehicles.csv"
    done = load_done(csv_path)
    todo = [s for s in inputs if Path(s).name not in done]
    logger.info("Model=%s workers=%d | %d image(s), %d already done, %d to process",
                args.model, args.workers, len(inputs), len(done), len(todo))

    thr = Thresholds(args.min_make_conf, args.min_model_conf, args.min_year_conf,
                     args.min_direction_conf)
    detector = Detector(args.detector, args.image_size, args.det_conf, args.det_iou)
    identifier = Identifier(args.model)
    lane_cfg = load_lane_config(args.lanes)
    if lane_cfg.get("cameras"):
        logger.info("lane geometry loaded for %d camera(s)", len(lane_cfg["cameras"]))

    new_file = not csv_path.exists()
    f = open(csv_path, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
    if new_file:
        writer.writeheader()
        f.flush()

    processed = ident = 0
    try:
        for src in todo:
            try:
                vehicles = process_image(src, detector, identifier, thr, args.crop_long_edge,
                                         args.verify, args.workers,
                                         (out_dir / "crops") if args.save_crops else None,
                                         lane_cfg=lane_cfg)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed on %s: %s", src, exc)
                continue
            name = Path(src).name
            for v in vehicles:
                writer.writerow(row_of(name, v))
            f.flush()
            if not args.no_json:
                jdir = out_dir / "json"
                jdir.mkdir(parents=True, exist_ok=True)
                (jdir / f"{Path(name).stem}.json").write_text(
                    json.dumps([asdict(v) for v in vehicles], indent=2))
            processed += 1
            ident += sum(1 for v in vehicles if v.model != UNKNOWN)
            if processed % 25 == 0:
                logger.info("progress: %d/%d images, %d models identified so far",
                            processed, len(todo), ident)
    finally:
        f.close()
    logger.info("Done. CSV=%s | processed=%d | model-identified=%d", csv_path, processed, ident)
    return 0


if __name__ == "__main__":
    sys.exit(main())
