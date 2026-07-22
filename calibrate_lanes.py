"""
calibrate_lanes.py  -  Set and verify the express-lane region for a camera.

Lane assignment is geometric: a vehicle is Express if its wheels-on-road point
(bottom-center of its detection box) falls inside the camera's fixed express-lane
polygon, else General Purpose. Because the camera view is fixed, you define that
polygon once. This tool renders it on a real frame so you can confirm/tune it.

Two things it draws on a sample image, saved as a PNG you can open:
  * the current express_polygon from lanes.json (semi-transparent), and
  * every detected vehicle, colored by the lane it would be assigned
    (blue = Express, orange = General Purpose), with its wheels point marked.

Workflow
--------
1. Pick a clear daytime frame from the camera.
2. Run:  python calibrate_lanes.py --image <frame.jpg> --lanes lanes.json --out preview.png
3. Open preview.png. Adjust the polygon coordinates in lanes.json until the shaded
   region covers exactly the two barrier-separated center (Express) lanes.
4. Re-run until the box colors match reality. Commit lanes.json.

Use --no-detect to just see the polygon over the frame (no YOLO needed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import identify_vehicles as iv


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Draw/verify a camera's express-lane region.")
    ap.add_argument("--image", required=True, help="Sample frame from the camera.")
    ap.add_argument("--lanes", default="lanes.json")
    ap.add_argument("--out", default="lane_preview.png")
    ap.add_argument("--detector", default="yolov8x.pt")
    ap.add_argument("--image-size", type=int, default=1536)
    ap.add_argument("--no-detect", action="store_true", help="Only draw the polygon.")
    args = ap.parse_args(argv)

    import cv2
    import numpy as np

    image = iv.load_image_bgr(args.image)
    overlay = image.copy()
    lane_cfg = iv.load_lane_config(args.lanes)
    poly, buf = iv.camera_polygon(args.image, lane_cfg)

    if poly:
        pts = np.array([[int(x), int(y)] for x, y in poly], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (200, 60, 20))          # filled EL region (BGR)
        image = cv2.addWeighted(overlay, 0.30, image, 0.70, 0)
        cv2.polylines(image, [pts], True, (255, 90, 40), 2)
        cv2.putText(image, "EXPRESS region", (pts[:, 0].min(), max(20, pts[:, 1].min() - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 90, 40), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "No express_polygon configured for this camera", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    n_el = n_gp = 0
    if not args.no_detect:
        det = iv.Detector(args.detector, args.image_size, 0.25, 0.60)
        for (x1, y1, x2, y2, dconf, coarse) in det.detect(image):
            lt, _, rx, ry = iv.classify_lane((x1, y1, x2, y2), poly, buf)
            color = (200, 60, 20) if lt == "Express" else (0, 140, 255) if lt == "General Purpose" \
                else (120, 120, 120)
            if lt == "Express":
                n_el += 1
            elif lt == "General Purpose":
                n_gp += 1
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.circle(image, (rx, ry), 4, color, -1)  # wheels-on-road reference point

    cv2.putText(image, f"Express={n_el}  GeneralPurpose={n_gp}", (20, image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, image)
    print(f"wrote {args.out}  (Express={n_el}, General Purpose={n_gp})")
    if not poly:
        print("Tip: add an 'express_polygon' for this camera to", args.lanes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
