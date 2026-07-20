"""
Compare first-pass (data/traffic_observations.csv) against second-pass
(data/validation_pass2.csv) and produce test-retest agreement statistics.

Because vehicle-level matching between two independent reads of the same frame
is ambiguous, agreement is computed at three levels:

  1. Frame level  : total vehicle count, Express-lane vehicle count
  2. Multiset level: per-frame overlap of (lane_type,) and (make, model) multisets
  3. Aggregate level: Express share, make-identification rate, body-type mix

Usage:
    python -m src.validation_report
Writes data/validation_report.md and prints the summary.
"""

from collections import Counter
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
P1 = REPO_ROOT / "data" / "traffic_observations.csv"
P2 = REPO_ROOT / "data" / "validation_pass2.csv"
OUT = REPO_ROOT / "data" / "validation_report.md"


def multiset_overlap(a: Counter, b: Counter) -> float:
    """Jaccard-style overlap of two multisets (1.0 = identical)."""
    inter = sum((a & b).values())
    union = sum((a | b).values())
    return inter / union if union else 1.0


def main() -> None:
    p1 = pd.read_csv(P1)
    p2 = pd.read_csv(P2)
    p2 = p2[~p2["sequence_number"].isin(["MISSING_IMAGE"])]

    frames = sorted(set(p1["image_path"]) & set(p2["image_path"]))
    print(f"frames in both passes: {len(frames)}")

    rows = []
    for f in frames:
        a = p1[p1["image_path"] == f]
        b = p2[(p2["image_path"] == f) & (p2["sequence_number"] != "NO_VEHICLES")]
        rows.append({
            "image_path": f,
            "n1": len(a), "n2": len(b),
            "ex1": (a["lane_type"] == "Express").sum(),
            "ex2": (b["lane_type"] == "Express").sum(),
            "lane_overlap": multiset_overlap(
                Counter(a["lane_type"]), Counter(b["lane_type"])),
            "mm_overlap": multiset_overlap(
                Counter(zip(a["vehicle_make"], a["vehicle_model"])),
                Counter(zip(b["vehicle_make"], b["vehicle_model"]))),
        })
    df = pd.DataFrame(rows)

    lines = ["# Test-retest validation report", ""]

    def add(s: str = "") -> None:
        lines.append(s)
        print(s)

    add(f"- Frames compared: **{len(df)}**")
    add(f"- Mean vehicle count: pass1 {df['n1'].mean():.2f}, pass2 {df['n2'].mean():.2f}")
    add(f"- Frames with identical vehicle count: "
        f"{(df['n1'] == df['n2']).mean():.1%}; within +/-1: "
        f"{((df['n1'] - df['n2']).abs() <= 1).mean():.1%}")
    add(f"- Frames with identical Express count: {(df['ex1'] == df['ex2']).mean():.1%}")
    add(f"- Mean lane-type multiset overlap: {df['lane_overlap'].mean():.3f}")
    add(f"- Mean make+model multiset overlap: {df['mm_overlap'].mean():.3f}")
    add()

    sub1 = p1[p1["image_path"].isin(frames)]
    sub2 = p2[p2["image_path"].isin(frames) & (p2["sequence_number"] != "NO_VEHICLES")]
    add("## Aggregate quantities (the ones the paper's findings rest on)")
    add(f"- Express share: pass1 {(sub1['lane_type'] == 'Express').mean():.3f}, "
        f"pass2 {(sub2['lane_type'] == 'Express').mean():.3f}")
    add(f"- Make identified: pass1 {(sub1['vehicle_make'] != 'UNKNOWN').mean():.3f}, "
        f"pass2 {(sub2['vehicle_make'] != 'UNKNOWN').mean():.3f}")
    for model in [("Toyota", "RAV4"), ("Ford", "F-150")]:
        c1 = ((sub1["vehicle_make"] == model[0]) & (sub1["vehicle_model"] == model[1])).sum()
        c2 = ((sub2["vehicle_make"] == model[0]) & (sub2["vehicle_model"] == model[1])).sum()
        add(f"- {model[0]} {model[1]} count: pass1 {c1}, pass2 {c2}")

    # worst-agreement frames for manual inspection
    add()
    add("## 15 lowest-agreement frames (inspect these by hand)")
    worst = df.nsmallest(15, "mm_overlap")
    for _, r in worst.iterrows():
        add(f"- `{r['image_path']}` n:{r['n1']}/{r['n2']} "
            f"express:{r['ex1']}/{r['ex2']} mm_overlap:{r['mm_overlap']:.2f}")

    OUT.write_text("\n".join(lines))
    print(f"\nreport written to {OUT}")


if __name__ == "__main__":
    main()
