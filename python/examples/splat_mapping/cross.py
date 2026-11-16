"""Cross-platform scoring: one vehicle's map seen from the other's viewpoints.

    overlap   annotate the held-out frames of dataset A with how much of each view dataset B's training views saw:
              frame["seen_by_<tag>"] = the share of the view's trainable pixels (every 8th) on surface that lies on a
              trained-on pixel of >= 3 of B's training depth maps (dataset.SeenTest), and frame["trainable_px"].
              Written back into A's transforms.json.
    table     cells from score.py's per-view lists: each cell is a map scored on one split of one dataset, over all
              of its scored views or over those with seen_by_<tag> >= a threshold. Means over the views (PSNR dB,
              SSIM), the view count, the mean trainable pixel count.

    py -3.14 cross.py overlap --data A --seen-by B --tag mariner [--splits on_path]
    py -3.14 cross.py table --out cross.json --cell "ROW|COLUMN|score.json|DATA|SPLIT|TAG|THRESHOLD" ...
        (TAG "-" = no restriction)
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dataset as ds  # noqa: E402
from manifest import write_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["overlap", "table"])
ap.add_argument("--data")
ap.add_argument("--seen-by")
ap.add_argument("--tag")
ap.add_argument("--splits", default="on_path")
ap.add_argument("--min-seen", type=int, default=3)
ap.add_argument("--cell", action="append", default=[])
ap.add_argument("--out")
args = ap.parse_args()


def frames_of(root, split):
    tf = json.load(open(os.path.join(root, "transforms.json"), encoding="utf-8"))
    return tf, [f for f in tf["frames"] if f["split"] == split]


if args.mode == "overlap":
    seen = ds.SeenTest(args.seen_by, args.min_seen)
    path = os.path.join(args.data, "transforms.json")
    tf = json.load(open(path, encoding="utf-8"))
    want = set(args.splits.split(","))
    for fr in tf["frames"]:
        if fr["split"] in want:
            share, npx = ds.frame_overlap(seen, fr, args.data, tf)
            fr[f"seen_by_{args.tag}"] = round(share, 4)
            fr["trainable_px"] = npx
    tf.setdefault("threepp", {})[f"seen_by_{args.tag}"] = {
        "rule": f"share of trainable pixels (every 8th) on surface on a trained-on pixel of >= {args.min_seen} "
                f"training depth maps of the other dataset, within 0.1 m + 1 % of depth",
        "other": os.path.basename(os.path.dirname(os.path.abspath(args.seen_by))), "splits": sorted(want)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tf, f, indent=1)
    for sp in sorted(want):
        v = [fr[f"seen_by_{args.tag}"] for fr in tf["frames"] if fr["split"] == sp]
        print(f"{sp}: {len(v)} views, seen by {args.tag}: median {np.median(v):.2f}, >= 0.30 in {sum(x >= 0.3 for x in v)}")
    write_manifest(os.path.dirname(os.path.abspath(args.data)), name=f"cross_overlap_{args.tag}",
                   extra={"data": args.data, "seen_by": args.seen_by})
else:
    cells = []
    for c in args.cell:
        row, col, score_path, root, split, tag, thr = c.split("|")
        sc = json.load(open(score_path, encoding="utf-8"))["splits"][split]
        _, frs = frames_of(root, split)
        assert len(frs) == len(sc["psnr"]), (score_path, split, len(frs), len(sc["psnr"]))
        pick = [j for j, fr in enumerate(frs) if sc["psnr"][j] is not None
                and (tag == "-" or fr[f"seen_by_{tag}"] >= float(thr))]
        p = np.array([sc["psnr"][j] for j in pick], float)
        s = np.array([sc["ssim"][j] for j in pick], float)
        cells.append({"row": row, "column": col, "score": score_path, "split": split, "restricted_by": None if tag == "-" else tag,
                      "threshold": None if tag == "-" else float(thr), "views": len(pick), "of_views": len(frs),
                      "psnr_mean": float(p.mean()) if len(p) else None, "psnr_min": float(p.min()) if len(p) else None,
                      "ssim_mean": float(s.mean()) if len(s) else None,
                      "trainable_px_mean": float(np.mean([sc["trainable_px"][j] for j in pick])) if pick and "trainable_px" in sc else None})
    cols = list(dict.fromkeys(c["column"] for c in cells))
    rows = list(dict.fromkeys(c["row"] for c in cells))
    print("| map | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 1) + "|")
    for r in rows:
        line = []
        for col in cols:
            c = next((c for c in cells if c["row"] == r and c["column"] == col), None)
            line.append("" if c is None or c["psnr_mean"] is None else f"{c['psnr_mean']:.2f} ({c['views']}/{c['of_views']})")
        print(f"| {r} | " + " | ".join(line) + " |")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(cells, f, indent=1)
