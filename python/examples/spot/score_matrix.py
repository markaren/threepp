"""Score one Spot checkpoint on the perception-robustness matrix, or aggregate a seed array.

    python score_matrix.py spot_steps.pt --json scores.jsonl        # 6 base cells -> one JSON line each
    python score_matrix.py spot_steps.pt --cells hard                # 5 HARD cells -> scores_hard.jsonl
    python score_matrix.py --aggregate /cluster/work/$USER/spot_runs  # mean +- 95% CI per condition x cell
    python score_matrix.py --aggregate ... --cells hard               # the same for the hard set

The cells are the deployment conditions a terrain policy can meet, whatever it was trained on:

    analytic     privileged scan from the closed-form tent formula (what spot_steps.pt was trained on)
    raycast      privileged scan read off the real collider (threepp.rl.raycast)
    perceived:N  the camera-limited scan (threepp.rl.perception) with N m of per-cell map error

Every cell is scored deterministically (act_mean, fixed seed) at the same K, so the only thing that
differs between two rows of one table is the checkpoint. Scoring a policy on a source it was not
trained on is deliberate: that mismatch IS the measurement.

Aggregation reads <run>/manifest.json (condition, seed, args) and <run>/scores.jsonl under a root
and prints, per condition x cell, mean and a 95% interval over seeds for track, fell/step and level.
"""
import argparse
import glob
import json
import math
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

# (label, extra CLI for train_spot_steps.py --score)
CELLS = [
    ("analytic",       ["--score-source", "analytic", "--score-perceive", "off"]),
    ("raycast",        ["--score-source", "raycast",  "--score-perceive", "off"]),
    ("perceived:0.00", ["--score-noise", "0.0"]),
    ("perceived:0.02", ["--score-noise", "0.02"]),
    ("perceived:0.05", ["--score-noise", "0.05"]),
    ("perceived:0.10", ["--score-noise", "0.10"]),
]
# The HARD set: every lane starts on the tallest risers (level 5 = 0.20 m) under 3 m/s shoves, so a
# converged policy is measured where it can still fail. The base set saturates at ~1.9 track / 0
# falls for every checkpoint because scoring climbs from level 0 and never gets past level 1.
HARD = ["--score-level", "5", "--score-push", "3.0"]
HARD_CELLS = [
    ("hard:analytic",       [*HARD, "--score-source", "analytic", "--score-perceive", "off"]),
    ("hard:raycast",        [*HARD, "--score-source", "raycast",  "--score-perceive", "off"]),
    ("hard:perceived:0.00", [*HARD, "--score-noise", "0.0"]),
    ("hard:perceived:0.05", [*HARD, "--score-noise", "0.05"]),
    ("hard:perceived:0.10", [*HARD, "--score-noise", "0.10"]),
]
CELL_SETS = {"base": CELLS, "hard": HARD_CELLS, "all": CELLS + HARD_CELLS}
SCORES_FILE = {"base": "scores.jsonl", "hard": "scores_hard.jsonl", "all": "scores_all.jsonl"}


def score(ckpt, json_path, envs=512, seed=0, cells=None):
    """One subprocess per cell: a fresh CUDA context and a fresh PhysX world each time, so a cell
    that dies (a raycast BVH that will not build, say) costs that cell and not the matrix."""
    failed = []
    for label, extra in (cells or CELLS):
        cmd = [sys.executable, os.path.join(_HERE, "train_spot_steps.py"), "--score", ckpt,
               "--score-envs", str(envs), "--seed", str(seed), "--score-json", json_path, *extra]
        print(f"\n== cell {label}: {' '.join(cmd[2:])}", flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            failed.append(label)
            print(f"!! cell {label} exited {rc}", flush=True)
    # Tag the lines with the cell label so the aggregate does not have to re-derive it.
    tagged = []
    for line in open(json_path):
        rec = json.loads(line)
        if "cell" not in rec:
            rec["cell"] = _cell_label(rec)
        tagged.append(rec)
    with open(json_path, "w") as f:
        for rec in tagged:
            f.write(json.dumps(rec) + "\n")
    return failed


def _cell_label(rec):
    hard = "hard:" if rec.get("start_level") is not None else ""
    if rec.get("score_perceive"):
        return f"{hard}perceived:{rec.get('score_noise', 0.0):.2f}"
    return hard + rec.get("score_source", "analytic")


def _ci(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan"), 0
    m = sum(xs) / n
    if n < 2:
        return m, float("nan"), n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    # t-quantile for small n (n-1 dof); falls back to 1.96 past the table.
    t = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26}.get(n - 1, 1.96)
    return m, t * sd / math.sqrt(n), n


def aggregate(root, out_md="", scores="scores.jsonl"):
    rows = {}     # (condition, cell) -> list of (track, fell, level, clear)
    runs = 0
    for man_path in sorted(glob.glob(os.path.join(root, "*", "manifest.json"))):
        run = os.path.dirname(man_path)
        man = json.load(open(man_path))
        sc = os.path.join(run, scores)
        if not os.path.exists(sc):
            print(f"(no scores yet: {run})")
            continue
        runs += 1
        for line in open(sc):
            rec = json.loads(line)
            key = (man["condition"], rec.get("cell") or _cell_label(rec))
            rows.setdefault(key, []).append((rec["track"], rec["fell"], rec["level"], rec.get("clear")))
    conds = sorted({c for c, _ in rows}, key=lambda c: [x[0] for x in CONDITIONS].index(c) if c in [x[0] for x in CONDITIONS] else 99)
    cells = [c for c, _ in CELL_SETS["all"] if any(c == k for _, k in rows)]
    hard = any(c.startswith("hard:") for c in cells)
    lines = [f"# Spot perception-robustness seed array: {runs} runs under {root} ({scores})", "",
             "Each cell: mean +- 95% CI over seeds of *track* (of 2.0), *fell/step*, *level* (of 5)"
             + (", *clear* (fraction of episodes clearing the tent)" if hard else "") + ".", "",
             "| trained on | " + " | ".join(cells) + " |",
             "|---|" + "---|" * len(cells)]
    for cond in conds:
        cellstr = []
        for cell in cells:
            xs = rows.get((cond, cell), [])
            if not xs:
                cellstr.append("-")
                continue
            t, tci, n = _ci([x[0] for x in xs])
            f, fci, _ = _ci([x[1] for x in xs])
            l, lci, _ = _ci([x[2] for x in xs])
            txt = f"{t:.3f}+-{tci:.3f} / {f:.4f}+-{fci:.4f} / {l:.2f}+-{lci:.2f}"
            if cell.startswith("hard:") and all(x[3] is not None for x in xs):
                c, cci, _ = _ci([x[3] for x in xs])
                txt += f" / {c:.2f}+-{cci:.2f}"
            cellstr.append(txt + f" (n={n})")
        lines.append(f"| {cond} | " + " | ".join(cellstr) + " |")
    text = "\n".join(lines)
    print(text)
    if out_md:
        with open(out_md, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\nwrote {out_md}")


# The training conditions of the array, the single source of truth the Slurm script reads
# (`python score_matrix.py --condition 3` prints the trainer flags for array index 3 % len).
CONDITIONS = [
    ("analytic",  []),
    ("raycast",   ["--height-source", "raycast"]),
    ("perc0.00",  ["--perceive"]),
    ("perc0.02",  ["--perceive", "--perceive-noise", "0.02"]),
    ("perc0.05",  ["--perceive", "--perceive-noise", "0.05"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="")
    ap.add_argument("--json", default="", help="scores.jsonl to append to (default: next to the checkpoint)")
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0, help="score seed (the command sampler); NOT the training seed")
    ap.add_argument("--aggregate", default="", help="root of run directories to summarise")
    ap.add_argument("--md", default="", help="with --aggregate: also write the table here")
    ap.add_argument("--cells", choices=tuple(CELL_SETS), default="base",
                    help="which cell set to score (default base); the JSON file name follows it")
    ap.add_argument("--scores", default="", help="with --aggregate: the per-run scores file name "
                    "(default follows --cells: scores.jsonl / scores_hard.jsonl)")
    ap.add_argument("--condition", type=int, default=-1,
                    help="print '<name>\\t<trainer flags>' for this array index and exit")
    ap.add_argument("--n-conditions", action="store_true")
    args = ap.parse_args()
    if args.n_conditions:
        print(len(CONDITIONS)); return
    if args.condition >= 0:
        name, flags = CONDITIONS[args.condition % len(CONDITIONS)]
        print(name + "\t" + " ".join(flags)); return
    if args.aggregate:
        aggregate(args.aggregate, args.md, args.scores or SCORES_FILE[args.cells]); return
    if not args.checkpoint:
        ap.error("give a checkpoint, --aggregate, or --condition")
    js = args.json or os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), SCORES_FILE[args.cells])
    failed = score(args.checkpoint, js, envs=args.envs, seed=args.seed, cells=CELL_SETS[args.cells])
    print(f"\nscores -> {js}" + (f"  (FAILED cells: {failed})" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
