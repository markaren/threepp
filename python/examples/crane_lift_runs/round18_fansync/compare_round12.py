"""Round 18 against round 12, run for run: the report lines the paper's E2 numbers come from, side
by side, and how far the seed-0 trajectory moved when the fan started reading its own frame.

    python compare_round12.py [round12_dir] [round18_dir]

Reads each run's .log (crane_lift.py's op_report lines) and .npz log. Log columns: 0 t, 1-3 tip,
4-6 hook, 17-18 swing estimate, 23 landed, 27-29 container centre (see analyze.py).
"""
import glob
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R12 = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "round12_protocol")
R18 = sys.argv[2] if len(sys.argv) > 2 else HERE
KEYS = ("arrival: swing", "pay-out  (", "LANDED", "sensor: the load", "seeds [")


def report_lines(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in open(path, encoding="utf-8", errors="replace"):
        for k in KEYS:
            if line.strip().startswith(k):
                out[k] = line.strip()
    return out


def landing(npz, meta):
    L = np.load(npz)["log"]
    g = meta["geom"]
    land = np.array(g["turbine"]) + np.array(g["land_off"])
    i = np.argmax(L[:, 23] > 0) if (L[:, 23] > 0).any() else None
    if i is None:
        return None, None
    return float(L[i, 0]), 1e3 * float(np.hypot(L[i, 27] - land[0], L[i, 29] - land[2]))


def main():
    names = ["op_s0_noas", "op_s0_a", "op_s0_film"] + [f"op_s{k}" for k in range(1, 10)]
    print(f"round 12: {R12}\nround 18: {R18}\n")
    for n in names:
        a, b = report_lines(os.path.join(R12, n + ".log")), report_lines(os.path.join(R18, n + ".log"))
        if not b:
            print(f"{n}: round 18 not finished")
            continue
        print(f"== {n}")
        for k in KEYS:
            if k in a or k in b:
                print(f"  r12 {a.get(k, '-')}")
                print(f"  r18 {b.get(k, '-')}")
    # the landing table over all seeds
    print("\nlanding (t_land s, centre off the mark mm):")
    for n in ["op_s0_noas", "op_s0_a"] + [f"op_s{k}" for k in range(1, 10)]:
        row = [n]
        for d in (R12, R18):
            j, z = os.path.join(d, n + ".json"), os.path.join(d, n + ".npz")
            if os.path.exists(j) and os.path.exists(z):
                t, off = landing(z, json.load(open(j))["meta"])
                row.append(f"{t:.2f} s {off:5.0f} mm" if t is not None else "not landed")
            else:
                row.append("-")
        print(f"  {row[0]:11s} r12 {row[1]:>18s}   r18 {row[2]:>18s}")
    # seed-0 trajectory: how far the container's path moved between the rounds
    za, zb = os.path.join(R12, "op_s0_a.npz"), os.path.join(R18, "op_s0_a.npz")
    if os.path.exists(za) and os.path.exists(zb):
        A, B = np.load(za)["log"], np.load(zb)["log"]
        n = min(len(A), len(B))
        d = np.hypot(A[:n, 27] - B[:n, 27], A[:n, 29] - B[:n, 29])
        first = int(np.argmax(d > 1e-9)) if (d > 1e-9).any() else None
        est = np.hypot(A[:n, 17] - B[:n, 17], A[:n, 18] - B[:n, 18])
        print(f"\nseed 0, round 18 vs round 12 over {n} rows: container centre first differs at row {first} "
              f"(t = {A[first, 0]:.2f} s), horizontal difference RMS {1e3 * np.sqrt((d ** 2).mean()):.1f} mm, max {1e3 * d.max():.1f} mm; "
              f"swing estimate differs RMS {1e3 * np.sqrt((est ** 2).mean()):.1f} mm, max {1e3 * est.max():.1f} mm"
              if first is not None else f"\nseed 0: the container's path is identical over {n} rows")
    # replay: the rows of the seed-0 processes
    ms = sorted(glob.glob(os.path.join(R18, "op_s0_[a-j].json"))) + glob.glob(os.path.join(R18, "op_s0_film.json"))
    if ms:
        rows = {}
        for m in ms:
            r = json.load(open(m))["rows"]
            for k, v in r.items():
                rows.setdefault(k, set()).add(v["fnv"] if isinstance(v, dict) else v)
        print(f"\nround 18 seed-0 processes: {len(ms)}; distinct hashes per row: " +
              ", ".join(f"{k} {len(v)}" for k, v in rows.items()))


if __name__ == "__main__":
    main()
