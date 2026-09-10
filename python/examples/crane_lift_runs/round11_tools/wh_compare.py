"""Group warm-up-hash runs by their rgb row and find the first frame at which the groups part.

    python wh_compare.py <dir> [prefix]
"""
import collections
import glob
import json
import os
import subprocess
import sys

d = sys.argv[1]
prefix = sys.argv[2] if len(sys.argv) > 2 else ""
runs = {}
for p in sorted(glob.glob(os.path.join(d, prefix + "*.json"))):
    n = os.path.basename(p)[:-5]
    if n == "diag_meta":
        continue
    runs[n] = json.load(open(p))
byvar = collections.defaultdict(dict)
for n, r in runs.items():
    byvar[n.rsplit("_", 1)[0]][n] = r
KEYS = ["rgb", "depth", "normals", "instance_ids", "motion", "albedo", "tip"]
for var, rs in byvar.items():
    groups = collections.defaultdict(list)
    for n, r in rs.items():
        groups[r["rows"]["rgb"]["fnv"]].append(n)
    print(f"== {var}: {len(rs)} runs, rgb groups: " + "  ".join(",".join(v) for v in groups.values()))
    reps = [v[0] for v in groups.values()]
    # within-group check: warm-up hashes identical?
    for g, names in groups.items():
        a = rs[names[0]]["diag"]
        for n in names[1:]:
            b = rs[n]["diag"]
            same = a["first"] == b["first"] and a["warmup"] == b["warmup"]
            print(f"   within group {g[:6]}: {names[0]} vs {n}: warm-up hashes {'identical' if same else 'DIFFER'}")
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            a, b = rs[reps[i]], rs[reps[j]]
            print(f"-- {reps[i]} vs {reps[j]}")
            fa, fb = a["diag"]["first"], b["diag"]["first"]
            print("   first render: " + "  ".join(f"{k}={'=' if fa.get(k) == fb.get(k) else 'X'}" for k in KEYS if k in fa))
            W = min(len(a["diag"]["warmup"]), len(b["diag"]["warmup"]))
            for k in KEYS:
                bad = [t for t in range(W) if a["diag"]["warmup"][t].get(k) != b["diag"]["warmup"][t].get(k)]
                print(f"   warm-up {k:12s}: {W - len(bad)}/{W} agree; differing {bad[:16]}{'...' if len(bad) > 16 else ''}")
            ca, cb = a["per_frame"]["rgb"], b["per_frame"]["rgb"]
            bad = [t for t in range(min(len(ca), len(cb))) if ca[t] != cb[t]]
            print(f"   capture rgb: {len(bad)} differing of {min(len(ca), len(cb))}; first {bad[:5]}")
            ta, tb = a["per_frame"]["tip.rgb"], b["per_frame"]["tip.rgb"]
            bad = [t for t in range(min(len(ta), len(tb))) if ta[t] != tb[t]]
            print(f"   capture tip: {len(bad)} differing of {min(len(ta), len(tb))}; first {bad[:5]}")
            # frame dumps
            here = os.path.dirname(os.path.abspath(__file__))
            for tag in ("first", "warm0", "warm1", "warm2", "warm3", "cap0", "cap0_tip"):
                pa = os.path.join(d, f"{reps[i]}_{tag}.npy")
                pb = os.path.join(d, f"{reps[j]}_{tag}.npy")
                if os.path.exists(pa) and os.path.exists(pb):
                    out = subprocess.run([sys.executable, os.path.join(here, "diff_frames.py"), pa, pb,
                                          os.path.join(d, f"diff_{reps[i]}_{reps[j]}_{tag}.png")],
                                         capture_output=True, text=True).stdout
                    lines = out.strip().splitlines()
                    print(f"   dump {tag:8s}: " + (lines[1] if len(lines) > 1 else out.strip()))
                    if len(lines) > 2 and "differing pixels: 0 " not in lines[1]:
                        for ln in lines[2:5]:
                            print("             " + ln)
