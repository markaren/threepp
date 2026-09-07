"""Read the crane_lift manifests and logs in a folder: fresh-process replay per row, the first
differing frame of the rendered rows, the seed spread of the lift, and the anti-swing control.

    python analyze.py <folder with audit_*.json, op_s0_{a,b,c}.json(.npz), op_s1.json, op_s0_noas.json>
"""
import glob
import json
import math
import os
import sys

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))


def load(name):
    with open(os.path.join(D, name)) as fh:
        return json.load(fh)


def replay(names, title):
    ms = [load(n) for n in names if os.path.exists(os.path.join(D, n))]
    if len(ms) < 2:
        return
    print(f"{title}: {len(ms)} fresh processes")
    for key in ms[0]["rows"]:
        vals = [m["rows"][key]["fnv"] if isinstance(m["rows"][key], dict) else m["rows"][key] for m in ms]
        n = len(set(vals))
        print(f"  {'OK  ' if n == 1 else 'DIFF'} {key:14s} {n} distinct of {len(ms)}")
    for key in ("rgb", "tip.rgb"):
        seqs = [m.get("per_frame", {}).get(key, []) for m in ms]
        if all(seqs) and len(set(tuple(s) for s in seqs)) > 1:
            first = next((i for i in range(min(map(len, seqs))) if len(set(s[i] for s in seqs)) > 1), None)
            n_diff = sum(1 for i in range(min(map(len, seqs))) if len(set(s[i] for s in seqs)) > 1)
            print(f"       {key}: first differing frame {first}, {n_diff} of {min(map(len, seqs))} frames differ")


def log_of(name):
    p = os.path.join(D, name[:-5] + ".npz")
    return np.load(p)["log"] if os.path.exists(p) else None


def swing(L):
    return np.hypot(L[:, 4] - L[:, 1], L[:, 6] - L[:, 3])


def cross_track(p, ref):
    a, b = ref[:-1], ref[1:]
    ab = b - a
    den = np.maximum((ab * ab).sum(1), 1e-12)
    out = np.empty(len(p))
    for i0 in range(0, len(p), 256):
        q = p[i0:i0 + 256]
        t = np.clip(((q[:, None, :] - a[None]) * ab[None]).sum(2) / den[None], 0.0, 1.0)
        c = a[None] + t[..., None] * ab[None]
        out[i0:i0 + 256] = np.sqrt(((q[:, None, :] - c) ** 2).sum(2)).min(1)
    return out


replay(["audit_a.json", "audit_b.json", "audit_c.json"], "audit, 120 frames")
replay(["op_s0_a.json", "op_s0_b.json", "op_s0_c.json"], "the lift, seed 0")

L0 = log_of("op_s0_a.json")
if L0 is not None:
    s0 = swing(L0)
    err = np.linalg.norm(L0[:, 1:4] - L0[:, 14:17], axis=1)
    print(f"lift seed 0: {L0[-1, 0]:.0f} s, swing RMS {1e3 * math.sqrt((s0 ** 2).mean()):.0f} mm, max {1e3 * s0.max():.0f} mm; "
          f"tip off its nominal target RMS {1e3 * math.sqrt((err ** 2).mean()):.0f} mm; heave span {L0[:, 11].max() - L0[:, 11].min():.2f} m, "
          f"roll +-{math.degrees(np.abs(L0[:, 13]).max()):.1f} deg")
    Ln = log_of("op_s0_noas.json")
    if Ln is not None:
        sn = swing(Ln)
        print(f"  without the anti-swing loop: swing RMS {1e3 * math.sqrt((sn ** 2).mean()):.0f} mm, max {1e3 * sn.max():.0f} mm "
              f"(hold phase after 44 s: {1e3 * math.sqrt((sn[Ln[:, 0] > 44] ** 2).mean()):.0f} vs {1e3 * math.sqrt((s0[L0[:, 0] > 44] ** 2).mean()):.0f} mm RMS)")
    seeds = [(k, log_of(f"op_s{k}.json")) for k in range(1, 10)]
    seeds = [(k, L) for k, L in seeds if L is not None]
    if seeds:
        ref = L0[:, [4, 6]]                                   # the hook's path, top view
        devs = [cross_track(L[:, [4, 6]], ref) for _, L in seeds]
        allv = np.concatenate(devs)
        print(f"  seeds {[k for k, _ in seeds]}: hook path deviation from seed 0, RMS {1e3 * math.sqrt((allv ** 2).mean()):.0f} mm, "
              f"max {1e3 * allv.max():.0f} mm; tip path " +
              ", ".join(f"{1e3 * math.sqrt((cross_track(L[:, [1, 3]], L0[:, [1, 3]]) ** 2).mean()):.0f} mm" for _, L in seeds))
