"""Read the crane_lift manifests and logs in a folder: fresh-process replay per row, the first
differing frame of the rendered rows, the seed spread of the lift, and the anti-swing control.

    python analyze.py <folder with audit_*.json, op_s0_{a,b,c}.json(.npz), op_s1.json, op_s0_noas.json>

Log columns (the .npz `log`): 0 t, 1-3 tip world, 4-6 hook world, 7-9 q (slew, luff, telescope),
10 wire length, 11 heave, 12 pitch, 13 roll, 14-16 nominal tip target, 17-18 sensor swing estimate,
19-21 the MRU feedforward joint velocities q_dot_ff (appended 2026-09-07; absent in rounds 1-3),
22 contact (0/1), 23 landed (0/1), 24-25 the anti-swing correction u in world x, z
(appended 2026-09-07, round 5; absent in rounds 1-4), 26 hoist tension in newtons (appended
2026-09-07, round 6; absent in rounds 1-5, and zero in any round-6 run made with --payload pbd,
which has no load cell), 27-29 the container's CENTRE in the world and 30 its tilt in degrees
(the angle its own up-axis makes with vertical; appended 2026-09-07, round 7, absent in rounds
1-6; under --payload pbd the centre is the anchor minus the sling-plus-half-height drop and the
tilt is identically zero, since that model's load cannot rotate). The npz also carries `imu` (N x 7 float64: t, gyro xyz,
accel xyz) from the hull IMU at 100 Hz, the motion reference unit, seeded MEMS noise as sensor_audit.py
seeds it, on a kinematic body that rides the vessel's pose; a manifest row `imu` of its own (appended
2026-09-07 late, absent before; empty under --payload pbd).

Column 4-6, "hook", is whichever point the wire ends at in the payload model that produced the
run: the pendulum bob under `--payload pbd` (rounds 1-5 and the pbd control of round 6), and the
CONTAINER'S OWN ANCHOR POINT - its top centre plus the sling height - under `--payload physx`.
Round 6's container is a rigid body on a DISTANCE joint, so it tilts, and the anchor and the
container's centre are no longer the same horizontal point: the centre hangs on an effective
pendulum of wire + 4.16 m while the anchor hangs on the wire alone. The tip range sensor sees
the CENTRE (columns 17-18), so |(17,18)| runs about 1.5x |swing| in a physx run by construction,
not by sensor error; crane_lift.py's own report prints both ratios.

Derived quantities the round-5 and round-6 tables use, all from those columns: the swing is the
horizontal hook-tip distance, hypot(4-1, 6-3); the hook's nominal point is the tip target with
the wire subtracted, (14, 15 - 10, 16); the wire slack is 10 - |(4,5,6) - (1,2,3)|; the sensor's
gain is the median of |(17,18)| / |swing| over the frames carrying an estimate; the realised
joint rates and accelerations are the first and second differences of 7-9 at 60 Hz, against
V_MAX and A_MAX; the tether is slack (the load is down) where 26 falls to zero.

The phase boundaries are in the manifest's meta.op. Rounds 1-4: hold 4 s, transfer 40 s, pay-out
12 s. Round 5: 4 / 60 / 12 s. Round 6 adds an ARRIVAL HOLD between the transfer and the pay-out
and the transfer's length follows from an acceleration-limited trapezoid rather than being
chosen, so meta.op carries `boundaries` = [T1, T2, T3, T4] (4.00 / 40.59 / 48.59 / 60.59 s at
the round-6 geometry) plus `t_arrive`, `profile`, `sweep_rad` and `xfer_peak_rate`. meta.geom
carries the turbine placement, the measured deck rectangle and the tower; meta.payload the model,
the container's mass, the PhysX substeps and the minimum container-to-tower clearance.

Round 7 flies a 10 ft box (`--load 10ft|20ft`) and moves the landing point to where the round-7
clearances hold (1.5 m to the tower surface, 1.0 m to every railing, no overlap with the three
door leaves, which are also taken out of the collider set). meta.geom gains `land_off`,
`clear_rect` (the rectangle the container's footprint must lie inside), `rail_margin`,
`tower_margin` and the measured `doors`; meta.payload gains `load`, the box's half-extents,
`doors_excluded`, the minimum railing and door clearances with the times they were crossed, and
the resting attitude (`rest_tilt_deg`, `rest_anchor_above_level`, `in_clear_rect`). meta gains
`joint_margin`: per joint, the smallest margin the run ever left to its position limits.
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


replay([f"audit_{k}.json" for k in "abcdefghij"], "audit, 120 frames")
replay([f"op_s0_{k}.json" for k in "abcdefghij"] + ["op_s0_film.json"], "the lift, seed 0")

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
        op = load("op_s0_a.json")["meta"].get("op", {})
        t2 = op.get("t_hold", 4.0) + op.get("t_xfer", 40.0)     # the pay-out begins: 44 s in rounds 1-4, 64 s in round 5
        print(f"  without the anti-swing loop: swing RMS {1e3 * math.sqrt((sn ** 2).mean()):.0f} mm, max {1e3 * sn.max():.0f} mm "
              f"(pay-out phase after {t2:.0f} s: {1e3 * math.sqrt((sn[Ln[:, 0] > t2] ** 2).mean()):.0f} vs {1e3 * math.sqrt((s0[L0[:, 0] > t2] ** 2).mean()):.0f} mm RMS)")
    seeds = [(k, log_of(f"op_s{k}.json")) for k in range(1, 10)]
    seeds = [(k, L) for k, L in seeds if L is not None]
    if seeds:
        ref = L0[:, [4, 6]]                                   # the hook's path, top view
        devs = [cross_track(L[:, [4, 6]], ref) for _, L in seeds]
        allv = np.concatenate(devs)
        print(f"  seeds {[k for k, _ in seeds]}: hook path deviation from seed 0, RMS {1e3 * math.sqrt((allv ** 2).mean()):.0f} mm, "
              f"max {1e3 * allv.max():.0f} mm; tip path " +
              ", ".join(f"{1e3 * math.sqrt((cross_track(L[:, [1, 3]], L0[:, [1, 3]]) ** 2).mean()):.0f} mm" for _, L in seeds))
