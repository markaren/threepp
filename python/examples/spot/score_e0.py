"""E0-lite: score Spot checkpoints on episode-counted cells in ONE SpotStepsEnv world, zero training.

    python score_e0.py spot_steps.pt --label ctrl:spot_steps --seed 1 --json e0_s1.jsonl
    python score_e0.py spot_steps.pt --envs 256 --blocks S20,P2,F --recount       # local smoke
    python score_e0.py --aggregate /cluster/work/$USER/spot_e0                     # CPU: the tables
    python score_e0.py spot_steps.pt --suite showpiece --envs 128 --json sp.jsonl --recount   # + sp_touchdowns.npz

Suites: `e0` (default) is every BLOCKS cell with the legacy termination, so the E0 job list and its numbers
reproduce; `showpiece` is S10, S20, S20V6, S23, S20X15, P2, P3, FP4, F plus the SLOPE_BLOCKS (planar up SU05-SU20,
down SD10-SD20, cross SC05-SC25, and SU15M / SC15M on the explicit 1.0 'min' material) with the honest termination.
Slope cells report falls per 100 m along the surface, drift, slip and a spawn gate (a cell whose settle health fails
more than 2% of the time is flagged in the header and is not a result). Both run
the instrument by default (spot_feet.py; --no-instrument skips it): touchdowns placed against the tread
closed form with the chance baseline beside every row, falls per 100 m by honest cause, drift, slip,
recovery, and p95 commanded torque / joint speed as validity gates. With --json, the per-touchdown rows go
to <json stem>_touchdowns.npz with the per-lane tread geometry.

One world per (checkpoint, score seed). The K lanes are split into contiguous BLOCKS, each one cell:
its own lane type, frozen level (one band per lane, so a lane meets exactly one tent and then flat
ground, as the top rung always has), held command, shove, foot friction and payload. A flat,
unscored buffer lane sits wherever the geometry changes between blocks. Every checkpoint is scored
against the raycast oracle scan with perception off, deterministic act_mean through its own
RunningNorm, whatever it was trained on. Episodes start together; the counters take each lane's
first E = 2 episodes, and the run stops once every lane has ended E (about 1600 steps).

Output: one JSON line per block (every SpotStepsEnv counter plus rates) after a header line with the
checkpoint meta, git sha, build/step timing and a hash of (obs, root_pos) at steps 100/600/1200. Two
runs of one checkpoint at one seed must agree counter for counter and hash for hash; if they do not,
PhysX-GPU is not deterministic here and every noise estimate has to be empirical.

What the numbers mean (SpotStepsEnv.STAT_KEYS): crossed = forward progress past the tent (CLEAR_DIST)
alone; cleared = crossed while within 1.3 m of the lane centre AT THE CROSSING; success = cleared AND
timed out (never fell). So a lost success decomposes into climbing (crossed) and holding the lane
(cleared/crossed), and out_of_lane stays what it is — where the robot ended up 10 s later. Falls per
push = falls within 100 control steps (2 s) of the one shove delivered; a lane that falls before
reaching its trigger x is never shoved and is not in the denominator.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))

# Levels 0-5 are the training ladder; 6-8 extend it past anything a 96-d checkpoint has seen.
E0_RISERS = (0.04, 0.07, 0.10, 0.13, 0.16, 0.20, 0.23, 0.26, 0.30)


def _blk(stairs, level, vx, dv=0.0, mu=None, payload=0.0, lanes=1, scan_dx=0.0, slope=None, deg=0.0,
         material=None, measure_only=False):
    return {"stairs": stairs, "level": level, "vx": vx, "push_dv": dv, "foot_mu": mu,
            "payload_kg": payload, "lanes": lanes, "scan_dx": scan_dx, "slope": slope, "deg": deg,
            "material": material, "measure_only": measure_only}


# Order matters, but not for buffers any more (there is one between every pair): the shove ladder
# sits at one end, so the cells everything else is read against — the low rung, S20 nominal, the
# contact pair, the tall rungs — never touch a lane that takes a 4-6 m/s sideways shove. `lanes` is
# a share, not a count: the FLAT shove ladder carries the positive control, whose margin is the
# tightest in the suite.
BLOCKS = {
    "S10":   _blk(True, 2, 0.8),                # the low rung. Every other stair cell sits at level >= 5,
                                                # where "cannot climb 0.20 m" and "climbs it and then
                                                # falls" both read success 0.000 — which is exactly what
                                                # hid the positive control's condition. This one is the
                                                # reference that tells the two apart on any checkpoint.
    "S20":   _blk(True, 5, 0.8),                # the nominal 0.20 m tent, no shove
    "S20f":  _blk(True, 5, 1.2),
    "MU5":   _blk(True, 5, 0.8, mu=0.5),        # the bridge: nominal friction, but through an EXPLICIT
    "MU3":   _blk(True, 5, 0.8, mu=0.3),        # foot material (restitution 0, 'min'), so MU3 - MU5 is
    "PAY7":  _blk(True, 5, 0.8, payload=7.0),   # friction alone and MU5 - S20 is the material itself
    "S23":   _blk(True, 6, 0.8),
    "MU3H":  _blk(True, 6, 0.8, mu=0.3),        # the same contact and load questions one rung up. At
    "PAY7H": _blk(True, 6, 0.8, payload=7.0),   # level 5 the shipped policy reads success 1.000 through
                                                # S20, MU5 and MU3 alike, so the level-5 trio answers
                                                # "what does the material cost" and nothing else; the
                                                # 0.23 m pair is where friction and payload have room
                                                # to show a difference (see R4 in the aggregate).
    "S26":   _blk(True, 7, 0.8),
    "S30":   _blk(True, 8, 0.8),
    "F":     _blk(False, 0, 0.8),
    "P1":    _blk(True, 5, 0.8, dv=1.0),        # one shove of exactly dv m/s at a random x on the tent
    "P2":    _blk(True, 5, 0.8, dv=2.0, lanes=2),
    "P3":    _blk(True, 5, 0.8, dv=3.0, lanes=2),
    "P4":    _blk(True, 5, 0.8, dv=4.0),
    "P5":    _blk(True, 5, 0.8, dv=5.0),
    "FP2":   _blk(False, 0, 0.8, dv=2.0),       # the same shove on flat ground, over [spawn+1, spawn+6].
    "FP3":   _blk(False, 0, 0.8, dv=3.0),       # THE POSITIVE CONTROL IS READ HERE, not on the tent:
    "FP4":   _blk(False, 0, 0.8, dv=4.0, lanes=2),   # spot_steps_push2.pt makes 0.93-1.21 m on a 0.20 m
    "FP5":   _blk(False, 0, 0.8, dv=5.0, lanes=2),   # tent, so 1-5 of its lanes per on-tent block reach
    "FP6":   _blk(False, 0, 0.8, dv=6.0, lanes=2),   # their trigger x: P2 reads 0 falls of 3-4 pushes.
                                                # Every flat lane is shoved, and there the instrument
                                                # separates the two checkpoints cleanly: falls per push
                                                # 0.270 / 0.670 / 0.760 (shipped) vs 0.000 / 0.010 /
                                                # 0.050 (push2) at dv 4 / 5 / 6, K=256.
}
# Cells only the showpiece suite uses. Kept out of BLOCKS so the E0 job list (every BLOCKS cell, in order) and
# its lane layout — and so every E0 number — are unchanged.
EXTRA_BLOCKS = {
    "S20V6":  _blk(True, 5, 0.6),               # the commensurate speed: stride = v * 0.5 s = one 0.30 m tread,
                                                # so the touchdown phase stops walking across the tread
    "S20X15": _blk(True, 5, 0.8, scan_dx=0.15), # THE PLACEMENT PROOF: S20 with the observation's scan shifted
                                                # +0.15 m (half a tread) forward. Reward, termination and every
                                                # counter stay on the real terrain; if placement follows the scan,
                                                # the touchdowns move off the real tread centre with it.
}
# The explicit terrain material of an added contact cell: friction 1.0, restitution 0, 'min' combine (= spot_slopes.GRIPPY,
# checked in score()). 'min' outranks the default ground's 'average', so the ramp only grips if the feet are 1.0 'min'
# too: the M cells carry foot_mu 1.0. The up approach is the shared ground and keeps the default material.
SLOPE_MATERIAL = (1.0, 1.0, 0.0, "min", "min")
# Slope cells (spot_slopes lanes, raycast only), all at vx 0.8 held. Up and down ramps are 20 m along the surface;
# a cross lane is tilted about x, so walking +x is level and gravity pulls across the lane. 0-20 deg is the target
# envelope (plans/spot-frontier.md, Target); SC25 sits one rung above it so the ladder cannot go blind.
SLOPE_BLOCKS = {
    **{f"SU{d:02d}": _blk(False, 0, 0.8, slope="up", deg=float(d)) for d in (5, 10, 15, 20)},
    **{f"SD{d:02d}": _blk(False, 0, 0.8, slope="down", deg=float(d)) for d in (10, 15, 20)},
    **{f"SC{d:02d}": _blk(False, 0, 0.8, slope="cross", deg=float(d)) for d in (5, 10, 15, 20)},
    "SC25": _blk(False, 0, 0.8, slope="cross", deg=25.0, measure_only=True),
    "SU15M": _blk(False, 0, 0.8, mu=1.0, slope="up", deg=15.0, material=SLOPE_MATERIAL),
    "SC15M": _blk(False, 0, 0.8, mu=1.0, slope="cross", deg=15.0, material=SLOPE_MATERIAL),
}
ALL_BLOCKS = {**BLOCKS, **EXTRA_BLOCKS, **SLOPE_BLOCKS}
SUITES = {"e0": list(BLOCKS),
          "showpiece": ["S10", "S20", "S20V6", "S23", "S20X15", "P2", "P3", "FP4", "F", *SLOPE_BLOCKS]}
LANE_FAMILY = {None: "flat", "up": "slope_up", "down": "slope_down", "cross": "cross"}   # SpotStepsEnv lane_family
# Unscored lanes of a slope block's own family and angle on each side of it. Two, measured: on the up ramps
# spot_steps.pt drifts one way (+y), ending +3.38 m from its lane on average at 20 deg and up to 4.48 m, which is the
# outer edge of ONE guard lane (4.5 m) and the start of the cliff beside it.
SLOPE_GUARD = 2
# Cross-slope geometry. 'offset' (one continuous plane per cross block, every lane pulling toward -y), measured
# 2026-09-12 on spot_steps.pt at K=256: drift toward gravity 0.108 / 0.234 / 0.364 / 0.521 / 0.885 m per metre at
# 5 / 10 / 15 / 20 / 25 deg. spot_slopes' 'corrugate' zig-zag puts a V gutter 1.5 m from every lane centre and read
# 0.099 / 0.104 / 0.105 / 0.105 / 0.106 on the same cells: every episode ended in the gutter (1.29-1.41 m), so it
# cannot tell a policy that holds its line from one that slides. The off-surface gate (spot_slopes.OFF_SURFACE_MAX)
# flags a cell whose robots slid off their block's plane (SC25 does).
CROSS_MODE = "offset"
SUITE_TERMINATION = {"e0": "legacy", "showpiece": "honest"}
# The plant each suite is measured on unless the CLI says otherwise: e0 keeps the impulse caps so its job list
# reproduces E0-lite; showpiece uses real torque limits, the agreed plant for every number from S0b on.
SUITE_DRIVE_LIMITS = {"e0": False, "showpiece": True}
HASH_STEPS = (100, 600, 1200)
BUFFER = "_buf"
LOW_RUNG = "S10"               # printed first everywhere: it is what makes a 0.000 row interpretable
GATE_FPP = ("FP4", "FP5")      # G3, the positive control, on flat ground where push2 is measurable
FRONTIER_FPP = ("P2", "P3")    # still reported — the on-tent frontier — but not what G3 is read from
R4_PAIR = ("MU3H", "S23")      # friction with headroom; the level-5 trio stays as the material control


def layout(names, k, buffer=1, guard=SLOPE_GUARD):
    """Per-lane arrays for the blocks `names`: contiguous lanes per block, sized by the block's
    `lanes` share (the remainder one lane at a time from the front), and `buffer` flat unscored lanes
    between EVERY pair of blocks — not only where the geometry changes. Lanes are SPACING (3 m)
    apart, nothing filters collisions between robots, and a shove of up to 5 m/s moves one further
    than that, so a shoved block must not sit against the block it is compared against.

    A slope block is also flanked, inside those buffers, by `guard` unscored lanes of its own family, angle and
    material on each side. Lanes of one family and angle tile with no step, and a lane of another kind meets them
    at a cliff (up to 20 m * sin(theta) beside a plateau, spot_slopes), so a scored slope lane never borders one.
    Guards take the buffer's block id and everything else from their block. Without slope blocks the arrays are
    exactly the historical ones."""
    slope = [ALL_BLOCKS[b]["slope"] is not None for b in names]
    n_guard = 2 * guard * sum(slope)
    n_buf = buffer * (len(names) - 1) + n_guard
    w = [ALL_BLOCKS[b]["lanes"] for b in names]
    avail = k - n_buf
    cnt = [avail * x // sum(w) for x in w]
    for i in range(avail - sum(cnt)):                # the remainder, one lane each from the front
        cnt[i % len(names)] += 1
    if min(cnt) < 1:
        raise ValueError(f"{k} envs cannot hold {len(names)} blocks + {n_buf} buffer lanes")
    bid, src = [], []                                # src = the spec a lane takes its geometry and command from
    for i, b in enumerate(names):
        if i > 0:
            bid += [len(names)] * buffer
            src += [len(names)] * buffer
        g = guard if slope[i] else 0
        bid += [len(names)] * g + [i] * cnt[i] + [len(names)] * g
        src += [i] * (cnt[i] + 2 * g)
    bid, src = np.array(bid, np.int64), np.array(src, np.int64)
    spec = [ALL_BLOCKS[b] for b in names] + [_blk(False, 0, 0.8)]        # last = the buffer
    pick = lambda key, dt: np.array([spec[j][key] for j in src], dt)
    mu = np.array([np.nan if spec[j]["foot_mu"] is None else spec[j]["foot_mu"] for j in src], np.float64)
    hold = np.zeros((k, 3), np.float32)
    hold[:, 0] = pick("vx", np.float32)
    return {"bid": bid, "stairs": pick("stairs", bool), "level": pick("level", np.int64), "hold": hold,
            "dv": pick("push_dv", np.float32), "mu": mu, "payload": pick("payload_kg", np.float64),
            "scan_dx": pick("scan_dx", np.float32), "names": list(names) + [BUFFER],
            "family": ["stairs" if spec[j]["stairs"] else LANE_FAMILY[spec[j]["slope"]] for j in src],
            "deg": pick("deg", np.float64), "material": [spec[j]["material"] for j in src], "guard_lanes": n_guard}


def _git():
    try:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
        sha = subprocess.check_output(["git", "-C", repo, "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "-C", repo, "status", "--porcelain"], text=True,
                                        stderr=subprocess.DEVNULL).count("\n")
        return sha, dirty
    except Exception:                                               # noqa: BLE001
        return "unknown", -1


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rates(r):
    """Rates for one counter row (None where the denominator is 0)."""
    n, p, t = r["episodes"], r["pushes"], r["terminations"]
    div = lambda a, b: (a / b) if b else None
    return {"success": div(r["success"], n), "crossed": div(r["crossed"], n),
            "cleared": div(r["cleared"], n), "falls": div(t, n),
            "falls_per_push": div(r["falls_after_push"], p), "term_low_frac": div(r["term_low"], t),
            "out_of_lane": div(r["out_of_lane"], n), "mean_progress": div(r["progress_sum"], n)}


def _f(v, n=3):
    return "-" if v is None else f"{v:.{n}f}"


def print_instrument(rec):
    """The placement, stability and validity lines under one block row; chance in brackets."""
    pl, st, gt = rec["placement"], rec["stability"], rec["gates"]
    ch = pl["chance"]
    if rec["lane_type"] == "stair":
        for dn in ("asc", "desc"):
            p = pl[dn]
            print(f"        place {dn:4s} td {p['touchdowns']:5d}  median d_edge {_f(p['median_d_edge'])} "
                  f"[{ch['median_d_edge']:.3f}]  P(d_edge<0.05) {_f(p['p_edge_lt05'])} [{ch['p_edge_lt05']:.3f}]  "
                  f"P(d_drop<0.028) {_f(p['p_drop_lt028'])} [{ch['p_drop_lt028']:.3f}]  stalls {p['stalls']}"
                  f"  perched {p.get('perched', '-')}"
                  f"  | excl stalls: median {_f(p['median_d_edge_excl_stalls'])} P(<0.05) "
                  f"{_f(p['p_edge_lt05_excl_stalls'])} P(drop) {_f(p['p_drop_lt028_excl_stalls'])}")
        ns = pl["nosing"]
        print(f"        nosing crossings {ns['crossings']}  p10 {_f(ns['p10'])} m  P(<0.02) {_f(ns['p_scrape_lt02'])}"
              f"   touchdowns {pl['touchdowns']} {pl['by_kind']}")
    bc = st["by_cause_per_100m"]
    # term_tilt / term_low are the legacy causes only when the legacy rule terminated (spot_feet.stability_summary)
    rule = (f"legacy tilt {_f(bc['legacy_tilt'], 2)} low {_f(bc['legacy_low'], 2)}"
            if st.get("termination", "legacy") == "legacy" else
            f"upright at term {_f(bc.get('upright_at_term'), 2)}, legacy rule true at term "
            f"{_f(bc.get('legacy_cond_at_term'), 2)}")
    line = (f"        stab path {st['path_m']:.0f} m  falls/100m {_f(st['falls_per_100m'], 2)} (tilt-held "
            f"{_f(bc['tilt_sustained'], 2)} base {_f(bc['body_base'], 2)} upper-leg {_f(bc['body_uleg'], 2)} | "
            f"{rule})  drift/m {_f(st['drift_per_m'])}"
            f"  slip {_f(st['slip_m_per_stance_s'])} m/stance-s  up/cos(slope) p05 {_f(st['uprel_p05'])}"
            + (f"  toward gravity/m {_f(st['drift_downhill_per_m'])}" if rec.get("slope") == "cross" else "")
            + (f"  off-surface {_f(st.get('offsurf_frac'))}" if rec.get("slope") else ""))
    if st["pushes"]:
        line += (f"  recovered {st['recovered']}/{st['pushes']} mean {_f(st['recovery_s_mean'], 2)} s "
                 f"p95 {_f(st['recovery_s_p95'], 2)} s")
    print(line)
    rc = st["reconcile"]
    print(f"        gates p95 |tau| hx {_f(gt['hx']['tau_p95'], 0)}/45 hy {_f(gt['hy']['tau_p95'], 0)}/45 "
          f"kn {_f(gt['kn']['tau_p95'], 0)}/115 Nm  p95 |qd| {_f(gt['hx']['qd_p95'], 1)}/{_f(gt['hy']['qd_p95'], 1)}/"
          f"{_f(gt['kn']['qd_p95'], 1)} rad/s   fired: legacy {rc['legacy_fired']} honest {rc['honest_fired']} "
          f"(legacy first {rc['legacy_first']}, honest first {rc['honest_first']})  knee end near terrain in "
          f"{st['knee']['fired_episodes']} eps  ray misses {st['ray_misses']}"
          + (f"  td recount {rec['td_recount']} ok={rec['td_recount_ok']}" if "td_recount" in rec else "")
          + (f"  npz rows {rec['npz_rows']} ok={rec['npz_ok']}" if "npz_rows" in rec else ""))


def _pool_instrument(recs):
    """Sum the raw instrument vectors of several block records (e.g. one checkpoint over score seeds)."""
    out = {}
    for r in recs:
        for key, v in r["instrument"].items():
            if isinstance(v, list):
                out[key] = [a + b for a, b in zip(out[key], v)] if key in out else list(v)
            else:
                out[key] = out.get(key, 0.0) + v
    return out


def score(ckpt, label, seed, k=2048, names=None, episodes=2, cap=1700, json_path="", recount=False,
          all_bands=False, repeat=False, device="cuda", drive_limits_are_forces=None, suite="e0",
          termination=None, instrument=True, td_every=64, cross_mode=None):
    import torch
    import threepp as tp
    from threepp.rl import load_policy
    from spot_steps_env import SpotStepsEnv, STAT_KEYS
    from spot_terrain_env import DT, SUBSTEPS
    import spot_feet as sf
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return None
    names = list(names or SUITES[suite])
    termination = termination or SUITE_TERMINATION[suite]
    if drive_limits_are_forces is None:
        drive_limits_are_forces = SUITE_DRIVE_LIMITS[suite]
    cross_mode = cross_mode or CROSS_MODE
    torch.manual_seed(seed)
    ac, norm, meta = load_policy(ckpt, device=device)
    pol = (lambda o: ac.act_mean(norm.norm(o))) if norm is not None else ac.act_mean
    lay = layout(names, k)
    slope_on = any(ALL_BLOCKS[n]["slope"] for n in names)
    if slope_on:
        import spot_slopes as ss
        assert SLOPE_MATERIAL == ss.GRIPPY, "score_e0.SLOPE_MATERIAL has drifted from spot_slopes.GRIPPY"
    # without a slope cell the env gets exactly the historical lane_types call, so every e0 number is unchanged
    lanes_kw = (dict(lane_family=lay["family"], slope_deg=lay["deg"], slope_material=lay["material"],
                     slope_cross_mode=cross_mode) if slope_on else dict(lane_types=lay["stairs"]))
    t0 = time.perf_counter()
    with torch.no_grad():
        env = SpotStepsEnv(num_envs=k, device=device, seed=seed, height_source="raycast", perceive=False,
                           risers=E0_RISERS, **lanes_kw, init_level=lay["level"],
                           freeze_level=True, single_band=not all_bands, hold_cmd=lay["hold"],
                           push_mode="once_on_tent", push_dv=lay["dv"], push_lanes=lay["dv"] > 0,
                           foot_mu=lay["mu"] if np.isfinite(lay["mu"]).any() else None,
                           payload_kg=lay["payload"] if (lay["payload"] != 0).any() else None,
                           block_id=lay["bid"], count_episodes=episodes,
                           drive_limits_are_forces=drive_limits_are_forces, termination=termination,
                           instrument=instrument,
                           scan_offset=lay["scan_dx"] if (lay["scan_dx"] != 0).any() else None)
        # What PhysX actually ran, read back off the articulation flag rather than trusted from the
        # argument: a result must never be attributed to the wrong plant.
        dlf_ran = bool(env.sim.robots[0].art.drive_limits_are_forces)
        torch.cuda.synchronize(); t1 = time.perf_counter()
        obs = env.reset()
        env.reset_stats()
        torch.cuda.synchronize(); t2 = time.perf_counter()
        hashes, steps = {}, 0
        bid = lay["bid"]
        ep_host = np.zeros(k, np.int64)
        tally = np.zeros((len(lay["names"]), 3), np.int64)          # episodes, terminations, timeouts
        recorder = sf.TouchdownRecorder(env, every=td_every) if (instrument and json_path) else None
        td_tally = np.zeros(len(lay["names"]), np.int64)            # touchdowns, recounted host-side
        st_h, tick_h, td_pend = np.zeros((k, 4), bool), np.zeros(k, np.int64), np.zeros(k, np.int64)
        for steps in range(1, cap + 1):
            obs, _, done, _, timeout = env.step(pol(obs))
            if recorder is not None:
                recorder.after_step()
            if recount:
                # An independent count from what step() returns, not from the counters' own path.
                d = done.cpu().numpy(); to = timeout.cpu().numpy()
                if instrument:
                    # touchdowns: the hysteresis rule re-run in numpy on this tick's tip clearances, with its
                    # own tick and episode bookkeeping from `done`
                    c = env.feet_clear.cpu().numpy()
                    tick_h += 1
                    new = np.where(st_h, c <= sf.TD_OFF, c < sf.TD_ON)
                    td_pend += (new & ~st_h & (tick_h > sf.TD_ARM_TICKS)[:, None]).sum(axis=1)
                    st_h = new
                i = np.nonzero(d)[0]
                ok = i[ep_host[i] < episodes]
                np.add.at(tally, (bid[ok], 0), 1)
                np.add.at(tally, (bid[ok], 1), (~to[ok]).astype(np.int64))
                np.add.at(tally, (bid[ok], 2), to[ok].astype(np.int64))
                np.add.at(td_tally, bid[ok], td_pend[ok])
                td_pend[i] = 0; st_h[i] = False; tick_h[i] = 0
                ep_host[i] += 1
            if steps in HASH_STEPS:
                blob = torch.cat([obs.reshape(-1), env.sim.root_position.reshape(-1)]).float().cpu().numpy()
                hashes[str(steps)] = hashlib.sha256(blob.tobytes()).hexdigest()[:16]
            if steps % 50 == 0 and bool((env.ep_index >= episodes).all()):
                break
        torch.cuda.synchronize(); t3 = time.perf_counter()
        stats = env.episode_stats()
        cols = recorder.columns(bid, env.ep_index.cpu().numpy(), episodes) if recorder is not None else None
    rows = {r["block"]: r for r in stats["rows"]}
    sha, dirty = _git()
    blocks_out, complete, recount_ok = [], True, True
    npz_ok = None if cols is None else True
    for b, name in enumerate(names):
        r = rows.get(b)
        lanes = int((bid == b).sum())
        spec = ALL_BLOCKS[name]
        rec = {"type": "block", "label": label, "seed": seed, "repeat": repeat, "block": name,
               "lane_type": "stair" if spec["stairs"] else "flat", "level": spec["level"],
               "riser": E0_RISERS[spec["level"]] if spec["stairs"] else 0.0, "vx": spec["vx"],
               "push_dv": spec["push_dv"], "foot_mu": spec["foot_mu"], "payload_kg": spec["payload_kg"],
               "scan_dx": spec["scan_dx"], "lanes": lanes, "E": episodes, "slope": spec["slope"],
               "slope_deg": spec["deg"], "material": None if spec["material"] is None else list(spec["material"]),
               "measure_only": spec["measure_only"]}
        rec.update({key: r[key] for key in STAT_KEYS})
        if "spawn" in r:
            rec["spawn"] = r["spawn"]
        rec["progress_sum"] = r["progress_sum"]
        rec["by_episode"] = r["by_episode"]
        rec["complete"] = r["episodes"] == lanes * episodes and r["timeouts"] + r["terminations"] == r["episodes"]
        complete &= rec["complete"]
        if recount:
            rec["recount"] = {"episodes": int(tally[b, 0]), "terminations": int(tally[b, 1]),
                              "timeouts": int(tally[b, 2])}
            rec["recount_ok"] = (rec["recount"]["episodes"] == r["episodes"]
                                 and rec["recount"]["terminations"] == r["terminations"]
                                 and rec["recount"]["timeouts"] == r["timeouts"])
            recount_ok &= rec["recount_ok"]
        if instrument:
            ins = r["instrument"]
            rec["instrument"] = ins
            rec["placement"] = sf.placement_summary(ins)
            rec["stability"] = sf.stability_summary(ins, r, DT, termination)
            rec["gates"] = sf.gates_summary(ins)
            n_td = int(round(ins["td"]))
            if recount:
                rec["td_recount"] = int(td_tally[b])
                rec["td_recount_ok"] = rec["td_recount"] == n_td
                recount_ok &= rec["td_recount_ok"]
            if cols is not None:
                rec["npz_rows"] = int((cols["block"] == b).sum())
                rec["npz_ok"] = rec["npz_rows"] == n_td
                npz_ok &= rec["npz_ok"]
        rec["rates"] = rates(rec)
        blocks_out.append(rec)
    head = {"type": "header", "label": label, "seed": seed, "repeat": repeat,
            "checkpoint": os.path.abspath(ckpt), "ckpt_sha256": _sha256(ckpt), "meta": meta,
            "envs": k, "E": episodes, "cap": cap, "blocks": names, "risers": list(E0_RISERS),
            "single_band": not all_bands, "height_source": "raycast", "perceive": False,
            "lanes_per_block": {n: int((bid == b).sum()) for b, n in enumerate(names)},
            "buffer_lanes": int((bid == len(names)).sum()),
            "git": sha, "git_dirty": dirty, "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
            "build_s": t1 - t0, "reset_s": t2 - t1, "steps_s": t3 - t2, "steps": steps,
            "hashes": hashes, "complete": complete, "recount_ok": recount_ok if recount else None,
            "drive_limits_are_forces": bool(drive_limits_are_forces), "drive_limits_ran": dlf_ran,
            # GpuSim's contact model: a direct-GPU PhysxWorld always runs TGS (the PhysxWorld tgs_pcm doc in
            # bind_physx.cpp), stepped SUBSTEPS times per control tick
            "solver": "TGS", "substep_dt": DT / SUBSTEPS, "substeps": SUBSTEPS, "control_dt": DT,
            "suite": suite, "termination": termination, "instrument": bool(instrument),
            "read_links": bool(env.sim.read_links),
            "scan_dx": {n: ALL_BLOCKS[n]["scan_dx"] for n in names if ALL_BLOCKS[n]["scan_dx"]}}
    if instrument:
        head.update({"touchdown": {"on": sf.TD_ON, "off": sf.TD_OFF, "arm_ticks": sf.TD_ARM_TICKS,
                                   "foot_r": sf.FOOT_R, "stall_dist": sf.STALL_DIST, "stall_stride": sf.STALL_STRIDE,
                                   "perch_rise": sf.PERCH_RISE, "slip_contact": sf.SLIP_CONTACT,
                                   "nosing": "every rising-nosing crossing between two touchdowns, stance included, "
                                             "clamped at 0"},
                     "honest": {"tilt_up": sf.TILT_UP, "tilt_ticks": sf.TILT_TICKS, "body_contact": sf.BODY_CONTACT},
                     "chance": sf.CHANCE, "npz_ok": npz_ok,
                     # the validity gates for the whole item (every scored block pooled, buffer lanes excluded)
                     "gates": sf.gates_summary(_pool_instrument(blocks_out)),
                     "touchdowns_npz": (os.path.splitext(os.path.abspath(json_path))[0] + "_touchdowns.npz"
                                        if cols is not None else None)})
    if slope_on:
        cells = [n for n in names if ALL_BLOCKS[n]["slope"]]
        by_name = {rec["block"]: rec for rec in blocks_out}
        flagged = [n for n in cells if (by_name[n].get("spawn") or {"flagged": True})["flagged"]]
        head["slopes"] = {
            "cells": {n: {"family": ALL_BLOCKS[n]["slope"], "deg": ALL_BLOCKS[n]["deg"],
                          "material": ALL_BLOCKS[n]["material"], "foot_mu": ALL_BLOCKS[n]["foot_mu"],
                          "measure_only": ALL_BLOCKS[n]["measure_only"]} for n in cells},
            "guard_lanes": lay["guard_lanes"], "cross_mode": cross_mode, "ramp_len_m": ss.RAMP_LEN,
            "spawn": "spot_slopes.spawn_pose (IK stance)", "spawn_u": ss.SPAWN_U, "spawn_body_tilt": ss.SPAWN_BODY_TILT,
            "spawn_clearance": ss.SPAWN_CLEARANCE, "settle_steps": env.settle_steps, "settle_holds": "spawn joints",
            "settle_up_frac": ss.SETTLE_UP_FRAC, "contact_tol": ss.CONTACT_TOL,
            "clearances": "perpendicular to the local plane on slope lanes (honest termination, touchdowns, slip)",
            "path": "along the surface on slope lanes"}
        head["spawn_gate"] = {"max_fail_frac": ss.SPAWN_FAIL_MAX, "flagged": flagged,
                              "flagged_le_20deg": [n for n in flagged if ALL_BLOCKS[n]["deg"] <= 20.0]}
        if instrument:
            off = [n for n in cells if (by_name[n]["stability"].get("offsurf_frac") or 0.0) > ss.OFF_SURFACE_MAX]
            head["surface_gate"] = {"max_offsurf_frac": ss.OFF_SURFACE_MAX, "flagged": off,
                                    "flagged_le_20deg": [n for n in off if ALL_BLOCKS[n]["deg"] <= 20.0],
                                    "offsurf_frac": {n: by_name[n]["stability"].get("offsurf_frac") for n in cells}}
    print(f"[e0] {label} seed {seed}: K={k}, {len(names)} blocks ({suite}, {termination} termination), build "
          f"{t1 - t0:.1f} s, reset {t2 - t1:.1f} s, {steps} steps {t3 - t2:.1f} s "
          f"({(t3 - t2) / max(steps, 1) * 1e3:.1f} ms/step)  complete={complete}"
          + (f"  recount_ok={recount_ok}" if recount else "") + (f"  npz_ok={npz_ok}" if npz_ok is not None else ""))
    for rec in blocks_out:
        rt = rec["rates"]
        fpp = rt["falls_per_push"]
        print(f"  {rec['block']:5s} {rec['lane_type']:5s} lanes {rec['lanes']:4d} eps {rec['episodes']:4d}  "
              f"success {_f(rt['success'])}  crossed {_f(rt['crossed'])}  "
              f"falls {_f(rt['falls'])}  {'low' if termination == 'legacy' else 'upright'} {rec['term_low']:3d}  "
              f"after-clear {rec['fell_after_clear']:3d}  out-of-lane {rec['out_of_lane']:3d}"
              + (f"  falls/push {fpp:.3f} ({rec['falls_after_push']}/{rec['pushes']})" if fpp is not None else ""))
        if rec.get("spawn"):
            sp = rec["spawn"]
            print(f"        spawn {rec['slope']} {rec['slope_deg']:.0f} deg{' (measurement only)' if rec['measure_only'] else ''}"
                  f"{' material ' + str(rec['material']) if rec['material'] else ''}: settle bad {sp['settle_bad']}/"
                  f"{sp['settle_checked']}  IK/gap bad {sp['unreachable']}/{sp['resets']}  min up/cos "
                  f"{_f(sp['min_up_over_cos'])}  max foot gap {_f(sp['max_foot_gap_m'], 4)} m"
                  + ("  !! FLAGGED: not a result" if sp["flagged"] else ""))
        if instrument:
            print_instrument(rec)
    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w") as f:
            f.write(json.dumps(head, default=str) + "\n")
            for rec in blocks_out:
                f.write(json.dumps(rec, default=str) + "\n")
        print(f"[e0] -> {json_path}")
        if cols is not None:
            sf.save_touchdowns(head["touchdowns_npz"], cols, env, lay["names"])
            print(f"[e0] -> {head['touchdowns_npz']} ({len(cols['env'])} touchdown rows)")
    return head, blocks_out


# ── aggregate (CPU) ───────────────────────────────────────────────────────────────────────────────
ARRAY_RE = re.compile(r"^(?P<cond>[A-Za-z0-9.]+)_s(?P<run>\d+):(?P<w>best|latest)$")
T975 = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23,
        12: 2.18, 15: 2.13, 20: 2.09, 24: 2.06, 30: 2.04}


# One-sided 95% upper bound on a Poisson count n: the exact chi-square limit tabulated to 10, beyond that the
# Wilson-Hilferty form, which matches the exact limit to 0.01 at n = 10.
POISSON_UP95 = (2.996, 4.744, 6.296, 7.754, 9.154, 10.513, 11.842, 13.148, 14.435, 15.705, 16.962)


def _poisson_up95(n):
    if n < len(POISSON_UP95):
        return POISSON_UP95[n]
    m = n + 1.0
    return m * (1.0 - 1.0 / (9.0 * m) + 1.645 / (3.0 * math.sqrt(m))) ** 3


def _t975(df):
    ks = [d for d in sorted(T975) if d <= df]
    return T975[ks[-1]] if ks else float("nan")


def arm_of(label):
    m = ARRAY_RE.match(label)
    if m:
        return "A" if m["w"] == "best" else "B"
    return label


def load(root):
    """{(label, seed, repeat, suite): {"header", "blocks": {name: rec}}} from every score_e0 JSONL under root. A JSONL
    from before suites existed is 'e0'. The suite is in the key because one label and seed can be scored by both
    suites into the same directory, and the showpiece cells share block names with E0 under another termination."""
    items = {}
    for path in sorted(glob.glob(os.path.join(root, "**", "*.jsonl"), recursive=True)):
        head, blocks = None, {}
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("type") == "header":
                    head = rec
                elif rec.get("type") == "block":
                    blocks[rec["block"]] = rec
        if head is not None:
            key = (head["label"], head["seed"], bool(head.get("repeat")), head.get("suite", "e0"))
            if key in items:
                print(f"!! {key} is scored twice: {items[key]['path']} and {path}; the later file is kept")
            items[key] = {"header": head, "blocks": blocks, "path": path}
    return items


def _metric(item, block, what):
    """(rate, n) for one item: what = 'success' | 'fpp' | 'falls'."""
    r = item["blocks"].get(block)
    if r is None:
        return None, 0
    if what == "fpp":
        return (r["falls_after_push"] / r["pushes"], r["pushes"]) if r["pushes"] else (None, 0)
    n = r["episodes"]
    if not n:
        return None, 0
    return (r["success"] if what == "success" else r["terminations"]) / n, n


def _arm_stat(items, block, what):
    """Mean over checkpoints of the per-checkpoint rate, between-checkpoint SD, binomial SE of the mean."""
    vals = [(p, n) for p, n in (_metric(it, block, what) for it in items) if p is not None]
    if not vals:
        return None
    ps = [p for p, _ in vals]
    m = sum(ps) / len(ps)
    sd = statistics.stdev(ps) if len(ps) > 1 else float("nan")
    se = math.sqrt(sum(p * (1 - p) / n for p, n in vals)) / len(vals)
    return m, sd, se, len(vals), sum(n for _, n in vals)


def aggregate(root, report_seed=2, select_seed=1):
    items = load(root)
    if not items:
        print(f"no score_e0 JSONL under {root}"); return
    main = {k: v for k, v in items.items() if not k[2] and k[3] == "e0"}     # sections 1-10 are the E0 tables
    at = lambda seed: {lbl: it for (lbl, s, _, _), it in main.items() if s == seed}
    rep = at(report_seed)
    arms = {}
    for lbl, it in rep.items():
        arms.setdefault(arm_of(lbl), []).append(it)
    blocks = [b for b in ALL_BLOCKS if any(b in it["blocks"] for it in rep.values())]
    blocks.sort(key=lambda b: b != LOW_RUNG)        # the low rung first, then BLOCKS order (stable)
    arm_order = [a for a in ("A", "B") if a in arms] + sorted(a for a in arms if a not in ("A", "B"))

    def _low(arm, b):
        """(share, n) of this block's terminations that were base_above < 0.18 with the robot still
        upright — the base came down rather than the robot tipping over. Legacy-termination items only: under
        the honest rule term_low is every upright termination, not this cause."""
        leg = [it for it in arms[arm] if b in it["blocks"] and it["header"].get("termination", "legacy") == "legacy"]
        t = sum(it["blocks"][b]["terminations"] for it in leg)
        lo = sum(it["blocks"][b]["term_low"] for it in leg)
        return (lo / t, t) if t else (None, 0)

    print(f"# E0-lite aggregate: {len(items)} scored items under {root}; report seed {report_seed}, "
          f"selection seed {select_seed}")
    print("  arm A = array in-run best (25), B = array latest (25); mean over checkpoints, sd between "
          "checkpoints, se = binomial SE of that mean")

    print("\n## 1. per-block success and falls per push (report seed)")
    print(f"  {LOW_RUNG} (the 0.10 m rung) is printed first on purpose: every other stair cell sits at "
          "level >= 5, where a policy that cannot climb the riser at all and one that climbs it and then "
          "falls both read success 0.000.")
    print("  low = share of the block's terminations that were base_above < 0.18 with the robot upright "
          "(the base came down). PAY7 is a termination-REFERENCE cell as much as a payload cell — read "
          "routing rule R3 off its low column, not off its success.")
    print(f"{'block':6s} {'arm':28s} {'n':>3s} {'success':>8s} {'sd':>6s} {'se':>6s} {'low':>5s}  "
          f"{'falls/push':>10s} {'sd':>6s} {'se':>6s} {'pushes':>7s}")
    for b in blocks:
        for a in arm_order:
            s = _arm_stat(arms[a], b, "success")
            if s is None:
                continue
            f = _arm_stat(arms[a], b, "fpp")
            ftxt = f"{f[0]:10.4f} {f[1]:6.4f} {f[2]:6.4f} {f[4]:7d}" if f else f"{'-':>10s}"
            lo, _ = _low(a, b)
            ltxt = f"{lo:5.2f}" if lo is not None else f"{'-':>5s}"
            print(f"{b:6s} {a[:28]:28s} {s[3]:3d} {s[0]:8.3f} {s[1]:6.3f} {s[2]:6.3f} {ltxt}  {ftxt}")

    print("\n## 2. paired best vs latest (latest - best, report seed)")
    tags = {}
    for lbl, it in rep.items():
        m = ARRAY_RE.match(lbl)
        if m:
            tags.setdefault(f"{m['cond']}_s{m['run']}", {})[m["w"]] = it
    pairs = {t: v for t, v in tags.items() if "best" in v and "latest" in v}
    paired = {}
    for block, what in (("S20", "success"), ("FP4", "fpp"), ("P2", "fpp"), ("P3", "fpp"),
                        ("S23", "success")):
        d = []
        for v in pairs.values():
            pb, _ = _metric(v["best"], block, what); pl, _ = _metric(v["latest"], block, what)
            if pb is not None and pl is not None:
                d.append(pl - pb)
        if len(d) >= 2:
            m, sd = sum(d) / len(d), statistics.stdev(d)
            se = sd / math.sqrt(len(d)); tq = _t975(len(d) - 1)
            paired[(block, what)] = (m, m - tq * se, m + tq * se)
            print(f"  {block} {what:7s}: {len(d)} pairs  mean {m:+.4f}  sd {sd:.4f}  t {m / se if se else float('nan'):+.2f}"
                  f"  95% CI [{m - tq * se:+.4f}, {m + tq * se:+.4f}]")
        else:
            print(f"  {block} {what}: {len(d)} pairs (need >= 2)")
    s20 = paired.get(("S20", "success"))
    weights = "latest" if (s20 is not None and s20[1] >= -0.025) else "best"
    print(f"  R1 suggestion: '{weights}' weights (latest if the lower 95% bound on S20 success >= -0.025 = "
          f"MEI/2; the paired falls/push rows are the other half of the rule, read them)")

    print(f"\n## 3. positive control (G3): push2 vs the 25-run median (arm A), falls per push, read on "
          f"the FLAT shove cells {' and '.join(GATE_FPP)}")
    print("   G3 is off the tent because push2 cannot climb the cells the on-tent ladder scores: it "
          "makes 0.93-1.21 m on a 0.20 m tent where shipped makes 9.5-19.0 m, so only 1-5 of its lanes "
          "per on-tent block ever reach their trigger x (P2 read 0 falls of 3-4 pushes) and the ladder "
          "cannot measure the one checkpoint it exists to detect. On flat ground every lane is shoved.")
    print(f"   The on-tent cells {' and '.join(FRONTIER_FPP)} stay the REPORTED frontier and are printed "
          "below the gate rows, but nothing is decided on them.")
    print("   se is binomial only — the comparator is an order statistic over 25 runs and carries no "
          "between-run term, so z reads high; take the difference and its CI with it")
    for block in tuple(GATE_FPP) + tuple(FRONTIER_FPP):
        gated = block in GATE_FPP
        a = [(p, n, lbl) for lbl, it in rep.items() if arm_of(lbl) == "A"
             for p, n in [_metric(it, block, "fpp")] if p is not None]
        if not a:
            print(f"  {block}: no arm-A data"); continue
        a.sort()
        pm, nm, lm = a[len(a) // 2]
        for ctl in ("ctrl:spot_steps_push2", "ctrl:spot_steps_push2_latest", "ctrl:spot_steps"):
            it = rep.get(ctl)
            if it is None:
                continue
            pc, nc = _metric(it, block, "fpp")
            pooled = [_metric(v, block, "fpp") for (lbl, s, _, _), v in main.items() if lbl == ctl]
            pooled = [(p, n) for p, n in pooled if p is not None]
            pp = sum(p * n for p, n in pooled) / max(sum(n for _, n in pooled), 1)
            npool = sum(n for _, n in pooled)
            if pc is None:
                continue
            se = math.sqrt(pm * (1 - pm) / nm + pc * (1 - pc) / nc)
            z = (pm - pc) / se if se > 0 else float("inf")
            d = pm - pc
            print(f"  {'G3' if gated else '  '} {block:4s} {ctl:30s} {pc:.4f} (n {nc})  vs median "
                  f"{pm:.4f} (n {nm}, {lm})  d {d:+.4f} [{d - 1.96 * se:+.4f}, {d + 1.96 * se:+.4f}]  "
                  f"z {z:+.2f} " + (f"{'> 3: PASS' if z > 3 else '<= 3: FAIL'}" if gated
                                    else "(frontier, not gated)")
                  + f"   pooled over {len(pooled)} seeds {pp:.4f} (n {npool})")

    rungs = (LOW_RUNG, "S20", "S23", "S26", "S30")
    print("\n## 4. per-rung success (report seed): 0.10 / 0.20 / 0.23 / 0.26 / 0.30 m at vx 0.8. The "
          "0.10 m rung makes the rest of the row readable: 0.000 there is a policy that does not climb, "
          "0.000 at 0.20 m with the low rung high is one that climbs and falls.")
    print(f"  {'arm':30s} " + "  ".join(f"{b:^11s}" for b in rungs))
    for a in arm_order:
        cells = [_arm_stat(arms[a], b, "success") for b in rungs]
        print(f"  {a[:30]:30s} " + "  ".join(f"{c[0]:.3f}+-{c[2]:.3f}" if c else "    -      " for c in cells))

    hi, ref = R4_PAIR
    print(f"\n## 5. contact and load. Routing rule R4 (friction) is read from {hi} — foot mu 0.3 at the "
          f"0.23 m rung — against {ref}, NOT from the level-5 trio: after the in-lane latch fix the "
          "nominal S20, the MU5 bridge and MU3 all read success 1.000 at K=256, so there is no headroom "
          "there. The trio stays as the material/bridge control (MU5 - S20 = the explicit foot material, "
          "MU3 - MU5 = friction alone). PAY7H is payload's headroom cell, and PAY7's low column in "
          "section 1 is the termination reference R3 is read with.")
    cols = ("S20", "MU5", "MU3", ref, hi, "PAY7", "PAY7H")
    print(f"  {'arm':30s} " + "  ".join(f"{b:>7s}" for b in cols) + f"    {hi} - {ref}")
    for a in arm_order:
        cells = [_arm_stat(arms[a], b, "success") for b in cols]
        s_hi, s_ref = _arm_stat(arms[a], hi, "success"), _arm_stat(arms[a], ref, "success")
        d = (s_hi[0] - s_ref[0]) if (s_hi and s_ref) else None
        print(f"  {a[:30]:30s} " + "  ".join(f"{c[0]:7.3f}" if c else f"{'-':>7s}" for c in cells)
              + (f"    {d:+.3f} -> friction DR {'IN' if d <= -0.10 else 'out of'} E1"
                 if d is not None else ""))

    print("\n## 6. termination cause: share of falls with the base down but upright (base_above < 0.18)")
    for b in blocks:
        parts = []
        for a in arm_order:
            lo, t = _low(a, b)
            if lo is not None:
                parts.append(f"{a[:22]} {lo:.2f} ({t})")
        if parts:
            print(f"  {b:5s} " + "   ".join(parts))

    print(f"\n## 7. E1 parent: analytic run with the MEDIAN composite z(S23 success) - z(P2 falls/push), "
          f"selection seed {select_seed}, '{weights}' weights")
    sel = at(select_seed)
    pool = [it for lbl, it in sel.items() if arm_of(lbl) in ("A", "B")]
    s23 = [p for p, _ in (_metric(it, "S23", "success") for it in pool) if p is not None]
    fpp = [p for p, _ in (_metric(it, "P2", "fpp") for it in pool) if p is not None]
    if len(s23) >= 2 and len(fpp) >= 2:
        z = lambda v, xs: (v - statistics.mean(xs)) / (statistics.stdev(xs) or 1.0)
        cand = []
        for lbl, it in sel.items():
            m = ARRAY_RE.match(lbl)
            if m and m["cond"] == "analytic" and m["w"] == weights:
                a, _ = _metric(it, "S23", "success"); f, _ = _metric(it, "P2", "fpp")
                if a is not None and f is not None:
                    cand.append((z(a, s23) - z(f, fpp), lbl, a, f))
        cand.sort()
        for c in cand:
            print(f"  {c[1]:22s} composite {c[0]:+.3f}   S23 success {c[2]:.3f}   P2 falls/push {c[3]:.4f}")
        if cand:
            print(f"  -> parent (median of {len(cand)}): {cand[len(cand) // 2][1]}  (z over {len(pool)} array checkpoints)")
    else:
        print("  not enough array checkpoints at the selection seed")

    print("\n## 8. same-seed determinism (exact counters + obs/root_pos hashes)")
    found = False
    for (lbl, s, r, su), it in items.items():
        if not r:
            continue
        base = items.get((lbl, s, False, su))
        if base is None:
            continue
        found = True
        diffs = []
        for b, rec in it["blocks"].items():
            o = base["blocks"].get(b, {})
            for key in list(rec.get("by_episode", [{}])[0].keys()) + ["progress_sum"]:
                if key in rec and rec.get(key) != o.get(key):
                    diffs.append(f"{b}.{key} {o.get(key)} vs {rec.get(key)}")
            if rec.get("by_episode") != o.get("by_episode"):
                diffs.append(f"{b}.by_episode")
        hb, hr = base["header"].get("hashes"), it["header"].get("hashes")
        verdict = "MATCH" if not diffs and hb == hr else "MISMATCH -> PhysX-GPU nondeterministic, use empirical SDs"
        print(f"  {lbl} seed {s}: {verdict}; hashes {'equal' if hb == hr else f'{hb} vs {hr}'}"
              + (f"; first diffs: {diffs[:4]}" if diffs else ""))
    if not found:
        print("  no repeat item found")

    print("\n## 9. score-seed noise (controls scored at >= 3 seeds): SD across seeds vs mean binomial SE")
    by_lbl = {}
    for (lbl, s, _, _), it in main.items():
        by_lbl.setdefault(lbl, {})[s] = it
    for lbl, seeds in sorted(by_lbl.items()):
        if len(seeds) < 3:
            continue
        parts = []
        for block, what in (("S20", "success"), ("S23", "success"), ("P2", "fpp"), ("P3", "fpp")):
            v = [_metric(it, block, what) for it in seeds.values()]
            v = [(p, n) for p, n in v if p is not None]
            if len(v) >= 2:
                ps = [p for p, _ in v]
                se = statistics.mean(math.sqrt(p * (1 - p) / n) for p, n in v)
                parts.append(f"{block}:{what} {statistics.mean(ps):.4f} sd {statistics.stdev(ps):.4f} (se {se:.4f})")
        print(f"  {lbl} ({len(seeds)} seeds): " + "; ".join(parts))

    print("\n## 10. reconciliation and steering side outputs")
    for path in sorted(glob.glob(os.path.join(root, "**", "legacy1.jsonl"), recursive=True)):
        for line in open(path):
            rec = json.loads(line)
            print(f"  legacy1 {os.path.basename(os.path.dirname(path))}: track {rec['track']:.4f} "
                  f"(target 1.6272, d {rec['track'] - 1.6272:+.4f})  fell {rec['fell']:.6f} "
                  f"(target 0.001314, d {rec['fell'] - 0.001314:+.6f})")
    for path in sorted(glob.glob(os.path.join(root, "**", "steer.json"), recursive=True)):
        rec = json.load(open(path))
        print(f"  steer {os.path.basename(os.path.dirname(path))}: worst ratio {rec['worst']:.3f} "
              f"({'pass' if rec['pass'] else 'FAIL'})")
    inst, heads = {}, {}
    for (lbl, s, rep_, su), it in sorted(items.items(), key=lambda kv: kv[0]):
        if rep_:
            continue
        tag = lbl if su == "e0" else f"{lbl} [{su}]"      # one suite's cells are never pooled with another's
        heads[(tag, s)] = it["header"]
        for b, rec in it["blocks"].items():
            if "instrument" in rec:
                inst.setdefault(tag, {}).setdefault(b, []).append(rec)
    if inst:
        import spot_feet as sf
        print("\n## 11. foot placement on the real 0.30 m treads, pooled over each checkpoint's score seeds. Chance "
              "(touchdowns uniform across the tread) in brackets. S20X15 is S20 with the scan shifted +0.15 m: a "
              "policy that places by the scan moves its touchdowns off the tread centre there; a rhythm does not.")
        print(f"  {'checkpoint':28s} {'block':7s} {'dir':4s} {'seeds':>5s} {'td':>6s} {'med d_edge':>15s} "
              f"{'P(d_edge<0.05)':>15s} {'P(d_drop<0.028)':>15s} {'stalls':>6s} {'perch':>5s} {'P<.05 no stall':>14s} "
              f"{'nose p10':>8s}")
        for lbl in sorted(inst):
            for b in [x for x in ALL_BLOCKS if x in inst[lbl]]:
                recs = inst[lbl][b]
                if recs[0]["lane_type"] != "stair":
                    continue
                pl = sf.placement_summary(_pool_instrument(recs))
                for dn in ("asc", "desc", "all"):
                    p = pl[dn]
                    print(f"  {lbl[:28]:28s} {b:7s} {dn:4s} {len(recs):5d} {p['touchdowns']:6d} "
                          f"{_f(p['median_d_edge']):>7s} [0.075] {_f(p['p_edge_lt05']):>7s} [0.333] "
                          f"{_f(p['p_drop_lt028']):>7s} [{sf.CHANCE['p_drop_lt028']:.3f}] {p['stalls']:6d} "
                          f"{('-' if p.get('perched') is None else str(p['perched'])):>5s} "
                          f"{_f(p['p_edge_lt05_excl_stalls']):>14s} "
                          + (f"{_f(pl['nosing']['p10']):>8s}" if dn == "all" else ""))
        print("\n## 12. stability and validity, pooled over score seeds: falls per 100 m walked by honest cause "
              "(tilt held 0.2 s / base / upper leg on the terrain), the legacy causes beside them (legacy-termination "
              "items only: '-' when the pool ran another rule, whose term_low is not base_above), lateral drift "
              "per metre, foot slip per stance-second, up_z / cos(local slope) p05, p95 |commanded torque| against "
              "45/45/115 Nm and p95 |joint speed|.")
        print(f"  {'checkpoint':28s} {'block':7s} {'term':6s} {'path m':>8s} {'f/100m':>7s} {'tilt':>6s} {'base':>6s} "
              f"{'uleg':>6s} {'L.tilt':>6s} {'L.low':>6s} {'drift/m':>7s} {'slip':>6s} {'up p05':>6s} "
              f"{'tau hx/hy/kn':>13s} {'qd hx/hy/kn':>15s}")
        for lbl in sorted(inst):
            for b in [x for x in ALL_BLOCKS if x in inst[lbl]]:
                recs = inst[lbl][b]
                pooled = _pool_instrument(recs)
                cnt = {key: sum(r[key] for r in recs) for key in ("terminations", "term_tilt", "term_low", "pushes")}
                term = {heads[(lbl, r["seed"])].get("termination", "legacy") for r in recs}
                dt = heads[(lbl, recs[0]["seed"])].get("control_dt", 0.02)
                rule = next(iter(term)) if len(term) == 1 else "mixed"
                st, gt = sf.stability_summary(pooled, cnt, dt, rule), sf.gates_summary(pooled)
                bc = st["by_cause_per_100m"]
                print(f"  {lbl[:28]:28s} {b:7s} {'/'.join(sorted(term)):6s} {st['path_m']:8.0f} "
                      f"{_f(st['falls_per_100m'], 2):>7s} {_f(bc['tilt_sustained'], 2):>6s} {_f(bc['body_base'], 2):>6s} "
                      f"{_f(bc['body_uleg'], 2):>6s} {_f(bc['legacy_tilt'], 2):>6s} {_f(bc['legacy_low'], 2):>6s} "
                      f"{_f(st['drift_per_m']):>7s} {_f(st['slip_m_per_stance_s']):>6s} {_f(st['uprel_p05']):>6s} "
                      f"{_f(gt['hx']['tau_p95'], 0):>4s}/{_f(gt['hy']['tau_p95'], 0)}/{_f(gt['kn']['tau_p95'], 0):<4s} "
                      f"{_f(gt['hx']['qd_p95'], 1)}/{_f(gt['hy']['qd_p95'], 1)}/{_f(gt['kn']['qd_p95'], 1)}")
        slope_rows = [(lbl, b) for lbl in sorted(inst) for b in ALL_BLOCKS if ALL_BLOCKS[b]["slope"] and b in inst[lbl]]
        if slope_rows:
            print("\n## 13. slopes at vx 0.8 held, pooled over score seeds: honest falls per 100 m walked ALONG THE "
                  "SURFACE with the one-sided 95% Poisson upper bound (so a zero reads as a bound, not a claim), lateral "
                  "drift per metre and, on cross-slopes, drift toward gravity per metre, foot slip per stance-second, "
                  "up_z / cos(local slope) p05. spawn = settle health failed / checked at the full reset, and IK or gap "
                  "failures / resets; a flagged cell is NOT a result. SC25 is above the 0-20 deg envelope (measurement "
                  "only); the M cells are the explicit 1.0 'min' material on the ramps and the feet.")
            print(f"  {'checkpoint':28s} {'block':6s} {'fam':5s} {'deg':>3s} {'eps':>4s} {'path m':>7s} {'falls':>5s} "
                  f"{'f/100m':>6s} {'up95':>6s} {'tilt':>5s} {'base':>5s} {'uleg':>5s} {'drift/m':>7s} {'grav/m':>7s} "
                  f"{'slip':>6s} {'up p05':>6s} {'offsurf':>7s} {'settle':>7s} {'IK':>7s} flag")
            for lbl, b in slope_rows:
                recs = inst[lbl][b]
                pooled = _pool_instrument(recs)
                cnt = {key: sum(r[key] for r in recs) for key in ("terminations", "term_tilt", "term_low", "pushes",
                                                                  "episodes")}
                hd = heads[(lbl, recs[0]["seed"])]
                term = {heads[(lbl, r["seed"])].get("termination", "legacy") for r in recs}
                st = sf.stability_summary(pooled, cnt, hd.get("control_dt", 0.02),
                                          next(iter(term)) if len(term) == 1 else "mixed")
                bc = st["by_cause_per_100m"]
                sp = [r.get("spawn") or {"flagged": True} for r in recs]
                ssum = lambda key: sum(s.get(key, 0) for s in sp)
                up95 = (100.0 * _poisson_up95(cnt["terminations"]) / st["path_m"]) if st["path_m"] > 0 else None
                spec = ALL_BLOCKS[b]
                print(f"  {lbl[:28]:28s} {b:6s} {spec['slope'][:5]:5s} {spec['deg']:3.0f} {cnt['episodes']:4d} "
                      f"{st['path_m']:7.0f} {cnt['terminations']:5d} {_f(st['falls_per_100m'], 2):>6s} {_f(up95, 2):>6s} "
                      f"{_f(bc['tilt_sustained'], 2):>5s} {_f(bc['body_base'], 2):>5s} {_f(bc['body_uleg'], 2):>5s} "
                      f"{_f(st['drift_per_m']):>7s} {_f(st.get('drift_downhill_per_m')) if spec['slope'] == 'cross' else '-':>7s} "
                      f"{_f(st['slip_m_per_stance_s']):>6s} {_f(st['uprel_p05']):>6s} {_f(st.get('offsurf_frac')):>7s} "
                      f"{ssum('settle_bad'):>3d}/{ssum('settle_checked'):<3d} {ssum('unreachable'):>3d}/{ssum('resets'):<3d} "
                      + ("FLAGGED spawn" if any(s["flagged"] for s in sp) else "ok")
                      + (" LEFT SURFACE" if (st.get("offsurf_frac") or 0.0) > 0.02 else "")
                      + f" [{hd.get('slopes', {}).get('cross_mode', '?')}]"
                      + (" (above envelope)" if spec["measure_only"] else ""))
    incomplete = [f"{k[0]} s{k[1]}" for k, it in items.items() if not it["header"].get("complete")]
    if incomplete:
        print(f"\n!! {len(incomplete)} items did not reach lanes x E episodes in every block: {incomplete[:8]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default="")
    ap.add_argument("--label", default="", help="item label (default: the checkpoint's file stem). The "
                    "aggregate reads '<cond>_s<N>:best|latest' as array arms A/B, anything else as a control")
    ap.add_argument("--seed", type=int, default=1, help="score seed (S_sel = 1, S_test = 2)")
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--blocks", default="", help=f"comma list, subset of {','.join(ALL_BLOCKS)} (default: the "
                    "suite's cells)")
    ap.add_argument("--suite", choices=sorted(SUITES), default="e0",
                    help="e0: every E0 cell, legacy termination (the E0 job list, numbers unchanged); showpiece: "
                    f"{','.join(SUITES['showpiece'])}, honest termination")
    ap.add_argument("--termination", choices=("legacy", "honest"), default=None,
                    help="override the suite's termination rule (e0: legacy, showpiece: honest)")
    ap.add_argument("--no-instrument", dest="instrument", action="store_false",
                    help="skip the placement / stability / validity instrument and the link reads it needs")
    ap.add_argument("--td-every", dest="td_every", type=int, default=64,
                    help="control ticks between flushes of the per-touchdown rows to the host")
    ap.add_argument("--cross-mode", dest="cross_mode", choices=("corrugate", "offset"), default=CROSS_MODE,
                    help="cross-slope lanes: offset (default) = each cross block one continuous plane pulling toward -y; "
                    "corrugate = alternating tilt, a V gutter at every lane edge (caps drift toward gravity near 1.5 m)")
    ap.add_argument("--episodes", type=int, default=2, help="episodes counted per lane (E)")
    ap.add_argument("--cap", type=int, default=1700, help="step cap (the run stops earlier once every "
                    "lane has ended E episodes)")
    ap.add_argument("--json", default="", help="write the header + block lines here (overwritten)")
    ap.add_argument("--recount", action="store_true", help="also count episodes host-side from the "
                    "per-step done/timeout tensors and check the counters against it (slow; smoke only)")
    ap.add_argument("--all-bands", dest="all_bands", action="store_true",
                    help="build every band per stair lane (the training ladder) instead of one")
    ap.add_argument("--repeat", action="store_true", help="mark this item as the same-seed repeat")
    ap.add_argument("--drive-limits-are-forces", dest="drive_limits_are_forces",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="enforce the joint effort caps as torques (45/45/115 N·m); --no-drive-limits-are-forces "
                    "reads them as impulses (max_force/dt), the plant every existing checkpoint was trained on. "
                    "Default by suite: off for e0 (reproduces E0-lite), on for showpiece (the agreed plant)")
    ap.add_argument("--aggregate", default="", help="print the E0 tables for every JSONL under this root")
    ap.add_argument("--report-seed", dest="report_seed", type=int, default=2)
    ap.add_argument("--select-seed", dest="select_seed", type=int, default=1)
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.aggregate, args.report_seed, args.select_seed); return
    if not args.checkpoint:
        ap.error("give a checkpoint or --aggregate")
    names = [b for b in args.blocks.split(",") if b] or list(SUITES[args.suite])
    unknown = [b for b in names if b not in ALL_BLOCKS]
    if unknown:
        ap.error(f"unknown blocks {unknown}")
    label = args.label or os.path.splitext(os.path.basename(args.checkpoint))[0]
    out = score(args.checkpoint, label, args.seed, k=args.envs, names=names, episodes=args.episodes,
                cap=args.cap, json_path=args.json, recount=args.recount, all_bands=args.all_bands,
                repeat=args.repeat, drive_limits_are_forces=args.drive_limits_are_forces, suite=args.suite,
                termination=args.termination, instrument=args.instrument, td_every=args.td_every,
                cross_mode=args.cross_mode)
    if out is None:
        sys.exit(1)
    head, _ = out
    ok = head["complete"] and head["recount_ok"] is not False and head.get("npz_ok") is not False
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
