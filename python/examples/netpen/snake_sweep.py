"""Validation sweeps: the snake model (snake_model.py) against the published TRENDS of
Kelasidi et al. (2015), Robotics and Biomimetics 2:8, https://pmc.ncbi.nlm.nih.gov/articles/PMC4679098/

The paper tabulates no absolute speeds, so the check is trend-level, as the paper's own claims are:

  T1 alpha  (omega 120 deg/s, delta 30): "the average forward velocity is increased by increasing the
            parameter alpha"; our verdict: speed rises 10 -> 30 deg and the argmax lies in [25, 40]
            (and we say whether it declines at 35-40).
  T2 omega  (alpha 30, delta 30): "the increase of the forward velocity is almost linear ... until the
            value of omega = 90 deg/s"; verdict: linear fit over 60-90 deg/s, R^2 >= 0.95, slope > 0.
  T3 delta  (alpha 30, omega 120): "there exists a value of the gait parameter delta which gives the
            maximum forward velocity"; verdict: the argmax is strictly inside (0, 90) deg.
  T4 power  "by increasing the parameter alpha the average power consumption is increased", "by
            increasing omega the power consumption is increased", "increase of the value of delta
            results in a decrease of the average power consumption"; verdict: Spearman signs.

Both gaits (lateral undulation, eel-like) x both coefficient sets (complex, control). 30 s per run;
speed = |p_CM(30 s) - p_CM(10 s)| / 20 s (the paper's start/stop definition), mean power over the
same window (the paper's absolute joint power; net power recorded too). No coefficient is tuned.

    python snake_sweep.py                # sweeps + figure + table   (4 worker processes)
    python snake_sweep.py --look         # the 9 s look mp4 + strip (headless Vulkan; snake_look.py)
    python snake_sweep.py --report-only  # re-plot from the JSONs
"""
import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.dirname(_HERE)
_PY = os.path.dirname(_EX)
for _p in (_HERE, _EX, _PY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

OUT = os.path.join(_HERE, "out", "sweep")
T_RUN, T0, T1 = 30.0, 10.0, 30.0
ALPHAS = [10, 15, 20, 25, 30, 35, 40]
OMEGAS = [60, 75, 90, 105, 120, 135, 150]
DELTAS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
REF = dict(alpha=30, omega=120, delta=30)
PATTERNS = ("lateral", "eel")
COEFS = ("complex", "control")
SPACING = 25.0                     # m between the snakes that share one PhysX world


def run_specs():
    """Unique runs (the reference gait is shared by the three sweeps)."""
    specs = {}
    for coef in COEFS:
        for pat in PATTERNS:
            pts = [(a, REF["omega"], REF["delta"]) for a in ALPHAS]
            pts += [(REF["alpha"], w, REF["delta"]) for w in OMEGAS]
            pts += [(REF["alpha"], REF["omega"], d) for d in DELTAS]
            for a, w, d in pts:
                specs[(coef, pat, a, w, d)] = dict(coef=coef, pattern=pat, alpha=a, omega=w, delta=d)
    return list(specs.values())


def run_batch(batch):
    """One process, one PhysX world, every snake of the batch side by side (independent islands)."""
    import gc
    import snake_model as sm
    dt = 1.0 / 240.0
    world = sm.make_world(dt)
    snakes = []
    side = int(math.ceil(math.sqrt(len(batch))))
    for j, spec in enumerate(batch):
        o = (SPACING * (j % side), 0.0, SPACING * (j // side))
        S = sm.Snake(world, sm.SnakeParams(coeffs=spec["coef"]), origin=o, heading=0.0)
        S.set_gait(spec["pattern"], math.radians(spec["alpha"]), math.radians(spec["omega"]),
                   math.radians(spec["delta"]))
        snakes.append(S)
    nsteps = int(round(T_RUN / dt)); i0 = int(round(T0 / dt))
    rec = [dict(e0=None, n0=None, c0=None, peak=0.0, yaw=[], tilt=0.0, vy=0.0) for _ in snakes]
    t_wall = time.time()
    for s in range(nsteps):
        if s == i0:
            for S, r in zip(snakes, rec):
                r["c0"] = S.com(); r["e0"] = S.energy_abs; r["n0"] = S.energy_net
                S.peak_torque = 0.0
        world.step(dt)
        if s >= i0 and s % 4 == 0:
            for S, r in zip(snakes, rec):
                y = S.link_yaws()
                r["yaw"].append(float(y[-1] - np.mean(np.unwrap(y))))
                if s % 48 == 0:
                    r["tilt"] = max(r["tilt"], sm.roll_pitch(S))
                    r["vy"] = max(r["vy"], abs(S.com()[1]))
    rows = []
    for spec, S, r in zip(batch, snakes, rec):
        c1 = S.com()
        disp = c1 - r["c0"]
        sp = float(np.linalg.norm(disp[[0, 2]])) / (T1 - T0)
        fwd = float(disp @ np.array([math.cos(S.mean_heading()), 0.0, -math.sin(S.mean_heading())]))
        L = S.n * S.p.link_length
        yaw = np.array(r["yaw"])
        rows.append(dict(spec, speed=sp, speed_bl=sp / L, head_first=bool(fwd > 0),
                         power_abs=(S.energy_abs - r["e0"]) / (T1 - T0),
                         power_net=(S.energy_net - r["n0"]) / (T1 - T0),
                         peak_torque=float(S.peak_torque), head_yaw_rms_deg=float(np.degrees(np.sqrt(np.mean(yaw ** 2)))),
                         max_tilt_deg=float(r["tilt"]), max_vertical_m=float(r["vy"]),
                         finite=bool(np.isfinite(c1).all())))
    for S in snakes:
        S.remove()
    del snakes, world
    gc.collect()
    return rows, time.time() - t_wall


# ---------------------------------------------------------------------- verdicts
def _spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _series(rows, coef, pat, key):
    fixed = {k: v for k, v in REF.items() if k != key}
    sel = [r for r in rows if r["coef"] == coef and r["pattern"] == pat and all(r[k] == v for k, v in fixed.items())]
    sel.sort(key=lambda r: r[key])
    return sel


def verdicts(rows, coef):
    out = {}
    for pat in PATTERNS:
        sa = _series(rows, coef, pat, "alpha"); so = _series(rows, coef, pat, "omega"); sd = _series(rows, coef, pat, "delta")
        a = np.array([r["alpha"] for r in sa]); va = np.array([r["speed"] for r in sa]); pa = np.array([r["power_abs"] for r in sa])
        w = np.array([r["omega"] for r in so]); vw = np.array([r["speed"] for r in so]); pw = np.array([r["power_abs"] for r in so])
        d = np.array([r["delta"] for r in sd]); vd = np.array([r["speed"] for r in sd]); pd = np.array([r["power_abs"] for r in sd])
        up = a <= 30
        rho_up = _spearman(a[up], va[up])
        amax = float(a[int(np.argmax(va))])
        declines = bool(va[-1] < va.max() - 1e-9 and amax < 40)
        t1 = bool(rho_up > 0.8 and va[a == 30][0] > va[a == 10][0] and 25 <= amax <= 40)
        lo = w <= 90
        coef_fit = np.polyfit(w[lo], vw[lo], 1)
        pred = np.polyval(coef_fit, w[lo])
        ss_res = float(((vw[lo] - pred) ** 2).sum()); ss_tot = float(((vw[lo] - vw[lo].mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        t2 = bool(r2 >= 0.95 and coef_fit[0] > 0)
        dmax = float(d[int(np.argmax(vd))])
        t3 = bool(0 < dmax < 90)
        ra, rw, rd = _spearman(a, pa), _spearman(w, pw), _spearman(d, pd)
        t4 = bool(ra > 0 and rw > 0 and rd < 0)
        out[pat] = dict(
            T1=dict(passed=t1, spearman_10_30=rho_up, argmax_alpha_deg=amax, declines_35_40=declines,
                    speeds=va.tolist()),
            T2=dict(passed=t2, r2_60_90=r2, slope_ms_per_degs=float(coef_fit[0]), speeds=vw.tolist()),
            T3=dict(passed=t3, argmax_delta_deg=dmax, speeds=vd.tolist()),
            T4=dict(passed=t4, spearman_alpha=ra, spearman_omega=rw, spearman_delta=rd))
    return out


def _diag(name, v, pat, coef):
    """One-line diagnosis for a FAIL (what the numbers show; no tuning)."""
    if name == "T1":
        return (f"speed argmax at alpha {v['argmax_alpha_deg']:.0f} deg, Spearman(10-30) {v['spearman_10_30']:+.2f}: "
                + ("monotone rise through 40 deg (no decline in this range)" if v["argmax_alpha_deg"] >= 40 else "early peak"))
    if name == "T2":
        return f"R^2 over 60-90 deg/s = {v['r2_60_90']:.3f}, slope {v['slope_ms_per_degs'] * 1000:.3f} mm/s per deg/s"
    if name == "T3":
        return f"speed argmax at delta {v['argmax_delta_deg']:.0f} deg (a boundary)"
    if name == "T4":
        return (f"Spearman power vs alpha {v['spearman_alpha']:+.2f}, omega {v['spearman_omega']:+.2f}, "
                f"delta {v['spearman_delta']:+.2f} (want +, +, -)")
    return ""


# ---------------------------------------------------------------------- outputs
def write_outputs(rows, elapsed=None):
    os.makedirs(OUT, exist_ok=True)
    allv = {}
    for coef in COEFS:
        v = verdicts(rows, coef)
        allv[coef] = v
        mine = [r for r in rows if r["coef"] == coef]
        ref = {pat: _series(rows, coef, pat, "alpha")[ALPHAS.index(30)] for pat in PATTERNS}
        doc = dict(source="https://pmc.ncbi.nlm.nih.gov/articles/PMC4679098/", coefficient_set=coef,
                   run_seconds=T_RUN, window=[T0, T1],
                   speed_definition="|p_CM(30 s) - p_CM(10 s)| / 20 s (horizontal)",
                   power_definition="mean of sum_i |u_i phi_dot_i| over the window (paper eq. 28, absolute); power_net = signed",
                   verdicts=v, reference_gait=ref, rows=mine)
        with open(os.path.join(OUT, f"sweep_{coef}.json"), "w") as f:
            json.dump(doc, f, indent=1)
    lines = []
    lines.append("# Snake model vs Kelasidi et al. (2015) trends\n")
    lines.append("Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC4679098/ . 9 links x 0.18 m, 0.8 kg/link, "
                 "PD kp 20 / kd 5 (+ the paper's feedforward), PhysX 240 Hz. 30 s runs, speed and mean |power| over 10-30 s.\n")
    lines.append("## Verdicts\n")
    lines.append("| trend | coef set | gait | verdict | numbers |")
    lines.append("|---|---|---|---|---|")
    for coef in COEFS:
        for pat in PATTERNS:
            vv = allv[coef][pat]
            for name in ("T1", "T2", "T3", "T4"):
                x = vv[name]
                if name == "T1":
                    num = (f"argmax {x['argmax_alpha_deg']:.0f} deg, Spearman(10-30) {x['spearman_10_30']:+.2f}, "
                           f"declines at 35-40: {'yes' if x['declines_35_40'] else 'no'}")
                elif name == "T2":
                    num = f"R^2(60-90) {x['r2_60_90']:.3f}, slope {x['slope_ms_per_degs'] * 1000:.3f} mm/s per deg/s"
                elif name == "T3":
                    num = f"argmax {x['argmax_delta_deg']:.0f} deg"
                else:
                    num = f"Spearman alpha {x['spearman_alpha']:+.2f}, omega {x['spearman_omega']:+.2f}, delta {x['spearman_delta']:+.2f}"
                verdict = "PASS" if x["passed"] else "FAIL: " + _diag(name, x, pat, coef)
                lines.append(f"| {name} | {coef} | {pat} | {verdict} | {num} |")
    lines.append("\n## The paper's reference gait (alpha 30, omega 120 deg/s, delta 30)\n")
    lines.append("| coef set | gait | speed m/s | bl/s | mean abs power W | net power W | peak torque N m | head yaw RMS deg |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for coef in COEFS:
        for pat in PATTERNS:
            r = _series(rows, coef, pat, "alpha")[ALPHAS.index(30)]
            lines.append(f"| {coef} | {pat} | {r['speed']:.4f} | {r['speed_bl']:.4f} | {r['power_abs']:.3f} | "
                         f"{r['power_net']:.3f} | {r['peak_torque']:.2f} | {r['head_yaw_rms_deg']:.2f} |")
    lines.append("\n## All runs\n")
    lines.append("| coef | gait | alpha deg | omega deg/s | delta deg | speed m/s | bl/s | P abs W | P net W | peak N m | head yaw RMS deg | head-first |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: (r["coef"], r["pattern"], r["alpha"], r["omega"], r["delta"])):
        lines.append(f"| {r['coef']} | {r['pattern']} | {r['alpha']} | {r['omega']} | {r['delta']} | {r['speed']:.4f} | "
                     f"{r['speed_bl']:.4f} | {r['power_abs']:.3f} | {r['power_net']:.3f} | {r['peak_torque']:.2f} | "
                     f"{r['head_yaw_rms_deg']:.2f} | {r['head_first']} |")
    with open(os.path.join(OUT, "validation.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    plot(rows, allv)
    for coef in COEFS:
        for pat in PATTERNS:
            vv = allv[coef][pat]
            print(f"{coef:8s} {pat:8s} " + "  ".join(f"{k} {'PASS' if vv[k]['passed'] else 'FAIL'}" for k in ("T1", "T2", "T3", "T4")))
    return allv


def plot(rows, allv):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid, surf = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    col = {"lateral": "#2a78d6", "eel": "#eb6834"}
    ls = {"complex": "-", "control": "--"}
    mk = {"complex": "o", "control": "s"}
    fig, axs = plt.subplots(2, 3, figsize=(15, 9.2), facecolor=surf)
    keys = [("alpha", "amplitude alpha (deg)", "omega 120 deg/s, delta 30 deg"),
            ("omega", "frequency omega (deg/s)", "alpha 30 deg, delta 30 deg"),
            ("delta", "phase shift delta (deg)", "alpha 30 deg, omega 120 deg/s")]
    claims_v = ["paper: speed increases with alpha",
                "paper: ~linear in omega up to 90 deg/s",
                "paper: an interior delta maximises speed"]
    claims_p = ["paper: power increases with alpha",
                "paper: power increases with omega",
                "paper: power decreases with delta"]
    tnames = ["T1", "T2", "T3"]
    for j, (key, xl, fixed) in enumerate(keys):
        for row, (ylab, fld, claims) in enumerate([("mean forward speed (m/s)", "speed", claims_v),
                                                    ("mean |joint power| (W)", "power_abs", claims_p)]):
            ax = axs[row, j]
            ax.set_facecolor(surf)
            for coef in COEFS:
                for pat in PATTERNS:
                    s = _series(rows, coef, pat, key)
                    ax.plot([r[key] for r in s], [r[fld] for r in s], ls[coef], color=col[pat], lw=2,
                            marker=mk[coef], ms=5, mfc=col[pat] if coef == "complex" else surf, mew=1.5,
                            label=f"{'lateral undulation' if pat == 'lateral' else 'eel-like'}, {coef} coeffs")
            ax.set_xlabel(xl, color=ink2); ax.set_ylabel(ylab, color=ink2)
            ax.grid(True, color=grid, lw=0.8); ax.tick_params(colors=ink2)
            for sp_ in ax.spines.values():
                sp_.set_color(grid)
            tn = tnames[j] if row == 0 else "T4"
            fails = [f"{coef}/{pat}" for coef in COEFS for pat in PATTERNS if not allv[coef][pat][tn]["passed"]]
            vt = "PASS for both gaits x both coefficient sets" if not fails else "FAIL: " + ", ".join(fails)
            ax.set_title(f"{claims[j]}\n({fixed})\n{tn} {vt}", fontsize=9.5, color=ink, loc="left")
            if key == "omega" and row == 0:
                ax.axvspan(60, 90, color=grid, alpha=0.5, lw=0)
    hl, lb = axs[0, 0].get_legend_handles_labels()
    fig.legend(hl, lb, loc="lower center", ncol=4, frameon=False, fontsize=9, labelcolor=ink2)
    fig.suptitle("Underwater snake robot (Mamba, 9 links) in PhysX with the Kelasidi et al. (2015) fluid model: "
                 "the paper's trends", color=ink, fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(os.path.join(OUT, "validation.png"), dpi=110, facecolor=surf)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--look", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.look:
        import snake_look
        snake_look.main([])
        return
    if args.report_only:
        rows = []
        for coef in COEFS:
            rows += json.load(open(os.path.join(OUT, f"sweep_{coef}.json")))["rows"]
        write_outputs(rows)
        return
    specs = run_specs()
    nw = max(1, min(4, args.workers))
    batches = [specs[i::nw] for i in range(nw)]
    t0 = time.time()
    print(f"{len(specs)} runs in {nw} worker processes ...", flush=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=nw, maxtasksperchild=1) as pool:     # a fresh process (one PhysX world) per batch
        res = pool.map(run_batch, batches, chunksize=1)
    rows = [r for rr, _ in res for r in rr]
    print(f"done in {time.time() - t0:.0f} s (batches {', '.join(f'{e:.0f}' for _, e in res)} s)", flush=True)
    bad = [r for r in rows if not r["finite"]]
    if bad:
        print(f"WARNING: {len(bad)} non-finite runs")
    write_outputs(rows)


if __name__ == "__main__":
    main()
