"""E3's two numbers: the repeatability row and the seed spread, from fresh processes.

    python netpen_e3_runs.py --out <dir>                       # 10 + 10 processes, then both figures
    python netpen_e3_runs.py --out <dir> --repeat 2 --seeds 3  # a small one, to prove the wiring
    python netpen_e3_runs.py --out <dir> --figs                # figures from runs already on disk
    python netpen_e3_runs.py --out <dir> --dry                 # print the commands and stop

1. Repeatability. `--repeat` fresh processes at the same seed. Every process
   writes a manifest in sensor_audit.py's format; a row that carries one hash
   across all of them replays to the bit, and a row that does not is named with
   the count. Judge `traj`, `sonar`, `net`, `rope`: the rendered rows (`rov.rgb`)
   share the GPU with whatever else is running and can differ for that alone.

2. Seed spread. `--seeds` processes, seeds 0..N-1, one process each. The seed is
   the only input that changes: it seeds the sonar's own RangeNoiseModel and the
   ranging noise on the derived standoff. Reported per seed and over the set:

   - lateral deviation: the cross-track distance from each logged position to the
     SEED-0 PATH (nearest point on that polyline) -- the fan-out, with along-track
     lag divided out.
   - same-time separation: |p_k(t) - p_0(t)|, which carries the lag as well.

   Figures: netpen_e3_paths.png (top view, every path over the pen outline) and
   netpen_e3_spread.png (deviation against time).

Every process runs on the simulation clock; nothing here reads a wall clock
except the printed durations.
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENE = os.path.join(HERE, "warp_netpen.py")
JUDGE = ("traj", "sonar", "net", "rope", "fish", "rov")   # streams the GPU's other tenants cannot move
RENDERED = ("rov.rgb",)


def run_one(py, seconds, seed, out, extra, dry=False, reuse=False):
    cmd = [py, SCENE, "--e3", str(seconds), "--e3-seed", str(seed), "--e3-out", out] + list(extra)
    if dry:
        print("  " + subprocess.list2cmdline(cmd))
        return None
    if reuse and os.path.exists(out) and os.path.exists(out[:-5] + ".npz"):
        print(f"  reuse {os.path.basename(out)}")
        return json.load(open(out))
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if p.returncode != 0 or not os.path.exists(out):
        print(p.stdout[-2000:])
        print(p.stderr[-2000:])
        raise SystemExit(f"run failed (seed {seed}, exit {p.returncode})")
    for line in p.stdout.splitlines():
        if line.startswith("e3:"):
            print("  " + line)
    print(f"  {os.path.basename(out)} in {time.perf_counter() - t0:.0f} s wall")
    return json.load(open(out))


def repeatability(mans, names):
    """One hash per row, or the split that says which stream moved first."""
    print(f"\nrepeatability: {len(mans)} fresh processes, same seed")
    verdict = {}
    keys = [k for k in mans[0]["rows"]]
    for k in keys:
        rs = [m["rows"].get(k) for m in mans]
        hs = [r if isinstance(r, str) else f"{r['fnv']}:{r['frames']}" for r in rs]
        uniq = sorted(set(hs))
        verdict[k] = len(uniq)
        tag = "OK  " if len(uniq) == 1 else ("DIFF" if k in JUDGE else "diff")
        note = "" if len(uniq) == 1 else "  <- " + ", ".join(
            f"{names[i]}={hs[i][:16]}" for i in range(len(hs)))
        if len(uniq) > 1 and k in RENDERED:
            note += "   (rendered: a shared GPU can move this on its own)"
        print(f"  {tag} {k:9s} {len(uniq)} distinct of {len(hs)}  {uniq[0][:16]}{note}")
    bad = [k for k in JUDGE if verdict.get(k, 1) > 1]
    print("  verdict: " + ("every judged row is one hash" if not bad
                           else "SPLIT in " + ", ".join(bad)))
    return verdict


def cross_track(ref, p):
    """Shortest distance from each point of `p` to the polyline `ref` (N,2 arrays)."""
    a, b = ref[:-1], ref[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(1), 1e-12)
    d = np.empty(len(p))
    for i, q in enumerate(p):
        t = np.clip(((q - a) * ab).sum(1) / L2, 0.0, 1.0)
        d[i] = np.sqrt((((a + t[:, None] * ab) - q) ** 2).sum(1).min())
    return d


def spread(paths, out_dir, pen_r=7.0):
    """The seed spread, its two figures, and the numbers for the table."""
    logs, seeds = [], []
    for s, npz in paths:
        z = np.load(npz, allow_pickle=True)
        logs.append(z["log"])
        seeds.append(s)
    cols = list(np.load(paths[0][1], allow_pickle=True)["cols"])
    ix, iz, it = cols.index("x"), cols.index("z"), cols.index("t")
    n = min(len(L) for L in logs)
    t = logs[0][:n, it]
    ref = logs[0][:n, [ix, iz]]
    rows = []
    for s, L in zip(seeds, logs):
        p = L[:n, [ix, iz]]
        lat = cross_track(ref, p)
        sep = np.linalg.norm(p - ref, axis=1)
        rows.append((s, lat, sep))
    print(f"\nseed spread: {len(rows)} seeds, {n} frames = {t[-1]:.1f} s")
    print(f"  {'seed':>4}  {'max lat':>8}  {'rms lat':>8}  {'max sep':>8}  {'rms sep':>8}")
    for s, lat, sep in rows:
        print(f"  {s:>4}  {lat.max():8.3f}  {np.sqrt((lat ** 2).mean()):8.3f}  "
              f"{sep.max():8.3f}  {np.sqrt((sep ** 2).mean()):8.3f}")
    nz = [r for r in rows if r[0] != seeds[0]]
    if nz:
        allat = np.concatenate([r[1] for r in nz])
        allsep = np.concatenate([r[2] for r in nz])
        print(f"  over the set (seed 0 excluded): max lateral {allat.max():.3f} m, "
              f"RMS lateral {np.sqrt((allat ** 2).mean()):.3f} m; "
              f"max separation {allsep.max():.3f} m, RMS {np.sqrt((allsep ** 2).mean()):.3f} m")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    th = np.linspace(0, 2 * np.pi, 361)
    ax.plot(pen_r * np.cos(th), pen_r * np.sin(th), color="0.35", lw=1.4, label="pen wall (rest)")
    ax.plot((pen_r - 1.5) * np.cos(th), (pen_r - 1.5) * np.sin(th), color="0.75", lw=0.8, ls="--",
            label="1.5 m standoff")
    cmap = plt.get_cmap("viridis")
    for i, (s, L) in enumerate(zip(seeds, logs)):
        ax.plot(L[:n, ix], L[:n, iz], lw=1.0, color=cmap(i / max(len(seeds) - 1, 1)), label=f"seed {s}")
    ax.plot(ref[0, 0], ref[0, 1], "o", ms=5, color="k")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title(f"E3: {len(seeds)} sonar seeds, {t[-1]:.0f} s of closed-loop inspection")
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    f1 = os.path.join(out_dir, "netpen_e3_paths.png")
    fig.savefig(f1, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(7.0, 5.4), sharex=True)
    for i, (s, lat, sep) in enumerate(rows):
        c = cmap(i / max(len(rows) - 1, 1))
        ax[0].plot(t, lat, lw=0.9, color=c, label=f"seed {s}")
        ax[1].plot(t, sep, lw=0.9, color=c)
    if nz:
        rms = np.sqrt(np.mean(np.stack([r[1] for r in nz]) ** 2, axis=0))
        ax[0].plot(t, rms, lw=2.0, color="k", label="RMS over seeds")
    ax[0].set_ylabel("lateral deviation [m]")
    ax[1].set_ylabel("same-time separation [m]")
    ax[1].set_xlabel("simulation time [s]")
    ax[0].set_title("E3: deviation from the seed-0 path")
    ax[0].legend(fontsize=7, ncol=3)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    f2 = os.path.join(out_dir, "netpen_e3_spread.png")
    fig.savefig(f2, dpi=150)
    plt.close(fig)
    print(f"  figures: {f1}\n            {f2}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "aaa_caps", "netpen", "e3"))
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--repeat", type=int, default=10, help="fresh processes at the same seed")
    ap.add_argument("--seeds", type=int, default=10, help="sensor-noise seeds, one process each")
    ap.add_argument("--seed", type=int, default=0, help="the seed the repeatability set uses")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--size", default="960x540", help="smaller than the film: E3 judges the loop, not the look")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--reuse", action="store_true", help="skip a run whose json + npz are already there")
    ap.add_argument("--figs", action="store_true", help="no runs: figures and numbers from what is on disk")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    extra = ["--size", args.size]
    print(f"E3 runs -> {out_dir}\n  {args.repeat} x seed {args.seed} (repeatability), "
          f"{args.seeds} seeds (spread), {args.seconds:.0f} s each")

    mans, names = [], []
    if not args.figs:
        for i in range(args.repeat):
            p = os.path.join(out_dir, f"rep{i}.json")
            m = run_one(args.python, args.seconds, args.seed, p, extra, args.dry, args.reuse)
            if m:
                mans.append(m)
                names.append(f"rep{i}")
        for s in range(args.seeds):
            p = os.path.join(out_dir, f"seed{s}.json")
            run_one(args.python, args.seconds, s, p, extra, args.dry, args.reuse)
    if args.dry:
        return
    if not mans:                                        # --figs: read the repeatability set off disk
        for i in range(args.repeat):
            p = os.path.join(out_dir, f"rep{i}.json")
            if os.path.exists(p):
                mans.append(json.load(open(p)))
                names.append(f"rep{i}")
    if len(mans) > 1:
        repeatability(mans, names)
    paths = [(s, os.path.join(out_dir, f"seed{s}.npz")) for s in range(args.seeds)
             if os.path.exists(os.path.join(out_dir, f"seed{s}.npz"))]
    if len(paths) > 1:
        spread(paths, out_dir)
    elif paths:
        print("\nseed spread: one seed on disk; nothing to spread")


if __name__ == "__main__":
    main()
