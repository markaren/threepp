"""Phase 0 offline spike: own BoxEye + OpticLobe vs the flyvis reference, T4/T5 selectivity.

    py -3.14 python/examples/flyeye/spike_offline.py --stimuli C:/dev/_flyeye/stimuli \
        --reference C:/dev/_flyeye/reference --own C:/dev/_flyeye/own

For every movie <m> under --stimuli (PNG frames rendered by render_edges.py) and every
reference file <reference>/<m>_<dtype>.npz written by tools/flyvis_reference.py, runs the
own path (no flyvis) on CPU float32, CUDA float32 and CPU float64, and prints max abs and
relative differences of (a) the luminance movie, (b) box_eye, (c) the post-fade-in state
and (d) all (T, 45669) responses. Then measures direction selectivity of the central
T4a-d (ON edges) and T5a-d (OFF edges) and draws figures/phase0_direction_selectivity.png.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message=".*[Ss]parse.*")

from flyeye.lattice import HexLattice, load_png_luminance  # noqa: E402
from flyeye.optic_lobe import OpticLobe  # noqa: E402

DIRS = ("right", "left", "up", "down")
OPP = dict(right="left", left="right", up="down", down="up")
T4 = ("T4a", "T4b", "T4c", "T4d")
T5 = ("T5a", "T5b", "T5c", "T5d")


def run_own(lat, lobe, movie, dt):
    """movie (T, H, W) luminance -> box (T, 721), initial state (N,), responses (T, N)."""
    movie = movie.to(device=lobe.device, dtype=lobe.dtype)
    box = lat.box_eye(movie)
    v0 = lobe.fade_in(box[0], dt).clone()
    out = torch.empty(len(box), lobe.n_nodes, dtype=lobe.dtype, device=lobe.device)
    for t in range(len(box)):
        out[t] = lobe.step(box[t], dt)
    return box, v0, out


def diffs(own, ref):
    ref = torch.as_tensor(ref)
    d = (own.cpu().to(ref.dtype) - ref).abs().max().item()
    return d, d / max(ref.abs().max().item(), 1e-30)


def compare(args, lat):
    movies = sorted(p.name for p in args.stimuli.iterdir() if p.is_dir())
    configs = [("cpu", torch.float32), ("cuda", torch.float32), ("cpu", torch.float64)]
    if not torch.cuda.is_available():
        configs.remove(("cuda", torch.float32))
    lobes = {c: OpticLobe(device=c[0], dtype=c[1]) for c in configs}
    report, traces = {}, {}
    for name in movies:
        files = sorted((args.stimuli / name).glob("*.png"))
        for dev, dtype in configs:
            tag = f"{dev}_{str(dtype).split('.')[-1]}"
            ref_path = args.reference / f"{name}_{str(dtype).split('.')[-1]}.npz"
            if not ref_path.exists():
                print(f"{name} {tag}: no reference {ref_path.name}, skipped")
                continue
            ref = np.load(ref_path)
            dt = float(ref["dt"])
            movie = torch.stack([load_png_luminance(f, dtype) for f in files])
            t0 = time.perf_counter()
            box, v0, resp = run_own(lat, lobes[(dev, dtype)], movie, dt)
            if dev == "cuda":
                torch.cuda.synchronize()
            el = time.perf_counter() - t0
            r = dict(
                movie=diffs(movie, ref["movie"]),
                boxeye=diffs(box, ref["boxeye"]),
                initial_state=diffs(v0, ref["initial_state"]),
                responses=diffs(resp, ref["responses"]),
                max_abs_ref=float(np.abs(ref["responses"]).max()),
                seconds=el,
            )
            report[f"{name}/{tag}"] = r
            print(
                f"{name:9s} {tag:13s} movie {r['movie'][0]:.1e}  box {r['boxeye'][0]:.2e}  "
                f"init {r['initial_state'][0]:.2e}  resp abs {r['responses'][0]:.2e} rel {r['responses'][1]:.2e}  "
                f"(|v_ref|max {r['max_abs_ref']:.2f}, {el:.1f} s)"
            )
            if dev == "cpu" and dtype == torch.float32:
                lobe = lobes[(dev, dtype)]
                types = T4 if name.startswith("on") else T5
                traces[name] = dict(
                    own={t: lobe.central(t, resp).cpu().numpy() for t in types},
                    ref={t: lobe.central(t, torch.from_numpy(ref["responses"])).numpy() for t in types},
                    dt=dt,
                )
                args.own.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(args.own / f"{name}_cpu_float32.npz", boxeye=box.numpy(),
                                    initial_state=v0.numpy(), responses=resp.numpy(), dt=dt)
    return report, traces


def selectivity(traces, preroll):
    """Peak above the pre-roll baseline (state at the last pre-roll frame), per direction."""
    rows = []
    for pol, types in (("on", T4), ("off", T5)):
        for t in types:
            peaks = {}
            for src in ("own", "ref"):
                peaks[src] = {}
                for d in DIRS:
                    tr = traces[f"{pol}_{d}"][src][t]
                    peaks[src][d] = float(tr[preroll:].max() - tr[preroll - 1])
            pk = peaks["own"]
            pref = max(DIRS, key=lambda d: pk[d])
            p, n = pk[pref], pk[OPP[pref]]
            dsi = (max(p, 0) - max(n, 0)) / (max(p, 0) + max(n, 0) + 1e-12)
            pr = peaks["ref"]
            ref_pref = max(DIRS, key=lambda d: pr[d])
            rows.append(dict(cell_type=t, polarity=pol, preferred_direction=pref, pref_peak=p, null_peak=n,
                             dsi=dsi, peaks=pk, ref_preferred=ref_pref,
                             ref_dsi=(max(pr[pref], 0) - max(pr[OPP[pref]], 0))
                             / (max(pr[pref], 0) + max(pr[OPP[pref]], 0) + 1e-12)))
    return rows


def figure(traces, rows, preroll, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(right="#2a78d6", left="#eb6834", up="#1baf7a", down="#eda100")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharex=True, constrained_layout=True)
    for ax, row in zip(axes.ravel(), rows):
        t, pol = row["cell_type"], row["polarity"]
        for d in DIRS:
            tr = traces[f"{pol}_{d}"]
            time_s = np.arange(len(tr["own"][t])) * tr["dt"]
            ax.plot(time_s, tr["own"][t], color=colors[d], lw=2, label=f"{d} (own)")
            ax.plot(time_s, tr["ref"][t], color="#222222", lw=1, ls=(0, (3, 3)), alpha=0.8)
        ax.axvline((preroll - 0.5) * traces[f"{pol}_right"]["dt"], color="#999999", lw=0.8)
        ax.set_title(f"{t} ({pol.upper()} edges): prefers {row['preferred_direction']}, DSI {row['dsi']:.2f}", fontsize=10)
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("time since first frame (s)")
    axes[0, 0].set_ylabel("central column v (a.u.)")
    axes[1, 0].set_ylabel("central column v (a.u.)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#222222", lw=1, ls=(0, (3, 3))))
    labels.append("flyvis reference (dashed)")
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("Phase 0: threepp-rendered edges (13 columns/s, dt 1/100) through the own optic-lobe runtime; "
                 "grey line = end of pre-roll", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    print("figure", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stimuli", type=Path, default=Path(r"C:\dev\_flyeye\stimuli"))
    ap.add_argument("--reference", type=Path, default=Path(r"C:\dev\_flyeye\reference"))
    ap.add_argument("--own", type=Path, default=Path(r"C:\dev\_flyeye\own"))
    ap.add_argument("--figure", type=Path, default=Path(__file__).parent / "figures" / "phase0_direction_selectivity.png")
    args = ap.parse_args(argv)

    torch.set_num_threads(6)
    lat = HexLattice()
    report, traces = compare(args, lat)
    preroll = int(json.loads((args.stimuli / "meta.json").read_text())["preroll_frames"])

    for tag in ("cpu_float32", "cuda_float32", "cpu_float64"):
        rs = [v for k, v in report.items() if k.endswith(tag)]
        if rs:
            print(f"ALL {tag:13s} box abs {max(r['boxeye'][0] for r in rs):.2e}  "
                  f"init abs {max(r['initial_state'][0] for r in rs):.2e}  "
                  f"resp abs {max(r['responses'][0] for r in rs):.2e}  rel {max(r['responses'][1] for r in rs):.2e}  "
                  f"({len(rs)} movies)")

    rows = selectivity(traces, preroll)
    print(f"{'type':5s} {'pref':6s} {'pref_pk':>8s} {'null_pk':>8s} {'DSI':>6s} {'refDSI':>6s}  peaks right/left/up/down")
    for r in rows:
        pk = " ".join(f"{r['peaks'][d]:+.3f}" for d in DIRS)
        print(f"{r['cell_type']:5s} {r['preferred_direction']:6s} {r['pref_peak']:8.3f} {r['null_peak']:8.3f} "
              f"{r['dsi']:6.2f} {r['ref_dsi']:6.2f}  {pk}  (ref prefers {r['ref_preferred']})")
    figure(traces, rows, preroll, args.figure)
    (args.own / "spike_report.json").write_text(json.dumps(dict(match=report, selectivity=rows), indent=1))


if __name__ == "__main__":
    main()
