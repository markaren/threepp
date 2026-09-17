"""Phase 1 gate: the online eye reproduces the Phase 0 figure, and what TAA does to it.

    py -3.14 python/examples/flyeye/online_gate.py            # a + b + flush sweep + figure
    py -3.14 python/examples/flyeye/online_gate.py --figure-only

a. frames-identical: the Phase 0 PNGs uploaded as BGRA uint8 CUDA tensors, through
   eye.receptors_from_bgra and OpticLobe on CUDA, against the own offline responses
   (C:/dev/_flyeye/own/<movie>_cpu_float32.npz). Same rule as Phase 0: max abs response
   difference relative to max|v| at most 1e-4 in float32.
b. live: the same 8 movies re-rendered with render_edges.Stage (same scene, calibration
   from stimuli/meta.json) through an add_view secondary view + FrameTensors + EyeView +
   OpticLobe on CUDA, at flush 16 (the Phase 0 setting) and at 3 and 1. Per movie: live
   colour vs the PNGs and vs the primary read in the same run (pixel diffs by distance to
   the model edge), receptors and responses vs offline, the edge position in the live
   colour vs the model, central T4/T5 traces. Pass = preferred directions identical and
   DSI within 0.02 of Phase 0 at flush 16.
   Control, before the view exists: the primary alone at flush 16 against the PNGs.

Writes C:/dev/_flyeye/online/{report.json,traces.npz}, figures/phase1_online_vs_offline.png
(Phase 0 layout, offline solid, live flush 16 dashed) and figures/phase1_taa_flush.png (live
minus offline central traces at flush 16, 3, 1).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore", message=".*[Ss]parse.*")

from flyeye.eye import EyeView, bgra_luma_u8, receptors_from_bgra  # noqa: E402
from flyeye.lattice import HexLattice, luma_u8  # noqa: E402
from flyeye.optic_lobe import OpticLobe  # noqa: E402
from flyeye.spike_offline import DIRS, T4, T5, selectivity  # noqa: E402

TYPES = T4 + T5
MOVIES = [f"{p}_{d}" for p in ("on", "off") for d in DIRS]


def load_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def rgb_to_bgra_cuda(rgb):
    a = np.full(rgb.shape[:2] + (1,), 255, np.uint8)
    return torch.from_numpy(np.ascontiguousarray(np.concatenate([rgb[..., ::-1], a], -1))).cuda()


def central_traces(lobe, resp):
    return {t: lobe.central(t, resp).cpu().numpy() for t in TYPES}


# -- a. frames identical ------------------------------------------------------------------
def frames_identical(args, lat, lobe, offline):
    rows = {}
    for name in MOVIES:
        files = sorted((args.stimuli / name).glob("*.png"))
        ref = offline[name]
        dt = ref["dt"]
        luma_equal = True
        rec = []
        for f in files:
            rgb = load_rgb(f)
            bgra = rgb_to_bgra_cuda(rgb)
            if f is files[0] or f is files[len(files) // 2]:
                luma_equal &= torch.equal(bgra_luma_u8(bgra).cpu(), luma_u8(torch.from_numpy(rgb.copy())))
            rec.append(receptors_from_bgra(bgra, lat))
        box = torch.stack(rec)
        v0 = lobe.fade_in(box[0], dt).clone()
        d_resp = torch.zeros((), device="cuda")
        for t in range(len(box)):
            d_resp = torch.maximum(d_resp, (lobe.step(box[t], dt) - ref["responses"][t]).abs().max())
        vmax = ref["responses"].abs().max().item()
        rows[name] = dict(
            luma_u8_equal=bool(luma_equal),
            receptors_max_abs=(box - ref["boxeye"]).abs().max().item(),
            initial_state_max_abs=(v0 - ref["initial_state"]).abs().max().item(),
            responses_max_abs=d_resp.item(),
            responses_rel=d_resp.item() / vmax,
        )
        r = rows[name]
        print(f"a {name:9s} luma==PIL {r['luma_u8_equal']}  receptors {r['receptors_max_abs']:.2e}  "
              f"init {r['initial_state_max_abs']:.2e}  resp {r['responses_max_abs']:.2e} rel {r['responses_rel']:.2e}")
    return rows


# -- b. live ------------------------------------------------------------------------------
def axis_distance(direction, size, e):
    """Signed distance (px) of each pixel centre along the motion axis from the model edge,
    positive = inside the advancing region's far side (not yet covered)."""
    c = np.arange(size) + 0.5
    p = c if direction in ("right", "down") else size - c
    return p - e


def diff_by_distance(a, b, direction, e, hist):
    """Pixel diffs of two (H, W, 3) frames, collapsed across the motion axis; bins by the
    distance of the differing lines to the model edge. Returns (max abs, n pixels)."""
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    if not d.any():
        return 0, 0
    line = d.max(axis=0) if direction in ("right", "left") else d.max(axis=1)
    dist = axis_distance(direction, len(line), e)
    for x in dist[line > 0]:
        k = int(math.floor(x)) if abs(x) < 4 else ("far<" if x < 0 else "far>")
        hist[str(k)] = hist.get(str(k), 0) + 1
    return int(d.max()), int((d > 0).sum())


def live(args, stage, eye, lat, lobe, meta, offline, flush):
    from flyeye.render_edges import MARGIN_PX, edge_position

    stage.r.set_flush_frames(flush)
    cal = meta["calibration"]
    lin = {k: v["linear_input"] for k, v in cal.items()}
    ppf, preroll, n_move, size = meta["px_per_frame"], meta["preroll_frames"], meta["moving_frames"], meta["size"]
    dt = meta["dt"]
    warm = math.ceil(meta["warmup_frames_unsaved"] * 16 / flush)  # >= 160 GPU frames of settled history
    rows, traces, receptors = {}, {}, {}
    for name in MOVIES:
        pol, d = name.split("_")
        ref = offline[name]
        stage.set_bg(lin["grey"])
        stage.set_region(lin[pol])
        stage.place_edge(d, -MARGIN_PX)
        stage.r.reset_temporal_history()
        for _ in range(warm):
            stage.render()
        files = sorted((args.stimuli / name).glob("*.png"))
        hist_png, hist_prim = {}, {}
        px_png = dict(max=0, pixels=0, frames=0)
        px_prim = dict(max=0, pixels=0, frames=0)
        prim_vs_png = dict(max=0, pixels=0)
        pos_live, pos_prim, expected, prof = [], [], [], None
        oor = dict(live=0, primary=0, png=0)  # pixels outside [region, grey]: RCAS over/undershoot
        d_rec = torch.zeros((), device="cuda")
        d_resp = torch.zeros((), device="cuda")
        rec_all = []
        cen = {t: np.empty(len(files), np.float32) for t in TYPES}
        idx = {t: lobe.type_index(t)[(lobe.u[lobe.type_index(t)] == 0) & (lobe.v_coord[lobe.type_index(t)] == 0)][0]
               for t in TYPES}
        cen_gpu = torch.empty(len(files), len(TYPES), device="cuda")
        cen_idx = torch.stack([idx[t] for t in TYPES])
        for k, f in enumerate(files):
            e = -MARGIN_PX + max(0, k - (preroll - 1)) * ppf
            stage.place_edge(d, e)
            prim = stage.render()  # renders `flush` GPU frames, reads the primary back
            x = eye.receptors()
            if k == 0:
                lobe.fade_in(x, dt)
            v = lobe.step(x, dt)
            cen_gpu[k] = v[cen_idx]
            d_rec = torch.maximum(d_rec, (x - ref["boxeye"][k]).abs().max())
            d_resp = torch.maximum(d_resp, (v - ref["responses"][k]).abs().max())
            rec_all.append(x)
            col = eye.color[..., [2, 1, 0]].cpu().numpy()
            png = load_rgb(f)
            m, n = diff_by_distance(col, png, d, e, hist_png)
            px_png["max"], px_png["pixels"], px_png["frames"] = max(px_png["max"], m), px_png["pixels"] + n, px_png["frames"] + (n > 0)
            m, n = diff_by_distance(col, prim, d, e, hist_prim)
            px_prim["max"], px_prim["pixels"], px_prim["frames"] = max(px_prim["max"], m), px_prim["pixels"] + n, px_prim["frames"] + (n > 0)
            m, n = diff_by_distance(prim, png, d, e, {})
            prim_vs_png["max"], prim_vs_png["pixels"] = max(prim_vs_png["max"], m), prim_vs_png["pixels"] + n
            lo, hi = cal[pol]["measured_8bit"], cal["grey"]["measured_8bit"]
            lo8, hi8 = sorted((round(lo), round(hi)))
            oor["live"] += int(((col[..., 0] < lo8) | (col[..., 0] > hi8)).sum())
            oor["primary"] += int(((prim[..., 0] < lo8) | (prim[..., 0] > hi8)).sum())
            oor["png"] += int(((png[..., 0] < lo8) | (png[..., 0] > hi8)).sum())
            pos_live.append(edge_position(col, d, lo, hi)[0])
            pos_prim.append(edge_position(prim, d, lo, hi)[0])
            expected.append(max(0.0, min(float(size), e)))
            if k == preroll + n_move // 2:
                # profile across the edge at mid-crossing, centre line, 6 px either side
                c = size // 2
                if d in ("right", "left"):
                    lp, lc, pp = col[c, :, 0], prim[c, :, 0], png[c, :, 0]
                else:
                    lp, lc, pp = col[:, c, 0], prim[:, c, 0], png[:, c, 0]
                i0 = int(round(e if d in ("right", "down") else size - e))
                sl = slice(max(0, i0 - 6), min(size, i0 + 6))
                prof = dict(index_range=[sl.start, sl.stop], live=lp[sl].tolist(), primary=lc[sl].tolist(), png=pp[sl].tolist())
        cen_np = cen_gpu.cpu().numpy()
        for j, t in enumerate(TYPES):
            cen[t] = cen_np[:, j]
        traces[name] = cen
        receptors[name] = torch.stack(rec_all).cpu().numpy()
        exp = np.array(expected)
        inside = (exp > 10) & (exp < size - 10)
        res_live = np.array(pos_live)[inside] - exp[inside]
        res_prim = np.array(pos_prim)[inside] - exp[inside]
        rows[name] = dict(
            live_vs_png=dict(px_png, by_edge_distance=hist_png),
            live_vs_primary=dict(px_prim, by_edge_distance=hist_prim),
            primary_vs_png=prim_vs_png,
            outside_levels_pixels=oor,
            edge_profile_mid=prof,
            receptors_max_abs=d_rec.item(),
            responses_max_abs=d_resp.item(),
            edge_live_minus_model_mean=float(res_live.mean()),
            edge_live_minus_model_max_abs=float(np.abs(res_live).max()),
            edge_primary_minus_model_mean=float(res_prim.mean()),
            edge_primary_minus_model_max_abs=float(np.abs(res_prim).max()),
        )
        r = rows[name]
        print(f"b f{flush:<2d} {name:9s} live-png max {px_png['max']:3d} px {px_png['pixels']:6d} "
              f"bins {dict(sorted(hist_png.items()))}  prim-png {prim_vs_png['max']}/{prim_vs_png['pixels']}  "
              f"live-prim {px_prim['max']}/{px_prim['pixels']}  outside {oor}  rec {r['receptors_max_abs']:.2e} resp {r['responses_max_abs']:.2e}  "
              f"edge live {r['edge_live_minus_model_mean']:+.3f}/{r['edge_live_minus_model_max_abs']:.3f} "
              f"prim {r['edge_primary_minus_model_mean']:+.3f}/{r['edge_primary_minus_model_max_abs']:.3f}")
    return rows, traces, receptors


def control_primary_only(args, stage, meta, movies=("on_right", "off_up")):
    """Before any view exists: the primary at flush 16 against the Phase 0 PNGs."""
    from flyeye.render_edges import MARGIN_PX

    cal = meta["calibration"]
    lin = {k: v["linear_input"] for k, v in cal.items()}
    out = {}
    for name in movies:
        pol, d = name.split("_")
        stage.set_bg(lin["grey"])
        stage.set_region(lin[pol])
        stage.place_edge(d, -MARGIN_PX)
        stage.r.reset_temporal_history()
        for _ in range(meta["warmup_frames_unsaved"]):
            stage.render()
        mx, n = 0, 0
        for k, f in enumerate(sorted((args.stimuli / name).glob("*.png"))):
            stage.place_edge(d, -MARGIN_PX + max(0, k - (meta["preroll_frames"] - 1)) * meta["px_per_frame"])
            dd = np.abs(stage.render().astype(np.int16) - load_rgb(f).astype(np.int16))
            mx, n = max(mx, int(dd.max())), n + int((dd.max(axis=2) > 0).sum())
        out[name] = dict(max_abs=mx, pixels=n)
        print(f"control primary-only flush 16 {name}: max {mx} pixels {n}")
    return out


def dsi_table(live_traces, offline_traces, preroll):
    tr = {m: dict(own=live_traces[m], ref=offline_traces[m], dt=0.01) for m in MOVIES}
    return selectivity(tr, preroll)


def trace_dev(a, b):
    """Per type: max over its polarity's 4 movies and all frames of |a - b| (central column)."""
    out = {}
    for t in TYPES:
        pol = "on" if t in T4 else "off"
        out[t] = float(max(np.abs(a[f"{pol}_{d}"][t] - b[f"{pol}_{d}"][t]).max() for d in DIRS))
    return out


# -- figure -------------------------------------------------------------------------------
def figure(offline_traces, live_traces, rows_live, rows_off, preroll, dt, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = dict(right="#2a78d6", left="#eb6834", up="#1baf7a", down="#eda100")
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.2), sharex=True, constrained_layout=True)
    for ax, row, roff in zip(axes.ravel(), rows_live, rows_off):
        t, pol = row["cell_type"], row["polarity"]
        for d in DIRS:
            off = offline_traces[f"{pol}_{d}"][t]
            lv = live_traces[f"{pol}_{d}"][t]
            time_s = np.arange(len(off)) * dt
            ax.plot(time_s, off, color=colors[d], lw=2.2, label=f"{d}")
            ax.plot(time_s, lv, color="#222222", lw=1, ls=(0, (3, 3)))
        ax.axvline((preroll - 0.5) * dt, color="#999999", lw=0.8)
        ax.set_title(f"{t} ({pol.upper()}): prefers {row['preferred_direction']}, "
                     f"DSI live {row['dsi']:.2f} / offline {roff['dsi']:.2f}", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel("time since first frame (s)")
    axes[0, 0].set_ylabel("central column v (a.u.)")
    axes[1, 0].set_ylabel("central column v (a.u.)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    labels = [f"{l} (offline, Phase 0 PNGs)" for l in labels]
    handles.append(plt.Line2D([], [], color="#222222", lw=1, ls=(0, (3, 3))))
    labels.append("live secondary view, flush 16 (dashed)")
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False)
    fig.suptitle("Phase 1: live EyeView (add_view + FrameTensors, CUDA) vs the offline Phase 0 path, "
                 "same 8 moving-edge movies; grey line = end of pre-roll", fontsize=11)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    print("figure", path)


def taa_figure(offline_traces, live_by_flush, rows_off, preroll, dt, path):
    """Per subtype, its preferred-direction movie: live minus offline central trace per flush."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {16: "#2a78d6", 3: "#eda100", 1: "#d6336c"}
    fig, axes = plt.subplots(2, 4, figsize=(16, 6.4), sharex=True, constrained_layout=True)
    for ax, row in zip(axes.ravel(), rows_off):
        t, pol, d = row["cell_type"], row["polarity"], row["preferred_direction"]
        off = offline_traces[f"{pol}_{d}"][t]
        time_s = np.arange(len(off)) * dt
        ax2 = ax.twinx()
        ax2.plot(time_s, off, color="#bbbbbb", lw=3, zorder=0)
        ax2.set_yticks([])
        for f in sorted(live_by_flush, reverse=True):
            ax.plot(time_s, live_by_flush[f][f"{pol}_{d}"][t] - off, color=colors.get(f, "#222222"), lw=1.3,
                    label=f"flush {f}")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        ax.axhline(0, color="#666666", lw=0.6)
        ax.set_title(f"{t} ({pol.upper()}, {d} = preferred), peak {off.max() - off[preroll - 1]:.2f}", fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
    for ax in axes[1]:
        ax.set_xlabel("time since first frame (s)")
    axes[0, 0].set_ylabel("live - offline, central v")
    axes[1, 0].set_ylabel("live - offline, central v")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(plt.Line2D([], [], color="#bbbbbb", lw=3))
    labels.append("offline trace (grey, own scale, right)")
    fig.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Phase 1: what the secondary view's TAA does to the optic lobe. Live minus offline central-column "
                 "response, preferred-direction edge, render() flushing 16, 3 or 1 GPU frames", fontsize=11)
    fig.savefig(path, dpi=110)
    print("figure", path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stimuli", type=Path, default=Path(r"C:\dev\_flyeye\stimuli"))
    ap.add_argument("--own", type=Path, default=Path(r"C:\dev\_flyeye\own"))
    ap.add_argument("--out", type=Path, default=Path(r"C:\dev\_flyeye\online"))
    ap.add_argument("--flush", default="16,3,1")
    ap.add_argument("--figure", type=Path, default=Path(__file__).parent / "figures" / "phase1_online_vs_offline.png")
    ap.add_argument("--figure-only", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((args.stimuli / "meta.json").read_text())
    preroll, dt = int(meta["preroll_frames"]), float(meta["dt"])
    flushes = [int(f) for f in args.flush.split(",")]

    if args.figure_only:
        z = np.load(args.out / "traces.npz")
        tr = lambda key: {m: {t: z[f"{key}/{m}/{t}"] for t in TYPES} for m in MOVIES}
        off, lv = tr("offline"), tr("live16")
        figure(off, lv, dsi_table(lv, off, preroll), dsi_table(off, off, preroll), preroll, dt, args.figure)
        live_by_flush = {f: tr(f"live{f}") for f in flushes if f"live{f}/on_right/T4a" in z.files}
        taa_figure(off, live_by_flush, dsi_table(off, off, preroll), preroll, dt,
                   args.figure.with_name("phase1_taa_flush.png"))
        return

    lobe = OpticLobe(device="cuda")
    lat = HexLattice()
    lat.centers = lat.centers.cuda()
    offline = {}
    for name in MOVIES:
        z = np.load(args.own / f"{name}_cpu_float32.npz")
        offline[name] = dict(dt=float(z["dt"]), boxeye=torch.from_numpy(z["boxeye"]).cuda(),
                             initial_state=torch.from_numpy(z["initial_state"]).cuda(),
                             responses=torch.from_numpy(z["responses"]).cuda())
    offline_traces = {m: central_traces(lobe, offline[m]["responses"]) for m in MOVIES}
    report = dict(frames_identical=frames_identical(args, lat, lobe, offline))
    a = report["frames_identical"].values()
    report["frames_identical_all"] = dict(
        receptors_max_abs=max(r["receptors_max_abs"] for r in a),
        responses_max_abs=max(r["responses_max_abs"] for r in a),
        responses_rel=max(r["responses_rel"] for r in a),
        luma_u8_equal=all(r["luma_u8_equal"] for r in a),
        passes=max(r["responses_rel"] for r in a) <= 1e-4,
    )
    print("a ALL", report["frames_identical_all"])

    import threepp as tp
    from flyeye.render_edges import Stage

    t0 = time.perf_counter()
    stage = Stage(SimpleNamespace(size=meta["size"], fov=meta["fov_deg"], flush=flushes[0], fps=meta["fps"]))
    stage.hide_region()
    stage.set_bg(0.2)
    for _ in range(5):
        stage.render()
    if flushes[0] == 16:
        report["control_primary_only_flush16"] = control_primary_only(args, stage, meta)
    eye_cam = tp.PerspectiveCamera(meta["fov_deg"], 1.0, 0.1, 10.0)  # the primary's twin
    eye = EyeView(stage.r, eye_cam, size=meta["size"], lattice=lat)
    stage.render()
    eye.arm()
    report["renderer_settings"] = stage.applied
    save = {f"offline/{m}/{t}": offline_traces[m][t] for m in MOVIES for t in TYPES}
    live_traces, off_rows = {}, dsi_table(offline_traces, offline_traces, preroll)
    for flush in flushes:
        rows, traces, rec = live(args, stage, eye, lat, lobe, meta, offline, flush)
        live_traces[flush] = traces
        sel = dsi_table(traces, offline_traces, preroll)
        report[f"live_flush{flush}"] = dict(movies=rows, selectivity=sel)
        save.update({f"live{flush}/{m}/{t}": traces[m][t] for m in MOVIES for t in TYPES})
        np.savez_compressed(args.out / f"receptors_flush{flush}.npz", **rec)
        print(f"flush {flush}: {time.perf_counter() - t0:.0f} s")
    eye.close()
    np.savez_compressed(args.out / "traces.npz", **save)

    # summary: DSI per subtype, preferred direction, trace deviations
    summary = {}
    for flush in flushes:
        sel = report[f"live_flush{flush}"]["selectivity"]
        s = dict(
            dsi={r["cell_type"]: round(r["dsi"], 4) for r in sel},
            preferred={r["cell_type"]: r["preferred_direction"] for r in sel},
            dsi_minus_offline={r["cell_type"]: round(r["dsi"] - o["dsi"], 4) for r, o in zip(sel, off_rows)},
            preferred_same=all(r["preferred_direction"] == o["preferred_direction"] for r, o in zip(sel, off_rows)),
            central_dev_vs_offline=trace_dev(live_traces[flush], offline_traces),
        )
        if 16 in live_traces:
            s["central_dev_vs_flush16"] = trace_dev(live_traces[flush], live_traces[16])
        rows = report[f"live_flush{flush}"]["movies"].values()
        s["receptors_max_abs"] = max(r["receptors_max_abs"] for r in rows)
        s["responses_max_abs"] = max(r["responses_max_abs"] for r in rows)
        s["edge_live_minus_model_mean"] = {m: round(r["edge_live_minus_model_mean"], 3) for m, r in report[f"live_flush{flush}"]["movies"].items()}
        s["edge_live_minus_model_max_abs"] = max(r["edge_live_minus_model_max_abs"] for r in rows)
        mv = report[f"live_flush{flush}"]["movies"]
        for src in ("live", "primary"):
            res = {m: mv[m][f"edge_{src}_minus_model_mean"] for m in MOVIES}
            # opposite directions: the common part is the temporal lag (behind the model),
            # the antisymmetric part a fixed image shift (+x right, +y down)
            s[f"edge_{src}_lag_px"] = {f"{p}_{ax}": round(-(res[f"{p}_{a}"] + res[f"{p}_{b}"]) / 2, 3)
                                       for p in ("on", "off") for ax, a, b in (("x", "right", "left"), ("y", "down", "up"))}
            s[f"edge_{src}_shift_px"] = {f"{p}_{ax}": round((res[f"{p}_{a}"] - res[f"{p}_{b}"]) / 2, 3)
                                         for p in ("on", "off") for ax, a, b in (("x", "right", "left"), ("y", "down", "up"))}
        s["edge_primary_minus_model_max_abs"] = max(r["edge_primary_minus_model_max_abs"] for r in rows)
        s["passes"] = s["preferred_same"] and max(abs(v) for v in s["dsi_minus_offline"].values()) <= 0.02
        summary[flush] = s
        print(f"flush {flush}: preferred same {s['preferred_same']}  DSI {s['dsi']}")
        print(f"   dDSI {s['dsi_minus_offline']}")
        print(f"   central dev vs offline {({k: round(v, 3) for k, v in s['central_dev_vs_offline'].items()})}")
        if "central_dev_vs_flush16" in s:
            print(f"   central dev vs flush16 {({k: round(v, 3) for k, v in s['central_dev_vs_flush16'].items()})}")
        print(f"   receptors {s['receptors_max_abs']:.3e} responses {s['responses_max_abs']:.3e}  "
              f"edge live max {s['edge_live_minus_model_max_abs']:.3f} primary max {s['edge_primary_minus_model_max_abs']:.3f}  "
              f"passes {s['passes']}")
        print(f"   edge lag live {s['edge_live_lag_px']} shift {s['edge_live_shift_px']}")
        print(f"   edge lag primary {s['edge_primary_lag_px']} shift {s['edge_primary_shift_px']}")
    report["summary"] = summary
    report["offline_selectivity"] = off_rows
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    if 16 in live_traces:
        figure(offline_traces, live_traces[16], dsi_table(live_traces[16], offline_traces, preroll), off_rows,
               preroll, dt, args.figure)
    taa_figure(offline_traces, live_traces, off_rows, preroll, dt, args.figure.with_name("phase1_taa_flush.png"))


if __name__ == "__main__":
    main()
