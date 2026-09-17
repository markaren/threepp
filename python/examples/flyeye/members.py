"""Other flyvis ensemble members, re-run offline over the Phase 2 recordings (no rendering).

    py -3.14 python/examples/flyeye/members.py                  # replay + edges, every member npz
    py -3.14 python/examples/flyeye/members.py --members 000 --verify-only
    py -3.14 python/examples/flyeye/members.py --edges-only

Inputs:
  --models  C:/dev/_flyeye/ensemble/models/flow_0000_NNN.npz, written by
            tools/export_flyvis.py --member NNN --out ... (same format as data/flyeye_model.npz)
  --phase2  C:/dev/_flyeye/phase2/<scenario>_<light>.npz, the 21 recordings

Replay, exactly what scenarios.run did per recording: OpticLobe(member, cuda, float32) and
MotionField(lobe) (rest = relu(v) after fade_in on uniform grey 0.5, single eye, dt 0.01);
fade_in on frame 0's recorded receptors (both eyes batched, (2, 721)), then one step per
frame at dt 0.01 with that frame's receptors, t4t5_raw[k] = MotionField.raw(v after frame k).
The recorded receptors are float16, the live recorder fed float32, so a replay of member
000 differs from the recorded t4t5_raw by the float16 rounding of the input (reported).

Output per member (C:/dev/_flyeye/ensemble/raw/<NNN>/):
  <scenario>_<light>.npz  np.savez_compressed:
      t4t5_raw (T, 2, 8, 721) float16  relu(v) - rest, order T4a T4b T4c T4d T5a T5b T5c T5d,
                                       frame-aligned 1:1 with the recording (same T, same eyes)
      rest     (8, 721) float32        this member's grey-0.5 rest rate that was subtracted
      member   str, run str, types (8,) str
  rest.npz   rest (8, 721) float32, types, member, model (npz path)
Report C:/dev/_flyeye/ensemble/members.json ("replay": per member per run stats, "edges": Phase 0
procedural edge peaks / DSI per member per type) + figure C:/dev/_flyeye/ensemble/edges_dsi.png.
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

from flyeye.lattice import HexLattice  # noqa: E402
from flyeye.optic_lobe import OpticLobe  # noqa: E402
from flyeye.readouts import MOTION_TYPES, MotionField  # noqa: E402

DT = 0.01
ROOT = Path("C:/dev/_flyeye")
SCENARIOS = ("straight_3", "straight_10", "straight_30", "yaw", "pitch", "roll", "approach")
LIGHTS = ("bright", "dim", "dark")

# Phase 0 edge geometry (dt_sweep.py / render_edges.py): 403 px, grey 128, ON 204, OFF 53,
# one-pixel area-coverage edge, 13 columns/s = 169 px/s, starts 0.19 s in, 261 frames at 100 Hz.
SIZE, GREY, ON, OFF, MARGIN = 403, 128, 204, 53, 2.0
PX_PER_S, T_START, N_MOVIE = 13 * 13.0, 0.19, 261
DIRS = ("right", "left", "up", "down")
OPP = dict(right="left", left="right", up="down", down="up")


def member_npz(models: Path, member: str) -> Path:
    return models / f"flow_0000_{member}.npz"


@torch.no_grad()
def replay(lobe: OpticLobe, motion: MotionField, receptors: np.ndarray) -> torch.Tensor:
    """receptors (T, B, 721) -> t4t5_raw (T, B, 8, 721) float16 on the lobe's device."""
    x = torch.from_numpy(receptors).to(device=lobe.device, dtype=lobe.dtype)
    T, B = x.shape[:2]
    out = torch.empty(T, B, 8, x.shape[2], dtype=torch.float16, device=lobe.device)
    for k in range(T):
        if k == 0:
            lobe.fade_in(x[0], DT)
        v = lobe.step(x[k], DT)
        out[k] = motion.raw(v).half()
    return out


def edge_frame(t, direction, level, device):
    e = -MARGIN + max(0.0, t - T_START) * PX_PER_S
    idx = torch.arange(SIZE, device=device, dtype=torch.float64)
    cov = (e - idx).clamp(0, 1)
    if direction in ("left", "up"):
        cov = cov.flip(0)
    lum = (GREY + cov * (level - GREY)) / 255
    if direction in ("right", "left"):
        return lum[None, :].expand(SIZE, SIZE).float()
    return lum[:, None].expand(SIZE, SIZE).float()


def edge_movies(device) -> tuple[list[str], torch.Tensor]:
    lat = HexLattice()
    names, movies = [], []
    for pol, level in (("on", ON), ("off", OFF)):
        for d in DIRS:
            names.append(f"{pol}_{d}")
            movies.append(torch.stack([lat.box_eye(edge_frame(k * DT, d, level, device)) for k in range(N_MOVIE)]))
    return names, torch.stack(movies, 1)  # (T, 8 movies, 721)


@torch.no_grad()
def edges(lobe: OpticLobe, names, movies) -> dict:
    """Central-column T4a-d (ON edges) / T5a-d (OFF edges): peak above the last pre-roll state per
    direction, preferred = argmax, DSI = (pref - null) / (pref + null) with null = opposite."""
    central = torch.stack([lobe.type_index(t)[(lobe.u[lobe.type_index(t)] == 0) & (lobe.v_coord[lobe.type_index(t)] == 0)][0]
                           for t in MOTION_TYPES])
    lobe.fade_in(movies[0], DT)
    tr = torch.empty(N_MOVIE, movies.shape[1], 8, dtype=torch.float64)
    for k in range(N_MOVIE):
        tr[k] = lobe.step(movies[k], DT)[:, central].double().cpu()
    t_axis = (np.arange(N_MOVIE) + 1) * DT
    b = int(np.searchsorted(t_axis, T_START))
    res = {}
    for j, t in enumerate(MOTION_TYPES):
        pol = "on" if t.startswith("T4") else "off"
        pk = {d: float(tr[b:, names.index(f"{pol}_{d}"), j].max() - tr[b - 1, names.index(f"{pol}_{d}"), j]) for d in DIRS}
        pref = max(pk, key=pk.get)
        p, n = max(pk[pref], 0.0), max(pk[OPP[pref]], 0.0)
        # sustained: last 20 frames (the central column has seen the uniform ON/OFF level for > 1 s)
        sus = {d: float(tr[-20:, names.index(f"{pol}_{d}"), j].mean() - tr[b - 1, names.index(f"{pol}_{d}"), j]) for d in DIRS}
        res[t] = dict(pref=pref, pref_peak=p, null_peak=n, dsi=(p - n) / (p + n) if p + n > 0 else 0.0, peaks=pk,
                      sustained=sus, sustained_mean=float(np.mean(list(sus.values()))))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=ROOT / "ensemble" / "models")
    ap.add_argument("--phase2", type=Path, default=ROOT / "phase2")
    ap.add_argument("--out", type=Path, default=ROOT / "ensemble" / "raw")
    ap.add_argument("--members", default="", help="comma list of NNN; default every npz in --models")
    ap.add_argument("--verify-only", action="store_true", help="replay member 000, compare, write nothing")
    ap.add_argument("--edges-only", action="store_true")
    ap.add_argument("--no-edges", action="store_true")
    args = ap.parse_args(argv)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    members = [m for m in args.members.split(",") if m] or sorted(p.stem.split("_")[-1] for p in args.models.glob("flow_0000_*.npz"))
    runs = [f"{s}_{l}" for s in SCENARIOS for l in LIGHTS]
    report_path = args.out.parent / "members.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}

    if not args.edges_only:
        for mem in members:
            if args.verify_only and mem != "000":
                continue
            t0 = time.time()
            lobe = OpticLobe(member_npz(args.models, mem), device=dev)
            motion = MotionField(lobe)
            rest = motion.rest.float().cpu().numpy()
            odir = args.out / mem
            if not args.verify_only:
                odir.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(odir / "rest.npz", rest=rest, types=np.array(MOTION_TYPES),
                                    member=np.array(mem), model=np.array(str(member_npz(args.models, mem))))
            info = dict(rest_mean_per_type={t: float(rest[j].mean()) for j, t in enumerate(MOTION_TYPES)}, runs={})
            t_sim = 0.0
            for run in runs:
                z = np.load(args.phase2 / f"{run}.npz")
                ts = time.time()
                raw = replay(lobe, motion, z["receptors"])
                torch.cuda.synchronize() if dev == "cuda" else None
                t_sim += time.time() - ts
                raw = raw.cpu().numpy()
                r = dict(T=int(raw.shape[0]), max_abs_rate=float(np.abs(raw.astype(np.float32)).max()))
                if mem == "000":
                    rec = z["t4t5_raw"]
                    d = np.abs(raw.astype(np.float32) - rec.astype(np.float32))
                    r.update(max_abs_diff_vs_recorded=float(d.max()), mean_abs_diff_vs_recorded=float(d.mean()),
                             frac_diff_le_4_9e_4=float((d <= 4.9e-4).mean()), frac_diff_le_1e_4=float((d <= 1e-4).mean()),
                             p999_abs_diff=float(np.quantile(d[::7], 0.999)), frac_f16_identical=float((raw == rec).mean()),
                             recorded_abs_p90=float(np.quantile(np.abs(rec[::7].astype(np.float32)), 0.9)))
                    print(f"  {mem} {run}: T {raw.shape[0]} max|replay - recorded| {d.max():.3e} mean {d.mean():.2e} "
                          f"identical f16 {r['frac_f16_identical']:.3f}, <=1e-4 {r['frac_diff_le_1e_4']:.5f}, "
                          f"recorded |raw| p90 {r['recorded_abs_p90']:.3f}", flush=True)
                if not args.verify_only:
                    np.savez_compressed(odir / f"{run}.npz", t4t5_raw=raw, rest=rest, member=np.array(mem),
                                        run=np.array(run), types=np.array(MOTION_TYPES))
                info["runs"][run] = r
            info["sim_seconds"] = t_sim
            info["wall_seconds"] = time.time() - t0
            print(f"member {mem}: sim {t_sim:.1f} s, wall {info['wall_seconds']:.1f} s", flush=True)
            if not args.verify_only:
                report.setdefault("replay", {})[mem] = info
                report_path.write_text(json.dumps(report, indent=1))

    if not args.no_edges and not args.verify_only:
        names, movies = edge_movies(dev)
        res = {}
        for mem in members:
            lobe = OpticLobe(member_npz(args.models, mem), device=dev)
            res[mem] = edges(lobe, names, movies)
            print(f"edges {mem}: " + ", ".join(f"{t} {e['pref']} {e['pref_peak']:.3f}/{e['dsi']:.2f}" for t, e in res[mem].items())
                  + f"; T4c on_right {res[mem]['T4c']['peaks']['right']:.3f}; sustained " + " ".join(f"{e['sustained_mean']:.2f}" for e in res[mem].values()), flush=True)
        report["edges"] = res
        report_path.write_text(json.dumps(report, indent=1))
        figure(res, args.out.parent / "edges_dsi.png")


def figure(res: dict, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mems = list(res)
    fig, ax = plt.subplots(4, 1, figsize=(12, 12), constrained_layout=True)
    x = np.arange(8)
    w = 0.8 / len(mems)
    expect = dict(T4a="left", T4b="right", T4c="up", T4d="down", T5a="left", T5b="right", T5c="up", T5d="down")
    for i, m in enumerate(mems):
        pk = [res[m][t]["pref_peak"] for t in MOTION_TYPES]
        dsi = [res[m][t]["dsi"] for t in MOTION_TYPES]
        wrong = [res[m][t]["pref"] != expect[t] for t in MOTION_TYPES]
        c = plt.cm.viridis(i / max(1, len(mems) - 1))
        ax[0].bar(x + (i - len(mems) / 2 + 0.5) * w, pk, w, color=c, label=m)
        ax[1].bar(x + (i - len(mems) / 2 + 0.5) * w, dsi, w, color=c)
        for j in np.nonzero(wrong)[0]:
            ax[1].text(x[j] + (i - len(mems) / 2 + 0.5) * w, 0.02, "x", ha="center", color="red", fontsize=9)
    ax[0].set_ylabel("preferred-direction peak")
    ax[0].legend(ncol=len(mems), fontsize=8, title="member")
    ax[1].set_ylabel("DSI (red x = pref not cardinal-expected)")
    for a in ax[:2]:
        a.set_xticks(x, [f"{t}\n({expect[t]})" for t in MOTION_TYPES])
    # per-direction peaks of T4c and T4d (the member-000 quirks)
    lab = [f"{t} {d}" for t in ("T4c", "T4d") for d in DIRS]
    xx = np.arange(len(lab))
    for i, m in enumerate(mems):
        vals = [res[m][t]["peaks"][d] for t in ("T4c", "T4d") for d in DIRS]
        ax[2].bar(xx + (i - len(mems) / 2 + 0.5) * w, vals, w, color=plt.cm.viridis(i / max(1, len(mems) - 1)))
    ax[2].set_xticks(xx, [s.replace(" ", "\nON ") for s in lab])
    ax[2].set_ylabel("central-column peak per ON edge direction")
    for i, m in enumerate(mems):
        ax[3].bar(x + (i - len(mems) / 2 + 0.5) * w, [res[m][t]["sustained_mean"] for t in MOTION_TYPES], w,
                  color=plt.cm.viridis(i / max(1, len(mems) - 1)))
    ax[3].axhline(0, color="k", lw=0.5)
    ax[3].set_xticks(x, [f"{t} ({'ON' if t.startswith('T4') else 'OFF'})" for t in MOTION_TYPES])
    ax[3].set_ylabel("sustained level after the edge\n(mean of 4 directions, minus pre-roll)")
    fig.suptitle("flyvis flow/0000 members: Phase 0 procedural edges, 13 columns/s, 100 Hz, central column")
    fig.savefig(path, dpi=110)
    print("figure", path)


if __name__ == "__main__":
    sys.exit(main())
