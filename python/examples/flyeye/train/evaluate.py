"""Score a retrained checkpoint the same way every earlier flyeye number was scored.

    py -3.14 python/examples/flyeye/train/evaluate.py score --name "000 unit" --ckpt <model.npz>
    py -3.14 python/examples/flyeye/train/evaluate.py score --name "retrained linear" \
        --ckpt C:/dev/_flyeye/retrain/linear/model_final.npz \
        --decoder linear --dec-state C:/dev/_flyeye/retrain/linear/decoder_final.pt
    py -3.14 python/examples/flyeye/train/evaluate.py gratings --name before --ckpt <model.npz>
    py -3.14 python/examples/flyeye/train/evaluate.py figure

`scoring.py` is the judge for everything and is not touched. What is reported per model and
per readout:

- **Held-out flights.** Train on `straight_3`, `straight_10`, `yaw`, `roll`; test on
  `straight_30`, `pitch`, `approach`, all three lightings (`train/__init__.py`). The gate is
  the fraction of scene column-frames within 45 deg of the true direction, in the best log2
  speed bin with N >= 20k, lag 20 ms, and also in the fixed 2-4 columns/s bin that every
  earlier table used. Handoff 5b: bright, dim and dark share scene and trajectory, so holding
  lighting out tests lighting only; trajectories have to be held out as well.
- **Its OWN motion-blind baseline.** The same model and the same readout driven by the run's
  own rest frames 30-100, ping-ponged, with 10 s of that repeated rest dropped first so the
  circuit has settled. A per-column static prior scores on this baseline too, so only
  gate minus baseline (the lift) is evidence of motion sensing.
- **Rotation** through `readouts.RotationReadout` and `scoring.score_rotation` on the yaw,
  pitch and roll runs (of those, only `pitch` is held out; the other two are training
  trajectories and are marked as such).
- **The static response**: |field| p50 and p90 over the rest segments of the held-out runs,
  scene columns, the first 0.3 s of each rest dropped.
- **Tuning**: `repair.py`'s procedural gratings on the checkpoint. 24 directions at 4 Hz and
  8 columns give the preferred angle, the first-harmonic amplitude and the DSI per subtype;
  8 directions over 0.25-16 Hz give the TF curve and its peak.

Readouts: `unit` is `readouts.MotionField` with unit gains, recomputing that model's own
grey-0.5 rest, exactly as Phase 2 and the repair round did. `dec` is the decoder trained with
the model; its `rest` buffer is replaced by the evaluated model's own rest so that member 000
and the retrained lobe are both read with their own resting rates.

Results accumulate in `C:/dev/_flyeye/retrain/results.json`; the figure is
`python/examples/flyeye/figures/retrain_smoke.png`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from flyeye import MODEL_NPZ, scoring as sc  # noqa: E402
from flyeye.lattice import HexLattice  # noqa: E402
from flyeye.optic_lobe import OpticLobe  # noqa: E402
from flyeye.readouts import MOTION_TYPES  # noqa: E402
from flyeye.repair import N_DIRS, grating_rates, vector_mean  # noqa: E402
from flyeye.train import RETRAIN_ROOT, TEST_SCEN, TRAIN_SCEN  # noqa: E402
from flyeye.train import decoder as dc  # noqa: E402
from flyeye.train.model import OUTPUT_TYPES, unit_field  # noqa: E402

DT = 0.01
LAG = 2  # frames, the Phase 2 gate lag
SETTLE = 1000  # frames of repeated rest before the baseline frames (repair.SETTLE)
FIXED_BIN = 3  # 2-4 columns/s in scoring.SPEED_EDGES
RESULTS = RETRAIN_ROOT / "results.json"
FIGURE = Path(__file__).resolve().parents[1] / "figures" / "retrain_smoke.png"
TF_HZ = (0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0)


# ---------------------------------------------------------------------------------------------- replay
def gather_index(lobe: OpticLobe, names) -> torch.Tensor:
    return torch.stack([lobe.type_index(n) for n in names])


@torch.no_grad()
def grey_rest(lobe: OpticLobe, index: torch.Tensor) -> torch.Tensor:
    """(K, 721) relu(v) at uniform grey 0.5, MotionField.rest for this model."""
    saved = lobe.v
    grey = torch.full((index.shape[1],), 0.5, dtype=lobe.dtype, device=lobe.device)
    v = lobe.fade_in(grey, DT)
    out = torch.relu(v[index]).clone()
    lobe.v = saved
    return out


@torch.no_grad()
def replay(lobe: OpticLobe, index: torch.Tensor, receptors: np.ndarray | torch.Tensor) -> torch.Tensor:
    """receptors (T, B, 721) -> relu(v) at `index`, (T, B, K, 721) float16 on the lobe's device.

    Exactly what members.replay does (fade_in on frame 0, then one step per frame at dt 0.01),
    gathering the wanted types instead of only T4/T5.
    """
    x = torch.as_tensor(np.asarray(receptors), device=lobe.device, dtype=lobe.dtype)
    T, B = x.shape[:2]
    out = torch.empty(T, B, index.shape[0], index.shape[1], dtype=torch.float16, device=lobe.device)
    for k in range(T):
        if k == 0:
            lobe.fade_in(x[0], DT)
        v = lobe.step(x[k], DT)
        out[k] = torch.relu(v[:, index]).half()
    return out


def rest_receptors(rec: np.ndarray, settle=SETTLE) -> np.ndarray:
    """repair.rest_input on the receptors: frames 30-100 ping-ponged for settle + T frames."""
    r = rec[sc.REST]
    cyc = np.concatenate([r, r[::-1]])
    return cyc[np.arange(settle + len(rec)) % len(cyc)]


# ---------------------------------------------------------------------------------------------- readouts
def apply_decoder(decoder, rates: torch.Tensor, chunk=64) -> np.ndarray:
    """(T, B, K, 721) float16 -> (T, B, 2, 721) float32 numpy, in frame chunks."""
    decoder.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rates), chunk):
            out.append(decoder(rates[i:i + chunk].float()).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def unit_readout(rates8: torch.Tensor, rest8: torch.Tensor, chunk=256) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(rates8), chunk):
            out.append(unit_field(rates8[i:i + chunk].float(), rest8).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def build_decoder(kind: str, state: Path, lobe: OpticLobe, index: torch.Tensor):
    rest = grey_rest(lobe, index)
    if kind == "linear":
        d = dc.LinearDecoder(rest, init=np.zeros((8, 2), np.float32), source=str(state))
    else:
        u = lobe.u[index[0]].cpu().numpy()
        v = lobe.v_coord[index[0]].cpu().numpy()
        d = dc.FlyvisDecoder(u, v, in_ch=len(OUTPUT_TYPES))
    d = d.to(lobe.device)
    d.load_state_dict(torch.load(state, map_location=lobe.device))
    if hasattr(d, "set_rest"):
        d.set_rest(rest)  # this model's own grey rest, not the one the decoder was trained with
    return d.eval()


# ---------------------------------------------------------------------------------------------- scoring
def rest_magnitude(field: np.ndarray, run: sc.Run) -> tuple[np.ndarray, int]:
    """|field| over the run's rest segments, scene columns, first 0.3 s of each rest dropped."""
    keep = np.zeros(run.T, bool)
    for name, a, b in run.segments:
        if name == "rest":
            keep[a + sc.SKIP:b] = True
    if not keep.any():
        return np.zeros(0, np.float32), 0
    f = field[keep]
    m = np.hypot(f[:, :, 0], f[:, :, 1])
    return m[run.scene[keep]].astype(np.float32), int(keep.sum())


def score_model(name: str, ckpt: Path, decoder_kind=None, dec_state=None, device="cuda",
                scenarios=TEST_SCEN, lag=LAG) -> dict:
    """Replay one checkpoint over all 21 recordings and score every readout it supports."""
    t0 = time.time()
    lobe = OpticLobe(ckpt, device=device, dtype=torch.float32)
    runs = sc.Runs()
    idx8 = gather_index(lobe, MOTION_TYPES)
    rest8 = grey_rest(lobe, idx8)
    dec = idx_dec = None
    if decoder_kind:
        names = MOTION_TYPES if decoder_kind == "linear" else OUTPUT_TYPES
        idx_dec = idx8 if decoder_kind == "linear" else gather_index(lobe, names)
        dec = build_decoder(decoder_kind, Path(dec_state), lobe, idx_dec)

    readouts = ["unit"] + (["dec"] if dec is not None else [])
    C = {r: {} for r in readouts}
    Cp = {r: {} for r in readouts}
    Cb = {r: {} for r in readouts}
    rot = {r: {} for r in readouts}
    restmag = {r: [] for r in readouts}
    test_keys = [(s, l) for l in sc.LIGHTS for s in scenarios]

    for key in runs.keys():
        run = runs[key]
        rec = run.arr("receptors")
        rates = replay(lobe, idx8 if idx_dec is None else
                       (idx8 if decoder_kind == "linear" else torch.cat([idx8, idx_dec])), rec)
        f = {}
        if decoder_kind == "linear" or idx_dec is None:
            f["unit"] = unit_readout(rates, rest8)
            if dec is not None:
                f["dec"] = apply_decoder(dec, rates)
        else:
            f["unit"] = unit_readout(rates[:, :, :8], rest8)
            f["dec"] = apply_decoder(dec, rates[:, :, 8:])
        del rates
        for r in readouts:
            if key[0] in sc.AXIS:
                rot[r][key] = sc.field_rotation(f[r], run)
            if key in test_keys:
                C[r][key] = sc.direction_counts(f[r], run, [lag])
                Cp[r][key] = sc.direction_counts(sc.pool_field(f[r], run.P), run, [lag], pooled=True)
                m, _ = rest_magnitude(f[r], run)
                restmag[r].append(m)
        if key in test_keys:  # the motion-blind baseline: this model on the run's own rest, repeated
            brates = replay(lobe, idx8 if decoder_kind != "flyvis" else torch.cat([idx8, idx_dec]),
                            rest_receptors(rec))[SETTLE:]
            fb = {}
            if decoder_kind == "flyvis":
                fb["unit"] = unit_readout(brates[:, :, :8], rest8)
                fb["dec"] = apply_decoder(dec, brates[:, :, 8:])
            else:
                fb["unit"] = unit_readout(brates, rest8)
                if dec is not None:
                    fb["dec"] = apply_decoder(dec, brates)
            del brates
            for r in readouts:
                Cb[r][key] = sc.direction_counts(fb[r], run, [lag])
        run.drop()
        print(f"  {name}: {key[0]}_{key[1]} done ({time.time() - t0:.0f} s)", flush=True)

    out = dict(name=name, ckpt=str(ckpt), decoder=decoder_kind, dec_state=str(dec_state) if dec_state else None,
               test_scenarios=list(scenarios), train_scenarios=list(TRAIN_SCEN), lag_ms=10 * lag, readouts={})
    for r in readouts:
        g = sc.gate_from_counts(C[r], scenarios=scenarios)
        b = sc.gate_from_counts(Cb[r], lag_index=0, bin_index=g["bin_index"], scenarios=scenarios)
        p = sc.gate_from_counts(Cp[r], lag_index=0, scenarios=scenarios)
        R = sc.score_rotation(rot[r], runs)
        m = np.concatenate(restmag[r]) if restmag[r] else np.zeros(1, np.float32)
        out["readouts"][r] = dict(
            gate=g["value"], bin=g["bin"], n=g["n"], baseline=b["value"], lift=round(g["value"] - b["value"], 4),
            fixed_bin_gate=g["by_speed"][FIXED_BIN], fixed_bin_base=b["by_speed"][FIXED_BIN],
            fixed_bin_lift=round(g["by_speed"][FIXED_BIN] - b["by_speed"][FIXED_BIN], 4),
            fixed_bin_n=g["n_by_speed"][FIXED_BIN],
            by_light=g["by_light"], baseline_by_light=b["by_light"], by_scenario=g["by_scenario"],
            by_speed=g["by_speed"], baseline_by_speed=b["by_speed"],
            pooled_19=p["value"], pooled_bin=p["bin"], overall=g["overall"],
            r2_seg={l: R[l]["r2_seg"] for l in R},
            gain_seg={l: {s: R[l][s]["gain_seg"] for s in sc.AXIS} for l in R},
            signs={l: R[l]["signs_right"] for l in R},
            rest_p50=round(float(np.median(m)), 5), rest_p90=round(float(np.quantile(m, 0.9)), 5),
            rest_mean=round(float(m.mean()), 5), rest_n=int(m.size))
    out["seconds"] = round(time.time() - t0, 1)
    return out


# ---------------------------------------------------------------------------------------------- tuning
def score_gratings(name: str, ckpt: Path, device="cuda") -> dict:
    """24-direction DSI and preferred angle at 4 Hz / 8 columns, plus the TF curve."""
    t0 = time.time()
    lat = HexLattice()
    lat.centers = lat.centers.to(device)
    lobe = OpticLobe(ckpt, device=device, dtype=torch.float32)
    idx = gather_index(lobe, MOTION_TYPES)
    rest = grey_rest(lobe, idx)
    u, v = lat.u.numpy(), lat.v.numpy()
    inner = (np.abs(u) + np.abs(v) + np.abs(u + v)) // 2 <= 11
    dirs = np.arange(N_DIRS) * 360.0 / N_DIRS
    r = grating_rates(lat, lobe, rest, dirs, 4.0, 8, 0.5, device=device)
    curves = r[:, :, inner].mean(2)  # (24, 8)
    per = {}
    for j, n in enumerate(MOTION_TYPES):
        ang, amp, dsi, peak, peak_ang = vector_mean(dirs, curves[:, j])
        # the classic preferred-minus-null DSI as well: the same 24 curves, null = 180 deg away
        null = float(curves[(int(np.argmax(curves[:, j])) + N_DIRS // 2) % N_DIRS, j])
        p, q = max(peak, 0.0), max(null, 0.0)
        per[n] = dict(angle_deg=round(ang, 2), amplitude=round(amp, 5), dsi=round(dsi, 3),
                      peak=round(peak, 5), peak_dir_deg=peak_ang, null=round(null, 5),
                      dsi_pref_null=round((p - q) / (p + q), 3) if p + q > 0 else 0.0)
    tf_dirs = np.arange(8) * 45.0
    tf = {}
    for f in TF_HZ:
        c = grating_rates(lat, lobe, rest, tf_dirs, f, 8, 0.5, device=device)[:, :, inner].mean(2)
        amps = [vector_mean(tf_dirs, c[:, j])[1] for j in range(8)]
        tf[str(f)] = dict(per_type=[round(a, 5) for a in amps],
                          t4=round(float(np.sum(amps[:4])), 5), t5=round(float(np.sum(amps[4:])), 5),
                          total=round(float(np.sum(amps)), 5))
        print(f"  {name}: TF {f} Hz total {tf[str(f)]['total']:.4f} ({time.time() - t0:.0f} s)", flush=True)
    tot = np.array([tf[str(f)]["total"] for f in TF_HZ])
    out = dict(name=name, ckpt=str(ckpt), subtypes=per, tf_hz=list(TF_HZ), tf=tf,
               tf_peak_hz=float(TF_HZ[int(np.argmax(tot))]), tf_peak=round(float(tot.max()), 5),
               dsi_min_ab=round(float(min(per[n]["dsi"] for n in ("T4a", "T4b", "T5a", "T5b"))), 3),
               dsi_pn_min_ab=round(float(min(per[n]["dsi_pref_null"] for n in ("T4a", "T4b", "T5a", "T5b"))), 3),
               curves=curves.round(6).tolist(), dirs_deg=dirs.tolist(), seconds=round(time.time() - t0, 1))
    return out


# ---------------------------------------------------------------------------------------------- store
def store(section: str, name: str, value: dict, path: Path = RESULTS):
    path.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(path.read_text()) if path.exists() else {}
    res.setdefault(section, {})[name] = value
    path.write_text(json.dumps(res, indent=1, default=float))
    print(f"wrote {path} [{section}][{name}]")
    return res


def print_row(name, r, key):
    d = r["readouts"][key]
    print(f"{name:26s} {key:4s} gate {d['gate']:.4f} ({d['bin']}) base {d['baseline']:.4f} lift {d['lift']:+.4f} | "
          f"2-4 col/s {d['fixed_bin_gate']:.4f}/{d['fixed_bin_base']:.4f} ({d['fixed_bin_lift']:+.4f}) | "
          f"pool19 {d['pooled_19']:.4f} | rest p50 {d['rest_p50']:.4f} p90 {d['rest_p90']:.4f} | "
          f"R2 {d['r2_seg']['bright']} signs {d['signs']}", flush=True)


# ---------------------------------------------------------------------------------------------- figure
def figure(path: Path = FIGURE, results: Path = RESULTS, logs=()):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = json.loads(results.read_text())
    S, G = res.get("score", {}), res.get("gratings", {})
    runs = {p.parent.name: json.loads(p.read_text()) for p in map(Path, logs) if p.exists()}
    names = list(S)
    fig = plt.figure(figsize=(22, 17))
    gs = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.3, height_ratios=[1, 1, 1.1])

    # 1. loss curves
    ax = fig.add_subplot(gs[0, 0])
    for run, log in runs.items():
        st = log["steps"]
        ax.plot([s["step"] for s in st], [s["loss"] for s in st], lw=1.2, label=f"{run} ({log['loss']})")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss (per window)")
    ax.set_title("Training loss")
    ax.legend(fontsize=7)

    # 2. static response at rest over training
    ax = fig.add_subplot(gs[0, 1])
    for run, log in runs.items():
        st = log["steps"]
        k = "rest_unit_p50" if "rest_unit_p50" in st[0] else "rest_dec_p50"
        ax.plot([s["step"] for s in st], [s[k] for s in st], lw=1.2, label=f"{run} {k[5:]}")
    ax.set_xlabel("step")
    ax.set_ylabel("|field| p50 on a fixed textured rest")
    ax.set_title("Static response during training")
    ax.legend(fontsize=7)

    # 3. gate and baseline on held-out flights
    ax = fig.add_subplot(gs[0, 2:4])
    rows = [(n, r, S[n]["readouts"][r]) for n in names for r in S[n]["readouts"]]
    x = np.arange(len(rows))
    ax.bar(x - 0.19, [d["gate"] for _, _, d in rows], 0.36, color="#0072b2", label="gate (best bin)")
    ax.bar(x + 0.19, [d["fixed_bin_gate"] for _, _, d in rows], 0.36, color="#56b4e9", label="gate (2-4 columns/s)")
    ax.plot(x - 0.19, [d["baseline"] for _, _, d in rows], "_", color="red", ms=16, mew=2.5,
            label="own motion-blind baseline")
    ax.plot(x + 0.19, [d["fixed_bin_base"] for _, _, d in rows], "_", color="#d55e00", ms=16, mew=2.5)
    for xi, (_, _, d) in zip(x, rows):
        ax.text(xi, max(d["gate"], d["fixed_bin_gate"]) + 0.008, f"{d['lift']:+.3f}", ha="center", fontsize=8)
    ax.axhline(0.25, color="k", lw=0.6, ls=":")
    ax.set_xticks(x, [f"{n}\n{r}" for n, r, _ in rows], fontsize=8)
    ax.set_ylabel("fraction within 45 deg")
    ax.set_title("Held-out flights (straight_30, pitch, approach; all lightings): gate vs its own baseline\n"
                 "the number above each pair is the lift in the best bin")
    ax.legend(fontsize=8)

    # 4. lift
    ax = fig.add_subplot(gs[1, 0])
    ax.bar(x, [d["lift"] for _, _, d in rows], 0.6, color="#009e73")
    ax.bar(x, [d["fixed_bin_lift"] for _, _, d in rows], 0.3, color="#004d3a")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(0.25, color="k", lw=0.6, ls="--")
    ax.text(len(rows) - 0.5, 0.255, "proposed +0.25", ha="right", fontsize=8)
    ax.set_xticks(x, [f"{n}\n{r}" for n, r, _ in rows], fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("gate minus own baseline")
    ax.set_title("Lift (wide = best bin, narrow = 2-4 columns/s)")

    # 5. static response at rest, held-out
    ax = fig.add_subplot(gs[1, 1])
    ax.bar(x - 0.19, [d["rest_p50"] for _, _, d in rows], 0.36, color="#444444", label="p50")
    ax.bar(x + 0.19, [d["rest_p90"] for _, _, d in rows], 0.36, color="#999999", label="p90")
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{n}\n{r}" for n, r, _ in rows], fontsize=7, rotation=20, ha="right")
    ax.set_ylabel("|field| at rest, held-out runs")
    ax.set_title("Static response on held-out rests")
    ax.legend(fontsize=8)

    # 6. rotation R^2
    ax = fig.add_subplot(gs[1, 2])
    w = 0.8 / max(1, 3 * len(rows))
    for i, (n, r, d) in enumerate(rows):
        for j, axis in enumerate(("yaw", "pitch", "roll")):
            val = d["r2_seg"]["bright"][axis] * np.sign(d["gain_seg"]["bright"][axis])
            ax.bar(j + (i - len(rows) / 2 + 0.5) * w * 3, val, w * 3, color=plt.cm.viridis(i / max(1, len(rows) - 1)),
                   label=f"{n} {r}" if j == 0 else None)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(3), ["yaw", "pitch", "roll"])
    ax.set_ylabel("R^2 x sign(gain), bright")
    ax.set_title("Rotation, six steady segments")
    ax.legend(fontsize=6)

    # 7. DSI before/after
    ax = fig.add_subplot(gs[1, 3])
    gn = list(G)
    xx = np.arange(8)
    w = 0.8 / max(1, len(gn))
    for i, n in enumerate(gn):
        ax.bar(xx + (i - len(gn) / 2 + 0.5) * w, [G[n]["subtypes"][t]["dsi"] for t in MOTION_TYPES], w,
               color=plt.cm.plasma(i / max(1, len(gn) - 1)), label=n)
    ax.axhline(0.5, color="k", lw=0.8, ls="--")
    ax.set_xticks(xx, MOTION_TYPES, rotation=45, fontsize=8)
    ax.set_ylabel("DSI, 24 directions, 4 Hz, 8 columns")
    ax.set_title("Direction selectivity (dashed = 0.5)")
    ax.legend(fontsize=7)

    # 8. TF curves
    ax = fig.add_subplot(gs[2, 0])
    for i, n in enumerate(gn):
        d = G[n]
        ax.plot(d["tf_hz"], [d["tf"][str(f)]["total"] for f in d["tf_hz"]], "o-", ms=4,
                color=plt.cm.plasma(i / max(1, len(gn) - 1)), label=f"{n} peak {d['tf_peak_hz']} Hz")
    ax.set_xscale("log")
    ax.set_xlabel("temporal frequency (Hz), 8-column gratings")
    ax.set_ylabel("sum of first-harmonic amplitudes, T4 + T5")
    ax.set_title("Temporal-frequency tuning")
    ax.legend(fontsize=7)

    # 9. preferred angles
    ax = fig.add_subplot(gs[2, 1], projection="polar")
    for i, n in enumerate(gn):
        for t in MOTION_TYPES:
            d = G[n]["subtypes"][t]
            ax.annotate("", xy=(math.radians(d["angle_deg"]), d["amplitude"]), xytext=(0, 0),
                        arrowprops=dict(arrowstyle="->", color=plt.cm.plasma(i / max(1, len(gn) - 1)), lw=1.4))
    ax.set_title("Preferred direction and amplitude per subtype\n" + " / ".join(gn), fontsize=9)

    # 10. parameter drift
    ax = fig.add_subplot(gs[2, 2])
    labels, vals = [], []
    for run, log in runs.items():
        s = log["steps"][-1]
        for k, lab in (("tau_max_rel", "tau max |rel|"), ("bias_max", "bias max |d|"),
                       ("strength_rel_l2", "syn L2 / L2(0)"), ("tau_l2", "tau rel L2"), ("bias_l2", "bias L2")):
            labels.append(f"{run}\n{lab}")
            vals.append(s[k])
    ax.bar(np.arange(len(vals)), vals, 0.6, color="#cc79a7")
    ax.set_xticks(np.arange(len(vals)), labels, fontsize=6, rotation=30, ha="right")
    ax.set_ylabel("drift from member 000")
    ax.set_title("Parameter drift per group")

    # 11. the numbers
    ax = fig.add_subplot(gs[2, 3])
    ax.axis("off")
    lines = [f"{'model / readout':30s} {'gate':>6s} {'base':>6s} {'lift':>6s} {'2-4':>6s} {'b2-4':>6s} {'rest50':>7s}"]
    for n, r, d in rows:
        lines.append(f"{(n + ' / ' + r)[:30]:30s} {d['gate']:6.3f} {d['baseline']:6.3f} {d['lift']:+6.3f} "
                     f"{d['fixed_bin_gate']:6.3f} {d['fixed_bin_base']:6.3f} {d['rest_p50']:7.4f}")
    lines += ["", "held out: straight_30, pitch, approach, all lightings; lag 20 ms",
              "base = the same model and readout on the run's own rest, repeated"]
    for n in gn:
        d = G[n]
        lines.append(f"{n[:20]:20s} DSI min(T4a,b,T5a,b) {d['dsi_min_ab']:.2f}  TF peak {d['tf_peak_hz']} Hz")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)

    fig.suptitle("flyeye retraining smoke: the flyvis free parameters refitted on rendered flight "
                 "(Phase 2 recordings, Geiranger)", fontsize=14)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=85, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["score", "gratings", "figure", "table"])
    ap.add_argument("--name", default="model")
    ap.add_argument("--ckpt", type=Path, default=MODEL_NPZ)
    ap.add_argument("--decoder", default=None, choices=[None, "linear", "flyvis"])
    ap.add_argument("--dec-state", type=Path, default=None)
    ap.add_argument("--logs", nargs="*", default=[], help="train log.json paths for the figure")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    if args.what == "score":
        r = score_model(args.name, args.ckpt, args.decoder, args.dec_state, args.device)
        for k in r["readouts"]:
            print_row(args.name, r, k)
        store("score", args.name, r)
    elif args.what == "gratings":
        r = score_gratings(args.name, args.ckpt, args.device)
        print(f"{args.name}: " + ", ".join(f"{t} {d['angle_deg']:+.0f} deg DSI {d['dsi']:.2f}/{d['dsi_pref_null']:.2f} "
                                           f"amp {d['amplitude']:.4f}" for t, d in r["subtypes"].items()))
        print(f"  TF peak {r['tf_peak_hz']} Hz, min DSI over T4a/b, T5a/b {r['dsi_min_ab']} (vector mean), "
              f"{r['dsi_pn_min_ab']} (preferred minus null)")
        store("gratings", args.name, r)
    elif args.what == "figure":
        figure(logs=args.logs)
    else:
        res = json.loads(RESULTS.read_text())
        for n, r in res.get("score", {}).items():
            for k in r["readouts"]:
                print_row(n, r, k)
    return 0


if __name__ == "__main__":
    sys.exit(main())
