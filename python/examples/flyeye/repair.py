"""Offline readout repair on the Phase 2 recordings (member 000): subtype directions, readout and looming variants.

    py -3.14 python/examples/flyeye/repair.py gratings   # T4/T5 subtype directions from procedural gratings (CUDA, ~1 min)
    py -3.14 python/examples/flyeye/repair.py variants   # every readout / looming variant through scoring.py (~15 min)
    py -3.14 python/examples/flyeye/repair.py figure     # figures/repair_member000.png from the saved results

No rendering, no threepp import. Reads C:/dev/_flyeye/phase2/*.npz (t4t5_raw, truth), writes
C:/dev/_flyeye/repair/{subtypes.npz, subtypes.json, variants.json}.

Every readout variant maps t4t5_raw (T, B, 8, 721; relu(v) minus the grey-0.5 rest, T4a..T4d,
T5a..T5d) to a field (T, B, 2, 721) in image x right / y up, so RotationReadout, Looming and
scoring.py apply unchanged. Anything fitted against renderer truth is fitted on the bright runs
only; dim and dark are the test split. Every variant is also scored on its own motion-blind
baseline: the same variant applied to the run's own rest frames 30-100 (ping-pong repeated, with
10 s of the same repeated rest before the scored frames so adaptation has settled).

Variants (member 000 results in C:/dev/_flyeye/repair/variants*.json, figure figures/repair_member000.png):
  V0   unit gains (the Phase 2 field)
  V1a  grating-measured subtype unit vectors, gain 1;  V1b  the same with gain 1 / first-harmonic amplitude
  V2   8 -> 2 least-squares map to the unit true direction, fitted on bright
  V3a  opponent normalisation per pair, (b - a) / (a + b + eps);  V3b  causal EMA adaptation, tau s
  V3c  V3a / V3b combined with V1a, V1b or a V2 refitted on the preprocessed bright runs
Looming on each: L0 plain Looming; L1 on the field minus the RotationReadout (lstsq) templates x estimate;
L2 Looming / (total field RMS + eps); L2rot Looming / (RMS of the templates x estimate field + eps);
eps = median over the bright rests.

Recommended for every member (see recommended()): readout "V3c adapt tau=2.0 + V2" (V2 refitted per
member on its bright runs) and looming "V3b adapt tau=0.5" read out as L2rot:

    from flyeye import repair as rp, scoring as sc
    runs = sc.Runs()
    raws = {key: t4t5_raw_of_member[key] for key in runs.keys()}   # float32 (T, B, 8, 721), same run keys
    readout, loom = rp.recommended(runs, raws)
    row = rp.summarise(rp.score_variant(readout, runs, raws))       # direction, own baseline, rotation
    loom_row = rp.summarise(rp.score_variant(loom, runs, raws))["looming"]["L2rot"]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyeye import scoring  # noqa: E402
from flyeye.lattice import HexLattice  # noqa: E402
from flyeye.readouts import MOTION_TYPES  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = Path(r"C:\dev\_flyeye\repair")
DT = 0.01
LAG = 2  # frames, the Phase 2 gate lag (20 ms)

# ---------------------------------------------------------------------------------------------- 1. gratings
N_DIRS = 24
GRATING = dict(tf_hz=4.0, lambda_columns=8, contrast=0.5)
INNER_RADIUS = 11
DROP_S, RUN_S = 0.5, 2.0


def grating_rates(lat, lobe, rest, dirs_deg, tf_hz, lam_cols, contrast, size=403, device="cuda"):
    """Time-averaged relu minus rest (n_dirs, 8, 721) for full-field drifting sinusoids.

    Luma frame 0.5 (1 + c sin(k . p - 2 pi f t)), p = (col, -row) px about the centre, k along the
    drift direction (image angle, 0 = right, 90 = up), wavelength lam_cols * 13 px. All directions run
    as one batch of eyes: fade_in on the first frame (1 s), DROP_S dropped, RUN_S averaged.
    """
    th = torch.tensor(np.radians(dirs_deg), dtype=torch.float32, device=device)
    rows = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    X, Y = rows[None, :].expand(size, size), -rows[:, None].expand(size, size)  # x = col, y = -row
    kx, ky = torch.cos(th), torch.sin(th)
    proj = kx[:, None, None] * X + ky[:, None, None] * Y  # (D, H, W) px along the drift
    lam_px = lam_cols * 13.0
    idx = torch.stack([lobe.type_index(n) for n in MOTION_TYPES])

    def frame(t):
        return 0.5 * (1 + contrast * torch.sin(2 * math.pi * (proj / lam_px - tf_hz * t)))

    lobe.fade_in(lat.box_eye(frame(0.0)), DT)
    n_drop, n_run = int(round(DROP_S / DT)), int(round(RUN_S / DT))
    acc = torch.zeros(len(dirs_deg), 8, 721, dtype=torch.float64, device=device)
    for s in range(n_drop + n_run):
        v = lobe.step(lat.box_eye(frame((s + 1) * DT)), DT)
        if s >= n_drop:
            acc += (torch.relu(v[:, idx]) - rest).double()
    return (acc / n_run).cpu().numpy()


def vector_mean(dirs_deg, curve):
    """curve (D,) -> (angle deg in (-180, 180], first-harmonic amplitude, DSI = |sum A e^i| / sum |A|, peak, peak angle)."""
    th = np.radians(dirs_deg)
    zc = (curve * np.exp(1j * th)).sum()
    return (float(np.degrees(np.angle(zc))), float(2 * abs(zc) / len(th)), float(abs(zc) / np.abs(curve).sum()),
            float(curve.max()), float(dirs_deg[int(np.argmax(curve))]))


def run_gratings(out=OUT):
    from flyeye.optic_lobe import OpticLobe
    from flyeye.readouts import MotionField

    t0 = time.time()
    dev = "cuda"
    lat = HexLattice()
    lat.centers = lat.centers.to(dev)
    lobe = OpticLobe(device=dev, dtype=torch.float32)
    mf = MotionField(lobe)
    rest = mf.rest  # (8, 721)
    u, v = lat.u.numpy(), lat.v.numpy()
    inner = (np.abs(u) + np.abs(v) + np.abs(u + v)) // 2 <= INNER_RADIUS
    dirs = np.arange(N_DIRS) * 360.0 / N_DIRS
    res = {}
    arrays = dict(dirs_deg=dirs, rest=rest.cpu().numpy(), inner=inner)
    for lam in (GRATING["lambda_columns"], 4):
        r = grating_rates(lat, lobe, rest, dirs, GRATING["tf_hz"], lam, GRATING["contrast"], device=dev)
        curves = r[:, :, inner].mean(2)  # (D, 8)
        arrays[f"curves_l{lam}"] = curves
        arrays[f"percol_l{lam}"] = r.astype(np.float32)
        per = {}
        for j, name in enumerate(MOTION_TYPES):
            ang, amp, dsi, peak, peak_ang = vector_mean(dirs, curves[:, j])
            # per-column vector means over inner columns: spread of the preferred angle
            zc = (r[:, j, :] * np.exp(1j * np.radians(dirs))[:, None]).sum(0)
            col_ang = np.degrees(np.angle(zc[inner]))
            spread = float(np.degrees(np.sqrt(-2 * np.log(max(abs(np.exp(1j * np.radians(col_ang)).mean()), 1e-9)))))
            per[name] = dict(angle_deg=round(ang, 2), amplitude=round(amp, 5), dsi=round(dsi, 3), peak=round(peak, 5),
                             peak_dir_deg=peak_ang, null=round(float(curves[(int(np.argmax(curves[:, j])) + N_DIRS // 2) % N_DIRS, j]), 5),
                             column_angle_circular_sd_deg=round(spread, 1))
        res[f"lambda_{lam}"] = per
    # lattice axes in image angles (x right, y up)
    ev = np.array([13.0, -6.5])  # e_v = (row 6.5, col 13) -> (x 13, y -6.5)
    res["lattice_axes_deg"] = dict(e_u=-90.0, e_v=round(float(np.degrees(np.arctan2(ev[1], ev[0]))), 2), note="neighbour axes: e_u, e_v, e_v - e_u",
                                   e_v_minus_e_u=round(float(np.degrees(np.arctan2(-6.5 + 13.0, 13.0))), 2))
    res["stimulus"] = dict(GRATING, directions=N_DIRS, inner_radius=INNER_RADIUS, drop_s=DROP_S, avg_s=RUN_S,
                           mean_luma=0.5, note="procedural numpy/torch frames 403 px through HexLattice.box_eye + OpticLobe (CUDA float32)")
    res["seconds"] = round(time.time() - t0, 1)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "subtypes.npz", **arrays)
    (out / "subtypes.json").write_text(json.dumps(res, indent=1))
    for lam in (GRATING["lambda_columns"], 4):
        print(f"lambda {lam} columns, 4 Hz, c 0.5")
        for n, d in res[f"lambda_{lam}"].items():
            print(f"  {n}: angle {d['angle_deg']:+7.1f} deg  amp {d['amplitude']:.4f}  dsi {d['dsi']:.2f}  peak {d['peak']:.4f} at {d['peak_dir_deg']:.0f}"
                  f"  null {d['null']:+.4f}  column sd {d['column_angle_circular_sd_deg']} deg")
    print(res["lattice_axes_deg"], f"{res['seconds']} s")
    return res


def load_subtypes(out=OUT, lam=GRATING["lambda_columns"]):
    """(unit vectors (8, 2) x right / y up, amplitudes (8,)) from subtypes.json."""
    d = json.loads((out / "subtypes.json").read_text())[f"lambda_{lam}"]
    ang = np.radians([d[n]["angle_deg"] for n in MOTION_TYPES])
    return np.stack([np.cos(ang), np.sin(ang)], 1), np.array([d[n]["amplitude"] for n in MOTION_TYPES])


# ---------------------------------------------------------------------------------------------- 2. readout variants
UNIT = np.array([[-1, 0], [1, 0], [0, 1], [0, -1]] * 2, dtype=np.float32)  # MotionField unit gains, (8, 2)
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7)]  # opponent pairs a/b, c/d for T4 and T5
SETTLE = 1000  # frames of repeated rest before the scored baseline frames (10 s)


def grey_rest(out=OUT) -> np.ndarray:
    """(8, 721) relu rate at uniform grey 0.5 (MotionField.rest), saved by the gratings step."""
    return np.load(out / "subtypes.npz")["rest"].astype(np.float32)


def map_field(raw: np.ndarray, U: np.ndarray, gains=None) -> np.ndarray:
    """raw (T, B, 8, N) -> field (T, B, 2, N) = sum_k g_k raw_k U_k; U (8, 2) x right / y up."""
    W = U * (1.0 if gains is None else np.asarray(gains, dtype=np.float32)[:, None])
    return np.einsum("tbkn,kc->tbcn", raw, W.astype(np.float32))


def opponent_norm(raw: np.ndarray, rest: np.ndarray, eps: float) -> np.ndarray:
    """Per column, per opponent pair (a, b): rate_a / (rate_a + rate_b + eps) minus the same at grey.

    rate = raw + rest (the relu rate). With the unit map, x = (b - a) / (a + b + eps) minus its grey
    value, per T4 and T5 pair. Stateless."""
    rate = np.maximum(raw + rest, 0)
    out = np.empty_like(raw)
    for i, j in PAIRS:
        s = rate[:, :, i] + rate[:, :, j] + eps
        s0 = rest[i] + rest[j] + eps
        out[:, :, i] = rate[:, :, i] / s - rest[i] / s0
        out[:, :, j] = rate[:, :, j] / s - rest[j] / s0
    return out


def adapt(raw: np.ndarray, tau_s: float, dt: float = DT, device="cuda") -> np.ndarray:
    """Causal subtractive adaptation per column and subtype: raw(t) - m(t),
    m(t) = m(t-1) + dt/tau (raw(t) - m(t-1)), m(0) = raw(0). No future frames."""
    x = torch.as_tensor(raw, device=device)
    m = x[0].clone()
    a = dt / tau_s
    out = torch.empty_like(x)
    for t in range(len(x)):
        m += a * (x[t] - m)
        out[t] = x[t] - m
    return out.cpu().numpy()


def rest_input(raw: np.ndarray, settle=SETTLE) -> np.ndarray:
    """Motion-blind input: frames 30-100 of the first rest, ping-pong repeated for settle + T frames."""
    r = raw[scoring.REST]
    cyc = np.concatenate([r, r[::-1]])
    return cyc[np.arange(settle + len(raw)) % len(cyc)]


class Variant:
    """A readout: pre (raw -> raw', may be stateful) then a direction map U (8, 2) with gains."""

    def __init__(self, name, U=UNIT, gains=None, pre=None, note=""):
        self.name, self.U, self.gains, self.pre, self.note = name, U, gains, pre, note
        self.fitted = None

    def field(self, raw):
        return map_field(self.pre(raw) if self.pre else raw, self.U, self.gains)

    def baseline(self, raw):
        """The variant on the run's rest frames repeated; the settle frames dropped."""
        return self.field(rest_input(raw))[SETTLE:]


def fit_linear(runs, raws, pre=None, lights=("bright",), lag=LAG, stride=2):
    """Least-squares 8 -> 2 map (8, 2) from (pre)raw to the unit true direction: scene columns,
    true speed >= MIN_SPEED, frames >= SKIP into their segment, circuit lag frames behind truth,
    every stride-th frame, the given lightings only."""
    XtX, XtY, n = np.zeros((8, 8)), np.zeros((8, 2)), 0
    for key in runs.keys(lights):
        run = runs[key]
        raw = raws[key]
        raw = pre(raw) if pre else raw
        k = np.nonzero((run.into >= scoring.SKIP) & (np.arange(run.T) >= lag))[0][::stride]
        kt = k - lag
        ang, sp, _ = run.truth(False)
        sel = (sp[kt] >= scoring.MIN_SPEED) & run.scene[kt]  # (K, B, N)
        X = np.moveaxis(raw[k], 2, -1)[sel].astype(np.float64)  # (n, 8)
        a = ang[kt][sel].astype(np.float64)
        XtX += X.T @ X
        XtY += X.T @ np.stack([np.cos(a), np.sin(a)], 1)
        n += len(X)
    return np.linalg.solve(XtX, XtY).astype(np.float32), n


class Raws:
    """Cache of float32 t4t5_raw per run."""

    def __init__(self, runs):
        self.runs, self._c = runs, {}

    def __getitem__(self, key):
        if key not in self._c:
            self._c[key] = self.runs[key].raw()
        return self._c[key]


_ROT = {}


def rot_projection(field, run):
    """(T, B, 2, N) -> (w (T, 3) lstsq estimate, the field of that rotation (T, B, 2, N): templates x estimate)."""
    from flyeye.readouts import RotationReadout

    m = run.meta
    if "lstsq" not in _ROT:
        _ROT["lstsq"] = RotationReadout(m["size"], m["fov_deg"], m["eye_rotations"], method="lstsq")
    ro = _ROT["lstsq"]
    f = torch.as_tensor(field, dtype=torch.float32, device="cuda")
    w = ro(f)
    rot = torch.einsum("kbcn,tk->tbcn", ro._on("templates", f), w)
    return w.cpu().numpy(), rot.cpu().numpy()


def score_variant(v: Variant, runs, raws, looming=True):
    """Direction (lag 20 ms, own baseline), rotation (matched) and looming (L0 plain, L1 rotation-gated,
    L2 divisive by total field RMS, L2rot divisive by the rotation-field RMS) of one variant, 21 runs."""
    t0 = time.time()
    fields, base = {}, {}
    rot_series, loom, energy = {}, {"L0": {}, "L1": {}, "L2": {}, "L2rot": {}}, {}
    for key in runs.keys():
        run, raw = runs[key], raws[key]
        f = v.field(raw)
        fields[key] = f
        base[key] = v.baseline(raw)
        if key[0] in scoring.AXIS:
            rot_series[key] = scoring.field_rotation(f, run)
        if looming:
            loom["L0"][key] = scoring.field_looming(f, run)
            w, frot = rot_projection(f, run)
            loom["L1"][key] = scoring.field_looming(f - frot, run)
            energy[key] = (np.sqrt((f ** 2).mean((1, 2, 3))), np.sqrt((frot ** 2).mean((1, 2, 3))))
    D = scoring.score_direction(fields, runs, baseline=base, pooled=True, lag=LAG)
    del fields, base
    R = scoring.score_rotation(rot_series, runs)
    out = dict(name=v.name, note=v.note, direction=D, rotation=R)
    if looming:
        # eps = median of the energy over the bright runs' rest frames 30-100 (the fitting split)
        e_all = np.concatenate([energy[(s, "bright")][0][scoring.REST] for s in runs.scenarios])
        e_rot = np.concatenate([energy[(s, "bright")][1][scoring.REST] for s in runs.scenarios])
        eps_all, eps_rot = float(np.median(e_all)), float(np.median(e_rot))
        for key in runs.keys():
            loom["L2"][key] = loom["L0"][key] / (energy[key][0] + eps_all)
            loom["L2rot"][key] = loom["L0"][key] / (energy[key][1] + eps_rot)
        out["loom_eps"] = dict(total_rms=eps_all, rotation_rms=eps_rot)
        out["looming"] = {}
        for name, series in loom.items():
            S = scoring.score_looming(series, runs)
            for l in S:
                S[l].pop("_series")
            out["looming"][name] = S
    out["seconds"] = round(time.time() - t0, 1)
    return out


def summarise(res: dict) -> dict:
    """The comparison row of one variant."""
    D, R = res["direction"], res["rotation"]
    g, b, p = D["gate"], D["baseline"], D["pooled_19"]

    def test(d):
        n = d["n_by_light"]
        return round((d["by_light"]["dim"] * n["dim"] + d["by_light"]["dark"] * n["dark"]) / max(n["dim"] + n["dark"], 1), 4)

    row = dict(name=res["name"], note=res["note"], gate=g["value"], bin=g["bin"], n=g["n"], baseline=b["value"], lift=D["lift"],
               train_bright=g["by_light"]["bright"], test_dim=g["by_light"]["dim"], test_dark=g["by_light"]["dark"],
               test_dim_dark=test(g), baseline_by_light=b["by_light"], baseline_test=test(b),
               pooled_19=p["value"], pooled_bin=p["bin"], pooled_by_light=p["by_light"], overall=g["overall"],
               overall_by_light=g["overall_by_light"], by_speed=g["by_speed"], baseline_by_speed=b["by_speed"],
               r2_seg={l: R[l]["r2_seg"] for l in R}, r2_ts={l: {s: R[l][s]["r2_ts"] for s in scoring.AXIS} for l in R},
               signs={l: R[l]["signs_right"] for l in R},
               gain_seg={l: {s: R[l][s]["gain_seg"] for s in scoring.AXIS} for l in R},
               gain_by_rate={l: {s: R[l][s]["gain_by_rate"] for s in scoring.AXIS} for l in R},
               cross_bright=R["bright"]["cross_talk_row_normalised"])
    if "looming" in res:
        row["looming"] = {name: {l: dict(peak=S[l]["approach"]["peak"], corr=S[l]["approach"]["corr_lag0"],
                                         threshold=S[l]["threshold"], rot_fa=S[l]["rotation_false_alarm_max"],
                                         margin_peak=S[l]["margin_peak"], margin_last=S[l]["margin_last"],
                                         margin_with_straight=S[l]["margin_peak_with_straight"],
                                         straight_max=max(S[l]["false_alarm"][s]["max"] for s in ("straight_3", "straight_10", "straight_30")))
                                  for l in S} for name, S in res["looming"].items()}
    return row


def print_row(r):
    print(f"{r['name']:30s} gate {r['gate']:.4f} ({r['bin']}) base {r['baseline']:.4f} lift {r['lift']:+.4f} | "
          f"bright {r['train_bright']:.4f} dim {r['test_dim']:.4f} dark {r['test_dark']:.4f} | pooled19 {r['pooled_19']:.4f} | "
          f"R2 {r['r2_seg']} signs {r['signs']}")
    for name, d in r.get("looming", {}).items():
        print(f"      {name:6s} " + "  ".join(f"{l}: peak {x['peak']:.4f} rotFA {x['rot_fa']:.4f} margin {x['margin_peak']} "
                                             f"(w/ straight {x['margin_with_straight']}) r {x['corr']}" for l, x in d.items()))
    sys.stdout.flush()


def variant_list(runs, raws, out=OUT):
    """Name -> factory of every variant (fits run lazily, bright only)."""
    rest = grey_rest(out)
    U1, amp = load_subtypes(out)
    V = {}
    V["V0 unit"] = lambda: Variant("V0 unit", note="MotionField unit gains (the Phase 2 field, from t4t5_raw)")
    V["V1a grating dirs g=1"] = lambda: Variant("V1a grating dirs g=1", U=U1, note="grating-measured unit vectors, gain 1")
    V["V1b grating dirs g=1/amp"] = lambda: Variant("V1b grating dirs g=1/amp", U=U1, gains=1 / amp,
                                                    note="grating-measured unit vectors, gain 1 / first-harmonic amplitude")

    def v2():
        W, n = fit_linear(runs, raws)
        v = Variant("V2 lstsq 8->2", U=W, note=f"8->2 map, least squares to the unit true direction on bright (N {n})")
        v.fitted = dict(W=W.tolist(), n=n, angles_deg=np.degrees(np.arctan2(W[:, 1], W[:, 0])).round(1).tolist(),
                        norms=np.hypot(W[:, 0], W[:, 1]).round(4).tolist())
        return v

    V["V2 lstsq 8->2"] = v2
    for eps in (0.01, 0.05, 0.2):
        V[f"V3a opp eps={eps}"] = lambda e=eps: Variant(f"V3a opp eps={e}", pre=lambda r: opponent_norm(r, rest, e),
                                                        note="opponent normalisation per pair, unit map")
    for tau in (0.25, 0.5, 1.0, 2.0):
        V[f"V3b adapt tau={tau}"] = lambda t=tau: Variant(f"V3b adapt tau={t}", pre=lambda r: adapt(r, t),
                                                          note="causal subtractive adaptation (EMA), unit map")

    # V3c: a static-pattern removal combined with V1 or V2 (V2 refitted on the preprocessed bright runs)
    def v3c_fit(name, pre, note):
        def make():
            W, n = fit_linear(runs, raws, pre=pre)
            v = Variant(name, U=W, pre=pre, note=f"{note}; 8->2 map refitted on bright (N {n})")
            v.fitted = dict(W=W.tolist(), n=n, angles_deg=np.degrees(np.arctan2(W[:, 1], W[:, 0])).round(1).tolist(),
                            norms=np.hypot(W[:, 0], W[:, 1]).round(4).tolist())
            return v
        return make

    for tau in (0.5, 1.0, 2.0):
        V[f"V3c adapt tau={tau} + V2"] = v3c_fit(f"V3c adapt tau={tau} + V2", lambda r, t=tau: adapt(r, t),
                                                 f"adaptation tau {tau} s")
        V[f"V3c adapt tau={tau} + V1a"] = lambda t=tau: Variant(f"V3c adapt tau={t} + V1a", U=U1, pre=lambda r: adapt(r, t),
                                                                note=f"adaptation tau {t} s, grating unit vectors")
        V[f"V3c adapt tau={tau} + V1b"] = lambda t=tau: Variant(f"V3c adapt tau={t} + V1b", U=U1, gains=1 / amp, pre=lambda r: adapt(r, t),
                                                                note=f"adaptation tau {t} s, grating unit vectors, gain 1/amplitude")
    for eps in (0.01, 0.05, 0.2):
        V[f"V3c opp eps={eps} + V2"] = v3c_fit(f"V3c opp eps={eps} + V2", lambda r, e=eps: opponent_norm(r, rest, e),
                                               f"opponent normalisation eps {eps}")
    return V, rest, U1, amp


def run_variants(out=OUT, only=None, names=None, json_name="variants.json"):
    runs = scoring.Runs()
    raws = Raws(runs)
    V, rest, U1, amp = variant_list(runs, raws, out)
    path = out / json_name
    results = json.loads(path.read_text()) if path.exists() else {}
    for name in names or list(V):
        if only and name not in only:
            continue
        v = V[name]()
        res = score_variant(v, runs, raws, looming=True)
        row = summarise(res)
        results[name] = dict(row=row, fitted=v.fitted, loom_eps=res.get("loom_eps"),
                             looming_full={k: {l: {kk: vv for kk, vv in d.items() if kk in ("false_alarm", "approach", "threshold",
                                                                                            "threshold_with_straight", "detect")}
                                                   for l, d in S.items()} for k, S in res.get("looming", {}).items()},
                             rotation_full=res["rotation"])
        path.write_text(json.dumps(results, indent=1, default=float))
        print_row(row)
        print(f"      {res['seconds']} s", flush=True)
    return results


def recommended(runs, raws, readout_tau=2.0, loom_tau=0.5):
    """(readout Variant, looming Variant) recommended by the member-000 repair round.

    readout: causal adaptation tau readout_tau s, then an 8 -> 2 map fitted by fit_linear on the bright runs
    of these raws (so each member gets its own map); looming: adaptation tau loom_tau s with the unit map,
    read from score_variant(...)["looming"]["L2rot"] (Looming divided by the rotation-field RMS + eps)."""
    pre = lambda r: adapt(r, readout_tau)
    W, n = fit_linear(runs, raws, pre=pre)
    readout = Variant(f"V3c adapt tau={readout_tau} + V2", U=W, pre=pre,
                      note=f"adaptation tau {readout_tau} s; 8->2 map refitted on bright (N {n})")
    readout.fitted = dict(W=W.tolist(), n=n)
    loom = Variant(f"V3b adapt tau={loom_tau}", pre=lambda r: adapt(r, loom_tau), note="causal adaptation, unit map; use L2rot")
    return readout, loom


# ---------------------------------------------------------------------------------------------- 3. figure
LIGHTS = scoring.LIGHTS
LCOL = dict(bright="#e69f00", dim="#0072b2", dark="#444444")
LOOM_NAMES = dict(L0="L0 plain", L1="L1 rotation-gated", L2="L2 / total RMS", L2rot="L2 / rotation RMS")


def load_results(out=OUT):
    """Merge variants*.json (later files win) into {name: entry}, in a stable order."""
    res = {}
    for p in sorted(out.glob("variants*.json")):
        res.update(json.loads(p.read_text()))
    order = lambda n: (n.split()[0], n)
    return {k: res[k] for k in sorted(res, key=order)}


def figure(out=OUT, path=HERE / "figures" / "repair_member000.png"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = json.loads((out / "subtypes.json").read_text())
    arr = np.load(out / "subtypes.npz")
    dirs = arr["dirs_deg"]
    res = load_results(out)
    names = list(res)
    rows = [res[n]["row"] for n in names]
    fig = plt.figure(figsize=(30, 30))
    gs = fig.add_gridspec(4, 8, height_ratios=[0.8, 0.8, 1.3, 1.9], hspace=0.45, wspace=0.35)
    th = np.radians(np.append(dirs, dirs[0]))
    axes_deg = [90.0, sub["lattice_axes_deg"]["e_v"], sub["lattice_axes_deg"]["e_v_minus_e_u"]]
    for j, name in enumerate(MOTION_TYPES):
        ax = fig.add_subplot(gs[j // 4, j % 4], projection="polar")
        for lam, ls in ((8, "-"), (4, ":")):
            c = arr[f"curves_l{lam}"][:, j]
            ax.plot(th, np.append(c, c[0]), ls, color="#222222" if lam == 8 else "#888888", lw=1.4, label=f"{lam} col")
        rmax = max(arr["curves_l8"][:, j].max(), arr["curves_l4"][:, j].max(), 1e-3) * 1.1
        for a in axes_deg:
            for s in (0, 180):
                ax.plot([np.radians(a + s)] * 2, [0, rmax], color="#56b4e9", lw=0.7, ls="--")
        d8 = sub["lambda_8"][name]
        ax.annotate("", xy=(np.radians(d8["angle_deg"]), rmax * 0.95), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#d55e00", lw=2))
        ax.set_ylim(min(0, arr["curves_l8"][:, j].min(), arr["curves_l4"][:, j].min()), rmax)
        ax.set_xticks(np.radians([0, 90, 180, 270]), ["R", "U", "L", "D"], fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_title(f"{name}: {d8['angle_deg']:+.0f} deg, amp {d8['amplitude']:.3f}, DSI {d8['dsi']:.2f}\n"
                     f"(4 col: {sub['lambda_4'][name]['angle_deg']:+.0f} deg)", fontsize=9)
        if j == 0:
            ax.legend(fontsize=7, loc="lower left", bbox_to_anchor=(-0.35, -0.2))
    ax = fig.add_subplot(gs[0:2, 4:8])
    ax.axis("off")
    lines = ["Subtype tuning: procedural drifting sinusoids, 4 Hz, contrast 0.5, 24 directions,",
             "time-averaged relu minus grey rest, inner columns (hex radius <= 11).",
             "Black = 8-column wavelength (the fit), grey dotted = 4 columns;",
             "orange arrow = vector-mean preferred direction at 8 columns;",
             f"blue dashed = hex lattice neighbour axes ({axes_deg[0]:.0f}, {axes_deg[1]:+.1f}, {axes_deg[2]:+.1f} deg).",
             "Image angles: 0 = right, 90 = up.", ""]
    lines.append(f"{'variant':28s} {'gate':>6s} {'base':>6s} {'lift':>6s} {'brt':>6s} {'dim':>6s} {'dark':>6s} {'pool19':>6s}")
    for r in rows:
        lines.append(f"{r['name'][:28]:28s} {r['gate']:6.3f} {r['baseline']:6.3f} {r['lift']:+6.3f} {r['train_bright']:6.3f} "
                     f"{r['test_dim']:6.3f} {r['test_dark']:6.3f} {r['pooled_19']:6.3f}")
    lines += ["", "gate: fraction within 45 deg, scene columns, best speed bin (N >= 20k), lag 20 ms;",
              "base: the same variant on the run's own rest frames repeated (settled);",
              "V2 is fitted on bright only: bright = train, dim/dark = test."]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9.5)

    x = np.arange(len(rows))
    w = 0.26
    ax = fig.add_subplot(gs[2, 0:4])
    for i, l in enumerate(LIGHTS):
        vals = [r["train_bright"] if l == "bright" else r[f"test_{l}"] for r in rows]
        bars = ax.bar(x + (i - 1) * w, vals, w, color=LCOL[l], label=f"{l} ({'train' if l == 'bright' else 'test'})",
                      hatch=None if l == "bright" else "//", edgecolor="white", lw=0)
        base = [r["baseline_by_light"][l] for r in rows]
        ax.plot(x + (i - 1) * w, base, "_", color="red", ms=14, mew=2, label="own motion-blind baseline" if i == 0 else None)
    ax.plot(x, [r["pooled_19"] for r in rows], "D", color="#56b4e9", ms=6, label="pooled-19 (all lights)")
    ax.axhline(0.25, color="k", lw=0.6, ls=":")
    ax.axhline(0.362, color="k", lw=0.6, ls="--")
    ax.text(len(rows) - 0.5, 0.365, "Phase 2 gate 0.362", ha="right", fontsize=8)
    ax.set_xticks(x, [r["name"] for r in rows], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0.15, 0.66)
    ax.set_ylabel("fraction within 45 deg, best speed bin")
    ax.set_title("Direction gate per variant and lighting (bars) vs its own motion-blind baseline (red ticks)")
    ax.legend(fontsize=8, ncol=3, loc="upper left")

    ax = fig.add_subplot(gs[2, 4:8])
    ax.bar(x, [r["lift"] for r in rows], 0.6, color="#009e73")
    for xi, r in zip(x, rows):
        ax.text(xi, r["lift"] + 0.002 * np.sign(r["lift"] or 1), f"{r['lift']:+.3f}", ha="center", fontsize=8,
                va="bottom" if r["lift"] >= 0 else "top")
    ax.axhline(0.0429, color="k", lw=0.6, ls="--")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x, [r["name"] for r in rows], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("gate minus own baseline (all lights)")
    ax.set_title("Lift over the motion-blind baseline (dashed = Phase 2, 0.043)")

    ax = fig.add_subplot(gs[3, 0:4])
    ww = 0.09
    for ai, axis in enumerate(("yaw", "pitch", "roll")):
        for i, l in enumerate(LIGHTS):
            off = (ai * 3 + i - 4) * ww
            vals = [r["r2_seg"][l][axis] * np.sign(r["gain_seg"][l][axis]) for r in rows]
            ax.bar(x + off, vals, ww, color=LCOL[l], alpha=[1.0, 0.6, 0.3][ai], label=f"{axis} {l}")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(0.7, color="k", lw=0.5, ls=":")
    ax.set_xticks(x, [r["name"] for r in rows], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("R^2 x sign(gain), six steady-segment means vs rate")
    ax.set_title("Rotation: signed R^2 per axis (yaw solid | pitch mid | roll pale; each bright, dim, dark); below 0 = wrong-sign gain")
    ax.legend(fontsize=7, ncol=9, loc="upper center", bbox_to_anchor=(0.5, -0.36))

    ax = fig.add_subplot(gs[3, 4:8])
    lnames = list(LOOM_NAMES)
    M = np.array([[r["looming"][ln][l]["margin_peak"] if "looming" in r else np.nan for ln in lnames for l in LIGHTS] for r in rows],
                 dtype=float)
    im = ax.imshow(np.log2(np.clip(M, 0.05, None)), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    fontweight="bold" if M[i, j] > 1 else "normal")
    ax.set_xticks(range(M.shape[1]), [f"{ln} {l}" for ln in lnames for l in LIGHTS], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows)), [r["name"] for r in rows], fontsize=8)
    for j in (3, 6, 9):
        ax.axvline(j - 0.5, color="k", lw=1)
    ax.set_title("Looming margin = approach peak / max false alarm (rotation runs + approach rest), per lighting\n"
                 "L0 plain, L1 rotation-gated (field minus templates x lstsq estimate), L2 / (total RMS + eps), "
                 "L2rot / (rotation-field RMS + eps); red > 1; Phase 2 = 0.30 / 0.11 / 0", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, label="log2 margin")
    fig.suptitle("flyeye member 000: readout repair on the Phase 2 recordings (21 runs, Geiranger)", fontsize=15)
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=80, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["gratings", "variants", "figure"])
    ap.add_argument("--only", nargs="*", help="variant names to (re)score; the rest are kept from the json")
    ap.add_argument("--json", default="variants.json")
    args = ap.parse_args(argv)
    if args.what == "gratings":
        run_gratings()
    elif args.what == "variants":
        run_variants(only=args.only, json_name=args.json)
    elif args.what == "figure":
        figure()


if __name__ == "__main__":
    main()
