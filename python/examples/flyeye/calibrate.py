"""Phase 2 figures a, c, d: the circuit against the renderer's ground truth on the recorded scenarios.

    py -3.14 python/examples/flyeye/calibrate.py        # reads C:/dev/_flyeye/phase2/*.npz (scenarios.py), about 2 min

CPU only, no rendering. Writes <data>/calibrate.json and figures/phase2_{direction,rotation,looming}.png.

Direction (figure a). Truth is flow_box (box-averaged angular flow on the column's unit
image-right/up tangents), mapped to the pixel velocity of the column centre (px/s, x right,
y up): the tangents are up to 18 deg off orthogonal in the corners, and the circuit's x/y
are pixel axes of the lattice. Speed is |pixel velocity| / 13 px, in columns/s.
- A column-frame counts when its true speed is at least MIN_SPEED columns/s and the circuit
  frame is at least SKIP frames into its segment (response latency; also drops fade-in).
- Hit: the circuit vector (MotionField, unit gains; T4-only is b-a, c-d on T4a-d, T5-only on
  T5a-d) within 45 deg of the true direction at frame k - lag. The lag (0..60 ms) that
  maximises the pooled T4+T5 accuracy is used everywhere.
- "scene" columns see geometry (sky_frac < 0.5); sky has no texture to see motion on.
- Time-averaged: per steady rotation segment, the direction of the column's mean field
  against its mean true velocity (the grating figure's time-averaged vector).
- Gate: accuracy of scene columns in the best speed bin (N >= GATE_MIN_N), all lightings and
  scenarios pooled, per frame. The equivalent TF is speed / wavelength for the receptor
  images' power-weighted mean spatial wavelength along lattice columns (lambda_eff).

Rotation (figure c). rot_matched and rot_lstsq against w_body on the yaw, pitch and roll runs.
Offset = the mean over frames 30-100 of the first rest. Per axis: a lagged linear fit over the
whole run (lag 0..300 ms, best R^2), steady-segment means (first SKIP frames dropped) per rate,
their gain per rad/s and R^2 against the rate, and the 3x3 cross-talk (row = true axis,
column = readout; least-squares slope of the six segment means through the origin). Straight
flights: the rest-subtracted readout in steady flight against the truth templates.

Looming (figure d). loom against 1/tau_depth on the approach (lag 0..300 ms, best r), binned
means and the saturation point (first 1/tau bin reaching 90 % of the largest bin mean);
false alarms = loom on the rotation runs and the approach rest from frame 30 on (straight
flights close on the wall at 1/tau 0.007-0.09 and are reported apart); threshold = the
largest false alarm per lighting, and the 1/tau from which the steady approach stays above it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flyeye.lattice import HexLattice  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = Path(r"C:\dev\_flyeye\phase2")
SCENARIOS = ("straight_3", "straight_10", "straight_30", "yaw", "pitch", "roll", "approach")
LIGHTS = ("bright", "dim", "dark")
VARIANTS = ("T4+T5", "T4", "T5", "pooled 19", "static")
SKIP = 30  # 0.3 s at 100 Hz
MIN_SPEED = 0.25  # columns/s
LAGS = range(7)  # frames, 0..60 ms
SPEED_EDGES = 2.0 ** np.arange(-2, 9)  # 0.25 .. 256 columns/s, 10 bins
CONTRAST_EDGES = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0001])
GATE_MIN_N = 20000
PX_PER_COLUMN = 13.0
AXIS = dict(pitch=0, yaw=1, roll=2)
AXIS_NAMES = ("wx pitch", "wy yaw", "wz roll")
GRATING_ACC = {4: 1.00, 8: 1.00, 16: 1.00}  # time-averaged, best TF 4 Hz (tuning.py report)
GRATING_ACC_FRAME = {4: 0.69, 8: 0.78, 16: 0.70}


def segments(z):
    """[(name, first, end)] from the labels."""
    out = []
    for s in z["labels"]:
        name, a, b = str(s).split("|")
        out.append((name, int(a), int(b)))
    return out


def into_segment(z) -> np.ndarray:
    """(T,) frames since the start of the frame's segment."""
    k = np.arange(len(z["t"]))
    first = np.array([a for _, a, _ in segments(z)])[z["seg"].astype(int)]
    return k - first


def pixel_map(lat: HexLattice, size=403, fov=90.0) -> np.ndarray:
    """(721, 2, 2): tangent projections (right, up) of angular flow, rad/s -> pixel velocity px/s (x right, y up)."""
    d = lat.column_rays(size, fov, torch.float64).numpy()
    T = lat.column_tangents(size, fov, torch.float64).numpy()  # (721, 2, 3)
    f = size / 2 / math.tan(math.radians(fov) / 2)
    J = np.zeros((len(d), 2, 3))
    J[:, 0, 0] = J[:, 1, 1] = f / -d[:, 2]
    J[:, 0, 2] = f * d[:, 0] / d[:, 2] ** 2
    J[:, 1, 2] = f * d[:, 1] / d[:, 2] ** 2
    G = np.einsum("nki,nli->nkl", T, T)
    return np.einsum("nki,nji,njl->nkl", J, T, np.linalg.inv(G))


def hex_pool(lat: HexLattice, radius: int) -> np.ndarray:
    """(721, 721) row-normalised mean over the columns within hex distance radius."""
    u, v = lat.u.numpy(), lat.v.numpy()
    du, dv = u[:, None] - u[None], v[:, None] - v[None]
    A = ((np.abs(du) + np.abs(dv) + np.abs(du + dv)) // 2 <= radius).astype(np.float32)
    return A / A.sum(1, keepdims=True)


def circuit_fields(z) -> dict:
    raw = z["t4t5_raw"].astype(np.float32)  # (T, B, 8, 721)
    t4 = np.stack([raw[:, :, 1] - raw[:, :, 0], raw[:, :, 2] - raw[:, :, 3]], axis=2)
    t5 = np.stack([raw[:, :, 5] - raw[:, :, 4], raw[:, :, 6] - raw[:, :, 7]], axis=2)
    return {"T4+T5": z["field"].astype(np.float32), "T4": t4, "T5": t5}


def angle(v):
    return np.arctan2(v[:, :, 1], v[:, :, 0])  # (T, B, 721)


def wrap_deg(a):
    return np.degrees((a + np.pi) % (2 * np.pi) - np.pi)


def load(scen, light):
    return np.load(DATA / f"{scen}_{light}.npz"), json.loads((DATA / f"{scen}_{light}.json").read_text())


# ---------------------------------------------------------------------------------------------- direction
def direction(lat, M):
    nS, nC, nL = len(SPEED_EDGES) - 1, len(CONTRAST_EDGES) - 1, len(LAGS)
    shape = (len(LIGHTS), len(SCENARIOS), len(VARIANTS), nL, 2, nS, nC)
    hits, total = np.zeros(shape), np.zeros(shape)
    hist_edges = np.linspace(-180, 180, 73)
    err_hist = np.zeros((len(LIGHTS), len(VARIANTS), nL, 2, 72))
    maps = {}  # (scen, light, seg) -> per-lag (B, 721) hits and totals
    tavg = []  # time-averaged per steady rotation segment
    spec = {l: [] for l in LIGHTS}
    MAP_SEGS = {("yaw", "bright"): "yaw +90", ("straight_30", "bright"): "fly 30 m/s"}
    vlines = [np.nonzero(lat.v.numpy() == v)[0] for v in range(-6, 7)]
    P = hex_pool(lat, 2)
    for li, light in enumerate(LIGHTS):
        for si, scen in enumerate(SCENARIOS):
            z, _ = load(scen, light)
            T = len(z["t"])
            vel = np.einsum("nkl,tbln->tbkn", M, z["flow_box"].astype(np.float64)).astype(np.float32)
            fields = circuit_fields(z)
            fields["static"] = np.broadcast_to(fields["T4+T5"][30:100].mean(0), fields["T4+T5"].shape)
            fields["pooled 19"] = np.einsum("mn,tbkn->tbkm", P, fields["T4+T5"])
            vel_pooled = np.einsum("mn,tbkn->tbkm", P, vel)
            contrast = z["contrast"].astype(np.float32)
            scene = z["sky_frac"].astype(np.float32) < 0.5
            into = into_segment(z)
            cbin = np.clip(np.digitize(contrast, CONTRAST_EDGES) - 1, 0, nC - 1)
            truth = {}
            for name, v in (("column", vel), ("pooled", vel_pooled)):
                sp = np.hypot(v[:, :, 0], v[:, :, 1]) / PX_PER_COLUMN  # (T, B, 721)
                truth[name] = (angle(v), sp, np.clip(np.digitize(sp, SPEED_EDGES) - 1, 0, nS - 1))
            speed = truth["column"][1]
            seg_names = [s[0] for s in segments(z)]
            for lag in LAGS:
                k = np.nonzero((into >= SKIP) & (np.arange(T) >= lag))[0]
                kt = k - lag
                for vi, var in enumerate(VARIANTS):
                    ang_t, sp, sbin = truth["pooled" if var == "pooled 19" else "column"]
                    ok = sp[kt] >= MIN_SPEED
                    idx = ((scene[kt].astype(np.int64) * nS + sbin[kt]) * nC + cbin[kt])[ok]
                    err = wrap_deg(angle(fields[var][k]) - ang_t[kt])
                    hit = np.abs(err) < 45
                    base = (li, si, vi, lag)
                    hits[base] += np.bincount(idx, weights=hit[ok], minlength=2 * nS * nC).reshape(2, nS, nC)
                    total[base] += np.bincount(idx, minlength=2 * nS * nC).reshape(2, nS, nC)
                    for g in (0, 1):
                        sel = ok & (scene[kt] == bool(g))
                        err_hist[li, vi, lag, g] += np.histogram(err[sel], hist_edges)[0]
                    want = MAP_SEGS.get((scen, light))
                    if var == "T4+T5" and want:
                        m = z["seg"][k] == seg_names.index(want)
                        maps.setdefault((scen, light, want), {})[lag] = (
                            (hit & ok)[m].sum(0), ok[m].sum(0), speed[kt][m].mean(0), scene[kt][m].mean(0))
            # time-averaged direction on steady rotation segments, lag 0 (a mean over the segment)
            if scen in AXIS:
                for segi, (name, a, b) in enumerate(segments(z)):
                    if name == "rest":
                        continue
                    kk = np.arange(a + SKIP, b)
                    mf, mv = fields["T4+T5"][kk].mean(0), vel[kk].mean(0)
                    sp = np.hypot(mv[:, 0], mv[:, 1]) / PX_PER_COLUMN
                    ok = sp >= MIN_SPEED
                    sc = z["sky_frac"][kk].astype(np.float32).mean(0) < 0.5
                    hit = np.abs(wrap_deg(angle(mf[None])[0] - angle(mv[None])[0])) < 45
                    tavg.append(dict(light=light, scen=scen, seg=name, rate=abs(int(name.split()[1])),
                                     acc_scene=float(hit[ok & sc].mean()), acc_all=float(hit[ok].mean()),
                                     n_scene=int((ok & sc).sum()), speed_p50=float(np.median(sp[ok & sc]))))
            # spatial spectrum of the receptor images along lattice columns (v const), every 10th frame
            r = z["receptors"][::10].astype(np.float32)  # (t, B, 721)
            for idxs in vlines:
                n = len(idxs)
                x = r[:, :, idxs] - r[:, :, idxs].mean(-1, keepdims=True)
                p = np.abs(np.fft.rfft(x * np.hanning(n), axis=-1)) ** 2
                spec[light].append((np.fft.rfftfreq(n), p.reshape(-1, p.shape[-1]).mean(0)))
            print(f"direction {scen}_{light}", flush=True)
    return dict(hits=hits, total=total, err_hist=err_hist, hist_edges=hist_edges, maps=maps, tavg=tavg, spec=spec)


def lambda_eff(spec_list):
    """Power-weighted mean spatial frequency (cycles/column, DC excluded) -> wavelength in columns."""
    num = den = 0.0
    for f, p in spec_list:
        num += (f[1:] * p[1:]).sum()
        den += p[1:].sum()
    return 1.0 / (num / den)


def direction_summary(D):
    hits, total = D["hits"], D["total"]
    acc_lag = [hits[:, :, 0, lag].sum() / total[:, :, 0, lag].sum() for lag in LAGS]
    acc_lag_scene = [hits[:, :, 0, lag, 1].sum() / total[:, :, 0, lag, 1].sum() for lag in LAGS]
    lag = int(np.argmax(acc_lag_scene))
    H, N = hits[:, :, :, lag], total[:, :, :, lag]  # (L, S, V, geo, sbin, cbin)
    lam = {l: lambda_eff(D["spec"][l]) for l in LIGHTS}
    lam_all = lambda_eff(sum(D["spec"].values(), []))
    speed_mid = np.sqrt(SPEED_EDGES[:-1] * SPEED_EDGES[1:])

    def acc(h, n):
        return (h / np.maximum(n, 1)).round(4).tolist(), n.astype(int).tolist()

    out = dict(lag_frames=lag, lag_ms=10 * lag, acc_by_lag_all=np.round(acc_lag, 4).tolist(),
               acc_by_lag_scene=np.round(acc_lag_scene, 4).tolist(), lambda_eff_columns=round(lam_all, 2),
               lambda_eff_by_light={l: round(v, 2) for l, v in lam.items()},
               speed_edges=SPEED_EDGES.tolist(), contrast_edges=CONTRAST_EDGES.tolist())
    # pooled over everything
    for g, gname in ((1, "scene"), (slice(None), "all")):
        h, n = H[:, :, 0, g], N[:, :, 0, g]
        h, n = (h, n) if gname == "scene" else (h.sum(2), n.sum(2))
        out[f"overall_{gname}"] = round(float(h.sum() / n.sum()), 4)
        out[f"overall_{gname}_n"] = int(n.sum())
        out[f"overall_{gname}_by_light"] = {l: round(float(h[i].sum() / n[i].sum()), 4) for i, l in enumerate(LIGHTS)}
        out[f"by_speed_{gname}"] = acc(h.sum((0, 1, 3)), n.sum((0, 1, 3)))
        out[f"by_speed_{gname}_by_light"] = {l: acc(h[i].sum((0, 2)), n[i].sum((0, 2))) for i, l in enumerate(LIGHTS)}
        out[f"by_contrast_{gname}_by_light"] = {l: acc(h[i].sum((0, 1)), n[i].sum((0, 1))) for i, l in enumerate(LIGHTS)}
        out[f"by_contrast_{gname}"] = acc(h.sum((0, 1, 2)), n.sum((0, 1, 2)))
        out[f"by_scenario_{gname}_by_light"] = {
            l: {s: round(float(h[i, j].sum() / max(n[i, j].sum(), 1)), 4) for j, s in enumerate(SCENARIOS)}
            for i, l in enumerate(LIGHTS)}
        out[f"by_scenario_{gname}_n"] = {s: int(n[:, j].sum()) for j, s in enumerate(SCENARIOS)}
    for vi, var in enumerate(VARIANTS):
        h, n = H[:, :, vi, 1], N[:, :, vi, 1]
        out.setdefault("by_variant_scene", {})[var] = dict(
            overall=round(float(h.sum() / n.sum()), 4), by_light={l: round(float(h[i].sum() / n[i].sum()), 4) for i, l in enumerate(LIGHTS)},
            by_speed=acc(h.sum((0, 1, 3)), n.sum((0, 1, 3)))[0])
    # median absolute error, scene columns, from the histograms
    centres = 0.5 * (D["hist_edges"][1:] + D["hist_edges"][:-1])
    def med_abs(hist):
        a = np.abs(centres)
        o = np.argsort(a)
        c = np.cumsum(hist[o])
        return float(a[o][np.searchsorted(c, c[-1] / 2)])
    out["median_abs_error_deg_scene"] = {
        var: {l: med_abs(D["err_hist"][li, vi, lag, 1]) for li, l in enumerate(LIGHTS)} for vi, var in enumerate(VARIANTS)}
    # gate
    h, n = H[:, :, 0, 1].sum((0, 1, 3)), N[:, :, 0, 1].sum((0, 1, 3))
    a = h / np.maximum(n, 1)
    b = int(np.argmax(np.where(n >= GATE_MIN_N, a, -1)))
    gate = dict(definition=("fraction of column-frames whose T4+T5 MotionField vector is within 45 deg of the true "
                            f"flow direction (flow_box as pixel velocity), scene columns (sky_frac < 0.5), true speed "
                            f">= {MIN_SPEED} columns/s, first 0.3 s of each segment dropped, lag {10 * lag} ms; best "
                            f"speed bin with N >= {GATE_MIN_N}; all scenarios and lightings pooled"),
                bin=f"{SPEED_EDGES[b]:g}-{SPEED_EDGES[b + 1]:g} columns/s", bin_index=b, value=round(float(a[b]), 4),
                n=int(n[b]), speed_mid=round(float(speed_mid[b]), 2),
                tf_hz_at_lambda_eff=round(float(speed_mid[b] / lam_all), 2),
                tf_hz_range_at_lambda_eff=[round(SPEED_EDGES[b] / lam_all, 2), round(SPEED_EDGES[b + 1] / lam_all, 2)],
                tf_hz_at_lambda_4_8_16=[round(float(speed_mid[b] / l), 2) for l in (4, 8, 16)],
                by_light={l: dict(value=round(float(H[i, :, 0, 1, b].sum() / max(N[i, :, 0, 1, b].sum(), 1)), 4),
                                  n=int(N[i, :, 0, 1, b].sum())) for i, l in enumerate(LIGHTS)},
                all_columns=dict(value=round(float(H[:, :, 0, :, b].sum() / N[:, :, 0, :, b].sum()), 4),
                                 n=int(N[:, :, 0, :, b].sum())),
                by_scenario={s: dict(value=round(float(H[:, j, 0, 1, b].sum() / max(N[:, j, 0, 1, b].sum(), 1)), 4),
                                     n=int(N[:, j, 0, 1, b].sum())) for j, s in enumerate(SCENARIOS)},
                grating_time_averaged_at_4hz=GRATING_ACC, grating_per_frame_at_4hz=GRATING_ACC_FRAME)
    vp = VARIANTS.index("pooled 19")
    hp, np_ = H[:, :, vp, 1].sum((0, 1, 3)), N[:, :, vp, 1].sum((0, 1, 3))
    ap = hp / np.maximum(np_, 1)
    bp = int(np.argmax(np.where(np_ >= GATE_MIN_N, ap, -1)))
    gate["pooled_19"] = dict(bin=f"{SPEED_EDGES[bp]:g}-{SPEED_EDGES[bp + 1]:g} columns/s", value=round(float(ap[bp]), 4), n=int(np_[bp]),
                             by_light={l: round(float(H[i, :, vp, 1, bp].sum() / max(N[i, :, vp, 1, bp].sum(), 1)), 4) for i, l in enumerate(LIGHTS)})
    vs = VARIANTS.index("static")
    gate["static_baseline_same_bin"] = round(float(H[:, :, vs, 1, b].sum() / N[:, :, vs, 1, b].sum()), 4)
    out["gate"] = gate
    # time-averaged rotation segments
    ta = {}
    for r in D["tavg"]:
        ta.setdefault(r["light"], {}).setdefault(str(r["rate"]), []).append(r["acc_scene"])
    out["time_averaged_rotation_scene"] = {l: {k: round(float(np.mean(v)), 4) for k, v in d.items()} for l, d in ta.items()}
    out["time_averaged_rotation_segments"] = D["tavg"]
    return out


# ---------------------------------------------------------------------------------------------- rotation
def fit(x, y):
    A = np.stack([x, np.ones_like(x)], 1)
    (g, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ [g, c]
    return float(g), float(c), float(1 - (r @ r) / max(((y - y.mean()) ** 2).sum(), 1e-30))


def rotation():
    out, series = {}, {}
    for light in LIGHTS:
        L = out.setdefault(light, {})
        for method in ("rot_matched", "rot_lstsq"):
            M = L.setdefault(method, {})
            cross = np.zeros((3, 3))
            for scen, ax in AXIS.items():
                z, _ = load(scen, light)
                r, w = z[method].astype(np.float64), z["w_body"]
                off = r[30:100].mean(0)
                rs = r - off
                best = None
                for lag in range(31):
                    k = np.arange(30 + lag, len(r))
                    g, c, r2 = fit(w[k - lag, ax], r[k, ax])
                    if best is None or r2 > best[3]:
                        best = (lag, g, c, r2)
                segs = []
                for name, a, b in segments(z):
                    if name == "rest":
                        continue
                    rate = math.radians(int(name.split()[1]))
                    segs.append((rate, rs[a + SKIP:b].mean(0)))
                rates = np.array([s[0] for s in segs])
                means = np.array([s[1] for s in segs])  # (6, 3)
                cross[ax] = (rates @ means) / (rates @ rates)
                g6, c6, r2_6 = fit(rates, means[:, ax])
                by_rate = {}
                for rate, m in segs:
                    by_rate.setdefault(f"{abs(round(math.degrees(rate)))}", []).append(m[ax] / rate)
                M[scen] = dict(offset=off.round(4).tolist(), lag_ms=10 * best[0], gain_ts=round(best[1], 4),
                               offset_ts=round(best[2], 4), r2_ts=round(best[3], 4),
                               gain_seg=round(g6, 4), r2_seg=round(r2_6, 4),
                               gain_by_rate={k: round(float(np.mean(v)), 4) for k, v in by_rate.items()},
                               seg_means=[[round(math.degrees(rt)), *m.round(4).tolist()] for rt, m in segs],
                               offaxis_over_onaxis_max=round(float(np.max(np.abs(np.delete(means, ax, 1)).max(1)
                                                                          / np.maximum(np.abs(means[:, ax]), 1e-9))), 3))
                if method == "rot_matched":
                    series[(light, scen)] = (rs[:, ax], w[:, ax], best[0])
            M["cross_talk"] = cross.round(4).tolist()
            M["cross_talk_row_normalised"] = (cross / np.diag(cross)[:, None]).round(3).tolist()
            M["signs_right"] = int(sum(np.sign(s[1 + AXIS[sc]]) == np.sign(s[0]) for sc in AXIS
                                       for s in M[sc]["seg_means"]))
        # straight flights: translation leakage
        for scen in ("straight_3", "straight_10", "straight_30"):
            z, _ = load(scen, light)
            name, a, b = [s for s in segments(z) if s[0] != "rest"][0]
            full = z["speed"] >= z["speed"].max() - 1e-9
            k = np.nonzero(full & (np.arange(len(full)) >= a + SKIP) & (np.arange(len(full)) < b))[0]
            r = z["rot_matched"].astype(np.float64)
            off = r[30:100].mean(0)
            yaw_gain30 = L["rot_matched"]["yaw"]["gain_by_rate"]["30"]
            L.setdefault("straight", {})[scen] = dict(
                circuit=(r[k] - off).mean(0).round(4).tolist(), truth_box=z["rot_truth_box"][k].mean(0).round(4).tolist(),
                circuit_rad_s_at_yaw30_gain=((r[k] - off).mean(0) / yaw_gain30).round(3).tolist())
        L["matched_vs_lstsq_max_abs"] = float(max(np.abs(load(s, light)[0]["rot_matched"] - load(s, light)[0]["rot_lstsq"]).max()
                                                  for s in SCENARIOS))
    return out, series


# ---------------------------------------------------------------------------------------------- looming
def looming():
    out, series = {}, {}
    for light in LIGHTS:
        L = out.setdefault(light, {})
        fa = {}
        for scen in SCENARIOS:
            z, _ = load(scen, light)
            T = len(z["t"])
            k = np.arange(30, T)
            if scen == "approach":
                name, a, b = segments(z)[1]
                k = np.arange(30, a)  # rest before the approach
            fa[scen] = dict(mean=float(z["loom"][k].mean()), p99=float(np.percentile(z["loom"][k], 99)),
                            max=float(z["loom"][k].max()), truth_max=float(z["loom_truth"][k].max()))
        L["false_alarm"] = {s: {kk: round(v, 5) for kk, v in d.items()} for s, d in fa.items()}
        # straight flights close on the wall too (1/tau 0.007-0.09): reported, not used as false alarms
        thr = max(d["max"] for s, d in fa.items() if not s.startswith("straight"))
        thr_truth = max(d["truth_max"] for s, d in fa.items() if not s.startswith("straight"))
        z, _ = load("approach", light)
        name, a, b = segments(z)[1]
        inv = np.where(np.isfinite(z["tau_depth"]), 1 / z["tau_depth"], 0.0)
        loom, loomt = z["loom"].astype(np.float64), z["loom_truth"].astype(np.float64)
        steady = z["speed"] >= z["speed"].max() - 1e-9
        k = np.nonzero(steady & (np.arange(len(steady)) >= a + SKIP) & (np.arange(len(steady)) < b))[0]

        def corr(x, y):
            return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")

        best = max(((lag, corr(inv[k - lag], loom[k])) for lag in range(31)),
                   key=lambda p: -1 if math.isnan(p[1]) else p[1])
        edges = np.arange(0.1, 0.75, 0.05)
        bi = np.digitize(inv[k], edges) - 1
        bins = [float(loom[k][bi == i].mean()) if (bi == i).any() else float("nan") for i in range(len(edges) - 1)]
        binst = [float(loomt[k][bi == i].mean()) if (bi == i).any() else float("nan") for i in range(len(edges) - 1)]
        bm = np.array(bins)
        sat = None
        if np.nanmax(bm) > 0:
            j = int(np.nonzero(bm >= 0.9 * np.nanmax(bm))[0][0])
            sat = round(float(0.5 * (edges[j] + edges[j + 1])), 3)

        def detect(x, th):
            above = x[k] > th
            if not above.any():
                return None
            stay = np.nonzero(~above)[0]
            j = 0 if len(stay) == 0 else stay[-1] + 1
            first = int(np.nonzero(above)[0][0])
            late = x[k][-30:]
            stays = j < len(k)
            return dict(first_cross_inv_tau=round(float(inv[k][first]), 3), first_cross_tau_s=round(1 / max(inv[k][first], 1e-9), 2),
                        stays_above_from_inv_tau=round(float(inv[k][j]), 3) if stays else None,
                        stays_above_from_tau_s=round(1 / max(inv[k][j], 1e-9), 2) if stays else None,
                        frames_above=int(above.sum()), frames=len(k), last_03s_mean=round(float(late.mean()), 5),
                        margin_last_03s_over_threshold=round(float(late.mean() / th), 2) if th > 0 else None)

        L["approach"] = dict(first_03s=round(float(loom[k[:30]].mean()), 5), steady_frames=[int(k[0]), int(k[-1]) + 1],
                             last_03s=round(float(loom[k[-30:]].mean()), 5), peak=round(float(loom[k].max()), 5),
                             peak_inv_tau=round(float(inv[k][np.argmax(loom[k])]), 3), inv_tau_range=[round(float(inv[k].min()), 3), round(float(inv[k].max()), 3)],
                             corr_lag0=round(corr(inv[k], loom[k]), 3), best_lag_ms=10 * best[0], corr_best=round(best[1], 3),
                             truth_corr=round(corr(inv[k], loomt[k]), 3), truth_peak=round(float(loomt[k].max()), 4),
                             bin_edges=edges.round(2).tolist(), bin_means=[round(v, 5) for v in bins],
                             truth_bin_means=[round(v, 4) for v in binst], saturation_inv_tau=sat,
                             linear_slope_per_inv_tau=round(fit(inv[k], loom[k])[0], 4))
        L["threshold"] = round(thr, 5)
        L["detect"] = detect(loom, thr) if thr > 0 or loom[k].max() > 0 else None
        L["threshold_truth"] = round(thr_truth, 5)
        L["detect_truth"] = detect(loomt, thr_truth)
        series[light] = (z["t"], loom, loomt, inv, a, b)
    return out, series


# ---------------------------------------------------------------------------------------------- figures
COLORS = dict(bright="#e69f00", dim="#0072b2", dark="#444444")
VCOL = {"T4+T5": "#222222", "T4": "#d55e00", "T5": "#009e73", "pooled 19": "#56b4e9", "static": "#aaaaaa"}


def fig_direction(D, S, lat, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lag = S["lag_frames"]
    H, N = D["hits"][:, :, :, lag], D["total"][:, :, :, lag]
    xs = np.sqrt(SPEED_EDGES[:-1] * SPEED_EDGES[1:])
    plt.rcParams["axes.titlesize"] = 9
    fig = plt.figure(figsize=(22, 15))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    for i, l in enumerate(LIGHTS):
        h, n = H[i, :, 0, 1].sum((0, 2)), N[i, :, 0, 1].sum((0, 2))
        ok = n >= 2000
        ax.plot(xs[ok], (h / np.maximum(n, 1))[ok], "o-", color=COLORS[l], label=f"{l}, scene columns")
        h, n = H[i, :, 0].sum((0, 1, 3)), N[i, :, 0].sum((0, 1, 3))
        ok = n >= 2000
        ax.plot(xs[ok], (h / np.maximum(n, 1))[ok], "o:", color=COLORS[l], alpha=0.6, label=f"{l}, all columns")
    g = S["gate"]
    ax.axvspan(SPEED_EDGES[g["bin_index"]], SPEED_EDGES[g["bin_index"] + 1], color="#cccccc", alpha=0.4, lw=0)
    ax.axhline(0.25, color="k", lw=0.6, ls="--")
    ax.text(0.3, 0.26, "chance 0.25", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("true speed, columns/s (13 px)")
    ax.set_ylabel("fraction within 45 deg (per frame)")
    ax.set_title(f"a1  accuracy vs true speed, T4+T5, lag {10 * lag} ms\ngate: {g['value']:.3f} at {g['bin']} "
                 f"(N {g['n']}), TF ~{g['tf_hz_at_lambda_eff']} Hz at lambda_eff {S['lambda_eff_columns']} col")
    ax.legend(fontsize=7, loc="lower right")
    top = ax.secondary_xaxis("top", functions=(lambda s: s / S["lambda_eff_columns"], lambda f: f * S["lambda_eff_columns"]))
    top.set_xlabel("equivalent TF at lambda_eff, Hz", fontsize=8)

    ax = fig.add_subplot(gs[0, 1])
    for vi, var in enumerate(VARIANTS):
        h, n = H[:, :, vi, 1].sum((0, 1, 3)), N[:, :, vi, 1].sum((0, 1, 3))
        ok = n >= 2000
        ax.plot(xs[ok], (h / np.maximum(n, 1))[ok], "o-", color=VCOL[var], label=f"{var} ({S['by_variant_scene'][var]['overall']:.3f} overall)")
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_ylim(0, 1)
    ax.set_xlabel("true speed, columns/s")
    ax.axhline(0.25, color="k", lw=0.6, ls=":")
    ax.set_title("a2  variants (scene columns, all lights): T4+T5, T4-only, T5-only,\n"
                 "19-column hex mean (vs pooled truth), static = the run's rest field held")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 2])
    cx = np.arange(len(CONTRAST_EDGES) - 1)
    for i, l in enumerate(LIGHTS):
        h, n = H[i, :, 0, 1].sum((0, 1)), N[i, :, 0, 1].sum((0, 1))
        ok = n >= 2000
        ax.plot(cx[ok], (h / np.maximum(n, 1))[ok], "o-", color=COLORS[l], label=l)
    ax.set_xticks(cx, [f"{CONTRAST_EDGES[j]:g}-{min(CONTRAST_EDGES[j + 1], 1):g}" for j in cx], fontsize=8)
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel("footprint Michelson contrast (luma/255, 13x13 px)")
    ax.set_title("a3  accuracy vs local contrast (scene columns, speed >= 0.25 col/s)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[0, 3])
    for vi, var in enumerate(VARIANTS):
        a = [D["hits"][:, :, vi, L, 1].sum() / D["total"][:, :, vi, L, 1].sum() for L in LAGS]
        ax.plot([10 * L for L in LAGS], a, "o-", color=VCOL[var], label=var)
    ax.axvline(10 * lag, color="k", lw=0.6, ls=":")
    ax.set_xlabel("lag of the circuit behind truth, ms")
    ax.set_ylabel("fraction within 45 deg")
    ax.set_title("a4  pooled accuracy vs lag (scene columns)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 0:2])
    w = 0.26
    for i, l in enumerate(LIGHTS):
        vals = [S["by_scenario_scene_by_light"][l][s] for s in SCENARIOS]
        ax.bar(np.arange(len(SCENARIOS)) + (i - 1) * w, vals, w, color=COLORS[l], label=l)
        for x, v in zip(np.arange(len(SCENARIOS)) + (i - 1) * w, vals):
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=7)
    ax.set_xticks(np.arange(len(SCENARIOS)), [f"{s}\nN {S['by_scenario_scene_n'][s] / 1e6:.2f} M" for s in SCENARIOS], fontsize=8)
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction within 45 deg")
    ax.set_title("a5  accuracy per scenario and lighting (scene columns, speed >= 0.25 col/s, T4+T5)")
    ax.legend(fontsize=8, ncol=3)

    ax = fig.add_subplot(gs[1, 2])
    c = 0.5 * (D["hist_edges"][1:] + D["hist_edges"][:-1])
    for i, l in enumerate(LIGHTS):
        hh = D["err_hist"][i, 0, lag, 1]
        ax.plot(c, hh / hh.sum() / 5, color=COLORS[l], label=f"{l}, median |err| {S['median_abs_error_deg_scene']['T4+T5'][l]:.0f} deg")
    hh = D["err_hist"][:, 0, lag, 0].sum(0)
    ax.plot(c, hh / hh.sum() / 5, color="#999999", ls=":", label="sky columns, all lights")
    ax.axvline(-45, color="k", lw=0.6, ls="--")
    ax.axvline(45, color="k", lw=0.6, ls="--")
    ax.axhline(1 / 360, color="k", lw=0.5, ls=":")
    ax.set_xlim(-180, 180)
    ax.set_xticks(range(-180, 181, 45))
    ax.set_xlabel("circuit minus true direction, deg")
    ax.set_ylabel("density per deg")
    ax.set_title("a6  per-column error distribution (T4+T5, scene columns)")
    ax.legend(fontsize=7)

    ax = fig.add_subplot(gs[1, 3])
    for l in LIGHTS:
        d = S["time_averaged_rotation_scene"][l]
        rates = sorted(d, key=int)
        ax.plot([int(r) for r in rates], [d[r] for r in rates], "o-", color=COLORS[l], label=f"{l}, segment-mean vector")
    ax.set_xscale("log")
    ax.set_xticks([30, 90, 180], ["30", "90", "180"])
    ax.axhline(0.5, color="k", lw=0.6, ls="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel("rotation rate, deg/s")
    ax.set_title("a7  time-averaged direction, steady rotation segments\n(mean of yaw, pitch, roll, both signs; scene columns)")
    ax.legend(fontsize=8)

    rc = lat.pixel_rc(403).numpy()
    panels = [(("yaw", "bright", "yaw +90"), 0, "left eye"), (("yaw", "bright", "yaw +90"), 1, "right eye"),
              (("straight_30", "bright", "fly 30 m/s"), 0, "left eye"), (("straight_30", "bright", "fly 30 m/s"), 1, "right eye")]
    for j, (key, b, eye) in enumerate(panels):
        ax = fig.add_subplot(gs[2, j])
        h, n, sp, sc = D["maps"][key][lag]
        a = np.where(n[b] > 0, h[b] / np.maximum(n[b], 1), np.nan)
        ax.scatter(rc[:, 1], -rc[:, 0], c="#dddddd", s=62, marker="h", lw=0)
        m = np.isfinite(a)
        im = ax.scatter(rc[m, 1], -rc[m, 0], c=a[m], cmap="viridis", vmin=0, vmax=1, s=62, marker="h", lw=0)
        sky = (sc[b] < 0.5) & (n[b] > 0)
        ax.scatter(rc[sky, 1], -rc[sky, 0], facecolors="none", edgecolors="w", s=14, marker="o", lw=0.6)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        acc = np.nansum(h[b][sc[b] >= 0.5]) / max(n[b][sc[b] >= 0.5].sum(), 1)
        ax.set_title(f"a8  {key[2]}, {key[1]}, {eye}: scene columns {acc:.2f}\n"
                     f"grey = true speed < 0.25 col/s; white ring = sky column", fontsize=9)
        if j == 3:
            fig.colorbar(im, ax=ax, fraction=0.04, label="fraction within 45 deg")
    fig.suptitle("Phase 2 (a): T4/T5 local motion direction vs the renderer's box-averaged flow, Geiranger scenarios, 21 runs", fontsize=14)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def fig_rotation(R, series, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(3, 3, figsize=(20, 14))
    fig.subplots_adjust(hspace=0.42, wspace=0.25)
    for j, scen in enumerate(("yaw", "pitch", "roll")):
        ax = axs[0, j]
        for l in LIGHTS:
            rs, w, lag = series[(l, scen)]
            t = np.arange(len(rs)) * 0.01
            ax.plot(t, rs, color=COLORS[l], lw=0.9, label=f"circuit {l}")
        rs, w, lag = series[("bright", scen)]
        g = R["bright"]["rot_matched"][scen]["gain_by_rate"]["30"]
        ax.plot(t, w * g, color="k", lw=0.8, ls="--", label=f"true rate x {g:.3f} (bright 30 deg/s gain)")
        ax.set_xlabel("t, s")
        ax.set_ylabel(f"{AXIS_NAMES[AXIS[scen]]} readout minus rest, a.u.")
        m = R["bright"]["rot_matched"][scen]
        ax.set_title(f"c{j + 1}  {scen} run, on-axis matched readout\nbright: lag {m['lag_ms']} ms, R^2 {m['r2_ts']:.2f} (lagged linear fit)")
        ax.legend(fontsize=7, loc="lower left")
        ax = axs[1, j]
        for l in LIGHTS:
            sm = np.array(R[l]["rot_matched"][scen]["seg_means"])
            o = np.argsort(sm[:, 0])
            ax.plot(sm[o, 0], sm[o, 1 + AXIS[scen]], "o-", color=COLORS[l],
                    label=f"{l}: gain/rad/s 30|90|180 = " + " | ".join(f"{R[l]['rot_matched'][scen]['gain_by_rate'][k]:.3f}" for k in ("30", "90", "180")))
            for ax2 in range(3):
                if ax2 != AXIS[scen]:
                    ax.plot(sm[o, 0], sm[o, 1 + ax2], "x:", color=COLORS[l], alpha=0.5, lw=0.7)
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.set_xlabel("true rate, deg/s")
        ax.set_ylabel("segment mean minus rest, a.u.")
        ax.set_title(f"c{j + 4}  {scen}: steady-segment means (o on-axis, x off-axis)")
        ax.legend(fontsize=7)
    ax = axs[2, 0]
    for i, l in enumerate(LIGHTS):
        C = np.array(R[l]["rot_matched"]["cross_talk_row_normalised"])
        ext = [i * 4, i * 4 + 3, 0, 3]
        ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, extent=ext)
        for a in range(3):
            for b in range(3):
                ax.text(ext[0] + b + 0.5, 2.5 - a, f"{C[a, b]:+.2f}", ha="center", va="center", fontsize=8)
        ax.text(i * 4 + 1.5, 3.2, l, ha="center")
    ax.set_xlim(-0.2, 11.2)
    ax.set_ylim(-0.3, 3.6)
    ax.set_xticks([0.5, 1.5, 2.5, 4.5, 5.5, 6.5, 8.5, 9.5, 10.5], ["wx", "wy", "wz"] * 3)
    ax.set_yticks([2.5, 1.5, 0.5], ["pitch run", "yaw run", "roll run"])
    ax.set_title("c7  cross-talk, row-normalised slopes of segment means\n(row = true axis, column = readout)")
    ax = axs[2, 1]
    w = 0.13
    for i, l in enumerate(LIGHTS):
        for s, scen in enumerate(("straight_3", "straight_10", "straight_30")):
            v = R[l]["straight"][scen]["circuit"]
            x = s * 1.0 + (i - 1) * 0.3
            ax.bar([x - w, x, x + w], v, w, color=["#cc79a7", "#56b4e9", "#999999"], edgecolor=COLORS[l], lw=1.5)
    for s, scen in enumerate(("straight_3", "straight_10", "straight_30")):
        tb = R["bright"]["straight"][scen]["truth_box"]
        ax.text(s, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 0.01, f"truth wy {tb[1]:+.3f} rad/s", ha="center", fontsize=7)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks([0, 1, 2], ["3 m/s", "10 m/s", "30 m/s"])
    ax.set_ylabel("readout minus rest, a.u.")
    ax.set_title("c8  straight flight: wx (pink), wy (blue), wz (grey); bars bright | dim | dark\n(edge colour = lighting)")
    ax = axs[2, 2]
    ax.axis("off")
    lines = ["gain = segment mean / true rate (a.u. per rad/s), matched = lstsq", ""]
    for l in LIGHTS:
        lines.append(f"{l}  (signs right {R[l]['rot_matched']['signs_right']}/18, offset yaw run {R[l]['rot_matched']['yaw']['offset']})")
        for scen in ("yaw", "pitch", "roll"):
            m = R[l]["rot_matched"][scen]
            lines.append(f"  {scen:5s} gain 30|90|180 {m['gain_by_rate']['30']:.3f}|{m['gain_by_rate']['90']:.3f}|{m['gain_by_rate']['180']:.3f}"
                         f"  R^2 seg {m['r2_seg']:.2f}  ts {m['r2_ts']:.2f}  lag {m['lag_ms']} ms  off/on max {m['offaxis_over_onaxis_max']:.2f}")
        lines.append("")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8)
    fig.suptitle("Phase 2 (c): RotationReadout (matched) vs true body rates, hover at Geiranger, three lightings", fontsize=14)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def fig_looming(Lm, series, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 3, figsize=(20, 11))
    fig.subplots_adjust(hspace=0.35, wspace=0.28)
    ax = axs[0, 0]
    for l in LIGHTS:
        t, loom, loomt, inv, a, b = series[l]
        ax.plot(t, loom, color=COLORS[l], label=f"circuit {l} (r {Lm[l]['approach']['corr_lag0']})")
    t, loom, loomt, inv, a, b = series["bright"]
    ax.axvspan(t[a], t[b - 1], color="#eeeeee", lw=0)
    ax2 = ax.twinx()
    ax2.plot(t, inv, color="k", ls="--", lw=0.8, label="1/tau (depth)")
    ax2.set_ylabel("1/tau, 1/s")
    ax.axhline(Lm["bright"]["threshold"], color=COLORS["bright"], ls=":", lw=0.8)
    ax.set_xlabel("t, s")
    ax.set_ylabel("Looming value, a.u.")
    ax.set_title("d1  approach run: circuit looming and 1/tau (shaded = approach segment incl. ramps)")
    ax.legend(fontsize=8, loc="upper left")
    ax2.legend(fontsize=8, loc="center left")

    ax = axs[0, 1]
    edges = np.array(Lm["bright"]["approach"]["bin_edges"])
    mid = 0.5 * (edges[1:] + edges[:-1])
    for l in LIGHTS:
        t, loom, loomt, inv, a, b = series[l]
        k = np.arange(a + SKIP, b)
        ax.scatter(inv[k], loom[k], s=3, color=COLORS[l], alpha=0.25)
        ap = Lm[l]["approach"]
        ax.plot(mid, ap["bin_means"], "o-", color=COLORS[l],
                label=f"{l}: r {ap['corr_lag0']}, best lag {ap['best_lag_ms']} ms r {ap['corr_best']}, 90 % sat {ap['saturation_inv_tau']}")
        ax.axhline(Lm[l]["threshold"], color=COLORS[l], ls=":", lw=0.8)
    ax.set_xlabel("1/tau from the depth AOV, 1/s")
    ax.set_ylabel("circuit Looming, a.u.")
    ax.set_title("d2  circuit looming vs 1/tau, binned means (dotted = false-alarm max per lighting)")
    ax.legend(fontsize=7)

    ax = axs[0, 2]
    for l in LIGHTS:
        t, loom, loomt, inv, a, b = series[l]
        k = np.arange(a + SKIP, b)
        ax.plot(inv[k], loomt[k], color=COLORS[l], lw=1, label=f"{l}: r {Lm[l]['approach']['truth_corr']}")
    ax.set_xlabel("1/tau, 1/s")
    ax.set_ylabel("Looming on the true flow, rad/s")
    ax.set_title("d3  the same readout on the renderer's flow (reference)")
    ax.legend(fontsize=8)

    ax = axs[1, 0]
    names = [s for s in SCENARIOS if s != "approach"] + ["approach rest"]
    w = 0.26
    for i, l in enumerate(LIGHTS):
        fa = Lm[l]["false_alarm"]
        vals = [fa[s]["max"] for s in SCENARIOS if s != "approach"] + [fa["approach"]["max"]]
        p99 = [fa[s]["p99"] for s in SCENARIOS if s != "approach"] + [fa["approach"]["p99"]]
        x = np.arange(len(names)) + (i - 1) * w
        ax.bar(x, vals, w, color=COLORS[l], alpha=0.5, label=f"{l} max")
        ax.bar(x, p99, w, color=COLORS[l], label=f"{l} p99")
        ax.plot([x[0] - w, x[-1] + w], [Lm[l]["approach"]["last_03s"]] * 2, color=COLORS[l], ls="--", lw=0.8)
    ax.set_xticks(np.arange(len(names)), names, fontsize=8)
    ax.set_ylabel("circuit Looming, a.u.")
    ax.set_title("d4  loom off the approach: max (light) and p99 (solid), frames >= 30\n"
                 "(dashed = steady approach, last 0.3 s mean; straight flights are slow approaches)")
    ax.legend(fontsize=7, ncol=3)

    ax = axs[1, 1]
    for l in LIGHTS:
        fa = Lm[l]["false_alarm"]
        vals = [fa[s]["truth_max"] for s in SCENARIOS if s != "approach"] + [fa["approach"]["truth_max"]]
        ax.plot(np.arange(len(names)), vals, "o-", color=COLORS[l], label=l)
    ax.set_xticks(np.arange(len(names)), names, fontsize=8)
    ax.set_ylabel("Looming on the true flow, max")
    ax.set_title("d5  false alarms of the same readout on the true flow")
    ax.legend(fontsize=8)

    ax = axs[1, 2]
    ax.axis("off")
    lines = []
    for l in LIGHTS:
        ap, d, dt_ = Lm[l]["approach"], Lm[l]["detect"], Lm[l]["detect_truth"]
        lines.append(f"{l}")
        lines.append(f"  approach first 0.3 s {ap['first_03s']:.4f}, last 0.3 s {ap['last_03s']:.4f}, peak {ap['peak']:.4f} at 1/tau {ap['peak_inv_tau']}")
        lines.append(f"  corr with 1/tau {ap['corr_lag0']} (lag {ap['best_lag_ms']} ms: {ap['corr_best']}), truth {ap['truth_corr']}")
        lines.append(f"  threshold (false-alarm max) {Lm[l]['threshold']:.4f}")
        if d:
            lines.append(f"  detect: first above at 1/tau {d['first_cross_inv_tau']}, stays above from {d['stays_above_from_inv_tau']}"
                         f" (tau {d['stays_above_from_tau_s']} s)")
            lines.append(f"          last 0.3 s / threshold = {d['margin_last_03s_over_threshold']}")
        else:
            lines.append("  detect: never above the false-alarm max")
        if dt_:
            lines.append(f"  truth: stays above {Lm[l]['threshold_truth']:.4f} from 1/tau {dt_['stays_above_from_inv_tau']}, margin {dt_['margin_last_03s_over_threshold']}")
        lines.append("")
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=8)
    fig.suptitle("Phase 2 (d): LPLC2-style Looming (opponent) vs 1/tau on the 15 m/s wall approach, and its false alarms", fontsize=14)
    fig.savefig(path, dpi=90, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    global DATA
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--figures", type=Path, default=HERE / "figures")
    args = ap.parse_args(argv)
    DATA = args.data
    lat = HexLattice()
    M = pixel_map(lat)
    D = direction(lat, M)
    S = direction_summary(D)
    R, rser = rotation()
    Lm, lser = looming()
    res = dict(direction=S, rotation=R, looming=Lm)
    (DATA / "calibrate.json").write_text(json.dumps(res, indent=1))
    args.figures.mkdir(exist_ok=True)
    fig_direction(D, S, lat, args.figures / "phase2_direction.png")
    fig_rotation(R, rser, args.figures / "phase2_rotation.png")
    fig_looming(Lm, lser, args.figures / "phase2_looming.png")
    g = S["gate"]
    print(f"gate {g['value']} at {g['bin']} N {g['n']} lag {S['lag_ms']} ms lambda_eff {S['lambda_eff_columns']} "
          f"TF {g['tf_hz_at_lambda_eff']} Hz; by light {g['by_light']}")
    print(f"wrote {DATA / 'calibrate.json'} and figures in {args.figures}")


if __name__ == "__main__":
    main()
