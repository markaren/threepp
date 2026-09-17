"""Phase 2 scoring against the renderer's ground truth, as importable functions.

Extracted from calibrate.py (which now uses this module and still prints gate 0.3619) so any
readout variant or model member is scored with the same rules. CPU numpy, no threepp import;
field_rotation / field_looming use torch through readouts.py.

    from flyeye import scoring as sc
    runs = sc.Runs()                                   # lazy cache of the 21 recordings
    fields = {key: my_field(runs[key]) for key in runs.keys()}   # (T, B, 2, 721), x right, y up
    D = sc.score_direction(fields, runs)               # gate, pooled-19, per lighting
    rot = {key: sc.field_rotation(fields[key], runs[key]) for key in runs.keys()}
    R = sc.score_rotation(rot, runs)                   # per lighting: R^2, gains, signs
    loom = {key: sc.field_looming(fields[key], runs[key]) for key in runs.keys()}
    L = sc.score_looming(loom, runs)                   # per lighting: approach peak, false alarms, margin

Keys are (scenario, light) with scenario in SCENARIOS and light in LIGHTS. Every series is
per frame: frame k is the render after pose k, dt 0.01 s.

Direction rules (the Phase 2 gate):
- Truth: flow_box (rad/s on the column's unit image tangents) mapped to the column centre's
  pixel velocity (px/s, x right, y up) through the exact pinhole Jacobian (pixel_map).
  Speed = |pixel velocity| / 13 px, in columns/s.
- A column-frame counts when true speed >= MIN_SPEED (0.25 columns/s) and the circuit frame
  is >= SKIP (30 frames = 0.3 s) into its segment. The circuit frame k is compared with truth
  at k - lag.
- Hit: |wrapped angle(field) - angle(truth)| < 45 deg. A field vector of exactly zero length has
  angle atan2(0, 0) = 0 (right) and can hit, as in Phase 2 (zero_is_miss=False, the default that
  reproduces 0.3619). The recorded f16 T5-only field is exactly zero often (T5 rests at 0 on grey,
  overall 0.252 vs 0.221 as a miss); T4+T5 almost never (third decimal of a few bins).
- Scene columns: sky_frac < 0.5.
- lag: the one in LAGS (0..60 ms) that maximises scene accuracy over all speeds, unless given.
- Gate: scene accuracy in the best log2 speed bin (SPEED_EDGES) with N >= GATE_MIN_N, all runs
  pooled. Pooled-19: the 19-column hex mean of the field against the same mean of the truth,
  its own best bin at the same lag.
- Motion-blind baselines: "held" = the field's mean over frames 30-100 of the first rest,
  held for the whole run (calibrate's "static"); a caller with a stateful readout should
  instead pass its own baseline fields (e.g. the readout run on rest frames repeated) through
  score_direction(baseline=...).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .lattice import HexLattice, default_lattice

DATA = Path(r"C:\dev\_flyeye\phase2")
SCENARIOS = ("straight_3", "straight_10", "straight_30", "yaw", "pitch", "roll", "approach")
LIGHTS = ("bright", "dim", "dark")
SKIP = 30  # 0.3 s at 100 Hz
MIN_SPEED = 0.25  # columns/s
LAGS = range(7)  # frames, 0..60 ms
SPEED_EDGES = 2.0 ** np.arange(-2, 9)  # 0.25 .. 256 columns/s, 10 bins
CONTRAST_EDGES = np.array([0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.0001])
GATE_MIN_N = 20000
PX_PER_COLUMN = 13.0
AXIS = dict(pitch=0, yaw=1, roll=2)
AXIS_NAMES = ("wx pitch", "wy yaw", "wz roll")
REST = slice(30, 100)  # frames of the first rest used for offsets and the held baseline
N_SPEED, N_CONTRAST = len(SPEED_EDGES) - 1, len(CONTRAST_EDGES) - 1


# ---------------------------------------------------------------------------------------------- basics
def segments(z):
    """[(name, first, end)] from a recording's labels ('name|first|end', end exclusive)."""
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


def pixel_map(lat: HexLattice | None = None, size=403, fov=90.0) -> np.ndarray:
    """(721, 2, 2): tangent projections (right, up) of angular flow, rad/s -> pixel velocity px/s (x right, y up)."""
    lat = lat or default_lattice()
    d = lat.column_rays(size, fov, torch.float64).numpy()
    T = lat.column_tangents(size, fov, torch.float64).numpy()  # (721, 2, 3)
    f = size / 2 / math.tan(math.radians(fov) / 2)
    J = np.zeros((len(d), 2, 3))
    J[:, 0, 0] = J[:, 1, 1] = f / -d[:, 2]
    J[:, 0, 2] = f * d[:, 0] / d[:, 2] ** 2
    J[:, 1, 2] = f * d[:, 1] / d[:, 2] ** 2
    G = np.einsum("nki,nli->nkl", T, T)
    return np.einsum("nki,nji,njl->nkl", J, T, np.linalg.inv(G))


def hex_pool(lat: HexLattice | None = None, radius: int = 2) -> np.ndarray:
    """(721, 721) row-normalised mean over the columns within hex distance radius (2 -> 19 columns)."""
    lat = lat or default_lattice()
    u, v = lat.u.numpy(), lat.v.numpy()
    du, dv = u[:, None] - u[None], v[:, None] - v[None]
    A = ((np.abs(du) + np.abs(dv) + np.abs(du + dv)) // 2 <= radius).astype(np.float32)
    return A / A.sum(1, keepdims=True)


def pool_field(field: np.ndarray, P: np.ndarray) -> np.ndarray:
    """(T, B, 2, 721) -> the same with every column replaced by its hex_pool mean."""
    return np.einsum("mn,tbkn->tbkm", P, field)


def angle(v):
    """(T, B, 2, N) -> (T, B, N) image angle, rad, 0 = right, +pi/2 = up."""
    return np.arctan2(v[:, :, 1], v[:, :, 0])


def wrap_deg(a):
    return np.degrees((a + np.pi) % (2 * np.pi) - np.pi)


def load(scen, light, data: Path = DATA):
    """(NpzFile, meta dict) of one recording."""
    return np.load(Path(data) / f"{scen}_{light}.npz"), json.loads((Path(data) / f"{scen}_{light}.json").read_text())


# ---------------------------------------------------------------------------------------------- runs
class Run:
    """One recording with its truth prepared once.

    z: the NpzFile (arrays load on access); meta: the json; key (scen, light).
    Lazily: vel (T, B, 2, 721) float32 px/s; truth(pooled) -> (angle (T,B,N), speed (T,B,N),
    speed bin (T,B,N)); scene (T,B,N) bool; cbin (T,B,N); into (T,); raw() t4t5_raw float32;
    field() the recorded unit-gain field float32.
    """

    def __init__(self, scen, light, data: Path = DATA, M=None, P=None):
        self.key = (scen, light)
        self.scen, self.light = scen, light
        self.z, self.meta = load(scen, light, data)
        self._M, self._P = M, P
        self._c = {}

    def arr(self, name):
        if name not in self._c:
            self._c[name] = self.z[name]
        return self._c[name]

    @property
    def T(self):
        return len(self.arr("t"))

    @property
    def M(self):
        if self._M is None:
            self._M = pixel_map()
        return self._M

    @property
    def P(self):
        if self._P is None:
            self._P = hex_pool()
        return self._P

    @property
    def vel(self):
        if "vel" not in self._c:
            self._c["vel"] = np.einsum("nkl,tbln->tbkn", self.M, self.arr("flow_box").astype(np.float64)).astype(np.float32)
        return self._c["vel"]

    def truth(self, pooled=False):
        key = "truth_pooled" if pooled else "truth"
        if key not in self._c:
            v = pool_field(self.vel, self.P) if pooled else self.vel
            sp = np.hypot(v[:, :, 0], v[:, :, 1]) / PX_PER_COLUMN
            self._c[key] = (angle(v), sp, np.clip(np.digitize(sp, SPEED_EDGES) - 1, 0, N_SPEED - 1))
        return self._c[key]

    @property
    def scene(self):
        if "scene" not in self._c:
            self._c["scene"] = self.arr("sky_frac").astype(np.float32) < 0.5
        return self._c["scene"]

    @property
    def cbin(self):
        if "cbin" not in self._c:
            c = self.arr("contrast").astype(np.float32)
            self._c["cbin"] = np.clip(np.digitize(c, CONTRAST_EDGES) - 1, 0, N_CONTRAST - 1)
        return self._c["cbin"]

    @property
    def into(self):
        if "into" not in self._c:
            self._c["into"] = into_segment(self.z)
        return self._c["into"]

    @property
    def segments(self):
        return segments(self.z)

    def raw(self):
        """t4t5_raw (T, B, 8, 721) float32: relu(v) minus the grey-0.5 rest, T4a..T4d, T5a..T5d."""
        return self.arr("t4t5_raw").astype(np.float32)

    def field(self):
        return self.arr("field").astype(np.float32)

    def drop(self):
        """Free the cached arrays (truth stays cheap to rebuild)."""
        self._c.clear()


class Runs:
    """Lazy {(scen, light): Run} over the recordings, sharing one pixel map and hex pool."""

    def __init__(self, data: Path = DATA, scenarios=SCENARIOS, lights=LIGHTS):
        self.data, self.scenarios, self.lights = Path(data), tuple(scenarios), tuple(lights)
        self.M, self.P = pixel_map(), hex_pool()
        self._runs = {}

    def keys(self, lights=None):
        return [(s, l) for l in (lights or self.lights) for s in self.scenarios]

    def __getitem__(self, key) -> Run:
        if key not in self._runs:
            self._runs[key] = Run(*key, data=self.data, M=self.M, P=self.P)
        return self._runs[key]


# ---------------------------------------------------------------------------------------------- direction
def held_baseline(field: np.ndarray) -> np.ndarray:
    """The field's mean over frames 30-100 of the first rest, held for every frame (calibrate's 'static')."""
    return np.broadcast_to(field[REST].mean(0), field.shape)


def frame_hits(field: np.ndarray, run: Run, lag: int, pooled=False, zero_is_miss=False, ang=None):
    """Per-frame hits at one lag.

    Returns (k, kt, err, hit, ok, idx): circuit frames k (>= SKIP into segment, >= lag), truth
    frames kt = k - lag, err (K, B, N) deg, hit (K, B, N) bool, ok (K, B, N) true speed >= MIN_SPEED,
    idx = flat (scene, speed bin, contrast bin) index of the ok entries. ang: precomputed angle(field).
    """
    T = run.T
    k = np.nonzero((run.into >= SKIP) & (np.arange(T) >= lag))[0]
    kt = k - lag
    ang_t, sp, sbin = run.truth(pooled)
    ok = sp[kt] >= MIN_SPEED
    scene = run.scene
    idx = ((scene[kt].astype(np.int64) * N_SPEED + sbin[kt]) * N_CONTRAST + run.cbin[kt])[ok]
    a = angle(field) if ang is None else ang
    err = wrap_deg(a[k] - ang_t[kt])
    hit = np.abs(err) < 45
    if zero_is_miss:
        hit &= (field[k, :, 0] != 0) | (field[k, :, 1] != 0)
    return k, kt, err, hit, ok, idx


def direction_counts(field: np.ndarray, run: Run, lags=LAGS, pooled=False, zero_is_miss=False) -> dict:
    """hits and total (nLags, 2 [sky, scene], N_SPEED, N_CONTRAST) for one run's field (T, B, 2, 721).

    pooled=True compares with the hex-pooled truth; pass the pooled field yourself (pool_field)."""
    lags = list(lags)
    shape = (len(lags), 2, N_SPEED, N_CONTRAST)
    hits, total = np.zeros(shape), np.zeros(shape)
    ang = angle(field)
    for i, lag in enumerate(lags):
        k, kt, err, hit, ok, idx = frame_hits(field, run, lag, pooled, zero_is_miss, ang)
        n = 2 * N_SPEED * N_CONTRAST
        hits[i] = np.bincount(idx, weights=hit[ok], minlength=n).reshape(2, N_SPEED, N_CONTRAST)
        total[i] = np.bincount(idx, minlength=n).reshape(2, N_SPEED, N_CONTRAST)
    return dict(hits=hits, total=total, lags=lags)


def _stack(counts: dict, lights=LIGHTS, scenarios=SCENARIOS):
    """{(scen, light): counts} -> hits, total (L, S, nLags, 2, nS, nC)."""
    H = np.stack([np.stack([counts[(s, l)]["hits"] for s in scenarios]) for l in lights])
    N = np.stack([np.stack([counts[(s, l)]["total"] for s in scenarios]) for l in lights])
    return H, N


def best_lag(counts: dict, lights=LIGHTS, scenarios=SCENARIOS) -> int:
    """Index into the counts' lags that maximises scene accuracy over all speeds, contrasts and runs."""
    H, N = _stack(counts, lights, scenarios)
    acc = [H[:, :, i, 1].sum() / N[:, :, i, 1].sum() for i in range(H.shape[2])]
    return int(np.argmax(acc))


def gate_from_counts(counts: dict, lag_index=None, bin_index=None, lights=LIGHTS, scenarios=SCENARIOS) -> dict:
    """Gate numbers from {(scen, light): direction_counts}.

    lag_index None = best_lag; bin_index None = the best speed bin with N >= GATE_MIN_N.
    Returns value, bin, bin_index, n, lag_ms, by_light {light: value}, n_by_light, overall (scene,
    all speeds), overall_by_light, by_speed (scene accuracies per bin), by_scenario {light: {scen: value}}.
    """
    H, N = _stack(counts, lights, scenarios)
    li = best_lag(counts, lights, scenarios) if lag_index is None else lag_index
    lags = counts[(scenarios[0], lights[0])]["lags"]
    h, n = H[:, :, li, 1], N[:, :, li, 1]  # (L, S, nS, nC) scene
    hs, ns = h.sum((0, 1, 3)), n.sum((0, 1, 3))
    a = hs / np.maximum(ns, 1)
    b = int(np.argmax(np.where(ns >= GATE_MIN_N, a, -1))) if bin_index is None else bin_index
    r = lambda x: round(float(x), 4)
    return dict(value=r(a[b]), bin=f"{SPEED_EDGES[b]:g}-{SPEED_EDGES[b + 1]:g} columns/s", bin_index=b, n=int(ns[b]),
                lag_index=li, lag_ms=10 * lags[li],
                by_light={l: r(h[i, :, b].sum() / max(n[i, :, b].sum(), 1)) for i, l in enumerate(lights)},
                n_by_light={l: int(n[i, :, b].sum()) for i, l in enumerate(lights)},
                overall=r(h.sum() / n.sum()), overall_by_light={l: r(h[i].sum() / n[i].sum()) for i, l in enumerate(lights)},
                by_speed=[r(x) for x in a], n_by_speed=ns.astype(int).tolist(),
                by_scenario={l: {s: r(h[i, j, b].sum() / max(n[i, j, b].sum(), 1)) for j, s in enumerate(scenarios)}
                             for i, l in enumerate(lights)})


def score_direction(fields, runs: Runs, baseline="held", pooled=True, lights=None, lag=None, zero_is_miss=False,
                    keep_counts=False) -> dict:
    """The Phase 2 gate for one readout.

    fields: {(scen, light): (T, B, 2, 721)} or a callable run -> field (called once per run).
    baseline: "held" (held_baseline of the field), None, or {(scen, light): field} / callable run -> field
      (e.g. the readout on repeated rest frames). Scored at the field's lag and gate bin.
    pooled: also score the 19-column hex mean (own best bin, same lag).
    lights: subset of LIGHTS (e.g. ("dim", "dark") for a test split); lag: fixed frames or None (best).
    Returns dict(gate=..., pooled_19=..., baseline=..., lift=gate - baseline) with gate_from_counts dicts.
    """
    lights = tuple(lights or runs.lights)
    keys = [(s, l) for l in lights for s in runs.scenarios]
    lags = LAGS if lag is None else [lag]
    C, Cp, Cb = {}, {}, {}
    for key in keys:
        run = runs[key]
        f = fields(run) if callable(fields) else fields[key]
        f = np.asarray(f, dtype=np.float32)
        C[key] = direction_counts(f, run, lags, zero_is_miss=zero_is_miss)
        if pooled:
            Cp[key] = direction_counts(pool_field(f, run.P), run, lags, pooled=True, zero_is_miss=zero_is_miss)
        if baseline is not None:
            if isinstance(baseline, str) and baseline == "held":
                fb = held_baseline(f)
            else:
                fb = baseline(run) if callable(baseline) else baseline[key]
            Cb[key] = direction_counts(np.asarray(fb, dtype=np.float32), run, lags, zero_is_miss=zero_is_miss)
    out = dict(gate=gate_from_counts(C, lights=lights, scenarios=runs.scenarios))
    li = out["gate"]["lag_index"]
    if pooled:
        out["pooled_19"] = gate_from_counts(Cp, lag_index=li, lights=lights, scenarios=runs.scenarios)
    if baseline is not None:
        out["baseline"] = gate_from_counts(Cb, lag_index=li, bin_index=out["gate"]["bin_index"], lights=lights,
                                           scenarios=runs.scenarios)
        out["lift"] = round(out["gate"]["value"] - out["baseline"]["value"], 4)
    if keep_counts:
        out["counts"] = dict(field=C, pooled=Cp, baseline=Cb)
    return out


# ---------------------------------------------------------------------------------------------- field -> series
def field_rotation(field, run: Run, method="matched", device="cpu") -> np.ndarray:
    """(T, B, 2, 721) -> (T, 3) RotationReadout estimate (wx pitch, wy yaw, wz roll) with the run's eye geometry."""
    from .readouts import RotationReadout

    m = run.meta
    ro = RotationReadout(m["size"], m["fov_deg"], m["eye_rotations"], method=method)
    f = torch.as_tensor(np.asarray(field), dtype=torch.float32, device=device)
    return ro(f).cpu().numpy()


_LOOM_CACHE = {}


def looming_readout(run: Run, opponent=True):
    from .readouts import Looming

    m = run.meta
    key = (m["size"], m["fov_deg"], json.dumps(m["eye_rotations"]), opponent)
    if key not in _LOOM_CACHE:
        _LOOM_CACHE[key] = Looming(m["size"], m["fov_deg"], m["eye_rotations"], opponent=opponent)
    return _LOOM_CACHE[key]


def field_looming(field, run: Run, device=None, chunk=64, opponent=True) -> np.ndarray:
    """(T, B, 2, 721) -> (T,) Looming value per frame (readouts.Looming, batched over frames)."""
    lo = looming_readout(run, opponent)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    f = torch.as_tensor(np.asarray(field), dtype=torch.float32)
    radial = lo.radial.to(device=device, dtype=torch.float32)
    quad = lo.quadrants.to(device=device, dtype=torch.float32)
    out = []
    for i in range(0, len(f), chunk):
        x = f[i:i + chunk].to(device)
        o = torch.einsum("cbkn,tbkn->tcbn", radial, x)
        if opponent:
            q = torch.relu(torch.einsum("cqbn,tcbn->tcq", quad, o))
        else:
            q = torch.einsum("cqbn,tcbn->tcq", quad, torch.relu(o))
        out.append(q.min(dim=2).values.max(dim=1).values.cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------------------------------------- rotation
def fit(x, y):
    """Least squares y = g x + c -> (g, c, R^2)."""
    A = np.stack([x, np.ones_like(x)], 1)
    (g, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ [g, c]
    return float(g), float(c), float(1 - (r @ r) / max(((y - y.mean()) ** 2).sum(), 1e-30))


def rotation_run(r: np.ndarray, run: Run, ax: int) -> dict:
    """One rotation run (yaw, pitch or roll): r (T, 3) readout series, ax the true axis.

    offset = mean of frames 30-100; lagged linear fit over frames >= 30 (lag 0..300 ms, best R^2);
    steady-segment means (first SKIP frames dropped) per rate; gain_seg / r2_seg = fit of the six
    on-axis means against the rate (rad/s); signs_right = on-axis mean has the rate's sign (of 6).
    Also returns seg_rates (6,) and seg_means (6, 3) for the cross-talk.
    """
    r = np.asarray(r, dtype=np.float64)
    w = run.arr("w_body")
    off = r[REST].mean(0)
    rs = r - off
    best = None
    for lag in range(31):
        k = np.arange(30 + lag, len(r))
        g, c, r2 = fit(w[k - lag, ax], r[k, ax])
        if best is None or r2 > best[3]:
            best = (lag, g, c, r2)
    segs = []
    for name, a, b in run.segments:
        if name == "rest":
            continue
        segs.append((math.radians(int(name.split()[1])), rs[a + SKIP:b].mean(0)))
    rates = np.array([s[0] for s in segs])
    means = np.array([s[1] for s in segs])
    g6, c6, r2_6 = fit(rates, means[:, ax])
    by_rate = {}
    for rate, m in segs:
        by_rate.setdefault(f"{abs(round(math.degrees(rate)))}", []).append(m[ax] / rate)
    return dict(offset=off.round(4).tolist(), lag_ms=10 * best[0], gain_ts=round(best[1], 4), offset_ts=round(best[2], 4),
                r2_ts=round(best[3], 4), gain_seg=round(g6, 4), r2_seg=round(r2_6, 4),
                gain_by_rate={k: round(float(np.mean(v)), 4) for k, v in by_rate.items()},
                seg_means=[[round(math.degrees(rt)), *m.round(4).tolist()] for rt, m in segs],
                offaxis_over_onaxis_max=round(float(np.max(np.abs(np.delete(means, ax, 1)).max(1)
                                                           / np.maximum(np.abs(means[:, ax]), 1e-9))), 3),
                signs_right=int(sum(np.sign(m[ax]) == np.sign(rt) for rt, m in segs)),
                _rates=rates, _means=means, _rs=rs[:, ax], _lag=best[0])


def score_rotation(series: dict, runs: Runs, lights=None) -> dict:
    """{(scen, light): (T, 3)} on the yaw, pitch and roll runs -> per light:
    {yaw|pitch|roll: rotation_run dict (without the private arrays)}, cross_talk (3x3, row = true
    axis, least-squares slope of the segment means through the origin), cross_talk_row_normalised,
    signs_right (of 18), r2_seg {axis: value}. Straight flights are not scored here (straight_leak)."""
    out = {}
    for light in lights or runs.lights:
        L = out.setdefault(light, {})
        cross = np.zeros((3, 3))
        for scen, ax in AXIS.items():
            d = rotation_run(series[(scen, light)], runs[(scen, light)], ax)
            cross[ax] = (d["_rates"] @ d["_means"]) / (d["_rates"] @ d["_rates"])
            L[scen] = {k: v for k, v in d.items() if not k.startswith("_")}
        L["cross_talk"] = cross.round(4).tolist()
        L["cross_talk_row_normalised"] = (cross / np.diag(cross)[:, None]).round(3).tolist()
        L["signs_right"] = int(sum(L[s]["signs_right"] for s in AXIS))
        L["r2_seg"] = {s: L[s]["r2_seg"] for s in AXIS}
    return out


def straight_leak(series: dict, runs: Runs, light: str, yaw_gain30: float) -> dict:
    """Straight flights: the rest-subtracted readout (T, 3) mean over steady full-speed frames against the truth."""
    out = {}
    for scen in ("straight_3", "straight_10", "straight_30"):
        run = runs[(scen, light)]
        name, a, b = [s for s in run.segments if s[0] != "rest"][0]
        speed = run.arr("speed")
        full = speed >= speed.max() - 1e-9
        k = np.nonzero(full & (np.arange(len(full)) >= a + SKIP) & (np.arange(len(full)) < b))[0]
        r = np.asarray(series[(scen, light)], dtype=np.float64)
        off = r[REST].mean(0)
        out[scen] = dict(circuit=(r[k] - off).mean(0).round(4).tolist(), truth_box=run.arr("rot_truth_box")[k].mean(0).round(4).tolist(),
                         circuit_rad_s_at_yaw30_gain=((r[k] - off).mean(0) / yaw_gain30).round(3).tolist())
    return out


# ---------------------------------------------------------------------------------------------- looming
def _corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 and y.std() > 0 else float("nan")


def score_looming(series: dict, runs: Runs, lights=None, truth_key="loom_truth") -> dict:
    """{(scen, light): (T,) looming value} over all seven scenarios -> per light:

    false_alarm {scen: mean, p99, max, truth_max} over frames >= 30 (approach: its first rest only);
    threshold = max false alarm over the rotation runs and the approach rest (calibrate's rule;
    straight flights close on the wall at 1/tau 0.007-0.09 and are kept apart);
    threshold_with_straight = the same including straight flights;
    approach: steady full-speed frames >= SKIP into the approach: first_03s, last_03s, peak,
    peak_inv_tau, corr_lag0 and corr_best (lag 0..300 ms) with 1/tau_depth, binned means,
    saturation_inv_tau, linear slope; truth_corr / truth_peak for the readout on the true flow;
    detect (first crossing, stays-above, last 0.3 s margin) at threshold;
    margin_peak = approach peak / threshold, margin_last = last 0.3 s mean / threshold,
    margin_peak_with_straight = peak / threshold_with_straight.
    """
    out = {}
    for light in lights or runs.lights:
        L = out.setdefault(light, {})
        fa = {}
        for scen in runs.scenarios:
            run = runs[(scen, light)]
            x = np.asarray(series[(scen, light)])
            truth = run.arr(truth_key)
            k = np.arange(30, run.T)
            if scen == "approach":
                name, a, b = run.segments[1]
                k = np.arange(30, a)
            fa[scen] = dict(mean=float(x[k].mean()), p99=float(np.percentile(x[k], 99)), max=float(x[k].max()),
                            truth_max=float(truth[k].max()))
        L["false_alarm"] = {s: {kk: round(v, 5) for kk, v in d.items()} for s, d in fa.items()}
        thr = max(d["max"] for s, d in fa.items() if not s.startswith("straight"))
        thr_all = max(d["max"] for d in fa.values())
        thr_truth = max(d["truth_max"] for s, d in fa.items() if not s.startswith("straight"))
        run = runs[("approach", light)]
        name, a, b = run.segments[1]
        tau = run.arr("tau_depth")
        inv = np.where(np.isfinite(tau), 1 / tau, 0.0)
        loom = np.asarray(series[("approach", light)], dtype=np.float64)
        loomt = run.arr(truth_key).astype(np.float64)
        speed = run.arr("speed")
        steady = speed >= speed.max() - 1e-9
        k = np.nonzero(steady & (np.arange(len(steady)) >= a + SKIP) & (np.arange(len(steady)) < b))[0]
        best = max(((lag, _corr(inv[k - lag], loom[k])) for lag in range(31)), key=lambda p: -1 if math.isnan(p[1]) else p[1])
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

        peak = float(loom[k].max())
        L["approach"] = dict(first_03s=round(float(loom[k[:30]].mean()), 5), steady_frames=[int(k[0]), int(k[-1]) + 1],
                             last_03s=round(float(loom[k[-30:]].mean()), 5), peak=round(peak, 5),
                             peak_inv_tau=round(float(inv[k][np.argmax(loom[k])]), 3),
                             inv_tau_range=[round(float(inv[k].min()), 3), round(float(inv[k].max()), 3)],
                             corr_lag0=round(_corr(inv[k], loom[k]), 3), best_lag_ms=10 * best[0], corr_best=round(best[1], 3),
                             truth_corr=round(_corr(inv[k], loomt[k]), 3), truth_peak=round(float(loomt[k].max()), 4),
                             bin_edges=edges.round(2).tolist(), bin_means=[round(v, 5) for v in bins],
                             truth_bin_means=[round(v, 4) for v in binst], saturation_inv_tau=sat,
                             linear_slope_per_inv_tau=round(fit(inv[k], loom[k])[0], 4))
        L["threshold"] = round(thr, 5)
        L["threshold_with_straight"] = round(thr_all, 5)
        L["detect"] = detect(loom, thr) if thr > 0 or loom[k].max() > 0 else None
        L["threshold_truth"] = round(thr_truth, 5)
        L["detect_truth"] = detect(loomt, thr_truth)
        L["margin_peak"] = round(peak / thr, 3) if thr > 0 else None
        L["margin_last"] = round(float(loom[k[-30:]].mean()) / thr, 3) if thr > 0 else None
        L["margin_peak_with_straight"] = round(peak / thr_all, 3) if thr_all > 0 else None
        L["rotation_false_alarm_max"] = round(max(fa[s]["max"] for s in ("yaw", "pitch", "roll")), 5)
        L["_series"] = (run.arr("t"), loom, loomt, inv, a, b)
    return out
