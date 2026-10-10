"""
The Matrice 350 heard by nine microphones in a field, and found again from what they record.

The aircraft, the wind, the autopilot and the mission are rigs/m350_rig.py, flown here without a
window or a renderer. Each step logs the four rotor speeds and the four hubs in NED; the rotors'
sound is then made offline, carried to every microphone with retarded time (which gives the
Doppler) and spherical spreading, the microphones' own noise is added, and a GCC-PHAT plus
SRP-PHAT localizer says where the drone is in every window. numpy only (np.fft), matplotlib Agg
for the figure.

    python m350_acoustic.py --checks                  # the three checks, a line each
    python m350_acoustic.py --flight A                # the tuning flight: a table, an npz, a csv, a figure
    python m350_acoustic.py --flight B --snr 10       # the held-out flight, noisier microphones
    python m350_acoustic.py --flight B --ground-reflection
    python m350_acoustic.py --report                  # the four rows: A 20 dB, B 20 dB, B 10 dB, B 20 dB + ground

Outputs go to out/acoustic/ beside this file (gitignored).

WHAT IS WHOSE
-------------
REAL (the rig's): the flight dynamics, the four rotor speeds at every step, the hub positions,
the wind, the autopilot. STANDARD (textbook): spherical spreading 1/r, retarded time, GCC-PHAT,
SRP-PHAT. ASSUMED, every number of it: the rotor's sound (a harmonic series at the blade-pass
frequency with a 1/k rolloff, a broadband part whose level rises with the rotor speed, the mix
between them) and the microphones' noise. No recording of an M350 was used. Units are
arbitrary: the signal-to-noise ratio at the 50 m reference distance is the only level that means
anything. The demo is about GEOMETRY and TIMING; the timbre is a placeholder. Not modelled: air
absorption, rotor directivity, wind noise on the microphones, occlusion; the ground reflection
is a flag, off by default.

The truth a window is scored against is the centroid of the four hubs at the time the sound
left them: the window's centre less the estimate's distance to the array's centre over c.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # python/ (threepp)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rigs"))   # examples/rigs/ (m350_rig)

from m350_rig import M350, Autopilot, Mission, Wind, load_spec  # noqa: E402

OUT = os.path.join(HERE, "out", "acoustic")

C = 343.0             # speed of sound, m/s
FS = 16000            # sample rate, Hz
REF = 50.0            # the reference distance of the level convention, m
HARMONICS = 12
BROADBAND = 0.5       # broadband RMS over tonal RMS at hover (ASSUMED; tuned on flight A only)
LOWPASS = 3000.0      # the broadband part's one-pole corner, Hz (ASSUMED)
REFLECTION = 0.6      # the ground's coefficient with --ground-reflection (ASSUMED)

# The array: a parameter, not a constant of nature.
# ring_alt_height: every other ring microphone on a mast of that height (None: a planar ring).
ARRAY = dict(ring_radius=15.0, ring_count=8, ring_height=1.5, ring_alt_height=None, jitter=1.0, jitter_seed=0,
             mast_height=6.0)

# The localizer, every number chosen on flight A.
LOC = dict(window=0.5, hop=0.125, upsample=4, band=(100.0, 4000.0),
           grid=dict(ne=(-100.0, 100.0), h=(2.0, 80.0), step=2.0),
           keep=2000, final=0.05, start=1.0)

FLIGHTS = {
    "A": dict(start=(-60.0, -40.0, 25.0), wind=(4.0, 250.0), seed=1, cap=80.0,
              legs=[dict(pos=(60.0, 40.0, 25.0), speed=5.0, yaw="track", hold=2.0),
                    dict(pos=(-40.0, 50.0, 35.0), speed=5.0, yaw="track", hold=1.0)]),
    "B": dict(start=(-70.0, 20.0, 40.0), wind=(8.0, 300.0), seed=3, cap=80.0,
              legs=[dict(pos=(30.0, 60.0, 40.0), speed=6.0, yaw="track", hold=0.0),
                    dict(pos=(50.0, -50.0, 30.0), speed=6.0, yaw="track", hold=1.0)]),
}


# --------------------------------------------------------------------------- #
#  the array, the flight
# --------------------------------------------------------------------------- #
def microphones(a=ARRAY):
    """(9, 3) NED: the ring (north first, clockwise seen from above), then the mast."""
    rng = np.random.default_rng(a["jitter_seed"])
    mics = []
    for k in range(a["ring_count"]):
        th = 2.0 * math.pi * k / a["ring_count"]
        jn, je = rng.uniform(-a["jitter"], a["jitter"], 2)
        h = a["ring_height"] if (k % 2 == 0 or a["ring_alt_height"] is None) else a["ring_alt_height"]
        mics.append((a["ring_radius"] * math.cos(th) + jn, a["ring_radius"] * math.sin(th) + je, -h))
    mics.append((0.0, 0.0, -a["mast_height"]))
    return np.array(mics)


def fly(name, spec=None):
    """Fly a flight from its hover; log every step: t, position, quaternion, rotor speeds, hubs."""
    F = FLIGHTS[name]
    spec = spec or load_spec()
    wind = Wind(F["wind"][0], F["wind"][1], turbulence=1.0, seed=F["seed"], spec=spec)
    m = M350(spec, wind=wind)
    n, e, h = F["start"]
    m.set_state(n=n, e=e, h=h)
    ap = Autopilot(m)
    mission = Mission(F["legs"], spec)
    t, x, hubs = [0.0], [list(m.x)], [m.rotor_hubs_ned()]
    while not mission.done and m.t < F["cap"]:
        mission.update(m, ap)
        ap.update()
        m.step()
        t.append(m.t)
        x.append(list(m.x))
        hubs.append(m.rotor_hubs_ned())
    x = np.array(x)
    return dict(t=np.array(t), pos=x[:, 0:3], quat=x[:, 6:10], w=x[:, 13:17], hubs=np.array(hubs),
                w_hover=m.hover_speed(), blades=spec["aircraft"]["propeller"]["blades"])


def flight_log(name, cache=True):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"flight{name}_log.npz")
    if cache and os.path.exists(path):
        z = np.load(path)
        return {k: (z[k] if z[k].ndim else z[k].item()) for k in z.files}
    log = fly(name)
    if cache:
        np.savez(path, **log)
    return log


# --------------------------------------------------------------------------- #
#  the source and the propagation
# --------------------------------------------------------------------------- #
def lowpass_unit(n, rng):
    """Unit white noise through a one-pole low-pass at LOWPASS, applied in the frequency domain."""
    a = math.exp(-2.0 * math.pi * LOWPASS / FS)
    X = np.fft.rfft(rng.standard_normal(n))
    z = np.exp(-2j * math.pi * np.arange(X.size) / n)
    return np.fft.irfft(X * (1.0 - a) / (1.0 - a * z), n)


def tonal_rms():
    return math.sqrt(0.5 * sum(1.0 / (k * k) for k in range(1, HARMONICS + 1)))


def sources(t_log, w_log, w_hover, blades, n, seed, broadband=BROADBAND):
    """(n_rotors, n) source signals on the emission clock k / FS, one rotor at hover = RMS 1."""
    a = math.exp(-2.0 * math.pi * LOWPASS / FS)
    bb_unit = math.sqrt((1.0 - a) / (1.0 + a))                      # the filtered noise's RMS
    t = np.arange(n) / FS
    rng = np.random.default_rng(seed)
    psi = rng.uniform(0.0, 2.0 * math.pi, (w_log.shape[1], HARMONICS))
    tr = tonal_rms()
    gain_bb = broadband * tr / bb_unit
    level = 1.0 / (tr * math.sqrt(1.0 + broadband * broadband))
    out = np.empty((w_log.shape[1], n))
    for i in range(w_log.shape[1]):
        w = np.interp(t, t_log, w_log[:, i])
        phi = blades * np.cumsum(w) / FS
        s = np.zeros(n)
        for k in range(1, HARMONICS + 1):
            s += np.cos(k * phi + psi[i, k - 1]) / k
        bb = lowpass_unit(n, np.random.default_rng(seed + 1 + i))
        out[i] = level * (s + gain_bb * (w / w_hover) ** 2.5 * bb)
    return out


def propagate(t_log, src_pos, src, mics, snr_db, seed, reflection=False):
    """Microphone signals (n_mics, n). src_pos (n_src, n_log, 3) NED, src (n_src, n) on the
    emission clock. Retarded time: an emission at t_e arrives at t_e + r(t_e) / c."""
    n = src.shape[1]
    t = np.arange(n) / FS
    out = np.zeros((len(mics), n))
    images = [(src_pos, 1.0)]
    if reflection:
        img = src_pos.copy()
        img[..., 2] *= -1.0                                         # mirrored in the ground plane d = 0
        images.append((img, REFLECTION))
    for m, mic in enumerate(mics):
        for pos, coef in images:
            for i in range(src.shape[0]):
                r = np.linalg.norm(pos[i] - mic, axis=1)
                ta = t_log + r / C
                te = np.interp(t, ta, t_log)
                g = coef * REF / np.interp(te, t_log, r)
                y = g * np.interp(te * FS, np.arange(n), src[i])
                y[t < ta[0]] = 0.0
                out[m] += y
    noise = 2.0 * 10.0 ** (-snr_db / 20.0)                          # four rotors at 50 m: RMS 2
    rng = np.random.default_rng(seed + 100)
    out += noise * rng.standard_normal(out.shape)
    return out


# --------------------------------------------------------------------------- #
#  the localizer: GCC-PHAT per pair, SRP-PHAT over a grid, two refinements
# --------------------------------------------------------------------------- #
class Localizer:
    def __init__(self, mics, window=LOC["window"], loc=LOC):
        self.mics = mics
        self.loc = loc
        self.nw = int(round(window * FS))
        self.nfft = 1 << int(math.ceil(math.log2(2 * self.nw)))
        self.up = loc["upsample"]
        self.fs_up = FS * self.up
        self.pairs = [(a, b) for a in range(len(mics)) for b in range(a + 1, len(mics))]
        base = max(np.linalg.norm(mics[a] - mics[b]) for a, b in self.pairs)
        self.L = int(math.ceil(base / C * self.fs_up)) + 2          # lags kept: +-L samples
        self.nlag = 2 * self.L + 1
        f = np.fft.rfftfreq(self.nfft, 1.0 / FS)
        self.band = (f >= loc["band"][0]) & (f <= loc["band"][1])
        self.hann = np.hanning(self.nw)
        g = loc["grid"]
        ne = np.arange(g["ne"][0], g["ne"][1] + 1e-9, g["step"])
        hh = np.arange(g["h"][0], g["h"][1] + 1e-9, g["step"])
        N, E, H = np.meshgrid(ne, ne, hh, indexing="ij")
        self.grid = np.stack([N.ravel(), E.ravel(), -H.ravel()], axis=1)
        self.levels = int(math.ceil(math.log2(2.5 * math.sqrt(3.0) * 0.5 * g["step"] / C * self.fs_up + 2.0))) + 1
        self.grid_idx = self._index(self.grid, 0.5 * math.sqrt(3.0) * g["step"])

    def _index(self, P, half_diag):
        """Per pair, the flat index into the stack of max-filtered GCCs: the lag the point
        predicts, at the level that covers the lags its cell spans."""
        d = [np.maximum(np.linalg.norm(P - m, axis=1), 1e-6) for m in self.mics]
        u = [(P - m) / dm[:, None] for m, dm in zip(self.mics, d)]
        idx = np.empty((len(self.pairs), len(P)), np.int32)
        for p, (a, b) in enumerate(self.pairs):
            lag = np.rint((d[a] - d[b]) / C * self.fs_up).astype(np.int64) + self.L
            W = np.linalg.norm(u[a] - u[b], axis=1) * half_diag / C * self.fs_up * 1.25 + (1.0 if half_diag > 0 else 0.0)
            lev = np.clip(np.ceil(np.log2(W + 1.0)), 0, self.levels - 1).astype(np.int64)
            idx[p] = lev * self.nlag + np.clip(lag, 0, self.nlag - 1)
        return idx

    def gcc(self, x):
        """(n_pairs, levels * nlag): per pair the PHAT GCC (mean cosine over the band's bins),
        then max-filtered over +-(2^k - 1) samples for level k."""
        X = np.fft.rfft(x * self.hann, self.nfft, axis=1)
        nb = int(self.band.sum())
        n_up = self.nfft * self.up
        stacks = np.empty((len(self.pairs), self.levels, self.nlag), np.float32)
        for p, (a, b) in enumerate(self.pairs):
            G = X[a] * np.conj(X[b])
            G = np.where(self.band, G / (np.abs(G) + 1e-12), 0.0)
            cc = np.fft.irfft(G, n_up) * (n_up / (2.0 * nb))
            M = np.concatenate([cc[-self.L:], cc[:self.L + 1]])
            stacks[p, 0] = M
            for k in range(1, self.levels):
                s = 1 << (k - 1)
                lo = np.concatenate([np.full(s, M[0]), M[:-s]])
                hi = np.concatenate([M[s:], np.full(s, M[-1])])
                M = np.maximum(M, np.maximum(lo, hi))
                stacks[p, k] = M
        return stacks.reshape(len(self.pairs), -1)

    def srp(self, stacks, idx):
        s = np.zeros(idx.shape[1], np.float32)
        for p in range(len(self.pairs)):
            s += stacks[p][idx[p]]
        return s

    def locate(self, x):
        """One window (n_mics, nw): the estimate (NED) and the coarse map's peak over its mean.

        A cell's max-filtered SRP is an UPPER bound on the SRP anywhere in it; the sharp SRP at
        its centre is a LOWER bound on the maximum. Branch and bound: keep the cells whose upper
        bound reaches the best centre, split each into eight, repeat down to `final` m. (The
        coarse map alone is a plateau many metres long along the range; one refine box about
        its argmax lands anywhere on it.)"""
        stacks = self.gcc(x)
        s = self.srp(stacks, self.grid_idx)
        conf = float(s.max() / max(s.mean(), 1e-9))
        K = self.loc["keep"]
        top = np.argpartition(-s, K)[:K] if len(s) > K else np.arange(len(s))
        P, U, step = self.grid[top], s[top], self.loc["grid"]["step"]
        best, best_v = None, -np.inf
        while True:
            Lc = self.srp(stacks, self._index(P, 0.0))
            k = int(np.argmax(Lc))
            if Lc[k] > best_v:
                best, best_v = P[k], float(Lc[k])
            if step <= self.loc["final"]:
                return best, conf
            P = P[U >= best_v]
            if len(P) > K:
                P = P[np.argsort(-U[U >= best_v])[:K]]
            step *= 0.5
            o = 0.5 * step * np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)], float)
            P = (P[:, None, :] + o[None, :, :]).reshape(-1, 3)
            P = P[-P[:, 2] >= self.loc["grid"]["h"][0] - 0.5 * step]
            U = self.srp(stacks, self._index(P, 0.5 * math.sqrt(3.0) * step))


def run(name, snr=20.0, reflection=False, window=LOC["window"], hop=LOC["hop"], seed=None, broadband=BROADBAND,
        cache=True, quiet=False, every=2.0, array=None):
    """Fly (or load the flight), synthesise, localize; returns a dict of everything."""
    seed = FLIGHTS[name]["seed"] if seed is None else seed
    log = flight_log(name, cache=cache)
    mics = microphones(ARRAY if array is None else array)
    t_log = log["t"]
    n = int(t_log[-1] * FS)
    src = sources(t_log, log["w"], float(log["w_hover"]), int(log["blades"]), n, seed, broadband)
    sig = propagate(t_log, np.transpose(log["hubs"], (1, 0, 2)), src, mics, snr, seed, reflection)
    loc = Localizer(mics, window)
    centre = mics.mean(axis=0)
    truth_log = log["hubs"].mean(axis=1)                            # the hubs' centroid, NED
    nh = int(round(hop * FS))
    rows = []
    i0 = int(round((LOC["start"]) * FS))
    while i0 + loc.nw <= n:
        tc = (i0 + 0.5 * loc.nw) / FS
        x = sig[:, i0:i0 + loc.nw]
        est, conf = loc.locate(x)
        te = tc - np.linalg.norm(est - centre) / C                  # the emission-time correction
        tru = np.array([np.interp(te, t_log, truth_log[:, k]) for k in range(3)])
        energy = (x * x).sum(axis=1)
        n2 = np.array([*(energy @ mics[:, :2] / energy.sum()), -30.0])
        n1 = np.array([centre[0], centre[1], -30.0])
        rows.append([tc, te, *est, *tru, np.linalg.norm(est - tru), np.linalg.norm(est[:2] - tru[:2]), conf,
                     np.linalg.norm(n1 - tru), *n2, np.linalg.norm(n2 - tru), tru[2] - est[2]])
        i0 += nh
    R = np.array(rows)
    res = dict(name=name, snr=snr, reflection=reflection, window=window, hop=hop, seed=seed, broadband=broadband,
               log=log, mics=mics, sig=sig, rows=R)
    if not quiet:
        print(f"flight {name}: {t_log[-1]:.1f} s flown, {len(R)} windows of {window} s every {hop} s, SNR {snr:g} dB at "
              f"{REF:g} m{', ground reflection' if reflection else ''}, acoustic seed {seed} (timbre and units ASSUMED)")
        print("     t   estimate n, e, h (m)        truth n, e, h (m)         3D err  horiz height  conf   null1  null2")
        nxt = 0.0
        for r in R:
            if r[0] >= nxt:
                print(f"{r[0]:6.2f}  {r[2]:7.2f} {r[3]:7.2f} {-r[4]:6.2f}   {r[5]:7.2f} {r[6]:7.2f} {-r[7]:6.2f}   "
                      f"{r[8]:6.2f} {r[9]:6.2f} {r[16]:+6.2f} {r[10]:5.1f}  {r[11]:6.1f} {r[15]:6.1f}")
                nxt = r[0] + every
        print(summary_line(res))
    return res


def summary(res):
    R = res["rows"]
    return dict(N=len(R), med=float(np.median(R[:, 8])), p90=float(np.percentile(R[:, 8], 90)),
                hmed=float(np.median(R[:, 9])), vmed=float(np.median(np.abs(R[:, 16]))), null1=float(np.median(R[:, 11])), null2=float(np.median(R[:, 15])))


def label(res):
    return (f"{res['name']} {res['snr']:g} dB{' + ground' if res['reflection'] else ''}"
            f"{' (tuning)' if res['name'] == 'A' else ''}")


def summary_line(res):
    s = summary(res)
    return (f"{label(res):<22} N {s['N']:4d}  median 3D {s['med']:6.2f} m  p90 {s['p90']:6.2f} m  horizontal median "
            f"{s['hmed']:6.2f} m  height median {s['vmed']:5.2f} m  Null 1 {s['null1']:6.1f} m  Null 2 {s['null2']:6.1f} m")


def tag(res):
    return (f"snr{res['snr']:g}{'_gr' if res['reflection'] else ''}_w{res['window']:g}_h{res['hop']:g}"
            f"_s{res['seed']}")


def versioned(stem, ext):
    k = 1
    while os.path.exists(f"{stem}_v{k:02d}{ext}"):
        k += 1
    return f"{stem}_v{k:02d}{ext}"


def save(res):
    """npz (log, microphone signals, estimates), csv (per window), png (plan view, error in time)."""
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, f"flight{res['name']}_{tag(res)}")
    log = res["log"]
    np.savez_compressed(stem + ".npz", t=log["t"], pos=log["pos"], quat=log["quat"], w=log["w"], hubs=log["hubs"],
                        mics=res["mics"], fs=FS, signals=res["sig"].astype(np.float32), windows=res["rows"])
    head = ("t_c,t_emission,est_n,est_e,est_d,truth_n,truth_e,truth_d,err3d,err_horizontal,confidence,null1_err,"
            "null2_n,null2_e,null2_d,null2_err,err_height")
    np.savetxt(stem + ".csv", res["rows"], delimiter=",", header=head, comments="", fmt="%.4f")
    png = figure(res, versioned(stem, ".png"))
    return stem + ".npz", stem + ".csv", png


def figure(res, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R, mics, log = res["rows"], res["mics"], res["log"]
    tr = log["hubs"].mean(axis=1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
    a1.plot(tr[:, 1], tr[:, 0], color="0.35", lw=1.2, label="truth (hub centroid)")
    a1.scatter(R[:, 13], R[:, 12], s=6, color="tab:orange", alpha=0.6, label="Null 2 (energy-weighted centroid)")
    a1.scatter(R[:, 3], R[:, 2], s=6, color="tab:blue", label="SRP-PHAT estimate")
    a1.scatter(mics[:, 1], mics[:, 0], marker="^", s=60, color="k", label="microphones")
    a1.set_xlabel("east (m)")
    a1.set_ylabel("north (m)")
    a1.set_aspect("equal")
    a1.grid(alpha=0.3)
    a1.legend(loc="best", fontsize=8)
    a1.set_title(f"flight {label(res)}: plan view")
    a2.semilogy(R[:, 0], R[:, 8], ".", ms=3, color="tab:blue", label="SRP-PHAT 3D error")
    a2.semilogy(R[:, 0], R[:, 9], ".", ms=2, color="tab:green", alpha=0.7, label="SRP-PHAT horizontal error")
    a2.semilogy(R[:, 0], np.abs(R[:, 16]), ".", ms=2, color="tab:purple", alpha=0.5, label="SRP-PHAT |height error|")
    a2.semilogy(R[:, 0], R[:, 11], "-", color="0.5", lw=1, label="Null 1: array centre at 30 m")
    a2.semilogy(R[:, 0], R[:, 15], "-", color="tab:orange", lw=1, label="Null 2: energy-weighted centroid at 30 m")
    a2.set_xlabel("window centre (s)")
    a2.set_ylabel("3D error (m)")
    a2.grid(alpha=0.3, which="both")
    a2.legend(loc="best", fontsize=8)
    s = summary(res)
    a2.set_title(f"median {s['med']:.2f} m, p90 {s['p90']:.2f} m; nulls {s['null1']:.0f} m and {s['null2']:.0f} m "
                 f"(timbre ASSUMED)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
#  the three checks
# --------------------------------------------------------------------------- #
def check_static():
    mics = microphones()
    p = np.array([20.0, -10.0, -25.0])
    w_h = M350().hover_speed()
    blades = load_spec()["aircraft"]["propeller"]["blades"]
    t_log = np.array([0.0, 2.0])
    n = int(2.0 * FS)
    src = sources(t_log, np.full((2, 1), w_h), w_h, blades, n, seed=5)
    sig = propagate(t_log, np.repeat(p[None, None, :], 2, axis=1), src, mics, 30.0, seed=5)
    loc = Localizer(mics)
    i0 = int(1.0 * FS)
    est, conf = loc.locate(sig[:, i0:i0 + loc.nw])
    err = float(np.linalg.norm(est - p))
    centre = mics.mean(axis=0)
    e1 = float(np.linalg.norm(np.array([centre[0], centre[1], -30.0]) - p))
    ok = err <= 0.3 and e1 > 20.0
    print(f"[1] static source at (20, -10, 25 up) m, one rotor's sound at hover ({blades * w_h / (2 * math.pi):.1f} Hz "
          f"blade-pass), SNR 30 dB, one {LOC['window']} s window: estimate ({est[0]:.2f}, {est[1]:.2f}, {-est[2]:.2f}), "
          f"error {err:.3f} m (asked <= 0.3), confidence {conf:.1f}; Null 1 error {e1:.1f} m (asked > 20): "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_determinism():
    a = run("A", cache=False, quiet=True)
    b = run("A", cache=False, quiet=True)
    same_sig = np.array_equal(a["sig"], b["sig"])
    same_est = np.array_equal(a["rows"], b["rows"])
    ok = same_sig and same_est
    print(f"[2] two runs of flight A at seed {a['seed']} from the flight on ({a['sig'].shape[1]} samples x "
          f"{a['sig'].shape[0]} microphones, {len(a['rows'])} windows): signals "
          f"{'bit-identical' if same_sig else 'DIFFER'}, estimates {'bit-identical' if same_est else 'DIFFER'}: "
          f"{'PASS' if ok else 'FAIL'}")
    return ok, a


def peak_freq(x, f_lo, f_hi):
    nfft = 1 << 21
    X = np.abs(np.fft.rfft(x * np.hanning(len(x)), nfft))
    f = np.fft.rfftfreq(nfft, 1.0 / FS)
    sel = np.where((f >= f_lo) & (f <= f_hi))[0]
    k = sel[int(np.argmax(X[sel]))]
    y0, y1, y2 = np.log(X[k - 1:k + 2])
    return (k + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)) * FS / nfft


def check_doppler(log, mics, span=1.0, offset=3.0, rotor=0, seed=1):
    """The mast microphone hears ONE rotor (fr) of flight A through propagate(); its strongest
    peak 3 s before the overhead pass over the same peak 3 s after, each divided by the rotor's
    own blade-pass at the emission (the log's; it changes a little between the two). The four
    rotors together will not do: in forward flight the front pair turns at about 247 and 251
    rad/s and the rear pair at 268 and 273, lines 1.5 Hz apart, as far apart as the Doppler
    shift, and the strongest peak of the mix jumps between rotors."""
    mast = mics[-1]
    t_log = log["t"]
    hub = log["hubs"][:, rotor]
    leg1 = t_log < t_log[-1] * 0.6
    hd = np.hypot(hub[:, 0] - mast[0], hub[:, 1] - mast[1])
    t_pass = t_log[leg1][int(np.argmin(hd[leg1]))]
    n = int(t_log[-1] * FS)
    src = sources(t_log, log["w"][:, rotor:rotor + 1], float(log["w_hover"]), int(log["blades"]), n, seed)
    sig = propagate(t_log, hub[None], src, mast[None], 20.0, seed)[0]
    r = np.linalg.norm(hub - mast, axis=1)
    rdot = np.gradient(r, t_log)
    bpf = log["blades"] * log["w"][:, rotor] / (2.0 * math.pi)
    out = []
    for tm in (t_pass - offset, t_pass + offset):
        ta = tm + np.interp(tm, t_log, r) / C                       # the receive time of the emission at tm
        i0 = int((ta - 0.5 * span) * FS)
        sel = (t_log >= tm - 0.5 * span) & (t_log <= tm + 0.5 * span)
        f0 = float(bpf[sel].mean())
        f = peak_freq(sig[i0:i0 + int(span * FS)], 0.9 * f0, 1.1 * f0)
        out.append((f, float(rdot[sel].mean()), f0))
    (fa, va, f0a), (fr, vr, f0r) = out
    v = 0.5 * (-va + vr)                                            # the mean radial speed, both positive
    pred = (C + v) / (C - v)
    meas = (fa / f0a) / (fr / f0r)
    rel = (meas - 1.0) / (pred - 1.0)
    ok = abs(rel - 1.0) <= 0.2
    print(f"[3] Doppler at the mast, flight A's overhead pass at {t_pass:.1f} s, rotor fr alone: strongest peak {fa:.3f} Hz "
          f"approaching (3 s before, radial speed {-va:.2f} m/s, source {f0a:.3f} Hz) and {fr:.3f} Hz receding (3 s after, "
          f"{vr:.2f} m/s, source {f0r:.3f} Hz): ratio {meas:.5f} against (c + v_r) / (c - v_r) = {pred:.5f}; the shift "
          f"{100 * (meas - 1):.3f} % against {100 * (pred - 1):.3f} % ({100 * (rel - 1):+.1f} %, asked within 20 %): "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def checks():
    t0 = time.time()
    ok1 = check_static()
    ok2, a = check_determinism()
    ok3 = check_doppler(a["log"], a["mics"])
    print(f"checks: {sum((ok1, ok2, ok3))} of 3 pass ({time.time() - t0:.0f} s)")
    return ok1 and ok2 and ok3


# --------------------------------------------------------------------------- #
#  the report
# --------------------------------------------------------------------------- #
def report():
    rows = [("A", 20.0, False), ("B", 20.0, False), ("B", 10.0, False), ("B", 20.0, True)]
    lines, figs = [], []
    res_all = []
    for name, snr, gr in rows:
        res = run(name, snr=snr, reflection=gr, quiet=True)
        figs.append(save(res)[2])
        res_all.append(res)
        print(summary_line(res))
    log = res_all[0]["log"]
    bpf = log["blades"] * float(log["w_hover"]) / (2.0 * math.pi)
    lines.append("M350 acoustic localization: GCC-PHAT + SRP-PHAT on nine microphones (8 on a 15 m ring 1.5 m up, "
                 "1 on a 6 m mast)")
    lines.append(f"window {LOC['window']} s, hop {LOC['hop']} s, BROADBAND {BROADBAND} (broadband / tonal RMS at hover), "
                 f"band {LOC['band'][0]:g}-{LOC['band'][1]:g} Hz, GCC upsampled x{LOC['upsample']}, grid "
                 f"{LOC['grid']['step']:g} m, branch and bound to {LOC['final']:g} m; hover blade-pass {bpf:.1f} Hz")
    lines.append("ASSUMED: the rotor timbre (12 harmonics at 1/k, broadband rising as speed^2.5), the noise; units "
                 "arbitrary, SNR at 50 m the only level. Truth: the hub centroid at the emission time.")
    lines.append("")
    lines.append("| flight | N | median 3D (m) | p90 3D (m) | horizontal median (m) | Null 1 median (m) | Null 2 median (m) |")
    lines.append("|---|---|---|---|---|---|---|")
    for res in res_all:
        s = summary(res)
        lines.append(f"| {label(res)} | {s['N']} | {s['med']:.2f} | {s['p90']:.2f} | {s['hmed']:.2f} | {s['null1']:.1f} | "
                     f"{s['null2']:.1f} |")
    sB = summary(res_all[1])
    gate = sB["med"] <= 2.0 and sB["null1"] >= 10.0 * sB["med"] and sB["null2"] >= 10.0 * sB["med"]
    lines.append("")
    lines.append(f"gate (B 20 dB: median 3D <= 2 m, each null >= 10x worse): {'PASS' if gate else 'FAIL'} "
                 f"(median {sB['med']:.2f} m; nulls {sB['null1'] / sB['med']:.0f}x and {sB['null2'] / sB['med']:.0f}x)")
    lines.append("figures: " + ", ".join(os.path.basename(f) for f in figs))
    text = "\n".join(lines)
    print()
    print(text)
    path = versioned(os.path.join(OUT, "report"), ".txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"written: {path}")
    return gate


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checks", action="store_true", help="the three checks")
    ap.add_argument("--flight", choices=("A", "B"), default=None)
    ap.add_argument("--snr", type=float, default=20.0, help="dB, the microphones' noise against the four rotors at 50 m")
    ap.add_argument("--ground-reflection", action="store_true")
    ap.add_argument("--window", type=float, default=LOC["window"], help="s")
    ap.add_argument("--hop", type=float, default=LOC["hop"], help="s")
    ap.add_argument("--seed", type=int, default=None, help="the acoustic seed (default: the flight's)")
    ap.add_argument("--broadband", type=float, default=BROADBAND, help="broadband over tonal RMS at hover (tuning)")
    ap.add_argument("--report", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    if a.checks:
        return 0 if checks() else 1
    if a.report:
        return 0 if report() else 1
    if a.flight:
        res = run(a.flight, snr=a.snr, reflection=a.ground_reflection, window=a.window, hop=a.hop, seed=a.seed,
                  broadband=a.broadband)
        for p in save(res):
            print(f"written: {p}")
        return 0
    parse(["--help"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
