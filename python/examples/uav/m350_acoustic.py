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
    python m350_acoustic.py --flight B --snr 35 --ambient pink    # the film's run
    python m350_acoustic.py --shot 18 --flight B      # one frame of the film (35 dB pink unless told)
    python m350_acoustic.py --film --flight B         # the film, its stereo track and four frames

Outputs go to out/acoustic/ beside this file (gitignored). The film is a headless GL view (the
x8_flight.py recipe) beside a matplotlib panel, piped to ffmpeg (imageio_ffmpeg) with the ring's
westmost and eastmost microphones as the left and right channels.

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


def pink(x):
    """The same white noise shaped by 1 / sqrt(f) (f >= 1 Hz, the DC bin zero), each row
    rescaled to the RMS the white noise had: a quiet field's floor, low-frequency."""
    f = np.fft.rfftfreq(x.shape[-1], 1.0 / FS)
    g = 1.0 / np.sqrt(np.maximum(f, 1.0))
    g[0] = 0.0
    y = np.fft.irfft(np.fft.rfft(x, axis=-1) * g, x.shape[-1], axis=-1)
    return y * (np.sqrt((x * x).mean(axis=-1)) / np.sqrt((y * y).mean(axis=-1)))[..., None]


def propagate(t_log, src_pos, src, mics, snr_db, seed, reflection=False, ambient="white"):
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
    white = rng.standard_normal(out.shape)
    out += noise * (pink(white) if ambient == "pink" else white)
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
        cache=True, quiet=False, every=2.0, array=None, ambient="white"):
    """Fly (or load the flight), synthesise, localize; returns a dict of everything."""
    seed = FLIGHTS[name]["seed"] if seed is None else seed
    log = flight_log(name, cache=cache)
    mics = microphones(ARRAY if array is None else array)
    t_log = log["t"]
    n = int(t_log[-1] * FS)
    src = sources(t_log, log["w"], float(log["w_hover"]), int(log["blades"]), n, seed, broadband)
    sig = propagate(t_log, np.transpose(log["hubs"], (1, 0, 2)), src, mics, snr, seed, reflection, ambient)
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
               ambient=ambient, log=log, mics=mics, sig=sig, rows=R)
    if not quiet:
        print(f"flight {name}: {t_log[-1]:.1f} s flown, {len(R)} windows of {window} s every {hop} s, SNR {snr:g} dB "
              f"{ambient} at "
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
    return (f"{res['name']} {res['snr']:g} dB{' pink' if res['ambient'] == 'pink' else ''}"
            f"{' + ground' if res['reflection'] else ''}{' (tuning)' if res['name'] == 'A' else ''}")


def summary_line(res):
    s = summary(res)
    return (f"{label(res):<22} N {s['N']:4d}  median 3D {s['med']:6.2f} m  p90 {s['p90']:6.2f} m  horizontal median "
            f"{s['hmed']:6.2f} m  height median {s['vmed']:5.2f} m  Null 1 {s['null1']:6.1f} m  Null 2 {s['null2']:6.1f} m")


def tag(res):
    return (f"snr{res['snr']:g}{'_pink' if res['ambient'] == 'pink' else ''}{'_gr' if res['reflection'] else ''}"
            f"_w{res['window']:g}_h{res['hop']:g}_s{res['seed']}")


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


# --------------------------------------------------------------------------- #
#  the film: a headless GL view of the field beside the localizer's panel, two ring
#  microphones as the stereo track
# --------------------------------------------------------------------------- #
FPS, VIEW_W, PANEL_W, FILM_H = 30, 1280, 640, 1080
INSET = (426, 360)          # the close-up, drawn at 1:1 scale, top-left of the 3D view (over the sky)
INSET_AT = (12, 12)         # its frame's top-left corner, px
CLOSE = 9.0                 # the close-up camera's distance from the aircraft, m
TRAIL = 10                  # the last estimates shown


def result(name, snr, ambient, reflection, window, hop, seed):
    """A tag's run from its npz when one was saved, else run it and save it."""
    seed = FLIGHTS[name]["seed"] if seed is None else seed
    res = dict(name=name, snr=snr, ambient=ambient, reflection=reflection, window=window, hop=hop, seed=seed,
               broadband=BROADBAND)
    path = os.path.join(OUT, f"flight{name}_{tag(res)}.npz")
    if not os.path.exists(path):
        res = run(name, snr=snr, reflection=reflection, window=window, hop=hop, seed=seed, ambient=ambient)
        save(res)
        return res
    z = np.load(path)
    res.update(log=flight_log(name), mics=z["mics"], sig=z["signals"].astype(np.float64), rows=z["windows"])
    print(summary_line(res))
    return res


def stereo(res, path):
    """The ring microphones with the smallest (left) and the largest (right) east coordinate,
    16 kHz, 16 bit, the pair normalised together to a -3 dBFS peak."""
    import wave
    mics, sig = res["mics"], res["sig"]
    ring = np.arange(len(mics) - 1)
    left, right = int(ring[np.argmin(mics[ring, 1])]), int(ring[np.argmax(mics[ring, 1])])
    x = np.stack([sig[left], sig[right]], axis=1)
    x = x * (10.0 ** (-3.0 / 20.0) / np.abs(x).max())
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(FS)
        w.writeframes(np.round(x * 32767.0).astype("<i2").tobytes())
    return left, right


class Panel:
    """640 x 1080, matplotlib Agg: the SRP-PHAT slice at the estimated height above, the 3D
    error against time below. Drawn once per window and reused between them."""

    def __init__(self, res):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        self.res, mics = res, res["mics"]
        self.loc = Localizer(mics, res["window"])
        g = np.arange(-100.0, 100.0 + 1e-9, 2.0)
        self.g = g
        N, E = np.meshgrid(g, g, indexing="ij")
        self.NE = np.stack([N.ravel(), E.ravel()], axis=1)
        self.nh = int(round(res["hop"] * FS))
        T = float(res["log"]["t"][-1])
        bg, fg = "#12161b", "#e6e9ee"
        plt.rcParams.update({"font.size": 12, "text.color": fg, "axes.labelcolor": fg, "xtick.color": fg,
                             "ytick.color": fg, "axes.edgecolor": "#6b7380", "axes.facecolor": bg})
        fig = self.fig = plt.figure(figsize=(PANEL_W / 100.0, FILM_H / 100.0), dpi=100, facecolor=bg)
        self.head = fig.text(0.05, 0.975, "", fontsize=17, fontweight="bold", va="top")
        self.sub = fig.text(0.05, 0.94, "", fontsize=12, va="top", color="#aab2bd")
        a1 = self.a1 = fig.add_axes([0.13, 0.42, 0.8, 0.45])
        self.img = a1.imshow(np.zeros((g.size, g.size)), origin="lower", extent=(-101, 101, -101, 101), cmap="magma",
                             vmin=0.0, vmax=1.0, interpolation="bilinear")
        tr = res["log"]["hubs"].mean(axis=1)
        a1.plot(tr[:, 1], tr[:, 0], color="w", lw=0.8, alpha=0.35)
        a1.scatter(mics[:, 1], mics[:, 0], marker="^", s=36, color="#7fe0ff", edgecolor="k", lw=0.5, zorder=3,
                   label="microphones")
        self.truth = a1.scatter([0], [0], marker="x", s=150, color="w", lw=2.5, zorder=4, label="truth")
        self.est = a1.scatter([0], [0], marker="o", s=60, color="#2fffb0", edgecolor="k", lw=0.8, zorder=5,
                              label="estimate")
        a1.set_xlabel("east (m)")
        a1.set_ylabel("north (m)")
        a1.set_xlim(-100, 100)
        a1.set_ylim(-100, 100)
        self.t1 = a1.set_title("", fontsize=12)
        a1.legend(loc="lower left", fontsize=10, framealpha=0.6, facecolor=bg, edgecolor="#6b7380")
        a2 = self.a2 = fig.add_axes([0.13, 0.06, 0.8, 0.26])
        a2.set_yscale("log")
        self.l_est, = a2.plot([], [], ".", ms=3, color="#2fffb0", label="SRP-PHAT 3D error")
        self.l_n1, = a2.plot([], [], "-", lw=1.2, color="#9aa3ad", label="Null 1: array centre, 30 m up")
        self.l_n2, = a2.plot([], [], "-", lw=1.2, color="#ffa94d", label="Null 2: energy centroid, 30 m up")
        a2.set_xlim(0.0, T)
        a2.set_ylim(0.03, 3000.0)                                           # room for the legend above the nulls
        a2.set_xlabel("time (s)")
        a2.set_ylabel("3D error (m)")
        a2.grid(alpha=0.25, which="major")
        a2.legend(loc="upper right", fontsize=9, framealpha=0.6, facecolor=bg, edgecolor="#6b7380", ncol=1)
        self.t2 = a2.set_title("", fontsize=12)
        fig.text(0.05, 0.005, "rotor sound ASSUMED (harmonics + broadband); truth: hub centroid when the sound left",
                 fontsize=8.5, color="#8a929c", va="bottom")
        self.j = None
        self.rgb = None

    def slice(self, j):
        """The SRP-PHAT over north and east on a 2 m grid at window j's estimated height, each
        cell its upper bound (the GCCs max-filtered over the lags the cell spans), per pair."""
        R, loc = self.res["rows"], self.loc
        i0 = int(round(R[j, 0] * FS - 0.5 * loc.nw))
        stacks = loc.gcc(self.res["sig"][:, i0:i0 + loc.nw])
        P = np.column_stack([self.NE, np.full(len(self.NE), R[j, 4])])
        s = loc.srp(stacks, loc._index(P, math.sqrt(2.0))) / len(loc.pairs)
        return s.reshape(self.g.size, self.g.size)

    def draw(self, j, t):
        res, R = self.res, self.res["rows"]
        self.head.set_text("Where is the drone? Listen.")
        self.sub.set_text(f"nine microphones, GCC-PHAT + SRP-PHAT, {res['window']:g} s windows\n"
                          f"every {res['hop']:g} s; flight {res['name']}, SNR {res['snr']:g} dB {res['ambient']} at 50 m")
        if j < 0:
            self.img.set_data(np.zeros((self.g.size, self.g.size)))
            self.t1.set_text("SRP-PHAT: listening...")
            for a in (self.truth, self.est):
                a.set_offsets(np.empty((0, 2)))
            self.t2.set_text(f"t {t:5.1f} s")
            for ln in (self.l_est, self.l_n1, self.l_n2):
                ln.set_data([], [])
        else:
            s = self.slice(j)
            self.img.set_data(s)
            self.img.set_clim(0.0, max(float(s.max()), 1e-6))
            self.truth.set_offsets([[R[j, 6], R[j, 5]]])
            self.est.set_offsets([[R[j, 3], R[j, 2]]])
            self.t1.set_text(f"SRP-PHAT slice at the estimated height, {-R[j, 4]:.1f} m up")
            k = slice(0, j + 1)
            self.l_est.set_data(R[k, 0], R[k, 8])
            self.l_n1.set_data(R[k, 0], R[k, 11])
            self.l_n2.set_data(R[k, 0], R[k, 15])
            self.t2.set_text(f"t {R[j, 0]:5.2f} s   error {R[j, 8]:.2f} m   median so far "
                             f"{np.median(R[k, 8]):.2f} m")
        self.fig.canvas.draw()
        self.rgb = np.asarray(self.fig.canvas.buffer_rgba())[:, :, :3].copy()
        self.j = j
        return self.rgb


class View:
    """The field in a headless GL canvas (the Scene recipe of x8_flight.py): the nine
    microphones, the M350 posed from the log, the truth track, the estimate, its trail and its
    stalk, the line from the estimate to the truth. The camera is fixed, from the south-west and
    above, fitted to the flight and the array."""

    def __init__(self, tp, res):
        from m350_rig import Visual, model_path, ned_to_world
        self.tp, self.res, self.w2 = tp, res, ned_to_world
        log, mics = res["log"], res["mics"]
        self.canvas = tp.Canvas("threepp - M350 acoustic", width=VIEW_W, height=FILM_H, antialiasing=4, headless=True,
                                vsync=False)
        r = self.r = tp.GLRenderer(self.canvas)
        r.tone_mapping = tp.ToneMapping.ACESFilmic
        r.tone_mapping_exposure = 1.0
        s = self.scene = tp.Scene()
        s.background = 0x9eb8d6
        s.set_fog(tp.Color(0x9eb8d6), 400.0, 2500.0)
        s.add(tp.HemisphereLight(0xdce6f2, 0x4a5a38, 1.1))
        sun = tp.DirectionalLight(0xfff4e0, 2.2)
        sun.position.set(-300.0, 500.0, 200.0)
        s.add(sun)
        ground = tp.Mesh(tp.PlaneGeometry(5000.0, 5000.0), self.mat(0x646c50, 0.95))
        ground.rotation.x = -math.pi / 2
        s.add(ground)
        reach = 20.0 * math.ceil(max(np.abs(log["pos"][:, :2]).max(), np.abs(mics[:, :2]).max()) / 20.0 + 0.5)
        grid = tp.GridHelper(int(2 * reach), int(2 * reach / 10.0), tp.Color(0x7c836a), tp.Color(0x7c836a))
        grid.position.y = 0.03                                              # a 10 m grid about the array
        s.add(grid)
        pole, knob = self.mat(0xf2f2f2, 0.6), self.mat(0xd8241c, 0.5, emissive=(0.6, 0.05, 0.03))
        for m in mics:
            h = -float(m[2])
            p = tp.Mesh(tp.CylinderGeometry(0.12, 0.12, h, 8, 1), pole)
            p.position.set(*ned_to_world(m[0], m[1], -0.5 * h))
            s.add(p)
            b = tp.Mesh(tp.SphereGeometry(0.45, 16, 10), knob)
            b.position.set(*ned_to_world(m[0], m[1], -h))
            s.add(b)
        tr = log["hubs"].mean(axis=1)
        self.track = np.array([ned_to_world(*p) for p in tr[::10]], np.float32)
        line, _ = self.polyline(self.track, 0xffffff, opacity=0.45)
        s.add(line)
        hot = self.mat(0x2fffb0, 0.4, emissive=(0.2, 1.0, 0.65))
        self.est = tp.Mesh(tp.SphereGeometry(1.0, 20, 12), hot)
        s.add(self.est)
        self.trail = [tp.Mesh(tp.SphereGeometry(1.0, 12, 8), hot) for _ in range(TRAIL)]
        for b in self.trail:
            s.add(b)
        self.stalk, self.stalk_g = self.polyline([(0, 0, 0), (0, 0, 0)], 0x2fffb0)
        s.add(self.stalk)
        self.miss, self.miss_g = self.polyline([(0, 0, 0), (0, 0, 0)], 0xff3b30)
        s.add(self.miss)
        self.model = tp.GLTFLoader().load(model_path()).scene
        s.add(self.model)
        self.vis = Visual(self.model)
        self.m = M350()
        t, w = log["t"], log["w"]
        self.angle = np.concatenate([np.zeros((1, 4)), np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t)[:, None], axis=0)])
        self.cam = tp.PerspectiveCamera(54.0, VIEW_W / FILM_H, 0.5, 6000.0)
        self.close = tp.PerspectiveCamera(30.0, VIEW_W / FILM_H, 0.05, 6000.0)
        ground_track = self.track.copy()
        ground_track[:, 1] = 0.0
        self.fit(np.vstack([self.track, ground_track, [ned_to_world(*m) for m in mics]]))

    def mat(self, color, roughness, emissive=None):
        m = self.tp.MeshStandardMaterial()
        m.color, m.roughness, m.metalness = color, roughness, 0.0
        if emissive is not None:
            m.emissive = self.tp.Color(*emissive)
            m.emissive_intensity = 1.0
        return m

    def polyline(self, pts, color, opacity=1.0):
        tp = self.tp
        g = tp.BufferGeometry()
        g.set_attribute("position", np.asarray(pts, np.float32))
        m = tp.LineBasicMaterial()
        m.color = color
        if opacity < 1.0:
            m.transparent, m.opacity = True, opacity
        line = tp.Line(g, m)
        line.frustum_culled = False
        return line, g

    def fit(self, pts, elevation=32.0, horizon=0.8, margin=0.92):
        """The eye on the south-west diagonal of the box's centre, `elevation` degrees up from
        it; the view looks north-east, pitched down just enough to put the horizon at `horizon`
        of the frame's half-height above the middle; the distance is the shortest that keeps
        every point inside `margin` of the frame and off the close-up and its caption."""
        c = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
        el = math.radians(elevation)
        d = np.array([-math.cos(el) / math.sqrt(2.0), math.sin(el), math.cos(el) / math.sqrt(2.0)])   # west, up, south
        tv = math.tan(math.radians(0.5 * self.cam.fov))
        th = tv * VIEW_W / FILM_H
        pitch = math.atan(horizon * tv)
        flat = np.array([1.0, 0.0, -1.0]) / math.sqrt(2.0)                  # north-east, level
        f = math.cos(pitch) * flat - math.sin(pitch) * np.array([0.0, 1.0, 0.0])
        right = np.cross(f, [0.0, 1.0, 0.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        x1 = 2.0 * (INSET_AT[0] + INSET[0] + 20) / VIEW_W - 1.0             # the close-up's corner in NDC
        y1 = 1.0 - 2.0 * (INSET_AT[1] + INSET[1] + 50) / FILM_H

        def inside(dist):
            v = pts - (c + dist * d)
            z = v @ f
            x, y = v @ right / (z * th), v @ up / (z * tv)
            return bool(np.all(z > 0) and np.all(np.abs(x) <= margin) and np.all(np.abs(y) <= margin)
                        and not np.any((x < x1) & (y > y1)))

        lo, hi = 1.0, 5000.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (lo, mid) if inside(mid) else (mid, hi)
        self.eye, self.dir = c + hi * d, d
        self.cam.position.set(*map(float, self.eye))
        self.cam.look_at(self.tp.Vector3(*map(float, self.eye + 100.0 * f)))
        print(f"[view] camera {hi:.0f} m from the box's centre ({c[0]:.0f}, {c[1]:.0f}, {c[2]:.0f}) world, "
              f"{elevation:g} deg up from the south-west, pitched {math.degrees(pitch):.1f} deg down")

    def state(self, te):
        """The logged state at te, into the M350 the Visual reads (position, quaternion, rotor angles)."""
        log, m = self.res["log"], self.m
        t = log["t"]
        k = int(np.clip(np.searchsorted(t, te) - 1, 0, len(t) - 2))
        a = float(np.clip((te - t[k]) / (t[k + 1] - t[k]), 0.0, 1.0))
        q0, q1 = log["quat"][k], log["quat"][k + 1]
        q = (1.0 - a) * q0 + a * (q1 if np.dot(q0, q1) >= 0 else -q1)
        q /= np.linalg.norm(q)
        p = (1.0 - a) * log["pos"][k] + a * log["pos"][k + 1]
        m.x[0:3] = [float(v) for v in p]
        m.x[6:10] = [float(v) for v in q]
        m.rotor_angle = [float(v) % (2.0 * math.pi) for v in (1.0 - a) * self.angle[k] + a * self.angle[k + 1]]
        return p

    def update(self, t, j):
        """Film time t is the time at the microphones: the aircraft is shown where it was when the
        sound now arriving at the array's centre left it, which is what window j is scored against."""
        res, R, w2 = self.res, self.res["rows"], self.w2
        centre = res["mics"].mean(axis=0)
        te = t
        for _ in range(3):
            te = t - np.linalg.norm(self.state(te) - centre) / C
        self.p = np.array(w2(*self.state(te)))
        self.vis.pose(self.m, None, h0=0.0)
        show = j >= 0
        for o in [self.est, self.stalk, self.miss, *self.trail]:
            o.visible = show
        if not show:
            return
        e, tru = np.array(w2(*R[j, 2:5])), np.array(w2(*R[j, 5:8]))
        self.e = e
        self.est.position.set(*map(float, e))
        self.stalk_g.update_attribute("position", np.array([e, [e[0], 0.0, e[2]]], np.float32))
        self.miss_g.update_attribute("position", np.array([e, tru], np.float32))
        for i, b in enumerate(self.trail):
            k = j - 1 - i
            b.visible = k >= 0
            if k >= 0:
                b.position.set(*map(float, w2(*R[k, 2:5])))

    def sizes(self, r_est, r_trail):
        self.est.scale.set(r_est, r_est, r_est)
        for i, b in enumerate(self.trail):
            k = r_trail * (1.0 - 0.06 * i)
            b.scale.set(k, k, k)

    def render(self):
        """The fixed view, with the close-up (1:1, the same south-west side, 9 m off) top-left."""
        self.sizes(0.9, 0.45)
        self.r.render(self.scene, self.cam)
        rgb = np.ascontiguousarray(self.r.read_pixels())
        self.sizes(0.1, 0.05)
        self.close.position.set(*map(float, self.p + CLOSE * self.dir))
        self.close.look_at(self.tp.Vector3(*map(float, self.p)))
        self.r.render(self.scene, self.close)
        cu = np.asarray(self.r.read_pixels())
        w, h = INSET
        y0, x0 = (FILM_H - 2 * h) // 2, (VIEW_W - 2 * w) // 2
        cu = cu[y0:y0 + 2 * h, x0:x0 + 2 * w].reshape(h, 2, w, 2, 3).mean(axis=(1, 3)).astype(np.uint8)
        b, (x0, y0) = 3, INSET_AT
        rgb[y0:y0 + h + 2 * b, x0:x0 + w + 2 * b] = 245
        rgb[y0 + b:y0 + b + h, x0 + b:x0 + b + w] = cu
        return rgb


def label_view(rgb, lines):
    """Text onto the 3D view (PIL)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=20)
    for (x, y), text in lines:
        d.text((x, y), text, fill=(255, 255, 255), font=font, stroke_width=2, stroke_fill=(20, 24, 30))
    return np.asarray(im)


def film(a):
    """--shot t: one frame of the film as a png; --film: all of it, with the stereo track."""
    import subprocess
    import threepp as tp
    from PIL import Image
    res = result(a.flight, a.snr, a.ambient, a.ground_reflection, a.window, a.hop, a.seed)
    R, T = res["rows"], float(res["log"]["t"][-1])
    view, panel = View(tp, res), Panel(res)

    def frame(t):
        j = int(np.searchsorted(R[:, 0], t, side="right")) - 1
        if panel.rgb is None or j != panel.j:
            panel.draw(j, t)
        view.update(t, j)
        rgb = view.render()
        x0, y0 = INSET_AT
        txt = [((18, FILM_H - 40), f"t {t:.1f} s. The aircraft is drawn where it was when the sound now at the "
                                   f"microphones left it."),
               ((x0 + 6, y0 + INSET[1] + 14), f"close-up from {CLOSE:g} m: " + (
                   f"estimate (green) {R[j, 8]:.2f} m from the truth" if j >= 0 else "no estimate yet"))]
        return np.concatenate([label_view(rgb, txt), panel.rgb], axis=1)

    if a.shot is not None:
        path = a.out or versioned(os.path.join(OUT, f"shot_{a.shot:g}"), ".png")
        Image.fromarray(frame(a.shot)).save(path)
        print(f"written: {path}")
        return 0
    stem = os.path.join(OUT, f"flight{a.flight}_{tag(res)}_stereo.wav")
    left, right = stereo(res, stem)
    print(f"written: {stem} (left microphone {left}, right {right}: the ring's westmost and eastmost)")
    import imageio_ffmpeg
    dst = versioned(os.path.join(OUT, "m350_acoustic"), ".mp4")
    tmp = dst[:-4] + ".part.mp4"
    ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo",
                           "-pix_fmt", "rgb24", "-s", f"{VIEW_W + PANEL_W}x{FILM_H}", "-r", str(FPS), "-i", "-",
                           "-i", stem, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac",
                           "-b:a", "160k", "-shortest", tmp], stdin=subprocess.PIPE)
    n = int(T * FPS)
    keep = {int(round(5.0 * FPS)): 5, int(round(18.0 * FPS)): 18, int(round(32.0 * FPS)): 32, n - 1: "end"}
    t0 = time.time()
    for i in range(n):
        rgb = frame(i / FPS)
        ff.stdin.write(rgb.tobytes())
        if i in keep:
            p = versioned(os.path.join(OUT, f"frame_{keep[i]}"), ".png")
            Image.fromarray(rgb).save(p)
            print(f"written: {p}")
        if i % (10 * FPS) == 0:
            print(f"[film] {i / FPS:5.1f} of {T:.1f} s ({time.time() - t0:.0f} s)", flush=True)
    ff.stdin.close()
    ff.wait()
    os.replace(tmp, dst)
    print(f"written: {dst} ({n} frames, {n / FPS:.1f} s, {os.path.getsize(dst) / 1e6:.1f} MB)")
    return 0


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checks", action="store_true", help="the three checks")
    ap.add_argument("--flight", choices=("A", "B"), default=None)
    ap.add_argument("--snr", type=float, default=None,
                    help="dB, the microphones' noise against the four rotors at 50 m (default 20; the film's 35)")
    ap.add_argument("--ground-reflection", action="store_true")
    ap.add_argument("--window", type=float, default=LOC["window"], help="s")
    ap.add_argument("--hop", type=float, default=LOC["hop"], help="s")
    ap.add_argument("--seed", type=int, default=None, help="the acoustic seed (default: the flight's)")
    ap.add_argument("--broadband", type=float, default=BROADBAND, help="broadband over tonal RMS at hover (tuning)")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--ambient", choices=("white", "pink"), default=None,
                    help="the microphones' noise: white (the default), or pink (1/sqrt(f), the same RMS; the film's)")
    ap.add_argument("--shot", type=float, default=None, help="s: one still of the film at that time")
    ap.add_argument("--out", default=None, help="the still's png (default out/acoustic/shot_<t>_vNN.png)")
    ap.add_argument("--film", action="store_true", help="the film of the flight (default SNR 35 dB pink)")
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    pictures = a.film or a.shot is not None
    a.snr = (35.0 if pictures else 20.0) if a.snr is None else a.snr
    a.ambient = ("pink" if pictures else "white") if a.ambient is None else a.ambient
    if a.checks:
        return 0 if checks() else 1
    if a.report:
        return 0 if report() else 1
    if a.flight and pictures:
        return film(a)
    if a.flight:
        res = run(a.flight, snr=a.snr, reflection=a.ground_reflection, window=a.window, hop=a.hop, seed=a.seed,
                  broadband=a.broadband, ambient=a.ambient)
        for p in save(res):
            print(f"written: {p}")
        return 0
    parse(["--help"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
