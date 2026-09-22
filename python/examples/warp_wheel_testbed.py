"""A single rigid wheel in a bin of MLS-MPM grains, CARRIED by the grains.

Phase 1 of taking the gravel lane of warp_mudsnow_drive.py from a one-way
display to a two-way terramechanics model. No car: one cylinder wheel, a dead
load W on a vertical linear bearing, a carriage that prescribes the horizontal
speed v and the spin omega from a slip ratio s = 1 - v / (omega r).

The wheel's height is NOT prescribed. Every MPM substep the momentum the
wheel's velocity boundary condition removes from the grid is the soil's force
on it, and a one-thread kernel integrates m y'' = F_soil_y - W with it -- the
wheel sinks until the grains carry W (threepp.granular_mpm).

    python warp_wheel_testbed.py --mode static        # sinkage vs load, closure
    python warp_wheel_testbed.py --mode esweep        # elastic rebound vs E
    python warp_wheel_testbed.py --mode plate         # two-plate Bekker calibration
    python warp_wheel_testbed.py --mode slip          # DP/W, torque, sinkage vs slip
    python warp_wheel_testbed.py --mode compare       # Wong-Reece on the plate fit
    python warp_wheel_testbed.py --mode render        # Vulkan stills + mp4 (--slip 0.3)
    python warp_wheel_testbed.py --mode bench         # ms per simulated second, N reps
    python warp_wheel_testbed.py --scale lab ...      # r=0.15 m lab wheel instead
    --soil loose|dense   loose (default) = DP + compaction cap; dense = the demo's
                         DP-only law. Output goes to <out>/<soil>.
    --out DIR            output root (default: _granular_wheel next to this script)
    --h, --E, --mu-wheel, --tag, --slips, --L        resolution / material / rim /
                         file tag / slip list / slip-bin length

Results, caveats and the phase-2 plan live with the output (RESULTS.md in the
output root); the car-side coupling is warp_mudsnow_drive.py's gravel lane.

Conventions: x is the travel direction, y up, z the wheel axis. Drawbar pull
DP = the soil's horizontal force on the wheel along +x (positive = the soil
pushes the wheel forward, i.e. net traction). Drive torque T = the soil's
moment on the wheel about its axle, sign chosen so a resisting (driven-wheel)
torque is positive: the motor must supply T. Sinkage z = the undisturbed
settled surface minus the rim bottom (r + collider skin).
"""
import csv
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from threepp import granular_mpm as gm

G = 9.81
FRAME = 1.0 / 60.0


def cli_arg(flag, default, cast):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


SCALE = cli_arg("--scale", "car", str)
if SCALE == "car":
    CFG = dict(r=0.36, b=0.24, h=0.025, L=1.8, L_slip=3.2, Wd=0.5, D=0.30,
               loads=[500.0, 1500.0, 3000.0, 4500.0], W_slip=4500.0, v=0.25,
               plates=[0.10, 0.20], plate_L=0.40, plate_bin=(1.0, 0.8))
else:  # lab
    CFG = dict(r=0.15, b=0.10, h=0.0125, L=1.0, L_slip=1.6, Wd=0.30, D=0.18,
               loads=[40.0, 100.0, 200.0, 300.0], W_slip=200.0, v=0.12,
               plates=[0.05, 0.10], plate_L=0.20, plate_bin=(0.6, 0.45))
CFG["h"] = cli_arg("--h", CFG["h"], float)
E = cli_arg("--E", 5.0e6, float)
PHI = cli_arg("--phi", 36.0, float)
COH = cli_arg("--coh", 0.0, float)
RHO = cli_arg("--rho", 1700.0, float)
NU = 0.3
CFL = cli_arg("--cfl", 0.35, float)
TAG = cli_arg("--tag", "", str)
MU_WHEEL = cli_arg("--mu-wheel", 0.6, float)     # rim-grain Coulomb friction
NO_DETAIL = "--no-detail" in sys.argv            # render: skip the per-pixel sand grain
NO_STABLE_FIX = "--no-stable-fix" in sys.argv    # render: A/B the soup motion-vector fix
# Loose soil: a compaction cap on the Drucker-Prager cone (see granular_mpm).
# --soil dense turns it off: a cohesionless DP material that can only flow,
# never compact -- the demo's gravel.
SOIL = cli_arg("--soil", "loose", str)
OUT_ROOT = cli_arg("--out", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "_granular_wheel"), str)
CAP_P0 = cli_arg("--cap-p0", 2.0e3 if SOIL == "loose" else 0.0, float)
CAP_LAM = cli_arg("--cap-lam", 0.025 if SOIL == "loose" else 0.0, float)
OUT = os.path.join(OUT_ROOT, SOIL)
os.makedirs(OUT, exist_ok=True)


def material(E_=None):
    return gm.Material(E=E if E_ is None else E_, nu=NU, rho=RHO, phi_deg=PHI, cohesion=COH,
                       cap_p0=CAP_P0, cap_lambda=CAP_LAM)


def make_sim(mat, length, width, depth, h=None):
    h = CFG["h"] if h is None else h
    dt0 = mat.cfl_dt(h, CFL, vmax=2.0)
    nsub = int(math.ceil(FRAME / dt0))
    dt = FRAME / nsub
    sim = gm.GranularMPM(mat, h, lo=(0.0, 0.0, 0.0), hi=(length, 0.0, width),
                         fill_height=depth, dt=dt, headroom=0.30, max_bodies=1)
    return sim, nsub


def settle(sim, nsub, seconds=0.2):
    for _ in range(int(round(seconds / FRAME))):
        sim.step(nsub)
    sim.wp.synchronize_device(sim.dev)


def run_wheel(W, s=None, E_=None, length=None, unload=False, frame_cb=None,
              t_ramp=0.4, t_hold=0.5, x_end_margin=None, log_name=None):
    """Load a wheel onto the bed; optionally drive it at slip s. Returns per-frame rows."""
    r, b = CFG["r"], CFG["b"]
    mat = material(E_)
    L = CFG["L"] if length is None else length
    sim, nsub = make_sim(mat, L, CFG["Wd"], CFG["D"])
    x0 = r + 0.12 if s is not None else 0.5 * L
    zc = 0.5 * CFG["Wd"]
    # Park the wheel above the grid while the bed settles.
    wid = sim.add_body(gm.CYLINDER, (r, 0.5 * b, 0.0), (x0, CFG["D"] + r + 1.0, zc),
                       axis=(0, 0, 1), mu=MU_WHEEL, mode=gm.KINEMATIC)
    settle(sim, nsub, 0.2)
    ys = sim.surface_height(0.0, L, 0.0, CFG["Wd"])
    y0 = ys + r + sim.eps + 0.002
    m_eff = W / G
    sim.set_body(wid, pos=(x0, y0, zc), vel=(0, 0, 0), mode=gm.FREE_Y, mass=m_eff, load=0.0)
    sim.take_mean_force()
    sim.take_wall_force()
    Ws = sim.soil_weight
    rows = []
    t = 0.0
    phase = "load"
    omega = 0.0
    wall = 0.0
    frames_timed = 0
    x_end = L - r - (0.20 if x_end_margin is None else x_end_margin)
    t_unload = 0.0
    while True:
        # Host-side schedule, per frame (the graph reads these arrays).
        if phase == "load":
            load = W * min(1.0, t / t_ramp)
            sim.set_body(wid, load=load)
            if t >= t_ramp + t_hold:
                if s is not None:
                    phase = "drive"
                    omega = CFG["v"] / ((1.0 - s) * r)
                    pos, vel = sim.body_state(wid)
                    sim.set_body(wid, vel=(CFG["v"], vel[1], 0.0), omega=(0, 0, -omega))
                elif unload:
                    phase = "unload"
                    t_unload = t
                else:
                    break
        elif phase == "unload":
            frac = min(1.0, (t - t_unload) / 0.3)
            load = W * (1.0 - 0.97 * frac)
            sim.set_body(wid, load=load)
            if t - t_unload > 0.7:
                break
        t0 = time.perf_counter()
        sim.step(nsub)
        f, tq = sim.take_mean_force()        # syncs
        dtw = time.perf_counter() - t0
        fw = sim.take_wall_force()
        if phase == "drive":
            wall += dtw
            frames_timed += 1
        t += FRAME
        pos, vel = sim.body_state(wid)
        sink = ys - (pos[1] - r - sim.eps)
        rows.append(dict(t=t, phase=phase, x=pos[0], y=pos[1], vy=vel[1], sink=sink,
                         load=float(sim.bload.numpy()[wid]), Fx=f[0, 0], Fy=f[0, 1],
                         Fz=f[0, 2], Tz=tq[0, 2], Fbin_y=fw[1], Fbin_x=fw[0],
                         floor_minus_soil=-fw[1] - Ws))
        if frame_cb is not None:
            frame_cb(sim, rows[-1], nsub)
        if phase == "drive" and pos[0] >= x_end:
            break
        if not np.isfinite(pos[1]) or pos[1] < -0.5:
            print("  !! wheel fell through / diverged")
            break
    info = dict(n=sim.n, nsub=nsub, dt=sim.dt, h=sim.h, pd=sim.pd, ys=ys, eps=sim.eps,
                grid=sim.dims, c_p=mat.p_wave,
                ms_per_sim_s=(1000.0 * wall / max(frames_timed, 1)) * 60.0 if frames_timed else None)
    if log_name:
        write_csv(os.path.join(OUT, log_name), rows)
    return rows, info


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def steady(rows, phase, key, tail=0.25):
    sel = [rw for rw in rows if rw["phase"] == phase]
    k = max(1, int(round(tail / FRAME)))
    vals = np.array([rw[key] for rw in sel[-k:]])
    return float(vals.mean()), float(vals.std())


def window(rows, key, x_lo, x_hi):
    vals = np.array([rw[key] for rw in rows if rw["phase"] == "drive" and x_lo <= rw["x"] <= x_hi])
    return float(vals.mean()), float(vals.std()), len(vals)


# --- experiments --------------------------------------------------------------

def mode_static():
    out = []
    for W in CFG["loads"]:
        rows, info = run_wheel(W, length=min(CFG["L"], 1.0), log_name=f"static_{SCALE}_{int(W)}{TAG}.csv")
        fy, fys = steady(rows, "load", "Fy")
        z, zs = steady(rows, "load", "sink")
        fb, fbs = steady(rows, "load", "floor_minus_soil")
        out.append(dict(W=W, Fy=fy, Fy_std=fys, closure=fy / W, sink=z, sink_std=zs,
                        bin_closure=fb / W, bin_closure_std=fbs / W))
        print(f"  W={W:7.0f} N  Fy/W={fy / W:.3f} (+-{fys / W:.3f})  (Fbin-Wsoil)/W={fb / W:.3f} "
              f"(+-{fbs / W:.3f})  sinkage={z * 1000:.1f} mm "
              f"(+-{zs * 1000:.1f})  n={info['n']} nsub={info['nsub']} dt={info['dt']:.2e}")
    write_csv(os.path.join(OUT, f"static_{SCALE}{TAG}.csv"), out)
    return out


def mode_esweep():
    out = []
    W = CFG["loads"][-1]
    for E_ in [1.0e6, 3.0e6, 5.0e6, 1.0e7]:
        t0 = time.perf_counter()
        rows, info = run_wheel(W, E_=E_, length=1.0, unload=True, log_name=f"esweep_{SCALE}_{E_:.0e}.csv")
        wall = time.perf_counter() - t0
        z_loaded, _ = steady(rows, "load", "sink", 0.2)
        z_unl, _ = steady(rows, "unload", "sink", 0.2)
        fy, _ = steady(rows, "load", "Fy", 0.2)
        rebound = z_loaded - z_unl
        simt = rows[-1]["t"] + 0.2
        d = dict(E=E_, c_p=info["c_p"], dt=info["dt"], nsub=info["nsub"], sink_loaded=z_loaded,
                 sink_unloaded=z_unl, rebound=rebound, elastic_frac=rebound / max(z_loaded, 1e-9),
                 closure=fy / W, wall_s=wall, sim_s=simt, n=info["n"])
        out.append(d)
        print(f"  E={E_:.1e}  c_p={info['c_p']:.1f} m/s  dt={info['dt']:.2e}  nsub={info['nsub']}  "
              f"sink={z_loaded * 1000:.1f} mm  rebound={rebound * 1000:.2f} mm "
              f"({100 * rebound / max(z_loaded, 1e-9):.1f}%)  closure={fy / W:.3f}  wall {wall:.1f}s/{simt:.2f}s sim")
    write_csv(os.path.join(OUT, f"esweep_{SCALE}.csv"), out)


def run_plate(bw, rate=0.05, zmax=0.10):
    Lb, Wb = CFG["plate_bin"]
    mat = material()
    sim, nsub = make_sim(mat, Lb, Wb, CFG["D"])
    hy = 0.15          # tall enough that the heave never flows over its top face
    pid = sim.add_body(gm.BOX, (0.5 * bw, hy, 0.5 * CFG["plate_L"]),
                       (0.5 * Lb, CFG["D"] + 1.0, 0.5 * Wb), mu=0.6, mode=gm.KINEMATIC)
    settle(sim, nsub, 0.2)
    ys = sim.surface_height(0.0, Lb, 0.0, Wb)
    sim.set_body(pid, pos=(0.5 * Lb, ys + hy + sim.eps + 0.003, 0.5 * Wb), vel=(0, -rate, 0))
    sim.take_mean_force()
    rows = []
    t = 0.0
    area = bw * CFG["plate_L"]
    while True:
        sim.step(nsub)
        f, _ = sim.take_mean_force()
        t += FRAME
        pos, _ = sim.body_state(pid)
        z = ys - (pos[1] - hy - sim.eps)
        rows.append(dict(t=t, z=z, Fy=f[0, 1], p=f[0, 1] / area))
        if z >= zmax:
            break
    write_csv(os.path.join(OUT, f"plate_{SCALE}_b{int(bw * 1000)}.csv"), rows)
    return rows


def fit_bekker(curves, zlo, zhi):
    """Standard two-plate method: p = (k_c/b + k_phi) z^n, common n."""
    ns, fits = [], []
    for bw, rows in curves:
        z = np.array([rw["z"] for rw in rows])
        p = np.array([rw["p"] for rw in rows])
        m = (z >= zlo) & (z <= zhi) & (p > 0)
        A = np.vstack([np.ones(m.sum()), np.log(z[m])]).T
        c, n = np.linalg.lstsq(A, np.log(p[m]), rcond=None)[0]
        ns.append(n)
        fits.append((bw, z[m], p[m]))
    n = float(np.mean(ns))
    keq = [float(np.exp(np.mean(np.log(p) - n * np.log(z)))) for _, z, p in fits]
    (b1, _, _), (b2, _, _) = fits
    kc = (keq[0] - keq[1]) / (1.0 / b1 - 1.0 / b2)
    kphi = keq[0] - kc / b1
    # Goodness of fit on the common-n model.
    r2 = []
    for (bw, z, p), k in zip(fits, keq):
        pr = k * z ** n
        r2.append(float(1.0 - np.sum((np.log(p) - np.log(pr)) ** 2)
                        / np.sum((np.log(p) - np.log(p).mean()) ** 2)))
    return dict(n=n, n_each=[float(x) for x in ns], k_eq=keq, k_c=float(kc), k_phi=float(kphi),
                b=[b1, b2], r2=r2, z_range=[zlo, zhi])


def mode_plate():
    curves = []
    for bw in CFG["plates"]:
        rows = run_plate(bw)
        curves.append((bw, rows))
        zs = [0.02, 0.04, 0.06, 0.08]
        msg = "  ".join(f"z={z * 1000:.0f}mm p={np.interp(z, [r['z'] for r in rows], [r['p'] for r in rows]) / 1000:.1f}kPa"
                        for z in zs)
        print(f"  plate b={bw:.2f} m: {msg}")
    zlo = 2.0 * CFG["h"] * 0.4
    fit = fit_bekker(curves, zlo, 0.08 if SCALE == "car" else 0.05)
    fit.update(E=E, phi=PHI, c=COH, rho=RHO, h=CFG["h"])
    with open(os.path.join(OUT, f"bekker_fit_{SCALE}.json"), "w") as fh:
        json.dump(fit, fh, indent=2)
    print("  Bekker fit:", json.dumps(fit))


def mode_slip():
    out = []
    W = CFG["W_slip"]
    slips = [float(x) for x in cli_arg("--slips", "0,0.1,0.2,0.3,0.5,0.7", str).split(",")]
    # A long bin: the wheel starts in its own static pit and needs a few
    # diameters to forget it. Steady state = the last 0.9 m (car) of travel.
    Ls = cli_arg("--L", CFG["L_slip"], float)
    x_hi = Ls - CFG["r"] - 0.20
    x_lo = x_hi - 1.25 * 2 * CFG["r"]
    for s in slips:
        rows, info = run_wheel(W, s=s, length=Ls, log_name=f"slip_{SCALE}_s{int(round(s * 100)):02d}{TAG}.csv")
        # Drift across the window: how far from steady the averages are.
        dz = [rw["sink"] for rw in rows if rw["phase"] == "drive" and x_lo <= rw["x"] <= x_hi]
        dp, dps, nwin = window(rows, "Fx", x_lo, x_hi)
        fy, _, _ = window(rows, "Fy", x_lo, x_hi)
        fb, _, _ = window(rows, "floor_minus_soil", x_lo, x_hi)
        fbx, _, _ = window(rows, "Fbin_x", x_lo, x_hi)
        tz, tzs, _ = window(rows, "Tz", x_lo, x_hi)
        z, zs, _ = window(rows, "sink", x_lo, x_hi)
        z_static, _ = steady(rows, "load", "sink", 0.2)
        d = dict(s=s, W=W, DP=dp, DP_std=dps, DP_W=dp / W, T=tz, T_std=tzs, sink=z, sink_std=zs,
                 sink_drift=(dz[-1] - dz[0]) if dz else float("nan"),
                 sink_static=z_static, closure=fy / W, bin_closure=fb / W,
                 bin_x_over_DP=-fbx / dp if abs(dp) > 1e-9 else float("nan"), n_frames=nwin, ms_per_sim_s=info["ms_per_sim_s"],
                 n=info["n"], nsub=info["nsub"])
        out.append(d)
        print(f"  s={s:.2f}  DP/W={dp / W:+.3f} (+-{dps / W:.3f})  T={tz:7.1f} N m  "
              f"sink={z * 1000:.1f} mm (static {z_static * 1000:.1f})  Fy/W={fy / W:.3f}  "
              f"frames={nwin}  {info['ms_per_sim_s']:.0f} ms/sim-s")
    write_csv(os.path.join(OUT, f"slip_{SCALE}{TAG}.csv"), out)


# --- Wong-Reece ---------------------------------------------------------------

def wong_reece(W, s, r, b, kc, kphi, n, c, phi_deg, K, c1=0.43, c2=0.32):
    """Sinkage and DP/W, T for a rigid driven wheel (Wong & Reece 1967), theta2 = 0."""
    tphi = math.tan(math.radians(phi_deg))
    keq = kc / b + kphi

    def forces(z):
        th1 = math.acos(max(-1.0, 1.0 - z / r))
        thm = (c1 + c2 * s) * th1
        th = np.linspace(0.0, th1, 400)
        sig = np.where(th >= thm,
                       keq * r ** n * np.maximum(np.cos(th) - math.cos(th1), 0.0) ** n,
                       keq * r ** n * np.maximum(np.cos(th1 - th / max(thm, 1e-9) * (th1 - thm))
                                                 - math.cos(th1), 0.0) ** n)
        j = r * ((th1 - th) - (1.0 - s) * (math.sin(th1) - np.sin(th)))
        tau = (c + sig * tphi) * (1.0 - np.exp(-np.maximum(j, 0.0) / K))
        Wz = r * b * np.trapezoid(sig * np.cos(th) + tau * np.sin(th), th)
        DP = r * b * np.trapezoid(tau * np.cos(th) - sig * np.sin(th), th)
        T = r * r * b * np.trapezoid(tau, th)
        return Wz, DP, T

    lo, hi = 1e-5, r
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if forces(mid)[0] < W:
            lo = mid
        else:
            hi = mid
    z = 0.5 * (lo + hi)
    _, DP, T = forces(z)
    return z, DP / W, T


def bekker_static(W, r, b, kc, kphi, n):
    """Static sinkage of a rigid wheel pressed straight down: Bekker pressure
    on BOTH arcs of the contact (a rolling wheel only has the front one)."""
    keq = kc / b + kphi

    def wz(z):
        th1 = math.acos(max(-1.0, 1.0 - z / r))
        th = np.linspace(0.0, th1, 400)
        sig = keq * r ** n * np.maximum(np.cos(th) - math.cos(th1), 0.0) ** n
        return 2.0 * r * b * np.trapezoid(sig * np.cos(th), th)

    lo, hi = 1e-6, r
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if wz(mid) < W else (lo, mid)
    return 0.5 * (lo + hi)


def mode_compare():
    with open(os.path.join(OUT, f"bekker_fit_{SCALE}.json")) as fh:
        fit = json.load(fh)
    with open(os.path.join(OUT, f"slip_{SCALE}{TAG}.csv")) as fh:
        mpm = list(csv.DictReader(fh))
    Ks = [float(k) for k in cli_arg("--K", "0.01,0.02,0.04", str).split(",")]
    rows = []
    for rw in mpm:
        s = float(rw["s"])
        d = dict(s=s, mpm_sink=float(rw["sink"]), mpm_DP_W=float(rw["DP_W"]), mpm_T=float(rw["T"]))
        for K in Ks:
            z, dpw, T = wong_reece(float(rw["W"]), s, CFG["r"], CFG["b"], fit["k_c"], fit["k_phi"],
                                   fit["n"], COH, PHI, K)
            d[f"wr_sink_K{int(K * 1000)}"] = z
            d[f"wr_DP_W_K{int(K * 1000)}"] = dpw
            d[f"wr_T_K{int(K * 1000)}"] = T
        rows.append(d)
    write_csv(os.path.join(OUT, f"compare_{SCALE}{TAG}.csv"), rows)
    # Static: Bekker sinkage vs load from the same fit.
    st = []
    p = os.path.join(OUT, f"static_{SCALE}.csv")
    if os.path.exists(p):
        with open(p) as fh:
            for rw in csv.DictReader(fh):
                z, _, _ = wong_reece(float(rw["W"]), 0.0, CFG["r"], CFG["b"], fit["k_c"], fit["k_phi"],
                                     fit["n"], COH, PHI, Ks[len(Ks) // 2])
                zb = bekker_static(float(rw["W"]), CFG["r"], CFG["b"], fit["k_c"], fit["k_phi"], fit["n"])
                st.append(dict(W=float(rw["W"]), mpm_sink=float(rw["sink"]), bekker_static_sink=zb,
                               wr_rolling_s0_sink=z))
        write_csv(os.path.join(OUT, f"compare_static_{SCALE}.csv"), st)
    for d in rows + st:
        print("  " + "  ".join(f"{k}={v:.4g}" for k, v in d.items()))
    plots(fit, rows, st, Ks)


def plots(fit, rows, st, Ks):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = [d["s"] for d in rows]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(s, [d["mpm_DP_W"] for d in rows], "o-", color="k", lw=2, label="MLS-MPM (two-way)")
    for K, ls in zip(Ks, [":", "--", "-."]):
        ax[0].plot(s, [d[f"wr_DP_W_K{int(K * 1000)}"] for d in rows], ls, color="C0",
                   label=f"Wong-Reece, K={K * 100:.0f} cm")
    ax[0].axhline(0, color="0.6", lw=0.8)
    ax[0].set_xlabel("slip s")
    ax[0].set_ylabel("DP / W")
    ax[0].legend(fontsize=8)
    ax[0].set_title("Drawbar pull")
    ax[1].plot(s, [1000 * d["mpm_sink"] for d in rows], "o-", color="k", lw=2, label="MLS-MPM")
    for K, ls in zip(Ks, [":", "--", "-."]):
        ax[1].plot(s, [1000 * d[f"wr_sink_K{int(K * 1000)}"] for d in rows], ls, color="C0",
                   label=f"Wong-Reece, K={K * 100:.0f} cm")
    ax[1].set_xlabel("slip s")
    ax[1].set_ylabel("sinkage, mm")
    ax[1].legend(fontsize=8)
    ax[1].set_title("Slip-sinkage")
    ax[2].plot(s, [d["mpm_T"] for d in rows], "o-", color="k", lw=2, label="MLS-MPM")
    for K, ls in zip(Ks, [":", "--", "-."]):
        ax[2].plot(s, [d[f"wr_T_K{int(K * 1000)}"] for d in rows], ls, color="C0",
                   label=f"Wong-Reece, K={K * 100:.0f} cm")
    ax[2].set_xlabel("slip s")
    ax[2].set_ylabel("drive torque, N m")
    ax[2].legend(fontsize=8)
    ax[2].set_title("Torque")
    fig.suptitle(f"{SOIL} soil, {SCALE} wheel r={CFG['r']} m b={CFG['b']} m, W={CFG['W_slip']:.0f} N, "
                 f"rim mu={MU_WHEEL}; "
                 f"Bekker from MPM plates: k_c={fit['k_c']:.3g}, k_phi={fit['k_phi']:.3g}, n={fit['n']:.2f}")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"slip_{SCALE}{TAG}.png"), dpi=110)
    if st:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax.plot([d["W"] / 1000 for d in st], [1000 * d["mpm_sink"] for d in st], "o-", color="k",
                lw=2, label="MLS-MPM (load-driven wheel)")
        ax.plot([d["W"] / 1000 for d in st], [1000 * d["bekker_static_sink"] for d in st], "--",
                color="C0", label="Bekker, same plates (static, both arcs)")
        ax.plot([d["W"] / 1000 for d in st], [1000 * d["wr_rolling_s0_sink"] for d in st], ":",
                color="C0", label="Wong-Reece rolling, s=0 (front arc)")
        ax.set_xlabel("wheel load W, kN")
        ax.set_ylabel("static sinkage, mm")
        ax.legend()
        ax.set_title("Sinkage vs load")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"sinkage_vs_load_{SCALE}.png"), dpi=110)
    # Plates.
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for bw in fit["b"]:
        p = os.path.join(OUT, f"plate_{SCALE}_b{int(bw * 1000)}.csv")
        with open(p) as fh:
            pr = list(csv.DictReader(fh))
        z = np.array([float(x["z"]) for x in pr])
        pp = np.array([float(x["p"]) for x in pr])
        ax.plot(1000 * z, pp / 1000, label=f"MPM plate b={bw * 100:.0f} cm")
        zz = np.linspace(0.002, z.max(), 100)
        ax.plot(1000 * zz, (fit["k_c"] / bw + fit["k_phi"]) * zz ** fit["n"] / 1000, "--", color="0.4")
    ax.set_xlabel("sinkage, mm")
    ax.set_ylabel("pressure, kPa")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.set_title("Plate sinkage (dashed: Bekker fit)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"plates_{SCALE}.png"), dpi=110)


# --- render -------------------------------------------------------------------

def build_wheel(tp, r, b):
    """A tyre with lugs and a hub face with spokes, axis along z, so spin reads."""
    g = tp.Group()
    rub = tp.MeshStandardMaterial()
    rub.color = 0x1b1b1d
    rub.roughness = 0.88
    tyre = tp.Mesh(tp.CylinderGeometry(r - 0.014, r - 0.014, b, 72), rub)
    tyre.rotation.x = math.pi / 2
    g.add(tyre)
    for k in range(26):
        a = 2.0 * math.pi * k / 26
        lug = tp.Mesh(tp.BoxGeometry(0.016, 0.034, b * 0.96), rub)
        lug.position.set((r - 0.008) * math.cos(a), (r - 0.008) * math.sin(a), 0.0)
        lug.rotation.z = a
        g.add(lug)
    steel = tp.MeshStandardMaterial()
    steel.color = 0xb9bdc3
    steel.metalness = 0.75
    steel.roughness = 0.35
    hub = tp.Mesh(tp.CylinderGeometry(0.21, 0.21, 0.012, 48), steel)
    hub.rotation.x = math.pi / 2
    hub.position.z = 0.5 * b + 0.004
    g.add(hub)
    dark = tp.MeshStandardMaterial()
    dark.color = 0x2a2d31
    dark.roughness = 0.6
    for k in range(5):
        a = 2.0 * math.pi * k / 5
        sp = tp.Mesh(tp.BoxGeometry(0.17, 0.035, 0.006), dark)
        sp.position.set(0.105 * math.cos(a), 0.105 * math.sin(a), 0.5 * b + 0.012)
        sp.rotation.z = a
        g.add(sp)
    mark = tp.MeshStandardMaterial()
    mark.color = 0xe8b418
    mark.roughness = 0.5
    m = tp.Mesh(tp.BoxGeometry(0.05, 0.05, 0.008), mark)
    m.position.set(0.0, r - 0.05, 0.5 * b + 0.001)
    g.add(m)
    return g


def mode_render():
    import threepp as tp
    import imageio_ffmpeg
    from PIL import Image, ImageDraw
    from warp_common import DensitySurface, encode_png_sequence, load_font

    s = cli_arg("--slip", 0.3, float)
    W = CFG["W_slip"]
    r, b = CFG["r"], CFG["b"]
    rdir = os.path.join(OUT_ROOT, "render")
    fdir = os.path.join(rdir, f"frames_s{int(round(s * 100)):02d}")
    os.makedirs(fdir, exist_ok=True)
    for f in os.listdir(fdir):
        os.remove(os.path.join(fdir, f))
    width, height = cli_arg("--width", 1280, int), cli_arg("--height", 720, int)
    canvas = tp.Canvas("wheel testbed", width=width, height=height, vsync=False, headless=True)
    renderer = tp.VulkanRenderer(canvas)
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    renderer.tone_mapping_exposure = cli_arg("--exposure", 0.9, float)
    scene = tp.Scene()
    scene.background = 0x8d9aa6
    scene.add(tp.HemisphereLight(0xd6e0ea, 0x4a4238, 0.9))
    sun = tp.DirectionalLight(0xfff1dc, 3.4)
    sun.position.set(-1.6, 2.4, 2.2)
    sun.cast_shadow = True
    sun.set_shadow_frustum(-3.0, 3.0, 3.0, -3.0)
    sun.set_shadow_bias(-0.0005)
    scene.add(sun)

    L, Wd, D = CFG["L"], CFG["Wd"], CFG["D"]
    concrete = tp.MeshStandardMaterial()
    concrete.color = 0x55585c
    concrete.roughness = 0.9
    floor = tp.Mesh(tp.PlaneGeometry(12.0, 12.0), concrete)
    floor.rotation.x = -math.pi / 2
    floor.position.set(0.5 * L, -0.001, 0.0)
    floor.receive_shadow = True
    scene.add(floor)
    wallm = tp.MeshStandardMaterial()
    wallm.color = 0x70767e
    wallm.metalness = 0.4
    wallm.roughness = 0.55
    for (sx, sy, sz, px, py, pz) in [(L + 0.04, D + 0.12, 0.02, 0.5 * L, 0.5 * (D + 0.12), -0.01),
                                     (0.02, D + 0.12, Wd, -0.01, 0.5 * (D + 0.12), 0.5 * Wd),
                                     (0.02, D + 0.12, Wd, L + 0.01, 0.5 * (D + 0.12), 0.5 * Wd)]:
        w = tp.Mesh(tp.BoxGeometry(sx, sy, sz), wallm)
        w.position.set(px, py, pz)
        w.cast_shadow = True
        w.receive_shadow = True
        scene.add(w)
    # The carriage: a rail over the bin and a fork down to the axle, far side.
    rail = tp.Mesh(tp.BoxGeometry(L + 0.3, 0.06, 0.06), wallm)
    rail.position.set(0.5 * L, D + 2 * r + 0.25, 0.5 * Wd - 0.5 * b - 0.06)
    scene.add(rail)
    strut = tp.Mesh(tp.BoxGeometry(0.05, 1.0, 0.04), wallm)
    scene.add(strut)
    axle = tp.Mesh(tp.CylinderGeometry(0.025, 0.025, 0.5 * b + 0.08, 16), wallm)
    axle.rotation.x = math.pi / 2
    scene.add(axle)
    wheel = build_wheel(tp, r, b)
    wheel.traverse(lambda o: setattr(o, "cast_shadow", True))
    scene.add(wheel)

    # Soil: the drive demo's DensitySurface route, host staging, grain normals.
    pd = 0.5 * CFG["h"]
    cell = 1.15 * pd
    origin = (-4 * cell, -2 * cell, -4 * cell)
    dims = (int((L + 4 * cell - origin[0]) / cell) + 1, int((D + 0.25 - origin[1]) / cell) + 1,
            int((Wd + 4 * cell - origin[2]) / cell) + 1)
    iso = 0.46 * (cell / pd) ** 3
    cap = 400_000
    geo = tp.BufferGeometry()
    geo.set_attribute("position", np.zeros((cap * 3, 3), np.float32))
    geo.set_attribute("normal", np.tile(np.float32([0, 1, 0]), (cap * 3, 1)))
    geo.set_draw_range(0, 3)
    sand = tp.MeshPhysicalMaterial()
    sand.color = 0x9c8a6c
    sand.roughness = 0.95
    sand.specular_intensity = 0.2
    # Per-PIXEL grain on the Vulkan detail layer: the marching-cubes vertex
    # normals can only carry relief coarser than ~2.5 cells (finer aliases into
    # diagonal streaks), so the sand's own grain lives here. ~7 mm stones.
    if not NO_DETAIL and hasattr(sand, "detail_normal_map"):
        from warp_common import grain_detail_texture
        sand.detail_normal_map = tp.data_texture(grain_detail_texture(256, 144), srgb=False)
        sand.detail_repeat = 12.0
        sand.detail_normal_scale = 1.0
        sand.detail_rough_strength = 0.6
    sand.side = tp.Side.Double
    soil = tp.Mesh(geo, sand)
    soil.cast_shadow = True
    soil.receive_shadow = True
    soil.frustum_culled = False
    scene.add(soil)
    # The soil is a marching-cubes SOUP re-uploaded every frame: once the wheel
    # changes the triangle count, every later vertex slot holds a different
    # surface point, so per-vertex motion vectors are noise and the temporal
    # passes boil everywhere downstream of the disturbance (the whole bin ahead
    # of the wheel, measured). Declare it: the soil reprojects as world-static.
    if not NO_STABLE_FIX and hasattr(renderer, "set_stable_correspondence"):
        renderer.set_stable_correspondence(soil, False)

    cam = tp.PerspectiveCamera(36.0, width / height, 0.05, 60.0)
    # Side-on from the near side, raised ~28 degrees: low enough that the bow
    # wave and the berms stand up in silhouette, high enough to see INTO the rut
    # over the near lip of the bin.
    cam.position.set(0.36 * L, 1.42, 2.45)
    cam.look_at(tp.Vector3(0.55 * L, 0.40, 0.25))
    cam2 = tp.PerspectiveCamera(34.0, width / height, 0.05, 60.0)

    state = dict(k=0, ang=0.0, surf=None, stage=None, rows=[])
    REPS = 4
    omega = CFG["v"] / ((1.0 - s) * r)
    stills = {}

    def cb(sim, row, nsub):
        wpm = sim.wp
        if state["surf"] is None:
            state["surf"] = DensitySurface(origin, cell, dims, sim.dev)
            state["stage"] = (wpm.zeros(cap * 3, dtype=wpm.vec3, device=sim.dev),
                              wpm.zeros(cap * 3, dtype=wpm.vec3, device=sim.dev))
        surf, (sp, sn) = state["surf"], state["stage"]
        nt = min(surf.build(sim.x, sim.n, iso), cap)
        if nt > 0:
            surf.expand(nt, sp, sn, sign=1.0, flip_winding=True, grain=0.55, grain_freq=70.0)
            geo.update_attribute("position", sp[:3 * nt].numpy())
            geo.update_attribute("normal", sn[:3 * nt].numpy())
        geo.set_draw_range(0, 3 * nt)
        if row["phase"] == "drive":
            state["ang"] -= omega * FRAME
        x, y = row["x"], row["y"]
        wheel.position.set(x, y, 0.5 * Wd)
        wheel.rotation.z = state["ang"]
        ztop = rail.position.y
        strut.position.set(x, 0.5 * (ztop + y), rail.position.z)
        strut.scale.y = max(ztop - y, 0.05)
        axle.position.set(x, y, 0.5 * Wd - 0.25 * b - 0.04)
        path = os.path.join(fdir, f"f{state['k']:05d}.png")
        # save_frame hands back a frame from 2 or 3 calls earlier (measured by
        # looking, and not constant). The scene is not touched between these
        # calls, so the LAST write is this state whatever the latency.
        for _ in range(REPS):
            renderer.save_frame(scene, cam, path)
        state["rows"].append(dict(row))
        state["k"] += 1
        if state["k"] % 30 == 0:
            print(f"  frame {state['k']}  {row['phase']}  x={x:.2f}  sink={row['sink'] * 1000:.1f} mm  "
                  f"tris={nt}")

    rows, info = run_wheel(W, s=s, frame_cb=cb, length=CFG["L"], log_name=f"render_run_s{int(round(s * 100)):02d}.csv")
    # Stills of the final state, each view rendered REPS + 2 times, last kept.
    x = rows[-1]["x"]
    tag = f"s{int(round(s * 100)):02d}"
    for name, eye, tgt in [("side_low", (0.5 * L, 0.95, 3.45), (0.5 * L, 0.42, 0.25)),
                           ("rut", (x - 1.35, 0.95, 1.05), (x - 0.30, 0.27, 0.25)),
                           ("bow", (x + 0.95, 0.50, 1.10), (x + 0.12, 0.30, 0.25)),
                           ("top", (x - 0.45, 1.9, 0.9), (x - 0.45, 0.28, 0.25))]:
        cam2.position.set(*eye)
        cam2.look_at(tp.Vector3(*tgt))
        p2 = os.path.join(rdir, f"{name}_{tag}.png")
        for _ in range(REPS + 2):
            renderer.save_frame(scene, cam2, p2)
        stills[name] = p2
    # Annotate every frame with the measured numbers (a running 0.25 s mean).
    font = load_font(22)
    small = load_font(17)
    rr = state["rows"]
    for k, row in enumerate(rr):
        p = os.path.join(fdir, f"f{k:05d}.png")
        im = Image.open(p).convert("RGB")
        d = ImageDraw.Draw(im)
        lo = max(0, k - 15)
        win = rr[lo:k + 1]
        fx = np.mean([w["Fx"] for w in win])
        fy = np.mean([w["Fy"] for w in win])
        lines = [f"MLS-MPM soil bin, one rigid wheel r={r:.2f} m b={b:.2f} m, load W={W / 1000:.1f} kN "
                 f"(dead weight, wheel height free)",
                 f"t={row['t']:.2f} s  phase={row['phase']}  slip s={s if row['phase'] == 'drive' else 0:.2f}  "
                 f"sinkage={row['sink'] * 1000:5.1f} mm  soil Fy/W={fy / W:5.3f}  DP/W={fx / W:+.3f}"]
        d.rectangle([0, 0, width, 62], fill=(20, 22, 26))
        d.text((14, 6), lines[0], fill=(235, 235, 235), font=small)
        d.text((14, 30), lines[1], fill=(250, 214, 110), font=font)
        d.text((14, height - 26), "2x slow motion (60 sim frames/s shown at 30 fps)",
               fill=(230, 230, 230), font=small)
        im.save(p)
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = os.path.join(rdir, f"wheel_s{int(round(s * 100)):02d}.mp4")
    encode_png_sequence(os.path.join(fdir, "f%05d.png"), mp4, 30, crf=18, ffmpeg=ff,
                        vf="tpad=stop_mode=clone:stop_duration=1.2")     # hold the end
    n = len(rr)
    picks = [int(round(i * (n - 1) / 7)) for i in range(8)]
    ims = [Image.open(os.path.join(fdir, f"f{k:05d}.png")).convert("RGB") for k in picks]
    tw, th = width // 2, height // 2
    sheet = Image.new("RGB", (tw * 2, th * 4), (0, 0, 0))
    for i, im in enumerate(ims):
        sheet.paste(im.resize((tw, th), Image.LANCZOS), ((i % 2) * tw, (i // 2) * th))
    sheet_p = os.path.join(rdir, f"contact_s{int(round(s * 100)):02d}.png")
    sheet.save(sheet_p)
    last = os.path.join(rdir, f"still_end_s{int(round(s * 100)):02d}.png")
    Image.open(os.path.join(fdir, f"f{n - 1:05d}.png")).save(last)
    mid = os.path.join(rdir, f"still_mid_s{int(round(s * 100)):02d}.png")
    Image.open(os.path.join(fdir, f"f{n // 2:05d}.png")).save(mid)
    print("render:", mp4, sheet_p, last, mid, stills)


def mode_bench():
    """ms per simulated second: graph vs eager, interleaved, N repeats, a rolling wheel in."""
    reps = cli_arg("--reps", 4, int)
    frames = 60
    out = []
    for E_ in [3.0e6, 5.0e6, 1.0e7]:
        mat = material(E_)
        sim, nsub = make_sim(mat, CFG["L"], CFG["Wd"], CFG["D"])
        r = CFG["r"]
        sim.add_body(gm.CYLINDER, (r, 0.5 * CFG["b"], 0.0), (0.6, CFG["D"] + r - 0.03, 0.5 * CFG["Wd"]),
                     vel=(0.25, 0, 0), omega=(0, 0, -0.25 / (0.7 * r)), mode=gm.KINEMATIC)
        settle(sim, nsub, 0.1)
        for rep in range(reps):
            for graph in (True, False):
                sim.set_body(0, pos=(0.6, CFG["D"] + r - 0.03, 0.5 * CFG["Wd"]))
                sim.step(nsub, graph=graph)
                sim.wp.synchronize_device(sim.dev)
                t0 = time.perf_counter()
                for _ in range(frames):
                    sim.step(nsub, graph=graph)
                    sim.take_mean_force()
                sim.wp.synchronize_device(sim.dev)
                ms = 1000.0 * (time.perf_counter() - t0) / frames * 60.0
                out.append(dict(E=E_, n=sim.n, nsub=nsub, graph=int(graph), rep=rep,
                                ms_per_sim_s=ms, ms_per_frame=ms / 60.0,
                                us_per_substep=1000.0 * ms / 60.0 / nsub))
                print(f"  E={E_:.0e} n={sim.n} nsub={nsub} graph={int(graph)} rep={rep}: "
                      f"{ms:.0f} ms/sim-s  ({ms / 60:.1f} ms/frame, {1000 * ms / 60 / nsub:.0f} us/substep)")
    write_csv(os.path.join(OUT, f"bench_{SCALE}.csv"), out)


if __name__ == "__main__":
    mode = cli_arg("--mode", "static", str)
    {"static": mode_static, "esweep": mode_esweep, "plate": mode_plate, "slip": mode_slip,
     "compare": mode_compare, "render": mode_render, "bench": mode_bench}.get(mode, lambda: None)()
