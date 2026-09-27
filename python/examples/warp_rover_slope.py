"""A VIPER-class rover wheel in GPU grains: real-time, plausible settings.

Phase A of the rover sand-slope demo (plans/rover-sand-slope-demo.md, the
REFRAME note): NASA's data is a plausibility band, not a gate. One wheel in
"VV mode" (carriage speed v fixed, wheel speed sets the slip, height free
under a dead load N) gives DP/N, the traction angle and the sinkage against
slip; the same wheel in GranularPatches gives the real 4-wheel cost.

Modes
  --mode wheel    the slip sweep (--slips, --soil, --lug, --h, --E, --load)
  --mode esweep   static load/unload at a few E: the elastic share of the sinkage
  --mode rt       4 wheels in GranularPatches (moving windows): ms per 1/60 s frame
  --mode plot     fig_traction.png (DP/N, angle, sinkage vs slip) and
                  fig_slope.png (the implied slip-vs-slope curve vs NASA MGRU3)
  --mode plate    two virtual plates on the same grains: Bekker k_c, k_phi, n

The wheel (MGRU3): r = 0.25 m to the rim, b = 0.20 m, 24 straight transverse
grousers 25 mm tall and 10 mm thick. A 25 mm lug cannot be resolved on a
30-40 mm grid, so the default (--lug coulomb) is the classic EFFECTIVE-RADIUS
model: a Coulomb cylinder at r_eff = r + h_g whose friction is the soil's own,
mu = tan(phi) -- the grousers trap soil, so the shear happens soil-on-soil at
the tips. Slip is measured at r_eff: s_eff = 1 - v / (omega r_eff).
Other variants, off by default (granular_mpm.LUGGED / ENVELOPE):
  --lug inflated  real lugs, inflated to one cell thick (rule in wheel_body());
  --lug envelope  a NO-SLIP cylinder at r + h_g (phase 1: too much traction);
  --lug smooth    the bare rim, for reference.

The envelope's friction is slip-mobilised (--jh-k 3, granular_mpm.wheel_mu: a
Janosi-Hanamoto law on the local slip ratio), so traction builds with slip
instead of jumping to the Coulomb ceiling (--jh-k 0) at the first sliding node.

Soil (--soil): a loose, cohesionless GRC-1-like sand (NASA's tilt bed: 15-20 %
relative density), rho = 1650 kg/m^3, phi = 38 deg, Drucker-Prager with a
BOUNDED compaction cap (SOILS below). Defaults = the phase-A choice: h = 30 mm,
8 particles per cell, E = 1 MPa (elastic share of the static sinkage 5 % at
h = 30 mm; 0.5 MPa is 13 % and 26 % fewer substeps, but puts DP/N(0) at +0.07
and the 50 %-slip slope at 22.0 deg: D:/dev/rover_rt/E05).

Output goes under --out (default D:/dev/rover_rt).

    PYTHONPATH=python python python/examples/warp_rover_slope.py --mode wheel
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np

from threepp import granular_mpm as gm

G = 9.81
FRAME = 1.0 / 60.0


def cli_arg(flag, default, cast):
    if flag in sys.argv:
        return cast(sys.argv[sys.argv.index(flag) + 1])
    return default


# --- the experiment -----------------------------------------------------------

# MGRU3 wheel (Hu et al. 2024, arXiv:2405.11001; python/examples/rover/rover_spec.json).
WHEEL = dict(r=0.25, b=0.20, hg=0.025, t=0.010, n=24)
R_EFF = WHEEL["r"] + WHEEL["hg"]        # grouser tips: the effective radius
N_LOAD = cli_arg("--load", 216.0, float)  # N per wheel: 88 kg on 4 wheels (TREC rig: 171.7)
V = 0.2                 # m/s, the carriage speed (VV mode)
SLIPS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]    # s_eff, at r_eff

# Loose GRC-1-like sand. Three laws, compared in phase A (h = 40 mm, phi = 35,
# plain Coulomb envelope; D:/dev/rover_rt/scan):
#   dp    pure Drucker-Prager, no cap: the tips never got below the surface
#         (sinkage -7..-4 mm), no rut;
#   gcap  a gentler, unbounded cap (lam 2x phase 1's, p0 1 kPa): DP/N +0.17 at
#         zero slip, flat 0.45 from s = 0.3 on, 5-8 mm sinkage;
#   bcap  a BOUNDED compaction cap, p_c = p0 exp(-e_vp / lam) <= p_max: p_max
#         sets the sinkage (a bearing pressure). CHOSEN, tuned to p_max = 5 kPa
#         and phi = 38 (GRC-1's upper end) with jh_k = 3 at h = 30 mm.
SOILS = {
    "dp": dict(rho=1650.0, phi=35.0, cap_p0=0.0, cap_lam=0.0, cap_pmax=0.0),
    "bcap": dict(rho=1650.0, phi=38.0, cap_p0=1.0e3, cap_lam=0.025, cap_pmax=5.0e3),
    "gcap": dict(rho=1650.0, phi=35.0, cap_p0=1.0e3, cap_lam=0.05, cap_pmax=0.0),
}

# NASA GRC TREC data as digitised in Hu et al. 2024, arXiv:2405.11001 (their
# Fig. 4): (slip %, traction angle atan(DBP/N) deg), GRC-3, N = 171.7 N, slip
# at the RIM (r = 0.25 m). A plausibility band here, not a gate.
NASA_TREC = np.array([
    (-0.206, -1.659), (4.919, 8.100), (9.799, 13.818), (9.884, 16.933),
    (20.028, 20.121), (30.008, 19.858), (30.012, 22.299), (39.991, 21.531),
    (40.156, 24.898), (40.073, 23.130), (50.135, 24.382), (60.117, 25.718),
    (60.111, 21.762), (60.037, 25.634), (70.096, 25.034), (70.104, 29.748),
    (80.166, 31.337), (80.090, 33.610)])
# NASA MGRU3 tilt bed, 2022 campaign (plan doc): (slope deg, slip %), slip at the rim.
NASA_SLOPE = np.array([(0.3, 2.5), (5.6, 5.1), (10.0, 10.4), (15.3, 35.5), (20.0, 73.8),
                       (25.2, 83.8), (30.1, 89.7)])

H = cli_arg("--h", 0.03, float)          # 4 wheels: 5.3 ms/frame (40 mm: 2.5, 25 mm: 8.9)
E = cli_arg("--E", 1.0e6, float)
NU = 0.3
CFL = cli_arg("--cfl", 0.5, float)        # GranularPatches' default, so both agree
PPC = cli_arg("--ppc", 0.5, float)        # particle spacing / h (0.5 = 8 per cell)
SOIL = cli_arg("--soil", "bcap", str)
SP = dict(SOILS[SOIL])
for _k in ("rho", "phi", "cap_p0", "cap_lam", "cap_pmax"):
    SP[_k] = cli_arg("--" + _k.replace("_", "-"), SP[_k], float)
RHO, PHI = SP["rho"], SP["phi"]
LUG = cli_arg("--lug", "coulomb", str)
# Wheel-soil friction: soil-on-soil (tan phi) for the effective-radius model.
MU_RIM = cli_arg("--mu", math.tan(math.radians(PHI)), float)
CONTACT = cli_arg("--contact", "none", str)
# Slip-mobilised friction on the envelope (granular_mpm wheel_mu, Janosi-Hanamoto
# on the local slip ratio): mu_eff = mu (1 - exp(-k s_loc)); 0 = plain Coulomb.
JH_K = cli_arg("--jh-k", 3.0, float)
TAG = cli_arg("--tag", "", str)
OUT = cli_arg("--out", "D:/dev/rover_rt", str)
os.makedirs(OUT, exist_ok=True)

# The bin: 3x the wheel width, 0.2 m deep (the 4-wheel patches use the same depth).
WD = cli_arg("--width", 0.60, float)
DEPTH = cli_arg("--depth", 0.20, float)
X0 = R_EFF + 0.10                             # hub x of the static pit
TRAVEL = cli_arg("--travel", 0.70, float)     # m driven after the static pit
SKIP = cli_arg("--skip", 0.35, float)         # m of travel before the steady window
CONTACTS = dict(none=gm.PC_NONE, project=gm.PC_PROJECT, velocity=gm.PC_VELOCITY)


def material(E_=None, rho=None, phi=None):
    return gm.Material(E=E if E_ is None else E_, nu=NU, rho=RHO if rho is None else rho,
                       phi_deg=PHI if phi is None else phi, cohesion=0.0,
                       cap_p0=SP["cap_p0"], cap_lambda=SP["cap_lam"], cap_pmax=SP["cap_pmax"])


def make_sim(mat, length, width, depth, h, contact):
    dt0 = mat.cfl_dt(h, CFL, vmax=2.0)
    nsub = int(math.ceil(FRAME / dt0))
    sim = gm.GranularMPM(mat, h, lo=(0.0, 0.0, 0.0), hi=(length, 0.0, width),
                         fill_height=depth, dt=FRAME / nsub, headroom=0.15, max_bodies=1,
                         particle_contact=CONTACTS[contact], ppc_spacing=PPC, jh_k=JH_K)
    return sim, nsub


def wheel_body(lug, h):
    """(shape, size, lug params, outer radius) for a grouser variant at cell size h.

    Inflation rule (--lug inflated): a lug's thickness becomes t' = max(t, h),
    one cell, so that at least one row of nodes lies inside it in every
    orientation. Its height h_g and the count (24) are KEPT: the area a lug
    sweeps per radian of spin, ((r + h_g)^2 - r^2) / 2, and the lug pitch are
    what set the soil a grouser drives and traps, and neither depends on the
    thickness. The cost is solid volume: the lug share of the grouser annulus
    goes from N t / (2 pi r) = 15 % to N t' / (2 pi r) (26 % at h = 16.7 mm,
    19 % at 12.5 mm).
    """
    r, b, hg = WHEEL["r"], WHEEL["b"], WHEEL["hg"]
    if lug == "inflated":
        t_eff = max(WHEEL["t"], h)
        return gm.LUGGED, (r, 0.5 * b, hg), (t_eff, float(WHEEL["n"]), 0.0), r + hg
    if lug == "envelope":
        return gm.ENVELOPE, (r, 0.5 * b, hg), (0.0, 0.0, 0.0), r + hg
    if lug == "coulomb":
        # The effective-radius model: a plain Coulomb cylinder at r + h_g.
        return gm.CYLINDER, (r + hg, 0.5 * b, 0.0), (0.0, 0.0, 0.0), r + hg
    return gm.CYLINDER, (r, 0.5 * b, 0.0), (0.0, 0.0, 0.0), r


def settle(sim, nsub, seconds):
    for _ in range(int(round(seconds / FRAME))):
        sim.step(nsub)
    sim.wp.synchronize_device(sim.dev)


def run_wheel(s=None, lug=LUG, contact=CONTACT, h=H, E_=None, rho=None, phi=None, W=N_LOAD,
              unload=False, length=None, t_ramp=0.4, t_hold=0.4, dump=None):
    """Settle the bed, load the wheel to W (FREE_Y), then drive at slip s (or
    unload for the elastic share). Returns per-frame rows and run info."""
    r = WHEEL["r"]
    mat = material(E_, rho, phi)
    L = length if length is not None else X0 + TRAVEL + r + WHEEL["hg"] + 0.15
    sim, nsub = make_sim(mat, L, WD, DEPTH, h, contact)
    x0 = X0 if s is not None else 0.5 * L
    zc = 0.5 * WD
    shape, size, lugp, r_out = wheel_body(lug, h)
    wid = sim.add_body(shape, size, (x0, DEPTH + r_out + 1.0, zc), axis=(0, 0, 1), mu=MU_RIM,
                       mode=gm.KINEMATIC, lug=lugp)
    settle(sim, nsub, 0.2)
    ys = sim.surface_height(0.0, L, 0.0, WD)
    sim.set_body(wid, pos=(x0, ys + r_out + sim.eps + 0.002, zc), vel=(0, 0, 0), mode=gm.FREE_Y,
                 mass=W / G, load=0.0)
    sim.take_mean_force()
    sim.take_wall_force()
    Ws = sim.soil_weight
    rows, t, phase, wall, nt, t_un = [], 0.0, "load", 0.0, 0, 0.0
    x_end = x0 + TRAVEL
    while True:
        if phase == "load":
            sim.set_body(wid, load=W * min(1.0, t / t_ramp))
            if t >= t_ramp + t_hold:
                if s is not None:
                    phase = "drive"
                    om = V / ((1.0 - s) * R_EFF)      # slip at the grouser tips
                    _, vel = sim.body_state(wid)
                    sim.set_body(wid, vel=(V, vel[1], 0.0), omega=(0, 0, -om))
                elif unload:
                    phase, t_un = "unload", t
                else:
                    break
        elif phase == "unload":
            sim.set_body(wid, load=W * (1.0 - 0.97 * min(1.0, (t - t_un) / 0.3)))
            if t - t_un > 0.7:
                break
        t0 = time.perf_counter()
        sim.step(nsub)
        f, tq = sim.take_mean_force()        # syncs
        dtw = time.perf_counter() - t0
        fw = sim.take_wall_force()
        if phase == "drive" and t > t_ramp + t_hold + 0.1:   # past the graph capture
            wall += dtw
            nt += 1
        t += FRAME
        pos, vel = sim.body_state(wid)
        # Sinkage of the grouser TIPS (r_eff): settled surface minus the tip
        # circle's bottom. The rim sits h_g higher (sink_rim).
        rows.append(dict(t=t, phase=phase, x=pos[0], y=pos[1], vy=vel[1], sink=ys - (pos[1] - R_EFF),
                         sink_rim=ys - (pos[1] - r),
                         Fx=f[0, 0], Fy=f[0, 1], Tz=tq[0, 2], floor_minus_soil=-fw[1] - Ws,
                         Fbin_x=fw[0]))
        if phase == "drive" and pos[0] >= x_end:
            break
        if not np.isfinite(pos[1]) or pos[1] < -0.5:
            print("  !! wheel fell through / diverged", flush=True)
            break
    info = dict(n=sim.n, nsub=nsub, dt=sim.dt, h=sim.h, pd=sim.pd, ys=ys, eps=sim.eps, L=L,
                c_p=mat.p_wave, ms_per_frame=1000.0 * wall / max(nt, 1),
                rtf=(wall / max(nt, 1)) / FRAME, r_out=r_out, x0=x0, zc=zc)
    if dump is not None:
        x = sim.x.numpy()
        pos, _ = sim.body_state(wid)
        np.save(os.path.join(OUT, dump + ".npy"), x)
        meta = dict(info, wheel=pos.tolist(), r=r, b=WHEEL["b"], hg=WHEEL["hg"], lug=lug,
                    contact=contact, pmass=float(sim.P.pmass), s=s)
        meta["grid"] = None
        with open(os.path.join(OUT, dump + ".json"), "w") as fh:
            json.dump(meta, fh)
    return rows, info


def write_csv(path, rows):
    if not rows:
        return
    keys = []
    for rw in rows:          # the union: the dumped slip carries the rut metrics too
        keys += [k for k in rw if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, restval="")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    def val(k, v):
        if k in ("phase", "lug", "contact", "soil"):
            return v
        return float(v) if v != "" else float("nan")
    with open(path) as fh:
        return [{k: val(k, v) for k, v in rw.items()} for rw in csv.DictReader(fh)]


def steady(rows, phase, key, tail=0.25):
    sel = [rw for rw in rows if rw["phase"] == phase]
    k = max(1, int(round(tail / FRAME)))
    vals = np.array([rw[key] for rw in sel[-k:]])
    return float(vals.mean()), float(vals.std())


# --- analysis of a dumped bed -------------------------------------------------

def rut_metrics(x, meta):
    """Penetration into the collider's outer contact surface, and the rut's
    bulk density (mass / volume under the free surface) against the untouched
    bed beside it."""
    xw, yw, zw = meta["wheel"]
    pd, h, eps = meta["pd"], meta["h"], meta["eps"]
    r_c = meta["r"] + (meta["hg"] if meta["lug"] in ("envelope", "coulomb") else 0.0)
    d = x - np.array([xw, yw, zw])
    rl = np.hypot(d[:, 0], d[:, 1])
    inw = np.abs(d[:, 2]) <= 0.5 * meta["b"]
    pen = (r_c + eps) - rl[inw]
    # Column tops on a pd grid -> local surface; mass per column below it.
    cb = 2.0 * pd
    L, Wd = meta["L"], WD
    nx, nz = int(math.ceil(L / cb)), int(math.ceil(Wd / cb))
    ix = np.clip((x[:, 0] / cb).astype(int), 0, nx - 1)
    iz = np.clip((x[:, 2] / cb).astype(int), 0, nz - 1)
    top = np.full((nx, nz), -np.inf)
    np.maximum.at(top, (ix, iz), x[:, 1])
    cnt = np.zeros((nx, nz))
    np.add.at(cnt, (ix, iz), 1.0)
    surf = top + 0.5 * pd
    rho_col = cnt * meta["pmass"] / (cb * cb * np.maximum(surf, 1e-6))
    xc = (np.arange(nx) + 0.5) * cb
    zcc = (np.arange(nz) + 0.5) * cb
    XX, ZZ = np.meshgrid(xc, zcc, indexing="ij")
    # The rut: from just past the static pit's centre to just behind the tread.
    x_lo = meta["x0"] + 0.15
    x_hi = xw - meta["r_out"] - 0.03
    rut = (XX > x_lo) & (XX < x_hi) & (np.abs(ZZ - zw) < 0.5 * meta["b"] - cb)
    far = (XX > x_lo) & (XX < x_hi) & (np.abs(ZZ - zw) > 0.5 * meta["b"] + 0.08) & \
          (ZZ > 2 * cb) & (ZZ < Wd - 2 * cb)
    return dict(max_pen_h=float(pen.max() / h), n_pen=int((pen > 0).sum()),
                rho_rut=float(rho_col[rut].mean()) if rut.any() else float("nan"),
                rho_far=float(rho_col[far].mean()) if far.any() else float("nan"),
                rut_depth_mm=float(1000 * (meta["ys"] - surf[rut].mean())) if rut.any() else float("nan"))


def rut_plot(tag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.load(os.path.join(OUT, tag + ".npy"))
    m = json.load(open(os.path.join(OUT, tag + ".json")))
    xw, yw, zw = m["wheel"]
    pd, ys, L, r = m["pd"], m["ys"], m["L"], m["r"]
    fig, ax = plt.subplots(3, 1, figsize=(11, 12.5), gridspec_kw=dict(height_ratios=[1.1, 1.0, 0.8]))
    # Top view: column top + pd/2 minus the settled surface.
    cb = pd
    nx, nz = int(math.ceil(L / cb)), int(math.ceil(WD / cb))
    ix = np.clip((x[:, 0] / cb).astype(int), 0, nx - 1)
    iz = np.clip((x[:, 2] / cb).astype(int), 0, nz - 1)
    Hm = np.full((nx, nz), -np.inf)
    np.maximum.at(Hm, (ix, iz), x[:, 1])
    Hm[~np.isfinite(Hm)] = np.nan
    dH = 1000 * (Hm + 0.5 * pd - ys)
    a = ax[0]
    im = a.imshow(dH.T, origin="lower", extent=[0, L, 0, WD], aspect="equal", cmap="RdBu_r",
                  vmin=-50, vmax=50)
    a.plot([xw], [zw], "k+", ms=12)
    xs_cut = 0.5 * (m["x0"] + 0.15 + xw - m["r_out"] - 0.03)
    a.axvline(xs_cut, color="k", ls=":", lw=1)
    a.set_title(f"{tag}: top view, surface minus settled bed (mm); + = hub; dotted = cross-section", fontsize=9)
    a.set_xlabel("x (m)")
    a.set_ylabel("z (m)")
    plt.colorbar(im, ax=a, fraction=0.02)
    # Cross-section of the rut behind the wheel: mean column top over +-2 cm.
    a = ax[1]
    sel = np.abs(x[:, 0] - xs_cut) < 0.02
    izs = np.clip((x[sel, 2] / cb).astype(int), 0, nz - 1)
    prof = np.full(nz, -np.inf)
    np.maximum.at(prof, izs, x[sel, 1])
    zc_ = (np.arange(nz) + 0.5) * cb
    a.plot(1000 * (zc_ - zw), 1000 * (prof + 0.5 * pd - ys), "k.-", ms=3)
    a.axvspan(-500 * m["b"], 500 * m["b"], color="C1", alpha=0.15, label="wheel width")
    a.axhline(0, color="0.5", lw=0.6)
    a.set_xlabel("z - z_wheel (mm)")
    a.set_ylabel("surface - settled bed (mm)")
    a.set_title(f"cross-section at x = {xs_cut:.2f} m (behind the wheel)", fontsize=9)
    a.legend(fontsize=8)
    # Side view of the centre slab around the wheel.
    a = ax[2]
    sl = (np.abs(x[:, 2] - zw) < 0.05) & (np.abs(x[:, 0] - xw) < 0.45)
    d = x[sl]
    rl = np.hypot(d[:, 0] - xw, d[:, 1] - yw)
    a.scatter(1000 * (d[:, 0] - xw), 1000 * (d[:, 1] - ys), s=2,
              c=np.where(rl < r + m["eps"], "C3", "0.4"))
    tt = np.linspace(0, 2 * np.pi, 400)
    for rr, ls in ((r, "-"), (r + m["hg"], "--")):
        a.plot(1000 * rr * np.cos(tt), 1000 * (yw - ys + rr * np.sin(tt)), "b" + ls, lw=0.8)
    a.set_xlim(-450, 450)
    a.set_ylim(-150, 80)
    a.set_aspect("equal")
    a.set_xlabel("x - x_wheel (mm)")
    a.set_ylabel("y - settled bed (mm)")
    a.set_title("centre slab |z - zc| < 50 mm; rim (solid), grouser tips (dashed); red = inside r + eps",
                fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT, f"rut_{tag}.png")
    fig.savefig(path, dpi=90)
    print("wrote", path, flush=True)


# --- modes --------------------------------------------------------------------

def sweep_name(soil=SOIL, lug=LUG, h=H, load=N_LOAD, tag=TAG):
    return f"sweep_{soil}_{lug}_h{h * 1000:.0f}_N{load:.0f}{tag}"


def mode_wheel():
    slips = [float(v) for v in cli_arg("--slips", ",".join(str(s) for s in SLIPS), str).split(",")]
    dump_slips = [float(v) for v in cli_arg("--dump-slips", "0.3,0.7", str).split(",") if v]
    name = sweep_name()
    # A sweep may be run in pieces: keep the other slips of an earlier summary.
    prev = os.path.join(OUT, name + ".csv")
    old = [d for d in read_csv(prev) if all(abs(d["s"] - s) > 1e-6 for s in slips)] \
        if os.path.exists(prev) else []
    out = []
    for s in slips:
        t0 = time.perf_counter()
        dump = f"bed_{name}_s{int(round(s * 100)):02d}" if any(abs(s - q) < 1e-6 for q in dump_slips) else None
        rows, info = run_wheel(s=s, dump=dump)
        wall = time.perf_counter() - t0
        write_csv(os.path.join(OUT, f"{name}_s{int(round(s * 100)):02d}.csv"), rows)
        dr = [rw for rw in rows if rw["phase"] == "drive"]
        xs = info["x0"] + SKIP
        win = [rw for rw in dr if rw["x"] >= xs]
        arr = lambda k: np.array([rw[k] for rw in win])   # noqa: E731
        dp, fy, tz, zz, fb = arr("Fx"), arr("Fy"), arr("Tz"), arr("sink"), arr("floor_minus_soil")
        half = len(win) // 2
        d = dict(s=s, soil=SOIL, lug=LUG, contact=CONTACT, h=H, E=E, rho=RHO, phi=PHI, mu=MU_RIM,
                 cap_p0=SP["cap_p0"], cap_lam=SP["cap_lam"], cap_pmax=SP["cap_pmax"], ppc=PPC, load=N_LOAD,
                 jh_k=JH_K,
                 DP=dp.mean(), DP_N=dp.mean() / N_LOAD, DP_std_N=dp.std() / N_LOAD,
                 angle=math.degrees(math.atan(dp.mean() / N_LOAD)),
                 angle_drift=math.degrees(math.atan(dp[half:].mean() / N_LOAD))
                 - math.degrees(math.atan(dp[:half].mean() / N_LOAD)),
                 T=tz.mean(), sink=zz.mean(), sink_rim=zz.mean() - WHEEL["hg"],
                 sink_drift=zz[half:].mean() - zz[:half].mean(),
                 sink_static=steady(rows, "load", "sink", 0.2)[0], closure=fy.mean() / N_LOAD,
                 bin_closure=fb.mean() / N_LOAD,
                 win_s=len(win) * FRAME, win_m=len(win) * FRAME * V,
                 n=info["n"], nsub=info["nsub"], ms_per_frame=info["ms_per_frame"], rtf=info["rtf"],
                 wall_s=wall)
        if dump:
            x = np.load(os.path.join(OUT, dump + ".npy"))
            meta = json.load(open(os.path.join(OUT, dump + ".json")))
            d.update(rut_metrics(x, meta))
            rut_plot(dump)
        out.append(d)
        print(f"  s_eff={s:.2f}  DP/N={d['DP_N']:+.3f} (std {d['DP_std_N']:.3f}, angle {d['angle']:+5.1f} deg,"
              f" drift {d['angle_drift']:+.1f})  T={d['T']:6.1f} N m  sink(tips)={1000 * d['sink']:5.1f} mm"
              f" (drift {1000 * d['sink_drift']:+.1f}, static {1000 * d['sink_static']:.1f})"
              f"  Fy/N={d['closure']:.3f}  bin={d['bin_closure']:.3f}  win={d['win_s']:.1f}s"
              f"  {d['ms_per_frame']:.1f} ms/frame  n={info['n']} nsub={info['nsub']}  wall {wall:.0f} s"
              + (f"  rut={d['rut_depth_mm']:.1f} mm rho_rut/far={d['rho_rut'] / d['rho_far']:.3f}" if dump else ""),
              flush=True)
        write_csv(os.path.join(OUT, name + ".csv"), sorted(old + out, key=lambda d: d["s"]))


def mode_esweep():
    out = []
    for E_ in [float(v) for v in cli_arg("--Es", "0.5e6,1e6,2e6,5e6", str).split(",")]:
        t0 = time.perf_counter()
        rows, info = run_wheel(s=None, E_=E_, unload=True, length=0.9)
        z_l, _ = steady(rows, "load", "sink", 0.2)
        z_u, _ = steady(rows, "unload", "sink", 0.2)
        fy, fys = steady(rows, "load", "Fy", 0.2)
        d = dict(E=E_, c_p=info["c_p"], nsub=info["nsub"], sink_loaded=z_l, sink_unloaded=z_u,
                 elastic_share=(z_l - z_u) / max(z_l + info["eps"], 1e-9),
                 closure=fy / N_LOAD, wall_s=time.perf_counter() - t0)
        # The share is of the travel INTO the bed of the collider's outer
        # contact surface (the grouser envelope or tips, plus the skin eps),
        # which touches first.
        out.append(d)
        print(f"  E={E_:.1e}  c_p={info['c_p']:.1f}  nsub={info['nsub']}  sink={1000 * z_l:.1f} mm"
              f"  unloaded={1000 * z_u:.1f} mm  elastic share={100 * d['elastic_share']:.1f} %"
              f"  Fy/N={fy / N_LOAD:.3f} (+-{fys / N_LOAD:.3f})  wall {d['wall_s']:.0f} s", flush=True)
    write_csv(os.path.join(OUT, f"esweep_{SOIL}_{LUG}_h{H * 1000:.0f}.csv"), out)


def run_plate(bw, plate_L=0.30, rate=0.05, zmax=0.06):
    Lb, Wb = 0.8, 0.8
    sim, nsub = make_sim(material(), Lb, Wb, DEPTH, H, CONTACT)
    hy = 0.10
    pid = sim.add_body(gm.BOX, (0.5 * bw, hy, 0.5 * plate_L), (0.5 * Lb, DEPTH + 1.0, 0.5 * Wb),
                       mu=MU_RIM, mode=gm.KINEMATIC)
    settle(sim, nsub, 0.2)
    ys = sim.surface_height(0.0, Lb, 0.0, Wb)
    sim.set_body(pid, pos=(0.5 * Lb, ys + hy + sim.eps + 0.003, 0.5 * Wb), vel=(0, -rate, 0))
    sim.take_mean_force()
    rows, t = [], 0.0
    while True:
        sim.step(nsub)
        f, _ = sim.take_mean_force()
        t += FRAME
        pos, _ = sim.body_state(pid)
        z = ys - (pos[1] - hy - sim.eps)
        rows.append(dict(t=t, z=z, Fy=f[0, 1], p=f[0, 1] / (bw * plate_L)))
        if z >= zmax:
            break
    return rows


def fit_bekker(curves, zlo, zhi):
    """Standard two-plate method, p = (k_c/b + k_phi) z^n with a common n (as
    warp_wheel_testbed.fit_bekker)."""
    ns, fits = [], []
    for bw, rows in curves:
        z = np.array([rw["z"] for rw in rows])
        p = np.array([rw["p"] for rw in rows])
        m = (z >= zlo) & (z <= zhi) & (p > 0)
        A = np.vstack([np.ones(m.sum()), np.log(z[m])]).T
        _, n = np.linalg.lstsq(A, np.log(p[m]), rcond=None)[0]
        ns.append(n)
        fits.append((bw, z[m], p[m]))
    n = float(np.mean(ns))
    keq = [float(np.exp(np.mean(np.log(p) - n * np.log(z)))) for _, z, p in fits]
    (b1, _, _), (b2, _, _) = fits
    kc = (keq[0] - keq[1]) / (1.0 / b1 - 1.0 / b2)
    kphi = keq[0] - kc / b1
    return dict(n=n, n_each=[float(v) for v in ns], k_eq=keq, k_c=float(kc), k_phi=float(kphi),
                b=[b1, b2], z_range=[zlo, zhi])


def mode_plate():
    curves = []
    for bw in (0.10, 0.20):
        rows = run_plate(bw)
        write_csv(os.path.join(OUT, f"plate_b{int(bw * 1000)}_{SOIL}_h{H * 1000:.0f}.csv"), rows)
        curves.append((bw, rows))
        zz = [0.01, 0.02, 0.03, 0.05]
        print(f"  plate b={bw:.2f} m: " + "  ".join(
            f"z={z * 1000:.0f}mm p={np.interp(z, [r['z'] for r in rows], [r['p'] for r in rows]) / 1000:.2f}kPa"
            for z in zz), flush=True)
    fit = fit_bekker(curves, max(0.8 * H, 0.008), 0.05)
    fit.update(E=E, phi=PHI, rho=RHO, h=H, soil=SOIL)
    with open(os.path.join(OUT, f"bekker_fit_{SOIL}_h{H * 1000:.0f}.json"), "w") as fh:
        json.dump(fit, fh, indent=2)
    print("  Bekker fit:", json.dumps(fit), flush=True)


def wong_reece(W, s, r, b, kc, kphi, n, c, phi_deg, K, c1=0.43, c2=0.32):
    """Rigid driven wheel, Wong & Reece 1967, theta_2 = 0. Copied from
    warp_wheel_testbed.wong_reece (that module parses argv at import).
    Returns sinkage, DP/W, T."""
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


def twin_curve(fit, ours, phi, slips):
    """Bekker twin (b): our plates, K fitted to OUR flat DP(s) at low slip
    (0 < s <= 0.2), grouser effective radius r + h_g."""
    r_eff, b = WHEEL["r"] + WHEEL["hg"], WHEEL["b"]
    # The grains' slip is already at r_eff (Wong-Reece's radius here).
    se = lambda s: s   # noqa: E731
    low = [(d["s"], d["DP"] / N_LOAD) for d in ours if 0.0 < d["s"] <= 0.2 + 1e-9]
    Ks = np.geomspace(0.001, 0.2, 80)
    err = [sum((wong_reece(N_LOAD, se(s), r_eff, b, fit["k_c"], fit["k_phi"], fit["n"], 0.0, phi, K)[1] - y) ** 2
               for s, y in low) for K in Ks]
    K = float(Ks[int(np.argmin(err))])
    curve = [(s,) + wong_reece(N_LOAD, se(s), r_eff, b, fit["k_c"], fit["k_phi"], fit["n"], 0.0, phi, K)
             for s in slips]
    return K, curve


def nasa_ref(s):
    """The NASA traction angle at slip s: the mean of the repeat points at each
    tested slip (rounded to 0.05), linearly interpolated between them."""
    keys = np.round(NASA_TREC[:, 0] / 5.0) * 0.05
    us = np.unique(keys)
    means = np.array([NASA_TREC[keys == u, 1].mean() for u in us])
    return float(np.interp(s, us, means))


def gate(rows):
    """Pass/fail of the phase-1 traction gate on one sweep."""
    errs = [(r["s"], r["angle"] - nasa_ref(r["s"])) for r in rows if 0.1 - 1e-9 <= r["s"] <= 0.8 + 1e-9]
    s0 = [r["angle"] for r in rows if abs(r["s"]) < 1e-9]
    sign_ok = bool(s0) and (s0[0] < 0.0) == (nasa_ref(0.0) < 0.0)
    worst = max((abs(e) for _, e in errs), default=float("nan"))
    return dict(errors=errs, worst=worst, sign_ok=sign_ok, passed=bool(errs) and worst <= 3.0 and sign_ok)


def s_rim_of(s_eff):
    """The same kinematics as a slip at the rim (r = 0.25 m), NASA's definition."""
    return 1.0 - (1.0 - np.asarray(s_eff)) * R_EFF / WHEEL["r"]


def s_eff_of(s_rim):
    return 1.0 - (1.0 - np.asarray(s_rim)) * WHEEL["r"] / R_EFF


def implied_slope(rows):
    """Steady slip on a slope theta, ignoring load transfer: DP/N(s) = tan(theta).
    Returns (s_eff, theta deg) along the (monotone-ised) DP/N curve."""
    ss = np.array([r["s"] for r in rows])
    dpn = np.maximum.accumulate(np.array([r["DP_N"] for r in rows]))
    return ss, np.degrees(np.arctan(dpn))


def slope_at_slip(rows, s):
    ss, th = implied_slope(rows)
    return float(np.interp(s, ss, th))


def mode_plot():
    import glob
    import re
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pat = re.compile(r"^sweep_(\w+?)_(\w+?)_h(\d+)_N(\d+)(.*)\.csv$")
    sweeps = []
    for f in sorted(glob.glob(os.path.join(OUT, "sweep_*.csv"))):
        mt = pat.match(os.path.basename(f))
        if not mt or re.search(r"_s\d\d\.csv$", f):      # a per-slip time series
            continue
        rows = sorted(read_csv(f), key=lambda r: r["s"])
        sweeps.append(dict(name=os.path.basename(f)[:-4], soil=mt.group(1), lug=mt.group(2),
                           h=int(mt.group(3)), N=int(mt.group(4)), tag=mt.group(5), rows=rows))
    full = [sw for sw in sweeps if len(sw["rows"]) >= 4]
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.6))
    se_n = s_eff_of(NASA_TREC[:, 0] / 100.0)
    tn = np.tan(np.radians(NASA_TREC[:, 1]))
    ax[0].scatter(se_n, tn, s=30, c="0.55", marker="s", zorder=1,
                  label="NASA TREC, GRC-3, N = 171.7 N (slip converted to r_eff)")
    ax[1].scatter(se_n, NASA_TREC[:, 1], s=30, c="0.55", marker="s", zorder=1, label="NASA TREC (slip at r_eff)")
    ax[1].scatter(NASA_TREC[:, 0] / 100.0, NASA_TREC[:, 1], s=22, facecolors="none", edgecolors="0.7",
                  marker="s", zorder=1, label="NASA TREC as published (slip at the rim)")
    styles = {}
    summary = {}
    for i, sw in enumerate(sweeps):
        rows = sw["rows"]
        col = "C%d" % (i % 10)
        full_ = len(rows) >= 4
        lab = f"{sw['soil']}, {sw['lug']}, h = {sw['h']} mm, N = {sw['N']} N{sw['tag']}"
        ss = [r["s"] for r in rows]
        kw = dict(color=col, ms=6 if full_ else 5, lw=2 if full_ else 0.8, alpha=1.0 if full_ else 0.6)
        ax[0].plot(ss, [r["DP_N"] for r in rows], "o-" if full_ else "x:", label=lab, **kw)
        ax[1].plot(ss, [r["angle"] for r in rows], "o-" if full_ else "x:", label=lab, **kw)
        ax[2].plot(ss, [1000 * r["sink"] for r in rows], "o-" if full_ else "x:", label=lab + " (tips)", **kw)
        if full_:
            ax[2].plot(ss, [1000 * r["sink_rim"] for r in rows], "--", color=col, lw=1, label="  rim = tips - 25 mm")
        styles[sw["name"]] = col
        summary[sw["name"]] = dict(rows=[dict(s_eff=r["s"], s_rim=float(s_rim_of(r["s"])), DP_N=r["DP_N"],
                                               angle=r["angle"], sink_tips_mm=1000 * r["sink"],
                                               sink_rim_mm=1000 * r["sink_rim"], closure=r["closure"])
                                          for r in rows])
        if full_:
            summary[sw["name"]].update(slope_at_50_eff=slope_at_slip(rows, 0.5),
                                       slope_at_50_rim=slope_at_slip(rows, float(s_eff_of(0.5))))
    ax[0].axhspan(0.45, 0.75, color="C2", alpha=0.07, lw=0, label="plausible saturation 0.45-0.75")
    ax[0].axhline(0, color="0.5", lw=0.6)
    ax[0].set_ylabel("DP / N")
    ax[1].set_ylabel("traction angle atan(DP / N) (deg)")
    ax[2].set_ylabel("steady sinkage (mm)")
    for a in ax:
        a.set_xlabel("slip at the grouser tips  s_eff = 1 - v / (omega r_eff),  r_eff = 0.275 m")
        a.grid(alpha=0.3)
        a.set_xlim(-0.05, 0.85)
        a.legend(fontsize=6.5, loc="best")
    ax[0].set_title("drawbar pull, v = 0.2 m/s (VV mode)", fontsize=10)
    ax[1].set_title("traction angle", fontsize=10)
    ax[2].set_title("sinkage vs slip", fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_traction.png")
    fig.savefig(path, dpi=100)
    print("wrote", path)
    # The implied slip-vs-slope curve.
    fig, a = plt.subplots(figsize=(9, 6.2))
    a.plot(NASA_SLOPE[:, 0], NASA_SLOPE[:, 1], "ks-", ms=7, lw=1.2, zorder=6,
           label="NASA MGRU3 tilt bed 2022 (slip at the rim)")
    a.axvspan(13, 22, color="C2", alpha=0.08, lw=0, label="plausible: 50 % slip at 13-22 deg")
    a.axhline(50, color="0.5", lw=0.6, ls=":")
    for sw in (full or sweeps):
        rows = sw["rows"]
        ss, th = implied_slope(rows)
        col = styles[sw["name"]]
        lab = f"grains {sw['soil']}, h = {sw['h']} mm, N = {sw['N']} N"
        a.plot(th, 100 * ss, "o-", color=col, lw=2, label=lab + ", slip at r_eff")
        a.plot(th, 100 * s_rim_of(ss), "--", color=col, lw=1, label=lab + ", same, as rim slip")
    a.set_xlabel("slope theta (deg), steady where DP/N(s) = tan(theta) (no load transfer)")
    a.set_ylabel("slip (%)")
    a.set_xlim(-5, 35)
    a.set_ylim(-15, 100)
    a.grid(alpha=0.3)
    a.legend(fontsize=7.5, loc="upper left")
    a.set_title("Implied slip vs slope from the single-wheel DP/N curve", fontsize=10)
    fig.tight_layout()
    path = os.path.join(OUT, "fig_slope.png")
    fig.savefig(path, dpi=100)
    print("wrote", path)
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    for k, v in summary.items():
        if "slope_at_50_eff" in v:
            print(f"{k}: 50 % slip at {v['slope_at_50_eff']:.1f} deg (s_eff) / {v['slope_at_50_rim']:.1f} deg (rim slip)")


def mode_rt():
    """The real cost: 4 wheels in GranularPatches (moving windows), loaded to
    N_LOAD, driving at v = 0.2 m/s at slip --rt-slip, CUDA graph, wall ms per
    1/60 s frame (step_frame: upload, window/seed/sort, substeps, readback).
    Several h are interleaved rep by rep."""
    hs = [float(v) for v in cli_arg("--hs", "0.04,0.03", str).split(",")]
    frames = cli_arg("--frames", 120, int)
    reps = cli_arg("--reps", 3, int)
    px, pz = cli_arg("--patch-x", 0.6, float), cli_arg("--patch-z", 0.4, float)
    pdp = cli_arg("--patch-depth", DEPTH, float)
    s = cli_arg("--rt-slip", 0.3, float)
    om = V / ((1.0 - s) * R_EFF)
    spec = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "rover", "rover_spec.json")))
    hubs = np.array([spec["hubs"][k] for k in ("FL", "FR", "RL", "RR")], float)
    axis = np.tile([0.0, 0.0, 1.0], (4, 1))
    zero = np.zeros((4, 3))
    hw = 0.5 * WHEEL["b"]
    runs = {}
    for h in hs:
        sim = gm.GranularPatches(material(), h, 4, patch_size=(px, pz), depth=pdp,
                                 hf_origin=(-2.0, -2.0), hf_size=(8.0, 4.0), floor_y=0.0,
                                 ppc_spacing=PPC, cfl=CFL, jh_k=JH_K)
        nsub = sim.nsub_for(om * R_EFF)
        pos = hubs.copy()
        pos[:, 1] = pdp + R_EFF + 0.05
        for _ in range(30):          # seed + settle under gravity, wheels hovering
            sim.set_wheels(pos, zero, zero, axis, 0.0, R_EFF, hw, mu=MU_RIM, mass=N_LOAD / G,
                           free_y=False, set_y=True)
            sim.step_frame(nsub)
        pos[:, 1] = pdp + R_EFF + sim.eps + 0.002
        sim.set_wheels(pos, zero, zero, axis, 0.0, R_EFF, hw, mu=MU_RIM, mass=N_LOAD / G,
                       free_y=True, set_y=True)
        sim.step_frame(nsub)
        runs[h] = dict(sim=sim, nsub=nsub, pos=pos, t=0.0, ms=[], outs=[])
        print(f"  h={1000 * h:.0f} mm: {sim.n_slots:,} slots, nsub={nsub}, blocks {sim.ix}x{sim.dims[1]}x{sim.iz} cells",
              flush=True)

    def advance(R, nframes, timed):
        sim = R["sim"]
        for _ in range(nframes):
            load = N_LOAD * min(1.0, R["t"] / 0.3)
            vel, omg = zero.copy(), zero.copy()
            if R["t"] > 0.6:
                vel[:, 0] = V
                omg[:, 2] = -om
            sim.set_wheels(R["pos"], vel, omg, axis, load, R_EFF, hw, mu=MU_RIM, mass=N_LOAD / G)
            t0 = time.perf_counter()
            out = sim.step_frame(R["nsub"])
            dt = time.perf_counter() - t0
            R["pos"][:, 0] += vel[:, 0] * FRAME
            if timed:
                R["ms"].append(1000.0 * dt)
            R["outs"].append(out)
            R["t"] += FRAME

    for R in runs.values():
        advance(R, 90, False)        # load, start driving, capture the graphs
    res = []
    for rep_i in range(reps):
        for h in hs:
            R = runs[h]
            n0 = len(R["ms"])
            advance(R, frames, True)
            print(f"  rep {rep_i} h={1000 * h:.0f} mm: {np.mean(R['ms'][n0:]):.2f} ms/frame "
                  f"(median {np.median(R['ms'][n0:]):.2f})", flush=True)
    for h in hs:
        R = runs[h]
        o = np.array(R["outs"])
        last = o[-frames:]
        finite = bool(np.isfinite(o).all())
        dpn = float(last[:, :, 0].mean() / N_LOAD)
        sink = float((pdp - (last[:, :, 9] - R_EFF)).mean())
        per_rep = [float(np.mean(R["ms"][i * frames:(i + 1) * frames])) for i in range(reps)]
        d = dict(h=h, nsub=R["nsub"], slots=R["sim"].n_slots, alive=float(last[-1, :, 12].sum()),
                 ms_mean=float(np.mean(R["ms"])), ms_median=float(np.median(R["ms"])),
                 ms_p95=float(np.percentile(R["ms"], 95)), ms_rep_min=min(per_rep), ms_rep_max=max(per_rep),
                 finite=finite, DP_N_patch=dpn, sink_tips_patch_mm=1000 * sink, patch_x=px, patch_z=pz,
                 patch_depth=pdp, s_eff=s, soil=SOIL, E=E, ppc=PPC, jh_k=JH_K, **SP)
        res.append(d)
        print(f"  h={1000 * h:.0f} mm  nsub={R['nsub']}  {d['alive']:.0f} live grains  ms/frame mean {d['ms_mean']:.2f}"
              f" median {d['ms_median']:.2f} p95 {d['ms_p95']:.2f} (reps {d['ms_rep_min']:.2f}-{d['ms_rep_max']:.2f})"
              f"  finite={finite}  patch DP/N={dpn:+.3f}  sink(tips)={1000 * sink:.1f} mm", flush=True)
    write_csv(os.path.join(OUT, f"rt_{SOIL}_E{E:.0e}_{px:.2f}x{pz:.2f}x{pdp:.2f}{TAG}.csv"), res)


if __name__ == "__main__":
    mode = cli_arg("--mode", "wheel", str)
    print(f"mode={mode} soil={SOIL} {SP} lug={LUG} mu={MU_RIM:.3f} jh_k={JH_K} contact={CONTACT} E={E:.2e} h={H} "
          f"ppc_spacing={PPC} cfl={CFL} N={N_LOAD}", flush=True)
    dict(wheel=mode_wheel, esweep=mode_esweep, plate=mode_plate, plot=mode_plot, rt=mode_rt)[mode]()
