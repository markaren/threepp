"""Phase-2 gate: four moving-window MLS-MPM patches under a car's wheels.

Measures threepp.granular_mpm.GranularPatches (K patches in ONE simulation,
one CUDA graph per frame, heightfield write-back / seeding):

    python granular_patch_gate.py --part g1     # cost per 1/60 s frame, interleaved
    python granular_patch_gate.py --part g2     # parity vs the phase-1 static bin
    python granular_patch_gate.py --part g3     # rut persistence + re-entry (+ render)
    python granular_patch_gate.py --part all

    --out DIR   output folder (default: _granular_wheel_gate next to this script)

Output: CSVs, PNG plots, renders and logs in --out.

Iteration 2 (2026-09-22): GranularPatches' defaults are now the shipped
configuration (0.8 x 0.44 m blocks, CFL 0.5, device re-sort every frame,
tiled P2G, pit-free write-back); make_car_sim() follows them and passes any
extra GranularPatches keyword through; the iteration-2 gate scripts (P1 cost,
P2 parity, P3 rut floor + render) import this module for Car / bring_up. The
car-side coupling this gated is warp_mudsnow_drive.py's gravel lane.
"""
import csv
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # python/
sys.path.insert(0, os.path.dirname(HERE))                      # python/examples

import numpy as np

from threepp import granular_mpm as gm

OUT = (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv
       else os.path.join(HERE, "_granular_wheel_gate"))
os.makedirs(OUT, exist_ok=True)
FRAME = 1.0 / 60.0
G = 9.81
R_W, B_W, W_LOAD = 0.36, 0.24, 4500.0
TRACK, BASE = 1.6, 2.66


def arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


def material(E=3.0e6):
    return gm.Material(E=E, nu=0.3, rho=1700.0, phi_deg=36.0, cohesion=0.0,
                       cap_p0=2.0e3, cap_lambda=0.025)


def write_csv(path, rows):
    if rows:
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


def gpu_status():
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        return q.stdout.strip(), a.stdout.strip() or "(no compute apps)"
    except Exception as e:  # noqa: BLE001
        return f"nvidia-smi failed: {e}", ""


def other_gpu_pids():
    """PIDs of OTHER python processes holding a GPU context (another session's jobs)."""
    try:
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10).stdout
    except Exception:  # noqa: BLE001
        return set()
    me = os.getpid()
    out = set()
    for line in a.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and "python" in parts[1].lower():
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            if pid != me:
                out.add(pid)
    return out


class Car:
    """Kinematic car (rear-axle reference) driving four wheels over the patches."""

    def __init__(self, sim, x0=3.0, z0=0.0, v=3.0, radius=None, slip=0.1, W=W_LOAD,
                 mass=None, damp=0.0):
        self.sim, self.v, self.R, self.s, self.W = sim, v, radius, slip, W
        self.x, self.z, self.psi = x0, z0, 0.0
        self.speed = 0.0
        self.mass, self.damp = mass, damp
        # FL, FR, RL, RR (body frame: +x forward, +z left)
        self.off = np.array([[BASE, TRACK / 2], [BASE, -TRACK / 2], [0.0, TRACK / 2], [0.0, -TRACK / 2]])

    def wheel_pos(self, x, z, psi):
        f = np.array([math.cos(psi), math.sin(psi)])
        a = np.array([-math.sin(psi), math.cos(psi)])
        return np.array([[x + o[0] * f[0] + o[1] * a[0], z + o[0] * f[1] + o[1] * a[1]] for o in self.off])

    def advance_pose(self, x, z, psi, ds):
        if self.R is None:
            return x + ds * math.cos(psi), z + ds * math.sin(psi), psi
        dpsi = ds / self.R
        cx, cz = x - self.R * math.sin(psi), z + self.R * math.cos(psi)
        psi2 = psi + dpsi
        return cx + self.R * math.sin(psi2), cz - self.R * math.cos(psi2), psi2

    def command(self, load, y=None):
        """Per-wheel hub pose / velocity / spin / axis for the next frame."""
        p0 = self.wheel_pos(self.x, self.z, self.psi)
        eps = 1e-3
        x1, z1, psi1 = self.advance_pose(self.x, self.z, self.psi, max(self.speed, 1e-4) * eps)
        p1 = self.wheel_pos(x1, z1, psi1)
        d = (p1 - p0)
        d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        vw = self.speed * np.linalg.norm(self.wheel_pos(*self.advance_pose(self.x, self.z, self.psi, 1e-3)) - p0,
                                         axis=1) / 1e-3
        pos = np.zeros((4, 3))
        pos[:, 0], pos[:, 2] = p0[:, 0], p0[:, 1]
        if y is not None:
            pos[:, 1] = y
        vel = np.zeros((4, 3))
        vel[:, 0], vel[:, 2] = d[:, 0] * vw, d[:, 1] * vw
        axis = np.zeros((4, 3))
        axis[:, 0], axis[:, 2] = -d[:, 1], d[:, 0]
        w = vw / ((1.0 - self.s) * R_W)
        omega = -axis * w[:, None]
        self.sim.set_wheels(pos, vel, omega, axis, load, R_W, 0.5 * B_W, mass=self.mass,
                            damp=self.damp, set_y=y is not None)

    def step(self, nsub, load):
        self.command(load)
        out = self.sim.step_frame(nsub)
        self.x, self.z, self.psi = self.advance_pose(self.x, self.z, self.psi, self.speed * FRAME)
        return out


def make_car_sim(E=3.0e6, ppc=0.63, patch=(0.8, 0.44), depth=0.3, h=0.04, hf=((-2.0, -6.0), (64.0, 16.0)),
                 **kw):
    sim = gm.GranularPatches(material(E), h, 4, patch_size=patch, depth=depth,
                             hf_origin=hf[0], hf_size=hf[1], ppc_spacing=ppc, **kw)
    return sim


def bring_up(car, v_target, nsub_drive, W=W_LOAD, t_load=0.3, t_hold=0.2, t_acc=0.25):
    """Seed, drop the wheels on the soil, ramp the load, accelerate. Returns rows."""
    sim = car.sim
    nsub0 = sim.nsub_for(2.0)
    y0 = sim.floor_y + sim.depth + R_W + sim.eps + 0.002
    car.command(0.0, y=y0 + 1.0)
    sim.step_frame(nsub0)                      # seeds all four blocks (fresh)
    for _ in range(8):
        car.command(0.0, y=y0 + 1.0)
        sim.step_frame(nsub0)
    car.command(0.0, y=y0)
    sim.step_frame(nsub0)
    rows = []
    n_load = int(round((t_load + t_hold) / FRAME))
    for f in range(n_load):
        out = car.step(nsub0, W * min(1.0, f * FRAME / t_load))
        rows.append(out)
    n_acc = int(round(t_acc / FRAME))
    for f in range(n_acc):
        car.speed = v_target * (f + 1) / n_acc
        out = car.step(nsub_drive, W)
        rows.append(out)
    return rows


# ------------------------------------------------------------------------------ G1
def part_g1():
    reps = arg("--reps", 6, int)
    nfr = arg("--frames", 30, int)
    util, apps = gpu_status()
    print("GPU before:", util, "|", apps.replace("\n", "; "))
    cases = [
        dict(name="v0.5", v=0.5),
        dict(name="v3", v=3.0),
        dict(name="v8", v=8.0),
        dict(name="v3_steer_R15", v=3.0, R=15.0),
        dict(name="v3_E5", v=3.0, E=5.0e6),
        dict(name="v3_8ppc", v=3.0, ppc=0.5),
        dict(name="v3_small_0.8x0.44", v=3.0, patch=(0.8, 0.44)),
        dict(name="v8_small_0.8x0.44", v=8.0, patch=(0.8, 0.44)),
    ]
    only = arg("--cases", None)
    if only:
        cases = [c for c in cases if c["name"] in only.split(",")]
    for c in cases:
        sim = make_car_sim(E=c.get("E", 3.0e6), ppc=c.get("ppc", 0.63), patch=c.get("patch", (1.0, 0.52)))
        car = Car(sim, x0=3.0, z0=0.0, v=c["v"], radius=c.get("R"), slip=0.1)
        c["nsub"] = sim.nsub_for(c["v"] / (1.0 - 0.1))
        c["sim"], c["car"] = sim, car
        rows = bring_up(car, c["v"], c["nsub"])
        for _ in range(20):
            rows.append(car.step(c["nsub"], W_LOAD))
        c["times"], c["rows"] = [], rows
        out = rows[-1]
        print(f"  {c['name']}: nsub={c['nsub']} slots={sim.n_slots} alive={out[:, gm.OUT_ALIVE].astype(int)} "
              f"ybot={np.round(out[:, gm.OUT_Y_BOTTOM], 3)} Fy/W={np.round(out[:, 1] / W_LOAD, 2)}")
    # Interleaved timed blocks: ms of wall time per 1/60 s frame, including the
    # host command, the graph (window/retire/seed + substeps + finalize) and the
    # synchronous readback.
    clean = []
    for rep in range(reps):
        # Another session on this machine sometimes renders: wait (<= 30 s) for
        # its python processes to leave the GPU, and flag the rep if they don't.
        t_wait = time.perf_counter()
        while True:
            others = other_gpu_pids()
            if not others or time.perf_counter() - t_wait > 30.0:
                break
            time.sleep(1.0)
        for c in cases:
            car = c["car"]
            ts = []
            for _ in range(nfr):
                t0 = time.perf_counter()
                out = car.step(c["nsub"], W_LOAD)
                ts.append(time.perf_counter() - t0)
                c["rows"].append(out)
            c["times"].append(1000.0 * float(np.median(ts)))
        others_after = other_gpu_pids()
        ok = not others and not others_after
        clean.append(ok)
        print(f"  rep {rep} {'clean' if ok else 'CONTENDED ' + str(others | others_after)}: "
              + "  ".join(f"{c['name']}={c['times'][-1]:.2f}" for c in cases))
    for c in cases:
        c["times_all"] = list(c["times"])
        if sum(clean) >= 3:
            c["times"] = [t for t, ok in zip(c["times"], clean) if ok]
    print(f"  clean reps: {sum(clean)} of {reps}")
    util2, apps2 = gpu_status()
    print("GPU after:", util2, "|", apps2.replace("\n", "; "))
    # Breakdown: the same sims, graphs of the parts alone (substeps only; the
    # window/retire/seed + finalize/readback only), interleaved.
    for c in cases:
        c["t_sub"], c["t_io"] = [], []
    for rep in range(3):
        t_wait = time.perf_counter()
        while other_gpu_pids() and time.perf_counter() - t_wait < 30.0:
            time.sleep(1.0)
        for c in cases:
            sim, car = c["sim"], c["car"]
            for key, parts in (("t_sub", (False, True, False)), ("t_io", (True, False, True))):
                car.command(W_LOAD)
                sim.step_frame(c["nsub"], parts=parts)       # capture once
                ts = []
                for _ in range(10):
                    car.command(W_LOAD)
                    t0 = time.perf_counter()
                    sim.step_frame(c["nsub"], parts=parts)
                    ts.append(time.perf_counter() - t0)
                c[key].append(1000.0 * float(np.median(ts)))
    res = []
    for c in cases:
        t = np.array(c["times"])
        rows = np.array(c["rows"][-reps * nfr:])
        fy = rows[:, :, 1] / W_LOAD
        sink_proxy = rows[:, :, gm.OUT_Y_BOTTOM]
        d = dict(case=c["name"], v=c["v"], nsub=c["nsub"], slots=c["sim"].n_slots,
                 n_clean_reps=len(c["times"]), ms_all_reps=" ".join(f"{t:.2f}" for t in c["times_all"]),
                 alive_mean=float(rows[:, :, gm.OUT_ALIVE].sum(1).mean()),
                 ms_median=float(np.median(t)), ms_min=float(t.min()), ms_max=float(t.max()),
                 x_realtime=float(np.median(t)) / (1000.0 * FRAME),
                 us_per_substep=1000.0 * float(np.median(t)) / c["nsub"],
                 ms_substeps_only=float(np.median(c["t_sub"])), ms_window_io_only=float(np.median(c["t_io"])),
                 FyW_mean=float(fy.mean()), FyW_std=float(fy.std()),
                 rim_bottom_mean=float(sink_proxy.mean()), overflow=float(rows[-1, 0, gm.OUT_OVERFLOW]),
                 reps=reps, frames_per_rep=nfr)
        res.append(d)
        print(f"{d['case']:>20s}: nsub={d['nsub']:3d}  {d['ms_median']:.2f} ms/frame "
              f"[{d['ms_min']:.2f}..{d['ms_max']:.2f}]  = {d['x_realtime']:.2f}x RT  "
              f"{d['us_per_substep']:.0f} us/sub  sub-only {d['ms_substeps_only']:.2f}  "
              f"io-only {d['ms_window_io_only']:.2f}  alive {d['alive_mean']:.0f}  Fy/W {d['FyW_mean']:.3f}"
              f"+-{d['FyW_std']:.2f}  ovf {d['overflow']:.0f}")
    write_csv(os.path.join(OUT, "g1_cost.csv"), res)
    with open(os.path.join(OUT, "g1_gpu.txt"), "w") as fh:
        fh.write(f"before: {util}\n{apps}\nafter: {util2}\n{apps2}\n")
    return res


# ------------------------------------------------------------------------------ G2
def part_g2():
    """One moving-window patch vs the phase-1 static bin, same h / ppc / E."""
    h, ppc, E, W, v = 0.04, 0.63, 3.0e6, W_LOAD, 0.25
    s = 0.3
    # Phase-1 testbed, with ppc_spacing forced to 0.63 (it has no flag for it).
    saved = sys.argv[:]
    sys.argv = [saved[0], "--h", str(h), "--E", str(E)]
    import warp_wheel_testbed as tb
    sys.argv = saved
    tb.CFG["Wd"] = 0.52

    def make_sim(mat, length, width, depth, h_=None):
        hh = tb.CFG["h"] if h_ is None else h_
        dt0 = mat.cfl_dt(hh, tb.CFL, vmax=2.0)
        nsub = int(math.ceil(FRAME / dt0))
        nsub += nsub & 1                           # the patch sim needs even; match it
        sim = gm.GranularMPM(mat, hh, lo=(0.0, 0.0, 0.0), hi=(length, 0.0, width), fill_height=depth,
                             dt=FRAME / nsub, headroom=0.30, max_bodies=1, ppc_spacing=ppc)
        return sim, nsub
    tb.make_sim = make_sim
    t0 = time.perf_counter()
    rows_st, info_st = tb.run_wheel(W, length=1.0)
    bin_static = tb.steady(rows_st, "load", "sink")[0]
    bin_fy = tb.steady(rows_st, "load", "Fy")[0] / W
    Ls = 3.2
    x_hi = Ls - R_W - 0.20
    x_lo = x_hi - 1.25 * 2 * R_W
    rows_sl, info_sl = tb.run_wheel(W, s=s, length=Ls)
    bin_dp = tb.window(rows_sl, "Fx", x_lo, x_hi)[0] / W
    bin_sink = tb.window(rows_sl, "sink", x_lo, x_hi)[0]
    bin_fy_sl = tb.window(rows_sl, "Fy", x_lo, x_hi)[0] / W
    bin_T = tb.window(rows_sl, "Tz", x_lo, x_hi)[0]
    print(f"  bin: nsub={info_st['nsub']} n={info_st['n']}  static {bin_static * 1000:.1f} mm Fy/W {bin_fy:.3f} | "
          f"s=0.3: DP/W {bin_dp:+.3f} sink {bin_sink * 1000:.1f} mm Fy/W {bin_fy_sl:.3f} T {bin_T:.0f}  "
          f"({time.perf_counter() - t0:.0f} s)")

    def patch_run(drive):
        sim = gm.GranularPatches(material(E), h, 1, patch_size=(1.0, 0.52), depth=0.3,
                                 hf_origin=(-2.0, -1.0), hf_size=(8.0, 2.0), ppc_spacing=ppc)
        nsub = sim.nsub_for(2.0)
        x0 = R_W + 0.12 if drive else 0.5
        zc = 0.26
        ax = np.array([[0.0, 0.0, 1.0]])
        y_park = 0.3 + R_W + 1.0

        def cmd(x, load, vx=0.0, w=0.0, y=None):
            p = np.array([[x, 0.0 if y is None else y, zc]])
            sim.set_wheels(p, np.array([[vx, 0.0, 0.0]]), np.array([[0.0, 0.0, -w]]), ax, load, R_W,
                           0.5 * B_W, set_y=y is not None)
        cmd(x0, 0.0, y=y_park)
        sim.step_frame(nsub)
        for _ in range(int(round(0.2 / FRAME))):
            cmd(x0, 0.0, y=y_park)
            sim.step_frame(nsub)
        xp = sim.particles(0)
        top = xp[:, 1][xp[:, 1] > xp[:, 1].max() - 0.8 * sim.pd]
        ys = float(top.mean() + 0.5 * sim.pd)
        cmd(x0, 0.0, y=ys + R_W + sim.eps + 0.002)
        sim.step_frame(nsub)
        rows = []
        t, x = 0.0, x0
        phase = "load"
        w = v / ((1.0 - s) * R_W)
        while True:
            if phase == "load":
                load = W * min(1.0, t / 0.4)
                cmd(x, load)
                if t >= 0.9:
                    if not drive:
                        break
                    phase = "drive"
            if phase == "drive":
                cmd(x, W, vx=v, w=w)
                x += v * FRAME
            out = sim.step_frame(nsub)
            t += FRAME
            rows.append(dict(t=t, phase=phase, x=float(out[0, 8]), y=float(out[0, 9]),
                             sink=ys - float(out[0, gm.OUT_Y_BOTTOM]), Fx=float(out[0, 0]),
                             Fy=float(out[0, 1]), Tz=float(out[0, 5]), Tax=float(out[0, 6]),
                             blk_x=float(out[0, 14]), alive=float(out[0, 12])))
            if phase == "drive" and x >= x_hi:
                break
        return rows, ys, nsub

    rows_ps, ys_p, nsub_p = patch_run(False)
    p_static = tb.steady(rows_ps, "load", "sink")[0]
    p_fy = tb.steady(rows_ps, "load", "Fy")[0] / W
    rows_pd, _, _ = patch_run(True)
    p_dp = tb.window(rows_pd, "Fx", x_lo, x_hi)[0] / W
    p_sink = tb.window(rows_pd, "sink", x_lo, x_hi)[0]
    p_fy_sl = tb.window(rows_pd, "Fy", x_lo, x_hi)[0] / W
    p_T = tb.window(rows_pd, "Tz", x_lo, x_hi)[0]
    shifts = len(set(r["blk_x"] for r in rows_pd))
    print(f"  patch: nsub={nsub_p}  static {p_static * 1000:.1f} mm Fy/W {p_fy:.3f} | s=0.3: DP/W {p_dp:+.3f} "
          f"sink {p_sink * 1000:.1f} mm Fy/W {p_fy_sl:.3f} T {p_T:.0f}  ({shifts} block positions)")
    res = [dict(quantity="static sinkage mm (4.5 kN)", bin=bin_static * 1000, patch=p_static * 1000),
           dict(quantity="static Fy/W", bin=bin_fy, patch=p_fy),
           dict(quantity="s=0.3 DP/W", bin=bin_dp, patch=p_dp),
           dict(quantity="s=0.3 sinkage mm", bin=bin_sink * 1000, patch=p_sink * 1000),
           dict(quantity="s=0.3 Fy/W", bin=bin_fy_sl, patch=p_fy_sl),
           dict(quantity="s=0.3 T N m (soil moment about +z)", bin=bin_T, patch=p_T)]
    for r in res:
        r["diff_pct"] = 100.0 * (r["patch"] - r["bin"]) / abs(r["bin"]) if r["bin"] else float("nan")
        print(f"    {r['quantity']:38s} bin {r['bin']:9.3f}  patch {r['patch']:9.3f}  {r['diff_pct']:+.1f} %")
    write_csv(os.path.join(OUT, "g2_parity.csv"), res)
    write_csv(os.path.join(OUT, "g2_patch_slip_run.csv"), rows_pd)
    write_csv(os.path.join(OUT, "g2_bin_slip_run.csv"), rows_sl)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for rows, lab in ((rows_sl, "static bin 3.2 m"), (rows_pd, "moving window 1.0 m")):
            d = [r for r in rows if r["phase"] == "drive"]
            ax[0].plot([r["x"] for r in d], [1000 * r["sink"] for r in d], label=lab)
            ax[1].plot([r["x"] for r in d], [r["Fx"] / W for r in d], label=lab, lw=0.8)
        for a in ax:
            a.axvspan(x_lo, x_hi, color="0.9")
            a.set_xlabel("wheel x (m)")
            a.legend()
        ax[0].set_ylabel("sinkage (mm)")
        ax[1].set_ylabel("DP/W (frame mean)")
        fig.suptitle("G2: s = 0.3, W = 4.5 kN, h = 40 mm, 4 ppc, E = 3 MPa (grey = averaging window)")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "g2_parity.png"), dpi=110)
    except Exception as e:  # noqa: BLE001
        print("plot failed", e)
    return res


# ------------------------------------------------------------------------------ G3 (+ G4)
def profile_particles(x, xs, zc, pd, zbins):
    """MPM free surface along z at x = xs: at each bin centre, the top of the
    particles within a disc of radius pd (+ half a spacing). At 4 ppc a plain
    20 mm bin often holds no top-layer particle, so a disc is the robust read."""
    band = x[np.abs(x[:, 0] - xs) < pd]
    zm = 0.5 * (zbins[1:] + zbins[:-1])
    prof = np.full(len(zm), np.nan)
    for i, z in enumerate(zm):
        m = (band[:, 0] - xs) ** 2 + (band[:, 2] - z) ** 2 < pd * pd
        if m.any():
            prof[i] = band[m, 1].max() + 0.5 * pd
    return prof


def part_g3():
    render = "--no-render" not in sys.argv
    sim = make_car_sim(hf=((-2.0, -3.0), (40.0, 6.0)))
    v = 3.0
    car = Car(sim, x0=3.0, z0=0.0, v=v, slip=0.1)
    nsub = sim.nsub_for(v / 0.9)
    bring_up(car, v, nsub)
    k_front, k_rear = 0, 2                     # left side: FL then RL on the same track
    xs = 3.0 + BASE + 2.0                      # the cross-section, 2 m ahead of FL's start
    zc = TRACK / 2
    xs_h, zs_h = sim.hf_coords()
    jlo, jhi = np.searchsorted(zs_h, zc - 0.26), np.searchsorted(zs_h, zc + 0.26)
    zb = np.concatenate([zs_h[jlo:jhi] - 0.5 * sim.hf_dx, [zs_h[jhi - 1] + 0.5 * sim.hf_dx]])
    zmid = 0.5 * (zb[1:] + zb[:-1])
    ci = int(np.argmin(np.abs(xs_h - xs)))
    got = {}
    rend = Render(sim, car) if render else None
    frames = 0
    while frames < 400:
        out = car.step(nsub, W_LOAD)
        frames += 1
        lo = sim.block_interior(k_front)
        if lo[0] < xs < lo[1] and out[k_front, 8] > xs + 0.1:
            got["mpm_front_before_retire"] = profile_particles(sim.particles(k_front), xs, zc, sim.pd, zb)
        elif "mpm_front_before_retire" in got and "hf_after_front" not in got and lo[0] > xs:
            got["hf_after_front"] = sim.heightfield()[ci, jlo:jhi].copy()
        lr = sim.block_interior(k_rear)
        if "hf_after_front" in got and "mpm_rear_seeded" not in got and lr[1] > xs + 2 * sim.pd:
            got["mpm_rear_seeded"] = profile_particles(sim.particles(k_rear), xs, zc, sim.pd, zb)
            got["rear_seed_frame_x"] = lr[1]
        if "mpm_rear_seeded" in got and lr[0] < xs < lr[1] and out[k_rear, 8] > xs + 0.1:
            got["mpm_rear_before_retire"] = profile_particles(sim.particles(k_rear), xs, zc, sim.pd, zb)
        if "mpm_rear_before_retire" in got and "hf_after_rear" not in got and lr[0] > xs:
            got["hf_after_rear"] = sim.heightfield()[ci, jlo:jhi].copy()
        if rend is not None and frames % 2 == 0 and frames <= 240:
            rend.frame(out)
        if "hf_after_rear" in got and (rend is None or frames > 240):
            break
    flat = 0.3
    a, b = got["mpm_front_before_retire"], got["hf_after_front"]
    m = np.isfinite(a)
    err = (b[m] - a[m]) * 1000
    c = got["mpm_rear_seeded"]
    mc = np.isfinite(c)
    err_seed = (c[mc] - b[mc]) * 1000
    cen = np.abs(zmid - zc) < 0.06
    res = dict(xs=xs, rut_depth_mpm_mm=(flat - np.nanmin(a)) * 1000, rut_depth_hf_mm=(flat - b.min()) * 1000,
               rut_centre_mpm_mm=float((flat - np.nanmean(a[cen])) * 1000),
               rut_centre_hf_mm=float((flat - np.mean(b[cen])) * 1000),
               writeback_rms_mm=float(np.sqrt(np.mean(err ** 2))), writeback_bias_mm=float(err.mean()),
               writeback_median_abs_mm=float(np.median(np.abs(err))),
               writeback_maxabs_mm=float(np.abs(err).max()),
               seeded_vs_hf_rms_mm=float(np.sqrt(np.mean(err_seed ** 2))),
               seeded_vs_hf_bias_mm=float(err_seed.mean()),
               rear_seeded_rut_depth_mm=(flat - np.nanmin(c)) * 1000,
               second_pass_rut_depth_hf_mm=(flat - got["hf_after_rear"].min()) * 1000 if "hf_after_rear" in got else float("nan"))
    for k, val in res.items():
        print(f"    {k}: {val:.2f}")
    rows = [dict(z=float(zmid[i]), mpm_front_before_retire=float(a[i]), hf_after_front=float(b[i]),
                 mpm_rear_seeded=float(c[i]),
                 mpm_rear_before_retire=float(got.get("mpm_rear_before_retire", np.full_like(a, np.nan))[i]),
                 hf_after_rear=float(got.get("hf_after_rear", np.full_like(a, np.nan))[i]))
            for i in range(len(zmid))]
    write_csv(os.path.join(OUT, "g3_profiles.csv"), rows)
    write_csv(os.path.join(OUT, "g3_summary.csv"), [res])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.2))
        zz = (zmid - zc) * 1000
        ax.axhline(flat * 1000, color="0.6", ls=":", label="virgin surface")
        ax.plot(zz, a * 1000, "o-", ms=3, label="FL patch MPM surface, just before retirement")
        ax.plot(zz, b * 1000, "-", lw=2, label="heightfield after write-back")
        ax.plot(zz, c * 1000, "s--", ms=3, label="RL patch MPM surface, just after seeding")
        if "hf_after_rear" in got:
            ax.plot(zz, got["hf_after_rear"] * 1000, "-", label="heightfield after the second pass")
        ax.set_xlabel("z across the track (mm, wheel centre = 0)")
        ax.set_ylabel("surface height (mm)")
        ax.set_title(f"G3: rut cross-section at x = {xs:.2f} m (3 m/s, W = 4.5 kN)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "g3_rut_profile.png"), dpi=110)
    except Exception as e:  # noqa: BLE001
        print("plot failed", e)
    if rend is not None:
        rend.finish()
    return res


class Render:
    """Headless Vulkan: heightfield mesh + the four patches' MC surfaces + wheels."""

    def __init__(self, sim, car, width=1280, height=720):
        import threepp as tp
        saved = sys.argv[:]
        sys.argv = [saved[0]]
        import warp_wheel_testbed as tb
        sys.argv = saved
        self.tp, self.sim, self.car, self.W, self.H = tp, sim, car, width, height
        self.fdir = os.path.join(OUT, "frames")
        os.makedirs(self.fdir, exist_ok=True)
        for f in os.listdir(self.fdir):
            os.remove(os.path.join(self.fdir, f))
        canvas = tp.Canvas("patch gate", width=width, height=height, vsync=False, headless=True)
        self.canvas = canvas
        r = tp.VulkanRenderer(canvas)
        r.tone_mapping = tp.ToneMapping.ACESFilmic
        r.tone_mapping_exposure = 0.9
        self.renderer = r
        sc = tp.Scene()
        sc.background = 0x8d9aa6
        sc.add(tp.HemisphereLight(0xd6e0ea, 0x4a4238, 0.9))
        sun = tp.DirectionalLight(0xfff1dc, 3.2)
        sun.position.set(-3.0, 6.0, 4.0)
        sun.cast_shadow = True
        sun.set_shadow_frustum(-8.0, 8.0, 8.0, -8.0)
        sun.set_shadow_bias(-0.0005)
        sc.add(sun)
        self.sun = sun
        self.scene = sc

        def sand():
            m = tp.MeshPhysicalMaterial()
            m.color = 0x9c8a6c
            m.roughness = 0.95
            m.specular_intensity = 0.2
            m.side = tp.Side.Double
            return m
        # Heightfield mesh (a window of the far field around the car).
        self.hx, self.hz = 520, 150                 # vertices; 20 mm spacing = 10.4 x 3 m
        self.hgeo = tp.PlaneGeometry(1.0, 1.0, self.hx - 1, self.hz - 1)
        hm = sand()
        hm.color = 0x8f7f63                          # a shade darker: the far field
        self.hmesh = tp.Mesh(self.hgeo, hm)
        self.hmesh.receive_shadow = True
        self.hmesh.cast_shadow = True
        self.hmesh.frustum_culled = False
        sc.add(self.hmesh)
        # Patch soil (marching-cubes soup, rebuilt every frame).
        cap = 300_000
        self.cap = cap
        geo = tp.BufferGeometry()
        geo.set_attribute("position", np.zeros((cap * 3, 3), np.float32))
        geo.set_attribute("normal", np.tile(np.float32([0, 1, 0]), (cap * 3, 1)))
        geo.set_draw_range(0, 3)
        self.sgeo = geo
        self.soil = tp.Mesh(geo, sand())
        self.soil.cast_shadow = True
        self.soil.receive_shadow = True
        self.soil.frustum_culled = False
        sc.add(self.soil)
        if hasattr(r, "set_stable_correspondence"):
            r.set_stable_correspondence(self.soil, False)
            r.set_stable_correspondence(self.hmesh, False)
        # Wheels + block outlines.
        self.wheels = []
        for _ in range(4):
            w = tb.build_wheel(tp, R_W, B_W)
            w.traverse(lambda o: setattr(o, "cast_shadow", True))
            sc.add(w)
            self.wheels.append(w)
        lm = tp.MeshBasicMaterial()
        lm.color = 0xffc233
        self.edges = []
        for _ in range(4):
            es = []
            for _ in range(4):
                e = tp.Mesh(tp.BoxGeometry(1.0, 1.0, 1.0), lm)
                sc.add(e)
                es.append(e)
            self.edges.append(es)
        self.cam = tp.PerspectiveCamera(40.0, width / height, 0.05, 80.0)
        self.k = 0
        self.ang = np.zeros(4)
        self.stage = None
        self.surf_cache = None

    def _update(self, out):
        tp, sim, car = self.tp, self.sim, self.car
        wpm = sim.wp
        from warp_common import DensitySurface
        # Heightfield window around the car, patch interiors pushed under the MPM soil.
        H = sim.heightfield()
        xs_h, zs_h = sim.hf_coords()
        cx = car.x + 0.5 * BASE
        i0 = int(np.clip(np.searchsorted(xs_h, cx - 6.4), 0, len(xs_h) - self.hx))
        j0 = int(np.clip(np.searchsorted(zs_h, car.z - 1.5), 0, len(zs_h) - self.hz))
        Hs = H[i0:i0 + self.hx, j0:j0 + self.hz].copy()
        X, Z = np.meshgrid(xs_h[i0:i0 + self.hx], zs_h[j0:j0 + self.hz], indexing="ij")
        for k in range(4):
            a, b, c, d = sim.block_interior(k)
            m = (X > a + 0.01) & (X < b - 0.01) & (Z > c + 0.01) & (Z < d - 0.01)
            Hs[m] -= 0.05
        gx, gz = np.gradient(Hs, sim.hf_dx)
        nrm = np.stack([-gx, np.ones_like(Hs), -gz], -1)
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        # PlaneGeometry order: row iy (plane y from +h/2 down), column ix.
        P = np.stack([X, Hs, Z], -1).transpose(1, 0, 2).reshape(-1, 3).astype(np.float32)
        N = nrm.transpose(1, 0, 2).reshape(-1, 3).astype(np.float32)
        self.hgeo.update_attribute("position", P)
        self.hgeo.update_attribute("normal", N)
        # Patch soil: live particles of all four patches -> MC.
        xp = sim.particles()
        cell = 1.15 * sim.pd
        lo = xp.min(0) - 4 * cell
        hi = xp.max(0) + 4 * cell
        lo[1] = -2 * cell
        dims = tuple(int((hi[i] - lo[i]) / cell) + 1 for i in range(3))
        key = dims
        if self.surf_cache is None or self.surf_cache[0] != key:
            self.surf_cache = (key, None)
        surf = DensitySurface(tuple(lo), cell, dims, sim.dev)
        if self.stage is None:
            self.stage = (wpm.zeros(self.cap * 3, dtype=wpm.vec3, device=sim.dev),
                          wpm.zeros(self.cap * 3, dtype=wpm.vec3, device=sim.dev))
        xw = wpm.array(xp.astype(np.float32), dtype=wpm.vec3, device=sim.dev)
        iso = 0.46 * (cell / sim.pd) ** 3
        nt = min(surf.build(xw, len(xp), iso), self.cap)
        if nt > 0:
            sp, sn = self.stage
            surf.expand(nt, sp, sn, sign=1.0, flip_winding=True, grain=0.55, grain_freq=70.0)
            self.sgeo.update_attribute("position", sp[:3 * nt].numpy())
            self.sgeo.update_attribute("normal", sn[:3 * nt].numpy())
        self.sgeo.set_draw_range(0, 3 * nt)
        # Wheels and block outlines.
        for k in range(4):
            w = self.wheels[k]
            p = out[k, 8:11]
            w.position.set(float(p[0]), float(p[1]), float(p[2]))
            self.ang[k] -= car.speed / (0.9 * R_W) * FRAME * 2
            w.rotation.y = -car.psi
            w.rotation.z = float(self.ang[k])
            a, b, c, d = sim.block_interior(k)
            y = sim.floor_y + sim.depth + 0.03
            es = self.edges[k]
            # Unit boxes scaled to each edge of the block interior.
            for e, (px, pz, sx, sz) in zip(es, [(0.5 * (a + b), c, b - a, 0.012), (0.5 * (a + b), d, b - a, 0.012),
                                                 (a, 0.5 * (c + d), 0.012, d - c), (b, 0.5 * (c + d), 0.012, d - c)]):
                e.position.set(px, y, pz)
                e.scale.set(sx, 0.012, sz)
        self.sun.position.set(car.x - 3.0, 6.0, car.z + 4.0)
        return nt

    def shoot(self, path, eye, tgt, reps=4):
        tp = self.tp
        self.cam.position.set(*eye)
        self.cam.look_at(tp.Vector3(*tgt))
        for _ in range(reps):
            self.renderer.save_frame(self.scene, self.cam, path)

    def frame(self, out):
        nt = self._update(out)
        car = self.car
        cx = car.x + 0.5 * BASE
        p = os.path.join(self.fdir, f"f{self.k:05d}.png")
        self.shoot(p, (cx - 4.6, 2.9, car.z + 2.6), (cx - 0.2, 0.2, car.z))
        self.k += 1
        self.last_out = out
        if self.k % 20 == 0:
            print(f"    render frame {self.k}  tris {nt}")

    def finish(self):
        import imageio_ffmpeg
        from warp_common import encode_png_sequence
        car = self.car
        self._update(self.last_out)
        cx = car.x + 0.5 * BASE
        stills = []
        for name, eye, tgt in [("behind_above", (cx - 4.6, 2.9, car.z + 2.6), (cx - 0.2, 0.2, car.z)),
                               ("top", (cx - 1.8, 5.5, car.z + 0.01), (cx - 1.8, 0.2, car.z)),
                               ("rut_low", (cx - 5.5, 1.2, car.z + 0.8), (cx, 0.25, car.z + 0.8))]:
            p = os.path.join(OUT, f"g4_{name}.png")
            self.shoot(p, eye, tgt, reps=6)
            stills.append(p)
        mp4 = os.path.join(OUT, "g4_patches.mp4")
        if self.k > 10:
            encode_png_sequence(os.path.join(self.fdir, "f%05d.png"), mp4, 30, crf=20,
                                ffmpeg=imageio_ffmpeg.get_ffmpeg_exe())
        print("render:", stills, mp4)


if __name__ == "__main__":
    part = arg("--part", "all")
    if part in ("g1", "all"):
        part_g1()
    if part in ("g2", "all"):
        part_g2()
    if part in ("g3", "all"):
        part_g3()
