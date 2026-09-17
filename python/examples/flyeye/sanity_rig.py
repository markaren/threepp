"""Phase 1 readout sign check: two live eyes on a scripted vehicle in a textured room.

    py -3.14 python/examples/flyeye/sanity_rig.py [--flush 3 --out C:/dev/_flyeye/sanity]

A vehicle Object3D at the centre of a closed box room (half side --room, every wall,
floor and ceiling an unlit multi-scale noise texture) carries two 403 px, 90 deg eye
cameras yawed +45 and -45 deg (vehicle -Z forward). At 100 Hz with sim time pinned it
rests 1 s, then rotates about body axes at --rate deg/s, 1 s each with 0.5 s rests:
yaw left (+wy), yaw right, pitch up (+wx), pitch down, roll +wz, roll -wz; then it flies
forward toward the -Z wall and stops --stop m short of it. Rates and speed ramp over
0.1 s (0.2 s for the approach).

Per frame: EyeView x2 -> OpticLobe batched (fade_in on the first frame) -> MotionField
-> RotationReadout (matched and lstsq) -> Looming (default, and opponent=False). A third
rotation line runs the renderer's own motion AOV (sampled at the column centres, turned
into angular flow in rad/s) through the same RotationReadout: it checks the rig geometry
and the template signs without the circuit. The AOV reads zero at flush > 1 (the extra
GPU frames repeat the scene state), so run --flush 1 for that line.

Writes <out>/sanity.npz, <out>/report.json, eye snapshots, and
figures/phase1_readout_sanity.png.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore", message=".*[Ss]parse.*")

import threepp as tp  # noqa: E402

from flyeye.eye import EyeView, configure_sensor_renderer  # noqa: E402
from flyeye.lattice import HexLattice  # noqa: E402
from flyeye.optic_lobe import OpticLobe  # noqa: E402
from flyeye.readouts import Looming, MotionField, RotationReadout  # noqa: E402
from flyeye.truth import AovFlow  # noqa: E402
from flyeye.video import VideoOut  # noqa: E402

SIZE, FOV, DT = 403, 90.0, 0.01
YAWS = (45.0, -45.0)
FIGURE = Path(__file__).resolve().parent / "figures" / "phase1_readout_sanity.png"


def noise_png(path, seed, n=1024, sigmas=(10, 30, 90)):
    """High-contrast grey noise with blobs at three scales, mean about 128."""
    from PIL import Image

    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(n)
    k2 = f[:, None] ** 2 + f[None, :] ** 2
    acc = np.zeros((n, n))
    for s in sigmas:
        band = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * np.exp(-2 * (np.pi * s) ** 2 * k2)))
        acc += (band - band.mean()) / band.std()
    g = 0.5 + 0.42 * np.tanh(1.2 * acc / acc.std())
    Image.fromarray((np.clip(g, 0, 1) * 255).astype(np.uint8)).convert("RGB").save(path)
    return path


def build_room(half, tex_dir):
    """Six inward-facing unlit textured planes, one noise seed each."""
    scene = tp.Scene()
    scene.background = tp.Color(0.5, 0.5, 0.5)
    walls = (  # position, rotation (x, y), all normals point into the room
        ((0, 0, -half), (0, 0)), ((0, 0, half), (0, math.pi)),
        ((-half, 0, 0), (0, math.pi / 2)), ((half, 0, 0), (0, -math.pi / 2)),
        ((0, -half, 0), (-math.pi / 2, 0)), ((0, half, 0), (math.pi / 2, 0)),
    )
    for i, (p, (rx, ry)) in enumerate(walls):
        tex = tp.TextureLoader().load(str(noise_png(tex_dir / f"wall{i}.png", seed=i)), tp.ColorSpace.SRGB)
        tex.anisotropy = 8
        m = tp.MeshBasicMaterial()
        m.map = tex
        m.side = tp.Side.Double
        w = tp.Mesh(tp.PlaneGeometry(2 * half, 2 * half), m)
        w.position.set(*p)
        w.rotation.x, w.rotation.y = rx, ry
        scene.add(w)
    return scene


def ramp(t, T, r):
    """Trapezoid 0 -> 1 -> 0 over [0, T] with ramps of r seconds."""
    return float(np.clip(min(t, T - t) / r, 0.0, 1.0)) if 0 <= t <= T else 0.0


def schedule(rate_deg, room, stop, speed):
    """Per-frame body angular velocity (T, 3) rad/s, forward speed (T,), and labelled segments."""
    W = math.radians(rate_deg)
    segs = [("rest", 1.0, None)]
    for name, w in (("yaw left", (0, W, 0)), ("yaw right", (0, -W, 0)), ("pitch up", (W, 0, 0)),
                    ("pitch down", (-W, 0, 0)), ("roll +z", (0, 0, W)), ("roll -z", (0, 0, -W))):
        segs += [(name, 1.0, np.array(w)), ("rest", 0.5, None)]
    t_app = (room - stop) / speed + 0.2  # trapezoid area = speed * (T - 0.2)
    segs += [("approach", t_app, None), ("rest", 0.3, None)]
    w_all, v_all, labels, k = [], [], [], 0
    for name, T, w in segs:
        n = int(round(T / DT))
        for i in range(n):
            t = (i + 0.5) * DT
            if name == "approach":
                w_all.append(np.zeros(3))
                v_all.append(speed * ramp(t, T, 0.2))
            else:
                w_all.append(np.zeros(3) if w is None else w * ramp(t, T, 0.1))
                v_all.append(0.0)
        labels.append((name, k, k + n))
        k += n
    return np.array(w_all), np.array(v_all), labels


def rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    x, y, z = axis
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def quat(R):
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return x, y, z, w


def to_rgb(color):
    return color[..., [2, 1, 0]].cpu().numpy()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flush", type=int, default=3)
    ap.add_argument("--rate", type=float, default=75.0, help="rotation rate, deg/s")
    ap.add_argument("--room", type=float, default=6.0, help="room half side, m")
    ap.add_argument("--speed", type=float, default=3.0, help="approach speed, m/s")
    ap.add_argument("--stop", type=float, default=1.2, help="distance to the wall at the end, m")
    ap.add_argument("--out", type=Path, default=Path(r"C:\dev\_flyeye\sanity"))
    ap.add_argument("--figure", type=Path, default=FIGURE)
    ap.add_argument("--figure-only", action="store_true")
    ap.add_argument("--video", type=Path, default=None, help="also write an mp4 of the run (renders, receptors, field)")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.figure_only:
        run(args)
    figure(args)


def run(args):
    from PIL import Image

    t0 = time.perf_counter()
    canvas = tp.Canvas("flyeye sanity", width=128, height=128, headless=True, vsync=False)
    r = tp.VulkanRenderer(canvas, flush_frames=args.flush)
    settings = configure_sensor_renderer(r)
    for name in ("deferred_ao", "probe_gi", "volumetric_fog", "shadow_map_enabled"):
        try:
            setattr(r, name, False)
        except Exception:
            pass
    scene = build_room(args.room, args.out)
    main_cam = tp.PerspectiveCamera(60, 1.0, 0.05, 100)
    vehicle = tp.Group()
    scene.add(vehicle)
    cams = []
    for yaw in YAWS:
        c = tp.PerspectiveCamera(FOV, 1.0, 0.05, 100)
        c.rotation.y = math.radians(yaw)
        vehicle.add(c)
        cams.append(c)
    main_cam.position.set(0, 0, 0.5 * args.room)
    vehicle.add(main_cam)

    t = [0.0]

    def render():
        t[0] += DT
        r.sim_time = t[0]
        r.render(scene, main_cam)

    render()
    lat = HexLattice()
    eyes = [EyeView(r, c, SIZE) for c in cams]
    render()
    for e in eyes:
        e.arm()
    for _ in range(30):  # TAA settles at rest
        render()

    R_eyes = np.stack([rot((0, 1, 0), math.radians(y)) for y in YAWS])
    lobe = OpticLobe(device="cuda")
    motion = MotionField(lobe)
    ro_m = RotationReadout(SIZE, FOV, R_eyes, method="matched")
    ro_l = RotationReadout(SIZE, FOV, R_eyes, method="lstsq")
    loom = Looming(SIZE, FOV, R_eyes)
    loom_col = Looming(SIZE, FOV, R_eyes, opponent=False)
    aov = AovFlow(lat, DT, SIZE, FOV)
    video = VideoOut(args.video, lat, len(eyes), size=SIZE, yaws=YAWS) if args.video else None

    w_true, speed, labels = schedule(args.rate, args.room, args.stop, args.speed)
    n = len(w_true)
    R = np.eye(3)
    pos = np.zeros(3)
    rec = {k: [] for k in ("matched", "lstsq", "aov", "loom", "loom_col", "loom_centre", "field", "aov_field")}
    dist = np.zeros(n)
    snaps = {"rest": 90, "yaw_left": 160, "approach_end": labels[-2][2] - 5}
    for k in range(n):
        R = R @ rot(np.array(w_true[k]) / (np.linalg.norm(w_true[k]) or 1), np.linalg.norm(w_true[k]) * DT)
        pos = pos + R @ np.array([0, 0, -speed[k] * DT])
        vehicle.quaternion.set(*quat(R))
        vehicle.position.set(*pos)
        dist[k] = args.room + pos[2]
        render()
        x = torch.stack([e.receptors() for e in eyes])
        if k == 0:
            lobe.fade_in(x, DT)
        v = lobe.step(x, DT)
        f = motion(v)
        lo = loom(f)
        a = torch.stack([aov(e.motion) for e in eyes]).float()
        rec["matched"].append(ro_m(f))
        rec["lstsq"].append(ro_l(f))
        rec["aov"].append(ro_m(a))
        rec["loom"].append(lo["value"])
        rec["loom_col"].append(loom_col(f)["value"])
        rec["loom_centre"].append(lo["centre"])
        rec["field"].append(f.half())
        rec["aov_field"].append(a.half())
        for tag, kk in snaps.items():
            if k == kk:
                row = np.concatenate([to_rgb(e.color) for e in eyes], axis=1)
                Image.fromarray(np.ascontiguousarray(row)).save(args.out / f"eyes_{tag}.png")
        if video:
            seg = next(name for name, a, b in labels if a <= k < b)
            est, tw = rec["matched"][-1].tolist(), w_true[k]
            video.write([e.color for e in eyes], x, f, f"t {k * DT:5.2f} s   {seg}   flush {args.flush}",
                        f"circuit w ({est[0]:+.3f}, {est[1]:+.3f}, {est[2]:+.3f}) a.u.   "
                        f"true w ({tw[0]:+.2f}, {tw[1]:+.2f}, {tw[2]:+.2f}) rad/s   looming {float(lo['value']):.3f}")
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    if video:
        video.close()
    for e in eyes:
        e.close()

    data = {k: torch.stack(v).cpu().numpy() for k, v in rec.items()}
    inv_tau = np.where(speed > 0, speed / np.maximum(dist, 1e-6), 0.0)
    np.savez(args.out / "sanity.npz", w_true=w_true, speed=speed, dist=dist, inv_tau=inv_tau,
             labels=np.array([f"{a}|{b}|{c}" for a, b, c in labels]), **data)
    json.dump(dict(settings={k: str(v) for k, v in settings.items()}, flush=args.flush, rate_deg=args.rate,
                   room=args.room, speed=args.speed, stop=args.stop, frames=n, wall_s=wall),
              open(args.out / "run.json", "w"), indent=1)
    print(f"{n} frames in {wall:.1f} s")


AXES = ("wx (pitch)", "wy (yaw)", "wz (roll)")
LOOM_KEYS = ("loom", "loom_col")  # Looming default (opponent) and opponent=False


def summarize(d, labels):
    """Per manoeuvre: mean of each readout over the segment minus its first 0.25 s."""
    w, inv_tau = d["w_true"], d["inv_tau"]
    rows, lat = [], int(0.25 / DT)
    rest0 = slice(labels[0][1] + 50, labels[0][2])
    for name, a, b in labels:
        if name == "rest":
            continue
        s = slice(a + lat, b)
        row = dict(name=name)
        if name == "approach":
            s = slice(a, b)
            row["inv_tau_range"] = [float(inv_tau[s].min()), float(inv_tau[s].max())]
            for key in LOOM_KEYS:
                x = d[key]
                row[key] = dict(corr_inv_tau=float(np.corrcoef(x[s], inv_tau[s])[0, 1]), peak=float(x[s].max()),
                                first_0p3s=float(x[a:a + 30].mean()), last_0p3s_before_stop=float(x[b - 50:b - 20].mean()))
            for m in ("matched", "lstsq"):
                row[m] = d[m][s].mean(0).round(4).tolist()
        else:
            ax = int(np.argmax(np.abs(w[s]).mean(0)))
            row["axis"] = AXES[ax]
            row["true_mean_rad_s"] = float(w[s, ax].mean())
            for m in ("matched", "lstsq", "aov"):
                mean = d[m][s].mean(0)
                row[m] = mean.round(4).tolist()
                row[m + "_rest_subtracted"] = (mean - d[m][rest0].mean(0)).round(4).tolist()
                row[m + "_sign_ok"] = bool(np.sign(mean[ax]) == np.sign(row["true_mean_rad_s"]))
                off = [i for i in range(3) if i != ax]
                row[m + "_offaxis_ratio"] = float(np.abs(mean[off]).max() / max(abs(mean[ax]), 1e-9))
            for key in LOOM_KEYS:
                row[key] = dict(mean=float(d[key][s].mean()), peak=float(d[key][s].max()))
        rows.append(row)
    return rows


def figure(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(args.out / "sanity.npz")
    d = {k: z[k] for k in z.files}
    labels = [(s.split("|")[0], int(s.split("|")[1]), int(s.split("|")[2])) for s in d["labels"]]
    rows = summarize(d, labels)
    run_info = json.load(open(args.out / "run.json"))
    rest0 = slice(labels[0][1] + 50, labels[0][2])
    app = [r for r in rows if r["name"] == "approach"][0]
    rot_rows = [r for r in rows if r["name"] != "approach"]
    report = dict(run=run_info, manoeuvres=rows, rest_mean=dict(
        matched=d["matched"][rest0].mean(0).round(4).tolist(), lstsq=d["lstsq"][rest0].mean(0).round(4).tolist(),
        aov=d["aov"][rest0].mean(0).round(4).tolist(), **{k: float(d[k][rest0].mean()) for k in LOOM_KEYS}))
    report["sign_pass"] = {m: all(r[m + "_sign_ok"] for r in rot_rows) for m in ("matched", "lstsq", "aov")}
    for k in LOOM_KEYS:  # rises 2x over the approach, tracks 1/tau, and ends 2x above any rotation's mean
        rot_max = max(r[k]["mean"] for r in rot_rows)
        a_ = app[k]
        report[k + "_pass"] = bool(a_["last_0p3s_before_stop"] > 2 * a_["first_0p3s"] and a_["corr_inv_tau"] > 0.5
                                   and a_["last_0p3s_before_stop"] > 2 * rot_max)
        report[k + "_rotation_max_mean"] = rot_max
    json.dump(report, open(args.out / "report.json", "w"), indent=1)
    print(json.dumps(report, indent=1))

    t = np.arange(len(d["w_true"])) * DT
    fig = plt.figure(figsize=(15, 10.5))
    gs = fig.add_gridspec(4, 4, hspace=0.5, wspace=0.5)
    colors = dict(matched="#1f77b4", lstsq="#ff7f0e", true="k", aov="#999999")
    for ax_i in range(3):
        ax = fig.add_subplot(gs[ax_i, :3])
        for name, a, b in labels:
            if name != "rest":
                ax.axvspan(a * DT, b * DT, color="#eeeeee" if name != "approach" else "#fbe9d0", zorder=0)
                if ax_i == 0:
                    ax.text((a + b) / 2 * DT, 1.02, name, transform=ax.get_xaxis_transform(), ha="center", fontsize=8)
        ym = d["matched"][:, ax_i]
        s = np.abs(ym).max() / max(np.abs(d["w_true"][:, ax_i]).max(), 1e-9)
        ax.plot(t, d["matched"][:, ax_i], color=colors["matched"], lw=1.2, label="circuit, matched filter")
        ax.plot(t, d["lstsq"][:, ax_i], color=colors["lstsq"], lw=0.9, alpha=0.8, label="circuit, lstsq")
        ax.plot(t, d["aov"][:, ax_i] * s, color=colors["aov"], lw=0.9, label=f"motion AOV via templates (rad/s) x {s:.2f}, reads 0 at flush > 1")
        ax.plot(t, d["w_true"][:, ax_i] * s, color=colors["true"], lw=1.4, ls="--",
                label=f"true (rad/s) x {s:.2f} = max|matched| / max|true|")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_ylabel(AXES[ax_i])
        ax.legend(loc="lower left", fontsize=7, ncol=2)
        ax.set_xlim(0, t[-1])
    ax = fig.add_subplot(gs[3, :3])
    for name, a, b in labels:
        if name != "rest":
            ax.axvspan(a * DT, b * DT, color="#eeeeee" if name != "approach" else "#fbe9d0", zorder=0)
    ax.plot(t, d["loom"], color="#2ca02c", lw=1.2, label="Looming (default: quadrant mean, then rectify)")
    ax.plot(t, d["loom_col"], color="#9467bd", lw=0.9, alpha=0.8, label="Looming opponent=False (rectify per column)")
    ax.set_ylabel("looming")
    ax.set_xlabel("time (s)")
    ax2 = ax.twinx()
    ax2.plot(t, d["inv_tau"], color="k", ls="--", lw=1.2, label="1/tau = speed / distance to wall (1/s)")
    ax2.set_ylabel("1/tau (1/s)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=7)
    ax.set_xlim(0, t[-1])

    ax = fig.add_subplot(gs[3, 3])
    a, b = [(x[1], x[2]) for x in labels if x[0] == "approach"][0]
    sc = ax.scatter(d["inv_tau"][a:b], d["loom"][a:b], c=t[a:b], s=6, cmap="viridis")
    ax.scatter(d["inv_tau"][a:b], d["loom_col"][a:b], c=t[a:b], s=6, cmap="viridis", marker="x", alpha=0.4)
    ax.set_xlabel("1/tau (1/s)")
    ax.set_ylabel("looming")
    ax.set_title(f"approach: dots default r = {app['loom']['corr_inv_tau']:.2f},\n"
                 f"crosses opponent=False r = {app['loom_col']['corr_inv_tau']:.2f}", fontsize=8)
    fig.colorbar(sc, ax=ax, label="t (s)")

    ax = fig.add_subplot(gs[0:3, 3])
    names = [r["name"] for r in rot_rows]
    y = np.arange(len(names))
    for j, m in enumerate(("matched", "lstsq", "aov")):
        vals = []
        for r in rot_rows:
            i = AXES.index(r["axis"])
            vals.append(np.sign(r["true_mean_rad_s"]) * r[m][i] / max(abs(r[m][i]) for r in rot_rows
                                                                    if AXES.index(r["axis"]) == i))
        ax.barh(y + (j - 1) * 0.27, vals, height=0.25, color=colors[m], label=m)
    ax.axvline(0, color="k", lw=0.6)
    ax.set_yticks(y, names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("readout x sign(true) on the dominant axis, per-axis normalised; > 0 = right sign", fontsize=7)
    ax.legend(fontsize=7, loc="lower left")
    ax.set_title("sign check per manoeuvre", fontsize=9)
    fig.suptitle(f"flyeye Phase 1 readout sanity: two 403 px 90 deg eyes at +-45 deg yaw, flush {run_info['flush']}, "
                 f"{run_info['rate_deg']:.0f} deg/s rotations, {run_info['speed']:.1f} m/s approach "
                 f"to {run_info['stop']:.1f} m", fontsize=11)
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=100)
    print("wrote", args.figure)


if __name__ == "__main__":
    main()
