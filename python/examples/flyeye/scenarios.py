"""Phase 2 scenarios: two live eyes on a scripted vehicle over the Geiranger fjord wall.

    py -3.14 python/examples/flyeye/scenarios.py --scout          # tone mapping, sky motion, stills
    py -3.14 python/examples/flyeye/scenarios.py [--only yaw,approach] [--lights bright,dim,dark]

Scene: geodata/geiranger (1 m Kartverket DTM, cliff shell, forest), a flat dark water
plane at sea level, a procedural sky (background and image light) and one sun from the
south-west at 35 deg elevation. The corridor is a shelf on the south fjord wall at about
230 m: it runs north-east to the base of a wall that rises about 190 m over 70 m, with
the mountainside rising on the right and falling to the fjord on the left.

Lighting levels scale the sun (6.0 at bright) and the sky (gain 1.4) together (bright 1,
dim 0.3, dark 0.1); exposure and tone mapping stay fixed. Tone mapping AgX, chosen by
--scout: at bright, no pixel of the eye views clips with AgX or Neutral, while
NoToneMapping has a channel at 255 in 0.4-1.4 % of pixels (sky near the horizon glow).
AgX keeps brighter terrain mid-tones than Neutral (luma p50 71-93 vs 42-61), at the cost
of a darker toe at dark (25-34 % of pixels at luma <= 5).

Rig (sanity_rig.py geometry): a vehicle Object3D (-Z forward, +Y up) with two 403 px,
90 deg eyes yawed +45 and -45 deg, 100 Hz, flush_frames 1, sim time pinned. Scripted body
rates and velocities integrate exactly as sanity_rig does: R <- R rot(w dt), p <- p + R v dt.

Scenarios, each preceded by 1 s of rest (the circuit fades in on frame 0):
  straight_3, straight_10, straight_30  level flight toward the wall at 3, 10, 30 m/s
  yaw, pitch, roll                      hovering, +-30, +-90, +-180 deg/s segments
  approach                              15 m/s toward the wall, 150 m to 18 m, 45 m above the ground

Per run: <out>/<scenario>_<light>.npz (+ .json), <renders>/<scenario>_<light>.mp4 and stills.

npz keys, T frames, B = 2 eyes (0 = yaw +45, left; 1 = yaw -45, right), 721 columns in
lattice order; frame k is the render after pose k was set (its motion is pose k-1 -> k):
  t (T,) s; seg (T,) index into labels; labels "name|first|end" (end exclusive)
  pos_world (T, 3) m; R_world (T, 3, 3) world-from-vehicle; w_body (T, 3) rad/s and
    v_body (T, 3) m/s in the vehicle frame (x right, y up, -z forward), exact; speed (T,)
  receptors (T, B, 721) f16; t4t5_raw (T, B, 8, 721) f16 (MotionField.raw, T4a-d T5a-d);
    field (T, B, 2, 721) f16 (MotionField, unit gains, x right y up)
  flow_box, flow_centre (T, B, 2, 721) f32 rad/s (truth.AovFlow, sky = -w x ray)
  contrast, sky_frac (T, B, 721) f16 (truth.Footprint)
  rot_matched, rot_lstsq (T, 3) circuit; rot_truth_box, rot_truth_centre (T, 3) rad/s
  loom (T,), loom_centre (T, 3) circuit; loom_truth (T,) Looming on flow_box
  range_depth, range_depth_min (T,) m (truth.ForwardRange); range_depth_rate (T,) m/s;
    tau_depth (T,) s; range_geom, tau_geom (T,) straight and approach only (else NaN / inf);
    nearest (T,) m; baking, tiles (T,) terrain streaming state
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

from flyeye.eye import EyeView, bgra_luma_u8, configure_sensor_renderer  # noqa: E402
from flyeye.lattice import HexLattice  # noqa: E402
from flyeye.optic_lobe import OpticLobe  # noqa: E402
from flyeye.readouts import Looming, MotionField, RotationReadout  # noqa: E402
from flyeye.sanity_rig import DT, FOV, SIZE, YAWS, quat, ramp, rot  # noqa: E402
from flyeye.truth import AovFlow, Footprint, ForwardRange, rotation_flow, smooth_tau  # noqa: E402
from flyeye.video import VideoOut  # noqa: E402

PACK = Path(__file__).resolve().parents[3] / "geodata" / "geiranger"
FACE = np.array([-132.0, 844.0])  # (x, z) base of the wall
HEADING = np.array([1.0, -1.0]) / math.sqrt(2)  # (x, z) toward the wall, north-east
SUN_ELEV = math.radians(35.0)
SUN_DIR = np.array([-math.cos(SUN_ELEV) / math.sqrt(2), math.sin(SUN_ELEV), math.cos(SUN_ELEV) / math.sqrt(2)])
SUN_INTENSITY = 6.0
SKY_GAIN = 1.4
LIGHTS = dict(bright=1.0, dim=0.3, dark=0.1)
TONE_MAPPING = "AgX"
NEAR, FAR = 0.5, 6000.0
AGL = 40.0  # hover and straight flight, above the highest ground under the path
CLEAR = 45.0  # approach: above the highest ground under the path (trees are up to ~30 m)
SCENARIOS = ("straight_3", "straight_10", "straight_30", "yaw", "pitch", "roll", "approach")
OUT = Path(r"C:\dev\_flyeye\phase2")
RENDERS = Path(r"C:\dev\_flyeye\renders\phase2")


def sky_rgba(sun_dir, level=1.0, w=512, h=256):
    """Procedural float equirect (warp_common.sky_env's gradient, a land haze below the
    horizon), times level. Background and image light."""
    s = np.asarray(sun_dir, np.float32)
    s = s / np.linalg.norm(s)
    elev = ((np.arange(h, dtype=np.float32) + 0.5) / h - 0.5) * math.pi
    az = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) * 2.0 * math.pi
    d = np.stack([np.cos(elev)[:, None] * np.cos(az)[None, :], np.broadcast_to(np.sin(elev)[:, None], (h, w)),
                  np.cos(elev)[:, None] * np.sin(az)[None, :]], axis=-1)
    y = d[..., 1:2]
    up, down = np.clip(y, 0, 1) ** 0.40, np.clip(-y, 0, 1) ** 0.60
    col = np.where(y >= 0, np.float32([0.60, 0.71, 0.90]) * (1 - up) + np.float32([0.09, 0.23, 0.56]) * up,
                   np.float32([0.42, 0.46, 0.50]) * (1 - down) + np.float32([0.12, 0.14, 0.15]) * down)
    col = col + np.exp(-(y * y) / (2 * 0.004)) * np.float32([0.36, 0.28, 0.20])
    ang = np.arccos(np.clip(d @ s, -1, 1))[..., None]
    col = col + (np.exp(-(ang / math.radians(1.7)) ** 2) * 42.0 + np.exp(-(ang / math.radians(12.0)) ** 2) * 2.6) \
        * np.float32([1.0, 0.96, 0.88])
    out = np.ones((h, w, 4), np.float32)
    out[..., :3] = col * level
    return out


def heading_rotation(heading):
    """World-from-vehicle rotation with vehicle -Z along the horizontal (x, z) heading."""
    return rot((0, 1, 0), math.atan2(-heading[0], -heading[1]))


class Stage:
    """Renderer, scene, streaming terrain and the two-eye vehicle."""

    def __init__(self, tone_mapping=TONE_MAPPING):
        self.canvas = tp.Canvas("flyeye scenarios", width=128, height=128, headless=True, vsync=False)
        r = self.renderer = tp.VulkanRenderer(self.canvas, flush_frames=1)
        self.settings = configure_sensor_renderer(r)
        for name in ("deferred_ao", "probe_gi", "volumetric_fog"):
            try:
                setattr(r, name, False)
            except Exception:
                pass
        self.set_tone_mapping(tone_mapping)
        self.scene = tp.Scene()
        self.sky = {k: tp.float_texture(sky_rgba(SUN_DIR, lv * SKY_GAIN)) for k, lv in LIGHTS.items()}
        self.sun = tp.DirectionalLight(tp.Color(1.0, 0.95, 0.86), SUN_INTENSITY)
        self.sun.position.set(*(SUN_DIR * 1000.0))
        self.scene.add(self.sun)
        t0 = time.perf_counter()
        focus = (float(FACE[0] - HEADING[0] * 200), 0.0, float(FACE[1] - HEADING[1] * 200))
        self.geo = tp.GeoScene(str(PACK), forest_focus=focus, scatter=False)
        self.scene.add(self.geo)
        print(f"terrain {PACK.name} loaded in {time.perf_counter() - t0:.1f} s, {self.geo.stats}")
        wm = tp.MeshStandardMaterial()
        wm.color = tp.Color(0.03, 0.06, 0.07)
        wm.roughness = 0.35
        water = tp.Mesh(tp.PlaneGeometry(self.geo.pack_world_size, self.geo.pack_world_size), wm)
        water.rotation.x = -math.pi / 2
        water.position.y = self.geo.sea_level
        self.scene.add(water)
        self.vehicle = tp.Group()
        self.scene.add(self.vehicle)
        self.cams = []
        for yaw in YAWS:
            c = tp.PerspectiveCamera(FOV, 1.0, NEAR, FAR)
            c.rotation.y = math.radians(yaw)
            self.vehicle.add(c)
            self.cams.append(c)
        self.main_cam = tp.PerspectiveCamera(60, 1.0, NEAR, FAR)
        self.vehicle.add(self.main_cam)
        self.t, self.pos = 0.0, np.zeros(3)
        self.set_light("bright")
        self.pose(np.array([FACE[0], self.ground(0.0) + AGL, FACE[1]]), heading_rotation(HEADING))
        self.render()
        self.eyes = [EyeView(r, c, SIZE) for c in self.cams]
        self.render()
        for e in self.eyes:
            e.arm()

    def set_tone_mapping(self, name):
        self.tone_mapping = name
        self.renderer.tone_mapping = getattr(tp.ToneMapping, name)

    def set_light(self, name):
        self.light = name
        self.scene.environment = self.sky[name]
        self.scene.background = self.sky[name]
        self.sun.intensity = SUN_INTENSITY * LIGHTS[name]

    def ground(self, s, lateral=0.0):
        """Terrain height at corridor coordinate s (m back from the wall base)."""
        x, z = FACE - HEADING * s + np.array([-HEADING[1], HEADING[0]]) * lateral
        return float(self.geo.height_at(float(x), float(z)))

    def pose(self, pos, R):
        self.pos = np.asarray(pos, float)
        self.vehicle.quaternion.set(*quat(R))
        self.vehicle.position.set(*self.pos)

    def render(self):
        self.t += DT
        self.renderer.sim_time = self.t
        self.geo.update(tp.Vector3(*self.pos))
        self.renderer.render(self.scene, self.main_cam)

    def warm(self, poses, calm=30, cap=3000):
        """Stream every pose until no tile is baking, then return frames rendered."""
        n = 0
        for pos, R in poses:
            self.pose(pos, R)
            still = 0
            while still < calm and n < cap:
                self.render()
                n += 1
                still = still + 1 if self.geo.stats["baking"] == 0 else 0
        return n

    def range_geom(self, pos, fwd, max_range=800.0, step=0.25):
        """Distance along fwd (unit, world) to the heightfield surface, m; nan if none."""
        for i in range(1, int(max_range / step)):
            p = pos + fwd * (i * step)
            if self.geo.height_at(float(p[0]), float(p[2])) >= p[1]:
                return i * step
        return float("nan")


# -- schedules ----------------------------------------------------------------------------------
def build(segs):
    """segs: (name, T, w (3,) rad/s, v (3,) m/s, ramp s). Body-frame w, v per frame and labels."""
    w_all, v_all, labels, k = [], [], [], 0
    for name, T, w, v, r in segs:
        n = int(round(T / DT))
        for i in range(n):
            a = ramp((i + 0.5) * DT, T, r) if r > 0 else 1.0
            w_all.append(np.asarray(w, float) * a)
            v_all.append(np.asarray(v, float) * a)
        labels.append((name, k, k + n))
        k += n
    return np.array(w_all), np.array(v_all), labels


def integrate(pos0, R0, w, v):
    pos, R, P, Rs = np.array(pos0, float), np.array(R0, float), [], []
    for k in range(len(w)):
        n = np.linalg.norm(w[k])
        if n > 0:
            R = R @ rot(w[k] / n, n * DT)
        pos = pos + R @ v[k] * DT
        P.append(pos.copy())
        Rs.append(R.copy())
    return np.array(P), np.array(Rs)


REST = ("rest", 1.0, (0, 0, 0), (0, 0, 0), 0)
ROT_RATES = ((30.0, 1.2), (90.0, 0.8), (180.0, 0.6))  # deg/s, segment length s (ramps 0.1 s)


def scenario(stage: Stage, name: str):
    """-> dict(pos (T, 3), R (T, 3, 3) world-from-vehicle, w, v (T, 3) body frame, labels, info)."""
    R0 = heading_rotation(HEADING)
    fwd = np.array([HEADING[0], 0.0, HEADING[1]])
    info = {}
    if name.startswith("straight_"):
        speed = float(name.split("_")[1])
        s0, length = 400.0, speed * 3.3
        y = max(stage.ground(s) for s in np.arange(s0 - length - 5, s0 + 5, 2.0)) + AGL
        segs = [REST, (f"fly {speed:g} m/s", 3.6, (0, 0, 0), (0, 0, -speed), 0.3), ("rest", 0.5, (0, 0, 0), (0, 0, 0), 0)]
        pos0 = np.array([FACE[0] - HEADING[0] * s0, y, FACE[1] - HEADING[1] * s0])
        info.update(speed=speed, start_s=s0, altitude=y)
    elif name in ("yaw", "pitch", "roll"):
        s0 = 350.0
        y = stage.ground(s0) + AGL
        axis = dict(pitch=0, yaw=1, roll=2)[name]
        sign = -1.0 if name == "pitch" else 1.0  # pitch down first: the eyes sweep the ground, not the sky
        segs = [REST]
        for deg, T in ROT_RATES:
            for sgn, tag in ((sign, "+" if sign > 0 else "-"), (-sign, "-" if sign > 0 else "+")):
                w = np.zeros(3)
                w[axis] = sgn * math.radians(deg)
                segs += [(f"{name} {tag}{deg:g}", T, w, (0, 0, 0), 0.1), ("rest", 0.4, (0, 0, 0), (0, 0, 0), 0)]
        pos0 = np.array([FACE[0] - HEADING[0] * s0, y, FACE[1] - HEADING[1] * s0])
        info.update(start_s=s0, altitude=y, rates_deg=[d for d, _ in ROT_RATES])
    elif name == "approach":
        speed, start, stop, r = 15.0, 150.0, 18.0, 0.5
        hit = 0.0
        for _ in range(3):  # altitude clears the ground under the path by CLEAR; the wall moves with it
            y = max(stage.ground(s) for s in np.arange(hit + stop - 5, hit + start + 20, 2.0)) + CLEAR
            probe = np.array([FACE[0] - HEADING[0] * 250, y, FACE[1] - HEADING[1] * 250])
            hit = 250.0 - stage.range_geom(probe, fwd)  # corridor s of the wall at this altitude
        T = (start - stop) / speed + r  # trapezoid area = speed (T - r)
        s0 = hit + start
        segs = [REST, ("approach", T, (0, 0, 0), (0, 0, -speed), r), ("rest", 0.3, (0, 0, 0), (0, 0, 0), 0)]
        pos0 = np.array([FACE[0] - HEADING[0] * s0, y, FACE[1] - HEADING[1] * s0])
        info.update(speed=speed, start_range=start, stop_range=stop, altitude=y, wall_s=hit)
    else:
        raise ValueError(name)
    w, v, labels = build(segs)
    pos, R = integrate(pos0, R0, w, v)
    info["min_agl"] = float(min(p[1] - stage.geo.height_at(float(p[0]), float(p[2])) for p in pos[::10]))
    return dict(pos=pos, R=R, w=w, v=v, labels=labels, info=info)


# -- recording ----------------------------------------------------------------------------------
class Instruments:
    """Circuit, readouts and truth, built once per stage."""

    def __init__(self):
        self.R_eyes = np.stack([rot((0, 1, 0), math.radians(y)) for y in YAWS])
        self.lat = HexLattice()
        self.lobe = OpticLobe(device="cuda")
        self.motion = MotionField(self.lobe)
        self.rot_m = RotationReadout(SIZE, FOV, self.R_eyes, method="matched")
        self.rot_l = RotationReadout(SIZE, FOV, self.R_eyes, method="lstsq")
        self.loom = Looming(SIZE, FOV, self.R_eyes)
        self.flow_box = AovFlow(self.lat, DT, SIZE, FOV, mode="box")
        self.flow_centre = AovFlow(self.lat, DT, SIZE, FOV, mode="centre")
        self.foot = Footprint(self.lat, SIZE)
        self.range = ForwardRange(self.R_eyes, NEAR, FAR, SIZE, FOV)


def eye_row_png(stage, path):
    from PIL import Image

    row = np.concatenate([e.color[..., [2, 1, 0]].cpu().numpy() for e in stage.eyes], axis=1)
    Image.fromarray(np.ascontiguousarray(row)).save(path)


def run(stage: Stage, ins: Instruments, name: str, light: str, out: Path, renders: Path, video_full=(0.4, 2.0)):
    sc = scenario(stage, name)
    pos, R, w, v, labels = sc["pos"], sc["R"], sc["w"], sc["v"], sc["labels"]
    n = len(pos)
    stage.set_light(light)
    t0 = time.perf_counter()
    step = max(1, int(10.0 / max(np.linalg.norm(v, axis=1).max() * DT, 1e-9)))  # a pose every 10 m
    warm = stage.warm([(pos[k], R[k]) for k in range(n - 1, -1, -step)] + [(pos[0], R[0])])
    for _ in range(60):  # TAA and exposure-free history settle at the start pose
        stage.render()
    t_warm = time.perf_counter() - t0
    fwd_world = -R[:, :, 2]
    geom = np.full(n, np.nan)
    if name == "approach" or name.startswith("straight"):
        r0 = stage.range_geom(pos[0], fwd_world[0])
        geom = r0 - np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))])
    speed = np.linalg.norm(v, axis=1)
    video = VideoOut(renders / f"{name}_{light}.mp4", ins.lat, len(stage.eyes), size=SIZE, yaws=YAWS, truth=True,
                     field_full=video_full[0], truth_full=video_full[1])
    keys = ("receptors", "t4t5_raw", "field", "flow_box", "flow_centre", "contrast", "sky_frac", "rot_matched",
            "rot_lstsq", "rot_truth_box", "rot_truth_centre", "loom", "loom_centre", "loom_truth", "range_depth",
            "range_depth_min", "nearest", "baking", "tiles")
    rec = {k: [] for k in keys}
    seg_of = np.zeros(n, dtype=np.int16)
    for i, (_, a, b) in enumerate(labels):
        seg_of[a:b] = i
    stills = {}
    main_seg = max(range(len(labels)), key=lambda i: labels[i][2] - labels[i][1] if labels[i][0] != "rest" else -1)
    for i, (lab, a, b) in enumerate(labels):
        if lab != "rest" and (name not in ("yaw", "pitch", "roll") or "90" in lab):
            stills.setdefault("mid", (a + b) // 2)
    stills["last"] = labels[main_seg][2] - 1
    stills["first"] = labels[main_seg][1]
    for k in range(n):
        stage.pose(pos[k], R[k])
        stage.render()
        x = torch.stack([e.receptors() for e in stage.eyes])
        if k == 0:
            ins.lobe.fade_in(x, DT)
        vv = ins.lobe.step(x, DT)
        raw = ins.motion.raw(vv)
        f = ins.motion(vv)
        lo = ins.loom(f)
        w_cam = ins.R_eyes.transpose(0, 2, 1) @ w[k]  # (B, 3) body rate in each eye's frame
        fb = torch.stack([ins.flow_box(e.motion, e.depth, w_cam[b]) for b, e in enumerate(stage.eyes)]).float()
        fc = torch.stack([ins.flow_centre(e.motion, e.depth, w_cam[b]) for b, e in enumerate(stage.eyes)]).float()
        lum = [bgra_luma_u8(e.color).float() / 255 for e in stage.eyes]
        rec["receptors"].append(x.half())
        rec["t4t5_raw"].append(raw.half())
        rec["field"].append(f.half())
        rec["flow_box"].append(fb)
        rec["flow_centre"].append(fc)
        rec["contrast"].append(torch.stack([ins.foot.contrast(l) for l in lum]).half())
        rec["sky_frac"].append(torch.stack([ins.foot.sky_fraction(e.depth) for e in stage.eyes]).half())
        rec["rot_matched"].append(ins.rot_m(f))
        rec["rot_lstsq"].append(ins.rot_l(f))
        rec["rot_truth_box"].append(ins.rot_m(fb))
        rec["rot_truth_centre"].append(ins.rot_m(fc))
        rec["loom"].append(lo["value"])
        rec["loom_centre"].append(lo["centre"])
        rec["loom_truth"].append(ins.loom(fb)["value"])
        zr, zmin = ins.range([e.depth for e in stage.eyes])
        rec["range_depth"].append(zr)
        rec["range_depth_min"].append(zmin)
        rec["nearest"].append(ins.range.nearest([e.depth for e in stage.eyes]))
        st = stage.geo.stats
        rec["baking"].append(st["baking"])
        rec["tiles"].append(st["tiles"])
        seg = labels[seg_of[k]][0]
        est, tw, tr = rec["rot_matched"][-1].tolist(), w[k], rec["rot_truth_box"][-1].tolist()
        inv_tau = speed[k] / zr if zr == zr and zr > 0 else 0.0
        video.write([e.color for e in stage.eyes], x, f,
                    f"{name} / {light}   t {k * DT:5.2f} s   {seg}   |v| {speed[k]:4.1f} m/s   true w "
                    f"({tw[0]:+.2f}, {tw[1]:+.2f}, {tw[2]:+.2f}) rad/s   depth range {zr:6.1f} m  speed/range {inv_tau:.2f} /s",
                    f"circuit w ({est[0]:+.3f}, {est[1]:+.3f}, {est[2]:+.3f}) a.u.   true flow via templates "
                    f"({tr[0]:+.2f}, {tr[1]:+.2f}, {tr[2]:+.2f}) rad/s   looming circuit {float(lo['value']):.3f}"
                    f"  true flow {float(rec['loom_truth'][-1]):.3f}", truth=fb)
        for tag, kk in stills.items():
            if k == kk:
                video.last.save(renders / f"{name}_{light}_{tag}_sheet.png")
                eye_row_png(stage, renders / f"{name}_{light}_{tag}_eyes.png")
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    video.close()

    def stack(key, dtype=None):
        vals = rec[key]
        a = torch.stack(vals).cpu().numpy() if torch.is_tensor(vals[0]) else np.asarray(vals)
        return a.astype(dtype) if dtype else a

    range_depth = stack("range_depth", np.float64)
    tau_depth, zdot = smooth_tau(range_depth, DT)
    tau_geom = np.where(speed > 1e-6, geom / np.maximum(speed, 1e-9), np.inf)
    data = dict(
        t=np.arange(n) * DT, seg=seg_of, labels=np.array([f"{a}|{b}|{c}" for a, b, c in labels]),
        pos_world=pos, R_world=R, w_body=w, v_body=v, speed=speed,
        range_geom=geom, tau_geom=tau_geom, range_depth=range_depth, range_depth_min=stack("range_depth_min", np.float64), range_depth_rate=zdot, nearest=stack("nearest", np.float64),
        tau_depth=tau_depth, baking=stack("baking", np.int32), tiles=stack("tiles", np.int32),
        **{k: stack(k) for k in ("receptors", "t4t5_raw", "field", "flow_box", "flow_centre", "contrast", "sky_frac",
                                 "rot_matched", "rot_lstsq", "rot_truth_box", "rot_truth_centre", "loom", "loom_centre",
                                 "loom_truth")})
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{name}_{light}.npz", **data)
    meta = dict(scenario=name, light=light, light_level=LIGHTS[light], frames=n, dt=DT, size=SIZE, fov_deg=FOV,
                eye_yaws_deg=list(YAWS), eye_rotations=ins.R_eyes.tolist(), near=NEAR, far=FAR, pack=PACK.name,
                face_xz=FACE.tolist(), heading_xz=HEADING.tolist(), sun_dir=SUN_DIR.tolist(), sun_intensity=SUN_INTENSITY,
                tone_mapping=stage.tone_mapping, flush_frames=1, info=sc["info"], warm_frames=warm, warm_s=t_warm,
                wall_s=wall, video_field_full=video_full[0], video_truth_full=video_full[1], range_pixels_per_eye=ins.range.count, min_nearest_m=float(np.min(data["nearest"])),
                max_baking_during_run=int(data["baking"].max()), stills=stills,
                settings={k: str(x) for k, x in stage.settings.items()})
    json.dump(meta, open(out / f"{name}_{light}.json", "w"), indent=1)
    print(f"{name}/{light}: {n} frames, warm {warm} frames {t_warm:.1f} s, run {wall - t_warm:.1f} s, "
          f"max baking {meta['max_baking_during_run']}")
    return data, meta


# -- scout --------------------------------------------------------------------------------------
def scout(args):
    from PIL import Image

    out = args.renders / "scout"
    out.mkdir(parents=True, exist_ok=True)
    stage = Stage("NoToneMapping")
    lat = HexLattice()
    R0 = heading_rotation(HEADING)
    fwd = np.array([HEADING[0], 0.0, HEADING[1]])
    report = dict(corridor=[(s, stage.ground(s), stage.ground(s, 150), stage.ground(s, -150)) for s in range(-100, 601, 50)])
    probes = {}
    for s, agl in ((400, AGL), (350, AGL), (130, 35), (40, 35)):
        p = np.array([FACE[0] - HEADING[0] * s, stage.ground(s) + agl, FACE[1] - HEADING[1] * s])
        probes[f"s{s}"] = p
    y_app = stage.ground(0.0) + 35.0
    probe = np.array([FACE[0] - HEADING[0] * 200, y_app, FACE[1] - HEADING[1] * 200])
    report["wall_s_at_approach_altitude"] = 200.0 - stage.range_geom(probe, fwd)
    stats = {}
    for pname, p in probes.items():
        if pname not in args.probes.split(","):
            continue
        report[f"{pname}_warm_frames"] = stage.warm([(p, R0)], calm=60)
        for tm in args.tone_mappings.split(","):
            stage.set_tone_mapping(tm)
            for light in LIGHTS:
                stage.set_light(light)
                for _ in range(30):
                    stage.render()
                for e in stage.eyes:
                    e.frames.sync()
                row = np.concatenate([e.color[..., [2, 1, 0]].cpu().numpy() for e in stage.eyes], axis=1)
                Image.fromarray(np.ascontiguousarray(row)).save(out / f"{pname}_{tm}_{light}_sun{SUN_INTENSITY:g}.png")
                rgb = row.astype(np.int32)
                luma = (rgb[..., 0] * 19595 + rgb[..., 1] * 38470 + rgb[..., 2] * 7471 + 0x8000) >> 16
                sky = np.concatenate([(e.depth == 0).cpu().numpy() for e in stage.eyes], axis=1)
                hist = np.histogram(luma, bins=16, range=(0, 256))[0] / luma.size
                stats[f"{pname}_{tm}_{light}"] = dict(
                    clip_any_channel_255=float((rgb == 255).any(-1).mean()), luma_ge_250=float((luma >= 250).mean()),
                    luma_le_5=float((luma <= 5).mean()), terrain_luma_p1_p50_p99=np.percentile(luma[~sky], [1, 50, 99]).tolist(),
                    sky_luma_p50=float(np.median(luma[sky])) if sky.any() else None, sky_frac=float(sky.mean()),
                    hist16=hist.round(4).tolist())
                print(pname, tm, light, {k: v for k, v in stats[f"{pname}_{tm}_{light}"].items() if k != "hist16"})
    report["tonemap"] = stats
    # motion AOV at sky and terrain columns under a pure yaw, against the analytic rotation flow
    stage.set_tone_mapping("NoToneMapping")
    stage.set_light("bright")
    p = probes["s350"]
    stage.warm([(p, R0)], calm=30)
    R = R0.copy()
    W = np.array([0.0, math.radians(90.0), 0.0])
    fc, fb = AovFlow(lat, DT, SIZE, FOV, "centre"), AovFlow(lat, DT, SIZE, FOV, "box")
    rows = lat.pixel_rc(SIZE)
    for k in range(40):
        R = R @ rot((0, 1, 0), W[1] * DT)
        stage.pose(p, R)
        stage.render()
        for e in stage.eyes:
            e.frames.sync()
    flow_c = torch.stack([fc(e.motion) for e in stage.eyes]).cpu()
    flow_b = torch.stack([fb(e.motion) for e in stage.eyes]).cpu()
    R_eyes = np.stack([rot((0, 1, 0), math.radians(y)) for y in YAWS])
    w_cam = R_eyes.transpose(0, 2, 1) @ W
    flow_bs = torch.stack([fb(e.motion, e.depth, w_cam[b]) for b, e in enumerate(stage.eyes)]).cpu()
    flow_cs = torch.stack([fc(e.motion, e.depth, w_cam[b]) for b, e in enumerate(stage.eyes)]).cpu()
    ana = rotation_flow(W, np.stack([rot((0, 1, 0), math.radians(y)) for y in YAWS]), lat, SIZE, FOV)
    sky = torch.stack([(e.depth[rows[:, 0].cuda(), rows[:, 1].cuda()] == 0).cpu() for e in stage.eyes])
    err_c = (flow_c.double() - ana).norm(dim=1)  # (B, 721)
    err_b = (flow_b.double() - ana).norm(dim=1)
    report["yaw90_sky_check"] = dict(
        analytic_mean_speed=float(ana.norm(dim=1).mean()), sky_columns=int(sky.sum()),
        sky_centre_err_mean=float(err_c[sky].mean()) if sky.any() else None,
        sky_centre_speed_mean=float(flow_c.norm(dim=1)[sky].mean()) if sky.any() else None,
        terrain_centre_minus_rotation_mean=float(err_c[~sky].mean()),
        terrain_box_minus_rotation_mean=float(err_b[~sky].mean()),
        sky_patched_centre_err_mean_all=float((flow_cs.double() - ana).norm(dim=1).mean()),
        sky_patched_box_err_mean_all=float((flow_bs.double() - ana).norm(dim=1).mean()),
        rot_readout_patched_box=RotationReadout(SIZE, FOV, R_eyes)(flow_bs.float()).tolist(),
        rot_readout_unpatched_box=RotationReadout(SIZE, FOV, R_eyes)(flow_b.float()).tolist(), true_w=W.tolist())
    print(report["yaw90_sky_check"])
    json.dump(report, open(out / f"scout_sun{SUN_INTENSITY:g}_sky{SKY_GAIN:g}.json", "w"), indent=1)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scout", action="store_true")
    ap.add_argument("--probes", default="s400,s350,s130,s40", help="--scout: corridor poses")
    ap.add_argument("--tone-mappings", default="NoToneMapping,AgX,ACESFilmic,Neutral", help="--scout: candidates")
    ap.add_argument("--only", default=",".join(SCENARIOS))
    ap.add_argument("--lights", default=",".join(LIGHTS))
    ap.add_argument("--tone-mapping", default=TONE_MAPPING)
    ap.add_argument("--field-full", type=float, default=0.4, help="video: circuit field magnitude at full brightness")
    ap.add_argument("--truth-full", type=float, default=2.0, help="video: true flow (rad/s) at full brightness")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--renders", type=Path, default=RENDERS)
    args = ap.parse_args(argv)
    args.renders.mkdir(parents=True, exist_ok=True)
    if args.scout:
        return scout(args)
    stage = Stage(args.tone_mapping)
    ins = Instruments()
    for light in args.lights.split(","):
        for name in args.only.split(","):
            run(stage, ins, name, light, args.out, args.renders, (args.field_full, args.truth_full))
    for e in stage.eyes:
        e.close()


if __name__ == "__main__":
    main()
