"""Moving-edge stimuli for the flyeye Phase 0 spike, rendered by threepp Vulkan.

A fixed perspective camera (square, vertical FOV --fov) looks straight down -Z
at two unlit planes: a uniform mid-grey backdrop at depth 2, and in front of it
at depth 1 a large ON (bright) or OFF (dark) quad whose leading edge sweeps
across the whole frame. The quad is geometry sliding past the camera, so the
renderer's own motion vectors describe it exactly.

Eight movies: {on, off} x {right, left, up, down}, directions in IMAGE space
(right = toward larger column, up = toward row 0). ON means a bright region
advances into the grey; OFF a dark one. Each movie is --preroll static frames
with the edge just off-screen, then one frame per step until the edge has left
the far side. One fly eye column is 13 px, so

    px_per_frame = speed_cols_per_s * 13 / fps

Output, one 8-bit RGB PNG of exactly size x size per frame:

    <out>/<movie>/frame_NNNN.png
    <out>/meta.json      (timing, geometry, renderer settings, measured values)

    py -3.14 python/examples/flyeye/render_edges.py --out C:/dev/_flyeye/stimuli

Renderer: tone mapping NONE (the post composite then only sRGB-encodes the
linear colour), no auto exposure, no bloom, no DoF, no DLSS/FSR, no sensor
noise, no AO/GI/fog. Two things have no Python switch and are measured
instead (meta.json): the built-in TAA resolve and the post-TAA RCAS sharpen.
TAA sets how well a sub-pixel edge position is resolved; every render() drives
--flush GPU frames of the same scene state, and more of them let TAA average
more jitter samples (edge position error vs the model, max over the crossing:
0.85 px at 1, 0.7 px at 3, 0.3 px at 8, 0.2 px at 16, hence the default 16).
RCAS leaves a one-pixel over/undershoot on each side of the edge (ON: 209 and
~121 next to 204 | 128; OFF: ~46 and ~139 next to 53 | 128); flat regions are
exact. The grey/ON/OFF inputs are chosen from the rendered 8-bit values before
the movies are drawn: the unlit colour passes an 8-bit LINEAR store, so only
k/255 inputs are distinct and OFF defaults to 53 (8/255 renders 50, 9/255 53).
Material colours need needs_update() on Vulkan or the change never uploads.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import threepp as tp  # noqa: E402

COLUMN_PX = 13
MARGIN_PX = 2.0          # the edge starts/ends this far outside the frame
DIRECTIONS = ("right", "left", "up", "down")
POLARITIES = ("on", "off")


def srgb_to_linear(x):
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def pin_renderer(r):
    """Switch off everything temporal/adaptive that the API exposes. Returns what was set."""
    applied = {}

    def put(name, value):
        try:
            setattr(r, name, value)
            applied[name] = str(getattr(r, name))
        except Exception as e:  # property missing in this build
            applied[name] = f"unavailable ({type(e).__name__})"

    put("auto_exposure", False)
    put("physical_camera", False)
    put("tone_mapping", tp.ToneMapping.NoToneMapping)
    put("tone_mapping_exposure", 1.0)
    put("bloom_intensity", 0.0)
    put("depth_of_field", False)
    for up in ("dlss", "fsr"):
        put(up, False)
    put("render_scale", 1.0)
    put("deferred_ao", False)
    put("probe_gi", False)
    put("volumetric_fog", False)
    put("shadow_map_enabled", False)
    put("occlusion_culling", False)
    put("auto_lod", False)
    put("env_sun_extraction", False)
    put("starfield", 0.0)
    try:
        r.set_sensor_noise(False)
        applied["sensor_noise"] = "off"
    except Exception as e:
        applied["sensor_noise"] = f"unavailable ({type(e).__name__})"
    try:
        r.set_lens_distortion("none")
        applied["lens_distortion"] = "none"
    except Exception as e:
        applied["lens_distortion"] = f"unavailable ({type(e).__name__})"
    applied["taa"] = "always on (no API switch); edge profile measured"
    applied["rcas_sharpen"] = "engine default 0.5 (not exposed to Python); measured"
    return applied


def basic(value):
    m = tp.MeshBasicMaterial()
    m.color = tp.Color(value, value, value)
    return m


class Stage:
    def __init__(self, args):
        self.size = args.size
        self.half = math.tan(math.radians(args.fov) / 2.0)   # world half-extent at depth 1
        self.world_per_px = 2.0 * self.half / self.size

        self.canvas = tp.Canvas("flyeye edges", width=self.size, height=self.size,
                                headless=True, vsync=False)
        self.r = tp.VulkanRenderer(self.canvas, flush_frames=args.flush)
        self.applied = pin_renderer(self.r)

        self.scene = tp.Scene()
        self.cam = tp.PerspectiveCamera(args.fov, 1.0, 0.1, 10.0)
        self.cam.position.set(0, 0, 0)   # default orientation looks down -Z, +Y up

        self.bg_mat = basic(0.2)
        bg_side = 4.0 * 2.0 * self.half + 10.0
        self.bg = tp.Mesh(tp.PlaneGeometry(bg_side, bg_side), self.bg_mat)
        self.bg.position.set(0, 0, -2.0)
        self.scene.add(self.bg)

        self.q_mat = basic(0.2)
        self.q_len = 4.0 * self.half + 1.0          # > 2 frame widths: trailing side never shows
        self.quad = tp.Mesh(tp.PlaneGeometry(self.q_len, self.q_len), self.q_mat)
        self.quad.position.set(0, 0, -1.0)
        self.scene.add(self.quad)

        self.t = 0.0
        self.dt = 1.0 / args.fps

    # Vulkan uploads material values only when the material version bumps:
    # a plain colour write without needs_update() never reaches the GPU.
    def set_bg(self, v):
        self.bg_mat.color = tp.Color(v, v, v)
        self.bg_mat.needs_update()
        self.scene.background = tp.Color(v, v, v)

    def set_region(self, v):
        self.q_mat.color = tp.Color(v, v, v)
        self.q_mat.needs_update()

    def hide_region(self):
        # park the quad far outside the frustum
        self.quad.position.set(1000.0, 1000.0, -1.0)

    def place_edge(self, direction, e):
        """e = edge position in pixels along the motion axis, measured from the
        side the edge enters (0 = that frame border, size = the far border)."""
        L2 = self.q_len / 2.0
        s = self.size
        if direction == "right":        # edge at column e, region at columns < e
            x = e * self.world_per_px - self.half
            self.quad.position.set(x - L2, 0.0, -1.0)
        elif direction == "left":       # edge at column s - e, region at columns > edge
            x = (s - e) * self.world_per_px - self.half
            self.quad.position.set(x + L2, 0.0, -1.0)
        elif direction == "down":       # edge at row e, region at rows < e (above it)
            y = self.half - e * self.world_per_px
            self.quad.position.set(0.0, y + L2, -1.0)
        elif direction == "up":         # edge at row s - e, region at rows > edge (below it)
            y = self.half - (s - e) * self.world_per_px
            self.quad.position.set(0.0, y - L2, -1.0)
        else:
            raise ValueError(direction)

    def render(self):
        self.t += self.dt
        self.r.sim_time = self.t
        self.r.render(self.scene, self.cam)
        img = np.asarray(self.r.read_pixels())
        if img.shape[:2] != (self.size, self.size):
            raise RuntimeError(f"read_pixels gave {img.shape}, expected {self.size}x{self.size}")
        return np.ascontiguousarray(img[:, :, :3])


def calibrate(stage, targets, warm=6):
    """Pick the linear input level per target so the rendered 8-bit value hits it.

    The unlit colour reaches the output through an 8-bit LINEAR store, so the
    inputs that matter are k/255; in the darks neighbouring codes are ~3 sRGB
    levels apart (8/255 -> 50, 9/255 -> 53). Try the codes around the analytic
    guess, keep the closest, and report what was actually rendered."""
    stage.hide_region()
    out = {}
    c = stage.size // 2
    for name, t8 in targets.items():
        k0 = int(round(float(srgb_to_linear(t8 / 255.0)) * 255.0))
        tried = []
        for k in (k0 - 1, k0, k0 + 1):
            if not 0 <= k <= 255:
                continue
            v = k / 255.0
            stage.set_bg(v)
            stage.r.reset_temporal_history()
            for _ in range(warm):
                img = stage.render()
            patch = img[c - 50:c + 50, c - 50:c + 50]
            tried.append({"linear_code": k, "linear_input": v, "measured_mean": float(patch.mean()),
                          "measured_min": int(patch.min()), "measured_max": int(patch.max())})
        best = min(tried, key=lambda d: abs(d["measured_mean"] - t8))
        out[name] = {"target_8bit": t8, "linear_input": best["linear_input"],
                     "linear_code": best["linear_code"], "measured_8bit": best["measured_mean"],
                     "tried": tried}
    return out


def edge_position(img, direction, lo, hi):
    """Sub-pixel edge position (pixels from the entering border) from the mean
    profile across the motion axis: count of pixels already covered by the region."""
    g = img.astype(np.float64).mean(axis=2)
    if direction in ("right", "left"):
        prof = g.mean(axis=0)
    else:
        prof = g.mean(axis=1)
    frac = np.clip((prof - hi) / (lo - hi), 0.0, 1.0)   # 1 where region, 0 where grey
    covered = float(frac.sum())
    return covered, prof


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=r"C:\dev\_flyeye\stimuli")
    ap.add_argument("--speed-cols-per-s", type=float, default=13.0)
    ap.add_argument("--fps", type=float, default=100.0)
    ap.add_argument("--preroll", type=int, default=20)
    ap.add_argument("--size", type=int, default=403)
    ap.add_argument("--fov", type=float, default=90.0)
    ap.add_argument("--grey", type=int, default=128, help="target 8-bit background level")
    ap.add_argument("--on", type=int, default=204, help="target 8-bit ON level")
    ap.add_argument("--off", type=int, default=53, help="target 8-bit OFF level")
    ap.add_argument("--warmup", type=int, default=10, help="unsaved frames before each movie")
    ap.add_argument("--flush", type=int, default=16,
                    help="GPU frames the Python facade drives per render() (same scene state)")
    ap.add_argument("--only", default="", help="comma-separated movie names to render (default all)")
    args = ap.parse_args()

    if not tp.HAS_VULKAN:
        sys.exit("this threepp build has no Vulkan backend")

    ppf = args.speed_cols_per_s * COLUMN_PX / args.fps
    n_move = int(math.ceil((args.size + 2.0 * MARGIN_PX) / ppf))
    os.makedirs(args.out, exist_ok=True)

    t_start = time.perf_counter()
    stage = Stage(args)
    stage.hide_region()
    stage.set_bg(0.2)
    for _ in range(5):                     # first frames build pipelines
        stage.render()

    cal = calibrate(stage, {"grey": args.grey, "on": args.on, "off": args.off})
    lin = {k: v["linear_input"] for k, v in cal.items()}
    print("calibration:", {k: (round(v["linear_input"], 5), round(v["measured_8bit"], 2))
                           for k, v in cal.items()})

    meta = {
        "fps": args.fps, "dt": 1.0 / args.fps,
        "speed_cols_per_s": args.speed_cols_per_s, "column_px": COLUMN_PX,
        "px_per_frame": ppf, "size": args.size, "fov_deg": args.fov,
        "deg_per_px_at_centre": math.degrees(math.atan(2.0 * stage.half / args.size)),
        "deg_per_column_at_centre": math.degrees(math.atan(2.0 * stage.half / args.size * COLUMN_PX)),
        "margin_px": MARGIN_PX, "preroll_frames": args.preroll, "moving_frames": n_move,
        "frames_per_movie": args.preroll + n_move, "warmup_frames_unsaved": args.warmup,
        "edge_position_model": "e(k) = -margin + max(0, k - (preroll - 1)) * px_per_frame, "
                               "pixels from the entering border; frame k is 0-based",
        "image_convention": "row 0 = top; up = toward row 0; right = toward larger column",
        "png": "8-bit RGB, R=G=B",
        "renderer": {"backend": "Vulkan headless", "read": "VulkanRenderer.read_pixels() after render()",
                     "flush_frames_per_render": args.flush, "sim_time": "pinned, += dt per saved/warm frame",
                     "settings": stage.applied},
        "calibration": cal,
        "movies": {},
    }

    for pol in POLARITIES:
        region = lin["on"] if pol == "on" else lin["off"]
        for d in DIRECTIONS:
            name = f"{pol}_{d}"
            if args.only and name not in args.only.split(","):
                continue
            mdir = os.path.join(args.out, name)
            os.makedirs(mdir, exist_ok=True)
            for f in os.listdir(mdir):
                if f.startswith("frame_") and f.endswith(".png"):
                    os.remove(os.path.join(mdir, f))
            stage.set_bg(lin["grey"])
            stage.set_region(region)
            stage.place_edge(d, -MARGIN_PX)
            stage.r.reset_temporal_history()
            for _ in range(args.warmup):
                stage.render()

            lo_v = cal[pol]["measured_8bit"]
            grey_v = cal["grey"]["measured_8bit"]
            positions, expected = [], []
            grey_px, region_px = [], []
            n_frames = args.preroll + n_move
            for k in range(n_frames):
                step = max(0, k - (args.preroll - 1))
                e = -MARGIN_PX + step * ppf
                stage.place_edge(d, e)
                img = stage.render()
                Image.fromarray(img, mode="RGB").save(os.path.join(mdir, f"frame_{k:04d}.png"))
                pos, _ = edge_position(img, d, lo_v, grey_v)
                positions.append(pos)
                expected.append(max(0.0, min(float(args.size), e)))
                if k == 0:
                    grey_px.append(img[..., 0])
                if k == n_frames - 1:
                    region_px.append(img[..., 0])
                if k == args.preroll + n_move // 2:
                    mid_range = [int(img[..., 0].min()), int(img[..., 0].max())]
            # displacement fit over frames where the edge is fully inside the frame
            ks = np.arange(n_frames)
            pos = np.array(positions)
            exp = np.array(expected)
            inside = (exp > 10) & (exp < args.size - 10)
            slope, icpt = np.polyfit(ks[inside], pos[inside], 1)
            resid = pos[inside] - exp[inside]
            step_d = np.diff(pos)[inside[1:] & inside[:-1]]
            g0 = grey_px[0]
            r1 = region_px[0]
            meta["movies"][name] = {
                "polarity": pol, "direction": d, "frames": n_frames,
                "preroll_frames": args.preroll, "moving_frames": n_move,
                "dir": mdir,
                "measured": {
                    "preroll_frame0_min_max_mean": [int(g0.min()), int(g0.max()), float(g0.mean())],
                    "last_frame_min_max_mean": [int(r1.min()), int(r1.max()), float(r1.mean())],
                    "fit_px_per_frame": float(slope),
                    "edge_minus_model_px_mean": float(resid.mean()),
                    "edge_minus_model_px_max_abs": float(np.abs(resid).max()),
                    "per_frame_step_px_mean_std_min_max": [float(step_d.mean()), float(step_d.std()),
                                                           float(step_d.min()), float(step_d.max())],
                    "mid_crossing_frame_min_max": mid_range,
                },
                "edge_position_px": [round(p, 3) for p in positions],
            }
            print(f"{name:10s} frames={n_frames} fit={slope:.4f} px/frame (model {ppf:.4f}) "
                  f"lag mean={resid.mean():+.3f} max|.|={np.abs(resid).max():.3f} step std={step_d.std():.3f}  "
                  f"mid {mid_range[0]}..{mid_range[1]}  "
                  f"f0 {g0.min()}..{g0.max()}  last {r1.min()}..{r1.max()}")

    meta["wall_seconds"] = time.perf_counter() - t_start
    with open(os.path.join(args.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"wrote {len(meta['movies'])} movies to {args.out} in {meta['wall_seconds']:.1f} s")


if __name__ == "__main__":
    main()
