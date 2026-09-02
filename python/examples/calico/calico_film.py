"""WP5 of the Calico demo: the shot list, the sightline test, and the film's bookkeeping.

`spot_calico.py --film OUT_DIR` owns the loop; this module owns everything about
it that is not simulation:

`Shot` / `build_shots`
    Six shots, ~61 s at 30 fps, every static pose taken from `shots.json`'s
    `cameras_v2` and mapped into the demo's Z-up world by the frame map
    `spot_calico.Frame` applies (world = (x, -z, y) + t). Moves are eased with a
    smoothstep on BOTH the eye and the look target, so a push-in starts and ends
    at zero velocity instead of snapping.

`ColumnField`
    WP0's sightline clearance test, kept as a callable instead of a note in a
    JSON comment. It decodes a coarse LOD of the SOG once (trail_analysis'
    decoder), bins the 98th percentile splat height into 0.5 m cells, caches the
    grid, and answers two questions: how far above the rock is this point, and
    what is the WORST clearance along this camera->target line. A negative
    answer means the shot is buried in the bank, which is exactly how WP0's
    first establishing_v2 attempt failed (-1.37 m at the -z brush mound). Every
    static keyframe and midpoint is checked before the film renders; the chase
    shot is checked every frame and LIFTS the camera when its offset pose is
    inside the bank.

`FrameLog`
    The sensor CSV (base pose, the 45-cell scan vector, a hash of the PiP depth
    image, per frame) and the determinism proof: every frame's RAW PIXELS are
    hashed (`renderer.read_pixels()`, not the PNG file, so a PNG encoder version
    cannot flatter the result), the CSV is hashed, and the lot is written to
    `hashes.txt` for a diff against a second run.

The film's clock: ONE accumulated float, `t_sim += DT` per frame, used for both
`world.step(DT)` and `renderer.sim_time = t_sim`. Nothing in the frame path is
allowed to read the wall clock -- that is what makes the two runs comparable at
all (feedback: cross-process determinism needs setSimTime).
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

FPS = 30
DT = 0.02                    # the policy's control step; the film's sim step too


# ── easing ────────────────────────────────────────────────────────────────────
def smoothstep(t):
    t = min(1.0, max(0.0, float(t)))
    return t * t * (3.0 - 2.0 * t)


def ease(a, b, t):
    s = smoothstep(t)
    return np.asarray(a, np.float64) * (1.0 - s) + np.asarray(b, np.float64) * s


def keyframes(keys, u):
    """`keys` = [(u0, pos, look), ...] sorted by u; smoothstep between them."""
    if u <= keys[0][0]:
        return np.asarray(keys[0][1], np.float64), np.asarray(keys[0][2], np.float64)
    for (u0, p0, l0), (u1, p1, l1) in zip(keys, keys[1:]):
        if u <= u1:
            t = (u - u0) / max(u1 - u0, 1e-9)
            return ease(p0, p1, t), ease(l0, l1, t)
    return np.asarray(keys[-1][1], np.float64), np.asarray(keys[-1][2], np.float64)


# ── the sightline test ────────────────────────────────────────────────────────
class ColumnField:
    """98th-percentile splat height per 0.5 m XY cell, in the demo's world frame.

    Built from a coarse LOD of the SOG (level 3 = 2.4M splats is enough for a
    0.5 m grid and decodes in a few seconds) and cached, keyed by the asset, the
    level and the world offset. `top()` is the rock; `clearance()` marches a
    segment and returns its worst height above the rock.
    """

    CELL = 0.5
    PCT = 0.98
    OPACITY_MIN = 0.5

    def __init__(self, asset, t_world, level=3, half_extent=40.0, cache_dir=None,
                 verbose=True):
        self.cell = self.CELL
        t = np.asarray(t_world, np.float64)
        cache_dir = cache_dir or os.environ.get("TEMP") or "."
        key = f"{os.path.basename(str(asset).rstrip('/'))}_l{level}_" \
              f"{t[0]:.3f}_{t[1]:.3f}_{t[2]:.3f}_{half_extent:.0f}"
        self.path = os.path.join(cache_dir, f"calico_columns_{key}.npz")
        if os.path.exists(self.path):
            z = np.load(self.path)
            self.F, self.x0, self.y0 = z["F"], float(z["x0"]), float(z["y0"])
            if verbose:
                print(f"[film] column field from cache {self.path} {self.F.shape}")
            return
        t0 = time.perf_counter()
        from pathlib import Path
        import trail_analysis as TA
        pts, opa, _s = TA.load_level(Path(str(asset)), level)
        keep = opa > self.OPACITY_MIN
        P = pts[keep].astype(np.float64)
        # Y-up post-flip -> this demo's Z-up world, the same map Frame applies.
        W = np.stack([P[:, 0] + t[0], -P[:, 2] + t[1], P[:, 1] + t[2]], axis=1)
        m = (np.abs(W[:, 0]) < half_extent) & (np.abs(W[:, 1]) < half_extent)
        W = W[m]
        self.x0 = float(W[:, 0].min()) - self.cell
        self.y0 = float(W[:, 1].min()) - self.cell
        nx = int((W[:, 0].max() - self.x0) / self.cell) + 2
        ny = int((W[:, 1].max() - self.y0) / self.cell) + 2
        ix = np.clip(((W[:, 0] - self.x0) / self.cell).astype(np.int64), 0, nx - 1)
        iy = np.clip(((W[:, 1] - self.y0) / self.cell).astype(np.int64), 0, ny - 1)
        k = ix * ny + iy
        # Per-cell percentile without a Python loop: sort by (cell, z) and index
        # each run at its own 98th percentile. The percentile, not the max, is
        # the point -- one stray splat 3 m up would otherwise close the wash.
        order = np.lexsort((W[:, 2], k))
        ks, zs = k[order], W[order, 2]
        uniq, start, cnt = np.unique(ks, return_index=True, return_counts=True)
        pick = start + np.minimum(cnt - 1, (self.PCT * (cnt - 1)).astype(np.int64))
        F = np.full(nx * ny, np.nan, np.float32)
        F[uniq] = zs[pick]
        self.F = F.reshape(nx, ny)
        np.savez_compressed(self.path, F=self.F, x0=self.x0, y0=self.y0)
        if verbose:
            print(f"[film] column field {self.F.shape} from level {level} "
                  f"({len(W)} splats in range) in {time.perf_counter() - t0:.1f}s "
                  f"-> {self.path}")

    def top(self, x, y):
        """Rock height at (x, y), or -inf where nothing was ever seen (open air)."""
        i = int(min(max((x - self.x0) / self.cell, 0), self.F.shape[0] - 1))
        j = int(min(max((y - self.y0) / self.cell, 0), self.F.shape[1] - 1))
        v = self.F[i, j]
        return -1e9 if not np.isfinite(v) else float(v)

    def clearance(self, cam, target, steps=48, skip_end=0.12):
        """Worst height of the cam->target segment above the rock under it.

        The last `skip_end` of the segment is skipped: the target IS on the
        ground (it is the robot), so the line necessarily reaches it.
        """
        a = np.asarray(cam, np.float64)
        b = np.asarray(target, np.float64)
        worst, at = 1e9, a
        n = max(2, int(steps * (1.0 - skip_end)))
        for s in np.linspace(0.0, 1.0 - skip_end, n):
            p = a + (b - a) * s
            c = p[2] - self.top(p[0], p[1])
            if c < worst:
                worst, at = c, p
        return float(worst), at


# ── shots ─────────────────────────────────────────────────────────────────────
class Shot:
    """One take. `camera(u, ctx)` returns (eye, target); `beat(u, ctx)` drives
    everything else the shot changes (the SLAM opacity ramp, the PiP zoom)."""

    def __init__(self, name, seconds, camera, walk=True, beat=None, note="",
                 turn_back=False):
        self.name = name
        self.seconds = float(seconds)
        self.frames = int(round(self.seconds * FPS))
        self.camera = camera
        self.walk = bool(walk)
        self.beat = beat
        self.note = note
        # `turn_back` is the film's TIME trigger for the auto-walk's turn-around
        # (spot_calico.turn_back). The walk's own distance trigger usually fires
        # first, on arrival at the last waypoint; this is the backstop that
        # guarantees the robot is turning by the time the sensor beat opens,
        # whatever the gait did on the way up.
        self.turn_back = bool(turn_back)
        self.worst_clearance = 1e9


def build_shots(cams, spine, ctx):
    """The six shots. `cams` maps a cameras_v2 name to (eye, look) in WORLD."""
    est_p, est_l = cams["establishing_v2"]
    fol_p, fol_l = cams["follow_v2"]
    low_p, low_l = cams["low_following_v2"]
    lb_p, lb_l = cams["lookback"]
    spawn = np.asarray(ctx.spawn, np.float64)

    # 1. ESTABLISHING. Three seconds of nothing moving -- the place first, the
    #    robot second -- then a push 35 % of the way down the sightline, with the
    #    look target easing off the wash and onto the robot.
    # 22 %, not 35: at 35 the eye ends 3 m from a standing robot and 43 degrees
    # above it, which is a picture of gravel with a robot in it. At 22 the wash
    # still reads as a place and the robot has grown to something you can see.
    est_end = est_p + (np.asarray(est_l) - est_p) * 0.22
    est_keys = [(0.00, est_p, est_l),
                (0.25, est_p, est_l),
                (1.00, est_end, spawn + [0, 0, 0.5])]

    def cam_establishing(u, c):
        return keyframes(est_keys, u)

    # 2. FOLLOW. A chase at a FIXED offset in the robot's own frame, lifted
    #    whenever that offset pose lands inside the bank. The wash is barely a
    #    metre wide in places (WP0: keep lateral offsets <= 0.55 m), so the
    #    offset is almost straight behind.
    def cam_follow(u, c):
        p, yaw = c.robot()
        f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        left = np.array([-f[1], f[0], 0.0])
        back = 3.0 - 0.6 * smoothstep(u)          # closes in slowly
        eye = p - f * back + left * 0.45 + [0, 0, 1.05]
        eye = c.lift(eye, p + [0, 0, 0.35])
        eye = c.smooth("follow", eye, 0.12)
        return eye, p + [0, 0, 0.35]

    # 3. TRACKSIDE. The camera does not move at all; the robot walks through
    #    frame. low_following_v2, verified by render in WP0's follow-up.
    def cam_trackside(u, c):
        p, _ = c.robot()
        return low_p, ease(low_l, p + [0, 0, 0.3], 0.0 if u < 0.15 else (u - 0.15) / 0.85)

    # 4. LOOKBACK, the hero. Static eye at WP0's -x end; the robot walks toward
    #    it; the look target eases from the wash onto the robot as it arrives.
    #    The SLAM map fades in from 3 s over 2 s.
    def cam_lookback(u, c):
        p, _ = c.robot()
        return lb_p, ease(lb_l, p + [0, 0, 0.35], min(1.0, u / 0.6))

    def beat_lookback(u, c):
        t = u * 14.0                              # seconds into the shot
        c.set_slam_fade(min(1.0, max(0.0, (t - 3.0) / 2.0)))

    # 5. SENSOR BEAT. The eye BACKS OFF instead of pushing in -- by this point
    #    the robot has walked the whole trail and is a couple of metres from the
    #    lens -- and the PiP grows to a quarter of the FRAME AREA: at 1280x720
    #    the panel is already 448x336 = 0.16 of the frame, so a quarter is
    #    zoom 1.25, not the 1.8 the first cut used (which buried the robot behind
    #    its own instrument).
    sen_end = lb_p - (np.asarray(lb_l) - lb_p) * 0.10
    PIP_QUARTER = 1.25

    def cam_sensor(u, c):
        p, _ = c.robot()
        return ease(lb_p, sen_end, u), p + [0, 0, 0.35]

    def beat_sensor(u, c):
        c.set_slam_fade(1.0)
        c.set_pip_zoom(1.0 + (PIP_QUARTER - 1.0) * smoothstep(u))

    # 6. FINAL. Hold everything.
    def cam_final(u, c):
        p, _ = c.robot()
        return sen_end, p + [0, 0, 0.35]

    def beat_final(u, c):
        c.set_slam_fade(1.0)
        c.set_pip_zoom(PIP_QUARTER)

    return [
        Shot("01_establishing", 14.0, cam_establishing, walk=False,
             note="establishing_v2, robot standing at spawn_v2: 3 s hold, then a "
                  "35 % push-in with the look easing onto the robot"),
        Shot("02_follow", 11.0, cam_follow, walk=True,
             note="follow_v2 geometry as a live chase: 3.0 -> 2.4 m behind, 0.45 m "
                  "left, 1.05 m up, lifted out of the bank when the column says so"),
        Shot("03_trackside", 7.0, cam_trackside, walk=True,
             note="low_following_v2, camera static, robot walks through frame"),
        Shot("04_lookback", 14.0, cam_lookback, walk=True, beat=beat_lookback,
             note="the hero: robot walks toward the camera, SLAM map fades in at 3 s"),
        Shot("05_sensor", 9.0, cam_sensor, walk=True, beat=beat_sensor,
             turn_back=True,
             note="sensor beat: the robot has reached the end of the spine, turns "
                  "around and walks back down the wash; the depth PiP grows to a "
                  "quarter of the frame, the map fully on"),
        Shot("06_hold", 6.0, cam_final, walk=True, beat=beat_final,
             note="hold on the lookback: the robot walks away down the trail it "
                  "just mapped, with map and PiP"),
    ]


# ── per-frame bookkeeping ─────────────────────────────────────────────────────
class FrameLog:
    """sensors.csv, the per-frame pixel hashes, and hashes.txt."""

    def __init__(self, out_dir):
        self.dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.csv_path = os.path.join(out_dir, "sensors.csv")
        self.csv = open(self.csv_path, "w", encoding="utf-8", newline="\n")
        self.csv.write("frame,shot,t_film,t_sim,x,y,z,qx,qy,qz,qw,pip_sha,"
                       + ",".join(f"s{i:02d}" for i in range(45)) + "\n")
        self.frames = []            # (name, sha256 of raw pixels)

    def row(self, i, shot, t_film, t_sim, rs, scan, pip_depth):
        ph = "-" if pip_depth is None else hashlib.sha256(
            np.ascontiguousarray(pip_depth, np.float32).tobytes()).hexdigest()[:16]
        self.csv.write(
            f"{i},{shot},{t_film:.4f},{t_sim:.4f},"
            + ",".join(f"{float(v):.6f}" for v in rs[:7]) + f",{ph},"
            + ",".join(f"{float(v):.6f}" for v in np.asarray(scan).ravel()[:45]) + "\n")

    def frame(self, name, pixels):
        self.frames.append((name, hashlib.sha256(
            np.ascontiguousarray(pixels).tobytes()).hexdigest()))

    def close(self):
        self.csv.close()
        with open(self.csv_path, "rb") as f:
            csv_sha = hashlib.sha256(f.read()).hexdigest()
        roll = hashlib.sha256()
        lines = []
        for name, sha in self.frames:
            roll.update(sha.encode())
            lines.append(f"{sha}  {name}")
        lines.append(f"{csv_sha}  sensors.csv")
        lines.append(f"{roll.hexdigest()}  ALL-FRAMES")
        with open(os.path.join(self.dir, "hashes.txt"), "w", encoding="utf-8",
                  newline="\n") as f:
            f.write("\n".join(lines) + "\n")
        return roll.hexdigest(), csv_sha


# ── ffmpeg ────────────────────────────────────────────────────────────────────
def stitch(out_dir, fps=FPS):
    """OUT_DIR/*.png -> OUT_DIR/film.mp4 if ffmpeg is on PATH. Returns the path."""
    import shutil
    import subprocess
    exe = shutil.which("ffmpeg")
    if not exe:
        # No ffmpeg on PATH, but imageio-ffmpeg ships one and is installed in
        # this Python. It is the same binary; use it rather than stopping at a
        # PNG sequence.
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
            print(f"[film] ffmpeg from imageio-ffmpeg: {exe}")
        except Exception:
            print("[film] no ffmpeg on PATH -- PNG sequence only")
            return None
    mp4 = os.path.join(out_dir, "film.mp4")
    cmd = [exe, "-y", "-framerate", str(fps), "-i", os.path.join(out_dir, "f%05d.png"),
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-r", str(fps), mp4]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("[film] ffmpeg failed:\n" + r.stderr[-1500:])
        return None
    print(f"[film] {mp4}  {os.path.getsize(mp4) / 1e6:.1f} MB")
    return mp4
