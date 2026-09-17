"""Chase-cam film: a quadcopter flying the Geiranger wall with the fly-eye panels as insets.

    py -3.14 python/examples/flyeye/film_chase.py --pilot      # short flight, stills of the chase view and each eye
    py -3.14 python/examples/flyeye/film_chase.py              # the full clip

Scene, lighting, streaming warm-up, eyes and truth exactly as scenarios.py (imported, not
modified) except the water: the benchmark's flat matte plane is replaced by a glassy FFT ocean
(fjord_ocean below, warp_netpen.py's recipe over a GeoScene pack at a 3 m/s wind); --flat-water
puts the plane back for comparison. The primary render is a 1920x1080, 70 deg chase camera in
world space, smoothed 4 m behind and 1.4 m above the vehicle (5 m / 1.7 m in the dive), wide
enough that the eye insets' content can be found inside the chase frame. The vehicle Group
carries the drone_rig.Drone model (its +Z forward turned to the vehicle's -Z, rotors ticked every
frame), the two 403 px, 90 deg eyes yawed +45/-45, mounted EYE_AHEAD m ahead of the hull nose so
the airframe is outside both fields of view, and one coloured frustum wedge per eye (cyan L,
orange R, the inset border colours) that only the chase view renders.

Flight: hover, approach the wall at 14 m/s 45 m above the highest ground under the path (277 m
above the fjord), brake, yaw 90 deg left, then dive at 15 m/s vertical (12 m/s forward) to 25 m
above the water and fly level for 3.5 s; the header states the altitude above the sea level every
frame and flight() audits the terrain clearance (the run aborts under 15 m AGL).

Circuit per frame: receptors -> OpticLobe (member 000, both eyes batched) -> the repaired readout
of repair.py (causal adaptation tau 2 s per column and subtype, then the 8 -> 2 map fitted on
the Phase 2 bright recordings of this scene, scores.json "rec adapt2+V2") -> RotationReadout
(matched) and Looming. Truth from the eyes' motion AOV via truth.AovFlow (box, flush 1).

Output 1920x1080, 30 fps, H.264 yuv420p crf 18; every --stride-th 100 Hz frame is written, so the
playback factor is 30 * stride / 100 (stride 2 = 0.6x, stride 1 = 0.3x), stated on screen.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
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
from threepp.torch_frames import FrameTensors  # noqa: E402

from drone_rig import Drone  # noqa: E402
from flyeye import scenarios as S  # noqa: E402
from flyeye.eye import EyeView, configure_sensor_renderer  # noqa: E402
from flyeye.lattice import HexLattice  # noqa: E402
from flyeye.optic_lobe import OpticLobe  # noqa: E402
from flyeye.readouts import Looming, MotionField, RotationReadout  # noqa: E402
from flyeye.sanity_rig import DT, FOV, SIZE, YAWS, rot  # noqa: E402
from flyeye.truth import AovFlow  # noqa: E402
from flyeye.video import VideoOut  # noqa: E402

OUT = Path(r"C:\dev\_flyeye\renders\linkedin")
SCORES = Path(r"C:\dev\_flyeye\repair\scores.json")
W_PX, H_PX = 1920, 1080
FPS = 30
EYE_AHEAD = 0.40  # m ahead of the vehicle origin (hull nose at 0.085 m); the eyes see nothing behind their own plane
CHASE_FOV = 70.0  # vertical; wide enough that the chase frame holds roughly the union of the two 90 deg eyes
CHASE_BACK, CHASE_UP = 4.0, 1.4
EXTRA_TURN = 0.0  # deg of further yaw at the start of the dive (set from the terrain probe, see flight())
EYE_COLORS = ((0, 230, 255), (255, 150, 30))  # eye L cyan, eye R orange: the wedges on the drone, the inset borders, the swatches
WEDGE_LEN = 1.6  # m along each eye axis for the 90 x 90 deg frustum wedge drawn on the vehicle (2.2 ran off the 70 deg frame)
FONT = r"C:\Windows\Fonts\segoeui.ttf"
FONT_B = r"C:\Windows\Fonts\segoeuib.ttf"


# -- stage ------------------------------------------------------------------------------------------
def fjord_ocean(geo, center, wind_speed=3.0, choppiness=0.5, specular=0.55, fetch=9e3, look="auto"):
    """warp_netpen.py's fjord ocean over a GeoScene pack, made glassy. One 512^2 sheet spans the
    pack (size 1.25 x pack), its vertices packed toward `center` (world x, z) by the mesh warp
    (coef_a 0.07: sub-metre cells at the centre, tens of metres at the pack edge); the three
    cascade tiles are pinned (320 / 40.6 / 2.98 m) so the sea does not scale with the sheet.
    Calm: wind ~3 m/s with a 9 km fetch gives short low chop, choppiness 0.5 keeps the crests
    round, no whitecaps. Fjord water from above: dark green-grey absorption, specular 0.55 so the
    reflected shore is dimmed while the sky mirror and the grazing horizon stay at full Fresnel."""
    size = geo.pack_world_size * 1.25
    ocean = tp.Ocean(size=size, resolution=512, wind_speed=wind_speed, wind_theta=0.6, choppiness=choppiness,
                     fft_size=512, fetch=fetch, look=look)
    ocean.params.foam_amount = 0.0
    ocean.params.tile_size_0 = 320.0
    ocean.params.tile_size_1 = 40.6
    ocean.params.tile_size_2 = 2.98
    ocean.warp.center_x = float(center[0])
    ocean.warp.center_z = float(center[1])
    ocean.warp.half_range = size * 0.5
    ocean.warp.coef_a = 0.07
    ocean.material.attenuation_color = tp.Color(0.16, 0.30, 0.24)
    ocean.material.attenuation_distance = 2.5
    ocean.material.specular_intensity = specular
    ocean.position.y = geo.sea_level
    return ocean


class ChaseStage(S.Stage):
    """scenarios.Stage with a 1920x1080 world-space chase camera and the drone model on the vehicle.
    water: dict of fjord_ocean keyword overrides, or None for the benchmark's flat matte plane."""

    def __init__(self, water: dict | None = None):
        self.canvas = tp.Canvas("flyeye chase", width=W_PX, height=H_PX, headless=True, vsync=False)
        r = self.renderer = tp.VulkanRenderer(self.canvas, flush_frames=1)
        self.settings = configure_sensor_renderer(r)
        for name in ("deferred_ao", "probe_gi", "volumetric_fog"):
            try:
                setattr(r, name, False)
            except Exception:
                pass
        self.set_tone_mapping(S.TONE_MAPPING)
        self.scene = tp.Scene()
        self.sky = {k: tp.float_texture(S.sky_rgba(S.SUN_DIR, lv * S.SKY_GAIN)) for k, lv in S.LIGHTS.items()}
        self.sun = tp.DirectionalLight(tp.Color(1.0, 0.95, 0.86), S.SUN_INTENSITY)
        self.sun.position.set(*(S.SUN_DIR * 1000.0))
        self.scene.add(self.sun)
        t0 = time.perf_counter()
        focus = (float(S.FACE[0] - S.HEADING[0] * 200), 0.0, float(S.FACE[1] - S.HEADING[1] * 200))
        self.geo = tp.GeoScene(str(S.PACK), forest_focus=focus, scatter=False)
        self.scene.add(self.geo)
        print(f"terrain {S.PACK.name} loaded in {time.perf_counter() - t0:.1f} s, {self.geo.stats}, "
              f"pack {self.geo.pack_world_size:.0f} m, sea level {self.geo.sea_level:.2f} m")
        if water is None:
            # the benchmark's water (scenarios.Stage): a flat matte plane at sea level
            wm = tp.MeshStandardMaterial()
            wm.color = tp.Color(0.03, 0.06, 0.07)
            wm.roughness = 0.35
            self.ocean = None
            plane = tp.Mesh(tp.PlaneGeometry(self.geo.pack_world_size, self.geo.pack_world_size), wm)
            plane.rotation.x = -math.pi / 2
            plane.position.y = self.geo.sea_level
            self.scene.add(plane)
        else:
            self.ocean = fjord_ocean(self.geo, (focus[0], focus[2]), **water)
            self.scene.add(self.ocean)
            print(f"ocean: {json.dumps(water)} at y {self.geo.sea_level:.2f}, warp centre "
                  f"({focus[0]:.0f}, {focus[2]:.0f}) half-range {self.ocean.warp.half_range:.0f} m")
        self.vehicle = tp.Group()
        self.scene.add(self.vehicle)
        self.cams = []
        for yaw in YAWS:
            c = tp.PerspectiveCamera(FOV, 1.0, S.NEAR, S.FAR)
            c.rotation.y = math.radians(yaw)
            c.position.set(0.0, 0.0, -EYE_AHEAD)
            self.vehicle.add(c)
            self.cams.append(c)
        # the eyes drawn on the drone for the chase view: one 90 x 90 deg frustum wedge per eye, apex at the
        # eye camera, WEDGE_LEN m along its axis, in the inset colour. Lines are overlay-pass only, so the
        # eyes (secondary views) do not see them.
        self.wedges = []
        for yaw, rgb in zip(YAWS, EYE_COLORS):
            Ry = rot((0, 1, 0), math.radians(yaw))
            apex = np.array([0.0, 0.0, -EYE_AHEAD])
            corners = [apex + Ry @ (np.array([sx, sy, -1.0]) * WEDGE_LEN) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            pts = []
            for c in corners:
                pts += [apex, c]
            for i in range(4):
                pts += [corners[i], corners[(i + 1) % 4]]
            g = tp.BufferGeometry()
            g.set_from_points([tp.Vector3(*[float(v) for v in p]) for p in pts])
            m = tp.LineBasicMaterial()
            m.color = tp.Color(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
            wedge = tp.LineSegments(g, m)
            self.vehicle.add(wedge)
            self.wedges.append(wedge)
        # the drone model: its authoring frame is forward +Z, the vehicle's forward is -Z
        self.holder = tp.Group()
        self.holder.rotation.y = math.pi
        self.vehicle.add(self.holder)
        self.drone = Drone(self.holder)
        self.drone.root.position.set(0.0, 0.0, 0.0)
        self.drone.root.rotation.set(0.0, 0.0, 0.0)
        self.main_cam = tp.PerspectiveCamera(CHASE_FOV, W_PX / H_PX, S.NEAR, S.FAR)
        self.scene.add(self.main_cam)
        self.t, self.pos = 0.0, np.zeros(3)
        self.set_light("bright")
        p0 = np.array([S.FACE[0], self.ground(0.0) + S.AGL, S.FACE[1]])
        R0 = S.heading_rotation(S.HEADING)
        self.pose(p0, R0)
        self.chase(p0 - R0[:, 2] * -CHASE_BACK + np.array([0, CHASE_UP, 0]), p0)
        self.render()
        self.eyes = [EyeView(r, c, SIZE) for c in self.cams]
        self.render()
        for e in self.eyes:
            e.arm()
        self.frames = FrameTensors(r, 0, ("color",))
        self.render()
        if not self.frames.bgra:
            raise RuntimeError("primary colour export is RGBA, expected BGRA")

    def chase(self, eye, aim):
        self.main_cam.position.set(*[float(v) for v in eye])
        self.main_cam.look_at(tp.Vector3(*[float(v) for v in aim]))

    def attribution(self):
        try:
            a = self.geo.attribution
            return a() if callable(a) else str(a)
        except Exception:
            return json.loads((S.PACK / "region.json").read_text())["attribution"]


# -- flight -----------------------------------------------------------------------------------------
def smooth01(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def flight(stage: S.Stage, cruise=14.0, start=320.0, stop=150.0, t_rest=1.0, t_acc=1.0, t_brake=1.5,
           v_turn=0.0, turn_rate=90.0, t_turn=1.15, v_out=12.0, t_out=3.5, dive_rate=15.0, alt_low=25.0,
           t_dive_ramp=3.0, extra_turn=0.0, pilot=False):
    """Approach the wall at `cruise` m/s from `start` m, brake to v_turn (0 = hover) by `stop` m, yaw
    left at turn_rate deg/s (0.15 s ramps, so 90 deg over 1.15 s) for t_turn s. Then, over the fjord:
    accelerate to v_out over 1 s while the vertical rate ramps (smoothstep, t_dive_ramp s) to
    dive_rate m/s down, hold it, and ramp out so the dive ends alt_low m above stage.geo.sea_level;
    extra_turn deg of further yaw at turn_rate (0.15 s ramps) at the start of the dive bends the
    heading along the fjord. Then fly level for t_out s.
    -> dict(pos, R, w, v, vy, speed, labels, info). Body frame as scenarios.py; exact integration.
    info carries the clearance audit: min_agl (all frames), forward range to terrain at the end of the
    level flight and the minimum forward range sampled every 0.5 s from the start of the dive."""
    R0 = S.heading_rotation(S.HEADING)
    fwd = np.array([S.HEADING[0], 0.0, S.HEADING[1]])
    hit = 0.0
    for _ in range(3):
        y = max(stage.ground(s) for s in np.arange(hit + stop - 5, hit + start + 20, 2.0)) + S.CLEAR
        probe = np.array([S.FACE[0] - S.HEADING[0] * 250, y, S.FACE[1] - S.HEADING[1] * 250])
        hit = 250.0 - stage.range_geom(probe, fwd)
    if pilot:
        start = stop + 30.0
    s0 = hit + start
    d_straight = start - stop
    d_acc = 0.5 * cruise * t_acc
    d_brake = 0.5 * (cruise + v_turn) * t_brake
    t_cruise = max(0.0, (d_straight - d_acc - d_brake) / cruise)
    sea = float(stage.geo.sea_level)
    drop = y - (sea + alt_low)
    t_flat = max(0.0, drop / dive_rate - t_dive_ramp)  # smoothstep ramps integrate to half their length
    t_dive = t_flat + 2 * t_dive_ramp
    t_extra = extra_turn / turn_rate + 0.15 if extra_turn > 0 else 0.0
    segs = [("hover", t_rest), (f"accelerate to {cruise:g} m/s", t_acc), (f"approach {cruise:g} m/s", t_cruise),
            (f"brake to {v_turn:g} m/s" if v_turn > 0 else "brake to hover", t_brake),
            (f"yaw turn {turn_rate:g} deg/s", t_turn),
            (f"dive to {alt_low:g} m, {dive_rate:g} m/s down, {v_out:g} m/s forward", t_dive),
            (f"level flight {v_out:g} m/s, {alt_low:g} m above the fjord", t_out)]
    bounds, k = [], 0
    for name, T in segs:
        n = int(round(T / DT))
        bounds.append((name, k, k + n))
        k += n
    n = k
    speed, yaw, vy = np.zeros(n), np.zeros(n), np.zeros(n)
    for name, a, b in bounds:
        u = (np.arange(b - a) + 0.5) / max(b - a, 1)
        t = (np.arange(b - a) + 0.5) * DT
        if name.startswith("accelerate to " + f"{cruise:g}"):
            speed[a:b] = cruise * smooth01(u)
        elif name.startswith("approach"):
            speed[a:b] = cruise
        elif name.startswith("brake"):
            speed[a:b] = cruise + (v_turn - cruise) * smooth01(u)
        elif name.startswith("yaw"):
            speed[a:b] = v_turn
            yaw[a:b] = math.radians(turn_rate) * np.array([S.ramp((i + 0.5) * DT, (b - a) * DT, 0.15) for i in range(b - a)])
        elif name.startswith("dive"):
            T = (b - a) * DT
            speed[a:b] = v_turn + (v_out - v_turn) * smooth01(t / 1.0)
            vy[a:b] = -dive_rate * np.minimum(smooth01(t / t_dive_ramp), smooth01((T - t) / t_dive_ramp))
            if t_extra > 0:
                yaw[a:b] = math.radians(turn_rate) * np.array([S.ramp(tt, t_extra, 0.15) for tt in t])
        elif name.startswith("level"):
            speed[a:b] = v_out
    w = np.stack([np.zeros(n), yaw, np.zeros(n)], 1)
    v = np.stack([np.zeros(n), vy, -speed], 1)  # yaw-only R: body y is world y
    pos0 = np.array([S.FACE[0] - S.HEADING[0] * s0, y, S.FACE[1] - S.HEADING[1] * s0])
    pos, R = S.integrate(pos0, R0, w, v)
    agl = np.array([p[1] - stage.geo.height_at(float(p[0]), float(p[2])) for p in pos])
    k_dive = bounds[5][1]
    ranges = []
    for k in range(k_dive, n, 50):
        ranges.append(stage.range_geom(pos[k], -R[k][:, 2]))
    ranges.append(stage.range_geom(pos[-1], -R[-1][:, 2]))
    ranges = np.array(ranges, float)
    ranges = np.where(np.isnan(ranges), np.inf, ranges)
    info = dict(altitude=float(y), sea_level=sea, alt_low=alt_low, wall_s=float(hit), start_range=start, stop_range=stop,
                cruise=cruise, v_out=v_out, dive_rate=dive_rate, t_dive=t_dive, extra_turn=extra_turn,
                min_agl=float(agl.min()), min_agl_frame=int(agl.argmin()), min_agl_t=float(agl.argmin() * DT),
                end_alt_above_sea=float(pos[-1][1] - sea), end_forward_range=float(ranges[-1]),
                min_forward_range_from_dive=float(ranges.min()), frames=n, seconds=n * DT,
                peak_yaw_deg_s=float(np.degrees(yaw.max())), turn_deg=float(np.degrees(yaw.sum() * DT)))
    return dict(pos=pos, R=R, w=w, v=v, vy=vy, speed=speed, labels=bounds, info=info)


def drone_attitude(speed, w, tau=0.12, vy=None, dive_rate=15.0):
    """Cosmetic airframe pitch/roll (rad) per frame from the body-frame profile: nose down by
    atan(a_fwd / g) plus a cruise trim, a further 12 deg nose down at the full dive rate, bank into
    the yaw turn by atan(v w_yaw / g). The eyes and the truth stay on the vehicle frame; only the
    model tilts."""
    n = len(speed)
    a_fwd = np.gradient(speed, DT)
    dive = np.zeros(n) if vy is None else np.clip(-np.asarray(vy) / dive_rate, 0.0, 1.0)
    pitch = np.clip(np.arctan2(a_fwd, 9.81) + np.radians(6.0) * speed / max(speed.max(), 1e-6) + np.radians(12.0) * dive,
                    -0.45, 0.45)
    roll = np.clip(np.arctan2(speed * w[:, 1], 9.81), -0.6, 0.6)
    k = 1 - math.exp(-DT / tau)
    out = np.zeros((n, 2))
    p = r = 0.0
    for i in range(n):
        p += (pitch[i] - p) * k
        r += (roll[i] - r) * k
        out[i] = (p, r)
    return out


class ChaseCam:
    """Behind and above the vehicle, smoothed, aimed a little ahead and off-axis so the drone sits
    small in the lower middle of the free area (the insets take the bottom and the right) and the
    70 deg frame holds most of what the two 90 deg eyes see."""

    def __init__(self, back=CHASE_BACK, up=CHASE_UP, tau_pos=0.45, tau_aim=0.2):
        self.back, self.up, self.tau_pos, self.tau_aim = back, up, tau_pos, tau_aim
        self.eye = self.aim = self.fwd = None

    def __call__(self, pos, R, back=None, up=None, ahead=3.0):
        """back / up (m) override the construction framing for this frame (the dive and the low
        flight pull the camera back and up so the water and the wall stay in frame with the drone);
        ahead is the aim point distance along the smoothed heading. The eye and aim targets are
        smoothed with tau_pos / tau_aim, so a change of framing eases in."""
        back = self.back if back is None else back
        up = self.up if up is None else up
        fwd = -R[:, 2]
        right = R[:, 0]
        if self.fwd is None:
            self.fwd = fwd.copy()
        self.fwd += (fwd - self.fwd) * (1 - math.exp(-DT / self.tau_pos))
        self.fwd /= np.linalg.norm(self.fwd)
        f = self.fwd
        rgt = np.cross(f, [0, 1, 0])
        rgt /= np.linalg.norm(rgt)
        eye_t = -f * back + np.array([0, up, 0]) + rgt * 0.30
        aim_t = f * ahead + rgt * 0.42 - np.array([0, 0.58, 0])
        if self.eye is None:
            self.eye, self.aim = eye_t.copy(), aim_t.copy()
        self.eye += (eye_t - self.eye) * (1 - math.exp(-DT / self.tau_pos))
        self.aim += (aim_t - self.aim) * (1 - math.exp(-DT / self.tau_aim))
        return pos + self.eye, pos + self.aim


# -- readout ----------------------------------------------------------------------------------------
class RepairedReadout:
    """repair.recommended() online: causal EMA adaptation (tau s) of the 8 rest-subtracted rates per
    column, then the fitted 8 -> 2 map W (scores.json, member 000, bright runs of this scene)."""

    def __init__(self, motion: MotionField, W, tau=2.0):
        self.motion = motion
        self.W = torch.as_tensor(np.asarray(W, np.float32), device="cuda")  # (8, 2)
        self.a = DT / tau
        self.m = None

    def __call__(self, v):
        raw = self.motion.raw(v).float()  # (B, 8, 721)
        if self.m is None:
            self.m = raw.clone()
        self.m += self.a * (raw - self.m)
        return torch.einsum("bkn,kc->bcn", raw - self.m, self.W), raw


# -- composition ------------------------------------------------------------------------------------
class Composer:
    P = 250  # hex panel px per eye
    PAIR = 2 * 250 + 4
    STRIP_Y = 770
    COL_X = 1620
    EYE = 300

    def __init__(self, lattice: HexLattice, path: Path, scales: dict, fps=FPS):
        import imageio_ffmpeg
        from PIL import Image, ImageDraw, ImageFont

        self.Image, self.ImageDraw = Image, ImageDraw
        self.f = {k: ImageFont.truetype(FONT, k) for k in (22, 24, 26, 28, 30, 32, 36, 40)}
        self.fb = {k: ImageFont.truetype(FONT_B, k) for k in (28, 32, 40, 48, 60, 72)}
        self.scales = scales
        P = self.P
        rc = (lattice.pixel_rc(SIZE).float() - SIZE / 2) * (P / SIZE) + P / 2
        rc = rc.cuda()
        yy, xx = torch.meshgrid(torch.arange(P, device="cuda"), torch.arange(P, device="cuda"), indexing="ij")
        pix = torch.stack([yy.flatten(), xx.flatten()], 1).float() + 0.5
        d, i = zip(*(torch.cdist(c, rc).min(1) for c in pix.split(16384)))
        self.index = torch.cat(i).view(P, P)
        self.inside = (torch.cat(d) <= 8.5 * P / SIZE).view(P, P)
        # legend disc: hue by angle, brightness by radius
        L = 100
        ly, lx = torch.meshgrid(torch.arange(L, device="cuda"), torch.arange(L, device="cuda"), indexing="ij")
        fx, fy = (lx.float() + 0.5 - L / 2), -(ly.float() + 0.5 - L / 2)
        disc = VideoOut.field_rgb(torch.stack([fx.flatten(), fy.flatten()]), L / 2 * 0.92).view(L, L, 3)
        self.disc = disc * ((fx * fx + fy * fy) <= (L / 2) ** 2)[..., None]
        self.proc = subprocess.Popen(
            [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", f"{W_PX}x{H_PX}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)], stdin=subprocess.PIPE)
        self.n_written = 0
        self.last = None

    # text helpers
    def box_text(self, dr, xy, text, font, fill=(255, 255, 255), pad=8, bg=(8, 9, 12)):
        x, y = xy
        l, t, r, b = dr.textbbox((x, y), text, font=font)
        dr.rectangle((l - pad, t - pad, r + pad, b + pad), fill=bg)
        dr.text((x, y), text, font=font, fill=fill)
        return b + pad

    def hex(self, img, x0, y0, rgb):
        """rgb (721, 3) uint8 -> the P x P hex image at (x0, y0)."""
        P = self.P
        img[y0:y0 + P, x0:x0 + P] = rgb[self.index] * self.inside[..., None]

    def eye_rgb(self, color, size):
        c = color[..., :3][..., [2, 1, 0]].float().permute(2, 0, 1)[None]
        c = torch.nn.functional.interpolate(c, size=(size, size), mode="bilinear", antialias=True, align_corners=False)
        return c[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)

    def frame(self, chase, eyes, receptors, field, truth, gauges: dict, texts: dict):
        """Compose one frame on the GPU, draw text on the CPU, write it."""
        sc = self.scales
        img = chase[..., :3][..., [2, 1, 0]].clone()  # (1080, 1920, 3)
        # backing boxes
        img[self.STRIP_Y:, :] = 10
        img[:, self.COL_X:] = 10
        # right column: eye renders
        for e in range(2):
            y0 = e * (self.EYE + 4)
            img[y0:y0 + self.EYE, self.COL_X:self.COL_X + self.EYE] = self.eye_rgb(eyes[e], self.EYE)
        # legend disc
        img[616:716, 1628:1728] = self.disc
        # bottom strip: three hex pairs
        P, y0 = self.P, self.STRIP_Y + 56
        for i, (kind, src, full) in enumerate((("rec", receptors, None), ("field", field, sc["field_full"]),
                                               ("truth", truth, sc["truth_full"]))):
            x0 = 20 + i * (self.PAIR + 20)
            for e in range(2):
                if kind == "rec":
                    g = (src[e].clamp(0, 1) * 255).to(torch.uint8)[:, None].expand(-1, 3)
                else:
                    g = VideoOut.field_rgb(src[e].float(), full)
                self.hex(img, x0 + e * (P + 4), y0, g)
        im = self.Image.fromarray(img.cpu().numpy())
        dr = self.ImageDraw.Draw(im)
        # header
        y = self.box_text(dr, (24, 16), texts["title"], self.fb[40])
        self.box_text(dr, (24, y + 6), texts["sub"], self.f[26])
        y = self.box_text(dr, (24, 118), texts["seg"], self.f[32], fill=(255, 220, 120))
        if texts.get("alt"):
            self.box_text(dr, (24, y + 6), texts["alt"], self.f[32], fill=(255, 220, 120))
        # time, right-aligned in the chase area
        t = texts["time"]
        w = dr.textlength(t, font=self.f[30])
        self.box_text(dr, (self.COL_X - 24 - w, 16), t, self.f[30])
        # eye labels, a colour swatch and a border in the eye's colour (matches its wedge on the drone)
        for e, (yaw, col) in enumerate(zip(YAWS, EYE_COLORS)):
            y0 = e * (self.EYE + 4)
            dr.rectangle((self.COL_X, y0 + self.EYE - 34, self.COL_X + self.EYE, y0 + self.EYE), fill=(8, 9, 12))
            dr.rectangle((self.COL_X + 10, y0 + self.EYE - 27, self.COL_X + 30, y0 + self.EYE - 7), fill=col)
            dr.text((self.COL_X + 40, y0 + self.EYE - 32), f"eye {'L' if e == 0 else 'R'}   yaw {yaw:+.0f}   90 deg",
                    font=self.f[22], fill=(255, 255, 255))
            dr.rectangle((self.COL_X, y0, self.COL_X + self.EYE - 1, y0 + self.EYE - 1), outline=col, width=5)
        # legend text
        dr.text((1736, 626), "hue = direction", font=self.f[22], fill=(255, 255, 255))
        dr.text((1736, 654), "(red = right,", font=self.f[22], fill=(200, 200, 200))
        dr.text((1736, 680), "green = up)", font=self.f[22], fill=(200, 200, 200))
        dr.text((1736, 706), "bright = fast", font=self.f[22], fill=(255, 255, 255))
        # panel labels
        labels = (("721 receptor columns per eye (L, R)", "BoxEye luminance, what the circuit sees"),
                  ("T4/T5 opponent motion field", f"readout calibrated on this scene, full {sc['field_full']:.2f} a.u."),
                  ("true flow, renderer ground truth", f"motion AOV per column, full {sc['truth_full']:.2f} rad/s"))
        for i, (l1, l2) in enumerate(labels):
            x0 = 20 + i * (self.PAIR + 20)
            dr.text((x0, self.STRIP_Y + 4), l1, font=self.f[24], fill=(255, 255, 255))
            dr.text((x0, self.STRIP_Y + 30), l2, font=self.f[22], fill=(200, 200, 205))
        # gauges
        gx, gy = self.COL_X + 12, self.STRIP_Y + 4
        dr.text((gx, gy), "wide-field rotation, deg/s", font=self.f[24], fill=(255, 255, 255))
        dr.text((gx, gy + 26), "white = true (scripted)", font=self.f[22], fill=(200, 200, 205))
        dr.text((gx, gy + 50), "orange = circuit (0.1 s EMA),", font=self.f[22], fill=(255, 160, 40))
        dr.text((gx, gy + 74), "calibrated on this scene", font=self.f[22], fill=(255, 160, 40))
        full = 120.0
        bw = 276
        for i, name in enumerate(("yaw", "pitch", "roll")):
            ry = gy + 104 + i * 48
            tv, cv = gauges["true"][i], gauges["circuit"][i]
            dr.text((gx, ry), name, font=self.f[22], fill=(255, 255, 255))
            dr.text((gx + 70, ry), f"{tv:+6.1f}", font=self.f[22], fill=(255, 255, 255))
            dr.text((gx + 170, ry), f"{cv:+6.1f}", font=self.f[22], fill=(255, 160, 40))
            cx = gx + bw // 2
            for j, (val, col) in enumerate(((tv, (255, 255, 255)), (cv, (255, 160, 40)))):
                by = ry + 26 + j * 9
                dr.rectangle((gx, by, gx + bw, by + 7), fill=(40, 42, 48))
                px = int(np.clip(val / full, -1, 1) * bw / 2)
                dr.rectangle((min(cx, cx + px), by, max(cx, cx + px), by + 7), fill=col)
            dr.line((cx, ry + 24, cx, ry + 46), fill=(120, 120, 130))
        ry = gy + 104 + 3 * 48
        lt, lc = gauges["loom_true"], gauges["loom_circuit"]
        dr.text((gx, ry), "loom", font=self.f[22], fill=(255, 255, 255))
        dr.text((gx + 58, ry), f"{lt:5.3f} rad/s", font=self.f[22], fill=(255, 255, 255))
        dr.text((gx + 178, ry), f"{lc:5.3f} a.u.", font=self.f[22], fill=(255, 160, 40))
        for j, (val, fullv, col) in enumerate(((lt, sc["loom_truth_full"], (255, 255, 255)),
                                                (lc, sc["loom_circuit_full"], (255, 160, 40)))):
            by = ry + 26 + j * 9
            dr.rectangle((gx, by, gx + bw, by + 7), fill=(40, 42, 48))
            dr.rectangle((gx, by, gx + int(np.clip(val / fullv, 0, 1) * bw), by + 7), fill=col)
        self.last = im
        self.proc.stdin.write(im.tobytes())
        self.n_written += 1
        return im

    def card(self, lines, seconds, attribution=None):
        """A dark card: lines = [(text, font_size, bold, colour)], centred, held for seconds."""
        im = self.Image.new("RGB", (W_PX, H_PX), (12, 14, 18))
        dr = self.ImageDraw.Draw(im)
        fonts = [self.fb[s] if b else self.f[s] for _, s, b, _ in lines]
        heights = [dr.textbbox((0, 0), t, font=f)[3] + 22 for (t, _, _, _), f in zip(lines, fonts)]
        y = (H_PX - sum(heights)) // 2 - 20
        for (t, _, _, col), f, h in zip(lines, fonts, heights):
            w = dr.textlength(t, font=f)
            dr.text(((W_PX - w) / 2, y), t, font=f, fill=col)
            y += h
        if attribution:
            dr.text((24, H_PX - 44), attribution, font=self.f[22], fill=(150, 150, 160))
        buf = im.tobytes()
        for _ in range(int(round(seconds * FPS))):
            self.proc.stdin.write(buf)
            self.n_written += 1
        return im

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


# -- main -------------------------------------------------------------------------------------------
def eye_png(eye, path):
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(eye.color[..., [2, 1, 0]].cpu().numpy())).save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", action="store_true", help="short flight, stills only, no cards")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--name", default=None)
    ap.add_argument("--field-full", type=float, default=0.30, help="repaired field a.u. at full brightness (p95 on approach_bright)")
    ap.add_argument("--truth-full", type=float, default=1.0, help="true flow rad/s at full brightness")
    ap.add_argument("--stride", type=int, default=2, help="write every stride-th 100 Hz frame at 30 fps")
    ap.add_argument("--flat-water", action="store_true", help="the benchmark's flat matte water plane instead of the FFT ocean")
    ap.add_argument("--wind", type=float, default=3.0, help="ocean wind speed m/s (glassy fjord: 2.5-4)")
    ap.add_argument("--chop", type=float, default=0.5, help="ocean choppiness (0.45 realistic, >= 0.8 folds crests)")
    ap.add_argument("--specular", type=float, default=0.55, help="ocean specular_intensity (dims the reflected shore and the sun glint)")
    ap.add_argument("--look", default="auto", choices=("auto", "ocean", "pond", "fjord"),
                    help="tp.Ocean material recipe; fjord adds volume scattering (glacial turquoise)")
    ap.add_argument("--extra-turn", type=float, default=EXTRA_TURN,
                    help="further yaw (deg) at the start of the dive, bending the heading along the fjord")
    ap.add_argument("--alt-low", type=float, default=25.0, help="altitude above the sea level at the end of the dive, m")
    ap.add_argument("--dive-rate", type=float, default=15.0, help="vertical rate in the dive, m/s")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    name = args.name or ("chase_pilot" if args.pilot else "flyeye_chase_geiranger")
    scores = json.load(open(SCORES))["rows"]["000"]["rec adapt2+V2"]
    Wmap = np.array(scores["fitted"]["W"], np.float32)
    gain = scores["gain_by_rate"]["bright"]  # a.u. per rad/s at 30 / 90 / 180 deg/s
    gain90 = np.array([gain["pitch"]["90"], gain["yaw"]["90"], gain["roll"]["90"]])
    offset = np.array(scores["rotation_offsets"]["bright"]["yaw"])
    scales = dict(field_full=args.field_full, truth_full=args.truth_full, loom_truth_full=0.20, loom_circuit_full=0.06)
    playback = f"{FPS * args.stride / 100:.1f}x"

    water = None if args.flat_water else dict(wind_speed=args.wind, choppiness=args.chop, specular=args.specular, look=args.look)
    stage = ChaseStage(water)
    attribution = stage.attribution()
    print("attribution:", attribution)
    lat = HexLattice()
    R_eyes = np.stack([rot((0, 1, 0), math.radians(y)) for y in YAWS])
    lobe = OpticLobe(device="cuda")
    motion = MotionField(lobe)
    readout = RepairedReadout(motion, Wmap, tau=2.0)
    rot_m = RotationReadout(SIZE, FOV, R_eyes, method="matched")
    loom = Looming(SIZE, FOV, R_eyes)
    flow_box = AovFlow(lat, DT, SIZE, FOV, mode="box")

    fl = flight(stage, pilot=args.pilot, extra_turn=args.extra_turn, alt_low=args.alt_low, dive_rate=args.dive_rate)
    pos, R, w, v, vy, speed, labels = fl["pos"], fl["R"], fl["w"], fl["v"], fl["vy"], fl["speed"], fl["labels"]
    n = len(pos)
    print("flight:", json.dumps(fl["info"]), "segments:", [(nm, a, b) for nm, a, b in labels])
    if fl["info"]["min_agl"] < 15.0:
        raise SystemExit(f"flight path within 15 m of terrain: min AGL {fl['info']['min_agl']:.1f} m at t {fl['info']['min_agl_t']:.2f} s")
    att = drone_attitude(speed, w, vy=vy, dive_rate=args.dive_rate)
    sea = float(stage.geo.sea_level)
    k_dive = labels[5][1]
    # chase framing: CHASE_BACK / CHASE_UP (4 m / 1.4 m) aimed 3 m ahead for the approach, eased over 2 s to
    # 5 m / 1.7 m aimed 3.5 m ahead for the dive and the low flight (water and wall in frame with the drone)
    u_frame = smooth01((np.arange(n) - k_dive) * DT / 2.0)
    cam_back, cam_up, cam_ahead = CHASE_BACK + 1.0 * u_frame, CHASE_UP + 0.3 * u_frame, 3.0 + 0.5 * u_frame
    seg_of = np.zeros(n, dtype=np.int32)
    for i, (_, a, b) in enumerate(labels):
        seg_of[a:b] = i

    # streaming warm-up along the whole path, as scenarios.run
    t0 = time.perf_counter()
    step = max(1, int(10.0 / max(speed.max() * DT, 1e-9)))
    cam = ChaseCam()
    stage.chase(*cam(pos[0], R[0]))
    warm = stage.warm([(pos[k], R[k]) for k in range(n - 1, -1, -step)] + [(pos[0], R[0])])
    for _ in range(60):
        stage.render()
    print(f"warm-up {warm} frames in {time.perf_counter() - t0:.1f} s")

    comp = Composer(lat, args.out / f"{name}.mp4", scales)
    if not args.pilot:
        comp.card([("A fly's optic lobe, flown along a fjord wall", 72, True, (255, 255, 255)),
                   ("threepp renderer  +  flyvis connectome model (Lappalainen et al. 2024)", 40, False, (230, 230, 235)),
                   ("FIB-25 / FIB-19 medulla connectome, member 000, two 90 deg eyes at 100 Hz", 32, False, (200, 200, 205)),
                   (f"playback {playback}", 32, False, (255, 220, 120))], 3.0, attribution)
    stills = {}
    for i, (nm, a, b) in enumerate(labels):
        if nm.startswith("approach"):
            stills["cruise"] = (a + b) // 2
        if nm.startswith("brake"):
            stills["brake"] = (a + b) // 2
        if nm.startswith("yaw"):
            stills["turn"] = (a + b) // 2
        if nm.startswith("dive"):
            stills["dive_start"] = a + 150
            stills["dive"] = (a + b) // 2
            stills["flare"] = b - 40
        if nm.startswith("level"):
            stills["level"] = (a + b) // 2
    if args.pilot:
        stills["first"] = 10
    t0 = time.perf_counter()
    tick = 0.0
    log = []
    circ_s, loom_s = np.zeros(3), 0.0
    k_s = 1 - math.exp(-DT / 0.1)
    for k in range(n):
        stage.pose(pos[k], R[k])
        stage.drone.root.rotation.set(float(att[k, 0]), 0.0, float(att[k, 1]))
        tick += DT
        stage.drone.tick(DT, tick, night=0.0)
        stage.chase(*cam(pos[k], R[k], back=float(cam_back[k]), up=float(cam_up[k]), ahead=float(cam_ahead[k])))
        stage.render()
        x = torch.stack([e.receptors() for e in stage.eyes])
        if k == 0:
            lobe.fade_in(x, DT)
        vv = lobe.step(x, DT)
        field, raw = readout(vv)
        est = rot_m(field).cpu().numpy()
        circuit = (est - offset) / gain90  # rad/s, calibrated on the Phase 2 bright runs of this scene
        lc = float(loom(field)["value"])
        w_cam = R_eyes.transpose(0, 2, 1) @ w[k]
        fb = torch.stack([flow_box(e.motion, e.depth, w_cam[b]) for b, e in enumerate(stage.eyes)]).float()
        lt = float(loom(fb)["value"])
        circ_s += (circuit - circ_s) * k_s
        loom_s += (lc - loom_s) * k_s
        stage.frames.sync()
        seg = labels[seg_of[k]][0]
        gauges = dict(true=np.degrees(w[k][[1, 0, 2]]), circuit=np.degrees(circ_s[[1, 0, 2]]), loom_true=lt, loom_circuit=loom_s)
        alt = float(pos[k][1] - sea)
        texts = dict(title="threepp + flyvis connectome model",
                     sub="Lappalainen et al. 2024, FIB-25 / FIB-19 medulla connectome, member 000",
                     seg=f"{seg}   |v| {math.hypot(speed[k], vy[k]):4.1f} m/s",
                     alt=f"alt {alt:.0f} m above the fjord" + (f"   sink {-vy[k]:.0f} m/s" if vy[k] < -0.5 else ""),
                     time=f"t {k * DT:5.2f} s   playback {playback}   100 Hz")
        if k % args.stride == 0:
            comp.frame(stage.frames.color, [e.color for e in stage.eyes], x, field, fb, gauges, texts)
            for tag, kk in stills.items():
                if kk - args.stride < k <= kk:
                    comp.last.save(args.out / f"{name}_{tag}.png")
                    for e, eye in enumerate(stage.eyes):
                        eye_png(eye, args.out / f"{name}_{tag}_eye{e}.png")
        log.append([k * DT, speed[k], *np.degrees(w[k]).tolist(), *np.degrees(circuit).tolist(), lt, lc,
                    stage.geo.stats["baking"]])
        if k % 500 == 0:
            print(f"frame {k}/{n} {time.perf_counter() - t0:.1f} s baking {stage.geo.stats['baking']}", flush=True)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    if not args.pilot:
        comp.card([("45,669 neurons, 1.5 M synapses", 72, True, (255, 255, 255)),
                   ("two eyes at 324 Hz on an RTX 4070", 40, False, (230, 230, 235)),
                   ("wide-field rotation reads out (R^2 0.9)", 40, False, (230, 230, 235)),
                   ("per-column flow does not (as measured against exact truth)", 40, False, (230, 230, 235)),
                   ("readout calibrated on this scene", 32, False, (255, 220, 120)),
                   ("threepp + flyvis connectome model, Lappalainen et al. 2024", 28, False, (170, 170, 180))],
                  4.0, attribution)
    comp.close()
    np.save(args.out / f"{name}_log.npy", np.array(log))
    meta = dict(name=name, frames=n, written=comp.n_written, seconds_out=comp.n_written / FPS, fps=FPS, dt=DT,
                stride=args.stride, playback=playback,
                flight=fl["info"], labels=[list(x) for x in labels], attribution=attribution, wall_s=wall,
                warm_frames=warm, scales=scales, gain90=gain90.tolist(), offset=offset.tolist(), eye_ahead=EYE_AHEAD,
                stills={k: v for k, v in stills.items()}, max_baking=int(max(r[-1] for r in log)),
                water=water or "flat plane", ms_per_frame=1000.0 * wall / n)
    json.dump(meta, open(args.out / f"{name}.json", "w"), indent=1)
    print(f"{name}: {n} flight frames, {comp.n_written} written ({comp.n_written / FPS:.1f} s), run {wall:.1f} s "
          f"({meta['ms_per_frame']:.1f} ms/frame), max baking {meta['max_baking']}")
    for e in stage.eyes:
        e.close()


if __name__ == "__main__":
    main()
