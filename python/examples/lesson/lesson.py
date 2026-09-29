"""lesson: a small toolkit for making explainer videos with threepp.

A lesson is rendered in two passes:

1. **Choreography.** Everything stateful (an IK solve, a simulation, a trail)
   runs once, front to back, and records per-frame state. No rendering.
2. **Frames.** `render(t)` is a pure function of the recorded state and the
   clock, so any frame can be rendered on its own: a still, a contact sheet,
   a low-res preview of one beat, or the final film.

The pieces, from bottom to top:

`Timeline`, `Keys`, `OrbitCamera`
    Named beats with start/end times. `tl.p("fk", t)` is the eased 0..1
    progress through a beat, `tl.fade("fk", t)` a fade-in/hold/fade-out
    envelope for things that belong to it. `Keys` interpolates keyframed
    vectors; `OrbitCamera` keyframes the camera as azimuth, elevation,
    distance, look point and fov, with a slow drift.

`Stage`
    Canvas + renderer + a dark studio set (key/rim light, soft IBL, a floor that
    dissolves into the background). GL by default so a lesson runs on any
    laptop; `renderer="vulkan"` for the ray-traced look. `frame()` returns the
    picture as an (H, W, 3) uint8 array and `project()` maps a world point to
    pixels so 2D callouts can point at 3D things.

`Arrow3D`, `Ring3D`, `Marker3D`, `Tube3D`
    The 3D annotation kit: thick arrows (shaft + cone), glowing rings around a
    joint axis, a target marker, and a tube trail that is rebuilt from points.
    All fade with `.opacity` and live wherever they are parented, so an arrow
    can ride on a robot link. `set_pose` places any object at a 4x4 numpy pose.

`Hud`
    The 2D layer, also drawn by threepp: an orthographic scene rendered over
    the 3D one. Panels, text (Text2D from the system TTF), LaTeX-style maths
    (matplotlib's mathtext outlines loaded through SVGLoader, no TeX install
    needed, and cached to JSON so a finished lesson needs no matplotlib), lines,
    arrows, bars, a log plot, line plots (`plot`), a panel of live values
    (`readout`), and callouts. The cards every lesson has are widgets:
    `captions`, `title_card`, `equation_card` and `summary`. `ghost()` makes a
    translucent copy of any object, and `Stage.follow()` keeps the key light's
    shadow area on a moving subject.

`Film`
    Pipes frames into ffmpeg (imageio_ffmpeg's binary) as H.264, writing to a
    temporary name and renaming on success so a half-written mp4 never sits at
    the final path.

`run`
    The command line every lesson shares: the film, a preview, one stretch,
    stills, a contact sheet, or the captions as an .srt. A lesson provides
    `setup(width, height) -> (render, captions)`.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import wave
from types import MappingProxyType

import numpy as np
import struct
import zlib

_HERE = os.path.dirname(os.path.abspath(__file__))
_PY = os.path.dirname(os.path.dirname(_HERE))
if _PY not in sys.path:
    sys.path.insert(0, _PY)
import threepp as tp  # noqa: E402


# ── colour ────────────────────────────────────────────────────────────────────
def hex_rgb(h):
    """0xRRGGBB or '#rrggbb' -> (r, g, b) ints."""
    if isinstance(h, str):
        h = int(h.lstrip("#"), 16)
    return (h >> 16) & 255, (h >> 8) & 255, h & 255


def to_hex(h):
    if isinstance(h, str):
        return int(h.lstrip("#"), 16)
    return int(h)


# ── easing & time ─────────────────────────────────────────────────────────────
def clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


# The easings take a scalar or a numpy array (per-point staggers).
def _c01(x):
    return clamp01(x) if np.isscalar(x) else np.clip(x, 0.0, 1.0)


def smooth(x):
    """Smoothstep: zero velocity at both ends."""
    x = _c01(x)
    return x * x * (3.0 - 2.0 * x)


def smoother(x):
    """Smootherstep: zero velocity AND acceleration at both ends."""
    x = _c01(x)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def ease_out(x):
    x = _c01(x)
    return 1.0 - (1.0 - x) ** 3


def ease_out_back(x, s=1.4):
    x = _c01(x) - 1.0
    return 1.0 + x * x * ((s + 1.0) * x + s)


def lerp(a, b, t):
    return a + (b - a) * t


def remap(t, a, b):
    """Linear 0..1 progress of t through [a, b], clamped."""
    return clamp01((t - a) / max(b - a, 1e-9))


class Timeline:
    """Named beats. Times are seconds from the start of the film."""

    def __init__(self):
        self.beats = {}
        self.order = []

    def add(self, name, start, end):
        self.beats[name] = (float(start), float(end))
        self.order.append(name)
        return self

    def then(self, name, duration, gap=0.0):
        """Append a beat right after the previous one."""
        start = self.beats[self.order[-1]][1] + gap if self.order else 0.0
        return self.add(name, start, start + duration)

    @property
    def duration(self):
        return max(e for _, e in self.beats.values()) if self.beats else 0.0

    def span(self, name):
        return self.beats[name]

    def start(self, name):
        return self.beats[name][0]

    def end(self, name):
        return self.beats[name][1]

    def local(self, name, t):
        """Seconds since the beat started (negative before it)."""
        return t - self.beats[name][0]

    def active(self, name, t, pad=0.0):
        s, e = self.beats[name]
        return s - pad <= t <= e + pad

    def p(self, name, t, ease=smoother):
        """Eased 0..1 progress through the beat."""
        s, e = self.beats[name]
        return ease(remap(t, s, e))

    def fade(self, name, t, fade_in=0.5, fade_out=0.5, delay=0.0, early_out=0.0):
        """0 -> 1 -> 0 envelope over the beat (smoothstepped edges)."""
        s, e = self.beats[name]
        s += delay
        e -= early_out
        if t < s or t > e:
            return 0.0
        a = smooth((t - s) / fade_in) if fade_in > 0 else 1.0
        b = smooth((e - t) / fade_out) if fade_out > 0 else 1.0
        return min(a, b)

    def which(self, t):
        for n in self.order:
            s, e = self.beats[n]
            if s <= t < e:
                return n
        return self.order[-1] if self.order else None


def envelope(t, t_in, t_out, fade_in=0.5, fade_out=0.5):
    """Standalone fade envelope: rises at t_in, falls to 0 at t_out."""
    if t < t_in or t > t_out:
        return 0.0
    a = smooth((t - t_in) / fade_in) if fade_in > 0 else 1.0
    b = smooth((t_out - t) / fade_out) if fade_out > 0 else 1.0
    return min(a, b)


class Keys:
    """Keyframed vector value with smoother-step interpolation between keys.

    keys = [(time, value), ...]. Values are anything numpy can add.
    """

    def __init__(self, keys, ease=smoother):
        self.keys = sorted(((float(t), np.asarray(v, np.float64)) for t, v in keys), key=lambda k: k[0])
        self.ease = ease

    def __call__(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1].copy()
        for (t0, v0), (t1, v1) in zip(ks, ks[1:]):
            if t <= t1:
                u = self.ease((t - t0) / max(t1 - t0, 1e-9))
                return v0 * (1 - u) + v1 * u
        return ks[-1][1].copy()


def fade_in_out(t, duration, fade_in=0.8, fade_out=0.9):
    """Veil for a film that rises from black and returns to it (1 = black), for Hud.fade."""
    return 1.0 - min(smooth(remap(t, 0.0, fade_in)), 1.0 - smooth(remap(t, duration - fade_out, duration)))


class OrbitCamera:
    """A keyframed orbit around a look point.

    keys = [(time, azimuth deg, elevation deg, distance m, look point, fov deg), ...].
    By default the look points are in a Z-up (robot) frame and azimuth is measured in
    its XY plane from +x. With `frame="yup"` they are in the stage's Y-up world and
    azimuth is measured in the XZ plane from +x towards -z (the same direction once
    Z-up is turned into Y-up). `follow(t)`, if given, returns a point (same frame) that
    the keyed look point is added to, and `heading(t)` an angle (rad) added to the
    azimuth: together a chase camera. `drift` is ((amplitude deg, rad/s) for azimuth,
    (amplitude deg, rad/s) for elevation): slow sinusoids so held shots never freeze.
    Called with t, returns (eye, look, fov) in the Y-up world, ready for `Stage.look`.
    """

    def __init__(self, keys, drift=((1.2, 0.21), (0.6, 0.17)), frame="zup", follow=None, heading=None):
        self.orbit = Keys([(k[0], [k[1], k[2], k[3], k[5]]) for k in keys])
        self.target = Keys([(k[0], k[4]) for k in keys])
        self.drift = drift
        self.frame = frame
        self.follow = follow
        self.heading = heading

    def __call__(self, t):
        az, el, dist, fov = (float(v) for v in self.orbit(t))
        look = self.target(t)
        if self.follow is not None:
            look = look + np.asarray(self.follow(t), float)
        (az_amp, az_rate), (el_amp, el_rate) = self.drift
        az += az_amp * math.sin(az_rate * t)
        el += el_amp * math.sin(el_rate * t + 1.0)
        a, e = math.radians(az), math.radians(el)
        if self.heading is not None:
            a += float(self.heading(t))
        if self.frame == "yup":
            eye = look + dist * np.array([math.cos(e) * math.cos(a), math.sin(e), -math.cos(e) * math.sin(a)])
            return eye, look, fov
        eye = look + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        zw = lambda p: np.array([p[0], p[2], -p[1]])  # noqa: E731  Z-up -> Y-up
        return zw(eye), zw(look), fov


# ── the stage ─────────────────────────────────────────────────────────────────
def data_roots():
    """Where threepp_data may be, in search order: THREEPP_DATA_DIR, a checkout next
    to the repo, then the copies CMake fetches into the build directories."""
    repo = os.path.dirname(_PY)
    roots = [os.environ.get("THREEPP_DATA_DIR", "")]
    roots += [os.path.join(os.path.dirname(repo), n) for n in ("threepp-data", "threepp_data")]
    roots += sorted(glob.glob(os.path.join(repo, "cmake-build-*", "_deps", "threepp_data-src")))
    return [r for r in roots if r and os.path.isdir(r)]


def data_file(*parts):
    """Path of a threepp_data file in the first root that has it. Roots are checked
    per file, so an older checkout that lacks it does not hide a newer copy."""
    rel = os.path.join(*parts)
    roots = data_roots()
    for r in roots:
        p = os.path.join(r, rel)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"threepp_data file '{rel}' not found in {roots or 'any data directory'}; "
                            f"set THREEPP_DATA_DIR to a threepp_data checkout that has it")


def standard(color, roughness=0.5, metalness=0.0, emissive=None, emissive_intensity=1.0):
    m = tp.MeshStandardMaterial()
    m.color = to_hex(color)
    m.roughness = roughness
    m.metalness = metalness
    if emissive is not None:
        m.emissive = to_hex(emissive)
        m.emissive_intensity = emissive_intensity
    return m


class Stage:
    """A headless canvas, a renderer, and a dark studio set.

    World is Y-up (three.js convention). Robots from URDF are Z-up; parent them
    under `stage.zup`, a group that turns Z-up into Y-up, and annotate in the
    robot's own frame.
    """

    BG = 0x0b0f17

    def __init__(self, width=1920, height=1080, renderer="gl", headless=True, msaa=8,
                 floor=True, env="empty_warehouse_01_2k.hdr", env_intensity=0.35, design=(1920, 1080),
                 fog=None, shadow_extent=1.6, far=60.0):
        """`fog` = (near, far) in metres (default 3.5..9 with the studio floor, none without).
        `shadow_extent` is the half-size of the key light's shadow area around its target;
        `follow()` moves that area with a subject. `far` is the camera's far plane."""
        self.W, self.H = int(width), int(height)
        self.design = design     # project() answers in these units, the Hud's
        self.kind = renderer
        if renderer == "vulkan":
            self.canvas = tp.Canvas("lesson", width=self.W, height=self.H, vsync=False, headless=headless)
            self.r = tp.VulkanRenderer(self.canvas)
            self.r.sun_angular_radius = 2.0
            self.r.gbuffer_msaa = 2
        else:
            self.canvas = tp.Canvas("lesson", width=self.W, height=self.H, antialiasing=msaa, headless=headless)
            self.r = tp.GLRenderer(self.canvas)
            self.r.shadow_map_enabled = True
        self.r.tone_mapping = tp.ToneMapping.ACESFilmic
        self.r.tone_mapping_exposure = 0.95
        self.frame_index = 0

        self.scene = tp.Scene()
        self.scene.background = tp.Background(self.BG)
        if renderer != "vulkan" and env:
            try:
                path = data_file("textures", "env", env)
            except FileNotFoundError as e:
                print(f"[stage] rendering without image-based lighting: {e}", file=sys.stderr)
            else:
                self.scene.environment = tp.RGBELoader().load(path)
                self.scene.environment_intensity = env_intensity

        self.hemi = tp.HemisphereLight(0x9fb6d8, 0x0c0e12, 0.35)
        self.scene.add(self.hemi)
        self.key = tp.DirectionalLight(0xfff1e0, 2.7)
        self.key.position.set(2.6, 6.0, 3.6)
        self.key.cast_shadow = True
        self.key.shadow.map_size = tp.Vector2(4096, 4096)
        self.key.shadow.radius = 5
        self.key.shadow.bias = -0.0003
        e = shadow_extent
        self.key.set_shadow_frustum(-e, e, e, -e)
        self.scene.add(self.key)
        self._key_offset = np.array([2.6, 6.0, 3.6])
        self._key_target = None
        self.rim = tp.DirectionalLight(0x86a8ff, 1.4)
        self.rim.position.set(-4.0, 3.0, -3.5)
        self.scene.add(self.rim)

        if floor:
            self.floor_mat = standard(0x0f1319, roughness=0.7)
            self.floor_mat.env_map_intensity = 0.06
            self.floor = tp.Mesh(tp.CircleGeometry(12.0, 128), self.floor_mat)
            self.floor.rotate_x(-math.pi / 2)
            self.floor.receive_shadow = True
            self.scene.add(self.floor)
            if fog is None:
                fog = (3.5, 9.0)     # the floor dissolves into the backdrop instead of ending at a rim
        if fog is not None:
            self.scene.set_fog(tp.Color(self.BG), float(fog[0]), float(fog[1]))

        self.zup = tp.Group()
        self.zup.rotate_x(-math.pi / 2)
        self.scene.add(self.zup)

        self.camera = tp.PerspectiveCamera(32, self.W / self.H, 0.03, far)
        self._look = np.zeros(3)

    def follow(self, p, height=None):
        """Centre the key light and its shadow area on world point p (Y-up). The light
        keeps its direction; `height` scales its offset (the shadow camera's depth)."""
        p = np.asarray(p, float)
        if self._key_target is None:
            self._key_target = tp.Object3D()
            self.scene.add(self._key_target)
            self.key.set_target(self._key_target)
        off = self._key_offset if height is None else self._key_offset * (height / self._key_offset[1])
        self._key_target.position.set(*p)
        self.key.position.set(*(p + off))

    # camera ----------------------------------------------------------------
    def look(self, eye, target, fov=None, roll=0.0):
        eye = np.asarray(eye, float)
        target = np.asarray(target, float)
        if fov is not None:
            self.camera.fov = float(fov)
            self.camera.update_projection_matrix()
        self.camera.position.set(*eye)
        if roll:
            fwd = target - eye
            fwd /= np.linalg.norm(fwd)
            right = np.cross(fwd, [0, 1, 0])
            right /= np.linalg.norm(right) + 1e-12
            up = np.cross(right, fwd)
            u = math.cos(roll) * up + math.sin(roll) * right
            self.camera.up.set(*u)
        else:
            self.camera.up.set(0, 1, 0)
        self.camera.look_at(*target)
        self._look = target
        self.camera.update_matrix_world()

    # frame -----------------------------------------------------------------
    def frame(self, t=None, hud=None):
        """Render the scene (and the Hud over it) and return (H, W, 3) uint8."""
        if self.kind == "vulkan" and t is not None:
            self.r.sim_time = float(t)
        self.r.render(self.scene, self.camera)
        if hud is not None:
            self.r.auto_clear = False
            self.r.render(hud.scene, hud.camera)
            self.r.auto_clear = True
        self.frame_index += 1
        return self.r.read_pixels()

    # projection --------------------------------------------------------------
    def _vp(self):
        self.camera.update_matrix_world()
        view = np.linalg.inv(self.camera.matrix_world.to_numpy())
        f = math.radians(self.camera.fov)
        a = self.W / self.H
        n, fa = self.camera.near, self.camera.far
        P = np.zeros((4, 4))
        P[0, 0] = 1.0 / (a * math.tan(f / 2))
        P[1, 1] = 1.0 / math.tan(f / 2)
        P[2, 2] = -(fa + n) / (fa - n)
        P[2, 3] = -2 * fa * n / (fa - n)
        P[3, 2] = -1.0
        return P @ view

    def project(self, p_world):
        """World point(s) -> pixel (x, y) and view depth. Accepts (3,) or (N, 3)."""
        p = np.atleast_2d(np.asarray(p_world, float))
        h = np.c_[p, np.ones(len(p))] @ self._vp().T
        w = h[:, 3:4]
        ndc = h[:, :3] / w
        dw, dh = self.design
        xy = np.c_[(ndc[:, 0] * 0.5 + 0.5) * dw, (0.5 - ndc[:, 1] * 0.5) * dh]
        out = np.c_[xy, w[:, 0]]
        return out[0] if np.asarray(p_world).ndim == 1 else out

    def zup_to_world(self, p):
        """Robot (Z-up) coordinates -> world (Y-up): (x, y, z) -> (x, z, -y)."""
        p = np.asarray(p, float)
        return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1)

    def project_zup(self, p):
        return self.project(self.zup_to_world(p))


# ── 3D annotation kit ─────────────────────────────────────────────────────────
def mat4(M):
    """A 4x4 numpy matrix as a tp.Matrix4."""
    m = tp.Matrix4()
    m.set(*[float(v) for v in np.asarray(M, float).reshape(-1)])
    return m


def set_pose(obj, M):
    """Place `obj` at a 4x4 numpy pose in its parent's frame (position and rotation)."""
    M = np.asarray(M, float)
    obj.position.set(*M[:3, 3])
    R = M[:3, :3]
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-4:
        x = (R[2, 1] - R[1, 2]) / (4 * w)
        y = (R[0, 2] - R[2, 0]) / (4 * w)
        z = (R[1, 0] - R[0, 1]) / (4 * w)
    else:  # 180 degree turn: pick the largest diagonal
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            x = math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
            y, z, w = (R[0, 1] + R[1, 0]) / (4 * x), (R[0, 2] + R[2, 0]) / (4 * x), (R[2, 1] - R[1, 2]) / (4 * x)
        elif i == 1:
            y = math.sqrt(max(0.0, 1 + R[1, 1] - R[0, 0] - R[2, 2])) / 2
            x, z, w = (R[0, 1] + R[1, 0]) / (4 * y), (R[1, 2] + R[2, 1]) / (4 * y), (R[0, 2] - R[2, 0]) / (4 * y)
        else:
            z = math.sqrt(max(0.0, 1 + R[2, 2] - R[0, 0] - R[1, 1])) / 2
            x, y, w = (R[0, 2] + R[2, 0]) / (4 * z), (R[1, 2] + R[2, 1]) / (4 * z), (R[1, 0] - R[0, 1]) / (4 * z)
    obj.quaternion.set(x, y, z, w)


def quat_y_to(d):
    """Quaternion (x, y, z, w) rotating +Y onto unit vector d."""
    y = np.array([0.0, 1.0, 0.0])
    d = np.asarray(d, float)
    c = float(np.dot(y, d))
    if c < -0.999999:
        return (1.0, 0.0, 0.0, 0.0)
    axis = np.cross(y, d)
    s = math.sqrt((1.0 + c) * 2.0)
    return (axis[0] / s, axis[1] / s, axis[2] / s, s / 2.0)


def _fade_material(m, opacity, xray=False):
    opacity = clamp01(opacity)
    m.transparent = opacity < 0.999 or xray
    m.opacity = opacity
    m.depth_write = opacity >= 0.999 and not xray


def xray(obj, order=10):
    """Draw `obj` (and children) on top of everything: no depth test, late render order."""
    def f(o):
        o.render_order = order
        mat = getattr(o, "material", None)
        if mat is not None:
            mat.depth_test = False
            mat.depth_write = False
            mat.transparent = True
    obj.traverse(f)


def ghost(obj, color, opacity=0.3, emissive=0.35):
    """A translucent copy of `obj` (recursive clone, every mesh on one shared material):
    a second pose of the same thing on screen. Returns (copy, material); fade it with
    `material.opacity` and hide it with `copy.visible`."""
    g = obj.clone(True)
    mat = standard(color, roughness=0.4, emissive=color, emissive_intensity=emissive)
    mat.transparent = True
    mat.depth_write = False
    mat.opacity = opacity

    def f(o):
        if type(o).__name__ == "Mesh":
            o.set_material(mat)
            o.cast_shadow = False
    g.traverse(f)
    return g, mat


class Arrow3D:
    """A thick arrow from `a` to `b` (in its parent's frame)."""

    def __init__(self, color, radius=0.008, head_radius=None, head_length=None, emissive=0.55,
                 parent=None):
        self.radius = radius
        self.head_r = head_radius or radius * 2.6
        self.head_l = head_length or radius * 6.5
        self.mat = standard(color, roughness=0.35, emissive=color, emissive_intensity=emissive)
        self.group = tp.Group()
        self.shaft = tp.Mesh(tp.CylinderGeometry(radius, radius, 1.0, 20), self.mat)
        self.head = tp.Mesh(tp.ConeGeometry(self.head_r, self.head_l, 28), self.mat)
        self.shaft.cast_shadow = True
        self.head.cast_shadow = True
        self.group.add(self.shaft)
        self.group.add(self.head)
        if parent is not None:
            parent.add(self.group)
        self.opacity = 1.0

    def set(self, a, b, opacity=None, scale=1.0):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        d = b - a
        L = float(np.linalg.norm(d))
        if opacity is not None:
            self.set_opacity(opacity)
        if L < 1e-6 or scale <= 1e-4:
            self.group.visible = False
            return
        self.group.visible = self.opacity > 0.003
        u = d / L
        hl = min(self.head_l * scale, 0.45 * L) if L < 2.2 * self.head_l * scale else self.head_l * scale
        sl = max(L - hl, 1e-4)
        self.group.position.set(*a)
        self.group.quaternion.set(*quat_y_to(u))
        self.shaft.scale.set(scale, sl, scale)
        self.shaft.position.set(0, sl / 2, 0)
        self.head.scale.set(scale, hl / self.head_l if self.head_l > 0 else 1.0, scale)
        self.head.position.set(0, sl + hl / 2, 0)

    def set_opacity(self, o):
        self.opacity = clamp01(o)
        _fade_material(self.mat, self.opacity)
        self.group.visible = self.opacity > 0.003

    def set_color(self, color, emissive=None):
        self.mat.color = to_hex(color)
        self.mat.emissive = to_hex(color)
        if emissive is not None:
            self.mat.emissive_intensity = emissive


class Ring3D:
    """A glowing torus in its parent's XY plane (i.e. around the local Z axis)."""

    def __init__(self, color, radius=0.08, tube=0.006, parent=None, emissive=0.9, arc=2 * math.pi):
        self.mat = standard(color, roughness=0.3, emissive=color, emissive_intensity=emissive)
        self.mesh = tp.Mesh(tp.TorusGeometry(radius, tube, 16, 96, arc), self.mat)
        if parent is not None:
            parent.add(self.mesh)
        self.set_opacity(0.0)

    def set_opacity(self, o):
        _fade_material(self.mat, o)
        self.mesh.visible = o > 0.003


class Marker3D:
    """The target: a glowing core, a translucent shell, and an optional axis triad."""

    def __init__(self, color, radius=0.022, parent=None, triad=True, triad_len=0.09):
        self.group = tp.Group()
        self.core_mat = standard(color, roughness=0.2, emissive=color, emissive_intensity=2.2)
        self.core = tp.Mesh(tp.SphereGeometry(radius, 40, 24), self.core_mat)
        self.shell_mat = tp.MeshBasicMaterial()
        self.shell_mat.color = to_hex(color)
        self.shell_mat.blending = tp.Blending.Additive
        self.shell_mat.transparent = True
        self.shell_mat.depth_write = False
        self.shell = tp.Mesh(tp.SphereGeometry(radius * 1.9, 40, 24), self.shell_mat)
        self.group.add(self.core)
        self.group.add(self.shell)
        self.triad = []
        if triad:
            for c, d in ((0xff5a5f, (1, 0, 0)), (0x5ee27a, (0, 1, 0)), (0x4f8dff, (0, 0, 1))):
                a = Arrow3D(c, radius=0.0045, parent=self.group, emissive=0.8)
                a.set((0, 0, 0), np.array(d, float) * triad_len)
                self.triad.append(a)
        if parent is not None:
            parent.add(self.group)
        self.set_opacity(0.0)

    def place(self, M):
        """Pose from a 4x4 numpy matrix (parent frame)."""
        set_pose(self.group, M)

    def set_opacity(self, o, pulse=0.0):
        o = clamp01(o)
        _fade_material(self.core_mat, o)
        self.shell_mat.opacity = 0.16 * o * (1.0 + 0.5 * pulse)
        self.shell.visible = o > 0.003
        for a in self.triad:
            a.set_opacity(o)
        self.group.visible = o > 0.003


def tube_arrays(points, radius, sides=12):
    """Positions, normals and indices for a tube through `points` (N, 3)."""
    P = np.asarray(points, np.float64)
    if len(P) >= 2:   # drop repeated points: they have no tangent
        keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]
        P = P[keep]
    n = len(P)
    if n < 2:
        return None
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True) + 1e-12
    # parallel-transport frames: no twisting
    ref = np.array([0.0, 0.0, 1.0]) if abs(T[0, 2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    N = np.zeros_like(P)
    N[0] = np.cross(T[0], ref)
    N[0] /= np.linalg.norm(N[0])
    for i in range(1, n):
        v = N[i - 1] - np.dot(N[i - 1], T[i]) * T[i]
        L = np.linalg.norm(v)
        N[i] = v / L if L > 1e-9 else N[i - 1]
    B = np.cross(T, N)
    ang = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    r = np.broadcast_to(np.asarray(radius, float), (n,))[:, None, None]
    nrm = ca[None, :, None] * N[:, None, :] + sa[None, :, None] * B[:, None, :]
    pos = P[:, None, :] + r * nrm
    i = np.arange(n - 1)[:, None]
    j = np.arange(sides)[None, :]
    a = i * sides + j
    b = i * sides + (j + 1) % sides
    c = (i + 1) * sides + j
    d = (i + 1) * sides + (j + 1) % sides
    idx = np.stack([a, c, b, b, c, d], axis=-1).reshape(-1)
    return (pos.reshape(-1, 3).astype(np.float32), nrm.reshape(-1, 3).astype(np.float32),
            idx.astype(np.uint32))


class Tube3D:
    """A tube through a list of points, rebuilt whenever `set_points` is called."""

    def __init__(self, color, radius=0.004, sides=12, parent=None, emissive=0.9, xray=False):
        self.xray = xray
        self.radius = radius
        self.sides = sides
        self.mat = standard(color, roughness=0.35, emissive=color, emissive_intensity=emissive)
        self.geom = tp.BufferGeometry()
        self.mesh = tp.Mesh(self.geom, self.mat)
        self.mesh.frustum_culled = False
        if parent is not None:
            parent.add(self.mesh)
        self._n = 0
        self.mesh.visible = False

    def set_points(self, pts, radius=None, opacity=1.0):
        pts = np.asarray(pts, float)
        if len(pts) < 2 or opacity <= 0.003:
            self.mesh.visible = False
            return
        arr = tube_arrays(pts, self.radius if radius is None else radius, self.sides)
        if arr is None:
            self.mesh.visible = False
            return
        pos, nrm, idx = arr
        # A fresh geometry per change: attribute sizes vary with the point count.
        g = tp.BufferGeometry()
        g.set_attribute("position", pos)
        g.set_attribute("normal", nrm)
        g.set_index(idx)
        self.mesh.set_geometry(g)
        self.geom = g
        _fade_material(self.mat, opacity, self.xray)
        if self.xray:
            self.mat.depth_test = False
            self.mesh.render_order = 10
        self.mesh.visible = True


def turbo(t):
    """Google's Turbo colormap (polynomial fit): t in [0, 1] -> (N, 3) float RGB in [0, 1]."""
    t = np.clip(np.asarray(t, np.float64), 0.0, 1.0)
    r = np.array([0.13572138, 4.61539260, -42.66032258, 132.13108234, -152.94239396, 59.28637943])
    g = np.array([0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604])
    b = np.array([0.10667330, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973])
    P = np.stack([t ** k for k in range(6)], axis=-1)
    return np.clip(np.stack([P @ r, P @ g, P @ b], axis=-1), 0.0, 1.0)


def turbo_hex(t):
    r, g, b = (turbo(np.array([t]))[0] * 255).round().astype(int)
    return (int(r) << 16) | (int(g) << 8) | int(b)


def disc_texture(size=64, soft=0.18):
    """A round sprite (white disc, soft edge) for Points."""
    y, x = np.mgrid[0:size, 0:size]
    r = np.hypot(x - (size - 1) / 2, y - (size - 1) / 2) / (size / 2)
    a = np.clip((1.0 - r) / soft, 0.0, 1.0)
    img = np.zeros((size, size, 4), np.uint8)
    img[..., :3] = 255
    img[..., 3] = (a * 255).astype(np.uint8)
    tex = tp.data_texture(img, True)
    return tex


class Cloud:
    """A point cloud with a fixed capacity, updated in place (positions, colours, count)."""

    _sprite = None

    def __init__(self, capacity, size=0.004, parent=None, sprite=True):
        self.capacity = int(capacity)
        self.geom = tp.BufferGeometry()
        self.geom.set_attribute("position", np.zeros((self.capacity, 3), np.float32))
        self.geom.set_attribute("color", np.zeros((self.capacity, 3), np.float32))
        self.geom.set_draw_range(0, 0)
        self.mat = tp.PointsMaterial()
        self.mat.size = size
        self.mat.size_attenuation = True
        self.mat.vertex_colors = True
        if sprite:
            if Cloud._sprite is None:
                Cloud._sprite = disc_texture()
            self.mat.map = Cloud._sprite
            self.mat.alpha_test = 0.5
        self.points = tp.Points(self.geom, self.mat)
        self.points.frustum_culled = False
        if parent is not None:
            parent.add(self.points)
        self._pos = np.zeros((self.capacity, 3), np.float32)
        self._col = np.zeros((self.capacity, 3), np.float32)

    def set(self, pos, col, opacity=1.0):
        n = min(len(pos), self.capacity)
        if n == 0 or opacity <= 0.003:
            self.geom.set_draw_range(0, 0)
            self.points.visible = False
            return
        self._pos[:n] = pos[:n]
        self._col[:n] = col[:n]
        self.geom.update_attribute("position", self._pos)
        self.geom.update_attribute("color", self._col)
        self.geom.set_draw_range(0, n)
        self.mat.transparent = opacity < 0.999
        self.mat.opacity = opacity
        self.points.visible = True


class Segments:
    """Many line segments (1 px), updated in place; additive by default so dense fans glow."""

    def __init__(self, capacity, parent=None, additive=True):
        self.capacity = int(capacity)
        self.geom = tp.BufferGeometry()
        self.geom.set_attribute("position", np.zeros((2 * self.capacity, 3), np.float32))
        self.geom.set_attribute("color", np.zeros((2 * self.capacity, 3), np.float32))
        self.geom.set_draw_range(0, 0)
        self.mat = tp.LineBasicMaterial()
        self.mat.vertex_colors = True
        self.mat.transparent = True
        self.mat.depth_write = False
        if additive:
            self.mat.blending = tp.Blending.Additive
        self.lines = tp.LineSegments(self.geom, self.mat)
        self.lines.frustum_culled = False
        if parent is not None:
            parent.add(self.lines)
        self._pos = np.zeros((2 * self.capacity, 3), np.float32)
        self._col = np.zeros((2 * self.capacity, 3), np.float32)

    def set(self, a, b, col_a, col_b=None, opacity=1.0):
        n = min(len(a), self.capacity)
        if n == 0 or opacity <= 0.003:
            self.lines.visible = False
            return
        self._pos[0:2 * n:2] = a[:n]
        self._pos[1:2 * n:2] = b[:n]
        self._col[0:2 * n:2] = col_a[:n]
        self._col[1:2 * n:2] = (col_a if col_b is None else col_b)[:n]
        self.geom.update_attribute("position", self._pos)
        self.geom.update_attribute("color", self._col)
        self.geom.set_draw_range(0, 2 * n)
        self.mat.opacity = opacity
        self.lines.visible = True


# ── 2D overlay ────────────────────────────────────────────────────────────────
TEXT = 0xf2f5fa     # body text and maths
DIM = 0x9fb2cc      # labels, notes, secondary text

FONT_DIRS = [os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts"),
             "/usr/share/fonts/truetype/dejavu", "/usr/share/fonts"]
FONT_FILES = {
    "regular": ["segoeui.ttf", "DejaVuSans.ttf"],
    "light": ["segoeuil.ttf", "segoeuisl.ttf", "DejaVuSans.ttf"],
    "semibold": ["seguisb.ttf", "segoeuib.ttf", "DejaVuSans-Bold.ttf"],
    "bold": ["segoeuib.ttf", "DejaVuSans-Bold.ttf"],
    "italic": ["segoeuii.ttf", "DejaVuSans-Oblique.ttf"],
    "numeric": ["bahnschrift.ttf", "DejaVuSansMono.ttf"],
    "mono": ["consola.ttf", "DejaVuSansMono.ttf"],
}


def _find_font(kind):
    for name in FONT_FILES[kind]:
        for d in FONT_DIRS:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return None


# ── code on screen ────────────────────────────────────────────────────────────
# Keywords per language (read-only: a lesson that wants more passes its own set, e.g.
# KEYWORDS["cpp"] | {"for"}), and the colours of each kind of token.
KEYWORDS = MappingProxyType({"py": frozenset({"import", "as", "def", "return", "for", "in", "if", "else", "True", "False", "None",
                             "from"}),
            "cpp": frozenset({"using", "namespace", "int", "auto", "return", "const", "new", "float"}),
            "js": frozenset({"const", "new", "let", "function"}), "cmake": frozenset(),
            "sh": frozenset({"pip", "python"})})
CODE_COLOURS = {"kw": 0xc792ea, "str": 0xc3e88d, "num": 0xf78c6c, "fn": 0x82aaff, "type": 0xffcb6b, "mod": 0x89ddff,
                "com": 0x6b7a90, "punc": 0x8fa3bf, "id": 0xe6ecf5, "pre": 0xc792ea}
CODE_ACCENT = 0x4cc9f0       # the bar beside a highlighted line
_TOK = re.compile(r'(//.*|#.*)|("[^"]*")|(\b0x[0-9a-fA-F]+\b|\b\d+\.?\d*f?\b)|([A-Za-z_][A-Za-z_0-9]*)|(\s+)|(.)')


def tokenize(line, lang, keywords=None, palette=None):
    """(column, text, colour) per token of one source line; whitespace is skipped. `keywords`
    replaces KEYWORDS[lang], `palette` replaces CODE_COLOURS."""
    kw = KEYWORDS.get(lang, ()) if keywords is None else keywords
    col = CODE_COLOURS if palette is None else palette
    out = []
    ms = list(_TOK.finditer(line))
    for k, m in enumerate(ms):
        s = m.group(0)
        if m.group(5):
            continue
        if m.group(1):
            if lang == "cpp" and s.startswith("#"):          # #include "..."
                word = s.split()[0]
                out.append((m.start(), word, col["pre"]))
                rest = s[len(word):]
                if rest.strip():
                    out.append((m.start() + len(word) + (len(rest) - len(rest.lstrip())), rest.strip(), col["str"]))
                continue
            if (lang == "cpp") != s.startswith("//"):         # a '#' in C++ or '//' elsewhere is not a comment
                out.append((m.start(), s, col["punc"]))
                continue
            out.append((m.start(), s, col["com"]))
        elif m.group(2):
            out.append((m.start(), s, col["str"]))
        elif m.group(3):
            out.append((m.start(), s, col["num"]))
        elif m.group(4):
            nxt = line[m.end():].lstrip()[:1]
            if s in kw:
                c = col["kw"]
            elif s in ("tp", "THREE", "threepp"):
                c = col["mod"]
            elif s[0].isupper():
                c = col["type"]
            elif nxt == "(":
                c = col["fn"]
            else:
                c = col["id"]
            out.append((m.start(), s, c))
        else:
            out.append((m.start(), s, col["punc"]))
    return out


class Hud:
    """The 2D layer, drawn by threepp: an orthographic scene rendered over the 3D one.

    Immediate-mode API: call `begin()`, then draw calls (`panel`, `text`, `math`,
    `line`, ...), then `end()`. Behind it every call reuses a pooled threepp mesh:
    glyphs are `Text2D` meshes built from the TTF, panels are `ShapeGeometry`
    rounded rectangles, lines are triangle strips, and equations are matplotlib's
    glyph outlines turned into an SVG path and loaded with `SVGLoader`, so they are
    real geometry too. Objects a frame does not touch are hidden, and each call is
    drawn on top of the previous one (render order = call order, no depth test).

    Coordinates are pixels in a fixed 1920x1080 design space, origin top-left,
    whatever the actual render size. Text is measured with the same threepp Font
    that draws it (`Font.advance`, `ascender`, `descender`), so layout and glyphs agree.
    """

    def __init__(self, width=1920, height=1080, math_cache=None):
        """`math_cache`: a JSON file of typeset equations. With it, a lesson whose
        equations are all cached renders without matplotlib; a new or edited
        equation is typeset once and appended."""
        self.W, self.H = int(width), int(height)
        self._math_file = math_cache
        self._math_disk = {}
        if math_cache and os.path.isfile(math_cache):
            import json
            with open(math_cache, encoding="utf-8") as f:
                self._math_disk = json.load(f)
        self.scene = tp.Scene()
        self.camera = tp.OrthographicCamera(0, self.W, self.H, 0, -10, 10)
        self._fonts_tp = {}
        self._pool = {}
        self._used = {}
        self._order = 0
        self._math_cache = {}

    # fonts -----------------------------------------------------------------
    def _tp_font(self, kind):
        f = self._fonts_tp.get(kind)
        if f is None:
            path = _find_font(kind)
            f = tp.FontLoader().load(path) if path else None
            if f is None:
                f = tp.FontLoader().default_font()
            self._fonts_tp[kind] = f
        return f

    def text_width(self, s, size=28, kind="regular"):
        return self._tp_font(kind).advance(s, size)

    # pooling ---------------------------------------------------------------
    def begin(self):
        self._used = {k: 0 for k in self._pool}
        self._order = 0
        return self

    def end(self):
        for k, objs in self._pool.items():
            n = self._used.get(k, 0)
            for o in objs[n:]:
                o.visible = False

    def _acquire(self, key, factory):
        objs = self._pool.setdefault(key, [])
        n = self._used.get(key, 0)
        if n < len(objs):
            o = objs[n]
        else:
            o = factory()
            self.scene.add(o)
            objs.append(o)
        self._used[key] = n + 1
        o.visible = True
        self._order += 1
        o.render_order = self._order
        return o

    @staticmethod
    def _material(color=0xffffff):
        m = tp.MeshBasicMaterial()
        m.color = to_hex(color)
        m.transparent = True
        m.depth_test = False
        m.depth_write = False
        m.tone_mapped = False
        m.side = tp.Side.Double
        return m

    def _paint(self, mesh, color, alpha):
        m = mesh.material
        m.color = to_hex(color)
        m.opacity = clamp01(alpha)

    def _Y(self, y):
        return self.H - y

    # primitives --------------------------------------------------------------
    @staticmethod
    def _rounded_shape(x, y, w, h, r):
        """Rounded rectangle, lower-left (x, y), y up."""
        r = max(0.0, min(r, w / 2, h / 2))
        s = tp.Shape()
        s.move_to(x + r, y)
        s.line_to(x + w - r, y)
        s.absarc(x + w - r, y + r, r, -math.pi / 2, 0, False)
        s.line_to(x + w, y + h - r)
        s.absarc(x + w - r, y + h - r, r, 0, math.pi / 2, False)
        s.line_to(x + r, y + h)
        s.absarc(x + r, y + h - r, r, math.pi / 2, math.pi, False)
        s.line_to(x, y + r)
        s.absarc(x + r, y + r, r, math.pi, 1.5 * math.pi, False)
        return s

    def _rrect(self, x, y, w, h, radius, color, alpha, outline_w=0.0):
        """A filled rounded rect (or just its outline ring) in design pixels."""
        key = ("rrect", round(w, 1), round(h, 1), round(radius, 1), round(outline_w, 2))

        def make():
            if outline_w > 0:
                outer = self._rounded_shape(0, 0, w, h, radius)
                inner = self._rounded_shape(outline_w, outline_w, w - 2 * outline_w, h - 2 * outline_w,
                                            max(radius - outline_w, 0.0))
                hole = tp.Path()
                pts = inner.get_points(12)
                hole.set_from_points(list(reversed(pts)))
                outer.holes = list(outer.holes) + [hole]
                geom = tp.ShapeGeometry([outer], 12)
            else:
                geom = tp.ShapeGeometry([self._rounded_shape(0, 0, w, h, radius)], 12)
            return tp.Mesh(geom, self._material())
        m = self._acquire(key, make)
        m.position.set(x, self._Y(y + h), 0)
        self._paint(m, color, alpha)
        return m

    def panel(self, x, y, w, h, radius=14, fill=0x0d131e, alpha=0.72, outline=None, outline_alpha=0.25,
              width=1.5):
        if alpha <= 0.003:
            return
        self._rrect(x, y, w, h, radius, fill, alpha)
        if outline is not None:
            self.outline(x, y, w, h, outline, outline_alpha * alpha / 0.72, width=width, radius=radius)

    def outline(self, x, y, w, h, color=0xffffff, alpha=1.0, width=1.5, radius=0):
        """A rounded-rectangle outline with no fill."""
        if alpha <= 0.003:
            return
        self._rrect(x, y, w, h, radius, color, alpha, outline_w=width)

    def _glyphs(self, s, size, kind):
        key = ("text", s, round(size, 1), kind)

        # tessellate by size: a 16 px label needs few segments per curve, a 96 px title many
        segs = int(min(32, max(8, size / 3)))

        def make():
            return tp.Text2D(self._tp_font(kind), s, size=size, curve_segments=segs, material=self._material())
        return self._acquire(key, make)

    def text(self, x, y, s, size=28, color=0xffffff, alpha=1.0, kind="regular", anchor="la", tracking=0.0):
        """anchor = horizontal (l/m/r) + vertical (a ascender, m middle, s baseline, d descender)."""
        if alpha <= 0.003 or not s:
            return
        f = self._tp_font(kind)
        asc, desc = f.ascender(size), -f.descender(size)
        v = anchor[1]
        base = y + (asc if v == "a" else (asc - desc) / 2 if v == "m" else -desc if v == "d" else 0.0)
        if tracking:
            widths = [f.advance(ch, size) for ch in s]
            total = sum(widths) + tracking * (len(s) - 1)
        else:
            total = f.advance(s, size)
        h = anchor[0]
        x0 = x - (total if h == "r" else total / 2 if h == "m" else 0.0)
        if tracking:
            cx = x0
            for ch, w in zip(s, widths):
                if ch != " ":
                    g = self._glyphs(ch, size, kind)
                    g.position.set(cx, self._Y(base), 0)
                    self._paint(g, color, alpha)
                cx += w + tracking
            return
        g = self._glyphs(s, size, kind)
        g.position.set(x0, self._Y(base), 0)
        self._paint(g, color, alpha)

    def rich(self, x, y, parts, size=28, alpha=1.0, anchor="ls"):
        widths = [self.text_width(t, size, k) for t, _, k in parts]
        total = sum(widths)
        cx = x - (total if anchor[0] == "r" else total / 2 if anchor[0] == "m" else 0)
        for (t, c, k), w in zip(parts, widths):
            self.text(cx, y, t, size=size, color=c, alpha=alpha, kind=k, anchor="l" + anchor[1])
            cx += w
        return total

    def _strip(self, pts, width, round_ends=True):
        """Triangles for a thick polyline (design pixels, y already flipped)."""
        P = np.asarray(pts, np.float64)
        keep = np.r_[True, np.linalg.norm(np.diff(P, axis=0), axis=1) > 1e-6]
        P = P[keep]
        if len(P) < 2:
            return None, None
        hw = width / 2.0
        pos, idx = [], []
        for a, b in zip(P[:-1], P[1:]):
            d = b - a
            n = np.array([-d[1], d[0]]) / np.linalg.norm(d) * hw
            k = len(pos)
            pos += [a + n, a - n, b + n, b - n]
            idx += [k, k + 1, k + 2, k + 1, k + 3, k + 2]
        # round joins and caps: small fans
        joins = P if round_ends else P[1:-1]
        seg = 10
        for c in joins:
            k = len(pos)
            pos.append(c)
            for s_ in range(seg + 1):
                ang = 2 * math.pi * s_ / seg
                pos.append(c + hw * np.array([math.cos(ang), math.sin(ang)]))
            for s_ in range(seg):
                idx += [k, k + 1 + s_, k + 2 + s_]
        pos = np.c_[np.asarray(pos), np.zeros(len(pos))].astype(np.float32)
        return pos, np.asarray(idx, np.uint32)

    def _dyn_mesh(self, pos, idx, color, alpha):
        m = self._acquire(("dyn",), lambda: tp.Mesh(tp.BufferGeometry(), self._material()))
        g = tp.BufferGeometry()
        g.set_attribute("position", pos)
        g.set_index(idx)
        m.set_geometry(g)
        m.frustum_culled = False
        m.position.set(0, 0, 0)
        self._paint(m, color, alpha)
        return m

    def line(self, pts, color=0xffffff, width=2.0, alpha=1.0):
        if alpha <= 0.003:
            return
        P = [(float(px), self._Y(float(py))) for px, py in pts]
        pos, idx = self._strip(P, width)
        if pos is not None:
            self._dyn_mesh(pos, idx, color, alpha)

    def dashed(self, p0, p1, color=0xffffff, width=2.0, alpha=1.0, dash=10, gap=7, phase=0.0):
        if alpha <= 0.003:
            return
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        L = float(np.linalg.norm(p1 - p0))
        if L < 1e-3:
            return
        u = (p1 - p0) / L
        s = -phase % (dash + gap) - (dash + gap)
        allpos, allidx = [], []
        while s < L:
            a, b = max(s, 0.0), min(s + dash, L)
            if b > a:
                A, B = p0 + u * a, p0 + u * b
                pos, idx = self._strip([(A[0], self._Y(A[1])), (B[0], self._Y(B[1]))], width, round_ends=False)
                if pos is not None:
                    allidx.append(idx + sum(len(q) for q in allpos))
                    allpos.append(pos)
            s += dash + gap
        if allpos:
            self._dyn_mesh(np.concatenate(allpos), np.concatenate(allidx), color, alpha)

    def circle(self, x, y, r, fill=None, outline=None, width=2.0, alpha=1.0):
        if alpha <= 0.003:
            return
        if fill is not None:
            m = self._acquire(("disc",), lambda: tp.Mesh(tp.CircleGeometry(1.0, 48), self._material()))
            m.position.set(x, self._Y(y), 0)
            m.scale.set(r, r, 1)
            self._paint(m, fill, alpha)
        if outline is not None:
            pts = [(x + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 64)]
            self.line(pts, outline, width, alpha)

    def arrow2d(self, p0, p1, color=0xffffff, width=2.5, head=12, alpha=1.0):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        d = p1 - p0
        L = np.linalg.norm(d)
        if L < 1e-3 or alpha <= 0.003:
            return
        u = d / L
        n = np.array([-u[1], u[0]])
        base = p1 - u * head
        self.line([p0, base + u * 1.0], color, width, alpha)
        tri = np.array([p1, base + n * head * 0.5, base - n * head * 0.5])
        pos = np.c_[tri[:, 0], self.H - tri[:, 1], np.zeros(3)].astype(np.float32)
        self._dyn_mesh(pos, np.array([0, 1, 2], np.uint32), color, alpha)

    def bar(self, x, y, w, h, frac, color, alpha=1.0, track=0x2a3342, radius=None):
        radius = h / 2 if radius is None else radius
        self._rrect(x, y, w, h, radius, track, alpha)
        fw = max(h, w * clamp01(frac))
        self._rrect(x, y, fw, h, radius, color, alpha)

    # maths ---------------------------------------------------------------------
    def _math_group(self, tex, size):
        """matplotlib mathtext -> glyph outlines -> one SVG path -> threepp meshes (cached per size)."""
        key = (tex, round(size, 1))
        hit = self._math_cache.get(key)
        if hit is not None:
            return hit
        dkey = f"{round(size, 1)}|{tex}"
        disk = self._math_disk.get(dkey)
        if disk is not None:
            info = (disk["svg"], disk["x0"], disk["y0"], disk["w"], disk["h"])
            self._math_cache[key] = info
            return info
        import matplotlib
        from matplotlib.font_manager import FontProperties
        from matplotlib.path import Path as MPath
        from matplotlib.textpath import TextPath
        with matplotlib.rc_context({"mathtext.fontset": "cm"}):
            tpath = TextPath((0, 0), tex, size=size, prop=FontProperties(size=size))
        V, C = tpath.vertices, tpath.codes
        d = []
        i = 0
        while i < len(V):
            c = C[i]
            x, y = V[i]
            if c == MPath.MOVETO:
                d.append(f"M{x:.3f},{-y:.3f}")
                i += 1
            elif c == MPath.LINETO:
                d.append(f"L{x:.3f},{-y:.3f}")
                i += 1
            elif c == MPath.CURVE3:
                (x1, y1), (x2, y2) = V[i], V[i + 1]
                d.append(f"Q{x1:.3f},{-y1:.3f} {x2:.3f},{-y2:.3f}")
                i += 2
            elif c == MPath.CURVE4:
                (x1, y1), (x2, y2), (x3, y3) = V[i], V[i + 1], V[i + 2]
                d.append(f"C{x1:.3f},{-y1:.3f} {x2:.3f},{-y2:.3f} {x3:.3f},{-y3:.3f}")
                i += 3
            elif c == MPath.CLOSEPOLY:
                d.append("Z")
                i += 1
            else:
                i += 1
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg"><path fill="#ffffff" fill-rule="nonzero" '
               f'd="{" ".join(d)}"/></svg>')
        ext = tpath.get_extents()
        info = (svg, float(ext.x0), float(ext.y0), float(ext.width), float(ext.height))
        self._math_cache[key] = info
        if self._math_file:
            import json
            self._math_disk[dkey] = dict(zip(("svg", "x0", "y0", "w", "h"), info))
            with open(self._math_file, "w", encoding="utf-8") as f:
                json.dump(self._math_disk, f, indent=0, sort_keys=True)
        return info

    def math_size(self, tex, size=34):
        _, _, _, w, h = self._math_group(tex, size)
        return w, h

    def math(self, x, y, tex, size=34, color=0xffffff, alpha=1.0, anchor="lm"):
        """Place maths; (x, y) is the anchor point ('l'/'m'/'r' + 't'/'m'/'b') of its ink box."""
        if alpha <= 0.003:
            return (0, 0)
        svg, x0, y0, w, h = self._math_group(tex, size)

        def make():
            g = tp.SVGLoader().parse(svg)          # y-flipped by the loader: back to y-up
            mat = self._material()
            for ch in g.children:
                ch.set_material(mat)
            return g
        g = self._acquire(("math", tex, round(size, 1)), make)
        for ch in g.children:
            ch.render_order = g.render_order
            self._paint(ch, color, alpha)
        left = x - (w if anchor[0] == "r" else w / 2 if anchor[0] == "m" else 0)
        top = y - (h if anchor[1] == "b" else h / 2 if anchor[1] == "m" else 0)
        # ink box: x0..x0+w, y0..y0+h in y-up TextPath units
        g.position.set(left - x0, self._Y(top) - (y0 + h), 0)
        return (w, h)

    # compound widgets ----------------------------------------------------------
    def caption(self, text, alpha, y=None, size=34, sub=None, width=1640, color=0xf2f5fa):
        """Lower-third caption: one or two balanced lines, centred, on a soft panel."""
        if alpha <= 0.003:
            return
        y = self.H - 118 if y is None else y
        lines = self.wrap_balanced(text, size, width)
        lh = size * 1.28
        widths = [self.text_width(l, size) for l in lines]
        w = max(widths) + 64
        h = lh * len(lines) + 34
        x = (self.W - w) / 2
        top = y - h / 2
        self.panel(x, top, w, h, radius=18, alpha=0.62 * alpha)
        for i, l in enumerate(lines):
            self.text(self.W / 2, top + 17 + i * lh + size * 0.02, l, size=size, color=color, alpha=alpha,
                      anchor="ma")

    def captions(self, t, captions):
        """All captions of a film, [(start, end, text)]: each fades in and out over its span.
        An entry (start, end, text, False) is spoken but not drawn (a line over a title card)."""
        for a0, b0, txt, *shown in captions:
            if shown and not shown[0]:
                continue
            al = envelope(t, a0, b0, 0.45, 0.4)
            if al > 0:
                self.caption(txt, al)

    def title_card(self, t, kicker, title, subtitle, accent, t_in=0.9, t_out=6.4, size=96, sub_dy=118):
        """Opening title, left-aligned: a small tracked kicker, the title rising into place,
        and a subtitle `sub_dy` below it that fades in after it."""
        a = envelope(t, t_in, t_out, 0.9, 0.8)
        if a <= 0:
            return
        y = 360 + 14 * (1 - ease_out(remap(t, t_in, t_in + 1.3)))
        self.text(128, y - 58, kicker, size=22, color=accent, alpha=a, kind="semibold", tracking=4)
        self.text(122, y, title, size=size, color=TEXT, alpha=a, kind="semibold")
        sub = a * smooth(remap(t, t_in + 0.7, t_in + 1.7))
        self.text(128, y + sub_dy, subtitle, size=38, color=DIM, alpha=sub, kind="light")

    def equation_card(self, t, rows, x=70, y=64, size=40, gap=12, min_width=470):
        """A card of equations, rows [(t_in, t_out, tex, note or None)]. Each row fades in and
        out over its span, with its note above it in small capitals, and the card is sized
        to the rows showing at t."""
        rows = [e for e in rows if e[0] - 0.1 <= t <= e[1] + 0.6]
        if not rows:
            return
        sizes = [self.math_size(tex, size) for _, _, tex, _ in rows]
        w = max(max(s[0] for s in sizes) + 60, min_width)
        h = sum(s[1] + (30 if note else gap - 4) for s, (_, _, _, note) in zip(sizes, rows)) + 40
        pa = max(envelope(t, a, b, 0.6, 0.6) for a, b, _, _ in rows)
        self.panel(x, y, w, h, radius=16, alpha=0.66 * pa, outline=0x8aa0c0, outline_alpha=0.16)
        yy = y + 22
        for (a, b, tex, note), (sw, sh) in zip(rows, sizes):
            al = envelope(t, a, b, 0.6, 0.6)
            if note:
                self.text(x + 30, yy, note.upper(), size=16, color=DIM, alpha=al, kind="semibold", tracking=2.2)
                yy += 24
            self.math(x + 30, yy + sh / 2, tex, size=size, color=TEXT, alpha=al, anchor="lm")
            yy += sh + gap

    def summary(self, t, t0, t1, rows, footer, accent, link_color, text_dx=260, link="github.com/markaren/threepp"):
        """Closing card over the darkened frame, from t0 to past t1: "IN SHORT", rows of
        (tex, sentence) that appear one after another, then a footer and a link."""
        a = envelope(t, t0 + 0.2, t1 + 1, 0.8, 0.1)
        if a <= 0:
            return
        self.panel(-20, -20, self.W + 40, self.H + 40, radius=0, fill=0x070a10, alpha=0.8 * a)
        x = 250
        self.text(x, 250, "IN SHORT", size=22, color=accent, alpha=a, kind="semibold", tracking=4)
        for k, (tex, txt) in enumerate(rows):
            al = a * smooth(remap(t, t0 + 0.6 + 0.5 * k, t0 + 1.3 + 0.5 * k))
            yy = 350 + k * 110
            self.math(x, yy, tex, size=46, color=TEXT, alpha=al, anchor="lm")
            self.text(x + text_dx, yy, txt, size=38, color=TEXT, alpha=al, anchor="lm")
        al = a * smooth(remap(t, t0 + 2.4, t0 + 3.2))
        self.text(x, 780, footer, size=28, color=DIM, alpha=al, kind="semibold")
        self.text(x, 824, link, size=26, color=link_color, alpha=al)

    def wrap_balanced(self, text, size, width, kind="regular"):
        """Wrap into the fewest lines, then even out their lengths (no orphans)."""
        n = len(self.wrap(text, size, width, kind))
        if n <= 1:
            return [text]
        lo, hi = 200.0, float(width)
        best = self.wrap(text, size, width, kind)
        for _ in range(18):
            mid = (lo + hi) / 2
            w = self.wrap(text, size, mid, kind)
            if len(w) <= n and all(self.text_width(l, size, kind) <= mid + 1 for l in w):
                best, hi = w, mid
            else:
                lo = mid
        return best

    def wrap(self, text, size, width, kind="regular"):
        words = [w for w in text.split(" ") if w]     # a no-break space (U+00A0) keeps its words together
        lines, cur = [], ""
        for w in words:
            cand = (cur + " " + w).strip()
            if self.text_width(cand, size, kind) <= width or not cur:
                cur = cand
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def callout(self, anchor, label_xy, text, color=0xffffff, alpha=1.0, size=24, kind="semibold",
                dot=5, grow=1.0):
        """Leader line from a projected 3D point to a label."""
        if alpha <= 0.003:
            return
        ax, ay = anchor
        lx, ly = label_xy
        ex = ax + (lx - ax) * grow
        ey = ay + (ly - ay) * grow
        self.circle(ax, ay, dot, fill=color, alpha=alpha)
        self.line([(ax, ay), (ex, ey)], color, 1.6, alpha * 0.9)
        if grow > 0.98:
            left = lx < ax
            self.text(lx + (-10 if left else 10), ly, text, size=size, color=color, alpha=alpha, kind=kind,
                      anchor="rm" if left else "lm")

    def logplot(self, x, y, w, h, values, alpha=1.0, color=0xffb347, ymin=1e-4, ymax=1.0, xmax=None,
                title=None, threshold=None, unit_fmt=None, marker_last=True):
        """Small log-scale plot of a positive series (error vs iteration)."""
        if alpha <= 0.003:
            return
        self.panel(x, y, w, h, alpha=0.70 * alpha, outline=0x8aa0c0, outline_alpha=0.18)
        px, py, pw, ph = x + 70, y + 50, w - 92, h - 80
        if title:
            self.text(x + 22, y + 16, title, size=17, color=0x9fb2cc, alpha=alpha, kind="semibold", tracking=2.0)
        lo, hi = math.log10(ymin), math.log10(ymax)

        def Y(v):
            v = max(float(v), ymin)
            return py + ph * (1 - (math.log10(v) - lo) / (hi - lo))

        # only decades inside [ymin, ymax]: Y() clamps, so one below ymin would sit at ymin's height
        for dec in range(math.ceil(lo - 1e-9), math.floor(hi + 1e-9) + 1):
            yy = Y(10.0 ** dec)
            self.line([(px, yy), (px + pw, yy)], 0x3a4558, 1.0, alpha * 0.8)
            lab = unit_fmt(10.0 ** dec) if unit_fmt else f"1e{dec}"
            self.text(px - 10, yy, lab, size=15, color=0x7f8ea6, alpha=alpha, kind="numeric", anchor="rm")
        if threshold is not None:
            yy = Y(threshold)
            self.dashed((px, yy), (px + pw, yy), 0x5ee27a, 1.4, alpha * 0.85, dash=6, gap=5)
        n = len(values)
        xm = max(xmax or n, 2)

        def X(i):
            return px + pw * i / (xm - 1)

        if n >= 2:
            self.line([(X(i), Y(v)) for i, v in enumerate(values)], color, 2.6, alpha)
        for i, v in enumerate(values):
            self.circle(X(i), Y(v), 4.2 if (i == n - 1 and marker_last) else 3.0, fill=color, alpha=alpha)

    def plot(self, x, y, w, h, series, xlim, ylim, alpha=1.0, title=None, xlog=False, ylog=False,
             xticks=(), yticks=(), xfmt=str, yfmt=str, legend=True, markers=(), width=2.4):
        """Line plot on linear or log axes. series: [(xs, ys, colour, label or None)];
        points outside xlim/ylim are clamped to the frame. xticks/yticks: values that get a
        label (xfmt/yfmt) and, for y, a grid line. markers: [(x, y, colour)] dots."""
        if alpha <= 0.003:
            return
        self.panel(x, y, w, h, alpha=0.70 * alpha, outline=0x8aa0c0, outline_alpha=0.18)
        top = 52 if title else 24
        px, py, pw, ph = x + 70, y + top, w - 94, h - top - 46
        if title:
            self.text(x + 22, y + 16, title, size=17, color=DIM, alpha=alpha, kind="semibold", tracking=2.0)
        fx = math.log10 if xlog else float
        fy = math.log10 if ylog else float
        x0, x1 = fx(xlim[0]), fx(xlim[1])
        y0, y1 = fy(ylim[0]), fy(ylim[1])

        def X(v):
            return px + pw * clamp01((fx(max(v, xlim[0]) if xlog else v) - x0) / (x1 - x0))

        def Y(v):
            return py + ph * (1 - clamp01((fy(max(v, ylim[0]) if ylog else v) - y0) / (y1 - y0)))

        for v in yticks:
            yy = Y(v)
            self.line([(px, yy), (px + pw, yy)], 0x3a4558, 1.0, alpha * 0.8)
            self.text(px - 10, yy, yfmt(v), size=15, color=0x7f8ea6, alpha=alpha, kind="numeric", anchor="rm")
        for v in xticks:
            self.text(X(v), py + ph + 12, xfmt(v), size=15, color=0x7f8ea6, alpha=alpha, kind="numeric", anchor="ma")
        for xs, ys, color, _ in series:
            if len(xs) >= 2:
                self.line([(X(a), Y(b)) for a, b in zip(xs, ys)], color, width, alpha)
        for mx, my, color in markers:
            self.circle(X(mx), Y(my), 5.0, fill=color, alpha=alpha)
        if legend:
            ly = py + 4
            for _, _, color, label in series:
                if label:
                    self.line([(px + pw - 150, ly + 9), (px + pw - 126, ly + 9)], color, 3.0, alpha)
                    self.text(px + pw - 118, ly + 9, label, size=15, color=TEXT, alpha=alpha, anchor="lm")
                    ly += 22

    def readout(self, x, y, w, title, rows, alpha=1.0, row_h=40, value_size=26):
        """A panel of live values: rows [(label, value text, value colour or None)]."""
        if alpha <= 0.003:
            return
        self.panel(x, y, w, 58 + row_h * len(rows), radius=16, alpha=0.66 * alpha, outline=0x8aa0c0,
                   outline_alpha=0.16)
        self.text(x + 24, y + 22, title, size=16, color=DIM, alpha=alpha, kind="semibold", tracking=2.2)
        for j, (label, value, color) in enumerate(rows):
            yy = y + 58 + j * row_h + row_h / 2
            self.text(x + 24, yy, label, size=19, color=DIM, alpha=alpha, anchor="lm")
            self.text(x + w - 24, yy, value, size=value_size, color=TEXT if color is None else color, alpha=alpha,
                      kind="numeric", anchor="rm")

    def fade(self, amount, color=0x000000):
        """Full-frame veil, drawn last: 1 = solid colour."""
        if amount <= 0.003:
            return
        self._rrect(-4, -4, self.W + 8, self.H + 8, 0, color, amount)

    def image(self, x, y, w, h, texture, alpha=1.0):
        """A texture as a w x h rectangle, top-left at (x, y)."""
        if alpha <= 0.003:
            return

        def make():
            mat = self._material()
            mat.map = texture
            return tp.Mesh(tp.PlaneGeometry(1, 1), mat)
        m = self._acquire(("image", id(texture)), make)
        m.position.set(x + w / 2, self._Y(y + h / 2), 0)
        m.scale.set(w, h, 1)
        m.material.opacity = clamp01(alpha)

    def colorbar(self, x, y, w, h, cmap, alpha=1.0, steps=48):
        """Horizontal gradient bar: cmap(t) -> 0xRRGGBB for t in [0, 1]."""
        if alpha <= 0.003:
            return
        sw = w / steps
        for k in range(steps):
            self._rrect(x + k * sw, y, sw + 0.6, h, 0, cmap((k + 0.5) / steps), alpha)

    def code(self, x, y, lines, lang, size=20, lh=27, alpha=1.0, line_alpha=None, hl=None, reveal=None,
             keywords=None, palette=None, accent=CODE_ACCENT):
        """Source lines in a monospace font with syntax colours (see `tokenize`). line_alpha(j),
        hl(j) (a highlight bar, 0..1) and reveal(j) (0..1 of the line's tokens shown) are
        optional per-line functions."""
        if alpha <= 0.003:
            return
        cw = self.text_width("0", size, "mono")
        for j, line in enumerate(lines):
            a = alpha * (line_alpha(j) if line_alpha else 1.0)
            if a <= 0.003 or not line.strip():
                continue
            yy = y + j * lh
            h = hl(j) if hl else 0.0
            if h > 0.003:
                self.panel(x - 16, yy - 2, len(max(lines, key=len)) * cw + 30, lh + 2, radius=5, fill=0x223452,
                           alpha=0.75 * h * alpha)
                self.panel(x - 16, yy - 2, 4, lh + 2, radius=2, fill=accent, alpha=h * alpha)
            toks = tokenize(line, lang, keywords, palette)
            n = len(toks) if reveal is None else int(math.ceil(reveal(j) * len(toks) - 1e-9))
            for col, s, c in toks[:n]:
                self.text(x + col * cw, yy + lh / 2, s, size=size, color=c, alpha=a, kind="mono", anchor="lm")


# ── images without an imaging library ──────────────────────────────────────────
def write_png(path, rgb):
    """Write an (H, W, 3) uint8 array as a PNG with only the standard library."""
    rgb = np.ascontiguousarray(rgb, np.uint8)
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))   # filter type 0 per row

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)))
        f.write(chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(chunk(b"IEND", b""))


def shrink(rgb, factor):
    """Box-filter downsample by an integer factor."""
    h, w = rgb.shape[0] // factor * factor, rgb.shape[1] // factor * factor
    a = rgb[:h, :w].astype(np.float32).reshape(h // factor, factor, w // factor, factor, -1)
    return a.mean(axis=(1, 3)).round().astype(np.uint8)


# ── film writer ───────────────────────────────────────────────────────────────
def ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


class Film:
    """Frames -> H.264 mp4 through an ffmpeg pipe. Written to <path>.part, renamed on close.
    `audio`: a WAV muxed in as AAC, starting `audio_offset` seconds into it."""

    def __init__(self, path, width, height, fps=60, crf=16, preset="slow", threads=0, audio=None, audio_offset=0.0):
        self.path = path
        self.tmp = path + ".part.mp4"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        cmd = [ffmpeg_exe(), "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
        if audio:
            cmd += ["-ss", f"{audio_offset:.3f}", "-i", audio, "-map", "0:v", "-map", "1:a",
                    "-c:a", "aac", "-b:a", "192k", "-shortest"]
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p",
                "-profile:v", "high", "-movflags", "+faststart", "-color_primaries", "bt709",
                "-color_trc", "bt709", "-colorspace", "bt709"]
        if threads:
            cmd += ["-threads", str(threads)]
        cmd += [self.tmp]
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.n = 0

    def write(self, frame):
        self.p.stdin.write(np.ascontiguousarray(frame, np.uint8).tobytes())
        self.n += 1

    def close(self):
        self.p.stdin.close()
        rc = self.p.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited {rc}")
        os.replace(self.tmp, self.path)
        return self.path


def write_srt(path, captions):
    """Captions [(start s, end s, text)] as a SubRip (.srt) subtitle file."""
    def stamp(s):
        ms = int(round(s * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"
    with open(path, "w", encoding="utf-8") as f:
        for k, (a, b, txt, *_) in enumerate(captions, 1):
            f.write(f"{k}\n{stamp(a)} --> {stamp(b)}\n{txt}\n\n")


def write_wav(path, samples, rate):
    """Mono float samples in [-1, 1] as a 16-bit WAV, with only the standard library."""
    pcm = (np.clip(np.asarray(samples, np.float32), -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())


def read_wav(path):
    with wave.open(path, "rb") as w:
        data = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32767
        return data, w.getframerate()


# ── the film's clock ──────────────────────────────────────────────────────────
class TimeMap:
    """The film's clock against a lesson's own (script) clock. A hold (a, d, tag) stops the
    script clock at a for d seconds of film: the picture stands still there, and render() is
    told which hold it is in and how far through (0..1)."""

    def __init__(self, holds=()):
        self.holds = sorted(holds, key=lambda h: h[0])

    def add(self, a, d, tag):
        self.holds = sorted(self.holds + [(a, d, tag)], key=lambda h: h[0])

    def film(self, ts):
        """Script time -> film time (a moment at a hold maps to before it)."""
        return ts + sum(d for a, d, _ in self.holds if a < ts)

    def script(self, tf):
        """Film time -> (script time, (tag, 0..1 progress) inside a hold, else None)."""
        shift = 0.0
        for a, d, tag in self.holds:
            if tf < a + shift:
                break
            if tf < a + shift + d:
                return a, (tag, (tf - a - shift) / d)
            shift += d
        return tf - shift, None


# ── narration ─────────────────────────────────────────────────────────────────
# Kokoro's G2P reads "PhysX" as "fize-ex" and runs "three.js" together; a [word](/phonemes/)
# link fixes the pronunciation outright.
SPOKEN_WORDS = {"IMU": "I M U", "GNSS": "G N S S", "fps": "frames per second", "threepp": "three p p",
                "FR3": "F R 3", "PhysX": "[PhysX](/fˈɪzˌɛks/)", "three.js": "three J S"}
_UNITS = [("m/s²", "metres per second squared"), ("°/s", "degrees per second"),
          ("ms", "milliseconds"), ("mm", "millimetres"), ("cm", "centimetres"), ("s", "seconds"),
          ("m", "metres"), ("%", "percent"), ("°", "degrees")]
_UNIT_RE = re.compile(r"(\d)[  ]?(" + "|".join(re.escape(u) for u, _ in _UNITS) + r")(?![A-Za-z²])")


def spoken(text, words=None):
    """Caption text as it should be read aloud: units after numbers, and a few acronyms, in words."""
    t = text.replace("×", " by ")
    units = dict(_UNITS)
    t = _UNIT_RE.sub(lambda m: m.group(1) + " " + units[m.group(2)], t)
    for k, v in {**SPOKEN_WORDS, **(words or {})}.items():
        t = re.sub(r"(?<![\w])" + re.escape(k) + r"(?![\w])", v, t)
    return t.replace(" ", " ")


class Narration:
    """Captions read aloud by Kokoro (offline neural TTS, hexgrad/Kokoro-82M), one clip per
    caption, cached as WAV by (voice, speed, spoken text) so a re-render only synthesises the
    lines that changed."""

    RATE = 24000

    def __init__(self, voice="af_heart", speed=1.0, cache_dir="lesson_out/voice_cache", words=None):
        self.voice, self.speed, self.cache_dir, self.words = voice, speed, cache_dir, words
        self._pipe = None
        os.makedirs(cache_dir, exist_ok=True)

    def _path(self, text):
        say = spoken(text, self.words)
        key = hashlib.sha1(f"{self.voice}|{self.speed}|{say}".encode("utf-8")).hexdigest()[:16]
        return say, os.path.join(self.cache_dir, f"{self.voice}_{key}.wav")

    def clip(self, text):
        say, path = self._path(text)
        if os.path.isfile(path):
            return read_wav(path)[0]
        self._synth(say, path)
        return read_wav(path)[0]

    def word_times(self, text):
        """[(word, start, end)] in seconds into the clip, as Kokoro aligned them (punctuation
        dropped), so a picture can cut on a spoken word. Cached beside the clip's WAV."""
        say, path = self._path(text)
        side = path[:-4] + ".words.json"
        if not os.path.isfile(side):
            self._synth(say, path)                  # the WAV again too, so the two always agree
        with open(side, encoding="utf-8") as f:
            return [tuple(w) for w in json.load(f)]

    def _synth(self, say, path):
        if self._pipe is None:
            try:
                from kokoro import KPipeline
            except ImportError as e:
                raise RuntimeError("narration needs Kokoro (pip install kokoro soundfile); "
                                   "or render without it with --no-voice") from e
            # Kokoro voice names start with their language code: a = American, b = British English, ...
            self._pipe = KPipeline(lang_code=self.voice[0], repo_id="hexgrad/Kokoro-82M")
        chunks, words, off = [], [], 0.0
        for r in self._pipe(say, voice=self.voice, speed=self.speed):
            a = np.asarray(r.audio, np.float32)
            for tk in r.tokens or ():
                if tk.start_ts is not None and any(ch.isalnum() for ch in tk.text):
                    words.append((tk.text.strip(".,;:!?"), off + tk.start_ts, off + tk.end_ts))
            chunks.append(a)
            off += len(a) / self.RATE
        write_wav(path, np.concatenate(chunks), self.RATE)
        with open(path[:-4] + ".words.json", "w", encoding="utf-8") as f:
            json.dump([[w, round(a, 3), round(b, 3)] for w, a, b in words], f)


# ── the house rules ───────────────────────────────────────────────────────────
TITLE_VOICE_BY = 2.0                     # the first spoken line starts this early: the title card is narrated
_NUMBER_RE = re.compile(r"\d(?![dD]\b)")   # a digit, except the one in "2D" / "3D"


def house_rule_warnings(captions):
    """The house rules every lesson film follows, as warnings: no specific numbers in the
    narration, and a narrated title card (a caption from the first seconds, spoken-only if the
    title card should stay clean). The picture is the lesson's own to check."""
    out = []
    if not captions or min(c[0] for c in captions) > TITLE_VOICE_BY:
        out.append(f"the title card is not narrated: no caption starts in the first {TITLE_VOICE_BY:.0f} s")
    for a, _, txt, *_ in captions:
        if _NUMBER_RE.search(txt):
            out.append(f"the caption at {a:.1f} s has a number in it: {txt[:70]!r}")
    return out


def card(ov, x, y, w, h, alpha, title=None):
    """The house card: a rounded panel with a small spaced-out title."""
    ov.panel(x, y, w, h, radius=16, alpha=0.74 * alpha, outline=0x8aa0c0, outline_alpha=0.16)
    if title:
        ov.text(x + 26, y + 22, title, size=16, color=DIM, alpha=alpha, kind="semibold", tracking=2.2)


def code_card(ov, t, t_in, t_out, x, y, lines, lang, title, size=20, lh=27, stagger=0.35, min_w=0, out=None,
              out_t=None, keywords=None, out_color=0xa6e3a1):
    """A titled card of code whose lines type in one after another from t_in. `out`: lines
    the program prints, shown under the code from `out_t` (default: once the code is in)."""
    a = envelope(t, t_in, t_out, 0.6, 0.6)
    if a <= 0.003:
        return
    cw = ov.text_width("0", size, "mono")
    w = max(max(len(l) for l in lines + (out or [])) * cw + 60, min_w)
    h = 64 + lh * len(lines) + (lh * len(out) + 26 if out else 0)
    card(ov, x, y, w, h, a, title)
    ov.code(x + 30, y + 50, lines, lang, size, lh, a,
            reveal=lambda j: remap(t, t_in + 0.3 + stagger * j, t_in + 0.9 + stagger * j), keywords=keywords)
    if out:
        oa = a * smooth(remap(t, out_t if out_t is not None else t_in + 0.9 + stagger * len(lines),
                              (out_t if out_t is not None else t_in + 0.9 + stagger * len(lines)) + 0.4))
        yy = y + 50 + lh * len(lines) + 10
        ov.panel(x + 12, yy - 4, w - 24, lh * len(out) + 12, radius=8, fill=0x070a10, alpha=0.8 * oa)
        for j, line in enumerate(out):
            ov.text(x + 30, yy + 2 + lh * j + lh / 2, line, size=size, color=out_color, alpha=oa, kind="mono",
                    anchor="lm")


# ── measured speech ───────────────────────────────────────────────────────────
VOICE_LEAD, VOICE_TAIL = 0.15, 0.45     # speech starts this far into a caption, and ends this far before its end
SLACK = VOICE_LEAD + VOICE_TAIL + 0.3    # a caption laid out from its speech outlasts it by this much


def overruns(spans, seconds, holds=(), margin=0.0):
    """[(i, t, extra)]: every caption i whose line needs more time than it gets, and how much
    more, placed at t, the moment its speech should have ended (VOICE_TAIL before its end).
    A line gets the time from VOICE_LEAD into its caption to t, plus any holds (t, d) in that
    window: the picture stands still there, the voice goes on. Each overrun found counts as a
    hold for the captions after it. `spans` [(start, end)] and `seconds` go in step."""
    holds, out = list(holds), []
    for i, ((a, b), need) in enumerate(zip(spans, seconds)):
        q, p = a + VOICE_LEAD, b - VOICE_TAIL
        have = p - q + sum(d for h, d in holds if q <= h < p)
        if need + margin > have:
            out.append((i, p, need + margin - have))
            holds.append((p, need + margin - have))
    return out


class Speech:
    """A lesson's lines by key, and how long each takes to say. A lesson whose narration sets
    its pace lays its timeline out from these lengths, so the lengths must not move by
    themselves: they are measured once (`--remeasure`) and committed next to the lesson as
    <lesson>.speech.json, {voice: {key: {"s": seconds, "said": spoken text}}}. A line with no
    measurement gets an estimate from its word count, reported; a line whose text changed
    since it was measured is reported by `run`. `words` is the lesson's own pronunciation
    table (see `spoken`); `voice_only` lines are read aloud but not drawn."""

    def __init__(self, lines, cache, voice="af_heart", words=None, voice_only=()):
        self.lines, self.cache, self.voice = dict(lines), cache, voice
        self.words, self.voice_only = dict(words or {}), set(voice_only)
        self.measured = {}
        if os.path.isfile(cache):
            with open(cache, encoding="utf-8") as fh:
                self.measured = json.load(fh).get(voice, {})
        self.estimated, self.final = [], {}

    def seconds(self, k):
        m = self.measured.get(str(k))
        if m is not None:
            return m["s"]
        if self.lines[k] is None:
            raise KeyError(f"line {k!r} has no text yet and no measurement in {self.cache}")
        if k not in self.estimated:
            self.estimated.append(k)
            print(f"[speech] line {k!r} is not measured for {self.voice}; estimating from its words "
                  f"(--remeasure measures it)")
        return 0.42 * len(self.lines[k].split()) + 0.4

    def span(self, k, start, slack=SLACK):
        """Caption k from `start`, as long as its line takes to say, plus `slack`."""
        return (start, start + self.seconds(k) + slack)

    def layout(self, plan, gap=0.2, slack=SLACK):
        """(Timeline, {key: (start, end)}) from a plan of beats [(name, lead-in, keys, tail)]:
        each beat's captions follow each other `gap` apart, after its lead-in, and the beat
        ends `tail` after its last one (or after its lead-in's start when it has none)."""
        tl, cap, t = Timeline(), {}, 0.0
        for name, lead, keys, tail in plan:
            c = t + lead
            for k in keys:
                cap[k] = (c, c + self.seconds(k) + slack)
                c = cap[k][1] + gap
            end = (c - gap if keys else t) + tail
            tl.add(name, t, end)
            t = end
        return tl, cap

    def stretch(self, spans, holds=(), margin=0.3):
        """For captions written at fixed times, {key: (start, end)}: a TimeMap whose `.film`
        maps the written clock onto one with more time just before a caption ends, wherever
        its line (plus `margin`) would run over. `holds` [(t, d)] on the written clock count
        as time to speak."""
        keys = list(spans)
        found = overruns([spans[k] for k in keys], [self.seconds(k) for k in keys], holds, margin)
        return TimeMap([(t, d, ("speech", keys[i])) for i, t, d in found])

    def captions(self, spans, fields=None, texts=None):
        """[(start, end, text)] for run(), one per {key: (start, end)}. `texts` holds whole
        lines written after the choreography (a line in `lines` may be None until then);
        `fields` fills the {names} in the rest. Voice-only lines get the not-drawn flag."""
        out = []
        for k, (a, b) in spans.items():
            txt = texts[k] if texts and k in texts else self.lines[k]
            if fields is not None:
                txt = txt.format(**fields)
            self.final[k] = txt
            out.append((a, b, txt) + ((False,) if k in self.voice_only else ()))
        return out

    def stale(self):
        """The lines whose final text is not what was measured."""
        return [k for k, txt in self.final.items()
                if str(k) in self.measured and self.measured[str(k)].get("said") != spoken(txt, self.words)]

    def remeasure(self, cache_dir, voice=None):
        """Say every final line with Kokoro (through the voice cache) and write the lengths."""
        voice = voice or self.voice
        narr = Narration(voice, cache_dir=cache_dir, words=self.words)
        data = {}
        if os.path.isfile(self.cache):
            with open(self.cache, encoding="utf-8") as fh:
                data = json.load(fh)
        old = data.get(voice, {})
        new = {str(k): {"s": round(len(narr.clip(txt)) / Narration.RATE, 2), "said": spoken(txt, self.words)}
               for k, txt in self.final.items()}
        for k, v in new.items():
            if old.get(k, {}).get("s") != v["s"]:
                print(f"[speech] {k}: {old.get(k, {}).get('s')} -> {v['s']} s")
        data[voice] = new
        with open(self.cache, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, indent=1, ensure_ascii=False)
            fh.write("\n")
        print(f"[speech] wrote {self.cache}: {len(new)} lines for {voice}")


# ── the command line ──────────────────────────────────────────────────────────


def _contact_sheet(thumbs, cols=6):
    th, tw = thumbs[0].shape[:2]
    rows = (len(thumbs) + cols - 1) // cols
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, im in enumerate(thumbs):
        y, x = (k // cols) * th, (k % cols) * tw
        sheet[y:y + th, x:x + tw] = im
    return sheet


def _write_gate(out, name, captions, clips, narr, tm, duration, film_duration, frame, W, every):
    """`--gate DIR`: what a refactor of the toolkit must leave alone, from one setup().
    <name>.srt, the captions on the film clock (as --srt writes them); <name>.voice.json,
    every caption's clocks, text and spoken text, and every hold; <name>_sheet.png, one frame
    every `every` s (as --sheet makes it); and stills/, a full-size frame at the middle of
    every caption. Frames are rendered in film order, so a lesson that streams its pictures
    (warp_threepp's subprocess) never has to rewind."""
    os.makedirs(os.path.join(out, "stills"), exist_ok=True)
    write_srt(os.path.join(out, f"{name}.srt"), [(tm.film(a), tm.film(b), txt) for a, b, txt, *_ in captions])
    words = narr.words if narr is not None else None
    record = {
        "voice": narr.voice if narr is not None else None,
        "duration": duration, "film_duration": film_duration,
        "captions": [{"start": a, "end": b, "film_start": tm.film(a), "film_end": tm.film(b),
                      "text": txt, "drawn": bool(rest[0]) if rest else True, "spoken": spoken(txt, words),
                      "samples": None if clips is None else int(len(clips[k]))}
                     for k, (a, b, txt, *rest) in enumerate(captions)],
        "holds": [[a, d, list(tag) if isinstance(tag, tuple) else tag] for a, d, tag in tm.holds],
    }
    with open(os.path.join(out, f"{name}.voice.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1, ensure_ascii=False, default=repr)

    f = max(1, W // 480)
    jobs = [(float(t), "sheet", k) for k, t in enumerate(np.arange(0.5, film_duration, every))]
    jobs += [(0.5 * (tm.film(a) + tm.film(b)), "still", k) for k, (a, b, *_) in enumerate(captions)]
    thumbs = {}
    t0 = time.time()
    for n, (t, kind, k) in enumerate(sorted(jobs)):
        img = frame(t)
        if kind == "sheet":
            thumbs[k] = shrink(img, f)
        else:
            write_png(os.path.join(out, "stills", f"{name}_cap{k:02d}_{t:07.2f}.png"), img)
        if n % 10 == 0:
            print(f"[gate] {n}/{len(jobs)} frames, {time.time() - t0:.0f}s", flush=True)
    write_png(os.path.join(out, f"{name}_sheet.png"), _contact_sheet([thumbs[k] for k in sorted(thumbs)]))
    print(f"[gate] wrote {out}: {len(captions)} captions, {len(tm.holds)} holds, "
          f"{len(thumbs)} sheet frames, {len(captions)} stills ({time.time() - t0:.0f}s)")


def run(name, duration, setup, fps=60, speech=None):
    """The command line every lesson shares.

    `setup(width, height)` builds the lesson at that render size and returns
    `(render, captions)`. `render(t)` gives the (H, W, 3) frame at t seconds on the
    lesson's own clock. A lesson that declares holds (`render.holds = [(t, seconds, tag)]`)
    takes `render(t, hold)`, where hold is (tag, 0..1) while the film stands still at t.
    `captions` is the [(start, end, text)] list the film shows, on the lesson's clock.
    An entry (start, end, text, False) is read aloud and written to the .srt but not drawn:
    use it to narrate the title card, so the film has sound from the start.

    The captions are read aloud (Kokoro, voice `--voice`) unless `--no-voice`. Where a
    line runs longer than its caption, the film holds the picture until it has been said.
    A lesson paced by its narration passes its `speech` (a Speech): its pronunciation table
    reaches the narrator, and `--remeasure` measures its final lines and rewrites the
    .speech.json its layout is read from.

        (no flags)            the film, 1920x1080 at `fps`, to <outdir>/<name>.mp4
        --preview             960x540 at 30 fps
        --from S --to S       one stretch of the film
        --stills 5,15,40      single frames, <outdir>/<name>_still_<t>.png
        --sheet [--every S]   contact sheet, one frame every S seconds from 0.5 s
        --srt                 the captions as <outdir>/<name>.srt
        --gate DIR            a regression record for refactors, into DIR (see `_write_gate`)
        --remeasure           say every line (Kokoro, --voice) and rewrite <lesson>.speech.json
    Times on the command line are film times.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help=f"the film (default: <outdir>/{name}.mp4)")
    ap.add_argument("--outdir", default="lesson_out", help="stills, sheets, subtitles and the default film")
    ap.add_argument("--stills", default=None, help="comma-separated times (s)")
    ap.add_argument("--sheet", action="store_true", help="contact sheet, one frame every --every seconds")
    ap.add_argument("--every", type=float, default=3.0)
    ap.add_argument("--srt", action="store_true", help=f"write the captions to <outdir>/{name}.srt")
    ap.add_argument("--voice", default="af_heart", help="Kokoro voice for the narration (default af_heart)")
    ap.add_argument("--no-voice", dest="no_voice", action="store_true", help="no narration: a silent film")
    ap.add_argument("--preview", action="store_true", help="960x540 @ 30 fps")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--gate", default=None, metavar="DIR",
                    help="write the captions, the spoken lines and holds, a sheet and a still per caption to DIR")
    ap.add_argument("--remeasure", action="store_true", help="measure every spoken line and rewrite the "
                    "lesson's .speech.json (the layout moves on the next run)")
    args = ap.parse_args()
    if args.remeasure and speech is None:
        ap.error("this lesson is not laid out from measured speech")
    if speech is not None and args.voice != speech.voice and not args.no_voice:
        print(f"[speech] the layout is timed for {speech.voice}; {args.voice} gets holds wherever it runs long")

    W, H = (960, 540) if args.preview else (1920, 1080)
    fps = args.fps or (30 if args.preview else fps)
    render, captions = setup(W, H)
    for msg in house_rule_warnings(captions):
        print("[house rules]", msg)
    os.makedirs(args.outdir, exist_ok=True)
    if speech is not None:
        if args.remeasure:
            speech.remeasure(os.path.join(args.outdir, "voice_cache"), args.voice)
            return
        stale = speech.stale()
        if stale:
            print(f"[speech] these lines changed since they were measured: {stale} (--remeasure)")

    tm = TimeMap(getattr(render, "holds", ()))
    clips = narr = None
    if not args.no_voice:
        narr = Narration(args.voice, cache_dir=os.path.join(args.outdir, "voice_cache"),
                         words=speech.words if speech is not None else None)
        t0 = time.time()
        clips = [narr.clip(c[2]) for c in captions]
        said = sum(len(c) for c in clips) / Narration.RATE
        for _, t, d in overruns([c[:2] for c in captions], [len(c) / Narration.RATE for c in clips],
                                [(a, d) for a, d, _ in tm.holds]):
            tm.add(t, d, ("voice", None))
        held = sum(d for _, d, tag in tm.holds if tag[0] == "voice")
        print(f"[voice] {len(clips)} lines, {said:.0f} s of speech ({args.voice}) in {time.time() - t0:.1f}s; "
              f"the film holds {held:.1f} s for it")
    film_duration = tm.film(duration)
    takes_hold = len(inspect.signature(render).parameters) >= 2

    def frame(t):
        ts, hold = tm.script(t)
        return render(ts, hold) if takes_hold else render(ts)

    if args.gate:
        _write_gate(args.gate, name, captions, clips, narr, tm, duration, film_duration, frame, W, args.every)
        return
    if args.srt:
        p = os.path.join(args.outdir, f"{name}.srt")
        write_srt(p, [(tm.film(a), tm.film(b), txt) for a, b, txt, *_ in captions])
        print("saved", p, f"({len(captions)} captions)")
        return
    if args.stills:
        for tok in args.stills.split(","):
            t = float(tok)
            p = os.path.join(args.outdir, f"{name}_still_{t:06.2f}.png")
            write_png(p, frame(t))
            print("saved", p)
        return
    if args.sheet:
        f = max(1, W // 480)
        thumbs = [shrink(frame(float(t)), f) for t in np.arange(0.5, film_duration, args.every)]
        p = os.path.join(args.outdir, f"{name}_sheet.png")
        write_png(p, _contact_sheet(thumbs))
        print("saved", p, f"({len(thumbs)} frames, every {args.every} s from 0.5 s, row-major)")
        return

    audio = None
    if clips is not None:
        track = np.zeros(int(math.ceil(film_duration * Narration.RATE)) + 1, np.float32)
        for (a, *_), c in zip(captions, clips):
            i0 = int(round((tm.film(a) + VOICE_LEAD) * Narration.RATE))
            n = min(len(c), len(track) - i0)
            track[i0:i0 + n] += c[:n]
        audio = os.path.join(args.outdir, f"{name}.voice.wav")
        write_wav(audio, track, Narration.RATE)
    t_to = args.t_to if args.t_to is not None else film_duration
    out = args.out or os.path.join(args.outdir, f"{name}.mp4")
    film = Film(out, W, H, fps=fps, crf=16 if not args.preview else 22,
                preset="slow" if not args.preview else "veryfast", audio=audio, audio_offset=args.t_from)
    nfr = int(round((t_to - args.t_from) * fps))
    t_start = time.time()
    for f in range(nfr):
        t = args.t_from + f / fps
        film.write(frame(t))
        if f % (fps * 5) == 0:
            el = time.time() - t_start
            print(f"[film] {t:6.1f}s  frame {f}/{nfr}  {el / max(f, 1) * 1000:.0f} ms/frame", flush=True)
    path = film.close()
    print(f"[film] wrote {path} ({nfr} frames, {time.time() - t_start:.0f}s)")
