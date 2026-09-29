"""The 3D side of a lesson: the studio (`Stage`), poses, and the annotation kit: arrows,
rings, markers, tube trails, point clouds and line fans that fade and ride on any object."""
from __future__ import annotations

import math
import os

import numpy as np

import threepp as tp

from .timing import clamp01

__all__ = ["hex_rgb", "to_hex", "standard", "Stage", "mat4", "set_pose", "quat_y_to", "xray", "ghost",
           "Arrow3D", "Ring3D", "Marker3D", "tube_arrays", "Tube3D", "turbo", "turbo_hex", "disc_texture",
           "Cloud", "Segments"]


def hex_rgb(h):
    """0xRRGGBB or '#rrggbb' -> (r, g, b) ints."""
    if isinstance(h, str):
        h = int(h.lstrip("#"), 16)
    return (h >> 16) & 255, (h >> 8) & 255, h & 255


def to_hex(h):
    if isinstance(h, str):
        return int(h.lstrip("#"), 16)
    return int(h)


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

    def __init__(self, width=1920, height=1080, renderer=None, headless=True, msaa=8,
                 floor=True, env=None, env_intensity=0.35, design=(1920, 1080),
                 fog=None, shadow_extent=1.6, far=60.0):
        """`renderer`: a GLRenderer whose canvas is width x height (the Stage sets its tone
        mapping, exposure and shadows), or None for a new headless canvas and renderer.
        `env`: the image-based light, an equirectangular HDR's path or a loaded texture
        (None: none). `fog` = (near, far) in metres (default 3.5..9 with the studio floor,
        none without). `shadow_extent` is the half-size of the key light's shadow area
        around its target; `follow()` moves that area with a subject. `far` is the camera's
        far plane."""
        self.W, self.H = int(width), int(height)
        self.design = design     # project() answers in these units, the Hud's
        if renderer is None:
            self.canvas = tp.Canvas("lesson", width=self.W, height=self.H, antialiasing=msaa, headless=headless)
            self.r = tp.GLRenderer(self.canvas)
        else:
            self.canvas, self.r = None, renderer
        self.r.shadow_map_enabled = True
        self.r.tone_mapping = tp.ToneMapping.ACESFilmic
        self.r.tone_mapping_exposure = 0.95
        self.frame_index = 0

        self.scene = tp.Scene()
        self.scene.background = tp.Background(self.BG)
        if env is not None:
            self.scene.environment = tp.RGBELoader().load(env) if isinstance(env, (str, os.PathLike)) else env
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
        """Render the scene (and the Hud over it) and return (H, W, 3) uint8. `t` is the
        film time; the GL renderer does not need it."""
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
