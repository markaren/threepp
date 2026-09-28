"""Hold still: PID control, flown on a rocket.  Part 4 of the threepp lessons.

    python rocket_pid.py                               # the film (1080p60), lesson_out/rocket_pid.mp4
    python rocket_pid.py --stills 10,50,85             # individual frames, into lesson_out/
    python rocket_pid.py --sheet                       # contact sheet
    python rocket_pid.py --preview --out preview.mp4   # 960x540 @ 30 fps
    python rocket_pid.py --srt                         # the captions as lesson_out/rocket_pid.srt

A test rocket (a "hopper": one engine, four legs) flies in PhysX. Its engine pushes along
the body with a thrust the controller sets 240 times a second; the engine answers a
fifth of a second late and cannot throttle below 35 %. A gimbal tips the thrust to keep
the rocket upright. The altitude controller is built up on screen, P, then D, then I,
and every flight, bounce and overshoot the film shows is what PhysX did with it. The
exhaust, smoke and dust are particles stepped by NVIDIA Warp kernels on the GPU, fed by
the recorded engine state, and drawn by threepp.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import lesson  # noqa: E402
from lesson import (DIM, TEXT, Arrow3D, Hud, OrbitCamera, Stage, Timeline, clamp01,  # noqa: E402
                    ease_out, envelope, fade_in_out, ghost, remap, run, smooth, standard, tp)

import warp as wp  # noqa: E402

FPS = 60
DT = 1 / 240                 # physics step = one control update
SUB = 4                      # physics steps per film frame
G = 9.81

# the vehicle
BODY_H, BODY_R = 12.0, 0.9
MASS = 6000.0
T_MAX_W = 1.5                # full thrust / weight
T_MIN = 0.35                 # the lowest throttle a lit engine can hold
TAU = 0.2                    # engine spool lag (s)
GIMBAL = math.radians(8.0)   # how far the engine can swing
NOZZLE_Y = 0.35              # the exit plane, body frame
LEG_R = 2.6                  # feet from the axis
TARGET = 20.0                # the hover height (m)

# the controller, in throttle units (1 = full thrust)
KP, KI, KD = 0.10, 0.03, 0.15
# attitude (gimbal angle per radian of tilt, per rad/s) and the lean back over the pad (rad per m, per m/s)
ATT_KP, ATT_KD = 2.0, 1.2
LAT_KP, LAT_KD, LEAN_MAX = 0.03, 0.10, 0.10

C_P = 0xff9f43
C_I = 0xb983ff
C_D = 0x4cc9f0
C_TGT = 0x5ee27a
C_BAD = 0xff5d73
C_TEXT = TEXT
C_DIM = DIM


# ── flight physics ────────────────────────────────────────────────────────────
def q2R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def hull_mesh():
    """The collision shape: the body and the four feet, as one convex hull."""
    pts = []
    for k in range(16):
        a = 2 * math.pi * k / 16
        for y in (1.2, BODY_H):
            pts.append((BODY_R * math.cos(a), y, BODY_R * math.sin(a)))
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        for y in (0.0, 0.15):
            pts.append((LEG_R * math.cos(a), y, LEG_R * math.sin(a)))
    g = tp.BufferGeometry()
    g.set_attribute("position", np.array(pts, np.float32))
    return tp.Mesh(g, tp.MeshBasicMaterial())


class Flight:
    """One flight, recorded per film frame (60 Hz) from a 240 Hz PhysX run."""
    pass


def fly(seconds, ignite=0.5, kp=KP, ki=0.0, kd=0.0, ki_from=None, anti_windup=True, target=None,
        gust=None, cut_on_impact=False, land_from=None):
    """Fly the hopper from the pad. Times are seconds from the start of the flight.
    `ki_from` switches the I term on part-way; `target(t)` moves the set point; `gust` is
    (t0, t1, sideways force / weight); `cut_on_impact` stops the engine when the rocket
    hits the ground on the way down; `land_from` cuts it at touchdown after that time."""
    world = tp.PhysxWorld(fixed_timestep=DT)
    ground = tp.Mesh(tp.BoxGeometry(400, 1, 400), tp.MeshBasicMaterial())
    ground.position.y = -0.5
    world.add_static(ground, world.create_material(0.8, 0.7, 0.05))
    probe = world.add_dynamic_convex(hull_mesh(), 1.0)
    density = MASS / probe.mass
    world.remove(probe)
    body = world.add_dynamic_convex(hull_mesh(), density)
    mass = body.mass
    t_max = T_MAX_W * mass * G

    n_frames = int(round(seconds * FPS)) + 1
    fl = Flight()
    fl.mass, fl.hover = mass, 1.0 / T_MAX_W
    fl.pos = np.zeros((n_frames, 3))
    fl.quat = np.zeros((n_frames, 4))
    fl.vel = np.zeros((n_frames, 3))
    fl.thr = np.zeros(n_frames)          # thrust / full thrust, as the engine delivers it
    fl.cmd = np.zeros(n_frames)          # what the controller asked for (unclipped)
    fl.terms = np.zeros((n_frames, 3))   # P, I, D contributions to the command
    fl.tgt = np.zeros(n_frames)
    fl.gim = np.zeros((n_frames, 2))
    fl.gust = np.zeros(n_frames)
    fl.lit = np.zeros(n_frames, bool)
    fl.impact = None                     # (t, vertical speed) of a hard landing
    fl.touchdown = None                  # (t, vertical speed) of a landing
    thr, integ, lit = 0.0, 0.0, False
    climbed = False
    for k in range(n_frames * SUB):
        t = k * DT
        p, q = body.position, body.quaternion
        v, w = body.linear_velocity, body.angular_velocity
        pos = np.array([p.x, p.y, p.z])
        vel = np.array([v.x, v.y, v.z])
        Rm = q2R((q.x, q.y, q.z, q.w))
        up = Rm[:, 1]
        climbed |= pos[1] > 3.0
        if not lit and t >= ignite and fl.impact is None and fl.touchdown is None:
            lit = True
        tgt = target(t) if target is not None else TARGET
        e = tgt - pos[1]
        ki_now = ki if (ki_from is None or t >= ki_from) else 0.0
        P, I, D = kp * e, ki_now * integ, -kd * vel[1]
        cmd = P + I + D
        if ki_now > 0 and lit:
            wind = (cmd > 1.0 and e > 0) or (cmd < T_MIN and e < 0)
            if not (anti_windup and wind):
                integ += e * DT
        if lit and cut_on_impact and climbed and pos[1] < 0.25 and vel[1] < -1.0:
            fl.impact = (t, float(vel[1]))
            lit = False
        if lit and land_from is not None and t >= land_from and pos[1] < 0.3:
            fl.touchdown = (t, float(vel[1]))
            lit = False
        want = min(max(cmd, T_MIN), 1.0) if lit else 0.0
        thr += (want - thr) * (DT / TAU)
        # attitude: the gimbal leans the thrust against the tilt, and a slow position loop
        # asks for a small lean back toward the pad
        lat, latv = pos[[0, 2]], vel[[0, 2]]
        lean = np.clip(-LAT_KP * lat - LAT_KD * latv, -LEAN_MAX, LEAN_MAX)
        tilt = np.array([up[0], up[2]])
        rate = np.array([-w.z, w.x])                      # d(up)/dt = w x up, near upright
        gim = np.clip(ATT_KP * (tilt - lean) + ATT_KD * rate, -GIMBAL, GIMBAL)
        d = up + np.array([gim[0], 0.0, gim[1]])
        d /= np.linalg.norm(d)
        F = thr * t_max
        if F > 1.0:
            nozzle = pos + Rm @ np.array([0.0, NOZZLE_Y, 0.0])
            body.add_force_at_pos(tp.Vector3(*(F * d)), tp.Vector3(*nozzle))
        g_on = gust is not None and gust[0] <= t < gust[1]
        if g_on:
            body.add_force(tp.Vector3(gust[2] * mass * G, 0.0, 0.0))
        world.step(DT)
        if k % SUB == SUB - 1:
            f = k // SUB
            p, q, v = body.position, body.quaternion, body.linear_velocity
            fl.pos[f] = (p.x, p.y, p.z)
            fl.quat[f] = (q.x, q.y, q.z, q.w)
            fl.vel[f] = (v.x, v.y, v.z)
            fl.thr[f] = thr
            fl.cmd[f] = cmd if lit else 0.0
            fl.terms[f] = (P, I, D) if lit else (0.0, 0.0, 0.0)
            fl.tgt[f] = tgt
            fl.gim[f] = gim
            fl.gust[f] = 1.0 if g_on else 0.0
            fl.lit[f] = lit
    fl.n = n_frames
    fl.t = np.arange(n_frames) / FPS
    return fl


# ── the set ───────────────────────────────────────────────────────────────────
def _grain(n, seed, scale=1.0):
    rng = np.random.default_rng(seed)
    g = rng.normal(0.0, 1.0, (n, n))
    for _ in range(2):
        g = (g + np.roll(g, 1, 0) + np.roll(g, 1, 1) + np.roll(g, (1, 1), (0, 1))) / 4
    return g * scale


def pad_texture(n=2048, radius=14.0):
    """The landing pad seen from above: concrete, a painted ring and cross, and a scorch."""
    y, x = (np.mgrid[0:n, 0:n] + 0.5) / n * 2 * radius - radius
    r = np.hypot(x, y)
    base = 0.58 + 0.04 * _grain(n, 3) + 0.015 * _grain(n // 4, 4).repeat(4, 0).repeat(4, 1)
    jx = np.clip(1.0 - np.abs(((x + 2.0) % 4.0) - 2.0) / 0.07, 0, 1)
    jy = np.clip(1.0 - np.abs(((y + 2.0) % 4.0) - 2.0) / 0.07, 0, 1)
    base = base * (1.0 - 0.22 * np.maximum(jx, jy))
    scorch = np.exp(-(r / 5.0) ** 2) * (0.55 + 0.25 * _grain(n, 5))
    v = base * (1.0 - 0.8 * np.clip(scorch, 0, 1))
    rgb = np.stack([v * 1.0, v * 0.98, v * 0.95], -1)
    paint = np.array([0.95, 0.72, 0.12])
    ring = np.clip(1.0 - np.abs(r - 9.0) / 0.22, 0, 1)
    xa, ya = np.abs(x - y) / math.sqrt(2), np.abs(x + y) / math.sqrt(2)
    cross = np.clip(1.0 - np.minimum(xa, ya) / 0.3, 0, 1) * (r < 6.0) * (r > 1.2)
    m = np.clip(ring + cross, 0, 1)[..., None] * (1.0 - 0.6 * np.clip(scorch, 0, 1))[..., None]
    rgb = rgb * (1 - m) + paint * m
    edge = np.clip((radius - 0.15 - r) / 0.1, 0, 1)[..., None]
    rgb = rgb * edge + 0.12 * (1 - edge)
    img = np.clip(np.concatenate([rgb, np.ones_like(rgb[..., :1])], -1) * 255, 0, 255).astype(np.uint8)
    tex = tp.data_texture(img, True)
    tex.anisotropy = 16
    return tex


def ground_texture(n=512):
    v = 0.34 + 0.05 * _grain(n, 7) + 0.03 * _grain(n // 8, 8).repeat(8, 0).repeat(8, 1)
    rgb = np.stack([v * 1.02, v, v * 0.96, np.ones_like(v)], -1)
    tex = tp.data_texture(np.clip(rgb * 255, 0, 255).astype(np.uint8), True)
    tex.wrap_s = tp.TextureWrapping.Repeat
    tex.wrap_t = tp.TextureWrapping.Repeat
    tex.anisotropy = 16
    return tex


SKY_ZENITH = np.array([0.012, 0.018, 0.045])
SKY_MID = np.array([0.06, 0.07, 0.14])
SKY_HORIZON = np.array([0.42, 0.2, 0.11])
FOG = 0x6b4a3c


def sky_dome(radius=900.0):
    """Blue hour: a gradient dome (vertex colours, unlit, no fog) and a few stars."""
    g = tp.SphereGeometry(radius, 64, 32)
    p = g.get_attribute("position")
    y = np.asarray(p)[:, 1] / radius
    u = np.clip(y, -1, 1)
    c = np.empty((len(u), 3), np.float32)
    lo = np.clip(u / 0.12, 0, 1)[:, None]
    hi = np.clip((u - 0.12) / 0.6, 0, 1)[:, None]
    c[:] = SKY_HORIZON * (1 - lo) + SKY_MID * lo
    c[:] = c * (1 - hi) + SKY_ZENITH * hi
    c[u < 0] = SKY_HORIZON * 0.35
    g.set_attribute("color", c)
    m = tp.MeshBasicMaterial()
    m.vertex_colors = True
    m.side = tp.Side.Back
    m.fog = False
    m.depth_write = False
    dome = tp.Mesh(g, m)
    dome.render_order = -10
    rng = np.random.default_rng(21)
    n = 700
    az = rng.uniform(0, 2 * math.pi, n)
    el = np.arcsin(rng.uniform(0.12, 1.0, n))
    sp = np.c_[np.cos(el) * np.cos(az), np.sin(el), np.cos(el) * np.sin(az)] * (radius * 0.95)
    sg = tp.BufferGeometry()
    sg.set_attribute("position", sp.astype(np.float32))
    b = (rng.uniform(0.3, 1.0, n) * np.clip((np.sin(el) - 0.12) / 0.4, 0, 1)) ** 1.5
    sg.set_attribute("color", np.c_[b, b, b * 1.1].astype(np.float32))
    sm = tp.PointsMaterial()
    sm.size = 2.4
    sm.size_attenuation = False
    sm.vertex_colors = True
    sm.transparent = True
    sm.blending = tp.Blending.Additive
    sm.depth_write = False
    sm.fog = False
    stars = tp.Points(sg, sm)
    return dome, stars


def build_rocket():
    """A stainless hopper from primitives: body, dome, engine skirt and bell, four legs."""
    root = tp.Group()
    steel = standard(0xc9ccd1, roughness=0.32, metalness=0.9)
    steel.env_map_intensity = 1.2
    dark = standard(0x1b1d22, roughness=0.55, metalness=0.4)
    soot = standard(0x2a2522, roughness=0.8, metalness=0.2)

    def add(geom, mat, y=0.0, x=0.0, z=0.0, shadow=True):
        m = tp.Mesh(geom, mat)
        m.position.set(x, y, z)
        m.cast_shadow = shadow
        m.receive_shadow = True
        root.add(m)
        return m

    add(tp.CylinderGeometry(BODY_R, BODY_R, 9.6, 64, 1), steel, y=1.6 + 4.8)
    nose = add(tp.SphereGeometry(BODY_R, 48, 16, 0, 2 * math.pi, 0, math.pi / 2), steel, y=11.2)
    nose.scale.set(1.0, 0.9, 1.0)
    for yy in (2.6, 4.4, 6.2, 8.0, 9.8):                      # weld seams
        add(tp.TorusGeometry(BODY_R + 0.004, 0.012, 6, 96), dark, y=yy).rotate_x(math.pi / 2)
    band = standard(0x15171c, roughness=0.45, metalness=0.3)
    add(tp.CylinderGeometry(BODY_R + 0.006, BODY_R + 0.006, 0.9, 64, 1, True), band, y=10.1)
    add(tp.CylinderGeometry(BODY_R + 0.05, BODY_R + 0.12, 0.8, 64, 1), soot, y=1.2)      # engine skirt
    bell = standard(0x3b3130, roughness=0.5, metalness=0.7)
    bell.side = tp.Side.Double
    add(tp.CylinderGeometry(0.22, 0.52, 0.75, 48, 1, True), bell, y=NOZZLE_Y + 0.375)
    # the glow inside the bell, lit by the engine
    glow = tp.MeshBasicMaterial()
    glow.color = tp.Color(0xffb060)
    glow.transparent = True
    glow.blending = tp.Blending.Additive
    glow.depth_write = False
    inner = add(tp.CircleGeometry(0.5, 32), glow, y=NOZZLE_Y + 0.02, shadow=False)
    inner.rotate_x(math.pi / 2)
    # RCS pods and a camera fairing
    for k in range(4):
        a = k * math.pi / 2
        add(tp.BoxGeometry(0.22, 0.4, 0.22), dark, y=10.9, x=(BODY_R + 0.08) * math.cos(a),
            z=(BODY_R + 0.08) * math.sin(a))
    # legs: an A-frame per foot, a foot pad
    leg = standard(0x2b2e35, roughness=0.5, metalness=0.6)
    for k in range(4):
        a = math.pi / 4 + k * math.pi / 2
        c, s = math.cos(a), math.sin(a)
        foot = np.array([LEG_R * c, 0.12, LEG_R * s])
        for top_y, rr in ((4.2, 0.075), (1.7, 0.06)):
            top = np.array([BODY_R * c, top_y, BODY_R * s])
            d = foot - top
            L = float(np.linalg.norm(d))
            m = add(tp.CylinderGeometry(rr, rr, L, 12), leg)
            mid = 0.5 * (top + foot)
            m.position.set(*mid)
            q = lesson._quat_y_to(d / L)
            m.quaternion.set(*q)
        add(tp.CylinderGeometry(0.34, 0.4, 0.14, 24), leg, y=0.07, x=foot[0], z=foot[2])
    return root, glow


# ── exhaust, smoke and dust: Warp kernels ─────────────────────────────────────
@wp.kernel
def _emit_exhaust(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), age: wp.array(dtype=float),
                  life: wp.array(dtype=float), kind: wp.array(dtype=int), start: int, cap: int, seed: int,
                  n0: wp.vec3, n1: wp.vec3, d: wp.vec3, speed: float, spread: float, dt: float, life0: float):
    tid = wp.tid()
    i = (start + tid) % cap
    s = wp.rand_init(seed, tid)
    u = wp.randf(s)
    j = wp.vec3(wp.randn(s), wp.randn(s), wp.randn(s))
    dirv = wp.normalize(d + j * spread)
    # a point on the exit plane: the nozzle mouth is ~0.5 m across
    side = wp.normalize(wp.cross(d, wp.vec3(0.3, 0.1, 0.9)))
    side2 = wp.cross(d, side)
    a = wp.randf(s) * 6.2831853
    rr = 0.45 * wp.sqrt(wp.randf(s))
    sp = speed * (0.8 + 0.4 * wp.randf(s))
    back = (1.0 - u) * dt
    pos[i] = n0 + (n1 - n0) * u + (side * wp.cos(a) + side2 * wp.sin(a)) * rr + dirv * sp * back
    vel[i] = dirv * sp
    age[i] = back
    life[i] = life0 * (0.7 + 0.6 * wp.randf(s))
    kind[i] = 0


@wp.kernel
def _emit_dust(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), age: wp.array(dtype=float),
               life: wp.array(dtype=float), kind: wp.array(dtype=int), start: int, cap: int, seed: int,
               c: wp.vec3, r0: float, vr: float):
    tid = wp.tid()
    i = (start + tid) % cap
    s = wp.rand_init(seed, tid)
    a = wp.randf(s) * 6.2831853
    rad = wp.vec3(wp.cos(a), 0.0, wp.sin(a))
    pos[i] = c + rad * (r0 * (0.4 + 0.8 * wp.randf(s))) + wp.vec3(0.0, 0.05 + 0.2 * wp.randf(s), 0.0)
    vel[i] = rad * (vr * (0.4 + 0.9 * wp.randf(s))) + wp.vec3(0.0, 0.6 + 2.8 * wp.randf(s), 0.0)
    age[i] = 0.0
    life[i] = 2.5 + 3.5 * wp.randf(s)
    kind[i] = 1


@wp.kernel
def _step(pos: wp.array(dtype=wp.vec3), vel: wp.array(dtype=wp.vec3), age: wp.array(dtype=float),
          life: wp.array(dtype=float), kind: wp.array(dtype=int), dt: float, seed: int):
    i = wp.tid()
    a = age[i]
    if a >= life[i]:
        return
    a = a + dt
    age[i] = a
    v = vel[i]
    p = pos[i]
    k = kind[i]
    if k == 0:                  # exhaust: fast, slows in the air, and the cooling gas starts to rise
        v = v * wp.exp(-dt * 3.2) + wp.vec3(0.0, 4.0 * dt * wp.min(a * 3.0, 1.0), 0.0)
    elif k == 2:                # exhaust turned along the ground: a rolling cloud
        v = v * wp.exp(-dt * 1.4) + wp.vec3(0.0, (0.8 + 1.6 * wp.min(a, 2.0)) * dt, 0.0)
    else:                       # dust: kicked out, drifts, settles slowly
        v = v * wp.exp(-dt * 1.1) + wp.vec3(0.0, -1.2 * dt, 0.0)
    p = p + v * dt
    if p[1] < 0.08:
        if k == 0:
            s = wp.rand_init(seed, i)
            h = wp.vec3(v[0], 0.0, v[2])
            hn = wp.length(h)
            vy = wp.abs(v[1])
            dirh = wp.vec3(1.0, 0.0, 0.0)
            if hn > 1.0e-4:
                dirh = h / hn
            v = dirh * (0.45 * vy + hn) * (0.6 + 0.6 * wp.randf(s)) + wp.vec3(0.0, 0.15 * vy * wp.randf(s), 0.0)
            kind[i] = 2
            life[i] = a + 1.5 + 2.5 * wp.randf(s)
        else:
            v = wp.vec3(v[0] * 0.7, wp.abs(v[1]) * 0.15, v[2] * 0.7)
        p = wp.vec3(p[0], 0.08, p[2])
    pos[i] = p
    vel[i] = v


class Exhaust:
    """GPU particles for the plume, the ground cloud and the dust, stepped one film frame
    (1/60 s, two substeps) at a time from the recorded engine state. The state is a pure
    function of the frame: `seek(f)` re-simulates from the nearest checkpoint."""

    CAP = 320_000
    RATE = 1000                 # exhaust particles per frame at full throttle
    DUST_RATE = 420            # dust particles per frame at full throttle, low over the pad
    CHECK = 120                # frames between checkpoints

    def __init__(self, engine, device="cuda:0"):
        self.engine = engine       # frame -> (nozzle position, thrust direction, throttle) or None
        self.dev = device
        z3 = np.zeros((self.CAP, 3), np.float32)
        self.pos = wp.array(z3, dtype=wp.vec3, device=device)
        self.vel = wp.array(z3, dtype=wp.vec3, device=device)
        self.age = wp.array(np.ones(self.CAP, np.float32), dtype=float, device=device)
        self.life = wp.array(np.zeros(self.CAP, np.float32), dtype=float, device=device)
        self.kind = wp.array(np.zeros(self.CAP, np.int32), dtype=int, device=device)
        self.head = 0
        self.frame = -1
        self.checks = {}
        self.prev_nozzle = None

    def _snapshot(self):
        return (wp.clone(self.pos), wp.clone(self.vel), wp.clone(self.age), wp.clone(self.life),
                wp.clone(self.kind), self.head, self.prev_nozzle)

    def _restore(self, snap):
        for dst, src in zip((self.pos, self.vel, self.age, self.life, self.kind), snap[:5]):
            wp.copy(dst, src)
        self.head, self.prev_nozzle = snap[5], snap[6]

    def _clear(self):
        self.age.fill_(1.0)
        self.life.fill_(0.0)
        self.head = 0
        self.prev_nozzle = None

    def _advance(self):
        f = self.frame + 1
        st = self.engine(f)
        if st == "reset":
            self._clear()
            st = None
        dtf = 1.0 / FPS
        if st is not None:
            nozzle, d, thr = st
            n0 = nozzle if self.prev_nozzle is None else self.prev_nozzle
            n = int(self.RATE * thr)
            if n > 0:
                speed = 30.0 + 26.0 * thr
                wp.launch(_emit_exhaust, dim=n, inputs=[self.pos, self.vel, self.age, self.life, self.kind,
                                                         self.head, self.CAP, 7919 * f + 1, wp.vec3(*n0),
                                                         wp.vec3(*nozzle), wp.vec3(*d), speed, 0.075, dtf, 0.55],
                          device=self.dev)
                self.head = (self.head + n) % self.CAP
            # dust: where the plume meets the pad, strongest when the engine is low
            h = nozzle[1]
            if d[1] < -0.3:
                hit = nozzle + d * (h / -d[1])
                near = clamp01(1.0 - h / 22.0) ** 1.6
                nd = int(self.DUST_RATE * thr * near)
                if nd > 0:
                    wp.launch(_emit_dust, dim=nd, inputs=[self.pos, self.vel, self.age, self.life, self.kind,
                                                          self.head, self.CAP, 104729 * f + 3,
                                                          wp.vec3(hit[0], 0.0, hit[2]), 1.0 + 0.25 * h,
                                                          7.0 + 11.0 * near], device=self.dev)
                    self.head = (self.head + nd) % self.CAP
            self.prev_nozzle = nozzle
        else:
            self.prev_nozzle = None
        for sub in range(2):
            wp.launch(_step, dim=self.CAP, inputs=[self.pos, self.vel, self.age, self.life, self.kind,
                                                   dtf / 2, 31 * f + sub], device=self.dev)
        self.frame = f
        if f % self.CHECK == 0:
            self.checks[f] = self._snapshot()

    def seek(self, f):
        if f < self.frame or f - self.frame > self.CHECK:
            base = max((c for c in self.checks if c <= f), default=None)
            if base is None:
                self._clear()
                self.frame = -1
            elif base > self.frame or f < self.frame:
                self._restore(self.checks[base])
                self.frame = base
        while self.frame < f:
            self._advance()

    def arrays(self):
        a, l = self.age.numpy(), self.life.numpy()
        alive = a < l
        return self.pos.numpy()[alive], a[alive], l[alive], self.kind.numpy()[alive]


def puff_texture(size=128):
    """A soft round sprite with a Gaussian falloff and no visible edge."""
    y, x = np.mgrid[0:size, 0:size]
    r = np.hypot(x - (size - 1) / 2, y - (size - 1) / 2) / (size / 2)
    a = np.exp(-(r * 2.1) ** 2) * np.clip((1.0 - r) / 0.15, 0.0, 1.0)
    img = np.zeros((size, size, 4), np.uint8)
    img[..., :3] = 255
    img[..., 3] = (a * 255).astype(np.uint8)
    return tp.data_texture(img, True)


class ExhaustView:
    """threepp points for the particles: glowing exhaust (additive), then smoke and dust in
    four age bands, each larger and fainter than the last, so the cloud spreads and thins
    out as it ages instead of fading to the fog colour blob by blob."""

    BANDS = ((0.0, 2.4, 0.15), (0.25, 3.8, 0.11), (0.5, 5.6, 0.075), (0.75, 7.8, 0.045))  # (from age/life, size, opacity)

    def __init__(self, scene, cap=Exhaust.CAP):
        puff = puff_texture()
        self.cap = cap

        def pts(size, blending, opacity, order):
            g = tp.BufferGeometry()
            buf = (np.zeros((cap, 3), np.float32), np.zeros((cap, 3), np.float32))
            g.set_attribute("position", buf[0])
            g.set_attribute("color", buf[1])
            g.set_draw_range(0, 0)
            m = tp.PointsMaterial()
            m.size = size
            m.size_attenuation = True
            m.vertex_colors = True
            m.map = puff
            m.transparent = True
            m.depth_write = False
            m.opacity = opacity
            m.blending = blending
            p = tp.Points(g, m)
            p.frustum_culled = False
            p.render_order = order
            scene.add(p)
            return g, buf
        # the oldest, largest band first, so younger smoke draws over it
        self.smoke = [pts(sz, tp.Blending.Normal, op, 4 - k) for k, (_, sz, op) in enumerate(self.BANDS)]
        self.hot = pts(2.2, tp.Blending.Additive, 0.5, 6)

    def _put(self, gp, P, C):
        g, (pa, ca) = gp
        n = min(len(P), self.cap)
        pa[:n], ca[:n] = P[:n], C[:n]
        g.update_attribute("position", pa)
        g.update_attribute("color", ca)
        g.set_draw_range(0, n)

    def update(self, ex: Exhaust, glow_at, glow):
        P, age, life, kind = ex.arrays()
        hot = (kind != 1) & (age < 0.22)
        T = np.clip(age[hot] / 0.22, 0, 1)[:, None]
        c0, c1, c2 = np.array([1.0, 0.93, 0.78]), np.array([1.0, 0.5, 0.14]), np.array([0.45, 0.08, 0.02])
        ch = np.where(T < 0.5, c0 * (1 - 2 * T) + c1 * 2 * T, c1 * (2 - 2 * T) + c2 * (2 * T - 1))
        ch *= (1.0 - T) ** 1.2 * 1.3
        self._put(self.hot, P[hot], ch)
        # smoke and dust: grey-brown, lit orange near the engine
        cold = ~hot
        Pc, ac, lc, kc = P[cold], age[cold], life[cold], kind[cold]
        u = np.clip(ac / lc, 0, 1)
        base = np.where((kc == 1)[:, None], np.array([0.2, 0.16, 0.12]), np.array([0.14, 0.135, 0.14]))
        dist = np.linalg.norm(Pc - np.asarray(glow_at), axis=1)[:, None]
        lit = glow * np.array([1.0, 0.45, 0.14]) * np.clip(1.0 - dist / 16.0, 0, 1) ** 2 * 1.6
        col = base + lit
        edges = [b[0] for b in self.BANDS] + [1.01]
        for k in range(len(self.BANDS)):
            sel = (u >= edges[k]) & (u < edges[k + 1])
            self._put(self.smoke[k], Pc[sel], col[sel])


class LaunchSite:
    """The pad at blue hour: sky, ground, pad, floodlights, distant tanks, and the rocket
    with its flame, engine light and exhaust."""

    def __init__(self, st: Stage):
        self.st = st
        sc = st.scene
        sc.set_fog(tp.Color(FOG), 90.0, 620.0)
        dome, stars = sky_dome()
        sc.add(dome)
        sc.add(stars)
        st.hemi.color = tp.Color(0x6a7ba8)
        st.hemi.ground_color = tp.Color(0x3a2a22)
        st.hemi.intensity = 1.1
        # the sun has just set behind the tanks: a low warm key, long soft shadows
        st.key.color = tp.Color(0xffa870)
        st.key.intensity = 1.6
        st._key_offset = np.array([-60.0, 29.0, -40.0])
        st.key.set_shadow_frustum(-40, 40, 40, -40)
        st.key.shadow.bias = -0.0008
        st.key.shadow.normal_bias = 0.06
        st.rim.color = tp.Color(0x7b93d6)
        st.rim.intensity = 0.5
        st.r.tone_mapping_exposure = 1.05
        st.camera.near = 0.3
        st.camera.update_projection_matrix()

        gm = standard(0xffffff, roughness=0.95)
        gm.map = ground_texture()
        gm.map.repeat.set(60, 60)
        gm.env_map_intensity = 0.03
        ground = tp.Mesh(tp.PlaneGeometry(1200, 1200), gm)
        ground.rotate_x(-math.pi / 2)
        ground.receive_shadow = True
        sc.add(ground)
        pm = standard(0xffffff, roughness=0.85)
        pm.map = pad_texture()
        pm.env_map_intensity = 0.05
        pad = tp.Mesh(tp.CircleGeometry(14.0, 128), pm)
        pad.rotate_x(-math.pi / 2)
        pad.position.y = 0.08             # clear of the ground's depth, seen from 60 m
        pad.receive_shadow = True
        sc.add(pad)

        # floodlight masts
        mast = standard(0x22252b, roughness=0.6, metalness=0.5)
        lamp = standard(0x111111, roughness=0.3, emissive=0xfff2dc, emissive_intensity=3.0)
        self.spots = []
        for k, a in enumerate((0.35, 2.45, 4.3)):
            x, z = 72 * math.cos(a), 72 * math.sin(a)
            pole = tp.Mesh(tp.CylinderGeometry(0.3, 0.45, 28, 12), mast)
            pole.position.set(x, 14, z)
            pole.cast_shadow = True
            sc.add(pole)
            head = tp.Mesh(tp.BoxGeometry(3.0, 1.4, 0.5), lamp)
            head.position.set(x * 0.99, 28.4, z * 0.99)
            head.look_at(0, 4, 0)
            sc.add(head)
            sp = tp.SpotLight(tp.Color(0xfff0dc), 2.4, 0.0, 0.24, 0.6, 0.0)
            sp.position.set(x, 28.8, z)
            aim = tp.Object3D()
            aim.position.set(0, 3, 0)
            sc.add(aim)
            sp.set_target(aim)
            sc.add(sp)
            if k == 0:
                sp.cast_shadow = True
                sp.shadow.map_size = tp.Vector2(2048, 2048)
                sp.shadow.bias = -0.0004
            self.spots.append(sp)
        # distant tanks and a tower, dark against the afterglow
        sil = standard(0x0d0e12, roughness=0.8)
        for x, z, r, h in ((-150, -120, 7, 16), (-132, -128, 5, 12), (-175, -96, 9, 20), (120, -170, 6, 11)):
            c = tp.Mesh(tp.CylinderGeometry(r, r, h, 32), sil)
            c.position.set(x, h / 2, z)
            sc.add(c)
            d = tp.Mesh(tp.SphereGeometry(r, 32, 8, 0, 2 * math.pi, 0, math.pi / 2), sil)
            d.position.set(x, h, z)
            sc.add(d)
        tower = tp.Mesh(tp.BoxGeometry(5, 46, 5), sil)
        tower.position.set(-95, 23, -150)
        sc.add(tower)
        warn = standard(0x111111, emissive=0xff2a1a, emissive_intensity=4.0)
        beacon = tp.Mesh(tp.SphereGeometry(0.6, 12, 8), warn)
        beacon.position.set(-95, 46.8, -150)
        sc.add(beacon)
        self.beacon = warn

        # the rocket, its flame and its light
        self.rocket, self.bell_glow = build_rocket()
        sc.add(self.rocket)
        self.flames = []
        for color, r, op in ((0xff7a2a, 0.5, 0.5), (0xffe6c0, 0.3, 0.85)):
            m = tp.MeshBasicMaterial()
            m.color = tp.Color(color)
            m.transparent = True
            m.blending = tp.Blending.Additive
            m.depth_write = False
            m.opacity = op
            m.fog = False
            cone = tp.Mesh(tp.ConeGeometry(r, 1.0, 32, 1, True), m)
            cone.visible = False
            cone.render_order = 5             # after the smoke: the flame glows through it
            sc.add(cone)
            self.flames.append((cone, m, op))
        self.light = tp.PointLight(tp.Color(0xff9a4a), 0.0, 90.0, 1.2)
        sc.add(self.light)
        self.view = ExhaustView(sc)
        self.ghosts = {}

    def add_ghost(self, key, color):
        g, m = ghost(self.rocket, color, opacity=0.0)
        g.visible = False
        self.st.scene.add(g)
        self.ghosts[key] = (g, m)

    def pose(self, obj, pos, quat):
        obj.position.set(*pos)
        obj.quaternion.set(*quat)

    def engine(self, pos, quat, gim, thr, t):
        """Place the flame, the bell glow and the engine light for thrust `thr` (0..1).
        Returns the exhaust direction (unit, world) and the nozzle position."""
        R = q2R(quat)
        up = R[:, 1]
        d = up + np.array([gim[0], 0.0, gim[1]])
        d = -d / np.linalg.norm(d)
        nozzle = np.asarray(pos) + R @ np.array([0.0, NOZZLE_Y, 0.0])
        flick = 1.0 + 0.08 * math.sin(t * 61.0) + 0.05 * math.sin(t * 97.0 + 1.3) + 0.04 * math.sin(t * 143.0)
        on = thr > 0.02
        for k, (cone, m, op) in enumerate(self.flames):
            cone.visible = on
            if on:
                L = (2.2 + 5.5 * thr) * (1.0 if k == 0 else 0.62) * flick
                cone.scale.set(0.8 + 0.4 * thr, L, 0.8 + 0.4 * thr)
                q = lesson._quat_y_to(d)              # the apex (+y) downstream, the base in the bell
                cone.quaternion.set(*q)
                cone.position.set(*(nozzle + d * (L / 2 - 0.1)))
                m.opacity = op * clamp01(thr * 2.5)
        self.bell_glow.opacity = clamp01(thr * 3.0)
        self.light.intensity = 9.0 * thr * flick
        self.light.position.set(*(nozzle + d * 2.0))
        return d, nozzle


# ── the script ────────────────────────────────────────────────────────────────
# The narration sets the pace, as in Part 0: every caption lasts as long as its spoken
# line (Kokoro, af_heart, measured in SPEECH) plus the voice's lead-in and tail and a
# margin, and each flight event is placed from those captions and from what the physics
# needs (the P flight's crash, the climb's overshoot, the landing). A line whose text
# changes should be re-measured (lesson.Narration(...).clip(text)); until then `run`
# holds the picture for any overrun.
CAPTION_TEXT = {
    "t0": "Hold still: PID control, flown on a rocket.",
    "t1": "This is a hopper: a test rocket with one engine. Its job is to lift off, and hold still at a set height.",
    "t2": "The engine can only push along the rocket. A computer sets the throttle many times a second, "
          "and the engine always answers a moment late.",
    "p1": "Idea one: the further below the target, the harder it pushes. That is P, for proportional.",
    "p2": "It shoots past, falls back, and every swing is bigger than the last. The engine answers late, "
          "so every push arrives late.",
    "p3": "Until it slams into the pad.",
    "d1": "Idea two: also push against the speed. Climbing fast? Ease off. Falling? Push harder. "
          "That is D, for derivative.",
    "d2": "No more swinging. But it settles below the target.",
    "d3": "To hover, the engine must carry the weight, and P only pushes that hard when the rocket is below "
          "the target. So the gap never closes.",
    "i1": "Idea three: add up the error over time, and push with that sum too. That is I, for integral.",
    "i2": "While any gap is left, the sum keeps growing, until the rocket sits right on the target, and stays there.",
    "w1": "One catch. On the climb the gap is large for seconds, and the sum piles up. "
          "So this rocket shoots far past the target.",
    "w2": "The fix: stop adding while the engine is already at full throttle. Now it barely overshoots.",
    "g1": "Two more loops keep it upright by swinging the engine. A gust shoves the rocket sideways; "
          "it leans into it, and flies back over the pad.",
    "l1": "To land, move the target down a little every step, and let the same three terms follow it.",
    "l2": "The engine cuts at touchdown, and the rocket settles softly on its legs.",
}
VOICE_ONLY = {"t0"}          # spoken over the title card, not drawn as a caption
lesson.SPOKEN_WORDS.update({"PID": "P I D"})
SPEECH = {"t0": 3.65, "t1": 6.75, "t2": 8.88, "p1": 6.1, "p2": 8.1, "p3": 2.3, "d1": 7.88, "d2": 3.48,
          "d3": 8.55, "i1": 6.22, "i2": 6.42, "w1": 7.95, "w2": 5.92, "g1": 8.82, "l1": 5.7, "l2": 4.83}   # s, af_heart
SLACK = lesson.VOICE_LEAD + lesson.VOICE_TAIL + 0.3


def speech(k):
    """The measured length of line k, or an estimate from its words until it is measured."""
    return SPEECH.get(k, 0.42 * len(CAPTION_TEXT[k].split()) + 0.4)


IGNITE = 1.2                 # seconds from a flight's start to ignition
GUST = (0.12, 1.5)           # sideways force / weight, seconds
LAND_FAST, LAND_SLOW, LAND_KNEE = 2.0, 0.6, 4.0   # m/s above and below the knee height (m)


def landing_target(t0):
    """The set point for a landing that starts at flight time t0: down at LAND_FAST to the
    knee, then LAND_SLOW, and on below the ground so the rocket keeps settling."""
    t_knee = (TARGET - LAND_KNEE) / LAND_FAST

    def target(t):
        if t < t0:
            return TARGET
        u = t - t0
        return TARGET - LAND_FAST * u if u < t_knee else LAND_KNEE - LAND_SLOW * (u - t_knee)
    return target


class Plan:
    """The flights and the timeline. Everything the film shows comes from here."""

    def __init__(self, verbose=True):
        t0 = time.time()
        # flights whose timing does not depend on the layout, flown first
        self.fp = fly(40.0, ignite=IGNITE, kp=KP, cut_on_impact=True)
        if self.fp.impact is None:
            raise RuntimeError("the P-only flight never came down: retune KP or TAU")
        self.fw = fly(26.0, ignite=IGNITE, kp=KP, ki=KI, kd=KD, anti_windup=False)
        naive_peak_t = float(self.fw.t[np.argmax(self.fw.pos[:, 1])])
        probe = fly(60.0, ignite=IGNITE, kp=KP, ki=KI, kd=KD, land_from=30.0, target=landing_target(30.0))
        td_rel = probe.touchdown[0] - 30.0
        t_imp = self.fp.impact[0]

        cap, tl = {}, Timeline()

        def say(k, a):
            cap[k] = (a, a + speech(k) + SLACK)
            return cap[k][1]
        c = say("t0", 1.2)                       # spoken over the title card
        tl.add("open", 0.0, max(6.6, c + 0.8))
        c = say("t1", tl.end("open") + 0.4)
        c = say("t2", c + 0.2)
        tl.add("task", tl.end("open"), c + 0.3)
        # P: a flight that ends on the pad
        s = tl.end("task")
        c = say("p1", s + 0.3)
        c = say("p2", c + 0.2)
        # the flight starts late enough that "until it hits the pad" is said as it hits
        self.F_P = max(s, c + 0.4 - t_imp)
        self.impact_t = self.F_P + t_imp
        c = say("p3", self.impact_t - 0.2)
        tl.add("p", s, max(c + 0.6, self.impact_t + 3.2))
        # D, then I on the same flight
        s = tl.end("p")
        self.F_D = s
        c = say("d1", s + 0.5)
        c = say("d2", max(c + 0.2, self.F_D + IGNITE + 9.0))
        c = say("d3", c + 0.2)
        tl.add("d", s, c + 0.2)
        s = tl.end("d")
        self.I_on = s + 0.75 * speech("i1")
        c = say("i1", s)
        c = say("i2", max(c + 0.2, self.I_on + 11.0))
        tl.add("i", s, c + 0.5)
        # windup, gust and landing: one flight, with the naive climb as a ghost
        s = tl.end("i")
        self.F_C = s
        c = say("w1", s + 0.4)
        c = say("w2", max(c + 0.2, self.F_C + naive_peak_t + 1.2))
        tl.add("w", s, c + 0.4)
        s = tl.end("w")
        c = say("g1", s + 0.2)
        self.gust_t = cap["g1"][0] + 1.4
        tl.add("g", s, max(c + 0.3, self.gust_t + 8.5))
        s = tl.end("g")
        c = say("l1", s + 0.2)
        self.land_t = cap["l1"][0] + 0.8
        self.td_t = self.land_t + td_rel
        c = say("l2", max(c + 0.2, self.td_t - 0.3))
        tl.add("l", s, c + 1.4)
        tl.add("outro", tl.end("l"), tl.end("l") + 9.0)
        self.tl, self.cap = tl, cap
        self.duration = tl.duration

        # the flights placed by the layout
        self.fd = fly(tl.end("i") - self.F_D + 0.1, ignite=IGNITE, kp=KP, ki=KI, kd=KD,
                      ki_from=self.I_on - self.F_D)
        g0 = self.gust_t - self.F_C
        l0 = self.land_t - self.F_C
        self.fc = fly(self.duration - self.F_C + 0.1, ignite=IGNITE, kp=KP, ki=KI, kd=KD,
                      gust=(g0, g0 + GUST[1], GUST[0]), land_from=l0, target=landing_target(l0))

        # what the captions say, measured
        fd, fc, fw = self.fd, self.fc, self.fw
        m = {}
        m["impact"] = abs(self.fp.impact[1])
        a = int((self.I_on - self.F_D - 2.0) * FPS)
        m["h_pd"] = float(fd.pos[a:a + 2 * FPS, 1].mean())
        m["hover"] = 100.0 * fd.hover
        m["sag"] = TARGET - m["h_pd"]
        b = int((tl.end("i") - self.F_D) * FPS) - 1
        m["hold_mm"] = 1000.0 * float(np.abs(fd.pos[b - 2 * FPS:b, 1] - TARGET).max())
        m["naive_peak"] = float(fw.pos[:, 1].max())
        pre = int(g0 * FPS)
        m["aw_over"] = float(fc.pos[:pre, 1].max()) - TARGET
        gw = slice(pre, int((g0 + 8.0) * FPS))
        up = np.array([q2R(q)[:, 1] for q in fc.quat[gw]])
        m["tilt"] = float(np.degrees(np.arccos(np.clip(up[:, 1], -1, 1))).max())
        m["drift"] = float(np.hypot(fc.pos[gw, 0], fc.pos[gw, 2]).max())
        m["td"] = abs(fc.touchdown[1]) if fc.touchdown else float("nan")
        self.m = m
        self.captions = [(a, b, CAPTION_TEXT[k].format(**m)) + ((False,) if k in VOICE_ONLY else ())
                         for k, (a, b) in cap.items()]
        if verbose:
            print(f"[flights] 5 PhysX flights in {time.time() - t0:.1f}s; mass {fc.mass:.0f} kg, "
                  f"hover throttle {m['hover']:.1f} %")
            print(f"[P]  swings grow; hits the pad {t_imp:.1f} s after ignition at {m['impact']:.1f} m/s")
            print(f"[PD] settles at {m['h_pd']:.2f} m: {m['sag']:.2f} m low (hover / Kp = {fd.hover / KP:.2f} m)")
            print(f"[PID] holds {TARGET:.0f} m within {m['hold_mm']:.1f} mm")
            print(f"[windup] naive peak {m['naive_peak']:.2f} m; with anti-windup {m['aw_over']:.2f} m over")
            print(f"[gust] {GUST[0]:.2f} x weight for {GUST[1]} s: tilt {m['tilt']:.1f} deg, "
                  f"drift {m['drift']:.2f} m")
            print(f"[land] touchdown at {m['td']:.2f} m/s")
            print(f"[film] {self.duration:.1f} s")

    def state(self, t):
        """The flight on screen at film time t: (flight, frame index, flight start, key)."""
        if t < self.F_D:
            fl, t0, key = self.fp, self.F_P, "p"
        elif t < self.F_C:
            fl, t0, key = self.fd, self.F_D, "d"
        else:
            fl, t0, key = self.fc, self.F_C, "c"
        i = min(max(int(round((t - t0) * FPS)), 0), fl.n - 1)
        return fl, i, t0, key


_PLAN = None


def plan():
    global _PLAN
    if _PLAN is None:
        _PLAN = Plan()
    return _PLAN


# ── the picture ───────────────────────────────────────────────────────────────
BADGES = {"p": ("P", "PROPORTIONAL", C_P), "d": ("D", "DERIVATIVE", C_D), "i": ("I", "INTEGRAL", C_I),
          "w": ("I", "ANTI-WINDUP", C_I), "g": ("PID", "GUST", C_TGT), "l": ("PID", "LANDING", C_TGT)}
SUMMARY = [(r"$K_p\,e$", "P pushes in proportion to the error."),
           (r"$-K_d\,v$", "D pushes against the speed: no swinging."),
           (r"$K_i\int e\,dt$", "I pushes with the error's running sum: no gap left.")]


def shake(t, amp):
    return amp * np.array([math.sin(t * 37.0) + 0.6 * math.sin(t * 71.0 + 1.1),
                           0.7 * math.sin(t * 43.0 + 2.0) + 0.4 * math.sin(t * 89.0),
                           math.sin(t * 29.0 + 0.4) + 0.5 * math.sin(t * 101.0 + 2.2)])


class Painter:
    def __init__(self, st: Stage, site: LaunchSite, pl: Plan, ov: Hud):
        self.st, self.site, self.pl, self.ov = st, site, pl, ov
        tl, cap = pl.tl, pl.cap
        site.add_ghost("naive", C_BAD)
        # the target: a hoop at the set point, around the rocket's axis
        self.hoop_holder = tp.Group()
        st.scene.add(self.hoop_holder)
        hold = tp.Group()
        hold.rotate_x(math.pi / 2)
        self.hoop_holder.add(hold)
        self.hoop = lesson.Ring3D(C_TGT, radius=3.4, tube=0.05, parent=hold, emissive=2.2)
        self.err = Arrow3D(C_P, radius=0.09, emissive=1.6, parent=st.scene)
        self.err.set_opacity(0.0)
        self.gust_arrow = Arrow3D(0x9fd3ff, radius=0.16, emissive=1.4, parent=st.scene)
        self.gust_arrow.set_opacity(0.0)
        # the look point follows each flight's rocket, smoothed
        self.follow = {}
        for key, fl in (("p", pl.fp), ("d", pl.fd), ("c", pl.fc)):
            p = fl.pos.copy()
            sm = np.empty_like(p)
            sm[0] = p[0]
            k = 1.0 - math.exp(-1.0 / (FPS * 0.6))
            for i in range(1, len(p)):
                sm[i] = sm[i - 1] + k * (p[i] - sm[i - 1])
            self.follow[key] = sm
        T = tl.start
        imp = pl.impact_t
        cam = [   # (time, azimuth, elevation, distance, look offset, fov)
            (0.0, 205, 4, 56, (0, 6, 0), 30), (6.6, 222, 6, 40, (0, 6, 0), 32),
            (tl.end("task") - 0.5, 238, 8, 36, (0, 5, 0), 34),
            (pl.F_P + IGNITE, 250, 5, 40, (0, 5, 0), 36), (imp - 3.0, 262, 7, 50, (0, 5, 0), 36),
            (imp + 0.2, 268, 4, 34, (0, 5, 0), 34), (tl.end("p"), 272, 5, 32, (0, 5, 0), 34),
            (T("d"), 150, 4, 38, (0, 5, 0), 34), (T("d") + 6.0, 140, 9, 46, (0, 5, 0), 34),
            (tl.end("d"), 128, 10, 46, (0, 5, 0), 34), (tl.end("i"), 110, 12, 50, (0, 5, 0), 34),
            (T("w"), 200, 4, 44, (0, 5, 0), 36), (T("w") + 5.0, 212, 8, 58, (0, 5, 0), 36),
            (tl.end("w"), 222, 9, 56, (0, 5, 0), 36),
            (T("g"), 262, 8, 46, (0, 5, 0), 34), (tl.end("g"), 272, 9, 46, (0, 5, 0), 34),
            (T("l"), 300, 6, 42, (0, 5, 0), 34), (pl.td_t - 3.0, 318, 4, 30, (0, 5, 0), 34),
            (tl.end("l"), 326, 5, 28, (0, 5, 0), 34),
            (tl.end("outro"), 340, 14, 60, (0, 5, 0), 32),
        ]
        self.cam = OrbitCamera(cam, drift=((1.2, 0.17), (0.6, 0.13)), frame="yup", follow=self._look)
        self.equations = [
            (cap["p1"][0] + 1.5, T("d") - 0.2, r"$u = K_p\,e$", "throttle, P"),
            (T("d") + 0.3, T("i") + 0.8, r"$u = K_p\,e - K_d\,v$", "throttle, PD"),
            (T("i") + 0.5, tl.end("l"), r"$u = K_p\,e + K_i\int e\,dt - K_d\,v$", "throttle, PID"),
            (cap["p1"][0] + 1.5, tl.end("l"), r"$e = h_{\mathrm{target}} - h$", "the error"),
            (cap["w2"][0] + 0.3, tl.end("w"), r"$u \geq 1:\ \mathrm{stop\ adding\ to}\ \int e\,dt$", "anti-windup"),
        ]

    def _look(self, t):
        fl, i, t0, key = self.pl.state(t)
        p = self.follow[key][i]
        return np.array([p[0], p[1], p[2]])

    # ---- 3D --------------------------------------------------------------------
    def set3d(self, t):
        pl, site, st = self.pl, self.site, self.st
        tl = pl.tl
        fl, i, t0, key = pl.state(t)
        pos, quat, thr = fl.pos[i], fl.quat[i], fl.thr[i]
        site.pose(site.rocket, pos, quat)
        d, nozzle = site.engine(pos, quat, fl.gim[i], thr, t)
        site.view.update(self.ex, nozzle + d * 2.0, thr)
        # centre the sun's shadow map where the rocket's shadow lands, not under the rocket
        off = st._key_offset
        mid = np.asarray(pos) + np.array([0.0, 6.0, 0.0])
        st.follow(mid - off * (mid[1] / off[1]), height=60.0)
        site.beacon.emissive_intensity = 4.0 if (t % 1.6) < 0.25 else 0.4
        # the camera, shaken by the engine when it is close
        eye, look, fov = self.cam(t)
        near = clamp01(1.0 - pos[1] / 28.0)
        amp = 0.05 * thr * near * clamp01(1.0 - (np.linalg.norm(eye - pos) - 25.0) / 60.0)
        amp += 0.5 * math.exp(-max(t - pl.impact_t, 0.0) * 5.0) * (t >= pl.impact_t) * (t < tl.end("p"))
        s = shake(t, amp)
        st.look(eye + s, look + 0.4 * s, fov)
        # the ghost: the same controller without anti-windup
        g, gm = site.ghosts["naive"]
        ga = envelope(t, tl.start("w") + 0.2, tl.end("w") - 0.3, 0.5, 0.8) if key == "c" else 0.0
        g.visible = ga > 0.003
        if g.visible:
            j = min(i, pl.fw.n - 1)
            site.pose(g, pl.fw.pos[j], pl.fw.quat[j])
            gm.opacity = 0.42 * ga
        # target hoop and the error arrow
        tgt = fl.tgt[i] if fl.lit[i] or key == "c" else TARGET
        ha = envelope(t, pl.cap["t1"][0] + 2.0, pl.td_t + 0.6, 0.8, 0.8)
        ha *= 0.0 if (tl.end("p") - 0.4 < t < tl.start("d") + 0.8) else 1.0
        self.hoop.set_opacity(ha * (0.55 + 0.15 * math.sin(t * 4.0)))
        self.hoop_holder.position.set(pos[0], tgt, pos[2])
        ea = envelope(t, pl.cap["p1"][0] + 1.0, tl.end("i"), 0.5, 0.5) * (1.0 if fl.lit[i] else 0.0)
        e = tgt - pos[1]
        if ea > 0.003 and abs(e) > 0.25:
            base = np.array([pos[0] + 2.2, pos[1] + 6.0, pos[2]])
            self.err.set(base, base + np.array([0.0, e, 0.0]), opacity=ea)
        else:
            self.err.set_opacity(0.0)
        if key == "c" and fl.gust[i] > 0:
            gp = np.array([pos[0] - 9.0, pos[1] + 7.0, pos[2]])
            self.gust_arrow.set(gp, gp + np.array([5.0, 0.0, 0.0]), opacity=1.0)
        else:
            self.gust_arrow.set_opacity(0.0)
        return fl, i, key

    # ---- 2D --------------------------------------------------------------------
    def draw2d(self, t, fl, i, key):
        ov, pl, st = self.ov, self.pl, self.st
        tl = pl.tl
        ov.title_card(t, "A THREEPP LESSON  ·  PART 4", "Hold Still", "PID control, flown on a rocket", C_P,
                      t_out=tl.end("open") - 0.3)
        ov.captions(t, pl.captions)
        ov.equation_card(t, self.equations)
        # the beat's badge, top centre
        for name, (letter, word, col) in BADGES.items():
            a = envelope(t, tl.start(name) + 0.15, tl.end(name) - 0.1, 0.4, 0.4)
            if a > 0.003:
                w = ov.text_width(word, 20, "semibold") + 20 * 2.4 + 90
                x0 = 960 - w / 2
                ov.panel(x0, 58, w, 62, radius=31, alpha=0.72 * a, outline=col, outline_alpha=0.6)
                ov.text(x0 + 30, 89, letter, size=34, color=col, alpha=a, kind="semibold", anchor="lm")
                ov.text(x0 + 30 + ov.text_width(letter, 34, "semibold") + 20, 90, word, size=20, color=C_TEXT,
                        alpha=a, kind="semibold", anchor="lm", tracking=2.4)
        # altitude, the last 20 s of this flight
        fa = envelope(t, pl.F_P + IGNITE - 0.5, tl.end("l") - 0.2, 0.6, 0.6)
        if tl.end("p") - 0.35 < t < tl.start("d") + 0.35 or tl.end("i") - 0.35 < t < tl.start("w") + 0.35:
            fa = 0.0
        t0 = pl.state(t)[2]
        tf = i / FPS
        if fa > 0:
            x1 = max(tf, 20.0)
            x0 = x1 - 20.0
            j0 = max(int(x0 * FPS), 0)
            xs = fl.t[j0:i + 1:2]
            series = [(xs, fl.tgt[j0:i + 1:2], C_TGT, "target")]
            if key == "c" and t < tl.end("w"):
                jn = min(i, pl.fw.n - 1)
                series.append((pl.fw.t[j0:jn + 1:2], pl.fw.pos[j0:jn + 1:2, 1], C_BAD, "no anti-windup"))
            series.append((xs, fl.pos[j0:i + 1:2, 1], 0xffffff, "altitude"))
            ov.plot(1380, 64, 476, 300, series, (x0, x1), (0.0, 32.0), alpha=fa, title="ALTITUDE  (m)",
                    yticks=(0, 10, 20, 30), xticks=[v for v in range(int(x0) + 1, int(x1) + 1) if v % 5 == 0],
                    xfmt=lambda v: f"{v:.0f} s", yfmt=lambda v: f"{v:.0f}")
            self.terms(t, fl, i, fa, key)
        # the P crash
        ia = envelope(t, pl.impact_t, tl.end("p") - 0.3, 0.05, 0.4)
        if ia > 0:
            ov.fade(0.5 * math.exp(-(t - pl.impact_t) * 6.0), color=0xffffff)
            p = st.project(fl.pos[i] + np.array([0.0, 7.0, 0.0]))
            ov.text(p[0] + 150, p[1] - 10, "HARD LANDING", size=46, color=C_BAD, alpha=ia, kind="semibold",
                    anchor="ls", tracking=3.0)
        # the landing
        la = envelope(t, pl.td_t, tl.end("l") + 0.5, 0.2, 0.6)
        if la > 0:
            p = st.project(fl.pos[i] + np.array([0.0, 7.0, 0.0]))
            ov.text(p[0] + 190, p[1] - 10, "TOUCHDOWN", size=40, color=C_TGT, alpha=la, kind="semibold",
                    anchor="ls", tracking=3.0)
        # the target's label on the hoop
        ha = envelope(t, pl.cap["t1"][0] + 2.4, tl.end("task") + 1.0, 0.6, 0.6)
        if ha > 0:
            p = st.project(np.array([fl.pos[i][0] + 3.4, TARGET, fl.pos[i][2]]))
            ov.callout(p[:2], (p[0] + 150, p[1] - 90), "target", C_TGT, ha, size=26)
        ov.summary(t, tl.start("outro"), tl.end("outro"), SUMMARY, "Flown, simulated and rendered with threepp",
                   C_P, C_P, text_dx=250)

    def terms(self, t, fl, i, alpha, key):
        """The controller's output, term by term: signed bars, and the throttle it asks for."""
        ov, pl = self.ov, self.pl
        tl = pl.tl
        x, y, w = 70, 560, 440
        shown = [("P", C_P, 0, 0.0)]
        if t >= tl.start("d"):
            shown.append(("D", C_D, 2, 0.0))
        if t >= pl.I_on - 0.2:
            shown.append(("I", C_I, 1, 0.0))
        h = 108 + 44 * len(shown)
        ov.panel(x, y, w, h, radius=16, alpha=0.66 * alpha, outline=0x8aa0c0, outline_alpha=0.16)
        ov.text(x + 24, y + 22, "THROTTLE, TERM BY TERM", size=16, color=C_DIM, alpha=alpha, kind="semibold",
                tracking=2.2)
        cx, half = x + 280, 130
        for j, (name, col, k, _) in enumerate(shown):
            yy = y + 72 + j * 44
            v = float(fl.terms[i, k])
            ov.text(x + 24, yy, name, size=24, color=col, alpha=alpha, kind="semibold", anchor="lm")
            ov.line([(cx - half, yy), (cx + half, yy)], 0x2a3342, 12.0, alpha)
            ov.line([(cx, yy), (cx + half * max(-1.0, min(1.0, v)), yy)], col, 12.0, alpha)
        ov.line([(cx, y + 52), (cx, y + 52 + 44 * len(shown))], 0x55627a, 1.4, alpha)
        yy = y + 72 + len(shown) * 44 + 8
        ov.text(x + 24, yy, "engine", size=19, color=C_DIM, alpha=alpha, anchor="lm")
        ov.bar(cx, yy - 7, half, 14, fl.thr[i], 0xffffff, alpha=alpha)
        hx = cx + half * pl.fd.hover
        ov.line([(hx, yy - 14), (hx, yy + 14)], C_TGT, 2.0, alpha)
        ov.text(hx, yy + 26, "hover", size=14, color=C_TGT, alpha=alpha, anchor="ma")


# ── main ──────────────────────────────────────────────────────────────────────
def setup(width, height):
    pl = plan()
    st = Stage(width, height, renderer="gl", floor=False, env="autumn_field_puresky_2k.hdr", env_intensity=0.15,
               fog=(90, 620), shadow_extent=24.0, far=1500.0)
    site = LaunchSite(st)
    ov = Hud(1920, 1080, math_cache=os.path.join(_HERE, "rocket_pid.math.json"))
    painter = Painter(st, site, pl, ov)
    starts = {int(round(pl.F_D * FPS)), int(round(pl.F_C * FPS))}

    def engine(f):
        if f in starts:
            return "reset"
        t = f / FPS
        fl, i, _, _ = pl.state(t)
        if fl.thr[i] < 0.02:
            return None
        d, nozzle = site.engine(fl.pos[i], fl.quat[i], fl.gim[i], fl.thr[i], t)
        return nozzle, d, fl.thr[i]
    painter.ex = Exhaust(engine)

    def render(t):
        painter.ex.seek(int(round(t * FPS)))
        fl, i, key = painter.set3d(t)
        ov.begin()
        painter.draw2d(t, fl, i, key)
        cut = min(abs(t - pl.F_D), abs(t - pl.F_C))
        ov.fade(max(fade_in_out(t, pl.duration), clamp01(1.0 - cut / 0.35)))
        ov.end()
        return st.frame(t, hud=ov)
    return render, pl.captions


if __name__ == "__main__":
    run("rocket_pid", plan().duration, setup, fps=FPS)
