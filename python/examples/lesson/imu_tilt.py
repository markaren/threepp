"""Which way is up? How a robot measures its own tilt.  Part 3 of the threepp lessons.

    python imu_tilt.py                               # the film (1080p60), lesson_out/imu_tilt.mp4
    python imu_tilt.py --stills 10,50,85             # individual frames, into lesson_out/
    python imu_tilt.py --sheet                       # contact sheet
    python imu_tilt.py --preview --out preview.mp4   # 960x540 @ 30 fps
    python imu_tilt.py --srt                         # the captions as lesson_out/imu_tilt.srt

A Range Rover (tp.PhysxVehicle) drives a hilly stadium track in PhysX, with a tp.Imu on
its centre console: ICM-42688-P white noise from the datasheet, plus a gyro turn-on bias
set in its noise model. Everything the film shows about the IMU is those samples: the gyro
integrated on its own, the accelerometer on its own, a complementary filter, and a Kalman
filter that also uses the wheel speeds to subtract the car's own acceleration. The true
tilt is the chassis pose, so every error on screen is measured, and the bias the Kalman
filter finds is scored against the one that was set.
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
from lesson import (DIM, TEXT, Arrow3D, Hud, OrbitCamera, Ring3D, Stage, Timeline, clamp01,  # noqa: E402
                    data_file, ease_out, envelope, fade_in_out, ghost, remap, run, smooth, standard, tp, xray)

FPS = 60
DT = 1 / 240                 # physics substep = one IMU sample
SUB = 4                      # substeps per film frame
BLOCK = 6                    # frames per point in the tilt-error plot (0.1 s means)
BIAS_DPS = np.array([0.50, -0.40, 0.45])    # gyro turn-on bias (sensor x right, y up, z forward), deg/s
R_WHEEL = 0.4
V_STRAIGHT, V_TURN = 13.0, 6.0
THROTTLE_MAX = 0.45           # keeps acceleration out of the turns SUV-like
MAX_STEER = 0.6               # PhysxVehicle default, rad at full lock
KF_SIG_Z = 0.3               # Kalman: how far (rad) the corrected accelerometer may be off

C_GYRO = 0x4f8dff
C_ACC = 0xffb347
C_FUSE = 0x5ee27a
C_KF = 0xd08bff
C_TEXT = TEXT
C_DIM = DIM
C_IMU = 0x4cc9f0
C_OWN = 0xff7a90                 # the car's own acceleration
AXIS_C = (0xff5a5f, 0x5ee27a, 0x4f8dff)      # x right, y up, z forward

# ── the script ────────────────────────────────────────────────────────────────
# Written on a 108 s clock (the times below). The narration then sets the pace: where a
# spoken line (Kokoro, af_heart, measured in SPEECH) runs past its caption, the script gets
# that much more time just before the caption ends, so the car drives on while the line is
# said instead of the film standing still. `at` maps a written time onto that clock. A line
# whose text changes should be re-measured (lesson.Narration(...).clip(text)); until then
# `run` holds the picture for any overrun.
_CAPTIONS = [
    (7.4, 13.4, "Phones, drones, cars and robots carry an IMU: a gyroscope and an accelerometer on one chip."),
    (13.8, 19.2, None),      # parked accelerometer reading
    (19.6, 23.6, "So it knows which way is up. The gyroscope reports how fast the car turns, axis by axis."),
    (24.2, 30.2, "Idea one: start from the right answer and add up every turn the gyroscope reports."),
    (30.6, 41.6, None),      # gyro bias and drift
    (42.0, 44.0, "Idea two: read up straight from the accelerometer. It cannot drift."),
    (44.4, 57.6, None),      # accelerometer fooled
    (58.0, 64.4, "Combine them: follow the gyroscope, and lean a little toward the accelerometer at every step."),
    (64.8, 75.6, None),      # best blend
    (76.4, 83.0, "The car also knows its speed from its wheels, so it can predict its own acceleration and remove it."),
    (83.4, 93.6, None),      # Kalman
    (94.4, 99.6, "None of this finds north: gravity says nothing about heading. That takes a compass, GNSS or a camera."),
]
SPEECH = [7.17, 9.88, 6.4, 5.55, 10.85, 4.65, 9.95, 5.88, 11.72, 6.65, 10.12, 7.33]    # seconds, af_heart
MARGIN = 0.3                 # a stretched caption outlasts its speech by this much

# Each ghost is introduced on its own: the film holds (the drive stands still), the car
# is hidden, the ghost blinks with a label, and the drive goes on. Everything in this file
# runs on the drive's clock; lesson.run maps it to the film's.
_INTRO = {"gyro": 24.8, "acc": 42.6, "comp": 58.6, "kf": 83.0}
PAUSE = 2.0
INTRO_LABEL = {"gyro": "the gyroscope's estimate", "acc": "the accelerometer's estimate",
               "comp": "the complementary filter", "kf": "the Kalman filter"}


def _stretch():
    """(written time, seconds added there) for every caption whose line would overrun it.
    Like lesson.run, an introduction's pause inside a caption counts as time to speak."""
    out = []
    for (a, b, _), need in zip(_CAPTIONS, SPEECH):
        p = b - lesson.VOICE_TAIL
        have = p - (a + lesson.VOICE_LEAD) + PAUSE * sum(a + lesson.VOICE_LEAD <= s < p for s in _INTRO.values())
        if need + MARGIN > have:
            out.append((p, need + MARGIN - have))
    return out


STRETCH = _stretch()


def at(t):
    """A written script time on the drive's clock."""
    return t + sum(d for p, d in STRETCH if p < t)


TL = Timeline()
for _name, _a, _b in (("open", 0.0, 7.0), ("sense", 7.0, 24.0), ("gyro", 24.0, 42.0), ("accel", 42.0, 58.0),
                      ("fuse", 58.0, 76.0), ("kalman", 76.0, 94.0), ("heading", 94.0, 100.0), ("outro", 100.0, 108.0)):
    TL.add(_name, at(_a), at(_b))
T_GO = at(20.0)              # the car pulls away
INTRO = {key: at(s) for key, s in _INTRO.items()}
T_GYRO0 = INTRO["gyro"]      # the gyro-only estimate starts from the true tilt here
CAPTIONS = [(at(a), at(b), txt) for a, b, txt in _CAPTIONS]

EQUATIONS = [(at(a), at(b), tex, note) for a, b, tex, note in [   # (t_in, t_out, tex, note)
    (14.2, 23.8, r"$\mathbf{f} = \mathbf{a} - \mathbf{g}$", "what the accelerometer measures"),
    (24.0, 41.8, r"$\mathbf{u} \leftarrow \mathbf{u} - (\boldsymbol{\omega}\,\Delta t) \times \mathbf{u}$",
     "integrate the gyroscope"),
    (41.9, 57.8, r"$\mathbf{u} = \mathbf{f}\,/\,|\mathbf{f}|$", "trust the accelerometer"),
    (57.9, 75.8, r"$\mathbf{u} \leftarrow \alpha\,(\mathbf{u} - \boldsymbol{\omega}\Delta t \times \mathbf{u})"
                 r" + (1 - \alpha)\,\hat{\mathbf{f}}$", "complementary filter"),
    (76.6, 93.8, r"$\mathbf{f} - (\dot{v}\,\hat{\mathbf{e}}_{\mathrm{fwd}} + \boldsymbol{\omega} \times \mathbf{v})$",
     "remove the car's own acceleration"),
    (83.6, 93.8, r"$\boldsymbol{\omega}_{\mathrm{true}} = \boldsymbol{\omega} - \mathbf{b}$", "Kalman filter: learn the bias"),
]]

SUMMARY = [   # (tex, sentence)
    (r"$\int \boldsymbol{\omega}\,dt$", "The gyroscope is smooth, but every small offset adds up."),
    (r"$\mathbf{f}\,/\,|\mathbf{f}|$", "The accelerometer never drifts, but acceleration fools it."),
    (r"$\alpha$", "Blend them, and subtract what you know about your own motion."),
]


# ── the track ─────────────────────────────────────────────────────────────────
STRAIGHT, RADIUS = 60.0, 15.0            # a stadium: two straights, two half circles
LAP = 2 * STRAIGHT + 2 * math.pi * RADIUS
TRACK_HALF = 3.2                          # half width of the asphalt


def smoothstep(e0, e1, x):
    u = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return u * u * (3 - 2 * u)


def ground(x, z):
    """Y-up terrain: long hills on the south straight (true pitch), a side slope on the north
    straight (true roll), flat turns, and a flat start at the west end of the south straight."""
    straight = 1.0 - smoothstep(24.0, 30.0, np.abs(x))
    south = 1.0 - smoothstep(-6.0, -2.0, z)
    north = smoothstep(2.0, 6.0, z)
    hills = 1.2 * np.sin((x + 16.0) / 10.0) ** 2 * smoothstep(-16.0, -8.0, x)   # 31 m period, <= 6.8 deg
    side = 0.10 * (z - 15.0)
    return straight * (south * hills + north * side)


def track_point(s):
    """Centre line at arc length s (m), counter-clockwise seen from above, from the west end
    of the south straight (z = -RADIUS) heading +x."""
    s = s % LAP
    h = STRAIGHT / 2
    if s < STRAIGHT:
        return np.array([-h + s, -RADIUS])
    s -= STRAIGHT
    if s < math.pi * RADIUS:
        a = -math.pi / 2 + s / RADIUS
        return np.array([h + RADIUS * math.cos(a), RADIUS * math.sin(a)])
    s -= math.pi * RADIUS
    if s < STRAIGHT:
        return np.array([h - s, RADIUS])
    s -= STRAIGHT
    a = math.pi / 2 + s / RADIUS
    return np.array([-h + RADIUS * math.cos(a), RADIUS * math.sin(a)])


def curvature(s):
    s = s % LAP
    on_straight = s < STRAIGHT or STRAIGHT + math.pi * RADIUS <= s < 2 * STRAIGHT + math.pi * RADIUS
    return 0.0 if on_straight else 1 / RADIUS


def track_distance(x, z):
    """Distance from (x, z) to the track's centre line."""
    h = STRAIGHT / 2
    cx = np.clip(x, -h, h)
    return np.abs(np.hypot(x - cx, z) - RADIUS)


def terrain_arrays(half_x=70.0, half_z=45.0, cell=0.5):
    xs = np.arange(-half_x, half_x + 1e-9, cell)
    zs = np.arange(-half_z, half_z + 1e-9, cell)
    X, Z = np.meshgrid(xs, zs)
    Y = ground(X, Z)
    pos = np.stack([X, Y, Z], -1).reshape(-1, 3).astype(np.float32)
    nz, nx = X.shape
    i = np.arange(nz - 1)[:, None] * nx + np.arange(nx - 1)[None, :]
    idx = np.stack([i, i + nx, i + 1, i + 1, i + nx, i + nx + 1], -1).reshape(-1).astype(np.uint32)
    # a tint over the ground texture: the asphalt band is lighter
    road = (1.0 - smoothstep(TRACK_HALF - 0.4, TRACK_HALF + 0.4, track_distance(X, Z)))[..., None]
    col = np.array([0.80, 0.86, 1.0]) * (1 - road) + np.array([1.45, 1.45, 1.5]) * road
    uv = np.stack([X / GRID, Z / GRID], -1).reshape(-1, 2)
    return pos, idx, col.reshape(-1, 3).astype(np.float32), uv.astype(np.float32)


GRID = 5.0                     # metres per ground tile


def ground_texture(n=512, line_cm=6.0, seed=5):
    """One GRID x GRID tile, repeated: dark grain with an anti-aliased grid line on its border."""
    px = n / (GRID * 100.0)                     # pixels per cm
    y, x = np.mgrid[0:n, 0:n] + 0.5
    edge = np.minimum(np.minimum(x, n - x), np.minimum(y, n - y))
    line = np.clip(line_cm * px / 2 + 0.75 - edge, 0.0, 1.0)
    rng = np.random.default_rng(seed)
    grain = rng.normal(0.0, 1.0, (n, n))
    grain = (grain + np.roll(grain, 1, 0) + np.roll(grain, 1, 1) + np.roll(grain, (1, 1), (0, 1))) / 4
    base = 0.155 + 0.018 * grain
    v = base * (1 - line) + 0.33 * line
    img = np.clip(np.stack([v * 0.92, v * 0.96, v * 1.04, np.ones_like(v)], -1) * 255, 0, 255).astype(np.uint8)
    tex = tp.data_texture(img, True)
    tex.wrap_s = tp.TextureWrapping.Repeat
    tex.wrap_t = tp.TextureWrapping.Repeat
    tex.anisotropy = 16
    return tex


def edge_ribbons(width=0.22, step=0.4, lift=0.03):
    """The two painted edge lines as flat ribbons following the track, just above the ground."""
    ss = np.arange(0.0, LAP + step, step)
    P = np.array([track_point(v) for v in ss])
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    N = np.c_[-T[:, 1], T[:, 0]]
    pos, idx = [], []
    for off in (TRACK_HALF - 0.3, -(TRACK_HALF - 0.3)):
        for side in (-0.5, 0.5):
            q = P + N * (off + side * width)
            pos.append(np.c_[q[:, 0], ground(q[:, 0], q[:, 1]) + lift, q[:, 1]])
    pos = np.concatenate(pos).astype(np.float32)
    m = len(ss)
    for r in range(2):
        a0, b0 = 2 * r * m, (2 * r + 1) * m
        i = np.arange(m - 1)
        idx.append(np.stack([a0 + i, b0 + i, a0 + i + 1, a0 + i + 1, b0 + i, b0 + i + 1], -1).reshape(-1))
    return pos, np.concatenate(idx).astype(np.uint32)


# ── rotations ─────────────────────────────────────────────────────────────────
def q2R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R2q(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    return ((R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w), w)


def align(a, b):
    """Smallest rotation matrix taking unit vector a onto unit vector b."""
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.999999:
        return -np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def tilt_error(U, up_true):
    c = np.clip(np.sum(U * up_true, 1) / np.linalg.norm(U, axis=1), -1, 1)
    return np.degrees(np.arccos(c))


# ── estimators: world up, expressed in the car's frame ────────────────────────
def gyro_only(W, up0, k0):
    u = up0.copy()
    out = np.zeros((len(W), 3))
    for k in range(len(W)):
        if k > k0:
            u = u - np.cross(W[k], u) * DT
            u /= np.linalg.norm(u)
        out[k] = u
    return out


def complementary(W, F, up0, alpha):
    u = up0.copy()
    out = np.zeros((len(W), 3))
    for k in range(len(W)):
        u = u - np.cross(W[k], u) * DT
        u = alpha * u + (1 - alpha) * F[k] / np.linalg.norm(F[k])
        u /= np.linalg.norm(u)
        out[k] = u
    return out


def own_accel(W, v):
    """The car's own acceleration in its frame, from wheel speed v: dv/dt forward plus w x v."""
    vb = np.zeros((len(v), 3))
    vb[:, 2] = v
    a = np.cross(W, vb)
    a[:, 2] += np.gradient(v, DT)
    return a


def kalman(W, F, up0, sig_g, sig_z, q_bias=(1e-4) ** 2):
    """State: up (3) and gyro bias (3). The bias-corrected gyro propagates up; the
    direction of the (corrected) specific force measures it with stddev sig_z."""
    x = np.r_[up0, 0, 0, 0]
    P = np.diag([1e-4] * 3 + [np.radians(1.0) ** 2] * 3)
    I6 = np.eye(6)
    H = np.hstack([np.eye(3), np.zeros((3, 3))])
    Rm = sig_z ** 2 * np.eye(3)
    Q = np.diag([(sig_g * DT) ** 2] * 3 + [q_bias * DT] * 3)
    out = np.zeros((len(W), 3))
    bias = np.zeros((len(W), 3))
    for k in range(len(W)):
        u, b = x[:3], x[3:]
        w = W[k] - b
        Fk = I6.copy()
        Fk[:3, :3] -= skew(w) * DT
        Fk[:3, 3:] = -skew(u) * DT
        x = np.r_[u - np.cross(w, u) * DT, b]
        P = Fk @ P @ Fk.T + Q
        z = F[k] / np.linalg.norm(F[k])
        K = P @ H.T @ np.linalg.inv(H @ P @ H.T + Rm)
        x = x + K @ (z - x[:3])
        P = (I6 - K @ H) @ P
        x[:3] /= np.linalg.norm(x[:3])
        out[k] = x[:3]
        bias[k] = x[3:]
    return out, bias


# ── the car ───────────────────────────────────────────────────────────────────
WHEEL_TAG = ("WheelFL", "WheelFR", "WheelBL", "WheelBR")   # PhysX order: FR, FL, RR, RL (see warp_mudsnow_drive)


def build_car():
    """The Evoque from threepp_data: body shell plus four wheel rigs, in the chassis frame."""
    chassis = tp.Group()
    rigs = [tp.Group() for _ in range(4)]
    for r in rigs:
        chassis.add(r)
    body = tp.ModelLoader().load(data_file("models", "gltf", "2015_land-rover_range_rover_evoque_coupe",
                                           "scene.gltf"))
    body.scale.set(100.0, 100.0, 100.0)      # the gltf bakes a 0.01 at its root
    body.position.y = -1.0                   # contact patches at y = 0; the chassis centre rides ~1 m up
    chassis.add(body)
    parts = [[] for _ in range(4)]
    lamps = []

    def sort(o):
        if type(o).__name__ != "Mesh":
            return
        try:
            o.material.ao_map = None         # AO and metal-roughness share one texture; R was never authored
        except Exception:                    # noqa: BLE001
            pass
        o.cast_shadow = True
        for i, tag in enumerate(WHEEL_TAG):
            if tag in o.name:
                parts[i].append(o)
                return
        if "lights_position_back" in o.name:
            lamps.append(o)
    body.traverse(sort)
    for i, group in enumerate(parts):
        lo, hi = np.full(3, 1e30), np.full(3, -1e30)
        for part in group:
            a = part.geometry.get_attribute("position")
            lo, hi = np.minimum(lo, a.min(0)), np.maximum(hi, a.max(0))
        hub = 0.5 * (lo + hi)
        for part in group:
            clone = part.clone()
            part.visible = False
            clone.visible = True
            clone.position.set(float(-hub[0]), float(-hub[1]), float(-hub[2]))
            rigs[i].add(clone)
    brake = standard(0x201a18, 0.6, emissive=0xff2a12, emissive_intensity=0.6)
    for m in lamps:
        m.set_material(brake)
    return chassis, rigs, brake


# ── choreography ──────────────────────────────────────────────────────────────
class Record:
    pass


def choreograph(verbose=True):
    world = tp.PhysxWorld(fixed_timestep=DT)
    pos, idx, _, _ = terrain_arrays()
    g = tp.BufferGeometry()
    g.set_attribute("position", pos)
    g.set_index(idx)
    world.add_static_trimesh(tp.Mesh(g, tp.MeshBasicMaterial()))

    s0 = 4.0
    p0 = track_point(s0)
    y0 = float(ground(p0[0], p0[1])) + 1.0
    q0 = tp.Quaternion(0, math.sin(math.pi / 4), 0, math.cos(math.pi / 4))   # forward +z turned to +x
    car = tp.PhysxVehicle(world, position=tp.Vector3(p0[0], y0, p0[1]), rotation=q0)
    node, mount = tp.Group(), tp.Group()
    mount.position.set(*IMU_MOUNT)
    node.add(mount)
    node.position.set(p0[0], y0, p0[1])
    node.quaternion.set(q0.x, q0.y, q0.z, q0.w)
    node.update_matrix_world()
    car.associate(node)
    imu = tp.Imu(mount)                      # ICM-42688-P white noise by default
    gn = imu.gyro_noise
    gn.constant_bias = tp.Vector3(*np.radians(BIAS_DPS))
    gn.seed = 11
    imu.gyro_noise = gn
    an = imu.accel_noise
    an.seed = 12
    imu.accel_noise = an
    imu.reset()
    world.register_sensor(imu)

    n_frames = int(round(TL.duration * FPS)) + 1
    n = n_frames * SUB
    rec = Record()
    rec.pos = np.zeros((n_frames, 3))
    rec.quat = np.zeros((n_frames, 4))
    rec.wheel = np.zeros((n_frames, 4, 7))
    rec.brake = np.zeros(n_frames)
    rec.pedals = np.zeros((n_frames, 3))       # throttle, brake, steer as applied
    truth_q = np.zeros((n, 4))
    wheel_v = np.zeros(n)
    speed = np.zeros(n)
    s_track = s0
    chunks = []
    brake = 0.0
    for k in range(n):
        t = k * DT
        p = car.position
        pp = np.array([p.x, p.z])
        while np.linalg.norm(track_point(s_track + 0.05) - pp) < np.linalg.norm(track_point(s_track) - pp):
            s_track += 0.05
        tgt = track_point(s_track + 7.0)
        qq = car.quaternion
        R = q2R((qq.x, qq.y, qq.z, qq.w))
        fwd, right = R @ np.array([0, 0, 1.0]), R @ np.array([1.0, 0, 0])
        d = np.array([tgt[0] - pp[0], 0.0, tgt[1] - pp[1]])
        steer = float(np.clip(math.atan2(np.dot(d, right), np.dot(d, fwd)) * 1.6, -1, 1))
        car.set_steer(steer)
        ahead = max(curvature(s_track + dd) for dd in (0.0, 10.0, 20.0))
        v_set = 0.0 if t < T_GO else (V_TURN if ahead > 0 else V_STRAIGHT)
        v = car.forward_speed
        if v_set == 0.0:
            throttle = 0.0
            brake = 1.0
        else:
            e = v_set - v
            brake = 0.0 if e > -1.0 else min(0.12, -0.03 * e)
            throttle = float(np.clip(0.25 * e + 0.12, 0.0, THROTTLE_MAX))
        car.set_throttle(throttle)
        car.set_brake(brake)
        world.step(DT)
        qq = car.quaternion
        truth_q[k] = (qq.x, qq.y, qq.z, qq.w)
        speed[k] = car.forward_speed
        wheel_v[k] = R_WHEEL * np.mean([car.wheel_angular_speed(i) for i in range(4)])
        if k % SUB == SUB - 1:
            f = k // SUB
            p = car.position
            rec.pos[f] = (p.x, p.y, p.z)
            rec.quat[f] = truth_q[k]
            for i in range(4):
                lp, lq = car.wheel_local_pose(i)
                rec.wheel[f, i] = (lp.x, lp.y, lp.z, lq.x, lq.y, lq.z, lq.w)
            rec.brake[f] = brake if t >= T_GO else 0.0
            rec.pedals[f] = (throttle, brake, steer)
        if k % 240 == 239:
            chunks.append(imu.drain_array())
    chunks.append(imu.drain_array())
    S = np.concatenate(chunks)[:n]
    rec.W, rec.F = S[:, 1:4], S[:, 4:7]
    rec.speed, rec.wheel_v = speed, wheel_v
    rec.n_frames = n_frames
    tt = np.arange(n) * DT
    Rt = np.array([q2R(q) for q in truth_q])
    # the car's own acceleration in the world: a = R f + g, averaged over the last 0.15 s
    a_world = np.einsum("nij,nj->ni", Rt, rec.F) + np.array([0.0, -9.81, 0.0])
    cs = np.cumsum(np.r_[np.zeros((1, 3)), a_world], 0)
    kk = np.arange(n_frames) * SUB + SUB - 1
    lo = np.maximum(kk - 35, 0)
    rec.acc = (cs[kk + 1] - cs[lo]) / (kk + 1 - lo)[:, None]
    rec.up_true = Rt[:, 1, :]                 # world up in the car frame: R^T e_y
    rec.fwd = Rt[:, :, 2]                     # the car's forward axis in the world
    driving = tt > T_GO + 2.0
    acc = np.gradient(speed, DT)
    m = {}
    m["top_speed"] = float(speed.max())
    m["laps"] = float(speed[tt > T_GO].sum() * DT / LAP)
    m["brake_peak"] = float(-acc[driving].min())

    # ---- the four estimates
    W, F = rec.W, rec.F
    parked = (tt > 2.0) & (tt < T_GO - 0.2)
    m["f_parked"] = float(np.linalg.norm(F[parked], axis=1).mean())
    m["gyro_parked"] = np.degrees(W[parked].mean(0))
    k0 = int(T_GYRO0 / DT)
    rec.u_gyro = gyro_only(W, rec.up_true[k0], k0)
    rec.u_acc = F / np.linalg.norm(F, axis=1, keepdims=True)
    e_g = tilt_error(rec.u_gyro, rec.up_true)
    e_a = tilt_error(rec.u_acc, rec.up_true)
    m["gyro_err_10s"] = float(e_g[int((T_GYRO0 + 10) / DT)])
    m["gyro_err_end"] = float(e_g[int(T_GYRO0 / DT):int((TL.end("gyro") - 0.5) / DT)].max())
    m["acc_err_parked"] = float(e_a[parked].mean())
    m["acc_err_brake"] = float(e_a[driving & (acc < -2)].max())
    m["acc_err_turn"] = float(e_a[driving & (np.abs(W[:, 1]) > 0.2) & (np.abs(acc) < 0.5)].max())
    sweep = []
    for alpha in (0.99, 0.995, 0.998, 0.999, 0.9995, 0.9998, 0.9999, 0.99995):
        e = tilt_error(complementary(W, F, rec.up_true[0], alpha), rec.up_true)
        sweep.append((DT * alpha / (1 - alpha), math.sqrt(np.mean(e[driving] ** 2)), alpha))
    rec.sweep = sweep
    tau, rms, alpha = min(sweep, key=lambda r: r[1])
    m["tau"], m["comp_rms"], m["alpha"] = tau, rms, alpha
    rec.u_comp = complementary(W, F, rec.up_true[0], alpha)
    sig_g = 2.8e-3 * math.pi / 180 * math.sqrt(1 / DT)      # ICM-42688-P gyro white noise per sample
    Fc = F - own_accel(W, wheel_v)
    rec.u_kf, rec.bias = kalman(W, Fc, rec.up_true[0], sig_g, KF_SIG_Z)
    e_k = tilt_error(rec.u_kf, rec.up_true)
    e_c = tilt_error(rec.u_comp, rec.up_true)
    m["kf_rms"] = float(math.sqrt(np.mean(e_k[driving] ** 2)))
    m["kf_max"] = float(e_k[driving].max())
    m["bias_est"] = np.degrees(rec.bias[int((TL.end("kalman") - 1.0) / DT)])
    # heading: the gyro integrated from the drive start, in full 3D
    R = Rt[int(T_GO / DT)].copy()
    yaw_err = np.zeros(n)
    k1 = int(T_GO / DT)
    for k in range(k1 + 1, n):
        w = W[k] * DT
        th = np.linalg.norm(w)
        K = skew(w / th) if th > 0 else np.zeros((3, 3))
        R = R @ (np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K)
        fe, ft = R[:, 2], Rt[k][:, 2]
        yaw_err[k] = math.degrees(math.atan2(fe[0] * ft[2] - fe[2] * ft[0], fe[0] * ft[0] + fe[2] * ft[2]))
    rec.yaw_err = yaw_err
    m["yaw_err_end"] = float(abs(yaw_err[int((TL.end("heading") - 0.5) / DT)]))
    # per-frame errors for the plots
    at = lambda e: e[SUB - 1::SUB][:n_frames]  # noqa: E731
    rec.err = {"gyro": at(e_g), "acc": at(e_a), "comp": at(e_c), "kf": at(e_k)}
    rec.meta = m
    if verbose:
        print(f"[drive] {n} IMU samples at {1 / DT:.0f} Hz; top speed {m['top_speed']:.1f} m/s, {m['laps']:.2f} laps, "
              f"hardest braking {m['brake_peak']:.1f} m/s^2")
        print(f"[parked] |f| {m['f_parked']:.3f} m/s^2, gyro {m['gyro_parked'].round(2)} deg/s (bias set {BIAS_DPS})")
        print(f"[gyro]  from truth at {T_GYRO0} s: {m['gyro_err_10s']:.2f} deg after 10 s, "
              f"up to {m['gyro_err_end']:.2f} deg within {TL.end('gyro') - 0.5 - T_GYRO0:.0f} s")
        print(f"[accel] parked {m['acc_err_parked']:.2f} deg; braking up to {m['acc_err_brake']:.1f}, "
              f"cornering up to {m['acc_err_turn']:.1f} deg")
        print("[comp]  " + ", ".join(f"tau {a:.2f} s: {b:.2f}" for a, b, _ in sweep) + " deg rms")
        print(f"[kalman] rms {m['kf_rms']:.2f} deg, max {m['kf_max']:.2f}; bias {m['bias_est'].round(2)} deg/s "
              f"vs set {BIAS_DPS}")
        print(f"[heading] gyro-integrated heading off by {m['yaw_err_end']:.1f} deg at {TL.end('heading') - 0.5:.0f} s")
    return rec


IMU_MOUNT = (0.0, 0.25, 0.3)     # on the centre console, in the chassis frame


# ── the picture ───────────────────────────────────────────────────────────────
class Film3D:
    def __init__(self, st, rec):
        self.st, self.rec = st, rec
        pos, idx, col, uv = terrain_arrays()
        g = tp.BufferGeometry()
        g.set_attribute("position", pos)
        g.set_attribute("color", col)
        g.set_attribute("uv", uv)
        g.set_index(idx)
        g.compute_vertex_normals()
        m = standard(0xffffff, 0.95)
        m.map = ground_texture()
        m.vertex_colors = True
        m.env_map_intensity = 0.05
        self.terrain = tp.Mesh(g, m)
        self.terrain.receive_shadow = True
        st.scene.add(self.terrain)
        rp, ri = edge_ribbons()
        rg = tp.BufferGeometry()
        rg.set_attribute("position", rp)
        rg.set_index(ri)
        rg.compute_vertex_normals()
        lines = standard(0x8f98a6, 0.8)
        lines.side = tp.Side.Double
        self.lines = tp.Mesh(rg, lines)
        self.lines.receive_shadow = True
        st.scene.add(self.lines)

        self.car, self.rigs, self.brake_mat = build_car()
        st.scene.add(self.car)
        # ghosts: the car at the true position, tilted by each estimate's error
        self.ghosts = {}
        self.ghost_rigs = {}
        for key, c in (("gyro", C_GYRO), ("acc", C_ACC), ("comp", C_FUSE), ("kf", C_KF)):
            gh, gm = ghost(self.car, c, opacity=0.0)
            gh.visible = False
            st.scene.add(gh)
            self.ghosts[key] = (gh, gm)
            self.ghost_rigs[key] = list(gh.children)[:4]     # a clone keeps child order: the wheel rigs first
        # the IMU chip, seen through the body
        self.chip = tp.Mesh(tp.BoxGeometry(0.16, 0.05, 0.16), standard(C_IMU, 0.3, emissive=C_IMU,
                                                                      emissive_intensity=1.6))
        self.chip.position.set(*IMU_MOUNT)
        self.car.add(self.chip)
        xray(self.chip, order=12)
        # the specific force, from the chip
        self.f_arrow = Arrow3D(0xffffff, radius=0.035, parent=self.car, emissive=0.9)
        xray(self.f_arrow.group, order=13)
        self.f_arrow.set_opacity(0.0)
        # gyro rings about the car's three axes
        self.rings = []
        for axis, c, r in ((0, AXIS_C[0], 2.45), (1, AXIS_C[1], 2.7), (2, AXIS_C[2], 1.4)):
            holder = tp.Group()
            holder.position.set(0, 0.1, 0.1)
            if axis == 0:
                holder.rotate_y(math.pi / 2)        # ring around x
            elif axis == 1:
                holder.rotate_x(math.pi / 2)        # ring around y
            self.car.add(holder)
            ring = Ring3D(c, radius=r, tube=0.03, parent=holder, emissive=1.2)
            self.rings.append(ring)
        # heading arrows on the ground
        self.head_true = Arrow3D(0xffffff, radius=0.06, emissive=0.8, parent=st.scene)
        self.acc_arrow = Arrow3D(C_OWN, radius=0.07, emissive=1.1, parent=st.scene)
        xray(self.acc_arrow.group, order=14)
        self.acc_arrow.set_opacity(0.0)
        self.head_gyro = Arrow3D(C_GYRO, radius=0.06, emissive=1.0, parent=st.scene)
        for a in (self.head_true, self.head_gyro):
            a.set_opacity(0.0)

    def pose(self, i):
        rec = self.rec
        self.car.position.set(*rec.pos[i])
        self.car.quaternion.set(*rec.quat[i])
        for r, w in zip(self.rigs, rec.wheel[i]):
            r.position.set(*w[:3])
            r.quaternion.set(*w[3:])
        self.brake_mat.emissive_intensity = 0.6 + 5.0 * min(1.0, rec.brake[i] / 0.12)


def heading_of(fwd):
    """Azimuth of a Y-up forward vector in OrbitCamera's yup convention (from +x toward -z)."""
    return math.atan2(-fwd[2], fwd[0])


class Painter:
    def __init__(self, st, f3: Film3D, rec, ov: Hud):
        self.st, self.f3, self.rec, self.ov = st, f3, rec, ov
        m = rec.meta
        fill = {
            1: f"Parked, the accelerometer does not read zero. It reads {m['f_parked']:.2f} m/s² straight up: "
               f"the ground holding the car against gravity.",
            4: f"But this gyroscope reads {np.linalg.norm(m['gyro_parked']):.1f} °/s while the car stands still. "
               f"Small, but it adds up: up to {m['gyro_err_end']:.1f}° off within {TL.end('gyro') - 0.5 - T_GYRO0:.0f} s.",
            6: f"But it cannot tell gravity from the car's own acceleration. A corner rolls it by up to "
               f"{m['acc_err_turn']:.0f}°, and braking tips it by up to {m['acc_err_brake']:.0f}°.",
            8: f"Lean too hard and every brake shows through; too little and the gyroscope drifts. The best blend "
               f"here trusts the gyroscope for about {m['tau']:.0f} s: {m['comp_rms']:.1f}° off on average.",
            10: f"A Kalman filter does that, and learns the gyroscope's offset as it goes: {m['kf_rms']:.1f}° "
                f"off on average, {m['comp_rms'] / m['kf_rms']:.0f} times closer than the best blend.",
        }
        self.captions = [(a, b, fill.get(k, txt)) for k, (a, b, txt) in enumerate(CAPTIONS)]
        # the chase camera follows the car and turns with it, smoothed
        n = rec.n_frames
        head = np.unwrap([heading_of(rec.fwd[i * SUB + SUB - 1]) for i in range(n)])
        sm = np.empty(n)
        sm[0] = head[0]
        k = 1.0 - math.exp(-1.0 / (FPS * 1.2))
        for i in range(1, n):
            sm[i] = sm[i - 1] + k * (head[i] - sm[i - 1])
        self.head = sm
        self.frame = lambda t: min(max(int(round(t * FPS)), 0), n - 1)  # noqa: E731
        # tilt errors as fixed 0.1 s block means for the plot
        nb = n // BLOCK
        self.blocks_t = (np.arange(nb) * BLOCK + (BLOCK - 1) / 2) / FPS
        self.blocks = {key: e[:nb * BLOCK].reshape(nb, BLOCK).mean(1) for key, e in rec.err.items()}
        self.camera = OrbitCamera(CAM, drift=((1.0, 0.19), (0.5, 0.15)), frame="yup",
                                  follow=lambda t: rec.pos[self.frame(t)], heading=lambda t: self.head[self.frame(t)])

    # ---- 3D --------------------------------------------------------------------
    def set3d(self, t, i, intro=None):
        rec, f3, st = self.rec, self.f3, self.st
        f3.pose(i)
        st.look(*self.camera(t))
        st.follow(rec.pos[i], height=14.0)
        k = i * SUB + SUB - 1
        R = q2R(rec.quat[i])

        # chip, force arrow and rings in the sense beat
        sense = envelope(t, TL.start("sense") + 0.6, TL.end("sense") - 0.2, 0.6, 0.6)
        f3.chip.visible = sense > 0.003 or envelope(t, at(76.0), at(84.0), 0.5, 0.5) > 0.003
        f3.chip.material.opacity = max(sense, envelope(t, at(76.0), at(84.0), 0.5, 0.5))
        fa = envelope(t, at(13.6), TL.end("sense") - 0.2, 0.5, 0.6)
        if fa > 0.003:
            f = rec.F[max(k - 12, 0):k + 1].mean(0)          # 50 ms of samples: the arrow reads, not buzzes
            base = np.array(IMU_MOUNT)
            f3.f_arrow.set(base, base + f * 0.16, opacity=fa)
        else:
            f3.f_arrow.set_opacity(0.0)
        rv = envelope(t, at(19.4), TL.end("sense") + 0.4, 0.6, 0.6)
        w = np.abs(rec.W[max(k - 12, 0):k + 1].mean(0))
        for j, ring in enumerate(f3.rings):
            glow = clamp01(w[j] / 0.35)
            ring.set_opacity(rv * (0.35 + 0.65 * glow))
            ring.mat.emissive_intensity = 0.6 + 3.0 * glow

        # ghosts: true position, true heading, the estimate's tilt
        show = {"gyro": envelope(t, INTRO["gyro"], TL.end("fuse") - 6.0, 0.05, 0.8),
                "acc": envelope(t, INTRO["acc"], TL.start("fuse") + 5.0, 0.05, 0.8),
                "comp": envelope(t, INTRO["comp"], TL.end("kalman") - 5.0, 0.05, 0.8),
                "kf": envelope(t, INTRO["kf"], TL.end("heading") + 0.4, 0.05, 0.8)}
        # an introduction: the car is hidden and only the new ghost is shown, blinking
        f3.car.visible = intro is None
        if intro is not None:
            show = {key: (0.5 - 0.5 * math.cos(2 * math.pi * 2.5 * intro[1])) * 1.6 if key == intro[0] else 0.0
                    for key in show}
        ests = {"gyro": rec.u_gyro, "acc": rec.u_acc, "comp": rec.u_comp, "kf": rec.u_kf}
        ut = rec.up_true[k]
        for key, (gh, gm) in f3.ghosts.items():
            op = show[key]
            if op <= 0.003:
                gh.visible = False
                continue
            gh.visible = True
            gm.opacity = min(0.38 * op, 0.6)
            ue = ests[key][k] / np.linalg.norm(ests[key][k])
            Re = R @ align(ue, ut)                     # the pose whose up, seen from the car, is the estimate
            gh.position.set(*rec.pos[i])
            gh.quaternion.set(*R2q(Re))
            for r, w in zip(f3.ghost_rigs[key], rec.wheel[i]):
                r.position.set(*w[:3])
                r.quaternion.set(*w[3:])

        # the car's own acceleration, above the roof, while the accelerometer is the subject
        self.acc_label = None
        av = envelope(t, TL.start("accel"), TL.end("kalman"), 0.6, 0.6) if intro is None else 0.0
        a = rec.acc[i] * np.array([1.0, 0.0, 1.0])
        mag = float(np.linalg.norm(a))
        op = av * clamp01((mag - 0.6) / 0.8)
        if op > 0.003:
            base = rec.pos[i] + np.array([0.0, 1.25, 0.0])
            tip = base + a * 0.42
            f3.acc_arrow.set(base, tip, opacity=op)
            fwd = rec.fwd[k] * np.array([1.0, 0.0, 1.0])
            fwd /= np.linalg.norm(fwd)
            lon = float(np.dot(a, fwd))
            lat = float(np.linalg.norm(a - lon * fwd))
            if lat > abs(lon):
                text = f"cornering {lat:.1f} m/s²"
            else:
                text = f"{'braking' if lon < 0 else 'speeding up'} {abs(lon):.1f} m/s²"
            self.acc_label = (0.5 * (base + tip), text, op)
        else:
            f3.acc_arrow.set_opacity(0.0)

        # heading beat: the true forward and the gyro-integrated forward, on the ground
        hv = envelope(t, TL.start("heading") + 0.2, TL.end("heading") + 0.4, 0.4, 0.6)
        if hv > 0.003:
            fwd = rec.fwd[k] * np.array([1, 0, 1.0])
            fwd /= np.linalg.norm(fwd)
            yaw = math.radians(rec.yaw_err[k])
            c, s = math.cos(yaw), math.sin(yaw)
            fg = np.array([c * fwd[0] + s * fwd[2], 0.0, -s * fwd[0] + c * fwd[2]])
            base = rec.pos[i] + np.array([0.0, 1.4, 0.0])
            f3.head_true.set(base, base + fwd * 5.0, opacity=hv)
            f3.head_gyro.set(base, base + fg * 5.0, opacity=hv)
        else:
            f3.head_true.set_opacity(0.0)
            f3.head_gyro.set_opacity(0.0)

    # ---- 2D --------------------------------------------------------------------
    def draw2d(self, t, i, intro=None):
        ov, st, rec = self.ov, self.st, self.rec
        m = rec.meta
        k = i * SUB + SUB - 1
        ov.title_card(t, "A THREEPP LESSON  ·  PART 3", "Which Way Is Up?", "How a robot measures its own tilt",
                      C_IMU)
        ov.captions(t, self.captions)
        ov.equation_card(t, EQUATIONS)

        # the IMU readout and its callout
        a = envelope(t, at(8.2), TL.end("sense") - 0.2, 0.6, 0.6)
        if a > 0:
            w = np.degrees(rec.W[k])
            f = rec.F[k]
            rows = [(f"gyro {n}", f"{v:+6.2f} °/s", AXIS_C[j]) for j, (n, v) in enumerate(zip("xyz", w))]
            rows += [(f"accel {n}", f"{v:+6.2f} m/s²", AXIS_C[j]) for j, (n, v) in enumerate(zip("xyz", f))]
            ov.readout(1488, 64, 368, "IMU  ·  ICM-42688-P", rows, alpha=a)
            chip = st.project(np.asarray(rec.pos[i]) + q2R(rec.quat[i]) @ np.array(IMU_MOUNT))
            c1 = envelope(t, at(8.6), at(13.4), 0.5, 0.4)
            if c1 > 0:
                ov.callout(chip[:2], (chip[0] + 170, chip[1] - 150), "IMU", C_IMU, c1, grow=ease_out(remap(t, at(8.6), at(9.3))))

        # tilt error, scrolling
        pe = envelope(t, T_GYRO0 + 0.4, TL.end("kalman") - 0.2, 0.6, 0.6)
        if pe > 0:
            t1 = max(t, T_GYRO0 + 20.0)
            t0 = t1 - 20.0
            done = (i + 1) // BLOCK                        # blocks fully in the past
            series = []
            for key, c, label in (("gyro", C_GYRO, "gyroscope"), ("acc", C_ACC, "accelerometer"),
                                  ("comp", C_FUSE, "complementary"), ("kf", C_KF, "Kalman")):
                start = INTRO[key]
                if t < start:
                    continue
                xs, ys = self.blocks_t[:done], self.blocks[key][:done]
                sel = xs >= max(start, t0)
                series.append((xs[sel], ys[sel], c, label))
            ov.plot(1380, 64, 476, 330, series, (t0, t1), (0.0, 30.0), alpha=pe, title="TILT ERROR  (deg)",
                    yticks=(0, 10, 20, 30), xticks=[v for v in range(int(t0) + 1, int(t1) + 1) if v % 5 == 0],
                    xfmt=lambda v: f"{v:.0f} s", yfmt=lambda v: f"{v:.0f}°")

        # the blend sweep
        sw = envelope(t, at(65.0), TL.end("fuse") - 0.2, 0.6, 0.6)
        if sw > 0:
            taus = [r[0] for r in rec.sweep]
            rmss = [r[1] for r in rec.sweep]
            grow = smooth(remap(t, at(65.0), at(68.0)))
            nshow = max(2, int(round(grow * len(taus))))
            ov.plot(1380, 420, 476, 300, [(taus[:nshow], rmss[:nshow], C_FUSE, None)], (0.3, 100.0), (0.0, 16.0),
                    alpha=sw, title="BLEND  vs  AVERAGE ERROR", xlog=True, xticks=(1, 10, 100), yticks=(0, 5, 10, 15),
                    xfmt=lambda v: f"{v:.0f} s", yfmt=lambda v: f"{v:.0f}°",
                    markers=[(m["tau"], m["comp_rms"], 0xffffff)] if grow > 0.99 else ())
            if grow > 0.99:
                ov.text(1380 + 238, 420 + 300 - 16, "how long to trust the gyroscope", size=15, color=C_DIM,
                        alpha=sw, anchor="ms")

        # the bias the Kalman filter learns
        bv = envelope(t, at(84.0), TL.end("kalman") + 0.4, 0.6, 0.6)
        if bv > 0:
            xs = np.arange(0, self.frameidx(t) + 1, 15) / FPS
            idx = (xs / DT).astype(int)
            series = [(xs, np.degrees(rec.bias[idx, j]), AXIS_C[j], f"axis {n}") for j, n in enumerate("xyz")]
            series += [((0.0, t), (BIAS_DPS[j], BIAS_DPS[j]), AXIS_C[j], None) for j in range(3)]
            ov.plot(1380, 420, 476, 300, series, (0.0, max(t, 30.0)), (-1.0, 1.0), alpha=bv,
                    title="GYRO BIAS: LEARNED vs SET  (deg/s)", yticks=(-1, -0.5, 0, 0.5, 1),
                    xticks=tuple(range(0, int(max(t, 30.0)) + 1, 25)), xfmt=lambda v: f"{v:.0f} s",
                    yfmt=lambda v: f"{v:+.1f}", width=2.0)

        # heading
        hv = envelope(t, TL.start("heading") + 0.4, TL.end("heading") + 0.4, 0.5, 0.6)
        if hv > 0:
            ov.readout(1488, 64, 368, "HEADING", [("gyroscope only", f"{abs(rec.yaw_err[k]):5.1f}° off", C_GYRO),
                                                   ("since", f"{T_GO:.0f} s", None)], alpha=hv)

        # introducing a ghost: name it while it blinks on its own
        if intro is not None:
            key, u = intro
            col = {"gyro": C_GYRO, "acc": C_ACC, "comp": C_FUSE, "kf": C_KF}[key]
            p = st.project(np.asarray(rec.pos[i]) + np.array([0.0, 0.75, 0.0]))
            la = clamp01(u / 0.12) * clamp01((1.0 - u) / 0.08)
            ov.callout(p[:2], (p[0] - 230, p[1] - 190), INTRO_LABEL[key], col, la, size=30,
                       grow=ease_out(remap(u, 0.0, 0.25)))

        if self.acc_label is not None:
            tip, text, op = self.acc_label
            p = st.project(tip)
            ov.text(p[0], p[1] - 34, text, size=26, color=C_OWN, alpha=op, kind="semibold", anchor="ms")

        # what the driver is doing
        dv = envelope(t, T_GO - 1.0, TL.end("kalman") + 0.4, 0.6, 0.6)
        if dv > 0:
            thr, brk, steer = rec.pedals[i]
            x, y, w = 70, 690, 340
            ov.panel(x, y, w, 196, radius=16, alpha=0.66 * dv, outline=0x8aa0c0, outline_alpha=0.16)
            ov.text(x + 24, y + 22, "DRIVER", size=16, color=C_DIM, alpha=dv, kind="semibold", tracking=2.2)
            for j, (label, v, c) in enumerate((("throttle", thr, C_FUSE), ("brake", brk, 0xff5d73))):
                yy = y + 62 + j * 42
                ov.text(x + 24, yy, label, size=19, color=C_DIM, alpha=dv, anchor="lm")
                ov.bar(x + 116, yy - 6, 140, 12, v, c, alpha=dv)
                ov.text(x + w - 24, yy, f"{v * 100:3.0f} %", size=21, color=C_TEXT, alpha=dv, kind="numeric", anchor="rm")
            yy = y + 146
            ov.text(x + 24, yy, "steering", size=19, color=C_DIM, alpha=dv, anchor="lm")
            ov.bar(x + 116, yy - 3, 140, 6, 1.0, 0x2a3342, alpha=dv)
            ov.line([(x + 186, yy - 10), (x + 186, yy + 10)], 0x55627a, 1.4, dv)
            ov.circle(x + 186 + 70 * steer, yy, 8.5, fill=C_IMU, alpha=dv)
            ov.text(x + w - 24, yy, f"{round(math.degrees(steer * MAX_STEER)) + 0:+3d}°", size=21, color=C_TEXT, alpha=dv,
                    kind="numeric", anchor="rm")

        ov.summary(t, TL.start("outro"), TL.end("outro"), SUMMARY, "Driven, sensed and rendered with threepp",
                   C_IMU, C_IMU, text_dx=240)

    def frameidx(self, t):
        return self.frame(t)


# camera: (time, azimuth deg relative to the car's heading, elevation deg, distance m, look offset, fov)
CAM = [
    (at(0.0), 125, 26, 24.0, (6.0, 0.0, 3.0), 32),
    (at(5.8), 115, 18, 14.0, (1.0, 0.4, 0.0), 32),
    (at(8.2), 105, 12, 8.5, (0.0, 0.7, 0.0), 30),
    (at(13.5), 110, 12, 8.0, (0.0, 0.7, 0.0), 30),
    (at(19.0), 135, 16, 10.5, (0.0, 0.6, 0.0), 30),
    (at(24.5), 150, 16, 12.5, (0.0, 0.6, 0.0), 30),
    (at(41.5), 150, 15, 12.5, (0.0, 0.6, 0.0), 30),
    (at(44.0), 100, 10, 12.0, (0.0, 0.6, 0.0), 30),
    (at(57.5), 100, 10, 12.0, (0.0, 0.6, 0.0), 30),
    (at(60.0), 135, 18, 13.0, (0.0, 0.6, 0.0), 30),
    (at(75.5), 140, 18, 13.0, (0.0, 0.6, 0.0), 30),
    (at(78.0), 160, 20, 12.5, (0.0, 0.6, 0.0), 30),
    (at(93.5), 165, 20, 13.0, (0.0, 0.6, 0.0), 30),
    (at(96.5), 180, 52, 20.0, (0.0, 0.0, 0.0), 32),
    (at(108.0), 200, 36, 32.0, (0.0, 0.0, 0.0), 32),
]


# ── main ──────────────────────────────────────────────────────────────────────
def setup(width, height):
    t0 = time.time()
    rec = choreograph()
    print(f"[choreo] {rec.n_frames} frames in {time.time() - t0:.1f}s")
    st = Stage(width, height, renderer="gl", floor=False, fog=(40.0, 110.0), shadow_extent=9.0, far=250.0)
    f3 = Film3D(st, rec)
    ov = Hud(1920, 1080, math_cache=os.path.join(_HERE, "imu_tilt.math.json"))
    painter = Painter(st, f3, rec, ov)

    def render(t, hold=None):
        intro = (hold[0][1], hold[1]) if hold is not None and hold[0][0] == "intro" else None
        i = painter.frameidx(t)
        painter.set3d(t, i, intro)
        ov.begin()
        painter.draw2d(t, i, intro)
        ov.fade(fade_in_out(t, TL.duration))
        ov.end()
        return st.frame(t, hud=ov)
    render.holds = [(a, PAUSE, ("intro", key)) for key, a in INTRO.items()]
    return render, painter.captions


if __name__ == "__main__":
    run("imu_tilt", TL.duration, setup, fps=FPS)
