"""Demolition: two chrome wrecking balls through two towers, 3,120 bricks.

    python demolition.py                        # live: the camera orbits, drag to take over
    python demolition.py --shot out.png --at 2.4,6.8   # headless stills at those sim times
    python demolition.py --video 25             # headless film -> out/demolition.mp4

Both towers come down, then the rubble flies home in a rising spiral and
rebuilds them, and it starts over. Every brick is one instance of ONE
InstancedMesh and one PhysX rigid body (`world.add_instanced`), so both towers
are a single draw call that the physics moves. The Vulkan renderer ray traces
it: the bricks shadow each other as they fall, and the balls and the polished
floor mirror the collapse. The impacts play in slow motion, which is nothing
more than a smaller dt handed to `world.step`. The rebuild is not simulated:
numpy flies every brick from where it came to rest to its slot, one
`set_matrices` call per frame.

The sim clock stops at 10.5 s, where the rebuild takes over, so `--at` takes
times below that.

Needs a build with Vulkan and PhysX (tp.HAS_VULKAN, tp.HAS_PHYSX).
"""
import math
import os
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

from demo_common import (Encoder, cli_arg, parse_size, resize_handler,
                         standard_material, write_radiance_hdr)

if not (tp.HAS_VULKAN and tp.HAS_PHYSX):
    print("This demo needs a threepp build with both the Vulkan renderer and PhysX.")
    sys.exit(0)

SHOT = cli_arg("--shot", None, str)
SHOT_AT = [float(v) for v in cli_arg("--at", "4.0", str).split(",")]
VIDEO = cli_arg("--video", 0.0, float)
W, H = parse_size(cli_arg("--size", "1280x720", str))
HEADLESS = SHOT is not None or VIDEO > 0.0
FPS = 60

# --------------------------------------------------------------------------- #
#  Sky: a low sun painted into an equirect HDR, so the demo needs no assets
# --------------------------------------------------------------------------- #
SUN_ELEVATION = math.radians(15.0)
SUN_AZIMUTH = math.radians(8.0)
SUN_DIR = np.array([math.cos(SUN_ELEVATION) * math.cos(SUN_AZIMUTH),
                    math.sin(SUN_ELEVATION),
                    math.cos(SUN_ELEVATION) * math.sin(SUN_AZIMUTH)])


def make_sky_hdr(path, w=2048, h=1024):
    j = np.arange(h).reshape(h, 1)
    i = np.arange(w).reshape(1, w)
    theta = (j + 0.5) / h * math.pi
    phi = (i + 0.5) / w * 2.0 * math.pi - math.pi
    y = np.broadcast_to(np.cos(theta), (h, w))
    sin_t = np.sin(theta)
    d = np.stack([sin_t * np.cos(phi), y, sin_t * np.sin(phi)], axis=-1)
    cosang = np.clip((d * SUN_DIR).sum(-1), -1.0, 1.0)
    ang = np.arccos(cosang)

    # The horizon is warm toward the sun and dusky opposite it.
    toward = (0.5 + 0.5 * cosang)[..., None] ** 2.5
    horizon = (np.array([1.00, 0.50, 0.22]) * toward
               + np.array([0.30, 0.32, 0.50]) * (1.0 - toward))
    zenith = np.array([0.04, 0.10, 0.30])
    haze = np.array([0.10, 0.10, 0.13])
    up = np.clip(y, 0.0, 1.0)[..., None] ** 0.32
    down = np.clip(-y, 0.0, 1.0)[..., None] ** 0.6
    sky = np.where(y[..., None] >= 0.0,
                   horizon * (1.0 - up) + zenith * up,
                   horizon * (1.0 - down) + haze * down)

    core = np.exp(-(ang / math.radians(1.6)) ** 2)
    halo = np.exp(-(ang / math.radians(9.0)) ** 2)
    sky = sky + (core * 40.0 + halo * 0.9)[..., None] * np.array([1.0, 0.76, 0.48])
    return write_radiance_hdr(path, sky)


# --------------------------------------------------------------------------- #
#  Renderer and scene
# --------------------------------------------------------------------------- #
canvas = tp.Canvas("threepp - demolition", width=W, height=H, vsync=False,
                   headless=HEADLESS)
# A stepped film wants one rendered frame per render() call.
renderer = tp.VulkanRenderer(canvas, flush_frames=1 if HEADLESS else 3)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 0.85
renderer.shadow_map_enabled = True

env = tp.RGBELoader().load(
    make_sky_hdr(os.path.join(tempfile.gettempdir(), "threepp_demolition_sky.hdr")))
scene = tp.Scene()
scene.background = env
scene.environment = env

sun = tp.DirectionalLight(0xffc890, 3.2)
sun.position.set(*(SUN_DIR * 80.0))
sun.cast_shadow = True
scene.add(sun)

# --------------------------------------------------------------------------- #
#  Physics
# --------------------------------------------------------------------------- #
world = tp.PhysxWorld(tp.Vector3(0.0, -9.81, 0.0), fixed_timestep=1.0 / 240.0,
                      max_substeps=4, num_threads=8, tgs_pcm=True)
stone = world.create_material(0.7, 0.6, 0.05)

# A polished floor to mirror the fall, and a pale plinth for the tower.
floor = tp.Mesh(tp.BoxGeometry(700.0, 2.0, 700.0),
                standard_material(tp.Color(0.035, 0.038, 0.045), roughness=0.16, metalness=0.6))
floor.position.y = -1.0
floor.receive_shadow = True
scene.add(floor)
world.add_static(floor, stone)

# Two towers of ivory bricks, each wound with two coloured ribbons:
# (centre x, centre z, bricks per ring, rings, ribbon colours from base to top).
BL, BH, BD = 1.0, 0.5, 0.5                      # brick length, height, depth (m)
PLINTH_H = 0.3
TOWERS = [
    (0.0, 0.0, 40, 48, [[0.55, 0.03, 0.08], [0.95, 0.22, 0.04], [1.00, 0.62, 0.10]]),
    (-20.0, -16.0, 30, 40, [[0.03, 0.08, 0.42], [0.02, 0.34, 0.60], [0.22, 0.78, 0.70]]),
]
N_BRICKS = sum(n_ring * layers for _, _, n_ring, layers, _ in TOWERS)


def ring_radius(n_ring):
    return n_ring * (BL + 0.07) / (2.0 * math.pi)


rng = np.random.default_rng(4)
bricks = tp.InstancedMesh(tp.BoxGeometry(BL, BH, BD),
                          standard_material(tp.Color(1.0, 1.0, 1.0), roughness=0.5),
                          N_BRICKS)
UP = tp.Vector3(0.0, 1.0, 0.0)
ONE = tp.Vector3(1.0, 1.0, 1.0)
IVORY = np.array([0.80, 0.78, 0.72])
home = []                                       # (position, quaternion) per brick
axis, order = [], []                            # its tower's centre, its height fraction
colors = np.empty((N_BRICKS, 3), np.float32)
m = tp.Matrix4()
for cx, cz, n_ring, layers, ribbon in TOWERS:
    radius = ring_radius(n_ring)
    ribbon = np.array(ribbon)

    # A pale plinth under the tower.
    side = 2.0 * radius + 7.5
    plinth = tp.Mesh(tp.BoxGeometry(side, PLINTH_H, side),
                     standard_material(tp.Color(0.55, 0.53, 0.50), roughness=0.55))
    plinth.position.set(cx, 0.5 * PLINTH_H, cz)
    plinth.cast_shadow = True
    plinth.receive_shadow = True
    scene.add(plinth)
    world.add_static(plinth, stone)

    # Rings of bricks, every other ring turned half a brick.
    for layer in range(layers):
        f = layer / (layers - 1)
        for k in range(n_ring):
            n = len(home)
            a = (k + 0.5 * (layer % 2)) * 2.0 * math.pi / n_ring
            p = tp.Vector3(cx + radius * math.cos(a), PLINTH_H + (layer + 0.5) * BH,
                           cz + radius * math.sin(a))
            q = tp.Quaternion().set_from_axis_angle(UP, 0.5 * math.pi - a)
            home.append((p, q))
            axis.append((cx, cz))
            order.append(f)
            m.compose(p, q, ONE)
            bricks.set_matrix_at(n, m)
            if (2 * k - layer) % n_ring < n_ring * 3 // 20:
                c = (ribbon[0] * (1.0 - 2.0 * f) + ribbon[1] * 2.0 * f if f < 0.5
                     else ribbon[1] * (2.0 - 2.0 * f) + ribbon[2] * (2.0 * f - 1.0))
            else:
                c = IVORY
            colors[n] = c * rng.uniform(0.84, 1.06)
bricks.instance_matrix_needs_update()
bricks.set_colors(colors)
bricks.cast_shadow = True
bricks.receive_shadow = True
bricks.frustum_culled = False                   # the bounds go stale once it falls
scene.add(bricks)
bodies = world.add_instanced(bricks, 500.0)

# What the rebuild needs, as arrays: where every brick belongs and when it flies.
HOME_P = np.array([[p.x, p.y, p.z] for p, _ in home])
HOME_Q = np.array([[q.x, q.y, q.z, q.w] for _, q in home])
AXIS = np.array(axis)
HOME_R = np.hypot(HOME_P[:, 0] - AXIS[:, 0], HOME_P[:, 2] - AXIS[:, 1])
HOME_TH = np.arctan2(HOME_P[:, 2] - AXIS[:, 1], HOME_P[:, 0] - AXIS[:, 0])
RISE, FLIGHT = 3.2, 1.8                         # s: base-to-top stagger, one brick's flight
DEPART = np.array(order) * RISE + rng.uniform(0.0, 0.3, N_BRICKS)
REBUILD_TIME = RISE + 0.3 + FLIGHT

# The wrecking balls:
# (tower it is aimed at, azimuth it comes from, aim height, launch time, radius).
SPEED, RANGE = 50.0, 75.0
BALL_SPECS = [(0, math.radians(180.0), 8.5, 0.6, 2.6),
              (1, math.radians(237.0), 6.5, 5.0, 2.4)]
PARKED = tp.Vector3(0.0, -60.0, 0.0)
ZERO = tp.Vector3(0.0, 0.0, 0.0)
chrome = standard_material(tp.Color(0.95, 0.96, 0.97), roughness=0.04, metalness=1.0)
balls = []
for tower, az, aim_y, t0, radius in BALL_SPECS:
    cx, cz, n_ring = TOWERS[tower][:3]
    mesh = tp.Mesh(tp.SphereGeometry(radius, 64, 40), chrome)
    mesh.position.copy(PARKED)
    mesh.cast_shadow = True
    mesh.visible = False
    scene.add(mesh)
    body = world.add(mesh, density=7800.0)
    body.set_kinematic(True)
    flight = RANGE / SPEED
    start = tp.Vector3(cx + RANGE * math.cos(az), aim_y + 3.0, cz + RANGE * math.sin(az))
    velocity = tp.Vector3(-SPEED * math.cos(az), (0.5 * 9.81 * flight * flight - 3.0) / flight,
                          -SPEED * math.sin(az))
    hit = t0 + (RANGE - ring_radius(n_ring) - 0.5 * BD - radius) / SPEED
    balls.append({"mesh": mesh, "body": body, "t0": t0, "start": start,
                  "velocity": velocity, "hit": hit, "flying": False})

# --------------------------------------------------------------------------- #
#  Timeline
# --------------------------------------------------------------------------- #
SLOW_RATE = 0.25                                # one 240 Hz substep per 60 Hz frame
REBUILD_AT = 10.5                               # sim s: both towers are down
state = {"t": 0.0, "t0": 0.0, "slow": True, "rebuild": None}


def smooth(a, b, x):
    u = min(max((x - a) / (b - a), 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def slowness(t):
    """0 in real time, 1 from just before each impact until the debris flies."""
    k = 0.0
    for ball, hold in zip(balls, (1.05, 0.9)):
        k = max(k, smooth(ball["hit"] - 0.14, ball["hit"] - 0.02, t)
                * (1.0 - smooth(ball["hit"] + hold, ball["hit"] + hold + 0.5, t)))
    return k


def time_scale(t):
    return 1.0 + (SLOW_RATE - 1.0) * slowness(t) if state["slow"] else 1.0


def slerp(a, b, t):
    """Rows of unit quaternions a -> b at t (N,), the short way round."""
    d = (a * b).sum(1)
    b = np.where(d[:, None] < 0.0, -b, b)
    th = np.arccos(np.clip(np.abs(d), 0.0, 1.0))
    s = np.sin(th)
    flat = s < 1e-5
    s = np.where(flat, 1.0, s)
    wa = np.where(flat, 1.0 - t, np.sin((1.0 - t) * th) / s)
    wb = np.where(flat, t, np.sin(t * th) / s)
    q = a * wa[:, None] + b * wb[:, None]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def pose_matrices(p, q):
    """(N, 16) column-major matrices from positions (N, 3) and unit quaternions (N, 4)."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    out = np.zeros((len(p), 16), np.float32)
    out[:, 0] = 1.0 - 2.0 * (y * y + z * z)
    out[:, 1] = 2.0 * (x * y + w * z)
    out[:, 2] = 2.0 * (x * z - w * y)
    out[:, 4] = 2.0 * (x * y - w * z)
    out[:, 5] = 1.0 - 2.0 * (x * x + z * z)
    out[:, 6] = 2.0 * (y * z + w * x)
    out[:, 8] = 2.0 * (x * z + w * y)
    out[:, 9] = 2.0 * (y * z - w * x)
    out[:, 10] = 1.0 - 2.0 * (x * x + y * y)
    out[:, 12:15] = p
    out[:, 15] = 1.0
    return out


def begin_rebuild():
    """Note where every brick came to rest. From here the bricks are flown, not simulated."""
    p0 = np.array([[v.x, v.y, v.z] for v in (b.position for b in bodies)])
    q0 = np.array([[q.x, q.y, q.z, q.w] for q in (b.quaternion for b in bodies)])
    dx, dz = p0[:, 0] - AXIS[:, 0], p0[:, 2] - AXIS[:, 1]
    r0 = np.hypot(dx, dz)
    turn = np.arctan2(dz, dx) - HOME_TH
    state["rebuild"] = {
        "tau": 0.0, "q0": q0, "y0": p0[:, 1], "r0": r0,
        # Near bricks all circle their tower the same way round on the way home,
        # which makes the vortex; far ones take the short way.
        "turn": np.where(r0 < 14.0, np.mod(turn, 2.0 * math.pi),
                         np.mod(turn + math.pi, 2.0 * math.pi) - math.pi),
        "lift": 2.5 + 0.3 * np.abs(r0 - HOME_R) + rng.uniform(0.0, 3.0, N_BRICKS),
        "balls": [(b["mesh"].position.x, b["mesh"].position.y, b["mesh"].position.z)
                  for b in balls],
    }


def rebuild_step(dt_wall):
    """Fly every brick home in a rising spiral, the base first, and the balls back out."""
    rb = state["rebuild"]
    rb["tau"] += dt_wall
    u = np.clip((rb["tau"] - DEPART) / FLIGHT, 0.0, 1.0)
    e = u * u * u * (u * (6.0 * u - 15.0) + 10.0)
    r = rb["r0"] + (HOME_R - rb["r0"]) * e
    th = HOME_TH + rb["turn"] * (1.0 - e)
    pos = np.stack([AXIS[:, 0] + r * np.cos(th),
                    rb["y0"] + (HOME_P[:, 1] - rb["y0"]) * e + rb["lift"] * np.sin(math.pi * e),
                    AXIS[:, 1] + r * np.sin(th)], axis=1)
    bricks.set_matrices(pose_matrices(pos, slerp(rb["q0"], HOME_Q, e)))
    g = smooth(0.0, 2.5, rb["tau"])
    for ball, (x, y, z) in zip(balls, rb["balls"]):
        s = ball["start"]
        ball["mesh"].position.set(x + (s.x - x) * g,
                                  y + (s.y - y) * g + 12.0 * math.sin(math.pi * g),
                                  z + (s.z - z) * g)
    if rb["tau"] >= REBUILD_TIME:
        reset()


def reset():
    state["rebuild"] = None
    for body, (p, q) in zip(bodies, home):
        body.set_pose(p, q)
        body.set_linear_velocity(ZERO)
        body.set_angular_velocity(ZERO)
    for ball in balls:
        ball["body"].set_kinematic(True)
        ball["body"].set_pose(PARKED)
        ball["mesh"].visible = False
        ball["flying"] = False
    state["t"], state["t0"] = 0.0, world.sim_time


def advance(dt_wall):
    """Step the world by one frame of wall time, slowed around the impacts."""
    if state["rebuild"] is not None:
        rebuild_step(dt_wall)
        return
    dt = dt_wall * time_scale(state["t"])
    for ball in balls:
        if not ball["flying"] and state["t"] >= ball["t0"]:
            ball["flying"] = True
            ball["mesh"].visible = True
            ball["body"].set_kinematic(False)
            ball["body"].set_pose(ball["start"])
            ball["body"].set_linear_velocity(ball["velocity"])
    world.step(dt)
    state["t"] = world.sim_time - state["t0"]   # the physics clock, in whole substeps
    if state["t"] >= REBUILD_AT:
        begin_rebuild()


# --------------------------------------------------------------------------- #
#  Camera
# --------------------------------------------------------------------------- #
TARGET = tp.Vector3(-9.0, 8.0, -7.0)
ORBIT_RADIUS, ORBIT_HEIGHT, ORBIT_RATE = 54.0, 7.0, 0.085
ORBIT_START = math.radians(96.0)
camera = tp.PerspectiveCamera(46, W / H, 0.1, 2000.0)


cam = {"push": 0.0, "rise": 0.0}


def place_camera(wall_t, dt_wall=0.0):
    """The film's camera: it orbits, pushes in low for the slow motion and rises for the rebuild."""
    ease = 1.0 - math.exp(-dt_wall / 0.7)
    rebuilding = state["rebuild"] is not None
    cam["push"] += ((0.0 if rebuilding else slowness(state["t"])) - cam["push"]) * ease
    cam["rise"] += ((1.0 if rebuilding else 0.0) - cam["rise"]) * ease
    radius = ORBIT_RADIUS - 8.0 * cam["push"] + 8.0 * cam["rise"]
    a = ORBIT_START + ORBIT_RATE * wall_t
    camera.position.set(TARGET.x + radius * math.cos(a),
                        ORBIT_HEIGHT - 1.0 * cam["push"] + 7.0 * cam["rise"],
                        TARGET.z + radius * math.sin(a))
    camera.look_at(TARGET.x, TARGET.y, TARGET.z)


place_camera(0.0)

# --------------------------------------------------------------------------- #
#  Headless: stills or a film, stepped at a fixed frame rate
# --------------------------------------------------------------------------- #
if HEADLESS:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    encoder = None
    if VIDEO > 0.0:
        os.makedirs(out_dir, exist_ok=True)
        encoder = Encoder(os.path.join(out_dir, "demolition.mp4"), W, H, FPS)
    pending = sorted(SHOT_AT) if SHOT else []
    stem, ext = os.path.splitext(SHOT) if SHOT else ("", "")
    frame, saved, t_step, t_draw = 0, 0, 0.0, 0.0
    while (pending and frame < 60 * FPS) or (encoder is not None and frame < VIDEO * FPS):
        wall_t = frame / FPS
        c0 = time.perf_counter()
        advance(1.0 / FPS)
        c1 = time.perf_counter()
        place_camera(wall_t, 1.0 / FPS)
        renderer.sim_time = wall_t              # pins the temporal passes to the film clock
        renderer.render(scene, camera)
        t_step += c1 - c0
        t_draw += time.perf_counter() - c1
        if encoder is not None:
            encoder.send(renderer.read_pixels())
        if pending and state["t"] >= pending[0]:
            path = SHOT if len(SHOT_AT) == 1 else f"{stem}_{saved}{ext}"
            renderer.save_frame(scene, camera, path)
            print(f"wrote {path} at sim t={state['t']:.2f} s (frame {frame})")
            pending.pop(0)
            saved += 1
        frame += 1
    if encoder is not None:
        encoder.close()
        print("wrote", os.path.join(out_dir, "demolition.mp4"))
    print(f"{N_BRICKS} bricks, {frame} frames: physics {1e3 * t_step / frame:.2f} ms, "
          f"render {1e3 * t_draw / frame:.2f} ms per frame")
    sys.exit(0)

# --------------------------------------------------------------------------- #
#  Live
# --------------------------------------------------------------------------- #
controls = tp.OrbitControls(camera, canvas)
controls.target = tp.Vector3(TARGET.x, TARGET.y, TARGET.z)
controls.enable_damping = True
controls.max_distance = 220.0
ui = tp.ImguiContext(canvas, renderer) if tp.HAS_IMGUI else None
canvas.on_window_resize(resize_handler(camera, renderer))
clock = tp.Clock()
live = {"orbit": True}


def draw_ui():
    tp.imgui.set_next_window_pos(10, 10)
    tp.imgui.set_next_window_size(270, 0)
    tp.imgui.begin("Demolition")
    tp.imgui.text(f"{N_BRICKS} bricks, one draw call")
    tp.imgui.text(f"t = {state['t']:5.2f} s   x{time_scale(state['t']):.2f}")
    if tp.imgui.button("Again"):
        reset()
    _, state["slow"] = tp.imgui.checkbox("slow motion at impact", state["slow"])
    _, live["orbit"] = tp.imgui.checkbox("orbit the camera", live["orbit"])
    tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps")
    tp.imgui.end()


def animate():
    dt = min(clock.get_delta(), 1.0 / 60.0)     # never more than the world's four substeps
    advance(dt)
    if ui is not None:
        controls.enabled = not ui.want_capture_mouse
    if live["orbit"]:
        # Turn the camera about the target; OrbitControls reads the new position.
        x, z = camera.position.x - controls.target.x, camera.position.z - controls.target.z
        c, s = math.cos(ORBIT_RATE * dt), math.sin(ORBIT_RATE * dt)
        camera.position.x = controls.target.x + c * x - s * z
        camera.position.z = controls.target.z + s * x + c * z
    controls.update()
    renderer.render(scene, camera)
    if ui is not None:
        ui.render(draw_ui)


canvas.animate(animate)
