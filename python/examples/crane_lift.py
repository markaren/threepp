"""Offshore crane lift: a knuckle-boom crane on a heaving vessel lands a container on a
wind-turbine platform, with the crane tip held world-stationary against the vessel's
motion (active motion compensation) and the payload swing damped from a tip-mounted
ray-traced range sensor. Every clock is the simulation clock; the vessel rides the
FFT ocean the renderer draws.

    python crane_lift.py                                    # window; drag to orbit, Esc quits
    python crane_lift.py --shot hero --out hero.png --seconds 12
    python crane_lift.py --audit 120 --audit-out a.json      # the sensor-determinism row of THIS scene
    python crane_lift.py --op 60 --op-out op_s0.json --seed 0   # the closed lift (tip sensor -> anti-swing)
    python sensor_audit.py --compare a.json b.json          # exit 0 = bit-identical

Assets (crane, vessel, turbine) are the co-author's Seaonics-derived models and are NOT
in the repository: point CRANE_ASSETS_DIR at the crane twin's assets/ folder.
Vulkan only.
"""
import json
import math
import os
import platform
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import threepp as tp
from demo_common import resize_handler
from warp_common import cli_arg, parse_size, sky_env, standard_material   # noqa: E402 (warp import is harmless)

SHOT = cli_arg("--shot", "", str)
AUDIT = cli_arg("--audit", 0, int)
AUDIT_OUT = cli_arg("--audit-out", "", str)
OP = cli_arg("--op", 0.0, float)                 # the closed lift, seconds
OP_OUT = cli_arg("--op-out", "", str)
OP_FILM = cli_arg("--op-film", "", str)          # every frame of the lift to this mp4
SEED = cli_arg("--seed", 0, int)
NO_ANTISWING = "--no-antiswing" in sys.argv
NO_AMC = "--no-amc" in sys.argv
NO_CLOUDS = "--no-clouds" in sys.argv          # bisection knob for the rendered-frame rows
HEADLESS = bool(SHOT) or AUDIT > 0 or OP > 0
SECONDS = cli_arg("--seconds", 6.0, float)
W, H = parse_size(cli_arg("--size", "1600x900", str))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CAP_DIR = os.path.join(ROOT, "aaa_caps", "crane")
OUT = cli_arg("--out", os.path.join(CAP_DIR, f"{SHOT or 'crane'}.png"), str)
ASSETS = os.environ.get("CRANE_ASSETS_DIR") or cli_arg("--assets", "", str)
if not ASSETS or not os.path.isfile(os.path.join(ASSETS, "crane.glb")):
    sys.exit("set CRANE_ASSETS_DIR (or --assets) to the folder holding crane.glb, Vessel_Fixed.glb, Wind_Turbine.glb")

FPS = 60
DT = 1.0 / FPS
WARMUP = 30

# ---- world ------------------------------------------------------------------
SUN_DIR = np.array([0.78, 0.38, -0.30]); SUN_DIR /= np.linalg.norm(SUN_DIR)
VESSEL_HALF_L, VESSEL_HALF_B = 43.0, 10.0
BUOY_PTS = [(-9.25, 41.35), (9.25, 41.35), (-9.25, -41.35), (9.25, -41.35)]   # port-stern, stbd-stern, port-bow, stbd-bow (vessel x, z)
VESSEL_Y0 = -3.0                       # waterline sits 3 m up the hull: a laden vessel
DECK_Y = 15.6 + VESSEL_Y0              # top of the hull mesh, where the crane pedestal stands
CRANE_ON_VESSEL = (6.0, DECK_Y - 8.0, 25.0)    # the glb's Base sits 8 m up its own frame
TURBINE_POS = (36.0, 0.0, 28.0)
PLATFORM = np.array([TURBINE_POS[0] - 9.04, 20.3, TURBINE_POS[2] - 3.18])   # the landing target on the platform
PICKUP = np.array([-11.6, DECK_Y, 38.2])        # where the container starts: the port quarter, 22 m from the king
HOOK_ABOVE = 2.4                       # hook point above the container centre (glb)
LOAD_H = 3.6 / 2                       # container half height (glb z)

# ---- crane (Seaonics C25 envelope, from the twin) ---------------------------
Q_MIN = np.array([-2.0 * math.pi, -1.379, 0.0]); Q_MAX = np.array([2.0 * math.pi, 0.873, 11.0])   # slew unwrapped; luff and telescope are the C25's
V_MAX = np.array([0.0838, 0.15, 2.5]); A_MAX = np.array([0.0838, 0.15, 2.5]); TAU = 0.1
WIRE_MIN = 0.5

# ---- renderer ---------------------------------------------------------------
canvas = tp.Canvas("threepp - offshore crane lift", width=W, height=H, vsync=False, headless=HEADLESS)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.AgX
renderer.tone_mapping_exposure = 0.80
renderer.render_scale = 1.0
renderer.gbuffer_msaa = 2
renderer.sun_angular_radius = 0.55
renderer.bloom_intensity = 0.08
renderer.bloom_clamp = 12.0
renderer.auto_exposure = True
renderer.set_auto_exposure_range(-1.5, 1.5)
renderer.set_auto_exposure_speed(1.2)
renderer.fog_anisotropy = 0.5

scene = tp.Scene()
sky = sky_env(SUN_DIR, below_horizon=(0.42, 0.50, 0.62), below_nadir=(0.10, 0.14, 0.20))
scene.environment = sky
scene.background = sky
sun = tp.DirectionalLight(0xfff1dc, 2.6)
sun.position.set(*(SUN_DIR * 1000.0))
scene.add(sun)
scene.set_fog_exp2(tp.Color(0.66, 0.72, 0.80), 0.0012)
renderer.set_height_fog(density=0.0, base_y=0.0, falloff=0.02, noise_amount=0.35)
if not NO_CLOUDS:
    renderer.set_clouds(coverage=0.42, density=0.9, bottom_y=420.0, top_y=1100.0,
                        wind=tp.Vector3(7.0, 0.0, 3.0), evolve_speed=1.0)

# ---- ocean -------------------------------------------------------------------
OCEAN_SIZE = 2400.0
ocean = tp.Ocean(size=OCEAN_SIZE, resolution=512, wind_speed=13.0, wind_theta=0.9,
                 choppiness=0.78, wave_scale=2.0, fft_size=512, fetch=8e4)
ocean.params.tile_size_0 = 520.0
ocean.params.tile_size_1 = 52.0
ocean.params.tile_size_2 = 4.1
ocean.params.foam_amount = 0.55
ocean.warp.center_x = 12.0
ocean.warp.center_z = 20.0
ocean.warp.half_range = OCEAN_SIZE * 0.5
ocean.warp.coef_a = 0.06
ocean.material.attenuation_color = tp.Color(0.10, 0.28, 0.34)
ocean.material.attenuation_distance = 4.0
ocean.material.specular_intensity = 0.7
scene.add(ocean)
floor = tp.Mesh(tp.PlaneGeometry(OCEAN_SIZE, OCEAN_SIZE), standard_material(0x03060a))
floor.rotate_x(-math.pi / 2)
floor.position.y = -60.0
scene.add(floor)

# ---- models ------------------------------------------------------------------
loader = tp.ModelLoader()
vessel = loader.load(os.path.join(ASSETS, "Vessel_Fixed.glb"))
vessel.position.set(0.0, VESSEL_Y0, 0.0)
scene.add(vessel)

turbine = loader.load(os.path.join(ASSETS, "Wind_Turbine.glb"))
turbine.position.set(*TURBINE_POS)
scene.add(turbine)
rotor = turbine.get_object_by_name("Rotor Axis")

crane_model = loader.load(os.path.join(ASSETS, "crane.glb"))
crane = tp.Group()                      # the twin's mounting: the glb is Z-up, boom along its -Y
crane.rotation.x = -math.pi / 2
crane.rotation.z = -math.pi
crane.position.set(*CRANE_ON_VESSEL)
crane.add(crane_model)
vessel.add(crane)

king = crane_model.get_object_by_name("King")
hinge = crane_model.get_object_by_name("C1").children[0]      # the inner 'C1' mesh carries the luff
telescope = crane_model.get_object_by_name("Telescope")
outlet = crane_model.get_object_by_name("Outlet")             # the tip
sheave = crane_model.get_object_by_name("BoomSheave")
in_target = crane_model.get_object_by_name("BoomInWireTarget")
out_target = crane_model.get_object_by_name("BoomOutWireTarget")
winch_in = crane_model.get_object_by_name("BoomWinchInlet")
winch_out = crane_model.get_object_by_name("BoomWinchOutlet")
for nm in ("BoomWinchInlet", "BoomWinchOutlet", "OutletVis"):
    g = crane_model.get_object_by_name(nm)
    for ch in g.children:               # the glb's stretched wire cylinders: replaced by our own below
        ch.visible = False
load_glb = crane_model.get_object_by_name("Load")
crane_model.remove(load_glb)

# The payload hangs in the world: pivot at the hook, container 2.4 m below it, facing the boom.
load_pivot = tp.Group()
load_frame = tp.Group()
load_frame.rotation.x = -math.pi / 2
load_frame.rotation.z = -math.pi
load_glb.position.set(0.0, 0.0, -HOOK_ABOVE)
load_glb.rotation.set(0.0, 0.0, 0.0)
load_frame.add(load_glb)
load_pivot.add(load_frame)
scene.add(load_pivot)
load_meshes = []
load_glb.traverse(lambda o: load_meshes.append(o) if isinstance(o, tp.Mesh) else None)
LOAD_ID = 4040
for mesh in load_meshes:
    renderer.set_instance_id(mesh, LOAD_ID)

# Wires: a unit cylinder along +Z, aimed with look_at and stretched with scale.z.
wire_mat = standard_material(0x2a2a2a, roughness=0.5, metalness=0.8)


def make_wire(radius):
    g = tp.Group()
    geo = tp.CylinderGeometry(radius, radius, 1.0, 10, 1)
    geo.rotate_x(math.pi / 2)
    geo.translate(0.0, 0.0, 0.5)
    g.add(tp.Mesh(geo, wire_mat))
    scene.add(g)
    return g


def aim_wire(w, a, b):
    d = b - a
    L = float(np.linalg.norm(d))
    w.position.set(*a)
    w.look_at(*b)
    w.scale.set(1.0, 1.0, max(L, 1e-3))


hoist_wire = make_wire(0.05)
boom_wire_in = make_wire(0.06)
boom_wire_out = make_wire(0.06)

# ---- the tip camera (PIP) and the tip range sensor --------------------------
CAM_W, CAM_H, HUD_M = 480, 270, 16
tip_cam = tp.PerspectiveCamera(70.0, CAM_W / CAM_H, 0.2, 400.0)
scene.add(tip_cam)
TIP_VIEW = 0
FAN_N, FAN_HALF = 28, math.radians(26.0)         # 28x28 rays in a +-26 deg cone straight down from the tip
_g = np.linspace(-FAN_HALF, FAN_HALF, FAN_N)
_ax, _az = np.meshgrid(_g, _g)
FAN_DIRS = np.stack([np.tan(_ax), -np.ones_like(_ax), np.tan(_az)], -1).reshape(-1, 3).astype(np.float32)
FAN_DIRS /= np.linalg.norm(FAN_DIRS, axis=1, keepdims=True)
fan_params = tp.LidarParams()
fan_params.max_range = 60.0
fan_params.detector_threshold = 0.0
FAN_LAST = {}


def wpos(o):
    p = o.get_world_position()
    return np.array([p.x, p.y, p.z], np.float64)


# ---- vessel motion: the hull follows the sea it is drawn on -----------------
ves = {"y": VESSEL_Y0, "pitch": 0.0, "roll": 0.0, "vy": 0.0, "wp": 0.0, "wr": 0.0, "live": False}
ocean.hull_exclusion.set_pose(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, VESSEL_HALF_L, VESSEL_HALF_B)
V_TAU = 1.6                            # the hull's inertia against the wave plane, s


def vessel_step(dt):
    if not ves["live"]:
        return
    h = [float(ocean.sample_height(x, z)) for x, z in BUOY_PTS]
    heave = 0.25 * sum(h) + VESSEL_Y0
    pitch = math.atan2((h[2] + h[3]) * 0.5 - (h[0] + h[1]) * 0.5, 82.7)     # bow up = +
    roll = math.atan2((h[1] + h[3]) * 0.5 - (h[0] + h[2]) * 0.5, 18.5)      # stbd up = +
    k = min(dt / V_TAU, 1.0)
    for key, tgt in (("y", heave), ("pitch", pitch), ("roll", roll)):
        ves[key] += (tgt - ves[key]) * k
    vessel.position.y = ves["y"]
    vessel.rotation.x = ves["pitch"]
    vessel.rotation.z = -ves["roll"]
    ocean.hull_exclusion.set_pose(0.0, 0.0, ves["y"] + 2.0, 0.0, ves["pitch"], ves["roll"])


# ---- crane kinematics --------------------------------------------------------
q = np.array([-math.pi / 2, 0.0, 3.0])          # slew, luff, telescope (realised)
q_vel = np.zeros(3)
q_filt = q.copy()
wire_len = 5.0


def set_joints(qq):
    king.rotation.z = -qq[0]
    hinge.rotation.x = qq[1]
    telescope.position.y = qq[2]


def fk(qq):
    set_joints(qq)
    crane.update_matrix_world(True)
    return wpos(outlet)


def ik(target, q0, iters=6):
    """Damped Gauss-Newton on a finite-difference Jacobian through the scene graph (the twin's solveTipIK)."""
    qq = q0.copy()
    eps, lam = 1e-3, 1e-4
    for _ in range(iters):
        p = fk(qq)
        r = p - target
        if float(np.linalg.norm(r)) < 1e-3:
            break
        J = np.zeros((3, 3))
        for i in range(3):
            dq = qq.copy(); dq[i] += eps
            J[:, i] = (fk(dq) - p) / eps
        step = np.linalg.solve(J.T @ J + lam * np.eye(3), J.T @ r)
        alpha = 1.0
        for _h in range(6):
            cand = np.clip(qq - alpha * step, Q_MIN, Q_MAX)
            if float(np.linalg.norm(fk(cand) - target)) < float(np.linalg.norm(r)):
                qq = cand
                break
            alpha *= 0.5
        else:
            break
    return qq


def actuate(q_cmd, dt):
    """First-order hydraulic lag, then the C25 rate and acceleration limits (the twin's stepJointActuator)."""
    global q, q_vel, q_filt
    q_filt += (q_cmd - q_filt) * min(1.0, dt / TAU)
    v_des = np.clip((q_filt - q) / dt, -V_MAX, V_MAX)
    acc = np.clip((v_des - q_vel) / dt, -A_MAX, A_MAX)
    q_vel = q_vel + acc * dt
    q = np.clip(q + q_vel * dt, Q_MIN, Q_MAX)


# ---- the payload: a spherical pendulum on the hoist wire -----------------------
pend = {"p": None, "v": np.zeros(3)}
PEND_DAMP = 0.08
PEND_SUB = 4


def pendulum_step(tip, dt):
    """Position-based: gravity, then the rod constraint to the tip, velocity from the corrected move."""
    if pend["p"] is None:
        pend["p"] = tip + np.array([0.0, -wire_len, 0.0])
        return pend["p"]
    h = dt / PEND_SUB
    p, v = pend["p"], pend["v"]
    for _ in range(PEND_SUB):
        v = v * (1.0 - PEND_DAMP * h) + np.array([0.0, -9.81, 0.0]) * h
        p_new = p + v * h
        d = p_new - tip
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            p_new = tip + d * (wire_len / n)
        v = (p_new - p) / h
        p = p_new
    pend["p"], pend["v"] = p, v
    return p


# ---- the operation: pickup on deck -> hover -> land on the platform ------------
def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def op_target(t):
    """Nominal tip target in the world and the wire length, as a function of simulation time."""
    hover_up = 3.0
    A = PICKUP + np.array([0.0, wire0 + HOOK_ABOVE + LOAD_H, 0.0])            # tip above the pickup
    B = PLATFORM + np.array([0.0, hover_up + HOOK_ABOVE + LOAD_H, 0.0])       # tip above the platform
    if t < 4.0:
        return A, wire0
    if t < 34.0:                                    # the transfer, lifting first then swinging out
        u = smooth((t - 4.0) / 30.0)
        p = A + (B - A) * u
        p[1] += 2.5 * math.sin(math.pi * u)         # clear the bulwark
        return p, wire0
    if t < 50.0:                                    # pay out until the container hovers 1 m over the grating
        u = smooth((t - 34.0) / 16.0)
        return B, wire0 + (hover_up - 1.0) * u
    return B, wire0 + hover_up - 1.0


wire0 = 10.0
rng = np.random.default_rng(SEED)
FAN_SIGMA = 0.02                       # the tip sensor's seeded range noise, m
AS_KP, AS_KD = 0.55, 0.65             # anti-swing: move the tip after the load
swing_est = {"s": np.zeros(2), "ds": np.zeros(2), "seen": False}
sensor_every = 3                       # 20 Hz
frame_i = 0
sim_t = 0.0
tip_world = np.zeros(3)
load_world = np.zeros(3)
log_rows = []
antiswing_on = [not NO_ANTISWING]


def tip_fan(tip):
    """The tip range sensor: one ray-traced dispatch straight down, the container found as the returns nearer than the deck."""
    origins = np.repeat(tip[None].astype(np.float32), len(FAN_DIRS), 0)
    r = renderer.scan_lidar(origins, FAN_DIRS, fan_params)
    hit = r["return_no"] > 0
    dist = r["distance"].astype(np.float64)
    dist = dist + rng.normal(0.0, FAN_SIGMA, dist.shape) * hit
    FAN_LAST["fan"] = np.concatenate([dist.astype(np.float32)[:, None], r["instance_id"][:, None].astype(np.float32)], 1)
    near = hit & (dist < wire_len + HOOK_ABOVE + 2.0 * LOAD_H + 1.0) & (dist > 0.5)
    if near.sum() < 4:
        return None
    pts = tip[None] + FAN_DIRS[near].astype(np.float64) * dist[near][:, None]
    return pts.mean(0)


def step(dt=DT):
    global frame_i, sim_t, tip_world, load_world, wire_len
    sim_t += dt
    renderer.sim_time = sim_t
    frame_i += 1
    vessel_step(dt)
    if rotor is not None:
        rotor.rotation.z = 0.25 * sim_t
    # the lift
    target, wl = op_target(sim_t) if OP or SHOT or AUDIT or True else (None, wire0)
    wire_len = wl
    cmd = target.copy()
    if antiswing_on[0] and swing_est["seen"]:
        corr = AS_KP * swing_est["s"] + AS_KD * swing_est["ds"]
        n = float(np.linalg.norm(corr))
        if n > 2.0:
            corr *= 2.0 / n
        cmd[0] += corr[0]
        cmd[2] += corr[1]
    if NO_AMC:
        q_cmd = q                                   # joints frozen: the tip rides the vessel
    else:
        q_cmd = ik(cmd, q)
    actuate(q_cmd, dt)
    tip_world = fk(q)
    hook = pendulum_step(tip_world, dt)
    load_world = hook
    load_pivot.position.set(*hook)
    load_pivot.rotation.y = -q[0]
    load_pivot.update_matrix_world(True)
    # wires
    aim_wire(hoist_wire, tip_world, hook)
    aim_wire(boom_wire_in, wpos(winch_in), wpos(in_target))
    aim_wire(boom_wire_out, wpos(winch_out), wpos(out_target))
    # the tip camera looks down the wire
    tip_cam.position.set(*(tip_world + np.array([0.0, -0.3, 0.0])))
    tip_cam.look_at(*(tip_world + np.array([0.0, -20.0, 0.0])))
    # the tip sensor and the swing estimate (20 Hz, its own clock)
    if frame_i % sensor_every == 0 and frame_i > 2:
        c = tip_fan(tip_world)
        if c is not None:
            s = np.array([c[0] - tip_world[0], c[2] - tip_world[2]])
            if swing_est["seen"]:
                swing_est["ds"] = (s - swing_est["s"]) / (sensor_every * dt)
            swing_est["s"] = s
            swing_est["seen"] = True
    log_rows.append(np.array([sim_t, *tip_world, *hook, *q, wire_len, ves["y"], ves["pitch"], ves["roll"],
                              *(target if target is not None else (0, 0, 0)),
                              swing_est["s"][0], swing_est["s"][1]], np.float64))


# ---- cameras ---------------------------------------------------------------
camera = tp.PerspectiveCamera(42.0, W / H, 0.3, 6000.0)
SHOTS = {
    "hero": ((-16.0, 9.0, 66.0), (15.0, 19.0, 26.0)),
    "deck": ((14.0, 24.0, 44.0), (10.0, 20.0, 26.0)),
    "turbine": ((60.0, 26.0, 62.0), (20.0, 22.0, 26.0)),
    "low": ((-8.0, 4.0, 60.0), (16.0, 22.0, 27.0)),
}


def place(cam, shot):
    p, t = SHOTS[shot]
    cam.position.set(*p)
    cam.look_at(*t)


def first_frames():
    """One frame so the ocean exists and the views can be added; then the vessel floats."""
    global TIP_VIEW, q, q_filt
    vessel.update_matrix_world(True)
    A0 = op_target(0.0)[0]                          # start ON the pickup, not in the parked pose: best of four slew seeds
    cands = [ik(A0, np.array([s0, 0.0, 3.0]), iters=14) for s0 in (-math.pi / 2, math.pi / 2, 0.0, math.pi)]
    q = min(cands, key=lambda c: float(np.linalg.norm(fk(c) - A0)))
    print(f"start pose: slew {math.degrees(q[0]):.0f} deg, luff {math.degrees(q[1]):.0f} deg, telescope {q[2]:.1f} m, "
          f"tip error {1e3 * float(np.linalg.norm(fk(q) - A0)):.0f} mm")
    q_filt = q.copy()
    pendulum_step(fk(q), DT)
    renderer.sim_time = 0.0
    renderer.render(scene, camera)
    ves["live"] = True
    TIP_VIEW = renderer.add_view(tip_cam, CAM_W, CAM_H)
    if TIP_VIEW:
        renderer.set_view_display_rect(TIP_VIEW, HUD_M, H - HUD_M - CAM_H, CAM_W, CAM_H)


def op_report():
    L = np.asarray(log_rows)
    if len(L) < 10:
        return
    s = np.hypot(L[:, 4] - L[:, 1], L[:, 6] - L[:, 3])         # true horizontal swing, hook vs tip
    err = np.linalg.norm(L[:, 1:4] - L[:, 14:17], axis=1)      # tip vs its nominal target
    print(f"lift: {L[-1, 0]:.1f} s, swing RMS {1e3 * math.sqrt((s ** 2).mean()):.0f} mm, max {1e3 * s.max():.0f} mm; "
          f"tip-target RMS {1e3 * math.sqrt((err ** 2).mean()):.0f} mm, max {1e3 * err.max():.0f} mm; "
          f"heave span {L[:, 11].max() - L[:, 11].min():.2f} m, roll +-{math.degrees(np.abs(L[:, 13]).max()):.1f} deg")


def run_manifest(n, out, mode):
    import sensor_audit as sa
    renderer.set_flush_frames(1)
    place(camera, "hero")
    renderer.set_auto_exposure_speed(12.0)
    for _ in range(WARMUP):
        step()
        renderer.render(scene, camera)
    renderer.set_auto_exposure_speed(1.2)
    renderer.set_event_camera_params(threshold=0.20, decay=0.88, min_luma=0.005, max_events_per_pixel=5,
                                     frame_time_us=int(sim_t * 1e6))
    renderer.event_camera_source = "final"
    renderer.event_camera_enabled = True
    keys = ["rgb", "aov.depth", "aov.normals", "aov.ids", "aov.motion", "aov.albedo",
            "tip.rgb", "fan", "events.raw", "events.sorted", "traj", "vessel"]
    rows = {k: sa.Fnv() for k in keys}
    n_events = 0
    wall0 = time.perf_counter()
    for f in range(n):
        step()
        renderer.set_event_camera_params(threshold=0.20, decay=0.88, min_luma=0.005, max_events_per_pixel=5,
                                         frame_time_us=int(sim_t * 1e6))
        aovs = renderer.read_aovs_typed(scene, camera, ["rgb", "depth", "normals", "instance_ids", "motion", "albedo"])
        rows["rgb"].update(sa.arr_bytes(aovs["rgb"]))
        rows["aov.depth"].update(sa.arr_bytes(aovs["depth"]))
        rows["aov.normals"].update(sa.arr_bytes(aovs["normals"]))
        rows["aov.ids"].update(sa.arr_bytes(aovs["instance_ids"]))
        rows["aov.motion"].update(sa.arr_bytes(aovs["motion"]))
        rows["aov.albedo"].update(sa.arr_bytes(aovs["albedo"]))
        if TIP_VIEW:
            rows["tip.rgb"].update(sa.arr_bytes(renderer.read_view_rgb_pixels(TIP_VIEW)))
        if "fan" in FAN_LAST:
            rows["fan"].update(sa.arr_bytes(FAN_LAST.pop("fan")))
        ev, _ov = renderer.read_event_stream(max_events=4000000)
        n_events += int(ev.shape[0])
        rows["events.raw"].update(sa.arr_bytes(np.ascontiguousarray(ev)))
        if ev.shape[0]:
            order = np.lexsort((ev[:, 2], ev[:, 0], ev[:, 1], ev[:, 3]))
            rows["events.sorted"].update(sa.arr_bytes(np.ascontiguousarray(ev[order])))
        else:
            rows["events.sorted"].update(b"")
        r = log_rows[-1]
        rows["traj"].update(r[:14].tobytes())
        rows["vessel"].update(r[11:14].tobytes())
    wall = time.perf_counter() - wall0
    manifest = {
        "meta": {
            "threepp": getattr(tp, "__version__", "?"), "platform": platform.platform(),
            "scene": "crane_lift", "mode": mode, "frames": n, "warmup": WARMUP, "fps": FPS, "size": [W, H],
            "seed": SEED, "antiswing": antiswing_on[0], "amc": not NO_AMC, "clouds": not NO_CLOUDS,
            "gpu": next((getattr(renderer, a) for a in ("gpu_name", "device_name") if hasattr(renderer, a)), ""),
            "pins": {"sim_time": True, "auto_exposure": bool(renderer.auto_exposure)},
            "events": n_events, "wall_seconds": round(wall, 2),
        },
        "rows": {k: v.row() if v.frames else "absent" for k, v in rows.items()},
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=1)
    for k, v in manifest["rows"].items():
        print(f"  {k:14s} " + (v if isinstance(v, str) else f"{v['fnv']}  frames={v['frames']}"))
    print(f"{mode}: {n} frames after {WARMUP} warm-up, {1e3 * wall / n:.1f} ms/f, {n_events} events -> {out}")
    op_report()
    if OP_OUT and mode == "op":
        np.savez_compressed(OP_OUT[:-5] + ".npz", log=np.asarray(log_rows))


place(camera, SHOT if SHOT in SHOTS else "hero")
first_frames()

if AUDIT:
    run_manifest(AUDIT, AUDIT_OUT or os.path.join(CAP_DIR, "crane_audit.json"), "audit")
elif OP:
    run_manifest(int(OP * FPS), OP_OUT or os.path.join(CAP_DIR, f"crane_op_s{SEED}.json"), "op")
elif HEADLESS:
    if SHOT not in SHOTS:
        sys.exit(f"unknown shot {SHOT!r}; one of {', '.join(SHOTS)}")
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    frames = int(SECONDS * FPS)
    t0 = time.perf_counter()
    for i in range(frames):
        step()
        renderer.render(scene, camera)
    renderer.save_frame(scene, camera, OUT)
    print(f"simulated {SECONDS:.1f} s ({frames} frames) at {1e3 * (time.perf_counter() - t0) / frames:.1f} ms/f, wrote {OUT}")
    op_report()
else:
    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target.set(*SHOTS["hero"][1])
    canvas.on_window_resize(resize_handler(camera, renderer))

    def animate():
        step()
        controls.update()
        renderer.render(scene, camera)

    canvas.animate(animate)
