"""Offshore crane lift: a knuckle-boom crane on a heaving vessel lands a container on a
wind-turbine platform, with the crane tip held world-stationary against the vessel's
motion (active motion compensation) and the payload swing damped from a tip-mounted
ray-traced range sensor. Every clock is the simulation clock; the vessel rides the
FFT ocean the renderer draws.

    python crane_lift.py                                    # window; drag to orbit, Esc quits
    python crane_lift.py --shot hero --out hero.png --seconds 12
    python crane_lift.py --audit 120 --audit-out a.json      # the sensor-determinism row of THIS scene
    python crane_lift.py --op 76 --op-out op_s0.json --seed 0   # the closed lift (tip sensor -> anti-swing)
    python crane_lift.py --op 76 --op-film lift.mp4             # ... and every second frame to an mp4 at 30 fps
    python crane_lift.py --op 76 --law legacy                   # ... the round-3 PD law instead of the integral one
    python crane_lift.py --op 65 --no-antiswing                 # ... the control run, loop open
    python crane_lift.py --op 76 --no-ff                        # ... with the MRU velocity feedforward off
    python crane_lift.py --op 106 --xfer-profile smooth --xfer 90   # ... the round-5 smoothstep transfer instead
    python crane_lift.py --op 65 --payload pbd                   # ... the round-5 position-based pendulum instead of PhysX
    python crane_lift.py --op 65 --engage always --as-k 0.5      # ... the loop closed through the transfer too
    python crane_lift.py --op 65 --geom old                      # ... the round-5 turbine placement (landing on the deck corner)
    python crane_lift.py --op 65 --load 20ft                     # ... the round-6 12 t 20 ft box instead of the 5 t 10 ft one
    python crane_lift.py --op 76 --pin-exposure                 # ... auto-exposure off, the exposure pinned
    python sensor_audit.py --compare a.json b.json          # exit 0 = bit-identical

Assets (crane, vessel, turbine) are the co-author's Seaonics-derived models and are NOT
in the repository: point CRANE_ASSETS_DIR at the crane twin's assets/ folder.
Vulkan only.
"""
import json
import math
import os
import struct
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
NO_FF = "--no-ff" in sys.argv                    # the MRU velocity feedforward off (position loop only)
FF_MODE = cli_arg("--ff-mode", "both", str)      # 'vessel' = cancel the vessel motion only, 'both' = + the target's own velocity
AMC_KP = cli_arg("--amc-kp", 3.0, float)         # position-loop gain, 1/s; 0 selects the twin's raw v_des = (q_filt - q) / dt
LAW = cli_arg("--law", "integral", str)          # anti-swing law: integral (the phase-1 design) | legacy (the round-3 PD) | off
AS_K = cli_arg("--as-k", 0.8, float)             # the integral law's gain k, 1/s
ENGAGE = cli_arg("--engage", "ondemand", str)    # when the law is closed: ondemand (on arrival, the cited use case) | always
GEOM = cli_arg("--geom", "new", str)             # turbine placement: new (the landing point inside the deck) | old (round 5's corner)
XFER_PROFILE = cli_arg("--xfer-profile", "trapezoid", str)   # the transfer's slew profile: trapezoid (operator) | smooth (round 5)
PAYLOAD = cli_arg("--payload", "physx", str)     # the payload model: physx (rigid body on a rope) | pbd (round 5's pendulum)
PHYSX_SUB = cli_arg("--physx-sub", 1, int)       # PhysX substeps per 60 Hz frame
PHYSX_DAMP = cli_arg("--physx-damp", 0.0, float)  # linear damping on the container, 1/s
LOAD = cli_arg("--load", "10ft", str)            # the payload: 10ft (a 5 t box, round 7) | 20ft (12 t, round 6)
OP_PNG = cli_arg("--op-png", "", str)            # write <prefix>_arrival.png and <prefix>_touchdown.png from the run's own frames
if LOAD not in ("10ft", "20ft"):
    sys.exit("--load is one of 10ft, 20ft")
if ENGAGE not in ("ondemand", "always"):
    sys.exit("--engage is one of ondemand, always")
if GEOM not in ("new", "old"):
    sys.exit("--geom is one of new, old")
if XFER_PROFILE not in ("trapezoid", "smooth"):
    sys.exit("--xfer-profile is one of trapezoid, smooth")
if PAYLOAD not in ("physx", "pbd"):
    sys.exit("--payload is one of physx, pbd")
PIN_EXPOSURE = "--adapt-exposure" not in sys.argv  # auto-exposure OFF by default (E1's rgb row is pinned; one of three lifts differed in rgb with it adapting); --adapt-exposure restores it
EXPOSURE = cli_arg("--exposure", 0.90, float)     # the pinned tone-mapping exposure
SUN = cli_arg("--sun", "", str)                    # "x,y,z" sun direction override for the look pass
NO_GULLS = "--no-gulls" in sys.argv               # the ambient flock off
SEA = cli_arg("--sea", "calm", str)               # calm (the default: a working sea for a crane lift) | fresh (the rounds 1-7 sea, too aggressive)
RAMP_S = cli_arg("--ramp", 1.5, float)            # the operator's joystick ramp, seconds (was 0.8 = the drive's own acceleration limit)
NO_CLOUDS = "--no-clouds" in sys.argv          # bisection knob for the rendered-frame rows
if LAW not in ("integral", "legacy", "off"):
    sys.exit("--law is one of integral, legacy, off")
if LAW == "off":
    NO_ANTISWING = True
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
SUN_DIR = np.array([float(v) for v in SUN.split(",")]) if SUN else np.array([-0.25, 0.50, 0.83])   # behind the hero camera's left shoulder: lights the stern, the crane and the load; the old (0.78, 0.38, -0.30) backlit the scene through the tower
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)
VESSEL_HALF_L, VESSEL_HALF_B = 43.0, 10.0
BUOY_PTS = [(-9.25, 41.35), (9.25, 41.35), (-9.25, -41.35), (9.25, -41.35)]   # port-stern, stbd-stern, port-bow, stbd-bow (vessel x, z)
VESSEL_Y0 = -3.0                       # waterline sits 3 m up the hull: a laden vessel
DECK_Y = 15.6 + VESSEL_Y0              # top of the hull mesh, where the crane pedestal stands
CRANE_ON_VESSEL = (6.0, DECK_Y - 8.0, 25.0)    # the glb's Base sits 8 m up its own frame
KING_XZ = np.array([CRANE_ON_VESSEL[0], CRANE_ON_VESSEL[2]])   # the slew axis in the world at rest
PICKUP = np.array([-6.0, DECK_Y, 36.0])         # where the container starts: the port quarter, 16 m from the king
SLING_H = 2.6                          # hook to the container's top corners
CONT_H = 3.11                          # the container (Cargo_Container_Blue_Fixed.glb), origin at its BOTTOM CENTRE
# The glb is a 20 ft box: measured, x +-1.425, y 0..3.115, z +-3.495, so its LONG axis is its own z.
# Round 7 flies a 10 ft box instead: the same corrugated shell with its long axis SCALED BY HALF
# (CONT_SCALE_Z on the visual, the same half-length on the PhysX proxy and on the sling feet), and
# 5 t instead of 12 - the density follows from the proxy's volume, so the box stays homogeneous.
# The reason is the platform: the clear arm of the turbine deck is 5.88 m wide between railing and
# tower, and a 7.0 m box only fits ACROSS it with 1.09 m to the tower. A 3.5 m box fits with room.
CONT_X = 1.43                          # half width (across the boom at the landing), both sizes
CONT_Z = 3.50 if LOAD == "20ft" else 1.75      # half length (along the boom's tangent at the landing)
CONT_SCALE_Z = 1.0 if LOAD == "20ft" else 0.5  # the visual glb's own long axis
LOAD_DROP = SLING_H + CONT_H           # hook to container bottom
CONT_MASS = 12000.0 if LOAD == "20ft" else 5000.0     # a laden 20 ft box / a laden 10 ft box, kg
CONT_DENSITY = CONT_MASS / (2 * CONT_X * CONT_H * 2 * CONT_Z)   # 193 / 161 kg/m^3 over the box proxy
ANCHOR_UP = 0.5 * CONT_H + SLING_H     # the container's centre to the hook point above it

# ---- where the turbine stands, and where on its deck the container goes ------
# Measured off Wind_Turbine.glb (every mesh's world bounds, the turbine at its own origin):
# the RAILING encloses x -8.96..+4.69, z -4.72..+5.62 about the turbine origin, the grating's
# walking surface is at y = 20.50 (plating 20.30, railing top 21.58), and the tower is a
# cylinder of radius 3.08 m standing ON that deck, ON the turbine's own axis. So the deck is
# 13.65 x 10.34 m with a 6.2 m tower in it, and only the -x arm is wide: 8.96 m from the near
# railing to the axis, 5.88 m of clear deck between the railing and the tower SURFACE.
#
# Round 5 landed on the glb's `CenterPoint (Should be at Landing Target)` node, which sits on
# the deck's south-west CORNER (offset -9.04, -3.18): the container came to rest half over the
# edge. The new placement moves the TURBINE instead - the crane cannot reach further in - so a
# landing point 18.50 m from the king lies on the clear -x arm with the container's 7.0 m length
# lying TANGENTIALLY (across the boom) and both ends on the deck.
#
# The plan asked for the landing point 4.5 m inside the near railing. That is not reachable
# here: 4.5 m in from -8.96 is local x = -4.46, and the container's 1.43 m half-width then puts
# its long face at 3.03 m from the tower axis, INSIDE the 3.08 m tower. The clear strip is only
# 5.88 m wide, so the centre must sit in local x -7.53..-4.51 for the box to fit between railing
# and tower at all. -5.60 is the compromise taken (round 6, the 20 ft box): 3.36 m inside the near
# railing, 1.09 m from its far face to the tower surface - and 1.09 m is less than the 1.07 m of
# arrival swing the operator's transfer leaves, which is why the 20 ft box STRUCK the tower.
#
# Round 7 asks for real clearances - 1.5 m to the tower surface, 1.0 m to every railing, no
# overlap with the three door meshes - and the 10 ft box is what buys them. The clear strip is
# 8.96 (railing to axis) - 1.0 (railing margin) - 3.08 (tower) - 1.5 (tower margin) = 3.38 m of
# allowed span for the box's 2.86 m width, so the landing tolerance ACROSS the strip is +-0.26 m
# and the centre must sit in local x -6.53..-6.01. Along the strip there is room, and the landing
# point is offset in +z of the turbine axis so the box sets down BESIDE the tower rather than on
# the axis: local (-6.25, +2.00), 17.92 m from the king (round 6: -5.60, +0.45 and 18.50 m). The
# turbine itself does NOT move - the clearances hold where it stands.
TOWER_R = 3.08                         # the tower's radius at deck height, measured from the glb (it tapers to 2.65 at y = 44)
DECK_LOCAL = (-8.96, 4.69, -4.72, 5.62)   # the railing rectangle about the turbine axis: x0, x1, z0, z1
RAIL_MARGIN, TOWER_MARGIN = 1.0, 1.5   # what round 7 demands of the landing: metres to a railing / to the tower surface
DOOR_MESHES = ("Door (1)", "Door (2)", "Door (3)")   # the three 1.09 m doors standing on the walking surface
if GEOM == "old":
    TURBINE_POS = (34.0, 0.0, 27.0)
    LAND_OFF = np.array([-9.04, 20.3, -3.18])       # the glb's CenterPoint node, on the deck corner
    LOAD_YAW = None                                 # round 5: the load turns with the king
else:
    TURBINE_POS = (30.10, 0.0, 24.55)
    LAND_OFF = np.array([-5.60, 20.50, 0.45]) if LOAD == "20ft" else np.array([-6.25, 20.50, 2.00])
    LOAD_YAW = 0.0                                  # the container's long axis along world z = across the boom
PLATFORM = np.array([TURBINE_POS[0] + LAND_OFF[0], LAND_OFF[1], TURBINE_POS[2] + LAND_OFF[2]])
TOWER_XZ = np.array([TURBINE_POS[0], TURBINE_POS[2]])
DECK_RECT = (TURBINE_POS[0] + DECK_LOCAL[0], TURBINE_POS[0] + DECK_LOCAL[1],
             TURBINE_POS[2] + DECK_LOCAL[2], TURBINE_POS[2] + DECK_LOCAL[3])
# The CLEAR RECTANGLE: where the container's whole footprint has to lie for the round-7 clearances
# to hold. Bounded by the three railings it can reach (inset RAIL_MARGIN) and, on the fourth side,
# by the tower's bounding circle (inset TOWER_MARGIN). It is a rectangle rather than the exact
# swept region because the exact one is a rectangle minus a disc and the box never gets near the
# curved part: the tower is 6.25 m away across the strip and only 2.0 m along it.
CLEAR_RECT = (DECK_RECT[0] + RAIL_MARGIN, TOWER_XZ[0] - TOWER_R - TOWER_MARGIN,
              DECK_RECT[2] + RAIL_MARGIN, DECK_RECT[3] - RAIL_MARGIN)
START_CLEAR = 0.3                      # the container hangs this far over the vessel deck at t = 0 (no deck collider)

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
# The temporal resolve is CHOSEN here, before the first frame, not read off afterwards: the
# paper's pinned rendered-frame row is the in-house TAA, and the upscaler in force is decided
# at the first frame (sensor_audit.py --resolve taa does exactly this).
for _name in ("fsr", "dlss"):
    if hasattr(renderer, _name):
        try:
            setattr(renderer, _name, False)
        except Exception as _e:                    # pragma: no cover - no binding on this build
            print(f"could not clear {_name}: {_e}")
PINNED_EXPOSURE = EXPOSURE
if PIN_EXPOSURE:                                   # the second pin of sensor_audit.py: nothing that adapts to image statistics
    renderer.auto_exposure = False
    renderer.tone_mapping_exposure = PINNED_EXPOSURE

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
SEA_P = {"calm": dict(wind_speed=9.0, choppiness=0.70, wave_scale=1.3, foam=0.40),
         "fresh": dict(wind_speed=13.5, choppiness=0.78, wave_scale=2.3, foam=0.55)}[SEA]
ocean = tp.Ocean(size=OCEAN_SIZE, resolution=512, wind_speed=SEA_P["wind_speed"], wind_theta=0.9,
                 choppiness=SEA_P["choppiness"], wave_scale=SEA_P["wave_scale"], fft_size=512, fetch=8e4)
ocean.params.tile_size_0 = 520.0
ocean.params.tile_size_1 = 52.0
ocean.params.tile_size_2 = 4.1
ocean.params.foam_amount = SEA_P["foam"]
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

# The ambient flock (extras/fauna, as the sailboat and netpen films use it): seeded, stepped on the sim
# clock in step(), so it replays; it lives off the stern quarter over the water, under the tip, and the
# fan is gated on the container's id so a bird can never enter the swing estimate.
gulls = None
if not NO_GULLS:
    gp = tp.FlockParams()
    gp.seed, gp.bird_count, gp.perching, gp.birds_cast_shadow = 4711, 18, False, False
    gp.home, gp.roam_radius, gp.cruise_altitude, gp.altitude_spread = tp.Vector3(-26.0, 15.0, 22.0), 24.0, 15.0, 0.4   # abeam to port, mid-distance, left of the crane in the hero frame
    gp.cruise_speed, gp.max_speed, gp.mass_kg = 9.5, 17.0, 0.95
    gp.shape.body_length, gp.shape.body_radius, gp.shape.wing_span, gp.shape.tail_fork = 0.58, 0.075, 1.42, 0.15
    gp.plumage.back, gp.plumage.belly = tp.Color(0.48, 0.52, 0.57), tp.Color(0.88, 0.88, 0.86)
    gp.plumage.cap, gp.plumage.leg, gp.plumage.wingtip_dark = tp.Color(0.80, 0.78, 0.72), tp.Color(0.85, 0.62, 0.22), 0.22
    gp.w_cohesion, gp.w_alignment, gp.loner_fraction = 0.30, 0.45, 0.30
    gulls = tp.Flock(gp)
    scene.add(gulls)
turbine.update_matrix_world(True)
# The three doors, measured where they stand rather than assumed: each is a 1.09 m leaf hinged
# open ON the walking surface at the railing line (measured in the turbine's own frame:
# Door (1) x 4.55..4.70, z -0.41..0.76; Door (2) x -0.89..0.28, z 5.27..5.39; Door (3)
# x -9.00..-7.83, z -2.72..-2.60). Round 6's container came to rest 0.49 m high on this furniture,
# so round 7 (a) keeps the landing footprint clear of all three and (b) takes them OUT of the
# static collider set - see build_physx - so nothing can perch on a door even if it lands wide.
DOOR_BOX = {}
for _nm in DOOR_MESHES:
    _d = turbine.get_object_by_name(_nm)
    if _d is None:
        continue
    _b = tp.Box3()
    _b.set_from_object(_d)
    _lo, _hi = _b.min(), _b.max()                # Box3.min / .max are methods on this binding
    DOOR_BOX[_nm] = (_lo.x, _hi.x, _lo.z, _hi.z, _lo.y, _hi.y)

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
crane_model.remove(load_glb)                    # the glb's grey cube: replaced by a 20 ft container on four slings

# The payload hangs in the world: pivot at the hook, the container SLING_H below it, facing the boom.
load_pivot = tp.Group()
container = loader.load(os.path.join(ASSETS, "Cargo_Container_Blue_Fixed.glb"))
for nm in ("LOD1", "LOD2"):
    lod = container.get_object_by_name(nm)
    if lod is not None:
        lod.visible = False
container.scale.set(1.0, 1.0, CONT_SCALE_Z)     # a 10 ft box out of the 20 ft glb: the corrugation stays, the length halves
container.position.set(0.0, -LOAD_DROP, 0.0)
load_pivot.add(container)
hook_mesh = tp.Mesh(tp.SphereGeometry(0.22, 12, 8), standard_material(0x202020, roughness=0.4, metalness=0.9))
load_pivot.add(hook_mesh)
scene.add(load_pivot)
load_meshes = []
container.traverse(lambda o: load_meshes.append(o) if isinstance(o, tp.Mesh) else None)
LOAD_ID = 4040
for mesh in load_meshes:
    renderer.set_instance_id(mesh, LOAD_ID)
    mesh.material.color = 0x2b57a6            # the asset's colour map does not come through the loader; a blue box
SLING_CORNERS = [np.array([sx * CONT_X, -SLING_H, sz * CONT_Z]) for sx in (-1, 1) for sz in (-1, 1)]
if PAYLOAD == "physx":
    # The container is no longer rigidly under the hook: it is a rigid body with its own pose,
    # so it hangs in the WORLD and the hook group keeps only the hook sphere and the slings.
    load_pivot.remove(container)
    scene.add(container)

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
slings = [make_wire(0.025) for _ in SLING_CORNERS]
boom_wire_in = make_wire(0.06)
boom_wire_out = make_wire(0.06)

# ---- the tip camera (PIP) and the tip range sensor --------------------------
CAM_W, CAM_H, HUD_M = 480, 270, 16
tip_cam = tp.PerspectiveCamera(70.0, CAM_W / CAM_H, 0.2, 400.0)
scene.add(tip_cam)
TIP_VIEW = 0
# 32x32 rays in a +-34 deg cone straight down from the tip. The half-angle is set by the estimator,
# not by the picture: the beams are a uniform grid in (tan ax, tan az), so on the container's
# horizontal top face they land on a uniform grid in world x, z and their mean is an UNBIASED
# estimate of the face's centroid - but only while the whole face is inside the footprint. At the
# container's 8.6 m below the head, +-26 deg reaches +-4.2 m against the face's 3.5 m half-length
# plus the 0.45 m head offset, so the far end fell outside and the centroid was pulled 0.1 to 0.2 m
# toward the head (measured: gain 2.7 against a 0.05 m true swing). +-34 deg reaches +-5.8 m, which
# holds the whole face plus about 1.3 m of swing.
#
# Round 6 widens it to +-42 deg. An operator-shaped transfer leaves METRES of residual swing at
# arrival, not the decimetres a smoothstep leaves, and 1.3 m of coverage is where the estimate
# would stop being unbiased exactly when the controller needs it. +-42 deg reaches +-6.9 m at the
# hover height, 3.0 m of swing past the face's far edge, and more as the load is paid out.
#
# Round 7 halves the load's length, so the half-angle comes back down: the estimator needs the
# WHOLE top face inside the footprint, and the face's far edge is now CONT_Z + SENSOR_OUT = 2.20 m
# from the head instead of 3.95 m. At the hover the face is 7.7 m under the head, so +-34 deg
# reaches 5.19 m: the same 3.0 m of swing margin past the face's edge that +-42 deg bought the
# 20 ft box, with the grid 1.4x denser on a face half the size (measured: ~100 returns a scan).
FAN_N, FAN_HALF = 32, math.radians(42.0 if LOAD == "20ft" else 34.0)
_g = np.linspace(-FAN_HALF, FAN_HALF, FAN_N)
_ax, _az = np.meshgrid(_g, _g)
FAN_DIRS = np.stack([np.tan(_ax), -np.ones_like(_ax), np.tan(_az)], -1).reshape(-1, 3).astype(np.float32)
FAN_DIRS /= np.linalg.norm(FAN_DIRS, axis=1, keepdims=True)
fan_params = tp.LidarParams()
fan_params.max_range = 60.0
fan_params.detector_threshold = 0.0
FAN_LAST = {}
SENSOR_DROP = np.array([0.0, -0.9, 0.0])      # the sensor head hangs under the boom tip, clear of its mesh
# ... and OUTBOARD of the hoist wire. On the wire's axis (rounds 1-4) the head sits INSIDE the
# 0.05 m wire cylinder: every ray leaves through the cylinder wall, so the fan measured the wire
# and nothing else. Rays near the vertical ran 0.5 to 6 m inside it before exiting, which is what
# passed the old range gate and became the "container" centroid - a point on the wire, at half
# the swing, which is the 0.84 / 0.53 gain of rounds 3-4. 0.45 m outboard clears it.
SENSOR_OUT = 0.45
fan_seen = [0, 0, 0]                           # (scans, scans that found the container, returns on the container)


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
    # The MRU surrogate: this motion is a first-order lag of the sampled sea, so the
    # rate is analytic, d(state)/dt = (sample - state) / V_TAU at the instant the step
    # begins -- exactly the increment the step then applies (k = dt / V_TAU). The twin's
    # header is explicit that a backward difference is NOT equivalent (half-sample lag).
    rate = {key: (tgt - ves[key]) / V_TAU for key, tgt in (("y", heave), ("pitch", pitch), ("roll", roll))}
    ves["vy"], ves["wp"], ves["wr"] = rate["y"], rate["pitch"], rate["roll"]
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


IK_EPS = 1e-3


def jacobian(qq, p=None):
    """d(tip world) / dq by forward differences through the scene graph, the vessel pose held.
    Shared by the IK and the MRU feedforward so both linearise about the same point."""
    if p is None:
        p = fk(qq)
    J = np.zeros((3, 3))
    for i in range(3):
        dq = qq.copy(); dq[i] += IK_EPS
        J[:, i] = (fk(dq) - p) / IK_EPS
    return J, p


def ik(target, q0, iters=6):
    """Damped Gauss-Newton on a finite-difference Jacobian through the scene graph (the twin's solveTipIK)."""
    qq = q0.copy()
    lam = 1e-4
    for _ in range(iters):
        p = fk(qq)
        r = p - target
        if float(np.linalg.norm(r)) < 1e-3:
            break
        J, _ = jacobian(qq, p)
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


def mru_feedforward(q_cmd, v_target):
    """The twin's MRU velocity feedforward (AMCController: qDotFF through the inverse Jacobian).

    With the joints frozen the tip rides the vessel at
        v_vessel = (0, vy, 0) + omega x (tip - vessel origin),   omega ~ (pitch', 0, -roll')
    (the hull rotates about its own origin at (0, ves.y, 0); rotation.x = pitch, rotation.z = -roll).
    The tip's world velocity is v_vessel + J q', so holding it on a target that itself moves at
    v_target needs q'_ff = J^-1 (v_target - v_vessel). The twin uses v_target = 0 (`--ff-mode vessel`);
    'both' also feeds the transfer forward."""
    J, p = jacobian(q_cmd)
    om = np.array([ves["wp"], 0.0, -ves["wr"]])
    v_ves = np.array([0.0, ves["vy"], 0.0]) + np.cross(om, p - np.array([0.0, ves["y"], 0.0]))
    b = (v_target - v_ves) if FF_MODE == "both" else -v_ves
    try:
        return np.linalg.solve(J, b)
    except np.linalg.LinAlgError:                   # a singular pose: no feedforward, the position loop carries it
        return np.zeros(3)


def actuate(q_cmd, dt, q_dot_ff=None):
    """First-order drive lag (the crane is electric), then the C25 rate and acceleration limits (the twin's stepJointActuator).
    The MRU feedforward is ADDED to the position-loop velocity before the rate limiter, as the twin does.

    Records which limit BOUND this step: the rate limit is the one that costs the anti-swing law its
    authority (the tip cannot follow), and it is what sets the law's anti-windup flag; the
    acceleration limit is counted separately because at 60 Hz a position loop clips it routinely."""
    global q, q_vel, q_filt
    q_filt += (q_cmd - q_filt) * min(1.0, dt / TAU)
    # The twin's position loop is v_des = (q_filt - q) / dt, a gain of 60 /s at 60 Hz. Against the
    # slew's 0.0838 rad/s^2 acceleration limit that demand is unreachable by three orders of
    # magnitude, so the slew runs bang-bang at its acceleration limit and the feedforward is lost
    # in the clip (measured: 97% of the hold frames at the limit). --amc-kp sets a finite gain
    # instead; --amc-kp 0 is the twin's behaviour, bit for bit.
    v_pos = (q_filt - q) / dt if AMC_KP <= 0.0 else AMC_KP * (q_filt - q)
    v_raw = v_pos if q_dot_ff is None else v_pos + q_dot_ff
    v_des = np.clip(v_raw, -V_MAX, V_MAX)
    a_raw = (v_des - q_vel) / dt
    acc = np.clip(a_raw, -A_MAX, A_MAX)
    q_vel = q_vel + acc * dt
    q = np.clip(q + q_vel * dt, Q_MIN, Q_MAX)
    rate_hit = bool(np.any(np.abs(v_raw) > V_MAX * (1.0 + 1e-9)))
    acc_hit = bool(np.any(np.abs(a_raw) > A_MAX * (1.0 + 1e-9)))
    rate_sat_frames[0] += rate_hit
    acc_sat_frames[0] += acc_hit
    as_state["sat"] = rate_hit or (acc_hit and SAT_FLAG_INCLUDES_ACC)


# ---- the payload: a spherical pendulum on the hoist wire (--payload pbd) --------
pend = {"p": None, "v": np.zeros(3)}
PEND_DAMP = 0.08
PEND_SUB = 4

# The grating the container is set down on: y = 20.50 (the glb's `Grating` walking surface) with
# the new placement, 20.30 (the plating, the CenterPoint node's own height) with `--geom old`.
# The footprint the PBD path accepts contact on is the measured RAILING rectangle inset by half
# a metre, except under `--geom old`, where the landing point is 0.2 m inside the deck's x edge
# and any honest rectangle would reject the landing point itself; there the round-5 4.5 m radius
# about PLATFORM is kept so the two geometries stay comparable runs of the same script.
GRATING_Y = PLATFORM[1]                # the container's bottom rests here
FOOT_R = 4.5                           # `--geom old`: the platform's footprint about PLATFORM's xz, m
HOOK_FLOOR = GRATING_Y + LOAD_DROP     # the hook can go no lower over the grating
LAND_HOLD, LAND_SLACK = 0.5, 0.3       # seconds in contact and metres of wire slack that call it landed
contact = {"on": False, "t0": None, "landed": False, "t_land": None, "xz": None, "slack": 0.0,
           "on_deck": None}


def on_deck(x, z, inset=0.5):
    """Is (x, z) over the platform's walking surface? The measured railing rectangle, inset."""
    if GEOM == "old":
        return bool(math.hypot(x - PLATFORM[0], z - PLATFORM[2]) < FOOT_R)
    return bool(DECK_RECT[0] + inset < x < DECK_RECT[1] - inset
                and DECK_RECT[2] + inset < z < DECK_RECT[3] - inset)


def pendulum_step(tip, dt):
    """Position-based: gravity, then the wire as a DISTANCE constraint (it can go slack, never push),
    then the grating: the container's bottom may not pass through it. Velocity from the corrected move.

    Contact is inelastic and sticking - the container is a 20 ft box of steel set down on a steel
    grating, it does not bounce and it does not slide - so the whole velocity is zeroed while it
    rests. The wire is unchanged by the contact: it may go slack, it never pushes."""
    if pend["p"] is None:
        pend["p"] = tip + np.array([0.0, -wire_len, 0.0])
        return pend["p"]
    h = dt / PEND_SUB
    p, v = pend["p"], pend["v"]
    on = False
    for _ in range(PEND_SUB):
        v = v * (1.0 - PEND_DAMP * h) + np.array([0.0, -9.81, 0.0]) * h
        p_new = p + v * h
        d = p_new - tip
        n = float(np.linalg.norm(d))
        if n > wire_len:                          # taut: pull the hook back onto the sphere; slack: fall free
            p_new = tip + d * (wire_len / n)
        v = (p_new - p) / h
        p = p_new
        if p[1] < HOOK_FLOOR and on_deck(p[0], p[2]):
            p = np.array([p[0], HOOK_FLOOR, p[2]])
            v = np.zeros(3)
            on = True
    pend["p"], pend["v"] = p, v
    update_contact(on, wire_len - float(np.linalg.norm(p - tip)), np.array([p[0], p[2]]))
    return p


def update_contact(on, slack, centre_xz):
    """The touchdown state machine, shared by both payload models.

    `on` is the contact flag (the grating test in PBD, the ContactSensor's latched state in
    PhysX), `slack` the metres the tether has gone slack by (PhysX: the tether's max distance
    minus the anchor separation, which is the tension going to zero), `centre_xz` the
    container's centre in the world. Landed = LAND_HOLD seconds in contact with the wire slack
    by more than LAND_SLACK: set down AND let go, not merely touching."""
    contact["on"], contact["slack"] = bool(on), float(slack)
    if on:
        if contact["t0"] is None:
            contact["t0"] = sim_t
        if (not contact["landed"]) and sim_t - contact["t0"] >= LAND_HOLD and slack > LAND_SLACK:
            contact["landed"] = True              # set down and the wire gone slack: the operator lets go
            contact["t_land"] = sim_t
            contact["xz"] = np.array([centre_xz[0] - PLATFORM[0], centre_xz[1] - PLATFORM[2]])
            contact["on_deck"] = on_deck(centre_xz[0], centre_xz[1], inset=0.0)
    else:
        contact["t0"] = None


# ---- the operation: pickup on deck -> hover -> land on the platform ------------
def smooth(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


wire0 = 6.0
HOVER_UP = 2.0                                     # the container's bottom over the grating during the transfer
# The two ends of the operation. A is above the pickup on the port quarter, with the container
# hanging START_CLEAR over the deck (the hold phase is the crane HOLDING it in the air, since
# there is no deck collider). B carries the WIRE as well as the sling drop - with the tip target
# only 2 m + LOAD_DROP over the platform (rounds 1-4) the hook hangs a wire length lower still
# and the container finished 5.5 m UNDER the grating, inside the turbine base. wire0 is 6.0 m
# rather than 7.0 so B stays inside the boom's reach with the vessel moving.
OP_A = PICKUP + np.array([0.0, START_CLEAR + wire0 + LOAD_DROP, 0.0])
OP_B = PLATFORM + np.array([0.0, HOVER_UP + LOAD_DROP + wire0, 0.0])
_RA, _RB = OP_A[[0, 2]] - KING_XZ, OP_B[[0, 2]] - KING_XZ
TH_A, TH_B = math.atan2(_RA[1], _RA[0]), math.atan2(_RB[1], _RB[0])
D_TH = (TH_B - TH_A + math.pi) % (2.0 * math.pi) - math.pi      # the slew sweep round the king, rad
R_A, R_B = float(np.linalg.norm(_RA)), float(np.linalg.norm(_RB))

# ---- the transfer: what a joystick operator actually does ---------------------
# Round 5's transfer was a 60 s smoothstep in the slew angle, chosen so its peak rate (0.0616
# rad/s) stayed under the C25's 0.0838 - a machine-planned profile whose acceleration is a
# smooth bump and which therefore leaves almost no swing behind. An operator does not do that:
# he pushes the joystick, the drive ramps at its acceleration limit to a comfortable rate, holds
# it, and ramps down. The residual swing at arrival is a step response of the pendulum to that
# final deceleration, and damping it is the whole job of the on-demand controller.
#
# So `trapezoid` is: ramp at A_MAX[0] = 0.0838 rad/s^2 to 0.8 x V_MAX[0] = 0.0670 rad/s, cruise,
# ramp down. The duration follows from the sweep rather than being chosen. Luff and telescope
# ride smoothsteps over the same interval (the radial and vertical parts of the tip target), so
# only the slew is operator-shaped; the bulwark lift is kept.
XFER_A = float(A_MAX[0])
XFER_V = 0.8 * float(V_MAX[0])
XFER_A = min(XFER_A, XFER_V / max(RAMP_S, 1e-3))   # --ramp: a gentler operator stop than the drive limit (1.5 s -> about 0.045 rad/s^2)
T_HOLD, T_ARRIVE, T_PAYOUT = 4.0, 8.0, 12.0
_t_ramp = XFER_V / XFER_A
_d_ramp = 0.5 * XFER_A * _t_ramp * _t_ramp
if 2.0 * _d_ramp >= abs(D_TH):                     # too short to reach the cruise rate: a triangle
    _t_ramp = math.sqrt(abs(D_TH) / XFER_A)
    _t_cruise = 0.0
    XFER_V = XFER_A * _t_ramp
    _d_ramp = 0.5 * abs(D_TH)
else:
    _t_cruise = (abs(D_TH) - 2.0 * _d_ramp) / XFER_V
T_XFER = (2.0 * _t_ramp + _t_cruise) if XFER_PROFILE == "trapezoid" else cli_arg("--xfer", 60.0, float)
T1, T2, T3, T4 = T_HOLD, T_HOLD + T_XFER, T_HOLD + T_XFER + T_ARRIVE, T_HOLD + T_XFER + T_ARRIVE + T_PAYOUT


def slew_frac(t):
    """Fraction of the sweep completed at time t into the transfer, 0 to 1."""
    if XFER_PROFILE != "trapezoid":
        return smooth(t / T_XFER)
    t = min(max(t, 0.0), T_XFER)
    if t < _t_ramp:
        d = 0.5 * XFER_A * t * t
    elif t < _t_ramp + _t_cruise:
        d = _d_ramp + XFER_V * (t - _t_ramp)
    else:
        td = T_XFER - t
        d = abs(D_TH) - 0.5 * XFER_A * td * td
    return d / abs(D_TH)


def op_target(t):
    """Nominal tip target in the world and the wire length, as a function of simulation time.

    Five phases: T_HOLD s holding the load in the air over the pickup, the transfer (whose
    length T_XFER FOLLOWS from the profile), T_ARRIVE s of arrival hold at the hover point -
    which is where the on-demand loop engages and has to earn its keep - then T_PAYOUT s of
    pay-out to touchdown, then hold. The pay-out starts at a FIXED time for every variant, so
    open loop and closed loop are compared at equal times.

    At HOVER_UP = 2 m the container clears the platform railing, whose top is 1.08 m over the
    grating, by 0.92 m on the way in."""
    if t < T1:
        return OP_A, wire0
    if t < T2:                                      # the transfer: a slew ROUND the king at the boom's reach, not a line over it
        s = (t - T1) / T_XFER                       # luff and telescope: smoothstep on the radius and the height
        u = smooth(s)
        th = TH_A + D_TH * slew_frac(t - T1)        # the slew: the operator's trapezoid (or round 5's smoothstep)
        r = R_A * (1.0 - u) + R_B * u
        p = np.array([KING_XZ[0] + r * math.cos(th), OP_A[1] * (1.0 - u) + OP_B[1] * u, KING_XZ[1] + r * math.sin(th)])
        p[1] += 1.0 * math.sin(math.pi * u)         # clear the bulwark
        return p, wire0
    if t < T3:                                      # the arrival hold: the tip stationary over the platform
        return OP_B, wire0
    if t < T4:                                      # pay out to the grating: the container lands and the wire goes slack
        return OP_B, wire0 + (HOVER_UP + 0.5) * smooth((t - T3) / T_PAYOUT)
    return OP_B, wire0 + HOVER_UP + 0.5


rng = np.random.default_rng(SEED)
FAN_SIGMA = 0.02                       # the tip sensor's seeded range noise, m
IMU_HZ = 0.0                           # the hull IMU (motion reference unit): 0 = every physics substep, 60 Hz at one substep, as the audit harness samples
AS_KP, AS_KD = 0.30, 0.40             # the LEGACY law (round 3): move the tip after the load. Kept for the record.
AS_CLAMP, AS_CUTOUT = 1.2, 3.0         # correction clamp, and the swing beyond which the loop opens (runaway guard)
sat_frames = [0]                       # frames with a joint at a POSITION limit
rate_sat_frames = [0]                  # frames with a drive at its RATE limit
acc_sat_frames = [0]                   # frames with a drive at its ACCELERATION limit
sat_joint = np.zeros(6, int)           # per joint: at min, at max


# ---- the anti-swing law (from crane_lift_runs/swing_surrogate.py, phase 1) ----
AS_GAMMA, AS_UMAX = 0.4, 1.5           # leak [1/s], correction clamp [m]; the gain k is --as-k
AS_CUTOUT_M = 3.0                      # runaway guard, kept from the legacy law
# The rate limit is what costs the law its authority; the acceleration limiter, at 60 Hz with a
# position loop, clips on nearly every frame and would freeze the integrator for the whole run
# (measured in the control run: see the round-5 table), so it does NOT set the anti-windup flag.
SAT_FLAG_INCLUDES_ACC = False


def antiswing_state():
    """The controller's whole memory: one integrator, one flag."""
    return {"u": np.zeros(2), "sat": False, "cutout": 0}


def antiswing_update(s_est, have_est, dt_sensor, state):
    """u' = k s - gamma u, integrated once per sensor update and HELD between updates.

    s_est          the tip sensor's horizontal swing estimate (hook - tip) in world x, z [m]
    have_est       False when the scan found no load: the correction is held, not zeroed
    dt_sensor      seconds since the previous estimate (0.05 s at 20 Hz)
    state          the dict from antiswing_state(); state["sat"] is set by actuate() on any
                   frame a drive is at its RATE limit

    Returns the world (x, z) correction to add to the tip target.

    Why an integrator and not the legacy PD: through a position servo the pivot velocity,
    not the pivot position, sets the swing damping.  s'' = -omega^2 s - k s' - p_target'',
    so a positive k is damping and any positive KD on s' is ANTI-damping (see plans/crane-lift-e3.md).
    The leak gamma bounds the correction and unwinds it after the transfer; the conditional
    integration keeps the integrator from winding up against a rate-saturated slew drive.
    """
    u = state["u"]
    if not have_est:
        return u.copy()                     # sensor dropout: hold the last correction
    s = np.asarray(s_est, np.float64)
    if float(np.linalg.norm(s)) > AS_CUTOUT_M:      # runaway guard: bleed out, do not jump to zero
        state["cutout"] += 1
        state["u"] = u * max(0.0, 1.0 - 2.0 * dt_sensor)
        return state["u"].copy()
    u_new = u + (AS_K * s - AS_GAMMA * u) * dt_sensor
    n = float(np.linalg.norm(u_new))
    if n > AS_UMAX:
        u_new = u_new * (AS_UMAX / n)
    if not (state["sat"] and n > float(np.linalg.norm(u))):   # anti-windup: may shrink, not grow
        state["u"] = u_new
    return state["u"].copy()


# ---- the payload as a PhysX rigid body on a rope (--payload physx) -------------
# One dynamic box (the container's collision proxy, invisible; the glb follows its pose), one
# kinematic sphere at the boom tip, and ONE DISTANCE joint between them: max distance enabled,
# min 0, no stiffness. That is a rope - it holds the load at the wire length, it never pushes,
# and it may go slack. The platform and tower are static trimeshes, so the load can bump the
# tower, catch the railing, and rest on the grating with friction, none of which the round-5
# pendulum could do. Joint.reaction() is the load cell; ContactSensor(proxy) is the touchdown
# switch. Both are logged.
#
# Two joint-frame details worth writing down.
#
# (1) The Joint constructor takes ONE world frame and derives BOTH local anchors from it, so
#     the tip-side anchor lands at (frame - tip pose) in the tip body's frame. To get the tether
#     anchored AT the tip and at the container's own hook point (its top centre plus SLING_H),
#     the container is CREATED with its hook point exactly at the tip - which bakes local
#     anchors of (0,0,0) and (0, ANCHOR_UP, 0) - and only then set_pose'd down to hang a wire
#     length below. Local frames are fixed at creation, so the drop costs nothing.
# (2) Params.upper is read once, at creation, and nothing in the API re-opens it. The wire pays
#     out 2.5 m during the landing, so either the joint is re-created whenever the length moves
#     (a fresh constraint every centimetre: the solver's warm-start cache is thrown away each
#     time, and Joint.reaction - the load cell the landing is detected with - restarts at zero
#     on every re-creation) or the tether keeps its length and the KINEMATIC ANCHOR is lowered
#     by the pay-out instead. The second is taken. It is the same motion of the load, the
#     pendulum keeps one period throughout so the controller does not see a plant change
#     mid-landing, and the visual hoist wire is drawn from the REAL tip to the real hook, so
#     nothing on screen betrays it.
px = {"world": None, "proxy": None, "body": None, "tip_body": None, "joint": None, "sensor": None,
      "imu": None, "imu_rows": [], "imu_frame": [], "hull_body": None,
      "statics": None, "anchor": np.zeros(3), "centre": np.zeros(3), "tension": 0.0, "build_s": 0.0,
      "quat": None, "tilt": 0.0, "doors_excluded": [],
      "tower_min": 1e9, "tower_hit_t": None, "first_contact_t": None, "payout": 0.0, "max_stretch": 0.0,
      "rail_min": 1e9, "rail_hit_t": None, "door_min": 1e9, "door_hit_t": None, "door_who": ""}
FURNITURE_TOP = 21.60                  # the railing top (21.58) and the door tops (21.59), measured off the glb


def set_quat(dst, src):
    """The Python Quaternion has set() but no copy()."""
    dst.set(src.x, src.y, src.z, src.w)


def build_physx(tip):
    """Build the PhysX world at the operation's start pose. Called once, from first_frames()."""
    t0 = time.perf_counter()
    w = tp.PhysxWorld(gravity=tp.Vector3(0.0, -9.81, 0.0), fixed_timestep=DT / PHYSX_SUB,
                      max_substeps=max(4, PHYSX_SUB), num_threads=1)      # CPU PhysX: the bit-exact path
    mat = w.create_material(static_friction=0.6, dynamic_friction=0.5, restitution=0.0)
    yaw = tp.Quaternion()
    yaw.set_from_axis_angle(tp.Vector3(0.0, 1.0, 0.0), LOAD_YAW if LOAD_YAW is not None else -q[0])
    proxy = tp.Mesh(tp.BoxGeometry(2.0 * CONT_X, CONT_H, 2.0 * CONT_Z), standard_material(0x2b57a6))
    proxy.visible = False                                      # never drawn and never in the scene: the glb is the picture
    set_quat(proxy.quaternion, yaw)
    proxy.position.set(tip[0], tip[1] - ANCHOR_UP, tip[2])     # hook point AT the tip, so the joint bakes a zero tip-side anchor
    body = w.add(proxy, density=CONT_DENSITY, material=mat)
    tip_mesh = tp.Mesh(tp.SphereGeometry(0.15, 8, 6), standard_material(0x202020))
    tip_mesh.visible = False
    tip_mesh.position.set(*tip)
    tip_body = w.add(tip_mesh, density=1000.0, material=mat)
    tip_body.set_kinematic(True)
    jp = tp.Joint.Params()
    jp.type = tp.Joint.Type.DISTANCE
    jp.lower = 0.0
    jp.upper = wire0                                           # a rope: max distance only, slack allowed, no spring
    joint = tp.Joint(w, tip_body, body, position=tp.Vector3(*tip), rotation=tp.Quaternion(), params=jp)
    body.set_pose(tp.Vector3(tip[0], tip[1] - wire0 - ANCHOR_UP, tip[2]), yaw)
    if PHYSX_DAMP > 0.0:
        body.set_linear_damping(PHYSX_DAMP)
    sensor = tp.ContactSensor(proxy, rate_hz=0.0)
    w.register_sensor(sensor)
    # The motion reference unit: an IMU on the hull at the vessel's origin (the point vessel_step
    # rotates about), 100 Hz, seeded MEMS-class noise exactly as the paper's proprioceptive harness
    # (sensor_audit.py) seeds it; a manifest row of its own, hashed sample by sample.
    # The IMU needs a PhysX body in its ancestry: a small kinematic body rides the hull's pose
    # (set_kinematic_target every frame from vessel_step's state), the way the twin surrogates
    # its MRU; it sits at the vessel's origin, 36 m from the pickup, and never meets the load.
    hull_mesh = tp.Mesh(tp.BoxGeometry(0.4, 0.4, 0.4), standard_material(0x202020))
    hull_mesh.visible = False
    hull_mesh.position.set(0.0, ves["y"], 0.0)
    set_quat(hull_mesh.quaternion, vessel.quaternion)
    hull_body = w.add(hull_mesh, density=1000.0, material=mat)
    hull_body.set_kinematic(True)
    px["hull_body"] = hull_body
    mount = tp.Group()
    hull_mesh.add(mount)
    imu = tp.Imu(mount, rate_hz=IMU_HZ)
    g = imu.gyro_noise
    g.seed = SEED * 7919 + 1
    imu.gyro_noise = g
    a = imu.accel_noise
    a.seed = (SEED * 7919 + 1) ^ 0x9E3779B97F4A7C15
    imu.accel_noise = a
    w.register_sensor(imu)
    px["imu"] = imu
    t_cook = time.perf_counter()
    # The platform, the railing and the tower - but NOT the three doors. They are 1.09 m leaves
    # standing on the walking surface, and round 6's container came down on one and rested 0.49 m
    # up, tilted, instead of bedding flat on the grating. A door is furniture, not structure; it
    # is kept in the PICTURE and taken out of the COLLIDER set. The binding has no filter
    # argument, so the leaves are detached for the length of the cook and put straight back.
    doors = [(d, d.parent) for d in (turbine.get_object_by_name(n) for n in DOOR_MESHES) if d is not None]
    for d, _p in doors:
        d.remove_from_parent()
    statics = w.add_static_trimesh_tree(turbine)
    for d, p in doors:
        p.add(d)
    turbine.update_matrix_world(True)
    px["doors_excluded"] = [n for n in DOOR_MESHES if turbine.get_object_by_name(n) is not None]
    cook = time.perf_counter() - t_cook
    px.update(world=w, proxy=proxy, body=body, tip_body=tip_body, joint=joint, sensor=sensor,
              statics=statics, mat=mat, tip_mesh=tip_mesh, build_s=time.perf_counter() - t0)
    print(f"physx: {LOAD} {CONT_MASS / 1e3:.1f} t container ({body.mass:.0f} kg at {CONT_DENSITY:.0f} kg/m3, "
          f"{2 * CONT_X:.2f} x {CONT_H:.2f} x {2 * CONT_Z:.2f} m) on a {wire0:.1f} m "
          f"DISTANCE tether from a kinematic tip, {len(statics)} static trimeshes ({cook:.2f} s to cook), "
          f"{PHYSX_SUB} substep(s) of {1e3 * DT / PHYSX_SUB:.2f} ms, damping {PHYSX_DAMP}; build {px['build_s']:.2f} s")
    print(f"physx: excluded from the collider set: {', '.join(px['doors_excluded']) or 'nothing'} "
          f"(kept visible; the doors are furniture, not structure)")


def box_yaw(quat):
    return math.atan2(2.0 * (quat.w * quat.y + quat.x * quat.z), 1.0 - 2.0 * (quat.y ** 2 + quat.x ** 2))


def box_to_tower(pos, quat):
    """Horizontal distance from the container box to the tower SURFACE (negative = overlapping)."""
    yaw = box_yaw(quat)
    d = TOWER_XZ - np.array([pos.x, pos.z])
    c, s = math.cos(-yaw), math.sin(-yaw)
    local = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])          # into the box's own frame
    over = np.maximum(np.abs(local) - np.array([CONT_X, CONT_Z]), 0.0)    # half width across, half length along
    return float(np.linalg.norm(over)) - TOWER_R


def box_span(pos, quat):
    """The container footprint's world x and z extent: (x0, x1, z0, z1).

    The axis-aligned span of the rotated box, which is exact while the box is level and its yaw is
    0 (it is: a DISTANCE tether transmits no torque) and conservative otherwise."""
    yaw = box_yaw(quat)
    c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
    hx = c * CONT_X + s * CONT_Z
    hz = s * CONT_X + c * CONT_Z
    return (pos.x - hx, pos.x + hx, pos.z - hz, pos.z + hz)


def box_to_railing(pos, quat):
    """Metres from the container footprint to the NEAREST railing, positive while it is inside."""
    x0, x1, z0, z1 = box_span(pos, quat)
    return min(x0 - DECK_RECT[0], DECK_RECT[1] - x1, z0 - DECK_RECT[2], DECK_RECT[3] - z1)


def box_to_doors(pos, quat):
    """(gap, name) to the nearest door leaf: 0 or less means the footprints overlap."""
    x0, x1, z0, z1 = box_span(pos, quat)
    best, who = 1e9, ""
    for nm, (dx0, dx1, dz0, dz1, _y0, _y1) in DOOR_BOX.items():
        gx = max(dx0 - x1, x0 - dx1, 0.0)
        gz = max(dz0 - z1, z0 - dz1, 0.0)
        g = math.hypot(gx, gz) if (gx > 0.0 or gz > 0.0) else -min(min(x1 - dx0, dx1 - x0), min(z1 - dz0, dz1 - z0))
        if g < best:
            best, who = g, nm
    return best, who


def in_clear_rect(pos, quat):
    """Is the whole container footprint inside the clear rectangle (1.5 m off the tower, 1.0 m off
    every railing, and therefore clear of all three doors, which stand on the railing line)?"""
    x0, x1, z0, z1 = box_span(pos, quat)
    return bool(x0 >= CLEAR_RECT[0] and x1 <= CLEAR_RECT[1] and z0 >= CLEAR_RECT[2] and z1 <= CLEAR_RECT[3])


def box_tilt_deg(quat):
    """How far the container's own up-axis has left vertical, degrees."""
    up = tp.Vector3(0.0, 1.0, 0.0)
    up.apply_quaternion(quat)
    return math.degrees(math.acos(max(-1.0, min(1.0, up.y))))


def physx_step(tip, dt):
    """One PhysX frame: drive the kinematic anchor, step, read the pose back.

    The anchor is the real tip LOWERED by the pay-out (see the note above), so the tether's
    fixed 6.0 m length plus that drop is the wire the operation is actually paying out."""
    px["payout"] = wire_len - wire0
    anchor_tip = tip - np.array([0.0, px["payout"], 0.0])
    px["tip_body"].set_kinematic_target(tp.Vector3(*anchor_tip))
    if px["hull_body"] is not None:                       # the hull rides vessel_step's pose; the IMU reads it
        px["hull_body"].set_kinematic_target(tp.Vector3(0.0, ves["y"], 0.0), vessel.quaternion)
    px["world"].step(dt)
    px["imu_frame"] = []
    if px["imu"] is not None:
        for s in px["imu"].drain():
            vals = (s.t, s.angular_velocity.x, s.angular_velocity.y, s.angular_velocity.z,
                    s.linear_acceleration.x, s.linear_acceleration.y, s.linear_acceleration.z)
            px["imu_frame"].append(vals)
        px["imu_rows"].extend(px["imu_frame"])
    pos, quat = px["body"].position, px["body"].quaternion
    px["quat"] = quat
    f, _t = px["joint"].reaction()
    px["tension"] = math.sqrt(f.x * f.x + f.y * f.y + f.z * f.z)
    # the hook for the visuals and for the swing: the container's own anchor point in the world
    up = tp.Vector3(0.0, ANCHOR_UP, 0.0)
    up.apply_quaternion(quat)
    anchor = np.array([pos.x + up.x, pos.y + up.y, pos.z + up.z])
    px["anchor"] = anchor
    px["centre"] = np.array([pos.x, pos.y, pos.z])
    stretch = float(px["joint"].position) - wire0             # the load cell's own length reading
    px["max_stretch"] = max(px["max_stretch"], stretch)
    slack = -min(stretch, 0.0)
    clear = box_to_tower(pos, quat)
    px["tower_min"] = min(px["tower_min"], clear)
    px["tilt"] = box_tilt_deg(quat)
    # The railing and the doors only exist between the walking surface and FURNITURE_TOP, so the
    # box can only touch them while its own bottom is below that; the horizontal clearance is
    # tracked whatever the height (it is the "how close did it come" number) but the HIT is not.
    # ... and only while the box is over the deck at all: for most of the lift it hangs over open
    # water thirty metres outside the railing rectangle, where "distance to the nearest railing"
    # is a large negative number that means nothing.
    low = (pos.y - 0.5 * CONT_H) < FURNITURE_TOP
    x0, x1, z0, z1 = box_span(pos, quat)
    over_deck = (x1 > DECK_RECT[0] and x0 < DECK_RECT[1] and z1 > DECK_RECT[2] and z0 < DECK_RECT[3])
    rail = box_to_railing(pos, quat)
    door, who = box_to_doors(pos, quat)
    if over_deck:
        px["rail_min"] = min(px["rail_min"], rail)
        if low and px["rail_hit_t"] is None and rail < 0.0:
            px["rail_hit_t"] = sim_t
    if door < px["door_min"]:
        px["door_min"], px["door_who"] = door, who
    if low and px["door_hit_t"] is None and door < 0.0:
        px["door_hit_t"] = sim_t
    on = bool(px["sensor"].in_contact)
    if px["tower_hit_t"] is None and clear < 0.02:
        px["tower_hit_t"] = sim_t                    # geometric, not sensor-gated: the box has reached the tower
    if on and px["first_contact_t"] is None:
        px["first_contact_t"] = sim_t                # anything at all: the grating, the railing, the tower
    update_contact(on, slack, np.array([pos.x, pos.z]))
    # the visual container: the glb's origin is its BOTTOM face, the proxy's is its centre
    dn = tp.Vector3(0.0, -0.5 * CONT_H, 0.0)
    dn.apply_quaternion(quat)
    container.position.set(pos.x + dn.x, pos.y + dn.y, pos.z + dn.z)
    set_quat(container.quaternion, quat)
    container.update_matrix_world(True)
    return anchor


def sling_corners_world():
    """The four sling feet: the container's top corners, from whichever payload model is live."""
    if PAYLOAD == "physx":
        pos, quat = px["body"].position, px["quat"]
        out = []
        for sx in (-1, 1):
            for sz in (-1, 1):
                v = tp.Vector3(sx * CONT_X, 0.5 * CONT_H, sz * CONT_Z)
                v.apply_quaternion(quat)
                out.append(np.array([pos.x + v.x, pos.y + v.y, pos.z + v.z]))
        return out
    out = []
    for c_ in SLING_CORNERS:
        v_ = load_pivot.local_to_world(tp.Vector3(*c_))
        out.append(np.array([v_.x, v_.y, v_.z]))
    return out


as_state = antiswing_state()
swing_est = {"s": np.zeros(2), "ds": np.zeros(2), "seen": False}
sensor_every = 3                       # 20 Hz
frame_i = 0
sim_t = 0.0
tip_world = np.zeros(3)
load_world = np.zeros(3)
log_rows = []
antiswing_on = [not NO_ANTISWING]
ff_last = np.zeros(3)                  # the MRU feedforward joint velocities of the last step (log cols 19-21)
gain_rec = []                          # per sensor update: |s_est|, |anchor - tip|, |container centre - tip|


def load_centre(pts):
    """The container's centre from the returns that carry its id: the MID-RANGE in the load's own frame.

    Not the centroid. The mean of the returns is biased by what the head cannot see through: the
    hoist wire, the hook and the four slings all hang between the head and the container's top face
    and their shadows fall INBOARD of the container's centre (the head is mounted outboard), so the
    surviving returns lean outboard. Measured open loop at the pickup: the centroid sat 0.27 m
    outboard of the hook, against a true swing of 0.05 m - a gain of 3.8 and a standing tip offset
    of 0.35 m once the integral law had chased it.

    The container's top face is a known 2 x CONT_X by 2 x CONT_Z rectangle (2.86 x 7.00 m for the
    20 ft box, 2.86 x 3.50 for round 7's 10 ft one), and the estimator knows its yaw
    from the slew encoder (the load hangs on four slings and turns with the king). Rotated into
    that frame the face is axis-aligned, so each edge is sampled along its whole length and the
    mid-range of the returns is the centre: the shadows are INTERIOR, and the half-cell the grid
    falls short at one edge it also falls short at the other, so it cancels."""
    # The load's yaw. Round 5 read it off the slew encoder, because the load hung rigidly under
    # the hook and turned with the king. It does not any more: a DISTANCE tether transmits no
    # torque, so a PhysX container keeps whatever heading it was picked up with (measured: the
    # quaternion's y component stays 0 to five decimals over a whole lift). The load is therefore
    # given a fixed heading, LOAD_YAW, chosen so its 7 m length lies ACROSS the boom at the
    # landing - the only way a 7 m box fits between the platform's railing and its tower - and
    # the estimator uses that. The mid-range is in any case first-order insensitive to the yaw:
    # the return set is centrally symmetric about the container's centre, so the mid-range of
    # its extent is that centre in any frame.
    a = -LOAD_YAW if LOAD_YAW is not None else q[0]
    c, s = math.cos(a), math.sin(a)
    u = c * pts[:, 0] + s * pts[:, 2]
    v = -s * pts[:, 0] + c * pts[:, 2]
    uu = 0.5 * (u.min() + u.max())
    vv = 0.5 * (v.min() + v.max())
    return np.array([c * uu - s * vv, pts[:, 1].mean(), s * uu + c * vv])


def sensor_origin(tip):
    """The sensor head: under the boom tip and outboard of the hoist wire, on the boom's radial line."""
    k = wpos(king)
    d = np.array([tip[0] - k[0], 0.0, tip[2] - k[2]])
    n = float(np.linalg.norm(d))
    return tip + SENSOR_DROP + (SENSOR_OUT / n) * d if n > 1e-6 else tip + SENSOR_DROP


def tip_fan(tip):
    """The tip range sensor: one ray-traced dispatch straight down, the container picked out by its instance id.

    Rounds 1-4 selected the load by RANGE (`dist < wire_len + LOAD_DROP + 1`), which admitted the
    deck; but the deck was never the problem, because no ray ever reached it. With the head on the
    wire's axis EVERY ray left through the inside of the hoist-wire cylinder and stopped there, and
    the near-vertical ones ran 0.5 to 6 m inside it before doing so - inside the old range window.
    Rounds 1-4 measured the WIRE, and a point half way down the wire is half the swing, which is
    the 0.84 / 0.53 gain the plan recorded. The head is now outboard (SENSOR_OUT) and the gate is
    the id: the tracer's `instance_id` is the stable per-object id, the same number the raster ids
    AOV writes and `renderer.set_instance_id` sets, so the hook, the four slings and the wires are
    dropped by name and only the container's meshes survive."""
    o = sensor_origin(tip)
    origins = np.repeat(o[None].astype(np.float32), len(FAN_DIRS), 0)
    r = renderer.scan_lidar(origins, FAN_DIRS, fan_params)
    hit = r["return_no"] > 0
    dist = r["distance"].astype(np.float64)
    dist = dist + rng.normal(0.0, FAN_SIGMA, dist.shape) * hit
    ids = r["instance_id"]
    FAN_LAST["fan"] = np.concatenate([dist.astype(np.float32)[:, None], ids[:, None].astype(np.float32)], 1)
    near = hit & (ids == LOAD_ID) & (dist > 0.5)
    fan_seen[0] += 1
    fan_seen[2] += int(near.sum())
    if near.sum() < 4:
        return None
    fan_seen[1] += 1
    pts = o[None] + FAN_DIRS[near].astype(np.float64) * dist[near][:, None]
    return load_centre(pts)


def engaged(t):
    """Is the swing-suppression loop closed right now?

    `always` is round 5: closed from the first frame, so the law fights the transfer as well.
    `ondemand` is the cited controller's own use case (espenakk2026demand): the operator flies
    the load across open loop and presses the button ON ARRIVAL, and the loop's whole job is to
    take the residual swing out during the arrival hold, before the pay-out starts. It engages
    at T2 with the hover point as its target and disengages on `landed` (which is handled
    above, with the 0.5 s bleed)."""
    if not antiswing_on[0] or contact["landed"]:
        return False
    return ENGAGE == "always" or t >= T2


def step(dt=DT):
    global frame_i, sim_t, tip_world, load_world, wire_len
    sim_t += dt
    renderer.sim_time = sim_t
    frame_i += 1
    vessel_step(dt)
    if gulls is not None:
        gulls.update(dt)
    if rotor is not None:
        rotor.rotation.z = 0.25 * sim_t
    # the lift
    target, wl = op_target(sim_t) if OP or SHOT or AUDIT or True else (None, wire0)
    wire_len = wl
    cmd = target.copy()
    corr = np.zeros(2)
    if contact["landed"]:
        # Set down: the operator lets the button go. The correction is not dropped in one frame
        # (a 1.5 m step in the tip target would run the slew at its rate limit for a second) but
        # bled out with a 0.5 s time constant.
        as_state["u"] *= max(0.0, 1.0 - 2.0 * dt)
        corr = as_state["u"].copy()
        cmd[0] += corr[0]
        cmd[2] += corr[1]
    elif engaged(sim_t) and LAW == "legacy" and swing_est["seen"]:
        corr = AS_KP * swing_est["s"] + AS_KD * swing_est["ds"]
        n = float(np.linalg.norm(corr))
        if n > AS_CLAMP:
            corr *= AS_CLAMP / n
        if float(np.linalg.norm(swing_est["s"])) > AS_CUTOUT:
            corr = np.zeros(2)
        cmd[0] += corr[0]
        cmd[2] += corr[1]
    elif engaged(sim_t) and LAW == "integral":
        corr = antiswing_update(swing_est["s"], swing_est["seen"], sensor_every * dt, as_state)
        cmd[0] += corr[0]
        cmd[2] += corr[1]
    if NO_AMC:
        q_cmd = q                                   # joints frozen: the tip rides the vessel
    else:
        q_cmd = ik(cmd, q)
    # The nominal target's own world velocity, centred on this instant (op_target is analytic,
    # so a central difference costs two evaluations and carries no half-sample lag).
    v_tgt = (op_target(sim_t + 0.5 * dt)[0] - op_target(sim_t - 0.5 * dt)[0]) / dt
    ff = np.zeros(3) if (NO_FF or NO_AMC) else mru_feedforward(q_cmd, v_tgt)
    ff_last[:] = ff
    actuate(q_cmd, dt, ff)
    if bool(np.any(q <= Q_MIN + 1e-6) or np.any(q >= Q_MAX - 1e-6)):
        sat_frames[0] += 1
        sat_joint[:3] += (q <= Q_MIN + 1e-6)
        sat_joint[3:] += (q >= Q_MAX - 1e-6)
    tip_world = fk(q)
    if PAYLOAD == "physx":
        hook = physx_step(tip_world, dt)             # the container's own anchor point in the world
        load_pivot.position.set(*hook)
        load_pivot.rotation.y = LOAD_YAW if LOAD_YAW is not None else -q[0]
    else:
        hook = pendulum_step(tip_world, dt)
        load_pivot.position.set(*hook)
        load_pivot.rotation.y = LOAD_YAW if LOAD_YAW is not None else -q[0]
        px["centre"] = hook - np.array([0.0, ANCHOR_UP, 0.0])    # the pendulum's load is rigid under the hook and never tilts
    load_world = hook
    load_pivot.update_matrix_world(True)
    # wires
    aim_wire(hoist_wire, tip_world, hook)
    for w_, c_ in zip(slings, sling_corners_world()):
        aim_wire(w_, hook, c_)
    aim_wire(boom_wire_in, wpos(winch_in), wpos(in_target))
    aim_wire(boom_wire_out, wpos(winch_out), wpos(out_target))
    # the tip camera sits at the sensor head and looks down it
    _so = sensor_origin(tip_world)
    tip_cam.position.set(*_so)
    tip_cam.look_at(*(_so + np.array([0.0, -20.0, 0.0])))
    # the tip sensor and the swing estimate (20 Hz, its own clock)
    if frame_i % sensor_every == 0 and frame_i > 2:
        c = tip_fan(tip_world)
        if c is not None:
            s = np.array([c[0] - tip_world[0], c[2] - tip_world[2]])
            if swing_est["seen"]:
                swing_est["ds"] = (s - swing_est["s"]) / (sensor_every * dt)
            swing_est["s"] = s
            swing_est["seen"] = True
            # What the fan MEASURES is the container's own centre; what the operation's swing
            # metric calls the swing is the ANCHOR (the hook point above it), and with a rigid
            # container on a rope those are not the same point: the body tilts with the rope, so
            # the centre hangs on an effective pendulum of wire + ANCHOR_UP while the anchor
            # hangs on the wire alone. Both ratios are recorded rather than argued about.
            gain_rec.append([float(np.linalg.norm(s)),
                             math.hypot(hook[0] - tip_world[0], hook[2] - tip_world[2]),
                             math.hypot(px["centre"][0] - tip_world[0], px["centre"][2] - tip_world[2])
                             if PAYLOAD == "physx" else math.hypot(hook[0] - tip_world[0], hook[2] - tip_world[2])])
    log_rows.append(np.array([sim_t, *tip_world, *hook, *q, wire_len, ves["y"], ves["pitch"], ves["roll"],
                              *(target if target is not None else (0, 0, 0)),
                              swing_est["s"][0], swing_est["s"][1], *ff_last,
                              float(contact["on"]), float(contact["landed"]), corr[0], corr[1],
                              px["tension"], *px["centre"], px["tilt"]], np.float64))


# ---- cameras ---------------------------------------------------------------
camera = tp.PerspectiveCamera(42.0, W / H, 0.3, 6000.0)
SHOTS = {
    "hero": ((-40.0, 9.0, 58.0), (18.0, 19.0, 27.0)),          # the paper's Fig. 1 framing (chosen 2026-09-07 from three candidates)
    "hero_wide": ((-44.0, 11.0, 60.0), (14.0, 19.0, 26.0)),
    "hero3": ((-30.0, 6.0, 52.0), (20.0, 20.0, 27.0)),
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
    global TIP_VIEW, q, q_filt, wire_len
    vessel.update_matrix_world(True)
    wire_len = op_target(0.0)[1]                    # the hook starts ON the wire, not 1 m of slack below it
    A0 = op_target(0.0)[0]                          # start ON the pickup, not in the parked pose: best of four slew seeds
    cands = [ik(A0, np.array([s0, 0.0, 3.0]), iters=14) for s0 in (-math.pi / 2, math.pi / 2, 0.0, math.pi)]
    q = min(cands, key=lambda c: float(np.linalg.norm(fk(c) - A0)))
    print(f"start pose: slew {math.degrees(q[0]):.0f} deg, luff {math.degrees(q[1]):.0f} deg, telescope {q[2]:.1f} m, "
          f"tip error {1e3 * float(np.linalg.norm(fk(q) - A0)):.0f} mm")
    q_filt = q.copy()
    if PAYLOAD == "physx":
        build_physx(fk(q))
    else:
        pendulum_step(fk(q), DT)
    print(f"operation: hold 0-{T1:.2f} s, transfer {T1:.2f}-{T2:.2f} s ({T_XFER:.2f} s, {XFER_PROFILE}, "
          f"sweep {math.degrees(abs(D_TH)):.1f} deg, peak slew {XFER_V:.4f} rad/s = "
          f"{100.0 * XFER_V / V_MAX[0]:.0f}% of the limit), arrival hold {T2:.2f}-{T3:.2f} s, "
          f"pay-out {T3:.2f}-{T4:.2f} s, then hold; --op >= {T4 + 2.0:.1f}")
    print(f"geometry ({GEOM}): turbine {TURBINE_POS}, landing point {np.round(PLATFORM, 2).tolist()}, "
          f"{float(np.linalg.norm(PLATFORM[[0, 2]] - KING_XZ)):.2f} m from the king; deck rectangle "
          f"x {DECK_RECT[0]:.2f}..{DECK_RECT[1]:.2f}, z {DECK_RECT[2]:.2f}..{DECK_RECT[3]:.2f}; "
          f"landing point {PLATFORM[0] - DECK_RECT[0]:.2f} m inside the near railing, "
          f"tip to the tower axis {float(np.linalg.norm(PLATFORM[[0, 2]] - TOWER_XZ)):.2f} m, "
          f"container's near face to the tower surface "
          f"{float(np.linalg.norm(PLATFORM[[0, 2]] - TOWER_XZ)) - CONT_X - TOWER_R:.2f} m")
    _p = tp.Vector3(PLATFORM[0], PLATFORM[1] + 0.5 * CONT_H, PLATFORM[2])
    _qi = tp.Quaternion()
    _yaw = tp.Quaternion()
    _yaw.set_from_axis_angle(tp.Vector3(0.0, 1.0, 0.0), LOAD_YAW if LOAD_YAW is not None else 0.0)
    _sx0, _sx1, _sz0, _sz1 = box_span(_p, _yaw)
    print(f"clearances with the {LOAD} box set down ON the landing point ({2 * CONT_X:.2f} x {2 * CONT_Z:.2f} m "
          f"footprint x {_sx0:.2f}..{_sx1:.2f}, z {_sz0:.2f}..{_sz1:.2f}):")
    print(f"  tower surface {box_to_tower(_p, _yaw):.2f} m (asked {TOWER_MARGIN:.1f}); railings "
          f"-x {_sx0 - DECK_RECT[0]:.2f}, +x {DECK_RECT[1] - _sx1:.2f}, -z {_sz0 - DECK_RECT[2]:.2f}, "
          f"+z {DECK_RECT[3] - _sz1:.2f} m (asked {RAIL_MARGIN:.1f})")
    for _nm, (_dx0, _dx1, _dz0, _dz1, _dy0, _dy1) in DOOR_BOX.items():
        _gx = max(_dx0 - _sx1, _sx0 - _dx1, 0.0)
        _gz = max(_dz0 - _sz1, _sz0 - _dz1, 0.0)
        print(f"  {_nm}: x {_dx0:.2f}..{_dx1:.2f}, z {_dz0:.2f}..{_dz1:.2f}, y {_dy0:.2f}..{_dy1:.2f} "
              f"-> gap {math.hypot(_gx, _gz):.2f} m, footprints {'OVERLAP' if _gx <= 0 and _gz <= 0 else 'clear'}")
    print(f"  clear rectangle for the footprint: x {CLEAR_RECT[0]:.2f}..{CLEAR_RECT[1]:.2f} "
          f"({CLEAR_RECT[1] - CLEAR_RECT[0]:.2f} m for a {2 * CONT_X:.2f} m width, tolerance "
          f"+-{0.5 * (CLEAR_RECT[1] - CLEAR_RECT[0] - 2 * CONT_X):.2f} m), z {CLEAR_RECT[2]:.2f}..{CLEAR_RECT[3]:.2f}; "
          f"nominal landing inside: {'YES' if in_clear_rect(_p, _yaw) else 'NO'}")
    del _p, _qi
    renderer.sim_time = 0.0
    renderer.render(scene, camera)
    ves["live"] = True
    TIP_VIEW = renderer.add_view(tip_cam, CAM_W, CAM_H)
    if TIP_VIEW:
        renderer.set_view_display_rect(TIP_VIEW, W - HUD_M - CAM_W, H - HUD_M - CAM_H, CAM_W, CAM_H)


def op_report():
    L = np.asarray(log_rows)
    if len(L) < 10:
        return
    s = np.hypot(L[:, 4] - L[:, 1], L[:, 6] - L[:, 3])         # true horizontal swing, hook vs tip
    err = np.linalg.norm(L[:, 1:4] - L[:, 14:17], axis=1)      # tip vs its nominal target
    n = len(L)
    rate = np.abs(np.diff(L[:, 7], prepend=L[0, 7])) / DT      # the realised slew rate, for the saturation fraction
    for name, lo, hi in (("hold    ", 0.0, T1), ("transfer", T1, T2),
                         ("arrival ", T2, T3), ("pay-out ", T3, 1e9)):
        m = (L[:, 0] >= lo) & (L[:, 0] < hi)
        if m.any():
            print(f"  {name} ({lo:.1f}-{min(hi, L[-1, 0]):.1f} s, {int(m.sum())} frames): "
                  f"swing RMS {1e3 * math.sqrt((s[m] ** 2).mean()):.0f} mm, max {1e3 * s[m].max():.0f} mm; "
                  f"tip-target RMS {1e3 * math.sqrt((err[m] ** 2).mean()):.0f} mm; "
                  f"slew at the rate limit {100.0 * (rate[m] > V_MAX[0] * 0.999).mean():.1f}%")
    if L[-1, 0] > T3:
        i2 = int(np.argmin(np.abs(L[:, 0] - T2)))
        i3 = int(np.argmin(np.abs(L[:, 0] - T3)))
        settle = [i for i in range(i2, i3 + 1) if s[i:i3 + 1].max() < 0.15]
        arr = (L[:, 0] >= T2) & (L[:, 0] < T2 + 1.0)
        print(f"  arrival: swing at T2 {1e3 * s[i2]:.0f} mm (first second RMS {1e3 * math.sqrt((s[arr] ** 2).mean()):.0f} mm, "
              f"peak {1e3 * s[(L[:, 0] >= T2) & (L[:, 0] < T3)].max():.0f} mm); at the start of the pay-out "
              f"{1e3 * s[i3]:.0f} mm; settled under 150 mm "
              + (f"{L[settle[0], 0] - T2:.2f} s after arrival" if settle else "never inside the arrival hold"))
    if PAYLOAD == "physx" and L[-1, 0] > T3:
        tc = px["first_contact_t"] or L[-1, 0]
        before = L[(L[:, 0] > tc - 1.0) & (L[:, 0] < tc), 26]
        after = L[L[:, 0] > tc + 1.5, 26]
        print(f"  tether: tension {L[60:i2, 26].mean() / 1e3:.1f} kN carrying the load "
              f"({CONT_MASS * 9.81 / 1e3:.1f} kN = mg), {before.mean() / 1e3 if len(before) else float('nan'):.1f} kN "
              f"in the second before first contact (t = {tc:.2f} s), "
              f"{after.mean() / 1e3 if len(after) else float('nan'):.1f} kN after; "
              f"max stretch over the {wire0:.1f} m length {1e3 * px['max_stretch']:.0f} mm")
        print(f"  clearance: container to the tower surface, min {px['tower_min']:.2f} m"
              + (f", TOUCHED at t = {px['tower_hit_t']:.2f} s" if px["tower_hit_t"] else ", never touched")
              + ("; first contact was BEFORE the pay-out (it struck the structure)" if tc < T3
                 else "; no contact before the pay-out"))
        print(f"  clearance: to the nearest railing, min {px['rail_min']:.2f} m"
              + (f", CROSSED at t = {px['rail_hit_t']:.2f} s" if px["rail_hit_t"] else " (never crossed below the railing top)")
              + f"; to the nearest door ({px['door_who'] or 'n/a'}), min {px['door_min']:.2f} m"
              + (f", OVERLAPPED at t = {px['door_hit_t']:.2f} s" if px["door_hit_t"] else " (never overlapped)"))
        # The resting attitude: how far the box's own up-axis has left vertical, and how high its
        # anchor sits against a box lying FLAT on the grating. Round 6's on-demand runs came to
        # rest 0.49 m high on the deck furniture; a flat landing reads 0.00.
        rest = L[L[:, 0] > L[-1, 0] - 1.0]
        if rest.shape[1] > 30:
            anchor_level = GRATING_Y + CONT_H + SLING_H
            pos_r, quat_r = px["body"].position, px["quat"]
            print(f"  at rest (last second): tilt {rest[:, 30].mean():.2f} deg (max {rest[:, 30].max():.2f}), "
                  f"anchor {rest[:, 5].mean():.3f} m = {rest[:, 5].mean() - anchor_level:+.3f} m against a box lying flat "
                  f"({anchor_level:.3f}); footprint x {box_span(pos_r, quat_r)[0]:.2f}..{box_span(pos_r, quat_r)[1]:.2f}, "
                  f"z {box_span(pos_r, quat_r)[2]:.2f}..{box_span(pos_r, quat_r)[3]:.2f}; inside the clear rectangle "
                  f"(x {CLEAR_RECT[0]:.2f}..{CLEAR_RECT[1]:.2f}, z {CLEAR_RECT[2]:.2f}..{CLEAR_RECT[3]:.2f}): "
                  f"{'YES' if in_clear_rect(pos_r, quat_r) else 'NO'}; tower {box_to_tower(pos_r, quat_r):.2f} m, "
                  f"railing {box_to_railing(pos_r, quat_r):.2f} m, door {box_to_doors(pos_r, quat_r)[0]:.2f} m")
    if L.shape[1] > 21:
        print(f"  MRU feedforward ({'off' if NO_FF else FF_MODE}): |q_dot_ff| RMS slew/luff/tel "
              + "/".join(f"{v:.4f}" for v in np.sqrt((L[:, 19:22] ** 2).mean(0))))
    est = np.abs(L[:, 17]) + np.abs(L[:, 18]) > 0
    if est.any():
        gain = np.median(np.hypot(L[est, 17], L[est, 18]) / np.maximum(s[est], 1e-6))
        G = np.asarray(gain_rec) if gain_rec else np.zeros((1, 3))
        ok = G[:, 1] > 0.05
        print(f"  sensor: the load in {fan_seen[1]}/{fan_seen[0]} scans, {fan_seen[2] / max(fan_seen[1], 1):.0f} returns per hit, "
              f"estimate gain (median |s_est|/|s_true|) {gain:.3f}; against the container's own centre "
              f"{np.median(G[ok, 0] / np.maximum(G[ok, 2], 1e-6)) if ok.any() else float('nan'):.3f}, "
              f"against the anchor {np.median(G[ok, 0] / np.maximum(G[ok, 1], 1e-6)) if ok.any() else float('nan'):.3f}")
    slack = L[:, 10] - np.linalg.norm(L[:, 4:7] - L[:, 1:4], axis=1)
    if contact["landed"]:
        print(f"  LANDED at t = {contact['t_land']:.2f} s ({contact['t_land'] - T3:.2f} s into the pay-out), container centre "
              f"{1e3 * float(np.linalg.norm(contact['xz'])):.0f} mm "
              f"from the platform point (dx {contact['xz'][0]:+.2f}, dz {contact['xz'][1]:+.2f} m), "
              f"{'INSIDE' if contact['on_deck'] else 'OUTSIDE'} the deck rectangle; "
              f"in contact for {int(L[:, 22].sum())}/{n} frames, max wire slack {slack.max():.2f} m")
    else:
        print(f"  NOT landed: in contact for {int(L[:, 22].sum()) if L.shape[1] > 22 else 0}/{n} frames, "
              f"max wire slack {slack.max():.2f} m, hook at the end {np.round(L[-1, 4:7], 2).tolist()}")
    print(f"lift: {L[-1, 0]:.1f} s, law {'off' if not antiswing_on[0] else LAW}"
          f"{f' k={AS_K}' if antiswing_on[0] and LAW == 'integral' else ''}, "
          f"swing RMS {1e3 * math.sqrt((s ** 2).mean()):.0f} mm, max {1e3 * s.max():.0f} mm; "
          f"tip-target RMS {1e3 * math.sqrt((err ** 2).mean()):.0f} mm, max {1e3 * err.max():.0f} mm; "
          f"heave span {L[:, 11].max() - L[:, 11].min():.2f} m, roll +-{math.degrees(np.abs(L[:, 13]).max()):.1f} deg; "
          f"a joint at a position limit in {sat_frames[0]}/{n} frames (slew/luff/tel at min {sat_joint[:3].tolist()}, "
          f"at max {sat_joint[3:].tolist()}), at a rate limit in {rate_sat_frames[0]}/{n} "
          f"({100.0 * rate_sat_frames[0] / n:.1f}%), at an acceleration limit in {acc_sat_frames[0]}/{n} "
          f"({100.0 * acc_sat_frames[0] / n:.1f}%)")
    # Every joint's smallest margin to its own position limits over the whole run. `sat_frames`
    # counts frames AT a limit; this says how close the run ever came, which is what tells you
    # whether moving the landing point has spent the crane's envelope.
    for j, nm in enumerate(("slew  ", "luff  ", "telesc")):
        lo = float((L[:, 7 + j] - Q_MIN[j]).min())
        hi = float((Q_MAX[j] - L[:, 7 + j]).min())
        unit = "rad" if j < 2 else "m"
        print(f"  {nm} range over the run [{L[:, 7 + j].min():+.4f}, {L[:, 7 + j].max():+.4f}] {unit}; "
              f"margin to min {lo:.4f}, to max {hi:.4f} {unit}"
              + (f" ({math.degrees(lo):.1f} / {math.degrees(hi):.1f} deg)" if j < 2 else ""))


def aa_path():
    """The temporal resolve actually running, for the manifest: DLSS, FSR 3.1, or the in-house TAA.
    `renderer.fsr` / `renderer.dlss` report the ACTIVE upscaler (decided at the first frame), so this
    is read after first_frames(). Nothing here changes the path."""
    if bool(getattr(renderer, "dlss", False)):
        return "dlss"
    if bool(getattr(renderer, "fsr", False)):
        return "fsr3.1"
    return "taa"


def _jsonable(o):
    """numpy scalars in the manifest meta. np.float64 subclasses float and passes; np.bool_ does
    NOT subclass bool and raised, which is why this exists."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serialisable: {type(o).__name__}")


def film_writer(path, fps, preset="veryfast", crf="18"):
    """The netpen film's encoder settings (warp_netpen.film_writer)."""
    import imageio.v2 as imageio
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=None, macro_block_size=None,
                              ffmpeg_params=["-crf", crf, "-pix_fmt", "yuv420p", "-preset", preset])


FILM_FPS = 30                          # every second 60 Hz frame
_png_done = {"arrival": False, "touchdown": False}
_png_peak = [-1.0, 0.0]                # the largest arrival-phase swing written so far, and when


def _op_png(f, rgb, tip_px):
    """Three stills out of the RUN's own frames (not a second simulation in --shot mode, whose
    phase against the sea differs): the moment the load arrives over the platform, the moment it
    is set down, and the worst swing of the arrival hold.

    The last one cannot be seen coming, so it is written whenever the arrival-phase swing sets a
    new maximum and the file is simply overwritten; what survives the run is the peak frame. It
    costs a few tens of PNG writes in the first swing cycle and nothing after."""
    import imageio.v2 as imageio
    s_now = math.hypot(load_world[0] - tip_world[0], load_world[2] - tip_world[2])
    peak = T2 <= sim_t < T3 and s_now > _png_peak[0]
    if peak:
        _png_peak[0], _png_peak[1] = s_now, sim_t
    for tag, when in (("arrival", sim_t >= T2 and not _png_done["arrival"]),
                      ("touchdown", (contact["landed"] or sim_t >= T4) and not _png_done["touchdown"]),
                      ("maxswing", peak)):
        if not when:
            continue
        _png_done[tag] = True
        path = f"{OP_PNG}_{tag}.png"
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        imageio.imwrite(path, film_composite(rgb, tip_px))
        if tag != "maxswing":
            print(f"png: {tag} at t = {sim_t:.2f} s (frame {f}) -> {path}")


def film_composite(rgb, tip):
    """The hero frame as the display shows it. The renderer's RGB read already carries the secondary
    view at its display rect (seen 2026-09-07 when the rect moved to the lower right and the frame
    showed two insets), so nothing is drawn here; the tip view is kept only for its own hash row."""
    return rgb


def _film_composite_unused(rgb, tip):
    if tip is None or tip.size == 0:
        return rgb
    out = rgb.copy()
    th, tw = tip.shape[:2]
    y0, x0 = out.shape[0] - HUD_M - th, HUD_M
    if y0 < 0 or x0 + tw > out.shape[1]:
        return out
    out[y0 - 2:y0 + th + 2, x0 - 2:x0 + tw + 2] = 20        # a thin frame round the inset
    out[y0:y0 + th, x0:x0 + tw] = tip
    return out


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
            "tip.rgb", "fan", "events.raw", "events.sorted", "traj", "vessel", "tension", "contact", "imu"]
    rows = {k: sa.Fnv() for k in keys}
    per_frame = {"rgb": [], "tip.rgb": []}
    import hashlib
    n_events = 0
    writer = film_writer(OP_FILM, FILM_FPS) if (OP_FILM and mode == "op") else None
    film_wall, film_frames = [0.0], [0]
    if writer is not None:
        print(f"film: every {FPS // FILM_FPS}nd frame of {n} to {OP_FILM} at {FILM_FPS} fps, {W}x{H} with the {CAM_W}x{CAM_H} tip inset")
    wall0 = time.perf_counter()
    for f in range(n):
        step()
        renderer.set_event_camera_params(threshold=0.20, decay=0.88, min_luma=0.005, max_events_per_pixel=5,
                                         frame_time_us=int(sim_t * 1e6))
        aovs = renderer.read_aovs_typed(scene, camera, ["rgb", "depth", "normals", "instance_ids", "motion", "albedo"])
        rows["rgb"].update(sa.arr_bytes(aovs["rgb"]))
        per_frame["rgb"].append(hashlib.sha256(sa.arr_bytes(aovs["rgb"])).hexdigest()[:16])
        rows["aov.depth"].update(sa.arr_bytes(aovs["depth"]))
        rows["aov.normals"].update(sa.arr_bytes(aovs["normals"]))
        rows["aov.ids"].update(sa.arr_bytes(aovs["instance_ids"]))
        rows["aov.motion"].update(sa.arr_bytes(aovs["motion"]))
        rows["aov.albedo"].update(sa.arr_bytes(aovs["albedo"]))
        tip_px = None
        if TIP_VIEW:
            tip_px = renderer.read_view_rgb_pixels(TIP_VIEW)
            tv = sa.arr_bytes(tip_px)
            rows["tip.rgb"].update(tv)
            per_frame["tip.rgb"].append(hashlib.sha256(tv).hexdigest()[:16])
        if writer is not None and f % (FPS // FILM_FPS) == 0:
            _fw = time.perf_counter()
            writer.append_data(film_composite(aovs["rgb"], tip_px))
            film_wall[0] += time.perf_counter() - _fw
            film_frames[0] += 1
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
        # The two new sensor rows of round 6: the hoist tether's load cell (Joint.reaction, N)
        # and the container's touchdown switch (ContactSensor's latched state).
        rows["tension"].update(np.float32(r[26]).tobytes())
        rows["contact"].update(np.uint8(r[22] > 0.5).tobytes())
        for vals in px["imu_frame"]:                     # the hull IMU, every sample since the last frame
            rows["imu"].update(struct.pack("<7d", *vals))
        if OP_PNG and mode == "op":
            _op_png(f, aovs["rgb"], tip_px)
    wall = time.perf_counter() - wall0
    if OP_PNG and mode == "op" and _png_peak[0] > 0.0:
        print(f"png: maxswing {1e3 * _png_peak[0]:.0f} mm at t = {_png_peak[1]:.2f} s -> {OP_PNG}_maxswing.png")
    if writer is not None:
        writer.close()
        print(f"film: {film_frames[0]} frames -> {OP_FILM}, {1e3 * film_wall[0] / max(film_frames[0], 1):.1f} ms/frame "
              f"of composite + encode ({100.0 * film_wall[0] / wall:.0f}% of the run)")
    _L = np.asarray(log_rows)
    _rest = _L[_L[:, 0] > _L[-1, 0] - 1.0] if len(_L) > 60 else _L
    _margin = [[float((_L[:, 7 + j] - Q_MIN[j]).min()), float((Q_MAX[j] - _L[:, 7 + j]).min())] for j in range(3)]
    manifest = {
        "meta": {
            "threepp": getattr(tp, "__version__", "?"), "platform": platform.platform(),
            "scene": "crane_lift", "mode": mode, "frames": n, "warmup": WARMUP, "fps": FPS, "size": [W, H],
            "seed": SEED, "antiswing": antiswing_on[0], "amc": not NO_AMC, "clouds": not NO_CLOUDS,
            "ff": not NO_FF, "ff_mode": "off" if NO_FF else FF_MODE, "amc_kp": AMC_KP or "1/dt", "aa": aa_path(),
            "law": LAW if antiswing_on[0] else "off", "as_k": AS_K, "as_gamma": AS_GAMMA, "as_umax": AS_UMAX,
            "engage": ENGAGE if antiswing_on[0] else "off",
            "op": {"t_hold": T_HOLD, "t_xfer": T_XFER, "t_arrive": T_ARRIVE, "t_payout": T_PAYOUT,
                   "profile": XFER_PROFILE, "sweep_rad": D_TH, "xfer_peak_rate": XFER_V,
                   "boundaries": [T1, T2, T3, T4], "wire0": wire0, "hover_up": HOVER_UP},
            "geom": {"name": GEOM, "turbine": list(TURBINE_POS), "platform": PLATFORM.tolist(),
                     "land_off": LAND_OFF.tolist(),
                     "deck_rect": list(DECK_RECT), "tower_xz": TOWER_XZ.tolist(), "tower_r": TOWER_R,
                     "king_range": float(np.linalg.norm(PLATFORM[[0, 2]] - KING_XZ)),
                     "inside_near_edge": float(PLATFORM[0] - DECK_RECT[0]),
                     "clear_rect": list(CLEAR_RECT), "rail_margin": RAIL_MARGIN, "tower_margin": TOWER_MARGIN,
                     "doors": {k: list(v) for k, v in DOOR_BOX.items()},
                     "tip_to_tower_axis": float(np.linalg.norm(PLATFORM[[0, 2]] - TOWER_XZ))},
            "payload": {"model": PAYLOAD, "load": LOAD, "mass": CONT_MASS, "density": CONT_DENSITY,
                        "half_x": CONT_X, "half_z": CONT_Z, "height": CONT_H,
                        "substeps": PHYSX_SUB, "damping": PHYSX_DAMP,
                        "build_s": round(px["build_s"], 3), "max_stretch": px["max_stretch"],
                        "doors_excluded": list(px["doors_excluded"]),
                        "tower_min": None if px["tower_min"] > 1e8 else px["tower_min"],
                        "tower_hit_t": px["tower_hit_t"], "first_contact_t": px["first_contact_t"],
                        "rail_min": None if px["rail_min"] > 1e8 else px["rail_min"],
                        "rail_hit_t": px["rail_hit_t"], "door_min": None if px["door_min"] > 1e8 else px["door_min"],
                        "door_who": px["door_who"], "door_hit_t": px["door_hit_t"],
                        "rest_tilt_deg": float(_rest[:, 30].mean()) if _L.shape[1] > 30 else None,
                        "rest_anchor_y": float(_rest[:, 5].mean()),
                        "rest_anchor_above_level": float(_rest[:, 5].mean() - (GRATING_Y + CONT_H + SLING_H)),
                        "in_clear_rect": (in_clear_rect(px["body"].position, px["quat"])
                                          if PAYLOAD == "physx" and px["quat"] is not None else None)},
            "joint_margin": {"slew": _margin[0], "luff": _margin[1], "telescope": _margin[2]},
            "landed": bool(contact["landed"]), "t_land": contact["t_land"], "on_deck": contact["on_deck"],
            "land_offset": None if contact["xz"] is None else [float(contact["xz"][0]), float(contact["xz"][1])],
            "rate_sat_frames": rate_sat_frames[0], "acc_sat_frames": acc_sat_frames[0], "pos_sat_frames": sat_frames[0],
            "gpu": next((getattr(renderer, a) for a in ("gpu_name", "device_name") if hasattr(renderer, a)), ""),
            "pins": {"sim_time": True, "auto_exposure": bool(renderer.auto_exposure),
                     "fsr": bool(getattr(renderer, "fsr", False)), "dlss": bool(getattr(renderer, "dlss", False)),
                     "fsr_available": bool(getattr(renderer, "fsr_available", False)),
                     "dlss_available": bool(getattr(renderer, "dlss_available", False))},
            "events": n_events, "wall_seconds": round(wall, 2),
            "fan_scans": fan_seen[0], "fan_saw_load": fan_seen[1],
        },
        "per_frame": per_frame,
        "rows": {k: v.row() if v.frames else "absent" for k, v in rows.items()},
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # The LOG first, then the manifest. A run of this scene is seven minutes of GPU time and the
    # log is the part that cannot be recomputed from anything else; a manifest that fails to
    # serialise (a numpy bool in the meta did exactly this once) must not take the log with it.
    if OP_OUT and mode == "op":
        np.savez_compressed(OP_OUT[:-5] + ".npz", log=np.asarray(log_rows),
                            imu=np.asarray(px["imu_rows"], np.float64).reshape(-1, 7))
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=1, default=_jsonable)
    for k, v in manifest["rows"].items():
        print(f"  {k:14s} " + (v if isinstance(v, str) else f"{v['fnv']}  frames={v['frames']}"))
    print(f"{mode}: {n} frames after {WARMUP} warm-up, {1e3 * wall / n:.1f} ms/f, {n_events} events -> {out}")
    op_report()


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
