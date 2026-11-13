"""A vessel standing by an offshore wind turbine in a storm, or in working weather with a USV and
a drone on the job: a still or a short film, from one of five viewpoints.

    python turbine_storm.py --shot storm.png
    python turbine_storm.py --film storm.mp4 [--seconds 12] [--view drone] [--from 15] [--camera-look]
    python turbine_storm.py --weather calm --film calm.mp4

What is in it: R/V Gunnerus (threepp-data's Gunnerus.glb), the IEA 15 MW turbine of this folder in
its film fit-out (turbine_film.glb, see below), an overcast sky (Poly Haven's mud_road_puresky,
CC0, fetched once), and with --weather calm the Mariner USV (../usv) and the M350 (../uav).

The views (--view): mast (the default: a 139 degree fisheye over her wheelhouse, riding her),
drone (a slow arc off her quarter), beam (low off her lee side), platform (from the turbine's
boarding deck, down on her) and wide (the fisheye from upwind, the whole turbine in frame).
--from SECONDS starts a shot later in the same sea, so shots from several views cut together as
one stretch of time.

The sea. --weather storm is wind 20 m/s over 26 km of fetch on a 3 m swell, DRAWN AT TWICE ITS
HEIGHT (wave_scale 2): a look setting. A sum of linear waves at a measured spectrum is too gentle
to pinch into crests, and at twice the height it does; do not read wave heights off this scene.
--weather calm is 8 m/s at natural height. Either way the wind sea is not the same everywhere
(Ocean.set_wave_climate, patches a few hundred metres across): one strength everywhere shows the
wave tiles' repeat as rows of caps toward the horizon.

The foam. The ocean's own crest foam, here started where the drawn surface is well pinched
(THREEPP_OCEAN_FOLD_FOAM_J, set before the renderer reads it). At the hull and round the pile the
scene lays foam into the ocean's wake field itself: where the water rises or falls fast against
them, and in the storm from sources on her waterline and on the pile (her thrusters' races, the
sea breaking against her side, the run-up on the pile and the eddies it sheds), each swelling up
and dying away, thrown outward and carried off by the field's own flow.

The turbine's film fit-out is this folder's turbine with a boarding deck toward the vessel, a few
more fittings and its paint weathered (turbine_weather.py). In the storm its rotor faces the wind
and turns. Build the model once:

    blender --background --factory-startup --python build_turbine_blender.py -- \\
        --overrides turbine_film_overrides.json --out turbine_film.glb

Needs a Vulkan build with ray queries. Headless.
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # python/ (threepp)
sys.path.insert(0, os.path.dirname(_HERE))                     # examples/ (demo_common)
sys.path.insert(0, _HERE)

from demo_common import Encoder, cli_arg, parse_size           # noqa: E402
from turbine_out import out_dir                                # noqa: E402
from turbine_weather import weather                            # noqa: E402

STORM = cli_arg("--weather", "storm", str) != "calm"
SHOT = cli_arg("--shot", "", str)
FILM = cli_arg("--film", "", str)
SECONDS = cli_arg("--seconds", 12.0, float)
W, H = parse_size(cli_arg("--size", "1920x1080", str))
CAMERA_LOOK = "--camera-look" in sys.argv       # the film as a small camera would give it: 720p, a little soft
if not (SHOT or FILM):
    sys.exit(__doc__)

# The crest foam's look constants are read once, when the foam is first made.
os.environ.setdefault("THREEPP_OCEAN_CREST_FOAM_MAX", "1.0")
if STORM:
    os.environ.setdefault("THREEPP_OCEAN_FOLD_FOAM_J", "0.55")
    os.environ.setdefault("THREEPP_OCEAN_FOLD_FOAM_RAMP", "0.15")

import threepp as tp                                           # noqa: E402
from threepp.utils import fetch_file                           # noqa: E402

WIND, FETCH, SWELL, WAVE_SCALE, CHOP, CALMEST = (20.0, 26000.0, 3.0, 2.0, 1.1, 0.5) if STORM else \
                                                (8.0, 0.0, 1.0, 1.0, 0.55, 0.6)
HFOV = math.radians(139.0)                      # across the frame's width
TURBINE_AT = (17.0, 22.0)                       # pile axis, world XZ; the vessel lies at the origin, bow toward +Z
TURBINE_LIFT = 6.0                              # the platform stands clear of a sea drawn at twice its height
PILE_R = 5.3
LENGTH, HALF_L, HALF_B = 31.25, 15.0, 4.3       # Gunnerus
CAM, AIM = np.array([0.0, 25.0, -15.0]), np.array([0.0, 3.0, 13.0])   # the mast view, in the vessel's frame
# Steps before the first frame: the sea and its foam settle. --from SECONDS starts later in the same sea.
WARM = max(420, int(round(60.0 * cli_arg("--from", 7.0, float))))


def data_dir():
    """threepp_data's checkout: THREEPP_DATA_DIR, or beside the repository."""
    env = os.environ.get("THREEPP_DATA_DIR")
    if env and os.path.isdir(env):
        return env
    repo = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
    for name in ("threepp-data", "threepp_data"):
        cand = os.path.join(os.path.dirname(repo), name)
        if os.path.isdir(cand):
            return cand
    sys.exit("Gunnerus.glb is in threepp_data: set THREEPP_DATA_DIR to its checkout")


def need(path, how):
    if not os.path.isfile(path):
        sys.exit(f"{os.path.basename(path)} is generated. {how}")
    return path


# ---------------------------------------------------------------- the scene
canvas = tp.Canvas("turbine storm", width=W, height=H, vsync=False, headless=True)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic   # an overcast day: the white steel keeps its shading
renderer.tone_mapping_exposure = 2.6
# The viewpoints (--view NAME): what the eye stands on ("ship": it rides her, in her frame;
# "world": it stands still), eye, aim, and the lens: None = the 139 degree fisheye, else an
# ordinary lens of that vertical field of view (degrees). All but the mast's are ASSUMED places.
VIEWS = {
    "mast": ("ship", CAM, AIM, None),                                    # over her wheelhouse, looking forward and down
    "drone": ("world", (62.0, 42.0, -48.0), (6.0, 2.0, 10.0), 34.0),     # off her port quarter, high: the wash and the pile
    "beam": ("world", (78.0, 10.0, -30.0), (9.0, 7.0, 13.0), 27.0),      # low off her lee quarter, the pile beside her
    "platform": ("world", (13.0, 27.5, 16.5), (-1.0, 2.0, -2.0), 62.0),  # from the turbine's deck, down on her
    "wide": ("world", (-95.0, 10.0, -75.0), (17.0, 95.0, 22.0), None),   # from upwind, close in: the whole turbine, rotor and all
}
VIEW = cli_arg("--view", "mast", str)
if VIEW not in VIEWS:
    sys.exit(f"--view: one of {', '.join(VIEWS)}")
VIEW_ON, VIEW_EYE, VIEW_AIM, VIEW_FOV = VIEWS[VIEW]
VIEW_EYE, VIEW_AIM = np.array(VIEW_EYE, dtype=float), np.array(VIEW_AIM, dtype=float)
if VIEW_FOV is None:
    renderer.set_lens_distortion("fisheye")     # equidistant, k = 0: the camera's own fov sets its focal length

scene = tp.Scene()
sky = tp.RGBELoader().load(fetch_file("https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/4k/mud_road_puresky_4k.hdr",
                                      os.path.join(out_dir("cache"))))
scene.background = sky
scene.environment = sky

bed_mat = tp.MeshStandardMaterial()
bed_mat.color, bed_mat.roughness = 0x050505, 1.0
bed = tp.Mesh(tp.PlaneGeometry(6000.0, 6000.0), bed_mat)
bed.rotate_x(-math.pi / 2)
bed.position.y = -60.0
scene.add(bed)

SIZE = 2000.0
# 1024 vertices a side over 2 km, packed toward the vessel each step (warp_toward below): about
# 0.3 m cells round her, metres at the horizon, where a wave is under a pixel anyway. A uniform
# 0.5 m grid (4096 a side) draws the same picture four times slower, and the fisheye draws the
# scene five times over: 118 ms a frame against 28 ms (RTX 5080, 1920x1080).
ocean = tp.Ocean(size=SIZE, resolution=1024, wind_speed=WIND, fetch=FETCH)
p = ocean.params
p.short_wave_spectrum = p.wave_reflections = True
p.swell_height, p.swell_period, p.swell_theta = SWELL, 11.0, p.wind_theta
p.wave_scale, p.choppiness = WAVE_SCALE, CHOP
ocean.wake_field.resolution, ocean.wake_field.patches = 1024, 2
ocean.wake_field.lane_life, ocean.wake_field.aeration_life = 3.0, 2.5
ocean.material.attenuation_color = tp.Color(0.105, 0.13, 0.115)   # a grey North Sea under this sky


def patchy(lo, cells=6, n=256, seed=3):
    """The wind sea's gain over the ocean, lo..1 in patches a few hundred metres across."""
    rng = np.random.default_rng(seed)

    def lattice(k):
        g = rng.random((k + 1, k + 1))
        u = np.linspace(0, k, n, endpoint=False)
        i = np.floor(u).astype(int)
        f = u - i
        f = f * f * (3 - 2 * f)
        r = g[:, i] * (1 - f) + g[:, i + 1] * f
        return r[i, :] * (1 - f)[:, None] + r[i + 1, :] * f[:, None]
    v = (lattice(cells) + 0.5 * lattice(2 * cells)) / 1.5
    v = np.clip((v - 0.30) / 0.40, 0.0, 1.0)
    return (lo + (1.0 - lo) * v * v * (3 - 2 * v)).astype(np.float32)


gain = patchy(CALMEST)
ocean.set_wave_climate(np.stack([np.ones_like(gain), gain, gain, np.zeros_like(gain)], axis=2),
                       -SIZE / 2, -SIZE / 2, SIZE, SIZE)          # before the first render: the foam follows it
scene.add(ocean)

ship = tp.Group()
hull = tp.GLTFLoader().load(os.path.join(data_dir(), "models", "gltf", "Gunnerus.glb")).scene
box = tp.Box3()
box.set_from_object(hull)
size = box.get_size()
if size.x > size.z:
    hull.rotate_y(-math.pi / 2)
hull.rotate_y(math.pi)
k = LENGTH / max(size.x, size.y, size.z)
hull.scale.set(k, k, k)
ship.add(hull)
scene.add(ship)

turbine = tp.GLTFLoader().load(need(os.path.join(_HERE, "turbine_film.glb"),
                                    "Build it once (see this file's header).")).scene
turbine.position.set(TURBINE_AT[0], TURBINE_LIFT, TURBINE_AT[1])
for name in ("marine_growth", "anode_cage"):    # what grows and hangs below the waterline stays there
    node = turbine.get_object_by_name(name)
    if node is not None:
        node.position.y -= TURBINE_LIFT
turbine.traverse(weather)
scene.add(turbine)

# The model is built in its inspection state (rotor locked, blades feathered, nacelle along +X).
# In the storm she is at work: nacelle turned so the rotor faces upwind, blades pitched as the
# reference turbine has them at 20 m/s (about 18 degrees), rotor at its rated 7.56 rpm.
ROTOR = turbine.get_object_by_name("rotor") if STORM else None
ROTOR_RATE = 7.56 * 2.0 * math.pi / 60.0         # rad/s
if ROTOR is not None:
    turbine.get_object_by_name("nacelle").rotation.y = math.atan2(math.sin(p.wind_theta), -math.cos(p.wind_theta))
    for i in (1, 2, 3):
        turbine.get_object_by_name(f"pitch_{i}").rotate_y(math.radians(18.0 - 90.0))

wash = tp.VesselWash(ocean, 0, length=LENGTH, half_length=HALF_L, half_beam=HALF_B, mass=5.0e5, draft=2.7, size=120.0)
# How she sits. SINK: metres under her design waterline (a LOOK setting for the storm: on a sea drawn
# at twice its height a hull that follows the mean of the water under her rides with her bottom
# showing). DECK_CAP: the ocean's plane for her is laid this far over her waterline, just under her
# working deck, so the sea stands as high against her side as it comes and is held down only where
# it would come up through the deck (laid on her waterline it pressed every higher crest down
# round her, a square pit in the sea at her stern).
SINK = 0.5 if STORM else 0.0
DECK_CAP = 1.0
POSE_Z = (-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0)   # stations along her the sea is read at
HULL_PTS = [(HALF_B * 1.05 * math.sin(a), HALF_L * 1.02 * math.cos(a)) for a in np.linspace(0, 2 * math.pi, 18, endpoint=False)]
PILE_PTS = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
DOWNWIND = (math.cos(p.wind_theta), math.sin(p.wind_theta))
WASH_GAIN = 0.8 if STORM else 1.0
# In the storm she holds station on her two propellers and her bow thruster against the wind and
# the waves' drift, and her wash boils: sources on her waterline and on the pile, each swelling up
# and dying away over a few seconds. All of these figures are ASSUMED, set by eye.
HOLD = 4.0e4                                     # N, the mean force she holds against
PROPS = (("prop_port", -1.6, -2.0, -13.0), ("prop_stbd", 1.6, -2.0, -13.0))   # x, depth, z in her frame; discs of 1 m radius
BOW_THRUSTER = (0.0, -1.6, 10.5)
BOILS, PILE_BOILS = (70, 22) if STORM else (0, 0)
BOIL_LIFE = (3.0, 6.0)                           # s
BOIL_FOAM, BOIL_AIR, BOIL_TURB = 0.85, 0.8, 4.0  # what a boil lays at its height: foam, air, turbulence (m/s)
BOIL_THROW, PILE_THROW = 3.0, 2.6                # m/s: outward from her side at most, and downwind off the pile
if STORM:
    wash.air = 4.0                               # a displacement hull's propellers take little air down as computed
    wf = ocean.wake_field                        # a wash that lasts, and that the field's flow tears rather than spreads
    wf.foam_life, wf.foam_thinning, wf.aeration_life = 14.0, 6.0, 9.0
    wf.turbulence_life, wf.velocity_life, wf.spread, wf.eddy_texels = 8.0, 6.0, 0.01, 30.0
boil_rng = np.random.default_rng(7)
boils = {}                                       # slot -> (x, z, vx, vz, radius, strength, ends, began)
pile_boils = {}


def swell(now, began, ends):
    """A boil's strength over its life: up from nothing and back to nothing (sin squared), so the
    foam it lays grows out of the water instead of appearing on it."""
    return math.sin(math.pi * min(max((now - began) / max(ends - began, 1e-3), 0.0), 1.0)) ** 2


mariner = drone = None
if not STORM:
    mariner = tp.GLTFLoader().load(need(os.path.join(_HERE, "..", "usv", "mariner.glb"),
                                        "Build it: examples/usv/build_mariner_blender.py.")).scene
    scene.add(mariner)
    mariner_wash = tp.VesselWash(ocean, 1, length=5.98, half_length=2.9, half_beam=1.0, mass=2400.0, draft=0.5, size=120.0)
    mariner_wash.air = 0.25                     # as computed, her jet leaves a grey smear seen from above
    drone = tp.GLTFLoader().load(need(os.path.join(_HERE, "..", "uav", "m350.glb"),
                                      "Build it: examples/uav/build_m350_blender.py.")).scene
    scene.add(drone)
    disc = tp.MeshStandardMaterial()            # a turning propeller is a blur, not two frozen blades
    disc.color, disc.roughness, disc.transparent, disc.opacity = 0x202020, 0.6, True, 0.28
    for name in ("prop_fr", "prop_fl", "prop_rl", "prop_rr"):
        prop = drone.get_object_by_name(name)
        if prop is not None:
            blur = tp.Mesh(tp.CylinderGeometry(0.267, 0.267, 0.006, 40), disc)
            blur.position.copy(prop.position)
            prop.parent.add(blur)
            prop.visible = False

camera = tp.PerspectiveCamera(math.degrees(2 * math.atan((H / 2) / ((W / 2) / (HFOV / 2)))) if VIEW_FOV is None else VIEW_FOV,
                              W / H, 0.1, 4000)
scene.add(camera)

# ---------------------------------------------------------------- one step
pose = np.zeros(3)          # the vessel's heave, pitch, roll
mariner_pose = np.zeros(2)  # heave, pitch
gap = {}                    # the water's height against the hull and the pile, last step
t = 0.0


def step(dt=1.0 / 60.0):
    global pose, mariner_pose, t
    t += dt
    sh = ocean.sample_height
    # The vessel rides the sea: a first-order follow of the surface under her.
    # Seven stations along her length (the sea's mean height and its slope under her, by least
    # squares) and one either side amidships.
    hz = [sh(0.0, z) for z in POSE_Z]
    slope = sum(z * h for z, h in zip(POSE_Z, hz)) / sum(z * z for z in POSE_Z)
    hp, hm = sh(3.5, 0.0), sh(-3.5, 0.0)
    want = np.array([sum(hz) / len(hz), 0.9 * math.atan(-slope), -0.6 * math.atan2(hp - hm, 7.0)])
    pose = pose + (want - pose) * min(1.0, dt / 0.8)
    ship.position.y = float(pose[0]) - SINK
    ship.rotation.x, ship.rotation.z = float(pose[1]), -float(pose[2])
    # Her plane as the ocean takes it (DECK_CAP over her waterline, see above): pitch is +bow up and
    # roll +starboard (+X) up there, while rotation.x = +a puts her +Z end DOWN and rotation.z = -a
    # puts her +X side DOWN, so both signs turn over.
    ocean.hull_exclusion.set_pose(0.0, 0.0, float(pose[0]) - SINK + DECK_CAP, 0.0, -float(pose[1]), -float(pose[2]),
                                  HALF_L, HALF_B)
    if ROTOR is not None:
        ROTOR.rotate_x(ROTOR_RATE * dt)

    ocean.clear_wake_sources()
    thrusters = []
    if STORM:
        # What she has to hold comes and goes with the gusts and the wave groups.
        hold = HOLD * (0.55 + 0.6 * (0.5 + 0.5 * math.sin(0.21 * t)) * (0.5 + 0.5 * math.sin(0.057 * t + 1.0))
                       + 0.35 * math.sin(0.83 * t + 2.0) ** 2)
        fx, fz = -hold * DOWNWIND[0], -hold * DOWNWIND[1]
        for name, x, y, z in PROPS:              # her length's share, and half the sideways one by their rudders
            thrusters.append((name, (x, float(pose[0]) - SINK + y, z), (0.25 * fx, 0.0, 0.5 * fz), 1.0, False))
        bx, by, bz = BOW_THRUSTER                # the other half of the sideways share, athwartships at her bow
        thrusters.append(("bow_thruster", (bx, float(pose[0]) - SINK + by, bz), (0.5 * fx, 0.0, 0.0), 0.5, True))
    wash.update(dt, (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, 0.0), thrusters)
    for slot in range(BOILS):
        b = boils.get(slot)
        if b is None or t >= b[6]:
            # A point on her waterline: more of them at her stern, where the propellers work,
            # and on her weather side, where the sea strikes her.
            a = boil_rng.uniform(0.0, 2.0 * math.pi)
            for _ in range(3):                   # the weights, by rejection
                ox, oz = math.sin(a), math.cos(a)
                weather = max(0.0, -(ox * DOWNWIND[0] + oz * DOWNWIND[1]))
                if boil_rng.random() < 0.35 + 0.4 * max(0.0, -oz) ** 2 + 0.35 * weather:
                    break
                a = boil_rng.uniform(0.0, 2.0 * math.pi)
            ox, oz = math.sin(a), math.cos(a)
            # Foam is MADE at her side and nowhere else: every source lies on her waterline
            # (within a metre or so of it), and what it makes is thrown outward and carried
            # off by the water's own flow in the wake field. How hard it is thrown differs
            # round her and wanders with time, which is what gives the wash its lobes.
            lobe = 0.25 + 0.75 * (0.5 + 0.5 * math.sin(3.0 * a + 0.11 * t + 1.3)) * (0.5 + 0.5 * math.sin(2.0 * a - 0.07 * t))
            out = boil_rng.uniform(0.2, 1.2)
            nx, nz = ox / HALF_B, oz / HALF_L    # the outline's outward normal
            nn = math.hypot(nx, nz)
            nx, nz = nx / nn, nz / nn
            s = boil_rng.uniform(0.5, 1.0)
            throw = BOIL_THROW * lobe * boil_rng.uniform(0.5, 1.0)
            b = boils[slot] = (HALF_B * ox + nx * out, HALF_L * oz + nz * out,
                               nx * throw + 0.8 * DOWNWIND[0], nz * throw + 0.8 * DOWNWIND[1],
                               boil_rng.uniform(0.56, 1.36), s, t + boil_rng.uniform(*BOIL_LIFE), t)
        x, z, vx, vz, r, s, t1, t0 = b
        s *= swell(t, t0, t1)
        wash.put(f"b{slot}", (x, 0.0, z), (vx, 0.0, vz), r, foam=BOIL_FOAM * s, aeration=BOIL_AIR * s,
                 turbulence=BOIL_TURB * s)
    for i, (x, z) in enumerate(HULL_PTS):       # foam where the water rises or falls fast against the hull
        g = sh(x, z) - (pose[0] - SINK - z * math.sin(pose[1]) - x * math.sin(pose[2]))
        v = (g - gap.get(i, g)) / dt
        gap[i] = g
        s = WASH_GAIN * min(max((abs(v) - 0.35) / 1.4, 0.0), 1.0)
        if s > 0.02:
            n = math.hypot(x / HALF_B, z / HALF_L) or 1.0
            wash.put(f"h{i}", (x, 0.0, z), (0.7 * s * x / HALF_B / n, 0.0, 0.7 * s * z / HALF_L / n), 1.3,
                     foam=0.9 * s, aeration=0.5 * s, turbulence=0.7 * s)
    g = sh(TURBINE_AT[0] + PILE_R + 1.0, TURBINE_AT[1])
    v = (g - gap.get("pile", g)) / dt
    gap["pile"] = g
    sp = WASH_GAIN * min(max((abs(v) - 0.3) / 1.6, 0.0), 1.0)
    for i, (cx, cz) in enumerate(PILE_PTS):     # and where the sea runs past the pile: its weather side and its lee
        s = sp * (0.35 + 0.65 * abs(cx * DOWNWIND[0] + cz * DOWNWIND[1]))
        if s > 0.02:
            wash.put(f"p{i}", (TURBINE_AT[0] + cx * (PILE_R + 0.9), 0.0, TURBINE_AT[1] + cz * (PILE_R + 0.9)),
                     (1.2 * s * DOWNWIND[0], 0.0, 1.2 * s * DOWNWIND[1]), 1.4, foam=0.9 * s, aeration=0.5 * s, turbulence=0.8 * s)
    # The pile stands in the sea's way. A crest runs up its weather side and falls back as
    # white water; the water that has passed it is torn into eddies shed from one side and
    # then the other, and they carry their foam and air off to leeward in a trail.
    for slot in range(PILE_BOILS):
        b = pile_boils.get(slot)
        if b is None or t >= b[6]:
            cross = (-DOWNWIND[1], DOWNWIND[0])
            if boil_rng.random() < 0.3:          # run-up: against the weather side
                a = boil_rng.uniform(-1.0, 1.0)
                dx, dz = -math.cos(a) * DOWNWIND[0] + math.sin(a) * cross[0], -math.cos(a) * DOWNWIND[1] + math.sin(a) * cross[1]
                px, pz = (PILE_R + 0.6) * dx, (PILE_R + 0.6) * dz
                vx, vz = 0.8 * dx + math.sin(a) * cross[0], 0.8 * dz + math.sin(a) * cross[1]
                r, s = boil_rng.uniform(0.4, 0.72), boil_rng.uniform(0.6, 1.0)
            else:                                # the lee: made at the pile's sides, shed from one and then the other
                side = 1.0 if int(t / 2.3 + slot) % 2 == 0 else -1.0
                a = side * boil_rng.uniform(0.9, 1.9)          # round toward the lee from the side it leaves by
                dx, dz = -math.cos(a) * DOWNWIND[0] + math.sin(a) * cross[0], -math.cos(a) * DOWNWIND[1] + math.sin(a) * cross[1]
                px, pz = (PILE_R + 0.7) * dx, (PILE_R + 0.7) * dz
                carry = PILE_THROW * boil_rng.uniform(0.6, 1.0)  # the flow past the pile takes it off downwind
                vx, vz = carry * DOWNWIND[0] - 0.6 * side * cross[0], carry * DOWNWIND[1] - 0.6 * side * cross[1]
                r, s = boil_rng.uniform(0.48, 1.2), boil_rng.uniform(0.5, 1.0)
            b = pile_boils[slot] = (TURBINE_AT[0] + px, TURBINE_AT[1] + pz, vx, vz, r, s, t + boil_rng.uniform(*BOIL_LIFE), t)
        x, z, vx, vz, r, s, t1, t0 = b
        s *= (0.45 + 0.55 * sp) * swell(t, t0, t1)   # more when the sea is working against the pile
        wash.put(f"t{slot}", (x, 0.0, z), (vx, 0.0, vz), r, foam=BOIL_FOAM * s, aeration=BOIL_AIR * s,
                 turbulence=BOIL_TURB * s)

    if mariner is not None:
        # The Mariner comes in from ahead at 4 m/s, bow toward the vessel (her model: X forward).
        mx, mz, fx, fz = -13.0, 92.0 - 4.0 * t, 0.0, -1.0
        hb, hs, hc = sh(mx + 2.2 * fx, mz + 2.2 * fz), sh(mx - 2.2 * fx, mz - 2.2 * fz), sh(mx, mz)
        want = np.array([(hb + hs + hc) / 3.0, math.atan2(hb - hs, 4.4)])
        mariner_pose = mariner_pose + (want - mariner_pose) * min(1.0, dt / 0.35)
        mariner.position.set(mx, float(mariner_pose[0]) - 0.30, mz)
        mariner.rotation.y, mariner.rotation.z = math.atan2(-fz, fx), float(mariner_pose[1]) + 0.05
        mariner_wash.update(dt, (mx, 0.0, mz), (fx, 0.0, fz), (4.0 * fx, 0.0, 4.0 * fz),
                            [("jet", (mx - 2.9 * fx, -0.25, mz - 2.9 * fz), (1500.0 * fx, 0.0, 1500.0 * fz), 0.14, True)])
        # The M350 holds station off the bow: a lean into the wind, and a position hold's small corrections.
        lean = math.radians(1.1 * WIND)
        gust = 0.5 * math.sin(0.43 * t) + 0.3 * math.sin(1.13 * t + 1.0) + 0.2 * math.sin(2.9 * t + 2.0)
        drone.position.set(-5.0 + 0.05 * WIND * (0.6 * math.sin(0.31 * t) + 0.4 * math.sin(0.83 * t + 1.0)),
                           21.0 + 0.03 * WIND * (0.6 * math.sin(0.47 * t + 0.4) + 0.4 * math.sin(1.31 * t)),
                           -4.0 + 0.05 * WIND * (0.6 * math.sin(0.37 * t + 2.0) + 0.4 * math.sin(0.97 * t)))
        yaw = math.radians(-100.0) + 0.05 * math.sin(0.6 * t)
        drone.rotation.y = yaw
        drone.rotation.z = lean * math.cos(p.wind_theta + yaw) * (1.0 + 0.25 * gust) + 0.02 * math.sin(3.7 * t)
        drone.rotation.x = lean * math.sin(p.wind_theta + yaw) * (1.0 + 0.25 * gust) + 0.02 * math.sin(4.3 * t + 1.0)

    # The camera rides the vessel.
    cx, sx, cz, sz = math.cos(pose[1]), math.sin(pose[1]), math.cos(-pose[2]), math.sin(-pose[2])
    R = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    if VIEW_ON == "ship":
        eye = R @ VIEW_EYE + [0.0, pose[0] - SINK, 0.0]
        at = R @ VIEW_AIM + [0.0, pose[0] - SINK, 0.0]
        up = R @ [0.0, 1.0, 0.0]
    else:
        eye, at, up = VIEW_EYE, VIEW_AIM, np.array([0.0, 1.0, 0.0])
        if VIEW == "drone":
            # A drone does not hang still: a slow arc round the two of them, sinking a little
            # and closing in, from her port quarter toward her beam.
            s = max(t - WARM / 60.0, 0.0)
            a = math.atan2(VIEW_EYE[2] - at[2], VIEW_EYE[0] - at[0]) + math.radians(3.2) * s
            r = math.hypot(VIEW_EYE[0] - at[0], VIEW_EYE[2] - at[2]) * (1.0 - 0.012 * s)
            eye = np.array([at[0] + r * math.cos(a), VIEW_EYE[1] - 0.7 * s, at[2] + r * math.sin(a)])
    camera.position.set(*eye.tolist())
    camera.up.set(*up.tolist())
    camera.look_at(tp.Vector3(*at.tolist()))

    renderer.sim_time = t
    ocean.warp_toward(0.0, 0.0, 0.15)
    renderer.render(scene, camera)


for _ in range(WARM):
    step()
if SHOT:
    from PIL import Image
    Image.fromarray(renderer.read_pixels()).save(SHOT)
    print("wrote", SHOT)
if FILM:
    # --camera-look: 720p, a slight blur, a little colour out, faint grain, as a small camera gives it.
    look = "scale=1280:720:flags=bicubic,gblur=sigma=0.75,eq=saturation=0.9,noise=alls=2:allf=t" if CAMERA_LOOK else None
    enc = Encoder(FILM, W, H, 30, crf=25 if CAMERA_LOOK else 20, faststart=True, vf=look)
    for i in range(int(SECONDS * 60)):
        step()
        if i % 2 == 1:
            enc.send(renderer.read_pixels())
    enc.close()
    print("wrote", FILM)
