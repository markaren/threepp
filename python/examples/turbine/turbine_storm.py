"""A vessel standing by an offshore wind turbine, seen through a 139 degree fisheye from above her
wheelhouse: a still or a short film of the sea in a storm, or in working weather with a USV and a
drone on the job.

    python turbine_storm.py --shot storm.png
    python turbine_storm.py --film storm.mp4 [--seconds 12] [--camera-look]
    python turbine_storm.py --weather calm --film calm.mp4

What is in it: R/V Gunnerus (threepp-data's Gunnerus.glb), the IEA 15 MW turbine of this folder in
its film fit-out (turbine_film.glb, see below), an overcast sky (Poly Haven's mud_road_puresky,
CC0, fetched once), and with --weather calm the Mariner USV (../usv) and the M350 (../uav).

The sea. --weather storm is wind 20 m/s over 26 km of fetch on a 3 m swell, DRAWN AT TWICE ITS
HEIGHT (wave_scale 2): a look setting. A sum of linear waves at a measured spectrum is too gentle
to pinch into crests, and at twice the height it does; do not read wave heights off this scene.
--weather calm is 8 m/s at natural height. Either way the wind sea is not the same everywhere
(Ocean.set_wave_climate, patches a few hundred metres across): one strength everywhere shows the
wave tiles' repeat as rows of caps toward the horizon.

The foam. Crest foam along the crest (THREEPP_OCEAN_CREST_LINE_FOAM), started where the drawn
surface is well pinched (THREEPP_OCEAN_FOLD_FOAM_J), set here before the renderer reads them. At
the hull and round the pile the scene lays foam into the ocean's wake field itself: where the water
rises or falls fast against the hull, and where the sea runs past the pile.

The turbine's film fit-out is this folder's turbine with a few more fittings and shells dense
enough to colour per vertex (run-off, splash-zone grime, rust runs: weather() below). Build it once:

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

STORM = cli_arg("--weather", "storm", str) != "calm"
SHOT = cli_arg("--shot", "", str)
FILM = cli_arg("--film", "", str)
SECONDS = cli_arg("--seconds", 12.0, float)
W, H = parse_size(cli_arg("--size", "1920x1080", str))
CAMERA_LOOK = "--camera-look" in sys.argv       # the film as a small camera would give it: 720p, a little soft
if not (SHOT or FILM):
    sys.exit(__doc__)

# The crest foam's look constants are read once, when the foam is first made.
os.environ.setdefault("THREEPP_OCEAN_CREST_LINE_FOAM", "1")
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
CAM, AIM = np.array([0.0, 25.0, -15.0]), np.array([0.0, 3.0, 13.0])   # in the vessel's frame
WARM = 420                                      # steps before the first frame: the sea and its foam settle


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


def weather(o):
    """Nothing at sea stays the colour it was painted: run-off streaks, chalking, splash-zone grime,
    rust runs below fittings and dried salt on the yellow, grime toward the foot of the tower. Per
    vertex, on the big shells only."""
    if not isinstance(o, tp.Mesh):
        return
    c = o.material.color
    yellow = c.r > 0.55 and c.g > 0.45 and c.b < 0.30
    grey = abs(c.r - c.g) < 0.06 and abs(c.g - c.b) < 0.08 and c.r > 0.45
    P = np.asarray(o.geometry.get_attribute("position"), dtype=np.float64).reshape(-1, 3)
    if not (yellow or grey) or len(P) < 3000:
        return
    y, az = P[:, 1], np.arctan2(P[:, 2], P[:, 0])
    rng = np.random.default_rng(11 if yellow else 23)

    def streaks(count, yscale):                 # narrow round the shell, long down it
        n = np.zeros_like(y)
        for _ in range(count):
            m, ph = int(rng.integers(9, 90)), rng.uniform(0, 6.28)
            yk, yp = rng.uniform(0.02, yscale), rng.uniform(0, 6.28)
            n += np.sin(m * az + ph + 0.6 * np.sin(yk * y + yp)) * np.sin(yk * 0.5 * y + yp) / (1 + 0.03 * m)
        return n / math.sqrt(count)
    run, chalk = streaks(14, 0.25), streaks(6, 0.9)
    col = np.tile([c.r, c.g, c.b], (len(P), 1))
    col *= (1.0 - 0.16 * np.clip(run, 0, None) - 0.07 * np.clip(-run, 0, None) + 0.06 * chalk)[:, None]
    if yellow:
        splash = np.clip((6.0 - (y + 1.5 * chalk)) / 9.0, 0.0, 1.0) ** 1.4
        col = col * (1.0 - 0.8 * splash)[:, None] + np.array([0.16, 0.15, 0.07]) * (0.8 * splash)[:, None]
        rust = np.zeros_like(y)
        for a0 in rng.uniform(-math.pi, math.pi, 22):
            top, ln = rng.uniform(2.0, 14.5), rng.uniform(1.5, 6.0)
            d = np.angle(np.exp(1j * (az - a0)))
            rust = np.maximum(rust, np.exp(-0.5 * (d / 0.012) ** 2) * np.clip((top - y) / 0.3, 0, 1)
                              * np.clip(1 - (top - y) / ln, 0, 1))
        col = col * (1.0 - 0.55 * rust)[:, None] + np.array([0.30, 0.13, 0.05]) * (0.55 * rust)[:, None]
        salt = np.clip(1 - np.abs(y - 6.5) / 4.0, 0, 1) * np.clip(chalk + 0.3, 0, 1)
        col = col * (1.0 - 0.18 * salt)[:, None] + 0.135 * salt[:, None]
    else:
        low = np.clip(1.0 - (y - y.min()) / 25.0, 0.0, 1.0)
        col *= (1.0 - 0.10 * low * (0.6 + 0.4 * np.clip(run, 0, None)))[:, None]
        col[:, 2] *= 1.0 - 0.05 * low
    o.geometry.set_attribute("color", np.clip(col, 0.0, 1.0).astype(np.float32))
    o.material.vertex_colors = True
    o.material.color = tp.Color(1.0, 1.0, 1.0)
    o.material.roughness = max(o.material.roughness, 0.7)


# ---------------------------------------------------------------- the scene
canvas = tp.Canvas("turbine storm", width=W, height=H, vsync=False, headless=True)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.Linear   # a high-key overcast day, the white steel near clipping
renderer.tone_mapping_exposure = 3.0
renderer.set_lens_distortion("fisheye")         # equidistant, k = 0: the camera's own fov sets its focal length

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
p.hs_fix = p.short_waves = p.short_wave_spectrum = True
p.crest_foam = p.chop_normals = p.fold_foam = p.wave_reflections = True
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

wash = tp.VesselWash(ocean, 0, length=LENGTH, half_length=HALF_L, half_beam=HALF_B, mass=5.0e5, draft=2.7, size=120.0)
HULL_PTS = [(HALF_B * 1.05 * math.sin(a), HALF_L * 1.02 * math.cos(a)) for a in np.linspace(0, 2 * math.pi, 18, endpoint=False)]
PILE_PTS = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]
DOWNWIND = (math.cos(p.wind_theta), math.sin(p.wind_theta))
WASH_GAIN = 0.8 if STORM else 1.0

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

camera = tp.PerspectiveCamera(math.degrees(2 * math.atan((H / 2) / ((W / 2) / (HFOV / 2)))), W / H, 0.1, 4000)
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
    hb, hs, hp, hm, hc = sh(0.0, -11.0), sh(0.0, 11.0), sh(3.5, 0.0), sh(-3.5, 0.0), sh(0.0, 0.0)
    want = np.array([0.8 * (hb + hs + hp + hm + hc) / 5.0, 0.8 * math.atan2(hb - hs, 22.0), -0.6 * math.atan2(hp - hm, 7.0)])
    pose = pose + (want - pose) * min(1.0, dt / 0.8)
    ship.position.y = float(pose[0])
    ship.rotation.x, ship.rotation.z = float(pose[1]), -float(pose[2])
    ocean.hull_exclusion.set_pose(0.0, 0.0, float(pose[0]), 0.0, float(pose[1]), float(pose[2]), HALF_L, HALF_B)

    ocean.clear_wake_sources()
    wash.update(dt, (0.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 0.0, 0.0))
    for i, (x, z) in enumerate(HULL_PTS):       # foam where the water rises or falls fast against the hull
        g = sh(x, z) - (pose[0] + z * math.sin(pose[1]) - x * math.sin(pose[2]))
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
    eye = R @ CAM + [0.0, pose[0], 0.0]
    at = R @ AIM + [0.0, pose[0], 0.0]
    up = R @ [0.0, 1.0, 0.0]
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
