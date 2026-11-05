"""Radar and sonar on the Mariner in Alesund harbour: what a sea drone sees, above and below.

The Mariner runs a loop round the harbour basin south of the town on the real
terrain (a Kartverket height pack with the town's buildings, quays and moored
boats from OpenStreetMap), following the 14 m depth contour so the sonar has a
seabed to look at. She carries the two sensors a harbour-inspection USV carries:

  * a MARINE RADAR on her mast (threepp.MarineRadar): an X-band antenna turning
    at 24 rpm. Every spoke is the radar equation over the rays of the scene:
    the NRL sea-clutter model on the rendered waves, per-material backscatter,
    two-ray multipath over the sea (low targets fade with range), the
    antenna's azimuth pattern and its sidelobes, kTBF noise, and the operator's
    gain, sea (STC) and rain (FTC) controls. The scope top right is its PPI;
    the moving dots are the other USVs, the bright points the buoys' radar
    reflectors, the coast the town's waterfront.
  * a FORWARD-LOOKING IMAGING SONAR under her bow (threepp.ImagingSonar, an
    Oculus M750d at 750 kHz, tilted down): the sonar equation with
    Francois-Garrison absorption, Lambert + facet-glint seabed scattering,
    coherent beamforming through a tapered 512-beam array (speckle, sidelobe
    arcs, curved range lines), the chirp's matched filter, reverberation, noise
    and the sea surface as a mirror. The fan bottom right is its display. Along
    the north shore she passes a lost Otter X lying on its side on the seabed.

Both sensors report ground truth next to the picture: every radar range cell
and every sonar (beam, bin) carries the instance id of what made its echo
(L shows it as colour). That label comes from the renderer, not a detector.

    python usv_radar_sonar.py                       # drive; the loop runs on autopilot until you steer
    python usv_radar_sonar.py --shot 12 --out rs.png  # headless: run the loop 12 s, write the frame + both displays

Options: --pack <dir> (default geodata/aalesund), --sea-state 3, --rain (a squall
over the basin), --array (a 4 ft open array instead of the Mariner's radome),
--start <metres along the loop>, --size 1600x900.

Keys: W/S throttle, A/D steer (either hands the helm back from the autopilot),
P autopilot, N north-up / head-up, L ground-truth labels, M sonar multipath,
1-4 radar range 0.25 / 0.5 / 0.75 / 1 nmi.

Needs a Vulkan build of threepp, mariner.glb and otterx.glb / otter.glb (see
build_*_blender.py), and the terrain pack (scripts/geodata/fetch_norway_terrain.py).
"""
import math
import os
import sys
import tempfile
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXAMPLES = os.path.dirname(_HERE)
sys.path.insert(0, os.path.dirname(_EXAMPLES))                 # python/ (threepp)
sys.path.insert(0, _EXAMPLES)                                  # examples/ (demo_common)
sys.path.insert(0, os.path.join(_EXAMPLES, "rigs"))            # examples/rigs/ (usv_rig)

import threepp as tp
from demo_common import cli_arg, parse_size, resize_handler, write_radiance_hdr
from usv_rig import StripHull, Wake, Wash, attitude, drive_for, rot_y, rot_z, wrap

if not tp.HAS_VULKAN:
    print("The sensors trace the Vulkan renderer's scene: this needs a Vulkan build of threepp.")
    sys.exit(0)

REPO = os.path.dirname(os.path.dirname(_EXAMPLES))
PACK = cli_arg("--pack", os.path.join(REPO, "geodata", "aalesund"), str)
for need in ("mariner.glb", "otterx.glb", "otter.glb"):
    if not os.path.exists(os.path.join(_HERE, need)):
        print(f"{need} is generated: build it once from this folder, see build_{need[:-4]}_blender.py")
        sys.exit(1)
if not os.path.isfile(os.path.join(PACK, "region.json")):
    print(f"no terrain pack at {PACK}: fetch one with scripts/geodata/fetch_norway_terrain.py "
          "(see geodata/README.md), or pass --pack <dir>")
    sys.exit(1)

SHOT_T = cli_arg("--shot", 0.0, float) if "--shot" in sys.argv else 0.0
HEADLESS = SHOT_T > 0.0
OUT = cli_arg("--out", "usv_radar_sonar.png", str)
W, H = parse_size(cli_arg("--size", "1600x900", str))
SEA_STATE = cli_arg("--sea-state", 3.0, float)
RAIN = "--rain" in sys.argv
OPEN_ARRAY = "--array" in sys.argv
START_S = cli_arg("--start", 760.0, float)
CONTOUR_DEPTH = 14.0          # m: the loop's depth contour
SPEED = 4.0                   # m/s on the autopilot (about 8 kn)
SUBSTEPS_HZ = 240.0
SEA_SIZE = 4000.0             # the ocean sheet: past the radar's 1 nmi in every direction

# Stable ids (the Ids AOV's and the sensors' labels). High, out of the way of the
# ids the renderer assigns on its own to the terrain's meshes.
ID_SEA, ID_OWN, ID_OTTERX, ID_OTTER, ID_BUOY, ID_WRECK, ID_ROCK = 60001, 60002, 60010, 60011, 60020, 60030, 60040
N_BUOYS = 4                   # ID_BUOY + k
# What the survey built, labelled by kind: pier decks and moored boats, buildings, road bridges.
ID_PROPS, ID_BUILDINGS, ID_BRIDGES = 61000, 61001, 61002

# --------------------------------------------------------------------------- #
#  Scene: sky, the real harbour, one ocean
# --------------------------------------------------------------------------- #
canvas = tp.Canvas("threepp - radar and sonar on the Mariner, Alesund", width=W, height=H, vsync=False,
                   headless=HEADLESS)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 0.7
renderer.set_sensor_only_surfaces(False)
ui = tp.ImguiContext(canvas, renderer) if (tp.HAS_IMGUI and not HEADLESS) else None

SUN_DIR = np.array([-0.45, 0.30, -0.84])      # a low afternoon sun in the south-west, over the basin
SUN_DIR /= np.linalg.norm(SUN_DIR)


def make_sky_hdr(path, w=2048, h=1024):
    """A clear west-coast afternoon: pale horizon, a deep zenith, a low sun."""
    j = np.arange(h).reshape(h, 1)
    i = np.arange(w).reshape(1, w)
    theta = (j / h) * math.pi
    phi = (i / w) * 2 * math.pi - math.pi
    y = np.broadcast_to(np.cos(theta), (h, w))
    x, z = np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)
    horizon, zenith, haze = np.array([0.62, 0.66, 0.72]), np.array([0.10, 0.22, 0.46]), np.array([0.30, 0.33, 0.37])
    up = np.clip(y, 0.0, 1.0)[..., None] ** 0.4
    down = np.clip(-y, 0.0, 1.0)[..., None] ** 0.6
    sky = np.where(y[..., None] >= 0.0, horizon * (1.0 - up) + zenith * up, horizon * (1.0 - down) + haze * down)
    sky = sky + np.exp(-(y * y) / (2 * 0.006))[..., None] * np.array([1.0, 0.82, 0.62]) * 0.3
    ang = np.arccos(np.clip(np.stack([x, y, z], axis=-1) @ SUN_DIR, -1.0, 1.0))
    sky = sky + (np.exp(-(ang / math.radians(1.4)) ** 2) * 55.0
                 + np.exp(-(ang / math.radians(12.0)) ** 2) * 3.0)[..., None] * np.array([1.0, 0.9, 0.75])
    return write_radiance_hdr(path, sky)


env = tp.RGBELoader().load(make_sky_hdr(os.path.join(tempfile.gettempdir(), "threepp_usv_rs_sky.hdr")))
scene = tp.Scene()
scene.background = env
scene.environment = env

sun = tp.DirectionalLight(0xffe8cc, 2.4)
sun.position.set(*(SUN_DIR * 600.0))
sun.cast_shadow = True
scene.add(sun)

_t0 = time.perf_counter()
GEO = tp.GeoScene(PACK, forest_focus=(400.0, 0.0, 400.0), scatter=False, cars=False, forest_extent=1500.0)
scene.add(GEO)
print(f"terrain: {os.path.basename(PACK)} {GEO.pack_world_size:.0f} m loaded in {time.perf_counter() - _t0:.1f} s "
      f"({GEO.stats['buildings']} buildings, {GEO.stats['boats']} boats)")
print(GEO.attribution)


def label_subtree(root, ident):
    """One instance id for everything under `root`: the sensors' labels only, the look is untouched."""
    if root is not None:
        root.traverse(lambda o: renderer.set_instance_id(o, ident))


# The terrain tiles stream in and out, so they keep the ids the renderer gives them.
for _name, _ident in (("geo_urban_props", ID_PROPS), ("geo_buildings", ID_BUILDINGS), ("geo_road_bridges", ID_BRIDGES)):
    label_subtree(GEO.get_object_by_name(_name), _ident)

ocean = tp.Ocean(size=SEA_SIZE, resolution=512, wind_speed=3.0 + 2.2 * SEA_STATE, wind_theta=0.5,
                 choppiness=0.6, fetch=8000.0)
# A sheet this size would ask cascade 0 for a 4 km swell: pin the tiles to a harbour sea.
ocean.params.tile_size_0 = 320.0
ocean.params.tile_size_1 = 40.6
ocean.params.tile_size_2 = 2.98
ocean.params.foam_amount = 0.0
ocean.warp.half_range = 0.5 * SEA_SIZE
ocean.warp.coef_a = 0.07
ocean.material.attenuation_color = tp.Color(0.14, 0.27, 0.25)
ocean.material.attenuation_distance = 3.0
ocean.wake_field.resolution, ocean.wake_field.patches = 1024, 1
ocean.wake_field.ripple_resolution = 512
scene.add(ocean)
renderer.set_instance_id(ocean, ID_SEA)


# --------------------------------------------------------------------------- #
#  The loop: the depth contour along the town's waterfront, back along the south shore
# --------------------------------------------------------------------------- #
def contour(xs, z_from, z_to, step):
    """For each x, the first z (walking from the shore) where the seabed passes the contour depth."""
    pts = []
    for x in xs:
        z = z_from
        while (z < z_to) if step > 0 else (z > z_to):
            if GEO.height_at(float(x), float(z)) < -CONTOUR_DEPTH:
                pts.append((float(x), float(z)))
                break
            z += step
    return pts


north = contour(np.arange(-200.0, 1001.0, 20.0), 150.0, 700.0, 2.0)
south = contour(np.arange(1000.0, -201.0, -20.0), 1250.0, 800.0, -2.0)
LOOP = np.array(north + south, float)
k = np.ones(5) / 5.0                      # smooth the contour a little: it follows a 2 m DTM
LOOP = np.stack([np.convolve(np.pad(LOOP[:, i], 2, mode="wrap"), k, mode="valid") for i in range(2)], 1)
seg = np.linalg.norm(np.diff(np.vstack([LOOP, LOOP[:1]]), axis=0), axis=1)
LOOP_S = np.concatenate([[0.0], np.cumsum(seg)])
LOOP_LEN = float(LOOP_S[-1])
print(f"loop: {len(LOOP)} points along the {CONTOUR_DEPTH:.0f} m contour, {LOOP_LEN:.0f} m round")


def loop_point(s):
    s = s % LOOP_LEN
    i = int(np.searchsorted(LOOP_S, s, side="right") - 1)
    a, b = LOOP[i % len(LOOP)], LOOP[(i + 1) % len(LOOP)]
    f = (s - LOOP_S[i]) / max(1e-6, LOOP_S[i + 1] - LOOP_S[i])
    return a + f * (b - a)


def nearest_s(p):
    d = np.linalg.norm(LOOP - p[None, :], axis=1)
    return float(LOOP_S[int(np.argmin(d))])


# --------------------------------------------------------------------------- #
#  Our boat and her sensors
# --------------------------------------------------------------------------- #
def tag(root, ident):
    def visit(o):
        if isinstance(o, tp.Mesh):
            renderer.set_instance_id(o, ident)
            o.cast_shadow = True
            o.receive_shadow = True
    root.traverse(visit)


boat = tp.GLTFLoader().load(os.path.join(_HERE, "mariner.glb")).scene
tag(boat, ID_OWN)
scene.add(boat)
hull = StripHull("mariner", ocean)
hull.drive = drive = drive_for(hull).bind(boat)
p0 = loop_point(START_S)
p1 = loop_point(START_S + 10.0)
heading0 = math.atan2(-(p1[1] - p0[1]), p1[0] - p0[0])
hull.R = rot_y(heading0) @ rot_z(math.radians(hull.design["trim_deg"]))
hull.p[0], hull.p[2] = float(p0[0]), float(p0[1])     # seated on the water after the first render
wake = Wake(hull, ocean)
wash = Wash(hull, drive, ocean, 0)

# The radar on the mast's radar node (local -Z = the bow, +Y up).
radar_model = tp.MarineRadarModel.open_array_4ft() if OPEN_ARRAY else tp.MarineRadarModel.radome_24in()
if not OPEN_ARRAY:
    rs = hull.spec["sensors"]["radar"]          # the Mariner's own: a 20 in radome class
    radar_model.horizontal_beamwidth = float(rs["h_beam_deg"])
    radar_model.vertical_beamwidth = float(rs["v_beam_deg"])
    radar_model.rpm = float(rs["rpm"])
radar_model.min_range = 25.0
radar_model.max_range = 1852.0
radar_model.range_bins = 1024
radar_model.pulse_length = 80e-9
radar_model.sea_state = SEA_STATE
if RAIN:
    radar_model.rain = [tp.RainCell(820.0, 760.0, radius=150.0, top=1400.0, rain_rate=16.0)]   # over the basin
radar = tp.MarineRadar(radar_model)
radar.materials.set(ID_SEA, tp.RadarMaterial.sea_surface())
radar.materials.set(ID_OTTERX, tp.RadarMaterial.fibreglass())
radar.materials.set(ID_OTTER, tp.RadarMaterial.fibreglass())
for _k in range(N_BUOYS):
    radar.materials.set(ID_BUOY + _k, tp.RadarMaterial.radar_reflector())
radar.materials.default_material = tp.RadarMaterial.rock()
boat.get_object_by_name("radar").add(radar)

# The forward-looking sonar on the gondola's flat, her deepest point (the spec's transducer mount),
# a hand below it: local -Z along her +X, tilted 24 deg down. Shallower, the wave troughs come
# within a foot of the face and take the fan's shallow edge.
sonar = tp.ImagingSonar(tp.ImagingSonarModel.oculus_m750d())
sm = sonar.model
sm.max_range = 50.0
sm.range_bins = 768
sm.water.temperature = 8.0
sm.water.salinity = 33.0
sm.surface_y = 0.0
sm.surface_rms_height = 0.02 + 0.03 * SEA_STATE
sonar.rebuild()
sonar.materials.default_material = tp.AcousticMaterial.sand()
sonar.materials.set(ID_SEA, tp.AcousticMaterial.sea_surface())
sonar.materials.set(ID_WRECK, tp.AcousticMaterial.steel())
sonar.materials.set(ID_ROCK, tp.AcousticMaterial.rock())
sonar.materials.set(ID_OTTERX, tp.AcousticMaterial.steel())
sonar.materials.set(ID_OTTER, tp.AcousticMaterial.steel())
for _k in range(N_BUOYS):
    sonar.materials.set(ID_BUOY + _k, tp.AcousticMaterial.steel())
SONAR_TILT = math.radians(24.0)
_mount = hull.spec["sensors"]["transducer"]["position"]
sonar.position.set(float(_mount[0]), float(_mount[1]) - 0.04, float(_mount[2]))
eu = tp.Euler(-SONAR_TILT, -0.5 * math.pi, 0.0)
eu.order = tp.RotationOrder.YXZ
sonar.quaternion.set_from_euler(eu)
boat.add(sonar)
PING_HZ = 10.0

# --------------------------------------------------------------------------- #
#  What there is to find: traffic, buoys, a lost Otter X, rocks
# --------------------------------------------------------------------------- #
otterx = tp.GLTFLoader().load(os.path.join(_HERE, "otterx.glb")).scene
tag(otterx, ID_OTTERX)
scene.add(otterx)
otter = tp.GLTFLoader().load(os.path.join(_HERE, "otter.glb")).scene
tag(otter, ID_OTTER)
scene.add(otter)
BASIN = np.array([430.0, 690.0])


def traffic_pose(t):
    """The Otter X circles the basin; the Otter shuttles along the south shore."""
    a = 0.012 * t + 1.0
    px = BASIN + 210.0 * np.array([math.cos(a), math.sin(a)])
    hx = math.atan2(-math.cos(a), -math.sin(a))          # tangent, counter-clockwise seen from above
    s = 0.5 * (1.0 - math.cos(0.004 * t * 2 * math.pi))
    so = loop_point(LOOP_LEN * 0.62 + 300.0 * s) + np.array([0.0, -45.0])
    ds = math.sin(0.004 * t * 2 * math.pi)
    ho = math.pi if ds >= 0.0 else 0.0
    return (px, hx, 0.365), (so, ho, 0.25)


def float_on_sea(obj, xz, heading, draft):
    x, z = float(xz[0]), float(xz[1])
    f, s = 1.2, 0.8
    c, sn = math.cos(heading), math.sin(heading)
    fx, fz = c, -sn                       # her +X along (cos a, 0, -sin a)
    hb = ocean.sample_height(x + f * fx, z + f * fz)
    hs = ocean.sample_height(x - f * fx, z - f * fz)
    hp = ocean.sample_height(x - s * fz, z + s * fx)
    hst = ocean.sample_height(x + s * fz, z - s * fx)
    obj.position.set(x, 0.25 * (hb + hs + hp + hst) - draft, z)
    e = tp.Euler(math.atan2(hp - hst, 2 * s) * 0.6, heading, math.atan2(hb - hs, 2 * f) * 0.6)
    e.order = tp.RotationOrder.YXZ
    obj.quaternion.set_from_euler(e)


def make_buoy(color, ident):
    """A lateral mark: float, mast, topmark, and the octahedral radar reflector on top."""
    g = tp.Group()
    paint = tp.MeshStandardMaterial()
    paint.color = color
    paint.roughness = 0.55
    steel = tp.MeshStandardMaterial()
    steel.color = 0x8a8f94
    steel.metalness = 0.8
    steel.roughness = 0.35
    body = tp.Mesh(tp.CylinderGeometry(0.75, 0.9, 1.3, 24), paint)
    body.position.y = 0.2
    g.add(body)
    mast = tp.Mesh(tp.CylinderGeometry(0.09, 0.12, 2.2, 10), paint)
    mast.position.y = 1.9
    g.add(mast)
    top = tp.Mesh(tp.ConeGeometry(0.35, 0.6, 16) if color == 0x1f7a3a else tp.CylinderGeometry(0.3, 0.3, 0.5, 16), paint)
    top.position.y = 3.25
    g.add(top)
    refl = tp.Mesh(tp.OctahedronGeometry(0.32), steel)
    refl.position.y = 2.65
    g.add(refl)
    tag(g, ident)
    return g


def in_water(p, toward, depth=3.0):
    """`p`, or the first point on the way to `toward` with at least `depth` of water."""
    p, toward = np.asarray(p, float), np.asarray(toward, float)
    for f in np.linspace(0.0, 1.0, 41):
        q = p + f * (toward - p)
        if GEO.height_at(float(q[0]), float(q[1])) < -depth:
            return q
    return toward


BUOYS = []
for s_along, side, color in ((360.0, -30.0, 0x1f7a3a), (900.0, -35.0, 0xb3261e), (1500.0, 40.0, 0x1f7a3a)):
    q = loop_point(s_along)
    b = make_buoy(color, ID_BUOY + len(BUOYS))
    scene.add(b)
    BUOYS.append((b, in_water(q + np.array([0.0, -side]), q)))
b = make_buoy(0xb3261e, ID_BUOY + len(BUOYS))
scene.add(b)
BUOYS.append((b, BASIN + np.array([-60.0, 40.0])))

# The lost Otter X: on her side on the seabed a little off the north contour.
wq = loop_point(START_S + 70.0) + np.array([0.0, 6.0])
wreck = tp.GLTFLoader().load(os.path.join(_HERE, "otterx.glb")).scene
tag(wreck, ID_WRECK)
wy = GEO.height_at(float(wq[0]), float(wq[1]))
wreck.position.set(float(wq[0]), wy + 0.3, float(wq[1]))
e = tp.Euler(math.radians(100.0), math.radians(35.0), math.radians(8.0))
e.order = tp.RotationOrder.YXZ
wreck.quaternion.set_from_euler(e)
scene.add(wreck)
print(f"wreck: Otter X on the seabed at ({wq[0]:.0f}, {wy:.1f}, {wq[1]:.0f})")

rock_mat = tp.MeshStandardMaterial()
rock_mat.color = 0x4a4740
rock_mat.roughness = 0.95
rng = np.random.default_rng(7)
ROCKS = []                    # (id, centre, radius): the sonar's ground truth
for i in range(14):
    q = loop_point(START_S + 30.0 + 14.0 * i) + rng.normal(0.0, 9.0, 2)
    r = float(rng.uniform(0.5, 1.6))
    rock = tp.Mesh(tp.IcosahedronGeometry(r, 1), rock_mat)
    rock.scale.set(1.0, float(rng.uniform(0.5, 0.8)), float(rng.uniform(0.8, 1.2)))
    rock.position.set(float(q[0]), GEO.height_at(float(q[0]), float(q[1])) + 0.3 * r, float(q[1]))
    rock.rotation.y = float(rng.uniform(0, 6.28))
    renderer.set_instance_id(rock, ID_ROCK + i)
    ROCKS.append((ID_ROCK + i, (float(q[0]), float(rock.position.y), float(q[1])), r))
    sonar.materials.set(ID_ROCK + i, tp.AcousticMaterial.rock())
    scene.add(rock)

# --------------------------------------------------------------------------- #
#  The two displays, as screen-space panels
# --------------------------------------------------------------------------- #
PPI = 470 if H >= 800 else 380
FAN_W, FAN_H = (600, 360) if H >= 800 else (480, 290)
MARGIN = 14
FONT = tp.FontLoader().default_font()


def panel(w, h, anchor_x, anchor_y, y_off, text):
    tex = tp.data_texture(np.zeros((h, w, 4), np.uint8), srgb=True)
    tex.generate_mipmaps = False
    tex.wrap_s = tex.wrap_t = tp.TextureWrapping.ClampToEdge
    mat = tp.SpriteMaterial()
    mat.map = tex
    sp = tp.Sprite(mat)
    sp.screen_space = True
    sp.screen_anchor.set(anchor_x, anchor_y)
    sp.center.set(anchor_x, anchor_y)
    sp.position.set(-MARGIN, y_off, 0.0)
    sp.scale.set(w, h, 1.0)
    scene.add(sp)
    lab = tp.TextSprite(FONT, world_scale=17.0)
    lab.set_text(text)
    lab.set_color(0xdfe8ee)
    lab.screen_space = True
    lab.screen_anchor.set(anchor_x, anchor_y)
    lab.center.set(1.0, 0.0 if anchor_y < 0.5 else 1.0)
    lab.position.set(-MARGIN, y_off + (h + 6 if anchor_y < 0.5 else -h - 6), 0.0)
    scene.add(lab)
    return tex, lab


ppi_tex, ppi_lab = panel(PPI, PPI, 1.0, 1.0, -MARGIN, "")
fan_tex, fan_lab = panel(FAN_W, FAN_H, 1.0, 0.0, MARGIN, "")
ppi = tp.RadarPpiDisplay(PPI)
ppi.style.trail_seconds = 6.0
fan = tp.SonarFanDisplay(FAN_W, FAN_H)
RANGES_NM = (0.25, 0.5, 0.75, 1.0)
state = {"range": 2, "north_up": False, "labels": False, "autopilot": True, "multipath": True,
         "last_ping": -1.0, "frame": None, "ping_ms": 0.0, "radar_ms": 0.0, "thr_i": 0.0, "fold_t0": 0.0, "ppi_ms": 0.0}


def captions():
    rng_nm = RANGES_NM[state["range"]]
    kind = "4 ft open array" if OPEN_ARRAY else f"{radar_model.horizontal_beamwidth:.1f} deg radome"
    ppi_lab.set_text(f"RADAR  X band, {kind}  |  {rng_nm:g} nmi  |  {'north up' if state['north_up'] else 'head up'}"
                     f"{'  |  labels' if state['labels'] else ''}")
    fan_lab.set_text(f"SONAR  750 kHz, 130 x 20 deg, 512 beams  |  {sm.max_range:.0f} m  |  "
                     f"tilt {math.degrees(SONAR_TILT):.0f} deg{'  |  labels' if state['labels'] else ''}")


captions()

# --------------------------------------------------------------------------- #
#  Camera
# --------------------------------------------------------------------------- #
camera = tp.PerspectiveCamera(48.0, W / H, 0.2, 9000.0)
controls = tp.OrbitControls(camera, canvas)
controls.enable_damping = True
controls.min_distance = 5.0
controls.max_distance = 400.0
canvas.on_window_resize(resize_handler(camera, renderer))
_anchor = np.zeros(3)
_cam = {"yaw": None}


def chase_camera(dt, first=False):
    """Behind her starboard quarter, high enough to see what lies ahead of her."""
    fwd = hull.R[:, 0]
    yaw = math.atan2(fwd[0], fwd[2])
    if _cam["yaw"] is None or first:
        _cam["yaw"] = yaw
    dy = (yaw - _cam["yaw"] + math.pi) % (2 * math.pi) - math.pi
    _cam["yaw"] += dy * (1.0 - math.exp(-dt / 1.5))
    a = _cam["yaw"] + math.radians(205.0)
    c = hull.p
    camera.position.set(float(c[0] + 19.0 * math.sin(a)), 7.5, float(c[2] + 19.0 * math.cos(a)))
    ahead = c + 14.0 * np.array([fwd[0], 0.0, fwd[2]])
    camera.look_at(tp.Vector3(float(ahead[0]), 0.5, float(ahead[2])))


def follow_camera():
    """Interactive: keep the orbit target on her without stealing the user's orbit."""
    global _anchor
    c = hull.p
    d = c - _anchor
    _anchor = c.copy()
    camera.position.set(camera.position.x + d[0], camera.position.y + 0.3 * d[1], camera.position.z + d[2])
    controls.target = tp.Vector3(float(c[0]), float(c[1]) + 1.0, float(c[2]))
    controls.update()


# --------------------------------------------------------------------------- #
#  One frame
# --------------------------------------------------------------------------- #
sim = {"t": 0.0, "afloat": False, "rendered": False}


def autopilot(dt):
    """Steer for a point 35 m ahead on the loop; hold her near SPEED (PI on the throttle)."""
    p = np.array([hull.p[0], hull.p[2]])
    tgt = loop_point(nearest_s(p) + 35.0)
    hd, _, _ = attitude(hull.R)
    want = math.atan2(-(tgt[1] - p[1]), tgt[0] - p[0])
    err = wrap(want - hd)
    drive.steer_cmd = max(-drive.steer_max, min(drive.steer_max, -1.4 * err))
    e = SPEED - hull.u
    state["thr_i"] = max(-0.2, min(0.4, state["thr_i"] + 0.04 * dt * e))
    drive.throttle_cmd = max(0.0, min(0.7, 0.18 + state["thr_i"] + 0.06 * e))


def step(dt):
    sim["t"] += dt
    t = sim["t"]
    renderer.sim_time = t
    if state["autopilot"]:
        autopilot(dt)
    drive.tick(dt)
    if sim["rendered"]:
        if not sim["afloat"]:
            hull.seat(float(p0[0]), float(p0[1]), heading0)
            sim["afloat"] = True
        hull.advance(dt, max(1, int(math.ceil(dt * SUBSTEPS_HZ))))
        wake.update(dt, True)
        ocean.clear_wake_sources()
        wash.update(dt, True)
    hull.place(boat)
    drive.pose()
    ocean.warp_toward(float(hull.p[0]), float(hull.p[2]), 0.07)
    (px, hx, dx), (po, ho, do) = traffic_pose(t)
    float_on_sea(otterx, px, hx, dx)
    float_on_sea(otter, po, ho, do)
    for bb, q in BUOYS:
        h = ocean.sample_height(float(q[0]), float(q[1]))
        bb.position.set(float(q[0]), h - 0.35, float(q[1]))
    GEO.update(camera.position)


# Where the sensors were when they fired: the ground-truth report's poses (--shot only).
POSES = {"radar": [], "sonar": {}}


def sense():
    """After render(): sweep the radar, ping the sonar, redraw both displays."""
    t = sim["t"]
    t0 = time.perf_counter()
    radar.sim_time = t
    radar.update(renderer)
    if HEADLESS:
        a = radar.get_world_position()
        POSES["radar"].append((t, a.x, a.y, a.z))
    t_mid = time.perf_counter()
    rng = RANGES_NM[state["range"]] * 1852.0
    ppi.style.show_labels = state["labels"]
    ppi.draw(radar.image, t, radar.heading_deg, radar.antenna_bearing_deg, state["north_up"], rng)
    ppi.upload(ppi_tex)
    state["radar_ms"] = 1000.0 * (t_mid - t0)
    state["ppi_ms"] = 1000.0 * (time.perf_counter() - t_mid)

    # The sonar folds its acoustics on a worker thread: the frame never waits for it.
    if sonar.ping_pending and sonar.ping_ready(renderer):
        sonar.ping_collect_async(renderer)
        state["fold_t0"] = time.perf_counter()
    if sonar.frame_ready:
        state["ping_ms"] = 1000.0 * (time.perf_counter() - state["fold_t0"])
        f = sonar.take_frame()
        state["frame"] = f
        fan.show_labels = state["labels"]
        fan.draw(f, sm.horizontal_fov)
        fan.upload(fan_tex)
    if not sonar.ping_pending and not sonar.folding and t - state["last_ping"] >= 1.0 / PING_HZ:
        sm.surface_multipath = state["multipath"]
        sonar.sim_time = t
        if HEADLESS:
            POSES["sonar"][t] = sonar.matrix_world.to_numpy()
        sonar.ping_begin(renderer)
        state["last_ping"] = t


# --------------------------------------------------------------------------- #
#  Input and the panel
# --------------------------------------------------------------------------- #
_prev = {}


def pressed(key):
    now = canvas.is_key_down(key)
    fired = now and not _prev.get(key, False)
    _prev[key] = now
    return fired


def handle_keys(dt):
    if ui is not None and ui.want_capture_keyboard:
        return
    if canvas.is_key_down("W") or canvas.is_key_down("S") or canvas.is_key_down("A") or canvas.is_key_down("D"):
        state["autopilot"] = False
    if canvas.is_key_down("W"):
        drive.throttle_cmd = min(1.0, drive.throttle_cmd + 0.5 * dt)
    if canvas.is_key_down("S"):
        drive.throttle_cmd = max(drive.throttle_min, drive.throttle_cmd - 0.5 * dt)
    if not state["autopilot"]:
        steer = (1.0 if canvas.is_key_down("D") else 0.0) - (1.0 if canvas.is_key_down("A") else 0.0)
        drive.steer_cmd = steer * drive.steer_max
    for key, what in (("P", "autopilot"), ("N", "north_up"), ("L", "labels"), ("M", "multipath")):
        if pressed(key):
            state[what] = not state[what]
            captions()
    for i in range(4):
        if pressed(str(i + 1)):
            state["range"] = i
            captions()


def draw_ui():
    tp.imgui.set_next_window_pos(10, 10)
    tp.imgui.set_next_window_size(340, 0)
    tp.imgui.begin("Mariner: radar + sonar")
    hd, _, _ = attitude(hull.R)
    tp.imgui.text(f"{hull.u / 0.5144:4.1f} kn  heading {math.degrees(hd) % 360:5.1f}  "
                  f"{'AUTOPILOT' if state['autopilot'] else 'manual'}")
    tp.imgui.separator()
    tp.imgui.text("Radar")
    ch, v = tp.imgui.slider_float("gain dB", radar.model.gain_db, -10.0, 20.0)
    if ch:
        radar.model.gain_db = v
    ch, v = tp.imgui.slider_float("sea (STC) dB", radar.model.sea_clutter_db, 0.0, 40.0)
    if ch:
        radar.model.sea_clutter_db = v
    ch, v = tp.imgui.slider_float("rain (FTC)", radar.model.rain_clutter, 0.0, 1.0)
    if ch:
        radar.model.rain_clutter = v
    ch, v = tp.imgui.slider_float("sea state", radar.model.sea_state, 0.0, 6.0)
    if ch:
        radar.model.sea_state = v
    tp.imgui.text(f"{radar.rays_per_slot} rays/spoke, radar {state['radar_ms']:.1f} ms, scope {state['ppi_ms']:.1f} ms")
    tp.imgui.separator()
    tp.imgui.text("Sonar")
    ch, v = tp.imgui.slider_float("sonar gain dB", sm.gain_db, -20.0, 20.0)
    if ch:
        sm.gain_db = v
    ch, v = tp.imgui.slider_float("TVG dB/decade", sm.tvg_spreading, 10.0, 40.0)
    if ch:
        sm.tvg_spreading = v
    ch, v = tp.imgui.checkbox("surface multipath (M)", state["multipath"])
    if ch:
        state["multipath"] = v
    tp.imgui.text(f"{sonar.ray_count} rays/ping, fold {state['ping_ms']:.1f} ms (worker thread)")
    f = state["frame"]
    if f is not None:
        tp.imgui.text(f"noise floor {f.noise_level_db:.0f} dB re 1 uPa")
    tp.imgui.separator()
    tp.imgui.text("P autopilot  N north-up  L labels  1-4 range")
    tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps")
    tp.imgui.end()


# --------------------------------------------------------------------------- #
#  Run
# --------------------------------------------------------------------------- #
def write_png(path, rgba):
    try:
        from PIL import Image
        Image.fromarray(rgba).save(path)
    except ImportError:
        np.save(path + ".npy", rgba)


def ground_truth(stem):
    """Score the last radar revolution and the last sonar ping against the height
    field and the targets' poses (rs_truth.py); write _truth.json and B-scans."""
    import json

    import rs_truth
    t0 = time.perf_counter()
    report = {}

    def half_size(obj):
        sz = tp.Box3().set_from_object(obj, False).get_size()
        return 0.5 * math.hypot(sz.x, sz.z)

    def fm(v, spec="+.1f"):
        return "n/a" if v is None else format(v, spec)

    def top_of(obj):
        return tp.Box3().set_from_object(obj, False).max().y - ocean.sample_height(obj.position.x, obj.position.z)

    radar_targets = {ID_OTTERX: ("Otter X", lambda tt: tuple(traffic_pose(tt)[0][0]), half_size(otterx), top_of(otterx)),
                     ID_OTTER: ("Otter", lambda tt: tuple(traffic_pose(tt)[1][0]), half_size(otter), top_of(otter))}
    for k, (bb, q) in enumerate(BUOYS):
        radar_targets[ID_BUOY + k] = (f"buoy {k}", lambda tt, q=q: (float(q[0]), float(q[1])), 0.9, top_of(bb))
    not_land = {ID_SEA, ID_OWN, ID_PROPS, ID_BRIDGES, ID_WRECK, *radar_targets, *(r[0] for r in ROCKS)}
    a = radar.get_world_position()
    grid = rs_truth.HeightGrid(GEO, a.x, a.z, radar_model.max_range + 60.0, 2.5)
    rt = rs_truth.radar_truth(radar.image, radar_model, POSES["radar"], grid, radar_targets, not_land,
                              radar_model.threshold_db)
    report["radar"] = rt
    c = rt["coast"]
    print(f"radar truth: {rt['spokes_filled']} spokes. Land echoes on land (within one resolution cell): "
          f"{100 * c['on_land_share']:.1f}% of {c['land_echo_cells']} cells.\n"
          f"    first land echo per spoke vs the height field's shore ({c['spokes_with_shore']} spokes): "
          f"at the shore {c['first_echo_at_shore']}, early (over water) {c['first_echo_early']}, behind it "
          f"{c['first_echo_behind_shore']} (flat aprons and beaches return little at grazing incidence); "
          f"error median {fm(c['median_err_m'])} m, p10 {fm(c['p10_err_m'])}, p90 {fm(c['p90_err_m'])}")
    for ex in c.get("early_examples", []):
        print(f"    early: bearing {ex['world_bearing_deg']:.1f}, echo {ex['echo_m']:.0f} m, shore {ex['shore_m']:.0f} m, "
              f"label {ex['label']}")
    for ident, r in rt["targets"].items():
        if r["seen"]:
            print(f"    {r['name']:8s} range {r['range_true_m']:6.0f} m: error {r['range_err_m']:+5.1f} m "
                  f"(tol {r['range_tol_m']:.1f}), bearing {r['bearing_err_deg']:+5.2f} deg "
                  f"(tol {r['bearing_tol_deg']:.2f}), SNR {r['snr_db']:.0f} dB  {'ok' if r['ok'] else 'OUT'}")
        else:
            print(f"    {r['name']:8s} not seen{'' if r['line_of_sight'] else ' (no line of sight over the terrain: not expected)'}")
        if "at_truth" in r:
            a_t = r["at_truth"]
            print(f"             at its true cell ({a_t['range_true_m']:.0f} m): max SNR {a_t['max_snr_db']:.0f} dB, "
                  f"label there {a_t['label_at_max']}, labels above threshold {a_t['labels_above_threshold']}")
    write_png(stem + "_radar_bscan.png", rs_truth.radar_bscan(radar.image, rt, radar_targets, radar_model.threshold_db))
    for r in rt["targets"].values():
        if r["seen"] and not r["line_of_sight"]:
            print(f"    note: {r['name']} seen without a line of sight on the height field")

    f = state["frame"]
    if f is not None and POSES["sonar"]:
        world = POSES["sonar"][min(POSES["sonar"], key=lambda tt: abs(tt - f.time))]
        sgrid = rs_truth.HeightGrid(GEO, float(world[0, 3]), float(world[2, 3]), sm.max_range + 5.0, 0.5)
        objs = {ID_WRECK: ("wreck", (float(wreck.position.x), float(wreck.position.y), float(wreck.position.z)),
                           half_size(wreck))}
        for rid, pos, rad in ROCKS:
            objs[rid] = (f"rock {rid - ID_ROCK}", pos, rad)
        st = rs_truth.sonar_truth(f, sm, world, sgrid, objs, not_seabed=(ID_SEA, ID_OWN))
        report["sonar"] = st
        b = st["seabed"]
        print(f"sonar truth: seabed edge between the fan's first null and its -3 dB edge on {b['edge_between_null_and_3db']}"
              f"/{b['beams_with_bottom']} beams ({100 * b['share']:.1f}%), error vs the -3 dB edge median "
              f"{fm(b['median_err_vs_3db_m'], '+.2f')} m (p10 {fm(b['p10_err_m'], '+.2f')}, p90 {fm(b['p90_err_m'], '+.2f')})")
        in_fan = []
        for ident, r in st["objects"].items():
            if abs(r["bearing_true_deg"]) > 0.5 * sm.horizontal_fov or not (sm.min_range < r["range_true_m"] < sm.max_range):
                continue
            in_fan.append((r["name"], r["bearing_true_deg"], r["range_true_m"]))
            if r["seen"]:
                print(f"    {r['name']:8s} range {r['range_true_m']:5.1f} m: error {r['range_err_m']:+5.2f} m "
                      f"(tol {r['range_tol_m']:.2f}), bearing {r['bearing_err_deg']:+5.2f} deg "
                      f"(tol {r['bearing_tol_deg']:.2f})  {'ok' if r['ok'] else 'OUT'}")
            else:
                if not r["in_fan"]:
                    why = f" (at {r['elevation_deg']:+.0f} deg: outside the vertical fan, not expected)"
                elif not r["in_main_lobe"]:
                    why = (f" (at {r['elevation_deg']:+.0f} deg, in the fan's skirt: two-way pattern "
                           f"{r['pattern_two_way_db']:.0f} dB)")
                else:
                    why = ""
                print(f"    {r['name']:8s} range {r['range_true_m']:5.1f} m, bearing {r['bearing_true_deg']:+.1f}: not seen{why}")
        write_png(stem + "_sonar_bscan.png", rs_truth.sonar_bscan(f, st, in_fan))
    img = radar.image
    np.savez_compressed(stem + "_data.npz", radar_snr=img.snr_db, radar_label=img.label, radar_time=img.time,
                        radar_bearing=img.world_bearing_deg, radar_poses=np.asarray(POSES["radar"]),
                        sonar_level=f.level_db if f is not None else np.zeros(0),
                        sonar_label=f.label if f is not None else np.zeros(0),
                        sonar_kind=f.label_kind if f is not None else np.zeros(0))
    with open(stem + "_truth.json", "w", encoding="utf-8") as fh:
        json.dump(rs_truth.strip_arrays(report), fh, indent=1)
    print(f"[rs] ground truth in {time.perf_counter() - t0:.1f} s: {stem}_truth.json, _radar_bscan, _sonar_bscan")


if HEADLESS:
    fps = 30
    step(1.0 / fps)
    chase_camera(1.0 / fps, first=True)
    n = int(round(SHOT_T * fps))
    for i in range(n):
        renderer.render(scene, camera)
        sim["rendered"] = True
        sense()
        step(1.0 / fps)
        chase_camera(1.0 / fps)
    # A ping and a fan from the final pose, so the panel matches the frame.
    while sonar.ping_pending or sonar.folding:
        renderer.render(scene, camera)
        sense()
    renderer.render(scene, camera)
    sense()
    renderer.save_frame(scene, camera, OUT)
    stem = os.path.splitext(OUT)[0]
    write_png(stem + "_ppi.png", ppi.rgba())
    write_png(stem + "_fan.png", fan.rgba())
    st = state["labels"]
    state["labels"] = True
    ppi.style.show_labels = True
    ppi.draw(radar.image, sim["t"], radar.heading_deg, radar.antenna_bearing_deg, False, RANGES_NM[state["range"]] * 1852.0)
    write_png(stem + "_ppi_labels.png", ppi.rgba())
    if state["frame"] is not None:
        fan.show_labels = True
        fan.draw(state["frame"], sm.horizontal_fov)
        write_png(stem + "_fan_labels.png", fan.rgba())
        f = state["frame"]
        lab = f.label
        ids, counts = np.unique(lab[lab >= 0], return_counts=True)
        print("sonar labels:", dict(zip(ids.tolist(), counts.tolist())),
              f"level {float(f.level_db.min()):.0f}..{float(f.level_db.max()):.0f} dB, noise {f.noise_level_db:.0f} dB")
    img = radar.image
    rl = img.label
    ids, counts = np.unique(rl[rl >= 0], return_counts=True)
    print("radar labels:", dict(zip(ids.tolist()[:12], counts.tolist()[:12])),
          f"snr max {float(img.snr_db.max()):.0f} dB, displayed cells {int((img.display > 0.05).sum())}")
    # Ground truth for the PPI's geometry: the terrain's land mask in the same
    # head-up frame, so a mirrored or rotated scope shows at a glance.
    rng = RANGES_NM[state["range"]] * 1852.0
    S = ppi.size
    yy, xx = np.mgrid[0:S, 0:S]
    c = 0.5 * (S - 1)
    dxp, dyp = xx - c, c - yy
    rp = np.hypot(dxp, dyp) / (0.5 * S - 3.0) * rng
    wb = np.radians(np.degrees(np.arctan2(dxp, dyp)) + radar.heading_deg)
    wx = hull.p[0] + rp * np.sin(wb)
    wz = hull.p[2] - rp * np.cos(wb)
    land = np.zeros((S, S), bool)
    for j in range(0, S, 2):
        for i in range(0, S, 2):
            if rp[j, i] <= rng:
                land[j:j + 2, i:i + 2] = GEO.height_at(float(wx[j, i]), float(wz[j, i])) > 0.3
    ppi.style.show_labels = False
    ppi.draw(radar.image, sim["t"], radar.heading_deg, radar.antenna_bearing_deg, False, rng)
    over = ppi.rgba().copy()
    over[land, 2] = np.maximum(over[land, 2], 150)
    over[land, 0] = np.maximum(over[land, 0], 90)
    write_png(stem + "_ppi_truth.png", over)
    ppi.draw(radar.image, sim["t"], radar.heading_deg, radar.antenna_bearing_deg, True, rng)
    write_png(stem + "_ppi_north.png", ppi.rgba())
    ground_truth(stem)
    hd, _, _ = attitude(hull.R)
    print(f"boat at ({hull.p[0]:.0f}, {hull.p[2]:.0f}), heading {math.degrees(hd) % 360:.0f} (math), "
          f"radar heading_deg {radar.heading_deg:.1f} (from north, clockwise)")
    print(f"[rs] wrote {OUT} (+ _ppi, _fan, _labels, _ppi_truth); radar {state['radar_ms']:.1f} ms, scope {state['ppi_ms']:.1f} ms")
else:
    clock = tp.Clock()
    step(1e-3)
    chase_camera(0.0, first=True)
    _anchor = hull.p.copy()

    def animate():
        dt = min(clock.get_delta(), 0.05)
        handle_keys(dt)
        if ui is not None:
            controls.enabled = not ui.want_capture_mouse
        step(dt)
        follow_camera()
        renderer.render(scene, camera)
        sim["rendered"] = True
        sense()
        if ui is not None:
            ui.render(draw_ui)

    canvas.animate(animate)
