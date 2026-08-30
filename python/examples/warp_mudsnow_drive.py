"""A Range Rover driven over Bekker-Wong ground: mud, snow and a packed clay track.

WASD drives a PxVehicle2 Range Rover Evoque down three lanes of deformable
terrain (threepp.terrain_deform, the Sumner/O'Brien/Hodgins displacement grid
with a Bekker-Wong bearing and Janosi-Hanamoto shear). The lanes are 48 m long:
mud on one side, packed snow on the other, a firm clay strip down the middle to
spawn on and to compare against. The wheels cut ruts, the ruts stay, and the
soil decides how deep the car sits and how much of the throttle reaches the
ground.

    pip install warp-lang
    python warp_mudsnow_drive.py                  # Vulkan if available
    python warp_mudsnow_drive.py --gl             # OpenGL
    python warp_mudsnow_drive.py --vsync          # cap to the display
    python warp_mudsnow_drive.py --interop        # zero-copy lane meshes (see Strip)
    python warp_mudsnow_drive.py --bench          # ms/frame, vsync off, headless
    python warp_mudsnow_drive.py --shot --script mud     # one acceptance frame
    python warp_mudsnow_drive.py --shot --script cine    # the cinematic frame
    python warp_mudsnow_drive.py --cell 0.08 --width 6   # cheaper ground

    W / S     throttle / brake        R    gear (forward <-> reverse)
    A / D     steer                   SPACE handbrake
    V         driver POV              C    cinematic orbit
    BACKSPACE respawn                 T    reset the ground
    F         save a frame

`--script` drives a fixed input sequence headless and saves the frame the
comparison needs: `mud` and `snow` are the same line at the same throttle down
either soft lane, `spin_mud` / `spin_clay` the same full-throttle launch, `lap`
a circle in mud twice, `cine` the look pass's own frame off the C camera, and
`pan --yaw D` one heading of a 360 from the driver's seat. Repeatable is the
point -- a hand-driven pass is not the same line twice, and the whole claim is
that only the SOIL differs.

How the ground and the vehicle are coupled
------------------------------------------
PhysX finds ground with a raycast, and the deformable terrain does not live in
the PhysX scene. Mirroring the grid into a collider every frame is hopeless, and
holding the car up on terrain reaction forces alone makes a boat -- no sticky
tires, no suspension. So the split is:

  * PhysX keeps the WHOLE vehicle: suspension, tire model, sticky tires,
    substeps, load transfer. Untouched.
  * The road its suspension sees is Bekker's. Per wheel, per frame, we invert
    the Bekker bearing integral for the current wheel load W and hand the
    suspension a plane at `grade - z_eq(W)` (PhysxVehicle.set_road_override).
    The wheel rides IN the ground by exactly the sinkage the load dictates.
  * The friction the tire model gets is Mohr-Coulomb's:
    mu = (c*A(z_eq) + tan_phi*W) / W -- about 0.37 in mud, 0.40 in snow and
    0.75 on clay, against the 2.0 of the C++ demo's asphalt. That difference IS
    the driving experience.
  * The terrain module carves the ruts: the wheels are its collider spheres, fed
    the contact-patch slip velocity, so the prints follow the wheels and the
    module's own Janosi state is live for the panel.

Double-count audit -- what the soil model applies and what it only reports:

    bearing (module)   NOT applied. PhysX's suspension carries the car; the
                       module's Bekker force would be the same load a second
                       time. It is displayed instead, next to the PhysX
                       suspension load, and the two agreeing to a few percent
                       is the point of the middle panel.
    shear (module)     NOT applied. The tire model does traction, under the
                       Mohr-Coulomb mu ceiling we hand it -- which saturates at
                       the same c*A + tan_phi*W the module's Janosi law does.
                       Displayed as traction utilisation.
    motion resistance  APPLIED, via add_force_at_pos: Bekker's compaction
                       resistance (the work of making the rut) plus the
                       first-order bulldozing wedge, opposing the wheel's FULL
                       horizontal travel -- a sideways slide ploughs soil too.
                       PhysX has no notion of either, so there is nothing to
                       double-count: this is why soft ground bleeds speed and a
                       slide dies instead of sailing.
    slip sinkage       APPLIED, through the road override: a spinning wheel
                       excavates, so the effective sinkage is z_eq scaled by a
                       dig factor that follows slip with a ~0.6 s time constant.
                       Floor it in mud and the wheel digs to where the motion
                       resistance exceeds the traction ceiling -- stuck -- and
                       easing off lets the dig relax and the car creep out.
                       Standard terramechanics (slip sinkage), and the whole
                       mud-driving skill loop. NOTE: the Bekker-vs-PhysX panel
                       is an identity only while dig ~ 1 -- a dug-in wheel sits
                       deliberately below static equilibrium, and the panel
                       then shows the module's excess bearing at that depth.
    suction, grade     module-internal; they shape the grid the wheels then read.

Keep that split. Applying the module's bearing or shear on top of PhysX's would
double the load the car already carries.
"""
import atexit
import math
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import warp as wp

import threepp as tp
from threepp.terrain_deform import MATERIALS, DeformableTerrain
from warp_common import cli_arg, parse_size, standard_material

try:
    from threepp.cuda_interop import VkInteropArray
except ImportError:                                 # build without the interop helper
    VkInteropArray = None

GL = "--gl" in sys.argv
INTEROP = "--interop" in sys.argv    # zero-copy publish; see the note on Strip
VSYNC = "--vsync" in sys.argv
SHOT = "--shot" in sys.argv
BENCH = "--bench" in sys.argv
SHOT_TIME = cli_arg("--shot", 8.0, float)
WIDTH, HEIGHT = parse_size(cli_arg("--size", "1600x900", str))
CELL = cli_arg("--cell", 0.05, float)
LANE_LEN = cli_arg("--length", 48.0, float)
LANE_W = cli_arg("--width", 8.0, float)
OUT_DIR = cli_arg("--out-dir", ".", str)
AA = cli_arg("--aa", 4, int)
WARM = cli_arg("--warm", 60, int)
NOSAN = "--no-sanitize" in sys.argv

DT = 1.0 / 60.0
R_WHEEL = 0.40                    # PhysX wheel radius, and the collider sphere
CHASSIS_MASS = 1500.0
REST_LOAD = CHASSIS_MASS * 9.81 / 4.0

# --- the yard -----------------------------------------------------------------
# 48 m of lane along x, three strips across z. The terrain module works in its
# own (x, y, height) frame and publishes y-up meshes as (x, y, z) -> (x, z, -y),
# so a lane's module-y is the NEGATIVE of its world z.

X0, X1 = -0.5 * LANE_LEN, 0.5 * LANE_LEN
NX = int(round((X1 - X0) / CELL)) + 1

CLAY_W = 4.0
LANES = ("mud", "clay", "snow")
LANE_Z = {"mud": (0.5 * CLAY_W, 0.5 * CLAY_W + LANE_W),
          "clay": (-0.5 * CLAY_W, 0.5 * CLAY_W),
          "snow": (-0.5 * CLAY_W - LANE_W, -0.5 * CLAY_W)}

# The presets are calibrated to a 5 cm Spot foot; the bearing numbers carry
# straight over to a 40 cm wheel (a 3.7 kN corner sits 22 mm into clay, 55 mm
# into mud, 64 mm into snow, which is what those materials do to a light 4x4).
# The one thing that does not carry over is the grade relaxation: at 0.15 /s a
# rut forgets it is a rut in about seven seconds, and a car comes back around
# for a second lap long before that. Slowed to 0.02 /s, the mud and snow ruts
# hold their datum for a minute, which is what makes the second pass ride in
# its own tracks instead of digging a fresh sinkage below them.
MATS = {"mud": replace(MATERIALS["mud"], grade_rate=0.02),
        "snow": replace(MATERIALS["snow"], grade_rate=0.02),
        "clay": MATERIALS["clay"]}

SPAWN_POS = tp.Vector3(-20.0, 1.05, 0.0)
SPAWN_ROT = tp.Quaternion().set_from_axis_angle(tp.Vector3(0, 1, 0), math.pi / 2)

# Motion-resistance coefficients. Bulldozing: the fraction of (frontal patch
# area x bearing pressure) pushed as the soil wedge -- around 460 N per wheel in
# mud and 830 N in snow at static sinkage. Compaction: Bekker's rut-making work
# (k_c + 2b*k_phi) * z^(n+1) / (n+1), scaled so that mud cruise costs ~0.18 g
# total -- enough that a slide dies and a coast bleeds off, while leaving
# ~0.18 g of traction margin to actually drive with (the mud ceiling is 0.37 g).
C_BULLDOZE = 0.5
C_COMPACT = 0.35
# The wedge cannot push harder than the material can bear before it FAILS --
# snow crushes at a few tens of kPa. Without this cap snow's n=1.6 makes the
# wedge pressure explode with depth (~120 kPa dug-in) and the lane is
# undrivable at its own static sinkage. Mud's cap is above anything reachable.
P_BULLDOZE_CAP = {"mud": 8.0e4, "snow": 3.5e4, "clay": 1.0e9}

# Slip sinkage: a spinning wheel excavates. The effective sinkage is
# z_eq * dig, where dig follows 1 + K_DIG * min(scrub / SCRUB_REF, 1) with a
# DIG_TAU time constant -- digging in is excavation, not instant, and so is
# climbing back out when you ease off. The driver is the contact patch's scrub
# SPEED, not the slip ratio: any launch from rest has ratio ~1, but a gentle
# launch scrubs ~1 m/s and barely digs, while wheelspin scrubs 10-25 m/s and
# buries the wheel. DIG_MAX caps the hole at a physical fraction of the wheel.
# At full spin in mud the resistance at DIG_MAX exceeds the mu*W traction
# ceiling: genuinely stuck until the dig relaxes. Clay barely digs.
SCRUB_DEAD = 1.5                  # m/s below which a wheel compacts, not digs --
                                  # without this dead zone the car can chicken-and-
                                  # egg itself stuck: drag keeps it slow, slow keeps
                                  # scrub up, scrub keeps the dig (and drag) up.
SCRUB_REF = 5.0                   # m/s of scrub PAST the dead zone for full rate
K_DIG = {"mud": 1.6, "snow": 1.3, "clay": 0.2}
DIG_MAX = {"mud": 2.5, "snow": 2.2, "clay": 1.3}
DIG_TAU = 0.6                     # seconds
DRAG_CAP_FRAC = 1.5               # per-wheel drag cap, x wheel load (stability)

RIGID_Y = -0.03                   # the packed ground the lanes sit on
FALLBACK_MU = 1.1                 # see "handling" below


# --- Bekker inversion ----------------------------------------------------------


def bekker_table(mat, radius=R_WHEEL, cell=CELL, z_max=0.30, n=241):
    """W(z) and contact area A(z) for a sphere pressed z into flat ground.

    The SAME discrete sum the terrain module's imprint kernel evaluates -- one
    cell of the grid at a time, the chord half-width b floored at a cell, the
    pressure (k_c/b + k_phi) * z^n over the cell area -- so inverting this table
    lands the wheel at exactly the depth where the module's own bearing balances
    the load, and the panel's Bekker-vs-PhysX column agrees by construction
    rather than by fitting. Closed forms for the same integral do not: the
    b-floor and the grid quantisation are both worth several percent at these
    depths.
    """
    zs = np.linspace(0.0, z_max, n)
    m = int(math.ceil(radius / cell)) + 2
    g = np.arange(-m, m + 1) * cell
    dx, dy = np.meshgrid(g, g, indexing="ij")
    d2 = dx * dx + dy * dy
    load = np.zeros(n)
    area = np.zeros(n)
    for i, z in enumerate(zs):
        # Height of the sphere's lower surface over each cell, +BIG outside it.
        bottom = np.where(d2 < radius * radius,
                          (radius - z) - np.sqrt(np.maximum(radius * radius - d2, 0.0)),
                          1.0e9)
        zc = np.maximum(-bottom, 0.0)
        b = np.sqrt(np.maximum(zc * (2.0 * radius - zc), cell * cell))
        p = (mat.k_c / b + mat.k_phi) * np.power(zc, mat.n)
        load[i] = float((p * cell * cell).sum())
        area[i] = float((zc > 0.0).sum()) * cell * cell
    return zs, load, area


LUT = {k: bekker_table(m) for k, m in MATS.items()}


def sinkage(lane, w):
    """(z_eq, mu) for wheel load `w` on `lane`."""
    zs, load, area = LUT[lane]
    m = MATS[lane]
    z = float(np.interp(w, load, zs))
    a = float(np.interp(z, zs, area))
    return z, (m.cohesion * a + m.tan_phi * w) / max(w, 1.0)


def motion_resistance(lane, z):
    """Soil drag (N) for a wheel travelling at sinkage z: compaction + bulldozing.

    Compaction is Bekker's rut-making work per unit distance,
    (k_c + w_eff*k_phi) * z^(n+1) / (n+1) -- the energy that went into the print
    you can see behind the car. Bulldozing is the wedge ahead of the wheel, 2b
    wide and z deep, pushed at the bearing pressure the wheel is generating.
    Both grow superlinearly in z, which is what makes slip sinkage a trap: dig
    twice as deep and the drag roughly triples.
    """
    if z <= 1.0e-4:
        return 0.0
    m = MATS[lane]
    b = math.sqrt(max(z * (2.0 * R_WHEEL - z), CELL * CELL))
    p = min((m.k_c / b + m.k_phi) * z ** m.n, P_BULLDOZE_CAP[lane])
    compact = C_COMPACT * (m.k_c + 2.0 * b * m.k_phi) * z ** (m.n + 1.0) / (m.n + 1.0)
    return compact + C_BULLDOZE * z * 2.0 * b * p


# --- the ground ----------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("no CUDA device -- the terrain will run on the CPU and this will crawl")

terrain = {}
for name in LANES:
    z0, z1 = LANE_Z[name]
    ny = int(round((z1 - z0) / CELL)) + 1
    terrain[name] = DeformableTerrain((X0, -z1), CELL, (NX, ny), K=1,
                                      material=MATS[name], device=device, c_max=4)
    print(f"  {name:5s} {terrain[name]}")
cells = sum(t.nx * t.ny for t in terrain.values())
print(f"  {cells:,} cells total")


def bilinear(t, flat, x, y):
    """Bilinear lookup into one of a terrain's [K, nx, ny] grids, [K, P] in/out.

    DeformableTerrain.heights() is this against `h`; the coupling wants the same
    gather against `grade`, because the road we hand the suspension is measured
    from the bearing datum, not from whatever the last pass left on the surface.
    """
    fx = ((x - t.origin_t[:, 0:1]) / t.cell).clamp(0.0, t.nx - 1.0001)
    fy = ((y - t.origin_t[:, 1:2]) / t.cell).clamp(0.0, t.ny - 1.0001)
    ix, iy = fx.long(), fy.long()
    tx, ty = fx - ix, fy - iy
    gat = lambda a, b: flat.gather(1, a * t.ny + b)                     # noqa: E731
    return ((1 - tx) * (1 - ty) * gat(ix, iy) + tx * (1 - ty) * gat(ix + 1, iy)
            + (1 - tx) * ty * gat(ix, iy + 1) + tx * ty * gat(ix + 1, iy + 1))


def lane_at(x, z):
    """The lane under world (x, z), or None -- and None PAST THE LANE ENDS in x
    too, or a car that leaves the far end of the strip would carry soft-ground
    handling (and its drag) onto the rigid ground forever."""
    if not (X0 <= x <= X0 + (NX - 1) * CELL):
        return None
    for name in LANES:
        lo, hi = LANE_Z[name]
        if lo <= z <= hi:
            return name
    return None


# --- the vehicle ---------------------------------------------------------------

world = tp.PhysxWorld()

# The rigid ground the lanes are bedded into: what a wheel that leaves the
# terrain falls back onto (the scene query still runs under every wheel, and is
# what the road override stands in for). 3 cm below the lane datum, so driving
# off the end of a lane is a kerb rather than a cliff.
rigid = tp.Mesh(tp.BoxGeometry(400.0, 0.4, 400.0),
                standard_material(0x2a2a26, 0.95))
rigid.position.y = RIGID_Y - 0.2
rigid.receive_shadow = True
world.add_static(rigid)

# tire_friction is the ceiling on the RIGID fallback and on anything the road
# override does not cover. The C++ demo's 2.0 out-grips the geometry -- a 1.65 m
# track under a CoM ~0.85 m up rolls at about 1 g, and 2.0 of mu reaches 2 --
# so this demo runs 1.1 everywhere. On the lanes it does not matter (the
# Mohr-Coulomb override is 0.37-0.75 and slides long before the car tips); on
# the clay strip and the fallback it is the difference between a scrub and a
# roll. Measured: full lock at 50 km/h on clay peaks at 2.2 degrees of tilt.
vehicle = tp.PhysxVehicle(world,
                          chassis_width=1.95, chassis_height=1.4, chassis_length=4.4,
                          chassis_mass=CHASSIS_MASS,
                          wheelbase=2.66, track_width=1.65, wheel_radius=R_WHEEL,
                          driven_wheels=[True, True, True, True],
                          max_throttle_torque=1500.0,
                          tire_friction=FALLBACK_MU,
                          longitudinal_stiffness=100_000.0,
                          suspension_travel=0.3, suspension_stiffness=35_000.0,
                          suspension_damping=4500.0, suspension_attachment_y=-0.4,
                          wheel_damping_rate=1.5,
                          position=SPAWN_POS, rotation=SPAWN_ROT)


def qrot(q, v):
    """Rotate the numpy vec3 `v` by a threepp Quaternion."""
    u = np.array([q.x, q.y, q.z])
    t = 2.0 * np.cross(u, v)
    return v + q.w * t + np.cross(u, t)


# --- scene ---------------------------------------------------------------------

if not GL and not tp.vulkan_available():
    print("vulkan not available on this machine; falling back to OpenGL")
    GL = True

canvas = tp.Canvas("threepp x warp - mud & snow drive", width=WIDTH, height=HEIGHT,
                   antialiasing=AA, vsync=VSYNC, headless=SHOT or BENCH)
renderer = tp.GLRenderer(canvas) if GL else tp.VulkanRenderer(canvas)
if GL:
    renderer.shadow_map_enabled = True
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 0.95

scene = tp.Scene()
# The backdrop IS the haze: background and fog the same warm grey-brown, so the
# far hills dissolve into it and there is no horizon LINE anywhere to fight the
# yard for attention (warp_mudsnow_mpm.py does the same at 1/100 the scale).
# Global fog, not a per-material one -- some Vulkan paths ignore those.
HAZE = 0x3a3329
scene.background = HAZE
scene.set_fog(HAZE, 55.0, 300.0)

# Low warm key raking ACROSS the lanes -- along z, while the lanes run along x.
# Berms are centimetres tall; under a high sun they vanish, and under a low one
# every rut wall throws a shadow as long as it is deep. 13 degrees of elevation
# is the compromise: shallow enough that a 3 cm berm draws a 12 cm shadow, steep
# enough that the car's own shadow does not lie down the lane and cover them.
# +z, not -z, and that sign is the whole shot: the acceptance cameras look
# ACROSS the lanes from the clay strip, so a key from behind them lights the
# face of every rut wall and hides its shadow, while a key from beyond throws
# each shadow toward the lens. Same relief, twice the contrast.
SUN_DIR = (-9.0, 8.5, 34.0)
scene.add(tp.HemisphereLight(0xc4cfdc, 0x39352e, 0.80))
sun = tp.DirectionalLight(0xffdca8, 4.0)
sun.position.set(*SUN_DIR)
sun.cast_shadow = True
sun.set_shadow_frustum(-40.0, 40.0, 40.0, -40.0)
sun.set_shadow_bias(-0.0008)
scene.add(sun)


def make_dusk_hdr(path, key_dir, W=512, HH=256):
    """A low-sun haze dome, written as a Radiance .hdr: what the paint reflects.

    The Evoque is the only glossy thing in the yard, and with no environment it
    reads as white plastic -- the body reflects nothing, so its panels have no
    edges. This is the rest of a low sun sky: a warm band low at the key's
    azimuth, cool dim blue overhead, and a dark warm ground half so horizontal
    panels do not mirror a sky that is underneath them.

    Deliberately carries NO sun DISC. The renderer extracts a disc out of an
    HDRI into an analytic light by default, and the scene's DirectionalLight is
    already the one sun -- a disc here would be a second one.
    """
    j = np.arange(HH).reshape(HH, 1)
    i = np.arange(W).reshape(1, W)
    theta = (j / HH) * math.pi
    phi = (i / W) * 2 * math.pi - math.pi
    y = np.broadcast_to(np.cos(theta), (HH, W))
    sin_t = np.sin(theta)
    d = np.stack([sin_t * np.cos(phi), y, sin_t * np.sin(phi)], axis=-1)
    up = np.clip(y, 0.0, 1.0)[..., None] ** 0.75
    sky = np.array([0.30, 0.26, 0.21]) * (1.0 - up) + np.array([0.07, 0.10, 0.17]) * up
    k = np.float32(key_dir)
    k = k / np.linalg.norm(k)
    ang = np.arccos(np.clip((d * k).sum(-1), -1.0, 1.0))
    sky = sky + np.exp(-(ang / math.radians(42.0)) ** 2)[..., None] \
        * np.array([1.05, 0.66, 0.33])
    sky = np.where((y < 0)[..., None], np.array([0.045, 0.040, 0.034]), sky)
    rgb = np.maximum(sky, 0.0)
    m = rgb.max(axis=2)
    mask = m >= 1e-32
    mant, exp = np.frexp(np.where(mask, m, 1.0))
    scale = np.where(mask, mant * 256.0 / np.where(mask, m, 1.0), 0.0)
    rgbe = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    for c in range(3):
        rgbe[..., c] = np.clip(rgb[..., c] * scale, 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(mask, np.clip(exp + 128, 0, 255), 0).astype(np.uint8)
    # An RGBE row starting (2, 2, <128) reads as "adaptive RLE" to a .hdr
    # reader; nudge the one pixel that could fake that signature.
    if rgbe[0, 0, 0] == 2 and rgbe[0, 0, 1] == 2 and rgbe[0, 0, 2] < 128:
        rgbe[0, 0, 0] = 3
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (HH, W))
        f.write(rgbe.tobytes())
    return path


try:
    import tempfile
    scene.environment = tp.RGBELoader().load(make_dusk_hdr(
        os.path.join(tempfile.gettempdir(), "threepp_mudsnow_drive_sky.hdr"), SUN_DIR))
except Exception as e:                              # noqa: BLE001 - garnish only
    print(f"  note: no environment ({e})")

# Mud eats the sunlight: rough AND weak-specular, so the raking key gives it a
# wet sheen in patches instead of a plastic gloss. Snow is the opposite --
# bright and near-lambertian, so what you read on it is pure shape. Clay is
# packed dry earth: lighter than mud and flatter than either.
mud_mat = tp.MeshPhysicalMaterial()
mud_mat.color = 0x40291a
mud_mat.roughness = 0.86
mud_mat.specular_intensity = 0.14
snow_mat = standard_material(0xdde5f0, 0.97)
clay_mat = standard_material(0x5a5142, 0.96)
LANE_MAT = {"mud": mud_mat, "snow": snow_mat, "clay": clay_mat}

# Micro-relief, as a slope bent into the published normal rather than geometry.
# A 5 cm grid over 48 m of lane is dead flat between the ruts, and dead flat is
# what makes the mud read as brown gloss paint under a raking key. This is the
# same trick warp_mudsnow_mpm.py's `grain` plays, and it costs four noise
# lookups a vertex on the strips a wheel actually touched. (freq 1/m, slope) --
# mud is gloopy, so large soft lumps; snow is crystalline, so fine and strong.
LANE_GRAIN = {"mud": (1.4, 0.22), "snow": (5.5, 0.13), "clay": (2.4, 0.19)}


# A lane is drawn as a row of STRIPS, not one mesh, and the ground is published
# through the plain attribute path by default. The strips are a real choice:
# the host route republishes only the strips a wheel touched (Strip.publish),
# and 16 m of lane is the granularity that makes that cheap. `--interop` stays
# opt-in only because the host route runs everywhere -- the zero-copy path
# needs an NVIDIA driver on the other end of the export.
#
# (Historical note: both routes used to be capped by a VulkanRenderer defect --
# enable_vertex_interop left the RT-side GeometryDescs pointing at the freed
# buffers of the record it swaps out at arm time, which read as a >65k-vertex
# "ceiling" and a device loss once a lane had been driven on. Fixed in
# VulkanCoreGeometry.cpp (the desc republish in enableVertexInterop); merged
# 154k-vertex lanes now arm and drive fine, so MAX_INTEROP_VERTS is a strip
# granularity knob, not a correctness limit.)
#
# Strips share their boundary column so there is no seam, and take their
# normals from the FULL grid so the shading crosses the joins too.
MAX_INTEROP_VERTS = 60_000


@wp.kernel
def strip_surface(h: wp.array3d(dtype=float),
                  i0: int, ox: float, oy: float, cell: float, z0: float,
                  gfreq: float, gamp: float, gseed: int,
                  pos: wp.array(dtype=wp.vec3),
                  nrm: wp.array(dtype=wp.vec3)):
    """terrain_deform's `surface` kernel over one column range of the grid.

    Same y-up map ((x, y, height) -> (x, height, -y), a proper rotation, so the
    index buffer's winding still faces out) and the same central-difference
    normal -- but the difference reads the whole grid while the write is
    strip-local, which is what makes the seam between two strips invisible.

    Plus the grain (see LANE_GRAIN): the material's micro-slope, added to the
    grid's own slope before the normal is built, because grain IS extra slope.
    Shading only -- the position written is exactly the grid's, so the ground
    the camera sees is still the ground the wheels read.
    """
    s, j = wp.tid()
    i = i0 + s
    nx = h.shape[1]
    ny = h.shape[2]
    ia = wp.min(i + 1, nx - 1)
    ib = wp.max(i - 1, 0)
    ja = wp.min(j + 1, ny - 1)
    jb = wp.max(j - 1, 0)
    dhx = (h[0, ia, j] - h[0, ib, j]) / (float(ia - ib) * cell)
    dhy = (h[0, i, ja] - h[0, i, jb]) / (float(ja - jb) * cell)
    gx = (ox + float(i) * cell) * gfreq
    gy = (oy + float(j) * cell) * gfreq
    d = float(0.5)
    st = wp.uint32(gseed)
    dhx += gamp * (wp.noise(st, wp.vec2(gx + d, gy)) - wp.noise(st, wp.vec2(gx - d, gy)))
    dhy += gamp * (wp.noise(st, wp.vec2(gx, gy + d)) - wp.noise(st, wp.vec2(gx, gy - d)))
    # A second octave at 3.7x: one scale of lump is a pattern, two is a surface.
    bx = gx * 3.7
    by = gy * 3.7
    ga = gamp * 0.55
    dhx += ga * (wp.noise(st, wp.vec2(bx + d, by)) - wp.noise(st, wp.vec2(bx - d, by)))
    dhy += ga * (wp.noise(st, wp.vec2(bx, by + d)) - wp.noise(st, wp.vec2(bx, by - d)))
    n = wp.normalize(wp.vec3(-dhx, -dhy, 1.0))
    t = s * ny + j
    pos[t] = wp.vec3(ox + float(i) * cell, z0 + h[0, i, j], -(oy + float(j) * cell))
    nrm[t] = wp.vec3(n[0], n[2], -n[1])


class Strip:
    """One column range of a lane as a threepp mesh, published device-side."""

    def __init__(self, terrain, material, i0, ncol, grain=(1.0, 0.0)):
        self.t = terrain
        self.i0 = i0
        self.ncol = ncol
        self.gfreq, self.gamp = grain
        ny = terrain.ny
        self.n = ncol * ny
        self.ox, self.oy = (float(v) for v in terrain.origin_np[0])
        gi, gj = np.meshgrid(np.arange(i0, i0 + ncol), np.arange(ny), indexing="ij")
        p0 = np.stack([self.ox + gi * terrain.cell,
                       np.full_like(gi, terrain.z0, dtype=np.float32),
                       -(self.oy + gj * terrain.cell)], axis=-1)
        a = ((gi[:-1, :-1] - i0) * ny + gj[:-1, :-1]).ravel()
        faces = np.stack([a, a + ny, a + ny + 1, a, a + ny + 1, a + 1],
                         axis=1).astype(np.uint32)
        self.geometry = tp.BufferGeometry()
        self.geometry.set_attribute("position",
                                    np.ascontiguousarray(p0.reshape(-1, 3), np.float32))
        self.geometry.set_attribute("normal",
                                    np.tile(np.float32([0, 1, 0]), (self.n, 1)))
        self.geometry.set_index(np.ascontiguousarray(faces.reshape(-1), np.uint32))
        self.mesh = tp.Mesh(self.geometry, material)
        self.mesh.frustum_culled = False      # the vertices move under the renderer
        self.mesh.receive_shadow = True
        self.vk = None
        self._host = None
        self.dirty = True
        self.x0 = self.ox + i0 * terrain.cell
        self.x1 = self.ox + (i0 + ncol - 1) * terrain.cell

    def launch(self, pos, nrm):
        wp.launch(strip_surface, dim=(self.ncol, self.t.ny), device=self.t._wp_device,
                  inputs=[self.t.h, self.i0, self.ox, self.oy, self.t.cell, self.t.z0,
                          self.gfreq, self.gamp, 0x5eed],
                  outputs=[pos, nrm])

    def arm(self, renderer):
        if not INTEROP or VkInteropArray is None                 or not hasattr(renderer, "enable_vertex_interop"):
            return False
        # Grid vertices keep their identity frame to frame, so the renderer's
        # per-vertex motion vectors are real motion: leave stable_correspondence
        # alone (a re-triangulating producer is the case that must opt out).
        h = renderer.enable_vertex_interop(self.mesh, self._on_frame,
                                          validate=not NOSAN)
        if h is None:
            return False
        (pos_h, pos_b), (nrm_h, nrm_b) = h
        try:
            self.vk = (VkInteropArray(pos_h, pos_b, wp.vec3, self.n, device),
                       VkInteropArray(nrm_h, nrm_b, wp.vec3, self.n, device))
        except Exception as e:                          # noqa: BLE001 - fall back
            print(f"  note: CUDA import failed ({e}) -- host route")
            renderer.disable_vertex_interop(self.mesh)
            self.vk = None
            return False
        self._on_frame()
        # Ordered teardown: the CUDA mappings go before the renderer frees the
        # memory they point at.
        atexit.register(self._release, renderer)
        return True

    def _on_frame(self):
        """Inside render(), post-fence and pre-record. The synchronize is
        MANDATORY: host ordering is all that sequences this write against the
        Vulkan frame that reads it."""
        self.launch(self.vk[0].array, self.vk[1].array)
        wp.synchronize_device(device)

    def publish(self):
        """Host route: device -> host -> geometry, only for a dirty strip.

        A height grid only changes where a wheel is, and a strip is 16 m of
        lane: republishing all eight every frame is 32 ms of copy for 4 m of
        change. Marked dirty under a wheel, plus one strip a frame round-robin
        so a slumping berm the car has already left still catches up.
        """
        if self.vk is not None or not self.dirty:
            return
        self.dirty = False
        if self._host is None:
            self._host = (wp.zeros(self.n, dtype=wp.vec3, device=self.t._wp_device),
                          wp.zeros(self.n, dtype=wp.vec3, device=self.t._wp_device))
        self.launch(*self._host)
        self.geometry.update_attribute("position", self._host[0].numpy())
        self.geometry.update_attribute("normal", self._host[1].numpy())

    def _release(self, renderer):
        if self.vk is None:
            return
        pair, self.vk = self.vk, None
        for a in pair:
            a.close()
        renderer.disable_vertex_interop(self.mesh)


def make_strips(name):
    t = terrain[name]
    per = max(2, min(t.nx, MAX_INTEROP_VERTS // t.ny))
    n_strip = int(math.ceil((t.nx - 1) / (per - 1)))
    per = int(math.ceil((t.nx - 1) / n_strip)) + 1
    out = []
    for k in range(n_strip):
        i0 = k * (per - 1)
        out.append(Strip(t, LANE_MAT[name], i0, min(per, t.nx - i0), LANE_GRAIN[name]))
    return out


strips = [s for name in LANES for s in make_strips(name)]
for s in strips:
    scene.add(s.mesh)
print(f"  {len(strips)} lane strips, {max(s.n for s in strips):,} vertices each at most")
_rr = 0


def mark_dirty(hub):
    """Whichever strips a wheel is standing in, plus one more, need a republish."""
    global _rr
    for s in strips:
        for i in range(4):
            if s.x0 - 0.6 <= hub[i, 0] <= s.x1 + 0.6:
                s.dirty = True
                break
    strips[_rr % len(strips)].dirty = True
    _rr += 1

# --- the proving ground --------------------------------------------------------
# The lanes used to sit on a 400 m grey slab under a white sky, which is the one
# thing about this demo that was wrong on sight: no scale, no horizon, no world.
# What replaces it is ONE heightfield -- dead flat and EXACTLY at the collider
# height out to the perimeter, because the visual ground and the ground the car
# drives on have to be the same ground, and beyond that rising into hills the
# fog takes away. One mesh, so there is no apron/surround seam to hide, and the
# lattice is stretched cubically so the metres beside the yard are fine and the
# quarter-kilometre out at the rim is cheap: 0.7 m spacing at the lanes, 7 m at
# the edge, 33k vertices for a 640 m yard.

SURROUND_R = 320.0                # to the rim
APRON_R = 62.0                    # flat, drivable, matches the rigid collider
rng = np.random.default_rng(7)


def vnoise(x, y, freq, seed, n=64):
    """Value noise on a wrapping n x n lattice: bilinear, smoothstep-faded."""
    g = np.random.default_rng(seed).random((n, n))
    fx, fy = np.asarray(x) * freq, np.asarray(y) * freq
    i0, j0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    tx, ty = fx - i0, fy - j0
    tx = tx * tx * (3.0 - 2.0 * tx)
    ty = ty * ty * (3.0 - 2.0 * ty)
    a = lambda di, dj: g[(i0 + di) % n, (j0 + dj) % n]                   # noqa: E731
    return ((1 - tx) * (1 - ty) * a(0, 0) + tx * (1 - ty) * a(1, 0)
            + (1 - tx) * ty * a(0, 1) + tx * ty * a(1, 1))


def fbm(x, y, freq, seed, octaves=4):
    v, amp, tot = 0.0, 1.0, 0.0
    for k in range(octaves):
        v = v + amp * vnoise(x, y, freq * 2.0 ** k, seed + 977 * k)
        tot += amp
        amp *= 0.5
    return v / tot


def build_surround():
    """The lattice, and the three meshes cut out of it.

    ONE material per mesh, because per-vertex colour would be the natural way
    to grade dirt into snow into haze and this renderer's deferred path does
    not have it: gbuffer.vert has no colour binding and writes vec3(1) (only
    the GPU-driven indirect path reads a "color" attribute). So the yard is
    split into regions instead, and the cuts are put where a material change is
    something rather than nothing -- the snow line, and the perimeter the hills
    start from.
    """
    n = 181
    u = np.linspace(-1.0, 1.0, n)
    ax = (0.20 * u + 0.80 * u ** 3) * SURROUND_R
    X, Z = np.meshgrid(ax, ax, indexing="ij")
    r = np.hypot(X, Z)

    # Nothing at all inside the perimeter: the apron is the collider's plane,
    # to the millimetre. Past it the hills ramp in over 50 m (a cliff at the
    # perimeter would read as a wall) and the rim lifts above the eyeline so
    # there is a silhouette in the haze instead of an edge of the world.
    t = np.clip((r - APRON_R) / 50.0, 0.0, 1.0) ** 2
    hills = 26.0 * fbm(X, Z, 1.0 / 190.0, 11) - 9.5
    hills += 7.0 * fbm(X, Z, 1.0 / 52.0, 31) - 3.0
    h = RIGID_Y + t * np.maximum(hills, 0.0)
    h += t * 14.0 * np.clip((r - 130.0) / 180.0, 0.0, 1.0) ** 1.4

    dhx = np.gradient(h, ax, axis=0)
    dhz = np.gradient(h, ax, axis=1)
    # The apron is geometrically flat and would mirror the sky as one sheet, so
    # its RELIEF is a slope bent into the normal -- 20 m lumps, the biggest
    # thing this lattice can carry without aliasing. Same trick as the lanes'
    # grain, one scale up.
    dhx = dhx + 0.13 * (fbm(X + 91.0, Z, 0.05, 53, 3) - 0.5) * 2.0
    dhz = dhz + 0.13 * (fbm(X, Z + 47.0, 0.05, 59, 3) - 0.5) * 2.0
    nrm = np.stack([-dhx, np.ones_like(h), -dhz], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)

    pos = np.ascontiguousarray(np.stack([X, h, Z], axis=-1).reshape(-1, 3), np.float32)
    nrm = np.ascontiguousarray(nrm.reshape(-1, 3), np.float32)
    a = (np.arange(n - 1)[:, None] * n + np.arange(n - 1)[None, :]).ravel()
    quads = np.stack([a, a + 1, a + n + 1, a, a + n + 1, a + n], axis=1)
    # Region tests on the quad's own centre, so a quad belongs to exactly one
    # mesh and the three meshes tile the lattice with no gap and no overlap.
    cx = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[:-1, 1:] + X[1:, 1:]).ravel()
    cz = 0.25 * (Z[:-1, :-1] + Z[1:, :-1] + Z[:-1, 1:] + Z[1:, 1:]).ravel()
    cr = np.hypot(cx, cz)
    yard = cr <= APRON_R + 7.0
    # A snow LINE, not a rectangle: the threshold wanders a few metres with the
    # ground so the field's edge reads as lying snow rather than a rug.
    snowline = LANE_Z["snow"][0] - 1.6 - 7.0 * fbm(cx, cz, 0.035, 89, 3)
    snow_ground = yard & (cz < snowline)

    def mesh_of(mask, material):
        idx = quads[mask]
        g = tp.BufferGeometry()
        g.set_attribute("position", pos)
        g.set_attribute("normal", nrm)
        g.set_index(np.ascontiguousarray(idx.ravel(), np.uint32))
        m = tp.Mesh(g, material)
        m.receive_shadow = True
        return m

    # None of the three casts: the sun's shadow frustum is +-40 m (tight, so
    # that a 3 cm berm gets a shadow map texel), the apron is flat, and the
    # hills start at 62 m. Everything they could cast falls outside the map, so
    # cast_shadow on them buys a 33k-vertex depth pass and nothing else.
    return (mesh_of(yard & ~snow_ground, apron_mat),
            mesh_of(snow_ground, field_snow_mat),
            mesh_of(~yard, hill_mat))


# The yard floor is churned wet dirt: darker than the clay strip so the lanes
# read as lighter cuts into it rather than carpets laid on it, and dull enough
# that the raking key does not turn 12,000 m2 of it into a mirror.
apron_mat = standard_material(0x2b241c, 0.96)
field_snow_mat = standard_material(0xc9d2de, 0.97)
# The hills carry the haze's own hue: half the job of a backdrop is being the
# thing the fog dissolves, and a backdrop in a different colour never does.
hill_mat = standard_material(0x554b3e, 0.98)
for part in build_surround():
    scene.add(part)


def boulder(radius, seed, squash=0.72):
    """A rock: an icosphere pushed around by three beat frequencies.

    IcosahedronGeometry is unindexed, so the displacement -- a pure function of
    the vertex position -- lands identically on every copy of a shared corner
    and the shell stays closed, while compute_vertex_normals then gives it the
    faceted read a broken rock has and a UV sphere never does.
    """
    g = tp.IcosahedronGeometry(1.0, 2)
    p = g.get_attribute("position")
    r = np.random.default_rng(seed)
    d = np.ones(len(p))
    for freq, amp in ((1.15, 0.30), (2.45, 0.15), (5.10, 0.065)):
        ph = r.uniform(0.0, 6.283, 3)
        d += amp * (np.sin(freq * p[:, 0] + ph[0]) * np.sin(freq * p[:, 1] + ph[1])
                    * np.sin(freq * p[:, 2] + ph[2]))
    q = p * d[:, None] * radius
    q[:, 1] *= squash
    g.set_attribute("position", np.ascontiguousarray(q, np.float32))
    g.compute_vertex_normals()
    return g


# Speed perception: without something standing still beside the lane, 70 km/h
# over a flat grid reads as 20. A weathered post line marks the perimeter and
# the lane shoulders; boulders scatter over the verges, dark and wet where they
# sit by the mud and snow-capped on the snow side. Built once -- adding or
# removing scene entries mid-drive rebuilds the Vulkan descriptor set and drops
# the TAA history.
#
# And built MERGED: ~170 pebbles and posts as 170 scene entries is 170 draws
# and 170 shadow-map draws for scenery that never moves relative to anything,
# and it measured as most of this pass's frame cost. Baked into one geometry a
# material, it is five draws.
post_mat = standard_material(0x2a2521, 0.92)
cap_mat = standard_material(0x9aa2ac, 0.9)
wet_rock = standard_material(0x312b25, 0.72)
dry_rock = standard_material(0x4b463d, 0.95)
snow_rock = standard_material(0x9fa8b2, 0.93)


class Batch:
    """Accumulates transformed copies of small geometries into one mesh."""

    def __init__(self, material, cast=False):
        self.material = material
        self.cast = cast
        self.pos, self.nrm, self.idx, self.n = [], [], [], 0

    def add(self, geometry, offset, rot_y=0.0, tilt=(0.0, 0.0)):
        p = geometry.get_attribute("position")
        nm = geometry.get_attribute("normal")
        i = geometry.get_index()
        ca, sa = math.cos(rot_y), math.sin(rot_y)
        cx, sx = math.cos(tilt[0]), math.sin(tilt[0])
        cz, sz = math.cos(tilt[1]), math.sin(tilt[1])
        # Rz . Rx . Ry -- the same order three.js applies an object's XYZ Euler
        # plus the yaw, so a batched copy sits exactly where the loose Mesh did.
        m = (np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
             @ np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
             @ np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]]))
        self.pos.append(p @ m.T + np.asarray(offset))
        self.nrm.append(nm @ m.T)
        self.idx.append((np.arange(len(p)) if i is None else i) + self.n)
        self.n += len(p)

    def mesh(self):
        g = tp.BufferGeometry()
        g.set_attribute("position",
                        np.ascontiguousarray(np.concatenate(self.pos), np.float32))
        g.set_attribute("normal",
                        np.ascontiguousarray(np.concatenate(self.nrm), np.float32))
        g.set_index(np.ascontiguousarray(np.concatenate(self.idx).ravel(), np.uint32))
        m = tp.Mesh(g, self.material)
        m.cast_shadow = self.cast
        m.receive_shadow = True
        return m


# The shoulder posts are inside the +-40 m shadow frustum and cast; the 62 m
# perimeter ring is outside it and cannot, so it is a separate batch that skips
# the depth pass entirely.
near_posts = Batch(post_mat, cast=True)
far_posts = Batch(post_mat)
caps = Batch(cap_mat, cast=True)
rocks = {id(m): Batch(m, cast=True) for m in (wet_rock, dry_rock, snow_rock)}
rock_mats = {id(m): m for m in (wet_rock, dry_rock, snow_rock)}

for x in np.arange(X0 + 2.0, X1, 6.0):
    for z in (LANE_Z["mud"][1] + 0.9, LANE_Z["snow"][0] - 0.9):
        hgt = float(rng.uniform(1.25, 1.65))
        near_posts.add(tp.CylinderGeometry(0.06, 0.10, hgt, 8),
                       (float(x), 0.5 * hgt + RIGID_Y, float(z)),
                       tilt=(float(rng.normal(0.0, 0.05)),      # nothing stands
                             float(rng.normal(0.0, 0.05))))     # straight
        if z < 0.0:                                             # snow on the tops
            cap = tp.SphereGeometry(0.10, 8, 6)
            cap.scale(1.0, 0.45, 1.0)
            caps.add(cap, (float(x), hgt + RIGID_Y - 0.02, float(z)))
# The perimeter itself, so the flat apron ends at something rather than at the
# fog: a ring of posts on the boundary the hills start from.
for a in np.arange(0.0, 2.0 * math.pi, 2.0 * math.pi / 96.0):
    hgt = float(rng.uniform(1.0, 1.4))
    far_posts.add(tp.CylinderGeometry(0.06, 0.09, hgt, 6),
                  (float(APRON_R * math.cos(a)), 0.5 * hgt + RIGID_Y,
                   float(APRON_R * math.sin(a))),
                  tilt=(0.0, float(rng.normal(0.0, 0.07))))

for k in range(72):
    r = float(rng.uniform(0.30, 1.15))
    side = 1.0 if rng.integers(0, 2) else -1.0
    z = float(rng.uniform(12.0, 46.0)) * side
    x = float(rng.uniform(X0 - 26.0, X1 + 26.0))
    mat = snow_rock if (z < 0.0 and rng.random() < 0.7) else         (wet_rock if abs(z) < 22.0 and side > 0 else dry_rock)
    rocks[id(mat)].add(boulder(r, 400 + k), (x, r * 0.34 + RIGID_Y, z),
                       rot_y=float(rng.uniform(0.0, 6.28)))
# NOTE: these boulders are scenery only. Making the big ones solid is one line
# (a hidden sphere proxy through world.add_static -- add_static reads
# Box/Sphere/Capsule geometry, not a displaced icosahedron), and it was tried
# and taken back out: four more static actors change the order PhysX
# accumulates its scene in, and the acceptance scripts' numbers moved in the
# third digit. This phase is the LOOK pass, and the sim has to print what it
# printed before it.
props = tp.Group()
for b in (near_posts, far_posts, caps, *rocks.values()):
    props.add(b.mesh())
scene.add(props)
print(f"  surround: {APRON_R:.0f} m apron, hills to {SURROUND_R:.0f} m")

# --- weather --------------------------------------------------------------------
# One ParticleField in Renderer ownership: the CPU writes a 64-byte emitter
# record a frame and never touches a flake, so the whole snowfall costs one
# scene entry and one indirect draw. Built ONCE and left in the scene (adding
# or removing entries mid-drive rebuilds the Vulkan descriptor set and throws
# away the TAA history), so there is no toggle -- parking it is
# set_live_count(0).
#
# The brief asked for snowfall over the SNOW LANE. That was tried first and it
# is worse: an 8 m wide emitter slab is a rectangular column of white specks
# with two visible vertical edges, and the drive camera sits on the clay strip
# looking straight through one of them. Snow falls out of a sky, not out of a
# box over one strip, so this follows the camera toroidally instead -- no edge
# anywhere, and which lane is the snow one is said by the GROUND, which is
# where this demo says everything else too.
# 3.6k flakes in a 28 m box, not 30k in a 56 m one: a flake has to subtend a
# pixel or two to read as a flake, and a wide box spends its whole budget on
# 40 m specks that composite into a star field. Close and sparse beats far and
# dense every time -- and dim beats bright: the quads are ADDITIVE against a
# dark dusk sky and outside auto-exposure, so white at 0.35 intensity reads as
# a bokeh star field that upstages the car (field report). Snow at dusk is
# sky-coloured, not white; it should be felt, not counted.
SNOW_CAP = 3_600
SNOW_HALF = 14.0
snowfall = None
if not GL:
    _sc = tp.ParticleField.Config()
    _sc.capacity = SNOW_CAP
    _sc.ownership = tp.ParticleField.Ownership.Renderer
    _sc.w_semantic = tp.ParticleField.WSemantic.Radius
    _sc.uniform_radius = 0.024
    snowfall = tp.ParticleField.create(_sc)
    snowfall.frustum_culled = False
    # Billboards only, unlit and additive. A lit mesh proxy would be the right
    # answer for flakes a metre from the lens and this camera is never there;
    # what it would cost is a G-buffer draw for every flake so that the two in
    # focus read as crystals.
    snowfall.set_billboard_repr(tp.Color(0.62, 0.64, 0.70), tp.Color(0.48, 0.51, 0.58),
                                0.22, 0.90)
    _sb = snowfall.billboard_repr
    _sb.lod_near = 0.0
    _sb.lod_fade = 0.0
    _sb.softness = 0.9
    _sb.fade_power = 0.0
    _sb.size_taper = 0.0
    _sb.bright_jitter = 0.35
    _sb.near_fade = 6.0                           # a flake at arm's length is a
                                                  # white plate across the frame
    _sb.stretch_seconds = 0.05
    _sb.stretch_max = 3.0
    _sb.stretch_max_screen = 0.02
    _sb.glow = 0.0
    _se = snowfall.emitter
    z_snowc = 0.5 * (LANE_Z["snow"][0] + LANE_Z["snow"][1])
    _se.spawn_center = tp.Vector3(0.0, 15.0, 0.0)
    _se.spawn_half_extent = tp.Vector3(SNOW_HALF, 0.5, SNOW_HALF)
    _se.velocity = tp.Vector3(0.0, -1.9, 0.0)     # a flake terminal-velocities slowly
    _se.speed_spread = 0.4
    _se.wind = tp.Vector3(0.55, 0.0, -0.2)
    _se.drift_amplitude = 0.28                    # the tumble, which is what says snow
    _se.drift_frequency = 0.5
    _se.drift_scale = 3.0
    _se.lifetime = 9.0
    _se.duty_cycle = 1.0
    _se.size = 0.024
    _se.size_jitter = 0.45
    _se.seed = 20260830
    _se.follow = True
    _se.follow_snap = 2.0
    snowfall.set_emitter(_se)
    snowfall.set_emitter_time(0.0, DT)
    scene.add(snowfall)

_weather_t = 0.0


def weather(dt, eye=None):
    """Advance the snowfall. Absolute time, so a still can seek to any t.

    The follow box is centred on the CAMERA -- weather that follows anything
    else is not weather -- and set_follow_center snaps, so the curtain does not
    swim under a moving chase cam.
    """
    global _weather_t
    if snowfall is None:
        return
    _weather_t += dt
    if eye is not None:
        snowfall.set_follow_center(tp.Vector3(*(float(v) for v in eye)))
    snowfall.set_emitter_time(_weather_t, dt)

# --- the rover -----------------------------------------------------------------


def data_dir():
    """threepp_data's checkout. THREEPP_DATA_DIR wins; otherwise the usual
    places -- note the checkout is commonly named with a hyphen."""
    env = os.environ.get("THREEPP_DATA_DIR")
    if env and os.path.isdir(env):
        return env
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for name in ("threepp-data", "threepp_data"):
        cand = os.path.join(os.path.dirname(repo), name)
        if os.path.isdir(cand):
            return cand
    return ""


chassis = tp.Group()
scene.add(chassis)

# PhysX index -> the model's own wheel tag. The model labels its wheels from
# inside the car, so its "FL" sits at +x, which is where PhysX puts wheel 0
# (front-RIGHT, +x +z). The tags therefore look swapped and are not.
WHEEL_TAG = ("WheelFL", "WheelFR", "WheelBL", "WheelBR")
wheel_rigs = [tp.Group() for _ in range(4)]
for rig in wheel_rigs:
    chassis.add(rig)

brake_mat = reverse_mat = None
model_path = os.path.join(data_dir(), "models", "gltf",
                          "2015_land-rover_range_rover_evoque_coupe", "scene.gltf")
if os.path.isfile(model_path):
    body = tp.ModelLoader().load(model_path)
    body.scale.set(100.0, 100.0, 100.0)     # the gltf bakes a 0.01 at its root
    # The model's contact patches sit at its own y = 0; the PhysX chassis centre
    # rides wheel_radius + |attachment_y| + travel - jounce = 0.996 m above the
    # road. Drop the shell by a metre and the wheel wells land on the rigs.
    body.position.y = -1.0
    chassis.add(body)

    parts = [[] for _ in range(4)]
    lamps = {"lights_position_back": [], "lights_reverse": []}

    def _sort(o):
        if type(o).__name__ != "Mesh":
            return
        # The .gltf packs AO and metal-roughness into ONE texture (R = AO,
        # G = rough, B = metal) and never authored R, so an aoMap read is a
        # constant zero and the whole body goes black. Drop it.
        try:
            o.material.ao_map = None
        except Exception:                               # noqa: BLE001 - not all have one
            pass
        for i, tag in enumerate(WHEEL_TAG):
            if tag in o.name:
                parts[i].append(o)
                return
        for key in lamps:
            if key in o.name:
                lamps[key].append(o)

    body.traverse(_sort)

    for i, group in enumerate(parts):
        if not group:
            continue
        # The per-wheel "pivot" nodes are identity transforms -- the wheel
        # positions are baked into the geometry -- so the hub is the centre of
        # the combined vertex AABB, and a clone recentred on it spins about its
        # own axle under a rig driven by wheel_local_pose(). The wheel is
        # axisymmetric about that axle, which is why the model's own axis
        # convention does not need undoing here.
        lo = np.full(3, 1.0e30)
        hi = np.full(3, -1.0e30)
        for part in group:
            a = part.geometry.get_attribute("position")
            lo = np.minimum(lo, a.min(0))
            hi = np.maximum(hi, a.max(0))
        hub = 0.5 * (lo + hi)
        for part in group:
            clone = part.clone()                # clone BEFORE hiding: copy() takes `visible`
            part.visible = False
            clone.visible = True
            clone.position.set(float(-hub[0]), float(-hub[1]), float(-hub[2]))
            wheel_rigs[i].add(clone)

    # Brake and reverse lamps. The model gives them their own meshes but ONE
    # shared emissive material, so they get a fresh material each and can flare
    # independently.
    for key, colour in (("lights_position_back", 0xff2a12), ("lights_reverse", 0xfff2e0)):
        if not lamps[key]:
            continue
        mat = standard_material(0x201a18, 0.6, emissive=colour)
        mat.emissive_intensity = 1.0
        for m in lamps[key]:
            m.set_material(mat)
        if key == "lights_position_back":
            brake_mat = mat
        else:
            reverse_mat = mat
else:
    print(f"  note: no Evoque at {model_path} -- driving a box "
          f"(set THREEPP_DATA_DIR)")
    shell = tp.Mesh(tp.BoxGeometry(1.95, 1.2, 4.4), standard_material(0x7a2f24, 0.5, 0.3))
    shell.cast_shadow = True
    chassis.add(shell)
    for rig in wheel_rigs:
        w = tp.Mesh(tp.CylinderGeometry(R_WHEEL, R_WHEEL, 0.3, 18), standard_material(0x1c1c1e, 0.8))
        w.rotate_z(math.pi / 2)
        w.cast_shadow = True
        rig.add(w)

# --- cameras -------------------------------------------------------------------

camera = tp.PerspectiveCamera(58, canvas.aspect(), 0.1, 400)
camera.position.set(-30.0, 4.0, 0.0)
pov = tp.PerspectiveCamera(72, canvas.aspect(), 0.05, 400)
pov.position.set(0.4, 0.45, -0.15)
pov.rotation.y = math.pi           # camera looks down -z; chassis forward is +z
chassis.add(pov)

CHASE_DIST, CHASE_PITCH = 9.5, 0.30
cam_pos = np.array([-30.0, 4.0, 0.0])
cam_tgt = np.array([-20.0, 1.0, 0.0])
view_mode = 0                      # 0 chase, 1 POV, 2 cinematic orbit
cine_t = 0.0


def resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    pov.aspect = camera.aspect
    pov.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(resize)

# --- input ---------------------------------------------------------------------

steer_cmd = 0.0
throttle_cmd = 0.0
brake_cmd = 0.0
edge = {}


def pressed(key):
    """True on the frame `key` goes down (canvas.is_key_down is a level)."""
    down = canvas.is_key_down(key)
    was = edge.get(key, False)
    edge[key] = down
    return down and not was


# --- coupling state ------------------------------------------------------------

w_ema = np.full(4, REST_LOAD)
prev_hub = None                   # full hub positions last frame, for drag direction
dig = np.ones(4)                  # slip-sinkage factor, EMA toward 1 + K_DIG*slip

# Traction control, on by default (the real Evoque has it -- "Terrain Response"
# is exactly this). The direct drive can put 1500 N*m on a wheel whose mud
# ceiling is ~530 N*m, so anything past ~35 % throttle just spins without it.
# Cuts throttle while any contact patch scrubs past TC_SCRUB, recovers when it
# grips. X toggles it off -- at which point flooring it in mud digs the car in,
# which is the other half of the demo.
TC_SCRUB = 2.5                    # m/s of scrub where the cut starts
tc_on = True
tc_cut = 1.0
radii = torch.full((4,), R_WHEEL, device=device)
centers = {n: torch.zeros(1, 4, 3, device=device) for n in LANES}
vels = {n: torch.zeros(1, 4, 3, device=device) for n in LANES}
c_host = {n: np.zeros((1, 4, 3), np.float32) for n in LANES}
v_host = {n: np.zeros((1, 4, 3), np.float32) for n in LANES}
hud = dict(lane=[None] * 4, z=np.zeros(4), mu=np.zeros(4), w=np.zeros(4),
           bek=np.zeros(4), slip=np.zeros(4), util=np.zeros(4), j=np.zeros(4),
           dig=np.ones(4), drag=np.zeros(4), over=[False] * 4)
_readback = 0


def coupled_step(throttle, steer, brake, relax_iters=2):
    """One 1/60 step: ground under the wheels, then PhysX over the top."""
    global w_ema, prev_hub, tc_cut, _readback

    # 1. chassis pose + wheel world centres
    p, q = vehicle.position, vehicle.quaternion
    origin = np.array([p.x, p.y, p.z])
    hub = np.empty((4, 3))
    for i in range(4):
        lp, _ = vehicle.wheel_local_pose(i)
        hub[i] = origin + qrot(q, np.array([lp.x, lp.y, lp.z]))
    fwd = qrot(q, np.array([0.0, 0.0, 1.0]))
    right = qrot(q, np.array([1.0, 0.0, 0.0]))
    v_fwd = vehicle.forward_speed

    # 2. the GRADE under each wheel -- the bearing datum, not the surface. One
    #    gather per lane and one transfer for all of them.
    lane_of = [lane_at(hub[i, 0], hub[i, 2]) for i in range(4)]
    grade = np.zeros(4)
    gathered = []
    for name in LANES:
        idx = [i for i in range(4) if lane_of[i] == name]
        if not idx:
            continue
        t = terrain[name]
        xs = torch.tensor([[hub[i, 0] for i in idx]], device=device, dtype=torch.float32)
        ys = torch.tensor([[-hub[i, 2] for i in idx]], device=device, dtype=torch.float32)
        gathered.append((idx, bilinear(t, t.grade_torch.view(1, t.nx * t.ny), xs, ys)))
    if gathered:
        flat = torch.cat([g[1][0] for g in gathered]).cpu().numpy()
        k = 0
        for idx, _ in gathered:
            for i in idx:
                grade[i] = flat[k]
                k += 1

    # 3. wheel load, smoothed. The road-height/load loop has a frame of lag in
    #    it and rings at the suspension frequency without this.
    load = np.array([vehicle.suspension_force(i) for i in range(4)])
    w_ema += (load - w_ema) * (DT / (0.15 + DT))

    # 4/5. invert Bekker for the sinkage, Mohr-Coulomb for the friction, and
    #      hand the suspension a road at grade - z_eq. The dig factor (updated
    #      in step 6 from last frame's slip) deepens the road under a spinning
    #      wheel: slip sinkage, one frame of lag like the load loop.
    z_eq = np.zeros(4)
    mu = np.zeros(4)
    for i in range(4):
        if lane_of[i] is None:
            vehicle.clear_road_override(i)
            dig[i] += (1.0 - dig[i]) * (DT / (DIG_TAU + DT))
            continue
        z_st, mu[i] = sinkage(lane_of[i], w_ema[i])
        z_eq[i] = min(z_st * dig[i], 0.45 * (2.0 * R_WHEEL))
        vehicle.set_road_override(i, float(grade[i] - z_eq[i]), float(mu[i]))

    # 6. carve. The wheels are the module's collider spheres; their bottoms are
    #    already at grade - z_eq, so the imprint cuts the rut to exactly the
    #    Bekker sinkage and the berm is whatever the material would not compact.
    #    Velocity is the CONTACT PATCH slip, not the hub velocity: that is what
    #    the Janosi integral is measured along.
    max_scrub = 0.0
    for name in LANES:
        c, v = c_host[name], v_host[name]
        for i in range(4):
            if lane_of[i] != name:
                c[0, i] = (0.0, -LANE_Z[name][1], 50.0)     # parked well clear
                v[0, i] = (0.0, 0.0, 0.0)
                continue
            c[0, i] = (hub[i, 0], -hub[i, 2], hub[i, 1])
            slip_long = vehicle.wheel_angular_speed(i) * R_WHEEL - v_fwd
            slip_lat = vehicle.tire_lateral_slip(i) * abs(v_fwd)
            s = slip_long * fwd + slip_lat * right
            vy = 0.0 if prev_hub is None else (hub[i, 1] - prev_hub[i, 1]) / DT
            v[0, i] = (s[0], -s[2], vy)
            # Slip sinkage follows the patch's scrub SPEED past a dead zone
            # (see SCRUB_DEAD / SCRUB_REF). Digging DEEPER also requires the
            # wheel to be carrying load: excavation is pressure ejecting
            # material, and an unloaded wheel just flings mud. Without that
            # gate there is a runaway -- the road drops faster than the body
            # can follow, the wheel unloads, spins up, and pins the dig.
            max_scrub = max(max_scrub, math.hypot(s[0], s[2]))
            scrub = max(math.hypot(s[0], s[2]) - SCRUB_DEAD, 0.0)
            target = 1.0 + K_DIG[name] * min(scrub / SCRUB_REF, 1.0)
            target = min(target, DIG_MAX[name])
            if target < dig[i] or w_ema[i] > 0.35 * REST_LOAD:
                dig[i] += (target - dig[i]) * (DT / (DIG_TAU + DT))
        centers[name].copy_(torch.from_numpy(c))
        vels[name].copy_(torch.from_numpy(v))
        terrain[name].deform(centers[name], radii, vels[name], DT)
        terrain[name].relax(relax_iters)

    # 7. motion resistance -- the one soil force actually applied (see the audit
    #    at the top of the file). Opposes each wheel's FULL horizontal travel:
    #    a sideways slide ploughs soil just like forward travel does, which is
    #    why a slide in mud dies instead of sailing. Tapered below 0.5 m/s so it
    #    never pushes a resting car backwards, capped per wheel for stability.
    hud["drag"][:] = 0.0
    for i in range(4):
        if lane_of[i] is None:
            continue
        vh = (np.zeros(3) if prev_hub is None else (hub[i] - prev_hub[i]) / DT)
        vh[1] = 0.0
        speed_h = float(np.linalg.norm(vh))
        f = motion_resistance(lane_of[i], z_eq[i])
        f = min(f, DRAG_CAP_FRAC * max(w_ema[i], 1.0)) * min(speed_h / 0.5, 1.0)
        hud["drag"][i] = f
        if f <= 0.0 or speed_h < 0.05:
            continue
        d = vh / speed_h
        vehicle.add_force_at_pos(tp.Vector3(float(-d[0] * f), 0.0, float(-d[2] * f)),
                                 tp.Vector3(*hub[i]))
    prev_hub = hub.copy()

    # 8. and PhysX runs the car -- through the traction control, which is the
    #    difference between "drivable at any throttle" and "any throttle spins".
    if tc_on:
        err = max_scrub - TC_SCRUB
        if err > 0.0:
            tc_cut = max(0.15, tc_cut - 4.0 * DT * min(err / 4.0, 1.0))
        else:
            tc_cut = min(1.0, tc_cut + 1.5 * DT)
        throttle = throttle * tc_cut
    else:
        tc_cut = 1.0
    vehicle.set_throttle(throttle)
    vehicle.set_steer(steer)
    vehicle.set_brake(brake)
    world.step(DT)

    # Panel telemetry. One small transfer every third frame: the module's own
    # Bekker bearing and Janosi shear next to what PhysX's suspension is
    # carrying. Nothing in the loop reads these back.
    _readback += 1
    if _readback % 3 == 0:
        pack = torch.stack([torch.cat([terrain[n].forces[0, :, 0:3].reshape(-1),
                                       terrain[n].slip_j[0]]) for n in LANES]).cpu().numpy()
        for i in range(4):
            n = lane_of[i]
            hud["bek"][i] = 0.0 if n is None else pack[LANES.index(n)][i * 3 + 2]
            hud["j"][i] = 0.0 if n is None else pack[LANES.index(n)][12 + i]
            if n is not None:
                ft = math.hypot(pack[LANES.index(n)][i * 3], pack[LANES.index(n)][i * 3 + 1])
                hud["util"][i] = ft / max(mu[i] * w_ema[i], 1.0)
    hud["lane"] = lane_of
    hud["z"] = z_eq
    hud["dig"] = dig.copy()
    hud["mu"] = mu
    hud["w"] = load
    hud["slip"] = np.array([vehicle.tire_longitudinal_slip(i) for i in range(4)])
    hud["over"] = [vehicle.road_override_active(i) for i in range(4)]
    return hub


# --- UI ------------------------------------------------------------------------

ui = tp.ImguiContext(canvas, renderer) if not (SHOT or BENCH) else None
fps_ema = 60.0
shot_no = 0
WHEEL_NAME = ("FR", "FL", "RR", "RL")


def bar(v, n=14):
    v = max(0.0, min(1.0, v))
    k = int(round(v * n))
    return "[" + "#" * k + "." * (n - k) + "]"


def draw_ui():
    tp.imgui.set_next_window_pos(8, 8)
    tp.imgui.set_next_window_size(376, 0)
    tp.imgui.begin("Bekker-Wong drive")
    tp.imgui.text(f"{vehicle.forward_speed * 3.6:6.1f} km/h   "
                  f"{'REVERSE' if vehicle.gear == tp.PhysxVehicle.Gear.REVERSE else 'FORWARD'}"
                  f"   {fps_ema:5.1f} fps")
    tp.imgui.text(f"throttle {bar(throttle_cmd)}  brake {bar(brake_cmd)}")
    tp.imgui.text(f"traction control {'ON ' if tc_on else 'OFF'} "
                  f"{bar(tc_cut) if tc_on else '[ floor it and dig ]'}")
    tp.imgui.text(f"steer    {bar(0.5 + 0.5 * steer_cmd)}")
    tp.imgui.separator()
    tp.imgui.text("wh lane  sink   load   mu   slip  util  dig  drag")
    for i in range(4):
        n = hud["lane"][i]
        tp.imgui.text(f"{WHEEL_NAME[i]} {(n or 'rigid'):5s} "
                      f"{hud['z'][i] * 1000:5.1f} {hud['w'][i] / 1000:6.2f} "
                      f"{(hud['mu'][i] if n else FALLBACK_MU):5.2f} "
                      f"{hud['slip'][i]:6.2f} {hud['util'][i]:5.2f} "
                      f"{hud['dig'][i]:4.2f} {hud['drag'][i] / 1000:5.2f}")
    tp.imgui.text("           mm     kN               -         kN")
    if (np.mean(hud["dig"]) > 1.6 and abs(vehicle.forward_speed) < 0.5
            and np.mean(hud["util"]) > 0.9):
        tp.imgui.text("DUG IN -- ease off the throttle and creep out")
    tp.imgui.separator()
    tp.imgui.text("soil bearing vs PhysX suspension load")
    for i in range(4):
        if hud["lane"][i] is None:
            tp.imgui.text(f"{WHEEL_NAME[i]}  -- off the lanes (rigid ground) --")
            continue
        err = 100.0 * (hud["bek"][i] / max(hud["w"][i], 1.0) - 1.0)
        tp.imgui.text(f"{WHEEL_NAME[i]} Bekker {hud['bek'][i]:7.0f} N   "
                      f"PhysX {hud['w'][i]:7.0f} N   {err:+5.1f} %")
    tp.imgui.separator()
    tp.imgui.text("W/S drive  A/D steer  R gear  SPACE handbrake  X tc")
    tp.imgui.text("V pov  C cinematic  BACKSPACE respawn  T reset  F frame")
    tp.imgui.end()


def save_shot(path):
    if GL:
        renderer.render(scene, camera)
        renderer.save_frame(path)
    else:
        renderer.save_frame(scene, active_camera(), path)
    print(f"  wrote {path}")


def active_camera():
    return pov if view_mode == 1 else camera


# --- frame ---------------------------------------------------------------------


def drive_inputs(dt):
    """Keyboard -> commands, with main.cpp's speed-sensitive steer and slew."""
    global steer_cmd, throttle_cmd, brake_cmd, view_mode, cine_t, shot_no
    left = canvas.is_key_down("A") or canvas.is_key_down("LEFT")
    rightk = canvas.is_key_down("D") or canvas.is_key_down("RIGHT")
    steer_in = (1.0 if left else 0.0) - (1.0 if rightk else 0.0)
    scale = 1.0 / (1.0 + abs(vehicle.forward_speed) * 3.6 * 0.015)
    steer_cmd += (steer_in * scale - steer_cmd) * min(1.0, dt * 2.0)
    throttle_cmd = 1.0 if (canvas.is_key_down("W") or canvas.is_key_down("UP")) else 0.0
    brake_cmd = 1.0 if (canvas.is_key_down("S") or canvas.is_key_down("DOWN")
                        or canvas.is_key_down("SPACE")) else 0.0

    if pressed("R"):
        vehicle.gear = (tp.PhysxVehicle.Gear.REVERSE
                        if vehicle.gear == tp.PhysxVehicle.Gear.FORWARD
                        else tp.PhysxVehicle.Gear.FORWARD)
    if pressed("X"):
        global tc_on
        tc_on = not tc_on
    if pressed("V"):
        view_mode = 0 if view_mode == 1 else 1
    if pressed("C"):
        view_mode = 0 if view_mode == 2 else 2
        cine_t = 0.0
    if pressed("BACKSPACE"):
        vehicle.respawn(SPAWN_POS, SPAWN_ROT)
        vehicle.gear = tp.PhysxVehicle.Gear.FORWARD
        steer_cmd = 0.0
        w_ema[:] = REST_LOAD
    if pressed("T"):
        # Reset the DEMO's soft-ground state along with the grids: a reset that
        # leaves the dig factor, the TC cut and the load EMA where they were is
        # not a reset a stuck car can feel.
        global prev_hub, tc_cut
        for t in terrain.values():
            t.reset()
        dig[:] = 1.0
        tc_cut = 1.0
        w_ema[:] = REST_LOAD
        prev_hub = None
    if pressed("F"):
        shot_no += 1
        save_shot(os.path.join(OUT_DIR, f"warp_mudsnow_drive_{shot_no:03d}.png"))


def cine_pose(c, a):
    """The cinematic orbit at phase `a`: eye and target.

    Low -- 1.2 to 2.2 m, i.e. inside a wheel's world rather than above the car
    -- and a full circle, so the orbit CROSSES the key's azimuth twice a lap.
    That is the whole point of it: at the two crossings the ruts are backlit and
    every berm on the lane draws its own shadow toward the lens, which is the
    frame this demo exists to produce. Anywhere else on the circle it is a car.
    """
    return (c + np.array([math.cos(a) * 9.5, 0.30 + 0.85 * (0.5 - 0.5 * math.cos(2.0 * a)),
                          math.sin(a) * 9.5]),
            c + np.array([0.0, -0.22, 0.0]))


# The phase of that orbit where the camera is behind the car and BEYOND it is
# the key: the hero framing, and the one --script cine parks on.
CINE_HERO = 3.55


def update_camera(dt):
    """main.cpp's chase: exponential lerp toward a chassis-relative offset, read
    off the interpolated visual rather than the raw actor pose."""
    global cam_pos, cam_tgt, cine_t
    p, q = vehicle.position, vehicle.quaternion
    c = np.array([p.x, p.y, p.z])
    if view_mode == 2:
        cine_t += dt
        want, want_t = cine_pose(c, cine_t * 0.16)
    else:
        off = qrot(q, np.array([0.0, CHASE_DIST * math.sin(CHASE_PITCH),
                                -CHASE_DIST * math.cos(CHASE_PITCH)]))
        want = c + off
        want_t = c + qrot(q, np.array([0.0, 0.9, 3.0]))
    k = 1.0 - math.exp(-(2.0 if view_mode == 2 else 5.0) * dt)
    cam_pos += (want - cam_pos) * k
    cam_tgt += (want_t - cam_tgt) * k
    camera.position.set(*cam_pos)
    camera.look_at(tp.Vector3(*cam_tgt))


brake_was = reverse_was = False


def frame():
    global fps_ema, brake_was, reverse_was
    t0 = time.perf_counter()
    drive_inputs(DT)
    mark_dirty(coupled_step(throttle_cmd, steer_cmd, brake_cmd))

    p, q = vehicle.position, vehicle.quaternion
    chassis.position.set(p.x, p.y, p.z)
    chassis.quaternion.set(q.x, q.y, q.z, q.w)
    for i in range(4):
        lp, lq = vehicle.wheel_local_pose(i)
        wheel_rigs[i].position.set(lp.x, lp.y, lp.z)
        wheel_rigs[i].quaternion.set(lq.x, lq.y, lq.z, lq.w)

    # Lamps: only touch the material when the state flips -- needs_update()
    # re-uploads it.
    on = brake_cmd > 0.05
    if brake_mat is not None and on != brake_was:
        brake_was = on
        brake_mat.emissive_intensity = 6.0 if on else 1.0
        brake_mat.needs_update()
    rev = vehicle.gear == tp.PhysxVehicle.Gear.REVERSE
    if reverse_mat is not None and rev != reverse_was:
        reverse_was = rev
        reverse_mat.emissive_intensity = 6.0 if rev else 1.0
        reverse_mat.needs_update()

    update_camera(DT)
    c = active_camera()
    weather(DT, (c.position.x, c.position.y, c.position.z))
    for s in strips:
        s.publish()
    renderer.render(scene, active_camera())
    if ui is not None:
        ui.render(draw_ui)
    fps_ema += (1.0 / max(time.perf_counter() - t0, 1e-4) - fps_ema) * 0.05


# The first render is what creates the records the vertex exports come from.
renderer.render(scene, camera)
if not GL:
    armed = sum(s.arm(renderer) for s in strips)
    print(f"lane mesh route: "
          f"{'zero-copy CUDA -> Vulkan' if armed == len(strips) else 'mixed / host copy'}"
          f" ({armed}/{len(strips)} armed)")
for s in strips:
    s.publish()

def pose_visuals():
    p, q = vehicle.position, vehicle.quaternion
    chassis.position.set(p.x, p.y, p.z)
    chassis.quaternion.set(q.x, q.y, q.z, q.w)
    for k in range(4):
        lp, lq = vehicle.wheel_local_pose(k)
        wheel_rigs[k].position.set(lp.x, lp.y, lp.z)
        wheel_rigs[k].quaternion.set(lq.x, lq.y, lq.z, lq.w)


def scripted(spawn_z, beats, eye, look, path, note=""):
    """Drive a fixed input script, park a camera, save the frame.

    The shots the acceptance asks for have to be repeatable and identical
    between lanes -- the same line at the same speed is the whole comparison --
    which a hand-driven pass is not. Each beat is (seconds, throttle, steer).
    `eye` None means the cinematic orbit at CINE_HERO, wherever the car ended.
    """
    vehicle.respawn(tp.Vector3(-20.0, 1.05, spawn_z), SPAWN_ROT)
    vehicle.gear = tp.PhysxVehicle.Gear.FORWARD
    w_ema[:] = REST_LOAD
    for _ in range(60):
        coupled_step(0.0, 0.0, 0.0)
    for secs, thr, st in beats:
        for _ in range(int(round(secs * 60.0))):
            mark_dirty(coupled_step(thr, st, 0.0))
            weather(DT, eye if eye is not None else (0.0, 2.0, 0.0))
    pose_visuals()
    for s in strips:
        s.dirty = True
        s.publish()
    if eye is None:
        p = vehicle.position
        eye, look = cine_pose(np.array([p.x, p.y, p.z]), CINE_HERO)
    # dt = 0 freezes the snowfall for the still, and this is also where the
    # follow box finally lands on the shot's own camera: the WARM loop below
    # renders the same instant sixty times to settle the temporal passes, and a
    # field reporting motion through frames that do not move is one TAA smears.
    weather(0.0, eye)
    camera.position.set(*eye)
    camera.look_at(tp.Vector3(*look))
    # The Vulkan pipeline is temporal (probe GI, denoisers, the upscaler): one
    # render after the sim would capture frame ONE of all of them.
    for _ in range(0 if GL else WARM):
        renderer.render(scene, camera)
    save_shot(path)
    print(f"  {os.path.basename(path)}: v={vehicle.forward_speed * 3.6:5.1f} km/h  "
          f"lane={hud['lane'][0]}  sink={np.round(hud['z'] * 1000, 1)} mm  "
          f"mu={np.round(hud['mu'], 2)}  slip={np.round(hud['slip'], 2)}  "
          f"util={np.round(hud['util'], 2)}  "
          f"Bekker/PhysX={np.round(100 * (hud['bek'] / np.maximum(hud['w'], 1.0) - 1.0), 1)} % "
          f"{note}")


if BENCH:
    # Honest wall clock: the whole frame the interactive loop runs, vsync off.
    for _ in range(60):
        mark_dirty(coupled_step(0.5, 0.15, 0.0))
        pose_visuals()
        weather(DT)
        for s in strips:
            s.publish()
        renderer.render(scene, camera)
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(240):
        mark_dirty(coupled_step(0.5, 0.15, 0.0))
        pose_visuals()
        weather(DT)
        for s in strips:
            s.publish()
        renderer.render(scene, camera)
    wp.synchronize_device(device)
    ms = (time.perf_counter() - t0) * 1000.0 / 240.0
    t1 = time.perf_counter()
    for _ in range(240):
        coupled_step(0.5, 0.15, 0.0)
    wp.synchronize_device(device)
    sim_ms = (time.perf_counter() - t1) * 1000.0 / 240.0
    print(f"bench [{'opengl' if GL else 'vulkan'}, "
          f"{'zero-copy interop' if INTEROP else 'host copy, dirty strips'}]: "
          f"{ms:.2f} ms/frame ({1000.0 / ms:.0f} fps), of which sim+coupling "
          f"{sim_ms:.2f} ms; {cells:,} cells, {len(strips)} strips")
elif SHOT:
    which = cli_arg("--script", "lanes", str)
    z_mud = 0.5 * (LANE_Z["mud"][0] + LANE_Z["mud"][1])
    z_snow = 0.5 * (LANE_Z["snow"][0] + LANE_Z["snow"][1])
    out = lambda n: os.path.join(OUT_DIR, f"warp_mudsnow_drive_{n}.png")   # noqa: E731

    # One script per process: VulkanRenderer.save_frame is a render + readback
    # and taking two of them in one run loses the device on the second.
    if which in ("mud", "snow"):
        # Same line, same throttle, both soft lanes: mud walls its trench,
        # snow swallows the wheel and barely rims.
        z = z_mud if which == "mud" else z_snow
        scripted(z, [(3.2, 0.45, 0.0)],
                 (-13.5, 1.15, z + (5.2 if z < 0 else -5.2)), (-15.5, -0.1, z),
                 out(f"1_{which}_rut"))
    if which in ("spin_mud", "spin_clay"):
        # Acceptance 2: the same full-throttle launch, mud vs the clay strip.
        # TC off -- unassisted wheelspin is what this shot is about.
        tc_on = False
        z = z_mud if which == "spin_mud" else 0.0
        scripted(z, [(2.0, 1.0, 0.0)],
                 (-17.0, 1.5, z + (5.0 if z <= 0 else -5.0)), (-18.5, 0.2, z),
                 out(f"2_{which}"))
    if which == "cine":
        # The look pass's own acceptance frame: the cinematic camera, parked at
        # the phase of its orbit that crosses the key, on a mud lane with a
        # trail of its own ruts leading back out of frame. Nothing here is a
        # special camera -- cine_pose is the same function C flies.
        scripted(z_mud, [(5.5, 0.45, 0.0)], None, None, out("4_cine"))
    if which == "pan":
        # Acceptance 3: no flat grey anywhere in a 360 pan from the driver's
        # seat. One heading per process (a second save_frame in one run loses
        # the device), so this is run four times with --yaw.
        yaw = math.radians(cli_arg("--yaw", 0.0, float))
        eye = np.array([-6.0, 1.35, 0.0])
        scripted(0.0, [(0.2, 0.0, 0.0)], eye,
                 eye + np.array([math.sin(yaw) * 20.0, -1.2, math.cos(yaw) * 20.0]),
                 out(f"5_pan_{int(round(math.degrees(yaw))):03d}"))
    if which == "diag":
        # Tuning telemetry: a 0.45-throttle launch on each soft lane, per-wheel
        # state every half second. No frame saved.
        for lane_name in ("mud", "snow"):
            z_lane = 0.5 * (LANE_Z[lane_name][0] + LANE_Z[lane_name][1])
            vehicle.respawn(tp.Vector3(-20.0, 1.05, z_lane), SPAWN_ROT)
            dig[:] = 1.0
            tc_cut = 1.0
            w_ema[:] = REST_LOAD
            for _ in range(60):
                coupled_step(0.0, 0.0, 0.0)
            print(f"  {lane_name}: t  v_kmh | W_kN | omega*r m/s | dig | sink_mm | tc")
            for k in range(int(4.0 * 60)):
                coupled_step(0.45, 0.0, 0.0)
                if k % 30 == 0:
                    w = [vehicle.suspension_force(i) / 1000 for i in range(4)]
                    wr = [vehicle.wheel_angular_speed(i) * R_WHEEL for i in range(4)]
                    print(f"  {k / 60.0:4.1f} {vehicle.forward_speed * 3.6:6.1f} | "
                          f"W {np.round(w, 2)} | wr {np.round(wr, 1)} | "
                          f"dig {np.round(dig, 2)} | z {np.round(hud['z'] * 1000, 0)} | "
                          f"{tc_cut:4.2f}")
    if which == "stuck":
        # The mud skill loop, measured, with TC off (TC exists to prevent
        # exactly this): floor it from rest (the wheels dig and the car goes
        # nowhere), then ease off to a crawl throttle (the dig relaxes and it
        # creeps out). No frame is saved -- this one is numbers.
        tc_on = False
        vehicle.respawn(tp.Vector3(-20.0, 1.05, z_mud), SPAWN_ROT)
        w_ema[:] = REST_LOAD
        for _ in range(60):
            coupled_step(0.0, 0.0, 0.0)
        x0 = vehicle.position.x
        for _ in range(int(4.0 * 60)):
            coupled_step(1.0, 0.0, 0.0)
        x1 = vehicle.position.x
        d1, dig1, z1 = x1 - x0, float(np.mean(hud["dig"])), float(np.mean(hud["z"]) * 1000)
        for _ in range(int(6.0 * 60)):
            coupled_step(0.35, 0.0, 0.0)
        x2 = vehicle.position.x
        d2, dig2 = x2 - x1, float(np.mean(hud["dig"]))
        print(f"  stuck test (mud): 4 s full throttle -> {d1:5.2f} m travelled, "
              f"dig {dig1:.2f}, sink {z1:.0f} mm")
        print(f"                    6 s at 0.35       -> {d2:5.2f} m travelled, "
              f"dig {dig2:.2f}, v {vehicle.forward_speed * 3.6:4.1f} km/h")
        print(f"  drag/wheel now {np.round(hud['drag'], 0)} N, "
              f"traction ceiling {np.round(hud['mu'] * hud['w'], 0)} N")
    if which == "lap":
        # Acceptance 3: a circle in mud, twice. Full lock at a walking pace
        # turns inside the lane; the readout is the depth of the wheel below
        # the LOCAL surface, which is a fresh sinkage on lap 1 and almost
        # nothing on lap 2 because the grade under the rut has not moved.
        def rut_depth():
            p = vehicle.position
            lp, _ = vehicle.wheel_local_pose(3)
            hubw = np.array([p.x, p.y, p.z]) + qrot(vehicle.quaternion,
                                                    np.array([lp.x, lp.y, lp.z]))
            t = terrain["mud"]
            x = torch.tensor([[hubw[0]]], device=device, dtype=torch.float32)
            y = torch.tensor([[-hubw[2]]], device=device, dtype=torch.float32)
            surf = float(t.heights(x, y)[0, 0])
            return (surf - (hubw[1] - R_WHEEL)) * 1000.0

        vehicle.respawn(tp.Vector3(-14.0, 1.05, z_mud), SPAWN_ROT)
        w_ema[:] = REST_LOAD
        for _ in range(60):
            coupled_step(0.0, 0.0, 0.0)
        lap = []
        for k in range(int(24.0 * 60)):
            mark_dirty(coupled_step(0.30, 1.0, 0.0))
            if k % 30 == 0:
                lap.append((k / 60.0, rut_depth(), float(hud["z"][3] * 1000.0)))
        pose_visuals()
        for s in strips:
            s.dirty = True
            s.publish()
        camera.position.set(-14.0, 6.0, z_mud - 6.5)
        camera.look_at(tp.Vector3(-15.5, -0.1, z_mud))
        for _ in range(0 if GL else WARM):
            renderer.render(scene, camera)
        save_shot(out("3_second_lap"))
        print("  circle in mud: t, wheel below LOCAL surface (mm), Bekker z_eq (mm)")
        for t_s, d, z in lap[::4]:
            print(f"    {t_s:5.1f}s  {d:6.1f}  {z:6.1f}")
else:
    canvas.animate(frame)
