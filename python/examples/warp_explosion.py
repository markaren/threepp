"""Blast Yard: a brick test range demolished by a charge -- PhysX + Warp + Vulkan.

A light-grey void with a grid floor, the way a test range looks in Unreal or
Omniverse, and a yard full of things that are *pre-broken by construction* --
a three-walled brick house, a shed, pillars of stacked segments, pallets of
crates, loose rubble. Roughly twelve hundred individual PhysX boxes, laid in
staggered courses with millimetre gaps so nothing starts interpenetrating,
settled for two seconds, and then hit.

The blast is not a fracture solve; it is an impulse field. At t0 every body
gets

    J(r) = J0 * exp(-r / L) / max(r, r0) ** 1.5

along (body - charge), with an upward bias near the floor (a real blast
reflects off the ground and lifts what is standing on it), a speed cap so the
bodies nearest the charge do not leave at kilometres per second, and a random
tumble. A decaying radial "wind" force follows for the next 0.45 s -- that is
what keeps the wall panels moving after the first frame instead of dropping
straight down, and it is the difference between "a wall fell over" and "a wall
was blown out". A point light flashes at the charge and the camera shakes.

The gas is four `tp.ParticleField`s, all created before the first frame and
parked at live count 0 -- creating one mid-run is a device idle and a cleared
TAA history, on the one frame where that shows.

  fire   HostRing, Warp-advected: a radial burst with heavy drag, curl stir and
         buoyancy. Rendered as a DensityRepr volume whose blackbody emission
         ramp (2700 K at the box floor, 1250 K at its top) IS the fireball --
         emission is sigma * L_e and sigma is already being marched, so the
         flame's shape comes from the particles and its colour from the ramp.
         Additive billboards with their own bloom pyramid carry the flash.
  smoke  HostRing, DensityRepr only. Early smoke is thrown outward and rolls
         over into a mushroom head; late smoke goes straight up as the stem.
         Real extinction, so it shadows itself and dims the yard behind it.
  dust   HostRing. Slots are born on the ground annulus at the shock radius
         r = c*t, so the ring races outward ahead of the debris for free.
  spark  HostRing. A TIGHT burst window (0.45 s) then long-lived embers that
         arc, land and burn out. Not the analytic emitter, which cannot express
         a one-shot burst at all -- see the note at the field.

Everything is driven off one sim clock and one blast timeline, so the later
phases (drums, slow-motion film) hang off the same t0 without re-deriving it.

    python warp_explosion.py                 # window; drag to orbit, Esc quits
    python warp_explosion.py --shot 4        # headless: sim 4 s, write png
    python warp_explosion.py --video 6       # headless: 6 s of frames (+mp4 if ffmpeg)
    python warp_explosion.py --bench         # timed phase breakdown, post-detonation
    python warp_explosion.py --yield 2400    # bigger charge (J0, kg m/s at 1 m)
    python warp_explosion.py --charge 0,0.6,-1.2 --t0 1.5
    python warp_explosion.py --courses 20 --shot 4      # taller walls, more bodies
    python warp_explosion.py --shot 4 --spin 0.25       # orbit while simulating
    python warp_explosion.py --gas 2.0                  # twice the gas particles
    python warp_explosion.py --sigma 1.6                # thicker smoke
    python warp_explosion.py --no-gas                   # phase 1 only
    python warp_explosion.py --no-ao                    # RT AO/GI off (~6.5 ms/frame back)
    python warp_explosion.py --msaa 1                   # the other render knob

Vulkan only: the fireball renders with tp.ParticleField, which draws nothing on
the GL path by design. The smoke rides the froxel FOG pass, so --no-fog is a
phase-1 debug flag -- it turns the gas volumes off with it.
"""
import math
import os
import subprocess
import sys
import tempfile
import time

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import (bench_loop, cli_arg, find_ffmpeg, parse_size,
                         resize_handler, standard_material)

SHOT = "--shot" in sys.argv
SHOT_T = cli_arg("--shot", 4.0, float)
VIDEO = cli_arg("--video", 0.0, float)     # seconds of frames to render
BENCH = "--bench" in sys.argv
HEADLESS = SHOT or VIDEO > 0 or BENCH
SEED = cli_arg("--seed", 7, int)

if HEADLESS:
    # Presenting to the hidden window is pure cost offline -- and in --bench it
    # is worse than cost: nothing pumps the window's message loop, so present
    # stalls and the numbers come out roughly a third of the real window rate.
    os.environ.setdefault("THREEPP_VULKAN_SUPPRESS_PRESENT", "1")

FPS = 60.0
DT = 1.0 / FPS

# --- the charge -------------------------------------------------------------

T0 = cli_arg("--t0", 2.0, float)           # detonation time; before it, the stack settles
YIELD = cli_arg("--yield", 1400.0, float)   # J0: impulse (kg m/s) at 1 m on a 1 kg body
CHARGE = tuple(float(v) for v in cli_arg("--charge", "0,0.6,-0.6", str).split(","))
BLAST_L = 5.0             # exponential range of the impulse field, m
BLAST_R0 = 0.8            # near-field softening radius, m
BLAST_P = 1.5             # geometric falloff exponent
BLAST_VMAX = 22.0         # speed cap on the kick, m/s -- the anti-NaN valve
BLAST_LIFT = 0.55         # ground-reflection upward bias, added to the unit direction
BLAST_SPIN = 0.55         # random tumble, rad/s per m/s of kick
WIND_A0 = 130.0            # radial "wind" acceleration at 1 m, m/s^2
WIND_TAU = 0.18           # its decay time, s
WIND_T = 0.45             # how long the wind blows, s
FLASH_I = 2600.0          # point-light flash peak intensity
FLASH_TAU = 0.075         # flash decay, s
SHAKE_A = 0.11            # camera shake amplitude at the default yield, m
SHAKE_TAU = 0.30
SHAKE_T = 1.4
FRONT_C = 28.0            # visual shock-front speed, m/s: the dust ring rides it
FLASH_BB = 7.0            # billboard brightness spike at t0, on top of the base gain

# --- the gas ----------------------------------------------------------------
#
# Counts are the bandwidth: one HostRing field costs 16 B/particle/frame on the
# readback AND again on the submit memcpy, and that is the only per-frame cost
# that scales with the particle count anywhere in this demo. --gas scales all
# three cohorts together; --sigma scales the optical mass the other way, so
# `--gas 0.5 --sigma 2` is very nearly the same picture for half the bus.
NO_GAS = "--no-gas" in sys.argv
GAS = cli_arg("--gas", 1.0, float)
SIGMA = cli_arg("--sigma", 1.0, float)
FIRE_N = int(120_000 * GAS)
SMOKE_N = int(300_000 * GAS)
DUST_N = int(180_000 * GAS)
SPARK_N = int(44_000 * GAS)

# Emission windows, in seconds since t0 (§2.6 of the plan). A slot's birth time
# is drawn ONCE at startup and sorted, so the sim is a pure function of tw with
# no free list, no compaction and no spawn kernel -- and the sorted births mean
# the submit only has to carry the prefix that has been born yet.
FIRE_EMIT, FIRE_LIFE = 0.26, (0.24, 0.70)
SMOKE_EMIT, SMOKE_LIFE = 3.40, (3.2, 7.5)
DUST_EMIT, DUST_LIFE = 1.10, (0.8, 1.9)
SMOKE_SKEW = 1.0          # births = EMIT * u**SKEW: front-loaded, so the head leads
SPARK_EMIT, SPARK_LIFE = 0.40, (0.9, 3.4)   # a TIGHT burst, then long-lived embers
SPARK_V0, SPARK_DRAG, SPARK_CURL = 18.0, 0.90, 0.9
SPARK_R = (0.075, 0.0)    # holds its size, then goes out (r_pow below)

FIRE_V0, FIRE_DRAG, FIRE_BUOY, FIRE_CURL = 21.0, 7.5, 26.0, 7.0
SMOKE_V0, SMOKE_DRAG, SMOKE_BUOY, SMOKE_CURL = 6.0, 1.5, 7.5, 3.4
DUST_V0, DUST_DRAG, DUST_GRAV, DUST_CURL = 7.5, 2.2, 9.81, 0.7
FIRE_R = (0.10, 0.55)     # world radius over life, m (w under WSemantic.Radius)
SMOKE_R = (0.30, 1.45)
DUST_R = (0.14, 0.70)

# sigma_t one particle contributes, before --sigma. These are TUNED BY EYE and
# are an order of magnitude above the pyi's weather guidance for a reason: a
# rain curtain fills a 50 m box and wants a whisper of extinction each, while a
# soot column is 300k particles inside a few hundred cubic metres and has to go
# genuinely opaque. Total optical mass is N * sigma, so halving a count and
# doubling its sigma here is very nearly the same picture.
FIRE_SIGMA, SMOKE_SIGMA, DUST_SIGMA = 0.40, 0.55, 0.24

# --- the yard ---------------------------------------------------------------

BRICK = (0.40, 0.20, 0.20)     # length (local x) x height (y) x depth (z)
GAP = 0.002                    # laid with a 2 mm joint: no initial interpenetration
COURSES = cli_arg("--courses", 16, int)
SHED_COURSES = cli_arg("--shed-courses", 11, int)
GROUND = 400.0                 # floor slab side, m: the edge sits out in the fog
JITTER = 0.0015                # per-brick placement noise, m

RHO_BRICK, RHO_CRATE, RHO_BLOCK = 900.0, 160.0, 1500.0

rng = np.random.default_rng(SEED)

# Each item is (position, size, yaw). One list per material batch: instance
# colours are a GL-only feature, so brick tone variation is three batches of
# bricks rather than three colours in one.
batches = {"brick0": [], "brick1": [], "brick2": [], "crate": [], "block": []}
_brick_tone = 0


def add_item(kind, pos, size, yaw=0.0):
    batches[kind].append((pos, size, yaw))


def add_brick(pos, yaw):
    """A brick, in whichever of the three clay tones comes next."""
    global _brick_tone
    add_item(f"brick{_brick_tone % 3}", pos, BRICK, yaw)
    _brick_tone += 7          # coprime with 3: adjacent bricks never share a tone run


def brick_wall(origin, direction, s0, s1, courses, y0=0.0):
    """A staggered brick wall from origin + direction*s0 to origin + direction*s1.

    `direction` is a unit (dx, dz); the brick's long axis runs along it. Odd
    courses are shifted half a brick, so the joints break like real bonding and
    the stack has something to interlock on when it is hit.
    """
    dx, dz = direction
    yaw = math.atan2(-dz, dx)                  # local +X -> (dx, 0, dz)
    pitch = BRICK[0] + GAP
    span = s1 - s0
    for c in range(courses):
        y = y0 + (c + 0.5) * (BRICK[1] + GAP) + 0.001
        shift = 0.5 * pitch if c % 2 else 0.0
        s = shift + 0.5 * pitch
        while s + 0.5 * pitch <= span + 1e-6:
            jx, jz = rng.normal(0.0, JITTER, 2)
            add_brick((origin[0] + dx * (s0 + s) + jx, y,
                       origin[2] + dz * (s0 + s) + jz),
                      yaw + float(rng.normal(0.0, 0.004)))   # under the joint width
            s += pitch


def brick_house(centre, lx, lz, courses):
    """Three walls in a U, open toward +Z: back wall plus two returns.

    The returns start one brick-depth past the back wall so the corners butt
    instead of overlapping.
    """
    cx, cz = centre
    inset = BRICK[2] + GAP
    brick_wall((cx - 0.5 * lx, 0.0, cz - 0.5 * lz), (1.0, 0.0), 0.0, lx, courses)
    for sx in (-1.0, 1.0):
        brick_wall((cx + sx * 0.5 * lx, 0.0, cz - 0.5 * lz), (0.0, 1.0),
                   inset, lz, courses)


def pillar(x, z, segments, seg=(0.46, 0.25, 0.46)):
    """A stacked column of segments, each turned a few degrees off its neighbour."""
    for k in range(segments):
        y = (k + 0.5) * (seg[1] + GAP) + 0.001
        jx, jz = rng.normal(0.0, JITTER, 2)
        add_item("block", (x + jx, y, z + jz), seg, float(rng.normal(0.0, 0.02)))


def pallet(x, z, nx, nz, levels, crate=0.5):
    """A block of crates on the ground.

    The yaw is deliberately tiny: a 0.5 m box turned 3 degrees pokes 12 mm past
    its own footprint, which is more than the joint, and a pallet that starts
    interpenetrating shakes itself apart during the settle.
    """
    yaw_max = 0.01
    pitch = crate * (1.0 + yaw_max) + 0.012
    for ly in range(levels):
        for ix in range(nx):
            for iz in range(nz):
                jx, jz = rng.normal(0.0, JITTER, 2)
                add_item("crate",
                         (x + (ix - 0.5 * (nx - 1)) * pitch + jx,
                          (ly + 0.5) * (crate + GAP) + 0.001,
                          z + (iz - 0.5 * (nz - 1)) * pitch + jz),
                         (crate, crate, crate), float(rng.uniform(-yaw_max, yaw_max)))


def rubble(n, r_min, r_max):
    """Loose bricks lying flat around the yard."""
    for _ in range(n):
        a = float(rng.uniform(0.0, 2.0 * math.pi))
        r = float(rng.uniform(r_min, r_max))
        add_brick((r * math.sin(a), 0.5 * BRICK[1] + 0.001, r * math.cos(a)),
                  float(rng.uniform(0.0, 2.0 * math.pi)))


brick_house((0.0, 0.0), 8.0, 5.5, COURSES)            # the main structure
brick_house((-7.6, 1.0), 3.6, 3.0, SHED_COURSES)      # a shed off to one side
brick_wall((6.6, 0.0, -3.0), (0.0, 1.0), 0.0, 6.0, COURSES - 4)   # free-standing wall
for px, pz in ((9.6, -2.2), (9.6, 1.8)):
    pillar(px, pz, 10)
pallet(-3.4, 4.6, 3, 3, 3)
pallet(4.2, 5.2, 3, 2, 4)
pallet(8.2, 4.4, 3, 3, 3)
pallet(-6.8, -2.6, 3, 2, 3)     # keep pallets wider than they are tall, or they settle over
rubble(80, 3.5, 11.0)

N_BODIES = sum(len(v) for v in batches.values())

# --- renderer ---------------------------------------------------------------

if not tp.vulkan_available():
    sys.exit("warp_explosion needs the Vulkan renderer (ParticleField is Vulkan-only).")

W, H = parse_size(cli_arg("--size", "1280x720", str))
MSAA = cli_arg("--msaa", 4, int)              # G-buffer MSAA (measured free here at 720p)
canvas = tp.Canvas("threepp x physx - blast yard", width=W, height=H,
                   antialiasing=MSAA, vsync=False, headless=HEADLESS)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 0.55    # pinned: phase 2's billboards bypass auto-exposure
renderer.shadow_map_enabled = True
# The two knobs that actually move the frame time here (measured, --bench):
# ray-traced AO/GI ~6.5 ms and the fog march ~2.5 ms at 1280x720. The body
# count is not the lever -- PhysX is under 2 ms of a 22 ms frame.
NO_AO = "--no-ao" in sys.argv
NO_FOG = "--no-fog" in sys.argv
if NO_AO:
    renderer.deferred_ao = False

VOID = tp.Color(0.80, 0.81, 0.83)
scene = tp.Scene()
scene.background = VOID
if not NO_FOG:
    scene.set_fog(VOID, 55.0, 200.0)          # dissolves the slab's far edge into the void

camera = tp.PerspectiveCamera(44, canvas.aspect(), 0.05, 800)
CAM_R, CAM_Y, CAM_TARGET = 30.0, 11.0, (0.0, 6.0, 0.0)
SPIN = cli_arg("--spin", 0.0, float)          # shot/video: orbit the camera, deg/frame
START_ANGLE = cli_arg("--angle", 22.0, float)  # orbit start angle, degrees
_cam_base = tp.Vector3()


def orbit(frame):
    """Place the camera on its orbit for `frame` (static at SPIN == 0)."""
    a = math.radians(START_ANGLE + SPIN * frame)
    _cam_base.set(CAM_R * math.sin(a), CAM_Y, CAM_R * math.cos(a))
    camera.position.set(_cam_base.x, _cam_base.y, _cam_base.z)
    camera.look_at(*CAM_TARGET)


# A test range is an overcast-bright void with one hard sun in it. The fill is
# deliberately weak: everything here is grey or clay, and a strong hemisphere
# fill flattens both into the same pale pink.
scene.add(tp.HemisphereLight(0xd7e3f4, 0x8d8a84, 0.45))
sun = tp.DirectionalLight(0xfff2df, 3.0)
sun.position.set(9.0, 15.0, 6.0)
sun.cast_shadow = True
scene.add(sun)

flash = tp.PointLight(0xffd6a0, 0.0, 45.0, 2.0)
flash.position.set(*CHARGE)
scene.add(flash)


def grid_texture(px=1024, tile=10.0, minor=1.0):
    """The test-range floor: a light grey with 1 m minor and 10 m major rules.

    Written to a temp PNG because TextureLoader takes a path; `tile` is the
    metres the image covers, which is what sets the repeat below.
    """
    from PIL import Image

    def rules(period, width):
        i = np.arange(px)
        return ((i + width // 2) % period) < width

    img = np.full((px, px, 3), 163, np.uint8)
    fine = rules(max(int(round(px * minor / tile)), 2), 2)
    coarse = rules(px, 6)
    for mask, tone in ((fine, 138), (coarse, 100)):
        img[mask, :, :] = tone
        img[:, mask, :] = tone
    path = os.path.join(tempfile.gettempdir(), f"threepp_blast_grid_{px}_{int(tile)}.png")
    Image.fromarray(img).save(path)
    return path, tile


grid_path, grid_tile = grid_texture()
grid = tp.TextureLoader().load(grid_path, tp.ColorSpace.SRGB)
grid.wrap_s = tp.TextureWrapping.Repeat
grid.wrap_t = tp.TextureWrapping.Repeat
grid.repeat.set(GROUND / grid_tile, GROUND / grid_tile)
grid.needs_update()

floor_mat = standard_material(tp.Color(1.0, 1.0, 1.0), 0.92)
floor_mat.map = grid
# A slab, not a plane: PhysX infers a collider from Box geometry, so the thing
# you see and the thing the bricks land on are one object.
floor = tp.Mesh(tp.BoxGeometry(GROUND, 0.4, GROUND), floor_mat)
floor.position.y = -0.2
floor.receive_shadow = True
scene.add(floor)

# --- physics ----------------------------------------------------------------

world = tp.PhysxWorld(tp.Vector3(0.0, -9.81, 0.0), fixed_timestep=DT, max_substeps=2,
                      num_threads=8, tgs_pcm=True)
ground_mat = world.create_material(0.62, 0.58, 0.06)
world.add_static(floor, ground_mat)

MATERIALS = {
    "brick0": (standard_material(tp.Color(0.42, 0.17, 0.12), 0.92), RHO_BRICK),
    "brick1": (standard_material(tp.Color(0.52, 0.24, 0.16), 0.92), RHO_BRICK),
    "brick2": (standard_material(tp.Color(0.35, 0.15, 0.12), 0.94), RHO_BRICK),
    "crate": (standard_material(tp.Color(0.46, 0.32, 0.17), 0.88), RHO_CRATE),
    "block": (standard_material(tp.Color(0.52, 0.52, 0.50), 0.95), RHO_BLOCK),
}

UP = tp.Vector3(0.0, 1.0, 0.0)
meshes, bodies = [], []
for kind, items in batches.items():
    if not items:
        continue
    material, density = MATERIALS[kind]
    im = tp.InstancedMesh(tp.BoxGeometry(1.0, 1.0, 1.0), material, len(items))
    m, q = tp.Matrix4(), tp.Quaternion()
    for i, (p, s, yaw) in enumerate(items):
        q.set_from_axis_angle(UP, yaw)
        m.compose(tp.Vector3(*p), q, tp.Vector3(*s))
        im.set_matrix_at(i, m)
    im.instance_matrix_needs_update()
    im.cast_shadow = True
    im.receive_shadow = True
    im.frustum_culled = False        # the bounds go stale the moment it detonates
    scene.add(im)
    meshes.append(im)
    bodies += world.add_instanced(im, density)

MASS = np.array([b.mass for b in bodies], dtype=np.float64)
print(f"blast yard: {N_BODIES} bodies in {len(meshes)} instanced batches "
      f"({MASS.sum():.0f} kg), charge at {CHARGE}, t0 = {T0:.2f} s")

# --- the blast timeline -----------------------------------------------------
#
# One clock, one timeline. Phases 2-4 (gas emission windows, shell coupling,
# the slow-motion ramp) read `blast_timeline(sim_time)` rather than testing
# sim_time against their own constants.

_shake_bank = [[(float(f), float(p)) for f, p in
                zip(rng.uniform(7.0, 23.0, 3), rng.uniform(0.0, 6.283, 3))]
               for _ in range(3)]


def blast_timeline(t):
    """The state of the detonation at sim time `t`, as a dict.

    tw       seconds since t0 (negative before it)
    armed    True once the charge has gone off
    flash    point-light intensity, 0 outside the pulse
    wind     radial acceleration scale (m/s^2 at 1 m), 0 outside the tail
    front    shock-front radius, m -- the dust ring is spawned on it
    shake    (x, y, z) camera offset, m
    bb_gain  additive-billboard brightness multiplier: the t0 spike, decaying
             over ~0.3 s. Billboards composite after the upscaler and are NOT
             seen by auto-exposure, so this is the whole exposure story for them
    fire_e   fireball blackbody emission scale, 0 once the flame is out
    """
    tw = t - T0
    if tw < 0.0:
        return dict(tw=tw, armed=False, flash=0.0, wind=0.0, front=0.0,
                    shake=(0.0, 0.0, 0.0), bb_gain=0.0, fire_e=0.0)
    scale = math.sqrt(YIELD / 1400.0)
    flash_i = FLASH_I * scale * math.exp(-tw / FLASH_TAU) if tw < 0.5 else 0.0
    wind = WIND_A0 * math.exp(-tw / WIND_TAU) if tw < WIND_T else 0.0
    shake = (0.0, 0.0, 0.0)
    if tw < SHAKE_T:
        a = SHAKE_A * scale * math.exp(-tw / SHAKE_TAU)
        shake = tuple(a * sum(math.sin(2.0 * math.pi * f * tw + p) for f, p in bank) / 3.0
                      for bank in _shake_bank)
    return dict(tw=tw, armed=True, flash=flash_i, wind=wind,
                front=FRONT_C * tw, shake=shake,
                bb_gain=1.0 + FLASH_BB * math.exp(-tw / 0.13),
                fire_e=math.exp(-max(tw - 0.16, 0.0) / 0.30))


def detonate():
    """The impulse field, applied once, to every body."""
    cx, cy, cz = CHARGE
    kick = tp.Vector3()
    spin = tp.Vector3()
    for b, m in zip(bodies, MASS):
        p = b.position
        dx, dy, dz = p.x - cx, p.y - cy, p.z - cz
        r = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        dx, dy, dz = dx / r, dy / r, dz / r
        # A blast reflects off the ground, so what is standing on it gets lifted.
        dy += BLAST_LIFT * math.exp(-p.y / 1.5)
        n = math.sqrt(dx * dx + dy * dy + dz * dz)
        dx, dy, dz = dx / n, dy / n, dz / n
        j = YIELD * math.exp(-r / BLAST_L) / max(r, BLAST_R0) ** BLAST_P
        dv = min(j / m, BLAST_VMAX)     # the cap is what keeps the near field finite
        b.wake_up()
        kick.set(dx * dv * m, dy * dv * m, dz * dv * m)
        b.add_impulse(kick)
        w = BLAST_SPIN * dv
        spin.set(*(w * v for v in rng.uniform(-1.0, 1.0, 3)))
        b.set_angular_velocity(spin)


def blast_wind(accel):
    """The drag tail: a decaying radial acceleration field for WIND_T seconds.

    Applied as force = a * m so it is a genuine acceleration -- light crates and
    heavy pillar segments are carried by the same wind at the same rate.
    """
    cx, cy, cz = CHARGE
    f = tp.Vector3()
    for b, m in zip(bodies, MASS):
        p = b.position
        dx, dy, dz = p.x - cx, p.y - cy, p.z - cz
        r = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        k = accel * m / max(r, BLAST_R0) ** BLAST_P / r
        f.set(dx * k, dy * k, dz * k)
        b.add_force(f)


# --- the gas: four ParticleFields, all created here, before the first frame --
#
# The churn contract (ParticleField.hpp): a field is created ONCE at its final
# capacity and never resized, and creating one is a STRUCTURAL scene change --
# entry re-expansion, a vkDeviceWaitIdle and a cleared TAA history. So all four
# exist from startup and sit at live count 0 until the charge goes off. Nothing
# is added to or removed from the scene after this point.

# The curl field is baked to an NG^3 grid once per frame and fetched
# trilinearly, exactly as warp_nebula does it: 110k grid threads instead of
# 560k 4D noise evaluations, and the grid is what makes the smoke roll rather
# than every particle wobbling on its own phase.
NG = 48
NR = 30.0                              # half-size of the cubic noise box, m
NCX, NCY, NCZ = 0.0, 14.0, 0.0         # its world centre: over the yard, up the plume
NG_SCALE = (NG - 1) / (2.0 * NR)
NOISE_FREQ = 0.11                      # ~9 m eddies
NOISE_RATE = 0.35                      # how fast the field itself evolves, Hz-ish


@wp.func
def gas_noise(f: wp.array3d(dtype=wp.vec3), p: wp.vec3) -> wp.vec3:
    """Trilinear fetch from the baked curl grid; outside it, clamp to the edge."""
    gx = wp.clamp((p[0] - NCX + NR) * NG_SCALE, 0.0, float(NG - 1) - 1.0e-3)
    gy = wp.clamp((p[1] - NCY + NR) * NG_SCALE, 0.0, float(NG - 1) - 1.0e-3)
    gz = wp.clamp((p[2] - NCZ + NR) * NG_SCALE, 0.0, float(NG - 1) - 1.0e-3)
    i, j, k = int(wp.floor(gx)), int(wp.floor(gy)), int(wp.floor(gz))
    fx, fy, fz = gx - float(i), gy - float(j), gz - float(k)
    c00 = f[i, j, k] * (1.0 - fx) + f[i + 1, j, k] * fx
    c10 = f[i, j + 1, k] * (1.0 - fx) + f[i + 1, j + 1, k] * fx
    c01 = f[i, j, k + 1] * (1.0 - fx) + f[i + 1, j, k + 1] * fx
    c11 = f[i, j + 1, k + 1] * (1.0 - fx) + f[i + 1, j + 1, k + 1] * fx
    return ((c00 * (1.0 - fy) + c10 * fy) * (1.0 - fz)
            + (c01 * (1.0 - fy) + c11 * fy) * fz)


@wp.kernel
def bake_noise(t: float, f: wp.array3d(dtype=wp.vec3)):
    i, j, k = wp.tid()
    s = wp.rand_init(11)
    x = (float(i) / NG_SCALE - NR) * NOISE_FREQ
    y = (float(j) / NG_SCALE - NR) * NOISE_FREQ
    z = (float(k) / NG_SCALE - NR) * NOISE_FREQ
    f[i, j, k] = wp.curlnoise(s, wp.vec4(x, y, z, t * NOISE_RATE))


@wp.kernel
def step_gas(pos: wp.array(dtype=wp.vec4),
             vel: wp.array(dtype=wp.vec3),
             birth: wp.array(dtype=float),
             life: wp.array(dtype=float),
             noise: wp.array3d(dtype=wp.vec3),
             cx: float, cy: float, cz: float,
             tw: float, dt: float, kind: int, seed: int,
             v0: float, drag: float, buoy: float, buoy_tau: float,
             grav: float, curl: float, r0: float, r1: float, r_pow: float,
             front_c: float):
    """One kernel, four cohorts. kind: 0 fire, 1 smoke, 2 dust, 3 sparks.

    A slot is born the first frame tw passes its (fixed, sorted) birth time and
    is dead for good once it passes birth + life -- so the whole cohort is a
    pure function of tw with no free list. The dead sentinel is the one rule
    every consumer of the position buffer tests: w < 0.
    """
    i = wp.tid()
    b = birth[i]
    age = tw - b
    if age < 0.0 or age >= life[i]:
        pos[i] = wp.vec4(0.0, -1000.0, 0.0, -1.0)
        return
    q = pos[i]
    p = wp.vec3(q[0], q[1], q[2])
    v = vel[i]
    if q[3] < 0.0:
        # ── birth ───────────────────────────────────────────────────────────
        s = wp.rand_init(seed, i)
        z = 2.0 * wp.randf(s) - 1.0
        a = wp.randf(s) * 6.2831853
        rxy = wp.sqrt(wp.max(1.0 - z * z, 0.0))
        d = wp.vec3(rxy * wp.cos(a), z, rxy * wp.sin(a))
        if kind == 2:
            # Dust rides the shock front: a slot born at tw = b appears on the
            # ground annulus of radius c*b, which IS the ring racing outward.
            # Ground-hugging by construction: the kick is almost all RADIAL,
            # the vertical component is a tenth of it, and gravity is full g.
            # Dust is what the shock front TEARS OFF the ground as it passes --
            # the smoke column is the only thing in this demo that rises.
            rr = front_c * b * (0.90 + 0.15 * wp.randf(s))
            p = wp.vec3(rr * wp.cos(a), 0.05 + 0.60 * wp.randf(s), rr * wp.sin(a))
            sp = v0 * (0.45 + 0.85 * wp.randf(s))
            v = wp.vec3(wp.cos(a) * sp, sp * (0.06 + 0.30 * wp.randf(s)),
                        wp.sin(a) * sp)
        else:
            p = wp.vec3(cx, cy, cz) + d * (0.20 + 0.95 * wp.pow(wp.randf(s), 0.3333))
            sp = v0 * (0.35 + 0.95 * wp.randf(s))
            if kind == 1:
                # The mushroom, for free: smoke born in the first fifth of a
                # second is thrown sideways and rolls over into the head; smoke
                # born later goes straight up and is the stem underneath it.
                lat = 0.40 + 0.75 * wp.exp(-b / 0.42)
                dd = wp.vec3(d[0] * lat, wp.abs(d[1]) * 0.55 + 0.55, d[2] * lat)
            else:
                dd = wp.vec3(d[0], d[1] + 0.32, d[2])
            v = wp.normalize(dd) * sp
    # ── advect ──────────────────────────────────────────────────────────────
    acc = gas_noise(noise, p) * curl
    acc = acc + wp.vec3(0.0, buoy * wp.exp(-age / buoy_tau) - grav, 0.0)
    acc = acc - v * drag
    v = v + acc * dt
    p = p + v * dt
    if p[1] < 0.10:                       # the yard's floor, cheaply
        p = wp.vec3(p[0], 0.10, p[2])
        # Dust that reaches the ground STAYS on it -- no bounce, or a settling
        # ring turns back into a fountain one frame later.
        if kind == 3:
            v = wp.vec3(0.0, 0.0, 0.0)        # a landed ember stays put and burns out
        else:
            vy = float(0.0)
            if kind != 2:
                vy = wp.abs(v[1]) * 0.20
            v = wp.vec3(v[0] * 0.92, vy, v[2] * 0.92)
    # Radius over life. r_pow shapes it: 1 is the linear growth the gas wants,
    # and > 1 with r1 < r0 is the ember's hold-then-drop -- the only fade a
    # HostRing field HAS, because age-based fade is the emitter's and there is
    # no emitter here.
    f = wp.pow(age / life[i], r_pow)
    pos[i] = wp.vec4(p[0], p[1], p[2], r0 + (r1 - r0) * f)
    vel[i] = v


class Cohort:
    """One HostRing field plus the Warp state that feeds it.

    The device buffer is `wp.vec4`, which is byte-identical to ParticlePos, so
    the (n, 4) float32 the readback lands in IS the submit buffer -- one DtoH
    copy into a pinned host array and one memcpy into the field's ring, with no
    repack anywhere in between.
    """

    def __init__(self, name, n, kind, emit, life, radius, seed, skew=1.0, r_pow=1.0):
        self.name, self.n, self.kind, self.r_pow = name, n, kind, r_pow
        gen = np.random.default_rng(seed)
        births = np.sort(emit * gen.random(n) ** skew).astype(np.float32)
        lives = gen.uniform(life[0], life[1], n).astype(np.float32)
        self.births = births
        self.t_last = float((births + lives).max())     # when the cohort is gone
        self.pos = wp.zeros(n, dtype=wp.vec4, device=device)
        self.vel = wp.zeros(n, dtype=wp.vec3, device=device)
        self.birth = wp.array(births, dtype=float, device=device)
        self.life = wp.array(lives, dtype=float, device=device)
        try:
            self.host = wp.zeros(n, dtype=wp.vec4, device="cpu", pinned=True)
        except TypeError:                               # older warp: plain host mem
            self.host = wp.zeros(n, dtype=wp.vec4, device="cpu")
        self.host_np = self.host.numpy()                # a VIEW of the host array
        self.field = None
        self.radius = radius

    def advance(self, tw, dt, params):
        wp.launch(step_gas, dim=self.n,
                  inputs=[self.pos, self.vel, self.birth, self.life, noise_grid,
                          CHARGE[0], CHARGE[1], CHARGE[2], tw, dt, self.kind,
                          9000 + self.kind, *params,
                          self.radius[0], self.radius[1], self.r_pow, FRONT_C],
                  device=device)

    def fetch(self, tw):
        """Issue the device->host copy. Births are SORTED, so only the prefix
        that has been born yet has to cross the bus -- which is most of the
        saving in the first half second, when the picture matters most."""
        n = int(np.searchsorted(self.births, tw, side="right"))
        if n:
            wp.copy(self.host, self.pos, count=n)
        return n

    def push(self, n):
        """One memcpy into the field's ring. submit() also sets the live count."""
        if n:
            self.field.submit(self.host_np[:n])
        else:
            self.field.set_live_count(0)


cohorts = []
fields = []

if not NO_GAS:
    wp.init()
    device = wp.get_preferred_device()
    noise_grid = wp.zeros((NG, NG, NG), dtype=wp.vec3, device=device)

    cohorts = [
        Cohort("fire", FIRE_N, 0, FIRE_EMIT, FIRE_LIFE, FIRE_R, SEED + 1),
        Cohort("smoke", SMOKE_N, 1, SMOKE_EMIT, SMOKE_LIFE, SMOKE_R, SEED + 2,
               skew=SMOKE_SKEW),
        Cohort("dust", DUST_N, 2, DUST_EMIT, DUST_LIFE, DUST_R, SEED + 3),
        Cohort("spark", SPARK_N, 3, SPARK_EMIT, SPARK_LIFE, SPARK_R, SEED + 4,
               skew=1.8, r_pow=1.5),
    ]
    fire, smoke, dust, spark = cohorts
    fire.params = (FIRE_V0, FIRE_DRAG, FIRE_BUOY, 0.30, 0.0, FIRE_CURL)
    smoke.params = (SMOKE_V0, SMOKE_DRAG, SMOKE_BUOY, 1.50, 0.0, SMOKE_CURL)
    dust.params = (DUST_V0, DUST_DRAG, 0.0, 1.0, DUST_GRAV, DUST_CURL)
    spark.params = (SPARK_V0, SPARK_DRAG, 0.0, 1.0, 9.81, SPARK_CURL)

    def make_field(n, radius):
        c = tp.ParticleField.Config()
        c.capacity = n
        c.ownership = tp.ParticleField.Ownership.HostRing
        c.w_semantic = tp.ParticleField.WSemantic.Radius   # w IS the world radius
        c.uniform_radius = radius
        f = tp.ParticleField.create(c)
        f.frustum_culled = False        # the bounds go stale the moment it detonates
        f.set_live_count(0)
        scene.add(f)
        return f

    cx, cy, cz = CHARGE
    VOLUMES = not NO_FOG                # DensityRepr rides the froxel FOG pass

    # ── fire ────────────────────────────────────────────────────────────────
    # The flame is the DENSITY volume's blackbody ramp, not the billboards: the
    # ramp is hot at the box FLOOR and cools toward its top, so the box has to
    # be the fireball and nothing else -- a box the size of the yard would put
    # the whole flame in the bottom two per cent of the ramp.
    fire.field = make_field(FIRE_N, FIRE_R[1])
    if VOLUMES:
        fire.field.set_density_repr(tp.Vector3(cx, 3.6, cz), tp.Vector3(9.0, 4.6, 9.0),
                                    FIRE_SIGMA * SIGMA, 96)
        _d = fire.field.density_repr
        _d.albedo = tp.Color(0.30, 0.22, 0.17)
        _d.anisotropy = 0.15
        _d.temp_bottom_k = 2700.0
        _d.temp_top_k = 1250.0
        _d.temp_falloff = 1.5
        _d.emissive_intensity = 0.0     # driven by the timeline
    fire.field.set_billboard_repr(tp.Color(1.00, 0.63, 0.24), tp.Color(1.00, 0.20, 0.03),
                                  0.0, 0.40)
    _b = fire.field.billboard_repr
    _b.softness = 0.80
    _b.fade_power = 0.0                 # no age on a HostRing field: w carries the life
    _b.size_taper = 0.0
    _b.bright_jitter = 0.70
    _b.near_fade = 1.5
    _b.glow = 1.10                      # this field's own bloom pyramid
    _b.glow_threshold = 0.0

    # ── smoke ───────────────────────────────────────────────────────────────
    # DensityRepr only. Soot is a DARK medium -- a bright albedo here and the
    # column reads as steam.
    smoke.field = make_field(SMOKE_N, SMOKE_R[1])
    if VOLUMES:
        smoke.field.set_density_repr(tp.Vector3(0.0, 12.0, 0.0),
                                     tp.Vector3(16.0, 13.0, 16.0), SMOKE_SIGMA * SIGMA, 128)
        _d = smoke.field.density_repr
        _d.albedo = tp.Color(0.115, 0.108, 0.100)
        _d.anisotropy = 0.28

    # ── dust ────────────────────────────────────────────────────────────────
    dust.field = make_field(DUST_N, DUST_R[1])
    if VOLUMES:
        dust.field.set_density_repr(tp.Vector3(0.0, 1.5, 0.0),
                                    tp.Vector3(36.0, 2.6, 36.0), DUST_SIGMA * SIGMA, 128)
        _d = dust.field.density_repr
        _d.albedo = tp.Color(0.62, 0.58, 0.52)
        _d.anisotropy = 0.10
    # NO billboards. The plan asked for "very dim billboards for sunlit
    # glints", and they were tried: once the ring is genuinely ground-hugging,
    # 170k additive quads packed into a half-metre layer read as a carpet of
    # static across the whole yard, not as glinting grit. The extinction volume
    # is the honest representation for dust and it is the only one here.

    # ── sparks ──────────────────────────────────────────────────────────────
    #
    # HostRing, not the analytic emitter, and that is a DEVIATION FROM THE PLAN
    # with a hard reason. Ownership::Renderer's emitter is a STEADY-STATE
    # generator by construction: particle_emit.comp gives every slot a random
    # phase into its own period (`float birth; // phase offset into the period,
    # [0,1)`), so at any emitter time a `dutyCycle` fraction of slots is alive
    # and WHICH slots those are changes continuously. Feed it a long period and
    # a short duty -- the obvious way to spell "one burst" -- and what you get
    # is a permanent thin drizzle of brand-new sparks out of the charge, minutes
    # after the flame is out. There is no parameterisation that fixes it: the
    # phase spread is scale-invariant, so no remapping of the clock turns a
    # steady state into a burst.
    #
    # What is lost by moving: age fade, size taper and the hot->cool colour ramp
    # are all derived from the emitter's closed form, and on a HostRing field
    # ageFrac is pinned to 0. The radius IS the fade here (an additive quad's
    # contribution goes as its area, so r -> 0 is a fade), and because this is
    # ONE burst rather than a steady state, the whole field ages together --
    # so cooling the field's own colour over tw below is a faithful stand-in
    # for cooling each ember over its own age.
    spark.field = make_field(SPARK_N, SPARK_R[0])
    spark.field.set_billboard_repr(tp.Color(1.00, 0.78, 0.36), tp.Color(1.00, 0.17, 0.02),
                                   0.0, 1.0)
    _b = spark.field.billboard_repr
    _b.softness = 0.32
    _b.fade_power = 0.0                 # inert on HostRing: the radius is the fade
    _b.size_taper = 0.0
    _b.bright_jitter = 0.65
    _b.stretch_seconds = 0.030          # smeared along (pos - prevPos) of the ring
    _b.stretch_max = 34.0
    _b.stretch_max_screen = 0.055
    _b.near_fade = 1.2
    _b.glow = 0.50
    SPARK_HOT = ((1.00, 0.80, 0.40), (1.00, 0.30, 0.06))   # new ember -> old ember

    # ── streaks: the ONE thing HostRing cannot do ───────────────────────────
    # ParticleFieldPass.cpp is explicit: "the stretch is a Renderer-mode
    # feature and is silently zero elsewhere", because a HostRing field's
    # prevPositions are the previous RING SLOT and their age is never
    # published, so there is no dt to divide by. Round sprites are what the
    # ember field above can be, and a blast that throws no streaks reads wrong.
    #
    # So a SECOND, small Renderer field does the streaks -- and only for the
    # burst window. Its slots respawn continuously (that is the steady-state
    # emitter's nature, see the note above), which is exactly right while the
    # charge is still ejecting and exactly wrong afterwards, so the field's
    # intensity is ramped out over 0.45-0.80 s and it is PARKED at 0.85 s.
    # Nothing is instantiated after the burst; the HostRing embers carry the
    # rest of the story.
    STREAK_N = int(20_000 * GAS)
    STREAK_T = 0.85
    _kc = tp.ParticleField.Config()
    _kc.capacity = STREAK_N
    _kc.ownership = tp.ParticleField.Ownership.Renderer
    _kc.w_semantic = tp.ParticleField.WSemantic.Radius
    _kc.uniform_radius = 0.05
    streaks = tp.ParticleField.create(_kc)
    streaks.frustum_culled = False
    streaks.set_billboard_repr(tp.Color(1.00, 0.84, 0.46), tp.Color(1.00, 0.24, 0.03),
                               0.0, 1.0)
    _b = streaks.billboard_repr
    _b.softness = 0.20
    _b.fade_power = 2.0                 # a Renderer field HAS an age: hold, then drop
    _b.size_taper = 0.60
    _b.bright_jitter = 0.60
    _b.stretch_seconds = 0.032          # the exact analytic velocity, smeared
    _b.stretch_max = 34.0
    _b.stretch_max_screen = 0.055
    _b.near_fade = 1.2
    _b.glow = 0.70
    _e = streaks.emitter                # NB: a COPY -- mutate and hand it back
    _e.spawn_center = tp.Vector3(cx, cy + 0.2, cz)
    _e.spawn_half_extent = tp.Vector3(0.45, 0.45, 0.45)
    _e.velocity = tp.Vector3(0.0, 8.0, 0.0)   # the ground reflection's upward bias
    _e.speed_spread = 22.0                    # isotropic: this IS the radial burst
    _e.accel = tp.Vector3(0.0, -9.81, 0.0)
    _e.lifetime = STREAK_T
    _e.lifetime_jitter = 0.25
    _e.duty_cycle = 1.0
    _e.size = 0.05
    _e.size_jitter = 0.65
    _e.seed = SEED * 1013 + 5
    streaks.set_emitter(_e)
    streaks.set_emitter_time(0.0, DT)
    streaks.set_live_count(0)
    scene.add(streaks)

    fields = [c.field for c in cohorts] + [streaks]
    GAS_END = max(c.t_last for c in cohorts) + 0.1

    # ── PREWARM: one throwaway frame with every field LIVE ───────────────────
    # Creating the fields up front is necessary but not sufficient. A density
    # volume's image is allocated (and its resolution latched) the first frame
    # the field actually has live particles, and a billboard field's glow
    # pyramid allocates its offscreen target the first time it draws -- so with
    # nothing but set_live_count(0) at startup, ALL of that lands on the
    # detonation frame. Measured in the window: a 34.9 ms worst frame at t0,
    # against 6-9 ms once the gas is running. One particle each, for one frame
    # nobody sees, moves it to startup where it belongs.
    _warm = np.array([[0.0, 1.0, 0.0, 0.01]], np.float32)
    for c in cohorts:
        c.field.submit(_warm)
    streaks.set_live_count(STREAK_N)
    renderer.render(scene, camera)
    for c in cohorts:
        c.field.set_live_count(0)
    streaks.set_live_count(0)
    print(f"gas: fire {FIRE_N:,} + smoke {SMOKE_N:,} + dust {DUST_N:,} HostRing "
          f"+ sparks {SPARK_N:,} HostRing "
          f"({16 * (FIRE_N + SMOKE_N + DUST_N + SPARK_N) / 1e6:.1f} MB/frame at full "
          f"count), on {device}")
else:
    GAS_END = 0.0

# Billboards are composited after the upscaler and are outside auto-exposure,
# so their intensity is tied to the pinned scene exposure BY HAND, once, here.
# Change tone_mapping_exposure and these follow it instead of blowing out.
_EXP = 0.55 / max(renderer.tone_mapping_exposure, 1e-3)
BB_FIRE = 0.055 * _EXP     # low ON PURPOSE: 120k additive quads inside a 4 m ball
BB_SPARK = 0.85 * _EXP     # emission ramp. The billboards are only its sparkle.
BB_STREAK = 1.6 * _EXP
FIRE_EMISSIVE = 4.5                     # emission is intensity * THIS field's sigma
_gas_live = False


def step_gas_frame(dt):
    """Advance the three HostRing cohorts and publish them. Returns (kernel, submit)."""
    global _gas_live
    if NO_GAS:
        return 0.0, 0.0
    state = blast_timeline(sim_time)
    tw = state["tw"]
    if not state["armed"] or tw > GAS_END:
        if _gas_live:                   # park, do not destroy: one entry, no churn
            for c in cohorts:
                c.field.set_live_count(0)
            streaks.set_live_count(0)
            _gas_live = False
        return 0.0, 0.0
    _gas_live = True
    t0 = time.perf_counter()
    wp.launch(bake_noise, dim=(NG, NG, NG), inputs=[tw, noise_grid], device=device)
    for c in cohorts:
        c.advance(tw, dt, c.params)
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    counts = [c.fetch(tw) for c in cohorts]
    wp.synchronize_device(device)                       # the DtoH copies
    for c, n in zip(cohorts, counts):
        c.push(n)
    t2 = time.perf_counter()

    gain = state["bb_gain"]
    fire.field.billboard_repr.intensity = BB_FIRE * gain
    if not NO_FOG:
        # NEVER exactly zero while the gas is live. Any emissive volume switches
        # the per-pixel dust march from 24 midpoint steps to 32 dithered ones,
        # and the midpoint form quantises the smoke column into hard horizontal
        # shells -- 2 m steps through a 10 m plume. The dither TAA converges;
        # the shells it does not. The floor costs nothing because the fire
        # volume's own sigma is zero once the flame's particles are dead, and
        # emission is intensity * THAT volume's sigma.
        fire.field.density_repr.emissive_intensity = max(FIRE_EMISSIVE * state["fire_e"],
                                                         1.0e-3)
    # One burst ages as one population, so cooling the FIELD is a faithful
    # stand-in for the per-particle colour ramp a HostRing field cannot have.
    k = min(tw / 2.6, 1.0)
    hot, cool = SPARK_HOT
    spark.field.billboard_repr.color_hot = tp.Color(*(a + (b - a) * k
                                                      for a, b in zip(hot, cool)))
    spark.field.billboard_repr.intensity = (BB_SPARK * (1.0 + 0.6 * (gain - 1.0))
                                            * (1.0 - 0.55 * k))
    # The streak field ejects only while the charge is still ejecting.
    if tw < STREAK_T:
        streaks.set_emitter_time(tw, dt)
        streaks.set_live_count(STREAK_N)
        streaks.billboard_repr.intensity = (BB_STREAK * gain
                                            * min(max((0.80 - tw) / 0.35, 0.0), 1.0))
    else:
        streaks.set_live_count(0)
    return t1 - t0, t2 - t1


# --- frame ------------------------------------------------------------------

sim_time = 0.0
prof = {"gas": 0.0, "submit": 0.0, "blast": 0.0, "physx": 0.0, "render": 0.0, "n": 0}


def step_frame(dt=DT):
    """Advance one frame of sim time. Returns the bench's phase durations.

    `dt` is a parameter, not a constant, because phase 4 ramps sim time through
    the detonation for the slow-motion cut.
    """
    global sim_time
    t_prev = sim_time
    sim_time += dt
    t0 = time.perf_counter()
    if t_prev < T0 <= sim_time:
        detonate()
    state = blast_timeline(sim_time)
    if state["wind"] > 0.0:
        blast_wind(state["wind"])
    flash.intensity = state["flash"]
    t1 = time.perf_counter()
    world.step(dt)
    t2 = time.perf_counter()
    gas, submit = step_gas_frame(dt)
    prof["blast"] += t1 - t0
    prof["physx"] += t2 - t1
    prof["gas"] += gas
    prof["submit"] += submit
    prof["n"] += 1
    return gas, submit, t1 - t0, t2 - t1


def apply_shake():
    """Offset the camera by the current shake; returns the offset applied."""
    sx, sy, sz = blast_timeline(sim_time)["shake"]
    camera.position.set(camera.position.x + sx, camera.position.y + sy,
                        camera.position.z + sz)
    return sx, sy, sz


def yard_stats():
    """Sanity: nothing at NaN, nothing a kilometre away, how much is still moving."""
    p = np.array([[b.position.x, b.position.y, b.position.z] for b in bodies])
    v = np.array([[b.linear_velocity.x, b.linear_velocity.y, b.linear_velocity.z]
                  for b in bodies])
    finite = np.isfinite(p).all(axis=1)
    r = np.linalg.norm(p[finite], axis=1)
    moving = int((np.linalg.norm(v[finite], axis=1) > 1.0).sum())
    return dict(nan=int((~finite).sum()), r_max=float(r.max()),
                y_max=float(p[finite][:, 1].max()), moving=moving)


def gas_stats():
    """Where the gas actually IS -- the only honest way to tune sigma, since a
    column that is twice as wide as you think is four times more transparent."""
    out = {}
    for c in cohorts:
        n = int(np.searchsorted(c.births, blast_timeline(sim_time)["tw"], side="right"))
        if n == 0:
            continue
        p = c.host_np[:n]
        live = p[:, 3] >= 0.0
        if not live.any():
            continue
        q = p[live]
        r = np.hypot(q[:, 0], q[:, 2])
        out[c.name] = (int(live.sum()), float(np.median(q[:, 1])),
                       float(np.percentile(q[:, 1], 95)), float(np.median(r)),
                       float(np.percentile(r, 95)))
    return out


def report():
    s = yard_stats()
    n = max(prof["n"], 1)
    live = sum(f.live_count for f in fields)
    print(f"  t={sim_time:5.2f}  nan={s['nan']}  r_max={s['r_max']:6.1f} m  "
          f"y_max={s['y_max']:5.2f} m  moving={s['moving']}/{len(bodies)}  "
          f"gas live={live:,}")
    for name, (k, ymed, y95, rmed, r95) in gas_stats().items():
        print(f"  {name:>5}: {k:7,} live  y {ymed:5.1f}/{y95:5.1f} m  "
              f"r {rmed:5.1f}/{r95:5.1f} m  (median/p95)")
    print(f"  per frame: gas {1e3 * prof['gas'] / n:.2f} ms, submit "
          f"{1e3 * prof['submit'] / n:.2f} ms, blast {1e3 * prof['blast'] / n:.2f} ms, "
          f"physx {1e3 * prof['physx'] / n:.2f} ms, render "
          f"{1e3 * prof['render'] / n:.2f} ms")


def save(path):
    renderer.save_frame(scene, camera, path)      # renders + reads back


# --- run --------------------------------------------------------------------

orbit(0)

if SHOT:
    frames = int(round(SHOT_T * FPS))
    t_start = time.perf_counter()
    for f in range(frames):
        orbit(f)
        step_frame()
        apply_shake()
        tr = time.perf_counter()
        renderer.render(scene, camera)            # keep the temporal history honest
        prof["render"] += time.perf_counter() - tr
        if f % 60 == 0:
            print(f"  t={sim_time:5.2f}  moving={yard_stats()['moving']}", flush=True)
    if SPIN == 0.0:
        for _ in range(16):                       # settle the accumulation on the last pose
            renderer.render(scene, camera)
    else:
        orbit(frames)
    out = cli_arg("--out", "warp_explosion.png", str)
    save(out)
    print(f"simulated {SHOT_T:.1f} s ({frames} frames) in "
          f"{time.perf_counter() - t_start:.1f}s, wrote {out}")
    report()
elif VIDEO > 0:
    outdir = tempfile.mkdtemp(prefix="warp_explosion_frames_")
    total = int(round(VIDEO * FPS))
    t_start = time.perf_counter()
    for _ in range(24):                           # warm the temporal history on frame 0
        renderer.render(scene, camera)
    for k in range(total):
        orbit(k)
        step_frame()
        apply_shake()
        save(os.path.join(outdir, f"f{k:05d}.png"))
        if k % 60 == 0:
            print(f"  frame {k}/{total}  ({time.perf_counter() - t_start:.0f}s elapsed)",
                  flush=True)
    print(f"rendered {total} frames in {time.perf_counter() - t_start:.0f}s -> {outdir}")
    report()
    ff = find_ffmpeg()
    if ff:
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "60",
                        "-i", os.path.join(outdir, "f%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
                        "warp_explosion.mp4"], check=False)
        print("wrote warp_explosion.mp4")
elif BENCH:
    # Bench the interesting state: bricks in flight, not a sleeping stack.
    # Headless on purpose (vsync off, no present), which also means each
    # render() waits on its own frame instead of pipelining -- so the render
    # column is a per-frame GPU upper bound, not the window's throughput. The
    # window prints its own live fps; that is the number to quote.
    while sim_time < T0 + 0.6:
        step_frame()
        renderer.render(scene, camera)
    prof.update(gas=0.0, submit=0.0, blast=0.0, physx=0.0, render=0.0, n=0)
    live = sum(f.live_count for f in fields)
    bench_loop(step_frame, lambda: renderer.render(scene, camera),
               ("gas", "submit", "blast", "physx"), 20, 200,
               f"{len(bodies)} bodies + {live:,} live particles (headless, serialized)")
    report()
else:
    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target.set(*CAM_TARGET)
    canvas.on_window_resize(resize_handler(camera, renderer))
    fps = {"t": time.perf_counter(), "n": 0, "shake": (0.0, 0.0, 0.0),
           "last": time.perf_counter(), "worst": 0.0}

    def animate():
        now0 = time.perf_counter()
        fps["worst"] = max(fps["worst"], now0 - fps["last"])
        fps["last"] = now0
        step_frame()
        # Undo last frame's shake before the controls read the camera back.
        sx, sy, sz = fps["shake"]
        camera.position.set(camera.position.x - sx, camera.position.y - sy,
                            camera.position.z - sz)
        controls.update()
        fps["shake"] = apply_shake()
        renderer.render(scene, camera)
        fps["n"] += 1
        if fps["n"] >= 120:
            now = time.perf_counter()
            # The worst frame in the window is the interesting one: a field
            # created mid-run would show up here as a device idle, and the
            # whole point of parking all four at startup is that it never does.
            print(f"  t={sim_time:5.2f}  {fps['n'] / (now - fps['t']):5.1f} fps  "
                  f"(worst {1e3 * fps['worst']:5.1f} ms, {len(bodies)} bodies, "
                  f"{sum(f.live_count for f in fields):,} particles)", flush=True)
            fps["t"], fps["n"], fps["worst"] = now, 0, 0.0

    canvas.animate(animate)
