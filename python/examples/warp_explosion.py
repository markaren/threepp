"""Blast Yard: a brick test range demolished by a charge -- PhysX + threepp/Vulkan.

Phase 1 of the explosion demo: the scene and the rigid demolition core. A
light-grey void with a grid floor, the way a test range looks in Unreal or
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

Everything is driven off one sim clock and one blast timeline, so the later
phases (gas, drums, slow-motion film) can hang off the same t0 without
re-deriving it.

    python warp_explosion.py                 # window; drag to orbit, Esc quits
    python warp_explosion.py --shot 4        # headless: sim 4 s, write png
    python warp_explosion.py --video 6       # headless: 6 s of frames (+mp4 if ffmpeg)
    python warp_explosion.py --bench         # timed phase breakdown, post-detonation
    python warp_explosion.py --yield 2400    # bigger charge (J0, kg m/s at 1 m)
    python warp_explosion.py --charge 0,0.6,-1.2 --t0 1.5
    python warp_explosion.py --courses 20 --shot 4      # taller walls, more bodies
    python warp_explosion.py --shot 4 --spin 0.25       # orbit while simulating
    python warp_explosion.py --no-ao                    # RT AO/GI off (~6.5 ms/frame back)
    python warp_explosion.py --no-fog --msaa 1          # the other two render knobs

Vulkan only: the later phases render the fireball with tp.ParticleField, which
draws nothing on the GL path by design.
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

camera = tp.PerspectiveCamera(40, canvas.aspect(), 0.05, 800)
CAM_R, CAM_Y, CAM_TARGET = 22.0, 7.5, (0.0, 1.8, 0.0)
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
    front    shock-front radius, m -- phase 2's dust ring rides this
    shake    (x, y, z) camera offset, m
    """
    tw = t - T0
    if tw < 0.0:
        return dict(tw=tw, armed=False, flash=0.0, wind=0.0, front=0.0,
                    shake=(0.0, 0.0, 0.0))
    scale = math.sqrt(YIELD / 1400.0)
    flash_i = FLASH_I * scale * math.exp(-tw / FLASH_TAU) if tw < 0.5 else 0.0
    wind = WIND_A0 * math.exp(-tw / WIND_TAU) if tw < WIND_T else 0.0
    shake = (0.0, 0.0, 0.0)
    if tw < SHAKE_T:
        a = SHAKE_A * scale * math.exp(-tw / SHAKE_TAU)
        shake = tuple(a * sum(math.sin(2.0 * math.pi * f * tw + p) for f, p in bank) / 3.0
                      for bank in _shake_bank)
    return dict(tw=tw, armed=True, flash=flash_i, wind=wind,
                front=28.0 * tw, shake=shake)


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


# --- frame ------------------------------------------------------------------

sim_time = 0.0
prof = {"blast": 0.0, "physx": 0.0, "render": 0.0, "n": 0}


def step_frame(dt=DT):
    """Advance one frame of sim time. Returns (blast s, physx s) for the bench.

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
    prof["blast"] += t1 - t0
    prof["physx"] += t2 - t1
    prof["n"] += 1
    return t1 - t0, t2 - t1


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


def report():
    s = yard_stats()
    n = max(prof["n"], 1)
    print(f"  t={sim_time:5.2f}  nan={s['nan']}  r_max={s['r_max']:6.1f} m  "
          f"y_max={s['y_max']:5.2f} m  moving={s['moving']}/{len(bodies)}")
    print(f"  per frame: blast {1e3 * prof['blast'] / n:.2f} ms, physx "
          f"{1e3 * prof['physx'] / n:.2f} ms, render {1e3 * prof['render'] / n:.2f} ms")


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
    prof.update(blast=0.0, physx=0.0, render=0.0, n=0)
    bench_loop(step_frame, lambda: renderer.render(scene, camera),
               ("blast", "physx"), 20, 200, f"{len(bodies)} bodies (headless, serialized)")
    report()
else:
    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target.set(*CAM_TARGET)
    canvas.on_window_resize(resize_handler(camera, renderer))
    fps = {"t": time.perf_counter(), "n": 0, "shake": (0.0, 0.0, 0.0)}

    def animate():
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
            print(f"  t={sim_time:5.2f}  {fps['n'] / (now - fps['t']):5.1f} fps  "
                  f"({len(bodies)} bodies)", flush=True)
            fps["t"], fps["n"] = now, 0

    canvas.animate(animate)
