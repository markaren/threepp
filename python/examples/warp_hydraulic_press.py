"""A chrome sphere crushed by a hydraulic press in a Cornell box: Warp + threepp.

The sphere is a thin metal shell with real bending stiffness: two icospheres a
few centimetres apart (outer skin, inner skin) braced into a truss -- edges in
each layer, a radial strut per vertex, and two diagonals per edge. Bending then
costs stretch in the layers, so the shell holds its shape under gravity and an
inverted dent is a genuine crease, not a free flip (a single layer with
distance-based "bending" is sign-blind and deflates like an empty balloon).
What makes it metal rather than rubber is plasticity -- any strut strained past
its yield creeps its rest length toward the current length, so the creases the
press puts in stay there when the plate lifts. No internal pressure: the shell
is vented. Everything runs in raw Warp kernels; threepp gets the outer skin's
positions and smooth normals once per frame.

The solve is Gauss-Seidel over a greedy graph colouring of the constraints (no
two constraints in a colour share a vertex, so each colour is one race-free
launch), with many small substeps. The whole frame is captured once into a
CUDA graph and replayed.

Rendered with the Vulkan path so the chrome mirrors the red and green walls and
the reflections warp across the buckling surface as it goes.

    pip install warp-lang
    python warp_hydraulic_press.py               # window; drag to orbit, Esc quits
    python warp_hydraulic_press.py --shot 7      # headless: sim 7 s, write png
    python warp_hydraulic_press.py --video 9     # headless: 9 s of frames (+mp4 if ffmpeg)
    python warp_hydraulic_press.py --gl          # OpenGL renderer (no mirror reflections)
    python warp_hydraulic_press.py --size 960x768               # smaller window
    python warp_hydraulic_press.py --subdiv 6 --substeps 160   # finer shell, more substeps
    python warp_hydraulic_press.py --video 9 --spin 0.1        # turntable: orbit 0.1 deg/frame
    python warp_hydraulic_press.py --shot 1 --freeze --spin 1.5 --angle -45   # no sim: camera only

Warp falls back to CPU if no CUDA device is present -- slower, same picture.
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
from warp_common import (accum_normals, cli_arg, find_ffmpeg, icosphere, integrate,
                         orbit_loop, parse_size, scatter_soup, standard_material)

SHOT = "--shot" in sys.argv
SHOT_T = cli_arg("--shot", 7.0, float)
VIDEO = cli_arg("--video", 0.0, float)      # seconds of frames to render
GL = "--gl" in sys.argv
SUBDIV = cli_arg("--subdiv", 5, int)       # icosphere subdivisions per skin

# --- tunables ---------------------------------------------------------------

RADIUS = 0.6
SUBSTEPS = cli_arg("--substeps", 96, int)  # many small steps with one sweep each
                                           # converge better than fewer steps
                                           # with more sweeps
DT = 1.0 / (60.0 * SUBSTEPS)
ITERATIONS = cli_arg("--iters", 1, int)    # Gauss-Seidel sweeps per substep
DAMPING = 0.002              # per substep
THICK = cli_arg("--thick", 0.015, float)   # shell thickness: spacing of the two skins
# Strut classes. A thin shell is a stiff membrane with cheap bending: the skins
# are (nearly) inextensible and rarely yield, the shear diagonals between the
# skins are soft and yield easily -- so wrinkling costs little, and once a
# wrinkle has formed the diagonals remember it.
STIFF_SKIN = 1.0
STIFF_DIAG = cli_arg("--diag", 0.25, float)
STIFF_RAD = 1.0              # radial struts hold the thickness
YIELD_SKIN = cli_arg("--yskin", 0.06, float)
YIELD_DIAG = cli_arg("--ydiag", 0.01, float)
YIELD_RAD = 0.06
PLASTIC_RATE = cli_arg("--plastic", 0.04, float)   # excess adopted per substep
MU = 0.4                     # friction against floor and plate
GRAVITY = wp.vec3(0.0, -cli_arg("--gravity", 9.81, float), 0.0)

ROOM = 3.2                   # Cornell box side
BED_T = 0.12                 # press bed (anvil) thickness; the sphere sits on it
FLOOR_Y = BED_T
PLATE_R = 0.6                # ram head radius: just wider than the crushed cap
PLATE_T = 0.10               # ram head thickness
PLATE_TOP = FLOOR_Y + 1.45   # plate underside, resting height
PLATE_MIN = FLOOR_Y + cli_arg("--gap", 0.30, float)   # plate underside at full stroke
# (t_start, t_end, y_from, y_to): hold, press, dwell, lift, hold
STROKE = [(0.0, 0.8, PLATE_TOP, PLATE_TOP),
          (0.8, 4.3, PLATE_TOP, PLATE_MIN),
          (4.3, 5.3, PLATE_MIN, PLATE_MIN),
          (5.3, 7.0, PLATE_MIN, PLATE_TOP),
          (7.0, 9.0, PLATE_TOP, PLATE_TOP)]


def plate_height(t):
    """Underside height of the press plate at sim time t, eased per segment."""
    for t0, t1, y0, y1 in STROKE:
        if t <= t1:
            s = min(max((t - t0) / max(t1 - t0, 1e-9), 0.0), 1.0)
            s = s * s * (3.0 - 2.0 * s)
            return y0 + (y1 - y0) * s
    return STROKE[-1][3]


# --- warp kernels -----------------------------------------------------------

@wp.func
def collide(p: wp.vec3, plate_y: float) -> wp.vec3:
    if p[1] < FLOOR_Y:
        p = wp.vec3(p[0], FLOOR_Y, p[2])
    if p[1] > plate_y and p[0] * p[0] + p[2] * p[2] < PLATE_R * PLATE_R:
        p = wp.vec3(p[0], plate_y, p[2])
    return p


@wp.kernel
def solve_color(pos: wp.array(dtype=wp.vec3),
                ci: wp.array(dtype=int),
                cj: wp.array(dtype=int),
                rests: wp.array(dtype=float),
                stiffs: wp.array(dtype=float),
                start: int):
    # One colour: no two constraints here share a vertex, so in-place
    # Gauss-Seidel writes are race-free.
    k = start + wp.tid()
    i = ci[k]
    j = cj[k]
    pi = pos[i]
    pj = pos[j]
    d = pj - pi
    l = wp.length(d)
    if l > 1.0e-9:
        corr = d * (0.5 * stiffs[k] * (l - rests[k]) / l)
        pos[i] = pi + corr
        pos[j] = pj - corr


@wp.kernel
def project(pos: wp.array(dtype=wp.vec3),
            plate_ys: wp.array(dtype=float), s: int):
    i = wp.tid()
    pos[i] = collide(pos[i], plate_ys[s])


@wp.kernel
def plastic(pos: wp.array(dtype=wp.vec3),
            ci: wp.array(dtype=int),
            cj: wp.array(dtype=int),
            rests: wp.array(dtype=float),
            yields: wp.array(dtype=float),
            rate: float):
    k = wp.tid()
    l = wp.length(pos[cj[k]] - pos[ci[k]])
    r = rests[k]
    y = yields[k]
    if l > r * (1.0 + y):
        rests[k] = r + (l - r * (1.0 + y)) * rate
    elif l < r * (1.0 - y):
        rests[k] = r + (l - r * (1.0 - y)) * rate


@wp.kernel
def contacts(pos: wp.array(dtype=wp.vec3),
             prev: wp.array(dtype=wp.vec3),
             plate_ys: wp.array(dtype=float), s: int):
    i = wp.tid()
    plate_y = plate_ys[s]
    p = collide(pos[i], plate_y)
    # Friction: on contact, take back a fraction of this substep's
    # tangential motion.
    if p[1] < FLOOR_Y + 1.0e-4 or (p[1] > plate_y - 1.0e-4 and
                                    p[0] * p[0] + p[2] * p[2] < PLATE_R * PLATE_R):
        t = p - prev[i]
        p = p - wp.vec3(t[0], 0.0, t[2]) * MU
    pos[i] = p


# --- mesh construction (numpy, once) ----------------------------------------

verts0, faces = icosphere(SUBDIV)
p0 = verts0 * RADIUS

# Two skins: outer at RADIUS, inner at RADIUS - THICK, same connectivity.
# Vertices 0..N-1 are the outer skin (the rendered one), N..2N-1 the inner.
N = len(p0)
p0 = np.concatenate([p0, verts0 * (RADIUS - THICK)]).astype(np.float32)

# Constraints, all plain distance struts: edges in each skin, a radial strut
# per vertex, and the two diagonals across each edge's prism. That is the
# truss; bending stiffness is what it buys.
edges = set()
for fa, fb, fc in faces:
    for i, j in ((fa, fb), (fb, fc), (fc, fa)):
        edges.add((min(i, j), max(i, j)))

cons = []          # (i, j, stiffness, yield)
for i, j in edges:
    cons.append((i, j, STIFF_SKIN, YIELD_SKIN))            # outer skin
    cons.append((N + i, N + j, STIFF_SKIN, YIELD_SKIN))    # inner skin
    cons.append((i, N + j, STIFF_DIAG, YIELD_DIAG))        # shear diagonals
    cons.append((j, N + i, STIFF_DIAG, YIELD_DIAG))
for i in range(N):
    cons.append((i, N + i, STIFF_RAD, YIELD_RAD))          # radial strut

n_verts = len(p0)
n_cons = len(cons)

# Greedy graph colouring: a constraint takes the lowest colour not yet used
# by any constraint at either of its vertices. Per-vertex bitmask of colours.
# Colouring class by class (skins, then diagonals, then radials) needs fewer
# colours than the interleaved order, and fewer colours is fewer launches.
cons.sort(key=lambda c: (0 if (c[0] < N) == (c[1] < N) else
                         (2 if abs(c[0] - c[1]) == N else 1)))
used = [0] * n_verts
color = np.empty(n_cons, dtype=np.int32)
for k, (i, j, _, _) in enumerate(cons):
    m = used[i] | used[j]
    c = 0
    while (m >> c) & 1:
        c += 1
    color[k] = c
    used[i] |= 1 << c
    used[j] |= 1 << c
order = np.argsort(color, kind="stable")
n_colors = int(color.max()) + 1
color_count = np.bincount(color, minlength=n_colors)
color_start = np.concatenate(([0], np.cumsum(color_count)[:-1]))

ci_np = np.array([cons[k][0] for k in order], dtype=np.int32)
cj_np = np.array([cons[k][1] for k in order], dtype=np.int32)
stiff_np = np.array([cons[k][2] for k in order], dtype=np.float32)
yield_np = np.array([cons[k][3] for k in order], dtype=np.float32)
rest_np = np.linalg.norm(p0[ci_np] - p0[cj_np], axis=1).astype(np.float32)

# Sit on the floor; a hair of jitter breaks the icosahedral symmetry so the
# buckling pattern is not a perfect, fake-looking pentagon.
rng = np.random.default_rng(11)
p0 = p0 + np.array((0.0, FLOOR_Y + RADIUS, 0.0), dtype=np.float32) \
        + rng.uniform(-5e-4, 5e-4, p0.shape).astype(np.float32)

wp.init()
device = wp.get_preferred_device()
print(f"hydraulic press: {n_verts} particles (2 skins), {len(faces)} faces, "
      f"{n_cons} constraints in {n_colors} colours on {device}")

x = wp.array(p0, dtype=wp.vec3, device=device)
prev = wp.array(p0, dtype=wp.vec3, device=device)
nrm = wp.zeros(n_verts, dtype=wp.vec3, device=device)
ci = wp.array(ci_np, dtype=int, device=device)
cj = wp.array(cj_np, dtype=int, device=device)
rests = wp.array(rest_np, dtype=float, device=device)
stiffs = wp.array(stiff_np, dtype=float, device=device)
yields = wp.array(yield_np, dtype=float, device=device)
tris = wp.array(faces.reshape(-1), dtype=int, device=device)
n_corners = faces.size
soup_pos = wp.zeros(n_corners, dtype=wp.vec3, device=device)
soup_nrm = wp.zeros(n_corners, dtype=wp.vec3, device=device)
plate_ys = wp.zeros(SUBSTEPS, dtype=float, device=device)

sim_time = 0.0
prof = {"sim": 0.0, "copy": 0.0, "render": 0.0, "n": 0}


def frame_launches():
    """Every launch of one 60 fps frame; captured into a CUDA graph below."""
    for s in range(SUBSTEPS):
        # Integrate in place: the Gauss-Seidel sweeps write pos directly.
        wp.launch(integrate, dim=n_verts, device=device,
                  inputs=[x, prev, x, DT, DAMPING, GRAVITY])
        for _ in range(ITERATIONS):
            for c in range(n_colors):
                wp.launch(solve_color, dim=int(color_count[c]), device=device,
                          inputs=[x, ci, cj, rests, stiffs, int(color_start[c])])
            wp.launch(project, dim=n_verts, device=device, inputs=[x, plate_ys, s])
        wp.launch(contacts, dim=n_verts, device=device, inputs=[x, prev, plate_ys, s])
        wp.launch(plastic, dim=n_cons, device=device,
                  inputs=[x, ci, cj, rests, yields, PLASTIC_RATE])
    nrm.zero_()
    wp.launch(accum_normals, dim=len(faces), device=device, inputs=[x, tris, nrm])
    wp.launch(scatter_soup, dim=n_corners, device=device,
              inputs=[x, nrm, tris, soup_pos, soup_nrm])


graph = None
if device.is_cuda:
    with wp.ScopedCapture(device) as cap:
        frame_launches()
    graph = cap.graph


def step_frame():
    """Advance one 60 fps frame of simulation and refresh the geometry."""
    global sim_time
    heights = np.array([plate_height(sim_time + (s + 1) * DT) for s in range(SUBSTEPS)],
                       dtype=np.float32)
    plate_ys.assign(heights)
    sim_time += SUBSTEPS * DT
    t0 = time.perf_counter()
    if graph is not None:
        wp.capture_launch(graph)
    else:
        frame_launches()
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    geometry.update_attribute("position", soup_pos.numpy())
    geometry.update_attribute("normal", soup_nrm.numpy())
    t2 = time.perf_counter()
    prof["sim"] += t1 - t0
    prof["copy"] += t2 - t1
    prof["n"] += 1
    # The press hardware follows the same curve the collider used.
    py = float(heights[-1])
    plate.position.y = py + 0.5 * PLATE_T
    rod.position.y = py + PLATE_T + 0.5 * ROD_L


# --- threepp scene ----------------------------------------------------------

VULKAN = not GL and tp.vulkan_available()
if not GL and not VULKAN:
    print("vulkan not available on this machine; falling back to OpenGL")

headless = SHOT or VIDEO > 0
W, H = parse_size(cli_arg("--size", "1280x1024", str))
canvas = tp.Canvas("threepp x warp - hydraulic press", width=W, height=H,
                   antialiasing=4, vsync=False, headless=headless)
if VULKAN:
    renderer = tp.VulkanRenderer(canvas)
else:
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = True
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 1.15

scene = tp.Scene()
scene.background = 0x000000

camera = tp.PerspectiveCamera(42, canvas.aspect(), 0.05, 60)
CAM_R, CAM_Y, CAM_TARGET = 0.5 * ROOM + 3.0, 1.45, (0.0, 1.0, 0.0)
SPIN = cli_arg("--spin", 0.0, float)       # shot/video: orbit the camera, deg/frame
START_ANGLE = cli_arg("--angle", 0.0, float)  # orbit start angle, degrees


def orbit(frame):
    """Place the camera on its orbit for `frame` (static at SPIN == 0)."""
    a = math.radians(START_ANGLE + SPIN * frame)
    camera.position.set(CAM_R * math.sin(a), CAM_Y, CAM_R * math.cos(a))
    camera.look_at(*CAM_TARGET)


orbit(0)


def wall(color, roughness=0.95):
    return standard_material(color, roughness)


S = ROOM
white = tp.Color(0.73, 0.73, 0.73)
floor = tp.Mesh(tp.PlaneGeometry(S, S), wall(white))
floor.rotation.x = -math.pi / 2
floor.receive_shadow = True
scene.add(floor)

ceiling = tp.Mesh(tp.PlaneGeometry(S, S), wall(white))
ceiling.rotation.x = math.pi / 2
ceiling.position.y = S
scene.add(ceiling)

back = tp.Mesh(tp.PlaneGeometry(S, S), wall(white))
back.position.set(0.0, S / 2, -S / 2)
scene.add(back)

left = tp.Mesh(tp.PlaneGeometry(S, S), wall(tp.Color(0.65, 0.05, 0.05)))
left.rotation.y = math.pi / 2
left.position.set(-S / 2, S / 2, 0.0)
scene.add(left)

right = tp.Mesh(tp.PlaneGeometry(S, S), wall(tp.Color(0.12, 0.45, 0.15)))
right.rotation.y = -math.pi / 2
right.position.set(S / 2, S / 2, 0.0)
scene.add(right)

# Emissive ceiling panel: the area light. On Vulkan it is the light source
# (NEE samples it); GL gets a point light in the same place as a stand-in.
panel = tp.Mesh(tp.PlaneGeometry(0.5 * S, 0.5 * S),
                standard_material(0xffffff, 1.0, emissive=tp.Color(1.0, 1.0, 1.0),
                                  emissive_intensity=18.0))
panel.rotation.x = math.pi / 2
panel.position.set(0.0, S - 0.01, 0.0)
scene.add(panel)
if not VULKAN:
    lamp = tp.PointLight(0xffffff, 40.0)
    lamp.position.set(0.0, S - 0.2, 0.0)
    lamp.cast_shadow = True
    scene.add(lamp)
    scene.add(tp.AmbientLight(0xffffff, 0.15))

# The sphere: non-indexed triangle soup; the scatter kernel writes per-corner
# positions and smooth accumulated normals in face order.
geometry = tp.BufferGeometry()
geometry.set_attribute("position", p0[faces.reshape(-1)])
geometry.set_attribute("normal", verts0[faces.reshape(-1)])
ball = tp.Mesh(geometry, standard_material(tp.Color(0.97, 0.97, 0.97), 0.03, 1.0))
ball.cast_shadow = True
ball.frustum_culled = False     # positions change under the renderer's feet
scene.add(ball)

# The press. An H-frame shop press: bed plate the sphere sits on, two
# uprights, a crossbeam carrying the hydraulic cylinder, and the ram
# telescoping out of a sleeve under it. Painted frame, bare-steel tooling.
paint = standard_material(tp.Color(0.30, 0.32, 0.36), 0.45, 0.5)
steel = standard_material(tp.Color(0.22, 0.23, 0.25), 0.38, 1.0)


def part(geom, mat, y, x=0.0, z=0.0):
    m = tp.Mesh(geom, mat)
    m.position.set(x, y, z)
    m.cast_shadow = True
    m.receive_shadow = True
    scene.add(m)
    return m


BEAM_Y0 = FLOOR_Y + 2.36                      # crossbeam underside
part(tp.BoxGeometry(2.4, BED_T, 1.5), steel, 0.5 * BED_T)                  # bed
for sx in (-1.0, 1.0):                                                      # uprights
    part(tp.BoxGeometry(0.18, BEAM_Y0 + 0.24 - FLOOR_Y, 0.18), paint,
         0.5 * (FLOOR_Y + BEAM_Y0 + 0.24), x=sx * 1.11)
part(tp.BoxGeometry(2.4, 0.24, 0.32), paint, BEAM_Y0 + 0.12)               # crossbeam
part(tp.CylinderGeometry(0.21, 0.21, 0.45, 48), paint, BEAM_Y0 - 0.225)    # cylinder housing
part(tp.CylinderGeometry(0.11, 0.11, 0.33, 48), steel, BEAM_Y0 - 0.45 - 0.165)  # sleeve
part(tp.CylinderGeometry(0.15, 0.15, 0.28, 48), paint, BEAM_Y0 + 0.24 + 0.14)   # cap
plate = part(tp.CylinderGeometry(PLATE_R, PLATE_R, PLATE_T, 96), steel, PLATE_TOP + 0.5 * PLATE_T)
ROD_L = 1.2                    # top always hidden inside sleeve/housing/cap
rod = part(tp.CylinderGeometry(0.075, 0.075, ROD_L, 48), steel,
           PLATE_TOP + PLATE_T + 0.5 * ROD_L)


def save(path):
    if VULKAN:
        renderer.save_frame(scene, camera, path)   # renders + reads back
    else:
        renderer.render(scene, camera)
        renderer.save_frame(path)


if SHOT:
    frames = int(round(SHOT_T * 60))
    t0 = time.perf_counter()
    FREEZE = "--freeze" in sys.argv   # no sim; camera only
    for f in range(frames):
        orbit(f)
        if not FREEZE:
            step_frame()
        tr = time.perf_counter()
        renderer.render(scene, camera)   # keep the temporal history honest
        prof["render"] += time.perf_counter() - tr
        if f % 30 == 0:
            xs = x.numpy()
            print(f"  t={sim_time:4.2f}  plate={plate_height(sim_time):.3f}  "
                  f"shell h={xs[:, 1].max() - xs[:, 1].min():.3f}  "
                  f"w={xs[:, 0].max() - xs[:, 0].min():.3f}", flush=True)
    if SPIN == 0.0:
        # Let the accumulation settle on the final pose before the read-back.
        for _ in range(24):
            renderer.render(scene, camera)
    else:
        orbit(frames)                    # keep moving into the saved frame
    out = cli_arg("--out", "warp_hydraulic_press.png", str)
    save(out)
    print(f"simulated {SHOT_T:.1f} s ({frames} frames) in "
          f"{time.perf_counter() - t0:.1f}s, wrote {out}")
    n = max(prof["n"], 1)
    print(f"  per frame: sim {1e3 * prof['sim'] / n:.1f} ms, copy "
          f"{1e3 * prof['copy'] / n:.1f} ms, render {1e3 * prof['render'] / n:.1f} ms")
elif VIDEO > 0:
    outdir = tempfile.mkdtemp(prefix="warp_press_frames_")
    total = int(round(VIDEO * 60))
    t0 = time.perf_counter()
    for _ in range(30):                  # warm the temporal history on frame 0
        renderer.render(scene, camera)
    for k in range(total):
        orbit(k)
        step_frame()
        save(os.path.join(outdir, f"f{k:05d}.png"))
        if k % 60 == 0:
            print(f"  frame {k}/{total}  ({time.perf_counter() - t0:.0f}s elapsed)",
                  flush=True)
    print(f"rendered {total} frames in {time.perf_counter() - t0:.0f}s -> {outdir}")
    ff = find_ffmpeg()
    if ff:
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "60",
                        "-i", os.path.join(outdir, "f%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
                        "warp_hydraulic_press.mp4"], check=False)
        print("wrote warp_hydraulic_press.mp4")
else:
    orbit_loop(canvas, renderer, scene, camera, step_frame, target=(0.0, 1.0, 0.0))
