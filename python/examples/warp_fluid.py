"""A liquid surface simulated with NVIDIA Warp and rendered by threepp.

Position Based Fluids on the GPU, surfaced with GPU marching cubes, drawn as a
refracting water mesh. A dam break collapses into a basin, then a paddle sweeps
back and forth through it so the water never settles into a still pond.

Nothing crosses host memory in the render path, on either backend: the
marching-cubes triangles and their normals are written straight into the
renderer's own vertex buffers -- GLRenderer.gl_buffer_id + wp.RegisteredGLBuffer
on OpenGL, VulkanRenderer.enable_vertex_interop + threepp.cuda_interop (a CUDA
import of the Vulkan allocation) on Vulkan. Both backends publish the live
triangle count with set_draw_range: the raster draw, the BLAS build and the
interop copies all clamp to it, so capacity beyond the live surface costs
nothing per frame.

    pip install warp-lang
    python warp_fluid.py                 # window; drag to orbit, Esc quits
    python warp_fluid.py --n 400000      # more particles (a taller column)
    python warp_fluid.py --scale 2       # 8x particles at half the spacing, same tank
    python warp_fluid.py --shot 6        # headless PNG at t=6s
    python warp_fluid.py --bench         # timed phase breakdown
    python warp_fluid.py --vulkan        # Vulkan renderer (RT reflections)
    python warp_fluid.py --vulkan --no-interop   # ... through pinned host memory
    python warp_fluid.py --points        # draw particles instead of the surface
    python warp_fluid.py --probe 3       # print an energy audit, no window
    python warp_fluid.py --iters 4 --rho 0   # plain Jacobi, no Chebyshev acceleration
    python warp_fluid.py --obstacle part.stl # collide an arbitrary mesh, not the box
    python warp_fluid.py --obstacle part.stl --obstacle-height 0.35 --sdf-res 128
    python warp_fluid.py --dump out/s2 --scale 2 --seconds 8   # simulate only, positions to disk
    python warp_fluid.py --replay out/s2 --vulkan --video 8    # render a dump, no simulation

--dump runs the simulation with no window and no renderer and writes one file
per frame into the directory: positions quantised to 16 bits against the
frame's own bounding box (0.03 mm over the tank, against a spacing of
millimetres) plus a meta.json with the resolution and step settings. --replay
reads them back in place of sim_step, so every render mode (window, --shot,
--video, --points) works unchanged on a machine that never ran the fluid. The
split exists because the two halves want different hardware: the simulation
scales with GPU memory bandwidth (an H100 holds tens of millions of
particles), while the Vulkan renderer needs VK_KHR_ray_query, which datacenter
GPUs do not expose. Velocities are not dumped, so --points replays with a flat
colour. A replay takes --scale, --n and --obstacle from meta.json.

Needs a CUDA device: the zero-copy surface path is CUDA/OpenGL interop.

--obstacle takes any model threepp's ModelLoader reads (.stl, .obj, .gltf/.glb,
.dae) and collides the fluid against a signed distance field baked from it once
at startup, in place of the analytic paddle box. A STEP/IGES solid has no
triangles of its own, so tessellate it first -- FreeCAD, OpenCascade, CAD
Assistant -- and pass the result. The mesh is auto-fitted (uniform scale, centred
where the box was, resting on the floor), because CAD arrives in millimetres as
often as metres. It sweeps like the paddle did, so the tank keeps circulating.

The mesh needs to be CLOSED for the sign to mean anything: the bake asks for a
winding number, and an open shell has none. An unclosed mesh still repels
particles at its surface, so it looks plausible while being an unsigned field --
check the "closed" note the bake prints.
"""
import atexit
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import (DensitySurface, Encoder, bench_loop, cli_arg,
                         open_display, parse_size, pbf_constants,
                         resize_handler, sky_env, standard_material,
                         write_radiance_hdr)
try:
    from threepp.cuda_interop import VkInteropArray
except ImportError:                  # older threepp builds have no CUDA<->Vulkan interop
    VkInteropArray = None

# Warp compiles with fast_math off, so every sqrt and divide in the neighbour
# loops is a multi-instruction IEEE sequence instead of one hardware op. Nothing
# in the fluid is subnormal-sensitive (the guards sit at 1e-9 and 1e-12), so the
# approximate forms are safe. Module-scoped: the surfacing kernels are untouched.
wp.set_module_options({"fast_math": True})

# Resolution multiplier. --scale k divides the particle spacing by k and, unless
# --n overrides it, multiplies the count by k^3, so the tank, the dam-break
# column and the paddle keep their geometry and the fluid is simply finer.
# Raising --n alone does something different: the fill is footprint-first
# (nx, nz from the tank, ny = N / (nx nz)), so extra particles at the same
# spacing stack into a TALLER column -- 3M at the default spacing is a 2.5 m
# head of water in a 1.1 m tank, which three Jacobi iterations cannot hold
# (measured on an H100: 150% mean compression, |v| peaking at 440 m/s).
SCALE = cli_arg("--scale", 1.0, float)

# --- the WIDE pool ------------------------------------------------------------
# --scale k refines the SAME tank: D shrinks and the surfacing grid is
# domain/CELL in each axis, so cost grows as k^3 and an 80 GB card runs out of
# int32 in the marching cubes (715,827,882 nodes) before it runs out of memory.
# --wide W does the other thing: it enlarges the tank FOOTPRINT at fixed D.
# Particles grow with AREA and the surfacing grid grows in two of its three
# axes, which is the only direction along which a big GPU buys a visibly bigger
# scene rather than a bigger number in a log. At equal bytes the difference is
# not close: 78 GB of widening is 834 m2 of water at 9 mm; 78 GB of refining is
# 2.31 m2.
#
# And it is the SAFE direction. The comment above about 3M particles boiling is
# about DEPTH, not count: --n is footprint-first, so 3M at D = 9 mm in a 1.1 m
# tank is a 2.5 m tower. PBF's density constraint is local (H = 2D spans one
# particle layer), so the residual follows the hydrostatic column. A pool a
# hundred times wider at the SAME depth is no harder to solve than this one.
WIDE = cli_arg("--wide", 1.0, float)        # footprint multiplier, both axes
WIDE_X = cli_arg("--wide-x", WIDE, float)   # per-axis overrides; a canal is
WIDE_Z = cli_arg("--wide-z", WIDE, float)   #   --wide-x 16 --wide-z 1
W_AREA = WIDE_X * WIDE_Z                    # particles, triangles, surface area
W_LIN = math.sqrt(W_AREA)                   # camera and sun distance
# --fill flat fills the whole footprint to --depth instead of damming a column
# into 38% of it. A dam break is a 2 m event: at --wide 16 the front travels
# 2*sqrt(g*h) = 3.3 m/s and needs 20 s to cross 33.6 m, and long before it
# arrives the tongue is thinner than the splat width, where water does not thin
# -- it vanishes. A pool that is already full, driven by a full-width piston,
# is the scene that survives being 34 m long.
FLAT = cli_arg("--fill", "dam", str).lower() == "flat"
DEPTH = cli_arg("--depth", 0.27, float)     # still water depth (m) under --fill flat
# IS_REF is the promise that nothing changed for anyone who passes no new flag.
# It gates the hash-grid dims, the camera, the look and the surfacing ceiling
# back to their shipped values -- bit-exactly, because bucket assignment fixes
# the neighbour iteration order and therefore the floating-point summation
# order in every kernel. Do not tidy these branches away: they are what keeps
# every previously recorded dump replaying to the same pixels.
IS_REF = (WIDE_X == 1.0 and WIDE_Z == 1.0 and not FLAT)
# WHICH END THE CAMERA STANDS AT, and it decides the whole shot.
# The piston makes a wave that LEAVES at sqrt(g*h) = 1.63 m/s. Standing at the
# piston end (--cam-from near, the first design) the camera watches the event
# depart: over a 14 s run the front travels ~23 m away and the near field goes
# glassy, which is exactly what the first 33.6 m film looked like. Standing at
# the FAR end instead, the front approaches, grows and arrives during the shot,
# and the crossing time becomes the drama instead of the problem.
# --cam-from all is the third option and the default for a wide run: pull back
# off the +x end and elevate until the ENTIRE pool is inside the frame. A low
# eye at the waterline cannot ever show a pool as a pool -- the far end
# compresses to a line and the side walls fall outside the lens -- so if you
# want the footprint legible you have to pay for it with height.
CAM_FROM = cli_arg("--cam-from", "near" if IS_REF else "all", str).lower()
CAM_ALL = CAM_FROM == "all"
CAM_FAR_END = CAM_FROM in ("far", "all")

N = cli_arg("--n", int(round(340_000 * SCALE ** 3 * W_AREA)), int)
# Simulate-only / render-only split (see the module docstring). A replay
# reproduces the dump's fill exactly -- same --scale, same requested --n, same
# obstacle -- because N is baked into the buffers the render paths address.
DUMP = cli_arg("--dump", "", str)
DUMP_SECONDS = cli_arg("--seconds", 6.0, float)
REPLAY = cli_arg("--replay", "", str)
_meta = None
if REPLAY:
    with open(os.path.join(REPLAY, "meta.json")) as _f:
        _meta = json.load(_f)
    SCALE = float(_meta["scale"])
    N = int(_meta["n_requested"])
N_REQUESTED = N
BENCH = "--bench" in sys.argv
SHOT = "--shot" in sys.argv
SHOT_TIME = cli_arg("--shot", 6.0, float)
POINTS = "--points" in sys.argv
PROBE = "--probe" in sys.argv
# Vulkan gets ray-traced reflections and a dedicated water/glass shader. Its zero
# copy is enable_vertex_interop: the renderer exports its own position/normal
# allocations, CUDA imports them, and the expand kernel writes them in place.
# The live triangle count is published with set_draw_range exactly as on GL.
VULKAN = "--vulkan" in sys.argv
# --no-interop forces the pinned-host round trip on Vulkan. It is also the
# automatic fallback when the export or the CUDA import does not come up.
INTEROP = "--no-interop" not in sys.argv
# --frames N times the real interactive frame model (canvas.animate_once).
# --bench is headless, where a Vulkan render() call runs several full GPU
# frames, so its render figure is not what a window costs.
FRAMES = cli_arg("--frames", 0, int)
# --video S renders S seconds at 60 fps to warp_fluid.mp4 (or PNG frames if
# ffmpeg is absent). Offline: every sim frame is rendered, so the temporal
# pipeline stays converged; works headless.
VIDEO = cli_arg("--video", 0.0, float)
# --egl forces the display-less EGL context (a cluster GPU node); --no-egl
# forces a window. Default: EGL when Linux has no DISPLAY and one is available,
# so a Slurm job needs no flag and a laptop is unaffected.
EGL = True if "--egl" in sys.argv else (False if "--no-egl" in sys.argv else None)
OPAQUE = "--opaque" in sys.argv   # debug: render the surface as a plain lit
                                  # solid, to separate a geometry problem
                                  # from a shading one

# --- fluid parameters ---------------------------------------------------------

D = 0.009 / SCALE            # rest particle spacing (m); the single
                             # resolution knob -- H, WALL_EPS, MAX_DP,
                             # rest density, CELL and the iso level all
                             # derive from it
H = 2.0 * D                  # SPH support radius
MASS = 1.0                   # unit mass; rest density is measured from a lattice
DT = 1.0 / 60.0
# A finer spacing needs proportionally shorter substeps: the per-substep travel
# at the tank's ~2-3 m/s must stay around one spacing, or particles skip whole
# neighbour shells between projections. Default scales with --scale.
SUBSTEPS = cli_arg("--substeps", max(3 if FLAT else 2, int(round(2 * SCALE))), int)
# A flat pool gets K = 9 (3 substeps x 3 iterations), not 6. Measured on the
# 4070: --wide 8 settles at 8.32% mean compression with K = 9, and --wide 4 sits
# at 18.15% with K = 6. The first Idun film ran the K = 6 default.
ITERATIONS = cli_arg("--iters", 3, int)               # density-constraint projections per substep
# Chebyshev acceleration of the Jacobi projection (Wang 2015). RHO estimates
# the spectral radius of the iteration; 0 disables it (plain Jacobi).
# Iteration 1 runs plain, then omega_2 = 2/(2-rho^2), omega_k = 4/(4 - rho^2 omega_{k-1}).
# Two guards in solve_delta keep it stable on a free surface: particles with
# lambda == 0 are never extrapolated, and the total accelerated step is
# re-clamped to MAX_DP.
RHO_CHEB = cli_arg("--rho", 0.8, float)
OMEGAS = [1.0]
for _k in range(1, max(ITERATIONS, 1)):
    _w = (2.0 / (2.0 - RHO_CHEB ** 2) if _k == 1
          else 4.0 / (4.0 - RHO_CHEB ** 2 * OMEGAS[-1]))
    OMEGAS.append(_w)
S_CORR_N = 4.0               # artificial-pressure exponent (solve_delta spells
                             # the power out as two multiplies)
S_CORR_DQ = 0.20 * H
JACOBI_RELAX = 0.4           # Jacobi projection over-corrects without this
XSPH_C = cli_arg("--xsph", 0.08, float)          # viscosity: how much a particle adopts neighbour flow
VORTICITY = cli_arg("--vorticity", 0.22, float)  # curl restored after projection damps it
V_MAX = 5.0                  # velocity clamp
MAX_DP = 0.35 * D            # per-iteration position-correction bound
GRAVITY = -9.81

# --- tank geometry ------------------------------------------------------------

X0, X1 = -1.05 * WIDE_X, 1.05 * WIDE_X   # tank interior; --wide scales the
Z0, Z1 = -0.55 * WIDE_Z, 0.55 * WIDE_Z   # FOOTPRINT at fixed spacing D
FLOOR = 0.0
WALL_EPS = 0.6 * D           # keep particle centres this far off a wall

# A sloping shore at the +x end. A closed 34 m box is a bathtub: the wave train
# reflects off a vertical wall with no loss and stands, and the whole pool
# turns into one 41 s seiche. A beach shoals the train, breaks it, and gives
# the far end of the shot something that reads as surf rather than as a wall.
BEACH_RUN = cli_arg("--beach", 0.0, float)            # metres of shore at +x
BEACH_SLOPE = (DEPTH / BEACH_RUN) if BEACH_RUN > 0.0 else 0.0
BEACH_TOE = X1 - max(BEACH_RUN, 1.0e-6)

# A block on the tank floor for the jet to break over. It must be a FINITE box:
# an unbounded half-space would teleport every particle beneath it onto its
# surface.
if FLAT:
    # One 0.27 m paddle cannot stir 591 m2. Under a flat fill the box becomes a
    # full-width piston at the -x end: the whole tank width moves, so the wave
    # it makes is planar and crosses the pool instead of dispersing.
    BX0, BX1 = X0 + 0.10, X0 + 0.55
    BZ0, BZ1 = Z0, Z1
    BY1 = DEPTH + 0.30
else:
    BX0, BX1 = 0.15 * WIDE_X, 0.42 * WIDE_X
    BZ0, BZ1 = -0.30 * WIDE_Z, 0.30 * WIDE_Z
    BY1 = 0.26          # a HEIGHT, not a footprint: --wide does not deepen the tank

# --obstacle PATH swaps that box for an arbitrary triangle mesh, collided
# through a signed distance field baked once at startup. The tank walls stay
# analytic in both modes -- they are six planes and a clamp is unbeatable; only
# the thing in the middle of the tank is worth a field.
#
# An SDF rather than a per-particle mesh query, because collide() runs
# (1 + ITERATIONS) times per particle per substep: a BVH descent per call is a
# real cost, a trilinear volume fetch is not. The bake pays the mesh query once
# per voxel instead, which is why it can afford the robust winding-number sign.
#
# The mesh arrives through threepp's own ModelLoader and is read back out of the
# BufferGeometry it produced, so .stl/.obj/.gltf/.glb/.dae all work and this file
# carries no parser. Millimetres are the CAD norm and metres are the tank's, so
# the mesh is fitted rather than trusted to arrive in the right units.
OBSTACLE = cli_arg("--obstacle", "", str)
OBSTACLE_H = cli_arg("--obstacle-height", 0.0, float)  # 0 = fit the box's envelope
SDF_RES = cli_arg("--sdf-res", 64, int)                # voxels on the long axis
if REPLAY and _meta.get("obstacle"):
    # The obstacle decides which lattice particles are dropped at seeding, so
    # the replay must bake the same field; it is also what the picture shows.
    if not OBSTACLE:
        OBSTACLE = _meta["obstacle"]
        OBSTACLE_H = float(_meta["obstacle_height"])
        SDF_RES = int(_meta["sdf_res"])
    if not os.path.isfile(OBSTACLE):
        raise SystemExit(f"--replay: the dump was made with --obstacle {_meta['obstacle']}; "
                         f"pass the same file (not found: {OBSTACLE})")

# The dam-break column occupies [X0, FILL_X1] and collapses in the first second.
# After that the paddle keeps the water moving.
FILL_X1 = -0.25 * WIDE_X
if FLAT:
    # Amplitude and period must scale together or not at all: their ratio sets
    # the wave height and steepness, which is a property of the water, not of
    # how much of it there is.
    PADDLE_AMP = cli_arg("--paddle-amp", 0.14, float)
    PADDLE_PERIOD = cli_arg("--paddle-period", 2.4, float)
else:
    PADDLE_AMP = cli_arg("--paddle-amp", 0.46 * WIDE_X, float)
    PADDLE_PERIOD = cli_arg("--paddle-period", 2.0 * WIDE_X, float)
# A piston that starts at full stroke puts a step into shallow water. Ramp it.
PADDLE_RAMP = cli_arg("--paddle-ramp", 1.0 if FLAT else 0.0, float)

# --- SPH kernels ----------------------------------------------------------------

PBF = pbf_constants(D, H, MASS, S_CORR_DQ, S_CORR_N)
POLY6, SPIKY = PBF["poly6"], PBF["spiky"]
RHO0, W_DQ, SG2_REST = PBF["rho0"], PBF["w_dq"], PBF["sum_grad2"]
EPS_CFM, S_CORR_K = PBF["eps_cfm"], PBF["s_corr_k"]


@wp.func
def w_poly6(r2: float) -> float:
    d = H * H - r2
    if d <= 0.0:
        return 0.0
    return POLY6 * d * d * d


@wp.func
def w_spiky_grad(rv: wp.vec3, r: float) -> wp.vec3:
    if r <= 1.0e-9 or r >= H:
        return wp.vec3(0.0, 0.0, 0.0)
    return rv * (SPIKY * (H - r) * (H - r) / r)


@wp.func
def collide(p: wp.vec3, bx0: float, bx1: float, vol: wp.uint64, ox: float,
            blo: wp.vec3, bhi: wp.vec3) -> wp.vec3:
    # The tank is six planes, so a clamp is the whole of it in both modes.
    x = wp.min(wp.max(p[0], X0 + WALL_EPS), X1 - WALL_EPS)
    z = wp.min(wp.max(p[2], Z0 + WALL_EPS), Z1 - WALL_EPS)
    y = wp.max(p[1], FLOOR + WALL_EPS)
    if BEACH_SLOPE > 0.0:
        # The shore is a plane, so it is a clamp like every other wall here.
        y = wp.max(y, FLOOR + WALL_EPS + (x - BEACH_TOE) * BEACH_SLOPE)
    if vol != wp.uint64(0):
        # --obstacle: one trilinear fetch yields distance AND gradient. The field
        # is baked in the mesh's REST frame, so the paddle sweep is undone on the
        # query point rather than re-baked -- a rigid translation is a subtract.
        # vol == 0 is the box path below; the branch is uniform across the launch,
        # so it costs no divergence.
        q = wp.vec3(x - ox, y, z)
        # Reject against the field's own bounds before sampling it: a NanoVDB
        # fetch is a sparse-tree descent, not a texture read, and most particles
        # are nowhere near the obstacle. Outside the grid the sample returns
        # bg_value anyway, so the reject changes no result.
        if (q[0] > blo[0] and q[0] < bhi[0] and q[1] < bhi[1]
                and q[2] > blo[2] and q[2] < bhi[2]):
            g = wp.vec3()
            d = wp.volume_sample_grad_f(vol, wp.volume_world_to_index(vol, q),
                                        wp.Volume.LINEAR, g)
            if d < WALL_EPS:
                gl = wp.length(g)
                if gl > 1.0e-8:
                    # Push out along the surface normal to the same standoff the
                    # box path keeps. The gradient comes back in INDEX space, but
                    # the voxels are cubic, so normalising recovers the world
                    # direction.
                    q = q + g * ((WALL_EPS - d) / gl)
                    # Re-clamp to the tank. A particle wedged between the mesh and
                    # a wall would otherwise be pushed straight through the wall,
                    # and leaked fluid never comes back; ending up slightly inside
                    # the obstacle is recoverable, because the next iteration
                    # pushes out again. Deliberately the lesser of two failures.
                    x = wp.min(wp.max(q[0] + ox, X0 + WALL_EPS), X1 - WALL_EPS)
                    y = wp.max(q[1], FLOOR + WALL_EPS)
                    z = wp.min(wp.max(q[2], Z0 + WALL_EPS), Z1 - WALL_EPS)
        return wp.vec3(x, y, z)
    # obstacle box: if inside, eject through the face of least penetration
    ax0 = bx0 - WALL_EPS
    ax1 = bx1 + WALL_EPS
    az0 = BZ0 - WALL_EPS
    az1 = BZ1 + WALL_EPS
    ay1 = BY1 + WALL_EPS
    if x > ax0 and x < ax1 and z > az0 and z < az1 and y < ay1:
        dxm = x - ax0
        dxp = ax1 - x
        dzm = z - az0
        dzp = az1 - z
        dyp = ay1 - y
        m = wp.min(wp.min(wp.min(dxm, dxp), wp.min(dzm, dzp)), dyp)
        if m == dyp:
            y = ay1
        elif m == dxm:
            x = ax0
        elif m == dxp:
            x = ax1
        elif m == dzm:
            z = az0
        else:
            z = az1
    return wp.vec3(x, y, z)


@wp.kernel
def predict(x: wp.array(dtype=wp.vec3),
            v: wp.array(dtype=wp.vec3),
            xp: wp.array(dtype=wp.vec3),
            dt: float, bx0: float, bx1: float, vol: wp.uint64, ox: float,
            blo: wp.vec3, bhi: wp.vec3):
    i = wp.tid()
    vi = v[i] + wp.vec3(0.0, GRAVITY, 0.0) * dt
    sp = wp.length(vi)
    if sp > V_MAX:
        vi = vi * (V_MAX / sp)
    v[i] = vi
    xp[i] = collide(x[i] + vi * dt, bx0, bx1, vol, ox, blo, bhi)


@wp.kernel
def solve_lambda(xp: wp.array(dtype=wp.vec3),
                 grid: wp.uint64,
                 rho0: float,
                 eps_cfm: float,
                 lam: wp.array(dtype=float)):
    # spatially-sorted thread order: neighbour gathers hit cache
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0:
        return
    p = xp[i]
    rho = float(0.0)
    grad_i = wp.vec3(0.0, 0.0, 0.0)
    sum_grad2 = float(0.0)
    q = wp.hash_grid_query(grid, p, H)
    for j in q:
        rv = p - xp[j]
        r2 = wp.dot(rv, rv)
        if r2 < H * H:
            rho += MASS * w_poly6(r2)
            g = w_spiky_grad(rv, wp.sqrt(r2)) * (MASS / rho0)
            grad_i += g
            sum_grad2 += wp.dot(g, g)
    sum_grad2 += wp.dot(grad_i, grad_i)
    # Resist compression only. A free-surface particle is under-dense by
    # definition; letting it produce a negative constraint gives it a large
    # lambda whose neighbour gradients are all one-sided (nothing cancels), and
    # the surface implodes. Cohesion is s_corr's job.
    c = wp.max(rho / rho0 - 1.0, 0.0)
    lam[i] = -c / (sum_grad2 + eps_cfm)


@wp.kernel
def solve_delta(xp: wp.array(dtype=wp.vec3),
                lam: wp.array(dtype=float),
                grid: wp.uint64,
                rho0: float,
                w_dq: float,
                k_corr: float,
                relax: float,
                omega: float,
                prev: wp.array(dtype=wp.vec3),
                bx0: float,
                bx1: float,
                vol: wp.uint64,
                ox: float,
                blo: wp.vec3,
                bhi: wp.vec3,
                out: wp.array(dtype=wp.vec3)):
    # spatially-sorted thread order: neighbour gathers hit cache
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0:
        return
    p = xp[i]
    li = lam[i]
    w_dq_inv = 1.0 / w_dq          # hoisted: one divide per thread, not per neighbour
    dp = wp.vec3(0.0, 0.0, 0.0)
    q = wp.hash_grid_query(grid, p, H)
    for j in q:
        rv = p - xp[j]
        r2 = wp.dot(rv, rv)
        if r2 < H * H and r2 > 1.0e-12:
            # Artificial pressure repels near-coincident particles, which is
            # what keeps droplets round instead of stringy. S_CORR_N == 4, so
            # the power is two multiplies in the hottest loop of the sim.
            qq = w_poly6(r2) * w_dq_inv
            q2 = qq * qq
            s_corr = -k_corr * (q2 * q2)
            dp += w_spiky_grad(rv, wp.sqrt(r2)) * (li + lam[j] + s_corr)
    d = dp * (MASS / rho0 * relax)
    dl = wp.length(d)
    if dl > MAX_DP:
        d = d * (MAX_DP / dl)
    # Chebyshev semi-iteration (Wang 2015): extrapolate the relaxed Jacobi
    # result through the PREVIOUS iterate. omega == 1 is plain Jacobi, kept as
    # its own branch. collide() runs after the extrapolation so momentum never
    # carries a particle through a wall.
    if omega == 1.0 or li == 0.0:
        # li == 0 means a free-surface particle: its one-sided constraint is
        # satisfied, so it has nothing to converge to and the extrapolation
        # below would be a pure kinetic kick into the surface.
        out[i] = collide(p + d, bx0, bx1, vol, ox, blo, bhi)
    else:
        # Re-clamp the TOTAL move (Jacobi step + extrapolation) to MAX_DP so an
        # accelerated iterate never travels farther than plain Jacobi allows;
        # the acceleration survives as a better direction, not a bigger step.
        e = prev[i] + (p + d - prev[i]) * omega - p
        el = wp.length(e)
        if el > MAX_DP:
            e = e * (MAX_DP / el)
        out[i] = collide(p + e, bx0, bx1, vol, ox, blo, bhi)


@wp.kernel
def finalize(x: wp.array(dtype=wp.vec3),
             xp: wp.array(dtype=wp.vec3),
             v: wp.array(dtype=wp.vec3),
             dt: float):
    i = wp.tid()
    vi = (xp[i] - x[i]) * (1.0 / dt)
    sp = wp.length(vi)
    if sp > V_MAX:
        vi = vi * (V_MAX / sp)
    v[i] = vi
    x[i] = xp[i]


@wp.kernel
def curl_and_viscosity(x: wp.array(dtype=wp.vec3),
                       v: wp.array(dtype=wp.vec3),
                       grid: wp.uint64,
                       rho0: float,
                       omega: wp.array(dtype=wp.vec3),
                       v_out: wp.array(dtype=wp.vec3)):
    # spatially-sorted thread order: neighbour gathers hit cache
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0:
        return
    p = x[i]
    vi = v[i]
    w = wp.vec3(0.0, 0.0, 0.0)
    dv = wp.vec3(0.0, 0.0, 0.0)
    q = wp.hash_grid_query(grid, p, H)
    for j in q:
        rv = p - x[j]
        r2 = wp.dot(rv, rv)
        if r2 < H * H:
            vij = v[j] - vi
            w += wp.cross(vij, w_spiky_grad(rv, wp.sqrt(r2)))
            dv += vij * w_poly6(r2)
    omega[i] = w * (MASS / rho0)
    v_out[i] = vi + dv * (XSPH_C * MASS / rho0)


@wp.kernel
def vorticity_confine(x: wp.array(dtype=wp.vec3),
                      omega: wp.array(dtype=wp.vec3),
                      v_in: wp.array(dtype=wp.vec3),
                      grid: wp.uint64,
                      dt: float,
                      v: wp.array(dtype=wp.vec3)):
    # spatially-sorted thread order: neighbour gathers hit cache
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0:
        return
    p = x[i]
    # grad|omega| points at the vortex core; the restoring force is N x omega
    eta = wp.vec3(0.0, 0.0, 0.0)
    q = wp.hash_grid_query(grid, p, H)
    for j in q:
        rv = p - x[j]
        r2 = wp.dot(rv, rv)
        if r2 < H * H and r2 > 1.0e-12:
            eta += w_spiky_grad(rv, wp.sqrt(r2)) * wp.length(omega[j])
    vi = v_in[i]
    le = wp.length(eta)
    if le > 1.0e-6:
        vi = vi + wp.cross(eta * (1.0 / le), omega[i]) * (VORTICITY * dt)
    v[i] = vi


@wp.kernel
def measure_compression(x: wp.array(dtype=wp.vec3),
                        grid: wp.uint64,
                        rho0: float,
                        err: wp.array(dtype=float)):
    # Post-solve residual of the density constraint (positive part only, the
    # same one-sided form the solver projects). This is the convergence metric
    # --probe reports, and what makes solver changes comparable.
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0:
        return
    p = x[i]
    rho = float(0.0)
    q = wp.hash_grid_query(grid, p, H)
    for j in q:
        rv = p - x[j]
        r2 = wp.dot(rv, rv)
        if r2 < H * H:
            rho += MASS * w_poly6(r2)
    err[i] = wp.max(rho / rho0 - 1.0, 0.0)


@wp.kernel
def shade_points(v: wp.array(dtype=wp.vec3), col: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    f = wp.min(wp.length(v[i]) / 2.6, 1.0)
    col[i] = wp.vec3(0.05, 0.28, 0.42) * (1.0 - f) + wp.vec3(0.92, 0.97, 1.0) * f


# --- surfacing: particles -> density grid -> marching cubes -------------------

CELL = cli_arg("--cell", 1.00 if IS_REF else 1.5, float) * D   # surface grid spacing (then blurred). Marching
                             # cubes cannot emit a sheet thinner than roughly
                             # the combined splat+blur kernel width -- thinner
                             # water does not thin, it VANISHES -- so the cell
                             # is fine and the second blur round is narrowed,
                             # at the cost of more triangles.
# That argument is the shipped dam break seen from 1.9 m, where a cell spans
# several pixels. A wide shot turns it upside down: at --wide 8 from 16 m a
# 9 mm cell is a fifth of a pixel, the thin tongue it protects cannot be seen,
# and what CELL = D actually buys is ONE particle per cell -- a density field
# made of shot noise, which the gradient normals turn into glints that re-roll
# every frame. The first Idun film was twenty seconds of that. 1.5 D holds
# ~3.4 particles per cell for 3.4x fewer nodes; --normal-blur does the rest.
GX0, GY0, GZ0 = X0 - 0.05, FLOOR - 0.035, Z0 - 0.05
# The shipped ceiling was 0.56 m with the comment "paddle spray never gets near
# this". It does: a --probe 3 run of the shipped scene measures y_max = 0.896 m
# at t = 1.20 s. A particle above the top plane deposits NO density at all
# (warp_common._splat returns early), so it vanishes from the surface and pops
# back on the way down, and marching cubes leaves the surface open up there.
# IS_REF keeps the old value so recorded dumps still replay identically;
# everything else gets a ceiling the water cannot reach.
GRID_TOP = cli_arg("--grid-top", 0.56 if IS_REF else DEPTH + 0.95, float)
NGX = int((X1 + 0.05 - GX0) / CELL) + 1
NGY = int((GRID_TOP - GY0) / CELL) + 1
NGZ = int((Z1 + 0.05 - GZ0) / CELL) + 1
if 3 * NGX * NGY * NGZ >= 2 ** 31:
    # warp's marching cubes indexes nodes as ti*ny*nz*3 + tj*nz*3 + tk*3 + side
    # in int32 and allocates wp.zeros(nx*ny*nz*3), so past 715,827,882 nodes it
    # raises a ValueError from deep inside check_array_shape. Say it here, in
    # terms of the flags that caused it.
    raise SystemExit(
        f"surfacing grid {NGX}x{NGY}x{NGZ} = {NGX * NGY * NGZ:,} nodes exceeds warp's "
        f"int32 marching-cubes limit (715,827,882). Raise --cell (nodes fall as "
        f"1/cell^3) or lower --grid-top.")

# --- the --obstacle mesh ------------------------------------------------------


def load_obstacle_mesh(path):
    """Anything ModelLoader handles -> (M, 3, 3) float32 triangles, world space.

    The triangles are read back out of the loaded BufferGeometry rather than
    re-parsed here, so every format the library already supports arrives the
    same way -- .stl, .obj, .gltf/.glb, .dae -- instead of this example growing
    a parser per format. A multi-part assembly comes through as the assembly:
    each node's world matrix is baked in, so a .glb's nested parts land where
    they belong instead of collapsing onto the origin.

    A STEP/IGES solid carries no triangles of its own -- it is NURBS and
    topology -- so tessellate it first (FreeCAD, OpenCascade, CAD Assistant)
    and pass what that writes out.
    """
    root = tp.ModelLoader().load(path)
    if root is None:
        raise SystemExit(f"--obstacle: threepp could not load {path}")
    root.update_matrix_world(True)
    parts = []

    def visit(node):
        # Meshes only. Points and Line carry a geometry too, and their
        # "triangles" would be nonsense read three vertices at a time.
        if not isinstance(node, tp.Mesh):
            return
        pos = node.geometry.get_attribute("position")
        if pos is None or len(pos) < 3:
            return
        idx = node.geometry.get_index()
        v = pos if idx is None else pos[idx]          # STL is already a soup
        m = node.matrix_world.to_numpy()
        parts.append((v @ m[:3, :3].T + m[:3, 3]).reshape(-1, 3, 3))

    root.traverse(visit)
    if not parts:
        raise SystemExit(f"--obstacle: {path} loaded but carries no triangles")
    return np.concatenate(parts).astype(np.float32)


def fit_obstacle(tris, target_h):
    """Scale/place a mesh so it sits on the tank floor where the box was.

    CAD arrives in millimetres about as often as metres, and a STEP export is
    positioned wherever the assembly origin happened to be -- so fit rather than
    trust. Uniform scale, always: an SDF is only a distance field under a rigid
    plus uniform map, so anisotropic scaling would make every baked distance lie.

    The default matches the box's HEIGHT, since "as tall as the block it
    replaces" is what puts the obstacle in the water rather than under it -- but
    with the footprint clamped, because height alone is wrong for anything flat:
    scaled up to the box's height a flat part spans the tank wall to wall. The
    clamp keeps the sweep inside the tank in x and leaves the water somewhere
    to go in z.
    """
    flat = tris.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    size = np.maximum(hi - lo, 1e-9)
    if target_h > 0.0:
        s = float(target_h / size[1])          # explicit: the caller's business
    else:
        s = float(min(BY1 / size[1],                                   # as tall as the box
                      (0.9 * (X1 - X0) - 2.0 * PADDLE_AMP) / size[0],  # sweep stays inside
                      0.8 * (Z1 - Z0) / size[2]))                      # leave a flow gap
    out = (flat - lo) * s
    out[:, 0] += 0.5 * (BX0 + BX1) - 0.5 * size[0] * s   # centred in x on the box
    out[:, 2] += 0.5 * (BZ0 + BZ1) - 0.5 * size[2] * s   # ... and in z
    out[:, 1] += FLOOR                                    # resting on the floor
    return out.reshape(-1, 3, 3).astype(np.float32), s, size


@wp.kernel
def sample_sdf_kernel(vol: wp.uint64, pts: wp.array(dtype=wp.vec3),
                      out: wp.array(dtype=float)):
    i = wp.tid()
    out[i] = wp.volume_sample_f(vol, wp.volume_world_to_index(vol, pts[i]),
                                wp.Volume.LINEAR)


@wp.kernel
def bake_sdf_kernel(mid: wp.uint64, out: wp.array3d(dtype=float),
                    lo: wp.vec3, h: float, far: float):
    i, j, k = wp.tid()
    p = lo + wp.vec3(float(i) * h, float(j) * h, float(k) * h)
    # Winding-number sign, not the pseudonormal one: a tessellated CAD solid is
    # a triangle SOUP with duplicated vertices and no adjacency, and it is
    # routinely a little bit open at the seams. The winding number needs neither
    # adjacency nor watertightness to get inside/outside right. It is the
    # expensive query, which is affordable precisely because this runs once.
    q = wp.mesh_query_point_sign_winding_number(mid, p, far)
    if q.result:
        out[i, j, k] = q.sign * wp.length(p - wp.mesh_eval_position(mid, q.face, q.u, q.v))
    else:
        out[i, j, k] = far


def bake_sdf(tris, res, dev):
    """Mesh -> NanoVDB signed distance volume, in the mesh's rest frame."""
    flat = tris.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    voxel = float(max(hi - lo) / max(res, 8))
    # Pad so the field carries a usable band OUTSIDE the surface: collide() only
    # acts within WALL_EPS of it, but the gradient at that distance must be real
    # and not clamped against the grid edge.
    pad = max(4.0 * voxel, 2.0 * WALL_EPS)
    g_lo = lo - pad
    dims = np.maximum(np.ceil((hi + pad - g_lo) / voxel).astype(int) + 1, 2)
    far = float(np.linalg.norm(hi - lo) + pad)

    mesh = wp.Mesh(points=wp.array(flat, dtype=wp.vec3, device=dev),
                   indices=wp.array(np.arange(len(flat), dtype=np.int32), device=dev),
                   support_winding_number=True)
    grid = wp.zeros(tuple(dims), dtype=float, device=dev)
    wp.launch(bake_sdf_kernel, dim=tuple(dims), device=dev,
              inputs=[mesh.id, grid, wp.vec3(*g_lo.tolist()), voxel, far])
    # bg_value keeps a query that falls outside the grid reading as "far
    # outside", so a particle that leaves the padded band simply never collides
    # rather than sampling garbage.
    field = grid.numpy()
    # A closed mesh has an interior, so the field must go negative somewhere. If
    # it never does, the winding number found nothing to be inside of -- an open
    # shell, or a surface with inconsistent facet winding. Worth saying out loud
    # because the failure is quiet: an UNSIGNED field still repels particles at
    # the surface, so the fluid looks right while nothing can ever be pushed back
    # OUT of the obstacle.
    closed = bool((field < 0.0).any())
    vol = wp.Volume.load_from_numpy(field, min_world=tuple(g_lo.tolist()),
                                    voxel_size=voxel, bg_value=far)
    g_hi = g_lo + (dims - 1) * voxel
    return vol, voxel, dims, mesh, g_lo, g_hi, closed


# --- particle initialisation --------------------------------------------------

wp.init()
device = wp.get_preferred_device()

obstacle_tris = None
obstacle_vol = None
OBSTACLE_VOL_ID = wp.uint64(0)
OBSTACLE_LO = wp.vec3(0.0, 0.0, 0.0)   # unused on the box path
OBSTACLE_HI = wp.vec3(0.0, 0.0, 0.0)
if OBSTACLE:
    if not os.path.exists(OBSTACLE):
        raise SystemExit(f"--obstacle: no such file: {OBSTACLE}")
    _raw_tris = load_obstacle_mesh(OBSTACLE)
    obstacle_tris, _fit_s, _raw_size = fit_obstacle(_raw_tris, OBSTACLE_H)
    _t0 = time.perf_counter()
    obstacle_vol, _vox, _dims, _obstacle_mesh, _glo, _ghi, _closed = bake_sdf(
        obstacle_tris, SDF_RES, device)
    OBSTACLE_VOL_ID = obstacle_vol.id
    OBSTACLE_LO = wp.vec3(*_glo.tolist())
    OBSTACLE_HI = wp.vec3(*_ghi.tolist())
    wp.synchronize_device(device)
    _fit_size = _raw_size * _fit_s
    print(f"obstacle: {os.path.basename(OBSTACLE)}  {len(_raw_tris):,} tris  "
          f"raw {_raw_size[0]:.4g} x {_raw_size[1]:.4g} x {_raw_size[2]:.4g} "
          f"-> x{_fit_s:.4g} -> {_fit_size[0]:.3f} x {_fit_size[1]:.3f} x "
          f"{_fit_size[2]:.3f} m")
    print(f"          sdf {_dims[0]}x{_dims[1]}x{_dims[2]} @ {_vox * 1000:.2f} mm/voxel, "
          f"baked in {(time.perf_counter() - _t0) * 1000:.0f} ms, "
          f"{'closed (signed)' if _closed else 'NOT CLOSED -- unsigned'}")
    if not _closed:
        print("          warning: no interior found, so the field cannot push a particle")
        print("                   back OUT of the mesh. Surface repulsion still works.")

nz = int((Z1 - Z0 - 2.0 * WALL_EPS) / D)
if FLAT:
    # DEPTH sets ny, not N: a flat pool is defined by how deep the water is,
    # and the particle count follows from the footprint. Starting clear of the
    # piston keeps the first frame from resolving an interpenetration.
    FILL_X0 = BX1 + WALL_EPS
    nx = int((X1 - WALL_EPS - FILL_X0) / D)
    ny = max(1, int(round(DEPTH / D)))
else:
    FILL_X0 = X0 + WALL_EPS
    nx = int((FILL_X1 - X0 - 2.0 * WALL_EPS) / D)
    ny = max(1, N // (nx * nz))


def seed_lattice(nx, ny, nz, chunk=1 << 22):
    """The fill lattice, in chunks, straight into float32.

    np.meshgrid + np.stack builds three int64 index grids and a float64
    intermediate before the cast, which is ~84 bytes per lattice point: 17 GB
    of HOST memory at 200M particles, single-threaded, for data that ends up as
    12 bytes on the GPU. That OOMs a compute node before the GPU is touched.
    Chunking writes the answer directly and peaks at a few hundred MB.
    """
    m = nx * ny * nz
    out = np.empty((m, 3), np.float32)
    for lo in range(0, m, chunk):
        hi = min(lo + chunk, m)
        i = np.arange(lo, hi, dtype=np.int64)
        ix, r = np.divmod(i, ny * nz)
        iy, iz = np.divmod(r, nz)
        out[lo:hi, 0] = FILL_X0 + (ix + 0.5) * D
        out[lo:hi, 1] = FLOOR + WALL_EPS + (iy + 0.5) * D
        out[lo:hi, 2] = Z0 + WALL_EPS + (iz + 0.5) * D
    return out


p0 = seed_lattice(nx, ny, nz)
if BEACH_SLOPE > 0.0:
    # Drop what the shore already occupies, before it is ever a particle.
    p0 = p0[p0[:, 1] >= FLOOR + WALL_EPS + (p0[:, 0] - BEACH_TOE) * BEACH_SLOPE]
# drop the particles that would start inside the obstacle. Seeding happens at
# t = 0, where the paddle offset is sin(0) == 0, so the rest frame the field was
# baked in IS the world frame here -- no offset to undo.
if obstacle_vol is not None:
    _q = wp.array(p0, dtype=wp.vec3, device=device)
    _d = wp.zeros(len(p0), dtype=float, device=device)
    wp.launch(sample_sdf_kernel, dim=len(p0), device=device,
              inputs=[OBSTACLE_VOL_ID, _q, _d])
    keep = _d.numpy() >= WALL_EPS
else:
    keep = ~((p0[:, 0] > BX0 - WALL_EPS) & (p0[:, 0] < BX1 + WALL_EPS) &
             (p0[:, 2] > BZ0 - WALL_EPS) & (p0[:, 2] < BZ1 + WALL_EPS) &
             (p0[:, 1] < BY1 + WALL_EPS))
p0 = p0[keep]
rng = np.random.default_rng(17)
p0 = (p0 + rng.uniform(-0.06 * D, 0.06 * D, p0.shape)).astype(np.float32)
N = len(p0)
if REPLAY and N != int(_meta["n"]):
    raise SystemExit(f"--replay: the fill produced {N:,} particles but the dump holds "
                     f"{int(_meta['n']):,}; the obstacle or the tank geometry differs "
                     f"from the run that made it")

# A cell fully inside the fluid collects (CELL/D)^3 particles; the surface sits
# near half of that, so the iso-threshold follows the resolution automatically.
ISO = 0.5 * (CELL / D) ** 3
# wp.MarchingCubes winds triangles so the CCW-implied normal points INTO the
# density, i.e. opposite the outward -grad normal DensitySurface computes. Both
# backends do `normal *= faceDirection` for a Side.Double material, so shipping
# un-negated normals leaves the water's shading normal pointing DOWN. That does
# not just darken the diffuse term -- it disables transmission: getIBLVolumeRefraction
# refracts along the normal, so the refracted ray goes UP into the sky, samples
# the backdrop above the horizon and returns nothing. Measured: with a bright RED
# pool floor and transmission 1.0, the water showed exactly zero red.
# Measured 2026-09-09: flipping this changes the shading (the water darkens)
# but does NOT restore transmission, so the winding/normal question is real
# but is NOT the see-through bug. Default left at the shipped value; the
# flag stays so the A/B can be repeated.
MC_SIGN = cli_arg("--mc-sign", 1.0, float)
# Reverse the marching-cubes winding so the water's visible surface is
# FRONT-facing. Without this a Side.Double transmissive material silently
# renders with transmissionFactor == 0 (see DensitySurface.expand).
# Reverse the marching-cubes winding so the water's visible surface is
# FRONT-facing. wp.MarchingCubes winds triangles INTO the density, and the
# deferred renderer traces the geometry to measure the glass chord through the
# water -- with the winding inverted it cannot tell inside from outside, the
# chord measures ~0, Beer-Lambert contributes nothing, and the pool renders as
# colourless quicksilver from above while staying turquoise from below.
# Flipping it restores the tropical colour and leaves every non-water surface
# untouched. With the winding corrected the shipped -grad normals are already
# aligned, so MC_SIGN stays 1.0.
MC_FLIP = bool(cli_arg("--mc-flip", 1, int))
# The default followed SCALE**2 and knew nothing about the footprint, so a
# widened run kept the 700k of a 2 m tank against tens of millions of live
# triangles. The clamp is safe (build_surface bounds it, _expand parks the
# overflow and set_draw_range clamps the draw) but it truncates x-major -- it
# cuts the pool off as a slab at the far end, which is exactly where a wide
# shot is looking. The 6.0 is the measured triangles-per-area-per-cell^2 law
# (4.7) with headroom.
_AREA = (X1 - X0) * (Z1 - Z0)
MAX_TRIS = cli_arg("--max-tris",
                   int(700_000 * SCALE ** 2) if IS_REF
                   else int(6.0 * _AREA / CELL ** 2), int)

x = wp.array(p0, dtype=wp.vec3, device=device)
v = wp.zeros(N, dtype=wp.vec3, device=device)
xp = wp.zeros(N, dtype=wp.vec3, device=device)
xtmp = wp.zeros(N, dtype=wp.vec3, device=device)
xprev = wp.zeros(N, dtype=wp.vec3, device=device)   # Chebyshev's k-1 iterate
lam = wp.zeros(N, dtype=float, device=device)
omega = wp.zeros(N, dtype=wp.vec3, device=device)
vtmp = wp.zeros(N, dtype=wp.vec3, device=device)
col = wp.zeros(N, dtype=wp.vec3, device=device)
# wp.HashGrid(128,128,128) is not a capacity, it is a PERIODIC TILE: warp folds
# world cells with x % dim_x, so at H = 18 mm cells the shipped grid tiles every
# 2.304 m. The tank outgrows it at --wide 1.10, and at --wide 16 it folds 15:1
# in x and 8:1 in z. Nothing breaks and nothing is reported -- every neighbour
# loop re-tests r2 < H*H, so aliased candidates are distance-rejected -- it just
# costs ~25x in the kernels that are already most of the frame.
GRID_CELLS = cli_arg("--grid-cells", 1 << 28, int)   # bucket budget (2.0 GB)


def hash_grid_dims():
    """Bucket dims that cover the tank at the neighbour radius."""
    if IS_REF and SCALE == 1.0:
        # The shipped grid, kept exactly. Bucket assignment fixes the sort
        # order of hash_grid_point_id, which fixes the summation order in every
        # neighbour loop, so this is what makes a reference run bit-identical
        # rather than merely equivalent. Do not tidy it away: it is what keeps
        # old dumps replaying to the same pixels.
        return [128, 128, 128]
    d = [max(8, math.ceil(s / H) + 2)
         for s in (X1 - X0, GRID_TOP - FLOOR, Z1 - Z0)]
    while d[0] * d[1] * d[2] > GRID_CELLS:
        # Fold Y first: the fluid is a shallow layer in a tall box, so a folded
        # Y bucket collides occupied cells against EMPTY air. X and Z are
        # densely occupied across the whole floor, where folding collides
        # occupied against occupied -- the expensive kind.
        if d[1] > 16:
            d[1] = max(16, d[1] // 2)
        else:
            d[0 if d[0] >= d[2] else 2] //= 2
    return d


GDX, GDY, GDZ = hash_grid_dims()
grid = wp.HashGrid(GDX, GDY, GDZ, device)
grid.reserve(N)

print(f"fluid: {N:,} particles on {device} | pool {X1 - X0:.2f} x {Z1 - Z0:.2f} m "
      f"x {ny * D:.3f} m deep (--wide {WIDE_X:g}x{WIDE_Z:g}, "
      f"{'flat' if FLAT else 'dam'}"
      f"{f', beach {BEACH_RUN:g} m' if BEACH_RUN else ''})\n"
      f"       surface grid {NGX}x{NGY}x{NGZ} = {NGX * NGY * NGZ / 1e6:.1f}M nodes "
      f"@ {CELL * 1000:.1f} mm, top {GRID_TOP:.2f} m | iso={ISO:.2f} | "
      f"max_tris {MAX_TRIS:,}\n"
      f"       hash grid {GDX}x{GDY}x{GDZ} = {GDX * GDY * GDZ / 1e6:.1f}M buckets "
      f"@ {H * 1000:.1f} mm cells, ~{N / 8e6:.1f}M occupied\n"
      f"       rho0={RHO0:.4g} sum_grad2={SG2_REST:.4g} "
      f"eps={EPS_CFM:.3g} k_corr={S_CORR_K:.3g} K={SUBSTEPS * ITERATIONS}")

# Round one of the blur stays binomial so grid-aligned noise is killed
# outright; round two is narrowed, which buys thinness without a new noise
# source.
# --normal-blur N: N more binomial rounds of the field AFTER marching cubes, so
# the triangles keep the detail above and only the shading normals see the
# wider kernel. In a wide shot the grain is in the NORMAL, not the geometry --
# the relief is sub-pixel, the tilt is not -- and a post-extraction blur cannot
# thin a sheet, because the sheet is already out. Off for the reference, which
# stays bit-identical. Measured with --tilt at the film's own 9 mm spacing and
# K = 6, normal noise against the wave's tilt: --cell 1.0 unblurred (the first
# Idun film) 0.303 rad vs 0.154, SNR 0.51; --cell 1.5 alone 0.137, SNR 0.93;
# +4 rounds 0.046, SNR 2.0; +8 rounds 0.028, SNR 3.0. Eight is the default
# because a cluster shot sits further away than any preview and has no MSAA.
NORMAL_BLUR = cli_arg("--normal-blur", 0 if IS_REF else 8, int)
surface = DensitySurface((GX0, GY0, GZ0), CELL, (NGX, NGY, NGZ), device,
                         blur=(0.25, 0.125), normal_blur=NORMAL_BLUR)

sim_time = 0.0
frame_no = 0


def paddle_offset(t):
    """Sweep displacement of the paddle along x at time t."""
    a = PADDLE_AMP
    if PADDLE_RAMP > 0.0:
        # A piston at full stroke from rest is a step function into 0.27 m of
        # water: the first half-cycle makes a bore, not a wave.
        a *= min(1.0, t / PADDLE_RAMP)
    return a * math.sin(2.0 * math.pi * t / PADDLE_PERIOD)


def paddle_extent(t):
    """World x-extent of the paddle box at time t."""
    c = paddle_offset(t)
    return BX0 + c, BX1 + c


# --- dump / replay frame format --------------------------------------------------
# One .npz per frame: uint16 positions quantised against the frame's own
# bounding box, plus that box in float32. 6 bytes per particle; the step is
# (extent / 65535), 0.03 mm for the 2.1 m tank, two orders under the spacing at
# any --scale this runs at. Frame k holds the state AFTER sim_step k, so
# sim_time = (k + 1) * DT when it is on screen -- the paddle is placed from
# sim_time, and the replay advances it the same way.

_dump_nonfinite_warned = False


def write_dump_frame(path, pos):
    global _dump_nonfinite_warned
    if not np.isfinite(pos).all():
        if not _dump_nonfinite_warned:
            _dump_nonfinite_warned = True
            print("  warning: non-finite positions in the dump (the simulation diverged); "
                  "written as 0")
        pos = np.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
    lo = pos.min(axis=0).astype(np.float32)
    hi = pos.max(axis=0).astype(np.float32)
    span = np.maximum(hi - lo, np.float32(1e-6))
    q = np.rint((pos - lo) * (np.float32(65535.0) / span)).astype(np.uint16)
    np.savez(path, q=q, lo=lo, hi=hi)


def read_dump_frame(path):
    with np.load(path) as z:
        lo, hi, q = z["lo"], z["hi"], z["q"]
    return (q.astype(np.float32) * ((hi - lo) / np.float32(65535.0)) + lo).astype(np.float32)


def dump_frame_path(directory, k):
    return os.path.join(directory, f"f{k:06d}.npz")


# Replay state: the reader thread keeps one frame ahead of the GPU, since
# np.load and the dequantise release the GIL and take tens of milliseconds at
# millions of particles.
_replay = {"pool": None, "pending": None, "held": False}


def replay_step():
    """Stand-in for sim_step: load dumped frame `frame_no` into x."""
    global sim_time, frame_no
    total = int(_meta["frames"])
    if frame_no >= total:
        if not _replay["held"]:
            _replay["held"] = True
            print(f"replay: past the last dumped frame ({total}); holding it")
        return
    if _replay["pool"] is None:
        _replay["pool"] = ThreadPoolExecutor(max_workers=1)
        _replay["pending"] = _replay["pool"].submit(read_dump_frame,
                                                    dump_frame_path(REPLAY, frame_no))
    pos = _replay["pending"].result()
    if frame_no + 1 < total:
        _replay["pending"] = _replay["pool"].submit(read_dump_frame,
                                                    dump_frame_path(REPLAY, frame_no + 1))
    wp.copy(x, wp.array(pos, dtype=wp.vec3, device=device))
    sim_time += DT
    frame_no += 1


def sim_step():
    """Advance one rendered frame of fluid."""
    global sim_time, frame_no
    if REPLAY:
        replay_step()
        return
    dt = DT / SUBSTEPS
    for _ in range(SUBSTEPS):
        bx0, bx1 = paddle_extent(sim_time)
        ox = paddle_offset(sim_time)
        wp.launch(predict, dim=N, device=device,
                  inputs=[x, v, xp, dt, bx0, bx1, OBSTACLE_VOL_ID, ox,
                          OBSTACLE_LO, OBSTACLE_HI])
        grid.build(points=xp, radius=H)
        # Three-buffer rotation instead of a ping-pong: solve_delta reads the
        # current iterate AND the previous one (for the Chebyshev extrapolation)
        # while writing a third. prev is cur on iteration 1, where omega == 1
        # makes the extrapolation a no-op and the read harmless.
        prev = cur = xp
        spare = [xtmp, xprev]
        for k in range(ITERATIONS):
            if k:
                # Rebuild on the current iterate instead of reusing the grid
                # predict left behind. A build is a locality investment, not
                # overhead: hash_grid_point_id hands every kernel a spatially
                # sorted thread order, and a grid built on positions the solver
                # has since moved gives the neighbour gathers a scrambled one.
                # The rebuild costs less than it returns in the two solve
                # kernels, and rebuilding LESS often loses by the same mechanism.
                grid.build(points=cur, radius=H)
            out = spare.pop(0)
            wp.launch(solve_lambda, dim=N, device=device,
                      inputs=[cur, grid.id, RHO0, EPS_CFM, lam])
            wp.launch(solve_delta, dim=N, device=device,
                      inputs=[cur, lam, grid.id, RHO0, W_DQ, S_CORR_K,
                              JACOBI_RELAX, OMEGAS[k], prev, bx0, bx1,
                              OBSTACLE_VOL_ID, ox, OBSTACLE_LO,
                              OBSTACLE_HI, out])
            if prev is not cur:
                spare.append(prev)
            prev, cur = cur, out
        wp.launch(finalize, dim=N, device=device, inputs=[x, cur, v, dt])
        sim_time += dt
    # Same trade for the two velocity passes, which query x -- a whole substep
    # of projection away from the positions the last build saw.
    grid.build(points=x, radius=H)
    wp.launch(curl_and_viscosity, dim=N, device=device,
              inputs=[x, v, grid.id, RHO0, omega, vtmp])
    wp.launch(vorticity_confine, dim=N, device=device,
              inputs=[x, omega, vtmp, grid.id, DT, v])
    frame_no += 1


_clamped = False


def build_surface():
    """Splat, blur, marching cubes; returns the (clamped) triangle count."""
    global _clamped
    got = surface.build(x, N, ISO)
    if got > MAX_TRIS and not _clamped:
        # Silent truncation would just look like a chunk of missing water.
        _clamped = True
        print(f"  note: marching cubes produced {got:,} triangles, clamped to "
              f"{MAX_TRIS:,} -- raise --max-tris if the surface looks cut off")
    return min(got, MAX_TRIS)


if PROBE:
    err = wp.zeros(N, dtype=float, device=device)
    print(f"{'t':>6} {'|v|mean':>8} {'|v|max':>8} {'y_mean':>7} {'y_max':>7} "
          f"{'comp%':>7} {'cmax%':>7} {'tris':>9}")
    comp_acc = []
    y_peak = 0.0
    for f in range(int(60 * cli_arg("--probe", 3.0, float))):
        sim_step()
        if f % 12 == 0:
            wp.launch(measure_compression, dim=N, device=device,
                      inputs=[x, grid.id, RHO0, err])
            e = err.numpy()
            comp_acc.append(e.mean())
            vv = np.linalg.norm(v.numpy(), axis=1)
            yy = x.numpy()[:, 1]
            y_peak = max(y_peak, float(yy.max()))
            print(f"{f/60.0:6.2f} {vv.mean():8.3f} {vv.max():8.3f} {yy.mean():7.3f} "
                  f"{yy.max():7.3f} {e.mean()*100:7.3f} {e.max()*100:7.2f} "
                  f"{build_surface():9,d}")
    print(f"mean compression over run: {np.mean(comp_acc)*100:.3f}%  "
          f"(iters={ITERATIONS} rho={RHO_CHEB} relax={JACOBI_RELAX} "
          f"substeps={SUBSTEPS} K={SUBSTEPS * ITERATIONS})")
    # The surfacing grid has a lid, and a particle above it contributes NO
    # density at all -- it does not thin, it vanishes, and marching cubes
    # leaves the surface open at the top plane. That is invisible in the
    # numbers and obvious in the render, so say it here where it is cheap.
    if y_peak > GRID_TOP:
        print(f"WARNING: water reached y = {y_peak:.3f} m but the surfacing grid ends "
              f"at {GRID_TOP:.3f} m.\n"
              f"         Everything above it is missing from the surface. "
              f"Pass --grid-top {y_peak + 0.2:.1f} (costs "
              f"{(int((y_peak + 0.2 - GY0) / CELL) + 1) / NGY:.1f}x the surfacing grid).")
    else:
        print(f"surfacing lid: water peaked at {y_peak:.3f} m, grid top {GRID_TOP:.3f} m -- clear.")
    sys.exit(0)


if DUMP:
    # Simulate-only, like --probe: no canvas, no renderer. The write of frame k
    # (device readback done here, quantise + savez on a worker thread) overlaps
    # the simulation of frame k + 1; the queue is one deep so a slow disk
    # throttles the loop instead of filling memory.
    os.makedirs(DUMP, exist_ok=True)
    total = int(round(DUMP_SECONDS * 60))
    meta = {"format": "warp_fluid dump", "version": 1,
            "scale": SCALE, "n_requested": N_REQUESTED, "n": N, "d": D,
            "dt": DT, "fps": 60, "substeps": SUBSTEPS, "iters": ITERATIONS,
            "rho": RHO_CHEB, "frames": total,
            "obstacle": OBSTACLE, "obstacle_height": OBSTACLE_H, "sdf_res": SDF_RES}
    with open(os.path.join(DUMP, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    pool = ThreadPoolExecutor(max_workers=1)
    pending = None
    t0 = time.perf_counter()
    for k in range(total):
        sim_step()
        pos = x.numpy()
        if pending is not None:
            pending.result()
        pending = pool.submit(write_dump_frame, dump_frame_path(DUMP, k), pos)
        if k % 60 == 0:
            el = time.perf_counter() - t0
            print(f"  frame {k}/{total}  ({el:.0f}s elapsed)", flush=True)
    if pending is not None:
        pending.result()
    pool.shutdown()
    el = time.perf_counter() - t0
    nbytes = sum(os.path.getsize(dump_frame_path(DUMP, k)) for k in range(total))
    print(f"dumped {total} frames of {N:,} particles in {el:.0f}s "
          f"({1000.0 * el / max(total, 1):.0f} ms/frame) -> {DUMP}  "
          f"{nbytes / 2**20:.0f} MB")
    sys.exit(0)


# --- environment assets --------------------------------------------------------
# The environment map is not decoration: what the water reflects and refracts IS
# the water's look, and a bare grey sky makes grey water. The demo fetches an
# indoor swimming pool HDRI (Poly Haven, CC0) and a mosaic tile texture
# (Wikimedia Commons) once into the temp dir; offline, the procedural sky below
# and a plain aqua liner stand in, so the demo still runs with no network.

POOL_HDR_URL = ("https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/"
                "indoor_pool_2k.hdr")
MOSAIC_URL = ("https://upload.wikimedia.org/wikipedia/commons/e/e8/"
              "Baby_blue_aqua_mosaic_swimming_pool_square_seamless_tiled_"
              "floor_texture.jpg")


def fetch_asset(url, name):
    """Cache a demo asset in the temp dir; None if offline."""
    path = os.path.join(tempfile.gettempdir(), name)
    if os.path.exists(path):
        return path
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "threepp-demo"})
        with urllib.request.urlopen(req, timeout=20) as r, open(path, "wb") as f:
            f.write(r.read())
        return path
    except Exception as e:
        print(f"  asset: {name} unavailable ({e}); using the built-in fallback")
        return None


SUN_DIR = np.array([0.42, 0.45, 0.79])
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)


def make_sky_hdr(path, W=2048, HH=1024):
    j = np.arange(HH).reshape(HH, 1)
    i = np.arange(W).reshape(1, W)
    theta = (j / HH) * math.pi
    phi = (i / W) * 2 * math.pi - math.pi
    y = np.broadcast_to(np.cos(theta), (HH, W))
    sin_t = np.sin(theta)
    xx = sin_t * np.cos(phi)
    zz = sin_t * np.sin(phi)

    up = np.clip(y, 0.0, 1.0)[..., None]
    t = up ** 0.4
    sky = np.array([0.62, 0.70, 0.82]) * (1.0 - t) + np.array([0.07, 0.22, 0.55]) * t
    glow = np.exp(-(y * y) / (2 * 0.006))[..., None]
    sky = sky + glow * np.array([1.0, 0.80, 0.55]) * 0.5
    # Below the horizon matters more than it looks: wave facets tilted toward
    # the camera reflect THIS hemisphere, so it tracks the ground rather than
    # going black.
    sky = np.where((y < 0)[..., None], np.array([0.40, 0.42, 0.46]), sky)

    d = np.stack([xx, y, zz], axis=-1)
    ang = np.arccos(np.clip((d * SUN_DIR).sum(-1), -1.0, 1.0))
    core = np.exp(-(ang / math.radians(1.6)) ** 2)
    halo = np.exp(-(ang / math.radians(12.0)) ** 2)
    sky = sky + (core * 70.0 + halo * 5.0)[..., None] * np.array([1.0, 0.95, 0.86])

    return write_radiance_hdr(path, sky)


# --- threepp scene ------------------------------------------------------------

if VULKAN and not tp.vulkan_available():
    print("vulkan not available on this machine; falling back to OpenGL")
    VULKAN = False

WIDTH, HEIGHT = parse_size(cli_arg("--size", "1280x800", str))
# On a compute node there is no window system at all, so the GL context comes
# from the driver through EGL and `canvas` is None. Everything below that needs
# a window -- OrbitControls, ImGui, resize -- lives in the interactive branch,
# which a headless run never reaches.
display = open_display("threepp x warp - fluid", WIDTH, HEIGHT,
                       vulkan=VULKAN, egl=EGL,
                       headless=(SHOT or BENCH or VIDEO > 0) and not FRAMES)
canvas = display.canvas
renderer = display.renderer
if display.context is not None:
    print(f"rendering headless: {display.describe()}")
    if not (VIDEO or SHOT or BENCH):
        raise SystemExit(
            "There is no window here, so there is nothing to interact with. Ask for a "
            "render instead: --video S (S seconds to mp4), --shot, or --bench.")
if not VULKAN:
    renderer.shadow_map_enabled = True
# --murk declares an underwater medium, which makes shadeWater's above branch
# use applyMurk and agree with the from-below path. It is OFF here on purpose:
# the murk is a horizontal PLANE, so it tints everything below the waterline --
# including the OUTSIDE of the tank, the table and the chairs, which are in air.
# That model fits an ocean, where below the surface really is water; it does not
# fit a pool standing on a table. The winding fix below colours the water
# without touching anything that is not water (measured: the tank's outer wall
# stays byte-identical).
MURK_SIGMA = cli_arg("--murk", 0.0, float)          # 1/m; 0 = off
MURK_COLOR = (0.055, 0.30, 0.36)
if VULKAN and MURK_SIGMA > 0.0:
    renderer.set_fog_water_surface_y(FLOOR + (DEPTH if FLAT else 0.29))
    renderer.set_underwater_murk(MURK_SIGMA, tp.Color(*MURK_COLOR))

renderer.tone_mapping = tp.ToneMapping.ACESFilmic
# Authored, not guessed: a film should state its exposure. The default is
# the shipped value for each backend; --exposure overrides it.
# Measured by sweeping exposure x env_map_intensity and looking at the frames
# (scripts in the session scratchpad). The shipped GL value of 1.15 was
# tuned for a 2 m tank lit by an indoor-pool HDRI and viewed from ABOVE,
# where you are mostly looking THROUGH the water at a tiled floor. On a
# grazing line over open water you are mostly looking at REFLECTIONS, and
# 1.15 blows the surface to white.
renderer.tone_mapping_exposure = cli_arg(
    "--exposure", (0.95 if VULKAN else 1.40) if IS_REF else 0.35, float)

SUN_POS = ((2.4, 3.2, 4.2) if IS_REF else
           ((X0 if CAM_FAR_END else X1) * 1.6, X1 * 0.30, Z1 * 0.40))
# Low (about 10 deg) and at the FAR end, so the specular path runs the whole
# length of the pool back into the lens. A high sun on flat water gives an
# even sheen and no glitter, which is most of why the first frame read as milk.

scene = tp.Scene()
# The indoor-pool HDRI is Vulkan-only: the ray-traced path turns its bright
# warm interior into reflections, refracted light and sparkle. GL's screen-space
# transmission just floods with it and washes the water out, so GL keeps the
# procedural sky.
_pool_hdr = fetch_asset(POOL_HDR_URL, "threepp_indoor_pool_2k.hdr") if VULKAN else None
if IS_REF or VULKAN:
    env = tp.RGBELoader().load(_pool_hdr if _pool_hdr else make_sky_hdr(
        os.path.join(tempfile.gettempdir(), "threepp_fluid_sky.hdr")))
else:
    # A real sun disc, on the same direction as the key light, so the
    # reflection, the glint and the shadows agree. This is the single
    # biggest difference between water and blue gel at a grazing angle.
    _sd = np.array(SUN_POS, dtype=np.float64)
    _sd = _sd / np.linalg.norm(_sd)
    env = sky_env(tuple(_sd), below_horizon=(0.22, 0.27, 0.31),
                  below_nadir=(0.05, 0.06, 0.07))
scene.environment = env
scene.background = env

# Length is the subject. The shipped camera sits 1.2 m from the origin with a
# 46 deg lens: point that at a 34 m pool and you get a featureless mat to the
# horizon with a visible far wall -- technically correct and completely flat.
# A low eye just above the waterline, a long lens, and a target pinned to the
# far end instead turn the length into the picture: the wave train recedes,
# and apparent size halving with distance is the depth cue no single prop can
# give you.
DOLLY = cli_arg("--dolly", 0.0, float)      # camera speed ALONG ITS OWN VIEW, m/s
WLINE = DEPTH if FLAT else 0.07             # the waterline the shot is built on
CAM_DIR = -1.0 if CAM_FAR_END else 1.0      # +1 looks toward +x, -1 toward -x
CAM_FOV = cli_arg("--fov", 46 if IS_REF else (40 if CAM_ALL else 30), float)
CAM_EL = cli_arg("--cam-el", 14.0, float)   # degrees above the water, --cam-from all
CAM_AZ = cli_arg("--cam-az", 12.0, float)   # degrees off the tank axis, ditto


ZERO3 = np.zeros(3)


def _view_fill(eye, tgt, pts, tanv, tanh):
    """How much of the frame `pts` fills, and how far off-centre it sits.

    Returns (fill, aim) with fill in units of "1.0 exactly touches an edge",
    and aim a world-space direction: adding aim*distance to the target nulls
    the offset, i.e. it is a tilt in both axes at once. BOTH axes matter --
    centring only the vertical left the pool at 70% of the frame width, since
    the rig sits a dozen degrees off the tank axis.
    """
    f = tgt - eye
    f = f / np.linalg.norm(f)
    r = np.cross(f, np.array([0.0, 1.0, 0.0]))
    nr = np.linalg.norm(r)
    if nr < 1e-9:
        return 1e9, ZERO3
    r = r / nr
    u = np.cross(r, f)
    d = pts - eye
    zv = d @ f
    if np.min(zv) <= 1e-3:                  # something is behind the lens
        return 1e9, ZERO3
    ax, ay = (d @ r) / zv, (d @ u) / zv
    fill = max(np.max(np.abs(ax)) / tanh, np.max(np.abs(ay)) / tanv)
    return fill, (0.5 * (np.max(ax) + np.min(ax)) * r
                  + 0.5 * (np.max(ay) + np.min(ay)) * u)


def fit_pool_camera():
    """Solve the eye distance so the whole tank fits, then tilt to centre it.

    --wide changes the footprint by a factor of sixteen in area, so a framing
    typed in by hand is right for exactly one value of it. Binary-search the
    distance along a fixed elevation/azimuth instead and the shot follows the
    flag. The second loop nulls the vertical offset: aiming at the centroid of
    a flat rectangle seen obliquely leaves it in the bottom third under an
    empty sky, because perspective makes the near edge much the larger.
    """
    top = max(BY1, WLINE + 0.35)            # the piston is the tallest thing
    pts = np.array([[x, y, z] for x in (X0, X1) for y in (FLOOR, top)
                    for z in (Z0, Z1)], np.float64)
    anchor = np.array([0.5 * (X0 + X1), WLINE, 0.5 * (Z0 + Z1)])
    el, az = math.radians(CAM_EL), math.radians(CAM_AZ)
    d = np.array([-CAM_DIR * math.cos(el) * math.cos(az),
                  math.sin(el),
                  math.cos(el) * math.sin(az)])
    tanv = math.tan(math.radians(CAM_FOV) * 0.5)
    tanh = tanv * display.aspect
    off, dist = ZERO3.copy(), 10.0
    for _ in range(6):
        lo, hi = 0.5, 400.0
        for _ in range(48):
            mid = 0.5 * (lo + hi)
            fill, _ = _view_fill(anchor + d * mid, anchor + off, pts, tanv, tanh)
            if fill > 0.94:                 # a margin, so a wave crest has room
                lo = mid
            else:
                hi = mid
        dist = hi
        _, aim = _view_fill(anchor + d * dist, anchor + off, pts, tanv, tanh)
        off = off + aim * dist
    return list(anchor + d * dist), tuple(anchor + off)


if CAM_ALL:
    CAM_EYE, CAM_TGT = fit_pool_camera()
    # The eye-level shot wants a low sun at the far end and gets a glitter path
    # for free. Lift the camera to 14 degrees and that same sun reflects clean
    # over its head: the pool went black. A mirror surface sends the sun to the
    # eye only from the MIRRORED direction, so derive it instead of typing it.
    # l = (-v.x, v.y, -v.z) for v the direction from the pool to the camera; the
    # +0.06 on the elevation walks the highlight away down the pool, turning a
    # blob under the lens into a path that leads the eye to the wave maker.
    _v = np.array(CAM_EYE) - np.array([0.0, WLINE, 0.0])
    _v = _v / np.linalg.norm(_v)
    _l = np.array([-_v[0], _v[1] + 0.06, -_v[2]])
    SUN_POS = tuple(_l / np.linalg.norm(_l) * (6.0 * W_LIN))
    if not VULKAN:
        # The sky was built above from the sun this block just replaced; build
        # it again, or the disc the water reflects is not where the light is.
        env = sky_env(tuple(_l / np.linalg.norm(_l)), below_horizon=(0.22, 0.27, 0.31),
                      below_nadir=(0.05, 0.06, 0.07))
        scene.environment = env
        scene.background = env
else:
    _cx = (X1 - 1.2) if CAM_FAR_END else (X0 + 1.2)
    _tx = X0 if CAM_FAR_END else X1
    CAM_EYE = [_cx, WLINE + 0.45, 0.06 * (Z1 - Z0)]
    CAM_TGT = (_tx, WLINE - 0.06, -0.02 * (Z1 - Z0))
camera = tp.PerspectiveCamera(CAM_FOV, display.aspect,
                              0.01 * W_LIN, 100 * W_LIN)
if IS_REF:
    camera.position.set(1.20, 0.63, 1.34)
    camera.look_at(-0.02, 0.07, 0.0)
else:
    camera.position.set(*CAM_EYE)
    camera.look_at(*CAM_TGT)


def camera_at(t):
    """The dolly. Height fixed, target pinned to the far end."""
    if DOLLY:
        camera.position.set(CAM_EYE[0] + DOLLY * CAM_DIR * t, CAM_EYE[1], CAM_EYE[2])
        camera.look_at(*CAM_TGT)

sun = tp.DirectionalLight(0xfff3e0, 2.6)
sun.position.set(*SUN_POS)
sun.cast_shadow = True
scene.add(sun)
scene.add(tp.HemisphereLight(0xbcd4ff, 0x2a2b28, 0.25))

# Everything the water should refract must be OPAQUE: threepp's transmission
# pre-pass renders only the opaque bucket plus the sky.
tank_mat = standard_material(0x8a8f96, 0.55, 0.1)
floor_mesh = tp.Mesh(tp.BoxGeometry(X1 - X0 + 0.06, 0.03, Z1 - Z0 + 0.06), tank_mat)
floor_mesh.position.set(0.5 * (X0 + X1), FLOOR - 0.015, 0.5 * (Z0 + Z1))
floor_mesh.receive_shadow = True
scene.add(floor_mesh)

# The pool liner. What the water refracts IS the water's colour -- on Vulkan
# almost entirely so (see the material below) -- and a grey floor can only ever
# produce grey water. It is inset inside the walls rather than painted onto the
# floor box because that box's sides are exposed below the walls, where a
# saturated colour reads as a bright stripe around the outside of the tank.
_mosaic = fetch_asset(MOSAIC_URL, "threepp_pool_mosaic.jpg")
TILE = 0.80            # world metres per mosaic sheet -> square tiles everywhere.
# One environment level for the whole scene. It was only ever applied to the
# water, so the liner and the walls kept full-strength image-based light --
# and a bright neutral sky added to their DIFFUSE term swamps the chroma of
# a blue mosaic. Measured: the tiles came back mean RGB 118/133/148 at
# saturation 33, i.e. a grey checkerboard, from a texture that is solidly
# blue. Lowering exposure only made it a darker grey (44/49/57, sat 15) --
# proof it was the light being ADDED, not the exposure.
ENV_I = cli_arg("--env-intensity", 1.0 if IS_REF else 0.25, float)
                       # Large rather than small: the ray-traced refraction
                       # samples the map at LOD 0, one ray per pixel, so fine
                       # tiles shatter into dark speckle under the rippled
                       # surface.


def liner_material(w, h, mosaic):
    # --floor-color is a debug knob: a strongly off-hue liner makes it obvious
    # at a glance whether the water is transmitting the floor or just tinted.
    m = standard_material(cli_arg("--floor-color", 0xffffff if mosaic else 0x4ec9de, lambda v: int(v, 0)), 0.5)
    m.env_map_intensity = ENV_I
    if mosaic:
        t = tp.TextureLoader().load(_mosaic, tp.ColorSpace.SRGB)
        t.wrap_s = t.wrap_t = tp.TextureWrapping.Repeat
        t.repeat = tp.Vector2(w / TILE, h / TILE)
        # Texture.anisotropy defaults to 1, i.e. none. A wall of 4 cm tiles seen
        # at a grazing angle then undersamples along the view direction and the
        # pattern beats into ~26 cm blocks -- it renders as a black-and-white
        # checkerboard rather than as a blue mosaic. Mipmaps alone do not fix it
        # (they blur the other axis instead); anisotropic filtering does.
        t.min_filter = tp.Filter.LinearMipmapLinear
        t.mag_filter = tp.Filter.Linear
        t.anisotropy = 16
        m.map = t
    return m


# Mosaic on the WALLS, plain aqua on the FLOOR (--floor-mosaic 1 to tile it).
# The floor is what the body of the water refracts, and this mosaic's saturated
# navy average drags the whole pool dark and busy; the plain aqua floor keeps
# the water luminous while the walls carry the tiled-pool identity at the rim
# and waterline.
#
# That matters MOST on Vulkan, where the water is ray-traced: darken what it
# refracts and the refracted term goes dark, leaving mostly specular, and the
# water reads as liquid metal instead of tropical. Measured as a regression
# when the floor was tiled by default -- do not turn it on for the Vulkan path
# without looking at a frame.
liner = tp.Mesh(tp.PlaneGeometry(X1 - X0, Z1 - Z0),
                liner_material(X1 - X0, Z1 - Z0,
                               mosaic=cli_arg("--floor-mosaic", 0, int) and _mosaic is not None))
liner.rotate_x(-math.pi / 2)
liner.position.set(0.5 * (X0 + X1), FLOOR + 0.0015, 0.5 * (Z0 + Z1))
liner.receive_shadow = True
scene.add(liner)

# The walls get the liner too. Refracted rays that miss the floor hit the tank's
# inner faces, and grey walls put grey right back into the water body. The tiles
# are an ordinary colour map, so both backends take them: GL's screen-space
# transmission really does show the liner through the water, which is where the
# tiled-pool identity reads.
for sx, sz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    if sx:
        wl = tp.Mesh(tp.PlaneGeometry(Z1 - Z0, 0.20),
                     liner_material(Z1 - Z0, 0.20, mosaic=_mosaic is not None))
        wl.rotate_y(-sx * math.pi / 2)
        wl.position.set((X1 - 0.002) if sx > 0 else (X0 + 0.002), 0.10,
                        0.5 * (Z0 + Z1))
    else:
        wl = tp.Mesh(tp.PlaneGeometry(X1 - X0, 0.20),
                     liner_material(X1 - X0, 0.20, mosaic=_mosaic is not None))
        if sz > 0:
            wl.rotate_y(math.pi)
        wl.position.set(0.5 * (X0 + X1), 0.10,
                        (Z1 - 0.002) if sz > 0 else (Z0 + 0.002))
    wl.receive_shadow = True
    scene.add(wl)

block_mat = standard_material(0x6d7480, 0.35, 0.45)
if obstacle_tris is not None:
    # The SAME triangles the field was baked from, so what you see is what the
    # fluid hits. Already in world coordinates from fit_obstacle, so the mesh
    # sits at the origin and only the paddle sweep moves it.
    _og = tp.BufferGeometry()
    _og.set_attribute("position", obstacle_tris.reshape(-1, 3))
    _og.compute_vertex_normals()
    block = tp.Mesh(_og, block_mat)
    block.position.set(0.0, 0.0, 0.0)
else:
    block = tp.Mesh(tp.BoxGeometry(BX1 - BX0, BY1, BZ1 - BZ0), block_mat)
    block.position.set(0.5 * (BX0 + BX1), 0.5 * BY1, 0.5 * (BZ0 + BZ1))
block.cast_shadow = True
block.receive_shadow = True
scene.add(block)

wall_mat = standard_material(0x9fa6ae, 0.28, 0.65)
WALL_H, WALL_T = 0.20, 0.025
for sx, sz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    if sx:
        w = tp.Mesh(tp.BoxGeometry(WALL_T, WALL_H, Z1 - Z0 + 2 * WALL_T), wall_mat)
        w.position.set(X1 + 0.5 * WALL_T if sx > 0 else X0 - 0.5 * WALL_T,
                       0.5 * WALL_H, 0.5 * (Z0 + Z1))
    else:
        w = tp.Mesh(tp.BoxGeometry(X1 - X0 + 2 * WALL_T, WALL_H, WALL_T), wall_mat)
        w.position.set(0.5 * (X0 + X1), 0.5 * WALL_H,
                       Z1 + 0.5 * WALL_T if sz > 0 else Z0 - 0.5 * WALL_T)
    w.cast_shadow = True
    w.receive_shadow = True
    scene.add(w)

# No ground plane on Vulkan. Wave facets tilted toward the camera reflect
# DOWNWARD, and refracted rays exiting the tank sweep BELOW it; a grey plane
# there answers "grey" to every one of them. With it gone the same rays miss,
# and an env miss samples the HDRI's below-horizon content at raw HDR radiance
# -- the pool hall's warm tiles and its real turquoise water. GL keeps the
# ground: its procedural sky is dull below the horizon, and its screen-space
# refraction is tuned with the plane in place.
if not VULKAN:
    _g = max(60.0, 3.0 * max(X1 - X0, Z1 - Z0))
    ground = tp.Mesh(tp.PlaneGeometry(_g, _g),
                     standard_material(0x848b94 if IS_REF else 0x3a4048, 0.8))
    ground.rotate_x(-math.pi / 2)
    ground.position.y = FLOOR - 0.032
    ground.receive_shadow = True
    scene.add(ground)

geometry = tp.BufferGeometry()
reg_pos = None
reg_nrm = None
stage_pos = None
stage_nrm = None
host_pos = host_nrm = hview_pos = hview_nrm = None
vk_slot = 0             # which pinned pair the GPU is filling
vk_pending = None       # (rows, ntris, slot) copied last frame, not yet uploaded
vk_interop = None       # (positions, normals) VkInteropArray pair, or None
vk_ntris = 0            # triangle count the in-render callback should expand
vk_route = "pinned host copy"
# Older threepp builds honour drawRange only on the line/points overlay, not on
# the mesh path -- the same builds that have no enable_vertex_interop. Detect
# one, get the other.
LEGACY_TAIL = VULKAN and not hasattr(renderer, "enable_vertex_interop")
tail_high = 0           # high-water row count, only used when LEGACY_TAIL
if LEGACY_TAIL:
    print("  note: this threepp build predates drawRange on the Vulkan mesh "
          "path -- padding the unused tail instead")

if POINTS:
    geometry.set_attribute("position", p0)
    geometry.set_attribute("color", np.tile(np.float32([0.05, 0.28, 0.42]), (N, 1)))
    pmat = tp.PointsMaterial()
    pmat.size = 1.6 * D
    pmat.size_attenuation = True
    pmat.vertex_colors = True
    water = tp.Points(geometry, pmat)
else:
    cap = MAX_TRIS * 3
    if VULKAN:
        stage_pos = wp.zeros(cap, dtype=wp.vec3, device=device)
        stage_nrm = wp.zeros(cap, dtype=wp.vec3, device=device)
        # Persistent PINNED host mirrors. Calling stage.numpy() per frame both
        # allocates a fresh multi-MB array and copies the whole capacity, not
        # just the live prefix. wp.copy with an explicit count DMAs only the
        # rows that changed, and a CPU array's .numpy() is a view, so the loop
        # allocates nothing. Two of each, so the DMA for frame N lands while
        # frame N-1 is being uploaded: the mandatory sync then waits on a copy
        # issued a whole frame ago and costs almost nothing. Geometry lags the
        # sim by one frame.
        host_pos = [wp.zeros(cap, dtype=wp.vec3, device="cpu", pinned=True)
                    for _ in range(2)]
        host_nrm = [wp.zeros(cap, dtype=wp.vec3, device="cpu", pinned=True)
                    for _ in range(2)]
        hview_pos = [a.numpy() for a in host_pos]
        hview_nrm = [a.numpy() for a in host_nrm]
    geometry.set_attribute("position", np.zeros((cap, 3), np.float32))
    geometry.set_attribute("normal", np.tile(np.float32([0, 1, 0]), (cap, 1)))
    # A marching-cubes soup has no natural parameterisation, so this geometry
    # never carried uvs. --debug-uv adds a dummy set to test whether their
    # ABSENCE is what stops the transmission branch running on this mesh.
    if cli_arg("--debug-uv", 0, int):
        geometry.set_attribute("uv", np.zeros((cap, 2), np.float32))
    geometry.set_draw_range(0, 3)
    wmat = tp.MeshStandardMaterial() if OPAQUE else tp.MeshPhysicalMaterial()
    # The backends tint transmitted light through different models, so the water
    # wants a different base colour on each. GL runs the glTF transmission chunk
    # and really does consume thickness and attenuation, so a tinted base is
    # part of a tuned whole. Vulkan's glass path multiplies everything seen
    # through the surface by the albedo ALONE (its Beer-Lambert chord measures
    # ~0 for this mesh, so attenuation contributes nothing), and colours are
    # sRGB, so a tinted base cuts most of the red out of the refracted image at
    # zero depth -- which reads as dark gel. A near-white base lets the liner
    # colour through.
    if OPAQUE:
        wmat.color = 0x3aa0c8
    else:
        wmat.color = cli_arg("--water-color",
                             0xcfeef5 if VULKAN else 0x8fd6e8,
                             lambda v: int(v, 0))
    # GL has no ray tracing: transmission is a screen-space pre-pass, so a
    # bright environment floods straight through and the water goes to milk.
    # Measured on the first wide frame: RGB 193/209/217, saturation 29. A
    # little roughness to spread the sun into a glitter track, and half the
    # environment, is what puts the colour back.
    wmat.roughness = cli_arg("--water-roughness", 0.04 if IS_REF else 0.05, float)
    # 0.25 for the wide pool, not 1.0: a full-strength environment on a
    # surface that big floods the far field pale and the glitter track
    # disappears into it. Always settable, so the look can be swept.
    wmat.env_map_intensity = ENV_I
    wmat.metalness = 0.0
    if not OPAQUE:
        wmat.transmission = cli_arg("--transmission", 1.0, float)
        wmat.ior = 1.333
        # GL-only, all four: the Vulkan glass path reads none of them for this
        # mesh. Kept because they are exactly what makes the GL water work.
        # How far you can SEE INTO the water. The volume term is
        # attenuation_color ^ (thickness / attenuation_distance), so 0.55 over
        # 0.30 is colour^1.83 -- about (0.02, 0.27, 0.36) for this teal, i.e.
        # red gone and two thirds of the rest with it. That is opaque by
        # construction, and no amount of light makes a floor visible through it.
        wmat.thickness = cli_arg("--water-thickness", 0.55 if IS_REF else (2.5 if VULKAN else 0.30), float)
        wmat.attenuation_color = tp.Color(0x1d7d92)
        # 0.30 m was tuned when GL's transmission contributed NOTHING, so the
        # absorption was free. With the winding corrected it contributes, and
        # colour^(0.55/0.30) eats ~98% of the red -- the water went dark and
        # refused to carry the floor's colour. Vulkan keeps 0.30: its chord is
        # traced, not the material thickness, and it already reads correctly.
        wmat.attenuation_distance = cli_arg(
            "--water-attenuation", (0.30 if VULKAN else 0.9) if IS_REF else
            (4.0 if VULKAN else 0.48), float)
        # GL, wide: 0.30 / 0.48 keeps the Beer exponent of the old 2.5 / 4.0
        # (0.625, so the tint is unchanged) with an eighth of the refraction
        # lever. GL offsets its screen-space refraction sample by thickness
        # along the refracted ray, so 2.5 m under 0.27 m of water threw every
        # normal error tens of pixels. Vulkan traces its chord and keeps 4.0.
        wmat.clearcoat = 0.25
        wmat.clearcoat_roughness = 0.12
    if cli_arg("--debug-glass", 0, int):
        # The exact material the isolated probe transmits with: white, smooth,
        # single-sided, thin, nothing else set. Same geometry, same scene --
        # this separates "the material config" from "the mesh".
        wmat = tp.MeshPhysicalMaterial()
        wmat.color = 0xffffff
        wmat.roughness = 0.0
        wmat.metalness = 0.0
        wmat.transmission = cli_arg("--transmission", 1.0, float)
        wmat.ior = 1.333
        # Side.Double, exactly like warp_water_balloon's water, which DOES
        # transmit with this same marching-cubes geometry. Leaving it
        # single-sided culls the surface (the winding is inward), which is what
        # invalidated the first version of this test.
        wmat.side = tp.Side.Double
        wmat.transparent = False
    else:
        # wp.MarchingCubes winds triangles INTO the density, so the water's
        # visible surface is BACK-facing. With Side.Double the shader does
        # normal *= faceDirection from gl_FrontFacing, which flips the normal on
        # exactly the fragments you can see -> NdotV clamps to 0 -> the
        # transmission chunk's Fresnel weight goes to 1 -> transmissionFactor is
        # exactly 0. That is why the water was never see-through.
        _side = cli_arg("--water-side", "double", str).lower()
        wmat.side = {"front": tp.Side.Front, "back": tp.Side.Back,
                     "double": tp.Side.Double}[_side]
    # NOT transparent: the transmissive bucket is selected by transmission > 0
    # and the shader forces alpha to 1, so flipping this only risks sort issues.
    wmat.transparent = False
    # --debug-plane-water swaps the marching-cubes surface for a flat quad at
    # the waterline, same material, same everything else. It exists to isolate
    # the GEOMETRY from the shading when transmission misbehaves.
    if cli_arg("--debug-plane-water", 0, int):
        _pg = tp.PlaneGeometry(X1 - X0, Z1 - Z0)
        water = tp.Mesh(_pg, wmat)
        water.rotate_x(-math.pi / 2)
        water.position.set(0.5 * (X0 + X1), FLOOR + 0.22, 0.5 * (Z0 + Z1))
    else:
        water = tp.Mesh(geometry, wmat)
    # A 34 m sheet casting into itself buys nothing and costs a shadow pass
    # over the whole pool.
    water.cast_shadow = IS_REF

water.frustum_culled = False          # the CPU-side bounds never see GPU writes
scene.add(water)

# --- the scale chain ----------------------------------------------------------
# A 34 m pool and a 3.4 km bay render identically when nothing in frame has a
# known size: the first wide shot came back looking like open ocean, which is
# pretty and says nothing about the thing the cluster was needed for. Apparent
# size halving along a RECEDING LINE is the cue that fixes it, and it beats any
# single prop -- a lone 1.8 m post at 33 m is four pixels.
#
# Two lines, both instanced (two draw calls for the lot): marker buoys at a
# 4 m pitch down one side, and a swimming-lane rope at a 0.5 m pitch down the
# other. The rope is the stronger cue because its spheres are small and dense,
# so the eye reads the spacing collapse directly; the buoys give the near field
# something with a legible diameter. Both ride the still waterline, and neither
# is simulated -- they are rulers, not physics.
MARKERS = cli_arg("--markers", 0 if IS_REF else 1, int)


def settled_waterline():
    """Where the free surface ends up, which is NOT --depth.

    --depth sets the FILL, and the beach then cuts a wedge out of it: the
    remaining water spreads over the same footprint and settles lower. At
    --wide 4 --beach 2 that is a ~3 cm drop, which floats 7.5 cm markers
    clear of the surface and makes them look airborne.

    Conserve volume instead. With a flat bed of length L and a beach of slope
    m rising from BEACH_TOE, the water volume at height h is
        V(h) = W * (L*h + h^2 / (2m))
    so h solves a quadratic. N*D^3 is the lattice volume actually seeded
    (already beach-cut), so this follows every flag automatically.
    """
    if not FLAT:
        return FLOOR + 0.22
    vol = N * D ** 3
    width = Z1 - Z0
    flat_len = max(BEACH_TOE - FILL_X0, 1e-6)
    if BEACH_SLOPE <= 0.0:
        return FLOOR + vol / (width * flat_len)
    a = 0.5 / BEACH_SLOPE
    b = flat_len
    c = -vol / width
    h = (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)
    return FLOOR + min(h, DEPTH)


if MARKERS:
    # The RENDERED surface is not the fill depth. DensitySurface blurs the
    # density field before marching cubes, which pulls the iso contour inward,
    # so the visible water sits roughly a cell or two below where the particles
    # are -- at --cell 2 that is 2-4 cm, half a lane-float diameter, and the
    # markers floated in mid-air. Sink them by a cell and a bit of their own
    # radius so they intersect the surface they are drawn against, not the one
    # the physics thinks it has.
    _wl = settled_waterline() - cli_arg("--marker-sink", 1.2 * CELL, float)

    # Both lines are laid out FROM THE CAMERA BACKWARD, not from a fixed end of
    # the tank. Reversing the shot moved the camera to +x, and a chain anchored
    # at X1 then begins behind the lens: at --wide 4 the first several lane
    # floats were behind the camera and the nearest buoy sat 0.8 m from it,
    # filling a tenth of the frame. Anchoring at the eye keeps the near element
    # at a fixed, legible distance whichever end the camera stands at, and the
    # far element stops clear of the piston stroke.
    _m_near = min(max(CAM_EYE[0] + 2.4 * CAM_DIR, X0 + 0.9), X1 - 0.4)
    _m_far = (X0 + 1.2) if CAM_FAR_END else (X1 - 0.6)
    _m_run = max(abs(_m_far - _m_near), 1e-6)

    # Buoys: 0.45 m across, half-submerged, every 4 m.
    _bstep = 4.0
    _nb = max(2, int(_m_run / _bstep) + 1)
    _bz = 0.79 * Z1
    buoys = tp.InstancedMesh(tp.SphereGeometry(0.225, 20, 14),
                             standard_material(0xff6a1f, 0.45), _nb)
    for i in range(_nb):
        _m = tp.Matrix4()
        _m.set_position(_m_near + i * _bstep * CAM_DIR, _wl - 0.25 * 0.225, _bz)
        buoys.set_matrix_at(i, _m)
    buoys.instance_matrix_needs_update()
    buoys.cast_shadow = True
    scene.add(buoys)

    # Lane rope: 0.15 m floats at a 0.5 m pitch, alternating blue and white the
    # way a real one is -- the alternation is what makes the pitch countable
    # once the spheres are only a pixel or two apart.
    _rstep = 0.5
    _nr = max(2, int(_m_run / _rstep) + 1)
    _rz = -0.68 * Z1
    rope = tp.InstancedMesh(tp.SphereGeometry(0.075, 14, 10),
                            standard_material(0xffffff, 0.5), _nr)
    for i in range(_nr):
        _m = tp.Matrix4()
        _m.set_position(_m_near + i * _rstep * CAM_DIR, _wl - 0.25 * 0.075, _rz)
        rope.set_matrix_at(i, _m)
        rope.set_color_at(i, tp.Color(0xf2f4f6 if (i // 2) % 2 == 0 else 0x1b6fb0))
    rope.instance_matrix_needs_update()
    rope.instance_color_needs_update()
    scene.add(rope)

    print(f"scale chain: {_nb} buoys @ {_bstep:g} m, {_nr} lane floats @ {_rstep:g} m, "
          f"waterline {_wl:.3f} m (fill depth {DEPTH:.3f})")

# The shot in one line, so a Slurm log says whether the wave could ever arrive.
_c = math.sqrt(9.81 * max(DEPTH, 1e-6))
# Under --cam-from all the eye sits OUTSIDE the tank, so the distance to it is
# not the interesting number: the whole pool is in frame, and what matters is
# when the front reaches the near wall. Clamp the eye into the tank to get it.
_reach = abs(min(max(CAM_EYE[0], X0), X1) - (BX1 if FLAT else FILL_X1))
print(f"shot: {CAM_FROM} "
      f"(eye {CAM_EYE[0]:.1f} {CAM_EYE[1]:.1f} {CAM_EYE[2]:.1f}, fov {CAM_FOV:g}), "
      f"wave speed {_c:.2f} m/s, front reaches the lens at t={_reach / _c:.1f} s "
      f"(warm-up {cli_arg('--warmup', 0.5 if IS_REF else 4.0, float):.1f} s "
      f"+ shot {VIDEO if VIDEO else SHOT_TIME:.1f} s)")




def vk_on_frame():
    """Fill the renderer's OWN vertex buffers, in place. Runs inside render().

    The renderer calls this once per frame, post-fence and pre-record, so the
    only thing this may do is write the imported arrays -- the marching-cubes
    output and density field are whatever build_surface() left behind a moment
    ago in frame().

    The synchronize is MANDATORY and is the whole contract: wp.launch is
    asynchronous on Warp's stream, and host ordering here is what sequences the
    CUDA write against the Vulkan frame that reads it (there is no shared
    semaphore).

    Only the LIVE triangles are expanded: the renderer's copies, BLAS build and
    raster draw all clamp to the drawRange write_surface published, so rows past
    vk_ntris are never consumed and shrinking frames need no re-degenerating.
    """
    if vk_ntris > 0:
        surface.expand(vk_ntris, vk_interop[0].array, vk_interop[1].array, sign=MC_SIGN,
                       flip_winding=MC_FLIP)
    wp.synchronize_device(device)


def arm_vulkan_interop():
    """Point the expand kernel straight at the renderer's vertex buffers. True if live.

    Must run AFTER the first render(): the renderer's record for a mesh (and so
    the allocation there is anything to export) is created on the frame the mesh
    is first drawn. Every failure falls back to the pinned-host route below,
    which needs neither the external-memory export nor the CUDA import.
    """
    global vk_interop, vk_route, stage_pos, stage_nrm, host_pos, host_nrm
    global hview_pos, hview_nrm
    if VkInteropArray is None or not hasattr(renderer, "enable_vertex_interop"):
        print("  note: this threepp build predates the CUDA->Vulkan vertex "
              "interop -- using the host route")
        return False
    h = renderer.enable_vertex_interop(water, vk_on_frame)
    if h is None:
        print("  note: vulkan vertex interop did not arm -- falling back to the "
              "host route (the renderer prints the reason on stderr)")
        return False
    (pos_handle, pos_bytes), (nrm_handle, nrm_bytes) = h
    cap = MAX_TRIS * 3
    try:
        # Tightly-packed float xyz on both sides, so wp.vec3 is the dtype -- see
        # the stride note in threepp/cuda_interop.py. The handles stay the
        # renderer's: never CloseHandle them here.
        vk_interop = (VkInteropArray(pos_handle, pos_bytes, wp.vec3, cap, device),
                      VkInteropArray(nrm_handle, nrm_bytes, wp.vec3, cap, device))
    except Exception as e:
        print(f"  note: CUDA import of the vulkan export failed ({e}) -- falling "
              f"back to the host route")
        renderer.disable_vertex_interop(water)
        vk_interop = None
        return False
    # Degenerate the WHOLE capacity once, as belt and braces. The renderer's
    # copies, BLAS build and raster draw all clamp to the drawRange, so nothing
    # SHOULD ever consume rows past the live surface -- but the exports are
    # fresh VRAM, and one launch here means a future consumer that forgets the
    # clamp reads a harmless off-screen point instead of uninitialised garbage
    # that happens to be finite.
    surface.expand(0, vk_interop[0].array, vk_interop[1].array, dim=MAX_TRIS, sign=MC_SIGN,
                   flip_winding=MC_FLIP)
    wp.synchronize_device(device)
    # The pinned-host staging is dead weight now. Dropping it also makes a
    # stray trip through the host branch fail loudly rather than quietly write
    # into buffers nothing uploads any more.
    stage_pos = stage_nrm = None
    host_pos = host_nrm = hview_pos = hview_nrm = None
    vk_route = "zero-copy CUDA -> Vulkan"
    # Ordered teardown: the CUDA mappings must go before the renderer frees the
    # Vulkan memory they point at. atexit runs while both are still alive;
    # interpreter shutdown alone would collect them in an arbitrary order.
    atexit.register(release_vulkan_interop)
    return True


def release_vulkan_interop():
    """Drop the CUDA mappings, then hand the mesh back to the CPU path."""
    global vk_interop
    if vk_interop is None:
        return
    pair, vk_interop = vk_interop, None
    for a in pair:
        a.close()
    renderer.disable_vertex_interop(water)


def write_surface(ntris):
    """Push marching-cubes output at the renderer."""
    global reg_pos, reg_nrm, vk_slot, vk_pending, vk_ntris
    if vk_interop is not None:
        # Zero copy: nothing to push. vk_on_frame() does the expand from inside
        # the renderer's frame, over exactly the surface built above; the
        # drawRange published here is what the renderer clamps its raster draw,
        # BLAS build and interop copies to.
        vk_ntris = ntris
        geometry.set_draw_range(0, 3 * ntris)
        return
    if VULKAN:
        # No CUDA/Vulkan interop is bound, so the triangles go out through host
        # memory -- live rows only, the drawRange keeps the renderer off the
        # stale tail.
        # Publish last frame's copy first. The sync is MANDATORY -- a copy into
        # PINNED host memory is asynchronous, unlike a pageable one, and without
        # it the upload reads a half-written buffer and the mesh tears. Waiting
        # here, on a copy issued a full frame ago, makes it nearly free. It also
        # guarantees the staging buffers are free before the expand below
        # reuses them.
        wp.synchronize_device(device)
        if vk_pending is not None:
            p_rows, p_tris, p_slot = vk_pending
            if LEGACY_TAIL:
                # This build's Vulkan mesh path ignores drawRange, so the rows
                # past the live surface keep LAST frame's triangles and render
                # as stale garbage. Collapse the shrinking tail to a point --
                # degenerate triangles rasterise to nothing -- and upload
                # through the high-water mark instead of the live count.
                global tail_high
                if p_rows < tail_high:
                    hview_pos[p_slot][p_rows:tail_high] = 0.0
                p_rows = tail_high = max(tail_high, p_rows)
            geometry.update_attribute("position", hview_pos[p_slot][:p_rows])
            geometry.update_attribute("normal", hview_nrm[p_slot][:p_rows])
            geometry.set_draw_range(0, 3 * p_tris)
        if ntris > 0:
            surface.expand(ntris, stage_pos, stage_nrm, sign=MC_SIGN, flip_winding=MC_FLIP)
            rows = ntris * 3
            wp.copy(host_pos[vk_slot], stage_pos, count=rows)
            wp.copy(host_nrm[vk_slot], stage_nrm, count=rows)
            vk_pending = (rows, ntris, vk_slot)
            vk_slot ^= 1
        return
    if reg_pos is None:
        if not hasattr(renderer, "gl_buffer_id"):
            raise SystemExit(
                "This threepp build has no GLRenderer.gl_buffer_id, so the GL "
                "zero-copy route is unavailable. Use --vulkan, or upgrade "
                "threepp.")
        pid = renderer.gl_buffer_id(geometry, "position")
        nid = renderer.gl_buffer_id(geometry, "normal")
        if pid is None or nid is None:
            return
        flags = wp.RegisteredGLBuffer.WRITE_DISCARD
        reg_pos = wp.RegisteredGLBuffer(int(pid), device, flags)
        reg_nrm = wp.RegisteredGLBuffer(int(nid), device, flags)
    if ntris > 0:
        dp = reg_pos.map(dtype=wp.vec3, shape=(MAX_TRIS * 3,))
        dn = reg_nrm.map(dtype=wp.vec3, shape=(MAX_TRIS * 3,))
        surface.expand(ntris, dp, dn, sign=MC_SIGN, flip_winding=MC_FLIP)
        reg_pos.unmap()
        reg_nrm.unmap()
    geometry.set_draw_range(0, 3 * ntris)


def refresh_surface():
    """Rebuild the render geometry from the current particle state."""
    if POINTS:
        wp.launch(shade_points, dim=N, device=device, inputs=[v, col])
        geometry.update_attribute("position", x.numpy())
        geometry.update_attribute("color", col.numpy())
        return N
    bx0, bx1 = paddle_extent(sim_time)
    # The mesh's vertices already carry its rest position, so it tracks the raw
    # sweep offset; the box's geometry is centred on its own origin, so it
    # tracks the swept centre.
    block.position.x = (bx0 - BX0) if obstacle_tris is not None else 0.5 * (bx0 + bx1)
    ntris = build_surface()
    write_surface(ntris)
    return ntris


def frame():
    """One simulation frame plus a refreshed surface."""
    sim_step()
    return refresh_surface()


timed_tris = []


def timed_step():
    """sim_step + surface, each synced and timed; for --bench and --frames."""
    t0 = time.perf_counter()
    sim_step()
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    ntris = build_surface()
    write_surface(ntris)
    wp.synchronize_device(device)
    timed_tris.append(ntris)
    return t1 - t0, time.perf_counter() - t1


def save_frame(path):
    if VULKAN:
        renderer.save_frame(scene, camera, path)   # renders + reads back
    else:
        renderer.render(scene, camera)
        renderer.save_frame(path)


BACKEND = "vulkan" if VULKAN else "opengl"

# Neither backend's vertex buffers exist until the renderer has drawn the mesh
# once: gl_buffer_id returns None before the first render, and the Vulkan record
# enable_vertex_interop exports from is built on the frame the mesh is first
# seen. So both zero-copy routes are armed by polling after one throwaway render.
renderer.render(scene, camera)

if VULKAN and not POINTS:
    if INTEROP:
        arm_vulkan_interop()
    print(f"vulkan surface route: {vk_route}")

if FRAMES:
    # Real interactive cadence: the canvas owns submit/present, so renderer
    # .render() records exactly one frame instead of driving flush_frames.
    WARMUP = 30
    acc = [0.0, 0.0, 0.0]
    state = {"n": 0}

    def timed():
        sim, surf = timed_step()
        t2 = time.perf_counter()
        renderer.render(scene, camera)
        if state["n"] >= WARMUP:
            acc[0] += sim
            acc[1] += surf
            acc[2] += time.perf_counter() - t2
        state["n"] += 1

    t_wall = time.perf_counter()
    for _ in range(FRAMES + WARMUP):
        canvas.animate_once(timed)
    wall = time.perf_counter() - t_wall
    ms = 1000.0 / FRAMES
    tot = sum(acc) * ms
    print(f"frames {N:,} particles [{BACKEND}] "
          f"substeps={SUBSTEPS} iters={ITERATIONS} rho={RHO_CHEB}: "
          f"sim {acc[0] * ms:.2f} | surface {acc[1] * ms:.2f} | render {acc[2] * ms:.2f} "
          f"= {tot:.2f} ms ({1000.0 / tot:.0f} fps)  wall {wall * 1000 / (FRAMES + WARMUP):.1f} ms")
elif BENCH:
    for _ in range(45):
        frame()
        renderer.render(scene, camera)
    timed_tris.clear()
    bench_loop(timed_step, lambda: renderer.render(scene, camera),
               ("sim", "surface"), warmup=0, timed=120,
               label=f"{N:,} particles [{BACKEND}]")
    print(f"  {int(np.mean(timed_tris)):,} tris/frame")
    save_frame("warp_fluid.png")
elif SHOT:
    nt = 0
    total = int(round(SHOT_TIME * 60))
    # The Vulkan pipeline is TEMPORAL: probe GI, the reflection denoiser and the
    # upscaler all converge over frames, and a single render after the sim loop
    # would capture frame ONE of all of them -- probes still dark, no history.
    # Render the last stretch of frames so the shot is the CONVERGED image.
    warm = min(total, 90) if VULKAN else 1
    for i in range(total):
        nt = frame()
        if i >= total - warm:
            renderer.render(scene, camera)
    if "--tilt" in sys.argv and nt > 0:
        # The rendered normals, measured instead of judged: each upward vertex
        # normal against the mean of its 5 cm bin. A wave's own normal turns by
        # far less than that across 5 cm, so "noise" is nearly all shot noise
        # and "wave" is the tilt of the bin means. No pixel size, MSAA or
        # camera enters it -- which is exactly what fooled the look check that
        # sent the first film to the cluster.
        _p = wp.zeros(3 * nt, dtype=wp.vec3, device=device)
        _n = wp.zeros(3 * nt, dtype=wp.vec3, device=device)
        surface.expand(nt, _p, _n, sign=MC_SIGN, flip_winding=MC_FLIP)
        P, Q = _p.numpy(), _n.numpy()
        m = ((Q[:, 1] > 0.5) & (P[:, 0] > X0 + 0.6) & (P[:, 0] < X1 - 0.6)
             & (np.abs(P[:, 2]) < Z1 - 0.3))
        if int(m.sum()) < 100:
            print("tilt: too few upward vertices to measure")
        else:
            P, Q = P[m], Q[m]
            key = (np.floor(P[:, 0] / 0.05).astype(np.int64) * 100003
                   + np.floor(P[:, 2] / 0.05).astype(np.int64))
            _, inv = np.unique(key, return_inverse=True)
            inv = inv.ravel()
            M = np.zeros((int(inv.max()) + 1, 3))
            np.add.at(M, inv, Q)
            M /= np.linalg.norm(M, axis=1, keepdims=True)
            noise = np.arccos(np.clip((Q * M[inv]).sum(1), -1.0, 1.0))
            wave = np.arccos(np.clip(M[:, 1], -1.0, 1.0))
            nr = float(np.sqrt((noise ** 2).mean()))
            wr = float(np.sqrt((wave ** 2).mean()))
            print(f"tilt: noise rms {nr:.4f} rad | wave rms (5 cm bins) {wr:.4f} rad | "
                  f"SNR {wr / max(nr, 1e-9):.2f} | {int(m.sum()):,} verts")
    save_frame("warp_fluid.png")
    print(f"simulated {SHOT_TIME:.1f} s, {nt:,} triangles [{BACKEND}], wrote warp_fluid.png")
elif VIDEO:
    # Offline video: every simulated frame is rendered, so the temporal pipeline
    # is always converged and no vsync, screen capture or window is involved --
    # the same path works on a headless cluster node.
    #
    # Straight down an x264 pipe, not a PNG per frame. At 3840x2160 a 10 s film
    # is 600 frames, and the PNG round trip costs more than the render does:
    # encode, write, read back, decode, for pixels that were already in memory.
    # It is also tens of GB of intermediates on a filesystem shared with
    # everyone else on the cluster.
    total = int(round(VIDEO * 60))
    warm = int(round(cli_arg("--warmup", 0.5 if IS_REF else 4.0, float) * 60))
    # --out-size WxH renders at --size and area-downsamples in the encoder. The
    # EGL pbuffer has no MSAA (a local Canvas gets 4x, which is why previews
    # look cleaner than the cluster), and averaging several fully shaded
    # samples per output pixel is the only anti-aliasing that averages SHADING.
    OUT_SIZE = cli_arg("--out-size", "", str)
    enc = Encoder("warp_fluid.mp4", WIDTH, HEIGHT, 60, crf=16, preset="slow",
                  vf=(f"scale={OUT_SIZE.lower().replace('x', ':')}:flags=area"
                      if OUT_SIZE else None))
    t0 = time.perf_counter()
    for i in range(warm):
        # A replay has nothing to settle and every dumped frame is wanted, so
        # the temporal passes converge on dump frame 0 instead of consuming
        # frames: load it once, then re-render it.
        if REPLAY and i:
            refresh_surface()
        elif VULKAN or i >= warm - 1:
            frame()
        else:
            # GL draws only the last warm-up frame, so surfacing the others is
            # pure cost: the first Idun film ran 240 full-grid marching-cubes
            # passes for pictures nobody drew. The sim never reads the surface,
            # so skipping it changes no particle.
            sim_step()
        # Vulkan's probes, denoiser and upscaler need history; GL does not.
        if i >= warm - (30 if VULKAN else 1):
            renderer.render(scene, camera)
    for k in range(total):
        if not (REPLAY and k == 0):     # replay: frame 0 is already loaded
            frame()
        camera_at(k / 60.0)
        renderer.render(scene, camera)
        enc.send(renderer.read_pixels())
        if k % 60 == 0:
            el = time.perf_counter() - t0
            print(f"  frame {k}/{total}  ({el:.0f}s elapsed, "
                  f"{wp.get_mempool_used_mem_high(device) / 2 ** 30:.1f} GB peak)",
                  flush=True)
    enc.close()
    print(f"rendered {total} frames in {time.perf_counter() - t0:.0f}s -> "
          f"warp_fluid.mp4 ({os.path.getsize('warp_fluid.mp4') // 1024} KB)")
else:
    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    canvas.on_window_resize(resize_handler(camera, renderer))

    ui = tp.ImguiContext(canvas, renderer) if tp.HAS_IMGUI else None
    route = vk_route if VULKAN else ("host copy" if POINTS else "zero-copy CUDA -> GL")
    # The window is vsync'd, so the frame rate imgui reports is capped at the
    # display and only tells the whole story once the sim is slower than a refresh.
    # The phase columns are the uncapped truth, so they are worth the two syncs it
    # takes to attribute them -- but a sync per frame would serialize the very
    # pipeline being measured, so only every SAMPLE_EVERY-th frame is sampled.
    SAMPLE_EVERY = 30
    prof = {"sim": 0.0, "surface": 0.0, "render": 0.0, "tris": 0, "n": 0}

    def draw_hud():
        tp.imgui.set_next_window_pos(10, 10)
        tp.imgui.set_next_window_size(272, 0)
        tp.imgui.begin("Warp PBF fluid")
        tp.imgui.text(f"{N:,} particles   {prof['tris']:,} tris")
        tp.imgui.text(f"{BACKEND}  |  {route}")
        tp.imgui.separator()
        tp.imgui.text(f"{tp.imgui.get_framerate():6.1f} fps   (vsync capped)")
        tp.imgui.separator()
        total = prof["sim"] + prof["surface"] + prof["render"]
        for name in ("sim", "surface", "render"):
            tp.imgui.text(f"  {name:<8}{prof[name]:6.2f} ms")
        if total > 0.0:
            tp.imgui.text(f"  {'uncapped':<8}{total:6.2f} ms  = {1000.0 / total:.0f} fps")
        tp.imgui.separator()
        tp.imgui.text(f"solver: {SUBSTEPS} substeps x {ITERATIONS} iters, rho {RHO_CHEB:g}")
        tp.imgui.text("drag = orbit, scroll = zoom")
        tp.imgui.end()

    def animate():
        prof["n"] += 1
        if ui is not None:
            # Don't orbit while the pointer is over the panel -- a drag that starts
            # on a widget belongs to imgui, not to the camera.
            controls.enabled = not ui.want_capture_mouse
        if prof["n"] % SAMPLE_EVERY:
            prof["tris"] = frame()
            controls.update()
            renderer.render(scene, camera)
            return
        t0 = time.perf_counter()
        sim_step()
        wp.synchronize_device(device)
        t1 = time.perf_counter()
        prof["tris"] = refresh_surface()
        wp.synchronize_device(device)
        t2 = time.perf_counter()
        controls.update()
        renderer.render(scene, camera)
        # No sync after render: this is the CPU-side submit, the same figure
        # --bench reports, not the GPU's own draw time.
        t3 = time.perf_counter()
        prof["sim"] = (t1 - t0) * 1000.0
        prof["surface"] = (t2 - t1) * 1000.0
        prof["render"] = (t3 - t2) * 1000.0

    def animate_with_ui():
        animate()
        ui.render(draw_hud)

    canvas.animate(animate if ui is None else animate_with_ui)
