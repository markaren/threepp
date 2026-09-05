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
    python warp_fluid.py --n 400000      # more particles
    python warp_fluid.py --shot 6        # headless PNG at t=6s
    python warp_fluid.py --bench         # timed phase breakdown
    python warp_fluid.py --vulkan        # Vulkan renderer (RT reflections)
    python warp_fluid.py --vulkan --no-interop   # ... through pinned host memory
    python warp_fluid.py --points        # draw particles instead of the surface
    python warp_fluid.py --probe 3       # print an energy audit, no window
    python warp_fluid.py --iters 4 --rho 0   # plain Jacobi, no Chebyshev acceleration
    python warp_fluid.py --obstacle part.stl # collide an arbitrary mesh, not the box
    python warp_fluid.py --obstacle part.stl --obstacle-height 0.35 --sdf-res 128

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
import math
import os
import sys
import tempfile
import time

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import (DensitySurface, bench_loop, cli_arg, encode_png_sequence,
                         find_ffmpeg, pbf_constants,
                         resize_handler, standard_material, write_radiance_hdr)
try:
    from threepp.cuda_interop import VkInteropArray
except ImportError:                  # older threepp builds have no CUDA<->Vulkan interop
    VkInteropArray = None

# Warp compiles with fast_math off, so every sqrt and divide in the neighbour
# loops is a multi-instruction IEEE sequence instead of one hardware op. Nothing
# in the fluid is subnormal-sensitive (the guards sit at 1e-9 and 1e-12), so the
# approximate forms are safe. Module-scoped: the surfacing kernels are untouched.
wp.set_module_options({"fast_math": True})

N = cli_arg("--n", 340_000, int)
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
OPAQUE = "--opaque" in sys.argv   # debug: render the surface as a plain lit
                                  # solid, to separate a geometry problem
                                  # from a shading one

# --- fluid parameters ---------------------------------------------------------

D = 0.009                    # rest particle spacing (m); the single
                             # resolution knob -- H, WALL_EPS, MAX_DP,
                             # rest density, CELL and the iso level all
                             # derive from it
H = 2.0 * D                  # SPH support radius
MASS = 1.0                   # unit mass; rest density is measured from a lattice
DT = 1.0 / 60.0
SUBSTEPS = cli_arg("--substeps", 2, int)
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
XSPH_C = 0.08                # viscosity: how much a particle adopts neighbour flow
VORTICITY = 0.22             # curl restored after projection damps it
V_MAX = 5.0                  # velocity clamp
MAX_DP = 0.35 * D            # per-iteration position-correction bound
GRAVITY = -9.81

# --- tank geometry ------------------------------------------------------------

X0, X1 = -1.05, 1.05         # tank interior
Z0, Z1 = -0.55, 0.55
FLOOR = 0.0
WALL_EPS = 0.6 * D           # keep particle centres this far off a wall

# A block on the tank floor for the jet to break over. It must be a FINITE box:
# an unbounded half-space would teleport every particle beneath it onto its
# surface.
BX0, BX1 = 0.15, 0.42
BZ0, BZ1 = -0.30, 0.30
BY1 = 0.26

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

# The dam-break column occupies [X0, FILL_X1] and collapses in the first second.
# After that the paddle keeps the water moving.
FILL_X1 = -0.25
PADDLE_AMP = 0.46            # sweep amplitude (m)
PADDLE_PERIOD = 2.0          # seconds per full stroke; fast enough to throw a bow wave

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

CELL = 1.00 * D              # surface grid spacing (then blurred). Marching
                             # cubes cannot emit a sheet thinner than roughly
                             # the combined splat+blur kernel width -- thinner
                             # water does not thin, it VANISHES -- so the cell
                             # is fine and the second blur round is narrowed,
                             # at the cost of more triangles.
GX0, GY0, GZ0 = X0 - 0.05, FLOOR - 0.035, Z0 - 0.05
NGX = int((X1 + 0.05 - GX0) / CELL) + 1
NGY = int((0.56 - GY0) / CELL) + 1   # paddle spray never gets near this
NGZ = int((Z1 + 0.05 - GZ0) / CELL) + 1

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

nx = int((FILL_X1 - X0 - 2.0 * WALL_EPS) / D)
nz = int((Z1 - Z0 - 2.0 * WALL_EPS) / D)
ny = max(1, N // (nx * nz))
ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
p0 = np.stack([
    X0 + WALL_EPS + (ix.ravel() + 0.5) * D,
    FLOOR + WALL_EPS + (iy.ravel() + 0.5) * D,
    Z0 + WALL_EPS + (iz.ravel() + 0.5) * D,
], axis=-1).astype(np.float32)
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

# A cell fully inside the fluid collects (CELL/D)^3 particles; the surface sits
# near half of that, so the iso-threshold follows the resolution automatically.
ISO = 0.5 * (CELL / D) ** 3
MAX_TRIS = cli_arg("--max-tris", 700_000, int)

print(f"fluid: {N:,} particles on {device} | grid {NGX}x{NGY}x{NGZ} cell={CELL} "
      f"iso={ISO:.2f}\n       rho0={RHO0:.4g} sum_grad2={SG2_REST:.4g} "
      f"eps={EPS_CFM:.3g} k_corr={S_CORR_K:.3g}")

x = wp.array(p0, dtype=wp.vec3, device=device)
v = wp.zeros(N, dtype=wp.vec3, device=device)
xp = wp.zeros(N, dtype=wp.vec3, device=device)
xtmp = wp.zeros(N, dtype=wp.vec3, device=device)
xprev = wp.zeros(N, dtype=wp.vec3, device=device)   # Chebyshev's k-1 iterate
lam = wp.zeros(N, dtype=float, device=device)
omega = wp.zeros(N, dtype=wp.vec3, device=device)
vtmp = wp.zeros(N, dtype=wp.vec3, device=device)
col = wp.zeros(N, dtype=wp.vec3, device=device)
grid = wp.HashGrid(128, 128, 128, device)
grid.reserve(N)

# Round one of the blur stays binomial so grid-aligned noise is killed
# outright; round two is narrowed, which buys thinness without a new noise
# source.
surface = DensitySurface((GX0, GY0, GZ0), CELL, (NGX, NGY, NGZ), device,
                         blur=(0.25, 0.125))

sim_time = 0.0
frame_no = 0


def paddle_offset(t):
    """Sweep displacement of the paddle along x at time t."""
    return PADDLE_AMP * math.sin(2.0 * math.pi * t / PADDLE_PERIOD)


def paddle_extent(t):
    """World x-extent of the paddle box at time t."""
    c = paddle_offset(t)
    return BX0 + c, BX1 + c


def sim_step():
    """Advance one rendered frame of fluid."""
    global sim_time, frame_no
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
    for f in range(int(60 * cli_arg("--probe", 3.0, float))):
        sim_step()
        if f % 12 == 0:
            wp.launch(measure_compression, dim=N, device=device,
                      inputs=[x, grid.id, RHO0, err])
            e = err.numpy()
            comp_acc.append(e.mean())
            vv = np.linalg.norm(v.numpy(), axis=1)
            yy = x.numpy()[:, 1]
            print(f"{f/60.0:6.2f} {vv.mean():8.3f} {vv.max():8.3f} {yy.mean():7.3f} "
                  f"{yy.max():7.3f} {e.mean()*100:7.3f} {e.max()*100:7.2f} "
                  f"{build_surface():9,d}")
    print(f"mean compression over run: {np.mean(comp_acc)*100:.3f}%  "
          f"(iters={ITERATIONS} rho={RHO_CHEB} relax={JACOBI_RELAX})")
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

canvas = tp.Canvas("threepp x warp - fluid", width=1280, height=800,
                   antialiasing=4,
                   headless=(SHOT or BENCH or VIDEO > 0) and not FRAMES)
if VULKAN:
    renderer = tp.VulkanRenderer(canvas)
else:
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = True
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 0.95 if VULKAN else 1.15

scene = tp.Scene()
# The indoor-pool HDRI is Vulkan-only: the ray-traced path turns its bright
# warm interior into reflections, refracted light and sparkle. GL's screen-space
# transmission just floods with it and washes the water out, so GL keeps the
# procedural sky.
_pool_hdr = fetch_asset(POOL_HDR_URL, "threepp_indoor_pool_2k.hdr") if VULKAN else None
env = tp.RGBELoader().load(_pool_hdr if _pool_hdr else make_sky_hdr(
    os.path.join(tempfile.gettempdir(), "threepp_fluid_sky.hdr")))
scene.environment = env
scene.background = env

camera = tp.PerspectiveCamera(46, canvas.aspect(), 0.01, 100)
camera.position.set(1.20, 0.63, 1.34)
camera.look_at(-0.02, 0.07, 0.0)

sun = tp.DirectionalLight(0xfff3e0, 2.6)
sun.position.set(2.4, 3.2, 4.2)
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
                       # Large rather than small: the ray-traced refraction
                       # samples the map at LOD 0, one ray per pixel, so fine
                       # tiles shatter into dark speckle under the rippled
                       # surface.


def liner_material(w, h, mosaic):
    m = standard_material(0xffffff if mosaic else 0x4ec9de, 0.5)
    if mosaic:
        t = tp.TextureLoader().load(_mosaic, tp.ColorSpace.SRGB)
        t.wrap_s = t.wrap_t = tp.TextureWrapping.Repeat
        t.repeat = tp.Vector2(w / TILE, h / TILE)
        m.map = t
    return m


# Mosaic on the WALLS, plain aqua on the FLOOR. The floor is what the body of
# the water refracts, and this mosaic's saturated navy average drags the whole
# pool dark and busy; the plain aqua floor keeps the water luminous while the
# walls carry the tiled-pool identity at the rim and waterline.
liner = tp.Mesh(tp.PlaneGeometry(X1 - X0, Z1 - Z0),
                liner_material(X1 - X0, Z1 - Z0, mosaic=False))
liner.rotate_x(-math.pi / 2)
liner.position.set(0.5 * (X0 + X1), FLOOR + 0.0015, 0.5 * (Z0 + Z1))
liner.receive_shadow = True
scene.add(liner)

# The walls get the liner too. Refracted rays that miss the floor hit the tank's
# inner faces, and grey walls put grey right back into the water body.
for sx, sz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
    if sx:
        wl = tp.Mesh(tp.PlaneGeometry(Z1 - Z0, 0.20),
                     liner_material(Z1 - Z0, 0.20, mosaic=VULKAN and _mosaic is not None))
        wl.rotate_y(-sx * math.pi / 2)
        wl.position.set((X1 - 0.002) if sx > 0 else (X0 + 0.002), 0.10,
                        0.5 * (Z0 + Z1))
    else:
        wl = tp.Mesh(tp.PlaneGeometry(X1 - X0, 0.20),
                     liner_material(X1 - X0, 0.20, mosaic=VULKAN and _mosaic is not None))
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
    ground = tp.Mesh(tp.PlaneGeometry(60, 60), standard_material(0x848b94, 0.8))
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
        wmat.color = 0xcfeef5 if VULKAN else 0x8fd6e8
    wmat.roughness = 0.04
    wmat.metalness = 0.0
    if not OPAQUE:
        wmat.transmission = 1.0      # real screen-space refraction on the GL path
        wmat.ior = 1.333
        # GL-only, all four: the Vulkan glass path reads none of them for this
        # mesh. Kept because they are exactly what makes the GL water work.
        wmat.thickness = 0.55
        wmat.attenuation_color = tp.Color(0x1d7d92)
        wmat.attenuation_distance = 0.30
        wmat.clearcoat = 0.25
        wmat.clearcoat_roughness = 0.12
    wmat.side = tp.Side.Double
    # NOT transparent: the transmissive bucket is selected by transmission > 0
    # and the shader forces alpha to 1, so flipping this only risks sort issues.
    wmat.transparent = False
    water = tp.Mesh(geometry, wmat)
    water.cast_shadow = True

water.frustum_culled = False          # the CPU-side bounds never see GPU writes
scene.add(water)


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
        surface.expand(vk_ntris, vk_interop[0].array, vk_interop[1].array)
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
    surface.expand(0, vk_interop[0].array, vk_interop[1].array, dim=MAX_TRIS)
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
            surface.expand(ntris, stage_pos, stage_nrm)
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
        surface.expand(ntris, dp, dn)
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
    save_frame("warp_fluid.png")
    print(f"simulated {SHOT_TIME:.1f} s, {nt:,} triangles [{BACKEND}], wrote warp_fluid.png")
elif VIDEO:
    # Offline video: every simulated frame is rendered and saved, so the
    # temporal pipeline is always converged and there is no vsync, screen
    # capture, or window involved -- the same path works on a headless server.
    # Frames land in a temp dir; ffmpeg (if present) muxes warp_fluid.mp4.
    outdir = tempfile.mkdtemp(prefix="warp_fluid_frames_")
    total = int(round(VIDEO * 60))
    warm = 30 if VULKAN else 1        # let the dam break settle history before frame 0
    t0 = time.perf_counter()
    for i in range(warm):
        frame()
        renderer.render(scene, camera)
    for k in range(total):
        frame()
        save_frame(os.path.join(outdir, f"f{k:05d}.png"))
        if k % 60 == 0:
            el = time.perf_counter() - t0
            print(f"  frame {k}/{total}  ({el:.0f}s elapsed)", flush=True)
    print(f"rendered {total} frames in {time.perf_counter() - t0:.0f}s -> {outdir}")
    ff = find_ffmpeg()
    if ff:
        encode_png_sequence(os.path.join(outdir, "f%05d.png"), "warp_fluid.mp4", 60,
                            crf=18, ffmpeg=ff)
        print(f"wrote warp_fluid.mp4 ({VIDEO:.0f}s @ 60fps, "
              f"{os.path.getsize('warp_fluid.mp4') // 1024} KB)")
    else:
        print("ffmpeg not found -- frames left as PNGs in", outdir)
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
