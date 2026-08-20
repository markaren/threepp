"""A liquid surface simulated with NVIDIA Warp and rendered by threepp.

Position Based Fluids on the GPU, surfaced with GPU marching cubes, drawn as a
refracting water mesh. A dam break collapses into a basin, then a paddle sweeps
back and forth through it so the water never settles into a still pond.

Nothing crosses host memory in the render path, on either backend: the
marching-cubes triangles and their normals are written straight into the
renderer's own vertex buffers -- GLRenderer.gl_buffer_id + wp.RegisteredGLBuffer
on OpenGL, VulkanRenderer.enable_vertex_interop + threepp.cuda_interop (a CUDA
import of the Vulkan allocation) on Vulkan. GL publishes the live triangle count
with set_draw_range; the Vulkan mesh path has no drawRange, so its unused tail is
padded with degenerate triangles instead.

    pip install warp-lang
    python warp_fluid.py                 # window; drag to orbit, Esc quits
    python warp_fluid.py --n 400000      # more particles
    python warp_fluid.py --shot 6        # headless PNG at t=6s
    python warp_fluid.py --bench         # timed phase breakdown
    python warp_fluid.py --vulkan        # Vulkan renderer (RT reflections)
    python warp_fluid.py --vulkan --no-interop   # ... via pinned host memory (A/B)
    python warp_fluid.py --points        # draw particles instead of the surface
    python warp_fluid.py --probe 3       # print an energy audit, no window

Needs a CUDA device: the zero-copy surface path is CUDA/OpenGL interop.
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
from threepp.cuda_interop import VkInteropArray


def cli_arg(flag, default, cast):
    if flag in sys.argv:
        k = sys.argv.index(flag)
        if k + 1 < len(sys.argv) and not sys.argv[k + 1].startswith("--"):
            return cast(sys.argv[k + 1])
    return default


N = cli_arg("--n", 340_000, int)
BENCH = "--bench" in sys.argv
SHOT = "--shot" in sys.argv
SHOT_TIME = cli_arg("--shot", 6.0, float)
POINTS = "--points" in sys.argv
PROBE = "--probe" in sys.argv
# Vulkan gets ray-traced reflections and a dedicated water/glass shader. Its zero
# copy is not gl_buffer_id but enable_vertex_interop: the renderer exports its
# own position/normal allocations, CUDA imports them, and the expand kernel
# writes them in place. Its deferred mesh path still ignores
# BufferGeometry::drawRange -- VulkanCoreRecord.cpp honours drawRange only in the
# line/points overlay branch -- so the unused tail is padded with degenerate
# triangles rather than left out of the draw call.
VULKAN = "--vulkan" in sys.argv
# Force the pinned-host round trip on Vulkan (the route this demo used before the
# interop existed). Kept as the A/B baseline, and as the automatic fallback on
# any machine where the export or the CUDA import does not come up.
INTEROP = "--no-interop" not in sys.argv
# --frames N times the REAL interactive frame model (canvas.animate_once).
# --bench is headless, where Vulkan runs flush_frames full GPU frames per
# render() call, so its render figure is not what a window costs.
FRAMES = cli_arg("--frames", 0, int)
OPAQUE = "--opaque" in sys.argv   # debug: render the surface as a
                                  # plain lit solid, to separate a
                                  # geometry problem from a shading one

# --- fluid parameters ---------------------------------------------------------

D = 0.009                    # rest particle spacing (m); the single
                             # resolution knob -- H, WALL_EPS, MAX_DP,
                             # rest density, CELL and the iso level all
                             # derive from it
H = 2.0 * D                  # SPH support radius (~38 neighbours)
MASS = 1.0                   # unit mass; rest density is measured from a lattice
DT = 1.0 / 60.0
SUBSTEPS = cli_arg("--substeps", 2, int)
ITERATIONS = cli_arg("--iters", 4, int)               # density-constraint projections per substep
S_CORR_N = 4.0               # EPS_CFM / S_CORR_K are derived from the lattice
                             # below: hand-guessed values are off by orders of
                             # magnitude because the SPH kernels carry a 1/h^9
                             # scale, so the published constants do not transfer.
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
# an unbounded half-space silently teleports every particle beneath it onto its
# surface, which reads as the whole fluid exploding on frame one.
BX0, BX1 = 0.15, 0.42
BZ0, BZ1 = -0.30, 0.30
BY1 = 0.26

# The dam-break column occupies [X0, FILL_X1] and collapses in the first second.
# After that the paddle keeps the water moving forever: a recycling spout was
# tried first and could not work, because a spout dense enough to surface under
# marching cubes needs far more inflow than it can drain, so it just parks a
# supersaturated blob in mid-air.
FILL_X1 = -0.25
PADDLE_AMP = 0.46            # sweep amplitude (m)
PADDLE_PERIOD = 2.0          # seconds per full stroke (Froude ~1.2:
                             # supercritical, so it throws a bow wave)

# --- SPH kernel coefficients --------------------------------------------------

POLY6 = 315.0 / (64.0 * math.pi * H ** 9)
SPIKY = -45.0 / (math.pi * H ** 6)


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
def collide(p: wp.vec3, bx0: float, bx1: float) -> wp.vec3:
    x = wp.min(wp.max(p[0], X0 + WALL_EPS), X1 - WALL_EPS)
    z = wp.min(wp.max(p[2], Z0 + WALL_EPS), Z1 - WALL_EPS)
    y = wp.max(p[1], FLOOR + WALL_EPS)
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
            dt: float, bx0: float, bx1: float):
    i = wp.tid()
    vi = v[i] + wp.vec3(0.0, GRAVITY, 0.0) * dt
    sp = wp.length(vi)
    if sp > V_MAX:
        vi = vi * (V_MAX / sp)
    v[i] = vi
    xp[i] = collide(x[i] + vi * dt, bx0, bx1)


@wp.kernel
def solve_lambda(xp: wp.array(dtype=wp.vec3),
                 grid: wp.uint64,
                 rho0: float,
                 eps_cfm: float,
                 lam: wp.array(dtype=float)):
    # spatially-sorted thread order: neighbour gathers hit cache.
    # This beats caching an explicit neighbour list, which was tried
    # and measured slower: indexing a cached list forces threads back
    # into original array order and loses exactly this locality.
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
    # the surface implodes at tens of m/s. Cohesion is s_corr's job.
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
                bx0: float,
                bx1: float,
                out: wp.array(dtype=wp.vec3)):
    # spatially-sorted thread order: neighbour gathers hit cache.
    # This beats caching an explicit neighbour list, which was tried
    # and measured slower: indexing a cached list forces threads back
    # into original array order and loses exactly this locality.
    i = wp.hash_grid_point_id(grid, wp.tid())
    if i < 0:
        return
    p = xp[i]
    li = lam[i]
    dp = wp.vec3(0.0, 0.0, 0.0)
    q = wp.hash_grid_query(grid, p, H)
    for j in q:
        rv = p - xp[j]
        r2 = wp.dot(rv, rv)
        if r2 < H * H and r2 > 1.0e-12:
            # artificial pressure repels near-coincident particles, which is
            # what keeps droplets round instead of stringy
            s_corr = -k_corr * wp.pow(w_poly6(r2) / w_dq, S_CORR_N)
            dp += w_spiky_grad(rv, wp.sqrt(r2)) * (li + lam[j] + s_corr)
    d = dp * (MASS / rho0 * relax)
    dl = wp.length(d)
    if dl > MAX_DP:
        d = d * (MAX_DP / dl)
    out[i] = collide(p + d, bx0, bx1)


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
    # spatially-sorted thread order: neighbour gathers hit cache.
    # This beats caching an explicit neighbour list, which was tried
    # and measured slower: indexing a cached list forces threads back
    # into original array order and loses exactly this locality.
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
    # spatially-sorted thread order: neighbour gathers hit cache.
    # This beats caching an explicit neighbour list, which was tried
    # and measured slower: indexing a cached list forces threads back
    # into original array order and loses exactly this locality.
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
def shade_points(v: wp.array(dtype=wp.vec3), col: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    f = wp.min(wp.length(v[i]) / 2.6, 1.0)
    col[i] = wp.vec3(0.05, 0.28, 0.42) * (1.0 - f) + wp.vec3(0.92, 0.97, 1.0) * f


# --- surfacing: particles -> density grid -> marching cubes -------------------

CELL = 1.18 * D              # surface grid spacing (coarser than D, then blurred)
GX0, GY0, GZ0 = X0 - 0.05, FLOOR - 0.035, Z0 - 0.05
NGX = int((X1 + 0.05 - GX0) / CELL) + 1
NGY = int((0.56 - GY0) / CELL) + 1   # paddle spray never gets near this
NGZ = int((Z1 + 0.05 - GZ0) / CELL) + 1


@wp.func
def sample(field: wp.array3d(dtype=float), gx: float, gy: float, gz: float) -> float:
    i = int(wp.floor(gx))
    j = int(wp.floor(gy))
    k = int(wp.floor(gz))
    if i < 0 or j < 0 or k < 0 or i > NGX - 2 or j > NGY - 2 or k > NGZ - 2:
        return 0.0
    fx = gx - float(i)
    fy = gy - float(j)
    fz = gz - float(k)
    c00 = field[i, j, k] * (1.0 - fx) + field[i + 1, j, k] * fx
    c10 = field[i, j + 1, k] * (1.0 - fx) + field[i + 1, j + 1, k] * fx
    c01 = field[i, j, k + 1] * (1.0 - fx) + field[i + 1, j, k + 1] * fx
    c11 = field[i, j + 1, k + 1] * (1.0 - fx) + field[i + 1, j + 1, k + 1] * fx
    a = c00 * (1.0 - fy) + c10 * fy
    b = c01 * (1.0 - fy) + c11 * fy
    return a * (1.0 - fz) + b * fz


@wp.kernel
def splat(x: wp.array(dtype=wp.vec3), field: wp.array3d(dtype=float)):
    # Trilinear deposit: 8 atomics per particle. Cheap enough that the blur,
    # not the splat, is what makes the surface smooth.
    t = wp.tid()
    p = x[t]
    gx = (p[0] - GX0) / CELL
    gy = (p[1] - GY0) / CELL
    gz = (p[2] - GZ0) / CELL
    i = int(wp.floor(gx))
    j = int(wp.floor(gy))
    k = int(wp.floor(gz))
    if i < 0 or j < 0 or k < 0 or i > NGX - 2 or j > NGY - 2 or k > NGZ - 2:
        return
    fx = gx - float(i)
    fy = gy - float(j)
    fz = gz - float(k)
    for a in range(2):
        wa = wp.where(a == 0, 1.0 - fx, fx)
        for b in range(2):
            wb = wp.where(b == 0, 1.0 - fy, fy)
            for c in range(2):
                wc = wp.where(c == 0, 1.0 - fz, fz)
                wp.atomic_add(field, i + a, j + b, k + c, wa * wb * wc)


@wp.kernel
def blur_axis(src: wp.array3d(dtype=float), dst: wp.array3d(dtype=float), axis: int):
    i, j, k = wp.tid()
    di = wp.where(axis == 0, 1, 0)
    dj = wp.where(axis == 1, 1, 0)
    dk = wp.where(axis == 2, 1, 0)
    im = wp.max(i - di, 0)
    jm = wp.max(j - dj, 0)
    km = wp.max(k - dk, 0)
    ip = wp.min(i + di, NGX - 1)
    jp = wp.min(j + dj, NGY - 1)
    kp = wp.min(k + dk, NGZ - 1)
    dst[i, j, k] = 0.25 * src[im, jm, km] + 0.5 * src[i, j, k] + 0.25 * src[ip, jp, kp]


@wp.kernel
def expand(verts: wp.array(dtype=wp.vec3),
           indices: wp.array(dtype=wp.int32),
           field: wp.array3d(dtype=float),
           ntris: int,
           out_pos: wp.array(dtype=wp.vec3),
           out_nrm: wp.array(dtype=wp.vec3)):
    # Python has no index-buffer binding, so de-index into a triangle soup and
    # give every vertex a smooth normal from the density-field gradient. A lit
    # mesh with no normal attribute renders BLACK, so this is not optional.
    t = wp.tid()
    if t >= ntris:
        # Collapse unused triangles to a point (zero area, rasterises nothing).
        # Vulkan ignores drawRange for meshes, so the tail must be padded rather
        # than simply left out of the draw call.
        for c in range(3):
            out_pos[t * 3 + c] = wp.vec3(0.0, -50.0, 0.0)
            out_nrm[t * 3 + c] = wp.vec3(0.0, 1.0, 0.0)
        return
    for c in range(3):
        p = verts[indices[t * 3 + c]]
        gx = (p[0] - GX0) / CELL
        gy = (p[1] - GY0) / CELL
        gz = (p[2] - GZ0) / CELL
        g = wp.vec3(sample(field, gx + 1.0, gy, gz) - sample(field, gx - 1.0, gy, gz),
                    sample(field, gx, gy + 1.0, gz) - sample(field, gx, gy - 1.0, gz),
                    sample(field, gx, gy, gz + 1.0) - sample(field, gx, gy, gz - 1.0))
        l = wp.length(g)
        # density rises inward, so the outward surface normal is -grad
        n = wp.where(l > 1.0e-9, g * (-1.0 / l), wp.vec3(0.0, 1.0, 0.0))
        out_pos[t * 3 + c] = p
        out_nrm[t * 3 + c] = n


# --- particle initialisation --------------------------------------------------

wp.init()
device = wp.get_preferred_device()

nx = int((FILL_X1 - X0 - 2.0 * WALL_EPS) / D)
nz = int((Z1 - Z0 - 2.0 * WALL_EPS) / D)
ny = max(1, N // (nx * nz))
ix, iy, iz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
p0 = np.stack([
    X0 + WALL_EPS + (ix.ravel() + 0.5) * D,
    FLOOR + WALL_EPS + (iy.ravel() + 0.5) * D,
    Z0 + WALL_EPS + (iz.ravel() + 0.5) * D,
], axis=-1).astype(np.float32)
# drop the particles that would start inside the obstacle
keep = ~((p0[:, 0] > BX0 - WALL_EPS) & (p0[:, 0] < BX1 + WALL_EPS) &
         (p0[:, 2] > BZ0 - WALL_EPS) & (p0[:, 2] < BZ1 + WALL_EPS) &
         (p0[:, 1] < BY1 + WALL_EPS))
p0 = p0[keep]
rng = np.random.default_rng(17)
p0 = (p0 + rng.uniform(-0.06 * D, 0.06 * D, p0.shape)).astype(np.float32)
N = len(p0)

# Rest density measured from the lattice itself, so the constraint is calibrated
# to the actual packing rather than a hand-guessed constant.
off = np.arange(-3, 4) * D
gxl, gyl, gzl = np.meshgrid(off, off, off, indexing="ij")
r2_lattice = (gxl ** 2 + gyl ** 2 + gzl ** 2).ravel()
r2_lattice = r2_lattice[r2_lattice < H * H]
RHO0 = float(MASS * (POLY6 * np.maximum(H * H - r2_lattice, 0.0) ** 3).sum())
W_DQ = float(POLY6 * (H * H - S_CORR_DQ ** 2) ** 3)

# The constraint gradients inherit the kernel's 1/h^9 scale, so published
# constants (eps ~ 1e-6, k ~ 0.1) are meaningless here. Derive both from the
# measured lattice: soften lambda by 5% of the rest gradient energy, and size
# artificial pressure at ~30% of lambda-at-10%-compression.
_r = np.sqrt(r2_lattice[r2_lattice > 1e-12])
_g = (MASS / RHO0) * abs(SPIKY) * (H - _r) ** 2
SG2_REST = float((_g ** 2).sum())
EPS_CFM = 0.05 * SG2_REST
_ratio_typ = (POLY6 * (H * H - D * D) ** 3) / W_DQ
S_CORR_K = 0.30 * (0.10 / SG2_REST) / _ratio_typ ** S_CORR_N

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
lam = wp.zeros(N, dtype=float, device=device)
omega = wp.zeros(N, dtype=wp.vec3, device=device)
vtmp = wp.zeros(N, dtype=wp.vec3, device=device)
col = wp.zeros(N, dtype=wp.vec3, device=device)
grid = wp.HashGrid(128, 128, 128, device)
grid.reserve(N)

field = wp.zeros((NGX, NGY, NGZ), dtype=float, device=device)
field2 = wp.zeros((NGX, NGY, NGZ), dtype=float, device=device)
mc = wp.MarchingCubes(
    NGX, NGY, NGZ,
    domain_bounds_lower_corner=wp.vec3(GX0, GY0, GZ0),
    domain_bounds_upper_corner=wp.vec3(GX0 + float(NGX - 1) * CELL,
                                       GY0 + float(NGY - 1) * CELL,
                                       GZ0 + float(NGZ - 1) * CELL),
)

sim_time = 0.0
frame_no = 0


def paddle_extent(t):
    """World x-extent of the paddle at time t."""
    c = PADDLE_AMP * math.sin(2.0 * math.pi * t / PADDLE_PERIOD)
    return BX0 + c, BX1 + c


def sim_step():
    """Advance one rendered frame of fluid."""
    global sim_time, frame_no
    dt = DT / SUBSTEPS
    for _ in range(SUBSTEPS):
        bx0, bx1 = paddle_extent(sim_time)
        wp.launch(predict, dim=N, device=device, inputs=[x, v, xp, dt, bx0, bx1])
        grid.build(points=xp, radius=H)
        a, b = xp, xtmp
        for _ in range(ITERATIONS):
            wp.launch(solve_lambda, dim=N, device=device,
                      inputs=[a, grid.id, RHO0, EPS_CFM, lam])
            wp.launch(solve_delta, dim=N, device=device,
                      inputs=[a, lam, grid.id, RHO0, W_DQ, S_CORR_K,
                              JACOBI_RELAX, bx0, bx1, b])
            a, b = b, a
        wp.launch(finalize, dim=N, device=device, inputs=[x, a, v, dt])
        sim_time += dt
    wp.launch(curl_and_viscosity, dim=N, device=device,
              inputs=[x, v, grid.id, RHO0, omega, vtmp])
    wp.launch(vorticity_confine, dim=N, device=device,
              inputs=[x, omega, vtmp, grid.id, DT, v])
    frame_no += 1


_clamped = False


def _note_clamp(got):
    # Silent truncation would just look like a chunk of missing water.
    global _clamped
    _clamped = True
    print(f"  note: marching cubes produced {got:,} triangles, clamped to "
          f"{MAX_TRIS:,} -- raise --max-tris if the surface looks cut off")


def build_surface():
    """Splat, blur, marching cubes; returns the triangle count."""
    field.zero_()
    wp.launch(splat, dim=N, device=device, inputs=[x, field])
    # Ping-pong the two grids instead of copying one onto the other: assign()
    # costs a full-grid copy per pass, which dominated the surfacing budget.
    # Six passes is even, so the result lands back in `field`.
    a, b = field, field2
    for _ in range(2):
        for axis in range(3):
            wp.launch(blur_axis, dim=(NGX, NGY, NGZ), device=device,
                      inputs=[a, b, axis])
            a, b = b, a
    mc.surface(field, ISO)
    got = mc.indices.shape[0] // 3
    if got > MAX_TRIS and not _clamped:
        _note_clamp(got)
    return min(got, MAX_TRIS)


if PROBE:
    print(f"{'t':>6} {'|v|mean':>8} {'|v|max':>8} {'y_mean':>7} {'y_max':>7} {'tris':>9}")
    for f in range(int(60 * cli_arg("--probe", 3.0, float))):
        sim_step()
        if f % 12 == 0:
            vv = np.linalg.norm(v.numpy(), axis=1)
            yy = x.numpy()[:, 1]
            print(f"{f/60.0:6.2f} {vv.mean():8.3f} {vv.max():8.3f} {yy.mean():7.3f} "
                  f"{yy.max():7.3f} {build_surface():9,d}")
    sys.exit(0)


# --- procedural HDR sky (numpy -> Radiance .hdr -> RGBELoader) -----------------
# Adapted from examples/pbr_showcase.py so the demo downloads nothing. The
# environment map is not decoration: with transmission, the sky reflection and
# the Fresnel rim are most of what makes the surface read as water.

SUN_DIR = np.array([0.42, 0.45, 0.79])
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)


def _encode_rgbe(rgb):
    rgb = np.maximum(np.asarray(rgb, np.float64), 0.0)
    m = rgb.max(axis=2)
    mask = m >= 1e-32
    safe = np.where(mask, m, 1.0)
    mant, exp = np.frexp(safe)
    scale = np.where(mask, mant * 256.0 / safe, 0.0)
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    for c in range(3):
        out[..., c] = np.clip(rgb[..., c] * scale, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(mask, np.clip(exp + 128, 0, 255), 0).astype(np.uint8)
    return out


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
    sky = np.where((y < 0)[..., None], np.array([0.16, 0.17, 0.19]), sky)

    d = np.stack([xx, y, zz], axis=-1)
    ang = np.arccos(np.clip((d * SUN_DIR).sum(-1), -1.0, 1.0))
    core = np.exp(-(ang / math.radians(1.6)) ** 2)
    halo = np.exp(-(ang / math.radians(12.0)) ** 2)
    sky = sky + (core * 70.0 + halo * 5.0)[..., None] * np.array([1.0, 0.95, 0.86])

    rgbe = _encode_rgbe(sky)
    if rgbe[0, 0, 0] == 2 and rgbe[0, 0, 1] == 2 and rgbe[0, 0, 2] < 128:
        rgbe[0, 0, 0] = 3
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (HH, W))
        f.write(rgbe.tobytes())
    return path


# --- threepp scene ------------------------------------------------------------

if VULKAN and not tp.vulkan_available():
    print("vulkan not available on this machine; falling back to OpenGL")
    VULKAN = False

canvas = tp.Canvas("threepp x warp - fluid", width=1280, height=800,
                   antialiasing=4, headless=(SHOT or BENCH) and not FRAMES)
if VULKAN:
    renderer = tp.VulkanRenderer(canvas)
    # a smaller ceiling: every frame refreshes the whole padded range (a host
    # upload on the fallback route, a device-to-device copy + BLAS refit on the
    # interop one), so capacity costs bandwidth here in a way it does not on GL
    MAX_TRIS = min(MAX_TRIS, cli_arg("--max-tris", 260_000, int))
else:
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = True
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 1.15

scene = tp.Scene()
env = tp.RGBELoader().load(make_sky_hdr(
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
tank_mat = tp.MeshStandardMaterial()
tank_mat.color = 0x8a8f96
tank_mat.roughness = 0.55
tank_mat.metalness = 0.1
floor_mesh = tp.Mesh(tp.BoxGeometry(X1 - X0 + 0.06, 0.03, Z1 - Z0 + 0.06), tank_mat)
floor_mesh.position.set(0.5 * (X0 + X1), FLOOR - 0.015, 0.5 * (Z0 + Z1))
floor_mesh.receive_shadow = True
scene.add(floor_mesh)

block_mat = tp.MeshStandardMaterial()
block_mat.color = 0x6d7480
block_mat.roughness = 0.35
block_mat.metalness = 0.45
block = tp.Mesh(tp.BoxGeometry(BX1 - BX0, BY1, BZ1 - BZ0), block_mat)
block.position.set(0.5 * (BX0 + BX1), 0.5 * BY1, 0.5 * (BZ0 + BZ1))
block.cast_shadow = True
block.receive_shadow = True
scene.add(block)

wall_mat = tp.MeshStandardMaterial()
wall_mat.color = 0x9fa6ae
wall_mat.roughness = 0.28
wall_mat.metalness = 0.65
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

ground_mat = tp.MeshStandardMaterial()
ground_mat.color = 0x848b94
ground_mat.roughness = 0.8
ground_mat.metalness = 0.0
ground = tp.Mesh(tp.PlaneGeometry(60, 60), ground_mat)
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
tail_hwm = 0            # high-water mark: how far the padded tail must reach
vk_interop = None       # (positions, normals) VkInteropArray pair, or None
vk_ntris = 0            # triangle count the in-render callback should expand
vk_route = "pinned host copy"

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
        # just the live prefix -- the same mistake that made the nebula hitch.
        # wp.copy with an explicit count DMAs only the rows that changed, and
        # a CPU array's .numpy() is a view, so the loop allocates nothing.
        # Two of each, so the DMA for frame N lands while frame N-1 is being
        # uploaded: the mandatory sync then waits on a copy issued a whole frame
        # ago and costs almost nothing. Geometry lags the sim by one frame.
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
    wmat.color = 0x8fd6e8 if not OPAQUE else 0x3aa0c8
    wmat.roughness = 0.04
    wmat.metalness = 0.0
    if not OPAQUE:
        wmat.transmission = 1.0      # real screen-space refraction on the GL path
        wmat.ior = 1.333
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
    only thing this may do is write the imported arrays -- mc.verts, mc.indices
    and `field` are whatever build_surface() left behind a moment ago in frame().

    The synchronize is MANDATORY and is the whole contract: wp.launch is
    asynchronous on Warp's stream, and host ordering here is what sequences the
    CUDA write against the Vulkan frame that reads it (there is no shared
    semaphore). It replaces -- rather than adds to -- the sync the pinned-host
    route already had to pay.
    """
    global tail_hwm
    n = max(vk_ntris, tail_hwm)   # shrinking frames must re-degenerate the tail
    tail_hwm = vk_ntris           # they left real triangles in
    if n > 0:
        wp.launch(expand, dim=n, device=device,
                  inputs=[mc.verts, mc.indices, field, vk_ntris,
                          vk_interop[0].array, vk_interop[1].array])
    wp.synchronize_device(device)


def arm_vulkan_interop():
    """Point `expand` straight at the renderer's vertex buffers. True if live.

    Must run AFTER the first render(): the renderer's record for a mesh (and so
    the allocation there is anything to export) is created on the frame the mesh
    is first drawn. Every failure falls back to the pinned-host route below,
    which needs neither the external-memory export nor the CUDA import.
    """
    global vk_interop, vk_route, stage_pos, stage_nrm, host_pos, host_nrm
    global hview_pos, hview_nrm
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
    # Degenerate the WHOLE capacity once. The exports are fresh VRAM, so every
    # triangle the surface never reaches would otherwise be uninitialised garbage
    # on the first frame -- and garbage that happens to be finite gets past the
    # renderer's sanitize pass. After this the per-frame launch only has to cover
    # the high-water mark, exactly as the host route does. (mc.verts/mc.indices
    # are still None here; expand never touches them on the pad path, and warp
    # passes a None array as an empty one.)
    wp.launch(expand, dim=MAX_TRIS, device=device,
              inputs=[mc.verts, mc.indices, field, 0,
                      vk_interop[0].array, vk_interop[1].array])
    wp.synchronize_device(device)
    # The pinned-host staging is dead weight now: ~37 MB of pinned host memory
    # and 19 MB of device memory at the default capacity. Dropping it also makes
    # a stray trip through the host branch fail loudly rather than quietly write
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
    global reg_pos, reg_nrm, tail_hwm, vk_slot, vk_pending, vk_ntris
    if vk_interop is not None:
        # Zero copy: nothing to push. vk_on_frame() does the expand from inside
        # the renderer's frame, over exactly the surface built above.
        vk_ntris = ntris
        return
    if VULKAN:
        # No CUDA/Vulkan interop is bound, so the triangles go out through host
        # memory. Expand over the high-water mark so shrinking frames overwrite
        # last frame's leftovers with degenerates.
        # Publish last frame's copy first. The sync is MANDATORY -- a copy into
        # PINNED host memory is asynchronous, unlike a pageable one, and without
        # it the upload reads a half-written buffer and the mesh tears
        # (measured: 40/40 stale reads without, 0/40 with). Waiting here, on a
        # copy issued a full frame ago, makes it nearly free. It also guarantees
        # the staging buffers are free before the expand below reuses them.
        wp.synchronize_device(device)
        if vk_pending is not None:
            p_rows, p_tris, p_slot = vk_pending
            geometry.update_attribute("position", hview_pos[p_slot][:p_rows])
            geometry.update_attribute("normal", hview_nrm[p_slot][:p_rows])
            geometry.set_draw_range(0, 3 * p_tris)
        n = max(ntris, tail_hwm)
        tail_hwm = ntris
        if n > 0:
            wp.launch(expand, dim=n, device=device,
                      inputs=[mc.verts, mc.indices, field, ntris,
                              stage_pos, stage_nrm])
            rows = n * 3
            wp.copy(host_pos[vk_slot], stage_pos, count=rows)
            wp.copy(host_nrm[vk_slot], stage_nrm, count=rows)
            vk_pending = (rows, ntris, vk_slot)
            vk_slot ^= 1
        return
    if reg_pos is None:
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
        wp.launch(expand, dim=ntris, device=device,
                  inputs=[mc.verts, mc.indices, field, ntris, dp, dn])
        reg_pos.unmap()
        reg_nrm.unmap()
    geometry.set_draw_range(0, 3 * ntris)


def frame():
    """One simulation frame plus a refreshed surface."""
    sim_step()
    if POINTS:
        wp.launch(shade_points, dim=N, device=device, inputs=[v, col])
        geometry.update_attribute("position", x.numpy())
        geometry.update_attribute("color", col.numpy())
        return N
    bx0, bx1 = paddle_extent(sim_time)
    block.position.x = 0.5 * (bx0 + bx1)
    ntris = build_surface()
    write_surface(ntris)
    return ntris


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
    sim = surf = rend = 0.0
    state = {"n": 0}

    def timed():
        t0 = time.perf_counter()
        sim_step()
        wp.synchronize_device(device)
        t1 = time.perf_counter()
        ntris = build_surface()
        write_surface(ntris)
        wp.synchronize_device(device)
        t2 = time.perf_counter()
        renderer.render(scene, camera)
        t3 = time.perf_counter()
        if state["n"] >= 30:          # skip warm-up
            nonlocal_add(t1 - t0, t2 - t1, t3 - t2)
        state["n"] += 1

    def nonlocal_add(a, b, c):
        global sim, surf, rend
        sim += a
        surf += b
        rend += c

    t_wall = time.perf_counter()
    for _ in range(FRAMES + 30):
        canvas.animate_once(timed)
    wall = time.perf_counter() - t_wall
    ms = 1000.0 / FRAMES
    tot = (sim + surf + rend) * ms
    print(f"frames {N:,} particles [{'vulkan' if VULKAN else 'opengl'}] "
          f"substeps={SUBSTEPS} iters={ITERATIONS}: "
          f"sim {sim * ms:.2f} | surface {surf * ms:.2f} | render {rend * ms:.2f} "
          f"= {tot:.2f} ms ({1000.0 / tot:.0f} fps)  wall {wall * 1000 / (FRAMES + 30):.1f} ms")
elif BENCH:
    for _ in range(45):
        frame()
        renderer.render(scene, camera)
    sim = surf = rend = 0.0
    TIMED = 120
    tris = 0
    for _ in range(TIMED):
        t0 = time.perf_counter()
        sim_step()
        wp.synchronize_device(device)
        t1 = time.perf_counter()
        ntris = build_surface()
        write_surface(ntris)
        wp.synchronize_device(device)
        t2 = time.perf_counter()
        renderer.render(scene, camera)
        tris += ntris
        sim += t1 - t0
        surf += t2 - t1
        rend += time.perf_counter() - t2
    ms = 1000.0 / TIMED
    total = (sim + surf + rend) * ms
    print(f"bench {N:,} particles, {tris // TIMED:,} tris/frame: "
          f"sim {sim * ms:.2f} | surface {surf * ms:.2f} | render {rend * ms:.2f} "
          f"= {total:.2f} ms/frame ({1000.0 / total:.0f} fps) "
          f"[{'vulkan' if VULKAN else 'opengl'}]")
    if VULKAN:
        renderer.save_frame(scene, camera, "warp_fluid.png")
    else:
        renderer.save_frame("warp_fluid.png")
elif SHOT:
    nt = 0
    for _ in range(int(round(SHOT_TIME * 60))):
        nt = frame()
    if VULKAN:
        renderer.render(scene, camera)      # drives flush_frames full frames
        renderer.save_frame(scene, camera, "warp_fluid.png")
    else:
        renderer.render(scene, camera)
        renderer.save_frame("warp_fluid.png")
    print(f"simulated {SHOT_TIME:.1f} s, {nt:,} triangles "
          f"[{'vulkan' if VULKAN else 'opengl'}], wrote warp_fluid.png")
else:
    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        renderer.set_size(w, h)

    canvas.on_window_resize(on_resize)

    def animate():
        frame()
        controls.update()
        renderer.render(scene, camera)

    canvas.animate(animate)
