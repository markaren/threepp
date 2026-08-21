"""How many deformable fish can one GPU convey? A Newton VBD benchmark.

Answers the only question worth asking before building a fish-handling
simulator: at what count does a tetrahedral FEM fish stop running in real time.
Everything else here exists to make that number honest.

The fish is whatever mesh you point --fish at, tetrahedralised into a coarse
cage. The conveyor is a bed of rollers -- kinematic cylinders spun about their
own axis, which is the one conveying mechanism that needs no tricks: a cylinder
rotating about its long axis occupies the same volume forever, so contact reads
the true surface velocity omega*r and nothing ever has to be teleported back.
The solver is Newton's SolverVBD (Vertex Block Descent, stable Neo-Hookean
tets), which is the solver a from-scratch effort would converge on anyway.

--fish IS REQUIRED and has no default. It takes any model threepp's ModelLoader
reads -- .usdz, .usd/.usda/.usdc, .obj, .stl, .gltf/.glb, .dae -- and nothing
about this benchmark is fish-specific beyond the auto-orientation, so any
closed solid works. The scans this was developed against are not
redistributable, which is why there is no bundled asset to fall back on.

    pip install newton
    python newton_fish_conveyor.py --fish cod.usdz --graph           # sanity check
    python newton_fish_conveyor.py --fish cod.usdz --graph --count 64
    python newton_fish_conveyor.py --fish cod.usdz --graph --sweep   # 1,4,16,64 table
    python newton_fish_conveyor.py --fish cod.usdz --graph --sweep --sweep-to 256 \
        --max-particles 800000
    python newton_fish_conveyor.py --fish cod.usdz --graph --count 32 --self-contact
    python newton_fish_conveyor.py --fish cod.usdz --count 16 --view # NOT the perf path
    python newton_fish_conveyor.py --fish cod.usdz --tet-res 24      # finer cage
    python newton_fish_conveyor.py --fish cod.usdz --young 3e4 --poisson 0.45
    python newton_fish_conveyor.py --fish cod.usdz --fish-length 0.6 # rescale to 60 cm

Reported per count: solve milliseconds per frame, the realtime factor at 60 Hz,
tet/particle counts, VRAM, whether the fish actually conveyed (mean speed along
the belt vs. the belt's own speed), and whether anything went unstable. The
conveying number is a physics check, not a perf number -- if it reads ~0 the
timings are still valid but the contact model is not transmitting drive.

TETRAHEDRALISATION is a voxel lattice carved by the mesh's own signed distance
field, split 5-tets-per-cell with alternating parity so the result is
conforming. No external mesher, no dependency, and the resolution is a single
knob -- which is what you want when the point is to sweep cost against fidelity.
It also happens to be the right shape for production: a coarse simulation cage
with the high-resolution scan skinned onto it, rather than simulating 15k
vertices of scan directly.

The fish is auto-oriented: longest extent along the belt, thinnest extent
vertical, i.e. lying on its side the way a fish lies on a real conveyor.

ALWAYS PASS --graph. Without CUDA graph capture this is launch-overhead bound,
not solve bound: one 290-tet fish measured 54 ms/frame ungraphed and 6.2 ms
graphed. Ungraphed numbers say nothing about how many fish fit.

Needs a CUDA device. --device cpu runs but is not a meaningful measurement.
The sweep stops at --max-particles rather than running until it OOMs, and the
SDF bake is launched in slabs so no single dispatch is long enough to trip a
display-driver timeout. Sustained GPU load on a thin laptop is its own risk --
keep an eye on the machine rather than walking away from a long sweep.
"""
import math
import os
import sys
import time

import numpy as np
import warp as wp

try:
    import newton
except ImportError:
    sys.exit("newton is not installed -- pip install newton (needs a CUDA GPU, driver 545+)")

sys.path.insert(0, __file__.rsplit("examples", 1)[0])
try:
    import threepp as tp
except ImportError:
    sys.exit("threepp is not importable -- the .usdz is read through threepp's ModelLoader")


def cli_arg(flag, default, cast):
    if flag in sys.argv:
        k = sys.argv.index(flag)
        if k + 1 < len(sys.argv) and not sys.argv[k + 1].startswith("--"):
            return cast(sys.argv[k + 1])
    return default


FISH = cli_arg("--fish", "", str)            # required -- see the module docstring
COUNT = cli_arg("--count", 1, int)
SWEEP = "--sweep" in sys.argv
SWEEP_TO = cli_arg("--sweep-to", 64, int)    # raise deliberately; see MAX_PARTICLES
# Hard ceiling on model size. This is a laptop with 8 GiB of VRAM whose sustained
# CPU+GPU draw has taken the whole machine down once; a sweep that runs until it
# OOMs is not a measurement, it is a stress test. Raise it on purpose or not at all.
MAX_PARTICLES = cli_arg("--max-particles", 400_000, int)
SDF_SLAB = cli_arg("--sdf-slab", 8, int)     # grid rows per SDF dispatch
TET_RES = cli_arg("--tet-res", 32, int)      # cells along the fish's longest axis
ITERATIONS = cli_arg("--iterations", 10, int)# VBD iterations per substep
SUBSTEPS = cli_arg("--substeps", 4, int)
FRAMES = cli_arg("--frames", 120, int)       # timed frames
WARMUP = cli_arg("--warmup", 30, int)        # settle + kernel compile, untimed
GRAPH = "--graph" in sys.argv
VIEW = "--view" in sys.argv
SELF_CONTACT = "--self-contact" in sys.argv
BELT_SPEED = cli_arg("--speed", 0.5, float)  # m/s along +X
ROLLER_R = cli_arg("--roller-radius", 0.05, float)
MU = cli_arg("--mu", 0.45, float)            # plastic module belt, not steel
YOUNG = cli_arg("--young", 5.0e4, float)     # Pa -- fish flesh is soft
POISSON = cli_arg("--poisson", 0.45, float)  # nearly incompressible
DENSITY = cli_arg("--density", 1050.0, float)
FISH_LENGTH = cli_arg("--fish-length", 0.0, float)  # 0 = keep the scan's own size
DEVICE = cli_arg("--device", "cuda:0", str)
FPS = 60.0


# ── The fish: load, orient, tetrahedralise ───────────────────────────────────

def load_triangles(path):
    """Every triangle of the model, welded into one (verts, faces) pair."""
    root = tp.ModelLoader().load(path)
    meshes = []

    def walk(n):
        if getattr(n, "geometry", None) is not None:
            meshes.append(n)
        for c in n.children:
            walk(c)

    walk(root)
    if not meshes:
        sys.exit(f"no meshes in {path}")

    verts, faces = [], []
    for m in meshes:
        g = m.geometry
        p = np.asarray(g.get_attribute("position"), dtype=np.float32)
        idx = g.get_index()
        idx = np.arange(len(p), dtype=np.int64) if idx is None else np.asarray(idx, dtype=np.int64)
        faces.append(idx.reshape(-1, 3) + len(verts))
        verts.append(p)
    return np.concatenate(verts), np.concatenate(faces)


def orient_fish(v):
    """Longest extent -> +X (along the belt), thinnest -> +Z (up).

    A fish on a conveyor lies on its side: its lateral (thinnest) axis vertical,
    its length along travel. Sorting the bbox extents gets that without knowing
    anything about how the scan was authored.
    """
    ext = v.max(0) - v.min(0)
    order = np.argsort(ext)              # [thinnest, middle, longest]
    perm = [order[2], order[1], order[0]]  # -> [X=longest, Y=middle, Z=thinnest]
    v = v[:, perm]
    # A permutation can be a reflection; keep it a rotation so the scan is not
    # mirrored (a fish is not symmetric, and neither is a fillet).
    if np.linalg.det(np.eye(3)[perm]) < 0:
        v = v * np.float32([1, 1, -1])
    return v


def is_closed(faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    _, cnt = np.unique(np.sort(e, 1), axis=0, return_counts=True)
    return bool((cnt == 2).all())


@wp.kernel
def _sdf_kernel(mesh: wp.uint64, origin: wp.vec3, h: float, max_dist: float, i0: int,
                out: wp.array3d(dtype=wp.float32)):
    di, j, k = wp.tid()
    i = i0 + di
    p = origin + wp.vec3((float(i) + 0.5) * h, (float(j) + 0.5) * h, (float(k) + 0.5) * h)
    q = wp.mesh_query_point_sign_winding_number(mesh, p, max_dist)
    d = max_dist
    if q.result:
        cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
        d = wp.length(p - cp) * q.sign
    out[i, j, k] = d


# The two 5-tet splits of a cube. Corner c has bits (dx, dy, dz) = (c&1, c>>1&1,
# c>>2&1). Alternating them by cell parity is what makes the lattice conforming:
# neighbouring cells then agree on the diagonal of every shared face.
_EVEN = np.array([[0, 1, 2, 4], [1, 2, 3, 7], [1, 4, 5, 7], [2, 4, 6, 7], [1, 2, 4, 7]])
_ODD = np.array([[0, 1, 3, 5], [0, 2, 3, 6], [0, 4, 5, 6], [3, 5, 6, 7], [0, 3, 5, 6]])


def tetrahedralise(verts, faces, res, device):
    """A conforming tet cage carved out of the mesh by its own SDF."""
    lo, hi = verts.min(0), verts.max(0)
    h = float((hi - lo).max()) / res
    lo = lo - 2.0 * h
    hi = hi + 2.0 * h
    dims = np.maximum(np.ceil((hi - lo) / h).astype(int), 1)

    mesh = wp.Mesh(points=wp.array(verts, dtype=wp.vec3, device=device),
                   indices=wp.array(faces.astype(np.int32).flatten(), dtype=wp.int32, device=device),
                   support_winding_number=True)
    sdf = wp.zeros(tuple(dims), dtype=wp.float32, device=device)
    # Launched in slabs, not one dispatch. A winding-number query walks the mesh
    # BVH, and on an open mesh it walks a lot of it; a single grid-wide launch is
    # exactly the multi-second dispatch this laptop answers with a device loss.
    # Slabs keep every dispatch short and cost nothing measurable.
    slab = max(1, SDF_SLAB)
    for i0 in range(0, int(dims[0]), slab):
        n = min(slab, int(dims[0]) - i0)
        wp.launch(_sdf_kernel, dim=(n, int(dims[1]), int(dims[2])), device=device,
                  inputs=[mesh.id, wp.vec3(*lo), h, float((hi - lo).max()), i0], outputs=[sdf])
        wp.synchronize_device(device)
    solid = sdf.numpy() < 0.0
    if not solid.any():
        sys.exit("tetrahedralise: nothing inside -- is the mesh closed? try a higher --tet-res")

    ci, cj, ck = np.nonzero(solid)
    nx, ny, nz = dims

    # Corner grid: number only the corners a solid cell actually touches.
    used = np.zeros((nx + 1, ny + 1, nz + 1), dtype=bool)
    for c in range(8):
        used[ci + (c & 1), cj + ((c >> 1) & 1), ck + ((c >> 2) & 1)] = True
    cid = -np.ones_like(used, dtype=np.int64)
    ui, uj, uk = np.nonzero(used)
    cid[ui, uj, uk] = np.arange(len(ui))
    tverts = (lo + np.stack([ui, uj, uk], 1) * h).astype(np.float32)

    corners = np.stack([cid[ci + (c & 1), cj + ((c >> 1) & 1), ck + ((c >> 2) & 1)]
                        for c in range(8)], 1)
    parity = (ci + cj + ck) % 2
    tets = np.concatenate([corners[parity == 0][:, _EVEN].reshape(-1, 4),
                           corners[parity == 1][:, _ODD].reshape(-1, 4)])

    # Positive orientation, whatever the split tables did.
    a, b, c_, d = (tverts[tets[:, i]] for i in range(4))
    neg = np.einsum("ij,ij->i", np.cross(b - a, c_ - a), d - a) < 0.0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]

    tverts, tets = _largest_component(tverts, tets)
    return tverts, tets.astype(np.int32), h


def _largest_component(verts, tets):
    """Voxel carving can shed islands (fin tips, scan noise). Keep the body."""
    parent = np.arange(len(verts))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for t in tets:
        r = find(t[0])
        for v in t[1:]:
            s = find(v)
            if s != r:
                parent[s] = r
    roots = np.array([find(i) for i in range(len(verts))])
    labels, counts = np.unique(roots[tets[:, 0]], return_counts=True)
    keep = labels[counts.argmax()]
    tets = tets[roots[tets[:, 0]] == keep]

    remap = -np.ones(len(verts), dtype=np.int64)
    live = np.unique(tets)
    remap[live] = np.arange(len(live))
    return verts[live], remap[tets]


def surface_of(tets):
    """Faces belonging to exactly one tet, wound outward."""
    f = tets[:, [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]]].reshape(-1, 3)
    _, idx, cnt = np.unique(np.sort(f, 1), axis=0, return_index=True, return_counts=True)
    return f[idx[cnt == 1]]


# ── The conveyor + the sim ───────────────────────────────────────────────────

@wp.kernel
def drive_rollers(t: wp.array(dtype=wp.float32), omega: float,
                  joint_q: wp.array(dtype=wp.float32), joint_qd: wp.array(dtype=wp.float32)):
    i = wp.tid()
    joint_q[i] = omega * t[0]
    joint_qd[i] = omega


@wp.kernel
def advance_time(dt: float, t: wp.array(dtype=wp.float32)):
    t[0] = t[0] + dt


class Sim:
    def __init__(self, cage, n_fish, device):
        cverts, ctets, cell = cage
        self.device = device
        self.n_fish = n_fish
        self.n_cage_verts = len(cverts)
        self.frame_dt = 1.0 / FPS
        self.sim_dt = self.frame_dt / SUBSTEPS

        ext = cverts.max(0) - cverts.min(0)
        fish_len, fish_wide = float(ext[0]), float(ext[1])

        # Lay the fish out in lanes across a belt sized to the fish.
        lanes = max(1, int(math.ceil(math.sqrt(n_fish * fish_len / (3.0 * fish_wide)))))
        lanes = min(lanes, 6)
        rows = int(math.ceil(n_fish / lanes))
        half_w = max(0.35, lanes * fish_wide * 0.75)
        bed_len = max(2.0, rows * fish_len * 1.35 + 1.0)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=(0.0, 0.0, -9.81))

        # --- roller bed: kinematic cylinders, axis along Y, tops at z = 0 ---
        roller_cfg = newton.ModelBuilder.ShapeConfig()
        roller_cfg.density = 0.0
        roller_cfg.mu = MU
        roller_cfg.ke, roller_cfg.kd = 1.0e5, 1.0e2
        roller_cfg.has_particle_collision = True
        roller_cfg.margin = 0.5 * cell

        spacing = 2.05 * ROLLER_R          # cylinders nearly touching: a tight bed
        n_rollers = max(2, int(bed_len / spacing))
        z_to_y = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), -0.5 * math.pi)
        self.half_w = half_w
        self.roller_x = []
        self.roller_joints = []
        for i in range(n_rollers):
            x = -0.5 + (i + 0.5) * spacing
            self.roller_x.append(x)
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01),
                                    is_kinematic=True, label=f"roller_{i}")
            builder.add_shape_cylinder(body, xform=wp.transform(wp.vec3(), z_to_y),
                                       radius=ROLLER_R, half_height=half_w, cfg=roller_cfg)
            j = builder.add_joint_revolute(
                parent=-1, child=body, axis=newton.Axis.Y,
                parent_xform=wp.transform(p=wp.vec3(x, 0.0, -ROLLER_R), q=wp.quat_identity()))
            builder.add_articulation([j], label=f"roller_{i}")
            self.roller_joints.append(j)

        # --- side rails, and a floor well below to catch anything that leaves ---
        rail_cfg = newton.ModelBuilder.ShapeConfig()
        rail_cfg.mu = 0.2
        rail_cfg.has_particle_collision = True
        rail_cfg.margin = 0.5 * cell
        for s in (-1.0, 1.0):
            builder.add_shape_box(body=-1, hx=0.5 * bed_len, hy=0.02, hz=0.12,
                                  xform=wp.transform(p=wp.vec3(0.5 * bed_len - 0.5,
                                                               s * (half_w + 0.02), 0.10),
                                                     q=wp.quat_identity()),
                                  cfg=rail_cfg)
        builder.add_ground_plane(height=-1.0)

        # --- the fish ---
        lam = YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
        mu_lame = YOUNG / (2.0 * (1.0 + POISSON))
        flat_tets = ctets.flatten().tolist()
        vert_list = cverts.tolist()
        drop = 0.02 + 0.5 * float(ext[2])
        self.spawn_x = []
        for k in range(n_fish):
            lane, row = k % lanes, k // lanes
            x = 0.2 + row * fish_len * 1.35
            y = (lane - 0.5 * (lanes - 1)) * (2.0 * half_w / max(lanes, 1)) * 0.85
            self.spawn_x.append(x)
            builder.add_soft_mesh(
                pos=wp.vec3(x, y, drop), rot=wp.quat_identity(), scale=1.0,
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=vert_list, indices=flat_tets,
                density=DENSITY, k_mu=mu_lame, k_lambda=lam, k_damp=1.0,
                add_surface_mesh_edges=SELF_CONTACT,
                particle_radius=0.4 * cell)

        builder.color()
        self.model = builder.finalize(device=device)
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e2
        self.model.soft_contact_mu = MU

        self.solver = newton.solvers.SolverVBD(
            self.model, iterations=ITERATIONS,
            particle_enable_self_contact=SELF_CONTACT,
            particle_self_contact_radius=0.4 * cell,
            particle_self_contact_margin=0.8 * cell,
            rigid_body_particle_contact_buffer_size=256)
        self.pipeline = newton.CollisionPipeline(
            self.model, broad_phase="sap", soft_contact_margin=0.5 * cell)

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)

        self.omega = BELT_SPEED / ROLLER_R
        self.n_rollers = n_rollers
        self.t = wp.zeros(1, dtype=wp.float32, device=device)
        self.graph = None

        # One dof per roller joint and no other joints: joint index == dof index.
        assert self.model.joint_dof_count == n_rollers, "roller joint layout changed"

    def simulate(self):
        for _ in range(SUBSTEPS):
            self.state_0.clear_forces()
            wp.launch(drive_rollers, dim=self.n_rollers, device=self.device,
                      inputs=[self.t, self.omega],
                      outputs=[self.state_0.joint_q, self.state_0.joint_qd])
            newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd,
                           self.state_0, body_flag_filter=newton.BodyFlags.KINEMATIC)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_time, dim=1, device=self.device, inputs=[self.sim_dt], outputs=[self.t])

    def capture(self):
        # simulate() ping-pongs state_0/state_1 per substep. Capture bakes that
        # sequence of device pointers into the graph, so a replay only lands the
        # result back in state_0 -- where the next replay starts reading -- if
        # the swap count is even. An odd count silently simulates from a stale
        # buffer every frame.
        if SUBSTEPS % 2:
            sys.exit(f"--graph needs an even --substeps (got {SUBSTEPS}): the state "
                     f"ping-pong would leave each frame reading a stale buffer")
        with wp.ScopedCapture(device=self.device) as cap:
            self.simulate()
        self.graph = cap.graph

    def step(self):
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

    def centroids(self):
        q = self.state_0.particle_q.numpy()[: self.n_fish * self.n_cage_verts]
        return q.reshape(self.n_fish, self.n_cage_verts, 3).mean(1)


# ── Measurement ──────────────────────────────────────────────────────────────

def vram_mb(device):
    try:
        d = wp.get_device(device)
        return (d.total_memory - d.free_memory) / (1 << 20)
    except Exception:
        return float("nan")


def run(cage, n_fish, device, verbose=True):
    t_build = time.perf_counter()
    sim = Sim(cage, n_fish, device)
    build_s = time.perf_counter() - t_build

    for _ in range(WARMUP):          # kernel compile + settle onto the rollers
        sim.simulate()
    wp.synchronize_device(device)
    if GRAPH:
        sim.capture()
        sim.step()
        wp.synchronize_device(device)

    x0 = sim.centroids()[:, 0].copy()
    used0 = vram_mb(device)

    times = []
    for _ in range(FRAMES):
        wp.synchronize_device(device)
        t0 = time.perf_counter()
        sim.step()
        wp.synchronize_device(device)
        times.append((time.perf_counter() - t0) * 1e3)

    c = sim.centroids()
    dx = c[:, 0] - x0
    elapsed = FRAMES / FPS
    finite = np.isfinite(c).all(1)
    onbelt = finite & (c[:, 2] > -0.5)

    times = np.array(times)
    r = {
        "n": n_fish,
        "tets": sim.model.tet_count,
        "particles": sim.model.particle_count,
        "rollers": sim.n_rollers,
        "ms": float(np.median(times)),
        "ms_p95": float(np.percentile(times, 95)),
        "rt": (1000.0 / FPS) / float(np.median(times)),
        "convey": float(np.mean(dx[onbelt]) / elapsed) if onbelt.any() else float("nan"),
        "belt": BELT_SPEED,
        "onbelt": int(onbelt.sum()),
        "vram": used0,
        "build_s": build_s,
    }
    if verbose:
        print(f"  {n_fish:>4} fish | {r['tets']:>7} tets {r['particles']:>7} parts | "
              f"{r['ms']:7.2f} ms/frame (p95 {r['ms_p95']:6.2f}) | {r['rt']:5.2f}x realtime | "
              f"convey {r['convey']:+.3f} m/s of {BELT_SPEED:.2f} | "
              f"{r['onbelt']}/{n_fish} on belt | {r['vram']:.0f} MB", flush=True)
    return sim, r


def flat_normals(p):
    """Per-face normals for a non-indexed triangle soup, one copy per vertex."""
    t = p.reshape(-1, 3, 3)
    n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(np.repeat(n, 3, axis=0), dtype=np.float32)


def view(sim, cage_surf, cage_verts):
    """Eyeball it. Host readback per frame -- deliberately not the perf path.

    Newton is Z-up and so is this scene, which is why the camera sits at -Y
    looking along +Y rather than the usual three.js Y-up placement.
    """
    canvas = tp.Canvas("newton fish conveyor", width=1280, height=720, antialiasing=4)
    renderer = tp.GLRenderer(canvas)

    scene = tp.Scene()
    scene.background = 0x171A1F
    scene.add(tp.AmbientLight(0xFFFFFF, 0.45))
    key = tp.DirectionalLight(0xFFFFFF, 2.5)
    key.position.set(2, -4, 6)
    scene.add(key)

    tri = cage_surf.flatten()
    rest = np.ascontiguousarray(cage_verts[tri], dtype=np.float32)
    # ALLOCATE with set_attribute, UPDATE with update_attribute. The renderer
    # caches its GPU buffers per geometry and invalidates that cache on the
    # geometry's version, which update_attribute bumps and set_attribute does
    # not -- a per-frame set_attribute leaves the old buffer live, so a fish
    # keeps being drawn at a position it has already left.
    geoms = []
    for _ in range(sim.n_fish):
        g = tp.BufferGeometry()
        g.set_attribute("position", rest.copy())
        g.set_attribute("normal", flat_normals(rest))
        mat = tp.MeshPhongMaterial()
        mat.color = 0xC08878
        mat.shininess = 60.0
        scene.add(tp.Mesh(g, mat))
        geoms.append(g)

    roller_geo = tp.CylinderGeometry(ROLLER_R, ROLLER_R, 2.0 * sim.half_w, 20)
    roller_mat = tp.MeshPhongMaterial()
    roller_mat.color = 0x707880
    for x in sim.roller_x:
        m = tp.Mesh(roller_geo, roller_mat)
        m.position.set(x, 0.0, -ROLLER_R)
        # CylinderGeometry's axis is +Y; the rollers lie along world +Y already,
        # so no rotation -- unlike a Y-up scene, where this would need one.
        scene.add(m)

    camera = tp.PerspectiveCamera(50, canvas.aspect(), 0.05, 100)
    camera.position.set(1.0, -2.0, 1.1)
    camera.up.set(0, 0, 1)
    controls = tp.OrbitControls(camera, canvas)
    controls.target.set(1.0, 0.0, 0.0)
    controls.enable_damping = True

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        renderer.set_size(w, h)

    canvas.on_window_resize(on_resize)

    n_cage = sim.n_cage_verts

    def animate():
        sim.step()
        q = sim.state_0.particle_q.numpy()[: sim.n_fish * n_cage].reshape(sim.n_fish, n_cage, 3)
        for i, g in enumerate(geoms):
            p = np.ascontiguousarray(q[i][tri], dtype=np.float32)
            g.update_attribute("position", p)
            g.update_attribute("normal", flat_normals(p))
        controls.update()
        renderer.render(scene, camera)

    canvas.animate(animate)


def main():
    if not FISH:
        sys.exit("--fish PATH is required: there is no bundled model.\n"
                 "  Any closed solid threepp's ModelLoader reads will do -- .usdz, .usd,\n"
                 "  .obj, .stl, .gltf/.glb, .dae. Nothing here is fish-specific beyond the\n"
                 "  auto-orientation (longest extent along the belt, thinnest one up).\n"
                 "  e.g. python newton_fish_conveyor.py --fish cod.usdz --graph")
    if not os.path.isfile(FISH):
        sys.exit(f"--fish: no such file: {FISH}")

    device = wp.get_device(DEVICE)
    if device.is_cpu:
        print("warning: --device cpu is not a meaningful measurement of GPU headroom")

    verts, faces = load_triangles(FISH)
    verts = orient_fish(verts)
    if FISH_LENGTH > 0:
        verts *= FISH_LENGTH / float((verts.max(0) - verts.min(0))[0])
    verts = verts - 0.5 * (verts.max(0) + verts.min(0))
    verts[:, 2] -= verts[:, 2].min()
    ext = verts.max(0) - verts.min(0)

    print(f"fish  {os.path.basename(FISH)}")
    print(f"      {len(verts)} verts, {len(faces)} tris, "
          f"{ext[0]:.3f} x {ext[1]:.3f} x {ext[2]:.3f} m (L x W x H), "
          f"{'closed' if is_closed(faces) else 'NOT CLOSED -- signs are unreliable'}")

    t0 = time.perf_counter()
    cverts, ctets, cell = tetrahedralise(verts, faces, TET_RES, device)
    print(f"cage  {len(cverts)} verts, {len(ctets)} tets @ {cell * 1000:.1f} mm cells "
          f"({time.perf_counter() - t0:.2f} s)")
    print(f"mat   E={YOUNG:.3g} Pa, nu={POISSON}, rho={DENSITY} kg/m3 | "
          f"belt {BELT_SPEED} m/s, mu={MU}, rollers r={ROLLER_R} m")
    print(f"solve VBD {ITERATIONS} iters x {SUBSTEPS} substeps @ {FPS:.0f} Hz"
          f"{', CUDA graph' if GRAPH else ''}{', self-contact' if SELF_CONTACT else ''}")
    print()

    if VIEW:
        # Warm up (compile + settle onto the rollers) but skip the timed loop --
        # the window is for looking at, and the host readback in view() would
        # make any number it produced meaningless anyway.
        sim = Sim((cverts, ctets, cell), COUNT, device)
        for _ in range(WARMUP):
            sim.simulate()
        wp.synchronize_device(device)
        if GRAPH:
            sim.capture()
        print(f"      {COUNT} fish, {sim.model.tet_count} tets -- drag to orbit, Esc quits",
              flush=True)
        view(sim, surface_of(ctets), cverts)
        return

    counts = [n for n in (1, 4, 16, 64, 128, 256, 512) if n <= SWEEP_TO] if SWEEP else [COUNT]
    rows = []
    for n in counts:
        if n * len(cverts) > MAX_PARTICLES:
            print(f"  {n:>4} fish | SKIPPED: {n * len(cverts)} particles exceeds "
                  f"--max-particles {MAX_PARTICLES}", flush=True)
            break
        print(f"  {n:>4} fish | building...", end="\r", flush=True)
        try:
            _, r = run((cverts, ctets, cell), n, device)
            rows.append(r)
        except Exception as e:                       # OOM or buffer overflow at high n
            print(f"  {n:>4} fish | FAILED: {type(e).__name__}: {e}", flush=True)
            break

    if len(rows) > 1:
        print()
        print("  realtime budget is 16.67 ms/frame at 60 Hz")
        last = [r for r in rows if r["rt"] >= 1.0]
        if last:
            print(f"  realtime ceiling on this GPU: ~{last[-1]['n']} fish "
                  f"({last[-1]['tets']} tets) at {last[-1]['ms']:.1f} ms/frame")
        else:
            print("  nothing hit realtime -- lower --tet-res or --iterations")


if __name__ == "__main__":
    main()
