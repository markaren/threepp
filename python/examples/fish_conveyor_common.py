"""Everything the fish-conveyor benchmarks share, so the engines can be compared.

Two examples sit on this: newton_fish_conveyor.py (Newton SolverVBD) and
physx_fish_conveyor.py (PhysX 5 deformable volumes). The only thing that differs
between them is the solver -- the scans, the orientation, the tetrahedral cage,
the barycentric bind, the belt layout, the lighting, the camera, the GPU skinning
and the quality metrics all live here and are identical on both sides. That is
the whole point of an A/B: two runs that share everything except the thing under
test.

Nothing here parses a command line by itself. Both callers read their own flags
and push the shared ones in through configure(), because a module-level constant
that two examples disagree about is exactly the unfairness this file exists to
prevent.
"""
import math
import os
import sys
import tempfile
import time

import numpy as np
import warp as wp

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


# -- Shared configuration ----------------------------------------------------
# Module globals rather than a config object, because the code below is lifted
# verbatim from the Newton benchmark and reads them as globals. configure() is
# the only writer; both examples call it once, after parsing argv.

FPS = 60.0
TET_RES = 14          # cells along the fish's longest axis (voxel cage)
CARVE = 0.25          # cells of dilation on the voxel cage
SDF_SLAB = 8          # grid rows per SDF dispatch
RENDER_RATIO = 0.4    # decimate the drawn scan to this fraction of its triangles
CAGE = False          # draw the simulation cage instead of the scan
FISH_LENGTH = 0.0     # 0 = keep the scan's own size
LANES = 5             # fish abreast; sets the belt width
BED = 30.0            # m; hard cap on the belt length
LAYERS = 2            # how deep the catch may be piled
BELT_SPEED = 0.5      # m/s along +X
ROLLER_R = 0.05       # m
MU = 0.45             # plastic module belt: the FHF parameters measured 0.43-0.56
BELT_STYLE = "rollers"  # rollers | flat -- flat is a plastic modular belt, the
                        # thing MU was actually measured on and what a real
                        # processing line runs; solvers that support it opt in
WARMUP = 90           # settle frames, untimed
FRAMES = 120          # timed frames
# How many frames the belt has to survive after warmup. Normally FRAMES, but the
# A/B runs a longer window and the runway below is sized from this, not from
# whatever the sweep happens to use.
MEASURED_FRAMES = 120
# Per-fish size jitter, as a fraction. Newton scales each soft mesh at spawn for
# free; PhysX's cook cache keys on the source geometry and applies only the
# spawn rotation and translation, so a per-fish scale would mean one cook per
# fish. The A/B therefore runs both engines at 0 and keeps the yaw and position
# jitter, which is where the contact-load variety actually comes from.
SCALE_JITTER = 0.07
SEED = 7              # layout rng seed
INTEROP = True        # zero-copy skinning straight into the GL vertex buffers
# Path to a physical-parameters JSON (the FHF database ships one): a dict keyed
# by scan stem, each entry carrying dimensions.total_length_mm and weight_kg.
# When set, every scan is rescaled to ITS OWN measured length -- photogrammetry
# scale is arbitrary, and these scans arrive 2.5x life size -- and Fish.weight_kg
# carries the measured mass for the solvers that can use it. "" = scan size.
PARAMS = ""

MODEL_EXT = (".usdz", ".usd", ".usda", ".usdc", ".obj", ".stl", ".gltf", ".glb", ".dae")


def configure(**kw):
    """Set the shared knobs from a caller's parsed CLI. An unknown name raises."""
    g = globals()
    for k, v in kw.items():
        if k not in g:
            raise KeyError("fish_conveyor_common.configure: no such setting " + repr(k))
        g[k] = v


# -- The fish: load, orient, tetrahedralise ----------------------------------

_params_cache = {}


def _params_entry(scan_path):
    """The physical-parameters entry for one scan, keyed by its filename stem.

    A missing file or a missing key is silent by design: the parameters are an
    upgrade, not a requirement, and a scan without an entry simply keeps its
    own scale like before.
    """
    if not PARAMS:
        return None
    if PARAMS not in _params_cache:
        import json
        try:
            with open(PARAMS, "r", encoding="utf-8") as f:
                _params_cache[PARAMS] = json.load(f)
        except (OSError, ValueError) as e:
            sys.exit(f"--params {PARAMS}: {e}")
    stem = os.path.splitext(os.path.basename(scan_path))[0]
    return _params_cache[PARAMS].get(stem)


def find_params(fish_spec):
    """Auto-discover a Physical_Parameters/parameters.json next to the scans."""
    first = fish_spec.split(",")[0].strip()
    base = first if os.path.isdir(first) else os.path.dirname(first)
    for root in (base, os.path.dirname(base)):
        p = os.path.join(root, "Physical_Parameters", "parameters.json")
        if os.path.isfile(p):
            return p
    return ""

def load_surface(path):
    """Every triangle of the model, welded into one (verts, faces, uv, material).

    The scan arrives as an indexed, UV-mapped mesh with its baked albedo, normal
    and AO maps already attached by threepp's loader -- all of which is kept, and
    all of which is what the render surface is for. World transforms are baked
    in so a scan authored under a scaled root still measures what it looks like.
    """
    root = tp.ModelLoader().load(path)
    if root is None:
        sys.exit(f"could not load {path} (unsupported format, or a broken file)")
    root.update_matrix_world(True)

    meshes = []

    def collect(node):
        if isinstance(node, tp.Mesh) and node.geometry is not None:
            meshes.append(node)

    root.traverse(collect)
    if not meshes:
        sys.exit(f"no meshes in {path}")

    verts, faces, uvs, material, base = [], [], [], None, 0
    for m in meshes:
        g = m.geometry
        p = g.get_attribute("position")
        if p is None or not len(p):
            continue
        p = np.asarray(p, np.float32)
        w = m.matrix_world.to_numpy()
        p = (p @ w[:3, :3].T + w[:3, 3]).astype(np.float32)
        idx = g.get_index()
        idx = np.arange(len(p), dtype=np.int64) if idx is None else np.asarray(idx, dtype=np.int64)
        uv = g.get_attribute("uv")
        uvs.append(np.zeros((len(p), 2), np.float32) if uv is None else np.asarray(uv, np.float32))
        faces.append(idx.reshape(-1, 3) + base)
        verts.append(p)
        base += len(p)
        if material is None:
            material = m.material
    if not verts:
        sys.exit(f"no geometry in {path}")
    return np.concatenate(verts), np.concatenate(faces), np.concatenate(uvs), material


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
    # mirrored (a fish is not symmetric, and neither is a fillet) and so the
    # surface winding -- which the renderer does care about -- survives.
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
    # Keep cells whose centre is within CARVE cells OUTSIDE the surface, not just
    # the ones strictly inside it. Carving at the zero level set sounds right and
    # is not: a cell only survives if its CENTRE is inside, so the cage ends up to
    # half a cell short on every face, and at a coarse resolution that is not a
    # rounding error -- a 14-cell cage of a 1.412 m cod measured 1.110 m, i.e. the
    # snout and the whole tail were outside the collider AND outside every tet the
    # skin binds to. Fish went through each other nose-first, and skin vertices
    # extrapolated from a crushed tet 300 mm away left the pile as needles.
    # At 0.25 cells the cage measures the fish exactly, for 1.7x the tets.
    solid = sdf.numpy() < CARVE * h
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


# ── Binding the scan to the cage ─────────────────────────────────────────────

# A bound on how much a surface vertex may amplify its tet's motion. The cage is
# carved from cells whose CENTRE is inside the mesh, so it ends up to half a cell
# short of the skin and better than half the surface sits outside its own tet;
# those vertices are extrapolated, and extrapolation multiplies. That is fine at
# |w| ~ 1.5 and lethal at 4: crush a tet in the middle of a heap and a fin tip
# bound at |w| = 3.9 leaves the fish as a metre-long needle. Seen exactly that.
WEIGHT_LIMIT = 2.2


def bind_surface(surf, cverts, ctets, cell):
    """Bind every surface vertex to a cage tet, by barycentric weights.

    Weights are UNCLAMPED and sum to one, so a vertex reproduces its rest
    position exactly even when it lies outside the cage -- which most of the
    surface does, since voxel carving only keeps cells whose centre is inside
    the mesh, leaving the cage a good half-cell short of the skin everywhere.
    Clamping instead is what rounds the corners off a bound mesh; the guard
    above only catches the pathological tail.

    The search is O(1) per vertex because the cage is a lattice: every tet lives
    inside one cell of a regular grid of side `cell`, so the tet holding a point
    is in one of the 27 cells around it. Everything else is a fallback for the
    fin tips the carve dropped on the floor.
    """
    surf = np.asarray(surf, np.float64)
    cverts64 = np.asarray(cverts, np.float64)
    n = len(surf)

    a = cverts64[ctets[:, 0]]
    edges = np.stack([cverts64[ctets[:, i]] - a for i in (1, 2, 3)], axis=2)  # columns
    einv = np.linalg.inv(edges)
    centroid = cverts64[ctets].mean(1)

    origin = cverts64.min(0)
    cellof = np.floor((centroid - origin) / cell).astype(np.int64)
    dims = cellof.max(0) + 1
    lin = (cellof[:, 0] * dims[1] + cellof[:, 1]) * dims[2] + cellof[:, 2]
    order = np.argsort(lin, kind="stable")
    lin_s = lin[order]
    starts = np.flatnonzero(np.r_[True, lin_s[1:] != lin_s[:-1]])
    counts = np.diff(np.r_[starts, len(lin_s)])
    slot = np.arange(len(lin_s)) - np.repeat(starts, counts)
    table = np.full((int(dims.prod()), int(counts.max())), -1, np.int64)
    table[lin_s, slot] = order

    def weights_for(points, tet_ids):
        d = points - a[tet_ids]
        uvw = np.einsum("nij,nj->ni", einv[tet_ids], d)
        return np.concatenate([1.0 - uvw.sum(1, keepdims=True), uvw], axis=1)

    base = np.clip(np.floor((surf - origin) / cell).astype(np.int64), 0, dims - 1)
    best_t = np.full(n, -1, np.int64)
    best_w = np.zeros((n, 4))
    best_score = np.full(n, -np.inf)         # max-min-weight = "most inside"
    for di in (0, -1, 1):
        for dj in (0, -1, 1):
            for dk in (0, -1, 1):
                c = base + (di, dj, dk)
                ok = ((c >= 0) & (c < dims)).all(1)
                if not ok.any():
                    continue
                q = np.where(ok, (c[:, 0] * dims[1] + c[:, 1]) * dims[2] + c[:, 2], 0)
                for s in range(table.shape[1]):
                    t = np.where(ok, table[q, s], -1)
                    rows = np.flatnonzero(t >= 0)
                    if not len(rows):
                        continue
                    ids = t[rows]
                    w = weights_for(surf[rows], ids)
                    score = w.min(1)
                    better = score > best_score[rows]
                    hit = rows[better]
                    best_score[hit] = score[better]
                    best_t[hit] = ids[better]
                    best_w[hit] = w[better]

    # Anything the lattice never reached: nearest tet by centroid, no filter.
    # (The AABB/neighbourhood pre-filter is a speed filter, never a correctness
    # bound -- a fin tip whose cells were all carved away has no local tet.)
    miss = np.flatnonzero(best_t < 0)
    for k in range(0, len(miss), 1024):
        blk = miss[k:k + 1024]
        ids = ((surf[blk][:, None, :] - centroid[None]) ** 2).sum(-1).argmin(1)
        best_t[blk] = ids
        best_w[blk] = weights_for(surf[blk], ids)

    # Over the bound, slide toward the clamped (inside-the-tet) weights -- but
    # only as far as the bound needs, not all the way. A full projection is what
    # rounds fins off; the shortest blend that satisfies |w| <= WEIGHT_LIMIT keeps
    # almost all of the shape and still cannot amplify.
    peak = np.abs(best_w).max(1)
    wild = peak > WEIGHT_LIMIT
    if wild.any():
        inside = np.clip(best_w[wild], 0.0, None)
        inside /= np.maximum(inside.sum(1, keepdims=True), 1e-12)
        t = ((peak[wild] - WEIGHT_LIMIT) / (peak[wild] - 1.0))[:, None]
        for _ in range(6):                       # the max is piecewise linear in t
            blend = (1.0 - t) * best_w[wild] + t * inside
            over = np.abs(blend).max(1)[:, None] > WEIGHT_LIMIT
            t = np.where(over, np.minimum(t + 0.1, 1.0), t)
        best_w[wild] = (1.0 - t) * best_w[wild] + t * inside

    tet4 = ctets[best_t].astype(np.int32)
    bary = best_w.astype(np.float32)
    rest = np.einsum("nkj,nk->nj", cverts64[tet4], best_w)
    err = np.linalg.norm(rest - surf, axis=1)
    return tet4, bary, (float(err.mean()), float(err.max())), int(wild.sum()), int(len(miss))


class Fish:
    """One scan: its render surface, its simulation cage, and the bind between.

    `tets` injects an externally cooked cage -- (rest vertices, tets) out of any
    other tetrahedraliser -- in place of the voxel carve. That is how the Newton
    run on PhysX-cooked tets is built: same solver, same skin, same belt, a
    different mesh underneath. When it is supplied the bind lattice is sized from
    the cage's own longest tet edge rather than a voxel pitch, which is the bound
    the neighbourhood search actually needs.
    """

    def __init__(self, path, device, tets=None):
        verts, faces, uv, material = load_surface(path)
        verts = orient_fish(verts)
        self.weight_kg = None
        entry = _params_entry(path)
        if entry is not None:
            # The measured fish, not the scanned artefact: true length (the rig's
            # scale is arbitrary) and true mass for whoever can apply it.
            L = float(entry["dimensions"]["total_length_mm"]) * 1e-3
            verts *= L / float((verts.max(0) - verts.min(0))[0])
            self.weight_kg = float(entry["weight_kg"])
        elif FISH_LENGTH > 0:
            verts *= FISH_LENGTH / float((verts.max(0) - verts.min(0))[0])
        verts = verts - 0.5 * (verts.max(0) + verts.min(0))
        verts[:, 2] -= verts[:, 2].min()

        self.path = path
        self.verts, self.faces, self.uv, self.material = verts, faces, uv, material
        self.extent = verts.max(0) - verts.min(0)
        self.closed = is_closed(faces)

        # The SDF bake wants every triangle the scan has; the renderer does not.
        # meshoptimizer takes 25k triangles of photogrammetry down to 10k with a
        # bounding box that does not move a micron -- the detail that matters at
        # this distance is in the baked normal map, not in the tessellation --
        # and every triangle dropped is one fewer vertex to skin, every frame,
        # per fish.
        self.rverts, self.rfaces, self.ruv = verts, faces.astype(np.int32), uv
        if 0.0 < RENDER_RATIO < 1.0:
            g = tp.BufferGeometry()
            g.set_attribute("position", verts)
            g.set_attribute("uv", uv)
            g.set_index(faces.astype(np.uint32).reshape(-1))
            g = tp.simplify_geometry(g, RENDER_RATIO, 0.02)
            self.rverts = np.ascontiguousarray(g.get_attribute("position"), np.float32)
            self.ruv = np.ascontiguousarray(g.get_attribute("uv"), np.float32)
            self.rfaces = np.asarray(g.get_index(), np.int32).reshape(-1, 3)

        t0 = time.perf_counter()
        if tets is None:
            self.cverts, self.ctets, self.cell = tetrahedralise(verts, faces, TET_RES, device)
            self.bind_cell = self.cell
            self.cage_kind = "voxel %d" % TET_RES
        else:
            # A callable gets the half-built Fish, so a cooker that wants the
            # decimated render surface (PhysX's does -- it remeshes anyway, and
            # cooking 25k photogrammetry triangles is minutes of nothing) can
            # reach it. Everything above this point is already final.
            cv, ct = tets(self) if callable(tets) else tets
            self.cverts = np.ascontiguousarray(cv, np.float32)
            self.ctets = np.ascontiguousarray(ct, np.int32)
            self.cell = contact_cell(self.cverts, self.ctets)
            self.bind_cell = lattice_cell(self.cverts, self.ctets)
            self.cage_kind = "supplied"
        self.cage_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        surf = self.cverts if CAGE else self.rverts
        self.n_bound = len(surf)
        self.tet4, self.bary, self.rest_err, self.clamped, self.orphans = \
            bind_surface(surf, self.cverts, self.ctets, self.bind_cell)
        self.bind_s = time.perf_counter() - t0

        # Rest tet volumes and rest length, once: the quality metrics divide by
        # them after every run.
        r = self.cverts[self.ctets].astype(np.float64)
        self.rest_vol = np.einsum("ij,ij->i", np.cross(r[:, 1] - r[:, 0], r[:, 2] - r[:, 0]),
                                  r[:, 3] - r[:, 0]) / 6.0
        self.rest_edge = np.stack([np.linalg.norm(r[:, a] - r[:, b], axis=1)
                                   for a, b in _EDGES], axis=1)
        self.rest_len = float((self.cverts.max(0) - self.cverts.min(0))[0])

    def render_surface(self):
        """(vertices, triangles, uv) of whatever this run draws."""
        if CAGE:
            return self.cverts, surface_of(self.ctets), np.zeros((len(self.cverts), 2), np.float32)
        return self.rverts, self.rfaces, self.ruv

    def report(self):
        e = self.extent
        print(f"fish  {os.path.basename(self.path)}")
        print(f"      {len(self.verts)} verts, {len(self.faces)} tris"
              f"{'' if len(self.rfaces) == len(self.faces) else f' -> {len(self.rfaces)} drawn'}, "
              f"{e[0]:.3f} x {e[1]:.3f} x {e[2]:.3f} m (L x W x H)"
              f"{'' if self.weight_kg is None else f', {self.weight_kg:.2f} kg measured'}, "
              f"{'closed' if self.closed else 'NOT CLOSED -- signs are unreliable'}")
        print(f"cage  {len(self.cverts)} verts, {len(self.ctets)} tets @ "
              f"{self.cell * 1000:.1f} mm cells ({self.cage_s:.2f} s)")
        mean_err, max_err = self.rest_err
        print(f"bind  {self.n_bound} surface verts -> cage, rest error "
              f"{mean_err * 1000:.3f} mm mean / {max_err * 1000:.1f} mm max, "
              f"{self.clamped} clamped, {self.orphans} outside the lattice "
              f"({self.bind_s:.2f} s)")


_EDGES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def contact_cell(verts, tets):
    """Characteristic particle spacing of a cage, however that cage was built.

    Every contact radius on the Newton side is quoted in cells of this, so it has
    to mean the same thing for a voxel carve and for a cooked conforming mesh.
    Handing it the LONGEST tet edge instead -- which is what the bind lattice
    wants -- makes "particle radius 0.40 cells" 133 mm on a 50 mm roller, and
    that is not a fish, it is a boulder. Measured: 16 fish went from 7.4 ms to
    26.4 ms with 25 tets inverted at 18x stretch, purely from that one number.

    Median of each tet's SHORTEST edge. On the voxel lattice that is exactly the
    voxel pitch -- four of the five tets in a cube split own a cube edge, the
    fifth is the h*sqrt(2) core -- so the tuned Newton defaults keep their
    meaning unchanged.
    """
    t = np.asarray(verts, np.float64)[tets]
    e = np.stack([np.linalg.norm(t[:, a] - t[:, b], axis=1) for a, b in _EDGES], axis=1)
    return float(np.median(e.min(axis=1)))


def lattice_cell(verts, tets):
    """A bin size for bind_surface's neighbourhood search, for arbitrary tets.

    The search only has to guarantee that the tet containing a point is binned
    within one cell of that point, and a point inside a tet is never further from
    that tet's centroid than the tet's longest edge -- so the longest edge in the
    cage is the bound. (A voxel cage passes its lattice pitch instead, which is
    the same bound by construction.)
    """
    t = verts[tets]
    longest = 0.0
    for a, b in _EDGES:
        longest = max(longest, float(np.linalg.norm(t[:, a] - t[:, b], axis=1).max()))
    return longest


def fish_paths(spec):
    """--fish takes a file, a directory of scans, or a comma-separated list."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if os.path.isdir(part):
            out += [os.path.join(part, f) for f in sorted(os.listdir(part))
                    if f.lower().endswith(MODEL_EXT)]
        else:
            out.append(part)
    return out


# -- Skinning: the scan rides the cage, on the GPU ---------------------------

@wp.kernel
def skin_surface(q: wp.array(dtype=wp.vec3),
                 part_base: wp.array(dtype=int),
                 tet: wp.array2d(dtype=int),
                 bary: wp.array2d(dtype=float),
                 n_surf: int,
                 out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    f = i // n_surf
    k = i - f * n_surf
    b = part_base[f]
    out[i] = (q[b + tet[k, 0]] * bary[k, 0] + q[b + tet[k, 1]] * bary[k, 1] +
              q[b + tet[k, 2]] * bary[k, 2] + q[b + tet[k, 3]] * bary[k, 3])


@wp.kernel
def clear_normals(nrm: wp.array(dtype=wp.vec3)):
    nrm[wp.tid()] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def accumulate_normals(pos: wp.array(dtype=wp.vec3), tri: wp.array2d(dtype=int),
                       n_surf: int, n_tri: int, nrm: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    f = i // n_tri
    t = i - f * n_tri
    b = f * n_surf
    ia = b + tri[t, 0]
    ib = b + tri[t, 1]
    ic = b + tri[t, 2]
    n = wp.cross(pos[ib] - pos[ia], pos[ic] - pos[ia])
    wp.atomic_add(nrm, ia, n)
    wp.atomic_add(nrm, ib, n)
    wp.atomic_add(nrm, ic, n)


@wp.kernel
def unitise_normals(nrm: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    n = nrm[i]
    l = wp.length(n)
    if l > 1.0e-12:
        nrm[i] = n / l
    else:
        nrm[i] = wp.vec3(0.0, 0.0, 1.0)



class School:
    """Every fish of one variant, as a single indexed mesh Warp rewrites in place.

    One mesh, not one per fish: the whole variant is one index buffer and one
    draw, and the skinning kernels see one flat vertex range. The alternative --
    a geometry per fish, updated from Python -- is a per-frame host round trip
    per fish, and it is what made the old viewer fall over at a dozen.

    `part_base` is the whole run's per-fish offset into the flat cage-vertex
    array, in fish order; the School takes the slice its own variant owns. The
    positions handed to update() may be a Warp array already on the device (what
    Newton's solver produces) or a host numpy array (what a PhysX readback
    produces) -- the numpy case is uploaded into a buffer owned here, so the
    skinning kernels below are the same code either way.
    """

    def __init__(self, fish, indices, part_base, device):
        verts, tris, uv = fish.render_surface()
        self.device = device
        self.n_fish = len(indices)
        self.n_surf = len(verts)
        self.n_tri = len(tris)
        self.n_vert = self.n_fish * self.n_surf

        self.part_base = wp.array(np.asarray(part_base, np.int32)[indices], dtype=int, device=device)
        self.tet = wp.array(fish.tet4, dtype=int, device=device)
        self.bary = wp.array(fish.bary, dtype=float, device=device)
        self.tri = wp.array(np.ascontiguousarray(tris, np.int32), dtype=int, device=device)

        # Rest pose, tiled once. The renderer needs a real buffer to allocate
        # against; every frame after this one is written on the device.
        rest = np.tile(verts.astype(np.float32), (self.n_fish, 1))
        self.geometry = tp.BufferGeometry()
        self.geometry.set_attribute("position", rest)
        self.geometry.set_attribute("normal", np.zeros_like(rest))
        self.geometry.set_attribute("uv", np.tile(uv.astype(np.float32), (self.n_fish, 1)))
        # aoMap samples uv2, not uv -- same unwrap here, but it must be present
        # or the scan's baked occlusion silently does nothing.
        self.geometry.set_attribute("uv2", np.tile(uv.astype(np.float32), (self.n_fish, 1)))
        self.geometry.set_index(
            (np.tile(tris.astype(np.uint32), (self.n_fish, 1)).reshape(self.n_fish, -1)
             + (np.arange(self.n_fish, dtype=np.uint32) * self.n_surf)[:, None]).reshape(-1))

        self.mesh = tp.Mesh(self.geometry, fish_material(fish))
        self.mesh.cast_shadow = True
        self.mesh.receive_shadow = True
        self.mesh.frustum_culled = False     # the vertices move; the bounds do not

        self.host_pos = wp.zeros(self.n_vert, dtype=wp.vec3, device=device)
        self.host_nrm = wp.zeros(self.n_vert, dtype=wp.vec3, device=device)
        self.upload = None                   # device staging for host-side positions
        self.reg_pos = self.reg_nrm = None

    def _device_positions(self, q):
        """A wp.vec3 device array, whichever side of the bus the caller had it on."""
        if isinstance(q, wp.array):
            return q
        q = np.ascontiguousarray(q, np.float32)
        if self.upload is None or len(self.upload) != len(q):
            self.upload = wp.zeros(len(q), dtype=wp.vec3, device=self.device)
        wp.copy(self.upload, wp.array(q, dtype=wp.vec3, device=self.device, copy=False))
        return self.upload

    def _skin_into(self, q, pos, nrm):
        wp.launch(skin_surface, dim=self.n_vert, device=self.device,
                  inputs=[q, self.part_base, self.tet, self.bary, self.n_surf], outputs=[pos])
        wp.launch(clear_normals, dim=self.n_vert, device=self.device, outputs=[nrm])
        wp.launch(accumulate_normals, dim=self.n_fish * self.n_tri, device=self.device,
                  inputs=[pos, self.tri, self.n_surf, self.n_tri], outputs=[nrm])
        wp.launch(unitise_normals, dim=self.n_vert, device=self.device, outputs=[nrm])

    def update(self, q, renderer):
        """Skin one frame. Straight into the GL vertex buffers when we can."""
        q = self._device_positions(q)
        if INTEROP and self.reg_pos is None:
            pid = renderer.gl_buffer_id(self.geometry, "position")
            nid = renderer.gl_buffer_id(self.geometry, "normal")
            if pid is not None and nid is not None:      # None until the first render
                flags = wp.RegisteredGLBuffer.WRITE_DISCARD
                self.reg_pos = wp.RegisteredGLBuffer(int(pid), self.device, flags)
                self.reg_nrm = wp.RegisteredGLBuffer(int(nid), self.device, flags)
        if self.reg_pos is not None:
            pos = self.reg_pos.map(dtype=wp.vec3, shape=(self.n_vert,))
            nrm = self.reg_nrm.map(dtype=wp.vec3, shape=(self.n_vert,))
            self._skin_into(q, pos, nrm)
            self.reg_pos.unmap()
            self.reg_nrm.unmap()
            return
        self._skin_into(q, self.host_pos, self.host_nrm)
        self.geometry.update_attribute("position", self.host_pos.numpy())
        self.geometry.update_attribute("normal", self.host_nrm.numpy())


# -- The belt, and where every fish starts on it ------------------------------

class Layout:
    """The conveyor's geometry and the catch's spawn poses. Engine-independent.

    Every number a solver needs to build the same scene twice: the roller
    positions, the rails, and per fish a spawn position, a yaw and a scale. Both
    engines call plan_layout() with the same fishes and the same count and get
    the same belt, which is the precondition for the comparison meaning anything.
    """

    def __init__(self, **kw):
        self.__dict__.update(kw)


def plan_layout(fishes, n_fish, variant, seed=None):
    """Size the belt to the fish and deal the catch onto it."""
    rng = np.random.default_rng(SEED if seed is None else seed)
    # Sized off the SCAN's bounding box, not the cage's. Two engines tetrahedralise
    # the same fish into cages of slightly different extent, and a belt whose
    # length depended on that would give them different beds -- and a different
    # pile -- which is the one thing an A/B may not do. (For the voxel cage at
    # --carve 0.25 the two agree to a fraction of a millimetre anyway.)
    ext = np.array([f.extent for f in fishes])
    fish_len, fish_wide = float(ext[:, 0].max()), float(ext[:, 1].max())

    # The belt is sized to the FISH, not to the catch: a real machine does not
    # get longer because today's haul was bigger. Lanes across, a capped run
    # along, and whatever does not fit in one layer is dropped on top of what
    # does -- which is what a bulk infeed looks like anyway. It also keeps the
    # count sweep honest: at 128 fish the old rule asked for a 45 m conveyor.
    # Pile depth is not a cosmetic choice: it is the load on the fish at the
    # bottom, and it is what the solver has to hold up. Measured at 100 fish,
    # same solver settings, only the belt changed -- cross-fish penetrations
    # went from 403 pairs at 19.0 mm on a 14 m belt to 34 pairs at 5.2 mm on a
    # 26 m one, and the frame got FASTER (34 -> 46 fps) because a shallow spread
    # generates far fewer contacts than a deep heap.
    lanes = max(1, LANES)
    half_w = max(0.35, lanes * fish_wide * 0.62)
    pitch = fish_len * 1.10
    rows_wanted = int(math.ceil(n_fish / (lanes * max(1, LAYERS))))
    rows = max(1, min(rows_wanted, int((BED - 1.0) / pitch)))
    # Plus runway. The belt has to outlast the measurement: at 0.5 m/s a fish
    # covers 1.75 m during warmup + 120 timed frames -- 4.75 m over an 8 s A/B
    # window -- and a bed sized to hold the catch and nothing more conveys it off
    # the end mid-window, which reads as "2 of 4 on belt" and a conveying number
    # averaged over fish lying on the floor.
    runway = BELT_SPEED * (WARMUP + MEASURED_FRAMES) / FPS + 1.0
    bed_len = max(2.0, rows * pitch + runway)
    per_layer = lanes * rows
    layer_h = 1.15 * float(ext[:, 2].max())
    layers = int(math.ceil(n_fish / per_layer))
    # Yaw as far as the rails allow: a fish turned by y takes L*sin(y) +
    # W*cos(y) across the belt, and the rails are unforgiving.
    room = (1.7 * half_w - fish_wide) / max(fish_len, 1e-6)
    yaw_max = min(math.radians(28.0), math.asin(float(np.clip(room, 0.0, 1.0))))

    spacing = 2.05 * ROLLER_R          # cylinders nearly touching: a tight bed
    n_rollers = max(2, int(bed_len / spacing))
    roller_x = [-0.5 + (i + 0.5) * spacing for i in range(n_rollers)]

    # Side guards sized to the heap, not to one fish: pile five layers of cod
    # behind a 240 mm rail and you are just measuring how fast they fall off.
    rail_y = half_w + 0.02
    rail_h = float(np.clip(0.55 * layers * layer_h, 0.12, 0.30))

    spawn = np.zeros((n_fish, 3), np.float32)
    yaw = np.zeros(n_fish, np.float32)
    scale = np.ones(n_fish, np.float32)
    for k in range(n_fish):
        fish = fishes[int(variant[k])]
        layer, slot = divmod(k, per_layer)
        lane, row = slot % lanes, slot // lanes
        s = 1.0 + float(rng.uniform(-SCALE_JITTER, SCALE_JITTER)) if SCALE_JITTER > 0 else 1.0
        y_rot = float(rng.uniform(-yaw_max, yaw_max)) + (math.pi if rng.random() < 0.5 else 0.0)
        x = 0.5 + row * pitch + float(rng.uniform(-0.08, 0.08)) * pitch
        y = ((lane - 0.5 * (lanes - 1)) * (2.0 * half_w / max(lanes, 1)) * 0.88
             + float(rng.uniform(-0.06, 0.06)) * half_w)
        z = 0.02 + 0.5 * s * float(fish.extent[2]) + layer * layer_h
        spawn[k] = (x, y, z)
        yaw[k] = y_rot
        scale[k] = s

    part_count = np.array([len(fishes[int(v)].cverts) for v in variant], np.int32)
    part_base = np.concatenate([[0], np.cumsum(part_count)[:-1]]).astype(np.int32)

    return Layout(lanes=lanes, half_w=half_w, bed_len=bed_len, rail_y=rail_y, rail_h=rail_h,
                  roller_x=roller_x, n_rollers=n_rollers, per_layer=per_layer, layer_h=layer_h,
                  layers=layers, pitch=pitch, spawn=spawn, yaw=yaw, scale=scale,
                  part_base=part_base, part_count=part_count,
                  n_particles=int(part_base[-1] + part_count[-1]))


def deal_variants(n_fish, n_variants):
    """Fish k is scan k % n_variants -- the same deal for every engine."""
    return np.arange(n_fish) % n_variants


# -- Quality: how well the fish behaved, not how fast they ran ----------------

@wp.kernel
def _count_overlaps(pos: wp.array(dtype=wp.vec3), grid: wp.uint64, owner: wp.array(dtype=int),
                    thick: float, hits: wp.array(dtype=int), worst: wp.array(dtype=float)):
    i = wp.tid()
    p = pos[i]
    mine = owner[i]
    q = wp.hash_grid_query(grid, p, thick)
    for j in q:
        if owner[j] != mine and j > i:
            d = wp.length(p - pos[j])
            if d < thick:
                wp.atomic_add(hits, 0, 1)
                wp.atomic_max(worst, 0, thick - d)


PENETRATION_THICKNESS = 0.030   # m. FIXED, so the number compares across configs


def measure(positions, fishes, variant, layout, device):
    """Five numbers describing what the solver did to the fish.

    inverted    tets with negative volume            -> disintegration
    volume      per-fish volume / rest volume        -> squash, and blow-up
    stretch     worst tet edge stretch ratio         -> "jelly"
    droop       fish bbox length / rest length       -> "jelly" (a fish is not a rope)
    penetration cross-fish cage-vertex pairs closer than a FIXED 30 mm, and the
                worst of those depths -> "stuck in each other"

    The penetration thickness is deliberately not derived from the cage pitch:
    two engines with different tetrahedralisations would then be scored against
    different rulers, and the whole point is one ruler.
    """
    q = np.asarray(positions, np.float64)
    n_fish = len(variant)
    finite = bool(np.isfinite(q).all())
    inverted = 0
    vr_lo, vr_hi, stretch, droop_lo = 1e9, 0.0, 0.0, 1e9
    for k in range(n_fish):
        f = fishes[int(variant[k])]
        b, c = int(layout.part_base[k]), int(layout.part_count[k])
        p = q[b:b + c]
        if not np.isfinite(p).all():
            continue
        t = p[f.ctets]
        vol = np.einsum("ij,ij->i", np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]),
                        t[:, 3] - t[:, 0]) / 6.0
        inverted += int((vol * np.sign(f.rest_vol.sum()) < 0).sum())
        tot, rtot = abs(vol.sum()), abs(f.rest_vol.sum())
        vr_lo, vr_hi = min(vr_lo, tot / rtot), max(vr_hi, tot / rtot)
        for e, (a, bb) in enumerate(_EDGES):
            cur = np.linalg.norm(t[:, a] - t[:, bb], axis=1)
            stretch = max(stretch, float((cur / np.maximum(f.rest_edge[:, e], 1e-9)).max()))
        droop_lo = min(droop_lo, float((p.max(0) - p.min(0))[0] / f.rest_len))

    npart = int(layout.n_particles)
    owner = wp.array(np.repeat(np.arange(n_fish, dtype=np.int32), layout.part_count),
                     dtype=int, device=device)
    grid = wp.HashGrid(64, 64, 64, device)
    pos = wp.array(np.ascontiguousarray(q[:npart], np.float32), dtype=wp.vec3, device=device)
    grid.build(points=pos, radius=PENETRATION_THICKNESS)
    hits = wp.zeros(1, dtype=int, device=device)
    worst = wp.zeros(1, dtype=float, device=device)
    wp.launch(_count_overlaps, dim=npart, device=device,
              inputs=[pos, grid.id, owner, PENETRATION_THICKNESS], outputs=[hits, worst])
    wp.synchronize_device(device)

    return {
        "finite": finite,
        "inverted": inverted,
        "vol_lo": float(vr_lo), "vol_hi": float(vr_hi),
        "stretch": float(stretch),
        "droop": float(droop_lo),
        "pen_pairs": int(hits.numpy()[0]),
        "pen_worst_mm": float(worst.numpy()[0]) * 1000.0,
    }

# -- The set: a processing hall, lit like one ---------------------------------

# ── The set: a processing hall, lit like one ─────────────────────────────────

def _encode_rgbe(rgb):
    """Vectorised linear-RGB float -> Radiance RGBE bytes, shape (H, W, 4)."""
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


def make_hall_hdr(path, W=2048, H=1024):
    """An equirect HDR of a processing hall: dark walls, ceiling strip lights.

    Not a sky. A wet fish is mostly specular, and what a specular surface shows
    you is the room -- so the room is the light. Four long ceiling strips give
    the drawn-out highlights along the flank that read as "wet"; a sky would
    give one round sun blob and the fish would look like painted plastic.
    """
    j = np.arange(H).reshape(H, 1)
    i = np.arange(W).reshape(1, W)
    theta = (j / H) * math.pi                    # 0 = straight up
    phi = (i / W) * 2 * math.pi - math.pi
    y = np.broadcast_to(np.cos(theta), (H, W))   # +1 zenith .. -1 nadir
    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)
    z = sin_t * np.sin(phi)

    up = np.clip(y, 0.0, 1.0)[..., None]
    down = np.clip(-y, 0.0, 1.0)[..., None]
    env = (np.array([0.055, 0.062, 0.072]) * (1.0 - up - down)      # walls
           + np.array([0.155, 0.168, 0.186]) * up                    # ceiling
           + np.array([0.022, 0.023, 0.025]) * down)                 # floor

    # Strip lights on a ceiling 3.2 m up, running along the hall (the belt's
    # axis), traced properly: where a ray through this texel hits the ceiling
    # plane. That is what puts a straight highlight on a curved wet flank.
    ceil_h, half_len, half_wide = 3.2, 7.0, 0.26
    t = ceil_h / np.maximum(y, 1e-4)
    hx, hz = x * t, z * t
    lit = np.zeros((H, W))
    for centre in (-3.0, -1.0, 1.0, 3.0):
        band = np.clip(1.0 - ((hz - centre) / half_wide) ** 2, 0.0, 1.0) ** 1.2
        run = np.clip(1.0 - (hx / half_len) ** 8, 0.0, 1.0)
        lit += band * run
    lit = np.where(y > 0.02, lit, 0.0)
    env = env + (lit * 34.0)[..., None] * np.array([1.0, 0.97, 0.92])

    # A lit end wall, so the room has a far side instead of a black void, and
    # the fish get a cool rim from behind.
    wall = np.clip(0.5 + 0.5 * np.cos(phi - 2.0), 0.0, 1.0) ** 6
    wall = wall * np.clip(1.0 - 2.2 * abs(y + 0.12), 0.0, 1.0)
    env = env + (wall * 1.4)[..., None] * np.array([0.62, 0.72, 0.92])

    rgbe = _encode_rgbe(env)
    # stb reads scanlines uncompressed only if the first pixel is not an RLE
    # run marker (2, 2, <128).
    if rgbe[0, 0, 0] == 2 and rgbe[0, 0, 1] == 2 and rgbe[0, 0, 2] < 128:
        rgbe[0, 0, 0] = 3
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (H, W))
        f.write(rgbe.tobytes())
    return path


def fish_material(fish):
    """The scan's own baked maps, on a wet-fish BSDF.

    The loader hands back a MeshStandardMaterial at roughness 0.9 -- correct for
    the dry matte thing a photogrammetry rig scanned, wrong for a fish that came
    out of ice ten minutes ago. Physical + clearcoat puts the water film back:
    a thin smooth coat over a rough body is exactly what wet skin is.
    """
    m = tp.MeshPhysicalMaterial()
    if CAGE or fish.material is None:
        m.color = 0xC08878
        m.roughness = 0.55
        m.metalness = 0.0
        m.flat_shading = CAGE
        return m
    src = fish.material
    for slot in ("map", "normal_map", "ao_map"):
        tex = getattr(src, slot, None)
        if tex is not None:
            setattr(m, slot, tex)
    m.roughness = 0.52
    m.metalness = 0.0
    m.clearcoat = 0.85
    m.clearcoat_roughness = 0.10
    m.env_map_intensity = 1.0
    # The scans are open surfaces -- there is a hole where the fish lay on the
    # rig -- so the far side of it has to draw or the fish reads as hollow.
    m.side = tp.Side.Double
    return m


def steel(color, metalness=0.9, roughness=0.3, flat=False, env=1.0):
    m = tp.MeshStandardMaterial()
    m.color = color
    m.metalness = metalness
    m.roughness = roughness
    m.flat_shading = flat
    m.env_map_intensity = env
    return m

def build_scene(sim, fishes, renderer, canvas, device):
    """The hall, the machine, and one School per scan. Returns (scene, camera, draw).

    `sim` is duck-typed on purpose -- it is whatever the engine's own driver
    class is, and all this needs from it is .layout, .variant, .omega,
    .sim_time() and .positions(). That is the entire surface the two solvers
    have to agree on to be drawn by the same code.
    """
    L = sim.layout
    scene = tp.Scene()
    env = tp.RGBELoader().load(make_hall_hdr(
        os.path.join(tempfile.gettempdir(), "threepp_fish_hall.hdr")))
    scene.environment = env
    # The environment lights the scene but is NOT the backdrop: a synthetic room
    # painted as a sky reads as an empty studio. A dark ground the floor fogs
    # into keeps the eye on the belt.
    scene.background = 0x0F1216
    scene.set_fog(tp.Color(0x0F1216), 2.0, 17.0)

    # The sim is Z-up (both engines are run that way here); three.js lighting, shadows and the environment map are all
    # Y-up. One group carries the whole conveyor from one to the other, so
    # everything below is authored in the sim's own coordinates and everything
    # outside it -- lights, camera -- is ordinary Y-up.
    rig = tp.Group()
    rig.rotate_x(-math.pi / 2)
    scene.add(rig)

    mid_x = 0.5 * L.bed_len - 0.5
    aim = tp.Object3D()
    aim.position.set(mid_x, 0.6, 0.0)
    scene.add(aim)

    key = tp.DirectionalLight(0xFFF1DC, 2.9)
    key.position.set(mid_x + 1.5, 5.0, 3.0)
    key.cast_shadow = True
    key.set_target(aim)
    reach = 0.6 * max(L.bed_len, 4.0 * L.half_w)
    key.set_shadow_frustum(-reach, reach, reach, -reach)
    key.set_shadow_bias(-0.00035)
    scene.add(key)
    fill = tp.DirectionalLight(0x9FC0E8, 0.7)
    fill.position.set(mid_x - 3.0, 2.0, -3.5)
    scene.add(fill)
    rim = tp.DirectionalLight(0xBBD4FF, 1.5)
    rim.position.set(mid_x - 4.0, 1.6, 2.5)
    rim.set_target(aim)
    scene.add(rim)
    scene.add(tp.HemisphereLight(0xAFC6DE, 0x14161A, 0.35))

    # --- the machine, in sim coordinates ---
    frame = steel(0x394049, 0.55, 0.5)
    rail_mat = steel(0x9AA3AC, 0.92, 0.28)
    roller_mat = steel(0x46505A, 0.90, 0.35, flat=True, env=0.9)

    rollers, slats = [], []
    slat_pitch, slat_h = 0.085, 0.018
    # The visual belt ends where the PHYSICS ends (bed_x1 = bed_len - 0.5 in
    # rig coordinates): an overhanging visual gives fish that appear to fall
    # through a belt that is still drawn under them.
    belt_span = L.bed_len
    if BELT_STYLE == "flat":
        # A plastic modular belt: slats gliding over a static bed plate. The
        # visible slat seams moving down the line are what sells that the belt
        # itself moves -- a featureless plane under moving fish reads as ice.
        slat_geo = tp.BoxGeometry(slat_pitch - 0.007, 2.0 * L.half_w, slat_h)
        slat_mat = steel(0x2E4A5A, 0.0, 0.55, env=0.6)   # blue food-grade plastic
        n_slat = int(belt_span / slat_pitch) + 1
        for i in range(n_slat):
            m = tp.Mesh(slat_geo, slat_mat)
            m.position.set(-0.5 + i * slat_pitch, 0.0, -0.5 * slat_h)
            m.cast_shadow = True
            m.receive_shadow = True
            rig.add(m)
            slats.append(m)
        plate = tp.Mesh(tp.BoxGeometry(belt_span, 2.0 * L.half_w + 0.04, 0.03),
                        steel(0x232A31, 0.3, 0.75))
        plate.position.set(-0.5 + 0.5 * belt_span - 0.5 * slat_pitch, 0.0,
                           -slat_h - 0.018)
        plate.receive_shadow = True
        rig.add(plate)
    else:
        # A flat-shaded prism, not a smooth cylinder. Circumscribed, so it
        # touches the true collider along the contact line rather than lying
        # about the radius -- and its facets are the only reason you can see
        # that the rollers are turning at all. A smooth cylinder at this
        # spacing renders as one corrugated sheet that never appears to move.
        sides = 24
        roller_geo = tp.CylinderGeometry(ROLLER_R / math.cos(math.pi / sides),
                                         ROLLER_R / math.cos(math.pi / sides),
                                         2.0 * L.half_w, sides)
        for x in L.roller_x:
            m = tp.Mesh(roller_geo, roller_mat)
            m.position.set(x, 0.0, -ROLLER_R)
            # CylinderGeometry's axis is +Y; the rollers lie along world +Y
            # already, so no rotation -- unlike a Y-up scene.
            m.cast_shadow = True
            m.receive_shadow = True
            rig.add(m)
            rollers.append(m)

    for s in (-1.0, 1.0):
        rail = tp.Mesh(tp.BoxGeometry(L.bed_len, 0.04, 2.0 * L.rail_h), rail_mat)
        rail.position.set(mid_x, s * L.rail_y, 0.10)
        rail.cast_shadow = True
        rail.receive_shadow = True
        rig.add(rail)

        beam = tp.Mesh(tp.BoxGeometry(L.bed_len, 0.06, 0.16), frame)
        beam.position.set(mid_x, s * (L.half_w + 0.06), -2.2 * ROLLER_R)
        beam.cast_shadow = True
        beam.receive_shadow = True
        rig.add(beam)

        for lx in np.linspace(0.0, L.bed_len - 1.0, max(2, int(L.bed_len / 1.2))):
            leg = tp.Mesh(tp.BoxGeometry(0.07, 0.07, 1.0 - 2.6 * ROLLER_R), frame)
            leg.position.set(lx, s * (L.half_w + 0.06), -0.5 - 1.3 * ROLLER_R)
            leg.cast_shadow = True
            rig.add(leg)

    floor = tp.Mesh(tp.PlaneGeometry(40, 40), steel(0x23272C, 0.0, 0.85))
    floor.position.set(mid_x, 0.0, -1.0)
    floor.receive_shadow = True
    rig.add(floor)

    # A corner of the hall. Two panels are enough: one across the far end of the
    # belt and one down its far side. Without them the belt floats in a black
    # void, which is a look, but not the look of a plant. The fog swallows both
    # before their edges, so the room has no visible seams.
    hall = steel(0x424B55, 0.0, 0.72)
    for size, pos in (((0.12, 26.0, 4.2), (-5.0, 0.0, 1.1)),
                      ((26.0, 0.12, 4.2), (mid_x, L.half_w + 4.0, 1.1))):
        wall = tp.Mesh(tp.BoxGeometry(*size), hall)
        wall.position.set(*pos)
        wall.receive_shadow = True
        rig.add(wall)

    # The fixtures the environment map is already lighting from. Nothing sees
    # them in the default framing -- they are for when you orbit up, and for the
    # reflection they put down the length of a wet fish.
    lamp = tp.MeshBasicMaterial()
    lamp.color = 0xFFF6E8
    # Two lamps for the room itself. The key light is aimed at the belt and the
    # walls would otherwise sit in the dark, which reads as a void with a wall
    # painted on it rather than as a room.
    for px in (0.35 * L.bed_len, 0.9 * L.bed_len):
        bulb = tp.PointLight(0xFFF3E2, 14.0, 16.0, 2.0)
        bulb.position.set(px, 0.0, 2.7)
        rig.add(bulb)
    for ly in (-1.0, 1.0):
        for lx in np.linspace(0.4, L.bed_len - 1.0, max(2, int(L.bed_len / 1.8))):
            fixture = tp.Mesh(tp.BoxGeometry(1.4, 0.16, 0.06), lamp)
            fixture.position.set(lx, ly * max(0.9, 0.7 * L.half_w), 2.9)
            rig.add(fixture)

    # --- the catch ---
    schools = []
    for v, fish in enumerate(fishes):
        idx = np.flatnonzero(sim.variant == v)
        if not len(idx):
            continue
        school = School(fish, idx, L.part_base, device)
        rig.add(school.mesh)
        schools.append(school)

    # Both rise with the pile: a belt loaded five fish deep behind 600 mm guards
    # hides its own load from the camera that framed it empty. And the camera
    # stands BEYOND the end of the bed, aimed slightly down the line, so the
    # drop zone -- fish ride off the end and lie on the floor -- is in frame
    # rather than directly underfoot.
    eye_h = 1.35 + 2.4 * L.rail_h
    end_x = L.bed_len - 0.5
    camera = tp.PerspectiveCamera(42, canvas.aspect(), 0.05, 100)
    camera.position.set(end_x + 3.4, eye_h, 1.85 + 0.75 * L.half_w)
    camera.look_at(mid_x + 0.24 * L.bed_len, -0.28, 0.0)

    def draw():
        if slats:
            run = sim.sim_time() * BELT_SPEED
            for i, m in enumerate(slats):
                m.position.x = -0.5 + (i * slat_pitch + run) % belt_span
        else:
            angle = sim.omega * sim.sim_time()
            for m in rollers:
                m.rotation.y = angle
        q = sim.positions()
        for s in schools:
            s.update(q, renderer)

    return scene, camera, draw


def stage(sim, fishes, headless, device, title="fish conveyor"):
    canvas = tp.Canvas(title, width=1600, height=900,
                       antialiasing=4, headless=headless)
    renderer = tp.GLRenderer(canvas)
    renderer.shadow_map_enabled = True
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    renderer.tone_mapping_exposure = 0.95
    scene, camera, draw = build_scene(sim, fishes, renderer, canvas, device)
    return canvas, renderer, scene, camera, draw


# -- The A/B run: one config, one JSON line ----------------------------------

def vram_mb(device):
    try:
        d = wp.get_device(device)
        return (d.total_memory - d.free_memory) / (1 << 20)
    except Exception:
        return float("nan")


def ab_run(sim, fishes, device, seconds, engine, seed, title,
           shot=None, shot_at=None, warmup=None):
    """Settle, convey, and report -- the identical measured path for both engines.

    The clock runs over the WHOLE window rather than per frame, because what is
    being compared is throughput on one machine and a per-frame timer around an
    asynchronous GL submit measures the submit, not the frame. Skinning and a
    headless render are inside the window on purpose: a solver that is quick and
    then needs a host round trip to be drawn has not saved anyone anything, and
    that difference is exactly what the two engines disagree about.

    Warm-up is untimed (kernel compile, and the catch dropping onto the rollers).
    Nothing is recycled during the window: conveying is measured as displacement
    over it, and a fish teleported back to the start reads as the belt running
    hard backwards.
    """
    canvas, renderer, scene, camera, draw = stage(sim, fishes, True, device, title)
    for _ in range(WARMUP if warmup is None else warmup):
        sim.step()
    draw()
    renderer.render(scene, camera)      # arms the GL interop on frame one
    wp.synchronize_device(device)

    x0 = sim.centroids()[:, 0].copy()
    used0 = vram_mb(device)
    n = int(round(seconds * FPS))
    shot_frame = None if shot is None else int(round((shot_at or seconds) * FPS))

    t0 = time.perf_counter()
    for k in range(n):
        sim.step()
        draw()
        renderer.render(scene, camera)
        if shot_frame is not None and k + 1 == shot_frame:
            wp.synchronize_device(device)
            renderer.save_frame(shot)
    wp.synchronize_device(device)
    ms = (time.perf_counter() - t0) * 1000.0 / max(n, 1)

    q = np.asarray(sim.positions_host())
    c = sim.centroids()
    dx = c[:, 0] - x0
    finite = np.isfinite(c).all(1)
    onbelt = finite & (c[:, 2] > -0.5)
    elapsed = n / FPS

    r = {
        "engine": engine,
        "count": int(sim.n_fish),
        "seed": int(seed),
        "ms_frame": float(ms),
        "fps": float(1000.0 / ms) if ms > 0 else float("nan"),
        "tets": int(sum(len(fishes[int(v)].ctets) for v in sim.variant)),
        "cage_verts": int(sim.layout.n_particles),
        "convey": float(np.mean(dx[onbelt]) / elapsed) if onbelt.any() else float("nan"),
        "belt": BELT_SPEED,
        "onbelt": int(onbelt.sum()),
        "vram": used0,
        "seconds": float(seconds),
    }
    r.update(measure(q, fishes, sim.variant, sim.layout, device))
    if shot:
        r["shot"] = shot
    return r


def ab_report(r):
    """One human line, then the JSON line the harness parses."""
    import json
    print(f"  {r['engine']:<18} {r['count']:>4} fish seed {r['seed']} | "
          f"{r['ms_frame']:7.2f} ms {r['fps']:5.1f} fps | inv {r['inverted']:>5} | "
          f"vol {r['vol_lo']:4.2f}-{r['vol_hi']:4.2f} | stretch {r['stretch']:5.2f} | "
          f"droop {r['droop']:4.2f} | pen {r['pen_pairs']:>6} worst {r['pen_worst_mm']:5.1f} mm | "
          f"convey {r['convey']:+.3f} of {r['belt']:.2f} | {r['onbelt']}/{r['count']} on belt | "
          f"{r['vram']:.0f} MB{'' if r['finite'] else '  NaN!'}", flush=True)
    print("AB_JSON " + json.dumps(r), flush=True)
