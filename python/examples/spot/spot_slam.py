"""Spot SLAM — procedural terrain + trees + depth camera + live map reconstruction.

Spot walks on TerrainGenerator terrain while a body-mounted forward depth camera
accumulates a 3-D point cloud. Every ~3 s a background thread runs marching cubes
to reconstruct the growing SLAM surface (semi-transparent blue) over the ground truth.

    python spot_slam.py
    python spot_slam.py --seed 7 --amplitude 0.20
    python spot_slam.py --shot out.png

Controls: W/S = fwd/back  A/D = strafe  Q/E = turn  |  SPACE = toggle auto-forward  |  R = reset
"""
import argparse, math, os, sys, threading, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import threepp as tp
from threepp.rl import load_policy
from spot_deploy import (build_spot, fetch_assets,
                         _quat_to_R, _quat_from_R,
                         default_q, isaac_to_add, add_to_isaac, ACTION_SCALE, Z0)
from spot_depth_scan import ForwardDepthScanner
from spot_terrain_env import VX_HI, VY_HI, WZ_HI

GRAV = np.array([0.0, 0.0, -1.0])

# ── constants ──────────────────────────────────────────────────────────────────
WORLD_SZ   = 80.0
AMPLITUDE  = 10.8
TREE_COUNT = 60
CLEAR_R    = 7.0     # no trees within this radius of spawn
SENSOR_W   = 160
SENSOR_H   = 120
SENSOR_FAR = 8.0
SCAN_EVERY = 3       # depth scan every N frames; result cached for policy (~17 Hz)
MC_FRAMES  = 90      # trigger SLAM rebuild every N rendered frames
GRASS_BLADES = 12000 # merged GrassMesh blade count (GPU-wind on Vulkan); tune for FPS
GRASS_RADIUS = 32.0  # grass disk radius around spawn (fog hides >25 m anyway)
HDR_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/noon_grass_2k.hdr"


def fetch_hdr(url, name):
    """Download an HDRI to ~/.cache/threepp/hdri once (cached) and return its path."""
    import pathlib, shutil, urllib.request
    cache = pathlib.Path.home() / ".cache" / "threepp" / "hdri"
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / name
    if dest.exists() and dest.stat().st_size > 0:
        return str(dest)
    print(f"[hdri] downloading {name} ...")
    req = urllib.request.Request(url, headers={"User-Agent": "threepp-spot-demo"})
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    tmp.replace(dest)
    return str(dest)


# ── terrain ────────────────────────────────────────────────────────────────────
def build_terrain_zup(field, world_size, amplitude):
    """Vectorised Z-up triangle soup (XY ground plane, Z = height) with UVs."""
    dim  = field.shape[0]
    half = world_size / 2.0
    lin  = np.linspace(-half, half, dim, dtype=np.float32)
    X, Y = np.meshgrid(lin, lin, indexing='ij')
    # field is stored field_[iz][ix] (Z outer, X inner) — transpose so first index → world X
    Z    = (field.T * amplitude).astype(np.float32)
    U, V = np.meshgrid(np.linspace(0, 1, dim, dtype=np.float32),
                       np.linspace(0, 1, dim, dtype=np.float32), indexing='ij')

    ii, jj = np.meshgrid(np.arange(dim - 1), np.arange(dim - 1), indexing='ij')
    ii, jj = ii.ravel(), jj.ravel()

    def corner(a, b):
        return np.stack([X[a, b], Y[a, b], Z[a, b]], 1)
    def uvcorner(a, b):
        return np.stack([U[a, b], V[a, b]], 1)

    v00, v10, v11, v01 = corner(ii, jj), corner(ii+1, jj), corner(ii+1, jj+1), corner(ii, jj+1)
    u00, u10, u11, u01 = uvcorner(ii, jj), uvcorner(ii+1, jj), uvcorner(ii+1, jj+1), uvcorner(ii, jj+1)

    verts = np.vstack([np.stack([v00, v10, v11], 1).reshape(-1, 3),
                       np.stack([v00, v11, v01], 1).reshape(-1, 3)])
    uvs   = np.vstack([np.stack([u00, u10, u11], 1).reshape(-1, 2),
                       np.stack([u00, u11, u01], 1).reshape(-1, 2)])

    g = tp.BufferGeometry()
    g.set_attribute("position", np.ascontiguousarray(verts))
    g.set_attribute("uv",       np.ascontiguousarray(uvs))
    g.compute_vertex_normals()
    return g


# ── scatter helpers ──────────────────────────────────────────────────────────────
def _in_pond(px, py, hz, pond):
    """True if (px,py) at terrain height hz is at/below the pond waterline and near
    its centre — used to keep vegetation out of open water. pond = (cx,cy,wl)|None."""
    if pond is None:
        return False
    cx, cy, wl = pond
    return math.hypot(px - cx, py - cy) < 17.0 and hz < wl + 0.2


# ── trees ──────────────────────────────────────────────────────────────────────
def scatter_trees(scene, gen, params, n=TREE_COUNT, seed=0, world=None, pond=None):
    """Scatter trees. When `world` is given, add a static box collider per trunk
    (a "tree stub") so Spot bumps into the trunks; the leafy canopy stays
    non-colliding. Returns the list of collider proxy meshes (keep them alive)."""
    rng  = np.random.default_rng(seed)
    half = params.world_size / 2.0 - 4.0
    proxies = []

    bark_alb, _ = tp.make_bark_textures(128, seed, [0.34, 0.22, 0.12])
    trunk_mat = tp.MeshStandardMaterial()
    trunk_mat.map       = bark_alb
    trunk_mat.roughness = 0.85

    leaf_mat = tp.MeshStandardMaterial()
    leaf_mat.map           = tp.make_leaf_texture(256, seed, [0.18, 0.42, 0.14])  # RGBA leaf-cluster cutout
    leaf_mat.alpha_test    = 0.5     # discard the transparent gaps → leafy silhouette, not solid cards
    leaf_mat.roughness     = 0.9
    leaf_mat.side          = tp.Side.Double
    leaf_mat.vertex_colors = True    # per-leaf tint variation baked into the geometry

    tpar = tp.TreeParams()
    tp.apply_tree_preset(0, tpar)     # Oak baseline
    tpar.max_iterations = 120         # cheaper for many trees
    tpar.trunk_height   = 4.0
    tpar.crown_radius_x = 2.5
    tpar.crown_radius_z = 2.5
    tpar.crown_height   = 3.0

    tgen = tp.TreeGenerator(0)        # reseeded per tree
    placed = 0
    attempts = 0
    while placed < n and attempts < n * 15:
        attempts += 1
        px, py = float(rng.uniform(-half, half)), float(rng.uniform(-half, half))
        if math.hypot(px, py) < CLEAR_R:
            continue
        hz = float(gen.height_at(px, py, params))
        if _in_pond(px, py, hz, pond):       # don't plant trees in open water
            continue
        tree_seed = int(rng.integers(0, 100_000))
        tgen.reseed(tree_seed)
        tgen.build_skeleton(tpar)
        trunk_geo = tgen.make_trunk_geometry(tpar)
        leaf_geo  = tgen.make_leaf_geometry(tpar)
        scale = float(rng.uniform(0.7, 1.3))
        for geo, mat in ((trunk_geo, trunk_mat), (leaf_geo, leaf_mat)):
            m = tp.Mesh(geo, mat)
            m.rotation.x = math.pi / 2   # Y-up tree → Z-up world
            m.position.set(px, py, hz)
            m.scale.set(scale, scale, scale)
            m.cast_shadow = True
            scene.add(m)
        if world is not None:
            # Tall thin box collider on the lower trunk (covers Spot's height);
            # the canopy above it is leaves → no collision.
            tw  = 0.35 * scale
            stub = tp.Mesh(tp.BoxGeometry(tw, tw, 2.5), trunk_mat)
            stub.position.set(px, py, hz + 1.25)
            world.add_static(stub)
            proxies.append(stub)
        placed += 1
    print(f"[trees] placed {placed}/{n}")
    return proxies


# ── stones ───────────────────────────────────────────────────────────────────────
def make_rock_geometry(seed):
    """Low-poly faceted boulder: a 5x7 sphere displaced by a few smooth lumps
    (port of forest_demo's makeRock). Non-indexed soup; pair with flat_shading."""
    rng = np.random.default_rng(seed)
    lat_segs, lon_segs = 5, 7
    p1, p2, p3 = rng.uniform(-math.pi, math.pi, 3)
    theta = np.linspace(0, math.pi, lat_segs + 1, dtype=np.float32)
    phi   = np.linspace(0, 2 * math.pi, lon_segs + 1, dtype=np.float32)
    T, P  = np.meshgrid(theta, phi, indexing='ij')              # (6,8)
    sinT, cosT = np.sin(T), np.cos(T)
    nx, ny, nz = sinT * np.cos(P), cosT, sinT * np.sin(P)
    disp = (1.0 + 0.30 * np.sin(2 * P + p1) * sinT + 0.24 * np.cos(3 * P + p2)
            + 0.22 * np.sin(3 * T + p3) + 0.14 * np.cos(5 * P + 4 * T + p1))
    disp = np.clip(disp, 0.6, 1.5)
    pos = np.stack([nx * disp, ny * disp, nz * disp], -1).reshape(-1, 3)
    nrm = np.stack([nx, ny, nz], -1).reshape(-1, 3)
    rowV = lon_segs + 1
    idx = []
    for la in range(lat_segs):
        for lo in range(lon_segs):
            a = la * rowV + lo; b = a + rowV
            idx += [a, a + 1, b, a + 1, b + 1, b]            # CCW from outside
    idx = np.array(idx, np.int64)
    g = tp.BufferGeometry()
    g.set_attribute("position", np.ascontiguousarray(pos[idx], np.float32))
    g.set_attribute("normal",   np.ascontiguousarray(nrm[idx], np.float32))
    return g


def scatter_stones(scene, gen, params, n=35, seed=0, world=None, pond=None):
    """Scatter boulders. When `world` is given, add a static sphere collider per
    stone so Spot bumps into them. Returns the collider proxy meshes."""
    rng  = np.random.default_rng(seed + 7)
    half = params.world_size / 2.0 - 4.0
    proxies = []
    rgeos = [make_rock_geometry(s) for s in (1, 2, 3)]
    mat = tp.MeshStandardMaterial()
    mat.color = 0x4e4a44
    mat.roughness = 1.0
    mat.flat_shading = True            # crisp facets on the low-poly boulders
    placed = attempts = 0
    while placed < n and attempts < n * 15:
        attempts += 1
        px, py = float(rng.uniform(-half, half)), float(rng.uniform(-half, half))
        if math.hypot(px, py) < 2.5:
            continue
        hz = float(gen.height_at(px, py, params))
        if _in_pond(px, py, hz, pond):
            continue
        s  = 0.30 + float(rng.uniform()) * 0.55
        m  = tp.Mesh(rgeos[int(rng.integers(0, len(rgeos)))], mat)
        m.position.set(px, py, hz + s * 0.30)   # mostly proud of the ground, partly embedded
        m.scale.set(s, s, s * 0.8)              # slightly squashed boulders
        m.rotation.x = float(rng.uniform(0, 2 * math.pi))
        m.rotation.y = float(rng.uniform(0, 2 * math.pi))
        m.rotation.z = float(rng.uniform(0, 2 * math.pi))
        m.cast_shadow = True
        m.receive_shadow = True
        scene.add(m)
        if world is not None:
            sph = tp.Mesh(tp.SphereGeometry(s * 0.8, 8, 6), mat)
            sph.position.set(px, py, hz + s * 0.30)
            world.add_static(sph)
            proxies.append(sph)
        placed += 1
    print(f"[stones] placed {placed}/{n}")
    return proxies


# ── bushes (shrub-variant trees, like forest_demo) ────────────────────────────────
def scatter_bushes(scene, gen, params, n=50, seed=0, pond=None):
    rng  = np.random.default_rng(seed + 3)
    half = params.world_size / 2.0 - 4.0

    bark_alb, _ = tp.make_bark_textures(64, seed, [0.30, 0.22, 0.12])
    trunk_mat = tp.MeshStandardMaterial(); trunk_mat.map = bark_alb; trunk_mat.roughness = 0.9
    leaf_mat  = tp.MeshStandardMaterial()
    leaf_mat.map           = tp.make_leaf_texture(256, seed + 5, [0.14, 0.36, 0.12])
    leaf_mat.alpha_test    = 0.5
    leaf_mat.roughness     = 0.9
    leaf_mat.side          = tp.Side.Double
    leaf_mat.vertex_colors = True

    # A few reusable shrub variants (short trunk, wide low hemisphere crown).
    tgen = tp.TreeGenerator(0)
    variants = []
    for s in (11, 22, 33):
        tpar = tp.TreeParams()
        tpar.seed = s
        tpar.trunk_height = 0.45; tpar.trunk_radius = 0.06
        tpar.crown_shape = tp.CrownShape.Hemisphere
        tpar.crown_radius_x = 1.1; tpar.crown_radius_z = 1.1; tpar.crown_height = 1.2
        tpar.attractor_count = 260; tpar.influence_distance = 2.0; tpar.kill_distance = 0.45
        tpar.segment_length = 0.22; tpar.max_iterations = 130; tpar.randomness = 0.12
        tpar.radial_segments = 5
        tpar.leaf_style = tp.LeafStyle.CrossQuad
        tpar.leaf_size = 0.45; tpar.leaf_density = 0.95
        tpar.leaves_per_cluster = 5; tpar.leaf_spread = 0.35
        tgen.reseed(s); tgen.build_skeleton(tpar)
        variants.append((tgen.make_trunk_geometry(tpar), tgen.make_leaf_geometry(tpar)))

    placed = attempts = 0
    while placed < n and attempts < n * 15:
        attempts += 1
        px, py = float(rng.uniform(-half, half)), float(rng.uniform(-half, half))
        if math.hypot(px, py) < 3.0:
            continue
        hz = float(gen.height_at(px, py, params))
        if _in_pond(px, py, hz, pond):       # keep bushes out of open water
            continue
        trunk_geo, leaf_geo = variants[int(rng.integers(0, len(variants)))]
        s  = 0.7 + float(rng.uniform()) * 0.7
        for geo, mat in ((trunk_geo, trunk_mat), (leaf_geo, leaf_mat)):
            m = tp.Mesh(geo, mat)
            m.rotation.x = math.pi / 2     # Y-up → Z-up (like the trees)
            m.position.set(px, py, hz)
            m.scale.set(s, s, s)
            m.cast_shadow = True
            scene.add(m)
        placed += 1
    print(f"[bushes] placed {placed}/{n}")


# ── grass ────────────────────────────────────────────────────────────────────────
def build_grass_field(gen, params, n_blades, radius, seed, clear_r=0.0, pond=None):
    """One merged GrassMesh geometry (non-indexed triangle soup) for the Vulkan
    GPU-wind path. Blades are Y-up in local space (the wind compute bends local
    X/Z, keeping Y), and the field is baked so rotation.x=pi/2 on the mesh stands
    them up at terrain height in the Z-up world — same convention as the trees.
    Returns (geometry, blade_count). Needs a per-vertex 'heightFrac' attribute
    (0 at base, 1 at tip) which drives the sway weighting."""
    rng = np.random.default_rng(seed + 4242)
    seg, wbase = 4, 0.05
    rows = seg + 1
    t = np.linspace(0.0, 1.0, rows, dtype=np.float32)          # height fraction per row
    w = wbase * (1.0 - t)                                       # taper to a point at the tip

    # 10-vertex blade template (Y-up): two verts (L/R) per row
    tpl_pos = np.empty((rows, 2, 3), np.float32)
    tpl_pos[:, 0] = np.stack([-w, t, np.zeros_like(t)], 1)
    tpl_pos[:, 1] = np.stack([ w, t, np.zeros_like(t)], 1)
    tpl_pos = tpl_pos.reshape(-1, 3)                            # (10,3)
    tpl_hf  = np.repeat(t, 2)                                   # heightFrac = blade-local y
    tpl_uv  = np.stack([np.tile([0.0, 1.0], rows), tpl_hf], 1).astype(np.float32)
    c_bot = np.array([0.06, 0.13, 0.04], np.float32)
    c_top = np.array([0.22, 0.40, 0.13], np.float32)
    tpl_col = (c_bot + (c_top - c_bot) * tpl_hf[:, None]).astype(np.float32)
    n0 = np.array([0.0, 0.85, 0.53], np.float32); n0 /= np.linalg.norm(n0)
    tpl_nrm = np.tile(n0, (rows * 2, 1))

    # expand 10 indexed verts → 24-vertex non-indexed soup (set_index isn't bound):
    # two triangles per segment, winding matching makeGrassField()
    soup = []
    for s in range(seg):
        a = s * 2
        soup += [a, a + 1, a + 2, a + 1, a + 3, a + 2]
    soup = np.array(soup, np.int64)                            # (24,)
    sp, sh, su, sc, sn = (tpl_pos[soup], tpl_hf[soup], tpl_uv[soup],
                          tpl_col[soup], tpl_nrm[soup])         # (24, …)
    V = soup.shape[0]

    # scatter blades in TUFTS (grass grows in clumps) over a disk, skipping a
    # clearing around spawn. One terrain-height query per tuft (blades in a tuft
    # are within a few cm, terrain is locally flat) keeps the build cheap.
    TUFT, SPREAD = 6, 0.12        # blades per tuft, max in-tuft offset (m)
    xs, ys, hs, yaws, sxs, hgts, tries = [], [], [], [], [], [], 0
    while len(xs) < n_blades and tries < n_blades * 4:
        tries += 1
        ang = rng.uniform(0, 2 * math.pi); rr = math.sqrt(rng.uniform(0, 1)) * radius
        cx, cy0 = math.cos(ang) * rr, math.sin(ang) * rr
        if math.hypot(cx, cy0) < clear_r:
            continue
        hz = float(gen.height_at(cx, cy0, params))
        if _in_pond(cx, cy0, hz, pond):      # no grass tufts in open water
            continue
        for _ in range(TUFT):
            xs.append(cx + rng.uniform(-SPREAD, SPREAD))
            ys.append(cy0 + rng.uniform(-SPREAD, SPREAD))
            hs.append(hz)
            yaws.append(rng.uniform(0, 2 * math.pi))
            sxs.append(0.7 + rng.uniform(0, 1) * 0.6)
            hgts.append(0.10 + rng.uniform(0, 1) * 0.14)   # short meadow grass
    n = len(xs)
    wx  = np.array(xs, np.float32);  wy = np.array(ys, np.float32);  h = np.array(hs, np.float32)
    yaw = np.array(yaws, np.float32); sx = np.array(sxs, np.float32); hg = np.array(hgts, np.float32)
    cy, sy = np.cos(yaw)[:, None], np.sin(yaw)[:, None]

    # per-blade: scale (sx,hg,sx) → R_y(yaw) → translate to merged-local (wx, h, -wy)
    P  = sp[None] * np.stack([sx, hg, sx], 1)[:, None, :]       # (n,V,3)
    px =  P[:, :, 0] * cy + P[:, :, 2] * sy + wx[:, None]
    py =  P[:, :, 1] +                         h[:, None]
    pz = -P[:, :, 0] * sy + P[:, :, 2] * cy + (-wy)[:, None]
    pos = np.stack([px, py, pz], -1).reshape(-1, 3).astype(np.float32)

    nx =  sn[None, :, 0] * cy + sn[None, :, 2] * sy
    ny =  np.tile(sn[:, 1][None], (n, 1))
    nz = -sn[None, :, 0] * sy + sn[None, :, 2] * cy
    nrm = np.stack([nx, ny, nz], -1).reshape(-1, 3).astype(np.float32)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)

    g = tp.BufferGeometry()
    g.set_attribute("position",   np.ascontiguousarray(pos))
    g.set_attribute("normal",     np.ascontiguousarray(nrm))
    g.set_attribute("uv",         np.ascontiguousarray(np.tile(su, (n, 1))))
    g.set_attribute("color",      np.ascontiguousarray(np.tile(sc, (n, 1))))
    g.set_attribute("heightFrac", np.ascontiguousarray(np.tile(sh, n)[:, None]))
    return g, n


# ── pond (Vulkan FFT ocean) ───────────────────────────────────────────────────────
def add_pond(scene, gen, params, max_r=20.0, exclude_r=7.0, depth=1.4):
    """Drop an FFT water surface into the deepest terrain basin within an annulus
    [exclude_r, max_r] around spawn (Vulkan only — tp.Ocean is absent on GL
    builds). Searching a disk (vs a forward box) lands in a real bowl with terrain
    rising on all sides, so the flat water plane reads as a contained pond rather
    than a hillside flood. Returns (pond, water_level) or (None, None). The
    renderer auto-animates the waves each frame."""
    if not hasattr(tp, "Ocean"):
        print("[pond] skipped (needs the Vulkan build)")
        return None, None
    best = None
    for x in np.linspace(-max_r, max_r, 49):
        for y in np.linspace(-max_r, max_r, 49):
            r = math.hypot(float(x), float(y))
            if not (exclude_r <= r <= max_r):
                continue
            h = float(gen.height_at(float(x), float(y), params))
            if best is None or h < best[2]:
                best = (float(x), float(y), h)
    lx, ly, lh = best
    wl = lh + depth
    pond = tp.Ocean(size=32.0, resolution=128, wind_speed=3.0, wind_theta=0.5,
                    choppiness=0.4, wave_scale=0.4, fft_size=256)
    pond.params.tile_size_0 = 28.0     # small tiles → pond-scale ripples, not ocean swell
    pond.params.tile_size_1 = 7.0
    pond.params.tile_size_2 = 0.0
    pond.rotation.x = math.pi / 2      # Y-up ocean → Z-up world (waves rise in +Z)
    pond.position.set(lx, ly, wl-1)
    scene.add(pond)
    print(f"[pond] basin ({lx:.1f},{ly:.1f})  water_level={wl:.2f} m")
    return pond, wl


# ── policy observation (94-d: mirrors SpotStepsEnv) ───────────────────────────
def v2_obs(art, last_act, cmd, ahead, h_here):
    rs, rv = art.root_state(), art.root_velocity()
    R = _quat_to_R(rs[3:7]); Rt = R.T
    lin_b = Rt @ rv[0:3]; ang_b = Rt @ rv[3:6]; proj_g = Rt @ GRAV
    jp_isaac = art.joint_positions()[isaac_to_add]
    jv_isaac = art.joint_velocities()[isaac_to_add]
    qpos = jp_isaac - default_q
    return np.concatenate([lin_b, ang_b, proj_g, cmd, qpos, jv_isaac, last_act,
                           [float(rs[2]) - h_here], ahead]).astype(np.float32)


def _scanner_pts(scanner):
    """Convert the accumulated elevation map H → world-space 3-D points for the SLAM mapper."""
    valid = ~np.isnan(scanner.H)
    if not valid.any():
        return np.empty((0, 3), np.float32)
    ix = np.where(valid)[0].astype(np.float32)
    iy = np.where(valid)[1].astype(np.float32)
    return np.stack([ix * scanner.cell + scanner.x0,
                     iy * scanner.cell + scanner.y0,
                     scanner.H[valid]], axis=1).astype(np.float32)


# ── SLAM mapper ────────────────────────────────────────────────────────────────
class SlamMapper:
    """Accumulates scan hits → VoxelGrid → marching-cubes surface (background thread)."""

    VOXEL = 0.12
    CELL  = 0.14
    RAD   = 0.22   # tight: surface hugs scan hits rather than ballooning above them
    ISO   = 0.55

    def __init__(self, scene):
        self.scene   = scene
        self.grid    = tp.VoxelGrid(self.VOXEL, max_points_per_voxel=3, min_spacing=0.12)
        self._surf   = [None]
        self._pending= [None]
        self._busy   = [False]
        self._lock   = threading.Lock()

    def insert(self, pts):
        if pts.shape[0] > 0:
            self.grid.insert_array(pts)

    def trigger_rebuild(self):
        with self._lock:
            if self._busy[0] or self.grid.voxel_count < 30:
                return
            pts = self.grid.collect()
            self._busy[0] = True
        threading.Thread(target=self._worker, args=(pts,), daemon=True).start()

    def _worker(self, pts):
        try:
            field = tp.splat_points_to_field(pts, self.CELL, self.RAD, max_nodes=6_000_000)
            iso   = tp.marching_cubes(field, self.ISO)
            with self._lock:
                self._pending[0] = iso if not iso.empty else None
        finally:
            with self._lock:
                self._busy[0] = False

    def apply_pending(self):
        with self._lock:
            iso = self._pending[0]; self._pending[0] = None
        if iso is None:
            return
        geo = tp.iso_mesh_to_geometry(iso)
        mat = tp.MeshStandardMaterial()
        mat.color       = 0x55bbff
        mat.side        = tp.Side.Double
        # Wireframe routes the mesh through the renderer's raster overlay path: on Vulkan it
        # gets the kSnapWire flag → EXCLUDED from the path-tracer TLAS (so the depth sensor
        # never hits its own reconstruction → no self-contamination), and is drawn depth-tested
        # over the PT frame. On GL it simply renders as wireframe (still hidden during scans).
        mat.wireframe   = True
        mesh = tp.Mesh(geo, mat); mesh.frustum_culled = False
        if self._surf[0] is not None:
            self.scene.remove(self._surf[0])
        self._surf[0] = mesh
        self.scene.add(mesh)

    def clear(self):
        """Remove the displayed surface and discard all accumulated data."""
        with self._lock:
            self._pending[0] = None
        self.grid.clear()
        if self._surf[0] is not None:
            self.scene.remove(self._surf[0])
            self._surf[0] = None

    @property
    def busy(self):  return self._busy[0]
    @property
    def voxels(self): return self.grid.voxel_count


# ── path trail ────────────────────────────────────────────────────────────────
class PathTrail:
    MAX = 3000

    def __init__(self, scene):
        g = tp.BufferGeometry()
        g.set_attribute("position", np.zeros((self.MAX, 3), np.float32))
        g.set_draw_range(0, 0)
        mat = tp.LineBasicMaterial(); mat.color = 0xffffff
        self.line = tp.Line(g, mat); self.line.frustum_culled = False
        self._g   = g; self._pts = []; self._prev = None
        scene.add(self.line)

    def update(self, rs):
        p = (float(rs[0]), float(rs[1]), float(rs[2]) + 0.06)
        if self._prev and math.dist(p, self._prev) < 0.15:
            return
        self._pts.append(p)
        if len(self._pts) > self.MAX:
            self._pts.pop(0)
        arr = np.array(self._pts, np.float32)
        buf = np.zeros((self.MAX, 3), np.float32)
        buf[:len(arr)] = arr
        self._g.update_attribute("position", buf)
        self._g.set_draw_range(0, len(self._pts))
        self._prev = p

    def clear(self):
        self._pts.clear(); self._prev = None
        self._g.update_attribute("position", np.zeros((self.MAX, 3), np.float32))
        self._g.set_draw_range(0, 0)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--amplitude", type=float, default=AMPLITUDE)
    ap.add_argument("--shot",      metavar="PNG")
    args = ap.parse_args()
    assert tp.HAS_PHYSX, "needs a PhysX-enabled threepp build"
    headless = bool(args.shot)

    # ── assets ────────────────────────────────────────────────────────────────
    assets = fetch_assets()
    model_path = os.path.join(_HERE, "spot_steps.pt")
    ac, _norm, _meta = load_policy(model_path, device="cpu")
    ac.eval()
    print(f"[policy] {os.path.basename(model_path)}")

    # ── terrain ───────────────────────────────────────────────────────────────
    print("[terrain] generating ...")
    tparams = tp.TerrainParams()
    tp.apply_terrain_preset(1, tparams)          # Rolling Hills base shape
    tparams.resolution    = 192                  # 192² cells — fast physics trimesh
    tparams.world_size    = WORLD_SZ
    tparams.amplitude     = float(args.amplitude)# physical height in metres (height_at must agree)
    tparams.feature_scale = 45.0                 # preset is 430 m for 1600 m world; scale to 80 m
    tparams.octaves       = 7                    # more detail layers
    tparams.warp          = 0.40                 # organic domain warp
    tparams.falloff       = tp.TerrainFalloff.Off# preset's radial bowl looks wrong at 80 m
    tparams.erosion       = tp.ErosionType.Off   # thermal grooves are too fine at this scale
    tparams.ao_strength   = 8.0

    gen  = tp.TerrainGenerator(args.seed)
    gen.build_field(tparams)
    field    = gen.get_field()
    terr_tex = gen.bake_splat_texture(tparams)
    terr_geo = build_terrain_zup(field, WORLD_SZ, float(args.amplitude))
    # Sample terrain under the four feet + base to find the highest point in the
    # footprint — prevents spawning any foot inside the mesh on sloped ground.
    _FEET = ((0.30, 0.17), (0.30, -0.17), (-0.30, 0.17), (-0.30, -0.17), (0.0, 0.0))
    h0 = max(float(gen.height_at(dx, dy, tparams)) for dx, dy in _FEET)
    print(f"[terrain] done  h_footprint={h0:.3f} m")

    # ── physics ───────────────────────────────────────────────────────────────
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.002, max_substeps=20)
    base  = tp.Mesh(tp.BoxGeometry(WORLD_SZ + 10, WORLD_SZ + 10, 0.4), tp.MeshStandardMaterial())
    base.position.set(0, 0, -0.2)
    world.add_static(base)
    terr_phys = tp.Mesh(terr_geo, tp.MeshStandardMaterial())
    world.add_static_trimesh(terr_phys)

    art, meshes = build_spot(world, assets)

    def settle(n=80):
        for _ in range(n):
            art.set_drive_targets(default_q[add_to_isaac].astype(np.float32))
            world.step(0.02)

    art.reset(tp.Vector3(0.0, 0.0, Z0 + h0 + 0.02))
    settle()
    print("[spot] standing")

    # ── canvas + renderer ─────────────────────────────────────────────────────
    canvas = tp.Canvas("threepp · Spot SLAM", width=1200, height=720,
                       antialiasing=4, headless=headless)
    rend = tp.VulkanRenderer(canvas)
    rend.shadow_map_enabled       = True
    rend.tone_mapping             = tp.ToneMapping.ACESFilmic
    rend.tone_mapping_exposure    = 1.1

    # ── scene ─────────────────────────────────────────────────────────────────
    scene = tp.Scene()
    # HDR environment: equirect Radiance map drives image-based lighting (the Vulkan
    # PT importance-samples it) and serves as the sky backdrop. Cached on first run.
    try:
        env = tp.RGBELoader().load(fetch_hdr(HDR_URL, "noon_grass_2k.hdr"))
        # The renderer samples env maps Y-up (latitude = asin(dir.y)); this scene is
        # Z-up, so pitch the HDRI -90° about X to stand the sky overhead.
        env = tp.rotate_equirect(env, -90.0, 0.0, 0.0)
        scene.environment = env
        scene.background  = env
        hemi_intensity = 0.25          # HDR provides the ambient fill; keep only a touch
    except Exception as e:
        print(f"[hdri] unavailable ({e}); falling back to flat sky")
        scene.background = tp.Background(0x8ab4d4)
        hemi_intensity = 0.9
    scene.set_fog(0x8ab4d4, 25.0, 80.0)
    scene.add(tp.HemisphereLight(0xd0e8ff, 0x3a4820, hemi_intensity))
    sun = tp.DirectionalLight(0xfff8e0, 2.8)
    sun.position.set(15, -10, 20)
    sun.cast_shadow = True
    sun.set_shadow_frustum(-18, 18, 18, -18)  # fog hides >25 m; no need for full world frustum
    sun.set_shadow_bias(-0.0005)
    scene.add(sun)

    terr_vis = tp.Mesh(terr_geo, tp.MeshStandardMaterial())
    terr_vis.material.map       = terr_tex
    terr_vis.material.roughness = 0.9
    terr_vis.receive_shadow = True
    scene.add(terr_vis)

    for m in meshes:
        m.cast_shadow = True
        scene.add(m)

    # Pond first, so vegetation can avoid standing in open water.
    print("[pond] searching for a basin ...")
    pond_obj, pond_wl = add_pond(scene, gen, tparams)
    pond = (pond_obj.position.x, pond_obj.position.y, pond_wl) if pond_obj else None

    print("[trees] scattering ...")
    # Trees + stones get static colliders (Spot bumps trunks/boulders); bushes stay
    # soft (walk-through). Keep the proxy meshes alive for the whole run.
    phys_proxies = []
    phys_proxies += scatter_trees(scene, gen, tparams, seed=args.seed, world=world, pond=pond)
    scatter_bushes(scene, gen, tparams, seed=args.seed, pond=pond)
    phys_proxies += scatter_stones(scene, gen, tparams, seed=args.seed, world=world, pond=pond)

    print("[grass] building ...")
    grass_geo, n_grass = build_grass_field(gen, tparams, GRASS_BLADES, GRASS_RADIUS,
                                           args.seed, clear_r=1.0, pond=pond)
    grass_mat = tp.MeshStandardMaterial()
    grass_mat.vertex_colors = True
    grass_mat.roughness     = 0.95
    grass_mat.side          = tp.Side.Double
    grass = tp.GrassMesh(grass_geo, grass_mat)
    grass.rotation.x   = math.pi / 2      # Y-up blades → Z-up world (like the trees)
    grass.wind_dir     = tp.Vector2(0.8, 0.6)
    grass.wind_strength = 0.15
    grass.frustum_culled = False
    scene.add(grass)
    print(f"[grass] {n_grass} blades")

    # ── camera (Z-up chase cam) ───────────────────────────────────────────────
    w, h = canvas.size()
    camera = tp.PerspectiveCamera(50, w / h, 0.05, 200)
    camera.up.set(0, 0, 1)
    spawn_z = Z0 + h0
    camera.position.set(-3.0, -3.0, spawn_z + 1.8)
    camera.look_at(0.0, 0.0, spawn_z + 0.3)
    BACK, HEIGHT, LAG = 3.5, 1.8, 0.08

    # ── SLAM objects ──────────────────────────────────────────────────────────
    half = WORLD_SZ / 2.0
    # Vulkan's depth sensor traces the TLAS from the last render(); GL re-renders internally.
    is_vulkan = type(rend).__name__ == "VulkanRenderer"
    scanner = ForwardDepthScanner(rend, scene, meshes,
                                  bounds=(-half, half, -half, half),
                                  cell=0.15, far=SENSOR_FAR,
                                  mount_fwd=0.95, mount_up=-0.10,
                                  pitch_deg=40.0, fov_y=90.0)
    if is_vulkan:
        rend.render(scene, camera)   # build an initial TLAS so prewarm's scans have geometry
    scanner.prewarm(art.root_state())
    slam    = SlamMapper(scene)
    trail   = PathTrail(scene)
    last_act    = np.zeros(12, np.float32)
    ahead_cache = [np.zeros(45, np.float32)]   # last sensor reading; reused on skipped frames
    h_here_cache= [h0]

    # ── state ─────────────────────────────────────────────────────────────────
    fc        = [0]
    auto_fwd  = [False]
    hdg_lock  = [None]
    r_held    = [False]
    space_held= [False]

    def reset():
        rh = max(float(gen.height_at(dx, dy, tparams)) for dx, dy in _FEET)
        art.reset(tp.Vector3(0.0, 0.0, Z0 + rh + 0.02))
        last_act[:] = 0.0
        hdg_lock[0] = None
        settle(40)
        scanner.clear_map(); scanner.prewarm(art.root_state())
        slam.clear()
        trail.clear()

    def _save_frame(path):
        """Renderer-agnostic screenshot: GL takes only path; Vulkan takes scene+camera+path."""
        try:
            rend.save_frame(path)
        except TypeError:
            rend.save_frame(scene, camera, path)

    # ── headless ──────────────────────────────────────────────────────────────
    if headless:
        cmd = np.array([1.0, 0.0, 0.0], np.float32)
        # Vulkan: scan() uses the TLAS from the last render(), so render before each scan.
        # GL: scan() re-renders internally anyway; the extra render() is a cheap no-op for screenshots.
        rend.render(scene, camera)
        for i in range(150):
            rs = art.root_state()
            if i % SCAN_EVERY == 0:
                rend.render(scene, camera)   # refresh TLAS for Vulkan
                ahead, h_here = scanner.scan(rs)
            obs = v2_obs(art, last_act, cmd, ahead, h_here)
            with torch.no_grad():
                a = ac.act_mean(torch.from_numpy(obs)[None])[0].numpy()
            last_act[:] = a
            art.set_drive_targets((default_q + ACTION_SCALE * a)[add_to_isaac].astype(np.float32))
            world.step(0.02)
        rs = art.root_state()
        trail.line.visible = False
        for _ in range(20):
            rend.render(scene, camera)
            ahead, h_here = scanner.scan(rs)
        trail.line.visible = True
        slam.insert(_scanner_pts(scanner))
        slam.trigger_rebuild()
        time.sleep(1.2)
        slam.apply_pending()
        p   = np.array(rs[:3], float)
        fwd = _quat_to_R(rs[3:7])[:, 0]; fwd[2] = 0
        fwd /= max(np.linalg.norm(fwd), 1e-6)
        eye = p - fwd * 4.5 + np.array([0.0, 0.0, 2.2])
        camera.position.set(float(eye[0]), float(eye[1]), float(eye[2]))
        camera.look_at(float(p[0]), float(p[1]), float(p[2]) + 0.3)
        rend.render(scene, camera)
        _save_frame(args.shot)
        print(f"saved {args.shot}")
        return

    # ── interactive ───────────────────────────────────────────────────────────
    ui = tp.ImguiContext(canvas, rend) if tp.HAS_IMGUI else None

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        rend.set_size(w, h)
    canvas.on_window_resize(on_resize)

    def down(*keys):
        return any(canvas.is_key_down(k) for k in keys)

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12)
        tp.imgui.set_next_window_size(300, 0)
        tp.imgui.begin("Spot SLAM")

        rs = art.root_state()
        tp.imgui.text(f"pos  x={rs[0]:+.1f}  y={rs[1]:+.1f}  z={rs[2]:.2f} m")
        _, auto_fwd[0] = tp.imgui.checkbox("auto-forward (SPACE)", auto_fwd[0])
        tp.imgui.separator()
        tp.imgui.text("Depth camera")
        chg, v = tp.imgui.slider_float("range noise (m)", scanner.sensor.range_noise, 0.0, 0.10)
        if chg: scanner.sensor.range_noise = v
        if scanner.cloud is not None:
            chg, v = tp.imgui.slider_float("point size", scanner.cloud.material.size, 0.02, 0.12)
            if chg: scanner.cloud.material.size = v
        _, scanner.show_cloud = tp.imgui.checkbox("show cloud", scanner.show_cloud)
        _, scanner.show_grid  = tp.imgui.checkbox("show scan grid", scanner.show_grid)
        tp.imgui.separator()
        tp.imgui.text(f"SLAM  voxels: {slam.voxels}  {'[rebuilding]' if slam.busy else ''}")
        if tp.imgui.button("rebuild surface now") and not slam.busy:
            slam.trigger_rebuild()
        if tp.imgui.button("reset (R)"):
            reset()
        tp.imgui.separator()
        tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps   |   WASD + QE")
        tp.imgui.end()

    def frame():
        fc[0] += 1

        # SPACE toggles auto-forward
        if down("SPACE"):
            if not space_held[0]: auto_fwd[0] = not auto_fwd[0]
            space_held[0] = True
        else:
            space_held[0] = False

        # keyboard command — WASD drive, QE turn (numpad kept as an alternate)
        vx = (1.5 if down("W", "KP8") else 0.0) - (1.0 if down("S", "KP2") else 0.0)
        vy = (1.0 if down("A", "KP4") else 0.0) - (1.0 if down("D", "KP6") else 0.0)
        wz_key = (1.5 if down("Q", "KP7") else 0.0) - (1.5 if down("E", "KP9") else 0.0)
        if auto_fwd[0] and vx == 0.0 and vy == 0.0 and wz_key == 0.0:
            vx = 1.0

        # heading hold
        rs  = art.root_state()
        R   = _quat_to_R(rs[3:7])
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
        if wz_key != 0.0:
            wz = wz_key; hdg_lock[0] = yaw
        else:
            if hdg_lock[0] is None: hdg_lock[0] = yaw
            err = (yaw - hdg_lock[0] + math.pi) % (2 * math.pi) - math.pi
            wz  = float(np.clip(-2.0 * err, -1.0, 1.0))

        # obs → policy → step
        cmd   = np.array([vx, vy, wz], np.float32)
        obs   = v2_obs(art, last_act, cmd, ahead_cache[0], h_here_cache[0])
        with torch.no_grad():
            a = ac.act_mean(torch.from_numpy(obs)[None])[0].numpy()
        last_act[:] = a
        art.set_drive_targets((default_q + ACTION_SCALE * a)[add_to_isaac].astype(np.float32))
        world.step(0.02)
        rs = art.root_state()
        if fc[0] % MC_FRAMES == 0:
            slam.trigger_rebuild()
        slam.apply_pending()
        trail.update(rs)

        # R = reset
        if down("R"):
            if not r_held[0]: reset(); print("[reset]")
            r_held[0] = True
        else:
            r_held[0] = False

        # chase cam
        p   = np.array(rs[:3], float)
        fwd = R[:, 0].copy(); fwd[2] = 0.0
        nrm = np.linalg.norm(fwd); fwd = fwd / nrm if nrm > 1e-6 else np.array([1.0, 0.0, 0.0])
        des = p - fwd * BACK + np.array([0.0, 0.0, HEIGHT])
        camera.position.lerp(tp.Vector3(float(des[0]), float(des[1]), float(des[2])), LAG)
        camera.look_at(float(p[0] + fwd[0] * 0.5), float(p[1] + fwd[1] * 0.5), float(p[2]) + 0.2)

        if fc[0] % SCAN_EVERY == 0:
            if is_vulkan:
                # The SLAM surface is wireframe → excluded from the PT TLAS; the trail (Line)
                # and point cloud (Points) are never ray-traced either. scan() traces the TLAS
                # from the previous frame's render() → clean, no hiding/extra render needed.
                ahead_cache[0], h_here_cache[0] = scanner.scan(rs)
            else:
                # GL: the sensor re-renders the scene, so hide the SLAM surface + trail (the
                # scanner already hides the robot + its own cloud/grid).
                _extra = [o for o in (slam._surf[0], trail.line) if o is not None]
                for o in _extra: o.visible = False
                ahead_cache[0], h_here_cache[0] = scanner.scan(rs)
                for o in _extra: o.visible = True
            slam.insert(_scanner_pts(scanner))

        grass.time = time.perf_counter() - t0  # advance GPU wind (Vulkan)
        rend.render(scene, camera)             # single render per frame — no flicker
        if ui:
            ui.render(draw_ui)

    print(__doc__)
    t0 = time.perf_counter()
    canvas.animate(frame)


if __name__ == "__main__":
    main()
