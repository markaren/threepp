"""Spot SLAM — procedural terrain + trees + depth camera + live map reconstruction.

Spot walks on TerrainGenerator terrain while a body-mounted forward depth camera
accumulates a 3-D point cloud. Every ~3 s a background thread runs marching cubes
to reconstruct the growing SLAM surface (semi-transparent blue) over the ground truth.

    python spot_slam.py
    python spot_slam.py --seed 7 --amplitude 0.20
    python spot_slam.py --shot out.png
    python spot_slam.py --film recovery.mp4       # headless scripted fall-recovery film (--film-dry checks it first)
    python spot_slam.py --model idun_runs/spot_unified2/unifiedB_s0/spot_steps.pt  # one policy walks and jumps: J, while walking

Controls: W/S = fwd/back  A/D = strafe  Q/E = turn  |  R = reset  |  mouse = orbit/zoom
The interactive window opens FULLSCREEN; pass --windowed for a 1200x720 window instead.
"""
import argparse, math, os, sys, threading, time
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))   # scratch_clock / scratch_env

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import threepp as tp
from threepp.rl import load_policy
from threepp.utils import fetch_file
from spot_deploy import (build_spot, fetch_assets,
                         _quat_to_R, _quat_from_R,
                         default_q, isaac_to_add, add_to_isaac, ACTION_SCALE, Z0, LIM, MASS)
from spot_depth_scan import ForwardDepthScanner
from spot_terrain_env import VX_HI, VY_HI, WZ_HI
from scratch_env import STIFF_GAINS
from scratch_clock import GAIT_PERIOD
from _common import v2_obs
from spot_feet import cpu_foot_tips

# Fall recovery (--recovery-model, spot_recovery_env.py): the walking policy hands over when the body tips past
# RECOVER_UP and takes control back once the robot has stood (up_z > 0.9, base 0.40 m above the ground) and been still
# for RECOVER_SETTLE control ticks.
# Pushes are sized as a velocity change of the whole robot (impulse = total mass x dv), in a random sideways direction
# jittered PUSH_JITTER_DEG. F = SHOVE: SHOVE_DV through the base's centre of mass; training shoves top out at 2 m/s, so the
# walking policy should usually ride it out. G = TRIP: TRIP_DV at TRIP_HEIGHT above the base's centre of mass (the top of
# the body), the smallest push that tips a standing Spot over reliably (--trip-probe); --knock-at uses the trip. The old F
# was 150 N·s 0.3 m above the base, ~5 m/s on 28 kg: "it's like a tornado" (Lars, 2026-09-14).
RECOVER_UP = 0.5
RECOVER_SETTLE = 25          # 0.5 s at 50 Hz
SHOVE_DV = 2.0               # m/s
# Measured 2026-09-14 with --trip-probe (shipped walking policy standing, the plant --recovery-model builds, 10 trials per
# step, +-10 deg jitter), at 0.095 m above the base's centre of mass: flat ground (--amplitude 0) tipped past up_z 0.5 in
# 0/10 at 1 m/s, 6/10 at 1.5, 10/10 at 2; the default spawn terrain in 0/10 at 1, 1.5 and 2 m/s (min up_z 0.62-0.78) and
# 10/10 at 2.5. So 2.5 m/s (70 N·s on 28 kg) is the smallest reliable trip on both, rounded up to 2.75. The 2 m/s shove
# through the centre of mass tipped 0/10 on either (min up_z 0.78-0.92).
TRIP_DV, TRIP_HEIGHT = 2.75, 0.095  # m/s, m
PUSH_JITTER_DEG = 10.0
TRIP_LADDER = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)   # --trip-probe dv steps, m/s


# SPOT_SCANSTATS=1 prints the 45-cell scan's spread every 60 frames — the check that the
# height scan the policy consumes is actually tracking terrain, independent of how it is drawn.
_SCANSTATS = bool(os.environ.get("SPOT_SCANSTATS"))
_RESET_AT  = int(os.environ.get("SPOT_BENCH_RESET_AT", "0"))   # bench-only reset smoke test

# ── constants ──────────────────────────────────────────────────────────────────
WORLD_SZ   = 80.0
AMPLITUDE  = 10.8
TREE_COUNT = 60
CLEAR_R    = 7.0     # no trees within this radius of spawn

# Species mix for scatter_trees: (preset, name, weight, needs_frond, translucency, translucency_color)
# preset indices match apply_tree_preset: 0=Oak, 1=Pine/Spruce (whorl conifer), 2=Birch, 3=Willow.
TREE_SPECIES = [
    (0, "oak",    0.30, False, 0.40, (0.55, 0.85, 0.30)),
    (1, "spruce", 0.35, True,  0.45, (0.50, 0.80, 0.28)),
    (2, "birch",  0.20, False, 0.42, (0.60, 0.85, 0.35)),
    (3, "willow", 0.15, False, 0.40, (0.55, 0.85, 0.32)),
]
VARIANTS_PER_SPECIES = 5   # a handful of distinct seeds per species (like fjord/forest_demo);
                           # placements reuse these prototypes so the pool stays cheap
WATER_BIAS_R = 14.0        # within this radius of the pond centre, favour willows
SENSOR_W   = 128   # 128x96 = 12.3k beams: ~12.5 cm footprint at far range, still
SENSOR_H   = 96    # denser than the 15 cm elevation-map cell — was 160x120
SENSOR_FAR = 8.0
SCAN_EVERY = 3       # depth scan every N frames; result cached for policy (~17 Hz)
MC_FRAMES  = 90      # trigger SLAM rebuild every N rendered frames
GRASS_BLADES = 12000 # merged GrassMesh blade count (GPU-wind on Vulkan); tune for FPS
GRASS_RADIUS = 42.0  # grass disk radius around spawn (fog hides >25 m anyway)
HDR_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/2k/noon_grass_2k.hdr"

# ── film (--film OUT.mp4) ─────────────────────────────────────────────────────
# One continuous take, one frame per 0.02 s control tick (real time at 50 fps). The script is in ticks; the second trip
# and the end are scheduled from the recovery hand-backs, so the take is as long as the recoveries make it.
FILM_W, FILM_H, FILM_FPS = 1920, 1080, 50
FILM_VX       = 1.0      # m/s walking command
FILM_STAND0   = 100      # ticks standing before the walk
FILM_SHOVE    = 300      # tick of the F shove
FILM_TRIP1    = 500      # tick of the first G trip
FILM_HB_STAND = 25       # ticks standing after a hand-back before walking on
FILM_TRIP2    = 225      # ticks after the first hand-back: the second trip, from the other side
FILM_END      = 325      # ticks after the second hand-back: the end of the take
FILM_TIMEOUT  = 500      # ticks after a trip without a hand-back: give up on it
FILM_FOV      = 40.0
FILM_AZ       = (125.0, 235.0)   # camera azimuth from the course heading, deg: left-rear quarter, then right-rear
FILM_DIST     = (3.6, 4.8)       # m to the target: walking, around a fall (6 m left the fallen robot too small)
FILM_ELEV     = (15.0, 20.0)     # deg above the target: walking, around a fall
FILM_EYE_CLEAR = 1.0             # m above the terrain under the eye, at least

# ── bench (--bench N) ─────────────────────────────────────────────────────────
BENCH_WARMUP = 120     # frames dropped from the statistics (pipeline/TAA/allocator warmup)
BENCH_BUDGET = 16.7    # ms — a frame above this missed 60 Hz
BENCH_PHASES = ("policy", "physics", "slam", "camera", "scan", "render", "ui")



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
    return math.hypot(px - cx, py - cy) < 12.0 and hz < wl + 0.2


# ── trees ──────────────────────────────────────────────────────────────────────
def build_tree_variants(seed):
    """Pre-build a small pool of fully-realized tree prototypes (a few seeds per
    species from TREE_SPECIES, like forest_demo/vulkan_fjord's variant pooling):
    geometry + bark/leaf textures are shared across every placement of a given
    prototype — only position/scale/yaw vary per instance. Bounds the number of
    unique meshes/textures while still giving each species its own silhouette,
    bark, needle/leaf cutout and foliage translucency."""
    rng  = np.random.default_rng(seed + 9001)
    tgen = tp.TreeGenerator(0)        # reseeded per variant
    variants = []
    for preset, name, weight, needs_frond, translucency, t_color in TREE_SPECIES:
        for _ in range(VARIANTS_PER_SPECIES):
            vseed = int(rng.integers(0, 1_000_000))
            tpar = tp.TreeParams()
            tpar.seed = vseed
            tp.apply_tree_preset(preset, tpar)
            if preset != 1:            # Colonise species (whorl conifer is O(whorls), not iterations)
                tpar.max_iterations = 140
            tgen.reseed(vseed)
            tgen.build_skeleton(tpar)
            trunk_geo = tgen.make_trunk_geometry(tpar)
            leaf_geo  = tgen.make_leaf_geometry(tpar)

            bark_alb, bark_nrm = tp.make_bark_textures(192, vseed, tpar.bark_color)
            trunk_mat = tp.MeshStandardMaterial()
            trunk_mat.map          = bark_alb
            trunk_mat.normal_map   = bark_nrm
            trunk_mat.normal_scale = tp.Vector2(0.7, 0.7)
            trunk_mat.roughness    = 0.88

            # Conifer fronds want the elongated needle cutout; broadleaf species the
            # round leaf-cluster atlas (mirrors vulkan_fjord.cpp's makeLeafTex).
            leaf_tex = (tp.make_needle_frond_texture(224, vseed, tpar.leaf_color) if needs_frond
                       else tp.make_leaf_texture(224, vseed, tpar.leaf_color))
            leaf_mat = tp.MeshStandardMaterial()
            leaf_mat.map                = leaf_tex
            leaf_mat.alpha_test         = 0.5   # discard the transparent gaps → leafy silhouette
            leaf_mat.roughness          = 0.88
            leaf_mat.side               = tp.Side.Double
            leaf_mat.vertex_colors      = True  # per-leaf tint variation baked into the geometry
            # Foliage translucency: backlit canopy glow (Vulkan deferred only; no-op on GL).
            leaf_mat.translucency       = translucency
            leaf_mat.translucency_color = tp.Color(*t_color)

            variants.append(dict(name=name, weight=weight, is_willow=(name == "willow"),
                                 trunk_geo=trunk_geo, leaf_geo=leaf_geo,
                                 trunk_mat=trunk_mat, leaf_mat=leaf_mat,
                                 trunk_radius=tpar.trunk_radius,
                                 # rough crown extent, for the film camera's sightline test only
                                 crown_r=max(float(tpar.crown_radius_x), float(tpar.crown_radius_z), 1.0),
                                 trunk_h=float(tpar.trunk_height),
                                 height=float(tpar.trunk_height) + float(tpar.crown_height)))
    print(f"[trees] built {len(variants)} variants ({len(TREE_SPECIES)} species x {VARIANTS_PER_SPECIES})")
    return variants


def scatter_trees(scene, gen, params, variants, n=TREE_COUNT, seed=0, world=None, pond=None, placements=None):
    """Scatter trees drawn from a pre-built variant pool (build_tree_variants).
    When `world` is given, add a static box collider per trunk (a "tree stub")
    so Spot bumps into the trunks; the leafy canopy stays non-colliding.
    `placements`, if a list, gets (x, y, ground z, crown radius, trunk height, height) per tree.
    Returns the list of collider proxy meshes (keep them alive)."""
    rng  = np.random.default_rng(seed)
    half = params.world_size / 2.0 - 4.0
    proxies = []

    base_w      = np.array([v["weight"] for v in variants], np.float64)
    willow_mask = np.array([v["is_willow"] for v in variants])
    pond_cx, pond_cy = (pond[0], pond[1]) if pond is not None else (None, None)

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

        w = base_w
        if pond_cx is not None and math.hypot(px - pond_cx, py - pond_cy) < WATER_BIAS_R:
            w = np.where(willow_mask, base_w * 6.0, base_w)   # willows favour the water's edge
        v = variants[int(rng.choice(len(variants), p=w / w.sum()))]

        scale = float(rng.uniform(0.75, 1.3))
        yaw   = float(rng.uniform(0, 2 * math.pi))
        for geo, mat in ((v["trunk_geo"], v["trunk_mat"]), (v["leaf_geo"], v["leaf_mat"])):
            m = tp.Mesh(geo, mat)
            m.rotation.y = yaw           # spin around the (still Y-up) trunk axis first...
            m.rotation.x = math.pi / 2   # ...then stand it up: Y-up tree → Z-up world
            m.position.set(px, py, hz)
            m.scale.set(scale, scale, scale)
            m.cast_shadow = True
            scene.add(m)
        if world is not None:
            # Tall thin box collider on the lower trunk (covers Spot's height);
            # the canopy above it is leaves → no collision. Width tracks the
            # species' actual trunk radius so spruces/oaks don't share one size.
            tw   = 0.35 * scale * max(v["trunk_radius"] / 0.15, 0.6)
            stub = tp.Mesh(tp.BoxGeometry(tw, tw, 2.5), v["trunk_mat"])
            stub.position.set(px, py, hz + 1.25)
            world.add_static(stub)
            proxies.append(stub)
        if placements is not None:
            placements.append((px, py, hz, v["crown_r"] * scale, v["trunk_h"] * scale, v["height"] * scale))
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
    # Every phi-dependent term is gated by sinT so it vanishes at the poles
    # (theta = 0, pi). Ungated cos(3phi)/cos(5phi) give each pole vertex a
    # DIFFERENT radius, smearing the single pole point into a fan of thin sliver
    # triangles whose sub-pixel coverage toggles with the TAA jitter and flickers
    # uniformly every frame. Gated, the pole vertices coincide -> zero-area
    # triangles that rasterize to nothing, leaving a clean pole fan.
    disp = (1.0 + sinT * (0.30 * np.sin(2 * P + p1) + 0.24 * np.cos(3 * P + p2)
                          + 0.14 * np.cos(5 * P + 4 * T + p1))
            + 0.22 * np.sin(3 * T + p3))
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


def scatter_stones(scene, gen, params, n=35, seed=0, world=None, pond=None, placements=None):
    """Scatter boulders. When `world` is given, add a static sphere collider per
    stone so Spot bumps into them. `placements`, if a list, gets (x, y, ground z, radius).
    Returns the collider proxy meshes."""
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
        if placements is not None:
            placements.append((px, py, hz, s * 0.8))
        placed += 1
    print(f"[stones] placed {placed}/{n}")
    return proxies


# ── bushes (shrub-variant trees, like forest_demo) ────────────────────────────────
def scatter_bushes(scene, gen, params, n=50, seed=0, pond=None, placements=None):
    """Short trunk + wide low crown shrubs. Each of a handful of prototypes gets
    its own seed, crown shape/size, leaf colour and bark texture (mirrors the
    tree variant pool) so the understory doesn't read as one shrub copy-pasted.
    `placements`, if a list, gets (x, y, ground z, crown radius, height) per bush."""
    rng  = np.random.default_rng(seed + 3)
    half = params.world_size / 2.0 - 4.0

    N_VARIANTS = 6
    tgen = tp.TreeGenerator(0)
    variants = []
    for _ in range(N_VARIANTS):
        s = int(rng.integers(0, 1_000_000))
        tpar = tp.TreeParams()
        tpar.seed = s
        tpar.trunk_height = 0.35 + float(rng.uniform(0, 0.30))
        tpar.trunk_radius = 0.05 + float(rng.uniform(0, 0.03))
        tpar.crown_shape  = tp.CrownShape.Hemisphere if rng.uniform() < 0.6 else tp.CrownShape.Sphere
        r = 0.85 + float(rng.uniform(0, 0.65))
        tpar.crown_radius_x = r; tpar.crown_radius_z = r
        tpar.crown_height   = r * (1.05 if tpar.crown_shape == tp.CrownShape.Hemisphere else 1.35)
        tpar.attractor_count = 220 + int(rng.integers(0, 100))
        tpar.influence_distance = 2.0; tpar.kill_distance = 0.45
        tpar.segment_length = 0.22; tpar.max_iterations = 130; tpar.randomness = 0.12
        tpar.radial_segments = 5
        tpar.leaf_style = tp.LeafStyle.CrossQuad
        tpar.leaf_size    = 0.32 + float(rng.uniform(0, 0.22))
        tpar.leaf_density = 0.82 + float(rng.uniform(0, 0.16))
        tpar.leaves_per_cluster = int(rng.integers(4, 7))
        tpar.leaf_spread   = 0.28 + float(rng.uniform(0, 0.22))
        hue = float(rng.uniform(-0.03, 0.06))       # per-bush leaf hue jitter (olive → fresh green)
        tpar.leaf_color = [0.13 + hue * 0.6, 0.33 + hue, 0.11 + hue * 0.3]
        tpar.bark_color = [0.30, 0.22, 0.12]
        tgen.reseed(s); tgen.build_skeleton(tpar)
        trunk_geo = tgen.make_trunk_geometry(tpar)
        leaf_geo  = tgen.make_leaf_geometry(tpar)

        bark_alb, _ = tp.make_bark_textures(64, s, tpar.bark_color)
        trunk_mat = tp.MeshStandardMaterial(); trunk_mat.map = bark_alb; trunk_mat.roughness = 0.9

        leaf_mat = tp.MeshStandardMaterial()
        leaf_mat.map                = tp.make_leaf_texture(224, s, tpar.leaf_color)
        leaf_mat.alpha_test         = 0.5
        leaf_mat.roughness          = 0.9
        leaf_mat.side               = tp.Side.Double
        leaf_mat.vertex_colors      = True
        leaf_mat.translucency       = 0.35   # backlit canopy glow (Vulkan deferred; no-op on GL)
        leaf_mat.translucency_color = tp.Color(0.55, 0.85, 0.30)
        variants.append((trunk_geo, leaf_geo, trunk_mat, leaf_mat,
                         r, float(tpar.trunk_height) + float(tpar.crown_height)))

    placed = attempts = 0
    while placed < n and attempts < n * 15:
        attempts += 1
        px, py = float(rng.uniform(-half, half)), float(rng.uniform(-half, half))
        if math.hypot(px, py) < 3.0:
            continue
        hz = float(gen.height_at(px, py, params))
        if _in_pond(px, py, hz, pond):       # keep bushes out of open water
            continue
        trunk_geo, leaf_geo, trunk_mat, leaf_mat, crown_r, top = variants[int(rng.integers(0, len(variants)))]
        s   = 0.7 + float(rng.uniform()) * 0.7
        if placements is not None:
            placements.append((px, py, hz, crown_r * s, top * s))
        yaw = float(rng.uniform(0, 2 * math.pi))
        for geo, mat in ((trunk_geo, trunk_mat), (leaf_geo, leaf_mat)):
            m = tp.Mesh(geo, mat)
            m.rotation.y = yaw
            m.rotation.x = math.pi / 2     # Y-up → Z-up (like the trees)
            m.position.set(px, py, hz)
            m.scale.set(s, s, s)
            m.cast_shadow = True
            scene.add(m)
        placed += 1
    print(f"[bushes] placed {placed}/{n} ({len(variants)} variants)")


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
    # Ocean is scale-aware now: size=32 alone derives pond-scale cascade tiles,
    # clear-ish freshwater absorption (the bottom is genuinely visible through
    # the surface via the shallow-water shading), and near-zero natural
    # whitecap foam — the manual tile_size_0/1 overrides that used to fake
    # this are gone. tile_size_2=0 stays: the deferred fine-chop normal is
    # sampled in world XZ (Y-up assumption), so on this ROTATED (Z-up) ocean
    # the finest cascade would render as 1-D stripes.
    pond = tp.Ocean(size=32.0, resolution=128, wind_speed=3.0, wind_theta=0.5,
                    choppiness=0.4, wave_scale=0.4, tile_size_2=0.0, fft_size=256)
    # Murkier than the freshwater default: this pond is only ~0.4 m deep, and
    # at that depth clear water reads as wet sand. A short attenuation distance
    # veils the bottom into a proper green-brown; thickness scales the veil's
    # own brightness (see MeshPhysicalMaterial.thickness).
    mat = pond.material
    mat.attenuation_color = 0x245238   # green-brown murk
    mat.attenuation_distance = 1.2     # ~a metre of visibility
    mat.thickness = 0.6
    pond.rotation.x = math.pi / 2      # Y-up ocean → Z-up world (waves rise in +Z)
    pond.position.set(lx, ly, wl-1)
    scene.add(pond)
    print(f"[pond] basin ({lx:.1f},{ly:.1f})  water_level={wl:.2f} m")
    return pond, wl


# ── SLAM mapper ────────────────────────────────────────────────────────────────
class SlamMapper:
    """Accumulates scan hits → VoxelGrid → marching-cubes surface (background thread)."""

    VOXEL = 0.12
    CELL  = 0.14
    RAD   = 0.22   # tight: surface hugs scan hits rather than ballooning above them
    ISO   = 0.55
    MAX_VERTS = 600_000   # preallocated surface capacity (200k triangles)

    def __init__(self, scene):
        self.scene   = scene
        self.grid    = tp.VoxelGrid(self.VOXEL, max_points_per_voxel=3, min_spacing=0.12)
        self._pending= [None]
        self._busy   = [False]
        self._visible= [True]
        self._sent   = None      # scanner cells already pushed into the grid
        self._lock   = threading.Lock()
        # Bumped by clear(). A rebuild carries the generation it started in and
        # refuses to publish into a newer one — otherwise a worker that was
        # already marching cubes when the user hit R finishes a second later and
        # hands apply_pending() the PRE-reset surface, which then reappears on
        # screen over the freshly cleared map. Blanking the draw range in
        # clear() cannot prevent that: the stale result arrives afterwards.
        self._gen    = 0

        # ONE mesh + geometry for the whole session, preallocated at capacity:
        # a rebuild overwrites the leading rows and moves the draw range.
        # Building a fresh Mesh per rebuild instead cost ~11 ms inside the NEXT
        # render() — a new multi-MB vertex buffer plus the scene add/remove
        # forcing a full entry re-expansion — i.e. a visible hitch every
        # MC_FRAMES. In-place updates keep the entry (and its TAA history) put.
        g = tp.BufferGeometry()
        g.set_attribute("position", np.zeros((self.MAX_VERTS, 3), np.float32))
        g.set_attribute("normal",   np.zeros((self.MAX_VERTS, 3), np.float32))
        g.set_draw_range(0, 0)
        self._draw_n = 0          # mirrors the geometry's draw count (not readable back)
        mat = tp.MeshStandardMaterial()
        mat.color       = 0x55bbff
        mat.side        = tp.Side.Double
        # Wireframe routes the mesh through the renderer's raster overlay path: on Vulkan it
        # gets the kSnapWire flag → EXCLUDED from the path-tracer TLAS (so the depth sensor
        # never hits its own reconstruction → no self-contamination), and is drawn depth-tested
        # over the PT frame. On GL it simply renders as wireframe (still hidden during scans).
        mat.wireframe   = True
        self._geo  = g
        self._surf = tp.Mesh(g, mat)
        self._surf.frustum_culled = False
        self._apply_visibility()   # nothing built yet -> hidden
        # Added to the scene on the FIRST surface, not here: a 600k-vertex
        # all-zero geometry parked at the origin is still a scene entry, and the
        # depth sensor traces the scene rather than the draw range.
        self._added = False

    def insert_scanner(self, scanner):
        """Push the cells the scanner has newly filled in since the last call.

        The scanner's H is the *accumulated* elevation map — 535x535 cells over the
        80 m world here — so converting all of it every scan means walking a third
        of a million cells and re-offering tens of thousands of points the voxel
        grid then rejects. Only a first-time cell can add a voxel: later EMA nudges
        move a cell by centimetres, far inside VOXEL / min_spacing. So track which
        cells have been sent and hand over just the delta, which is bounded by the
        depth camera's footprint instead of growing with everything explored.
        """
        valid = ~np.isnan(scanner.H)
        if self._sent is None:
            self._sent = np.zeros_like(valid)
        new = valid & ~self._sent
        if not new.any():
            return
        self._sent |= new
        ix, iy = np.nonzero(new)
        self.grid.insert_array(np.stack([ix * scanner.cell + scanner.x0,
                                         iy * scanner.cell + scanner.y0,
                                         scanner.H[new]], axis=1).astype(np.float32))

    def trigger_rebuild(self):
        with self._lock:
            if self._busy[0] or self.grid.voxel_count < 30:
                return
            pts = self.grid.collect()
            self._busy[0] = True
            gen = self._gen
        threading.Thread(target=self._worker, args=(pts, gen), daemon=True).start()

    def _worker(self, pts, gen):
        try:
            field = tp.splat_points_to_field(pts, self.CELL, self.RAD, max_nodes=6_000_000)
            iso   = tp.marching_cubes(field, self.ISO)
            with self._lock:
                if gen != self._gen:
                    return               # cleared while we were building — drop it
                self._pending[0] = iso if not iso.empty else None
        finally:
            with self._lock:
                self._busy[0] = False

    def apply_pending(self):
        with self._lock:
            iso = self._pending[0]; self._pending[0] = None
        if iso is None:
            return
        pos, nrm = iso.positions, iso.normals
        n = min(pos.shape[0], self.MAX_VERTS)
        n -= n % 3                       # whole triangles only
        if n <= 0:
            self._geo.set_draw_range(0, 0)
            self._draw_n = 0
            self._apply_visibility()
            return
        if pos.shape[0] > self.MAX_VERTS:
            print(f"[slam] surface clipped to {self.MAX_VERTS} of {pos.shape[0]} verts")
        self._geo.update_attribute("position", pos[:n])
        self._geo.update_attribute("normal",   nrm[:n])
        self._geo.set_draw_range(0, n)
        self._draw_n = n
        self._apply_visibility()
        if not self._added:
            self.scene.add(self._surf)   # once per session, not per rebuild
            self._added = True

    def clear(self):
        """Hide the displayed surface and discard all accumulated data.

        Also invalidates any rebuild currently running, so its result is dropped
        rather than restoring the pre-clear surface a moment later.
        """
        with self._lock:
            self._gen += 1
            self._pending[0] = None
        self.grid.clear()
        self._sent = None
        self._geo.set_draw_range(0, 0)
        self._draw_n = 0
        self._apply_visibility()

    @property
    def busy(self):  return self._busy[0]
    @property
    def voxels(self): return self.grid.voxel_count

    def _apply_visibility(self):
        """Show the surface only when it actually has triangles.

        Dropping the draw range to 0 is NOT enough to blank it. The wireframe material routes
        this mesh through the renderer's overlay line path, which caches the line list it built
        from the geometry and refreshes that cache on the geometry's VERSION. update_attribute
        bumps the version (so a rebuild shows up), but set_draw_range alone does not — so after
        clear() the renderer keeps drawing the last cached surface until the next rebuild
        replaces it. That is what made the SLAM surface survive a reset. Visibility is evaluated
        per frame during traversal, so gating on it blanks the mesh immediately.
        """
        self._surf.visible = bool(self._visible[0] and self._draw_n > 0)

    @property
    def visible(self): return self._visible[0]
    @visible.setter
    def visible(self, v):
        self._visible[0] = v
        self._apply_visibility()


# ── path trail ────────────────────────────────────────────────────────────────
class PathTrail:
    MAX = 3000

    def __init__(self, scene):
        g = tp.BufferGeometry()
        g.set_attribute("position", np.zeros((self.MAX, 3), np.float32))
        g.set_draw_range(0, 0)
        mat = tp.LineBasicMaterial(); mat.color = 0xffffff
        self.line = tp.Line(g, mat); self.line.frustum_culled = False
        self.line.visible = False
        self._g   = g; self._pts = []; self._prev = None
        scene.add(self.line)

    @property
    def visible(self): return self.line.visible
    @visible.setter
    def visible(self, v): self.line.visible = v

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


# ── bench report ──────────────────────────────────────────────────────────────
def _bench_report(bt, total, period, scan_only, gpu):
    """Print the --bench summary. `bt`/`total`/`period`/`gpu` are per-frame and
    index-aligned with the warmup already trimmed; `scan_only` holds the scan
    phase of scan frames only. Seconds everywhere except `gpu`, which is the
    renderer's frame_timings dict and already milliseconds."""
    n = len(total)
    if n == 0:
        print(f"[bench] nothing measured — N must exceed the {BENCH_WARMUP}-frame warmup")
        return

    def ms(a):
        return np.asarray(a, np.float64) * 1000.0

    per, tot = ms(period), ms(total)
    over = int((per > BENCH_BUDGET).sum())
    print()
    print(f"== bench: {n} frames measured ({BENCH_WARMUP} warmup frames discarded) ==")
    print(f"overall   {1000.0 / max(float(per.mean()), 1e-9):.2f} fps")
    print(f"  period ms  mean {per.mean():8.3f}  p50 {np.percentile(per, 50):8.3f}"
          f"  p95 {np.percentile(per, 95):8.3f}  p99 {np.percentile(per, 99):8.3f}"
          f"  max {per.max():8.3f}")
    print(f"  over {BENCH_BUDGET} ms: {over}/{n} frames ({100.0 * over / n:.1f}%)")

    P = {k: ms(bt[k]) for k in BENCH_PHASES}
    print()
    print("cpu phases (ms)")
    print(f"  {'phase':<10}{'mean':>10}{'p50':>10}{'p95':>10}{'max':>10}")
    for k in BENCH_PHASES + ("total",):
        a = tot if k == "total" else P[k]
        print(f"  {k:<10}{a.mean():10.3f}{np.percentile(a, 50):10.3f}"
              f"{np.percentile(a, 95):10.3f}{a.max():10.3f}")

    smask = P["scan"] > 0.0
    print()
    print("scan vs non-scan frames (frame total, ms)")
    for label, sel in (("scan", smask), ("non-scan", ~smask)):
        if int(sel.sum()):
            a = tot[sel]
            print(f"  {label:<10}n={int(sel.sum()):<7}mean {a.mean():8.3f}  p95 {np.percentile(a, 95):8.3f}")
    if len(scan_only):
        a = ms(scan_only)
        print(f"  scan phase itself: mean {a.mean():8.3f}  p95 {np.percentile(a, 95):8.3f}")

    keys = sorted(gpu[0].keys()) if gpu and gpu[0] else []
    if keys:
        print()
        print("renderer frame_timings (ms)")
        print(f"  {'key':<22}{'mean':>10}{'p95':>10}")
        for k in keys:
            a = np.asarray([float(g.get(k, 0.0)) for g in gpu], np.float64)
            print(f"  {k:<22}{a.mean():10.3f}{np.percentile(a, 95):10.3f}")

    # hitch attribution — which phase dominated each over-budget frame
    stack = np.stack([P[k] for k in BENCH_PHASES])          # (phase, frame)
    hits  = np.nonzero(per > BENCH_BUDGET)[0]
    print()
    print(f"hitch blame ({len(hits)} frames over {BENCH_BUDGET} ms)")
    if len(hits):
        top    = stack[:, hits].argmax(axis=0)
        counts = sorted(((int((top == j).sum()), k) for j, k in enumerate(BENCH_PHASES)),
                        reverse=True)
        for c, k in counts:
            if c:
                print(f"  {k:<10}{c:7d}  ({100.0 * c / len(hits):.1f}%)")
        print("worst 5 frames")
        for i in hits[np.argsort(per[hits])[::-1][:5]]:
            order  = np.argsort(stack[:, i])[::-1][:3]
            phases = "  ".join(f"{BENCH_PHASES[int(j)]} {stack[int(j), i]:.2f}" for j in order)
            gt     = float(gpu[int(i)].get("gpu_total_ms", 0.0)) if int(i) < len(gpu) else 0.0
            print(f"  frame {BENCH_WARMUP + int(i) + 1:<7} period {per[i]:8.3f}   "
                  f"{phases}   gpu_total {gt:.2f}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",      type=int,   default=42)
    ap.add_argument("--amplitude", type=float, default=AMPLITUDE)
    ap.add_argument("--model",     default=os.path.join(_HERE, "spot_steps.pt"),
                    help="policy checkpoint (falls back to <model>_latest.pt)")
    ap.add_argument("--shot",      metavar="PNG")
    ap.add_argument("--windowed",  action="store_true",
                    help="1200x720 window instead of fullscreen")
    ap.add_argument("--bench",     type=int, default=0, metavar="N",
                    help="profile N frames: vsync off, auto-walk forward, print timing summary, exit")
    ap.add_argument("--recovery-model", dest="recovery_model", default=os.path.join(_HERE, "spot_recovery.pt"),
                    help="fall-recovery policy (train_spot_recovery.py): takes over when Spot tips past up_z "
                         f"{RECOVER_UP}, hands back once it has stood still for {RECOVER_SETTLE} ticks. F shoves, G trips. "
                         "Default: the shipped spot_recovery.pt beside this file (skipped if missing)")
    ap.add_argument("--no-recovery", dest="no_recovery", action="store_true",
                    help="walking policy only: no recovery hand-over, and the legacy viewer plant unless the walking "
                         "checkpoint trained with real torque limits")
    ap.add_argument("--self-collision", dest="self_collision", choices=("auto", "on", "off"), default="auto",
                    help="auto: on when a recovery policy is loaded (it trained with it); a walk + jump walker "
                         "(train_spot_steps.py --jump) trained without it")
    ap.add_argument("--headless-vx", dest="headless_vx", type=float, default=1.0,
                    help="with --shot: the headless walk's forward command (0 = stand; --jump-at then jumps from a stand)")
    ap.add_argument("--jump-at", dest="jump_at", type=int, default=0, metavar="TICK",
                    help="with --shot and a walk + jump walker: press J at this control tick of the headless walk and log "
                         "the flight, clearance, height, up_z and joint speeds over the jump window")
    ap.add_argument("--knock-at", dest="knock_at", type=int, default=0, metavar="TICK",
                    help="with --shot: trip Spot (G's push) at this control tick of the headless walk (tests the hand-over)")
    ap.add_argument("--trip-probe", dest="trip_probe", default="", metavar="OUT.json",
                    help="headless: 10 trips of the standing walking policy per dv of TRIP_LADDER up to the first that "
                         "tips it past up_z 0.5 in >= 8 of 10, then 10 F shoves; writes OUT.json and exits")
    ap.add_argument("--film", default="", metavar="OUT.mp4",
                    help=f"headless: render the scripted fall-recovery take (stand, walk, F shove, G trip, recovery, a "
                         f"trip from the other side, recovery, walk away) at {FILM_W}x{FILM_H} {FILM_FPS} fps to OUT.mp4 "
                         "(H.264 yuv420p); PNG frames go to OUT_frames/ and are deleted after the encode")
    ap.add_argument("--film-keep-frames", dest="film_keep_frames", action="store_true",
                    help="with --film: keep the PNG frames after the encode")
    ap.add_argument("--film-dry", dest="film_dry", action="store_true",
                    help="with --film: run the take without keeping frames and print the up_z / control-mode timeline "
                         "and the camera sightline check")
    ap.add_argument("--film-stills", dest="film_stills", default="", metavar="T1,T2,...",
                    help="with --film: keep only the frames at these ticks, as OUT_still_TTTT.png, and no video")
    ap.add_argument("--film-frames", dest="film_frames", type=int, default=0, metavar="N",
                    help="with --film: stop after N frames (a cost measurement)")
    ap.add_argument("--film-scale", dest="film_scale", type=float, default=0.75,
                    help="with --film: renderer render_scale (FSR reconstructs to the film size)")
    args = ap.parse_args()
    if args.no_recovery or (args.recovery_model and not os.path.exists(args.recovery_model)):
        args.recovery_model = ""
    assert tp.HAS_PHYSX, "needs a PhysX-enabled threepp build"
    headless = bool(args.shot) or bool(args.trip_probe) or bool(args.film)
    bench    = int(args.bench)

    # ── assets ────────────────────────────────────────────────────────────────
    assets = fetch_assets()
    model_path = args.model
    if not os.path.exists(model_path):
        _latest = os.path.splitext(model_path)[0] + "_latest.pt"
        model_path = _latest if os.path.exists(_latest) else model_path
    ac, norm, _meta = load_policy(model_path, device="cpu")
    ac.eval()
    # A stand_mode checkpoint saw a (0,0) clock with the phase held whenever its command was exactly zero; without
    # the same sentinel here it would trot in place off-distribution (play_spot_steps.py does the same).
    stand_mode = bool(_meta.get("stand_mode", False))
    print(f"[policy] {os.path.basename(model_path)}" + ("  (stand mode: no keys = stand still)" if stand_mode else ""))
    # A walk + jump checkpoint (train_spot_steps.py --jump) reads 98 inputs: the walking 96 plus the jump flag and phase over
    # the jump window that J starts, and it trained with a joint target rate limit (meta qd_max), applied here too.
    walk_jump = bool(_meta.get("jump_obs", False))
    walk_qd = float(_meta["qd_max"]) if _meta.get("qd_max") else None
    walk_win = int(_meta.get("jump_window", 60))
    if walk_jump or walk_qd:
        print("[policy] " + (f"walk + jump: J jumps ({walk_win}-tick window, standing or walking), " if walk_jump else "")
              + (f"joint targets <= {walk_qd:g} rad/s" if walk_qd else "no joint target limit"))
    rec = None
    if args.recovery_model:
        rec_ac, rec_norm, _rec_meta = load_policy(args.recovery_model, device="cpu")
        rec_ac.eval()
        # The recovery checkpoint's own action scale and target clip while it drives (wave 2 trains at 0.5 / 1.0,
        # clipped to the joint limits; a pilot meta has neither = 0.2 unclipped). The walking policy keeps ACTION_SCALE.
        rec_scale = float(_rec_meta.get("action_scale", ACTION_SCALE))
        rec_lo = np.array([LIM[j][0] for _ in range(4) for j in ("hx", "hy", "kn")], np.float32) \
            if _rec_meta.get("target_clip", False) else None      # add order, as set_drive_targets takes them
        rec_hi = np.array([LIM[j][1] for _ in range(4) for j in ("hx", "hy", "kn")], np.float32)
        # and its target rate limit (wave 3, spot_recovery_env.act), none in a wave 2 or pilot meta
        rec_qd = float(_rec_meta["qd_max"]) if _rec_meta.get("qd_max") else None
        rec = (rec_ac, rec_norm, rec_scale, rec_lo, rec_hi, rec_qd, float(_rec_meta.get("dt", 0.02)))
        print(f"[recovery] {os.path.basename(args.recovery_model)} takes over below up_z {RECOVER_UP}, action scale "
              f"{rec_scale}{' clipped to the joint limits' if rec_lo is not None else ''}, "
              f"{f'joint targets <= {rec_qd:g} rad/s' if rec_qd else 'no joint target speed limit'}  (F = shove, G = trip)")

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
    SPAWN_X, SPAWN_Y = -2.0, 0.0   # 2 m back from origin, facing +Y (90° CCW)
    _FEET = ((0.30, 0.17), (0.30, -0.17), (-0.30, 0.17), (-0.30, -0.17), (0.0, 0.0))
    h0 = max(float(gen.height_at(SPAWN_X + dx, SPAWN_Y + dy, tparams)) for dx, dy in _FEET)
    print(f"[terrain] done  h_footprint={h0:.3f} m")

    # ── physics ───────────────────────────────────────────────────────────────
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.002, max_substeps=20)
    base  = tp.Mesh(tp.BoxGeometry(WORLD_SZ + 10, WORLD_SZ + 10, 0.4), tp.MeshStandardMaterial())
    base.position.set(0, 0, -0.2)
    world.add_static(base)
    terr_phys = tp.Mesh(terr_geo, tp.MeshStandardMaterial())
    world.add_static_trimesh(terr_phys)

    # The plant the policies trained on: real torque limits when the walking checkpoint says so (the course policies), and
    # self-collision when a recovery policy is loaded (it trained with it on). Without them the viewer robot is stronger and
    # its legs pass through its own body, which flatters a recovery (29 of 29 knock-overs got up that way, 2026-09-14).
    _dlf = bool(_meta.get("drive_limits_are_forces", False)) or bool(args.recovery_model)
    _selfc = bool(args.recovery_model) and bool(_rec_meta.get("self_collision", True))
    if args.self_collision != "auto":
        _selfc = args.self_collision == "on"
    art, meshes = build_spot(world, assets, gains=STIFF_GAINS,   # stiff gains (90) = base gait's plant
                             drive_limits_are_forces=_dlf, self_collision=_selfc)
    print(f"[spot] plant: torque limits {'real' if _dlf else 'impulse (legacy)'}, self-collision {'on' if _selfc else 'off'}")
    # Pushes are sized from the total mass. The bindings expose no articulation link mass, so this sums what build_spot built,
    # link by link: each link's density is MASS / its collider's volume (box; capsule pi r^2 l + 4/3 pi r^3, the volumes PhysX
    # integrates), so these are the masses PhysX computed.
    _link_mass = [MASS["base"]] + [MASS[("hip", "uleg", "lleg")[(i - 1) % 3]] for i in range(1, len(art.links))]
    robot_mass = float(sum(_link_mass))
    print(f"[spot] total mass {robot_mass:.2f} kg over {len(_link_mass)} links: F shoves {SHOVE_DV:g} m/s "
          f"({robot_mass * SHOVE_DV:.0f} N·s), G trips {TRIP_DV:g} m/s ({robot_mass * TRIP_DV:.0f} N·s at {TRIP_HEIGHT:g} m)")

    def settle(n=80):
        for _ in range(n):
            art.set_drive_targets(default_q[add_to_isaac].astype(np.float32))
            world.step(0.02)

    _s = math.sqrt(2) / 2
    _spawn_quat = tp.Quaternion(0.0, 0.0, _s, _s)  # 90° CCW around Z
    # set_joint_positions writes PhysX DOF slots (every hip, then every upper leg, then every knee), NOT the add order
    # joint_positions() reads back. Learn the slot of each add-order joint once on the fresh robot rather than trust a
    # layout (measured 2026-09-13: add index i -> slot [0,4,8,1,5,9,2,6,10,3,7,11]).
    art.set_joint_positions((np.arange(12) + 1).astype(np.float32))
    _dof_slot = np.rint(np.asarray(art.joint_positions())).astype(int) - 1
    STANCE_Z = 0.505   # base height that puts the stance's foot tips 5 mm above flat ground (spot_slopes.STANCE_PRESET)

    def spawn_stance(ground_z):
        """Teleport into the trained stance just above the ground. art.reset() alone zeroes every joint (straight
        legs) and the old spawn dropped from Z0 = 0.72 m while settle() swung the legs into the stance mid-fall: the
        base sagged to 0.32 m. In the stance at 0.505 m it bottoms out at 0.42 m, up_z 0.998 (CPU probe, flat)."""
        art.reset(tp.Vector3(SPAWN_X, SPAWN_Y, ground_z + STANCE_Z), _spawn_quat)
        dof = np.empty(12, np.float32)
        dof[_dof_slot] = default_q[add_to_isaac]
        art.set_joint_positions(dof)

    spawn_stance(h0)
    settle()
    print("[spot] standing")

    # ── canvas + renderer ─────────────────────────────────────────────────────
    # --bench profiles the REAL interactive frame (window + ImGui); only vsync goes,
    # so the measured period is the frame's own cost rather than the display's.
    _bw, _bh = (FILM_W, FILM_H) if args.film else (1200, 720)
    # Interactive runs go fullscreen (--windowed opts out); --shot stays at the fixed
    # offscreen size. SPOT_BENCH_SIZE sizes the bench window explicitly, so it wins.
    _fullscreen = False #not headless and not args.windowed
    if bench and os.environ.get("SPOT_BENCH_SIZE"):
        _bw, _bh = (int(v) for v in os.environ["SPOT_BENCH_SIZE"].split("x"))
        _fullscreen = False
    _vsync = (not bench) or bool(os.environ.get("SPOT_BENCH_VSYNC"))
    canvas = tp.Canvas("threepp · Spot SLAM", width=_bw, height=_bh,
                       antialiasing=4, headless=headless, vsync=_vsync,
                       fullscreen=_fullscreen)
    rend = tp.VulkanRenderer(canvas)
    rend.shadow_map_enabled       = True
    rend.tone_mapping             = tp.ToneMapping.ACESFilmic
    rend.tone_mapping_exposure    = 1.1
    # Render below native and let FSR 3.1 reconstruct back up: the deferred shade is the
    # dominant GPU cost here and it scales with the internal resolution. Measured on an
    # RTX 4070 at 1920x1200 fullscreen with vsync, 0.5 is the largest scale whose GPU
    # frame (p95) still fits the 16.7 ms vblank budget in foliage-dense views — 0.58
    # already drops ~3% of vblanks. SPOT_BENCH_KNOBS (below) can still override
    # render_scale so perf sweeps keep working.
    rend.render_scale             = float(args.film_scale) if args.film else 0.5
    # if hasattr(rend, "gbuffer_msaa"):
    #     rend.gbuffer_msaa = 2
    if bench and os.environ.get("SPOT_BENCH_KNOBS"):
        # e.g. SPOT_BENCH_KNOBS="render_scale=0.67,probe_gi=0" — renderer attr
        # overrides for perf A/B sweeps; bench-only, floats coerced to bool for
        # bool-typed properties by pybind.
        for kv in os.environ["SPOT_BENCH_KNOBS"].split(","):
            k, v = kv.split("=")
            cur = getattr(rend, k.strip())
            val = float(v) if not isinstance(cur, bool) else bool(float(v))
            setattr(rend, k.strip(), val)
            print(f"[bench] knob {k.strip()} = {val}")

    # ── scene ─────────────────────────────────────────────────────────────────
    scene = tp.Scene()
    # HDR environment: equirect Radiance map drives image-based lighting (the Vulkan
    # PT importance-samples it) and serves as the sky backdrop. Cached on first run.
    try:
        env = tp.RGBELoader().load(fetch_file(HDR_URL, "~/.cache/threepp/hdri", "noon_grass_2k.hdr"))
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
    # Volumetric exponential fog (Beer-Lambert + single-scattering on Vulkan).
    # density = σ_t (extinction per metre); 0.02 ≈ 61% transmitted at 25 m.
    _fog_density  = [0.02]
    _fog_color    = 0x8ab4d4
    scene.set_fog_exp2(_fog_color, _fog_density[0])
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

    print("[trees] building species variants ...")
    tree_variants = build_tree_variants(args.seed)
    print("[trees] scattering ...")
    # Trees + stones get static colliders (Spot bumps trunks/boulders); bushes stay
    # soft (walk-through). Keep the proxy meshes alive for the whole run.
    phys_proxies = []
    tree_places, bush_places, stone_places = [], [], []     # the --film camera's sightline test
    phys_proxies += scatter_trees(scene, gen, tparams, tree_variants, seed=args.seed, world=world, pond=pond,
                                  placements=tree_places)
    scatter_bushes(scene, gen, tparams, seed=args.seed, pond=pond, placements=bush_places)
    phys_proxies += scatter_stones(scene, gen, tparams, seed=args.seed, world=world, pond=pond,
                                   placements=stone_places)

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

    # ── camera + orbit controls ───────────────────────────────────────────────
    w, h = canvas.size()
    camera = tp.PerspectiveCamera(50, w / h, 0.05, 200)
    camera.up.set(0, 0, 1)
    spawn_z = Z0 + h0
    camera.position.set(SPAWN_X - 3.0, SPAWN_Y - 3.0, spawn_z + 1.8)
    _init_tgt = np.array([SPAWN_X, SPAWN_Y, spawn_z + 0.3])
    _prev_tgt = [_init_tgt.copy()]    # tracks last target so we can apply delta to camera
    controls = tp.OrbitControls(camera, canvas)
    controls.target.set(*_init_tgt.tolist())
    controls.enable_damping = True
    controls.damping_factor = 0.10
    controls.enable_pan    = False    # pan would shift target away from Spot
    controls.min_distance  = 1.5
    controls.max_distance  = 30.0
    controls.update()

    # ── SLAM objects ──────────────────────────────────────────────────────────
    half = WORLD_SZ / 2.0
    # Vulkan's depth sensor traces the TLAS from the last render(); GL re-renders internally.
    is_vulkan = type(rend).__name__ == "VulkanRenderer"
    if is_vulkan:
        # scene.set_fog_exp2 above IS the single volumetric knob now: the froxel
        # sun shafts + aerial glow follow automatically wherever the fog medium is
        # present (the old set_volumetric_fog opt-in is a deprecated no-op). Only
        # the phase anisotropy is still worth setting — forward scatter gives the
        # sun-ward glow / god-ray shafts through the trees.
        rend.fog_anisotropy = 0.6
    scanner = ForwardDepthScanner(rend, scene, meshes,
                                  bounds=(-half, half, -half, half),
                                  cell=0.15, far=SENSOR_FAR,
                                  width=SENSOR_W, height=SENSOR_H,
                                  mount_fwd=0.95, mount_up=-0.10,
                                  pitch_deg=40.0, fov_y=90.0)
    if is_vulkan:
        rend.render(scene, camera)   # build an initial TLAS so prewarm's scans have geometry
    scanner.prewarm(art.root_state(), foot_tips=cpu_foot_tips(art))   # seed the ground under the body from the feet
    print(f"[scan] ground under the body seeded from the feet: h_here {scanner.h_here_last:.3f} m")
    slam    = SlamMapper(scene)
    trail   = PathTrail(scene)
    last_act    = np.zeros(12, np.float32)
    # Start from a real reading of the seeded map, not zeros and h0: h0 is the generator's HIGHEST point under the
    # footprint (6.564 m at the default spawn) while the feet stand at 6.19 m, so until the first in-loop scan landed
    # the policy was told its body sat ~5 cm off the ground, and the first spawn fell (R worked: its caches were real).
    _ahead0, _h_here0 = scanner.scan(art.root_state())
    ahead_cache = [_ahead0]       # last sensor reading; reused on skipped frames
    h_here_cache= [_h_here0]

    # ── state ─────────────────────────────────────────────────────────────────
    fc        = [0]
    _nh       = [0]      # harvests delivered (SPOT_SCANSTATS diagnostic)
    hdg_lock  = [None]
    r_held    = [False]
    gphi      = [0.0]    # gait phase clock ∈ [0,1); advanced +0.02/GAIT_PERIOD per control tick
    cmd_s     = np.zeros(2, np.float32)   # vx, vy after the acceleration limit (keys step 0 -> full speed in one tick)
    vx_hi     = [float(VX_HI)]    # live forward-speed cap (UI slider sets it)

    # ── bench state (untouched unless --bench) ────────────────────────────────
    bench_t     = {k: [] for k in BENCH_PHASES}   # per-phase seconds, one entry per frame
    bench_total = []              # whole frame() body, seconds
    bench_per   = []              # wall delta between frame() entries (true frame period)
    bench_scan  = []              # scan phase on scan frames only
    bench_gpu   = []              # rend.frame_timings snapshot per frame
    _t_prev     = [None]          # wall clock at the previous frame() entry
    _tm         = [0.0]           # section stopwatch
    _has_ft     = hasattr(rend, "frame_timings")
    if bench and os.environ.get("SPOT_BENCH_GC") == "off":
        import gc
        gc.freeze(); gc.disable()   # isolate GC pauses from the timing signal
        print("[bench] gc disabled")

    def _mark(key):
        now = time.perf_counter()
        bench_t[key].append(now - _tm[0])
        _tm[0] = now

    mode      = ["walk"]   # "walk" | "recover" (--recovery-model)
    settled   = [0]        # consecutive recovered-and-still ticks
    f_held    = [False]
    g_held    = [False]
    j_held    = [False]
    wjump     = [-1]       # a walk + jump walker: ticks into the jump window J started (-1: none)

    def recovery_tick(rs):
        """Hand-over logic, once per control tick: True when the recovery policy drives this tick (it has then acted).
        The recovery policy sees the walking observation with a zero command, the (0,0) stand clock, a flat scan and
        base_above against the terrain under the base, as it was trained (spot_recovery_env.observe)."""
        if rec is None:
            return False
        x, y, z, qx, qy = float(rs[0]), float(rs[1]), float(rs[2]), float(rs[3]), float(rs[4])
        up = 1.0 - 2.0 * (qx * qx + qy * qy)
        ground = float(gen.height_at(x, y, tparams))
        if mode[0] == "walk":
            if up >= RECOVER_UP:
                return False
            mode[0] = "recover"; settled[0] = 0; wjump[0] = -1
            # last_act is in the driving policy's action units: keep the joint targets it stands for across the hand-over
            last_act[:] *= ACTION_SCALE / rec[2]
            print(f"[recovery] up_z {up:.2f}: recovery policy takes over (action scale {rec[2]}"
                  f"{', clipped' if rec[3] is not None else ''}"
                  f"{f', joint targets <= {rec[5]:g} rad/s starting from the walking targets' if rec[5] else ''})")
        speed = float(np.linalg.norm(art.root_velocity()[:3]))
        stood = up > 0.9 and z - ground > 0.40 and speed < 0.2
        settled[0] = settled[0] + 1 if stood else 0
        if settled[0] >= RECOVER_SETTLE:
            mode[0] = "walk"; hdg_lock[0] = None; cmd_s[:] = 0.0
            last_act[:] *= rec[2] / ACTION_SCALE
            print(f"[recovery] upright and settled: walking policy takes back (action scale {ACTION_SCALE})")
            return False
        obs = v2_obs(art, last_act, np.zeros(3, np.float32), np.zeros(45, np.float32), ground, None)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs)[None]
            if rec[1] is not None:
                obs_t = rec[1].norm(obs_t)
            a = rec[0].act_mean(obs_t)[0].numpy()
        if rec[5]:
            # the training env's rate limit (spot_recovery_env.act): the target moves at most qd_max * dt per tick. last_act
            # holds the action of the last targets, rescaled at the takeover, so the limiter starts from the walking targets
            da = rec[5] * rec[6] / rec[2]
            a = (last_act + np.clip(a - last_act, -da, da)).astype(np.float32)
        last_act[:] = a
        tgt = (default_q + rec[2] * a)[add_to_isaac].astype(np.float32)
        if rec[3] is not None:
            tgt = np.clip(tgt, rec[3], rec[4])      # as spot_recovery_env.act clips them in training
        art.set_drive_targets(tgt)
        return True

    def walk_step(cmd, ahead, h_here, stand):
        """One tick of the walking policy: its observation (plus the jump flag and phase for a walk + jump walker, as
        spot_steps_env.observe appends them), the action, the joint target rate limit it trained with, the drive targets;
        then the J window advances (the policy saw phase 0 .. window-1 / window, spot_steps_env._jump_step)."""
        obs = v2_obs(art, last_act, cmd, ahead, h_here, None if stand else gphi[0])
        if walk_jump:
            w = 1.0 if wjump[0] >= 0 else 0.0
            obs = np.concatenate([obs, np.array([w, max(wjump[0], 0) / walk_win * w], np.float32)])
        with torch.no_grad():
            obs_t = torch.from_numpy(obs)[None]
            if norm is not None:
                obs_t = norm.norm(obs_t)
            a = ac.act_mean(obs_t)[0].numpy()
        if walk_qd:
            da = walk_qd * 0.02 / ACTION_SCALE
            a = (last_act + np.clip(a - last_act, -da, da)).astype(np.float32)
        last_act[:] = a
        art.set_drive_targets((default_q + ACTION_SCALE * a)[add_to_isaac].astype(np.float32))
        if wjump[0] >= 0:
            wjump[0] = wjump[0] + 1 if wjump[0] + 1 < walk_win else -1

    def feet_clearance(rs):
        """The lowest foot's height above the terrain under it (m), from the joint angles and the base pose through
        build_spot's capsule chain (spot_recovery_env.leg_segments): the headless jump log's airborne test."""
        from spot_recovery_env import leg_segments
        q = torch.from_numpy(np.asarray(art.joint_positions(), np.float32))[None]
        tips = leg_segments(q, samples=2)[0][0, 8:12, 1].numpy()     # lower legs' foot-end samples, body frame
        w = tips @ _quat_to_R(rs[3:7]).T + np.asarray(rs[:3], float)
        return min(float(t[2]) - 0.028 - float(gen.height_at(float(t[0]), float(t[1]), tparams)) for t in w)

    push_rng = np.random.default_rng(args.seed + 1)

    def push(kind, dv=None, quiet=False, side=None):
        """F ("shove"): SHOVE_DV through the base's centre of mass. G ("trip"): TRIP_DV at TRIP_HEIGHT above it. The
        impulse is robot_mass x dv toward a random side of the body, jittered PUSH_JITTER_DEG, for one substep:
        add_force_at_pos is consumed by the next fixed substep, so the force carrying the impulse is impulse / fixed_timestep.
        side = +1 / -1 pushes exactly toward the body's left / right instead (--film). -> (impulse N·s, dv m/s, height m)."""
        dv = float(dv if dv is not None else (SHOVE_DV if kind == "shove" else TRIP_DV))
        h = 0.0 if kind == "shove" else TRIP_HEIGHT
        rs = art.root_state()
        left = _quat_to_R(rs[3:7])[:, 1].copy()
        if side is not None:
            ang = math.atan2(float(left[1]), float(left[0])) + (0.0 if side > 0 else math.pi)
        else:
            ang = (math.atan2(float(left[1]), float(left[0])) + (0.0 if push_rng.random() < 0.5 else math.pi)
                   + math.radians(PUSH_JITTER_DEG) * push_rng.uniform(-1.0, 1.0))
        J = robot_mass * dv
        f = np.array([math.cos(ang), math.sin(ang), 0.0]) * (J / 0.002)
        art.link(0).add_force_at_pos(tp.Vector3(*[float(v) for v in f]),
                                     tp.Vector3(float(rs[0]), float(rs[1]), float(rs[2]) + h))
        if not quiet:
            print(f"[{kind}] {dv:g} m/s sideways = {J:.1f} N·s on {robot_mass:.2f} kg, {h:g} m above the base's centre of mass")
        return J, dv, h

    def reset():
        rh = max(float(gen.height_at(SPAWN_X + dx, SPAWN_Y + dy, tparams)) for dx, dy in _FEET)
        spawn_stance(rh)
        mode[0] = "walk"; settled[0] = 0; wjump[0] = -1
        last_act[:] = 0.0
        hdg_lock[0] = None
        gphi[0] = 0.0
        cmd_s[:] = 0.0
        settle(40)
        # A scan fired before the teleport is still in flight; harvesting it after the map is
        # cleared would fuse pre-reset terrain under the new pose. Collect and discard it.
        if getattr(scanner.sensor, "scan_pending", False):
            scanner.sensor.scan_collect(rend)
        scanner.clear_map(); scanner.prewarm(art.root_state(), foot_tips=cpu_foot_tips(art))
        ahead_cache[0], h_here_cache[0] = scanner.scan(art.root_state())   # the first ticks read this map, not the old pose's
        slam.clear()
        trail.clear()

    def _save_frame(path):
        """Renderer-agnostic screenshot: GL takes only path; Vulkan takes scene+camera+path."""
        try:
            rend.save_frame(path)
        except TypeError:
            rend.save_frame(scene, camera, path)

    # ── headless ──────────────────────────────────────────────────────────────
    def stand_ticks(n):
        """n control ticks of the walking policy with a zero command (the stand clock for a stand-mode checkpoint),
        scanning as the headless walk does -> the smallest up_z seen."""
        up_min = 1.0
        for i in range(n):
            rs = art.root_state()
            up_min = min(up_min, 1.0 - 2.0 * (float(rs[3]) ** 2 + float(rs[4]) ** 2))
            if i % SCAN_EVERY == 0:
                rend.render(scene, camera)
                ahead_cache[0], h_here_cache[0] = scanner.scan(rs)
            obs = v2_obs(art, last_act, np.zeros(3, np.float32), ahead_cache[0], h_here_cache[0],
                         None if stand_mode else gphi[0])
            with torch.no_grad():
                obs_t = torch.from_numpy(obs)[None]
                a = ac.act_mean(norm.norm(obs_t) if norm is not None else obs_t)[0].numpy()
            last_act[:] = a
            art.set_drive_targets((default_q + ACTION_SCALE * a)[add_to_isaac].astype(np.float32))
            world.step(0.02)
            if not stand_mode:
                gphi[0] = (gphi[0] + 0.02 / GAIT_PERIOD) % 1.0
        return up_min

    def run_trip_probe(path):
        """--trip-probe: per dv of TRIP_LADDER, 10 trips of a standing Spot (reset, 0.5 s standing, the push, 2 s more),
        stopping at the first dv that takes up_z below RECOVER_UP in >= 8 of 10; then 10 F shoves. No recovery policy
        acts (the plant is still the one --recovery-model builds). -> JSON."""
        import json

        def trials(kind, dv, n=10):
            ups = []
            for _ in range(n):
                reset()
                stand_ticks(25)
                push(kind, dv, quiet=True)
                ups.append(stand_ticks(100))
            fell = int(sum(u < RECOVER_UP for u in ups))
            h = 0.0 if kind == "shove" else TRIP_HEIGHT
            print(f"[trip-probe] {kind} {dv:g} m/s = {robot_mass * dv:.1f} N·s, {h:g} m above the base: up_z < {RECOVER_UP} "
                  f"in {fell}/{n}  (min up_z {sorted(round(u, 2) for u in ups)})")
            return {"kind": kind, "dv": dv, "impulse": robot_mass * dv, "height": h, "fell": fell, "trials": n, "up_min": ups}

        res = {"amplitude": float(args.amplitude), "seed": args.seed, "robot_mass": robot_mass,
               "trip_height": TRIP_HEIGHT, "jitter_deg": PUSH_JITTER_DEG, "policy": os.path.basename(model_path),
               "plant": {"drive_limits_are_forces": _dlf, "self_collision": _selfc}, "trip": []}
        for dv in TRIP_LADDER:
            res["trip"].append(trials("trip", dv))
            if res["trip"][-1]["fell"] >= 8:
                break
        res["shove"] = trials("shove", SHOVE_DV)
        with open(path, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[trip-probe] -> {path}")

    def run_film(out_path):
        """--film: the scripted take (FILM_* above), one frame per control tick. Three modes run the identical loop, and
        differ only in which frames they read back: all (PNG frames -> H.264), none (--film-dry: the up_z / control-mode
        timeline and the sightline check) or the listed ticks (--film-stills). A dry run that rendered only the scan ticks
        drifted from the rendered take by 80 ticks at the first hand-back (2026-09-14), so every mode renders every tick.
        The SLAM wireframe and the 45-cell scan markers are off: at 1920x1080 the wireframe covers the ground and the
        tree crowns and buries the fall; the depth camera's point cloud stays."""
        import shutil, subprocess
        from concurrent.futures import ThreadPoolExecutor
        from PIL import Image
        DT = 0.02
        dry = bool(args.film_dry)
        stills = sorted({int(s) for s in args.film_stills.split(",") if s.strip()})
        full = not dry and not stills
        base_path = os.path.splitext(os.path.abspath(out_path))[0]
        frame_dir = base_path + "_frames"
        # The walk's heading and the camera azimuth's zero: the spawn heading (+Y), re-chosen at every hand-back as the
        # clear corridor nearest the way Spot faces (the old course if it faces within 30 deg of it). Turning far back at
        # a crawl knocked it over again, and walking on blindly took it into a spruce (2026-09-14). The camera swings
        # round after it on a smoothed copy.
        course = [math.pi / 2]
        cw, ch = canvas.size()
        if (cw, ch) != (FILM_W, FILM_H):
            print(f"[film] WARNING: canvas is {cw}x{ch}, not {FILM_W}x{FILM_H}")
        camera.fov = FILM_FOV
        camera.aspect = cw / max(ch, 1)
        camera.update_projection_matrix()
        slam.visible = False
        scanner.show_grid = False
        if getattr(scanner, "markers", None):
            # show_grid only takes effect at the next scan, after frame 0: without this the pre-film reset's markers
            # popped up for one frame. The group is in the scanner's self-filter list, so no scan ever saw it anyway.
            scanner.marker_group.visible = False

        # Everything that can hide the robot, as vertical cylinders (x, y, radius, z0, z1): trunks, crowns, bushes, stones.
        cyl = []
        for x, y, z, cr, th, ht in tree_places:
            # crowns from near the ground and at least 0.3 x the height wide: the spruce skirts reach the grass, and the
            # preset crown radius missed one Spot walked into (tick 1300, 2026-09-14)
            cyl += [(x, y, 0.35, z, z + ht), (x, y, max(cr, 0.3 * ht), z + 0.3, z + ht)]
        cyl += [(x, y, cr, z, z + ht) for x, y, z, cr, ht in bush_places]
        cyl += [(x, y, r, z, z + 1.4 * r) for x, y, z, r in stone_places]
        CY = np.asarray(cyl, np.float64).reshape(-1, 5)
        solid = np.asarray([(x, y, 0.35) for x, y, *_ in tree_places] + [(x, y, r) for x, y, _, r in stone_places],
                           np.float64).reshape(-1, 3)          # colliders the robot can walk into
        _S = np.linspace(0.0, 1.0, 24)

        def sight_blocked(eye, tgt):
            """The eye -> target segment passes through an obstacle (its last 0.5 m skipped: that is the robot), or the eye
            brushes one: within 1.5 m of the eye every obstacle counts 0.8 m bigger. Without that pad the opening shot sat
            in a bush's leaves, which filled the right quarter of the frame for three seconds (2026-09-14)."""
            d = tgt - eye
            L = max(float(np.linalg.norm(d)), 1e-6)
            s = _S * max(0.0, 1.0 - 0.5 / L)
            P = eye[None] + s[:, None] * d[None]
            pad = np.where(s * L < 1.5, 0.8, 0.0)[:, None]
            dx, dy, pz = P[:, 0:1] - CY[None, :, 0], P[:, 1:2] - CY[None, :, 1], P[:, 2:3]
            R = CY[None, :, 2] + pad
            return bool(((dx * dx + dy * dy < R * R) & (pz > CY[None, :, 3] - pad) & (pz < CY[None, :, 4] + pad)).any())

        def corridor(x, y, z, heading):
            """Smallest gap (m) between the next 7 m of a straight walk on `heading` and anything at body height."""
            k = np.arange(1, 15)[:, None] * 0.5
            C = CY[CY[:, 3] < z + 0.3]
            if not len(C):
                return 1e9
            return float((np.hypot(x + k * math.cos(heading) - C[None, :, 0],
                                   y + k * math.sin(heading) - C[None, :, 1]) - C[None, :, 2]).min())

        def ground(x, y):
            return float(gen.height_at(float(x), float(y), tparams))

        def lift(eye, tgt):
            """Raise the eye FILM_EYE_CLEAR above the terrain under it, then until the sightline clears the terrain by 0.4 m."""
            eye = eye.copy()
            eye[2] = max(eye[2], ground(eye[0], eye[1]) + FILM_EYE_CLEAR)
            for s in (0.15, 0.3, 0.45, 0.6, 0.75):
                p = eye + s * (tgt - eye)
                need = ground(p[0], p[1]) + 0.4 - p[2]
                if need > 0.0:
                    eye[2] += need / (1.0 - s)
            return eye

        def spring(x, v, goal, w):
            """Critically damped second-order follow: no overshoot, no snap."""
            v = v + (w * w * (goal - x) - 2.0 * w * v) * DT
            return x + v * DT, v

        def eye_for(tgt, az_deg, dist, elev_deg):
            a, e = cam["course"] + math.radians(az_deg), math.radians(elev_deg)
            return lift(tgt + dist * np.array([math.cos(a) * math.cos(e), math.sin(a) * math.cos(e), math.sin(e)]), tgt)

        cam = {"off": 0.0, "nom_ok": 0, "blocked_goal": 0, "blocked_view": 0}

        def camera_tick(rs, pulled, az_nom, first=False):
            """Chase camera: target, distance, elevation and azimuth each spring to their goal, then the eye springs to
            the pose they give. The azimuth goal leaves the nominal quarter only when that sightline is blocked, for the
            nearest clear offset, and returns once the nominal has been clear for 25 ticks."""
            p = np.array([float(rs[0]), float(rs[1]), float(rs[2]) + 0.1])
            dist_goal, elev_goal = (FILM_DIST[1], FILM_ELEV[1]) if pulled else (FILM_DIST[0], FILM_ELEV[0])
            if first:
                cam.update(tgt=p, tv=np.zeros(3), dist=dist_goal, dv=0.0, elev=elev_goal, ev=0.0, az=az_nom, av=0.0,
                           course=course[0], cv=0.0)
            else:
                cam["course"], cam["cv"] = spring(cam["course"], cam["cv"], course[0], 1.2)
                cam["tgt"], cam["tv"] = spring(cam["tgt"], cam["tv"], p, 3.0)
                cam["dist"], cam["dv"] = spring(cam["dist"], cam["dv"], dist_goal, 1.8)
                cam["elev"], cam["ev"] = spring(cam["elev"], cam["ev"], elev_goal, 1.8)
            tgt = cam["tgt"]
            if sight_blocked(eye_for(tgt, az_nom + cam["off"], cam["dist"], cam["elev"]), tgt):
                cam["blocked_goal"] += 1
                for off in (0.0, 20.0, -20.0, 40.0, -40.0, 60.0, -60.0, 80.0, -80.0):
                    if not sight_blocked(eye_for(tgt, az_nom + off, cam["dist"], cam["elev"]), tgt):
                        cam["off"], cam["nom_ok"] = off, 0
                        break
            elif cam["off"] != 0.0:
                clear = not sight_blocked(eye_for(tgt, az_nom, cam["dist"], cam["elev"]), tgt)
                cam["nom_ok"] = cam["nom_ok"] + 1 if clear else 0
                if cam["nom_ok"] >= 25:
                    cam["off"] = 0.0
            if first:
                cam["az"] = az_nom + cam["off"]
            else:
                cam["az"], cam["av"] = spring(cam["az"], cam["av"], az_nom + cam["off"], 1.6)
            goal = eye_for(tgt, cam["az"], cam["dist"], cam["elev"])
            if first:
                cam["eye"], cam["eyev"] = goal, np.zeros(3)
            else:
                cam["eye"], cam["eyev"] = spring(cam["eye"], cam["eyev"], goal, 4.0)
                cam["eye"][2] = max(cam["eye"][2], ground(cam["eye"][0], cam["eye"][1]) + 0.5)   # never inside a hill
            if sight_blocked(cam["eye"], tgt):
                cam["blocked_view"] += 1
            camera.position.set(*(float(v) for v in cam["eye"]))
            camera.look_at(*(float(v) for v in tgt))

        def up_of(rs):
            return 1.0 - 2.0 * (float(rs[3]) ** 2 + float(rs[4]) ** 2)

        if full:
            os.makedirs(frame_dir, exist_ok=True)
            for f in os.listdir(frame_dir):
                if f.startswith("f") and f.endswith(".png"):
                    os.remove(os.path.join(frame_dir, f))
        pool = None if dry else ThreadPoolExecutor(max_workers=4)
        pending = []
        has_simtime = hasattr(rend, "sim_time")

        trips, events, log = [], [], []          # trips: {tick, side, takeover, handback, up_min}
        ev = {"trip2": None, "end": None}
        last_hb, stand_until, shove_up = None, FILM_STAND0, 1.0
        t_sim, n_frames, closest = 0.0, 0, 1e9
        cost = {"render": 0.0, "read": 0.0, "save": 0.0}
        rs = art.root_state()
        camera_tick(rs, False, FILM_AZ[0], first=True)
        grass.time = 0.0
        if has_simtime:
            rend.sim_time = 0.0
        for _ in range(8):        # pipelines, and the FSR/TAA history at the first pose: no first-frame garbage
            rend.render(scene, camera)
        t_wall = time.perf_counter()
        i = 0
        while True:
            if ev["end"] is not None and i >= ev["end"]:
                break
            if (args.film_frames and n_frames >= args.film_frames) or (stills and i > stills[-1]):
                break
            rs = art.root_state()
            if i == FILM_SHOVE:
                push("shove", side=-1)                           # away from the camera's side
                events.append((i, "F shove toward the body's right"))
            if i == FILM_TRIP1 or i == ev["trip2"]:
                side = -1 if not trips else 1                    # the second from the other side
                push("trip", side=side)
                trips.append({"tick": i, "side": side, "takeover": None, "handback": None, "up_min": 1.0})
                events.append((i, f"G trip {len(trips)} toward the body's {'right' if side < 0 else 'left'}"))
            # the command: stand, or walk the course heading (slowing to turn when far off it); zero while recovering
            R = _quat_to_R(rs[3:7])
            err = (math.atan2(float(R[1, 0]), float(R[0, 0])) - course[0] + math.pi) % (2 * math.pi) - math.pi
            vx_goal = 0.0 if (i < stand_until or mode[0] == "recover") else FILM_VX * float(np.clip(math.cos(err), 0.2, 1.0))
            cmd_s[:] = cmd_s + np.clip(np.array([vx_goal, 0.0], np.float32) - cmd_s, -0.06, 0.06)
            stand = stand_mode and float(cmd_s[0]) == 0.0 and vx_goal == 0.0
            wz = 0.0 if stand else float(np.clip(-2.0 * err, -1.0, 1.0))
            mode_before = mode[0]
            recovering = recovery_tick(rs)
            if mode[0] != mode_before:
                tr = trips[-1] if trips else None
                if mode[0] == "recover":
                    if tr is not None and tr["takeover"] is None:
                        tr["takeover"] = i
                    events.append((i, f"recovery takes over (up_z {up_of(rs):.2f})"))
                else:
                    last_hb, stand_until = i, i + FILM_HB_STAND
                    facing = course[0] + err                     # unwrapped: the camera swings the short way round
                    cands = ([course[0]] if abs(err) <= math.radians(30.0) else []) + \
                        [facing + math.radians(d) for d in (0, 15, -15, 30, -30, 45, -45, 60, -60, 75, -75, 90, -90)]
                    gaps = [corridor(rs[0], rs[1], rs[2], h) for h in cands]
                    k = next((j for j, g in enumerate(gaps) if g >= 1.2), int(np.argmax(gaps)))
                    course[0] = cands[k]
                    events.append((i, f"hand-back to walking, facing {math.degrees(err):+.0f} deg off the course; walks on "
                                      f"{math.degrees(course[0] - facing):+.0f} deg from its facing ({gaps[k]:.1f} m clear)"))
                    err = facing - course[0]
                    if tr is not None and tr["handback"] is None:
                        tr["handback"] = i
                        if len(trips) == 1:
                            ev["trip2"] = i + FILM_TRIP2
                        else:
                            ev["end"] = i + FILM_END
            for k, tr in enumerate(trips):                      # a trip that did not floor it, or never got up
                if tr["handback"] is None and i - tr["tick"] in (150, FILM_TIMEOUT):
                    if tr["takeover"] is None and i - tr["tick"] == 150:
                        print(f"[film] WARNING trip {k + 1} at tick {tr['tick']} did not tip Spot past up_z {RECOVER_UP}")
                        tr["handback"] = -1
                        if k == 0:
                            ev["trip2"] = i + FILM_TRIP2
                        else:
                            ev["end"] = i + FILM_END
                    elif i - tr["tick"] == FILM_TIMEOUT:
                        print(f"[film] WARNING trip {k + 1}: no hand-back within {FILM_TIMEOUT} ticks")
                        ev["end"] = i + 50
            if not recovering:
                walk_step(np.array([float(cmd_s[0]), 0.0, wz], np.float32), ahead_cache[0], h_here_cache[0], stand)
            world.step(DT)
            t_sim += DT
            if not stand and not recovering:
                gphi[0] = (gphi[0] + DT / GAIT_PERIOD) % 1.0
            rs = art.root_state()
            up = up_of(rs)
            if trips and trips[-1]["handback"] is None:
                trips[-1]["up_min"] = min(trips[-1]["up_min"], up)
            if FILM_SHOVE <= i < FILM_TRIP1:
                shove_up = min(shove_up, up)
            if len(solid):
                closest = min(closest, float(np.min(np.hypot(solid[:, 0] - rs[0], solid[:, 1] - rs[1]) - solid[:, 2])))

            nxt = FILM_TRIP1 if not trips else (ev["trip2"] if len(trips) == 1 else None)
            pulled = ((nxt is not None and i >= nxt - 40) or mode[0] == "recover"
                      or (bool(trips) and i - trips[-1]["tick"] < 40) or (last_hb is not None and i - last_hb < 50))
            az_nom = FILM_AZ[1] if (trips and trips[0]["handback"] is not None and i >= trips[0]["handback"] + 25) \
                else FILM_AZ[0]
            camera_tick(rs, pulled, az_nom)
            grass.time = t_sim
            if has_simtime:
                rend.sim_time = t_sim

            t0 = time.perf_counter()
            rend.render(scene, camera)
            t1 = time.perf_counter()
            cost["render"] += t1 - t0
            if full or i in stills:
                px = rend.read_pixels()
                t2 = time.perf_counter()
                cost["read"] += t2 - t1
                if n_frames == 0:
                    print(f"[film] read_pixels {px.shape} {px.dtype}")
                if full:
                    path = os.path.join(frame_dir, f"f{n_frames:05d}.png")
                    pending.append(pool.submit(lambda a=px, p=path: Image.fromarray(a).save(p, compress_level=1)))
                    while len(pending) > 8:
                        pending.pop(0).result()
                else:
                    path = f"{base_path}_still_{i:04d}.png"
                    Image.fromarray(px).save(path)
                    print(f"[film] still tick {i} (t {i * DT:.2f} s, up_z {up:.2f}, {mode[0]}) -> {path}")
                cost["save"] += time.perf_counter() - t2
                n_frames += 1
            if i % SCAN_EVERY == 0:
                # no point cloud while recovering: the tumbling depth camera sprays it over trees and sky (tick 950)
                scanner.show_cloud = mode[0] != "recover"
                ahead_cache[0], h_here_cache[0] = scanner.scan(rs)   # traces the TLAS of the render just above
            if i % (25 if dry else 50) == 0:
                el = time.perf_counter() - t_wall
                line = (f"[film] tick {i:5d}  t {i * DT:6.2f} s  up_z {up:+.2f}  {mode[0]:<7}  pos ({rs[0]:+6.2f}, "
                        f"{rs[1]:+6.2f}, {rs[2]:5.2f})  yaw off course {math.degrees(err):+4.0f}  "
                        f"cam off {cam['off']:+.0f} dist {cam['dist']:.2f}"
                        + (f"  {el / max(n_frames, 1) * 1000:.0f} ms/frame" if n_frames else ""))
                print(line)
            log.append((i, up, mode[0]))
            i += 1

        for f in pending:
            f.result()
        if pool is not None:
            pool.shutdown()
        wall = time.perf_counter() - t_wall
        print(f"\n[film] {i} ticks = {i * DT:.2f} s simulated, {n_frames} frames, {wall:.1f} s wall")
        if n_frames:
            print("[film] per frame: " + "  ".join(f"{k} {v / n_frames * 1000:.1f} ms" for k, v in cost.items())
                  + f"  | all-in {wall / n_frames * 1000:.1f} ms")
        print("[film] events:")
        for tick, what in events:
            print(f"  tick {tick:5d}  t {tick * DT:6.2f} s  {what}")
        print(f"  shove: min up_z {shove_up:+.2f} before the first trip")
        ok = len(trips) == 2
        for k, tr in enumerate(trips):
            floored, got_up = tr["up_min"] < RECOVER_UP, (tr["handback"] or -1) >= 0
            ok = ok and floored and got_up
            print(f"  trip {k + 1} at tick {tr['tick']}: min up_z {tr['up_min']:+.2f} ({'floored' if floored else 'NOT floored'}), "
                  f"takeover {tr['takeover']}, hand-back {tr['handback'] if got_up else 'NONE'}")
        print("[film] control by tick: " + ", ".join(f"{t}:{m}" for k, (t, _, m) in enumerate(log)
                                                     if k == 0 or log[k - 1][2] != m))
        print(f"[film] camera: goal sightline blocked {cam['blocked_goal']} ticks, the view itself {cam['blocked_view']} "
              f"ticks; the robot's base came within {closest:.2f} m of a trunk or stone")
        print(f"[film] CHECK {'PASS' if ok else 'FAIL'}: both trips floor Spot and both recoveries hand back")
        if not full or not n_frames:
            return
        exe = shutil.which("ffmpeg")
        if not exe:
            try:
                import imageio_ffmpeg
                exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                print(f"[film] no ffmpeg on PATH and no imageio-ffmpeg: frames kept in {frame_dir}")
                return
        cmd = [exe, "-y", "-hide_banner", "-loglevel", "error", "-framerate", str(FILM_FPS),
               "-i", os.path.join(frame_dir, "f%05d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "20",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", os.path.abspath(out_path)]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[film] ffmpeg failed, frames kept in {frame_dir}:\n{r.stderr[-1500:]}")
            return
        print(f"[film] {out_path}: {n_frames} frames at {FILM_FPS} fps = {n_frames / FILM_FPS:.2f} s, "
              f"{os.path.getsize(out_path) / 1e6:.1f} MB, encoded in {time.perf_counter() - t0:.1f} s")
        if not args.film_keep_frames:
            shutil.rmtree(frame_dir, ignore_errors=True)

    if headless:
        # Spawn the way R and the interactive loop do. Without it the headless walk started from the startup spawn, which
        # had already tipped to up_z 0.45 by the first tick and ended the 150-tick walk upside down (measured 2026-09-13,
        # cb737743 + the recovery hand-over's [headless] line).
        reset()
        if args.trip_probe:
            run_trip_probe(args.trip_probe)
            return
        if args.film:
            run_film(args.film)
            return
        cmd = np.array([args.headless_vx, 0.0, 0.0], np.float32)
        # Vulkan: scan() uses the TLAS from the last render(), so render before each scan.
        # GL: scan() re-renders internally anyway; the extra render() is a cheap no-op for screenshots.
        rend.render(scene, camera)
        n_ticks = max(150, args.knock_at + 250) if (args.knock_at and rec is not None) else 150
        if args.jump_at and walk_jump:
            n_ticks = max(n_ticks, args.jump_at + 350)
        up_min, modes, v_pre = 1.0, [], None
        jl = {"ticks": 0, "air": 0, "clear": 0.0, "height": 0.0, "up": 1.0, "qd": np.zeros(3)}
        for i in range(n_ticks):
            rs = art.root_state()
            up_min = min(up_min, 1.0 - 2.0 * (float(rs[3]) ** 2 + float(rs[4]) ** 2))
            if args.knock_at and i == args.knock_at:
                print(f"[trip] tick {i}")
                v_pre = np.asarray(art.root_velocity()[:3], float).copy()
                push("trip")
            if v_pre is not None and i == args.knock_at + 1:
                dvb = np.asarray(art.root_velocity()[:3], float) - v_pre
                print(f"[trip] base velocity change over that tick: {np.linalg.norm(dvb[:2]):.2f} m/s horizontal, "
                      f"{dvb[2]:+.2f} m/s vertical (whole-robot dv {TRIP_DV:g} m/s)")
            if args.jump_at and walk_jump and i == args.jump_at:
                wjump[0] = 0
                print(f"[jump] tick {i}: J, command {cmd.tolist()}")
            if not modes or modes[-1][1] != mode[0]:
                modes.append((i, mode[0]))
            if i % SCAN_EVERY == 0:
                rend.render(scene, camera)   # refresh TLAS for Vulkan
                ahead, h_here = scanner.scan(rs)
            if recovery_tick(rs):
                world.step(0.02)
                continue
            stand_now = stand_mode and not cmd.any()
            in_win = wjump[0] >= 0
            walk_step(cmd, ahead, h_here, stand_now)
            world.step(0.02)
            if in_win:
                rs2 = art.root_state()
                c = feet_clearance(rs2)
                jl["ticks"] += 1; jl["air"] += int(c > 0.02); jl["clear"] = max(jl["clear"], c)
                jl["height"] = max(jl["height"], float(rs2[2]) - float(gen.height_at(float(rs2[0]), float(rs2[1]), tparams)))
                jl["up"] = min(jl["up"], 1.0 - 2.0 * (float(rs2[3]) ** 2 + float(rs2[4]) ** 2))
                jl["qd"] = np.maximum(jl["qd"], np.abs(np.asarray(art.joint_velocities(), float)).reshape(4, 3).max(0))
            if not stand_now:
                gphi[0] = (gphi[0] + 0.02 / GAIT_PERIOD) % 1.0
        rs = art.root_state()
        print(f"[headless] {n_ticks} ticks, min up_z {up_min:.2f}, final up_z "
              f"{1.0 - 2.0 * (float(rs[3]) ** 2 + float(rs[4]) ** 2):.2f}"
              + (f", trip at tick {args.knock_at}" if args.knock_at else "")
              + (f", control by tick {modes + [(n_ticks, mode[0])]}" if rec is not None else ""))
        if args.jump_at and walk_jump:
            print(f"[jump] headless: {jl['ticks']} jump-window ticks, all feet clear of the ground {jl['air']} ticks "
                  f"({jl['air'] * 0.02:.2f} s), lowest-foot clearance max {jl['clear']:.3f} m, base height above the ground "
                  f"max {jl['height']:.3f} m, up_z min {jl['up']:.2f}, joint speed max hx/hy/kn "
                  f"{np.round(jl['qd'], 1).tolist()} rad/s")
        for _ in range(20):
            rend.render(scene, camera)
            ahead, h_here = scanner.scan(rs)
        slam.insert_scanner(scanner)
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
        if rec is not None or walk_jump:
            tp.imgui.text(f"control: {mode[0]}   (F shove, G trip{', J jump' if walk_jump else ''})")
        _, vx_hi[0] = tp.imgui.slider_float("forward speed vx (VX_HI)", vx_hi[0], 0.0, VX_HI)
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
        _, slam.visible  = tp.imgui.checkbox("show SLAM mesh", slam.visible)
        _, trail.visible = tp.imgui.checkbox("show path trail", trail.visible)
        if tp.imgui.button("rebuild surface now") and not slam.busy:
            slam.trigger_rebuild()
        if tp.imgui.button("reset (R)"):
            reset()
        tp.imgui.separator()
        tp.imgui.text("Fog (volumetric on Vulkan)")
        chg, v = tp.imgui.slider_float("density", _fog_density[0], 0.002, 0.15)
        if chg:
            _fog_density[0] = v
            scene.set_fog_exp2(_fog_color, v)
        if is_vulkan:
            # Sun shafts are automatic with the fog medium now (the old
            # "sun shafts" toggle was a deprecated no-op — removed). The density
            # slider above is the shaft on/off knob; anisotropy shapes the glow.
            chg, v = tp.imgui.slider_float("anisotropy g", rend.fog_anisotropy, -0.9, 0.9)
            if chg:
                rend.fog_anisotropy = v
        tp.imgui.separator()
        tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps   |   WASD+QE  F shove  G trip  J jump  mouse=orbit/zoom")
        tp.imgui.end()

    def frame():
        fc[0] += 1
        if bench:
            _t_frame = time.perf_counter()
            bench_per.append(0.0 if _t_prev[0] is None else _t_frame - _t_prev[0])
            _t_prev[0] = _t_frame

        # keyboard command — WASD drive, QE turn (numpad kept as an alternate)
        if bench:
            vx, vy, wz_key = vx_hi[0], 0.0, 0.0   # auto-walk forward; heading hold below
        else:
            vx = (vx_hi[0] if down("W", "KP8") else 0.0) - (1.0 if down("S", "KP2") else 0.0)
            vy = (1.0 if down("A", "KP4") else 0.0) - (1.0 if down("D", "KP6") else 0.0)
            wz_key = (1.5 if down("Q", "KP7") else 0.0) - (1.5 if down("E", "KP9") else 0.0)
        # Joystick-style smoothing: at most 3 m/s^2 (0 -> 1.5 m/s in 0.5 s). The clip lands exactly on the target,
        # so releasing the keys still reaches an exact zero command and stand mode engages.
        cmd_s[:] = cmd_s + np.clip(np.array([vx, vy], np.float32) - cmd_s, -0.06, 0.06)   # in place: frame() closes over it
        vx, vy = float(cmd_s[0]), float(cmd_s[1])

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
        if bench: _tm[0] = time.perf_counter()
        stand = stand_mode and vx == 0.0 and vy == 0.0 and wz_key == 0.0
        if stand:
            wz = 0.0                             # training's stand command is exactly zero: no heading-hold turn
        if not bench:                            # F = shove, G = trip (edge-triggered)
            for key, held, kind in (("F", f_held, "shove"), ("G", g_held, "trip")):
                if down(key):
                    if not held[0]: push(kind)
                    held[0] = True
                else:
                    held[0] = False
            if walk_jump:                         # J = jump, standing or walking (edge-triggered): the walking policy jumps
                if down("J"):
                    if not j_held[0] and mode[0] == "walk" and wjump[0] < 0:
                        wjump[0] = 0
                        print(f"[jump] J: a {walk_win}-tick jump window at command vx {vx:.2f} vy {vy:.2f} wz {wz:.2f}")
                    j_held[0] = True
                else:
                    j_held[0] = False
        recovering = recovery_tick(rs)           # the recovery policy acted this tick
        if not recovering:
            walk_step(np.array([vx, vy, wz], np.float32), ahead_cache[0], h_here_cache[0], stand)
        if bench: _mark("policy")
        world.step(0.02)
        if bench: _mark("physics")
        if not stand and not recovering:         # stand mode holds the phase where it stopped
            gphi[0] = (gphi[0] + 0.02 / GAIT_PERIOD) % 1.0
        rs = art.root_state()
        if bench: _tm[0] = time.perf_counter()
        if fc[0] % MC_FRAMES == 0 and not (bench and os.environ.get("SPOT_BENCH_NO_MC")):
            slam.trigger_rebuild()
        slam.apply_pending()
        trail.update(rs)
        if bench: _mark("slam")

        # SPOT_BENCH_RESET_AT=N exercises the reset path (incl. dropping an in-flight scan and
        # invalidating a running SLAM rebuild) without a keyboard — see reset().
        if bench and _RESET_AT and fc[0] == _RESET_AT:
            print(f"[reset-test] f{fc[0]} before: voxels={slam.voxels} busy={slam.busy} "
                  f"draw_verts={slam._draw_n}")
            reset()
            print(f"[reset-test] f{fc[0]} after:  voxels={slam.voxels} busy={slam.busy} "
                  f"draw_verts={slam._draw_n}")
        if bench and _RESET_AT and _RESET_AT < fc[0] <= _RESET_AT + 120 and fc[0] % 15 == 0:
            print(f"[reset-test] f{fc[0]} +{fc[0]-_RESET_AT:3d}: voxels={slam.voxels} "
                  f"busy={slam.busy} draw_verts={slam._draw_n}")

        # R = reset (no keyboard in bench mode)
        if not bench:
            if down("R"):
                if not r_held[0]: reset(); print("[reset]")
                r_held[0] = True
            else:
                r_held[0] = False

        # Follow Spot: shift camera by same delta as target so OrbitControls
        # sees an unchanged spherical offset (stays in orbit around Spot
        # rather than fixed in world space while Spot walks away).
        if bench: _tm[0] = time.perf_counter()
        p = np.array(rs[:3], float)
        new_tgt = np.array([float(p[0]), float(p[1]), float(p[2]) + 0.3])
        delta = new_tgt - _prev_tgt[0]
        _prev_tgt[0] = new_tgt
        cp = camera.position
        camera.position.set(cp.x + delta[0], cp.y + delta[1], cp.z + delta[2])
        controls.target.set(float(new_tgt[0]), float(new_tgt[1]), float(new_tgt[2]))
        controls.enabled = not (ui and ui.want_capture_mouse)
        controls.update()
        if bench: _mark("camera")

        # ── depth scan, PRE-render half: harvest ──────────────────────────────────────────
        # Vulkan scan() is synchronous: it fires a GPU trace and then waits behind the frame
        # already in flight (~17-30 ms here). Pipelined instead — the fire goes AFTER this
        # frame's render (below), the harvest happens on a LATER frame, by which time the
        # trace has retired and the collect is ~free. The elevation map is world-anchored, so
        # fusing a cloud a few frames old is correct; the 45 query points use the current rs.
        _scan_dt = 0.0
        _scan_work = False
        if bench: _t_scan = time.perf_counter()
        if is_vulkan:
            # The SLAM surface is wireframe → excluded from the PT TLAS; the trail (Line)
            # and point cloud (Points) are never ray-traced either → nothing to hide here.
            res = scanner.scan_harvest(rs)
            if res is not None:
                ahead_cache[0], h_here_cache[0] = res
                slam.insert_scanner(scanner)
                _scan_work = True
                _nh[0] += 1
                if _SCANSTATS and _nh[0] % 20 == 0:
                    _a = ahead_cache[0]
                    print(f"[scanstats] f{fc[0]:5d} ahead min={_a.min():+.3f} max={_a.max():+.3f} "
                          f"std={_a.std():.4f} nonzero={int((_a != 0).sum())}/45 "
                          f"h_here={h_here_cache[0]:.3f} voxels={slam.voxels}")
        elif fc[0] % SCAN_EVERY == 0:
            # GL: the sensor re-renders the scene and blocks anyway, so keep the fully
            # synchronous path — hide the SLAM surface + trail (the scanner already hides
            # the robot + its own cloud/grid).
            _extra = [(o, o.visible) for o in (slam._surf, trail.line) if o is not None]
            for o, _ in _extra: o.visible = False
            ahead_cache[0], h_here_cache[0] = scanner.scan(rs)
            for o, v in _extra: o.visible = v
            slam.insert_scanner(scanner)
            _scan_work = True
        if bench: _scan_dt += time.perf_counter() - _t_scan

        if bench: _tm[0] = time.perf_counter()
        grass.time = time.perf_counter() - t0  # advance GPU wind (Vulkan)
        rend.render(scene, camera)             # single render per frame — no flicker
        if bench: _mark("render")

        # ── depth scan, POST-render half: fire ────────────────────────────────────────────
        # Right after render() so the trace runs against the TLAS this render just built.
        # The beams snapshot the sensor pose now; a later frame harvests the cloud.
        if is_vulkan and fc[0] % SCAN_EVERY == 0:
            if bench: _t_scan = time.perf_counter()
            scanner.scan_fire(rs)
            if bench: _scan_dt += time.perf_counter() - _t_scan
            _scan_work = True
        if bench:
            # one entry per frame (phase lists stay frame-aligned); harvest + fire share it
            bench_t["scan"].append(_scan_dt if _scan_work else 0.0)
            if _scan_work:
                bench_scan.append(_scan_dt)

        if ui:
            if bench: _tm[0] = time.perf_counter()
            ui.render(draw_ui)
            if bench: _mark("ui")
        elif bench:
            bench_t["ui"].append(0.0)

        if bench:
            bench_total.append(time.perf_counter() - _t_frame)
            bench_gpu.append(dict(rend.frame_timings) if _has_ft else {})
            if (os.environ.get("SPOT_BENCH_SHOT")
                    and fc[0] == int(os.environ.get("SPOT_BENCH_SHOT_FRAME", "1000"))):
                _save_frame(os.environ["SPOT_BENCH_SHOT"])   # quality A/B capture
            if fc[0] >= bench:
                canvas.close()

    print(__doc__)
    if bench:
        print(f"[bench] profiling {bench} frames (first {BENCH_WARMUP} are warmup) ...")
    # Spawn exactly the way R does, as the last thing before the loop. The startup spawn runs seconds earlier, before
    # the scene, trees and scanner exist, and still fell on its first ticks with the stance spawn and a seeded map;
    # reset() is the path watched to stand (2026-09-13).
    reset()
    t0 = time.perf_counter()
    canvas.animate(frame)

    if bench:
        w = BENCH_WARMUP
        warm_scans = sum(1 for v in bench_t["scan"][:w] if v > 0.0)
        _bench_report({k: v[w:] for k, v in bench_t.items()},
                      bench_total[w:], bench_per[w:], bench_scan[warm_scans:], bench_gpu[w:])


if __name__ == "__main__":
    main()
