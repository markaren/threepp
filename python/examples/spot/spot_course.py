"""Mixed-terrain TRAINING COURSE for Spot: hills, cross-tilted bands and rough ground next to the stair strip.

Until 2026-09-13 the policy trained on SpotStepsEnv's tents (70% of lanes) and flat steering replay (30%). The showpiece
before rows (score_e0.py --suite showpiece, K=2048, Idun) measured what that world never shows it: placement near chance,
0.12-0.68 m/m downhill drift on 20 deg cross-slopes, 0.4-0.6 falls per 4 m/s shove. This module is the terrain half of
the course and nothing else: the family registry, the lane layout, one band per curriculum level along +x (the stair
strip's idiom), the builders and the per-family distances the curriculum reads. SpotStepsEnv(course=True) and
lane_family='hills' | 'cross_bands' | 'rough' wire it.

Families (share of lanes in the default course, lanes laid out as CONTIGUOUS blocks in BLOCK_ORDER):
    flat    0.15  steering replay, unchanged (no geometry)
    stairs  0.35  today's tents and risers (SpotStepsEnv._add_steps builds them)
    hills   0.20  per band: flat approach, an up ramp, a landing, a down ramp, a run-out, at HILL_DEGS
    cross   0.15  per band a section tilted about the lane centreline at CROSS_DEGS
    rough   0.15  per band smooth 2-D noise at ROUGH_AMPS (spot_heightfield_env.make_hf_grids)

Why contiguous blocks and not the stair env's per-lane coin flip: every box here is exactly one lane wide, so two
neighbouring lanes of one family have identical geometry at every x and meet without a step. A lane of another family
meets them at a cliff (up to HILL_RUN * tan 25 = 1.03 m beside a hill landing, CROSS_PIVOT + 1.5 tan 20 = 1.15 m
beside a cross band's high edge). The coin flip would put a cliff at ~60% of lane edges; blocks put one at 4 edges.
BLOCK_ORDER puts the gentlest neighbours together (flat | stairs | rough | hills | cross).

Geometry, per family (x from the strip start at 0; band j starts at j * band_len; SPAWN_OFF into the band):
  hills  HILL_APPROACH flat, a planar ramp of HILL_RUN horizontal metres (2.21-2.43 m along the surface, so the 0.7 m
         footprint has >= 1.5 m of walking wholly on it), HILL_LAND at the top, the mirror ramp down, HILL_RUNOUT flat.
         Every band starts and ends at ground level, so bands join without a cliff. Ramps are spot_slopes' rotated
         slabs (solid wedges: the underside passes below the ground under the whole top face), landings solid boxes.
  cross  the centreline sits at CROSS_PIVOT for the whole strip, so it is continuous from band to band and only the
         lane edges step, by 1.5 * (tan th_j+1 - tan th_j) <= 0.108 m with the angles in increasing order (the order that
         minimises it). Band 0 reaches back CROSS_BACK and the top band runs on to the end of the ground, so no lane ever
         walks off a ledge. The tilt alternates lane to lane inside a run of cross lanes (spot_slopes 'corrugate'): lane
         edges meet at one height, a V gutter or a ridge, never a 2 * 1.5 tan th cliff. The pivot height is what keeps
         the low edge (CROSS_PIVOT - 1.5 tan 20 = 0.054 m) above the shared ground box.
  rough  one noise tile per (lane, band) from ROUGH_SHAPES shapes, amplitude ROUGH_AMPS[j] (peak to valley, heights >= 0
         on top of the ground), edge-tapered to 0 in x and y so bands and neighbouring rough lanes join at 0. One static
         collider per lane over its whole strip, on a ROUGH_CELL grid: a trimesh (default, one PhysX cook per lane) or
         a height field. The raycast BVH must see the surface the feet collide with: a trimesh goes through
         CollectedWorld, so the BVH is built from the very Mesh PhysX cooked; a height field has no Mesh, so its BVH
         copy is triangulated here with ROUGH_DIAGONAL and is exact only if PhysX splits cells the same way.
         MEASURED 2026-09-13 (course smoke, RTX 4070, K=2048, 307 rough lanes x 5400 tris, every ingredient on,
         raycast_s4 act_mean): trimesh builds in 7.3 s, 1965 MiB after the build / 2207 peak, 28.2k env steps/s, and
         stance feet read a median -0.5 mm (p01 -2.5 mm) against the raycast; the height field builds in 7.1 s,
         2310 / 2471 MiB, 25.7k env steps/s, and its stance feet read a median -8.5 mm (p01 -30 mm): the BVH copy is
         NOT the surface PhysX collides with (diagonal or row mapping). Trimesh is the default; the height field stays
         an option until that is fixed.

Curriculum distances (forward progress from the spawn, SpotStepsEnv.ep_max_climb): feature_dist is the end of the band's
feature (hills: past the down ramp; cross and rough: the band's end); crossing it promotes. stairs keep the tent's
CLEAR_DIST and REACH_DIST rule. See SpotStepsEnv._course_done for the demotions.
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
import spot_slopes as ss

# --------------------------------------------------------------------------- #
#  Registry
# --------------------------------------------------------------------------- #
FAMILIES = ("flat", "stairs", "hills", "cross", "rough")
FLAT, STAIRS, HILLS, CROSS, ROUGH = range(len(FAMILIES))
# SpotStepsEnv lane_family names. 'cross' there is spot_slopes' single-angle scoring lane, so the course one is named
# after what it is: a strip of tilted bands.
LANE_FAMILY = {"flat": "flat", "stairs": "stairs", "hills": "hills", "cross": "cross_bands", "rough": "rough"}
CODE_OF_LANE_FAMILY = {"hills": HILLS, "cross_bands": CROSS, "rough": ROUGH}
DEFAULT_SHARES = {"flat": 0.15, "stairs": 0.35, "hills": 0.20, "cross": 0.15, "rough": 0.15}
BLOCK_ORDER = ("flat", "stairs", "rough", "hills", "cross")
N_LEVELS = 6
SPAWN_OFF = 0.7                  # = spot_steps_env.SPAWN_OFF (asserted there; importing it here would be circular)

HILL_DEGS = (4.0, 8.0, 12.0, 16.0, 20.0, 25.0)
HILL_APPROACH = 1.6              # = the stair band's FLAT_APPROACH: the spawn and the line-up are the same
HILL_RUN = 2.2                   # horizontal ramp length
HILL_LAND = 1.0
HILL_RUNOUT = 0.9
HILL_BAND = HILL_APPROACH + 2.0 * HILL_RUN + HILL_LAND + HILL_RUNOUT      # 7.9 m

CROSS_DEGS = (4.0, 8.0, 12.0, 15.0, 18.0, 20.0)
CROSS_BAND = 5.0
CROSS_PIVOT = 0.60               # centreline height (> 1.5 tan 20 = 0.546, so the low edge clears the ground)
CROSS_BACK = 3.0                 # band 0's slab reaches this far behind the strip start

ROUGH_AMPS = (0.02, 0.05, 0.08, 0.11, 0.14, 0.17)
ROUGH_BAND = 6.0
ROUGH_CELL = 0.2                 # one grid for both backends (a height field needs the same spacing in x and y)
ROUGH_SHAPES = 16                # = spot_heightfield_env.NUM_SHAPES
ROUGH_TAPER = 0.25               # raised-cosine edge window share: 1.2 m in x, 0.6 m in y at the lane edges
ROUGH_DIAGONAL = "ll_hh"         # BVH copy of a height field: each cell split (x0,y0)-(x1,y1)
ROUGH_BACKENDS = ("trimesh", "heightfield")

# Level ladders. 'pilot' is the ladder above (the 2026-09-13 pilot trained on it, and score_e0's course cells RG05 /
# RG11 / RG17 / H08 / H16 / H25 name its values, so it stays the default of every world that is not course=True). 'hard'
# is the wave-2 training ladder: the pilot reached the top level on stairs, cross and rough within 300 iterations, so the
# top of those three moves up. Hills are unchanged: 25 deg is still unclimbed. The stair risers only apply to a
# course=True world (SpotStepsEnv passes them as its riser ladder, so the tread closed form and the placement reward follow).
LADDERS = {
    "pilot": {"hills": HILL_DEGS, "cross": CROSS_DEGS, "rough": ROUGH_AMPS, "risers": None},
    "hard": {"hills": HILL_DEGS, "cross": (5.0, 9.0, 13.0, 17.0, 21.0, 25.0),
             "rough": (0.03, 0.07, 0.11, 0.15, 0.20, 0.25), "risers": (0.05, 0.09, 0.13, 0.17, 0.20, 0.23)},
}
EDGE_FOOT_DY = 0.21              # a stance foot's lateral offset (spot_feet: settled tips dy +-0.208..0.215)


def cross_pivot(degs, spacing):
    """The cross strip's centreline height for a ladder: CROSS_PIVOT, or the lowest 5 cm step that keeps the steepest
    band's low edge 5 cm above the ground (0.60 m for the pilot's 20 deg, 0.75 m for 25 deg)."""
    need = 0.5 * spacing * math.tan(math.radians(max(degs))) + 0.05
    return max(CROSS_PIVOT, round(math.ceil(round(need / 0.05, 6)) * 0.05, 6))


def cross_edge_steps(degs, spacing, dy=None):
    """[len-1] height step where cross band j meets band j+1, at lateral offset dy (default the lane edge). Lane to lane
    the corrugation makes it 0 at every x: neighbouring lanes' edges meet at one height (a gutter or a ridge)."""
    dy = 0.5 * spacing if dy is None else dy
    t = [math.tan(math.radians(d)) for d in degs]
    return [dy * (t[j + 1] - t[j]) for j in range(len(t) - 1)]


BAND_LEN = {HILLS: HILL_BAND, CROSS: CROSS_BAND, ROUGH: ROUGH_BAND}
FEATURE_DIST = {HILLS: (HILL_APPROACH - SPAWN_OFF) + 2.0 * HILL_RUN + HILL_LAND,   # 6.3 m: past the down ramp
                CROSS: CROSS_BAND - SPAWN_OFF,                                     # 4.3 m: the band's end
                ROUGH: ROUGH_BAND - SPAWN_OFF}                                     # 5.3 m
DEMOTE_PROGRESS = 0.4            # hills/cross/rough: a timeout under this share of the commanded forward distance demotes
LEVEL_VALUE = {HILLS: HILL_DEGS, CROSS: CROSS_DEGS, ROUGH: ROUGH_AMPS}
LEVEL_UNIT = {HILLS: "deg", CROSS: "deg", ROUGH: "m"}
IK_SPAWN = (HILLS, CROSS, ROUGH)  # spawned on spot_slopes' IK stance and held through the settle


def parse_shares(text):
    """'flat=0.15,stairs=0.35,...' (or a dict) -> {family: share}, every family present, normalised to sum 1."""
    if text is None or text == "":
        d = dict(DEFAULT_SHARES)
    elif isinstance(text, dict):
        d = {k: float(v) for k, v in text.items()}
    else:
        d = {}
        for part in str(text).split(","):
            k, v = part.split("=")
            d[k.strip()] = float(v)
    unknown = sorted(set(d) - set(FAMILIES))
    if unknown:
        raise ValueError(f"course shares: unknown families {unknown}, expected {list(FAMILIES)}")
    d = {f: max(0.0, float(d.get(f, 0.0))) for f in FAMILIES}
    tot = sum(d.values())
    if tot <= 0.0:
        raise ValueError("course shares must not all be 0")
    return {f: v / tot for f, v in d.items()}


def lane_families(k, shares=None):
    """[k] SpotStepsEnv lane_family names: contiguous blocks in BLOCK_ORDER sized by share (largest remainder; a family
    with a share > 0 gets at least one lane when k allows)."""
    sh = parse_shares(shares)
    order = [f for f in BLOCK_ORDER if sh[f] > 0.0]
    raw = np.array([sh[f] * k for f in order])
    cnt = np.floor(raw).astype(np.int64)
    for i in np.argsort(-(raw - cnt))[: k - int(cnt.sum())]:
        cnt[i] += 1
    for i in range(len(order)):                       # nobody at 0 while someone can spare one
        if cnt[i] == 0 and cnt.max() > 1:
            cnt[int(np.argmax(cnt))] -= 1
            cnt[i] += 1
    out = []
    for f, n in zip(order, cnt):
        out += [LANE_FAMILY[f]] * int(n)
    return out


def family_codes(lane_family):
    """SpotStepsEnv lane_family names -> [K] course codes (every non-course name, slope lanes included, is FLAT but
    'stairs', which is STAIRS)."""
    return np.array([STAIRS if f == "stairs" else CODE_OF_LANE_FAMILY.get(f, FLAT) for f in lane_family], np.int64)


# --------------------------------------------------------------------------- #
#  Lanes
# --------------------------------------------------------------------------- #
class CourseLanes:
    """The hills / cross / rough lanes of one world: builds their colliders and answers the per-lane band length and
    feature distance. codes [K] course codes (FLAT and STAIRS lanes build nothing here); bands [K] or None: a lane builds
    only that band (an eval world with frozen levels) instead of all N_LEVELS. ladder: a LADDERS key."""

    def __init__(self, codes, spacing, bands=None, seed=0, rough_backend="trimesh", ladder="pilot"):
        if rough_backend not in ROUGH_BACKENDS:
            raise ValueError(f"rough_backend must be one of {ROUGH_BACKENDS}, got {rough_backend!r}")
        if ladder not in LADDERS:
            raise ValueError(f"ladder must be one of {list(LADDERS)}, got {ladder!r}")
        self.codes = np.asarray(codes, np.int64).reshape(-1)
        self.k, self.spacing = self.codes.shape[0], float(spacing)
        self.bands = None if bands is None else np.asarray(bands, np.int64).reshape(-1)
        self.rough_backend = rough_backend
        self.ladder = ladder
        lad = LADDERS[ladder]
        self.hill_degs, self.cross_degs, self.rough_amps = lad["hills"], lad["cross"], lad["rough"]
        self.level_value = {HILLS: self.hill_degs, CROSS: self.cross_degs, ROUGH: self.rough_amps}
        self.cross_pivot = cross_pivot(self.cross_degs, self.spacing)
        # corrugation: alternate the tilt inside each contiguous run of cross lanes, so lane edges meet at one height
        run = np.zeros(self.k, np.int64)
        for i in range(1, self.k):
            if self.codes[i] == CROSS and self.codes[i - 1] == CROSS:
                run[i] = run[i - 1] + 1
        self.cross_sign = np.where(run % 2 == 0, 1.0, -1.0)
        rng = np.random.default_rng(int(seed) + 4099)
        self.rough_shape = rng.integers(0, ROUGH_SHAPES, size=(self.k, N_LEVELS))
        self._grids = None
        self._materials = []
        self.n_boxes = self.n_rough_tris = self.n_rough_colliders = 0

    def any(self):
        return bool(np.isin(self.codes, IK_SPAWN).any())

    def band_len(self):
        """[K] band length per lane (0 on flat and stair lanes: the env keeps its own for those)."""
        return np.array([BAND_LEN.get(int(c), 0.0) for c in self.codes], np.float64)

    def feature_dist(self):
        """[K] forward progress from the spawn that walks the band's feature (nan on flat and stair lanes)."""
        return np.array([FEATURE_DIST.get(int(c), np.nan) for c in self.codes], np.float64)

    def spawn_x_max(self):
        """The largest spawn x any course lane can have."""
        xs = [((N_LEVELS - 1) if self.bands is None else int(self.bands[i])) * BAND_LEN[int(c)] + SPAWN_OFF
              for i, c in enumerate(self.codes) if int(c) in BAND_LEN]
        return max(xs) if xs else 0.0

    def _levels(self, i):
        return range(N_LEVELS) if self.bands is None else (int(self.bands[i]),)

    # ---- builders --------------------------------------------------------------------------------------------
    def build(self, world, material=None, x_end=60.0, bvh_sink=None):
        """Add every hills / cross / rough collider to `world` (a PhysxWorld or a CollectedWorld). material: None (the
        world's default contact, as every existing terrain box) or a world.create_material object; the rough colliders
        always take the default (add_static_trimesh and add_static_heightfield take no material from Python). x_end: the
        shared ground's far end, where a cross lane's top band stops. bvh_sink: the CollectedWorld's mesh list, for the
        height-field backend's BVH copy."""
        _box = ss.SlopeLanes._box
        W = self.spacing
        for i in range(self.k):
            c, y = int(self.codes[i]), i * self.spacing
            if c == HILLS:
                for j in self._levels(i):
                    th = math.radians(self.hill_degs[j])
                    s, co, t = math.sin(th), math.cos(th), math.tan(th)
                    L, rise = HILL_RUN / co, HILL_RUN * t
                    T = HILL_RUN * s + ss.SLAB_MARGIN        # vertical drop T / cos exceeds the rise at the ramp top
                    xf = j * HILL_BAND + HILL_APPROACH
                    top, n = (xf + 0.5 * HILL_RUN, y, 0.5 * rise), (-s, 0.0, co)       # R_y(-th): local x up the slope
                    _box(world, (L, W, T), [top[a] - n[a] * 0.5 * T for a in range(3)], "y", -th, material)
                    xl = xf + HILL_RUN
                    _box(world, (HILL_LAND, W, rise - ss.BOX_BOTTOM),
                         (xl + 0.5 * HILL_LAND, y, 0.5 * (rise + ss.BOX_BOTTOM)), None, 0.0, material)
                    xd = xl + HILL_LAND
                    top, n = (xd + 0.5 * HILL_RUN, y, 0.5 * rise), (s, 0.0, co)        # R_y(+th): local x down the slope
                    _box(world, (L, W, T), [top[a] - n[a] * 0.5 * T for a in range(3)], "y", th, material)
                    self.n_boxes += 3
            elif c == CROSS:
                sg = float(self.cross_sign[i])
                lv = list(self._levels(i))
                for j in lv:
                    th = math.radians(self.cross_degs[j])
                    s, co, t = math.sin(th), math.cos(th), math.tan(th)
                    x0 = -CROSS_BACK if j == lv[0] else j * CROSS_BAND
                    x1 = x_end if j == lv[-1] else (j + 1) * CROSS_BAND
                    # thick enough that the underside is below the ground under the high edge: T/cos >= pivot + 1.5 tan + 0.1
                    T = co * (self.cross_pivot + 0.5 * W * t + 0.1) + W * s
                    top, n = (0.5 * (x0 + x1), y, self.cross_pivot), (0.0, -sg * s, co)
                    _box(world, (x1 - x0, W / co, T), [top[a] - n[a] * 0.5 * T for a in range(3)], "x", sg * th,
                         material)
                    self.n_boxes += 1
            elif c == ROUGH:
                self._add_rough(world, i, y, bvh_sink)

    def rough_grid(self, i):
        """Lane i's rough strip as (x0, z [nx, ny]): heights on the ROUGH_CELL grid, x0 = world x of row 0, y from
        -spacing/2 to +spacing/2 across the lane. Bands share their boundary row (0 from the taper on both sides)."""
        if self._grids is None:
            from spot_heightfield_env import make_hf_grids
            nx = int(round(ROUGH_BAND / ROUGH_CELL)) + 1
            ny = int(round(self.spacing / ROUGH_CELL)) + 1
            self._grids = make_hf_grids(num_shapes=ROUGH_SHAPES, seed=0, x0=0.0, x1=ROUGH_BAND, nx=nx, ny=ny,
                                        half_w=0.5 * self.spacing, taper=ROUGH_TAPER)[0]
        H = self._grids
        per = H.shape[1] - 1
        lv = list(self._levels(i))
        z = np.zeros((per * len(lv) + 1, H.shape[2]), np.float32)
        for b, j in enumerate(lv):
            z[b * per: b * per + per + 1] = np.maximum(z[b * per: b * per + per + 1],
                                                       H[int(self.rough_shape[i, j])] * self.rough_amps[j])
        return lv[0] * ROUGH_BAND, z

    def _add_rough(self, world, i, y, bvh_sink):
        x0, z = self.rough_grid(i)
        nx, ny = z.shape
        half = 0.5 * self.spacing
        mesh = grid_mesh(z, ROUGH_CELL, (x0, -half))
        mesh.position.set(0.0, float(y), 0.0)             # lane-local vertices: float32 stays precise at y ~ 6 km
        if self.rough_backend == "trimesh":
            world.add_static_trimesh(mesh)
        else:
            # rows are y, origin at the (x0, y - half) corner (bind_physx add_static_heightfield); the BVH gets a copy
            w = world.world if hasattr(world, "world") else world
            w.add_static_heightfield(np.ascontiguousarray(z.T), ROUGH_CELL, tp.Vector3(float(x0), float(y - half), 0.0))
            if bvh_sink is not None:
                bvh_sink.append(mesh)
        self.n_rough_colliders += 1
        self.n_rough_tris += 2 * (nx - 1) * (ny - 1)


def grid_mesh(z, cell, origin):
    """A Mesh over a height grid z [nx, ny] at (origin[0] + ix*cell, origin[1] + iy*cell, z), indexed, two triangles per
    cell split along (x0,y0)-(x1,y1), wound counter-clockwise seen from +z (the heightfield env's soup order)."""
    nx, ny = z.shape
    gx, gy = np.meshgrid(origin[0] + cell * np.arange(nx), origin[1] + cell * np.arange(ny), indexing="ij")
    verts = np.stack([gx, gy, z], axis=-1).reshape(-1, 3).astype(np.float32)
    v = np.arange(nx * ny, dtype=np.uint32).reshape(nx, ny)
    v00, v10, v11, v01 = v[:-1, :-1], v[1:, :-1], v[1:, 1:], v[:-1, 1:]
    faces = np.stack([np.stack([v00, v10, v11], -1), np.stack([v00, v11, v01], -1)], axis=2).reshape(-1, 3)
    g = tp.BufferGeometry()
    g.set_attribute("position", verts)
    g.set_index(np.ascontiguousarray(faces.reshape(-1)))
    return tp.Mesh(g, tp.MeshStandardMaterial())


def grid_height_np(z, cell, origin, x, y):
    """Closed form of grid_mesh's surface at world-local (x, y) (numpy): the validation reference for the raycast."""
    fx = (np.asarray(x, np.float64) - origin[0]) / cell
    fy = (np.asarray(y, np.float64) - origin[1]) / cell
    ix = np.clip(np.floor(fx).astype(np.int64), 0, z.shape[0] - 2)
    iy = np.clip(np.floor(fy).astype(np.int64), 0, z.shape[1] - 2)
    u, w = fx - ix, fy - iy
    h00, h10, h11, h01 = z[ix, iy], z[ix + 1, iy], z[ix + 1, iy + 1], z[ix, iy + 1]
    lower = u >= w                                      # triangle (v00, v10, v11)
    return np.where(lower, h00 + u * (h10 - h00) + w * (h11 - h10), h00 + w * (h01 - h00) + u * (h11 - h01))


def level_label(code, level):
    if code == STAIRS or code not in LEVEL_VALUE:
        return str(level)
    v = LEVEL_VALUE[code][int(level)]
    return f"{v:g} {LEVEL_UNIT[code]}"
