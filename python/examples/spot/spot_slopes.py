"""Slope terrain for Spot: planar up-ramps, down-ramps and cross-slopes built from ROTATED STATIC BOXES.

No Spot env has a slope, and "no falls on 0-20 deg slopes and cross-slopes" is the first target
(plans/spot-frontier.md, Target). This module is the terrain half of that measurement and nothing else:
lane builders, the closed-form surface of every lane (height and normal), a slope-aware spawn and a
settle-health check. SpotStepsEnv(lane_family=, slope_deg=, slope_material=) and score_e0.py --suite showpiece wire
it as the INTEGRATION notes below describe (line numbers there are as of the implementation, before the wiring).

Lane i runs along +x at y = i * spacing (the SpotStepsEnv layout); u = x - x0 is the distance along it.

    up     the shared ground is the flat approach -> a planar ramp at theta from u = RAMP_START, RAMP_LEN
           measured ALONG the surface (16 s at 1.2 m/s is 19.2 m) -> a plateau PLATEAU_LEN long
    down   a raised plateau from u = -PLATEAU_BACK (the spawn is on it) -> a planar ramp down, RAMP_LEN
           along the surface -> the shared ground as the flat run-out
    cross  the whole lane tilted about x: walking +x is level, gravity pulls across the lane

Why rotated boxes: add_static builds a PxBoxGeometry from the BoxGeometry's width/height/depth and takes
the pose from the mesh's matrixWorld (PhysxWorld.hpp:457-459, :1186-1205), so a rotated box IS a planar
collider; it takes a material from Python (bind_physx.cpp:826-832); and CollectedWorld hands the same
Mesh to the raycast BVH (raycast.py:125-128), so the scan reads exactly what PhysX collides with.

Every ramp is ONE slab, thick enough that its underside is below the ground everywhere under its top
face: the ramp is a solid wedge, not a roof a drifting robot could walk under. Plateaus are solid boxes.

Neighbours. Lanes of one family and angle tile with no step (every box is exactly one lane wide). Cross
lanes alternate the tilt ('corrugate', the default): neighbouring edges meet at 0 and at 3*tan(theta), so a
run of cross lanes is one continuous zig-zag. Even lanes rise with y (gravity pulls toward -y), odd lanes
fall (pulls toward +y); `downhill` says which. 'offset' keeps every lane rising with y and lifts each lane
of a contiguous run by 3*tan(theta): one continuous tilted plane, every lane pulling toward -y, but its
height grows with the run. Where a slope lane meets a DIFFERENT lane there is a step: up to
RAMP_LEN*sin(theta) beside an up plateau or a down spawn plateau (8.5 m at 25 deg), up to 3*tan(theta)
beside a cross lane. So put an unscored buffer lane of the SAME family and angle on each side of a slope
block; the drop then sits against a lower lane, which only sees a wall.

A 0 deg lane builds nothing: it is the shared ground, the flat reference cell, never a coplanar second
collider on top of the ground box.

INTEGRATION (what a SpotStepsEnv / score_e0 integrator wires; nothing here does it):
  1. Lane family flag: SpotStepsEnv(slope_family=None, slope_deg=None, slope_material=None), [K] arrays;
     None builds exactly today's world. Slope lanes are lane_types False (no tents, `_add_steps` skips
     them) and count under the 'flat' lane type with their own block id. In `_build`, after `_add_steps`:
     `self.slopes = SlopeLanes(fam, deg, spacing=SPACING); self.slopes.build(world, slope_material)`.
  2. Height source: slope lanes REQUIRE height_source='raycast' (assert it). The analytic branch
     (spot_steps_env.py:473-484) returns 0 off the tents and cannot represent a plane. Keep the raycast;
     `SlopeLanes.heights` is the closed form for validation and for counters, not a replacement.
  3. Ground: `SlopeLanes.ground_x_extent()` is the x-range the shared ground box must cover (down-ramp
     run-outs, up-ramp approaches). The legacy 60 m box (x -10..50) covers every default lane.
  4. Spawn hook, in on_reset (spot_steps_env.py:550-558), for slope rows only (leave the default rows and
     the default path untouched): `sp = spawn_pose(self.rays.heights, sx, self.lane_y[idx])` with
     sx = x0 + SPAWN_U; pose[:, 0:4] = sp.quat; pose[:, 4:7] = sp.pos (PhysX layout, sim.py:132-139); and
     set_joint_state(idx, sp.joint_q[:, self.a2i], 0) INSTEAD of stand_q_add (add order on the GPU,
     sim.py:213-221; the CPU set_joint_positions takes the Isaac = DOF-cache order, Articulation.hpp:364-374).
     On a 0 deg row sp.joint_q is within 0.046 rad of default_q (the IK puts the front and hind preset
     tips, 2 cm apart in height, at one clearance); it grows to 0.41-0.53 rad at 20 deg and 0.59-0.73 at 25,
     which is what a policy is handed at the start of a slope episode.
  5. Settle: hold the SPAWN joints, not default_q. VecTask settles only on a FULL reset, feeding zero
     actions for settle_steps = 20 (task.py:314-322); slope rows must be fed `settle_action(sp.joint_q)`
     instead (keep it per env; act() turns it into exactly sp.joint_q). Partial resets have no settle at
     all (task.py:282-286), so for them the joint preset is the whole story. Evaluate `settle_health` in
     on_settled and count failures per block (e.g. a `spawn_bad` stat); refuse to score a cell above 2%.
     GPU tips: foot_contact.foot_world (links 9..12, BFS order); CPU tips: `cpu_foot_tips` (links
     3/6/9/12, add order). Same local offset TIP_OFFSET; mixing the two orders reads the hips.
     Why not default_q (measured 2026-09-12, 20 settles per cell): a body aligned with the slope and
     default joints shifts the support polygon uphill by ~0.4*tan(theta) (12 cm at 15 deg), the downhill
     legs take the load and sag at kp 90, and the stance folds downhill: 12-20 of 20 settles fail from
     10 deg (up, cross) and 15 deg (down), flag on or off, grippy or not. kp 300 holds; so does the
     gravity-referenced footprint below, with no plant change.
  6. Block ids: `SLOPE_BLOCKS` below, in score_e0's _blk shape plus 'slope'/'deg'/'material'.
  7. The imitation gate (change > 0.10, spot_steps_env.py:607-609) is zero on any slope above ~5 deg and
     the roll penalty is world-frame (:643); both are training-side decisions, not terrain.
"""
import collections
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from spot_terrain_env import SPACING    # NOT spot_steps_env: that module is the one that will import this
from spot_deploy import HIP_X, HIP_Y, HY_Y, KN, LIM, ACTION_SCALE, default_q

# --------------------------------------------------------------------------- #
#  Lane families and geometry (lane-local u = x - x0)
# --------------------------------------------------------------------------- #
FLAT, UP, DOWN, CROSS = 0, 1, 2, 3
FAMILIES = {"flat": FLAT, "up": UP, "down": DOWN, "cross": CROSS}
SLOPE_DEGS = (5.0, 10.0, 15.0, 20.0, 25.0)

RAMP_START = 1.6         # = spot_steps_env.FLAT_APPROACH: the ramp starts where a tent would
RAMP_LEN = 20.0          # along the surface; 16 s (STEPS_EPISODE_S) at 1.2 m/s is 19.2 m
PLATEAU_LEN = 6.0        # up: past the ramp top, so 16 s at VX_HI 1.5 from the spawn stays on it at any angle
PLATEAU_BACK = 3.0       # down: plateau behind the spawn
RUNOUT = 6.0             # down: flat run-out past the ramp foot, on the shared ground
CROSS_BACK = 3.0         # cross: lane surface behind the spawn ...
CROSS_LEN = 27.0         # ... and ahead of the lane origin (0.7 + 16 s * 1.5 m/s = 24.7 m)
SPAWN_U = 0.7            # = spot_steps_env.SPAWN_OFF: flat approach (up), plateau (down), on the slope (cross)
SLAB_MARGIN = 0.5        # extra slab thickness below the deepest point it must fill
BOX_BOTTOM = -0.5        # plateaus and supports reach down into the ground box (top z=0, 1 m thick)
GRIPPY = (1.0, 1.0, 0.0, "min", "min")   # the explicit terrain material, as an ADDED cell. PhysX uses the
# higher-ranked combine mode of the pair ('min' outranks 'average'), so against DEFAULT feet this still
# yields min(1.0, 0.5): the grippy cell needs foot_mu 1.0 'min' too (score_e0 foot_mu) to change friction.

# --------------------------------------------------------------------------- #
#  Stance (body frame, fl/fr/hl/hr) — MEASURED, not FOOT_DX/FOOT_DY (spot_steps_env.py:81-82)
# --------------------------------------------------------------------------- #
FOOT_R = 0.028                   # lower-leg capsule radius (spot_deploy.py:155)
TIP_OFFSET = (0.0, 0.15, 0.0)    # lleg-local capsule end nearest the ground at stance (foot_contact.py:59)
CPU_FOOT_LINKS = (3, 6, 9, 12)   # CPU articulation, add order (bind_physx.cpp:418-422); GPU is 9..12 (BFS)
# Footprint the terrain plane is fitted through: the settled stance tips from the design review
# (dx +0.279 / -0.335, dy +-0.205). On a plane the fit does not depend on the footprint at
# all; it only matters where the four points straddle a crease. NOTE a CPU stance probe of 2026-09-12
# (flat, TGS 0.005, 1 s settle) settled the hind tips at dx -0.39..-0.41 and
# dy +-0.21..0.22 depending on the spawn height, front +0.278..0.291: the hind number does not reproduce.
STANCE_FIT_DX = (0.279, 0.279, -0.335, -0.335)
STANCE_FIT_DY = (0.205, -0.205, 0.205, -0.205)
# Where the tips ARE at spawn: joints preset to default_q, before the first step (same probe). The
# front tips sit 2 cm lower than the hind ones. Settled, the base is 0.459 m up and the tips at
# (+0.278, +-0.213, -0.435) / (-0.412, +-0.211, -0.427): the legs sag 3.5 cm under load.
STANCE_PRESET = ((0.2434, 0.2134, -0.4691), (0.2434, -0.2134, -0.4691),
                 (-0.4471, 0.2114, -0.4488), (-0.4471, -0.2114, -0.4488))
SPAWN_CLEARANCE = 0.005   # lowest capsule surface this far above the terrain along the normal. On flat
                          # ground a +3 mm start was the gentlest settle of four spawn heights (base z
                          # 0.440..0.498), against 0.414..0.528 for the training rule's 4.7 cm penetration.
SPAWN_BODY_TILT = 0.5     # spawn_pose: body tilted by this fraction of the local slope (0 level, 1 aligned).
                          # Measured, 20 spawns per cell, up/down/cross 0-25 deg: 0 and 0.5 both settle with
                          # zero failures; at 0 the downhill legs run out of reach at 25 deg (21 legs
                          # clamped, feet up to 35 mm high), at 0.5 none do. 1.0 (IK joints) fails down 25 and
                          # cross 25 and dips at down 20 (17 of 20 at 0.2 s).
CONTACT_TOL = 0.02        # a settled foot is in contact when its capsule surface is within this of the terrain
SETTLE_UP_FRAC = 0.98     # settle health: up_z >= SETTLE_UP_FRAC * cos(theta)
SPAWN_FAIL_MAX = 0.02     # a cell whose spawns fail settle health (or IK) more often than this is flagged, not scored
OFF_SURFACE_MAX = 0.02    # ... and so is one whose robots spend more than this share of their ticks over another surface
FD_STEP = 0.02            # finite-difference step for a normal off a height function


def _plane_fit_pinv():
    a = np.column_stack([np.ones(4), STANCE_FIT_DX, STANCE_FIT_DY])
    return np.linalg.pinv(a)                                           # [3,4]: h -> (h0, dh/dfwd, dh/dleft)


def _tip_level():
    """Body-frame rotation that takes the plane through the four PRESET tips onto body z, so a spawn
    aligned with it puts all four feet at the same height above a plane (the hind tips would otherwise
    start 2 cm higher than the front ones). A 1.7 deg pitch, nose up."""
    p = np.asarray(STANCE_PRESET, np.float64)
    cf = np.linalg.lstsq(np.column_stack([np.ones(4), p[:, 0], p[:, 1]]), p[:, 2], rcond=None)[0]
    m = np.array([-cf[1], -cf[2], 1.0]); m /= np.linalg.norm(m)
    v = np.cross(m, [0.0, 0.0, 1.0]); s2 = float(v @ v)
    if s2 < 1e-18:
        return np.eye(3)
    k = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + k + k @ k * ((1.0 - m[2]) / s2)


_FIT_PINV = _plane_fit_pinv()
_TIP_LEVEL = _tip_level()


# --------------------------------------------------------------------------- #
#  Lanes
# --------------------------------------------------------------------------- #
class SlopeLanes:
    """Per-lane slope geometry for K lanes: builds the boxes, and answers height / normal in closed form.

    family  [K] of 'flat' | 'up' | 'down' | 'cross' (or the FLAT/UP/DOWN/CROSS codes); flat builds nothing
    deg     scalar or [K] slope angle in degrees (0 = the shared ground, whatever the family)
    """

    def __init__(self, family, deg, spacing=SPACING, x0=0.0, cross_mode="corrugate", ramp_start=RAMP_START,
                 ramp_len=RAMP_LEN, plateau_len=PLATEAU_LEN, plateau_back=PLATEAU_BACK, runout=RUNOUT,
                 cross_back=CROSS_BACK, cross_len=CROSS_LEN):
        fam = np.array([FAMILIES[f] if isinstance(f, str) else int(f) for f in family], np.int64)
        k = fam.shape[0]
        d = np.asarray(deg, np.float64)
        d = np.full(k, float(d)) if d.ndim == 0 else d.reshape(-1).copy()
        if d.shape[0] != k:
            raise ValueError(f"deg: expected a scalar or {k} values, got {d.shape[0]}")
        if (d < 0.0).any() or (d >= 45.0).any():
            raise ValueError("deg must lie in [0, 45)")
        if cross_mode not in ("corrugate", "offset"):
            raise ValueError(f"cross_mode must be 'corrugate' or 'offset', got {cross_mode!r}")
        self.k, self.spacing, self.x0, self.cross_mode = k, float(spacing), float(x0), cross_mode
        self.ramp_start, self.ramp_len, self.plateau_len = float(ramp_start), float(ramp_len), float(plateau_len)
        self.plateau_back, self.runout = float(plateau_back), float(runout)
        self.cross_back, self.cross_len = float(cross_back), float(cross_len)
        self.family = fam.copy()                    # as asked, for reporting
        self.deg = d
        self._fam = np.where(d > 0.0, fam, FLAT)    # what is built and evaluated: a 0 deg lane is the ground
        th = np.radians(d)
        self._tan, self._sin, self._cos = np.tan(th), np.sin(th), np.cos(th)
        cross = self._fam == CROSS
        half = 0.5 * self.spacing
        if cross_mode == "corrugate":
            self._sign = np.where(np.arange(k) % 2 == 0, 1.0, -1.0)
            self._zc = half * self._tan                                     # edges at 0 and 2*half*tan
        else:
            self._sign = np.ones(k)
            run = np.zeros(k)                                               # index within a contiguous run
            for i in range(1, k):                                           # of cross lanes at one angle
                if cross[i] and cross[i - 1] and d[i] == d[i - 1]:
                    run[i] = run[i - 1] + 1.0
            self._zc = (half + run * self.spacing) * self._tan
        self._sign = np.where(cross, self._sign, 1.0)
        self._zc = np.where(cross, self._zc, 0.0)
        self._materials = {}                        # keep every created material alive with the lanes
        self._cache = {}

    # ---- closed form ---------------------------------------------------------------
    def _params(self, device, dtype):
        key = (str(device), dtype)
        if key not in self._cache:
            f = lambda a: torch.as_tensor(a, dtype=dtype, device=device)
            self._cache[key] = (torch.as_tensor(self._fam, dtype=torch.long, device=device),
                                f(self._tan), f(self._sin), f(self._cos), f(self._zc), f(self._sign))
        return self._cache[key]

    def _lookup(self, x, y):
        """Lane parameters for the lane each point lies in (by its y, not by the row it came from: a robot
        that drifted into its neighbour's lane is answered with the neighbour's surface)."""
        y = y.to(x.dtype)
        p = self._params(x.device, x.dtype)
        jf = torch.round(y / self.spacing)
        valid = (jf >= 0.0) & (jf <= float(self.k - 1))
        j = jf.clamp(0.0, float(self.k - 1)).long()
        return [t[j] for t in p], valid, x - self.x0, y - j.to(x.dtype) * self.spacing

    def _surface(self, fam, t, s, c, zc, sg, u, dy):
        zero = torch.zeros_like(u)
        run = self.ramp_len * c                                              # horizontal ramp length
        rise = self.ramp_len * s
        a = u - self.ramp_start
        h_up = torch.where(a < run + self.plateau_len, torch.minimum(a.clamp(min=0.0), run) * t, zero)
        h_dn = torch.where(u < -self.plateau_back, zero,
                           torch.where(a < 0.0, rise, torch.where(a < run, rise - a * t, zero)))
        h_x = torch.where((u >= -self.cross_back) & (u < self.cross_len), zc + sg * dy * t, zero)
        return torch.where(fam == UP, h_up, torch.where(fam == DOWN, h_dn, torch.where(fam == CROSS, h_x, zero)))

    def heights(self, x, y):
        """Terrain height at world (x, y), any matching shape, torch (the device and dtype of x). Points
        outside every lane's authored surface read 0, the shared ground. No host sync, no boolean
        indexing: usable inside a captured step."""
        g, valid, u, dy = self._lookup(x, y)
        h = self._surface(*g, u, dy)
        return torch.where(valid, h, torch.zeros_like(h))

    def normals(self, x, y):
        """Unit surface normal at world (x, y) -> [..., 3] (flat where the lane is flat)."""
        (fam, t, s, c, zc, sg), valid, u, dy = self._lookup(x, y)
        zero, one = torch.zeros_like(u), torch.ones_like(u)
        a = u - self.ramp_start
        run = self.ramp_len * c
        on_up = valid & (fam == UP) & (a > 0.0) & (a < run)
        on_dn = valid & (fam == DOWN) & (a > 0.0) & (a < run)
        on_x = valid & (fam == CROSS) & (u >= -self.cross_back) & (u < self.cross_len)
        nx = torch.where(on_up, -s, torch.where(on_dn, s, zero))
        ny = torch.where(on_x, -sg * s, zero)
        nz = torch.where(on_up | on_dn | on_x, c, one)
        return torch.stack([nx, ny, nz], dim=-1)

    def heights_np(self, x, y):
        return self.heights(torch.as_tensor(np.asarray(x, np.float64)),
                            torch.as_tensor(np.asarray(y, np.float64))).numpy()

    def normals_np(self, x, y):
        return self.normals(torch.as_tensor(np.asarray(x, np.float64)),
                            torch.as_tensor(np.asarray(y, np.float64))).numpy()

    def downhill(self, i):
        """Unit horizontal downhill direction (dx, dy) on lane i's slope face, for drift counters."""
        f = int(self._fam[i])
        if f == UP:
            return (-1.0, 0.0)
        if f == DOWN:
            return (1.0, 0.0)
        if f == CROSS:
            return (0.0, -float(self._sign[i]))
        return (0.0, 0.0)

    def lane_x_extent(self, i):
        """World x-range over which lane i's surface is authored (ground-supported parts included), or None."""
        f = int(self._fam[i])
        run = self.ramp_len * float(self._cos[i])
        if f == UP:
            return (self.x0 + SPAWN_U - 2.0, self.x0 + self.ramp_start + run + self.plateau_len)
        if f == DOWN:
            return (self.x0 - self.plateau_back, self.x0 + self.ramp_start + run + self.runout)
        if f == CROSS:
            return (self.x0 - self.cross_back, self.x0 + self.cross_len)
        return None

    def ground_x_extent(self):
        """The x-range the SHARED ground box must cover for these lanes (up approaches, down run-outs), or
        None when no lane leans on the ground."""
        lo, hi = [], []
        for i in range(self.k):
            f = int(self._fam[i])
            run = self.ramp_len * float(self._cos[i])
            if f == UP:
                lo.append(self.x0 + SPAWN_U - 2.0); hi.append(self.x0 + self.ramp_start)
            elif f == DOWN:
                lo.append(self.x0 + self.ramp_start + run); hi.append(self.x0 + self.ramp_start + run + self.runout)
        return (min(lo), max(hi)) if lo else None

    def edge_steps(self, n=4001):
        """[K-1] largest height step where lane i meets lane i+1, over the union of their x-extents, each
        lane evaluated with its OWN formula at the shared edge. 0 means the lanes meet without a cliff."""
        ext = [e for e in (self.lane_x_extent(i) for i in range(self.k)) if e is not None]
        if not ext or self.k < 2:
            return np.zeros(max(self.k - 1, 0))
        u = torch.linspace(min(e[0] for e in ext), max(e[1] for e in ext), n, dtype=torch.float64) - self.x0
        p = self._params("cpu", torch.float64)
        out = np.zeros(self.k - 1)
        for i in range(self.k - 1):
            gi = [t[i].expand(n) for t in p]
            gj = [t[i + 1].expand(n) for t in p]
            hi = self._surface(*gi, u, torch.full((n,), 0.5 * self.spacing, dtype=torch.float64))
            hj = self._surface(*gj, u, torch.full((n,), -0.5 * self.spacing, dtype=torch.float64))
            out[i] = float((hi - hj).abs().max())
        return out

    def surface_ids(self):
        """[K] int64: lanes with one id are one continuous walkable surface across their shared edges; a lane with
        another id meets them at a step or a crease. The shared ground (flat, stair and 0 deg lanes) is 0. Up and
        down lanes of one angle share an id; cross lanes of one angle share one in 'offset' mode (one plane per
        contiguous run) and split by tilt direction in 'corrugate' mode (the two faces of the zig-zag). Two separate
        runs at one angle also share an id, but only a lane of another id lies between them."""
        tenth = np.round(self.deg * 10.0).astype(np.int64)
        face = ((self._fam == CROSS) & (self._sign < 0)).astype(np.int64)
        return np.where(self._fam == FLAT, 0, self._fam * 100000 + tenth * 2 + face).astype(np.int64)

    # ---- builder -------------------------------------------------------------------
    def _lane_material(self, world, i, material):
        spec = material[i] if isinstance(material, (list, np.ndarray)) else material
        if spec is None:
            return None
        if isinstance(spec, tuple):                  # (static, dynamic, restitution, f_combine, r_combine)
            if spec not in self._materials:
                self._materials[spec] = world.create_material(*spec)
            return self._materials[spec]
        return spec                                   # already a world.create_material object

    @staticmethod
    def _box(world, size, center, axis, angle, mat):
        m = tp.Mesh(tp.BoxGeometry(*(float(v) for v in size)), tp.MeshStandardMaterial())
        m.position.set(*(float(v) for v in center))
        if axis == "x":
            m.rotation.x = float(angle)
        elif axis == "y":
            m.rotation.y = float(angle)
        if mat is None:
            world.add_static(m)                       # the exact call the default terrain makes
        else:
            world.add_static(m, mat)
        return m

    def build(self, world, material=None):
        """Add every slope lane's boxes to `world` (a PhysxWorld or a CollectedWorld) and return the meshes.

        material: None (the world's default contact, as every existing terrain box), a 5-tuple for
        world.create_material, a material object, or a list/array with one of those per lane. An
        explicit material is an ADDED cell, never the new default. The up approach and the down run-out
        are the shared ground and keep ITS material."""
        meshes = []
        for i in range(self.k):
            f = int(self._fam[i])
            if f == FLAT:
                continue
            mat = self._lane_material(world, i, material)
            th = math.radians(float(self.deg[i]))
            s, c, t = math.sin(th), math.cos(th), math.tan(th)
            L, W, y = self.ramp_len, self.spacing, i * self.spacing
            run, rise = L * c, L * s
            xr = self.x0 + self.ramp_start
            if f in (UP, DOWN):
                # Slab thickness so its underside passes below the ground under the whole top face:
                # the vertical drop through the slab is T/cos, and it must exceed the rise at the top.
                T = L * s * c + SLAB_MARGIN
                if f == UP:
                    top, n = (xr + 0.5 * run, y, 0.5 * rise), (-s, 0.0, c)      # R_y(-th): local x up the slope
                    meshes.append(self._box(world, (L, W, T), [top[k] - n[k] * 0.5 * T for k in range(3)],
                                            "y", -th, mat))
                    meshes.append(self._box(world, (self.plateau_len, W, rise - BOX_BOTTOM),
                                            (xr + run + 0.5 * self.plateau_len, y, 0.5 * (rise + BOX_BOTTOM)),
                                            None, 0.0, mat))
                else:
                    back = self.ramp_start + self.plateau_back
                    meshes.append(self._box(world, (back, W, rise - BOX_BOTTOM),
                                            (xr - 0.5 * back, y, 0.5 * (rise + BOX_BOTTOM)), None, 0.0, mat))
                    top, n = (xr + 0.5 * run, y, 0.5 * rise), (s, 0.0, c)       # R_y(+th): local x down the slope
                    meshes.append(self._box(world, (L, W, T), [top[k] - n[k] * 0.5 * T for k in range(3)],
                                            "y", th, mat))
            else:
                sg, zc = float(self._sign[i]), float(self._zc[i])
                T = W * s + SLAB_MARGIN                                         # fills the lane's own height span
                length = self.cross_back + self.cross_len
                top, n = (self.x0 + 0.5 * (self.cross_len - self.cross_back), y, zc), (0.0, -sg * s, c)
                meshes.append(self._box(world, (length, W / c, T), [top[k] - n[k] * 0.5 * T for k in range(3)],
                                        "x", sg * th, mat))                    # R_x(+-th): local y across the tilt
                zmin = zc - 0.5 * W * t
                if zmin > 1e-6:                                                 # 'offset' lanes stand on a support
                    meshes.append(self._box(world, (length, W, zmin - BOX_BOTTOM),
                                            (top[0], y, 0.5 * (zmin + BOX_BOTTOM)), None, 0.0, mat))
        return meshes


# --------------------------------------------------------------------------- #
#  Spawn and settle health (torch, batched; CPU or GPU, eager)
# --------------------------------------------------------------------------- #
def quat_from_matrix(R):
    """[n,3,3] rotation matrices (columns = body axes in world) -> [n,4] quaternions (qx,qy,qz,qw), w >= 0."""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    sq = torch.stack([1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
                      1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=1)    # 4 w^2, 4 x^2, 4 y^2, 4 z^2
    r = 0.5 * sq.clamp(min=1e-12).sqrt()                                        # 2|w|, 2|x|, 2|y|, 2|z| / 2
    cand = torch.stack([
        torch.stack([(m21 - m12), (m02 - m20), (m10 - m01), sq[:, 0]], 1) / (4.0 * r[:, 0:1]),
        torch.stack([sq[:, 1], (m01 + m10), (m02 + m20), (m21 - m12)], 1) / (4.0 * r[:, 1:2]),
        torch.stack([(m01 + m10), sq[:, 2], (m12 + m21), (m02 - m20)], 1) / (4.0 * r[:, 2:3]),
        torch.stack([(m02 + m20), (m12 + m21), sq[:, 3], (m10 - m01)], 1) / (4.0 * r[:, 3:4])], dim=1)
    q = cand.gather(1, sq.argmax(dim=1).view(-1, 1, 1).expand(-1, 1, 4)).squeeze(1)
    q = q / q.norm(dim=1, keepdim=True)
    return torch.where(q[:, 3:4] < 0.0, -q, q)


Spawn = collections.namedtuple("Spawn", "pos quat joint_q gap reachable normal")
Spawn.__doc__ = """pos [n,3] base position; quat [n,4] (qx,qy,qz,qw); joint_q [n,12] joint angles in Isaac order
(= the CPU set_joint_positions DOF-cache order; the GPU add order is joint_q[:, add_to_isaac]); gap [n,4]
each foot's capsule-surface clearance at spawn; reachable [n,4] False where IK hit a joint limit (the foot
then starts higher than `clearance`); normal [n,3] the fitted terrain normal."""

# --------------------------------------------------------------------------- #
#  Leg kinematics in the body frame, matched to PhysX: FK(default_q) = STANCE_PRESET within 0.03 mm, and
#  FK of every IK spawn reproduces the PhysX tips (checked 2026-09-12 against settled PhysX stances)
# --------------------------------------------------------------------------- #
LEG_SIGN = ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0))   # (front, left) for fl, fr, hl, hr
SHIN_TIP = 0.32     # knee joint to the capsule-end centre: half the 0.34 shin + TIP_OFFSET's 0.15


def _ry(x, z, a):
    return x * torch.cos(a) + z * torch.sin(a), -x * torch.sin(a) + z * torch.cos(a)


def leg_fk(q):
    """Foot tips [..., 4, 3] in the body frame from joint angles q [..., 4, 3] = (hx, hy, kn) per leg,
    fl/fr/hl/hr. From Isaac order [n, 12]: q.view(n, 3, 4).transpose(1, 2)."""
    sx = q.new_tensor([v[0] for v in LEG_SIGN]); sy = q.new_tensor([v[1] for v in LEG_SIGN])
    hx, hy, kn = q[..., 0], q[..., 1], q[..., 2]
    tx, tz = _ry(torch.full_like(hy, float(KN[0])), torch.full_like(hy, float(KN[2])), hy)
    kx, kz = _ry(torch.zeros_like(hy), torch.full_like(hy, -SHIN_TIP), hy + kn)
    xl, zl = tx + kx, tz + kz
    y = sy * HY_Y * torch.cos(hx) - zl * torch.sin(hx)
    z = sy * HY_Y * torch.sin(hx) + zl * torch.cos(hx)
    return torch.stack([sx * HIP_X + xl, sy * HIP_Y + y, z], dim=-1)


def leg_ik(p):
    """Closed-form IK: body-frame tips p [..., 4, 3] -> (q [..., 4, 3], reachable [..., 4]). Knee-back branch
    (Spot's), clamped to the URDF joint limits; reachable is False where a clamp was needed."""
    sx = p.new_tensor([v[0] for v in LEG_SIGN]); sy = p.new_tensor([v[1] for v in LEG_SIGN])
    px, py, pz = p[..., 0] - sx * HIP_X, p[..., 1] - sy * HIP_Y, p[..., 2]
    zl = -torch.sqrt((py * py + pz * pz - HY_Y * HY_Y).clamp(min=1e-9))   # leg length in its own sagittal plane
    hx = torch.remainder(torch.atan2(pz, py) - torch.atan2(zl, sy * HY_Y) + math.pi, 2.0 * math.pi) - math.pi
    t0, t2 = float(KN[0]), float(KN[2])
    a, b = -t2 * SHIN_TIP, -t0 * SHIN_TIP                                  # thigh . R(kn) shin = a cos kn + b sin kn
    c = (px * px + zl * zl - (t0 * t0 + t2 * t2) - SHIN_TIP * SHIN_TIP) / (2.0 * math.hypot(a, b))
    ok = (c >= -1.0) & (c <= 1.0)
    kn = math.atan2(b, a) - torch.acos(c.clamp(-1.0, 1.0))
    v_x, v_z = t0 - SHIN_TIP * torch.sin(kn), t2 - SHIN_TIP * torch.cos(kn)
    hy = torch.remainder(torch.atan2(v_z, v_x) - torch.atan2(zl, px) + math.pi, 2.0 * math.pi) - math.pi
    q = torch.stack([hx.clamp(*LIM["hx"]), hy.clamp(*LIM["hy"]), kn.clamp(*LIM["kn"])], dim=-1)
    ok = ok & (q[..., 0] == hx) & (q[..., 1] == hy) & (q[..., 2] == kn)
    return q, ok


def _fit_normal(height_fn, x, y, cy, sy):
    """Normal of the plane fitted through the terrain under the settled footprint (STANCE_FIT_*)."""
    dev, dt = x.device, x.dtype
    fdx = torch.tensor(STANCE_FIT_DX, device=dev, dtype=dt)
    fdy = torch.tensor(STANCE_FIT_DY, device=dev, dtype=dt)
    px = x[:, None] + cy[:, None] * fdx - sy[:, None] * fdy
    py = y[:, None] + sy[:, None] * fdx + cy[:, None] * fdy
    coef = height_fn(px, py).to(dt) @ torch.as_tensor(_FIT_PINV.T, device=dev, dtype=dt)   # [n,3]
    gf, gl = coef[:, 1], coef[:, 2]                                   # slope along the heading / to its left
    n = torch.stack([-(gf * cy - gl * sy), -(gf * sy + gl * cy), torch.ones_like(gf)], dim=1)
    return n / n.norm(dim=1, keepdim=True)


def _frame(nb, cy, sy):
    """Body frame with z = nb whose x keeps its horizontal projection exactly on the heading (cy, sy)."""
    bx = torch.stack([cy, sy, -(cy * nb[:, 0] + sy * nb[:, 1]) / nb[:, 2]], dim=1)
    bx = bx / bx.norm(dim=1, keepdim=True)
    return torch.stack([bx, torch.linalg.cross(nb, bx, dim=1), nb], dim=2)


def spawn_pose(height_fn, x, y, yaw=None, body_tilt=SPAWN_BODY_TILT, clearance=SPAWN_CLEARANCE, normal_fn=None):
    """Slope-aware spawn for n robots at world (x, y) facing `yaw` (default +x) -> Spawn.

    height_fn(px, py) -> heights of the same shape (SlopeLanes.heights, TerrainRays.heights, anything);
    normal_fn(px, py) -> [..., 3] if known, else the fitted plane's normal is used for every foot.
    The feet go on the terrain under the PRESET footprint taken horizontally around the base (gravity-
    referenced, as on flat ground), each capsule `clearance` above the terrain along the normal; the body
    is tilted by `body_tilt` of the fitted slope with its heading kept; the base sits so the legs keep
    their mean preset length; the joints come from leg_ik. Hold sp.joint_q through the settle
    (settle_action). Eager only: three height_fn calls and small matmuls, no host sync."""
    dev, dt = x.device, x.dtype
    y = y.to(dt)
    yaw = torch.zeros_like(x) if yaw is None else torch.as_tensor(yaw, device=dev).to(dt)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    n = _fit_normal(height_fn, x, y, cy, sy)
    nb = (1.0 - body_tilt) * n.new_tensor([0.0, 0.0, 1.0]) + body_tilt * n
    R = _frame(nb / nb.norm(dim=1, keepdim=True), cy, sy)
    pre = torch.as_tensor(STANCE_PRESET, device=dev, dtype=dt)                # [4,3]
    fx = x[:, None] + cy[:, None] * pre[:, 0] - sy[:, None] * pre[:, 1]
    fy = y[:, None] + sy[:, None] * pre[:, 0] + cy[:, None] * pre[:, 1]
    nz = (lambda a, b: normal_fn(a, b)[..., 2].to(dt)) if normal_fn is not None else (lambda a, b: n[:, 2:3].expand_as(a))
    tz = height_fn(fx, fy).to(dt) + (FOOT_R + clearance) / nz(fx, fy)
    bz = tz.mean(dim=1) - pre[:, 2].mean()
    base = torch.stack([x, y, bz], dim=1)
    q, ok = leg_ik((torch.stack([fx, fy, tz], dim=-1) - base[:, None, :]) @ R)
    tw = base[:, None, :] + leg_fk(q) @ R.transpose(1, 2)                      # the tips of the (clamped) solution
    hh = height_fn(tw[..., 0], tw[..., 1]).to(dt)
    nzz = nz(tw[..., 0], tw[..., 1])
    lift = ((FOOT_R + clearance) / nzz - (tw[..., 2] - hh)).max(dim=1).values.clamp(min=0.0)
    gap = (tw[..., 2] + lift[:, None] - hh) * nzz - FOOT_R
    return Spawn(torch.stack([x, y, bz + lift], dim=1), quat_from_matrix(R), q.transpose(1, 2).reshape(-1, 12),
                 gap, ok, n)


def spawn_pose_aligned(height_fn, x, y, yaw=None, clearance=SPAWN_CLEARANCE, level_tips=True):
    """The literal spawn: body aligned with the fitted plane, joints at default_q, base lifted until the lowest
    preset capsule is `clearance` above the terrain. Exact geometry (every foot 5.0 mm on a plane), but its
    passive settle folds downhill from 10-15 deg (see INTEGRATION 5): kept as the reference, not for use.
    level_tips pitches the body 1.7 deg so the front and hind preset tips start at one height -> Spawn."""
    dev, dt = x.device, x.dtype
    y = y.to(dt)
    yaw = torch.zeros_like(x) if yaw is None else torch.as_tensor(yaw, device=dev).to(dt)
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    n = _fit_normal(height_fn, x, y, cy, sy)
    R = _frame(n, cy, sy)
    if level_tips:
        R = R @ torch.as_tensor(_TIP_LEVEL, device=dev, dtype=dt)
    rel = torch.as_tensor(STANCE_PRESET, device=dev, dtype=dt) @ R.transpose(1, 2)   # [n,4,3] tips from the base
    ht = height_fn(x[:, None] + rel[..., 0], y[:, None] + rel[..., 1]).to(dt)
    nz = n[:, 2:3]
    bz = ((FOOT_R + clearance) / nz - rel[..., 2] + ht).max(dim=1).values
    gap = (bz[:, None] + rel[..., 2] - ht) * nz - FOOT_R
    jq = torch.as_tensor(default_q, device=dev, dtype=dt).expand(x.shape[0], 12).clone()
    return Spawn(torch.stack([x, y, bz], dim=1), quat_from_matrix(R), jq, gap,
                 torch.ones(x.shape[0], 4, dtype=torch.bool, device=dev), n)


def settle_action(joint_q):
    """The policy-space action whose drive targets are exactly joint_q (Isaac order): what a settle loop feeds
    SpotStepsEnv.act() (targets = default_q + ACTION_SCALE * a) to hold a spawn."""
    return (joint_q - torch.as_tensor(default_q, device=joint_q.device, dtype=joint_q.dtype)) / ACTION_SCALE


def settle_health(up_z, tips, height_fn, cos_theta, normal_z=None, tol=CONTACT_TOL):
    """Spawn-settle health: up_z >= SETTLE_UP_FRAC * cos(theta) AND all four feet in contact.

    up_z [n] (spot_terrain_env.up_z of the root quat), tips [n,4,3] world foot tips, cos_theta scalar or [n]
    (the cell's slope angle). Contact is kinematic, as everywhere in this stack (no GRF reader): the
    capsule surface within `tol` of the terrain along the normal. normal_z [n,4] if known, else a
    finite difference of height_fn. Returns (ok [n], contact [n,4], gap [n,4])."""
    tx, ty, tz = tips[..., 0], tips[..., 1], tips[..., 2]
    h = height_fn(tx, ty).to(tz.dtype)
    if normal_z is None:
        gx = (height_fn(tx + FD_STEP, ty) - height_fn(tx - FD_STEP, ty)).to(tz.dtype) / (2.0 * FD_STEP)
        gy = (height_fn(tx, ty + FD_STEP) - height_fn(tx, ty - FD_STEP)).to(tz.dtype) / (2.0 * FD_STEP)
        normal_z = 1.0 / torch.sqrt(1.0 + gx * gx + gy * gy)
    gap = (tz - h) * normal_z - FOOT_R
    contact = gap < tol
    ok = (up_z >= SETTLE_UP_FRAC * cos_theta) & contact.all(dim=-1)
    return ok, contact, gap


def cpu_foot_tips(art):
    """World foot tips [4,3] (fl, fr, hl, hr) of a CPU PhysX Spot (spot_deploy.build_spot)."""
    links = art.links
    out = np.empty((4, 3))
    for f, li in enumerate(CPU_FOOT_LINKS):
        p = links[li].world_point(tp.Vector3(*TIP_OFFSET))
        out[f] = (p.x, p.y, p.z)
    return out


# --------------------------------------------------------------------------- #
#  Suggested scorer cells (score_e0 _blk shape + slope keys); not wired anywhere
# --------------------------------------------------------------------------- #
def _slope_blk(fam, deg, vx=0.8, material=None, mu=None):
    return {"stairs": False, "level": 0, "vx": vx, "push_dv": 0.0, "foot_mu": mu, "payload_kg": 0.0,
            "lanes": 1, "slope": fam, "deg": float(deg), "material": material}


SLOPE_BLOCKS = {f"{tag}{int(d):02d}": _slope_blk(fam, d)
                for tag, fam in (("U", "up"), ("D", "down"), ("X", "cross")) for d in SLOPE_DEGS}
SLOPE_BLOCKS.update({f"{tag}{int(d):02d}G": _slope_blk(fam, d, material=GRIPPY, mu=1.0)
                     for tag, fam in (("U", "up"), ("D", "down"), ("X", "cross")) for d in (20.0, 25.0)})
