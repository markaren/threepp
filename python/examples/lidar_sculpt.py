"""Sculpting against a sensor: an optimiser reshapes a body to hide from -- or shout at -- the engine's own LIDAR.

The companion piece to warp_hull_sculpt. There the loss was physics, written out
as integrals I could differentiate. Here there is no loss function of mine at
all. The objective IS `DepthSensor.scan_rgbd` on the Vulkan backend: a GPU ray
trace through the renderer's own acceleration structure, evaluating the LIDAR
equation in the closest-hit shader that ships with the engine
(`shaders/lidar.rchit`) --

    I = P_tx . f_back . cos(theta) . exp(-2 sigma r) / r^2 . (pi R_ref^2)

    f_back = albedo (1 - metal) / pi  +  F . D(N.H) / (4 cos^2 theta)

-- Cook-Torrance at the back-scatter geometry, where sensor = transmitter =
receiver so the half-vector is the surface normal. What comes back to Python is
a point cloud with a normalised intensity per return, and returns the engine's
own `detectorThreshold` could not lift above the noise floor never come back at
all. I count what survives. That count, over a fixed set of poses, is the whole
objective. No gradients exist and none are asked for: a CPU CMA-ES proposes
shapes, the GPU says how visible each one is.

Two runs of the same machinery, differing only in a sign:

  STEALTH (minimise) discovers FACETING. The specular lobe is the reason it can.
  D is a GGX distribution evaluated at N.H = cos(theta), so at roughness 0.62 a
  surface tilted ~25 degrees off the beam has already lost most of its return,
  and past ~45 degrees it falls under the detector threshold and the beam simply
  reports nothing. A sphere cannot escape this: from any direction some patch of
  it is square-on. A faceted body can, if no facet is normal to any pose in the
  set -- which is the F-117 argument, made by an optimiser that was never told
  about aircraft.

  CONSPICUOUS (maximise) discovers FRONTAL PANELLING: broad flats aimed at the
  poses, because a flat normal to the beam returns its whole area at cos(theta)
  = 1 instead of the single square-on point a sphere offers.

FIXED VOLUME. Every candidate is projected back to the starting enclosed volume
by scaling all radii by (V0/V)^(1/3) before it is ever scanned. Without it both
answers are the same cheese: stealth shrinks, conspicuity grows, and neither
tells you anything about shape. The constraint is exact, not a penalty term, and
the run prints the volume error alongside the score.

HONESTY, in order of how much they matter:

  * SINGLE BOUNCE. The tracer records the first return along a beam (glass and
    water aside). Corner-cube retroreflection -- three mutually perpendicular
    faces sending a beam back the way it came -- is a multi-bounce effect this
    sensor cannot see, so the conspicuous optimum here is panelling, not corner
    cubes. A real corner-reflector story is not what this demo shows.
  * STAR-SHAPED BY CONSTRUCTION. The parameters are RADII along fixed
    directions, so no candidate can fold back on itself. That bounds how
    concave the conspicuous answer is allowed to get, and it is why the stealth
    answer is a faceted convex-ish body rather than a cavity.
  * FACE NORMALS, NOT SHADING NORMALS. `lidar.rchit` derives its normal from
    the hit triangle's own vertices and deliberately ignores the interpolated
    `normal` attribute. The score therefore cannot be gamed by writing a
    flattering normal buffer; the `normal` attribute here exists only so the
    raster picture is shaded. That is the engine's choice, not mine, and it is
    what makes "the sensor is the loss" mean something.
  * ONE MATERIAL. Both answers are the same painted metal (roughness 0.62,
    metalness 1.0). Stealth by absorbing coatings is a different demo; this one
    is only allowed to move geometry.
  * THERE IS A FLOOR UNDER STEALTH, AND IT IS THE ENGINE'S. `lidar.rchit` says
    so in its own comments: "We omit Smith G". Without the shadowing-masking
    term the back-scatter specular is D.F / (4 cos^2 theta), and as cos theta
    goes to zero that denominator wins -- so intensity falls 12x from square-on
    to the trough near 55 degrees and then climbs back up, saturating at
    grazing. Measured on the baseline sphere at 9 m: 0.208 at 0 deg, 0.017 at
    55 deg, over 0.16 again past 80 deg. Every closed body therefore keeps a
    bright SILHOUETTE RIM that no amount of faceting can tilt away, because the
    rim is where the surface turns away from the beam by definition. The
    stealth answer removes the square-on cap and keeps the rim, and that rim is
    most of what it still returns. A real sensor model with Smith G would not
    hand it back; this one does, and the number below is honest about it.
  * THE POSE SET IS THE THREAT MODEL. 8 azimuths at 8 m and 4 at 12 m, all
    aimed at the body's centre. A shape optimised against this set owes nothing
    to a pose outside it, and `--sweep` is where that bill comes due.

    python lidar_sculpt.py                  # window: watch it sculpt, then the finale
    python lidar_sculpt.py --mode stealth   # one act only
    python lidar_sculpt.py --mode conspicuous
    python lidar_sculpt.py --sweep          # detection-range table, headless
    python lidar_sculpt.py --tune           # numbers only, no renderer window
    python lidar_sculpt.py --gens 200       # longer search
    python lidar_sculpt.py --shot out.png   # headless, finale straight to a PNG
    python lidar_sculpt.py --selftest       # headless acceptance run, exit code

Keys: SPACE pause/resume   Esc quit

Needs a Vulkan build (-DTHREEPP_WITH_VULKAN=ON): the raster DepthSensor path has
no intensity to report, so there is nothing to optimise against on GL.
"""
import math
import os
import sys
import time

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import threepp as tp
from warp_common import cli_arg, icosphere, parse_size, resize_handler, signed_volume

# ---- flags -------------------------------------------------------------------
SELFTEST = "--selftest" in sys.argv
TUNE = "--tune" in sys.argv            # optimise only, no window, print numbers
SWEEP = "--sweep" in sys.argv          # detection-range table and stop
CLIP = "--clip" in sys.argv            # 12 s share clip: 3 s per body + all three
MODE = ("both" if (SELFTEST or CLIP)
        else cli_arg("--mode", "both", str))  # stealth|conspicuous|both
SEED = int(cli_arg("--seed", 7, float))
# The selftest runs the SAME population as the demo and only shortens the search.
# Dropping the population to 16 to save time was a false economy: it slows the
# search per generation as well as per second, and 120 x 16 lands at 0.602 --
# on the wrong side of a gate that 160 x 20 clears at 0.436. Budget at 160 x 20
# is ~50 s per act, well inside the 4-minute cap.
GENS = int(cli_arg("--gens", 160 if SELFTEST else 180, float))
POP = int(cli_arg("--pop", 20, float))
SIGMA0 = cli_arg("--sigma", 0.16, float)
SHOT = cli_arg("--shot", "", str)
SHOT_DIR = cli_arg("--shots", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "doc", "screenshots"), str)
FRAME_BUDGET = int(cli_arg("--frames", 0, float))
WIN_W, WIN_H = parse_size(cli_arg("--size", "640x360" if (SELFTEST or SWEEP)
                                 else ("1920x1080" if CLIP else "1440x810"), str))
HEADLESS = SELFTEST or SWEEP or TUNE or CLIP or bool(SHOT)
# The optimised parameter vectors, cached so a clip re-render does not pay the
# ~2 min of optimisation again. Keyed on n_param/seed/gens; a mismatch simply
# re-optimises.
SHAPES_CACHE = cli_arg("--shapes-cache", os.path.join(
    os.environ.get("TEMP", "."), "lidar_sculpt_shapes.npz"), str)

# ---- the body ----------------------------------------------------------------
SUBDIV = int(cli_arg("--subdiv", 4, float))       # render mesh: 4 = 2562 v / 5120 f
CTRL_SUBDIV = int(cli_arg("--ctrl", 2, float))    # control mesh: 2 = 162 directions
R_BODY = cli_arg("--radius", 1.0, float)          # the baseline sphere, metres
R_MIN = cli_arg("--r-min", 0.50, float)           # radius clamp, in units of R_BODY
R_MAX = cli_arg("--r-max", 1.60, float)
LAP_SWEEPS = int(cli_arg("--lap", 2, float))      # Laplacian sweeps on the radius field
NO_SYM = "--no-sym" in sys.argv                   # let it go lopsided

# ---- the sensor --------------------------------------------------------------
# Angular resolution is FIXED and the ranges are chosen around it, which is the
# right way round: a real scanner has a beam pattern, not a zoom. 24 deg over 56
# rows is 0.43 deg/beam -- 6.7 cm at 9 m, 11 cm at 15 m, comparable to an
# automotive sensor looking at a car-sized object.
SFOV = cli_arg("--sensor-fov", 24.0, float)
SW = int(cli_arg("--sensor-w", 72, float))
SH = int(cli_arg("--sensor-h", 56, float))
# The two stand-off ranges. They are close in because `LidarParams::referenceRange`
# is 5 m and is NOT reachable from Python (DepthSensor builds its own LidarParams
# and exposes none of it), so the 1/r^2 term puts this sensor's useful window at
# roughly 5-25 m for a 2 m body. See the notes in plans/lidar-sculpt.md.
R_NEAR = cli_arg("--r-near", 8.0, float)
R_FAR = cli_arg("--r-far", 12.0, float)
EL_NEAR = cli_arg("--el-near", 10.0, float)
EL_FAR = cli_arg("--el-far", -5.0, float)
N_AZ_NEAR = int(cli_arg("--az-near", 8, float))
N_AZ_FAR = int(cli_arg("--az-far", 4, float))
# The detection threshold, in the engine's own normalised intensity. The tracer
# has already dropped everything under LidarParams::detectorThreshold = 0.005,
# so this is a STRICTER detector laid on top of the engine's, and it is the
# single most important number in the demo: it is what decides whether the
# sensor is ANGLE-SELECTIVE or merely geometric.
#
# Measured on the baseline sphere at 9 m (the sweep is in plans/lidar-sculpt.md):
# intensity runs 0.208 at normal incidence, 0.062 at 25 deg, 0.017 in a trough
# near 55 deg, and back up past 0.16 at grazing. At 0.010 EVERY return on the
# body clears the bar, incidence stops mattering, and the objective collapses to
# "how many beams hit you" -- which fixed volume already pins down, so stealth
# has nothing to find (it sat exactly on the baseline for a whole run). At 0.035
# only surfaces within ~27 deg of square-on survive, and tilting a facet is
# worth something again.
THRESH = cli_arg("--threshold", 0.035, float)
# Width of the linear ramp below THRESH, as a fraction of it. The ramp is the
# only smoothing anywhere in the objective -- a pure count is piecewise constant
# and hands CMA-ES a plateau to wander on -- but it has to be NARROW, or the
# half-credit handed to the 0.017 trough drowns the thing being measured.
RAMP = cli_arg("--ramp", 0.25, float)
MASK_R = cli_arg("--mask-r", 2.6, float)          # returns beyond this from a body's
                                                  # centre are not that body's
ROUGHNESS = cli_arg("--roughness", 0.62, float)
METALNESS = cli_arg("--metalness", 1.0, float)

# ---- the finale --------------------------------------------------------------
# The three lanes are 11 m apart so that a scanner orbiting ONE body at the 8 m
# stand-off never ends up inside its neighbour. Each body is scanned from its
# own copy of the same relative pose, which is what makes the finale a
# shape-only comparison: identical range, identical elevation, identical beam
# pattern, three different shapes. One shared sensor orbiting the whole row
# would have handed the outer two a range penalty the middle one never pays.
SPACING = cli_arg("--spacing", 10.0, float)
ORBIT_R = cli_arg("--orbit-r", 8.0, float)        # per-body stand-off, = R_NEAR
ORBIT_EL = cli_arg("--orbit-el", 10.0, float)
ORBIT_RATE = cli_arg("--orbit-rate", 14.0, float) # deg/s
AZ0 = cli_arg("--orbit-az0", 18.0, float)         # start near the camera's side

SPHERE_COLOR = 0x8d959f
STEALTH_COLOR = 0x59677a
CONSP_COLOR = 0xc2884a
DETECTED_RGB = (1.00, 0.72, 0.22)                 # above threshold: the sensor sees it
FAINT_RGB = (0.13, 0.26, 0.42)                    # returned, but under threshold

if not tp.HAS_VULKAN or not tp.vulkan_available():
    print("This example needs the Vulkan backend (configure with "
          "-DTHREEPP_WITH_VULKAN=ON) and a working Vulkan loader: the raster "
          "DepthSensor reports no intensity, so there is no sensor to optimise "
          "against.")
    sys.exit(0)

if HEADLESS:
    os.environ.setdefault("THREEPP_VULKAN_SUPPRESS_PRESENT", "1")


# --------------------------------------------------------------------------- #
#  Geometry: a radial field on a control mesh, resampled onto the render mesh.
#
#  The optimiser never touches a vertex. It sets one RADIUS per control
#  direction, and the render mesh is that field interpolated and smoothed. 162
#  control directions with a z-mirror pairing leave ~91 free numbers, which is
#  a search space CMA-ES can cover in a couple of thousand evaluations; 2562
#  free vertices is not.
# --------------------------------------------------------------------------- #
class Shape:

    def __init__(self):
        rv, rf = icosphere(SUBDIV)
        cv, _ = icosphere(CTRL_SUBDIV)
        self.faces = rf.astype(np.int32)
        self.rdir = (rv / np.linalg.norm(rv, axis=1, keepdims=True)).astype(np.float64)
        self.cdir = (cv / np.linalg.norm(cv, axis=1, keepdims=True)).astype(np.float64)
        self.nv = len(self.rdir)
        self.nc = len(self.cdir)

        # control -> render: inverse-square-angle blend of the 3 nearest control
        # directions. The plan called for the containing spherical triangle;
        # this is the same idea without the point-in-triangle bookkeeping, and
        # the Laplacian sweeps below wash out the difference between them.
        dots = np.clip(self.rdir @ self.cdir.T, -1.0, 1.0)
        idx = np.argsort(-dots, axis=1)[:, :3]
        ang = np.arccos(np.take_along_axis(dots, idx, axis=1))
        w = 1.0 / (ang * ang + 1.0e-6)
        self.w_idx = idx
        self.w = w / w.sum(axis=1, keepdims=True)

        # render-mesh 1-ring, as CSR, for the radius-field Laplacian.
        nbr = [set() for _ in range(self.nv)]
        for a, b, c in self.faces:
            for i, j in ((a, b), (b, c), (c, a)):
                nbr[i].add(j)
                nbr[j].add(i)
        self.adj_off = np.zeros(self.nv + 1, np.int32)
        flat = []
        for i, s in enumerate(nbr):
            flat.extend(sorted(s))
            self.adj_off[i + 1] = len(flat)
        self.adj = np.array(flat, np.int32)

        # z-mirror pairing on the CONTROL mesh: exact on an icosphere, so
        # symmetry is a PROJECTION (both halves of a pair get the same number)
        # and not a term competing with the objective.
        key = {tuple(np.round(d, 5)): i for i, d in enumerate(self.cdir)}
        self.pair = np.arange(self.nc)
        for i, d in enumerate(self.cdir):
            j = key.get(tuple(np.round((d[0], d[1], -d[2]), 5)))
            if j is not None:
                self.pair[i] = j
        if NO_SYM:
            self.group = np.arange(self.nc)
        else:
            self.group = np.minimum(np.arange(self.nc), self.pair)
        self.gid, self.ginv = np.unique(self.group, return_inverse=True)
        self.n_param = len(self.gid)

        # face-adjacency for the vertex normals (picture only -- the LIDAR uses
        # face normals it derives itself).
        self.v0 = self.faces[:, 0]
        self.v1 = self.faces[:, 1]
        self.v2 = self.faces[:, 2]

        self.pos0 = (self.rdir * R_BODY).astype(np.float32)
        self.v_target = signed_volume(self.pos0.astype(np.float64), self.faces)

    # -- the map: parameters -> positions, at fixed volume ------------------
    def build(self, x):
        """x (n_param,) -> (positions (nv,3) float32, volume error)."""
        rc = np.clip(1.0 + x[self.ginv], R_MIN, R_MAX)
        r = (rc[self.w_idx] * self.w).sum(axis=1)
        for _ in range(LAP_SWEEPS):
            acc = np.add.reduceat(r[self.adj], self.adj_off[:-1])
            cnt = np.diff(self.adj_off)
            r = 0.5 * r + 0.5 * (acc / cnt)
        pos = self.rdir * (r * R_BODY)[:, None]
        v = signed_volume(pos, self.faces)
        # EXACT volume projection, not a penalty: scale every radius by the one
        # number that puts the enclosed volume back on target. Without it both
        # answers are trivial -- stealth shrinks, conspicuity grows.
        pos *= (self.v_target / max(v, 1.0e-9)) ** (1.0 / 3.0)
        err = abs(signed_volume(pos, self.faces) / self.v_target - 1.0)
        return pos.astype(np.float32), err

    def normals(self, pos):
        """Area-weighted vertex normals. For the raster picture only."""
        p = pos.astype(np.float64)
        fn = np.cross(p[self.v1] - p[self.v0], p[self.v2] - p[self.v0])
        n = np.zeros_like(p)
        for c in (self.v0, self.v1, self.v2):
            np.add.at(n, c, fn)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        return (n / np.maximum(ln, 1.0e-12)).astype(np.float32)

    def facet_rms(self, pos):
        """Dihedral-angle RMS over the mesh edges, in degrees.

        A number for the thing the pictures are supposed to show: a smooth
        sphere at this resolution sits near 2 deg, a genuinely faceted body
        runs several times that because the creases carry the whole budget.
        """
        p = pos.astype(np.float64)
        fn = np.cross(p[self.v1] - p[self.v0], p[self.v2] - p[self.v0])
        fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
        edges = {}
        for fi, (a, b, c) in enumerate(self.faces):
            for i, j in ((a, b), (b, c), (c, a)):
                edges.setdefault((min(i, j), max(i, j)), []).append(fi)
        d = [np.clip(np.dot(fn[f[0]], fn[f[1]]), -1.0, 1.0)
             for f in edges.values() if len(f) == 2]
        return float(np.degrees(np.sqrt(np.mean(np.arccos(d) ** 2))))


# --------------------------------------------------------------------------- #
#  The sensor, which is the loss.
# --------------------------------------------------------------------------- #
def pose_offsets():
    """The threat model: sensor origins relative to a body's centre."""
    out = []
    for k in range(N_AZ_NEAR):
        out.append(_sph(360.0 * k / N_AZ_NEAR, EL_NEAR, R_NEAR))
    for k in range(N_AZ_FAR):
        out.append(_sph(360.0 * k / N_AZ_FAR + 45.0, EL_FAR, R_FAR))
    return out


def _sph(az_deg, el_deg, r):
    a, e = math.radians(az_deg), math.radians(el_deg)
    return np.array([r * math.cos(e) * math.sin(a),
                     r * math.sin(e),
                     r * math.cos(e) * math.cos(a)], np.float64)


def aim(sensor, origin, target):
    """Place a DepthSensor at `origin` and point its beams at `target`.

    A straight look_at: sensors override the lookAt convention so the beams
    (local -Z) turn toward the target, exactly as for a camera. Older threepp
    releases applied the plain-Object3D convention -- +Z toward the target,
    beams exactly backwards, an empty cloud from every pose -- which is why
    this helper once aimed at the mirror point 2*origin - target.
    """
    sensor.position.set(float(origin[0]), float(origin[1]), float(origin[2]))
    sensor.look_at(float(target[0]), float(target[1]), float(target[2]))


def _pinhole_dirs(fov_y_deg, w, h):
    """Unit ray directions of a pinhole grid looking down local -Z, y up."""
    t = math.tan(math.radians(fov_y_deg) * 0.5)
    ys = t * (2.0 * (np.arange(h) + 0.5) / h - 1.0)
    xs = t * (w / h) * (2.0 * (np.arange(w) + 0.5) / w - 1.0)
    gx, gy = np.meshgrid(xs, ys)
    d = np.stack([gx, gy, -np.ones_like(gx)], axis=-1).reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


class Detector:
    """Scores a body by scanning it, and hands back the cloud for the picture.

    The whole 12-pose threat model is ONE beam table fired through
    `renderer.scan_lidar` in a single dispatch. It used to be 12 sequential
    DepthSensor scans, and that shape is round-trip-bound, not GPU-bound: each
    pose is a tiny trace followed by a synchronous readback, and the GPU sat
    ~70% idle between them. The table is precomputed RELATIVE to the body
    centre once (the poses never change); per evaluation only the origins are
    offset. `scan_pose` keeps the single-pose DepthSensor path for the sweep
    and detection-range utilities, where one pose at a time IS the question.
    """

    def __init__(self, renderer, scene):
        self.renderer = renderer
        self.scene = scene
        self.sensor = tp.DepthSensor(fov_y=SFOV, width=SW, height=SH,
                                     near=0.2, far=400.0)
        # A zero noise model: the score has to be a property of the SHAPE, not
        # of where the pseudo-random stream happened to be. Two evaluations of
        # the same mesh return bit-identical clouds (asserted in --selftest).
        self.sensor.noise = tp.RangeNoiseModel()
        self.offsets = pose_offsets()
        local = _pinhole_dirs(SFOV, SW, SH)
        origins, dirs = [], []
        for off in self.offsets:
            f = -off / np.linalg.norm(off)               # aim at the centre
            right = np.cross(f, [0.0, 1.0, 0.0])
            right /= np.linalg.norm(right)               # elevations are +/-10 deg:
            up = np.cross(right, f)                      # never parallel to world-up
            m = np.stack([right, up, -f], axis=1)        # local axes -> world
            dirs.append(local @ m.T)
            origins.append(np.broadcast_to(off, (len(local), 3)))
        self.beam_origins_rel = np.ascontiguousarray(np.concatenate(origins), np.float32)
        self.beam_dirs = np.ascontiguousarray(np.concatenate(dirs), np.float32)
        self.params = tp.LidarParams()
        self.params.max_range = 400.0
        self.params.min_range = 0.2
        self.params.detector_threshold = 0.0             # thresholds live in
        #                                                  python, same as before
        self.evals = 0
        self.eval_ms = 0.0

    def scan_pose(self, origin, centre):
        aim(self.sensor, origin, centre)
        pts, cols = self.sensor.scan_rgbd(self.renderer, self.scene)
        if not len(pts):
            return pts, np.zeros(0, np.float32)
        d = pts - centre.astype(np.float32)
        m = (d * d).sum(1) <= MASK_R * MASK_R
        return pts[m], cols[m, 0]

    def score(self, centre=np.zeros(3), collect=False):
        """(soft score, hard detected count, cloud, detected mask).

        `soft` ramps a return linearly from the engine's own detector floor up
        to THRESH and holds at 1 above it; `hard` is the plain count at THRESH.
        The optimiser reads soft, the report quotes hard.
        """
        t0 = time.perf_counter()
        c32 = centre.astype(np.float32)
        d = self.renderer.scan_lidar(self.beam_origins_rel + c32,
                                     self.beam_dirs, self.params)
        pts, inten = d["position"], d["intensity"]
        ok = d["return_no"] > 0
        rel = pts - c32
        ok &= (rel * rel).sum(1) <= MASK_R * MASK_R
        pts, inten = pts[ok], inten[ok]
        soft = float(np.clip((inten - (1.0 - RAMP) * THRESH)
                             / (RAMP * THRESH), 0.0, 1.0).sum())
        hit = inten >= THRESH
        hard = int(hit.sum())
        self.evals += 1
        self.eval_ms = (time.perf_counter() - t0) * 1000.0
        if collect:
            return soft, hard, pts, hit
        return soft, hard, None, None


# --------------------------------------------------------------------------- #
#  CMA-ES, ~70 lines of numpy. Reuse-first: no pip install for one optimiser.
#  Hansen's reference update, minimising; the conspicuity act flips the sign of
#  the objective rather than the algorithm.
# --------------------------------------------------------------------------- #
class CMAES:

    def __init__(self, n, sigma, seed, lam):
        self.n = n
        self.sigma = sigma
        self.lam = lam
        self.mu = lam // 2
        w = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.w = w / w.sum()
        self.mueff = 1.0 / (self.w ** 2).sum()
        self.cc = (4.0 + self.mueff / n) / (n + 4.0 + 2.0 * self.mueff / n)
        self.cs = (self.mueff + 2.0) / (n + self.mueff + 5.0)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1.0 - self.c1,
                       2.0 * (self.mueff - 2.0 + 1.0 / self.mueff)
                       / ((n + 2.0) ** 2 + self.mueff))
        self.damps = 1.0 + 2.0 * max(0.0, math.sqrt((self.mueff - 1.0) / (n + 1.0))
                                     - 1.0) + self.cs
        self.mean = np.zeros(n)
        self.pc = np.zeros(n)
        self.ps = np.zeros(n)
        self.C = np.eye(n)
        self.B = np.eye(n)
        self.D = np.ones(n)
        self.chiN = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))
        self.rng = np.random.default_rng(seed)
        self.gen = 0
        self.since_eig = 0

    def ask(self):
        z = self.rng.standard_normal((self.lam, self.n))
        self.z = z
        self.y = z @ (self.B * self.D).T
        return self.mean + self.sigma * self.y

    def tell(self, fitness):
        n, lam = self.n, self.lam
        order = np.argsort(fitness)
        ysel = self.y[order[:self.mu]]
        yw = self.w @ ysel
        self.mean = self.mean + self.sigma * yw

        invsqrtC = self.B @ np.diag(1.0 / self.D) @ self.B.T
        self.ps = (1.0 - self.cs) * self.ps + \
            math.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * (invsqrtC @ yw)
        self.gen += 1
        hsig = (np.linalg.norm(self.ps)
                / math.sqrt(1.0 - (1.0 - self.cs) ** (2 * self.gen))
                / self.chiN) < (1.4 + 2.0 / (n + 1.0))
        self.pc = (1.0 - self.cc) * self.pc + \
            (1.0 if hsig else 0.0) * math.sqrt(self.cc * (2.0 - self.cc) * self.mueff) * yw

        delta = (1.0 - (1.0 if hsig else 0.0)) * self.cc * (2.0 - self.cc)
        self.C = ((1.0 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + delta * self.C)
                  + self.cmu * (ysel.T * self.w) @ ysel)
        self.sigma *= math.exp((self.cs / self.damps)
                               * (np.linalg.norm(self.ps) / self.chiN - 1.0))
        self.sigma = min(self.sigma, 1.0)

        self.since_eig += lam
        if self.since_eig > lam * max(1, int(1.0 / (10.0 * n * (self.c1 + self.cmu)))):
            self.since_eig = 0
            self.C = np.triu(self.C) + np.triu(self.C, 1).T
            d2, self.B = np.linalg.eigh(self.C)
            self.D = np.sqrt(np.maximum(d2, 1.0e-20))
        return order[0]


# --------------------------------------------------------------------------- #
#  The optimisation run.
# --------------------------------------------------------------------------- #
class Run:
    """One act: stealth or conspicuous, against one live mesh in the scene."""

    def __init__(self, shape, geom, det, mode, baseline, seed):
        self.shape = shape
        self.geom = geom
        self.det = det
        self.mode = mode
        self.sign = -1.0 if mode == "conspicuous" else 1.0   # CMA-ES minimises
        self.baseline = baseline
        self.es = CMAES(shape.n_param, SIGMA0, seed, POP)
        self.best_x = np.zeros(shape.n_param)
        self.best_f = self.sign * baseline
        self.best_hard = None
        self.best_verr = 0.0
        self.gen = 0
        self.done = False
        self.t0 = time.perf_counter()
        self.cloud = np.zeros((0, 3), np.float32)
        self.det_mask = np.zeros(0, bool)
        # Called after every candidate evaluation with the candidate's OWN
        # cloud already stashed on self. The windowed path uses it to paint
        # the returns and present the matched mesh+cloud pair: publish()
        # PRESENTS each candidate (its render is what rebuilds the TLAS), so
        # without this the window flashes twenty candidate geometries per
        # generation against a cloud scanned from a different shape.
        self.on_eval = None

    def publish(self, x):
        pos, err = self.shape.build(x)
        self.geom.update_attribute("position", pos)
        self.geom.update_attribute("normal", self.shape.normals(pos))
        # ONE render rebuilds the BLAS/TLAS the tracer reads. Without it every
        # candidate is scored against the previous candidate's geometry.
        self.det.renderer.render(SCENE, CAMERA)
        return pos, err

    def step(self, collect=False):
        """One generation. Returns True while there is more to do."""
        if self.done:
            return False
        xs = self.es.ask()
        fit = np.empty(self.es.lam)
        verr = np.empty(self.es.lam)
        for i, x in enumerate(xs):
            _, verr[i] = self.publish(x)
            soft, hard, cloud, det = self.det.score(collect=self.on_eval is not None)
            fit[i] = self.sign * soft
            if self.on_eval is not None:
                self.cloud, self.det_mask = cloud, det
                self.on_eval(self)
        k = self.es.tell(fit)
        if fit[k] < self.best_f:
            self.best_f = float(fit[k])
            self.best_x = xs[k].copy()
            self.best_verr = float(verr[k])
        self.gen += 1
        # Re-publish the incumbent so the window (and the next act) is looking
        # at the best shape and not at the last candidate tried.
        pos, _ = self.publish(self.best_x)
        soft, hard, cloud, det = self.det.score(collect=collect)
        self.best_hard = hard
        if collect:
            self.cloud, self.det_mask = cloud, det
        self.done = self.gen >= GENS
        return not self.done

    @property
    def score(self):
        return abs(self.best_f)

    @property
    def ratio(self):
        return self.score / max(self.baseline, 1e-9)


# --------------------------------------------------------------------------- #
#  Scene
# --------------------------------------------------------------------------- #
SCENE = None
CAMERA = None


def make_geom(shape, pos):
    g = tp.BufferGeometry()
    g.set_attribute("position", pos.copy())
    g.set_attribute("normal", shape.normals(pos))
    g.set_index(shape.faces.reshape(-1).astype(np.uint32))
    return g


def make_material(color):
    m = tp.MeshStandardMaterial()
    m.color = color
    m.roughness = ROUGHNESS
    m.metalness = METALNESS
    return m


def make_cloud(capacity):
    g = tp.BufferGeometry()
    g.set_attribute("position", np.zeros((capacity, 3), np.float32))
    g.set_attribute("color", np.zeros((capacity, 3), np.float32))
    g.set_draw_range(0, 0)
    pm = tp.PointsMaterial()
    pm.size = 0.055
    pm.size_attenuation = True
    pm.vertex_colors = True
    pts = tp.Points(g, pm)
    pts.frustum_culled = False
    return pts, g


def paint_cloud(geom, capacity, cloud, det):
    """Detected points warm and bright, sub-threshold returns dim blue.

    Points render through the overlay pass and are not triangles, so they are
    not in the ray-traced acceleration structure: the cloud cannot scatter the
    beams that drew it. That is what makes it safe to leave in the scoring
    scene rather than toggling its visibility every evaluation (which would
    churn the entry list and force a rebuild + waitIdle per candidate).
    """
    n = min(len(cloud), capacity)
    if n:
        col = np.empty((n, 3), np.float32)
        col[:] = FAINT_RGB
        col[det[:n]] = DETECTED_RGB
        geom.update_attribute("position", cloud[:n].astype(np.float32))
        geom.update_attribute("color", col)
    geom.set_draw_range(0, n)


def build_scene(shape):
    global SCENE, CAMERA
    scene = tp.Scene()
    scene.background = 0x0b0e13
    scene.add(tp.HemisphereLight(0xc8d8ff, 0x1a1f28, 1.2))
    sun = tp.DirectionalLight(0xffffff, 2.4)
    sun.position.set(5, 9, 6)
    scene.add(sun)
    rim = tp.DirectionalLight(0x88aaff, 1.1)
    rim.position.set(-7, 3, -5)
    scene.add(rim)
    SCENE = scene
    camera = tp.PerspectiveCamera(45, WIN_W / max(WIN_H, 1), 0.1, 400)
    CAMERA = camera
    return scene, camera


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def print_model():
    print("  sensor      DepthSensor(fov_y=%.0f, %dx%d) -> scan_rgbd on Vulkan; "
          "intensity is the engine's own LIDAR equation" % (SFOV, SW, SH))
    print("  material    roughness %.2f  metalness %.2f  (identical on all three "
          "bodies)" % (ROUGHNESS, METALNESS))
    print("  poses       %d az at %.0f m / el %+.0f deg,  %d az at %.0f m / el "
          "%+.0f deg  = %d scans/eval"
          % (N_AZ_NEAR, R_NEAR, EL_NEAR, N_AZ_FAR, R_FAR, EL_FAR,
             N_AZ_NEAR + N_AZ_FAR))
    print("  detector    engine floor 0.005 (LidarParams::detectorThreshold), "
          "demo threshold %.3f" % THRESH)


def azimuth_table(det, centre, shapes, radius, el=0.0, n_az=12):
    """Detected returns per azimuth, one row per shape. `shapes` = [(name, pub)]."""
    azs = [360.0 * k / n_az for k in range(n_az)]
    print(f"  detected returns vs azimuth at {radius:.0f} m, elevation {el:+.0f} deg")
    print("      az       " + " ".join(f"{a:5.0f}" for a in azs) + "    total")
    for name, publish in shapes:
        publish()
        row = []
        for a in azs:
            pts, inten = det.scan_pose(centre + _sph(a, el, radius), centre)
            row.append(int((inten >= THRESH).sum()) if len(inten) else 0)
        print(f"      {name:<9}" + " ".join(f"{v:5d}" for v in row)
              + f"  {sum(row):7d}")


def detection_range(det, centre, publish, floor, lo=5.0, hi=40.0, step=0.5, n_az=8):
    """The largest range at which the mean detected count over azimuth is >= floor.

    Walked outward and reported at the LAST range that still clears the floor,
    so a shape with a hole in one azimuth is not rescued by its good side.
    """
    publish()
    best = 0.0
    r = lo
    while r <= hi + 1e-9:
        tot = 0
        for k in range(n_az):
            pts, inten = det.scan_pose(centre + _sph(360.0 * k / n_az, 0.0, r), centre)
            tot += int((inten >= THRESH).sum()) if len(inten) else 0
        if tot / n_az >= floor:
            best = r
        r += step
    return best


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    shape = Shape()
    print(__doc__.split("\n")[0])
    print(f"  mesh        {shape.nv} verts / {len(shape.faces)} faces; "
          f"{shape.nc} control directions -> {shape.n_param} free parameters"
          f"{'' if NO_SYM else ' (z-mirror paired)'}")
    print_model()
    print(f"  volume      {shape.v_target:.4f} m^3, held exactly by radial "
          f"projection every candidate")

    # vsync OFF is load-bearing, not preference: every evaluation presents two
    # frames (the candidate render that rebuilds the TLAS, and the matched
    # mesh+cloud pair), and with vsync each present blocks on the 60 Hz swap --
    # ~33 ms of pure waiting per ~7 ms of work, which read as "generation is
    # slow, GPU at 30%".
    canvas = tp.Canvas("threepp - lidar sculpt", width=WIN_W, height=WIN_H,
                       headless=HEADLESS, vsync=False)
    renderer = tp.VulkanRenderer(canvas)
    scene, camera = build_scene(shape)

    pos0, _ = shape.build(np.zeros(shape.n_param))
    geom = make_geom(shape, pos0)
    mesh = tp.Mesh(geom, make_material(SPHERE_COLOR))
    scene.add(mesh)

    cap = (N_AZ_NEAR + N_AZ_FAR) * SW * SH
    cloud_obj, cloud_geom = make_cloud(cap)
    scene.add(cloud_obj)

    camera.position.set(4.4, 2.6, 4.4)
    camera.look_at(0, 0, 0)
    # Material state reaches the GPU one render late (the per-entry material
    # patch lands with the next frame's uploads), so warm up before scoring or
    # the baseline is measured against the default material.
    for _ in range(3):
        renderer.render(scene, camera)

    det = Detector(renderer, scene)
    t0 = time.perf_counter()
    base_soft, base_hard, base_cloud, base_det = det.score(collect=True)
    print(f"\n  BASELINE sphere: soft {base_soft:.1f}, {base_hard} detected "
          f"returns over {len(det.offsets)} poses   ({det.eval_ms:.1f} ms/eval)")
    if base_hard < 40:
        print("  WARNING: the baseline sphere is barely detected at all; the "
              "objective has little to work with. Move --r-near in.")
    paint_cloud(cloud_geom, cap, base_cloud, base_det)

    if SWEEP:
        run_sweep(shape, geom, det, renderer, canvas, camera)
        return

    modes = {"stealth": ["stealth"], "conspicuous": ["conspicuous"],
             "both": ["stealth", "conspicuous"]}.get(MODE)
    if modes is None:
        print(f"  unknown --mode {MODE!r}; use stealth | conspicuous | both")
        return

    results = {"sphere": (base_soft, base_hard, np.zeros(shape.n_param), 0.0)}
    state = dict(run=None, label="")

    if not HEADLESS:
        canvas.on_window_resize(resize_handler(camera, renderer))
        controls = tp.OrbitControls(camera, canvas)
        controls.enable_damping = True
    else:
        controls = None
    ui = tp.ImguiContext(canvas, renderer) if (tp.HAS_IMGUI and not HEADLESS) else None

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12)
        tp.imgui.set_next_window_size(380, 0)
        tp.imgui.begin("the sensor is the loss")
        r = state["run"]
        if r is not None:
            tp.imgui.text(f"{r.mode.upper()}   generation {r.gen} / {GENS}")
            tp.imgui.text(f"score  {r.score:8.1f}   =  {r.ratio:5.3f} x sphere")
            tp.imgui.text(f"detected returns   {r.best_hard}  "
                          f"(sphere {base_hard})")
            tp.imgui.text(f"volume error   {100.0 * r.best_verr:.3f} %")
            tp.imgui.text(f"{det.eval_ms:.1f} ms / evaluation   "
                          f"{det.evals} evaluations")
        else:
            tp.imgui.text(state["label"])
        tp.imgui.separator()
        tp.imgui.text(f"{len(det.offsets)} poses  |  threshold {THRESH:.3f}")
        tp.imgui.text("orange = detected   blue = under threshold")
        tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps   drag to orbit")
        tp.imgui.end()

    def draw():
        if controls is not None:
            controls.update()
        renderer.render(scene, camera)
        if ui is not None:
            if controls is not None:
                controls.enabled = not ui.want_capture_mouse
            ui.render(draw_ui)

    alive = [True]

    def pump():
        """One window frame; False once the window has gone away."""
        if HEADLESS:
            draw()
            return True
        alive[0] = canvas.animate_once(draw)
        return alive[0]

    clip_loaded = False
    if CLIP and os.path.exists(SHAPES_CACHE):
        z = np.load(SHAPES_CACHE)
        if (int(z["n_param"]) == shape.n_param and int(z["seed"]) == SEED
                and int(z["gens"]) == GENS):
            for nm in ("stealth", "conspicuous"):
                results[nm] = (float(z[nm + "_soft"]), int(z[nm + "_hard"]),
                               z[nm + "_x"], float(z[nm + "_err"]))
            clip_loaded = True
            print(f"  clip: optimised shapes loaded from {SHAPES_CACHE}")

    for mode in (() if clip_loaded else modes):
        mesh.material.color = STEALTH_COLOR if mode == "stealth" else CONSP_COLOR
        run = Run(shape, geom, det, mode, base_soft, SEED)
        state["run"] = run

        def on_eval(r):
            # Candidate k's returns over candidate k's mesh, presented as a
            # pair: paint the cloud the scan just produced, then render once
            # more. publish()'s own render presented this candidate with the
            # PREVIOUS cloud (the scan needs the TLAS before the cloud can
            # exist), so without this second present the window spends the
            # whole generation showing meshes with someone else's returns.
            # Costs ~3 ms on a ~15 ms eval and keeps the window interactive
            # through the ~300 ms a generation takes.
            paint_cloud(cloud_geom, cap, r.cloud, r.det_mask)
            if not pump():
                r.done = True

        run.on_eval = on_eval if not HEADLESS else None
        print(f"\n  --- {mode.upper()} " + "-" * 52)
        t1 = time.perf_counter()
        while run.step(collect=True):
            paint_cloud(cloud_geom, cap, run.cloud, run.det_mask)
            if not pump():                       # gen boundary: incumbent + its cloud
                break
            if run.gen % 10 == 0 or run.gen == 1:
                print(f"    gen {run.gen:4d}   score {run.score:8.1f}  "
                      f"{run.ratio:6.3f} x sphere   detected {run.best_hard:5d}   "
                      f"dV {100.0 * run.best_verr:.3f} %")
        pos, err = run.publish(run.best_x)
        soft, hard, cloud, dm = det.score(collect=True)
        paint_cloud(cloud_geom, cap, cloud, dm)
        print(f"    gen {run.gen:4d}   score {soft:8.1f}  "
              f"{soft / base_soft:6.3f} x sphere   detected {hard:5d}   "
              f"dV {100.0 * err:.3f} %   [{time.perf_counter() - t1:.1f} s]")
        print(f"    facet RMS dihedral {shape.facet_rms(pos):.2f} deg "
              f"(sphere {shape.facet_rms(pos0):.2f} deg)")
        results[mode] = (soft, hard, run.best_x.copy(), err)
        if not alive[0]:
            break

    print(f"\n  total {time.perf_counter() - t0:.1f} s, {det.evals} evaluations "
          f"at {det.eval_ms:.1f} ms")
    summary(results, base_soft, base_hard)

    if CLIP:
        if not clip_loaded and "stealth" in results and "conspicuous" in results:
            np.savez(SHAPES_CACHE, n_param=shape.n_param, seed=SEED, gens=GENS,
                     **{f"{nm}_{k}": v for nm in ("stealth", "conspicuous")
                        for k, v in zip(("soft", "hard", "x", "err"), results[nm])})
            print(f"  clip: shapes cached to {SHAPES_CACHE}")
        render_clip(shape, renderer, scene, camera, results, mesh, cloud_obj)
        return

    # The selftest runs BEFORE the finale: the finale retires the optimisation
    # mesh, and the determinism check needs it still in the scene.
    if SELFTEST:
        def publish(x):
            pos, err = shape.build(x)
            geom.update_attribute("position", pos)
            geom.update_attribute("normal", shape.normals(pos))
            renderer.render(scene, camera)
            return err
        code = selftest_verdict(shape, det, results, base_soft, publish)
        sys.exit(code)

    # ---- the finale: three bodies, one orbiting scanner -------------------
    if alive[0] and len(results) > 2:
        finale(shape, det, renderer, canvas, scene, camera, results, base_soft,
               controls, ui, mesh, cloud_obj)


def summary(results, base_soft, base_hard):
    print("\n  " + "-" * 66)
    print(f"  {'shape':<14}{'score':>10}{'x sphere':>11}{'detected':>11}"
          f"{'dV':>10}")
    for name in ("sphere", "stealth", "conspicuous"):
        if name not in results:
            continue
        soft, hard, _, err = results[name]
        print(f"  {name:<14}{soft:10.1f}{soft / base_soft:11.3f}{hard:11d}"
              f"{100.0 * err:9.3f}%")
    print("  " + "-" * 66)


# --------------------------------------------------------------------------- #
#  The share clip (--clip): 12 seconds, four beats -- 3 s per body under a
#  sweeping scanner, then all three under the same pose. Headless and
#  deterministic (the azimuth is a function of the frame index, never the wall
#  clock), raw RGB streamed straight into x264 -- the pipeline the hull film
#  settled on after paying 5.4 GB of PNG intermediates to learn it.
# --------------------------------------------------------------------------- #
def _ffmpeg_exe():
    import shutil
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = None
    return exe


def render_clip(shape, renderer, scene, camera, results, old_mesh, old_cloud):
    import subprocess
    print("\n  --- CLIP " + "-" * 54)
    fps, secs = 30, 3.0
    names = ("sphere", "stealth", "conspicuous")
    colors = (SPHERE_COLOR, STEALTH_COLOR, CONSP_COLOR)
    centres = [np.array([(i - 1) * SPACING, 0.0, 0.0]) for i in range(3)]

    scene.remove(old_mesh)
    scene.remove(old_cloud)
    bodies = []
    for i, name in enumerate(names):
        pos, _ = shape.build(results[name][2])
        g = make_geom(shape, pos)
        m = tp.Mesh(g, make_material(colors[i]))
        m.position.set(*centres[i])
        scene.add(m)
        bodies.append((name, centres[i]))

    orbit = tp.DepthSensor(fov_y=SFOV, width=SW, height=SH, near=0.2, far=400.0)
    orbit.noise = tp.RangeNoiseModel()
    cap = 3 * SW * SH
    cloud_obj, cloud_geom = make_cloud(cap)
    cloud_obj.material.size = 0.11
    scene.add(cloud_obj)
    for _ in range(3):
        renderer.render(scene, camera)      # material warm-up, as everywhere

    exe = _ffmpeg_exe()
    if exe is None:
        print("  no ffmpeg on PATH and no imageio-ffmpeg; cannot encode the clip")
        return
    out = os.path.join(SHOT_DIR, "lidar_sculpt.mp4")
    os.makedirs(SHOT_DIR, exist_ok=True)
    log = os.path.join(os.environ.get("TEMP", "."), "lidar_sculpt_ffmpeg.log")
    cmd = [exe, "-y", "-hide_banner", "-loglevel", "warning",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{WIN_W}x{WIN_H}", "-r", str(fps), "-i", "-",
           "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", "-preset", "medium", "-movflags", "+faststart", out]
    logf = open(log, "w")
    enc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=logf, stderr=logf)

    fpseg = int(fps * secs)
    t0 = time.perf_counter()
    for f in range(4 * fpseg):
        seg, fu = divmod(f, fpseg)
        u = (fu + 0.5) / fpseg                  # (0, 1) inside the beat
        # A 240-degree sweep centred on the camera's side of each body. A full
        # 360 spends half the beat with the returns on the far side, hidden
        # behind the very body they landed on -- the same lesson the finale's
        # AZ0 comment records.
        az = AZ0 - 120.0 + 240.0 * u
        off = _sph(az, ORBIT_EL, ORBIT_R)
        featured = bodies if seg == 3 else [bodies[seg]]
        allp, alld = [], []
        for name, c in featured:
            aim(orbit, c + off, c)
            pts, cols = orbit.scan_rgbd(renderer, scene)
            if not len(pts):
                continue
            d = pts - c.astype(np.float32)
            msk = (d * d).sum(1) <= MASK_R * MASK_R
            if msk.any():
                allp.append(pts[msk])
                alld.append(cols[msk, 0] >= THRESH)
        paint_cloud(cloud_geom, cap,
                    np.concatenate(allp) if allp else np.zeros((0, 3), np.float32),
                    np.concatenate(alld) if alld else np.zeros(0, bool))

        if seg < 3:
            c = bodies[seg][1]
            th = -0.30 + 0.60 * u               # one slow arc per beat, no cuts
            eye = c + np.array([5.0 * math.sin(th), 2.1, 5.0 * math.cos(th)])
            look = c + np.array([0.0, 0.15, 0.0])
        else:
            eye = np.array([0.0, 5.0 + 1.2 * u, 19.5 + 2.5 * u])
            look = np.zeros(3)
        camera.position.set(float(eye[0]), float(eye[1]), float(eye[2]))
        camera.look_at(float(look[0]), float(look[1]), float(look[2]))

        renderer.render(scene, camera)
        rgb = renderer.read_pixels()
        if f == 0:
            assert rgb.shape[:2] == (WIN_H, WIN_W), f"readback {rgb.shape}"
        enc.stdin.write(np.ascontiguousarray(rgb, np.uint8).tobytes())

    enc.stdin.close()
    rc = enc.wait()
    logf.close()
    dt = time.perf_counter() - t0
    sz = os.path.getsize(out) / 1e6 if os.path.exists(out) else 0.0
    print(f"  wrote {out}  ({4 * fpseg} frames in {dt:.1f} s = "
          f"{4 * fpseg / dt:.1f} fps, {sz:.1f} MB, x264 rc={rc})")


# --------------------------------------------------------------------------- #
#  Finale: baseline sphere, stealth and conspicuous side by side, one LIDAR
#  orbiting all three at once. Same volume, same material, same beams -- the
#  only difference on the screen is shape, and the clouds say what that buys.
# --------------------------------------------------------------------------- #
def finale(shape, det, renderer, canvas, scene, camera, results, base_soft,
           controls, ui, old_mesh, old_cloud):
    print("\n  --- FINALE " + "-" * 52)
    names = ("sphere", "stealth", "conspicuous")
    colors = (SPHERE_COLOR, STEALTH_COLOR, CONSP_COLOR)
    centres = [np.array([(i - 1) * SPACING, 0.0, 0.0]) for i in range(3)]

    # The optimisation mesh and its cloud are retired; all three answers get
    # their own mesh so they can stand next to each other.
    scene.remove(old_mesh)
    scene.remove(old_cloud)

    bodies = []
    for i, name in enumerate(names):
        if name not in results:
            continue
        pos, _ = shape.build(results[name][2])
        g = make_geom(shape, pos)
        m = tp.Mesh(g, make_material(colors[i]))
        m.position.set(*centres[i])
        scene.add(m)
        bodies.append(dict(name=name, centre=centres[i], mesh=m))

    # The same beam pattern the optimisation was scored with, so what the
    # finale draws is the objective and not a prettier cousin of it.
    orbit = tp.DepthSensor(fov_y=SFOV, width=SW, height=SH, near=0.2, far=400.0)
    orbit.noise = tp.RangeNoiseModel()
    cap = 3 * SW * SH
    cloud_obj, cloud_geom = make_cloud(cap)
    cloud_obj.material.size = 0.11
    scene.add(cloud_obj)

    camera.position.set(0.0, 5.0, 19.5)
    camera.look_at(0, 0, 0)
    if controls is not None:
        controls.target = tp.Vector3(0, 0, 0)
    for _ in range(3):
        renderer.render(scene, camera)

    # Start the scanner on the CAMERA's side of the row (azimuth 0 is +z, which
    # is where the camera stands). Facing it away puts every detected return on
    # the far side of its own body, where the body then hides it -- which is
    # exactly what the first version of this shot did.
    live = dict(az=AZ0, counts={b["name"]: 0 for b in bodies})

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12)
        tp.imgui.set_next_window_size(400, 0)
        tp.imgui.begin("same pose, three shapes")
        tp.imgui.text(f"scanner at {ORBIT_R:.0f} m on each body, "
                      f"azimuth {live['az'] % 360.0:5.1f} deg")
        tp.imgui.text(f"equal volume {shape.v_target:.3f} m^3, equal material")
        tp.imgui.separator()
        tp.imgui.text("detected returns, this instant:")
        for b in bodies:
            tp.imgui.text(f"   {b['name']:<12}{live['counts'][b['name']]:5d}")
        tp.imgui.separator()
        tp.imgui.text("pose-set score (the objective):")
        for b in bodies:
            s = results[b["name"]][0]
            tp.imgui.text(f"   {b['name']:<12}{s:8.1f}   "
                          f"{s / base_soft:5.3f} x sphere")
        tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps   drag to orbit")
        tp.imgui.end()

    t0 = time.perf_counter()
    frames = 0
    shot_written = False
    while True:
        live["az"] = AZ0 + ORBIT_RATE * (time.perf_counter() - t0)
        off = _sph(live["az"], ORBIT_EL, ORBIT_R)
        # One pose, applied to each body in turn: same range, same elevation,
        # same beams. Returns are attributed by MASK_R, so a beam that clips a
        # neighbour on its way past is not credited to the body it was aimed at.
        allp, alld = [], []
        for b in bodies:
            c = b["centre"]
            aim(orbit, c + off, c)
            pts, cols = orbit.scan_rgbd(renderer, scene)
            live["counts"][b["name"]] = 0
            if not len(pts):
                continue
            d = pts - c.astype(np.float32)
            m = (d * d).sum(1) <= MASK_R * MASK_R
            if not m.any():
                continue
            hit = cols[m, 0] >= THRESH
            live["counts"][b["name"]] = int(hit.sum())
            allp.append(pts[m])
            alld.append(hit)
        paint_cloud(cloud_geom, cap,
                    np.concatenate(allp) if allp else np.zeros((0, 3), np.float32),
                    np.concatenate(alld) if alld else np.zeros(0, bool))

        if controls is not None:
            controls.update()
        renderer.render(scene, camera)
        if ui is not None:
            controls.enabled = not ui.want_capture_mouse
            ui.render(draw_ui)

        frames += 1
        if not shot_written and frames >= 12:
            png = SHOT or os.path.join(SHOT_DIR, "lidar_sculpt_finale.png")
            os.makedirs(os.path.dirname(png) or ".", exist_ok=True)
            renderer.save_frame(scene, camera, png)
            print(f"    wrote {png}")
            shot_written = True
        if frames % 60 == 0:
            print("    az %5.1f  " % (live["az"] % 360.0)
                  + "  ".join(f"{b['name']} {live['counts'][b['name']]:4d}"
                              for b in bodies))
        if HEADLESS:
            if frames >= max(24, FRAME_BUDGET):
                break
        else:
            if not canvas.animate_once(lambda: None):
                break
            if FRAME_BUDGET and frames >= FRAME_BUDGET:
                break


# --------------------------------------------------------------------------- #
#  --sweep: the bill for optimising against one pose set.
# --------------------------------------------------------------------------- #
def run_sweep(shape, geom, det, renderer, canvas, camera):
    print("\n  optimising all three shapes first (this is the same run as the "
          "demo, headless)")
    base_soft, base_hard, _, _ = det.score()
    xs = {"sphere": np.zeros(shape.n_param)}
    for mode in ("stealth", "conspicuous"):
        run = Run(shape, geom, det, mode, base_soft, SEED)
        while run.step():
            pass
        xs[mode] = run.best_x.copy()
        print(f"    {mode:<12}{run.ratio:6.3f} x sphere")

    def publisher(x):
        def go():
            pos, _ = shape.build(x)
            geom.update_attribute("position", pos)
            geom.update_attribute("normal", shape.normals(pos))
            renderer.render(SCENE, CAMERA)
        return go

    pubs = [(n, publisher(xs[n])) for n in ("sphere", "stealth", "conspicuous")]
    print()
    azimuth_table(det, np.zeros(3), pubs, R_NEAR, el=0.0, n_az=12)

    floor = cli_arg("--range-floor", 6.0, float)
    print(f"\n  detection RANGE -- the furthest stand-off at which the mean "
          f"detected count over 8 azimuths still reaches {floor:.0f}:")
    out = {}
    for name, pub in pubs:
        out[name] = detection_range(det, np.zeros(3), pub, floor)
        print(f"      {name:<14}{out[name]:6.1f} m")
    print(f"\n  HEADLINE: sphere seen out to {out['sphere']:.1f} m, "
          f"stealth {out['stealth']:.1f} m, conspicuous {out['conspicuous']:.1f} m")
    return out


# --------------------------------------------------------------------------- #
#  --selftest
# --------------------------------------------------------------------------- #
def selftest_verdict(shape, det, results, base_soft, publish):
    ok = True

    def check(name, passed, detail):
        nonlocal ok
        ok = ok and passed
        print(f"    {'PASS' if passed else 'FAIL'}  {name}: {detail}")

    print("\n  --- SELFTEST " + "-" * 50)
    s_soft, s_hard, s_x, s_err = results["stealth"]
    c_soft, c_hard, c_x, c_err = results["conspicuous"]

    check("stealth hides", s_soft <= STEALTH_GATE * base_soft,
          f"{s_soft / base_soft:.3f} x sphere (gate <= {STEALTH_GATE:.2f})")
    check("conspicuous shouts", c_soft >= CONSP_GATE * base_soft,
          f"{c_soft / base_soft:.3f} x sphere (gate >= {CONSP_GATE:.2f})")
    check("volume held", max(s_err, c_err) < 0.005,
          f"worst {100.0 * max(s_err, c_err):.4f} % (gate < 0.5 %)")

    # Determinism, in two parts: the sensor must replay a scan of the SAME mesh
    # bit for bit (zero noise model, same beam order), and the optimiser must
    # propose the same population from the same seed. Together those are what
    # make a reported score a property of the shape.
    publish(s_x)
    a = det.score()[:2]
    b = det.score()[:2]
    check("scan replays", a == b, f"soft/hard {a} twice")

    publish(s_x)
    c = det.score()[:2]
    check("republish replays", a == c,
          f"{a} then {c} after a rebuild of the same parameters")

    es_a = CMAES(shape.n_param, SIGMA0, SEED, POP)
    es_b = CMAES(shape.n_param, SIGMA0, SEED, POP)
    check("optimiser seeded", np.array_equal(es_a.ask(), es_b.ask()),
          "two CMA-ES at one seed propose the same population")

    check("stealth is faceted",
          shape.facet_rms(shape.build(s_x)[0]) > 1.5 * shape.facet_rms(
              shape.build(np.zeros(shape.n_param))[0]),
          f"dihedral RMS {shape.facet_rms(shape.build(s_x)[0]):.2f} deg vs "
          f"sphere {shape.facet_rms(shape.build(np.zeros(shape.n_param))[0]):.2f} deg")
    print("  " + "-" * 66)
    print("  SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# The gates are set from MEASURED behaviour at the selftest's budget (160
# generations x 20), with margin, and not from the plan's guesses. The plan
# asked for stealth <= 0.35x. It is not reachable: the grazing rim described in
# the docstring is a floor the engine's own BRDF puts under every closed body,
# and a 200-generation run lands at 0.416x and is still creeping down slowly.
# The selftest budget measures 0.436, so 0.55 is a gate it clears with room and
# a broken run would not.
STEALTH_GATE = cli_arg("--gate-stealth", 0.55, float)
CONSP_GATE = cli_arg("--gate-conspicuous", 1.40, float)


if __name__ == "__main__":
    main()
