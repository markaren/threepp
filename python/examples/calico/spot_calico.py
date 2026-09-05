"""Spot walks a real place: the Calico Tanks Gaussian-splat scan.

The scene is ONE asset. `tp.SplatCloud.from_sog_lod` streams the scan (7 LOD
levels, 19.3M splats at level 0) and that is what the primary camera draws.
`tp.bake_surface` fuses a triangle surface out of the same splats along the
trail; `tp.make_sensor_mesh` puts those triangles on the renderer's sensor-only
layer and PhysX takes the very same mesh as a static trimesh. So one object is

  * the ground Spot's feet stand on (PhysX),
  * the ground the forward DepthSensor measures (the lidar trace sees the
    sensor-only layer once set_sensor_only_surfaces(True)),
  * and never drawn in the picture -- the real splats are what you see.

    set PYTHONPATH=C:/dev/threepp/python
    py -3.14 python/examples/calico/spot_calico.py                 # interactive
    py -3.14 python/examples/calico/spot_calico.py --auto          # auto-walk the spine
    py -3.14 python/examples/calico/spot_calico.py --shot out.png --cam lookback --walk-m 8
    py -3.14 python/examples/calico/spot_calico.py --verify-frame v.png

Controls: W/S = fwd/back  A/D = strafe  Q/E = turn  |  R = reset  |  G = auto-walk
          mouse = orbit/zoom.  Close the window to exit (killing it loses the
          Vulkan pipeline cache and the next start pays the ~95 s compile again).

FRAME, decided once (plans/calico-splat-demo.md):
  the scan is Y-DOWN like the COLMAP capture it came from. `rotation.x = pi`
  undoes that and gives the "post-flip Y-up" frame every SuperSplat pose and
  all of WP0's shots.json is quoted in. The Spot stack on top is Z-UP (gravity
  (0,0,-9.81), camera.up = (0,0,1)), so the cloud takes one MORE quarter turn:
  rotation.x = pi + pi/2. The composite maps

      (x, y, z)_yup  ->  (x, -z, y)_zup

  and then a translation puts WP0's spawn at the world origin with its floor at
  z = 0. `to_world()` below is that map; every waypoint, camera and bake pose in
  this file goes through it. bake_surface emits WORLD-space vertices, so the
  bake happens AFTER the cloud has its final transform and the sensor mesh is
  added at the scene ROOT, never under the cloud.
"""
import argparse, json, math, os, sys, time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPOT = os.path.join(os.path.dirname(_HERE), "spot")
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))       # repo/python
sys.path.insert(0, _SPOT)
sys.path.insert(0, os.path.join(_SPOT, "scratch_distillation"))   # scratch_clock / scratch_env

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import threepp as tp
from threepp.rl import load_policy
from spot_deploy import (build_spot, fetch_assets, _quat_to_R,
                         default_q, isaac_to_add, add_to_isaac, ACTION_SCALE, Z0)
from spot_depth_scan import ForwardDepthScanner
from calico_slam_pip import SlamSurface, DepthPip
import calico_collider as CC
import calico_look
from spot_terrain_env import VX_HI
from scratch_env import STIFF_GAINS
from scratch_clock import GAIT_PERIOD
from _common import v2_obs
# The bench report is spot_slam's; no reason to own a second copy.
from spot_slam import _bench_report, BENCH_WARMUP, BENCH_PHASES

_SCANSTATS = bool(os.environ.get("SPOT_SCANSTATS"))

DEFAULT_ASSET = os.environ.get("THREEPP_CALICO_ASSET",
                               os.path.expanduser("~/.cache/threepp/splats/calico_tanks"))
SHOTS_JSON = os.path.join(_HERE, "shots.json")

SENSOR_W, SENSOR_H = 128, 96
SENSOR_FAR = 8.0
SCAN_EVERY = 3          # depth scan every N frames (~17 Hz), pipelined fire/harvest
MAP_CELL   = 0.15       # elevation-map cell, matches spot_slam
MC_FRAMES  = 90         # marching-cubes SLAM rebuild every N rendered frames
PIP_EVERY  = 3          # depth PiP readback every N frames (a device wait each time)
PIP_W, PIP_H = 224, 168 # secondary-view resolution; drawn at PIP_SCALE:1

# Bake recipe PROVEN by the judge after WP1 (plans/calico-splat-demo.md): explicit
# poses along the spine beat every built-in pose_set, whose Interior stations sit at
# the cloud's robust centre 60 m away and never look at the trail at all.
BAKE_EYE_UP   = 1.6     # m above each waypoint
BAKE_SIDE_M   = 4.0     # side looks 4 m left/right...
BAKE_SIDE_UP  = 0.5     # ...at +0.5 m
BAKE_SIDE_EVERY = 3     # ...on every third waypoint
BAKE_FOV      = 75.0
BAKE_VOXEL    = 0.05
BAKE_MAX_DEPTH = 15.0


# ── frame ─────────────────────────────────────────────────────────────────────
class Frame:
    """WP0's post-flip Y-up frame -> this demo's Z-up world.

    (x, y, z)_yup -> (x, -z, y)_zup + t, with t chosen so WP0's spawn is at the
    origin and the floor under it is z = 0.
    """

    def __init__(self, shots):
        sx, sy, sz = shots["spawn"]["position"]
        floor_y = float(shots["spawn"]["floor_y"])
        # raw map of the spawn, then the offset that sends it to (0, 0, 0)
        self.t = np.array([-sx, sz, -floor_y], np.float64)

    def __call__(self, p):
        return (float(p[0]) + self.t[0], -float(p[2]) + self.t[1], float(p[1]) + self.t[2])

    def dir(self, d):
        """A DIRECTION carries no translation."""
        return (float(d[0]), -float(d[2]), float(d[1]))


def _v3(p):
    return tp.Vector3(float(p[0]), float(p[1]), float(p[2]))


# ── bake poses ────────────────────────────────────────────────────────────────
def spine_bake_poses(wp):
    """The judged recipe. `wp` is the spine in WORLD coordinates, (N,3)."""
    poses = []
    up = tp.Vector3(0, 0, 1)
    n = len(wp)
    for i, w in enumerate(wp):
        eye = np.array([w[0], w[1], w[2] + BAKE_EYE_UP])
        nxt = wp[min(i + 1, n - 1)]
        prv = wp[max(i - 1, 0)]
        if i + 1 < n:
            poses.append(tp.BakePose(_v3(eye), _v3(nxt), up, BAKE_FOV))
        if i > 0:
            poses.append(tp.BakePose(_v3(eye), _v3(prv), up, BAKE_FOV))
        # straight down: the up vector has to be HORIZONTAL or look_at degenerates
        poses.append(tp.BakePose(_v3(eye), _v3(w), tp.Vector3(-1, 0, 0), BAKE_FOV))
        if i % BAKE_SIDE_EVERY == 0:
            tan = np.array(nxt, float) - np.array(prv, float)
            tan[2] = 0.0
            nrm = np.linalg.norm(tan)
            tan = tan / nrm if nrm > 1e-6 else np.array([1.0, 0.0, 0.0])
            left = np.array([-tan[1], tan[0], 0.0])
            for s in (+1.0, -1.0):
                tgt = np.array(w, float) + left * (s * BAKE_SIDE_M) + [0, 0, BAKE_SIDE_UP]
                poses.append(tp.BakePose(_v3(eye), _v3(tgt), up, BAKE_FOV))
    return poses


def lattice_bake_poses(cells, spacing=2.0):
    """Bake stations on a lattice across WP0's walkable region. `cells` is (N,3) world.

    The spine set walks a LINE and measures a ribbon about 2.5 m wide (97.5 % of the
    grid cells within 1.25 m of the spine come from real samples; at 3 m that is
    69 % and at 5 m it is 45 %). Everything else in the height grid is inpainted
    from the nearest cell that WAS seen, which is a fine way to avoid a hole and a
    poor way to describe rock. A user with a keyboard leaves the ribbon at once, so
    the bake has to leave it too.

    One station per `spacing` lattice cell -- the walkable cell nearest that cell's
    centre, chosen by an argmin over a fixed ordering, so the set is deterministic.
    Each station gets the SAME look set the spine stations get: straight down, and
    four horizontal looks 4 m out at +0.5 m (the four compass directions rather than
    a tangent's left/right, because a lattice point has no tangent).
    """
    P = np.asarray(cells, np.float64)
    if len(P) == 0:
        return []
    k = np.floor(P[:, :2] / float(spacing)).astype(np.int64)
    key = k[:, 0] * 100003 + k[:, 1]
    ctr = (k + 0.5) * float(spacing)
    d2 = ((P[:, :2] - ctr) ** 2).sum(axis=1)
    order = np.lexsort((d2, key))
    ks = key[order]
    first = np.concatenate([[True], ks[1:] != ks[:-1]])
    stations = P[order][first]
    poses = []
    up = tp.Vector3(0, 0, 1)
    for w in stations:
        eye = np.array([w[0], w[1], w[2] + BAKE_EYE_UP])
        poses.append(tp.BakePose(_v3(eye), _v3(w), tp.Vector3(-1, 0, 0), BAKE_FOV))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tgt = np.array([w[0] + dx * BAKE_SIDE_M, w[1] + dy * BAKE_SIDE_M,
                            w[2] + BAKE_SIDE_UP])
            poses.append(tp.BakePose(_v3(eye), _v3(tgt), up, BAKE_FOV))
    return poses, stations


# ── the LOD frame-time budget ─────────────────────────────────────────────────
class LodBudget:
    """A frame-time budget wrapped around `select_lod`.

    `select_lod` is a QUALITY policy: it keeps the coarsest resident level whose
    splats still cover the picture at ~`target_splats_per_pixel`, and it has no
    idea what that costs. In this scene the cost is wildly asymmetric. Standing
    at the spawn and looking down the open wash (-x) submits 1.07 M splats and
    renders in 15 ms; turning round to face the brushy end (+x) puts EVERY chunk
    of level 1 inside the frustum -- 4.81 M splats, 71 ms. Same policy, same spp,
    4.7x the frame. Raising spp makes it worse (spp 16 picks level 0: 18 M
    submitted, 203 ms), and the far plane changes nothing: the tile rasterizer
    does not clip chunks by camera far.

    The one lever that biases that policy coarser without touching C++ is the
    VIEWPORT HEIGHT it is told about. The footprint test scales with it, so
    telling it the picture is 0.7x as tall makes it step down a level -- exactly
    what dropping render_scale to 0.25 did by accident (level 2, 1.2 M, 27 ms).

    So: hold a scale in [0.25, 1], measure the render's wall time, and move the
    scale on it. Wall time is the honest signal here; `frame_timings.gpu_total_ms`
    read 22 ms on a frame whose wall time was 71 ms (it covers the deferred passes
    it instruments, not the splat submit-and-blend), so a controller fed from the
    GPU counter would never see the stall it exists to remove.

    Feedback (deliberately asymmetric, the way any drop-quality-fast controller
    is): over budget by more than TOL, multiply by DOWN at once; under LOW of the
    budget for HOLD_UP consecutive frames, divide by DOWN. `select_lod`'s own
    1.25 hysteresis sits underneath, and a COOLDOWN after every change stops the
    EMA's lag from spending three steps of scale on one stall.
    """

    EMA_N = 10          # frames in the smoothing window
    TOL = 1.15          # over budget by more than this -> coarser, now
    LOW = 0.85          # under this much of the budget -> a candidate for finer.
                        # 0.60 was the first try and it RATCHETS: about 10 ms of
                        # this frame is base cost (deferred shade, TAA/FSR, the
                        # robot, the SLAM mesh) and does not move with the splat
                        # count, so a settled 13 ms frame never reaches 0.6 x 18
                        # and the controller never gives quality back -- it sat
                        # on the coarsest resident level for 73 % of a revolution
                        # with 5 ms of its own budget unspent.
    HOLD_UP = 20        # ...for this many consecutive frames
    DOWN = 0.7          # the multiplicative step, both ways
    MIN_SCALE = 0.25    # and no lower: see `cap` for what happens past it
    MAX_STEPS = 4       # steps of DOWN a single over-budget frame may take
    MIN_CAP = 250_000   # splats; below the coarsest resident level nothing helps
    CAP_OFF = 30_000_000    # a cap above the whole cloud is no cap at all
    PAYOFF = 0.90       # a coarsening must buy at least 10 % or it is undone
    LOCKOUT = 600       # frames of silence after a change that bought nothing
    COOLDOWN = 5        # frames to let the EMA re-seed after a change
    UP_COOLDOWN = 45    # ...longer after going finer, which is the risky direction
    HOLD_UP_MAX = 160   # backoff ceiling (see `up_hold`)

    def __init__(self, budget_ms):
        self.budget = float(budget_ms)
        self.reset()

    def reset(self, budget_ms=None):
        if budget_ms is not None:
            self.budget = float(budget_ms)
        self.scale = 1.0
        self.ema = None
        self.n_low = 0
        self.cool = 0
        self.changes = 0
        self.last_ms = 0.0
        self.floor = None       # cheapest frame seen: with vsync this IS the cap
        self.up_hold = self.HOLD_UP
        self.since_up = 10 ** 9
        self.cap = None         # splats; None = whatever select_lod asked for
        self.last_submitted = 0
        self.pending = None     # (scale, cap, ema) to judge the last coarsening by
        self.lock = 0           # frames left of a no-payoff lockout
        self.enforced = 0       # frames where the cap actually replaced the ranges
        self.reverts = 0

    @property
    def on(self):
        return self.budget > 0.0

    def viewport(self, vh):
        """The height to hand `select_lod` this frame."""
        if not self.on:
            return int(vh)
        return max(1, int(round(float(vh) * self.scale)))

    def note(self, ms, submitted=None):
        """Feed one measured render wall time in ms, and what that frame drew."""
        self.last_ms = float(ms)
        if not self.on:
            return
        if submitted is not None and submitted > 0:
            self.last_submitted = int(submitted)
        a = 2.0 / (self.EMA_N + 1.0)
        self.ema = ms if self.ema is None else self.ema + a * (ms - self.ema)
        self.floor = ms if self.floor is None else min(self.floor, ms)
        self.since_up += 1
        if self.cool > 0:
            self.cool -= 1
            return
        # Did the last coarsening actually buy anything? A frame is only splat
        # bound some of the time -- at 1600x1000 this scene costs ~30 ms with
        # deferred shading, TAA/FSR, the robot and the SLAM mesh before a single
        # splat is drawn -- and against a budget it cannot reach, a controller
        # with no payoff test walks all the way down to the coarsest level and
        # sits there, having bought nothing and spent the whole picture. So:
        # judge the change, undo it if it did not pay, and go quiet for a while.
        if self.pending is not None:
            prev_scale, prev_cap, prev_ema = self.pending
            self.pending = None
            cur = self.ema
            if cur is not None and cur > self.PAYOFF * prev_ema:
                self.scale, self.cap = prev_scale, prev_cap
                self.reverts += 1
                self.lock = self.LOCKOUT
                self.ema = None
                print(f"[lod] budget guard STANDS DOWN: coarsening bought "
                      f"{100.0 * (1.0 - cur / prev_ema):.0f}% ({prev_ema:.1f} -> "
                      f"{cur:.1f} ms), so this frame is not splat bound -- "
                      f"reverting to scale {prev_scale:.2f} and going quiet for "
                      f"{self.LOCKOUT} frames")
                return
        if self.lock > 0:
            self.lock -= 1
            return
        # Under budget, OR pinned at the presentation cap. The second clause is
        # there because interactive runs are vsync'd: render() blocks in present,
        # so a frame with 6 ms of GPU work still measures ~16.7 ms and the plain
        # "< 0.6 * budget" test would never fire -- the scale would ratchet
        # coarse and stay there for the rest of the session.
        cheap = (self.ema < self.LOW * self.budget
                 or (self.floor is not None and self.ema <= self.floor * 1.10
                     and self.ema <= self.budget))
        if self.ema > self.TOL * self.budget:
            self.n_low = 0
            # One step per decision was measured to be too slow to matter: facing
            # +x costs 200 ms a frame, and five 12-frame steps down is seven
            # SECONDS of stall before the scale arrives. Take as many steps as
            # the overshoot is worth, at once; cost is near-linear in the splats
            # submitted, so the overshoot ratio in DOWN-sized steps is the honest
            # first guess.
            k = 1
            if self.ema > 2.0 * self.TOL * self.budget:
                k = int(min(self.MAX_STEPS,
                            math.ceil(math.log(self.ema / (self.TOL * self.budget))
                                      / math.log(1.0 / self.DOWN))))
            new = max(self.MIN_SCALE, self.scale * self.DOWN ** k)
            # ...and the cap, which is the lever that still bites at the scale's
            # floor. Cost is very nearly linear in the splats actually submitted
            # (measured: 1.07 M = 15 ms, 4.81 M = 71 ms, 18.0 M = 203 ms, i.e.
            # ~15 ms per million), so "draw less than you just drew" is a
            # well-posed instruction in splats even where it is not in viewport
            # height. It has to exist because the height lever is NOT monotone:
            # past roughly 0.25 the per-node selection fragments into more than
            # the 64 submit ranges the backend takes, selectLodPerNode bridges
            # the gaps to fit, and a bridged range carries every splat between
            # its ends -- so pushing the height further down put 18 M splats back
            # on the GPU and made the frame WORSE than no policy at all.
            cap = None
            if self.last_submitted > 0:
                cap = max(self.MIN_CAP, int(self.last_submitted * self.DOWN))
                cap = cap if self.cap is None else min(self.cap, cap)
            if new != self.scale or cap != self.cap:
                # If we only just went finer and are already back over budget,
                # this is the boundary case that oscillates. Back off: demand
                # twice as long a quiet spell before trying finer again.
                if self.since_up < 4 * self.HOLD_UP:
                    self.up_hold = min(self.HOLD_UP_MAX, self.up_hold * 2)
                if self.scale >= 1.0 and self.cap is None:
                    print(f"[lod] budget guard ENGAGES: {self.ema:.1f} ms "
                          f"smoothed over a {self.budget:.0f} ms budget, "
                          f"{self.last_submitted / 1e6:.2f} M splats submitted")
                self.pending = (self.scale, self.cap, self.ema)
                self.scale = new
                self.cap = cap
                self.changes += 1
                self.cool = self.COOLDOWN
                self.ema = None     # the old regime's frames say nothing now
        elif cheap:
            self.n_low += 1
            if self.n_low >= self.up_hold:
                self.n_low = 0
                new = min(1.0, self.scale / self.DOWN)
                cap = None if self.cap is None else int(self.cap / self.DOWN)
                if cap is not None and cap > self.CAP_OFF:
                    cap = None
                if new != self.scale or cap != self.cap:
                    self.scale = new
                    self.cap = cap
                    self.changes += 1
                    self.cool = self.UP_COOLDOWN
                    self.since_up = 0
                    self.ema = None
        else:
            self.n_low = 0


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=DEFAULT_ASSET)
    ap.add_argument("--model", default=os.path.join(_SPOT, "spot_steps.pt"))
    ap.add_argument("--shot", metavar="PNG")
    ap.add_argument("--cam", default="low_following",
                    help="shots.json camera for --shot (establishing/low_following/topdown/lookback),"
                         " 'follow' for a chase pose behind the robot, 'trackside' for WP0's"
                         " low_following geometry rebuilt at whatever waypoint we spawned on,"
                         " or 'back' for the spawn +1.5 m looking +x (the expensive view)")
    ap.add_argument("--strafe-m", type=float, default=0.0,
                    help="--shot: strafe sideways off the spine this far before capturing")
    ap.add_argument("--walk-m", type=float, default=0.0,
                    help="--shot: auto-walk this many metres along the spine before capturing")
    ap.add_argument("--shot-size", default="1600x1000")
    ap.add_argument("--verify-frame", metavar="PNG",
                    help="render the cloud alone at WP0's lookback pose and exit "
                         "(the one-time check that the Y-up -> Z-up map is right)")
    ap.add_argument("--auto", action="store_true", help="auto-walk the spine from the start")
    ap.add_argument("--spp", type=float, default=4.0,
                    help="select_lod target_splats_per_pixel. 4, not 8: on the "
                         "per-node selector (splats::selectLodPerNode, which "
                         "select_lod now dispatches to because this asset carries "
                         "the SSOG tree) the near field stays at level 0 either "
                         "way, and 4 costs 20.5 ms looking +x where 8 costs 43.8.")
    ap.add_argument("--near-lod-m", type=float, default=0.0,
                    help="within this distance of the nearest spine waypoint, force the "
                         "FINEST resident level, every chunk, no frustum trim. MEASURED at "
                         "1200x720 render_scale 0.5: 5.7 fps, 173 ms of it CPU-side submit. "
                         "select_lod at a high --spp keeps the chunk culling and is the "
                         "default; 0 disables the force.")
    ap.add_argument("--lod-budget-ms", type=float, default=0.0, metavar="MS",
                    help="frame-time budget for the splat pass, in milliseconds of "
                         "measured RENDER wall time. OFF by default (0): the per-node LOD "
                         "selector already holds the brush wall at ~16 ms, and the guard's "
                         "coarser level is a visible loss (woolly brush). 25 is a sane value "
                         "when you turn it on; 0 disables the feedback and leaves "
                         "select_lod alone). See LodBudget: the budget does not "
                         "pick levels itself, it scales the viewport height that "
                         "select_lod is told about, which is the only lever that "
                         "biases that policy coarser from Python.")
    ap.add_argument("--bench-spin", type=float, default=0.0, metavar="SECONDS",
                    help="headless A/B bench: stand at the spawn +1.5 m and turn the "
                         "camera through 360 degrees over SECONDS, once with the LOD "
                         "budget OFF and once ON, reporting frame-time p50/p95/max, "
                         "the level histogram and the level-switch rate for each.")
    ap.add_argument("--no-bake", action="store_true", help="skip the bake (splats only, no physics ground)")
    ap.add_argument("--flat", action="store_true",
                    help="CONTROL: replace the baked collider with a flat plane at the spawn's "
                         "floor. The splats still draw. If Spot cannot stand HERE the fault is "
                         "in this script, not in the surface.")
    ap.add_argument("--collider", default="grid", choices=("bake", "grid", "heightfield"),
                    help="what PhysX stands on. 'bake' is the raw marching-cubes shell -- the "
                         "SENSOR surface, and a bad collider: it has holes where the bake's "
                         "poses disagreed, it is a zero-thickness sheet a fast foot plant "
                         "tunnels through, and it fuses twigs into spikes that kick a foot "
                         "into the air. 'grid' derives a 2.5D height field from those same "
                         "triangles (calico_collider.build_height_grid), watertight by "
                         "construction, and hands PhysX a regular-grid trimesh. "
                         "'heightfield' is the same field as a real PxHeightField, which also "
                         "has THICKNESS. The sensor mesh is unchanged in every case -- what "
                         "the scanner sees is still the bake, brush and all.")
    ap.add_argument("--grid-area", default="full", choices=("corridor", "full"),
                    help="how far the height-grid collider reaches. 'corridor' is what "
                         "WP6 phase 1 shipped: the spine's bounding box plus 1.25 m plus "
                         "1 m of margin, i.e. x -17.9..2.3, y -2.3..5.9. The BAKE covers "
                         "-27.2..9.6 by -14.4..10.4, so a user driving with W/A/S/D left "
                         "the collider entirely within a couple of metres and fell into "
                         "nothing -- which is why the grid appeared to change nothing "
                         "interactively. 'full' builds over the bake's own AABB and adds "
                         "an edge-clamped skirt past it. 'corridor' is kept as the A/B.")
    ap.add_argument("--grid-map", metavar="PNG",
                    help="save a top-down map of the collider grid (height as colour, "
                         "inpainted cells hatched, spine and walkable region overlaid)")
    ap.add_argument("--grid-out", metavar="NPZ",
                    help="save the collider grid (with its measured mask) for offline work")
    ap.add_argument("--no-skirt", action="store_true",
                    help="CONTROL: no edge-clamped skirt past the grid, so walking off "
                         "the scanned area is a cliff into nothing again")
    ap.add_argument("--bake-poses", default="spine+lattice",
                    choices=("spine", "spine+lattice"),
                    help="'spine' is the 61-pose set judged after WP1: eye 1.6 m over each "
                         "spine waypoint, looking ahead/back/down plus side looks every "
                         "third waypoint. It measures a 2.5 m ribbon and leaves the rest "
                         "of the wash unmeasured, so the height grid out there is "
                         "inpainted guesswork. 'spine+lattice' adds a 2 m lattice of "
                         "stations across WP0's walkable region with the same look set.")
    ap.add_argument("--lattice-m", type=float, default=2.0,
                    help="lattice spacing for the extra bake stations")
    ap.add_argument("--spawn-wp", type=int, default=4,
                    help="spine waypoint index to spawn on. NOT 0: WP0's own spawn sits at the "
                         "brushy +x end where the bake is twigs over a shell with holes in it "
                         "-- probed floors there are +0.00..+0.07 while the baked vertices read "
                         "+0.275 (20 cm of brush), and the robot drops straight through. wp4 is "
                         "open wash rock: 8.03 m walked, zero falls.")
    ap.add_argument("--drive-test", metavar="SCRIPT", default=None,
                    choices=("full", "strafe3"),
                    help="headless INTERACTIVE-STYLE test: drive the robot with the same "
                         "velocity commands the W/A/S/D keys produce, on a fixed script, "
                         "and report base-z-minus-grid, roll, pitch and falls. 'full' is "
                         "the 90 s tour that leaves the corridor (fwd / strafe left / fwd "
                         "/ turn 90 and walk / strafe right / seeded random walk); "
                         "'strafe3' is the user's complaint reproduced: strafe sideways "
                         "off the spine and watch the base z.")
    ap.add_argument("--drive-seconds", type=float, default=90.0)
    ap.add_argument("--drive-seed", type=int, default=11)
    ap.add_argument("--drive-out", metavar="NPZ", default=None)
    ap.add_argument("--probe", action="store_true",
                    help="drop a ball on the baked collider and report where it rests")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--bench", type=int, default=0, metavar="N")
    # ---- WP3: the live SLAM surface and the depth picture-in-picture ----
    ap.add_argument("--no-slam", action="store_true",
                    help="skip the marching-cubes reconstruction of what the scanner has seen")
    ap.add_argument("--slam-mat", default="basic", choices=("basic", "standard"),
                    help="'basic' = unlit transparent -> Vulkan's raster overlay route, which is "
                         "the only path that actually BLENDS a solid mesh here and is excluded "
                         "from the traced scene (so the robot cannot map its own map). "
                         "'standard' = the deferred G-buffer, which has no blend: it comes out "
                         "OPAQUE and the depth sensor can see it. Diagnostic only.")
    ap.add_argument("--slam-opacity", type=float, default=0.62)
    ap.add_argument("--slam-lift", type=float, default=0.0,
                    help="metres to raise the SLAM surface off the rock it reconstructs")
    ap.add_argument("--mc-frames", type=int, default=MC_FRAMES)
    ap.add_argument("--no-pip", action="store_true",
                    help="skip the depth secondary view entirely (the A/B for its frame cost)")
    ap.add_argument("--cloud-update", action="store_true",
                    help="call SplatCloud.update() every frame. It is the GL path's per-frame "
                         "entry and runs a CPU counting sort over all 25.6M resident splats; "
                         "on Vulkan the tile rasterizer sorts on the GPU. Here for the A/B "
                         "that measured it at 178 ms of a 183 ms frame.")
    ap.add_argument("--pip-pitch", type=float, default=18.0,
                    help="PiP camera pitch below horizontal. The policy's own depth camera "
                         "pitches 40 deg (it is measuring the next footfall), which fills the "
                         "frame with the floor within a metre; the PICTURE wants the trail.")
    ap.add_argument("--slam-keep-above", type=float, default=0.5,
                    help="drop drawn SLAM geometry more than this far above the local floor "
                         "(brush and canyon wall). It stays in the VoxelGrid.")
    ap.add_argument("--pip-scale", type=int, default=2,
                    help="the PiP is drawn at this many window pixels per sensor pixel")
    # ---- WP4's look, adopted here (calico_look.py) ----
    ap.add_argument("--no-look", action="store_true",
                    help="CONTROL: keep WP2's placeholder lighting (flat hemisphere + a "
                         "sun pointing nowhere in particular, no sky, exposure 1.1) instead "
                         "of calico_look.apply_look. The A/B for the whole WP4 look.")
    ap.add_argument("--sky", default="proc", choices=("proc", "env", "none"),
                    help="calico_look sky: 'proc' (threepp's own, Z-up), 'env' (the scan's "
                         "own backdrop sphere -- a panorama band, see calico_look's docstring), "
                         "or 'none'")
    ap.add_argument("--no-shadows", action="store_true",
                    help="CONTROL: no contact shadows under the feet and body. The splats "
                         "receive nothing, so this is the whole ground-contact cue.")
    ap.add_argument("--exposure", type=float, default=None,
                    help="override calico_look.EXPOSURE (pinned; auto-exposure stays off)")
    # ---- WP5: film mode ----
    ap.add_argument("--film", metavar="OUT_DIR",
                    help="headless deterministic film render: fixed sim step, renderer sim_time "
                         "pinned to the physics clock, synchronous SLAM rebuilds, the shot list "
                         "in calico_film.py, a PNG sequence + sensors.csv + hashes.txt")
    ap.add_argument("--film-fps", type=int, default=30)
    ap.add_argument("--film-size", default="1600x900")
    ap.add_argument("--film-seed", type=int, default=7)
    ap.add_argument("--film-scale", type=float, default=1.0,
                    help="renderer render_scale for the film (1.0 = native)")
    ap.add_argument("--film-speed", type=float, default=0.42,
                    help="scale on the auto-walk's forward command for the film. "
                         "The walkable trail is 12.6 m and the film asks for ~28 s "
                         "of simulated walking, so the robot strolls at ~0.45 m/s "
                         "instead of the 1.1 m/s the waypoint follower defaults to.")
    ap.add_argument("--film-frames", type=int, default=0, metavar="N",
                    help="render only the first N frames of the shot list (hashes and "
                         "sensors.csv still written) -- the short loop for a "
                         "determinism A/B")
    ap.add_argument("--no-upscaler", action="store_true",
                    help="turn FSR off for the film. The upscaler's temporal pass is a "
                         "suspect for the frame-hash mismatch between two otherwise "
                         "bit-identical runs; TAA and the overlay MSAA stay on.")
    ap.add_argument("--film-shots", default="",
                    help="comma-separated shot names to render (default: all)")
    args = ap.parse_args()
    assert tp.HAS_PHYSX, "needs a PhysX-enabled threepp build"
    film = bool(args.film)
    if args.drive_test:
        args.no_slam = True
        args.no_pip = True
    spin_s = float(args.bench_spin)
    headless = bool(args.shot) or bool(args.verify_frame) or bool(args.drive_test) \
        or spin_s > 0.0
    bench = int(args.bench)
    # The film is a fixed-quality product: a controller that trades level for
    # frame time would make shot 3 look different from shot 1 for no reason the
    # viewer can see. The drive test never looks at its own picture.
    if film or args.drive_test:
        if args.lod_budget_ms > 0.0:
            print(f"[lod] budget disabled for "
                  f"{'--film' if film else '--drive-test'} (fixed quality)")
        args.lod_budget_ms = 0.0
    budget = LodBudget(args.lod_budget_ms)
    if film:
        # Everything downstream of here that could make one run differ from the
        # next: numpy's global RNG (nothing in this script draws from it, but the
        # policy's own init and torch's fallbacks do), torch's, and the thread
        # pools whose scheduling changes reduction order in a matmul.
        import random
        random.seed(args.film_seed)
        np.random.seed(args.film_seed)
        torch.manual_seed(args.film_seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_num_threads(1)

    with open(SHOTS_JSON) as f:
        shots = json.load(f)
    F = Frame(shots)
    spine = np.array([F(w) for w in shots["spine"]["waypoints"]], np.float64)
    _wp0 = max(0, min(int(args.spawn_wp), len(spine) - 2))
    if _wp0 == 0:
        spawn_fwd = np.array(F.dir(shots["spawn"]["forward"]))
    else:
        spawn_fwd = spine[_wp0 + 1] - spine[_wp0]
    spawn_yaw = math.atan2(spawn_fwd[1], spawn_fwd[0])
    SPAWN = spine[_wp0].copy()
    print(f"[frame] spawn on waypoint {_wp0} at "
          f"({SPAWN[0]:+.2f}, {SPAWN[1]:+.2f}, {SPAWN[2]:+.2f})")
    print(f"[frame] t = {tuple(round(v, 3) for v in F.t)}  "
          f"spine {len(spine)} waypoints  spawn_yaw = {math.degrees(spawn_yaw):.1f} deg")
    print(f"[frame] spine world x {spine[:,0].min():.2f}..{spine[:,0].max():.2f}  "
          f"y {spine[:,1].min():.2f}..{spine[:,1].max():.2f}  "
          f"z {spine[:,2].min():.2f}..{spine[:,2].max():.2f}")

    def shot_cam(name):
        c = shots["cameras"][name]
        return F(c["pos"]), F(c["look"])

    # ── canvas + renderer ─────────────────────────────────────────────────────
    if args.drive_test:
        _bw, _bh = 640, 400     # nothing is looked at; the render is only there to
                                # rebuild the TLAS the depth scan traces
    elif args.shot or args.verify_frame:
        _bw, _bh = (int(v) for v in args.shot_size.split("x"))
    elif film:
        _bw, _bh = (int(v) for v in args.film_size.split("x"))
    else:
        _bw, _bh = 1200, 720
    if bench and os.environ.get("SPOT_BENCH_SIZE"):
        _bw, _bh = (int(v) for v in os.environ["SPOT_BENCH_SIZE"].split("x"))
    _vsync = (not bench and spin_s <= 0.0) or bool(os.environ.get("SPOT_BENCH_VSYNC"))
    canvas = tp.Canvas("threepp - Spot on Calico Tanks", width=_bw, height=_bh,
                       antialiasing=4, headless=headless or film, vsync=_vsync)
    rend = tp.VulkanRenderer(canvas)
    rend.shadow_map_enabled = True
    rend.tone_mapping = tp.ToneMapping.ACESFilmic
    rend.tone_mapping_exposure = 1.1
    rend.render_scale = float(args.film_scale) if film else 0.5
    if bench and os.environ.get("SPOT_BENCH_KNOBS"):
        for kv in os.environ["SPOT_BENCH_KNOBS"].split(","):
            k, v = kv.split("=")
            cur = getattr(rend, k.strip())
            setattr(rend, k.strip(), float(v) if not isinstance(cur, bool) else bool(float(v)))
            print(f"[bench] knob {k.strip()} = {v}")

    # ── the scan ──────────────────────────────────────────────────────────────
    scene = tp.Scene()
    # ── the look (WP4, calico_look.py) ────────────────────────────────────────
    # What used to be here was a placeholder: a flat blue background, a broad
    # hemisphere and a sun aimed wherever looked plausible -- with the result the
    # judge named, that Spot read as pasted onto the scan. calico_look gives the
    # scene a sky, and ONE DirectionalLight pointing where the scan's own baked
    # sun points (fitted off the splats themselves, calico_look.sun_direction),
    # with the exposure pinned. --no-look puts the placeholder back for the A/B.
    _look = None
    if args.no_look:
        scene.background = tp.Background(0x9fb6cc)
        scene.add(tp.HemisphereLight(0xd8e6ff, 0x6b5a44, 0.55))
        sun = tp.DirectionalLight(0xfff4e0, 2.6)
        sun.position.set(12, -14, 22)
        sun.cast_shadow = True
        sun.set_shadow_frustum(-14, 14, 14, -14)
        sun.set_shadow_bias(-0.0005)
        scene.add(sun)
    else:
        _look = calico_look.apply_look(
            scene, rend, sky=args.sky, asset=args.asset,
            cloud_position=(float(F.t[0]), float(F.t[1]), float(F.t[2])),
            exposure=(calico_look.EXPOSURE if args.exposure is None
                      else float(args.exposure)))

    print(f"[splat] loading {args.asset} (dynamic LOD) ...")
    t0 = time.perf_counter()
    cloud = tp.SplatCloud.from_sog_lod(args.asset)
    cloud.rotation.x = math.pi + math.pi / 2          # Y-down capture -> Z-up world
    cloud.position.set(float(F.t[0]), float(F.t[1]), float(F.t[2]))
    scene.add(cloud)
    lod_levels = list(cloud.lod_levels)
    print(f"[splat] loaded in {time.perf_counter() - t0:.2f}s, "
          f"{len(lod_levels)} resident levels: {[l['count'] for l in lod_levels]}")

    w, h = canvas.size()
    camera = tp.PerspectiveCamera(50, w / max(h, 1), 0.05, 400)
    camera.up.set(0, 0, 1)

    # ---- the one-time frame check: WP0's lookback pose, cloud only ----
    if args.verify_frame:
        pos, look = shot_cam("lookback")
        camera.fov = 75.0
        camera.update_projection_matrix()
        camera.position.set(*pos)
        camera.look_at(*look)
        tp.select_lod(cloud, camera, h, target_splats_per_pixel=args.spp)
        rend.render(scene, camera)
        tp.select_lod(cloud, camera, h, target_splats_per_pixel=args.spp)
        rend.render(scene, camera)
        try:
            rend.save_frame(args.verify_frame)
        except TypeError:
            rend.save_frame(scene, camera, args.verify_frame)
        print(f"saved {args.verify_frame}  (compare by eye with shots/lookback.png)")
        return

    # ── physics + bake ────────────────────────────────────────────────────────
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.002, max_substeps=20)

    sensor_mesh = None
    surf_pos = None
    hgrid = [None]          # the height-grid collider, when --collider is not 'bake'
    skirt_body = [None]     # the edge-clamped ring past it
    if args.flat:
        _fg = tp.BufferGeometry()
        _q = [(-20, -20), (20, -20), (20, 20), (-20, -20), (20, 20), (-20, 20)]
        _fg.set_attribute("position", np.array(
            [[SPAWN[0] + a, SPAWN[1] + b, SPAWN[2]] for a, b in _q], np.float32))
        _flat = tp.Mesh(_fg, tp.MeshStandardMaterial())
        world.add_static_trimesh(_flat)
        print(f"[flat] CONTROL collider: 40x40 m plane at z = {SPAWN[2]:+.3f}")
    elif not args.no_bake:
        # Warm the pipeline once so the bake's own renders are not the first draw.
        camera.position.set(2.0, -2.0, 2.0)
        camera.look_at(-4.0, 1.0, 0.0)
        # Bake against the FINEST resident level, every chunk submitted. The bake
        # renders depth from each pose through the ordinary splat path, so whatever
        # select_lod left in submit_ranges is what it fuses: leave a coarse level in
        # there and the surface comes out thin and full of holes the robot drops
        # through, which is exactly what happened the first time.
        if lod_levels:
            cloud.submit_ranges = [(int(lod_levels[0]["base"]), int(lod_levels[0]["count"]))]
        if args.cloud_update:
            cloud.update(camera)
        t0 = time.perf_counter()
        rend.render(scene, camera)
        print(f"[splat] first render {time.perf_counter() - t0:.2f}s (shader compile included)")

        poses = spine_bake_poses(spine)
        n_spine = len(poses)
        n_lat = 0
        if args.bake_poses == "spine+lattice":
            cw = np.array([F(c) for c in shots["walkable_region"]["cells"]], np.float64)
            lat, stations = lattice_bake_poses(cw, args.lattice_m)
            poses = poses + lat
            n_lat = len(lat)
            print(f"[bake] + {len(stations)} lattice stations at {args.lattice_m} m over "
                  f"the walkable region (x {stations[:,0].min():+.1f}..{stations[:,0].max():+.1f}, "
                  f"y {stations[:,1].min():+.1f}..{stations[:,1].max():+.1f}) "
                  f"= {n_lat} extra poses")
        print(f"[bake] {len(poses)} poses ({n_spine} spine + {n_lat} lattice), "
              f"voxel={BAKE_VOXEL} max_depth={BAKE_MAX_DEPTH} ...")
        t0 = time.perf_counter()
        surface = tp.bake_surface(rend, cloud, poses=poses,
                                  voxel_size=BAKE_VOXEL, max_depth=BAKE_MAX_DEPTH)
        bake_s = time.perf_counter() - t0
        st = surface.stats
        print(f"[bake] {surface.triangle_count} triangles, {surface.vertex_count} vertices "
              f"in {bake_s:.2f}s")
        print(f"[bake]   depth_samples={st['depth_samples']} observed_voxels={st['observed_voxels']} "
              f"components={st['components']} culled={st['culled_components']}")
        print(f"[bake]   render={st['render_ms']:.0f}ms fuse={st['fuse_ms']:.0f}ms "
              f"mesh={st['mesh_ms']:.0f}ms")
        print(f"[bake]   aabb {tuple(round(v, 2) for v in (st['aabb_min'].x, st['aabb_min'].y, st['aabb_min'].z))}"
              f" .. {tuple(round(v, 2) for v in (st['aabb_max'].x, st['aabb_max'].y, st['aabb_max'].z))}")
        assert surface.triangle_count > 0, "empty bake -- see the stats above"

        # Did the bake actually cover the trail? The failure mode that costs an
        # hour is a mesh full of far canyon wall and a HOLE where the robot
        # stands, so report the surface height under every waypoint before
        # anything is built on top of it.
        P = surf_pos = surface.positions
        for i, wpt in enumerate(spine):
            near = P[(np.abs(P[:, 0] - wpt[0]) < 0.4) & (np.abs(P[:, 1] - wpt[1]) < 0.4)]
            if i == 0 or i % 4 == 0 or near.shape[0] == 0:
                zs = f"{near[:,2].min():+.2f}..{near[:,2].max():+.2f}" if near.shape[0] else "NONE"
                print(f"[bake]   wp{i:2d} ({wpt[0]:+6.2f},{wpt[1]:+6.2f},{wpt[2]:+5.2f}) "
                      f"verts={near.shape[0]:6d} z {zs}")

        # The bake is the SENSOR surface: added at the scene root, on the
        # sensor-only layer, never drawn in the picture. Unchanged, brush and all,
        # because that is what the scan looks like and what the scanner should see.
        sensor_mesh = tp.make_sensor_mesh(surface)
        scene.add(sensor_mesh)
        rend.set_sensor_only_surfaces(True)
        print(f"[bake] sensor mesh added at the scene root; "
              f"sensor_only_surfaces={rend.sensor_only_surfaces}")

        # The COLLIDER is a separate question, and the demo used to answer it with
        # the same triangles -- which is why the user watched Spot fall through the
        # ground and get kicked into the air. See --collider and calico_collider.py.
        if args.collider == "bake":
            body = world.add_static_trimesh(sensor_mesh)
            print(f"[collider] bake: the raw marching-cubes shell as a static trimesh "
                  f"({surface.triangle_count} triangles) = {body}")
        else:
            if args.grid_area == "corridor":
                _cw = CC.CORRIDOR_HALF_W
                _gb = (float(spine[:, 0].min()) - _cw, float(spine[:, 0].max()) + _cw,
                       float(spine[:, 1].min()) - _cw, float(spine[:, 1].max()) + _cw)
                _fc = CC.FLOOR_CELL
            else:
                # The WHOLE baked area, not the spine's corridor. The bake is what
                # was measured; anything narrower leaves the robot standing on
                # nothing the moment it steps sideways.
                _gb = CC.bake_bounds(surf_pos)
                # 0.5 m rather than 1 m: the local floor is a per-cell MINIMUM, and
                # over the whole scan the cells are no longer all wash floor. A 1 m
                # cell on a canyon flank has a metre of relief inside it, so its
                # minimum sits far below its own surface and the 0.30 m brush test
                # throws the upper half of the cell away. Halving the cell halves
                # that relief. On the wash it changes nothing measurable: the wp4
                # spawn moves by 0.0005 m.
                _fc = 0.5
            print(f"[collider] grid area '{args.grid_area}': "
                  f"x {_gb[0]:+.2f}..{_gb[1]:+.2f}  y {_gb[2]:+.2f}..{_gb[3]:+.2f}")
            hgrid[0] = CC.build_height_grid(surf_pos, surface.indices,
                                            bounds=_gb, floor_cell=_fc)
            if args.grid_map or args.grid_out:
                _M = CC.measured_mask(surf_pos, surface.indices, hgrid[0],
                                      floor_cell=_fc)
                if args.grid_out:
                    np.savez_compressed(args.grid_out, H=hgrid[0].H, x0=hgrid[0].x0,
                                        y0=hgrid[0].y0, cell=hgrid[0].cell, measured=_M)
                    print(f"[collider] grid saved to {args.grid_out}")
                if args.grid_map:
                    _cwm = np.array([F(c) for c in shots["walkable_region"]["cells"]])
                    CC.grid_map_png(args.grid_map, hgrid[0], measured=_M, spine=spine,
                                    walkable=_cwm,
                                    title=f"calico collider grid  area={args.grid_area}  "
                                          f"{hgrid[0].shape[0]}x{hgrid[0].shape[1]} @ "
                                          f"{hgrid[0].cell} m   measured "
                                          f"{100.0 * _M.mean():.1f}%  (dim+hatched = "
                                          f"inpainted)   cyan spine, magenta walkable")
                    print(f"[collider] map -> {args.grid_map}")
            _gs = hgrid[0].stats
            print(f"[collider]   {100.0 * _gs['frac_measured']:.1f}% of cells measured, "
                  f"{100.0 * _gs['frac_inpainted']:.1f}% inpainted from the nearest "
                  f"measured cell")
            if args.collider == "heightfield":
                if not hasattr(world, "add_static_heightfield"):
                    raise SystemExit("--collider heightfield needs a pyd with "
                                     "PhysxWorld.add_static_heightfield (build threepp_py)")
                body = world.add_static_heightfield(
                    hgrid[0].H.T, cell=hgrid[0].cell,
                    origin=tp.Vector3(hgrid[0].x0, hgrid[0].y0, 0.0))
                print(f"[collider] heightfield: PxHeightField {hgrid[0].shape} at "
                      f"{hgrid[0].cell} m, thickness 0.5 = {body}")
            else:
                body = world.add_static_trimesh(hgrid[0].to_mesh())
                print(f"[collider] grid: the height field as a regular-grid trimesh "
                      f"{hgrid[0].shape} at {hgrid[0].cell} m = {body}")
            # The skirt: the SAME field carried outward at its edge value, so a
            # robot driven off the scan gets a flat shelf instead of the void.
            # Disjoint from the grid in plan view -- the two together are one
            # ground surface, not two stacked ones.
            if not args.no_skirt:
                skirt_body[0] = world.add_static_trimesh(hgrid[0].skirt_mesh())
                _sb = hgrid[0].stats["skirt_bounds"]
                print(f"[collider] skirt: {hgrid[0].stats['skirt_tris']} triangles at "
                      f"{CC.SKIRT_CELL} m out to x {_sb[0]:+.1f}..{_sb[1]:+.1f} "
                      f"y {_sb[2]:+.1f}..{_sb[3]:+.1f} (edge-clamped) = {skirt_body[0]}")

        if args.probe:
            idx = surface.indices
            print(f"[probe] indices {idx.shape} {idx.dtype} min={idx.min()} max={idx.max()} "
                  f"verts={P.shape[0]} pos dtype={P.dtype}")
            world.remove(body)

            def drop(label, mesh):
                b = world.add_static_trimesh(mesh)
                ball = tp.Mesh(tp.SphereGeometry(0.15, 12, 8), tp.MeshStandardMaterial())
                ball.position.set(0.0, 0.0, 2.0)
                rb = world.add(ball, 200.0)
                for _ in range(200):
                    world.step(0.02)
                print(f"[probe] {label:<12} body={b is not None}  ball rest z = {rb.position.z:8.3f}")
                world.remove(rb)
                world.remove(b)

            _pg = tp.BufferGeometry()
            _pg.set_attribute("position", np.array([[-10, -10, 0.2], [10, -10, 0.2], [10, 10, 0.2],
                                                    [-10, -10, 0.2], [10, 10, 0.2], [-10, 10, 0.2]],
                                                   np.float32))
            drop("control-plane", tp.Mesh(_pg, tp.MeshStandardMaterial()))
            drop("sensor_mesh", sensor_mesh)
            drop("to_geometry", tp.Mesh(surface.to_geometry(), tp.MeshStandardMaterial()))
            soup = np.ascontiguousarray(P[idx.reshape(-1)], np.float32)
            gsoup = tp.BufferGeometry(); gsoup.set_attribute("position", soup)
            drop("soup", tp.Mesh(gsoup, tp.MeshStandardMaterial()))
            # RESTORE the collider that --collider asked for. This line used to read
            # `add_static_trimesh(sensor_mesh)` unconditionally, so `--collider grid
            # --probe` removed the height grid and put the raw BAKE back as the
            # ground -- the diagnostic quietly swapped in the very collider the grid
            # exists to replace. Every `drop()` above adds and removes its own body
            # inside the call, so after this line there is exactly ONE ground body
            # (plus the disjoint skirt).
            if args.collider == "bake":
                body = world.add_static_trimesh(sensor_mesh)
            elif args.collider == "heightfield":
                body = world.add_static_heightfield(
                    hgrid[0].H.T, cell=hgrid[0].cell,
                    origin=tp.Vector3(hgrid[0].x0, hgrid[0].y0, 0.0))
            else:
                body = world.add_static_trimesh(hgrid[0].to_mesh())
            print(f"[probe] restored the '{args.collider}' collider = {body}")

    _FEET = ((0.30, 0.17), (0.30, -0.17), (-0.30, 0.17), (-0.30, -0.17), (0.0, 0.0))

    def probe_floor(x, y, drop_from=1.2, r=0.06):
        """Drop a small hard ball at (x, y) and report what it lands ON.

        The vertex-max floor is a TRAP here: within 0.4 m of WP0's spawn the baked
        vertices run to +0.375 but a ball rests at +0.083, because those vertices are
        brush -- twig geometry the 5 cm bake fused above the real rock. Spawning the
        robot on the twigs drops it 30 cm onto spikes and it never gets up. The ball
        finds the surface PhysX will actually push the feet off.
        """
        b = tp.Mesh(tp.SphereGeometry(r, 10, 8), tp.MeshStandardMaterial())
        b.position.set(float(x), float(y), float(SPAWN[2]) + drop_from)
        rb = world.add(b, 500.0)
        for _ in range(160):
            world.step(0.01)
        z = float(rb.position.z) - r
        world.remove(rb)
        return z

    def surface_h(x, y, r=0.12, default=0.0):
        """Baked surface height at (x, y): the 90th percentile of vertices in a small
        square, which is the rock a foot rests on. The MAX over a 0.45 m box instead
        picks up the lip of the boulder next to the trail and spawns the robot half a
        metre in the air; the max over a 0.12 m box is still a spike magnet, so trim it."""
        if surf_pos is None:
            return default
        m = (np.abs(surf_pos[:, 0] - x) < r) & (np.abs(surf_pos[:, 1] - y) < r)
        return float(np.percentile(surf_pos[m, 2], 90.0)) if m.any() else default

    def footprint_h(x, y, yaw):
        """Highest floor over the four feet + base.

        On the height-grid collider this is a LOOKUP, not a probe, and the swap is
        the difference between standing and being thrown. probe_floor's ball ROLLS:
        it is a 6 cm sphere given 1.6 s on a real wash, and by the end it reports
        the bottom of whatever slope it landed on. At the wp0 spawn it read +0.059
        while the collider under the front feet is +0.35, so the robot was reset a
        third of a metre INSIDE its own ground and PhysX depenetrated it to
        (+13.4, -12.0, +9.97) -- ten metres up -- before the first policy step.
        The grid IS the collider, so ask the grid; the probe stays as the cross
        check and is still the only measurement available for --collider bake.
        """
        c, sn = math.cos(yaw), math.sin(yaw)
        pts = [(x + dx * c - dy * sn, y + dx * sn + dy * c) for dx, dy in _FEET]
        hs = [probe_floor(px, py) for px, py in pts]
        print("[spot] probed floor per foot: " + "  ".join(f"{v:+.3f}" for v in hs))
        if hgrid[0] is not None:
            gs = [float(hgrid[0].height_at(px, py)) for px, py in pts]
            print("[spot]   grid floor per foot: " + "  ".join(f"{v:+.3f}" for v in gs)
                  + f"   (grid max - probe max = {max(gs) - max(hs):+.3f} m)")
            return max(gs)
        return max(hs)

    # ── the floor, as a lookup ────────────────────────────────────────────────
    # The contact shadows need a floor height under a moving foot, every frame,
    # five times a frame. probe_floor is 160 physics steps and surface_h scans
    # every baked vertex, so neither can be called from the frame loop: bin the
    # baked vertices ONCE into a coarse XY grid (mean z per cell, which is the
    # local rock -- a foot-sized cell of a 5 cm bake has no room for a boulder
    # face to skew it), then shift the whole grid by the offset the BALL PROBE
    # measures at the spawn. That offset is the brush the bake fused above the
    # rock (WP2 found 20 cm of it at wp0), and it is the difference between a
    # shadow lying on the ground and one hovering over it.
    _FLOOR_CELL = 0.25

    def _floor_lookup():
        if hgrid[0] is not None:
            # The collider IS a height field: no need to bin vertices and guess.
            # The probe bias below comes out near zero, which is itself the check
            # that the grid sits where PhysX pushes the feet off.
            g = hgrid[0]
            return (lambda x, y: float(g.height_at(x, y)) + _floor_bias[0]), \
                   float(np.median(g.H))
        if surf_pos is None:
            return (lambda x, y: float(SPAWN[2])), 0.0
        P = surf_pos
        x0 = float(P[:, 0].min()) - _FLOOR_CELL
        y0 = float(P[:, 1].min()) - _FLOOR_CELL
        nx = int((P[:, 0].max() - x0) / _FLOOR_CELL) + 2
        ny = int((P[:, 1].max() - y0) / _FLOOR_CELL) + 2
        ix = np.clip(((P[:, 0] - x0) / _FLOOR_CELL).astype(np.int64), 0, nx - 1)
        iy = np.clip(((P[:, 1] - y0) / _FLOOR_CELL).astype(np.int64), 0, ny - 1)
        k = ix * ny + iy
        s = np.zeros(nx * ny, np.float64)
        c = np.zeros(nx * ny, np.float64)
        np.add.at(s, k, P[:, 2])
        np.add.at(c, k, 1.0)
        F = np.where(c > 0, s / np.maximum(c, 1.0), np.nan).reshape(nx, ny)
        med = float(np.nanmedian(F))

        def at(x, y):
            i = int(min(max((x - x0) / _FLOOR_CELL, 0), nx - 1))
            j = int(min(max((y - y0) / _FLOOR_CELL, 0), ny - 1))
            v = F[i, j]
            return (med if not np.isfinite(v) else float(v)) + _floor_bias[0]
        return at, med

    _floor_bias = [0.0]
    floor_at, _floor_med = _floor_lookup()

    h0 = footprint_h(SPAWN[0], SPAWN[1], spawn_yaw)
    if hgrid[0] is not None:
        # No bias: the floor lookup and the collider are the SAME height field, so
        # a shift here would only make the contact shadows disagree with the feet.
        _floor_bias[0] = 0.0
        print(f"[floor] the collider's own height grid is the floor lookup "
              f"(no bias; probe at the spawn centre reads "
              f"{probe_floor(SPAWN[0], SPAWN[1]) - floor_at(SPAWN[0], SPAWN[1]):+.3f} m "
              f"against it)")
    else:
        _floor_bias[0] = h0 - floor_at(SPAWN[0], SPAWN[1])
        print(f"[floor] baked-vertex grid ({_FLOOR_CELL} m cells) shifted by "
              f"{_floor_bias[0]:+.3f} m so it matches the probed floor at the spawn")
    print(f"[spot] probed floor under the spawn footprint: z = {h0:+.3f}  "
          f"(baked-vertex 90th pct says {surface_h(SPAWN[0], SPAWN[1]):+.3f})")

    # ── Spot ──────────────────────────────────────────────────────────────────
    assets = fetch_assets()
    model_path = args.model
    if not os.path.exists(model_path):
        _latest = os.path.splitext(model_path)[0] + "_latest.pt"
        model_path = _latest if os.path.exists(_latest) else model_path
    ac, norm, _meta = load_policy(model_path, device="cpu")
    ac.eval()
    print(f"[policy] {os.path.basename(model_path)}")

    art, meshes = build_spot(world, assets, gains=STIFF_GAINS)
    for m in meshes:
        m.cast_shadow = True
        scene.add(m)

    # Contact shadows: the splats are not geometry, so no shadow map and no
    # traced occluder reaches them -- the discs under the feet are the ONLY thing
    # that puts the robot on the ground rather than in front of it.
    shadows = None
    if not (args.no_shadows or args.no_look):
        shadows = calico_look.ContactShadows(scene, art, floor_fn=floor_at,
                                             floor_z=float(SPAWN[2]))
        print(f"[look] contact shadows on the probed baked floor "
              f"(sun elev {calico_look.SUN_ELEV_DEG:.0f} deg -> the discs slide "
              f"{abs(float(np.hypot(*shadows.slide))):.2f} m per metre of height)")

    def look_update(rs):
        """Per-frame half of the look: the shadow frustum rides the robot (the
        SUN does not turn -- Look.follow only moves the light's anchor along its
        own line), and each disc re-reads its foot's height off the floor grid."""
        if _look is not None:
            _look.follow((float(rs[0]), float(rs[1]), float(rs[2])))
        if shadows is not None:
            shadows.update()

    _hs = math.sin(spawn_yaw / 2.0)
    _spawn_quat = tp.Quaternion(0.0, 0.0, _hs, math.cos(spawn_yaw / 2.0))

    last_act = np.zeros(12, np.float32)
    ahead_cache = [np.zeros(45, np.float32)]
    h_here_cache = [0.0]

    def step_policy(cmd, gphi):
        obs = v2_obs(art, last_act, cmd, ahead_cache[0], h_here_cache[0], gphi)
        with torch.no_grad():
            obs_t = torch.from_numpy(obs)[None]
            if norm is not None:
                obs_t = norm.norm(obs_t)
            a = ac.act_mean(obs_t)[0].numpy()
        last_act[:] = a
        art.set_drive_targets((default_q + ACTION_SCALE * a)[add_to_isaac].astype(np.float32))
        world.step(0.02)

    def settle(n=80, policy_steps=120):
        """Drop onto the rock, then let the POLICY hold station.

        spot_slam settles with raw default-pose drive targets because its terrain is
        smooth. A 5 cm-voxel bake of gravel is not: the stiff default pose lands on a
        10 cm step under one foot and the robot tips over and never gets up. Holding
        the default pose for the touchdown and then handing the standing controller a
        zero velocity command is what keeps it upright here."""
        for _ in range(n):
            art.set_drive_targets(default_q[add_to_isaac].astype(np.float32))
            world.step(0.02)
        zero = np.zeros(3, np.float32)
        phi = 0.0
        for _ in range(policy_steps):
            step_policy(zero, phi)
            phi = (phi + 0.02 / GAIT_PERIOD) % 1.0

    art.reset(tp.Vector3(float(SPAWN[0]), float(SPAWN[1]), Z0 + h0 + 0.03), _spawn_quat)
    settle()
    rs0 = art.root_state()
    print(f"[spot] standing at ({rs0[0]:+.3f}, {rs0[1]:+.3f}, {rs0[2]:.3f}) "
          f"yaw={math.degrees(math.atan2(*_quat_to_R(rs0[3:7])[[1, 0], 0])):.1f} deg")

    # WP0's low_following geometry -- 1.2 m to the robot's right, 0.6 m above the
    # floor, level with the spawn -- rebuilt wherever we actually spawned. The eye
    # height comes from the PROBED floor at that exact spot, not from the spawn's,
    # because committing a pose without checking the column under it is how WP0's
    # own follow pose ended up inside a bush.
    _right = np.array([math.sin(spawn_yaw), -math.cos(spawn_yaw), 0.0])
    _tsxy = SPAWN[:2] + _right[:2] * 1.2
    trackside_eye = np.array([_tsxy[0], _tsxy[1],
                              probe_floor(_tsxy[0], _tsxy[1]) + 0.6])
    print(f"[cam] trackside eye {tuple(round(float(v), 3) for v in trackside_eye)} "
          f"(probed floor {trackside_eye[2] - 0.6:+.3f})")

    # ── camera + orbit controls ───────────────────────────────────────────────
    camera.position.set(SPAWN[0] - 3.0 * math.cos(spawn_yaw), SPAWN[1] - 3.0 * math.sin(spawn_yaw),
                        SPAWN[2] + 1.8)
    _init_tgt = np.array([SPAWN[0], SPAWN[1], SPAWN[2] + 0.4])
    _prev_tgt = [_init_tgt.copy()]
    controls = tp.OrbitControls(camera, canvas)
    controls.target.set(*_init_tgt.tolist())
    controls.enable_damping = True
    controls.damping_factor = 0.10
    controls.enable_pan = False
    controls.min_distance = 1.5
    controls.max_distance = 40.0
    controls.update()

    # ── depth scanner ─────────────────────────────────────────────────────────
    # The Vulkan DepthSensor traces the renderer's TLAS through the LIDAR path,
    # whose cull mask is 0xFF -- so set_sensor_only_surfaces(True) above is all it
    # takes for the scan to hit the baked surface. There is no secondary view here
    # (that is WP3's picture-in-picture), so set_view_sensor_surfaces has nothing
    # to be called on: handle 0, the primary, always refuses.
    bx = (float(spine[:, 0].min()) - 12.0, float(spine[:, 0].max()) + 12.0)
    by = (float(spine[:, 1].min()) - 12.0, float(spine[:, 1].max()) + 12.0)
    scanner = ForwardDepthScanner(rend, scene, meshes,
                                  bounds=(bx[0], bx[1], by[0], by[1]),
                                  cell=MAP_CELL, far=SENSOR_FAR,
                                  width=SENSOR_W, height=SENSOR_H,
                                  mount_fwd=0.95, mount_up=-0.10,
                                  pitch_deg=40.0, fov_y=90.0)
    rend.render(scene, camera)          # build a TLAS the prewarm scans can hit
    scanner.prewarm(art.root_state())
    print(f"[scan] map {scanner.nx}x{scanner.ny} cells over "
          f"x {bx[0]:.1f}..{bx[1]:.1f}  y {by[0]:.1f}..{by[1]:.1f}; "
          f"observed cells after prewarm = {int((~np.isnan(scanner.H)).sum())}")

    # ── WP3: the live SLAM surface ────────────────────────────────────────────
    # Same reconstruction spot_slam runs (VoxelGrid -> splat_points_to_field ->
    # marching_cubes on a background thread, generation-counted so a rebuild
    # that was in flight when the map was cleared drops its result), fed by the
    # SAME accumulated elevation map -- only the look is new: a solid,
    # semi-transparent, height-banded mesh instead of the blue wireframe.
    slam = None
    if not args.no_slam:
        slam = SlamSurface(scene, mat=args.slam_mat, opacity=args.slam_opacity,
                           lift=args.slam_lift, keep_above=args.slam_keep_above)
        # 'standard' keeps the mesh in the traced scene, so the depth sensor
        # would map its own reconstruction. Hiding it here only helps on the
        # GL/synchronous path (the Vulkan trace runs against the TLAS the
        # PREVIOUS render() built), which is one more reason the overlay route
        # is the default rather than a style choice.
        if slam.kind == "standard":
            scanner.hide.append(slam.mesh)
        print(f"[slam] {slam.kind} material, opacity {args.slam_opacity}, "
              f"rebuild every {args.mc_frames} frames, band +/-{SlamSurface.BAND} m")

    # ── WP3: the depth picture-in-picture ─────────────────────────────────────
    pip = None
    if not args.no_pip:
        pip = DepthPip(rend, scene, scanner, width=PIP_W, height=PIP_H,
                       fov_y=90.0, near=0.06, far=SENSOR_FAR,
                       scale=max(1, int(args.pip_scale)),
                       pitch_deg=args.pip_pitch)
        pip.place(art.root_state())
        rend.render(scene, camera)          # add_view needs one frame to exist first
        if not pip.attach():
            pip = None
        else:
            scanner.hide.append(pip.sprite)  # never let the scan see its own readout
            rend.render(scene, camera)
            _sd = rend.read_gbuffer_aov_raw("splat_depth")
            print(f"[splat] splat_depth AOV after the first frames: "
                  f"{'ALLOCATED ' + str(_sd.shape) if _sd is not None else 'not allocated'} "
                  f"(there is no splat_depth_aov property bound in Python; this raw read is "
                  f"the only way to ask)")

    # ── LOD ───────────────────────────────────────────────────────────────────
    _lod_report = [-1]
    _lod_switches = [0]          # level changes since the last reset
    _lod_t0 = [time.perf_counter()]

    def submitted_splats():
        """What the cloud will actually draw: the sum of its submit ranges."""
        try:
            return int(sum(int(c) for _b, c in cloud.submit_ranges))
        except Exception:
            return -1

    def pick_lod(vh):
        """Per-frame level choice.

        select_lod picks the coarsest resident level that still covers the cloud's
        projected footprint at ~target_splats_per_pixel. Standing ON the scan the
        near field must be level 0 or the rock beside a foot reads as blobs, so
        when the camera is inside --near-lod-m of the spine the finest resident
        level is forced outright (all its chunks submitted, no frustum trim).

        `vh` is the RAW viewport height; LodBudget scales it, which is how the
        frame-time budget biases this policy coarser (see the class).
        """
        cp = camera.position
        d = float(np.min(np.linalg.norm(spine - np.array([cp.x, cp.y, cp.z]), axis=1)))
        if args.near_lod_m > 0.0 and d < args.near_lod_m and lod_levels:
            lv = lod_levels[0]
            cloud.submit_ranges = [(int(lv["base"]), int(lv["count"]))]
            lvl = 0
        else:
            lvl = tp.select_lod(cloud, camera, budget.viewport(vh),
                                target_splats_per_pixel=args.spp)
            # The budget's hard backstop. select_lod has just written its ranges;
            # if they add up to more splats than the budget can pay for, replace
            # them with the WHOLE of the finest resident level that fits, as one
            # range. One range is deliberate: it is the fragmentation of the
            # per-node selection that bridges gaps and submits everything, so the
            # fallback must not be another scattered set. Frustum culling is lost
            # in the fallback, but in the direction where it fires the frustum
            # held the whole cloud anyway.
            if budget.cap is not None and lod_levels:
                n = submitted_splats()
                if n > budget.cap:
                    pick = None
                    for i, lv in enumerate(lod_levels):
                        if int(lv["count"]) <= budget.cap:
                            pick = i
                            break
                    if pick is None:
                        pick = len(lod_levels) - 1
                    if int(lod_levels[pick]["count"]) < n:
                        cloud.submit_ranges = [(int(lod_levels[pick]["base"]),
                                                int(lod_levels[pick]["count"]))]
                        lvl = pick
                        budget.enforced += 1
        if lvl != _lod_report[0]:
            _lod_report[0] = lvl
            _lod_switches[0] += 1
            n = lod_levels[lvl]["count"] if lvl < len(lod_levels) else 0
            print(f"[lod] level {lvl} ({n} splats, {submitted_splats() / 1e6:.2f} M "
                  f"submitted), {d:.1f} m from the spine, "
                  f"{budget.last_ms:.1f} ms/frame, vh scale {budget.scale:.2f}, cap "
                  f"{'-' if budget.cap is None else f'{budget.cap / 1e6:.2f} M'}")
        # NO cloud.update() here. SplatCloud::update runs sortByDepth, a CPU
        # counting sort over every resident splat whenever the camera moved
        # (SplatCloud.cpp:764-800) -- that is the GL path's per-frame entry.
        # The Vulkan tile rasterizer sorts on the GPU and the pre-render hook
        # does not sort at all, so on this backend the call buys nothing and
        # costs 25.6M splats' worth of CPU every frame: MEASURED at 178 ms of
        # the 183 ms frame, i.e. it WAS the whole "render phase". --cloud-update
        # puts it back for the A/B.
        if args.cloud_update:
            cloud.update(camera)
        return lvl

    # ── waypoint following ────────────────────────────────────────────────────
    # The route is a LIST, not the spine itself, because the walk does not end at
    # the spine's last waypoint any more: it turns around there and walks back.
    # Running out of waypoints used to leave the policy holding a zero command on
    # the slab under the hero camera, and the steps policy does not stand still --
    # it shuffles, and on that side slope it shuffled downhill for fifteen seconds
    # looking about to fall (film final_a, frames 1500..1829: y 3.19 -> 2.22 and
    # z 0.98 -> 0.49, i.e. half a metre of slide). Turning around gives it
    # somewhere to go and gives the static hero camera a walking robot to watch.
    auto = [bool(args.auto)]
    route = [spine]                  # the waypoint list currently being followed
    wp_i = [_wp0 + 1]                # the spawn waypoint itself; steer to the next
    walked = [0.0]
    turned = [False]                 # has the turn-back already been taken?
    turning = [False]                # mid-turn: the forward command is held at zero
    range_gate = [None]              # film mode installs a keep-in-frame taper here
    _last_p = [np.array(art.root_state()[:3], float)]
    falls = [0]

    TURN_TRIGGER_M = 1.5     # ...of the route's last waypoint
    TURN_WZ_MAX = 0.9        # rad/s command cap while turning (training range is +-1.2)
    TURN_RESUME_DEG = 20.0   # heading error at which the forward command comes back

    def turn_back(why=""):
        """Reverse the remaining route: walk back along the spine toward its start.

        The new route is every waypoint already PASSED, nearest first, so the robot
        retraces ground the bake covered and the scanner has already mapped. Only
        ever taken once (`turned`), whichever trigger fires first.
        """
        if turned[0]:
            return False
        cur = route[0]
        i = max(0, min(int(wp_i[0]), len(cur)))
        back = np.ascontiguousarray(cur[:i][::-1])
        if len(back) == 0:
            return False
        route[0] = back
        wp_i[0] = 0
        turned[0] = True
        turning[0] = True
        print(f"[auto] TURN-BACK ({why}): {len(back)} waypoints back down the spine, "
              f"first target ({back[0][0]:+.2f}, {back[0][1]:+.2f})")
        return True

    def auto_cmd(rs):
        """Velocity command steering toward the next route waypoint. -> (vx, vy, wz).

        Near the end of the route the route REVERSES and the robot turns around.
        The turn is a walking turn only in the sense that the gait keeps running:
        the forward command is zeroed until the heading error is back under
        TURN_RESUME_DEG, because asking this policy for 0.45 m/s forward while it
        yaws through 180 degrees on gravel is how it trips.
        """
        p = np.array(rs[:3], float)
        R = _quat_to_R(rs[3:7])
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
        wps = route[0]
        if not turned[0] and len(wps) and \
                float(np.linalg.norm(wps[-1, :2] - p[:2])) < TURN_TRIGGER_M:
            if turn_back(f"within {TURN_TRIGGER_M:.1f} m of the last waypoint"):
                wps = route[0]
        while wp_i[0] < len(wps) and np.linalg.norm(wps[wp_i[0], :2] - p[:2]) < 0.8:
            wp_i[0] += 1
        if wp_i[0] >= len(wps):
            return 0.0, 0.0, 0.0
        d = wps[wp_i[0], :2] - p[:2]
        err = (math.atan2(d[1], d[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
        if turning[0]:
            if abs(err) <= math.radians(TURN_RESUME_DEG):
                turning[0] = False
                print(f"[auto] turn complete, heading error {math.degrees(err):+.1f} deg "
                      f"-- forward command resumes")
            else:
                return 0.0, 0.0, float(np.clip(1.5 * err, -TURN_WZ_MAX, TURN_WZ_MAX))
        wz = float(np.clip(2.0 * err, -1.2, 1.2))
        # slow down while turning hard so the policy keeps its feet
        vx = float(VX_HI) * float(np.clip(1.0 - abs(err) / 1.0, 0.15, 1.0))
        if range_gate[0] is not None:
            vx *= float(range_gate[0](p))
        return vx, 0.0, wz

    attitude = [0.0, 0.0]            # max |roll|, max |pitch| in degrees

    def track(rs):
        p = np.array(rs[:3], float)
        walked[0] += float(np.linalg.norm(p[:2] - _last_p[0][:2]))
        _last_p[0] = p
        R = _quat_to_R(rs[3:7])
        # Tipped over, not "low": the trail is rough enough that a height threshold
        # fires on every boulder. The body's own up axis is unambiguous.
        if float(R[2, 2]) < 0.5:
            falls[0] += 1
        # Attitude off the body axes: pitch is how far the forward axis leaves the
        # horizontal, roll how far the left axis does. The collider's roughness
        # shows up here long before it shows up as a fall.
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -float(R[2, 0])))))
        roll = math.degrees(math.asin(max(-1.0, min(1.0, float(R[2, 1])))))
        attitude[0] = max(attitude[0], abs(roll))
        attitude[1] = max(attitude[1], abs(pitch))
        return p

    def _save_frame(path):
        try:
            rend.save_frame(path)
        except TypeError:
            rend.save_frame(scene, camera, path)

    _scan_hist = []

    def scanstats(tag, fc):
        a = ahead_cache[0]
        _scan_hist.append((float(a.min()), float(a.max()), float(a.std())))
        print(f"[scanstats] {tag} f{fc:5d} ahead min={a.min():+.3f} max={a.max():+.3f} "
              f"std={a.std():.4f} nonzero={int((a != 0).sum())}/45 "
              f"h_here={h_here_cache[0]:+.3f} cells={int((~np.isnan(scanner.H)).sum())}")

    # ── WP5: film mode ────────────────────────────────────────────────────────
    # Headless, fixed step, one clock. Everything that could make two runs of the
    # same simulation differ is either pinned or removed here:
    #
    #   * ONE accumulated float drives both world.step(DT) and renderer.sim_time,
    #     so every renderer-side animated field (TAA's blend dt included) sees
    #     the same time the physics does, however long the frame took to render.
    #   * the SLAM rebuild runs SYNCHRONOUSLY on a frame cadence instead of on a
    #     background thread, so which frame the map grows on is a function of the
    #     frame index and not of how busy the machine was.
    #   * the depth scan is the synchronous scan(), not the pipelined
    #     fire/harvest pair, and the PiP is read every frame.
    #   * seeds for numpy/torch/random are set at startup; auto-exposure is off
    #     (calico_look pins it), so no frame's exposure depends on the last one's.
    if film:
        import calico_film as CF
        from PIL import Image
        out_dir = os.path.abspath(args.film)
        os.makedirs(out_dir, exist_ok=True)

        cams_world = {n: (np.array(F(c["pos"]), np.float64),
                          np.array(F(c["look"]), np.float64))
                      for n, c in shots["cameras_v2"].items()}
        column = CF.ColumnField(args.asset, F.t)

        class FilmCtx:
            """What a shot's camera function is allowed to know."""

            spawn = SPAWN.copy()

            def __init__(self):
                self._ema = {}

            def robot(self):
                rs = art.root_state()
                R = _quat_to_R(rs[3:7])
                return (np.array([rs[0], rs[1], rs[2]], np.float64),
                        math.atan2(float(R[1, 0]), float(R[0, 0])))

            def lift(self, eye, target, margin=0.35):
                """Raise a chase pose out of the bank. The wash corridor is ~1 m
                wide at x ~ 8.4 (WP0), so an offset that is fine on open rock is
                inside the shoulder 3 m later; the column field says so before
                the frame is rendered instead of after it is looked at."""
                e = np.array(eye, np.float64)
                for _ in range(8):
                    c, _at = column.clearance(e, target)
                    if c >= margin:
                        break
                    e[2] += max(margin - c, 0.15)
                return e

            def smooth(self, key, v, alpha):
                """First-order lag, so a chase pose reacting to a trotting base
                does not shake. Deterministic: its state is only ever the
                previous value of the same deterministic sequence."""
                p = self._ema.get(key)
                v = np.asarray(v, np.float64)
                if p is None:
                    self._ema[key] = v.copy()
                    return v
                q = p + (v - p) * float(alpha)
                self._ema[key] = q
                return q

            def set_slam_fade(self, a):
                if slam is not None:
                    slam.set_opacity(rend, args.slam_opacity * float(a))
                    slam.visible = a > 0.01

            def set_pip_zoom(self, z):
                if pip is not None:
                    pip.zoom = float(z)

        ctx = FilmCtx()
        shot_list = CF.build_shots(cams_world, spine, ctx)

        # Keep the robot in the hero frame once it has turned around. The hero eye
        # sits 1.6 m over the spine's LAST waypoint, so walking back is walking
        # away from the lens; the look target tracks the base, so nothing ever
        # leaves the frame sideways, but a robot 10 m down the wash is a speck.
        # Taper the forward command to nothing at FRAME_KEEP_M.
        FRAME_KEEP_M = 8.0

        def _keep_in_frame(p):
            if not turned[0]:
                return 1.0
            cp = camera.position
            d = math.hypot(float(p[0]) - float(cp.x), float(p[1]) - float(cp.y))
            return float(np.clip((FRAME_KEEP_M - d) / 1.5, 0.0, 1.0))
        range_gate[0] = _keep_in_frame
        want = {s.strip() for s in args.film_shots.split(",") if s.strip()}
        if want:
            shot_list = [s for s in shot_list if s.name in want]
        if args.no_upscaler:
            # The only upscaler bound in Python is FSR (there is no dlss / taa
            # property; DLSS reports itself unavailable on this GPU anyway).
            # What is LEFT after this: the deferred TAA resolve (frame_timings
            # still reports taa_ms), the canvas's antialiasing=4 on the overlay
            # pass, and gbuffer_msaa (1 = off by default here).
            rend.fsr = False
        print(f"[film] upscaler: fsr={rend.fsr} (available={rend.fsr_available}) "
              f"gbuffer_msaa={rend.gbuffer_msaa} canvas antialiasing=4 "
              f"render_scale={rend.render_scale}")
        total = sum(s.frames for s in shot_list)
        if args.film_frames > 0:
            total = min(total, int(args.film_frames))
        print(f"[film] {len(shot_list)} shots, {total} frames at {CF.FPS} fps = "
              f"{total / CF.FPS:.1f} s of film ({total * CF.DT:.1f} s of simulation: "
              f"the sim step is the policy's {CF.DT}s, so the film runs "
              f"{CF.DT * CF.FPS:.2f}x real time)")
        for s in shot_list:
            print(f"[film]   {s.name:<16} {s.seconds:5.1f}s {s.frames:5d}f  "
                  f"walk={int(s.walk)}  {s.note}")

        # Pre-flight: every static keyframe and every midpoint, checked against
        # the splat columns before a single frame is rendered.
        print("[film] sightline pre-flight (worst clearance along cam -> target):")
        for s in shot_list:
            worst = 1e9
            for u in (0.0, 0.25, 0.5, 0.75, 1.0):
                eye, tgt = s.camera(u, ctx)
                c, at = column.clearance(eye, tgt)
                worst = min(worst, c)
            flag = "OK " if worst > 0.0 else "BURIED"
            print(f"[film]   {flag} {s.name:<16} worst {worst:+.2f} m")
        ctx._ema.clear()

        log = CF.FrameLog(out_dir)
        camera.fov = 75.0
        camera.update_projection_matrix()
        scanner.show_cloud = False
        scanner.show_grid = False
        if scanner.cloud is not None:
            scanner.cloud.visible = False
        if scanner.markers:
            scanner.marker_group.visible = False
        if slam is not None:
            slam.set_opacity(rend, 0.0)
            slam.visible = False
        if pip is not None:
            pip.zoom = 1.0

        t_sim = 0.0
        fidx = 0
        gphi = 0.0
        t_wall = time.perf_counter()
        zero_cmd = np.zeros(3, np.float32)
        for shot in shot_list:
            shot_worst = 1e9
            for k in range(shot.frames):
                u = k / max(shot.frames - 1, 1)
                rs = art.root_state()
                # The time trigger for the turn-back, "whichever comes first"
                # against auto_cmd's distance trigger: the sensor beat opens with
                # the robot turning if it has not already turned on arrival.
                if k == 0 and getattr(shot, "turn_back", False):
                    turn_back(f"{shot.name} begins")
                if shot.walk:
                    vx, vy, wz = auto_cmd(rs)
                    cmd = np.array([vx * float(args.film_speed), vy, wz], np.float32)
                else:
                    cmd = zero_cmd
                step_policy(cmd, gphi)          # world.step(0.02) inside
                gphi = (gphi + CF.DT / GAIT_PERIOD) % 1.0
                t_sim += CF.DT
                rs = art.root_state()
                track(rs)

                eye, tgt = shot.camera(u, ctx)
                shot_worst = min(shot_worst, column.clearance(eye, tgt)[0])
                camera.position.set(*(float(v) for v in eye))
                camera.look_at(*(float(v) for v in tgt))
                if shot.beat is not None:
                    shot.beat(u, ctx)
                else:
                    ctx.set_slam_fade(0.0)
                    ctx.set_pip_zoom(1.0)

                if pip is not None:
                    pip.place(rs)
                look_update(rs)
                pick_lod(canvas.size()[1])
                rend.sim_time = t_sim
                if k == 0:
                    # A cut is a disocclusion of the whole frame. Three throwaway
                    # renders at the SAME sim time let TAA converge on the new
                    # pose before the first frame of the shot is kept -- the sim
                    # does not advance, so this costs the film nothing but time.
                    for _ in range(3):
                        rend.render(scene, camera)
                rend.render(scene, camera)

                if fidx % SCAN_EVERY == 0:
                    ahead_cache[0], h_here_cache[0] = scanner.scan(rs)
                    if slam is not None:
                        slam.insert_scanner(scanner)
                if slam is not None and fidx % max(1, int(args.mc_frames)) == 0:
                    slam.rebuild_sync(band_center=h_here_cache[0])
                if pip is not None:
                    pip.update(canvas.size())

                px = rend.read_pixels()
                name = f"f{fidx:05d}.png"
                Image.fromarray(np.ascontiguousarray(px)).save(
                    os.path.join(out_dir, name), compress_level=1)
                log.frame(name, px)
                log.row(fidx, shot.name, fidx / float(CF.FPS), t_sim, rs,
                        ahead_cache[0], getattr(pip, "last_depth", None))
                fidx += 1
                if fidx % 60 == 0:
                    print(f"[film] {fidx}/{total} frames  {shot.name}  "
                          f"walked {walked[0]:.2f} m  "
                          f"{(time.perf_counter() - t_wall) / fidx * 1000:.0f} ms/frame")
                if args.film_frames > 0 and fidx >= int(args.film_frames):
                    break
            shot.worst_clearance = shot_worst
            print(f"[film] {shot.name} done: worst sightline clearance "
                  f"{shot_worst:+.2f} m, walked {walked[0]:.2f} m, falls={falls[0]}")
            if args.film_frames > 0 and fidx >= int(args.film_frames):
                break

        _ft = dict(rend.frame_timings) if hasattr(rend, "frame_timings") else {}
        print("[film] last frame's AA/upscale passes: "
              + "  ".join(f"{k}={_ft[k]:.3f}ms" for k in
                          ("taa_ms", "upscale_ms", "fsr_ms", "dlss_ms") if k in _ft))
        roll, csv_sha = log.close()
        print(f"[film] {fidx} frames in {time.perf_counter() - t_wall:.1f}s "
              f"-> {out_dir}")
        print(f"[film] FRAME HASH  {roll}")
        print(f"[film] SENSOR HASH {csv_sha}")
        CF.stitch(out_dir, CF.FPS)
        canvas.close()
        return

    # ── the interactive-style drive test ──────────────────────────────────────
    # The film walked the spine and never left the corridor, which is exactly why
    # it looked fixed. What the user does is hold W and A. This drives the SAME
    # command vector the key handler builds -- (vx, vy, wz) straight into
    # step_policy -- on a script that leaves the corridor within ten seconds, and
    # measures the one number that says whether there is ground: base z minus the
    # collider's own height under the base.
    if args.drive_test:
        rng = np.random.default_rng(args.drive_seed)
        DT = 0.02
        gphi = 0.0
        n = int(round(float(args.drive_seconds) / DT))
        rs0 = art.root_state()
        yaw0 = math.atan2(*_quat_to_R(rs0[3:7])[[1, 0], 0])
        turn_target = [yaw0 + math.pi / 2]
        rw = [np.zeros(3)]

        def script(t, rs):
            """(vx, vy, wz) exactly as W/A/S/D/Q/E would produce them."""
            R = _quat_to_R(rs[3:7])
            yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
            if args.drive_test == "strafe3":
                # the complaint, minimal: hold A and walk off the side of the spine
                return (0.0, 1.0, 0.0)
            if t < 10.0:
                return (float(VX_HI), 0.0, 0.0)            # W
            if t < 20.0:
                return (0.0, 1.0, 0.0)                     # A
            if t < 30.0:
                return (float(VX_HI), 0.0, 0.0)            # W
            if t < 40.0:                                   # Q until 90 deg, then W
                err = (turn_target[0] - yaw + math.pi) % (2 * math.pi) - math.pi
                if abs(err) > math.radians(10.0):
                    return (0.0, 0.0, float(np.clip(1.5 * np.sign(err), -1.5, 1.5)))
                return (float(VX_HI), 0.0, 0.0)
            if t < 50.0:
                return (0.0, -1.0, 0.0)                    # D
            # a seeded random walk: a new key combination every 1.5 s, held
            if int((t - 50.0) / 1.5) != int((t - 50.0 - DT) / 1.5) or t <= 50.0 + DT:
                rw[0] = np.array([rng.choice([0.0, float(VX_HI), -1.0]),
                                  rng.choice([0.0, 1.0, -1.0]),
                                  rng.choice([0.0, 1.5, -1.5])])
            return tuple(rw[0])

        rec = np.zeros((n, 8), np.float64)   # t x y z dz roll pitch fallen
        gh_at = (lambda x, y: float(hgrid[0].height_at(x, y))) if hgrid[0] is not None \
            else (lambda x, y: float(SPAWN[2]))
        t_wall = time.perf_counter()
        for i in range(n):
            t = i * DT
            rs = art.root_state()
            track(rs)
            if i % SCAN_EVERY == 0:
                look_update(rs)
                rend.render(scene, camera)
                ahead_cache[0], h_here_cache[0] = scanner.scan(rs)
            R = _quat_to_R(rs[3:7])
            rec[i] = (t, rs[0], rs[1], rs[2], rs[2] - gh_at(rs[0], rs[1]),
                      math.degrees(math.asin(max(-1.0, min(1.0, float(R[2, 1]))))),
                      math.degrees(math.asin(max(-1.0, min(1.0, -float(R[2, 0]))))),
                      1.0 if float(R[2, 2]) < 0.5 else 0.0)
            step_policy(np.array(script(t, rs), np.float32), gphi)
            gphi = (gphi + DT / GAIT_PERIOD) % 1.0
            if rs[2] < gh_at(rs[0], rs[1]) - 5.0:
                print(f"[drive] LOST at t={t:.1f}s ({rs[0]:+.2f}, {rs[1]:+.2f}) "
                      f"z={rs[2]:+.2f}, grid {gh_at(rs[0], rs[1]):+.2f} -- stopping")
                rec = rec[:i + 1]
                break

        dz = rec[:, 4]
        # A "drop" is the base falling more than 0.5 m below where it was standing:
        # the median offset IS the standing height, so measure against that.
        base = float(np.median(dz[:min(len(dz), 250)]))
        drops = np.where(dz < base - 0.5)[0]
        nfall = int(rec[:, 7].sum())
        print()
        print(f"  ==== DRIVE TEST '{args.drive_test}' "
              f"(grid_area={args.grid_area} skirt={not args.no_skirt} "
              f"bake={args.bake_poses}) ====")
        print(f"  {len(rec)} steps = {rec[-1,0]:.1f} s in "
              f"{time.perf_counter() - t_wall:.0f}s wall")
        print(f"  base z - grid height:  min {dz.min():+.3f}  max {dz.max():+.3f}  "
              f"median {np.median(dz):+.3f}  (standing offset {base:+.3f})")
        print(f"  max |roll| {np.abs(rec[:,5]).max():.1f} deg   "
              f"max |pitch| {np.abs(rec[:,6]).max():.1f} deg")
        print(f"  falls (body up axis < 0.5): {nfall} steps"
              + (f", first at t={rec[rec[:,7]>0][0,0]:.1f}s "
                 f"({rec[rec[:,7]>0][0,1]:+.2f}, {rec[rec[:,7]>0][0,2]:+.2f})"
                 if nfall else ""))
        print(f"  drops > 0.5 m below the standing offset: {len(drops)} steps"
              + (f", first at t={rec[drops[0],0]:.1f}s "
                 f"({rec[drops[0],1]:+.2f}, {rec[drops[0],2]:+.2f}) dz={dz[drops[0]]:+.2f}"
                 if len(drops) else ""))
        print(f"  travelled x {rec[:,1].min():+.2f}..{rec[:,1].max():+.2f}  "
              f"y {rec[:,2].min():+.2f}..{rec[:,2].max():+.2f}  "
              f"z {rec[:,3].min():+.2f}..{rec[:,3].max():+.2f}   walked {walked[0]:.2f} m")
        print("  base z every 5 s: " + " ".join(
            f"{rec[k,3]:+.2f}" for k in range(0, len(rec), int(5.0 / DT))))
        if args.drive_out:
            np.savez_compressed(args.drive_out, rec=rec,
                                grid_area=args.grid_area, base=base)
            print(f"  -> {args.drive_out}")
        canvas.close()
        return

    # ── the spin bench: what the frame costs in every direction ───────────────
    # The complaint this exists to measure is directional. Standing where the
    # robot spawns and turning round is the whole experiment: the wash is open
    # toward -x and every chunk of the cloud is in frame toward +x, so one
    # revolution walks the frustum through the best and the worst case the demo
    # has. Physics does not run here and neither do the scanner or the SLAM
    # rebuild: the question is what the SPLAT PASS costs, and a policy step or a
    # marching-cubes hitch in the same samples would only blur the answer.
    if spin_s > 0.0:
        SPIN_FPS = 60.0
        n_spin = max(30, int(round(spin_s * SPIN_FPS)))
        eye = np.array([SPAWN[0], SPAWN[1], SPAWN[2] + 1.5], float)
        camera.fov = 50.0
        camera.update_projection_matrix()
        if pip is not None:
            pip.place(art.root_state())

        def _aim(i, n):
            th = 2.0 * math.pi * (i / float(n))
            camera.position.set(*eye)
            camera.look_at(float(eye[0] + 10.0 * math.cos(th)),
                           float(eye[1] + 10.0 * math.sin(th)),
                           float(eye[2] - 0.6))

        def spin_pass(label, budget_ms, revs=2):
            """One or more revolutions. The FIRST is the controller learning the
            scene from scale 1.0 -- the honest cost of walking into the bad
            heading cold -- and the second is what the demo settles at; they are
            reported separately, because averaging them would flatter one and
            slander the other."""
            budget.reset(budget_ms)
            _lod_report[0] = -1
            ms, lv, sub, gpu, sw, enf = [], [], [], [], [], []
            for i in range(n_spin * revs):
                _aim(i % n_spin, n_spin)
                _prev = _lod_report[0]
                _e0 = budget.enforced
                lvl = pick_lod(canvas.size()[1])
                enf.append(1 if budget.enforced > _e0 else 0)
                t0 = time.perf_counter()
                rend.render(scene, camera)
                dt = (time.perf_counter() - t0) * 1e3
                nsub = submitted_splats()
                budget.note(dt, nsub)
                ms.append(dt); lv.append(lvl); sub.append(nsub)
                sw.append(1 if (_prev >= 0 and lvl != _prev) else 0)
                _ft = dict(rend.frame_timings) if hasattr(rend, "frame_timings") else {}
                gpu.append(float(_ft.get("gpu_total_ms", 0.0)))
            for r in range(revs):
                lo, hi = r * n_spin, (r + 1) * n_spin
                nenf = int(sum(enf[lo:hi]))
                a = np.array(ms[lo:hi])
                g = np.array(gpu[lo:hi])
                hist = {int(k): int(v) for k, v in
                        zip(*np.unique(np.array(lv[lo:hi]), return_counts=True))}
                secs = float(a.sum()) / 1e3
                nsw = int(sum(sw[lo:hi]))
                tag = f"{label} r{r + 1}"
                print(f"[spin] {tag:<14} budget={budget_ms:5.1f}ms  "
                      f"p50 {np.percentile(a, 50):6.1f}  p95 {np.percentile(a, 95):6.1f}  "
                      f"max {a.max():6.1f}  mean {a.mean():6.1f} ms   "
                      f"gpu_total p95 {np.percentile(g, 95):5.1f} ms")
                print(f"[spin] {'':<14} levels {hist}   submitted "
                      f"{np.min(sub[lo:hi]) / 1e6:.2f}..{np.max(sub[lo:hi]) / 1e6:.2f} M "
                      f"(mean {np.mean(sub[lo:hi]) / 1e6:.2f} M)")
                print(f"[spin] {'':<14} level switches {nsw} in {secs:.1f} s = "
                      f"{nsw / max(secs, 1e-6):.2f}/s   scale changes "
                      f"{budget.changes}, final vh scale {budget.scale:.2f}, "
                      f"guard enforced on {100.0 * nenf / n_spin:.0f}% of frames, "
                      f"{budget.reverts} stand-downs")
            return np.array(ms[-n_spin:]), lv[-n_spin:]

        print(f"[spin] {n_spin} frames per revolution at {canvas.size()} "
              f"render_scale {rend.render_scale}, eye "
              f"({eye[0]:+.2f}, {eye[1]:+.2f}, {eye[2]:+.2f}), spp {args.spp}")
        for i in range(40):                      # warmup: TAA, pipelines, allocator
            _aim(i, 120)
            pick_lod(canvas.size()[1])
            rend.render(scene, camera)
        off_ms, off_lv = spin_pass("BUDGET OFF", 0.0, revs=1)   # nothing to learn
        on_ms, on_lv = spin_pass("BUDGET ON", float(args.lod_budget_ms) or 25.0)
        # The heading histogram is the point: the average hides the +x arc.
        print("[spin] frame ms by heading (deg from +x, 12 sectors):")
        for lab, arr, lvs in (("off", off_ms, off_lv), ("on", on_ms, on_lv)):
            secs = []
            for s in range(12):
                lo, hi = s * n_spin // 12, (s + 1) * n_spin // 12
                secs.append(f"{np.percentile(arr[lo:hi], 50):5.0f}"
                            f"/{int(np.median(lvs[lo:hi]))}")
            print(f"[spin]   {lab:<3} " + " ".join(secs))
        canvas.close()
        return

    # ── headless capture ──────────────────────────────────────────────────────
    if headless:
        gphi = 0.0
        fc = 0
        nharvest = 0
        target_m = float(args.walk_m)
        # settle a moment on the surface with the policy holding station, then walk
        max_frames = 200 + int(target_m * 90)
        while True:
            fc += 1
            rs = art.root_state()
            track(rs)
            if fc % SCAN_EVERY == 0:
                if pip is not None:
                    pip.place(rs)
                look_update(rs)
                rend.render(scene, camera)
                if pip is not None and fc % (SCAN_EVERY * 5) == 0:
                    pip.update(canvas.size())
                ahead_cache[0], h_here_cache[0] = scanner.scan(rs)
                nharvest += 1
                if slam is not None:
                    slam.insert_scanner(scanner)
                    if nharvest % 12 == 0:
                        slam.trigger_rebuild(band_center=h_here_cache[0])
                    slam.apply_pending()
                if nharvest % 15 == 0:
                    scanstats("walk", fc)
            if args.strafe_m > 0.0:
                # Walk SIDEWAYS off the spine, which is the whole point: the film's
                # camera never saw this ground and neither did the corridor grid.
                off = float(np.linalg.norm(np.array(rs[:2]) - SPAWN[:2]))
                cmd = np.array((0.0, 0.0, 0.0) if fc < 60 else
                               ((0.0, 1.0, 0.0) if off < args.strafe_m else (0.0, 0.0, 0.0)),
                               np.float32)
                if off >= args.strafe_m and fc > 60 + 250:
                    break
            else:
                cmd = np.array((0.0, 0.0, 0.0) if (target_m <= 0.0 or fc < 60)
                               else auto_cmd(rs), np.float32)
            step_policy(cmd, gphi)
            gphi = (gphi + 0.02 / GAIT_PERIOD) % 1.0
            if args.strafe_m > 0.0:
                if fc >= 60 + int(args.strafe_m * 220) + 250:
                    break
            elif target_m <= 0.0:
                if fc >= 120:
                    break
            elif walked[0] >= target_m or fc >= max_frames or wp_i[0] >= len(route[0]):
                break
        rs = art.root_state()
        print(f"[walk] {walked[0]:.2f} m in {fc} frames, waypoint {wp_i[0]}/{len(route[0])}"
              f"{' (turned back)' if turned[0] else ''}, "
              f"falls={falls[0]}, final z={rs[2]:.3f}, "
              f"max |roll| {attitude[0]:.1f} deg, max |pitch| {attitude[1]:.1f} deg")
        if _scan_hist:
            arr = np.array(_scan_hist)
            print(f"[scanstats] over {len(arr)} samples: std mean={arr[:,2].mean():.4f} "
                  f"max={arr[:,2].max():.4f}   spread(max-min) mean="
                  f"{(arr[:,1]-arr[:,0]).mean():.3f} max={(arr[:,1]-arr[:,0]).max():.3f}")

        if args.cam == "trackside":
            pos = trackside_eye
            look = np.array(rs[:3], float) + np.array([0.0, 0.0, 0.15])
        elif args.cam == "back":
            # The user's complaint pose: stand where the robot spawns, 1.5 m up,
            # and look back down the wash toward the brushy end. The Y-up +x of
            # the analysis frame is world +x here (world = (x, -z, y) + t), so
            # "looking back" is simply +x.
            pos = np.array([SPAWN[0], SPAWN[1], SPAWN[2] + 1.5])
            look = pos + np.array([10.0, 0.0, -0.6])
        elif args.cam == "follow":
            p = np.array(rs[:3], float)
            fwd = _quat_to_R(rs[3:7])[:, 0]; fwd[2] = 0
            fwd /= max(np.linalg.norm(fwd), 1e-6)
            pos = p - fwd * 3.2 + np.array([0.0, 0.0, 1.4])
            look = p + np.array([0.0, 0.0, 0.2])
        else:
            pos, look = shot_cam(args.cam)
        camera.fov = 75.0
        camera.update_projection_matrix()
        camera.position.set(*pos)
        camera.look_at(*look)
        scanner.show_cloud = False
        scanner.show_grid = False
        if scanner.cloud is not None:
            scanner.cloud.visible = False
        if scanner.markers:
            scanner.marker_group.visible = False
        if slam is not None:
            # One last reconstruction of everything walked, then wait for it:
            # the worker is a thread, and a capture taken before it lands shows
            # the map as it was a dozen metres ago.
            slam.trigger_rebuild(band_center=h_here_cache[0])
            t_mc = time.perf_counter()
            while slam.busy and time.perf_counter() - t_mc < 20.0:
                time.sleep(0.05)
            tris = slam.apply_pending()
            print(f"[slam] {slam.voxels} voxels -> {tris} triangles "
                  f"in {time.perf_counter() - t_mc:.2f}s, banded on "
                  f"z = {h_here_cache[0]:+.3f} +/- {SlamSurface.BAND}")
        # Three frames let TAA settle on the new pose. The budget needs more than
        # that: it is a feedback loop over a 10-frame EMA, and a capture taken
        # three frames after a cut shows the level the PREVIOUS pose deserved.
        _n_settle = 3 if not budget.on else 90
        for _k in range(_n_settle):             # let TAA settle on the new pose
            pick_lod(h)
            if pip is not None:
                pip.place(rs)
            look_update(rs)
            _t_r = time.perf_counter()
            rend.render(scene, camera)
            budget.note((time.perf_counter() - _t_r) * 1e3, submitted_splats())
            if pip is not None:
                pip.update(canvas.size())
        if budget.on:
            print(f"[lod] capture settled at level {_lod_report[0]}, "
                  f"{submitted_splats() / 1e6:.2f} M submitted, vh scale "
                  f"{budget.scale:.2f}, {budget.ema:.1f} ms/frame smoothed")
        _sd2 = rend.read_gbuffer_aov_raw("splat_depth")
        print(f"[splat] splat_depth AOV at capture time: "
              f"{'ALLOCATED ' + str(_sd2.shape) if _sd2 is not None else 'NOT allocated'} "
              f"-- the overlay occlusion stamp is gated on it")
        if pip is not None:
            rend.render(scene, camera)          # composite the freshest readout
            x, y, pw, ph = pip.rect(canvas.size())
            d = getattr(pip, "last_depth", None)
            if d is not None:
                print(f"[pip] depth {d.min():.2f}..{d.max():.2f} m, "
                      f"{100.0 * float((d < pip.far * 0.999).mean()):.0f}% of pixels on "
                      f"geometry; rect ({x},{y}) {pw}x{ph}; "
                      f"{pip.updates}/{pip.frames} readbacks differed from the one before "
                      f"(a stale view would read 1)")
        _save_frame(args.shot)
        print(f"saved {args.shot}  (cam={args.cam} pos={tuple(round(v,3) for v in pos)} "
              f"look={tuple(round(v,3) for v in look)})")
        return

    # ── interactive ───────────────────────────────────────────────────────────
    ui = tp.ImguiContext(canvas, rend) if tp.HAS_IMGUI else None

    def on_resize(nw, nh):
        camera.aspect = nw / max(nh, 1)
        camera.update_projection_matrix()
        rend.set_size(nw, nh)
    canvas.on_window_resize(on_resize)

    def down(*keys):
        return any(canvas.is_key_down(k) for k in keys)

    fc = [0]
    _nh = [0]
    hdg_lock = [None]
    r_held = [False]
    g_held = [False]
    gphi = [0.0]
    vx_hi = [float(VX_HI)]

    bench_t = {k: [] for k in BENCH_PHASES}
    bench_total, bench_per, bench_scan, bench_gpu = [], [], [], []
    _t_prev = [None]
    _tm = [0.0]
    _has_ft = hasattr(rend, "frame_timings")

    def _mark(key):
        now = time.perf_counter()
        bench_t[key].append(now - _tm[0])
        _tm[0] = now

    def reset():
        art.reset(tp.Vector3(float(SPAWN[0]), float(SPAWN[1]), Z0 + h0 + 0.03), _spawn_quat)
        last_act[:] = 0.0
        hdg_lock[0] = None
        gphi[0] = 0.0
        route[0] = spine
        wp_i[0] = _wp0 + 1
        turned[0] = False
        turning[0] = False
        walked[0] = 0.0
        falls[0] = 0
        settle(40)
        if getattr(scanner.sensor, "scan_pending", False):
            scanner.sensor.scan_collect(rend)
        scanner.clear_map()
        scanner.prewarm(art.root_state())
        if slam is not None:
            slam.clear()
        _last_p[0] = np.array(art.root_state()[:3], float)

    def draw_ui():
        tp.imgui.set_next_window_pos(12, 12)
        tp.imgui.set_next_window_size(320, 0)
        tp.imgui.begin("Spot on Calico Tanks")
        rs = art.root_state()
        tp.imgui.text(f"pos  x={rs[0]:+.2f}  y={rs[1]:+.2f}  z={rs[2]:.2f} m")
        tp.imgui.text(f"walked {walked[0]:.1f} m   waypoint {wp_i[0]}/{len(route[0])}"
                      f"{'  (turned back)' if turned[0] else ''}")
        _, auto[0] = tp.imgui.checkbox("auto-walk the spine (G)", auto[0])
        _, vx_hi[0] = tp.imgui.slider_float("forward speed vx", vx_hi[0], 0.0, VX_HI)
        tp.imgui.separator()
        tp.imgui.text(f"splat LOD level {_lod_report[0]}")
        tp.imgui.text(f"depth map cells {int((~np.isnan(scanner.H)).sum())}")
        _, scanner.show_cloud = tp.imgui.checkbox("show scan cloud", scanner.show_cloud)
        _, scanner.show_grid = tp.imgui.checkbox("show scan grid", scanner.show_grid)
        if slam is not None:
            tp.imgui.text(f"SLAM  voxels {slam.voxels}  tris {slam.triangles}"
                          f"{'  [rebuilding]' if slam.busy else ''}")
            _, slam.visible = tp.imgui.checkbox("show SLAM surface", slam.visible)
            if tp.imgui.button("rebuild surface now") and not slam.busy:
                slam.trigger_rebuild(band_center=h_here_cache[0])
        if pip is not None:
            _, pip.visible = tp.imgui.checkbox("depth PiP (sensor surfaces)", pip.visible)
        if tp.imgui.button("reset (R)"):
            reset()
        tp.imgui.separator()
        tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps  |  WASD+QE  G=auto  mouse=orbit")
        tp.imgui.end()

    def frame():
        fc[0] += 1
        if bench:
            _t_frame = time.perf_counter()
            bench_per.append(0.0 if _t_prev[0] is None else _t_frame - _t_prev[0])
            _t_prev[0] = _t_frame

        rs = art.root_state()
        R = _quat_to_R(rs[3:7])
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))

        if not bench:
            if down("G"):
                if not g_held[0]:
                    auto[0] = not auto[0]
                    print(f"[auto] {'on' if auto[0] else 'off'}")
                g_held[0] = True
            else:
                g_held[0] = False

        if bench or auto[0]:
            vx, vy, wz = auto_cmd(rs)
            if bench and wp_i[0] >= len(route[0]):
                vx, vy, wz = vx_hi[0], 0.0, 0.0
            hdg_lock[0] = yaw
        else:
            vx = (vx_hi[0] if down("W", "KP8") else 0.0) - (1.0 if down("S", "KP2") else 0.0)
            vy = (1.0 if down("A", "KP4") else 0.0) - (1.0 if down("D", "KP6") else 0.0)
            wz_key = (1.5 if down("Q", "KP7") else 0.0) - (1.5 if down("E", "KP9") else 0.0)
            if wz_key != 0.0:
                wz = wz_key; hdg_lock[0] = yaw
            else:
                if hdg_lock[0] is None: hdg_lock[0] = yaw
                err = (yaw - hdg_lock[0] + math.pi) % (2 * math.pi) - math.pi
                wz = float(np.clip(-2.0 * err, -1.0, 1.0))

        if bench: _tm[0] = time.perf_counter()
        step_policy(np.array([vx, vy, wz], np.float32), gphi[0])
        if bench: _mark("policy"); _mark("physics")
        gphi[0] = (gphi[0] + 0.02 / GAIT_PERIOD) % 1.0
        rs = art.root_state()
        track(rs)   # the "slam" bench phase is marked further down, at the rebuild

        if not bench:
            if down("R"):
                if not r_held[0]: reset(); print("[reset]")
                r_held[0] = True
            else:
                r_held[0] = False

        if bench: _tm[0] = time.perf_counter()
        p = np.array(rs[:3], float)
        new_tgt = np.array([float(p[0]), float(p[1]), float(p[2]) + 0.4])
        delta = new_tgt - _prev_tgt[0]
        _prev_tgt[0] = new_tgt
        cp = camera.position
        camera.position.set(cp.x + delta[0], cp.y + delta[1], cp.z + delta[2])
        controls.target.set(*(float(v) for v in new_tgt))
        controls.enabled = not (ui and ui.want_capture_mouse)
        controls.update()
        if bench: _mark("camera")

        _scan_dt = 0.0
        _scan_work = False
        if bench: _t_scan = time.perf_counter()
        res = scanner.scan_harvest(rs)
        if res is not None:
            ahead_cache[0], h_here_cache[0] = res
            _scan_work = True
            _nh[0] += 1
            if slam is not None:
                slam.insert_scanner(scanner)
            if _SCANSTATS and _nh[0] % 20 == 0:
                scanstats("live", fc[0])
        if bench: _scan_dt += time.perf_counter() - _t_scan

        # ── SLAM: rebuild on a cadence, publish whatever the worker finished ──
        if bench: _tm[0] = time.perf_counter()
        if slam is not None:
            if fc[0] % max(1, int(args.mc_frames)) == 0:
                slam.trigger_rebuild(band_center=h_here_cache[0])
            slam.apply_pending()
        if bench: _mark("slam")

        if bench: _tm[0] = time.perf_counter()
        pick_lod(canvas.size()[1])
        if pip is not None:
            pip.place(rs)                       # the view rides the same body mount
        look_update(rs)
        # The budget's signal. Wall time around render() and nothing else: it is
        # the number the user feels, and the only one that saw the 71 ms stall
        # (frame_timings.gpu_total_ms read 22 ms on the same frame).
        _t_r = time.perf_counter()
        rend.render(scene, camera)
        budget.note((time.perf_counter() - _t_r) * 1e3, submitted_splats())
        if bench: _mark("render")
        if pip is not None and fc[0] % PIP_EVERY == 0:
            pip.update(canvas.size())           # reads the frame just rendered

        if fc[0] % SCAN_EVERY == 0:
            if bench: _t_scan = time.perf_counter()
            scanner.scan_fire(rs)
            if bench: _scan_dt += time.perf_counter() - _t_scan
            _scan_work = True
        if bench:
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
            if fc[0] >= bench:
                canvas.close()

    print(__doc__)
    if bench:
        print(f"[bench] profiling {bench} frames (first {BENCH_WARMUP} are warmup) ...")
    canvas.animate(frame)

    print(f"[walk] {walked[0]:.2f} m, waypoint {wp_i[0]}/{len(route[0])}, "
          f"turned_back={turned[0]}, falls={falls[0]}")
    if bench:
        wpad = BENCH_WARMUP
        warm_scans = sum(1 for v in bench_t["scan"][:wpad] if v > 0.0)
        _bench_report({k: v[wpad:] for k, v in bench_t.items()},
                      bench_total[wpad:], bench_per[wpad:], bench_scan[warm_scans:],
                      bench_gpu[wpad:])


if __name__ == "__main__":
    main()
