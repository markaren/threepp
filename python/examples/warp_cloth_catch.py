"""Four arms catch a cannonball in a cloth, tracking it with the engine's own event camera.

Aim the cannon, fire, and a rig of four anchors holding a sheet works out where
the ball will be and gets there in time. The rig is never told the ball's true
position: it sees only what a 640x480 DVS sees -- a sparse stream of brightness
changes -- and everything else is inferred from that.

Three things make this work, and each of them is a measurement, not a guess:

  * BEARINGS ONLY, with gravity supplying the scale. Splitting the estimator's
    error in two showed the bearing was unbiased at 4.3 mrad while the range --
    from the event cloud's apparent radius -- carried a systematic -0.19 m bias,
    3.4% of the distance and more than twice its own noise. The RMS radius of an
    event cloud is only PROPORTIONAL to the ball's apparent radius, and the
    constant depends on how the ball happens to look; a centroid has no such
    constant. So the arc is fitted to the bearings and the range is thrown away.

    Bearings alone would leave the scale free -- every scaled trajectory
    projects identically -- except that gravity is not free. Fixing |g| at 9.81
    pins it, and six unknowns (p0, v0) come out of bearings alone. Against
    fitting reconstructed 3-D points (--size-range) this is 4x better on the
    final call (0.06 m vs 0.26 m) and 11x on the first (0.08 m vs 0.93 m), which
    matters more: it means the rig can commit early instead of chasing.

    No depth sensor is needed for this, and one would not be free -- read_depth
    re-renders, which would feed the detector phantom frames. Where depth WOULD
    earn its place is a target whose size is unknown, or one that does not fly a
    ballistic arc; neither is true here.

  * Timestamps, not frame indices. The detector interpolates each event's stamp
    along the log-intensity ramp between two frames, so the stream carries real
    microsecond times. Fitting the ballistic arc against those stamps means the
    two-frame readback latency is handled for free rather than biasing the
    prediction.

  * The rig arrives MATCHING THE BALL'S HORIZONTAL VELOCITY. The bowl in a slack
    sheet is only ~0.3 m deep, worth about 3 J/kg, while a ball crossing it at
    3 m/s carries 4.5 J/kg -- park the rig at the landing spot and the ball skips
    straight out the far side. So the approach is a cubic Hermite (position AND
    velocity at arrival), re-solved every frame from the rig's current state as
    the track sharpens. That constraint is also what makes it look like four
    things cooperating rather than four things moving.

The sensor camera is static, because a moving event camera fires events over the
whole frame. That is both realistic and a better shot.

In the window the cannon carries its aim visibly and draws the arc the shot will
actually take, with a ring where it crosses the catch plane. That preview is
ground truth and is the PLAYER's aid -- it is hidden the instant the shot leaves,
so it can never be confused with, or leak into, what the tracker has to work out
for itself. Watching the tracker's cyan reticle converge onto that amber ring is
the whole perception story in one picture.

    python warp_cloth_catch.py                      # window; A/D aim, W/S elevate,
                                                    # Q/E power, SPACE fire, R reset
    python warp_cloth_catch.py --frames 900 --autofire   # bounded window run, for testing
    python warp_cloth_catch.py --tune               # headless, numbers only
    python warp_cloth_catch.py --clip catch.mp4     # headless mp4
    python warp_cloth_catch.py --clip c.mp4 --split # beauty | event stream, side by side
    python warp_cloth_catch.py --az -12 --el 58 --speed 7.0
    python warp_cloth_catch.py --oracle             # ground truth instead of the sensor,
                                                    # to separate tracking error from control

Prints the predicted intercept against where the ball actually crossed, so the
perception can be judged separately from the catch.
"""
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import cli_arg, find_ffmpeg, parse_size, standard_material

# ---- flags -------------------------------------------------------------------
TUNE = "--tune" in sys.argv
CLIP = cli_arg("--clip", "", str)
SEQ = cli_arg("--seq", "", str)
SPLIT = "--split" in sys.argv          # beauty | event visualisation, side by side
# Film pass: full render scale, a bigger frame, a proper key/fill/rim set, the
# dead air before the shot trimmed, and the catch itself in slow motion. The
# sheet still has to settle for the tracker to bootstrap -- it just does not
# have to be watched doing it.
FILM = "--film" in sys.argv
ORACLE = "--oracle" in sys.argv        # bypass the sensor; use true ball state
SIZE_RANGE = "--size-range" in sys.argv  # the old apparent-size fit, for the A/B
TRACE = "--trace" in sys.argv
# Bounded interactive run: drives the real window loop -- ImGui, HUD, orbit
# controls and all -- for N frames and exits, so the interactive path can be
# exercised rather than assumed. --autofire shoots as soon as it reads READY.
FRAMES = int(cli_arg("--frames", 0, float))
AUTOFIRE = "--autofire" in sys.argv
HEADLESS = bool(TUNE or CLIP or SEQ) and not FRAMES

# ---- the cannon --------------------------------------------------------------
CANNON = np.array([float(x) for x in cli_arg("--cannon", "-2.40,0.55,0", str).split(",")])
AZ = cli_arg("--az", 0.0, float)             # degrees; 0 fires along +x
EL = cli_arg("--el", 62.0, float)            # degrees above horizontal
MUZZLE = cli_arg("--speed", 6.5, float)      # m/s
aim = {"az": AZ, "el": EL, "speed": MUZZLE}
# The sheet must be STILL before the ball flies. A DVS reports change, so a
# ringing sheet floods the frame and the tracker locks onto cloth instead of
# ball -- measured centre 300 px from the true one. Once it has settled the
# ball is the only mover and the same tracker is accurate to ~1.4 px.
FIRE_AT = cli_arg("--fire-at", 1.2, float)   # headless: seconds of settle first

# ---- the rig -----------------------------------------------------------------
HOME = np.array([float(x) for x in cli_arg("--home", "0.50,0", str).split(",")])  # x, z
ARM_SPEED = cli_arg("--arm-speed", 4.5, float)   # m/s cap on the rig centre
ARM_REACH = cli_arg("--arm-reach", 2.0, float)   # m from home the rig may travel
ABSORB = cli_arg("--absorb", 0.35, float)        # s to bleed the ball's motion off
GIVE = cli_arg("--give", 0.30, float)            # m the sheet may drop while absorbing
RECOVER = cli_arg("--recover", 0.55, float)      # s to lift back to the ready height
PRESENT = cli_arg("--present", 0.75, float)      # how taut to pull on recovery, in
                                                 # units of the slack (1 = dead flat)

# ---- the sheet ---------------------------------------------------------------
N = int(cli_arg("--res", 48, float))
CLOTH = cli_arg("--cloth", 1.50, float)
SPAN = cli_arg("--span", 1.30, float)
CATCH_Y = cli_arg("--catch-y", 1.10, float)
CLOTH_KG = cli_arg("--cloth-kg", 0.45, float)

BALL_R = cli_arg("--ball-r", 0.10, float)
BALL_KG = cli_arg("--ball-kg", 0.40, float)
MU = cli_arg("--mu", 0.55, float)            # cloth-on-ball Coulomb friction

# ---- rates -------------------------------------------------------------------
# The sensor rate IS the loop rate: the detector samples once per render, so a
# fast tracker means a fast loop. The substep rate is held constant across it,
# because that -- not the frame rate -- is what the contact needs.
SENSOR_HZ = cli_arg("--sensor-hz", 240.0, float)
SUBSTEP_HZ = cli_arg("--substep-hz", 2880.0, float)
ITERS = int(cli_arg("--iters", 32, float))
SUBSTEPS = max(1, int(round(SUBSTEP_HZ / SENSOR_HZ)))
DT = 1.0 / (SENSOR_HZ * SUBSTEPS)
CLIP_FPS = 60.0
SECONDS = cli_arg("--seconds", 3.0, float)
DAMPING = cli_arg("--damping", 0.010, float)

SENSOR_W, SENSOR_H = parse_size(cli_arg("--sensor", "640x480", str))
# 4:3, to MATCH the 640x480 sensor. When the two aspects differ the detector's
# downsample is anisotropic and one focal length no longer describes both axes;
# rather than model the resampling, keep them the same shape. It is also the
# better aspect for a social clip.
VIEW_W, VIEW_H = parse_size(cli_arg("--size", "1280x960" if FILM else "960x720", str))
EV_THRESHOLD = cli_arg("--ev-threshold", 0.15, float)
# Render is ~65% of the loop, so this is the lever that matters. The worry was
# that it would cost accuracy -- the detector samples the post-TAA frame, so a
# lower scale softens the edges it fires on -- but measured across 0.5..1.0 the
# bearing stays at 2.4-2.6 px and every scale still catches. Interleaved A/B
# (the run-to-run spread is ~15%, so singles are not worth quoting): 25.8 ms at
# 1.0 against 18.7 ms at 0.6.
RENDER_SCALE = cli_arg("--render-scale", 1.0 if FILM else 0.8, float)
PROFILE = "--profile" in sys.argv
# The GI machinery is built for many-light and emissive-geometry scenes. This
# one has two lights and no emitters, so it is paying for convergence it does
# not need. Individually flagged so the cost of each can be measured.
NO_GI = "--no-gi" in sys.argv
NO_AO = "--no-ao" in sys.argv
NO_DENOISE = "--no-denoise" in sys.argv
NO_RESTIR = "--no-restir" in sys.argv
prof = {}
MIN_EVENTS = int(cli_arg("--min-events", 40, float))

V = (N + 1) * (N + 1)
REST = CLOTH / N
M_PARTICLE = CLOTH_KG / V
GRAVITY = wp.vec3(0.0, -9.81, 0.0)
G_NP = np.array([0.0, -9.81, 0.0])

# ---- warp: the sheet ----------------------------------------------------------


@wp.func
def grid_index(ix: int, iy: int, nx: int) -> int:
    return iy * (nx + 1) + ix


@wp.func
def spring(p: wp.vec3, pos: wp.array(dtype=wp.vec3), im: wp.array(dtype=float),
           w_self: float, ix: int, iy: int, nx: int, ny: int,
           rest: float, stiffness: float) -> wp.vec3:
    if ix < 0 or ix > nx or iy < 0 or iy > ny:
        return wp.vec3(0.0, 0.0, 0.0)
    j = grid_index(ix, iy, nx)
    w_nb = im[j]
    denom = w_self + w_nb
    if denom < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)
    d = pos[j] - p
    l = wp.length(d)
    if l < 1.0e-9:
        return wp.vec3(0.0, 0.0, 0.0)
    return d * (stiffness * (w_self / denom) * (l - rest) / l)


@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
              pred: wp.array(dtype=wp.vec3), im: wp.array(dtype=float),
              dt: float, damping: float):
    i = wp.tid()
    p = pos[i]
    prev_p = prev[i]
    prev[i] = p
    if im[i] == 0.0:
        pred[i] = p
        return
    vel = (p - prev_p) * (1.0 - damping)
    pred[i] = p + vel + GRAVITY * dt * dt


@wp.kernel
def set_anchors(pred: wp.array(dtype=wp.vec3), idx: wp.array(dtype=int),
                target: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    pred[idx[k]] = target[k]


@wp.kernel
def solve(p_in: wp.array(dtype=wp.vec3), p_out: wp.array(dtype=wp.vec3),
          im: wp.array(dtype=float), nx: int, ny: int, rest: float,
          ball: wp.array(dtype=wp.vec3), ball_prev: wp.array(dtype=wp.vec3),
          ball_r: float, mu: float,
          prev_pos: wp.array(dtype=wp.vec3),
          anchors: wp.array(dtype=wp.vec3), lra: wp.array(dtype=float),
          impulse: wp.array(dtype=wp.vec3), m_particle: float, relax: float):
    i = wp.tid()
    p = p_in[i]
    w = im[i]
    if w == 0.0:
        p_out[i] = p
        return
    ix = i % (nx + 1)
    iy = i // (nx + 1)
    rd = rest * wp.sqrt(2.0)
    c = wp.vec3(0.0, 0.0, 0.0)
    c += spring(p, p_in, im, w, ix - 1, iy, nx, ny, rest, 1.0)
    c += spring(p, p_in, im, w, ix + 1, iy, nx, ny, rest, 1.0)
    c += spring(p, p_in, im, w, ix, iy - 1, nx, ny, rest, 1.0)
    c += spring(p, p_in, im, w, ix, iy + 1, nx, ny, rest, 1.0)
    c += spring(p, p_in, im, w, ix - 1, iy - 1, nx, ny, rd, 0.85)
    c += spring(p, p_in, im, w, ix + 1, iy - 1, nx, ny, rd, 0.85)
    c += spring(p, p_in, im, w, ix - 1, iy + 1, nx, ny, rd, 0.85)
    c += spring(p, p_in, im, w, ix + 1, iy + 1, nx, ny, rd, 0.85)
    c += spring(p, p_in, im, w, ix - 2, iy, nx, ny, 2.0 * rest, 0.35)
    c += spring(p, p_in, im, w, ix + 2, iy, nx, ny, 2.0 * rest, 0.35)
    c += spring(p, p_in, im, w, ix, iy - 2, nx, ny, 2.0 * rest, 0.35)
    c += spring(p, p_in, im, w, ix, iy + 2, nx, ny, 2.0 * rest, 0.35)
    p = p + c * relax

    # Long-range attachment: no particle may be further from an anchor than the
    # sheet's own material distance to it. Jacobi propagates tension one edge per
    # iteration, far too slow to stop a 49-wide grid stretching like a trampoline
    # under an impact; this says it in one step. Unilateral, so it adds no energy.
    for k in range(4):
        ak = anchors[k]
        dk = p - ak
        lk = wp.length(dk)
        lmax = lra[i * 4 + k]
        if lk > lmax:
            p = ak + dk * (lmax / lk)

    # Ball contact. The pushout delta is the only thing the ball did to this
    # particle, so m * delta summed over the sheet IS the momentum transfer --
    # the spring corrections above cancel pairwise and cannot leak into it.
    bc = ball[0]
    d = p - bc
    l = wp.length(d)
    if l < ball_r:
        nrm_c = d / wp.max(l, 1.0e-6)
        push = nrm_c * ball_r - d
        p = p + push

        # Coulomb friction, position-based. Without it the pushout is purely
        # radial, so nothing can slow the ball ALONG the sheet -- it skates
        # across and out the far side no matter where the rig is. The sheet
        # holds a ball with friction, not with geometry.
        # Tangential slip is this particle's motion relative to the ball's over
        # the substep, projected out of the normal, and it is limited by the
        # penetration depth exactly as Macklin's PBD friction is.
        slip = (p - prev_pos[i]) - (bc - ball_prev[0])
        slip = slip - nrm_c * wp.dot(slip, nrm_c)
        lt = wp.length(slip)
        fric = wp.vec3(0.0, 0.0, 0.0)
        if lt > 1.0e-9:
            fric = slip * (-wp.min(1.0, mu * (ball_r - l) / lt))
            p = p + fric
        wp.atomic_add(impulse, 0, (push + fric) * m_particle)

    p_out[i] = wp.vec3(p[0], wp.max(p[1], 0.005), p[2])


@wp.kernel
def ball_predict(bp: wp.array(dtype=wp.vec3), bv: wp.array(dtype=wp.vec3),
                 bpred: wp.array(dtype=wp.vec3), h: float):
    bpred[0] = bp[0] + bv[0] * h + GRAVITY * h * h


@wp.kernel
def ball_finish(bp: wp.array(dtype=wp.vec3), bv: wp.array(dtype=wp.vec3),
                impulse: wp.array(dtype=wp.vec3), h: float, inv_bm: float,
                radius: float):
    v = bv[0] + GRAVITY * h - impulse[0] * (inv_bm / h)
    p = bp[0] + v * h
    if p[1] < radius:
        p = wp.vec3(p[0], radius, p[2])
        v = wp.vec3(v[0] * 0.6, wp.abs(v[1]) * 0.3, v[2] * 0.6)
    bv[0] = v
    bp[0] = p


@wp.kernel
def compute_normals(pos: wp.array(dtype=wp.vec3), nrm: wp.array(dtype=wp.vec3),
                    nx: int, ny: int):
    i = wp.tid()
    ix = i % (nx + 1)
    iy = i // (nx + 1)
    xm = pos[grid_index(wp.max(ix - 1, 0), iy, nx)]
    xp = pos[grid_index(wp.min(ix + 1, nx), iy, nx)]
    zm = pos[grid_index(ix, wp.max(iy - 1, 0), nx)]
    zp = pos[grid_index(ix, wp.min(iy + 1, ny), nx)]
    n = wp.cross(zp - zm, xp - xm)
    nrm[i] = n / wp.max(wp.length(n), 1.0e-9)


# ---- sheet state --------------------------------------------------------------

wp.init()
device = wp.get_preferred_device()

xs = np.linspace(-SPAN / 2, SPAN / 2, N + 1, dtype=np.float32)
gx, gz = np.meshgrid(xs, xs)
r2 = (gx / (SPAN / 2)) ** 2 + (gz / (SPAN / 2)) ** 2
gy = CATCH_Y - 0.5 * (CLOTH - SPAN) * np.clip(1.0 - r2, 0.0, 1.0)
p0 = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)

CORNERS = [(0, 0), (N, 0), (0, N), (N, N)]
anchor_idx_np = np.array([iy * (N + 1) + ix for ix, iy in CORNERS], dtype=np.int32)
anchor_local = np.array([[float(xs[ix]), 0.0, float(xs[iy])] for ix, iy in CORNERS],
                        dtype=np.float32)
p0[anchor_idx_np] = anchor_local + np.array([0.0, CATCH_Y, 0.0], np.float32)
p0[:, 0] += HOME[0]
p0[:, 2] += HOME[1]

fx = np.linspace(-CLOTH / 2, CLOTH / 2, N + 1, dtype=np.float32)
mx, mz = np.meshgrid(fx, fx)
flat = np.stack([mx, np.zeros_like(mx), mz], axis=-1).reshape(-1, 3)
lra_np = np.linalg.norm(flat[:, None, :] - flat[anchor_idx_np][None, :, :],
                        axis=2).astype(np.float32).reshape(-1)

im_np = np.full(V, 1.0, dtype=np.float32)
im_np[anchor_idx_np] = 0.0

pos = wp.array(p0, dtype=wp.vec3, device=device)
prev = wp.array(p0, dtype=wp.vec3, device=device)
pred = wp.zeros(V, dtype=wp.vec3, device=device)
scratch = wp.zeros(V, dtype=wp.vec3, device=device)
nrm = wp.zeros(V, dtype=wp.vec3, device=device)
im = wp.array(im_np, dtype=float, device=device)
lra = wp.array(lra_np, dtype=float, device=device)
anchor_idx = wp.array(anchor_idx_np, dtype=int, device=device)
anchor_tgt = wp.array(p0[anchor_idx_np].copy(), dtype=wp.vec3, device=device)

bp = wp.array(np.array([CANNON], np.float32), dtype=wp.vec3, device=device)
bv = wp.zeros(1, dtype=wp.vec3, device=device)
bpred = wp.zeros(1, dtype=wp.vec3, device=device)
impulse = wp.zeros(1, dtype=wp.vec3, device=device)

solve_graph = None
if device.is_cuda and ITERS % 2 == 0:
    try:
        with wp.ScopedCapture(device) as _cap:
            _a, _b = pred, scratch
            for _ in range(ITERS):
                wp.launch(solve, dim=V, device=device,
                          inputs=[_a, _b, im, N, N, REST, bpred, bp, BALL_R + 0.012, MU,
                                  prev, anchor_tgt, lra, impulse, M_PARTICLE, 0.35])
                _a, _b = _b, _a
        solve_graph = _cap.graph
    except Exception as exc:                       # noqa: BLE001 - capture is optional
        print(f"  (no graph capture: {exc})")


# ---- the rig ------------------------------------------------------------------

class Rig:
    """Four anchors on a square that translates. Position and velocity of the
    centre are the state; the corners ride along rigidly."""

    def __init__(self):
        self.p = np.array([HOME[0], CATCH_Y, HOME[1]])
        self.v = np.zeros(3)
        self.travel = 0.0
        self.peak_speed = 0.0
        self.starved = 0          # frames the speed cap bit
        self.p_pending = self.p.copy()
        self.spread = 0.0

    def targets(self):
        # Spreading the anchors re-tensions the sheet. Left slack it simply
        # crumples over the ball and hides it, which is the same lesson the
        # throw taught in reverse: a slack sheet has no shape to speak of.
        scale = 1.0 + self.spread * (CLOTH - SPAN) / SPAN
        return (anchor_local * scale + self.p).astype(np.float32)

    def goto(self, dt, p_cmd, v_cmd):
        """Move toward the commanded pose, capped at ARM_SPEED."""
        step = p_cmd - self.p
        d = float(np.linalg.norm(step))
        if d > ARM_SPEED * dt:
            step *= ARM_SPEED * dt / d
            self.starved += 1
        self.p_pending = self.p + step
        self.v = step / dt if dt > 0 else v_cmd
        self._commit()

    def _commit(self):
        home3 = np.array([HOME[0], CATCH_Y, HOME[1]])
        nxt = self.p_pending
        off = nxt - home3
        d = float(np.linalg.norm(off))
        if d > ARM_REACH:                       # the arms simply run out of arm
            nxt = home3 + off * (ARM_REACH / d)
            self.v[:] = 0.0
        self.travel += float(np.linalg.norm(nxt - self.p))
        self.peak_speed = max(self.peak_speed, float(np.linalg.norm(self.v)))
        self.p = nxt

    def step(self, dt, accel):
        self.v += accel * dt
        sp = float(np.linalg.norm(self.v))
        if sp > ARM_SPEED:
            self.v *= ARM_SPEED / sp
        self.p_pending = self.p + self.v * dt
        self._commit()


rig = Rig()


# ---- the sensor ---------------------------------------------------------------

class EventTracker:
    """Ball state from a DVS stream: cluster, centre, size -> bearing and range.

    Initialisation is unguarded (before the rig moves the ball is the only thing
    generating events); after that the search is gated to a window around the
    prediction, which is what keeps the sheet's own motion out of the track.
    """

    def __init__(self, fx, fy, eye, fwd, right, up, w, h):
        # Separate focal lengths per axis. The detector runs at its own
        # resolution while the view renders at another, and when the two aspect
        # ratios differ the downsample is ANISOTROPIC -- here 0.5x in x against
        # 0.667x in y. Using one focal length for both biased every range by
        # ~14%, which is 0.7 m at 5 m and put the predicted crossing 0.3 m short.
        self.fx, self.fy = fx, fy
        self.eye, self.fwd, self.right, self.up = eye, fwd, right, up
        self.w, self.h = w, h
        self.obs = []                # (t, x, y, z) in world space
        self.fit = None              # (p0, v0) of the ballistic fit
        self.gate = None             # (u, v, radius_px) to search inside
        self.rejected = 0
        self.bearing_rms = float('nan')
        self.last_px = None
        self.recent = []
        self.bear = []      # (t, u, v): the measurement that is not biased

    def _angular(self, xy):
        """Pixels -> tangent-plane angles, which are isotropic even when the
        pixels are not. The ball's angular radius is then simply R / z."""
        return np.stack([(xy[:, 0] - self.w / 2.0) / self.fx,
                         (xy[:, 1] - self.h / 2.0) / self.fy], axis=1)

    def observe(self, ev, t_now):
        if ev.shape[0] < MIN_EVENTS:
            return None
        xy = ev[:, :2].astype(np.float64)
        if self.gate is not None:
            gu, gv, gr = self.gate
            keep = (np.abs(xy[:, 0] - gu) < gr) & (np.abs(xy[:, 1] - gv) < gr)
            if keep.sum() >= MIN_EVENTS:
                xy = xy[keep]

        # Seed on the DENSEST patch, not the centroid. The sheet is a far bigger
        # event source than the ball -- a moving rig outlines the whole cloth --
        # so a centroid or median over the raw stream lands on cloth and the
        # range collapses. But the cloth fires as a thin sparse OUTLINE while the
        # ball fires as a solid disc, so events-per-cell separates them cleanly.
        cell = 8
        ncx = self.w // cell + 1
        key = (xy[:, 1] // cell).astype(np.int64) * ncx + (xy[:, 0] // cell).astype(np.int64)
        counts = np.bincount(key)
        seed = int(counts.argmax())
        c = np.array([(seed % ncx + 0.5) * cell, (seed // ncx + 0.5) * cell])

        # Collect around the seed and let the radius settle: start wide enough
        # for a close ball, then shrink onto whatever is actually there.
        radius = 40.0
        sel = xy
        for _ in range(3):
            d = np.linalg.norm(xy - c, axis=1)
            sel = xy[d < radius]
            if sel.shape[0] < MIN_EVENTS:
                self.rejected += 1
                return None
            c = sel.mean(axis=0)
            rr = float(np.sqrt(((sel - c) ** 2).sum(axis=1).mean()))
            radius = min(max(2.5 * rr, 8.0), 60.0)
        xy = sel
        c = xy.mean(axis=0)
        r_rms = float(np.sqrt(((xy - c) ** 2).sum(axis=1).mean()))
        if r_rms < 1.0:
            self.rejected += 1
            return None
        ang = self._angular(xy)
        c_ang = ang.mean(axis=0)
        r_ang = float(np.sqrt(((ang - c_ang) ** 2).sum(axis=1).mean()))
        if r_ang < 1e-5:
            self.rejected += 1
            return None
        # Events carry interpolated microsecond stamps, so use the stream's own
        # median time rather than "now" -- that absorbs the readback latency.
        t_ev = float(np.median(ev[:, 3])) * 1e-6
        if t_ev <= 0.0:
            t_ev = t_now
        self.last_px = (c[0], c[1], r_rms, xy.shape[0])
        z = BALL_R / r_ang
        p = (self.eye + self.fwd * z
             + self.right * (c_ang[0] * z) - self.up * (c_ang[1] * z))
        self.obs.append((t_ev, p[0], p[1], p[2]))
        self.bear.append((t_ev, c[0], c[1]))
        # Tight gate: the sheet is a large, bright, fast mover and once the ball
        # nears it a loose window swallows cloth events and the range collapses.
        self.gate = (c[0], c[1], max(4.0 * r_rms, 30.0))
        return p

    def solve_fit(self):
        """Least squares p(t) = p0 + v0 t + g t^2 / 2 over the observations,
        with one Huber-style reweighting pass so the occasional collapsed-cluster
        frame cannot drag the arc."""
        if len(self.obs) < 24:
            return None
        a = np.array(self.obs)
        if a[-1, 0] - a[0, 0] < 0.12:
            return None
        t = a[:, 0] - a[0, 0]
        y = a[:, 1:] - 0.5 * G_NP[None, :] * (t ** 2)[:, None]
        A = np.stack([np.ones_like(t), t], axis=1)
        w = np.ones_like(t)
        for _ in range(2):
            Aw = A * w[:, None]
            sol, *_ = np.linalg.lstsq(Aw, y * w[:, None], rcond=None)
            res = np.linalg.norm(A @ sol - y, axis=1)
            s = max(float(np.median(res)), 1e-3)
            w = 1.0 / (1.0 + (res / (3.0 * s)) ** 2)
        self.fit = (sol[0], sol[1], a[0, 0])
        return self.fit

    def _project(self, p):
        d = p - self.eye
        z = np.dot(d, self.fwd)
        z = np.where(np.abs(z) < 1e-3, 1e-3, z)
        return np.stack([self.w / 2.0 + self.fx * np.dot(d, self.right) / z,
                         self.h / 2.0 - self.fy * np.dot(d, self.up) / z], axis=-1)

    def solve_fit_bearings(self):
        """Fit the arc to the BEARINGS, with gravity fixed.

        Range from apparent size carries a systematic bias -- measured at
        -0.19 m, which is 3.4% of the distance and dwarfs its 0.08 m noise --
        because the RMS radius of an event cloud is only proportional to the
        ball's apparent radius, and the constant depends on how the ball happens
        to look. The bearing has no such constant: it is a centroid, and it came
        out unbiased at 4.3 mrad.

        Bearings alone would leave the scale free -- every scaled trajectory
        projects identically -- except that gravity is NOT free. Fixing |g| at
        9.81 pins the scale, so six unknowns (p0, v0) are recoverable from
        bearings alone. This is Gauss-Newton on those six, started from the
        size-based fit, which is close enough to converge from.
        """
        if len(self.bear) < 24:
            return None
        a = np.array(self.bear)
        t = a[:, 0] - a[0, 0]
        if t[-1] < 0.12:
            return None
        seed = self.solve_fit()
        if seed is None:
            return None
        x = np.concatenate([seed[0], seed[1]])
        half_g = 0.5 * G_NP[None, :] * (t ** 2)[:, None]

        def resid(x):
            p = x[None, :3] + x[None, 3:] * t[:, None] + half_g
            return (self._project(p) - a[:, 1:]).reshape(-1)

        r = resid(x)
        for _ in range(6):
            J = np.empty((r.size, 6))
            for k in range(6):
                dx = np.zeros(6)
                dx[k] = 1e-4 if k < 3 else 1e-3
                J[:, k] = (resid(x + dx) - r) / dx[k]
            # Huber-ish reweighting: one collapsed cluster should not steer six
            # parameters that a few hundred good bearings agree on.
            e = np.linalg.norm(r.reshape(-1, 2), axis=1)
            wgt = np.repeat(1.0 / (1.0 + (e / max(3.0 * np.median(e), 1e-6)) ** 2), 2)
            Jw, rw = J * wgt[:, None], r * wgt
            try:
                step = np.linalg.lstsq(Jw, -rw, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
            x = x + step
            r = resid(x)
            if np.linalg.norm(step[:3]) < 1e-4:
                break
        self.fit = (x[:3], x[3:], a[0, 0])
        self.bearing_rms = float(np.sqrt((r ** 2).reshape(-1, 2).sum(axis=1).mean()))
        return self.fit

    def predict_crossing(self, y_plane):
        """Where and when the fitted arc next crosses y_plane going down."""
        if self.fit is None:
            return None
        p0, v0, t0 = self.fit
        a, b, c = 0.5 * G_NP[1], v0[1], p0[1] - y_plane
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        r = math.sqrt(disc)
        ts = sorted(((-b + r) / (2 * a), (-b - r) / (2 * a)))
        t_hit = ts[-1]                       # the descending root
        if t_hit <= 0:
            return None
        p = p0 + v0 * t_hit + 0.5 * G_NP * t_hit * t_hit
        v = v0 + G_NP * t_hit
        if not np.all(np.isfinite(p)) or np.max(np.abs(p)) > 12.0:
            return None                       # outside any arena: not a prediction
        # Publish only once the answer stops moving. A ballistic fit over a short
        # baseline is confident and wrong -- the first call here came out 13 m
        # away -- and the rig will happily chase it. Agreement across consecutive
        # frames is the cheapest honest confidence signal available.
        self.recent.append(p)
        if len(self.recent) > 5:
            self.recent.pop(0)
        if len(self.recent) < 5:
            return None
        spread = float(np.max(np.linalg.norm(np.array(self.recent) -
                                             np.mean(self.recent, axis=0), axis=1)))
        if spread > 0.15:
            return None
        return t0 + t_hit, p, v


# ---- scene --------------------------------------------------------------------

if not tp.HAS_VULKAN:
    print("This demo needs the Vulkan backend (the event camera lives there).")
    sys.exit(0)

canvas = tp.Canvas("threepp - cloth catch", width=VIEW_W, height=VIEW_H,
                   headless=HEADLESS, vsync=False)
renderer = tp.VulkanRenderer(canvas)

scene = tp.Scene()
scene.background = 0x0a0d11

# The sensor IS the shot camera, and it is static: a moving event camera fires
# events across the whole frame, so there is nothing to be gained by flying it.
# High enough to see INTO the bowl -- from a low angle the sheet's near lip
# hides the very thing the shot is about -- while still framing the whole arc.
# The film pass sits closer: the arms are the subject and the wide framing left
# most of the frame as empty floor. Still wide enough that the cannon and the
# whole arc stay in shot.
# Framed on the RIG, not on the whole arena. The cannon sits at the edge and the
# ball flies in from off-frame, which reads better than watching it be launched --
# and it stops most of the frame being empty floor between the two.
EYE = np.array([1.75, 2.35, 4.25]) if FILM else np.array([0.60, 3.15, 5.60])
TGT = np.array([0.55, 1.05, 0.0]) if FILM else np.array([-0.30, 1.05, 0.0])
camera = tp.PerspectiveCamera(45, VIEW_W / VIEW_H, 0.1, 100)
camera.position.set(*EYE)
camera.look_at(*TGT)

_fwd = TGT - EYE
_fwd /= np.linalg.norm(_fwd)
_right = np.cross(_fwd, np.array([0.0, 1.0, 0.0]))
_right /= np.linalg.norm(_right)
_up = np.cross(_right, _fwd)
_tan_v = math.tan(math.radians(45) / 2.0)
FY_PX = (SENSOR_H / 2.0) / _tan_v
FX_PX = (SENSOR_W / 2.0) / (_tan_v * VIEW_W / VIEW_H)

scene.add(tp.HemisphereLight(0xffffff, 0x1a1e24, 0.55))
sun = tp.DirectionalLight(0xffffff, 3.2)
sun.position.set(3.5, 7.0, 4.0)
sun.cast_shadow = True
scene.add(sun)
if FILM:
    # White arms on a dark floor lose their silhouette under a single lamp.
    _fill = tp.DirectionalLight(0x9fb6d8, 0.9)      # cool fill from the shadow side
    _fill.position.set(-5.0, 2.5, 2.0)
    scene.add(_fill)
    _rim = tp.DirectionalLight(0xffd7a8, 1.7)       # warm rim, from behind
    _rim.position.set(-1.5, 3.0, -6.0)
    scene.add(_rim)

ground = tp.Mesh(tp.PlaneGeometry(40, 40), standard_material(0x24282e, 0.95))
ground.rotate_x(-math.pi / 2)
ground.receive_shadow = True
scene.add(ground)

geometry = tp.PlaneGeometry(CLOTH, CLOTH, N, N)
cloth_mesh = tp.Mesh(geometry, standard_material(0xd4542e, 0.88, side=tp.Side.Double))
cloth_mesh.cast_shadow = True
scene.add(cloth_mesh)

def _ball_texture(w=768, h=384):
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    U, V = np.meshgrid(u, v)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (243, 246, 252)
    img[((np.floor(U * 12) + np.floor(V * 6)) % 2).astype(bool)] = (22, 26, 34)
    img[np.abs(V - 0.5) < 0.030] = (0, 208, 255)      # equator
    img[np.abs(U - 0.5) < 0.010] = (255, 96, 30)      # meridian
    return img


ball_mat = standard_material(0xffffff, 0.34)
ball_mat.map = tp.data_texture(_ball_texture(), True)
ball_mesh = tp.Mesh(tp.SphereGeometry(BALL_R, 48, 32), ball_mat)
ball_mesh.cast_shadow = True
scene.add(ball_mesh)

anchor_meshes = []
for _ in range(4):
    m = tp.Mesh(tp.SphereGeometry(0.045, 18, 14), standard_material(0x1b1f25, 0.45, 0.4))
    scene.add(m)
    anchor_meshes.append(m)

# ---- the cannon ---------------------------------------------------------------
# A barrel that visibly carries the aim, so a shot can be lined up before it is
# taken rather than discovered afterwards. CANNON is the trunnion; the ball
# leaves the MUZZLE, so the visual and the ballistics agree.
BARREL_L = 0.62
BARREL_R = 0.075

cannon = tp.Group()
cannon.position.set(*[float(x) for x in CANNON])
scene.add(cannon)

# The trunnion sits at CANNON (0.55 m up, which is where the shot leaves from),
# so the carriage has to reach the floor FROM there: wheel bottom at local
# -0.55 puts it exactly on y = 0 rather than hovering a quarter of a metre up.
_AXLE_Y, _WHEEL_R = -0.27, 0.28
for _z in (-0.145, 0.145):
    _cheek = tp.Mesh(tp.BoxGeometry(0.34, 0.34, 0.035),
                     standard_material(0x343b45, 0.6, 0.45))
    _cheek.position.set(-0.02, -0.15, _z)
    _cheek.cast_shadow = True
    cannon.add(_cheek)
for _z in (-0.19, 0.19):
    _w = tp.Mesh(tp.CylinderGeometry(_WHEEL_R, _WHEEL_R, 0.05, 24),
                 standard_material(0x15181d, 0.55, 0.3))
    _w.rotate_x(math.pi / 2)
    _w.position.set(0.0, _AXLE_Y, _z)
    _w.cast_shadow = True
    cannon.add(_w)
    _hub = tp.Mesh(tp.CylinderGeometry(0.055, 0.055, 0.075, 14),
                   standard_material(0x6b7480, 0.4, 0.8))
    _hub.rotate_x(math.pi / 2)
    _hub.position.set(0.0, _AXLE_Y, _z)
    cannon.add(_hub)
_axle = tp.Mesh(tp.CylinderGeometry(0.022, 0.022, 0.40, 12),
                standard_material(0x22262d, 0.6, 0.5))
_axle.rotate_x(math.pi / 2)
_axle.position.set(0.0, _AXLE_Y, 0.0)
cannon.add(_axle)
# Trail spar down to the ground behind, so it stands on three points like a gun
# rather than on an invisible plinth.
_trail = tp.Mesh(tp.BoxGeometry(0.82, 0.065, 0.11),
                 standard_material(0x343b45, 0.65, 0.4))
_trail.position.set(-0.42, -0.40, 0.0)
_trail.rotate_z(math.radians(-24.0))
_trail.cast_shadow = True
cannon.add(_trail)

barrel_pivot = tp.Group()
cannon.add(barrel_pivot)
_barrel = tp.Mesh(tp.CylinderGeometry(BARREL_R * 0.80, BARREL_R, BARREL_L, 24),
                  standard_material(0x4a515b, 0.30, 0.85))
_barrel.position.set(0.0, BARREL_L / 2, 0.0)
_barrel.cast_shadow = True
barrel_pivot.add(_barrel)
_band = tp.Mesh(tp.CylinderGeometry(BARREL_R * 1.02, BARREL_R * 1.02, 0.05, 24),
                standard_material(0x6b7480, 0.35, 0.9))
_band.position.set(0.0, BARREL_L - 0.035, 0.0)
barrel_pivot.add(_band)
_breech = tp.Mesh(tp.SphereGeometry(BARREL_R * 1.15, 20, 14),
                  standard_material(0x4a515b, 0.30, 0.85))
barrel_pivot.add(_breech)


def aim_dir():
    el, az = math.radians(aim["el"]), math.radians(aim["az"])
    return np.array([math.cos(el) * math.cos(az), math.sin(el),
                     math.cos(el) * math.sin(az)])


def muzzle():
    return CANNON + aim_dir() * BARREL_L


def point_barrel():
    barrel_pivot.rotation.set(0.0, -math.radians(aim["az"]),
                              math.radians(aim["el"]) - math.pi / 2)


point_barrel()

# ---- the aim preview ----------------------------------------------------------
# Ground truth, and deliberately so: this is the PLAYER's aid, not the robot's.
# It is hidden the moment the shot is fired, so nothing it draws can be confused
# with -- or leak into -- what the tracker has to work out for itself. Watching
# the cyan reticle converge onto this amber one IS the perception story.
ARC_N = 72
_arc_geo = tp.BufferGeometry()
_arc_geo.set_attribute("position", np.zeros((ARC_N, 3), np.float32))
_arc_mat = tp.LineBasicMaterial()
_arc_mat.color = 0xffae3a
_arc_mat.transparent = True
_arc_mat.opacity = 0.55
aim_arc = tp.Line(_arc_geo, _arc_mat)
scene.add(aim_arc)

_hit_mat = tp.MeshBasicMaterial()
_hit_mat.color = 0xffae3a
_hit_mat.transparent = True
_hit_mat.opacity = 0.6
_hit_mat.side = tp.Side.Double
aim_hit = tp.Mesh(tp.RingGeometry(0.13, 0.17, 40), _hit_mat)
aim_hit.rotate_x(-math.pi / 2)
scene.add(aim_hit)


def true_crossing(p0, v0, y_plane):
    """Where an unobstructed shot would cross y_plane on the way down."""
    a, b, c = 0.5 * G_NP[1], v0[1], p0[1] - y_plane
    disc = b * b - 4 * a * c
    if disc < 0:
        return None, None
    r = math.sqrt(disc)
    t = max((-b + r) / (2 * a), (-b - r) / (2 * a))
    return (t, p0 + v0 * t + 0.5 * G_NP * t * t) if t > 0 else (None, None)


def update_aim_preview(visible):
    aim_arc.visible = visible
    aim_hit.visible = visible
    if not visible:
        return
    p0, v0 = muzzle(), aim["speed"] * aim_dir()
    t_hit, p_hit = true_crossing(p0, v0, CATCH_Y)
    span = t_hit if t_hit else 1.4
    ts = np.linspace(0.0, span, ARC_N)[:, None]
    pts = p0[None, :] + v0[None, :] * ts + 0.5 * G_NP[None, :] * ts * ts
    _arc_geo.update_attribute("position", pts.astype(np.float32))
    if p_hit is not None:
        aim_hit.visible = True
        aim_hit.position.set(float(p_hit[0]), CATCH_Y + 0.004, float(p_hit[2]))
    else:
        aim_hit.visible = False


if RENDER_SCALE < 0.999:
    renderer.render_scale = RENDER_SCALE
if NO_GI:
    renderer.probe_gi = False
if NO_AO:
    renderer.deferred_ao = False
if NO_DENOISE:
    renderer.denoise = False
# ReSTIR earns its keep with many lights and emissive geometry. This scene has
# two analytic lights and no emitters, which is exactly the case the legacy
# per-light NEE loops handle more cheaply -- and at 1 spp with two lights the
# two paths should agree, so unlike the GI toggles this one is not a look/perf
# trade. (Needs a threepp built after the restir_di binding was added.)
if NO_RESTIR and hasattr(renderer, "restir_di"):
    renderer.restir_di = False


# ---- the arms -----------------------------------------------------------------
# Four 5-DOF arms, defined as a URDF STRING and parsed by the engine's own loader
# rather than fetched: no external asset, but real joints with real limits and
# real speed caps. They are not set dressing -- the cloth corner follows each
# arm's ACHIEVED tool pose, so if an arm cannot get there, the sheet does not go
# there either, and the catch fails for a reason you can point at.
#
# Mounted on pedestals at plate height, which is the whole trick. Floor-mounted
# they would need ~1.25 m of reach just to climb to the catch plane before
# covering any of the corner's ~1.25 m sweep; measured, this arm's practical
# envelope is ~0.85 m horizontal (well under its 1.14 m nominal -- joint limits
# and the base link eat the rest). On a pedestal it only has to cover the sweep,
# and base-to-corner distance stays between 0.5 and 0.95 m throughout.
ARMS = "--no-arms" not in sys.argv
ARM_KIND = cli_arg("--arm", "fr3", str)        # fr3 | iiwa | proc
PEDESTAL_Y = cli_arg("--pedestal-y", 0.50, float)
ARM_OUT = cli_arg("--arm-out", 0.50, float)    # outward from the sheet, in z
ARM_LEAD = cli_arg("--arm-lead", 0.48, float)  # toward the intercept, in x
JOINT_SPEED = cli_arg("--joint-speed", 5.0, float)   # rad/s cap per joint

_L = dict(base=0.16, upper=0.50, fore=0.44, wrist=0.12)


def _link(name, radius, length, rgba, origin_z):
    return f"""
  <link name="{name}">
    <visual>
      <origin xyz="0 0 {origin_z}" rpy="0 0 0"/>
      <geometry><cylinder radius="{radius}" length="{length}"/></geometry>
      <material name="{name}_m"><color rgba="{rgba}"/></material>
    </visual>
    <inertial><mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>"""


def arm_urdf():
    L = _L
    return f"""<?xml version="1.0"?>
<robot name="catcher">
  {_link("base", 0.085, L['base'], "0.13 0.15 0.18 1", L['base'] / 2)}
  {_link("shoulder", 0.070, 0.14, "0.78 0.80 0.85 1", 0.0)}
  {_link("upper", 0.055, L['upper'], "0.78 0.80 0.85 1", L['upper'] / 2)}
  {_link("fore", 0.045, L['fore'], "0.26 0.30 0.36 1", L['fore'] / 2)}
  {_link("wrist", 0.034, L['wrist'], "0.78 0.80 0.85 1", L['wrist'] / 2)}
  {_link("tool", 0.026, 0.06, "0.95 0.42 0.14 1", 0.03)}
  <joint name="pan" type="revolute">
    <parent link="base"/><child link="shoulder"/>
    <origin xyz="0 0 {L['base']}" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="200" velocity="4.0"/>
  </joint>
  <joint name="lift" type="revolute">
    <parent link="shoulder"/><child link="upper"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="200" velocity="4.0"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="upper"/><child link="fore"/>
    <origin xyz="0 0 {L['upper']}" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.6" upper="2.6" effort="200" velocity="5.0"/>
  </joint>
  <joint name="wrist1" type="revolute">
    <parent link="fore"/><child link="wrist"/>
    <origin xyz="0 0 {L['fore']}" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.6" upper="2.6" effort="120" velocity="6.0"/>
  </joint>
  <joint name="wrist2" type="revolute">
    <parent link="wrist"/><child link="tool"/>
    <origin xyz="0 0 {L['wrist']}" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="120" velocity="6.0"/>
  </joint>
</robot>"""


# threepp_data ships a Franka FR3 and a KUKA iiwa. Both measure ~0.90 m of
# practical reach against a corner sweep of ~1.25 m, so the pedestal is doing the
# work: at 0.50 m high and 0.35-0.50 m outboard, both track the corner through
# the whole sweep to under a millimetre. Falls back to the procedural arm when
# the data checkout is not present, so the demo still runs from a bare clone.
_ARM_URDFS = {"fr3": (os.path.join("urdf", "franka", "fr3.urdf"), "fr3_hand_tcp"),
              "iiwa": (os.path.join("urdf", "lbr_iiwa_14_r820.urdf"), None)}
# Franka's ready pose, so the redundant joints settle somewhere an arm would
# actually sit rather than wherever the solver happens to leave them.
_REST = {"fr3": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
         "iiwa": [0.0, 0.6, 0.0, -1.5, 0.0, 1.0, 0.0],
         "proc": [0.0, -0.95, 1.75, -0.80, 0.0]}


def data_dir():
    """threepp_data's checkout. THREEPP_DATA_DIR wins; otherwise the usual
    places -- note the checkout is commonly named with a hyphen."""
    env = os.environ.get("THREEPP_DATA_DIR")
    if env and os.path.isdir(env):
        return env
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for name in ("threepp-data", "threepp_data"):
        cand = os.path.join(os.path.dirname(repo), name)
        if os.path.isdir(cand):
            return cand
    return ""


def _tool_pos(m):
    """Translation of a Matrix4. tool_transform reports in the robot's parent
    frame, which -- measured -- INCLUDES the Robot object's own placement, so
    these come out directly in world coordinates."""
    a = m.to_numpy().reshape(4, 4)
    return a[:3, 3] if abs(a[3, 3] - 1.0) < 1e-6 else a[3, :3]


class Arm:
    def __init__(self, base_xyz, kind):
        rel = _ARM_URDFS.get(kind)
        path = os.path.join(data_dir(), rel[0]) if (rel and data_dir()) else ""
        if path and os.path.isfile(path):
            self.robot = tp.URDFLoader().load(path)
            if rel[1]:
                self.robot.set_end_effector(rel[1])
            self.kind = kind
        else:
            self.robot = tp.URDFLoader().parse(os.getcwd(), arm_urdf())
            self.kind = "proc"
        # The URDF loader gives every collision shape a white wireframe material
        # and adds it alongside the visual mesh, so each link wears a triangulated
        # shell of itself. show_colliders() hides the collider GROUP; the meshes
        # get hidden too so it holds regardless of how visibility is inherited.
        self.robot.show_colliders(False)
        self.robot.position.set(*[float(v) for v in base_xyz])
        self.robot.rotate_x(-math.pi / 2)   # URDF is Z-up; the scene is Y-up
        self.robot.update_matrix()
        opts = tp.IkOptions()
        opts.max_joint_speed = JOINT_SPEED
        opts.position_tolerance = 0.002
        # A 3-DOF position task on a 5-DOF arm leaves two joints free, and left
        # to itself the solver folds them into the shortest pose -- which sits
        # bolt upright against the pedestal and reads as a post, not an arm.
        # Biasing the null space toward shoulder-out / elbow-bent costs nothing
        # in reach and is the difference between four posts and four arms.
        rest = list(_REST.get(self.kind, []))
        if rest:
            rest = (rest + [0.0] * self.robot.num_dof)[:self.robot.num_dof]
            opts.rest_pose = rest
            opts.rest_pose_gain = 0.06
        self.solver = tp.IkSolver(self.robot, opts)
        self.q = list(opts.rest_pose) if opts.rest_pose else [0.0] * self.robot.num_dof
        self.err = 0.0
        scene.add(self.robot)
        ped = tp.Mesh(tp.CylinderGeometry(0.11, 0.13, base_xyz[1], 20),
                      standard_material(0x1b1f25, 0.7, 0.35))
        ped.position.set(float(base_xyz[0]), float(base_xyz[1]) / 2, float(base_xyz[2]))
        ped.cast_shadow = True
        scene.add(ped)

    def settle(self, target):
        """Converge onto a corner with the speed cap OFF, once, before the run.

        The arms are born at q = 0 (straight up) while the sheet corners are
        elsewhere; with a 5 rad/s cap they spend the first second of every run
        catching up, and that transient IS the worst tracking error. A real rig
        starts already holding the cloth."""
        t = tp.Vector3(float(target[0]), float(target[1]), float(target[2]))
        for _ in range(300):
            self.q, _r = self.solver.solve(self.q, t, 0.0)
        self.robot.set_joint_values(self.q)
        return _tool_pos(self.solver.tool_transform(self.q))

    def track(self, target, dt):
        """Drive toward `target` and return where the tool ACTUALLY got to.

        ONE solve per tick. The max_joint_speed cap is applied per CALL, so
        looping the solver N times over the same dt quietly multiplies the cap by
        N -- measured, six calls at a 5 rad/s cap let joints run at 30 rad/s, and
        that is what made the arms look janky. solve() already iterates
        internally (IkOptions.max_iterations, default 100); the outer loop was
        buying nothing but a broken speed limit.
        """
        t = tp.Vector3(float(target[0]), float(target[1]), float(target[2]))
        self.q, _r = self.solver.solve(self.q, t, dt)
        self.robot.set_joint_values(self.q)
        got = _tool_pos(self.solver.tool_transform(self.q))
        self.err = float(np.linalg.norm(got - np.asarray(target, float)))
        return got


arms = []
if ARMS:
    for _k, _c in enumerate(anchor_local):
        arms.append(Arm((HOME[0] + _c[0] + ARM_LEAD,
                         PEDESTAL_Y,
                         HOME[1] + _c[2] + math.copysign(ARM_OUT, _c[2])), ARM_KIND))
    _home = np.array(p0[anchor_idx_np], np.float64)
    _res = np.array([a.settle(_home[k]) for k, a in enumerate(arms)])
    _kind = arms[0].kind
    print(f"arms: 4 x {_kind} ({arms[0].robot.num_dof} dof, tool "
          f"{arms[0].robot.end_effector_link}), pedestals at {PEDESTAL_Y:.2f} m, "
          f"settled onto the corners to {max(a.err for a in arms):.4f} m"
          + ("   [threepp_data not found - procedural fallback]"
             if _kind == "proc" and ARM_KIND != "proc" else ""))


renderer.event_camera_source = "shaded"
renderer.set_event_camera_resolution(SENSOR_W, SENSOR_H)
renderer.event_camera_enabled = True


# ---- state --------------------------------------------------------------------

anchor_from = np.array(p0[anchor_idx_np], np.float64)
anchor_to = anchor_from.copy()

sim_time = 0.0
state = "idle"                # idle -> flight -> catch -> settled / missed
t_fire = None
t_contact = None
v_contact = np.zeros(3)
ball_vy_at_contact = 0.0
t_recover = None
dip_depth = 0.0


def held(p, v):
    """Is the ball still in the sheet? Distance from the rig centre against the
    sheet's own half-width, at rest, and clear of the floor -- not an absolute
    height, which just measures how far the rig happened to dip."""
    return (math.hypot(p[0] - rig.p[0], p[2] - rig.p[2]) < 0.5 * CLOTH
            and p[1] > 2.5 * BALL_R
            and float(np.linalg.norm(v)) < 0.6)
truth_cross = None            # where the ball really crossed the catch plane
plan = None                   # (t_hit, p_hit, v_hit) from the tracker
first_plan = None             # the earliest usable prediction, for the report
approach = None               # (t0, p0, v0) the approach cubic departs from
tracker = EventTracker(FX_PX, FY_PX, EYE, _fwd, _right, _up, SENSOR_W, SENSOR_H)
frames_tracked = 0
ev_overflows = 0
_prev_ball_y = None
obs_err = []
arm_worst = [0.0]
arm_peak = [0.0, 0.0, 'idle']
bearing_err = []
range_err = []


def fire(basis=None):
    global state, t_fire, tracker, frames_tracked, plan, first_plan
    el, az = math.radians(aim["el"]), math.radians(aim["az"])
    v0 = aim["speed"] * np.array([math.cos(el) * math.cos(az), math.sin(el),
                                  math.cos(el) * math.sin(az)])
    bp.assign(np.array([muzzle()], np.float32))
    bv.assign(np.array([v0], np.float32))
    eye, fwd, right, up = basis if basis is not None else (EYE, _fwd, _right, _up)
    tracker = EventTracker(FX_PX, FY_PX, eye, fwd, right, up, SENSOR_W, SENSOR_H)
    frames_tracked, plan, first_plan = 0, None, None
    globals()['approach'] = None
    t_fire, state = sim_time, "flight"


def hermite(p0, v0, p1, v1, T, t):
    """Position and velocity at time t along the cubic that leaves (p0,v0) and
    arrives at (p1,v1) after T.

    The rig RIDES this curve rather than being pushed along it by the
    acceleration at s=0. Applying only a(0) of a plan that is re-solved every
    frame is a lagging controller -- the rig never actually follows the curve it
    just planned, and here it topped out at 1.35 m/s against the 3.05 m/s it
    needed to arrive matched. Riding the curve makes arrival exact by
    construction; the arm's speed limit is then enforced against the curve, so
    an impossible shot shows up as the rig falling behind and missing rather
    than as a controller that quietly cannot track."""
    T = max(T, 1e-3)
    s_ = min(max(t / T, 0.0), 1.0)
    s2, s3 = s_ * s_, s_ * s_ * s_
    h00, h10 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s_
    h01, h11 = -2 * s3 + 3 * s2, s3 - s2
    g00, g10 = 6 * s2 - 6 * s_, 3 * s2 - 4 * s_ + 1
    g01, g11 = -6 * s2 + 6 * s_, 3 * s2 - 2 * s_
    pos_ = h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1
    vel_ = (g00 * p0 + g10 * T * v0 + g01 * p1 + g11 * T * v1) / T
    return pos_, vel_


def substep(alpha=1.0):
    """One physics substep. `alpha` walks the anchors from where they were at the
    start of this tick to where the arms got them, so a 240 Hz IK update does not
    arrive as a step change the cloth has to absorb."""
    anchor_tgt.assign((anchor_from + (anchor_to - anchor_from) * alpha).astype(np.float32))
    impulse.zero_()
    wp.launch(ball_predict, dim=1, device=device, inputs=[bp, bv, bpred, DT])
    wp.launch(integrate, dim=V, device=device, inputs=[pos, prev, pred, im, DT, DAMPING])
    wp.launch(set_anchors, dim=4, device=device, inputs=[pred, anchor_idx, anchor_tgt])
    if solve_graph is not None:
        wp.capture_launch(solve_graph)
        wp.copy(pos, pred)
    else:
        a, b = pred, scratch
        for _ in range(ITERS):
            wp.launch(solve, dim=V, device=device,
                      inputs=[a, b, im, N, N, REST, bpred, bp, BALL_R + 0.012, MU,
                              prev, anchor_tgt, lra, impulse, M_PARTICLE, 0.35])
            a, b = b, a
        wp.copy(pos, a)
    if state != "idle":
        wp.launch(ball_finish, dim=1, device=device,
                  inputs=[bp, bv, impulse, DT, 1.0 / BALL_KG, BALL_R])


def step_frame():
    """Sim, render, read the sensor, re-plan, move the rig. One sensor tick."""
    global sim_time, state, t_contact, v_contact, ball_vy_at_contact
    global t_recover, dip_depth
    global plan, first_plan, frames_tracked, ev_overflows, truth_cross, _prev_ball_y

    import time as _t
    _mark = _t.perf_counter()

    def _lap(name):
        nonlocal _mark
        if PROFILE:
            now = _t.perf_counter()
            prof[name] = prof.get(name, 0.0) + (now - _mark) * 1000.0
            _mark = now

    point_barrel()
    update_aim_preview(state in ("idle", "settled", "missed"))

    dt = 1.0 / SENSOR_HZ
    global anchor_from, anchor_to
    anchor_from = anchor_to.copy()
    cmd = rig.targets()
    if arms:
        anchor_to = np.array([a.track(cmd[k], dt) for k, a in enumerate(arms)])
    else:
        anchor_to = cmd.astype(np.float64)
    for _i in range(SUBSTEPS):
        substep((_i + 1) / SUBSTEPS)
        sim_time += DT
    _lap("sim")

    p_true = bp.numpy()[0].astype(np.float64)
    v_true = bv.numpy()[0].astype(np.float64)
    contact = float(np.linalg.norm(impulse.numpy()[0])) > 0.0
    if arms:
        e = max(a.err for a in arms)
        arm_worst.append(e)
        if e > arm_peak[0]:
            arm_peak[:] = [e, sim_time, state]
    _lap("readback")

    # Ground truth of the crossing, recorded once, purely so the prediction can
    # be scored afterwards. Nothing in the control path reads it.
    if state == "flight" and _prev_ball_y is not None:
        if _prev_ball_y > CATCH_Y >= p_true[1] and truth_cross is None:
            truth_cross = (sim_time, p_true.copy(), v_true.copy())
    _prev_ball_y = p_true[1]

    # --- meshes ---------------------------------------------------------------
    wp.launch(compute_normals, dim=V, device=device, inputs=[pos, nrm, N, N])
    geometry.update_attribute("position", pos.numpy())
    geometry.update_attribute("normal", nrm.numpy())
    ball_mesh.visible = state != "idle"      # loaded in the barrel until fired
    ball_mesh.position.set(*[float(x) for x in p_true])
    for k, m in enumerate(anchor_meshes):
        m.position.set(float(anchor_to[k][0]), float(anchor_to[k][1]), float(anchor_to[k][2]))
        m.visible = not arms          # the tool link IS the gripper once arms exist
    _lap("mesh upload")

    # --- render, then read the sensor ----------------------------------------
    renderer.set_event_camera_params(threshold=EV_THRESHOLD, decay=0.80,
                                     min_luma=0.005, max_events_per_pixel=5,
                                     frame_time_us=int(round(sim_time * 1e6)))
    renderer.render(scene, camera)
    _lap("render")
    ev, overflowed = renderer.read_event_stream(200000)
    if overflowed:
        ev_overflows += 1
    _lap("event read")

    # --- perceive and plan ----------------------------------------------------
    if state == "flight":
        if ORACLE:
            tracker.obs.append((sim_time, p_true[0], p_true[1], p_true[2]))
            frames_tracked += 1
        else:
            est = tracker.observe(ev, sim_time)
            if est is not None:
                frames_tracked += 1
                obs_err.append(est - p_true)
                # Split the error into the two things the estimator actually
                # produces: a BEARING (where in the image) and a RANGE (how far).
                # They come from completely different measurements -- the cluster
                # centre and the cluster size -- and fixing the wrong one is easy.
                b = tracker
                d_ = p_true - b.eye
                zt = float(np.dot(d_, b.fwd))
                tu = SENSOR_W / 2.0 + b.fx * float(np.dot(d_, b.right)) / zt
                tv = SENSOR_H / 2.0 - b.fy * float(np.dot(d_, b.up)) / zt
                lp = b.last_px
                if lp is not None:
                    bearing_err.append(math.hypot(lp[0] - tu, lp[1] - tv))
                    range_err.append(float(np.dot(est - b.eye, b.fwd)) - zt)
        fitted = (tracker.solve_fit() if (ORACLE or SIZE_RANGE)
                  else tracker.solve_fit_bearings())
        if fitted is not None:
            hit = tracker.predict_crossing(CATCH_Y)
            if hit is not None:
                plan = hit
                if first_plan is None:
                    first_plan = (sim_time, hit)

    _lap("track + fit")

    # --- control --------------------------------------------------------------
    if state == "flight" and plan is not None:
        global approach
        t_hit, p_hit, v_hit = plan
        if approach is None:
            approach = (sim_time, rig.p.copy(), rig.v.copy())
        t0, p_start, v_start = approach
        target_p = np.array([p_hit[0], CATCH_Y, p_hit[2]])
        target_v = np.array([v_hit[0], 0.0, v_hit[2]])
        p_cmd, v_cmd = hermite(p_start, v_start, target_p, target_v,
                               t_hit - t0, sim_time - t0)
        rig.goto(dt, p_cmd, v_cmd)
    elif state == "catch":
        u = min((sim_time - t_contact) / ABSORB, 1.0)
        # Ride with the ball, then bleed off. Zero relative velocity at first
        # contact is what stops it skipping straight back out of the sheet.
        v_des = v_contact * (1.0 - u)
        v_des[1] = -dip_depth * (math.pi / ABSORB) * 0.5 * math.sin(math.pi * u)
        rig.step(dt, (v_des - rig.v) / dt)
    elif state == "recover":
        # Cradle, then lift back to the ready height -- which is what a person
        # does, and without it the sheet just stays parked at the bottom of its
        # give with the ball sitting in a pit.
        u = min((sim_time - t_recover) / RECOVER, 1.0)
        v_des = np.zeros(3)
        v_des[1] = dip_depth * (math.pi / RECOVER) * 0.5 * math.sin(math.pi * u)
        rig.spread = PRESENT * 0.5 * (1.0 - math.cos(math.pi * u))
        rig.step(dt, (v_des - rig.v) / dt)
    else:
        rig.step(dt, -rig.v / dt)

    if state == "flight" and contact:
        state = "catch"
        t_contact = sim_time
        v_contact = np.array([rig.v[0], 0.0, rig.v[2]])
        ball_vy_at_contact = float(v_true[1])
        # Give the ball room proportional to how hard it arrived, up to the
        # stroke the arms actually have.
        dip_depth = min(abs(ball_vy_at_contact) * ABSORB / math.pi, GIVE)
    elif state == "catch" and sim_time - t_contact > ABSORB:
        state, t_recover = "recover", sim_time
    elif state == "recover" and sim_time - t_recover > RECOVER + 0.25:
        state = "settled" if held(p_true, v_true) else "missed"

    _lap("control")
    if TRACE and state != "idle":
        pe = "  -" if plan is None else f"{plan[1][0]:+.2f}"
        oe = obs_err[-1] if obs_err else np.zeros(3)
        d_ = p_true - EYE
        zt = float(np.dot(d_, _fwd))
        tu = SENSOR_W / 2.0 + FX_PX * float(np.dot(d_, _right)) / zt
        tv = SENSOR_H / 2.0 - FY_PX * float(np.dot(d_, _up)) / zt
        rt = FY_PX * BALL_R / zt
        lp = tracker.last_px
        print(f"t={sim_time:6.3f} {state:8s} ev={ev.shape[0]:5d} obs={len(tracker.obs):4d} "
              f"pred_x={pe} rig_x={rig.p[0]:+.2f} ball=({p_true[0]:+.2f},{p_true[1]:.2f}) "
              f"obs_err=({oe[0]:+.3f},{oe[1]:+.3f},{oe[2]:+.3f})"
              + (f" | meas uv=({lp[0]:6.1f},{lp[1]:6.1f}) r={lp[2]:5.1f} n={lp[3]:4d}"
                 f"  true uv=({tu:6.1f},{tv:6.1f}) r={rt:5.1f} depth={zt:.2f}"
                 if lp else ""))
    return p_true, v_true


def report(p_true, v_true):
    src = "GROUND TRUTH (--oracle)" if ORACLE else f"{SENSOR_W}x{SENSOR_H} event camera"
    print(f"\n  sensor      {src} at {SENSOR_HZ:.0f} Hz")
    print(f"  sheet       {V} particles, {SUBSTEPS} substeps x {ITERS} iters per tick")
    print(f"  shot        az {aim['az']:+.1f} deg, el {aim['el']:.1f} deg, "
          f"{aim['speed']:.2f} m/s")
    if bearing_err:
        be, re = np.array(bearing_err), np.array(range_err)
        halfway = len(be) // 2
        print(f"  bearing     median {np.median(be):.2f} px, 90th {np.percentile(be, 90):.2f} px"
              f"   (= {np.median(be) / FY_PX * 1000:.2f} mrad)")
        tag = "USED for the fit" if SIZE_RANGE else "SEED ONLY - the fit uses bearings"
        print(f"  size-range  median {np.median(np.abs(re)):.3f} m, bias {re.mean():+.3f} m "
              f"({tag})")
        if not SIZE_RANGE and np.isfinite(tracker.bearing_rms):
            print(f"  fit residual {tracker.bearing_rms:.2f} px rms over "
                  f"{len(tracker.bear)} bearings")
    if obs_err:
        e = np.array(obs_err)
        print(f"  obs error   mean ({e[:,0].mean():+.3f},{e[:,1].mean():+.3f},"
              f"{e[:,2].mean():+.3f}) m, |err| median {np.median(np.linalg.norm(e,axis=1)):.3f} m")
    print(f"  tracked     {frames_tracked} frames, {len(tracker.obs)} observations, "
          f"{tracker.rejected} rejected" + (f", {ev_overflows} OVERFLOWS" if ev_overflows else ""))
    if truth_cross is None:
        print("  RESULT      the ball never reached the catch plane")
        return
    tt, tp_, tv = truth_cross
    print(f"  truth       crossed y={CATCH_Y:.2f} at x{tp_[0]:+.3f} z{tp_[2]:+.3f}, "
          f"t={tt:.3f} s, horizontal {math.hypot(tv[0], tv[2]):.2f} m/s")
    if first_plan is not None:
        t_at, (th, ph, vh) = first_plan
        print(f"  first call  at t={t_at:.3f} s ({1000 * (t_at - t_fire):.0f} ms after firing): "
              f"x{ph[0]:+.3f} z{ph[2]:+.3f}, err {np.linalg.norm(ph - tp_):.3f} m")
    if plan is not None:
        th, ph, vh = plan
        print(f"  final call  x{ph[0]:+.3f} z{ph[2]:+.3f}, "
              f"err {np.linalg.norm(ph - tp_):.3f} m, timing err {1000 * (th - tt):+.0f} ms")
    if arms:
        print(f"  arms        4 x {arms[0].kind}, worst IK error this "
              f"run {max(arm_worst):.4f} m at t={arm_peak[1]:.2f} s ({arm_peak[2]})"
              + ("  (an arm could not hold its corner)" if max(arm_worst) > 0.10 else
                 "  (lag at peak speed, within the joint limits)" if max(arm_worst) > 0.01 else ""))
    print(f"  rig         travelled {rig.travel:.2f} m, peak {rig.peak_speed:.2f} m/s "
          f"(cap {ARM_SPEED:.1f})"
          + (f", SPEED-CAPPED on {rig.starved} frames" if rig.starved else ""))
    if t_contact is not None:
        miss = math.hypot(p_true[0] - rig.p[0], p_true[2] - rig.p[2])
        print(f"  contact     t={t_contact:.3f} s, ball {miss:.3f} m from the sheet centre")
    resting = float(np.linalg.norm(v_true))
    verdict = {"settled": "CAUGHT", "missed": "MISSED", "catch": "still absorbing",
               "recover": "caught, still lifting",
               "flight": "NEVER CONTACTED", "idle": "never fired"}.get(state, state)
    dx = math.hypot(p_true[0] - rig.p[0], p_true[2] - rig.p[2])
    print(f"  RESULT      {verdict} -- ball {dx:.3f} m off the sheet centre "
          f"(half-width {0.5 * CLOTH:.2f}), y={p_true[1]:.2f}, {resting:.2f} m/s")


# ---- run ----------------------------------------------------------------------

TOTAL = int(SECONDS * SENSOR_HZ)


def compose(ev_img):
    """Beauty frame, optionally with the event stream shown alongside it.

    Side by side halves the width the actual subject gets, which on a 4:3 frame
    leaves the rig small. In the film pass the sensor goes in as a corner inset
    instead: same story, and the arms keep the frame.
    """
    rgb = renderer.read_pixels()
    if not SPLIT:
        return rgb
    from PIL import Image
    e = np.stack([ev_img] * 3, axis=-1) if ev_img.ndim == 2 else ev_img
    if not FILM:
        h = rgb.shape[0]
        e = np.asarray(Image.fromarray(e).resize(
            (int(h * e.shape[1] / e.shape[0]), h), Image.NEAREST))
        return np.concatenate([rgb, e], axis=1)
    H, W = rgb.shape[:2]
    iw = int(W * 0.30)
    ih = int(iw * e.shape[0] / e.shape[1])
    e = np.asarray(Image.fromarray(e).resize((iw, ih), Image.BILINEAR))
    m, b = int(W * 0.022), 2                      # margin, border
    out = rgb.copy()
    y0, x0 = H - ih - m, m
    out[y0 - b:y0 + ih + b, x0 - b:x0 + iw + b] = 235
    out[y0:y0 + ih, x0:x0 + iw] = e
    return out


if TUNE:
    import time
    print(f"cloth catch: {V} particles on {device}, {SENSOR_HZ:.0f} Hz sensor tick, "
          f"render scale {RENDER_SCALE:.2f}")
    pt = vt = None
    for i in range(TOTAL):
        if state == "idle" and sim_time >= FIRE_AT:
            fire()
        if i == 60:                        # warm up before timing
            t0 = time.perf_counter()
        pt, vt = step_frame()
    loop_ms = 1000.0 * (time.perf_counter() - t0) / (TOTAL - 60)
    report(pt, vt)
    print(f"  loop        {loop_ms:.2f} ms/tick ({1000.0 / loop_ms:.0f} Hz achievable, "
          f"{SENSOR_HZ:.0f} Hz asked)")
    if PROFILE:
        n = TOTAL
        for k, v in sorted(prof.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<12} {v / n:6.2f} ms/tick  ({100 * v / (loop_ms * n):4.1f}%)")
    sys.exit(0)

if (CLIP or SEQ) and not FRAMES:
    import shutil
    import tempfile
    from PIL import Image
    outdir = SEQ or os.path.join(tempfile.mkdtemp(prefix="clothcatch_"), "f")
    os.makedirs(os.path.dirname(outdir) or ".", exist_ok=True)
    every = max(1, int(round(SENSOR_HZ / CLIP_FPS)))
    written = 0
    pt = vt = None
    for i in range(TOTAL):
        if state == "idle" and sim_time >= FIRE_AT:
            fire()
        pt, vt = step_frame()
        # Real time on the approach, then every tick through the catch -- the sim
        # runs at SENSOR_HZ and the clip plays at CLIP_FPS, so keeping every tick
        # is free slow motion at exactly that ratio. And nothing before the shot
        # is worth watching: the settle is a requirement, not a beat.
        slowmo = FILM and (state in ("catch", "recover")
                           or (plan is not None and 0 < plan[0] - sim_time < 0.22))
        if FILM and sim_time < FIRE_AT - 0.35:
            keep = False
        else:
            keep = slowmo or (i % every == 0)
        if keep:
            Image.fromarray(compose(renderer.read_event_camera_visualisation())).save(
                f"{outdir}_{written:04d}.png")
            written += 1
    report(pt, vt)
    print(f"  wrote {written} frames at {outdir}_*.png")
    if CLIP:
        ff = find_ffmpeg()
        if ff is None:
            print("  no ffmpeg on PATH; frames kept")
        else:
            subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", f"{CLIP_FPS:.0f}",
                            "-i", f"{outdir}_%04d.png", "-an", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                            "-movflags", "+faststart", CLIP], check=True)
            print(f"  wrote {CLIP}")
            if not SEQ:
                shutil.rmtree(os.path.dirname(outdir), ignore_errors=True)
    sys.exit(0)

# --- interactive -----------------------------------------------------------------
# Two things a headless run never exercises, and both bite the same way.
#
# The sheet must be QUIET before a shot. A DVS reports change, so a sheet still
# ringing floods the frame and the tracker locks onto cloth rather than ball.
# The old reset() slammed the cloth back to its domed initial state and fired on
# the same frame, which is why interactive shots tracked to the wrong place while
# headless ones (which settle for FIRE_AT seconds first) were fine. Nothing here
# resets the cloth any more: it is left hanging where it is, and firing is gated
# on measured quiescence instead of on a hopeful delay.
#
# The camera may be moved freely, but the same rule applies to it: while it is
# moving every pixel changes and the detector reports the whole scene. So the
# ready gate wants a still camera too, and the shot is armed from the camera
# pose at the instant it fires.

# Worth knowing before adding any of this: the detector's source is "shaded",
# the raster G-buffer's own shade, which is resolved BEFORE the overlay pass.
# Unlit transparent things -- the sprite panel, the reticle -- and ImGui all land
# in that later pass, so none of them appear in the event stream. The debug view
# does not perturb the measurement it is showing. Verified by eye: the panel does
# not contain a picture of itself.
controls = tp.OrbitControls(camera, canvas)
controls.enable_damping = True
controls.target.set(*[float(x) for x in TGT])
ui = tp.ImguiContext(canvas, renderer)

QUIET_SHEET = cli_arg("--quiet-sheet", 0.08, float)   # m/s, max particle speed
QUIET_FRAMES = int(cli_arg("--quiet-frames", 10, float))

_prev_pos = None
_prev_cam = None
_still_for = 0
sheet_speed = 1e9
_dbg_tick = 0
cam_moved_in_flight = False
last_result = ""


def camera_basis():
    """Eye and orthonormal frame of the LIVE camera, so a shot is aimed from
    wherever the user has actually put it."""
    eye = np.array([camera.position.x, camera.position.y, camera.position.z])
    tgt = np.array([controls.target.x, controls.target.y, controls.target.z])
    fwd = tgt - eye
    n = float(np.linalg.norm(fwd))
    fwd = fwd / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    rn = float(np.linalg.norm(right))
    right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])
    return eye, fwd, right, np.cross(right, fwd)


def project_sensor(p, basis):
    eye, fwd, right, up = basis
    d = np.asarray(p, float) - eye
    z = float(np.dot(d, fwd))
    if z < 0.05:
        return None
    return (SENSOR_W / 2.0 + FX_PX * float(np.dot(d, right)) / z,
            SENSOR_H / 2.0 - FY_PX * float(np.dot(d, up)) / z)


# --- the debug panel ------------------------------------------------------------
# Parented to the CAMERA, not drawn in a second ortho pass: VulkanRenderer has
# no auto_clear, and its screen-space layer is the overlay pass, which unlit
# transparent materials (a SpriteMaterial) land in on their own. One unit in
# front of the camera the frustum is 2*tan(fov/2) high, so the panel is sized
# and cornered in those units.
_h1 = 2.0 * math.tan(math.radians(45) / 2.0)      # frustum height at z = -1
_w1 = _h1 * VIEW_W / VIEW_H
DBG_H3 = 0.30 * _h1
DBG_W3 = DBG_H3 * SENSOR_W / SENSOR_H
dbg_tex = tp.data_texture(np.zeros((SENSOR_H, SENSOR_W, 3), np.uint8), False)
_spr_mat = tp.SpriteMaterial()
_spr_mat.map = dbg_tex
panel = tp.Sprite(_spr_mat)
panel.scale.set(DBG_W3, DBG_H3, 1)
panel.position.set(_w1 / 2 - DBG_W3 / 2 - 0.02, _h1 / 2 - DBG_H3 / 2 - 0.02, -1.0)
camera.add(panel)
scene.add(camera)          # so the camera's children get their transforms updated

# Where the tracker currently thinks the ball will cross the catch plane. A flat
# reticle lying IN that plane, not a sphere -- a sphere out in the scene just
# reads as a second ball to catch.
_ring_mat = tp.MeshBasicMaterial()
_ring_mat.color = 0x2ad4ff
_ring_mat.transparent = True
_ring_mat.opacity = 0.75
_ring_mat.side = tp.Side.Double
aim_mark = tp.Mesh(tp.RingGeometry(0.17, 0.21, 40), _ring_mat)
aim_mark.rotate_x(-math.pi / 2)
aim_mark.visible = False
scene.add(aim_mark)


def _box(img, cx, cy, r, col):
    h, w = img.shape[:2]
    x0, x1 = int(max(0, cx - r)), int(min(w - 1, cx + r))
    y0, y1 = int(max(0, cy - r)), int(min(h - 1, cy + r))
    if x1 <= x0 or y1 <= y0:
        return
    img[y0, x0:x1] = col
    img[y1, x0:x1] = col
    img[y0:y1, x0] = col
    img[y0:y1, x1] = col


def _cross(img, cx, cy, r, col):
    h, w = img.shape[:2]
    x, y = int(cx), int(cy)
    if 0 <= y < h:
        img[y, int(max(0, x - r)):int(min(w, x + r))] = col
    if 0 <= x < w:
        img[int(max(0, y - r)):int(min(h, y + r)), x] = col


def build_debug(ev_img, basis):
    """The raw event frame with what the tracker made of it drawn on top:
    green = the cluster it locked onto and its search gate, blue = the predicted
    crossing. When those disagree with the ball, this is where you see it."""
    if ev_img is None or ev_img.size == 0:
        return
    rgb = np.repeat(ev_img[:, :, None], 3, axis=2).astype(np.float32)
    rgb *= 0.7                                   # dim, so the overlays read
    rgb = rgb.astype(np.uint8)
    lp = tracker.last_px
    if lp is not None:
        _box(rgb, lp[0], lp[1], max(2.5 * lp[2], 10), (40, 230, 120))
        _cross(rgb, lp[0], lp[1], 7, (40, 230, 120))
    if plan is not None:
        uv = project_sensor(np.array([plan[1][0], CATCH_Y, plan[1][2]]), basis)
        if uv is not None:
            _cross(rgb, uv[0], uv[1], 11, (60, 200, 255))
    # data_texture takes row 0 as v = 0 (bottom) while the sensor hands back row
    # 0 = top, so it goes up the other way without this flip.
    dbg_tex.update_data(np.ascontiguousarray(rgb[::-1]))


# --- input ----------------------------------------------------------------------
_held = {}


def pressed(k):
    now = canvas.is_key_down(k)
    fired = now and not _held.get(k, False)
    _held[k] = now
    return fired


def ready():
    return (state in ("idle", "settled", "missed")
            and sheet_speed < QUIET_SHEET and _still_for >= QUIET_FRAMES)


def arm():
    """Fire from the live camera pose. The cloth is deliberately NOT reset --
    it is already hanging quiet, and resetting it is what broke this path."""
    global cam_moved_in_flight, last_result
    bp.assign(np.array([muzzle()], np.float32))
    bv.zero_()
    rig.__init__()
    cam_moved_in_flight = False
    last_result = ""
    fire(camera_basis())


def draw_ui():
    tp.imgui.set_next_window_pos(14, 14)
    tp.imgui.set_next_window_size(420, 0)
    tp.imgui.begin("cloth catch")
    tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps   sensor {SENSOR_HZ:.0f} Hz")
    tp.imgui.separator()
    tp.imgui.text(f"azimuth    {aim['az']:+6.1f} deg      A / D")
    tp.imgui.text(f"elevation  {aim['el']:6.1f} deg      W / S")
    tp.imgui.text(f"muzzle     {aim['speed']:6.2f} m/s      Q / E")
    tp.imgui.separator()
    if state == "flight":
        n = len(tracker.obs)
        if plan is None:
            tp.imgui.text(f"TRACKING   {n} obs, no confident fit yet")
        else:
            tp.imgui.text(f"TRACKING   {n} obs")
            tp.imgui.text(f"intercept  x {plan[1][0]:+.2f}  z {plan[1][2]:+.2f}"
                          f"   in {max(plan[0] - sim_time, 0.0) * 1000:.0f} ms")
        if cam_moved_in_flight:
            tp.imgui.text("camera moved mid-flight - tracking degraded")
    elif ready():
        tp.imgui.text("READY - SPACE to fire")
    else:
        why = []
        if sheet_speed >= QUIET_SHEET:
            why.append(f"sheet settling ({sheet_speed:.2f} m/s)")
        if _still_for < QUIET_FRAMES:
            why.append("camera moving")
        if state in ("catch", "recover"):
            why.append("catching")
        tp.imgui.text("WAIT - " + ", ".join(why or ["..."]))
        tp.imgui.text("a DVS needs a still scene to see one moving thing")
    if last_result:
        tp.imgui.separator()
        tp.imgui.text(last_result)
    tp.imgui.separator()
    tp.imgui.text("mouse orbits - R resets")
    tp.imgui.end()


def animate():
    global _prev_pos, _prev_cam, _still_for, sheet_speed, _dbg_tick
    global cam_moved_in_flight, last_result, state

    if canvas.is_key_down("A"):
        aim["az"] -= 0.6
    if canvas.is_key_down("D"):
        aim["az"] += 0.6
    if canvas.is_key_down("W"):
        aim["el"] = min(aim["el"] + 0.4, 85.0)
    if canvas.is_key_down("S"):
        aim["el"] = max(aim["el"] - 0.4, 10.0)
    if canvas.is_key_down("Q"):
        aim["speed"] = max(aim["speed"] - 0.03, 2.0)
    if canvas.is_key_down("E"):
        aim["speed"] = min(aim["speed"] + 0.03, 12.0)

    cam = np.array([camera.position.x, camera.position.y, camera.position.z,
                    controls.target.x, controls.target.y, controls.target.z])
    if _prev_cam is not None and np.abs(cam - _prev_cam).max() < 1e-4:
        _still_for += 1
    else:
        _still_for = 0
        if state == "flight":
            cam_moved_in_flight = True
    _prev_cam = cam

    if (pressed("SPACE") or (AUTOFIRE and state in ("idle",))) and ready():
        arm()
    if pressed("R"):
        bp.assign(np.array([muzzle()], np.float32))
        bv.zero_()
        rig.__init__()
        state, last_result = "idle", ""

    controls.update()
    basis = camera_basis()
    aim_mark.visible = plan is not None and state == "flight"
    if aim_mark.visible:
        aim_mark.position.set(float(plan[1][0]), CATCH_Y, float(plan[1][2]))

    was = state
    p_true, v_true = step_frame()      # this is the render the detector samples
    if state in ("settled", "missed") and was not in ("settled", "missed"):
        dx = math.hypot(p_true[0] - rig.p[0], p_true[2] - rig.p[2])
        last_result = ("CAUGHT" if state == "settled" else "MISSED") + \
                      f" - ball {dx:.2f} m off the sheet centre"

    now = pos.numpy()
    if _prev_pos is not None and _prev_pos.shape == now.shape:
        sheet_speed = float(np.abs(now - _prev_pos).max()) * SENSOR_HZ
    _prev_pos = now.copy()

    global _dbg_tick
    _dbg_tick += 1
    if _dbg_tick % max(1, int(round(SENSOR_HZ / 60.0))) == 0:
        build_debug(renderer.read_event_camera_visualisation(), basis)
    ui.render(draw_ui)


print("cloth catch:  A/D aim   W/S elevation   Q/E power   SPACE fire   R reset"
      "   mouse orbits")

if FRAMES:
    from PIL import Image
    n = 0
    while n < FRAMES and canvas.animate_once(animate):
        n += 1
        if SEQ and n % 20 == 0:
            Image.fromarray(renderer.read_pixels()).save(f"{SEQ}_{n:05d}.png")
    print(f"  ran {n} interactive frames, state={state}, "
          f"sheet {sheet_speed:.3f} m/s, still {_still_for} frames")
    print(f"  {last_result or 'no shot completed'}")
else:
    canvas.animate(animate)
