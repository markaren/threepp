"""Four arms catch a cannonball in a cloth, tracking it with the engine's own event camera.

Aim the cannon, fire, and a rig of four anchors holding a sheet works out where
the ball will be and gets there in time. The rig is never told the ball's true
position: it sees only what a 640x480 DVS sees -- a sparse stream of brightness
changes -- and everything else is inferred from that.

Three things make this work, and each of them is a measurement, not a guess:

  * Range from apparent size. Only the PRIMARY Vulkan view runs the event
    detector (secondary views are "measurement cameras... no lens or sensor
    model"), so stereo is off the table. A DVS fires on a moving ball's
    silhouette, so the RMS radius of the event cloud is its apparent radius --
    measured at 0.989 of ground truth, i.e. essentially uncalibrated -- and a
    known ball radius turns that into depth.

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

    python warp_cloth_catch.py                      # window; A/D aim, W/S elevate,
                                                    # Q/E power, SPACE fire, R reset
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
ORACLE = "--oracle" in sys.argv        # bypass the sensor; use true ball state
TRACE = "--trace" in sys.argv
HEADLESS = bool(TUNE or CLIP or SEQ)

# ---- the cannon --------------------------------------------------------------
CANNON = np.array([float(x) for x in cli_arg("--cannon", "-2.40,0.55,0", str).split(",")])
AZ = cli_arg("--az", 0.0, float)             # degrees; 0 fires along +x
EL = cli_arg("--el", 62.0, float)            # degrees above horizontal
MUZZLE = cli_arg("--speed", 6.5, float)      # m/s
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
VIEW_W, VIEW_H = parse_size(cli_arg("--size", "960x720", str))
EV_THRESHOLD = cli_arg("--ev-threshold", 0.15, float)
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
        self.last_px = None
        self.recent = []

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
EYE = np.array([0.40, 3.05, 4.90])
TGT = np.array([-0.50, 1.05, 0.0])
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

ground = tp.Mesh(tp.PlaneGeometry(40, 40), standard_material(0x24282e, 0.95))
ground.rotate_x(-math.pi / 2)
ground.receive_shadow = True
scene.add(ground)

geometry = tp.PlaneGeometry(CLOTH, CLOTH, N, N)
cloth_mesh = tp.Mesh(geometry, standard_material(0xd4542e, 0.88, side=tp.Side.Double))
cloth_mesh.cast_shadow = True
scene.add(cloth_mesh)

# Bright ball on a dark set: that contrast is what the detector actually runs on.
ball_mesh = tp.Mesh(tp.SphereGeometry(BALL_R, 40, 28), standard_material(0xf2f5ff, 0.35))
ball_mesh.cast_shadow = True
scene.add(ball_mesh)

anchor_meshes = []
for _ in range(4):
    m = tp.Mesh(tp.SphereGeometry(0.045, 18, 14), standard_material(0x1b1f25, 0.45, 0.4))
    scene.add(m)
    anchor_meshes.append(m)

cannon_mesh = tp.Mesh(tp.BoxGeometry(0.42, 0.30, 0.30), standard_material(0x3a4048, 0.7, 0.3))
cannon_mesh.position.set(*[float(x) for x in CANNON])
cannon_mesh.cast_shadow = True
scene.add(cannon_mesh)

renderer.event_camera_source = "shaded"
renderer.set_event_camera_resolution(SENSOR_W, SENSOR_H)
renderer.event_camera_enabled = True


# ---- state --------------------------------------------------------------------

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
aim = {"az": AZ, "el": EL, "speed": MUZZLE}
tracker = EventTracker(FX_PX, FY_PX, EYE, _fwd, _right, _up, SENSOR_W, SENSOR_H)
frames_tracked = 0
ev_overflows = 0
_prev_ball_y = None
obs_err = []


def fire():
    global state, t_fire, tracker, frames_tracked, plan, first_plan
    el, az = math.radians(aim["el"]), math.radians(aim["az"])
    v0 = aim["speed"] * np.array([math.cos(el) * math.cos(az), math.sin(el),
                                  math.cos(el) * math.sin(az)])
    bp.assign(np.array([CANNON], np.float32))
    bv.assign(np.array([v0], np.float32))
    tracker = EventTracker(FX_PX, FY_PX, EYE, _fwd, _right, _up, SENSOR_W, SENSOR_H)
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


def substep():
    """One physics substep. The rig is kinematic, so it is simply written in."""
    anchor_tgt.assign(rig.targets())
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

    dt = 1.0 / SENSOR_HZ
    for _ in range(SUBSTEPS):
        substep()
        sim_time += DT

    p_true = bp.numpy()[0].astype(np.float64)
    v_true = bv.numpy()[0].astype(np.float64)
    contact = float(np.linalg.norm(impulse.numpy()[0])) > 0.0

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
    ball_mesh.position.set(*[float(x) for x in p_true])
    tg = rig.targets()
    for k, m in enumerate(anchor_meshes):
        m.position.set(float(tg[k][0]), float(tg[k][1]), float(tg[k][2]))

    # --- render, then read the sensor ----------------------------------------
    renderer.set_event_camera_params(threshold=EV_THRESHOLD, decay=0.80,
                                     min_luma=0.005, max_events_per_pixel=5,
                                     frame_time_us=int(round(sim_time * 1e6)))
    renderer.render(scene, camera)
    ev, overflowed = renderer.read_event_stream(200000)
    if overflowed:
        ev_overflows += 1

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
        if tracker.solve_fit() is not None:
            hit = tracker.predict_crossing(CATCH_Y)
            if hit is not None:
                plan = hit
                if first_plan is None:
                    first_plan = (sim_time, hit)

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
    """Beauty frame, optionally with the event stream beside it."""
    rgb = renderer.read_pixels()
    if not SPLIT:
        return rgb
    from PIL import Image
    h = rgb.shape[0]
    e = np.stack([ev_img] * 3, axis=-1) if ev_img.ndim == 2 else ev_img
    e = np.asarray(Image.fromarray(e).resize((int(h * e.shape[1] / e.shape[0]), h),
                                             Image.NEAREST))
    return np.concatenate([rgb, e], axis=1)


if TUNE:
    print(f"cloth catch: {V} particles on {device}, {SENSOR_HZ:.0f} Hz sensor tick")
    pt = vt = None
    for i in range(TOTAL):
        if state == "idle" and sim_time >= FIRE_AT:
            fire()
        pt, vt = step_frame()
    report(pt, vt)
    sys.exit(0)

if CLIP or SEQ:
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
        if i % every == 0:
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

# --- interactive ---------------------------------------------------------------
print("cloth catch:  A/D aim   W/S elevation   Q/E power   SPACE fire   R reset")
_held = {}


def pressed(k):
    now = canvas.is_key_down(k)
    fired = now and not _held.get(k, False)
    _held[k] = now
    return fired


def reset():
    global state, sim_time, t_contact, truth_cross, _prev_ball_y
    pos.assign(p0)
    prev.assign(p0)
    bp.assign(np.array([CANNON], np.float32))
    bv.zero_()
    rig.__init__()
    state, t_contact, truth_cross, _prev_ball_y = "idle", None, None, None


def animate():
    if canvas.is_key_down("A"):
        aim["az"] += 0.6
    if canvas.is_key_down("D"):
        aim["az"] -= 0.6
    if canvas.is_key_down("W"):
        aim["el"] = min(aim["el"] + 0.4, 85.0)
    if canvas.is_key_down("S"):
        aim["el"] = max(aim["el"] - 0.4, 10.0)
    if canvas.is_key_down("Q"):
        aim["speed"] = max(aim["speed"] - 0.02, 2.0)
    if canvas.is_key_down("E"):
        aim["speed"] = min(aim["speed"] + 0.02, 12.0)
    if pressed("SPACE") and state in ("idle", "settled", "missed"):
        reset()
        fire()
    if pressed("R"):
        reset()
    step_frame()


canvas.animate(animate)
