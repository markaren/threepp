"""The snake robot's inspection mission: sonar wall-following, a tear detector, the state machine.

Pure code, like netpen_e3.py: no scene import, no GPU. The scene (snake_netpen.py --mission) feeds
it the head sonar's images and the robot's own state; this module answers with a gait and a
steering offset phi0 for snake_model.Snake. On its own:

    python snake_mission.py --selftest            # CPU: detector + a unicycle stand-in, 3 seeds
    python snake_mission.py --selftest --physx    # + the real PhysX snake on the synthetic sonar

GUIDANCE (FOLLOW_WALL / INSPECT / RETURN). A line-of-sight wall follower in the snake's MEAN-heading
frame. The sonar rides on the head, which yaws with the gait (head-yaw RMS 31 deg lateral, 13 deg
eel-like), so the wall tangent the sonar measures in the head frame is re-expressed in the world
with the robot's own joint angles: head yaw - mean yaw = c_(n-1) - mean_k c_k, c = cumsum(phi)
(the IMU gives the head's yaw; nothing here reads the truth for this). Then, per sonar image,

    psi_wall = (psi_mean + dpsi_head) - tangent                   (world yaw of the wall ahead)
    e        = standoff_meas - standoff                           (+ = too far off the net)
    z       += k_i e dt, |z| <= z_max                             (slow integral: the current)
    psi_ref  = psi_wall - side * atan((e + z) / Delta)            (side +1 = wall to starboard)

and every scene step phi0 = clip(k_theta * wrap(psi_ref - psi_mean), +-25 deg). phi0 > 0 turns
toward +yaw (port). standoff_meas = min(first-echo standoff, the fitted wall line's perpendicular
distance), low-passed over a gait period: the min range alone swings with the head's yaw.

TEAR DETECTOR. Per image, on the wall side: the wall's expected range per beam is a wide
nan-median of the first echoes (the gap's neighbours carry it across the gap); a beam is a gap
beam where the first echo is missing or > expected + 1.0 m, or where the wall echo is < 25 % of
the local wall level (the 28 deg vertical fan straddles a 1.3 m hole except near abeam, so a
partly covered beam still echoes, only weaker). A contiguous run of gap beams with the wall on
both sides, at an expected range < 3 m, whose chord on the wall is 0.4-1.8 m, is a candidate;
its centre goes to the world with the head's pose at the image. It fires when >= 3 of the last
5 images hold a candidate within 6 deg of pen bearing of each other. If the head passes the
reported sector (+-20 deg around the reported tear bearing) without a fire, it fires in 'sector'
mode on the best sonar candidate inside the sector ('sector+sonar') or on the reported bearing.

STATE MACHINE. DOCKED -> UNDOCK (eel-like, heading held on the cradle axis, until the tail is out)
-> ACQUIRE (turn toward the wall until it is seen) -> FOLLOW_WALL (lateral, 1.5 m) -> TEAR (the
detector fired) -> INSPECT (eel-like, 1.0 m, until 1.5 m past the tear) -> UTURN (turn inboard)
-> INSPECT_2 (eel-like, 1.0 m, the wall now on the other side) -> RETURN (lateral, 1.5 m) ->
APPROACH (LOS onto the cradle axis from a USBL-like fix: true pose + seeded 5 cm noise at 1 Hz,
dead-reckoned between fixes) -> DOCK (slow eel-like; latch when the nose is within 0.15 m of the
latch point and the mean heading within 15 deg of the axis) -> DOCKED. The latch then pulls the
nose the last few cm with a capped spring (<= 3 N, reported) while the gait amplitude ramps to
zero and the joint drives straighten the body. That pull is the only non-paper force.

Conventions (the scene's): forward +x at yaw 0, starboard +z, y up; yaw = atan2(-z, x); pen
bearing = atan2(z, x); sonar bearings positive to starboard (netpen_e3).
"""
import argparse
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import netpen_e3 as e3

D2R = math.radians
R2D = math.degrees

# ---- the pen and the mission geometry (mirrors warp_netpen.py / snake_netpen.py) ------------------
PEN_R = 7.0
DEPTH_Y = -3.0
TEAR_R = 0.65
_SUN = np.array([0.62, 0.30])
TEAR_BEARING = math.atan2(-_SUN[1], -_SUN[0])             # -154.2 deg, warp_netpen.TEAR_TH
DOCK_BEARING = D2R(-95.0)
R_DOCK = 5.5
DOCK_LEN = 2.0
LINK_N, LINK_L, LINK_R = 9, 0.18, 0.0525
SON_TILT, SON_EL = -6.0, 14.0
STANDOFF_FOLLOW, STANDOFF_INSPECT = 1.5, 1.0


def polar(th, r, y=DEPTH_Y):
    return np.array([r * math.cos(th), y, r * math.sin(th)])


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_of(v):
    return math.atan2(-float(v[2]), float(v[0]))


def yaw_dir(h):
    return np.array([math.cos(h), 0.0, -math.sin(h)])


def bearing_of(p):
    return math.atan2(float(p[2]), float(p[0]))


DOCK_C = polar(DOCK_BEARING, R_DOCK)
DOCK_U = np.array([math.sin(DOCK_BEARING), 0.0, -math.cos(DOCK_BEARING)])   # cradle axis, toward the tear
DOCK_A = -DOCK_U                                          # the return direction through the cradle
DOCK_N = np.cross(DOCK_A, [0.0, 1.0, 0.0])                # horizontal normal of the approach line
LATCH_P = DOCK_C + 0.5 * LINK_N * LINK_L * DOCK_A         # the docked nose: body centred in the cradle
HEADING_OUT = yaw_of(DOCK_U)

# ---- gaits: measured on snake_model (complex coefficients, 20-30 s runs, still water) -------------
#   name        pattern    alpha  omega  delta   speed    head-yaw RMS   yaw rate per deg phi0
GAITS = {
    "cruise":  ("lateral", 30.0, 150.0, 30.0),       # 0.35 m/s   31 deg        ~2.2 /s
    "inspect": ("eel",     40.0, 150.0, 30.0),       # 0.22 m/s   13 deg        ~1.4 /s
    "dock":    ("eel",     30.0, 120.0, 30.0),       # 0.15 m/s   11 deg        ~0.9 /s
    "undock":  ("lateral", 20.0, 150.0, 30.0),       # 0.26 m/s   20 deg        (the cradle: headway into the current)
    "stop":    ("eel",      0.0,   0.0, 30.0),
}
GAIT_SPEED = {"cruise": 0.35, "inspect": 0.22, "dock": 0.15, "undock": 0.26, "stop": 0.0}
GAIT_TURN = {"cruise": 2.2, "inspect": 1.4, "dock": 0.93, "undock": 1.5, "stop": 0.0}
GAIT_HEADYAW = {"cruise": D2R(31.0), "inspect": D2R(13.0), "dock": D2R(11.0), "undock": D2R(20.0), "stop": 0.0}
GAIT_OMEGA = {k: D2R(v[2]) for k, v in GAITS.items()}

PHASES = ("DOCKED", "UNDOCK", "ACQUIRE", "FOLLOW_WALL", "TEAR", "INSPECT", "UTURN", "INSPECT_2",
          "RETURN", "APPROACH", "DOCK", "LATCHED", "ABORT")


# ---- the robot's own state, as the mission sees it -------------------------------------------------
class RobotState:
    """What the robot knows about itself each step (IMU/compass on the head, joint encoders, and
    the true pose only through Mission's USBL fix)."""
    __slots__ = ("t", "links", "nose", "head_pos", "head_yaw", "yaw_mean", "dpsi_head", "com", "speed")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


def dpsi_from_joints(phi):
    """Head yaw minus mean link yaw from the joint angles alone (phi[k] = yaw(k+1) - yaw(k), tail first)."""
    c = np.concatenate([[0.0], np.cumsum(np.asarray(phi, np.float64))])
    return float(c[-1] - c.mean())


# ---- sonar interpretation ----------------------------------------------------------------------------
def _nanmed_window(r, k):
    import warnings
    pad = np.pad(r, k // 2, mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(pad, k)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(win, axis=1)


def wall_ranges(intensity, rel=0.06, abs_floor=0.01, max_r=10.0, smooth=3):
    """Per-beam range of the LAST echo above threshold inside max_r (NaN where none). Inside the pen
    everything else (fish, the collar's shadow) is nearer than the net, so the last echo is the wall
    and the school cannot pull the standoff in; a beam through the hole has no last echo either."""
    a = np.asarray(intensity, np.float32)
    bins = a.shape[1]
    bin_m = e3.SON_RANGE / bins
    thr = max(abs_floor, rel * float(a.max()) if a.size else 0.0)
    b0, b1 = int(math.ceil(e3.SON_MIN_RANGE / bin_m)), min(bins, int(max_r / bin_m))
    hit = a[:, b0:b1] >= thr
    last = hit.shape[1] - 1 - np.argmax(hit[:, ::-1], axis=1)
    r = (b0 + last + 0.5) * bin_m
    r[~hit.any(axis=1)] = np.nan
    if smooth and smooth > 1:
        med = _nanmed_window(r, int(smooth) | 1)
        r = np.where(np.isnan(r), np.nan, med)
    return r


class TearDetector:
    """Gap in the net wall from the head sonar; see the module docstring."""

    def __init__(self, jump=1.0, weak=0.25, max_ref=3.0, min_w=0.4, max_w=1.8, win=41,
                 persist=3, window=5, agree=D2R(6.0), min_side=D2R(12.0)):
        self.jump, self.weak, self.max_ref = jump, weak, max_ref
        self.min_w, self.max_w, self.win = min_w, max_w, win
        self.persist, self.window, self.agree, self.min_side = persist, window, agree, min_side
        self.hist = []                                    # per image: candidate pen bearing or None
        self.all_cands = []                               # (t, bearing, width, range) for the log
        self.fired, self.bearing, self.mode, self.t_fire = False, float("nan"), "", float("nan")
        self.last = None

    def candidates(self, intensity, r, b, side, pos, yaw):
        """Candidate gaps in one image: list of (pen bearing, chord width m, expected range m)."""
        if side not in (-1, 1):
            return []
        a = np.asarray(intensity, np.float32)
        bins = a.shape[1]
        bin_m = e3.SON_RANGE / bins
        wall_side = side * b > self.min_side
        rr = np.where(wall_side, r, np.nan)
        ref = _nanmed_window(rr, self.win)
        # the wall echo level per beam, around the expected range
        rb = np.clip(np.nan_to_num(ref / bin_m, nan=0).astype(int), 0, bins - 1)
        idx = np.clip(rb[:, None] + np.arange(-4, 5)[None, :], 0, bins - 1)
        level = a[np.arange(len(r))[:, None], idx].max(1)
        lref = _nanmed_window(np.where(wall_side & np.isfinite(r), level, np.nan), self.win)
        ok = wall_side & np.isfinite(ref) & (ref < self.max_ref)
        gap = ok & (~np.isfinite(r) | (r > ref + self.jump) | (level < self.weak * np.nan_to_num(lref, nan=0.0)))
        out = []
        n = len(r)
        k = 0
        while k < n:
            if not gap[k]:
                k += 1
                continue
            j = k
            while j + 1 < n and gap[j + 1]:
                j += 1
            lo, hi = k, j
            k = j + 1
            # the wall on both sides of the run
            left = [i for i in range(max(0, lo - 8), lo) if np.isfinite(r[i]) and abs(r[i] - ref[i]) < 0.5]
            right = [i for i in range(hi + 1, min(n, hi + 9)) if np.isfinite(r[i]) and abs(r[i] - ref[i]) < 0.5]
            if len(left) < 3 or len(right) < 3:
                continue
            rc = float(np.nanmedian(ref[lo:hi + 1]))
            p_lo = pos + rc * yaw_dir(yaw - (b[lo] - 0.5 * abs(b[1] - b[0])))
            p_hi = pos + rc * yaw_dir(yaw - (b[hi] + 0.5 * abs(b[1] - b[0])))
            w = float(np.linalg.norm((p_hi - p_lo)[[0, 2]]))
            if not (self.min_w <= w <= self.max_w):
                continue
            pc = pos + rc * yaw_dir(yaw - 0.5 * (b[lo] + b[hi]))
            out.append((bearing_of(pc), w, rc))
        return out

    def update(self, t, intensity, r, b, side, pos, yaw):
        c = self.candidates(intensity, r, b, side, pos, yaw)
        best = min(c, key=lambda q: q[2]) if c else None  # the nearest: the one the fan sees whole
        for q in c:
            self.all_cands.append((t, *q))
        self.hist.append(None if best is None else best[0])
        self.hist = self.hist[-self.window:]
        self.last = best
        if self.fired or best is None:
            return False
        near = [h for h in self.hist if h is not None and abs(wrap(h - best[0])) < self.agree]
        if len(near) >= self.persist:
            self.fired, self.mode, self.t_fire = True, "sonar", t
            self.bearing = float(np.median(near))
            return True
        return False

    def fire_sector(self, t, reported, sector=D2R(20.0)):
        inside = [q[1] for q in self.all_cands if abs(wrap(q[1] - reported)) < sector]
        if inside:
            self.bearing, self.mode = float(np.median(inside)), "sector+sonar"
        else:
            self.bearing, self.mode = reported, "sector"
        self.fired, self.t_fire = True, t


class WallFollower:
    """LOS/ILOS wall follower in the mean-heading frame (module docstring)."""

    def __init__(self, standoff=STANDOFF_FOLLOW, side=1, delta=1.2, k_theta=0.5, ki=0.05, z_max=0.6,
                 phi0_max=D2R(25.0), lp_tau=0.8):
        self.standoff, self.side, self.delta, self.k_theta = standoff, side, delta, k_theta
        self.ki, self.z_max, self.phi0_max, self.lp_tau = ki, z_max, phi0_max, lp_tau
        self.z, self.meas, self.psi_ref, self.misses, self.e = 0.0, float("nan"), None, 0, float("nan")
        self.psi_wall = float("nan")

    def observe(self, est, yaw_head, dt):
        if est.seen and np.isfinite(est.tangent):
            m = est.standoff if not np.isfinite(est.perp) else min(est.standoff, est.perp)
            a = 1.0 - math.exp(-dt / self.lp_tau)
            self.meas = m if not np.isfinite(self.meas) else self.meas + a * (m - self.meas)
            self.e = self.meas - self.standoff
            if abs(self.e) < 0.5:                                # anti-windup: integrate near the setpoint only
                self.z = float(np.clip(self.z + self.ki * self.e * dt, -self.z_max, self.z_max))
            pw = yaw_head - est.tangent
            self.psi_wall = pw if not np.isfinite(self.psi_wall) else self.psi_wall + a * wrap(pw - self.psi_wall)
            corr = float(np.clip(math.atan((self.e + self.z) / self.delta), -D2R(45.0), D2R(45.0)))
            self.psi_ref = self.psi_wall - self.side * corr
            self.misses = 0
        else:
            self.misses += 1

    def phi0(self, yaw_mean):
        if self.psi_ref is None or self.misses > 30:
            ref = yaw_mean - self.side * D2R(15.0)          # lost the wall: turn gently toward its side
        else:
            ref = self.psi_ref
        return float(np.clip(self.k_theta * wrap(ref - yaw_mean), -self.phi0_max, self.phi0_max))


class Mission:
    """The state machine. Call on_sonar() with each image (and the sonar pose it was taken from),
    then step() every scene step; step() returns (gait name, phi0, latch_target or None)."""

    def __init__(self, seed=0, tear_reported=TEAR_BEARING, t_docked=4.0, usbl_sigma=0.05, usbl_period=1.0,
                 beam_sign=e3.BEAM_SIGN, t_cap=300.0):
        self.seed, self.tear_reported, self.t_docked, self.t_cap = seed, tear_reported, t_docked, t_cap
        self.rng = np.random.default_rng(1000 + seed)
        self.usbl_sigma, self.usbl_period = usbl_sigma, usbl_period
        self.beam_sign = beam_sign
        self.b = e3.beam_bearings(e3.SON_BEAMS, sign=beam_sign)
        self.phase, self.t_phase, self.events = "DOCKED", 0.0, [(0.0, "DOCKED")]
        self.gait, self.phi0_cmd = "stop", 0.0
        self.follow = WallFollower(STANDOFF_FOLLOW, side=1)
        self.det = TearDetector()
        self.est = e3.WallEst()
        self.images = 0
        self.fix, self.t_fix = None, -1e9
        self.latched, self.t_latch, self.done = False, float("nan"), False
        self.uturn_yaw0 = 0.0
        self.on_wall, self.station = False, False
        self.side_check = []                              # (geometric side, wall_side_of) per image
        self.dock_err = None
        self.los = dict(s=float("nan"), e=float("nan"), z=0.0)
        self.zi = {}

    # -- helpers
    def goto(self, phase, t):
        self.phase, self.t_phase = phase, t
        self.events.append((t, phase))

    def nav_fix(self, st):
        """USBL-like fix of the head: true pose + seeded noise, 1 Hz, dead-reckoned in between."""
        if st.t - self.t_fix >= self.usbl_period - 1e-9:
            self.fix = st.head_pos + self.rng.normal(0.0, self.usbl_sigma, 3) * [1.0, 0.0, 1.0]
            self.t_fix = st.t
        return self.fix + GAIT_SPEED.get(self.gait, 0.0) * yaw_dir(st.yaw_mean) * (st.t - self.t_fix)

    def line_phi0(self, p, yaw_mean, origin, a, delta, dt, key, ki=0.15, z_max=0.5, cap=D2R(20.0)):
        """ILOS onto the line through origin along a: psi_ref = yaw(delta a - (e + z) n)."""
        n = np.cross(a, [0.0, 1.0, 0.0])
        e = float((p - origin) @ n)
        s = float((p - origin) @ a)
        z = float(np.clip(self.zi.get(key, 0.0) + ki * e * dt, -z_max, z_max))
        self.zi[key] = z
        self.los = dict(s=s, e=e, z=z)
        psi_ref = yaw_of(delta * a - (e + z) * n)
        return float(np.clip(0.7 * wrap(psi_ref - yaw_mean), -cap, cap))

    # -- sonar
    def on_sonar(self, intensity, son_pos, son_yaw, st, dt_img):
        self.images += 1
        r = wall_ranges(intensity)
        side = self.follow.side
        self.est = e3.wall_estimate(r, side, self.b, fit_span=1.0)
        self.side_check.append((side, e3.wall_side_of(r, self.b)))
        self.follow.observe(self.est, son_yaw, dt_img)
        if self.phase in ("FOLLOW_WALL", "ACQUIRE") and not self.det.fired:
            if self.det.update(st.t, intensity, r, self.b, side, np.asarray(son_pos, float), son_yaw):
                self.goto("TEAR", st.t)
        elif self.phase in ("INSPECT", "INSPECT_2", "TEAR"):
            self.det.candidates(intensity, r, self.b, side, np.asarray(son_pos, float), son_yaw)

    def near_dock_line(self, st):
        """3.2-1.2 m short of the cradle centre along its axis, within 1.2 m of the axis line."""
        p = self.nav_fix(st)
        s = float((p - DOCK_C) @ DOCK_A)
        e = float((p - DOCK_C) @ DOCK_N)
        return -3.2 <= s < -1.2 and abs(e) < 1.2 and abs(wrap(bearing_of(p) - DOCK_BEARING)) < D2R(40.0)

    def station_gait(self):
        """Eel-like (the steady head the sonar wants) once on station; the lateral gait to get
        there: the current pushes the robot inboard at the tear faster than eel-like can close."""
        e = self.follow.e
        if not np.isfinite(e) or e > 0.45:
            self.station = False
        elif e < 0.25:
            self.station = True
        return "inspect" if self.station else "cruise"

    # -- control
    def step(self, st, dt):
        t = st.t
        tp_ = t - self.t_phase
        latch = None
        ph = self.phase
        if ph == "DOCKED":
            self.gait, self.phi0_cmd = "stop", 0.0
            if not self.latched and t >= self.t_docked:
                self.goto("UNDOCK", t)
        elif ph == "UNDOCK":
            self.gait = "undock"
            self.phi0_cmd = self.line_phi0(self.nav_fix(st), st.yaw_mean, DOCK_C, DOCK_U, 1.0, dt, "undock",
                                           cap=D2R(15.0))
            tail_s = float((st.links[0] - DOCK_C) @ DOCK_U)
            if tail_s > 0.5 * DOCK_LEN + 0.1:
                self.goto("ACQUIRE", t)
        elif ph == "ACQUIRE":
            self.gait = "cruise"
            self.follow.standoff, self.follow.side = STANDOFF_FOLLOW, 1
            self.phi0_cmd = self.follow.phi0(st.yaw_mean)
            if self.follow.misses == 0 and np.isfinite(self.follow.meas) and tp_ > 1.0:
                self.goto("FOLLOW_WALL", t)
        elif ph == "FOLLOW_WALL":
            self.gait = "cruise"
            self.phi0_cmd = self.follow.phi0(st.yaw_mean)
            b_head = bearing_of(self.nav_fix(st))
            # the fallback: the head is about to leave the reported sector with no sonar fire
            if not self.det.fired and wrap(b_head - self.tear_reported) < D2R(8.0):
                self.det.fire_sector(t, self.tear_reported)
                self.goto("TEAR", t)
        elif ph == "TEAR":
            self.gait = "inspect"
            self.follow.standoff = STANDOFF_INSPECT
            self.phi0_cmd = self.follow.phi0(st.yaw_mean)
            if tp_ >= 1.0:
                self.goto("INSPECT", t)
        elif ph == "INSPECT":
            self.gait = self.station_gait()
            self.follow.standoff = STANDOFF_INSPECT
            self.phi0_cmd = self.follow.phi0(st.yaw_mean)
            if wrap(bearing_of(self.nav_fix(st)) - self.det.bearing) < -1.5 / 6.0:   # 1.5 m past the tear
                self.uturn_yaw0 = st.yaw_mean
                self.goto("UTURN", t)
        elif ph == "UTURN":
            self.gait = "cruise"
            self.phi0_cmd = self.follow.side * D2R(20.0)   # inboard: away from the wall
            if abs(wrap(st.yaw_mean - self.uturn_yaw0)) > D2R(150.0) or tp_ > 40.0:
                self.follow = WallFollower(STANDOFF_INSPECT, side=-1)
                self.goto("INSPECT_2", t)
        elif ph == "INSPECT_2":
            self.gait = self.station_gait()
            self.phi0_cmd = self.follow.phi0(st.yaw_mean)
            if np.isfinite(self.follow.e) and abs(self.follow.e) < 0.4:
                self.on_wall = True
            if self.near_dock_line(st):                     # the tailwind can carry it home during the pass
                self.goto("APPROACH", t)
            elif self.on_wall and wrap(bearing_of(self.nav_fix(st)) - self.det.bearing) > 1.5 / 6.0:
                self.follow.standoff = STANDOFF_FOLLOW
                self.goto("RETURN", t)
        elif ph == "RETURN":
            self.gait = "cruise"
            self.follow.standoff = STANDOFF_FOLLOW
            self.phi0_cmd = self.follow.phi0(st.yaw_mean)
            if self.near_dock_line(st):
                self.goto("APPROACH", t)
        elif ph in ("APPROACH", "DOCK"):
            p = self.nav_fix(st)
            self.phi0_cmd = self.line_phi0(p, st.yaw_mean, DOCK_C, DOCK_A, 0.7, dt, "dock")
            s = self.los["s"]
            self.gait = "inspect" if (ph == "APPROACH" and s < -2.2) else "dock"
            if ph == "APPROACH" and s >= -0.5 * DOCK_LEN - 0.3:
                self.goto("DOCK", t)
            if ph == "DOCK":
                d = float(np.linalg.norm((st.nose - LATCH_P)[[0, 2]]))
                herr = abs(wrap(st.yaw_mean - yaw_of(DOCK_A)))
                if d < 0.15 and herr < D2R(15.0):
                    self.latched, self.t_latch = True, t
                    self.dock_err = (d, R2D(herr))
                    self.goto("LATCHED", t)
                elif float((st.nose - LATCH_P) @ DOCK_A) > 0.3:  # overshot the latch: stop, report, no retry
                    self.dock_err = (d, R2D(herr))
                    self.goto("ABORT", t)
        elif ph == "ABORT":
            self.gait, self.phi0_cmd = "stop", 0.0
            if tp_ > 5.0:
                self.done = True
        elif ph == "LATCHED":
            self.gait, self.phi0_cmd = "stop", 0.0
            latch = LATCH_P
            if tp_ >= 6.0:
                self.done = True
                self.goto("DOCKED", t)
        if self.phase == "DOCKED" and self.latched:
            latch = LATCH_P
        return self.gait, self.phi0_cmd, latch


# ---- the robot side: snake_model.Snake <-> the mission ------------------------------------------------
class SnakeAdapter:
    """Applies the mission's commands to a snake_model.Snake and reads its state (tail-first order)."""

    def __init__(self, snake, latch_k=40.0, latch_c=20.0, latch_cap=3.0):
        self.s = snake
        self.gait = None
        self.latch_k, self.latch_c, self.latch_cap = latch_k, latch_c, latch_cap
        self.latch_impulse, self.latch_pull0, self.latch_force = 0.0, None, np.zeros(3)
        self.phi0 = 0.0

    def state(self, t):
        s = self.s
        pos, q = s.link_poses()
        hp, hq, fwd = s.head_pose()
        yaws = np.unwrap(s.link_yaws())
        v, _ = s.velocities(pos, q)
        m = s.mass
        return RobotState(t=t, links=pos, nose=hp + 0.5 * LINK_L * fwd, head_pos=hp, head_yaw=float(yaws[-1]),
                          yaw_mean=float(yaws.mean()), dpsi_head=dpsi_from_joints(s.joint_angles()),
                          com=(m[:, None] * pos).sum(0) / m.sum(),
                          speed=float(np.linalg.norm(((m[:, None] * v).sum(0) / m.sum())[[0, 2]])))

    def apply(self, gait, phi0, latch, st, dt):
        s = self.s
        if gait != self.gait:
            pat, a, w, d = GAITS[gait]
            if gait == "stop" and self.gait is None:
                pass                                      # never started: the drives hold zero
            else:
                s.set_gait(pat, D2R(a), D2R(w), D2R(d), phi0=phi0)
            self.gait = gait
        elif self.gait != "stop":
            s.set_phi0(phi0)
        self.phi0 = phi0
        if latch is not None:
            if self.latch_pull0 is None:
                self.latch_pull0 = float(np.linalg.norm((latch - st.nose)[[0, 2]]))
            vh = s.velocities()[0][-1]
            f = self.latch_k * (np.asarray(latch) - st.nose) - self.latch_c * vh
            f[1] = 0.0
            n = float(np.linalg.norm(f))
            if n > self.latch_cap:
                f *= self.latch_cap / n
            F = np.zeros((s.n, 3))
            F[-1] = f
            s.extra_force = F
            self.latch_force = f
            self.latch_impulse += float(np.linalg.norm(f)) * dt
        else:
            s.extra_force = None
            self.latch_force = np.zeros(3)


# ---- the synthetic world for the CPU harness -------------------------------------------------------------
def current_at(t, seed=0):
    """warp_netpen.current_at, rotated by snake_netpen's --seed jitter."""
    sp = 0.15 * (1.0 + 0.35 * math.sin(0.21 * t))
    ang = 0.35 + 0.25 * math.sin(0.09 * t)
    if seed:
        ang += float(np.random.default_rng(seed).normal(0.0, 0.15))
    return np.array([sp * math.cos(ang), 0.0, sp * math.sin(ang)])


class SynthWall:
    """A pen wall bowed inward on the upstream side (Gaussian in bearing), with a round hole."""

    def __init__(self, bow=0.8, th_up=D2R(-160.0), width=0.6, tear_th=TEAR_BEARING, hole=True):
        self.bow, self.th_up, self.width, self.tear_th, self.hole = bow, th_up, width, tear_th, hole

    def radius(self, th):
        d = (np.asarray(th) - self.th_up + np.pi) % (2 * np.pi) - np.pi
        return PEN_R - self.bow * np.exp(-(d / self.width) ** 2)

    def dist(self, p):
        """Radial distance of points (k,3) to the wall (m, + inside)."""
        p = np.atleast_2d(p)
        return self.radius(np.arctan2(p[:, 2], p[:, 0])) - np.hypot(p[:, 0], p[:, 2])


class Fish:
    def __init__(self, rng, n=40):
        self.rng = rng
        th = rng.uniform(-np.pi, np.pi, n)
        r = rng.uniform(2.5, 6.0, n)
        self.p = np.stack([r * np.cos(th), r * np.sin(th)], 1)
        self.v = rng.normal(0, 0.3, (n, 2))

    def step(self, dt):
        self.v += self.rng.normal(0, 0.3, self.v.shape) * dt
        self.p += self.v * dt
        r = np.hypot(self.p[:, 0], self.p[:, 1])
        out = r > 6.2
        self.p[out] *= (6.2 / r[out])[:, None]
        self.v[out] *= -1


def synth_sonar(pos, yaw, wall, fish=None, rng=None, y=None):
    """(256 beams, 512 bins) intensity of the bowed wall with its hole and some fish, from the
    sonar at pos looking along yaw; beam order = +bearing to starboard (BEAM_SIGN = 1)."""
    B, NB = e3.SON_BEAMS, e3.SON_BINS
    b = e3.beam_bearings(B, sign=1)
    yb = yaw - b
    u = np.stack([np.cos(yb), -np.sin(yb)], 1)            # (x, z)
    p = np.array([pos[0], pos[2]])
    R = np.full(B, PEN_R)
    for _ in range(4):                                    # the ray-circle hit, R at the hit's bearing
        pu = u @ p
        t = -pu + np.sqrt(np.maximum(pu ** 2 + R ** 2 - p @ p, 0.0))
        h = p + t[:, None] * u
        R = wall.radius(np.arctan2(h[:, 1], h[:, 0]))
    th = np.arctan2(h[:, 1], h[:, 0])
    amp = np.ones(B)
    if wall.hole:
        x = R * ((th - wall.tear_th + np.pi) % (2 * np.pi) - np.pi)
        ins = np.abs(x) < TEAR_R
        hh = np.sqrt(np.maximum(TEAR_R ** 2 - x ** 2, 0.0))
        ys = (DEPTH_Y + 0.07) if y is None else y
        lo = ys + t * math.tan(D2R(SON_TILT - SON_EL))
        hi = ys + t * math.tan(D2R(SON_TILT + SON_EL))
        ov = np.clip(np.minimum(hi, DEPTH_Y + hh) - np.maximum(lo, DEPTH_Y - hh), 0.0, None) / (hi - lo)
        amp = np.where(ins, 1.0 - ov, 1.0)
    img = np.zeros((B, NB), np.float32)
    k = np.clip((t / e3.SON_RANGE * NB).astype(int), 0, NB - 1)
    ok = (t > e3.SON_MIN_RANGE) & (amp > 0.02)
    sp = 1.0 if rng is None else rng.uniform(0.7, 1.3, B)
    img[np.arange(B)[ok], k[ok]] = (amp * sp)[ok]
    if fish is not None:
        d = fish.p[None, :, :] - p[None, None, :]                  # (1, F, 2)
        tf = (d * u[:, None, :]).sum(-1)                           # (B, F)
        perp2 = (d ** 2).sum(-1) - tf ** 2
        hit = (perp2 < 0.15 ** 2) & (tf > e3.SON_MIN_RANGE) & (tf < t[:, None])
        bb, ff = np.nonzero(hit)
        kf = np.clip((tf[bb, ff] / e3.SON_RANGE * NB).astype(int), 0, NB - 1)
        img[bb, kf] = np.maximum(img[bb, kf], 0.35)
    return img


class Unicycle:
    """The snake's mean motion as a unicycle with the measured per-gait speed and turn rate, plus
    the gait's head-yaw swing and the current's drift; links on a straight line behind the COM."""

    def __init__(self, seed=0):
        self.seed = seed
        self.com = DOCK_C.copy()
        self.yaw = HEADING_OUT
        self.gait, self.phi0, self.phase_g, self.amp = "stop", 0.0, 0.0, 0.0
        self.v = 0.0
        self.latch_pull0, self.latch_impulse = None, 0.0

    def state(self, t):
        f = yaw_dir(self.yaw)
        ks = np.arange(LINK_N) - (LINK_N - 1) / 2.0
        links = self.com + np.outer(ks * LINK_L, f)
        dpsi = self.amp * math.sqrt(2.0) * math.sin(self.phase_g)
        return RobotState(t=t, links=links, nose=self.com + 0.5 * LINK_N * LINK_L * f, head_pos=links[-1],
                          head_yaw=self.yaw + dpsi, yaw_mean=self.yaw, dpsi_head=dpsi, com=self.com.copy(),
                          speed=self.v)

    def apply(self, gait, phi0, latch, st, dt):
        self.gait = gait
        self.phi0 += (phi0 - self.phi0) * (1.0 - math.exp(-dt / 0.5))
        v0 = GAIT_SPEED[gait]
        self.v += (v0 * (1.0 - 0.35 * (self.phi0 / D2R(20.0)) ** 2) - self.v) * (1.0 - math.exp(-dt / 2.0))
        self.amp += (GAIT_HEADYAW[gait] - self.amp) * (1.0 - math.exp(-dt / 2.0))
        self.phase_g += GAIT_OMEGA[gait] * dt
        self.yaw += GAIT_TURN[gait] * self.phi0 * (self.v / max(v0, 1e-6) if v0 > 0 else 0.0) * dt
        cur = current_at(st.t, self.seed)
        if latch is not None:
            if self.latch_pull0 is None:
                self.latch_pull0 = float(np.linalg.norm((latch - st.nose)[[0, 2]]))
            f = latch - st.nose
            f[1] = 0.0
            self.com += f * min(1.0, dt / 1.0)
            self.yaw += wrap(yaw_of(DOCK_A) - self.yaw) * min(1.0, dt / 1.0)
            return
        self.com = self.com + (self.v * yaw_dir(self.yaw) + cur) * dt


class PhysxRobot:
    """The real snake_model.Snake in its own PhysX world, on the synthetic water."""

    def __init__(self, seed=0):
        import snake_model as sm
        self.sm = sm
        self.seed = seed
        self.world = sm.make_world()
        self.snake = sm.Snake(self.world, sm.SnakeParams(), origin=tuple(DOCK_C), heading=HEADING_OUT)
        self.ad = SnakeAdapter(self.snake)

    def state(self, t):
        return self.ad.state(t)

    def apply(self, gait, phi0, latch, st, dt):
        self.ad.apply(gait, phi0, latch, st, dt)
        self.snake.set_current(current_at(st.t, self.seed))
        self.world.step(dt)

    @property
    def latch_pull0(self):
        return self.ad.latch_pull0

    @property
    def latch_impulse(self):
        return self.ad.latch_impulse


def run_synthetic(robot, seed=0, t_cap=300.0, dt=1.0 / 60.0, every=3, hole=True, verbose=True):
    """The mission on the synthetic pen: the robot (Unicycle or PhysxRobot) + synth_sonar."""
    rng = np.random.default_rng(seed)
    wall = SynthWall(hole=hole)
    fish = Fish(rng)
    ms = Mission(seed=seed)
    rows = []
    min_d = float("inf")
    n = int(t_cap / dt)
    for f in range(n):
        t = f * dt
        st = robot.state(t)
        if f % every == 0:
            fish.step(every * dt)
            yaw_s = st.head_yaw
            son = st.nose - 0.05 * yaw_dir(yaw_s)
            ms.on_sonar(synth_sonar(son, yaw_s, wall, fish, rng), son, yaw_s, st, every * dt)
        gait, phi0, latch = ms.step(st, dt)
        robot.apply(gait, phi0, latch, st, dt)
        d = float(wall.dist(st.links).min()) - LINK_R
        if ms.phase not in ("DOCKED",) or ms.latched:
            min_d = min(min_d, d)
        true_d = float(wall.dist(st.head_pos[None])[0])
        rows.append((t, PHASES.index(ms.phase), *st.com[[0, 2]], st.yaw_mean, phi0, ms.follow.meas, true_d, d))
        if ms.done:
            break
    L = np.array(rows)
    res = dict(seed=seed, done=ms.done, t_end=float(L[-1, 0]), min_d=min_d, tear_mode=ms.det.mode,
               tear_err_deg=R2D(wrap(ms.det.bearing - TEAR_BEARING)) if ms.det.fired else float("nan"),
               tear_t=ms.det.t_fire, dock_err=ms.dock_err, events=[(round(a, 1), b) for a, b in ms.events],
               latch_pull0=robot.latch_pull0, latch_impulse=robot.latch_impulse)
    fw = L[:, 1] == PHASES.index("FOLLOW_WALL")
    if fw.any():
        res["follow_meas_minus_true_rms"] = float(np.sqrt(np.nanmean((L[fw, 6] - L[fw, 7]) ** 2)))
        res["follow_true_mean"] = float(np.nanmean(L[fw, 7]))
    if verbose:
        print(f"  seed {seed}: done {res['done']} t {res['t_end']:.1f} s, min link-to-net {min_d:.2f} m, tear "
              f"{res['tear_mode']} err {res['tear_err_deg']:+.1f} deg at {res['tear_t']:.1f} s, dock err {res['dock_err']}")
        print("   events " + ", ".join(f"{b}@{a}" for a, b in res["events"]))
    return res, L


PHASE_COLORS = {"DOCKED": "#555555", "UNDOCK": "#8c564b", "ACQUIRE": "#bcbd22", "FOLLOW_WALL": "#1f77b4",
                "TEAR": "#d62728", "INSPECT": "#ff7f0e", "UTURN": "#9467bd", "INSPECT_2": "#e377c2",
                "RETURN": "#17becf", "APPROACH": "#2ca02c", "DOCK": "#006400", "LATCHED": "#000000", "ABORT": "#ff0000"}


def summarize(tel, mission, adapter, mass):
    """The per-seed numbers from the telemetry (see snake_netpen.run_mission for the columns)."""
    L, cols = tel["log"], list(tel["cols"])
    c = {k: i for i, k in enumerate(cols)}
    t = L[:, c["t"]]
    ph = L[:, c["phase"]].astype(int)
    gait = L[:, c["gait"]].astype(int)
    gaits = list(tel["gaits"])
    names = list(tel["phases"])
    com = L[:, [c["com_x"], c["com_z"]]]
    seg = np.linalg.norm(np.diff(com, axis=0), axis=1)
    moving = ph[1:] != names.index("DOCKED")
    out = dict(completed=bool(mission.done and mission.latched), mission_time_s=round(float(t[-1]), 1),
               path_length_m=round(float(seg[moving].sum()), 2),
               events=[[round(a, 2), b] for a, b in mission.events])
    per = {}
    for g in ("cruise", "inspect", "dock", "undock"):
        k = gaits.index(g)
        m = gait[1:] == k
        if m.sum() < 60:
            continue
        dt_ = np.diff(t)[m]
        dist = float(seg[m].sum())
        e = float(np.sum(L[1:, c["p_abs"]][m] * dt_))
        hy = L[1:, c["dpsi_head"]][m]
        per[g] = dict(time_s=round(float(dt_.sum()), 1), dist_m=round(dist, 2),
                      mean_speed_ms=round(dist / max(float(dt_.sum()), 1e-9), 3),
                      energy_J=round(e, 1), cot_J_per_kg_m=round(e / (mass * max(dist, 1e-9)), 3),
                      head_yaw_rms_deg=round(R2D(float(np.sqrt(np.mean(hy ** 2)))), 1))
    out["per_gait"] = per
    fw = ph == names.index("FOLLOW_WALL")
    if fw.any():
        err = L[fw, c["meas"]] - L[fw, c["d_head_true"]]
        err = err[np.isfinite(err)]
        out["follow_standoff_err_rms_m"] = round(float(np.sqrt(np.mean(err ** 2))), 3) if len(err) else None
        out["follow_true_mean_m"] = round(float(np.nanmean(L[fw, c["d_head_true"]])), 3)
        out["follow_meas_mean_m"] = round(float(np.nanmean(L[fw, c["meas"]])), 3)
    ins = (ph == names.index("INSPECT")) | (ph == names.index("INSPECT_2"))
    if ins.any():
        out["inspect_true_mean_m"] = round(float(np.nanmean(L[ins, c["d_head_true"]])), 3)
    und = ph != names.index("DOCKED")
    out["min_link_to_net_m"] = round(float(np.nanmin(L[und, c["d_min_links"]])), 3) if und.any() else None
    det = mission.det
    out["tear"] = dict(detected=bool(det.fired), mode=det.mode, t_s=round(float(det.t_fire), 1) if det.fired else None,
                       bearing_deg=round(R2D(det.bearing), 2) if det.fired else None,
                       bearing_err_deg=round(R2D(wrap(det.bearing - float(tel["tear_true"]))), 2) if det.fired else None,
                       candidates=len(det.all_cands))
    out["dock_error"] = None if mission.dock_err is None else dict(nose_m=round(mission.dock_err[0], 3),
                                                                   heading_deg=round(mission.dock_err[1], 1))
    out["latch"] = dict(pull_start_m=None if adapter.latch_pull0 is None else round(adapter.latch_pull0, 3),
                        impulse_Ns=round(adapter.latch_impulse, 3), cap_N=adapter.latch_cap)
    out["peak_joint_torque_Nm"] = round(float(L[-1, c["peak_torque"]]), 2)
    out["energy_abs_J"] = round(float(L[-1, c["e_abs"]]), 1)
    out["sonar_images"] = int(L[-1, c["images"]])
    sc = mission.side_check
    out["wall_side_agreement"] = round(float(np.mean([a == b for a, b in sc if b != 0])), 3) if sc else None
    return out


def plot_top(tel, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    L, cols = tel["log"], list(tel["cols"])
    c = {k: i for i, k in enumerate(cols)}
    names = list(tel["phases"])
    ph = L[:, c["phase"]].astype(int)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=110)
    th = np.linspace(-np.pi, np.pi, 361)
    ax.plot(PEN_R * np.cos(th), PEN_R * np.sin(th), color="#999999", lw=1.0, label="pen r 7 (rest)")
    ring = tel.get("net_y3")
    if ring is not None and len(ring):
        ax.scatter(ring[:, 0], ring[:, 1], s=1, color="#444444", label="net at y -3 (end)")
    dc, du = np.asarray(tel["dock_c"]), np.asarray(tel["dock_u"])
    a, b = dc - du * DOCK_LEN / 2, dc + du * DOCK_LEN / 2
    ax.plot([a[0], b[0]], [a[2], b[2]], color="#e0641c", lw=5, alpha=0.6, label="cradle")
    tt = float(tel["tear_true"])
    ax.plot(PEN_R * math.cos(tt), PEN_R * math.sin(tt), "rx", ms=12, mew=3, label="tear (true)")
    for k, nm in enumerate(names):
        m = ph == k
        if not m.any():
            continue
        col = PHASE_COLORS.get(nm, "k")
        ax.plot(np.where(m, L[:, c["head_x"]], np.nan), np.where(m, L[:, c["head_z"]], np.nan), color=col, lw=0.7, alpha=0.6)
        ax.plot(np.where(m, L[:, c["com_x"]], np.nan), np.where(m, L[:, c["com_z"]], np.nan), color=col, lw=2.2, label=nm)
    tb = L[:, c["tear_bearing"]]
    if np.isfinite(tb).any():
        be = float(tb[np.isfinite(tb)][-1])
        ax.plot([0, PEN_R * math.cos(be)], [0, PEN_R * math.sin(be)], "r--", lw=0.8, label="tear bearing (est)")
    ax.set_aspect("equal")
    ax.set_xlim(-8, 8)
    ax.set_ylim(8, -8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)  (top view, looking down)")
    ax.set_title(f"snake mission seed {int(tel['seed'])}: COM (thick) and head (thin) by phase")
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def detector_unit():
    """Hole abeam at 1.0 / 1.5 m standoffs fires; the intact wall never does."""
    ok = True
    rng = np.random.default_rng(3)
    for hole in (True, False):
        for so in (1.0, 1.5):
            wall = SynthWall(hole=hole)
            det = TearDetector()
            fish = Fish(rng)
            fired = False
            th = TEAR_BEARING + D2R(25.0)
            b_all = e3.beam_bearings(sign=1)
            while th > TEAR_BEARING - D2R(20.0):
                rr = float(wall.radius(th)) - so
                pos = polar(th, rr)
                yaw = yaw_of(np.array([math.sin(th), 0.0, -math.cos(th)]))   # decreasing bearing: wall to starboard
                fish.step(0.05)
                img = synth_sonar(pos, yaw, wall, fish, rng)
                r = wall_ranges(img)
                fired |= det.update(0.0, img, r, b_all, 1, pos, yaw)
                th -= 0.25 * 0.05 / rr
            good = fired == hole
            err = R2D(wrap(det.bearing - TEAR_BEARING)) if det.fired else float("nan")
            if hole:
                good = good and abs(err) < 5.0
            ok &= good
            print(f"  {'ok  ' if good else 'FAIL'} detector hole={hole} standoff {so}: fired {fired} bearing err {err:+.1f} deg")
    return ok


def selftest(physx=False):
    ok = detector_unit()
    print("unicycle stand-in on the synthetic pen:")
    for seed in (0, 1, 2):
        res, _ = run_synthetic(Unicycle(seed), seed)
        good = res["done"] and res["min_d"] >= 0.6 and res["tear_mode"] == "sonar" and abs(res["tear_err_deg"]) < 5
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} unicycle seed {seed}")
    if physx:
        print("PhysX snake on the synthetic pen:")
        t0 = time.perf_counter()
        res, L = run_synthetic(PhysxRobot(0), 0)
        good = res["done"] and res["min_d"] >= 0.6
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} physx seed 0 ({time.perf_counter() - t0:.0f} s wall)")
    print("selftest: " + ("all ok" if ok else "FAILURES"))
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--physx", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.physx) else 1)
    print(__doc__)
