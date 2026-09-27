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
In the mission the along-wall DIRECTION is the pen's tangent at the navigation fix's bearing (the
sonar line fit swings in the concave corners the current squashes into the pen and sent the robot
round in loops); the sonar sets the cross-track error e only:

    psi_ref  = crab(Delta * tangent + clip(e + z, +-Delta) * outward)

where crab() turns the wanted ground direction into a heading through water that cancels the
current across it (the station's current meter; |cross| / gait speed <= 0.8): at the tear the
current sets the robot onto the pen's centre at ~0.15 m/s, which the integral alone never caught
in a 10 s pass. The guidance steers on a 0.6 s low-pass of the mean link yaw (it wobbles with the
lateral wave) and phi0 is rate limited to 40 deg/s (a step reaches the joints as a phi_ddot*
spike through the paper's feed-forward (23): 50 N m peaks without it, <= 10 N m with it).
NAVIGATION: a USBL fix of the head at 1 Hz (seeded 5 cm noise), DVL dead reckoning in between (the
error is the fix's noise, held for the second); the body centre adds the chain shape from the
joint encoders.

TEAR DETECTOR. Per image, across the whole fan (the gait swings the head so far, +-40-65 deg,
that the wall and the hole sweep through the port half too): the wall's expected range per beam
is a 41-beam nan-median of the last echoes, carried linearly across a longer no-echo run from the
wall on its two sides; a beam is a gap beam where the echo is missing or > expected + 1.0 m, or
where the wall echo is < 25 % of the local wall level (the vertical fan straddles a 1.3 m hole
except near abeam, so a partly covered beam still echoes, only weaker). A contiguous run of gap
beams with the wall on both sides, at an expected range < 3 m, whose chord on the wall (its two
edges, each at its own expected range: the wall is oblique to the fan) is 0.4-1.8 m, is a
candidate; the chord's midpoint goes to the world with the head's pose at the image. It fires
when >= 3 of the last 5 images hold a candidate within 6 deg of pen bearing of each other. If the
head passes the reported sector (+-20 deg around the reported tear bearing) without a fire, it
fires in 'sector' mode on the best sonar candidate inside the sector ('sector+sonar') or on the
reported bearing; the close passes keep collecting candidates and upgrade a bare 'sector' fire to
'sector+sonar' once >= 3 agree. On the scene's rendered sonar (seed 0, 600 images ACQUIRE ..
INSPECT_2) the hole is a no-echo run of 8-30 beams at 1.8-2.1 m, 30-48 at 1.3-1.7 m, none beyond
~2.2 m: 111 candidates, all 0.6-1.3 deg from the true tear bearing, none elsewhere (wall, fish,
cradle), a 'sonar' fire at 30.1 s (-1.1 deg). A starboard-only mask (b > 12 deg) and the bare 41-beam median
gave 7: the hole sat in the port half or straddled the mask's edge (no wall beside it), and a hole
wider than the window left the median NaN in its middle, splitting the run.

STATE MACHINE. DOCKED (latched: the latch line holds the head at the latch point, spring 40 N/m,
damper 20 N s/m, cap 10 N; the cradle bore holds the links inside it, see CRADLE) -> UNDOCK (the
latch releases; lateral at 20 deg, into the current, on the cradle axis until the nose clears the
outer hoop, then on an exit line turned 38 deg inboard of the axis, 20 deg inboard of the pen
tangent, so the body stays off the bowed net) -> ACQUIRE (turn toward the wall until it is seen)
-> FOLLOW_WALL (lateral, 1.5 m) -> TEAR (the detector fired) -> INSPECT (1.25 m, until 1.5 m past
the tear: eel-like on station, lateral at 20 deg when off it) -> UTURN (inboard, until heading back
along the pen) -> INSPECT_2 (the same, back past the tear, on the pen-centred circle at the wall
radius the first pass measured minus 1.25 m: USBL fix + circle ILOS) -> RETURN -> APPROACH.

DOCKING, the way a real dock is flown. The cradle is open at both ends; the robot enters at the
DOWNSTREAM end so the final approach heads INTO the current (the station's current meter picks the
end): the headway keeps the steering authority and a miss drifts back out of the funnel instead of
through it. APPROACH: an ILOS pass line 1.2 m inboard of the axis, with the current, until the
body centre is 3.0 m beyond the cradle centre -> TURN_IN (turn outboard, 25 deg phi0, onto the
axis) -> FINAL (ILOS on the axis from the USBL fix and the joint encoders, lateral cruise, lateral at
20 deg for the last metre so the body fits the funnel) -> at the mouth plane the FUNNEL decides:
nose within 0.30 m of the axis (the 0.35 m mouth hoop less the body radius) and mean heading
within 25 deg = CAPTURE (the funnel cone narrows to the 0.13 m bore), else RETREAT (gait off, the current carries it back
1.4 m) and retry, up to 3 attempts, else ABORT. CAPTURE: the gait fades out and the latch line hauls
the nose along the axis to the latch point at 0.15 m/s (target moving along the axis, the same
capped spring) -> DOCKED (latched) and the mission ends 5 s later.

CRADLE. While DOCKED and in CAPTURE the bore's contact acts on every link inside the cradle: a
one-sided lateral spring-damper beyond 0.04 m of free play (150 N/m, 10 N s/m, cap 6 N per link),
the rails and hoops pushing back. It is off while swimming out (lateral undulation needs the lateral
motion). These are the only non-paper forces: no thrusters, the latch line, the bore contact.

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
# The cradle hangs 4.7 m out (its axis tangent there): the current flattens this side of the pen to
# r ~6.2-6.4 at the cradle's bearings (the cloth at y -3, measured), so at 5.5 m the downstream
# staging turn came within 0.3 m of the net; at 4.7 m everything within 3 m of the cradle along
# its axis is >= 0.8 m off it.
R_DOCK = 4.7
DOCK_LEN = 2.0
# The funnel mouth takes the gait's swept width: at lateral 20 deg the head swings ~+-0.15 m about
# the body's line (0.105 m body), so the mouth is 0.70 m across and narrows to the 0.26 m bore.
MOUTH_R, BORE_R = 0.35, 0.13                              # funnel mouth hoop radius, bore radius (m)
LINK_N, LINK_L, LINK_R = 9, 0.18, 0.0525
SON_TILT, SON_EL = -6.0, 14.0
STANDOFF_FOLLOW, STANDOFF_INSPECT = 1.5, 1.25


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
DOCK_IN = -np.array([math.cos(DOCK_BEARING), 0.0, math.sin(DOCK_BEARING)])  # inboard (toward the pen centre)
# The cradle axis is yawed 18 deg off the tangent: its downstream (+x) end points inboard, so (a) the
# scene's current (heading 0.35 +- 0.25 rad) runs within ~10 deg of the axis and the robot docks
# nearly head-on into it, and (b) the staging turn beyond that end stays >= 0.9 m off the net.
DOCK_TILT = D2R(18.0)
_U_TAN = np.array([math.sin(DOCK_BEARING), 0.0, -math.cos(DOCK_BEARING)])
DOCK_U = math.cos(DOCK_TILT) * _U_TAN - math.sin(DOCK_TILT) * DOCK_IN      # cradle axis, toward the tear
DOCK_A = -DOCK_U
DOCK_N = np.cross(DOCK_A, [0.0, 1.0, 0.0])                # the axis's horizontal normal, inboard
HALF_BODY = 0.5 * LINK_N * LINK_L                         # 0.81 m: nose ahead of the body centre


def latch_point(a_in):
    """The docked nose for a body that entered along a_in: the body centred in the cradle."""
    return DOCK_C + HALF_BODY * np.asarray(a_in, float)


LATCH_P = latch_point(DOCK_U)                             # the start pose's nose (the current sets the end's)
HEADING_OUT = yaw_of(DOCK_U)
EXIT_BEND = D2R(38.0)                                     # the exit line past the outer hoop, turned inboard (20 deg off the tangent)
EXIT_U = math.cos(EXIT_BEND) * DOCK_U + math.sin(EXIT_BEND) * DOCK_IN
EXIT_O = DOCK_C + 0.5 * DOCK_LEN * DOCK_U
# docking: pass inboard of the cradle, turn outboard onto its axis downstream, swim in against the current
PASS_E = 1.2                                              # pass line: this far inboard of the axis
S_TURN = -3.0                                             # turn when the COM is this far along the entry axis
CAPTURE_E, CAPTURE_PSI = MOUTH_R - LINK_R, D2R(25.0)       # funnel capture window at the mouth plane (0.30 m)
V_HAUL = 0.15                                             # m/s: the latch line hauls the nose to the latch
LATCH_K, LATCH_C, LATCH_CAP = 40.0, 20.0, 10.0            # head latch spring N/m, damper N s/m, cap N
GUIDE_FREE, GUIDE_K, GUIDE_C, GUIDE_CAP = 0.04, 150.0, 10.0, 6.0   # cradle bore contact per link
MAX_ATTEMPTS = 3
PHI0_RATE = D2R(40.0)                                     # rad/s: the steering offset's rate limit

# ---- gaits: measured on snake_model (complex coefficients, 20-30 s runs, still water) -------------
#   name        pattern    alpha  omega  delta   speed    head-yaw RMS   yaw rate per deg phi0
GAITS = {
    "cruise":  ("lateral", 30.0, 150.0, 30.0),       # 0.35 m/s   31 deg        ~2.2 /s
    "inspect": ("eel",     40.0, 150.0, 30.0),       # 0.22 m/s   13 deg        ~1.4 /s
    "creep":   ("lateral", 20.0, 150.0, 30.0),       # 0.26 m/s   20 deg        inspection off station
    "dock":    ("lateral", 20.0, 150.0, 30.0),       # 0.26 m/s   20 deg        the last metre into the funnel
    "undock":  ("lateral", 20.0, 150.0, 30.0),       # 0.26 m/s   20 deg        (the cradle: headway into the current)
    "stop":    ("eel",      0.0,   0.0, 30.0),
}
GAIT_SPEED = {"cruise": 0.35, "inspect": 0.22, "creep": 0.26, "dock": 0.26, "undock": 0.26, "stop": 0.0}
GAIT_TURN = {"cruise": 2.2, "inspect": 1.4, "creep": 1.5, "dock": 1.5, "undock": 1.5, "stop": 0.0}
GAIT_HEADYAW = {"cruise": D2R(31.0), "inspect": D2R(13.0), "creep": D2R(20.0), "dock": D2R(20.0),
                "undock": D2R(20.0), "stop": 0.0}
GAIT_OMEGA = {k: D2R(v[2]) for k, v in GAITS.items()}

PHASES = ("DOCKED", "UNDOCK", "ACQUIRE", "FOLLOW_WALL", "TEAR", "INSPECT", "UTURN", "INSPECT_2",
          "RETURN", "APPROACH", "TURN_IN", "FINAL", "CAPTURE", "RETREAT", "ABORT")


# ---- the robot's own state, as the mission sees it -------------------------------------------------
class RobotState:
    """What the robot knows about itself each step (IMU/compass on the head, joint encoders, and
    the true pose only through Mission's USBL fix); cur = the dock station's current meter."""
    __slots__ = ("t", "links", "nose", "head_pos", "head_yaw", "yaw_mean", "dpsi_head", "com", "speed", "cur")

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


def _fill_across(v, mask):
    """v with its NaN runs inside mask that have finite values on both sides filled linearly."""
    k = np.arange(len(v))
    idx = np.flatnonzero(mask & np.isfinite(v))
    if len(idx) < 2:
        return v
    inner = mask & ~np.isfinite(v) & (k > idx[0]) & (k < idx[-1])
    out = v.copy()
    out[inner] = np.interp(k[inner], idx, v[idx])
    return out


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
                 persist=3, window=5, agree=D2R(6.0), min_side=None):
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
        # the whole fan by default: the gait swings the head so far that the wall and the hole sweep
        # through the port half too (min_side = a starboard-only mask, radians)
        wall_side = np.ones(len(b), bool) if self.min_side is None else side * b > self.min_side
        rr = np.where(wall_side, r, np.nan)
        # carried across a no-echo run wider than the window from the wall on both sides
        ref = _fill_across(_nanmed_window(rr, self.win), wall_side)
        # the wall echo level per beam, around the expected range
        rb = np.clip(np.nan_to_num(ref / bin_m, nan=0).astype(int), 0, bins - 1)
        idx = np.clip(rb[:, None] + np.arange(-4, 5)[None, :], 0, bins - 1)
        level = a[np.arange(len(r))[:, None], idx].max(1)
        lref = _fill_across(_nanmed_window(np.where(wall_side & np.isfinite(r), level, np.nan), self.win), wall_side)
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
            # the chord on the wall between the run's edges, each at its own expected range (the
            # wall is oblique to the fan: one expected range for both undercounts it)
            p_lo = pos + ref[lo] * yaw_dir(yaw - (b[lo] - 0.5 * abs(b[1] - b[0])))
            p_hi = pos + ref[hi] * yaw_dir(yaw - (b[hi] + 0.5 * abs(b[1] - b[0])))
            w = float(np.linalg.norm((p_hi - p_lo)[[0, 2]]))
            if not (self.min_w <= w <= self.max_w):
                continue
            pc = 0.5 * (p_lo + p_hi)                      # its centre on the wall, not the mid beam
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

    def refine(self, t, intensity, r, b, side, pos, yaw, reported, sector=D2R(20.0)):
        """During the close passes: log the candidates; a sector fire takes the sonar's bearing once
        >= 3 candidates agree inside the sector (mode 'sector+sonar')."""
        for q in self.candidates(intensity, r, b, side, pos, yaw):
            self.all_cands.append((t, *q))
        if self.mode == "sector":
            inside = [q[1] for q in self.all_cands if abs(wrap(q[1] - reported)) < sector]
            if len(inside) >= 3:
                self.bearing, self.mode = float(np.median(inside)), "sector+sonar"

    def fire_sector(self, t, reported, sector=D2R(20.0)):
        inside = [q[1] for q in self.all_cands if abs(wrap(q[1] - reported)) < sector]
        if inside:
            self.bearing, self.mode = float(np.median(inside)), "sector+sonar"
        else:
            self.bearing, self.mode = reported, "sector"
        self.fired, self.t_fire = True, t


def crab_heading(d, cur, v):
    """The heading that makes good the ground direction d through water of speed v in the current
    cur (the station's current meter): cancel the current across d, |that| / v <= 0.8 (53 deg)."""
    d = np.asarray(d, float)
    d = d / max(np.linalg.norm(d), 1e-9)
    if cur is None or v <= 0.0:
        return yaw_of(d)
    c = np.asarray(cur, float).copy()
    c[1] = 0.0
    cp = c - float(c @ d) * d
    k = float(np.linalg.norm(cp)) / v
    if k > 0.8:
        cp *= 0.8 / k
        k = 0.8
    h = math.sqrt(1.0 - k * k) * d - cp / v
    return yaw_of(h)


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

    def phi0(self, yaw_mean, p_fix=None, cur=None, v=0.0):
        """With p_fix (the USBL head fix) the along-wall DIRECTION is the pen's tangent at the fix's
        bearing and the sonar only sets the cross-track error: the sonar's own line fit swings in the
        pen's concave corners (the current squashes the pen), where it sent the robot round in loops."""
        if p_fix is not None:
            th = bearing_of(p_fix)
            tan = self.side * np.array([math.sin(th), 0.0, -math.cos(th)])   # side +1: toward decreasing bearing
            out = np.array([math.cos(th), 0.0, math.sin(th)])
            ez = (self.e + self.z) if (np.isfinite(self.e) and self.misses <= 30) else 0.0
            ref = crab_heading(self.delta * tan + float(np.clip(ez, -self.delta, self.delta)) * out, cur, v)
        elif self.psi_ref is None or self.misses > 30:
            ref = yaw_mean - self.side * D2R(15.0)          # lost the wall: turn gently toward its side
        else:
            ref = self.psi_ref
        return float(np.clip(self.k_theta * wrap(ref - yaw_mean), -self.phi0_max, self.phi0_max))


class Mission:
    """The state machine. Call on_sonar() with each image (and the sonar pose it was taken from),
    then step() every scene step; step() returns (gait name, phi0, hold or None), where hold =
    dict(target=nose target point or None, cap=N, axis=unit, guide=bool) for SnakeAdapter."""

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
        self.engaged = True                               # the latch holds the head (the mission starts docked)
        self.a_in, self.latch = DOCK_U.copy(), LATCH_P.copy()
        self.docked_ok, self.returned, self.done = False, False, False
        self.uturn_yaw0 = 0.0
        self.on_wall, self.station = False, False
        self.side_check = []                              # (geometric side, wall_side_of) per image
        self.dock_err = None                              # at the end: (nose to latch m, heading err deg)
        self.capture_err = None                           # at the mouth: (nose off-axis m, heading err deg)
        self.attempts, self.attempt_log = 0, []
        self.los = dict(s=float("nan"), e=float("nan"), z=0.0)
        self.zi = {}
        self.turn_sign, self.s_cap, self.t_cap_start = 1.0, 0.0, 0.0
        self.r_wall = 5.7
        self.psi_f = None
        self.phi0_prev = 0.0

    # -- helpers
    def goto(self, phase, t):
        self.phase, self.t_phase = phase, t
        self.events.append((t, phase))

    def nav_err(self, st):
        """USBL + DVL navigation: a USBL fix of the head at 1 Hz (seeded 5 cm noise per axis), dead
        reckoned in between on the DVL's velocity over ground (drift-free over a second); the error
        is the fix's noise, held until the next fix. (The old dead reckoning on the still-water gait
        speed jumped 0.1-0.2 m at every fix into the current and aliased the head's swing into the
        steering: the body thrashed at +-20 deg phi0 in FINAL, 40 W against 4 W along the wall.)"""
        if st.t - self.t_fix >= self.usbl_period - 1e-9:
            self.fix = self.rng.normal(0.0, self.usbl_sigma, 3) * [1.0, 0.0, 1.0]
            self.t_fix = st.t
        return self.fix

    def nav_fix(self, st):
        return st.head_pos + self.nav_err(st)

    def com_fix(self, st):
        """The body centre: the head's navigation solution + the chain shape from the joint encoders."""
        return st.com + self.nav_err(st)

    def line_phi0(self, p, yaw_mean, origin, a, delta, dt, key, ki=0.15, z_max=0.5, cap=D2R(20.0)):
        """ILOS onto the line through origin along a: psi_ref = yaw(delta a - (e + z) n)."""
        n = np.cross(a, [0.0, 1.0, 0.0])
        e = float((p - origin) @ n)
        s = float((p - origin) @ a)
        z = float(np.clip(self.zi.get(key, 0.0) + ki * e * dt, -z_max, z_max))
        self.zi[key] = z
        self.los = dict(s=s, e=e, z=z)
        psi_ref = yaw_of(delta * a - float(np.clip(e + z, -delta, delta)) * n)   # intercept <= 45 deg: no overshoot
        return float(np.clip(0.7 * wrap(psi_ref - yaw_mean), -cap, cap))

    def circle_phi0(self, p, yaw_mean, r_ref, dt, delta=1.0, ki=0.1, z_max=0.5, cap=D2R(20.0), cur=None, v=0.0,
                    e_min=-np.inf):
        """ILOS on the pen-centred circle r_ref, travelling toward increasing bearing; e_min (the sonar's
        'too close' error, + = inboard wanted) overrides the circle where the wall bows in."""
        th = bearing_of(p)
        out = np.array([math.cos(th), 0.0, math.sin(th)])
        tan = np.array([-math.sin(th), 0.0, math.cos(th)])
        e = max(math.hypot(p[0], p[2]) - r_ref, e_min)
        z = float(np.clip(self.zi.get("circle", 0.0) + ki * e * dt, -z_max, z_max))
        self.zi["circle"] = z
        psi_ref = crab_heading(delta * tan - float(np.clip(e + z, -delta, delta)) * out, cur, v)
        return float(np.clip(0.7 * wrap(psi_ref - yaw_mean), -cap, cap)), e

    def axis_coords(self, p, a):
        """(along a from the cradle centre, horizontal offset from the axis) of point p."""
        d = np.asarray(p, float) - DOCK_C
        s = float(d @ a)
        lat = d - s * a
        lat[1] = 0.0
        return s, float(np.linalg.norm(lat))

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
            self.det.refine(st.t, intensity, r, self.b, side, np.asarray(son_pos, float), son_yaw, self.tear_reported)

    def station_gait(self):
        """Eel-like (the steady head the sonar wants) once on station; lateral at reduced amplitude
        (creep: more headway) to get there: the current pushes the robot inboard at the tear."""
        e = self.follow.e
        if not np.isfinite(e) or abs(e) > 0.40:
            self.station = False
        elif abs(e) < 0.20:
            self.station = True
        return "inspect" if self.station else "creep"

    def choose_entry(self, st):
        """Dock INTO the current: enter the open-ended cradle at its downstream end."""
        cur = np.zeros(3) if st.cur is None else np.asarray(st.cur, float)
        self.a_in = DOCK_U.copy() if float(cur @ DOCK_U) <= 0.0 else DOCK_A.copy()
        self.latch = latch_point(self.a_in)
        out = -DOCK_N
        self.turn_sign = 1.0 if wrap(yaw_of(out) - yaw_of(-self.a_in)) > 0 else -1.0

    # -- control
    def step(self, st, dt):
        t = st.t
        tp_ = t - self.t_phase
        # the mean link yaw wobbles with the lateral wave; the guidance steers on its 0.6 s low-pass
        if self.psi_f is None:
            self.psi_f = st.yaw_mean
        self.psi_f += (1.0 - math.exp(-dt / 0.6)) * wrap(st.yaw_mean - self.psi_f)
        yaw_g = self.psi_f
        hold = None
        ph = self.phase
        if ph == "DOCKED":
            self.gait, self.phi0_cmd = "stop", 0.0
            hold = dict(target=self.latch, cap=LATCH_CAP, axis=self.a_in, guide=True)
            if not self.returned and t >= self.t_docked:
                self.engaged = False                      # release
                self.goto("UNDOCK", t)
            elif self.returned and tp_ >= 5.0:
                self.done = True
        elif ph == "UNDOCK":
            self.gait = "undock"
            s_nose, _ = self.axis_coords(st.nose, DOCK_U)
            p = self.com_fix(st)                          # the body's line, not the swinging head
            if s_nose < 0.5 * DOCK_LEN + 0.15:            # through the cradle on its axis
                self.phi0_cmd = self.line_phi0(p, yaw_g, DOCK_C, DOCK_U, 1.0, dt, "undock", cap=D2R(15.0))
            else:                                         # past the outer hoop: the exit line, bent inboard
                self.phi0_cmd = self.line_phi0(p, yaw_g, EXIT_O, EXIT_U, 1.0, dt, "exit", cap=D2R(15.0))
            tail_s, _ = self.axis_coords(st.links[0], DOCK_U)
            if tail_s > 0.5 * DOCK_LEN + 0.2:                  # the tail is clear of the outer hoop
                self.goto("ACQUIRE", t)
        elif ph == "ACQUIRE":
            self.gait = "cruise"
            self.follow.standoff, self.follow.side = STANDOFF_FOLLOW, 1
            self.phi0_cmd = self.follow.phi0(yaw_g, self.nav_fix(st), st.cur, GAIT_SPEED.get(self.gait, 0.2))
            if self.follow.misses == 0 and np.isfinite(self.follow.meas) and tp_ > 1.0:
                self.goto("FOLLOW_WALL", t)
        elif ph == "FOLLOW_WALL":
            self.gait = "cruise"
            self.phi0_cmd = self.follow.phi0(yaw_g, self.nav_fix(st), st.cur, GAIT_SPEED.get(self.gait, 0.2))
            b_head = bearing_of(self.nav_fix(st))
            # the fallback: the head is about to leave the reported sector with no sonar fire
            if not self.det.fired and wrap(b_head - self.tear_reported) < D2R(8.0):
                self.det.fire_sector(t, self.tear_reported)
                self.goto("TEAR", t)
        elif ph == "TEAR":
            self.gait = "creep"
            self.follow.standoff = STANDOFF_INSPECT
            self.phi0_cmd = self.follow.phi0(yaw_g, self.nav_fix(st), st.cur, GAIT_SPEED.get(self.gait, 0.2))
            if tp_ >= 1.0:
                self.goto("INSPECT", t)
        elif ph == "INSPECT":
            self.gait = self.station_gait()
            self.follow.standoff = STANDOFF_INSPECT
            self.phi0_cmd = self.follow.phi0(yaw_g, self.nav_fix(st), st.cur, GAIT_SPEED.get(self.gait, 0.2))
            if wrap(bearing_of(self.nav_fix(st)) - self.det.bearing) < -1.5 / 5.0:   # 1.5 m past the tear
                # turn until heading back along the wall (the sonar's wall yaw reversed), not 180 deg
                # off the crabbed heading: the current pushes inboard here, so the robot crabs 25-40 deg
                p = self.nav_fix(st)
                th = bearing_of(p)
                meas = self.follow.meas if np.isfinite(self.follow.meas) else STANDOFF_INSPECT
                self.r_wall = float(np.clip(math.hypot(p[0], p[2]) + meas, 5.0, 7.2))   # the wall here, from the pass
                self.uturn_yaw0 = yaw_of(np.array([-math.sin(th), 0.0, math.cos(th)]))  # back along the pen (+bearing)
                self.goto("UTURN", t)
        elif ph == "UTURN":
            self.gait = "cruise"
            self.phi0_cmd = self.follow.side * D2R(20.0)   # inboard: away from the wall
            if abs(wrap(st.yaw_mean - self.uturn_yaw0)) < D2R(25.0) or tp_ > 40.0:
                self.follow = WallFollower(STANDOFF_INSPECT, side=-1)
                self.station = False
                self.goto("INSPECT_2", t)
        elif ph == "INSPECT_2":
            # back past the tear on the wall radius the first pass measured (USBL fix, circle ILOS):
            # after the U-turn the wall is on the port side, 2-3 m off and 40 deg off the bow, where
            # the sonar wall fit is not yet trustworthy; the sonar keeps imaging the tear
            p = self.nav_fix(st)
            fm = self.follow.meas                         # the port-side sonar standoff (side -1 follower)
            e_son = (STANDOFF_INSPECT - fm) if (np.isfinite(fm) and self.follow.misses <= 30) else -np.inf
            self.phi0_cmd, e = self.circle_phi0(p, yaw_g, self.r_wall - STANDOFF_INSPECT, dt, cur=st.cur,
                                                v=GAIT_SPEED.get(self.gait, 0.2), e_min=e_son)
            if abs(e) > 0.40:
                self.station = False
            elif abs(e) < 0.20:
                self.station = True
            self.gait = "inspect" if self.station else "creep"
            self.on_wall |= abs(e) < 0.5
            if (self.on_wall or tp_ > 20.0) and wrap(bearing_of(p) - self.det.bearing) > 1.5 / 5.0:
                self.follow.standoff = STANDOFF_FOLLOW
                self.goto("RETURN", t)
        elif ph == "RETURN":
            self.gait = "cruise"
            self.follow.standoff = STANDOFF_FOLLOW
            self.phi0_cmd = self.follow.phi0(yaw_g, self.nav_fix(st), st.cur, GAIT_SPEED.get(self.gait, 0.2))
            if wrap(bearing_of(self.nav_fix(st)) - DOCK_BEARING) > -D2R(45.0):    # leave the wall short of the cradle
                self.choose_entry(st)
                s_c, _ = self.axis_coords(self.com_fix(st), self.a_in)
                if s_c > S_TURN + 0.5:
                    self.goto("APPROACH", t)
                else:                                     # already downstream of the cradle: straight in
                    self.attempts += 1
                    self.goto("FINAL", t)
        elif ph == "APPROACH":                            # the pass line inboard of the cradle, with the current
            self.gait = "cruise"
            o = DOCK_C + PASS_E * DOCK_N
            self.phi0_cmd = self.line_phi0(self.com_fix(st), yaw_g, o, -self.a_in, 1.0, dt, "pass")
            s_c, _ = self.axis_coords(self.com_fix(st), self.a_in)
            if s_c < S_TURN:
                self.goto("TURN_IN", t)
        elif ph == "TURN_IN":                             # outboard, onto the axis, heading into the current
            self.gait = "cruise"
            self.phi0_cmd = self.turn_sign * D2R(25.0)
            if abs(wrap(st.yaw_mean - yaw_of(self.a_in))) < D2R(40.0) or tp_ > 25.0:
                self.zi.pop("final", None)
                self.attempts += 1
                self.goto("FINAL", t)
        elif ph == "FINAL":
            s_nose, e_nose = self.axis_coords(st.nose, self.a_in)
            self.gait = "cruise" if s_nose < -0.5 * DOCK_LEN - 1.0 else "dock"
            # track the point halfway from the body centre to the head: the body's line and the nose both
            p_tr = 0.5 * (self.com_fix(st) + self.nav_fix(st))
            self.phi0_cmd = self.line_phi0(p_tr, yaw_g, DOCK_C, self.a_in, 0.8, dt, "final")
            herr = abs(wrap(st.yaw_mean - yaw_of(self.a_in)))
            if s_nose >= -0.5 * DOCK_LEN:                 # the nose at the mouth plane: the funnel decides
                ok = e_nose < CAPTURE_E and herr < CAPTURE_PSI
                self.attempt_log.append(dict(t=round(t, 1), e=round(e_nose, 3), psi_deg=round(R2D(herr), 1), captured=ok))
                if ok:
                    self.capture_err = (e_nose, R2D(herr))
                    self.s_cap, self.t_cap_start = s_nose, t
                    self.engaged = True
                    self.goto("CAPTURE", t)
                else:
                    self.goto("RETREAT", t)
            elif s_nose > -0.5 * DOCK_LEN - 0.6 and e_nose > 0.6:   # clearly wide of the funnel: back off early
                self.attempt_log.append(dict(t=round(t, 1), e=round(e_nose, 3), psi_deg=round(R2D(herr), 1), captured=False))
                self.goto("RETREAT", t)
        elif ph == "RETREAT":                             # gait off: the current carries it back downstream
            self.gait, self.phi0_cmd = "stop", 0.0
            s_nose, _ = self.axis_coords(st.nose, self.a_in)
            if s_nose < -0.5 * DOCK_LEN - 1.4 or tp_ > 12.0:
                if self.attempts >= MAX_ATTEMPTS:
                    self.goto("ABORT", t)
                else:
                    self.attempts += 1
                    self.zi.pop("final", None)
                    self.goto("FINAL", t)
        elif ph == "CAPTURE":                             # the funnel holds the nose; the latch line hauls it in
            self.gait, self.phi0_cmd = "stop", 0.0
            s_t = min(self.s_cap + V_HAUL * tp_, HALF_BODY)
            hold = dict(target=DOCK_C + s_t * self.a_in, cap=LATCH_CAP, axis=self.a_in, guide=True)
            d = float(np.linalg.norm((st.nose - self.latch)[[0, 2]]))
            # latched once the haul is in and the nose sits within 6 cm of the latch point: the 40 N/m
            # spring holds it ~4 cm short against the current's 1.5-2 N on the body
            if (s_t >= HALF_BODY and d < 0.06) or tp_ > 40.0:
                self.dock_err = (d, R2D(abs(wrap(st.yaw_mean - yaw_of(self.a_in)))))
                self.docked_ok = d < 0.10
                self.returned = True
                self.goto("DOCKED", t)
        elif ph == "ABORT":
            self.gait, self.phi0_cmd = "stop", 0.0
            if tp_ > 5.0:
                self.done = True
        # rate limit on the steering offset: a step in phi0 (a guidance line switch, the U-turn
        # entry) otherwise reaches the joints as a phi_ddot* spike through the paper's feed-forward
        # (23): 50 N m peaks at the undock's line switch, against 10 N m for the PD part
        d = self.phi0_cmd - self.phi0_prev
        lim = PHI0_RATE * dt
        self.phi0_cmd = self.phi0_prev + float(np.clip(d, -lim, lim))
        self.phi0_prev = self.phi0_cmd
        return self.gait, self.phi0_cmd, hold


# ---- the robot side: snake_model.Snake <-> the mission ------------------------------------------------
class SnakeAdapter:
    """Applies the mission's commands to a snake_model.Snake and reads its state (tail-first order)."""

    def __init__(self, snake, latch_k=LATCH_K, latch_c=LATCH_C, latch_cap=LATCH_CAP):
        self.s = snake
        self.gait = None
        self.latch_k, self.latch_c, self.latch_cap = latch_k, latch_c, latch_cap
        self.latch_impulse, self.latch_pull0, self.latch_force = 0.0, None, np.zeros(3)
        self.guide_force, self.guide_max, self.latch_max = 0.0, 0.0, 0.0
        self.phi0 = 0.0

    def state(self, t, cur=None):
        s = self.s
        pos, q = s.link_poses()
        hp, hq, fwd = s.head_pose()
        yaws = np.unwrap(s.link_yaws())
        v, _ = s.velocities(pos, q)
        m = s.mass
        return RobotState(t=t, links=pos, nose=hp + 0.5 * LINK_L * fwd, head_pos=hp, head_yaw=float(yaws[-1]),
                          yaw_mean=float(yaws.mean()), dpsi_head=dpsi_from_joints(s.joint_angles()),
                          com=(m[:, None] * pos).sum(0) / m.sum(),
                          speed=float(np.linalg.norm(((m[:, None] * v).sum(0) / m.sum())[[0, 2]])), cur=cur)

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
        s.extra_force, self.latch_force, self.guide_force = None, np.zeros(3), 0.0
        if latch is None:
            return
        F = np.zeros((s.n, 3))
        v = s.velocities()[0]
        tgt = latch.get("target")
        if tgt is not None:                               # the latch line on the head module
            f = self.latch_k * (np.asarray(tgt) - st.nose) - self.latch_c * v[-1]
            f[1] = 0.0
            n = float(np.linalg.norm(f))
            cap = float(latch.get("cap", self.latch_cap))
            if n > cap:
                f *= cap / n
            F[-1] += f
            self.latch_force = f
            self.latch_impulse += float(np.linalg.norm(f)) * dt
            self.latch_max = max(self.latch_max, float(np.linalg.norm(f)))
        if latch.get("guide"):
            G = cradle_guide(st.links, v)
            F += G
            self.guide_force = float(np.linalg.norm(G, axis=1).max())
            self.guide_max = max(self.guide_max, self.guide_force)
        s.extra_force = F


def cradle_guide(links, v):
    """The cradle bore's contact on the links inside it (one-sided lateral spring-damper beyond a
    free play, capped per link): the rails and funnel hoops, not a propulsor. links/v (n, 3)."""
    F = np.zeros((len(links), 3))
    for k, (p, vk) in enumerate(zip(links, v)):
        d = np.asarray(p, float) - DOCK_C
        s = float(d @ DOCK_U)
        if abs(s) > 0.5 * DOCK_LEN:
            continue
        lat = d - s * DOCK_U
        lat[1] = 0.0
        rho = float(np.linalg.norm(lat))
        if rho <= GUIDE_FREE or rho > 0.5:
            continue
        nrm = lat / rho
        f = -GUIDE_K * (rho - GUIDE_FREE) * nrm - GUIDE_C * float(vk @ nrm) * nrm
        n = float(np.linalg.norm(f))
        if n > GUIDE_CAP:
            f *= GUIDE_CAP / n
        F[k] = f
    return F


# ---- the synthetic world for the CPU harness -------------------------------------------------------------
def current_at(t, seed=0):
    """warp_netpen.current_at, rotated by snake_netpen's --seed jitter."""
    sp = 0.15 * (1.0 + 0.35 * math.sin(0.21 * t))
    ang = 0.35 + 0.25 * math.sin(0.09 * t)
    if seed:
        ang += float(np.random.default_rng(seed).normal(0.0, 0.15))
    return np.array([sp * math.cos(ang), 0.0, sp * math.sin(ang)])


# The live cloth at y -3 after ~75 s of the scene's current (snake_netpen --mission seed 0, the
# telemetry's net ring): the nearest net node per 10 deg bin of pen bearing, bin centres -175..175.
# The current squashes the pen: ~5.7 m on the upstream side (the tear), ~6.2-6.4 m around the
# cradle, 7.0-7.4 m downstream.
NET_R_BINS = np.array([6.09, 5.62, 5.76, 5.66, 5.69, 5.86, 6.21, 6.20, 6.28, 6.42, 6.35, 6.30, 6.31, 6.56,
                       6.61, 6.61, 6.84, 6.82, 7.02, 7.07, 7.40, 7.16, 7.10, 6.74, 6.56, 6.51, 6.37, 6.42,
                       6.45, 6.35, 6.28, 6.31, 6.23, 6.23, 6.25, 6.07])


class SynthWall:
    """The pen wall as the live cloth stands under the current (NET_R_BINS, circular interpolation),
    with a round hole at the tear."""

    def __init__(self, tear_th=TEAR_BEARING, hole=True, r_const=None):
        self.tear_th, self.hole = tear_th, hole
        self.th = np.radians(np.arange(-175.0, 180.0, 10.0))
        self.r = NET_R_BINS if r_const is None else np.full(len(NET_R_BINS), float(r_const))

    def radius(self, th):
        th = (np.asarray(th, np.float64) + np.pi) % (2 * np.pi) - np.pi
        return np.interp(th, self.th, self.r, period=2 * np.pi)

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
    the gait's head-yaw swing and the current's drift; links on a straight line behind the COM.
    A hold (latch target) pulls the nose to the target and the heading onto the hold's axis."""

    def __init__(self, seed=0):
        self.seed = seed
        self.com = DOCK_C.copy()
        self.yaw = HEADING_OUT
        self.gait, self.phi0, self.phase_g, self.amp = "stop", 0.0, 0.0, 0.0
        self.v = 0.0
        self.latch_pull0, self.latch_impulse, self.latch_max, self.guide_max = None, 0.0, 0.0, 0.0

    def state(self, t, cur=None):
        f = yaw_dir(self.yaw)
        ks = np.arange(LINK_N) - (LINK_N - 1) / 2.0
        links = self.com + np.outer(ks * LINK_L, f)
        dpsi = self.amp * math.sqrt(2.0) * math.sin(self.phase_g)
        return RobotState(t=t, links=links, nose=self.com + HALF_BODY * f, head_pos=links[-1],
                          head_yaw=self.yaw + dpsi, yaw_mean=self.yaw, dpsi_head=dpsi, com=self.com.copy(),
                          speed=self.v, cur=cur)

    def apply(self, gait, phi0, hold, st, dt):
        self.gait = gait
        self.phi0 += (phi0 - self.phi0) * (1.0 - math.exp(-dt / 0.5))
        v0 = GAIT_SPEED[gait]
        self.v += (v0 * (1.0 - 0.35 * (self.phi0 / D2R(20.0)) ** 2) - self.v) * (1.0 - math.exp(-dt / 2.0))
        self.amp += (GAIT_HEADYAW[gait] - self.amp) * (1.0 - math.exp(-dt / 2.0))
        self.phase_g += GAIT_OMEGA[gait] * dt
        self.yaw += GAIT_TURN[gait] * self.phi0 * (self.v / max(v0, 1e-6) if v0 > 0 else 0.0) * dt
        cur = current_at(st.t, self.seed)
        if hold is not None and hold.get("target") is not None:
            f = np.asarray(hold["target"]) - st.nose
            f[1] = 0.0
            self.com += f * min(1.0, dt / 1.0)
            self.yaw += wrap(yaw_of(hold["axis"]) - self.yaw) * min(1.0, dt / 1.0)
            self.v = 0.0
            return
        if hold is not None and hold.get("guide"):        # in the bore: along the axis only
            a = np.asarray(hold["axis"], float)
            self.com = self.com + float((self.v * yaw_dir(self.yaw) + cur) @ a) * a * dt
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

    def state(self, t, cur=None):
        return self.ad.state(t, cur)

    def apply(self, gait, phi0, hold, st, dt):
        self.ad.apply(gait, phi0, hold, st, dt)
        self.snake.set_current(current_at(st.t, self.seed))
        self.world.step(dt)

    def close(self):
        """PhysX allows one foundation per process: free this world before the next robot's."""
        import gc
        self.snake.remove()
        self.ad = self.snake = self.world = None
        gc.collect()

    @property
    def latch_pull0(self):
        return self.ad.latch_pull0

    @property
    def latch_impulse(self):
        return self.ad.latch_impulse

    @property
    def latch_max(self):
        return self.ad.latch_max

    @property
    def guide_max(self):
        return self.ad.guide_max


def run_synthetic(robot, seed=0, t_cap=300.0, dt=1.0 / 60.0, every=3, hole=True, verbose=True):
    """The mission on the synthetic pen: the robot (Unicycle or PhysxRobot) + synth_sonar."""
    rng = np.random.default_rng(seed)
    wall = SynthWall(hole=hole)
    fish = Fish(rng)
    ms = Mission(seed=seed)
    rows = []
    min_d, min_where = float("inf"), ""
    n = int(t_cap / dt)
    for f in range(n):
        t = f * dt
        st = robot.state(t, current_at(t, seed))
        if f % every == 0:
            fish.step(every * dt)
            yaw_s = st.head_yaw
            son = st.nose - 0.05 * yaw_dir(yaw_s)
            ms.on_sonar(synth_sonar(son, yaw_s, wall, fish, rng), son, yaw_s, st, every * dt)
        gait, phi0, hold = ms.step(st, dt)
        robot.apply(gait, phi0, hold, st, dt)
        d = float(wall.dist(st.links).min()) - LINK_R
        if d < min_d:
            min_d, min_where = d, f"{ms.phase} t {t:.1f}"
        true_d = float(wall.dist(st.head_pos[None])[0])
        rows.append((t, PHASES.index(ms.phase), *st.com[[0, 2]], st.yaw_mean, phi0, ms.follow.meas, true_d, d))
        if ms.done:
            break
    L = np.array(rows)
    res = dict(seed=seed, done=ms.done, docked=ms.docked_ok, t_end=float(L[-1, 0]), min_d=min_d, min_where=min_where,
               tear_mode=ms.det.mode,
               tear_err_deg=R2D(wrap(ms.det.bearing - TEAR_BEARING)) if ms.det.fired else float("nan"),
               tear_t=ms.det.t_fire, dock_err=ms.dock_err, capture_err=ms.capture_err, attempts=ms.attempts,
               attempt_log=ms.attempt_log, events=[(round(a, 1), b) for a, b in ms.events],
               latch_max=robot.latch_max, guide_max=robot.guide_max)
    fw = L[:, 1] == PHASES.index("FOLLOW_WALL")
    if fw.any():
        res["follow_meas_minus_true_rms"] = float(np.sqrt(np.nanmean((L[fw, 6] - L[fw, 7]) ** 2)))
        res["follow_true_mean"] = float(np.nanmean(L[fw, 7]))
    if verbose:
        print(f"  seed {seed}: docked {res['docked']} t {res['t_end']:.1f} s, min link-to-net {min_d:.2f} m ({min_where}), "
              f"tear {res['tear_mode']} err {res['tear_err_deg']:+.1f} deg at {res['tear_t']:.1f} s, attempts {ms.attempts}, "
              f"capture {res['capture_err']}, dock err {res['dock_err']}, latch max {res['latch_max']:.1f} N, "
              f"guide max {res['guide_max']:.1f} N")
        print("   events " + ", ".join(f"{b}@{a}" for a, b in res["events"]))
        if ms.attempt_log:
            print("   attempts " + "; ".join(str(a) for a in ms.attempt_log))
    return res, L


PHASE_COLORS = {"DOCKED": "#555555", "UNDOCK": "#8c564b", "ACQUIRE": "#bcbd22", "FOLLOW_WALL": "#1f77b4",
                "TEAR": "#d62728", "INSPECT": "#ff7f0e", "UTURN": "#9467bd", "INSPECT_2": "#e377c2",
                "RETURN": "#17becf", "APPROACH": "#2ca02c", "TURN_IN": "#98df8a", "FINAL": "#006400",
                "CAPTURE": "#000000", "RETREAT": "#ff9896", "ABORT": "#ff0000"}


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
    out = dict(completed=bool(mission.done and mission.docked_ok), mission_time_s=round(float(t[-1]), 1),
               path_length_m=round(float(seg[moving].sum()), 2),
               events=[[round(a, 2), b] for a, b in mission.events])
    per = {}
    for g in ("cruise", "inspect", "creep", "dock", "undock"):
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
    dmin = L[:, c["d_min_links"]]
    k = int(np.nanargmin(dmin))
    out["min_link_to_net_m"] = round(float(dmin[k]), 3)                  # the whole mission, docked included
    out["min_link_to_net_where"] = f"{names[ph[k]]} t {t[k]:.1f} s"
    det = mission.det
    sonar = det.fired and det.mode != "sector"
    out["tear"] = dict(detected=bool(det.fired),
                       mode=("sector (sonar did not confirm)" if det.mode == "sector" else det.mode),
                       t_s=round(float(det.t_fire), 1) if det.fired else None,
                       bearing_deg=round(R2D(det.bearing), 2) if sonar else None,
                       bearing_err_deg=round(R2D(wrap(det.bearing - float(tel["tear_true"]))), 2) if sonar else None,
                       candidates=len(det.all_cands))
    if "d_head_tear" in c:
        for nm in ("INSPECT", "INSPECT_2"):
            m = ph == names.index(nm)
            if m.any():
                k = int(np.nanargmin(np.where(m, L[:, c["d_head_tear"]], np.inf)))
                out[f"{nm.lower()}_closest_head_to_tear_m"] = round(float(L[k, c["d_head_tear"]]), 2)
                out[f"{nm.lower()}_closest_t_s"] = round(float(t[k]), 1)
    out["dock_error"] = None if mission.dock_err is None else dict(nose_m=round(mission.dock_err[0], 3),
                                                                   heading_deg=round(mission.dock_err[1], 1))
    out["capture_error"] = None if mission.capture_err is None else dict(nose_off_axis_m=round(mission.capture_err[0], 3),
                                                                         heading_deg=round(mission.capture_err[1], 1))
    out["dock_attempts"] = mission.attempts
    out["attempt_log"] = mission.attempt_log
    out["entry"] = "downstream end, heading into the current" if float(mission.a_in @ DOCK_U) > 0 else "upstream end"
    lf = L[:, c["latch_f"]]
    out["latch"] = dict(max_N=round(float(np.nanmax(lf)), 2), max_N_after_undock=round(float(np.nanmax(np.where(t > mission.t_docked + 0.1, lf, 0.0))), 2),
                        impulse_Ns=round(adapter.latch_impulse, 3), cap_N=adapter.latch_cap,
                        guide_max_N_per_link=round(adapter.guide_max, 2), guide_cap_N=GUIDE_CAP)
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
    if "tear_x" in c:                                     # where the hole is on the bowed wall (tear_now)
        ax.plot(L[-1, c["tear_x"]], L[-1, c["tear_z"]], "rx", ms=12, mew=3, label="tear (true, bowed wall)")
    else:
        ax.plot(PEN_R * math.cos(tt), PEN_R * math.sin(tt), "rx", ms=12, mew=3, label="tear (true, rest)")
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
    """Hole abeam at the 1.5 m follow standoff fires (1.25 m reported only: the 28 deg vertical fan
    straddles the hole closer in); the intact wall never does. The wall here is a smooth circle at
    the cloth's radius around the tear (5.7 m): the 10 deg binned table kinks by 0.1 m at the hole."""
    ok = True
    rng = np.random.default_rng(3)
    fires = []
    for hole, so, gate in ((True, 1.5, False), (True, 1.25, False), (False, 1.25, True), (False, 1.5, True)):
        wall = SynthWall(hole=hole, r_const=5.7)
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
        if hole and fired:
            good = good and abs(err) < 5.0
        if gate:
            ok &= good
        elif good:
            fires.append(so)
        tag = ("ok  " if good else "FAIL") if gate else "info"
        print(f"  {tag} detector hole={hole} standoff {so}: fired {fired} bearing err {err:+.1f} deg")
    print(f"  {'ok  ' if fires else 'FAIL'} detector: the hole fires within 5 deg at standoff(s) {fires} m")
    return ok and bool(fires)


def selftest(physx=False, seeds=(0, 1, 2)):
    t0 = time.perf_counter()
    ok = detector_unit()

    def gate(res):
        # 0.5 m here: the synthetic wall is the NEAREST cloth node per 10 deg bin (an envelope inside
        # the scalloped cloth); the scene gates 0.6 m on the live cloth itself
        return res["done"] and res["docked"] and res["min_d"] >= 0.5

    print("unicycle stand-in on the synthetic pen (the live cloth's shape, the scene's current):")
    for seed in seeds:
        res, _ = run_synthetic(Unicycle(seed), seed)
        good = gate(res)
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} unicycle seed {seed}")
    if physx:
        print("PhysX snake on the synthetic pen:")
        for seed in seeds:
            t1 = time.perf_counter()
            rob = PhysxRobot(seed)
            res, L = run_synthetic(rob, seed)
            rob.close()
            del rob
            good = gate(res)
            ok &= good
            print(f"  {'ok  ' if good else 'FAIL'} physx seed {seed} ({time.perf_counter() - t1:.0f} s wall)")
    print(f"selftest: {'all ok' if ok else 'FAILURES'} ({time.perf_counter() - t0:.0f} s)")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--physx", action="store_true")
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest(a.physx, tuple(int(s) for s in a.seeds.split(","))) else 1)
    print(__doc__)
