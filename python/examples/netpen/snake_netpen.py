"""An eel-like snake robot (NTNU Mamba style) patrols the net pen of warp_netpen.py.

The BlueROV2 is retired; a 9-link yellow snake lives in a cradle hung from the inner collar,
undocks, follows the net wall at ~1.5 m, passes the tear slowly at ~1.25 m, U-turns, passes it
again, swims back and docks into the current through a funnel (snake_mission.py). This file is the
SCENE: the snake visual, the dock, the hooks into warp_netpen (head camera, head sonar, fish
avoidance) and a kinematic placeholder for the motion. The swimming physics (snake_model.py)
plugs in through the same pose-provider interface.

    python snake_netpen.py                                   # window; drag to orbit, Esc quits
    python snake_netpen.py --snake-shot snake_hud --out x.png --seconds 8 --size 1600x900
    python snake_netpen.py --snake-shot all --out D:/dev/snake_out/shots --tag _v1

Flag contract
  Own flags (warp_netpen never sees them):
    --snake-shot NAME   headless still: snake_dock snake_school snake_tear snake_hud snake_top, a comma
                        list of them, or `all` (one process renders the list in order, top last)
    --out PATH          the PNG (one shot) or the directory (a list; default D:/dev/snake_out/shots)
    --tag SUFFIX        appended to each still's name (e.g. _v1)
    --seed N            current heading jitter: N = 0 is the scene's own current, N > 0 rotates
                        warp_netpen.current_at by a seeded N(0, 0.15) rad
    --snake-film PATH   reserved for the film phase (P4); exits with a message today
    --mission           headless closed-loop mission (PhysX snake + snake_mission), telemetry, summary
    --mission-shots     ... plus a still a moment after each phase change
    --dump-sonar        with --mission: save up to 600 sonar images near the wall (seedN_sonar_dump.npz)
    --t-cap S  --render-every N   mission time cap; main-view render cadence
  Passed through to warp_netpen (read by it at import):
    --size WxH  --seconds S (scene warm-up before a still)  --fish N  --terrain [dir]
    --no-interop  --no-ocean
  Set for it: headless stills import warp_netpen with `--shot p3_hud` (a valid key: HEADLESS and
  the HUD insets on); a still that does not want the insets parks them. Nothing else reaches it.

Pose-provider interface (KinematicSnake here, the physics in P3; link 0 = HEAD, link 8 = tail):
    poses(t)     -> (pos (9, 3) float64 link centres, quat (9, 4) float64 x, y, z, w), world frame,
                    link forward = local +x, up = local +y (scene convention); t = mission seconds,
                    called once per scene frame with non-decreasing t
    head_pose()  -> (pos (3,), R (3, 3)) of the head link at the last poses() call;
                    R columns = forward, up, starboard
    nose()       -> (3,) the front tip of the head (camera, lamps and sonar mount here)
    mid_point()  -> (3,) link 4's centre: the fish school's single avoidance point
    phase        -> str, the mission phase for the HUD
"""
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.dirname(_HERE)
_PY = os.path.dirname(_EX)
for _p in (_HERE, _EX, _PY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from demo_common import cli_arg
import snake_mission as MS                                # pure: the mission, the cradle geometry

# ---- flags: ours first, then the argv warp_netpen will see at import --------------------------
SNAKE_SHOT = cli_arg("--snake-shot", "", str)
MISSION = "--mission" in sys.argv                         # headless closed-loop mission, telemetry + summary
MISSION_SHOTS = "--mission-shots" in sys.argv             # ... plus a still at every phase change
KINEMATIC = "--kinematic" in sys.argv                     # the window runs the kinematic placeholder
T_CAP = cli_arg("--t-cap", 300.0, float)
RENDER_EVERY = cli_arg("--render-every", 3, int)
SNAKE_FILM = "--snake-film" in sys.argv
DUMP_SONAR = "--dump-sonar" in sys.argv                   # --mission: save the sonar images near the wall (npz)
SEED = cli_arg("--seed", 0, int)
TAG = cli_arg("--tag", "", str)
SIZE = cli_arg("--size", "1600x900", str)
SECONDS = cli_arg("--seconds", 8.0, float)
SHOT_NAMES = ("snake_dock", "snake_school", "snake_tear", "snake_hud", "snake_top")
OUT_DIR_DEFAULT = "D:/dev/snake_out/shots"
OUT = cli_arg("--out", "", str)
if SNAKE_FILM:
    sys.exit("--snake-film is the film phase's (P4); not implemented in the scene phase")
SHOT_LIST = SHOT_NAMES if SNAKE_SHOT == "all" else tuple(n for n in SNAKE_SHOT.split(",") if n)
if any(n not in SHOT_NAMES for n in SHOT_LIST):
    sys.exit(f"unknown --snake-shot {SNAKE_SHOT!r}; one or a comma list of {', '.join(SHOT_NAMES)}, or all")
HEADLESS = bool(SNAKE_SHOT) or MISSION or MISSION_SHOTS

_w_argv = [os.path.join(_HERE, "warp_netpen.py"), "--size", SIZE, "--seconds", str(SECONDS)]
if HEADLESS:
    _w_argv += ["--shot", "p3_hud"]
for _flag in ("--fish",):
    if _flag in sys.argv:
        _w_argv += [_flag, str(cli_arg(_flag, 400, int))]
if "--terrain" in sys.argv:
    _k = sys.argv.index("--terrain")
    _w_argv += ["--terrain"] + ([sys.argv[_k + 1]] if _k + 1 < len(sys.argv) and not sys.argv[_k + 1].startswith("--") else [])
for _flag in ("--no-interop", "--no-ocean"):
    if _flag in sys.argv:
        _w_argv.append(_flag)
_argv_saved, sys.argv = sys.argv, _w_argv
import warp_netpen as W                                   # builds the whole scene, renders once, adds the ROV view
sys.argv = _argv_saved
import threepp as tp

standard_material = W.standard_material

# ---- geometry of the mission --------------------------------------------------------------------
LINK_N, LINK_L, LINK_R = 9, 0.18, 0.0525
BODY_L = LINK_N * LINK_L                                  # 1.62 m
DEPTH_Y = W.TEAR_Y                                        # -3.0: the tear's depth
R_PATROL, R_INSPECT = 5.5, 6.0                            # nominal radii: 1.5 m / 1.0 m off the rest wall (r 7)
STANDOFF_FOLLOW, STANDOFF_INSPECT = 1.5, 1.0              # the kinematic keeps at least these off the live cloth
DOCK_BEARING = math.radians(-95.0)                        # 59 deg of arc from the tear (bearing -154.2 deg)
TEAR_BEARING = float(W.TEAR_TH)
DOCK_LEN = 2.0                                            # hoop plane to hoop plane
V_CRUISE, V_INSPECT = 0.15, 0.10                          # m/s
T_DOCKED = 6.0                                            # mission seconds held in the cradle before undocking
T_UNDOCK_RAMP = 4.0
OMEGA = 1.6                                               # rad/s gait frequency at cruise (~92 deg/s)
WAVELEN = 1.3                                             # m: ~1.25 body waves
A_TAIL = 0.12                                             # m lateral amplitude at the tail; eel-like envelope
UTURN_R = 0.6
PUSH_SLOPE = 0.35                                         # max inboard push per metre of path (kinematic only)
XS_DENSE = np.linspace(0.0, 1.6 * 9 * 0.18, 145)           # body stations for the dense target centre line


def polar(th, r, y=DEPTH_Y):
    return np.array([r * math.cos(th), y, r * math.sin(th)])


def e_r(th):
    return np.array([math.cos(th), 0.0, math.sin(th)])


DOCK_C = MS.DOCK_C.copy()                                 # r 4.7, the axis yawed 18 deg (snake_mission.py says why)
DOCK_U = MS.DOCK_U.copy()                                 # dock axis, pointing toward the tear
D2R = math.radians


def _catmull(P, n=48):
    """Uniform Catmull-Rom through control points P (k, 3) -> dense polyline and the dense index of each P."""
    P = np.asarray(P, np.float64)
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out, marks = [], []
    u = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        marks.append(len(out) * n)
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * u ** 3))
    dense = np.vstack(out + [P[-1:]])
    marks.append(len(dense) - 1)
    return dense, marks


def build_path():
    """The head's path through the whole mission and the arc length of every named waypoint."""
    TT, DB, u = TEAR_BEARING, DOCK_BEARING, DOCK_U
    pts, names = [], []

    def add(p, name=""):
        pts.append(np.asarray(p, np.float64))
        names.append(name)

    for d, nm in ((-2.6, ""), (-1.6, ""), (-0.81, ""), (0.0, ""), (0.81, "head_docked")):
        add(DOCK_C + d * u, nm)
    add(polar(DB - D2R(15.0), R_PATROL), "dock_exit")     # the r 5.5 arc is tangent to the cradle axis at the dock
    th = DB - D2R(21.0)
    while th > TT + D2R(20.0):
        add(polar(th, R_PATROL))
        th -= D2R(6.0)
    add(polar(TT + D2R(14.0), 0.5 * (R_PATROL + R_INSPECT)), "inspect_1")
    for a in (8.0, 2.0, -4.0, -10.0):
        add(polar(TT + D2R(a), R_INSPECT))
    th_e = TT - D2R(16.0)
    add(polar(th_e, R_INSPECT), "uturn")
    c = polar(th_e, R_INSPECT - UTURN_R)
    ue = np.array([math.sin(th_e), 0.0, -math.cos(th_e)])
    for psi in (40.0, 90.0, 140.0):
        add(c + UTURN_R * (math.cos(D2R(psi)) * e_r(th_e) + math.sin(D2R(psi)) * ue))
    add(polar(th_e, R_INSPECT - 2.0 * UTURN_R), "inspect_2")
    for a, r in ((-10.0, 5.3), (-4.0, 5.8), (2.0, R_INSPECT), (8.0, R_INSPECT)):
        add(polar(TT + D2R(a), r))
    add(polar(TT + D2R(15.0), 0.5 * (R_PATROL + R_INSPECT)), "return")
    th = TT + D2R(21.0)
    while th < DB - D2R(20.0):
        add(polar(th, R_PATROL))
        th += D2R(6.0)
    add(polar(DB - D2R(15.0), R_PATROL), "dock_entry")
    for d, nm in ((0.81, ""), (0.0, ""), (-0.81, "head_home"), (-1.6, ""), (-2.6, "")):
        add(DOCK_C + d * u, nm)
    dense, marks = _catmull(pts)
    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    tan = np.gradient(dense, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-12)
    keys = {nm: float(s[m]) for nm, m in zip(names, marks) if nm}
    return dense, s, tan, keys, np.asarray(pts)


PATH_P, PATH_S, PATH_T, PATH_KEYS, PATH_CTRL = build_path()


def path_at(s):
    s = np.atleast_1d(np.asarray(s, np.float64))
    p = np.stack([np.interp(s, PATH_S, PATH_P[:, k]) for k in range(3)], -1)
    t = np.stack([np.interp(s, PATH_S, PATH_T[:, k]) for k in range(3)], -1)
    # beyond either end: straight on along the end tangent (the tail's history behind the cradle)
    lo, hi = s < 0.0, s > PATH_S[-1]
    p[lo] = PATH_P[0] + np.outer(s[lo], PATH_T[0])
    p[hi] = PATH_P[-1] + np.outer(s[hi] - PATH_S[-1], PATH_T[-1])
    return p, t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)


def smoothstep(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def speed_at(s, t):
    """Commanded along-path speed: undock ramp in time, slow pass at the tear, creep into the cradle."""
    k = PATH_KEYS
    v = V_CRUISE * smoothstep((t - T_DOCKED) / T_UNDOCK_RAMP)
    # slow from 0.6 m before the first inspection waypoint to 0.6 m after the return one
    a = smoothstep((s - (k["inspect_1"] - 1.2)) / 1.2) * (1.0 - smoothstep((s - k["return"]) / 1.2))
    v = min(v, V_CRUISE + (V_INSPECT - V_CRUISE) * a)
    left = k["head_home"] - s
    if left <= 0.0:
        return 0.0
    return min(v, max(0.012, V_CRUISE * left / 1.5))


def build_clock(dt=1.0 / 240.0, t_max=400.0):
    """s(t) and the gait phase phi(t) integrated once at 240 Hz; poses() interpolates between rows."""
    k = PATH_KEYS
    ts, ss, ph, vs = [0.0], [k["head_docked"]], [0.0], [0.0]
    t, s, phi = 0.0, k["head_docked"], 0.0
    while t < t_max:
        v = speed_at(s, t)
        s = min(s + v * dt, k["head_home"])
        phi += OMEGA * min(v / V_CRUISE, 1.0) * dt
        t += dt
        ts.append(t); ss.append(s); ph.append(phi); vs.append(v)
        if s >= k["head_home"] and t > T_DOCKED + 10.0:
            break
    return np.array(ts), np.array(ss), np.array(ph), np.array(vs)


CLK_T, CLK_S, CLK_PHI, CLK_V = build_clock()
T_MISSION = float(CLK_T[-1])


def quat_from_R(R):
    """(x, y, z, w) of a proper rotation matrix."""
    m00, m11, m22 = R[0, 0], R[1, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0.0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return np.array([(R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s])
    if m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        return np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
    if m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        return np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
    s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
    return np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s])


def frame_from_fwd(f):
    up = np.array([0.0, 1.0, 0.0])
    up = up - f * float(f @ up)
    up /= max(np.linalg.norm(up), 1e-12)
    return np.column_stack([f, up, np.cross(f, up)])


def net_positions():
    return W.net_host[0] if W.net_host[0] is not None else W.net.pos.numpy()


class KinematicSnake:
    """Placeholder motion with the physics' interface: a travelling-wave body riding the mission path.

    Follow-the-leader: the head is at path arc length s(t) (a 240 Hz table, interpolated); body
    station x (0 at the nose) sits at s - x, displaced along the horizontal path normal by
    a(x) sin(k x - phi(t)), eel-like envelope a(x) = A_TAIL (0.2 + 0.8 x / L). Where the live cloth
    bows in closer than the phase's standoff (+ radius + half the tail swing), the line is pushed
    inboard: looked up to 3 m ahead, at most PUSH_SLOPE m per metre travelled, zero at the cradle,
    recorded against s so the body follows the head's line. Joints are placed by arc length along
    that dense target line and re-chained at exactly LINK_L. Stateful: poses() wants
    non-decreasing t (the stills' pre-roll replays it from 20 s before the roll)."""

    n_links, link_len, radius = LINK_N, LINK_L, LINK_R

    def __init__(self):
        self.x = np.arange(LINK_N + 1) * LINK_L
        self.rec_s, self.rec_off = [], []
        self.off, self.off_rl, self.t_last, self.s_prev = 0.0, 0.0, None, None
        self.pos = np.zeros((LINK_N, 3))
        self.quat = np.tile([0.0, 0.0, 0.0, 1.0], (LINK_N, 1))
        self.R = np.zeros((LINK_N, 3, 3))
        self.joints = np.zeros((LINK_N + 1, 3))
        self.phase, self.speed, self.s = "DOCKED", 0.0, PATH_KEYS["head_docked"]
        self.max_joint = 0.0

    def state(self, t):
        t = min(max(t, 0.0), T_MISSION)
        return (float(np.interp(t, CLK_T, CLK_S)), float(np.interp(t, CLK_T, CLK_PHI)), float(np.interp(t, CLK_T, CLK_V)))

    def phase_of(self, s, t):
        k = PATH_KEYS
        if t < T_DOCKED:
            return "DOCKED"
        if s >= k["head_home"] - 1e-6:
            return "DOCKED"
        for name, key in (("UNDOCK", "dock_exit"), ("FOLLOW_WALL", "inspect_1"), ("INSPECT", "uturn"),
                          ("U_TURN", "inspect_2"), ("INSPECT", "return"), ("RETURN", "dock_entry")):
            if s < k[key]:
                return name
        return "DOCK"

    def standoff_at(self, s):
        k = PATH_KEYS
        return STANDOFF_INSPECT if k["inspect_1"] - 0.5 <= s <= k["return"] + 0.5 else STANDOFF_FOLLOW

    def offset_at(self, s):
        if not self.rec_s:
            return np.zeros_like(s)
        return np.interp(s, self.rec_s, self.rec_off)

    def poses(self, t):
        s, phi, v = self.state(t)
        self.t_last = t
        k = PATH_KEYS
        # live-wall push: only between the dock exit and the dock entry, only ever inboard
        want = 0.0
        q = net_positions()
        for sa in (s, s + 1.0, s + 2.0, s + 3.0):         # look ahead: the push is slope-limited below
            w = smoothstep((sa - k["dock_exit"]) / 1.5) * smoothstep((k["dock_entry"] - sa) / 1.5)
            if w > 0.0:                                   # never near the cradle: it hangs where it hangs
                p_nom = path_at(sa)[0][0]
                d = float(np.sqrt(((q - p_nom) ** 2).sum(1).min()))
                want = max(want, w * (self.standoff_at(sa) + LINK_R + 0.5 * A_TAIL - d))
        if self.s_prev is None:
            self.off_rl = self.off = want
        else:
            # per metre travelled, not per second, and at most 0.35 m of push per metre: a steeper
            # push folds the body where the path heads radially (the U-turn)
            ds = max(s - self.s_prev, 0.0)
            self.off_rl += min(max(want - self.off_rl, -PUSH_SLOPE * ds), PUSH_SLOPE * ds)
        # zero in and near the cradle, ramping in and out at the same slope
        self.off_rl = min(self.off_rl, PUSH_SLOPE * max(min(s - k["dock_exit"], k["dock_entry"] - s), 0.0))
        if self.s_prev is None:
            self.off = self.off_rl
        else:
            self.off += (self.off_rl - self.off) * (1.0 - math.exp(-ds / 0.5))
        self.s_prev = s
        if not self.rec_s or s > self.rec_s[-1] + 1e-4:
            self.rec_s.append(s)
            self.rec_off.append(self.off)
        elif self.rec_s:
            self.rec_off[-1] = self.off
        # the target centre line, densely: the inboard push shortens it (a smaller radius), so the
        # joints are placed by ARC LENGTH along it, never at fixed stations (those overshoot and fold)
        xs = XS_DENSE
        sx = s - xs
        P, T = path_at(sx)
        rr = np.hypot(P[:, 0], P[:, 2])
        inb = -np.stack([P[:, 0] / rr, np.zeros_like(rr), P[:, 2] / rr], -1)
        P = P + inb * self.offset_at(sx)[:, None]
        N = np.stack([T[:, 2], np.zeros_like(rr), -T[:, 0]], -1)
        N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
        amp = min(v / 0.06, 1.0) if t >= T_DOCKED else 0.0
        lat = amp * A_TAIL * (0.2 + 0.8 * np.minimum(xs, BODY_L) / BODY_L) * np.sin(2.0 * math.pi / WAVELEN * xs - phi)
        Qd = P + N * lat[:, None]
        cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(Qd, axis=0), axis=1))])
        Q = np.stack([np.interp(self.x, cum, Qd[:, c]) for c in range(3)], -1)
        j = np.empty_like(Q)
        j[0] = Q[0]
        for i in range(LINK_N):
            d = Q[i + 1] - j[i]
            j[i + 1] = j[i] + LINK_L * d / max(np.linalg.norm(d), 1e-12)
        self.joints = j
        for i in range(LINK_N):
            f = j[i] - j[i + 1]
            f /= np.linalg.norm(f)
            self.R[i] = frame_from_fwd(f)
            self.quat[i] = quat_from_R(self.R[i])
        self.pos = 0.5 * (j[:-1] + j[1:])
        fw = self.R[:, :, 0]
        if t >= T_DOCKED:
            ang = np.arccos(np.clip((fw[:-1] * fw[1:]).sum(1), -1.0, 1.0))
            self.max_joint = max(self.max_joint, float(ang.max()))
        self.phase, self.speed, self.s = self.phase_of(s, t), v, s
        return self.pos.copy(), self.quat.copy()

    def head_pose(self):
        return self.pos[0].copy(), self.R[0].copy()

    def nose(self):
        return self.joints[0].copy()

    def mid_point(self):
        return self.pos[LINK_N // 2].copy()


class PhysicsSnake:
    """The swimming physics (snake_model.Snake, PhysX, 240 Hz: 4 substeps per 1/60 s scene step) closed
    around snake_mission.Mission on the head sonar's images. Same interface as KinematicSnake.
    Model links are tail-first with the link axis on local +Y; the scene wants head-first with
    forward on local +x and up on local +y, so the order is reversed and each rotation is
    right-multiplied by C = [t_loc, up_loc, t_loc x up_loc]. Cannot rewind: t only moves forward."""

    n_links, link_len, radius = LINK_N, LINK_L, LINK_R

    def __init__(self, seed=0):
        import snake_model as SM
        import snake_mission as MS
        self.MS = MS
        self.world = SM.make_world(1.0 / 240.0)
        self.snake = SM.Snake(self.world, SM.SnakeParams(), origin=tuple(float(v) for v in MS.DOCK_C),
                              heading=MS.HEADING_OUT)
        self.ad = MS.SnakeAdapter(self.snake)
        self.mission = MS.Mission(seed=seed, tear_reported=TEAR_BEARING)
        s = self.snake
        self.C = np.column_stack([s.t_loc, s.up_loc, np.cross(s.t_loc, s.up_loc)])
        self.t_last, self.st, self.images = None, None, 0
        self.phase, self.speed, self.max_joint, self.off = "DOCKED", 0.0, 0.0, 0.0
        self.y0 = float(MS.DOCK_C[1])
        self.max_dy = 0.0
        self._read()
        self.st = self.ad.state(0.0)

    def _read(self):
        from snake_model import quat_to_R
        pos, q = self.snake.link_poses()
        R = quat_to_R(q) @ self.C
        self.pos = pos[::-1].copy()
        self.R = R[::-1].copy()
        self.quat = np.array([quat_from_R(r) for r in self.R])
        f = self.R[0][:, 0]
        self.joints = np.vstack([self.pos[0] + 0.5 * LINK_L * f, self.pos - 0.5 * LINK_L * self.R[:, :, 0]])
        self.max_dy = max(self.max_dy, float(np.abs(pos[:, 1] - self.y0).max()))

    def poses(self, t):
        if t < 0.0:
            return self.pos.copy(), self.quat.copy()
        if self.t_last is None:
            self.t_last = t
            return self.pos.copy(), self.quat.copy()
        n = int(round((t - self.t_last) * 240.0))
        if n > 0:
            h = n / 240.0
            st = self.ad.state(self.t_last, W.current_at(W.world_t))   # cur: the station's current meter
            for img, o, yaw in SONAR_IN:                  # the last image(s), with the pose they were fired from
                self.mission.on_sonar(img, o, yaw, st, W.SON_EVERY / 60.0)
                self.images += 1
            SONAR_IN.clear()
            gait, phi0, hold = self.mission.step(st, h)
            self.ad.apply(gait, phi0, hold, st, h)
            self.snake.set_current(W.current_at(W.world_t))
            self.world.step(h)
            self.t_last += h
            self.st = st
            self.phase, self.speed = self.mission.phase, st.speed
            self._read()
        return self.pos.copy(), self.quat.copy()

    def head_pose(self):
        return self.pos[0].copy(), self.R[0].copy()

    def nose(self):
        return self.joints[0].copy()

    def mid_point(self):
        return self.pos[LINK_N // 2].copy()


def mission_time_of(key, frac=0.0):
    """Mission time at which the head reaches waypoint `key` (+ frac of the way to the next metre)."""
    s = PATH_KEYS[key] + frac
    i = int(np.searchsorted(CLK_S, s))
    return float(CLK_T[min(i, len(CLK_T) - 1)])


def mission_time_head_at_bearing(th, first_after=0.0):
    """First mission time after `first_after` at which the head's bearing crosses `th` (outbound)."""
    P = path_at(CLK_S)[0]
    b = np.arctan2(P[:, 2], P[:, 0])
    ok = (CLK_T >= first_after) & (np.abs(((b - th + math.pi) % (2 * math.pi)) - math.pi) < D2R(1.0))
    i = int(np.argmax(ok))
    return float(CLK_T[i])


# ---- the snake visual --------------------------------------------------------------------------
SON_ID_SNAKE = 9
yellow = tp.MeshPhysicalMaterial()
yellow.color = 0xffc20e                                  # safety yellow
yellow.roughness = 0.45
yellow.metalness = 0.0
yellow.specular_intensity = 0.4                           # full F0 chromes a dark-ish surface under the bright env
rubber = standard_material(0x141516, roughness=0.8)
smoked = tp.MeshPhysicalMaterial()
smoked.color = 0x0c0f12
smoked.roughness = 0.08
smoked.metalness = 0.0
smoked.clearcoat = 1.0
smoked.clearcoat_roughness = 0.05
lens_mat = standard_material(0x05070c, roughness=0.12)
ring_mat = standard_material(0x9aa0a6, roughness=0.35, metalness=0.9)
LAMP_I = 550.0                                            # 800 put a backscatter disc over the head-cam inset
lamp_disc_mat = standard_material(0xfff2dc, roughness=0.3, emissive=0xffe6c0, emissive_intensity=90.0)

snake_links, snake_meshes = [], []


def _add(parent, mesh, shadow=True):
    mesh.cast_shadow = shadow
    parent.add(mesh)
    snake_meshes.append(mesh)
    return mesh


def _along_x(mesh):
    mesh.rotation.z = -math.pi / 2                        # geometry +Y -> +X
    return mesh


# Machined modules, rubber boots at the joints. Per link: a yellow module shell (slimmer than the
# head), a thin seam with a bolt ring, raised end rings with a chamfer down to the joint, and half a
# ridged boot at each joint end; the two halves ride their own links, so a bent joint opens the boot
# on the outside of the bend (a dark core fills it) and closes it on the inside. Merged per material.
MOD_R = 0.049                                             # module shell radius (the head keeps LINK_R)
MOD_HALF = 0.056                                          # shell half length
RING_R, RING_W = 0.0515, 0.006                            # raised end rings
CHAMF_W, BOOT_R = 0.009, 0.036                            # chamfer to the boot, boot core radius
X_CH = MOD_HALF + RING_W                                  # 0.062: chamfer start
X_BOOT = X_CH + CHAMF_W                                   # 0.071: boot start; the joint is at 0.09
metal = standard_material(0xa3a9ae, roughness=0.38, metalness=0.6)
duct_mat = standard_material(0x2a2e33, roughness=0.5, metalness=0.3, side=tp.Side.Double)
ROT_X = (0.0, 0.0, -math.pi / 2)                          # a +Y primitive laid along +X
ROT_MX = (0.0, 0.0, math.pi / 2)                          # ... along -X
ROT_TORUS_X = (0.0, math.pi / 2, 0.0)                     # a torus (axis +Z) turned onto +X


def cyl_x(r0, r1, x0, x1, seg=40, open_=False):
    """Frustum from radius r0 at x0 to r1 at x1, as a merged() item."""
    if x1 < x0:
        return (tp.CylinderGeometry(r0, r1, x0 - x1, seg, 1, open_), (0.5 * (x0 + x1), 0.0, 0.0), ROT_MX)
    return (tp.CylinderGeometry(r1, r0, x1 - x0, seg, 1, open_), (0.5 * (x0 + x1), 0.0, 0.0), ROT_X)


def module_parts(i):
    """(yellow, metal, rubber, duct) merged() item lists for link i, in link-local coordinates."""
    Y, M, R, D = [], [], [], []
    ends = []
    if i > 0:
        ends.append(+1.0)                                 # front end has a joint
    ends.append(-1.0)                                     # rear end: a joint, or the tail cap
    if i > 0:
        Y.append(cyl_x(MOD_R, MOD_R, -MOD_HALF, MOD_HALF))
        R.append(cyl_x(MOD_R + 0.0006, MOD_R + 0.0006, -0.0013, 0.0013))   # the seam
        for k in range(10):                               # bolt ring beside the seam
            a = 2.0 * math.pi * (k + 0.5) / 10
            M.append((tp.CylinderGeometry(0.0034, 0.0034, 0.004, 8, 1), (0.0075, MOD_R * math.cos(a), MOD_R * math.sin(a)), (a, 0.0, 0.0)))
    for sg in ends:
        rr = LINK_R + 0.002 if i == 0 else RING_R
        M.append(cyl_x(rr, rr, sg * MOD_HALF, sg * X_CH))
        if i == LINK_N - 1 and sg < 0:
            continue                                      # the tail cap takes over from the ring
        M.append(cyl_x(rr, BOOT_R + 0.004, sg * X_CH, sg * X_BOOT))
        R.append(cyl_x(BOOT_R, BOOT_R, sg * X_BOOT, sg * 0.5 * LINK_L, seg=24))
        for xr in (0.0765, 0.0855):                       # two ridges per half: four per joint
            R.append((tp.TorusGeometry(BOOT_R + 0.001, 0.0052, 8, 32), (sg * xr, 0.0, 0.0), ROT_TORUS_X))
    if i < LINK_N - 1:
        R.append((tp.SphereGeometry(BOOT_R + 0.0005, 24, 12), (-0.5 * LINK_L, 0.0, 0.0), None))   # boot core
    if i == 6:                                            # sensor pod on the crown (altimeter / DVL stand-in)
        R.append((tp.BoxGeometry(0.056, 0.020, 0.034), (0.0, MOD_R + 0.007, 0.0), None))
        M.append((tp.CylinderGeometry(0.009, 0.009, 0.004, 16, 1), (0.029, MOD_R + 0.007, 0.0), ROT_X))
    return Y, M, R, D


for i in range(LINK_N):
    g = tp.Group()
    if i == 0:                                            # head: shell, rubber collar, smoked dome to the nose
        x_dome = 0.5 * LINK_L - LINK_R * 0.98             # dome centre: its pole is the nose joint
        m = _add(g, _along_x(tp.Mesh(tp.CylinderGeometry(LINK_R, LINK_R, x_dome + MOD_HALF, 40, 1), yellow)))
        m.position.x = 0.5 * (x_dome - MOD_HALF)
        collar_ring = _add(g, _along_x(tp.Mesh(tp.CylinderGeometry(LINK_R + 0.003, LINK_R + 0.003, 0.012, 40, 1), rubber)))
        collar_ring.position.x = x_dome
        dome = _add(g, tp.Mesh(tp.SphereGeometry(LINK_R * 0.98, 32, 20, 0.0, 2 * math.pi, 0.0, 0.5 * math.pi), smoked))
        dome.rotation.z = -math.pi / 2                    # the hemisphere's +Y pole -> +X (forward)
        dome.position.x = x_dome
        bezel = _add(g, _along_x(tp.Mesh(tp.CylinderGeometry(0.017, 0.017, 0.006, 20, 1), ring_mat)))
        bezel.position.x = 0.5 * LINK_L - 0.006
        lens = _add(g, _along_x(tp.Mesh(tp.CylinderGeometry(0.012, 0.012, 0.008, 20, 1), lens_mat)))
        lens.position.x = 0.5 * LINK_L - 0.001
        son = _add(g, tp.Mesh(tp.BoxGeometry(0.05, 0.018, 0.04), rubber))   # sonar head on the crown
        son.position.set(-0.01, LINK_R + 0.004, 0.0)
        lamps = []
        for z in (-0.045, 0.045):                         # two side lamps just behind the dome
            hous = _add(g, _along_x(tp.Mesh(tp.CylinderGeometry(0.014, 0.014, 0.03, 16, 1), rubber)))
            hous.position.set(x_dome - 0.012, -0.012, z)
            disc = tp.Mesh(tp.CircleGeometry(0.011, 20), lamp_disc_mat)
            disc.rotation.y = math.pi / 2
            disc.position.set(x_dome + 0.0035, -0.012, z)
            _add(g, disc, shadow=False)
            spot = tp.SpotLight(tp.Color(1.0, 0.90, 0.74), LAMP_I, 3.5, math.radians(24.0), 0.6, 3.0)
            spot.position.set(x_dome + 0.006, -0.012, z)
            tgt = tp.Group()
            tgt.position.set(3.0, -0.2, z)
            g.add(tgt)
            spot.set_target(tgt)
            g.add(spot)
            lamps.append((spot, tgt))
    Y, M, R, D = module_parts(i)
    for items, mat in ((Y, yellow), (M, metal), (R, rubber), (D, duct_mat)):
        if items:
            _add(g, W.merged(items, mat))
    if i == LINK_N - 1:                                   # tail: tapered cap
        cap = _add(g, tp.Mesh(tp.ConeGeometry(MOD_R, 0.12, 40, 1), yellow))
        cap.rotation.z = math.pi / 2                      # cone apex (+Y) -> -X (backwards)
        cap.position.x = -X_CH - 0.06
        tip = _add(g, tp.Mesh(tp.SphereGeometry(0.012, 12, 8), rubber))
        tip.position.x = -X_CH - 0.118
    W.scene.add(g)
    snake_links.append(g)
HEAD_LAMPS = lamps
for _m in snake_meshes:
    W.renderer.set_instance_id(_m, SON_ID_SNAKE)
W.sonar.reflectivity.set(SON_ID_SNAKE, 0.0)               # the snake's own body never echoes in its sonar


def pose_links(pos, quat):
    for g, p, q in zip(snake_links, pos, quat):
        g.position.set(float(p[0]), float(p[1]), float(p[2]))
        g.quaternion.set(float(q[0]), float(q[1]), float(q[2]), float(q[3]))


# ---- the dock: an open-both-ends cradle on two ropes from the inner collar --------------------
dock = tp.Group()
DOCK_SIDE = np.cross(DOCK_U, [0.0, 1.0, 0.0])
_UP = np.array([0.0, 1.0, 0.0])
dock_grey = standard_material(0x4a525a, roughness=0.5, metalness=0.6)
dock_white = standard_material(0xd8dcdf, roughness=0.55)
dock_orange = standard_material(0xe0641c, roughness=0.55)
led_mat = standard_material(0x103010, roughness=0.3, emissive=0x30ff50, emissive_intensity=40.0)


def dpt(a, s, h):
    """Dock-local (along axis, starboard-ish side, up) -> world."""
    return DOCK_C + a * DOCK_U + s * DOCK_SIDE + h * _UP


def _dock_add(m):
    m.cast_shadow = True
    dock.add(m)
    return m


HOOP_R, RAIL_S, RAIL_H = MS.MOUTH_R, 0.085, -0.072          # funnel mouth 0.35 m: the gait's swept width
for sgn in (-1.0, 1.0):                                    # the two rails the body rests on
    _dock_add(W.tube(dpt(-0.5 * DOCK_LEN, sgn * RAIL_S, RAIL_H), dpt(0.5 * DOCK_LEN, sgn * RAIL_S, RAIL_H), 0.014, dock_grey, 12))
for a in (-0.55, 0.0, 0.55):                               # U-ties under the body
    pts = [dpt(a, s, h) for s, h in ((-RAIL_S, RAIL_H), (-0.07, -0.12), (0.07, -0.12), (RAIL_S, RAIL_H))]
    for p, q in zip(pts[:-1], pts[1:]):
        _dock_add(W.tube(p, q, 0.011, dock_grey, 8))
for a in (-0.5 * DOCK_LEN, 0.5 * DOCK_LEN):                # funnel hoops, both ends open
    sgn = 1.0 if a > 0 else -1.0
    hoop = _dock_add(tp.Mesh(tp.TorusGeometry(HOOP_R, 0.017, 12, 48), dock_orange))
    hoop.position.set(*dpt(a, 0.0, 0.0))
    # torus lies in XY with its axis on +Z: turn +Z onto the dock axis
    zaxis = np.array([0.0, 0.0, 1.0])
    ax = np.cross(zaxis, DOCK_U)
    hoop.quaternion.set_from_axis_angle(tp.Vector3(*(ax / np.linalg.norm(ax))), math.acos(float(zaxis @ DOCK_U)))
    fun = tp.Mesh(tp.CylinderGeometry(HOOP_R, MS.BORE_R, 0.25, 32, 1, True), standard_material(0xc9cdd0, roughness=0.6, side=tp.Side.Double))
    fun.position.set(*dpt(a - sgn * 0.125, 0.0, 0.0))
    W.align_y(fun, sgn * DOCK_U)                           # wide end outward
    _dock_add(fun)
    for s in (-1.0, 1.0):                                  # hoop to rail struts
        _dock_add(W.tube(dpt(a, s * RAIL_S, RAIL_H), dpt(a, s * 0.707 * HOOP_R, -0.707 * HOOP_R), 0.010, dock_grey, 8))
# latch post with its LED, mid-cradle on the starboard side
_dock_add(W.tube(dpt(0.0, RAIL_S + 0.03, RAIL_H), dpt(0.0, RAIL_S + 0.03, 0.10), 0.013, dock_grey, 10))
latch = _dock_add(tp.Mesh(tp.BoxGeometry(0.05, 0.03, 0.05), dock_white))
latch.position.set(*dpt(0.0, RAIL_S + 0.03, 0.10))
led = tp.Mesh(tp.SphereGeometry(0.011, 12, 8), led_mat)
led.position.set(*dpt(0.0, RAIL_S + 0.03, 0.125))
dock.add(led)
# spreader bar over the cradle and the two ropes up to the inner collar
_dock_add(W.tube(dpt(-0.5 * DOCK_LEN, 0.0, HOOP_R + 0.02), dpt(0.5 * DOCK_LEN, 0.0, HOOP_R + 0.02), 0.012, dock_grey, 8))
for a in (-0.5 * DOCK_LEN, 0.5 * DOCK_LEN):
    top = dpt(a, 0.0, HOOP_R + 0.02)
    th = math.atan2(top[2], top[0])
    anchor = np.array([(W.PEN_R - 0.05) * math.cos(th), W.WATER_Y - 0.12, (W.PEN_R - 0.05) * math.sin(th)])
    _dock_add(W.tube(top, anchor, 0.009, W.rope_mat, 6))
    _dock_add(W.tube(dpt(a, 0.0, 0.0) + HOOP_R * _UP, top, 0.010, dock_grey, 8))
W.scene.add(dock)


def led_update(phase):
    led_mat.emissive = 0x30ff50 if phase == "DOCKED" else 0xffa020


# ---- geometry-check overlay (snake_top only) ----------------------------------------------------
path_line = None


def build_path_line():
    global path_line
    pts = PATH_P[::4]
    v, n = W.tube_verts(pts, 0.05, 6)
    geo = tp.BufferGeometry()
    geo.set_attribute("position", v)
    geo.set_attribute("normal", n)
    geo.set_index(W.tube_index(len(pts), 6))
    path_line = tp.Mesh(geo, standard_material(0x00ffff, roughness=0.9, emissive=0x00e0ff, emissive_intensity=6.0))
    path_line.frustum_culled = False
    W.scene.add(path_line)
    mark = tp.Mesh(tp.SphereGeometry(0.25, 16, 10), standard_material(0xff2020, roughness=0.9, emissive=0xff2020, emissive_intensity=8.0))
    mark.position.set(*W.TEAR_C)
    W.scene.add(mark)
    return path_line, mark


# ---- retire the ROV ------------------------------------------------------------------------------
W.rov_yaw.visible = False
for _k, _l in enumerate(W.rov_lights):
    if _k % 2:
        _l.intensity = 0.0
    else:
        _l.emissive_intensity = 0.0
W.tether.visible = False
W.bubbles.visible = False
W.bubbles.set_live_count(0)
W.props = []                                              # step() spins every entry: nothing left to spin
W.rov_thrust = np.zeros(3)
_PARK = (np.array([W.PEN_R - 0.4, W.WATER_Y - 0.3, 0.0]), np.array([W.PEN_R - 0.4, W.WATER_Y - 0.3, 0.6]))


def _no_bubbles(dt):
    return None


def _no_tether_upload():
    return None


def _parked_tether(dt):
    return _PARK                                          # the hidden rope hangs parked at the collar


W.bubbles_step = _no_bubbles
W.tether_upload = _no_tether_upload
W.tether_pins = _parked_tether
W._hud_a[1].set_text("SNAKE HEAD CAM")

# ---- --seed: jitter the current heading --------------------------------------------------------
_current0 = W.current_at
CUR_DH = float(np.random.default_rng(SEED).normal(0.0, 0.15)) if SEED else 0.0


def current_at(t):
    c = _current0(t)
    if CUR_DH == 0.0:
        return c
    ch, sh = math.cos(CUR_DH), math.sin(CUR_DH)
    return np.array([c[0] * ch - c[2] * sh, 0.0, c[0] * sh + c[2] * ch], np.float32)


W.current_at = current_at

# ---- hooks: the snake's head is the "ROV" for every consumer in warp_netpen ---------------------
SNAKE = [KinematicSnake()]                                # run_mission / the window swap in PhysicsSnake
MISSION_T0 = [0.0]                                        # mission clock = W.world_t + MISSION_T0
STATS = {"min_d": float("inf")}


def mission_t():
    return W.world_t + MISSION_T0[0]


def wall_side(p, fwd):
    """Unit vector perpendicular to fwd, horizontal, pointing at the net (outboard)."""
    out = np.array([p[0], 0.0, p[2]])
    out = out - fwd * float(out @ fwd)
    n = np.linalg.norm(out)
    return out / n if n > 1e-6 else np.cross(fwd, _UP)


def snake_rov_pose(t, dt=1.0 / 60.0):
    sn = SNAKE[0]
    pos, quat = sn.poses(mission_t())
    pose_links(pos, quat)
    p, R = sn.head_pose()
    W.rov_pos, W.rov_R, W.rov_thrust = p, R, np.zeros(3)
    q = net_positions()
    d2 = ((q[None, :, :] - pos[:, None, :]) ** 2).sum(-1)
    d = float(np.sqrt(d2.min())) - LINK_R
    W.ROV_STATS["d"] = d
    STATS["min_d"] = min(STATS["min_d"], d)
    W.ROV_STATS["min_d"] = STATS["min_d"]
    # lamps: aim with the head camera, forward and 45 deg to the wall side, in the head frame
    look = R.T @ (math.cos(math.radians(45.0)) * R[:, 0] + math.sin(math.radians(45.0)) * wall_side(p, R[:, 0]))
    for k, (spot, tgt) in enumerate(HEAD_LAMPS):
        tgt.position.set(*(0.055 * np.array([1.0, 0.0, 0.0]) + 3.0 * look + [0.0, -0.15, (k - 0.5) * 0.3]))
    led_update(sn.phase)


SONAR_IN = []                                             # (intensity, sonar pos, sonar yaw) for the mission
SON_POSE = [None]


def snake_sonar_aim():
    sn = SNAKE[0]
    p, R = sn.head_pose()
    o = sn.nose() - 0.05 * R[:, 0] + (LINK_R + 0.015) * R[:, 1]
    SON_POSE[0] = (o.copy(), math.atan2(-float(R[2, 0]), float(R[0, 0])))
    Rs = R @ W.rot_z(math.radians(W.SON_TILT))
    fwd, up = Rs[:, 0], Rs[:, 1]
    W.sonar.position.set(float(o[0]), float(o[1]), float(o[2]))
    W.sonar.up = tp.Vector3(float(up[0]), float(up[1]), float(up[2]))
    W.sonar.look_at(float(o[0] + fwd[0]), float(o[1] + fwd[1]), float(o[2] + fwd[2]))


def snake_cam_place():
    sn = SNAKE[0]
    p, R = sn.head_pose()
    o = sn.nose() + 0.02 * R[:, 0]
    look = math.cos(math.radians(45.0)) * R[:, 0] + math.sin(math.radians(45.0)) * wall_side(p, R[:, 0])
    W.rov_cam.position.set(*o)
    W.rov_cam.look_at(*(o + look))


def snake_fish_step(t, dt):
    cp = W.camera.position
    W.school.step(t, dt, SNAKE[0].mid_point(), np.array([cp.x, cp.y, cp.z]))   # one point, 2 m falloff: the body
    if W.fish_vk is not None:
        return
    W.school.skin()
    W.fish_geo.update_attribute("position", W.school.out_p.numpy())
    W.fish_geo.update_attribute("normal", W.school.out_n.numpy())


_sonar_draw0 = W.sonar_draw


SONAR_DUMP = []                                           # (t, phase, image f16, sonar pos, sonar yaw)
DUMP_PHASES = ("ACQUIRE", "FOLLOW_WALL", "TEAR", "INSPECT", "UTURN", "INSPECT_2")


def snake_sonar_draw(frame_i):
    """The sonar's image as it lands (the same one the inset draws), with the pose it was fired from."""
    if SON_POSE[0] is not None:
        img = W.son_hist[(frame_i // W.SON_EVERY) % 3].copy()
        SONAR_IN.append((img, *SON_POSE[0]))
        del SONAR_IN[:-2]
        ph = getattr(SNAKE[0], "phase", "")
        if DUMP_SONAR and ph in DUMP_PHASES and len(SONAR_DUMP) < 600:
            SONAR_DUMP.append((mission_t(), ph, img.astype(np.float16), SON_POSE[0][0], SON_POSE[0][1]))
    _sonar_draw0(frame_i)


W.sonar_draw = snake_sonar_draw
W.rov_pose = snake_rov_pose
W.sonar_aim = snake_sonar_aim
W.rov_cam_place = snake_cam_place
W.fish_step = snake_fish_step

# ---- HUD: the mission phase over the camera inset ----------------------------------------------
phase_lab = tp.TextSprite(W.FONT, world_scale=18.0)
phase_lab.set_text("DOCKED")
phase_lab.set_color(0xffd23c)
phase_lab.screen_space = True
phase_lab.screen_anchor.set(0.0, 0.0)
phase_lab.center.set(0.0, 0.0)
phase_lab.position.set(W.HUD_M, W.HUD_M + W.CAM_H + 34, 0.0)
W.scene.add(phase_lab)


def hud_phase():
    sn = SNAKE[0]
    txt = f"MISSION {sn.phase}   t {mission_t():5.1f} s   {sn.speed:.2f} m/s   net {W.ROV_STATS.get('d', 0.0):.2f} m"
    phase_lab.set_text(txt)
    phase_lab.position.x = W.HUD_M if W.HUD_LIVE[0] else -9000.0


def step():
    W.step()
    hud_phase()


# ---- stills ---------------------------------------------------------------------------------------
def aim(eye, tgt, fov):
    cam = W.camera
    cam.fov = fov
    cam.update_projection_matrix()
    cam.position.set(*W.net_clear(np.asarray(eye, np.float64)))
    cam.look_at(*np.asarray(tgt, np.float64))


def mean_fwd(sn):
    f = sn.pos[0] - sn.pos[-1]
    f[1] = 0.0
    return f / max(np.linalg.norm(f), 1e-9)


def place_shot(name):
    sn = SNAKE[0]
    mid, hp = sn.mid_point(), sn.head_pose()[0]
    fwd = mean_fwd(sn)
    inb = -np.array([mid[0], 0.0, mid[2]]) / math.hypot(mid[0], mid[2])
    if name == "snake_dock":
        aim(DOCK_C - 1.9 * e_r(DOCK_BEARING) + 0.75 * DOCK_U + [0, 0.55, 0], DOCK_C + 0.15 * DOCK_U + [0, -0.05, 0], 55.0)
    elif name == "snake_school":
        aim(mid - 2.1 * fwd + 1.2 * inb + [0, 1.5, 0], mid + 0.5 * fwd + [0, -0.1, 0], 58.0)
    elif name == "snake_tear":
        tc, tn = W.tear_now(), W.tear_normal()
        side = np.cross(_UP, tn)
        side *= 1.0 if float(side @ (hp - tc)) > 0 else -1.0
        aim(tc - 3.3 * tn + 0.9 * side + [0, 0.55, 0], 0.55 * tc + 0.45 * mid, 52.0)
    elif name == "snake_hud":
        aim(mid + 2.4 * inb + 0.3 * fwd + [0, 0.9, 0], mid + 0.25 * fwd + [0, -0.5, 0], 60.0)
    elif name == "snake_top":
        W.camera.up = tp.Vector3(0.0, 0.0, -1.0)
        W.camera.fov = 60.0
        W.camera.update_projection_matrix()
        W.camera.position.set(0.0, 17.0, 0.0)
        W.camera.look_at(0.0, DEPTH_Y, 0.0)


def shot_mission_time(name):
    if name == "snake_dock":
        return 3.0
    if name == "snake_school":
        return mission_time_head_at_bearing(DOCK_BEARING - D2R(36.0), T_DOCKED)
    if name == "snake_tear":
        return mission_time_head_at_bearing(TEAR_BEARING - D2R(4.0), T_DOCKED)
    if name == "snake_hud":
        return mission_time_head_at_bearing(TEAR_BEARING + D2R(26.0), T_DOCKED)
    return mission_time_head_at_bearing(TEAR_BEARING - D2R(8.0), T_DOCKED)


def mission_sweep(rate=15.0):
    """The whole kinematic mission against the cloth as it is now (no scene stepping): the joint
    angles it asks of the body, the closest any link comes to the net, the wall push it needed."""
    sn = KinematicSnake()
    q = net_positions()
    worst, dmin, offmax, where = 0.0, float("inf"), 0.0, ""
    for t in np.arange(0.0, T_MISSION + 1e-9, 1.0 / rate):
        pos, _ = sn.poses(float(t))
        fw = sn.R[:, :, 0]
        ang = float(np.degrees(np.arccos(np.clip((fw[:-1] * fw[1:]).sum(1), -1.0, 1.0))).max())
        if ang > worst:
            worst, where = ang, f"{sn.phase} t {t:.1f} s"
        d = float(np.sqrt(((q[None] - pos[:, None]) ** 2).sum(-1).min())) - LINK_R
        dmin, offmax = min(dmin, d), max(offmax, sn.off)
    print(f"mission sweep ({T_MISSION:.0f} s at {rate:.0f} Hz, cloth frozen at scene t {W.world_t:.1f} s): "
          f"max joint {worst:.1f} deg ({where}), min link-to-net {dmin:.2f} m, max wall push {offmax:.2f} m")


def run_shots(names, out_for):
    warm = int(SECONDS * 60)
    live0 = W.HUD_LIVE[0]
    W.hud_park(False)
    t0 = time.perf_counter()
    for _ in range(warm):                                 # the net bows, the school mills; nothing to look at yet
        step()
    print(f"warm-up: {warm} scene frames in {time.perf_counter() - t0:.1f} s (no render)")
    mission_sweep()
    render_ms = []
    for name in names:
        if name == "snake_top":
            build_path_line()
            if hasattr(W, "ocean"):
                W.ocean.visible = False
            W.renderer.set_underwater_murk(0.002, W.MURK_COLOR)
        W.hud_park(name == "snake_hud" and live0)
        n_roll = 150
        t_save = shot_mission_time(name)
        SNAKE[0] = KinematicSnake()
        MISSION_T0[0] = t_save - (W.world_t + n_roll / 60.0)
        for k in range(int(20 * 15), 0, -1):              # the kinematic's wall-push history, before the roll
            SNAKE[0].poses(max(t_save - n_roll / 60.0 - k / 15.0, 0.0))
        STATS["min_d"] = float("inf")
        tb = None
        for i in range(n_roll):
            if i == 30:
                tb = time.perf_counter()
            step()
            place_shot(name)
            W.renderer.render(W.scene, W.camera)
        render_ms.append(1e3 * (time.perf_counter() - tb) / (n_roll - 30))
        out = out_for(name)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        W.renderer.save_frame(W.scene, W.camera, out)
        sn = SNAKE[0]
        print(f"{name}: mission t {mission_t():.1f} s, phase {sn.phase}, head {np.round(sn.head_pose()[0], 2)}, "
              f"net {W.ROV_STATS.get('d', float('nan')):.2f} m (min over roll {STATS['min_d']:.2f}), "
              f"wall offset {sn.off:.2f} m, max joint {math.degrees(sn.max_joint):.0f} deg, {render_ms[-1]:.1f} ms/f -> {out}")
    print(f"frame path: {np.mean(render_ms):.1f} ms/f mean over {len(names)} shots at {SIZE} (step + render)")


MISSION_SHOT_PHASES = {"UNDOCK": ("undock", 3.0), "FOLLOW_WALL": ("follow_wall", 4.0), "TEAR": ("tear_detected", 0.5),
                       "INSPECT": ("inspect", 4.0), "UTURN": ("uturn", 3.0), "INSPECT_2": ("inspect_2", 3.0),
                       "APPROACH": ("approach", 3.0), "TURN_IN": ("turn_in", 2.0), "FINAL": ("final", 4.0),
                       "CAPTURE": ("capture", 2.0), "DOCKED": ("docked", 3.0)}


def place_mission_shot(name):
    sn = SNAKE[0]
    mid = sn.mid_point()
    fwd = mean_fwd(sn)
    inb = -np.array([mid[0], 0.0, mid[2]]) / math.hypot(mid[0], mid[2])
    if name in ("undock", "approach", "turn_in", "final", "capture", "docked"):
        aim(mid - 1.2 * fwd + 2.0 * inb + [0, 0.9, 0], mid + 0.2 * fwd + [0, -0.1, 0], 58.0)
    elif name in ("tear_detected", "inspect"):
        tc = W.tear_now()
        aim(mid + 2.6 * inb - 0.6 * fwd + [0, 0.8, 0], 0.5 * tc + 0.5 * mid, 58.0)
    else:
        aim(mid + 2.4 * inb + 0.3 * fwd + [0, 0.9, 0], mid + 0.25 * fwd + [0, -0.3, 0], 60.0)


def _net_ring():
    q = net_positions()
    sel = np.abs(q[:, 1] - DEPTH_Y) < 0.06
    return q[sel][:, [0, 2]].astype(np.float32)


def run_mission(shots=False):
    """Headless: the physics snake flies the mission from DOCKED to DOCKED (or the time cap) on the
    scene's own sonar; telemetry every frame, the summary, the top-down plot; stills on request.
    The main view renders every --render-every frames (the sonar's trace rides on the renderer) and
    every frame of a still's 40-frame lead-in."""
    import json
    import snake_mission as MS
    out_dir = OUT or "D:/dev/snake_out/mission"
    os.makedirs(out_dir, exist_ok=True)
    W.renderer.set_flush_frames(1)
    W.hud_park(True)                                      # the sonar only runs while the HUD is live
    prov = PhysicsSnake(SEED)
    SNAKE[0] = prov
    warm = int(SECONDS * 60)
    MISSION_T0[0] = -(W.world_t + warm / 60.0)
    aim(DOCK_C - 3.0 * e_r(DOCK_BEARING) + [0, 1.5, 0], DOCK_C, 60.0)
    t0 = time.perf_counter()
    for i in range(warm):                                 # the net settles, the snake sits in the cradle
        step()
        if i % RENDER_EVERY == 0:
            W.renderer.render(W.scene, W.camera)
    print(f"mission: warm-up {warm} frames in {time.perf_counter() - t0:.1f} s", flush=True)
    cols = ["t", "phase", "gait", "phi0", "psi_mean", "psi_head", "dpsi_head", "speed", "p_abs", "p_net", "e_abs",
            "e_net", "meas", "d_head_true", "d_min_links", "tangent", "psi_wall", "wall_seen", "tear_fired",
            "tear_bearing", "cand_bearing", "cur_x", "cur_z", "com_x", "com_y", "com_z", "head_x", "head_y", "head_z",
            "los_s", "los_e", "latch_f", "images", "peak_torque", "guide_f", "tear_x", "tear_z", "d_head_tear"]
    rows, links, quats = [], [], []
    shots_todo, shot_paths, shot_names = [], [], set()
    wall0 = time.perf_counter()
    f, last_phase = 0, "DOCKED"
    gait_ids = {g: k for k, g in enumerate(MS.GAITS)}
    while True:
        step()
        t = mission_t()
        ms = prov.mission
        if ms.phase != last_phase:
            print(f"  t {t:6.1f} s  {last_phase} -> {ms.phase}  head {np.round(prov.pos[0], 2)}  "
                  f"net {W.ROV_STATS.get('d', 0):.2f} m", flush=True)
            if shots and ms.phase in MISSION_SHOT_PHASES and MISSION_SHOT_PHASES[ms.phase][0] not in shot_names:
                nm, lag = MISSION_SHOT_PHASES[ms.phase]
                shot_names.add(nm)
                shots_todo.append((f + int(lag * 60), nm))
            last_phase = ms.phase
        pend = [s for s in shots_todo if s[0] - 40 <= f]
        if pend:
            place_mission_shot(pend[0][1])
        elif f % RENDER_EVERY == 0:
            aim(prov.pos[4] + np.array([0.0, 6.0, 0.01]), prov.pos[4], 60.0)
        if pend or f % RENDER_EVERY == 0:
            W.renderer.render(W.scene, W.camera)
        for s in [s for s in shots_todo if s[0] <= f]:
            p = os.path.join(out_dir, f"seed{SEED}_{s[1]}{TAG}.png")
            W.renderer.save_frame(W.scene, W.camera, p)
            shot_paths.append(p)
            shots_todo.remove(s)
            print(f"  still {p}  (t {t:.1f} s, {ms.phase})", flush=True)
        hp = prov.pos[0]
        tc = W.tear_now()
        qn = net_positions()
        d_head = float(np.sqrt(((qn - hp) ** 2).sum(1).min()))
        sn = prov.snake
        cur = W.current_at(W.world_t)
        est = ms.est
        cand = ms.det.last[0] if ms.det.last is not None else float("nan")
        st = prov.st
        rows.append([t, MS.PHASES.index(ms.phase), gait_ids.get(ms.gait, -1), ms.phi0_cmd, st.yaw_mean, st.head_yaw,
                     st.dpsi_head, st.speed, sn.power_abs, sn.power_net, sn.energy_abs, sn.energy_net, ms.follow.meas,
                     d_head, W.ROV_STATS.get("d", float("nan")), est.tangent, ms.follow.psi_wall, float(est.seen),
                     float(ms.det.fired), ms.det.bearing, cand, float(cur[0]), float(cur[2]), *st.com, *hp,
                     ms.los.get("s", float("nan")), ms.los.get("e", float("nan")),
                     float(np.linalg.norm(prov.ad.latch_force)), prov.images, sn.peak_torque, prov.ad.guide_force,
                     float(tc[0]), float(tc[2]), float(np.linalg.norm(prov.nose() - tc))])
        links.append(prov.pos.astype(np.float32))
        quats.append(prov.quat.astype(np.float32))
        f += 1
        if (ms.done and not shots_todo) or t >= T_CAP:
            break
        if f % 600 == 0:
            print(f"  t {t:6.1f} s  {ms.phase:11s} {(time.perf_counter() - wall0) / f * 1e3:.1f} ms/f  "
                  f"images {prov.images}  meas {ms.follow.meas:.2f}  head-net {d_head:.2f}", flush=True)
    wall = time.perf_counter() - wall0
    L = np.asarray(rows, np.float64)
    stem = os.path.join(out_dir, f"seed{SEED}{TAG}")
    tel = dict(cols=np.array(cols), log=L, links=np.asarray(links), quats=np.asarray(quats),
               events=np.array([f"{a:.2f} {b}" for a, b in prov.mission.events]),
               tear_true=np.float64(TEAR_BEARING), dock_c=MS.DOCK_C, dock_u=MS.DOCK_U, latch_p=MS.LATCH_P,
               gaits=np.array(list(MS.GAITS)), phases=np.array(MS.PHASES), net_y3=_net_ring(), seed=np.int64(SEED))
    np.savez_compressed(stem + "_telemetry.npz", **tel)
    if DUMP_SONAR and SONAR_DUMP:
        np.savez_compressed(stem + "_sonar_dump.npz", t=np.array([d[0] for d in SONAR_DUMP]),
                            phase=np.array([d[1] for d in SONAR_DUMP]), img=np.stack([d[2] for d in SONAR_DUMP]),
                            pos=np.stack([d[3] for d in SONAR_DUMP]), yaw=np.array([d[4] for d in SONAR_DUMP]),
                            net_y3=_net_ring())
        print(f"mission: {len(SONAR_DUMP)} sonar images -> {stem}_sonar_dump.npz")
    summ = MS.summarize(tel, prov.mission, prov.ad, mass=float(prov.snake.mass.sum()))
    summ.update(seed=SEED, frames=f, wall_s=round(wall, 1), ms_per_frame=round(wall / max(f, 1) * 1e3, 1),
                render_every=RENDER_EVERY, size=SIZE, max_vertical_drift_m=round(prov.max_dy, 4),
                current_heading_jitter_rad=CUR_DH, stills=shot_paths)
    with open(stem + ".json", "w") as fh:
        json.dump(summ, fh, indent=1)
    MS.plot_top(tel, stem + "_top.png")
    print(json.dumps(summ, indent=1))
    print(f"mission: {f} frames in {wall:.0f} s ({wall / max(f, 1) * 1e3:.1f} ms/f) -> {stem}_telemetry.npz, .json, _top.png")


def main():
    print(f"snake: path {PATH_S[-1]:.1f} m, mission {T_MISSION:.1f} s, dock bearing {math.degrees(DOCK_BEARING):.1f} deg, "
          f"tear bearing {math.degrees(TEAR_BEARING):.1f} deg ({math.degrees(DOCK_BEARING - TEAR_BEARING):.1f} deg of arc), "
          f"seed {SEED} (current heading {CUR_DH:+.3f} rad)")
    print("snake: waypoints " + ", ".join(f"{k} s={v:.2f} t={mission_time_of(k):.1f}" for k, v in PATH_KEYS.items()))
    if MISSION or MISSION_SHOTS:
        run_mission(shots=MISSION_SHOTS)
        return
    if HEADLESS:
        if len(SHOT_LIST) > 1:
            d = OUT or OUT_DIR_DEFAULT
            run_shots(SHOT_LIST, lambda n: os.path.join(d, f"{n}{TAG}.png"))
        else:
            run_shots(SHOT_LIST, lambda n: OUT or os.path.join(OUT_DIR_DEFAULT, f"{n}{TAG}.png"))
        print(f"snake: max joint angle over the rolls {math.degrees(SNAKE[0].max_joint):.1f} deg")
        return
    if not KINEMATIC:
        SNAKE[0] = PhysicsSnake(SEED)
        MISSION_T0[0] = -W.world_t
    aim(DOCK_C - 2.4 * e_r(DOCK_BEARING) + 0.9 * DOCK_U + [0, 0.7, 0], DOCK_C, 60.0)
    W.orbit_loop(W.canvas, W.renderer, W.scene, W.camera, step, target=tuple(DOCK_C))


if __name__ == "__main__":
    main()
