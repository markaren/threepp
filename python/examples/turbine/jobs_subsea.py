"""Phase 3B: the underwater jobs. The tethered ROV flies an inspection pass around the anode cage,
the thruster snake finds the abrasion where the cable sleeve first lies on the rock, and the two
work one site together (the ROV lights it, the snake reads it).

    python jobs_subsea.py --jobs snake,rov,pair [--outdir <dir>]

Importable for the film: build_subsea(renderer, S, F) returns J, then per frame
    J.snake_job(tau)     poses the snake (film time tau, 0..SNAKE_T), its lamps and head camera
    J.rov_job(tau, dt)   steps the ROV carrot follower (call once per frame, in order)
    J.pair(tau)          the ROV parked over the work site, lamps on the patch
    J.tether(top, tail)  the hanging tether from the Mariner's moonpool to the ROV
Headless only. Frame: X upwind, Y up, origin on the pile axis at MSL.
"""
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import threepp as tp
from demo_common import Encoder, cli_arg, standard_material

import turbine_site as ts
import turbine_fleet as tf
import fleet_vehicles as fv
from usv_rig import quat_of

UP = fv.UP
_unit = fv._unit
FPS = 30
L = fv.LINK_L
SNAKE_T, ROV_T, PAIR_T = 14.0, 12.0, 6.0
SKIN_GAP = 0.62                    # snake skin to sleeve skin while it travels
NOSE_GAP = 0.52                    # nose to the wear patch at the hold
PATCH_HALF = 0.21                  # the worn length is 2 x this along the sleeve
SNAKE_RAD = fv.LINK_R + 0.03       # shell + side ducts
ROV_V, ROV_A, ROV_YAW = 0.45, 0.30, math.radians(25.0)
CAGE = ts.spec["underwater"]["anode_cage"]
CAGE_FACE = CAGE["ring_radius"] + 0.5 * CAGE["anode_size"][0]
ROV_STANDOFF = 2.0
TETHER_SLACK = 1.06
TETHER_R = 0.005                   # 10 mm neutrally buoyant tether, yellow (Fathom is 7.6 mm)
ROV_FLOOD_I = 3.0
ROV_JOB_I = 38.0                   # v02 at 60 took the anode in the beam near white
TETHER_LOCAL = np.array([-0.19, 0.11, 0.0])      # warp_netpen: tail, top


def smooth(e0, e1, x):
    u = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def min_jerk(u):
    u = np.clip(u, 0.0, 1.0)
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def slerp_dir(a, b, w):
    a, b = _unit(a), _unit(b)
    ang = math.acos(float(np.clip(np.dot(a, b), -1.0, 1.0)))
    if ang < 1e-6:
        return a
    return _unit(math.sin((1 - w) * ang) * a + math.sin(w * ang) * b)


# --------------------------------------------------------------------------- #
#  The sleeve: arc length, tangent, the touchdown
# --------------------------------------------------------------------------- #
class Sleeve:
    def __init__(self, S):
        self.P = np.asarray(S.cable_path, float)
        self.s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.P, axis=0), axis=1))])
        T = np.gradient(self.P, axis=0)
        self.T = T / np.linalg.norm(T, axis=1)[:, None]
        az = math.radians(S.spec["empties"]["cable_entry"]["azimuth_deg"])
        rn = np.array([math.cos(az), 0.0, math.sin(az)])
        r = self.P @ rn
        gap = self.P[:, 1] - ts.CPS_R - ts.seabed_height(self.P[:, 0], self.P[:, 2])
        self.i_td = int(np.argmax((r > r[0] + 2.0) & (gap < 0.05)))      # fleet_vehicles.snake_station
        self.s_td = float(self.s[self.i_td])

    def at(self, s):
        s = np.clip(s, 0.0, self.s[-1])
        p = np.stack([np.interp(s, self.s, self.P[:, k]) for k in range(3)], -1)
        t = np.stack([np.interp(s, self.s, self.T[:, k]) for k in range(3)], -1)
        return p, t / np.linalg.norm(t, axis=-1, keepdims=True)

    def radius(self, s):
        return ts.cps_radius(np.clip(s, 0.0, self.s[-1]))

    def nearest(self, pts):
        d = np.linalg.norm(pts[:, None, :] - self.P[None, :, :], axis=2)
        i = np.argmin(d, axis=1)
        return d[np.arange(len(pts)), i], self.s[i]


def perp(v, t):
    return _unit(v - np.dot(v, t) * t)


# --------------------------------------------------------------------------- #
#  The wear patch: jacket worn through at the touchdown, scrape lines, chips on the rock
# --------------------------------------------------------------------------- #
def _hash2(a, b):
    return np.modf(np.sin(a * 127.1 + b * 311.7) * 43758.5453)[0] % 1.0


def build_wear(scene, SL, out):
    """A polar patch on the sleeve: worn-through core (pale liner, scrape lines along the sleeve),
    a torn rim of fresh jacket, a grime halo fading into the fouled jacket. Edges are smooth."""
    s0 = SL.s_td
    c, t = SL.at(s0)
    pdir = perp(out + 0.3 * UP, t)                                     # the flank that faces the snake
    nr, nf, amax = 48, 220, math.radians(60.0)
    rho = np.linspace(0.0, 1.35, nr)                                   # 1 = the torn edge, beyond: grime halo
    phi = np.linspace(0.0, 2 * math.pi, nf)
    RHO, PHI = np.meshgrid(rho, phi, indexing="ij")
    edge = 1.0 + 0.16 * np.sin(3 * PHI + 1.0) + 0.09 * np.sin(7 * PHI + 2.3) + 0.05 * np.sin(13 * PHI) + 0.03 * np.sin(29 * PHI + 0.7)
    u = RHO * edge * np.cos(PHI)
    v = RHO * edge * np.sin(PHI) * 0.8
    ss = s0 + u * PATCH_HALF
    TH = v * amax
    C, T = SL.at(ss.reshape(-1))
    C, T = C.reshape(nr, nf, 3), T.reshape(nr, nf, 3)
    D = pdir - np.sum(pdir * T, axis=-1, keepdims=True) * T
    D /= np.linalg.norm(D, axis=-1, keepdims=True)
    E = np.cross(T, D)
    dirs = np.cos(TH)[..., None] * D + np.sin(TH)[..., None] * E
    R = SL.radius(ss) + 0.006 + 0.002 * smooth(1.0, 0.9, RHO)         # above the 18-gon's flats and the collars
    pos = C + R[..., None] * dirs
    fouled = np.array([0.22, 0.13, 0.04])                              # the site's CPS colour
    liner = np.array([0.23, 0.225, 0.20])                               # worn through: pale grey liner
    jacket = np.array([0.40, 0.21, 0.06])                              # fresh jacket at the torn rim
    grime = np.array([0.09, 0.065, 0.035])
    scr = np.abs(np.sin(TH * 55.0 + 1.3 * np.sin(u * 4.0))) ** 10      # scrape lines along the sleeve
    fine = 0.5 + 0.5 * np.sin(TH * 180.0 + 3.0 * u)
    core = smooth(0.95, 0.80, RHO)
    rim = smooth(0.80, 0.93, RHO) * smooth(1.05, 0.97, RHO)
    halo = smooth(1.0, 1.08, RHO) * smooth(1.35, 1.12, RHO)
    col = core[..., None] * liner * (1.0 - 0.6 * scr - 0.15 * fine)[..., None]
    col = col + rim[..., None] * jacket
    w_out = np.clip(1.0 - core - rim, 0.0, 1.0)
    col = col + w_out[..., None] * (halo[..., None] * grime + (1.0 - halo[..., None]) * fouled)
    col = col * (0.9 + 0.2 * _hash2(ss * 40.0, TH * 13.0))[..., None]
    P = pos.reshape(-1, 3).astype(np.float32)
    N = dirs.reshape(-1, 3).astype(np.float32)
    i, j = np.meshgrid(np.arange(nr - 1), np.arange(nf - 1), indexing="ij")
    a, b = (i * nf + j).ravel(), ((i + 1) * nf + j).ravel()
    cc, d = ((i + 1) * nf + j + 1).ravel(), (i * nf + j + 1).ravel()
    tri = np.concatenate([np.stack([a, b, cc], 1), np.stack([a, cc, d], 1)])
    n_t = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]])
    flip = np.sum(n_t * N[tri[:, 0]], axis=1) < 0.0                    # every face outward
    tri[flip] = tri[flip][:, [0, 2, 1]]
    geo = tp.BufferGeometry()
    geo.set_attribute("position", P)
    geo.set_attribute("normal", N)
    geo.set_attribute("color", col.reshape(-1, 3).astype(np.float32))
    geo.set_index(tri.reshape(-1).astype(np.uint32))
    mat = tp.MeshStandardMaterial()
    mat.vertex_colors = True
    mat.roughness = 0.9
    patch = tp.Mesh(geo, mat)
    patch.receive_shadow = True
    scene.add(patch)
    # chips of jacket on the rock below the rim
    chips = tp.Group()
    cm = standard_material(0x2e1c0c, roughness=0.95)       # dull fouled flakes
    rng = np.random.default_rng(7)
    for k in range(4):
        a = rng.uniform(-0.5, 0.5)
        q = c + perp(out, t) * (SL.radius(s0) + rng.uniform(0.08, 0.35)) + t * a * 0.6
        q[1] = float(ts.seabed_height(q[0], q[2]))
        sz = rng.uniform(0.015, 0.03)
        ch = tp.Mesh(tp.BoxGeometry(sz, 0.006, sz * rng.uniform(0.5, 1.2)), cm)
        ch.position.set(float(q[0]), float(q[1]) + 0.004, float(q[2]))
        ch.rotation.set(rng.uniform(-0.4, 0.4), rng.uniform(0, 6.28), rng.uniform(-0.4, 0.4))
        chips.add(ch)
    scene.add(chips)
    look = c + SL.radius(s0) * pdir
    return patch, chips, look, pdir


# --------------------------------------------------------------------------- #
#  Tether: a hanging chain with slack
# --------------------------------------------------------------------------- #
def catenary(a_pt, b_pt, slack=TETHER_SLACK, n=160):
    """Points of a hanging chain from a_pt to b_pt, length slack x the straight run."""
    a_pt, b_pt = np.asarray(a_pt, float), np.asarray(b_pt, float)
    dh = (b_pt - a_pt) * np.array([1.0, 0.0, 1.0])
    h = float(np.linalg.norm(dh))
    v = float(b_pt[1] - a_pt[1])
    Lc = slack * float(np.linalg.norm(b_pt - a_pt))
    u = np.linspace(0.0, 1.0, n)
    if h < 1e-3:
        return a_pt + u[:, None] * (b_pt - a_pt)
    k = math.sqrt(max(Lc * Lc - v * v, h * h * 1.0001))
    lo, hi = 1e-3, 1e6                                                 # solve 2 a sinh(h / 2a) = k for a
    for _ in range(100):
        mid = math.sqrt(lo * hi)
        if 2 * mid * math.sinh(min(h / (2 * mid), 300.0)) > k:
            lo = mid
        else:
            hi = mid
    a = math.sqrt(lo * hi)
    x0 = 0.5 * h - a * math.asinh(v / (2 * a * math.sinh(h / (2 * a))))
    x = u * h
    y = a * np.cosh((x - x0) / a)
    y = y - y[0] + a_pt[1]
    e = dh / h
    return a_pt[None, :] * np.array([1.0, 0.0, 1.0]) + x[:, None] * e[None, :] + y[:, None] * UP[None, :]


def catmull(ctrl, n=160):
    """A smooth curve through control points (centripetal enough for a tether route), resampled."""
    C = np.asarray(ctrl, float)
    C = np.vstack([2 * C[0] - C[1], C, 2 * C[-1] - C[-2]])
    pts = []
    for i in range(1, len(C) - 2):
        p0, p1, p2, p3 = C[i - 1], C[i], C[i + 1], C[i + 2]
        for t in np.linspace(0.0, 1.0, 40, endpoint=False):
            pts.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    pts.append(C[-2])
    P = np.array(pts)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    si = np.linspace(0.0, s[-1], n)
    return np.stack([np.interp(si, s, P[:, k]) for k in range(3)], 1)


def tether_clearance(P):
    """(pile, cage) clearance of the tether's skin, m."""
    r = np.hypot(P[:, 0], P[:, 2])
    wet = P[:, 1] < -0.5
    pile = float(np.min(r[wet] - ts.PILE_R - 0.22)) - fv.TETHER_R
    band = (P[:, 1] > CAGE["ring_y"][1] - 0.3) & (P[:, 1] < CAGE["ring_y"][0] + 0.3)
    cage = float(np.min(np.abs(r[band] - CAGE["ring_radius"]) - 0.5 * CAGE["anode_size"][0] - 0.1)) - fv.TETHER_R if band.any() else 99.0
    return pile, cage


# --------------------------------------------------------------------------- #
#  The jobs
# --------------------------------------------------------------------------- #
def build_subsea(renderer, S, F):
    import types
    J = types.SimpleNamespace()
    V = F.vehicles
    SL = Sleeve(S)
    out = np.asarray(V["snake_out"], float)
    J.sleeve, J.out = SL, out
    J.patch, J.chips, J.look, J.pdir = build_wear(S.scene, SL, out)
    links, lamps = V["snake_links"], V["snake_lamps"]
    off_dir = _unit(0.9 * out + 0.4 * UP)                              # beside, a little above

    # ---- the snake's rail: the sleeve offset to the open side, bowing out near the pile
    s_rail = np.linspace(-2.5, SL.s[-1], 1400)

    def rail(bow_w, s_head):
        c, t = SL.at(s_rail)
        c = c + np.minimum(s_rail, 0.0)[:, None] * SL.T[0][None, :]          # past the entry: straight on
        D = SL.radius(s_rail) + fv.LINK_R + SKIN_GAP + 0.45 * smooth(1.8, 0.0, s_rail)
        base = c + D[:, None] * off_dir[None, :]
        bow = bow_w * 0.20 * np.sin(np.pi * np.clip((s_head - s_rail) / 1.5, 0.0, 1.0))   # held C around the patch
        return base + bow[:, None] * out[None, :]

    def body_joints(s_head, bow_w):
        B = rail(bow_w, s_head)
        sig = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(B, axis=0), axis=1))])
        sh = float(np.interp(s_head, s_rail, sig))
        js = sh - np.arange(fv.LINK_N - 1) * L
        return np.stack([np.interp(js, sig, B[:, k]) for k in range(3)], 1)

    def pose_at(s_head, bow_w, neck_w, target):
        body = body_joints(s_head, bow_w)
        d_body = _unit(body[0] - body[1])
        d_look = _unit(target - body[0])
        d1 = slerp_dir(d_body, _unit(d_body + 1.2 * d_look), neck_w)
        j1 = body[0] + L * d1
        d0 = slerp_dir(d1, _unit(target - j1), neck_w)
        j0 = j1 + L * d0
        return np.vstack([j0, j1, body])

    # travel view: during the transit the head looks ahead and a little down onto the sleeve
    def transit_target(s_head):
        c, t = SL.at(s_head + 0.3)
        return c + SL.radius(s_head + 0.3) * perp(off_dir, t)

    # the hold: the rail point whose neck puts the nose NOSE_GAP off the patch
    best = None
    for sh in np.linspace(SL.s_td - 0.9, SL.s_td + 0.3, 121):
        j = pose_at(sh, 1.0, 1.0, J.look)
        err = abs(np.linalg.norm(j[0] - J.look) - NOSE_GAP)
        if best is None or err < best[0]:
            best = (err, sh)
    J.s_hold = best[1]
    J.s_start = J.s_hold - 2.3
    T_STOP = 11.3

    # ---- a thruster vehicle: rigid on a smooth path, then ONE ease in joint space, then hold
    def chain(dirs, anchor, k_anchor=5):
        """Joints (nose first) from link directions, joint k_anchor pinned at anchor."""
        j = np.zeros((fv.LINK_N + 1, 3))
        for i in range(fv.LINK_N - 1, -1, -1):
            j[i] = j[i + 1] + L * dirs[i]
        return j + (anchor - j[k_anchor])

    def dirs_of(j):
        return np.stack([_unit(j[i] - j[i + 1]) for i in range(fv.LINK_N)])

    work0 = pose_at(J.s_hold, 1.0, 1.0, J.look)
    Hx = _unit(work0[2] - work0[-1])
    Hz = _unit(np.cross(Hx, UP))
    HR = np.stack([Hx, np.cross(Hz, Hx), Hz], axis=1)                 # the hold frame
    side = -1.0 if np.dot(Hz, out) > 0 else 1.0                       # local z toward the sleeve
    # straight shape (local): body a rod, the neck held in a slight look-down toward the sleeve
    loc = np.zeros((fv.LINK_N + 1, 3))
    for k in range(2, fv.LINK_N + 1):
        loc[k] = [-(k - 2) * L, 0.0, 0.0]
    d1 = _unit([0.93, -0.15, side * 0.30])                          # the neck looks down onto the sleeve beside it
    d0 = _unit([0.70, -0.33, side * 0.62])
    loc[1] = loc[2] + L * d1
    loc[0] = loc[1] + L * d0
    loc = loc - loc[5]
    J.anchor = work0[5]
    T_STOP, T_EASE0, T_EASE1 = 9.0, 9.2, 11.7
    lift = 0.16 * UP                                                  # the straight rod clears the rock; the ease settles it
    P_end = J.anchor + lift
    P_start = P_end - 1.6 * Hx + 0.2 * UP + 0.15 * out
    P_ctl = P_end - 0.8 * Hx + 0.05 * UP
    hd_state = {}

    def rigid(tau):
        u = float(min_jerk(tau / T_STOP))
        p = (1 - u) ** 2 * P_start + 2 * (1 - u) * u * P_ctl + u * u * P_end
        tg = _unit(2 * (1 - u) * (P_ctl - P_start) + 2 * u * (P_end - P_ctl))
        return p, tg

    def heading(tau, dt):
        _, tg = rigid(tau)
        if "x" not in hd_state or dt <= 0:
            hd_state["x"] = rigid(T_STOP)[1] if tau >= T_STOP else tg
        else:
            hd_state["x"] = slerp_dir(hd_state["x"], tg, 1 - math.exp(-dt / 0.6))   # low-pass heading
        if tau >= T_STOP:
            hd_state["x"] = slerp_dir(hd_state["x"], Hx, 1 - math.exp(-max(dt, 1e-3) / 0.3))
        x = hd_state["x"]
        z = _unit(np.cross(x, UP))
        return np.stack([x, np.cross(z, x), z], axis=1)

    def snake_pose(tau, scan=0.0, dt=1.0 / FPS):
        drift = np.array([0.012 * math.sin(0.5 * tau), 0.008 * math.sin(0.8 * tau + 1.0), 0.010 * math.cos(0.45 * tau)]) * float(smooth(T_EASE1, T_EASE1 + 1.0, tau))
        if tau < T_EASE0:
            p, _ = rigid(tau)
            R = heading(tau, dt)
            return p[None, :] + loc @ R.T + drift
        straight = P_end + loc @ HR.T
        work = pose_at(J.s_hold, 1.0, 1.0, J.look + scan * SL.at(SL.s_td)[1])
        w = float(min_jerk((tau - T_EASE0) / (T_EASE1 - T_EASE0)))
        ds, dw = dirs_of(straight), dirs_of(work)
        dirs = np.stack([slerp_dir(ds[i], dw[i], w) for i in range(fv.LINK_N)])
        return chain(dirs, J.anchor + (1 - w) * lift) + drift

    J.snake_pose = snake_pose
    head_cam = tp.PerspectiveCamera(62.0, 16.0 / 9.0, 0.02, 60.0)
    J.head_cam = head_cam
    lamp0 = fv.SNAKE_LAMP_I
    for sp in lamps:                                              # aim fixed in the head frame: straight ahead
        sp.distance = 1.4                                         # short throw: the pool stays under the nose
        q = sp.position
        sp.get_target().position.set(q.x + 0.9, q.y, q.z)

    def hit_distance(j):
        """How far the head axis runs to the sleeve or the bed (for the lamp dimmer)."""
        d = _unit(j[0] - j[1])
        for s in np.arange(0.0, 3.0, 0.01):
            p = j[0] + s * d
            dc, sc = SL.nearest(p[None, :])
            if dc[0] < SL.radius(sc[0]) or p[1] < float(ts.seabed_height(p[0], p[2])):
                return s, p
        return 3.0, j[0] + 3.0 * d

    def snake_job(tau, scan=0.0):
        j = snake_pose(tau, scan)
        fv.pose_snake(links, j)
        hd, hp = hit_distance(j)                                  # reported only; the lamps never read it
        for sp in lamps:                                          # one planned dim-down as the head closes in
            sp.intensity = lamp0 * 0.5 * 0.3 ** float(smooth(7.0, 11.0, tau))    # log ramp: < 2 %/frame
        # head camera at the lens, looking along the head
        f = _unit(j[0] - j[1])
        r = _unit(np.cross(f, UP))
        u = np.cross(r, f)
        head_cam.position.set(*map(float, j[0] + 0.004 * f))
        head_cam.quaternion.set(*map(float, quat_of(np.stack([r, u, -f], axis=1))))
        J.snake_joints, J.snake_hit = j, (hd, hp)
        return j

    J.snake_job = snake_job
    J.snake_lamps_i = lambda: float(lamps[0].intensity)

    def snake_cam(j, tau):
        """Low three-quarter front on the open side, tracking the head."""
        c, t = SL.at(J.s_hold)
        fwd = _unit(t * np.array([1.0, 0.0, 1.0]))
        head = j[0]
        w = float(smooth(6.0, 11.0, tau))
        eye = J.look + 0.95 * fwd + 1.1 * out
        eye[1] = float(ts.seabed_height(eye[0], eye[2])) + 0.5
        tgt = 0.45 * head + 0.3 * j[4] + 0.25 * (w * J.look + (1 - w) * j[2])
        tgt[1] = min(tgt[1], eye[1] - 0.08)                           # a little down: no horizon seam
        return eye, tgt, 44.0

    J.snake_cam = snake_cam

    # ---- the ROV: carrot follower (bounded accel, speed, yaw rate; lagged roll/pitch from thrust)
    rov = V["rov"]
    flood = tp.SpotLight(tp.Color(1.0, 0.96, 0.9), ROV_FLOOD_I, 8.0, math.radians(62.0), 1.0, 2.0)   # wide, weak: spill on the neighbours
    flood.position.set(0.215, 0.02, 0.0)
    ftg = tp.Group()
    ftg.position.set(4.0, -0.2, 0.0)
    rov.add(ftg)
    flood.set_target(ftg)
    rov.add(flood)
    V["rov_lights"] = list(V["rov_lights"])
    J.flood = flood
    R_C = CAGE_FACE + ROV_STANDOFF + 0.23
    Y_TOP = CAGE["ring_y"][0]

    def carrot(tau):
        a0, a1 = 101.0, 80.0                                  # ends nose-on to the anode at 80 deg
        y = -8.6 - 2.4 * float(np.clip(tau / 5.5, 0.0, 1.0)) - 0.4 * float(smooth(5.5, 10.5, tau))
        az = a0 + (a1 - a0) * float(min_jerk((tau - 3.0) / 7.8))
        r = R_C + 0.4 * float(smooth(4.0, 0.0, tau))
        a = math.radians(az)
        return np.array([r * math.cos(a), y, r * math.sin(a)])

    st = {}

    def rov_reset(tau):
        st["p"] = carrot(tau)
        st["v"] = (carrot(tau + 0.05) - carrot(tau)) / 0.05
        st["yaw"] = yaw_to_pile(st["p"])
        st["r"] = 0.0
        st["pitch"] = st["roll"] = 0.0
        st["peak"] = [0.0, 0.0, 0.0, 0.0, 0.0]              # speed, accel, yaw rate, |pitch|, |roll|
        place_rov(st["p"], rot_y(st["yaw"]))

    def yaw_to_pile(p):
        f = -np.array([p[0], 0.0, p[2]])
        return math.atan2(-f[2], f[0])                       # rot_y(yaw) @ +X = f

    def rov_step(tau, dt):
        n = 4
        h = dt / n
        for k in range(n):
            tk = tau + k * h
            c = carrot(tk + 0.6)                             # the carrot runs ahead
            vc = (carrot(tk + 0.65) - carrot(tk + 0.55)) / 0.1
            v_des = vc + 0.6 * (c - st["p"])
            sp = np.linalg.norm(v_des)
            if sp > ROV_V:
                v_des *= ROV_V / sp
            a = (v_des - st["v"]) / 0.8
            am = np.linalg.norm(a)
            if am > ROV_A:
                a *= ROV_A / am
            st["v"] = st["v"] + a * h
            st["p"] = st["p"] + st["v"] * h
            e = (yaw_to_pile(st["p"]) - st["yaw"] + math.pi) % (2 * math.pi) - math.pi
            r_des = float(np.clip(1.2 * e, -ROV_YAW, ROV_YAW))
            st["r"] += float(np.clip(r_des - st["r"], -math.radians(40.0) * h, math.radians(40.0) * h))
            st["yaw"] += st["r"] * h
            f = np.array([math.cos(st["yaw"]), 0.0, -math.sin(st["yaw"])])
            sd = np.cross(UP, f)
            p_t = -0.16 * float(np.dot(a, f)) - 0.05 * float(st["v"][1])      # nose dips as it thrusts ahead
            r_t = 0.16 * float(np.dot(a, sd))
            st["pitch"] += (p_t - st["pitch"]) * (1 - math.exp(-h / 0.7))
            st["roll"] += (r_t - st["roll"]) * (1 - math.exp(-h / 0.7))
            pk = st["peak"]
            pk[0] = max(pk[0], float(np.linalg.norm(st["v"])))
            pk[1] = max(pk[1], float(np.linalg.norm(a)))
            pk[2] = max(pk[2], abs(st["r"]))
            pk[3] = max(pk[3], abs(st["pitch"]))
            pk[4] = max(pk[4], abs(st["roll"]))
        place_rov(st["p"], rot_y(st["yaw"]) @ rot_z(st["pitch"]) @ rot_x(st["roll"]))

    def place_rov(p, R):
        rov.position.set(*map(float, p))
        rov.quaternion.set(*map(float, quat_of(R)))
        V["rov_pos"], V["rov_fwd"] = p, R[:, 0]
        J.rov_R = R

    J.rov_reset, J.rov_step, J.rov_state = rov_reset, rov_step, st

    def rov_job(tau, dt):
        if "p" not in st:
            rov_reset(tau)
        elif dt > 0.0:
            rov_step(tau, dt)
        return st["p"]

    J.rov_job = rov_job

    # ---- the tether mesh (ours; the phase-2 Bezier is hidden)
    tmat = V["tether_mat"]
    J.tether_mesh = None
    J.tether_min = [99.0, 99.0, 0.0]

    def tether(P):
        geo = ts.tube(P, lambda s: np.full_like(s, TETHER_R), sides=8)
        if J.tether_mesh is None:
            J.tether_mesh = tp.Mesh(geo, tmat)
            S.scene.add(J.tether_mesh)
        else:
            g = J.tether_mesh.geometry
            g.update_attribute("position", np.asarray(geo.get_attribute("position"), np.float32))
            g.update_attribute("normal", np.asarray(geo.get_attribute("normal"), np.float32))
        pile, cage = tether_clearance(P)
        J.tether_min[0] = min(J.tether_min[0], pile)
        J.tether_min[1] = min(J.tether_min[1], cage)
        return P

    def moonpool():
        return F.mariner.to_world([0.2, 0.0, 0.0])

    def rov_tether():
        tail = st["p"] + J.rov_R @ TETHER_LOCAL
        top = moonpool()
        lean = (tail - top) * np.array([1.0, 0.0, 1.0])
        guide = top - 1.2 * UP + 0.03 * _unit(lean)          # the moonpool's guide tube (a hair off vertical: the tube frame needs it)
        cat = catenary(guide, tail, n=148)
        chord = guide + np.linspace(0.0, 1.0, 148)[:, None] * (tail - guide)
        sag = np.linalg.norm(cat - chord, axis=1)
        drift = _unit(np.array([-0.08, 0.0, 0.015]))                   # the slack streams with the tidal current, not down
        cat = chord + sag[:, None] * drift[None, :]
        P = np.vstack([top + np.linspace(0.0, 1.0, 12, endpoint=False)[:, None] * (guide - top), cat])
        keel = P[P[:, 1] > top[1] - 1.0]                     # inside the moonpool well and just under the keel
        J.tether_min[2] = max(J.tether_min[2], float(np.max(np.linalg.norm((keel - top) * np.array([1.0, 0.0, 1.0]), axis=1))))
        return tether(P)

    J.rov_tether, J.moonpool = rov_tether, moonpool

    def rov_cam(tau):
        p, R = st["p"], J.rov_R
        f = _unit(R[:, 0] * np.array([1.0, 0.0, 1.0]))
        sd = np.cross(UP, f)
        tang = -sd if np.dot(-sd, np.array([-p[2], 0.0, p[0]])) < 0 else sd   # the way it travels (az decreasing)
        eye = p + 2.3 * tang + 0.35 * f + np.array([0.0, 0.8, 0.0])        # beside it: the tether runs off outward and up
        lit = p + 2.0 * f + np.array([0.0, -0.45, 0.0])
        tgt = 0.72 * p + 0.28 * lit + np.array([0.0, 0.1, 0.0])
        return eye, tgt, 42.0

    J.rov_cam = rov_cam

    # ---- the pair: ROV parked up and to the side, lamps on the patch; tether routed round the pile
    def pair(tau):
        j = snake_job(SNAKE_T, 0.12 * math.sin(0.9 * tau))           # the neck scans across the patch
        c, t = SL.at(SL.s_td)
        fwdc = _unit(t * np.array([1.0, 0.0, 1.0]))
        p = J.look + 2.1 * out + 2.4 * UP + 0.9 * fwdc + np.array([0.06 * math.sin(0.5 * tau), 0.04 * math.sin(0.9 * tau), 0.05 * math.cos(0.4 * tau)])
        f = _unit((J.look - p) * np.array([1.0, 0.0, 1.0]))
        yaw = math.atan2(-f[2], f[0]) + math.radians(2.5) * math.sin(0.35 * tau)
        R = rot_y(yaw) @ rot_z(math.radians(-8.0)) @ rot_x(math.radians(1.5 * math.sin(0.7 * tau)))
        st["p"] = p
        place_rov(p, R)
        # lamps on tilt brackets: aim both at the patch
        Rinv = R.T
        dist = float(np.linalg.norm(J.look - p))
        for k, sp in enumerate(V["rov_lights"]):
            tg = sp.get_target()
            q = Rinv @ (J.look + (0.25 if k else -0.25) * fwdc - p)
            tg.position.set(*map(float, q))
            sp.intensity = 0.8 * fv.ROV_SPOT_I                          # v01 scaled for distance and bleached the site
        # tether: up off the ROV, round the pile on the far side of the berm, up to the moonpool
        top = moonpool()
        tail = p + R @ TETHER_LOCAL
        a_rov = math.atan2(p[2], p[0])
        a_top = math.atan2(top[2], top[0])
        da = (a_top - a_rov) % (2 * math.pi)
        if da > math.pi:
            da -= 2 * math.pi                                   # the shorter way round
        ctrl = [tail, tail + np.array([0.0, 1.5, 0.0]) - 0.3 * R[:, 0]]
        for w in np.linspace(0.15, 0.85, 5):
            a = a_rov + w * da
            sway = 0.35 * math.sin(0.6 * tau + 4.0 * w)                  # the tether sways in the current
            ctrl.append(np.array([(10.5 + sway) * math.cos(a), -24.0 + 20.0 * w, (10.5 + sway) * math.sin(a)]))
        ctrl += [top + np.array([0.0, -3.0, 0.0]), top]
        return j, tether(catmull(ctrl, n=160))

    J.pair = pair

    def pair_cam(tau):
        c, t = SL.at(SL.s_td)
        fwdc = _unit(t * np.array([1.0, 0.0, 1.0]))
        eye = J.look + 4.6 * _unit(1.0 * fwdc + 0.55 * out) + np.array([0.0, 0.7, 0.0])
        tgt = J.look + 1.6 * UP + 0.9 * out
        return eye, tgt, 58.0

    J.pair_cam = pair_cam
    return J


def snake_clearance(J, j):
    """(rock, sleeve, pile) skin clearance, m, sampled along every link."""
    pts = []
    for i in range(fv.LINK_N):
        for u in np.linspace(0.0, 1.0, 10):
            pts.append(j[i + 1] + (j[i] - j[i + 1]) * u)
    P = np.array(pts)
    bed = P[:, 1] - ts.seabed_height(P[:, 0], P[:, 2]) - SNAKE_RAD
    d, s = J.sleeve.nearest(P)
    sleeve = d - J.sleeve.radius(s) - SNAKE_RAD
    pile = np.hypot(P[:, 0], P[:, 2]) - ts.PILE_R - 0.22 - SNAKE_RAD
    return float(bed.min()), float(sleeve.min()), float(pile.min())


def joint_bends(j):
    return [math.degrees(math.acos(float(np.clip(np.dot(_unit(j[i] - j[i + 1]), _unit(j[i + 1] - j[i + 2])), -1, 1))))
            for i in range(len(j) - 2)]


# --------------------------------------------------------------------------- #
#  Headless capture
# --------------------------------------------------------------------------- #
def next_name(outdir, stem, ext):
    k = 1
    while any(os.path.exists(os.path.join(outdir, f"{stem}_v{k:02d}.{e}")) for e in ("mp4", "png")):
        k += 1
    return os.path.join(outdir, f"{stem}_v{k:02d}.{ext}"), k


def save_png(rgb, path):
    from PIL import Image
    d, b = os.path.split(path)
    tmp = os.path.join(d, "_tmp_" + b)
    Image.fromarray(rgb).save(tmp)
    os.replace(tmp, path)
    print(f"[subsea] wrote {path}")


def sheet(frames, path):
    tiles = [f[::4, ::4] for f in frames]
    h, w = tiles[0].shape[:2]
    cols = 4
    rows = (len(tiles) + cols - 1) // cols
    img = np.zeros((rows * h + (rows - 1) * 4, cols * w + (cols - 1) * 4, 3), np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        img[r * (h + 4):r * (h + 4) + h, c * (w + 4):c * (w + 4) + w] = t
    save_png(img, path)


def pip_dress(rgb, x, y, w, h, label):
    """A thin frame and a small label on the composited inset."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(rgb)
    dr = ImageDraw.Draw(im)
    dr.rectangle([x - 2, y - 2, x + w + 1, y + h + 1], outline=(200, 205, 200), width=2)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    dr.text((x + 8, y + 6), label, fill=(225, 230, 225), font=font)
    return np.asarray(im)


def main():
    jobs = cli_arg("--jobs", "snake,rov,pair", str).split(",")
    outdir = cli_arg("--outdir", ts.out_dir("jobs"), str)
    os.makedirs(outdir, exist_ok=True)
    W, H = 1920, 1080
    canvas = tp.Canvas("threepp - subsea jobs", width=W, height=H, vsync=False, headless=True)
    renderer = tp.VulkanRenderer(canvas)
    renderer.tone_mapping = tp.ToneMapping.AgX
    renderer.tone_mapping_exposure = 1.4
    renderer.sun_angular_radius = 0.6
    renderer.bloom_intensity = 0.06
    renderer.auto_exposure = False
    t0 = time.perf_counter()
    for attempt in range(2):
        try:
            S = ts.build_site(renderer, sheet=12000.0)
            break
        except Exception as e:                                  # 3A may be rewriting turbine.glb
            if attempt:
                raise
            print("[subsea] site load failed, retry in 20 s:", e)
            time.sleep(20.0)
    camera = tp.PerspectiveCamera(45.0, W / H, 0.03, 20000.0)
    F = tf.build_fleet(renderer, S)
    J = build_subsea(renderer, S, F)
    V = F.vehicles
    print(f"[subsea] built in {time.perf_counter() - t0:.1f} s; touchdown at s {J.sleeve.s_td:.2f} m "
          f"{np.round(J.look, 2)}, hold s {J.s_hold:.2f}")
    S.set_lamps((0, 0, 0), (1, 0, 0), False)

    def aim(eye, tgt, fov):
        camera.fov = fov
        camera.update_projection_matrix()
        camera.position.set(*map(float, eye))
        camera.look_at(tp.Vector3(*map(float, tgt)))

    # settle the boats (the moonpool heaves with the Mariner) and give the ocean its first frame
    T0 = tf.T_SHOT
    aim(np.array([0.0, -12.0, 12.0]), np.array([0.0, -12.0, 0.0]), 50.0)
    n_set = int(cli_arg("--settle", 3.0, float) * FPS)
    for k in range(n_set + 1):
        t = T0 - (n_set - k) / FPS
        renderer.sim_time = t
        S.update(t, camera, True, 1.0 / FPS)
        renderer.render(S.scene, camera)
        F.update(t, 1.0 / FPS)
    if V["tether"] is not None:
        V["tether"].visible = False
    view = [0]
    PIP = (W - 24 - 480, 24, 480, 270)

    def film(stem, dur, pose, cam, keys, pip=False):
        path, ver = next_name(outdir, stem, "mp4")
        d, b = os.path.split(path)
        tmp = os.path.join(d, "_tmp_" + b)
        enc = Encoder(tmp, W, H, FPS, crf=18, preset="medium", faststart=True)
        n = int(round(dur * FPS))
        pick = set(np.linspace(0, n - 1, 8).round().astype(int).tolist())
        keyk = {int(round(k * FPS)) if k >= 0 else n + int(round(k * FPS)): nm for k, nm in keys}
        tiles, t1 = [], time.perf_counter()
        eye_s = tgt_s = None
        for k in range(n):
            tau = k / FPS
            t = T0 + tau
            pose(tau, 1.0 / FPS if k else 0.0)
            eye, tgt, fov = cam(tau)
            a = 1.0 - math.exp(-(1.0 / FPS) / 0.5)
            eye_s = eye if eye_s is None else eye_s + a * (eye - eye_s)
            tgt_s = tgt if tgt_s is None else tgt_s + a * (tgt - tgt_s)
            aim(eye_s, tgt_s, fov)
            if k == 0:
                S.update(t, camera, True, 0.0)
                S.settle_snow(camera, t)                    # once: then it drifts with the film
            renderer.sim_time = t
            S.update(t, camera, True, 1.0 / FPS)
            renderer.sim_time = t
            renderer.render(S.scene, camera)
            rgb = renderer.read_pixels()
            if pip and view[0]:
                x, y, w, h = PIP
                if k == 0:                                  # is the device composite in the readback?
                    inset = renderer.read_view_rgb_pixels(view[0])
                    diff = float(np.abs(rgb[y:y + h, x:x + w].astype(int) - inset.astype(int)).mean()) if inset.size else 99.0
                    view.append(diff > 12.0)
                    print(f"[subsea] PiP composite vs view readback: mean diff {diff:.1f} -> {'paste' if view[-1] else 'device composite'}")
                if view[-1]:
                    rgb = rgb.copy()
                    rgb[y:y + h, x:x + w] = renderer.read_view_rgb_pixels(view[0])
                rgb = pip_dress(rgb, x, y, w, h, "SNAKE CAM")
            enc.send(rgb)
            if k in pick:
                tiles.append(rgb)
            if k in keyk:
                save_png(rgb, os.path.join(outdir, f"{stem}_{keyk[k]}_v{ver:02d}.png"))
            F.update(t, 1.0 / FPS)
            if V["tether"] is not None:
                V["tether"].visible = False
        rc = enc.close()
        if rc == 0:
            os.replace(tmp, path)
            print(f"[subsea] wrote {path} ({n} frames, {time.perf_counter() - t1:.0f} s)")
        else:
            print(f"[subsea] encoder rc {rc} for {tmp}")
        sheet(tiles, os.path.join(outdir, f"{stem}_sheet_v{ver:02d}.png"))

    def do_snake():
        rov_hide = V["rov"].visible
        V["rov"].visible = False                               # the snake's own light only
        for sp in V["rov_lights"]:
            sp.intensity = 0.0
        J.flood.intensity = 0.0
        if J.tether_mesh is not None:
            J.tether_mesh.visible = False
        stats = {"clr": [9.0, 9.0, 9.0], "bend": 0.0, "v": 0.0}
        prev = [None]

        def pose(tau, dt):
            j = J.snake_job(tau)
            bends = np.array(joint_bends(j))
            if stats.get("b") is not None and dt > 0:
                dj = np.abs(bends - stats["b"])
                if tau < 9.2:
                    stats["transit_body"] = max(stats.get("transit_body", 0.0), float(dj[1:].max()))
                else:
                    stats["ease_rate"] = max(stats.get("ease_rate", 0.0), float(dj[1:].max()) / dt)
            stats["b"] = bends
            li = J.snake_lamps_i()
            if stats.get("li") is not None:
                stats["li_jump"] = max(stats.get("li_jump", 0.0), abs(li - stats["li"]) / max(stats["li"], 1e-6))
            stats["li"] = li
            c = snake_clearance(J, j)
            stats["clr"] = [min(a, b) for a, b in zip(stats["clr"], c)]
            stats["bend"] = max(stats["bend"], max(joint_bends(j)))
            if prev[0] is not None and dt > 0:
                stats["v"] = max(stats["v"], float(np.linalg.norm(j[0] - prev[0])) / dt)
            prev[0] = j[0].copy()

        pose(0.0, 0.0)
        if cli_arg("--pip", 1, int):
            renderer.render(S.scene, camera)
            view[0] = renderer.add_view(J.head_cam, PIP[2], PIP[3])
            if view[0]:
                renderer.set_view_display_rect(view[0], *PIP)
            print(f"[subsea] head camera view {view[0]}")
        film("snake_job", SNAKE_T, pose, lambda tau: J.snake_cam(J.snake_joints, tau),
             [(4.0, "transit"), (-1.0 / FPS, "hold")], pip=True)
        j = J.snake_joints
        print(f"[subsea] snake: min clearance rock {stats['clr'][0]:.3f} sleeve {stats['clr'][1]:.3f} pile {stats['clr'][2]:.3f} m; "
              f"peak joint bend {stats['bend']:.0f} deg; peak nose speed {stats['v']:.2f} m/s; nose to patch at the hold "
              f"{np.linalg.norm(j[0] - J.look):.2f} m; head axis meets the sleeve {J.snake_hit[0]:.2f} m out, "
              f"{np.linalg.norm(J.snake_hit[1] - J.look):.2f} m from the patch centre")
        print(f"[subsea] snake joints: max body-joint change per frame in transit {stats.get('transit_body', 0):.3f} deg; "
              f"max body-joint rate in the ease/hold {stats.get('ease_rate', 0):.1f} deg/s; lamp max frame-to-frame change "
              f"{100 * stats.get('li_jump', 0):.2f} %, target fixed in the head frame")
        if view[0]:
            renderer.set_view_display_rect(view[0], -9000, 0, PIP[2], PIP[3])
        V["rov"].visible = rov_hide
        for sp in V["rov_lights"]:
            sp.intensity = fv.ROV_SPOT_I
        J.flood.intensity = ROV_FLOOD_I
        if J.tether_mesh is not None:
            J.tether_mesh.visible = True

    def do_rov():
        J.rov_reset(0.0)
        for sp in V["rov_lights"]:
            sp.intensity = ROV_JOB_I

        def pose(tau, dt):
            J.rov_job(tau, dt)
            J.rov_tether()

        film("rov_job", ROV_T, pose, J.rov_cam, [(1.0, "arrive"), (-1.0 / FPS, "pass")])
        pk = J.rov_state["peak"]
        print(f"[subsea] ROV: peak speed {pk[0]:.2f} m/s, accel {pk[1]:.2f} m/s^2, yaw rate {math.degrees(pk[2]):.1f} deg/s, "
              f"|pitch| {math.degrees(pk[3]):.1f} |roll| {math.degrees(pk[4]):.1f} deg; tether min clearance pile "
              f"{J.tether_min[0]:.2f} m, cage {J.tether_min[1]:.2f} m, max drift in the moonpool well {J.tether_min[2]:.2f} m")

    def do_pair():
        J.tether_min[:2] = [99.0, 99.0]

        def pose(tau, dt):
            J.pair(tau)

        film("subsea_pair", PAIR_T, pose, J.pair_cam, [(-1.0 / FPS, "still")])
        print(f"[subsea] pair: tether min clearance pile {J.tether_min[0]:.2f} m, cage {J.tether_min[1]:.2f} m")


    for nm in jobs:                                         # in the order asked
        {"snake": do_snake, "rov": do_rov, "pair": do_pair}[nm]()

if __name__ == "__main__":
    main()
