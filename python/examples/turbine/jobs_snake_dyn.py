"""Phase 3C: the snake job flown on its physics model (netpen/snake_model.py, PhysX, planar).

Position and heading come only from the thrusters, drag, added mass, the current and a controller.
Modes (setpoints only): TRANSIT (ILOS along the sleeve, crabbing into the current) -> APPROACH
(decelerate onto the hold point) -> HOLD (station keeping) -> SHAPE (joint targets ramped over 2.5 s,
setpoints moved with them) -> INSPECT (neck scan through the joint PD; a small visual neck pitch).

    python jobs_snake_dyn.py --traj            # physics only: numbers + the trajectory plot
    python jobs_snake_dyn.py --render          # the clip (headless), sheet, stills

FILM API (see build_snake_dyn):
    job = build_snake_dyn(renderer, S, F, J)   # after ts.build_site, tf.build_fleet, js.build_subsea
    job.reset(); job.step_to(tau)              # physics to time tau (monotonic), poses links/lamps/props/J.head_cam
    eye, tgt, fov = job.cam(tau)               # low-passed three-quarter-front camera, once per frame
    job.T["TRANSIT"|"APPROACH"|"HOLD"|"SHAPE"|"INSPECT"]   # mode start times, s
Headless only.
"""
import hashlib
import math
import os
import sys
import time
import types

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.dirname(_HERE)
for _p in (os.path.dirname(_EX), _EX, _HERE, os.path.join(_EX, "netpen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import threepp as tp
import turbine_site as ts
import fleet_vehicles as fv
import jobs_subsea as js
import snake_model as sm

UP = fv.UP
_unit = fv._unit
FPS = 30
DT = 1.0 / 240.0
SUB = 8                                   # physics substeps per 30 fps frame
CTRL_EVERY = 4                            # controller at 60 Hz
L = sm.SnakeParams().link_length
N = 9
CURRENT = 0.12 * np.array([1.0, 0.0, 0.0])   # m/s toward +X: the reverse tide of the spec's 180 deg; 45 deg against/across the sleeve
U_TRANSIT = 0.22                          # m/s over ground
GAP = 0.60                                # skin to sleeve skin in transit
SKIN = fv.LINK_R + 0.03                   # shell + ducts
Y_CLEAR = 0.37                            # body axis to the highest rock under it: skin clearance >= 0.27 m
LAMP_HI, LAMP_LO = 1.5, 0.40              # one planned dim as the head closes (fv.SNAKE_LAMP_I 3 burnt the patch)
NECK_PITCH = (math.radians(4.0), math.radians(6.0))   # visual only (planar model): neck link, head link
T_TOTAL = 34.0
NECK_TRANSIT = [math.radians(11.0), math.radians(14.0)]   # neck joints in transit and approach (aim, not gait)
CAM_TILT = math.radians(24.0)            # head camera mount, pitched down in the head frame (fixed): the
HEAD_FOV = 46.0                          # secondary view shows its unmurked background above the horizon
                                         # (a light grey band, not fixed by a 12 m far plane): keep it out of frame


def smooth(e0, e1, x):
    u = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return float(u * u * (3.0 - 2.0 * u))


def min_jerk(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u ** 3 * (10.0 - 15.0 * u + 6.0 * u * u)


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_of(d):
    return math.atan2(-d[2], d[0])


def ydir(h):
    return np.array([math.cos(h), 0.0, -math.sin(h)])


def site_like():
    return types.SimpleNamespace(cable_path=ts.cable_path()[0], spec=ts.spec)


def snake_out(S):
    """fleet_vehicles.build_fleet's choice of the snake's open side (no renderer needed)."""
    for sgn in (1.0, -1.0):
        joints, look, out = fv.snake_station(S, sgn)
        eye, tgt, _ = fv.snake_cam({"snake_joints": joints, "snake_look": look, "snake_out": out})
        right = _unit(np.cross(_unit(tgt - eye), UP))
        if np.dot(look - joints[4], right) < 0.0:
            break
    return np.asarray(out, float)


def bed_max(x, z, r=0.12):
    """Highest rock under a small footprint (the depth hold's sonar altimeter)."""
    h = -1e9
    for dx in (-r, 0.0, r):
        for dz in (-r, 0.0, r):
            h = np.maximum(h, ts.seabed_height(np.asarray(x) + dx, np.asarray(z) + dz))
    return h


class SnakeDyn:
    def __init__(self, S, out, look=None, seed=0):
        self.rng = np.random.default_rng(seed)                  # fixed seed (nothing random is drawn today)
        self.SL = SL = js.Sleeve(S)
        self.out = out = np.asarray(out, float)
        c, t = SL.at(SL.s_td)
        self.pdir = js.perp(out + 0.3 * UP, t)
        self.look = c + SL.radius(SL.s_td) * self.pdir if look is None else np.asarray(look, float)
        # the rail: sleeve centre, horizontally GAP off its skin to the open side
        s = np.linspace(0.0, SL.s[-1], 900)
        cc, tt = SL.at(s)
        th = _unit_rows(tt * [1.0, 0.0, 1.0])
        no = _unit_rows(np.cross(UP[None, :], th))
        no *= np.sign(np.sum(no * out, axis=1))[:, None]
        self.rail = cc * [1.0, 0.0, 1.0] + (SL.radius(s) + SKIN + GAP)[:, None] * no
        self.rail_t, self.rail_n = th, no
        self.rail_s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.rail, axis=0), axis=1))])
        self.world = sm.make_world(DT)
        p = sm.SnakeParams(thrusters=sm.thruster_layout(), current=tuple(CURRENT))
        s0 = 0.9                                                # start: the tail just clear of the pile growth
        i0 = int(np.searchsorted(s, s0))
        h0 = yaw_of(U_TRANSIT * th[i0] - CURRENT)                 # nose into the transit crab
        com0 = self.rail[i0]
        self.sn = sn = sm.Snake(self.world, p, origin=(float(com0[0]), 0.0, float(com0[2])), heading=h0)
        sn.set_gait("lateral", 0.0, 0.0, 0.0, 0.0, ramp=False)    # gait amplitude ZERO throughout
        sn.set_shape(np.zeros(N - 1))
        sn._shape.w = 6.64 / 1.0                                # the PD targets follow a 1 s smoother (ramped by us too)
        self.ct, self.cn = sn.c_t, sn.c_n
        self.t = 0.0
        self.k = 0
        self.mode, self.t_mode = "TRANSIT", 0.0
        self.events = [(0.0, "TRANSIT")]
        self.z_los = 0.0
        self.zi = np.zeros(3)
        self.shape_w = 0.0
        self.pitch_w = 0.0
        self.scan = 0.0
        self.psi_ref = sn.mean_heading()
        self._plan_hold()
        pos, _ = sn.link_poses()
        self.y = float(self._y_ref(pos))
        self.yd = 0.0
        self.cmd_log = np.zeros(4)
        self.v_d = np.zeros(3)
        self.psi_d = self.psi_ref
        self.t_app = None
        self.lamp = LAMP_HI
        self.prop_ang = np.zeros(4)
        # start in steady transit (not from rest): every link moving at the transit ground velocity,
        # the thrusters already spun up to the controller's first command
        v0 = U_TRANSIT * th[i0]
        for kk, lk in enumerate(sn.links):
            lk.add_impulse(sm._v3(sn.mass[kk] * v0))
        self._control(DT * CTRL_EVERY)
        sn.thr = sn.thr_cmd.copy()
        self.t_steady = 0.0

    # ---------------------------------------------------------------- the hold, planned once
    def _plan_hold(self):
        sn = self.sn
        pos, q = sn.link_poses()
        R0 = sm.quat_to_R(q)[0]
        self._R_build = R0
        # the final shape: body a gentle C toward the sleeve, the neck turned onto the patch
        self.sgn = self._toward_sleeve_sign()
        b, n1, n2 = math.radians(3.5), math.radians(17.0), math.radians(19.0)
        self.q_f = self.sgn * np.array([b] * 6 + [n1, n2])
        self.q_tr = self.sgn * np.array([0.0] * 6 + NECK_TRANSIT)     # transit: body straight, the neck looks at the sleeve beside it
        cf, Rf = sn.fk(np.zeros(3), R0, self.q_f)
        tf = Rf @ sn.t_loc
        yaws = np.unwrap(np.arctan2(-tf[:, 2], tf[:, 0]))
        com_l, psi_l = cf.mean(0), float(yaws.mean())
        nose_l, f_l = cf[-1] + 0.5 * L * tf[-1], tf[-1]
        c, t = self.SL.at(self.SL.s_td)
        th = _unit(t * [1.0, 0.0, 1.0])
        ph = _unit(self.pdir * [1.0, 0.0, 1.0])
        self.hold_plan = None
        for k_ob in np.linspace(0.2, 1.4, 25):                  # the head looks in obliquely, from the pile side
            d_h = _unit(-ph + k_ob * th)
            dpsi = wrap(yaw_of(d_h) - yaw_of(f_l))
            Rz = _roty(dpsi)
            for D_h in np.linspace(0.28, 0.50, 23):                # horizontal nose to look point
                nose_w = self.look * [1.0, 0.0, 1.0] - D_h * d_h
                p_f = nose_w + Rz @ (com_l - nose_l)
                p_f[1] = 0.0
                psi_f = psi_l + dpsi
                jf = self._joints_at(self.q_f, p_f, psi_f, 1.0)
                js0 = self._joints_at(self.q_tr, p_f, psi_f, 0.0)      # body straight, same setpoint: before SHAPE
                cf_, cs_ = clearances(self, jf), clearances(self, js0)
                sl = min(cf_[1], cs_[1])
                dist = float(np.linalg.norm(jf[0] - self.look))
                ok = sl >= 0.32 and min(cf_[0], cs_[0]) >= 0.25
                score = abs(dist - 0.40) + 0.02 * k_ob
                if ok and (self.hold_plan is None or score < self.hold_plan[0]):
                    self.hold_plan = (score, k_ob, D_h, p_f, psi_f, dist, cf_, cs_)
        _, k_ob, D_h, self.p_f, self.psi_f, dist, cf_, cs_ = self.hold_plan
        print(f"[snake_dyn] hold plan: look obliqueness {k_ob:.2f}, nose {D_h:.2f} m off (horizontal), {dist:.2f} m 3D; "
              f"clearance rock/sleeve/pile straight {np.round(cs_, 3)}, shaped {np.round(cf_, 3)}")
        # ONE station-keeping setpoint from the end of the approach to the end: the shape change happens around it
        self.psi_0, self.p_0 = self.psi_f, self.p_f.copy()
        # where the approach starts (along-rail)
        self.s_hold = float(self._rail_project(self.p_0)[0])

    def _joints_at(self, q, p_com, psi, pitch_w):
        """Scene joints of the planned pose: shape q, COM at p_com, mean heading psi (for the plan's checks)."""
        sn = self.sn
        R0 = self._R_build
        c, Rk = sn.fk(np.zeros(3), R0, q)
        tl = Rk @ sn.t_loc
        yaws = np.unwrap(np.arctan2(-tl[:, 2], tl[:, 0]))
        Rz = _roty(wrap(psi - float(yaws.mean())))
        c = (c - c.mean(0)) @ Rz.T + p_com
        tl = tl @ Rz.T
        return self._joints_from(c, tl, self._y_ref(c), pitch_w)

    def _joints_from(self, pos, tl, y, pitch_w):
        fr = pos + 0.5 * L * tl                                  # front end of each physics link
        j = np.vstack([fr[::-1], pos[0] - 0.5 * L * tl[0]])      # nose .. tail
        j[:, 1] = y
        a1, a2 = NECK_PITCH[0] * pitch_w, (NECK_PITCH[0] + NECK_PITCH[1]) * pitch_w
        d1 = _unit((j[1] - j[2]) * [1, 0, 1])                    # visual only: the planar model has no pitch
        d0 = _unit((j[0] - j[1]) * [1, 0, 1])
        j[1] = j[2] + L * (math.cos(a1) * d1 - math.sin(a1) * UP)
        j[0] = j[1] + L * (math.cos(a2) * d0 - math.sin(a2) * UP)
        return j

    def _toward_sleeve_sign(self):
        """+1 if a positive joint angle turns the headward link toward the sleeve (-out)."""
        sn = self.sn
        pos, q = sn.link_poses()
        R0 = sm.quat_to_R(q)[0]
        c, R = sn.fk(np.zeros(3), R0, np.array([0.0] * 7 + [0.3]))
        f = R[-1] @ sn.t_loc
        return 1.0 if np.dot(f, -self.out) > 0 else -1.0

    def _rail_project(self, p):
        d = np.linalg.norm((self.rail - p * [1.0, 0.0, 1.0]), axis=1)
        i = int(np.argmin(d))
        e = float(np.dot(p * [1, 0, 1] - self.rail[i], self.rail_n[i]))
        return self.rail_s[i], e, self.rail_t[i], self.rail_n[i]

    # ---------------------------------------------------------------- vertical (not in the planar model)
    def _y_ref(self, pos):
        f = _unit(pos[-1] - pos[0])
        pts = np.vstack([pos, pos[-1] + 0.3 * f])               # the body + 0.3 m ahead of the nose
        return float(np.max(bed_max(pts[:, 0], pts[:, 2]))) + Y_CLEAR

    # ---------------------------------------------------------------- the controller
    def _state(self):
        sn = self.sn
        pos, q = sn.link_poses()
        v, w = sn.velocities(pos, q)
        m = sn.mass
        com = (m[:, None] * pos).sum(0) / m.sum()
        vc = (m[:, None] * v).sum(0) / m.sum()
        psi = sn.mean_heading()
        r = float(np.mean(w[:, 1]))
        return pos, q, com, vc, psi, r

    def _guidance(self, com, vc, psi, dt):
        t = self.t
        md = self.mode
        tm = t - self.t_mode
        if md in ("TRANSIT", "APPROACH"):
            s, e, th, no = self._rail_project(com)
            d_go = float(np.linalg.norm((self.p_0 - com) * [1, 0, 1]))
            U = U_TRANSIT
            if md == "TRANSIT":
                if self.s_hold - s < 1.0:
                    self._goto("APPROACH")
                    self.psi_app0 = self.psi_d
                self.z_los = float(np.clip(self.z_los + 0.12 * e * dt, -0.3, 0.3))
                Dl = 1.0
                v_d = U * _unit(Dl * th - float(np.clip(e + self.z_los, -Dl, Dl)) * no)
                psi_d = yaw_of(v_d - CURRENT)                        # nose into the water it swims through
            else:
                sp = min(U, math.sqrt(2.0 * 0.03 * max(d_go - 0.02, 0.0)), 0.35 * d_go + 0.01)
                self.z_los = float(np.clip(self.z_los + 0.12 * e * dt, -0.3, 0.3))
                d_rail = _unit(th - float(np.clip(e + self.z_los, -1.0, 1.0)) * no)
                wa = smooth(0.0, 2.5, tm)                           # the mode switch ramps, no setpoint step
                v_d = sp * _unit((1 - wa) * d_rail + wa * _unit((self.p_0 - com) * [1, 0, 1]))
                w = smooth(0.8, 0.2, d_go)
                crab = yaw_of(v_d - CURRENT) if sp > 0.05 else self.psi_0
                psi_d = crab + w * wrap(self.psi_0 - crab)
                if d_go < 0.06 and np.linalg.norm(vc * [1, 0, 1]) < 0.03 and abs(wrap(self.psi_0 - psi)) < math.radians(4):
                    self._goto("HOLD")
            return v_d, psi_d, None
        # station keeping modes
        if md == "HOLD" and tm > 2.0:
            self._goto("SHAPE")
        if self.mode == "SHAPE":
            self.shape_w = min_jerk((t - self.t_mode) / 2.5)
            if t - self.t_mode > 3.2:
                self._goto("INSPECT")
        if self.mode == "INSPECT":
            ti = t - self.t_mode
            self.pitch_w = min_jerk(ti / 1.5)
            self.scan = math.radians(5.0) * math.sin(2 * math.pi * max(ti - 1.5, 0.0) / 7.0) * smooth(1.5, 3.0, ti)
        w = self.shape_w
        p_t = (1 - w) * self.p_0 + w * self.p_f
        psi_t = self.psi_0 + w * wrap(self.psi_f - self.psi_0)
        return None, psi_t, p_t

    def _goto(self, m):
        self.mode, self.t_mode = m, self.t
        self.events.append((round(self.t, 3), m))

    def _control(self, dt):
        sn = self.sn
        pos, q, com, vc, psi, r = self._state()
        v_d, psi_d, p_t = self._guidance(com, vc, psi, dt)
        if p_t is not None:                                         # station keeping: position loop + slow integral
            e = (p_t - com) * [1, 0, 1]
            self.zi = np.clip(self.zi + 1.2 * e * dt, -2.0, 2.0)
            v_d = e * 0.6
            n = np.linalg.norm(v_d)
            if n > 0.10:
                v_d *= 0.10 / n
        else:
            self.zi *= 0.0
        # heading setpoint: rate limited (the tunnel pair's reach)
        dpsi = wrap(psi_d - self.psi_ref)
        rmax = math.radians(12.0) * dt
        self.psi_ref = wrap(self.psi_ref + float(np.clip(dpsi, -rmax, rmax)))
        self.v_d, self.psi_d = v_d, psi_d
        tdir = ydir(psi)
        ndir = np.cross(tdir, UP)
        u = v_d - CURRENT                                             # through the water
        ut, un = float(u @ tdir), float(u @ ndir)
        nn = sn.n
        Ft = nn * self.ct * (ut + abs(ut) * ut) + 10.0 * float((v_d - vc) @ tdir)
        Fn = nn * self.cn * (un + abs(un) * un) + 18.0 * float((v_d - vc) @ ndir)
        F = Ft * tdir + Fn * ndir + self.zi
        # yaw
        r_d = float(np.clip(0.8 * wrap(self.psi_ref - psi), -0.25, 0.25))
        xk = (np.arange(nn) - (nn - 1) / 2.0) * L
        M = self.cn * float(np.sum(xk ** 2 * r_d + np.abs(xk) ** 3 * r_d * abs(r_d))) + nn * sn.lam2 * r_d
        M += 4.0 * (r_d - r)
        cmd = self._alloc(pos, q, com, np.array([F[0], F[2], M]))
        sn.set_thrust(cmd)
        self.cmd_log = sn.thr_cmd.copy()
        # joint targets: straight in transit (+ a small bend into a turn), the ramped shape, the neck scan
        phi = self.shape_w * self.q_f + (1.0 - self.shape_w) * self.q_tr
        if self.mode in ("TRANSIT", "APPROACH"):
            k = float(np.clip(L * r_d / max(np.linalg.norm(v_d), 0.15), -math.radians(10), math.radians(10)))  # BEND_MAX
            phi = phi + k
        phi = phi.copy()
        phi[6:] += self.scan * np.array([0.5, 1.0])
        sn.set_shape(phi)

    def _alloc(self, pos, q, com, tau):
        """Least-norm thrust for (Fx, Fz, My) from the thrusters' actual positions and axes, clipped."""
        sn = self.sn
        R = sm.quat_to_R(q)
        B = np.zeros((3, 4))
        for j in range(4):
            k = sn.thr_link[j]
            a = R[k] @ sn.thr_axis[j]
            rr = pos[k] + R[k] @ sn.thr_pos[j] - com
            B[:, j] = [a[0], a[2], np.cross(rr, a)[1]]
        lo = np.array([t.t_min for t in sn.thrusters])
        hi = np.array([t.t_max for t in sn.thrusters])
        u = np.zeros(4)
        free = np.ones(4, bool)
        for _ in range(3):
            res = tau - B[:, ~free] @ u[~free]
            u[free] = np.linalg.pinv(B[:, free]) @ res
            sat = free & ((u < lo) | (u > hi))
            if not sat.any():
                break
            u = np.clip(u, lo, hi)
            free &= ~sat
            if not free.any():
                break
        return np.clip(u, lo, hi)

    # ---------------------------------------------------------------- stepping
    def step_frame(self):
        """One 1/30 s frame: SUB physics substeps, the controller at 60 Hz, the depth hold, the lamp."""
        for _ in range(SUB):
            if self.k % CTRL_EVERY == 0:
                self._control(DT * CTRL_EVERY)
            self.world.step(DT)
            self.t += DT
            self.k += 1
        pos, _ = self.sn.link_poses()
        # depth hold: critically damped follow of the altimeter reference (2 s)
        w = 6.64 / 2.0
        h = 1.0 / FPS
        yr = self._y_ref(pos)
        self.yd += (w * w * (yr - self.y) - 2.0 * w * self.yd) * h
        self.y += self.yd * h
        # lamp: constant, ONE planned dim that starts with the approach (log ramp over 5 s), constant after
        if self.t_app is None and self.mode != "TRANSIT":
            self.t_app = self.t
        u = 0.0 if self.t_app is None else smooth(0.0, 5.0, self.t - self.t_app)
        self.lamp = LAMP_HI * (LAMP_LO / LAMP_HI) ** u
        self.prop_ang += np.sign(self.sn.thr) * 2.6 * np.sqrt(np.abs(self.sn.thr)) * 2 * math.pi * h

    def advance(self, t_film):
        while self.t < t_film - 1e-9:
            self.step_frame()

    def joints(self):
        """Scene joint points (nose first, N+1) with the visual y and the eased neck pitch."""
        pos, q = self.sn.link_poses()
        tl = sm.quat_to_R(q) @ self.sn.t_loc
        return self._joints_from(pos, tl, self.y, self.pitch_w)

    def close(self):
        """Tear the PhysX world down (one PhysX foundation per process)."""
        import gc
        if self.sn is not None:
            self.sn.remove()
        self.sn = None
        self.world = None
        gc.collect()


def _unit_rows(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def _roty(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def clearances(D, j):
    """(rock, sleeve, pile) skin clearance, m."""
    P = np.array([j[i + 1] + (j[i] - j[i + 1]) * u for i in range(N) for u in np.linspace(0, 1, 10)])
    bed = P[:, 1] - ts.seabed_height(P[:, 0], P[:, 2]) - SKIN
    d, s = D.SL.nearest(P)
    sl = d - D.SL.radius(s) - SKIN
    pile = np.hypot(P[:, 0], P[:, 2]) - ts.PILE_R - 0.22 - SKIN
    return float(bed.min()), float(sl.min()), float(pile.min())


# ---------------------------------------------------------------------- record (no rendering)
def record(T=T_TOTAL, S=None, out=None):
    S = S or site_like()
    out = snake_out(S) if out is None else out
    D = SnakeDyn(S, out)
    rec = {k: [] for k in ("t", "com", "vc", "psi", "cmd", "thr", "q", "qd", "mode", "lamp", "nose", "dlook",
                           "clr", "amp", "links", "y", "psi_d", "vd", "track")}
    nf = int(round(T * FPS))
    for f in range(nf + 1):
        if f:
            D.step_frame()
        pos, q, com, vc, psi, r = D._state()
        j = D.joints()
        rec["t"].append(D.t); rec["com"].append(com); rec["vc"].append(vc); rec["psi"].append(psi)
        rec["cmd"].append(D.cmd_log.copy()); rec["thr"].append(D.sn.thr.copy())
        rec["q"].append(D.sn.joint_angles()); rec["qd"].append(np.asarray(D.sn.art.joint_velocities(), float))
        rec["mode"].append(D.mode); rec["lamp"].append(D.lamp); rec["nose"].append(j[0])
        rec["dlook"].append(float(np.linalg.norm(j[0] - D.look))); rec["clr"].append(clearances(D, j))
        rec["amp"].append(float(np.abs(D.sn._amp.x).max())); rec["links"].append(pos); rec["y"].append(D.y)
        rec["psi_d"].append(D.psi_ref); rec["vd"].append(D.v_d.copy() if D.v_d is not None else np.zeros(3))
        rec["track"].append(D._rail_project(com)[1])
    R = {k: (np.array(v) if k != "mode" else v) for k, v in rec.items()}
    pos, q = D.sn.link_poses()
    R["hash"] = hashlib.sha1(np.round(np.concatenate([pos.ravel(), q.ravel()]), 9).tobytes()).hexdigest()[:12]
    R["events"] = D.events
    R["D"] = D
    return R


def summarize(R):
    D = R["D"]
    t, md = R["t"], np.array(R["mode"])
    lim_hi = np.array([th.t_max for th in D.sn.thrusters])
    lim_lo = np.array([th.t_min for th in D.sn.thrusters])
    vg0 = np.linalg.norm(R["vc"][:, [0, 2]], axis=1)
    tr = np.isin(md, ["TRANSIT", "APPROACH"]) & (t > 4.0) & (vg0 > 0.15)
    vg = np.linalg.norm(R["vc"][:, [0, 2]], axis=1)
    vw = np.linalg.norm((R["vc"] - CURRENT)[:, [0, 2]], axis=1)
    crab = np.degrees([wrap(R["psi"][i] - yaw_of(R["vc"][i])) for i in range(len(t))])
    q = np.degrees(R["q"])
    print(f"events: {R['events']}; start s {D.rail_s[0]:.2f}, hold s {D.s_hold:.2f}")
    print(f"current {np.linalg.norm(CURRENT):.2f} m/s toward +X (45 deg against/across the sleeve)")
    print(f"transit (t>4 s, v>0.15): speed over ground {vg[tr].mean():.3f} m/s, through water {vw[tr].mean():.3f} m/s, "
          f"crab (heading - course) mean {crab[tr].mean():+.1f} deg, range {crab[tr].min():+.1f}..{crab[tr].max():+.1f}; "
          f"cross-track rms {np.sqrt(np.mean(R['track'][tr] ** 2)):.3f} m")
    print(f"transit body joints max |phi| {np.abs(q[tr][:, :6]).max():.2f} deg (all joints {np.abs(q[tr]).max():.2f}); gait amplitude max {R['amp'].max():.2e} rad")
    names = [th.name for th in D.sn.thrusters]
    hold = np.isin(md, ["HOLD", "SHAPE", "INSPECT"])
    for k, nm in enumerate(names):
        print(f"  {nm:13s} cmd peak {R['cmd'][:, k].max():+.2f} / {R['cmd'][:, k].min():+.2f} N "
              f"(limits {lim_lo[k]:+.1f}..{lim_hi[k]:+.1f}); hold mean {R['cmd'][hold, k].mean():+.2f}, "
              f"hold |max| {np.abs(R['cmd'][hold, k]).max():.2f}")
    t_ins = [e[0] for e in R["events"] if e[1] == "INSPECT"]
    ins = (md == "INSPECT") & (t > (t_ins[0] + 2.0 if t_ins else 1e9))
    if ins.any():
        c = R["com"][ins]
        e = np.linalg.norm((c - D.p_f)[:, [0, 2]], axis=1)
        print(f"station keeping (INSPECT from +2 s, COM vs setpoint): rms {np.sqrt(np.mean(e ** 2)):.3f} m, max {e.max():.3f} m; "
              f"nose motion std {np.linalg.norm(R['nose'][ins].std(0)):.3f} m")
        print(f"head (nose) to patch at hold: mean {R['dlook'][ins].mean():.3f} m, range {R['dlook'][ins].min():.3f}..{R['dlook'][ins].max():.3f}")
    sh = md == "SHAPE"
    if sh.any():
        qdv = np.degrees(np.abs(R["qd"][sh]))
        print(f"shape change: joint rates max {qdv.max():.1f} deg/s (body {qdv[:, :6].max():.1f}, neck {qdv[:, 6:].max():.1f}); final q {np.round(q[-1], 1)}")
    clr = R["clr"]
    print(f"clearance min: rock {clr[:, 0].min():.3f} sleeve {clr[:, 1].min():.3f} pile {clr[:, 2].min():.3f} m")
    lamp = R["lamp"]
    print(f"lamp {lamp[0]:.2f} -> {lamp[-1]:.2f}, largest frame step {100 * np.max(np.abs(np.diff(lamp)) / lamp[:-1]):.2f} %")
    print(f"final pose hash {R['hash']}")


def plot(R, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    D = R["D"]
    t = R["t"]
    fig = plt.figure(figsize=(16, 10))
    ax = fig.add_subplot(2, 3, 1)
    SLp = D.SL.P
    ax.plot(SLp[:, 0], SLp[:, 2], color="#8a5a2b", lw=6, alpha=0.5, label="sleeve")
    ax.plot(D.rail[:, 0], D.rail[:, 2], ":", color="grey", lw=1, label="rail")
    ax.plot(R["com"][:, 0], R["com"][:, 2], color="#1f77b4", label="COM")
    ax.plot(R["nose"][:, 0], R["nose"][:, 2], color="#d62728", lw=0.8, label="nose")
    for i in range(0, len(t), 60):
        P = R["links"][i]
        ax.plot(P[:, 0], P[:, 2], color="#e0a000", lw=2, alpha=0.6)
    ax.plot(*D.look[[0, 2]], "k*", ms=12, label="patch")
    c0 = R["com"][0]
    ax.annotate("", xy=(c0[0] - 0.6 + 0.6, c0[2] + 0.8), xytext=(c0[0] - 0.6, c0[2] + 0.8),
                arrowprops=dict(arrowstyle="->", color="teal", lw=2))
    ax.text(c0[0] - 0.6, c0[2] + 0.9, f"current {np.linalg.norm(CURRENT):.2f} m/s", color="teal")
    ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xlabel("x (m)"); ax.set_ylabel("z (m)")
    ax.set_title("top down (links every 2 s)"); ax.legend(fontsize=7, loc="lower left")
    x0, x1 = min(R["com"][:, 0].min(), D.look[0]) - 1, max(R["com"][:, 0].max(), D.look[0]) + 1
    z0, z1 = min(R["com"][:, 2].min(), D.look[2]) - 1, max(R["com"][:, 2].max(), D.look[2]) + 1
    ax.set_xlim(x0, x1); ax.set_ylim(z1, z0)

    def modes(a):
        for tt, nm in R["events"]:
            a.axvline(tt, color="grey", lw=0.6, ls="--")
            a.text(tt, a.get_ylim()[1], nm, fontsize=7, rotation=90, va="top")

    ax = fig.add_subplot(2, 3, 2)
    ax.plot(t, np.linalg.norm(R["vc"][:, [0, 2]], axis=1), label="over ground")
    ax.plot(t, np.linalg.norm((R["vc"] - CURRENT)[:, [0, 2]], axis=1), label="through water")
    crab = np.degrees([wrap(R["psi"][i] - yaw_of(R["vc"][i])) if np.linalg.norm(R["vc"][i]) > 0.03 else np.nan for i in range(len(t))])
    ax2 = ax.twinx(); ax2.plot(t, crab, color="grey", lw=0.7); ax2.set_ylabel("crab (deg)")
    ax.set_ylabel("speed (m/s)"); ax.legend(fontsize=7); ax.set_title("speed"); modes(ax)
    ax = fig.add_subplot(2, 3, 3)
    for k, th in enumerate(D.sn.thrusters):
        ax.plot(t, R["cmd"][:, k], label=th.name, lw=0.9)
    ax.axhline(4.0, color="k", lw=0.5, ls=":"); ax.axhline(-2.5, color="k", lw=0.5, ls=":"); ax.axhline(2.5, color="k", lw=0.5, ls=":")
    ax.set_ylabel("thrust cmd (N)"); ax.legend(fontsize=7); ax.set_title("thrusters (limits dotted)"); modes(ax)
    ax = fig.add_subplot(2, 3, 4)
    q = np.degrees(R["q"])
    for k in range(8):
        ax.plot(t, q[:, k], lw=0.9, label=f"j{k}" + (" neck" if k >= 6 else ""))
    ax.set_ylabel("joint angle (deg)"); ax.legend(fontsize=6, ncol=2); ax.set_title("joints (tail j0 .. head j7)"); modes(ax)
    ax = fig.add_subplot(2, 3, 5)
    ax.plot(t, R["dlook"], label="nose to patch")
    ax.plot(t, R["clr"][:, 0], label="rock clr")
    ax.plot(t, R["clr"][:, 1], label="sleeve clr")
    ax.set_ylim(0, 3); ax.set_ylabel("m"); ax.legend(fontsize=7); ax.set_title("distances"); modes(ax)
    ax = fig.add_subplot(2, 3, 6)
    md = np.array(R["mode"])
    ins = np.isin(md, ["HOLD", "SHAPE", "INSPECT"])
    if ins.any():
        e = (R["com"] - D.p_f) * [1, 0, 1]
        ax.plot(t[ins], 100 * e[ins, 0], label="COM err x (cm) vs final")
        ax.plot(t[ins], 100 * e[ins, 2], label="COM err z (cm)")
        ax.plot(t[ins], R["lamp"][ins] * 10, label="lamp x10")
    ax.plot(t, R["lamp"] * 10, color="grey", lw=0.6)
    ax.legend(fontsize=7); ax.set_title("hold error, lamp"); modes(ax)
    fig.suptitle(f"snake on its physics model: current {np.linalg.norm(CURRENT):.2f} m/s, thrusters only, gait amplitude 0; hash {R['hash']}")
    fig.tight_layout()
    d, b = os.path.split(path)
    tmp = os.path.join(d, "_tmp_" + b)
    fig.savefig(tmp, dpi=90)
    os.replace(tmp, path)
    print("wrote", path)


def next_ver(outdir, stem):
    k = 1
    while any(f.startswith(f"{stem}_") and f.endswith(f"v{k:02d}.png") or f == f"{stem}_v{k:02d}.mp4"
              for f in os.listdir(outdir)):
        k += 1
    return k



# ---------------------------------------------------------------------- the film API
def build_snake_dyn(renderer, S, F, J):
    """The film's entry point. Returns job with
         job.reset()          a fresh physics run from t = 0 (steady transit)
         job.step_to(tau)     steps the physics to physics time tau (monotonic; reset() to go back) and poses
                              the snake links, lamps, props and J.head_cam; returns the scene joints
         job.cam(tau)         (eye, target, fov) of the low three-quarter-front film camera; low-passed,
                              stateful: call once per rendered frame, in order
         job.T                mode start times (s) of the deterministic run, e.g. job.T["HOLD"]
         job.D                the SnakeDyn (numbers: D.mode, D.lamp, D.sn.thr, D.events)
       The marine snow is set to drift with CURRENT. Nothing else in the scene is touched."""
    V = F.vehicles
    me = S.motes.emitter
    me.wind = tp.Vector3(float(CURRENT[0]), 0.0, float(CURRENT[2]))
    S.motes.set_emitter(me)
    job = types.SimpleNamespace()
    links, lamps = V["snake_links"], V["snake_lamps"]
    props = add_props(links)
    head_cam = J.head_cam
    head_cam.far = 10.0                                      # visibility ~5 m: nothing beyond is legitimately visible
    head_cam.fov = HEAD_FOV
    head_cam.update_projection_matrix()
    out = V["snake_out"]
    dry = SnakeDyn(S, out, look=J.look)                     # one dry run for the mode times (deterministic)
    dry.advance(T_TOTAL)
    job.T = {nm: t for t, nm in dry.events}
    dry.close()
    job.D = None
    cam_state = {}

    def reset():
        if job.D is not None:
            job.D.close()
        job.D = SnakeDyn(S, out, look=J.look)
        cam_state.clear()

    def step_to(tau):
        D = job.D
        D.advance(tau)
        j = D.joints()
        fv.pose_snake(links, j)
        for sp in lamps:
            sp.intensity = D.lamp
        for g, sd, kind, idx in props:
            a = float(D.prop_ang[idx])
            if kind == "side":
                g.rotation.x = a
            else:
                g.rotation.z = a
        f0 = _unit(j[0] - j[1])
        r = _unit(np.cross(f0, UP))
        u0 = np.cross(r, f0)
        f = math.cos(CAM_TILT) * f0 - math.sin(CAM_TILT) * u0    # camera mount pitched down, fixed in the head
        u = np.cross(r, f)
        head_cam.position.set(*map(float, j[0] + 0.004 * f0))
        head_cam.quaternion.set(*map(float, js.quat_of(np.stack([r, u, -f], axis=1))))
        job.joints = j
        return j

    def cam(tau):
        """Low (0.5 m over the rock), three-quarter front: beyond the patch along the sleeve, a little
        to the open side, so the snake comes toward it and the patch sits in the foreground."""
        D = job.D
        j = job.joints
        t_s = _unit(D.SL.at(D.SL.s_td)[1] * [1, 0, 1])
        eye_hold = D.look + 1.45 * _unit(1.0 * t_s + 0.55 * D.out)
        s_h, _, th, _ = D._rail_project(j[0])
        eye_tr = j[0] * [1, 0, 1] + 1.7 * _unit(1.0 * th + 0.45 * D.out)
        w = smooth(0.0, 5.0, D.t - (job.T["APPROACH"] - 2.0))
        eye = (1 - w) * eye_tr + w * eye_hold
        eye[1] = float(bed_max(eye[0], eye[2], 0.25)) + 0.5
        head_ref = D._joints_at(D.q_f * D.shape_w, D.p_f, D.psi_f, D.pitch_w)[0]   # the SETPOINT's head: the robot's own motion stays visible
        tgt = (1 - w) * (0.6 * j[1] + 0.4 * j[4]) + w * (0.45 * D.look + 0.55 * head_ref + 0.0 * UP)
        tgt[1] = min(tgt[1], eye[1] - 0.15)
        a_e = 1.0 - math.exp(-(1.0 / FPS) / 1.0)
        a_t = 1.0 - math.exp(-(1.0 / FPS) / 0.7)
        if "e" not in cam_state:
            cam_state["e"], cam_state["t"] = eye, tgt
        cam_state["e"] = cam_state["e"] + a_e * (eye - cam_state["e"])
        cam_state["t"] = cam_state["t"] + a_t * (tgt - cam_state["t"])
        return cam_state["e"].copy(), cam_state["t"].copy(), 44.0

    job.reset, job.step_to, job.cam = reset, step_to, cam
    reset()
    step_to(0.0)
    return job

# ---------------------------------------------------------------------- render
def render(clip_t0, dur, outdir, ver):
    from demo_common import Encoder
    import turbine_fleet as tf
    W, H = 1920, 1080
    canvas = tp.Canvas("threepp - snake dyn", width=W, height=H, vsync=False, headless=True)
    renderer = tp.VulkanRenderer(canvas)
    renderer.tone_mapping = tp.ToneMapping.AgX
    renderer.tone_mapping_exposure = 1.4
    renderer.sun_angular_radius = 0.6
    renderer.bloom_intensity = 0.06
    renderer.auto_exposure = False
    S = ts.build_site(renderer, sheet=12000.0)
    F = tf.build_fleet(renderer, S)
    J = js.build_subsea(renderer, S, F)
    V = F.vehicles
    S.set_lamps((0, 0, 0), (1, 0, 0), False)
    V["rov"].visible = False                                # the snake's own light only
    for sp in V["rov_lights"]:
        sp.intensity = 0.0
    J.flood.intensity = 0.0
    if V["tether"] is not None:
        V["tether"].visible = False
    job = build_snake_dyn(renderer, S, F, J)
    camera = tp.PerspectiveCamera(44.0, W / H, 0.03, 20000.0)

    def aim(eye, tgt, fov):
        camera.fov = fov
        camera.update_projection_matrix()
        camera.position.set(*map(float, eye))
        camera.look_at(tp.Vector3(*map(float, tgt)))

    T0 = ts.T_SHOT
    aim(np.array([0.0, -12.0, 12.0]), np.array([0.0, -12.0, 0.0]), 50.0)
    for k in range(int(3.0 * FPS) + 1):                    # settle the boats, first ocean frame
        t = T0 - 3.0 + k / FPS
        renderer.sim_time = t
        S.update(t, camera, True, 1.0 / FPS)
        renderer.render(S.scene, camera)
        F.update(t, 1.0 / FPS)
    if V["tether"] is not None:
        V["tether"].visible = False
    job.step_to(clip_t0)                                    # the early transit, unrendered (real time from here)
    D = job.D
    head_cam = J.head_cam
    stats = dict(clip=[])
    pose = lambda: job.step_to(D.t)
    film_cam = lambda j, tau: job.cam(tau)
    j = pose()
    PIP = (W - 24 - 480, 24, 480, 270)
    renderer.render(S.scene, camera)
    view = renderer.add_view(head_cam, PIP[2], PIP[3])
    if view:
        renderer.set_view_display_rect(view, *PIP)
    stem = "snake_dyn"
    path = os.path.join(outdir, f"{stem}_v{ver:02d}.mp4")
    tmp = os.path.join(outdir, f"_tmp_{stem}_v{ver:02d}.mp4")
    enc = Encoder(tmp, W, H, FPS, crf=18, preset="medium", faststart=True)
    n = int(round(dur * FPS))
    pick = set(np.linspace(0, n - 1, 8).round().astype(int).tolist())
    t_sh = [e[0] for e in D.events]
    keys = {}
    tiles, t1 = [], time.perf_counter()
    paste = None
    for k in range(n):
        if k:
            D.step_frame()
        tau = D.t
        t = T0 + tau
        j = pose()
        eye, tgt, fov = film_cam(j, tau)
        aim(eye, tgt, fov)
        if k == 0:
            S.update(t, camera, True, 0.0)
            S.settle_snow(camera, t)
        renderer.sim_time = t
        S.update(t, camera, True, 1.0 / FPS)
        renderer.sim_time = t
        renderer.render(S.scene, camera)
        rgb = renderer.read_pixels()
        if view:
            x, y, w_, h_ = PIP
            if paste is None:
                inset = renderer.read_view_rgb_pixels(view)
                paste = inset.size and float(np.abs(rgb[y:y + h_, x:x + w_].astype(int) - inset.astype(int)).mean()) > 12.0
            if paste:
                rgb = rgb.copy()
                rgb[y:y + h_, x:x + w_] = renderer.read_view_rgb_pixels(view)
            rgb = js.pip_dress(rgb, x, y, w_, h_, "SNAKE CAM")
        enc.send(rgb)
        stats["clip"].append(float((rgb.max(axis=2) >= 250).mean()))
        if k in pick:
            tiles.append(rgb)
        nm = None
        if k == int(2.0 * FPS):
            nm = "transit"
        if D.mode == "SHAPE" and "shape" not in keys and D.t - D.t_mode > 1.25:
            nm = "shape"
        if k == n - 1:
            nm = "hold"
        if nm and nm not in keys:
            keys[nm] = k
            js.save_png(rgb, os.path.join(outdir, f"{stem}_{nm}_v{ver:02d}.png"))
        F.update(t, 1.0 / FPS)
        if V["tether"] is not None:
            V["tether"].visible = False
        if k % 60 == 0:
            print(f"  frame {k}/{n} t {D.t:.2f} {D.mode} ({time.perf_counter() - t1:.0f} s)", flush=True)
    rc = enc.close()
    if rc == 0:
        os.replace(tmp, path)
        print(f"wrote {path} ({n} frames, {time.perf_counter() - t1:.0f} s)")
    js.sheet(tiles, os.path.join(outdir, f"{stem}_sheet_v{ver:02d}.png"))
    cl = np.array(stats["clip"])
    print(f"clipped (>=250) pixel fraction: max {cl.max() * 100:.3f} %, mean {cl.mean() * 100:.3f} %; keys {keys}")
    pos, q = D.sn.link_poses()
    print("render final hash", hashlib.sha1(np.round(np.concatenate([pos.ravel(), q.ravel()]), 9).tobytes()).hexdigest()[:12])


def add_props(links):
    """snake_netpen's side props (3 blades, pitched) + a rotor in each tunnel; spun from the thrust."""
    pm = js.standard_material(0x1c1f22, roughness=0.45, metalness=0.4) if hasattr(js, "standard_material") else None
    out = []
    for sd, idx in ((-1.0, 0), (1.0, 1)):
        pg = tp.Group()
        pg.position.set(0.012, 0.0, sd * fv.SIDE_Y)
        for k in range(3):
            b = tp.Mesh(tp.BoxGeometry(0.0025, 0.021, 0.011), pm)
            hd = tp.Group()
            hd.rotation.x = 2.0 * math.pi * k / 3.0
            b.position.y = 0.0135
            b.rotation.y = math.radians(35.0)
            hd.add(b)
            pg.add(hd)
        links[fv.SCENE_SIDE_THRUSTERS].add(pg)
        out.append((pg, sd, "side", idx))
    for li, idx in ((fv.SCENE_TUNNEL_BOW, 2), (fv.SCENE_TUNNEL_STERN, 3)):
        pg = tp.Group()
        for k in range(3):
            b = tp.Mesh(tp.BoxGeometry(0.018, 0.0022, 0.008), pm)
            hd = tp.Group()
            hd.rotation.z = 2.0 * math.pi * k / 3.0
            b.position.x = 0.010
            b.rotation.x = math.radians(35.0)
            hd.add(b)
            pg.add(hd)
        links[li].add(pg)
        out.append((pg, 0.0, "tunnel", idx))
    return out


def main():
    from demo_common import cli_arg
    outdir = cli_arg("--outdir", ts.out_dir("jobs"), str)
    os.makedirs(outdir, exist_ok=True)
    if "--traj" in sys.argv:
        R = record()
        summarize(R)
        ver = next_ver(outdir, "snake_dyn")
        plot(R, os.path.join(outdir, f"snake_dyn_traj_v{ver:02d}.png"))
        R["D"].close()
        if "--twice" in sys.argv:
            R2 = record()
            R2["D"].close()
            print("determinism: run 1", R["hash"], "run 2", R2["hash"], "same" if R["hash"] == R2["hash"] else "DIFFERENT")
    if "--render" in sys.argv:
        ver = cli_arg("--ver", 0, int) or next_ver(outdir, "snake_dyn")
        render(cli_arg("--t0", 12.0, float), cli_arg("--dur", 18.0, float), outdir, ver)


if __name__ == "__main__":
    main()
