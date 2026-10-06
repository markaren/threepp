"""
remus_rig -- the REMUS 100 AUV: its published 6-DOF model, an autopilot and a guidance law on
top of it, and its 3D model posed from the model's state.

Four things, and nothing scene-specific:

  * ``Remus``     -- the model of T. Prestero, "Verification of a Six-Degree of Freedom Simulation
                     Model for the REMUS Autonomous Underwater Vehicle", S.M. thesis, MIT / WHOI
                     Joint Program, 2001, http://hdl.handle.net/1721.1/65068: its rigid body with
                     the centre of gravity below the centre of buoyancy (3.8), the hydrostatics
                     (4.2), the force and moment sums (4.49) with the coefficients of its Appendix B
                     (axial and crossflow drag, added mass and its cross terms, body lift and its
                     moment, fin lift, the propeller's thrust and torque), solved through the mass
                     matrix (6.8)/(6.9), the kinematics (3.1)-(3.5), fourth-order Runge-Kutta
                     (6.16)-(6.18) at a fixed step. Numbers in auv/remus100_spec.json.
  * ``Autopilot`` -- OURS, not the thesis's: stern planes on pitch, pitch on depth, rudder on
                     heading or course, the propeller speed from the commanded speed; gains from
                     the model's own linearisation. ``command(course=..., depth=..., speed=...)`` is
                     the hook a guidance law drives (the X8's, with depth for altitude).
  * ``LOS``       -- OURS: line-of-sight guidance on a polyline of waypoints (x8_rig.LOS's law).
  * ``Visual``    -- remus100.glb (auv/build_remus_blender.py) posed from a Remus: the hull from the
                     state, the stern-plane and rudder pairs by their angles, the propeller by its
                     speed.

What the thesis does not give is kept out of its equations and marked OURS or ASSUMED where it
appears: g, the fin servo and propeller-speed lags, the thrust and torque away from 1500 RPM, the
current, and the guards outside the model's range of validity (``Remus.guards``, ``Remus.flags``).

Nothing here imports threepp: ``Visual`` is handed the loaded model's root node.

FRAMES
------
Inside the model everything is the thesis's: NED position (N, E, D), D the depth; body axes x
forward, y starboard, z down, origin at the centre of buoyancy; attitude (phi, theta, psi), psi
from north toward east. The one mapping to threepp's world (x east, y up, z south; the X8 rig's
and the drone rig's) is at the boundary, in ``ned_to_world``,
``Remus.world_position`` and ``Remus.world_rotation``::

    world = (E, h0 - D, -N)            h0: the world height of the NED origin (the sea surface)

so psi = 0 points the nose along world -z (north) and psi = 90 deg along +x (east). The .glb's own
frame is X forward, Y up, Z starboard (the boats' and the X8's): a body vector (x, y, z) is
(x, -z, y) in it.

THE CURRENT (OURS)
------------------
The thesis assumes still water (its Section 1.5.1). A current enters here the standard way for a
current that is steady and uniform in the earth frame (Fossen, Handbook of Marine Craft
Hydrodynamics and Motion Control, 2011, Section 10.3): the velocity state (u, v, w) is the
velocity through the water, nu_r = nu - nu_c, and every term of the thesis's equations takes it,
while the kinematics add the current back::

    M nu_r' = tau(nu_r, eta, delta)          (6.9), every term on (u_r, v_r, w_r, p, q, r)
    (N, E, D)' = J1(eta2) nu_r + V_c          V_c the current, NED

The hydrodynamic terms (drag, lift, fin lift, added mass and its cross terms) depend on the
motion relative to the water by their derivation. The rigid-body terms take nu_r exactly, not as
an approximation: with V_c constant in NED its body-frame components change only by the
rotation, nu_c' = -omega x nu_c, so nu' + omega x nu = nu_r' + omega x nu_r, and the same in
m r_G x (nu' + omega x nu). The hydrostatics do not depend on velocity. A callable current is
taken as locally steady and uniform (its rate of change along the path is not added). With no
current the state is the thesis's.

TYPICAL USE
-----------
::

    from remus_rig import Remus, Autopilot, LOS, Visual, model_path

    auv = Remus(current=(0.0, 0.3, 0.0))            # 0.3 m/s setting east
    ap = Autopilot(auv)                              # the spec's 1.75 m/s
    auv.trim(ap.rpm, n=0.0, e=0.0, d=5.0, course=0.0)
    los = LOS([(0, 0), (150, 0), (150, 100)], auv=auv)
    while ...:
        vg = auv.ground_velocity()
        ap.command(course=los.update(auv.n, auv.e, math.hypot(vg[0], vg[1])), depth=5.0)
        ap.update()                                  # fin and propeller commands
        auv.step()                                   # one fixed step, auv.dt
    visual = Visual(tp.GLTFLoader().load(model_path()).scene)
    visual.pose(auv, h0=0.0)
"""
import json
import math
import os

import numpy as np

AUV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "auv")
SPEC_PATH = os.path.join(AUV_DIR, "remus100_spec.json")
TWO_PI = 2.0 * math.pi


def load_spec(path=None):
    with open(path or SPEC_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def model_path():
    """The generated remus100.glb, or None where it has not been built."""
    p = os.path.join(AUV_DIR, "remus100.glb")
    return p if os.path.isfile(p) else None


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# --------------------------------------------------------------------------- #
#  The frames
# --------------------------------------------------------------------------- #
C_WN = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])    # NED vector -> world (x east, y up, z south)
C_BM = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])    # model (X fwd, Y up, Z stbd) -> body (FRD)


def ned_to_world(n, e, d, h0=0.0):
    return (e, h0 - d, -n)


def world_to_ned(x, y, z, h0=0.0):
    return (-z, x, h0 - y)


def rot_nb(phi, theta, psi):
    """J1(eta2) of eq. (3.2): body -> NED, zyx Euler angles."""
    cf, sf = math.cos(phi), math.sin(phi)
    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(psi), math.sin(psi)
    return np.array([[ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp],
                     [ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp],
                     [-st, sf * ct, cf * ct]])


def quat_of(R):
    """(x, y, z, w) of a rotation matrix."""
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        return ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2.0
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    q[3] = (R[k, j] - R[j, k]) / s
    return tuple(q)


def quat_mul(a, b):
    """Hamilton product of (x, y, z, w) quaternions: the rotation b, then a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz)


# --------------------------------------------------------------------------- #
#  The vehicle: the thesis's model
# --------------------------------------------------------------------------- #
# Full state, in this order. The first 12 are the thesis's (its x of eq. 6.11, reordered as the
# X8's): position, velocity through the water and angular rate in body axes, the Euler angles.
# Then the actuators: the stern-plane and rudder angles (rad) and the propeller speed (RPM).
STATE = ("n", "e", "d", "u", "v", "w", "phi", "theta", "psi", "p", "q", "r", "stern", "rudder", "rpm")
N_RIGID = 12

# The coefficient names of Appendix B, in the order the force sums use them.
COEFFS = ("Xuu", "Xud", "Xwq", "Xqq", "Xvr", "Xrr",
          "Yvv", "Yrr", "Yuv", "Yvd", "Yrd", "Yur", "Ywp", "Ypq", "Yuudr",
          "Zww", "Zqq", "Zuw", "Zwd", "Zqd", "Zuq", "Zvp", "Zrp", "Zuuds",
          "Kpp", "Kpd",
          "Mww", "Mqq", "Muw", "Mwd", "Mqd", "Muq", "Mvp", "Mrp", "Muuds",
          "Nvv", "Nrr", "Nuv", "Nvd", "Nrd", "Nur", "Nwp", "Npq", "Nuudr")


class Remus:
    """The REMUS vehicle as the thesis models it.

    spec     remus100_spec.json as a dict (default: load it)
    dt       the fixed step, s (default 1/120: 60, 40, 30 and 24 frames a second are whole
             numbers of steps)
    current  NED current, m/s: a 3-tuple, or a callable (t, n, e, d) -> (cn, ce, cd)
    guards   the envelope guards on (OURS; ``guards=False`` runs the thesis's equations bare)
    fin_tau, rpm_tau   override the spec's actuator lags (0: the command is the state)
    coeffs   override coefficients, {name: value}, for comparisons; the spec is not changed

    The state is ``self.x`` (STATE order); ``self.cmd`` holds the stern-plane and rudder commands
    (rad) and the propeller command (RPM) that the next step() applies. ``self.out`` is what the
    last step evaluated at its start (forces by origin, angles of attack, guards)."""

    def __init__(self, spec=None, dt=1.0 / 120.0, current=(0.0, 0.0, 0.0), guards=True, fin_tau=None,
                 rpm_tau=None, coeffs=None):
        self.spec = spec = load_spec() if spec is None else spec
        P, C, PR, F, ENV = spec["physical"], spec["coefficients"], spec["propulsion"], spec["fins"], spec["envelope"]
        self.c = {k: float(C[k]) for k in COEFFS}
        if coeffs:
            self.c.update({k: float(v) for k, v in coeffs.items()})
        self.W, self.B, self.g = P["W"], P["B"], P["g"]
        self.m = self.W / self.g
        self.rg = (P["xg"], P["yg"], P["zg"])
        self.rb = (P["xb"], P["yb"], P["zb"])
        self.Ixx, self.Iyy, self.Izz = P["Ixx"], P["Iyy"], P["Izz"]
        self.Xprop, self.Kprop, self.rpm_ref, self.u_ref = PR["Xprop"], PR["Kprop"], PR["rpm_ref"], PR["u_ref"]
        self.rpm_max = PR["rpm_max"]
        self.rpm_tau = PR["rpm_tau"] if rpm_tau is None else rpm_tau
        self.fin_tau = F["tau"] if fin_tau is None else fin_tau
        self.delta_max = math.radians(F["delta_max_deg"])
        self.x_fin = F["x_fin"]
        self.guards_on = guards
        self.stall = math.radians(ENV["fin_stall_deg"])
        self.u_min = ENV["u_min"]
        self.cos_min = math.cos(math.radians(ENV["pitch_max_deg"]))
        self.speed_range = tuple(ENV["speed_range"])
        self.alpha_max = math.radians(ENV["alpha_max_deg"])
        self.guards = {"stern_stall": 0, "rudder_stall": 0, "pitch": 0}
        self.flags = {"speed": 0, "alpha": 0, "beta": 0}
        self.Minv = np.linalg.inv(self.mass_matrix())
        self.dt = dt
        self.current = current
        self.x = [0.0] * len(STATE)
        self.cmd = [0.0, 0.0, 0.0]
        self.t = 0.0
        self.steps = 0
        self.prop_angle = 0.0             # the drawn propeller's angle: the speed integrated, for Visual
        self.out = {}
        self._out_at = -1

    # ---- named access
    def __getattr__(self, name):
        if name in _INDEX:
            return self.x[_INDEX[name]]
        raise AttributeError(name)

    @property
    def depth(self):
        return self.x[2]

    def current_ned(self, t=None, n=None, e=None, d=None):
        c = self.current
        if callable(c):
            x = self.x
            return c(self.t if t is None else t, x[0] if n is None else n, x[1] if e is None else e,
                     x[2] if d is None else d)
        return c

    # ---- the thesis's equations
    def mass_matrix(self):
        """The rigid body's and the added mass's acceleration terms, eq. (6.8)."""
        m, (xg, yg, zg), c = self.m, self.rg, self.c
        return np.array([
            [m - c["Xud"], 0.0, 0.0, 0.0, m * zg, -m * yg],
            [0.0, m - c["Yvd"], 0.0, -m * zg, 0.0, m * xg - c["Yrd"]],
            [0.0, 0.0, m - c["Zwd"], m * yg, -m * xg - c["Zqd"], 0.0],
            [0.0, -m * zg, m * yg, self.Ixx - c["Kpd"], 0.0, 0.0],
            [m * zg, 0.0, -m * xg - c["Mwd"], 0.0, self.Iyy - c["Mqd"], 0.0],
            [-m * yg, m * xg - c["Nvd"], 0.0, 0.0, 0.0, self.Izz - c["Nrd"]]])

    def thrust_torque(self, rpm):
        """The propeller's thrust and torque on the hull: the thesis's Xprop and Kprop at 1500
        RPM (Table C.9), scaled by (n / 1500)^2 away from it (ASSUMED, spec `propulsion.law`)."""
        k = rpm * abs(rpm) / (self.rpm_ref * self.rpm_ref)
        return self.Xprop * k, self.Kprop * k

    def forces(self, nu, eta2, ds, dr, rpm, out=None):
        """The six external force and moment sums (6.7)'s right-hand sides, the rigid body's
        velocity terms moved to them, at the water-relative velocity nu = (u, v, w, p, q, r), the
        attitude eta2 = (phi, theta, psi), the fin angles ds, dr (rad) and the propeller speed
        (RPM). Returns [X, Y, Z, K, M, N]; out gets each origin's share."""
        u, v, w, p, q, r = nu
        phi, th = eta2[0], eta2[1]
        c = self.c
        m, (xg, yg, zg), (xb, yb, zb) = self.m, self.rg, self.rb
        W, B = self.W, self.B
        # (4.1)-(4.3) hydrostatics: f_G - f_B and r_G x f_G - r_B x f_B, with f = J1^-1 (0, 0, W)
        cf, sf = math.cos(phi), math.sin(phi)
        ct, st = math.cos(th), math.sin(th)
        fx, fy, fz = -st, ct * sf, ct * cf
        wb = W - B
        ax, ay, az = W * xg - B * xb, W * yg - B * yb, W * zg - B * zb
        hs = (wb * fx, wb * fy, wb * fz, ay * fz - az * fy, az * fx - ax * fz, ax * fy - ay * fx)
        # (4.46), (4.47) the propeller
        Xp, Kp = self.thrust_torque(rpm)
        # (4.49) the hydrodynamic sums: crossflow and axial drag, body and fin lift, added-mass
        # cross terms, fin lift, rolling drag, as Appendix B combines them
        uu = u * u
        X = c["Xuu"] * u * abs(u) + c["Xwq"] * w * q + c["Xqq"] * q * q + c["Xvr"] * v * r + c["Xrr"] * r * r + Xp
        Y = (c["Yvv"] * v * abs(v) + c["Yrr"] * r * abs(r) + c["Yuv"] * u * v + c["Ywp"] * w * p + c["Yur"] * u * r
             + c["Ypq"] * p * q + c["Yuudr"] * uu * dr)
        Z = (c["Zww"] * w * abs(w) + c["Zqq"] * q * abs(q) + c["Zuw"] * u * w + c["Zuq"] * u * q + c["Zvp"] * v * p
             + c["Zrp"] * r * p + c["Zuuds"] * uu * ds)
        K = c["Kpp"] * p * abs(p) + Kp
        M = (c["Mww"] * w * abs(w) + c["Mqq"] * q * abs(q) + c["Muw"] * u * w + c["Muq"] * u * q + c["Mvp"] * v * p
             + c["Mrp"] * r * p + c["Muuds"] * uu * ds)
        N = (c["Nvv"] * v * abs(v) + c["Nrr"] * r * abs(r) + c["Nuv"] * u * v + c["Nur"] * u * r + c["Nwp"] * w * p
             + c["Npq"] * p * q + c["Nuudr"] * uu * dr)
        # guard (OURS): the fin pairs' lift held at its value at the stall angle. A pair's lift is
        # Yuudr u^2 (dr - (v + x_fin r) / u) and Zuuds u^2 (ds + (w - x_fin q) / u), eqs.
        # (4.41)-(4.44) with Table C.10's fin coefficients; only the excess is taken off.
        g_s = g_r = False
        a_s = a_r = 0.0
        if u > self.u_min:
            xf = self.x_fin
            a_r = dr - (v + xf * r) / u
            a_s = ds + (w - xf * q) / u
            if self.guards_on:
                lim = self.stall
                if abs(a_r) > lim:
                    dY = c["Yuudr"] * uu * (math.copysign(lim, a_r) - a_r)
                    Y += dY
                    N += xf * dY
                    g_r = True
                if abs(a_s) > lim:
                    dZ = c["Zuuds"] * uu * (math.copysign(lim, a_s) - a_s)
                    Z += dZ
                    M -= xf * dZ
                    g_s = True
        # (3.8) the rigid body's velocity terms: m (omega x nu1 + omega x (omega x r_G)) and
        # omega x (I omega) + m r_G x (omega x nu1), taken to the right-hand side
        cx, cy, cz = q * w - r * v, r * u - p * w, p * v - q * u
        rr = p * xg + q * yg + r * zg
        w2 = p * p + q * q + r * r
        rb_f = (m * (cx + p * rr - xg * w2), m * (cy + q * rr - yg * w2), m * (cz + r * rr - zg * w2))
        rb_m = ((self.Izz - self.Iyy) * q * r + m * (yg * cz - zg * cy),
                (self.Ixx - self.Izz) * r * p + m * (zg * cx - xg * cz),
                (self.Iyy - self.Ixx) * p * q + m * (xg * cy - yg * cx))
        tau = [X + hs[0] - rb_f[0], Y + hs[1] - rb_f[1], Z + hs[2] - rb_f[2],
               K + hs[3] - rb_m[0], M + hs[4] - rb_m[1], N + hs[5] - rb_m[2]]
        if out is not None:
            out.update(hydro=(X, Y, Z, K, M, N), hydrostatic=hs, rigid=rb_f + rb_m, tau=tuple(tau),
                       thrust=Xp, prop_torque=Kp, alpha_s=a_s, alpha_r=a_r, g_stern=g_s, g_rudder=g_r)
        return tau

    def rigid_rates(self, xr, ds, dr, rpm, vc=(0.0, 0.0, 0.0), out=None):
        """d/dt of the 12 states xr = (n, e, d, u, v, w, phi, theta, psi, p, q, r) for the fin
        angles ds, dr (rad), the propeller speed (RPM) and the NED current vc: (6.9) and the
        kinematics (3.1), (3.4), the current added to the position's rate."""
        n_, e_, d_, u, v, w, phi, th, psi, p, q, r = xr
        tau = self.forces((u, v, w, p, q, r), (phi, th, psi), ds, dr, rpm, out)
        Mi = self.Minv
        acc = [Mi[i, 0] * tau[0] + Mi[i, 1] * tau[1] + Mi[i, 2] * tau[2] + Mi[i, 3] * tau[3]
               + Mi[i, 4] * tau[4] + Mi[i, 5] * tau[5] for i in range(6)]
        cf, sf = math.cos(phi), math.sin(phi)
        ct, st = math.cos(th), math.sin(th)
        cp, sp = math.cos(psi), math.sin(psi)
        ndot = ct * cp * u + (sf * st * cp - cf * sp) * v + (cf * st * cp + sf * sp) * w + vc[0]
        edot = ct * sp * u + (sf * st * sp + cf * cp) * v + (cf * st * sp - sf * cp) * w + vc[1]
        ddot = -st * u + sf * ct * v + cf * ct * w + vc[2]
        g_p = abs(ct) < self.cos_min
        if g_p and self.guards_on:                        # guard: (3.5) is singular at theta = +-90 deg
            ct = math.copysign(self.cos_min, ct)
        phidot = p + (sf * q + cf * r) * st / ct
        thdot = cf * q - sf * r
        psidot = (sf * q + cf * r) / ct
        if out is not None:
            out["g_pitch"] = g_p
        return [ndot, edot, ddot] + acc[:3] + [phidot, thdot, psidot] + acc[3:]

    def _vc(self, x, t):
        c = self.current_ned(t, x[0], x[1], x[2])
        return (float(c[0]), float(c[1]), float(c[2]))

    def rates(self, x, c, t):
        """d/dt of the full state under the commands c = (stern, rudder, rpm) held over the step:
        the rigid body, then the actuators (first-order lags, ASSUMED; with a zero time constant
        the state is set to the command at the start of the step and held)."""
        dx = self.rigid_rates(x[:N_RIGID], x[12], x[13], x[14], self._vc(x, t))
        ft, rt = self.fin_tau, self.rpm_tau
        dx += [(c[0] - x[12]) / ft if ft > 0.0 else 0.0,
               (c[1] - x[13]) / ft if ft > 0.0 else 0.0,
               (c[2] - x[14]) / rt if rt > 0.0 else 0.0]
        return dx

    def evaluate(self):
        """Forces, angles and the rest at the current state (fills self.out)."""
        x = self.x
        out = {}
        dx = self.rigid_rates(x[:N_RIGID], x[12], x[13], x[14], self._vc(x, self.t), out)
        u, v, w = x[3], x[4], x[5]
        U = math.sqrt(u * u + v * v + w * w)
        out.update(U=U, alpha=math.atan2(w, u), beta=math.asin(clamp(v / U, -1.0, 1.0)) if U > 1e-9 else 0.0,
                   current=self._vc(x, self.t), stern=x[12], rudder=x[13], rpm=x[14], acc=dx[3:6] + dx[9:12])
        self.out = out
        self._out_at = self.steps
        return out

    # ---- stepping
    def _clip_cmd(self):
        lim = self.delta_max
        return (clamp(self.cmd[0], -lim, lim), clamp(self.cmd[1], -lim, lim), clamp(self.cmd[2], 0.0, self.rpm_max))

    def step(self):
        """One fixed step: the commands clipped (the thesis's +-13.6 deg, Table 2.2), RK4 over dt
        (6.16)-(6.18) with the commands held, the guards' tally."""
        dt = self.dt
        c = self._clip_cmd()
        if self.fin_tau <= 0.0:
            self.x[12], self.x[13] = c[0], c[1]
        if self.rpm_tau <= 0.0:
            self.x[14] = c[2]
        out = self.out if self._out_at == self.steps else self.evaluate()
        if out["g_stern"]:
            self.guards["stern_stall"] += 1
        if out["g_rudder"]:
            self.guards["rudder_stall"] += 1
        if out["g_pitch"]:
            self.guards["pitch"] += 1
        if not (self.speed_range[0] <= out["U"] <= self.speed_range[1]):
            self.flags["speed"] += 1
        if abs(out["alpha"]) > self.alpha_max:
            self.flags["alpha"] += 1
        if abs(out["beta"]) > self.alpha_max:
            self.flags["beta"] += 1
        x, t = self.x, self.t
        k1 = self.rates(x, c, t)
        x2 = [a + 0.5 * dt * b for a, b in zip(x, k1)]
        k2 = self.rates(x2, c, t + 0.5 * dt)
        x3 = [a + 0.5 * dt * b for a, b in zip(x, k2)]
        k3 = self.rates(x3, c, t + 0.5 * dt)
        x4 = [a + dt * b for a, b in zip(x, k3)]
        k4 = self.rates(x4, c, t + dt)
        self.x = [a + dt / 6.0 * (b1 + 2.0 * b2 + 2.0 * b3 + b4) for a, b1, b2, b3, b4 in zip(x, k1, k2, k3, k4)]
        self.prop_angle = (self.prop_angle + self.x[14] / 60.0 * TWO_PI * dt) % TWO_PI
        self.t = t + dt
        self.steps += 1

    def run(self, seconds, before_step=None):
        """Step for `seconds`; before_step(self) runs before every step (an autopilot's update)."""
        for _ in range(int(round(seconds / self.dt))):
            if before_step is not None:
                before_step(self)
            self.step()

    # ---- setting a state
    def set_state(self, n=0.0, e=0.0, d=0.0, u=1.54, v=0.0, w=0.0, phi=0.0, theta=0.0, psi=0.0,
                  p=0.0, q=0.0, r=0.0, stern=0.0, rudder=0.0, rpm=1500.0):
        """The state (u, v, w through the water), with the actuators at rest on the given angles
        and speed and the commands equal to them."""
        self.x = [n, e, d, u, v, w, phi, theta, psi, p, q, r, stern, rudder, rpm]
        self.cmd = [stern, rudder, rpm]
        self._out_at = -1
        self.evaluate()

    def steady_speed(self, rpm):
        """The straight-line speed at which the propeller's thrust balances the axial drag,
        Xprop (n / 1500)^2 = -Xuu u|u| (the thesis's 1.544 m/s at 1500 RPM)."""
        T, _ = self.thrust_torque(rpm)
        return math.copysign(math.sqrt(abs(T) / -self.c["Xuu"]), T)

    def rpm_for(self, speed):
        """The propeller speed whose steady straight-line speed is `speed` (the inverse of the above)."""
        return math.copysign(self.rpm_ref * math.sqrt(abs(speed) * abs(speed) * -self.c["Xuu"] / self.Xprop), speed)

    def trim(self, rpm=1500.0, n=0.0, e=0.0, d=0.0, psi=0.0, apply=True, iters=40, course=None):
        """Straight motion at constant depth through still water at propeller speed `rpm`: solves
        the six equations of motion (p = q = r = 0) and a zero depth rate for u, v, w, phi, theta
        and the stern-plane and rudder angles. (REMUS needs both: it is 7 N buoyant, and the
        propeller's torque rolls it to -5.3 deg, which turns part of the buoyancy and the fin
        forces sideways.) With apply, the vehicle is put there at heading psi or, given a course,
        at the heading whose ground track in the current is that course. Returns the solution."""
        z = np.array([self.steady_speed(rpm), 0.0, 0.0, math.radians(-5.3), 0.0, 0.0, 0.0])
        rest = [0.0, 0.0, 0.0]

        def resid(z):
            u, v, w, ph, th, ds, dr = z
            f = self.rigid_rates([0.0, 0.0, d, u, v, w, ph, th, 0.0] + rest, ds, dr, rpm)
            return np.array(f[3:6] + f[9:12] + [f[2]])
        for _ in range(iters):
            r0 = resid(z)
            J = np.zeros((7, 7))
            for k in range(7):
                hk = 1e-7 * max(1.0, abs(z[k]))
                zp, zm = z.copy(), z.copy()
                zp[k] += hk
                zm[k] -= hk
                J[:, k] = (resid(zp) - resid(zm)) / (2.0 * hk)
            step = np.linalg.solve(J, -r0)
            z = z + step
            if np.max(np.abs(step)) < 1e-13:
                break
        u, v, w, ph, th, ds, dr = (float(a) for a in z)
        sol = {"rpm": rpm, "u": u, "v": v, "w": w, "phi": ph, "theta": th, "stern": ds, "rudder": dr,
               "alpha": math.atan2(w, u), "residual": float(np.max(np.abs(resid(z))))}
        if apply:
            def place(psi):
                self.set_state(n, e, d, u, v, w, ph, th, psi, stern=ds, rudder=dr, rpm=rpm)
            if course is not None:
                psi = course
                for _ in range(8):
                    place(psi)
                    psi += wrap(course - self.course())
            place(psi)
        return sol

    # ---- what a scene reads
    def euler(self):
        return self.x[6], self.x[7], self.x[8]

    def water_velocity_ned(self):
        """The velocity through the water, NED: J1 nu_r."""
        x = self.x
        return rot_nb(x[6], x[7], x[8]) @ np.array(x[3:6])

    def ground_velocity(self):
        """NED velocity over the ground: J1 nu_r + V_c."""
        return self.water_velocity_ned() + np.asarray(self.current_ned(), float)

    def course(self):
        """The ground track's direction chi, from north toward east (rad)."""
        v = self.ground_velocity()
        return math.atan2(v[1], v[0])

    def world_position(self, h0=0.0):
        return ned_to_world(self.x[0], self.x[1], self.x[2], h0)

    def world_rotation(self):
        """The .glb's rotation in the world: model -> body -> NED -> world."""
        x = self.x
        return C_WN @ rot_nb(x[6], x[7], x[8]) @ C_BM

    def energy(self):
        """Kinetic energy of the body and the added mass, 0.5 nu_r^T M nu_r (M of eq. 6.8, which is
        symmetric with Appendix B's numbers), plus the potential of weight and buoyancy taken at
        the centres of gravity and buoyancy, J. In still water."""
        x = self.x
        nu = np.array(x[3:6] + x[9:12])
        T = 0.5 * float(nu @ self.mass_matrix() @ nu)
        R = rot_nb(x[6], x[7], x[8])
        dG = x[2] + float(R[2] @ np.array(self.rg))
        dB = x[2] + float(R[2] @ np.array(self.rb))
        return T - self.W * dG + self.B * dB

    def jacobian(self, xr, ui, eps=None):
        """Numerical Jacobians of rigid_rates at the 12 states xr and the inputs ui = (stern,
        rudder, rpm): central differences. Returns (A, B), 12 x 12 and 12 x 3."""
        xr, ui = np.asarray(xr, float), np.asarray(ui, float)

        def f(xv, uv):
            return np.array(self.rigid_rates(list(xv), uv[0], uv[1], uv[2]))
        A = np.zeros((12, 12))
        Bm = np.zeros((12, 3))
        for k in range(12):
            hk = 1e-6 * max(1.0, abs(xr[k]))
            xp, xm = xr.copy(), xr.copy()
            xp[k] += hk
            xm[k] -= hk
            A[:, k] = (f(xp, ui) - f(xm, ui)) / (2.0 * hk)
        for k in range(3):
            hk = 1e-6 * max(1.0, abs(ui[k]))
            up, um = ui.copy(), ui.copy()
            up[k] += hk
            um[k] -= hk
            Bm[:, k] = (f(xr, up) - f(xr, um)) / (2.0 * hk)
        return A, Bm

    def turn_radius(self, rudder, rpm=1500.0, seconds=60.0):
        """The steady turning radius at a fixed rudder angle (rad) and propeller speed, from the
        model itself: from the straight trim, the rudder held, the stern planes at their trim,
        still water; the horizontal speed over the yaw rate, averaged over the last third. m."""
        probe = Remus(self.spec, dt=1.0 / 60.0, guards=self.guards_on, fin_tau=0.0, rpm_tau=0.0, coeffs=self.c)
        tr = probe.trim(rpm, d=50.0)
        probe.cmd = [tr["stern"], rudder, rpm]
        vs, rs = [], []
        n = int(round(seconds / probe.dt))
        for i in range(n):
            probe.step()
            if i >= 2 * n // 3:
                vg = probe.ground_velocity()
                vs.append(math.hypot(vg[0], vg[1]))
                rs.append(abs(probe.x[11]) * math.cos(probe.x[7]) / max(math.cos(probe.x[6]), 1e-3))
        return float(np.mean(vs) / max(np.mean(rs), 1e-6))


_INDEX = {k: i for i, k in enumerate(STATE)}


# --------------------------------------------------------------------------- #
#  The autopilot: OURS
# --------------------------------------------------------------------------- #
def place3(A, b, i_angle, i_rate, wn, zeta, sign):
    """Output feedback u = kp x[i_angle] + kd x[i_rate] on a 3-state model x' = A x + b u: kp and
    kd that put a closed-loop pair at (wn, zeta), and the third pole that results (it must come
    out negative). sign picks the root whose kp has that sign. Returns (kp, kd, third pole)."""
    target = lambda p3: np.poly([complex(-zeta * wn, wn * math.sqrt(max(0.0, 1.0 - zeta * zeta))),
                                 complex(-zeta * wn, -wn * math.sqrt(max(0.0, 1.0 - zeta * zeta))), -p3]).real

    def resid(z):
        kp, kd, p3 = z
        K = np.zeros(3)
        K[i_angle], K[i_rate] = kp, kd
        return np.poly(A + np.outer(b, K))[1:] - target(p3)[1:]
    best = None
    for p0 in (0.5, 1.0, 2.0, 4.0, 8.0):
        z = np.array([sign * 1.0, sign * 1.0, p0])
        for _ in range(60):
            r0 = resid(z)
            J = np.zeros((3, 3))
            for k in range(3):
                h = 1e-7 * max(1.0, abs(z[k]))
                zp = z.copy()
                zp[k] += h
                J[:, k] = (resid(zp) - r0) / h
            z = z - np.linalg.solve(J, r0)
            if np.max(np.abs(r0)) < 1e-12:
                break
        if np.max(np.abs(resid(z))) < 1e-9 and z[2] > 0.0 and np.sign(z[0]) == sign and (best is None or z[2] < best[2]):
            best = z
    if best is None:
        raise RuntimeError(f"no stable output-feedback gains for wn {wn}, zeta {zeta}")
    return float(best[0]), float(best[1]), float(-best[2])


class Autopilot:
    """Successive loop closure on the model's own linearisation at the commanded speed:

        pitch    delta_s = delta_s* + kp (theta - theta_c) + kd q
        depth    theta_c = theta* + kz (D - D_c) + ki int(D - D_c), saturated outside a zone
        heading  delta_r = delta_r* + kp e + ki int e - kd r,  e = chi_c - chi (or psi_c - psi)
        speed    n = n* + kp (U_c - u) + ki int(U_c - u)

    The starred values are the straight trim (Remus.trim) at the propeller speed n* whose trim
    runs at the commanded speed. The model is open-loop unstable in both planes at speed (the
    Munk moments Muw and Nuv outweigh the fins' restoring moments: the heave-pitch pair at
    +0.37 +- 0.40i /s and the sway-yaw root at +0.85 /s at 1.54 m/s), so the pitch and heading
    gains come from the 3-state models (w, q, theta) and (v, r, psi) of the Jacobian at the trim,
    not from the angle and rate alone: kp and kd put a closed-loop pair at the spec's wn and zeta
    (``place3``; the third pole is reported as p3_th, p3_psi). The depth loop's gains place a pair
    on D' = -U theta, the speed loop's pole on u' = a_u u + b_n n. The spec's autopilot block
    holds the frequencies, damping, integrals and limits, nothing else. The heading integral acts
    only within zone_deg of the set point (it carries the rudder trim and the crab in a current).

    hold: "course" (the ground track, which turns into a current and compensates for it) or
    "heading" (the nose, which a cross current carries off the line)."""

    def __init__(self, auv, speed=None, hold="course"):
        self.auv = auv
        cfg = auv.spec["autopilot"]
        self.cfg = cfg
        self.hold = hold
        self.u_c = speed or cfg["speed"]
        self.chi_c = auv.x[8]
        self.d_c = auv.x[2]
        self.design(self.u_c)
        self.i_d = self.i_chi = self.i_u = 0.0
        self.theta_c = 0.0

    def design(self, speed):
        """The gains at `speed` (m/s)."""
        auv, cfg = self.auv, self.cfg
        rpm = auv.rpm_for(speed)
        # the linearisation without the guards: at the trim the stern planes' effective angle can
        # sit on the stall guard's plateau, where d(q')/d(delta_s) is zero
        probe = Remus(auv.spec, dt=auv.dt, guards=False, fin_tau=0.0, rpm_tau=0.0, coeffs=auv.c)
        for _ in range(6):                                 # the propeller speed whose trim has u = speed
            tr = probe.trim(rpm, d=10.0, apply=False)
            rpm *= speed / tr["u"]
        tr = probe.trim(rpm, d=10.0, apply=False)
        self.trim, self.rpm = tr, rpm
        xr = [0.0, 0.0, 10.0, tr["u"], tr["v"], tr["w"], tr["phi"], tr["theta"], 0.0, 0.0, 0.0, 0.0]
        A, B = probe.jacobian(xr, [tr["stern"], tr["rudder"], rpm])
        I = {k: i for i, k in enumerate(STATE[:12])}
        self.A, self.B = A, B
        # pitch: delta_s = kp (theta - theta_c) + kd q on the heave-pitch model (w, q, theta)
        c = cfg["pitch"]
        sub = [I["w"], I["q"], I["theta"]]
        self.kp_th, self.kd_th, self.p3_th = place3(A[np.ix_(sub, sub)], B[sub, 0], 2, 1, c["wn"], c["zeta"], 1.0)
        self.th_max = math.radians(c["limit_deg"])
        # depth (D' = -U theta): theta_c = kz e + ki int e, the pair at wn, zeta
        c = cfg["depth"]
        self.kz = 2.0 * c["zeta"] * c["wn"] / tr["u"]
        self.ki_z = c["wn"] * c["wn"] / tr["u"]
        self.zone = c["zone"]
        # heading: delta_r = -kp psi - kd r (+ the integral) on the sway-yaw model (v, r, psi)
        c = cfg["heading"]
        sub = [I["v"], I["r"], I["psi"]]
        kp, kd, self.p3_psi = place3(A[np.ix_(sub, sub)], B[sub, 1], 2, 1, c["wn"], c["zeta"], 1.0)
        self.kp_psi, self.kd_psi = -kp, -kd               # delta_r = kp_psi e - kd_psi r, e = psi_c - psi
        self.ki_psi = c["ki"] * self.kp_psi
        self.i_chi_max = math.radians(c["integral_deg"]) / max(abs(self.ki_psi), 1e-9)
        self.psi_zone = math.radians(c["zone_deg"])
        # speed on the propeller: u' = a_u (u - u*) + b_n (n - n*) from the Jacobian, the pole at wn
        c = cfg["speed_loop"]
        self.a_u, self.b_n = A[I["u"], I["u"]], B[I["u"], 2]
        self.kp_u = (c["wn"] + self.a_u) / self.b_n
        self.ki_u = c["ki"] * self.kp_u
        self.i_u_max = c["integral_rpm"] / max(self.ki_u, 1e-9)

    def command(self, course=None, heading=None, depth=None, speed=None):
        """The guidance hook: a course (ground track) or a heading, rad from north toward east; a
        depth, m below the NED origin; a speed through the water, m/s. What is not given is kept."""
        if course is not None:
            self.chi_c, self.hold = course, "course"
        if heading is not None:
            self.chi_c, self.hold = heading, "heading"
        if depth is not None:
            self.d_c = depth
        if speed is not None and abs(speed - self.u_c) > 1e-9:
            self.u_c = speed
            self.design(speed)

    def aim_at(self, n, e, depth=None, speed=None):
        """Pure pursuit: the course straight at the point (n, e)."""
        x = self.auv.x
        self.command(course=math.atan2(e - x[1], n - x[0]), depth=depth, speed=speed)

    def update(self):
        """Writes the vehicle's stern-plane, rudder and propeller commands for its next step."""
        auv, dt = self.auv, self.auv.dt
        x = auv.x
        tr = self.trim
        theta, psi, q, r = x[7], x[8], x[10], x[11]
        # depth -> pitch
        e_d = x[2] - self.d_c
        if abs(e_d) > self.zone:
            th_c = math.copysign(self.th_max, e_d)
        else:
            th_c = tr["theta"] + self.kz * e_d + self.ki_z * self.i_d
            if abs(th_c) < self.th_max or th_c * e_d < 0.0:
                self.i_d += e_d * dt                       # anti-windup: hold the integral while saturated
        self.theta_c = th_c = clamp(th_c, -self.th_max, self.th_max)
        # pitch -> stern planes
        ds = tr["stern"] + self.kp_th * (theta - th_c) + self.kd_th * q
        # course (or heading) -> rudder
        if self.hold == "course":
            vg = auv.ground_velocity()
            chi = math.atan2(vg[1], vg[0]) if math.hypot(vg[0], vg[1]) > 0.05 else psi
        else:
            chi = psi
        e = wrap(self.chi_c - chi)
        dr = tr["rudder"] + self.kp_psi * e + self.ki_psi * self.i_chi - self.kd_psi * r
        if abs(e) < self.psi_zone:                         # the integral only near the set point (trim, crab)
            self.i_chi = clamp(self.i_chi + e * dt, -self.i_chi_max, self.i_chi_max)
        # speed -> propeller
        e_u = self.u_c - x[3]
        n = self.rpm + self.kp_u * e_u + self.ki_u * self.i_u
        if 0.0 < n < auv.rpm_max or n * e_u < 0.0:
            self.i_u = clamp(self.i_u + e_u * dt, -self.i_u_max, self.i_u_max)
        auv.cmd = [ds, dr, clamp(n, 0.0, auv.rpm_max)]

    def status(self):
        return (f"chi_c {math.degrees(self.chi_c):6.1f}  theta_c {math.degrees(self.theta_c):+5.1f}  "
                f"D_c {self.d_c:5.1f}  U_c {self.u_c:4.2f}")


# --------------------------------------------------------------------------- #
#  Guidance: OURS
# --------------------------------------------------------------------------- #
class LOS:
    """Line of sight along a polyline of waypoints [(n, e), ...] (m, NED), x8_rig.LOS's law.

    On the leg from p_k to p_k+1 (path angle a_k), with the along-track distance s and the
    cross-track error e (positive to the right of the path):

        chi_d = a_k + atan(-e / delta)

    the course to the point `delta` metres ahead of the vehicle's projection on the leg
    (``aim``). A leg is done when s comes within `accept` metres of its end. accept "turn" (the
    default) is the distance a turn needs to come round onto the next leg: R tan(|dchi| / 2),
    dchi the corner's course change and R the model's steady turning radius at turn_rudder x the
    rudder limit (``radius``, from Remus.turn_radius; pass it to skip that), at most half the leg.
    With loop the route closes back to its first point."""

    def __init__(self, waypoints, lookahead=None, accept=None, loop=False, spec=None, radius=None, auv=None):
        spec = spec or (auv.spec if auv is not None else load_spec())
        g = spec["guidance"]
        self.wp = [tuple(map(float, p[:2])) for p in waypoints]
        self.delta = g["lookahead"] if lookahead is None else lookahead
        self.accept = g["accept"] if accept is None else accept
        if self.accept == "turn" and radius is None:
            probe = auv or Remus(spec)
            radius = probe.turn_radius(g["turn_rudder"] * probe.delta_max, probe.rpm_for(spec["autopilot"]["speed"]))
        self.radius = radius
        self.loop = loop
        self.k = 0
        self.e = self.s = 0.0
        self.aim = self.wp[1] if len(self.wp) > 1 else self.wp[0]
        self.done = False
        self.chi_d = self.path_angle = 0.0

    def switch_distance(self, k):
        """How far before the end of leg k it hands over to leg k + 1."""
        if self.accept != "turn":
            return float(self.accept)
        (an, ae), (bn, be) = self.leg(k)
        _, (cn, ce) = self.leg(k + 1)
        dchi = wrap(math.atan2(ce - be, cn - bn) - math.atan2(be - ae, bn - an))
        return min(self.radius * math.tan(0.5 * abs(dchi)), 0.5 * math.hypot(bn - an, be - ae))

    def leg(self, k=None):
        k = self.k if k is None else k
        a = self.wp[k % len(self.wp)]
        b = self.wp[(k + 1) % len(self.wp)]
        return a, b

    def legs(self):
        return len(self.wp) if self.loop else len(self.wp) - 1

    def update(self, n, e, speed=None):
        """The course command for the vehicle at (n, e). (speed is accepted for x8_rig.LOS's
        signature; the steady turning radius at a fixed rudder does not depend on it.)"""
        while True:
            (an, ae), (bn, be) = self.leg()
            L = math.hypot(bn - an, be - ae)
            ak = math.atan2(be - ae, bn - an)
            ca, sa = math.cos(ak), math.sin(ak)
            s = (n - an) * ca + (e - ae) * sa
            more = self.loop or self.k < self.legs() - 1
            if more and s >= L - self.switch_distance(self.k):
                self.k = (self.k + 1) % self.legs() if self.loop else self.k + 1
                continue
            break
        self.done = not self.loop and self.k == self.legs() - 1 and s >= L
        self.s = s
        self.e = -(n - an) * sa + (e - ae) * ca
        sa_ = s + self.delta
        self.aim = (an + sa_ * ca, ae + sa_ * sa)
        self.path_angle = ak
        self.chi_d = ak + math.atan2(-self.e, self.delta)
        return self.chi_d


def cross_track(waypoints, n, e, loop=False):
    """Distance from (n, e) to the nearest point of the polyline (m, unsigned)."""
    best = float("inf")
    pts = [tuple(p[:2]) for p in waypoints]
    pts = pts + ([pts[0]] if loop else [])
    for (an, ae), (bn, be) in zip(pts[:-1], pts[1:]):
        dn, de = bn - an, be - ae
        L2 = dn * dn + de * de
        t = clamp(((n - an) * dn + (e - ae) * de) / L2, 0.0, 1.0) if L2 > 0 else 0.0
        best = min(best, math.hypot(n - an - t * dn, e - ae - t * de))
    return best


# --------------------------------------------------------------------------- #
#  The 3D model
# --------------------------------------------------------------------------- #
class Visual:
    """remus100.glb's nodes posed from a Remus's state.

    root    the loaded model's scene node (tp.GLTFLoader().load(model_path()).scene)

    pose(auv, h0) places the root from the model's position and attitude (world = (E, h0 - D,
    -N)), turns each fin node about its hinge (local +Z) by its pair's angle, and turns the
    propeller about local +X by the integrated propeller speed. Each fin node's local +Z is the
    body axis its pair turns on (+y for the stern planes, +z for the rudders), so a positive turn
    is the thesis's positive delta_s (trailing edge down) and delta_r (trailing edge to port).
    The nodes' rest rotations are kept and the fin turn is applied after them."""

    NODE_NAMES = {"stern_port": "stern_port", "stern_starboard": "stern_starboard",
                  "rudder_upper": "rudder_upper", "rudder_lower": "rudder_lower", "propeller": "propeller"}

    def __init__(self, root):
        self.root = root
        self.nodes = {k: root.get_object_by_name(n) for k, n in self.NODE_NAMES.items()}
        missing = [k for k, v in self.nodes.items() if v is None]
        if missing:
            raise KeyError(f"remus100.glb has no node(s) {missing}: rebuild it with auv/build_remus_blender.py")
        self.rest = {k: (n.quaternion.x, n.quaternion.y, n.quaternion.z, n.quaternion.w) for k, n in self.nodes.items()}

    def _turn(self, key, axis, angle):
        h = 0.5 * angle
        s = math.sin(h)
        q = quat_mul(self.rest[key], (axis[0] * s, axis[1] * s, axis[2] * s, math.cos(h)))
        self.nodes[key].quaternion.set(*q)

    def pose(self, auv, h0=0.0):
        x = auv.x
        self.root.position.set(*ned_to_world(x[0], x[1], x[2], h0))
        self.root.quaternion.set(*quat_of(auv.world_rotation()))
        for k in ("stern_port", "stern_starboard"):
            self._turn(k, (0.0, 0.0, 1.0), x[12])
        for k in ("rudder_upper", "rudder_lower"):
            self._turn(k, (0.0, 0.0, 1.0), x[13])
        self._turn("propeller", (1.0, 0.0, 0.0), auv.prop_angle)
