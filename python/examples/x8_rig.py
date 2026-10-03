"""
x8_rig -- the Skywalker X8 flying wing: its published flight model, an autopilot and a guidance
law on top of it, and its 3D model posed from the model's state.

Four things, and nothing scene-specific:

  * ``X8``        -- the 6-DOF model of B. Løw-Hansen, R. Hann, K. Gryte, T. A. Johansen and
                     C. Deiler, "Modeling and identification of a small fixed-wing UAV using
                     estimated aerodynamic angles", CEAS Aeronautical Journal 16, 501-523 (2025),
                     https://doi.org/10.1007/s13272-025-00816-3 (open access, CC BY 4.0): its
                     equations (5)-(29) as written, with the numbers of its Tables 1, 6, 7, 8 and 9
                     (uav/x8_spec.json). Air-relative velocity against a wind vector (15), (16);
                     stability-frame lift and drag rotated to the body (17)-(21); the propeller's
                     thrust and torque polynomials (5)-(9) with the gyroscopic term (10), (11) and
                     the rear-propeller sign (12), (13); the motor (28), (29); the rigid body (24),
                     (25), (26) and the Euler-angle kinematics (27); the elevon mixing (14); the
                     second-order servos and the first-order throttle with their input delays
                     (Table 9). Fixed-step fourth-order Runge-Kutta.
  * ``Autopilot`` -- OURS, not the article's: successive loop closure on the elevons and the
                     throttle (roll and pitch inside, course or heading on roll, altitude on
                     pitch, airspeed on throttle), its inner gains computed from the article's
                     coefficients. ``command(course=..., altitude=..., airspeed=...)`` is the hook
                     a guidance law drives; ``aim_at(n, e)`` flies at a point.
  * ``LOS``       -- OURS: line-of-sight guidance along a polyline of waypoints: the course to a
                     look-ahead point on the current leg, the cross-track error, the aim point.
  * ``Visual``    -- the x8.glb (uav/build_x8_blender.py) posed from an X8: the airframe from
                     the state, the elevons by their deflections, the propeller by its speed.

What the article does not give is kept out of its equations and marked OURS or ASSUMED where
it appears: the air density, gravity and the battery voltage (x8_spec.json says why), and the
guards that keep the model finite outside the envelope it was identified in (``X8.guards``).

Nothing here imports threepp: ``Visual`` is handed the loaded model's root node.

FRAMES
------
Inside the model everything is the article's: NED position (N, E, D), body axes x forward, y
right, z down, attitude (phi, theta, psi) with psi from north toward east. The one mapping to
threepp's world (x east, y up, z south; the drone rig's and the Nørvasundet twin's) is at the
boundary, in ``ned_to_world``, ``X8.world_position`` and ``X8.world_rotation``::

    world = (E, h0 - D, -N)            h0: the height of the NED origin in the world

so heading psi = 0 points the nose along world -z (north) and psi = 90 deg along +x (east). The
.glb's own frame is X forward, Y up, Z right (the boats' convention): a body vector (x, y, z) is
(x, -z, y) in it.

Wind is the velocity of the air over the ground, NED, m/s (the article's (u_w, v_w, w_w) is the
same vector in body axes): (0, 5, 0) is air moving east at 5 m/s, a wind FROM the west.

TYPICAL USE
-----------
::

    from x8_rig import X8, Autopilot, LOS, Visual, model_path

    x8 = X8(wind=(0.0, -6.0, 0.0))                 # 6 m/s from the east
    x8.trim(18.0, n=0.0, e=0.0, h=60.0, psi=0.0)   # level at 18 m/s, 60 m up, heading north
    ap = Autopilot(x8)
    los = LOS([(0, 0), (400, 0), (400, 300)])
    while ...:
        ap.command(course=los.update(x8.n, x8.e), altitude=60.0, airspeed=18.0)
        ap.update()                                  # elevon and throttle commands
        x8.step()                                    # one fixed step, x8.dt
    visual = Visual(tp.GLTFLoader().load(model_path()).scene)
    visual.pose(x8, h0=0.0)
"""
import json
import math
import os

import numpy as np

UAV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uav")
SPEC_PATH = os.path.join(UAV_DIR, "x8_spec.json")
TWO_PI = 2.0 * math.pi


def load_spec(path=None):
    with open(path or SPEC_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def model_path():
    """The generated x8.glb, or None where it has not been built."""
    p = os.path.join(UAV_DIR, "x8.glb")
    return p if os.path.isfile(p) else None


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def isa_density(h, rho0=1.225):
    """ISA troposphere density at h metres above mean sea level."""
    return rho0 * (1.0 - 0.0065 * h / 288.15) ** 4.2559


# --------------------------------------------------------------------------- #
#  The frames
# --------------------------------------------------------------------------- #
C_WN = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])    # NED vector -> world (x east, y up, z south)
C_BM = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])    # model (X fwd, Y up, Z right) -> body (FRD)


def ned_to_world(n, e, d, h0=0.0):
    return (e, h0 - d, -n)


def world_to_ned(x, y, z, h0=0.0):
    return (-z, x, h0 - y)


def rot_nb(phi, theta, psi):
    """R_b^n: body (FRD) -> NED, zyx Euler angles."""
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


# --------------------------------------------------------------------------- #
#  The aircraft: the article's model
# --------------------------------------------------------------------------- #
# Full state, in this order. The first 13 are the article's: position, velocity and angular rate
# in body axes, the Euler angles, the propeller speed. Then the actuators of its Table 9.
STATE = ("n", "e", "d", "u", "v", "w", "phi", "theta", "psi", "p", "q", "r", "omega",
         "elevon_r", "elevon_r_rate", "elevon_l", "elevon_l_rate", "throttle")
N_RIGID = 13


class X8:
    """The Skywalker X8 as the article models it.

    spec     x8_spec.json as a dict (default: load it)
    dt       the fixed step, s (default 1/300: the servo and throttle delays are whole steps,
             and 60, 50 and 30 frames a second are whole numbers of steps)
    wind     NED wind, m/s: a 3-tuple, or a callable (t, n, e, d) -> (wn, we, wd)
    h0       the NED origin's height above mean sea level (the air density follows it)
    rho      a fixed air density in place of the spec's (number, or None)

    The state is ``self.x`` (STATE order); ``self.cmd`` holds the elevon and throttle commands
    (right elevon, left elevon in rad, throttle 0..1) that the next step() pushes into the
    delay lines. ``self.out`` is what the last step evaluated at its start (airspeed, angles,
    thrust, torque, current, coefficients, guards)."""

    def __init__(self, spec=None, dt=1.0 / 300.0, wind=(0.0, 0.0, 0.0), h0=0.0, rho=None):
        self.spec = spec = load_spec() if spec is None else spec
        P, A, PR, PU, AC, EN, ENV = (spec["physical"], spec["aero"], spec["propeller"], spec["propulsion"],
                                     spec["actuators"], spec["environment"], spec["envelope"])
        self.m, self.Ix, self.Iy, self.Iz, self.Ixz = P["mass"], P["Ix"], P["Iy"], P["Iz"], P["Ixz"]
        self.cbar, self.b, self.S = P["cbar"], P["b"], P["S"]
        G = self.Ix * self.Iz - self.Ixz * self.Ixz
        self.Gx, self.Gxz, self.Gz = self.Iz / G, self.Ixz / G, self.Ix / G      # I^-1, x-z block
        self.a = A
        self.ct = (PR["CT0"], PR["CT1"], PR["CT2"], PR["CT3"])
        self.cq = (PR["CQ0"], PR["CQ1"], PR["CQ2"])
        self.KE, self.Rm, self.D, self.Ip = PU["KE"], PU["R"], PU["D"], PU["Ip"]
        self.Ub = PU["battery_voltage"]
        self.wn_s, self.zeta_s = AC["servo_wn"], AC["servo_zeta"]
        self.tau_t = AC["throttle_tau"]
        self.elevon_limit = math.radians(AC["elevon_limit_deg"])
        self.g = EN["g"]
        self.rho0 = EN["rho_sea_level"]
        self.rho_fixed = rho if rho is not None else (None if EN["rho"] == "isa" else float(EN["rho"]))
        self.h0 = h0
        # guards (OURS, envelope block of the spec): what enters the article's equations is held
        # inside the range it was identified in; the equations themselves are untouched
        self.alpha_lim = tuple(math.radians(a) for a in ENV["alpha_deg"])
        self.beta_lim = tuple(math.radians(a) for a in ENV["beta_deg"])
        self.va_min, self.j_max = ENV["va_min"], ENV["j_max"]
        self.guards = {"alpha": 0, "beta": 0, "va": 0, "j": 0, "omega": 0}
        self.dt = dt
        self.n_servo = self._steps(AC["servo_delay"])
        self.n_throttle = self._steps(AC["throttle_delay"])
        self.wind = wind
        self.x = [0.0] * len(STATE)
        self.cmd = [0.0, 0.0, 0.0]
        self.t = 0.0
        self.steps = 0
        self.prop_angle = 0.0             # the drawn propeller's angle: Omega integrated, for Visual
        self.out = {}
        self._out_at = -1                 # the step count self.out was evaluated at
        self._reset_delays()

    def _steps(self, delay):
        k = delay / self.dt
        n = int(round(k))
        if abs(k - n) > 1e-6:
            raise ValueError(f"a delay of {delay} s is not a whole number of {self.dt} s steps")
        return n

    def _reset_delays(self):
        """Fill the delay lines with the current commands, as if they had been held forever."""
        self.dl_er = [self.cmd[0]] * self.n_servo
        self.dl_el = [self.cmd[1]] * self.n_servo
        self.dl_t = [self.cmd[2]] * self.n_throttle

    # ---- named access
    def __getattr__(self, name):
        if name in _INDEX:
            return self.x[_INDEX[name]]
        raise AttributeError(name)

    @property
    def h(self):
        """Height above the NED origin, m."""
        return -self.x[2]

    def rho_at(self, d):
        return self.rho_fixed if self.rho_fixed is not None else isa_density(self.h0 - d, self.rho0)

    def wind_ned(self, t=None, n=None, e=None, d=None):
        w = self.wind
        if callable(w):
            x = self.x
            return w(self.t if t is None else t, x[0] if n is None else n, x[1] if e is None else e,
                     x[2] if d is None else d)
        return w

    # ---- the article's equations
    def rigid_rates(self, xr, de, da, dt_, wb, out=None):
        """d/dt of the 13 rigid-body and propeller states xr = (n, e, d, u, v, w, phi, theta, psi,
        p, q, r, Omega_p) for elevator and aileron deflections de, da (rad), throttle dt_ (0..1) and
        the wind in body axes wb = (u_w, v_w, w_w). Equation numbers are the article's."""
        n_, e_, d_, u, v, w, phi, th, psi, p, q, r, Om = xr
        A = self.a
        rho = self.rho_at(d_)
        # (15) air-relative velocity, (16) air data
        ua, va, wa = u - wb[0], v - wb[1], w - wb[2]
        Va = math.sqrt(ua * ua + va * va + wa * wa)
        Vg = max(Va, self.va_min)                          # guard: divisions by V_a below va_min
        alpha = math.atan2(wa, ua)                         # tan^-1(w_a / u_a), on all four quadrants
        beta = math.asin(clamp(va / Vg, -1.0, 1.0))
        al = clamp(alpha, *self.alpha_lim)                 # guard: coefficients inside the identified range
        be = clamp(beta, *self.beta_lim)
        # (5)-(9) propeller
        Omg = max(Om, 0.0)
        J = TWO_PI * Vg / (Omg * self.D) if Omg > 1e-9 else self.j_max
        Jg = min(J, self.j_max)                            # guard: C_T(J) turns back up past its minimum
        c0, c1, c2, c3 = self.ct
        CT = c0 + Jg * (c1 + Jg * (c2 + Jg * c3))
        q0, q1, q2 = self.cq
        CQ = q0 + Jg * (q1 + Jg * q2)
        k4 = rho * self.D ** 4 / (4.0 * math.pi * math.pi)
        T = k4 * CT * Omg * Omg                            # (7)
        Q = k4 * self.D * CQ * Omg * Omg                   # (6)
        # (28), (29) motor
        U = dt_ * self.Ub
        Im = (U - Omg * self.KE) / self.Rm
        Qm = Im * self.KE
        Om_dot = (Qm - Q) / self.Ip
        # (17), (18) coefficients
        b2v = self.b / (2.0 * Vg)
        ps, qs, rs = p * b2v, q * self.cbar / (2.0 * Vg), r * b2v
        CL = A["CL0"] + A["CLa"] * al + A["CLq"] * qs + A["CLde"] * de
        CD = A["CD0"] + A["CDq"] * qs + A["CDCT"] * CT + A["CDk1"] * CL + A["CDk2"] * CL * CL
        Cm = A["Cm0"] + A["Cma"] * al + A["Cmq"] * qs + A["Cmde"] * de
        CY = A["CY0"] + A["CYb"] * be + A["CYp"] * ps + A["CYr"] * rs + A["CYda"] * da
        Cl = A["Cl0"] + A["Clb"] * be + A["Clp"] * ps + A["Clr"] * rs + A["Clda"] * da
        Cn = A["Cn0"] + A["Cnb"] * be + A["Cnp"] * ps + A["Cnr"] * rs + A["Cnda"] * da
        # (19), (20) stability -> body: [CX, CY, CZ] = R_b^s(alpha)^T [-CD, CY, -CL]
        ca, sa = math.cos(alpha), math.sin(alpha)
        CX = -CD * ca + CL * sa
        CZ = -CD * sa - CL * ca
        # (21), (22) forces and moments in body axes
        qS = 0.5 * rho * Va * Va * self.S
        cf, sf = math.cos(phi), math.sin(phi)
        ct, st = math.cos(th), math.sin(th)
        mg = self.m * self.g
        Fx = qS * CX + T - mg * st                         # F_gravity = R_b^n^T [0, 0, m g]
        Fy = qS * CY + mg * sf * ct
        Fz = qS * CZ + mg * cf * ct
        Ip_Om = self.Ip * Omg
        L = qS * self.b * Cl - (Q + self.Ip * Om_dot)      # (11)-(13): M_prop,rear = -M_prop
        M = qS * self.cbar * Cm - Ip_Om * r
        N = qS * self.b * Cn + Ip_Om * q
        # (25a) translational, (25b) rotational, (24) the inertia
        udot = Fx / self.m - (q * w - r * v)
        vdot = Fy / self.m - (r * u - p * w)
        wdot = Fz / self.m - (p * v - q * u)
        h1, h2, h3 = self.Ix * p - self.Ixz * r, self.Iy * q, -self.Ixz * p + self.Iz * r
        L2, M2, N2 = L - (q * h3 - r * h2), M - (r * h1 - p * h3), N - (p * h2 - q * h1)
        pdot = self.Gx * L2 + self.Gxz * N2
        qdot = M2 / self.Iy
        rdot = self.Gxz * L2 + self.Gz * N2
        # (26) position, (27) Euler angles
        cp, sp = math.cos(psi), math.sin(psi)
        r00, r01, r02 = ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp
        r10, r11, r12 = ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp
        r20, r21, r22 = -st, sf * ct, cf * ct
        ndot = r00 * u + r01 * v + r02 * w
        edot = r10 * u + r11 * v + r12 * w
        ddot = r20 * u + r21 * v + r22 * w
        tt = st / ct
        phidot = p + sf * tt * q + cf * tt * r
        thdot = cf * q - sf * r
        psidot = (sf * q + cf * r) / ct
        if out is not None:
            out.update(Va=Va, alpha=alpha, beta=beta, rho=rho, J=J, CT=CT, CQ=CQ, T=T, Q=Q, U=U, Im=Im,
                       Qm=Qm, CL=CL, CD=CD, Cm=Cm, CY=CY, Cl=Cl, Cn=Cn, de=de, da=da, throttle=dt_,
                       Fx=Fx, Fy=Fy, Fz=Fz, L=L, M=M, N=N,
                       g_alpha=al != alpha, g_beta=be != beta, g_va=Va < self.va_min, g_j=J > self.j_max)
        return [ndot, edot, ddot, udot, vdot, wdot, phidot, thdot, psidot, pdot, qdot, rdot, Om_dot]

    def wind_body(self, xr, wn):
        """The NED wind in body axes: R_b^n^T w."""
        R = rot_nb(xr[6], xr[7], xr[8])
        return R.T @ np.asarray(wn, float)

    def rates(self, x, c, t):
        """d/dt of the full state under the delayed commands c = (right elevon, left elevon,
        throttle) held over the step: the rigid body, then the actuators of Table 9."""
        wn = self.wind_ned(t, x[0], x[1], x[2])
        if wn[0] == 0.0 and wn[1] == 0.0 and wn[2] == 0.0:
            wb = (0.0, 0.0, 0.0)
        else:
            cf, sf = math.cos(x[6]), math.sin(x[6])
            ct, st = math.cos(x[7]), math.sin(x[7])
            cp, sp = math.cos(x[8]), math.sin(x[8])
            a, b_, d = wn
            wb = (ct * cp * a + ct * sp * b_ - st * d,
                  (sf * st * cp - cf * sp) * a + (sf * st * sp + cf * cp) * b_ + sf * ct * d,
                  (cf * st * cp + sf * sp) * a + (cf * st * sp - sf * cp) * b_ + cf * ct * d)
        er, el = x[13], x[15]
        de, da = 0.5 * (er + el), 0.5 * (el - er)         # (14)
        dx = self.rigid_rates(x[:N_RIGID], de, da, x[17], wb)
        wn2, z2 = self.wn_s * self.wn_s, 2.0 * self.zeta_s * self.wn_s
        dx += [x[14], wn2 * (c[0] - er) - z2 * x[14],     # second-order servos
               x[16], wn2 * (c[1] - el) - z2 * x[16],
               (c[2] - x[17]) / self.tau_t]               # first-order throttle
        return dx

    def evaluate(self):
        """Air data, forces and the rest at the current state (fills self.out)."""
        x = self.x
        wn = self.wind_ned()
        wb = self.wind_body(x, wn)
        er, el = x[13], x[15]
        out = {}
        self.rigid_rates(x[:N_RIGID], 0.5 * (er + el), 0.5 * (el - er), x[17], wb, out)
        out["wind"] = tuple(wn)
        self.out = out
        self._out_at = self.steps
        return out

    # ---- stepping
    def step(self):
        """One fixed step: the commands into the delay lines, RK4 over dt, the guards' tally."""
        dt = self.dt
        lim = self.elevon_limit
        self.dl_er.append(clamp(self.cmd[0], -lim, lim))
        self.dl_el.append(clamp(self.cmd[1], -lim, lim))
        self.dl_t.append(clamp(self.cmd[2], 0.0, 1.0))
        c = (self.dl_er.pop(0), self.dl_el.pop(0), self.dl_t.pop(0))
        out = self.out if self._out_at == self.steps else self.evaluate()
        for k in ("alpha", "beta", "va", "j"):
            if out["g_" + k]:
                self.guards[k] += 1
        x, t = self.x, self.t
        k1 = self.rates(x, c, t)
        x2 = [a + 0.5 * dt * b for a, b in zip(x, k1)]
        k2 = self.rates(x2, c, t + 0.5 * dt)
        x3 = [a + 0.5 * dt * b for a, b in zip(x, k2)]
        k3 = self.rates(x3, c, t + 0.5 * dt)
        x4 = [a + dt * b for a, b in zip(x, k3)]
        k4 = self.rates(x4, c, t + dt)
        self.x = [a + dt / 6.0 * (b1 + 2.0 * b2 + 2.0 * b3 + b4) for a, b1, b2, b3, b4 in zip(x, k1, k2, k3, k4)]
        if self.x[12] < 0.0:                               # guard: the ESC drives one way
            self.x[12] = 0.0
            self.guards["omega"] += 1
        self.prop_angle = (self.prop_angle + self.x[12] * dt) % TWO_PI
        self.t = t + dt
        self.steps += 1

    def run(self, seconds, before_step=None):
        """Step for `seconds`; before_step(self) runs before every step (an autopilot's update)."""
        for _ in range(int(round(seconds / self.dt))):
            if before_step is not None:
                before_step(self)
            self.step()

    # ---- setting a state
    def set_state(self, n=0.0, e=0.0, d=0.0, u=18.0, v=0.0, w=0.0, phi=0.0, theta=0.0, psi=0.0,
                  p=0.0, q=0.0, r=0.0, omega=0.0, de=0.0, da=0.0, throttle=0.0):
        """The state, with the actuators at rest on the given deflections and throttle, and the
        delay lines full of the same commands."""
        er, el = de - da, de + da
        self.x = [n, e, d, u, v, w, phi, theta, psi, p, q, r, omega, er, 0.0, el, 0.0, throttle]
        self.cmd = [er, el, throttle]
        self._reset_delays()
        self._out_at = -1
        self.evaluate()

    def motor_speed(self, va, throttle, d=0.0):
        """The propeller speed at which the motor's torque (28) balances the propeller's (6) at
        airspeed va and this throttle (Omega_p' = 0 in (29)), by bisection."""
        xr = [0.0, 0.0, d, va, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        lo, hi = 0.0, throttle * self.Ub / self.KE
        if hi <= 0.0:
            return 0.0
        for _ in range(80):
            xr[12] = 0.5 * (lo + hi)
            if self.rigid_rates(xr, 0.0, 0.0, throttle, (0.0, 0.0, 0.0))[12] > 0.0:
                lo = xr[12]
            else:
                hi = xr[12]
        return 0.5 * (lo + hi)

    def trim(self, va, n=0.0, e=0.0, h=0.0, psi=0.0, gamma=0.0, apply=True, iters=40, course=None):
        """Straight flight at airspeed va and air-path angle gamma (rad), wings held by the
        elevons: solves the six rigid-body equations, the motor's balance and the path angle for
        alpha, beta, phi, theta, de, da, throttle and Omega_p, in still air. (The X8 needs a
        little aileron, bank and sideslip to fly straight: Cl0, Cn0 and CY0 are not zero, and
        the propeller's torque rolls it.) With apply, the aircraft is put there (in the wind it
        has, at the same air-relative velocity), heading psi, or, given a course, at the heading
        that makes that ground track in the wind. Returns the solution as a dict."""
        z = np.array([0.08, 0.0, 0.0, 0.08 + gamma, -0.04, 0.0, 0.4, 450.0])   # alpha beta phi theta de da dt Om
        scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
        d = -h

        def resid(z):
            al, be, ph, th, de, da, dt_, om = z
            u, v, w = va * math.cos(al) * math.cos(be), va * math.sin(be), va * math.sin(al) * math.cos(be)
            xr = [0.0, 0.0, d, u, v, w, ph, th, 0.0, 0.0, 0.0, 0.0, om]
            f = self.rigid_rates(xr, de, da, dt_, (0.0, 0.0, 0.0))
            climb = -f[2]
            return np.array(f[3:6] + f[9:12] + [f[12] / 100.0, climb - va * math.sin(gamma)])
        for _ in range(iters):
            r0 = resid(z)
            Jm = np.zeros((8, 8))
            for k in range(8):
                hk = 1e-6 * scale[k]
                zp, zm = z.copy(), z.copy()
                zp[k] += hk
                zm[k] -= hk
                Jm[:, k] = (resid(zp) - resid(zm)) / (2.0 * hk)
            step = np.linalg.solve(Jm, -r0)
            z = z + step
            if np.max(np.abs(step / scale)) < 1e-12:
                break
        al, be, ph, th, de, da, dt_, om = (float(a) for a in z)
        sol = {"va": va, "alpha": al, "beta": be, "phi": ph, "theta": th, "de": de, "da": da, "throttle": dt_,
               "omega": om, "residual": float(np.max(np.abs(resid(z))))}
        if apply:
            u, v, w = va * math.cos(al) * math.cos(be), va * math.sin(be), va * math.sin(al) * math.cos(be)

            def place(psi):
                R = rot_nb(ph, th, psi)
                wb = R.T @ np.asarray(self.wind_ned(0.0, n, e, d), float)
                self.set_state(n, e, d, u + wb[0], v + wb[1], w + wb[2], ph, th, psi, omega=om, de=de, da=da, throttle=dt_)
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

    def ground_velocity(self):
        """NED velocity over the ground (eq. 26)."""
        x = self.x
        return rot_nb(x[6], x[7], x[8]) @ np.array(x[3:6])

    def air_velocity_ned(self):
        return self.ground_velocity() - np.asarray(self.wind_ned(), float)

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
        """Kinetic (translational over the ground and rotational) plus potential energy per the
        NED origin, J. The propeller's own spin is not in it."""
        x = self.x
        u, v, w, p, q, r = x[3], x[4], x[5], x[9], x[10], x[11]
        rot = 0.5 * (self.Ix * p * p + self.Iy * q * q + self.Iz * r * r) - self.Ixz * p * r
        return 0.5 * self.m * (u * u + v * v + w * w) + rot - self.m * self.g * x[2]

    def jacobian(self, xr, ui, eps=None):
        """Numerical Jacobians of rigid_rates at the 13 states xr and the inputs ui = (de, da,
        throttle, u_w, v_w, w_w) (wind in body axes): central differences. Returns (A, B),
        13 x 13 and 13 x 6."""
        xr, ui = np.asarray(xr, float), np.asarray(ui, float)

        def f(xv, uv):
            return np.array(self.rigid_rates(list(xv), uv[0], uv[1], uv[2], tuple(uv[3:6])))
        A = np.zeros((13, 13))
        B = np.zeros((13, 6))
        for k in range(13):
            hk = 1e-6 * max(1.0, abs(xr[k]))
            xp, xm = xr.copy(), xr.copy()
            xp[k] += hk
            xm[k] -= hk
            A[:, k] = (f(xp, ui) - f(xm, ui)) / (2.0 * hk)
        for k in range(6):
            hk = 1e-6 * max(1.0, abs(ui[k]))
            up, um = ui.copy(), ui.copy()
            up[k] += hk
            um[k] -= hk
            B[:, k] = (f(xr, up) - f(xr, um)) / (2.0 * hk)
        return A, B


_INDEX = {k: i for i, k in enumerate(STATE)}


def quasi_steady(A, B, k=12):
    """Eliminate state k (the propeller speed) as quasi-steady: A11 - A12 A22^-1 A21, and the
    same for B. The motor's time constant is about 0.02 s."""
    keep = [i for i in range(A.shape[0]) if i != k]
    a22 = A[k, k]
    Ar = A[np.ix_(keep, keep)] - np.outer(A[keep, k], A[k, keep]) / a22
    Br = B[keep, :] - np.outer(A[keep, k], B[k, :]) / a22
    return Ar, Br


# --------------------------------------------------------------------------- #
#  The autopilot: OURS
# --------------------------------------------------------------------------- #
class Autopilot:
    """Successive loop closure, as in Beard and McLain, Small Unmanned Aircraft (2012), ch. 6:

        roll     d_a = d_a* + kp (phi_c - phi) + ki int(phi_c - phi) - kd p
        course   phi_c = kp (chi_c - chi) + ki int(chi_c - chi)      (or heading psi in place of chi)
        pitch    d_e = d_e* + kp (theta_c - theta) + ki int(theta_c - theta)
        altitude theta_c = theta* + kp (h_c - h) + ki int(h_c - h), saturated outside a zone
        airspeed d_t = d_t* + kp (V_c - V_a) + ki int(V_c - V_a)

    The starred values are the still-air trim at the commanded airspeed. The roll and pitch
    gains come from the article's coefficients at that airspeed (the second-order models
    phi'' = -a_phi1 phi' + a_phi2 d_a and theta'' = -a_th1 theta' - a_th2 theta + a_th3 d_e):
    roll from its natural frequency, pitch as a multiple of the airframe's own stiffness (the
    X8's is above what the servo delay lets a pitch loop reach; x8_spec.json says why). The
    course loop's gains come from the roll loop's and a bandwidth ratio, the altitude loop's
    from its own natural frequency (h' = V_a theta), the airspeed loop's from the model's own
    d(u')/du and d(u')/d(throttle) at the trim, the motor taken as quasi-steady. The spec's
    autopilot block holds the frequencies, stiffness, damping, integrals and limits, nothing else.

    Measured from the 18 m/s trim in still air (uav/README.md): +10 m of height in 4.7 s (90 %),
    0.55 m over; a 90 deg course change in 5.2 s, 5.4 deg over, the height within 0.9 m; +2 m/s of
    airspeed in 1.3 s, 0.34 m/s over.

    hold: "course" (the ground track, which turns into the wind and compensates for it) or
    "heading" (the nose, which drifts with the wind)."""

    def __init__(self, x8, airspeed=None, hold="course"):
        self.x8 = x8
        cfg = x8.spec["autopilot"]
        self.cfg = cfg
        self.hold = hold
        self.va_c = airspeed or cfg["airspeed"]
        self.chi_c = x8.x[8]
        self.h_c = x8.h
        self.design(self.va_c)
        self.i_phi = self.i_chi = self.i_h = self.i_v = self.i_th = 0.0
        self.phi_c = self.theta_c = 0.0

    def design(self, va):
        """The gains at airspeed va (and the air density at the aircraft's height)."""
        x8, A, cfg = self.x8, self.x8.a, self.cfg
        probe = X8(x8.spec, dt=x8.dt, h0=x8.h0, rho=x8.rho_fixed)
        tr = probe.trim(va, h=x8.h, apply=False)
        self.trim = tr
        rho = x8.rho_at(x8.x[2])
        qS = 0.5 * rho * va * va * x8.S
        # roll: C_p = Gamma_3 C_l + Gamma_4 C_n in units of the inertia inverse (Ix Iz - Ixz^2 folded in)
        a1 = -qS * x8.b * (x8.Gx * A["Clp"] + x8.Gxz * A["Cnp"]) * x8.b / (2.0 * va)
        a2 = qS * x8.b * (x8.Gx * A["Clda"] + x8.Gxz * A["Cnda"])
        c = cfg["roll"]
        w, z = c["wn"], c["zeta"]
        self.a_phi = (a1, a2)
        self.kp_phi = w * w / a2
        # rate feedback only where the airframe's own roll damping is short of 2 zeta wn; the
        # X8's (a_phi1 = 17 /s at 18 m/s, the article's roll mode) is not, and positive rate
        # feedback through the 0.07 s servo delay would cost phase
        self.kd_phi = max(0.0, (2.0 * z * w - a1) / a2)
        self.ki_phi = c["ki"] * self.kp_phi
        self.i_phi_max = math.radians(c["integral_deg"]) / self.ki_phi    # the integral's share of the aileron
        self.phi_max = math.radians(c["limit_deg"])
        # course (chi' = g / V_g tan phi)
        c = cfg["course"]
        self.w_chi = w / c["bandwidth_ratio"]
        self.z_chi = c["zeta"]
        # pitch: theta'' = -a_th1 theta' - a_th2 theta + a_th3 d_e. The X8's open-loop pitch
        # stiffness a_th2 (105 /s^2 at 18 m/s, 10 rad/s) is above any bandwidth the 0.07 s delay
        # allows, and its short period is well damped (the article: zeta 0.78); so no rate
        # feedback, and a proportional gain that adds `stiffness` times the airframe's own:
        # kp a_th3 = stiffness a_th2, i.e. kp = -stiffness Cma / Cmde, the same at every speed.
        # The integral takes the steady error out.
        a1 = -qS * x8.cbar * A["Cmq"] * x8.cbar / (2.0 * va) / x8.Iy
        a2 = -qS * x8.cbar * A["Cma"] / x8.Iy
        a3 = qS * x8.cbar * A["Cmde"] / x8.Iy
        self.a_th = (a1, a2, a3)
        c = cfg["pitch"]
        self.kp_th = c["stiffness"] * a2 / a3
        self.ki_th = c["wi"] * self.kp_th
        self.i_th_max = math.radians(c["integral_deg"]) / abs(self.ki_th)
        self.th_max = math.radians(c["limit_deg"])
        # altitude (h' = V_a theta, the pitch loop's DC gain 1 by its integral)
        c = cfg["altitude"]
        wh = c["wn"]
        self.kp_h = 2.0 * c["zeta"] * wh / va
        self.ki_h = wh * wh / va
        self.h_zone = c["zone"]
        # airspeed on throttle: V_a' = -a_V1 V_a + a_V2 d_t, from the model's Jacobian at the trim
        u, v, ww = (va * math.cos(tr["alpha"]) * math.cos(tr["beta"]), va * math.sin(tr["beta"]),
                    va * math.sin(tr["alpha"]) * math.cos(tr["beta"]))
        xr = [0.0, 0.0, x8.x[2], u, v, ww, tr["phi"], tr["theta"], 0.0, 0.0, 0.0, 0.0, tr["omega"]]
        Aj, Bj = probe.jacobian(xr, [tr["de"], tr["da"], tr["throttle"], 0.0, 0.0, 0.0])
        Ar, Br = quasi_steady(Aj, Bj)
        av1, av2 = -Ar[3, 3], Br[3, 2]
        c = cfg["speed"]
        w, z = c["wn"], c["zeta"]
        self.kp_v = (2.0 * z * w - av1) / av2
        self.ki_v = w * w / av2
        self.t_lim = tuple(c["throttle"])
        self.a_v = (av1, av2)

    def command(self, course=None, heading=None, altitude=None, airspeed=None):
        """The guidance hook: a course (ground track) or a heading, rad from north toward east;
        a height above the NED origin, m; an airspeed, m/s. What is not given is kept."""
        if course is not None:
            self.chi_c, self.hold = course, "course"
        if heading is not None:
            self.chi_c, self.hold = heading, "heading"
        if altitude is not None:
            self.h_c = altitude
        if airspeed is not None and abs(airspeed - self.va_c) > 1e-9:
            self.va_c = airspeed
            self.design(airspeed)

    def aim_at(self, n, e, altitude=None, airspeed=None):
        """Pure pursuit: the course straight at the point (n, e)."""
        x = self.x8.x
        self.command(course=math.atan2(e - x[1], n - x[0]), altitude=altitude, airspeed=airspeed)

    def update(self):
        """Writes the X8's elevon and throttle commands for its next step."""
        x8, dt = self.x8, self.x8.dt
        x = x8.x
        tr = self.trim
        out = x8.out if x8._out_at == x8.steps else x8.evaluate()
        phi, theta, psi, p, q = x[6], x[7], x[8], x[9], x[10]
        # course (or heading) -> roll
        if self.hold == "course":
            vg = x8.ground_velocity()
            chi = math.atan2(vg[1], vg[0])
            spd = max(math.hypot(vg[0], vg[1]), 5.0)
        else:
            chi, spd = psi, max(out["Va"], 5.0)
        e_chi = wrap(self.chi_c - chi)
        kp = 2.0 * self.z_chi * self.w_chi * spd / x8.g
        ki = self.w_chi * self.w_chi * spd / x8.g
        phi_c = kp * e_chi + ki * self.i_chi
        if abs(phi_c) < self.phi_max or phi_c * e_chi < 0.0:
            self.i_chi += e_chi * dt                       # anti-windup: hold the integral while saturated
        self.phi_c = phi_c = clamp(phi_c, -self.phi_max, self.phi_max)
        # roll -> aileron
        e_phi = phi_c - phi
        da = tr["da"] + self.kp_phi * e_phi + self.ki_phi * self.i_phi - self.kd_phi * p
        self.i_phi = clamp(self.i_phi + e_phi * dt, -self.i_phi_max, self.i_phi_max)
        # altitude -> pitch
        e_h = self.h_c - x8.h
        if abs(e_h) > self.h_zone:
            th_c = math.copysign(self.th_max, e_h)
        else:
            th_c = tr["theta"] + self.kp_h * e_h + self.ki_h * self.i_h
            if abs(th_c) < self.th_max or th_c * e_h < 0.0:
                self.i_h += e_h * dt
        self.theta_c = th_c = clamp(th_c, -self.th_max, self.th_max)
        # pitch -> elevator
        e_th = th_c - theta
        de = tr["de"] + self.kp_th * e_th + self.ki_th * self.i_th
        self.i_th = clamp(self.i_th + e_th * dt, -self.i_th_max, self.i_th_max)
        # airspeed -> throttle
        e_v = self.va_c - out["Va"]
        thr = tr["throttle"] + self.kp_v * e_v + self.ki_v * self.i_v
        if self.t_lim[0] < thr < self.t_lim[1] or thr * e_v < 0.0:
            self.i_v += e_v * dt
        thr = clamp(thr, *self.t_lim)
        # mixing (14) inverted: the elevons; their throw is clipped in X8.step
        x8.cmd = [de - da, de + da, thr]

    def status(self):
        return (f"chi_c {math.degrees(self.chi_c):6.1f}  phi_c {math.degrees(self.phi_c):+5.1f}  "
                f"theta_c {math.degrees(self.theta_c):+5.1f}  h_c {self.h_c:5.1f}  V_c {self.va_c:4.1f}")


# --------------------------------------------------------------------------- #
#  Guidance: OURS
# --------------------------------------------------------------------------- #
class LOS:
    """Line of sight along a polyline of waypoints [(n, e), ...] (m, NED).

    On the leg from p_k to p_k+1 (path angle a_k), with the along-track distance s and the
    cross-track error e (positive to the right of the path):

        chi_d = a_k + atan(-e / delta)

    the course to the point `delta` metres ahead of the aircraft's projection on the leg
    (``aim``). A leg is done when s comes within `accept` metres of its end. accept "turn" (the
    default) is the distance a turn needs to come round onto the next leg: R tan(|dchi| / 2),
    dchi the corner's course change and R = V_g^2 / (g tan(turn_bank x the roll limit)) at the
    ground speed update() is given (else the autopilot's airspeed), at most half the leg. With
    loop the route closes back to its first point."""

    def __init__(self, waypoints, lookahead=None, accept=None, loop=False, spec=None):
        spec = spec or load_spec()
        g = spec["guidance"]
        self.wp = [tuple(map(float, p)) for p in waypoints]
        self.delta = g["lookahead"] if lookahead is None else lookahead
        self.accept = g["accept"] if accept is None else accept
        self.turn_bank = g["turn_bank"] * math.radians(spec["autopilot"]["roll"]["limit_deg"])
        self.g, self.va = spec["environment"]["g"], spec["autopilot"]["airspeed"]
        self.loop = loop
        self.k = 0
        self.e = self.s = 0.0
        self.aim = self.wp[1] if len(self.wp) > 1 else self.wp[0]
        self.done = False
        self.chi_d = self.path_angle = 0.0

    def switch_distance(self, k, speed):
        """How far before the end of leg k it hands over to leg k + 1."""
        if self.accept != "turn":
            return float(self.accept)
        (an, ae), (bn, be) = self.leg(k)
        _, (cn, ce) = self.leg(k + 1)
        dchi = wrap(math.atan2(ce - be, cn - bn) - math.atan2(be - ae, bn - an))
        R = speed * speed / (self.g * math.tan(self.turn_bank))
        return min(R * math.tan(0.5 * abs(dchi)), 0.5 * math.hypot(bn - an, be - ae))

    def leg(self, k=None):
        k = self.k if k is None else k
        a = self.wp[k % len(self.wp)]
        b = self.wp[(k + 1) % len(self.wp)]
        return a, b

    def legs(self):
        return len(self.wp) if self.loop else len(self.wp) - 1

    def update(self, n, e, speed=None):
        """The course command for the aircraft at (n, e), flying at `speed` over the ground."""
        speed = self.va if speed is None else speed
        while True:
            (an, ae), (bn, be) = self.leg()
            L = math.hypot(bn - an, be - ae)
            ak = math.atan2(be - ae, bn - an)
            ca, sa = math.cos(ak), math.sin(ak)
            s = (n - an) * ca + (e - ae) * sa
            more = self.loop or self.k < self.legs() - 1
            if more and s >= L - self.switch_distance(self.k, speed):
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
    pts = list(waypoints) + ([waypoints[0]] if loop else [])
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
    """x8.glb's nodes posed from an X8's state.

    root    the loaded model's scene node (tp.GLTFLoader().load(model_path()).scene)

    pose(x8, h0) places the root from the model's position and attitude (world = (E, h0 - D,
    -N)), turns each elevon node about its hinge (local +Z) by that elevon's deflection
    (positive trailing edge down), and turns the propeller about local +X by the model's
    integrated Omega_p (its spin vector along body +x, the sense eq. (13) implies)."""

    NODE_NAMES = {"elevon_left": "elevon_left", "elevon_right": "elevon_right", "propeller": "propeller"}

    def __init__(self, root):
        self.root = root
        self.nodes = {k: root.get_object_by_name(n) for k, n in self.NODE_NAMES.items()}
        missing = [k for k, v in self.nodes.items() if v is None]
        if missing:
            raise KeyError(f"x8.glb has no node(s) {missing}: rebuild it with uav/build_x8_blender.py")

    def pose(self, x8, h0=0.0):
        x = x8.x
        self.root.position.set(*ned_to_world(x[0], x[1], x[2], h0))
        self.root.quaternion.set(*quat_of(x8.world_rotation()))
        self.nodes["elevon_right"].rotation.z = x[13]
        self.nodes["elevon_left"].rotation.z = x[15]
        self.nodes["propeller"].rotation.x = x8.prop_angle
