"""
aerosonde_rig -- the Aerosonde UAV as Beard and McLain's textbook models it: the book's flight
model with the parameters its authors give, an autopilot and a guidance law on top of it, and
its 3D model posed from the model's state.

Four things, and nothing scene-specific:

  * ``Aerosonde`` -- the 6-DOF model of R. W. Beard and T. W. McLain, Small Unmanned Aircraft:
                     Theory and Practice (Princeton University Press, 2012), with the numbers its
                     authors give for the Aerosonde in the book's repository
                     (github.com/byu-magicc/mavsim_public, aerosonde_parameters.py; copied into
                     uav/aerosonde_spec.json). Chapter 3: the rigid body in Euler-angle form, twelve
                     states, the inertia's x-z product through the Gamma constants. Chapter 4: air
                     data against a wind vector (section 4.4), lift with the blended flat-plate stall
                     and the induced-drag polar (section 4.2.2), the linear lateral coefficients
                     (section 4.2.3), the stability-to-body rotation, gravity (section 4.1). The
                     authors' addendum to chapter 4: the propeller's thrust and torque polynomials
                     and the electric motor, its speed the root of a quadratic (quasi-steady: the
                     rotor has no state). Fixed-step fourth-order Runge-Kutta.
  * ``Autopilot`` -- OURS: successive loop closure as in the book's chapter 6 (roll on aileron,
                     course or heading on roll, pitch on elevator, altitude on pitch, airspeed on
                     throttle, a yaw damper on the rudder), the gains computed from the spec's
                     coefficients at the commanded airspeed. ``command(course=..., altitude=...,
                     airspeed=...)`` is the hook a guidance law drives; ``aim_at(n, e)`` flies at
                     a point. It is fed the exact state: no estimator, no sensor noise.
  * ``LOS``       -- x8_rig's line-of-sight guidance on a polyline, with this aircraft's spec.
  * ``Visual``    -- the aerosonde.glb posed from an Aerosonde: the airframe from the state, the
                     ailerons and the inverted V-tail's two ruddervators from the deflections, the
                     propeller from its speed.

The repository the numbers come from is GPL-3.0: the numbers and the book's equations are cited,
none of its code is copied or ported. What the book does not give is kept out of its equations and
marked OURS or ASSUMED where it appears: the servos and the throttle lag (the book's Aerosonde
has none), the guards that keep the model finite (``Aerosonde.guards``), the autopilot, the
guidance. Nothing here is validated against a real Aerosonde: it is the textbook's aircraft.

Nothing here imports threepp: ``Visual`` is handed the loaded model's root node.

FRAMES AND SIGNS
----------------
The book's: NED position (pn, pe, pd), body axes x out of the nose, y out of the right wing, z
out of the belly, attitude (phi, theta, psi) with psi from north toward east. The one mapping to
threepp's world (x east, y up, z south) is x8_rig's, at the boundary::

    world = (E, h0 - D, -N)            h0: the height of the NED origin in the world

The .glb's own frame is X forward, Y up, Z right: a body vector (x, y, z) is (x, -z, y) in it.

Control surfaces, the book's signs: a positive elevator is trailing edge down and pitches the
nose down (Cmde < 0); a positive aileron is the left aileron's trailing edge down and the right
one's up and rolls the right wing down (Clda > 0); a positive rudder is trailing edge to port and
yaws the nose to port (Cndr < 0). The real aircraft has an inverted V-tail: the book's elevator
and rudder are virtual surfaces, and only ``Visual`` mixes them onto the two ruddervators.

Wind is the velocity of the air over the ground, NED, m/s: (0, 5, 0) is air moving east at
5 m/s, a wind FROM the west. A 3-tuple, a callable (t, n, e, d) -> (wn, we, wd), or an
m350_rig.Wind (a mean profile with Dryden gusts). It is sampled once a step, at the state the
step starts from, and held across the Runge-Kutta stages.

TYPICAL USE
-----------
::

    from aerosonde_rig import Aerosonde, Autopilot, LOS, Visual, model_path

    a = Aerosonde(wind=(0.0, -8.0, 0.0))            # 8 m/s from the east
    a.trim(25.0, n=0.0, e=0.0, h=100.0, course=0.0) # level at 25 m/s, 100 m up, tracking north
    ap = Autopilot(a)
    los = LOS([(0, 0), (1000, 0), (1000, 800)])
    while ...:
        ap.command(course=los.update(a.n, a.e), altitude=100.0, airspeed=25.0)
        ap.update()                                  # surface and throttle commands
        a.step()                                     # one fixed step, a.dt
    visual = Visual(tp.GLTFLoader().load(model_path()).scene)
    visual.pose(a, h0=0.0)
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)                                 # examples/rigs/ (x8_rig, m350_rig)

from m350_rig import Wind  # noqa: E402
from x8_rig import (C_BM, C_WN, LOS as _LOS, clamp, cross_track, isa_density, ned_to_world,  # noqa: E402,F401
                    quat_of, rot_nb, world_to_ned, wrap)

UAV_DIR = os.path.join(os.path.dirname(HERE), "uav")
SPEC_PATH = os.path.join(UAV_DIR, "aerosonde_spec.json")
TWO_PI = 2.0 * math.pi


def load_spec(path=None):
    with open(path or SPEC_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def model_path():
    """The generated aerosonde.glb, or None where it has not been built."""
    p = os.path.join(UAV_DIR, "aerosonde.glb")
    return p if os.path.isfile(p) else None


# --------------------------------------------------------------------------- #
#  The aircraft: the book's model
# --------------------------------------------------------------------------- #
# Full state, in this order. The first 12 are the book's (chapter 3): NED position, body
# velocity, Euler angles, body rates. Then the actuators, which are ASSUMED (the book's Aerosonde
# has none): elevator, aileron and rudder with their rates, and the throttle.
STATE = ("n", "e", "d", "u", "v", "w", "phi", "theta", "psi", "p", "q", "r",
         "elevator", "elevator_rate", "aileron", "aileron_rate", "rudder", "rudder_rate", "throttle")
N_RIGID = 12
LON = (3, 5, 10, 7)                      # u, w, q, theta: the longitudinal block of the Jacobian
LAT = (4, 9, 11, 6)                      # v, p, r, phi: the lateral block


class Aerosonde:
    """The Aerosonde as the book models it.

    spec     aerosonde_spec.json as a dict (default: load it)
    dt       the fixed step, s (default 1/300: 60, 50 and 30 frames a second are whole numbers
             of steps)
    wind     NED wind, m/s: a 3-tuple, a callable (t, n, e, d) -> (wn, we, wd), or an
             m350_rig.Wind; sampled once a step and held across the Runge-Kutta stages
    h0       the NED origin's height above mean sea level (only an ISA air density follows it)
    rho      the air density in place of the spec's choice: a number, or "isa". The spec's
             default is the source's constant 1.2682 kg/m^3 at every height.

    The state is ``self.x`` (STATE order); ``self.cmd`` holds the commands the next step() uses
    (elevator, aileron, rudder in rad, throttle 0..1). ``self.out`` is what the last step
    evaluated at its start (airspeed, angles, the propeller's speed, thrust and torque, the
    coefficients, the guards)."""

    def __init__(self, spec=None, dt=1.0 / 300.0, wind=(0.0, 0.0, 0.0), h0=0.0, rho=None):
        self.spec = spec = load_spec() if spec is None else spec
        P, A, PR, PU, AC, EN, ENV = (spec["physical"], spec["aero"], spec["propeller"], spec["propulsion"],
                                     spec["actuators"], spec["environment"], spec["envelope"])
        self.m, self.Jx, self.Jy, self.Jz, self.Jxz = P["mass"], P["Jx"], P["Jy"], P["Jz"], P["Jxz"]
        self.S, self.b, self.c, self.e_os = P["S"], P["b"], P["c"], P["e"]
        self.AR = self.b * self.b / self.S
        self.g = P["gravity"]
        # the Gamma constants of chapter 3 (the inverse of the inertia with its x-z product)
        G = self.Jx * self.Jz - self.Jxz * self.Jxz
        self.gamma = (self.Jxz * (self.Jx - self.Jy + self.Jz) / G,
                      (self.Jz * (self.Jz - self.Jy) + self.Jxz * self.Jxz) / G,
                      self.Jz / G, self.Jxz / G,
                      (self.Jz - self.Jx) / self.Jy, self.Jxz / self.Jy,
                      ((self.Jx - self.Jy) * self.Jx + self.Jxz * self.Jxz) / G, self.Jx / G)
        self.a = A
        self.ct = (PR["CT0"], PR["CT1"], PR["CT2"])
        self.cq = (PR["CQ0"], PR["CQ1"], PR["CQ2"])
        self.D = PR["D"]
        self.KV = 60.0 / (TWO_PI * PU["KV_rpm_per_volt"])     # back-emf constant, V s/rad
        self.KQ = self.KV                                     # torque constant, N m/A: the source takes them equal
        self.Rm, self.i0, self.V_max = PU["R_motor"], PU["i0"], PU["V_max"]
        # ASSUMED: the actuators
        self.wn_s, self.zeta_s = AC["servo_wn"], AC["servo_zeta"]
        self.tau_t = AC["throttle_tau"]
        self.limits = (math.radians(AC["elevator_limit_deg"]), math.radians(AC["aileron_limit_deg"]),
                       math.radians(AC["rudder_limit_deg"]))
        # the air: the source's constant unless the spec or the caller asks for the ISA
        self.rho0 = EN["rho_sea_level"]
        choice = EN["rho"] if rho is None else rho
        self.rho_fixed = None if choice == "isa" else float(P["rho"] if choice == "source" else choice)
        self.h0 = h0
        # guards (OURS, envelope block of the spec)
        self.va_min = ENV["va_min"]
        self.guards = {"va": 0, "omega": 0}
        self.dt = dt
        self.x = [0.0] * len(STATE)
        self.cmd = [0.0, 0.0, 0.0, 0.0]
        self.t = 0.0
        self.steps = 0
        self.prop_angle = 0.0             # the drawn propeller's angle: Omega integrated, for Visual
        self.out = {}
        self._out_at = -1                 # the step count self.out was evaluated at
        self._w = (0.0, 0.0, 0.0)         # the wind at the current state, held across the stages
        self.wind = wind

    # ---- named access
    def __getattr__(self, name):
        if name in _INDEX:
            return self.x[_INDEX[name]]
        raise AttributeError(name)

    @property
    def h(self):
        """Height above the NED origin, m."""
        return -self.x[2]

    @property
    def wind(self):
        return self._wind

    @wind.setter
    def wind(self, w):
        self._wind = w
        self._steady = not (callable(w) or isinstance(w, Wind))
        self._w = self._wind_at(self.t, self.x[0], self.x[1], self.x[2])
        self._out_at = -1

    def rho_at(self, d):
        return self.rho_fixed if self.rho_fixed is not None else isa_density(self.h0 - d, self.rho0)

    def wind_ned(self):
        """The wind at the aircraft that the next step uses, NED m/s."""
        return self._w

    def _wind_at(self, t, n, e, d):
        """The wind at a point without moving a Wind's gusts on (its mean there)."""
        w = self._wind
        if isinstance(w, Wind):
            return w.mean(n, e, -d)
        if callable(w):
            w = w(t, n, e, d)
        return (float(w[0]), float(w[1]), float(w[2]))

    # ---- the book's equations
    def propeller(self, Va, throttle, rho):
        """The propeller's speed (rad/s), thrust (N) and torque (N m) at airspeed Va and this
        throttle: the authors' addendum to chapter 4, as their chapter 4 slides sum it up
        ("Propeller Thrust and Torque: Summary"). The motor's torque
        KQ ((V_in - KV Omega) / R - i0) equals the propeller's rho n^2 D^5 C_Q(J) at every
        instant (quasi-steady: the rotor has no state of its own), which is a quadratic in
        Omega, a Omega^2 + b Omega + c = 0; its positive root is taken. Returns (Omega, T, Q,
        guard)."""
        D = self.D
        q0, q1, q2 = self.cq
        V_in = self.V_max * throttle
        a = q0 * rho * D ** 5 / (TWO_PI * TWO_PI)
        b = q1 * rho * D ** 4 * Va / TWO_PI + self.KQ * self.KV / self.Rm
        c = q2 * rho * D ** 3 * Va * Va - self.KQ * V_in / self.Rm + self.KQ * self.i0
        disc = b * b - 4.0 * a * c
        guard = disc < 0.0
        # (-b + sqrt(b^2 - 4ac)) / (2a), written as -2c / (b + sqrt(..)): the same root without
        # the cancellation (b^2 is hundreds of times 4ac), and finite where there is no air (a = 0)
        Om = -2.0 * c / (b + math.sqrt(disc if disc > 0.0 else 0.0))
        if Om < 0.0:                                       # guard: the motor cannot turn the propeller
            Om, guard = 0.0, True                          # (throttle near zero at low airspeed); it stands still
        n = Om / TWO_PI
        vd = Va / D
        c0, c1, c2 = self.ct
        # T = rho n^2 D^4 C_T(J) and Q = rho n^2 D^5 C_Q(J) with J = Va / (n D), multiplied out
        # so that a stopped propeller (J infinite) needs no division
        T = rho * D ** 4 * (c0 * n * n + c1 * n * vd + c2 * vd * vd)
        Q = rho * D ** 5 * (q0 * n * n + q1 * n * vd + q2 * vd * vd)
        return Om, T, Q, guard

    def rigid_rates(self, xr, de, da, dr, thr, wb, out=None):
        """d/dt of the 12 rigid-body states xr = (pn, pe, pd, u, v, w, phi, theta, psi, p, q, r)
        for elevator, aileron and rudder deflections de, da, dr (rad), throttle thr (0..1) and
        the wind in body axes wb. Chapter and section numbers are the book's."""
        n_, e_, d_, u, v, w, phi, th, psi, p, q, r = xr
        A = self.a
        rho = self.rho_at(d_)
        # 4.4 air-relative velocity and air data
        ur, vr, wr = u - wb[0], v - wb[1], w - wb[2]
        Va = math.sqrt(ur * ur + vr * vr + wr * wr)
        Vg = max(Va, self.va_min)                          # guard: divisions by V_a below va_min
        alpha = math.atan2(wr, ur)
        beta = math.asin(clamp(vr / Vg, -1.0, 1.0))
        # addendum to 4.3: the propeller and the motor
        Om, T, Q, g_om = self.propeller(Va, thr, rho)
        # 4.2.2 lift: the linear wing blended into a flat plate past alpha0 (the stall)
        ca, sa = math.cos(alpha), math.sin(alpha)
        M, a0 = A["M"], A["alpha0"]
        e1, e2 = math.exp(-M * (alpha - a0)), math.exp(M * (alpha + a0))
        sig = (1.0 + e1 + e2) / ((1.0 + e1) * (1.0 + e2))
        cl_lin = A["CL0"] + A["CLalpha"] * alpha
        CLa = (1.0 - sig) * cl_lin + sig * (2.0 * math.copysign(1.0, alpha) * sa * sa * ca)
        # 4.2.2 drag: parasitic plus induced, the induced part from the LINEAR lift at every alpha.
        # (CD0 and CDalpha of the spec belong to the book's linear model and are not used here.)
        CDa = A["CDp"] + cl_lin * cl_lin / (math.pi * self.e_os * self.AR)
        qs = q * self.c / (2.0 * Vg)
        CL = CLa + A["CLq"] * qs + A["CLde"] * de
        CD = CDa + A["CDq"] * qs + A["CDde"] * de
        Cm = A["Cm0"] + A["Cmalpha"] * alpha + A["Cmq"] * qs + A["Cmde"] * de
        # 4.2.3 lateral: linear in beta, the rates and the surfaces
        b2v = self.b / (2.0 * Vg)
        ps, rs = p * b2v, r * b2v
        CY = A["CY0"] + A["CYbeta"] * beta + A["CYp"] * ps + A["CYr"] * rs + A["CYda"] * da + A["CYdr"] * dr
        Cl = A["Cl0"] + A["Clbeta"] * beta + A["Clp"] * ps + A["Clr"] * rs + A["Clda"] * da + A["Cldr"] * dr
        Cn = A["Cn0"] + A["Cnbeta"] * beta + A["Cnp"] * ps + A["Cnr"] * rs + A["Cnda"] * da + A["Cndr"] * dr
        # stability -> body: lift and drag turned through alpha; 4.1 gravity; the thrust along
        # body x through the centre of mass
        qS = 0.5 * rho * Va * Va * self.S
        cf, sf = math.cos(phi), math.sin(phi)
        ct, st = math.cos(th), math.sin(th)
        mg = self.m * self.g
        Fx = qS * (-CD * ca + CL * sa) + T - mg * st
        Fy = qS * CY + mg * ct * sf
        Fz = qS * (-CD * sa - CL * ca) + mg * ct * cf
        # the propeller turns positively about body x (clockwise seen from behind); the air's
        # torque on it, which the motor passes to the airframe, is the other way: -Q about x (4.3)
        L = qS * self.b * Cl - Q
        Mm = qS * self.c * Cm
        N = qS * self.b * Cn
        # chapter 3: translation, rotation (the Gamma form), position, Euler angles
        udot = r * v - q * w + Fx / self.m
        vdot = p * w - r * u + Fy / self.m
        wdot = q * u - p * v + Fz / self.m
        G1, G2, G3, G4, G5, G6, G7, G8 = self.gamma
        pdot = G1 * p * q - G2 * q * r + G3 * L + G4 * N
        qdot = G5 * p * r - G6 * (p * p - r * r) + Mm / self.Jy
        rdot = G7 * p * q - G1 * q * r + G4 * L + G8 * N
        cp, sp = math.cos(psi), math.sin(psi)
        ndot = ct * cp * u + (sf * st * cp - cf * sp) * v + (cf * st * cp + sf * sp) * w
        edot = ct * sp * u + (sf * st * sp + cf * cp) * v + (cf * st * sp - sf * cp) * w
        ddot = -st * u + sf * ct * v + cf * ct * w
        tt = st / ct
        phidot = p + sf * tt * q + cf * tt * r
        thdot = cf * q - sf * r
        psidot = (sf * q + cf * r) / ct
        if out is not None:
            nrev = Om / TWO_PI
            V_in = self.V_max * thr
            Im = (V_in - self.KV * Om) / self.Rm
            k4 = rho * nrev * nrev * self.D ** 4
            out.update(Va=Va, alpha=alpha, beta=beta, rho=rho, omega=Om, T=T, Q=Q,
                       J=Va / (nrev * self.D) if nrev > 1e-9 else math.inf,
                       CT=T / k4 if k4 > 1e-12 else 0.0, CQ=Q / (k4 * self.D) if k4 > 1e-12 else 0.0,
                       V_in=V_in, Im=Im, P_el=V_in * Im, sigma=sig, CL=CL, CD=CD, Cm=Cm, CY=CY, Cl=Cl, Cn=Cn,
                       lift=qS * CL, drag=qS * CD, de=de, da=da, dr=dr, throttle=thr,
                       Fx=Fx, Fy=Fy, Fz=Fz, L=L, M=Mm, N=N, g_va=Va < self.va_min, g_omega=g_om)
        return [ndot, edot, ddot, udot, vdot, wdot, phidot, thdot, psidot, pdot, qdot, rdot]

    def wind_body(self, xr, wn):
        """The NED wind in body axes: R_b^n^T w."""
        return rot_nb(xr[6], xr[7], xr[8]).T @ np.asarray(wn, float)

    def rates(self, x, c):
        """d/dt of the full state under the commands c = (elevator, aileron, rudder, throttle)
        held over the step, in the held wind: the rigid body, then the ASSUMED actuators."""
        a, b_, d = self._w
        if a == 0.0 and b_ == 0.0 and d == 0.0:
            wb = (0.0, 0.0, 0.0)
        else:
            cf, sf = math.cos(x[6]), math.sin(x[6])
            ct, st = math.cos(x[7]), math.sin(x[7])
            cp, sp = math.cos(x[8]), math.sin(x[8])
            wb = (ct * cp * a + ct * sp * b_ - st * d,
                  (sf * st * cp - cf * sp) * a + (sf * st * sp + cf * cp) * b_ + sf * ct * d,
                  (cf * st * cp + sf * sp) * a + (cf * st * sp - sf * cp) * b_ + cf * ct * d)
        dx = self.rigid_rates(x[:N_RIGID], x[12], x[14], x[16], x[18], wb)
        wn2, z2 = self.wn_s * self.wn_s, 2.0 * self.zeta_s * self.wn_s
        dx += [x[13], wn2 * (c[0] - x[12]) - z2 * x[13],  # second-order servos
               x[15], wn2 * (c[1] - x[14]) - z2 * x[15],
               x[17], wn2 * (c[2] - x[16]) - z2 * x[17],
               (c[3] - x[18]) / self.tau_t]               # first-order throttle
        return dx

    def evaluate(self):
        """Air data, forces and the rest at the current state (fills self.out)."""
        x = self.x
        out = {}
        self.rigid_rates(x[:N_RIGID], x[12], x[14], x[16], x[18], self.wind_body(x, self._w), out)
        out["wind"] = tuple(self._w)
        self.out = out
        self._out_at = self.steps
        return out

    # ---- stepping
    def step(self):
        """One fixed step: the commands clipped to the surfaces' throws and the throttle's
        range, the guards' tally, RK4 over dt in the held wind, the wind sampled for the next."""
        dt = self.dt
        le, la, lr = self.limits
        c = (clamp(self.cmd[0], -le, le), clamp(self.cmd[1], -la, la), clamp(self.cmd[2], -lr, lr),
             clamp(self.cmd[3], 0.0, 1.0))
        out = self.out if self._out_at == self.steps else self.evaluate()
        if out["g_va"]:
            self.guards["va"] += 1
        if out["g_omega"]:
            self.guards["omega"] += 1
        x = self.x
        k1 = self.rates(x, c)
        k2 = self.rates([a + 0.5 * dt * b for a, b in zip(x, k1)], c)
        k3 = self.rates([a + 0.5 * dt * b for a, b in zip(x, k2)], c)
        k4 = self.rates([a + dt * b for a, b in zip(x, k3)], c)
        self.x = x = [a + dt / 6.0 * (b1 + 2.0 * b2 + 2.0 * b3 + b4) for a, b1, b2, b3, b4 in zip(x, k1, k2, k3, k4)]
        self.prop_angle = (self.prop_angle + out["omega"] * dt) % TWO_PI
        self.t += dt
        self.steps += 1
        w = self._wind
        if isinstance(w, Wind):                            # the gusts move on one step, at the new state
            v = self.ground_velocity()
            self._w = w.advance(dt, x[0], x[1], -x[2], (float(v[0]), float(v[1]), float(v[2])))
        elif not self._steady:
            self._w = self._wind_at(self.t, x[0], x[1], x[2])

    def run(self, seconds, before_step=None):
        """Step for `seconds`; before_step(self) runs before every step (an autopilot's update)."""
        for _ in range(int(round(seconds / self.dt))):
            if before_step is not None:
                before_step(self)
            self.step()

    # ---- setting a state
    def set_state(self, n=0.0, e=0.0, d=0.0, u=25.0, v=0.0, w=0.0, phi=0.0, theta=0.0, psi=0.0,
                  p=0.0, q=0.0, r=0.0, de=0.0, da=0.0, dr=0.0, throttle=0.0):
        """The state, with the actuators at rest on the given deflections and throttle and the
        commands the same."""
        self.x = [n, e, d, u, v, w, phi, theta, psi, p, q, r, de, 0.0, da, 0.0, dr, 0.0, throttle]
        self.cmd = [de, da, dr, throttle]
        self._w = self._wind_at(self.t, n, e, d)
        self._out_at = -1
        self.evaluate()

    def trim(self, va, n=0.0, e=0.0, h=0.0, psi=0.0, gamma=0.0, apply=True, iters=60, course=None, throttle=None):
        """Straight flight at airspeed va and air-path angle gamma (rad) with no sideslip: solves
        the six rigid-body equations and the path angle for alpha, phi, theta, the three surfaces
        and the throttle, in still air. (The propeller's torque needs a little aileron, that a
        little rudder, and the two surfaces' side force a fraction of a degree of bank.) Given a
        throttle, that is held and the path angle is solved for instead: throttle=0.0 is the
        glide. With apply, the aircraft is put there (in the wind it has, at the same air-relative
        velocity), heading psi, or, given a course, at the heading that makes that ground track
        in the wind. Returns the solution as a dict; `limited` says a surface or the throttle is
        past what step() allows."""
        z = np.array([0.05, 0.0, 0.05 + gamma, -0.1, 0.0, 0.0, 0.4 if throttle is None else gamma])
        d = -h

        def split(z):
            al, ph, th, de, da, dr, last = (float(a) for a in z)
            thr, gam = (last, gamma) if throttle is None else (throttle, last)
            return al, ph, th, de, da, dr, thr, gam

        def resid(z, out=None):
            al, ph, th, de, da, dr, thr, gam = split(z)
            xr = [0.0, 0.0, d, va * math.cos(al), 0.0, va * math.sin(al), ph, th, 0.0, 0.0, 0.0, 0.0]
            f = self.rigid_rates(xr, de, da, dr, thr, (0.0, 0.0, 0.0), out)
            return np.array(f[3:6] + f[9:12] + [-f[2] - va * math.sin(gam)])
        for _ in range(iters):
            r0 = resid(z)
            Jm = np.zeros((7, 7))
            for k in range(7):
                zp, zm = z.copy(), z.copy()
                zp[k] += 1e-6
                zm[k] -= 1e-6
                Jm[:, k] = (resid(zp) - resid(zm)) / 2e-6
            step = np.linalg.solve(Jm, -r0)
            big = float(np.max(np.abs(step)))
            z = z + step * min(1.0, 0.3 / big) if big > 0.0 else z      # at most 0.3 rad (or 0.3 of throttle) a turn
            if big < 1e-12:
                break
        o = {}
        res = float(np.max(np.abs(resid(z, o))))
        al, ph, th, de, da, dr, thr, gam = split(z)
        le, la, lr = self.limits
        sol = {"va": va, "alpha": al, "beta": 0.0, "phi": ph, "theta": th, "gamma": gam, "de": de, "da": da, "dr": dr,
               "throttle": thr, "omega": o["omega"], "T": o["T"], "Q": o["Q"], "J": o["J"], "CL": o["CL"], "CD": o["CD"],
               "lift": o["lift"], "drag": o["drag"], "sigma": o["sigma"], "Im": o["Im"], "P_el": o["P_el"], "residual": res,
               "limited": abs(de) > le or abs(da) > la or abs(dr) > lr or not 0.0 <= thr <= 1.0}
        if apply:
            u, w = va * math.cos(al), va * math.sin(al)

            def place(psi):
                wb = rot_nb(ph, th, psi).T @ np.asarray(self._wind_at(self.t, n, e, d), float)
                self.set_state(n, e, d, u + wb[0], wb[1], w + wb[2], ph, th, psi, de=de, da=da, dr=dr, throttle=thr)
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
        """NED velocity over the ground."""
        x = self.x
        return rot_nb(x[6], x[7], x[8]) @ np.array(x[3:6])

    def air_velocity_ned(self):
        return self.ground_velocity() - np.asarray(self._w, float)

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
        NED origin, J. The propeller's own spin is not in it (the model gives it no inertia)."""
        x = self.x
        u, v, w, p, q, r = x[3], x[4], x[5], x[9], x[10], x[11]
        rot = 0.5 * (self.Jx * p * p + self.Jy * q * q + self.Jz * r * r) - self.Jxz * p * r
        return 0.5 * self.m * (u * u + v * v + w * w) + rot - self.m * self.g * x[2]

    def angular_momentum_ned(self):
        """The angular momentum about the centre of mass, NED axes."""
        x = self.x
        p, q, r = x[9], x[10], x[11]
        hb = np.array([self.Jx * p - self.Jxz * r, self.Jy * q, self.Jz * r - self.Jxz * p])
        return rot_nb(x[6], x[7], x[8]) @ hb

    def jacobian(self, xr=None, ui=None):
        """Numerical Jacobians of rigid_rates at the 12 states xr and the inputs ui = (de, da,
        dr, throttle, u_w, v_w, w_w) (wind in body axes): central differences. Default: the
        current state, in still air. Returns (A, B), 12 x 12 and 12 x 7."""
        x = self.x
        xr = np.asarray(x[:N_RIGID] if xr is None else xr, float)
        ui = np.asarray([x[12], x[14], x[16], x[18], 0.0, 0.0, 0.0] if ui is None else ui, float)

        def f(xv, uv):
            return np.array(self.rigid_rates(list(xv), uv[0], uv[1], uv[2], uv[3], tuple(uv[4:7])))
        A = np.zeros((N_RIGID, N_RIGID))
        B = np.zeros((N_RIGID, 7))
        for k in range(N_RIGID):
            hk = 1e-6 * max(1.0, abs(xr[k]))
            xp, xm = xr.copy(), xr.copy()
            xp[k] += hk
            xm[k] -= hk
            A[:, k] = (f(xp, ui) - f(xm, ui)) / (2.0 * hk)
        for k in range(7):
            hk = 1e-6 * max(1.0, abs(ui[k]))
            up, um = ui.copy(), ui.copy()
            up[k] += hk
            um[k] -= hk
            B[:, k] = (f(xr, up) - f(xr, um)) / (2.0 * hk)
        return A, B


_INDEX = {k: i for i, k in enumerate(STATE)}
AIRCRAFT = Aerosonde


def modes(A):
    """The five classical modes of a 12 x 12 Jacobian from Aerosonde.jacobian, from its
    decoupled blocks (u, w, q, theta) and (v, p, r, phi): the eigenvalues, by name. The short
    period is the faster longitudinal pair and the phugoid the slower (a pair that has split
    into two real roots is given as the root nearer zero); the roll mode is the faster real
    lateral root, the spiral the slower, the dutch roll the pair."""
    el = np.linalg.eigvals(A[np.ix_(LON, LON)])
    ea = np.linalg.eigvals(A[np.ix_(LAT, LAT)])
    cl = sorted([e for e in el if e.imag > 1e-9], key=lambda e: -abs(e))
    rl = sorted([e for e in el if abs(e.imag) <= 1e-9], key=lambda e: abs(e))
    ca = [e for e in ea if e.imag > 1e-9]
    ra = sorted([e.real for e in ea if abs(e.imag) <= 1e-9])
    return {"short_period": cl[0] if cl else rl[-1], "phugoid": cl[1] if len(cl) > 1 else rl[0],
            "roll": ra[0] if ra else None, "dutch_roll": ca[0] if ca else None, "spiral": ra[-1] if ra else None}


# --------------------------------------------------------------------------- #
#  The autopilot: OURS
# --------------------------------------------------------------------------- #
class Autopilot:
    """Successive loop closure, as in the book's chapter 6:

        roll     d_a = d_a* + kp (phi_c - phi) + ki int(phi_c - phi) - kd p
        course   phi_c = kp (chi_c - chi) + ki int(chi_c - chi)      (or heading psi in place of chi)
        pitch    d_e = d_e* + kp (theta_c - theta) + ki int(theta_c - theta) - kd q
        altitude theta_c = theta* + kp (h_c - h) + ki int(h_c - h), saturated outside a zone
        airspeed d_t = d_t* + kp (V_c - V_a) + ki int(V_c - V_a)
        rudder   d_r = d_r* + k_r (r - its own low-pass): a yaw damper

    The starred values are the still-air trim at the commanded airspeed. The roll and pitch
    gains come from the spec's coefficients at that airspeed, through the book's second-order
    models phi'' = -a_phi1 phi' + a_phi2 d_a and theta'' = -a_th1 theta' - a_th2 theta + a_th3 d_e:
    roll from its natural frequency (rate feedback only where the airframe's own roll damping is
    short of 2 zeta wn), pitch from a natural frequency set as a multiple of the airframe's own
    stiffness, sqrt((1 + stiffness) a_th2), so that it follows the airspeed and the proportional
    gain keeps its sign at every speed, with the book's rate feedback for the damping. An
    integral in the pitch loop makes its DC gain 1 (the book leaves that to the altitude loop).
    The course loop's gains come from the roll loop's and a bandwidth ratio, the altitude loop's
    from its own natural frequency (h' = V_a theta), the airspeed loop's from the model's own
    d(u')/du and d(u')/d(throttle) at the trim.

    The rudder: a yaw damper, not a sideslip hold. The model's steady sideslip in a 30 deg banked
    turn is under a degree without any rudder, so there is little for a sideslip loop to do,
    while the dutch roll is lightly damped; and a rudder proportional to beta stiffens that mode
    without damping it. The gain adds what the dutch roll of the linearised model lacks to the
    spec's damping, through d(r')/d(d_r); the washout keeps the rudder out of a steady turn.

    The spec's autopilot block holds the frequencies, dampings, integrals and limits, nothing
    else. Everything is fed the EXACT state: no estimator, no sensor noise.

    hold: "course" (the ground track, which turns into the wind and compensates for it),
    "heading" (the nose, which drifts with the wind) or "roll" (``command(roll=...)``: the roll
    angle is handed in, for an outer course loop of the caller's own)."""

    def __init__(self, aircraft, airspeed=None, hold="course"):
        self.ac = aircraft
        cfg = aircraft.spec["autopilot"]
        self.cfg = cfg
        self.hold = hold
        self.va_c = airspeed or cfg["airspeed"]
        self.chi_c = aircraft.x[8]
        self.h_c = aircraft.h
        self.design(self.va_c)
        self.i_phi = self.i_chi = self.i_h = self.i_v = self.i_th = 0.0
        self.phi_c = self.theta_c = self.phi_cmd = 0.0
        self.r_lp = aircraft.x[11]

    def design(self, va):
        """The gains at airspeed va (and the air density at the aircraft's height)."""
        ac, A, cfg = self.ac, self.ac.a, self.cfg
        probe = Aerosonde(ac.spec, dt=ac.dt, h0=ac.h0, rho="isa" if ac.rho_fixed is None else ac.rho_fixed)
        tr = probe.trim(va, h=ac.h, apply=False)
        self.trim = tr
        rho = ac.rho_at(ac.x[2])
        qS = 0.5 * rho * va * va * ac.S
        G3, G4 = ac.gamma[2], ac.gamma[3]
        # roll: C_p = Gamma_3 C_l + Gamma_4 C_n (the book's chapter 5 coefficients)
        a1 = -qS * ac.b * (G3 * A["Clp"] + G4 * A["Cnp"]) * ac.b / (2.0 * va)
        a2 = qS * ac.b * (G3 * A["Clda"] + G4 * A["Cnda"])
        c = cfg["roll"]
        w, z = c["wn"], c["zeta"]
        self.a_phi = (a1, a2)
        self.kp_phi = w * w / a2
        # rate feedback only where the airframe's own roll damping is short of 2 zeta wn (at
        # 25 m/s a_phi1 is 22.6 /s and it is not)
        self.kd_phi = max(0.0, (2.0 * z * w - a1) / a2)
        self.ki_phi = c["ki"] * self.kp_phi
        self.i_phi_max = math.radians(c["integral_deg"]) / self.ki_phi    # the integral's share of the aileron
        self.phi_max = math.radians(c["limit_deg"])
        # course (chi' = g / V_g tan phi)
        c = cfg["course"]
        self.w_chi = w / c["bandwidth_ratio"]
        self.z_chi = c["zeta"]
        # pitch: theta'' = -a_th1 theta' - a_th2 theta + a_th3 d_e, closed to wn^2 = (1 + stiffness)
        # a_th2: kp = stiffness a_th2 / a_th3 = -stiffness Cmalpha / Cmde, the same at every speed
        a1 = -qS * ac.c * A["Cmq"] * ac.c / (2.0 * va) / ac.Jy
        a2 = -qS * ac.c * A["Cmalpha"] / ac.Jy
        a3 = qS * ac.c * A["Cmde"] / ac.Jy
        self.a_th = (a1, a2, a3)
        c = cfg["pitch"]
        self.w_th = math.sqrt((1.0 + c["stiffness"]) * a2)
        self.kp_th = c["stiffness"] * a2 / a3
        self.kd_th = (2.0 * c["zeta"] * self.w_th - a1) / a3
        self.ki_th = c["wi"] * self.kp_th
        self.i_th_max = math.radians(c["integral_deg"]) / abs(self.ki_th)
        self.th_max = math.radians(c["limit_deg"])
        # altitude (h' = V_a theta, the pitch loop's DC gain 1 by its integral)
        c = cfg["altitude"]
        wh = c["wn"]
        self.kp_h = 2.0 * c["zeta"] * wh / va
        self.ki_h = wh * wh / va
        self.h_zone = c["zone"]
        # the linearised model at the trim: the airspeed loop and the yaw damper read it
        xr = [0.0, 0.0, ac.x[2], va * math.cos(tr["alpha"]), 0.0, va * math.sin(tr["alpha"]), tr["phi"], tr["theta"],
              0.0, 0.0, 0.0, 0.0]
        Aj, Bj = probe.jacobian(xr, [tr["de"], tr["da"], tr["dr"], tr["throttle"], 0.0, 0.0, 0.0])
        self.modes = modes(Aj)
        # airspeed on throttle: V_a' = -a_V1 V_a + a_V2 d_t
        av1, av2 = -Aj[3, 3], Bj[3, 3]
        c = cfg["speed"]
        w, z = c["wn"], c["zeta"]
        self.kp_v = (2.0 * z * w - av1) / av2
        self.ki_v = w * w / av2
        self.t_lim = tuple(c["throttle"])
        self.a_v = (av1, av2)
        # yaw damper: d(r')/d(d_r) is negative (Cndr < 0), so a positive gain on r adds damping
        c = cfg["yaw_damper"]
        dr_mode = self.modes["dutch_roll"]
        w_dr = abs(dr_mode)
        z_dr = -dr_mode.real / w_dr
        self.k_r = max(0.0, 2.0 * w_dr * (c["zeta"] - z_dr) / -Bj[11, 2])
        self.p_wo = c["washout"]

    def command(self, course=None, heading=None, altitude=None, airspeed=None, roll=None):
        """The guidance hook: a course (ground track) or a heading, rad from north toward east,
        or a roll angle, rad (right wing down +); a height above the NED origin, m; an airspeed,
        m/s. What is not given is kept."""
        if course is not None:
            self.chi_c, self.hold = course, "course"
        if heading is not None:
            self.chi_c, self.hold = heading, "heading"
        if roll is not None:
            self.phi_cmd, self.hold = roll, "roll"
        if altitude is not None:
            self.h_c = altitude
        if airspeed is not None and abs(airspeed - self.va_c) > 1e-9:
            self.va_c = airspeed
            self.design(airspeed)

    def aim_at(self, n, e, altitude=None, airspeed=None):
        """Pure pursuit: the course straight at the point (n, e)."""
        x = self.ac.x
        self.command(course=math.atan2(e - x[1], n - x[0]), altitude=altitude, airspeed=airspeed)

    def update(self):
        """Writes the aircraft's surface and throttle commands for its next step."""
        ac, dt = self.ac, self.ac.dt
        x = ac.x
        tr = self.trim
        out = ac.out if ac._out_at == ac.steps else ac.evaluate()
        phi, theta, psi, p, q, r = x[6], x[7], x[8], x[9], x[10], x[11]
        # course (or heading) -> roll
        if self.hold == "roll":
            phi_c = self.phi_cmd                           # the caller's own outer loop gives the roll angle
        else:
            if self.hold == "course":
                vg = ac.ground_velocity()
                chi = math.atan2(vg[1], vg[0])
                spd = max(math.hypot(vg[0], vg[1]), 5.0)
            else:
                chi, spd = psi, max(out["Va"], 5.0)
            e_chi = wrap(self.chi_c - chi)
            kp = 2.0 * self.z_chi * self.w_chi * spd / ac.g
            ki = self.w_chi * self.w_chi * spd / ac.g
            phi_c = kp * e_chi + ki * self.i_chi
            if abs(phi_c) < self.phi_max or phi_c * e_chi < 0.0:
                self.i_chi += e_chi * dt                   # anti-windup: hold the integral while saturated
        self.phi_c = phi_c = clamp(phi_c, -self.phi_max, self.phi_max)
        # roll -> aileron
        e_phi = phi_c - phi
        da = tr["da"] + self.kp_phi * e_phi + self.ki_phi * self.i_phi - self.kd_phi * p
        self.i_phi = clamp(self.i_phi + e_phi * dt, -self.i_phi_max, self.i_phi_max)
        # altitude -> pitch
        e_h = self.h_c - ac.h
        if abs(e_h) > self.h_zone:
            th_c = math.copysign(self.th_max, e_h)
        else:
            th_c = tr["theta"] + self.kp_h * e_h + self.ki_h * self.i_h
            if abs(th_c) < self.th_max or th_c * e_h < 0.0:
                self.i_h += e_h * dt
        self.theta_c = th_c = clamp(th_c, -self.th_max, self.th_max)
        # pitch -> elevator
        e_th = th_c - theta
        de = tr["de"] + self.kp_th * e_th + self.ki_th * self.i_th - self.kd_th * q
        self.i_th = clamp(self.i_th + e_th * dt, -self.i_th_max, self.i_th_max)
        # airspeed -> throttle
        e_v = self.va_c - out["Va"]
        thr = tr["throttle"] + self.kp_v * e_v + self.ki_v * self.i_v
        if self.t_lim[0] < thr < self.t_lim[1] or thr * e_v < 0.0:
            self.i_v += e_v * dt
        thr = clamp(thr, *self.t_lim)
        # yaw rate, washed out -> rudder
        dr = tr["dr"] + self.k_r * (r - self.r_lp)
        self.r_lp += self.p_wo * (r - self.r_lp) * dt
        ac.cmd = [de, da, dr, thr]                         # the throws are clipped in Aerosonde.step

    def status(self):
        return (f"chi_c {math.degrees(self.chi_c):6.1f}  phi_c {math.degrees(self.phi_c):+5.1f}  "
                f"theta_c {math.degrees(self.theta_c):+5.1f}  h_c {self.h_c:5.1f}  V_c {self.va_c:4.1f}")


# --------------------------------------------------------------------------- #
#  Guidance: x8_rig's, with this aircraft's spec
# --------------------------------------------------------------------------- #
class LOS(_LOS):
    """x8_rig.LOS (line of sight along a polyline of waypoints [(n, e), ...], OURS) reading its
    look-ahead, its hand-over rule, the roll limit, gravity and the airspeed from
    aerosonde_spec.json: x8_rig.LOS(spec=None) would load the X8's."""

    def __init__(self, waypoints, lookahead=None, accept=None, loop=False, spec=None):
        super().__init__(waypoints, lookahead=lookahead, accept=accept, loop=loop, spec=spec or load_spec())


# --------------------------------------------------------------------------- #
#  The 3D model
# --------------------------------------------------------------------------- #
class Visual:
    """aerosonde.glb's nodes posed from an Aerosonde's state.

    root    the loaded model's scene node (tp.GLTFLoader().load(model_path()).scene); anything
            with position.set, quaternion.set and get_object_by_name whose nodes have rotation

    pose(aircraft, h0) places the root (the .glb's origin is the centre of mass) from the
    state's position and attitude (world = (E, h0 - D, -N)) and turns

      * ``aileron_left``, ``aileron_right`` about their hinges (local +Z, positive trailing
        edge down) by +d_a and -d_a: the book's positive aileron, which rolls the right wing down;
      * ``ruddervator_left``, ``ruddervator_right`` about theirs (local +Z, positive trailing
        edge toward the panel's lower face) by (d_e - d_r) / 2 and (d_e + d_r) / 2: the halves
        of the book's V-tail mixing (chapter 4, control surfaces: d_e = d_rr + d_rl), with the
        rudder's sense the one an INVERTED V needs, which is the opposite of the book's upright
        V. A positive elevator puts both trailing edges down (nose down); a positive rudder puts
        both to port, which pushes the tail to starboard and yaws the nose to port (Cndr < 0);
      * ``propeller`` about local +X by the model's integrated Omega (its spin vector along body
        +x, clockwise seen from behind: the sense that makes the airframe's reaction -Q)."""

    NODE_NAMES = ("aileron_left", "aileron_right", "ruddervator_left", "ruddervator_right", "propeller")

    def __init__(self, root):
        self.root = root
        self.nodes = {n: root.get_object_by_name(n) for n in self.NODE_NAMES}
        missing = [k for k, v in self.nodes.items() if v is None]
        if missing:
            raise KeyError(f"aerosonde.glb has no node(s) {missing}: rebuild it with uav/build_aerosonde_blender.py")

    def pose(self, aircraft, h0=0.0):
        x, N = aircraft.x, self.nodes
        self.root.position.set(*(float(v) for v in aircraft.world_position(h0)))
        self.root.quaternion.set(*(float(v) for v in quat_of(aircraft.world_rotation())))
        de, da, dr = x[12], x[14], x[16]
        N["aileron_left"].rotation.z = da
        N["aileron_right"].rotation.z = -da
        N["ruddervator_left"].rotation.z = 0.5 * (de - dr)
        N["ruddervator_right"].rotation.z = 0.5 * (de + dr)
        N["propeller"].rotation.x = aircraft.prop_angle
