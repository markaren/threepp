"""
bixler3_rig -- the HobbyKing Bixler 3 (the "Bix3" of a Virginia Tech thesis): its identified
flight model, a throttle the thesis does not have, an autopilot and a guidance law on top, and
its 3D model posed from the model's state.

Four things, and nothing scene-specific:

  * ``Bixler3``   -- the 6-DOF model of B. M. Simmons, "System Identification of a Nonlinear
                     Flight Dynamics Model for a Small, Fixed-Wing UAV", Master's thesis,
                     Virginia Tech, 2018, http://hdl.handle.net/10919/95324: the rigid body of
                     its eqs. (2.7)-(2.12), the dynamic pressure (2.13), and the body-axis force
                     and moment coefficients of its eqs. (4.2)-(4.7) in the non-dimensional
                     states of (4.1), with the estimates of its Tables 4.5 and 4.7 and the mass,
                     geometry and inertia of its Tables 3.2 and 3.4 (uav/bixler3_spec.json). The
                     repository page states no licence: the equations and numbers are cited,
                     no text, figure or code is copied. Fixed-step fourth-order Runge-Kutta.
  * ``Autopilot`` -- OURS, not the thesis's: successive loop closure on the three surfaces and
                     the throttle (roll and pitch inside, course or heading on roll, altitude
                     on pitch, airspeed on throttle, the rudder holding the sideslip at zero),
                     its inner gains computed from the thesis's coefficients.
  * ``LOS``       -- OURS: x8_rig's line-of-sight guidance with this aircraft's spec.
  * ``Visual``    -- the bixler3.glb posed from a Bixler3: the airframe from the state, the
                     ailerons, the elevator and the rudder by their deflections, the propeller
                     by its speed.

THE THESIS HAS NO THROTTLE. It did not separate the propeller's thrust from the airframe's drag:
its C_X is the two together, at whatever constant throttle each manoeuvre held (not recorded in
it). On its own the identified model flies level at one speed only, the one where its C_X
balances (``Bixler3.level_speed()``: 15.3 m/s). What is added here to make it an aircraft
one can fly, all of it OURS or ASSUMED and marked so where it appears:

  * the throttle: X = qbar S C_X + [T(Omega_p, V_a) - T(Omega_ref, V_a)], T an ASSUMED propeller
    law for the vendor's 7x5 propeller and Omega_ref the speed it turns at the throttle the
    thesis's flights are TAKEN to have held (``throttle_ref``, from an ASSUMED lift-to-drag
    ratio; the spec's `propulsion` block). At that throttle the bracket is zero and the model
    is the thesis's exactly. The split of C_X into thrust and drag is an assumption: nothing
    about the propulsion is the thesis's, and glide and climb performance follow from it;
  * the servos, the deflection limits, the air density and gravity (the thesis gives none);
  * the wind: the thesis took the air as still, so its body velocity is the air-relative one,
    and with wind the air-relative body velocity stands where the thesis has u, v, w;
  * the guards that hold what enters the coefficients inside a range (``Bixler3.guards``): the
    model has no stall, and its w^2 terms bend the lift over far outside the data.

To fly the thesis's model untouched: leave the throttle at ``throttle_ref`` and stay off the
guards::

    b = Bixler3()
    b.level_speed(apply=True, h=60.0)      # its one level speed, its trim there, throttle_ref held
    b.run(10.0)                            # b.cmd[3] stays b.throttle_ref: out["dT"] is 0.0, b.guards all 0

Nothing here imports threepp: ``Visual`` is handed the loaded model's root node.

FRAMES
------
Inside the model everything is the thesis's: NED position (N, E, D), body axes x out of the
nose, y out of the right wing, z out of the belly, attitude (phi, theta, psi) with psi from
north toward east. The mapping to threepp's world (x east, y up, z south) is x8_rig's, at the
boundary: world = (E, h0 - D, -N). The .glb's own frame is X forward, Y up, Z right.

SIGNS of the deflections (rad). The thesis states none; they follow from its coefficients and
are the NASA (Klein and Morelli) ones: positive elevator is trailing edge down (Cmde < 0, nose
down); positive aileron rolls LEFT (Clda < 0: right aileron trailing edge down, left one up);
positive rudder is trailing edge to port and yaws the nose left (Cndr < 0).

Wind is the velocity of the air over the ground, NED, m/s: (0, 4, 0) is air moving east at
4 m/s, a wind FROM the west. A 3-tuple, a callable (t, n, e, d) -> (wn, we, wd), or an
m350_rig.Wind (a mean wind profile and Dryden gusts, moved on once a step).

TYPICAL USE
-----------
::

    from bixler3_rig import Bixler3, Autopilot, LOS, Visual, model_path

    b = Bixler3(wind=(0.0, 4.0, 0.0))              # 4 m/s from the west
    b.trim(12.0, n=0.0, e=0.0, h=60.0, course=0.0) # level at 12 m/s, 60 m up, tracking north
    ap = Autopilot(b)
    los = LOS([(0, 0), (300, 0), (300, 250)])
    while ...:
        ap.command(course=los.update(b.n, b.e), altitude=60.0, airspeed=12.0)
        ap.update()                                  # surface and throttle commands
        b.step()                                     # one fixed step, b.dt
    visual = Visual(tp.GLTFLoader().load(model_path()).scene)
    visual.pose(b, h0=0.0)
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)                                 # examples/rigs/ (x8_rig, m350_rig)

from m350_rig import Wind  # noqa: E402,F401
from x8_rig import LOS as _LOS  # noqa: E402
from x8_rig import (C_BM, C_WN, clamp, cross_track, isa_density, ned_to_world, quasi_steady,  # noqa: E402,F401
                    quat_of, rot_nb, world_to_ned, wrap)

UAV_DIR = os.path.join(os.path.dirname(HERE), "uav")
SPEC_PATH = os.path.join(UAV_DIR, "bixler3_spec.json")
TWO_PI = 2.0 * math.pi


def load_spec(path=None):
    with open(path or SPEC_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def model_path():
    """The generated bixler3.glb, or None where it has not been built."""
    p = os.path.join(UAV_DIR, "bixler3.glb")
    return p if os.path.isfile(p) else None


# --------------------------------------------------------------------------- #
#  The aircraft: the thesis's model, and a throttle
# --------------------------------------------------------------------------- #
# Full state, in this order. The first 12 are the thesis's: position, velocity and angular rate
# in body axes, the Euler angles. The propeller's speed and the servos are OURS (the thesis has
# neither a propeller nor actuator dynamics).
STATE = ("n", "e", "d", "u", "v", "w", "phi", "theta", "psi", "p", "q", "r", "omega",
         "elevator", "elevator_rate", "aileron", "aileron_rate", "rudder", "rudder_rate")
N_RIGID = 13


class Bixler3:
    """The Bixler 3 as the thesis models it, with a throttle that is ours.

    spec     bixler3_spec.json as a dict (default: load it)
    dt       the fixed step, s (default 1/300: 60, 50 and 30 frames a second are whole numbers
             of steps)
    wind     NED wind, m/s: a 3-tuple, a callable (t, n, e, d) -> (wn, we, wd), or an
             m350_rig.Wind (sampled once a step and held across the Runge-Kutta stages)
    h0       the NED origin's height above mean sea level (the air density follows it)
    rho      a fixed air density in place of the spec's (number, or None)

    The state is ``self.x`` (STATE order); ``self.cmd`` holds the commands the next step()
    takes: elevator, aileron, rudder (rad, the thesis's signs) and throttle (0..1, the
    propeller's speed as a share of its full-throttle speed). ``self.out`` is what the last
    step evaluated at its start (airspeed, angles, the six coefficients, our thrust and its
    difference from the reference, the guards). ``self.throttle_ref`` is the throttle at which
    the model is the thesis's."""

    def __init__(self, spec=None, dt=1.0 / 300.0, wind=(0.0, 0.0, 0.0), h0=0.0, rho=None):
        self.spec = spec = load_spec() if spec is None else spec
        P, A, PU, AC, EN, ENV = (spec["physical"], spec["aero"], spec["propulsion"], spec["actuators"],
                                 spec["environment"], spec["envelope"])
        self.m, self.Ix, self.Iy, self.Iz, self.Ixz = P["mass"], P["Ix"], P["Iy"], P["Iz"], P["Ixz"]
        self.cbar, self.b, self.S = P["cbar"], P["b"], P["S"]
        G = self.Ix * self.Iz - self.Ixz * self.Ixz
        self.Gx, self.Gxz, self.Gz = self.Iz / G, self.Ixz / G, self.Ix / G      # I^-1, x-z block
        self.a = A
        self.Vo = A["Vo"]                                    # the thesis's FIXED reference airspeed, (4.1)
        # the propeller and the motor: ASSUMED (propulsion block of the spec)
        self.D, self.CT0, self.J0 = PU["D"], PU["CT0"], PU["J0"]
        self.om_max = TWO_PI * PU["n_max"]                   # rad/s at full throttle
        self.tau_m = PU["motor_tau"]
        self.throttle_ref = PU["throttle_ref"]
        self.om_ref = self.throttle_ref * self.om_max
        # servos and throws: ASSUMED (actuators block)
        self.wn_s, self.zeta_s = AC["servo_wn"], AC["servo_zeta"]
        self.de_lim = math.radians(AC["elevator_limit_deg"])
        self.da_lim = math.radians(AC["aileron_limit_deg"])
        self.dr_lim = math.radians(AC["rudder_limit_deg"])
        self.g = EN["g"]
        self.rho0 = EN["rho_sea_level"]
        self.rho_fixed = rho if rho is not None else (None if EN["rho"] == "isa" else float(EN["rho"]))
        self.h0 = h0
        # guards (OURS, envelope block of the spec): what enters the thesis's coefficients is
        # held inside a range; the equations themselves are untouched
        self.wh_lim = tuple(ENV["wh"])
        self.vh_lim = tuple(ENV["vh"])
        self.guards = {"wh": 0, "vh": 0, "j": 0}
        self.dt = dt
        self.wind = wind
        self._w = (0.0, 0.0, 0.0)             # the wind at the aircraft now, NED: held over a step
        self.x = [0.0] * len(STATE)
        self.cmd = [0.0, 0.0, 0.0, self.throttle_ref]
        self.t = 0.0
        self.steps = 0
        self.prop_angle = 0.0                 # the drawn propeller's angle: Omega_p integrated, for Visual
        self.out = {}
        self._out_at = -1                     # the step count self.out was evaluated at
        self.x[3] = self.Vo
        self.x[12] = self.om_ref

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

    # ---- the wind
    def wind_ned(self):
        """The wind at the aircraft, NED m/s: what the next step flies in."""
        return self._w

    def _wind_at(self, t, n, e, d):
        """The wind at a point; of a Wind its mean, without moving the gusts on."""
        w = self.wind
        if isinstance(w, Wind):
            return w.mean(n, e, -d)
        if callable(w):
            w = w(t, n, e, d)
        return (float(w[0]), float(w[1]), float(w[2]))

    def _sample_wind(self):
        """The wind for the step to come; a Wind's gusts move on by one step here."""
        w = self.wind
        x = self.x
        if isinstance(w, Wind):
            v = self.ground_velocity()
            return w.advance(self.dt, x[0], x[1], -x[2], (v[0], v[1], v[2]))
        return self._wind_at(self.t, x[0], x[1], x[2])

    # ---- the propeller: ASSUMED
    def thrust(self, om, va, rho):
        """T = rho D^4 CT0 n^2 (1 - J / J0), n = Omega_p / 2 pi in rev/s and J = V_a / (n D): a
        static thrust coefficient falling in a straight line to zero at the advance ratio J0.
        Past J0 the thrust is held at zero (a windmilling propeller's drag is not modelled)."""
        n = om / TWO_PI
        if n <= 0.0:
            return 0.0
        T = rho * self.D ** 4 * self.CT0 * n * (n - va / (self.D * self.J0))
        return T if T > 0.0 else 0.0

    def reference_throttle(self, lift_to_drag=None, h=0.0):
        """The throttle the thesis's flights are TAKEN to have held (ASSUMED): the one at which
        our propeller gives m g / (L/D) at the speed where the thesis's model flies level
        hands-off. The spec's `propulsion.throttle_ref` is this number for its `lift_to_drag`."""
        ld = self.spec["propulsion"]["lift_to_drag"] if lift_to_drag is None else lift_to_drag
        va = self.level_speed(h=h)["va"]
        k = self.rho_at(-h) * self.D ** 4 * self.CT0
        c = va / (self.D * self.J0)
        n = 0.5 * (c + math.sqrt(c * c + 4.0 * self.m * self.g / ld / k))    # k n (n - c) = m g / (L/D)
        return TWO_PI * n / self.om_max

    # ---- the thesis's equations
    def rigid_rates(self, xr, de, da, dr, thr, wb, out=None):
        """d/dt of the 13 states xr = (n, e, d, u, v, w, phi, theta, psi, p, q, r, Omega_p) for
        elevator, aileron and rudder deflections de, da, dr (rad), throttle thr (0..1) and the
        wind in body axes wb. Equation numbers are the thesis's; what has none is ours."""
        n_, e_, d_, u, v, w, phi, th, psi, p, q, r, Om = xr
        A = self.a
        Vo = self.Vo
        rho = self.rho_at(d_)
        # the thesis assumed still air: the air-relative body velocity stands where it has u, v, w (OURS)
        ua, va, wa = u - wb[0], v - wb[1], w - wb[2]
        Va2 = ua * ua + va * va + wa * wa
        Va = math.sqrt(Va2)
        # (4.1) the non-dimensional states, over the FIXED Vo = 12 m/s, not over V_a
        uh, vh, wh = ua / Vo, va / Vo, wa / Vo
        ph, qh, rh = p * self.b / (2.0 * Vo), q * self.cbar / (2.0 * Vo), r * self.b / (2.0 * Vo)
        whg = clamp(wh, *self.wh_lim)                      # guard: the lift's quadratic turns over at wh = 0.38
        vhg = clamp(vh, *self.vh_lim)                      # guard: sideslip past what a doublet reaches
        # (4.2)-(4.7) the coefficients, total (not perturbations from a trim)
        CX = A["CXu"] * uh + A["CXw"] * whg + A["CXw2"] * whg * whg + A["CXo"]
        CZ = A["CZw"] * whg + A["CZq"] * qh + A["CZde"] * de + A["CZw2"] * whg * whg + A["CZo"]
        Cm = A["Cmw"] * whg + A["Cmq"] * qh + A["Cmde"] * de + A["Cmo"]
        CY = A["CYv"] * vhg + A["CYp"] * ph + A["CYr"] * rh + A["CYda"] * da + A["CYdr"] * dr + A["CYo"]
        Cl = A["Clv"] * vhg + A["Clp"] * ph + A["Clr"] * rh + A["Clda"] * da + A["Cldr"] * dr + A["Clo"]
        Cn = (A["Cnv"] * vhg + A["Cnp"] * ph + A["Cnr"] * rh + A["Cnda"] * da + A["Cndr"] * dr
              + A["Cnv2"] * vhg * vhg + A["Cno"])
        # OURS: the throttle. The thesis's C_X holds the thrust of its flights' throttle; ours adds
        # the difference between the propeller at its speed now and at the reference speed.
        T = self.thrust(Om, Va, rho)
        T_ref = self.thrust(self.om_ref, Va, rho)
        dT = T - T_ref
        Om_dot = (thr * self.om_max - Om) / self.tau_m     # the motor: a first-order lag (ASSUMED)
        # (2.13) forces and moments in body axes, qbar at the airspeed of the moment
        qS = 0.5 * rho * Va2 * self.S
        cf, sf = math.cos(phi), math.sin(phi)
        ct, st = math.cos(th), math.sin(th)
        mg = self.m * self.g
        Fx = qS * CX + dT - mg * st                        # (2.7): X + T - m g sin(theta), T inside X
        Fy = qS * CY + mg * ct * sf                        # (2.8)
        Fz = qS * CZ + mg * ct * cf                        # (2.9)
        L = qS * self.b * Cl
        M = qS * self.cbar * Cm
        N = qS * self.b * Cn
        # (2.7)-(2.9) translational, (2.10)-(2.12) rotational (the thesis sets Ixz to zero)
        udot = Fx / self.m - (q * w - r * v)
        vdot = Fy / self.m - (r * u - p * w)
        wdot = Fz / self.m - (p * v - q * u)
        h1, h2, h3 = self.Ix * p - self.Ixz * r, self.Iy * q, -self.Ixz * p + self.Iz * r
        L2, M2, N2 = L - (q * h3 - r * h2), M - (r * h1 - p * h3), N - (p * h2 - q * h1)
        pdot = self.Gx * L2 + self.Gxz * N2
        qdot = M2 / self.Iy
        rdot = self.Gxz * L2 + self.Gz * N2
        # position and Euler angles (zyx)
        cp, sp = math.cos(psi), math.sin(psi)
        ndot = ct * cp * u + (sf * st * cp - cf * sp) * v + (cf * st * cp + sf * sp) * w
        edot = ct * sp * u + (sf * st * sp + cf * cp) * v + (cf * st * sp - sf * cp) * w
        ddot = -st * u + sf * ct * v + cf * ct * w
        tt = st / ct
        phidot = p + sf * tt * q + cf * tt * r
        thdot = cf * q - sf * r
        psidot = (sf * q + cf * r) / ct
        if out is not None:
            n = Om / TWO_PI
            out.update(Va=Va, alpha=math.atan2(wa, ua), beta=math.asin(clamp(va / Va, -1.0, 1.0)) if Va > 1e-9 else 0.0,
                       rho=rho, uh=uh, vh=vh, wh=wh, CX=CX, CY=CY, CZ=CZ, Cl=Cl, Cm=Cm, Cn=Cn,
                       T=T, T_ref=T_ref, dT=dT, J=Va / (n * self.D) if n > 1e-9 else float("inf"),
                       throttle=thr, omega=Om, de=de, da=da, dr=dr,
                       X=qS * CX, Y=qS * CY, Z=qS * CZ, Fx=Fx, Fy=Fy, Fz=Fz, L=L, M=M, N=N,
                       g_wh=whg != wh, g_vh=vhg != vh, g_j=T == 0.0 and Om > 0.0)
        return [ndot, edot, ddot, udot, vdot, wdot, phidot, thdot, psidot, pdot, qdot, rdot, Om_dot]

    def wind_body(self, xr, wn):
        """The NED wind in body axes: R_b^n^T w."""
        if wn[0] == 0.0 and wn[1] == 0.0 and wn[2] == 0.0:
            return (0.0, 0.0, 0.0)
        return tuple(rot_nb(xr[6], xr[7], xr[8]).T @ np.asarray(wn, float))

    def rates(self, x, c, t=None):
        """d/dt of the full state under the commands c = (elevator, aileron, rudder, throttle)
        held over the step, in the wind held over the step: the rigid body and the propeller,
        then the servos (second order, ASSUMED)."""
        dx = self.rigid_rates(x[:N_RIGID], x[13], x[15], x[17], c[3], self.wind_body(x, self._w))
        wn2, z2 = self.wn_s * self.wn_s, 2.0 * self.zeta_s * self.wn_s
        dx += [x[14], wn2 * (c[0] - x[13]) - z2 * x[14],
               x[16], wn2 * (c[1] - x[15]) - z2 * x[16],
               x[18], wn2 * (c[2] - x[17]) - z2 * x[18]]
        return dx

    def _clipped(self):
        c = self.cmd
        return (clamp(c[0], -self.de_lim, self.de_lim), clamp(c[1], -self.da_lim, self.da_lim),
                clamp(c[2], -self.dr_lim, self.dr_lim), clamp(c[3], 0.0, 1.0))

    def evaluate(self):
        """Air data, coefficients, forces and the rest at the current state (fills self.out)."""
        x = self.x
        out = {}
        self.rigid_rates(x[:N_RIGID], x[13], x[15], x[17], self._clipped()[3], self.wind_body(x, self._w), out)
        out["wind"] = tuple(self._w)
        self.out = out
        self._out_at = self.steps
        return out

    # ---- stepping
    def step(self):
        """One fixed step: the commands clipped to the throws, the guards' tally, RK4 over dt
        in the wind held, then the wind for the next step."""
        dt = self.dt
        c = self._clipped()
        out = self.out if self._out_at == self.steps else self.evaluate()
        for k in ("wh", "vh", "j"):
            if out["g_" + k]:
                self.guards[k] += 1
        x, t = self.x, self.t
        k1 = self.rates(x, c)
        x2 = [a + 0.5 * dt * b for a, b in zip(x, k1)]
        k2 = self.rates(x2, c)
        x3 = [a + 0.5 * dt * b for a, b in zip(x, k2)]
        k3 = self.rates(x3, c)
        x4 = [a + dt * b for a, b in zip(x, k3)]
        k4 = self.rates(x4, c)
        self.x = [a + dt / 6.0 * (b1 + 2.0 * b2 + 2.0 * b3 + b4) for a, b1, b2, b3, b4 in zip(x, k1, k2, k3, k4)]
        self.prop_angle = (self.prop_angle + self.x[12] * dt) % TWO_PI
        self.t = t + dt
        self.steps += 1
        self._w = self._sample_wind()

    def run(self, seconds, before_step=None):
        """Step for `seconds`; before_step(self) runs before every step (an autopilot's update)."""
        for _ in range(int(round(seconds / self.dt))):
            if before_step is not None:
                before_step(self)
            self.step()

    # ---- setting a state
    def set_state(self, n=0.0, e=0.0, d=0.0, u=12.0, v=0.0, w=0.0, phi=0.0, theta=0.0, psi=0.0,
                  p=0.0, q=0.0, r=0.0, omega=None, de=0.0, da=0.0, dr=0.0, throttle=None):
        """The state, with the servos at rest on the given deflections, the throttle command
        given (default: throttle_ref, the thesis's model) and the propeller at that throttle's
        speed unless omega says otherwise."""
        throttle = self.throttle_ref if throttle is None else throttle
        omega = throttle * self.om_max if omega is None else omega
        self.x = [n, e, d, u, v, w, phi, theta, psi, p, q, r, omega, de, 0.0, da, 0.0, dr, 0.0]
        self.cmd = [de, da, dr, throttle]
        self._w = self._wind_at(self.t, n, e, d)
        self._out_at = -1
        self.evaluate()

    def _trim_solve(self, va, gamma, h, throttle, iters=60):
        """Straight flight without sideslip, in still air: the six rigid-body equations and the
        path angle, solved for alpha, phi, theta, the three deflections and one more: the
        throttle where the airspeed is given, the airspeed where the throttle is."""
        d = -h
        free_speed = va is None
        z = np.array([0.05, 0.0, 0.05 + gamma, -0.02, 0.0, 0.0, 14.0 if free_speed else 0.7])
        scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0 if free_speed else 1.0])

        def unpack(z):
            al, ph, th, de, da, dr, s = (float(a) for a in z)
            V, thr = (s, throttle) if free_speed else (va, s)
            return al, ph, th, de, da, dr, V, thr

        def resid(z, out=None):
            al, ph, th, de, da, dr, V, thr = unpack(z)
            xr = [0.0, 0.0, d, V * math.cos(al), 0.0, V * math.sin(al), ph, th, 0.0, 0.0, 0.0, 0.0, thr * self.om_max]
            f = self.rigid_rates(xr, de, da, dr, thr, (0.0, 0.0, 0.0), out)
            return np.array(f[3:6] + f[9:12] + [-f[2] - V * math.sin(gamma)])
        for _ in range(iters):
            r0 = resid(z)
            Jm = np.zeros((7, 7))
            for k in range(7):
                hk = 1e-6 * scale[k]
                zp, zm = z.copy(), z.copy()
                zp[k] += hk
                zm[k] -= hk
                Jm[:, k] = (resid(zp) - resid(zm)) / (2.0 * hk)
            try:
                step = np.linalg.solve(Jm, -r0)
            except np.linalg.LinAlgError:                  # no trim there: the caller reads the residual
                break
            z = z + step
            if np.max(np.abs(step / scale)) < 1e-12:
                break
        out = {}
        res = resid(z, out)
        al, ph, th, de, da, dr, V, thr = unpack(z)
        return {"va": V, "alpha": al, "beta": 0.0, "phi": ph, "theta": th, "de": de, "da": da, "dr": dr,
                "throttle": thr, "omega": thr * self.om_max, "T": out["T"], "dT": out["dT"], "X": out["X"],
                "Z": out["Z"], "wh": out["wh"], "guarded": out["g_wh"] or out["g_vh"],
                "residual": float(np.max(np.abs(res)))}

    def _place(self, sol, n, e, h, psi, course):
        al, V, d = sol["alpha"], sol["va"], -h
        u, w = V * math.cos(al), V * math.sin(al)

        def place(psi):
            wb = self.wind_body([0.0] * 6 + [sol["phi"], sol["theta"], psi], self._wind_at(self.t, n, e, d))
            self.set_state(n, e, d, u + wb[0], wb[1], w + wb[2], sol["phi"], sol["theta"], psi,
                           de=sol["de"], da=sol["da"], dr=sol["dr"], throttle=sol["throttle"])
        if course is not None:
            psi = course
            for _ in range(8):
                place(psi)
                psi += wrap(course - self.course())
        place(psi)

    def trim(self, va, n=0.0, e=0.0, h=0.0, psi=0.0, gamma=0.0, apply=True, course=None):
        """Straight flight at airspeed va and air-path angle gamma (rad), OUR throttle balancing
        it: alpha, bank, pitch, the three deflections and the throttle, in still air, with no
        sideslip. (CYo is not zero, so the model flies straight a few degrees banked.) With
        apply, the aircraft is put there (in the wind it has, at the same air-relative
        velocity), heading psi, or, given a course, at the heading that makes that ground track
        in the wind. Returns the solution as a dict; check its "residual", its "throttle"
        against 0..1 and its deflections against the throws where the speed is far from 12 m/s."""
        sol = self._trim_solve(va, gamma, h, None)
        if apply:
            self._place(sol, n, e, h, psi, course)
        return sol

    def level_speed(self, n=0.0, e=0.0, h=0.0, psi=0.0, apply=False, course=None, throttle=None):
        """The thesis's model untouched: the airspeed at which it flies level with the throttle
        at throttle_ref (our thrust difference zero), and its trim there. Another throttle
        gives that throttle's level speed. Returns the solution as a dict ("va" is the speed)."""
        sol = self._trim_solve(None, 0.0, h, self.throttle_ref if throttle is None else throttle)
        if apply:
            self._place(sol, n, e, h, psi, course)
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
        NED origin, J."""
        x = self.x
        u, v, w, p, q, r = x[3], x[4], x[5], x[9], x[10], x[11]
        rot = 0.5 * (self.Ix * p * p + self.Iy * q * q + self.Iz * r * r) - self.Ixz * p * r
        return 0.5 * self.m * (u * u + v * v + w * w) + rot - self.m * self.g * x[2]

    def jacobian(self, xr, ui, eps=None):
        """Numerical Jacobians of rigid_rates at the 13 states xr and the inputs ui = (de, da,
        dr, throttle, u_w, v_w, w_w) (wind in body axes): central differences. Returns (A, B),
        13 x 13 and 13 x 7."""
        xr, ui = np.asarray(xr, float), np.asarray(ui, float)

        def f(xv, uv):
            return np.array(self.rigid_rates(list(xv), uv[0], uv[1], uv[2], uv[3], tuple(uv[4:7])))
        A = np.zeros((13, 13))
        B = np.zeros((13, 7))
        for k in range(13):
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


AIRCRAFT = Bixler3
_INDEX = {k: i for i, k in enumerate(STATE)}


# --------------------------------------------------------------------------- #
#  The autopilot: OURS
# --------------------------------------------------------------------------- #
class Autopilot:
    """Successive loop closure, as in Beard and McLain, Small Unmanned Aircraft (2012), ch. 6,
    with the thesis's signs of the deflections (so the aileron and elevator gains are negative):

        roll     d_a = d_a* + kp (phi_c - phi) + ki int(phi_c - phi) - kd p
        course   phi_c = phi* + kp (chi_c - chi) + ki int(chi_c - chi)   (or heading psi in place of chi)
        pitch    d_e = d_e* + kp (theta_c - theta) + ki int(theta_c - theta) - kd q
        altitude theta_c = theta* + kp (h_c - h) + ki int(h_c - h), saturated outside a zone
        airspeed d_t = d_t* + kp (V_c - V_a) + ki int(V_c - V_a)
        sideslip d_r = d_r* + kp beta + ki int(beta)

    The starred values are the still-air trim at the commanded airspeed. The roll and pitch
    gains come from the thesis's coefficients at that airspeed (the second-order models
    phi'' = -a_phi1 phi' + a_phi2 d_a and theta'' = -a_th1 theta' - a_th2 theta + a_th3 d_e,
    with the thesis's rates over 2 Vo and its w over Vo) and the spec's natural frequencies and
    dampings. The course loop's gains come from the roll loop's and a bandwidth ratio, the
    altitude loop's from its own natural frequency (h' = V_a theta), the airspeed loop's from
    the model's own d(u')/du and d(u')/d(throttle) at the trim, the motor taken as quasi-steady.

    The rudder holds the sideslip at zero rather than damping yaw: the thesis's dutch roll is
    damped well enough on its own (0.32), and its CYo would otherwise leave a standing sideslip
    of several degrees in straight flight. The side force of the rudder (CYdr) is too small, and
    of too uncertain a sign, for the textbook's sideslip hold, so the loop is laid on the yawing
    moment: kp adds `stiffness` times the airframe's own weathercock stiffness (kp = stiffness
    Cnv V_a / (Vo Cndr)), and the integral takes the standing part out.

    It is fed the EXACT state (position, attitude, rates, airspeed, sideslip): no estimator, no
    sensor noise. The spec's autopilot block holds the frequencies, dampings, integrals and
    limits, nothing else.

    hold: "course" (the ground track, which turns into the wind and compensates for it),
    "heading" (the nose, which drifts with the wind) or "roll" (``command(roll=...)``: the roll
    angle is handed in, for an outer loop of the caller's own)."""

    def __init__(self, aircraft, airspeed=None, hold="course"):
        self.ac = aircraft
        cfg = aircraft.spec["autopilot"]
        self.cfg = cfg
        self.hold = hold
        self.va_c = airspeed or cfg["airspeed"]
        self.chi_c = aircraft.course() if hold == "course" else aircraft.x[8]
        self.h_c = aircraft.h
        self.design(self.va_c)
        self.i_phi = self.i_chi = self.i_h = self.i_v = self.i_th = self.i_beta = 0.0
        self.phi_c = self.theta_c = self.phi_cmd = 0.0

    def design(self, va):
        """The gains at airspeed va (and the air density at the aircraft's height)."""
        ac, A, cfg = self.ac, self.ac.a, self.cfg
        probe = Bixler3(ac.spec, dt=ac.dt, h0=ac.h0, rho=ac.rho_fixed)
        tr = probe.trim(va, h=ac.h, apply=False)
        self.trim = tr
        Vo = ac.Vo
        rho = ac.rho_at(ac.x[2])
        qS = 0.5 * rho * va * va * ac.S
        # roll: p' = L / Ix, the damping through p b / (2 Vo). a_phi2 < 0: positive aileron rolls left
        a1 = -qS * ac.b * (ac.Gx * A["Clp"] + ac.Gxz * A["Cnp"]) * ac.b / (2.0 * Vo)
        a2 = qS * ac.b * (ac.Gx * A["Clda"] + ac.Gxz * A["Cnda"])
        c = cfg["roll"]
        w, z = c["wn"], c["zeta"]
        self.a_phi = (a1, a2)
        self.kp_phi = w * w / a2
        self.kd_phi = max(0.0, 2.0 * z * w - a1) / a2      # rate feedback only for what the airframe's own damping lacks
        self.ki_phi = c["ki"] * self.kp_phi
        self.i_phi_max = math.radians(c["integral_deg"]) / abs(self.ki_phi)   # the integral's share of the aileron
        self.phi_max = math.radians(c["limit_deg"])
        # course (chi' = g / V_g tan phi)
        c = cfg["course"]
        self.w_chi = w / c["bandwidth_ratio"]
        self.z_chi = c["zeta"]
        # pitch: theta'' = -a_th1 theta' - a_th2 theta + a_th3 d_e; the stiffness through
        # Cmw w / Vo with w = V_a alpha, the damping through q cbar / (2 Vo). a_th3 < 0.
        a1 = -qS * ac.cbar * A["Cmq"] * ac.cbar / (2.0 * Vo) / ac.Iy
        a2 = -qS * ac.cbar * A["Cmw"] * (va / Vo) / ac.Iy
        a3 = qS * ac.cbar * A["Cmde"] / ac.Iy
        self.a_th = (a1, a2, a3)
        c = cfg["pitch"]
        w, z = c["wn"], c["zeta"]
        self.kp_th = (w * w - a2) / a3
        self.kd_th = (2.0 * z * w - a1) / a3
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
        xr = [0.0, 0.0, ac.x[2], va * math.cos(tr["alpha"]), 0.0, va * math.sin(tr["alpha"]), tr["phi"], tr["theta"],
              0.0, 0.0, 0.0, 0.0, tr["omega"]]
        Aj, Bj = probe.jacobian(xr, [tr["de"], tr["da"], tr["dr"], tr["throttle"], 0.0, 0.0, 0.0])
        Ar, Br = quasi_steady(Aj, Bj)
        av1, av2 = -Ar[3, 3], Br[3, 3]
        c = cfg["speed"]
        w, z = c["wn"], c["zeta"]
        self.kp_v = (2.0 * z * w - av1) / av2
        self.ki_v = w * w / av2
        self.t_lim = tuple(c["throttle"])
        self.a_v = (av1, av2)
        # sideslip on rudder, through the yawing moment: beta'' = -n_r beta' - n_b beta - n_dr d_r,
        # n_b = qS b Cnv V_a / (Vo Iz) the weathercock stiffness, n_dr = qS b Cndr / Iz < 0
        c = cfg["sideslip"]
        n_b = qS * ac.b * A["Cnv"] * (va / Vo) / ac.Iz
        n_dr = qS * ac.b * A["Cndr"] / ac.Iz
        self.a_beta = (n_b, n_dr)
        self.kp_beta = c["stiffness"] * n_b / n_dr
        self.ki_beta = c["wi"] * (1.0 + c["stiffness"]) * n_b / n_dr
        self.i_beta_max = math.radians(c["integral_deg"]) / abs(self.ki_beta)

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
        phi, theta, psi, p, q = x[6], x[7], x[8], x[9], x[10]
        # course (or heading) -> roll
        if self.hold == "roll":
            phi_c = self.phi_cmd                           # the caller's own outer loop gives the roll angle
        else:
            if self.hold == "course":
                vg = ac.ground_velocity()
                chi = math.atan2(vg[1], vg[0])
                spd = max(math.hypot(vg[0], vg[1]), 4.0)
            else:
                chi, spd = psi, max(out["Va"], 4.0)
            e_chi = wrap(self.chi_c - chi)
            kp = 2.0 * self.z_chi * self.w_chi * spd / ac.g
            ki = self.w_chi * self.w_chi * spd / ac.g
            phi_c = tr["phi"] + kp * e_chi + ki * self.i_chi
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
        # sideslip -> rudder
        beta = out["beta"]
        dr = tr["dr"] + self.kp_beta * beta + self.ki_beta * self.i_beta
        self.i_beta = clamp(self.i_beta + beta * dt, -self.i_beta_max, self.i_beta_max)
        # the throws are clipped in Bixler3.step
        ac.cmd = [de, da, dr, thr]

    def status(self):
        return (f"chi_c {math.degrees(self.chi_c):6.1f}  phi_c {math.degrees(self.phi_c):+5.1f}  "
                f"theta_c {math.degrees(self.theta_c):+5.1f}  h_c {self.h_c:5.1f}  V_c {self.va_c:4.1f}")


# --------------------------------------------------------------------------- #
#  Guidance: OURS
# --------------------------------------------------------------------------- #
class LOS(_LOS):
    """x8_rig.LOS (line of sight along a polyline of waypoints [(n, e), ...]: the course to a
    look-ahead point on the current leg, the cross-track error ``e``, the aim point ``aim``)
    with THIS aircraft's spec as the default: its `guidance` block is sized for 12 m/s."""

    def __init__(self, waypoints, lookahead=None, accept=None, loop=False, spec=None):
        super().__init__(waypoints, lookahead=lookahead, accept=accept, loop=loop, spec=spec or load_spec())


# --------------------------------------------------------------------------- #
#  The 3D model
# --------------------------------------------------------------------------- #
class Visual:
    """bixler3.glb's nodes posed from a Bixler3's state.

    root    the loaded model's scene node (tp.GLTFLoader().load(model_path()).scene)

    pose(aircraft, h0) places the root from the model's position and attitude (world = (E,
    h0 - D, -N)) and turns each surface about its hinge (local +Z) by the thesis's deflection.
    The ailerons' and the elevator's local +Z runs along the hinge toward starboard, so a
    positive turn puts the trailing edge down: the elevator takes d_e, the right aileron +d_a
    and the left one -d_a (positive aileron rolls left). The rudder's local +Z points down its
    hinge, so a positive turn moves the trailing edge to port: it takes d_r. The propeller (a
    pusher on the pylon) turns about local +X by the integrated speed of OUR propeller model."""

    NODE_NAMES = {"aileron_left": "aileron_left", "aileron_right": "aileron_right", "elevator": "elevator",
                  "rudder": "rudder", "propeller": "propeller"}

    def __init__(self, root):
        self.root = root
        self.nodes = {k: root.get_object_by_name(n) for k, n in self.NODE_NAMES.items()}
        missing = [k for k, v in self.nodes.items() if v is None]
        if missing:
            raise KeyError(f"bixler3.glb has no node(s) {missing}: rebuild it with uav/build_bixler3_blender.py")

    def pose(self, aircraft, h0=0.0):
        x = aircraft.x
        self.root.position.set(*ned_to_world(x[0], x[1], x[2], h0))
        self.root.quaternion.set(*quat_of(aircraft.world_rotation()))
        self.nodes["elevator"].rotation.z = x[13]
        self.nodes["aileron_right"].rotation.z = x[15]
        self.nodes["aileron_left"].rotation.z = -x[15]
        self.nodes["rudder"].rotation.z = x[17]
        self.nodes["propeller"].rotation.x = aircraft.prop_angle
