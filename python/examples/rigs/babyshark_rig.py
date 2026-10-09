"""
babyshark_rig -- the Foxtech Babyshark 260 VTOL, a 2.5 m quadplane: its published flight model,
what a flight from a hover to the cruise and back needs on top of it, an autopilot for both
halves and the transition between them, a mission, and its 3D model posed from the state.

Five things of its own, and nothing scene-specific:

  * ``Babyshark`` -- the model of B. P. Graesdal, "Full Nonlinear System Identification for a
                     Vertical-Takeoff-and-Landing Unmanned Aerial Vehicle", Master's thesis,
                     NTNU, 2021, https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/2981320: its
                     equations (6.6)-(6.16) as written, with the numbers of its Tables 6.2-6.5
                     (uav/babyshark_spec.json). The rigid body (6.6) with the inertia's Gamma
                     constants (Table 6.2); lift, drag, side force and the three moments (6.7)-
                     (6.10) with the rates made non-dimensional by the trim airspeed (6.11) and
                     the surfaces measured from their trim (6.13); the lift rotors' thrusts,
                     torques and moments (6.14)-(6.16); the pusher's thrust (6.15a); the servos
                     (3.41). Fixed-step fourth-order Runge-Kutta.
  * ``Autopilot`` -- OURS: the multirotor cascade of m350_rig for the hover, the successive
                     loop closure of x8_rig for the wing, and a transition between them laid
                     out as PX4's standard-VTOL one (the thesis's aircraft flew PX4).
  * ``Mission``   -- OURS: take-off, hover legs, a cruise along a route, a landing, with the
                     transitions where they belong.
  * ``Pilot``     -- OURS: four sticks (or a keyboard) on the autopilot's setpoints, the same
                     four hovering and on the wing (uav/babyshark_fly.py flies it in a window).
  * ``Visual``    -- the babyshark.glb (uav/build_babyshark_blender.py) posed from a Babyshark.

Used as they are: ``Wind`` of m350_rig (a mean wind and Dryden turbulence) and ``LOS`` of x8_rig
(line of sight along a polyline), both re-exported here.

WHAT IS WHOSE
-------------
The thesis's, untouched wherever it identified them: every coefficient, the mass and inertia,
the servo, the propellers' static thrust and torque coefficients, the rotor positions. The
thesis flew the wing around 21 m/s with the lift rotors stopped, in air taken as still; its
data reach alpha -13..+16 deg and its propellers were measured with no incoming air. Inside
that, ``Babyshark.rates`` IS the thesis's model (uav/babyshark_flight.py --checks 2 compares its
linearisation with the thesis's printed matrices, entry by entry).

OURS or ASSUMED, each in the spec with its reason, none of it fitted to a flight:

  * wind: the air-relative velocity stands where the thesis has the body velocity in the
    airspeed, alpha and beta (6.12);
  * outside the identified alpha and beta (``envelope``): the forces and moments go over to a
    bluff body (the airframe's projected areas, each at its centre of pressure), so the wing
    stalls and the hover and the transitions are finite. Inside, nothing changes;
  * the pusher's thrust falls with the advance ratio, T = rho D^4 cT n^2 (1 - J / J0): the
    thesis's law at J = 0. The thesis's static law at every speed is ``thesis_propeller=True``;
  * a lift rotor's thrust loses kz n v_ax to the air through its disc and has a small in-plane
    force (m350_rig's rotor): in a still hover it is the thesis's law;
  * motors: first-order lags, speed limits, an idle; two battery packs and their power;
  * the landing gear on a ground you hand in (m350_rig's contact model);
  * the air density follows the height (ISA; 1.225 at the sea, the thesis's constant).

NOT in the model: the propellers' gyroscopic moment and the pusher's torque (the thesis
neglects both), translational lift and rotor-wing interference in the transition, ground
effect, any sensor. Nothing here is validated against a flight log beyond what the thesis did.

Nothing here imports threepp: ``Visual`` is handed the loaded model's root node.

FRAMES
------
The thesis's, which are x8_rig's and m350_rig's: NED position (N, E, D), body axes x forward,
y right, z down, zyx Euler angles (phi, theta, psi). One mapping to threepp's world (x east,
y up, z south), at the boundary::

    world = (E, h0 - D, -N)            h0: the height of the NED origin in the world

The .glb's frame is X forward, Y up, Z right with its origin at the centre of gravity: a body
vector (x, y, z) is (x, -z, y) in it. Heights (``h``, the ground's, a waypoint's) are metres
above the NED origin. Wind is the velocity of the air over the ground, NED, m/s. Propeller
speeds are rev/s, as in the thesis (whose inputs are the squared rev/s).

TYPICAL USE
-----------
::

    from babyshark_rig import Babyshark, Wind, Autopilot, Mission, Visual, model_path

    a = Babyshark(wind=Wind(6.0, from_deg=250.0, spec=load_spec(), seed=3))
    a.place_on_ground(0.0, 0.0, yaw=math.radians(250.0))     # nose into the wind
    ap = Autopilot(a)
    mission = Mission([dict(kind="takeoff", height=60.0),
                       dict(kind="cruise", route=[(400.0, -300.0), (400.0, 300.0)], altitude=60.0),
                       dict(kind="land", pos=(0.0, 0.0))])
    while not mission.done:
        mission.update(a, ap)                         # setpoints for the autopilot
        ap.update()                                   # surface, pusher and rotor commands
        a.step()                                      # one fixed step, a.dt
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
    sys.path.insert(0, HERE)                                 # examples/rigs/ (m350_rig, x8_rig)

from m350_rig import (C_BM, C_WN, KEEP, Wind, clamp, flat_ground, ground_normal, isa_density,  # noqa: E402,F401
                      ned_to_world, quat_of, rot_nb, world_to_ned, wrap)
from x8_rig import LOS, cross_track  # noqa: E402,F401

UAV_DIR = os.path.join(os.path.dirname(HERE), "uav")
SPEC_PATH = os.path.join(UAV_DIR, "babyshark_spec.json")
TWO_PI = 2.0 * math.pi


def load_spec(path=None):
    with open(path or SPEC_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def model_path():
    """The generated babyshark.glb, or None where it has not been built."""
    p = os.path.join(UAV_DIR, "babyshark.glb")
    return p if os.path.isfile(p) else None


def body_of(v):
    """A layout vector (X forward, Y up, Z right) in body axes (x forward, y right, z down)."""
    return (v[0], v[2], -v[1])


def smoothstep(t):
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    return t * t * (3.0 - 2.0 * t)


def euler_of(R):
    """(phi, theta, psi) of a body-to-NED rotation matrix, zyx."""
    return (math.atan2(R[2][1], R[2][2]), -math.asin(clamp(R[2][0], -1.0, 1.0)), math.atan2(R[1][0], R[0][0]))


# --------------------------------------------------------------------------- #
#  The aircraft
# --------------------------------------------------------------------------- #
# The integrated state, in this order. The first 15 are the thesis's (its code's order): NED
# position, body velocity, body rates, Euler angles, the three control surfaces. Then OURS: the
# pusher's speed and the four lift rotors' (the thesis's order: 1 front right, 2 rear left,
# 3 front left, 4 rear right), rev/s.
STATE = ("n", "e", "d", "u", "v", "w", "p", "q", "r", "phi", "theta", "psi", "delta_a", "delta_e", "delta_r",
         "eta_fw", "eta_1", "eta_2", "eta_3", "eta_4")
N_RIGID = 12
ROTORS = ("1", "2", "3", "4")


class Babyshark:
    """The Babyshark 260 VTOL as the thesis models it, with what a hover and a transition need.

    spec      babyshark_spec.json as a dict (default: load it); every number is read from it
    dt        the fixed step, s (default 1/300: 60, 50 and 30 frames a second are whole steps)
    wind      a Wind; or a NED 3-tuple; or a callable (t, n, e, d) -> (wn, we, wd); or None.
              Sampled once per step and held across the Runge-Kutta stages.
    ground    ground(n, e) -> height above the NED origin (default flat at 0)
    h0        the NED origin's height above mean sea level (the air density follows it)
    rho       a fixed air density in place of the ISA's (0 takes the air away)
    thesis_propeller   True: the pusher's thrust is the thesis's static law at every airspeed

    With V, alpha, beta from the AIR-RELATIVE body velocity (u_a, v_a, w_a) (6.12; OURS: the
    wind), qbar = rho V^2 / 2, the rates p^ = b p / (2 V*), q^ = cbar q / (2 V*), r^ = b r /
    (2 V*) (6.11, V* the trim airspeed) and the surfaces from their trim, D_ = delta - delta*
    (6.13), the thesis's coefficients are

        cD = cD0 + cDa alpha + cDa2 alpha^2 + cDq q^ + (cDde + cDdea alpha) D_e          (6.8d)
        cL = cL0 + cLa alpha + cLa2 alpha^2 + cLde D_e                                   (6.8e)
        cY = cY0 + cYb beta + cYp p^ + cYda D_a + cYdr D_r                               (6.8b)
        cl = cl0 + clb beta + clp p^ + clr r^ + clda D_a                                 (6.10a)
        cm = cm0 + cma alpha + cmq q^ + cmde D_e + cmdr2 delta_r^2                       (6.10b)
        cn = cn0 + cnb beta + cnp p^ + cnr r^ + cndr D_r                                 (6.10c)

    the forces X = qbar S (-cD cos alpha + cL sin alpha), Y = qbar S cY, Z = qbar S (-cD sin
    alpha - cL cos alpha) (6.7), (6.8a, c) and the moments l = qbar S b cl, m = qbar S cbar cm,
    n = qbar S b cn (6.9). Lift rotor i at (r_ix, r_iy) from the centre of gravity, turning at
    n_i rev/s: T_i = rho D^4 cT n_i^2 and Q_i = rho D^5 cQ n_i^2 (6.15b, c), moments -r_iy T_i
    about x, r_ix T_i about y, +-Q_i about z (6.14). The pusher: T = rho D^4 cT n^2 (6.15a).
    The rigid body is (6.6a-i) with the Gamma constants of Table 6.2.

    OURS, each switched off where the thesis identified its model (the module's docstring and
    the spec say what and why): outside ``alpha_lim`` and ``beta_lim`` the coefficients are
    held at the edge and the forces and moments blend (``sigma``, 0 inside) to a bluff body's;
    the pusher's thrust is multiplied by (1 - J / J0); a lift rotor's thrust loses kz n v_ax and
    it has the in-plane force -kh n v_perp; the landing gear's four feet on the ground.

    ``x`` is the state (STATE order). ``cmd`` holds what the next step() uses: the three
    surface setpoints (rad), the pusher's speed command and the four lift rotors' (rev/s).
    ``out`` is what the last step evaluated at its start (air data, coefficients, thrusts,
    powers, the ground's force)."""

    def __init__(self, spec=None, dt=1.0 / 300.0, wind=None, ground=None, h0=0.0, rho=None,
                 thesis_propeller=False):
        self.spec = spec = load_spec() if spec is None else spec
        P, A, TR, AC, PU, ENV, EN, GR = (spec["physical"], spec["aero"], spec["trim"], spec["actuators"],
                                         spec["propulsion"], spec["envelope"], spec["environment"], spec["ground"])
        self.dt = dt
        self.wind = wind
        self.ground = flat_ground if ground is None or ground is flat_ground else (lambda n, e: float(ground(n, e)))
        self.h0 = h0
        self.g = EN["g"]
        self.rho0 = EN["rho_sea_level"]
        self.rho_fixed = rho if rho is not None else (None if EN["rho"] == "isa" else float(EN["rho"]))
        # ---- Table 6.2: the airframe
        self.b, self.cbar, self.S, self.m = P["b"], P["cbar"], P["S"], P["mass"]
        self.Jxx, self.Jyy, self.Jzz, self.Jxz = P["Jxx"], P["Jyy"], P["Jzz"], P["Jxz"]
        G = self.Jxx * self.Jzz - self.Jxz * self.Jxz
        self.gamma = (self.Jxz * (self.Jxx - self.Jyy + self.Jzz) / G,
                      (self.Jzz * (self.Jzz - self.Jyy) + self.Jxz * self.Jxz) / G,
                      self.Jzz / G, self.Jxz / G,
                      (self.Jzz - self.Jxx) / self.Jyy, self.Jxz / self.Jyy,
                      (self.Jxx * (self.Jxx - self.Jyy) + self.Jxz * self.Jxz) / G, self.Jxx / G)
        self.J = np.array([[self.Jxx, 0.0, -self.Jxz], [0.0, self.Jyy, 0.0], [-self.Jxz, 0.0, self.Jzz]])
        # ---- Table 6.4: the coefficients; Table 6.5: the trim they are written about
        self.a = A
        self._co = tuple(A[k] for k in ("cD0", "cDa", "cDa2", "cDq", "cDde", "cDdea", "cL0", "cLa", "cLa2", "cLde",
                                        "cm0", "cma", "cmq", "cmde", "cmdr2", "cY0", "cYb", "cYp", "cYda", "cYdr",
                                        "cl0", "clb", "clp", "clr", "clda", "cn0", "cnb", "cnp", "cnr", "cndr"))
        self.V0 = TR["V"]
        self.da0, self.de0, self.dr0 = (math.radians(TR["delta_a_deg"]), math.radians(TR["delta_e_deg"]),
                                        math.radians(TR["delta_r_deg"]))
        # ---- Table 6.3: the actuators
        self.T_servo = AC["T_servo"]
        self.rate_lim = math.radians(AC["rate_limit_dps"])
        self.d_max = (math.radians(AC["delta_a_max_deg"]), math.radians(AC["delta_e_max_deg"]),
                      math.radians(AC["delta_r_max_deg"]))
        self.D_fw, self.D_mr = AC["D_fw"], AC["D_mr"]
        self.kTf_rho = self.D_fw ** 4 * AC["cT_fw"]                    # T / (rho n^2), the pusher
        self.kTm_rho = self.D_mr ** 4 * AC["cT_mr"]                    # T / (rho n^2), a lift rotor
        self.kQm_rho = self.D_mr ** 5 * AC["cQ_mr"]                    # Q / (rho n^2)
        self._rot = [(AC["rotors"][k]["x"], AC["rotors"][k]["y"], float(AC["rotors"][k]["spin"])) for k in ROTORS]
        self.spin = tuple(r[2] for r in self._rot)
        self.rotor_names = tuple(AC["rotors"][k]["name"] for k in ROTORS)
        # ---- OURS: the propulsion around the thesis's static laws
        self.eta_fw_max, self.eta_mr_max, self.eta_idle = PU["eta_fw_max"], PU["eta_mr_max"], PU["eta_mr_idle"]
        self.tau_fw, self.tau_mr = PU["tau_fw"], PU["tau_mr"]
        self.J0 = math.inf if thesis_propeller else PU["J0_fw"]
        self.kQf_rho = self.D_fw ** 5 * PU["cQ_fw"]
        R = 0.5 * self.D_mr
        Ad = math.pi * R * R
        solidity = PU["blades_mr"] * PU["blade_chord_mr"] / (math.pi * R)
        lam_h = math.sqrt(0.5 * (self.kTm_rho / (4.0 * math.pi * math.pi)) / (Ad * R * R))
        a_s = PU["lift_slope_mr"] * solidity
        self.kz_rho = TWO_PI * (a_s * Ad * R / 4.0) / (1.0 + a_s / (16.0 * lam_h))     # per rev/s
        self.kh_rho = TWO_PI * solidity * PU["cd0_blade_mr"] * Ad * R / 4.0            # per rev/s
        self.eta_drive, self.avionics = PU["drive_efficiency"], PU["avionics_w"]
        self.lift_wh, self.cruise_wh = PU["battery_lift"]["energy_wh"], PU["battery_cruise"]["energy_wh"]
        self.lift_j, self.cruise_j = self.lift_wh * 3600.0, self.cruise_wh * 3600.0
        # ---- OURS: outside the identified range
        self.alpha_lim = tuple(math.radians(v) for v in ENV["alpha_deg"])
        self.beta_lim = tuple(math.radians(v) for v in ENV["beta_deg"])
        self.blend = math.radians(ENV["blend_deg"])
        B = ENV["bluff"]
        self.cd_bluff = B["drag_coefficient"]
        self._area = (B["area_front"], B["area_side"], B["area_top"])
        self._cpf, self._cps, self._cpt = body_of(B["cp_front"]), body_of(B["cp_side"]), body_of(B["cp_top"])
        self.guards = {"alpha": 0, "beta": 0}
        # ---- the ground: the four feet (m350_rig's contact model)
        self._feet = [tuple(float(c) for c in body_of(f)) for f in spec["layout"]["gear"]["feet"]]
        self._gk, self._gc = GR["stiffness_n_m"], GR["damping_ns_m"]
        self._gkt, self._gct, self._mu = GR["tangential_stiffness_n_m"], GR["tangential_damping_ns_m"], GR["friction"]
        self._gcheck = GR["check_height_m"]
        self._anchor = [None, None, None, None]
        self._near = False
        self._gnorm = (0.0, 0.0, -1.0)
        # ---- state
        self.x = [0.0] * len(STATE)
        self.x[12], self.x[13], self.x[14] = self.da0, self.de0, self.dr0
        self.cmd = [self.da0, self.de0, self.dr0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.rotor_angle = [0.0, 0.0, 0.0, 0.0]       # each lift rotor's own angle (its speed integrated), for Visual
        self.pusher_angle = 0.0
        self.t = 0.0
        self.steps = 0
        self._w = (0.0, 0.0, 0.0)
        self.out = {}
        self._air(self.rho_at(0.0))

    # ---- named access
    def __getattr__(self, name):
        if name in _INDEX:
            return self.x[_INDEX[name]]
        raise AttributeError(name)

    @property
    def h(self):
        """Height of the centre of gravity above the NED origin, m."""
        return -self.x[2]

    def rho_at(self, d):
        return self.rho_fixed if self.rho_fixed is not None else isa_density(self.h0 - d, self.rho0)

    def _air(self, rho):
        self.rho = rho
        self._kTf, self._kTm, self._kQm = self.kTf_rho * rho, self.kTm_rho * rho, self.kQm_rho * rho
        self._kz, self._kh, self._kQf = self.kz_rho * rho, self.kh_rho * rho, self.kQf_rho * rho

    def wind_ned(self):
        """The wind the last step used at the aircraft, NED m/s."""
        return self._w

    def _sample_wind(self):
        w = self.wind
        if w is None:
            return (0.0, 0.0, 0.0)
        x = self.x
        if isinstance(w, Wind):
            v = self.ground_velocity()
            return w.advance(self.dt, x[0], x[1], -x[2], (v[0], v[1], v[2]))
        if callable(w):
            w = w(self.t, x[0], x[1], x[2])
        return (float(w[0]), float(w[1]), float(w[2]))

    # ---- the equations
    def rates(self, x, c, rec=None):
        """d/dt of the state x under the commands c (three surface setpoints, the pusher's and
        the four lift rotors' speed commands), in the held wind and air density. Equation
        numbers are the thesis's. With rec (a dict) it also writes what it evaluated."""
        n, e, d, u, v, w, p, q, r, phi, th, psi, da, de, dr, nf, n1, n2, n3, n4 = x
        cf, sf = math.cos(phi), math.sin(phi)
        ct, st = math.cos(th), math.sin(th)
        cp, sp = math.cos(psi), math.sin(psi)
        r00, r01, r02 = ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp
        r10, r11, r12 = ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp
        r20, r21, r22 = -st, sf * ct, cf * ct
        # OURS: the air-relative velocity in body axes; (6.12) on it
        wn, we, wd = self._w
        ua = u - (r00 * wn + r10 * we + r20 * wd)
        va = v - (r01 * wn + r11 * we + r21 * wd)
        wa = w - (r02 * wn + r12 * we + r22 * wd)
        V = math.sqrt(ua * ua + va * va + wa * wa)
        rho = self.rho
        if V > 1e-6:
            alpha = math.atan2(wa, ua)                       # tan^-1(w / u), on all four quadrants
            beta = math.asin(clamp(va / V, -1.0, 1.0))
        else:
            alpha = beta = 0.0
        # OURS: how far outside the identified range (0 inside it)
        (alo, ahi), (blo, bhi) = self.alpha_lim, self.beta_lim
        al = alo if alpha < alo else ahi if alpha > ahi else alpha
        be = blo if beta < blo else bhi if beta > bhi else beta
        if al == alpha and be == beta:
            sg = 0.0
        else:
            sg = 1.0 - (1.0 - smoothstep(abs(alpha - al) / self.blend)) * (1.0 - smoothstep(abs(beta - be) / self.blend))
        Fx = Fy = Fz = Mx = My = Mz = 0.0
        cD = cL = cm = cY = cl = cn = 0.0
        if sg < 1.0:
            (cD0, cDa, cDa2, cDq, cDde, cDdea, cL0, cLa, cLa2, cLde, cm0, cma, cmq, cmde, cmdr2,
             cY0, cYb, cYp, cYda, cYdr, cl0, clb, clp, clr, clda, cn0, cnb, cnp, cnr, cndr) = self._co
            k2 = 0.5 / self.V0
            ph, qh, rh = self.b * p * k2, self.cbar * q * k2, self.b * r * k2          # (6.11)
            dda, dde, ddr = da - self.da0, de - self.de0, dr - self.dr0                # (6.13)
            cD = cD0 + cDa * al + cDa2 * al * al + cDq * qh + (cDde + cDdea * al) * dde    # (6.8d)
            cL = cL0 + cLa * al + cLa2 * al * al + cLde * dde                              # (6.8e)
            cY = cY0 + cYb * be + cYp * ph + cYda * dda + cYdr * ddr                       # (6.8b)
            cl = cl0 + clb * be + clp * ph + clr * rh + clda * dda                         # (6.10a)
            cm = cm0 + cma * al + cmq * qh + cmde * dde + cmdr2 * dr * dr                  # (6.10b)
            cn = cn0 + cnb * be + cnp * ph + cnr * rh + cndr * ddr                         # (6.10c)
            qS = 0.5 * rho * V * V * self.S * (1.0 - sg)
            ca, sa = math.cos(alpha), math.sin(alpha)
            Fx = qS * (-cD * ca + cL * sa)                                                 # (6.7a), (6.8a)
            Fy = qS * cY                                                                   # (6.7b)
            Fz = qS * (-cD * sa - cL * ca)                                                 # (6.7c), (6.8c)
            Mx, My, Mz = qS * self.b * cl, qS * self.cbar * cm, qS * self.b * cn           # (6.9)
        if sg > 0.0:
            # OURS: the bluff body, each projected area at its centre of pressure
            k = -0.5 * rho * self.cd_bluff * V * sg
            af, as_, at = self._area
            (_, yf, zf), (xs, _, zs), (xt, yt, _) = self._cpf, self._cps, self._cpt
            bx = k * af * (ua + q * zf - r * yf)
            by = k * as_ * (va + r * xs - p * zs)
            bz = k * at * (wa + p * yt - q * xt)
            Fx += bx
            Fy += by
            Fz += bz
            Mx += yt * bz - zs * by
            My += zf * bx - xt * bz
            Mz += xs * by - yf * bx
        # the pusher (6.15a); OURS: the advance ratio's (1 - J / J0), J = u_a / (n D)
        T = self._kTf * (nf * nf - (nf * ua / (self.D_fw * self.J0) if ua > 0.0 else 0.0))
        if T < 0.0:
            T = 0.0
        Fx += T
        # the lift rotors (6.14)-(6.16); OURS: the air through the disc (kz), the in-plane force (kh)
        kTm, kQm, kz, kh = self._kTm, self._kQm, self._kz, self._kh
        thrust = [] if rec is not None else None
        for (rx, ry, s), ni in zip(self._rot, (n1, n2, n3, n4)):
            Ti = kTm * ni * ni + kz * ni * (wa + p * ry - q * rx)
            if Ti < 0.0:
                Ti = 0.0
            hx, hy = -kh * ni * (ua - r * ry), -kh * ni * (va + r * rx)
            Fx += hx
            Fy += hy
            Fz -= Ti
            Mx -= ry * Ti
            My += rx * Ti
            Mz += rx * hy - ry * hx + s * kQm * ni * ni
            if thrust is not None:
                thrust.append(Ti)
        # OURS: the ground under the four feet, a spring and damper along its normal and a
        # bristle with Coulomb's limit across it
        gforce = 0.0
        if self._near:
            gnd, (nn, ne, nd) = self.ground, self._gnorm
            gk, gc, gkt, gct, mu = self._gk, self._gc, self._gkt, self._gct, self._mu
            vn = r00 * u + r01 * v + r02 * w
            ve = r10 * u + r11 * v + r12 * w
            vd = r20 * u + r21 * v + r22 * w
            for i, (cx, cy, cz) in enumerate(self._feet):
                pn = n + r00 * cx + r01 * cy + r02 * cz
                pe = e + r10 * cx + r11 * cy + r12 * cz
                pd = d + r20 * cx + r21 * cy + r22 * cz
                pen = -(gnd(pn, pe) + pd) * nd                           # depth along the normal
                if pen <= 0.0:
                    continue
                ox, oy, oz = q * cz - r * cy, r * cx - p * cz, p * cy - q * cx
                un = vn + r00 * ox + r01 * oy + r02 * oz
                ue = ve + r10 * ox + r11 * oy + r12 * oz
                ud = vd + r20 * ox + r21 * oy + r22 * oz
                vnn = un * nn + ue * ne + ud * nd
                Fn = gk * pen - gc * vnn
                if Fn < 0.0:
                    Fn = 0.0
                tn, te, td = -gct * (un - vnn * nn), -gct * (ue - vnn * ne), -gct * (ud - vnn * nd)
                a = self._anchor[i]
                if a is not None:
                    dn, de_, dd = pn - a[0], pe - a[1], pd - a[2]
                    dot = dn * nn + de_ * ne + dd * nd
                    tn -= gkt * (dn - dot * nn)
                    te -= gkt * (de_ - dot * ne)
                    td -= gkt * (dd - dot * nd)
                ft = math.sqrt(tn * tn + te * te + td * td)
                if ft > mu * Fn:
                    kf = mu * Fn / ft
                    tn, te, td = tn * kf, te * kf, td * kf
                gn_, ge_, gd_ = Fn * nn + tn, Fn * ne + te, Fn * nd + td
                bx = r00 * gn_ + r10 * ge_ + r20 * gd_
                by = r01 * gn_ + r11 * ge_ + r21 * gd_
                bz = r02 * gn_ + r12 * ge_ + r22 * gd_
                Fx += bx
                Fy += by
                Fz += bz
                Mx += cy * bz - cz * by
                My += cz * bx - cx * bz
                Mz += cx * by - cy * bx
                gforce += Fn
        # (6.6a-f): the rigid body, with gravity in body axes
        g, im = self.g, 1.0 / self.m
        udot = r * v - q * w + Fx * im - g * st
        vdot = p * w - r * u + Fy * im + g * sf * ct
        wdot = q * u - p * v + Fz * im + g * cf * ct
        G1, G2, G3, G4, G5, G6, G7, G8 = self.gamma
        pdot = G1 * p * q - G2 * q * r + G3 * Mx + G4 * Mz
        qdot = G5 * p * r - G6 * (p * p - r * r) + My / self.Jyy
        rdot = G7 * p * q - G1 * q * r + G4 * Mx + G8 * Mz
        # the position, and (6.6g-i) the Euler angles
        tt = st / ct
        qr = q * sf + r * cf
        # (3.41) the servos: a first-order response inside the rate limit, the setpoint inside the throw
        its, rl = 1.0 / self.T_servo, self.rate_lim
        sd = []
        for dlt, c_, lim in zip((da, de, dr), c[:3], self.d_max):
            sd.append(clamp((clamp(c_, -lim, lim) - dlt) * its, -rl, rl))
        if rec is not None:
            shaft_mr = TWO_PI * kQm * (n1 * n1 * n1 + n2 * n2 * n2 + n3 * n3 * n3 + n4 * n4 * n4)
            shaft_fw = TWO_PI * self._kQf * nf * nf * nf
            rec.update(Va=V, alpha=alpha, beta=beta, sigma=sg, rho=rho, air_body=(ua, va, wa),
                       cD=cD, cL=cL, cm=cm, cY=cY, cl=cl, cn=cn, T=T,
                       delta_t=(T / self._kTf if self._kTf > 0.0 else 0.0),
                       J=(ua / (nf * self.D_fw) if nf > 1e-6 else 0.0), thrust=thrust,
                       shaft_mr_w=shaft_mr, shaft_fw_w=shaft_fw,
                       lift_w=shaft_mr / self.eta_drive, cruise_w=shaft_fw / self.eta_drive + self.avionics,
                       ground_n=gforce, wind=self._w,
                       power_nc=Fx * u + Fy * v + Fz * w + Mx * p + My * q + Mz * r,
                       accel_body=(Fx * im, Fy * im, Fz * im))
        return [r00 * u + r01 * v + r02 * w, r10 * u + r11 * v + r12 * w, r20 * u + r21 * v + r22 * w,
                udot, vdot, wdot, pdot, qdot, rdot,
                p + tt * qr, q * cf - r * sf, qr / ct,
                sd[0], sd[1], sd[2],
                (c[3] - nf) / self.tau_fw, (c[4] - n1) / self.tau_mr, (c[5] - n2) / self.tau_mr,
                (c[6] - n3) / self.tau_mr, (c[7] - n4) / self.tau_mr]

    def evaluate(self):
        """Air data, coefficients, thrusts and the rest at the current state (fills self.out)."""
        out = {}
        self.rates(self.x, self._clipped(), out)
        self.out = out
        return out

    def _clipped(self):
        c = self.cmd
        fm, mm = self.eta_fw_max, self.eta_mr_max
        return (c[0], c[1], c[2], clamp(c[3], 0.0, fm), clamp(c[4], 0.0, mm), clamp(c[5], 0.0, mm),
                clamp(c[6], 0.0, mm), clamp(c[7], 0.0, mm))

    # ---- stepping
    def step(self):
        """One fixed step: the wind and the air density sampled and held, the commands clipped
        to the motors' range, RK4 over dt, the surfaces held inside their throw, the bristles
        moved, the batteries drawn."""
        dt, x = self.dt, self.x
        self._w = self._sample_wind()
        self._air(self.rho_at(x[2]))
        self._near = (-x[2] - self.ground(x[0], x[1])) < self._gcheck
        if self._near:
            self._gnorm = ground_normal(self.ground, x[0], x[1])
        c = self._clipped()
        out = {}
        k1 = self.rates(x, c, out)
        self.out = out
        if out["sigma"] > 0.0:
            (alo, ahi), (blo, bhi) = self.alpha_lim, self.beta_lim
            if not alo <= out["alpha"] <= ahi:
                self.guards["alpha"] += 1
            if not blo <= out["beta"] <= bhi:
                self.guards["beta"] += 1
        h2 = 0.5 * dt
        k2 = self.rates([a + h2 * b for a, b in zip(x, k1)], c)
        k3 = self.rates([a + h2 * b for a, b in zip(x, k2)], c)
        k4 = self.rates([a + dt * b for a, b in zip(x, k3)], c)
        h6 = dt / 6.0
        x = [a + h6 * (b1 + 2.0 * (b2 + b3) + b4) for a, b1, b2, b3, b4 in zip(x, k1, k2, k3, k4)]
        for i, lim in enumerate(self.d_max):
            x[12 + i] = clamp(x[12 + i], -lim, lim)
        for i in range(15, 20):
            if x[i] < 0.0:
                x[i] = 0.0
        self.pusher_angle = (self.pusher_angle + TWO_PI * x[15] * dt) % TWO_PI
        for i in range(4):
            self.rotor_angle[i] = (self.rotor_angle[i] + TWO_PI * x[16 + i] * dt) % TWO_PI
        self.x = x
        self.lift_j -= out["lift_w"] * dt
        self.cruise_j -= out["cruise_w"] * dt
        if self._near:
            self._move_anchors()
        elif any(a is not None for a in self._anchor):
            self._anchor = [None, None, None, None]
        self.t += dt
        self.steps += 1

    def _move_anchors(self):
        """Each foot's bristle: planted where the foot first touches, dragged along when its
        spring would pass Coulomb's limit (the foot is sliding), dropped when it lifts."""
        x = self.x
        R = rot_nb(x[9], x[10], x[11]).tolist()
        nn, ne, nd = self._gnorm
        for i, (cx, cy, cz) in enumerate(self._feet):
            pn = x[0] + R[0][0] * cx + R[0][1] * cy + R[0][2] * cz
            pe = x[1] + R[1][0] * cx + R[1][1] * cy + R[1][2] * cz
            pd = x[2] + R[2][0] * cx + R[2][1] * cy + R[2][2] * cz
            pen = -(self.ground(pn, pe) + pd) * nd
            if pen <= 0.0:
                self._anchor[i] = None
                continue
            a = self._anchor[i]
            if a is None:
                self._anchor[i] = (pn, pe, pd)
                continue
            dn, de, dd = pn - a[0], pe - a[1], pd - a[2]
            dot = dn * nn + de * ne + dd * nd
            dn, de, dd = dn - dot * nn, de - dot * ne, dd - dot * nd
            dist = math.sqrt(dn * dn + de * de + dd * dd)
            lim = self._mu * self._gk * pen / self._gkt
            if dist > lim:
                k = lim / dist
                self._anchor[i] = (pn - dn * k, pe - de * k, pd - dd * k)

    def run(self, seconds, before_step=None):
        """Step for `seconds`; before_step(self) runs before every step (an autopilot's update)."""
        for _ in range(int(round(seconds / self.dt))):
            if before_step is not None:
                before_step(self)
            self.step()

    # ---- setting a state
    def set_state(self, n=0.0, e=0.0, h=0.0, u=0.0, v=0.0, w=0.0, p=0.0, q=0.0, r=0.0, phi=0.0, theta=0.0, psi=0.0,
                  delta_a=None, delta_e=None, delta_r=None, eta_fw=0.0, eta_mr=0.0):
        """The state (h up; the rest in STATE's terms), the surfaces at rest on the given
        deflections (default: their trim) and the motors at the given speeds (eta_mr a number
        or four), all commanded to stay there."""
        da = self.da0 if delta_a is None else delta_a
        de = self.de0 if delta_e is None else delta_e
        dr = self.dr0 if delta_r is None else delta_r
        mr = [float(eta_mr)] * 4 if np.isscalar(eta_mr) else [float(a) for a in eta_mr]
        self.x = [n, e, -h, u, v, w, p, q, r, phi, theta, psi, da, de, dr, eta_fw] + mr
        self.cmd = [da, de, dr, eta_fw] + mr
        self._anchor = [None, None, None, None]
        self._air(self.rho_at(-h))

    def hover_speeds(self, rho=None):
        """The four lift rotor speeds that carry the weight level in still air, rev/s: the
        front pair and the rear pair differ, since the centre of gravity is not midway between
        them (the front axes are 0.353 m ahead of it, the rear ones 0.447 m behind)."""
        kT = self.kTm_rho * (self.rho if rho is None else rho)
        Bm = np.array([[1.0, 1.0, 1.0, 1.0], [-r[1] for r in self._rot], [r[0] for r in self._rot],
                       [r[2] for r in self._rot]])
        T = np.linalg.solve(Bm, np.array([self.m * self.g, 0.0, 0.0, 0.0]))
        return [math.sqrt(max(float(t), 0.0) / kT) for t in T]

    def set_hover(self, n=0.0, e=0.0, h=0.0, yaw=0.0):
        """Level and still at (n, e, h), heading yaw, on the lift rotors."""
        self.set_state(n, e, h, psi=yaw, eta_mr=self.hover_speeds(self.rho_at(-h)))

    def place_on_ground(self, n=0.0, e=0.0, yaw=0.0, eta_mr=0.0):
        """At rest on its feet at (n, e), lying on the ground's slope, the motors stopped (or
        the lift rotors at eta_mr) and commanded to stay so."""
        nn, ne, nd = ground_normal(self.ground, n, e)
        b3 = np.array([-nn, -ne, -nd])                                   # body z: into the ground
        xc = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        b2 = np.cross(b3, xc)
        b2 /= np.linalg.norm(b2)
        b1 = np.cross(b2, b3)
        phi, theta, psi = euler_of(np.column_stack([b1, b2, b3]))
        sag = self.m * self.g * (-nd) / (4.0 * self._gk)
        pos = np.array([n, e, -self.ground(n, e)]) - (self._feet[0][2] - sag) * b3
        self.set_state(float(pos[0]), float(pos[1]), -float(pos[2]), phi=phi, theta=theta, psi=psi, eta_mr=eta_mr)

    def trim(self, va, n=0.0, e=0.0, h=0.0, psi=0.0, gamma=0.0, apply=True, iters=60, course=None):
        """Straight flight on the wing at airspeed va and air-path angle gamma (rad), without
        sideslip, the lift rotors stopped: solves the six rigid-body equations and the path
        angle for alpha, phi, theta, the three surfaces and the pusher's speed, in still air.
        With apply, the aircraft is put there (in the wind it has, at the same air-relative
        velocity), heading psi, or, given a course, at the heading that makes that ground track
        in the wind. Returns the solution as a dict; ``limited`` names a surface or the pusher
        if the solution is outside its range (the model cannot fly this steadily)."""
        z = np.array([0.05, 0.0, 0.05 + gamma, self.de0, self.da0, self.dr0, 90.0])   # alpha phi theta de da dr eta
        scale = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0])
        keep = (self.x, self.cmd, self._w, self.rho, self._near)
        self._w, self._near = (0.0, 0.0, 0.0), False
        self._air(self.rho_at(-h))

        def state(z):
            al, ph, th, de, da, dr, eta = (float(a) for a in z)
            return [0.0, 0.0, -h, va * math.cos(al), 0.0, va * math.sin(al), 0.0, 0.0, 0.0, ph, th, 0.0,
                    da, de, dr, eta, 0.0, 0.0, 0.0, 0.0]

        def resid(z):
            x = state(z)
            f = self.rates(x, x[12:])
            return np.array(f[3:9] + [-f[2] - va * math.sin(gamma)])
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
            except np.linalg.LinAlgError:                   # no steady flight here (the pusher's thrust at its floor)
                break
            z = z + step / max(1.0, float(np.max(np.abs(step / (scale * 0.3)))))   # damped: at most 0.3 rad a pass
            if np.max(np.abs(step / scale)) < 1e-12:
                break
        al, ph, th, de, da, dr, eta = (float(a) for a in z)
        out = {}
        x = state(z)
        self.rates(x, x[12:], out)
        limited = [k for k, val, lim in (("aileron", da, self.d_max[0]), ("elevator", de, self.d_max[1]),
                                         ("rudder", dr, self.d_max[2])) if abs(val) > lim]
        if eta > self.eta_fw_max:
            limited.append("pusher")
        if not self.alpha_lim[0] <= al <= self.alpha_lim[1]:
            limited.append("alpha")
        sol = {"va": va, "alpha": al, "phi": ph, "theta": th, "de": de, "da": da, "dr": dr, "eta": eta,
               "T": out["T"], "delta_t": out["delta_t"], "J": out["J"], "cL": out["cL"], "cD": out["cD"],
               "cruise_w": out["cruise_w"], "residual": float(np.max(np.abs(resid(z)))), "limited": limited,
               "rho": self.rho}
        self.x, self.cmd, self._w, rho_, self._near = keep
        self._air(rho_)
        if apply:
            u, w = va * math.cos(al), va * math.sin(al)

            def place(psi):
                wb = rot_nb(ph, th, psi).T @ np.asarray(self._wind_at(n, e, -h), float)
                self.set_state(n, e, h, u + wb[0], wb[1], w + wb[2], phi=ph, theta=th, psi=psi,
                               delta_a=da, delta_e=de, delta_r=dr, eta_fw=eta)
            if course is not None:
                psi = course
                for _ in range(8):
                    place(psi)
                    psi += wrap(course - self.course())
            place(psi)
        return sol

    def _wind_at(self, n, e, d):
        """The mean wind at a point, without moving a Wind's turbulence on."""
        w = self.wind
        if w is None:
            return (0.0, 0.0, 0.0)
        if isinstance(w, Wind):
            self._w = w.mean(n, e, -d)
            return self._w
        if callable(w):
            w = w(self.t, n, e, d)
        self._w = (float(w[0]), float(w[1]), float(w[2]))
        return self._w

    # ---- what a scene reads
    def euler(self):
        return self.x[9], self.x[10], self.x[11]

    def rotation(self):
        """R_b^n as a 3 x 3 array."""
        return rot_nb(self.x[9], self.x[10], self.x[11])

    def ground_velocity(self):
        """NED velocity over the ground."""
        return self.rotation() @ np.array(self.x[3:6])

    def air_velocity_ned(self):
        return self.ground_velocity() - np.asarray(self._w, float)

    def airdata(self):
        """(V, alpha, beta, u_a) at the current state in the wind the last step used."""
        x = self.x
        ua, va, wa = np.array(x[3:6]) - self.rotation().T @ np.asarray(self._w, float)
        V = math.sqrt(ua * ua + va * va + wa * wa)
        if V < 1e-6:
            return 0.0, 0.0, 0.0, 0.0
        return V, math.atan2(wa, ua), math.asin(clamp(va / V, -1.0, 1.0)), float(ua)

    def course(self):
        """The ground track's direction chi, from north toward east (rad)."""
        v = self.ground_velocity()
        return math.atan2(v[1], v[0])

    def tilt(self):
        """The angle between the lift rotors' axis and the vertical, rad."""
        return math.acos(clamp(math.cos(self.x[9]) * math.cos(self.x[10]), -1.0, 1.0))

    def height_above_ground(self):
        """The lowest foot's height above the ground, m (negative: the springs' sag)."""
        x = self.x
        R = self.rotation().tolist()
        best = float("inf")
        for cx, cy, cz in self._feet:
            pn = x[0] + R[0][0] * cx + R[0][1] * cy + R[0][2] * cz
            pe = x[1] + R[1][0] * cx + R[1][1] * cy + R[1][2] * cz
            pd = x[2] + R[2][0] * cx + R[2][1] * cy + R[2][2] * cz
            best = min(best, -pd - self.ground(pn, pe))
        return best

    @property
    def on_ground(self):
        """The ground carried a tenth of the weight or more at the last step's start."""
        return self.out.get("ground_n", 0.0) > 0.1 * self.m * self.g

    @property
    def lift_fraction(self):
        """What is left of the lift rotors' battery, 0..1."""
        return self.lift_j / (self.lift_wh * 3600.0)

    @property
    def cruise_fraction(self):
        return self.cruise_j / (self.cruise_wh * 3600.0)

    def world_position(self, h0=0.0):
        """The centre of gravity (the .glb's origin) in the world."""
        return ned_to_world(self.x[0], self.x[1], self.x[2], h0)

    def world_rotation(self):
        """The .glb's rotation in the world: model -> body -> NED -> world."""
        return C_WN @ self.rotation() @ C_BM

    def energy(self):
        """Kinetic (translation and rotation of the airframe) plus potential energy per the NED
        origin, J. The propellers' own spin is not in it (nor in the model)."""
        x = self.x
        w = np.array(x[6:9])
        return 0.5 * self.m * (x[3] * x[3] + x[4] * x[4] + x[5] * x[5]) + 0.5 * float(w @ self.J @ w) - self.m * self.g * x[2]

    def angular_momentum_ned(self):
        return self.rotation() @ (self.J @ np.array(self.x[6:9]))

    def carry_alpha(self, V):
        """The angle of attack at which the wing's lift (6.8e, the elevator at its trim) is the
        weight at airspeed V, or None where the identified range has none (too slow)."""
        if V < 1.0:
            return None
        cL0, cLa, cLa2 = self.a["cL0"], self.a["cLa"], self.a["cLa2"]
        c0 = cL0 - self.m * self.g / (0.5 * self.rho * V * V * self.S)
        disc = cLa * cLa - 4.0 * cLa2 * c0
        if disc < 0.0:
            return None
        al = (-cLa + math.sqrt(disc)) / (2.0 * cLa2)
        return al if self.alpha_lim[0] <= al <= self.alpha_lim[1] else None

    def jacobian(self, x=None):
        """Numerical Jacobians of rates at the state x (default: the current one), by central
        differences, in the held wind: A, 12 x 12 on the rigid states, and B, 12 x 5 on the
        actuators' own states (aileron, elevator, rudder, the pusher's speed in rev/s and its
        SQUARE, the thesis's delta_t)."""
        x = np.array(self.x if x is None else x, float)
        keep = self._near
        self._near = False

        def f(xv):
            xl = [float(a) for a in xv]
            return np.array(self.rates(xl, xl[12:])[:N_RIGID])
        A = np.zeros((N_RIGID, N_RIGID))
        B = np.zeros((N_RIGID, 5))
        for k in range(N_RIGID):
            hk = 1e-6 * max(1.0, abs(x[k]))
            xp, xm = x.copy(), x.copy()
            xp[k] += hk
            xm[k] -= hk
            A[:, k] = (f(xp) - f(xm)) / (2.0 * hk)
        for j, k in enumerate((12, 13, 14, 15)):
            hk = 1e-6 * max(1.0, abs(x[k]))
            xp, xm = x.copy(), x.copy()
            xp[k] += hk
            xm[k] -= hk
            B[:, j] = (f(xp) - f(xm)) / (2.0 * hk)
        B[:, 4] = B[:, 3] / (2.0 * x[15]) if x[15] > 1e-9 else 0.0
        self._near = keep
        return A, B


_INDEX = {k: i for i, k in enumerate(STATE)}
LON = (3, 5, 7, 10)            # u, w, q, theta: the thesis's longitudinal state (6.17)
LAT = (4, 6, 8, 9)             # v, p, r, phi: its lateral-directional one (6.18)


# --------------------------------------------------------------------------- #
#  The autopilot: OURS
# --------------------------------------------------------------------------- #
class Autopilot:
    """OURS, not PX4: two autopilots and the way between them.

    ``mode`` is one of

        "mc"     the hover: m350_rig.Autopilot's cascade on the four lift rotors (position P,
                 velocity PI, the thrust vector inside a tilt limit, quaternion attitude P,
                 body-rate PID, mixer). The pusher is stopped and the surfaces rest at their
                 trim. Without a yaw command of its own it weathervanes: the nose turns into
                 the wind the velocity loop's integral has learned to lean against.
        "fw"     the wing: x8_rig.Autopilot's successive loop closure (roll on the aileron,
                 pitch on the elevator, course or heading on roll, altitude on pitch, airspeed
                 on the pusher), with the rudder holding the coordinated turn's yaw rate and
                 working against the sideslip. The lift rotors are stopped.
        "front"  hover to wing. The pusher ramps up; the lift rotors hold the height, and hold
                 the wings on the course's bank and the nose at the transition's pitch with a
                 weight that fades from 1 at ``blend`` m/s of airspeed to 0 at ``speed``; the
                 surfaces fly the same attitude throughout. At ``speed`` it is "fw".
        "back"   wing to hover. The pusher stops; the lift rotors take the attitude and the
                 height at once and slow the aircraft along its track; while the wing can
                 still carry the weight the nose stays a margin under the angle of attack at
                 which it would, so the braking does not balloon it. Below ``back_speed`` it
                 is "mc".

    command() takes both halves' setpoints, and what is not given is kept:

        position=(n, e, h), velocity=(vn, ve, vh), yaw=rad, yaw_rate=rad/s     the hover's
        course=rad or heading=rad, altitude=m, airspeed=m/s, roll=rad          the wing's

    position=None flies the velocity; yaw=None gives the yaw back to the weathervane. All in
    NED axes with heights and climb rates UP. transition("fw", course=rad) and
    transition("mc") start the two transitions. ``armed`` False stops every motor; ``idle``
    True holds the lift rotors at idle with the loops reset (on the ground).

    STATE FEEDBACK IS EXACT: position, velocity, attitude, rates and the airspeed are read from
    the model, with no estimator and no sensor noise. It does not know the wind. It does know
    the aircraft's mass, inertia, rotor geometry and coefficients (the wing's gains are
    computed from the thesis's coefficients; the spec's `autopilot` and `control` blocks hold
    the frequencies, damping, integrals and limits)."""

    def __init__(self, aircraft, airspeed=None):
        self.a = a = aircraft
        spec = a.spec
        self.fw_cfg = F = spec["autopilot"]
        self.mc_cfg = C = spec["control"]
        # ---- the hover's loops
        P, V, A, R, W, T = C["position"], C["velocity"], C["attitude"], C["rate"], C["weathervane"], C["transition"]
        self.kp_p, self.kp_h, self.v_xy, self.brake = P["kp_xy"], P["kp_z"], P["speed_xy_max_ms"], P["brake_ms2"]
        self.v_up, self.v_down = P["climb_max_ms"], P["descent_max_ms"]
        self.kp_v, self.ki_v, self.kp_vh, self.ki_vh = V["kp_xy"], V["ki_xy"], V["kp_z"], V["ki_z"]
        self.i_xy_max, self.i_h_max, self.e_i = V["integral_xy_ms2"], V["integral_z_ms2"], V["integral_error_ms"]
        self.kp_att, self.kp_yaw, self.yaw_acc = A["kp_tilt"], A["kp_yaw"], A["yaw_accel_rads2"]
        self.rate_max, self.tilt_max = math.radians(A["rate_max_dps"]), math.radians(A["tilt_max_deg"])
        self.yaw_rate_max = math.radians(A["yaw_rate_max_dps"])
        self.kp_r, self.ki_r, self.kd_r = R["kp"], R["ki"], R["kd"]
        self.kp_ry, self.ki_ry, self.i_r_max = R["kp_yaw"], R["ki_yaw"], R["integral_rads2"]
        self.d_alpha = 1.0 - math.exp(-TWO_PI * R["d_cutoff_hz"] * a.dt)
        self.wv_gain, self.wv_min, self.wv_rate = W["gain_per_s"], math.radians(W["min_tilt_deg"]), math.radians(W["rate_max_dps"])
        # the mixer: [T, Mx, My, Mz] = B [T_1..T_4]
        cq = a.kQm_rho / a.kTm_rho
        Bm = np.array([[1.0, 1.0, 1.0, 1.0], [-r[1] for r in a._rot], [r[0] for r in a._rot],
                       [r[2] * cq for r in a._rot]])
        self._mix = [tuple(float(v) for v in row) for row in np.linalg.inv(Bm)]
        self._Jd = (a.Jxx, a.Jyy, a.Jzz)
        # ---- the transition
        self.tr_speed, self.tr_blend, self.tr_pitch = T["speed_ms"], T["blend_ms"], math.radians(T["pitch_deg"])
        self.tr_throttle, self.tr_ramp, self.tr_timeout = T["throttle_front"], T["ramp_s"], T["timeout_s"]
        self.tr_decel, self.tr_back_speed = T["decel_ms2"], T["back_speed_ms"]
        self.tr_margin, self.surf_speed = math.radians(T["pitch_margin_deg"]), T["surface_speed_ms"]
        # ---- the wing's loops
        self.va_range = tuple(F["airspeed_range"])
        self.va_c = clamp(airspeed or F["airspeed"], *self.va_range)
        self.hold = "course"
        self.design(self.va_c)
        self.mode = "mc"
        self.armed = True
        self.idle = False
        self.aborted = False
        self.reset()

    # ---- the wing's gains, from the thesis's coefficients
    def design(self, va):
        """The wing's gains at airspeed va (and the air density at the aircraft's height)."""
        a, A, cfg = self.a, self.a.a, self.fw_cfg
        probe = Babyshark(a.spec, dt=a.dt, h0=a.h0, rho=a.rho_fixed)
        probe.J0 = a.J0
        tr = probe.trim(va, h=a.h, apply=True)
        self.trim = tr
        rho = a.rho_at(a.x[2])
        qS = 0.5 * rho * va * va * a.S
        G3, G4 = a.gamma[2], a.gamma[3]
        k2 = 0.5 / a.V0
        # roll: phi'' = -a1 phi' + a2 delta_a
        a1 = -qS * a.b * (G3 * A["clp"] + G4 * A["cnp"]) * a.b * k2
        a2 = qS * a.b * G3 * A["clda"]
        c = cfg["roll"]
        w, z = c["wn"], c["zeta"]
        self.a_phi = (a1, a2)
        self.kp_phi = w * w / a2
        self.kd_phi = max(0.0, (2.0 * z * w - a1) / a2)
        self.ki_phi = c["ki"] * self.kp_phi
        self.i_phi_max = math.radians(c["integral_deg"]) / self.ki_phi
        self.phi_max = math.radians(c["limit_deg"])
        c = cfg["course"]
        self.w_chi, self.z_chi = w / c["bandwidth_ratio"], c["zeta"]
        # pitch: theta'' = -d theta' - a2 theta + a3 delta_e, d the short period's own damping
        # (the pitch damping and the lift's, the trace of the (w, q) block), a3 < 0
        Aj, Bj = probe.jacobian()
        a2 = -qS * a.cbar * A["cma"] / a.Jyy
        a3 = qS * a.cbar * A["cmde"] / a.Jyy
        d_sp = -(Aj[5, 5] + Aj[7, 7])
        c = cfg["pitch"]
        self.a_th = (d_sp, a2, a3)
        self.kp_th = c["stiffness"] * a2 / a3
        w_cl = math.sqrt(a2 * (1.0 + c["stiffness"]))
        self.kd_th = max(0.0, (2.0 * c["zeta"] * w_cl - d_sp) / -a3)
        self.ki_th = c["wi"] * self.kp_th
        self.i_th_max = math.radians(c["integral_deg"]) / abs(self.ki_th)
        self.th_max = math.radians(c["limit_deg"])
        # altitude (h' = V theta)
        c = cfg["altitude"]
        self.kp_h_fw = 2.0 * c["zeta"] * c["wn"] / va
        self.ki_h_fw = c["wn"] * c["wn"] / va
        self.h_zone = c["zone"]
        # the steepest climb the pusher can hold at this airspeed: what its thrust at full speed
        # has over the trim's, as an angle; the pitch command stops at a share of it, so a long
        # climb costs height rate and not airspeed
        nm = a.eta_fw_max
        t_full = a.kTf_rho * rho * (nm * nm - nm * va / (a.D_fw * a.J0))
        self.climb_max = max(0.0, (t_full - tr["T"]) / (a.m * a.g))
        self.th_up = min(self.th_max, tr["theta"] + c["climb_fraction"] * self.climb_max)
        # a turn at bank phi needs 1 / cos(phi) of the lift: that much more angle of attack, fed
        # forward into the pitch command (cL / cL_alpha at the trim, per unit of extra load)
        self.turn_ff = tr["cL"] / (A["cLa"] + 2.0 * A["cLa2"] * tr["alpha"])
        # airspeed on the pusher's speed: V' = -a_V1 V + a_V2 n, from the model's Jacobian at the trim
        av1, av2 = -Aj[3, 3], Bj[3, 3]
        c = cfg["speed"]
        w, z = c["wn"], c["zeta"]
        self.kp_va = (2.0 * z * w - av1) / av2
        self.ki_va = w * w / av2
        self.a_v = (av1, av2)
        c = cfg["yaw"]
        self.kr, self.kb, self.dr_max = c["kr"], c["kb"], math.radians(c["limit_deg"])

    def reset(self):
        """Setpoints at the aircraft's own position, heading and height; the integrals emptied."""
        a = self.a
        x = a.x
        self.pos_c = (x[0], x[1], -x[2])
        self.vel_c = (0.0, 0.0, 0.0)
        self.acc_c = (0.0, 0.0, 0.0)
        self.yaw_c = x[11]
        self.yaw_rate_c = 0.0
        self.yaw_free = True                               # the weathervane has the yaw
        self.hold_position = True
        self.i_v = [0.0, 0.0, 0.0]
        self.i_r = [0.0, 0.0, 0.0]
        self._wd = [0.0, 0.0, 0.0]
        self._w_prev = (x[6], x[7], x[8])
        self.tilt_c = 0.0
        self.thrust_c = a.m * a.g
        self.vel_sp = (0.0, 0.0, 0.0)
        self.saturated = 0
        self.chi_c = x[11]
        self.h_c = -x[2]
        self.phi_cmd = 0.0
        self.i_phi = self.i_chi = self.i_h = self.i_va = self.i_th = 0.0
        self.phi_c = self.theta_c = 0.0
        self.t_tr = 0.0
        self.w_mc = 1.0
        self._v0 = 0.0

    def command(self, position=KEEP, velocity=KEEP, yaw=KEEP, acceleration=KEEP, yaw_rate=KEEP,
                course=None, heading=None, altitude=None, airspeed=None, roll=None):
        if position is not KEEP:
            self.hold_position = position is not None
            if position is not None:
                self.pos_c = (float(position[0]), float(position[1]), float(position[2]))
                self.h_c = self.pos_c[2]
        if velocity is not KEEP:
            self.vel_c = (0.0, 0.0, 0.0) if velocity is None else tuple(float(v) for v in velocity)
        if acceleration is not KEEP:
            self.acc_c = (0.0, 0.0, 0.0) if acceleration is None else tuple(float(v) for v in acceleration)
        if yaw is not KEEP:
            self.yaw_free = yaw is None
            if yaw is not None:
                self.yaw_c = float(yaw)
        if yaw_rate is not KEEP:
            self.yaw_rate_c = 0.0 if yaw_rate is None else float(yaw_rate)
        if (course is not None or heading is not None) and self.hold == "roll":
            self.i_chi = 0.0                               # the course loop starts anew after a roll command
        if course is not None:
            self.chi_c, self.hold = course, "course"
        if heading is not None:
            self.chi_c, self.hold = heading, "heading"
        if roll is not None:
            self.phi_cmd, self.hold = roll, "roll"
        if altitude is not None:
            self.h_c = altitude
            self.pos_c = (self.pos_c[0], self.pos_c[1], altitude)
        if airspeed is not None:
            airspeed = clamp(airspeed, *self.va_range)
            if abs(airspeed - self.va_c) > 1e-9:
                self.va_c = airspeed
                self.design(airspeed)

    def transition(self, to, course=None):
        """Start the front transition (to="fw", along `course`, default the heading it has) or
        the back transition (to="mc", along the ground track it has)."""
        a = self.a
        if to == "fw" and self.mode in ("mc", "back"):
            self.chi_c = a.x[11] if course is None else course
            self.hold = "course"
            self.yaw_c, self.yaw_free = self.chi_c, False
            self.h_c = self.pos_c[2] if self.hold_position else a.h
            self.i_phi = self.i_chi = self.i_h = self.i_va = self.i_th = 0.0
            self.mode, self.t_tr, self.aborted = "front", 0.0, False
        elif to == "mc" and self.mode in ("fw", "front"):
            vg = a.ground_velocity()
            self._v0 = math.hypot(vg[0], vg[1])
            self._track = math.atan2(vg[1], vg[0]) if self._v0 > 1.0 else a.x[11]
            self.yaw_c, self.yaw_free = self._track, False
            self.pos_c = (a.x[0], a.x[1], self.h_c)
            self.hold_position = False
            self.i_v = [0.0, 0.0, 0.0]
            self.i_r = [0.0, 0.0, 0.0]
            self._w_prev = (a.x[6], a.x[7], a.x[8])
            self.mode, self.t_tr = "back", 0.0

    # ---- the hover's pieces (m350_rig.Autopilot's, on this airframe)
    def _vertical(self, vh, ff=0.0, wing=False):
        """Height -> climb rate -> the vertical specific force fz (up, m/s^2, gravity in) asked
        of the lift rotors: not under 0.3 g in a hover; in a transition (wing) down to what
        the rotors give at idle, since the wing carries the rest."""
        a, dt = self.a, self.a.dt
        vch = clamp(ff + self.kp_h * (self.pos_c[2] - a.h), -self.v_down, self.v_up)
        eh = vch - vh
        ah = self.acc_c[2] + self.kp_vh * eh + self.i_v[2]
        fz = clamp(ah + a.g, 4.0 * a._kTm * a.eta_idle * a.eta_idle / a.m if wing else 0.3 * a.g, 2.0 * a.g)
        if fz == ah + a.g or eh * self.i_v[2] < 0.0:
            self.i_v[2] = clamp(self.i_v[2] + self.ki_vh * clamp(eh, -self.e_i, self.e_i) * dt, -self.i_h_max, self.i_h_max)
        return fz, vch

    def _horizontal(self, vn, ve, vcn, vce, fz, pitch_up_max=None):
        """Velocity -> the horizontal acceleration (an, ae), inside the tilt limit and, in the
        back transition, inside the nose-up limit along the yaw setpoint."""
        dt = self.a.dt
        en, ee = vcn - vn, vce - ve
        iv = self.i_v
        an = self.acc_c[0] + self.kp_v * en + iv[0]
        ae = self.acc_c[1] + self.kp_v * ee + iv[1]
        fh = math.hypot(an, ae)
        fmax = fz * math.tan(self.tilt_max)
        sat = fh > fmax
        if sat:
            an, ae = an * fmax / fh, ae * fmax / fh
        if pitch_up_max is not None:
            cy, sy = math.cos(self.yaw_c), math.sin(self.yaw_c)
            af, ar = an * cy + ae * sy, -an * sy + ae * cy
            lim = -fz * math.tan(pitch_up_max)
            if af < lim:
                af, sat = lim, True
                an, ae = af * cy - ar * sy, af * sy + ar * cy
        ei = self.e_i
        if not sat or en * iv[0] < 0.0:
            iv[0] = clamp(iv[0] + self.ki_v * clamp(en, -ei, ei) * dt, -self.i_xy_max, self.i_xy_max)
        if not sat or ee * iv[1] < 0.0:
            iv[1] = clamp(iv[1] + self.ki_v * clamp(ee, -ei, ei) * dt, -self.i_xy_max, self.i_xy_max)
        return an, ae

    def _lift(self, qd, fz, w_att=1.0):
        """The attitude wanted (a quaternion w, x, y, z) and the vertical specific force ->
        body rates -> moments (times w_att) -> the four lift rotors' speed commands."""
        a = self.a
        x, dt = a.x, a.dt
        p, q, r, phi, th, psi = x[6:12]
        ch, sh = math.cos(0.5 * phi), math.sin(0.5 * phi)
        ct, st = math.cos(0.5 * th), math.sin(0.5 * th)
        cp, sp = math.cos(0.5 * psi), math.sin(0.5 * psi)
        qw, qx = ch * ct * cp + sh * st * sp, sh * ct * cp - ch * st * sp
        qy, qz = ch * st * cp + sh * ct * sp, ch * ct * sp - sh * st * cp
        dw, dx, dy, dz = qd
        # the rotation from the attitude it has to the one wanted, in body axes: conj(q) x q_d
        ew = qw * dw + qx * dx + qy * dy + qz * dz
        ex = qw * dx - qx * dw - qy * dz + qz * dy
        ey = qw * dy + qx * dz - qy * dw - qz * dx
        ez = qw * dz - qx * dy + qy * dx - qz * dw
        if ew < 0.0:
            ex, ey, ez = -ex, -ey, -ez
        rm = self.rate_max
        pc = clamp(2.0 * self.kp_att * ex, -rm, rm)
        qc = clamp(2.0 * self.kp_att * ey, -rm, rm)
        eyaw = 2.0 * ez
        lin = self.yaw_acc / (self.kp_yaw * self.kp_yaw)
        if abs(eyaw) > lin:
            rc = math.copysign(math.sqrt(2.0 * self.yaw_acc * (abs(eyaw) - 0.5 * lin)), eyaw)
        else:
            rc = self.kp_yaw * eyaw
        rc = clamp(rc + self.yaw_rate_c, -self.yaw_rate_max, self.yaw_rate_max)
        # rates -> moments
        a_ = self.d_alpha
        wd = self._wd
        wd[0] += a_ * ((p - self._w_prev[0]) / dt - wd[0])
        wd[1] += a_ * ((q - self._w_prev[1]) / dt - wd[1])
        wd[2] += a_ * ((r - self._w_prev[2]) / dt - wd[2])
        self._w_prev = (p, q, r)
        ir = self.i_r
        ep, eq, er = pc - p, qc - q, rc - r
        Mx = w_att * self._Jd[0] * (self.kp_r * ep + ir[0] - self.kd_r * wd[0])
        My = w_att * self._Jd[1] * (self.kp_r * eq + ir[1] - self.kd_r * wd[1])
        Mz = w_att * self._Jd[2] * (self.kp_ry * er + ir[2])
        # collective: the vertical part of the thrust is m fz at the attitude it has
        T = a.m * fz / max(math.cos(phi) * math.cos(th), 0.5)
        self.thrust_c = T
        kT = a._kTm
        tmin, tmax = kT * a.eta_idle * a.eta_idle, kT * a.eta_mr_max * a.eta_mr_max
        A = self._mix
        base = [A[i][0] * T + A[i][1] * Mx + A[i][2] * My for i in range(4)]
        lo, hi = min(base), max(base)
        sat = 0
        if hi - lo > tmax - tmin:                         # roll and pitch alone do not fit: scale them
            k = (tmax - tmin) / (hi - lo)
            mean = 0.25 * (base[0] + base[1] + base[2] + base[3])
            base = [mean + k * (b - mean) for b in base]
            lo, hi = min(base), max(base)
            sat = 3
        if hi > tmax:                                     # then the collective gives way
            base = [b - (hi - tmax) for b in base]
            sat = max(sat, 2)
        elif lo < tmin:
            base = [b + (tmin - lo) for b in base]
            sat = max(sat, 2)
        ky = 1.0                                          # yaw is given up first
        for i in range(4):
            dy_ = A[i][3] * Mz
            if dy_ > 1e-12:
                ky = min(ky, max(0.0, (tmax - base[i]) / dy_))
            elif dy_ < -1e-12:
                ky = min(ky, max(0.0, (tmin - base[i]) / dy_))
        if ky < 1.0:
            sat = max(sat, 1)
        self.saturated = sat
        ki = self.ki_r * dt
        ir[0] = clamp(ir[0] + ki * ep, -self.i_r_max, self.i_r_max)
        ir[1] = clamp(ir[1] + ki * eq, -self.i_r_max, self.i_r_max)
        if ky >= 1.0 or er * ir[2] < 0.0:
            ir[2] = clamp(ir[2] + self.ki_ry * dt * er, -self.i_r_max, self.i_r_max)
        return [math.sqrt(max(base[i] + ky * A[i][3] * Mz, 0.0) / kT) for i in range(4)]

    def _thrust_attitude(self, an, ae, fz):
        """The attitude whose thrust gives (an, ae, fz): body z along -f, body x toward the yaw
        setpoint. Returns the quaternion and the (roll, pitch) it amounts to."""
        fn_ = math.sqrt(an * an + ae * ae + fz * fz)
        b3n, b3e, b3d = -an / fn_, -ae / fn_, fz / fn_
        self.tilt_c = math.acos(clamp(b3d, -1.0, 1.0))
        cy, sy = math.cos(self.yaw_c), math.sin(self.yaw_c)
        b2n, b2e, b2d = -b3d * sy, b3d * cy, b3n * sy - b3e * cy
        k = 1.0 / math.sqrt(b2n * b2n + b2e * b2e + b2d * b2d)
        b2n, b2e, b2d = b2n * k, b2e * k, b2d * k
        b1n, b1e, b1d = b2e * b3d - b2d * b3e, b2d * b3n - b2n * b3d, b2n * b3e - b2e * b3n
        tr = b1n + b2e + b3d
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            qd = (0.25 * s, (b2d - b3e) / s, (b3n - b1d) / s, (b1e - b2n) / s)
        else:                                             # more than 90 deg off level and north: never here
            dx, dy, dz, dw = quat_of(np.array([[b1n, b2n, b3n], [b1e, b2e, b3e], [b1d, b2d, b3d]]))
            qd = (dw, dx, dy, dz)
        return qd, (math.atan2(b2d, b3d), -math.asin(clamp(b1d, -1.0, 1.0)))

    @staticmethod
    def _euler_quat(phi, th, psi):
        ch, sh = math.cos(0.5 * phi), math.sin(0.5 * phi)
        ct, st = math.cos(0.5 * th), math.sin(0.5 * th)
        cp, sp = math.cos(0.5 * psi), math.sin(0.5 * psi)
        return (ch * ct * cp + sh * st * sp, sh * ct * cp - ch * st * sp,
                ch * st * cp + sh * ct * sp, ch * ct * sp - sh * st * cp)

    # ---- the wing's pieces (x8_rig.Autopilot's, on three surfaces)
    def _course_roll(self, V):
        """Course (or heading) -> the roll command."""
        a, dt = self.a, self.a.dt
        if self.hold == "roll":
            phi_c = self.phi_cmd
        else:
            vg = a.ground_velocity()
            gs = math.hypot(vg[0], vg[1])
            if self.hold == "course" and gs > 3.0:
                chi, spd = math.atan2(vg[1], vg[0]), max(gs, 5.0)
            else:
                chi, spd = a.x[11], max(V, 5.0)
            e_chi = wrap(self.chi_c - chi)
            phi_c = (2.0 * self.z_chi * self.w_chi * e_chi + self.w_chi * self.w_chi * self.i_chi) * spd / a.g
            if abs(phi_c) < self.phi_max or phi_c * e_chi < 0.0:
                self.i_chi += e_chi * dt
        return clamp(phi_c, -self.phi_max, self.phi_max)

    def _surfaces(self, phi_c, th_c, V, beta=0.0):
        """Roll -> aileron, pitch -> elevator, and the rudder on the coordinated turn's yaw rate
        and against the sideslip."""
        a, dt, tr = self.a, self.a.dt, self.trim
        p, q, r, phi, th = a.x[6:11]
        e_phi = phi_c - phi
        da = tr["da"] + self.kp_phi * e_phi + self.ki_phi * self.i_phi - self.kd_phi * p
        self.i_phi = clamp(self.i_phi + e_phi * dt, -self.i_phi_max, self.i_phi_max)
        e_th = th_c - th
        de = tr["de"] + self.kp_th * e_th + self.ki_th * self.i_th + self.kd_th * q
        self.i_th = clamp(self.i_th + e_th * dt, -self.i_th_max, self.i_th_max)
        r_turn = a.g * math.sin(phi) * math.cos(th) / max(V, 10.0)
        dr = tr["dr"] + clamp(self.kr * (r - r_turn) - self.kb * beta, -self.dr_max, self.dr_max)
        return da, de, dr

    # ---- one update
    def update(self):
        """Writes the aircraft's surface, pusher and lift rotor commands for its next step."""
        a = self.a
        x, dt = a.x, a.dt
        tr = self.trim
        neutral = [a.da0, a.de0, a.dr0]
        if not self.armed:
            a.cmd = neutral + [0.0, 0.0, 0.0, 0.0, 0.0]
            return
        if self.idle:
            a.cmd = neutral + [0.0] + [a.eta_idle] * 4
            self.i_v = [0.0, 0.0, 0.0]
            self.i_r = [0.0, 0.0, 0.0]
            self._w_prev = (x[6], x[7], x[8])
            return
        V, _, beta, _ = a.airdata()
        vg = a.ground_velocity()
        vn, ve, vh = float(vg[0]), float(vg[1]), -float(vg[2])
        mode = self.mode
        if mode == "fw":
            # course -> roll, altitude -> pitch, airspeed -> the pusher
            self.phi_c = phi_c = self._course_roll(V)
            e_h = self.h_c - a.h
            if abs(e_h) > self.h_zone:
                th_c = self.th_up if e_h > 0.0 else -self.th_max
            else:
                th_c = tr["theta"] + self.kp_h_fw * e_h + self.ki_h_fw * self.i_h
                if -self.th_max < th_c < self.th_up or th_c * e_h < 0.0:
                    self.i_h += e_h * dt
                th_c = clamp(th_c, -self.th_max, self.th_up)
            th_c += self.turn_ff * (1.0 / max(math.cos(x[9]), 0.5) - 1.0)
            self.theta_c = th_c = clamp(th_c, -self.th_max, self.th_max)
            e_v = self.va_c - V
            eta = tr["eta"] + self.kp_va * e_v + self.ki_va * self.i_va
            if 0.0 < eta < a.eta_fw_max or eta * e_v < 0.0:
                self.i_va += e_v * dt
            a.cmd = list(self._surfaces(phi_c, th_c, V, beta)) + [clamp(eta, 0.0, a.eta_fw_max), 0.0, 0.0, 0.0, 0.0]
            return
        if mode == "front":
            self.t_tr += dt
            if V >= self.tr_speed:                         # the wing has it
                self.mode = "fw"
                self.i_h = self.i_va = 0.0
                return self.update()
            if self.t_tr > self.tr_timeout:                # it never got there: back to a hover
                self.aborted = True
                self.transition("mc")
                return self.update()
            self.w_mc = w = clamp((self.tr_speed - V) / (self.tr_speed - self.tr_blend), 0.0, 1.0)
            self.phi_c = phi_c = clamp(self._course_roll(V), -0.5 * self.phi_max, 0.5 * self.phi_max)
            self.theta_c = th_c = self.tr_pitch
            fz, vch = self._vertical(vh, wing=True)
            self.vel_sp = (vn, ve, vch)
            lift = self._lift(self._euler_quat(phi_c, th_c, self.yaw_c), fz, w)
            eta = self.tr_throttle * a.eta_fw_max * clamp(self.t_tr / self.tr_ramp, 0.0, 1.0)
            surf = list(self._surfaces(phi_c, th_c, V)) if V > self.surf_speed else neutral
            a.cmd = surf + [eta] + lift
            return
        # the hover, and the back transition into it
        vcn, vce, vch_ff = self.vel_c
        pitch_up = None
        if mode == "back":
            self.t_tr += dt
            gs = math.hypot(vn, ve)
            if gs < self.tr_back_speed or self.t_tr > self.tr_timeout:
                self.mode = mode = "mc"
                self.pos_c = (x[0] + vn * abs(vn) / (2.0 * self.brake), x[1] + ve * abs(ve) / (2.0 * self.brake), self.h_c)
                self.hold_position, self.vel_c = True, (0.0, 0.0, 0.0)
                vcn = vce = vch_ff = 0.0
            else:
                vc = max(self._v0 - self.tr_decel * self.t_tr, 0.0)
                vcn, vce, vch_ff = vc * math.cos(self._track), vc * math.sin(self._track), 0.0
                al = a.carry_alpha(V)
                if al is not None:
                    pitch_up = max(al - self.tr_margin, 0.0)
        if mode == "mc" and self.hold_position:
            dn, de = self.pos_c[0] - x[0], self.pos_c[1] - x[1]
            k = self.kp_p
            dist, lin = math.hypot(dn, de), self.brake / (k * k)
            if dist > lin:                                # far out: the speed it can still brake from
                k = math.sqrt(2.0 * self.brake * (dist - 0.5 * lin)) / dist
            vcn += k * dn
            vce += k * de
        sp = math.hypot(vcn, vce)
        vmax = self.v_xy if mode == "mc" else max(self._v0, self.v_xy)
        if sp > vmax:
            vcn, vce = vcn * vmax / sp, vce * vmax / sp
        fz, vch = self._vertical(vh, vch_ff, wing=mode == "back")
        self.vel_sp = (vcn, vce, vch)
        an, ae = self._horizontal(vn, ve, vcn, vce, fz, pitch_up)
        # the weathervane: the yaw setpoint turns toward the side the velocity loop's integral
        # leans it to (the wind it has learned, not a manoeuvre's lean)
        if mode == "mc" and self.yaw_free:
            lean = math.atan2(-self.i_v[0] * math.sin(x[11]) + self.i_v[1] * math.cos(x[11]), a.g)
            if abs(lean) > self.wv_min:
                self.yaw_c = wrap(self.yaw_c + clamp(self.wv_gain * lean, -self.wv_rate, self.wv_rate) * dt)
        qd, (phi_c, th_c) = self._thrust_attitude(an, ae, fz)
        self.phi_c, self.theta_c = phi_c, th_c
        lift = self._lift(qd, fz)
        surf = list(self._surfaces(phi_c, th_c, V)) if mode == "back" and V > self.surf_speed else neutral
        a.cmd = surf + [0.0] + lift

    def status(self):
        if self.mode == "fw":
            return (f"fw     chi_c {math.degrees(self.chi_c):6.1f}  phi_c {math.degrees(self.phi_c):+5.1f}  "
                    f"theta_c {math.degrees(self.theta_c):+5.1f}  h_c {self.h_c:6.1f}  V_c {self.va_c:4.1f}")
        v = self.vel_sp
        return (f"{self.mode:<6} p_c ({self.pos_c[0]:7.1f} {self.pos_c[1]:7.1f} {self.pos_c[2]:6.1f})  "
                f"v_c ({v[0]:+5.1f} {v[1]:+5.1f} {v[2]:+5.1f})  yaw_c {math.degrees(self.yaw_c):6.1f}  "
                f"tilt_c {math.degrees(self.tilt_c):4.1f}  w_mc {self.w_mc:4.2f}  "
                f"mixer {('free', 'yaw', 'collective', 'roll/pitch')[self.saturated]}")


# --------------------------------------------------------------------------- #
#  The mission: OURS
# --------------------------------------------------------------------------- #
class Mission:
    """A list of legs, each a dict:

        dict(kind="takeoff", height=m above where it stands, hold=s)
        dict(kind="hover", pos=(n, e, h), yaw=deg or None, hold=s)
        dict(kind="cruise", route=[(n, e), ...], altitude=m, airspeed=m/s)
        dict(kind="land", pos=(n, e))

    A take-off spools the lift rotors at idle and climbs straight up. A hover leg flies to its
    point on the lift rotors (yaw None: the weathervane's). A cruise flies its route with
    x8_rig.LOS at its altitude and airspeed, starting from where the aircraft is; hovering, it
    first makes the front transition along the route's first leg, level at the height it has:
    the altitude is the wing's to reach, at the wing's own climb rate. A landing on the wing flies
    straight at its point, starts the back transition ``back_distance_m`` short of it, hovers
    the rest, comes down and stops the motors on the ground; hovering, it goes straight there.

    update(aircraft, autopilot) once per step, before the autopilot's. ``done`` when the last
    leg is; ``phase`` names what it is doing; ``los`` is the route's guidance while there is one."""

    def __init__(self, legs, spec=None):
        self.legs = [dict(l) for l in legs]
        self.spec = spec
        self.k = -1
        self.done = False
        self.phase = "start"
        self.los = None
        self._t = 0.0
        self._sub = 0

    def _next(self, a, ap):
        self.k += 1
        self._t, self._sub, self.los = 0.0, 0, None
        if self.k >= len(self.legs):
            self.done, self.phase = True, "done"
            return None
        return self.legs[self.k]

    def update(self, a, ap):
        if self.done:
            return
        cfg = (self.spec or a.spec)["control"]["mission"]
        if self.k < 0:
            self._next(a, ap)
        if self.done:
            return
        leg = self.legs[self.k]
        kind = leg.get("kind", "hover")
        dt = a.dt
        self._t += dt
        x = a.x
        if kind == "takeoff":
            if self._sub == 0:
                self.phase = "spool"
                ap.armed, ap.idle = True, True
                if self._t >= cfg["spool_s"]:
                    ap.idle = False
                    ap.reset()
                    ap.mode = "mc"
                    self._h1 = a.h + leg["height"]
                    self._hc = a.h
                    self._pos = (x[0], x[1])
                    self._sub = 1
            elif self._sub == 1:
                self.phase = "climb"
                self._hc = min(self._hc + cfg["climb_ms"] * dt, self._h1)
                ap.command(position=(self._pos[0], self._pos[1], self._hc),
                           velocity=(0.0, 0.0, cfg["climb_ms"] if self._hc < self._h1 else 0.0))
                if self._hc >= self._h1 and abs(a.h - self._h1) < cfg["accept_radius_m"]:
                    self._sub, self._t = 2, 0.0
            elif self._t >= leg.get("hold", 0.0):
                self._next(a, ap)
        elif kind == "hover":
            self.phase = "hover"
            if self._sub == 0:
                yaw = leg.get("yaw")
                ap.command(position=leg["pos"], velocity=None, yaw=None if yaw is None else math.radians(yaw))
                self._sub = 1
            vg = a.ground_velocity()
            d = math.hypot(leg["pos"][0] - x[0], leg["pos"][1] - x[1])
            if self._sub == 1 and d < cfg["accept_radius_m"] and abs(leg["pos"][2] - a.h) < cfg["accept_radius_m"] \
                    and float(np.linalg.norm(vg)) < cfg["accept_speed_ms"]:
                self._sub, self._t = 2, 0.0
            if self._sub == 2 and self._t >= leg.get("hold", 0.0):
                self._next(a, ap)
        elif kind == "cruise":
            if self._sub == 0:
                self.los = LOS([(x[0], x[1])] + [tuple(p) for p in leg["route"]], spec=self.spec or a.spec)
                ap.command(airspeed=leg.get("airspeed"))
                if ap.mode != "fw":
                    (an, ae), (bn, be) = self.los.leg(0)
                    ap.transition("fw", course=math.atan2(be - ae, bn - an))
                self._sub = 1
            vg = a.ground_velocity()
            chi = self.los.update(x[0], x[1], math.hypot(vg[0], vg[1]) if ap.mode == "fw" else None)
            if ap.mode == "fw":
                self.phase = "cruise"
                ap.command(course=chi, altitude=leg.get("altitude"))
            else:
                self.phase = "front transition" if ap.mode == "front" else "transition aborted"
            if self.los.done:
                self._next(a, ap)
        elif kind == "land":
            pn, pe = leg["pos"]
            if self._sub == 0:
                if ap.mode == "fw":
                    self.los = LOS([(x[0], x[1]), (pn, pe)], spec=self.spec or a.spec)
                    self._sub = 1
                else:
                    self._sub = 2
            if self._sub == 1:
                self.phase = "approach"
                vg = a.ground_velocity()
                ap.command(course=self.los.update(x[0], x[1], math.hypot(vg[0], vg[1])))
                if math.hypot(pn - x[0], pe - x[1]) <= cfg["back_distance_m"]:
                    ap.transition("mc")
                    self._sub = 2
            elif self._sub == 2:
                self.phase = "back transition" if ap.mode == "back" else "hover to the pad"
                if ap.mode == "mc":
                    ap.command(position=(pn, pe, ap.h_c), velocity=None, yaw=None)
                    vg = a.ground_velocity()
                    if math.hypot(pn - x[0], pe - x[1]) < cfg["accept_radius_m"] \
                            and float(np.linalg.norm(vg)) < cfg["accept_speed_ms"]:
                        self._hc = a.h
                        self._sub = 3
            elif self._sub == 3:
                self.phase = "descend"
                slow = a.height_above_ground() < cfg["land_slow_height_m"]
                rate = cfg["land_slow_rate_ms"] if slow else cfg["land_rate_ms"]
                self._hc -= rate * dt
                ap.command(position=(pn, pe, self._hc), velocity=(0.0, 0.0, -rate))
                if a.on_ground:
                    ap.idle = True
                    self._sub, self._t = 4, 0.0
            elif self._sub == 4:
                self.phase = "landed"
                if self._t >= cfg["land_settle_s"]:
                    ap.armed = False
                    self._next(a, ap)


# --------------------------------------------------------------------------- #
#  The pilot: OURS
# --------------------------------------------------------------------------- #
class Pilot:
    """A pilot's four sticks on the autopilot's setpoints, the same four hovering and on the
    wing (OURS; the assisted modes of a VTOL's autopilot are what it is modelled on). Each
    is -1..1:

                  hovering                                 on the wing
        forward   the speed along the nose (back: slower)  moves the airspeed setpoint
        turn      the yaw rate, to the right               the bank; released: wings level,
                                                           then the course it has
        side      the speed to the right                   -
        up        climb and descent on the lift rotors     moves the height setpoint: up at
                                                           what the pusher can hold, or down

    Sticks at rest, the hover brakes and holds its place, its height and its heading, and
    once it has stood still for some seconds the autopilot's weathervane has the nose (the
    lift rotors cannot hold the tail across a wind; ``weathervane`` False: the heading it
    has, as far as they can); the wing holds its course, its height and its airspeed. The wing's airspeed setpoint stays at or above the transition's speed, and under
    it the bank the turn stick gets shrinks to nothing: a level turn needs 1 / cos(bank) of
    the lift, and near the slowest steady flight the elevator has no more to give.

    On the ground, up spools the lift rotors and lifts off; down held on the ground stops
    them. toggle() starts the front transition from a hover and the back transition from the
    wing, or turns a transition around; the transitions themselves are the autopilot's.
    fly(legs) hands the aircraft to a Mission until it is done or a stick moves. On the wing,
    touching the ground stops every motor (``state`` "down": it is over).

    sticks(...) or keys(keyboard), then update() once per step before the autopilot's.
    ``state`` is "ground", "spool", "fly", "auto" (a Mission has it), "landed" or "down";
    ``phase`` says it in words."""

    AXES = ("forward", "turn", "side", "up")
    KEYS = {"forward": (("W", "UP"), ("S", "DOWN")), "turn": (("D", "RIGHT"), ("A", "LEFT")),
            "side": (("E",), ("Q",)), "up": (("SPACE",), ("SHIFT",))}

    def __init__(self, aircraft, autopilot, spec=None, weathervane=True):
        self.a, self.ap = aircraft, autopilot
        self.spec = spec or aircraft.spec
        self.cfg = self.spec["control"]["pilot"]
        self.weathervane = weathervane
        self.raw = [0.0, 0.0, 0.0, 0.0]
        self.s = [0.0, 0.0, 0.0, 0.0]
        self.mission = None
        self.state = "ground" if aircraft.on_ground or not autopilot.armed else "fly"
        self.phase = "on the ground"
        self._t = self._down = self._still = 0.0
        self._mode = None
        self._hc = aircraft.h
        self._va = autopilot.va_c
        self._held = self._level = self._yawing = False
        self._v = [0.0, 0.0]
        self._rate = 0.0

    # ---- what the pilot does
    def sticks(self, forward=0.0, turn=0.0, side=0.0, up=0.0):
        self.raw = [clamp(float(v), -1.0, 1.0) for v in (forward, turn, side, up)]

    def keys(self, keyboard):
        """The sticks from a keyboard, anything with is_key_down(name) (a tp.Canvas): W and S
        or the up and down arrows are forward, D and A or right and left the turn, E and Q the
        side, SPACE and SHIFT up and down. A key is all or nothing: update() eases it in."""
        down = keyboard.is_key_down
        self.raw = [(1.0 if any(down(k) for k in plus) else 0.0) - (1.0 if any(down(k) for k in minus) else 0.0)
                    for plus, minus in (self.KEYS[n] for n in self.AXES)]

    def toggle(self):
        """Hover to wing, wing to hover; in a transition, back the way it came. False where
        there is nothing to toggle (on the ground, or a Mission has the aircraft)."""
        if self.state != "fly" or self.mission is not None:
            return False
        self.ap.transition("fw" if self.ap.mode in ("mc", "back") else "mc")
        return True

    def fly(self, legs):
        """Hand the aircraft to a Mission of these legs (in the air already, without their
        take-offs; a Mission that has it is replaced). False where it cannot: spooling, just
        landed, down, or a Mission of its own still on the ground."""
        ap = self.ap
        if ap.armed and not ap.idle and not self.a.on_ground and self.state in ("fly", "auto"):
            legs = [leg for leg in legs if leg.get("kind") != "takeoff"]
        elif self.state != "ground":
            return False
        if not legs:
            return False
        self.mission, self.state = Mission(legs, self.spec), "auto"
        return True

    # ---- one update
    def update(self):
        a, ap, dt, c = self.a, self.ap, self.a.dt, self.cfg
        k = min(1.0, dt / c["stick_time_s"])
        self.s = [s + (r - s) * k for s, r in zip(self.s, self.raw)]
        m = self.spec["control"]["mission"]
        if self.mission is not None:
            if any(self.raw) and ap.armed and not ap.idle:              # a stick moved: the pilot has it
                self.mission, self.state, self._mode = None, "fly", None
            else:
                self.mission.update(a, ap)
                self.phase = "auto: " + self.mission.phase
                if self.mission.done:
                    self.mission, self._mode = None, None
                    self.state = "fly" if ap.armed else "ground"
                return
        if self.state == "ground":
            self.phase = "on the ground"
            ap.armed = False
            if self.raw[3] > 0.0:
                ap.armed = ap.idle = True
                self.state, self._t = "spool", 0.0
            return
        if self.state in ("spool", "landed"):
            self._t += dt
            spool = self.state == "spool"
            self.phase = "spooling" if spool else "landed"
            if (spool and self._t >= m["spool_s"]) or (not spool and self.raw[3] > 0.0):
                ap.idle = False
                ap.reset()
                ap.mode = "mc"
                self.state, self._mode, self._down = "fly", None, 0.0
            elif not spool and self._t >= m["land_settle_s"]:
                ap.armed = False
                self.state = "ground"
            return
        if self.state == "down":
            self.phase = "down"
            ap.armed = False
            return
        mode = ap.mode
        if mode != self._mode:                                          # the autopilot changed halves, or was taken over
            self._mode = mode
            lead = c["wing_height_lead_m"] if mode in ("fw", "front") else c["height_lead_m"]
            self._hc = clamp(ap.h_c if mode in ("fw", "front") else ap.pos_c[2], a.h - lead, a.h + lead)
            self._va, self._level, self._held, self._yawing = ap.va_c, ap.hold == "roll", False, False
            self._rate = self._still = 0.0
        if mode == "mc":
            self._hover()
        elif mode == "fw":
            self._wing()
        else:
            self.phase = "front transition" if mode == "front" else "back transition"
        if mode in ("fw", "front") and a.on_ground:
            ap.armed = False
            self.state = "down"

    def _hover(self):
        a, ap, dt, c = self.a, self.ap, self.a.dt, self.cfg
        m = self.spec["control"]["mission"]
        fwd, turn, side, up = self.s
        x = a.x
        psi = x[11]
        if self.raw[1] != 0.0 or abs(turn) > 0.02:
            # the yaw setpoint goes round at the stick's rate, never far ahead of the nose
            r = turn * ap.yaw_rate_max
            lead = math.radians(c["yaw_lead_deg"])
            ap.command(yaw=wrap(psi + clamp(wrap(ap.yaw_c + r * dt - psi), -lead, lead)), yaw_rate=r)
            self._yawing = True
        elif self._yawing:                                              # let go: the setpoint waits for the nose to stop
            ap.command(yaw=psi, yaw_rate=0.0)
            self._yawing = abs(x[8]) > math.radians(c["yaw_stop_dps"])
        else:
            vane = self.weathervane and self._still >= c["weathervane_after_s"]
            ap.command(yaw=None if vane else ap.yaw_c, yaw_rate=0.0)
        vz = up * (ap.v_up if up > 0.0 else ap.v_down)
        if vz < 0.0 and a.height_above_ground() < m["land_slow_height_m"]:
            vz = max(vz, -m["land_slow_rate_ms"])
        lead = c["height_lead_m"]
        self._hc = clamp(self._hc + vz * dt, a.h - lead, a.h + lead)
        if self.raw[0] != 0.0 or self.raw[2] != 0.0 or abs(fwd) > 0.02 or abs(side) > 0.02:
            cy, sy = math.cos(psi), math.sin(psi)
            if self._held or ap.hold_position:                          # from the speed it has, along and across the nose
                g = a.ground_velocity()
                self._v = [g[0] * cy + g[1] * sy, -g[0] * sy + g[1] * cy]
            dv = c["hover_accel_ms2"] * dt
            self._v[0] += clamp(fwd * (ap.v_xy if fwd > 0.0 else c["back_ms"]) - self._v[0], -dv, dv)
            self._v[1] += clamp(side * c["side_ms"] - self._v[1], -dv, dv)
            vf, vr = self._v
            ap.command(position=None, velocity=(vf * cy - vr * sy, vf * sy + vr * cy, vz), altitude=self._hc)
            self._held = False
            self.phase = "hover"
        else:
            if not self._held:                                          # where it can stop from this speed
                g, b = a.ground_velocity(), ap.brake
                ap.command(position=(x[0] + g[0] * abs(g[0]) / (2.0 * b), x[1] + g[1] * abs(g[1]) / (2.0 * b), self._hc))
                self._held = True
            ap.command(velocity=(0.0, 0.0, vz), altitude=self._hc)
            self.phase = "hover, holding"
        g = a.ground_velocity()
        still = self._held and not self._yawing and self.raw[1] == 0.0 and math.hypot(g[0], g[1]) < c["weathervane_speed_ms"]
        self._still = self._still + dt if still else 0.0
        if a.on_ground and self.raw[3] < 0.0:
            self._down += dt
            if self._down >= c["land_hold_s"]:
                ap.idle = True
                self.state, self._t = "landed", 0.0
        else:
            self._down = 0.0

    def _wing(self):
        a, ap, dt, c = self.a, self.ap, self.a.dt, self.cfg
        fwd, turn, _, up = self.s
        V = a.airdata()[0]
        if self.raw[1] != 0.0 or abs(turn) > 0.02:
            v0 = c["bank_none_ms"]
            ap.command(roll=turn * ap.phi_max * clamp((V - v0) / (ap.tr_speed - v0), 0.0, 1.0))
            self._level = True
        elif self._level:
            ap.command(roll=0.0)
            if abs(a.x[9]) < math.radians(c["level_deg"]):
                ap.command(course=a.course())
                self._level = False
        rate = up * (ap.fw_cfg["altitude"]["climb_fraction"] * ap.climb_max * V if up > 0.0 else c["wing_descent_ms"])
        dr = c["wing_accel_ms2"] * dt                                   # the height rate is eased: a wing levels off, it does not stop
        self._rate += clamp(rate - self._rate, -dr, dr)
        lead = c["wing_height_lead_m"]
        self._hc = clamp(self._hc + self._rate * dt, a.h - lead, a.h + lead)
        ap.command(altitude=self._hc)
        lo, hi = max(ap.va_range[0], ap.tr_speed), ap.va_range[1]
        self._va = clamp(self._va + fwd * c["airspeed_rate_ms2"] * dt, lo, hi)
        step = c["airspeed_step_ms"]
        va = clamp(round(self._va / step) * step, lo, hi)
        if abs(va - ap.va_c) > 1e-9:
            ap.command(airspeed=va)                                     # the wing's gains are designed anew at it
        self.phase = "wing"


# --------------------------------------------------------------------------- #
#  The 3D model
# --------------------------------------------------------------------------- #
class Visual:
    """babyshark.glb's nodes posed from a Babyshark.

    root    the loaded model's scene node (tp.GLTFLoader().load(model_path()).scene); anything
            with position.set, quaternion.set and get_object_by_name whose nodes have rotation

    pose(aircraft, h0) places the root (the .glb's origin is the centre of gravity) from the
    state's position and attitude (world = (E, h0 - D, -N)) and turns

      * ``aileron_left``, ``aileron_right`` about their hinges (local +Z, positive trailing
        edge down) by +delta_a and -delta_a: a positive aileron rolls right wing down;
      * ``ruddervator_left``, ``ruddervator_right`` about theirs (local +Z, positive trailing
        edge toward the panel's lower, inner face) by (delta_e - delta_r) / 2 and (delta_e +
        delta_r) / 2: the halves of the thesis's mixing (3.40), with the rudder's sense the
        one an inverted V needs (a positive rudder puts both trailing edges to port, which
        yaws the nose to port, the thesis's negative yawing moment);
      * ``lift_fr``, ``lift_fl``, ``lift_rl``, ``lift_rr`` about local +Y by each rotor's
        angle times its spin sense (+1 counter-clockwise seen from above);
      * ``pusher`` about local +X by its angle (its spin vector along body +x)."""

    SURFACES = ("aileron_left", "aileron_right", "ruddervator_left", "ruddervator_right")

    def __init__(self, root):
        self.root = root
        names = self.SURFACES + ("pusher",) + tuple("lift_" + n for n in ("fr", "rl", "fl", "rr"))
        self.nodes = {n: root.get_object_by_name(n) for n in names}
        missing = [k for k, v in self.nodes.items() if v is None]
        if missing:
            raise KeyError(f"babyshark.glb has no node(s) {missing}: rebuild it with uav/build_babyshark_blender.py")

    def pose(self, a, h0=0.0):
        x, N = a.x, self.nodes
        self.root.position.set(*(float(v) for v in a.world_position(h0)))
        self.root.quaternion.set(*(float(v) for v in quat_of(a.world_rotation())))
        N["aileron_left"].rotation.z = x[12]
        N["aileron_right"].rotation.z = -x[12]
        N["ruddervator_left"].rotation.z = 0.5 * (x[13] - x[14])
        N["ruddervator_right"].rotation.z = 0.5 * (x[13] + x[14])
        N["pusher"].rotation.x = a.pusher_angle
        for name, angle, spin in zip(a.rotor_names, a.rotor_angle, a.spin):
            N["lift_" + name].rotation.y = angle * spin
