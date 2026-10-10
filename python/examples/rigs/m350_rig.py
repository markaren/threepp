"""
m350_rig -- the DJI Matrice 350 RTK as a flown rigid body in wind: a flight model built from
the product page's figures, a wind with turbulence, an autopilot, a mission, a gimbal, and the
3D model posed from the state.

Six things, and nothing scene-specific:

  * ``M350``      -- a rigid body with four rotors. Mass, centre of gravity and inertia from
                     the spec's mass budget (uav/m350_spec.json); per rotor a static thrust
                     kT w^2 less the blade-element and momentum-theory loss to the air's speed
                     through the disc, a linear in-plane (H) force (P. Martin and E. Salaun,
                     "The true role of accelerometer feedback in quadrotor control", ICRA 2010;
                     M. Faessler, A. Franchi and D. Scaramuzza, "Differential flatness of
                     quadrotor dynamics subject to rotor drag for accurate tracking of
                     high-speed trajectories", IEEE RA-L 3(2), 2018), the shaft
                     torque from a figure of merit, a first-order rotor speed, the rotors'
                     angular momentum in the rigid-body equation; a bluff-body drag at a centre
                     of pressure; four skid contact points on a ground you hand in; the battery.
                     Fixed-step fourth-order Runge-Kutta.
  * ``Wind``      -- a mean wind (a logarithmic profile over the ground, or a field you hand
                     in) and the Dryden turbulence of MIL-F-8785C / MIL-HDBK-1797 in its
                     low-altitude form, as shaping filters on seeded white noise.
  * ``Autopilot`` -- OURS, not DJI's: the cascade multirotor autopilots are built as (position
                     P, velocity PI, thrust vector with a tilt limit, quaternion attitude P,
                     body-rate PID, mixer). It does not know the wind: it rejects it by
                     feedback. Its state feedback is EXACT: no estimator, no sensor noise.
  * ``Mission``   -- OURS: straight legs between waypoints on a jerk-limited S-curve, the yaw
                     setpoint inside a rate and an acceleration, take-off and landing.
  * ``Gimbal``    -- pan, roll and tilt toward a world target inside the payload's ranges.
  * ``Visual``    -- the m350.glb posed from an M350 and a Gimbal.

WHAT IS WHOSE
-------------
The product pages' (m350_spec.json "aircraft", "payload"): the masses, the wheelbase, the
battery, the limits (tilt, rates, speeds, wind, flight time, hover accuracy), the gimbal's ranges.

ASSUMED (the spec says why beside each): the split of the mass and where it sits, the propeller's
thrust coefficient, figure of merit, chord and lift slope, the drive's efficiency, the motor's lag
and thrust limit, the airframe's drag coefficient and areas, the skid contact points, the ground's
stiffness and friction, the terrain's roughness length, the gimbal's slew rate. DJI publishes no
thrust, torque, drag or controller figure: nothing here is validated against a flight log.

CALIBRATED, which is not validated: the figure of merit and the drive efficiency are chosen
together so the hover draws what the product page's 55 min implies (``M350.hover_power``), and
the rotors' in-plane drag kh is solved so 23 m/s needs 30 deg of tilt, the product page's point
(``M350.calibrate``). Both reproduce the product page by construction.

OURS: the model's structure, the autopilot, the mission, the turbulence's convection speed at a
hover (the speed of the air past the aircraft, where the standard has an airspeed).

Nothing here imports threepp: ``Visual`` is handed the loaded model's root node.

FRAMES
------
As x8_rig: NED position (N, E, D), body axes x forward, y right, z down, the attitude a unit
quaternion (w, x, y, z), body to NED, renormalised each step. One mapping to threepp's world
(x east, y up, z south), at the boundary::

    world = (E, h0 - D, -N)            h0: the height of the NED origin in the world

The .glb's and the spec's frame is X forward, Y up, Z right: a body vector (x, y, z) is
(x, -z, y) in it. The .glb's origin is NOT the centre of gravity: ``Visual`` places the root so
the centre of gravity is where the state says.

Heights (``h``, the ground's, a waypoint's) are metres above the NED origin; ``h0`` is that
origin's height above mean sea level, which only the air density reads.

Wind is the velocity of the air over the ground, NED, m/s; ``Wind(speed, from_deg)`` takes
where it blows FROM, degrees from north, clockwise.

TYPICAL USE
-----------
::

    from m350_rig import M350, Wind, Autopilot, Mission, Gimbal, Visual, model_path

    wind = Wind(8.0, from_deg=250.0, turbulence=1.5, seed=3)
    m = M350(wind=wind)                               # payload on, flat ground at 0
    m.place_on_ground(0.0, 0.0, yaw=0.0)
    ap, gimbal = Autopilot(m), Gimbal(m.spec)
    mission = Mission([dict(kind="takeoff", height=12.0),
                       dict(pos=(40.0, 10.0, 18.0), speed=4.0, yaw="poi", poi=(60.0, 10.0, 15.0), hold=5.0),
                       dict(kind="land")])
    while not mission.done:
        mission.update(m, ap)                         # setpoints for the autopilot
        ap.update()                                   # rotor speed commands
        m.step()                                      # one fixed step, m.dt
        gimbal.update(m, mission.look)
    visual = Visual(tp.GLTFLoader().load(model_path()).scene)
    visual.pose(m, gimbal, h0=0.0)
"""
import json
import math
import os

import numpy as np

UAV_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "uav")
SPEC_PATH = os.path.join(UAV_DIR, "m350_spec.json")
TWO_PI = 2.0 * math.pi
FT = 0.3048
KEEP = object()                      # "leave this setpoint as it is" in Autopilot.command


def load_spec(path=None):
    with open(path or SPEC_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def model_path():
    """The generated m350.glb, or None where it has not been built."""
    p = os.path.join(UAV_DIR, "m350.glb")
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


def body_of(v):
    """A vector of the spec's layout (X forward, Y up, Z right) in body axes (x, y, z)."""
    return (v[0], v[2], -v[1])


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


def quat_wxyz(R):
    """(w, x, y, z), the state's order, of a rotation matrix."""
    x, y, z, w = quat_of(np.asarray(R, float))
    return (float(w), float(x), float(y), float(z))


def rot_of_quat(qw, qx, qy, qz):
    """R_b^n of a quaternion (w, x, y, z)."""
    s = 2.0 / (qw * qw + qx * qx + qy * qy + qz * qz)
    return np.array([[1.0 - s * (qy * qy + qz * qz), s * (qx * qy - qw * qz), s * (qx * qz + qw * qy)],
                     [s * (qx * qy + qw * qz), 1.0 - s * (qx * qx + qz * qz), s * (qy * qz - qw * qx)],
                     [s * (qx * qz - qw * qy), s * (qy * qz + qw * qx), 1.0 - s * (qx * qx + qy * qy)]])


def flat_ground(n, e):
    return 0.0


def _plain(ground):
    """A ground callable whose heights are plain floats: a numpy scalar in the state would slow
    every step after it."""
    if ground is None or ground is flat_ground:
        return flat_ground
    return lambda n, e: float(ground(n, e))


def ground_normal(ground, n, e, eps=0.25):
    """The ground's unit normal at (n, e), NED, pointing out of it (up: its D part negative)."""
    gn = (ground(n + eps, e) - ground(n - eps, e)) / (2.0 * eps)
    ge = (ground(n, e + eps) - ground(n, e - eps)) / (2.0 * eps)
    k = 1.0 / math.sqrt(gn * gn + ge * ge + 1.0)
    return (-gn * k, -ge * k, -k)


# --------------------------------------------------------------------------- #
#  The wind
# --------------------------------------------------------------------------- #
class Wind:
    """A mean wind and Dryden turbulence on it.

    speed, from_deg   the mean wind 10 m above the ground and where it blows FROM (deg from
                      north, clockwise); below and above, the logarithmic profile
                      U(z) = speed ln(z / z0) / ln(10 / z0), z0 the spec's roughness length (ASSUMED)
    ground            ground(n, e) -> height, the same callable the aircraft has (default flat at 0)
    field             field(n, e, h) -> (wn, we, wd), NED m/s: replaces the profile (a scene's
                      terrain-following flow); speed and from_deg are then unused
    turbulence        scales the standard's sigmas (1.0: MIL-HDBK-1797 as written; mountain
                      terrain is rougher, the scene sets it, ASSUMED)
    seed              the white noise's; the same seed gives the same gusts, bit for bit

    The turbulence is the low-altitude Dryden model: with h the height above the ground in feet
    (floor 10 ft, held past 1000 ft where the form ends) and W20 the mean wind 20 ft up,

        L_w = h                      L_u = L_v = h / (0.177 + 0.000823 h)^1.2
        sigma_w = 0.1 W20            sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 h)^0.4

    as shaping filters on unit white noise, V the speed the turbulence is convected past the
    aircraft at:

        longitudinal   H(s) = sigma sqrt(2 V / L) / (s + V / L)                 (an Ornstein-Uhlenbeck process)
        lateral, vertical   H(s) = sigma sqrt(3 V / L) (s + V / (sqrt(3) L)) / (s + V / L)^2

    longitudinal along the local mean wind, lateral across it in the horizontal, vertical the
    third axis. Each filter is advanced with its exact discrete form for the step (the state
    transition and the noise covariance of the frozen coefficients), so the variances are the
    sigmas' at any step and any V / L.

    OURS: the standard takes V as the aircraft's airspeed through frozen turbulence. A hovering
    aircraft has none, and the turbulence is carried past it by the wind: V is the speed of the
    mean air past the aircraft, |v - mean wind|, with a floor of 1 m/s (the spec's).

    advance(dt, n, e, h, v) moves the gusts one step and returns the wind at the aircraft; the
    aircraft calls it once per step and holds the value across the Runge-Kutta stages.
    add_gust(t0, duration, (gn, ge, gd)) lays a one-minus-cosine gust on top."""

    def __init__(self, speed=0.0, from_deg=0.0, ground=None, field=None, turbulence=1.0, seed=1, spec=None):
        cfg = (spec or load_spec())["wind"]
        self.cfg = cfg
        self.speed = float(speed)
        a = math.radians(from_deg)
        self.dir = (-math.cos(a), -math.sin(a))             # the way the air moves, (N, E)
        self.ground = _plain(ground)
        self.field = field
        self.turbulence = float(turbulence)
        self.z0 = cfg["roughness_length_m"]
        self.z_ref = cfg["reference_height_m"]
        self.v_min = cfg["min_convection_speed_ms"]
        self.h_min_ft, self.h_max_ft = cfg["dryden_height_ft"]
        self.h20 = cfg["w20_height_ft"] * FT
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._buf, self._bi = [], 0
        self.z = [0.0, 0.0, 0.0, 0.0, 0.0]                  # unit-variance filter states: u, v1, v2, w1, w2
        self.t = 0.0
        self.gusts = []
        self.mean_ned = (0.0, 0.0, 0.0)
        self.gust_ned = (0.0, 0.0, 0.0)
        self.total = (0.0, 0.0, 0.0)
        self.sigma = (0.0, 0.0, 0.0)
        self.length = (1.0, 1.0, 1.0)
        self.V = self.v_min
        self.uvw = (0.0, 0.0, 0.0)
        # start the filters in their stationary state, not at zero
        self.z = [self._xi() for _ in range(5)]

    def _xi(self):
        if self._bi >= len(self._buf):
            self._buf = self.rng.standard_normal(4096).tolist()
            self._bi = 0
        v = self._buf[self._bi]
        self._bi += 1
        return v

    def profile(self, z):
        """The mean speed z metres above the ground (the logarithmic profile)."""
        if z <= self.z0:
            return 0.0
        return self.speed * math.log(z / self.z0) / math.log(self.z_ref / self.z0)

    def mean(self, n, e, h):
        """The mean wind at a point, NED m/s."""
        if self.field is not None:
            w = self.field(n, e, h)
            return (float(w[0]), float(w[1]), float(w[2]))
        s = self.profile(h - self.ground(n, e))
        return (s * self.dir[0], s * self.dir[1], 0.0)

    def w20(self, n, e):
        """The mean wind speed 20 ft above the ground under (n, e)."""
        if self.field is not None:
            w = self.field(n, e, self.ground(n, e) + self.h20)
            return math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
        return self.profile(self.h20)

    def dryden(self, hg, w20):
        """(sigma_u, sigma_v, sigma_w), (L_u, L_v, L_w) in m/s and m, hg metres above the ground."""
        hf = clamp(hg / FT, self.h_min_ft, self.h_max_ft)
        f = 0.177 + 0.000823 * hf
        sw = 0.1 * w20 * self.turbulence
        su = sw / f ** 0.4
        lw = hf * FT
        lu = lw / f ** 1.2
        return (su, su, sw), (lu, lu, lw)

    def add_gust(self, t0, duration, amplitude):
        """A discrete one-minus-cosine gust: amplitude (NED, m/s) at its peak, t0 + duration / 2."""
        self.gusts.append((float(t0), float(duration), tuple(float(a) for a in amplitude)))

    @staticmethod
    def _second_order(z1, z2, th, x1, x2):
        """One exact step of the lateral / vertical filter in unit-variance states: transition
        e^-th [[1 + th, th], [-th, 1 - th]], noise covariance I - Phi Phi^T. th = V dt / L."""
        ph = math.exp(-th)
        E = ph * ph
        if th < 0.02:                                       # 1 - E (1 + 2 th + 2 th^2) cancels: its series
            zz = 2.0 * th
            q11 = zz * zz * zz / 6.0 * (1.0 - 0.75 * zz + 0.3 * zz * zz - zz * zz * zz / 12.0)
        else:
            q11 = 1.0 - E * (1.0 + 2.0 * th + 2.0 * th * th)
        q12 = 2.0 * th * th * E
        q22 = 1.0 - E * (1.0 - 2.0 * th + 2.0 * th * th)
        l11 = math.sqrt(q11)
        l21 = q12 / l11
        l22 = math.sqrt(max(q22 - l21 * l21, 0.0))
        n1 = ph * ((1.0 + th) * z1 + th * z2) + l11 * x1
        n2 = ph * (-th * z1 + (1.0 - th) * z2) + l21 * x1 + l22 * x2
        return n1, n2

    def advance(self, dt, n, e, h, v=(0.0, 0.0, 0.0)):
        """One step of dt for an aircraft at (n, e, h) moving at v (NED): the wind there."""
        mn, me, md = self.mean(n, e, h)
        (su, sv, sw), (lu, lv, lw) = self.dryden(h - self.ground(n, e), self.w20(n, e))
        rn, re, rd = v[0] - mn, v[1] - me, v[2] - md
        V = math.sqrt(rn * rn + re * re + rd * rd)
        if V < self.v_min:
            V = self.v_min
        z = self.z
        ph = math.exp(-V * dt / lu)
        z[0] = ph * z[0] + math.sqrt(1.0 - ph * ph) * self._xi()
        z[1], z[2] = self._second_order(z[1], z[2], V * dt / lv, self._xi(), self._xi())
        z[3], z[4] = self._second_order(z[3], z[4], V * dt / lw, self._xi(), self._xi())
        gu = su * z[0]
        gv = sv * (0.5 * z[1] + 0.8660254037844386 * z[2])
        gw = sw * (0.5 * z[3] + 0.8660254037844386 * z[4])
        # the gust's axes: along the mean wind, across it in the horizontal, the third
        m = math.sqrt(mn * mn + me * me + md * md)
        mh = math.hypot(mn, me)
        if mh > 1e-6:
            t1 = (mn / m, me / m, md / m)
            t2 = (-me / mh, mn / mh, 0.0)
        else:
            t1, t2 = (self.dir[0], self.dir[1], 0.0), (-self.dir[1], self.dir[0], 0.0)
        t3 = (t1[1] * t2[2] - t1[2] * t2[1], t1[2] * t2[0] - t1[0] * t2[2], t1[0] * t2[1] - t1[1] * t2[0])
        gn = gu * t1[0] + gv * t2[0] + gw * t3[0]
        ge = gu * t1[1] + gv * t2[1] + gw * t3[1]
        gd = gu * t1[2] + gv * t2[2] + gw * t3[2]
        for t0, dur, amp in self.gusts:
            s = (self.t - t0) / dur
            if 0.0 < s < 1.0:
                k = 0.5 * (1.0 - math.cos(TWO_PI * s))
                gn += k * amp[0]
                ge += k * amp[1]
                gd += k * amp[2]
        self.t += dt
        self.mean_ned, self.gust_ned = (mn, me, md), (gn, ge, gd)
        self.sigma, self.length, self.V, self.uvw = (su, sv, sw), (lu, lv, lw), V, (gu, gv, gw)
        self.total = (mn + gn, me + ge, md + gd)
        return self.total


# --------------------------------------------------------------------------- #
#  The aircraft
# --------------------------------------------------------------------------- #
# The integrated state, in this order: NED position and velocity, the quaternion (w, x, y, z)
# body to NED, the body rates, the four rotor speeds (the spec's order: fr, fl, rl, rr).
STATE = ("n", "e", "d", "vn", "ve", "vd", "qw", "qx", "qy", "qz", "p", "q", "r", "w_fr", "w_fl", "w_rl", "w_rr")
ROTORS = ("fr", "fl", "rl", "rr")


class M350:
    """The Matrice 350 RTK as a rigid body with four rotors.

    spec      m350_spec.json as a dict (default: load it); every number is read from it
    dt        the fixed step, s (default 1/300: 60, 50 and 30 frames a second are whole steps)
    payload   True: with the H20T (7.298 kg); False: without (6.47 kg, the product page's point)
    wind      a Wind; or a NED 3-tuple; or a callable (t, n, e, d) -> (wn, we, wd); or None.
              Sampled once per step and held across the Runge-Kutta stages.
    ground    ground(n, e) -> height above the NED origin (default flat at 0)
    h0        the NED origin's height above mean sea level (the air density follows it)
    rho       a fixed air density in place of the ISA's (0 takes the air away)

    Per rotor i at its hub r_i (from the centre of gravity), axis = body up, w_i its speed,
    v_i = v_air + omega x r_i the air-relative velocity of the hub, v_ax its part along the axis
    (positive when the disc moves up through the air), v_perp the rest:

        T_i = kT w_i^2 - kz w_i v_ax   (not below zero)         kT = C_T rho D^4 / (4 pi^2)
        kz  = (rho a sigma A R / 4) / (1 + a sigma / (16 lambda_h))
              sigma = blades chord / (pi R), lambda_h = sqrt(C_T' / 2), C_T' = kT / (rho A R^2)
        H_i = -kh w_i v_perp, at the hub                         kh: calibrate()
        Q_i = kQ w_i^2                                           kQ = kT^1.5 / (sqrt(2 rho A) FM)
        on the airframe about body z: s_i (Q_i + J_r dw_i/dt), s_i = +1 for a rotor turning
        counter-clockwise seen from above; the rotors' angular momentum -J_r sum(s_i w_i) along
        body z is in the rigid-body equation (the gyroscopic term)
        dw_i/dt = (command - w_i) / tau, the command held between zero and the speed of the
        spec's maximum thrust (``w_max``); idle (``w_idle``) is the spec's fraction of the hover
        speed at the maximum take-off weight, and the autopilot's floor in flight

    The airframe's drag is -0.5 rho Cd |v| (A_front v_x, A_side v_y, A_top v_z) in body axes at
    the spec's centre of pressure. All four rotor coefficients scale with the air density.

    ``x`` is the state (STATE order), ``cmd`` the four rotor speed commands (rad/s) the next
    step() uses, ``out`` what the last step evaluated at its start (thrusts, powers, air data,
    the ground's force)."""

    def __init__(self, spec=None, dt=1.0 / 300.0, payload=True, wind=None, ground=None, h0=0.0, rho=None,
                 calibrate=True):
        self.spec = spec = load_spec() if spec is None else spec
        AC, PU, AE, AT, GR = spec["aircraft"], spec["propulsion"], spec["aero"], spec["atmosphere"], spec["ground"]
        L = spec["layout"]
        self.payload = payload
        self.dt = dt
        self.g = AT["g"]
        self.rho0 = AT["rho_sea_level"]
        self.rho_fixed = rho
        self.h0 = h0
        self.wind = wind
        self.ground = _plain(ground)
        # ---- mass, centre of gravity, inertia: the budget's boxes, parallel-axis to the CG
        items = [(k, v) for k, v in spec["mass_budget"].items() if k != "note" and (payload or k != "payload")]
        m = sum(v["mass"] for _, v in items)
        cg = np.zeros(3)
        for _, v in items:
            cg += v["mass"] * np.array(body_of(v["at"]))
        cg /= m
        I = np.zeros((3, 3))
        for _, v in items:
            bx, by, bz = v["box"][0], v["box"][2], v["box"][1]          # the box's sides along body x, y, z
            mi = v["mass"]
            I += np.diag([mi / 12.0 * (by * by + bz * bz), mi / 12.0 * (bx * bx + bz * bz), mi / 12.0 * (bx * bx + by * by)])
            r = np.array(body_of(v["at"])) - cg
            I += mi * (float(r @ r) * np.eye(3) - np.outer(r, r))
        self.m, self.cg_body, self.I = float(m), cg, I
        self.cg_model = (float(cg[0]), float(-cg[2]), float(cg[1]))     # in the .glb's frame
        self._I = tuple(float(a) for a in I.ravel())
        self._Ii = tuple(float(a) for a in np.linalg.inv(I).ravel())
        # ---- rotors
        P = AC["propeller"]
        self.D = P["diameter_m"]
        self.R = R = 0.5 * self.D
        self.A = A = math.pi * R * R
        self.Jr = PU["rotor_inertia_kgm2"]
        self.tau = PU["motor_time_constant_s"]
        self.fm, self.eta, self.avionics = PU["figure_of_merit"], PU["drive_efficiency"], PU["avionics_w"]
        self.kT_rho = PU["thrust_coefficient"] * self.D ** 4 / (4.0 * math.pi * math.pi)      # kT / rho
        solidity = P["blades"] * PU["blade_chord_m"] / (math.pi * R)
        lam_h = math.sqrt(0.5 * self.kT_rho / (A * R * R))
        a_s = PU["lift_slope"] * solidity
        self.kz_rho = (a_s * A * R / 4.0) / (1.0 + a_s / (16.0 * lam_h))                      # kz / rho
        self.kQ_rho = self.kT_rho ** 1.5 / (math.sqrt(2.0 * A) * self.fm)                     # kQ / rho
        self.kh_rho = 0.0                                                                     # kh / rho: calibrate()
        self.solidity, self.lambda_h = solidity, lam_h
        self._rot = []
        for k in ROTORS:
            r = np.array(body_of(L["rotors"][k]["pos"])) - cg
            self._rot.append((float(r[0]), float(r[1]), float(r[2]), float(L["rotors"][k]["spin"])))
        self.spin = tuple(r[3] for r in self._rot)
        kT0 = self.kT_rho * self.rho0
        self.w_max = math.sqrt(PU["max_thrust_n"] / kT0)
        # idle: a fraction of the hover speed at the maximum take-off weight, at sea level (OURS:
        # the spec gives the fraction, this is what it is taken of)
        self.w_hover_mtow = math.sqrt(AC["max_takeoff_kg"] * self.g / (4.0 * kT0))
        self.w_idle = PU["idle_speed_fraction"] * self.w_hover_mtow
        # ---- airframe drag
        ar = AE["projected_area_m2"]
        self.cd = AE["drag_coefficient"]
        self._cda = (self.cd * ar["front"], self.cd * ar["side"], self.cd * ar["top"])
        cp = np.array(body_of(AE["centre_of_pressure"])) - cg
        self._cp = (float(cp[0]), float(cp[1]), float(cp[2]))
        # ---- ground contact: the four ends of the two skids, where the model's generator has
        # published them (model.gear: the foot bar's centre and the skid's length), else the
        # spec's ASSUMED ones; their underside is the layout's skid_bottom_y_m
        gear = spec.get("model", {}).get("gear", {})
        if "foot" in gear and "length" in gear.get("skid", {}):
            sx, sz = 0.5 * gear["skid"]["length"], abs(gear["foot"][2])
        else:
            sx, sz = GR["skid_half_length_m"], GR["skid_z_m"]
        sy = L["skid_bottom_y_m"]
        self.skid_points = [(px, sy, pz) for px, pz in ((sx, sz), (sx, -sz), (-sx, -sz), (-sx, sz))]   # layout frame
        self._skid = []
        for px, pz in ((sx, sz), (sx, -sz), (-sx, -sz), (-sx, sz)):
            r = np.array(body_of((px, sy, pz))) - cg
            self._skid.append((float(r[0]), float(r[1]), float(r[2])))
        self._gk, self._gc = GR["stiffness_n_m"], GR["damping_ns_m"]
        self._gkt, self._gct, self._mu = GR["tangential_stiffness_n_m"], GR["tangential_damping_ns_m"], GR["friction"]
        self._gcheck = GR["check_height_m"]
        self._anchor = [None, None, None, None]
        self._near = False
        self._gnorm = (0.0, 0.0, -1.0)
        # ---- battery
        B = AC["battery"]
        self.battery_wh = B["count"] * B["energy_wh"]
        self.battery_j = self.battery_wh * 3600.0
        # ---- state
        self.x = [0.0] * len(STATE)
        self.x[6] = 1.0
        self.cmd = [0.0, 0.0, 0.0, 0.0]
        self.rotor_angle = [0.0, 0.0, 0.0, 0.0]       # each rotor's own angle (its speed integrated), for Visual
        self.t = 0.0
        self.steps = 0
        self._w = (0.0, 0.0, 0.0)
        self.out = {}
        self._air(self.rho_at(0.0))
        self.calibration = None
        if calibrate:
            self.calibrate()

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
        self._kT, self._kz, self._kh, self._kQ = (self.kT_rho * rho, self.kz_rho * rho, self.kh_rho * rho,
                                                  self.kQ_rho * rho)
        self._dk = -0.5 * rho

    def wind_ned(self):
        """The wind the last step used at the aircraft, NED m/s."""
        return self._w

    def _sample_wind(self):
        w = self.wind
        if w is None:
            return (0.0, 0.0, 0.0)
        x = self.x
        if isinstance(w, Wind):
            return w.advance(self.dt, x[0], x[1], -x[2], (x[3], x[4], x[5]))
        if callable(w):
            w = w(self.t, x[0], x[1], x[2])
        return (float(w[0]), float(w[1]), float(w[2]))

    # ---- the equations
    def rates(self, x, cmd, rec=None):
        """d/dt of the state x under the rotor speed commands cmd, in the held wind and air
        density. With rec (a dict) it also writes what it evaluated."""
        n, e, d, vn, ve, vd, qw, qx, qy, qz, p, q, r, w0, w1, w2, w3 = x
        s = 2.0 / (qw * qw + qx * qx + qy * qy + qz * qz)
        r00, r01, r02 = 1.0 - s * (qy * qy + qz * qz), s * (qx * qy - qw * qz), s * (qx * qz + qw * qy)
        r10, r11, r12 = s * (qx * qy + qw * qz), 1.0 - s * (qx * qx + qz * qz), s * (qy * qz - qw * qx)
        r20, r21, r22 = s * (qx * qz - qw * qy), s * (qy * qz + qw * qx), 1.0 - s * (qx * qx + qy * qy)
        # air-relative velocity in body axes
        wn, we, wd = self._w
        an, ae, ad = vn - wn, ve - we, vd - wd
        ub = r00 * an + r10 * ae + r20 * ad
        vb = r01 * an + r11 * ae + r21 * ad
        wb = r02 * an + r12 * ae + r22 * ad
        # the airframe's drag, at the centre of pressure
        V = math.sqrt(ub * ub + vb * vb + wb * wb)
        k = self._dk * V
        ca, cs, ct = self._cda
        Fx, Fy, Fz = k * ca * ub, k * cs * vb, k * ct * wb
        cx, cy, cz = self._cp
        Mx, My, Mz = cy * Fz - cz * Fy, cz * Fx - cx * Fz, cx * Fy - cy * Fx
        # the rotors
        kT, kz, kh, kQ, Jr, itau = self._kT, self._kz, self._kh, self._kQ, self.Jr, 1.0 / self.tau
        Q = hs = 0.0
        wdot = []
        thrust = [] if rec is not None else None
        for (rx, ry, rz, sp), w, c in zip(self._rot, (w0, w1, w2, w3), cmd):
            vix = ub + q * rz - r * ry
            viy = vb + r * rx - p * rz
            viz = wb + p * ry - q * rx                 # v_ax = -viz
            T = kT * w * w + kz * w * viz
            if T < 0.0:
                T = 0.0
            hx, hy = -kh * w * vix, -kh * w * viy
            Fx += hx
            Fy += hy
            Fz -= T
            Mx -= ry * T + rz * hy
            My += rz * hx + rx * T
            Mz += rx * hy - ry * hx
            wd_i = (c - w) * itau
            Q += sp * (kQ * w * w + Jr * wd_i)
            hs += sp * w
            wdot.append(wd_i)
            if thrust is not None:
                thrust.append(T)
        Mz += Q
        fn = r00 * Fx + r01 * Fy + r02 * Fz
        fe = r10 * Fx + r11 * Fy + r12 * Fz
        fd = r20 * Fx + r21 * Fy + r22 * Fz
        # the ground: a spring and damper along its normal, a bristle with Coulomb's limit across it
        gforce = 0.0
        if self._near:
            gnd, (nn, ne, nd) = self.ground, self._gnorm
            gk, gc, gkt, gct, mu = self._gk, self._gc, self._gkt, self._gct, self._mu
            for i, (cx, cy, cz) in enumerate(self._skid):
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
                    dn, de, dd = pn - a[0], pe - a[1], pd - a[2]
                    dot = dn * nn + de * ne + dd * nd
                    tn -= gkt * (dn - dot * nn)
                    te -= gkt * (de - dot * ne)
                    td -= gkt * (dd - dot * nd)
                ft = math.sqrt(tn * tn + te * te + td * td)
                if ft > mu * Fn:
                    kf = mu * Fn / ft
                    tn, te, td = tn * kf, te * kf, td * kf
                gn_, ge_, gd_ = Fn * nn + tn, Fn * ne + te, Fn * nd + td
                fn += gn_
                fe += ge_
                fd += gd_
                bx = r00 * gn_ + r10 * ge_ + r20 * gd_
                by = r01 * gn_ + r11 * ge_ + r21 * gd_
                bz = r02 * gn_ + r12 * ge_ + r22 * gd_
                Mx += cy * bz - cz * by
                My += cz * bx - cx * bz
                Mz += cx * by - cy * bx
                gforce += Fn
        # the rigid body: I w' = M - w x (I w + h_rotors)
        I = self._I
        Hx = I[0] * p + I[1] * q + I[2] * r
        Hy = I[3] * p + I[4] * q + I[5] * r
        Hz = I[6] * p + I[7] * q + I[8] * r - Jr * hs
        Lx, Ly, Lz = Mx - (q * Hz - r * Hy), My - (r * Hx - p * Hz), Mz - (p * Hy - q * Hx)
        Ii = self._Ii
        pd_ = Ii[0] * Lx + Ii[1] * Ly + Ii[2] * Lz
        qd_ = Ii[3] * Lx + Ii[4] * Ly + Ii[5] * Lz
        rd_ = Ii[6] * Lx + Ii[7] * Ly + Ii[8] * Lz
        im = 1.0 / self.m
        if rec is not None:
            shaft = kQ * (w0 * w0 * w0 + w1 * w1 * w1 + w2 * w2 * w2 + w3 * w3 * w3)
            rec.update(thrust=thrust, airspeed=V, air_body=(ub, vb, wb), rho=self.rho, shaft_w=shaft,
                       battery_w=shaft / self.eta + self.avionics, ground_n=gforce,
                       power_nc=fn * vn + fe * ve + fd * vd + Mx * p + My * q + Mz * r,
                       accel=(fn * im, fe * im, fd * im + self.g), wind=self._w)
        return [vn, ve, vd, fn * im, fe * im, fd * im + self.g,
                -0.5 * (qx * p + qy * q + qz * r), 0.5 * (qw * p + qy * r - qz * q),
                0.5 * (qw * q + qz * p - qx * r), 0.5 * (qw * r + qx * q - qy * p),
                pd_, qd_, rd_, wdot[0], wdot[1], wdot[2], wdot[3]]

    # ---- stepping
    def step(self):
        """One fixed step: the wind and the air density sampled and held, the commands clipped
        to the rotors' range, RK4 over dt, the quaternion renormalised, the bristles moved."""
        dt, x = self.dt, self.x
        self._w = self._sample_wind()
        self._air(self.rho_at(x[2]))
        self._near = (-x[2] - self.ground(x[0], x[1])) < self._gcheck
        if self._near:
            self._gnorm = ground_normal(self.ground, x[0], x[1])
        lo, hi = 0.0, self.w_max
        c = [lo if a < lo else hi if a > hi else a for a in self.cmd]
        out = {}
        k1 = self.rates(x, c, out)
        self.out = out
        h2 = 0.5 * dt
        k2 = self.rates([a + h2 * b for a, b in zip(x, k1)], c)
        k3 = self.rates([a + h2 * b for a, b in zip(x, k2)], c)
        k4 = self.rates([a + dt * b for a, b in zip(x, k3)], c)
        h6 = dt / 6.0
        x = [a + h6 * (b1 + 2.0 * (b2 + b3) + b4) for a, b1, b2, b3, b4 in zip(x, k1, k2, k3, k4)]
        qn = 1.0 / math.sqrt(x[6] * x[6] + x[7] * x[7] + x[8] * x[8] + x[9] * x[9])
        x[6] *= qn
        x[7] *= qn
        x[8] *= qn
        x[9] *= qn
        for i in range(4):
            if x[13 + i] < 0.0:
                x[13 + i] = 0.0
            self.rotor_angle[i] = (self.rotor_angle[i] + x[13 + i] * dt) % TWO_PI
        self.x = x
        self.battery_j -= out["battery_w"] * dt
        if self._near:
            self._move_anchors()
        elif self._anchor[0] is not None or self._anchor[1] is not None or self._anchor[2] is not None \
                or self._anchor[3] is not None:
            self._anchor = [None, None, None, None]
        self.t += dt
        self.steps += 1

    def _move_anchors(self):
        """Each skid point's bristle: planted where the point first touches, dragged along when
        its spring would pass Coulomb's limit (the point is sliding), dropped when it lifts."""
        x = self.x
        R = rot_of_quat(x[6], x[7], x[8], x[9]).tolist()
        nn, ne, nd = self._gnorm
        for i, (cx, cy, cz) in enumerate(self._skid):
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
    def set_state(self, n=0.0, e=0.0, h=0.0, yaw=0.0, roll=0.0, pitch=0.0, velocity=(0.0, 0.0, 0.0),
                  rates=(0.0, 0.0, 0.0), rotor_speed=None):
        """The state: position (h up), zyx Euler angles (rad), NED velocity, body rates, and all
        four rotors at rotor_speed (default: the hover's), commanded to stay there."""
        w = self.hover_speed() if rotor_speed is None else rotor_speed
        q = quat_wxyz(rot_nb(roll, pitch, yaw))
        self.x = [n, e, -h, velocity[0], velocity[1], velocity[2], q[0], q[1], q[2], q[3],
                  rates[0], rates[1], rates[2], w, w, w, w]
        self.cmd = [w, w, w, w]
        self._anchor = [None, None, None, None]
        self._air(self.rho_at(-h))

    def place_on_ground(self, n=0.0, e=0.0, yaw=0.0, rotor_speed=0.0):
        """At rest on its skids at (n, e), lying on the ground's slope, the rotors stopped (or
        at rotor_speed) and commanded to stay so."""
        nn, ne, nd = ground_normal(self.ground, n, e)
        b3 = np.array([-nn, -ne, -nd])                                   # body z: into the ground
        xc = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        b2 = np.cross(b3, xc)
        b2 /= np.linalg.norm(b2)
        b1 = np.cross(b2, b3)
        R = np.column_stack([b1, b2, b3])
        q = quat_wxyz(R)
        # the centre of gravity above the ground: the skids' depth below it, less the springs' sag
        sag = self.m * self.g * (-nd) / (4.0 * self._gk)
        depth = self._skid[0][2] - sag
        pos = np.array([n, e, -self.ground(n, e)]) - depth * b3
        self.x = [float(pos[0]), float(pos[1]), float(pos[2]), 0.0, 0.0, 0.0, q[0], q[1], q[2], q[3],
                  0.0, 0.0, 0.0, rotor_speed, rotor_speed, rotor_speed, rotor_speed]
        self.cmd = [rotor_speed] * 4
        self._anchor = [None, None, None, None]
        self._air(self.rho_at(self.x[2]))

    # ---- what a scene reads
    def rotation(self):
        """R_b^n as a 3 x 3 array."""
        x = self.x
        return rot_of_quat(x[6], x[7], x[8], x[9])

    def euler(self):
        """(roll, pitch, yaw), rad, zyx."""
        _, _, _, _, _, _, qw, qx, qy, qz = self.x[:10]
        return (math.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy)),
                math.asin(clamp(2.0 * (qw * qy - qz * qx), -1.0, 1.0)),
                math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))

    def tilt(self):
        """The angle between the rotor axis and the vertical, rad."""
        x = self.x
        return math.acos(clamp(1.0 - 2.0 * (x[7] * x[7] + x[8] * x[8]), -1.0, 1.0))

    def ground_velocity(self):
        return tuple(self.x[3:6])

    def air_velocity_ned(self):
        x, w = self.x, self._w
        return (x[3] - w[0], x[4] - w[1], x[5] - w[2])

    def height_above_ground(self):
        """The lowest skid point's height above the ground, m (negative: the springs' sag)."""
        x = self.x
        R = rot_of_quat(x[6], x[7], x[8], x[9]).tolist()
        best = float("inf")
        for cx, cy, cz in self._skid:
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
    def battery_fraction(self):
        return self.battery_j / (self.battery_wh * 3600.0)

    def world_position(self, h0=0.0):
        """The centre of gravity in the world."""
        return ned_to_world(self.x[0], self.x[1], self.x[2], h0)

    def world_rotation(self):
        """The .glb's rotation in the world: model -> body -> NED -> world."""
        return C_WN @ self.rotation() @ C_BM

    def point_ned(self, layout_point):
        """A point of the spec's layout (X forward, Y up, Z right, from the .glb's origin) in NED."""
        r = np.array(body_of(layout_point)) - self.cg_body
        return np.array(self.x[:3]) + self.rotation() @ r

    def rotor_hubs_ned(self):
        """The four hubs (fr, fl, rl, rr) in NED from the current state, a 4 x 3 array, m."""
        x = self.x
        r = np.array([h[:3] for h in self._rot])
        return np.array(x[:3]) + r @ rot_of_quat(x[6], x[7], x[8], x[9]).T

    def energy(self):
        """Kinetic (translation and rotation of the airframe) plus potential energy per the NED
        origin, J. The rotors' own spin is not in it."""
        x = self.x
        w = np.array(x[10:13])
        return (0.5 * self.m * (x[3] * x[3] + x[4] * x[4] + x[5] * x[5]) + 0.5 * float(w @ self.I @ w)
                - self.m * self.g * x[2])

    def angular_momentum_ned(self):
        """The airframe's and the rotors' angular momentum about the centre of gravity, NED."""
        x = self.x
        H = self.I @ np.array(x[10:13])
        H[2] -= self.Jr * sum(s * w for s, w in zip(self.spin, x[13:17]))
        return self.rotation() @ H

    # ---- the hover, the trim, the calibration
    def hover_speed(self, mass=None, rho=None):
        """The rotor speed that carries the weight in still air, rad/s."""
        rho = self.rho if rho is None else rho
        return math.sqrt((self.m if mass is None else mass) * self.g / (4.0 * self.kT_rho * rho))

    def hover_power(self, mass=None, rho=None):
        """The hover in still air at this mass (default: the aircraft's) and density (default:
        sea level): rotor speed, thrust, shaft and battery power, and the time the battery
        lasts at that power. CALIBRATED against the product page's flight time, not validated:
        the spec's figure of merit and drive efficiency were chosen so it comes out."""
        rho = self.rho0 if rho is None else rho
        mass = self.m if mass is None else mass
        w = self.hover_speed(mass, rho)
        shaft = 4.0 * self.kQ_rho * rho * w ** 3
        battery = shaft / self.eta + self.avionics
        return {"mass": mass, "rho": rho, "omega": w, "rpm": w * 60.0 / TWO_PI, "tip_speed": w * self.R,
                "thrust_per_rotor": mass * self.g / 4.0, "induced_w": shaft * self.fm, "shaft_w": shaft,
                "battery_w": battery, "endurance_min": self.battery_wh / battery * 60.0}

    def trim(self, air_velocity=(0.0, 0.0, 0.0), yaw=0.0, rho=None, pitch=None, iters=60):
        """Steady straight flight at an air-relative velocity (NED, m/s) and heading: solves the
        three force and three moment equations for roll, pitch and the four rotor speeds. Given
        a pitch, it is held and kh is solved in its place (calibrate's use). Returns a dict;
        the aircraft's state is not touched. Holding position in a steady wind w is
        trim(air_velocity=-w)."""
        rho = self.rho0 if rho is None else rho
        keep = (self.x, self.cmd, self._w, self.rho, self._near, self.kh_rho)
        self._w, self._near = (0.0, 0.0, 0.0), False
        # the first guess: leaning into the air by what the airframe's drag asks, the rotors fast
        # enough to carry the weight against the air through the tilted discs
        vh = math.hypot(air_velocity[0], air_velocity[1])
        th0 = math.atan2(0.5 * rho * self._cda[0] * vh * vh * 1.5, self.m * self.g) if pitch is None else abs(pitch)
        beta = math.atan2(air_velocity[1], air_velocity[0]) - yaw
        kT, kzv = self.kT_rho * rho, self.kz_rho * rho * vh * math.sin(th0)
        w0 = (kzv + math.sqrt(kzv * kzv + 4.0 * kT * self.m * self.g / (4.0 * math.cos(th0)))) / (2.0 * kT)
        z = (np.array([th0 * math.sin(beta), -th0 * math.cos(beta), w0, w0, w0, w0]) if pitch is None
             else np.array([0.0, self.kh_rho, w0, w0, w0, w0]))
        scale = np.array([1.0, 1.0 if pitch is None else 1e-3, 100.0, 100.0, 100.0, 100.0])

        def resid(z):
            th = z[1] if pitch is None else pitch
            if pitch is not None:
                self.kh_rho = float(z[1])
            self._air(rho)
            q = quat_wxyz(rot_nb(float(z[0]), float(th), yaw))
            ws = [float(a) for a in z[2:6]]
            f = self.rates([0.0, 0.0, 0.0, air_velocity[0], air_velocity[1], air_velocity[2], q[0], q[1], q[2], q[3],
                            0.0, 0.0, 0.0] + ws, ws)
            return np.array(f[3:6] + f[10:13])
        step = np.ones(6)
        for _ in range(iters):
            r0 = resid(z)
            J = np.zeros((6, 6))
            for k in range(6):
                hk = 1e-6 * scale[k]
                zp, zm = z.copy(), z.copy()
                zp[k] += hk
                zm[k] -= hk
                J[:, k] = (resid(zp) - resid(zm)) / (2.0 * hk)
            step = np.linalg.solve(J, -r0)
            lim = np.max(np.abs(step / (scale * np.array([0.3, 0.3 if pitch is None else 1e9, 1.0, 1.0, 1.0, 1.0]))))
            z = z + step / max(1.0, lim)                    # damped: at most 0.3 rad or 100 rad/s a pass
            if np.max(np.abs(step / scale)) < 1e-12:
                break
        res = float(np.max(np.abs(resid(z))))
        out = {}
        th = float(z[1]) if pitch is None else pitch
        q = quat_wxyz(rot_nb(float(z[0]), th, yaw))
        ws = [float(a) for a in z[2:6]]
        self.rates([0.0, 0.0, 0.0, air_velocity[0], air_velocity[1], air_velocity[2], q[0], q[1], q[2], q[3],
                    0.0, 0.0, 0.0] + ws, ws, out)
        # the horizontal force the rotors' H force and the airframe's drag each carry, along the air velocity
        R = rot_nb(float(z[0]), th, yaw)
        ub, vb, wb = out["air_body"]
        V = out["airspeed"]
        body = R @ (self._dk * V * np.array([self._cda[0] * ub, self._cda[1] * vb, self._cda[2] * wb]))
        hforce = R @ np.array([-self._kh * sum(ws) * ub, -self._kh * sum(ws) * vb, 0.0])
        av = np.array(air_velocity, float)
        sp = float(np.linalg.norm(av[:2]))
        along = av[:2] / sp if sp > 1e-9 else np.zeros(2)
        sol = {"roll": float(z[0]), "pitch": th, "tilt": math.acos(clamp(R[2, 2], -1.0, 1.0)), "omega": ws,
               "kh_rho": self.kh_rho, "battery_w": out["battery_w"], "thrust": out["thrust"], "residual": res,
               "body_drag": -float(body[:2] @ along), "rotor_drag": -float(hforce[:2] @ along),
               "saturated": max(ws) > self.w_max, "rho": rho}
        self.x, self.cmd, self._w, rho_, self._near, self.kh_rho = keep
        self._air(rho_)
        if pitch is not None:
            sol["kh_rho_solved"] = sol["kh_rho"]
            sol["kh_rho"] = self.kh_rho
        return sol

    def calibrate(self):
        """kh, the rotors' in-plane drag: solved so steady level flight at the product page's
        maximum speed needs exactly its maximum tilt (23 m/s at 30 deg), at the mass without a
        payload (6.47 kg) and sea-level density: the full trim, moments included, with the pitch
        held at -30 deg and kh in the unknowns. If the airframe's drag alone needs more than
        that tilt, kh stays at zero and the tilt the model needs is reported. A calibration, not
        a validation: the model meets this point by construction. Returns (and keeps, as
        ``calibration``) kh, the rotors' share of the drag there, and the rotor speed the point
        asks for against the speed of the spec's maximum thrust."""
        lim = self.spec["aircraft"]["limits"]
        V, tilt = lim["max_horizontal_ms"], math.radians(lim["max_tilt_deg"])
        probe = M350(self.spec, dt=self.dt, payload=False, calibrate=False)
        sol = probe.trim((V, 0.0, 0.0), pitch=-tilt)
        kh = sol["kh_rho_solved"]
        cal = {"speed": V, "tilt_target": tilt, "mass": probe.m}
        if kh < 0.0 or sol["residual"] > 1e-6:
            kh = 0.0
            cal["note"] = "the airframe's drag alone needs more than the product's tilt: kh = 0"
        probe.kh_rho = kh
        sol = probe.trim((V, 0.0, 0.0))
        total = sol["body_drag"] + sol["rotor_drag"]
        cal.update(kh_rho=kh, kh=kh * self.rho0, tilt=sol["tilt"], rotor_share=sol["rotor_drag"] / total,
                   body_drag=sol["body_drag"], rotor_drag=sol["rotor_drag"], omega=max(sol["omega"]),
                   omega_max=self.w_max, saturated=sol["saturated"], battery_w=sol["battery_w"],
                   residual=sol["residual"])
        self.kh_rho = kh
        self._air(self.rho)
        self.calibration = cal
        return cal


_INDEX = {k: i for i, k in enumerate(STATE)}


# --------------------------------------------------------------------------- #
#  The autopilot: OURS
# --------------------------------------------------------------------------- #
class Autopilot:
    """OURS, not DJI's flight controller: the cascade multirotor autopilots are built as.

        position   v_c = v_ff + kp (p_c - p), inside the speed limits
        velocity   a_c = a_ff + kp (v_c - v) + ki int(v_c - v)
        thrust     the specific force f = a_c - g; its tilt from the vertical held inside the
                   tilt limit; the body's z axis along -f, its x axis toward the yaw setpoint;
                   the collective so the vertical part of the thrust is m f_z at the attitude
                   the aircraft has
        attitude   body rates = 2 kp (the vector part of the quaternion from the attitude it
                   has to the one it wants), yaw on a square-root law inside its acceleration
        rates      angular acceleration = kp (w_c - w) + ki int - kd w', times the inertia
        mixer      four thrusts from the collective and the three moments, held inside the
                   rotors' range by giving up yaw first, then collective, then roll and pitch;
                   each thrust to a rotor speed by the static kT

    It does not know the wind (nor the thrust's loss to the air through the discs): it leans
    into it because the velocity loop's integral says so. It does know the aircraft's mass,
    inertia and rotor geometry. STATE FEEDBACK IS EXACT: position, velocity, attitude and rates
    are read from the model, with no estimator and no sensor noise, so the hover it holds is
    better than a real aircraft's. Gains and limits: the spec's "control" block, which says how
    each was chosen against the motor's lag.

    command(position=(n, e, h), velocity=(vn, ve, vh), yaw=rad, acceleration=(an, ae, ah),
    yaw_rate=rad/s): what is not given is kept. With a position, velocity and acceleration are
    feed-forward (None: zero). position=None switches the position loop off and flies the
    velocity. All in NED axes with the third component UP (a height, a climb rate).
    ``idle`` True holds the rotors at idle with the loops reset (on the ground)."""

    def __init__(self, m350):
        self.m350 = m = m350
        c = self.cfg = m.spec["control"]
        lim = m.spec["aircraft"]["limits"]
        self.tilt_max = math.radians(lim["max_tilt_forward_vision_deg"] - c["attitude"]["tilt_margin_deg"])
        self.v_up, self.v_down = lim["max_ascent_ms"], lim["max_descent_ms"]
        self.yaw_rate_max = math.radians(lim["max_yaw_rate_dps"])
        P, V, A, R = c["position"], c["velocity"], c["attitude"], c["rate"]
        self.kp_p, self.kp_h, self.v_xy = P["kp_xy"], P["kp_z"], P["speed_xy_max_ms"]
        self.brake = P["brake_ms2"]
        self.kp_v, self.ki_v, self.kp_vh, self.ki_vh = V["kp_xy"], V["ki_xy"], V["kp_z"], V["ki_z"]
        self.i_xy_max, self.i_h_max, self.e_i = V["integral_xy_ms2"], V["integral_z_ms2"], V["integral_error_ms"]
        self.kp_att, self.kp_yaw, self.yaw_acc = A["kp_tilt"], A["kp_yaw"], A["yaw_accel_rads2"]
        self.rate_max = math.radians(A["rate_max_dps"])
        self.kp_r, self.ki_r, self.kd_r = R["kp"], R["ki"], R["kd"]
        self.kp_ry, self.ki_ry = R["kp_yaw"], R["ki_yaw"]
        self.i_r_max = R["integral_rads2"]
        self.d_alpha = 1.0 - math.exp(-TWO_PI * R["d_cutoff_hz"] * m.dt)
        # the mixer: [T, Mx, My, Mz] = B [T_1..T_4]
        cq = m.kQ_rho / m.kT_rho
        B = np.array([[1.0, 1.0, 1.0, 1.0], [-r[1] for r in m._rot], [r[0] for r in m._rot],
                      [r[3] * cq for r in m._rot]])
        self._mix = [tuple(float(a) for a in row) for row in np.linalg.inv(B)]
        self._Id = (float(m.I[0, 0]), float(m.I[1, 1]), float(m.I[2, 2]))
        self.idle = False
        self.reset()

    def reset(self):
        """Setpoints at the aircraft's own position and heading; the integrals emptied."""
        m = self.m350
        x = m.x
        self.pos_c = (x[0], x[1], -x[2])
        self.vel_c = (0.0, 0.0, 0.0)
        self.acc_c = (0.0, 0.0, 0.0)
        self.yaw_c = m.euler()[2]
        self.yaw_rate_c = 0.0
        self.hold_position = True
        self.i_v = [0.0, 0.0, 0.0]
        self.i_r = [0.0, 0.0, 0.0]
        self._wd = [0.0, 0.0, 0.0]
        self._w_prev = (x[10], x[11], x[12])
        self.tilt_c = 0.0
        self.thrust_c = m.m * m.g
        self.vel_sp = (0.0, 0.0, 0.0)
        self.rate_sp = (0.0, 0.0, 0.0)
        self.saturated = 0

    def command(self, position=KEEP, velocity=KEEP, yaw=KEEP, acceleration=KEEP, yaw_rate=KEEP):
        if position is not KEEP:
            self.hold_position = position is not None
            if position is not None:
                self.pos_c = (float(position[0]), float(position[1]), float(position[2]))
        if velocity is not KEEP:
            self.vel_c = (0.0, 0.0, 0.0) if velocity is None else (float(velocity[0]), float(velocity[1]), float(velocity[2]))
        if acceleration is not KEEP:
            self.acc_c = (0.0, 0.0, 0.0) if acceleration is None else (float(acceleration[0]), float(acceleration[1]), float(acceleration[2]))
        if yaw is not KEEP and yaw is not None:
            self.yaw_c = float(yaw)
        if yaw_rate is not KEEP:
            self.yaw_rate_c = 0.0 if yaw_rate is None else float(yaw_rate)

    def update(self):
        """Writes the M350's four rotor speed commands for its next step."""
        m = self.m350
        x, dt, g = m.x, m.dt, m.g
        if self.idle:
            m.cmd = [m.w_idle] * 4
            self.i_v = [0.0, 0.0, 0.0]
            self.i_r = [0.0, 0.0, 0.0]
            self._w_prev = (x[10], x[11], x[12])
            return
        n, e, d, vn, ve, vd, qw, qx, qy, qz, p, q, r = x[:13]
        vh = -vd
        # position -> velocity
        vcn, vce, vch = self.vel_c
        if self.hold_position:
            dn, de = self.pos_c[0] - n, self.pos_c[1] - e
            k = self.kp_p
            dist, lin = math.hypot(dn, de), self.brake / (k * k)
            if dist > lin:                                # far out: the speed it can still brake from
                k = math.sqrt(2.0 * self.brake * (dist - 0.5 * lin)) / dist
            vcn += k * dn
            vce += k * de
            vch += self.kp_h * (self.pos_c[2] + d)
        sp = math.hypot(vcn, vce)
        if sp > self.v_xy:
            vcn, vce = vcn * self.v_xy / sp, vce * self.v_xy / sp
        vch = clamp(vch, -self.v_down, self.v_up)
        self.vel_sp = (vcn, vce, vch)
        # velocity -> acceleration (PI, with the feed-forward)
        en, ee, eh = vcn - vn, vce - ve, vch - vh
        iv = self.i_v
        an = self.acc_c[0] + self.kp_v * en + iv[0]
        ae = self.acc_c[1] + self.kp_v * ee + iv[1]
        ah = self.acc_c[2] + self.kp_vh * eh + iv[2]
        # thrust vector: vertical first, then the tilt limit on what is left
        fz = clamp(ah + g, 0.3 * g, 2.0 * g)
        sat_h = fz != ah + g
        fh = math.hypot(an, ae)
        fmax = fz * math.tan(self.tilt_max)
        sat_xy = fh > fmax
        if sat_xy:
            an, ae = an * fmax / fh, ae * fmax / fh
        # anti-windup: an integral stands still while its axis is limited and the error would grow it
        # and takes in no more than e_i of error: a commanded manoeuvre's error is not the wind's
        ei = self.e_i
        if not sat_xy or en * iv[0] < 0.0:
            iv[0] = clamp(iv[0] + self.ki_v * clamp(en, -ei, ei) * dt, -self.i_xy_max, self.i_xy_max)
        if not sat_xy or ee * iv[1] < 0.0:
            iv[1] = clamp(iv[1] + self.ki_v * clamp(ee, -ei, ei) * dt, -self.i_xy_max, self.i_xy_max)
        if not sat_h or eh * iv[2] < 0.0:
            iv[2] = clamp(iv[2] + self.ki_vh * clamp(eh, -ei, ei) * dt, -self.i_h_max, self.i_h_max)
        fn_ = math.sqrt(an * an + ae * ae + fz * fz)
        b3n, b3e, b3d = -an / fn_, -ae / fn_, fz / fn_
        self.tilt_c = math.acos(clamp(b3d, -1.0, 1.0))
        # the attitude wanted: z along b3, x toward the yaw setpoint
        cy, sy = math.cos(self.yaw_c), math.sin(self.yaw_c)
        b2n, b2e, b2d = -b3d * sy, b3d * cy, b3n * sy - b3e * cy
        k = 1.0 / math.sqrt(b2n * b2n + b2e * b2e + b2d * b2d)
        b2n, b2e, b2d = b2n * k, b2e * k, b2d * k
        b1n, b1e, b1d = b2e * b3d - b2d * b3e, b2d * b3n - b2n * b3d, b2n * b3e - b2e * b3n
        tr = b1n + b2e + b3d
        if tr > 0.0:
            s = math.sqrt(tr + 1.0) * 2.0
            dw, dx, dy, dz = 0.25 * s, (b2d - b3e) / s, (b3n - b1d) / s, (b1e - b2n) / s
        else:                                             # more than 90 deg off level and north: never here
            dx, dy, dz, dw = quat_of(np.array([[b1n, b2n, b3n], [b1e, b2e, b3e], [b1d, b2d, b3d]]))
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
        # yaw: proportional near the setpoint, the square root of the distance beyond (what its
        # acceleration can stop from)
        eyaw = 2.0 * ez
        lin = self.yaw_acc / (self.kp_yaw * self.kp_yaw)
        if abs(eyaw) > lin:
            rc = math.copysign(math.sqrt(2.0 * self.yaw_acc * (abs(eyaw) - 0.5 * lin)), eyaw)
        else:
            rc = self.kp_yaw * eyaw
        rc = clamp(rc + self.yaw_rate_c, -self.yaw_rate_max, self.yaw_rate_max)
        self.rate_sp = (pc, qc, rc)
        # rates -> moments
        a_ = self.d_alpha
        wd = self._wd
        wd[0] += a_ * ((p - self._w_prev[0]) / dt - wd[0])
        wd[1] += a_ * ((q - self._w_prev[1]) / dt - wd[1])
        wd[2] += a_ * ((r - self._w_prev[2]) / dt - wd[2])
        self._w_prev = (p, q, r)
        ir = self.i_r
        ep, eq, er = pc - p, qc - q, rc - r
        Mx = self._Id[0] * (self.kp_r * ep + ir[0] - self.kd_r * wd[0])
        My = self._Id[1] * (self.kp_r * eq + ir[1] - self.kd_r * wd[1])
        Mz = self._Id[2] * (self.kp_ry * er + ir[2])
        # collective: the vertical part of the thrust is m fz at the attitude it has
        r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
        T = m.m * fz / max(r22, 0.5)
        self.thrust_c = T
        # mixer
        kT = m._kT
        tmin, tmax = kT * m.w_idle * m.w_idle, kT * m.w_max * m.w_max
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
        m.cmd = [math.sqrt(max(base[i] + ky * A[i][3] * Mz, 0.0) / kT) for i in range(4)]

    def status(self):
        v = self.vel_sp
        return (f"p_c ({self.pos_c[0]:7.2f} {self.pos_c[1]:7.2f} {self.pos_c[2]:6.2f})  v_c ({v[0]:+5.2f} {v[1]:+5.2f} {v[2]:+5.2f})  "
                f"yaw_c {math.degrees(self.yaw_c):6.1f}  tilt_c {math.degrees(self.tilt_c):4.1f}  "
                f"T_c {self.thrust_c:5.1f} N  mixer {('free', 'yaw', 'collective', 'roll/pitch')[self.saturated]}")


# --------------------------------------------------------------------------- #
#  The mission: OURS
# --------------------------------------------------------------------------- #
def ap_lag(ap):
    """The lag of the autopilot's attitude loop, s: 1 / kp_tilt (the ratio of the two lowest
    coefficients of tau s^3 + (1 + kd) s^2 + kp s + kp kp_tilt)."""
    return 1.0 / ap.kp_att


class Mission:
    """Straight legs between waypoints, take-off and landing. A leg is a dict:

        dict(pos=(n, e, h), speed=m/s, yaw="track" | "poi" | degrees | None, poi=(n, e, h), hold=s,
             accel=m/s^2, jerk=m/s^3, yaw_rate=deg/s)
        dict(kind="takeoff", height=m above where it stands, speed=m/s, hold=s)
        dict(kind="land", rate=m/s)

    A leg runs from where the last one ended along a straight line on an S-curve: the jerk, the
    acceleration and the speed along it are limited (the spec's control.mission, or the leg's
    own accel and jerk; the climb and descent limits hold the speed back on a steep leg), so
    the acceleration handed to the autopilot, and with it the lean, builds and decays without a
    step. The profile's position, velocity and acceleration all go to the autopilot: the
    acceleration as the profile has it, the position and the velocity 1 / kp_tilt later (the
    attitude loop's lag, ``ap_lag``), which is when that acceleration has become a lean, so the
    feedback has next to nothing to add and the lean is the profile's. At the end the leg waits until the aircraft is within the accept radius and slow, then holds for
    `hold` seconds. The yaw setpoint goes to the leg's (along the track, at the point of
    interest, or a heading in degrees from north) inside a yaw rate (the spec's, or the leg's
    yaw_rate) and a yaw acceleration: no step in the yaw rate either.

    Take-off: the rotors spool up at idle on the ground, then the setpoint climbs (the same
    S-curve). Landing: the setpoint descends at `rate`, slower below the spec's height above the
    ground, and the rotors are cut to idle on contact. The height above the ground is read from
    the model (exact, as the autopilot's feedback is).

    update(m350, autopilot) once per step, before autopilot.update(). ``leg`` is the index of
    the current leg, ``phase`` what it is doing, ``done`` True after the last; ``setpoint`` the
    position setpoint, ``look`` the point the camera should look at (the leg's poi, or None);
    ``v_peak`` and ``a_peak`` the speed and the acceleration the current leg reaches."""

    def __init__(self, legs, spec=None):
        self.cfg = (spec or load_spec())["control"]["mission"]
        self.legs = [dict(l) for l in legs]
        self.leg = -1
        self.phase = "start"
        self.done = len(self.legs) == 0
        self.setpoint = None
        self.look = None
        self.t_leg = 0.0
        self.t_hold = 0.0
        self.yaw_c = None
        self.yaw_rate = 0.0
        self.yaw_target = None
        self.touchdown = None

    def _begin(self, m, ap):
        self.leg += 1
        if self.leg >= len(self.legs):
            self.done = True
            self.phase = "done"
            return
        L = self.legs[self.leg]
        kind = L.get("kind", "goto")
        self.t_leg = self.t_hold = 0.0
        if self.setpoint is None:
            self.setpoint = (m.x[0], m.x[1], -m.x[2])
        self.p0 = self.setpoint
        self.look = L.get("poi")
        self.yaw_target = None                            # a leg that names no heading lets the nose come to rest
        if kind == "takeoff":
            ap.idle = True
            self.phase = "spool"
            self.p0 = (m.x[0], m.x[1], -m.x[2])
            self.p1 = (self.p0[0], self.p0[1], self.p0[2] + L["height"])
        elif kind == "land":
            self.phase = "descend"
            return
        else:
            self.p1 = tuple(float(a) for a in L["pos"])
            self.phase = "fly"
        self._profile(L, ap)

    @staticmethod
    def _ramp(v, a, j):
        """The S-shaped ramp from rest to the speed v inside the acceleration a and the jerk j:
        (how long the jerk acts at each of its ends, how long the whole ramp takes). It covers
        v times half its time."""
        if v >= a * a / j:
            tj = a / j
            return tj, tj + v / a
        tj = math.sqrt(v / j)
        return tj, 2.0 * tj

    def _profile(self, L, ap):
        cfg = self.cfg
        d = [b - a for a, b in zip(self.p0, self.p1)]
        self.length = math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2])
        self.dir = tuple(a / self.length for a in d) if self.length > 1e-9 else (0.0, 0.0, 0.0)
        v = float(L.get("speed", cfg["speed_ms"]))
        if self.dir[2] > 1e-9:
            v = min(v, cfg["climb_fraction"] * ap.v_up / self.dir[2])
        elif self.dir[2] < -1e-9:
            v = min(v, cfg["climb_fraction"] * ap.v_down / -self.dir[2])
        a, j = float(L.get("accel", cfg["accel_ms2"])), float(L.get("jerk", cfg["jerk_ms3"]))
        if self.length <= 1e-9:
            v = 0.0
        tj, ta = self._ramp(v, a, j)
        if v * ta > self.length:                          # no room to reach the speed: the peak that fits
            lo, hi = 0.0, v
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                lo, hi = (lo, mid) if mid * self._ramp(mid, a, j)[1] > self.length else (mid, hi)
            v = lo
            tj, ta = self._ramp(v, a, j)
        tc = (self.length - v * ta) / v if v > 1e-9 else 0.0
        self.v_peak, self.a_peak = v, j * tj
        # seven stretches of constant jerk: up, hold, off; cruise; and the mirror of the first three
        self._segs = []
        t0 = s = vv = aa = 0.0
        for dur, jk in ((tj, j), (ta - 2.0 * tj, 0.0), (tj, -j), (tc, 0.0), (tj, -j), (ta - 2.0 * tj, 0.0), (tj, j)):
            self._segs.append((t0, dur, jk, s, vv, aa))
            s += vv * dur + 0.5 * aa * dur * dur + jk * dur * dur * dur / 6.0
            vv += aa * dur + 0.5 * jk * dur * dur
            aa += jk * dur
            t0 += dur
        self.t_end = t0

    def _along(self, t):
        """Distance, speed and acceleration along the leg t seconds into it."""
        if t <= 0.0:
            return 0.0, 0.0, 0.0
        if t < self.t_end:
            for t0, dur, jk, s, v, a in self._segs:
                u = t - t0
                if u < dur:
                    return s + u * (v + u * (0.5 * a + u * jk / 6.0)), v + u * (a + 0.5 * u * jk), a + jk * u
        return self.length, 0.0, 0.0

    def _yaw(self, m, L, target):
        """The yaw setpoint toward a target heading (None: come to rest) inside the yaw rate and
        the yaw acceleration: the rate wanted is the target's own rate plus (acceleration / rate
        limit) times the error, so it closes without a step in the rate and without overshoot."""
        cfg, dt, x = self.cfg, m.dt, m.x
        rmax = math.radians(float(L.get("yaw_rate", cfg["yaw_rate_dps"])))
        acc = math.radians(cfg["yaw_accel_dps2"])
        want = 0.0
        if target is not None:
            want = acc / rmax * wrap(target - self.yaw_c)
            poi = L.get("poi") if L.get("yaw") == "poi" else None
            if poi is not None:                           # the bearing's own rate as the aircraft moves
                dn, de = poi[0] - x[0], poi[1] - x[1]
                want += (de * x[3] - dn * x[4]) / (dn * dn + de * de)
            want = clamp(want, -rmax, rmax)
        self.yaw_rate += clamp(want - self.yaw_rate, -acc * dt, acc * dt)
        self.yaw_c = wrap(self.yaw_c + self.yaw_rate * dt)
        return self.yaw_rate

    def update(self, m, ap):
        if self.done:
            return
        if self.leg < 0:
            self.yaw_c = m.euler()[2]
            self._begin(m, ap)
            if self.done:
                return
        cfg, dt = self.cfg, m.dt
        L = self.legs[self.leg]
        x = m.x
        if self.phase == "spool":
            self.t_leg += dt
            if self.t_leg >= cfg["spool_s"]:
                ap.idle = False
                ap.reset()
                self.yaw_c, self.yaw_rate = ap.yaw_c, 0.0
                self.t_leg = 0.0
                self.phase = "fly"
            return
        if self.phase == "descend":
            hg = m.height_above_ground()
            rate = float(L.get("rate", cfg["land_rate_ms"]))
            if hg < cfg["land_slow_height_m"]:
                rate = min(rate, cfg["land_slow_rate_ms"])
            sp = self.setpoint
            # the setpoint leads the aircraft down by no more than the rate's own lag
            hc = max(sp[2] - rate * dt, -x[2] - rate / ap.kp_h - 0.5)
            self.setpoint = (sp[0], sp[1], hc)
            yaw_rate = self._yaw(m, L, None)
            ap.command(position=self.setpoint, velocity=(0.0, 0.0, -rate), acceleration=None, yaw=self.yaw_c, yaw_rate=yaw_rate)
            if m.on_ground:
                self.touchdown = {"t": m.t, "sink": x[5], "speed": math.sqrt(x[3] ** 2 + x[4] ** 2 + x[5] ** 2)}
                ap.idle = True
                self.phase = "landed"
            return
        if self.phase == "landed":
            self.t_hold += dt
            if self.t_hold >= float(L.get("hold", cfg["land_settle_s"])):
                self._begin(m, ap)
            return
        # a flown leg (and a take-off's climb)
        mode = L.get("yaw", "track" if L.get("kind", "goto") == "goto" else None)
        if mode == "track":
            if math.hypot(self.dir[0], self.dir[1]) * self.length > cfg["track_min_m"]:
                self.yaw_target = math.atan2(self.dir[1], self.dir[0])
        elif mode == "poi":
            poi = L["poi"]
            if math.hypot(poi[0] - x[0], poi[1] - x[1]) > cfg["track_min_m"]:
                self.yaw_target = math.atan2(poi[1] - x[1], poi[0] - x[0])
        elif mode is not None:
            self.yaw_target = math.radians(float(mode))
        yaw_rate = self._yaw(m, L, self.yaw_target)
        if self.phase == "fly":
            self.t_leg += dt
            # the acceleration goes out as the profile has it; the position and the velocity it
            # is held to are the profile's 1 / kp_tilt earlier, the attitude loop's lag, which is
            # when that acceleration has become a lean: the feedback then has nothing to add
            s, v, _ = self._along(self.t_leg - ap_lag(ap))
            a = self._along(self.t_leg)[2]
            dn, de, dh = self.dir
            self.setpoint = (self.p0[0] + dn * s, self.p0[1] + de * s, self.p0[2] + dh * s)
            ap.command(position=self.setpoint, velocity=(dn * v, de * v, dh * v), acceleration=(dn * a, de * a, dh * a),
                       yaw=self.yaw_c, yaw_rate=yaw_rate)
            if s >= self.length and v == 0.0:
                self.setpoint = self.p1
                self.phase = "settle"
            return
        ap.command(position=self.p1, velocity=None, acceleration=None, yaw=self.yaw_c, yaw_rate=yaw_rate)
        if self.phase == "settle":
            err = math.sqrt((x[0] - self.p1[0]) ** 2 + (x[1] - self.p1[1]) ** 2 + (-x[2] - self.p1[2]) ** 2)
            spd = math.sqrt(x[3] ** 2 + x[4] ** 2 + x[5] ** 2)
            if err < cfg["accept_radius_m"] and spd < cfg["accept_speed_ms"]:
                self.phase = "hold"
                self.t_hold = 0.0
        if self.phase == "hold":
            self.t_hold += dt
            if self.t_hold >= float(L.get("hold", 0.0)):
                self._begin(m, ap)


# --------------------------------------------------------------------------- #
#  The gimbal
# --------------------------------------------------------------------------- #
class Gimbal:
    """The H20T's three axes: pan about the airframe's up axis, then roll about the forward
    axis, then tilt about the starboard axis (the .glb's nodes gimbal_pan about +Y, gimbal_roll
    about +X, gimbal_tilt about +Z, each the parent of the next).

    update(m350, target) turns the line of sight toward a point (n, e, h) with the horizon
    level in the picture, each axis slewing at the spec's rate (ASSUMED there) inside its
    range (pan and tilt the controllable ranges, roll the mechanical one). With no target it
    looks along the aircraft's heading, `tilt_hold` (rad, 0 level, negative down) from the
    horizon, still stabilised against the airframe's roll and pitch.

    ``pan``, ``roll``, ``tilt``: the nodes' rotations, rad, relative to the airframe. As the
    nodes turn: pan positive to PORT (counter-clockwise seen from above), roll positive right
    side down, tilt positive raises the lens. ``los``: the line of sight it has, a NED unit
    vector; ``error``: its angle from the one asked for, rad."""

    def __init__(self, spec=None):
        g = (spec or load_spec())["payload"]["gimbal"]
        self.pan_lim = tuple(math.radians(a) for a in g["controllable_deg"]["pan"])
        self.tilt_lim = tuple(math.radians(a) for a in g["controllable_deg"]["tilt"])
        self.roll_lim = tuple(math.radians(a) for a in g["mechanical_deg"]["roll"])
        self.rate = math.radians(g["max_rate_dps"])
        self.pivot = (spec or load_spec())["layout"]["gimbal_pivot"]
        self.pan = self.roll = self.tilt = 0.0
        self.tilt_hold = 0.0
        self.los = (1.0, 0.0, 0.0)
        self.error = 0.0

    @staticmethod
    def _matrix(pan, roll, tilt):
        """Ry(pan) Rx(roll) Rz(tilt), the camera in the .glb's frame."""
        cp, sp, cr, sr, ct, st = math.cos(pan), math.sin(pan), math.cos(roll), math.sin(roll), math.cos(tilt), math.sin(tilt)
        return np.array([[cp * ct + sp * sr * st, -cp * st + sp * sr * ct, sp * cr],
                         [cr * st, cr * ct, -sr],
                         [-sp * ct + cp * sr * st, sp * st + cp * sr * ct, cp * cr]])

    def update(self, m, target=None, dt=None):
        dt = m.dt if dt is None else dt
        R = m.rotation()
        if target is not None:
            p = m.point_ned(self.pivot)
            d = np.array([target[0] - p[0], target[1] - p[1], -target[2] - p[2]])
        else:
            yaw = m.euler()[2]
            d = np.array([math.cos(yaw) * math.cos(self.tilt_hold), math.sin(yaw) * math.cos(self.tilt_hold),
                          -math.sin(self.tilt_hold)])
        d = d / np.linalg.norm(d)
        right = np.cross([0.0, 0.0, 1.0], d)                 # the picture's right: horizontal
        if np.linalg.norm(right) < 1e-6:                      # straight down or up: the airframe's right
            right = R[:, 1]
        right = right / np.linalg.norm(right)
        Rnc = np.column_stack([d, right, np.cross(d, right)])
        M = C_BM.T @ (R.T @ Rnc) @ C_BM                       # the camera in the .glb's frame
        roll = -math.asin(clamp(M[1, 2], -1.0, 1.0))
        tilt = math.atan2(M[1, 0], M[1, 1])
        pan = math.atan2(M[0, 2], M[2, 2])
        pan = self.pan + wrap(pan - self.pan)                 # the nearest turn to where it is
        if pan > self.pan_lim[1]:
            pan -= TWO_PI
        elif pan < self.pan_lim[0]:
            pan += TWO_PI
        s = self.rate * dt
        self.pan += clamp(clamp(pan, *self.pan_lim) - self.pan, -s, s)
        self.roll += clamp(clamp(roll, *self.roll_lim) - self.roll, -s, s)
        self.tilt += clamp(clamp(tilt, *self.tilt_lim) - self.tilt, -s, s)
        los = R @ (C_BM @ self._matrix(self.pan, self.roll, self.tilt)[:, 0])
        self.los = (float(los[0]), float(los[1]), float(los[2]))
        self.error = math.acos(clamp(float(los @ d), -1.0, 1.0))
        return self.los


# --------------------------------------------------------------------------- #
#  The 3D model
# --------------------------------------------------------------------------- #
class Visual:
    """m350.glb's nodes posed from an M350 (and a Gimbal).

    root    the loaded model's scene node (tp.GLTFLoader().load(model_path()).scene); anything
            with position.set, quaternion.set and get_object_by_name whose nodes have rotation

    pose(m350, gimbal, h0) places the root so the model's centre of gravity (not the .glb's
    origin) is at the state's position (world = (E, h0 - D, -N)), turns each propeller node
    (prop_fr, prop_fl, prop_rl, prop_rr) about its local +Y by that rotor's angle times its spin
    sense (+1 counter-clockwise seen from above), and the gimbal's nodes by its angles
    (gimbal_pan about +Y, gimbal_roll about +X, gimbal_tilt about +Z)."""

    PROPS = ("prop_fr", "prop_fl", "prop_rl", "prop_rr")
    GIMBAL = ("gimbal_pan", "gimbal_roll", "gimbal_tilt")

    def __init__(self, root):
        self.root = root
        self.props = [root.get_object_by_name(n) for n in self.PROPS]
        missing = [n for n, v in zip(self.PROPS, self.props) if v is None]
        if missing:
            raise KeyError(f"m350.glb has no node(s) {missing}: rebuild it with uav/build_m350_blender.py")
        self.gimbal = [root.get_object_by_name(n) for n in self.GIMBAL]      # None without the payload

    def pose(self, m, gimbal=None, h0=0.0):
        R = m.world_rotation()
        cg = np.array(m.world_position(h0))
        pos = cg - R @ np.array(m.cg_model)
        self.root.position.set(float(pos[0]), float(pos[1]), float(pos[2]))
        self.root.quaternion.set(*(float(a) for a in quat_of(R)))
        for node, angle, spin in zip(self.props, m.rotor_angle, m.spin):
            node.rotation.y = angle * spin
        if gimbal is not None and all(g is not None for g in self.gimbal):
            self.gimbal[0].rotation.y = gimbal.pan
            self.gimbal[1].rotation.x = gimbal.roll
            self.gimbal[2].rotation.z = gimbal.tilt
