"""
racer_rig -- a 5 inch racing quad as a flown rigid body: a flight controller and its four motors,
top down, each assist entering at its own layer and everything below it shared, and the 3D model
posed from the state.

Four things, and nothing scene-specific:

  * ``Racer``            -- the airframe (racer_spec.json): four motors that lag their commands,
                            slower down than up, thrust with the square of the rotor speed, props
                            that unload as the air comes into the discs faster, rotor drag across
                            the discs, the frame's quadratic drag on each axis, propwash in a sink,
                            a pack that sags, a sphere on a ground you hand in, a hard hit a crash.
                            Semi-implicit Euler at 240 Hz, the core it was ported from.
  * ``FlightController`` -- the layers above the motors, top down: the assist (AUTO a line, ARCADE
                            a heading, a speed and a height, ANGLE a tilt, ACRO body rates through
                            a radio's rate curve), a velocity layer with the drag and the props'
                            unloading in it, attitude, the rate loop, an airmode mixer.
  * ``Line``             -- a closed flown line and the speed its curvature allows, for AUTO.
  * ``Visual``           -- the racer.glb posed from a Racer, the rotors as overlay discs.

``Wind`` is m350_rig's, used as it is.

WHAT IS WHOSE
-------------
Nothing here is a product's or a paper's: it was tuned for a simulator of the class, and the spec
says so. It is not one measured craft, it has not been checked against a flight log or by a pilot,
and the flight controller is OURS, fed the exact state. Not in the model: ground effect, motor or
gyro noise, a filter stack, the air density's change with height, dirty air behind another drone.

Nothing here imports threepp: ``Visual`` is handed the loaded model's root node, and the threepp
module if it is to draw the discs.

FRAMES
------
m350_rig's: NED position (N, E, D), body axes x forward, y right, z down, the attitude a unit
quaternion (w, x, y, z), body to NED; world = (E, h0 - D, -N). The state ``x`` is the M350's STATE,
so a scene reads either the same way. The .glb and the spec's layout are X forward, Y up, Z right,
in metres, the origin at the centre of gravity.

TYPICAL USE
-----------
::

    from racer_rig import Racer, FlightController, Line, Sticks, Wind, Visual, model_path

    r = Racer(wind=Wind(4.0, from_deg=250.0, turbulence=1.0, seed=3))
    r.set_state(h=3.0, yaw=0.0)
    fc = FlightController(r, assist="acro")
    while flying:
        fc.sticks = Sticks(throttle=0.4, pitch=0.2)
        fc.update()                                   # the motors' commands
        r.step()                                      # one fixed step, r.dt
    visual = Visual(tp.GLTFLoader().load(model_path()).scene, tp=tp)
    visual.pose(r, h0=0.0)

    fc = FlightController(r, assist="auto", line=Line([(0, 0, 3), (60, 0, 5), (60, 40, 4), (0, 40, 6)]))
"""
import json
import math
import os
import random
import sys
from dataclasses import dataclass

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)                                 # examples/rigs/ (m350_rig)

from m350_rig import (C_BM, C_WN, STATE, Wind, _plain, clamp, flat_ground, ground_normal,  # noqa: E402,F401
                      ned_to_world, quat_of, quat_wxyz, rot_nb, rot_of_quat)

UAV_DIR = os.path.join(os.path.dirname(HERE), "uav")
SPEC_PATH = os.path.join(UAV_DIR, "racer_spec.json")
TWO_PI = 2.0 * math.pi
ROTORS = ("fr", "fl", "rl", "rr")
# Each rotor's side along body x and y, in the spec's order; its spin sense is the spec's.
SIDES = ((1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0))


def load_spec(path=None):
    with open(path or SPEC_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def model_path():
    """The generated racer.glb, or None where it has not been built."""
    p = os.path.join(UAV_DIR, "racer.glb")
    return p if os.path.isfile(p) else None


def clamp01(v):
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def approach(cur, to, max_step):
    return min(to, cur + max_step) if cur < to else max(to, cur - max_step)


def stick_rate(s, center, full):
    """A stick to a body rate: gentle round the centre, the full rate at the end (an FPV radio's cubic)."""
    return s * center + s * s * s * (full - center)


def quat_of_axes(xn, xe, xd, yn, ye, yd, zn, ze, zd):
    """(w, x, y, z) of the rotation matrix with these columns."""
    t = xn + ye + zd
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        q = (0.25 * s, (yd - ze) / s, (zn - xd) / s, (xe - yn) / s)
    elif xn > ye and xn > zd:
        s = math.sqrt(1.0 + xn - ye - zd) * 2.0
        q = ((yd - ze) / s, 0.25 * s, (yn + xe) / s, (zn + xd) / s)
    elif ye > zd:
        s = math.sqrt(1.0 + ye - xn - zd) * 2.0
        q = ((zn - xd) / s, (yn + xe) / s, 0.25 * s, (ze + yd) / s)
    else:
        s = math.sqrt(1.0 + zd - xn - ye) * 2.0
        q = ((xe - yn) / s, (zn + xd) / s, (ze + yd) / s, 0.25 * s)
    k = 1.0 / math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return (q[0] * k, q[1] * k, q[2] * k, q[3] * k)


# --------------------------------------------------------------------------- #
#  The airframe
# --------------------------------------------------------------------------- #
class Racer:
    """The racing quad as a rigid body on four motors.

    spec    racer_spec.json as a dict (default: load it)
    dt      the fixed step, s (default 1/240: 60, 48 and 30 frames a second are whole steps)
    wind    a Wind; or a NED 3-tuple; or a callable (t, n, e, d) -> (wn, we, wd); or None.
            Sampled once per step and held across it.
    ground  ground(n, e) -> height above the NED origin (default flat at 0)
    seed    the propwash's and a crash's random numbers

    ``u`` is the motors' commands, each a share of its full thrust, which the next step() uses; a
    FlightController writes it. Per motor, with T = thrust_to_weight m g / 4 its full static thrust,
    s its speed as a share of full, v the air's speed along the thrust axis:

        ds/dt   toward sqrt(u) with the time constant motor_up or motor_down
        thrust  T s^2 (1 - sag (1 - charge)) clamp(1 - v / (pitch_speed max(s, 0.15)), 0, 1.2) (1 - loss wash)

    The drag is -(cd s + k) u forward and sideways, -cd s u up the thrust axis, with k = rotor_drag
    sqrt(thrust / weight), u the air-relative velocity in body axes, s its size. ``x`` is the state
    (m350_rig's STATE: the rotors in rad/s), ``thrust`` the four rotors' now, ``charge`` the pack,
    ``wash`` how far into its own wash (0..1). ``parked``: on the pad, no physics until release().
    ``crashed``: the motors stopped, it falls and tumbles until a set_state or place_on_ground."""

    def __init__(self, spec=None, dt=1.0 / 240.0, wind=None, ground=None, seed=1):
        self.spec = spec = load_spec() if spec is None else spec
        AF, PU, AE, PA, WA, CO = (spec["airframe"], spec["propulsion"], spec["aero"], spec["pack"], spec["propwash"],
                                  spec["contact"])
        self.dt = dt
        self.g = spec["atmosphere"]["g"]
        self.wind = wind
        self.ground = _plain(ground)
        self.m = AF["mass_kg"]
        self.weight = self.m * self.g
        self.twr = AF["thrust_to_weight"]
        self.full = 0.25 * self.twr * self.weight          # N, a rotor at full speed
        self.arm, self.yaw_arm = AF["arm_m"], AF["yaw_arm_m"]
        I = AF["inertia_kgm2"]
        self.Ixx, self.Iyy, self.Izz = I["roll"], I["pitch"], I["yaw"]
        self.w_max = TWO_PI * PU["max_rev_s"]
        self.tau_up, self.tau_down = PU["motor_up_s"], PU["motor_down_s"]
        self.idle, self.pitch_speed = PU["idle"], PU["pitch_speed_ms"]
        d = AE["drag_kg_m"]
        self.cd = (d["front"], d["side"], d["top"])
        self.rotor_drag, self.rate_damp = AE["rotor_drag_ns_m"], AE["rate_damp_nms"]
        self.sag, self.pack_s = PA["sag"], PA["pack_s"]
        self.wash_speed, self.wash_span, self.wash_clear = WA["wash_speed_ms"], WA["wash_span_ms"], WA["wash_clear_ms"]
        self.wash_loss, self.wash_shake = WA["loss"], WA["shake_rads2"]
        self.radius, self.bounce, self.scrape, self.crash_speed = (CO["radius_m"], CO["bounce"], CO["scrape"],
                                                                   CO["crash_speed_ms"])
        self.rotors = tuple((sx, sy, float(spec["layout"]["rotors"][k]["spin"]))
                            for (sx, sy), k in zip(SIDES, ROTORS))
        self.rng = random.Random(seed)
        self.x = [0.0] * len(STATE)
        self.x[6] = 1.0
        self.u = [0.0, 0.0, 0.0, 0.0]
        self.t = 0.0
        self.steps = 0
        self.thrust = 0.0
        self.charge = 1.0
        self.wash = 0.0
        self.shake = (0.0, 0.0)
        self.parked = False
        self.crashed = False
        self.crash_t = 0.0
        self.grounded = False
        self.hit = 0.0                                     # m/s: the hardest contact since take_hit
        self._w = (0.0, 0.0, 0.0)

    # ---- setting a state
    def hover_share(self):
        """The share of full thrust that carries the weight."""
        return 1.0 / self.twr

    def set_state(self, n=0.0, e=0.0, h=0.0, yaw=0.0, roll=0.0, pitch=0.0, velocity=(0.0, 0.0, 0.0),
                  rates=(0.0, 0.0, 0.0)):
        """In the air at this state, the rotors at the hover's speed. The pack keeps what it has."""
        q = quat_wxyz(rot_nb(roll, pitch, yaw))
        w = math.sqrt(self.hover_share()) * self.w_max
        self.x = [n, e, -h, velocity[0], velocity[1], velocity[2], q[0], q[1], q[2], q[3],
                  rates[0], rates[1], rates[2], w, w, w, w]
        self.u = [self.hover_share()] * 4
        self.thrust = self.weight
        self.parked = self.crashed = self.grounded = False
        self.crash_t = 0.0
        self.shake = (0.0, 0.0)

    def place_on_ground(self, n=0.0, e=0.0, yaw=0.0):
        """Parked on its pad at (n, e), level, a full pack, the motors idling, until release()."""
        self.set_state(n, e, self.ground(n, e) + self.radius, yaw)
        w = math.sqrt(self.idle) * self.w_max
        self.x[13:17] = [w, w, w, w]
        self.u = [self.idle] * 4
        self.thrust = 0.0
        self.charge = 1.0
        self.parked = self.grounded = True

    def release(self):
        self.parked = False

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
        """The angle between the thrust axis and the vertical, rad."""
        x = self.x
        return math.acos(clamp(1.0 - 2.0 * (x[7] * x[7] + x[8] * x[8]), -1.0, 1.0))

    def speed(self):
        x = self.x
        return math.sqrt(x[3] * x[3] + x[4] * x[4] + x[5] * x[5])

    def rev_s(self):
        """The rotors' mean speed, rev/s."""
        return sum(self.x[13:17]) / (4.0 * TWO_PI)

    def height_above_ground(self):
        """The centre's height above the ground, m."""
        x = self.x
        return -x[2] - self.ground(x[0], x[1])

    def wind_ned(self):
        """The wind the last step used at the aircraft, NED m/s."""
        return self._w

    def world_position(self, h0=0.0):
        """The centre of gravity in the world."""
        return ned_to_world(self.x[0], self.x[1], self.x[2], h0)

    def world_rotation(self):
        """The .glb's rotation in the world: model -> body -> NED -> world."""
        return C_WN @ self.rotation() @ C_BM

    def take_hit(self):
        """The hardest contact since the last take, m/s along its normal."""
        h, self.hit = self.hit, 0.0
        return h

    # ---- the air
    def aero(self):
        """The air on it, thrust aside, NED, N: the frame's quadratic drag on each body axis and
        the rotors' across their discs, linear in the speed and growing with the thrust."""
        x, w = self.x, self._w
        an, ae, ad = x[3] - w[0], x[4] - w[1], x[5] - w[2]
        qw, qx, qy, qz = x[6], x[7], x[8], x[9]
        r00, r01, r02 = 1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)
        r10, r11, r12 = 2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)
        r20, r21, r22 = 2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)
        ub = r00 * an + r10 * ae + r20 * ad
        vb = r01 * an + r11 * ae + r21 * ad
        wb = r02 * an + r12 * ae + r22 * ad
        s = math.sqrt(an * an + ae * ae + ad * ad)
        k = self.rotor_drag * math.sqrt(max(self.thrust, 0.0) / self.weight)
        cf, cs, ct = self.cd
        fx, fy, fz = -(cf * s + k) * ub, -(cs * s + k) * vb, -ct * s * wb
        return (r00 * fx + r01 * fy + r02 * fz, r10 * fx + r11 * fy + r12 * fz, r20 * fx + r21 * fy + r22 * fz)

    def unloaded(self):
        """The share of a rotor's static thrust it has at this inflow and pack, at the rotors' mean speed."""
        x, w = self.x, self._w
        qw, qx, qy, qz = x[6], x[7], x[8], x[9]
        un, ue, ud = -2.0 * (qx * qz + qw * qy), -2.0 * (qy * qz - qw * qx), 2.0 * (qx * qx + qy * qy) - 1.0
        va = (x[3] - w[0]) * un + (x[4] - w[1]) * ue + (x[5] - w[2]) * ud
        s = sum(x[13:17]) / (4.0 * self.w_max)
        inflow = clamp01(1.0 - va / (self.pitch_speed * max(s, 0.15)))
        return max(inflow * (1.0 - self.sag * (1.0 - self.charge)), 0.2)

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

    # ---- stepping
    def step(self):
        """One fixed step: the motors, the air, the body, what it hits."""
        self._w = self._sample_wind()
        self.t += self.dt
        self.steps += 1
        if self.parked:
            return
        if self.crashed:
            self.crash_t += self.dt
            self.u = [0.0, 0.0, 0.0, 0.0]
        self._integrate(self.dt)
        self._collide()

    def _integrate(self, dt):
        x, wind = self.x, self._w
        qw, qx, qy, qz = x[6], x[7], x[8], x[9]
        un, ue, ud = -2.0 * (qx * qz + qw * qy), -2.0 * (qy * qz - qw * qx), 2.0 * (qx * qx + qy * qy) - 1.0
        an, ae, ad = x[3] - wind[0], x[4] - wind[1], x[5] - wind[2]
        va = an * un + ae * ue + ad * ud                   # + = air into the discs from above, flying up the thrust axis
        bn, be, bd = an - un * va, ae - ue * va, ad - ud * va
        across = math.sqrt(bn * bn + be * be + bd * bd)
        self.wash = wash = (clamp01((-va - self.wash_speed) / self.wash_span)
                            * clamp01(1.0 - across / self.wash_clear))
        pack = 1.0 - self.sag * (1.0 - self.charge)
        full, ws, arm, yarm = self.full, self.w_max, self.arm, self.yaw_arm
        keep = 1.0 - self.wash_loss * wash
        total = mx = my = mz = 0.0
        for i, (sx, sy, sp) in enumerate(self.rotors):
            to = math.sqrt(clamp01(self.u[i]))             # the ESC: thrust-linear commands
            s = x[13 + i] / ws
            s += (to - s) * (1.0 - math.exp(-dt / (self.tau_up if to > s else self.tau_down)))
            x[13 + i] = s * ws
            load = clamp(1.0 - va / (self.pitch_speed * max(s, 0.15)), 0.0, 1.2)
            f = full * s * s * pack * load * keep
            total += f
            mx -= sy * arm * f
            my += sx * arm * f
            mz += sp * yarm * f
        self.thrust = total
        self.charge = max(0.0, self.charge - (total / (4.0 * full)) ** 1.1 * dt / self.pack_s)
        # propwash: the rotors in their own turbulence, a shake that does not average out in a frame
        if wash > 0.0:
            sk = wash * self.wash_shake
            k = 1.0 - math.exp(-dt * 40.0)
            sq, sp_ = self.shake
            self.shake = sh = (sq + (self.rng.uniform(-sk, sk) - sq) * k, sp_ + (self.rng.uniform(-sk, sk) - sp_) * k)
        else:
            self.shake = sh = (0.0, 0.0)
        p, q, r = x[10], x[11], x[12]
        damp, Ixx, Iyy, Izz = self.rate_damp, self.Ixx, self.Iyy, self.Izz
        mx -= damp * p
        my -= damp * q
        mz -= damp * r
        hx, hy, hz = Ixx * p, Iyy * q, Izz * r
        pn = p + ((mx - (q * hz - r * hy)) / Ixx + sh[1]) * dt
        qn = q + ((my - (r * hx - p * hz)) / Iyy + sh[0]) * dt
        rn = r + (mz - (p * hy - q * hx)) / Izz * dt
        x[10], x[11], x[12] = pn, qn, rn
        fn, fe, fd = self.aero()
        k = total / self.m
        im = 1.0 / self.m
        vn, ve, vd = (x[3] + (un * k + fn * im) * dt, x[4] + (ue * k + fe * im) * dt,
                      x[5] + (ud * k + fd * im + self.g) * dt)
        x[3], x[4], x[5] = vn, ve, vd
        x[0] += vn * dt
        x[1] += ve * dt
        x[2] += vd * dt
        h = 0.5 * dt
        qw1 = qw + (-qx * pn - qy * qn - qz * rn) * h
        qx1 = qx + (qw * pn + qy * rn - qz * qn) * h
        qy1 = qy + (qw * qn + qz * pn - qx * rn) * h
        qz1 = qz + (qw * rn + qx * qn - qy * pn) * h
        k = 1.0 / math.sqrt(qw1 * qw1 + qx1 * qx1 + qy1 * qy1 + qz1 * qz1)
        x[6], x[7], x[8], x[9] = qw1 * k, qx1 * k, qy1 * k, qz1 * k

    # ---- what it hits
    def _collide(self):
        x = self.x
        self.grounded = False
        h = self.ground(x[0], x[1])
        if -x[2] - self.radius < h:
            nn, ne, nd = ground_normal(self.ground, x[0], x[1]) if self.ground is not flat_ground else (0.0, 0.0, -1.0)
            self._contact(nn, ne, nd, (h + self.radius + x[2]) * -nd)
            # lying on its side on the ground: that is a crash too
            qx, qy = x[7], x[8]
            if not self.crashed and 1.0 - 2.0 * (qx * qx + qy * qy) < 0.25 and self.speed() < 2.0:
                self._crash()

    def _contact(self, nn, ne, nd, depth):
        x = self.x
        x[0] += nn * depth
        x[1] += ne * depth
        x[2] += nd * depth
        vn = x[3] * nn + x[4] * ne + x[5] * nd
        if vn >= 0.0:
            return
        impact = -vn
        tn, te, td = x[3] - nn * vn, x[4] - ne * vn, x[5] - nd * vn
        k = 1.0 - self.scrape
        x[3], x[4], x[5] = (tn * k + nn * impact * self.bounce, te * k + ne * impact * self.bounce,
                            td * k + nd * impact * self.bounce)
        self.hit = max(self.hit, impact)
        # the strike turns the airframe: about the axis across its motion
        s = math.sqrt(tn * tn + te * te + td * td)
        tu = (tn / s, te / s, td / s) if s > 1e-9 else (0.0, 1.0, 0.0)
        axis = self.rotation().T @ np.array((ne * tu[2] - nd * tu[1], nd * tu[0] - nn * tu[2], nn * tu[1] - ne * tu[0]))
        k = impact * 0.9
        x[10] += float(axis[0]) * k
        x[11] += float(axis[1]) * k
        x[12] += float(axis[2]) * k
        if impact > self.crash_speed:
            self._crash()
        else:
            self.grounded = True

    def _crash(self):
        if self.crashed:
            return
        self.crashed = True
        self.crash_t = 0.0
        self.thrust = 0.0
        self.u = [0.0, 0.0, 0.0, 0.0]
        x, r = self.x, self.rng
        x[13:17] = [0.0, 0.0, 0.0, 0.0]
        x[10] += r.uniform(-9.0, 9.0)
        x[11] += r.uniform(-9.0, 9.0)
        x[12] += r.uniform(-6.0, 6.0)


# --------------------------------------------------------------------------- #
#  The flight controller: OURS
# --------------------------------------------------------------------------- #
@dataclass
class Sticks:
    """One frame's inputs. throttle: ACRO, ANGLE the collective's stick, 0..1; ARCADE the share of
    top speed. brake: ARCADE, 0..1. yaw: -1..1, + = right. pitch: ACRO, ANGLE + = nose down; ARCADE
    climb, + = up. roll: ACRO, ANGLE + = roll right; ARCADE sideways, + = right. boost: ARCADE."""
    throttle: float = 0.0
    brake: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    boost: bool = False


class FlightController:
    """The layers above the motors, top down, each assist entering at its own:

        "auto"    the line: a point ahead on it, the speed its profile gives -> a velocity
        "arcade"  the sticks as a heading, a speed along it and a height over the ground -> a velocity
        velocity  -> a thrust vector, drag and gravity included -> an up axis and a collective
        "angle"   the sticks as a tilt, the throttle as the collective -> an up axis and a collective
        attitude  -> body rates
        "acro"    the sticks as body rates (a radio's rate curve), the throttle as the collective
        rate PI   -> torque demands, as a flight controller's rate loop
        mixer     -> a command per motor, airmode: the torques first, the collective shifted to make
                     room, so authority is what the thrust leaves: a roll at full throttle takes
                     thrust away, a flip at zero throttle adds it

    ``sticks`` is read by update(), which writes the Racer's motor commands; call it once before
    each step(). ``assist`` may change between updates. It is fed the EXACT state: no estimator,
    no filter stack, no sensor noise, and the true wind in the drag of its velocity layer.
    ``psi`` is the heading it flies to: the body's plane of symmetry holds that direction, so at a
    racing lean the nose points down it, well away from psi by Euler's angles, with a few degrees
    of sideslip.
    reset() after a set_state or a place_on_ground."""

    def __init__(self, racer, assist="acro", line=None):
        self.m = racer
        self.assist = assist
        self.line = line
        c = racer.spec["control"]
        R, RA, A, V, AR, L = c["rate"], c["radio"], c["attitude"], c["velocity"], c["arcade"], c["line"]
        self.kp_r, self.ki_r, self.kp_y, self.ki_y, self.i_max = (R["kp"], R["ki"], R["kp_yaw"], R["ki_yaw"],
                                                                  R["integral_rads"])
        self.r_center, self.r_max, self.r_yaw, self.th_power = (RA["center_rads"], RA["max_rads"], RA["yaw_max_rads"],
                                                                RA["throttle_power"])
        self.kp_att, self.att_max, self.angle_max = A["kp"], A["rate_max_rads"], A["angle_max_rad"]
        self.V, self.AR = V, AR
        self.look_min, self.look_t, self.pace = L["look_min_m"], L["look_t_s"], L["pace"]
        self.sticks = Sticks()
        self.reset()

    def reset(self):
        """Setpoints at the aircraft's own heading, speed and height; the integral emptied."""
        m = self.m
        self.psi = m.euler()[2]
        self.speed_cmd = 0.0
        self.hold = max(self.AR["hold_start_m"], m.height_above_ground())
        self.rate_cmd = (0.0, 0.0, 0.0)                    # p, q, r
        self.rate_int = [0.0, 0.0, 0.0]
        self.share = 0.0 if m.parked else m.hover_share()
        self.i = -1                                        # its sample of the line

    def update(self):
        m = self.m
        if m.parked or m.crashed:
            return
        a = self.assist
        if a == "acro":
            self._acro()
        elif a == "angle":
            self._angle()
        elif a == "auto":
            self._auto()
        else:
            self._arcade()
        self._mix()

    def _collective(self):
        return clamp01(self.sticks.throttle) ** self.th_power

    def _acro(self):
        s, c, f = self.sticks, self.r_center, self.r_max
        self.rate_cmd = (stick_rate(s.roll, c, f), -stick_rate(s.pitch, c, f), stick_rate(s.yaw, c * 0.8, self.r_yaw))
        self.share = self._collective()

    def _angle(self):
        # the stick tilts the up axis about the heading the yaw stick moves; the throttle is the collective
        m, s = self.m, self.sticks
        x = m.x
        qw, qx, qy, qz = x[6], x[7], x[8], x[9]
        fn, fe = 1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz)
        psi = math.atan2(fe, fn) if math.hypot(fn, fe) > 0.15 else self.psi
        self.psi = psi
        a, b = math.tan(clamp(s.pitch, -1.0, 1.0) * self.angle_max), math.tan(clamp(s.roll, -1.0, 1.0) * self.angle_max)
        cp, sp = math.cos(psi), math.sin(psi)
        un, ue, uu = cp * a - sp * b, sp * a + cp * b, 1.0
        k = 1.0 / math.sqrt(un * un + ue * ue + uu * uu)
        self._attitude(un * k, ue * k, -uu * k, psi)
        p, q, r = self.rate_cmd
        self.rate_cmd = (p, q, r + stick_rate(s.yaw, self.r_center * 0.8, self.r_yaw))
        self.share = self._collective()

    def _arcade(self):
        m, s, K = self.m, self.sticks, self.AR
        x, dt = m.x, m.dt
        vn, ve = x[3], x[4]
        hs = math.hypot(vn, ve)
        self.psi += s.yaw * min(K["turn_rate_rads"], K["turn_accel_ms2"] / max(hs, 1.0)) * dt
        top = K["v_boost_ms"] if s.boost else K["v_max_ms"]
        if s.brake > 0.02:
            self.speed_cmd = approach(self.speed_cmd, -K["v_back_ms"], K["brake_ms2"] * s.brake * dt)
        elif s.throttle > 0.02 and self.speed_cmd < s.throttle * top:
            self.speed_cmd = approach(self.speed_cmd, s.throttle * top, K["accel_ms2"] * dt)
        else:
            self.speed_cmd = approach(self.speed_cmd, 0.0, K["coast_ms2"] * dt)
        cp, sp = math.cos(self.psi), math.sin(self.psi)
        # the height: released, the stick holds the height over the ground; held, it moves that height
        ahead = 0.45 * hs / max(hs, 1.0)
        g = max(m.ground(x[0], x[1]), m.ground(x[0] + vn * ahead, x[1] + ve * ahead))
        alt = -x[2]
        if abs(s.pitch) > 0.04:
            vy = s.pitch * K["climb_ms"]
            self.hold = clamp(alt - g, K["hold_min_m"], K["ceiling_m"])
        else:
            lim = 1.3 * K["climb_ms"]
            vy = clamp((g + self.hold - alt) * K["hold_gain"], -lim, lim)
        side = s.roll * K["strafe_ms"]
        self._track_velocity(cp * self.speed_cmd - sp * side, sp * self.speed_cmd + cp * side, vy, self.psi,
                             self.V["tilt_arcade_rad"], self.V["kp_arcade"])

    def _auto(self):
        m, L = self.m, self.line
        x = m.x
        self.i = L.follow(self.i, x[0], x[1], 6, 40)
        sp = m.speed()
        j = (self.i + int(max(self.look_min, self.look_t * sp) / L.step + 0.5)) % L.size
        # the speed: the line's profile a little ahead, so the braking starts before the corner does
        j2 = (self.i + int(0.15 * sp / L.step + 0.5)) % L.size
        vs = min(max(L.v[j2] * self.pace, 6.0), sp + 10.0)    # off the pad: no leap at the line
        dn, de, dh = L.n[j] - x[0], L.e[j] - x[1], L.h[j] + x[2]
        d = math.sqrt(dn * dn + de * de + dh * dh)
        if d > 1e-9:
            dn, de, dh = dn / d, de / d, dh / d
        else:
            dn, de, dh = math.cos(self.psi), math.sin(self.psi), 0.0
        vn, ve, vh = dn * vs, de * vs, dh * vs
        if math.hypot(vn, ve) > 0.5:
            self.psi = math.atan2(ve, vn)
        self._track_velocity(vn, ve, vh, self.psi, self.V["tilt_auto_rad"], self.V["kp_auto"])

    def _track_velocity(self, vcn, vce, vch, psi, tilt_max, gain):
        """A velocity (north, east, up) to an up axis and a collective."""
        m, V = self.m, self.V
        x, g = m.x, m.g
        an, ae, au = (vcn - x[3]) * gain, (vce - x[4]) * gain, (vch + x[5]) * gain
        ah = math.hypot(an, ae)
        if ah > V["accel_h_ms2"]:
            k = V["accel_h_ms2"] / ah
            an, ae = an * k, ae * k
        au = clamp(au, -0.85 * g, V["accel_up_ms2"])
        fn, fe, fd = m.aero()
        tn, te, tu = an * m.m - fn, ae * m.m - fe, (au + g) * m.m + fd
        tu = max(tu, 0.15 * m.weight)
        # the tilt's limit: the horizontal part cut to what that tilt allows at this lift
        h, lim = math.hypot(tn, te), tu * math.tan(tilt_max)
        if h > lim:
            k = lim / h
            tn, te = tn * k, te * k
        t = math.sqrt(tn * tn + te * te + tu * tu)
        self.share = clamp01(t / (m.twr * m.weight * m.unloaded()))     # what the props can give at this speed
        self._attitude(tn / t, te / t, -tu / t, psi)

    def _attitude(self, un, ue, ud, psi):
        """Body rates that bring the up axis (NED) and the heading to the aircraft."""
        fn, fe = math.cos(psi), math.sin(psi)
        ln, le, ld = -ud * fe, ud * fn, un * fe - ue * fn  # up x forward: the left axis
        k = math.sqrt(ln * ln + le * le + ld * ld)
        if k > 1e-9:
            ln, le, ld = ln / k, le / k, ld / k
        else:
            ln, le, ld = fe, -fn, 0.0
        bn, be, bd = le * ud - ld * ue, ld * un - ln * ud, ln * ue - le * un     # left x up: forward
        dw, dx, dy, dz = quat_of_axes(bn, be, bd, -ln, -le, -ld, -un, -ue, -ud)
        x = self.m.x
        qw, qx, qy, qz = x[6], x[7], x[8], x[9]
        # the rotation from the attitude it has to the one wanted, in body axes: conj(q) x q_d
        ew = qw * dw + qx * dx + qy * dy + qz * dz
        ex = qw * dx - qx * dw - qy * dz + qz * dy
        ey = qw * dy + qx * dz - qy * dw - qz * dx
        ez = qw * dz - qx * dy + qy * dx - qz * dw
        if ew < 0.0:
            ex, ey, ez = -ex, -ey, -ez
        k = 2.0 * self.kp_att
        p, q, r = ex * k, ey * k, ez * k
        n = math.sqrt(p * p + q * q + r * r)
        if n > self.att_max:
            k = self.att_max / n
            p, q, r = p * k, q * k, r * k
        self.rate_cmd = (p, q, r)

    def _mix(self):
        m = self.m
        x, dt, lim = m.x, m.dt, self.i_max
        ep, eq, er = self.rate_cmd[0] - x[10], self.rate_cmd[1] - x[11], self.rate_cmd[2] - x[12]
        ri = self.rate_int
        ri[0] = clamp(ri[0] + ep * dt, -lim, lim)
        ri[1] = clamp(ri[1] + eq * dt, -lim, lim)
        ri[2] = clamp(ri[2] + er * dt, -lim, lim)
        full = m.full
        scale = 1.0 / (4.0 * m.arm * full)
        roll = m.Ixx * (self.kp_r * ep + self.ki_r * ri[0]) * scale
        pitch = m.Iyy * (self.kp_r * eq + self.ki_r * ri[1]) * scale
        yaw = m.Izz * (self.kp_y * er + self.ki_y * ri[2]) / (4.0 * m.yaw_arm * full)
        # the motors' span, floor to full: roll and pitch first, yaw from what they leave, as a mixer's priorities
        idle = m.idle
        half = 0.5 * (1.0 - idle)
        rp = abs(roll) + abs(pitch)
        if rp > half:
            roll, pitch = roll * half / rp, pitch * half / rp
        left = min(half - min(rp, half), 0.3)
        yaw = clamp(yaw, -left, left)
        u = [self.share - sy * roll + sx * pitch + sp * yaw for sx, sy, sp in m.rotors]
        hi, lo = max(u), min(u)
        # airmode: the whole set shifted to fit, so the torques stand at either end of the throttle
        shift = 1.0 - hi if hi > 1.0 else idle - lo if lo < idle else 0.0
        m.u = [clamp01(a + shift) for a in u]


# --------------------------------------------------------------------------- #
#  The line
# --------------------------------------------------------------------------- #
class Line:
    """A closed flown line through points (n, e, h), a closed Catmull-Rom curve sampled ``step``
    metres apart, and the speed it allows: what its curvature in three dimensions allows inside
    the spec's lateral acceleration, a pass back for the braking and one forward for the power.
    ``n``, ``e``, ``h``, ``v``: the samples, plain lists; ``size`` their number, ``length`` metres."""

    def __init__(self, points, step=1.0, spec=None):
        cfg = (spec or load_spec())["control"]["line"]
        c = self._resample(np.asarray(points, np.float64), step)
        self.size = size = len(c)
        self.length = size * step
        self.step = step
        k = 7
        curv = np.linalg.norm(np.roll(c, -k, 0) - 2.0 * c + np.roll(c, k, 0), axis=1) / (k * step) ** 2
        curv = np.convolve(np.concatenate([curv[-3:], curv, curv[:3]]), np.ones(7) / 7.0, "valid")
        v = np.minimum(np.sqrt(cfg["lateral_ms2"] / np.maximum(curv, 1e-5)), cfg["top_ms"])
        for _ in range(2):                                 # twice round: the closed line's seam settles
            for i in range(size - 1, -1, -1):
                v[i] = min(v[i], math.sqrt(v[(i + 1) % size] ** 2 + 2.0 * cfg["brake_ms2"] * step))
            for i in range(size):
                v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2.0 * cfg["gain_ms2"] * step))
        self.n, self.e, self.h, self.v = c[:, 0].tolist(), c[:, 1].tolist(), c[:, 2].tolist(), v.tolist()

    @staticmethod
    def _resample(p, step):
        t = np.linspace(0.0, 1.0, 48, endpoint=False)[:, None]
        p0, p1, p2, p3 = np.roll(p, 1, 0), p, np.roll(p, -1, 0), np.roll(p, -2, 0)
        seg = [0.5 * (2.0 * a + (c - z) * t + (2.0 * z - 5.0 * a + 4.0 * c - d) * t * t
                      + (3.0 * a - z - 3.0 * c + d) * t * t * t) for z, a, c, d in zip(p0, p1, p2, p3)]
        fine = np.concatenate(seg)
        fine = np.concatenate([fine, fine[:1]])
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(fine, axis=0), axis=1))])
        count = int(round(s[-1] / step))
        at = np.linspace(0.0, s[-1], count, endpoint=False)
        return np.stack([np.interp(at, s, fine[:, k]) for k in range(3)], -1)

    def follow(self, last, pn, pe, back, ahead):
        """The sample nearest (pn, pe), searched from ``last`` over back and ahead samples."""
        n, e, size = self.n, self.e, self.size
        best, bd = -1, 1e300
        if last < 0:
            for i in range(size):
                dn, de = n[i] - pn, e[i] - pe
                d = dn * dn + de * de
                if d < bd:
                    bd, best = d, i
            return best
        for k in range(-back, ahead + 1):
            i = (last + k) % size
            dn, de = n[i] - pn, e[i] - pe
            d = dn * dn + de * de
            if d < bd:
                bd, best = d, i
        return self.follow(-1, pn, pe, 0, 0) if bd > 2500.0 else best

    def lap_time(self):
        """The profile's own time round the line, s."""
        return sum(self.step / max(a, 1e-3) for a in self.v)


# --------------------------------------------------------------------------- #
#  The 3D model
# --------------------------------------------------------------------------- #
class Visual:
    """racer.glb's nodes posed from a Racer.

    root  the loaded model's scene node (tp.GLTFLoader().load(model_path()).scene)
    spec  racer_spec.json as a dict (default: load it)
    tp    the threepp module: with it the modelled blades are hidden and each rotor is drawn as
          unlit transparent rings and ghost blades, because a real blade at this rate is sharp and
          frozen under motion blur. Those materials are the combination the Vulkan renderer draws on
          its raster overlay, after the tone map and never traced, so they cost no rays and do not
          smear in its TAA, and their colours are display values. Without tp the blades stay.

    pose(racer, h0) places the root at the state (world = (E, h0 - D, -N)) and turns the motors'
    bells, at most model.visible_rev_s (faster only aliases)."""

    SPINNING = ("bell", "vents", "prop_hub", "spinner_base", "spinner")

    def __init__(self, root, spec=None, tp=None):
        self.root = root
        spec = spec or load_spec()
        self.model = spec["model"]
        self.spinners, self.pivots = [], []
        parts = self.SPINNING if tp is not None else self.SPINNING + ("prop_blades",)
        rings = self._rings(tp) if tp is not None else None
        for k in ROTORS:
            rotor = spec["layout"]["rotors"][k]
            for part in parts:
                node = root.get_object_by_name(f"{part}_{k}")
                if node is None:
                    raise KeyError(f"racer.glb has no node {part}_{k}: rebuild it with uav/build_racer_blender.py")
                self.spinners.append((node, float(rotor["spin"])))
            if tp is not None:
                root.get_object_by_name(f"prop_blades_{k}").visible = False
                self.pivots.append((self._mount(tp, rings, rotor), float(rotor["spin"])))
        self.spin = 0.0
        self._t = None

    def _rings(self, tp):
        """The rings' and the ghost's geometries and materials, shared by the four rotors."""
        D = self.model["discs"]
        radius, bell = self.model["prop_radius_m"], self.model["bell_radius_m"]
        n = D["rings"]
        edges = np.linspace(0.0, radius, n + 1)
        mids = 0.5 * (edges[:-1] + edges[1:])
        s = mids / radius

        def smooth(e0, e1, v):
            t = np.clip((v - e0) / (e1 - e0), 0.0, 1.0)
            return t * t * (3.0 - 2.0 * t)
        prof = (1.0 - D["falloff"] * s) * (1.0 - smooth(0.80, 1.0, s)) * (0.35 + 0.65 * smooth(0.7 * bell, 1.6 * bell, mids))
        c0, c1 = tp.Color(int(D["color"], 16)), tp.Color(int(D["rim_color"], 16))
        out = []
        for k in range(n):
            m = self._overlay(tp, int(D["color"], 16), D["opacity"] * float(prof[k]))
            m.color = c0.clone().lerp(c1, float(s[k]))
            out.append((self._annulus(tp, float(edges[k]), float(edges[k + 1])), m))
        g = D["ghosts"]
        ghost = self._blades(tp, 0.0054, radius * 0.97)
        mats = [self._overlay(tp, int(D["ghost_color"], 16), D["ghost_opacity"] * (1.0 - c / g) ** 1.5) for c in range(g)]
        return out, ghost, mats

    def _mount(self, tp, rings, rotor):
        spin = float(rotor["spin"])
        out, ghost, mats = rings
        node = tp.Group()
        node.position.set(*(float(a) for a in rotor["pos"]))
        self.root.add(node)
        for geo, mat in out:
            node.add(tp.Mesh(geo, mat))
        pivot = tp.Group()
        node.add(pivot)
        arc = math.radians(80.0)
        for c, mat in enumerate(mats):
            mesh = tp.Mesh(ghost, mat)
            mesh.rotation.y = -spin * arc * c / (len(mats) - 1)
            pivot.add(mesh)
        return pivot

    @staticmethod
    def _overlay(tp, color, opacity):
        m = tp.MeshBasicMaterial()
        m.color = tp.Color(color)
        m.transparent = True
        m.opacity = float(opacity)
        m.depth_write = False
        m.side = tp.Side.Double
        return m

    @staticmethod
    def _flat(tp, pos, idx):
        g = tp.BufferGeometry()
        g.set_attribute("position", np.ascontiguousarray(pos, dtype=np.float32))
        nrm = np.zeros_like(pos, dtype=np.float32)
        nrm[:, 1] = 1.0
        g.set_attribute("normal", nrm)
        g.set_index(np.ascontiguousarray(idx.reshape(-1), dtype=np.uint32))
        return g

    @classmethod
    def _annulus(cls, tp, r0, r1, seg=64):
        a = np.linspace(0.0, TWO_PI, seg, endpoint=False)
        c, s = np.cos(a), np.sin(a)
        pos = np.concatenate([np.stack([r0 * c, 0 * c, r0 * s], -1), np.stack([r1 * c, 0 * c, r1 * s], -1)])
        i = np.arange(seg)
        j = (i + 1) % seg
        return cls._flat(tp, pos, np.stack([np.stack([i, seg + i, seg + j], -1), np.stack([i, seg + j, j], -1)], 1))

    @classmethod
    def _blades(cls, tp, r0, r1, seg=10):
        """The outline of a two-blade propeller in the XZ plane: chord tapering root to tip, a little sweep."""
        t = np.linspace(0.0, 1.0, seg + 1)
        r = r0 + (r1 - r0) * t
        chord = r1 * (0.18 * (1 - t) + 0.09 * t)
        sweep = r1 * 0.045 * t * t
        pos, idx = [], []
        for b in range(2):
            ca, sa = math.cos(math.pi * b), math.sin(math.pi * b)
            for edge in (np.stack([r, -0.5 * chord - sweep], -1), np.stack([r, 0.5 * chord - sweep], -1)):
                x, z = edge[:, 0] * ca - edge[:, 1] * sa, edge[:, 0] * sa + edge[:, 1] * ca
                pos.append(np.stack([x, np.zeros_like(x), z], -1))
            base = b * 2 * (seg + 1)
            k = np.arange(seg)
            lo, hi = base + k, base + (seg + 1) + k
            idx.append(np.stack([np.stack([lo, hi, hi + 1], -1), np.stack([lo, hi + 1, lo + 1], -1)], 1))
        return cls._flat(tp, np.concatenate(pos), np.concatenate(idx))

    def pose(self, m, h0=0.0):
        pos = m.world_position(h0)
        self.root.position.set(float(pos[0]), float(pos[1]), float(pos[2]))
        self.root.quaternion.set(*(float(a) for a in quat_of(m.world_rotation())))
        dt = 0.0 if self._t is None else m.t - self._t
        self._t = m.t
        self.spin = (self.spin + TWO_PI * min(m.rev_s(), self.model["visible_rev_s"]) * dt) % TWO_PI
        for node, s in self.spinners:
            node.rotation.y = s * self.spin
        ghost = TWO_PI * self.model["discs"]["ghost_rev_s"] * dt
        for pivot, s in self.pivots:
            pivot.rotation.y = pivot.rotation.y + s * ghost
