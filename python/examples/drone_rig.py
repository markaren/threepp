"""
drone_rig -- a survey quadcopter, an authored flight route and a follow camera.

Three things, and nothing scene-specific:

  * ``Drone``       -- the 0.45 m X-quad: hull, four arm pods, spinning rotor
                       discs, a three-axis gimbal, nav lights and a strobe.
                       Attitude is DERIVED from the path, never authored.
  * ``Route``       -- an authored multi-leg flight (beziers with trapezoid or
                       linear speed warps) sampled densely, Hann-smoothed as a
                       signal, and then held above a rate-limited terrain /
                       canopy floor applied with a softplus rather than max().
  * ``FollowCamera``-- the third-person rig, precomputed over the whole route
                       and filtered the same way, with a land-vs-water probe
                       that decides whether the eye rides high or low.

Nothing here imports warp, and nothing reads a module global belonging to a
scene: every scene-specific quantity is a constructor or function parameter,
and the defaults are the values one finished film settled on. The one thing a
caller MUST supply for the clearance machinery to mean anything is a terrain
height field::

    ground(x, z) -> height in metres

which is called with numpy arrays and should return an array of the same shape
(a scalar-only callable is wrapped automatically, more slowly); a `GeoScene`
supplies ``geo.height_at``. Pass ``ground=None`` and everything clamps against
sea level.

WORLD FRAME
-----------
Metres, Y-up, x = east, z = -north, sea level y = 0. The drone's authoring
frame is forward = +Z at yaw 0, up = +Y, which is the same convention
``Object3D.look_at`` and ``tp.PerspectiveCamera`` use, so a pose that aims the
machine also aims anything parented to its gimbal.

TYPICAL USE
-----------
::

    from drone_rig import Drone, Leg, LegRoute, Route, FollowCamera, trap

    legs = LegRoute([
        Leg(8.0,  pos=(P0, C0, P1),      aim=AIM_A, warp=trap(0.45, 0.0)),
        Leg(21.5, pos=(P1, C1, P2),      aim=(AIM_A, AIM_B)),
        Leg(9.0,  pos=(P2, C2, C3, P3),  aim=AIM_B, warp=ramp(5.0, 1.0)),
    ])
    route = Route(legs, ground=geo.height_at)
    cam   = FollowCamera(route, ground=geo.height_at)

    drone = Drone(scene)
    ...
    p, a = route.pose(t)
    drone.set_pose(p, a, dt)
    drone.tick(dt, t, night=1.0)
    eye, aim = cam.pose(t)
    camera.position.set(*eye); camera.look_at(tp.Vector3(*aim))
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Appended, not inserted, so a caller run from elsewhere still gets its own
# modules first and `demo_common` below resolves either way.
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import threepp as tp
from demo_common import standard_material


# --------------------------------------------------------------------------- #
#  Small maths
# --------------------------------------------------------------------------- #
def smoothstep(a, b, x):
    """Vectorised smoothstep."""
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smoothstep_f(a, b, x):
    """Scalar smoothstep."""
    t = min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def yxz_matrix(ex, ey, ez):
    """three.js Euler order YXZ composes as Ry * Rx * Rz."""
    cx, sx = math.cos(ex), math.sin(ex)
    cy, sy = math.cos(ey), math.sin(ey)
    cz, sz = math.cos(ez), math.sin(ez)
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return ry @ rx @ rz


# --------------------------------------------------------------------------- #
#  The machine.
#
#  Authoring frame: forward = +Z at yaw 0, up = +Y.
# --------------------------------------------------------------------------- #
DRONE_SPAN = 0.45
DRONE_ARM = DRONE_SPAN * 0.5 * 0.7071
DRONE_G = 9.81
# The eye sits on the gimbal, forward of and below the hull, so an FPV shot can
# never see the machine it is flying on.
DRONE_EYE = np.array([0.0, -0.090, 0.300])


class Drone:
    """A 0.45 m survey quad. ``Drone(scene, span=0.45)``.

    The only call a shot makes is ``set_pose(pos, look_at, dt)``: attitude is
    derived from the motion, so the flight is authored purely as a path.
    """

    def __init__(self, scene, span=DRONE_SPAN, eye_offset=DRONE_EYE,
                 body_color=0x39404d, trim_color=0x59616e,
                 prop_color=0x0c0e11, lens_color=0x07090c,
                 led_colors=(0xff1408, 0x18ff3c), strobe_color=0xffffff,
                 start_pos=(0.0, 6.0, 14.0), start_look=(0.0, 3.0, 0.0)):
        self.span = float(span)
        self.arm = self.span * 0.5 * 0.7071
        k = self.span / DRONE_SPAN          # everything below is drawn at 0.45 m
        self.eye_offset = np.asarray(eye_offset, np.float64) * k

        self.root = tp.Group()
        self.root.rotation.order = tp.RotationOrder.YXZ
        scene.add(self.root)

        # Not black. At blue hour a 0x1b1f26 airframe against a treeline is a
        # hole in the picture; this is still dark grey but it catches the sky
        # and the rim light, so the hull and booms have form.
        body = standard_material(body_color, 0.38, 0.35)
        trim = standard_material(trim_color, 0.45)
        # Bare blades, no blur disc: at this span and filming distance a disc
        # reads as a grey plate bolted to the arm, while the spinning blades
        # alias into the slow apparent counter-rotation a filmed prop has.
        prop = standard_material(prop_color, 0.60, side=tp.Side.Double)
        lens = standard_material(lens_color, 0.05, 0.2)

        hull = tp.Mesh(tp.BoxGeometry(0.115 * k, 0.052 * k, 0.170 * k), body)
        hull.cast_shadow = True
        self.root.add(hull)
        canopy = tp.Mesh(tp.SphereGeometry(0.055 * k, 14, 10), trim)
        canopy.scale.set(1.0, 0.52, 1.30)
        canopy.position.set(0.0, 0.026 * k, 0.005 * k)
        self.root.add(canopy)
        for s in (1.0, -1.0):
            boom = tp.Mesh(tp.BoxGeometry(self.span, 0.015 * k, 0.021 * k), trim)
            boom.rotation.y = s * math.radians(45.0)
            boom.cast_shadow = True
            self.root.add(boom)

        self.props = []
        for sx, sz in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            px, pz = sx * self.arm, sz * self.arm
            pod = tp.Mesh(tp.CylinderGeometry(0.017 * k, 0.020 * k,
                                              0.032 * k, 10, 1), body)
            pod.position.set(px, 0.013 * k, pz)
            self.root.add(pod)
            hub = tp.Group()
            hub.position.set(px, 0.035 * k, pz)
            self.root.add(hub)
            blade = tp.Mesh(tp.BoxGeometry(0.126 * k, 0.0024 * k, 0.016 * k), prop)
            blade.rotation.z = math.radians(9.0)   # pitch, so it reads as a blade
            hub.add(blade)
            self.props.append((hub, 1.0 if sx * sz > 0 else -1.0))

        self.leds = []
        for col, x, z in ((led_colors[0], -1.02, 1.02),
                          (led_colors[1], 1.02, 1.02)):
            m = standard_material(0x0a0a0a, 1.0, emissive=col,
                                  emissive_intensity=0.0)
            s = tp.Mesh(tp.SphereGeometry(0.012 * k, 10, 8), m)
            s.position.set(x * self.arm, 0.004 * k, z * self.arm)
            self.root.add(s)
            self.leds.append(m)
        self.strobe_mat = standard_material(0x0a0a0a, 1.0, emissive=strobe_color,
                                            emissive_intensity=0.0)
        st = tp.Mesh(tp.SphereGeometry(0.013 * k, 10, 8), self.strobe_mat)
        st.position.set(0.0, -0.030 * k, -0.030 * k)
        self.root.add(st)

        self.gimbal = tp.Group()
        self.gimbal.position.set(0.0, -0.038 * k, 0.082 * k)
        self.root.add(self.gimbal)
        gb = tp.Mesh(tp.SphereGeometry(0.027 * k, 14, 10), body)
        gb.scale.set(1.0, 1.0, 0.85)
        self.gimbal.add(gb)
        gl = tp.Mesh(tp.CylinderGeometry(0.014 * k, 0.017 * k, 0.018 * k, 12, 1),
                     lens)
        gl.rotate_x(math.pi / 2)
        gl.position.set(0.0, 0.0, 0.022 * k)
        self.gimbal.add(gl)

        self.pos = np.array(start_pos, np.float64)
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.look = np.array(start_look, np.float64)
        self.yaw = self.pitch = self.roll = 0.0
        self.spin = self.spool = 0.0
        self.throttle = 0.5
        self.face_w = 0.0        # 0 = nose follows the track, 1 = nose on the work
        self.have = False

    @staticmethod
    def _wrap_toward(a, b, k):
        d = (b - a + math.pi) % (2.0 * math.pi) - math.pi
        return a + d * k

    def set_pose(self, pos, look_at, dt=1.0 / 60.0):
        """Put the drone at `pos` looking at `look_at`. THE call a shot makes.

        Attitude is DERIVED, never authored: the last few poses give a smoothed
        velocity and acceleration, and a machine accelerating forward must be
        nose-down by atan2(a_fwd, g) to be doing it. Yaw follows the velocity
        heading once it is actually moving and the look-at bearing when it is
        not, so a hover does not spin on numerical noise.
        """
        p = np.asarray(pos, np.float64).copy()
        tgt = np.asarray(look_at, np.float64).copy()
        dt = max(float(dt), 1e-4)
        if not self.have:
            self.have = True
            self.vel[:] = 0.0
            self.acc[:] = 0.0
            to0 = tgt - p
            self.yaw = math.atan2(to0[0], to0[2])
        else:
            v = (p - self.pos) / dt
            vprev = self.vel.copy()
            self.vel += (v - self.vel) * (1.0 - math.exp(-dt / 0.12))
            a = (self.vel - vprev) / dt
            self.acc += (a - self.acc) * (1.0 - math.exp(-dt / 0.22))
        self.pos = p
        self.look = tgt

        sp = float(np.linalg.norm(self.vel[[0, 2]]))
        to = tgt - p
        bearing = math.atan2(to[0], to[2])
        heading = math.atan2(self.vel[0], self.vel[2]) if sp > 1e-3 else bearing
        yaw_t = self._wrap_toward(bearing, heading, smoothstep_f(0.6, 2.5, sp))
        # SURVEYING, the nose is ON THE WORK. Yaw-follows-heading is right for
        # a machine in transit and wrong for one orbiting its subject: on the
        # orbit the heading is tangential, so the airframe flew sideways-on to
        # the building it was mapping and, from the follow camera, sideways-on
        # to the audience. A real survey quad STRAFES the circle with its nose
        # (and its sensors) held on the target. `face_w` is that mode as a
        # weight, eased in and out by the film clock so the hand-back to
        # heading-follow is a turn, not a snap.
        yaw_t = self._wrap_toward(yaw_t, bearing, self.face_w)
        self.yaw = self._wrap_toward(self.yaw, yaw_t, 1.0 - math.exp(-dt / 0.15))

        fwd = np.array([math.sin(self.yaw), 0.0, math.cos(self.yaw)])
        lat = np.array([math.cos(self.yaw), 0.0, -math.sin(self.yaw)])
        lim = math.radians(25.0)
        # +x nose-down, -z starboard-down: the signs that bank it INTO the turn.
        pt = max(-lim, min(lim, math.atan2(float(np.dot(self.acc, fwd)), DRONE_G)))
        rl = max(-lim, min(lim, -math.atan2(float(np.dot(self.acc, lat)), DRONE_G)))
        k = 1.0 - math.exp(-dt / 0.10)
        self.pitch += (pt - self.pitch) * k
        self.roll += (rl - self.roll) * k
        self.throttle = float(np.clip(0.5 + self.acc[1] / 9.0, 0.0, 1.0))

        self.root.position.set(float(p[0]), float(p[1]), float(p[2]))
        self.root.rotation.set(self.pitch, self.yaw, self.roll)

        # The gimbal points AT the look target, in three axes.
        #
        # It used to set pitch only, with yaw hard zero, which silently means
        # "aim along the airframe's heading". That is fine whenever the machine
        # flies at what it is filming and WRONG the moment it does not -- which
        # is the entire orbit, where the heading is tangential and the target is
        # 90 degrees off the bow. The camera got away with it because the FPV
        # shot bypasses the gimbal and calls look_at itself; the LIDAR did not,
        # because the sensor is a real child of this node. The survey was
        # pointing off the side of the flight path the whole way round, which is
        # why the map came back full of hillside and contained no building.
        #
        # Solve it properly: take the world direction to the target into the
        # airframe's own frame, then read the Euler that aims local +Z along it.
        d = tgt - p
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            dl = yxz_matrix(self.pitch, self.yaw, self.roll).T @ (d / n)
            g_yaw = math.atan2(float(dl[0]), float(dl[2]))
            g_pitch = -math.asin(float(np.clip(dl[1], -1.0, 1.0)))
            # Roll is the one axis NOT derived: a gimbal's job is to cancel the
            # airframe's roll and hold the horizon. Under YXZ a Z-rotation
            # leaves local +Z alone, so this cannot disturb the aim above.
            self.gimbal.rotation.set(g_pitch, g_yaw, -self.roll)

    def tick(self, dt, t, night=1.0):
        """Props, LEDs and the strobe."""
        self.spool += (1.0 - self.spool) * (1.0 - math.exp(-dt / 0.55))
        rate = (100.0 + 55.0 * self.throttle) * self.spool
        self.spin += rate * dt
        for hub, sgn in self.props:
            hub.rotation.y = sgn * self.spin
        # Driven HARD. These are 12 mm spheres on a 0.45 m machine, and in the
        # wide shots that machine is ninety metres out: at sane intensities it
        # is a grey speck and the shot has no hero in it. On Vulkan an emissive
        # mesh is a real light source, so cranking them also throws a red and a
        # green wash onto the booms, which is what nav lights actually do and
        # what makes the silhouette read as a machine rather than as a dot.
        lit = 40.0 + 180.0 * night
        for m in self.leds:
            if abs(m.emissive_intensity - lit) > 0.5:
                m.emissive_intensity = lit
                m.needs_update()
        # A short, hot anti-collision flash. Short so it reads as a strobe and
        # not as a lamp; hot so it survives the auto-exposure clamp at range.
        blink = 1400.0 if (t % 1.05) < 0.055 else 0.0
        if blink != self.strobe_mat.emissive_intensity:
            self.strobe_mat.emissive_intensity = blink
            self.strobe_mat.needs_update()

    def eye(self):
        """World position of the gimbal eye -- where an FPV camera sits."""
        return self.pos + yxz_matrix(self.pitch, self.yaw,
                                     self.roll) @ self.eye_offset


# --------------------------------------------------------------------------- #
#  Speed warps.  Each maps a leg's TIME fraction u in [0, 1] to the DISTANCE
#  fraction covered by then, so the same authored curve can be flown fast,
#  slow, or handed over to the next leg still moving.
# --------------------------------------------------------------------------- #
def ease(u):
    """Sine-shaped speed profile. Peak is 1.5x the mean."""
    u = min(max(u, 0.0), 1.0)
    return u * u * (3.0 - 2.0 * u)


def trap_at(u, ain=0.15, aout=0.15):
    """Distance covered at fraction `u` of a leg flown with a TRAPEZOID speed
    profile: accelerate for `ain`, cruise, decelerate for `aout`. Returns 0..1.

    `ease` is a sine-shaped speed profile whose peak is 1.5x its mean, which is
    fine over eight seconds and wrong over thirty: a 400 m ferry would have to
    touch 19 m/s in the middle to average 13. A trapezoid peaks at 1.1-1.3x, so
    the long run cruises instead of surging.

    Either ramp can be ZERO, which is how two legs are joined at cruise speed
    instead of both stopping dead at the seam: the leg before ends with no
    deceleration and the leg after starts with no acceleration.
    """
    u = min(max(u, 0.0), 1.0)
    norm = 1.0 - 0.5 * ain - 0.5 * aout
    if ain > 0.0 and u < ain:
        s = 0.5 * u * u / ain
    elif aout <= 0.0 or u < 1.0 - aout:
        s = 0.5 * ain + (u - ain)
    else:
        w = (1.0 - u) / aout
        s = norm - 0.5 * w * w * aout
    return s / norm


def ramp_at(u, v0, v1):
    """Distance covered at fraction `u` of a leg flown with a LINEAR speed ramp
    from v0 to v1 (relative units; only their ratio matters). Returns 0..1.

    `trap_at` always ends at a dead stop, which is exactly wrong for a leg whose
    job is to hand the machine over to another leg still moving -- the arrival
    has to arrive at orbit speed, not at zero and then start again. A linear
    ramp has constant acceleration, so the attitude the kinematic model derives
    from it is a constant lean rather than a lurch.
    """
    u = min(max(u, 0.0), 1.0)
    return (v0 * u + 0.5 * (v1 - v0) * u * u) / max(0.5 * (v0 + v1), 1e-9)


def trap(ain=0.15, aout=0.15):
    """`trap_at` as a one-argument warp, for `Leg(warp=...)`."""
    return lambda u: trap_at(u, ain, aout)


def ramp(v0, v1):
    """`ramp_at` as a one-argument warp, for `Leg(warp=...)`."""
    return lambda u: ramp_at(u, v0, v1)


def speed_warp(speed, n=257):
    """A warp built from an arbitrary SPEED profile `speed(ts) -> array`.

    This is the general form of `ferry_warp`: a run with a BEAT in the middle
    of it, where the machine comes out of the opening turn at moderate speed,
    opens up across empty water, SLOWS to about half for the seconds either
    side of something it passes, and eases off again at the far end so the
    next leg picks it up near the speed it brakes from. Flat-out, such a pass
    crosses in under a second and reads as a glitch.

    `speed` is evaluated once on a uniform time grid, integrated, normalised
    and cached in the returned closure, so the warp itself is a table lookup.
    Only the SHAPE of `speed` matters; its scale cancels.
    """
    ts = np.linspace(0.0, 1.0, n)
    v = np.asarray(speed(ts), np.float64)
    d = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]))])
    d = d / d[-1]
    return lambda u: float(np.interp(min(max(u, 0.0), 1.0), ts, d))


def ferry_warp(pass_at=0.80, pass_width=0.085, pass_depth=0.55,
               out_of_turn=0.22, hand_off=0.28):
    """A ferry profile with a beat in it, where you put it.

    Defaults: 62% speed out of the turn rising to full by 22%
    of the leg, a Gaussian slow-down to 45% centred four fifths of the way
    along, and a 28% ease-off over the last tenth for the hand-over.
    """
    def _v(ts):
        v = 0.62 + 0.38 * smoothstep(0.0, out_of_turn, ts)
        v = v * (1.0 - pass_depth * np.exp(-((ts - pass_at) / pass_width) ** 2))
        return v * (1.0 - hand_off * smoothstep(0.90, 1.0, ts))
    return speed_warp(_v)


# --------------------------------------------------------------------------- #
#  Curves
# --------------------------------------------------------------------------- #
def bez(u, p0, p1, p2):
    """Quadratic bezier, in three dimensions."""
    v = 1.0 - u
    return v * v * p0 + 2.0 * v * u * p1 + u * u * p2


def bez3(u, p0, p1, p2, p3):
    """Cubic bezier. The one the JOINING legs need: a quadratic can match the
    direction it leaves on or the direction it arrives on, not both."""
    v = 1.0 - u
    return (v ** 3 * p0 + 3.0 * v * v * u * p1
            + 3.0 * v * u * u * p2 + u ** 3 * p3)


def dir_to(a, b):
    """Unit direction a -> b; +Z if they coincide."""
    d = np.asarray(b, np.float64) - np.asarray(a, np.float64)
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def join_cubic(a, b, d_in, d_out, tension=0.38):
    """Control points for a cubic that leaves `a` along `d_in` and arrives at
    `b` along `d_out`. Returns (a, c1, c2, b), ready for `Leg(pos=...)`.

    This is what turns a corner between two legs that each already have a
    direction into a curve: a ferry leg arrives heading down an inlet at 18 m/s
    and the orbit after it leaves heading across it, and a quadratic can match
    one end or the other, not both.
    """
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    L = float(np.linalg.norm(b - a))
    return (a, a + np.asarray(d_in, np.float64) * (tension * L),
            b - np.asarray(d_out, np.float64) * (tension * L), b)


# --------------------------------------------------------------------------- #
#  Legs
# --------------------------------------------------------------------------- #
class Leg:
    """One authored leg of a route.

    ``Leg(duration, pos, aim, warp=None)``

      duration  seconds this leg lasts
      pos       one of
                  * a callable ``f(u) -> (3,)``, u the DISTANCE fraction
                  * a 2-tuple  (p0, p1)          -- a straight line
                  * a 3-tuple  (p0, c, p1)       -- a quadratic bezier
                  * a 4-tuple  (p0, c0, c1, p1)  -- a cubic bezier
                  * a single point               -- a hold
      aim       one of
                  * a callable ``f(u, p) -> (3,)`` -- p is the position, so an
                    aim can be stated as a constraint (a bearing off wherever
                    the machine actually is) rather than as a coordinate
                  * a 2-tuple of points  -- cross-faded with `ease` over the leg
                  * a single point       -- held
      warp      ``f(u_time) -> u_distance``; default is a symmetric trapezoid.
                Use ``trap(a, 0.0)`` to hand over at speed, ``ramp(v0, v1)`` to
                arrive still moving, ``ferry_warp()`` for a run with a beat in
                it, or ``lambda u: u`` for dead-constant speed.
    """

    def __init__(self, duration, pos, aim, warp=None):
        self.duration = float(duration)
        self.pos = self._as_pos(pos)
        self.aim = self._as_aim(aim)
        self.warp = warp if warp is not None else trap()

    @staticmethod
    def _as_pos(pos):
        if callable(pos):
            return pos
        if isinstance(pos, (tuple, list)) and len(pos) in (2, 3, 4) \
                and not np.isscalar(pos[0]):
            pts = [np.asarray(p, np.float64) for p in pos]
            if len(pts) == 2:
                return lambda u: pts[0] * (1.0 - u) + pts[1] * u
            if len(pts) == 3:
                return lambda u: bez(u, *pts)
            return lambda u: bez3(u, *pts)
        p = np.asarray(pos, np.float64)
        return lambda u: p

    @staticmethod
    def _as_aim(aim):
        if callable(aim):
            return aim
        if isinstance(aim, (tuple, list)) and len(aim) == 2 \
                and not np.isscalar(aim[0]):
            a0 = np.asarray(aim[0], np.float64)
            a1 = np.asarray(aim[1], np.float64)
            return lambda u, p: a0 * (1.0 - ease(u)) + a1 * ease(u)
        a = np.asarray(aim, np.float64)
        return lambda u, p: a

    def at(self, ut):
        """(position, aim) at TIME fraction `ut` of this leg."""
        u = self.warp(min(max(float(ut), 0.0), 1.0))
        p = np.asarray(self.pos(u), np.float64)
        return p, np.asarray(self.aim(min(max(float(ut), 0.0), 1.0), p),
                             np.float64)


class LegRoute:
    """A list of `Leg` as one ``f(t) -> (position, aim)``, THE ROUTE AS
    AUTHORED. Feed it to `Route`; do not fly it directly -- what the machine
    actually flies is the smoothed version, for the reasons written there."""

    def __init__(self, legs):
        self.legs = list(legs)
        self.starts = []
        s = 0.0
        for leg in self.legs:
            self.starts.append(s)
            s += leg.duration
        self.duration = s

    def __call__(self, t):
        t = float(t)
        for leg, t0 in zip(self.legs, self.starts):
            if t < t0 + leg.duration:
                return leg.at((t - t0) / max(leg.duration, 1e-9))
        return self.legs[-1].at(1.0)


# --------------------------------------------------------------------------- #
#  Signal filters, over the sampled route
# --------------------------------------------------------------------------- #
def blur(a, w):
    """Symmetric Hann blur over `w` samples, edge-padded."""
    w = max(3, int(w) | 1)
    k = np.hanning(w + 2)[1:-1]
    k /= k.sum()
    return np.convolve(np.pad(a, w // 2, mode="edge"), k, mode="valid")[:len(a)]


def close_1d(a, w):
    """Grey-scale CLOSING (dilate, then erode) over an odd window of w samples.

    Fills valleys narrower than the window and leaves everything else exactly
    where it was -- which is the difference between a machine that climbs to
    its working height and holds it, and one that follows the ground back
    down the far side of every rise.

    This is the operator the approach needed and neither blurring nor a soft
    max could supply. Both of those are LOCAL AVERAGES: they round a corner
    but they cannot remove a dip, because the dip is genuinely there in the
    signal -- the route floor crests over the treeline and falls away behind
    it, so the flown path climbed, sank half a metre into the hollow, and
    climbed again. On screen that is the machine bobbing, and it survived two
    rounds of wider kernels for the simple reason that averaging a valley
    keeps a (shallower) valley. A closing removes it outright.
    """
    w = max(3, int(w) | 1)
    pad = w // 2
    sw = np.lib.stride_tricks.sliding_window_view
    dil = sw(np.pad(a, pad, mode="edge"), w).max(axis=1)
    return sw(np.pad(dil, pad, mode="edge"), w).min(axis=1)


def soft_max(a, b, m):
    """max(a, b) without the corner. Never below either input.

    `max()` is C0: at the crossing the slope steps, and a slope step in Y is a
    vertical jolt the attitude model banks to. softplus is smooth in every
    derivative and within a few centimetres of the true max once the curves
    separate by more than `m`, at the cost of about 0.7*m of extra height where
    they cross -- clearance spent on smoothness, which is the trade a route
    wants.
    """
    return b + m * np.logaddexp(0.0, (a - b) / m)


def _field(fn, default=0.0):
    """Wrap a scalar-or-vector height callable so it is safe on arrays."""
    if fn is None:
        return lambda x, z: np.full(np.shape(x), float(default))

    def f(x, z):
        x = np.asarray(x, np.float64)
        z = np.asarray(z, np.float64)
        try:
            r = np.asarray(fn(x, z), np.float64)
            if r.shape == x.shape:
                return r
        except Exception:
            pass
        return np.array([float(fn(float(a), float(b)))
                         for a, b in zip(x.ravel(), z.ravel())]).reshape(x.shape)
    return f


# --------------------------------------------------------------------------- #
#  THE ROUTE, SMOOTHED.
#
#  The legs above are each eased in their own parameter, so at every seam the
#  SPEED steps. And this machine's attitude is KINEMATIC: `Drone.set_pose`
#  reads the acceleration off the path and leans the airframe to match, so a
#  velocity step is not a subtle continuity error, it is a visible snap of the
#  whole aircraft and of every camera bolted to it.
#
#  Two fixes, and both are needed. The legs themselves join on matched tangents
#  (`join_cubic`, and a warp that hands over at speed). That fixes DIRECTION.
#  What is left is the speed magnitude at each seam, and the honest way to kill
#  that is to stop treating the route as a piecewise function evaluated per
#  frame and treat it as a SIGNAL: sample it densely, run a symmetric Hann
#  kernel over it, and fly the result.
#
#  A symmetric kernel is the point. It does not shift the beats -- a cut list
#  and a scan gate still land where they were authored -- it only rounds the
#  corners, and a 1 s kernel at cruise rounds them over about fifteen metres of
#  a four-hundred-metre run. The AIM is smoothed with the same kernel, because
#  a gimbal target that jumps at a leg boundary snaps the lens just as visibly
#  as a position that does.
#
#  Cost: a few thousand evaluations at construction, once, well under a second.
# --------------------------------------------------------------------------- #
PATH_DT = 1.0 / 60.0         # two samples per film frame
PATH_SMOOTH = 1.00           # kernel width, seconds
ROUTE_CLEAR = 4.0            # metres the route keeps over ground AND canopy
CLIMB_RATE = 2.5             # ... and the most climb its floor may ask for, m/s


class Route:
    """An authored route, sampled, smoothed, and held above the ground.

    ``Route(legs, duration=None, ground=None, canopy=None, ...)``

      legs      a `LegRoute`, or any ``f(t) -> (position, aim)``
      duration  seconds; taken from a `LegRoute` if omitted
      ground    ``ground(x, z) -> height`` in metres, called on arrays.
                None = sea level everywhere and the floor is just `clear`.
      canopy    ``canopy(x, z) -> height of standing cover ABOVE the ground``.
                None = bare ground.
      clear     metres kept over ground AND canopy
      climb_rate the most climb the floor may ask for, m/s. The raw requirement
                has a CLIFF in it wherever a clearing's cut line is, and
                clamping a path against a cliff produces a path with a cliff in
                it. Rate-limiting the requirement BACKWARDS in time makes the
                floor start rising long before the obstacle, so the machine
                goes up the way a pilot would -- early and gently.

    ``route.pose(t)`` -> (position, aim), both (3,) float arrays, metres.
    """

    def __init__(self, legs, duration=None, ground=None, canopy=None,
                 dt=PATH_DT, smooth=PATH_SMOOTH, clear=ROUTE_CLEAR,
                 climb_rate=CLIMB_RATE, soft_m=2.0, floor_blur=1.50,
                 close_seconds=2.20, post_blur=1.20, sea_y=0.0):
        self.dt = float(dt)
        self.duration = float(duration if duration is not None
                              else getattr(legs, "duration", 60.0))
        self.sea_y = float(sea_y)
        self.ground = _field(ground, sea_y)
        self.canopy = _field(canopy, 0.0)
        self.clear = float(clear)

        n = int(round(self.duration / self.dt)) + 1
        raw = np.empty((n, 6))
        for i in range(n):
            p, a = legs(i * self.dt)
            raw[i, :3] = p
            raw[i, 3:] = a

        w = max(3, int(round(smooth / self.dt)) | 1)
        k = np.hanning(w + 2)[1:-1]
        k /= k.sum()
        # Edge padding, so the first and last frames are held rather than faded
        # toward zero -- a film opens on a stationed drift and ends on a hold,
        # and both want their authored value.
        pad = np.pad(raw, ((w // 2, w // 2), (0, 0)), mode="edge")
        sm = np.empty_like(raw)
        for c in range(6):
            sm[:, c] = np.convolve(pad[:, c], k, mode="valid")[:n]

        # AND THEN THE GROUND GETS A VETO. Rounding a corner MOVES the path,
        # and the corner that matters is usually an arrival: a machine coming
        # down a valley at eighteen metres a second against a treeline standing
        # seventeen metres tall on ground that is already climbing. Rather than
        # hand-tune a waypoint until it happens to clear, state the constraint
        # and enforce it.
        need = self.floor_at(sm[:, 0], sm[:, 2])
        lim = float(climb_rate) * self.dt
        for i in range(n - 2, -1, -1):
            if need[i] < need[i + 1] - lim:
                need[i] = need[i + 1] - lim
        # Applied with a SOFT max, not max(): see `soft_max`.
        fl = blur(need, max(3, int(round(floor_blur / self.dt)) | 1))
        sm[:, 1] = soft_max(sm[:, 1], fl, float(soft_m))
        # ... then FILL THE HOLLOWS, then round what is left. Closing first,
        # because it is a max/min filter and leaves corners of its own; the
        # blur after it is what turns the filled plateau into a flown curve.
        sm[:, 1] = close_1d(sm[:, 1], round(close_seconds / self.dt))
        sm[:, 1] = blur(sm[:, 1], max(3, int(round(post_blur / self.dt)) | 1))

        self.path = sm
        self.legs = legs

    def floor_at(self, x, z):
        """Lowest height the route may fly at (x, z): ground, plus whatever is
        standing on it, plus `clear`. Vectorised."""
        return self.ground(x, z) + self.canopy(x, z) + self.clear

    def pose(self, t):
        """(position, aim) of the FLOWN route at time t. THE call a film makes."""
        return _sample(self.path, t, self.dt, self.duration)

    def __call__(self, t):
        return self.pose(t)


def _sample(tab, t, dt, duration):
    u = min(max(float(t), 0.0), duration) / dt
    i = int(u)
    if i >= len(tab) - 1:
        r = tab[-1]
    else:
        f = u - i
        r = tab[i] * (1.0 - f) + tab[i + 1] * f
    return r[:3].copy(), r[3:].copy()


# --------------------------------------------------------------------------- #
#  THE CAMERA, SMOOTHED THE SAME WAY.
#
#  Making the route spline-smooth is only half of it: HALF THE MOTION ON SCREEN
#  IS THE CAMERA'S. The follow rig is an offset from the machine, clamped
#  afterwards against the ground and the canopy -- and a canopy field steps by
#  seventeen metres the instant the eye crosses a clearing's cut line, which is
#  a hard edge on purpose. Measured on the film this was built for: the rig's own
#  target moved at up to 9.4 m/s vertically where the route it follows never
#  exceeds 2.6. An exponential filter cannot save that; it turns a step into a
#  fast ramp, and a camera ramping upward is indistinguishable, on screen, from
#  the subject dropping.
#
#  The fix is the one the route already uses, and it works here for the same
#  reason: this rig is a pure FUNCTION of the flown path, so the whole camera
#  track can be built up front and filtered as a signal rather than chased
#  frame by frame. Requirement first (dilate, so the smoothed curve can never
#  fall below the raw clearance), then the authored offset joined with a soft
#  max, then a closing to fill the hollows, then a blur.
# --------------------------------------------------------------------------- #
CAM_SMOOTH = 0.90            # position kernel, seconds
CAM_AIM_SMOOTH = 1.10        # the aim is allowed to lag further than the eye
LAND_RANGES = (18.0, 42.0, 80.0, 150.0)


def land_beyond(p, dirh, ground=None, canopy=None, ranges=LAND_RANGES,
                lo=1.5, hi=6.0):
    """0 = open water behind the machine, 1 = land. THE knob that decides
    whether a third-person eye rides under the drone or over it.

    A follow rig that sits BELOW the machine has a good reason: over water
    there is nothing behind the drone but sea and haze, and from underneath it
    stands against the sky, which is the brightest thing in the frame. Over
    LAND it is exactly wrong -- looking up puts the whole frame above the
    horizon, and an arrival at a building comes back as drone and sky with the
    thing the film is about below the bottom edge.

    So ask the terrain. Probe the ground (and its canopy) along the view axis
    PAST the drone, at the ranges a background actually occupies, and report
    how much of what is out there is land standing above the water. It is a
    property of the scenery, not of the clock, so it works on every leg
    including the ones nobody thought about.
    """
    g = _field(ground, 0.0)
    c = _field(canopy, 0.0)
    d = np.asarray(dirh, np.float64)
    d = np.array([d[0], 0.0, d[2]])
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return 0.0
    d /= n
    hit = 0.0
    for r in ranges:
        x = np.array([float(p[0] + d[0] * r)])
        z = np.array([float(p[2] + d[2] * r)])
        # "Land" means standing clear of the water, not merely above datum: a
        # metre of wet shingle 150 m out is still a seascape.
        hit += float(smoothstep(lo, hi, g(x, z) + c(x, z))[0])
    return hit / len(ranges)


class FollowCamera:
    """The third-person rig for a whole `Route`, precomputed and filtered.

    ``FollowCamera(route, ground=..., canopy=..., back=6.0, ...)``

      back        metres the eye stands OUT from the machine, on the far side of
                  it from whatever its gimbal is on
      rise_water  metres the eye rides above the machine over open water ...
      rise_land   ... and over land, blended by `land_beyond`. Both positive:
                  the eye sits above the drone everywhere, which is what makes
                  it look DOWN on the machine rather than up at its belly.
      aim_bias    the aim is `route_aim * b + drone_pos * (1 - b)`, so the shot
                  is composed on the work with the machine in the near frame
      ceiling     the eye is soft-min'd against drone height + this, so the
                  follow can never become a bird's eye over a stand
      hold_deg    after all the smoothing, wherever clearance lifted the eye the
                  aim is rotated back onto the drone by exactly the excess past
                  this half-angle -- KEEP THE MACHINE IN FRAME

    ``cam.pose(t)`` -> (eye, aim), both (3,) float arrays, metres.
    """

    def __init__(self, route, ground=None, canopy=None, back=6.0,
                 rise_water=0.8, rise_land=2.6, aim_bias=0.72, eye_clear=3.2,
                 ceiling=7.0, hold_deg=13.0, smooth=CAM_SMOOTH,
                 aim_smooth=CAM_AIM_SMOOTH, dilate=1.60, close_seconds=2.00,
                 sea_clear=1.6, ranges=LAND_RANGES):
        self.route = route
        self.dt = route.dt
        self.duration = route.duration
        g = _field(ground, route.sea_y) if ground is not None else route.ground
        c = _field(canopy, 0.0) if canopy is not None else route.canopy

        path = route.path
        n = len(path)
        pos, look = path[:, :3].copy(), path[:, 3:].copy()

        # Which way is OUT from the subject, in plan. The rig stands on the far
        # side of the machine from whatever its gimbal is on.
        out = pos - look
        out[:, 1] = 0.0
        d = np.linalg.norm(out, axis=1, keepdims=True)
        out = np.divide(out, np.maximum(d, 1e-6))

        # `land_beyond`, vectorised over the whole flight.
        w = np.zeros(n)
        for r in ranges:
            px = pos[:, 0] - out[:, 0] * r
            pz = pos[:, 2] - out[:, 2] * r
            w += smoothstep(1.5, 6.0, g(px, pz) + c(px, pz))
        w /= float(len(ranges))
        self.land = w

        eye = pos + out * float(back)
        eye[:, 1] = pos[:, 1] + (float(rise_water)
                                 + (float(rise_land) - float(rise_water)) * w)
        aim = look * float(aim_bias) + pos * (1.0 - float(aim_bias))

        # The requirement, dilated then blurred so the climb starts early and
        # the curve is never below the raw clearance it was built from.
        floor = np.maximum(g(eye[:, 0], eye[:, 2]) + float(eye_clear)
                           + c(eye[:, 0], eye[:, 2]),
                           route.sea_y + float(sea_clear))
        wdil = max(3, int(round(dilate / self.dt)) | 1)
        sw = np.lib.stride_tricks.sliding_window_view
        floor = sw(np.pad(floor, wdil // 2, mode="edge"), wdil).max(axis=1)
        floor = blur(floor, wdil)

        y = soft_max(eye[:, 1], floor, 1.5)
        # NEVER A BIRD'S EYE: a soft min against the machine's own height plus
        # a ceiling, so the follow cannot become a top-down over a stand.
        y = -soft_max(-y, -(pos[:, 1] + float(ceiling)), 1.0)
        y = close_1d(y, round(close_seconds / self.dt))
        kp = max(3, int(round(smooth / self.dt)) | 1)
        ka = max(3, int(round(aim_smooth / self.dt)) | 1)
        eye[:, 1] = blur(y, kp)
        for ci in (0, 2):
            eye[:, ci] = blur(eye[:, ci], kp)
        for ci in range(3):
            aim[:, ci] = blur(aim[:, ci], ka)

        # KEEP THE MACHINE IN FRAME, after the smoothing rather than before it:
        # wherever the clearance lifted the eye, rotate the aim back onto the
        # drone by exactly the excess. Blurred again afterwards, lightly, so the
        # cone's threshold cannot put a corner back into the aim.
        v = pos - eye
        wv = aim - eye
        nv = np.linalg.norm(v, axis=1, keepdims=True)
        nw = np.linalg.norm(wv, axis=1, keepdims=True)
        vu = v / np.maximum(nv, 1e-6)
        wu = wv / np.maximum(nw, 1e-6)
        ang = np.arccos(np.clip((vu * wu).sum(axis=1), -1.0, 1.0))
        lim = math.radians(float(hold_deg))
        kk = np.clip((ang - lim) / np.maximum(ang, 1e-6), 0.0, 1.0)[:, None]
        m = wu * (1.0 - kk) + vu * kk
        aim = eye + m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True),
                                   1e-6) * nw
        for ci in range(3):
            aim[:, ci] = blur(aim[:, ci], max(3, int(round(0.50 / self.dt)) | 1))

        self.track = np.concatenate([eye, aim], axis=1)

    def pose(self, t):
        """(eye, aim) of the smoothed follow camera at time t."""
        return _sample(self.track, t, self.dt, self.duration)

    def __call__(self, t):
        return self.pose(t)


def off_axis(cam, aim, target):
    """Angle between the view axis and the machine, in degrees.

    The one number that says whether a third-person shot is actually holding
    its subject: half the vertical field is FOV/2, so anything past that is out
    of the top or bottom of frame no matter how good the offset looked in the
    comment above it.
    """
    a = np.asarray(aim, np.float64) - np.asarray(cam, np.float64)
    b = np.asarray(target, np.float64) - np.asarray(cam, np.float64)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0,
                                                float(np.dot(a, b)) / (na * nb)))))


# --------------------------------------------------------------------------- #
#  A route sanity check, with no renderer: the kit's own smoke test.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    def ground(x, z):
        # A ridge running north-south, so the floor has something to veto.
        return 40.0 * np.exp(-((np.asarray(x) - 300.0) / 260.0) ** 2)

    A = np.array([-400.0, 40.0, 400.0])
    B = np.array([100.0, 30.0, -60.0])
    C = np.array([300.0, 60.0, -200.0])
    legs = LegRoute([
        Leg(20.0, pos=(A, A + np.array([200.0, 0.0, -300.0]), B),
            aim=(B, C), warp=ferry_warp()),
        Leg(12.0, pos=join_cubic(B, C, dir_to(A, B), dir_to(B, C)),
            aim=C, warp=ramp(5.0, 1.0)),
        Leg(8.0, pos=C, aim=C + np.array([0.0, 0.0, -200.0])),
    ])
    route = Route(legs, ground=ground)
    cam = FollowCamera(route, ground=ground)

    print("duration %.1f s, %d samples" % (route.duration, len(route.path)))
    prev = None
    vmax = amax = 0.0
    for i in range(len(route.path)):
        p = route.path[i, :3]
        if prev is not None:
            v = (p - prev) / route.dt
            vmax = max(vmax, float(np.linalg.norm(v)))
            if i > 1:
                amax = max(amax, float(np.linalg.norm((v - pv) / route.dt)))
            pv = v
        prev = p
    print("peak speed %.1f m/s, peak accel %.2f g" % (vmax, amax / 9.81))
    for t in (0.0, 10.0, 20.0, 30.0, 39.9):
        p, a = route.pose(t)
        e, ea = cam.pose(t)
        print("t=%5.1f  drone (%7.1f %6.1f %7.1f)  clear %5.1f m  "
              "eye (%7.1f %6.1f %7.1f)  off-axis %4.1f deg"
              % (t, p[0], p[1], p[2],
                 p[1] - float(ground(np.array([p[0]]), np.array([p[2]]))[0]),
                 e[0], e[1], e[2], off_axis(e, ea, p)))
