"""E3: the ROV inspection loop -- perception, controller and vehicle, as pure code.

The scene (warp_netpen.py) owns the world; this module owns the loop that closes
around it. Nothing here imports the scene, opens a window or touches the GPU, so
every part of it runs (and is checked) on its own:

    python netpen_e3.py --selftest

The chain, once per sonar image (20 Hz; the vehicle runs at 60 Hz):

    intensity (256 beams x 512 bins)
      -> first_echo_ranges   per-beam range of the first bin above a threshold
      -> wall_estimate       standoff, its bearing, the wall's tangent, a seen flag
      -> RangeNoise          the seeded ranging noise (the ONLY thing a seed changes)
      -> Controller          PD on the standoff, heading on the tangent, speed, depth
      -> Rov                 thrust-limited, drag-damped, at 60 Hz

Conventions. Body frame: +x forward, +y to starboard, angles measured from
forward toward starboard, so a bearing is positive to starboard. `side` is which
side of the hull the net wall is on (+1 starboard, -1 port); the controller
latches it from the first image it sees rather than assuming, because which way
the sonar's beam index runs is the engine's business, not this file's.

No wall clock anywhere: every rate is per-call with the caller's dt, and the
only random numbers come from a seeded numpy Generator whose draw sequence is
fixed by the call order.
"""
import math
import warnings

import numpy as np

# ---- the sonar's geometry (mirrors warp_netpen.py's SON_* constants) ---------
SON_BEAMS, SON_BINS, SON_RANGE, SON_FOV = 256, 512, 20.0, 130.0
SON_MIN_RANGE = 0.35                                  # sonar.params.min_range: the hull behind the transducer
# Which way the engine's beam index runs: +1 = the index increases to starboard.
# Not a guess -- warp_netpen.py --e3 prints the estimated wall bearing beside the
# true one from net_nearest() every frame, so one run says whether it holds.
BEAM_SIGN = 1

# ---- the vehicle (the constants of build_patrol, so the two models agree) ----
A_MAX = 0.3                                           # thrust-limited acceleration, m/s^2
C_DRAG = 0.3 / 0.8 ** 2                               # quadratic drag: 0.8 m/s terminal on full thrust
YAW_RATE_MAX = math.radians(25.0)
YAW_TAU = 0.4                                         # first-order lag from yaw-rate command to yaw rate
TRIM_TAU = 0.6                                        # the passively stable hull's trim lag
PITCH_CAP, ROLL_CAP = math.radians(6.0), math.radians(3.0)

# ---- the mission --------------------------------------------------------------
STANDOFF = 1.5                                        # metres of water the controller holds off the cloth
V_PATROL = 0.35                                       # m/s along the wall
NET_CLEAR = 0.95                                      # the scene's geometric safety push; never the setpoint


def beam_bearings(beams=SON_BEAMS, fov_deg=SON_FOV, sign=BEAM_SIGN):
    """Body bearing of each beam centre, radians, in beam-index order. Positive is
    to starboard; `sign` is the engine's beam order (see BEAM_SIGN)."""
    fov = math.radians(fov_deg)
    return sign * ((np.arange(beams, dtype=np.float64) + 0.5) / beams * fov - 0.5 * fov)


def first_echo_ranges(intensity, thresh=None, rel=0.06, abs_floor=0.01,
                      min_range=SON_MIN_RANGE, son_range=SON_RANGE, smooth=3):
    """Per-beam range of the FIRST range bin above a threshold, metres; NaN where none.

    `intensity` is (beams, bins) as SonarImage delivers it. The bin size is
    son_range / bins, and a bin's range is taken at its centre. With `thresh`
    None the threshold is adaptive -- max(abs_floor, rel * image max) -- because
    the intensity scale is the model's (attenuation, reflectivity), not a fixed
    unit; the result is still a pure function of the image.

    `smooth` (odd, 0/1 = off) runs a median across neighbouring beams AFTER the
    first echo is picked: a fish inside the pen lights a handful of beams and
    would otherwise win the min over the sector, while the wall is a broad arc
    that a 3-beam median leaves alone.
    """
    a = np.asarray(intensity, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"intensity must be (beams, bins), got {a.shape}")
    bins = a.shape[1]
    bin_m = son_range / bins
    if thresh is None:
        peak = float(np.nanmax(a)) if a.size else 0.0
        thresh = max(abs_floor, rel * peak)
    b0 = int(math.ceil(min_range / bin_m))             # the near field is the sensor's own min range
    hit = np.zeros_like(a, dtype=bool)
    hit[:, b0:] = a[:, b0:] >= thresh
    idx = np.argmax(hit, axis=1)
    r = (idx.astype(np.float64) + 0.5) * bin_m
    r[~hit.any(axis=1)] = np.nan
    if smooth and smooth > 1:
        k = int(smooth) | 1
        pad = np.pad(r, k // 2, mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, k)
        with warnings.catch_warnings():                # an all-NaN window is a hole, not a problem
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(win, axis=1)
        r = np.where(np.isnan(r), np.nan, med)         # a beam with no echo stays a hole
    return r


class WallEst:
    """What one sonar image says about the net wall, in the body frame."""

    __slots__ = ("seen", "standoff", "bearing", "perp", "tangent", "hits", "side")

    def __init__(self, seen=False, standoff=float("nan"), bearing=float("nan"),
                 perp=float("nan"), tangent=float("nan"), hits=0, side=0):
        self.seen, self.standoff, self.bearing = seen, standoff, bearing
        self.perp, self.tangent, self.hits, self.side = perp, tangent, hits, side

    def __repr__(self):
        return (f"<WallEst seen={self.seen} standoff={self.standoff:.2f} "
                f"bearing={math.degrees(self.bearing):.0f} tangent={math.degrees(self.tangent):.0f} "
                f"hits={self.hits} side={self.side:+d}>")


def wall_side_of(ranges, bearings=None, dead=math.radians(15.0)):
    """Which half of the fan the wall is in: +1 = the +bearing end, -1 = the -bearing end.

    The wall is the near, dense half. Compares the 20th percentile of the valid
    ranges either side of the fan centre (the dead band around dead-ahead is left
    out of both), and breaks a tie on which half has more returns. 0 = no idea.
    """
    r = np.asarray(ranges, dtype=np.float64)
    b = beam_bearings(len(r)) if bearings is None else np.asarray(bearings, dtype=np.float64)
    ok = np.isfinite(r)
    out = []
    for s in (-1, 1):
        m = ok & (s * b > dead)
        out.append((float(np.percentile(r[m], 20.0)) if m.sum() >= 8 else float("inf"), int(m.sum())))
    (rm, nm), (rp, np_) = out
    if not np.isfinite(rm) and not np.isfinite(rp):
        return 0
    if abs(rp - rm) < 0.05:
        return 1 if np_ > nm else (-1 if nm > np_ else 0)
    return 1 if rp < rm else -1


def wall_estimate(ranges, side, bearings=None, sector=math.radians(10.0),
                  fit_lo=math.radians(20.0), fit_span=2.5, min_hits=8):
    """Standoff, its bearing, and the wall's tangent, from one image's beam ranges.

    - standoff: the minimum valid range over the forward sector on the wall's side
      (bearings with `side * bearing > -sector`), and the bearing where it is.
    - tangent: the angle of the wall relative to the heading, from a total-least-
      squares line through the near returns on the wall side (bearings beyond
      `fit_lo`, ranges within `fit_span` of the standoff). Negative = the wall
      falls away ahead, which is what the inside of the pen looks like from a
      tangential heading, so the number carries the curvature the controller needs.
    - perp: that line's perpendicular distance -- the standoff without the fan-edge
      bias, reported for the log, not used as the setpoint.
    """
    r = np.asarray(ranges, dtype=np.float64)
    b = beam_bearings(len(r)) if bearings is None else np.asarray(bearings, dtype=np.float64)
    if side not in (-1, 1):
        return WallEst(side=0)
    ok = np.isfinite(r) & (side * b > -sector)
    if int(ok.sum()) < min_hits:
        return WallEst(hits=int(ok.sum()), side=side)
    j = int(np.flatnonzero(ok)[np.argmin(r[ok])])
    standoff, bearing = float(r[j]), float(b[j])
    fit = ok & (side * b > fit_lo) & (r < standoff + fit_span)
    tangent, perp = float("nan"), float("nan")
    if int(fit.sum()) >= 4:
        # body-frame points with y taken toward the wall, so the fit is side-agnostic
        x, y = r[fit] * np.cos(b[fit]), side * r[fit] * np.sin(b[fit])
        p = np.stack([x, y], 1)
        c = p.mean(0)
        d = np.linalg.svd(p - c, full_matrices=False)[2][0]
        if d[0] < 0.0:                                 # point it forward
            d = -d
        tangent = float(side * math.atan2(d[1], d[0]))
        n = np.array([-d[1], d[0]])
        perp = float(abs(n @ c))
    return WallEst(True, standoff, bearing, perp, tangent, int(ok.sum()), side)


class RangeNoise:
    """The ranging noise on the measured standoff: the engine's RangeNoiseModel,
    applied to a scalar.

    tp.RangeNoiseModel is a config struct -- the sensors that own one apply it
    inside C++ (SonarSensor.noise perturbs every ray), and nothing in the Python
    bindings applies one to a number. So this reproduces its documented contract
    exactly -- sigma = hypot(stddev, r * stddev_per_metre) metres, plus a fixed
    bias, one draw per return, seeded -- on a numpy Generator. `from_model()`
    takes the parameters straight off a tp.RangeNoiseModel so there is one
    definition of them.

    The seed is the ONLY thing that differs between the seed-spread runs.
    """

    def __init__(self, stddev=0.02, stddev_per_metre=0.004, bias=0.0, seed=0):
        self.stddev, self.stddev_per_metre, self.bias = float(stddev), float(stddev_per_metre), float(bias)
        self.seed = int(seed)
        self.reset()

    @classmethod
    def from_model(cls, model, seed=None):
        return cls(model.stddev, model.stddev_per_metre, model.bias,
                   model.seed if seed is None else seed)

    def reset(self):
        self._rng = np.random.Generator(np.random.PCG64(self.seed))
        self.draws = 0

    def __call__(self, r):
        """One seeded draw per call; NaN passes through without consuming one."""
        if not np.isfinite(r):
            return r
        sigma = math.hypot(self.stddev, r * self.stddev_per_metre)
        self.draws += 1
        return float(r + self.bias + sigma * self._rng.standard_normal())


class Rov:
    """The vehicle: thrust-limited, drag-damped, yaw-rate limited, at the caller's dt.

    Same constants and the same integration as warp_netpen.py's build_patrol, so
    the live vehicle and the baked patrol are the same machine; the difference is
    only who writes the thrusts. Commands are the bake's own three normalised
    thrusts (forward, lateral/starboard, vertical, each in [-1, 1]) plus a yaw
    rate in rad/s.
    """

    def __init__(self, pos, yaw, vel=None, yaw_rate=0.0, pitch=0.0, roll=0.0):
        self.pos = np.array(pos, np.float64)
        self.vel = np.zeros(3) if vel is None else np.array(vel, np.float64)
        self.yaw, self.yaw_rate = float(yaw), float(yaw_rate)
        self.pitch, self.roll = float(pitch), float(roll)
        self.cmd = (0.0, 0.0, 0.0, 0.0)

    # the scene's convention: forward is +x at yaw 0, starboard is +z
    @property
    def fwd(self):
        return np.array([math.cos(self.yaw), 0.0, -math.sin(self.yaw)])

    @property
    def right(self):
        return np.array([math.sin(self.yaw), 0.0, math.cos(self.yaw)])

    def body_vel(self):
        """(forward, starboard, up) components of the velocity."""
        return float(self.vel @ self.fwd), float(self.vel @ self.right), float(self.vel[1])

    def step(self, cmd, dt):
        tf, tl, tv, yaw_cmd = (float(np.clip(cmd[0], -1.0, 1.0)), float(np.clip(cmd[1], -1.0, 1.0)),
                               float(np.clip(cmd[2], -1.0, 1.0)),
                               float(np.clip(cmd[3], -YAW_RATE_MAX, YAW_RATE_MAX)))
        self.cmd = (tf, tl, tv, yaw_cmd)
        a = A_MAX * (tf * self.fwd + tl * self.right + tv * np.array([0.0, 1.0, 0.0]))
        self.vel = self.vel + (a - C_DRAG * float(np.linalg.norm(self.vel)) * self.vel) * dt
        self.pos = self.pos + self.vel * dt
        self.yaw_rate += (yaw_cmd - self.yaw_rate) * (1.0 - math.exp(-dt / YAW_TAU))
        self.yaw += self.yaw_rate * dt
        # trim: the thrust moment against the metacentric restoring moment, lagged
        k = 1.0 - math.exp(-dt / TRIM_TAU)
        self.pitch += (max(-PITCH_CAP, min(PITCH_CAP, -math.radians(4.0) * tf)) - self.pitch) * k
        self.roll += (max(-ROLL_CAP, min(ROLL_CAP, math.radians(3.0) * tl)) - self.roll) * k
        return self

    def row(self):
        """The nine numbers rov_pose() wants: x, y, z, yaw, pitch, roll, thrust f/l/v."""
        tf, tl, tv, _ = self.cmd
        return (float(self.pos[0]), float(self.pos[1]), float(self.pos[2]),
                self.yaw, self.pitch, self.roll, tf, tl, tv)


class Controller:
    """Wall-follower: hold a standoff off the net, keep the heading on the wall's
    tangent, a constant speed along it, and the depth of the cut.

    - standoff: P on the measured error, D on the vehicle's OWN lateral velocity.
      Rate feedback and not d(range)/dt on purpose -- differentiating a noisy
      20 Hz range is the one place a sensible gain turns into a buzz.
    - heading: a rate command proportional to the measured tangent. On the inside
      of the pen a tangential heading measures a tangent that falls away from the
      wall, so the same term is the curvature feed-forward: no radius is coded in.
    - the ROV is holonomic, so heading and standoff do not have to be one loop.
    - between sonar images `command()` is not called again: the caller holds the
      last command, which is what `last` carries.
    """

    def __init__(self, standoff=STANDOFF, depth=None, v_patrol=V_PATROL, side=0,
                 kp=1.6, kd=2.6, ki=0.30, k_psi=0.6, kv=3.0, kz=1.2, kzd=2.2,
                 lat_max=0.85, i_max=0.35, search_rate=math.radians(6.0),
                 beam_sign=BEAM_SIGN):
        self.standoff, self.depth, self.v_patrol = standoff, depth, v_patrol
        self.side = int(side)                          # 0 = latch it from the first image
        self.kp, self.kd, self.ki, self.k_psi = kp, kd, ki, k_psi
        self.kv, self.kz, self.kzd = kv, kz, kzd
        self.lat_max, self.i_max, self.search_rate = lat_max, i_max, search_rate
        self.beam_sign, self.acc = int(beam_sign), 0.0
        self.last = (0.0, 0.0, 0.0, 0.0)
        self.est, self.meas, self.clean, self.misses = WallEst(), float("nan"), float("nan"), 0

    def observe(self, intensity, noise=None, thresh=None):
        """One sonar image -> the wall estimate and the noisy standoff. Latches the
        side on the first image with a wall in it."""
        r = first_echo_ranges(intensity, thresh=thresh)
        b = beam_bearings(len(r), sign=self.beam_sign)
        if self.side == 0:
            self.side = wall_side_of(r, b)
        est = wall_estimate(r, self.side, b) if self.side else WallEst()
        self.est = est
        self.clean = est.standoff if est.seen else float("nan")
        self.meas = noise(self.clean) if (noise is not None and est.seen) else self.clean
        self.misses = 0 if est.seen else self.misses + 1
        return est

    def command(self, rov, dt):
        """The four commands, from the latest estimate and the vehicle's own state."""
        vf, vl, vy = rov.body_vel()
        depth = rov.pos[1] if self.depth is None else self.depth
        fwd = self.kv * (self.v_patrol - vf)
        vert = self.kz * (depth - rov.pos[1]) - self.kzd * vy
        if self.est.seen and np.isfinite(self.meas):
            e = self.standoff - self.meas              # +ve = too close to the cloth
            side = self.side
            # a slow integrator takes out the standing offset: holding a curved
            # wall needs a steady outward thrust, and P alone can only make one
            # by sitting inside its own setpoint
            self.acc = float(np.clip(self.acc + self.ki * e * dt, -self.i_max, self.i_max))
            lat = np.clip(-side * (self.kp * e + self.acc) - self.kd * vl, -self.lat_max, self.lat_max)
            psi = self.est.tangent if np.isfinite(self.est.tangent) else 0.0
            yaw_rate = np.clip(-self.k_psi * psi, -YAW_RATE_MAX, YAW_RATE_MAX)
        else:
            # no wall in the fan: creep forward, stop sliding, and turn gently
            # toward the side it was last seen on until it comes back
            lat = float(np.clip(-self.kd * vl, -self.lat_max, self.lat_max))
            yaw_rate = self.search_rate * (self.side if self.side else 1)
            fwd = min(fwd, 0.3)
        self.last = (float(np.clip(fwd, -1.0, 1.0)), float(lat), float(np.clip(vert, -1.0, 1.0)),
                     float(yaw_rate))
        return self.last


# ---- selftest: no GPU, no scene ---------------------------------------------
def _synth_image(pos, yaw, pen_r=7.0, flip=False, beams=SON_BEAMS, bins=SON_BINS,
                 son_range=SON_RANGE, amp=1.0):
    """A sonar image of the inside of a cylinder of radius pen_r, traced from the
    real pose with the scene's own conventions (forward +x at yaw 0, starboard
    +z). `flip` reverses the beam order, which is the engine convention this file
    refuses to assume: the estimator has to find the wall either way."""
    p = np.asarray(pos, np.float64)
    p = np.array([p[0], 0.0, p[2]])                    # the cylinder is vertical: work in the plane
    fwd = np.array([math.cos(yaw), 0.0, -math.sin(yaw)])
    right = np.array([math.sin(yaw), 0.0, math.cos(yaw)])
    b = beam_bearings(beams, sign=1)                   # body bearing, +ve to starboard
    u = np.cos(b)[:, None] * fwd + np.sin(b)[:, None] * right
    pu = u @ p
    disc = pu ** 2 + pen_r ** 2 - float(p @ p)
    t = np.where(disc > 0.0, -pu + np.sqrt(np.maximum(disc, 0.0)), np.nan)
    img = np.zeros((beams, bins), np.float32)
    ok = np.isfinite(t) & (t > SON_MIN_RANGE) & (t < son_range)
    k = np.clip(np.nan_to_num(t) / son_range * bins, 0, bins - 1).astype(int)
    img[np.arange(beams)[ok], k[ok]] = amp
    return (img[::-1] if flip else img), t


def _selftest():
    ok = True

    def check(name, cond, extra=""):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {name} {extra}")

    # Two poses on the pen wall (wall to starboard, wall to port) x the two beam
    # orders the engine could have: the estimator finds the wall and its side in all four.
    for wall_stbd in (True, False):
        pos = np.array([7.0 - 1.5, -3.0, 0.0])         # on the +x meridian: the wall is outboard, +x
        yaw = math.pi / 2 if wall_stbd else -math.pi / 2   # tangential, either way round
        for flip in (False, True):
            sgn = -1 if flip else 1
            tag = f"stbd={int(wall_stbd)} flip={int(flip)}"
            img, t = _synth_image(pos, yaw, flip=flip)
            r = first_echo_ranges(img)
            b = beam_bearings(len(r), sign=sgn)
            tt = t[::-1] if flip else t
            check(f"first_echo_ranges {tag}", np.nanmax(np.abs(r - tt)) < 0.05,
                  f"max |dr| {np.nanmax(np.abs(r - tt)):.3f} m")
            side = wall_side_of(r, b)
            check(f"wall_side_of {tag}", side == (1 if wall_stbd else -1), f"-> {side:+d}")
            e = wall_estimate(r, side, b)
            check(f"wall_estimate {tag}",
                  e.seen and abs(e.standoff - 1.62) < 0.12 and side * e.tangent < 0.0,
                  f"standoff {e.standoff:.2f} perp {e.perp:.2f} tangent {math.degrees(e.tangent):.1f} deg")

    # the standoff must be monotone in the truth (and read long: the abeam point
    # is outside a 130 deg forward fan, so the nearest return is the fan edge)
    s = [wall_estimate(first_echo_ranges(_synth_image([7.0 - x, -3.0, 0.0], math.pi / 2)[0]),
                       1, beam_bearings()).standoff for x in (1.0, 1.5, 2.0, 2.5)]
    check("standoff monotone", all(a < b for a, b in zip(s, s[1:])), " ".join(f"{v:.2f}" for v in s))
    check("standoff reads long", all(a > b for a, b in zip(s, (1.0, 1.5, 2.0, 2.5))),
          " ".join(f"{a - b:+.2f}" for a, b in zip(s, (1.0, 1.5, 2.0, 2.5))))

    # empty image -> not seen
    check("empty image", not wall_estimate(first_echo_ranges(np.zeros((SON_BEAMS, SON_BINS), np.float32)),
                                           1, beam_bearings()).seen)

    # noise: seeded, repeatable, and the seed is the only difference
    a, b, c = RangeNoise(seed=0), RangeNoise(seed=0), RangeNoise(seed=1)
    va, vb, vc = [a(1.5) for _ in range(8)], [b(1.5) for _ in range(8)], [c(1.5) for _ in range(8)]
    check("RangeNoise same seed", va == vb)
    check("RangeNoise other seed", va != vc, f"rms {np.std(np.array(va) - np.array(vc)):.4f} m")

    # the vehicle: full forward thrust settles at the drag terminal speed
    r_ = Rov([0.0, -1.0, 0.0], 0.0)
    for _ in range(6000):
        r_.step((1.0, 0.0, 0.0, 0.0), 1.0 / 60.0)
    check("Rov terminal speed", abs(r_.body_vel()[0] - 0.8) < 0.02, f"{r_.body_vel()[0]:.3f} m/s")
    r_ = Rov([0.0, -1.0, 0.0], 0.0)
    for _ in range(600):
        r_.step((0.0, 0.0, 0.0, YAW_RATE_MAX * 2), 1.0 / 60.0)
    check("Rov yaw-rate limit", abs(r_.yaw_rate - YAW_RATE_MAX) < 1e-6, f"{math.degrees(r_.yaw_rate):.1f} deg/s")

    # closed loop against the synthetic cylinder: does it converge and go round?
    for x0, seed, flip in ((2.4, 0, False), (1.0, 3, True), (1.5, 7, False)):
        rov = Rov([7.0 - x0, -3.0, 0.0], -math.pi / 2)
        ctl = Controller(depth=-3.0, beam_sign=-1 if flip else 1)
        noise = RangeNoise(seed=seed)
        dt, every, hist = 1.0 / 60.0, 3, []
        for f in range(60 * 90):
            if f % every == 0:
                img, _ = _synth_image(rov.pos, rov.yaw, flip=flip)
                ctl.observe(img, noise)
                ctl.command(rov, every * dt)
            rov.step(ctl.last, dt)
            hist.append((7.0 - math.hypot(rov.pos[0], rov.pos[2]),
                         math.atan2(rov.pos[2], rov.pos[0]), ctl.meas))
        d = np.array([h[0] for h in hist])
        m = np.array([h[2] for h in hist])
        turned = np.unwrap(np.array([h[1] for h in hist]))
        tag = f"from {x0:.1f} m flip={int(flip)}"
        # the controller's job is the MEASURED standoff; the truth sits inside it
        # by the fan-edge bias, which is the sensor's, not the loop's
        check(f"loop holds the setpoint {tag}", abs(np.nanmean(m[-600:]) - STANDOFF) < 0.06,
              f"measured {np.nanmean(m[-600:]):.2f} m (setpoint {STANDOFF:.2f})")
        check(f"loop converges {tag}", NET_CLEAR < d[-600:].mean() < STANDOFF and d[-600:].std() < 0.03,
              f"true standoff {d[-600:].mean():.2f} m, ripple {d[-600:].std():.3f} m")
        check(f"loop travels {tag}", abs(turned[-1] - turned[0]) > math.radians(120),
              f"{math.degrees(turned[-1] - turned[0]):.0f} deg round the pen")
    print("selftest: " + ("all ok" if ok else "FAILURES"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        raise SystemExit(_selftest())
    print(__doc__)
