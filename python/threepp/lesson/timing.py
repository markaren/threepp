"""Time in a lesson: easing curves, named beats (`Timeline`), keyframes (`Keys`,
`OrbitCamera`), and the film's clock with holds (`TimeMap`)."""
from __future__ import annotations

import math

import numpy as np

__all__ = ["clamp01", "smooth", "smoother", "ease_out", "ease_out_back", "lerp", "remap", "Timeline",
           "envelope", "Keys", "fade_in_out", "OrbitCamera", "TimeMap"]


def clamp01(x):
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else float(x)


# The easings take a scalar or a numpy array (per-point staggers).
def _c01(x):
    return clamp01(x) if np.isscalar(x) else np.clip(x, 0.0, 1.0)


def smooth(x):
    """Smoothstep: zero velocity at both ends."""
    x = _c01(x)
    return x * x * (3.0 - 2.0 * x)


def smoother(x):
    """Smootherstep: zero velocity AND acceleration at both ends."""
    x = _c01(x)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def ease_out(x):
    x = _c01(x)
    return 1.0 - (1.0 - x) ** 3


def ease_out_back(x, s=1.4):
    x = _c01(x) - 1.0
    return 1.0 + x * x * ((s + 1.0) * x + s)


def lerp(a, b, t):
    return a + (b - a) * t


def remap(t, a, b):
    """Linear 0..1 progress of t through [a, b], clamped."""
    return clamp01((t - a) / max(b - a, 1e-9))


class Timeline:
    """Named beats. Times are seconds from the start of the film."""

    def __init__(self):
        self.beats = {}
        self.order = []

    def add(self, name, start, end):
        """Add a beat; adding a name again redefines its span and keeps its place in the order."""
        if name not in self.beats:
            self.order.append(name)
        self.beats[name] = (float(start), float(end))
        return self

    def then(self, name, duration, gap=0.0):
        """Append a beat right after the previous one."""
        start = self.beats[self.order[-1]][1] + gap if self.order else 0.0
        return self.add(name, start, start + duration)

    @property
    def duration(self):
        return max(e for _, e in self.beats.values()) if self.beats else 0.0

    def span(self, name):
        return self.beats[name]

    def start(self, name):
        return self.beats[name][0]

    def end(self, name):
        return self.beats[name][1]

    def local(self, name, t):
        """Seconds since the beat started (negative before it)."""
        return t - self.beats[name][0]

    def active(self, name, t, pad=0.0):
        s, e = self.beats[name]
        return s - pad <= t <= e + pad

    def p(self, name, t, ease=smoother):
        """Eased 0..1 progress through the beat."""
        s, e = self.beats[name]
        return ease(remap(t, s, e))

    def fade(self, name, t, fade_in=0.5, fade_out=0.5, delay=0.0, early_out=0.0):
        """0 -> 1 -> 0 envelope over the beat (smoothstepped edges)."""
        s, e = self.beats[name]
        s += delay
        e -= early_out
        if t < s or t > e:
            return 0.0
        a = smooth((t - s) / fade_in) if fade_in > 0 else 1.0
        b = smooth((e - t) / fade_out) if fade_out > 0 else 1.0
        return min(a, b)

    def which(self, t):
        for n in self.order:
            s, e = self.beats[n]
            if s <= t < e:
                return n
        return self.order[-1] if self.order else None


def envelope(t, t_in, t_out, fade_in=0.5, fade_out=0.5):
    """Standalone fade envelope: rises at t_in, falls to 0 at t_out."""
    if t < t_in or t > t_out:
        return 0.0
    a = smooth((t - t_in) / fade_in) if fade_in > 0 else 1.0
    b = smooth((t_out - t) / fade_out) if fade_out > 0 else 1.0
    return min(a, b)


class Keys:
    """Keyframed vector value with smoother-step interpolation between keys.

    keys = [(time, value), ...]. Values are anything numpy can add.
    """

    def __init__(self, keys, ease=smoother):
        self.keys = sorted(((float(t), np.asarray(v, np.float64)) for t, v in keys), key=lambda k: k[0])
        self.ease = ease

    def __call__(self, t):
        ks = self.keys
        if t <= ks[0][0]:
            return ks[0][1].copy()
        for (t0, v0), (t1, v1) in zip(ks, ks[1:]):
            if t <= t1:
                u = self.ease((t - t0) / max(t1 - t0, 1e-9))
                return v0 * (1 - u) + v1 * u
        return ks[-1][1].copy()


def fade_in_out(t, duration, fade_in=0.8, fade_out=0.9):
    """Veil for a film that rises from black and returns to it (1 = black), for Hud.fade."""
    return 1.0 - min(smooth(remap(t, 0.0, fade_in)), 1.0 - smooth(remap(t, duration - fade_out, duration)))


class OrbitCamera:
    """A keyframed orbit around a look point.

    keys = [(time, azimuth deg, elevation deg, distance m, look point, fov deg), ...].
    By default the look points are in a Z-up (robot) frame and azimuth is measured in
    its XY plane from +x. With `frame="yup"` they are in the stage's Y-up world and
    azimuth is measured in the XZ plane from +x towards -z (the same direction once
    Z-up is turned into Y-up). `follow(t)`, if given, returns a point (same frame) that
    the keyed look point is added to, and `heading(t)` an angle (rad) added to the
    azimuth: together a chase camera. `drift` is ((amplitude deg, rad/s) for azimuth,
    (amplitude deg, rad/s) for elevation): slow sinusoids so held shots never freeze.
    Called with t, returns (eye, look, fov) in the Y-up world, ready for `Stage.look`.
    """

    def __init__(self, keys, drift=((1.2, 0.21), (0.6, 0.17)), frame="zup", follow=None, heading=None):
        self.orbit = Keys([(k[0], [k[1], k[2], k[3], k[5]]) for k in keys])
        self.target = Keys([(k[0], k[4]) for k in keys])
        self.drift = drift
        self.frame = frame
        self.follow = follow
        self.heading = heading

    def __call__(self, t):
        az, el, dist, fov = (float(v) for v in self.orbit(t))
        look = self.target(t)
        if self.follow is not None:
            look = look + np.asarray(self.follow(t), float)
        (az_amp, az_rate), (el_amp, el_rate) = self.drift
        az += az_amp * math.sin(az_rate * t)
        el += el_amp * math.sin(el_rate * t + 1.0)
        a, e = math.radians(az), math.radians(el)
        if self.heading is not None:
            a += float(self.heading(t))
        if self.frame == "yup":
            eye = look + dist * np.array([math.cos(e) * math.cos(a), math.sin(e), -math.cos(e) * math.sin(a)])
            return eye, look, fov
        eye = look + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        zw = lambda p: np.array([p[0], p[2], -p[1]])  # noqa: E731  Z-up -> Y-up
        return zw(eye), zw(look), fov


class TimeMap:
    """The film's clock against a lesson's own (script) clock. A hold (a, d, tag) stops the
    script clock at a for d seconds of film: the picture stands still there, and render() is
    told which hold it is in and how far through (0..1)."""

    def __init__(self, holds=()):
        self.holds = sorted(holds, key=lambda h: h[0])

    def add(self, a, d, tag):
        self.holds = sorted(self.holds + [(a, d, tag)], key=lambda h: h[0])

    def film(self, ts):
        """Script time -> film time (a moment at a hold maps to before it)."""
        return ts + sum(d for a, d, _ in self.holds if a < ts)

    def script(self, tf):
        """Film time -> (script time, (tag, 0..1 progress) inside a hold, else None)."""
        shift = 0.0
        for a, d, tag in self.holds:
            if tf < a + shift:
                break
            if tf < a + shift + d:
                return a, (tag, (tf - a - shift) / d)
            shift += d
        return tf - shift, None
