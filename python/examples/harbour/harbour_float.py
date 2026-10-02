"""Floating marks for the harbour scene, and small helpers the scene shares.

BuoyFloat puts one mark from buoys_hydro.json (schema threepp.buoy_hydro/1) on the FFT sea:

- heave: the draft table (waterline_y -> displaced volume) against the mark's weight plus the
  chain's pull at the eye, both vertical, with the file's heave added mass. The design point of
  the table is the mark's equilibrium, so the file's heave period comes out of it.
- roll / pitch: small angles, restoring moment rho g V GM (the file's GM, which already counts the
  chain) toward a fraction of the local wave slope, the file's roll inertia (added inertia
  included). The file's roll period comes out of it.
- surge / sway: the Froude-Krylov push of the wave slope on the displaced water (decayed over the
  draft), against a horizontal spring to the anchor (the chain) and a linear drag: the mark drifts
  in a small watch circle.
- yaw is held (a demo yaws the mark so its numbers face the fairway).

Fixed 60 Hz steps, semi-implicit Euler; every natural frequency is well under the step rate.
Sample the sea AFTER the frame's render (sample_height reads the last rendered field), as
usv_rig.StripHull does.

Frame of a mark (buoys README): origin on its axis at the lowest point of the buoyant body, +Y up.
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))          # examples/ (usv_rig)

from usv_rig import quat_of, rot_x, rot_y, rot_z  # noqa: E402

BUOYS_HYDRO = os.path.join(HERE, "buoys_hydro.json")
SEA_MASK = 0b011                   # swell + mid band, as the boats (usv_rig.BUOY_MASK)
# The wave pressure a float feels decays with depth as exp(-k d) (the Smith effect): a 4.6 m spar
# barely feels the short chop, a 0.3 m float rides all of it. Each band is weighted by exp(-k d)
# at one representative wavelength (ASSUMED: the swell cascade's energy near 20 m, the mid
# cascade's near 6 m, for the harbour's 5 m/s wind over ~4 km of fetch), d = the mark's draft.
BAND_LAMBDA = (20.0, 6.0)          # cascade 0, cascade 1 (m)


def load_marks(path=BUOYS_HYDRO):
    with open(path, encoding="utf-8") as fh:
        H = json.load(fh)
    if H.get("schema") != "threepp.buoy_hydro/1":
        raise ValueError(f"{path}: schema {H.get('schema')!r}, expected threepp.buoy_hydro/1")
    return H


class BuoyFloat:
    """One mark on the sea, held in a watch circle around its anchor.

    mark        a key of buoys_hydro.json "marks" (lateral_port, special, mooring_buoy, ...)
    anchor_xz   world (x, z) of the anchor; the mark rests right above it
    yaw         heading of the mark's +X in world (radians, about +Y, as rot_y)
    zeta_*      fractions of critical damping (light: a spar rings)
    watch_period  period of the surge/sway spring to the anchor (s)
    slope_follow  how far the restoring moment leans toward the local wave slope (0 = upright)
    fk_decay    Froude-Krylov decay of the wave's horizontal push over the draft
    depth_decay weight each band's level and slope by exp(-k draft) (BAND_LAMBDA); False = the
                surface level as it is (phase A: a spar then heaves with the short chop)"""

    def __init__(self, mark, ocean, anchor_xz, yaw=0.0, hydro=None, *, zeta_heave=0.08, zeta_tilt=0.06,
                 zeta_drift=0.6, watch_period=40.0, slope_follow=0.5, fk_decay=0.5, depth_decay=True):
        H = hydro if hydro is not None else load_marks()
        m = H["marks"][mark]
        D = m["design"]
        self.mark, self.ocean = mark, ocean
        self.rho, self.g = H["rho"], H["g"]
        self.mass = m["mass"]
        self.line_load = m["line_load_at_eye"]
        self.weight = (self.mass + self.line_load) * self.g          # the chain's pull is vertical at rest
        tab = np.array(m["draft_table"]["rows"], float)
        cols = m["draft_table"]["columns"]
        self.t_wl = tab[:, cols.index("waterline_y")]
        self.t_vol = tab[:, cols.index("displaced_volume")]
        self.wl = D["waterline_y"]                                    # design waterline above the origin
        self.m_heave = self.mass + D["heave_added_mass"]
        self.vol0 = D["displaced_volume"]
        self.k_tilt = self.rho * self.g * self.vol0 * D["gm"]         # N m / rad
        self.i_tilt = D["roll_inertia_incl_added"]
        k_heave = self.rho * self.g * D["waterplane_area"]
        self.c_heave = 2.0 * zeta_heave * math.sqrt(k_heave * self.m_heave)
        self.c_tilt = 2.0 * zeta_tilt * math.sqrt(self.k_tilt * self.i_tilt)
        # horizontal: the body plus the water it pushes (Ca = 1, a vertical cylinder)
        self.m_drift = self.mass + self.rho * self.vol0
        wn = 2.0 * math.pi / watch_period
        self.k_chain = self.m_drift * wn * wn
        self.c_drift = 2.0 * zeta_drift * math.sqrt(self.k_chain * self.m_drift)
        self.fk = self.rho * self.vol0 * 2.0 * fk_decay               # (1 + Ca) rho V, decayed
        self.slope_follow = slope_follow
        self.anchor = np.array(anchor_xz, float)
        self.yaw = yaw
        self.periods = (D["heave_period_s"], D["roll_period_s"])
        self.band_w = tuple(math.exp(-2.0 * math.pi / lam * self.wl) if depth_decay else 1.0 for lam in BAND_LAMBDA)
        # state: the axis point at the design waterline (world), its velocity, two tilt angles + rates
        self.p = np.array([self.anchor[0], 0.0, self.anchor[1]])
        self.v = np.zeros(3)
        self.tilt = np.zeros(2)            # (about world X, about world Z), radians
        self.tilt_w = np.zeros(2)
        self.eta = 0.0
        self.grad = np.zeros(2)
        self.log = []                      # (heave above the local sea, roll deg, pitch deg, drift m)

    # ---- the sea at the mark
    def level(self, x, z):
        """The sea level the float feels at (x, z): each band decayed over the draft."""
        w0, w1 = self.band_w
        oc = self.ocean
        if w0 == 1.0 and w1 == 1.0:
            return oc.sample_height(x, z, SEA_MASK)
        return w0 * oc.sample_height(x, z, 0b001) + w1 * oc.sample_height(x, z, 0b010)

    def sample_water(self, h=0.75):
        x, z = float(self.p[0]), float(self.p[2])
        self.eta = self.level(x, z)
        gx = (self.level(x + h, z) - self.level(x - h, z)) / (2.0 * h)
        gz = (self.level(x, z + h) - self.level(x, z - h)) / (2.0 * h)
        self.grad = np.array([gx, gz])

    def seat(self):
        """At rest on its design waterline over the anchor, on the current sea."""
        self.p[0], self.p[2] = self.anchor
        self.sample_water()
        self.p[1] = self.eta
        self.v[:] = 0.0
        self.tilt[:] = 0.0
        self.tilt_w[:] = 0.0

    def volume(self, immersion):
        return float(np.interp(immersion, self.t_wl, self.t_vol, left=0.0, right=self.t_vol[-1]))

    def step(self, dt):
        cos_t = math.cos(min(float(np.hypot(*self.tilt)), 1.2))
        immersion = self.wl + (self.eta - self.p[1]) / cos_t
        fb = self.rho * self.g * self.volume(immersion)
        a_y = (fb - self.weight - self.c_heave * self.v[1]) / self.m_heave
        gx, gz = self.grad
        off = self.p[[0, 2]] - self.anchor
        a_h = (-self.fk * self.g * np.array([gx, gz]) - self.k_chain * off - self.c_drift * self.v[[0, 2]]) / self.m_drift
        # the axis leans toward the local surface normal (-gx, 1, -gz): about X by -gz, about Z by +gx
        target = self.slope_follow * np.array([-gz, gx])
        alpha = (-self.k_tilt * (self.tilt - target) - self.c_tilt * self.tilt_w) / self.i_tilt
        self.v += np.array([a_h[0], a_y, a_h[1]]) * dt
        self.p += self.v * dt
        self.tilt_w += alpha * dt
        self.tilt += self.tilt_w * dt

    def advance(self, dt, substeps=1):
        """Sample the (already rendered) sea, then step; logs heave / roll / pitch / drift."""
        self.sample_water()
        for _ in range(substeps):
            self.step(dt / substeps)
        r = self.readout()
        self.log.append((r["heave"], r["roll"], r["pitch"], r["drift"]))

    # ---- scene
    def rotation(self):
        return rot_z(self.tilt[1]) @ rot_x(self.tilt[0]) @ rot_y(self.yaw)

    def place(self, obj):
        R = self.rotation()
        o = self.p - R @ np.array([0.0, self.wl, 0.0])
        obj.position.set(float(o[0]), float(o[1]), float(o[2]))
        obj.quaternion.set(*quat_of(R))

    def to_world(self, pv):
        """A point in the mark's frame (origin at the bottom of the body) to world."""
        R = self.rotation()
        return self.p + R @ (np.asarray(pv, float) - np.array([0.0, self.wl, 0.0]))

    def readout(self):
        """heave: the design waterline point above the local sea (m, + = riding high);
        roll about the mark's +X, pitch about its +Z (deg); drift from the anchor (m)."""
        R = self.rotation()
        ax = R[:, 1]
        fx, fz = math.cos(self.yaw), -math.sin(self.yaw)        # the mark's +X in world xz
        sx, sz = math.sin(self.yaw), math.cos(self.yaw)         # its +Z
        roll = math.degrees(math.atan2(ax[0] * sx + ax[2] * sz, ax[1]))
        pitch = math.degrees(math.atan2(-(ax[0] * fx + ax[2] * fz), ax[1]))
        return {"heave": float(self.p[1] - self.eta), "roll": roll, "pitch": pitch,
                "drift": float(np.hypot(*(self.p[[0, 2]] - self.anchor)))}

    def summary(self, skip=0):
        a = np.array(self.log[skip:]) if len(self.log) > skip else np.zeros((1, 4))
        return (f"{self.mark}: heave {a[:, 0].min():+.3f}..{a[:, 0].max():+.3f} m, roll {a[:, 1].min():+.2f}.."
                f"{a[:, 1].max():+.2f} deg, pitch {a[:, 2].min():+.2f}..{a[:, 2].max():+.2f} deg, "
                f"drift {a[:, 3].max():.2f} m (periods in the file: heave {self.periods[0]:.2f} s, "
                f"roll {self.periods[1]:.2f} s)")


def sag_line(p0, p1, sag, n=24):
    """Points of a hanging line from p0 to p1, a parabola sagging `sag` metres at mid-span."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    t = np.linspace(0.0, 1.0, n)[:, None]
    P = p0 + (p1 - p0) * t
    P[:, 1] -= sag * 4.0 * t[:, 0] * (1.0 - t[:, 0])
    return P
