"""The film turbine a few years at sea: turbine_storm.py calls weather(o) on every object of
turbine_film.glb once, at load time (turbine.traverse(weather)).

What it does. The yellow transition piece and the foot of the white tower get a colour map,
painted here with numpy and wrapped round them by azimuth and height (their UVs are replaced: u =
azimuth, v = height). On it: the paint a little uneven, dirty run-off down the shell from the
platform's edge and from the flange above it, thin brown rust runs below the steel that is
welded or bolted on (the knee braces' feet, the J-tube clamps, the boat landing's stubs, the
seams, the tower flange's bolts), grime and slime in the splash zone, and on the tower a dirtier
foot with a few brown stains. Everything else is left as built.

Kept on purpose to a minimum: rust only where it runs DOWN from a source, no speckle or chips.

ASSUMED, all of it: placed and sized by eye to photographs of transition pieces and to another
simulator's weathered turbine. The splash zone is placed for turbine_storm.py, which raises the
turbine 6 m (the sea's mean level at the model's y = -6).
"""
import json
import math
import os

import numpy as np
from PIL import Image

import threepp as tp

_HERE = os.path.dirname(os.path.abspath(__file__))
WATER_Y = -6.0                       # the sea's mean level in the model's frame (turbine_storm.py lifts it 6 m)
SEAM_AZ = math.radians(52.7)         # where the map's u wraps: on the side away from the vessel, between vertices
TP_Y = (-10.0, 15.6)                 # the transition piece map's height range (model y)
TW_Y = (15.0, 45.0)                  # the tower map's: its foot only


def _spec():
    with open(os.path.join(_HERE, "turbine_spec.json"), "r", encoding="utf-8") as fh:
        spec = json.load(fh)
    with open(os.path.join(_HERE, "turbine_film_overrides.json"), "r", encoding="utf-8") as fh:
        over = json.load(fh)

    def lay(dst, src):
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                lay(dst[k], v)
            else:
                dst[k] = v
    lay(spec, over)
    return spec


def _val(d):
    return d["value"] if isinstance(d, dict) and "value" in d else d


def _srgb(x):
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


class Canvas:
    """A colour map wrapped round the structure: column = azimuth, row = height (row 0 at y0)."""

    def __init__(self, base, y0, y1, W, H, seed):
        self.y0, self.y1, self.W, self.H = y0, y1, W, H
        self.rng = np.random.default_rng(seed)
        self.col = np.empty((H, W, 3), np.float32)
        self.col[:] = base

    def noise(self, cy, cx, octaves=3):
        """Smooth value noise in 0..1, wrapping round in azimuth; cy x cx cells at the coarsest."""
        out = np.zeros((self.H, self.W), np.float32)
        amp, tot = 1.0, 0.0
        for _ in range(octaves):
            g = self.rng.random((cy + 1, cx + 1)).astype(np.float32)
            g[:, -1] = g[:, 0]
            out += amp * np.asarray(Image.fromarray(g, "F").resize((self.W, self.H), Image.BICUBIC))
            tot += amp
            amp, cy, cx = amp * 0.5, cy * 2, cx * 2
        return out / tot

    def row(self, y):
        return (y - self.y0) / (self.y1 - self.y0) * self.H

    def column(self, az):
        return ((az - SEAM_AZ) % (2 * math.pi)) / (2 * math.pi) * self.W

    def lerp(self, target, alpha, rows=slice(None), cols=slice(None)):
        a = alpha[..., None]
        self.col[rows, cols] = self.col[rows, cols] * (1.0 - a) + np.asarray(target, np.float32) * a

    def run(self, az, y_top, length, half_w, strength, colour, wobble=1.0, bloom=0.0, head=0.15, sharp=1.5):
        """A run down from (az, y_top): strongest a little below its source, fading to nothing at
        its foot; it meanders a few cm and narrows as it goes. half_w in pixels."""
        r0, r1 = int(self.row(y_top - length)), int(self.row(y_top))
        r0, r1 = max(r0, 0), min(r1, self.H - 1)
        if r1 <= r0:
            return
        rows = np.arange(r0, r1 + 1)
        t = (rows - r0) / max(r1 - r0, 1)                       # 0 at the foot, 1 at the source
        ph = self.rng.uniform(0, 6.28, 3)
        x0 = self.column(az)
        xc = x0 + wobble * (1.5 * np.sin(rows * 0.031 + ph[0]) + 0.8 * np.sin(rows * 0.093 + ph[1]))
        hw = half_w * (0.55 + 0.45 * t)
        fade = np.clip(t / 0.85, 0.0, 1.0) ** 0.8 * np.clip((1.0 - t) / head, 0.0, 1.0) ** 0.5
        fade *= 0.75 + 0.25 * np.sin(rows * 0.21 + ph[2]) ** 2
        R = int(math.ceil(half_w)) + 4
        dx = np.arange(-R, R + 1)
        cols = (np.floor(x0).astype(int) + dx) % self.W
        prof = np.clip(1.0 - np.abs((np.floor(x0) + dx)[None, :] - xc[:, None]) / hw[:, None], 0.0, 1.0) ** sharp
        alpha = strength * prof * fade[:, None]
        sub = self.col[r0:r1 + 1][:, cols]
        sub = sub * (1.0 - alpha[..., None]) + np.asarray(colour, np.float32) * alpha[..., None]
        self.col[r0:r1 + 1, cols] = sub
        if bloom > 0.0:                                         # the stain where it leaves the steel
            rb = int(self.row(y_top))
            ry, rx = max(int(half_w * 1.6), 3), max(int(half_w * 2.2), 4)
            yy = np.arange(rb - ry, rb + ry + 1)
            yy = yy[(yy >= 0) & (yy < self.H)]
            xx = np.arange(-rx, rx + 1)
            d = (((yy - rb) / ry) ** 2)[:, None] + ((xx / rx) ** 2)[None, :]
            a = bloom * np.clip(1.0 - d, 0.0, 1.0) ** 1.2
            cc = (np.floor(x0).astype(int) + xx) % self.W
            sub = self.col[yy][:, cc]
            self.col[yy[:, None], cc[None, :]] = sub * (1.0 - a[..., None]) + np.asarray(colour, np.float32) * a[..., None]

    def texture(self):
        img = (_srgb(self.col) * 255.0 + 0.5).astype(np.uint8)
        return tp.data_texture(img, True)


RUST = (0.16, 0.085, 0.045)          # linear: a dull brown, not orange
DIRT = (0.09, 0.083, 0.065)           # run-off: a grey-brown film
GRIME = (0.17, 0.16, 0.11)
SLIME = (0.10, 0.115, 0.06)


def _paint_tp(base, spec):
    TPs = spec["transition_piece"]
    R0 = 0.5 * _val(TPs["outer_diameter"])
    py = _val(TPs["platform_y"])
    FT = TPs.get("fittings", {})
    W = 2048
    H = int(round((TP_Y[1] - TP_Y[0]) / (2 * math.pi * R0 / W)))
    cv = Canvas(base, TP_Y[0], TP_Y[1], W, H, 11)
    rng = cv.rng
    ys = cv.y0 + (np.arange(H) + 0.5) / H * (cv.y1 - cv.y0)
    # paint not quite even: broad, faint
    n1, n2 = cv.noise(4, 10), cv.noise(3, 80, 2)                 # blotches; long vertical weathering
    cv.col *= (0.92 + 0.09 * n1 + 0.15 * (n2 - 0.5))[..., None]
    # what reads from a vessel's distance: broad stains that run a long way down the column
    # from under the deck, a third of a metre to more than a metre wide, soft-edged and uneven
    for _ in range(26):
        cv.run(rng.uniform(-math.pi, math.pi), py - 0.4 - rng.uniform(0, 1.5), rng.uniform(6.0, 17.0),
               rng.uniform(22, 80), rng.uniform(0.22, 0.42), DIRT, wobble=1.6, sharp=1.0)
    # splash zone: grime from the sea's reach up, slime round the mean level
    nz = cv.noise(6, 24)
    nv = cv.noise(2, 70, 2)                                   # its upper edge licks up in streaks
    g = np.clip((WATER_Y + 5.0 + 2.0 * (nz - 0.5) + 3.0 * (nv - 0.5) - ys[:, None]) / 6.0, 0.0, 1.0) ** 1.5
    cv.lerp(GRIME, 0.5 * g)
    s = np.clip((WATER_Y + 1.0 + 1.2 * (nz - 0.5) + 1.5 * (nv - 0.5) - ys[:, None]) / 2.5, 0.0, 1.0)
    cv.lerp(SLIME, 0.45 * s)
    # run-off: water off the platform runs down the shell under its edge, from the flange above
    for _ in range(70):
        bleed = rng.random() < 0.4
        cv.run(rng.uniform(-math.pi, math.pi), py - 0.3 - rng.uniform(0, 0.6), rng.uniform(1.5, 6.0) * (1 + bleed),
               rng.uniform(22, 50) if bleed else rng.uniform(6.0, 12.0), rng.uniform(0.45, 0.6) if bleed else
               rng.uniform(0.55, 0.75), DIRT, sharp=1.0 if bleed else 1.5)
    nd = cv.noise(2, 60, 2)
    under = np.clip((ys[:, None] - (py - 2.2 - 1.6 * nd)) / 1.6, 0.0, 1.0) * (ys[:, None] < py)
    cv.lerp(DIRT, 0.22 * under)
    top = _val(TPs["top_y"])
    for _ in range(14):                                       # above the deck, from the tower flange
        cv.run(rng.uniform(-math.pi, math.pi), top - 0.4, rng.uniform(0.8, 2.3), rng.uniform(4.0, 10.0),
               rng.uniform(0.35, 0.55), DIRT)
    # rust: thin runs from steel bolted or welded on
    src = []
    for k in range(12):                                       # the knee braces' feet
        if rng.random() < 0.6:
            src.append((2 * math.pi * (k + 0.5) / 12 + rng.uniform(-0.01, 0.01), py - 3.25))
    for a_deg in FT.get("j_tube_azimuths_deg", []):          # the J-tubes' clamps
        for yk in np.arange(_val(TPs["bottom_y"]) + 2.0, py - 1.0, 3.2):
            if rng.random() < 0.5 and yk > WATER_Y + 1.0:
                src.append((math.radians(a_deg) + rng.choice([-1, 1]) * 0.012, yk - 0.1))
    az_bl = math.radians(_val(TPs["boat_landing_azimuth_deg"]))
    BL = TPs["boat_landing"]
    ya, yb = BL["y_range"]
    for sgn in (-1.0, 1.0):                                   # the boat landing's stubs
        a = az_bl + sgn * 0.5 * BL["fender_spacing"] / R0
        for yk in (0.5 * (ya + yb) + 1.0, yb - 0.6):
            if rng.random() < 0.7:
                src.append((a, yk - 0.18))
    LD = FT.get("landing_deck")
    if LD:                                                    # the landing deck's beam and brace roots
        a0, hw = math.radians(LD["azimuth_deg"]), 0.62 * 0.5 * LD["width"]
        for sgn in (-1.0, 1.0):
            a = a0 - sgn * math.asin(hw / R0)
            src += [(a, py - 0.7), (a + 0.01, py - 4.15)]
    for yk in FT.get("seam_y", [])[1:]:                       # a weld here and there
        for _ in range(2):
            src.append((rng.uniform(-math.pi, math.pi), yk))
    for a, y in src:
        cv.run(a, y, rng.uniform(1.0, 4.0), rng.uniform(4.0, 6.5), rng.uniform(0.7, 0.85), RUST,
               wobble=0.6, bloom=0.3)
    for _ in range(9):                                        # the tower flange's bolts, onto the yellow above the deck
        cv.run(rng.uniform(-math.pi, math.pi), top - 0.36, rng.uniform(0.6, 1.8), rng.uniform(2.5, 4.0),
               rng.uniform(0.55, 0.75), RUST, wobble=0.4, bloom=0.25)
    return cv


def _paint_tower(base):
    rt = 0.5 * 10.0                                           # the tower's foot radius (turbine_spec: 10 m)
    W = 2048
    H = int(round((TW_Y[1] - TW_Y[0]) / (2 * math.pi * rt / W)))
    cv = Canvas(base, TW_Y[0], TW_Y[1], W, H, 23)
    rng = cv.rng
    ys = cv.y0 + (np.arange(H) + 0.5) / H * (cv.y1 - cv.y0)
    plain = np.clip((cv.y1 - 4.0 - ys) / 6.0, 0.0, 1.0)[:, None]   # the map fades to plain paint at its top
    n1, n2 = cv.noise(3, 10), cv.noise(3, 80, 2)
    cv.col *= (1.0 + plain * (0.07 * (n1 - 0.5) + 0.05 * (n2 - 0.5)))[..., None]
    # the foot: dirtier for a couple of metres over the flange, its upper edge streaky
    nf = cv.noise(4, 60)
    foot = np.clip((cv.y0 + 1.6 + 1.6 * nf - ys[:, None]) / 1.6, 0.0, 1.0)
    cv.lerp((0.30, 0.27, 0.21), 0.45 * foot)
    for _ in range(7):                                        # brown stains, soft
        cv.run(rng.uniform(-math.pi, math.pi), cv.y0 + rng.uniform(2.0, 5.0), rng.uniform(1.0, 3.0),
               rng.uniform(10, 22), rng.uniform(0.4, 0.55), (0.22, 0.13, 0.07), sharp=1.0)
    for _ in range(18):                                       # a faint grey run-off
        cv.run(rng.uniform(-math.pi, math.pi), cv.y0 + rng.uniform(4.0, 16.0), rng.uniform(2.0, 9.0),
               rng.uniform(3, 9), rng.uniform(0.06, 0.12), (0.30, 0.30, 0.28))
    return cv


def _wrap(o, cv):
    """Replace o's UVs with (azimuth, height) on the canvas, and its colour by the canvas."""
    P = np.asarray(o.geometry.get_attribute("position"), dtype=np.float64).reshape(-1, 3)
    az = np.arctan2(P[:, 2], P[:, 0])
    u = ((az - SEAM_AZ) % (2 * math.pi)) / (2 * math.pi)
    v = np.clip((P[:, 1] - cv.y0) / (cv.y1 - cv.y0), 0.5 / cv.H, 1.0 - 0.5 / cv.H)
    o.geometry.set_attribute("uv", np.stack([u, v], 1).astype(np.float32))
    o.material.map = cv.texture()
    o.material.color = tp.Color(1.0, 1.0, 1.0)


_SPEC = None


def weather(o):
    global _SPEC
    if not isinstance(o, tp.Mesh):
        return
    name = o.material.name
    if name not in ("tp_yellow", "tower_grey"):
        return
    if _SPEC is None:
        _SPEC = _spec()
    c = o.material.color
    base = np.array([c.r, c.g, c.b], np.float32)
    _wrap(o, _paint_tp(base, _SPEC) if name == "tp_yellow" else _paint_tower(base))
