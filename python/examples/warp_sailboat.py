"""A sailboat whose sails are Warp cloth, sailing the FFT ocean in the mist.

The two sails are real cloth: a Warp GPU solver holds ~9k particles together
with structural / shear / bending constraints, and the only thing pushing them
around is the wind. Every particle feels the flat-plate membrane pressure

    a = (rho_air / 2) * (n . v_rel) * |n . v_rel| * n / sigma

along its own normal, quadratic in the normal component of the apparent wind and
divided by the cloth's areal density -- so the answer does not depend on how
finely the sail happens to be meshed. Sum that force over the sail and you have
the load on the rig, and THAT is what drives the boat: the component along the
hull is thrust, the component across it heels her over. Ease the mainsheet and
the sail spills wind, the boat stands up and slows. Point too close to the wind
and the luff starts to shake, the drive collapses, and she stops. None of that is
scripted -- it falls out of the cloth.

Honest about the model: this is normal pressure only, no circulation. A real sail
is a thin foil and makes far more lift close-hauled than a flat plate does, so
this boat is strong on a reach and comes upwind more grudgingly than the real
thing. The "rig efficiency" slider is the fudge factor that buys some of it back.

The sea is threepp's 3-cascade FFT ocean. It supplies wave height for buoyancy
(the hull pitches and rolls on the slope it is actually sitting on), it takes the
wake foam, and it follows the boat -- the vertex-density warp centre relocates
the whole grid in world space, so the water never runs out. The fog is the
unified Vulkan air medium, and the rig cuts shafts through it when you look
toward the sun.

The gulls are threepp's own ambient flock (extras/fauna): boids that fly, bank
into their turns, and land. The islet is scanned for landable rock at startup,
so they sit on it between flights and scatter when the boat runs them down.

The islet and its lighthouse are procedural -- a ridged-noise crag emitted as a
non-indexed soup so every triangle keeps its own normal (hard facets; the usual
`flatShading` lever is a no-op on the Vulkan backend), and a lofted surface of
revolution for the tower. The lamp is an emissive mesh, which on Vulkan is a
real light source, so it throws light into the fog.

The day runs. `celestial()` puts the sun and the moon where they belong for
~59 N in mid-August, and the sky is a Preetham bake in numpy that is BOTH the
background and the IBL -- so the light on the sails always comes from the sky
you can see, whether that is a dawn haze, a black squall or a moon. The weather
is a small state machine with ramps in seconds: cloud closes, wind builds, the
sea gets up, rain starts, the deck goes dark and glossy, and when it clears
there is a rainbow opposite the sun because the geometry says there should be.

    pip install warp-lang
    python warp_sailboat.py                     # window; drag to orbit, Esc quits
    python warp_sailboat.py --tod 19.7 --weather clearing
    python warp_sailboat.py --wind 14           # override the weather's wind
    python warp_sailboat.py --birds 60          # more gulls
    python warp_sailboat.py --timelapse             # run the day at 0.35 h/s
    python warp_sailboat.py --shot 25 --tod 23.5 --weather night --out night.png

Helm:  A / D steer      Q / E ease / trim main     Z / C ease / trim jib
       T auto-trim      P autopilot (hold course)  R reset
Sky:   1-5 jump to dawn / morning / squall / golden hour / night
       N toggle rain    K time-lapse the day       L fire a lightning strike
Drone: G fly the camera drone    V switch to its own view (FPV)

Needs a Vulkan build (-DTHREEPP_WITH_VULKAN=ON). Warp falls back to CPU if no
CUDA device is present -- slower, same picture.
"""
import math
import os
import sys
import time

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp


def cli_arg(flag, default, cast):
    if flag in sys.argv:
        k = sys.argv.index(flag)
        if k + 1 < len(sys.argv) and not sys.argv[k + 1].startswith("--"):
            return cast(sys.argv[k + 1])
    return default


SHOT = "--shot" in sys.argv
SHOT_T = cli_arg("--shot", 25.0, float)
TOD0 = cli_arg("--tod", 5.2, float) % 24.0        # clock time, hours
WEATHER0 = cli_arg("--weather", "mist", str)
WIND0 = cli_arg("--wind", 0.0, float)             # 0 = let the weather decide

if not tp.HAS_VULKAN or not tp.vulkan_available():
    print("This example needs the Vulkan backend (configure with "
          "-DTHREEPP_WITH_VULKAN=ON) and a working Vulkan loader.")
    sys.exit(0)

# --------------------------------------------------------------------------- #
#  Conventions. Bow at +Z, starboard at +X, up at +Y -- the same frame the C++
#  ocean showcase integrates in, so heading is (sin yaw, 0, cos yaw) and the
#  lateral axis is (cos yaw, 0, -sin yaw).
# --------------------------------------------------------------------------- #
LOA, BEAM, DRAFT, FREEBOARD = 9.6, 3.0, 0.62, 0.85
NS, NP = 40, 26              # hull stations, points per half-section

MAST_Z = 0.55                # mast step, m forward of the hull origin
MAST_H = 12.2                # waterline to masthead
GOOSE_Y = 1.95               # gooseneck (boom pivot) height
DECK_Y = 1.05                # working deck height near the mast
LUFF = 9.4                   # mainsail luff, up the mast
FOOT = 3.9                   # mainsail foot, along the boom
ROACH = 0.55                 # leech curve outboard of the head-clew line
TACK_Z = 4.30                # jib tack, on the foredeck
JIB_HEAD = 10.9              # forestay hound height
JIB_FOOT = 3.45              # jib tack to clew
JIB_CLEW_Y = 0.85            # jib clew height above its tack

# Cloth grids: (chordwise, along the luff). ~9k particles total.
MAIN_NU, MAIN_NV = 48, 64
JIB_NU, JIB_NV = 36, 50

DT = 1.0 / 480.0
# A sail under 9 m/s of wind sees ~200 m/s2 of pressure -- two orders above
# gravity. Small steps are what let an under-relaxed Jacobi solve keep up with
# that; at 1/240 the free leech flogged hard enough that neighbouring normals
# cancelled and the rig lost two thirds of its drive.
SUBSTEPS = 8                 # 8 x 1/480 = one 60 fps frame of sim time
ITERATIONS = 10              # constraint projections per substep
RHO_AIR = 1.225

# Hull: a 9.6 m fin-keel cruiser, ~3.5 t. Hull speed 1.34*sqrt(LWL_ft) knots.
MASS = 3500.0
HULL_SPEED = 1.34 * math.sqrt(LOA * 0.9 / 0.3048) * 0.5144   # ~3.6 m/s, 7 kn
GM = 1.10                    # metacentric height -- the stiffness of the boat
I_ROLL = MASS * 1.3 ** 2     # roll inertia; with GM gives a ~2.5 s roll period
ROLL_DAMP = 0.35 * 2.0 * math.sqrt(I_ROLL * MASS * 9.81 * GM)
# Centre of effort sits ~40% up the luff (5.7 m over the water); the centre of
# lateral resistance is ~1.1 m BELOW it, mid-keel. The heeling lever is the gap.
CE_HEIGHT = 6.8
DRAG_Q = 64.0                # quadratic hull drag, N/(m/s)^2
DRAG_WAVE = 1100.0           # wave-making wall past hull speed
KEEL_DAMP = 2600.0           # lateral resistance of keel + hull
LEEWAY_FRAC = 0.16           # share of sail side force that becomes drift
TURN_GAIN = 0.085            # yaw rate per (m/s * rad of rudder)
WEATHER_HELM = 0.055         # heel-induced round-up
RUDDER_MAX = math.radians(35.0)
BUOY_MASK = 0b011            # swells + mid band; cascade 2 is chop the hull ignores
SAIL_WIND_MAX = 16.0         # apparent wind the cloth solver is allowed to see


# --------------------------------------------------------------------------- #
#  Sky, sun, moon: a day/night cycle for ~59 N in mid-August.
#
#  The sky is a Preetham daylight model with a night extension, baked to a
#  float equirect in numpy and handed to the renderer as scene.environment --
#  which on Vulkan is BOTH the background and the IBL, so whatever the bake
#  says is what lights the boat. The atmosphere is therefore the single source
#  of truth: the sun light's colour is the same Beer-Lambert transmittance the
#  sky was painted with, and the fog colour is the sky's own horizon.
#
#  Frame: +Z is north, +X is east, +Y is up -- the same compass the wind
#  bearings already use ("wind from 0 deg" blows from +Z).
# --------------------------------------------------------------------------- #
LAT = math.radians(59.0)          # a skerry off the Norwegian coast
DECL = math.radians(17.5)         # solar declination, mid-August
SOLAR_NOON = 12.85                # clock time of the sun's transit
MOON_DECL = math.radians(5.0)     # a summer moon rides low and to the south
_SIN_LAT, _COS_LAT = math.sin(LAT), math.cos(LAT)


def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0.0, 1.0) if isinstance(x, np.ndarray) \
        else min(max((x - a) / (b - a), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _body_dir(decl, hour_angle):
    """Unit vector toward a body at the given declination and hour angle.

    Straight spherical astronomy, which is what makes the arc read as a real
    day rather than a sine wave: at 59 N in August the sun rises in the NE,
    transits 48 deg up in the south, sets in the NW, and then only dips ~12 deg
    below the horizon -- so the "night" is a long blue twilight, not a void.
    """
    sd, cd = math.sin(decl), math.cos(decl)
    ch = math.cos(hour_angle)
    up = _SIN_LAT * sd + _COS_LAT * cd * ch
    north = sd * _COS_LAT - cd * _SIN_LAT * ch
    east = -cd * math.sin(hour_angle)
    v = np.array([east, up, north], np.float64)
    return v / max(float(np.linalg.norm(v)), 1e-9)


class Celestial:
    """Where the sun and the moon are, and how much daylight there is."""

    __slots__ = ("t", "sun_dir", "moon_dir", "sun_y", "daylight", "moon_up",
                 "sun_elev_deg", "sun_az_deg")

    def __init__(self, t_hours):
        self.t = t_hours % 24.0
        ha = math.radians(15.0 * (self.t - SOLAR_NOON))
        self.sun_dir = _body_dir(DECL, ha)
        # Roughly opposite the sun, so it is up when the sun is down.
        self.moon_dir = _body_dir(MOON_DECL, ha + math.pi)
        self.sun_y = float(self.sun_dir[1])
        self.daylight = smoothstep(-0.10, 0.05, self.sun_y)
        self.moon_up = min(max(float(self.moon_dir[1]) / 0.30, 0.0), 1.0)
        self.sun_elev_deg = math.degrees(math.asin(max(-1.0, min(1.0, self.sun_y))))
        self.sun_az_deg = math.degrees(math.atan2(self.sun_dir[0], self.sun_dir[2]))


def celestial(t_hours):
    return Celestial(t_hours)


# ---- the atmosphere --------------------------------------------------------- #
SKY_TURBIDITY = 3.6               # nominal; the weather moves it (see turbidity())
SKY_RAYLEIGH = 3.0
SKY_MIE_COEFF = 0.006
SKY_MIE_G = 0.80
_TOTAL_RAYLEIGH = np.array([5.804542996261093e-6, 1.3562911419845635e-5,
                            3.0265902468824876e-5], np.float32)
_MIE_CONST = np.array([1.8399918514433978e14, 2.7798023919660528e14,
                       4.0790479543861094e14], np.float32)
_LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)


def turbidity(wx):
    """How much junk is in the air, from the weather that put it there.

    A fixed turbidity means the clear noon and the dawn haze are the same
    atmosphere, and one of the two always looks wrong -- a milky blue noon or a
    thin dawn. Coverage and fog density already say how thick the air is, so
    they say it here too.
    """
    return 2.8 + 3.0 * wx["coverage"] + 300.0 * wx["fog"]


def scatter_coeffs(sun_y, turb=SKY_TURBIDITY):
    """Rayleigh / Mie coefficients for the current sun height.

    ONE source for both the sky radiance and the sun light's transmittance
    tint -- computing them twice is how a renderer ends up with a sun that is
    a different colour from the sky it hangs in.
    """
    sunfade = 1.0 - min(max(1.0 - math.exp(sun_y), 0.0), 1.0)
    beta_r = _TOTAL_RAYLEIGH * np.float32(SKY_RAYLEIGH - (1.0 - sunfade))
    beta_m = _MIE_CONST * np.float32(0.434 * (0.2 * turb) * 1e-17 * SKY_MIE_COEFF)
    return beta_r, beta_m, sunfade


def extinction(beta_r, beta_m, dir_y):
    """Beer-Lambert transmittance along a slant path toward elevation dir_y."""
    y = np.asarray(dir_y, np.float32)
    zen = np.arccos(np.maximum(y, 0.0))
    inv = 1.0 / (np.cos(zen) + 0.15 * np.power(93.885 - zen * (180.0 / math.pi), -1.253))
    return np.exp(-(beta_r * (8400.0 * inv)[..., None]
                    + beta_m * (1250.0 * inv)[..., None])).astype(np.float32)


def _sun_intensity(zenith_cos):
    cutoff = 1.6110731556870734           # pi / 1.95
    zc = min(max(zenith_cos, -1.0), 1.0)
    return 1000.0 * max(0.0, 1.0 - math.exp(-((cutoff - math.acos(zc)) / 1.5)))


# Rainbow colours from the inner (violet) to the outer (red) edge of the bow.
_BOW_F = np.array([0.0, 0.25, 0.50, 0.75, 1.0], np.float32)
_BOW_RGB = np.array([[0.42, 0.10, 0.62], [0.10, 0.35, 0.95], [0.20, 0.85, 0.25],
                     [1.00, 0.92, 0.15], [1.00, 0.22, 0.08]], np.float32)


def _bow_colour(f):
    return np.stack([np.interp(f, _BOW_F, _BOW_RGB[:, c]) for c in range(3)],
                    -1).astype(np.float32)


def sky_radiance(dirs, cs, wx, y_rows=None):
    """Linear sky radiance toward `dirs` ((..., 3) unit vectors).

    Preetham daylight + the demo's own directional horizon glow at low sun +
    a night extension (deep blue gradient, moon disc and halo) + the storm's
    darkening and greying + the rainbow. The lower hemisphere fades out of the
    SAME horizon value the upper one ends on, so there is no dark band at the
    sea line and grazing reflections never go black.

    Written for the bake, which is half a million directions at a time. Three
    things keep it inside a frame: everything that depends only on ELEVATION is
    passed in as `y_rows` and stays a COLUMN (extinction, the horizon band, the
    below-horizon fade); the two features that occupy a handful of texels (the
    sun disc, the rainbow arc) are found with a threshold on the cosine and
    evaluated on that MASK rather than over the frame; and `x ** k` is spelled
    exp(k * log(x)), because np.power costs twice what that does at this size.
    Written naively the same model took 90-200 ms a bake.
    """
    d = np.asarray(dirs, np.float32)
    y = d[..., 1]
    yb = y if y_rows is None else y_rows       # elevation-only, still a column
    sun = cs.sun_dir.astype(np.float32)
    sun_y = float(sun[1])
    sun_e = np.float32(_sun_intensity(sun_y))
    beta_r, beta_m, sunfade = scatter_coeffs(sun_y, turbidity(wx))
    fex = extinction(beta_r, beta_m, yb)

    cos_t = np.clip(d @ sun, -1.0, 1.0)
    ct = cos_t * np.float32(0.5) + np.float32(0.5)
    ct *= ct
    ct *= np.float32(0.05968310365946075)
    r_phase = ct + np.float32(0.05968310365946075)
    g2 = SKY_MIE_G * SKY_MIE_G
    den = np.float32(1.0 + g2) - np.float32(2.0 * SKY_MIE_G) * cos_t
    m_phase = np.float32(0.07957747154594767 * (1.0 - g2)) / (den * np.sqrt(den))

    # beta / (betaR + betaM) folded into constants: the per-pixel divide goes.
    tot = beta_r + beta_m
    q = (beta_r / tot) * r_phase[..., None] + (beta_m / tot) * m_phase[..., None]
    q *= sun_e
    b = q * fex
    a = q - b
    np.maximum(a, 0.0, out=a)
    np.maximum(b, 0.0, out=b)
    mix_t = np.float32(min(max((1.0 - sun_y) ** 5, 0.0), 1.0))
    a *= np.sqrt(a)                                     # x ** 1.5
    np.sqrt(b, out=b)
    b *= mix_t
    b += np.float32(1.0) - mix_t
    a *= b                                              # a is `lin` from here

    # The solar disc lives IN the bake (env-sun policy Auto + an explicit
    # DirectionalLight = the env supplies sky and ambience, the disc is clamped
    # out of the glossy mips). Under a storm deck there is no disc to see.
    murk = smoothstep(0.30, 0.92, wx["coverage"]) * min(wx["cloud_density"], 1.0)
    disc_cos = 0.9999566769464484
    a += fex * np.float32(0.1)
    hit = cos_t > disc_cos
    if murk < 0.99 and hit.any():
        w = np.clip((cos_t[hit] - disc_cos) / 2.0e-5, 0.0, 1.0)
        w = (w * w * (3.0 - 2.0 * w) * (sun_e * 19000.0 * (1.0 - 0.98 * murk)))
        a[hit] += np.broadcast_to(fex, a.shape)[hit] * w[..., None].astype(np.float32)

    a *= np.float32(0.04)
    a += np.array([0.0, 0.0003, 0.00075], np.float32)
    np.maximum(a, 1e-9, out=a)
    np.log(a, out=a)
    a *= np.float32(1.0 / (1.2 + 1.2 * sunfade))
    col = np.exp(a, out=a)
    # The Preetham pow-curve lifts the deep-night residual to a grey haze; fade
    # the daylight model out once the sun is well down, so night belongs to the
    # (dark) night model below.
    col *= np.float32(0.05 + 0.95 * smoothstep(-0.25, -0.02, sun_y))

    # ---- the demo's dawn haze, kept ----------------------------------------
    # A horizon glow concentrated around the sun's BEARING rather than smeared
    # evenly round the compass: an all-round band reads as overcast noon, a
    # directional one reads as low sun, and that is this demo's hero look.
    glow_amp = ((1.0 - smoothstep(0.06, 0.34, sun_y)) * cs.daylight
                * (1.0 - 0.85 * murk))
    if glow_amp > 1e-3:
        sun_hz = sun[[0, 2]] / max(float(np.linalg.norm(sun[[0, 2]])), 1e-6)
        hx, hz = d[..., 0], d[..., 2]
        toward = ((hx * sun_hz[0] + hz * sun_hz[1])
                  / np.sqrt(np.maximum(hx * hx + hz * hz, 1e-12)))
        np.clip(toward, -1.0, 1.0, out=toward)
        toward *= np.float32(0.5)
        toward += np.float32(0.5)
        toward *= toward * toward
        toward *= np.float32(0.52 * glow_amp)
        toward += np.float32(0.09 * glow_amp)
        toward = toward * np.exp((yb * yb) * np.float32(-1.0 / (2.0 * 0.0075)))
        col += toward[..., None] * np.array([1.0, 0.62, 0.34], np.float32)

    # ---- night --------------------------------------------------------------
    night = 1.0 - cs.daylight
    if night > 1e-3:
        horizon = np.exp(np.maximum(yb, 0.0) * np.float32(-4.5)) * (yb >= 0.0)
        base = (np.array([0.0035, 0.0050, 0.0100], np.float32)
                + horizon[..., None] * np.array([0.016, 0.022, 0.042], np.float32))
        col += base * np.float32(night)
        # The moon is a 1 deg disc in a 2 pi sky: find it with one comparison
        # and paint it on the few hundred texels it actually covers.
        cos_m = d @ cs.moon_dir.astype(np.float32)
        near = cos_m > 0.92                             # ~23 deg: disc plus halo
        if cs.moon_up > 0.0 and near.any():
            ang_m = np.arccos(np.clip(cos_m[near], -1.0, 1.0))
            disc = (ang_m < 0.018).astype(np.float32) * cs.moon_up
            halo = 0.28 * np.exp(-ang_m * ang_m * 110.0) * cs.moon_up * (1.0 - disc)
            col[near] += ((disc[..., None] * np.array([16.0, 18.0, 23.0], np.float32)
                           + halo[..., None] * np.array([0.65, 0.75, 1.0], np.float32))
                          * np.float32(night))

    # ---- the storm deck -----------------------------------------------------
    # The volumetric cloud deck is a RASTER effect; the IBL knows nothing about
    # it. Without darkening and greying the bake ITSELF, a boat under a black
    # ceiling is still lit like a sunny day -- so the sky the storm shows and
    # the sky the storm lights by are made the same thing here.
    if murk > 1e-3:
        lum = (col @ _LUMA)[..., None] * np.float32(0.80 * murk)
        col *= np.float32(1.0 - 0.80 * murk)
        col += lum * np.array([0.90, 0.94, 1.00], np.float32)
        col *= np.float32(1.0 - 0.66 * murk)

    # ---- the rainbow --------------------------------------------------------
    # Geometry, not decoration: the bow is an arc of fixed angular radius about
    # the ANTISOLAR point, which is why it is only there with the sun behind you
    # and low, and why it sinks below the sea as the sun climbs. Primary
    # 40.5-42.5 deg (violet in, red out), secondary 51-53 deg and reversed.
    bow = wx["rainbow"] * (1.0 - smoothstep(0.26, 0.40, sun_y)) * cs.daylight
    if bow > 0.01:
        anti = -cos_t                    # cosine of the angle from the antisolar point
        yall = np.broadcast_to(y, anti.shape)
        for lo, hi, amp, flip in ((40.5, 42.5, 0.34 * bow, False),
                                  (51.0, 53.0, 0.115 * bow, True)):
            m = ((anti < math.cos(math.radians(lo)))
                 & (anti > math.cos(math.radians(hi))))
            if not m.any():
                continue
            f = (np.degrees(np.arccos(np.clip(anti[m], -1.0, 1.0))) - lo) / (hi - lo)
            np.clip(f, 0.0, 1.0, out=f)
            env = np.maximum(np.sin(np.pi * f), 0.0)
            env *= np.clip(yall[m] * 9.0, 0.0, 1.0) * amp   # stops at the sea line
            col[m] += _bow_colour(1.0 - f if flip else f) * env[..., None].astype(np.float32)

    # Below the horizon: fade toward a dark sea tone, out of exactly the horizon
    # value above it (fade == 1 at y == 0), so the two hemispheres meet.
    low = yb < 0.0
    if low.any():
        col *= np.where(low, np.exp(np.minimum(yb, 0.0) * np.float32(2.6)), 1.0)[..., None]
        col += low[..., None] * np.array([0.002, 0.0025, 0.003], np.float32)

    col *= np.float32(0.62)      # sun:sky contrast; auto exposure re-normalises
    return np.minimum(col, 300.0)


SKY_W, SKY_H = 1024, 512
_sv = (np.arange(SKY_H, dtype=np.float32) + 0.5) / SKY_H
_su = (np.arange(SKY_W, dtype=np.float32) + 0.5) / SKY_W
_selev = (_sv - 0.5) * math.pi
_saz = (_su - 0.5) * 2.0 * math.pi
# v = 0.5 + asin(dir.y)/pi and u = 0.5 + atan2(z, x)/tau is what the shader
# samples with (deferred_shade_10_lighting_utils.glsl), so row 0 is the nadir.
_SKY_DIRS = np.empty((SKY_H, SKY_W, 3), np.float32)
_SKY_DIRS[..., 0] = np.cos(_selev)[:, None] * np.cos(_saz)[None, :]
_SKY_DIRS[..., 1] = np.sin(_selev)[:, None]
_SKY_DIRS[..., 2] = np.cos(_selev)[:, None] * np.sin(_saz)[None, :]
_SKY_YROWS = np.sin(_selev)[:, None]
_HZ_AZ = (np.arange(16, dtype=np.float32) + 0.5) / 16.0 * 2.0 * math.pi
_HZ_DIRS = np.stack([np.cos(0.026) * np.cos(_HZ_AZ),
                     np.full(16, math.sin(0.026), np.float32),
                     np.cos(0.026) * np.sin(_HZ_AZ)], -1).astype(np.float32)


def bake_sky(cs, wx):
    """The whole sky as a float32 (H, W, 4) equirect, ready for float_texture."""
    out = np.empty((SKY_H, SKY_W, 4), np.float32)
    out[..., :3] = sky_radiance(_SKY_DIRS, cs, wx, y_rows=_SKY_YROWS)
    out[..., 3] = 1.0
    return out


def horizon_color(cs, wx):
    """Average radiance round the horizon -- the fog colour that always fits."""
    c = sky_radiance(_HZ_DIRS, cs, wx).mean(axis=0)
    return tp.Color(float(c[0]), float(c[1]), float(c[2]))


# --------------------------------------------------------------------------- #
#  Weather. Presets are TARGETS; the live state chases them with per-field time
#  constants measured in seconds, because a squall that arrives in one frame
#  reads as a cut, not as weather.
# --------------------------------------------------------------------------- #
WEATHER_KEYS = ("coverage", "cloud_density", "deck_bottom", "deck_top", "wind",
                "wave_scale", "choppy", "fog", "fog_h", "rain", "rainbow")

PRESETS = {
    # dawn: thick mist, a glassy sea, the light only just arriving
    "mist": dict(coverage=0.30, cloud_density=0.8, deck_bottom=620.0, deck_top=1500.0,
                 wind=6.0, wave_scale=0.72, choppy=0.48, fog=0.0062, fog_h=52.0,
                 rain=0.0, rainbow=0.0),
    "clear": dict(coverage=0.22, cloud_density=0.8, deck_bottom=700.0, deck_top=1600.0,
                  wind=9.5, wave_scale=1.00, choppy=0.58, fog=0.0011, fog_h=140.0,
                  rain=0.0, rainbow=0.0),
    "overcast": dict(coverage=0.74, cloud_density=1.15, deck_bottom=420.0, deck_top=1250.0,
                     wind=12.0, wave_scale=1.20, choppy=0.68, fog=0.0040, fog_h=130.0,
                     rain=0.0, rainbow=0.0),
    "storm": dict(coverage=0.97, cloud_density=1.70, deck_bottom=240.0, deck_top=1500.0,
                  wind=17.0, wave_scale=1.65, choppy=0.86, fog=0.0078, fog_h=150.0,
                  rain=1.0, rainbow=0.0),
    # the squall walking away: broken cloud, a low sun, and a bow behind you
    "clearing": dict(coverage=0.38, cloud_density=0.95, deck_bottom=520.0, deck_top=1350.0,
                     wind=8.5, wave_scale=1.15, choppy=0.62, fog=0.0032, fog_h=95.0,
                     rain=0.0, rainbow=1.0),
    # thin sea mist, because a lighthouse beam is only visible if there is
    # something in the air for it to be visible IN
    "night": dict(coverage=0.18, cloud_density=0.8, deck_bottom=700.0, deck_top=1600.0,
                  wind=7.0, wave_scale=0.90, choppy=0.55, fog=0.0062, fog_h=60.0,
                  rain=0.0, rainbow=0.0),
}
PRESETS["night-clear"] = PRESETS["night"]
PRESETS["golden"] = PRESETS["clearing"]

# Seconds to close ~63% of the gap. Cloud and sea are slow, rain is quick, and
# a wet deck dries far more slowly than it wets.
RAMP_TAU = dict(coverage=5.0, cloud_density=5.0, deck_bottom=6.0, deck_top=6.0,
                wind=7.0, wave_scale=9.0, choppy=9.0, fog=5.0, fog_h=6.0,
                rain=2.2, rainbow=3.5)

# The five acts: key -> (time of day, weather preset).
ACTS = ((4.833, "mist"), (8.5, "clear"), (15.0, "storm"),
        (19.667, "clearing"), (23.5, "night"))

if WEATHER0 not in PRESETS:
    print(f"unknown --weather {WEATHER0!r}; choose from {sorted(PRESETS)}")
    sys.exit(2)

# `weather` is the live state; `wtarget` is what it is chasing. Everything the
# sky, the sea, the fog, the clouds and the rain read comes out of `weather`.
weather = dict(PRESETS[WEATHER0])
weather["wetness"] = 1.0 if weather["rain"] > 0.5 else 0.0
wtarget = dict(PRESETS[WEATHER0])
if WIND0 > 0.0:
    weather["wind"] = wtarget["wind"] = WIND0
weather_name = WEATHER0
time_of_day = TOD0
# Hours of sky per second of wall clock. K toggles it in the window; 0.35
# runs the whole day in about 68 s, which is the cadence to watch the bake
# throttle at.
day_speed = 0.35 if "--timelapse" in sys.argv else 0.0
_rain_was_on = weather["rain"] > 0.3
_bow_timer = 0.0           # seconds left in the post-rain rainbow window


def set_weather(name):
    """Aim the weather at a preset. The ramps do the rest."""
    global weather_name
    if name not in PRESETS:
        return
    weather_name = name
    wtarget.update(PRESETS[name])


# --------------------------------------------------------------------------- #
#  Hull, deck, cabin and foils: structured grid lofts authored in numpy and
#  handed to BufferGeometry. Procedural rather than a loaded model because the
#  rig has to know exactly where the mast steps and where the sails attach.
# --------------------------------------------------------------------------- #
def _grid_indices(ns, nc):
    i = np.arange(ns - 1)[:, None] * nc + np.arange(nc - 1)[None, :]
    a, b, c, d = i, i + 1, i + nc, i + nc + 1
    return np.stack([np.stack([a, c, b], -1),
                     np.stack([b, c, d], -1)], 1).reshape(-1).astype(np.uint32)


def _orient(verts, idx, inside):
    """Flip the winding if the faces mostly point back at `inside`.

    Cheaper than hand-deriving the handedness of every loft, and the lofts are
    closed enough that "away from an interior point" is unambiguous.
    """
    tri = idx.reshape(-1, 3)
    p0, p1, p2 = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    out = (p0 + p1 + p2) / 3.0 - np.asarray(inside, np.float64)
    if float((n * out).sum()) < 0.0:
        tri = tri[:, ::-1]
    return np.ascontiguousarray(tri.reshape(-1)).astype(np.uint32)


def _mesh(verts, idx, mat, inside=(0.0, 0.0, 0.0)):
    g = tp.BufferGeometry()
    g.set_attribute("position", np.ascontiguousarray(verts, np.float32))
    g.set_index(_orient(verts, idx, inside))
    g.compute_vertex_normals()
    m = tp.Mesh(g, mat)
    m.cast_shadow = True
    m.receive_shadow = True
    return m


def _station(t):
    """Half-beam, canoe depth, sheer height and section fullness at station t.

    t = 0 at the bow, 1 at the transom: a fine entry, maximum beam just aft of
    midships, and a broad but not scow-like transom.
    """
    t = np.clip(t, 0.0, 1.0)
    # Beam: nothing at the stem, full at t = 0.62, and still 72% of it at the
    # transom. A curve that returns to zero aft would make her a double-ender.
    rise = np.clip(t / 0.62, 0.0, 1.0) ** 0.62
    fall = 1.0 - 0.28 * np.clip((t - 0.62) / 0.38, 0.0, 1.0) ** 1.6
    w = np.maximum(BEAM * 0.5 * rise * fall, 1e-3)
    # Rocker: deepest just aft of midships, rising to the stem, and the transom
    # sits a little above its deepest point rather than pinching to nothing.
    d = np.maximum(DRAFT * (np.clip(np.sin(np.pi * t ** 0.80), 0, 1) ** 0.5
                            * (1.0 - 0.55 * np.clip((t - 0.70) / 0.30, 0, 1) ** 2)
                            + 0.30 * np.clip((t - 0.55) / 0.45, 0, 1) ** 1.2), 0.02)
    # Sheer: a spring at the bow, a dip amidships, a small lift aft. A dead
    # straight sheer over this much freeboard is what makes a hull read as a barge.
    h = FREEBOARD * (1.0 + 0.62 * (1.0 - t) ** 2.0 - 0.17 * np.sin(np.pi * t))
    p = 1.35 + 1.15 * t ** 0.8                     # V forward, rounded bilge aft
    return w, d, h, p


def _half_section(t, n=NP):
    """Keel to deck edge in the x-y plane: superellipse below, flare above."""
    w, d, h, p = _station(t)
    u = np.linspace(0.0, 1.0, n)
    split = 0.66                                   # fraction of the girth underwater
    xs = np.empty(n)
    ys = np.empty(n)
    lo = u <= split
    a = (u[lo] / split) * (np.pi / 2)
    xs[lo] = w * np.sin(a) ** (2.0 / p)
    ys[lo] = -d * np.cos(a) ** (2.0 / p)
    hi = ~lo
    b = (u[hi] - split) / (1.0 - split)
    xs[hi] = w * (1.0 + 0.11 * b ** 1.6)           # topside flare
    ys[hi] = h * b ** 0.92
    return xs, ys


def hull_grid():
    """The hull surface as an (NS, 2*NP-1, 3) grid: stbd deck edge -> keel -> port."""
    t = np.linspace(0.0, 1.0, NS)
    X = np.empty((NS, NP))
    Y = np.empty((NS, NP))
    for i, ti in enumerate(t):
        X[i], Y[i] = _half_section(ti)
    # Starboard half reversed, then port. The keel column is shared, so the
    # surface stays continuous across the centreline.
    Xf = np.concatenate([X[:, ::-1], -X[:, 1:]], axis=1)
    Yf = np.concatenate([Y[:, ::-1], Y[:, 1:]], axis=1)
    nc = Xf.shape[1]
    Z = np.repeat(((0.5 - t) * LOA)[:, None], nc, axis=1)
    return np.stack([Xf, Yf, Z], -1)


def band(grid, c0, c1):
    """One paint band: columns [c0, c1] of the hull grid as its own mesh.

    Splitting the hull by girth rather than colouring it uniformly is what
    stops a white boat reading as a bathtub -- antifouling, a boot stripe and
    topsides are three different paints on a real hull, and the eye reads the
    waterline off the stripe.
    """
    sub = np.ascontiguousarray(grid[:, c0:c1 + 1])
    return sub.reshape(-1, 3), _grid_indices(sub.shape[0], sub.shape[1])


def build_deck():
    t = np.linspace(0.0, 1.0, NS)
    nc = 2 * NP - 1
    fx = np.linspace(1.0, -1.0, nc)
    V = np.empty((NS, nc, 3))
    for i, ti in enumerate(t):
        xs, ys = _half_section(ti)
        # Cockpit well, aft of the cabin: a smooth dish sunk into the deck.
        along = np.clip((ti - 0.72) / 0.25, 0.0, 1.0)
        well = np.sin(np.pi * along) ** 1.2 if 0.0 < along < 1.0 else 0.0
        # Steep sides, flat sole: a shallow dish reads as a dent in the deck.
        sunk = 0.52 * well * np.clip(1.0 - (np.abs(fx) / 0.72) ** 5, 0.0, 1.0)
        V[i, :, 0] = fx * xs[-1]
        V[i, :, 1] = ys[-1] + 0.055 * xs[-1] * (1.0 - fx ** 2) - sunk   # crown, less well
        V[i, :, 2] = (0.5 - ti) * LOA
    return V.reshape(-1, 3), _grid_indices(NS, nc)


def build_transom():
    """Close the stern.

    The boundary is the last hull station PLUS the deck's aft edge coming back
    the other way -- a closed loop. Fanning the hull curve alone leaves the gap
    between the two deck edges unfilled, which is a V-shaped hole you can see
    straight through into the inside of the boat.
    """
    xs, ys = _half_section(1.0)
    sec_x = np.concatenate([xs[::-1], -xs[1:]])          # stbd deck edge -> keel -> port
    sec_y = np.concatenate([ys[::-1], ys[1:]])
    fx = np.linspace(-1.0, 1.0, 2 * NP - 1)              # port -> stbd, matching the deck
    deck_x = fx * xs[-1]
    deck_y = ys[-1] + 0.055 * xs[-1] * (1.0 - fx ** 2)
    # Drop the shared end points so the loop has no duplicated vertices.
    x = np.concatenate([sec_x, deck_x[1:-1]])
    y = np.concatenate([sec_y, deck_y[1:-1]])
    z = np.full(len(x), -0.5 * LOA)
    ring = np.stack([x, y, z], -1)
    hub = ring.mean(axis=0)
    verts = np.concatenate([hub[None, :], ring], axis=0)
    n = len(ring)
    i0 = np.arange(1, n + 1, dtype=np.uint32)
    i1 = np.roll(i0, -1)                                 # wrap: the loop is closed
    idx = np.stack([np.zeros(n, np.uint32), i0, i1], -1).reshape(-1)
    return verts, idx


def fan_cap(ring):
    """Close an open section curve with a straight chord and fan it flat.

    Used for the coachroof's forward and aft bulkheads: the loft stops dead at
    both ends, and without a cap the trunk is a tunnel.
    """
    ring = np.asarray(ring, np.float64)
    hub = ring.mean(axis=0)
    verts = np.concatenate([hub[None, :], ring], axis=0)
    n = len(ring)
    i0 = np.arange(1, n + 1, dtype=np.uint32)
    idx = np.stack([np.zeros(n, np.uint32), i0, np.roll(i0, -1)], -1).reshape(-1)
    return verts, idx


def cabin_grid():
    """Coachroof: near-vertical sides under a rounded top, narrowing forward.

    A superellipse rather than a half-circle -- a domed section reads as a
    bubble on the deck, while flat sides and a crowned top read as a cabin.
    """
    t0, t1, ns, nc = 0.32, 0.72, 26, 30
    ts = np.linspace(t0, t1, ns)
    V = np.empty((ns, nc, 3))
    a = np.linspace(0.0, np.pi, nc)
    ca = np.cos(a)
    roof = (1.0 - np.abs(ca) ** 3.4) ** (1.0 / 3.4)
    for i, ti in enumerate(ts):
        xs, ys = _half_section(ti)
        f = (ti - t0) / (t1 - t0)
        w = 0.56 * xs[-1] * (0.66 + 0.34 * np.sin(np.pi * f) ** 0.45)
        h = 0.50 + 0.22 * np.sin(np.pi * np.clip(f, 0, 1)) ** 0.5
        V[i, :, 0] = w * ca
        V[i, :, 1] = ys[-1] + h * roof
        V[i, :, 2] = (0.5 - ti) * LOA
    return V


def build_foil(chord_root, chord_tip, span, thick, sweep, ns=14, nc=20):
    """Tapered NACA-00xx foil: chord along z, span down -y, thickness along x."""
    V = np.empty((ns, nc, 3))
    s = np.linspace(0.0, 2.0 * np.pi, nc)
    xc = 0.5 * (1.0 + np.cos(s))
    for i in range(ns):
        f = i / (ns - 1)
        c = chord_root + (chord_tip - chord_root) * f
        tt = 5.0 * thick * c * (0.2969 * np.sqrt(np.clip(xc, 0, 1)) - 0.1260 * xc
                                - 0.3516 * xc ** 2 + 0.2843 * xc ** 3 - 0.1015 * xc ** 4)
        V[i, :, 0] = np.sign(np.sin(s)) * tt
        V[i, :, 1] = -span * f
        V[i, :, 2] = -sweep * f + (xc - 0.5) * c
    return V.reshape(-1, 3), _grid_indices(ns, nc)


def _flat_soup(grid, inside):
    """A grid as a NON-INDEXED soup with per-face normals -- hard facets.

    Rock has to be faceted or it reads as a beanbag, and the usual lever for
    that (`flatShading`) is a no-op on the Vulkan backend -- it has no
    occurrences there at all. The portable way to get a crease is to stop
    sharing vertices: every triangle carries its own copy, so every triangle
    gets its own normal and the silhouette breaks into planes.
    """
    ns, nc = grid.shape[0], grid.shape[1]
    a = grid[:-1, :-1]
    b = grid[:-1, 1:]
    c = grid[1:, :-1]
    d = grid[1:, 1:]
    tris = np.concatenate([np.stack([a, c, b], axis=2),
                           np.stack([b, c, d], axis=2)], axis=2)
    tris = tris.reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    # Orient outward from an interior point, then flip the wound-in triangles
    # rather than the whole sheet: a crag is not a height field and parts of it
    # genuinely face different ways.
    out = tris.mean(axis=1) - np.asarray(inside, np.float64)
    flip = (n * out).sum(-1) < 0.0
    tris[flip] = tris[flip][:, ::-1]
    n[flip] = -n[flip]
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    verts = tris.reshape(-1, 3)
    nrm = np.repeat(n, 3, axis=0)
    return np.ascontiguousarray(verts, np.float32), np.ascontiguousarray(nrm, np.float32)


# (frequency in theta, frequency in t, amplitude) -- six octaves of crag.
_ROCK_OCTAVES = ((3.0, 2.3, 0.44), (7.0, -5.1, 0.26), (17.0, 11.0, 0.13),
                 (37.0, -23.0, 0.07), (79.0, 47.0, 0.035), (151.0, -89.0, 0.02))


def _ridged(theta, t, ph):
    """Ridged multifractal. Sharp crests, and detail only where there is rock.

    `1 - |sin|` gives a crease where a plain sine would give a dome. Weighting
    each octave by the PREVIOUS one is what turns that from a bed of nails into
    a crag: high-frequency detail is suppressed in the gullies and piles up on
    the ridges, which is how rock actually erodes.
    """
    total = np.zeros(np.broadcast_shapes(theta.shape, t.shape))
    weight = np.ones_like(total)
    norm = 0.0
    for k, (ft, fv, amp) in enumerate(_ROCK_OCTAVES):
        n = 1.0 - np.abs(np.sin(ft * theta + fv * t + ph[k % len(ph)]))
        total = total + amp * n * weight
        weight = np.clip(n * 1.10, 0.0, 1.0)
        norm += amp
    return total / norm


def build_islet(seed, radius, height, pad_r, na=220, nr=52):
    """A crag rising straight out of the water, with a pad cut on top.

    Radial grid: the RIM is displaced as well as the height, so the waterline
    is ragged instead of a circle, and the summit is flattened into a platform
    because a lighthouse needs somewhere to stand.
    """
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0.0, 6.28, 8)
    theta = np.linspace(0.0, 2.0 * np.pi, na)[None, :]
    t = np.linspace(0.0, 1.0, nr)[:, None]
    # Ragged outline: the rock is nowhere near round.
    rim = radius * (1.0 + 0.30 * np.sin(3.0 * theta + ph[5])
                    + 0.18 * np.sin(7.0 * theta + ph[6])
                    + 0.11 * np.sin(13.0 * theta + ph[7])
                    + 0.06 * np.sin(29.0 * theta + ph[5] * 2.1))
    relief = (_ridged(theta, t, ph) - 0.5) * 2.0
    # Dome, then a flat pad punched through the middle of it.
    dome = np.clip(1.0 - t ** 3.4, 0.0, 1.0) ** 0.50   # steep sides, solid body
    pad = np.clip((pad_r - t) / max(pad_r * 0.55, 1e-6), 0.0, 1.0)
    pad = pad * pad * (3.0 - 2.0 * pad)
    y = height * dome * (1.0 + 0.24 * relief * np.clip(t * 3.0, 0.0, 1.0))
    step = height * 0.155
    y = y + 0.45 * (np.round(y / step) * step - y)     # settle onto bedding planes
    y = y * (1.0 - pad) + height * 1.02 * pad
    # Radial displacement too, so the crag has overhangs in silhouette.
    r = t * rim * (1.0 + 0.06 * relief)
    grid = np.stack([r * np.cos(theta), y - 5.0, r * np.sin(theta)], -1)
    return grid


def _val_noise(rng, h, w, cy, cx):
    """Bilinear value noise on a (cy, cx) lattice, wrapping in u."""
    g = rng.random((cy + 1, cx + 1))
    g[:, -1] = g[:, 0]                                  # seamless around the tower
    fy = np.linspace(0.0, cy, h, endpoint=False)
    fx = np.linspace(0.0, cx, w, endpoint=False)
    y0 = np.floor(fy).astype(int)
    x0 = np.floor(fx).astype(int)
    ty = fy - y0
    tx = fx - x0
    ty = (ty * ty * (3.0 - 2.0 * ty))[:, None]
    tx = (tx * tx * (3.0 - 2.0 * tx))[None, :]
    g00 = g[y0][:, x0]
    g01 = g[y0][:, x0 + 1]
    g10 = g[y0 + 1][:, x0]
    g11 = g[y0 + 1][:, x0 + 1]
    return (g00 * (1 - tx) + g01 * tx) * (1 - ty) + (g10 * (1 - tx) + g11 * tx) * ty


def _fbm(rng, h, w, cy, cx, octaves=5):
    out = np.zeros((h, w))
    amp, norm = 1.0, 0.0
    for k in range(octaves):
        out += amp * _val_noise(rng, h, w, cy * 2 ** k, cx * 2 ** k)
        norm += amp
        amp *= 0.5
    return out / norm


def lighthouse_maps(seed, W=1024, H=1024):
    """Albedo + roughness for a light nobody has painted in forty years.

    Everything here is a WEATHERING PROCESS rather than a texture effect, which
    is what keeps it reading as age instead of as dirt: water runs downward from
    ledges, iron bleeds rust, lime render fails in patches and drops off, algae
    climbs as far as the spray reaches, and the sun bleaches whatever is left.
    """
    rng = np.random.default_rng(seed)
    v = np.linspace(0.0, 1.0, H)[:, None]                 # 0 = base, 1 = lantern
    u = np.linspace(0.0, 1.0, W)[None, :]

    grain = _fbm(rng, H, W, 3, 3, 6)
    # Lime render: warm off-white, sun-bleached toward the top.
    col = np.empty((H, W, 3))
    render = np.array([0.44, 0.41, 0.36]) + 0.22 * v[..., None] * np.array([1.0, 0.99, 0.93])
    col[:] = render * (0.78 + 0.44 * grain[..., None])

    # Render loss: thresholded blotches, showing the rubble stone beneath. The
    # edges are noisy because failing render spalls, it does not get cut out.
    blotch = _fbm(rng, H, W, 11, 9, 5) + 0.12 * grain
    loss = np.clip((blotch - (0.60 - 0.16 * (1.0 - v))) * 7.0, 0.0, 1.0)
    stone = np.array([0.40, 0.36, 0.31]) * (0.75 + 0.5 * _fbm(rng, H, W, 24, 20, 4)[..., None])
    # Masonry courses only show where the render is gone.
    course = (np.abs(np.sin(v * 46.0 * np.pi)) ** 12)
    stone = stone * (1.0 - 0.45 * course[..., None])
    col = col * (1.0 - loss[..., None]) + stone * loss[..., None]

    # Rain streaks: stretched hard in v so they run, and each one only starts
    # below the ledge that sheds it.
    streak_n = (0.55 * _val_noise(rng, H, W, 2, 130)
                + 0.30 * _val_noise(rng, H, W, 3, 260)
                + 0.15 * _val_noise(rng, H, W, 5, 520))
    for start, strength, tint in ((0.955, 0.62, (0.45, 0.41, 0.36)),
                                  (0.72, 0.34, (0.48, 0.45, 0.40)),
                                  (0.30, 0.26, (0.44, 0.43, 0.39))):
        run = np.clip((start - v) / 0.42, 0.0, 1.0) ** 0.7 * (v < start)
        mask = np.clip((streak_n - 0.55) * 5.5, 0.0, 1.0) * run * strength
        col = col * (1.0 - mask[..., None]) + np.array(tint) * mask[..., None]

    # Rust: narrower, fewer, and only under the ironwork at the gallery.
    rust_n = 0.6 * _val_noise(rng, H, W, 2, 300) + 0.4 * _val_noise(rng, H, W, 4, 620)
    run = np.clip((0.945 - v) / 0.30, 0.0, 1.0) ** 0.8 * (v < 0.945)
    rust = np.clip((rust_n - 0.66) * 9.0, 0.0, 1.0) * run * 0.9
    col = col * (1.0 - rust[..., None]) + np.array([0.34, 0.16, 0.07]) * rust[..., None]

    # Algae and salt: the splash zone, as high as the sea throws it.
    wet = np.clip((0.13 - v) / 0.13, 0.0, 1.0) ** 1.4 * (0.55 + 0.45 * grain)
    col = col * (1.0 - 0.72 * wet[..., None]) + np.array([0.13, 0.16, 0.11]) * (0.72 * wet[..., None])
    salt = np.clip((_fbm(rng, H, W, 9, 8, 4) - 0.58) * 4.0, 0.0, 1.0) * np.clip((0.34 - v) / 0.34, 0.0, 1.0)
    col = col + salt[..., None] * np.array([0.16, 0.16, 0.15])

    # Roughness: bare stone and salt bloom are matte, algae is damp and slick.
    rough = 0.74 + 0.14 * grain
    rough = rough + 0.16 * loss + 0.10 * salt - 0.30 * wet + 0.08 * rust
    rough = np.clip(rough, 0.06, 1.0)

    alb = np.clip(col, 0.0, 1.0)
    return ((alb ** (1.0 / 2.2)) * 255.0).astype(np.uint8), \
        (np.repeat(rough[..., None], 3, axis=2) * 255.0).astype(np.uint8)


def build_tower(profile, na=48):
    """A surface of revolution from a (y, radius) profile."""
    theta = np.linspace(0.0, 2.0 * np.pi, na)[None, :]
    ys = np.asarray([p[0] for p in profile], np.float64)[:, None]
    rs = np.asarray([p[1] for p in profile], np.float64)[:, None]
    # Nothing built by hand and left in the weather for a century is still a
    # perfect solid of revolution: lean the courses a little and let the radius
    # wander, so the silhouette is not a lathe turning.
    wob = 1.0 + 0.012 * np.sin(3.0 * theta + 1.7 * ys) + 0.008 * np.sin(7.0 * theta - 2.3 * ys)
    lean = 0.055 * (ys / max(float(ys.max()), 1e-6)) ** 2.0
    grid = np.stack([rs * wob * np.cos(theta) + lean,
                     np.broadcast_to(ys, (len(profile), na)),
                     rs * wob * np.sin(theta)], -1)
    # u wraps once round; v is the true height, so the streaks do not stretch
    # differently on the drum and on the shaft.
    uu = np.broadcast_to(np.linspace(0.0, 1.0, na)[None, :], (len(profile), na))
    vv = np.broadcast_to(ys / max(float(ys.max()), 1e-6), (len(profile), na))
    return grid, np.stack([uu, vv], -1)


def strut(p0, p1, radius, mat):
    """A thin cylinder between two points -- stays, shrouds, pushpit rails."""
    p0 = np.asarray(p0, np.float64)
    p1 = np.asarray(p1, np.float64)
    d = p1 - p0
    length = float(np.linalg.norm(d))
    m = tp.Mesh(tp.CylinderGeometry(radius, radius, length, 8, 1), mat)
    mid = 0.5 * (p0 + p1)
    m.position.set(float(mid[0]), float(mid[1]), float(mid[2]))
    # look_at aims local -Z at the target; the cylinder's axis is +Y, so tip it
    # a quarter turn about X to line the axis up with the aim.
    m.look_at(float(p1[0]), float(p1[1]), float(p1[2]))
    m.rotate_x(math.pi / 2)
    m.cast_shadow = True
    return m


# --------------------------------------------------------------------------- #
#  Sail rest shapes.
#
#  The rest surface is CAMBERED. A sailmaker cuts broadseam into flat panels so
#  the finished sail has belly, and we do the same -- but we only ever read rest
#  LENGTHS out of it, never rest positions. That makes the sail bistable: it
#  fills to whichever side the wind is on and flips over when you tack. Cut it
#  flat instead and the cloth is a taut sheet that cannot billow without
#  stretching, which reads on screen as a plank.
# --------------------------------------------------------------------------- #
DRAFT_FRAC = 0.13            # maximum belly as a fraction of chord


def _camber(fu, fv):
    """Belly profile: draft peaks ~40% aft of the luff, fading out at the head."""
    return np.sin(np.pi * np.clip(fu, 0, 1) ** 0.78) * (1.0 - fv) ** 0.55


def main_rest(nu, nv):
    """Mainsail at zero trim: luff up the mast, foot aft along the boom."""
    fu = (np.arange(nu) / (nu - 1.0))[None, :]
    fv = (1.0 - np.arange(nv) / (nv - 1.0))[:, None]   # PlaneGeometry rows run top-down
    chord = FOOT * (1.0 - fv) + ROACH * np.sin(np.pi * fv) ** 1.3
    flat = np.empty((nv, nu, 3))
    flat[..., 0] = 0.0
    flat[..., 1] = np.broadcast_to(GOOSE_Y + LUFF * fv, (nv, nu))
    flat[..., 2] = MAST_Z - chord * fu
    cam = flat.copy()
    cam[..., 0] = DRAFT_FRAC * chord * _camber(fu, fv)
    return flat.reshape(-1, 3), cam.reshape(-1, 3)


def jib_rest(nu, nv):
    """Jib at zero trim: luff on the forestay, clew aft on the centreline."""
    fu = (np.arange(nu) / (nu - 1.0))[None, :, None]
    fv = (1.0 - np.arange(nv) / (nv - 1.0))[:, None, None]
    tack = np.array([0.0, DECK_Y, TACK_Z])
    head = np.array([0.0, JIB_HEAD, MAST_Z])
    clew = np.array([0.0, DECK_Y + JIB_CLEW_Y, TACK_Z - JIB_FOOT])
    luff = tack + (head - tack) * fv                   # (nv, 1, 3)
    span = fu * (1.0 - fv) * (clew - luff)             # (nv, nu, 3)
    flat = luff + span
    chord = np.linalg.norm(clew - luff, axis=-1) * (1.0 - fv[..., 0])
    cam = flat.copy()
    cam[..., 0] = DRAFT_FRAC * 1.15 * chord * _camber(fu[..., 0], fv[..., 0])
    return flat.reshape(-1, 3), cam.reshape(-1, 3)


def vertex_areas(flat, nu, nv):
    """Rest area carried by each particle: a third of every incident triangle."""
    tri = _grid_indices(nv, nu).reshape(-1, 3).astype(np.int64)
    p0, p1, p2 = flat[tri[:, 0]], flat[tri[:, 1]], flat[tri[:, 2]]
    a = 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)
    area = np.zeros(len(flat))
    for k in range(3):
        np.add.at(area, tri[:, k], a / 3.0)
    return area.astype(np.float32)


# --------------------------------------------------------------------------- #
#  Warp kernels. Anything a slider can move arrives through a preallocated
#  uniform array: scalar kernel arguments are frozen at CUDA-graph capture time.
# --------------------------------------------------------------------------- #
@wp.func
def gidx(iu: int, iv: int, nu: int) -> int:
    return iv * nu + iu


@wp.func
def spring(p: wp.vec3, pos: wp.array(dtype=wp.vec3), rest: wp.array(dtype=wp.vec3),
           i: int, iu: int, iv: int, nu: int, nv: int, stiffness: float) -> wp.vec3:
    # A neighbour outside the grid contributes nothing.
    if iu < 0 or iu >= nu or iv < 0 or iv >= nv:
        return wp.vec3(0.0, 0.0, 0.0)
    j = gidx(iu, iv, nu)
    d = pos[j] - p
    l = wp.length(d)
    if l < 1.0e-9:
        return wp.vec3(0.0, 0.0, 0.0)
    # Rest length read off the cambered cut, so a tapered sail with a curved
    # leech needs no per-edge rest table.
    r = wp.length(rest[j] - rest[i])
    # Half the correction: the neighbour computes the other half for itself.
    return d * (0.5 * stiffness * (l - r) / l)


@wp.kernel
def apply_pins(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
               pin_idx: wp.array(dtype=int), pin_pos: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    i = pin_idx[k]
    p = pin_pos[k]
    # Carry prev along with the pin. These attachments MOVE (the boom swings),
    # and leaving prev behind would hand the cloth a bogus one-substep velocity.
    prev[i] = prev[i] + (p - pos[i])
    pos[i] = p


@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
              pred: wp.array(dtype=wp.vec3), nrm: wp.array(dtype=wp.vec3),
              inv_mass: wp.array(dtype=float),
              vec_uni: wp.array(dtype=wp.vec3), sc_uni: wp.array(dtype=float),
              dt: float):
    i = wp.tid()
    p = pos[i]
    prev_p = prev[i]
    prev[i] = p
    if inv_mass[i] == 0.0:
        pred[i] = p
        return
    wind = vec_uni[0]                  # apparent wind, boat frame
    grav = vec_uni[1]                  # gravity, boat frame -- so heel tilts it
    aero = sc_uni[0]                   # (rho_air / 2) / areal density
    tang = sc_uni[1]                   # skin friction along the surface
    damp = sc_uni[2]
    step = p - prev_p
    v_rel = wind - step / dt
    n = nrm[i]
    vn = wp.dot(n, v_rel)
    # Flat-plate membrane pressure: signed, quadratic, along the local normal.
    # The tangential term is skin friction -- it is what settles a luffing sail
    # instead of letting it ring forever.
    acc = grav + n * (aero * vn * wp.abs(vn)) + (v_rel - n * vn) * tang
    pred[i] = p + step * (1.0 - damp) + acc * dt * dt


@wp.kernel
def solve(p_in: wp.array(dtype=wp.vec3), p_out: wp.array(dtype=wp.vec3),
          rest: wp.array(dtype=wp.vec3), inv_mass: wp.array(dtype=float),
          sc_uni: wp.array(dtype=float), nu: int, nv: int):
    i = wp.tid()
    p = p_in[i]
    if inv_mass[i] == 0.0:
        p_out[i] = p
        return
    iu = i % nu
    iv = i // nu
    st = sc_uni[3]
    sh = sc_uni[4]
    bd = sc_uni[5]
    # Boltrope: the leech, head and foot are free edges, and a real sail carries
    # a reinforcing tape along each of them. Stiffening those rows is both the
    # honest model and what stops the leech shaking itself apart.
    if iu == nu - 1 or iv == 0 or iv == nv - 1:
        st = st * 1.9
        bd = bd * 1.6
    c = wp.vec3(0.0, 0.0, 0.0)
    # structural
    c += spring(p, p_in, rest, i, iu - 1, iv, nu, nv, st)
    c += spring(p, p_in, rest, i, iu + 1, iv, nu, nv, st)
    c += spring(p, p_in, rest, i, iu, iv - 1, nu, nv, st)
    c += spring(p, p_in, rest, i, iu, iv + 1, nu, nv, st)
    # shear
    c += spring(p, p_in, rest, i, iu - 1, iv - 1, nu, nv, sh)
    c += spring(p, p_in, rest, i, iu + 1, iv - 1, nu, nv, sh)
    c += spring(p, p_in, rest, i, iu - 1, iv + 1, nu, nv, sh)
    c += spring(p, p_in, rest, i, iu + 1, iv + 1, nu, nv, sh)
    # bending -- what keeps the leech from folding into knife creases
    c += spring(p, p_in, rest, i, iu - 2, iv, nu, nv, bd)
    c += spring(p, p_in, rest, i, iu + 2, iv, nu, nv, bd)
    c += spring(p, p_in, rest, i, iu, iv - 2, nu, nv, bd)
    c += spring(p, p_in, rest, i, iu, iv + 2, nu, nv, bd)
    # Jacobi projection: everyone moves at once, so under-relax to converge.
    p_out[i] = p + c * 0.3


@wp.kernel
def compute_normals(pos: wp.array(dtype=wp.vec3), nrm: wp.array(dtype=wp.vec3),
                    nu: int, nv: int):
    i = wp.tid()
    iu = i % nu
    iv = i // nu
    xm = pos[gidx(wp.max(iu - 1, 0), iv, nu)]
    xp = pos[gidx(wp.min(iu + 1, nu - 1), iv, nu)]
    ym = pos[gidx(iu, wp.max(iv - 1, 0), nu)]
    yp = pos[gidx(iu, wp.min(iv + 1, nv - 1), nu)]
    # Rows run top-down in PlaneGeometry, so cross(dv, du) matches the front face.
    n = wp.cross(yp - ym, xp - xm)
    nrm[i] = n / wp.max(wp.length(n), 1.0e-9)


@wp.kernel
def clear_force(out: wp.array(dtype=wp.vec3)):
    out[wp.tid()] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def accum_force(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
                nrm: wp.array(dtype=wp.vec3), area: wp.array(dtype=float),
                vec_uni: wp.array(dtype=wp.vec3), dt: float,
                out: wp.array(dtype=wp.vec3)):
    """Newtons on the rig, from the same pressure the particles just felt."""
    i = wp.tid()
    v_rel = vec_uni[0] - (pos[i] - prev[i]) / dt
    n = nrm[i]
    vn = wp.dot(n, v_rel)
    wp.atomic_add(out, 0, n * (0.5 * RHO_AIR * area[i] * vn * wp.abs(vn)))


wp.init()
device = wp.get_preferred_device()


class Sail:
    """One cloth panel: its Warp state, its geometry, and where it is pinned."""

    def __init__(self, name, nu, nv, flat, cam, pin_idx, colour):
        self.name, self.nu, self.nv = name, nu, nv
        self.n = nu * nv
        self.flat = np.ascontiguousarray(flat, np.float32)
        self.cam = np.ascontiguousarray(cam, np.float32)
        self.area_total = float(vertex_areas(flat, nu, nv).sum())
        # Born slightly bellied, so the sheet does not start in a flat,
        # perfectly unstable plane and dither about which way to fill.
        p0 = self.flat + (self.cam - self.flat) * 0.35
        rng = np.random.default_rng(11)
        p0 = np.ascontiguousarray(p0 + rng.uniform(-2e-3, 2e-3, p0.shape), np.float32)

        inv_mass = np.ones(self.n, np.float32)
        inv_mass[pin_idx] = 0.0

        self.pos = wp.array(p0, dtype=wp.vec3, device=device)
        self.prev = wp.array(p0, dtype=wp.vec3, device=device)
        self.pred = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.scratch = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.nrm = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.rest = wp.array(self.cam, dtype=wp.vec3, device=device)
        self.inv_mass = wp.array(inv_mass, dtype=float, device=device)
        self.area = wp.array(vertex_areas(flat, nu, nv), dtype=float, device=device)
        self.pin_idx = wp.array(np.asarray(pin_idx, np.int32), dtype=int, device=device)
        self.pin_pos = wp.zeros(len(pin_idx), dtype=wp.vec3, device=device)
        self.force = wp.zeros(1, dtype=wp.vec3, device=device)
        self.vec_uni = wp.zeros(2, dtype=wp.vec3, device=device)
        self.sc_uni = wp.zeros(8, dtype=float, device=device)
        self.pin_host = np.zeros((len(pin_idx), 3), np.float32)

        mat = tp.MeshStandardMaterial()
        mat.color = colour
        mat.roughness = 0.58
        mat.metalness = 0.0
        mat.side = tp.Side.Double
        # Sailcloth is thin, and the sun behind it glows straight THROUGH: a
        # wrapped back-light plus a forward-scatter halo when you look toward
        # the sun past the leech. Without it a sail is a white cardboard cutout,
        # so this is turned up hard and the cloth albedo pulled down off pure
        # white -- the transmitted light has to out-read the reflected light.
        mat.translucency = 0.92
        mat.translucency_color = 0xffe6bc
        self.geometry = tp.PlaneGeometry(1.0, 1.0, nu - 1, nv - 1)
        self.geometry.update_attribute("position", p0)
        self.mesh = tp.Mesh(self.geometry, mat)
        self.mesh.cast_shadow = True
        self.mesh.frustum_culled = False   # positions change under the renderer's feet

    def push_uniforms(self, wind_local, grav_local, aero, tang, damp, st, sh, bd):
        self.vec_uni.assign(np.asarray([wind_local, grav_local], np.float32))
        self.sc_uni.assign(np.asarray([aero, tang, damp, st, sh, bd, 0.0, 0.0], np.float32))

    def push_pins(self):
        self.pin_pos.assign(self.pin_host)

    def upload(self):
        # Device -> host -> geometry. The tier-1 handoff: one copy per frame.
        self.geometry.update_attribute("position", self.pos.numpy())
        self.geometry.update_attribute("normal", self.nrm.numpy())

    def read_force(self):
        return self.force.numpy()[0].astype(np.float64)


def sail_launches(sails):
    """Every launch of one 60 fps frame; captured into a CUDA graph below."""
    for s in sails:
        wp.launch(clear_force, dim=1, device=device, inputs=[s.force])
    for _ in range(SUBSTEPS):
        for s in sails:
            wp.launch(apply_pins, dim=len(s.pin_host), device=device,
                      inputs=[s.pos, s.prev, s.pin_idx, s.pin_pos])
            wp.launch(compute_normals, dim=s.n, device=device,
                      inputs=[s.pos, s.nrm, s.nu, s.nv])
            wp.launch(integrate, dim=s.n, device=device,
                      inputs=[s.pos, s.prev, s.pred, s.nrm, s.inv_mass,
                              s.vec_uni, s.sc_uni, DT])
            a, b = s.pred, s.scratch
            for _ in range(ITERATIONS):
                wp.launch(solve, dim=s.n, device=device,
                          inputs=[a, b, s.rest, s.inv_mass, s.sc_uni, s.nu, s.nv])
                a, b = b, a
            # Ping-pong parity is resolved at trace time; copy back either way.
            wp.copy(s.pos, a)
    for s in sails:
        wp.launch(compute_normals, dim=s.n, device=device,
                  inputs=[s.pos, s.nrm, s.nu, s.nv])
        wp.launch(accum_force, dim=s.n, device=device,
                  inputs=[s.pos, s.prev, s.nrm, s.area, s.vec_uni, DT, s.force])


# --------------------------------------------------------------------------- #
#  Build the two sails.
#
#  Both are pinned along the whole luff -- the main slides on a mast track, the
#  jib is hanked to the forestay -- plus the single clew corner. A modern
#  loose-footed main is attached to the boom at tack and clew only, and that is
#  exactly what gives it the deep curve along the foot.
# --------------------------------------------------------------------------- #
def luff_and_clew(nu, nv):
    luff = np.arange(nv) * nu                     # column iu = 0, every row
    clew = np.array([(nv - 1) * nu + (nu - 1)])   # bottom row, leech edge
    return np.concatenate([luff, clew])


main_pins = luff_and_clew(MAIN_NU, MAIN_NV)
jib_pins = luff_and_clew(JIB_NU, JIB_NV)
main_flat, main_cam = main_rest(MAIN_NU, MAIN_NV)
jib_flat, jib_cam = jib_rest(JIB_NU, JIB_NV)

mainsail = Sail("main", MAIN_NU, MAIN_NV, main_flat, main_cam, main_pins, 0xdcd5c6)
jib = Sail("jib", JIB_NU, JIB_NV, jib_flat, jib_cam, jib_pins, 0xd6cfc0)
SAILS = [mainsail, jib]

print(f"sailboat: {sum(s.n for s in SAILS)} sail particles "
      f"({mainsail.area_total + jib.area_total:.1f} m2 of cloth) on {device}, "
      f"{SUBSTEPS * ITERATIONS * len(SAILS)} constraint launches per frame")


def set_pin_targets(boom_angle, jib_angle):
    """Where the rig holds the cloth this frame, in boat-local coordinates.

    The luff of the main sits ON the mast axis, so swinging the boom does not
    move it -- only the clew sweeps. That asymmetry is what gives a real sail
    its twist, and here it comes for free.
    """
    nv = MAIN_NV
    fv = 1.0 - np.arange(nv) / (nv - 1.0)
    mainsail.pin_host[:nv, 0] = 0.0
    mainsail.pin_host[:nv, 1] = GOOSE_Y + LUFF * fv
    mainsail.pin_host[:nv, 2] = MAST_Z
    mainsail.pin_host[nv] = (-math.sin(boom_angle) * FOOT,
                             GOOSE_Y,
                             MAST_Z - math.cos(boom_angle) * FOOT)

    nv = JIB_NV
    fv = (1.0 - np.arange(nv) / (nv - 1.0))[:, None]
    tack = np.array([0.0, DECK_Y, TACK_Z])
    head = np.array([0.0, JIB_HEAD, MAST_Z])
    jib.pin_host[:nv] = tack + (head - tack) * fv
    jib.pin_host[nv] = (-math.sin(jib_angle) * JIB_FOOT,
                        DECK_Y + JIB_CLEW_Y,
                        TACK_Z - math.cos(jib_angle) * JIB_FOOT)


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
#  Scene
# --------------------------------------------------------------------------- #
W, H = (int(v) for v in cli_arg("--size", "1600x900", str).lower().split("x"))
canvas = tp.Canvas("threepp x warp - sailboat", width=W, height=H,
                   vsync=False, headless=SHOT)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 0.72
renderer.render_scale = 0.9
renderer.gbuffer_msaa = 2            # steadies the rigging wires against the sky
renderer.sun_angular_radius = 0.6    # soft ray-traced sun shadows
renderer.bloom_intensity = 0.11
# A day that runs from a moonlit sea to a noon glare is ~10 stops wide, and no
# single tone_mapping_exposure covers it (auto exposure IGNORES that knob while
# it is on). The clamp is what keeps night reading as night: metering to 18%
# grey would happily turn midnight into an overcast afternoon, so the eye is
# only allowed +3 EV of dilation -- enough to find the lighthouse beam and the
# nav lights, nowhere near enough to fake daylight -- and -2.5 EV of
# constriction, which is what a sunlit sea needs to keep its highlights.
# 1.2 EV/s (0.6 dilating) settles a cut in about a second without pumping on a
# lightning flash; bloom_clamp stops a flash from smearing the whole frame.
renderer.auto_exposure = True
renderer.set_auto_exposure_range(-2.5, 3.0)
renderer.set_auto_exposure_speed(1.2)
renderer.bloom_clamp = 12.0

ui = tp.ImguiContext(canvas, renderer) if (tp.HAS_IMGUI and not SHOT) else None

scene = tp.Scene()

# ---- the sky, ping-ponged --------------------------------------------------- #
#  Replacing scene.environment is not a per-frame material patch: it re-uploads
#  the texture, re-runs PMREM, idles the device and cold-starts the ReSTIR
#  reservoirs. So there are two float textures, they are allocated ONCE, each
#  bake writes the back one in place, and a bake only happens when the sun has
#  actually moved (0.25 deg of elevation / 0.6 deg of azimuth) or the weather
#  changed -- the same throttle vulkan_fjord uses.
sky_tex = [tp.float_texture(np.zeros((SKY_H, SKY_W, 4), np.float32)) for _ in range(2)]
sky_front = 0
_bake_count = 0
_bake_ms = 0.0
_last_bake_elev, _last_bake_az, _since_bake = -99.0, -99.0, 1e9
_last_bake_sig = None
SKY_LOG = "--quiet" not in sys.argv


def _weather_sig():
    return (round(weather["coverage"], 2), round(weather["cloud_density"], 2),
            round(weather["rainbow"], 2))


def apply_sky(cs, force=False):
    """Re-bake the environment if it is worth it, and swap the ping-pong."""
    global sky_front, _bake_count, _bake_ms
    global _last_bake_elev, _last_bake_az, _since_bake, _last_bake_sig
    sig = _weather_sig()
    if not force:
        if _since_bake < 0.30:
            return False
        if (abs(cs.sun_elev_deg - _last_bake_elev) < 0.25
                and abs(cs.sun_az_deg - _last_bake_az) < 0.6
                and sig == _last_bake_sig):
            return False
    _last_bake_elev, _last_bake_az, _last_bake_sig = cs.sun_elev_deg, cs.sun_az_deg, sig
    _since_bake = 0.0

    t0 = time.perf_counter()
    data = bake_sky(cs, weather)
    back = 1 - sky_front
    sky_tex[back].update_float(data)
    sky_front = back
    scene.environment = sky_tex[back]
    scene.background = sky_tex[back]
    ms = (time.perf_counter() - t0) * 1e3
    _bake_count += 1
    _bake_ms += ms
    if SKY_LOG:
        print(f"sky bake {_bake_count:3d}  t={_last_tod():05.2f}h  "
              f"elev {cs.sun_elev_deg:+5.1f} deg  az {cs.sun_az_deg % 360:5.1f} deg  "
              f"cover {weather['coverage']:.2f}  {ms:5.1f} ms")
    return True


def _last_tod():
    return time_of_day


cs0 = celestial(time_of_day)
apply_sky(cs0, force=True)

# One sun, aligned with the sun painted into the sky, so the specular highlight
# on the water and the shafts through the rig agree with the background. And a
# moon, which is the same light with a different colour temperature and only
# ever on when the sun is not.
sun = tp.DirectionalLight(0xffdcaa, 3.2)
sun.position.set(*(cs0.sun_dir * 1200.0))
scene.add(sun)
moon = tp.DirectionalLight(0x9fb4ff, 0.0)
moon.position.set(*(cs0.moon_dir * 1200.0))
scene.add(moon)

SEA = 1400.0
ocean = tp.Ocean(size=SEA, resolution=640, wind_speed=weather["wind"], wind_theta=0.6,
                 choppiness=weather["choppy"], fft_size=1024)
scene.add(ocean)

# Dark water below the surface so refraction has something to read against.
floor_mat = tp.MeshStandardMaterial()
floor_mat.color = 0x04070a
floor_mat.roughness = 1.0
floor = tp.Mesh(tp.PlaneGeometry(SEA, SEA), floor_mat)
floor.rotate_x(-math.pi / 2)
floor.position.y = -6.0
scene.add(floor)

# ---- the islet and its light ----------------------------------------------- #
#  A crag with a lighthouse on it -- the one thing on this horizon that is not
#  water, and the reason the gulls have somewhere to sit.
ISLET_X, ISLET_Z = -95.0, 115.0
ISLET_R, ISLET_H = 32.0, 15.5
PAD_FRAC = 0.24

rock_mat = tp.MeshStandardMaterial()
rock_mat.color = 0xbdb3a0
rock_mat.roughness = 0.92
rock_mat.metalness = 0.0

# The tower's weathering is a generated texture, not a flat colour -- see
# lighthouse_maps(). color stays white so the map is not tinted twice.
_alb, _rgh = lighthouse_maps(17)
stone_mat = tp.MeshStandardMaterial()
stone_mat.color = 0xffffff
stone_mat.map = tp.data_texture(_alb, srgb=True)
stone_mat.roughness_map = tp.data_texture(_rgh, srgb=False)
stone_mat.roughness = 1.0                    # the map scales this
stone_mat.metalness = 0.0
stone_mat.side = tp.Side.Double

lamp_glass = tp.MeshStandardMaterial()
lamp_glass.color = 0x1a1d20
lamp_glass.roughness = 0.08
lamp_glass.side = tp.Side.Double

metal_mat = tp.MeshStandardMaterial()
metal_mat.color = 0x4a3a2c                   # iron that gave up on paint
metal_mat.roughness = 0.72
metal_mat.metalness = 0.55
metal_mat.side = tp.Side.Double

islet = tp.Group()
islet.position.set(ISLET_X, 0.0, ISLET_Z)
scene.add(islet)

ig = build_islet(5, ISLET_R, ISLET_H, PAD_FRAC)
rv, rn = _flat_soup(ig, inside=(0.0, -30.0, 0.0))
rock_geom = tp.BufferGeometry()
rock_geom.set_attribute("position", rv)
rock_geom.set_attribute("normal", rn)
rock = tp.Mesh(rock_geom, rock_mat)
rock.receive_shadow = True
rock.cast_shadow = True
islet.add(rock)

# A pale shelf just under the shallows. The ocean is transmissive, so what the
# water reads as is mostly whatever is on the bottom -- sand here, deep dark
# everywhere else, which is the whole turquoise-round-a-rock effect.
shelf_mat = tp.MeshStandardMaterial()
shelf_mat.color = 0xb9c7a8
shelf_mat.roughness = 1.0
# Kept tight to the rock and set deep: a broad shelf shows its own rim through
# the water as a hard-edged patch, which reads as a bug rather than as a bank.
shelf = tp.Mesh(tp.CylinderGeometry(ISLET_R * 1.25, ISLET_R * 1.55, 1.0, 40, 1), shelf_mat)
shelf.position.set(0.0, -4.4, 0.0)
islet.add(shelf)

# The lighthouse: plinth, drum, tapered tower, gallery, lantern, dome.
PAD_Y = ISLET_H * 1.02 - 5.0
TOWER = [(0.0, 4.55), (1.5, 4.15), (1.7, 4.05), (1.9, 3.62),
         (3.2, 3.50), (3.4, 3.34), (12.4, 2.42), (13.0, 2.38),
         (13.2, 3.16), (13.9, 3.20), (14.05, 2.10), (17.0, 2.02),
         (17.3, 2.28), (18.0, 1.30), (18.7, 0.34), (19.1, 0.10)]
BANDS = [(0, 10, stone_mat), (10, 12, lamp_glass), (12, 15, metal_mat)]

light_house = tp.Group()
light_house.position.set(0.0, PAD_Y - 0.6, 0.0)
islet.add(light_house)

tg, tuv = build_tower(TOWER)
for r0, r1, paint in BANDS:
    sub = np.ascontiguousarray(tg[r0:r1 + 1])
    suv = np.ascontiguousarray(tuv[r0:r1 + 1])
    g = tp.BufferGeometry()
    g.set_attribute("position", np.ascontiguousarray(sub.reshape(-1, 3), np.float32))
    g.set_attribute("uv", np.ascontiguousarray(suv.reshape(-1, 2), np.float32))
    g.set_index(_orient(sub.reshape(-1, 3),
                        _grid_indices(sub.shape[0], sub.shape[1]),
                        (0.0, sub[:, :, 1].mean(), 0.0)))
    g.compute_vertex_normals()
    m = tp.Mesh(g, paint)
    m.cast_shadow = True
    m.receive_shadow = True
    light_house.add(m)

# Gallery rail round the lantern, and the lamp itself. On Vulkan an emissive
# mesh IS a light source, so this one really does throw light into the mist.
for k in range(16):
    a0 = 2.0 * math.pi * k / 16.0
    p0 = (3.05 * math.cos(a0), 13.9, 3.05 * math.sin(a0))
    light_house.add(strut(p0, (p0[0], p0[1] + 1.05, p0[2]), 0.035, metal_mat))
for h in (14.35, 14.95):
    prev = None
    for k in range(33):
        a0 = 2.0 * math.pi * k / 32.0
        p = (3.05 * math.cos(a0), h, 3.05 * math.sin(a0))
        if prev is not None:
            light_house.add(strut(prev, p, 0.022, metal_mat))
        prev = p

lamp_mat = tp.MeshStandardMaterial()
lamp_mat.color = 0x120e08
lamp_mat.emissive = 0xffc978
lamp_mat.emissive_intensity = 46.0
lamp = tp.Mesh(tp.SphereGeometry(0.72, 20, 14), lamp_mat)
lamp.position.set(0.0, 15.5, 0.0)
light_house.add(lamp)

# ---- the boat -------------------------------------------------------------- #
boat = tp.Group()
boat.rotation.order = tp.RotationOrder.YXZ   # yaw, then pitch, then roll
scene.add(boat)

# The hull bands, the coachroof and the deck are all OPEN lofts -- strips and a
# trunk with open ends, not solids. Seen from behind, a one-sided face is not
# drawn at all, so you look straight through the cabin top into the boat.
hull_mat = tp.MeshStandardMaterial()
hull_mat.color = 0xf4f5f3
hull_mat.roughness = 0.30                    # painted topsides: glossy dielectric
hull_mat.metalness = 0.0
hull_mat.side = tp.Side.Double

boot_mat = tp.MeshStandardMaterial()
boot_mat.color = 0x14202e                    # boot stripe / cove line
boot_mat.roughness = 0.38
boot_mat.side = tp.Side.Double

anti_mat = tp.MeshStandardMaterial()
anti_mat.color = 0x1d3550                    # antifouling below the waterline
anti_mat.roughness = 0.62
anti_mat.side = tp.Side.Double

deck_mat = tp.MeshStandardMaterial()
deck_mat.color = 0xb9a382                    # teak
deck_mat.roughness = 0.80
# The deck is a single open shell, not a solid: seen from below (or through the
# cockpit well from a low camera) a one-sided deck simply is not there.
deck_mat.side = tp.Side.Double

spar_mat = tp.MeshStandardMaterial()
spar_mat.color = 0xc9ccd0                    # anodised alloy
spar_mat.roughness = 0.34
spar_mat.metalness = 0.85

tiller_mat = tp.MeshStandardMaterial()
tiller_mat.color = 0x8a6033                  # varnished ash
tiller_mat.roughness = 0.28

glass_mat = tp.MeshStandardMaterial()
glass_mat.color = 0x0d1418                   # smoked cabin windows
glass_mat.roughness = 0.08
glass_mat.metalness = 0.0
glass_mat.side = tp.Side.Double

wire_mat = tp.MeshStandardMaterial()
wire_mat.color = 0x9aa0a6
wire_mat.roughness = 0.30
wire_mat.metalness = 0.95

# Girth index runs 0 (starboard deck edge) .. NP-1 (keel) .. 2*NP-2 (port).
HG = hull_grid()
KEEL_C = NP - 1
BOOT_LO, BOOT_HI = 15, 18                 # girth rows straddling the waterline
for c0, c1, paint in (
        (KEEL_C - BOOT_LO, KEEL_C + BOOT_LO, anti_mat),     # antifouling
        (KEEL_C - BOOT_HI, KEEL_C - BOOT_LO, boot_mat),     # boot stripe, stbd
        (KEEL_C + BOOT_LO, KEEL_C + BOOT_HI, boot_mat),     # boot stripe, port
        (0, KEEL_C - BOOT_HI, hull_mat),                    # topsides, stbd
        (KEEL_C + BOOT_HI, 2 * NP - 2, hull_mat)):          # topsides, port
    bv, bi = band(HG, c0, c1)
    boat.add(_mesh(bv, bi, paint, inside=(0.0, 0.2, 0.0)))

# Toe rail along the sheer: a hard edge where deck meets topside, so the deck
# line reads as a line instead of dissolving into the hull.
rail = HG[:, 0:1, :].copy()
for sgn, col in ((1.0, 0), (-1.0, 2 * NP - 2)):
    edge = HG[:, col, :].copy()
    top = edge.copy()
    top[:, 1] += 0.055
    top[:, 0] += sgn * 0.012
    rv = np.stack([edge, top], axis=1).reshape(-1, 3)
    ri = _grid_indices(NS, 2)
    boat.add(_mesh(rv, ri, boot_mat, inside=(0.0, -2.0, 0.0)))
dv, di = build_deck()
boat.add(_mesh(dv, di, deck_mat, inside=(0.0, -6.0, 0.0)))
# Coachroof, with a smoked window strip down each side. The windows are just
# the near-vertical columns of the same loft, painted dark -- and they are what
# makes the trunk read as a cabin rather than a blister.
CG = cabin_grid()
CNC = CG.shape[1]
for c0, c1, paint in ((0, 3, hull_mat), (3, 7, glass_mat), (7, CNC - 8, hull_mat),
                      (CNC - 8, CNC - 4, glass_mat), (CNC - 4, CNC - 1, hull_mat)):
    bv, bi = band(CG, c0, c1)
    boat.add(_mesh(bv, bi, paint, inside=(0.0, 1.0, 0.0)))
for ring, inside in ((CG[0], (0.0, 1.0, -1.0)), (CG[-1], (0.0, 1.0, 1.0))):
    fv_, fi_ = fan_cap(ring)
    boat.add(_mesh(fv_, fi_, hull_mat, inside=inside))
tv, ti_ = build_transom()
boat.add(_mesh(tv, ti_, hull_mat, inside=(0.0, 0.2, 2.0)))

kv, ki = build_foil(1.35, 0.85, 1.80, 0.12, 0.28)
keel = _mesh(kv, ki, anti_mat, inside=(0.0, -0.9, 0.0))
keel.position.set(0.0, -0.45, 0.15)
boat.add(keel)
# Ballast bulb: most of the 3.5 t lives down here, and it is the reason the
# righting moment in the heel equation is as strong as it is.
bulb = tp.Mesh(tp.SphereGeometry(0.30, 20, 12), anti_mat)
bulb.scale.set(1.0, 0.72, 3.1)
bulb.position.set(0.0, -2.22, -0.10)
boat.add(bulb)

# Rudder assembly. Everything hangs off one pivot so blade, stock and tiller
# swing together -- the tiller is the visible half of the same rotation that
# the yaw equation reads out of `rudder`.
rv, ri = build_foil(0.72, 0.46, 1.25, 0.11, 0.10)
rudder_pivot = tp.Group()
rudder_pivot.position.set(0.0, -0.15, -3.80)
rudder_pivot.add(_mesh(rv, ri, anti_mat, inside=(0.0, -0.6, 0.0)))
stock = tp.Mesh(tp.CylinderGeometry(0.045, 0.052, 1.32, 10, 1), spar_mat)
stock.position.set(0.0, 0.60, 0.0)
stock.cast_shadow = True
rudder_pivot.add(stock)
# Tiller: up out of the stock head and forward into the cockpit, where a hand
# would be. On a 9.6 m boat this is the helm.
rudder_pivot.add(strut((0.0, 1.24, 0.02), (0.0, 1.46, 1.55), 0.038, tiller_mat))
grip = tp.Mesh(tp.CylinderGeometry(0.031, 0.031, 0.30, 10, 1), boot_mat)
grip.position.set(0.0, 1.45, 1.42)
grip.rotate_x(math.pi / 2 - 0.14)
rudder_pivot.add(grip)
boat.add(rudder_pivot)

mast = tp.Mesh(tp.CylinderGeometry(0.055, 0.085, MAST_H - DECK_Y, 14, 1), spar_mat)
mast.position.set(0.0, DECK_Y + 0.5 * (MAST_H - DECK_Y), MAST_Z)
mast.cast_shadow = True
boat.add(mast)

# The boom hangs off a pivot at the gooseneck; swinging the pivot is the
# mainsheet, and the sail's clew pin follows the same angle.
boom_pivot = tp.Group()
boom_pivot.position.set(0.0, GOOSE_Y, MAST_Z)
boat.add(boom_pivot)
boom = tp.Mesh(tp.CylinderGeometry(0.048, 0.048, FOOT + 0.25, 12, 1), spar_mat)
boom.rotate_x(math.pi / 2)
boom.position.set(0.0, 0.0, -(FOOT + 0.25) * 0.5)
boom.cast_shadow = True
boom_pivot.add(boom)

def deck_edge(t, side):
    """The point on the sheer at fore-aft fraction t, port (-1) or starboard (+1)."""
    xs, ys = _half_section(t)
    return np.array([side * xs[-1], ys[-1], (0.5 - t) * LOA])


def build_rails():
    """Stanchions and lifelines round the deck, plus a pulpit and a pushpit.

    Nothing here is structural, but a yacht without guardrails reads as a model
    of a yacht. The wires follow the sheer because they are strung between the
    stanchion tops, so the sheerline gets drawn twice and the eye picks it up.
    """
    parts = []
    ts = np.linspace(0.12, 0.90, 7)
    for side in (-1.0, 1.0):
        tops, mids = [], []
        for t in ts:
            base = deck_edge(t, side)
            base[0] -= side * 0.045                    # inboard of the toe rail
            top = base + np.array([0.0, 0.60, 0.0])
            parts.append(strut(base, top, 0.017, spar_mat))
            tops.append(top)
            mids.append(base + np.array([0.0, 0.30, 0.0]))
        # Upper and lower guard wires, threaded stanchion to stanchion.
        for line in (tops, mids):
            for a, b in zip(line[:-1], line[1:]):
                parts.append(strut(a, b, 0.008, wire_mat))
        # Pulpit forward and pushpit aft: the wire turns up and inboard and
        # closes on the centreline.
        bow = deck_edge(0.035, side * 0.35) + np.array([0.0, 0.60, 0.0])
        parts.append(strut(tops[0], bow, 0.014, spar_mat))
        parts.append(strut(deck_edge(0.035, side * 0.35), bow, 0.014, spar_mat))
        stern = deck_edge(0.975, side * 0.55) + np.array([0.0, 0.60, 0.0])
        parts.append(strut(tops[-1], stern, 0.014, spar_mat))
        parts.append(strut(deck_edge(0.975, side * 0.55), stern, 0.014, spar_mat))
    # Close the pulpit and the pushpit across the centreline.
    parts.append(strut(deck_edge(0.035, -0.35) + np.array([0.0, 0.60, 0.0]),
                       deck_edge(0.035, 0.35) + np.array([0.0, 0.60, 0.0]), 0.014, spar_mat))
    parts.append(strut(deck_edge(0.975, -0.55) + np.array([0.0, 0.60, 0.0]),
                       deck_edge(0.975, 0.55) + np.array([0.0, 0.60, 0.0]), 0.014, spar_mat))
    return parts


for part in build_rails():
    boat.add(part)

masthead = (0.0, MAST_H, MAST_Z)
boat.add(strut((0.0, JIB_HEAD, MAST_Z), (0.0, DECK_Y - 0.05, TACK_Z), 0.016, wire_mat))
boat.add(strut(masthead, (0.0, DECK_Y - 0.05, -4.55), 0.016, wire_mat))
for side in (-1.0, 1.0):
    boat.add(strut(masthead, (side * 1.18, DECK_Y, MAST_Z + 0.15), 0.013, wire_mat))
    boat.add(strut((0.0, 6.6, MAST_Z), (side * 1.22, DECK_Y, MAST_Z - 1.35), 0.011, wire_mat))

for s in SAILS:
    boat.add(s.mesh)

# ---- gulls ------------------------------------------------------------------ #
#  threepp's own ambient flock (extras/fauna): boids that fly, bank into their
#  turns, land, walk about and lift off again. The islet is scanned for landable
#  rock at startup, so they really do sit on it.
gull = tp.FlockParams()
gull.seed = 4711
gull.bird_count = cli_arg("--birds", 34, int)
gull.home = tp.Vector3(ISLET_X, ISLET_H + 12.0, ISLET_Z)
gull.roam_radius = 76.0
gull.cruise_altitude = 15.0
gull.cruise_speed = 10.5
gull.max_speed = 19.0
gull.mass_kg = 0.95                # herring gull, not a finch
gull.gait = tp.Gait.Walk
gull.shape.body_length = 0.58
gull.shape.body_radius = 0.075
gull.shape.wing_span = 1.42
gull.shape.tail_fork = 0.15
gull.plumage.back = tp.Color(0.48, 0.52, 0.57)      # grey mantle
gull.plumage.belly = tp.Color(0.88, 0.88, 0.86)     # white underside
gull.plumage.cap = tp.Color(0.80, 0.78, 0.72)
gull.plumage.leg = tp.Color(0.85, 0.62, 0.22)
gull.plumage.wingtip_dark = 0.22                    # black primaries
gull.w_cohesion = 0.30                              # gulls are loose, not starlings
gull.w_alignment = 0.45
gull.loner_fraction = 0.30
gull.perching = True
gull.perch_search_radius = 70.0
gull.max_perched_fraction = 0.45
gull.birds_cast_shadow = False
gulls = tp.Flock(gull)
scene.add(gulls)

# ---- the night kit ---------------------------------------------------------- #
#  Everything here is pre-created and driven by intensity alone. Adding or
#  removing a scene object is a structural change (entry-list rebuild, device
#  idle, TAA history clear), so nothing in this demo is ever added after
#  startup -- the night simply turns the day's zeros into numbers.
LANTERN = (ISLET_X, PAD_Y - 0.6 + 15.5, ISLET_Z)

# Two beams 180 deg apart out of one lantern, which is what an optic with two
# lens panels actually throws. decay 2 = physical inverse square, so the
# intensity has to be lighthouse-sized (O(10^5) cd) before the far end of the
# beam lights anything; the cone is narrow enough to read as a shaft in the
# mist rather than a floodlight.
BEAM_INTENSITY = 380000.0
BEAM_PERIOD = 6.0                          # seconds per revolution
beams, beam_targets = [], []
for k in range(2):
    b = tp.SpotLight(tp.Color(1.0, 0.93, 0.74), 0.0, 2500.0,
                     math.radians(4.0), 0.5, 2.0)
    b.position.set(*LANTERN)
    t = tp.Group()
    t.position.set(LANTERN[0] + 900.0, LANTERN[1] - 22.0, LANTERN[2])
    scene.add(t)
    b.set_target(t)
    scene.add(b)
    beams.append(b)
    beam_targets.append(t)

# Navigation lights: red to port, green to starboard, white astern. Tiny
# emissive spheres, which on Vulkan are real light sources -- so they put a
# coloured wash on the deck beside them, which is exactly what they do at sea.
NAV_INTENSITY = 55.0
nav_mats = []
for colour, side, along, name in ((0xff1408, -1.0, 0.34, "port"),
                                  (0x18ff3c, 1.0, 0.34, "stbd"),
                                  (0xfff0d8, 0.0, 0.965, "stern")):
    m = tp.MeshStandardMaterial()
    m.color = 0x101010
    m.emissive = colour
    m.emissive_intensity = 0.0
    p = deck_edge(along, side * 0.92) if side != 0.0 else deck_edge(along, 0.0)
    s = tp.Mesh(tp.SphereGeometry(0.075, 12, 8), m)
    s.position.set(float(p[0]), float(p[1]) + 0.30, float(p[2]))
    boat.add(s)
    nav_mats.append(m)

# Cabin glow: the smoked window strips are already their own material band, so
# the light below decks is one emissive ramp on a material that exists.
glass_mat.emissive = 0xffb45a
glass_mat.emissive_intensity = 0.0

# Wet surfaces. A wet deck is not a darker deck, it is a SMOOTHER one: the
# water fills the grain, roughness collapses, and the sky arrives as a sheen
# instead of a diffuse wash. Darkening the albedo on its own reads as dirt.
WET_MATS = ((deck_mat, (0.725, 0.639, 0.510), 0.80, 0.20),
            (hull_mat, (0.957, 0.961, 0.953), 0.30, 0.14),
            (boot_mat, (0.078, 0.125, 0.180), 0.38, 0.16),
            (spar_mat, (0.788, 0.800, 0.816), 0.34, 0.18))
_wetness_applied = -1.0


def apply_wetness(w):
    """Tween the deck, hull, boot stripe and spars between dry and streaming."""
    global _wetness_applied
    if abs(w - _wetness_applied) < 0.02:
        return
    _wetness_applied = w
    for mat, base, dry, wet in WET_MATS:
        mat.roughness = dry + (wet - dry) * w
        k = 1.0 - 0.35 * w                       # wet paint reads ~a third down
        mat.color = tp.Color(base[0] * k, base[1] * k, base[2] * k)
        mat.needs_update()


# ---- rain ------------------------------------------------------------------- #
#  ONE field, created at its final capacity and never resized, parked with
#  set_live_count(0) when it is not raining. The emitter is a closed form on
#  the GPU -- the CPU writes one 64-byte record a frame and never touches a
#  drop. `follow` wraps the whole slab toroidally around the camera, so the
#  rain is everywhere the boat sails rather than a patch over the origin.
RAIN_CAP = 200_000
RAIN_HALF = 26.0                    # lateral half-extent = the wrap period / 2
RAIN_LOD_NEAR, RAIN_LOD_FAR = 6.0, 12.0

_rc = tp.ParticleField.Config()
_rc.capacity = RAIN_CAP
_rc.ownership = tp.ParticleField.Ownership.Renderer
_rc.w_semantic = tp.ParticleField.WSemantic.Radius
_rc.uniform_radius = 0.013
rain = tp.ParticleField.create(_rc)
rain.frustum_culled = False

# Near the camera a drop is a lit SOLID -- which is the whole point of the LOD
# split here: a lightning flash (Phase 3) reaches a mesh proxy and cannot reach
# an unlit additive quad. Past 12 m a drop is 2 px and the octahedron buys
# nothing, so the billboard streak takes over across the same band.
rain_mat = tp.MeshStandardMaterial()
rain_mat.color = tp.Color(0.58, 0.64, 0.72)
rain_mat.roughness = 0.06
rain_mat.metalness = 0.0
rain.set_mesh_repr(tp.OctahedronGeometry(0.013, 0), rain_mat)
_mr = rain.mesh_repr
_mr.lod_far = RAIN_LOD_FAR
_mr.lod_fade = RAIN_LOD_FAR - RAIN_LOD_NEAR
# near_cull is "how big may the NEAREST drop get": inside the band the linear
# scale ramp and the 1/d projection cancel, so every proxy lands at r/near_cull
# radians. At 2.2 m that is 0.006 rad -- a 12 px white diamond hanging in front
# of the lens, which is exactly what the first storm still showed. 9 m puts the
# nearest drop at 3 px, which is a drop.
_mr.near_cull = 9.0

# A drop crosses ~20 px a frame; drawn as a solid it reads as hail. The quad is
# smeared along the emitter's own analytic velocity, and both caps (world and
# screen) are there to stop the nearest drop painting a bar across the frame.
RAIN_BB_BASE = 0.135
rain.set_billboard_repr(tp.Color(0.72, 0.79, 0.90), tp.Color(0.60, 0.67, 0.78),
                        RAIN_BB_BASE, 0.30)
_br = rain.billboard_repr
_br.lod_near = RAIN_LOD_NEAR
_br.lod_fade = RAIN_LOD_FAR - RAIN_LOD_NEAR
_br.stretch_seconds = 0.024
_br.stretch_max = 30.0
_br.stretch_max_screen = 0.045
_br.near_fade = 1.20
_br.softness = 0.95
_br.fade_power = 0.0
_br.size_taper = 0.0
_br.bright_jitter = 0.55
_br.splash_ring_width = 0.28
_br.glow = 0.0

_re = rain.emitter                    # NB: a COPY -- mutate and hand it back
_re.spawn_center = tp.Vector3(0.0, 16.0, 0.0)
_re.spawn_half_extent = tp.Vector3(RAIN_HALF, 0.35, RAIN_HALF)
_re.velocity = tp.Vector3(0.0, -9.0, 0.0)
_re.speed_spread = 0.35
_re.wind = tp.Vector3(1.30, 0.0, 0.45)
_re.drift_amplitude = 0.03
_re.drift_frequency = 0.9
_re.drift_scale = 6.0
_re.lifetime = 2.4
_re.duty_cycle = 0.92
_re.size = 0.013
_re.size_jitter = 0.30
_re.seed = 20260822
_re.follow = True
# An integer number of density voxels: the curtain volume below is 52 m across
# 96 voxels, so half a voxel of snap would re-phase the haze and make it swim.
_re.follow_snap = 8.0 * (2.0 * RAIN_HALF / 96.0)
_rs = _re.surface
_rs.enabled = True                    # drops land on the deck AND on the sea
_rs.resolution = 512
_rs.splash_seconds = 0.30
_rs.splash_grow = 12.0
_rs.bias = 0.010
_rs.rest_seconds = 0.0
_rs.fade_seconds = 0.05
_re.surface = _rs
rain.set_emitter(_re)
rain.set_emitter_time(0.0, 1.0 / 60.0)

# The curtain itself: a rain squall is a DARK medium, and without it heavy rain
# is a lot of bright streaks in perfectly clear air.
rain.set_density_repr(tp.Vector3(0.0, 8.0, 0.0), tp.Vector3(RAIN_HALF, 9.0, RAIN_HALF),
                      0.014, 96)
_dr = rain.density_repr
_dr.albedo = tp.Color(0.42, 0.46, 0.52)
_dr.anisotropy = 0.35
_dr.enabled = False
rain.set_live_count(0)
scene.add(rain)

# --------------------------------------------------------------------------- #
#  Lightning.
#
#  Eight bolts are built ONCE at startup and never touched again. A strike is
#  three cheap things: two TRANSFORMS (park / un-park the channel), one
#  emissive number a frame, and two point-light intensities. Building geometry
#  per strike would rebuild the entry list, idle the device and cold-start the
#  TAA history -- on the one frame in the film where that is most visible.
#
#  The emissive channel is what lights the SURFACES (sails, sea, and the rain's
#  lit mesh proxies near the camera); emissive triangles are not in the froxel
#  pass, so the AIR flash -- the rain curtain, the mist, the cloud base -- needs
#  real clustered point lights. That is why there are both.
# --------------------------------------------------------------------------- #
LRNG = np.random.default_rng(cli_arg("--seed", 90210, int))
CLOUD_BASE = 300.0          # where a channel leaves the deck (storm deck is 240 m)
BOLT_PARK = -9000.0         # inactive channels live down here; a transform, not a rebuild
FLASH_RANGE = 8000.0        # point-light cull radius: must clear the strike distance
STRIKE_SLOTS = 2            # two concurrent strikes, four bolts each


def _perp_frame(d):
    """Unit tangent plus two perpendiculars for each row of `d`."""
    t = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    ref = np.tile(np.array([0.0, 0.0, 1.0]), (len(t), 1))
    ref[np.abs(t[:, 2]) > 0.9] = np.array([1.0, 0.0, 0.0])
    u = np.cross(t, ref)
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
    return t, u, np.cross(t, u)


def _displace(rng, a, b, jitter, levels):
    """Midpoint displacement between two points -- the shape of a leader."""
    pts = np.stack([np.asarray(a, np.float64), np.asarray(b, np.float64)])
    for lv in range(levels):
        seg = pts[1:] - pts[:-1]
        _, u, v = _perp_frame(seg)
        amp = jitter * (0.58 ** lv)
        mid = 0.5 * (pts[:-1] + pts[1:])
        mid = mid + (rng.normal(size=(len(seg), 1)) * u
                     + rng.normal(size=(len(seg), 1)) * v) * amp
        out = np.empty((2 * len(pts) - 1, 3))
        out[0::2], out[1::2] = pts, mid
        pts = out
    return pts


def _tube(pts, r0, r1, sides):
    """A tapering tube round a polyline, as a non-indexed triangle soup."""
    seg = np.diff(pts, axis=0)
    tang = np.zeros_like(pts)
    tang[:-1] += seg
    tang[1:] += seg
    _, u, v = _perp_frame(tang)
    rad = (r0 + (r1 - r0) * np.linspace(0.0, 1.0, len(pts)) ** 0.7)[:, None, None]
    ang = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    ring = pts[:, None, :] + rad * (np.cos(ang)[None, :, None] * u[:, None, :]
                                    + np.sin(ang)[None, :, None] * v[:, None, :])
    a, b = ring[:-1], ring[1:]
    a2, b2 = np.roll(a, -1, axis=1), np.roll(b, -1, axis=1)
    return np.concatenate([np.stack([a, b, a2], axis=2),
                           np.stack([a2, b, b2], axis=2)], axis=1).reshape(-1, 3)


def _soup_mesh(v, mat):
    tri = v.reshape(-1, 3, 3)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    g = tp.BufferGeometry()
    g.set_attribute("position", np.ascontiguousarray(v, np.float32))
    g.set_attribute("normal", np.ascontiguousarray(np.repeat(n, 3, axis=0), np.float32))
    m = tp.Mesh(g, mat)
    m.cast_shadow = False              # a light source does not cast a shadow
    m.receive_shadow = False
    m.frustum_culled = False
    return m


def build_bolt(rng, ground_y, r_main, branches, spread):
    """One channel: main stroke plus branches, all tapering toward their tips.

    The main radius is metres, not centimetres, on purpose: at 0.8-2.5 km a
    1.5 m channel is ~2 px at 1080p, which is the thinnest a bolt may be and
    still survive TAA and the upscalers.
    """
    top = np.array([rng.normal() * 14.0, CLOUD_BASE * rng.uniform(0.90, 1.20),
                    rng.normal() * 14.0])
    bot = np.array([rng.normal() * spread, ground_y, rng.normal() * spread])
    main = _displace(rng, top, bot, spread * 0.42, 6)
    parts = [_tube(main, r_main, r_main * 0.30, 5)]
    for _ in range(branches):
        i = int(rng.integers(len(main) // 4, len(main) - 8))
        p0 = main[i]
        d = np.array([rng.normal(), -abs(rng.normal()) - 0.6, rng.normal()])
        d /= np.linalg.norm(d)
        ln = max(28.0, (p0[1] - ground_y) * rng.uniform(0.30, 0.65))
        p1 = p0 + d * ln
        p1[1] = max(p1[1], ground_y + 8.0)
        parts.append(_tube(_displace(rng, p0, p1, ln * 0.18, 4),
                           r_main * 0.42, r_main * 0.10, 4))
    return np.concatenate(parts, axis=0)


BOLT_EMISSIVE = tp.Color(0.78, 0.84, 1.0)
bolt_mats, bolt_nodes, strike_anchors, flash_lights = [], [], [], []
_bolt_tris = 0
for _a in range(STRIKE_SLOTS):
    _anc = tp.Group()
    _anc.position.set(0.0, -20000.0, 0.0)
    scene.add(_anc)
    strike_anchors.append(_anc)
    for _k in range(4):
        _rng = np.random.default_rng(7000 + _a * 10 + _k)
        # #3 of each four is an INTRA-CLOUD channel: it stops 300 m up, inside
        # the storm deck, so it reads as a sheet glow rather than a fork.
        _v = build_bolt(_rng, 300.0 if _k == 3 else 0.0,
                        (2.0, 1.6, 1.25, 1.8)[_k], 2 + _k % 3, 55.0)
        _bolt_tris += len(_v) // 3
        _m = tp.MeshStandardMaterial()
        _m.color = 0x000000                 # black at rest; only emissive ever shows
        _m.emissive = BOLT_EMISSIVE
        _m.emissive_intensity = 0.0
        _m.roughness = 1.0
        _m.metalness = 0.0
        _m.side = tp.Side.Double            # the soup's winding is not worth deriving
        _node = tp.Group()
        _node.position.set(0.0, BOLT_PARK, 0.0)
        _node.add(_soup_mesh(_v, _m))
        _anc.add(_node)
        bolt_mats.append(_m)
        bolt_nodes.append(_node)
    pair = []
    for _k in range(2):
        _pl = tp.PointLight(tp.Color(0.80, 0.86, 1.0), 0.0, FLASH_RANGE, 2.0)
        _pl.position.set(0.0, -20000.0, 0.0)
        scene.add(_pl)
        pair.append(_pl)
    flash_lights.append(pair)

# What a kind of strike IS. `lux` is the illuminance the two air lights deliver
# AT THE BOAT at peak, so the point-light intensity is lux * d^2 and a strike
# looks the same from 600 m or from 2.5 km -- the kind, not the geometry, is
# what says how big it was. For reference the storm's own sun runs at ~0.55 and
# a clear noon at ~4.6 in the same units.
STRIKE_KINDS = {
    "sheet": dict(bolts=(), emis=0.0, lux=1.8, strokes=(2, 4),
                  dist=(1500.0, 2500.0), mid=520.0, top=820.0),
    "intra": dict(bolts=(3,), emis=1800.0, lux=2.4, strokes=(2, 3),
                  dist=(1300.0, 2200.0), mid=430.0, top=700.0),
    "mid": dict(bolts=(1, 2), emis=2800.0, lux=3.6, strokes=(2, 3),
                dist=(1000.0, 1900.0), mid=140.0, top=300.0),
    "hero": dict(bolts=(0,), emis=4200.0, lux=9.0, strokes=(3, 4),
                 dist=(600.0, 900.0), mid=130.0, top=300.0),
}
strikes = [dict(on=False, t=0.0, dur=0.0, bolt=-1, strokes=(),
                emis=0.0, lux=0.0, dist=1000.0) for _ in range(STRIKE_SLOTS)]
_next_strike = 3.0
flash_level = 0.0            # 0..1, the frame's brightest active stroke
# Auto exposure is a 10-stop-wide lever in this demo and a bolt is the brightest
# thing in the film; without a floor the meter constricts on the flash and the
# two seconds AFTER it come back grey. -1.2 EV is as far down as the eye may go
# while a channel is lit.
AE_RANGE = (-2.5, 3.0)
AE_FLASH_MIN = -1.2
_ae_flashing = False


def lightning_anchor(bearing_deg, distance_m, slot=0):
    """Put a strike pool at a bearing / distance from the boat (+Z is north).

    The pool is REPOSITIONED, never rebuilt, and given a random yaw so the same
    eight channels never present the same silhouette twice.
    """
    a = math.radians(bearing_deg)
    g = strike_anchors[slot]
    g.position.set(boat_state["x"] + distance_m * math.sin(a), 0.0,
                   boat_state["z"] + distance_m * math.cos(a))
    g.rotation.y = float(LRNG.uniform(0.0, 2.0 * math.pi))
    return g


def fire_strike(kind="mid", bearing=None, distance=None, slot=None):
    """Fire one strike NOW. Returns the thunder delay in seconds (no audio yet).

    This is the call a scripted shot makes. `kind` is sheet / intra / mid /
    hero; bearing is degrees from +Z, distance is metres from the boat.
    """
    spec = STRIKE_KINDS.get(kind, STRIKE_KINDS["mid"])
    if slot is None:
        slot = next((i for i, s in enumerate(strikes) if not s["on"]), 0)
    if bearing is None:
        bearing = float(LRNG.uniform(0.0, 360.0))
    if distance is None:
        distance = float(LRNG.uniform(*spec["dist"]))
    anc = lightning_anchor(bearing, distance, slot)

    if strikes[slot]["on"]:
        _end_strike(slot)
    bolt = -1
    if spec["bolts"]:
        bolt = slot * 4 + int(LRNG.choice(spec["bolts"]))
        bolt_nodes[bolt].position.set(0.0, 0.0, 0.0)

    # 2-4 return strokes down the same channel, 55-160 ms apart, each weaker.
    n = int(LRNG.integers(spec["strokes"][0], spec["strokes"][1] + 1))
    t, st = 0.0, [(0.0, 1.0)]
    for _ in range(n - 1):
        t += float(LRNG.uniform(0.055, 0.130))
        st.append((t, float(LRNG.uniform(0.60, 0.80))))
    strikes[slot].update(on=True, t=0.0, dur=t + 0.24, bolt=bolt, strokes=st,
                         emis=spec["emis"], lux=spec["lux"], dist=distance)
    for pl, y in zip(flash_lights[slot], (spec["mid"], spec["top"])):
        pl.position.set(anc.position.x, y, anc.position.z)
    return distance / 343.0


def _end_strike(slot):
    s = strikes[slot]
    s["on"] = False
    if s["bolt"] >= 0:
        bolt_mats[s["bolt"]].emissive_intensity = 0.0
        bolt_mats[s["bolt"]].needs_update()
        bolt_nodes[s["bolt"]].position.set(0.0, BOLT_PARK, 0.0)
        s["bolt"] = -1
    for pl in flash_lights[slot]:
        pl.intensity = 0.0
        pl.position.set(0.0, -20000.0, 0.0)   # out of every cluster cell


def _stroke_shape(u, tau=0.050):
    """One return stroke: a dim stepped leader, a spike, an exponential tail.

    The leader is 5% of the peak, not 50%: on an ACES curve anything above ~3
    is white, so a leader that is only one stop down from the stroke is not a
    leader at all -- it is the same white line arriving early.
    """
    if u < 0.0:
        return 0.0
    if u < 0.030:
        return 0.05 * (u / 0.030)
    if u < 0.046:
        return 0.05 + 0.95 * (u - 0.030) / 0.016
    return math.exp(-(u - 0.046) / tau)


def update_lightning(dt):
    """Schedule strikes in a storm, then advance whatever is burning."""
    global _next_strike, flash_level, _ae_flashing
    stormy = weather["coverage"] > 0.85 and weather["rain"] > 0.30
    if stormy:
        _next_strike -= dt
        if _next_strike <= 0.0:
            fire_strike(str(LRNG.choice(("sheet", "sheet", "intra", "mid", "mid", "hero"))))
            _next_strike = float(LRNG.uniform(5.0, 10.0))
    else:
        _next_strike = min(_next_strike, 4.0)

    flash_level = 0.0
    for i, s in enumerate(strikes):
        if not s["on"]:
            continue
        s["t"] += dt
        env = 0.0
        for t0, a in s["strokes"]:
            env = max(env, a * _stroke_shape(s["t"] - t0))
        if s["t"] > s["dur"] or env < 6e-3:
            _end_strike(i)
            continue
        env *= 0.72 + 0.28 * float(LRNG.random())     # a channel is never steady
        flash_level = max(flash_level, env)
        if s["bolt"] >= 0:
            # The CHANNEL fades faster than the flash does: the plasma cools in
            # microseconds while the lit cloud keeps glowing, and squaring is
            # also the only way to get a channel that visibly dims instead of
            # sitting at pure white until it snaps off (radiance 40 and
            # radiance 4000 tone-map to the same pixel).
            m = bolt_mats[s["bolt"]]
            m.emissive_intensity = s["emis"] * env * env
            m.needs_update()
        # Inverse square, no window (the backend's atten is 1/d^decay), so the
        # intensity has to carry the distance itself.
        inten = s["lux"] * env * s["dist"] ** 2
        for pl in flash_lights[i]:
            pl.intensity = inten

    hot = flash_level > 0.02
    if hot != _ae_flashing:
        _ae_flashing = hot
        renderer.set_auto_exposure_range(AE_FLASH_MIN if hot else AE_RANGE[0], AE_RANGE[1])


# --------------------------------------------------------------------------- #
#  The filming drone: a camera vehicle, not a quadrotor sim. The attitude is
#  kinematic -- it leans the way the acceleration says it must be leaning for
#  the path it is on -- which is all the eye reads at 15 m.
# --------------------------------------------------------------------------- #
DRONE_SPAN = 0.45
DRONE_ARM = DRONE_SPAN * 0.5 * 0.7071
DRONE_G = 9.81
# The eye sits on the gimbal, forward of and below the hull, so an FPV shot can
# never see the machine it is flying on.
DRONE_EYE = np.array([0.0, -0.090, 0.300])

drone = tp.Group()
drone.rotation.order = tp.RotationOrder.YXZ
drone.visible = False
scene.add(drone)

body_mat = tp.MeshStandardMaterial()
body_mat.color = 0x1b1f26
body_mat.roughness = 0.42
body_mat.metalness = 0.25
trim_mat = tp.MeshStandardMaterial()
trim_mat.color = 0x343b45
trim_mat.roughness = 0.55
prop_mat = tp.MeshStandardMaterial()
prop_mat.color = 0x0c0e11
prop_mat.roughness = 0.60
prop_mat.side = tp.Side.Double
# The blur disc is the one translucent thing in the demo. A TRANSPARENT
# MeshBasicMaterial is routed to the post-TAA UI overlay pass on this backend
# (VulkanCoreScene.cpp:26, kSnapUiBlend) and would draw straight through the
# boat; a transparent MeshStandardMaterial stays in the traced scene and gets
# stochastic alpha (VulkanCoreScene.cpp:3386), which is what a prop disc wants.
disc_mat = tp.MeshStandardMaterial()
disc_mat.color = 0xa8b0bb
disc_mat.roughness = 0.45
disc_mat.transparent = True
disc_mat.opacity = 0.0
disc_mat.side = tp.Side.Double
lens_mat = tp.MeshStandardMaterial()
lens_mat.color = 0x07090c
lens_mat.roughness = 0.05
lens_mat.metalness = 0.2

_hull = tp.Mesh(tp.BoxGeometry(0.115, 0.052, 0.170), body_mat)
_hull.cast_shadow = True
drone.add(_hull)
_canopy = tp.Mesh(tp.SphereGeometry(0.055, 14, 10), trim_mat)
_canopy.scale.set(1.0, 0.52, 1.30)
_canopy.position.set(0.0, 0.026, 0.005)
drone.add(_canopy)
for _s in (1.0, -1.0):
    _bm = tp.Mesh(tp.BoxGeometry(DRONE_SPAN, 0.015, 0.021), trim_mat)
    _bm.rotation.y = _s * math.radians(45.0)
    _bm.cast_shadow = True
    drone.add(_bm)

props = []
for _sx, _sz in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
    _px, _pz = _sx * DRONE_ARM, _sz * DRONE_ARM
    _pod = tp.Mesh(tp.CylinderGeometry(0.017, 0.020, 0.032, 10, 1), body_mat)
    _pod.position.set(_px, 0.013, _pz)
    drone.add(_pod)
    _hub = tp.Group()
    _hub.position.set(_px, 0.035, _pz)
    drone.add(_hub)
    _bl = tp.Mesh(tp.BoxGeometry(0.126, 0.0024, 0.016), prop_mat)
    _bl.rotation.z = math.radians(9.0)      # a little pitch, so it reads as a blade
    _hub.add(_bl)
    _disc = tp.Mesh(tp.CylinderGeometry(0.064, 0.064, 0.0016, 22, 1), disc_mat)
    _disc.position.set(_px, 0.035, _pz)
    drone.add(_disc)
    props.append((_hub, 1.0 if _sx * _sz > 0 else -1.0))

led_mats = []
for _col, _x, _z in ((0xff1408, -1.02, 1.02), (0x18ff3c, 1.02, 1.02)):
    _m = tp.MeshStandardMaterial()
    _m.color = 0x0a0a0a
    _m.emissive = _col
    _m.emissive_intensity = 0.0
    _s = tp.Mesh(tp.SphereGeometry(0.012, 10, 8), _m)
    _s.position.set(_x * DRONE_ARM, 0.004, _z * DRONE_ARM)
    drone.add(_s)
    led_mats.append(_m)
strobe_mat = tp.MeshStandardMaterial()
strobe_mat.color = 0x0a0a0a
strobe_mat.emissive = 0xffffff
strobe_mat.emissive_intensity = 0.0
_st = tp.Mesh(tp.SphereGeometry(0.013, 10, 8), strobe_mat)
_st.position.set(0.0, -0.030, -0.030)
drone.add(_st)

gimbal = tp.Group()
gimbal.position.set(0.0, -0.038, 0.082)
drone.add(gimbal)
_gb = tp.Mesh(tp.SphereGeometry(0.027, 14, 10), body_mat)
_gb.scale.set(1.0, 1.0, 0.85)
gimbal.add(_gb)
_lens = tp.Mesh(tp.CylinderGeometry(0.014, 0.017, 0.018, 12, 1), lens_mat)
_lens.rotate_x(math.pi / 2)
_lens.position.set(0.0, 0.0, 0.022)
gimbal.add(_lens)

drone_state = {
    "pos": np.array([0.0, 6.0, 14.0]), "vel": np.zeros(3), "acc": np.zeros(3),
    "look": np.array([0.0, 3.0, 0.0]),
    "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
    "spin": 0.0, "spool": 0.0, "throttle": 0.5, "have": False,
}
drone_on = ("--drone" in sys.argv) or ("--fpv" in sys.argv)
drone_auto = True          # False = a script owns drone_set_pose (Phase 4)
view_mode = "fpv" if "--fpv" in sys.argv else "orbit"
drone.visible = drone_on
_drone_phase = 0.0
_night_level = 0.0


def _wrap_toward(a, b, k):
    d = (b - a + math.pi) % (2.0 * math.pi) - math.pi
    return a + d * k


def drone_set_pose(pos, look_at, dt=1.0 / 60.0):
    """Put the drone at `pos` looking at `look_at`. THE call a shot makes.

    Attitude is derived, never authored: the last few poses give a smoothed
    velocity and acceleration, and a machine that is accelerating forward must
    be nose-down by atan2(a_fwd, g) to be doing it. Yaw is the velocity heading
    once it is actually moving and the look-at bearing when it is not, so a
    hover does not spin on numerical noise.
    """
    d = drone_state
    p = np.asarray(pos, np.float64).copy()
    tgt = np.asarray(look_at, np.float64).copy()
    dt = max(float(dt), 1e-4)
    if not d["have"]:
        d["have"] = True
        d["vel"][:] = 0.0
        d["acc"][:] = 0.0
        to0 = tgt - p
        d["yaw"] = math.atan2(to0[0], to0[2])
    else:
        v = (p - d["pos"]) / dt
        vprev = d["vel"].copy()
        d["vel"] += (v - d["vel"]) * (1.0 - math.exp(-dt / 0.12))
        a = (d["vel"] - vprev) / dt
        d["acc"] += (a - d["acc"]) * (1.0 - math.exp(-dt / 0.22))
    d["pos"] = p
    d["look"] = tgt

    sp = float(np.linalg.norm(d["vel"][[0, 2]]))
    to = tgt - p
    yaw_t = _wrap_toward(math.atan2(to[0], to[2]),
                         math.atan2(d["vel"][0], d["vel"][2]) if sp > 1e-3 else
                         math.atan2(to[0], to[2]), smoothstep(0.6, 2.5, sp))
    d["yaw"] = _wrap_toward(d["yaw"], yaw_t, 1.0 - math.exp(-dt / 0.15))
    fwd = np.array([math.sin(d["yaw"]), 0.0, math.cos(d["yaw"])])
    lat = np.array([math.cos(d["yaw"]), 0.0, -math.sin(d["yaw"])])
    lim = math.radians(25.0)
    # +x nose-down, -z starboard-down: the signs that make it bank INTO the turn
    pt = max(-lim, min(lim, math.atan2(float(np.dot(d["acc"], fwd)), DRONE_G)))
    rl = max(-lim, min(lim, -math.atan2(float(np.dot(d["acc"], lat)), DRONE_G)))
    k = 1.0 - math.exp(-dt / 0.10)
    d["pitch"] += (pt - d["pitch"]) * k
    d["roll"] += (rl - d["roll"]) * k
    d["throttle"] = float(np.clip(0.5 + d["acc"][1] / 9.0, 0.0, 1.0))

    drone.position.set(float(p[0]), float(p[1]), float(p[2]))
    drone.rotation.set(d["pitch"], d["yaw"], d["roll"])
    # The gimbal holds the horizon and points where the shot points.
    dh = math.hypot(float(to[0]), float(to[2]))
    gp = math.atan2(-float(to[1]), max(dh, 1e-3))
    gimbal.rotation.set(gp - d["pitch"], 0.0, -d["roll"])


def drone_tick(dt):
    """Props, LEDs and the strobe. Runs whenever the machine is in the world."""
    d = drone_state
    d["spool"] += (1.0 - d["spool"]) * (1.0 - math.exp(-dt / 0.55))
    rate = (100.0 + 55.0 * d["throttle"]) * d["spool"]
    d["spin"] += rate * dt
    for hub, sgn in props:
        hub.rotation.y = sgn * d["spin"]
    op = 0.55 * smoothstep(12.0, 70.0, rate)
    if abs(op - disc_mat.opacity) > 0.02:
        disc_mat.opacity = op
        disc_mat.needs_update()
    lit = 10.0 + 70.0 * _night_level
    for m in led_mats:
        if abs(m.emissive_intensity - lit) > 0.5:
            m.emissive_intensity = lit
            m.needs_update()
    blink = 340.0 if (world_time % 1.05) < 0.065 else 0.0
    if blink != strobe_mat.emissive_intensity:
        strobe_mat.emissive_intensity = blink
        strobe_mat.needs_update()


def drone_demo_path(dt):
    """Key G / --drone: a lazy orbit round the boat so the machine can be seen."""
    global _drone_phase
    _drone_phase += dt * (2.0 * math.pi / 22.0)
    bx, bz, by = boat_state["x"], boat_state["z"], boat_state["y"]
    r = 13.0 + 2.5 * math.sin(_drone_phase * 0.7)
    h = 6.0 + 1.7 * math.sin(_drone_phase * 1.3 + 0.8)
    drone_set_pose((bx + r * math.sin(_drone_phase), by + h, bz + r * math.cos(_drone_phase)),
                   (bx, by + 4.2, bz), dt)


def drone_camera_apply():
    """FPV: the render camera IS the gimbal eye."""
    d = drone_state
    eye = d["pos"] + yxz_matrix(d["pitch"], d["yaw"], d["roll"]) @ DRONE_EYE
    camera.position.set(float(eye[0]), float(eye[1]), float(eye[2]))
    camera.look_at(float(d["look"][0]), float(d["look"][1]), float(d["look"][2]))


camera = tp.PerspectiveCamera(48, canvas.aspect(), 0.1, 4000)
# Start downsun of the boat, looking back toward the sun: that is the one
# viewpoint where the sails glow with transmitted light and the rig cuts
# shafts through the mist. Orbit away from it and the scene goes flat.
_sun_hz = cs0.sun_dir[[0, 2]]
_sun_hz = _sun_hz / max(float(np.linalg.norm(_sun_hz)), 1e-6)
camera.position.set(-_sun_hz[0] * 21.0 + 12.0, 8.0, -_sun_hz[1] * 21.0 - 6.0)

# --------------------------------------------------------------------------- #
#  Live state. Everything an imgui slider owns lives in one dict, because the
#  widgets are pure immediate mode: value in, (changed, value) out.
# --------------------------------------------------------------------------- #
knob = {
    "wind": weather["wind"],   # true wind speed, m/s -- driven by the weather
    "wind_from": 215.0,        # bearing the wind blows FROM, deg (0 = from +Z)
    "boom": 42.0,              # mainsheet: boom angle off the centreline, deg
    "jib": 30.0,               # jib sheet angle, deg
    "auto_trim": True,
    "autopilot": True,
    "course": 340.0,           # autopilot heading, deg
    "rig_eff": 2.40,           # flat-plate model is shy upwind; buy some back
    "cloth": 420.0,            # sailcloth areal density, g/m2
    "stiff": 1.0,
    "sea_follows": True,
    "wave_scale": weather["wave_scale"],
    "choppy": weather["choppy"],
    "foam": 1.0,
    "fog": weather["fog"],     # sigma_t of the air medium, 1/m
    "fog_h": weather["fog_h"],  # height falloff of the mist, m
    "fog_g": 0.72,             # Henyey-Greenstein: + = forward god rays
    "cloud": weather["coverage"],
}

START_X, START_Z, START_HDG = 0.0, 0.0, 340.0

boat_state = {
    "x": START_X, "z": START_Z, "y": 0.0, "vy": 0.0,
    "yaw": math.radians(START_HDG), "yaw_rate": 0.0,
    "u": 0.0, "sway": 0.0,
    "heel": 0.0, "heel_rate": 0.0,
    "wave_pitch": 0.0, "wave_roll": 0.0,
    "rudder": 0.0,
    "boom": math.radians(42.0), "jib": math.radians(30.0),
    "drive": 0.0, "side": 0.0, "awa": 0.0, "aws": 0.0,
}


def apply_ocean_wind():
    """Point the wave spectrum where the wind is actually blowing."""
    blow = math.radians(knob["wind_from"]) + math.pi
    dx, dz = math.sin(blow), math.cos(blow)
    ocean.set_wind(max(knob["wind"], 0.5) if knob["sea_follows"] else 9.0,
                   math.atan2(dz, dx))          # Ocean measures theta from +X


apply_ocean_wind()
ocean.params.choppiness = knob["choppy"]
ocean.params.wave_scale = knob["wave_scale"]

scene.set_fog_exp2(tp.Color(0.72, 0.74, 0.78), knob["fog"])
# Profile only: density MUST be passed as 0 here or it would override the
# scene fog and take the live slider out of the loop.
renderer.set_height_fog(density=0.0, base_y=0.0, falloff=knob["fog_h"], noise_amount=0.45)
renderer.fog_anisotropy = knob["fog_g"]
renderer.set_clouds(coverage=knob["cloud"], density=1.0, bottom_y=380.0, top_y=1050.0,
                    wind=tp.Vector3(6.0, 0.0, 3.0), evolve_speed=1.0)

gulls.bake_perches_blocking(islet)      # the rock and the gallery are landable
gulls.set_disturbance_source(boat)      # they scatter when she runs them down
print(f"islet: {gulls.perch_count()} perches on the rock")

set_pin_targets(boat_state["boom"], boat_state["jib"])
for s in SAILS:
    s.push_pins()

# ---- CUDA graph ------------------------------------------------------------ #
graph = None
if device.is_cuda:
    with wp.ScopedCapture(device) as cap:
        sail_launches(SAILS)
    graph = cap.graph


def step_sails(wind_local, grav_local):
    aero = 0.5 * RHO_AIR / max(knob["cloth"] * 1e-3, 0.02)   # g/m2 -> kg/m2
    st = knob["stiff"]
    for s in SAILS:
        s.push_uniforms(wind_local, grav_local, aero, 0.11, 0.022, st, 0.85 * st, 0.42 * st)
        s.push_pins()
    if graph is not None:
        wp.capture_launch(graph)
    else:
        sail_launches(SAILS)
    for s in SAILS:
        s.upload()


# --------------------------------------------------------------------------- #
#  One frame of sailing.
# --------------------------------------------------------------------------- #
def hull_half_width(t):
    """Half-beam at fore-aft fraction t (0 = bow, 1 = stern), for foam splats."""
    u = 2.0 * t - 1.0
    b = BEAM * 0.5
    return b * (1.0 - u * u) ** 0.6 if u <= 0.0 else b * (1.0 - 0.25 * u * u)


sim_time = 0.0
first_render_done = False


def step_boat(dt):
    global sim_time
    bs = boat_state
    sim_time += dt

    cy, sy = math.cos(bs["yaw"]), math.sin(bs["yaw"])
    fwd = np.array([sy, 0.0, cy])            # heading
    stb = np.array([cy, 0.0, -sy])           # lateral

    # --- wind ---------------------------------------------------------------
    blow = math.radians(knob["wind_from"]) + math.pi
    gust = 1.0 + 0.10 * math.sin(0.37 * sim_time) + 0.05 * math.sin(1.13 * sim_time + 1.7)
    true_wind = knob["wind"] * gust * np.array([math.sin(blow), 0.0, math.cos(blow)])
    boat_vel = fwd * bs["u"] + stb * bs["sway"]
    app_wind = true_wind - boat_vel

    pitch = bs["wave_pitch"] + 0.020 * (bs["u"] / max(HULL_SPEED, 0.1)) ** 2
    roll = bs["wave_roll"] + bs["heel"]
    rot = yxz_matrix(-pitch, bs["yaw"], roll)
    wind_local = rot.T @ app_wind
    # The storm blows 17 m/s and the cloth solver was tuned at 9: past ~16 the
    # free leech flogs hard enough that neighbouring normals cancel and the rig
    # loses its drive to numerical noise. Cap what the SAILS see (the ocean and
    # the gulls still get the real thing) -- a reefed boat is the honest reading
    # of that anyway.
    _aw = float(np.linalg.norm(wind_local))
    if _aw > SAIL_WIND_MAX:
        wind_local = wind_local * (SAIL_WIND_MAX / _aw)
    grav_local = rot.T @ np.array([0.0, -9.81, 0.0])

    # Apparent wind angle off the bow; positive means the wind is on the
    # starboard side, which is the tack the sail should be to port on.
    bs["awa"] = math.atan2(-wind_local[0], -wind_local[2])
    bs["aws"] = float(np.linalg.norm(app_wind))

    # --- trim ---------------------------------------------------------------
    if knob["auto_trim"]:
        # The old rule of thumb: let the boom out to about half the apparent
        # wind angle, and sheet the jib a little harder than the main.
        want = math.copysign(min(max(abs(bs["awa"]) * 0.5, math.radians(6.0)),
                                 math.radians(85.0)), bs["awa"] or 1.0)
        knob["boom"] = math.degrees(abs(want))
        knob["jib"] = math.degrees(abs(want) * 0.72)
        target_b, target_j = want, want * 0.72
    else:
        sgn = 1.0 if (bs["awa"] >= 0.0) else -1.0
        target_b = sgn * math.radians(knob["boom"])
        target_j = sgn * math.radians(knob["jib"])
    # The boom swings at a finite rate: sheets are winches, not teleports.
    for key, target in (("boom", target_b), ("jib", target_j)):
        d = target - bs[key]
        bs[key] += max(-1.6 * dt, min(1.6 * dt, d))

    set_pin_targets(bs["boom"], bs["jib"])
    boom_pivot.rotation.y = bs["boom"]

    # --- the sails, and what they do to the boat ----------------------------
    step_sails(wind_local.astype(np.float32), grav_local.astype(np.float32))
    f_local = mainsail.read_force() + jib.read_force()
    f_world = rot @ f_local
    bs["drive"] = float(np.dot(f_world, fwd)) * knob["rig_eff"]
    bs["side"] = float(np.dot(f_world, stb)) * knob["rig_eff"]

    # Surge: thrust against quadratic drag, plus the wave-making wall that a
    # displacement hull runs into at 1.34*sqrt(LWL).
    u = bs["u"]
    over = max(abs(u) - HULL_SPEED, 0.0)
    drag = DRAG_Q * u * abs(u) + math.copysign(DRAG_WAVE * over * over, u)
    turn_drag = 220.0 * abs(bs["yaw_rate"]) * u
    bs["u"] += (bs["drive"] - drag - turn_drag) / MASS * dt
    bs["u"] = max(-1.5, min(bs["u"], HULL_SPEED * 1.25))

    # Leeway: the keel eats most of the side force, the rest is drift.
    bs["sway"] += (bs["side"] * LEEWAY_FRAC
                   - KEEL_DAMP * bs["sway"] * (abs(u) + 1.0)) / MASS * dt

    # Heel: sail moment against the ballast's righting moment, lightly damped.
    # This is a real second-order system, which is why she rolls after a gust
    # instead of snapping to an angle.
    m_heel = bs["side"] * CE_HEIGHT * math.cos(bs["heel"])
    m_right = MASS * 9.81 * GM * math.sin(bs["heel"])
    bs["heel_rate"] += (m_heel - m_right - ROLL_DAMP * bs["heel_rate"]) / I_ROLL * dt
    bs["heel"] += bs["heel_rate"] * dt
    bs["heel"] = max(-1.05, min(bs["heel"], 1.05))

    # Yaw: rudder, plus the weather helm a heeled hull generates on its own.
    r_target = (TURN_GAIN * max(abs(u), 0.4) * bs["rudder"]
                - WEATHER_HELM * bs["heel"] * max(abs(u), 0.3))
    t_yaw = min(max(LOA / max(abs(u), 1.2), 1.5), 6.0)
    bs["yaw_rate"] += (r_target - bs["yaw_rate"]) * min(dt / t_yaw, 1.0)
    bs["yaw"] += bs["yaw_rate"] * dt
    rudder_pivot.rotation.y = bs["rudder"]

    bs["x"] += (sy * bs["u"] + cy * bs["sway"]) * dt
    bs["z"] += (cy * bs["u"] - sy * bs["sway"]) * dt

    # --- the sea under the hull ---------------------------------------------
    if first_render_done:
        def sample(dx, dz):
            # dx = starboard offset, dz = forward offset.
            return ocean.sample_height(bs["x"] + sy * dz + cy * dx,
                                       bs["z"] + cy * dz - sy * dx, BUOY_MASK)
        h_c = sample(0.0, 0.0)
        h_bow, h_stern = sample(0.0, LOA * 0.5), sample(0.0, -LOA * 0.5)
        h_port, h_stbd = sample(-BEAM * 0.5, 0.0), sample(BEAM * 0.5, 0.0)
        # A spring-damped follower, not the surface height itself: tracking the
        # water exactly reads as a yo-yo riding the crests.
        omega = 2.0 * math.pi * 0.85
        bs["vy"] += ((h_c - bs["y"]) * omega * omega - bs["vy"] * 2.0 * 0.7 * omega) * dt
        bs["y"] += bs["vy"] * dt
        # atan2 over the FULL baseline, not the half -- the samples straddle it.
        wp_ = math.atan2(h_bow - h_stern, LOA)
        wr_ = math.atan2(h_stbd - h_port, BEAM)
        a = 1.0 - math.exp(-2.0 * math.pi * dt)
        bs["wave_pitch"] += (wp_ - bs["wave_pitch"]) * a
        bs["wave_roll"] += (wr_ - bs["wave_roll"]) * a

    # Sink her to her marks: the design waterline is local y = 0, and the boot
    # stripe is cut just above it so the stripe stays out of the water.
    boat.position.set(bs["x"], bs["y"] - 0.02, bs["z"])
    # Pitch is negated: positive Euler.x tips +Z toward -Y, i.e. bow DOWN, and
    # a positive wave pitch means the bow is on a crest.
    boat.rotation.set(-pitch, bs["yaw"], roll)

    # --- wake ---------------------------------------------------------------
    ocean.clear_foam_disturbances()
    spd = abs(bs["u"])
    norm = min(spd / 3.0, 1.0)
    base = (0.14 + 0.52 * norm) * knob["foam"]
    if base > 0.02:
        for side in (-1.0, 1.0):
            for i in range(7):
                t = i / 6.0
                lz = LOA * 0.5 - LOA * t
                lx = side * (hull_half_width(t) + 0.25)
                ocean.add_foam_disturbance(bs["x"] + cy * lx + sy * lz,
                                           bs["z"] - sy * lx + cy * lz,
                                           0.85, base)
        # Bow wave and the quarter wave off the transom.
        ocean.add_foam_disturbance(bs["x"] + sy * (LOA * 0.5 + 0.4),
                                   bs["z"] + cy * (LOA * 0.5 + 0.4),
                                   1.5, min(base * 1.3, 1.0))
        if norm > 0.15:
            lz = -LOA * 0.5 - 1.1
            ocean.add_foam_disturbance(bs["x"] + sy * lz, bs["z"] + cy * lz,
                                       2.4, min(0.35 + 0.5 * norm, 1.0))

    # The vertex-density warp centre relocates the whole grid in world space,
    # so pointing it at the boat is what makes the ocean effectively endless.
    ocean.warp_toward(bs["x"], bs["z"], 0.30)


# --------------------------------------------------------------------------- #
#  Input and the panel
# --------------------------------------------------------------------------- #
_prev_keys = {}


def pressed(key):
    now = canvas.is_key_down(key)
    fired = now and not _prev_keys.get(key, False)
    _prev_keys[key] = now
    return fired


def handle_keys(dt):
    global day_speed, drone_on, view_mode
    bs = boat_state
    if ui is not None and ui.want_capture_keyboard:
        return
    steer = 0.0
    if canvas.is_key_down("A"):
        steer -= 1.0
    if canvas.is_key_down("D"):
        steer += 1.0
    if steer != 0.0:
        knob["autopilot"] = False
        bs["rudder"] = max(-RUDDER_MAX, min(RUDDER_MAX, bs["rudder"] + steer * 1.1 * dt))
    elif not knob["autopilot"]:
        bs["rudder"] -= max(-0.9 * dt, min(0.9 * dt, bs["rudder"]))
    if canvas.is_key_down("Q"):
        knob["auto_trim"] = False
        knob["boom"] = min(85.0, knob["boom"] + 28.0 * dt)
    if canvas.is_key_down("E"):
        knob["auto_trim"] = False
        knob["boom"] = max(0.0, knob["boom"] - 28.0 * dt)
    if canvas.is_key_down("Z"):
        knob["auto_trim"] = False
        knob["jib"] = min(85.0, knob["jib"] + 28.0 * dt)
    if canvas.is_key_down("C"):
        knob["auto_trim"] = False
        knob["jib"] = max(0.0, knob["jib"] - 28.0 * dt)
    if pressed("T"):
        knob["auto_trim"] = not knob["auto_trim"]
    if pressed("P"):
        knob["autopilot"] = not knob["autopilot"]
        knob["course"] = math.degrees(bs["yaw"]) % 360.0
    if pressed("R"):
        reset_boat()
    for i in range(len(ACTS)):
        if pressed(str(i + 1)):
            jump_to_act(i)
    if pressed("N"):
        wtarget["rain"] = 0.0 if wtarget["rain"] > 0.05 else 1.0
    if pressed("K"):
        day_speed = 0.0 if day_speed != 0.0 else 0.35   # 24 h in ~68 s
    if pressed("L"):
        # A hero fork, now, in front of wherever the camera happens to look.
        cam = camera.position
        fwd = camera.get_world_direction()
        fire_strike("hero", bearing=math.degrees(math.atan2(fwd.x, fwd.z))
                    + float(LRNG.uniform(-14.0, 14.0)), distance=780.0)
    if pressed("G"):
        drone_on = not drone_on
        drone.visible = drone_on            # structural, so only on a keypress
        if not drone_on:
            view_mode = "orbit"
        else:
            drone_state["have"] = False     # re-seed the kinematics on the way in
    if pressed("V") and drone_on:
        view_mode = "orbit" if view_mode == "fpv" else "fpv"


def reset_boat():
    bs = boat_state
    bs.update({"x": START_X, "z": START_Z, "y": 0.0, "vy": 0.0, "u": 0.0,
               "sway": 0.0, "yaw_rate": 0.0, "heel": 0.0, "heel_rate": 0.0,
               "rudder": 0.0})
    bs["yaw"] = math.radians(knob["course"])


def run_autopilot(dt):
    if not knob["autopilot"]:
        return
    bs = boat_state
    err = math.radians(knob["course"]) - bs["yaw"]
    err = (err + math.pi) % (2.0 * math.pi) - math.pi
    cmd = 1.5 * err - 3.2 * bs["yaw_rate"]
    cmd = max(-RUDDER_MAX, min(RUDDER_MAX, cmd))
    d = cmd - bs["rudder"]
    bs["rudder"] += max(-1.1 * dt, min(1.1 * dt, d))


# --------------------------------------------------------------------------- #
#  The world clock: sky, weather, lights, rain. Everything that is not the boat.
# --------------------------------------------------------------------------- #
world_time = 0.0
_applied = {"cloud": None, "fog_h": None, "wind": None,
            "wave": None, "choppy": None, "glow": None}


def _ramp(dt):
    """Chase the target weather. Seconds, not frames -- see RAMP_TAU."""
    global _rain_was_on, _bow_timer
    for k in WEATHER_KEYS:
        a = 1.0 - math.exp(-dt / RAMP_TAU[k])
        weather[k] += (wtarget[k] - weather[k]) * a
    # Wetness lags the rain badly on the way down: a deck wets in seconds and
    # dries in minutes, and that asymmetry is most of what sells "it rained".
    wet_target = min(weather["rain"] * 1.25, 1.0)
    tau = 6.0 if wet_target > weather["wetness"] else 45.0
    weather["wetness"] += (wet_target - weather["wetness"]) * (1.0 - math.exp(-dt / tau))

    # The rainbow is not a preset flag, it is a consequence: about 40 s of bow
    # after the rain stops, and only while the sun is low enough to put the
    # antisolar point above the sea.
    raining = weather["rain"] > 0.30
    if _rain_was_on and not raining:
        _bow_timer = 40.0
    _rain_was_on = raining
    if _bow_timer > 0.0:
        _bow_timer = max(0.0, _bow_timer - dt)
        wtarget["rainbow"] = 1.0 if _bow_timer > 0.0 else 0.0


def update_world(dt):
    global world_time, time_of_day, _since_bake
    world_time += dt
    _since_bake += dt
    if day_speed != 0.0:
        time_of_day = (time_of_day + day_speed * dt) % 24.0
    _ramp(dt)

    cs = celestial(time_of_day)
    apply_sky(cs)
    murk = smoothstep(0.30, 0.92, weather["coverage"]) * min(weather["cloud_density"], 1.0)

    # ---- sun: same atmosphere as the sky it hangs in -----------------------
    sun.position.set(*(cs.sun_dir * 1200.0))
    beta_r, beta_m, _ = scatter_coeffs(cs.sun_y, turbidity(weather))
    fex = extinction(beta_r, beta_m, np.array([max(cs.sun_y, 0.0)], np.float32))[0]
    max_c = float(fex.max())
    if max_c > 1e-4:
        # Normalised, or raw Beer-Lambert kills the low sun's energy along with
        # its hue; then softened toward warm white, or the whole boat goes pink.
        t = fex / max_c
        t = t + (np.array([1.0, 0.80, 0.62], np.float32) - t) * 0.35
        # ...and then further toward white as the sun climbs. The raw
        # transmittance ratio at 48 deg is still (1, 0.78, 0.50), which is a
        # sunset colour: the whole picture goes amber at noon and the eye reads
        # the mismatch against a blue sky instantly.
        t = t + (1.0 - t) * (0.55 * smoothstep(0.10, 0.55, cs.sun_y))
    else:
        t = np.array([1.0, 0.80, 0.62], np.float32)
    sun.color = tp.Color(float(t[0]), float(t[1]), float(t[2]))
    sun.intensity = (4.6 * smoothstep(-0.03, 0.12, cs.sun_y) * (0.35 + 0.65 * max_c)
                     * (1.0 - 0.88 * murk))

    moon.position.set(*(cs.moon_dir * 1200.0))
    moon.intensity = 0.42 * cs.moon_up * (1.0 - cs.daylight) * (1.0 - 0.85 * murk)

    # Stars only once the sun is well down, and never through a storm deck.
    renderer.starfield = 1.15 * (1.0 - smoothstep(-0.18, -0.05, cs.sun_y)) * (1.0 - murk)

    # ---- air: the fog is the sky's own horizon -----------------------------
    knob["fog"] = weather["fog"]
    scene.set_fog_exp2(horizon_color(cs, weather), knob["fog"])
    if _applied["fog_h"] is None or abs(weather["fog_h"] - _applied["fog_h"]) > 1.0:
        _applied["fog_h"] = knob["fog_h"] = weather["fog_h"]
        renderer.set_height_fog(density=0.0, base_y=0.0,
                                falloff=knob["fog_h"], noise_amount=0.45)

    if _applied["cloud"] is None or abs(weather["coverage"] - _applied["cloud"]) > 0.006:
        _applied["cloud"] = knob["cloud"] = weather["coverage"]
        renderer.set_clouds(coverage=weather["coverage"], density=weather["cloud_density"],
                            bottom_y=weather["deck_bottom"], top_y=weather["deck_top"],
                            wind=tp.Vector3(6.0, 0.0, 3.0),
                            evolve_speed=1.0 + 1.4 * murk)

    # ---- sea ---------------------------------------------------------------
    if _applied["wind"] is None or abs(weather["wind"] - knob["wind"]) > 0.05:
        knob["wind"] = weather["wind"]
        _applied["wind"] = weather["wind"]
        if knob["sea_follows"]:
            apply_ocean_wind()
    if _applied["wave"] is None or abs(weather["wave_scale"] - knob["wave_scale"]) > 0.01:
        _applied["wave"] = knob["wave_scale"] = weather["wave_scale"]
        ocean.params.wave_scale = knob["wave_scale"]
    if _applied["choppy"] is None or abs(weather["choppy"] - knob["choppy"]) > 0.01:
        _applied["choppy"] = knob["choppy"] = weather["choppy"]
        ocean.params.choppiness = knob["choppy"]

    # ---- the light on the rock ---------------------------------------------
    global _night_level
    night = _night_level = 1.0 - smoothstep(0.0, 0.30, cs.daylight)
    ang = 2.0 * math.pi * (world_time / BEAM_PERIOD)
    for k, (b, tg) in enumerate(zip(beams, beam_targets)):
        a = ang + k * math.pi
        tg.position.set(LANTERN[0] + 900.0 * math.sin(a), LANTERN[1] - 22.0,
                        LANTERN[2] + 900.0 * math.cos(a))
        b.intensity = BEAM_INTENSITY * night
    glow = night * (1.0 + 0.05 * math.sin(world_time * 9.0) * math.sin(world_time * 3.7))
    if _applied["glow"] is None or abs(glow - _applied["glow"]) > 0.015:
        _applied["glow"] = glow
        # Both of these are REAL light sources on Vulkan, and auto exposure
        # lifts the night by up to 3 EV -- the pre-cycle constants (lamp 46,
        # window 26) came back as a white-hot rock and one yellow smear along
        # the deck once the eye was allowed to open.
        lamp_mat.emissive_intensity = 3.0 + 19.0 * glow
        lamp_mat.needs_update()
        glass_mat.emissive_intensity = 9.0 * glow
        glass_mat.needs_update()
        on = 1.0 - smoothstep(0.35, 0.60, cs.daylight)
        for m in nav_mats:
            m.emissive_intensity = NAV_INTENSITY * on
            m.needs_update()

    # ---- rain --------------------------------------------------------------
    amount = weather["rain"]
    live = int(RAIN_CAP * min(max(amount, 0.0), 1.0))
    rain.set_live_count(live)
    if live > 0:
        cp = camera.position
        rain.set_follow_center(tp.Vector3(cp.x, cp.y, cp.z))
        rain.set_emitter_time(world_time, dt)
        # The quads are composited AFTER the upscalers and are NOT touched by
        # auto exposure, so their brightness has to be walked against whatever
        # exposure the scene is running at -- otherwise the rain that reads
        # right at noon is invisible at midnight.
        br = rain.billboard_repr
        br.intensity = RAIN_BB_BASE * (1.0 + 4.5 * (1.0 - cs.daylight) ** 2)
        dr = rain.density_repr
        fc = rain.follow_center
        dr.center = tp.Vector3(fc.x, 8.0, fc.z)
        dr.sigma_per_particle = 0.014
        dr.enabled = True
    else:
        rain.density_repr.enabled = False

    apply_wetness(weather["wetness"])
    update_lightning(dt)
    return cs


def jump_to_act(i):
    """Keys 1-5: the five acts, each with its own hour and its own weather."""
    global time_of_day
    t, name = ACTS[i]
    time_of_day = t
    set_weather(name)
    weather.update(PRESETS[name])          # a cut, not a dissolve
    weather["wetness"] = 1.0 if weather["rain"] > 0.5 else 0.0
    update_world(1e-4)
    apply_sky(celestial(time_of_day), force=True)


def draw_ui():
    global time_of_day, day_speed
    bs = boat_state
    tp.imgui.set_next_window_pos(12, 12)
    tp.imgui.set_next_window_size(322, 0)
    tp.imgui.begin("Helm & rig")

    kn = bs["u"] / 0.5144
    tp.imgui.text(f"boat   {kn:5.2f} kn    heel {math.degrees(bs['heel']):+5.1f} deg")
    tp.imgui.text(f"wind   {bs['aws'] / 0.5144:5.2f} kn apparent, "
                  f"{abs(math.degrees(bs['awa'])):5.1f} deg "
                  f"{'stbd' if bs['awa'] >= 0 else 'port'}")
    tp.imgui.text(f"drive  {bs['drive']:7.0f} N     side {bs['side']:+7.0f} N")
    tp.imgui.text(f"heading {math.degrees(bs['yaw']) % 360.0:5.1f} deg    "
                  f"{tp.imgui.get_framerate():.0f} fps")
    tp.imgui.separator()

    ch, knob["wind"] = tp.imgui.slider_float("wind (m/s)", knob["wind"], 0.0, 22.0)
    if ch and knob["sea_follows"]:
        apply_ocean_wind()
    ch, knob["wind_from"] = tp.imgui.slider_float("wind from (deg)", knob["wind_from"], 0.0, 360.0)
    if ch:
        apply_ocean_wind()
    _, knob["sea_follows"] = tp.imgui.checkbox("sea state follows wind", knob["sea_follows"])
    tp.imgui.separator()

    _, knob["auto_trim"] = tp.imgui.checkbox("auto-trim sheets", knob["auto_trim"])
    _, knob["boom"] = tp.imgui.slider_float("mainsheet (deg)", knob["boom"], 0.0, 85.0)
    _, knob["jib"] = tp.imgui.slider_float("jib sheet (deg)", knob["jib"], 0.0, 85.0)
    tp.imgui.separator()

    _, knob["autopilot"] = tp.imgui.checkbox("autopilot", knob["autopilot"])
    _, knob["course"] = tp.imgui.slider_float("course (deg)", knob["course"], 0.0, 360.0)
    _, bs["rudder"] = tp.imgui.slider_float("rudder (rad)", bs["rudder"],
                                            -RUDDER_MAX, RUDDER_MAX)
    tp.imgui.separator()

    _, knob["cloth"] = tp.imgui.slider_float("cloth (g/m2)", knob["cloth"], 60.0, 900.0)
    _, knob["stiff"] = tp.imgui.slider_float("cloth stiffness", knob["stiff"], 0.15, 1.0)
    _, knob["rig_eff"] = tp.imgui.slider_float("rig efficiency", knob["rig_eff"], 0.5, 3.0)
    tp.imgui.text("A/D steer  Q/E main  Z/C jib  T trim  P pilot  R reset")
    tp.imgui.text("1-5 acts  N rain  K time-lapse  L strike  G drone  V FPV")
    tp.imgui.end()

    tp.imgui.set_next_window_pos(12, 470)
    tp.imgui.set_next_window_size(322, 0)
    tp.imgui.begin("Sea & sky")
    _cs = celestial(time_of_day)
    tp.imgui.text(f"{int(time_of_day):02d}:{int(time_of_day % 1.0 * 60):02d}   "
                  f"sun {_cs.sun_elev_deg:+5.1f} deg   {weather_name}")
    ch, time_of_day = tp.imgui.slider_float("time of day", time_of_day, 0.0, 24.0)
    if ch:
        apply_sky(celestial(time_of_day), force=True)
    _, day_speed_on = tp.imgui.checkbox("time-lapse (K)", day_speed != 0.0)
    day_speed = 0.35 if day_speed_on else 0.0
    for i, name in enumerate(("mist", "clear", "overcast", "storm", "clearing", "night")):
        if i:
            tp.imgui.same_line()
        if tp.imgui.button(name):
            set_weather(name)
    ch, wtarget["rain"] = tp.imgui.slider_float("rain", wtarget["rain"], 0.0, 1.0)
    tp.imgui.text(f"rain {int(rain.live_count / 1000)}k live   "
                  f"wet {weather['wetness']:.2f}   bow {weather['rainbow']:.2f}")
    tp.imgui.separator()
    ch, knob["wave_scale"] = tp.imgui.slider_float("wave scale", knob["wave_scale"], 0.0, 2.0)
    if ch:
        weather["wave_scale"] = wtarget["wave_scale"] = knob["wave_scale"]
        ocean.params.wave_scale = knob["wave_scale"]
    ch, knob["choppy"] = tp.imgui.slider_float("choppiness", knob["choppy"], 0.0, 1.0)
    if ch:
        weather["choppy"] = wtarget["choppy"] = knob["choppy"]
        ocean.params.choppiness = knob["choppy"]
    _, knob["foam"] = tp.imgui.slider_float("wake foam", knob["foam"], 0.0, 2.0)
    tp.imgui.separator()
    ch, knob["fog"] = tp.imgui.slider_float("fog density", knob["fog"], 0.0, 0.016)
    if ch:
        weather["fog"] = wtarget["fog"] = knob["fog"]
    ch, knob["fog_h"] = tp.imgui.slider_float("mist height (m)", knob["fog_h"], 8.0, 400.0)
    if ch:
        weather["fog_h"] = wtarget["fog_h"] = knob["fog_h"]
    ch, knob["fog_g"] = tp.imgui.slider_float("fog forward-scatter", knob["fog_g"], -0.9, 0.9)
    if ch:
        renderer.fog_anisotropy = knob["fog_g"]
    ch, knob["cloud"] = tp.imgui.slider_float("cloud cover", knob["cloud"], 0.0, 1.0)
    if ch:
        weather["coverage"] = wtarget["coverage"] = knob["cloud"]
    tp.imgui.separator()
    tp.imgui.text(f"gulls: {gulls.flying_count()} flying, "
                  f"{gulls.perched_count()} on the rock")
    if tp.imgui.button("flush the flock"):
        gulls.startle(tp.Vector3(boat_state["x"], 2.0, boat_state["z"]), 200.0, 1.0)
    tp.imgui.separator()
    if tp.imgui.button("strike (L)"):
        fire_strike("hero", distance=780.0)
    tp.imgui.same_line()
    if tp.imgui.button("sheet"):
        fire_strike("sheet")
    tp.imgui.same_line()
    if tp.imgui.button("fork"):
        fire_strike("mid")
    live = sum(1 for s in strikes if s["on"])
    tp.imgui.text(f"lightning: {live} burning   flash {flash_level:.2f}   "
                  f"next in {max(_next_strike, 0.0):4.1f} s")
    ch, dr = tp.imgui.checkbox("drone (G)", drone_on)
    if ch:
        _set_drone(dr)
    tp.imgui.same_line()
    ch, fp = tp.imgui.checkbox("FPV (V)", view_mode == "fpv")
    if ch and drone_on:
        _set_view("fpv" if fp else "orbit")
    tp.imgui.text("look toward the sun for shafts through the rig")
    tp.imgui.end()


def _set_drone(on):
    global drone_on, view_mode
    drone_on = bool(on)
    drone.visible = drone_on
    if not drone_on:
        view_mode = "orbit"
    else:
        drone_state["have"] = False


def _set_view(mode):
    global view_mode
    view_mode = mode


# --------------------------------------------------------------------------- #
#  Run
# --------------------------------------------------------------------------- #
controls = tp.OrbitControls(camera, canvas)
controls.enable_damping = True
controls.min_distance = 8.0
controls.max_distance = 220.0
controls.target = tp.Vector3(0.0, 3.0, 0.0)

clock = tp.Clock()
_cam_anchor = np.array([0.0, 0.0, 0.0])


def follow_camera():
    """Keep the boat under the orbit target without stealing the user's orbit."""
    global _cam_anchor
    p = np.array([boat_state["x"], 0.0, boat_state["z"]])
    d = p - _cam_anchor
    _cam_anchor = p
    camera.position.set(camera.position.x + d[0], camera.position.y,
                        camera.position.z + d[2])
    controls.target = tp.Vector3(float(p[0]), 3.0, float(p[2]))


def on_resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(on_resize)


def frame(dt):
    if not SHOT:
        handle_keys(dt)
    update_world(dt)
    run_autopilot(dt)
    step_boat(dt)
    blow = math.radians(knob["wind_from"]) + math.pi
    gulls.set_wind(math.sin(blow), math.cos(blow))   # perched birds face the wind
    gulls.update(dt)
    if drone_on:
        if drone_auto:
            drone_demo_path(dt)              # Phase 4 drives drone_set_pose itself
        drone_tick(dt)
    # FPV means the render camera IS the drone: the orbit rig has to keep its
    # hands off it, or controls.update() puts the camera back on its sphere.
    fpv = drone_on and view_mode == "fpv"
    # Offsetting the eye forward and below the hull is not enough on its own --
    # the gimbal swings off the airframe's axis and an arm walks back into the
    # corner of the frame. In FPV the machine is simply not in the world. The
    # visibility flip is a structural change, but it only ever happens on a
    # mode switch, which in the film is a CUT.
    want_vis = drone_on and not fpv
    if drone.visible != want_vis:
        drone.visible = want_vis
    if fpv:
        drone_camera_apply()
    else:
        follow_camera()


if SHOT:
    frames = int(round(SHOT_T * 60.0))
    # Where the still looks. `islet` is the demo's own framing; `sun` and
    # `anti` swing the camera round the boat so the shot is either INTO the
    # light (glitter path, sail translucency) or away from it, which is the
    # only half of the sky a rainbow -- or a moon opposite the sun -- lives in.
    FACE = cli_arg("--face", "islet", str)
    # A strike, on the clock, for the headless stress test. `--seq PREFIX`
    # writes the frames either side of it so the envelope can be LOOKED at
    # rather than asserted: -1 (before), 0 (leader), +2 (peak), +5, +10, +30.
    STRIKE_AT = cli_arg("--strike", -1.0, float)
    STRIKE_KIND = cli_arg("--strike-kind", "hero", str)
    STRIKE_BRG = cli_arg("--strike-bearing", 322.0, float)
    STRIKE_DIST = cli_arg("--strike-dist", 800.0, float)
    SEQ = cli_arg("--seq", "", str)
    strike_frame = int(round(STRIKE_AT * 60.0)) if STRIKE_AT > 0.0 else -1
    seq = {}
    if SEQ and strike_frame > 0:
        for o in (-1, 0, 2, 5, 10, 30, 75):
            seq[strike_frame + o] = f"{SEQ}_{'m' if o < 0 else 'p'}{abs(o):02d}.png"
    t_hot, n_hot, t_cold, n_cold = 0.0, 0, 0.0, 0
    for f in range(frames):
        if f == strike_frame:
            delay = fire_strike(STRIKE_KIND, bearing=STRIKE_BRG, distance=STRIKE_DIST)
            print(f"strike at frame {f} ({STRIKE_KIND}, {STRIKE_DIST:.0f} m), "
                  f"thunder in {delay:.1f} s")
        _t0 = time.perf_counter()
        frame(1.0 / 60.0)
        # Orbit slowly so the still is not shot from the launch pose.
        # Stand off her starboard quarter so the boat sits against the light.
        bx, bz = boat_state["x"], boat_state["z"]
        if FACE in ("sun", "anti"):
            cs_ = celestial(time_of_day)
            hz = cs_.sun_dir[[0, 2]]
            hz = hz / max(float(np.linalg.norm(hz)), 1e-6)
            if FACE == "anti":
                hz = -hz
            camera.position.set(bx - hz[0] * 30.0, 11.0, bz - hz[1] * 30.0)
            # Pitched well up: the primary bow's apex sits 42.5 deg above the
            # ANTISOLAR point, so at a low sun it is most of the way to the
            # zenith and a level camera frames only its shoulders.
            camera.look_at(bx + hz[0] * 260.0, 62.0, bz + hz[1] * 260.0)
        elif FACE == "drone":
            # Stand outside the drone's orbit and look in, so the boat is the
            # background and the machine is the subject.
            dp = drone_state["pos"]
            to = np.array([dp[0] - bx, 0.0, dp[2] - bz])
            to = to / max(float(np.linalg.norm(to)), 1e-6)
            camera.position.set(dp[0] + to[0] * 5.2 + 1.4, dp[1] + 1.15,
                                dp[2] + to[2] * 5.2 + 1.4)
            camera.look_at(float(dp[0]), float(dp[1]) - 0.12, float(dp[2]))
        elif FACE == "bow":
            # Bow-on from a little to port: the one angle where the jib is not
            # behind the main.
            cy, sy = math.cos(boat_state["yaw"]), math.sin(boat_state["yaw"])
            camera.position.set(bx + sy * 21.0 - cy * 8.0, 5.2, bz + cy * 21.0 + sy * 8.0)
            camera.look_at(bx, 6.4, bz)
        else:
            camera.position.set(bx + 27.0, 10.5, bz - 20.0)
            camera.look_at(bx * 0.35 + ISLET_X * 0.65, 7.0, bz * 0.35 + ISLET_Z * 0.65)
        if drone_on and view_mode == "fpv":
            drone_camera_apply()              # the shot framing does not own FPV
        renderer.render(scene, camera)        # keeps the temporal history honest
        first_render_done = True
        _dt_ms = (time.perf_counter() - _t0) * 1e3
        if f > 90:                            # skip the warm-up
            if flash_level > 0.02:
                t_hot += _dt_ms
                n_hot += 1
            else:
                t_cold += _dt_ms
                n_cold += 1
        if f in seq:
            try:
                from PIL import Image
                Image.fromarray(renderer.read_pixels()).save(seq[f])
            except Exception as exc:          # noqa: BLE001 - a still is not the demo
                print(f"  seq write failed: {exc}")
            print(f"  wrote {seq[f]}  (flash {flash_level:.3f})")
    # Let the accumulation settle on the final pose before the read-back.
    if not seq:
        for _ in range(24):
            renderer.render(scene, camera)
        out = cli_arg("--out", "warp_sailboat.png", str)
        renderer.save_frame(scene, camera, out)
        print(f"sailed {SHOT_T:.1f} s ({frames} frames), wrote {out}")
    if n_cold:
        print(f"frame time: {t_cold / n_cold:5.1f} ms over {n_cold} quiet frames"
              + (f", {t_hot / n_hot:5.1f} ms over {n_hot} lit frames" if n_hot else ""))
else:
    def animate():
        global first_render_done
        dt = min(clock.get_delta(), 0.1)      # a pause must not teleport the boat
        fpv = drone_on and view_mode == "fpv"
        if ui is not None:
            controls.enabled = (not ui.want_capture_mouse) and not fpv
        frame(dt)
        if not fpv:
            controls.update()
        renderer.render(scene, camera)
        first_render_done = True
        if ui is not None:
            ui.render(draw_ui)                # overlay: after render(), same frame

    canvas.animate(animate)
