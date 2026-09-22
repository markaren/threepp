"""A Range Rover driven over Bekker-Wong ground: mud, snow and a packed clay track.

WASD drives a PxVehicle2 Range Rover Evoque down three lanes of deformable
terrain (threepp.terrain_deform, the Sumner/O'Brien/Hodgins displacement grid
with a Bekker-Wong bearing and Janosi-Hanamoto shear). The lanes are 48 m long:
mud on one side, packed snow on the other, a firm clay strip down the middle to
spawn on and to compare against. The wheels cut ruts, the ruts stay, and the
soil decides how deep the car sits and how much of the throttle reaches the
ground.

    pip install warp-lang
    python warp_mudsnow_drive.py                  # Vulkan if available
    python warp_mudsnow_drive.py --gl             # OpenGL
    python warp_mudsnow_drive.py --vsync          # cap to the display
    python warp_mudsnow_drive.py --interop        # zero-copy lane meshes (see Strip)
    python warp_mudsnow_drive.py --bench          # ms/frame, vsync off, headless
    python warp_mudsnow_drive.py --shot --script mud     # one acceptance frame
    python warp_mudsnow_drive.py --shot --script cine    # the cinematic frame
    python warp_mudsnow_drive.py --cell 0.08 --width 6   # cheaper ground
    python warp_mudsnow_drive.py --no-lane-color         # flat lane materials
    python warp_mudsnow_drive.py --gravel                # the fourth lane: two-way grains
    python warp_mudsnow_drive.py --gravel --frames 600   # honest windowed fps, then exit
    python warp_mudsnow_drive.py --gravel --shot --script gravel_take   # launch mp4 + stills
    python warp_mudsnow_drive.py --film --interop        # the film (see below)
    python warp_mudsnow_drive.py --film --gravel --interop   # + the spread's take
    python warp_mudsnow_drive.py --film --dry            # its route, no renderer
    python warp_mudsnow_drive.py --film --takes mud,dig  # re-render two takes

    W / S     throttle / brake        R    gear (forward <-> reverse)
    A / D     steer                   SPACE handbrake
    V         driver POV              C    cinematic orbit
    BACKSPACE respawn                 T    reset the ground
    F         save a frame

`--script` drives a fixed input sequence headless and saves the frame the
comparison needs: `mud` and `snow` are the same line at the same throttle down
either soft lane, `spin_mud` / `spin_snow` / `spin_clay` the same full-throttle
launch, `lap`
a circle in mud twice, `crest` the car on the rise out of the spawn dip, `cine`
the look pass's own frame off the C camera, and `pan --yaw D` one heading of a
360 from the driver's seat. Repeatable is the point -- a hand-driven pass is not
the same line twice, and the whole claim is that only the SOIL differs.

`--gravel` adds a FOURTH lane whose ground is grains rather than a heightfield,
and the grains CARRY THE CAR: under each wheel a moving window of MLS-MPM soil
(threepp.granular_mpm.GranularPatches, four patches in one CUDA graph a frame)
takes the wheel's load, pose and spin from PhysX and hands back where the rim
bottom sits (the road override height), the gross thrust (the tyre's friction
ceiling) and the rolling resistance (add_force) -- one channel per force, no
Bekker anywhere on that lane. Opt-in, because it is a second solver: on an RTX
4070 at 1600x900, driving the lane in real time measures ~19 fps with the
default --aa 4 and ~24 fps with --aa 1 (--frames 900 --frames-log; the grains
are ~8-9 ms of every 16.7 ms physics step, the other lanes ~2 ms, and the
frame time holds steady over a full lane). Off the lane the grains are not
stepped at all. Its scripts are `spin_gravel`, `gravel`, `gravel_rest` and `gravel_take` (the
launch mp4 and stills). See "the gravel lane".

`--film` renders the demo's story as one ~52 s 60 fps mp4: seven scripted TAKES
over ONE continuously advancing sim (the ruts accumulate across the whole film;
nothing is ever reset), hard cuts between them, thirty frames discarded after
every cut for the temporal passes, and off-screen transits to move the car
between shots. With `--gravel` it is eight takes and ~59 s: the spread gets its
own launch, and the route detours to keep it virgin until the camera is there.
See "the film" at the bottom of this file.

The ground has a colour, and the wheels throw it
------------------------------------------------
Every surface in the yard is a WHITE material with a per-vertex `color`
attribute doing the work: the apron mixes dry dirt, wet earth and faded grass
over two scales of noise, darkens into a moisture halo along the soft lanes,
carries one worn wheel pair down its natural driving line, and drifts into the
snow field across a threshold that wanders instead of a mesh edge that does not.
Boulders and posts get a tint each, a value mottle across their facets and a
dirt line at the foot. The lanes carry it too -- a static attribute survives the
vertex-interop arm, which was the open question. Colours are written in sRGB and
decoded to linear by hand; see `to_linear`, and read it before adding a palette.

And when a contact patch scrubs past 2 m/s in a soft lane the wheel throws what
it is scrubbing loose: lit 3 cm clods in mud, powder in snow, host-owned
ballistics in numpy over two 1.5k ParticleFields. It is gated on the soil
model's own scrub, so it fires on a launch and on a slip-out and stays quiet on
a cruise.

The ground is a road, not a plane
---------------------------------
One analytic profile, `base(x, z)`, is the height of every drivable surface in
the yard: the three lanes get it as their DeformableTerrain `init_height` (which
the constructor copies into `grade`, the datum the road override already reads),
the apron mesh is drawn from it, and the apron's static PhysX trimesh collider
is cooked from the same vertices. Nothing samples a second opinion, so there is
no surface in the yard that disagrees with another about where the ground is.
Long swells along the lanes (27-61 m, +-0.4 m, grades to 5.7 %) are the crests
and dips; a crown falls a few cm from the clay strip to the shoulders; 3 cm of
8 m chatter is what the suspension hears. Nothing shorter than 4 m: the lanes
are a 5 cm grid read through a 40 cm wheel, and finer washboard aliases.

What a grade does to the suspension damper -- READ THIS BEFORE TUNING
--------------------------------------------------------------------
`PhysxVehicleBase::applyRoadOverrides` declares the overridden road plane
STATIONARY (`s.velocity = PxVec3(0)`), which was exactly true while the ground
was flat. On a grade it is not: the road under a wheel travelling at v rises at
v*slope, and PhysX's suspension damper reads the chassis climbing away from a
motionless plane as the suspension EXTENDING. The damper then subtracts
`damping * v * slope` -- 4500 * 7 m/s * 0.05 = ~1.6 kN per wheel -- from the
suspension force. Measured: cruising up the 5 % climb out of the spawn dip at
25 km/h the total normal load reads ~9.3 kN against a static 14.7 kN, and the
same amount MORE on the way down.

It is smooth, not a ringing loop (per-frame jounce is monotone, sd is a tenth of
the mean), and it is self-consistent -- z_eq is inverted from the load PhysX
reports, so the module's bearing at that depth still matches it and the panel's
Bekker-vs-PhysX column is unaffected. What it costs is absolute sinkage: the
car rides ~15 mm shallower uphill and deeper downhill than the soil says. The
fix is one line of C++ and a wider override API (pass the road's own vertical
velocity into `s.velocity` instead of zero); it is deliberately NOT worked
around here, because every Python-side workaround is a lie about where the
wheel is that the rut-carving imprint would then have to be told about too.

How the ground and the vehicle are coupled
------------------------------------------
PhysX finds ground with a raycast, and the deformable terrain does not live in
the PhysX scene. Mirroring the grid into a collider every frame is hopeless, and
holding the car up on terrain reaction forces alone makes a boat -- no sticky
tires, no suspension. So the split is:

  * PhysX keeps the WHOLE vehicle: suspension, tire model, sticky tires,
    substeps, load transfer. Untouched.
  * The road its suspension sees is Bekker's. Per wheel, per frame, we invert
    the Bekker bearing integral for the current wheel load W and hand the
    suspension a plane at `grade - z_eq(W)` (PhysxVehicle.set_road_override).
    The wheel rides IN the ground by exactly the sinkage the load dictates.
  * The friction the tire model gets is Mohr-Coulomb's:
    mu = (c*A(z_eq) + tan_phi*W) / W -- about 0.37 in mud, 0.40 in snow and
    0.75 on clay, against the 2.0 of the C++ demo's asphalt. That difference IS
    the driving experience.
  * The terrain module carves the ruts: the wheels are its collider spheres, fed
    the contact-patch slip velocity, so the prints follow the wheels and the
    module's own Janosi state is live for the panel.
  * EXCEPT on the gravel lane (--gravel), where none of the above runs: MPM
    grains replace the Bekker inversion, the Mohr-Coulomb mu, the motion
    resistance and the dig factor, wheel by wheel. The audit below is the
    Bekker lanes'; the gravel lane's own one-channel rule is at "the gravel
    lane".

Double-count audit -- what the soil model applies and what it only reports:

    bearing (module)   NOT applied. PhysX's suspension carries the car; the
                       module's Bekker force would be the same load a second
                       time. It is displayed instead, next to the PhysX
                       suspension load, and the two agreeing to a few percent
                       is the point of the middle panel.
    shear (module)     NOT applied. The tire model does traction, under the
                       Mohr-Coulomb mu ceiling we hand it -- which saturates at
                       the same c*A + tan_phi*W the module's Janosi law does.
                       Displayed as traction utilisation.
    motion resistance  APPLIED, via add_force_at_pos: Bekker's compaction
                       resistance (the work of making the rut) plus the
                       first-order bulldozing wedge, opposing the wheel's FULL
                       horizontal travel -- a sideways slide ploughs soil too.
                       PhysX has no notion of either, so there is nothing to
                       double-count: this is why soft ground bleeds speed and a
                       slide dies instead of sailing.
    slip sinkage       APPLIED, through the road override: a spinning wheel
                       excavates, so the effective sinkage is z_eq scaled by a
                       dig factor that follows slip with a ~0.6 s time constant.
                       Floor it in mud and the wheel digs to where the motion
                       resistance exceeds the traction ceiling -- stuck -- and
                       easing off lets the dig relax and the car creep out.
                       Standard terramechanics (slip sinkage), and the whole
                       mud-driving skill loop. NOTE: the Bekker-vs-PhysX panel
                       is an identity only while dig ~ 1 -- a dug-in wheel sits
                       deliberately below static equilibrium, and the panel
                       then shows the module's excess bearing at that depth.
    suction, grade     module-internal; they shape the grid the wheels then read.

Keep that split. Applying the module's bearing or shear on top of PhysX's would
double the load the car already carries.
"""
import atexit
import math
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import warp as wp

import threepp as tp
from threepp.terrain_deform import MATERIALS, DeformableTerrain
from threepp import granular_mpm as gm
from warp_common import (DensitySurface, cli_arg, encode_png_sequence, find_ffmpeg,
                         grain_detail_texture, parse_size,
                         standard_material, write_radiance_hdr)

try:
    from threepp.cuda_interop import VkInteropArray
except ImportError:                                 # build without the interop helper
    VkInteropArray = None

GL = "--gl" in sys.argv
INTEROP = "--interop" in sys.argv    # zero-copy publish; see the note on Strip
VSYNC = "--vsync" in sys.argv
SHOT = "--shot" in sys.argv
BENCH = "--bench" in sys.argv
FILM = "--film" in sys.argv          # the scripted film; see "the film" at the bottom
SHOT_TIME = cli_arg("--shot", 8.0, float)
WIDTH, HEIGHT = parse_size(cli_arg("--size", "1600x900", str))
CELL = cli_arg("--cell", 0.05, float)
LANE_LEN = cli_arg("--length", 48.0, float)
LANE_W = cli_arg("--width", 8.0, float)
OUT_DIR = cli_arg("--out-dir", ".", str)
AA = cli_arg("--aa", 4, int)
WARM = cli_arg("--warm", 60, int)
NOSAN = "--no-sanitize" in sys.argv
# The gravel patches' marching-cubes winding. wp.MarchingCubes winds INTO the
# density; --mc-flip reverses it so the outside is front-facing and the shipped
# -grad normals are already right. With it off the normals must go out negated
# (sign -1) for the raster's double-sided flip to land them outward -- which
# fixes the raster and nothing else: whatever traces the geometry (probe GI
# reads the stored normal, the lidar the winding) still sees it inside-out. On
# the spread the two render the same (at the renderer's noise floor); the flip
# is the one that is right for every consumer.
MC_FLIP = bool(cli_arg("--mc-flip", 1, int))
MC_SIGN = 1.0 if MC_FLIP else -1.0
LANE_COLOR = "--no-lane-color" not in sys.argv   # static colour under interop
# The fourth lane. Opt-in: its four MPM patches are a whole second solver in a
# frame that already spends ~25 ms elsewhere. See "the gravel lane" below.
GRAVEL = "--gravel" in sys.argv
# Honest windowed timing: `--frames N` runs the INTERACTIVE loop (the real
# frame(), wall-clock sim accumulator, UI, vsync off unless --vsync) for N frames
# on an autopilot (--auto-throttle as a speed hold at --auto-kmh; the keyboard
# is ignored), on the gravel lane with --gravel (the clay strip otherwise), then
# prints the mean frame time, the sim steps per frame and the real-time factor.
FRAMES = cli_arg("--frames", 0, int)
# Per-rendered-frame phase times to a CSV (written at exit): wall, sim steps, the
# steps' time, the gravel grains' own time and substeps, soup/strip publish,
# render. For "the frame rate falls over time" -- one number per phase per frame.
FRAMES_LOG = cli_arg("--frames-log", "", str)
# The gravel soup (four marching-cubes boxes, ~40 launches and a host copy) is
# rebuilt every SOUP_EVERY-th rendered frame in the interactive loop.
SOUP_EVERY = max(1, cli_arg("--soup-every", 2, int))
_pub_n = [0]
AUTO_THR = cli_arg("--auto-throttle", 0.30, float)
AUTO_KMH = cli_arg("--auto-kmh", 3.5, float)      # ...as a speed hold at this speed
TRACE = "--trace" in sys.argv        # scripted(): per-0.25 s state trace

DT = 1.0 / 60.0
R_WHEEL = 0.40                    # PhysX wheel radius, and the collider sphere
CHASSIS_MASS = 1500.0
REST_LOAD = CHASSIS_MASS * 9.81 / 4.0

# --- the yard -----------------------------------------------------------------
# 48 m of lane along x, three strips across z. The terrain module works in its
# own (x, y, height) frame and publishes y-up meshes as (x, y, z) -> (x, z, -y),
# so a lane's module-y is the NEGATIVE of its world z.

X0, X1 = -0.5 * LANE_LEN, 0.5 * LANE_LEN
NX = int(round((X1 - X0) / CELL)) + 1

CLAY_W = 4.0
# The gravel lane sits beyond the mud lane's post line, 5 m wide: past the posts
# so the yard still reads as three lanes and a spread, narrow because the MPM
# domain that covers it is priced by the square metre.
GRAVEL_W = 5.0
GRAVEL_Z = (0.5 * CLAY_W + LANE_W + 1.6, 0.5 * CLAY_W + LANE_W + 1.6 + GRAVEL_W)
LANES = ("mud", "clay", "snow") + (("gravel",) if GRAVEL else ())
LANE_Z = {"mud": (0.5 * CLAY_W, 0.5 * CLAY_W + LANE_W),
          "clay": (-0.5 * CLAY_W, 0.5 * CLAY_W),
          "snow": (-0.5 * CLAY_W - LANE_W, -0.5 * CLAY_W),
          "gravel": GRAVEL_Z}

# The presets are calibrated to a 5 cm Spot foot; the bearing numbers carry
# straight over to a 40 cm wheel (a 3.7 kN corner sits 22 mm into clay, 55 mm
# into mud, 64 mm into snow, which is what those materials do to a light 4x4).
# The one thing that does not carry over is the grade relaxation: at 0.15 /s a
# rut forgets it is a rut in about seven seconds, and a car comes back around
# for a second lap long before that. Slowed to 0.02 /s, the mud and snow ruts
# hold their datum for a minute, which is what makes the second pass ride in
# its own tracks instead of digging a fresh sinkage below them.
MATS = {"mud": replace(MATERIALS["mud"], grade_rate=0.02),
        "snow": replace(MATERIALS["snow"], grade_rate=0.02),
        "clay": MATERIALS["clay"],
        # Crushed aggregate, as a Bekker preset: the sand family with a graded
        # gravel's numbers. NOTHING drives on it any more -- the gravel lane is
        # carried by MLS-MPM grains (see "the gravel lane") -- it survives as
        # the material of the lane's DeformableTerrain, whose grid is only the
        # far field the strips draw, written from the grains every step.
        "gravel": replace(MATERIALS["sand"], n=1.1, k_c=1.2e3, k_phi=6.5e6,
                          cohesion=0.7e3, tan_phi=0.70, janosi_K=0.025,
                          tan_repose=0.78, compression=0.08, flow_rate=0.14,
                          grade_rate=0.40)}

SURROUND_R = 320.0                # to the rim of the world
APRON_R = 62.0                    # the drivable yard: flat-ISH, and it is a road


# --- the road ------------------------------------------------------------------
# ONE analytic profile, `base(x, z)` metres above the yard datum, and EVERY
# drivable surface samples it: the two soft lanes (as their DeformableTerrain
# init_height, which the constructor also copies into `grade`, so the road
# override already reads it), the clay strip, the apron mesh, and the apron's
# static trimesh collider. Two surfaces are never allowed to disagree about
# where the ground is -- that is the whole architecture of this section, and it
# is why the coupling loop needed no change at all: the suspension reads
# `grade - z_eq` exactly as before, and grade now has a road in it.
#
# Longitudinal swells run ALONG the lanes (x), because that is the axis you
# drive: three cosines anchored so x = SPAWN_X sits in the bottom of a dip
# (a car spawned on a grade rolls away before the script's first beat), the
# combined +-0.40 m reaching about 5.5 % where the crest comes out of the dip.
# Wavelengths 27-61 m: long enough to be a road and not a ramp, short enough
# that 48 m of lane carries a whole dip and a whole crest.


def vnoise(x, y, freq, seed, n=64):
    """Value noise on a wrapping n x n lattice: bilinear, smoothstep-faded."""
    g = np.random.default_rng(seed).random((n, n))
    fx, fy = np.asarray(x) * freq, np.asarray(y) * freq
    i0, j0 = np.floor(fx).astype(np.int64), np.floor(fy).astype(np.int64)
    tx, ty = fx - i0, fy - j0
    tx = tx * tx * (3.0 - 2.0 * tx)
    ty = ty * ty * (3.0 - 2.0 * ty)
    a = lambda di, dj: g[(i0 + di) % n, (j0 + dj) % n]                   # noqa: E731
    return ((1 - tx) * (1 - ty) * a(0, 0) + tx * (1 - ty) * a(1, 0)
            + (1 - tx) * ty * a(0, 1) + tx * ty * a(1, 1))


def fbm(x, y, freq, seed, octaves=4):
    v, amp, tot = 0.0, 1.0, 0.0
    for k in range(octaves):
        v = v + amp * vnoise(x, y, freq * 2.0 ** k, seed + 977 * k)
        tot += amp
        amp *= 0.5
    return v / tot


def smoothstep(a, b, v):
    t = np.clip((np.asarray(v, dtype=np.float64) - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def to_linear(c):
    """sRGB -> linear, and the one thing about vertex colour that will cost you
    an hour if nobody says it.

    A material's `color` is DECODED by the renderer: 0x2b241c is the sRGB
    number off a paint chip and reaches the shader as 0.023 linear. A `color`
    ATTRIBUTE is not -- it is taken as linear and used as it stands. Hand the
    attribute the same 0.169 the hex quoted and the ground comes back seven
    times too bright: the first pass of this phase turned a dusk yard into pale
    desert and the fog into haze over sand. Every palette in this file is
    written in sRGB, the numbers a human can picture, and decoded here.
    """
    c = np.clip(np.asarray(c, dtype=np.float64), 0.0, 1.0)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


SPAWN_X = -20.0
# (amplitude m, wavelength m). Written as -A*cos((x - SPAWN_X)*2pi/lam) so every
# term is at its MINIMUM and its derivative is zero at the spawn: the car starts
# in a dip, level, and has to climb out of it.
SWELL = ((0.20, 47.0), (0.12, 27.0), (0.08, 61.0))
CROWN = 0.08                      # crown fall, clay strip out to the shoulders
CROWN_HALF = 12.0
BUMP_AMP = 0.030                  # +-3 cm of road chatter...
BUMP_LAM = 8.0                    # ...at 8 m and 4 m. NOT shorter: the lanes are
                                  # a 5 cm grid read through a 40 cm wheel, and
                                  # anything under ~4 m aliases into noise the
                                  # suspension cannot tell from a step.
FADE_R = (APRON_R - 12.0, APRON_R)  # profile -> 0 before the hills take over


def base(x, z):
    """The road surface, metres above the yard datum, at world (x, z).

    Vectorised (numpy broadcasting) so the lane grids, the apron lattice and a
    single spawn point all go through the same function -- a second
    implementation is a second opinion about where the ground is, and one of
    them would be wrong.
    """
    x = np.asarray(x, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    h = np.zeros(np.broadcast(x, z).shape)
    for amp, lam in SWELL:
        h = h - amp * np.cos(2.0 * math.pi * (x - SPAWN_X) / lam)
    # Cross-fall: a crown on the clay strip falling a few cm to the shoulders,
    # so the lanes have camber and a stopped car leans off the centreline.
    h = h - CROWN * np.minimum(np.abs(z) / CROWN_HALF, 1.0) ** 2
    # Chatter, weighted by where you are: the clay strip is the graded track and
    # is nearly smooth, the soft lanes a little rougher, the apron beside them
    # roughest. Faded out past 26-46 m because the apron LATTICE coarsens with
    # radius and 8 m bumps on 2 m quads is aliasing, not texture.
    rough = ((0.30 + 0.70 * smoothstep(2.5, 13.0, np.abs(z)))
             * (1.0 - smoothstep(26.0, 46.0, np.hypot(x, z))))
    h = h + BUMP_AMP * rough * (2.0 * fbm(x, z, 1.0 / BUMP_LAM, 4441, 2) - 1.0)
    # And flat again before the perimeter, so the surround hills join the yard
    # at exactly the datum they joined it at when the yard was a plane.
    return h * (1.0 - smoothstep(FADE_R[0], FADE_R[1], np.hypot(x, z)))


def ride_y(x, z, over=1.05):
    """Spawn height at (x, z): the road plus the chassis' resting ride height."""
    return float(base(x, z)) + over


SPAWN_POS = tp.Vector3(SPAWN_X, ride_y(SPAWN_X, 0.0), 0.0)
SPAWN_ROT = tp.Quaternion().set_from_axis_angle(tp.Vector3(0, 1, 0), math.pi / 2)

# Motion-resistance coefficients. Bulldozing: the fraction of (frontal patch
# area x bearing pressure) pushed as the soil wedge -- around 460 N per wheel in
# mud and 830 N in snow at static sinkage. Compaction: Bekker's rut-making work
# (k_c + 2b*k_phi) * z^(n+1) / (n+1), scaled so that mud cruise costs ~0.18 g
# total -- enough that a slide dies and a coast bleeds off, while leaving
# ~0.18 g of traction margin to actually drive with (the mud ceiling is 0.37 g).
C_BULLDOZE = 0.5
C_COMPACT = 0.35
# The wedge cannot push harder than the material can bear before it FAILS --
# snow crushes at a few tens of kPa. Without this cap snow's n=1.6 makes the
# wedge pressure explode with depth (~120 kPa dug-in) and the lane is
# undrivable at its own static sinkage. Mud's cap is above anything reachable.
P_BULLDOZE_CAP = {"mud": 8.0e4, "snow": 3.5e4, "clay": 1.0e9}

# Slip sinkage: a spinning wheel excavates. The effective sinkage is
# z_eq * dig, where dig follows 1 + K_DIG * min(scrub / SCRUB_REF, 1) with a
# DIG_TAU time constant -- digging in is excavation, not instant, and so is
# climbing back out when you ease off. The driver is the contact patch's scrub
# SPEED, not the slip ratio: any launch from rest has ratio ~1, but a gentle
# launch scrubs ~1 m/s and barely digs, while wheelspin scrubs 10-25 m/s and
# buries the wheel. DIG_MAX caps the hole at a physical fraction of the wheel.
# At full spin in mud the resistance at DIG_MAX exceeds the mu*W traction
# ceiling: genuinely stuck until the dig relaxes. Clay barely digs.
SCRUB_DEAD = 1.5                  # m/s below which a wheel compacts, not digs --
                                  # without this dead zone the car can chicken-and-
                                  # egg itself stuck: drag keeps it slow, slow keeps
                                  # scrub up, scrub keeps the dig (and drag) up.
SCRUB_REF = 5.0                   # m/s of scrub PAST the dead zone for full rate
# (Gravel has no dig factor: its slip sinkage is the grains' own.)
K_DIG = {"mud": 1.6, "snow": 1.3, "clay": 0.2}
DIG_MAX = {"mud": 2.5, "snow": 2.2, "clay": 1.3}
DIG_TAU = 0.6                     # seconds
DRAG_CAP_FRAC = 1.5               # per-wheel drag cap, x wheel load (stability)

RIGID_Y = -0.03                   # the packed ground the lanes sit on
FALLBACK_MU = 1.1                 # see "handling" below


# --- Bekker inversion ----------------------------------------------------------


def bekker_table(mat, radius=R_WHEEL, cell=CELL, z_max=0.30, n=241):
    """W(z) and contact area A(z) for a sphere pressed z into flat ground.

    The SAME discrete sum the terrain module's imprint kernel evaluates -- one
    cell of the grid at a time, the chord half-width b floored at a cell, the
    pressure (k_c/b + k_phi) * z^n over the cell area -- so inverting this table
    lands the wheel at exactly the depth where the module's own bearing balances
    the load, and the panel's Bekker-vs-PhysX column agrees by construction
    rather than by fitting. Closed forms for the same integral do not: the
    b-floor and the grid quantisation are both worth several percent at these
    depths.
    """
    zs = np.linspace(0.0, z_max, n)
    m = int(math.ceil(radius / cell)) + 2
    g = np.arange(-m, m + 1) * cell
    dx, dy = np.meshgrid(g, g, indexing="ij")
    d2 = dx * dx + dy * dy
    load = np.zeros(n)
    area = np.zeros(n)
    for i, z in enumerate(zs):
        # Height of the sphere's lower surface over each cell, +BIG outside it.
        bottom = np.where(d2 < radius * radius,
                          (radius - z) - np.sqrt(np.maximum(radius * radius - d2, 0.0)),
                          1.0e9)
        zc = np.maximum(-bottom, 0.0)
        b = np.sqrt(np.maximum(zc * (2.0 * radius - zc), cell * cell))
        p = (mat.k_c / b + mat.k_phi) * np.power(zc, mat.n)
        load[i] = float((p * cell * cell).sum())
        area[i] = float((zc > 0.0).sum()) * cell * cell
    return zs, load, area


LUT = {k: bekker_table(m) for k, m in MATS.items()}


def sinkage(lane, w):
    """(z_eq, mu) for wheel load `w` on `lane`."""
    zs, load, area = LUT[lane]
    m = MATS[lane]
    z = float(np.interp(w, load, zs))
    a = float(np.interp(z, zs, area))
    return z, (m.cohesion * a + m.tan_phi * w) / max(w, 1.0)


def motion_resistance(lane, z):
    """Soil drag (N) for a wheel travelling at sinkage z: compaction + bulldozing.

    Compaction is Bekker's rut-making work per unit distance,
    (k_c + w_eff*k_phi) * z^(n+1) / (n+1) -- the energy that went into the print
    you can see behind the car. Bulldozing is the wedge ahead of the wheel, 2b
    wide and z deep, pushed at the bearing pressure the wheel is generating.
    Both grow superlinearly in z, which is what makes slip sinkage a trap: dig
    twice as deep and the drag roughly triples.
    """
    if z <= 1.0e-4:
        return 0.0
    m = MATS[lane]
    b = math.sqrt(max(z * (2.0 * R_WHEEL - z), CELL * CELL))
    p = min((m.k_c / b + m.k_phi) * z ** m.n, P_BULLDOZE_CAP[lane])
    compact = C_COMPACT * (m.k_c + 2.0 * b * m.k_phi) * z ** (m.n + 1.0) / (m.n + 1.0)
    return compact + C_BULLDOZE * z * 2.0 * b * p


# --- the ground ----------------------------------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("no CUDA device -- the terrain will run on the CPU and this will crawl")

terrain = {}
BASE_GRID = {}                    # the road, on each lane's own grid
BASE_T = {}                       # ...and on the device, for T
for name in LANES:
    z0, z1 = LANE_Z[name]
    ny = int(round((z1 - z0) / CELL)) + 1
    # The module works in (x, y, height) with y = -world z, so grid point (i, j)
    # is world (X0 + i*cell, z1 - j*cell). init_height goes into BOTH h and the
    # bearing datum `grade`, which is what the road override reads -- so this one
    # array is the whole of "the suspension drives on the profile".
    gx = (X0 + np.arange(NX) * CELL)[:, None]
    gz = (z1 - np.arange(ny) * CELL)[None, :]
    BASE_GRID[name] = np.ascontiguousarray(base(gx, gz)[None], np.float32)
    terrain[name] = DeformableTerrain((X0, -z1), CELL, (NX, ny), K=1,
                                      material=MATS[name], device=device, c_max=4,
                                      init_height=BASE_GRID[name])
    BASE_T[name] = torch.from_numpy(BASE_GRID[name]).to(device)
    print(f"  {name:5s} {terrain[name]}")
cells = sum(t.nx * t.ny for t in terrain.values())
print(f"  {cells:,} cells total")


def bilinear(t, flat, x, y):
    """Bilinear lookup into one of a terrain's [K, nx, ny] grids, [K, P] in/out.

    DeformableTerrain.heights() is this against `h`; the coupling wants the same
    gather against `grade`, because the road we hand the suspension is measured
    from the bearing datum, not from whatever the last pass left on the surface.
    """
    fx = ((x - t.origin_t[:, 0:1]) / t.cell).clamp(0.0, t.nx - 1.0001)
    fy = ((y - t.origin_t[:, 1:2]) / t.cell).clamp(0.0, t.ny - 1.0001)
    ix, iy = fx.long(), fy.long()
    tx, ty = fx - ix, fy - iy
    gat = lambda a, b: flat.gather(1, a * t.ny + b)                     # noqa: E731
    return ((1 - tx) * (1 - ty) * gat(ix, iy) + tx * (1 - ty) * gat(ix + 1, iy)
            + (1 - tx) * ty * gat(ix, iy + 1) + tx * ty * gat(ix + 1, iy + 1))


def lane_at(x, z):
    """The lane under world (x, z), or None -- and None PAST THE LANE ENDS in x
    too, or a car that leaves the far end of the strip would carry soft-ground
    handling (and its drag) onto the rigid ground forever."""
    if not (X0 <= x <= X0 + (NX - 1) * CELL):
        return None
    for name in LANES:
        lo, hi = LANE_Z[name]
        if lo <= z <= hi:
            return name
    return None


# --- the vehicle ---------------------------------------------------------------

world = tp.PhysxWorld()

# The deep catch plane: the floor of last resort, past the edge of the surround
# lattice or under anything the trimesh apron somehow misses. It used to sit at
# RIGID_Y, level with a dead-flat apron -- with a road in the apron it has to go
# BELOW the deepest dip (-0.48 m) or it would poke up through the crown of every
# hollow. What a wheel that leaves a lane actually lands on now is the apron
# trimesh (built with the surround, below), 3 cm under the lane datum: still a
# kerb rather than a cliff, and now a kerb that follows the road.
CATCH_DROP = 1.0
rigid = tp.Mesh(tp.BoxGeometry(400.0, 0.4, 400.0),
                standard_material(0x2a2a26, 0.95))
rigid.position.y = RIGID_Y - CATCH_DROP - 0.2
rigid.receive_shadow = True
world.add_static(rigid)

# tire_friction is the ceiling on the RIGID fallback and on anything the road
# override does not cover. The C++ demo's 2.0 out-grips the geometry -- a 1.65 m
# track under a CoM ~0.85 m up rolls at about 1 g, and 2.0 of mu reaches 2 --
# so this demo runs 1.1 everywhere. On the lanes it does not matter (the
# Mohr-Coulomb override is 0.37-0.75 and slides long before the car tips); on
# the clay strip and the fallback it is the difference between a scrub and a
# roll. Measured: full lock at 50 km/h on clay peaks at 2.2 degrees of tilt.
vehicle = tp.PhysxVehicle(world,
                          chassis_width=1.95, chassis_height=1.4, chassis_length=4.4,
                          chassis_mass=CHASSIS_MASS,
                          wheelbase=2.66, track_width=1.65, wheel_radius=R_WHEEL,
                          driven_wheels=[True, True, True, True],
                          max_throttle_torque=1500.0,
                          tire_friction=FALLBACK_MU,
                          longitudinal_stiffness=100_000.0,
                          suspension_travel=0.3, suspension_stiffness=35_000.0,
                          suspension_damping=4500.0, suspension_attachment_y=-0.4,
                          wheel_damping_rate=1.5,
                          position=SPAWN_POS, rotation=SPAWN_ROT)


def qrot(q, v):
    """Rotate the numpy vec3 `v` by a threepp Quaternion.

    Scalar arithmetic, not np.cross: this runs ~15 times a sim step, and on a
    3-vector numpy's per-call overhead was ~1 ms a step. Same formula, same
    operation order (v + w t + u x t, t = 2 u x v), so the same doubles."""
    x, y, z, w = q.x, q.y, q.z, q.w
    vx, vy, vz = float(v[0]), float(v[1]), float(v[2])
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return np.array([vx + w * tx + (y * tz - z * ty),
                     vy + w * ty + (z * tx - x * tz),
                     vz + w * tz + (x * ty - y * tx)])


# --- scene ---------------------------------------------------------------------

if not GL and not tp.vulkan_available():
    print("vulkan not available on this machine; falling back to OpenGL")
    GL = True

canvas = tp.Canvas("threepp x warp - mud & snow drive", width=WIDTH, height=HEIGHT,
                   antialiasing=AA, vsync=VSYNC, headless=SHOT or BENCH or FILM)
renderer = tp.GLRenderer(canvas) if GL else tp.VulkanRenderer(canvas)
if GL:
    renderer.shadow_map_enabled = True
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 0.95

scene = tp.Scene()
# The backdrop IS the haze: background and fog the same warm grey-brown, so the
# far hills dissolve into it and there is no horizon LINE anywhere to fight the
# yard for attention (warp_mudsnow_mpm.py does the same at 1/100 the scale).
# Global fog, not a per-material one -- some Vulkan paths ignore those.
HAZE = 0x3a3329
# The Vulkan pipeline auto-exposes this dusk up ~4x, so the authored haze
# reads as a pale warm sky on screen; GL has no auto-exposure and clears the
# background raw. Author the GL sky at the value the filmic frame actually
# shows (sampled off the Vulkan acceptance still), so the design -- ground
# dissolving into the haze, no horizon line anywhere -- survives the backend.
SKY = 0xe6d8c4 if GL else HAZE
scene.background = SKY
scene.set_fog(SKY, 55.0, 300.0)

# Low warm key raking ACROSS the lanes -- along z, while the lanes run along x.
# Berms are centimetres tall; under a high sun they vanish, and under a low one
# every rut wall throws a shadow as long as it is deep. 13 degrees of elevation
# is the compromise: shallow enough that a 3 cm berm draws a 12 cm shadow, steep
# enough that the car's own shadow does not lie down the lane and cover them.
# +z, not -z, and that sign is the whole shot: the acceptance cameras look
# ACROSS the lanes from the clay strip, so a key from behind them lights the
# face of every rut wall and hides its shadow, while a key from beyond throws
# each shadow toward the lens. Same relief, twice the contrast.
SUN_DIR = (-9.0, 8.5, 34.0)
scene.add(tp.HemisphereLight(0xc4cfdc, 0x39352e, 0.80))
sun = tp.DirectionalLight(0xffdca8, 4.0)
sun.position.set(*SUN_DIR)
sun.cast_shadow = True
sun.set_shadow_frustum(-40.0, 40.0, 40.0, -40.0)
sun.set_shadow_bias(-0.0008)
scene.add(sun)


def make_dusk_hdr(path, key_dir, W=512, HH=256):
    """A low-sun haze dome, written as a Radiance .hdr: what the paint reflects.

    The Evoque is the only glossy thing in the yard, and with no environment it
    reads as white plastic -- the body reflects nothing, so its panels have no
    edges. This is the rest of a low sun sky: a warm band low at the key's
    azimuth, cool dim blue overhead, and a dark warm ground half so horizontal
    panels do not mirror a sky that is underneath them.

    Deliberately carries NO sun DISC. The renderer extracts a disc out of an
    HDRI into an analytic light by default, and the scene's DirectionalLight is
    already the one sun -- a disc here would be a second one.
    """
    j = np.arange(HH).reshape(HH, 1)
    i = np.arange(W).reshape(1, W)
    theta = (j / HH) * math.pi
    phi = (i / W) * 2 * math.pi - math.pi
    y = np.broadcast_to(np.cos(theta), (HH, W))
    sin_t = np.sin(theta)
    d = np.stack([sin_t * np.cos(phi), y, sin_t * np.sin(phi)], axis=-1)
    up = np.clip(y, 0.0, 1.0)[..., None] ** 0.75
    sky = np.array([0.30, 0.26, 0.21]) * (1.0 - up) + np.array([0.07, 0.10, 0.17]) * up
    k = np.float32(key_dir)
    k = k / np.linalg.norm(k)
    ang = np.arccos(np.clip((d * k).sum(-1), -1.0, 1.0))
    sky = sky + np.exp(-(ang / math.radians(42.0)) ** 2)[..., None] \
        * np.array([1.05, 0.66, 0.33])
    sky = np.where((y < 0)[..., None], np.array([0.045, 0.040, 0.034]), sky)
    return write_radiance_hdr(path, sky)


try:
    import tempfile
    scene.environment = tp.RGBELoader().load(make_dusk_hdr(
        os.path.join(tempfile.gettempdir(), "threepp_mudsnow_drive_sky.hdr"), SUN_DIR))
except Exception as e:                              # noqa: BLE001 - garnish only
    print(f"  note: no environment ({e})")

# Mud eats the sunlight: rough AND weak-specular, so the raking key gives it a
# wet sheen in patches instead of a plastic gloss. Snow is the opposite --
# bright and near-lambertian, so what you read on it is pure shape. Clay is
# packed dry earth: lighter than mud and flatter than either.
mud_mat = tp.MeshPhysicalMaterial()
mud_mat.color = 0x40291a
mud_mat.roughness = 0.86
mud_mat.specular_intensity = 0.14
snow_mat = standard_material(0xdde5f0, 0.97)
clay_mat = standard_material(0x5a5142, 0.96)
# The gravel lane's strip is the far field of the MPM patches' heightfield (the
# stone the wheels have not reached, and the ruts they left); the soup over a
# live patch shares its albedo and its detail-layer stones, further down.
gravel_mat = standard_material(0x4c4740, 0.95)
LANE_MAT = {"mud": mud_mat, "snow": snow_mat, "clay": clay_mat,
            "gravel": gravel_mat}
# --lane-color: the same trick as the yard, on the LANES -- white materials and
# a static per-vertex albedo. Off by default; see the note in Strip.__init__.
LANE_ALBEDO = {"mud": (0.251, 0.161, 0.102), "snow": (0.867, 0.898, 0.941),
               "clay": (0.353, 0.318, 0.259), "gravel": (0.286, 0.267, 0.235)}
if LANE_COLOR:
    for _m in LANE_MAT.values():
        _m.color = 0xffffff
        _m.vertex_colors = True

# Micro-relief, as a slope bent into the published normal rather than geometry.
# A 5 cm grid over 48 m of lane is dead flat between the ruts, and dead flat is
# what makes the mud read as brown gloss paint under a raking key. This is the
# same trick warp_mudsnow_mpm.py's `grain` plays, and it costs four noise
# lookups a vertex on the strips a wheel actually touched. (freq 1/m, slope) --
# mud is gloopy, so large soft lumps; snow is crystalline, so fine and strong.
LANE_GRAIN = {"mud": (1.4, 0.22), "snow": (5.5, 0.13), "clay": (2.4, 0.19),
              "gravel": (3.6, 0.26)}


# A lane is drawn as a row of STRIPS, not one mesh, and the ground is published
# through the plain attribute path by default. The strips are a real choice:
# the host route republishes only the strips a wheel touched (Strip.publish),
# and 16 m of lane is the granularity that makes that cheap. `--interop` stays
# opt-in only because the host route runs everywhere -- the zero-copy path
# needs an NVIDIA driver on the other end of the export.
#
# (Historical note: both routes used to be capped by a VulkanRenderer defect --
# enable_vertex_interop left the RT-side GeometryDescs pointing at the freed
# buffers of the record it swaps out at arm time, which read as a >65k-vertex
# "ceiling" and a device loss once a lane had been driven on. Fixed in
# VulkanCoreGeometry.cpp (the desc republish in enableVertexInterop); merged
# 154k-vertex lanes now arm and drive fine, so MAX_INTEROP_VERTS is a strip
# granularity knob, not a correctness limit.)
#
# Strips share their boundary column so there is no seam, and take their
# normals from the FULL grid so the shading crosses the joins too.
MAX_INTEROP_VERTS = 60_000


@wp.kernel
def strip_surface(h: wp.array3d(dtype=float),
                  i0: int, ox: float, oy: float, cell: float, z0: float,
                  gfreq: float, gamp: float, gseed: int,
                  pos: wp.array(dtype=wp.vec3),
                  nrm: wp.array(dtype=wp.vec3)):
    """terrain_deform's `surface` kernel over one column range of the grid.

    Same y-up map ((x, y, height) -> (x, height, -y), a proper rotation, so the
    index buffer's winding still faces out) and the same central-difference
    normal -- but the difference reads the whole grid while the write is
    strip-local, which is what makes the seam between two strips invisible.

    Plus the grain (see LANE_GRAIN): the material's micro-slope, added to the
    grid's own slope before the normal is built, because grain IS extra slope.
    Shading only -- the position written is exactly the grid's, so the ground
    the camera sees is still the ground the wheels read.
    """
    s, j = wp.tid()
    i = i0 + s
    nx = h.shape[1]
    ny = h.shape[2]
    ia = wp.min(i + 1, nx - 1)
    ib = wp.max(i - 1, 0)
    ja = wp.min(j + 1, ny - 1)
    jb = wp.max(j - 1, 0)
    dhx = (h[0, ia, j] - h[0, ib, j]) / (float(ia - ib) * cell)
    dhy = (h[0, i, ja] - h[0, i, jb]) / (float(ja - jb) * cell)
    gx = (ox + float(i) * cell) * gfreq
    gy = (oy + float(j) * cell) * gfreq
    d = float(0.5)
    st = wp.uint32(gseed)
    dhx += gamp * (wp.noise(st, wp.vec2(gx + d, gy)) - wp.noise(st, wp.vec2(gx - d, gy)))
    dhy += gamp * (wp.noise(st, wp.vec2(gx, gy + d)) - wp.noise(st, wp.vec2(gx, gy - d)))
    # A second octave at 3.7x: one scale of lump is a pattern, two is a surface.
    bx = gx * 3.7
    by = gy * 3.7
    ga = gamp * 0.55
    dhx += ga * (wp.noise(st, wp.vec2(bx + d, by)) - wp.noise(st, wp.vec2(bx - d, by)))
    dhy += ga * (wp.noise(st, wp.vec2(bx, by + d)) - wp.noise(st, wp.vec2(bx, by - d)))
    n = wp.normalize(wp.vec3(-dhx, -dhy, 1.0))
    t = s * ny + j
    pos[t] = wp.vec3(ox + float(i) * cell, z0 + h[0, i, j], -(oy + float(j) * cell))
    nrm[t] = wp.vec3(n[0], n[2], -n[1])


class Strip:
    """One column range of a lane as a threepp mesh, published device-side."""

    def __init__(self, terrain, material, i0, ncol, base0, grain=(1.0, 0.0), name=""):
        self.t = terrain
        self.i0 = i0
        self.ncol = ncol
        self.gfreq, self.gamp = grain
        ny = terrain.ny
        self.n = ncol * ny
        self.ox, self.oy = (float(v) for v in terrain.origin_np[0])
        gi, gj = np.meshgrid(np.arange(i0, i0 + ncol), np.arange(ny), indexing="ij")
        # Seeded with the ROAD, not with the datum: the first render happens
        # before any strip is armed or published, and a flat lane for one frame
        # is a flat lane in a saved still.
        p0 = np.stack([self.ox + gi * terrain.cell,
                       terrain.z0 + base0[i0:i0 + ncol, :],
                       -(self.oy + gj * terrain.cell)], axis=-1)
        a = ((gi[:-1, :-1] - i0) * ny + gj[:-1, :-1]).ravel()
        faces = np.stack([a, a + ny, a + ny + 1, a, a + ny + 1, a + 1],
                         axis=1).astype(np.uint32)
        self.geometry = tp.BufferGeometry()
        self.geometry.set_attribute("position",
                                    np.ascontiguousarray(p0.reshape(-1, 3), np.float32))
        self.geometry.set_attribute("normal",
                                    np.tile(np.float32([0, 1, 0]), (self.n, 1)))
        if LANE_COLOR:
            # A lane is a Warp-owned position and normal buffer that the
            # renderer exports and CUDA writes, and whether a THIRD, static
            # attribute survives the arm -- which swaps the GeometryDesc out
            # from under the entry -- was an open question. It does: 8/8 strips
            # arm, the ruts still carve, and the tint is there. The colour is
            # the lane's own albedo with 20 % of value mottle at 3.5 m, which
            # is what stops eight metres of mud between two ruts being one
            # brown rectangle. `--no-lane-color` puts the flat materials back.
            gx, gz = p0[..., 0], p0[..., 2]
            c = np.asarray(LANE_ALBEDO[name], np.float64)
            v = (0.80 + 0.40 * fbm(gx, gz, 1.0 / 3.5, 131, 2))
            self.geometry.set_attribute(
                "color", np.ascontiguousarray(
                    to_linear(c[None, None, :] * v[..., None]).reshape(-1, 3),
                    np.float32))
        self.geometry.set_index(np.ascontiguousarray(faces.reshape(-1), np.uint32))
        self.mesh = tp.Mesh(self.geometry, material)
        self.mesh.frustum_culled = False      # the vertices move under the renderer
        self.mesh.receive_shadow = True
        self.vk = None
        self._host = None
        self.dirty = True
        self.x0 = self.ox + i0 * terrain.cell
        self.x1 = self.ox + (i0 + ncol - 1) * terrain.cell

    def launch(self, pos, nrm):
        wp.launch(strip_surface, dim=(self.ncol, self.t.ny), device=self.t._wp_device,
                  inputs=[self.t.h, self.i0, self.ox, self.oy, self.t.cell, self.t.z0,
                          self.gfreq, self.gamp, 0x5eed],
                  outputs=[pos, nrm])

    def arm(self, renderer):
        if not INTEROP or VkInteropArray is None                 or not hasattr(renderer, "enable_vertex_interop"):
            return False
        # Grid vertices keep their identity frame to frame, so the renderer's
        # per-vertex motion vectors are real motion: leave stable_correspondence
        # alone (a re-triangulating producer is the case that must opt out).
        h = renderer.enable_vertex_interop(self.mesh, self._on_frame,
                                          validate=not NOSAN)
        if h is None:
            return False
        (pos_h, pos_b), (nrm_h, nrm_b) = h
        try:
            self.vk = (VkInteropArray(pos_h, pos_b, wp.vec3, self.n, device),
                       VkInteropArray(nrm_h, nrm_b, wp.vec3, self.n, device))
        except Exception as e:                          # noqa: BLE001 - fall back
            print(f"  note: CUDA import failed ({e}) -- host route")
            renderer.disable_vertex_interop(self.mesh)
            self.vk = None
            return False
        self._on_frame()
        # Ordered teardown: the CUDA mappings go before the renderer frees the
        # memory they point at.
        atexit.register(self._release, renderer)
        return True

    def _on_frame(self):
        """Inside render(), post-fence and pre-record. The synchronize is
        MANDATORY: host ordering is all that sequences this write against the
        Vulkan frame that reads it."""
        self.launch(self.vk[0].array, self.vk[1].array)
        wp.synchronize_device(device)

    def publish(self):
        """Host route: device -> host -> geometry, only for a dirty strip.

        A height grid only changes where a wheel is, and a strip is 16 m of
        lane: republishing all eight every frame is 32 ms of copy for 4 m of
        change. Marked dirty under a wheel, plus one strip a frame round-robin
        so a slumping berm the car has already left still catches up.
        """
        if self.vk is not None or not self.dirty:
            return
        self.dirty = False
        if self._host is None:
            self._host = (wp.zeros(self.n, dtype=wp.vec3, device=self.t._wp_device),
                          wp.zeros(self.n, dtype=wp.vec3, device=self.t._wp_device))
        self.launch(*self._host)
        self.geometry.update_attribute("position", self._host[0].numpy())
        self.geometry.update_attribute("normal", self._host[1].numpy())

    def _release(self, renderer):
        if self.vk is None:
            return
        pair, self.vk = self.vk, None
        for a in pair:
            a.close()
        renderer.disable_vertex_interop(self.mesh)


def make_strips(name):
    t = terrain[name]
    per = max(2, min(t.nx, MAX_INTEROP_VERTS // t.ny))
    n_strip = int(math.ceil((t.nx - 1) / (per - 1)))
    per = int(math.ceil((t.nx - 1) / n_strip)) + 1
    out = []
    b0 = BASE_GRID[name][0]
    for k in range(n_strip):
        i0 = k * (per - 1)
        out.append(Strip(t, LANE_MAT[name], i0, min(per, t.nx - i0), b0,
                         LANE_GRAIN[name], name))
    return out


strips = [s for name in LANES for s in make_strips(name)]
for s in strips:
    scene.add(s.mesh)
print(f"  {len(strips)} lane strips, {max(s.n for s in strips):,} vertices each at most")
_rr = 0


def mark_dirty(hub):
    """Whichever strips a wheel is standing in, plus one more, need a republish."""
    global _rr
    for s in strips:
        for i in range(4):
            if s.x0 - 0.6 <= hub[i, 0] <= s.x1 + 0.6:
                s.dirty = True
                break
    strips[_rr % len(strips)].dirty = True
    _rr += 1

# --- the proving ground --------------------------------------------------------
# The lanes used to sit on a 400 m grey slab under a white sky, which is the one
# thing about this demo that was wrong on sight: no scale, no horizon, no world.
# What replaces it is ONE heightfield -- the ROAD (base(x, z)) out to the
# perimeter, because the visual ground and the ground the car drives on have to
# be the same ground, and beyond that rising into hills the fog takes away. One
# mesh, so there is no apron/surround seam to hide, and the lattice is stretched
# cubically so the metres beside the yard are fine and the quarter-kilometre out
# at the rim is cheap: 0.7 m spacing at the lanes, 7 m at the edge, 33k vertices
# for a 640 m yard.
#
# And the whole lattice is cooked as ONE static trimesh collider, so the apron
# the car drives on off the lanes is exactly the apron it is looking at -- the
# 0.7 m quads beside the lanes carry an 8 m bump to well under a millimetre, so
# the lane edge has no step in it that the profile put there.

rng = np.random.default_rng(7)


def yard_color(X, Z, ny, t):
    """Per-vertex colour of the lattice: what stops the yard being one brown.

    Absolute colours, not a modulation: the three materials are white and this
    function IS the ground's albedo. That is the only way the snow line can be
    a soft edge -- a multiplier on 0x2b241c cannot reach grey-white without
    going over 1 -- and it puts dirt, wet, grass, snow and rock in one palette
    that is mixed per vertex instead of cut per quad.

    Value first, hue second: this is dusk under fog, and a saturated ground at
    3 lux reads as a cartoon. The spread between the dry and the wet dirt is
    2.1x in VALUE and a few percent in hue.

      * two scales of patchiness -- 22 m fields of dry / wet / faded grass, 7 m
        of mottle inside them, the same value-noise family the road is made of;
      * a moisture halo a few metres out from the soft lanes' long edges, so
        the lanes belong to the ground instead of being decals on it;
      * a worn wheel pair down the apron's natural driving line (something has
        been here before, which is also the only cue that says this is a yard
        and not a field);
      * snow dusting carried by the same wandering snowline the mesh cut uses,
        broken up by noise and by how far up a face points, so the field's edge
        is drifted rather than sawn;
      * the hills going rockier and greyer with altitude before the fog takes
        them.
    """
    dry = np.array([0.238, 0.198, 0.146])       # sun-dried dirt, warm
    wet = np.array([0.105, 0.088, 0.072])       # churned wet earth, near-black
    grass = np.array([0.196, 0.190, 0.122])     # faded winter grass, olive
    snow = np.array([0.780, 0.816, 0.868])      # the lying-snow albedo
    rock = np.array([0.318, 0.282, 0.234])      # the hills' own brown
    scree = np.array([0.395, 0.396, 0.383])     # and their grey tops

    # 22 m fields, 7 m mottle. Two independent noises, so dry/wet and
    # dirt/grass are not the same map read twice.
    a = fbm(X + 13.0, Z - 5.0, 1.0 / 22.0, 71, 3)
    b = fbm(X - 61.0, Z + 29.0, 1.0 / 7.0, 83, 2)
    wetness = np.clip(1.35 * (a - 0.42) + 0.35 * (b - 0.5), 0.0, 1.0)
    grassy = smoothstep(0.52, 0.78, fbm(X + 200.0, Z + 90.0, 1.0 / 26.0, 97, 3))

    c = dry + (wet - dry) * wetness[..., None]
    c = c + (grass - c) * (0.75 * grassy)[..., None]
    c = c * (0.80 + 0.40 * b)[..., None]        # the 7 m mottle, as value

    # Moisture halo. The soft lanes wick outward: within ~4.5 m of a lane's
    # long edge the ground darkens and cools toward the wet end of the palette.
    halo = np.zeros_like(X)
    for z0, z1 in (LANE_Z["mud"], LANE_Z["snow"]):
        d = np.maximum(np.maximum(z0 - Z, Z - z1), 0.0)          # distance in z
        dx = np.maximum(np.maximum(X0 - X, X - X1), 0.0)
        halo = np.maximum(halo, (1.0 - smoothstep(0.0, 4.5, d))
                          * (1.0 - smoothstep(0.0, 7.0, dx)))
    halo *= 0.55 + 0.45 * b
    c = c + (wet * np.array([0.95, 1.0, 1.12]) - c) * (0.62 * halo)[..., None]

    # The worn track: one wheel pair down the apron beside the mud lane, on a
    # line that wanders the way a driven line does. Faint -- 0.78 of value --
    # because a hard black pair reads as paint.
    zc = LANE_Z["mud"][1] + 3.4 + 1.6 * np.sin(X / 17.0) + 0.7 * np.sin(X / 5.3)
    worn = np.zeros_like(X)
    for side in (-1.0, 1.0):
        worn = np.maximum(worn, 1.0 - smoothstep(0.10, 0.42, np.abs(Z - zc - side * 0.86)))
    worn *= (1.0 - smoothstep(30.0, 46.0, np.abs(X))) * (0.6 + 0.4 * b)
    c = c * (1.0 - 0.30 * worn)[..., None]

    # Snow. The SAME wandering threshold the mesh cut uses, so the two agree,
    # but read per vertex and softened over 3.5 m with the mottle eating into
    # it -- a snow field ends in drifts and bare patches, not on a line.
    snowline = LANE_Z["snow"][0] - 1.6 - 7.0 * fbm(X, Z, 0.035, 89, 3)
    cover = smoothstep(-0.8, 3.2, snowline - Z)
    cover = np.clip(cover * (0.55 + 0.75 * fbm(X + 7.0, Z - 3.0, 1.0 / 9.0, 101, 2))
                    * (1.0 - 0.35 * ny), 0.0, 1.0)               # ny: 0 flat, big = steep
    lit_snow = snow * (0.86 + 0.28 * b)[..., None]
    c = c + (lit_snow - c) * cover[..., None]

    # The hills: out of the yard the palette goes rock, and up the rock goes
    # scree-grey. The fog eats the far half of this; what it buys is the near
    # half not being the yard's own brown at 200 m.
    return c, rock, scree, snowline


def build_surround():
    """The lattice, the three meshes cut out of it, and its collider.

    The three materials are WHITE and a per-vertex `color` attribute carries
    the whole ground (see yard_color). This used to be impossible -- the note
    that stood here said the deferred path had no colour binding -- and it was
    half true: `set_attribute` was truncating indexed draw ranges and auto-LOD
    was pinning a partial one, so a colour attribute took the geometry off the
    path that draws it. Both are fixed (dev 5825467c, dfa43b60); the region
    cuts stay because they still carry the roughness split, but the snow line
    is now a blend across them rather than a seam between them.

    The collider is the WHOLE lattice, hills included, not just the yard: the
    alternative is a trimesh that stops at the perimeter and a metre of air
    beyond it, which is the invisible cliff this phase exists to remove.
    """
    n = 181
    u = np.linspace(-1.0, 1.0, n)
    ax = (0.20 * u + 0.80 * u ** 3) * SURROUND_R
    X, Z = np.meshgrid(ax, ax, indexing="ij")
    r = np.hypot(X, Z)

    # Inside the perimeter it is the road and nothing else, 3 cm under the lane
    # datum (the kerb the lanes are bedded into). base() has already faded
    # itself to zero by APRON_R and the hill ramp `t` is still zero there, so
    # the two never overlap and the join is exact rather than blended. Past it
    # the hills ramp in over 50 m (a cliff at the perimeter would read as a
    # wall) and the rim lifts above the eyeline so there is a silhouette in the
    # haze instead of an edge of the world.
    t = np.clip((r - APRON_R) / 50.0, 0.0, 1.0) ** 2
    hills = 26.0 * fbm(X, Z, 1.0 / 190.0, 11) - 9.5
    hills += 7.0 * fbm(X, Z, 1.0 / 52.0, 31) - 3.0
    h = RIGID_Y + base(X, Z) + t * np.maximum(hills, 0.0)
    h += t * 14.0 * np.clip((r - 130.0) / 180.0, 0.0, 1.0) ** 1.4

    dhx = np.gradient(h, ax, axis=0)
    dhz = np.gradient(h, ax, axis=1)
    # The apron is geometrically flat and would mirror the sky as one sheet, so
    # its RELIEF is a slope bent into the normal -- 20 m lumps, the biggest
    # thing this lattice can carry without aliasing. Same trick as the lanes'
    # grain, one scale up.
    dhx = dhx + 0.13 * (fbm(X + 91.0, Z, 0.05, 53, 3) - 0.5) * 2.0
    dhz = dhz + 0.13 * (fbm(X, Z + 47.0, 0.05, 59, 3) - 0.5) * 2.0
    nrm = np.stack([-dhx, np.ones_like(h), -dhz], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)

    # Colour, per vertex, over the SAME lattice: the yard's palette inside the
    # perimeter, ramped into rock and then scree over the hills by the same `t`
    # that lifts them, so there is one join and it is not visible.
    col, rock, scree, _ = yard_color(X, Z, 1.0 - nrm[..., 1], t)
    alt = smoothstep(2.0, 34.0, h - RIGID_Y)
    hue = fbm(X * 1.3 + 400.0, Z * 1.3, 1.0 / 60.0, 113, 3)
    hill_c = rock + (scree - rock) * alt[..., None]
    hill_c = hill_c * (0.72 + 0.56 * hue)[..., None]
    col = col + (hill_c - col) * (t ** 0.6)[..., None]

    pos = np.ascontiguousarray(np.stack([X, h, Z], axis=-1).reshape(-1, 3), np.float32)
    nrm = np.ascontiguousarray(nrm.reshape(-1, 3), np.float32)
    col = np.ascontiguousarray(to_linear(col).reshape(-1, 3), np.float32)
    a = (np.arange(n - 1)[:, None] * n + np.arange(n - 1)[None, :]).ravel()
    quads = np.stack([a, a + 1, a + n + 1, a, a + n + 1, a + n], axis=1)
    # Region tests on the quad's own centre, so a quad belongs to exactly one
    # mesh and the three meshes tile the lattice with no gap and no overlap.
    cx = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[:-1, 1:] + X[1:, 1:]).ravel()
    cz = 0.25 * (Z[:-1, :-1] + Z[1:, :-1] + Z[:-1, 1:] + Z[1:, 1:]).ravel()
    cr = np.hypot(cx, cz)
    yard = cr <= APRON_R + 7.0
    # A snow LINE, not a rectangle: the threshold wanders a few metres with the
    # ground so the field's edge reads as lying snow rather than a rug.
    snowline = LANE_Z["snow"][0] - 1.6 - 7.0 * fbm(cx, cz, 0.035, 89, 3)
    snow_ground = yard & (cz < snowline)

    def mesh_of(mask, material):
        idx = quads[mask]
        g = tp.BufferGeometry()
        g.set_attribute("position", pos)
        g.set_attribute("normal", nrm)
        g.set_attribute("color", col)
        g.set_index(np.ascontiguousarray(idx.ravel(), np.uint32))
        m = tp.Mesh(g, material)
        m.receive_shadow = True
        return m

    # None of the three casts: the sun's frustum is +-40 m (tight, so that a
    # 3 cm berm gets a shadow map texel), the hills start at 62 m, and the road
    # never self-shadows -- its steepest grade is 3.1 degrees against a key at
    # 13.6 degrees of elevation, so no crest can throw a shadow into its own
    # dip. cast_shadow here would buy a 33k-vertex depth pass and nothing else.
    #
    # The fourth mesh is never added to the scene: it is every quad of the
    # lattice, handed to PhysX as one cooked trimesh and then dropped. Same
    # vertices as the three visible ones, so the ground the wheels raycast into
    # is the ground to the last float.
    return (mesh_of(yard & ~snow_ground, apron_mat),
            mesh_of(snow_ground, field_snow_mat),
            mesh_of(~yard, hill_mat),
            mesh_of(np.ones(len(quads), bool), apron_mat))


# The yard floor is churned wet dirt: darker than the clay strip so the lanes
# read as lighter cuts into it rather than carpets laid on it, and dull enough
# that the raking key does not turn 12,000 m2 of it into a mirror. All three
# are WHITE now and carry only their roughness -- dirt, snow and rock all come
# out of the one `color` attribute (see yard_color), which is what lets the
# snow line drift across a mesh boundary instead of stopping at it.
apron_mat = standard_material(0xffffff, 0.96, vertex_colors=True)
field_snow_mat = standard_material(0xffffff, 0.97, vertex_colors=True)
hill_mat = standard_material(0xffffff, 0.98, vertex_colors=True)
*_surround, _ground_collider = build_surround()
for part in _surround:
    scene.add(part)
_t_cook = time.perf_counter()
world.add_static_trimesh(_ground_collider)
print(f"  apron collider: {len(_ground_collider.geometry.get_index()) // 3:,} triangles "
      f"cooked in {(time.perf_counter() - _t_cook) * 1000:.0f} ms")
del _ground_collider                    # cooked into PhysX; the mesh is dead weight


def boulder(radius, seed, squash=0.72):
    """A rock: an icosphere pushed around by three beat frequencies.

    IcosahedronGeometry is unindexed, so the displacement -- a pure function of
    the vertex position -- lands identically on every copy of a shared corner
    and the shell stays closed, while compute_vertex_normals then gives it the
    faceted read a broken rock has and a UV sphere never does.
    """
    g = tp.IcosahedronGeometry(1.0, 2)
    p = g.get_attribute("position")
    r = np.random.default_rng(seed)
    d = np.ones(len(p))
    for freq, amp in ((1.15, 0.30), (2.45, 0.15), (5.10, 0.065)):
        ph = r.uniform(0.0, 6.283, 3)
        d += amp * (np.sin(freq * p[:, 0] + ph[0]) * np.sin(freq * p[:, 1] + ph[1])
                    * np.sin(freq * p[:, 2] + ph[2]))
    q = p * d[:, None] * radius
    q[:, 1] *= squash
    g.set_attribute("position", np.ascontiguousarray(q, np.float32))
    g.compute_vertex_normals()
    return g


# Speed perception: without something standing still beside the lane, 70 km/h
# over a flat grid reads as 20. A weathered post line marks the perimeter and
# the lane shoulders; boulders scatter over the verges, dark and wet where they
# sit by the mud and snow-capped on the snow side. Built once -- adding or
# removing scene entries mid-drive rebuilds the Vulkan descriptor set and drops
# the TAA history.
#
# And built MERGED: ~170 pebbles and posts as 170 scene entries is 170 draws
# and 170 shadow-map draws for scenery that never moves relative to anything,
# and it measured as most of this pass's frame cost. Baked into one geometry a
# material, it is five draws.
#
# And coloured per vertex for the same reason the ground is: forty copies of
# one boulder under one material IS forty copies, and the eye finds that faster
# than it finds anything else in the frame. Each copy gets its own tint out of
# a spread, a value mottle across its facets, and a dirt line where it meets
# the ground -- the three cues that make a batch read as a scatter.
ROCK_ALBEDO = {"wet": (0.148, 0.126, 0.106), "dry": (0.300, 0.281, 0.244),
               "snow": (0.628, 0.663, 0.702)}
MUD_LINE = (0.088, 0.073, 0.058)          # what a foot in the wet yard looks like
post_mat = standard_material(0xffffff, 0.92, vertex_colors=True)
cap_mat = standard_material(0xffffff, 0.9, vertex_colors=True)
wet_rock = standard_material(0xffffff, 0.72, vertex_colors=True)
dry_rock = standard_material(0xffffff, 0.95, vertex_colors=True)
snow_rock = standard_material(0xffffff, 0.93, vertex_colors=True)


class Batch:
    """Accumulates transformed copies of small geometries into one mesh."""

    def __init__(self, material, cast=False):
        self.material = material
        self.cast = cast
        self.pos, self.nrm, self.idx, self.col, self.n = [], [], [], [], 0

    def add(self, geometry, offset, rot_y=0.0, tilt=(0.0, 0.0),
            color=(1.0, 1.0, 1.0), dirt=None, dirt_frac=0.45, mottle=0.0):
        """`color` is this copy's albedo, `dirt` the colour its foot is buried
        in -- blended over the lowest `dirt_frac` of the copy, which is the
        cheapest possible fix for props that look posted onto the ground rather
        than standing in it. `mottle` adds a per-vertex value jitter, which is
        what stops forty copies of one boulder reading as forty copies."""
        p = geometry.get_attribute("position")
        nm = geometry.get_attribute("normal")
        i = geometry.get_index()
        ca, sa = math.cos(rot_y), math.sin(rot_y)
        cx, sx = math.cos(tilt[0]), math.sin(tilt[0])
        cz, sz = math.cos(tilt[1]), math.sin(tilt[1])
        # Rz . Rx . Ry -- the same order three.js applies an object's XYZ Euler
        # plus the yaw, so a batched copy sits exactly where the loose Mesh did.
        m = (np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
             @ np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
             @ np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]]))
        q = p @ m.T + np.asarray(offset)
        self.pos.append(q)
        self.nrm.append(nm @ m.T)
        self.idx.append((np.arange(len(p)) if i is None else i) + self.n)
        c = np.tile(np.asarray(color, np.float64), (len(p), 1))
        if mottle > 0.0:
            # A hash of the LOCAL vertex, so a shared icosphere corner gets one
            # value and the shell does not crack along its seams.
            s = np.sin(p @ np.array([12.99, 78.23, 37.72])) * 43758.5453
            c = c * (1.0 + mottle * (2.0 * (s - np.floor(s)) - 1.0))[:, None]
        if dirt is not None:
            y0, y1 = q[:, 1].min(), q[:, 1].max()
            w = 1.0 - smoothstep(y0, y0 + max(dirt_frac * (y1 - y0), 1e-6), q[:, 1])
            c = c + (np.asarray(dirt, np.float64) - c) * (0.68 * w)[:, None]
        self.col.append(np.clip(c, 0.0, 1.0))
        self.n += len(p)

    def mesh(self):
        g = tp.BufferGeometry()
        g.set_attribute("position",
                        np.ascontiguousarray(np.concatenate(self.pos), np.float32))
        g.set_attribute("normal",
                        np.ascontiguousarray(np.concatenate(self.nrm), np.float32))
        g.set_attribute("color",
                        np.ascontiguousarray(to_linear(np.concatenate(self.col)),
                                             np.float32))
        g.set_index(np.ascontiguousarray(np.concatenate(self.idx).ravel(), np.uint32))
        m = tp.Mesh(g, self.material)
        m.cast_shadow = self.cast
        m.receive_shadow = True
        return m


# The shoulder posts are inside the +-40 m shadow frustum and cast; the 62 m
# perimeter ring is outside it and cannot, so it is a separate batch that skips
# the depth pass entirely.
near_posts = Batch(post_mat, cast=True)
far_posts = Batch(post_mat)
caps = Batch(cap_mat, cast=True)
rocks = {id(m): Batch(m, cast=True) for m in (wet_rock, dry_rock, snow_rock)}
rock_mats = {id(m): m for m in (wet_rock, dry_rock, snow_rock)}


def ground_y(x, z):
    """Where the apron surface is at (x, z) -- what a prop stands on."""
    return RIGID_Y + float(base(x, z))


# A SEPARATE stream for the tints. `rng` draws the placements, and a colour
# pass that borrows it would move every post and boulder in the yard -- which
# is a scene change wearing a colour change's clothes, and the acceptance
# stills would be comparing two different yards.
crng = np.random.default_rng(1907)


def tint(base_rgb, spread=0.22, hue=0.05):
    """One copy's albedo: a value draw plus a small independent hue wobble."""
    v = float(crng.normal(1.0, spread))
    return tuple(max(0.01, c * v * (1.0 + hue * float(crng.normal())))
                 for c in base_rgb)


for x in np.arange(X0 + 2.0, X1, 6.0):
    for z in (LANE_Z["mud"][1] + 0.9, LANE_Z["snow"][0] - 0.9):
        hgt = float(rng.uniform(1.25, 1.65))
        # Planted in the road, not at a datum the road left behind: a post line
        # that ignores the profile floats over the dips and buries itself in the
        # crests, and it is the one thing in frame that says where the ground is.
        y0 = ground_y(x, z)
        near_posts.add(tp.CylinderGeometry(0.06, 0.10, hgt, 8),
                       (float(x), 0.5 * hgt + y0, float(z)),
                       tilt=(float(rng.normal(0.0, 0.05)),      # nothing stands
                             float(rng.normal(0.0, 0.05))),     # straight
                       color=tint((0.152, 0.132, 0.115), 0.26, 0.07),
                       dirt=MUD_LINE, dirt_frac=0.30, mottle=0.06)
        if z < 0.0:                                             # snow on the tops
            cap = tp.SphereGeometry(0.10, 8, 6)
            cap.scale(1.0, 0.45, 1.0)
            caps.add(cap, (float(x), hgt + y0 - 0.02, float(z)),
                     color=tint((0.604, 0.635, 0.675), 0.10, 0.02))
# The perimeter itself, so the apron ends at something rather than at the fog:
# a ring of posts on the boundary the hills start from.
for a in np.arange(0.0, 2.0 * math.pi, 2.0 * math.pi / 96.0):
    hgt = float(rng.uniform(1.0, 1.4))
    px, pz = APRON_R * math.cos(a), APRON_R * math.sin(a)
    far_posts.add(tp.CylinderGeometry(0.06, 0.09, hgt, 6),
                  (float(px), 0.5 * hgt + ground_y(px, pz), float(pz)),
                  tilt=(0.0, float(rng.normal(0.0, 0.07))),
                  color=tint((0.150, 0.132, 0.118), 0.24, 0.06), mottle=0.05)

for k in range(72):
    r = float(rng.uniform(0.30, 1.15))
    side = 1.0 if rng.integers(0, 2) else -1.0
    z = float(rng.uniform(12.0, 46.0)) * side
    x = float(rng.uniform(X0 - 26.0, X1 + 26.0))
    mat = snow_rock if (z < 0.0 and rng.random() < 0.7) else         (wet_rock if abs(z) < 22.0 and side > 0 else dry_rock)
    key = "snow" if mat is snow_rock else ("wet" if mat is wet_rock else "dry")
    # Snow-capped rocks keep their tint tight (snow is snow); the dirt ones get
    # a wide value spread, and every one of them stands in the mud it sits in.
    _rot = float(rng.uniform(0.0, 6.28))
    _col = tint(ROCK_ALBEDO[key], 0.09 if key == "snow" else 0.26,
                0.02 if key == "snow" else 0.07)
    # The gravel lane's band starts inside the boulder field: drop the two or
    # three that would stand in it. Every draw above has already been taken, so
    # both rng streams stay bit-identical and no other rock in the yard moves
    # or changes colour because the fourth lane is on.
    if GRAVEL and GRAVEL_Z[0] - 1.0 < z < GRAVEL_Z[1] + 1.0:
        continue
    rocks[id(mat)].add(boulder(r, 400 + k), (x, r * 0.34 + ground_y(x, z), z),
                       rot_y=_rot, color=_col,
                       dirt=MUD_LINE if key != "snow" else (0.30, 0.31, 0.33),
                       dirt_frac=0.38, mottle=0.09 if key != "snow" else 0.04)
# NOTE: these boulders are scenery only. Making the big ones solid is one line
# (a hidden sphere proxy through world.add_static -- add_static reads
# Box/Sphere/Capsule geometry, not a displaced icosahedron), and it was tried
# and taken back out: four more static actors change the order PhysX
# accumulates its scene in, and the acceptance scripts' numbers moved in the
# third digit. This phase is the LOOK pass, and the sim has to print what it
# printed before it.
props = tp.Group()
for b in (near_posts, far_posts, caps, *rocks.values()):
    props.add(b.mesh())
scene.add(props)
print(f"  surround: {APRON_R:.0f} m apron, hills to {SURROUND_R:.0f} m")

# --- weather --------------------------------------------------------------------
# One ParticleField in Renderer ownership: the CPU writes a 64-byte emitter
# record a frame and never touches a flake, so the whole snowfall costs one
# scene entry and one indirect draw. Built ONCE and left in the scene (adding
# or removing entries mid-drive rebuilds the Vulkan descriptor set and throws
# away the TAA history), so there is no toggle -- parking it is
# set_live_count(0).
#
# The brief asked for snowfall over the SNOW LANE. That was tried first and it
# is worse: an 8 m wide emitter slab is a rectangular column of white specks
# with two visible vertical edges, and the drive camera sits on the clay strip
# looking straight through one of them. Snow falls out of a sky, not out of a
# box over one strip, so this follows the camera toroidally instead -- no edge
# anywhere, and which lane is the snow one is said by the GROUND, which is
# where this demo says everything else too.
# 3.6k flakes in a 28 m box, not 30k in a 56 m one: a flake has to subtend a
# pixel or two to read as a flake, and a wide box spends its whole budget on
# 40 m specks that composite into a star field. Close and sparse beats far and
# dense every time -- and dim beats bright: the quads are ADDITIVE against a
# dark dusk sky and outside auto-exposure, so white at 0.35 intensity reads as
# a bokeh star field that upstages the car (field report). Snow at dusk is
# sky-coloured, not white; it should be felt, not counted.
SNOW_CAP = 3_600
SNOW_HALF = 14.0
snowfall = None
if not GL:
    _sc = tp.ParticleField.Config()
    _sc.capacity = SNOW_CAP
    _sc.ownership = tp.ParticleField.Ownership.Renderer
    _sc.w_semantic = tp.ParticleField.WSemantic.Radius
    _sc.uniform_radius = 0.024
    snowfall = tp.ParticleField.create(_sc)
    snowfall.frustum_culled = False
    # Billboards only, unlit and additive. A lit mesh proxy would be the right
    # answer for flakes a metre from the lens and this camera is never there;
    # what it would cost is a G-buffer draw for every flake so that the two in
    # focus read as crystals.
    snowfall.set_billboard_repr(tp.Color(0.62, 0.64, 0.70), tp.Color(0.48, 0.51, 0.58),
                                0.22, 0.90)
    _sb = snowfall.billboard_repr
    _sb.lod_near = 0.0
    _sb.lod_fade = 0.0
    _sb.softness = 0.9
    _sb.fade_power = 0.0
    _sb.size_taper = 0.0
    _sb.bright_jitter = 0.35
    _sb.near_fade = 6.0                           # a flake at arm's length is a
                                                  # white plate across the frame
    _sb.stretch_seconds = 0.05
    _sb.stretch_max = 3.0
    _sb.stretch_max_screen = 0.02
    _sb.glow = 0.0
    _se = snowfall.emitter
    z_snowc = 0.5 * (LANE_Z["snow"][0] + LANE_Z["snow"][1])
    _se.spawn_center = tp.Vector3(0.0, 15.0, 0.0)
    _se.spawn_half_extent = tp.Vector3(SNOW_HALF, 0.5, SNOW_HALF)
    _se.velocity = tp.Vector3(0.0, -1.9, 0.0)     # a flake terminal-velocities slowly
    _se.speed_spread = 0.4
    _se.wind = tp.Vector3(0.55, 0.0, -0.2)
    _se.drift_amplitude = 0.28                    # the tumble, which is what says snow
    _se.drift_frequency = 0.5
    _se.drift_scale = 3.0
    _se.lifetime = 9.0
    _se.duty_cycle = 1.0
    _se.size = 0.024
    _se.size_jitter = 0.45
    _se.seed = 20260830
    _se.follow = True
    _se.follow_snap = 2.0
    snowfall.set_emitter(_se)
    snowfall.set_emitter_time(0.0, DT)
    scene.add(snowfall)

_weather_t = 0.0


def weather(dt, eye=None):
    """Advance the snowfall. Absolute time, so a still can seek to any t.

    The follow box is centred on the CAMERA -- weather that follows anything
    else is not weather -- and set_follow_center snaps, so the curtain does not
    swim under a moving chase cam.
    """
    global _weather_t
    if snowfall is None:
        return
    _weather_t += dt
    if eye is not None:
        snowfall.set_follow_center(tp.Vector3(*(float(v) for v in eye)))
    snowfall.set_emitter_time(_weather_t, dt)

# --- wheel spray ----------------------------------------------------------------
# What a wheel throws, and the reason it is an EVENT and not a fountain.
#
# The soil model already knows when a wheel is throwing soil: it is the same
# contact-patch SCRUB the Janosi integral is measured along and the slip
# sinkage follows. So the spray is gated on scrub past 2 m/s and on the wheel
# standing in a soft lane, which with traction control on means it fires when
# the car is launched, when it is provoked, and when the driver turns TC off --
# the three moments the demo is about. Cruising down the mud lane at 25 km/h
# throws almost nothing, which is correct and is the "modest" in the brief.
#
# Ownership.HostRing: 1.5k particles whose whole life is a dozen lines of numpy
# is not a thing the GPU should be told about. One (cap, 4) memcpy a frame,
# host_stable_slots so slot i is the same clod every frame -- which is what
# lets the radius taper with age (a HostRing field gets no emitter age fade)
# and what makes the billboard's velocity streak legal.
#
# The clods are the high-fidelity half: a lit icosahedron in the G-buffer near
# the camera, crossing over to an additive sprite at 7-11 m where a 3 cm clod
# is four pixels and the mesh is buying nothing. Snow spray is billboard only,
# stretched a little, in the snowfall's own sky-grey -- powder has no facets.
SPRAY_CAP = 1_500
SPRAY_SCRUB = 2.0                 # m/s of patch scrub below which nothing flies
SPRAY_G = np.array([0.0, -9.81, 0.0])


class Spray:
    """One host-owned pool of thrown soil. Ballistics in numpy, no GPU sim."""

    def __init__(self, name, drag, life, radius, seed):
        self.name = name
        self.drag = drag
        self.life_range = life
        self.radius = radius
        self.rng = np.random.default_rng(seed)
        self.field = None
        self.p = np.zeros((SPRAY_CAP, 3))
        self.v = np.zeros((SPRAY_CAP, 3))
        self.y0 = np.zeros(SPRAY_CAP)          # the ground it was thrown off
        self.age = np.full(SPRAY_CAP, 1e9)
        self.life = np.ones(SPRAY_CAP)
        self.r = np.zeros(SPRAY_CAP)
        self.buf = np.zeros((SPRAY_CAP, 4), np.float32)
        self.cursor = 0

    def spawn(self, n, pos, vel, spread, up, y0):
        """n particles at `pos` with `vel` plus an upward cone of half-angle
        `spread`. Round-robin over the pool: a cursor, not a free list, because
        stable slots mean the oldest particle is the one worth overwriting."""
        n = int(n)
        if n <= 0 or self.field is None:
            return
        i = (self.cursor + np.arange(n)) % SPRAY_CAP
        self.cursor = int((self.cursor + n) % SPRAY_CAP)
        j = self.rng.normal(0.0, spread, (n, 3))
        j[:, 1] = np.abs(j[:, 1]) + up
        self.p[i] = pos + self.rng.normal(0.0, 0.055, (n, 3))
        self.v[i] = vel + j
        self.y0[i] = y0
        self.age[i] = 0.0
        self.life[i] = self.rng.uniform(*self.life_range, n)
        self.r[i] = self.radius * self.rng.uniform(0.55, 1.45, n)

    def step(self, dt):
        if self.field is None:
            return
        live = self.age < self.life
        if not live.any():
            self.field.set_live_count(0)
            return
        self.age[live] += dt
        self.v[live] += SPRAY_G * dt
        self.v[live] *= math.exp(-self.drag * dt)
        self.p[live] += self.v[live] * dt
        # Two ways to die: it lands, or it runs out of life. Landing is tested
        # against the ground it was thrown off, not against base() -- 1.5k
        # analytic profile samples a frame to place a clod that is about to
        # vanish anyway is a lot of arithmetic for a centimetre.
        self.age[live & (self.p[:, 1] < self.y0 + 0.015)] = 1e9
        t = np.clip(self.age / np.maximum(self.life, 1e-6), 0.0, 1.0)
        # The taper IS the fade: no emitter, so age has to reach the picture
        # through the only per-particle channel a host field has, the radius.
        w = self.r * (1.0 - t) ** 0.45
        self.buf[:, :3] = self.p
        self.buf[:, 3] = np.where(self.age < self.life, w, -1.0)
        self.field.submit(self.buf, dt)

    def clear(self):
        self.age[:] = 1e9
        self.cursor = 0
        if self.field is not None:
            self.field.set_live_count(0)


def _spray_field(mesh_material=None, hot=(0.20, 0.14, 0.10), cool=(0.09, 0.07, 0.05),
                 intensity=1.0, radius=0.03, stretch=0.0, opacity=0.5):
    cfg = tp.ParticleField.Config()
    cfg.capacity = SPRAY_CAP
    cfg.ownership = tp.ParticleField.Ownership.HostRing
    cfg.w_semantic = tp.ParticleField.WSemantic.Radius
    cfg.uniform_radius = radius
    cfg.host_stable_slots = True
    f = tp.ParticleField.create(cfg)
    f.frustum_culled = False
    if mesh_material is not None:
        f.set_mesh_repr(tp.IcosahedronGeometry(radius, 0), mesh_material)
        mr = f.mesh_repr
        # The crossover has to sit BEYOND every camera the demo actually flies.
        # It used to be 11 m, and the chase camera sits at CHASE_DIST = 9.5 --
        # dead centre of the fade -- so interactive driving watched the clods
        # exactly where lit geometry dissolves into billboards, and they read
        # BRIGHT while the parked 2-5 m acceptance rigs (pure mesh) read dark.
        # The billboard is the reason: BillboardRepr composites AFTER the
        # upscalers and OUTSIDE auto-exposure, so its painted value is absolute
        # and does not come down with the AE-adapted dusk ground. Pushed out to
        # 28 m, every camera in this demo sees real lit geometry; 1.5k
        # subdiv-0 icosahedra is ~30k triangles worst case, which is nothing.
        mr.lod_far = 28.0
        mr.lod_fade = 6.0
        mr.near_cull = 0.9
    f.set_billboard_repr(tp.Color(*hot), tp.Color(*cool), intensity, 1.0)
    b = f.billboard_repr
    b.lod_near = 22.0 if mesh_material is not None else 0.0
    b.lod_fade = 6.0 if mesh_material is not None else 0.0
    b.alpha_over = True               # thrown soil OCCLUDES; additive mud glows
    b.opacity = opacity
    b.lit = True                      # in the same key the car is in
    b.lit_phase_g = 0.25
    b.lit_ambient = 0.14
    b.softness = 0.85
    b.fade_power = 1.4
    b.bright_jitter = 0.30
    b.glow = 0.0
    b.near_fade = 1.2                 # nothing ever splats on the lens
    if stretch > 0.0:
        b.stretch_seconds = stretch
        b.stretch_max = 2.5
        b.stretch_max_screen = 0.05
    f.set_live_count(0)
    scene.add(f)
    return f


# Wet clods are nearly black and take a specular from the key; powder is the
# snowfall's own sky-grey, because it IS the snowfall, briefly airborne.
clod_mat = standard_material(0x2a1e14, 0.62)
# Gravel throws STONES: drier, lighter, less drag than a wet clod and a shorter
# life, because a 2 cm chip leaves the tread and is down again in half a second.
grit_mat = standard_material(0x585044, 0.78)
sprays = {"mud": Spray("mud", 0.30, (0.70, 1.20), 0.030, 5150),
          "snow": Spray("snow", 1.60, (0.55, 1.00), 0.036, 5151)}
if GRAVEL:
    sprays["gravel"] = Spray("gravel", 0.16, (0.45, 0.85), 0.022, 5152)
if not GL:
    sprays["mud"].field = _spray_field(clod_mat, (0.115, 0.082, 0.058),
                                       (0.052, 0.038, 0.028), 1.0, 0.030,
                                       stretch=0.02, opacity=0.85)
    sprays["snow"].field = _spray_field(None, (0.66, 0.69, 0.75),
                                        (0.44, 0.47, 0.54), 1.0, 0.036,
                                        stretch=0.05, opacity=0.55)
    if GRAVEL:
        sprays["gravel"].field = _spray_field(grit_mat, (0.178, 0.161, 0.132),
                                              (0.088, 0.080, 0.068), 1.0, 0.022,
                                              stretch=0.015, opacity=0.90)


def spray_step(dt, hub):
    """Throw what the wheels are scrubbing loose, then fly what is airborne.

    Reads the coupling loop's own record of which lane each wheel is in and how
    fast its patch is scrubbing (hud, display state) plus the wheel's angular
    speed. Writes nothing the sim reads -- the whole of this function is
    downstream of the step that already happened.
    """
    if GL or hub is None:
        return
    q = vehicle.quaternion
    fwd = qrot(q, np.array([0.0, 0.0, 1.0]))
    right = qrot(q, np.array([1.0, 0.0, 0.0]))
    for i in range(4):
        lane = hud["lane"][i]
        if lane not in sprays:
            continue
        scrub = float(hud["scrub"][i])
        if scrub < SPRAY_SCRUB:
            continue
        s = sprays[lane]
        # Rate rises with scrub AND with load: an unloaded spinning wheel is
        # not excavating, it is waving at the sky (same gate the dig factor
        # uses). ~14 clods a frame per wheel flat out, which at 60 Hz and a
        # second of life is a pool of ~800 -- half the cap, four wheels in.
        rate = min((scrub - SPRAY_SCRUB) / 6.0, 1.0) * min(hud["w"][i] / REST_LOAD, 1.3)
        n = s.rng.poisson(14.0 * rate)
        if n <= 0:
            continue
        # The BACK of the contact patch, a wheel radius under the hub: soil
        # leaves where the tread lifts out of the ground, which is what makes a
        # rooster tail a tail and not a halo.
        rim = float(vehicle.wheel_angular_speed(i)) * R_WHEEL
        # The ground a clod lands back on. In gravel that is the TOP of the
        # loose bed, not the graded base the suspension is riding on.
        # (On gravel, hud["z"] is the sinkage under the undisturbed stone, so
        # this is the top of the loose layer there too.)
        gy = hub[i, 1] - R_WHEEL + hud["z"][i]
        pos = hub[i] - fwd * 0.34 + np.array([0.0, -0.30, 0.0])
        # Thrown against the direction of travel at a fraction of rim speed --
        # a clod leaves the tread, it is not launched from it -- with the
        # lateral component of the scrub carried across.
        vel = (-fwd * np.clip(rim, -14.0, 14.0) * 0.34
               + right * float(np.clip(hud["slip_lat"][i], -6.0, 6.0)) * 0.30)
        # Spread and lift by material: mud leaves in a fat wet arc, powder in a
        # wide soft cloud, gravel in a tight fast fan (stones do not billow).
        spread, up = {"mud": (0.9, 2.4), "snow": (1.25, 1.9),
                      "gravel": (0.70, 2.1)}[lane]
        s.spawn(n, pos, vel, spread, up, gy)
    for s in sprays.values():
        s.step(dt)

# --- the gravel lane: grains, not a heightfield ---------------------------------
# The fourth lane is the only one whose ground is not a Bekker grid. Under each
# wheel runs a moving window of MLS-MPM soil -- threepp.granular_mpm's
# GranularPatches: four 0.8 x 0.44 m patches, one per wheel, in ONE simulation
# and one CUDA graph per frame -- and the rest of the lane is a heightfield the
# patches seed from where they arrive and write the grains back into where they
# leave. The car is CARRIED by the grains, two-way, per wheel, per frame.
#
# ONE CHANNEL PER SOIL FORCE -- read this before believing anything on screen.
# Each force the grains put on a wheel reaches the car through exactly one
# channel, so nothing is counted twice:
#
#   bearing     W = PhysX's suspension force + the unsprung weight goes INTO the
#               patch; its wheel sinks until the grains carry W, and the rim
#               bottom comes back as the road override HEIGHT. The bearing
#               reaches the car ONLY through the suspension. The grains' vertical
#               force is displayed next to W, never add_force'd.
#   thrust      the grains' moment on the axle, T, is the gross thrust T / r. It
#               reaches the car ONLY through the PhysX tyre, as the friction
#               ceiling mu = (|T| / r) / W the road override hands it (EMA'd,
#               clamped to GR_MU_RANGE). A CEILING is what the soil can give,
#               and T / r is that only while the wheel slips (past GR_SLIP_SAT
#               the grains' shear is fully mobilised): so mu is sampled there
#               and HELD otherwise. Sampled at every slip, a coasting wheel's
#               T / r is its rolling resistance, ~0.2 W, and the car lost its
#               lateral grip the moment you lifted (measured: mu 0.30 on the
#               clamp and a slide off the lane in gravel_rest).
#   resistance  R = T / r - DP, DP = the grains' force on the wheel along its
#               rolling direction: the rut-making and the bulldozed wedge. It
#               reaches the car ONLY through add_force_at_pos, opposing the
#               wheel's travel -- the channel motion_resistance() uses on the
#               other lanes.
#
# Nothing on this lane comes from Bekker: no z_eq, no Mohr-Coulomb mu, no dig
# factor. Slip sinkage is emergent -- a spinning wheel throws grains out from
# under itself and the road override follows it down.
#
# What this route (phase 2, "Route A") cannot do: PhysX's own slip curve still
# shapes the thrust below saturation -- the grains set the ceiling, one frame
# late, and the tyre model decides how fast it is approached -- and the lateral
# soil force goes through that same tyre ceiling (a sideways plough gets no
# add_force here). Giving the grains the thrust directly needs a per-wheel
# torque input PhysxVehicle does not have yet.
#
# The patches run in a SHEARED frame: local y = world y - base(x, z) + GR_DEPTH.
# Their floor is flat (local 0) and the undisturbed gravel top is local
# GR_DEPTH, which is world base(x, z): the lane's graded base, the same profile
# every other surface in the yard reads, carried by the map rather than by a
# floor that would need 0.8 m of extra grains to follow the swells. The road
# override is a horizontal plane per wheel per frame on every lane, so a grade
# is a per-frame height change to PhysX anyway; the grains see the same thing
# (a 5.7 % grade tilts gravity 3.3 degrees against a 36-degree friction angle).
#
# An off-lane wheel's patch keeps following its wheel with the wheel HOVERING
# over it (kinematic, zero load, no spin) and its outputs ignored -- one config,
# one graph, and a patch that is already in place when the wheel comes back
# onto the stone. With no wheel on the lane at all the grains are not stepped.
#
# Look: the lane's own Strip mesh draws the far field (terrain["gravel"].h is
# written from the patches' heightfield every step, so interop and the host
# route both just work), and each patch's live grains are a marching-cubes soup
# on top. Under a live patch the strip is set to the grains' own top, sunk
# GR_SEAM below the soup and ramped back up to the true surface at the patch
# edge, so the soup covers it and the edge has no step or groove.

GR_H = 0.04                     # MPM cell: the phase-2 gate's shipped config
GR_DEPTH = 0.30                 # loose layer under the wheel, m
GR_PATCH = (0.8, 0.44)          # patch interior (x, z), m
GR_HALF_W = 0.12                # tyre half-width, m (a 240 mm tread)
GR_MASS, GR_DAMP = 45.0, 5000.0  # the wheel's vertical DOF: unsprung mass, near-critical damper
M_UNSPRUNG = 45.0               # W = suspension force + M_UNSPRUNG * g
GR_MU_RANGE = (0.30, 0.70)      # the tyre ceiling the grains may set
GR_MU0 = 0.54                   # ...before the first saturated sample (phase 1's saturation)
GR_SLIP_SAT = 0.25              # |slip| past which the gross thrust IS the ceiling...
GR_SCRUB_SAT = 0.5              # ...and the rim must also scrub this fast, m/s (a
                                # settling wheel's twitch at rest is not saturation)
GR_TAU = 0.10                   # s, EMA on mu and R (the grains' frame-to-frame scatter)
GR_NSUB_Q = 8                   # substeps rounded UP to a multiple: one CUDA graph each
GR_SEAM, GR_SEAM_RAMP = 0.025, 0.04   # strip sunk under a live patch, ramp from its edge
GR_LIFT = 0.25                  # an off-lane wheel hovers this far over its patch, m
GR_VTOP = 2.0                   # m/s: faster grains are airborne, not ground
GR_MAX_TRIS = 24_000            # per patch soup


def gravel_material():
    """warp_wheel_testbed.py's material() at E = 3 MPa: the phase-1 loose soil
    (Drucker-Prager 36 deg + a compaction cap) the phase-2 gate measured."""
    return gm.Material(E=3.0e6, nu=0.3, rho=1700.0, phi_deg=36.0, cohesion=0.0,
                       cap_p0=2.0e3, cap_lambda=0.025)


@wp.func
def gr_base(B: wp.array2d(dtype=float), x: float, z: float, bx0: float, bz1: float,
            cell: float) -> float:
    """base(x, z) off the gravel lane's own grid (i, j) -> (X0 + i*cell, z1 - j*cell)."""
    u = wp.clamp((x - bx0) / cell, 0.0, float(B.shape[0]) - 1.001)
    w = wp.clamp((bz1 - z) / cell, 0.0, float(B.shape[1]) - 1.001)
    i = int(u)
    j = int(w)
    fu = u - float(i)
    fw = w - float(j)
    return ((B[i, j] * (1.0 - fu) + B[i + 1, j] * fu) * (1.0 - fw)
            + (B[i, j + 1] * (1.0 - fu) + B[i + 1, j + 1] * fu) * fw)


@wp.kernel
def gr_particles(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3),
                 alive: wp.array(dtype=int), cap: int,
                 B: wp.array2d(dtype=float), bx0: float, bz1: float, cell: float,
                 depth: float, lx0: float, lx1: float, lz0: float, lz1: float,
                 pc: wp.array2d(dtype=int), marg: int, h: float, pd: float, vtop: float,
                 top: wp.array3d(dtype=float), out: wp.array(dtype=wp.vec3)):
    """Live grains -> world positions for the soup (sheared back onto the road,
    clipped to the lane), and each patch's column tops (local y) for the strip."""
    p = wp.tid()
    far = wp.vec3(0.0, -1.0e6, 0.0)
    if alive[p] == 0:
        out[p] = far
        return
    xp = x[p]
    if xp[0] < lx0 or xp[0] > lx1 or xp[2] < lz0 or xp[2] > lz1:
        out[p] = far
        return
    out[p] = wp.vec3(xp[0], xp[1] - depth + gr_base(B, xp[0], xp[2], bx0, bz1, cell), xp[2])
    if wp.length(v[p]) > vtop:
        return
    k = p // cap
    ci = int(wp.floor((xp[0] - float(pc[k, 0] + marg) * h) / h))
    cj = int(wp.floor((xp[2] - float(pc[k, 1] + marg) * h) / h))
    if ci >= 0 and cj >= 0 and ci < top.shape[1] and cj < top.shape[2]:
        wp.atomic_max(top, k, ci, cj, xp[1] + 0.5 * pd)


@wp.kernel
def gr_lane_h(H: wp.array2d(dtype=float), hox: float, hoz: float, hdx: float,
              B: wp.array2d(dtype=float), top: wp.array3d(dtype=float),
              pc: wp.array2d(dtype=int), marg: int, h: float, ix: int, iz: int,
              seam: float, ramp: float, depth: float, z0: float,
              ox: float, oy: float, cell: float, out: wp.array3d(dtype=float)):
    """The lane strip's grid from the patches' heightfield (sheared back onto the
    road); under a live patch, the grains' own top, sunk under the soup."""
    i, j = wp.tid()
    x = ox + float(i) * cell
    z = -(oy + float(j) * cell)
    # far field: bilinear in the cell-centred heightfield (local y)
    u = (x - hox) / hdx - 0.5
    w = (z - hoz) / hdx - 0.5
    i0 = int(wp.floor(u))
    j0 = int(wp.floor(w))
    fu = u - float(i0)
    fw = w - float(j0)
    i0c = wp.clamp(i0, 0, H.shape[0] - 1)
    i1c = wp.clamp(i0 + 1, 0, H.shape[0] - 1)
    j0c = wp.clamp(j0, 0, H.shape[1] - 1)
    j1c = wp.clamp(j0 + 1, 0, H.shape[1] - 1)
    y = ((H[i0c, j0c] * (1.0 - fu) + H[i1c, j0c] * fu) * (1.0 - fw)
         + (H[i0c, j1c] * (1.0 - fu) + H[i1c, j1c] * fu) * fw)
    wx = float(ix) * h
    wz = float(iz) * h
    for k in range(pc.shape[0]):
        ex = x - float(pc[k, 0] + marg) * h
        ez = z - float(pc[k, 1] + marg) * h
        if ex >= 0.0 and ex < wx and ez >= 0.0 and ez < wz:
            ci = wp.min(int(ex / h), top.shape[1] - 1)
            cj = wp.min(int(ez / h), top.shape[2] - 1)
            t = wp.max(top[k, ci, cj], 0.0)
            d = wp.min(wp.min(ex, wx - ex), wp.min(ez, wz - ez))
            a = wp.clamp(d / ramp, 0.0, 1.0)
            y = t - seam * a * a * (3.0 - 2.0 * a)
    out[0, i, j] = y - depth + B[i, j] - z0


def _base_host(x, z):
    """base() at a few points, off the lane grid (the same numbers gr_base reads)."""
    Bg = BASE_GRID["gravel"][0]
    u = np.clip((np.asarray(x) - X0) / CELL, 0.0, Bg.shape[0] - 1.001)
    w = np.clip((GRAVEL_Z[1] - np.asarray(z)) / CELL, 0.0, Bg.shape[1] - 1.001)
    i, j = u.astype(int), w.astype(int)
    fu, fw = u - i, w - j
    return ((Bg[i, j] * (1 - fu) + Bg[i + 1, j] * fu) * (1 - fw)
            + (Bg[i, j + 1] * (1 - fu) + Bg[i + 1, j + 1] * fu) * fw)


class GranularLane:
    """The gravel lane's four MPM patches, their coupling state and their look."""

    def __init__(self):
        t = terrain["gravel"]
        self.t = t
        self.dev = t._wp_device
        self.sim = None
        self._make_sim()
        self.base = wp.array(np.ascontiguousarray(BASE_GRID["gravel"][0]), dtype=float,
                             device=self.dev)
        sim = self.sim
        self.top = wp.zeros((4, sim.ix + 1, sim.iz + 1), dtype=float, device=self.dev)
        self.xs = wp.zeros(sim.n_slots, dtype=wp.vec3, device=self.dev)
        # One marching-cubes box per patch, cropped to the patch interior (the
        # soup then has no side walls to hide) and moved with it every frame.
        c = 1.15 * sim.pd
        self.mc_cell = c
        self.mc_dims = (int(GR_PATCH[0] / c) + 1,
                        int((GR_DEPTH + 0.2 + 0.12) / c) + 3,
                        int(GR_PATCH[1] / c) + 1)
        self.surfs = [DensitySurface((0.0, 0.0, 0.0), c, self.mc_dims, self.dev)
                      for _ in range(4)]
        self.iso = 0.46 * (c / sim.pd) ** 3
        cap = GR_MAX_TRIS * 3 * 4
        self.stage = (wp.zeros(cap, dtype=wp.vec3, device=self.dev),
                      wp.zeros(cap, dtype=wp.vec3, device=self.dev))
        self.geometry = tp.BufferGeometry()
        self.geometry.set_attribute("position", np.zeros((cap, 3), np.float32))
        self.geometry.set_attribute("normal", np.tile(np.float32([0, 1, 0]), (cap, 1)))
        self.geometry.set_draw_range(0, 0)
        # The lane's own albedo (sRGB, decoded by the material -- the same
        # number LANE_ALBEDO carries for the strip), so the soup and the strip
        # it lies on are one surface; the stones are the detail layer's.
        m = tp.MeshPhysicalMaterial()
        m.color = 0x49443c
        m.roughness = 0.94
        m.specular_intensity = 0.18
        m.side = tp.Side.Double
        self.material = m
        self.mesh = tp.Mesh(self.geometry, m)
        self.mesh.cast_shadow = True
        self.mesh.receive_shadow = True
        self.mesh.frustum_culled = False     # the CPU-side bounds never see the soup
        self.ntris = 0
        self.dirty = False                   # the grains moved since the last publish
        self.on = np.zeros(4, bool)          # wheel on the lane (its outputs are used)
        self.mu = np.full(4, GR_MU0)
        self.R = np.zeros(4)
        self.fy = np.zeros(4)
        self.thrust = np.zeros(4)
        self.out = np.zeros((4, 16), np.float32)
        self.nsub = 0
        self.ms = 0.0
        self.pub_ms = 0.0                    # soup rebuild + host copy, ms (EMA)
        print(f"  gravel lane: 4 MLS-MPM patches, {sim.n_slots:,} particle slots, "
              f"h = {GR_H * 1000:.0f} mm, {GR_PATCH[0]:.2f} x {GR_PATCH[1]:.2f} x "
              f"{GR_DEPTH:.2f} m each; heightfield {sim.hf.shape[0]}x{sim.hf.shape[1]} "
              f"@ {sim.hf_dx * 1000:.0f} mm")

    def _make_sim(self):
        self.sim = gm.GranularPatches(
            gravel_material(), GR_H, 4, patch_size=GR_PATCH, depth=GR_DEPTH,
            hf_origin=(X0 - 1.0, GRAVEL_Z[0] - 1.0), hf_size=(LANE_LEN + 2.0, GRAVEL_W + 2.0),
            floor_y=0.0, device=self.dev)
        # Pre-capture every substep count the lane can ask for, with the wheels
        # hovering over the spawn end: a first-use capture mid-drive is a hitch.
        zg = 0.5 * (GRAVEL_Z[0] + GRAVEL_Z[1])
        pos = np.array([[SPAWN_X + dx, 0.0, zg + dz] for dx, dz in
                        ((1.33, -0.82), (1.33, 0.82), (-1.33, -0.82), (-1.33, 0.82))])
        zero = np.zeros((4, 3))
        axis = np.tile([0.0, 0.0, 1.0], (4, 1))
        for n in sorted({self.nsub_q(s) for s in (0.0, 10.0, 20.0, 30.0, 40.0)}):
            pos[:, 1] = GR_DEPTH + R_WHEEL + GR_LIFT
            self.sim.set_wheels(pos, zero, zero, axis, 0.0, R_WHEEL, GR_HALF_W,
                                mass=GR_MASS, damp=GR_DAMP, free_y=False, set_y=True)
            self.sim.step_frame(n)
            self.sim.step_frame(n)
        self.was_on = np.zeros(4, bool)

    def nsub_q(self, speed):
        n = self.sim.nsub_for(min(float(speed), 40.0))
        return GR_NSUB_Q * int(math.ceil(n / GR_NSUB_Q))

    def reset(self):
        self._make_sim()
        self.geometry.set_draw_range(0, 0)
        self.ntris = 0
        self.dirty = False
        self.mu[:] = GR_MU0
        self.R[:] = 0.0

    def step(self, hub, vel, omega, axis, load, on):
        """One frame of the four patches. Returns the (4, 16) readback, or None
        when no wheel is on the lane (the grains are then not stepped)."""
        if not any(on):
            self.on[:] = False
            self.was_on[:] = False
            return None
        t0 = time.perf_counter()
        sim = self.sim
        sim.set_wheels(hub, vel, omega, axis, load, R_WHEEL, GR_HALF_W,
                       mass=GR_MASS, damp=GR_DAMP)
        c = sim.cmd
        for k in range(4):
            if not on[k]:
                # Hover: kinematic, unloaded, not spinning, over its own patch.
                c[k, 1] = GR_DEPTH + R_WHEEL + GR_LIFT
                c[k, 4] = 0.0
                c[k, 6:9] = 0.0
                c[k, 12] = 0.0
                c[k, 18] = 0.0
                c[k, 19] = 1.0
            elif not self.was_on[k]:
                # Just arrived (or the patch was hovering): set it down ON the
                # stone; from here on its height is the grains'.
                c[k, 1] = GR_DEPTH + R_WHEEL + sim.eps + 0.002
                c[k, 4] = 0.0
                c[k, 19] = 1.0
        speed = 0.0
        for k in range(4):
            if on[k]:
                speed = max(speed, float(np.hypot(vel[k][0], vel[k][2])),
                            float(np.linalg.norm(omega[k])) * R_WHEEL)
        self.nsub = self.nsub_q(speed)
        self.out = sim.step_frame(self.nsub)
        self.on = np.array(on, bool)
        self.was_on = self.on.copy()
        self.ms = (time.perf_counter() - t0) * 1000.0
        self.dirty = True
        # The strip's grid follows every step (cheap, no sync); the soup is
        # rebuilt lazily in publish(), once per RENDERED frame.
        sim_ = self.sim
        self.top.fill_(-1.0)
        wp.launch(gr_particles, dim=sim_.n_slots, device=self.dev,
                  inputs=[sim_.x, sim_.v, sim_.alive, sim_.cap, self.base, float(X0),
                          float(GRAVEL_Z[1]), float(CELL), GR_DEPTH, float(X0),
                          float(X0 + (NX - 1) * CELL), float(GRAVEL_Z[0]), float(GRAVEL_Z[1]),
                          sim_.pc, sim_.marg, sim_.h, sim_.pd, GR_VTOP, self.top, self.xs])
        t = self.t
        wp.launch(gr_lane_h, dim=(t.nx, t.ny), device=self.dev,
                  inputs=[sim_.hf, sim_.hf_origin[0], sim_.hf_origin[1], sim_.hf_dx,
                          self.base, self.top, sim_.pc, sim_.marg, sim_.h, sim_.ix, sim_.iz,
                          GR_SEAM, GR_SEAM_RAMP, GR_DEPTH, float(t.z0),
                          float(t.origin_np[0][0]), float(t.origin_np[0][1]), float(t.cell),
                          t.h])
        return self.out

    def publish(self):
        """The soup: one marching-cubes box per patch, into ONE mesh (host route)."""
        if not self.dirty:
            return
        t0 = time.perf_counter()
        self.dirty = False
        sim = self.sim
        c = self.mc_cell
        total = 0
        pc = self.out[:, gm.OUT_BLOCK]
        sp, sn = self.stage
        for k in range(4):
            lo_x = (pc[k, 0] + sim.marg) * sim.h
            lo_z = (pc[k, 1] + sim.marg) * sim.h
            b = _base_host([lo_x, lo_x + GR_PATCH[0]], [lo_z, lo_z + GR_PATCH[1]])
            oy = float(b.min()) - GR_DEPTH + 0.02
            s = self.surfs[k]
            o = (float(lo_x), oy, float(lo_z))
            s.origin = wp.vec3(*o)
            s.mc.domain_bounds_lower_corner = wp.vec3(*o)
            s.mc.domain_bounds_upper_corner = wp.vec3(o[0] + (self.mc_dims[0] - 1) * c,
                                                      o[1] + (self.mc_dims[1] - 1) * c,
                                                      o[2] + (self.mc_dims[2] - 1) * c)
            n = min(s.build(self.xs[k * sim.cap:(k + 1) * sim.cap], sim.cap, self.iso),
                    GR_MAX_TRIS)
            if n > 0:
                s.expand(n, sp[3 * total:3 * (total + n)], sn[3 * total:3 * (total + n)],
                         sign=MC_SIGN, flip_winding=MC_FLIP, grain=0.55, grain_freq=70.0)
            total += n
        self.ntris = total
        if total > 0:
            self.geometry.update_attribute("position", sp[:3 * total].numpy())
            self.geometry.update_attribute("normal", sn[:3 * total].numpy())
        self.geometry.set_draw_range(0, 3 * total)
        self.pub_ms += ((time.perf_counter() - t0) * 1000.0 - self.pub_ms) * 0.1


gravel = GranularLane() if GRAVEL else None
if gravel is not None:
    scene.add(gravel.mesh)
    if hasattr(gravel.material, "detail_normal_map"):
        # The stones: a per-PIXEL detail layer (triplanar, world-anchored, so
        # the strip and the soup carry the SAME stones across the seam). One
        # stone is 1 / (detail_repeat * 12) m: 28 mm. GL ignores the layer.
        _grain = tp.data_texture(grain_detail_texture(256, 144), srgb=False)
        for _m in (gravel.material, gravel_mat):
            _m.detail_normal_map = _grain
            _m.detail_repeat = 3.0
            _m.detail_normal_scale = 1.0


def gravel_couple(hub, lane_of, grade, w_load, q):
    """The grains' step for this frame, BEFORE PhysX: per gravel wheel the road
    override height (the rim bottom the grains hold up), the tyre ceiling (their
    gross thrust over W) and the resistance to add_force. Returns
    {i: (road_y, mu, R)} for the wheels on the lane."""
    if gravel is None:
        return {}
    on = [lane_of[i] == "gravel" for i in range(4)]
    vel = np.zeros((4, 3))
    om = np.zeros((4, 3))
    axis = np.zeros((4, 3))
    fwd_w = np.zeros((4, 3))
    for i in range(4):
        # The STEERED axle: the wheel's own local rotation (steer, then spin
        # about the axle, which leaves the axle where it is) under the chassis.
        _, lq = vehicle.wheel_local_pose(i)
        a = qrot(q, qrot(lq, np.array([1.0, 0.0, 0.0])))
        a[1] = 0.0
        a /= max(np.linalg.norm(a), 1e-9)
        # MPM convention: omega = -axis * spin for forward rolling, and the axle
        # moment T_AXLE is + when it resists that. PhysX's right vector with
        # omega = right * wheel_angular_speed is the same wheel (see the old
        # bed's note: a wheel rolling true stands still against the stone).
        axis[i] = -a
        om[i] = a * float(vehicle.wheel_angular_speed(i))
        fwd_w[i] = np.cross(a, [0.0, 1.0, 0.0])
        if prev_hub is not None:
            vel[i] = (hub[i] - prev_hub[i]) / DT
            vel[i, 1] = 0.0
    W = np.maximum(w_load, 0.0) + M_UNSPRUNG * 9.81
    out = gravel.step(hub, vel, om, axis, W, on)
    if out is None:
        return {}
    res = {}
    k_ema = DT / (GR_TAU + DT)
    for i in range(4):
        if not on[i]:
            continue
        f = out[i, gm.OUT_F]
        t_ax = float(out[i, gm.OUT_T_AXLE])
        # Which way the wheel is going (or trying to): travel, else spin.
        v_roll = float(np.dot(vel[i], fwd_w[i]))
        s = 1.0 if (v_roll + float(vehicle.wheel_angular_speed(i)) * R_WHEEL) >= 0.0 else -1.0
        thrust = s * t_ax / R_WHEEL                    # gross thrust H, + = propulsive
        dp = s * float(f[0] * fwd_w[i][0] + f[2] * fwd_w[i][2])
        mu_raw = abs(t_ax) / R_WHEEL / W[i]
        rim = float(vehicle.wheel_angular_speed(i)) * R_WHEEL
        slip = (rim - v_roll) / max(abs(rim), abs(v_roll), 0.5)
        if abs(slip) > GR_SLIP_SAT and abs(rim - v_roll) > GR_SCRUB_SAT:
            gravel.mu[i] += (float(np.clip(mu_raw, *GR_MU_RANGE)) - gravel.mu[i]) * k_ema
        gravel.R[i] += (max(thrust - dp, 0.0) - gravel.R[i]) * k_ema
        gravel.fy[i] = float(f[1]) / W[i]
        gravel.thrust[i] = mu_raw
        road = float(out[i, gm.OUT_Y_BOTTOM]) - GR_DEPTH + grade[i]
        res[i] = (road, gravel.mu[i], gravel.R[i])
    return res


# --- the rover -----------------------------------------------------------------


def data_dir():
    """threepp_data's checkout. THREEPP_DATA_DIR wins; otherwise the usual
    places -- note the checkout is commonly named with a hyphen."""
    env = os.environ.get("THREEPP_DATA_DIR")
    if env and os.path.isdir(env):
        return env
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for name in ("threepp-data", "threepp_data"):
        cand = os.path.join(os.path.dirname(repo), name)
        if os.path.isdir(cand):
            return cand
    return ""


chassis = tp.Group()
scene.add(chassis)

# PhysX index -> the model's own wheel tag. The model labels its wheels from
# inside the car, so its "FL" sits at +x, which is where PhysX puts wheel 0
# (front-RIGHT, +x +z). The tags therefore look swapped and are not.
WHEEL_TAG = ("WheelFL", "WheelFR", "WheelBL", "WheelBR")
wheel_rigs = [tp.Group() for _ in range(4)]
for rig in wheel_rigs:
    chassis.add(rig)

brake_mat = reverse_mat = None
model_path = os.path.join(data_dir(), "models", "gltf",
                          "2015_land-rover_range_rover_evoque_coupe", "scene.gltf")
if os.path.isfile(model_path):
    body = tp.ModelLoader().load(model_path)
    body.scale.set(100.0, 100.0, 100.0)     # the gltf bakes a 0.01 at its root
    # The model's contact patches sit at its own y = 0; the PhysX chassis centre
    # rides wheel_radius + |attachment_y| + travel - jounce = 0.996 m above the
    # road. Drop the shell by a metre and the wheel wells land on the rigs.
    body.position.y = -1.0
    chassis.add(body)

    parts = [[] for _ in range(4)]
    lamps = {"lights_position_back": [], "lights_reverse": []}

    def _sort(o):
        if type(o).__name__ != "Mesh":
            return
        # The .gltf packs AO and metal-roughness into ONE texture (R = AO,
        # G = rough, B = metal) and never authored R, so an aoMap read is a
        # constant zero and the whole body goes black. Drop it.
        try:
            o.material.ao_map = None
        except Exception:                               # noqa: BLE001 - not all have one
            pass
        for i, tag in enumerate(WHEEL_TAG):
            if tag in o.name:
                parts[i].append(o)
                return
        for key in lamps:
            if key in o.name:
                lamps[key].append(o)

    body.traverse(_sort)

    for i, group in enumerate(parts):
        if not group:
            continue
        # The per-wheel "pivot" nodes are identity transforms -- the wheel
        # positions are baked into the geometry -- so the hub is the centre of
        # the combined vertex AABB, and a clone recentred on it spins about its
        # own axle under a rig driven by wheel_local_pose(). The wheel is
        # axisymmetric about that axle, which is why the model's own axis
        # convention does not need undoing here.
        lo = np.full(3, 1.0e30)
        hi = np.full(3, -1.0e30)
        for part in group:
            a = part.geometry.get_attribute("position")
            lo = np.minimum(lo, a.min(0))
            hi = np.maximum(hi, a.max(0))
        hub = 0.5 * (lo + hi)
        for part in group:
            clone = part.clone()                # clone BEFORE hiding: copy() takes `visible`
            part.visible = False
            clone.visible = True
            clone.position.set(float(-hub[0]), float(-hub[1]), float(-hub[2]))
            wheel_rigs[i].add(clone)

    # Brake and reverse lamps. The model gives them their own meshes but ONE
    # shared emissive material, so they get a fresh material each and can flare
    # independently.
    for key, colour in (("lights_position_back", 0xff2a12), ("lights_reverse", 0xfff2e0)):
        if not lamps[key]:
            continue
        mat = standard_material(0x201a18, 0.6, emissive=colour)
        mat.emissive_intensity = 1.0
        for m in lamps[key]:
            m.set_material(mat)
        if key == "lights_position_back":
            brake_mat = mat
        else:
            reverse_mat = mat
else:
    print(f"  note: no Evoque at {model_path} -- driving a box "
          f"(set THREEPP_DATA_DIR)")
    shell = tp.Mesh(tp.BoxGeometry(1.95, 1.2, 4.4), standard_material(0x7a2f24, 0.5, 0.3))
    shell.cast_shadow = True
    chassis.add(shell)
    for rig in wheel_rigs:
        w = tp.Mesh(tp.CylinderGeometry(R_WHEEL, R_WHEEL, 0.3, 18), standard_material(0x1c1c1e, 0.8))
        w.rotate_z(math.pi / 2)
        w.cast_shadow = True
        rig.add(w)

# --- cameras -------------------------------------------------------------------

camera = tp.PerspectiveCamera(58, canvas.aspect(), 0.1, 400)
camera.position.set(-30.0, 4.0, 0.0)
pov = tp.PerspectiveCamera(72, canvas.aspect(), 0.05, 400)
pov.position.set(0.4, 0.45, -0.15)
pov.rotation.y = math.pi           # camera looks down -z; chassis forward is +z
chassis.add(pov)

CHASE_DIST, CHASE_PITCH = 9.5, 0.30
cam_pos = np.array([-30.0, 4.0, 0.0])
cam_tgt = np.array([-20.0, 1.0, 0.0])
view_mode = 0                      # 0 chase, 1 POV, 2 cinematic orbit
cine_t = 0.0


def resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    pov.aspect = camera.aspect
    pov.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(resize)

# --- input ---------------------------------------------------------------------

steer_cmd = 0.0
throttle_cmd = 0.0
brake_cmd = 0.0
edge = {}


def pressed(key):
    """True on the frame `key` goes down (canvas.is_key_down is a level)."""
    down = canvas.is_key_down(key)
    was = edge.get(key, False)
    edge[key] = down
    return down and not was


# --- coupling state ------------------------------------------------------------

w_ema = np.full(4, REST_LOAD)
prev_hub = None                   # full hub positions last frame, for drag direction
dig = np.ones(4)                  # slip-sinkage factor, EMA toward 1 + K_DIG*slip
LANE_IDLE_S = 2.0                 # a heightfield lane untouched this long stops stepping
lane_idle = {n: 0.0 for n in LANES}

# Traction control, on by default (the real Evoque has it -- "Terrain Response"
# is exactly this). The direct drive can put 1500 N*m on a wheel whose mud
# ceiling is ~530 N*m, so anything past ~35 % throttle just spins without it.
# Cuts throttle while any contact patch scrubs past TC_SCRUB, recovers when it
# grips. X toggles it off -- at which point flooring it in mud digs the car in,
# which is the other half of the demo.
TC_SCRUB = 2.5                    # m/s of scrub where the cut starts
tc_on = True
tc_cut = 1.0
radii = torch.full((4,), R_WHEEL, device=device)
centers = {n: torch.zeros(1, 4, 3, device=device) for n in LANES}
vels = {n: torch.zeros(1, 4, 3, device=device) for n in LANES}
c_host = {n: np.zeros((1, 4, 3), np.float32) for n in LANES}
v_host = {n: np.zeros((1, 4, 3), np.float32) for n in LANES}
hud = dict(lane=[None] * 4, z=np.zeros(4), mu=np.zeros(4), w=np.zeros(4),
           bek=np.zeros(4), slip=np.zeros(4), util=np.zeros(4), j=np.zeros(4),
           dig=np.ones(4), drag=np.zeros(4), over=[False] * 4,
           # display-only, and the wheel spray's whole trigger: the contact
           # patch's scrub speed and its lateral component, recorded where the
           # coupling loop already computes them for the Janosi integral.
           scrub=np.zeros(4), slip_lat=np.zeros(4))
_readback = 0
PHASE = dict(on=False, gravel=0.0)   # --bench: time spent in gravel_couple


def coupled_step(throttle, steer, brake, relax_iters=2):
    """One 1/60 step: ground under the wheels, then PhysX over the top."""
    global w_ema, prev_hub, tc_cut, _readback

    # 1. chassis pose + wheel world centres
    p, q = vehicle.position, vehicle.quaternion
    origin = np.array([p.x, p.y, p.z])
    hub = np.empty((4, 3))
    for i in range(4):
        lp, _ = vehicle.wheel_local_pose(i)
        hub[i] = origin + qrot(q, np.array([lp.x, lp.y, lp.z]))
    fwd = qrot(q, np.array([0.0, 0.0, 1.0]))
    right = qrot(q, np.array([1.0, 0.0, 0.0]))
    v_fwd = vehicle.forward_speed

    # 2. the GRADE under each wheel -- the bearing datum, not the surface. One
    #    gather per lane and one transfer for all of them.
    lane_of = [lane_at(hub[i, 0], hub[i, 2]) for i in range(4)]
    grade = np.zeros(4)
    gathered = []
    for name in LANES:
        idx = [i for i in range(4) if lane_of[i] == name]
        if not idx:
            continue
        if name == "gravel" and gravel is not None:
            # The grains, not this grid, carry the gravel wheels; the grade is
            # only the HUD's datum there, so read the road profile on the host
            # and skip a device gather + sync per step.
            for i in idx:
                grade[i] = float(base(hub[i, 0], hub[i, 2]))
            continue
        t = terrain[name]
        xs = torch.tensor([[hub[i, 0] for i in idx]], device=device, dtype=torch.float32)
        ys = torch.tensor([[-hub[i, 2] for i in idx]], device=device, dtype=torch.float32)
        gathered.append((idx, bilinear(t, t.grade_torch.view(1, t.nx * t.ny), xs, ys)))
    if gathered:
        flat = torch.cat([g[1][0] for g in gathered]).cpu().numpy()
        k = 0
        for idx, _ in gathered:
            for i in idx:
                grade[i] = flat[k]
                k += 1

    # 3. wheel load, smoothed. The road-height/load loop has a frame of lag in
    #    it and rings at the suspension frequency without this.
    load = np.array([vehicle.suspension_force(i) for i in range(4)])
    w_ema += (load - w_ema) * (DT / (0.15 + DT))

    # 4/5. invert Bekker for the sinkage, Mohr-Coulomb for the friction, and
    #      hand the suspension a road at grade - z_eq. The dig factor (updated
    #      in step 6 from last frame's slip) deepens the road under a spinning
    #      wheel: slip sinkage, one frame of lag like the load loop.
    z_eq = np.zeros(4)
    mu = np.zeros(4)
    for i in range(4):
        if lane_of[i] is None or lane_of[i] == "gravel":
            if lane_of[i] is None:
                vehicle.clear_road_override(i)
            dig[i] += (1.0 - dig[i]) * (DT / (DIG_TAU + DT))
            continue
        z_st, mu[i] = sinkage(lane_of[i], w_ema[i])
        z_eq[i] = min(z_st * dig[i], 0.45 * (2.0 * R_WHEEL))
        vehicle.set_road_override(i, float(grade[i] - z_eq[i]), float(mu[i]))

    # 4b. the gravel lane: the GRAINS carry these wheels (see "the gravel lane"
    #     for the one-channel-per-force rule). One MPM frame for all four
    #     patches, then per gravel wheel: road = the rim bottom the grains hold
    #     up, mu = their gross thrust over W, and R for step 7.
    if PHASE["on"]:
        wp.synchronize_device(device)
        _tg = time.perf_counter()
    gr = gravel_couple(hub, lane_of, grade, w_ema, q)
    if PHASE["on"]:
        wp.synchronize_device(device)
        PHASE["gravel"] += time.perf_counter() - _tg
    for i, (road, mu_g, _r) in gr.items():
        z_eq[i] = grade[i] - road
        mu[i] = mu_g
        vehicle.set_road_override(i, float(road), float(mu_g))

    # 6. carve. The wheels are the module's collider spheres; their bottoms are
    #    already at grade - z_eq, so the imprint cuts the rut to exactly the
    #    Bekker sinkage and the berm is whatever the material would not compact.
    #    Velocity is the CONTACT PATCH slip, not the hub velocity: that is what
    #    the Janosi integral is measured along.
    max_scrub = 0.0
    hud["scrub"][:] = 0.0
    hud["slip_lat"][:] = 0.0
    for name in LANES:
        c, v = c_host[name], v_host[name]
        for i in range(4):
            if lane_of[i] != name:
                c[0, i] = (0.0, -LANE_Z[name][1], 50.0)     # parked well clear
                v[0, i] = (0.0, 0.0, 0.0)
                continue
            c[0, i] = (hub[i, 0], -hub[i, 2], hub[i, 1])
            slip_long = vehicle.wheel_angular_speed(i) * R_WHEEL - v_fwd
            slip_lat = vehicle.tire_lateral_slip(i) * abs(v_fwd)
            s = slip_long * fwd + slip_lat * right
            vy = 0.0 if prev_hub is None else (hub[i, 1] - prev_hub[i, 1]) / DT
            v[0, i] = (s[0], -s[2], vy)
            # Slip sinkage follows the patch's scrub SPEED past a dead zone
            # (see SCRUB_DEAD / SCRUB_REF). Digging DEEPER also requires the
            # wheel to be carrying load: excavation is pressure ejecting
            # material, and an unloaded wheel just flings mud. Without that
            # gate there is a runaway -- the road drops faster than the body
            # can follow, the wheel unloads, spins up, and pins the dig.
            max_scrub = max(max_scrub, math.hypot(s[0], s[2]))
            hud["scrub"][i] = math.hypot(s[0], s[2])
            hud["slip_lat"][i] = slip_lat
            if name == "gravel":
                continue                # slip sinkage is the grains' own
            scrub = max(math.hypot(s[0], s[2]) - SCRUB_DEAD, 0.0)
            target = 1.0 + K_DIG[name] * min(scrub / SCRUB_REF, 1.0)
            target = min(target, DIG_MAX[name])
            if target < dig[i] or w_ema[i] > 0.35 * REST_LOAD:
                dig[i] += (target - dig[i]) * (DT / (DIG_TAU + DT))
        if name == "gravel":
            # The one lane whose surface is NOT this module's: the scrub above
            # still runs (traction control and the spray read it), the carve
            # does not -- the grains cut the rut, and gravel_couple writes the
            # lane's grid from them.
            continue
        # A lane nobody has touched for LANE_IDLE_S seconds is left alone: its
        # parked-collider deform + relax changed nothing but the slow grade
        # memory, and cost ~0.5 ms a lane a step. The ruts get LANE_IDLE_S to
        # settle after the last wheel leaves; the grade memory pauses while
        # the lane is idle (a rut datum is kept, not forgotten, in between).
        if any(lane_of[i] == name for i in range(4)):
            lane_idle[name] = 0.0
        else:
            lane_idle[name] += DT
            if lane_idle[name] > LANE_IDLE_S:
                continue
        centers[name].copy_(torch.from_numpy(c))
        vels[name].copy_(torch.from_numpy(v))
        terrain[name].deform(centers[name], radii, vels[name], DT)
        terrain[name].relax(relax_iters)

    # 7. motion resistance -- the one soil force actually applied (see the audit
    #    at the top of the file). Opposes each wheel's FULL horizontal travel:
    #    a sideways slide ploughs soil just like forward travel does, which is
    #    why a slide in mud dies instead of sailing. Tapered below 0.5 m/s so it
    #    never pushes a resting car backwards, capped per wheel for stability.
    hud["drag"][:] = 0.0
    for i in range(4):
        if lane_of[i] is None:
            continue
        vh = (np.zeros(3) if prev_hub is None else (hub[i] - prev_hub[i]) / DT)
        vh[1] = 0.0
        speed_h = float(np.linalg.norm(vh))
        # Gravel: the grains' R = T/r - DP (step 4b), the one channel their
        # resistance has. Everywhere else: Bekker's.
        f = gr[i][2] if i in gr else motion_resistance(lane_of[i], z_eq[i])
        f = min(f, DRAG_CAP_FRAC * max(w_ema[i], 1.0)) * min(speed_h / 0.5, 1.0)
        hud["drag"][i] = f
        if f <= 0.0 or speed_h < 0.05:
            continue
        d = vh / speed_h
        vehicle.add_force_at_pos(tp.Vector3(float(-d[0] * f), 0.0, float(-d[2] * f)),
                                 tp.Vector3(*hub[i]))
    prev_hub = hub.copy()

    # 8. and PhysX runs the car -- through the traction control, which is the
    #    difference between "drivable at any throttle" and "any throttle spins".
    if tc_on:
        err = max_scrub - TC_SCRUB
        if err > 0.0:
            tc_cut = max(0.15, tc_cut - 4.0 * DT * min(err / 4.0, 1.0))
        else:
            tc_cut = min(1.0, tc_cut + 1.5 * DT)
        throttle = throttle * tc_cut
    else:
        tc_cut = 1.0
    vehicle.set_throttle(throttle)
    vehicle.set_steer(steer)
    vehicle.set_brake(brake)
    world.step(DT)

    # Panel telemetry. One small transfer every third frame: the module's own
    # Bekker bearing and Janosi shear next to what PhysX's suspension is
    # carrying. Nothing in the loop reads these back.
    _readback += 1
    if _readback % 3 == 0:
        pack = torch.stack([torch.cat([terrain[n].forces[0, :, 0:3].reshape(-1),
                                       terrain[n].slip_j[0]]) for n in LANES]).cpu().numpy()
        for i in range(4):
            n = lane_of[i]
            hud["bek"][i] = 0.0 if n is None else pack[LANES.index(n)][i * 3 + 2]
            hud["j"][i] = 0.0 if n is None else pack[LANES.index(n)][12 + i]
            if n is not None:
                ft = math.hypot(pack[LANES.index(n)][i * 3], pack[LANES.index(n)][i * 3 + 1])
                hud["util"][i] = ft / max(mu[i] * w_ema[i], 1.0)
    for i in gr:
        # The grains' vertical force next to the load (the panel's second
        # column), and how much of the ceiling they set the tyre is using.
        hud["bek"][i] = gravel.fy[i] * (w_ema[i] + M_UNSPRUNG * 9.81)
        hud["util"][i] = gravel.thrust[i] / max(gravel.mu[i], 1e-3)
    hud["lane"] = lane_of
    hud["z"] = z_eq
    hud["dig"] = dig.copy()
    hud["mu"] = mu
    hud["w"] = load
    hud["slip"] = np.array([vehicle.tire_longitudinal_slip(i) for i in range(4)])
    hud["over"] = [vehicle.road_override_active(i) for i in range(4)]
    return hub


# --- UI ------------------------------------------------------------------------

ui = tp.ImguiContext(canvas, renderer) if not (SHOT or BENCH or FILM) else None
fps_ema = 60.0
shot_no = 0
WHEEL_NAME = ("FR", "FL", "RR", "RL")


def bar(v, n=14):
    v = max(0.0, min(1.0, v))
    k = int(round(v * n))
    return "[" + "#" * k + "." * (n - k) + "]"


def draw_ui():
    tp.imgui.set_next_window_pos(8, 8)
    tp.imgui.set_next_window_size(376, 0)
    tp.imgui.begin("Bekker-Wong drive")
    tp.imgui.text(f"{vehicle.forward_speed * 3.6:6.1f} km/h   "
                  f"{'REVERSE' if vehicle.gear == tp.PhysxVehicle.Gear.REVERSE else 'FORWARD'}"
                  f"   {fps_ema:5.1f} fps")
    tp.imgui.text(f"throttle {bar(throttle_cmd)}  brake {bar(brake_cmd)}")
    tp.imgui.text(f"traction control {'ON ' if tc_on else 'OFF'} "
                  f"{bar(tc_cut) if tc_on else '[ floor it and dig ]'}")
    tp.imgui.text(f"steer    {bar(0.5 + 0.5 * steer_cmd)}")
    tp.imgui.separator()
    tp.imgui.text("wh lane  sink   load   mu   slip  util  dig  drag")
    for i in range(4):
        n = hud["lane"][i]
        tp.imgui.text(f"{WHEEL_NAME[i]} {(n or 'rigid'):5s} "
                      f"{hud['z'][i] * 1000:5.1f} {hud['w'][i] / 1000:6.2f} "
                      f"{(hud['mu'][i] if n else FALLBACK_MU):5.2f} "
                      f"{hud['slip'][i]:6.2f} {hud['util'][i]:5.2f} "
                      f"{hud['dig'][i]:4.2f} {hud['drag'][i] / 1000:5.2f}")
    tp.imgui.text("           mm     kN               -         kN")
    if (np.mean(hud["dig"]) > 1.6 and abs(vehicle.forward_speed) < 0.5
            and np.mean(hud["util"]) > 0.9):
        tp.imgui.text("DUG IN -- ease off the throttle and creep out")
    tp.imgui.separator()
    tp.imgui.text("soil bearing vs PhysX suspension load")
    for i in range(4):
        if hud["lane"][i] is None:
            tp.imgui.text(f"{WHEEL_NAME[i]}  -- off the lanes (rigid ground) --")
            continue
        if hud["lane"][i] == "gravel":
            tp.imgui.text(f"{WHEEL_NAME[i]} grains {hud['bek'][i]:7.0f} N   "
                          f"PhysX {hud['w'][i]:7.0f} N  (+{M_UNSPRUNG * 9.81:.0f} unsprung)")
            continue
        err = 100.0 * (hud["bek"][i] / max(hud["w"][i], 1.0) - 1.0)
        tp.imgui.text(f"{WHEEL_NAME[i]} Bekker {hud['bek'][i]:7.0f} N   "
                      f"PhysX {hud['w'][i]:7.0f} N   {err:+5.1f} %")
    if gravel is not None:
        tp.imgui.separator()
        # The gravel lane is two-way: the grains carry these wheels (bearing
        # through the suspension, thrust through the tyre ceiling, resistance
        # through add_force -- one channel each; see "the gravel lane").
        tp.imgui.text("gravel lane  MLS-MPM, TWO-WAY -- the grains carry the car")
        on = [i for i in range(4) if hud["lane"][i] == "gravel"]
        if on:
            tp.imgui.text(f"  {gravel.nsub} substeps  {gravel.ms:5.1f} ms  "
                          f"{gravel.ntris:,} tris")
            tp.imgui.text("   wh  Fy/W   H/W    mu     R (N)")
            for i in on:
                tp.imgui.text(f"   {WHEEL_NAME[i]}  {gravel.fy[i]:5.2f}  "
                              f"{gravel.thrust[i]:5.2f}  {gravel.mu[i]:5.2f}  "
                              f"{gravel.R[i]:8.0f}")
        else:
            tp.imgui.text("   no wheel on the stone (grains not stepped)")
    tp.imgui.separator()
    tp.imgui.text("W/S drive  A/D steer  R gear  SPACE handbrake  X tc")
    tp.imgui.text("V pov  C cinematic  BACKSPACE respawn  T reset  F frame")
    tp.imgui.end()


def save_shot(path):
    if GL:
        renderer.render(scene, camera)
        renderer.save_frame(path)
    else:
        renderer.save_frame(scene, active_camera(), path)
    print(f"  wrote {path}")


def active_camera():
    return pov if view_mode == 1 else camera


# --- frame ---------------------------------------------------------------------


def drive_inputs(dt):
    """Keyboard -> commands, with main.cpp's speed-sensitive steer and slew."""
    global steer_cmd, throttle_cmd, brake_cmd, view_mode, cine_t, shot_no
    if FRAMES > 0:
        # Timed run: the autopilot only. The window takes focus when it opens,
        # and keys typed into it by accident must not drive the car.
        steer_cmd, brake_cmd = 0.0, 0.0
        throttle_cmd = AUTO_THR if vehicle.forward_speed * 3.6 < AUTO_KMH else 0.0
        return
    left = canvas.is_key_down("A") or canvas.is_key_down("LEFT")
    rightk = canvas.is_key_down("D") or canvas.is_key_down("RIGHT")
    steer_in = (1.0 if left else 0.0) - (1.0 if rightk else 0.0)
    scale = 1.0 / (1.0 + abs(vehicle.forward_speed) * 3.6 * 0.015)
    steer_cmd += (steer_in * scale - steer_cmd) * min(1.0, dt * 2.0)
    throttle_cmd = 1.0 if (canvas.is_key_down("W") or canvas.is_key_down("UP")) else 0.0
    brake_cmd = 1.0 if (canvas.is_key_down("S") or canvas.is_key_down("DOWN")
                        or canvas.is_key_down("SPACE")) else 0.0

    if pressed("R"):
        vehicle.gear = (tp.PhysxVehicle.Gear.REVERSE
                        if vehicle.gear == tp.PhysxVehicle.Gear.FORWARD
                        else tp.PhysxVehicle.Gear.FORWARD)
    if pressed("X"):
        global tc_on
        tc_on = not tc_on
    if pressed("V"):
        view_mode = 0 if view_mode == 1 else 1
    if pressed("C"):
        view_mode = 0 if view_mode == 2 else 2
        cine_t = 0.0
    if pressed("BACKSPACE"):
        vehicle.respawn(SPAWN_POS, SPAWN_ROT)
        vehicle.gear = tp.PhysxVehicle.Gear.FORWARD
        steer_cmd = 0.0
        w_ema[:] = REST_LOAD
    if pressed("T"):
        # Reset the DEMO's soft-ground state along with the grids: a reset that
        # leaves the dig factor, the TC cut and the load EMA where they were is
        # not a reset a stuck car can feel.
        #
        # reset() only fills a SCALAR, and the ground is a road now -- calling
        # it alone would iron the lane flat and leave a 40 cm ledge where it
        # meets the apron. So: reset() for the state it clears (carry, slip,
        # contact area), then the stored profile straight back into h AND grade,
        # which are zero-copy views of the warp arrays the kernels read.
        global prev_hub, tc_cut
        for name, t in terrain.items():
            t.reset()
            t.h_torch.copy_(BASE_T[name])
            t.grade_torch.copy_(BASE_T[name])
        for s in strips:
            s.dirty = True
        dig[:] = 1.0
        tc_cut = 1.0
        w_ema[:] = REST_LOAD
        prev_hub = None
        for _s in sprays.values():          # airborne soil off a ground that
            _s.clear()                      # no longer has a rut in it
        if gravel is not None:
            # The grains are state too: a fresh set of patches over an
            # undisturbed heightfield (the lane grid went back to base above).
            gravel.reset()
    if pressed("F"):
        shot_no += 1
        save_shot(os.path.join(OUT_DIR, f"warp_mudsnow_drive_{shot_no:03d}.png"))


def cine_pose(c, a):
    """The cinematic orbit at phase `a`: eye and target.

    Low -- 1.2 to 2.2 m, i.e. inside a wheel's world rather than above the car
    -- and a full circle, so the orbit CROSSES the key's azimuth twice a lap.
    That is the whole point of it: at the two crossings the ruts are backlit and
    every berm on the lane draws its own shadow toward the lens, which is the
    frame this demo exists to produce. Anywhere else on the circle it is a car.
    """
    return (c + np.array([math.cos(a) * 9.5, 0.30 + 0.85 * (0.5 - 0.5 * math.cos(2.0 * a)),
                          math.sin(a) * 9.5]),
            c + np.array([0.0, -0.22, 0.0]))


# The phase of that orbit where the camera is behind the car and BEYOND it is
# the key: the hero framing, and the one --script cine parks on.
CINE_HERO = 3.55


def update_camera(dt):
    """main.cpp's chase: exponential lerp toward a chassis-relative offset, read
    off the interpolated visual rather than the raw actor pose -- `chassis` IS
    that interpolated visual (posed by frame() from the blended sim states);
    reading the actor here would re-introduce the 60 Hz stepping the blend
    exists to hide."""
    global cam_pos, cam_tgt, cine_t
    p, q = chassis.position, chassis.quaternion
    c = np.array([p.x, p.y, p.z])
    if view_mode == 2:
        cine_t += dt
        want, want_t = cine_pose(c, cine_t * 0.16)
    else:
        off = qrot(q, np.array([0.0, CHASE_DIST * math.sin(CHASE_PITCH),
                                -CHASE_DIST * math.cos(CHASE_PITCH)]))
        want = c + off
        want_t = c + qrot(q, np.array([0.0, 0.9, 3.0]))
    k = 1.0 - math.exp(-(2.0 if view_mode == 2 else 5.0) * dt)
    cam_pos += (want - cam_pos) * k
    cam_tgt += (want_t - cam_tgt) * k
    camera.position.set(*cam_pos)
    camera.look_at(tp.Vector3(*cam_tgt))


brake_was = reverse_was = False
wall_prev = None                  # last frame's wall clock, for the accumulator
sim_debt = 0.0                    # wall time owed to the sim, in seconds
pose_prev = pose_cur = None       # car pose at sim steps N-1 and N, for drawing
_ft_steps = [0]                   # sim steps taken by frame() (--frames reports it)
_flog = []                        # --frames-log rows


def _write_frames_log():
    if not FRAMES_LOG or not _flog:
        return
    import csv
    with open(FRAMES_LOG, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "wall_ms", "sim_steps", "steps_ms", "gravel_ms", "gravel_nsub",
                    "gravel_pub_ms", "pub_ms", "render_ms", "x", "z", "kmh", "wheels_on_gravel",
                    "soup_tris"])
        w.writerows(_flog)
    print(f"  wrote {FRAMES_LOG} ({len(_flog)} frames)")


atexit.register(_write_frames_log)


def nlerp(q0, q1, a):
    """Normalised quaternion lerp, sign-fixed so it never takes the long way.
    The rotations this blends are one sim step apart -- a few degrees at the
    worst wheelspin -- where nlerp and slerp are the same curve."""
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    q = q0 + (q1 - q0) * a
    return q / max(float(np.linalg.norm(q)), 1e-12)


def capture_pose():
    """Chassis pose + the four wheel local poses, as plain arrays."""
    p, q = vehicle.position, vehicle.quaternion
    wheels = []
    for i in range(4):
        lp, lq = vehicle.wheel_local_pose(i)
        wheels.append((np.array([lp.x, lp.y, lp.z]),
                       np.array([lq.x, lq.y, lq.z, lq.w])))
    return (np.array([p.x, p.y, p.z]), np.array([q.x, q.y, q.z, q.w]), wheels)


def frame():
    global fps_ema, brake_was, reverse_was, wall_prev, sim_debt, pose_prev, pose_cur
    t0 = time.perf_counter()
    # The sim runs at 60 Hz on the WALL clock, not one step per rendered frame:
    # welded to the render loop, the car lived at (fps/60)x real time -- slow
    # motion at 33 fps under the filmic look, fast-forward on an uncapped GL
    # frame. An accumulator pays sim steps out of wall time instead, capped at
    # 4 a frame so a hitch buys slow motion rather than a spiral of catch-up
    # steps that make the next frame later still. Interactive only -- the
    # script, film and bench loops are 60 Hz by construction, one step per
    # emitted frame, and stay on fixed DT.
    wall = 0.0 if wall_prev is None else t0 - wall_prev
    wall_prev = t0
    drive_inputs(min(wall, 0.1))      # once per RENDERED frame: R/T/F edge-trigger
    sim_debt = min(sim_debt + wall, 4.0 * DT)
    hub = None
    t_steps = time.perf_counter()
    n_steps = 0
    gr_ms = 0.0
    gr_nsub = 0
    while sim_debt >= DT:
        sim_debt -= DT
        hub = coupled_step(throttle_cmd, steer_cmd, brake_cmd)
        _ft_steps[0] += 1
        n_steps += 1
        if gravel is not None and gravel.on.any():
            gr_ms += gravel.ms
            gr_nsub = max(gr_nsub, gravel.nsub)
        mark_dirty(hub)
        spray_step(DT, hub)
        pose_prev, pose_cur = pose_cur, capture_pose()
    t_steps = (time.perf_counter() - t_steps) * 1000.0

    # The DRAWN pose is the two newest sim states blended by the leftover
    # debt, so the car advances a little every RENDERED frame instead of by
    # whole 1/60 steps -- one or two per frame at 33 fps, zero or one above
    # 60. Raw stepping plus a wall-time chase camera read as the car shaking
    # against its own camera (field report); the cost of the blend is up to
    # one sim step of visual latency, which nothing here can feel.
    if pose_cur is None:
        pose_cur = capture_pose()
    if pose_prev is None or np.linalg.norm(pose_cur[0] - pose_prev[0]) > 5.0:
        pose_prev = pose_cur          # first frame or a respawn teleport: snap
    a = sim_debt / DT
    p = pose_prev[0] + (pose_cur[0] - pose_prev[0]) * a
    q = nlerp(pose_prev[1], pose_cur[1], a)
    chassis.position.set(*p)
    chassis.quaternion.set(*q)
    for i in range(4):
        lp = pose_prev[2][i][0] + (pose_cur[2][i][0] - pose_prev[2][i][0]) * a
        lq = nlerp(pose_prev[2][i][1], pose_cur[2][i][1], a)
        wheel_rigs[i].position.set(*lp)
        wheel_rigs[i].quaternion.set(*lq)

    # Lamps: only touch the material when the state flips -- needs_update()
    # re-uploads it.
    on = brake_cmd > 0.05
    if brake_mat is not None and on != brake_was:
        brake_was = on
        brake_mat.emissive_intensity = 6.0 if on else 1.0
        brake_mat.needs_update()
    rev = vehicle.gear == tp.PhysxVehicle.Gear.REVERSE
    if reverse_mat is not None and rev != reverse_was:
        reverse_was = rev
        reverse_mat.emissive_intensity = 6.0 if rev else 1.0
        reverse_mat.needs_update()

    # Presentation runs on wall time: the chase lerp's 1-exp(-k*dt) is frame
    # rate independent given the real dt, and snowfall advancing by wall time
    # is what stops the curtain stuttering on frames the sim did not step.
    update_camera(min(wall, 0.1))
    c = active_camera()
    weather(min(wall, 0.1), (c.position.x, c.position.y, c.position.z))
    t_pub = time.perf_counter()
    _pub_n[0] += 1
    if gravel is not None and _pub_n[0] % SOUP_EVERY == 0:
        gravel.publish()      # the soup, before the strips it lies on
    t_gpub = (time.perf_counter() - t_pub) * 1000.0
    for s in strips:
        s.publish()
    t_pub = (time.perf_counter() - t_pub) * 1000.0
    t_rend = time.perf_counter()
    renderer.render(scene, active_camera())
    t_rend = (time.perf_counter() - t_rend) * 1000.0
    if ui is not None:
        ui.render(draw_ui)
    fps_ema += (1.0 / max(time.perf_counter() - t0, 1e-4) - fps_ema) * 0.05
    if FRAMES_LOG:
        _flog.append((time.perf_counter(), wall * 1000.0, n_steps, t_steps, gr_ms, gr_nsub,
                      t_gpub, t_pub, t_rend, vehicle.position.x, vehicle.position.z,
                      vehicle.forward_speed * 3.6,
                      int(gravel.on.sum()) if gravel is not None else 0,
                      gravel.ntris if gravel is not None else 0))


# The first render is what creates the records the vertex exports come from.
renderer.render(scene, camera)
if not GL:
    armed = sum(s.arm(renderer) for s in strips)
    print(f"lane mesh route: "
          f"{'zero-copy CUDA -> Vulkan' if armed == len(strips) else 'mixed / host copy'}"
          f" ({armed}/{len(strips)} armed)")
    if gravel is not None and hasattr(renderer, "set_stable_correspondence"):
        # The patches' soup is re-triangulated every frame (one changed cell
        # shifts every later vertex slot) and always goes the host route, so
        # per-vertex motion history is noise: declare it, or the soup boils.
        renderer.set_stable_correspondence(gravel.mesh, False)
if gravel is not None:
    gravel.publish()      # the soup, before the strips it lies on
for s in strips:
    s.publish()

def pose_visuals():
    p, q = vehicle.position, vehicle.quaternion
    chassis.position.set(p.x, p.y, p.z)
    chassis.quaternion.set(q.x, q.y, q.z, q.w)
    for k in range(4):
        lp, lq = vehicle.wheel_local_pose(k)
        wheel_rigs[k].position.set(lp.x, lp.y, lp.z)
        wheel_rigs[k].quaternion.set(lq.x, lq.y, lq.z, lq.w)


def scripted(spawn_z, beats, eye, look, path, note=""):
    """Drive a fixed input script, park a camera, save the frame.

    The shots the acceptance asks for have to be repeatable and identical
    between lanes -- the same line at the same speed is the whole comparison --
    which a hand-driven pass is not. Each beat is (seconds, throttle, steer).
    `eye` None means the cinematic orbit at CINE_HERO, wherever the car ended.
    Spawn height comes off the road, not off a datum -- SPAWN_X is the bottom of
    a dip and dropping the car from a flat 1.05 m would be a 40 cm fall.
    """
    vehicle.respawn(tp.Vector3(SPAWN_X, ride_y(SPAWN_X, spawn_z), spawn_z), SPAWN_ROT)
    vehicle.gear = tp.PhysxVehicle.Gear.FORWARD
    w_ema[:] = REST_LOAD
    for _ in range(60):
        coupled_step(0.0, 0.0, 0.0)
    t_tr = 0
    for secs, thr, st in beats:
        for _ in range(int(round(secs * 60.0))):
            hub = coupled_step(thr, st, 0.0)
            mark_dirty(hub)
            spray_step(DT, hub)
            weather(DT, eye if eye is not None else (0.0, 2.0, 0.0))
            if TRACE and t_tr % 15 == 0:
                print(f"    t{t_tr / 60.0:5.2f} x{vehicle.position.x:7.2f} "
                      f"v{vehicle.forward_speed * 3.6:6.1f} km/h  "
                      f"sink {np.round(hud['z'] * 1000, 1)} mm  mu {np.round(hud['mu'], 2)}  "
                      f"wr {np.round([vehicle.wheel_angular_speed(i) * R_WHEEL for i in range(4)], 1)}"
                      f"  drag {np.round(hud['drag'], 0)}  tc {tc_cut:4.2f}"
                      + (f"  Fy/W {np.round(gravel.fy, 2)}  H/W {np.round(gravel.thrust, 2)}"
                         f"  nsub {gravel.nsub}" if gravel is not None else ""), flush=True)
            t_tr += 1
    pose_visuals()
    if gravel is not None:
        gravel.publish()      # the soup, before the strips it lies on
    for s in strips:
        s.dirty = True
        s.publish()
    if eye is None:
        p = vehicle.position
        eye, look = cine_pose(np.array([p.x, p.y, p.z]), CINE_HERO)
    # dt = 0 freezes the snowfall for the still, and this is also where the
    # follow box finally lands on the shot's own camera: the WARM loop below
    # renders the same instant sixty times to settle the temporal passes, and a
    # field reporting motion through frames that do not move is one TAA smears.
    weather(0.0, eye)
    camera.position.set(*eye)
    camera.look_at(tp.Vector3(*look))
    # The Vulkan pipeline is temporal (probe GI, denoisers, the upscaler): one
    # render after the sim would capture frame ONE of all of them.
    for _ in range(0 if GL else WARM):
        renderer.render(scene, camera)
    save_shot(path)
    print(f"  {os.path.basename(path)}: v={vehicle.forward_speed * 3.6:5.1f} km/h  "
          f"lane={hud['lane'][0]}  sink={np.round(hud['z'] * 1000, 1)} mm  "
          f"mu={np.round(hud['mu'], 2)}  slip={np.round(hud['slip'], 2)}  "
          f"util={np.round(hud['util'], 2)}  "
          f"Bekker/PhysX={np.round(100 * (hud['bek'] / np.maximum(hud['w'], 1.0) - 1.0), 1)} % "
          f"{note}")
    # Full precision, for before/after regression proofs (the line above rounds).
    _p, _q = vehicle.position, vehicle.quaternion
    print(f"  state: pos=({_p.x!r}, {_p.y!r}, {_p.z!r}) quat=({_q.x!r}, {_q.y!r}, {_q.z!r}, {_q.w!r}) "
          f"v={vehicle.forward_speed!r} load={[float(vehicle.suspension_force(i)) for i in range(4)]}")


if BENCH:
    # Honest wall clock: the whole frame the interactive loop runs, vsync off.
    # With --gravel the car drives the gravel lane (the clay strip otherwise).
    _zg = 0.5 * (GRAVEL_Z[0] + GRAVEL_Z[1])

    def bench_respawn():
        # Each timed pass starts at the spawn end, so 4 s at 0.3 stays on the lane.
        if GRAVEL:
            vehicle.respawn(tp.Vector3(SPAWN_X, ride_y(SPAWN_X, _zg), _zg), SPAWN_ROT)
            w_ema[:] = REST_LOAD

    bench_respawn()
    B_THR, B_STEER = (0.3, 0.0) if GRAVEL else (0.5, 0.15)   # --gravel: stay on the lane
    for _ in range(60):
        hub = coupled_step(B_THR, B_STEER, 0.0)
        mark_dirty(hub)
        spray_step(DT, hub)
        pose_visuals()
        weather(DT)
        if gravel is not None:
            gravel.publish()      # the soup, before the strips it lies on
        for s in strips:
            s.publish()
        renderer.render(scene, camera)
    bench_respawn()
    for _ in range(30):
        coupled_step(0.0, 0.0, 0.0)
    wp.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(240):
        hub = coupled_step(B_THR, B_STEER, 0.0)
        mark_dirty(hub)
        spray_step(DT, hub)
        pose_visuals()
        weather(DT)
        if gravel is not None:
            gravel.publish()      # the soup, before the strips it lies on
        for s in strips:
            s.publish()
        renderer.render(scene, camera)
    wp.synchronize_device(device)
    ms = (time.perf_counter() - t0) * 1000.0 / 240.0
    bench_respawn()
    for _ in range(30):
        coupled_step(0.0, 0.0, 0.0)
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    for _ in range(240):
        coupled_step(B_THR, B_STEER, 0.0)
    wp.synchronize_device(device)
    sim_ms = (time.perf_counter() - t1) * 1000.0 / 240.0
    # Breakdown pass: the gravel lane's own share (synchronised around it).
    bench_respawn()
    for _ in range(30):
        coupled_step(0.0, 0.0, 0.0)
    PHASE["on"], PHASE["gravel"] = True, 0.0
    for _ in range(240):
        coupled_step(B_THR, B_STEER, 0.0)
    PHASE["on"] = False
    g_ms = PHASE["gravel"] * 1000.0 / 240.0
    print(f"bench [{'opengl' if GL else 'vulkan'}, "
          f"{'zero-copy interop' if INTEROP else 'host copy, dirty strips'}]: "
          f"{ms:.2f} ms/frame ({1000.0 / ms:.0f} fps), of which sim+coupling "
          f"{sim_ms:.2f} ms (gravel lane {g_ms:.2f} ms, the rest {sim_ms - g_ms:.2f} ms), "
          f"render+publish {ms - sim_ms:.2f} ms; {cells:,} cells, {len(strips)} strips; "
          f"car at x {vehicle.position.x:.1f} z {vehicle.position.z:.1f} "
          f"lanes {hud['lane']} v {vehicle.forward_speed * 3.6:.1f} km/h"
          + (f"; gravel: MPM step {gravel.ms:.2f} ms at {gravel.nsub} substeps, soup "
             f"{gravel.pub_ms:.2f} ms, {gravel.ntris:,} tris" if gravel is not None else ""))
elif SHOT:
    which = cli_arg("--script", "lanes", str)
    z_mud = 0.5 * (LANE_Z["mud"][0] + LANE_Z["mud"][1])
    z_snow = 0.5 * (LANE_Z["snow"][0] + LANE_Z["snow"][1])
    out = lambda n: os.path.join(OUT_DIR, f"warp_mudsnow_drive_{n}.png")   # noqa: E731

    # Every parked camera is quoted RELATIVE to the road under its subject, so
    # the framings survived the profile unchanged: the shots were composed on a
    # plane, and a plane is what base() is measured from.
    def rig(sx, sz, eye_xyz, look_xyz):
        y0 = float(base(sx, sz))
        return ((eye_xyz[0], y0 + eye_xyz[1], eye_xyz[2]),
                (look_xyz[0], y0 + look_xyz[1], look_xyz[2]))

    # One script per process, one frame each. This used to be forced -- a
    # second save_frame in one run lost the device -- and is not any more (the
    # vertex-interop UAF fix; --film takes thousands of them in one run). It
    # stays one script per process because each of these shots wants the ground
    # UNDRIVEN, and the shots share a yard.
    if which in ("mud", "snow"):
        # Same line, same throttle, both soft lanes: mud walls its trench,
        # snow swallows the wheel and barely rims.
        z = z_mud if which == "mud" else z_snow
        eye, look = rig(-15.5, z, (-13.5, 1.15, z + (5.2 if z < 0 else -5.2)),
                        (-15.5, -0.1, z))
        scripted(z, [(3.2, 0.45, 0.0)], eye, look, out(f"1_{which}_rut"))
    if which in ("spin_mud", "spin_clay", "spin_snow"):
        # Acceptance 2: the same full-throttle launch, mud vs the clay strip.
        # TC off -- unassisted wheelspin is what this shot is about.
        # spin_snow is the spray's snow-side acceptance: the gentle `snow`
        # cruise correctly throws nothing, so the white spray needs this shot.
        tc_on = False
        z = (z_mud if which == "spin_mud" else
             z_snow if which == "spin_snow" else 0.0)
        eye, look = rig(-18.5, z, (-17.0, 1.5, z + (5.0 if z <= 0 else -5.0)),
                        (-18.5, 0.2, z))
        scripted(z, [(2.0, 1.0, 0.0)], eye, look, out(f"2_{which}"))
    if which == "burnout":
        # The clod fix's own acceptance (the mesh->billboard crossover used to
        # sit at 11 m, right on top of CHASE_DIST). Same mud launch as
        # spin_mud, shot from where the CHASE CAMERA stands -- 9.5 m back and a
        # little up -- because 2-5 m proves nothing about a fade at 9.5.
        tc_on = False
        eye, look = rig(-18.5, z_mud, (-18.5 - 9.5 * math.cos(0.30), 1.05 + 9.5 * math.sin(0.30),
                                       z_mud - 2.2), (-17.4, 0.55, z_mud))
        scripted(z_mud, [(2.0, 1.0, 0.0)], eye, look, out("8_burnout_chase"),
                 note="clods at chase distance")
    if which in ("spin_gravel", "gravel", "gravel_rest"):
        if gravel is None:
            print("  --script gravel* needs --gravel")
            raise SystemExit(0)
        z_g = 0.5 * (GRAVEL_Z[0] + GRAVEL_Z[1])
        if which == "spin_gravel":
            # The launch: TC off, floored, on the spread. What this shot is for
            # is the ROOSTER TAIL of stone and the trough opening under the rear
            # wheels -- the grains are the whole subject, so the camera is low
            # and close to the side the spray leaves on.
            tc_on = False
            eye, look = rig(-15.5, z_g, (-11.6, 1.10, z_g - 5.0), (-15.6, 0.30, z_g))
            scripted(z_g, [(1.5, 1.0, 0.0)], eye, look, out("7_spin_gravel"),
                     note="two-way MPM grains")
        elif which == "gravel":
            # The roll-through: TC on, a gentle 3.5 s crawl the length of the
            # spread. Grains PART at the wheel and pool behind it instead of
            # being thrown, which is the half of granular behaviour a launch
            # hides.
            eye, look = rig(-15.0, z_g, (-12.4, 1.35, z_g - 5.0), (-16.0, 0.05, z_g))
            scripted(z_g, [(3.5, 0.28, 0.0)], eye, look, out("7_gravel_roll"),
                     note="slow roll-through")
        else:
            # Before and after, one camera, one process: the undisturbed spread,
            # then the same frame once a pass has been driven through it and the
            # stone has had two seconds to stop moving.
            eye = (-13.2, float(base(-13.2, z_g)) + 2.30, z_g - 6.6)
            look = (-17.5, float(base(-17.5, z_g)) + 0.02, z_g)
            gravel.reset()
            vehicle.respawn(tp.Vector3(SPAWN_X, ride_y(SPAWN_X, -20.0), -20.0),
                            SPAWN_ROT)               # the car parked off frame
            for _ in range(90):
                coupled_step(0.0, 0.0, 0.0)
            camera.position.set(*eye)
            camera.look_at(tp.Vector3(*look))
            weather(0.0, eye)
            for _ in range(0 if GL else WARM):
                renderer.render(scene, camera)
            save_shot(out("7_gravel_before"))
            scripted(z_g, [(3.0, 0.55, 0.0), (2.0, 0.0, 1.0)], eye, look,
                     out("7_gravel_after"), note="the same frame, driven")
    if which == "gravel_take":
        # The two-way gravel lane's own take: a TC-on full-throttle launch, a
        # cruise and a stop, all on the lane, from a close chase camera; every
        # other sim frame saved and encoded at 30 fps (--take-secs, default
        # 8 s), then three stills: the launch frame, a low side view of a rear
        # wheel sunk in the grains, and the ruts behind the stopped car.
        # Frames go to <out-dir>/gravel_take/.
        if gravel is None:
            print("  --script gravel_take needs --gravel")
            raise SystemExit(0)
        import imageio_ffmpeg
        z_g = 0.5 * (GRAVEL_Z[0] + GRAVEL_Z[1])
        fdir = os.path.join(OUT_DIR, "gravel_take")
        os.makedirs(fdir, exist_ok=True)
        for _f in os.listdir(fdir):
            if _f.endswith(".png"):
                os.remove(os.path.join(fdir, _f))
        vehicle.respawn(tp.Vector3(SPAWN_X, ride_y(SPAWN_X, z_g), z_g), SPAWN_ROT)
        vehicle.gear = tp.PhysxVehicle.Gear.FORWARD
        w_ema[:] = REST_LOAD
        for _ in range(60):
            coupled_step(0.0, 0.0, 0.0)
        secs = cli_arg("--take-secs", 8.0, float)
        n_take = int(round(secs * 60.0))

        def take_eye():
            p, q = vehicle.position, vehicle.quaternion
            c = np.array([p.x, p.y, p.z])
            # 6.5 m back, 2.2 m out to the right, 1.9 m up: the right-hand
            # wheels and what they throw are in frame, the car is not a wall.
            eye = c + qrot(q, np.array([-2.2, 1.9, -6.5]))
            tgt = c + qrot(q, np.array([0.0, -0.2, 1.0]))
            return eye, tgt

        eye, tgt = take_eye()
        cam_pos[:], cam_tgt[:] = eye, tgt
        camera.position.set(*cam_pos)
        camera.look_at(tp.Vector3(*cam_tgt))
        for _ in range(0 if GL else 30):
            renderer.render(scene, camera)
        t_start = time.perf_counter()
        launch_still = None
        for k in range(n_take):
            t = k / 60.0
            thr, br = (0.0, 0.0) if t < 0.4 else (1.0, 0.0) if t < 3.0 else \
                (0.3, 0.0) if t < 5.0 else (0.0, 1.0)
            hub = coupled_step(thr, 0.0, br)
            mark_dirty(hub)
            spray_step(DT, hub)
            pose_visuals()
            eye, tgt = take_eye()
            kk = 1.0 - math.exp(-6.0 * DT)
            cam_pos += (eye - cam_pos) * kk
            cam_tgt += (tgt - cam_tgt) * kk
            camera.position.set(*cam_pos)
            camera.look_at(tp.Vector3(*cam_tgt))
            weather(DT, cam_pos)
            if gravel is not None:
                gravel.publish()
            for s in strips:
                s.publish()
            if k % 2:
                continue
            path = os.path.join(fdir, f"f{k // 2:05d}.png")
            renderer.save_frame(scene, camera, path)
            if k == 76:
                launch_still = path
            if k % 30 == 0:
                print(f"    take t{t:5.2f} x{vehicle.position.x:7.2f} "
                      f"v{vehicle.forward_speed * 3.6:6.1f} km/h  sink {np.round(hud['z'] * 1000, 0)} mm  "
                      f"mu {np.round(hud['mu'], 2)}  wr {np.round([vehicle.wheel_angular_speed(i) * R_WHEEL for i in range(4)], 1)}"
                      f"  tc {tc_cut:4.2f}  Fy/W {np.round(gravel.fy, 2)}  nsub {gravel.nsub}", flush=True)
        print(f"  take: {n_take} frames in {time.perf_counter() - t_start:.1f} s")
        import shutil
        if launch_still:
            shutil.copy(launch_still, out("9_gravel_launch_chase"))
            print(f"  wrote {out('9_gravel_launch_chase')}")
        mp4 = os.path.join(OUT_DIR, "warp_mudsnow_drive_9_gravel_take.mp4")
        encode_png_sequence(os.path.join(fdir, "f%05d.png"), mp4, 30, crf=19,
                            ffmpeg=imageio_ffmpeg.get_ffmpeg_exe())
        print(f"  wrote {mp4}")
        # Stills off the stopped car. A few frames of rest first so the grains
        # stop moving, then each camera warmed for the temporal passes.
        for _ in range(45):
            hub = coupled_step(0.0, 0.0, 1.0)
            mark_dirty(hub)
            spray_step(DT, hub)
        pose_visuals()
        if gravel is not None:
            gravel.publish()
        for s in strips:
            s.dirty = True
            s.publish()
        p, q = vehicle.position, vehicle.quaternion
        c = np.array([p.x, p.y, p.z])
        lp, _ = vehicle.wheel_local_pose(2)          # rear right
        hub_rr = c + qrot(q, np.array([lp.x, lp.y, lp.z]))
        right = qrot(q, np.array([1.0, 0.0, 0.0]))
        fwd = qrot(q, np.array([0.0, 0.0, 1.0]))
        g0 = float(base(hub_rr[0], hub_rr[2]))
        shots = [("9_gravel_wheel_low",
                  hub_rr + right * 1.55 - fwd * 0.55 + np.array([0.0, g0 + 0.10 - hub_rr[1], 0.0]),
                  hub_rr + np.array([0.0, g0 - 0.02 - hub_rr[1], 0.0])),
                 ("9_gravel_ruts_behind",
                  c - fwd * 7.5 + right * 1.2 + np.array([0.0, float(base(c[0] - 7.5, c[2])) + 1.55 - c[1], 0.0]),
                  c - fwd * 2.0 + np.array([0.0, float(base(c[0] - 2.0, c[2])) - c[1], 0.0]))]
        for name, e, lk in shots:
            weather(0.0, e)
            camera.position.set(*e)
            camera.look_at(tp.Vector3(*lk))
            for _ in range(0 if GL else WARM):
                renderer.render(scene, camera)
            save_shot(out(name))
        print(f"  gravel take: stopped at x {vehicle.position.x:.2f}, sink "
              f"{np.round(hud['z'] * 1000, 1)} mm, Fy/W {np.round(gravel.fy, 3)}")
    if which == "crest":
        # The road's own frame: the car ON the rise out of the spawn dip, shot
        # from the clay strip across the lanes so the chassis PITCH is a
        # silhouette against the grade behind it rather than something you take
        # on trust. 2.6 s at 0.45 in mud lands it at x ~ -13, which is the
        # steepest part of the climb (5.7 %), and the key is beyond the lanes
        # so the ruts it just cut throw their shadows at the lens.
        # Both ends of the shot are quoted against THEIR OWN ground, which is
        # the point of it: the camera stands 0.9 m over the crest at x = -3.5
        # and the car is 10 m back and 0.45 m lower, so the road between them
        # is a skyline and the pitch is read against it.
        eye = (-3.5, float(base(-3.5, z_mud - 4.5)) + 0.90, z_mud - 4.5)
        look = (-13.0, float(base(-13.0, z_mud)) + 0.55, z_mud)
        scripted(z_mud, [(2.6, 0.45, 0.0)], eye, look, out("6_crest"))
    if which == "cine":
        # The look pass's own acceptance frame: the cinematic camera, parked at
        # the phase of its orbit that crosses the key, on a mud lane with a
        # trail of its own ruts leading back out of frame. Nothing here is a
        # special camera -- cine_pose is the same function C flies.
        scripted(z_mud, [(5.5, 0.45, 0.0)], None, None, out("4_cine"))
    if which == "pan":
        # Acceptance 3: no flat grey anywhere in a 360 pan from the driver's
        # seat. One heading per process, so this is run four times with --yaw.
        yaw = math.radians(cli_arg("--yaw", 0.0, float))
        eye = np.array([-6.0, float(base(-6.0, 0.0)) + 1.35, 0.0])
        scripted(0.0, [(0.2, 0.0, 0.0)], eye,
                 eye + np.array([math.sin(yaw) * 20.0, -1.2, math.cos(yaw) * 20.0]),
                 out(f"5_pan_{int(round(math.degrees(yaw))):03d}"))
    if which == "diag":
        # Tuning telemetry: a 0.45-throttle launch on each soft lane, per-wheel
        # state every half second. No frame saved.
        for lane_name in ("mud", "snow"):
            z_lane = 0.5 * (LANE_Z[lane_name][0] + LANE_Z[lane_name][1])
            vehicle.respawn(tp.Vector3(SPAWN_X, ride_y(SPAWN_X, z_lane), z_lane),
                            SPAWN_ROT)
            dig[:] = 1.0
            tc_cut = 1.0
            w_ema[:] = REST_LOAD
            for _ in range(60):
                coupled_step(0.0, 0.0, 0.0)
            print(f"  {lane_name}: t  v_kmh | W_kN | omega*r m/s | dig | sink_mm | tc")
            _tot = []
            for k in range(int(4.0 * 60)):
                coupled_step(0.45, 0.0, 0.0)
                _tot.append(sum(vehicle.suspension_force(i) for i in range(4)))
                if k % 30 == 0:
                    w = [vehicle.suspension_force(i) / 1000 for i in range(4)]
                    wr = [vehicle.wheel_angular_speed(i) * R_WHEEL for i in range(4)]
                    print(f"  {k / 60.0:4.1f} {vehicle.forward_speed * 3.6:6.1f} | "
                          f"W {np.round(w, 2)} | wr {np.round(wr, 1)} | "
                          f"dig {np.round(dig, 2)} | z {np.round(hud['z'] * 1000, 0)} | "
                          f"{tc_cut:4.2f}")
            # Total normal load, which on a road is no longer the static 14.7 kN
            # (see "What a grade does to the suspension damper" at the top of
            # this file). Printed as mean/extremes/sd so the difference between
            # a car breathing with the road and a coupling loop RINGING is a
            # number: sd is a tenth of the mean here, and the per-frame trace is
            # monotone, not oscillatory.
            _w = np.array(_tot[-120:]) / 1000.0
            print(f"    total load over the last 2 s: mean {_w.mean():5.2f} kN  "
                  f"[{_w.min():5.2f}, {_w.max():5.2f}]  sd {_w.std():4.2f} "
                  f"(static 14.72)")
    if which == "apron":
        # The road profile's own acceptance, in numbers: drive the length of the
        # yard 20 m OFF both lanes, where every wheel is on the rigid fallback,
        # and print the ride height above RIGID_Y + base(x, z). If the trimesh
        # collider were missing (or cooked from different vertices) the car
        # would be a metre down on the catch plane and this column would read
        # zero; if the apron ignored the profile it would drift by 0.5 m. It
        # holds to a centimetre, and the centimetre IS the chatter -- the bumps
        # are felt off-lane because the collider has them.
        vehicle.respawn(tp.Vector3(SPAWN_X, ride_y(SPAWN_X, 20.0), 20.0), SPAWN_ROT)
        for _ in range(90):
            coupled_step(0.0, 0.0, 0.0)
        print("   t     x      y   y-apron  lane   (apron = RIGID_Y + base)")
        for k in range(int(9.0 * 60)):
            coupled_step(0.35, 0.0, 0.0)
            if k % 45 == 0:
                p = vehicle.position
                print(f"  {k/60.0:4.1f} {p.x:7.2f} {p.y:6.3f} "
                      f"{p.y - (RIGID_Y + float(base(p.x, p.z))):7.3f}  "
                      f"{hud['lane']}  v={vehicle.forward_speed*3.6:5.1f}")
    if which == "stuck":
        # The mud skill loop, measured, with TC off (TC exists to prevent
        # exactly this): floor it from rest (the wheels dig and the car goes
        # nowhere), then ease off to a crawl throttle (the dig relaxes and it
        # creeps out). No frame is saved -- this one is numbers.
        tc_on = False
        vehicle.respawn(tp.Vector3(SPAWN_X, ride_y(SPAWN_X, z_mud), z_mud), SPAWN_ROT)
        w_ema[:] = REST_LOAD
        for _ in range(60):
            coupled_step(0.0, 0.0, 0.0)
        x0 = vehicle.position.x
        for _ in range(int(4.0 * 60)):
            coupled_step(1.0, 0.0, 0.0)
        x1 = vehicle.position.x
        d1, dig1, z1 = x1 - x0, float(np.mean(hud["dig"])), float(np.mean(hud["z"]) * 1000)
        for _ in range(int(6.0 * 60)):
            coupled_step(0.35, 0.0, 0.0)
        x2 = vehicle.position.x
        d2, dig2 = x2 - x1, float(np.mean(hud["dig"]))
        print(f"  stuck test (mud): 4 s full throttle -> {d1:5.2f} m travelled, "
              f"dig {dig1:.2f}, sink {z1:.0f} mm")
        print(f"                    6 s at 0.35       -> {d2:5.2f} m travelled, "
              f"dig {dig2:.2f}, v {vehicle.forward_speed * 3.6:4.1f} km/h")
        print(f"  drag/wheel now {np.round(hud['drag'], 0)} N, "
              f"traction ceiling {np.round(hud['mu'] * hud['w'], 0)} N")
    if which == "lap":
        # Acceptance 3: a circle in mud, twice. Full lock at a walking pace
        # turns inside the lane; the readout is the depth of the wheel below
        # the LOCAL surface, which is a fresh sinkage on lap 1 and almost
        # nothing on lap 2 because the grade under the rut has not moved.
        def rut_depth():
            p = vehicle.position
            lp, _ = vehicle.wheel_local_pose(3)
            hubw = np.array([p.x, p.y, p.z]) + qrot(vehicle.quaternion,
                                                    np.array([lp.x, lp.y, lp.z]))
            t = terrain["mud"]
            x = torch.tensor([[hubw[0]]], device=device, dtype=torch.float32)
            y = torch.tensor([[-hubw[2]]], device=device, dtype=torch.float32)
            surf = float(t.heights(x, y)[0, 0])
            return (surf - (hubw[1] - R_WHEEL)) * 1000.0

        vehicle.respawn(tp.Vector3(-14.0, ride_y(-14.0, z_mud), z_mud), SPAWN_ROT)
        w_ema[:] = REST_LOAD
        for _ in range(60):
            coupled_step(0.0, 0.0, 0.0)
        lap = []
        for k in range(int(24.0 * 60)):
            mark_dirty(coupled_step(0.30, 1.0, 0.0))
            if k % 30 == 0:
                lap.append((k / 60.0, rut_depth(), float(hud["z"][3] * 1000.0)))
        pose_visuals()
        if gravel is not None:
            gravel.publish()      # the soup, before the strips it lies on
        for s in strips:
            s.dirty = True
            s.publish()
        y0 = float(base(-15.5, z_mud))
        camera.position.set(-14.0, y0 + 6.0, z_mud - 6.5)
        camera.look_at(tp.Vector3(-15.5, y0 - 0.1, z_mud))
        for _ in range(0 if GL else WARM):
            renderer.render(scene, camera)
        save_shot(out("3_second_lap"))
        print("  circle in mud: t, wheel below LOCAL surface (mm), Bekker z_eq (mm)")
        for t_s, d, z in lap[::4]:
            print(f"    {t_s:5.1f}s  {d:6.1f}  {z:6.1f}")
elif FILM:
    # --- the film -------------------------------------------------------------
    # ONE process, ONE continuously advancing sim, a list of TAKES. The ruts
    # accumulating across the whole film ARE the story, so nothing is ever reset:
    # what the second-lap take rides in is what the takes before it cut, and the
    # dig-in take digs into ground the drive already softened. A take is (camera
    # rig, duration, scripted inputs); between takes the camera hard-cuts.
    #
    # Two things a cut costs, and both are paid here:
    #   * the temporal stack. Probe GI, the reflection/AO denoisers and TAA/FSR
    #     all carry history; frame one after a cut is frame one of all of them.
    #     LEAD frames are rendered and thrown away after every cut.
    #   * continuity. The sim does not stop for the cut -- the lead frames step
    #     the car too -- and TRANSIT segments (capture=False) are the film's
    #     off-screen time: the sim runs, nothing is saved, and the car gets from
    #     where the last take left it to where the next one starts. No
    #     teleporting, ever, and no slowmo: every take is DT=1/60 real sim time.
    #
    # Frames land in --film-dir as PNGs, one take at a time, each take encoded to
    # its own segment and its PNGs deleted before the next take starts (a 48 s
    # film at 1600x900 is ~7 GB of PNG otherwise), then the segments are
    # concatenated. --takes renders only the named takes (the others still RUN,
    # as transits, so the route and the ruts are identical) -- that is the
    # dailies loop. --dry runs the whole route with no renderer at all, which is
    # how the choreography below was written.
    import subprocess

    FFMPEG = find_ffmpeg()
    FILM_DIR = cli_arg("--film-dir", os.path.join(OUT_DIR, "film"), str)
    LEAD = cli_arg("--lead", 30, int)          # frames discarded after every cut
    ONLY = [s.strip() for s in cli_arg("--takes", "", str).split(",") if s.strip()]
    DRY = "--dry" in sys.argv                  # route only: sim, no render
    PROBE = "--probe" in sys.argv
    SHEET_EVERY = 120                          # a contact-sheet thumb every 2 s

    Z_MUD = 0.5 * (LANE_Z["mud"][0] + LANE_Z["mud"][1])
    Z_SNOW = 0.5 * (LANE_Z["snow"][0] + LANE_Z["snow"][1])
    Z_GRV = 0.5 * (GRAVEL_Z[0] + GRAVEL_Z[1])

    def road(x, z, dy=0.0):
        """Camera height quoted against the ROAD under it, never a datum."""
        return float(base(x, z)) + dy

    def car_frame():
        """Car position and its flattened forward / side axes.

        `side` is the car's +z when it is heading +x -- the mud lane is at +z, so
        `side` is which way to look for it.
        """
        p, q = vehicle.position, vehicle.quaternion
        c = np.array([p.x, p.y, p.z])
        f = qrot(q, np.array([0.0, 0.0, 1.0]))
        f[1] = 0.0
        f = f / max(float(np.linalg.norm(f)), 1e-6)
        return c, f, np.array([-f[2], 0.0, f[0]])

    def rut_state(wheel=3):
        """(rut floor below the VIRGIN road, wheel below the local surface), mm.

        The second-lap claim measured. The floor is the surface the grid holds
        now against `base()`, the road as it was cut -- so a first pass drives it
        from 0 down to the Bekker sinkage, and a second pass over the same ground
        is riding on a floor that is already there. If soft ground simply
        remembered nothing, the floor would double on the second pass; it does
        not, and the difference between the two numbers is why the car rides
        higher in its own tracks than it does in fresh mud.
        """
        p = vehicle.position
        lp, _ = vehicle.wheel_local_pose(wheel)
        hubw = np.array([p.x, p.y, p.z]) + qrot(vehicle.quaternion,
                                                np.array([lp.x, lp.y, lp.z]))
        n = lane_at(hubw[0], hubw[2])
        if n is None:
            return float("nan"), float("nan")
        t = terrain[n]
        x = torch.tensor([[hubw[0]]], device=device, dtype=torch.float32)
        y = torch.tensor([[-hubw[2]]], device=device, dtype=torch.float32)
        surf = float(t.heights(x, y)[0, 0])
        return ((surf - float(base(hubw[0], hubw[2]))) * 1000.0,
                (surf - (hubw[1] - R_WHEEL)) * 1000.0)

    def rut_log(tag):
        floor, into = rut_state()
        return (f"rut floor {floor:6.1f} mm below the virgin road, wheel "
                f"{into:5.1f} mm into it ({tag})")

    def film_step(thr, st, br):
        """One sim frame with the visuals posed -- everything frame() does except
        the camera, the UI and the render."""
        global brake_was
        hub = coupled_step(thr, st, br)
        mark_dirty(hub)
        spray_step(DT, hub)
        pose_visuals()
        on = br > 0.05
        if brake_mat is not None and on != brake_was:
            brake_was = on
            brake_mat.emissive_intensity = 6.0 if on else 1.0
            brake_mat.needs_update()

    if PROBE:
        # DE-RISK, before a single frame of choreography. The header of this file
        # used to say that two save_frame calls in one Vulkan run lose the
        # device; that note predates the vertex-interop UAF fix, and a film is
        # thousands of them. Prove it in THIS scene, with the interop armed --
        # a probe in an empty scene would prove nothing about the lane strips.
        os.makedirs(FILM_DIR, exist_ok=True)
        t0 = time.perf_counter()
        for k in range(10):
            film_step(0.6, 0.0, 0.0)
            c, f, s = car_frame()
            eye = c - f * 8.0 + np.array([0.0, 2.0, 0.0]) + s * 2.0
            camera.position.set(*eye)
            camera.look_at(tp.Vector3(*(c + np.array([0.0, 0.4, 0.0]))))
            weather(DT, eye)
            for st in strips:
                st.publish()
            path = os.path.join(FILM_DIR, f"probe_{k:02d}.png")
            renderer.save_frame(scene, camera, path)
            print(f"  probe {k:2d}: {os.path.getsize(path) / 1e6:5.2f} MB  "
                  f"x={c[0]:6.2f}  v={vehicle.forward_speed * 3.6:5.1f} km/h  "
                  f"{(time.perf_counter() - t0) * 1000.0 / (k + 1):6.1f} ms/frame",
                  flush=True)
        print("probe: 10 repeated save_frame calls in one Vulkan run -- survived")
        sys.exit(0)

    # --- the route ------------------------------------------------------------
    # Steering sign: +1 is a left turn, and the car spawns heading +x, so +1
    # swings it toward -z (the snow lane) and -1 toward +z (the mud lane).
    UP = np.array([0.0, 1.0, 0.0])
    M = {}                    # what an `enter` hook leaves for its take's rig

    # Nothing in this film is a hand-written throttle number held for N seconds.
    # The first cut of it was, and it drove into the next county: the clay strip
    # has grip, and full throttle puts the car at 63 km/h in three seconds --
    # the whole 48 m yard is gone before a beat lands. Worse, the SAME number is
    # a different journey on every lane, which is the demo's entire point. So
    # every take that has to BE somewhere drives to WAYPOINTS at a target speed
    # instead, and the soil decides what that costs. Still one fixed script per
    # run: no input, no randomness, the same run twice is the same film twice.
    def pilot_fn(wps, v_kmh, loop=False, arrive=5.0, halt=False, gain=1.7):
        """A waypoint follower: steer at the next point, hold a speed."""
        st = {"i": 0, "dist": 1e9, "arrived": False}

        def go(t):
            c, f, s = car_frame()
            last = st["i"] >= len(wps) - 1
            d = np.array([wps[st["i"]][0], 0.0, wps[st["i"]][1]]) - c
            d[1] = 0.0
            st["dist"] = float(np.linalg.norm(d))
            if st["dist"] < arrive:
                if loop:
                    st["i"] = (st["i"] + 1) % len(wps)
                elif not last:
                    st["i"] += 1
                else:
                    st["arrived"] = True
            d = d / max(st["dist"], 1e-6)
            # Signed angle from the car's nose to the waypoint. `side` is the
            # car's right, and positive steer is a LEFT turn, hence the minus.
            err = math.atan2(float(np.dot(d, s)), float(np.dot(d, f)))
            want = 0.0 if (halt and st["arrived"]) else v_kmh
            v = vehicle.forward_speed * 3.6
            return (max(0.0, min(1.0, 0.12 * (want - v))),
                    max(-1.0, min(1.0, -gain * err)),
                    max(0.0, min(1.0, 0.10 * (v - want - 3.0))))

        go.state = st
        return go

    def transit(name, wps, v=24.0, secs=40.0, arrive=5.0, halt=True):
        """Off-screen time: the sim runs, the camera is not there. This is how
        the car gets from the end of one take to the start of the next without a
        teleport -- and the ruts it cuts on the way are as real as any other.

        It ends on ARRIVAL, not on a clock: `arrive` metres short of the last
        waypoint, which is why the last two waypoints of every transit are a
        LINE -- the stop lands on that line, pointing along it, and the take
        that follows knows where its subject is without being told.
        """
        go = pilot_fn(wps, v, arrive=arrive, halt=halt)
        return dict(name=name, capture=False, secs=secs, drive=go,
                    done=lambda: go.state["arrived"] and (not halt or
                                                          abs(vehicle.forward_speed) < 0.4))

    # The money shot is a RETRACE, not a circle. The brief asked for a circle in
    # the mud and the car riding higher on lap two; the car cannot do it. Minimum
    # turn radius is about 4.5 m (2.7 m wheelbase, ~33 degrees of lock) and the
    # mud lane is 8 m wide, so any circle the car can actually drive puts half of
    # its ring on the clay strip and the apron -- and a second lap that is half
    # on rigid ground is not the claim. Driven the other way, the SAME claim is
    # exact: the wheels come back down the lane in the two ruts they cut on the
    # way up (a car retracing its own centreline puts each wheel in the other
    # side's track), and rut_state() reads the rut floor under the wheel on both
    # passes. Fresh mud gives the full Bekker sinkage; the second pass is riding
    # on ground the first pass already pushed down.
    MUD_LINE = 6.0                    # the mud lane's centreline, both passes
    retrace_pass = pilot_fn([(-30.0, MUD_LINE)], 17.0)
    retrace_out = pilot_fn([(-26.0, 10.6)], 17.0)     # out over its own berm
    # The gravel take's line: a waypoint the take never reaches, so the pilot
    # holds the lane dead straight for the whole shot instead of arriving and
    # hunting the point it just passed on camera.
    gravel_pass = pilot_fn([(-44.0, Z_GRV)], 34.0)

    def enter_orbit(c, f, s):
        """Start the orbit behind the car, wherever the car ended up."""
        M["a"] = math.atan2(f[2], f[0]) + math.pi + 0.45

    TAKES = [
        # 1. Cold open. No car anywhere in frame: 5 s of virgin mud, the key
        #    raking across it from beyond the lane and the snow drifting through.
        #    The push-in is a dolly along the lane, not a zoom.
        #    Aimed DOWN about 15 degrees: at eye height on a road this flat the
        #    horizon eats the frame and the lane becomes a strip, and the subject
        #    of this shot is the ground.
        dict(name="open", secs=5.0, fov=36.0, drive=lambda t: (0.0, 0.0, 1.0),
             rig=lambda t, c, f, s: ((-8.0 + 0.9 * t, road(-8.0 + 0.9 * t, 3.4, 1.55), 3.4),
                                     (-2.0 + 0.9 * t, road(-2.0 + 0.9 * t, 7.2, -0.35), 7.2)),
             vf="fade=t=in:st=0:d=1.2"),

        # 2. The launch, on the clay strip: the one lane with grip. Full throttle
        #    out of the spawn dip, a lift and a dab of brake (the lamps flare),
        #    then back on it over the crest. The camera is low on the snow side
        #    looking ACROSS the lanes into the key, dollying gently the way the
        #    car goes so the pass has some length to it. Six seconds, not eight:
        #    at 60 km/h the clay strip is 48 m of yard and then it is gone.
        dict(name="launch", secs=6.0, fov=42.0,
             drive=lambda t: ((1.0, 0.0, 0.0) if t < 2.4 else
                              (0.0, 0.0, 0.75) if t < 3.3 else (0.32, 0.0, 0.0)),
             #  Aimed at the wheels, not the roof: this road is flat enough that
             #  a lens level with the car puts the horizon through the middle of
             #  the frame and gives half of it to an empty sky.
             rig=lambda t, c, f, s: ((-1.5 + 0.5 * t, road(-1.5, -6.2, 1.50), -6.2),
                                     c + UP * 0.10), lag=6.0),

        # Off screen: out onto the apron, back down the yard beyond the mud lane
        # (z = 14 is apron then, so the loop cuts no rut it would then drive on)
        # and around onto the mud lane's centreline, stopped, pointing up it.
        # With --gravel, z = 14 is the SPREAD, and that loop would plough the
        # take's subject off screen -- so the road goes through the take
        # instead: the same east-end loop onto the gravel lane's graded base
        # (its Bekker carve is off, the particles are the surface, and east of
        # the spread there are no particles to disturb), stopped short of the
        # virgin stone. The take launches over it, and a hook around the west
        # end lands the car on the mud line the next take needs either way.
        *([transit("to_mud", [(34.0, 8.0), (36.0, 16.0), (-26.0, 14.0),
                              (-27.0, 6.0), (-14.0, MUD_LINE)], v=26.0)]
          if not GRAVEL else
          [transit("to_gravel", [(34.0, 8.0), (36.0, 16.0), (24.0, Z_GRV),
                                 (4.0, Z_GRV), (-4.0, Z_GRV)], v=24.0),

           # 2g. The grains. TC off, floored west over the spread: the bed is
           #     MLS-MPM stone over the same Bekker road as every other lane,
           #     the wheels enter the solver as kinematic spheres, and the
           #     subject is what aggregate does that soil does not -- the
           #     trough opening under a spinning wheel, the fan of thrown
           #     stone, the pile that will not hold a wall. The mud take's
           #     trailing rig, lower and tighter, offset to the LANE side:
           #     the only near post line on this half of the yard is the mud
           #     shoulder's at z = 10.9, and any camera south of it shoots the
           #     whole take through the posts (tried; it does). From inside
           #     the corridor nothing stands between the lens and the stone.
           dict(name="gravel", secs=7.0, fov=42.0, tc=False,
                drive=lambda t: (gravel_pass(t) if t < 5.2 else
                                 (0.0, gravel_pass(t)[1], 0.5)),
                rig=lambda t, c, f, s: (c - f * 6.0 - s * 2.4 + UP * 0.75,
                                        c + f * 1.0 + UP * 0.25), lag=3.5),

           transit("to_mud", [(-33.0, 10.0), (-29.0, 5.5), (-14.0, MUD_LINE)],
                   v=20.0)]),

        # 3. The mud run, up the centreline. From rest: the pilot floors it, the
        #    mud will not take it, TC catches the scrub and feeds it back in --
        #    the tc column below is that argument in one number. Low trailing
        #    camera, offset a wheel-track to the side the key is NOT on, so the
        #    two ruts it just cut run from the lens back to the car with their
        #    own shadow in them. This is the pass the money shot rides back down.
        dict(name="mud", secs=9.0, fov=44.0,
             drive=pilot_fn([(30.0, MUD_LINE)], 22.0),
             rig=lambda t, c, f, s: (c - f * 7.0 - s * 2.4 + UP * 1.05,
                                     c + f * 1.0 + UP * 0.35), lag=3.5,
             log=lambda: rut_log("pass 1, fresh mud")),

        # Off screen: across to the top of the snow lane, pointing back down it.
        transit("to_snow", [(28.0, 2.0), (30.0, -8.0), (22.0, -6.0), (10.0, -6.0)],
                v=24.0),

        # 4. The snow lane: the same car, the same pilot, a lane that swallows
        #    the wheel instead of walling a trench (sink reads ~80 mm here
        #    against mud's ~55). Static low camera BEYOND the lane so the key is
        #    behind the car; it drives past and away, and the shot ends on the
        #    prints rather than on the car.
        dict(name="snow", secs=7.0, fov=42.0,
             drive=pilot_fn([(-26.0, -6.0)], 20.0),
             #  Eight metres off the lane, not four: at four the car crosses the
             #  lens as a white wall and the prints -- the subject -- are behind
             #  it. The pass wants to fit in the frame.
             rig=lambda t, c, f, s: ((3.0, road(3.0, -14.5, 1.05), -14.5),
                                     c + UP * 0.4), lag=5.0),

        # Off screen: the long way round the far end and back onto the mud lane's
        # centreline -- pointing DOWN it this time, and still rolling (halt=False)
        # so the money shot opens on a car already in its tracks.
        transit("to_retrace", [(-16.0, -11.0), (-18.0, -16.0), (26.0, -16.0),
                               (34.0, -8.0), (36.0, 4.0), (28.0, MUD_LINE),
                               (17.0, MUD_LINE)], v=24.0, halt=False),

        # 5. The physics money shot: back down the lane in the two ruts the
        #    mud take cut, then out ACROSS them -- up over its own berm and
        #    away. The camera orbits the car so the trench reads as a trench.
        #    The rut number in the log is the whole claim, and it is the same
        #    number the mud take printed on fresh ground.
        dict(name="retrace", secs=10.0, fov=40.0,
             drive=lambda t: (retrace_pass if t < 7.0 else retrace_out)(t),
             enter=enter_orbit,
             rig=lambda t, c, f, s: (c + np.array([math.cos(M["a"] + 0.14 * t) * 11.0,
                                                   3.4,
                                                   math.sin(M["a"] + 0.14 * t) * 11.0]),
                                     c + UP * 0.45), lag=3.0,
             log=lambda: rut_log("pass 2, its own ruts")),

        # Off screen: the long way round the apron again, back to the BOTTOM of
        # the mud lane on a line nothing has driven, stopped and pointing up it.
        # The dig-in take needs fresh ground under the wheels and the whole lane
        # in front of it: it buries the car, and then it has to crawl out.
        transit("to_dig", [(-27.0, 11.0), (-32.0, 5.0), (-28.0, 0.5), (-20.0, 2.0),
                           (-16.0, 9.2), (-2.0, 9.2)], v=20.0),

        # 6. TC OFF -- the X moment. Four seconds of full throttle with nothing
        #    between the engine and the mud: the wheels dig to where the motion
        #    resistance passes the traction ceiling and the car stops going
        #    anywhere. Hold the beat with the throttle shut (no slowmo, the CAMERA
        #    holds), then 0.35 and it creeps out of its own hole. Close, side-on,
        #    on the side the key is BEYOND, so the thrown soil is backlit.
        dict(name="dig", secs=10.0, fov=40.0, tc=False,
             drive=lambda t: ((1.0, 0.0, 0.0) if t < 4.0 else
                              (0.0, 0.0, 0.0) if t < 5.5 else (0.35, 0.0, 0.0)),
             enter=lambda c, f, s: M.update(
                 d=(s if (c + s * 5.5)[2] < c[2] else -s), p=c),
             rig=lambda t, c, f, s: (M["p"] + M["d"] * (5.6 - 0.12 * t) + UP * 0.62,
                                     c + UP * 0.45), lag=5.0,
             log=lambda: f"dig {np.mean(hud['dig']):4.2f}  sink {np.mean(hud['z']) * 1000:5.1f} mm  "
                         f"util {np.mean(hud['util']):4.2f}"),

        # 7. Outro: away up the rise into the haze, camera down IN the ruts. The
        #    fade is on this segment so the concat stays a stream copy.
        dict(name="outro", secs=5.0, fov=44.0,
             drive=pilot_fn([(24.0, 7.4)], 26.0),
             enter=lambda c, f, s: M.update(o=c, of=f),
             rig=lambda t, c, f, s: (M["o"] - M["of"] * 7.0 + UP * 0.38,
                                     M["o"] + M["of"] * 12.0 + UP * 0.7),
             vf="fade=t=out:st=3.4:d=1.6"),
    ]

    # --- run it ---------------------------------------------------------------
    frames_dir = os.path.join(FILM_DIR, "frames")
    sheet_dir = os.path.join(FILM_DIR, "sheet")
    if not DRY:
        for d in (FILM_DIR, frames_dir, sheet_dir):
            os.makedirs(d, exist_ok=True)
    segments = []
    saved = 0
    for _ in range(60):        # the car settles onto its springs before frame one
        coupled_step(0.0, 0.0, 0.0)
    t_start = time.perf_counter()
    print(f"film: {sum(t['secs'] for t in TAKES if t.get('capture', True)):.0f} s in "
          f"{sum(1 for t in TAKES if t.get('capture', True))} takes, "
          f"{sum(t['secs'] for t in TAKES if not t.get('capture', True)):.0f} s of transit, "
          f"lead {LEAD} frames/cut -> {FILM_DIR}")

    for take_no, take in enumerate(TAKES):
        name, secs = take["name"], take["secs"]
        capture = take.get("capture", True) and not DRY and (not ONLY or name in ONLY)
        tc_on = take.get("tc", True)
        n = int(round(secs * 60.0))
        lead = LEAD if capture else 0
        out_dir = os.path.join(frames_dir, name)
        if capture:
            if os.path.isdir(out_dir):
                for f_old in os.listdir(out_dir):
                    os.remove(os.path.join(out_dir, f_old))
            else:
                os.makedirs(out_dir)
            camera.fov = take.get("fov", 45.0)
            camera.update_projection_matrix()
        c0, f0, s0 = car_frame()
        if take.get("enter"):
            take["enter"](c0, f0, s0)
        first = True
        t_take = time.perf_counter()
        for k in range(-lead, n):
            t = max(k, 0) / 60.0
            thr, st, br = take["drive"](t)
            film_step(thr, st, br)
            c, f, s = car_frame()
            if take.get("tick"):
                take["tick"](t, c)
            if take.get("rig") is not None:
                # Rigs that follow the car are LAGGED, not glued to it: the
                # chassis is a suspension away from smooth. `first` snaps -- it
                # is the cut, and the lead frames absorb it.
                eye, tgt = take["rig"](t, c, f, s)
                eye, tgt = np.asarray(eye, float), np.asarray(tgt, float)
                if first or take.get("lag") is None:
                    cam_pos[:], cam_tgt[:] = eye, tgt
                else:
                    kk = 1.0 - math.exp(-take["lag"] * DT)
                    cam_pos += (eye - cam_pos) * kk
                    cam_tgt += (tgt - cam_tgt) * kk
                camera.position.set(*cam_pos)
                camera.look_at(tp.Vector3(*cam_tgt))
            first = False
            weather(DT, cam_pos)
            if capture:
                if gravel is not None:
                    gravel.publish()      # the soup, before the strips it lies on
                for strip in strips:
                    strip.publish()
                if k >= 0:
                    path = os.path.join(out_dir, f"f{k:05d}.png")
                    renderer.save_frame(scene, camera, path)
                    saved += 1
            if k >= 0 and k % 60 == 0:
                extra = take["log"]() if take.get("log") else ""
                print(f"  [{name:6s} {t:4.1f}s] x{c[0]:7.2f} z{c[2]:6.2f} "
                      f"hdg{math.degrees(math.atan2(f[2], f[0])):7.1f}  "
                      f"v{vehicle.forward_speed * 3.6:6.1f} km/h  "
                      f"lane {str(hud['lane'][0] or 'rigid'):5s} "
                      f"sink{np.mean(hud['z']) * 1000:5.1f}mm dig{np.mean(hud['dig']):4.2f} "
                      f"slip{np.mean(np.abs(hud['slip'])):5.2f} tc{tc_cut:4.2f} {extra}",
                      flush=True)
            if k >= 0 and take.get("done") and take["done"]():
                print(f"  [{name:6s} {t:4.1f}s] arrived  x{c[0]:7.2f} z{c[2]:6.2f} "
                      f"hdg{math.degrees(math.atan2(f[2], f[0])):7.1f}", flush=True)
                break
        if capture:
            # Named by the take's place in the ROUTE, not by this run's order,
            # so re-rendering one take (--takes) overwrites its own segment and
            # the concat below reassembles the whole film around it.
            seg = os.path.join(FILM_DIR, f"seg_{take_no:02d}_{name}.mp4")
            encode_png_sequence(os.path.join(out_dir, "f%05d.png"), seg, 60, crf=18,
                                preset="slow", vf=take.get("vf"), ffmpeg=FFMPEG)
            segments.append(seg)
            for f_old in os.listdir(out_dir):
                os.remove(os.path.join(out_dir, f_old))
            os.rmdir(out_dir)
            print(f"  -> {os.path.basename(seg)} ({n} frames, "
                  f"{(time.perf_counter() - t_take) / max(n + lead, 1) * 1000:.0f} ms/frame)",
                  flush=True)

    if segments:
        # Every segment in the working dir, in route order -- not just the ones
        # this run produced. Re-render one take with --takes and the film is
        # reassembled around it.
        import glob
        segments = sorted(glob.glob(os.path.join(FILM_DIR, "seg_*.mp4")))
        lst = os.path.join(FILM_DIR, "segments.txt")
        with open(lst, "w") as fh:
            for seg in segments:
                fh.write("file '%s'\n" % seg.replace("\\", "/"))
        mp4 = os.path.join(FILM_DIR, cli_arg("--out", "warp_mudsnow_drive.mp4", str))
        subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-f", "concat",
                        "-safe", "0", "-i", lst, "-c", "copy", mp4], check=True)
        print(f"wrote {mp4}  ({saved} frames, {saved / 60.0:.1f} s, "
              f"{os.path.getsize(mp4) / 1e6:.1f} MB) in "
              f"{(time.perf_counter() - t_start) / 60.0:.1f} min")
        try:
            # Contact sheet off the FINISHED mp4, one frame every 2 s: read back
            # from the deliverable itself, so the sheet is what is in the film
            # even when only some takes were re-rendered into it.
            from PIL import Image
            for old in os.listdir(sheet_dir):
                os.remove(os.path.join(sheet_dir, old))
            subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", mp4,
                            "-vf", f"fps=1/{SHEET_EVERY // 60}", "-fps_mode", "passthrough",
                            os.path.join(sheet_dir, "t%03d.png")], check=True)
            thumbs = sorted(os.listdir(sheet_dir))
            cols = 6
            tw, th = 400, int(round(400 * HEIGHT / WIDTH))
            rows = int(math.ceil(len(thumbs) / cols))
            sheet = Image.new("RGB", (cols * tw, rows * th), (16, 15, 13))
            for i, thumb in enumerate(thumbs):
                im = Image.open(os.path.join(sheet_dir, thumb))
                sheet.paste(im.resize((tw, th), Image.LANCZOS),
                            ((i % cols) * tw, (i // cols) * th))
            sp = os.path.join(FILM_DIR, "contact_sheet.png")
            sheet.save(sp)
            print(f"contact sheet: {sp} ({len(thumbs)} frames, one per "
                  f"{SHEET_EVERY // 60} s)")
        except Exception as exc:              # noqa: BLE001 - the film is the deliverable
            print(f"contact sheet skipped: {exc}")
elif FRAMES > 0:
    # Honest windowed timing (see FRAMES): the real interactive frame, the real
    # wall-clock accumulator, the UI; the first 60 frames are discarded.
    _zf = 0.5 * (GRAVEL_Z[0] + GRAVEL_Z[1]) if GRAVEL else 0.0
    vehicle.respawn(tp.Vector3(SPAWN_X, ride_y(SPAWN_X, _zf), _zf), SPAWN_ROT)
    _ft = dict(n=0, t0=None, steps=0)

    def _timed():
        frame()
        _ft["n"] += 1
        if _ft["n"] == 60:
            wp.synchronize_device(device)
            _ft["t0"] = time.perf_counter()
            _ft["s0"] = _ft_steps[0]
        if _ft["n"] == 60 + FRAMES:
            wp.synchronize_device(device)
            dt_ms = (time.perf_counter() - _ft["t0"]) * 1000.0 / FRAMES
            print(f"frames [{'opengl' if GL else 'vulkan'}, {WIDTH}x{HEIGHT}, "
                  f"vsync {'on' if VSYNC else 'off'}]: {FRAMES} frames, {dt_ms:.2f} ms/frame "
                  f"= {1000.0 / dt_ms:.1f} fps; car x {vehicle.position.x:.1f} "
                  f"lanes {hud['lane']} v {vehicle.forward_speed * 3.6:.1f} km/h; "
                  f"{(_ft_steps[0] - _ft['s0']) / FRAMES:.2f} sim steps/frame = "
                  f"{(_ft_steps[0] - _ft['s0']) * DT * 1000.0 / (dt_ms * FRAMES):.2f}x real time", flush=True)
            canvas.close()

    canvas.animate(_timed)
    if _ft["n"] < 60 + FRAMES:
        print(f"frames: window closed after {_ft['n']} frames -- no measurement", flush=True)
else:
    canvas.animate(frame)
