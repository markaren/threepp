"""Blast Yard: a brick test range demolished by a charge -- PhysX + Warp + Vulkan.

A light-grey void with a grid floor, the way a test range looks in Unreal or
Omniverse, and a yard full of things that are *pre-broken by construction* --
a three-walled brick house, a shed, a ten-metre chimney, a stone gateway, a
concrete bunker, a rack of steel pipe (capsule bodies), pillars, pallets of
crates, boundary walls, loose rubble. Roughly seventeen hundred individual
PhysX bodies, laid in staggered courses with millimetre gaps so nothing starts
interpenetrating, settled for two seconds, spread out so every charge has its
own victim -- and then hit.

The blast is not a fracture solve; it is an impulse field with a SHOCK FRONT.
Nothing happens at t0 except the flash: the front leaves the charge at
--front-speed and each body is hit when the front ARRIVES,

    t_hit = t0 + r / C_FRONT,      J(r) = J0 * exp(-r / L) / max(r, r0) ** 1.5

along (body - charge), with an upward bias near the floor (a real blast
reflects off the ground and lifts what is standing on it), a speed cap so the
bodies nearest the charge do not leave at kilometres per second, and a random
tumble. The near wall therefore leaves before the far one and the yard comes
apart as a travelling ripple, which is the whole read. A decaying radial "wind"
force follows each body's OWN arrival for 0.45 s -- that is what keeps the wall
panels moving after the first frame instead of dropping straight down, and it
is the difference between "a wall fell over" and "a wall was blown out". A
point light flashes at the charge, and the camera shakes when the front reaches
the LENS, not when the charge goes off.

There can be up to eight charges. `--charges "x,z,J@t; ..."` places them; in the
window B detonates at a structure that is still standing and D at the orbit
target, as often as you like. A new charge creates nothing -- no field, no
buffer, no scene entry -- it takes a slot of a fixed device-side charge array
and a share of the shared particle pools, so it costs no more than the frame it
lands on.

The gas is nine `tp.ParticleField`s at NEBULA SCALE -- ~15 million particles by
default -- all created before the first frame and parked at live count 0.
Creating one mid-run is a device idle and a cleared TAA history, on the one
frame where that shows.

Fifteen million particles do not fit on the bus. They ride
`renderer.enable_particle_field_interop` instead: the renderer exports each
field's positions allocation as an OS handle, CUDA imports it once
(`threepp.cuda_interop.VkInteropArray`), and the Warp sim writes it device to
device inside render(). Zero host particle traffic, ever. Without the extension
(or with --no-interop) the demo falls back to HostRing `submit()` at ~4% of the
counts and still runs.

THE MODEL: an explosion is a fireball that turns warmth into thick smoke
pushing outward. ALL matter is born inside the burst window; nothing is
instantiated afterwards. Everything you see later is that same matter evolving.

  fire   Born in a 0.3 s burst out of the charge: a radial blast with heavy
         drag, curl stir and buoyancy. Each parcel carries a cooling time.
         Rendered as a DensityRepr volume whose blackbody emission ramp
         (2700 K at the box floor, 1250 K at its top) IS the fireball.
  smoke  NEVER EMITTED. A fire parcel that cools past its threshold DIES and
         CONVERTS into SMOKE_K soot parcels at the same place, inheriting most
         of its outward momentum -- so the smoke pushes outward from the
         fireball, thick, and only then does buoyancy roll it up into a head.
         The column and the stem are the plume's own dynamics; no ground
         emitter pumps smoke after the blast. DensityRepr only.
  dust   Born on the ground annulus at the shock radius r = c*t while the front
         is still crossing the yard (~0.9 s), so the ring races outward ahead
         of the debris for free. DensityRepr only.
  ring   The Wilson cloud: a thin condensation shell ON the front, emitted at
         r = C_FRONT*t AND thrown outward at C_FRONT so each parcel rides the
         wave for the quarter second it lives instead of being left behind as a
         widening band. Bright, translucent, above the dust skirt -- the only
         part of a shockwave you can actually photograph. DensityRepr only.
  ember  A tight 0.4 s spawn, then long-lived embers that arc, land and burn
         out. Additive billboards. Not the analytic emitter, which cannot
         express a one-shot burst at all -- see the note at the field.
  flare  A small hot subset of the burst carrying the additive billboard flash
         with its own bloom pyramid. 15 M additive quads is not a look, it is
         a fill-rate accident; a quarter million of them is the sparkle.

Emission, recycling and the fire->smoke conversion are all device kernels
structured around a fixed-max array of 8 charges. The CPU never walks a
particle: per frame it computes one integer per cohort per charge and hands the
ring cursor to a launch. The pools are SHARED between charges -- a ninth
overlapping burst overwrites the oldest slots, which is the recycling policy.

Everything is driven off one sim clock and one blast timeline, so the later
phases (drums, slow-motion film) hang off the same t0 without re-deriving it.

    python warp_explosion.py                 # window; drag to orbit, Esc quits
    python warp_explosion.py --shot 4        # headless: sim 4 s, write png
    python warp_explosion.py --video 6       # headless: 6 s of frames (+mp4 if ffmpeg)
    python warp_explosion.py --bench         # timed phase breakdown, post-detonation
    python warp_explosion.py --yield 2400    # bigger charge (J0, kg m/s at 1 m)
    python warp_explosion.py --charge 0,0.6,-1.2 --t0 1.5
    python warp_explosion.py --charges "0,-0.6,1400@2.0; -14.5,-6,1700@2.6"
    python warp_explosion.py --auto-charge 2.5     # keep blowing things up
    python warp_explosion.py --front-speed 35      # a slower, more readable ripple
    python warp_explosion.py --courses 20 --shot 4      # taller walls, more bodies
    python warp_explosion.py --shot 4 --spin 0.25       # orbit while simulating
    python warp_explosion.py --gas 0.3                  # 4.4 M particles, not 15 M
    python warp_explosion.py --sigma 1.6                # thicker smoke
    python warp_explosion.py --no-interop               # HostRing fallback, reduced counts
    python warp_explosion.py --no-gas                   # phase 1 only
    python warp_explosion.py --no-ao                    # RT AO/GI off (~6.5 ms/frame back)
    python warp_explosion.py --msaa 1                   # the other render knob

Vulkan only: the fireball renders with tp.ParticleField, which draws nothing on
the GL path by design. The smoke rides the froxel FOG pass, so --no-fog is a
phase-1 debug flag -- it turns the gas volumes off with it.
"""
import math
import os
import subprocess
import sys
import tempfile
import time

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import (bench_loop, cli_arg, find_ffmpeg, parse_size,
                         resize_handler, standard_material)

SHOT = "--shot" in sys.argv
SHOT_T = cli_arg("--shot", 4.0, float)
VIDEO = cli_arg("--video", 0.0, float)     # seconds of frames to render
BENCH = "--bench" in sys.argv
HEADLESS = SHOT or VIDEO > 0 or BENCH
SEED = cli_arg("--seed", 7, int)

if HEADLESS:
    # Presenting to the hidden window is pure cost offline -- and in --bench it
    # is worse than cost: nothing pumps the window's message loop, so present
    # stalls and the numbers come out roughly a third of the real window rate.
    os.environ.setdefault("THREEPP_VULKAN_SUPPRESS_PRESENT", "1")

FPS = 60.0
DT = 1.0 / FPS

# --- the charges ------------------------------------------------------------
#
# A charge is (position, yield, t0). Up to MAX_CHARGES of them are live at once
# -- the device-side gas emission has been indexing a fixed 8-slot charge array
# since phase 2.6, and this is what fills it. The ninth overlapping burst takes
# the oldest slot; that is the recycling policy, not an error.

T0 = cli_arg("--t0", 2.0, float)           # detonation time; before it, the stack settles
YIELD = cli_arg("--yield", 1400.0, float)   # J0: impulse (kg m/s) at 1 m on a 1 kg body
CHARGE = tuple(float(v) for v in cli_arg("--charge", "0,0.6,-0.6", str).split(","))
CHARGE_SPEC = cli_arg("--charges", "", str)  # "x,z,J@t; x,y,z,J@t; ..." -- overrides --charge
# Fire a charge at something still standing every N seconds. It is what the B
# key does, on a timer -- a demo mode, and the honest way to measure that a
# mid-run detonation costs nothing structural (watch the worst-frame print).
AUTO_CHARGE = cli_arg("--auto-charge", 0.0, float)
_auto = [T0 + 1.0]
MAX_CHARGES = 8           # slots in the device charge array; a ring
# THE SHOCK FRONT. Nothing happens at t0 except the flash: every body takes its
# impulse when the front ARRIVES, t_hit = t0 + r / C_FRONT, and the wind tail
# follows ITS arrival rather than t0. That stagger IS the shockwave -- the near
# wall leaves, then the far one, and the yard comes apart as a ripple.
C_FRONT = cli_arg("--front-speed", 60.0, float)
BLAST_L = 5.0             # exponential range of the impulse field, m
BLAST_R0 = 0.8            # near-field softening radius, m
BLAST_P = 1.5             # geometric falloff exponent
BLAST_VMAX = 22.0         # speed cap on the kick, m/s -- the anti-NaN valve
BLAST_LIFT = 0.55         # ground-reflection upward bias, added to the unit direction
BLAST_SPIN = 0.55         # random tumble, rad/s per m/s of kick
WIND_A0 = 130.0            # radial "wind" acceleration at 1 m, m/s^2
WIND_TAU = 0.18           # its decay time, s
WIND_T = 0.45             # how long the wind blows, s
FLASH_I = 2600.0          # point-light flash peak intensity
FLASH_TAU = 0.075         # flash decay, s
SHAKE_A = 0.11            # camera shake amplitude at the default yield, m
SHAKE_TAU = 0.30
SHAKE_T = 1.4
# The DUST skirt is slower than the front by design: the front is the pressure
# wave, the dust is what it tears off the ground behind itself. The condensation
# ring below rides C_FRONT and is the front you can see.
DUST_C = 28.0             # visual dust-ring speed, m/s
FLASH_BB = 7.0            # billboard brightness spike at t0, on top of the base gain

# --- the gas ----------------------------------------------------------------
#
# Counts are VRAM and kernel bandwidth now, not bus bandwidth: the zero-copy
# interop path never moves a particle across PCIe. Each cohort costs
# 16 B (the renderer's exported positions) + 16 B (the sim's own positions, the
# copy source) + 12 B (velocity) + 8 B (age/life) = 52 B/particle resident.
# --gas scales every count; --sigma scales the optical mass the other way.
NO_GAS = "--no-gas" in sys.argv
NO_INTEROP = "--no-interop" in sys.argv
GAS = cli_arg("--gas", 1.0, float)
SIGMA = cli_arg("--sigma", 1.0, float)
SMOKE_K = cli_arg("--smoke-k", 3, int)      # soot parcels one cooled fire parcel becomes

# The capacity decision needs the device, because it is the interop path that
# makes fifteen million affordable: without it every particle crosses the bus
# TWICE a frame, so the fallback runs at phase 2's counts, which is what the
# HostRing path was measured to carry (1.79 ms of submit at 644 k live).
device = None
INTEROP = False
if not NO_GAS:
    wp.init()
    device = wp.get_preferred_device()
    INTEROP = (not NO_INTEROP) and device.is_cuda
FALLBACK = 0.04
_SCALE = GAS * (1.0 if INTEROP else FALLBACK)
FIRE_N = int(3_000_000 * _SCALE)
DUST_N = int(2_000_000 * _SCALE)
# Embers do NOT scale with the rest, and that is a finding, not an oversight: an
# ember is a DISCRETE POINT on screen, so its count is a dot density, not an
# optical mass. Half a million of them, thrown outward and landing on the yard,
# read as a carpet of white static thirty metres across -- exactly what phase 2
# rejected dust billboards for. Three times phase 2's count is the ceiling.
EMBER_N = int(70_000 * _SCALE)
FLARE_N = int(250_000 * _SCALE)
# The Wilson cloud: a thin condensation ring ON the front. It is emitted at the
# front radius AND thrown outward at the front speed, so each parcel rides the
# wave for its own short life instead of being left behind as a widening band.
WILSON_N = int(600_000 * _SCALE)
# Smoke is not emitted, it is CONVERTED, and the mapping is static: fire slot i
# becomes smoke slots [i*K, i*K+K). That is why there is no free list and no
# atomic anywhere in this sim -- the smoke pool inherits the fire ring's
# recycling for free.
SMOKE_N = FIRE_N * SMOKE_K

# Emission windows, in seconds since a charge's t0. NOTHING IS INSTANTIATED
# OUTSIDE THEM -- that is the model, not an optimisation. The longest window is
# the dust ring's, and it is the shock front crossing the yard, not a fountain.
FIRE_EMIT, FIRE_LIFE = 0.30, (0.30, 1.15)   # "life" here IS the cooling time
DUST_EMIT, DUST_LIFE = 0.90, (0.9, 2.2)
EMBER_EMIT, EMBER_LIFE = 0.40, (0.9, 3.4)   # a TIGHT burst, then long-lived embers
FLARE_EMIT, FLARE_LIFE = 0.22, (0.14, 0.46)
SMOKE_LIFE = (4.5, 9.5)                     # from the moment it was fire
# The ring is emitted for as long as the front is worth watching -- at 60 m/s
# that is 27 m, past the far props -- and each parcel lives a quarter second.
WILSON_EMIT, WILSON_LIFE = 0.45, (0.20, 0.42)
EMBER_SKEW, FLARE_SKEW = 1.8, 1.4           # emitted = N * (tau/EMIT)**(1/skew)

FIRE_V0, FIRE_DRAG, FIRE_BUOY, FIRE_CURL = 21.0, 7.5, 26.0, 7.0
SMOKE_DRAG, SMOKE_BUOY, SMOKE_CURL = 1.30, 4.5, 3.4
SMOKE_INHERIT = 0.75      # of the fire parcel's velocity: this is the outward push
SMOKE_SPREAD = 1.35       # m/s of isotropic scatter added to each soot child
DUST_V0, DUST_DRAG, DUST_GRAV, DUST_CURL = 7.5, 2.2, 9.81, 0.7
EMBER_V0, EMBER_DRAG, EMBER_CURL = 18.0, 0.90, 0.9
FLARE_V0, FLARE_DRAG, FLARE_BUOY, FLARE_CURL = 19.0, 6.5, 20.0, 6.0
# Low drag on purpose: the ring has to hold the front's speed for its whole
# life or it falls behind and the ring stops being a ring.
WILSON_DRAG, WILSON_BUOY, WILSON_CURL = 0.55, 1.2, 0.9
FIRE_R = (0.15, 0.80)     # world radius over life, m (w under WSemantic.Radius)
SMOKE_R = (0.30, 1.20)
DUST_R = (0.14, 0.70)
EMBER_R = (0.075, 0.0)    # holds its size, then goes out (r_pow below)
FLARE_R = (0.22, 0.85)
WILSON_R = (0.55, 1.70)   # grows as it dies: the ring thins out as it expands

# Volume resolution, LATCHED at the first frame the field has live particles.
# RAISING IT DOES NOT BUY DETAIL HERE, and that is worth knowing: the scatter is
# per-VOXEL OCCUPANCY, so doubling the resolution divides the particles per voxel
# by eight and the medium goes thin and grainy at the same sigma. 192**3 was
# tried against 128**3 for the smoke at nine million parcels and read WORSE --
# more scatter noise, no more structure -- because the plume's detail is set by
# the particles, not the grid. 128 stays, and the sigma is tuned against it.
FIRE_RES = cli_arg("--fire-res", 128, int)
SMOKE_RES = cli_arg("--smoke-res", 128, int)
DUST_RES = cli_arg("--dust-res", 128, int)
WILSON_RES = cli_arg("--wilson-res", 128, int)

# sigma_t one particle contributes, before --sigma.
#
# The scatter is an 8-tap trilinear splat into ONE voxel neighbourhood and the
# particle radius is never consulted, so what the fog march sees is
#   density = (particles in the voxel) * sigma = rho * V_voxel * sigma.
# Both factors move when the counts do, which is why these are NOT simply
# phase 2's values over N: going from 300 k at res 128 to 9 M at res 192 divides
# rho*V_voxel by 30 and multiplies it by 2.34, and the sigma that keeps the same
# picture follows. Then tuned by eye from there.
FIRE_SIGMA, SMOKE_SIGMA, DUST_SIGMA = 0.22, 0.32, 0.040
WILSON_SIGMA = 0.055
# Single-scatter albedo per medium (kMaxDensityFields slots: fire, smoke, dust,
# condensation ring). SOOT IS DARK -- and it has to be dark in the RIGHT way:
# the in-scatter is albedo x (ambient + sun x phase), so a neutral grey column
# under a white sky is a WHITE column no matter how low you take it. What reads
# as soot is a dirty warm grey with the BLUE end pulled down hardest, which is
# what tints the lit side toward brown instead of toward the sky.
#
# What the knob CANNOT fix, and it is worth knowing before spending an hour on
# it: the plume's sunlit side is bright because the density march lights it with
# shadowVis() sampled from the SCENE's shadow map, and particle volumes are not
# in the shadow map -- so a nine-million-parcel column is fully sunlit all the
# way through, with no self-shadowing to darken its own far side. Albedo scales
# that, it does not shade it. Below about 0.05 the column reads as cold ash
# rather than soot and the brown goes with it; 0.10/0.086/0.066 is the darkest
# value that still reads dirty rather than dead.
SMOKE_ALBEDO = tuple(float(v) for v in
                     cli_arg("--smoke-albedo", "0.10,0.086,0.066", str).split(","))

# --- the yard ---------------------------------------------------------------

BRICK = (0.40, 0.20, 0.20)     # length (local x) x height (y) x depth (z)
GAP = 0.002                    # laid with a 2 mm joint: no initial interpenetration
COURSES = cli_arg("--courses", 16, int)
SHED_COURSES = cli_arg("--shed-courses", 11, int)
GROUND = 400.0                 # floor slab side, m: the edge sits out in the fog
JITTER = 0.0015                # per-brick placement noise, m

RHO_BRICK, RHO_CRATE, RHO_BLOCK = 900.0, 160.0, 1500.0
RHO_STONE, RHO_CONC, RHO_PIPE = 1900.0, 2100.0, 420.0

rng = np.random.default_rng(SEED)

# Each item is (position, size, yaw). One list per material batch: instance
# colours are a GL-only feature (no `instanceColor` anywhere on the Vulkan
# path), so the yard's colour variety is SEVEN batches rather than seven
# colours in one, and each batch is still one draw and one add_instanced.
batches = {"brick0": [], "brick1": [], "brick2": [], "crate": [], "block": [],
           "stone": [], "conc": [], "pipe": []}
_brick_tone = 0

# Which structure each item belongs to, so the B key can pick a target that is
# still standing. `group` is the name currently being built; GROUPS maps it to
# (batch, index-within-batch) pairs, resolved to global body indices once the
# bodies exist.
GROUPS = {}
_group = "yard"


def group(name):
    global _group
    _group = name


def add_item(kind, pos, size, yaw=0.0):
    GROUPS.setdefault(_group, []).append((kind, len(batches[kind])))
    batches[kind].append((pos, size, yaw))


def add_brick(pos, yaw):
    """A brick, in whichever of the three clay tones comes next."""
    global _brick_tone
    add_item(f"brick{_brick_tone % 3}", pos, BRICK, yaw)
    _brick_tone += 7          # coprime with 3: adjacent bricks never share a tone run


def brick_wall(origin, direction, s0, s1, courses, y0=0.0):
    """A staggered brick wall from origin + direction*s0 to origin + direction*s1.

    `direction` is a unit (dx, dz); the brick's long axis runs along it. Odd
    courses are shifted half a brick, so the joints break like real bonding and
    the stack has something to interlock on when it is hit.
    """
    dx, dz = direction
    yaw = math.atan2(-dz, dx)                  # local +X -> (dx, 0, dz)
    pitch = BRICK[0] + GAP
    span = s1 - s0
    for c in range(courses):
        y = y0 + (c + 0.5) * (BRICK[1] + GAP) + 0.001
        shift = 0.5 * pitch if c % 2 else 0.0
        s = shift + 0.5 * pitch
        while s + 0.5 * pitch <= span + 1e-6:
            jx, jz = rng.normal(0.0, JITTER, 2)
            add_brick((origin[0] + dx * (s0 + s) + jx, y,
                       origin[2] + dz * (s0 + s) + jz),
                      yaw + float(rng.normal(0.0, 0.004)))   # under the joint width
            s += pitch


def brick_house(centre, lx, lz, courses):
    """Three walls in a U, open toward +Z: back wall plus two returns.

    The returns start one brick-depth past the back wall so the corners butt
    instead of overlapping.
    """
    cx, cz = centre
    inset = BRICK[2] + GAP
    brick_wall((cx - 0.5 * lx, 0.0, cz - 0.5 * lz), (1.0, 0.0), 0.0, lx, courses)
    for sx in (-1.0, 1.0):
        brick_wall((cx + sx * 0.5 * lx, 0.0, cz - 0.5 * lz), (0.0, 1.0),
                   inset, lz, courses)


def pillar(x, z, segments, seg=(0.46, 0.25, 0.46)):
    """A stacked column of segments, each turned a few degrees off its neighbour."""
    for k in range(segments):
        y = (k + 0.5) * (seg[1] + GAP) + 0.001
        jx, jz = rng.normal(0.0, JITTER, 2)
        add_item("block", (x + jx, y, z + jz), seg, float(rng.normal(0.0, 0.02)))


def pallet(x, z, nx, nz, levels, crate=0.5):
    """A block of crates on the ground.

    The yaw is deliberately tiny: a 0.5 m box turned 3 degrees pokes 12 mm past
    its own footprint, which is more than the joint, and a pallet that starts
    interpenetrating shakes itself apart during the settle.
    """
    yaw_max = 0.01
    pitch = crate * (1.0 + yaw_max) + 0.012
    for ly in range(levels):
        for ix in range(nx):
            for iz in range(nz):
                jx, jz = rng.normal(0.0, JITTER, 2)
                add_item("crate",
                         (x + (ix - 0.5 * (nx - 1)) * pitch + jx,
                          (ly + 0.5) * (crate + GAP) + 0.001,
                          z + (iz - 0.5 * (nz - 1)) * pitch + jz),
                         (crate, crate, crate), float(rng.uniform(-yaw_max, yaw_max)))


def rubble(n, r_min, r_max):
    """Loose bricks lying flat around the yard."""
    for _ in range(n):
        a = float(rng.uniform(0.0, 2.0 * math.pi))
        r = float(rng.uniform(r_min, r_max))
        add_brick((r * math.sin(a), 0.5 * BRICK[1] + 0.001, r * math.cos(a)),
                  float(rng.uniform(0.0, 2.0 * math.pi)))


def tower(x, z, courses, seg=(1.05, 0.55, 1.05)):
    """A chimney: a tall stack of heavy stone blocks.

    The tallest thing in the yard and the one that TOPPLES rather than
    scattering -- a stack this slender goes over as a whole before it comes
    apart, which is the silhouette the yard was missing.
    """
    for k in range(courses):
        y = (k + 0.5) * (seg[1] + GAP) + 0.001
        taper = 1.0 - 0.012 * k                   # a hair narrower every course
        jx, jz = rng.normal(0.0, JITTER, 2)
        add_item("stone", (x + jx, y, z + jz),
                 (seg[0] * taper, seg[1], seg[2] * taper),
                 float(rng.normal(0.0, 0.01)))


def archway(x, z, span=3.2, courses=8, seg=(0.75, 0.60, 0.75)):
    """A gateway: two block pillars carrying a lintel.

    The span runs along X, across the camera's line of sight -- an arch seen
    end-on is one column, and this one is out at the edge of the yard where
    that is exactly how it would be seen.
    """
    for sx in (-0.5, 0.5):
        for k in range(courses):
            y = (k + 0.5) * (seg[1] + GAP) + 0.001
            jx, jz = rng.normal(0.0, JITTER, 2)
            add_item("stone", (x + sx * span + jx, y, z + jz), seg,
                     float(rng.normal(0.0, 0.01)))
    # THE LINTEL SPANS THE GAP -- three beams laid side by side ACROSS both
    # pillars, not four blocks laid along it. The obvious spelling (a row of
    # blocks between the pillars) leaves every one of them cantilevered off one
    # pillar, and the gateway falls down before the charge is even armed.
    top = courses * (seg[1] + GAP) + 0.001
    beam = span + seg[0] - 0.10
    for i in range(3):
        add_item("stone", (x, top + 0.21, z + (i - 1.0) * 0.24), (beam, 0.42, 0.23), 0.0)
    for sx in (-1.0, 1.0):                        # a cap, for something to shed
        add_item("stone", (x + sx * 0.27 * beam, top + 0.42 + 0.18, z),
                 (0.44 * beam, 0.36, 0.80), 0.0)


def pipe_rack(x, z, length=6.4, radius=0.24):
    """Capsule bodies stacked in a cradle on two block trestles.

    PhysX infers a capsule collider straight off CapsuleGeometry (the threepp
    capsule is Y-aligned, PhysX's is X-aligned, and PhysxWorld's shape
    inference already carries the -PI/2 local pose that reconciles them), so a
    pipe is one instance of one InstancedMesh like everything else here. The
    only thing it needs that a brick does not is a real ROTATION: the instance
    is tipped a quarter turn onto the Z axis, which is why the pipe batch
    composes its own quaternion in the build loop below.

    A PIPE ROLLS, which is the whole difficulty. Laid straight on a flat
    trestle the rack shakes itself empty during the two-second settle -- the
    first thing this demo taught me about capsules. So: side rails on the
    trestles, a bottom row of three touching pipes between them, and two more
    nested in the valleys, which is how pipe actually sits in a yard.
    """
    top = 3.0 * (0.62 + GAP)
    for sz in (-1.0, 1.0):
        for k in range(3):
            add_item("conc", (x, (k + 0.5) * (0.62 + GAP) + 0.001,
                              z + sz * 0.40 * length), (2.4, 0.62, 0.55), 0.0)
        for sx in (-1.0, 1.0):          # the rails that stop the row rolling off
            add_item("conc", (x + sx * 1.05, top + 0.16, z + sz * 0.40 * length),
                     (0.26, 0.32, 0.55), 0.0)
    pitch = 2.0 * radius + 0.02
    for col in range(3):
        add_item("pipe", (x + (col - 1.0) * pitch, top + radius + 0.004, z),
                 (1.0, 1.0, 1.0), 0.0)
    for col in range(2):                # nested in the valleys of the row below
        add_item("pipe", (x + (col - 0.5) * pitch,
                          top + radius + 0.004 + 0.866 * pitch, z),
                 (1.0, 1.0, 1.0), 0.0)


def bunker(x, z, courses=3, block=(1.20, 0.55, 0.60)):
    """A low concrete bunker: three walls under a roof of slabs.

    The heaviest thing in the yard by far, and deliberately so -- it is what
    stands when the front has flattened everything around it, and the roof
    slabs are the pieces a near miss actually moves.
    """
    for c in range(courses):
        y = (c + 0.5) * (block[1] + GAP) + 0.001
        for i in range(4):                        # back wall, along x
            add_item("conc", (x + (i - 1.5) * (block[0] + GAP), y, z - 2.1), block, 0.0)
        for sx in (-1.0, 1.0):                    # returns, along z
            for j in range(3):
                add_item("conc", (x + sx * 2.1, y, z + (j - 1.0) * (block[0] + GAP)),
                         (block[2], block[1], block[0]), 0.0)
    # The slabs have to land INSIDE the returns' span at both ends: a roof
    # overhanging its own wall tips off during the settle, which is how the
    # first version of this bunker came apart before anything hit it.
    roof = courses * (block[1] + GAP) + 0.001
    for i in range(3):
        add_item("conc", (x, roof + 0.16, z + (i - 1.0) * (block[0] + GAP)),
                 (5.4, 0.32, 1.15), 0.0)


group("house")
brick_house((0.0, 0.0), 8.0, 5.5, COURSES)            # the main structure
group("shed")
brick_house((-7.6, 1.0), 3.6, 3.0, SHED_COURSES)      # a shed off to one side
group("wall")
brick_wall((6.6, 0.0, -3.0), (0.0, 1.0), 0.0, 6.0, COURSES - 4)   # free-standing wall
group("pillars")
for px, pz in ((9.6, -2.2), (9.6, 1.8)):
    pillar(px, pz, 10)
group("pallets")
pallet(-3.4, 4.6, 3, 3, 3)
pallet(4.2, 5.2, 3, 2, 4)
pallet(8.2, 4.4, 3, 3, 3)
pallet(-6.8, -2.6, 3, 2, 3)     # keep pallets wider than they are tall, or they settle over
# ── the dressing: one silhouette per charge, spread across the yard ──────────
group("tower")
tower(-14.5, -6.0, 19)
group("arch")
archway(13.5, -6.5)
group("pipes")
pipe_rack(-6.5, 10.5)
group("bunker")
bunker(8.5, 11.0)
group("far_pallets")
pallet(-12.5, 4.6, 3, 2, 3)
pallet(13.0, 3.4, 2, 3, 3)
pallet(1.0, 13.5, 4, 2, 2)
group("perimeter")
brick_wall((-17.5, 0.0, -11.0), (0.0, 1.0), 0.0, 9.6, 6)          # boundary walls: the
brick_wall((4.0, 0.0, -13.5), (1.0, 0.0), 0.0, 10.0, 6)           # front's far victims
group("yard")
rubble(90, 3.5, 15.0)

N_BODIES = sum(len(v) for v in batches.values())

# --- renderer ---------------------------------------------------------------

if not tp.vulkan_available():
    sys.exit("warp_explosion needs the Vulkan renderer (ParticleField is Vulkan-only).")

W, H = parse_size(cli_arg("--size", "1280x720", str))
MSAA = cli_arg("--msaa", 4, int)              # G-buffer MSAA (measured free here at 720p)
canvas = tp.Canvas("threepp x physx - blast yard", width=W, height=H,
                   antialiasing=MSAA, vsync=False, headless=HEADLESS)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 0.55    # pinned: phase 2's billboards bypass auto-exposure
renderer.shadow_map_enabled = True
# The two knobs that actually move the frame time here (measured, --bench):
# ray-traced AO/GI ~6.5 ms and the fog march ~2.5 ms at 1280x720. The body
# count is not the lever -- PhysX is under 2 ms of a 22 ms frame.
NO_AO = "--no-ao" in sys.argv
NO_FOG = "--no-fog" in sys.argv
if NO_AO:
    renderer.deferred_ao = False

VOID = tp.Color(0.80, 0.81, 0.83)
scene = tp.Scene()
scene.background = VOID
if not NO_FOG:
    scene.set_fog(VOID, 55.0, 200.0)          # dissolves the slab's far edge into the void

camera = tp.PerspectiveCamera(46, canvas.aspect(), 0.05, 800)
# Framed for the PLUME, not the yard: at nebula counts the column is the
# subject and it climbs past 25 m, so the yard only occupies the bottom third.
CAM_R, CAM_Y, CAM_TARGET = 38.0, 13.0, (0.0, 9.0, 0.0)
SPIN = cli_arg("--spin", 0.0, float)          # shot/video: orbit the camera, deg/frame
START_ANGLE = cli_arg("--angle", 22.0, float)  # orbit start angle, degrees
_cam_base = tp.Vector3()


def orbit(frame):
    """Place the camera on its orbit for `frame` (static at SPIN == 0)."""
    a = math.radians(START_ANGLE + SPIN * frame)
    _cam_base.set(CAM_R * math.sin(a), CAM_Y, CAM_R * math.cos(a))
    camera.position.set(_cam_base.x, _cam_base.y, _cam_base.z)
    camera.look_at(*CAM_TARGET)


# A test range is an overcast-bright void with one hard sun in it. The fill is
# deliberately weak: everything here is grey or clay, and a strong hemisphere
# fill flattens both into the same pale pink.
scene.add(tp.HemisphereLight(0xd7e3f4, 0x8d8a84, 0.45))
sun = tp.DirectionalLight(0xfff2df, 3.0)
sun.position.set(9.0, 15.0, 6.0)
sun.cast_shadow = True
scene.add(sun)

# One point light per concurrent charge, created here and parked at zero. THREE,
# not MAX_CHARGES: every point light in the scene costs the particle-density
# march a per-step in-scatter evaluation whether it is lit or not, and three
# flashes overlapping inside one 0.3 s decay is already more than the demo asks
# for. A fourth charge re-uses the oldest light, which by then is dark.
FLASHES = [tp.PointLight(0xffd6a0, 0.0, 45.0, 2.0) for _ in range(3)]
for _f in FLASHES:
    _f.position.set(*CHARGE)
    scene.add(_f)


def grid_texture(px=1024, tile=10.0, minor=1.0):
    """The test-range floor: a light grey with 1 m minor and 10 m major rules.

    Written to a temp PNG because TextureLoader takes a path; `tile` is the
    metres the image covers, which is what sets the repeat below.
    """
    from PIL import Image

    def rules(period, width):
        i = np.arange(px)
        return ((i + width // 2) % period) < width

    img = np.full((px, px, 3), 163, np.uint8)
    fine = rules(max(int(round(px * minor / tile)), 2), 2)
    coarse = rules(px, 6)
    for mask, tone in ((fine, 138), (coarse, 100)):
        img[mask, :, :] = tone
        img[:, mask, :] = tone
    path = os.path.join(tempfile.gettempdir(), f"threepp_blast_grid_{px}_{int(tile)}.png")
    Image.fromarray(img).save(path)
    return path, tile


grid_path, grid_tile = grid_texture()
grid = tp.TextureLoader().load(grid_path, tp.ColorSpace.SRGB)
grid.wrap_s = tp.TextureWrapping.Repeat
grid.wrap_t = tp.TextureWrapping.Repeat
grid.repeat.set(GROUND / grid_tile, GROUND / grid_tile)
grid.needs_update()

floor_mat = standard_material(tp.Color(1.0, 1.0, 1.0), 0.92)
floor_mat.map = grid
# A slab, not a plane: PhysX infers a collider from Box geometry, so the thing
# you see and the thing the bricks land on are one object.
floor = tp.Mesh(tp.BoxGeometry(GROUND, 0.4, GROUND), floor_mat)
floor.position.y = -0.2
floor.receive_shadow = True
scene.add(floor)

# --- physics ----------------------------------------------------------------

world = tp.PhysxWorld(tp.Vector3(0.0, -9.81, 0.0), fixed_timestep=DT, max_substeps=2,
                      num_threads=8, tgs_pcm=True)
ground_mat = world.create_material(0.62, 0.58, 0.06)
world.add_static(floor, ground_mat)

PIPE_R, PIPE_L = 0.24, 6.4
MATERIALS = {
    "brick0": (standard_material(tp.Color(0.42, 0.17, 0.12), 0.92), RHO_BRICK),
    "brick1": (standard_material(tp.Color(0.52, 0.24, 0.16), 0.92), RHO_BRICK),
    "brick2": (standard_material(tp.Color(0.35, 0.15, 0.12), 0.94), RHO_BRICK),
    "crate": (standard_material(tp.Color(0.46, 0.32, 0.17), 0.88), RHO_CRATE),
    "block": (standard_material(tp.Color(0.52, 0.52, 0.50), 0.95), RHO_BLOCK),
    "stone": (standard_material(tp.Color(0.56, 0.52, 0.44), 0.93), RHO_STONE),
    "conc": (standard_material(tp.Color(0.68, 0.68, 0.66), 0.90), RHO_CONC),
    "pipe": (standard_material(tp.Color(0.20, 0.34, 0.40), 0.42), RHO_PIPE),
}
GEOMETRY = {"pipe": lambda: tp.CapsuleGeometry(PIPE_R, PIPE_L, 6, 14)}

UP = tp.Vector3(0.0, 1.0, 0.0)
XAXIS = tp.Vector3(1.0, 0.0, 0.0)
meshes, bodies = [], []
_offsets = {}                       # batch -> first global body index
for kind, items in batches.items():
    if not items:
        continue
    material, density = MATERIALS[kind]
    geometry = GEOMETRY.get(kind, lambda: tp.BoxGeometry(1.0, 1.0, 1.0))()
    im = tp.InstancedMesh(geometry, material, len(items))
    m, q = tp.Matrix4(), tp.Quaternion()
    for i, (p, s, yaw) in enumerate(items):
        if kind == "pipe":
            # A threepp capsule stands on Y; a pipe lies along Z. The collider
            # follows the instance transform, so tipping it here tips both.
            q.set_from_axis_angle(XAXIS, -0.5 * math.pi)
        else:
            q.set_from_axis_angle(UP, yaw)
        m.compose(tp.Vector3(*p), q, tp.Vector3(*s))
        im.set_matrix_at(i, m)
    im.instance_matrix_needs_update()
    im.cast_shadow = True
    im.receive_shadow = True
    im.frustum_culled = False        # the bounds go stale the moment it detonates
    scene.add(im)
    meshes.append(im)
    _offsets[kind] = len(bodies)
    bodies += world.add_instanced(im, density)

MASS = np.array([b.mass for b in bodies], dtype=np.float64)
# Each structure's global body indices and where it started, so a charge can be
# aimed at something that is still standing (the B key).
TARGETS = {}
for _name, _items in GROUPS.items():
    _idx = np.array([_offsets[k] + i for k, i in _items], np.int64)
    _p = np.array([batches[k][i][0] for k, i in _items], np.float64)
    TARGETS[_name] = (_idx, _p, _p.mean(0))
HOME = np.zeros((len(bodies), 3))
for _idx, _p, _ in TARGETS.values():
    HOME[_idx] = _p

print(f"blast yard: {N_BODIES} bodies in {len(meshes)} instanced batches "
      f"({MASS.sum():.0f} kg), {len(TARGETS)} structures")

# --- the blast timeline -----------------------------------------------------
#
# One clock, one timeline. Phases 2-4 (gas emission windows, shell coupling,
# the slow-motion ramp) read `blast_timeline(sim_time)` rather than testing
# sim_time against their own constants.

_shake_bank = [[(float(f), float(p)) for f, p in
                zip(rng.uniform(7.0, 23.0, 3), rng.uniform(0.0, 6.283, 3))]
               for _ in range(3)]


def body_positions():
    return np.array([[b.position.x, b.position.y, b.position.z] for b in bodies])


class Charge:
    """One detonation: where, how big, when -- and when its front gets where.

    NOTHING IS APPLIED AT t0 except the flash. The front leaves the charge at
    C_FRONT and each body takes its impulse when the front ARRIVES:

        t_hit = t0 + r / C_FRONT

    which is the whole shockwave read. The near wall leaves, then the far one;
    at 60 m/s a body 25 m out waits four tenths of a second for its turn, and
    the yard comes apart as a travelling ripple rather than all at once. The
    wind tail follows each body's OWN arrival, not t0, so the push behind the
    front travels with it.

    The cost is one argsort at arm time. Per frame it is a searchsorted plus a
    walk over the bodies that arrived since the last one -- the schedule is
    sorted by r, so both the arrivals and the wind window are contiguous slices.
    """

    def __init__(self, pos, j0=None, t0=None):
        self.pos = tuple(float(v) for v in pos)
        self.j0 = YIELD if j0 is None else float(j0)
        self.t0 = T0 if t0 is None else float(t0)
        self.scale = math.sqrt(self.j0 / 1400.0)
        self.armed = False
        self.slot = 0                 # its slot in the 8-wide device charge array
        self.cursor = 0               # how far down the arrival schedule we are
        self.shake_t = self.t0        # when the front reaches the CAMERA
        self.light = None
        self.streaks = None
        self.order = self.t_sorted = self.kick = self.spin = self.wind_u = None

    def arm(self):
        """Freeze the yard's geometry into this charge's arrival schedule.

        Everything per-body is computed ONCE, vectorised, here: the kick, the
        tumble, the wind direction and the arrival time. What runs per frame is
        only the handing of those numbers to PhysX.
        """
        p = body_positions()
        d = p - np.array(self.pos)
        r = np.sqrt((d * d).sum(1)) + 1e-6
        rad = d / r[:, None]                       # pure radial: the wind's direction
        u = rad.copy()
        # A blast reflects off the ground, so what is standing on it gets lifted.
        u[:, 1] += BLAST_LIFT * np.exp(-np.maximum(p[:, 1], 0.0) / 1.5)
        u /= np.linalg.norm(u, axis=1)[:, None]
        j = self.j0 * np.exp(-r / BLAST_L) / np.maximum(r, BLAST_R0) ** BLAST_P
        dv = np.minimum(j / MASS, BLAST_VMAX)      # the cap keeps the near field finite
        self.kick = u * (dv * MASS)[:, None]
        self.spin = (BLAST_SPIN * dv)[:, None] * rng.uniform(-1.0, 1.0, (len(MASS), 3))
        self.wind_u = rad / (np.maximum(r, BLAST_R0) ** BLAST_P)[:, None]
        t_hit = self.t0 + r / C_FRONT
        self.order = np.argsort(t_hit)
        self.t_sorted = t_hit[self.order]
        self.cursor = 0
        # The shake re-triggers when the front reaches the CAMERA -- a blast
        # 30 m away does not shake the lens until the wave gets there.
        c = camera.position
        self.shake_t = self.t0 + math.dist((c.x, c.y, c.z), self.pos) / C_FRONT
        self.light = FLASHES[_fired[0] % len(FLASHES)]
        self.light.position.set(self.pos[0], self.pos[1] + 0.3, self.pos[2])
        claim_charge_slot(self)                    # device charge array + streak field
        _fired[0] += 1
        self.armed = True

    def retire(self):
        """Drop the per-body schedule; the charge is inert from here on."""
        self.order = self.t_sorted = self.kick = self.spin = self.wind_u = None

    def flash_i(self, t):
        tw = t - self.t0
        if tw < 0.0 or tw >= 0.5:
            return 0.0
        return FLASH_I * self.scale * math.exp(-tw / FLASH_TAU)

    def shake(self, t):
        tw = t - self.shake_t
        if tw < 0.0 or tw >= SHAKE_T:
            return (0.0, 0.0, 0.0)
        a = SHAKE_A * self.scale * math.exp(-tw / SHAKE_TAU)
        return tuple(a * sum(math.sin(2.0 * math.pi * f * tw + p) for f, p in bank) / 3.0
                     for bank in _shake_bank)


_fired = [0]              # how many charges have gone off: the slot/light ring cursor
charges = []              # every charge that has not yet been retired


def parse_charges(spec):
    """--charges "x,z,J@t; x,y,z,J@t; x,z; ..." -> a list of Charges.

    y is optional (ground level, where a charge sits), J and @t both fall back
    to --yield and --t0. Two numbers is the shortest useful form: a place.
    """
    out = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        body, _, when = item.partition("@")
        v = [float(x) for x in body.split(",")]
        t0 = float(when) if when.strip() else None
        if len(v) == 2:
            out.append(Charge((v[0], 0.6, v[1]), None, t0))
        elif len(v) == 3:
            out.append(Charge((v[0], 0.6, v[1]), v[2], t0))
        elif len(v) == 4:
            out.append(Charge((v[0], v[1], v[2]), v[3], t0))
        else:
            sys.exit(f"--charges: cannot read '{item}' (want x,z[,J] or x,y,z,J, "
                     f"optionally @t)")
    return out


def fire_charge(pos, j0=None, at=None, why=""):
    """Arm a new charge -- from the CLI at startup, or from a key mid-run.

    Mid-run this is the whole story: no field is created, no buffer is
    allocated, nothing is added to the scene. The gas pools are shared and the
    new charge simply takes the next slot of the device charge array (the ring
    cursor overwrites the oldest emission if it has to), and the rigid side
    builds one schedule. That is why it does not hitch.
    """
    ch = Charge(pos, j0, sim_time + 0.5 * DT if at is None else at)
    charges.append(ch)
    if why:
        print(f"  [{sim_time:5.2f}] charge at ({pos[0]:5.1f},{pos[2]:5.1f}) -- {why}",
              flush=True)
    return ch


def intact_target():
    """A structure that is still standing, for the B key.

    "Still standing" is measured, not remembered: the median displacement of a
    structure's bodies from where they were built. A wall that has already been
    blown across the yard is not a target worth re-detonating.
    """
    ok = []
    p = body_positions()
    for name, (idx, home, centre) in TARGETS.items():
        if len(idx) < 8:
            continue
        d = np.linalg.norm(p[idx] - home, axis=1)
        if float(np.median(d)) < 0.35:
            ok.append((name, centre))
    if not ok:
        return None, None
    name, centre = ok[int(rng.integers(len(ok)))]
    return name, (float(centre[0]), 0.6, float(centre[2]))


def blast_timeline(t):
    """The state of the whole yard at sim time `t`, aggregated over the charges.

    tw       seconds since the MOST RECENT charge (negative before the first)
    armed    True once anything has gone off
    front    the newest front's radius, m
    shake    (x, y, z) camera offset, m: SUMMED, each starting when that
             charge's front reaches the camera
    bb_gain  additive-billboard brightness multiplier -- the flash spike,
             summed over charges, decaying over ~0.3 s each. Billboards
             composite after the upscaler and are NOT seen by auto-exposure, so
             this is the whole exposure story for them
    fire_e   fireball blackbody emission scale, the max over live charges

    Per-charge quantities that used to live here -- the flash intensity and the
    wind -- moved onto the Charge itself, because there is no longer one of
    them: each charge drives its own point light, and the wind is applied per
    BODY from the arrival schedule.
    """
    tw, gain, fire_e = -1.0e9, 0.0, 0.0
    shake = [0.0, 0.0, 0.0]
    for ch in charges:
        b = t - ch.t0
        if b < 0.0:
            continue
        tw = b if tw < -1.0e8 else min(tw, b)     # the newest charge's clock
        gain += FLASH_BB * math.exp(-b / 0.13)
        fire_e = max(fire_e, math.exp(-max(b - 0.16, 0.0) / 0.30))
        s = ch.shake(t)
        shake = [a + c for a, c in zip(shake, s)]
    if tw < -1.0e8:
        return dict(tw=-1.0, armed=False, front=0.0, shake=(0.0, 0.0, 0.0),
                    bb_gain=0.0, fire_e=0.0)
    return dict(tw=tw, armed=True, front=C_FRONT * tw, shake=tuple(shake),
                bb_gain=1.0 + gain, fire_e=fire_e)


def blast_frame(t):
    """Arm what is due, deliver the front's impulses, blow the wind behind it.

    This is the one place the rigid side hears about the blast, and it is
    O(bodies the front crossed this frame) plus O(bodies inside the wind
    window) -- not O(bodies) every frame.
    """
    kick, spin, force = tp.Vector3(), tp.Vector3(), tp.Vector3()
    done = []
    for ch in charges:
        if not ch.armed:
            if t < ch.t0:
                continue
            ch.arm()
        ch.light.intensity = ch.flash_i(t)
        if ch.order is None:
            continue
        # ── the front arrives ────────────────────────────────────────────────
        hi = int(np.searchsorted(ch.t_sorted, t, "right"))
        if hi > ch.cursor:
            idx = ch.order[ch.cursor:hi]
            for i in idx:
                i = int(i)
                b = bodies[i]
                b.wake_up()
                kick.set(*ch.kick[i])
                b.add_impulse(kick)
                spin.set(*ch.spin[i])
                b.set_angular_velocity(spin)
            ch.cursor = hi
        # ── and the wind follows it ──────────────────────────────────────────
        # force = a * m, so it is a genuine acceleration: light crates and heavy
        # pillar segments are carried by the same wind at the same rate.
        lo = int(np.searchsorted(ch.t_sorted, t - WIND_T))
        if hi > lo:
            idx = ch.order[lo:hi]
            amp = (WIND_A0 * ch.scale
                   * np.exp(-(t - ch.t_sorted[lo:hi]) / WIND_TAU) * MASS[idx])
            fv = ch.wind_u[idx] * amp[:, None]
            for k, i in enumerate(idx):
                force.set(*fv[k])
                bodies[int(i)].add_force(force)
        elif t - ch.t0 > GAS_END + SHAKE_T:
            done.append(ch)
    for ch in done:                    # inert: drop the schedule, stop iterating it
        ch.retire()
        charges.remove(ch)


# --- the gas: five ParticleFields, all created here, before the first frame --
#
# The churn contract (ParticleField.hpp): a field is created ONCE at its final
# capacity and never resized, and creating one is a STRUCTURAL scene change --
# entry re-expansion, a vkDeviceWaitIdle and a cleared TAA history. So all five
# exist from startup and sit at live count 0 until the charge goes off. Nothing
# is added to or removed from the scene after this point.

# The charge array is FIXED at MAX_CHARGES slots and lives on the device
# (xyz + t0). Every emission kernel indexes it, so a new detonation is a 128-byte
# upload and nothing else: no field, no buffer, no scene entry, no device idle.
CHARGES = parse_charges(CHARGE_SPEC) if CHARGE_SPEC else [Charge(CHARGE)]
charges.extend(CHARGES)
# How much of a pool ONE detonation may claim. Never fewer than two shares, so
# a charge fired from the keyboard in a single-charge run still has budget.
N_SHARES = max(len(CHARGES), 2)

# The curl field is baked to an NG^3 grid once per frame and fetched
# trilinearly, exactly as warp_nebula does it: 110k grid threads instead of
# 560k 4D noise evaluations, and the grid is what makes the smoke roll rather
# than every particle wobbling on its own phase.
NG = 48
NR = 30.0                              # half-size of the cubic noise box, m
NCX, NCY, NCZ = 0.0, 14.0, 0.0         # its world centre: over the yard, up the plume
NG_SCALE = (NG - 1) / (2.0 * NR)
NOISE_FREQ = 0.11                      # ~9 m eddies
NOISE_RATE = 0.35                      # how fast the field itself evolves, Hz-ish


@wp.func
def gas_noise(f: wp.array3d(dtype=wp.vec3), p: wp.vec3) -> wp.vec3:
    """Trilinear fetch from the baked curl grid; outside it, clamp to the edge."""
    gx = wp.clamp((p[0] - NCX + NR) * NG_SCALE, 0.0, float(NG - 1) - 1.0e-3)
    gy = wp.clamp((p[1] - NCY + NR) * NG_SCALE, 0.0, float(NG - 1) - 1.0e-3)
    gz = wp.clamp((p[2] - NCZ + NR) * NG_SCALE, 0.0, float(NG - 1) - 1.0e-3)
    i, j, k = int(wp.floor(gx)), int(wp.floor(gy)), int(wp.floor(gz))
    fx, fy, fz = gx - float(i), gy - float(j), gz - float(k)
    c00 = f[i, j, k] * (1.0 - fx) + f[i + 1, j, k] * fx
    c10 = f[i, j + 1, k] * (1.0 - fx) + f[i + 1, j + 1, k] * fx
    c01 = f[i, j, k + 1] * (1.0 - fx) + f[i + 1, j, k + 1] * fx
    c11 = f[i, j + 1, k + 1] * (1.0 - fx) + f[i + 1, j + 1, k + 1] * fx
    return ((c00 * (1.0 - fy) + c10 * fy) * (1.0 - fz)
            + (c01 * (1.0 - fy) + c11 * fy) * fz)


@wp.kernel
def bake_noise(t: float, f: wp.array3d(dtype=wp.vec3)):
    i, j, k = wp.tid()
    s = wp.rand_init(11)
    x = (float(i) / NG_SCALE - NR) * NOISE_FREQ
    y = (float(j) / NG_SCALE - NR) * NOISE_FREQ
    z = (float(k) / NG_SCALE - NR) * NOISE_FREQ
    f[i, j, k] = wp.curlnoise(s, wp.vec4(x, y, z, t * NOISE_RATE))


@wp.kernel
def emit_gas(pos: wp.array(dtype=wp.vec4),
             vel: wp.array(dtype=wp.vec3),
             st: wp.array(dtype=wp.vec2),
             charge: wp.array(dtype=wp.vec4),
             ci: int, base: int, cap: int, b: float,
             kind: int, seed: int, v0: float,
             life0: float, life1: float, r0: float, front_c: float):
    """Hand out `dim` ring slots starting at `base`, born of charge `ci`.

    Device-side, per frame, and the ONLY place a particle comes into existence.
    The ring cursor is the whole allocator: a slot is reused only after the
    cohort has emitted `cap` more particles, which for the shipped budgets is
    longer than any lifetime -- so recycling is free and there is no free list
    to compact and no atomic to contend on. `charge` is a fixed 8-slot array
    (xyz + t0) so phase 2.5 only has to fill more of it.
    """
    i = wp.tid()
    slot = (base + i) % cap
    s = wp.rand_init(seed, base + i)
    ch = charge[ci]
    a = wp.randf(s) * 6.2831853
    q = wp.randf(s)                       # 0 = the core, 1 = the outer shell
    life = life0 + (life1 - life0) * wp.randf(s)
    if kind == 5:
        # THE CONDENSATION RING (Wilson cloud). Emitted ON the front -- a thin
        # annulus at exactly c*b -- and thrown outward AT the front speed, which
        # is the part that matters: a ring whose parcels stand still spreads into
        # a band as wide as (speed x lifetime), 25 m at these numbers. Riding the
        # wave instead, each parcel stays on the front for the quarter second it
        # lives, so what expands is a ring and not a disc. It sits a metre or so
        # up, above the dust skirt the same front is tearing off the ground.
        rr = front_c * b * (0.985 + 0.030 * wp.randf(s))
        p = wp.vec3(ch[0] + rr * wp.cos(a), 0.85 + 2.10 * wp.randf(s),
                    ch[2] + rr * wp.sin(a))
        sp = v0 * (0.92 + 0.16 * wp.randf(s))
        v = wp.vec3(wp.cos(a) * sp, sp * 0.04 * wp.randf(s), wp.sin(a) * sp)
    elif kind == 2:
        # Dust rides the shock front: a slot emitted at b seconds after the
        # charge appears on the ground annulus of radius c*b, which IS the ring
        # racing outward. Ground-hugging by construction -- the kick is almost
        # all RADIAL, the vertical part is a tenth of it, gravity is full g.
        rr = front_c * b * (0.90 + 0.15 * wp.randf(s))
        p = wp.vec3(ch[0] + rr * wp.cos(a), 0.05 + 0.60 * wp.randf(s),
                    ch[2] + rr * wp.sin(a))
        sp = v0 * (0.45 + 0.85 * wp.randf(s))
        v = wp.vec3(wp.cos(a) * sp, sp * (0.06 + 0.30 * wp.randf(s)),
                    wp.sin(a) * sp)
    else:
        z = 2.0 * wp.randf(s) - 1.0
        rxy = wp.sqrt(wp.max(1.0 - z * z, 0.0))
        d = wp.vec3(rxy * wp.cos(a), z, rxy * wp.sin(a))
        p = wp.vec3(ch[0], ch[1], ch[2]) + d * (0.20 + 0.95 * wp.pow(q, 0.3333))
        sp = v0 * (0.35 + 0.95 * wp.randf(s))
        v = wp.normalize(wp.vec3(d[0], d[1] + 0.32, d[2])) * sp
        if kind == 0 or kind == 4:
            # A fire parcel's "life" IS its cooling time, and the core cools
            # SLOWEST: q ~ 0 is deep inside the ball and holds its warmth,
            # q ~ 1 is the outer shell and turns to soot almost at once. That
            # gradient is what makes the conversion below read as a fireball
            # peeling into smoke from the outside in, rather than a population
            # that all goes dark together.
            life = life1 + (life0 - life1) * q
    pos[slot] = wp.vec4(p[0], p[1], p[2], r0)
    vel[slot] = v
    st[slot] = wp.vec2(0.0, life)


@wp.kernel
def step_gas(pos: wp.array(dtype=wp.vec4),
             vel: wp.array(dtype=wp.vec3),
             st: wp.array(dtype=wp.vec2),
             noise: wp.array3d(dtype=wp.vec3),
             dt: float, kind: int, seed: int,
             drag: float, buoy: float, buoy_tau: float, grav: float,
             curl: float, r0: float, r1: float, r_pow: float,
             s_pos: wp.array(dtype=wp.vec4),
             s_vel: wp.array(dtype=wp.vec3),
             s_st: wp.array(dtype=wp.vec2),
             sk: int, s_cap: int, s_life0: float, s_life1: float,
             s_r0: float, s_inherit: float, s_spread: float):
    """Advect one cohort, and turn cooled fire into smoke.

    kind: 0 fire, 1 smoke, 2 dust, 3 ember, 4 flare. The dead sentinel is the
    one rule every consumer of the position buffer tests -- w < 0 -- so a dead
    or never-emitted slot costs this kernel a single 16-byte load and an exit.

    THE CONVERSION is the model: nothing emits smoke. A fire parcel that
    reaches its cooling time dies and writes SMOKE_K soot parcels into the
    slots statically mapped to it, at its own position, carrying `s_inherit` of
    its velocity. The outward momentum of the fireball becomes the outward push
    of the smoke, and only then does buoyancy roll it over. The mapping
    (fire slot i -> smoke slots [i*K, i*K+K)) is what lets the smoke pool
    inherit the fire ring's recycling without a free list of its own.
    """
    i = wp.tid()
    q = pos[i]
    if q[3] < 0.0:
        return
    a = st[i]
    age = a[0] + dt
    life = a[1]
    p = wp.vec3(q[0], q[1], q[2])
    v = vel[i]
    if age >= life:
        pos[i] = wp.vec4(0.0, -1000.0, 0.0, -1.0)
        if kind == 0:
            s = wp.rand_init(seed + 7717, i)
            scat = q[3] * 1.4
            for k in range(sk):
                j = (i * sk + k) % s_cap
                o = wp.vec3(wp.randf(s) - 0.5, wp.randf(s) - 0.5, wp.randf(s) - 0.5)
                jv = wp.vec3(wp.randf(s) - 0.5, wp.randf(s) - 0.5, wp.randf(s) - 0.5)
                s_pos[j] = wp.vec4(p[0] + o[0] * scat, p[1] + o[1] * scat,
                                   p[2] + o[2] * scat, s_r0)
                s_vel[j] = v * s_inherit + jv * s_spread
                s_st[j] = wp.vec2(0.0, s_life0 + (s_life1 - s_life0) * wp.randf(s))
        return
    # ── advect ──────────────────────────────────────────────────────────────
    acc = gas_noise(noise, p) * curl
    acc = acc + wp.vec3(0.0, buoy * wp.exp(-age / buoy_tau) - grav, 0.0)
    acc = acc - v * drag
    v = v + acc * dt
    p = p + v * dt
    # The yard's floor, cheaply -- but dust gets a per-particle REST HEIGHT
    # spread over most of a metre instead of one shared plane. Two million
    # settled particles pinned to a single 10 cm layer is a sheet three voxels
    # thick and optically solid, and the froxel march steps ~2 m: the result is
    # a carpet of dither speckle across the whole yard that no amount of sigma
    # tuning fixes, because the sheet is opaque at any sigma. Spread over ~20
    # voxel layers it reads as settling dust and samples cleanly.
    fl = float(0.10)
    if kind == 2:
        sf = wp.rand_init(seed + 31, i)
        fl = 0.10 + 0.90 * wp.randf(sf)
    if p[1] < fl:
        p = wp.vec3(p[0], fl, p[2])
        # Dust that reaches the ground STAYS on it -- no bounce, or a settling
        # ring turns back into a fountain one frame later.
        if kind == 3:
            v = wp.vec3(0.0, 0.0, 0.0)        # a landed ember stays put and burns out
        else:
            vy = float(0.0)
            if kind != 2:
                vy = wp.abs(v[1]) * 0.20
            v = wp.vec3(v[0] * 0.92, vy, v[2] * 0.92)
    # Radius over life. r_pow shapes it: 1 is the linear growth the gas wants,
    # and > 1 with r1 < r0 is the ember's hold-then-drop -- the only fade an
    # Interop or HostRing field HAS, because age fade is the analytic emitter's
    # and there is no emitter here.
    f = wp.pow(age / life, r_pow)
    pos[i] = wp.vec4(p[0], p[1], p[2], r0 + (r1 - r0) * f)
    vel[i] = v
    st[i] = wp.vec2(age, life)


@wp.kernel
def sample_gas(pos: wp.array(dtype=wp.vec4), out: wp.array(dtype=wp.vec4), stride: int):
    """A strided sample of a cohort, for the tuning report only.

    The whole point of the interop path is that positions never cross the bus,
    so the diagnostic that used to read the submit buffer has to pull its own
    few thousand slots. Called from report(), never per frame.
    """
    i = wp.tid()
    out[i] = pos[i * stride]


DEAD = wp.vec4(0.0, -1000.0, 0.0, -1.0)
# Seconds spent inside the renderer's per-frame device-to-device callbacks. It
# is a list because it is written from inside render(), before `prof` exists.
GAS_COPY = [0.0, 0.0, 0.0]      # total, issue, synchronize


class Cohort:
    """One ParticleField plus the Warp state that feeds it.

    The device buffer is `wp.vec4`, which is byte-identical to ParticlePos, so
    on the interop path the renderer's exported allocation and this sim's own
    positions are the same 16-byte layout and the per-frame publish is one
    cuMemcpyDtoD with no repack and no host round trip. On the fallback path the
    same buffer is read back and memcpy'd into the field's host ring instead,
    which is why the counts drop by 25x there.

    `live` is a PREFIX of the ring: the cursor is monotonic, so slots
    [0, min(cursor, capacity)) is exactly the set that has ever been handed out.
    Everything -- the advect launch, the device copy, the field's live count --
    is trimmed to it, so the first frames after t0 cost a fraction of full
    capacity and the pre-detonation frames cost nothing at all.
    """

    def __init__(self, name, n, kind, emit, life, radius, seed,
                 skew=1.0, r_pow=1.0, per_charge=None, front_c=0.0):
        self.name, self.n, self.kind, self.r_pow = name, n, kind, r_pow
        self.emit_win, self.life, self.radius, self.skew = emit, life, radius, skew
        self.seed, self.front_c = seed, front_c
        self.per_charge = n if per_charge is None else per_charge
        self.pos = wp.empty(n, dtype=wp.vec4, device=device)
        self.pos.fill_(DEAD)                 # w < 0: nothing is alive until it is emitted
        self.vel = wp.zeros(n, dtype=wp.vec3, device=device)
        self.st = wp.zeros(n, dtype=wp.vec2, device=device)
        self.emitted = [0] * MAX_CHARGES     # per charge, so the ring cursor is exact
        self.cursor = 0
        self.live = 0
        self.field = None
        self.imported = None                 # VkInteropArray on the zero-copy path
        self.host = self.host_np = None      # the fallback's staging buffer
        self.params = ()

    # ── emission ─────────────────────────────────────────────────────────────
    def emit(self, ci, b):
        """Emit charge `ci`'s share due by `b` seconds after it fired.

        The cumulative form is what makes this exact and stateless per frame:
        `emitted(tau) = N * (tau/EMIT)**(1/skew)` is the same distribution phase
        2 drew as sorted birth times, evaluated instead of stored -- and unlike
        the sorted prefix it survives charges firing out of order.
        """
        if b <= 0.0 or self.per_charge == 0:
            return 0
        u = min(b / self.emit_win, 1.0)
        want = int(self.per_charge * u ** (1.0 / self.skew))
        n = want - self.emitted[ci]
        if n <= 0:
            return 0
        self.emitted[ci] = want
        wp.launch(emit_gas, dim=n,
                  inputs=[self.pos, self.vel, self.st, charge_arr, ci,
                          self.cursor, self.n, b, self.kind,
                          self.seed, self.params[0], self.life[0], self.life[1],
                          self.radius[0], self.front_c],
                  device=device)
        self.cursor += n
        self.live = min(self.cursor, self.n)
        return n

    # ── advection ────────────────────────────────────────────────────────────
    def advance(self, dt, child=None):
        if not self.live:
            return
        c = child if child is not None else self
        wp.launch(step_gas, dim=self.live,
                  inputs=[self.pos, self.vel, self.st, noise_grid, dt, self.kind,
                          self.seed, *self.params[1:],
                          self.radius[0], self.radius[1], self.r_pow,
                          c.pos, c.vel, c.st, SMOKE_K, c.n,
                          SMOKE_LIFE[0], SMOKE_LIFE[1], SMOKE_R[0],
                          SMOKE_INHERIT, SMOKE_SPREAD],
                  device=device)

    # ── publish ──────────────────────────────────────────────────────────────
    def device_copy(self):
        """The renderer's per-frame callback: one device-to-device copy of the
        live prefix, then a synchronize. It runs INSIDE render(), pre-record,
        and the host ordering it provides is the ONLY thing sequencing this
        write against the frame that reads it -- there is no shared semaphore.
        """
        t = time.perf_counter()
        if self.live:
            wp.copy(self.imported.array, self.pos, count=self.live)
        t1 = time.perf_counter()
        wp.synchronize_device(device)
        t2 = time.perf_counter()
        GAS_COPY[0] += t2 - t
        GAS_COPY[1] += t1 - t
        GAS_COPY[2] += t2 - t1

    def publish(self):
        """One integer on the interop path. A readback plus a memcpy otherwise."""
        if self.imported is not None or self.host is None or not self.live:
            self.field.set_live_count(self.live)
            return
        wp.copy(self.host, self.pos, count=self.live)
        wp.synchronize_device(device)
        self.field.submit(self.host_np[:self.live])

    def reset(self):
        self.pos.fill_(DEAD)
        self.emitted = [0] * MAX_CHARGES
        self.cursor = 0
        self.live = 0

    def sample(self, k=4096):
        """A strided read of the live prefix, for report() only."""
        if self.live < 2:
            return None
        stride = max(self.live // k, 1)
        n = self.live // stride
        out = wp.empty(n, dtype=wp.vec4, device=device)
        wp.launch(sample_gas, dim=n, inputs=[self.pos, out, stride], device=device)
        return out.numpy()


cohorts = []
fields = []

if not NO_GAS:
    noise_grid = wp.zeros((NG, NG, NG), dtype=wp.vec3, device=device)
    _ch = np.zeros((MAX_CHARGES, 4), np.float32)
    charge_arr = wp.array(_ch, dtype=wp.vec4, device=device)

    # per_charge: how much of a pool ONE detonation may claim. An overlapping
    # ninth burst simply overwrites the oldest slots in the ring.
    fire = Cohort("fire", FIRE_N, 0, FIRE_EMIT, FIRE_LIFE, FIRE_R, SEED + 1,
                  per_charge=FIRE_N // N_SHARES)
    smoke = Cohort("smoke", SMOKE_N, 1, 1.0, SMOKE_LIFE, SMOKE_R, SEED + 2,
                   per_charge=0)        # NEVER emitted: fire converts into it
    dust = Cohort("dust", DUST_N, 2, DUST_EMIT, DUST_LIFE, DUST_R, SEED + 3,
                  per_charge=DUST_N // N_SHARES, front_c=DUST_C)
    ember = Cohort("ember", EMBER_N, 3, EMBER_EMIT, EMBER_LIFE, EMBER_R, SEED + 4,
                   skew=EMBER_SKEW, r_pow=1.5, per_charge=EMBER_N // N_SHARES)
    flare = Cohort("flare", FLARE_N, 4, FLARE_EMIT, FLARE_LIFE, FLARE_R, SEED + 5,
                   skew=FLARE_SKEW, per_charge=FLARE_N // N_SHARES)
    wilson = Cohort("wilson", WILSON_N, 5, WILSON_EMIT, WILSON_LIFE, WILSON_R,
                    SEED + 6, per_charge=WILSON_N // N_SHARES, front_c=C_FRONT)
    # (v0, drag, buoy, buoy_tau, grav, curl) -- v0 is the emit kernel's, the
    # rest are the advect kernel's, in its argument order.
    fire.params = (FIRE_V0, FIRE_DRAG, FIRE_BUOY, 0.30, 0.0, FIRE_CURL)
    smoke.params = (0.0, SMOKE_DRAG, SMOKE_BUOY, 2.00, 0.0, SMOKE_CURL)
    dust.params = (DUST_V0, DUST_DRAG, 0.0, 1.0, DUST_GRAV, DUST_CURL)
    ember.params = (EMBER_V0, EMBER_DRAG, 0.0, 1.0, 9.81, EMBER_CURL)
    flare.params = (FLARE_V0, FLARE_DRAG, FLARE_BUOY, 0.26, 0.0, FLARE_CURL)
    wilson.params = (C_FRONT, WILSON_DRAG, WILSON_BUOY, 0.5, 0.0, WILSON_CURL)
    # fire is advected FIRST and smoke LAST, so a parcel that converts this
    # frame is advected the same frame instead of hanging for one.
    cohorts = [fire, flare, dust, ember, wilson, smoke]
    emitters = [fire, flare, dust, ember, wilson]

    OWNERSHIP = (tp.ParticleField.Ownership.Interop if INTEROP
                 else tp.ParticleField.Ownership.HostRing)

    def make_field(n, radius):
        c = tp.ParticleField.Config()
        c.capacity = n
        c.ownership = OWNERSHIP
        c.w_semantic = tp.ParticleField.WSemantic.Radius   # w IS the world radius
        c.uniform_radius = radius
        f = tp.ParticleField.create(c)
        f.frustum_culled = False        # the bounds go stale the moment it detonates
        f.set_live_count(0)
        scene.add(f)
        return f

    cx, cy, cz = CHARGE
    VOLUMES = not NO_FOG                # DensityRepr rides the froxel FOG pass

    # ── fire ────────────────────────────────────────────────────────────────
    # The flame is the DENSITY volume's blackbody ramp, not the billboards: the
    # ramp is hot at the box FLOOR and cools toward its top, so the box has to
    # be the fireball and nothing else -- a box the size of the yard would put
    # the whole flame in the bottom two per cent of the ramp.
    fire.field = make_field(FIRE_N, FIRE_R[1])
    if VOLUMES:
        fire.field.set_density_repr(tp.Vector3(cx, 3.6, cz), tp.Vector3(9.0, 4.6, 9.0),
                                    FIRE_SIGMA * SIGMA, FIRE_RES)
        _d = fire.field.density_repr
        _d.albedo = tp.Color(0.30, 0.22, 0.17)
        _d.anisotropy = 0.15
        _d.temp_bottom_k = 2700.0
        _d.temp_top_k = 1250.0
        _d.temp_falloff = 1.5
        _d.emissive_intensity = 0.0     # driven by the timeline
    # NO billboards on the fire field at this scale. Three million additive
    # quads inside a four-metre ball is not a look, it is a fill-rate accident:
    # the overdraw is hundreds deep and the result is a flat white ball. The
    # `flare` cohort below is the fire SUBSET that carries the additive layer.

    # ── smoke ───────────────────────────────────────────────────────────────
    # DensityRepr only -- nine million additive quads would be worse than three.
    # Soot is a DARK medium; a bright albedo here and the column reads as steam.
    smoke.field = make_field(SMOKE_N, SMOKE_R[1])
    if VOLUMES:
        # The box has to contain the WHOLE plume: soot outside it is not
        # scattered and simply is not there, which reads as a cloud with its top
        # sheared off. 48 x 30 x 48 m at 192**3 is a 25 cm voxel.
        smoke.field.set_density_repr(tp.Vector3(0.0, 13.0, 0.0),
                                     tp.Vector3(24.0, 15.0, 24.0),
                                     SMOKE_SIGMA * SIGMA, SMOKE_RES)
        _d = smoke.field.density_repr
        _d.albedo = tp.Color(*SMOKE_ALBEDO)
        _d.anisotropy = 0.28

    # ── dust ────────────────────────────────────────────────────────────────
    dust.field = make_field(DUST_N, DUST_R[1])
    if VOLUMES:
        dust.field.set_density_repr(tp.Vector3(0.0, 1.5, 0.0),
                                    tp.Vector3(36.0, 2.6, 36.0),
                                    DUST_SIGMA * SIGMA, DUST_RES)
        _d = dust.field.density_repr
        _d.albedo = tp.Color(0.50, 0.43, 0.33)   # the granular demo's "dirty", verbatim
        _d.anisotropy = 0.10
    # NO billboards. The plan asked for "very dim billboards for sunlit
    # glints", and they were tried: once the ring is genuinely ground-hugging,
    # 170k additive quads packed into a half-metre layer read as a carpet of
    # static across the whole yard, not as glinting grit. The extinction volume
    # is the honest representation for dust and it is the only one here.

    # ── the condensation ring ───────────────────────────────────────────────
    # The FOURTH and last density slot (kMaxDensityFields is 4). A real blast
    # front briefly drops the pressure behind it enough to condense the air's
    # water into a visible shell -- the Wilson cloud -- and that shell is the
    # only part of a shockwave you can actually photograph. Here it is a thin
    # bright annulus riding r = C_FRONT * t, translucent and short-lived, ABOVE
    # the dust skirt the same front is tearing off the ground.
    #
    # This is front-tracking emission, the one exception to "nothing spawns
    # after the burst" -- the same exception the dust ring already holds, and
    # for the same reason: the wave is still travelling, so its edge is still
    # making matter. It stops at WILSON_EMIT and nothing is emitted after it.
    wilson.field = make_field(WILSON_N, WILSON_R[1])
    if VOLUMES:
        _reach = C_FRONT * WILSON_EMIT + 6.0
        wilson.field.set_density_repr(tp.Vector3(0.0, 2.0, 0.0),
                                      tp.Vector3(_reach, 4.0, _reach),
                                      WILSON_SIGMA * SIGMA, WILSON_RES)
        _d = wilson.field.density_repr
        _d.albedo = tp.Color(0.90, 0.92, 0.96)   # condensed water, not soot
        _d.anisotropy = 0.35

    # ── flare: the fire subset that carries the additive layer ──────────────
    # A quarter of a million hot parcels on the same burst trajectory as the
    # fire, with a much shorter life, drawn as additive billboards with their
    # own bloom pyramid. This is how the flash survives the jump to nebula
    # counts: the DENSITY volume gets all three million fire parcels and the
    # BILLBOARDS get a subset small enough that the overdraw stays sane.
    flare.field = make_field(FLARE_N, FLARE_R[1])
    flare.field.set_billboard_repr(tp.Color(1.00, 0.63, 0.24), tp.Color(1.00, 0.20, 0.03),
                                   0.0, 0.40)
    _b = flare.field.billboard_repr
    _b.softness = 0.80
    _b.fade_power = 0.0                 # no age on an Interop field: w carries the life
    _b.size_taper = 0.0
    _b.bright_jitter = 0.70
    _b.near_fade = 1.5
    _b.glow = 1.10                      # this field's own bloom pyramid
    _b.glow_threshold = 0.0

    # ── embers ──────────────────────────────────────────────────────────────
    #
    # A device-fed field, not the analytic emitter, and that is a DEVIATION FROM
    # THE PLAN with a hard reason. Ownership::Renderer's emitter is a STEADY-STATE
    # generator by construction: particle_emit.comp gives every slot a random
    # phase into its own period (`float birth; // phase offset into the period,
    # [0,1)`), so at any emitter time a `dutyCycle` fraction of slots is alive
    # and WHICH slots those are changes continuously. Feed it a long period and
    # a short duty -- the obvious way to spell "one burst" -- and what you get
    # is a permanent thin drizzle of brand-new sparks out of the charge, minutes
    # after the flame is out. There is no parameterisation that fixes it: the
    # phase spread is scale-invariant, so no remapping of the clock turns a
    # steady state into a burst.
    #
    # What is lost by moving: age fade, size taper and the hot->cool colour ramp
    # are all derived from the emitter's closed form, and on a HostRing or
    # Interop field ageFrac is pinned to 0. The radius IS the fade here (an
    # additive quad's contribution goes as its area, so r -> 0 is a fade), and
    # because this is ONE burst rather than a steady state, the whole field ages
    # together -- so cooling the field's own colour over tw below is a faithful
    # stand-in for cooling each ember over its own age.
    ember.field = make_field(EMBER_N, EMBER_R[0])
    ember.field.set_billboard_repr(tp.Color(1.00, 0.78, 0.36), tp.Color(1.00, 0.17, 0.02),
                                   0.0, 1.0)
    _b = ember.field.billboard_repr
    _b.softness = 0.32
    _b.fade_power = 0.0                 # inert off the emitter: the radius is the fade
    _b.size_taper = 0.0
    _b.bright_jitter = 0.65
    _b.stretch_seconds = 0.030          # smeared along (pos - prevPos) of the ring
    _b.stretch_max = 34.0
    _b.stretch_max_screen = 0.055
    _b.near_fade = 1.2
    _b.glow = 0.50
    EMBER_HOT = ((1.00, 0.80, 0.40), (1.00, 0.30, 0.06))   # new ember -> old ember

    # ── streaks: the ONE thing a device-fed field cannot do ─────────────────
    # ParticleFieldPass.cpp is explicit: "the stretch is a Renderer-mode
    # feature and is silently zero elsewhere", because a HostRing field's
    # prevPositions are the previous RING SLOT and their age is never
    # published, so there is no dt to divide by. Round sprites are what the
    # ember field above can be, and a blast that throws no streaks reads wrong.
    #
    # So a SECOND, small Renderer field does the streaks -- and only for the
    # burst window. Its slots respawn continuously (that is the steady-state
    # emitter's nature, see the note above), which is exactly right while the
    # charge is still ejecting and exactly wrong afterwards, so the field's
    # intensity is ramped out over 0.45-0.80 s and it is PARKED at 0.85 s.
    # Nothing is instantiated after the burst; the HostRing embers carry the
    # rest of the story.
    # A RING OF THEM, one per concurrent charge and all created here. A Renderer
    # field is a closed form in ONE spawn_center and ONE clock, so two charges
    # cannot share one; and creating one per detonation is exactly the scene
    # churn (entry re-expansion, device idle, cleared TAA history) this whole
    # design exists to avoid. Three fields, assigned round-robin as charges fire,
    # a fourth burst stealing the oldest -- which by then has been parked for
    # its whole life anyway.
    STREAK_N = int(20_000 * GAS)
    STREAK_T = 0.85
    STREAK_RING = []
    for _k in range(3):
        _kc = tp.ParticleField.Config()
        _kc.capacity = STREAK_N
        _kc.ownership = tp.ParticleField.Ownership.Renderer
        _kc.w_semantic = tp.ParticleField.WSemantic.Radius
        _kc.uniform_radius = 0.05
        _s = tp.ParticleField.create(_kc)
        _s.frustum_culled = False
        _s.set_billboard_repr(tp.Color(1.00, 0.84, 0.46), tp.Color(1.00, 0.24, 0.03),
                              0.0, 1.0)
        _b = _s.billboard_repr
        _b.softness = 0.20
        _b.fade_power = 2.0             # a Renderer field HAS an age: hold, then drop
        _b.size_taper = 0.60
        _b.bright_jitter = 0.60
        _b.stretch_seconds = 0.032      # the exact analytic velocity, smeared
        _b.stretch_max = 34.0
        _b.stretch_max_screen = 0.055
        _b.near_fade = 1.2
        _b.glow = 0.70
        _e = _s.emitter                 # NB: a COPY -- mutate and hand it back
        _e.spawn_center = tp.Vector3(cx, cy + 0.2, cz)
        _e.spawn_half_extent = tp.Vector3(0.45, 0.45, 0.45)
        _e.velocity = tp.Vector3(0.0, 8.0, 0.0)   # the ground reflection's upward bias
        _e.speed_spread = 22.0                    # isotropic: this IS the radial burst
        _e.accel = tp.Vector3(0.0, -9.81, 0.0)
        _e.lifetime = STREAK_T
        _e.lifetime_jitter = 0.25
        _e.duty_cycle = 1.0
        _e.size = 0.05
        _e.size_jitter = 0.65
        _e.seed = SEED * 1013 + 5 + 97 * _k
        _s.set_emitter(_e)
        _s.set_emitter_time(0.0, DT)
        _s.set_live_count(0)
        scene.add(_s)
        STREAK_RING.append(_s)

    fields = [c.field for c in cohorts] + STREAK_RING
    # The last thing that can still be on screen: a fire parcel emitted at the
    # end of the burst, cooling for its full life, converting, and its soot
    # living out the longest smoke life. Nothing is emitted after DUST_EMIT.
    GAS_END = FIRE_EMIT + FIRE_LIFE[1] + SMOKE_LIFE[1] + 0.1
    EMIT_END = max(c.emit_win for c in emitters)     # nothing is born after this
    GAS_N = sum(c.n for c in cohorts)

    def claim_charge_slot(ch):
        """Give a charge its slot in the device charge array and its streak field.

        This is EVERYTHING a new detonation costs the gas: one 128-byte upload
        of the 8-slot array, five emission counters zeroed, one emitter's spawn
        centre moved (O(1) push constants). No allocation, no field creation, no
        device idle -- which is why the B key does not hitch.
        """
        ch.slot = _fired[0] % MAX_CHARGES
        _ch[ch.slot] = (ch.pos[0], ch.pos[1], ch.pos[2], ch.t0)
        charge_arr.assign(_ch)
        for c in cohorts:
            c.emitted[ch.slot] = 0        # the slot is re-used: start its budget over
        ch.streaks = STREAK_RING[_fired[0] % len(STREAK_RING)]
        e = ch.streaks.emitter
        e.spawn_center = tp.Vector3(ch.pos[0], ch.pos[1] + 0.2, ch.pos[2])
        ch.streaks.set_emitter(e)

    # ── ARM THE ZERO-COPY PATH ──────────────────────────────────────────────
    # enable_particle_field_interop returns None until after the FIRST render():
    # the field's device state and the renderer's field pass are both created on
    # the frame the field is first seen. So: render once (nothing is live, so
    # nothing is drawn), then export, then import into CUDA.
    #
    # Every leg of the failure path lands on the same fallback. No CUDA device,
    # --no-interop, an empty handle (the device cannot export memory), or a
    # CUDA-side import that throws: the fields were created HostRing in the
    # first two cases and the renderer has set host_fallback() in the third, so
    # submit() is legal either way and the counts were already scaled down.
    renderer.render(scene, camera)
    HOST_RING = not INTEROP             # is submit() a legal path for these fields?
    if INTEROP:
        from threepp.cuda_interop import VkInteropArray
        exported = False
        try:
            for c in cohorts:
                h = renderer.enable_particle_field_interop(c.field, c.device_copy)
                if h is None:
                    raise RuntimeError("this device cannot export memory "
                                       f"(host_fallback={c.field.host_fallback})")
                exported = True
                c.imported = VkInteropArray(h[0], h[1], wp.vec4, c.n, device)
            # Confirm the path is actually device-to-device by TIMING it: eight
            # full-capacity copies outside any frame. A number in the hundreds
            # of GB/s is device-local memory; anything near 20 would mean the
            # export had landed in host memory and the whole exercise was moot.
            for c in cohorts:
                wp.copy(c.imported.array, c.pos)
            wp.synchronize_device(device)
            _t = time.perf_counter()
            for _ in range(8):
                for c in cohorts:
                    wp.copy(c.imported.array, c.pos)
            wp.synchronize_device(device)
            _dt = (time.perf_counter() - _t) / 8.0
            print(f"gas: DIRECT device-to-device path armed -- {GAS_N:,} particles, "
                  f"{16 * GAS_N / 1e6:.0f} MB exported, 0 B/frame across the bus\n"
                  f"     full-capacity device copy: {1e3 * _dt:.2f} ms "
                  f"({32 * GAS_N / 1e9 / _dt:.0f} GB/s read+write)")
        except Exception as e:
            INTEROP = False
            for c in cohorts:
                c.imported = None
            if exported:
                # The renderer DID export and only the CUDA import failed, so
                # the fields are Interop and NOT in host_fallback: submit() will
                # throw on them and there is no second path from here. Park the
                # gas rather than crash, and name the flag that works.
                print(f"gas: CUDA could not import the export ({e}).\n"
                      f"     Re-run with --no-interop for the host-ring path.")
                for c in cohorts:
                    c.per_charge = 0
            else:
                # No external memory: the renderer has already put the fields in
                # host_fallback(), which makes submit() legal on them. Keep the
                # allocations (capacity is latched) but emit only what the bus
                # can carry -- 16 B/particle out and 16 B/particle back, twice a
                # frame, is what caps this path at a few hundred thousand.
                print(f"gas: no zero-copy export ({e}); host ring at {FALLBACK:.0%} "
                      f"of the counts")
                HOST_RING = True
                for c in cohorts:
                    c.per_charge = int(c.per_charge * FALLBACK)
    if HOST_RING:
        for c in cohorts:
            try:
                c.host = wp.zeros(c.n, dtype=wp.vec4, device="cpu", pinned=True)
            except TypeError:                           # older warp: plain host mem
                c.host = wp.zeros(c.n, dtype=wp.vec4, device="cpu")
            c.host_np = c.host.numpy()                  # a VIEW of the host array

    # ── PREWARM: one throwaway frame with every field LIVE ───────────────────
    # Creating the fields up front is necessary but not sufficient. A density
    # volume's image is allocated (and its resolution latched) the first frame
    # the field actually has live particles, and a billboard field's glow
    # pyramid allocates its offscreen target the first time it draws -- so with
    # nothing but set_live_count(0) at startup, ALL of that lands on the
    # detonation frame. Measured in the window at phase 2's counts: a 34.9 ms
    # worst frame at t0 against 6-9 ms once the gas is running, and at fifteen
    # million a 192**3 volume allocation is not something to do mid-shot. One
    # particle each, for one frame nobody sees, moves it to startup.
    _warm = wp.array(np.array([[0.0, 1.0, 0.0, 0.01]], np.float32),
                     dtype=wp.vec4, device=device)
    for c in cohorts:
        wp.copy(c.pos, _warm, count=1)
        c.live = 1
        c.publish()
    for _s in STREAK_RING:
        _s.set_live_count(STREAK_N)
    renderer.render(scene, camera)
    for c in cohorts:
        c.reset()
        c.field.set_live_count(0)
    for _s in STREAK_RING:
        _s.set_live_count(0)
    print(f"gas: fire {FIRE_N:,} -> smoke {SMOKE_N:,} (x{SMOKE_K} on cooling) "
          f"+ dust {DUST_N:,} + ember {EMBER_N:,} + flare {FLARE_N:,} "
          f"+ ring {WILSON_N:,} = {GAS_N:,} particles "
          f"({52 * GAS_N / 1e6:.0f} MB resident), on {device}")
    print(f"charges: {len(CHARGES)} at startup, {N_SHARES} shares of every pool, "
          f"{MAX_CHARGES} device slots, front {C_FRONT:.0f} m/s")
else:
    GAS_END = 0.0
    GAS_N = 0
    STREAK_RING = []

    def claim_charge_slot(ch):
        ch.slot = _fired[0] % MAX_CHARGES

# Billboards are composited after the upscaler and are outside auto-exposure,
# so their intensity is tied to the pinned scene exposure BY HAND, once, here.
# Change tone_mapping_exposure and these follow it instead of blowing out.
_EXP = 0.55 / max(renderer.tone_mapping_exposure, 1e-3)
# Scaled off phase 2's tuned values by the nebula's 1/sqrt(N) law: the flare
# cohort is 250 k against the old fire field's 120 k, the ember field 500 k
# against 44 k. Overlapping additive quads saturate long before they sum, which
# is why the exponent is a half and not one.
BB_FLARE = 0.038 * _EXP    # low ON PURPOSE: the FLAME is the density ramp.
BB_EMBER = 0.40 * _EXP     # The billboards are only its sparkle.
BB_STREAK = 1.6 * _EXP
FIRE_EMISSIVE = 4.5                     # emission is intensity * THIS field's sigma
_gas_live = False


def step_gas_frame(dt):
    """Advance every cohort and publish it. Returns (kernel, publish) seconds.

    On the interop path `publish` is five integers and the real cost is the
    device_copy the renderer invokes inside render(); on the fallback it is the
    readback plus the ring memcpy, which is why it is still its own column.
    """
    global _gas_live
    if NO_GAS:
        return 0.0, 0.0
    state = blast_timeline(sim_time)
    tw = state["tw"]
    if not state["armed"] or tw > GAS_END:
        if _gas_live:                   # park, do not destroy: one entry, no churn
            for c in cohorts:
                c.live = 0
                c.field.set_live_count(0)
            for s in STREAK_RING:
                s.set_live_count(0)
            _gas_live = False
        return 0.0, 0.0
    _gas_live = True
    t0 = time.perf_counter()
    wp.launch(bake_noise, dim=(NG, NG, NG), inputs=[tw, noise_grid], device=device)
    # EMISSION. The host loop is over CHARGES, never over particles: per cohort
    # per charge it computes one integer and hands the ring cursor to a launch.
    # A charge past its longest emission window is skipped entirely -- emit()
    # would return 0 anyway, but a burst that finished five seconds ago should
    # not cost five cohort calls a frame for the rest of the shot.
    for ch in charges:
        if not ch.armed:
            continue
        b = sim_time - ch.t0
        if b > EMIT_END:
            continue
        for c in emitters:
            c.emit(ch.slot, b)
    # Smoke slots are claimed by conversion, and the mapping is static, so the
    # live prefix of the smoke pool is exactly the fire ring's prefix times K.
    smoke.live = min(fire.cursor * SMOKE_K, smoke.n)
    for c in cohorts:
        c.advance(dt, child=smoke)
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    for c in cohorts:
        c.publish()
    t2 = time.perf_counter()

    gain = state["bb_gain"]
    flare.field.billboard_repr.intensity = BB_FLARE * gain
    if not NO_FOG:
        # NEVER exactly zero while the gas is live. Any emissive volume switches
        # the per-pixel dust march from 24 midpoint steps to 32 dithered ones,
        # and the midpoint form quantises the smoke column into hard horizontal
        # shells -- 2 m steps through a 10 m plume. The dither TAA converges;
        # the shells it does not. The floor costs nothing because the fire
        # volume's own sigma is zero once the flame's particles are dead, and
        # emission is intensity * THAT volume's sigma.
        fire.field.density_repr.emissive_intensity = max(FIRE_EMISSIVE * state["fire_e"],
                                                         1.0e-3)
    # One burst ages as one population, so cooling the FIELD is a faithful
    # stand-in for the per-particle colour ramp a device-fed field cannot have.
    k = min(tw / 2.6, 1.0)
    hot, cool = EMBER_HOT
    ember.field.billboard_repr.color_hot = tp.Color(*(a + (b - a) * k
                                                      for a, b in zip(hot, cool)))
    ember.field.billboard_repr.intensity = (BB_EMBER * (1.0 + 0.6 * (gain - 1.0))
                                            * (1.0 - 0.55 * k))
    # The streak fields eject only while their own charge is still ejecting.
    # Park them all first and let the live charges claim theirs back: with the
    # ring shared, a charge whose field has been stolen by a newer one must not
    # be the one that parks it, and iterating oldest-to-newest means the newest
    # claim is the one that stands.
    for s in STREAK_RING:
        s.set_live_count(0)
    for ch in charges:
        if ch.streaks is None:
            continue
        b = sim_time - ch.t0
        if 0.0 <= b < STREAK_T:
            ch.streaks.set_emitter_time(b, dt)
            ch.streaks.set_live_count(STREAK_N)
            ch.streaks.billboard_repr.intensity = (
                BB_STREAK * (1.0 + FLASH_BB * math.exp(-b / 0.13))
                * min(max((0.80 - b) / 0.35, 0.0), 1.0))
    return t1 - t0, t2 - t1


# --- frame ------------------------------------------------------------------

sim_time = 0.0
prof = {"gas": 0.0, "submit": 0.0, "blast": 0.0, "physx": 0.0, "render": 0.0, "n": 0}


def step_frame(dt=DT):
    """Advance one frame of sim time. Returns the bench's phase durations.

    `dt` is a parameter, not a constant, because phase 4 ramps sim time through
    the detonation for the slow-motion cut.
    """
    global sim_time
    sim_time += dt
    if AUTO_CHARGE > 0.0 and sim_time > _auto[0]:
        _auto[0] = sim_time + AUTO_CHARGE
        name, pos = intact_target()
        if pos is not None:
            fire_charge(pos, why=f"{name} [auto]")
    t0 = time.perf_counter()
    blast_frame(sim_time)      # arm what is due, deliver the front, blow the wind
    t1 = time.perf_counter()
    world.step(dt)
    t2 = time.perf_counter()
    gas, submit = step_gas_frame(dt)
    prof["blast"] += t1 - t0
    prof["physx"] += t2 - t1
    prof["gas"] += gas
    prof["submit"] += submit
    prof["n"] += 1
    return gas, submit, t1 - t0, t2 - t1


def apply_shake():
    """Offset the camera by the current shake; returns the offset applied."""
    sx, sy, sz = blast_timeline(sim_time)["shake"]
    camera.position.set(camera.position.x + sx, camera.position.y + sy,
                        camera.position.z + sz)
    return sx, sy, sz


def yard_stats():
    """Sanity: nothing at NaN, nothing a kilometre away, how much is still moving."""
    p = body_positions()
    v = np.array([[b.linear_velocity.x, b.linear_velocity.y, b.linear_velocity.z]
                  for b in bodies])
    finite = np.isfinite(p).all(axis=1)
    r = np.linalg.norm(p[finite], axis=1)
    moving = int((np.linalg.norm(v[finite], axis=1) > 1.0).sum())
    return dict(nan=int((~finite).sum()), r_max=float(r.max()),
                y_max=float(p[finite][:, 1].max()), moving=moving)


def gas_stats():
    """Where the gas actually IS -- the only honest way to tune sigma, since a
    column that is twice as wide as you think is four times more transparent.

    A STRIDED SAMPLE of the live prefix, scaled back up, because on the interop
    path the positions never cross the bus and there is no host mirror to read.
    """
    out = {}
    for c in cohorts:
        p = c.sample()
        if p is None:
            continue
        live = p[:, 3] >= 0.0
        if not live.any():
            continue
        q = p[live]
        r = np.hypot(q[:, 0], q[:, 2])
        out[c.name] = (int(round(live.mean() * c.live)), float(np.median(q[:, 1])),
                       float(np.percentile(q[:, 1], 95)), float(np.median(r)),
                       float(np.percentile(r, 95)))
    return out


def report():
    s = yard_stats()
    n = max(prof["n"], 1)
    live = sum(f.live_count for f in fields)
    print(f"  t={sim_time:5.2f}  nan={s['nan']}  r_max={s['r_max']:6.1f} m  "
          f"y_max={s['y_max']:5.2f} m  moving={s['moving']}/{len(bodies)}  "
          f"gas live={live:,}")
    for name, (k, ymed, y95, rmed, r95) in gas_stats().items():
        print(f"  {name:>5}: {k:7,} live  y {ymed:5.1f}/{y95:5.1f} m  "
              f"r {rmed:5.1f}/{r95:5.1f} m  (median/p95)")
    print(f"  per frame: gas {1e3 * prof['gas'] / n:.2f} ms, publish "
          f"{1e3 * prof['submit'] / n:.2f} ms, device_copy "
          f"{1e3 * GAS_COPY[0] / n:.2f} ms ({1e3 * GAS_COPY[1] / n:.2f} issue + "
          f"{1e3 * GAS_COPY[2] / n:.2f} sync), blast {1e3 * prof['blast'] / n:.2f} ms, "
          f"physx {1e3 * prof['physx'] / n:.2f} ms, render "
          f"{1e3 * prof['render'] / n:.2f} ms")


def save(path):
    renderer.save_frame(scene, camera, path)      # renders + reads back


# --- run --------------------------------------------------------------------

orbit(0)

if SHOT:
    frames = int(round(SHOT_T * FPS))
    t_start = time.perf_counter()
    for f in range(frames):
        orbit(f)
        step_frame()
        apply_shake()
        tr = time.perf_counter()
        renderer.render(scene, camera)            # keep the temporal history honest
        prof["render"] += time.perf_counter() - tr
        if f % 60 == 0:
            print(f"  t={sim_time:5.2f}  moving={yard_stats()['moving']}", flush=True)
    if SPIN == 0.0:
        for _ in range(16):                       # settle the accumulation on the last pose
            renderer.render(scene, camera)
    else:
        orbit(frames)
    out = cli_arg("--out", "warp_explosion.png", str)
    save(out)
    print(f"simulated {SHOT_T:.1f} s ({frames} frames) in "
          f"{time.perf_counter() - t_start:.1f}s, wrote {out}")
    report()
elif VIDEO > 0:
    outdir = tempfile.mkdtemp(prefix="warp_explosion_frames_")
    total = int(round(VIDEO * FPS))
    t_start = time.perf_counter()
    for _ in range(24):                           # warm the temporal history on frame 0
        renderer.render(scene, camera)
    for k in range(total):
        orbit(k)
        step_frame()
        apply_shake()
        save(os.path.join(outdir, f"f{k:05d}.png"))
        if k % 60 == 0:
            print(f"  frame {k}/{total}  ({time.perf_counter() - t_start:.0f}s elapsed)",
                  flush=True)
    print(f"rendered {total} frames in {time.perf_counter() - t_start:.0f}s -> {outdir}")
    report()
    ff = find_ffmpeg()
    if ff:
        subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "60",
                        "-i", os.path.join(outdir, "f%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17",
                        "warp_explosion.mp4"], check=False)
        print("wrote warp_explosion.mp4")
elif BENCH:
    # Bench the interesting state: bricks in flight, not a sleeping stack.
    # Headless on purpose (vsync off, no present), which also means each
    # render() waits on its own frame instead of pipelining -- so the render
    # column is a per-frame GPU upper bound, not the window's throughput. The
    # window prints its own live fps; that is the number to quote.
    while sim_time < T0 + 0.6:
        step_frame()
        renderer.render(scene, camera)
    prof.update(gas=0.0, submit=0.0, blast=0.0, physx=0.0, render=0.0, n=0)
    GAS_COPY[:] = [0.0, 0.0, 0.0]
    live = sum(f.live_count for f in fields)
    bench_loop(step_frame, lambda: renderer.render(scene, camera),
               ("gas", "submit", "blast", "physx"), 20, 200,
               f"{len(bodies)} bodies + {live:,} live particles (headless, serialized)")
    report()
else:
    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target.set(*CAM_TARGET)
    canvas.on_window_resize(resize_handler(camera, renderer))
    fps = {"t": time.perf_counter(), "n": 0, "shake": (0.0, 0.0, 0.0),
           "last": time.perf_counter(), "worst": 0.0}
    held = {"B": False, "D": False}
    print("keys: B = detonate at a structure that is still standing, "
          "D = detonate at the orbit target")

    def poll_keys():
        """B and D, on the edge -- is_key_down is a poll, not an event."""
        for k in ("B", "D"):
            down = canvas.is_key_down(k)
            if down and not held[k]:
                if k == "D":
                    t = controls.target
                    fire_charge((t.x, 0.6, t.z), why="orbit target [D]")
                else:
                    name, pos = intact_target()
                    if pos is None:
                        print("  nothing left standing to blow up", flush=True)
                    else:
                        fire_charge(pos, why=f"{name} [B]")
            held[k] = down

    def animate():
        now0 = time.perf_counter()
        fps["worst"] = max(fps["worst"], now0 - fps["last"])
        fps["last"] = now0
        poll_keys()
        step_frame()
        # Undo last frame's shake before the controls read the camera back.
        sx, sy, sz = fps["shake"]
        camera.position.set(camera.position.x - sx, camera.position.y - sy,
                            camera.position.z - sz)
        controls.update()
        fps["shake"] = apply_shake()
        renderer.render(scene, camera)
        fps["n"] += 1
        if fps["n"] >= 120:
            now = time.perf_counter()
            # The worst frame in the window is the interesting one: a field
            # created mid-run would show up here as a device idle, and the
            # whole point of parking all four at startup is that it never does.
            print(f"  t={sim_time:5.2f}  {fps['n'] / (now - fps['t']):5.1f} fps  "
                  f"(worst {1e3 * fps['worst']:5.1f} ms, {len(bodies)} bodies, "
                  f"{sum(f.live_count for f in fields):,} particles)", flush=True)
            fps["t"], fps["n"], fps["worst"] = now, 0, 0.0

    canvas.animate(animate)
