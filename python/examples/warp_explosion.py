"""Blast Yard: a brick test range demolished by a charge -- PhysX + Warp + Vulkan.

A light-grey void with a grid floor, the way a test range looks in Unreal or
Omniverse, and a yard full of things that are *pre-broken by construction* --
a three-walled brick house, a shed, a ten-metre chimney, a stone gateway, a
concrete bunker, a rack of steel pipe (capsule bodies), pillars, pallets of
crates, boundary walls, loose rubble. Roughly seventeen hundred individual
PhysX bodies, laid in staggered courses with millimetre gaps so nothing starts
interpenetrating, settled for two seconds, spread out so every charge has its
own victim -- and then hit.

The void is a GRADE, not a colour: `--grade` moves the sky, the floor tone, the
exposure, the fill and the sun together, because they set where black and white
are and the sun is also what lights the smoke. The default puts the sky a stop
under the floor and the floor a stop under the plume, so the fireball, the column
and the embers are the only things at the top of the range. `--grade flat` is the
near-white sky this had before, kept for comparison; it is what "washed out"
looks like.

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
  ring   The Wilson cloud: a thin condensation shell ON the front, and a DOME
         rather than a curtain -- it stands taller as it grows and is packed
         toward its own foot. The front DECAYS (speed = C_FRONT * e^(-t/1.2),
         so radius asymptotes instead of scaling at a constant rate like a
         script), and each parcel is launched at the local front speed with a
         drag equal to that decay, so it rides the wave for the quarter second
         it lives instead of being left behind as a widening band. Bright,
         translucent, above the dust skirt -- the only part of a shockwave you
         can actually photograph. DensityRepr only.
  ember  A 0.12 s throw of ballistic sparks that arc, land and burn out.
         Additive billboards, one FIELD per burst so each population dims on
         its own clock. Not the analytic emitter, which cannot express a
         one-shot burst at all -- see the note at the field.
  flare  A small hot subset of the burst carrying the additive billboard flash
         with its own bloom pyramid. 15 M additive quads is not a look, it is
         a fill-rate accident; a quarter million of them is the sparkle.

THE VOLUMES FOLLOW THE MATTER. A DensityRepr contributes nothing outside its
box -- the scatter SKIPS a particle out there, it does not clamp it -- and all
four boxes used to be nailed down (three at the world origin, the fireball's at
whatever --charge said, which --charges never wrote back). An expanding ring
poking through the four flat walls of a 66 m square around the origin is
amputated there, and what is left is four arcs in the corners of an axis-aligned
square over a floor whose grid rules run the same way: "the shockwave is locked
to a grid around world center". `center` and `half_extent` are per-frame writes
by design, so all four now track their own matter -- the fireball the newest
live charge, the ring and the dust skirt the union of the live fronts, the
plume its own measured centroid. Moving a box changes the voxel SIZE, and the
splat is per-voxel OCCUPANCY, so sigma is scaled as 1/h^2 to hold the optical
depth the march integrates invariant while the box breathes. Get that law
wrong and the ring flashes opaque the moment the box is small.

And the plume's box GROWS, which is the counter-intuitive half: parcels per
voxel is the only thing the density splat measures, and it collapsed 117 -> 3
between t=5 and t=11 because the plume inflated elevenfold inside a box whose
voxels never changed. Three parcels a voxel is not a medium, it is visible
discrete scatter -- which is why the old column dissolved into a sparse point
cloud and then blinked out. It now grows with the plume, fades out on a
field-wide sigma ramp instead of being deleted, and shreds as it ages (the curl
stir grows with a parcel's own age and gains a finer octave). There is ambient
wind, too, so the head leans downwind and shears against the still stem instead
of standing as a perfectly axisymmetric fountain over its own crater.

THE YARD IS SOLID. A gas parcel used to collide with exactly one thing, a flat
plane at y = 0.10 m: smoke went through the house, the bunker, the chimney and
the pipe rack as if they were not there. A coarse signed-distance field is
baked ONCE from the settled yard (one 3D launch, ~750 M box tests, tens of ms)
and sampled per parcel as a single vec4 fetch -- unit outward gradient plus
signed distance -- exactly the way the curl grid already is. Smoke is pushed out
of anything it is inside, loses the INTO-surface part of its velocity but keeps
the tangent (so it climbs a wall and rolls over the top rather than stopping
dead), clings a little and curls in the shear layer. Gaps are free: at 0.35 m
the doorway, the gateway's span and the pipe rack are simply open. It is
re-baked once per charge, 1.6 s after it fires, so the smoke stops breaking
around a house that has already been thrown across the yard.

Three steel drums stand in the yard, and they are the one thing here that does
NOT come apart by construction. Each is a two-skin truss shell in raw Warp --
the hydraulic press's cylinder, plus the water balloon's tearing -- so the wave
dents it, the dents STAY (plasticity: a strut past yield creeps its rest length
toward the length it has), struts stretched past TEAR die, faces left without
three live edges leave the mesh, and the flap that is still hanging peels back
on the wind. The vertex kicks are staggered by the same front the bricks feel.
Both skins are published as fixed-capacity triangle soups straight into the
renderer's own vertex buffers (`enable_vertex_interop`), the outer one in bare
steel that takes the ray-traced reflection of the flash, the inner one dark, so
a tear shows the drum's inside.

Debris that lands hard kicks up the floor: once a frame the host reads a THIRD
of the bodies (round-robin, so every one is sampled at 20 Hz for a third of the
cost), calls a body that was falling fast and has just stopped a STRIKE, and
hands the hardest two dozen of them to a kernel that raises a small collar of
dust parcels there -- in the dust cohort's own ring, with the dust cohort's own
gravity and settling. It is the one thing here that is not born in the burst
window, and it earns that the same way the shock ring does: it is not new matter
appearing, it is the yard being kicked.

Emission, recycling and the fire->smoke conversion are all device kernels
structured around a fixed-max array of 8 charges. The CPU never walks a
particle: per frame it computes one integer per cohort per charge and hands the
ring cursor to a launch. The pools are SHARED between charges -- a ninth
overlapping burst overwrites the oldest slots, which is the recycling policy.

`--video S` is S seconds of FILM, not of yard. The playback RAMPS: the settle
runs at 1.9x because nothing is happening in it, the ramp drops to a sixth speed
as the fuse burns down, holds through the flash, the fireball opening, the
condensation ring leaving and the first wall going, then comes back out over
three quarters of a second of yard. The physics timestep never changes -- PhysX
keeps its fixed 1/60 against an accumulator and the shells keep a fixed substep
size, which is what makes the slow motion honest rather than a different
simulation (get that wrong and the shells come out six times as damped: the
first cut of this had three pristine drums).

    python warp_explosion.py                 # window; drag to orbit, Esc quits
    python warp_explosion.py --video 9       # the film -> warp_explosion.mp4
    python warp_explosion.py --shot 2.1      # headless: sim 2.1 s, write png
    python warp_explosion.py --bench         # timed phase breakdown, post-detonation
    python warp_explosion.py --slowmo 0.08   # slower slow motion
    python warp_explosion.py --no-ramp --video 8    # real time, as phases 1-3 shot it
    python warp_explosion.py --grade dusk    # flat | yard | range (default) | dusk
    python warp_explosion.py --drum-cam --dof --shot 2.24   # macro on the near drum
    python warp_explosion.py --yield 2400    # bigger charge (J0, kg m/s at 1 m)
    python warp_explosion.py --charge 0,0.6,-1.2 --t0 1.5
    python warp_explosion.py --charges "0,-0.6,1400@2.0; -14.5,-6,1700@2.6"
    python warp_explosion.py --auto-charge 2.5     # keep blowing things up
    python warp_explosion.py --front-speed 35      # a slower, more readable ripple
    python warp_explosion.py --courses 20 --shot 4      # taller walls, more bodies
    python warp_explosion.py --shot 4 --spin 0.25       # orbit while simulating
    python warp_explosion.py --gas 0.3                  # 4.4 M particles, not 15 M
    python warp_explosion.py --sigma 1.6                # thicker smoke
    python warp_explosion.py --wind 0,0,0               # still air (default 1.9 m/s)
    python warp_explosion.py --smoke-aniso 0.3          # forward-scatter the plume
    python warp_explosion.py --no-sdf                   # gas ignores the structures
    python warp_explosion.py --sdf-cell 0.25            # finer obstacle field (more VRAM)
    python warp_explosion.py --no-interop               # HostRing fallback, reduced counts
    python warp_explosion.py --tear 1.18 --drum-gain 1400   # shred the drums
    python warp_explosion.py --no-puff                  # no impact dust
    python warp_explosion.py --no-drums                 # no shells at all
    python warp_explosion.py --no-gas                   # phase 1 only
    python warp_explosion.py --no-ao                    # RT AO/GI off (~6.5 ms/frame back)
    python warp_explosion.py --msaa 1                   # the other render knob

Vulkan only: the fireball renders with tp.ParticleField, which draws nothing on
the GL path by design. The smoke rides the froxel FOG pass, so --no-fog is a
phase-1 debug flag -- it turns the gas volumes off with it.
"""
import atexit
import math
import os
import sys
import tempfile
import time

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import (bench_loop, cli_arg, encode_png_sequence, find_ffmpeg, parse_size,
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

# THE SLOW-MOTION RAMP (--video only; the window is always real time).
#
# The water balloon's trick: the physics timestep never changes, only how much
# SIM time one rendered frame covers. So `--video S` is S seconds of FILM, and
# the sim time it covers is whatever the ramp integrates to -- the settle runs
# fast because nothing is happening in it, the frame the front leaves the charge
# runs at a sixth speed because everything is, and the drifting column runs a
# little fast again on the way out. Everything downstream follows for free:
# PhysX accumulates and interpolates against its own fixed 1/60 step, the gas and
# the shells take dt as an argument, the streak emitter is seeded with absolute
# time, and the shake and the timeline are functions of sim time.
SLOWMO = cli_arg("--slowmo", 0.16, float)   # sim seconds per film second at the floor
NO_RAMP = "--no-ramp" in sys.argv           # --video in real time, as phases 1-3 shot it
SLOW_LEAD = 0.30          # sim s before t0 that the ramp starts (the fuse)
SLOW_HOLD = 0.45          # sim s held at the floor: the bloom and the first wall
SLOW_OUT = 0.75           # sim s the ramp back out takes
PLAY_PRE = 1.9            # before the ramp: the settle is not the film
PLAY_TAIL = 1.35          # after it: the column drifts, so let it drift faster

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
# ── THE RISING SMOKE IS OFF ─────────────────────────────────────────────────
# A blast this size makes its picture in the first second: fireball, shock
# dome, debris, embers, the dust the front tears off the ground. The rising
# soot column that followed never earned its place -- every version of it read
# as a grey mass hanging over the yard rather than as smoke, and the aftermath
# is stronger without it: the dome expands, the dust ring runs out, the debris
# lands, and the yard is left standing in settling dust. `--smoke` puts the
# column back for anyone who wants to keep working on it.
SMOKE_ON = "--smoke" in sys.argv
SMOKE_K = cli_arg("--smoke-k", 3, int) if SMOKE_ON else 0
                                            # soot parcels one cooled fire parcel becomes

# The capacity decision needs the device, because it is the interop path that
# makes fifteen million affordable: without it every particle crosses the bus
# TWICE a frame, so the fallback runs at phase 2's counts, which is what the
# HostRing path was measured to carry (1.79 ms of submit at 644 k live).
NO_DRUMS = "--no-drums" in sys.argv
device = None
INTEROP = False
if not (NO_GAS and NO_DRUMS):
    wp.init()
    device = wp.get_preferred_device()
    INTEROP = (not NO_GAS) and (not NO_INTEROP) and device.is_cuda
FALLBACK = 0.04
_SCALE = GAS * (1.0 if INTEROP else FALLBACK)
FIRE_N = int(3_000_000 * _SCALE)
DUST_N = int(2_000_000 * _SCALE)
# Embers do NOT scale with the rest, and that is a finding, not an oversight: an
# ember is a DISCRETE POINT on screen, so its count is a dot density, not an
# optical mass. Half a million of them, thrown outward and landing on the yard,
# read as a carpet of white static thirty metres across -- exactly what phase 2
# rejected dust billboards for. Three times phase 2's count is the ceiling.
# ... and a burst gets a WHOLE ember field of its own (EMBER_RING_N of them,
# round-robin, exactly as the streak fields work). One field is one intensity and
# one colour, and the only fade an Interop billboard field HAS is that field-wide
# scalar -- so two bursts sharing one pool means the newer one's flash re-lights
# the older one's dead ground sparks. Three small fields cost less than one large
# one did and make "one burst ages as one population" literally true.
#
# PHASE 5: 20 k per burst still read as a HALO OF ROUND DOTS, not as sparks --
# measured 13-20 k live simultaneously around the crater, and once the streak
# fields park (0.22 s) nothing left in the shot has stretch, because stretch is
# a Renderer-emitter property and these are Interop billboards. "Sparse" is the
# word in the complaint and count is the only lever that reaches it: 6 k spread
# over a 25 m throw is discrete embers, 20 k is dandruff.
EMBER_N = int(6_000 * _SCALE)
EMBER_RING_N = cli_arg("--ember-ring", 3, int)
FLARE_N = int(250_000 * _SCALE)
# The Wilson cloud: a thin condensation ring ON the front. It is emitted at the
# front radius AND thrown outward at the front speed, so each parcel rides the
# wave for its own short life instead of being left behind as a widening band.
WILSON_N = int(600_000 * _SCALE)
# Smoke is not emitted, it is CONVERTED, and the mapping is static: fire slot i
# becomes smoke slots [i*K, i*K+K). That is why there is no free list and no
# atomic anywhere in this sim -- the smoke pool inherits the fire ring's
# recycling for free.
SMOKE_N = FIRE_N * SMOKE_K if SMOKE_ON else 1024
                                            # stub when off: ParticleField.create
                                            # rejects capacity 0, and a parked
                                            # field costs one entry and no march

# Emission windows, in seconds since a charge's t0. NOTHING IS INSTANTIATED
# OUTSIDE THEM -- that is the model, not an optimisation. The longest window is
# the dust ring's, and it is the shock front crossing the yard, not a fountain.
FIRE_EMIT, FIRE_LIFE = 0.30, (0.30, 1.15)   # "life" here IS the cooling time
DUST_EMIT, DUST_LIFE = 0.90, (0.9, 2.2)
# EMBERS: THROWN, NOT POURED. Every window here is quoted in SIM seconds and the
# film plays the detonation at --slowmo (0.16x), so 0.40 s of emission is two and
# a half SCREEN seconds of new sparks leaving the charge and a 3.4 s life is six
# screen seconds of them hanging about. That -- not late instantiation -- is what
# "it keeps spewing embers" was: the windows were tuned in yard time and watched
# in film time. Divide the numbers by the ramp instead of arguing with it.
#
# PHASE 5: the LOW end of the life range is what sets when the population starts
# to fall, and 0.55 s meant the count sat pinned at capacity for a full screen
# second after emission stopped -- nothing could die before then. Widening it
# downward starts the decay on the frame the last spark is thrown.
EMBER_EMIT, EMBER_LIFE = 0.12, (0.16, 0.85)  # ejecta: one throw, then they land
# The whole cohort is gone by EMBER_EMIT + EMBER_LIFE[1], and that is also the
# time the field's own brightness takes to reach ZERO (it used to floor at 45%).
EMBER_FADE_T = EMBER_EMIT + EMBER_LIFE[1]
EMBER_LAND_T = 0.30       # s a spark keeps burning after it hits the ground
FLARE_EMIT, FLARE_LIFE = 0.22, (0.14, 0.46)
# From the moment it was fire. A parcel's death is INSTANT (the kernel writes
# the dead sentinel and returns) and the density splat ignores its radius, so a
# long tail does not fade -- it deletes the plume parcel by parcel while it is
# still holding its shape, and the last third of it blinks out covering a
# third of the sky. Shorter tail + a field-wide sigma ramp (SMOKE_FADE below)
# is the honest exit: the plume thins to nothing instead of vanishing.
SMOKE_LIFE = (3.0, 13.0)  # phase 5: WIDE, so parcels expire staggered over the
                          # whole aftermath. A narrow window kills one burst's
                          # soot as one population and the cloud pops out.
SMOKE_BPOW = 2.2          # skew of the per-parcel buoyancy draw: >1 puts most
                          # soot near zero lift (the base) and a few near BGAIN
                          # (the head). This is the shape of the column.
SMOKE_BGAIN = 2.6
# The ring is emitted for as long as the front is worth watching -- at 60 m/s
# that is 27 m, past the far props -- and each parcel lives a quarter second.
WILSON_EMIT, WILSON_LIFE = 0.45, (0.20, 0.42)
# THE FRONT DECELERATES, and that is what stops it reading as a circle being
# scaled by a script. A blast wave decays toward the sound speed; r = c*t is a
# machine. One time constant is the whole model: speed = C_FRONT * e^(-b/TD),
# so radius = C_FRONT * TD * (1 - e^(-b/TD)) and it asymptotes at 51 m instead
# of running off across the yard for ever. Each ring parcel is LAUNCHED at the
# local front speed and given drag exactly 1/TD -- the same deceleration the
# front itself has -- so it rides the front for its whole life with no
# per-parcel bookkeeping at all. Reach drops 52 m -> 37 m, which is also what
# makes the follow box below affordable.
#
# TD is deliberately gentle rather than Sedov-steep, because the RIGID arrival
# schedule (t_hit = t0 + r/C_FRONT, built once per charge in Charge.arm) is
# still linear: the visible ring must not lag the wall it is supposed to be
# knocking over. At 1.2 s the ring reaches 20 m 0.06 s behind the impulse and
# has halved its speed by the time it dies, which reads as a decaying wave
# without desynchronising the demolition.
WILSON_TD = 1.2
# ... and it is a DOME, not a two-metre curtain. The shell stands taller as it
# grows and is densest at its foot, which is where a real Mach stem is.
# ... except that it was not one yet: measured median height 2.3 m at r = 18 m
# and 3.1 m at r = 34 m, which is 1:9 and FLATTER as it expands, because the
# band saturated at 6.2 m while the radius ran on to 37. A hemisphere wants its
# height to grow WITH its radius, so the band is now a fraction of the front
# radius with a floor, not an asymptote -- 0.30 r is a squat dome, which is what
# a surface burst actually throws, and the p1.7 packing keeps it brightest at
# the foot where the Mach stem is.
WILSON_H0, WILSON_HG = 2.2, 0.30     # spawn band: floor in m, then x front radius
WILSON_HP = 1.7                      # >1 packs the parcels toward the ground
EMBER_SKEW, FLARE_SKEW = 2.6, 1.4           # emitted = N * (tau/EMIT)**(1/skew)

# THE BLAST REACHES THE GAS THAT IS ALREADY IN THE AIR. Until now a charge only
# touched a gas parcel at BIRTH (emit_gas placed it); every parcel already alive
# was deaf to it, so a second charge beside a standing plume did nothing to that
# plume. step_gas now walks the same 8-slot device charge array the drums do and
# gives every live parcel the same two-part push the rigids get: one IMPULSE as
# the front sweeps past it (the half-open shell [C*b0, C*b1) tiles the radius
# axis as the frames tile time, so it lands exactly once at ANY dt), then a
# decaying WIND behind the front, keyed on that parcel's own arrival time --
# which is read off the geometry, r/C_FRONT, rather than stored per particle.
GAS_BLAST = cli_arg("--gas-blast", 1.0, float)   # scale on the whole coupling
# NOT the rigids' exp(-r/L)/r^1.5: on a massless parcel that is 0.004 at 10 m and
# invisible. A pressure wave wants a long reach -- L^2/(L^2+r^2) is 0.80 at 3 m,
# 0.27 at 10 m, 0.08 at 20 m.
GAS_L = 11.0              # m, the falloff length
# The two halves do NOT touch the same matter, which is what makes them
# separately tunable. A charge's OWN cohorts are never in its own shell -- the
# fireball is inside GAS_RMIN, the dust skirt rides at DUST_C (28 m/s) and the
# ring at C_FRONT with the velocity gate on it, so all of them are behind R0 and
# only ever feel the TAIL. The IMPULSE is therefore a cross-charge term, and it
# is the one that has to be big enough to shove a standing plume: at 18 m it is
# ~5.4 m/s, which against SMOKE_DRAG relaxes into about four metres of travel.
GAS_KICK = 20.0           # m/s of delta-v at r << L, x the charge's own scale
GAS_VMAX = 18.0           # m/s cap on one arrival -- the anti-NaN valve
GAS_RMIN = 4.0            # m: inside the fireball there is no thin shell
GAS_LIFT = 0.55           # the ground reflection lifts what is low, as it does bricks
GAS_WIND_A0 = 20.0        # m/s^2 of tail at r << L (this one DOES hit its own smoke)
GAS_STIR = 5.0            # m/s^2 of EXTRA curl behind the front: pushed smoke TUMBLES
GAS_BLAST_T = 1.6         # s a charge's front is worth testing against (45 m + tail)

# IMPACT DUST. Debris that lands hard kicks up the floor, and it is the cheapest
# "alive" trick in the demo: without it the yard goes quiet the moment the ring
# has passed, and a brick arcing forty metres lands in silence. The host reads
# the bodies ONCE a frame -- a third of them per frame, round-robin, which is
# what keeps a 1700-body Python loop off the frame time -- and a strike is a body
# that was falling fast and has just stopped. See impact_dust().
NO_PUFF = "--no-puff" in sys.argv
IMPACT_V = 5.5            # m/s of fall that counts as a strike
IMPACT_STOP = 0.45        # ... and has to have lost this much of it in one scan
IMPACT_Y = 1.8            # ... near enough the ground to raise anything, m
IMPACT_STRIDE = 3         # bodies scanned per frame: len(bodies) // this
IMPACT_T = 4.5            # s after a charge that debris is still coming down
PUFF_MAX = 24             # strikes taken per frame, the hardest first
PUFF_V0 = 3.2             # m/s at a reference strike
PUFF_LIFE = (0.55, 1.60)
# A puff has to be a LOCAL concentration to read, and the density splat is
# per-voxel occupancy with the particle radius ignored (phase 2's finding), so
# the only two knobs are how many parcels and how tightly they are packed. The
# dust volume's voxel is 0.56 x 0.04 x 0.56 m: 220 parcels in a metre-wide collar
# came out at under one parcel a voxel and was invisible in an A/B. 1400 in a
# half-metre collar is about six, and reads.
PUFF_P = max(int(1400 * _SCALE), 2)   # dust parcels one strike raises

FIRE_V0, FIRE_DRAG, FIRE_BUOY, FIRE_CURL = 21.0, 7.5, 26.0, 7.0
# The plume: tighter than it was, and it BREAKS UP as it ages instead of
# ballooning uniformly. Terminal stir is CURL/DRAG -- 2.6 m/s before, 1.4 now --
# because a divergence-free field stretches a blob into filaments without
# entraining anything, so all the extra stir bought was a wider, thinner,
# grainier column. What replaces it is SMOKE_CURL_AGE: the stir GROWS with a
# parcel's own age and gains a finer second octave, so young soot stays a
# coherent thick mass and old soot shreds. Occupancy (parcels per voxel) is the
# only thing the density splat measures, and it is what "thick" means here.
SMOKE_DRAG, SMOKE_BUOY, SMOKE_CURL = 1.45, 5.2, 3.0
SMOKE_CURL_AGE = 1.3      # extra stir at end of life, x SMOKE_CURL
SMOKE_ENT = 2.6           # m/s^2 of entrainment: the column WIDENS as it climbs
SMOKE_ENT_K = 0.34        # ... to a radius this multiple of its height above the crater
SMOKE_INHERIT = 0.75      # of the fire parcel's velocity: this is the outward push
SMOKE_SPREAD = 2.10       # m/s of isotropic scatter added to each soot child
DUST_V0, DUST_DRAG, DUST_GRAV, DUST_CURL = 7.5, 2.2, 9.81, 0.7
EMBER_V0, EMBER_DRAG, EMBER_CURL = 18.0, 0.90, 0.9
FLARE_V0, FLARE_DRAG, FLARE_BUOY, FLARE_CURL = 19.0, 6.5, 20.0, 6.0
# The drag IS the front's deceleration (1/WILSON_TD): a parcel launched at the
# local front speed and relaxed at exactly that rate stays on the front for its
# whole life, which is what keeps the ring a ring instead of a widening band.
WILSON_DRAG, WILSON_BUOY, WILSON_CURL = 1.0 / WILSON_TD, 1.2, 0.9
FIRE_R = (0.15, 0.80)     # world radius over life, m (w under WSemantic.Radius)
SMOKE_R = (0.30, 1.20)
DUST_R = (0.14, 0.70)
# An ember quad NEVER goes sub-pixel: the fragment falloff peaks at 1.0 at the
# sprite centre whatever the world radius, so shrinking one does not dim it, it
# only makes a same-brightness dot smaller until it aliases into twinkle outside
# TAA. It shrinks to a floor (50 mm = ~2 px at the shipped 40 m framing) and the
# DIMMING is done by the field scalar and by the population dying.
EMBER_R = (0.115, 0.050)
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
# 72, not 128, AND THAT IS THE ANTI-SPECKLE KNOB. Once the medium is thick
# enough to shadow itself, voxel-to-voxel OCCUPANCY noise stops being a rounding
# error and becomes shadow noise -- one extra parcel in a voxel now casts a
# visibly darker hole -- which is why the shipped build read as salt and pepper
# no matter what sigma did. Bigger voxels hold more parcels, and the relative
# noise falls as 1/sqrt of that. Measured at t=7.5 s, same shot, sigma held
# invariant: res 128 -> 96 -> 72 -> 56 moved the plume's high-frequency energy
# 15.8 -> 13.0 -> 9.4 -> 5.6 levels and its blown-out fraction 4.2% -> 0.01%.
# (The same bracket on the OLD thin build moved grain by 7% in total, which is
# why phase 4 concluded occupancy was not the lever. It is -- but only once the
# volume is thick.) 72 keeps a 0.67 m voxel over a 48 m box, which the linear
# filter at both ends of the splat carries without a visible lattice.
SMOKE_RES = cli_arg("--smoke-res", 96, int)
SMOKE_RES_REF = 72        # what SMOKE_SIGMA below is normalised at
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
FIRE_SIGMA, SMOKE_SIGMA, DUST_SIGMA = 0.22, 1.07, 0.040
WILSON_SIGMA = 0.048      # phase 5: the dome spreads 600 k over 2x the height
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
#
# ... and that last paragraph is now HISTORY. Renderer commit 02229a23 put the
# density volumes' own extinction into the sun leg of the march, so a soot
# column self-shadows: its far side is genuinely dark (measured p05 105 against
# a 167 sky). What did NOT change is the AMBIENT leg, which carries no
# transmittance at all -- so a saturated medium converges to
# albedo * (ambient + sun * T_medium), which is BRIGHTER than a thin one.
# Measured on this build at t=8: sigma x3 moved the plume mean 141 -> 157 and
# p95 199 -> 253. Optical depth buys brightness here, not thickness. Thickness
# is bought with OCCUPANCY (parcels per voxel, the only thing the splat sees)
# and spent through the shadowed side, which is why the follow box below
# matters more than either of these two numbers.
#
# PHASE 5 CORRECTS THAT PARAGRAPH. It was measured on a build whose smoke was
# optically thin everywhere -- transmittance through the plume head was 1.00 and
# 0.36 at its very base -- and a thin medium IS just albedo * ambient, which is
# why raising albedo there only brightened it. What was missing is the other
# half of "thick": at sigma 0.24 the volume never shadowed ITSELF, so with
# 02229a23 landed the sun reached every parcel and the column rendered as a
# uniformly lit white spray, brighter than the sky it sat on (measured p50 140
# against a sky of 166 -- smoke that is only 15% darker than the sky is a veil).
#
# The fix is BOTH knobs, in opposite directions: eight times the optical depth
# so the head actually blocks the sky and the underside falls into its own
# shadow, and a THIRD of the albedo so the lit rim does not blow out. That is
# also what kills the speckle -- the per-pixel march dither is a fraction of the
# in-scatter, so once the medium saturates the noise saturates with it (measured
# lag-1 autocorrelation 0.59 -> 0.90) -- and it is why the resolution bracket
# never worked: 2.5x the occupancy moved the grain by 7%, this moves it by 50%.
SMOKE_SIGMA_K = cli_arg("--smoke-sigma", 0.8, float)   # bracketing knob, x SMOKE_SIGMA
SMOKE_ALBEDO = tuple(float(v) for v in
                     cli_arg("--smoke-albedo", "0.34,0.31,0.28", str).split(","))
SMOKE_ANISO = cli_arg("--smoke-aniso", 0.20, float)

# --- ambient wind -----------------------------------------------------------
#
# There was no wind anywhere in the gas, and it shows: once buoyancy is spent
# the only motion left is a divergence-free curl field, so the column inflates
# in place, stays axisymmetric about its own crater and never shears. Three
# metres a second is enough to lean the head downwind, shear it against the
# still stem and walk the mass off the crater -- and the height ramp keeps it
# out of the ground dust, which must stay where it settled.
AMB_WIND = tuple(float(v) for v in cli_arg("--wind", "1.9,0,0.6", str).split(","))
AMB_Y0, AMB_Y1 = 2.0, 8.0     # m: no wind below Y0, full above Y1

# --- the density boxes FOLLOW ----------------------------------------------
#
# Every one of the four volumes used to be nailed to the world origin (or, for
# the fireball, to whatever `--charge` said, which `--charges` never writes
# back). A DensityRepr contributes NOTHING outside its box -- the scatter skips
# the particle, it does not clamp it -- so an expanding ring poking through the
# four flat walls of a 66 m square around the origin is amputated there, and
# what is left is four arcs sitting in the corners of an axis-aligned square
# over a floor whose grid rules run the same way. That is the whole of "the
# shockwave is locked to a grid around world center".
#
# `center` and `half_extent` are read/write per frame by design ("the volume is
# re-scattered from scratch every frame, so moving the box is free" --
# ParticleField.hpp); only `resolution` is latched. So the fix costs no slot,
# no field, no allocation: move the boxes.
#
# THE SIGMA LAW IS THE PRICE. The splat is per-voxel OCCUPANCY, so
# particles-per-voxel scales with the voxel VOLUME and the extinction the march
# integrates over a metre goes as occupancy * sigma / voxel_size, i.e. as
# sigma * h^2 at fixed resolution. Holding the look invariant while the box
# breathes therefore needs sigma proportional to 1/h^2, normalised at the
# half-extent each of these was tuned at. Get that wrong and the ring flashes
# opaque the moment the box is small.
WILSON_H_REF, WILSON_H_MIN = 33.0, 15.0
WILSON_HY, WILSON_CY = 9.0, 6.0       # the dome is taller than the old curtain
# DUST_H_MAX was 40 m, and two charges 12 m apart in x already want 43 -- the
# union box would have been clamped and the skirt amputated by a flat wall, i.e.
# exactly the artefact the whole box-follow pass exists to remove. The clamp
# only ever existed to bound the march's step budget, and this slab is 5.2 m
# thick, so it can afford to be generous.
DUST_H_REF, DUST_H_MIN, DUST_H_MAX = 36.0, 26.0, 60.0
DUST_REACH = DUST_C * DUST_EMIT + 9.0
# The smoke box TRACKS THE PLUME and GROWS WITH IT, which is the opposite of
# the intuition and the only thing that fixes the sparse-dots ending. Occupancy
# fell 117 -> 3 parcels per voxel between t=5 and t=11 because the plume
# inflated 11x inside a box whose voxels never changed; a box that grows with
# it holds the voxel count over the plume roughly constant, and the sigma law
# above keeps the optical depth honest while it does.
# The floor is TODAY'S half-extent, not something smaller: a box that shrinks to
# fit the plume divides the parcels per voxel by the cube of the shrink, and
# occupancy is the entire signal. The box may only ever GROW from here.
SMOKE_H_REF, SMOKE_H_MIN, SMOKE_H_MAX = 24.0, 26.0, 36.0
SMOKE_BOX_K = 2.35        # half-extent as a multiple of the plume's p90 spread
SMOKE_BOX_LAG = 0.30      # how fast the box chases the plume (per sample)
SMOKE_BOX_EVERY = 8       # frames between plume samples (a 64 kB strided read)
# ... and it FADES rather than being deleted. The ramp finishes before GAS_END
# so the plume thins to nothing instead of the park predicate popping it out.
# PHASE 5: it starts when the population starts DYING (SMOKE_LIFE[0] plus the
# fastest fire parcel's cooling time, i.e. 4.3 s) rather than a second later,
# and it ends exactly at GAS_END so the ramp finishes on the frame the park
# predicate fires instead of leaving 0.45 s of invisible-but-still-advected gas.
SMOKE_FADE = (2.6, 8.0)   # phase 5: the aftermath dissolves instead of hanging as a slab

# --- the gas meets the yard -------------------------------------------------
#
# Until now a gas parcel collided with exactly one thing: a flat plane at
# y = 0.10 m. It went through the house, the bunker, the chimney, the arch and
# the pipe rack as if they were not there -- a fireball lit inside the sealed
# bunker poured its smoke straight up through the roof slabs.
#
# The fix is a coarse signed-distance grid baked ONCE from the settled yard and
# sampled per parcel exactly the way the curl grid already is: one vec4 fetch
# (unit outward gradient + signed distance), no host loop, no per-parcel state,
# no broad phase. 0.35 m resolves the house doorway, the gateway's span and the
# pipe rack's gaps, so smoke pours through them with no authoring at all.
#
# STATIC, from the pre-blast layout, with ONE lazy re-bake per charge once its
# front has finished rearranging the yard. Tracking 1720 live poses every frame
# would mean the O(bodies) host readback in the sim path that every phase of
# this demo has kept out of it; and the static field is right exactly where it
# matters, because the front crosses the yard in a third of a second and the
# plume is down among the structures for the first second or two, while the
# walls are still standing or only just leaving.
NO_SDF = "--no-sdf" in sys.argv
SDF_CELL = cli_arg("--sdf-cell", 0.35, float)
SDF_YTOP = 12.0           # m: the grid's ceiling. Above it there is nothing to hit
#
# PHASE 5 MADE IT VISIBLE. The first cut of this was measurably real and
# optically invisible: the whole collision system displaced the average smoke
# parcel by 2.5 cm, against a parcel RADIUS of 0.30-1.20 m and a 0.35 m voxel,
# so --no-sdf and the SDF rendered the same picture in every scene that could be
# built, including a charge lit inside a standing three-walled house. Three
# things were wrong. The skin was 0.45 m -- a THIRD of one parcel -- so a plume
# only felt a wall it was already touching; there was no repulsion at all, only
# a push-out of the penetration that had already happened; and fire was exempt,
# while every soot parcel in the demo is born at a dying fire parcel's position,
# so smoke was INSTANTIATED inside the roof it should have been stopped by.
SDF_SKIN = 1.8            # m: the band a parcel feels the surface through
SDF_REST = 0.0            # pure slide. Anything above zero reads as rubber
SDF_REPEL = 26.0          # m/s^2 of standoff INSIDE the band: a real deflection
SDF_TANG = 2.6            # 1/s of tangential drag at the surface: smoke CLINGS
SDF_STIR = 6.0            # m/s^2 of shear turbulence at the surface
SDF_PUSH = 0.45           # m/frame ceiling on the push-out, so nothing teleports
SDF_FIRE_W = 0.55         # the fireball feels it too, at a little over half
SDF_REBAKE = (0.35, 2.4)  # s after a charge that the yard is re-measured

# --- the drums --------------------------------------------------------------
#
# Three vertical steel drums, each a TWO-SKIN TRUSS SHELL: an outer skin, an
# inner skin 13 mm behind it, and a truss between them -- edges in each skin, a
# radial strut per vertex pair, two shear diagonals across every edge's prism.
# That is the hydraulic press's shell wrapped onto a cylinder, and it is the
# whole reason a drum dents instead of deflating: bending costs stretch in the
# skins, so an inverted dent is a real crease and not a free sign flip.
#
# What makes it STEEL rather than rubber is plasticity (the press again): a
# strut strained past its yield creeps its rest length toward the length it
# actually has, so the crease the wave puts in is still there a second later.
# What makes it TEAR is the water balloon: a skin strut stretched past TEAR
# times its ORIGINAL rest length dies, and a face whose edges are not all alive
# leaves the mesh -- so a run of broken struts opens a slit, the slit runs, and
# the flap left hanging peels back on the wind behind the front.
#
# The blast coupling is per VERTEX and staggered by the front exactly like the
# rigid side (t_hit = t0 + r / C_FRONT), with a facing weight: a blast is a
# pressure on a surface, so the skin whose normal looks INTO the wave takes the
# push and the far skin is shadowed. That difference IS the cave-in. The whole
# thing lives on the device -- the host never touches a drum vertex.
DRUM_R, DRUM_H = 0.295, 0.88        # a 200 litre drum, near enough
DRUM_G = cli_arg("--drum-grid", 9, int)     # cap grid side; the body has 4*(g-1) segments
DRUM_NV = cli_arg("--drum-rings", 14, int)  # rings up the side
DRUM_THICK = cli_arg("--drum-thick", 0.013, float)   # spacing of the two skins, m
DRUM_SUBSTEPS = cli_arg("--drum-substeps", 48, int)
DRUM_SUB = DT / DRUM_SUBSTEPS       # the substep SIZE, fixed forever (see step_drums)
DRUM_DAMP = 0.002                   # per substep
DRUM_MU = 0.55                      # friction against the ground: this is what topples it
DRUM_GRAV = wp.vec3(0.0, -9.81, 0.0) if device is not None else None
# Strut classes, per the press: the skins are nearly inextensible and rarely
# yield, the shear diagonals are soft and yield early -- so a wrinkle is cheap
# to form and, once formed, the diagonals remember it.
#
# THE YIELDS ARE TINY AND THAT IS THE POINT. The press's 6% skin yield is a
# number a distance projection can never reach: at full stiffness every strut
# is pulled back inside a percent of its rest length on every sweep, so a 6%
# gate is never opened, nothing creeps, and the drum DENTS ELASTICALLY AND
# SPRINGS BACK -- which is exactly what the first version did, at every blast
# gain I tried. Thin steel yields at a couple of tenths of a percent anyway;
# what the press's number really encodes is a shell squeezed slowly by a plate,
# not one hit by a wave. These are the real thing.
STIFF_SKIN, STIFF_DIAG, STIFF_RAD = 1.0, 0.28, 1.0
YIELD_SKIN = cli_arg("--yskin", 0.0020, float)
YIELD_DIAG = cli_arg("--ydiag", 0.0008, float)
YIELD_RAD = 0.010
PLASTIC_RATE = cli_arg("--plastic", 0.05, float)     # excess adopted per substep
# THE TEAR IS DUCTILE, NOT ELASTIC. The first spelling of this measured the
# instantaneous stretch of a skin strut against its original rest length, and
# NOTHING EVER TORE, at any gain: a full-stiffness distance projection holds
# every strut within a percent or two of its rest length, and a crushed
# cylinder buckles rather than stretches. What actually breaks a drum is the
# plastic strain it has ACCUMULATED -- so every metre of rest-length creep, in
# either direction, is damage, and a skin strut dies when it has taken TEAR of
# its own original length in creep. That is a real ductile-failure model and it
# puts the tears where the creases are, which is where a drum actually splits.
TEAR = cli_arg("--tear", 0.090, float)
TEAR_SNAP = 1.45          # ... and a strut yanked this far past rest0 in one go
# The shell's own blast gain. It is NOT the rigid J0: an impulse per unit MASS
# is what a rigid body takes, while a pressure wave on a thin skin delivers
# P*A/(rho*A*t) -- a velocity that does not depend on how finely the shell is
# tessellated. So the drums get their own number, in m/s at 1 m, tuned by eye.
DRUM_GAIN = cli_arg("--drum-gain", 1500.0, float)
DRUM_VMAX = 26.0                    # speed cap on the kick, m/s: the anti-NaN valve
DRUM_FACE_MIN = 0.22                # what the SHADOWED side still feels, 0..1
DRUM_WIND = cli_arg("--drum-wind", 2.2, float)   # wind-tail gain over the rigid side
DRUM_SETTLE = 0.35                  # seconds of solve before the shells are frozen
DRUM_RUN_T = 8.0                    # seconds a charge keeps them running afterwards
# (x, z, yaw degrees). Two in the open ground outside the house doorway, where
# the camera sees them against the fireball, and one out by the chimney so a
# second charge has a drum of its own -- which is also the stagger test: the
# near pair caves at t0 + 0.11 s and this one does not move until 0.21 s.
DRUM_SITES = [(2.6, 5.4, 0.0), (4.3, 6.6, 40.0), (-12.2, -4.6, 15.0)]
DRUM_CAM = "--drum-cam" in sys.argv  # frame the near drum instead of the yard

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

# THE GRADE. Phases 1-3 shot a light-grey yard under a near-white sky at
# exposure 0.55 and the frames came out light-grey-on-white: a soot column whose
# darkest value was barely under the void behind it, and a floor that was the
# brightest thing in frame. The fix is not one knob, it is the five that set
# where black and white ARE -- and they have to move together, because the sun
# is what lights the smoke (the density march reads the scene's sun and ambient)
# and the floor tone is what the plume is read AGAINST.
#
#   void   background AND fog colour: the sky the column is silhouetted on
#   floor  the grid texture's base tone, 0-255 sRGB
#   exp    tone_mapping_exposure, pinned (auto-exposure is off: the billboards
#          bypass it, so BB_* below is tied to this number by hand)
#   hemi   the fill. Weak on purpose -- fill is what flattens a blast
#   sun    the one hard light. Contrast is (sun - hemi), not exposure
#   fog    linear near/far, m. Far too near and the yard dissolves with the void
# Bracketed by eye on one frame (t = 4.4, the column up and the yard scattered),
# measured as mean sky / floor / plume tone in the same three windows:
#   flat  251 / 175 / 226   phases 1-3: everything in the top third, plume
#                           DARKER than the sky it is drawn on. The complaint.
#   yard  187 / 139 / 208
#   range 160 / 126 / 198   SHIPPED
#   dusk  150 /  97 / 184   the floor stops being a light-grey test range
# `range` is the one that keeps the Omniverse floor and still puts the plume at
# the top of the range: sky a real grey, floor a stop under it, and the column,
# the fireball and the embers the only things above both.
GRADE = cli_arg("--grade", "range", str)
GRADES = {
    # phase 3's look, kept so its acceptance frames reproduce exactly
    "flat": dict(void=(0.800, 0.810, 0.830), floor=163, exp=0.55, hemi=0.45,
                 sun=3.0, fog=(55.0, 200.0)),
    "yard": dict(void=(0.400, 0.428, 0.482), floor=150, exp=0.50, hemi=0.26,
                 sun=3.9, fog=(80.0, 320.0)),
    "range": dict(void=(0.300, 0.325, 0.385), floor=152, exp=0.52, hemi=0.24,
                  sun=3.2, fog=(85.0, 340.0)),
    # the far end of the bracket -- overcast dusk, almost a lightbox
    "dusk": dict(void=(0.245, 0.268, 0.320), floor=118, exp=0.46, hemi=0.20,
                 sun=4.4, fog=(60.0, 260.0)),
}
if GRADE not in GRADES:
    sys.exit(f"--grade: want one of {', '.join(GRADES)}")
G = GRADES[GRADE]

canvas = tp.Canvas("threepp x physx - blast yard", width=W, height=H,
                   antialiasing=MSAA, vsync=False, headless=HEADLESS)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = G["exp"]  # pinned: the billboards bypass auto-exposure
renderer.shadow_map_enabled = True
# Thin-lens bokeh, off by default (it is not free and a yard-wide establishing
# shot does not want it). The macro drum shot does: --dof --drum-cam.
DOF = "--dof" in sys.argv
FSTOP = cli_arg("--fstop", 2.4, float)
if DOF:
    renderer.set_camera_exposure(FSTOP, 1.0 / 125.0, 100.0)   # f-number: the CoC
    renderer.depth_of_field = True
# The two knobs that actually move the frame time here (measured, --bench):
# ray-traced AO/GI ~6.5 ms and the fog march ~2.5 ms at 1280x720. The body
# count is not the lever -- PhysX is under 2 ms of a 22 ms frame.
NO_AO = "--no-ao" in sys.argv
NO_FOG = "--no-fog" in sys.argv
if NO_AO:
    renderer.deferred_ao = False

VOID = tp.Color(*G["void"])
scene = tp.Scene()
scene.background = VOID
if not NO_FOG:
    scene.set_fog(VOID, *G["fog"])            # dissolves the slab's far edge into the void

camera = tp.PerspectiveCamera(46, canvas.aspect(), 0.05, 800)
# Framed for the PLUME, not the yard: at nebula counts the column is the
# subject and it climbs past 25 m, so the yard only occupies the bottom third.
CAM_R, CAM_Y, CAM_TARGET = 38.0, 13.0, (0.0, 9.0, 0.0)
CAM_O = (0.0, 0.0)                            # what the orbit is centred on, (x, z)
_ANGLE0 = 22.0
if DRUM_CAM:
    # The macro shot: a close orbit on the near drum, from the CHARGE's side of
    # it, because the face that caves in is the face that saw the wave.
    _dx, _dz = DRUM_SITES[0][0], DRUM_SITES[0][1]
    CAM_O = (_dx, _dz)
    CAM_R, CAM_Y, CAM_TARGET = 2.7, 1.05, (_dx, 0.46, _dz)
    _ANGLE0 = math.degrees(math.atan2(-_dx, -_dz)) + 34.0
SPIN = cli_arg("--spin", 0.0, float)          # shot/video: orbit the camera, deg/frame
START_ANGLE = cli_arg("--angle", _ANGLE0, float)  # orbit start angle, degrees
_cam_base = tp.Vector3()


_focus = [CAM_O[0], CAM_TARGET[1], CAM_O[1]]     # what --drum-cam tracks


def orbit(frame):
    """Place the camera on its orbit for `frame` (static at SPIN == 0).

    In --drum-cam the orbit CENTRE follows the near drum, because a drum next
    to a charge that size does not stay where it was put -- it is thrown, and a
    macro shot that does not follow it is a shot of an empty patch of grid."""
    a = math.radians(START_ANGLE + SPIN * frame)
    ox, oz = (_focus[0], _focus[2]) if DRUM_CAM else CAM_O
    _cam_base.set(ox + CAM_R * math.sin(a), CAM_Y + (_focus[1] - CAM_TARGET[1]),
                  oz + CAM_R * math.cos(a))
    camera.position.set(_cam_base.x, _cam_base.y, _cam_base.z)
    tgt = _focus if DRUM_CAM else CAM_TARGET
    camera.look_at(tgt[0], tgt[1], tgt[2])
    if DOF:
        # The lens focuses on what the camera is pointed at. On the macro shot
        # that is the drum, which is MOVING, so the focus has to be recomputed
        # rather than dialled in once -- it rides the same lagging dolly.
        renderer.focus_distance = max(math.dist((_cam_base.x, _cam_base.y, _cam_base.z),
                                                tuple(tgt)), 0.2)


# A test range is an overcast-bright void with one hard sun in it. The fill is
# deliberately weak: everything here is grey or clay, and a strong hemisphere
# fill flattens both into the same pale pink.
scene.add(tp.HemisphereLight(0xd7e3f4, 0x8d8a84, G["hemi"]))
sun = tp.DirectionalLight(0xfff2df, G["sun"])
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


def grid_texture(px=1024, tile=10.0, minor=1.0, base=163):
    """The test-range floor: a light grey with 1 m minor and 10 m major rules.

    Written to a temp PNG because TextureLoader takes a path; `tile` is the
    metres the image covers, which is what sets the repeat below. `base` is the
    grade's floor tone and the RULES FOLLOW IT -- the grid has to stay a fixed
    fraction under the floor or a darker grade turns it into a black lattice.
    The cache path carries the tone, or the first grade rendered wins forever.
    """
    from PIL import Image

    def rules(period, width):
        i = np.arange(px)
        return ((i + width // 2) % period) < width

    img = np.full((px, px, 3), base, np.uint8)
    fine = rules(max(int(round(px * minor / tile)), 2), 2)
    coarse = rules(px, 6)
    for mask, tone in ((fine, int(base * 0.85)), (coarse, int(base * 0.61))):
        img[mask, :, :] = tone
        img[:, mask, :] = tone
    path = os.path.join(tempfile.gettempdir(),
                        f"threepp_blast_grid_{px}_{int(tile)}_{base}.png")
    Image.fromarray(img).save(path)
    return path, tile


grid_path, grid_tile = grid_texture(base=G["floor"])
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
# Every body's AABB half-extent, in the SAME global order as `bodies`, which is
# what lets the SDF be re-baked later from body_positions() alone. A yawed box
# is inflated to its axis-aligned hull (two lines, exact) and a pipe is the
# capsule's own hull, tipped onto Z the way the instance transform tips it.
_hs = []
for kind, items in batches.items():
    if not items:
        continue
    for _p, _s, _yaw in items:
        if kind == "pipe":
            _hs.append((PIPE_R, PIPE_R, 0.5 * PIPE_L + PIPE_R))
        elif _yaw:
            _c, _sn = abs(math.cos(_yaw)), abs(math.sin(_yaw))
            _hs.append((0.5 * (_s[0] * _c + _s[2] * _sn), 0.5 * _s[1],
                        0.5 * (_s[0] * _sn + _s[2] * _c)))
        else:
            _hs.append((0.5 * _s[0], 0.5 * _s[1], 0.5 * _s[2]))
HALF = np.array(_hs, np.float32)
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

# The SDF grid's geometry, sized to the yard it has to describe. These are
# module-level Python numbers ON PURPOSE: Warp folds them into the kernels
# below as compile-time constants, which is how the sampler stays a single vec4
# fetch instead of eight more kernel arguments threaded through step_gas.
_lo = (HOME - HALF).min(0) - 1.2
_hi = (HOME + HALF).max(0) + 1.2
SDF_OX, SDF_OZ = float(_lo[0]), float(_lo[2])
SDF_OY = -0.6
SDF_INV = 1.0 / SDF_CELL
SDF_NX = int(math.ceil((float(_hi[0]) - SDF_OX) / SDF_CELL)) + 1
SDF_NY = int(math.ceil((min(float(_hi[1]), SDF_YTOP) - SDF_OY) / SDF_CELL)) + 1
SDF_NZ = int(math.ceil((float(_hi[2]) - SDF_OZ) / SDF_CELL)) + 1
SDF_TOP = SDF_OY + (SDF_NY - 1) * SDF_CELL
SDF_BAND = 3.0            # m: only the band near a surface is ever read
SDF_ON = not (NO_SDF or NO_GAS)

# --- the drums: mesh, truss, kernels ----------------------------------------


def drum_mesh(g, nv, R, H):
    """A closed cylinder whose CAPS ARE QUAD GRIDS, not triangle fans.

    The obvious cap -- a fan of nu triangles around one centre vertex -- gives
    that vertex nu incident struts, and the Gauss-Seidel colouring needs at
    least that many colours because no two constraints in a colour may share a
    vertex. One colour is one launch and there are DRUM_SUBSTEPS of them per
    frame, so a 24-fan cap costs about fifty launches a substep where the rest
    of the shell costs sixteen. A cap built as a g x g quad grid keeps every
    degree at six.

    The grid is mapped to the disc with the elliptical square-to-circle map,
    which sends the square's BOUNDARY exactly onto the unit circle -- so with
    nu = 4*(g-1) the cap's rim vertices ARE the body's top ring, shared, no
    stitching. The ring is then spaced by the map rather than uniformly in
    angle; the error is under three degrees and invisible.
    """
    nu = 4 * (g - 1)

    def disc(s, t):
        return (s * math.sqrt(max(1.0 - 0.5 * t * t, 0.0)),
                t * math.sqrt(max(1.0 - 0.5 * s * s, 0.0)))

    # the square's perimeter, counter-clockwise, starting at the (-1, -1) corner
    perim = ([(i, 0) for i in range(g - 1)] + [(g - 1, j) for j in range(g - 1)]
             + [(i, g - 1) for i in range(g - 1, 0, -1)]
             + [(0, j) for j in range(g - 1, 0, -1)])
    ax = [-1.0 + 2.0 * i / (g - 1) for i in range(g)]
    ring = [disc(ax[i], ax[j]) for i, j in perim]          # unit vectors, in order
    verts = []
    for k in range(nv + 1):                                # the body: nv+1 rings
        y = H * k / nv
        for dx, dz in ring:
            verts.append((R * dx, y, R * dz))
    cap_in = {}
    for cap, y in ((0, 0.0), (1, H)):                      # cap interiors only
        for j in range(1, g - 1):
            for i in range(1, g - 1):
                x, z = disc(ax[i], ax[j])
                cap_in[(cap, i, j)] = len(verts)
                verts.append((R * x, y, R * z))
    base = {0: 0, 1: nv * nu}
    pat = {p: u for u, p in enumerate(perim)}

    def cid(cap, i, j):
        return base[cap] + pat[(i, j)] if (i, j) in pat else cap_in[(cap, i, j)]

    faces = []
    for k in range(nv):                                    # the body
        for u in range(nu):
            a, b = k * nu + u, k * nu + (u + 1) % nu
            c, d = (k + 1) * nu + (u + 1) % nu, (k + 1) * nu + u
            faces += [(a, b, c), (a, c, d)]
    for cap in (0, 1):                                     # the caps
        for j in range(g - 1):
            for i in range(g - 1):
                a, b = cid(cap, i, j), cid(cap, i + 1, j)
                c, d = cid(cap, i + 1, j + 1), cid(cap, i, j + 1)
                faces += [(a, b, c), (a, c, d)]
    v = np.array(verts, np.float32)
    f = np.array(faces, np.int32)
    # Outward winding, per face and exactly: a drum is CONVEX, so a face points
    # outward iff its normal agrees with (face centroid - body centre). Cheaper
    # to get right than three regions of hand-checked index order.
    c0 = v.mean(0)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    out = ((v[f].mean(1) - c0) * n).sum(1) < 0.0
    f[out] = f[out][:, ::-1]
    return v, f


DRUMS = (not NO_DRUMS) and device is not None
if DRUMS:
    _v1, _f1 = drum_mesh(DRUM_G, DRUM_NV, DRUM_R, DRUM_H)
    NV1, NF1 = len(_v1), len(_f1)
    # The inner skin: every vertex pushed DRUM_THICK along its own area-weighted
    # normal. At the rim that normal is a 45 degree mix of body and cap, which
    # mitres the corner instead of leaving a gap there.
    _n1 = np.zeros((NV1, 3), np.float64)
    _fn = np.cross(_v1[_f1[:, 1]] - _v1[_f1[:, 0]], _v1[_f1[:, 2]] - _v1[_f1[:, 0]])
    for _c in range(3):
        np.add.at(_n1, _f1[:, _c], _fn)
    _n1 /= np.linalg.norm(_n1, axis=1)[:, None]
    _skin1 = np.concatenate([_v1, _v1 - DRUM_THICK * _n1]).astype(np.float32)

    _edges = sorted({(min(int(t[a]), int(t[b])), max(int(t[a]), int(t[b])))
                     for t in _f1 for a, b in ((0, 1), (1, 2), (2, 0))})
    _eid = {e: k for k, e in enumerate(_edges)}
    NE1 = len(_edges)
    _fe1 = np.array([[_eid[(min(int(t[a]), int(t[b])), max(int(t[a]), int(t[b])))]
                      for a, b in ((0, 1), (1, 2), (2, 0))] for t in _f1], np.int32)

    # The truss. Every constraint remembers WHICH MESH EDGE it belongs to, so a
    # tear retires all four of an edge's struts (both skins and both diagonals)
    # in one write and the faces can be tested against the edges alone.
    # (i, j, stiffness, yield, edge, kind) with kind 0 skin / 1 diagonal / 2 radial.
    _cons1 = []
    for _i, _j in _edges:
        _e = _eid[(_i, _j)]
        _cons1.append((_i, _j, STIFF_SKIN, YIELD_SKIN, _e, 0))
        _cons1.append((NV1 + _i, NV1 + _j, STIFF_SKIN, YIELD_SKIN, _e, 0))
        _cons1.append((_i, NV1 + _j, STIFF_DIAG, YIELD_DIAG, _e, 1))
        _cons1.append((_j, NV1 + _i, STIFF_DIAG, YIELD_DIAG, _e, 1))
    for _i in range(NV1):
        _cons1.append((_i, NV1 + _i, STIFF_RAD, YIELD_RAD, -1, 2))

    # Three copies, placed. One set of arrays holds all of them: the solve does
    # not care that the constraint graph has three components, and one launch
    # over three drums beats three launches over one.
    N_DRUM = len(DRUM_SITES)
    VPD = 2 * NV1                                  # vertices per drum, both skins
    _pos, _cons, _tri_o, _tri_i, _fe = [], [], [], [], []
    for _d, (_x, _z, _yaw) in enumerate(DRUM_SITES):
        _a = math.radians(_yaw)
        _rot = np.array([[math.cos(_a), 0.0, math.sin(_a)], [0.0, 1.0, 0.0],
                         [-math.sin(_a), 0.0, math.cos(_a)]], np.float32)
        _pos.append(_skin1 @ _rot.T + np.float32([_x, 0.0012, _z]))
        _vb, _eb = _d * VPD, _d * NE1
        for _i, _j, _s, _y, _e, _k in _cons1:
            _cons.append((_vb + _i, _vb + _j, _s, _y, -1 if _e < 0 else _eb + _e, _k))
        _tri_o.append(_f1 + _vb)
        _tri_i.append(_f1[:, ::-1] + _vb + NV1)    # inner skin: wound to face inward
        _fe.append(_fe1 + _eb)
    _p0 = np.concatenate(_pos).astype(np.float32)
    _tri_o = np.concatenate(_tri_o).reshape(-1).astype(np.int32)
    _tri_i = np.concatenate(_tri_i).reshape(-1).astype(np.int32)
    _fe = np.concatenate(_fe).reshape(-1).astype(np.int32)
    N_DV, N_DF, N_DE = len(_p0), NF1 * N_DRUM, NE1 * N_DRUM
    N_DC = len(_cons)
    N_CORNERS = N_DF * 3
    # Which outer vertex a vertex belongs to: the blast's facing weight uses the
    # OUTER skin's normal for both skins (they are 13 mm apart and share a face).
    _outer = np.concatenate([np.concatenate([np.arange(NV1), np.arange(NV1)])
                             + _d * VPD for _d in range(N_DRUM)]).astype(np.int32)

    # Greedy colouring, class by class, exactly as the press does it: a
    # constraint takes the lowest colour unused at either of its vertices, and
    # colouring skins first, then diagonals, then radials needs fewer colours
    # than the interleaved order.
    _cons.sort(key=lambda c: c[5])
    _used = [0] * N_DV
    _col = np.empty(N_DC, np.int32)
    for _k, _c in enumerate(_cons):
        _m = _used[_c[0]] | _used[_c[1]]
        _q = 0
        while (_m >> _q) & 1:
            _q += 1
        _col[_k] = _q
        _used[_c[0]] |= 1 << _q
        _used[_c[1]] |= 1 << _q
    _ord = np.argsort(_col, kind="stable")
    N_COL = int(_col.max()) + 1
    COL_N = np.bincount(_col, minlength=N_COL)
    COL_0 = np.concatenate(([0], np.cumsum(COL_N)[:-1]))

    _ci = np.array([_cons[k][0] for k in _ord], np.int32)
    _cj = np.array([_cons[k][1] for k in _ord], np.int32)
    _cs = np.array([_cons[k][2] for k in _ord], np.float32)
    _cy = np.array([_cons[k][3] for k in _ord], np.float32)
    _ce = np.array([_cons[k][4] for k in _ord], np.int32)
    _ck = np.array([_cons[k][5] for k in _ord], np.int32)
    _cr = np.linalg.norm(_p0[_ci] - _p0[_cj], axis=1).astype(np.float32)

    dx_ = wp.array(_p0, dtype=wp.vec3, device=device)
    dprev = wp.array(_p0, dtype=wp.vec3, device=device)
    dnrm = wp.zeros(N_DV, dtype=wp.vec3, device=device)
    dci = wp.array(_ci, dtype=int, device=device)
    dcj = wp.array(_cj, dtype=int, device=device)
    drest = wp.array(_cr, dtype=float, device=device)      # creeps: plasticity
    drest0 = wp.array(_cr, dtype=float, device=device)     # never moves: the tear datum
    dstiff = wp.array(_cs, dtype=float, device=device)
    ddmg = wp.zeros(N_DC, dtype=float, device=device)      # accumulated plastic strain
    dyield = wp.array(_cy, dtype=float, device=device)
    dcedge = wp.array(_ce, dtype=int, device=device)
    dckind = wp.array(_ck, dtype=int, device=device)
    dtri_o = wp.array(_tri_o, dtype=int, device=device)
    dtri_i = wp.array(_tri_i, dtype=int, device=device)
    dfe = wp.array(_fe, dtype=int, device=device)
    douter = wp.array(_outer, dtype=int, device=device)
    dealive = wp.array(np.ones(N_DE, np.int32), dtype=int, device=device)
    dfalive = wp.array(np.ones(N_DF, np.int32), dtype=int, device=device)
    dsoup_op = wp.zeros(N_CORNERS, dtype=wp.vec3, device=device)
    dsoup_on = wp.zeros(N_CORNERS, dtype=wp.vec3, device=device)
    dsoup_ip = wp.zeros(N_CORNERS, dtype=wp.vec3, device=device)
    dsoup_in = wp.zeros(N_CORNERS, dtype=wp.vec3, device=device)
    dtsub = wp.array(np.full(DRUM_SUBSTEPS, -1.0e9, np.float32), dtype=float,
                     device=device)
    ddt = wp.array(np.float32([DT / DRUM_SUBSTEPS]), dtype=float, device=device)
    # The device charge array the SHELLS read: (x, y, z, t0) and (j0, scale).
    # Separate from the gas's, because the drums must work with --no-gas and
    # because they need the yield, which the gas array has no room for.
    _dch = np.tile(np.float32([0.0, 0.0, 0.0, -1.0e9]), (MAX_CHARGES, 1))
    dchp = wp.array(_dch, dtype=wp.vec4, device=device)
    dchj = wp.zeros(MAX_CHARGES, dtype=wp.vec2, device=device)
    # When each vertex was hit by each charge slot, or BIG for "not yet". This is
    # what makes the arrival EXACT rather than a window test: a drum already in
    # flight changes its own r between substeps, so a "did t cross t_hit this
    # substep" test can miss the crossing entirely. Recording the hit instead
    # fires once, on the first substep at or after arrival, and then doubles as
    # the clock for that vertex's own wind tail.
    dhit = wp.array(np.full(N_DV * MAX_CHARGES, 1.0e9, np.float32), dtype=float,
                    device=device)


@wp.kernel
def drum_predict(x: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
                 dt_a: wp.array(dtype=float), damping: float, gravity: wp.vec3):
    """Verlet predict, with dt read from the DEVICE so the substep loop can be
    captured once and replayed at any frame length (phase 4's slow-motion ramp
    changes dt every frame; the graph must not care)."""
    i = wp.tid()
    dt = dt_a[0]
    p = x[i]
    v = (p - prev[i]) * (1.0 - damping)
    prev[i] = p
    x[i] = p + v + gravity * dt * dt


@wp.kernel
def drum_blast(x: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
               nrm: wp.array(dtype=wp.vec3), outer: wp.array(dtype=int),
               chp: wp.array(dtype=wp.vec4), chj: wp.array(dtype=wp.vec2),
               hit: wp.array(dtype=float), tsub: wp.array(dtype=float),
               dt_a: wp.array(dtype=float), s: int):
    """The front arrives at this VERTEX, and then the wind follows it.

    Same schedule as the rigid side -- t_hit = t0 + r/C_FRONT -- so a drum next
    to a wall caves when that wall goes, not when the charge does. The facing
    weight is what turns a uniform push into a CRUSH: the skin whose normal
    looks into the wave takes it all, the far skin is shadowed and only feels
    DRUM_FACE_MIN of it, and the difference closes the drum.

    A kick is applied by moving PREV, not x: in Verlet the velocity is
    (x - prev)/dt, so prev -= u*dv*dt adds dv with no teleport.
    """
    i = wp.tid()
    t = tsub[s]
    dt = dt_a[0]
    p = x[i]
    n = nrm[outer[i]]
    ln = wp.length(n)
    if ln > 1.0e-12:
        n = n / ln
    else:
        n = wp.vec3(0.0, 1.0, 0.0)
    d = wp.vec3(0.0, 0.0, 0.0)
    for c in range(MAX_CHARGES):
        cp = chp[c]
        if cp[3] > -1.0e8:
            q = p - wp.vec3(cp[0], cp[1], cp[2])
            r = wp.max(wp.length(q), 1.0e-4)
            u = q / r
            # the pressure only reaches the skin that can see the charge
            w = DRUM_FACE_MIN + (1.0 - DRUM_FACE_MIN) * wp.max(-wp.dot(u, n), 0.0)
            fall = wp.exp(-r / BLAST_L) / wp.pow(wp.max(r, BLAST_R0), BLAST_P)
            h = hit[i * MAX_CHARGES + c]
            if h > 1.0e8:
                if t >= cp[3] + r / C_FRONT:
                    dv = wp.min(DRUM_GAIN * chj[c][1] * fall, DRUM_VMAX) * w
                    d = d + u * (dv * dt)
                    hit[i * MAX_CHARGES + c] = t
            else:
                b = t - h
                if b < WIND_T:
                    a = (WIND_A0 * DRUM_WIND * chj[c][1] * w * fall
                         * wp.exp(-b / WIND_TAU))
                    d = d + u * (a * dt * dt)
    if wp.length(d) > 0.0:
        prev[i] = prev[i] - d


@wp.kernel
def drum_solve(x: wp.array(dtype=wp.vec3), ci: wp.array(dtype=int),
               cj: wp.array(dtype=int), rests: wp.array(dtype=float),
               stiffs: wp.array(dtype=float), start: int):
    """One colour: no two constraints here share a vertex, so the in-place
    Gauss-Seidel writes are race-free. A torn strut has stiffness zero."""
    k = start + wp.tid()
    s = stiffs[k]
    if s > 0.0:
        i, j = ci[k], cj[k]
        pi, pj = x[i], x[j]
        dv = pj - pi
        l = wp.length(dv)
        if l > 1.0e-9:
            corr = dv * (0.5 * s * (l - rests[k]) / l)
            x[i] = pi + corr
            x[j] = pj - corr


@wp.kernel
def drum_contacts(x: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3)):
    """Ground plane at y = 0 with Coulomb-ish friction. This is the whole
    toppling model: the shell's own vertices are what stands on the floor, so a
    drum knocked off balance goes over because its footprint left."""
    i = wp.tid()
    p = x[i]
    if p[1] < 0.0:
        p = wp.vec3(p[0], 0.0, p[2])
        t = p - prev[i]
        p = p - wp.vec3(t[0], 0.0, t[2]) * DRUM_MU
    x[i] = p


@wp.kernel
def drum_plastic(x: wp.array(dtype=wp.vec3), ci: wp.array(dtype=int),
                 cj: wp.array(dtype=int), rests: wp.array(dtype=float),
                 rest0: wp.array(dtype=float), stiffs: wp.array(dtype=float),
                 yields: wp.array(dtype=float), dmg: wp.array(dtype=float),
                 rate: float):
    """Past yield, the rest length creeps toward the length it actually has.
    That is what makes a crease PERSIST after the wave has gone by -- and every
    metre of that creep, stretch or squash alike, is logged as damage, which is
    what eventually tears the shell open."""
    k = wp.tid()
    if stiffs[k] > 0.0:
        l = wp.length(x[cj[k]] - x[ci[k]])
        r = rests[k]
        y = yields[k]
        d = float(0.0)
        if l > r * (1.0 + y):
            d = (l - r * (1.0 + y)) * rate
        elif l < r * (1.0 - y):
            d = (l - r * (1.0 - y)) * rate
        if d != 0.0:
            rests[k] = r + d
            dmg[k] = dmg[k] + wp.abs(d) / rest0[k]


@wp.kernel
def drum_break(x: wp.array(dtype=wp.vec3), ci: wp.array(dtype=int),
               cj: wp.array(dtype=int), rest0: wp.array(dtype=float),
               stiffs: wp.array(dtype=float), dmg: wp.array(dtype=float),
               cedge: wp.array(dtype=int), ckind: wp.array(dtype=int),
               ealive: wp.array(dtype=int)):
    """A SKIN strut that has worked itself to death, or been yanked apart.

    Several struts on one mesh edge can fire at once; they all write the same
    zero, so the race is benign and no atomic is needed."""
    k = wp.tid()
    if ckind[k] == 0 and stiffs[k] > 0.0:
        if (dmg[k] > TEAR
                or wp.length(x[cj[k]] - x[ci[k]]) > rest0[k] * TEAR_SNAP):
            ealive[cedge[k]] = 0


@wp.kernel
def drum_clear_slot(hit: wp.array(dtype=float), slot: int):
    """Every vertex's arrival state against one charge slot, reset. The slot is
    a ring: a ninth detonation re-uses the oldest and starts it over."""
    hit[wp.tid() * MAX_CHARGES + slot] = 1.0e9


@wp.kernel
def drum_cut(cedge: wp.array(dtype=int), ealive: wp.array(dtype=int),
             stiffs: wp.array(dtype=float)):
    """Retire every strut of a dead edge -- both skins and both diagonals."""
    k = wp.tid()
    e = cedge[k]
    if e >= 0:
        if ealive[e] == 0:
            stiffs[k] = 0.0


@wp.kernel
def drum_faces(fe: wp.array(dtype=int), ealive: wp.array(dtype=int),
               falive: wp.array(dtype=int)):
    """A face lives while ALL THREE of its edges do. One broken edge therefore
    opens a two-triangle slit, and a run of them is a tear."""
    f = wp.tid()
    n = ealive[fe[f * 3 + 0]] + ealive[fe[f * 3 + 1]] + ealive[fe[f * 3 + 2]]
    falive[f] = wp.where(n == 3, 1, 0)


@wp.kernel
def drum_normals(x: wp.array(dtype=wp.vec3), tris: wp.array(dtype=int),
                 falive: wp.array(dtype=int), nrm: wp.array(dtype=wp.vec3)):
    """Area-weighted face normals onto the vertices -- LIVE faces only, so the
    lip of a tear is shaded by the metal that is still there."""
    f = wp.tid()
    if falive[f] != 0:
        ia, ib, ic = tris[f * 3 + 0], tris[f * 3 + 1], tris[f * 3 + 2]
        n = wp.cross(x[ib] - x[ia], x[ic] - x[ia])
        wp.atomic_add(nrm, ia, n)
        wp.atomic_add(nrm, ib, n)
        wp.atomic_add(nrm, ic, n)


@wp.kernel
def drum_soup(x: wp.array(dtype=wp.vec3), nrm: wp.array(dtype=wp.vec3),
              tri_o: wp.array(dtype=int), tri_i: wp.array(dtype=int),
              falive: wp.array(dtype=int),
              op: wp.array(dtype=wp.vec3), on: wp.array(dtype=wp.vec3),
              ip: wp.array(dtype=wp.vec3), inn: wp.array(dtype=wp.vec3)):
    """De-index BOTH skins into their triangle soups in ONE pass.

    A dead face collapses to a point far below the floor -- zero area, nothing
    rasterised, nothing in the BLAS. It is NOT compacted out of the soup and
    that is deliberate: the renderer keeps a previous-frame vertex buffer to
    build motion vectors from, so a soup whose triangles change slot between
    frames hands TAA garbage exactly where the hero shot is. A stable soup with
    holes in it costs a few thousand degenerate triangles and keeps the
    reprojection honest.
    """
    k = wp.tid()
    f = k / 3
    if falive[f] == 0:
        dead = wp.vec3(0.0, -60.0, 0.0)
        op[k] = dead
        ip[k] = dead
        on[k] = wp.vec3(0.0, 1.0, 0.0)
        inn[k] = wp.vec3(0.0, 1.0, 0.0)
    else:
        a = tri_o[k]
        op[k] = x[a]
        na = nrm[a]
        on[k] = na / wp.max(wp.length(na), 1.0e-9)
        b = tri_i[k]
        ip[k] = x[b]
        nb = nrm[b]
        inn[k] = nb / wp.max(wp.length(nb), 1.0e-9)


if DRUMS:
    def drum_launches(n=DRUM_SUBSTEPS):
        """Every launch of one frame of shell simulation; captured into a graph.

        `n` is the substep count and it is ONLY ever less than DRUM_SUBSTEPS off
        the captured path, for the slow-motion ramp -- see step_drums. The
        substep SIZE never changes, which is the whole point.
        """
        for s in range(n):
            wp.launch(drum_predict, dim=N_DV, device=device,
                      inputs=[dx_, dprev, ddt, DRUM_DAMP, DRUM_GRAV])
            wp.launch(drum_blast, dim=N_DV, device=device,
                      inputs=[dx_, dprev, dnrm, douter, dchp, dchj, dhit,
                              dtsub, ddt, s])
            for c in range(N_COL):
                wp.launch(drum_solve, dim=int(COL_N[c]), device=device,
                          inputs=[dx_, dci, dcj, drest, dstiff, int(COL_0[c])])
            wp.launch(drum_contacts, dim=N_DV, device=device, inputs=[dx_, dprev])
            wp.launch(drum_plastic, dim=N_DC, device=device,
                      inputs=[dx_, dci, dcj, drest, drest0, dstiff, dyield, ddmg,
                              PLASTIC_RATE])
            if s % 8 == 7:      # tearing does not need 48 Hz x 60; six a frame is plenty
                wp.launch(drum_break, dim=N_DC, device=device,
                          inputs=[dx_, dci, dcj, drest0, dstiff, ddmg, dcedge,
                                  dckind, dealive])
                wp.launch(drum_cut, dim=N_DC, device=device,
                          inputs=[dcedge, dealive, dstiff])
        wp.launch(drum_faces, dim=N_DF, device=device, inputs=[dfe, dealive, dfalive])
        dnrm.zero_()
        wp.launch(drum_normals, dim=N_DF, device=device,
                  inputs=[dx_, dtri_o, dfalive, dnrm])
        wp.launch(drum_normals, dim=N_DF, device=device,
                  inputs=[dx_, dtri_i, dfalive, dnrm])
        wp.launch(drum_soup, dim=N_CORNERS, device=device,
                  inputs=[dx_, dnrm, dtri_o, dtri_i, dfalive,
                          dsoup_op, dsoup_on, dsoup_ip, dsoup_in])

    def drum_boot():
        """Warm the Warp module, then capture the substep loop into a graph.

        Deliberately NOT run here. Warp builds the whole module the first time
        anything in it launches, and every kernel defined afterwards changes the
        module's hash -- so warming the shells before the gas kernels below even
        exist would compile the module twice and, worse, leave the captured
        graph pointing into the copy that got unloaded. This runs once from the
        boot block after the last kernel in the file.

        The warm pass runs at dt = 0: every kernel compiles, every buffer ends
        up holding the rest pose, and nothing moves.
        """
        global d_graph
        ddt.assign(np.float32([0.0]))
        drum_launches()
        wp.launch(drum_clear_slot, dim=N_DV, device=device, inputs=[dhit, 0])
        wp.synchronize_device(device)
        ddt.assign(np.float32([DT / DRUM_SUBSTEPS]))
        if device.is_cuda:
            with wp.ScopedCapture(device) as cap:
                drum_launches()
            d_graph = cap.graph
        lpf = DRUM_SUBSTEPS * (N_COL + 4) + 4
        print(f"drums: {N_DRUM} shells, {N_DV:,} vertices (2 skins), {N_DF:,} faces, "
              f"{N_DC:,} struts in {N_COL} colours, {DRUM_SUBSTEPS} substeps "
              f"({lpf:,} launches/frame{', CUDA graph' if d_graph else ''})")

    # ── the meshes: two fixed-capacity triangle soups ───────────────────────
    # OUTER is bare steel and the reason this scene is on Vulkan -- it takes the
    # ray-traced reflection of the yard and the flash. INNER is the same faces
    # wound the other way in a dark, nearly matte paint, so the moment the shell
    # tears open the lip shows the drum's unlit inside. It is the cheapest thing
    # that reads as "torn METAL" rather than "a hole in a balloon".
    drum_geo_o = tp.BufferGeometry()
    drum_geo_o.set_attribute("position", _p0[_tri_o])
    drum_geo_o.set_attribute("normal", np.tile(np.float32([0, 1, 0]), (N_CORNERS, 1)))
    drum_geo_o.set_draw_range(0, N_CORNERS)
    drum_out = tp.Mesh(drum_geo_o, standard_material(tp.Color(0.63, 0.65, 0.68),
                                                     0.17, 1.0))
    drum_geo_i = tp.BufferGeometry()
    drum_geo_i.set_attribute("position", _p0[_tri_i])
    drum_geo_i.set_attribute("normal", np.tile(np.float32([0, -1, 0]), (N_CORNERS, 1)))
    drum_geo_i.set_draw_range(0, N_CORNERS)
    drum_in = tp.Mesh(drum_geo_i, standard_material(tp.Color(0.055, 0.048, 0.042),
                                                    0.72, 0.30))
    for _m in (drum_out, drum_in):
        _m.cast_shadow = True
        _m.receive_shadow = True
        _m.frustum_culled = False       # the bounds go stale the moment it detonates
        scene.add(_m)

    def claim_drum_slot(ch):
        """A new charge's row in the shells' charge array, and the arrival state
        of every vertex against it cleared. Two host writes and one launch --
        nothing here walks a vertex."""
        _dch[ch.slot] = (ch.pos[0], ch.pos[1], ch.pos[2], ch.t0)
        dchp.assign(_dch)
        _j = dchj.numpy()
        _j[ch.slot] = (ch.j0, ch.scale)
        dchj.assign(_j)
        wp.launch(drum_clear_slot, dim=N_DV, device=device, inputs=[dhit, ch.slot])
else:
    N_DV = N_DF = N_DC = N_DE = N_COL = 0

    def claim_drum_slot(ch):
        pass

    def drum_boot():
        pass


DRUM_ROUTE = "off"
d_vk = None
d_graph = None


def arm_drum_interop():
    """Point the soup kernels straight at the renderer's own vertex buffers.

    Must run AFTER the first render(): a mesh's record -- and so the allocation
    there is anything to export -- is created on the frame it is first drawn.
    Any failure falls back to the press's host route, which needs neither the
    external-memory export nor the CUDA import.
    """
    global d_vk, DRUM_ROUTE
    if not DRUMS:
        return
    DRUM_ROUTE = "host copy"
    if not device.is_cuda or not hasattr(renderer, "enable_vertex_interop"):
        return
    try:
        from threepp.cuda_interop import VkInteropArray
    except ImportError:
        return
    handles = []
    try:
        for m in (drum_out, drum_in):
            h = renderer.enable_vertex_interop(m, drum_on_frame if m is drum_out
                                               else drum_on_frame_in)
            if h is None:
                raise RuntimeError("no external-memory export for this mesh")
            handles.append(h)
        d_vk = [VkInteropArray(hh[0], hh[1], wp.vec3, N_CORNERS, device)
                for h in handles for hh in h]
    except Exception as e:
        print(f"  note: drum vertex interop did not arm ({e}) -- host copy route")
        for m in (drum_out, drum_in):
            try:
                renderer.disable_vertex_interop(m)
            except Exception:
                pass
        d_vk = None
        return
    DRUM_ROUTE = "zero-copy CUDA -> Vulkan"
    atexit.register(release_drum_interop)


def release_drum_interop():
    """Drop the CUDA mappings BEFORE the renderer frees the memory they map."""
    global d_vk
    if d_vk is None:
        return
    arrays, d_vk = d_vk, None
    for a in arrays:
        a.close()
    for m in (drum_out, drum_in):
        renderer.disable_vertex_interop(m)


# Frames of publishing still owed after the last solve. A FROZEN shell's soup is
# byte-identical to the one already in the renderer's buffers, and copying it
# again costs two device-to-device copies AND two wp.synchronize_device calls
# inside render() -- which is a WAIT on the whole Vulkan frame, not a copy. That
# is the entire pre-detonation cost of the drums (measured: 211 fps against 307
# with them frozen), so the callbacks are gated on this. It counts DOWN rather
# than being a bool because the renderer keeps a previous-frame vertex buffer to
# build motion vectors from: the first frames after the shells stop still have to
# publish, or the last motion the drum made never leaves the history.
_dpub = [3]
_dacc = [0.0]        # sim seconds owed to the shells, not yet stepped
_dclk = [0.0]        # the shells' own clock: sim time at the last substep taken


def drum_on_frame():
    """Inside render(), post-fence and pre-record: fill the outer skin's buffers.

    The synchronize is the whole contract -- wp.launch is asynchronous on Warp's
    stream and host ordering here is the only thing sequencing this write
    against the Vulkan frame that reads it."""
    if d_vk is not None and _dpub[0] > 0:
        wp.copy(d_vk[0].array, dsoup_op)
        wp.copy(d_vk[1].array, dsoup_on)
        wp.synchronize_device(device)


def drum_on_frame_in():
    if d_vk is not None and _dpub[0] > 0:
        wp.copy(d_vk[2].array, dsoup_ip)
        wp.copy(d_vk[3].array, dsoup_in)
        wp.synchronize_device(device)


def drums_active():
    """The shells run while they are settling and for DRUM_RUN_T after the
    newest charge; between those they are FROZEN, because a thousand launches a
    frame to hold three drums still is a thousand launches wasted."""
    if sim_time < DRUM_SETTLE:
        return True
    st = blast_timeline(sim_time)
    return st["armed"] and st["tw"] < DRUM_RUN_T


def step_drums(dt):
    """One frame of shell simulation, at a FIXED substep size. Wall-clock secs.

    THE SUBSTEP SIZE IS A MATERIAL PROPERTY HERE, not a discretisation choice,
    and the slow-motion ramp is what proves it: the Verlet damping (1 - 0.002 per
    substep) and the plastic creep (5% of the excess per substep) are both
    per-SUBSTEP rates, so running the same 48 substeps over a sixth of the sim
    time makes the shells six times as damped and six times as ductile per second
    of yard. The first slow-motion cut came out with three pristine drums.
    So the shells keep a fixed DRUM_SUB and an accumulator, exactly as PhysX
    does: at full speed that is one captured-graph replay a frame, and at a sixth
    speed it is seven or eight uncaptured substeps a frame -- fine granularity
    where the film needs it, and the identical material either way. The graph is
    still the fast path and its shape never changes.
    """
    if not DRUMS or not drums_active():
        if _dpub[0] > 0:            # let the last motion out of the history, then stop
            _dpub[0] -= 1
        _dacc[0] = 0.0
        _dclk[0] = sim_time
        return 0.0
    t0 = time.perf_counter()
    _dacc[0] += dt
    n = int(_dacc[0] / DRUM_SUB + 1.0e-9)
    if n <= 0:                      # slow motion: not a whole substep's worth yet
        return time.perf_counter() - t0
    n = min(n, 4 * DRUM_SUBSTEPS)   # no spiral of death on a compile hitch
    _dacc[0] -= n * DRUM_SUB
    _dpub[0] = 3
    while n > 0:
        k = min(n, DRUM_SUBSTEPS)
        dtsub.assign((_dclk[0] + (np.arange(DRUM_SUBSTEPS) + 1)
                      * DRUM_SUB).astype(np.float32))
        _dclk[0] += k * DRUM_SUB
        if k == DRUM_SUBSTEPS and d_graph is not None:
            wp.capture_launch(d_graph)
        else:
            drum_launches(k)
        n -= k
        if n > 0:                   # the next batch rewrites dtsub from the host
            wp.synchronize_device(device)
    wp.synchronize_device(device)
    if d_vk is None:            # the fallback: four attribute uploads a frame
        drum_geo_o.update_attribute("position", dsoup_op.numpy())
        drum_geo_o.update_attribute("normal", dsoup_on.numpy())
        drum_geo_i.update_attribute("position", dsoup_ip.numpy())
        drum_geo_i.update_attribute("normal", dsoup_in.numpy())
    if DRUM_CAM:                # 26 kB of readback, in the macro mode only
        c = dx_.numpy()[:VPD].mean(0)
        for k in range(3):
            _focus[k] += (float(c[k]) - _focus[k]) * 0.25    # a lagging dolly
    return time.perf_counter() - t0


def drum_stats():
    """Torn edges, dead faces and each shell's remaining height and footprint."""
    if not DRUMS:
        return None
    alive = int(dealive.numpy().sum())
    faces = int(dfalive.numpy().sum())
    p = dx_.numpy().reshape(N_DRUM, -1, 3)
    hw = [(float(d[:, 1].max()), float(np.hypot(d[:, 0] - d[:, 0].mean(),
                                                d[:, 2] - d[:, 2].mean()).max() * 2.0))
          for d in p]
    g = ddmg.numpy()[_ck == 0]           # the skin struts: what the tear tests
    return dict(edges=alive, torn=N_DE - alive, faces=faces, gone=N_DF - faces,
                shells=hw, nan=int((~np.isfinite(p)).sum()),
                dmg=(float(np.percentile(g, 99)), float(g.max())))


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
        self.embers = None            # its own ember field, from the ring
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
        claim_drum_slot(self)                      # the shells' own charge array
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
    bb_gain  additive-billboard brightness multiplier -- the flash spike, the
             MAX over live charges (never the sum: a camera does not respond
             twice to two flashes, and summing let a new charge re-light an old
             charge's matter). Billboards composite after the upscaler and are
             NOT seen by auto-exposure, so this is the whole exposure story for
             them. Anything with its OWN clock (the ember ring) must use that
             clock instead -- see step_gas_frame.
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
        gain = max(gain, FLASH_BB * math.exp(-b / 0.13))   # the NEWEST flash, not the sum
        fire_e = max(fire_e, math.exp(-max(b - 0.22, 0.0) / 0.55))
        s = ch.shake(t)
        shake = [a + c for a, c in zip(shake, s)]
    if tw < -1.0e8:
        return dict(tw=-1.0, armed=False, front=0.0, shake=(0.0, 0.0, 0.0),
                    bb_gain=0.0, fire_e=0.0)
    return dict(tw=tw, armed=True, front=C_FRONT * tw, shake=tuple(shake),
                bb_gain=1.0 + gain, fire_e=fire_e)


def blast_frame(t, dt=DT):
    """Arm what is due, deliver the front's impulses, blow the wind behind it.

    This is the one place the rigid side hears about the blast, and it is
    O(bodies the front crossed this frame) plus O(bodies inside the wind
    window) -- not O(bodies) every frame.

    `dt` is here for ONE reason and it is the slow-motion ramp. PhysX runs a
    FIXED 1/60 step against an accumulator, so at a sixth speed six rendered
    frames go by per physics step -- and PxRigidBody::addForce ACCUMULATES until
    simulate() consumes it. Six frames of the same wind force therefore deliver
    six times the impulse over that one step. Scaling by dt/DT hands each frame
    its own share, which comes out identical at any playback speed (verified the
    hard way: the first slow-motion cut threw a brick 287 m).
    """
    fscale = dt / DT
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
            amp = (WIND_A0 * ch.scale * fscale
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
NOISE_FREQ = 0.085                     # ~12 m eddies: puffs, not grain
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
        #
        # THE FRONT DECAYS, and the parcel is launched at the front's CURRENT
        # speed with a drag equal to that decay -- so it stays on the front for
        # its whole life without a shred of per-parcel bookkeeping, and the
        # ring visibly slows instead of scaling at a constant rate like a
        # script. It is a DOME too: the shell stands taller as it grows and is
        # packed toward its own foot, where a real Mach stem is brightest.
        dec = wp.exp(-b / WILSON_TD)
        rr = front_c * WILSON_TD * (1.0 - dec) * (0.985 + 0.030 * wp.randf(s))
        # A HEMISPHERE, not a ring on stilts. Sampling a horizontal radius and
        # an independent height builds a CYLINDER however the velocities are
        # dressed up -- the silhouette is a wall with a flat top, which is what
        # it looked like. A blast front is a spherical shell: pick a DIRECTION
        # on the hemisphere and put the parcel at exactly rr along it, so the
        # shape is a sphere by construction and stays one as it grows.
        # sin(elevation) = u with u packed toward 0 keeps the parcels dense at
        # the foot -- where a real Mach stem is brightest -- without bending
        # the geometry: it is a density gradient over a true sphere, not a
        # squashed one.
        u = wp.pow(wp.randf(s), WILSON_HP)          # 0 = horizon, 1 = crown
        hz = wp.sqrt(wp.max(1.0 - u * u, 0.0))      # horizontal component
        d = wp.vec3(wp.cos(a) * hz, u, wp.sin(a) * hz)
        p = wp.vec3(ch[0] + d[0] * rr, 0.45 + d[1] * rr, ch[2] + d[2] * rr)
        sp = v0 * dec * (0.92 + 0.16 * wp.randf(s))
        # Purely RADIAL along the same direction: every parcel rides its own
        # normal, so the sphere expands as a sphere instead of shearing.
        v = wp.vec3(d[0] * sp, d[1] * sp, d[2] * sp)
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
def emit_puff(pos: wp.array(dtype=wp.vec4),
              vel: wp.array(dtype=wp.vec3),
              st: wp.array(dtype=wp.vec2),
              pts: wp.array(dtype=wp.vec4),
              base: int, cap: int, per: int, seed: int,
              v0: float, life0: float, life1: float, r0: float):
    """IMPACT DUST: a small ground puff where a piece of debris just landed.

    The second exception to "nothing is instantiated after the burst", and it
    earns it the same way the dust ring does -- this is not new matter appearing
    out of nowhere, it is the floor of the yard being kicked up by something that
    hit it. `pts` is (x, _, z, strength) for the strikes the host found this
    frame, at most PUFF_MAX of them, and the launch is 2D so each strike gets
    `per` parcels without an integer division in the kernel.

    The parcels go into the DUST cohort's ring at its cursor: same pool, same
    allocator, same advection (kind 2 -- gravity, low drag, a per-particle rest
    height), so a puff settles exactly like the shock skirt does. It just starts
    where a brick landed instead of on the front.
    """
    k, j = wp.tid()
    idx = k * per + j
    slot = (base + idx) % cap
    s = wp.rand_init(seed, base + idx)
    hit = pts[k]
    a = wp.randf(s) * 6.2831853
    rr = 0.07 + 0.40 * wp.sqrt(wp.randf(s))
    p = wp.vec3(hit[0] + rr * wp.cos(a), 0.04 + 0.20 * wp.randf(s),
                hit[2] + rr * wp.sin(a))
    # Mostly UP and a little out: a strike throws a collar, not a fountain.
    sp = v0 * hit[3] * (0.30 + 0.95 * wp.randf(s))
    v = wp.vec3(wp.cos(a) * sp * 0.85, sp * (0.55 + 0.85 * wp.randf(s)),
                wp.sin(a) * sp * 0.85)
    pos[slot] = wp.vec4(p[0], p[1], p[2], r0)
    vel[slot] = v
    st[slot] = wp.vec2(0.0, life0 + (life1 - life0) * wp.randf(s))


@wp.kernel
def bake_sdf(bc: wp.array(dtype=wp.vec3), bh: wp.array(dtype=wp.vec3), nb: int,
             g: wp.array3d(dtype=float)):
    """Exact AABB signed distance to the nearest of `nb` boxes, per voxel.

    One 3D launch over the grid, one linear scan over the yard inside it. It is
    ~750 M box tests, which is tens of milliseconds ONCE -- and it is the whole
    reason smoke can be told about twelve structures without a broad phase, an
    obstacle list or a single host loop in the frame.
    """
    i, j, k = wp.tid()
    p = wp.vec3(SDF_OX + float(i) * SDF_CELL,
                SDF_OY + float(j) * SDF_CELL,
                SDF_OZ + float(k) * SDF_CELL)
    d = float(1.0e9)
    for b in range(nb):
        c = bc[b]
        h = bh[b]
        q = wp.vec3(wp.abs(p[0] - c[0]) - h[0],
                    wp.abs(p[1] - c[1]) - h[1],
                    wp.abs(p[2] - c[2]) - h[2])
        o = wp.vec3(wp.max(q[0], 0.0), wp.max(q[1], 0.0), wp.max(q[2], 0.0))
        d = wp.min(d, wp.length(o) + wp.min(wp.max(q[0], wp.max(q[1], q[2])), 0.0))
    g[i, j, k] = wp.clamp(d, -2.0, SDF_BAND)


@wp.kernel
def bake_sdf_grad(g: wp.array3d(dtype=float), out: wp.array3d(dtype=wp.vec4)):
    """Central differences into the xyz of a vec4 whose w is the distance."""
    i, j, k = wp.tid()
    i0, i1 = wp.max(i - 1, 0), wp.min(i + 1, SDF_NX - 1)
    j0, j1 = wp.max(j - 1, 0), wp.min(j + 1, SDF_NY - 1)
    k0, k1 = wp.max(k - 1, 0), wp.min(k + 1, SDF_NZ - 1)
    n = wp.vec3(g[i1, j, k] - g[i0, j, k],
                g[i, j1, k] - g[i, j0, k],
                g[i, j, k1] - g[i, j, k0])
    l = wp.length(n)
    if l > 1.0e-6:
        n = n / l
    else:
        n = wp.vec3(0.0, 1.0, 0.0)
    out[i, j, k] = wp.vec4(n[0], n[1], n[2], g[i, j, k])


@wp.func
def sdf_sample(g: wp.array3d(dtype=wp.vec4), p: wp.vec3) -> wp.vec4:
    """Trilinear fetch. Outside the grid this returns "far", NOT the edge value:
    a clamp would smear the yard's outermost wall across the rest of the world.
    """
    gx = (p[0] - SDF_OX) * SDF_INV
    gy = (p[1] - SDF_OY) * SDF_INV
    gz = (p[2] - SDF_OZ) * SDF_INV
    if (gx < 0.0 or gy < 0.0 or gz < 0.0 or gx > float(SDF_NX - 1) - 1.0e-3
            or gy > float(SDF_NY - 1) - 1.0e-3 or gz > float(SDF_NZ - 1) - 1.0e-3):
        return wp.vec4(0.0, 1.0, 0.0, 1.0e9)
    i, j, k = int(wp.floor(gx)), int(wp.floor(gy)), int(wp.floor(gz))
    fx, fy, fz = gx - float(i), gy - float(j), gz - float(k)
    c00 = g[i, j, k] * (1.0 - fx) + g[i + 1, j, k] * fx
    c10 = g[i, j + 1, k] * (1.0 - fx) + g[i + 1, j + 1, k] * fx
    c01 = g[i, j, k + 1] * (1.0 - fx) + g[i + 1, j, k + 1] * fx
    c11 = g[i, j + 1, k + 1] * (1.0 - fx) + g[i + 1, j + 1, k + 1] * fx
    return ((c00 * (1.0 - fy) + c10 * fy) * (1.0 - fz)
            + (c01 * (1.0 - fy) + c11 * fy) * fz)


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
             s_r0: float, s_inherit: float, s_spread: float,
             gch: wp.array(dtype=wp.vec4), gcs: wp.array(dtype=float),
             t: float, bw: float,
             wind: wp.vec3, curl_gain: float,
             hub: wp.vec3, ent: float, ent_k: float,
             sdf: wp.array3d(dtype=wp.vec4), sdfw: float,
             bpow: float, bgain: float):
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

    AND EVERY LIVE PARCEL FEELS EVERY BLAST. `gch` is the same fixed 8-slot
    charge array the emission kernel indexes (xyz + t0, w <= -1e8 = empty) and
    `t` is the ABSOLUTE sim time at the END of this frame; `bw` is this cohort's
    weight, and zero on the frames when no front is open, which is what keeps
    the loop off fourteen million parcels for the rest of the shot.
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
                # RETAINED MOMENTUM VARIES TOO. Handing every soot parcel the
                # same fraction of the fireball's climb launches the whole
                # population at one speed, and no spread of buoyancy afterwards
                # can pull it back apart -- they coast up together and the
                # cloud leaves the crater behind. Soot that was entrained and
                # cooled kept very little of the fireball's motion; soot from
                # the core kept nearly all of it. Drawing that fraction is what
                # puts mass at the base, mass in the head, and a stem between.
                s_vel[j] = v * (s_inherit * (0.18 + 0.82 * wp.pow(wp.randf(s), 1.8)))                            + jv * s_spread
                s_st[j] = wp.vec2(0.0, s_life0 + (s_life1 - s_life0) * wp.randf(s))
        return
    # ── advect ──────────────────────────────────────────────────────────────
    # The stir GROWS with a parcel's own age, and the growth arrives as a finer
    # octave. Young soot keeps the coherent, thick mass that the density splat
    # rewards; old soot shreds into wisps and breaks up, which is what a plume
    # actually does when its buoyancy is spent. `curl_gain` is a per-cohort
    # scalar, so this branch is uniform across a launch, not divergent.
    acc = gas_noise(noise, p) * curl
    if curl_gain > 0.0:
        ag = curl * curl_gain * wp.min(age / life, 1.0)
        acc = acc + gas_noise(noise, p * 2.4) * ag
    # ── the front sweeps past, and the wind follows it ──────────────────────
    # The impulse is NOT multiplied by dt: it is applied once because the shell
    # test fires once, not because a frame is 1/60 s. The tail IS an
    # acceleration and is added to `acc`, which the integration below already
    # multiplies by dt. (Do not copy the rigid side's dt/DT scale -- that exists
    # only because PhysX's addForce accumulates between simulate() calls.)
    if bw > 0.0:
        for c in range(MAX_CHARGES):
            cp = gch[c]
            if cp[3] > -1.0e8:
                b1 = t - cp[3]
                if b1 >= 0.0 and b1 < GAS_BLAST_T:
                    b0 = b1 - dt
                    dq = p - wp.vec3(cp[0], cp[1], cp[2])
                    rr = wp.max(wp.length(dq), 1.0e-4)
                    u = dq / rr
                    fall = GAS_L * GAS_L / (GAS_L * GAS_L + rr * rr)
                    R0 = C_FRONT * b0
                    R1 = C_FRONT * b1
                    if rr >= R0 and rr < R1 and rr > GAS_RMIN:
                        # The velocity gate is load-bearing, not decoration: the
                        # condensation ring is LAUNCHED at ~C_FRONT and rides the
                        # shell, so without it the ring is kicked every frame and
                        # accelerates off its own front.
                        if wp.dot(v, u) < 0.75 * C_FRONT:
                            dv = wp.min(GAS_KICK * gcs[c] * fall, GAS_VMAX) * bw
                            lift = GAS_LIFT * wp.exp(-wp.max(p[1], 0.0) / 2.0)
                            v = v + u * dv + wp.vec3(0.0, dv * lift, 0.0)
                    elif rr < R0:
                        tb = b1 - rr / C_FRONT      # since THIS parcel's arrival
                        if tb < WIND_T:
                            e = wp.exp(-tb / WIND_TAU)
                            acc = acc + u * (GAS_WIND_A0 * gcs[c] * fall * e * bw)
                            acc = acc + gas_noise(noise, p * 2.0) * (GAS_STIR * e * bw)
    # ── ENTRAINMENT: the column widens as it climbs ─────────────────────────
    # Measured on the shipped build, the plume grew 2.3 m wider while it climbed
    # 9.3 m -- a narrow finger, not a cloud, because the only lateral motion in
    # the model was a divergence-free curl field, and a curl field MIXES without
    # spreading. A real plume entrains still air and its radius grows roughly
    # linearly with height above the source. That is one number: push outward
    # from the column's own axis while the parcel is inside the radius its
    # height entitles it to, and stop. `hub` is the box-follow's own centroid,
    # already measured every eighth frame for free, so this costs no readback.
    if ent > 0.0:
        dxz = wp.vec3(p[0] - hub[0], 0.0, p[2] - hub[2])
        rl = wp.length(dxz)
        want = 2.5 + ent_k * wp.max(p[1] - hub[1], 0.0)
        if rl > 0.25 and rl < want:
            acc = acc + dxz * (ent * (1.0 - rl / want) / rl)
    # BUOYANCY IS PER-PARCEL, and that is what keeps the column joined to its
    # own crater. Every parcel is born inside the same half-second burst, so a
    # SINGLE buoyancy curve keyed on age lifts the whole population in lockstep:
    # it leaves the ground as one bubble and drifts off as a blob with clear air
    # underneath -- which is exactly what "decoupled from the explosion" looks
    # like. A real plume is a DISTRIBUTION. The hot core climbs, entrained
    # cooler soot hangs around the crater, and everything in between is the
    # stem that joins them. Skewed so the strong risers are the minority they
    # are in a real column. Hashed on the parcel index, so it is fixed for that
    # parcel's whole life and costs no state to carry.
    bs = 1.0
    if bpow > 0.0:
        sb = wp.rand_init(seed + 4242, i)
        bs = wp.pow(wp.randf(sb), bpow) * bgain
    acc = acc + wp.vec3(0.0, buoy * bs * wp.exp(-age / buoy_tau) - grav, 0.0)
    # AMBIENT WIND, on a height ramp. The drag term is what carries it -- a
    # parcel relaxes toward the local air, not toward zero -- so at SMOKE_DRAG
    # the head reaches wind speed in two thirds of a second and shears against
    # the still stem below it. The ramp keeps the ground dust where it settled.
    wr = wp.clamp((p[1] - AMB_Y0) / (AMB_Y1 - AMB_Y0), 0.0, 1.0)
    acc = acc - (v - wind * wr) * drag
    v = v + acc * dt
    p = p + v * dt
    # ── the yard is solid ───────────────────────────────────────────────────
    # One vec4 fetch from the baked SDF: unit outward gradient and signed
    # distance. Push out of anything it is inside, kill the INTO-surface part of
    # its velocity but keep the tangent (which is what makes smoke climb a wall
    # and roll over the top instead of stopping dead against it), drag the
    # tangent a little so it clings, and stir the shear layer so the slide reads
    # as a curl rather than a decal. Gaps -- the doorway, the gateway's span,
    # the pipe rack -- are free: at 0.35 m they are simply open.
    if sdfw > 0.0 and p[1] < SDF_TOP:
        sd = sdf_sample(sdf, p)
        d = sd[3]
        if d < SDF_SKIN:
            n = wp.vec3(sd[0], sd[1], sd[2])
            if d < 0.0:
                p = p + n * wp.min(-d, SDF_PUSH)
            w = (SDF_SKIN - wp.max(d, 0.0)) / SDF_SKIN
            # STANDOFF, applied over the whole band and quadratic in it, is what
            # a viewer actually sees: the plume is turned BEFORE it reaches the
            # wall, over the metre and a half a parcel's own radius spans, so a
            # 3 m wall parts a column instead of nudging it 2 cm. Killing the
            # into-surface velocity alone can only ever act on the frame of
            # contact, which is why it never reached the density splat.
            v = v + n * (SDF_REPEL * w * w * sdfw * dt)
            vn = wp.dot(v, n)
            if vn < 0.0:
                v = v - n * (vn * (1.0 + SDF_REST))
            vt = v - n * wp.dot(v, n)
            v = v - vt * wp.min(SDF_TANG * w * dt, 1.0)
            v = v + gas_noise(noise, p * 2.0) * (SDF_STIR * w * dt * sdfw)
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
            # A landed ember stays put and BURNS OUT -- fast. Parking it without
            # touching its life is what built the static carpet of glowing dots
            # across the yard: 88% of the live population was lying on the floor
            # at full radiance waiting out a three-second lifetime.
            v = wp.vec3(0.0, 0.0, 0.0)
            life = wp.min(life, age + EMBER_LAND_T)
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
        self.blast_w = 0.0                   # how hard a passing front pushes this
        self.b0 = -1.0e9                     # t0 of the burst that owns it (ember ring)
        self.wind = (0.0, 0.0, 0.0)          # ambient wind this cohort feels
        self.curl_gain = 0.0                 # extra stir as a parcel ages
        self.hub = (0.0, 0.0, 0.0)           # the column's axis, for entrainment
        self.entrain = 0.0                   # m/s^2 of outward spread
        self.ent_k = 0.0                     # ... up to radius = ent_k * height
        self.sdf_w = 0.0                     # does the yard's geometry stop it
        self.bpow = 0.0                      # 0 = one buoyancy for the cohort;
        self.bgain = 1.0                     # >0 = per-parcel skewed draw

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
    def advance(self, dt, t, blasting, child=None):
        if not self.live:
            return
        c = child if child is not None else self
        bw = self.blast_w * GAS_BLAST if blasting else 0.0
        wp.launch(step_gas, dim=self.live,
                  inputs=[self.pos, self.vel, self.st, noise_grid, dt, self.kind,
                          self.seed, *self.params[1:],
                          self.radius[0], self.radius[1], self.r_pow,
                          c.pos, c.vel, c.st, SMOKE_K, c.n,
                          SMOKE_LIFE[0], SMOKE_LIFE[1], SMOKE_R[0],
                          SMOKE_INHERIT, SMOKE_SPREAD,
                          charge_arr, charge_scale, t, bw,
                          wp.vec3(*self.wind), self.curl_gain,
                          wp.vec3(*self.hub), self.entrain, self.ent_k,
                          sdf_grid, self.sdf_w, self.bpow, self.bgain],
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
    # SENTINELLED, like the drums' array at the top of the file: now that a
    # device-side loop walks all eight slots every frame, an unfired slot of
    # zeros would read as a phantom charge at the world origin that fired at
    # t = 0, and every parcel in the yard would be kicked by it during the
    # settle. w <= -1e8 is "empty".
    _ch = np.tile(np.float32([0.0, 0.0, 0.0, -1.0e9]), (MAX_CHARGES, 1))
    charge_arr = wp.array(_ch, dtype=wp.vec4, device=device)
    # ... and each slot's yield scale, sqrt(J/1400), so a big charge pushes the
    # air harder. (The drums keep their own copy; this one is the gas's.)
    _chs = np.ones(MAX_CHARGES, np.float32)
    charge_scale = wp.array(_chs, dtype=float, device=device)
    # This frame's impact points, (x, _, z, strength). Fixed capacity, uploaded
    # only on frames that actually found a strike.
    puff_pts = wp.zeros(PUFF_MAX, dtype=wp.vec4, device=device)

    # ── the yard, as a signed-distance field ────────────────────────────────
    # Baked once from the settled layout and re-baked once per charge after its
    # front has finished rearranging things (SDF_REBAKE). A 1x1x1 dummy stands
    # in when --no-sdf is passed, because the kernel argument still has to be a
    # real array -- the branch is `sdf_w > 0`, which is a host-side scalar.
    if SDF_ON:
        sdf_d = wp.zeros((SDF_NX, SDF_NY, SDF_NZ), dtype=float, device=device)
        sdf_grid = wp.zeros((SDF_NX, SDF_NY, SDF_NZ), dtype=wp.vec4, device=device)
        sdf_bc = wp.array(HOME.astype(np.float32), dtype=wp.vec3, device=device)
        sdf_bh = wp.array(HALF, dtype=wp.vec3, device=device)
    else:
        sdf_grid = wp.zeros((1, 1, 1), dtype=wp.vec4, device=device)

    def bake_yard_sdf(centres=None):
        """(Re)build the distance field. Two launches and no host particle work.

        The re-bake is the answer to "smoke keeps breaking around a house that
        has already been blown across the yard": one body_positions() -- the arm
        path's own O(bodies) loop, on ONE frame, not in the sim path -- and the
        field is the yard as it now stands, rubble piles included.
        """
        if not SDF_ON:
            return 0.0
        t = time.perf_counter()
        if centres is not None:
            sdf_bc.assign(np.ascontiguousarray(centres, np.float32))
        wp.launch(bake_sdf, dim=(SDF_NX, SDF_NY, SDF_NZ),
                  inputs=[sdf_bc, sdf_bh, len(bodies), sdf_d], device=device)
        wp.launch(bake_sdf_grad, dim=(SDF_NX, SDF_NY, SDF_NZ),
                  inputs=[sdf_d, sdf_grid], device=device)
        wp.synchronize_device(device)
        return time.perf_counter() - t

    if SDF_ON:
        _ms = bake_yard_sdf()
        print(f"yard SDF: {SDF_NX}x{SDF_NY}x{SDF_NZ} at {SDF_CELL:.2f} m "
              f"({SDF_NX * SDF_NY * SDF_NZ / 1e3:.0f} k voxels, "
              f"{16 * SDF_NX * SDF_NY * SDF_NZ / 1e6:.1f} MB) "
              f"over {len(bodies)} bodies in {1e3 * _ms:.0f} ms")

    # per_charge: how much of a pool ONE detonation may claim. An overlapping
    # ninth burst simply overwrites the oldest slots in the ring.
    fire = Cohort("fire", FIRE_N, 0, FIRE_EMIT, FIRE_LIFE, FIRE_R, SEED + 1,
                  per_charge=FIRE_N // N_SHARES)
    smoke = Cohort("smoke", SMOKE_N, 1, 1.0, SMOKE_LIFE, SMOKE_R, SEED + 2,
                   per_charge=0)        # NEVER emitted: fire converts into it
    dust = Cohort("dust", DUST_N, 2, DUST_EMIT, DUST_LIFE, DUST_R, SEED + 3,
                  per_charge=DUST_N // N_SHARES, front_c=DUST_C)
    # THE EMBER RING. One burst, one field, its own clock -- see EMBER_N above.
    # per_charge is the WHOLE field because a field is never shared.
    EMBER_RING = [Cohort(f"ember{k}", EMBER_N, 3, EMBER_EMIT, EMBER_LIFE, EMBER_R,
                         SEED + 4 + 131 * k, skew=EMBER_SKEW, r_pow=1.5,
                         per_charge=EMBER_N)
                  for k in range(EMBER_RING_N)]
    flare = Cohort("flare", FLARE_N, 4, FLARE_EMIT, FLARE_LIFE, FLARE_R, SEED + 5,
                   skew=FLARE_SKEW, per_charge=FLARE_N // N_SHARES)
    wilson = Cohort("wilson", WILSON_N, 5, WILSON_EMIT, WILSON_LIFE, WILSON_R,
                    SEED + 6, per_charge=WILSON_N // N_SHARES, front_c=C_FRONT)
    # (v0, drag, buoy, buoy_tau, grav, curl) -- v0 is the emit kernel's, the
    # rest are the advect kernel's, in its argument order.
    fire.params = (FIRE_V0, FIRE_DRAG, FIRE_BUOY, 0.30, 0.0, FIRE_CURL)
    smoke.params = (0.0, SMOKE_DRAG, SMOKE_BUOY, 1.70, 0.45, SMOKE_CURL)
    smoke.bpow, smoke.bgain = SMOKE_BPOW, SMOKE_BGAIN
    dust.params = (DUST_V0, DUST_DRAG, 0.0, 1.0, DUST_GRAV, DUST_CURL)
    for _e in EMBER_RING:
        _e.params = (EMBER_V0, EMBER_DRAG, 0.0, 1.0, 9.81, EMBER_CURL)
    flare.params = (FLARE_V0, FLARE_DRAG, FLARE_BUOY, 0.26, 0.0, FLARE_CURL)
    wilson.params = (C_FRONT, WILSON_DRAG, WILSON_BUOY, 0.5, 0.0, WILSON_CURL)
    # How hard a PASSING front pushes each cohort. Fire and flare are zero: they
    # are born WITH the burst velocity and kicking them again just doubles the
    # fireball's expansion. Smoke takes it in full -- that is the whole point,
    # "another explosion should push any old smoke in the scene". Dust and the
    # ring take a fraction because both are already emitted ON a front and their
    # radius law is tuned; the ember is ejecta and rides the wind.
    smoke.blast_w = 1.00
    for _e in EMBER_RING:
        _e.blast_w = 0.80
    dust.blast_w = 0.45
    wilson.blast_w = 0.35
    # Ambient wind reaches the SMOKE and nothing else: it is the only cohort
    # that lives long enough to care, and blowing the ground dust or the ring
    # would only smear two things that are already tuned.
    smoke.wind = AMB_WIND
    smoke.curl_gain = SMOKE_CURL_AGE
    smoke.entrain, smoke.ent_k = SMOKE_ENT, SMOKE_ENT_K
    # What the yard's geometry stops. Fire and flare are excluded on purpose --
    # they live inside the fireball for a second, they are the densest cohorts,
    # and colliding a fireball with the house it is destroying reads wrong.
    if SDF_ON:
        smoke.sdf_w = 1.0
        dust.sdf_w = 1.0
        wilson.sdf_w = 1.0
        for _e in EMBER_RING:
            _e.sdf_w = 1.0
        # ... and fire and flare at SDF_FIRE_W. Excluding them was the single
        # biggest hole in "smoke interacts with the world", because NOTHING
        # emits smoke in this demo: a soot parcel is written at the position of
        # the fire parcel that just cooled, so a fireball that walked through a
        # roof handed the plume a birth site on the far side of it. Partial
        # rather than full because a fireball genuinely does not bounce off the
        # house it is taking apart -- it pours around it.
        fire.sdf_w = SDF_FIRE_W
        flare.sdf_w = SDF_FIRE_W
    # fire is advected FIRST and smoke LAST, so a parcel that converts this
    # frame is advected the same frame instead of hanging for one.
    cohorts = [fire, flare, dust] + EMBER_RING + [wilson, smoke]
    emitters = [fire, flare, dust, wilson]      # the ember ring emits per CHARGE

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
    if VOLUMES and SMOKE_ON:
        # The box has to contain the WHOLE plume: soot outside it is not
        # scattered and simply is not there, which reads as a cloud with its top
        # sheared off. 48 x 30 x 48 m at 192**3 is a 25 cm voxel.
        smoke.field.set_density_repr(tp.Vector3(cx, 13.0, cz),
                                     tp.Vector3(24.0, 15.0, 24.0),
                                     SMOKE_SIGMA * SMOKE_SIGMA_K * SIGMA,
                                     SMOKE_RES)
        _d = smoke.field.density_repr
        _d.albedo = tp.Color(*SMOKE_ALBEDO)
        _d.anisotropy = SMOKE_ANISO

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
        # The box is written EVERY FRAME (see density_boxes()) -- these are only
        # what it is created with. It has to follow the front, because the ring
        # travels three times the distance the old static box covered.
        wilson.field.set_density_repr(tp.Vector3(0.0, WILSON_CY, 0.0),
                                      tp.Vector3(WILSON_H_MIN, WILSON_HY,
                                                 WILSON_H_MIN),
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
    # What is lost by moving: age fade, size taper, velocity stretch and the
    # hot->cool colour ramp are ALL derived from the emitter's closed form.
    # ParticleFieldPass only writes bp.lifetime and bp.stretchOverDt for a
    # rendererOwned field, so on this one lifetime == 0, the vertex stage pins
    # ageFrac = 0, and fade_power / size_taper / stretch_seconds are silently
    # zero -- authoring them here is a no-op that invites the next reader to tune
    # nothing. THE ONLY BRIGHTNESS CHANNEL AN INTEROP BILLBOARD FIELD HAS IS THE
    # FIELD-WIDE `intensity`, and the radius is a SIZE, not a radiance (the
    # fragment falloff peaks at 1.0 at the sprite centre whatever the quad's
    # world size, so shrinking one only makes a same-brightness dot smaller).
    #
    # Which is why there are THREE of these fields, one per concurrent burst,
    # claimed round-robin exactly as the streak fields are. Then "one burst ages
    # as one population" is literally true per field and the field scalar IS the
    # per-particle dimmer -- instead of a second charge's flash re-lighting the
    # first charge's dead sparks lying on the floor thirty metres away.
    EMBER_HOT = ((1.00, 0.80, 0.40), (1.00, 0.30, 0.06))   # new ember -> old ember
    for _e in EMBER_RING:
        _e.field = make_field(EMBER_N, EMBER_R[0])
        _e.field.set_billboard_repr(tp.Color(*EMBER_HOT[0]),
                                    tp.Color(1.00, 0.17, 0.02), 0.0, 1.0)
        _b = _e.field.billboard_repr
        _b.softness = 0.32
        _b.bright_jitter = 0.65
        _b.near_fade = 1.2
        _b.glow = 0.62

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
    # STREAK_T IS QUOTED IN SIM SECONDS AND WATCHED IN FILM SECONDS. Every slot
    # of this field is alive at all times (duty 1.0) and each is reborn once per
    # `lifetime` AT THE CHARGE, so the window is not "how long a streak lives",
    # it is HOW LONG SPARKS KEEP ERUPTING FROM THE ORIGIN. At 0.85 s the film's
    # 0.16x ramp played that as four screen seconds of garden sprinkler -- which
    # is what the user saw three times. 0.22 s is 1.4 screen seconds, and because
    # the emitter's period shortens with the lifetime each slot still cycles
    # about once through the window: the DENSITY of sparks is unchanged, only the
    # duration collapses.
    # Two numbers, and conflating them is what made the old fountain: the
    # emitter's LIFETIME is how far a ray reaches (its slots' ages are spread
    # over [0, lifetime) from the very first frame, so the spray is full-depth
    # instantly -- a steady state has no build-up), and the WINDOW is how long
    # sparks keep leaving the charge. The shipped build had both at 0.85 s.
    STREAK_N = int(20_000 * GAS)
    STREAK_T = 0.22           # the window: how long the charge keeps ejecting
    STREAK_LIFE = 0.70        # the reach: how far one ray gets before it fades
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
        _e.lifetime = STREAK_LIFE
        _e.lifetime_jitter = 0.35
        _e.duty_cycle = 1.0
        _e.size = 0.05
        _e.size_jitter = 0.65
        _e.seed = SEED * 1013 + 5 + 97 * _k
        _s.set_emitter(_e)
        _s.set_emitter_time(0.0, DT)
        _s.set_live_count(0)
        scene.add(_s)
        STREAK_RING.append(_s)

    STREAK_B0 = [-1.0e9] * len(STREAK_RING)   # t0 of the burst that owns each
    fields = [c.field for c in cohorts] + STREAK_RING
    # The last thing that can still be on screen: a fire parcel emitted at the
    # end of the burst, cooling for its full life, converting, and its soot
    # living out the longest smoke life. Nothing is emitted after DUST_EMIT.
    GAS_END = FIRE_EMIT + FIRE_LIFE[1] + SMOKE_LIFE[1] + 0.1
    EMIT_END = max(c.emit_win for c in emitters + EMBER_RING)   # nothing is born after
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
        _chs[ch.slot] = ch.scale          # the advect kernel's front reads this too
        charge_scale.assign(_chs)
        for c in cohorts:
            c.emitted[ch.slot] = 0        # the slot is re-used: start its budget over
        # ... and the ring is claimed OLDEST-FIRST, never round-robin. Round
        # robin picks by fire order, so four detonations inside EMBER_FADE_T --
        # trivial with --auto-charge 0.4 or a held B key -- delete a cohort that
        # is still glowing, in one frame, while an already-dark field sits idle
        # beside it. Least-recently-claimed is always the dimmest one there is.
        k = min(range(len(STREAK_RING)), key=lambda i: STREAK_B0[i])
        STREAK_B0[k] = ch.t0
        ch.streaks = STREAK_RING[k]
        e = ch.streaks.emitter
        e.spawn_center = tp.Vector3(ch.pos[0], ch.pos[1] + 0.2, ch.pos[2])
        ch.streaks.set_emitter(e)
        # ... and its own ember field, wiped so it carries ONE burst only. The
        # fourth overlapping charge steals the oldest, which by then has been
        # dark for over a second (EMBER_FADE_T).
        ch.embers = min(EMBER_RING, key=lambda c: c.b0)
        ch.embers.reset()
        ch.embers.field.set_live_count(0)
        ch.embers.b0 = ch.t0

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
          f"+ dust {DUST_N:,} + ember {EMBER_RING_N}x{EMBER_N:,} + flare {FLARE_N:,} "
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

# --- boot the drums ---------------------------------------------------------
#
# LAST, and in this order for two reasons. The Warp module is built the first
# time anything in it launches and re-hashed whenever a kernel is added, so the
# shells' warm pass and graph capture have to come after the last @wp.kernel in
# the file. And the vertex export does not exist until the mesh has been drawn
# once, so the interop needs a render behind it -- the gas block above already
# did two, and with --no-gas there is one here.
if DRUMS:
    if NO_GAS:
        renderer.render(scene, camera)
    drum_boot()
    arm_drum_interop()
    print(f"drums: publishing {2 * N_CORNERS:,} soup vertices via {DRUM_ROUTE}")

# Billboards are composited after the upscaler and are outside auto-exposure,
# so their intensity is tied to the pinned scene exposure BY HAND, once, here.
# Change tone_mapping_exposure and these follow it instead of blowing out.
_EXP = 0.55 / max(renderer.tone_mapping_exposure, 1e-3)
# Scaled off phase 2's tuned values by the nebula's 1/sqrt(N) law: the flare
# cohort is 250 k against the old fire field's 120 k, the ember field 500 k
# against 44 k. Overlapping additive quads saturate long before they sum, which
# is why the exponent is a half and not one.
BB_FLARE = 0.038 * _EXP    # low ON PURPOSE: the FLAME is the density ramp.
BB_EMBER = 0.95 * _EXP     # The billboards are only its sparkle. (x1.7: N/3.3)
BB_STREAK = 1.6 * _EXP
FIRE_EMISSIVE = 4.5                     # emission is intensity * THIS field's sigma
_gas_live = False

# --- impact dust ------------------------------------------------------------
#
# The only host loop over bodies left in the frame, and it is a THIRD of them:
# subset k = frame % 3 is scanned, so every body is sampled at 20 Hz and the
# Python attribute walk costs a third of what it would. Velocity comes from the
# position difference over that subset's own span (`_imp_t`), which is why the
# slow-motion ramp does not confuse it -- the span shrinks with dt and the
# quotient does not. A strike is: it WAS falling faster than IMPACT_V, it has
# just lost most of that, and it is near the ground.
_imp_p = HOME.copy()                       # last sampled position per body
_imp_vy = np.zeros(len(bodies))            # ... and the fall speed it had there
_imp_t = [0.0] * IMPACT_STRIDE
_imp_k = [0]
_imp_pts = np.zeros((PUFF_MAX, 4), np.float32)
PUFFS = [0]                                # strikes raised, for report()


def impact_dust():
    """Find this frame's strikes and kick up a puff at each. Returns the count.

    Called from step_gas_frame, i.e. AFTER world.step, so the positions read here
    are the ones the renderer is about to draw.
    """
    if NO_PUFF or NO_GAS or dust.per_charge == 0:
        return 0
    k = _imp_k[0] % IMPACT_STRIDE
    _imp_k[0] += 1
    sub = bodies[k::IMPACT_STRIDE]
    p = np.array([(b.position.x, b.position.y, b.position.z) for b in sub])
    span = max(sim_time - _imp_t[k], 1.0e-4)
    _imp_t[k] = sim_time
    vy = (p[:, 1] - _imp_p[k::IMPACT_STRIDE][:, 1]) / span
    pvy = _imp_vy[k::IMPACT_STRIDE]
    hit = ((pvy < -IMPACT_V) & (vy > IMPACT_STOP * pvy) & (p[:, 1] < IMPACT_Y)
           & np.isfinite(p).all(1))
    _imp_p[k::IMPACT_STRIDE] = p
    _imp_vy[k::IMPACT_STRIDE] = vy
    n = int(hit.sum())
    if n == 0:
        return 0
    speed = -pvy[hit]
    take = np.argsort(-speed)[:PUFF_MAX]           # the hardest strikes, not the first
    q = p[hit][take]
    m = len(take)
    _imp_pts[:m, 0] = q[:, 0]
    _imp_pts[:m, 2] = q[:, 2]
    _imp_pts[:m, 3] = np.clip(speed[take] / 13.0, 0.25, 1.7)
    puff_pts.assign(_imp_pts)
    wp.launch(emit_puff, dim=(m, PUFF_P),
              inputs=[dust.pos, dust.vel, dust.st, puff_pts, dust.cursor, dust.n,
                      PUFF_P, SEED + 91, PUFF_V0, PUFF_LIFE[0], PUFF_LIFE[1],
                      DUST_R[0]],
              device=device)
    dust.cursor += m * PUFF_P                      # the same ring, the same cursor
    dust.live = min(dust.cursor, dust.n)
    PUFFS[0] += m
    return m


def front_radius(b):
    """The decelerating front's radius, m -- ONE law, used by the ring's
    emission and by the box that has to contain it."""
    return C_FRONT * WILSON_TD * (1.0 - math.exp(-max(b, 0.0) / WILSON_TD))


def smoke_fade(tw):
    """The field-wide sigma ramp that lets the plume THIN OUT instead of being
    deleted. A parcel's death is instantaneous and the splat ignores its
    radius, so without this the last third of the column blinks out while it
    still covers a third of the sky. One burst ages as one population, which is
    what makes a field-wide scalar a faithful stand-in for a per-parcel fade
    (the same argument the ember field's dimmer rests on).

    AND IT RATCHETS. `tw` is time since the NEWEST charge, so firing a second
    charge beside a seven-second-old plume used to step that plume's field-wide
    sigma from 0.32 straight back to 1.00 in a single frame -- measured, with the
    old smoke's own velocity flat across the step, i.e. nothing physical happened
    and the smoke simply got three times thicker because a bang went off 14 m
    away. There is exactly one scalar for a pool that holds several bursts, so
    the only honest invariant available is that it never RISES; the fresh burst's
    soot is then a little thinner than it would be alone, which is a fraction of
    a stop and reads as haze, not as a pop. Reset on park, with the box.
    """
    a, b = SMOKE_FADE
    x = min(max((tw - a) / (b - a), 0.0), 1.0)
    f = 1.0 - x * x * (3.0 - 2.0 * x)
    if f < _sbox["fade"]:
        _sbox["fade"] = f
    return _sbox["fade"]


_sbox = {"c": None, "h": float(SMOKE_H_REF), "n": -1, "fade": 1.0}
_sdf_baked = [-1.0e9, 0]      # (t0 of the burst being tracked, re-bakes done)


def _quant(v, step):
    """Quantise a box edge. A box that moves a hair every frame re-scatters
    every parcel into a different voxel and hands TAA new noise to chase."""
    return round(float(v) / step) * step


def density_boxes(tw):
    """Put the four density volumes ON the matter they are supposed to hold.

    All four used to be nailed down: three at the world origin and the fireball
    at whatever `--charge` said, which `--charges` never writes back. A volume
    contributes nothing outside its box -- the scatter SKIPS the particle, it
    does not clamp it -- so the condensation ring, which travels to 33 m, was
    amputated by the four flat walls of a 66 m square around the origin and
    what survived was four arcs in the corners of that square, over a floor
    whose grid rules run the same way. Hence "locked to a grid around world
    center". Nothing here allocates: center and half_extent are per-frame
    writes by design, and sigma is scaled as 1/h^2 to hold the optical depth
    the march integrates invariant while the box breathes.
    """
    live = [ch for ch in charges if sim_time >= ch.t0]
    if not live:
        return
    # ── the fireball: the NEWEST charge that still has flame ────────────────
    # Its blackbody ramp is a function of height INSIDE this box, so the box
    # floor has to be that charge's own ground or the flame colour is wrong.
    hot = [ch for ch in live if sim_time - ch.t0 <= FIRE_EMIT + FIRE_LIFE[1]]
    if hot:
        ch = max(hot, key=lambda c: c.t0)
        fire.field.density_repr.center = tp.Vector3(ch.pos[0], ch.pos[1] + 3.0,
                                                   ch.pos[2])
    # ── the condensation ring: the union of the live fronts ─────────────────
    ring = [ch for ch in live
            if sim_time - ch.t0 <= WILSON_EMIT + WILSON_LIFE[1] + 0.05]
    if ring:
        rs = [front_radius(sim_time - ch.t0) + 3.0 for ch in ring]
        x0 = min(c.pos[0] - r for c, r in zip(ring, rs))
        x1 = max(c.pos[0] + r for c, r in zip(ring, rs))
        z0 = min(c.pos[2] - r for c, r in zip(ring, rs))
        z1 = max(c.pos[2] + r for c, r in zip(ring, rs))
        h = _quant(max(0.5 * (x1 - x0), 0.5 * (z1 - z0), WILSON_H_MIN), 0.5)
        d = wilson.field.density_repr
        d.center = tp.Vector3(_quant(0.5 * (x0 + x1), 0.25), WILSON_CY,
                              _quant(0.5 * (z0 + z1), 0.25))
        # The dome's crown rides at ~h now that it is a real hemisphere, so a
        # fixed 9 m ceiling would slice its top off flat.
        vy = max(WILSON_HY, 0.66 * h)
        d.center = tp.Vector3(d.center.x, vy * 0.92, d.center.z)
        d.half_extent = tp.Vector3(h, vy, h)
        d.sigma_per_particle = WILSON_SIGMA * SIGMA * (WILSON_H_REF / h) ** 2
    else:
        # The ring is over, but its live PREFIX never shrinks, so the box stays
        # bound and its span keeps feeding the fog march's step budget for the
        # rest of the shot. Collapse it.
        wilson.field.density_repr.half_extent = tp.Vector3(WILSON_H_MIN,
                                                           WILSON_HY,
                                                           WILSON_H_MIN)
    # ── the dust: settled skirt + impact puffs, so it never shrinks ─────────
    x0 = min(c.pos[0] for c in live) - DUST_REACH
    x1 = max(c.pos[0] for c in live) + DUST_REACH
    z0 = min(c.pos[2] for c in live) - DUST_REACH
    z1 = max(c.pos[2] for c in live) + DUST_REACH
    h = _quant(min(max(0.5 * (x1 - x0), 0.5 * (z1 - z0), DUST_H_MIN),
                   DUST_H_MAX), 0.5)
    d = dust.field.density_repr
    d.center = tp.Vector3(_quant(0.5 * (x0 + x1), 0.25), 1.5,
                          _quant(0.5 * (z0 + z1), 0.25))
    d.half_extent = tp.Vector3(h, 2.6, h)
    d.sigma_per_particle = DUST_SIGMA * SIGMA * (DUST_H_REF / h) ** 2
    # ── the plume: follow it, and GROW WITH IT ──────────────────────────────
    # The counter-intuitive half. Occupancy -- parcels per voxel, the only
    # thing the splat measures -- fell 117 -> 3 between t=5 and t=11 because
    # the plume inflated eleven-fold inside a box whose voxels never changed,
    # and three parcels a voxel is not a medium, it is visible discrete
    # scatter. Shrinking the box to "concentrate" it makes that WORSE (finer
    # voxels hold fewer parcels); growing it with the plume is what holds
    # occupancy roughly constant, and the 1/h^2 sigma keeps the optical depth
    # the same while it does. The sample is 4096 strided vec4 read AFTER the
    # frame's own synchronize, so it costs a launch and a 64 kB copy, and only
    # every SMOKE_BOX_EVERY frames.
    # SEEDED FROM THE CHARGE, not from the world origin. The box below is only
    # re-measured every SMOKE_BOX_EVERY frames and only once there is smoke to
    # measure, so a charge fired 30 m out used to pour its first eight frames of
    # soot outside a box still sitting at (0, 13, 0) -- invisible, and then a
    # SNAP onto the plume. Under the 0.16x ramp those eight frames are most of a
    # screen second, and it repeats on every fresh burst because park clears it.
    if _sbox["c"] is None:
        ch = max(live, key=lambda c: c.t0)
        _sbox["c"] = np.array([ch.pos[0], ch.pos[1] + 6.0, ch.pos[2]], np.float64)
        _sbox["h"] = float(SMOKE_H_MIN)
    _sbox["n"] += 1
    if smoke.live > 8192 and _sbox["n"] % SMOKE_BOX_EVERY == 0:
        a = smoke.sample(2048)
        if a is not None:
            a = a[a[:, 3] >= 0.0]
            if len(a) > 64:
                med = np.median(a[:, :3], axis=0)
                dev = np.percentile(np.abs(a[:, :3] - med), 96.0, axis=0)
                want = min(max(SMOKE_BOX_K * float(dev.max()), SMOKE_H_MIN),
                           SMOKE_H_MAX)
                _sbox["c"] = _sbox["c"] + (med - _sbox["c"]) * SMOKE_BOX_LAG
                _sbox["h"] += (want - _sbox["h"]) * SMOKE_BOX_LAG
    if _sbox["c"] is not None:
        h = _quant(_sbox["h"], 0.5)
        c = _sbox["c"]
        d = smoke.field.density_repr
        d.center = tp.Vector3(_quant(c[0], 0.25), _quant(c[1], 0.25),
                              _quant(c[2], 0.25))
        d.half_extent = tp.Vector3(h, 0.80 * h, h)
        # sigma ~ R^2 / h^2: what the march integrates over a metre is
        # n_density * voxel_side^2 * sigma, so both the box and the resolution
        # have to appear here or --smoke-res silently changes the exposure.
        d.sigma_per_particle = (SMOKE_SIGMA * SMOKE_SIGMA_K * SIGMA
                                * (SMOKE_H_REF / h) ** 2
                                * (SMOKE_RES / SMOKE_RES_REF) ** 2
                                * smoke_fade(tw))


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
            _sbox["c"] = None           # the next burst re-acquires its own box
            _sbox["fade"] = 1.0
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
    blasting = False
    for ch in charges:
        if not ch.armed:
            continue
        b = sim_time - ch.t0
        if 0.0 <= b < GAS_BLAST_T:
            blasting = True             # a front is open: the advect loop earns its cost
        if b > EMIT_END:
            continue
        for c in emitters:
            c.emit(ch.slot, b)
        if ch.embers is not None:       # the ember ring is per CHARGE, not shared
            ch.embers.emit(ch.slot, b)
    # IMPACT DUST rides the dust cohort's ring, so it has to go in before the
    # advect launch trims itself to the live prefix -- a puff emitted after it
    # would sit still for a frame.
    if tw < IMPACT_T:
        impact_dust()
    # Smoke slots are claimed by conversion, and the mapping is static, so the
    # live prefix of the smoke pool is exactly the fire ring's prefix times K.
    smoke.live = min(fire.cursor * SMOKE_K, smoke.n)
    # The entrainment axis: the box-follow's own centroid in xz (already
    # measured, every eighth frame, for free) and the newest charge's ground as
    # the height the widening is counted from. Until there is a plume to
    # measure it is simply the charge.
    _hot = max((c for c in charges if sim_time >= c.t0), key=lambda c: c.t0,
               default=None)
    if _hot is not None:
        cxz = _sbox["c"]
        smoke.hub = ((float(cxz[0]), _hot.pos[1], float(cxz[2]))
                     if cxz is not None else tuple(_hot.pos))
    for c in cohorts:
        c.advance(dt, sim_time, blasting, child=smoke)
    wp.synchronize_device(device)
    t1 = time.perf_counter()
    for c in cohorts:
        c.publish()
    t2 = time.perf_counter()

    # THE FLASH GAIN IS THE NEWEST CHARGE'S, NOT THE SUM. bb_gain used to be
    # summed over every live charge, which is wrong twice over: two charges
    # firing together doubled a spike that is a single camera response, and an
    # OLD charge's still-live additive matter was re-lit by a NEW charge's
    # flash. The flare pool is shared, so the honest key is the newest flash --
    # a flare parcel outlives its own charge by 0.68 s at most, and within that
    # window "the newest flash" and "its own flash" are the same number.
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
        density_boxes(tw)
    # THE YARD CHANGED SHAPE. One re-bake per charge, once its front has
    # finished throwing the walls across the yard -- otherwise smoke keeps
    # breaking around a house that is no longer there. It is one
    # body_positions() (the arm path's own O(bodies) host loop) plus two
    # launches, on ONE frame, and never in the steady state.
    if SDF_ON:
        newest = max((ch.t0 for ch in charges if sim_time >= ch.t0),
                     default=-1.0e9)
        if newest > _sdf_baked[0]:
            _sdf_baked[0], _sdf_baked[1] = newest, 0
        # TWO re-bakes per charge, not one. One at 1.6 s meant the whole
        # slow-motion section of the film -- the money shot -- ran against a
        # yard that had already been thrown across itself, and after that single
        # bake nothing was ever measured again, so settling rubble never
        # appeared. 0.35 s is right after the front has passed the near
        # structures, 2.4 s is once the yard has come to rest.
        k = _sdf_baked[1]
        if k < len(SDF_REBAKE) and sim_time >= newest + SDF_REBAKE[k]:
            bake_yard_sdf(body_positions())
            _sdf_baked[1] = k + 1
    # EACH ember field ages on ITS OWN burst's clock, which is the whole reason
    # there is a ring of them: the field scalar is the only brightness channel an
    # Interop billboard field has, so one shared field means one shared clock and
    # a new charge re-lighting an old charge's dead ground sparks (measured: an
    # 8x step, on 18,000 embers already lying still, in one frame). (1 - k)^2
    # reaches ZERO at EMBER_FADE_T instead of flooring at 45% for ever.
    #
    # PHASE 5: and its flash gain is ITS OWN burst's too. The ring gave every
    # field its own (1-k)^2 clock and then multiplied it by the GLOBAL flash
    # gain, so the bug simply moved: measured, a charge fired 99 m away and
    # entirely off camera stepped a 0.65 s old, mostly-grounded cohort's
    # intensity 0.099 -> 0.364 in one frame. A spark on the ground does not care
    # what happened on the far side of the yard.
    hot, cool = EMBER_HOT
    for c in EMBER_RING:
        bo = max(sim_time - c.b0, 0.0)
        k = min(bo / EMBER_FADE_T, 1.0)
        bb = c.field.billboard_repr
        bb.color_hot = tp.Color(*(a + (b - a) * k for a, b in zip(hot, cool)))
        bb.intensity = (BB_EMBER * (1.0 + 0.6 * FLASH_BB * math.exp(-bo / 0.13))
                        * (1.0 - k) ** 2)
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
            # The ramp reaches zero exactly at STREAK_T, so the field is never
            # live-but-black (it used to bottom out 0.02 s before the window
            # closed). `b` is its OWN charge's clock, so this is already
            # per-burst and no other charge's flash reaches it.
            ch.streaks.billboard_repr.intensity = (
                BB_STREAK * (1.0 + FLASH_BB * math.exp(-b / 0.13))
                * min(max((STREAK_T - b) / (0.42 * STREAK_T), 0.0), 1.0))
    return t1 - t0, t2 - t1


# --- frame ------------------------------------------------------------------

sim_time = 0.0
prof = {"gas": 0.0, "submit": 0.0, "blast": 0.0, "physx": 0.0, "drums": 0.0,
        "render": 0.0, "n": 0}


FILM_T0 = min((ch.t0 for ch in CHARGES), default=T0)   # the ramp is keyed on the FIRST charge


def playback_speed(t):
    """Sim seconds per film second at sim time `t` -- the slow-motion ramp.

    Keyed on SIM time, not on film time, so where the ramp sits relative to the
    detonation is exact and the same in every run: the floor starts LEAD seconds
    of sim before the first charge and holds HOLD seconds after it, which at the
    shipped numbers is the flash, the fireball opening, the condensation ring
    leaving and the near wall going -- about three and a half seconds of film out
    of three quarters of a second of yard.
    """
    if NO_RAMP:
        return 1.0

    def ease(a, b, u):
        u = min(max(u, 0.0), 1.0)
        return a + (b - a) * u * u * (3.0 - 2.0 * u)

    lead = FILM_T0 - SLOW_LEAD
    if t <= lead:
        return PLAY_PRE
    if t < FILM_T0:
        return ease(PLAY_PRE, SLOWMO, (t - lead) / SLOW_LEAD)
    hold = FILM_T0 + SLOW_HOLD
    if t < hold:
        return SLOWMO
    return ease(SLOWMO, PLAY_TAIL, (t - hold) / SLOW_OUT)


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
    blast_frame(sim_time, dt)  # arm what is due, deliver the front, blow the wind
    t1 = time.perf_counter()
    world.step(dt)
    t2 = time.perf_counter()
    drums = step_drums(dt)     # the shells: their own clock, their own arrivals
    gas, submit = step_gas_frame(dt)
    prof["blast"] += t1 - t0
    prof["physx"] += t2 - t1
    prof["drums"] += drums
    prof["gas"] += gas
    prof["submit"] += submit
    prof["n"] += 1
    return gas, submit, t1 - t0, t2 - t1, drums


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
          f"gas live={live:,}  impacts={PUFFS[0]:,}")
    for name, (k, ymed, y95, rmed, r95) in gas_stats().items():
        print(f"  {name:>5}: {k:7,} live  y {ymed:5.1f}/{y95:5.1f} m  "
              f"r {rmed:5.1f}/{r95:5.1f} m  (median/p95)")
    d = drum_stats()
    if d is not None:
        print(f"  drums: {d['torn']:,}/{N_DE:,} edges torn, {d['gone']:,}/{N_DF:,} "
              f"faces gone, damage p99/max {d['dmg'][0]:.3f}/{d['dmg'][1]:.3f}, "
              f"nan={d['nan']}, "
              + "  ".join(f"#{i} h {h:4.2f} w {w:4.2f} m"
                          for i, (h, w) in enumerate(d["shells"])))
    print(f"  per frame: gas {1e3 * prof['gas'] / n:.2f} ms, publish "
          f"{1e3 * prof['submit'] / n:.2f} ms, device_copy "
          f"{1e3 * GAS_COPY[0] / n:.2f} ms ({1e3 * GAS_COPY[1] / n:.2f} issue + "
          f"{1e3 * GAS_COPY[2] / n:.2f} sync), blast {1e3 * prof['blast'] / n:.2f} ms, "
          f"physx {1e3 * prof['physx'] / n:.2f} ms, drums "
          f"{1e3 * prof['drums'] / n:.2f} ms, render "
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
        # THE RAMP. One line: the film's frame length in SIM seconds. Everything
        # else -- PhysX's accumulator, the gas advection, the shells' substep
        # size, the streak emitter's clock, the shake -- takes it from here.
        step_frame(DT * playback_speed(sim_time))
        apply_shake()
        save(os.path.join(outdir, f"f{k:05d}.png"))
        if k % 60 == 0:
            print(f"  frame {k}/{total}  t={sim_time:5.2f} "
                  f"x{playback_speed(sim_time):4.2f}  "
                  f"({time.perf_counter() - t_start:.0f}s elapsed)", flush=True)
    print(f"rendered {total} frames of film ({sim_time:.2f} s of yard) in "
          f"{time.perf_counter() - t_start:.0f}s -> {outdir}")
    report()
    ff = find_ffmpeg()
    mp4 = cli_arg("--out", "warp_explosion.mp4", str)
    if ff:
        encode_png_sequence(os.path.join(outdir, "f%05d.png"), mp4, 60, crf=17,
                            ffmpeg=ff, check=False)
        print(f"wrote {mp4}")
elif BENCH:
    # Bench the interesting state: bricks in flight, not a sleeping stack.
    # Headless on purpose (vsync off, no present), which also means each
    # render() waits on its own frame instead of pipelining -- so the render
    # column is a per-frame GPU upper bound, not the window's throughput. The
    # window prints its own live fps; that is the number to quote.
    while sim_time < T0 + 0.6:
        step_frame()
        renderer.render(scene, camera)
    prof.update(gas=0.0, submit=0.0, blast=0.0, physx=0.0, drums=0.0, render=0.0, n=0)
    GAS_COPY[:] = [0.0, 0.0, 0.0]
    live = sum(f.live_count for f in fields)
    bench_loop(step_frame, lambda: renderer.render(scene, camera),
               ("gas", "submit", "blast", "physx", "drums"), 20, 200,
               f"{len(bodies)} bodies + {live:,} live particles + {N_DF:,} shell "
               f"faces (headless, serialized)")
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
