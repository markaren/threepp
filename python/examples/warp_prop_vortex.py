"""A ship's propeller CAVITATING underwater -- NVIDIA Warp sim, Vulkan rendering.

warp_nebula_vk.py proved the thesis on a cloud: particles carry the IMAGE, the
field's own density volume carries the LIGHT TRANSPORT. A nebula is the easy
case for it, because a cloud has no shape you can be wrong about. This is the
hard case, and the one the feature was built for: a five-bladed controllable-
pitch propeller on a ship's shaft three metres down, shedding tip vortices into
the classic interleaved helices -- and, when it is pushed hard enough, filling
their cores with VAPOUR.

THIS DEMO USED TO BE A PROP IN AIR, and git keeps that version (5fb670d3 and
back). Every structure below was built there and none of it changed in kind:
the closed form, the two legs, the helm ring, the volume. What changed is the
MEDIUM, and with it the story. Water carries no tracer. A propeller turning
quietly under a hull is very nearly invisible -- the wake is water in water and
the photographs of it are black. What you DO see, and the reason every naval
architect knows the shape of a propeller wake by heart, is CAVITATION: the tip
cores drop below the vapour pressure of water and BOIL, and the vortex draws
itself in white. That is not a look laid over the wake. It is a THRESHOLD, and
crossing it is now what the two levers are for.

WHY IT NEEDS THE VOLUME. A helix is a curve that passes IN FRONT OF ITSELF.
Additive sprites cannot say which loop is nearer -- every crossing is the same
sum from either side -- so a flat render of this is a tangle of uniform glowing
wire. With the two marches on, the near loop occludes the far one (T_cam) and
the bubbly wake shadows its own far side (T_sun), and the same particles
resolve into vapour ropes wound around a lit column of bubbles. --flat at the
same timestamp is the whole acceptance test.

    pip install warp-lang
    python warp_prop_vortex.py                   # window; PUSH EITHER SLIDER
    python warp_prop_vortex.py --shot 5.5        # 120 rpm: JUST BELOW inception
    python warp_prop_vortex.py --shot 5.5 --rps 3    # 180 rpm: ropes that END
    python warp_prop_vortex.py --shot 5.5 --rps 5    # 300 rpm: the whole wake boils
    python warp_prop_vortex.py --shot 5.5 --pitch 35 # same gate, other lever
    python warp_prop_vortex.py --shot 5.5 --flat # the SAME frame, knobs at 0
    python warp_prop_vortex.py --depth 1.2       # shallower: it cavitates sooner
    python warp_prop_vortex.py --view mouth      # across the disc: water IN, ropes OUT
    python warp_prop_vortex.py --view prop       # the bronze, close
    python warp_prop_vortex.py --pitch 8         # feathered: a nearly dead wake
    python warp_prop_vortex.py --shot 5.0 --rps 5 --rps-to 2 --rps-at 4.6
                                                 # the throttle's step, 0.4 s later
    python warp_prop_vortex.py --n 5000000       # more particles: a grainless far wake
    python warp_prop_vortex.py --bench           # frame time, knobs on vs --flat

    NOTE the 5.5 s: the wake is 2.4 s deep and the shaft takes 2 s to spin up,
    so anything before ~4.5 s is still showing the ramp in its far half. The
    air version's 4.0 s was right for a 0.85 s wake and is now too early.

CAVITATION IS A THRESHOLD AND THE DEMO IS TUNED TO SIT UNDER IT. At the default
helm -- 120 rpm, beta 22 deg, 3 m down -- the tip cores reach 83% of the
available pressure margin and NOTHING happens: the water is clear, the wake is
a faint bubbly column, and the helices are barely a hint. Push either slider
and the ropes switch on, because the core suction goes as (L omega R)^2 and the
margin does not move. That is the beat. It is the same beat a ship's engineer
lives with: cavitation is not a setting, it is an operating point you cross.

    ca      = K_CAV L(beta)^2 (omega R_tip)^2      the tip core's suction, m2/s2
    margin  = (p_atm + rho g DEPTH - p_vapour)/rho the water's own resistance
    e       = max(0, ca / margin - 1)              the EXCESS, per parcel

Both are evaluated AT BIRTH, from the helm ring, so a parcel that was shed
cavitating stays cavitating as it convects and a parcel shed clear stays clear.
The excess drives three things and they are deliberately different curves:
INTENSITY saturates (1 - exp(-e/0.45)), CORE RADIUS grows with it, and visible
rope LENGTH grows with it -- see COLLAPSE below.

    rho     1025 kg/m3      seawater
    DEPTH   3 m, --depth    submergence of the shaft centreline. THE HONEST
            LEVER: static head is most of the margin, so moving the prop up
            moves the inception point down. --depth 1.2 cavitates at the
            default helm; --depth 12 will not cavitate at 300 rpm.

COLLAPSE, NOT DIFFUSION, is the other thing the water changed. In air the far
wake DIFFUSED: the core burst, the grain ballooned, the rope dissolved into
smoke. Vapour does not dissolve. It CONDENSES, the instant the core pressure
recovers downstream, and it does so over a few centimetres -- the rope simply
STOPS. So the tip population's visible length is an age gate scaled by the
excess (a small excess lives a fraction of a loop and ends; a large one runs
the whole frame), with a short smoothstep at the end so the tail is a collapse
and not a fade. The turbulence still wiggles the cores while they live.

THE HUB VORTEX is the single most recognisable feature of real CPP footage, and
it cavitates FIRST: the flow behind the cap is slower, the pressure lower, and
the head start is worth about 12% here (HUB_CAV). So at the default the hub
core sits at 93% of margin against the tips' 83%, and a nudge on either slider
lights the axial rope a few rpm before the helices. A small share of slots is
taken out of the sheet's budget and born on the axis, aft of the cap.

IT IS A CONTROLLABLE-PITCH PROPELLER, and that is the demo's second thesis.
The first (below) is that particles carry the image and a volume carries the
light. The second is that a CLOSED-FORM wake can still be INTERACTIVE and still
be honest about causality. Drag the slider: five bronze blades turn in their
flanges at once, and the wake does NOT change at once, because the wake is made
of parcels that were shed under whatever pitch was set when each of them
crossed the disc. The boundary between the old slipstream and the new one
convects away downstream at the slipstream's own speed and the old wake
persists until it ages out, two and a half seconds later. That lag is the whole
point -- and now the boundary is also a boundary in CAVITATION NUMBER, so a
throttle step lights the new wake and leaves the old one dark.

    beta   the blade angle at 0.7 R, 0..35 deg, BETA_DESIGN = 22 the neutral
    L      the loading it makes: smoothstep(sin(beta - 2 deg)), 0 at feather
           and 1 at 32 deg, and the ONLY channel from the pitch to the flow
           -- to the wake speed as sqrt(L), and to the cavitation number as
           L^2, which is why the pitch lever alone can cross inception
    omega  the shaft rate, 60..600 rpm, RPS_REF = 6 rev/s the reference.
           Tip speed 11 m/s at the default 120 rpm and 34 at 360, and it is
           the SQUARE of that number that decides whether the wake is white
    hist   HIST_N vec3 (L, omega, theta), one per frame, one full slot period
           deep. Written on the host at frame_no % HIST_N, read in the kernel
           at the parcel's OWN crossing frame. That is the whole added state.

    u ~ omega sqrt(L)  induced velocity. Static-thrust momentum theory, which
                  does not care what the fluid is: at a fixed pitch T ~ omega^2
                  so v_i ~ omega, and the sqrt is spent on the loading. Wake
                  speed, funnel length and (through the distance-keyed flare)
                  funnel WIDTH all follow, so a feathered OR idling prop's
                  funnel collapses onto the disc. The ANCHOR is water: the
                  default helm makes a 2.0 m/s slipstream, which is what a
                  ship at the bollard actually throws.
    swirl ~ omega L    circulation is linear in both
    ca ~ (L omega)^2   the cavitation number, and the reason both levers are
                  so much sharper here than they were in air

WHAT THE THROTTLE DOES NOT DO IS THE INTERESTING PART. Advance per revolution
is u / n, and with u ~ omega that is INDEPENDENT OF RPM: change the shaft speed
and the helix's loop spacing in metres does not move at all. What moves is how
far 2.4 s of wake reaches (halve the rpm and the slipstream is half as long),
how fast the loops are laid down, and -- only in water -- whether they are
VISIBLE at all. Loop SPACING is a property of the PITCH alone. That is what a
screw is: it advances its geometric pitch per turn whatever speed you turn it
at, and the rpm-step crops show exactly that -- a short stub of new wake whose
loops line up with the old wake's, a rarefaction between them, and the old wake
convecting off to age out.

THETA CANNOT BE A CLOSED FORM ONCE OMEGA IS A LEVER. theta_of(t) = OMEGA t^2 /
2 T_RAMP was exact while omega was a constant; the moment it is a slider, any
azimuth derived from the CURRENT omega is retroactive, and dragging the
throttle re-places every filament in four metres of existing wake at the
azimuth it would have had if the shaft had always been at the new speed. The
helix twists bodily off its blade tips and the credibility detail dies. So
theta is integrated on the host once per frame (trapezoid, which is exact for
the linear spin-up ramp and so reproduces the old closed form bit for bit) and
carried in the ring; the mesh and the kernel read the same number.

THE ONE ASYMMETRY IS DELIBERATE: the wake reads history and the funnel does
not. A wake parcel's crossing time is in the past, so the ring has it. An
inflow parcel's is in the FUTURE, and rather than invent one the kernel clamps
to now -- which is the physics, not a fudge, because the flow ahead of a disc
is a pressure field and a pressure field is not convected. Pull the lever and
the funnel collapses in one frame while a metre of old slipstream is still
spiralling away under the pitch that made it. Those two rates differing on
screen is what a CPP looks like.

A HARD STEP TEARS THE SLIPSTREAM, and the black gap that opens in the
--pitch-at shots is the model being right rather than the model breaking. Past
the disc there is no force on a parcel, so it keeps the speed it was shed with
for the rest of its life: drop the throttle and the fast slug already
downstream keeps going at 5.0 m/s while the flow behind it leaves at 2.0, and
the two separate. What fills the tear is ambient water that never went through
the disc, which carries no bubbles and no vapour and is therefore black -- an
easier thing to believe underwater than it ever was in a room. A servo takes
seconds to
swing and a slider drag takes a human moment, so both stretch that tear into a
visible rarefaction instead of a void; only the scripted instantaneous step
opens it fully, and it is the clearest single frame of the causal claim.

A PROPELLER PULLS, and for a long time this one did not look like it: every
parcel was born ON the disc, so the water in front of the propeller was inert
and the whole thrust of the picture began at the blade. It does not, in a real
one. The disc draws its water from a streamtube WIDER THAN ITSELF, and
underwater there is something to SEE it with that a wind tunnel has to inject:
SUSPENDED SEDIMENT. Silt and plankton are in the water already, they are lit by
the same downwelling sun, and every diver's video of a working screw shows them
streaming in from well outside the tip radius, accelerating as they converge,
necking down through the disc and only THEN spiralling away. So each slot's
life has two legs, an INFLOW leg and the wake, joined at the disc. What arrives
is PLAIN WATER carrying motes, and it has none of the wake's identity -- no
swirl, no helix, no blade phase, no tight core -- because the blade is what
makes those, at the instant it cuts. --view mouth is the framing that shows it:
the mouth twice the disc's diameter, the disc plane splitting the frame,
converging motes one side and vapour the other.

THE SIM IS A CLOSED FORM, like the renderer's own emitter and unlike the
nebula's integrator: slot s is reborn every `period` seconds and its whole
state is f(seed, slot, t). Nothing accumulates, so --shot 4.0 is 240 frames of
evaluation rather than 240 frames of integration, and the wake is the same wake
whether you seek to it or walk to it. The inflow leg had to keep that: it is
walked BACKWARD from the crossing the wake leg already defines, so it costs no
state, no integration and no loss of seeking.

    phi   the slot's phase in [0, T_IN + LIFETIME); the disc is at T_IN
    tb    the CROSSING time -- past for the wake, future for the funnel
   -- upstream, phi < T_IN, w = T_IN - phi still to run --
    x     -d(w), slope pinned to the disc velocity at the crossing and falling
          to a fifth of it far upstream: slow drift, then the rush in
    r      r0 / sqrt(u(d)/u_disc), mass conservation through the actuator
          disc's own induced velocity -- past 2x the crossing radius at the
          mouth, and per-slot hashed so the funnel has body
    phi_a  one of a fixed rake of streamline azimuths, STATIC in the world:
          no omega, no blade offset, nothing that turns. It converges onto
          the crossing azimuth only in the last few centimetres
   -- downstream, phi >= T_IN, tau = phi - T_IN of age --
    phi_a  theta(tb) + blade * 2pi/B + swirl(tau)  -- swirl DECAYS with age
    x     U_wake * tau, eased from the disc velocity over the first ~0.1 s
    r     r0 contracting toward 0.78 r0 (slipstream contraction)
    w     a tight vapour tube while the parcel cavitates, widening with the
          excess; the burst-and-diffuse core is what a CLEAR parcel does
    vap   (1 - exp(-e/0.45)) x a collapse gate on age -- the rope ENDS
    +     a curl-noise displacement growing as tau^1.9, sampled at the
          Lagrangian LABEL rather than at the position, so the wobble convects
          WITH the filament instead of standing still in the world while the
          wake slides through it.

POSITION IS CONTINUOUS AT THE CROSSING AND IDENTITY DELIBERATELY IS NOT. The
axial speed matches (d's slope is pinned to u_disc), the radius matches and the
azimuth matches, so nothing pops or teleports; but the parcel arrives as a fat,
dim mote of silt in plain water and leaves as a 1 cm vapour filament welded to
a blade tip. That step IS the propeller. Read the frame left to right and the
blade is unmistakably the thing that caused the wake, which is the entire point
of drawing the water in front of it at all.

THE CREDIBILITY DETAIL is theta(t): the visible propeller's rotation and the
kernel's birth azimuth are the same function of the same clock, so every
filament is rooted at a blade tip and stays welded to it. Get that wrong by one
frame and the whole thing reads as a bubble curtain behind a fan.

THE STAGE IS THE ENGINE'S OWN and no new rendering was written for it: a deep
blue-green background, `set_fog_water_surface_y(DEPTH)` to put the waterline
three metres up, and `set_underwater_murk` for the column's extinction and
in-scatter, so distance falls into murk exactly as it does in the sailboat and
hull films. The sun is DOWNWELLING -- from above, blue-shifted and dimmed by
three metres of water -- which is what gives the bubble column a bright top and
a dark belly, and it stays brighter than the rake because the billboards take
the single brightest DirectionalLight as their sun.

BillboardRepr::stretchSeconds is NOT used, and not for lack of trying: the
backend gates the streak on `rendererOwned || hostPrevIsPrevStep`, and an
Interop field is neither, so it is a hard no-op here (ParticleFieldPass.cpp).
The rope's thickness comes from the per-particle tube offset instead, which is
where it belongs -- a vortex core is a tube, not a smeared point.

DETERMINISM is forfeit on the interop leg exactly as it is for the nebula (the
bytes are authored past the import); the sim clock is nevertheless the frame
index times DT and never a wall clock, so a shot is reproducible modulo the
import. --no-interop falls back to the host ring at a fraction of the count.
"""
import math
import os
import sys
import time

# Make the built `threepp` module (in the parent python/ dir) importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warp as wp

import threepp as tp
from warp_common import cli_arg

# 2M -> 1.7M, and it is the 25 fps floor that moved it, not taste. Fixing the
# negative-age bug (see the kernel) put a sixth of the parcels BACK into the
# frame that the rasteriser had been silently discarding as NaN, which is 5.6
# ms of work that was always supposed to be there; the underwater medium is
# another 3. Both are honest, neither is optional, and the count is the dial
# that pays for them. Nothing in the look moves with it: BRIGHT and SIGMA are
# both anchored at 2M and both go as 1/N, so total radiance and total optical
# depth are invariant and the only thing 1.7M buys less of is smoothness. See
# THE N CONTRACT below.
N = cli_arg("--n", 1_700_000, int)
BENCH = "--bench" in sys.argv
SHOT = "--shot" in sys.argv
SHOT_TIME = cli_arg("--shot", 4.0, float)
FLAT = "--flat" in sys.argv
INTEROP = "--no-interop" not in sys.argv
W, H = cli_arg("--width", 1280, int), cli_arg("--height", 800, int)
HEADLESS = SHOT or BENCH
FPS = 60
DT = 1.0 / FPS

# ── The propeller ───────────────────────────────────────────────────────────
# Axis is +X: the disc lies in YZ at x = 0 and the wake blows toward +X. A
# point at (0, r, 0) rotated about X by theta lands at (0, r cos, r sin), which
# is exactly the kernel's (y, z) -- so the mesh's rotation.x and the kernel's
# azimuth are the same number with no sign to get wrong.
BLADES = cli_arg("--blades", 5, int)
R_TIP = 0.90                        # m
R_HUB = 0.27       # the bulbous hub: a 5-blade CPP carries the whole pitch
                   # mechanism inside it, so hub/diameter is ~0.30, not 0.15
R_PALM = 0.245     # radius at which a blade root meets its own flange
# ── THE WATER, AND THE PRESSURE IT HAS TO GIVE UP ───────────────────────────
# Everything cavitation-related is here, and there is very little of it: three
# fluid constants, one geometric one, and the margin they make. That margin is
# a SPECIFIC ENERGY (m2/s2 = J/kg), because dividing the pressure by rho is
# what lets it be compared with a velocity squared without carrying rho into
# every term of the kernel.
RHO = 1025.0            # kg/m3, seawater
P_ATM = 101325.0        # Pa at the surface
P_VAP = 2340.0          # Pa, the vapour pressure of water at 20 C
G_ACC = 9.81
# DEPTH IS THE HONEST LEVER and the reason it is a CLI knob rather than a
# constant. Static head is 23% of the margin at 3 m and it is LINEAR in depth,
# while the suction that has to beat it is quadratic in tip speed -- so moving
# the propeller up moves the inception point down, hard. --depth 1.2 cavitates
# at the default helm; --depth 12 does not cavitate at 300 rpm. A real ship
# meets this as a loaded/ballasted difference and as the reason a propeller
# racing in a swell howls.
DEPTH = cli_arg("--depth", 3.0, float)      # m of water over the shaft
CAV_MARGIN = (P_ATM + RHO * G_ACC * DEPTH - P_VAP) / RHO    # m2/s2
# K_CAV IS SET BY THE DEMO'S BEAT, and that is a legitimate way to fix a
# coefficient this crude: the model says the tip-core suction goes as the
# circulation SQUARED (Gamma ~ L omega R for a lifting section), so
# ca = K_CAV (L omega R)^2, and K_CAV is the one number that says how tight the
# core is. Rather than invent a core radius, it is chosen so the DEFAULT helm
# -- 120 rpm, beta 22, 3 m down -- lands at 0.83 of the margin. Just under.
# Everything the demo is for happens in the last 17%.
K_CAV = cli_arg("--k-cav", 1.40, float)
# The hub vortex's head start. Behind the cap the axial flow is slower and the
# static pressure lower, so the hub core reaches vapour pressure BEFORE the tips
# do -- which is why the axial rope is the first thing to appear in real CPP
# footage and the last thing to go. 1.12 puts the hub at 0.93 of margin at the
# default against the tips' 0.83, so a few rpm separate the two inceptions.
HUB_CAV = 1.12
CAV_KNEE = 0.45      # excess at which the vapour is 63% of its full brightness
CAV_R = 2.6          # x core radius at full vapour: a fat rope, not a wire
# How long the vapour SURVIVES, as a fraction of LIFETIME, against the excess.
# See COLLAPSE in the docstring: this is the length knob and it is deliberately
# not the intensity knob, and it is the LEVER'S OWN RANGE that sets the slope.
# 0.50 saturated at e ~ 1.7 -- 230 rpm -- which put every interesting helm
# setting at "the rope runs off the frame" and made the collapse invisible at
# anything you would actually want to look at. 0.30 saturates at e ~ 2.9
# (about 245 rpm at the design pitch) and spreads the useful part over the
# 130-240 rpm band, where a rope that ENDS is the whole point.
CAV_L0, CAV_LK = 0.14, 0.30
CAV_COLLAPSE = 0.30  # the last 30% of that life is the condensation ramp
# ── CONTROLLABLE SHAFT SPEED ────────────────────────────────────────────────
# RPS_REF is the ANCHOR and not the default: every flow coupling below is
# normalised by it, so a prop turning at RPS_REF with beta at BETA_DESIGN
# evaluates to exactly the numbers this demo had before either lever existed.
# RPS is merely where the throttle starts.
RPS_REF = 6.0                       # rev/s, the tuning reference
# 2 rev/s = 120 rpm is where the DEMO starts, and it is an honest ship speed:
# a large slow-speed diesel turns its screw at 60-120 rpm and this is the top
# of that. It is also, by construction, the operating point that sits just
# under inception -- see K_CAV.
RPS = cli_arg("--rps", 2.0, float)  # revolutions per second commanded
OMEGA = 2.0 * math.pi * RPS
OMEGA_REF = 2.0 * math.pi * RPS_REF
RPM_MIN, RPM_MAX = 60.0, 600.0      # the slider's range
RPS_TO = cli_arg("--rps-to", RPS, float)      # headless: the step it takes
RPS_AT = cli_arg("--rps-at", 1.0e9, float)    # ... at this sim time
T_RAMP = cli_arg("--ramp", 2.0, float)   # spin-up, seconds
# The spin-up is no longer a special case in the kernel: it is simply the first
# two seconds of the SCHEDULE, multiplying whatever speed is commanded. That is
# what keeps the ring defined at t = 0 with no separate "before the prop
# started" branch, and it is why the slider is also ramped for its first two
# seconds rather than snapping the wake into existence on frame one.
OM_G_FLOOR = 0.04    # radiance left at a barely-turning shaft


def ramp_frac(t):
    return min(max(t / T_RAMP, 0.0), 1.0) if T_RAMP > 0.0 else 1.0


def scheduled_rps(t):
    """Headless shaft speed: hold RPS, step to RPS_TO at RPS_AT, both under the
    spin-up ramp. A hard step for the same reason the pitch one is hard."""
    return (RPS_TO if t >= RPS_AT else RPS) * ramp_frac(t)

# ── CONTROLLABLE PITCH ──────────────────────────────────────────────────────
# The blades turn in their flanges and the wake has to answer -- see the
# docstring. BETA_DESIGN is the angle the loft is AUTHORED at, so the pivot
# groups sit at zero rotation there and the designed twist is what you see.
BETA_DESIGN = 22.0                  # deg, at 0.7 R
BETA_MIN, BETA_MAX = 0.0, 35.0      # the slider's range
# The loading curve. L is the fraction of design thrust the blade is making,
# and it is the ONLY thing the flow reads: a smoothstep of sin(beta - beta0),
# zero at the feathered angle and saturating at the design-plus angle.
BETA_0 = 2.0                        # deg: feathered, no lift, no wake
BETA_1 = 32.0                       # deg: full loading
# Floors, and they earn their keep. U_FLOOR keeps a feathered prop's wake
# CREEPING rather than stacking every parcel of every slot on the disc plane
# in one bright annulus, which is what a hard zero does. G_FLOOR is the
# matching brightness floor -- and it has to be well BELOW the speed floor,
# because parcels per unit length goes as 1/u: at L = 0 the line density is
# 5.8x and the per-parcel radiance 0.05x, so the wake's flux per metre lands
# at a quarter of design. That is the number that reads as "near-dead".
U_FLOOR = 0.03
G_FLOOR = 0.05
IN_G_FLOOR = 0.06
PITCH = cli_arg("--pitch", BETA_DESIGN, float)      # headless: the pitch held
PITCH_TO = cli_arg("--pitch-to", PITCH, float)      # ... and the step it takes
PITCH_AT = cli_arg("--pitch-at", 1.0e9, float)      # ... at this sim time


def loading(beta_deg):
    """L(beta): 0 at the feathered angle, 1 at design-plus. Momentum theory
    only needs a monotone, smooth, saturating shape; sin(alpha) is the lift
    slope's own argument and the smoothstep takes the corners off both ends."""
    x = math.sin(math.radians(max(beta_deg - BETA_0, 0.0))) \
        / math.sin(math.radians(BETA_1 - BETA_0))
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


# THE FLOW IS SCALED RELATIVE TO DESIGN, NOT TO L. L(BETA_DESIGN) is 0.76, not
# 1 -- 22 deg is the designed cruise setting and there is deliberately more
# pitch available above it. If the kernel used L raw, the DEFAULT wake would
# come out 12% slower and 23% dimmer than the wake this demo spent four
# revisions tuning, and every crop against the pre-pitch baseline would be
# comparing two different pictures. So every flow term is divided by its own
# value at the design point: at beta = BETA_DESIGN the kernel evaluates to
# exactly the numbers it evaluated to before pitch existed, and the slider
# reads as a departure from design in both directions.
USC_DESIGN = math.sqrt(U_FLOOR + (1.0 - U_FLOOR) * loading(BETA_DESIGN))
GAIN_DESIGN = G_FLOOR + (1.0 - G_FLOOR) * loading(BETA_DESIGN)
IN_GAIN_DESIGN = IN_G_FLOOR + (1.0 - IN_G_FLOOR) * loading(BETA_DESIGN)
SWIRL_DESIGN = loading(BETA_DESIGN)


def scheduled_beta(t):
    """Headless pitch: hold PITCH, step to PITCH_TO at PITCH_AT. A hard step
    and not a servo ramp, deliberately -- the acceptance evidence is the
    BOUNDARY between old and new wake convecting downstream, and a ramp smears
    the one edge the shots exist to show."""
    return PITCH_TO if t >= PITCH_AT else PITCH

# ── The wake, CALIBRATED FOR WATER ──────────────────────────────────────────
# The anchors are quoted at RPS_REF (360 rpm) as everything else is, and they
# are set by the OPERATING point: 6.0 m/s at the reference is 2.0 m/s at the
# default 120 rpm, which is the slipstream a large screw actually throws at the
# bollard. In air the same numbers were 5.6 / 3.6 and the default was the
# reference speed; the ratio between disc and developed slipstream is unchanged
# because momentum theory does not care what the fluid is.
#
# THE LIFETIME HAD TO GROW WITH THE CALIBRATION, and this is the one change
# that is arithmetic rather than judgement. Reach is u x LIFETIME. At 2.0 m/s
# the old 0.85 s reached 1.7 m, which in this framing is a stub against an
# empty frame; 2.4 s puts it back at 4.8 m, within a hand's width of the reach
# the air version had at its own default, so the crops stay comparable and the
# box barely moves. It is also free: the same N over a proportionally longer
# period leaves parcels-per-metre where it was, which is what BRIGHT and SIGMA
# are actually tuned against. The helm ring is sized from PERIOD, so it grew
# with it and nothing else had to be told.
LIFETIME = 2.4           # s of wake held on screen; also the slot recycle period
U_DISC = 3.86            # axial velocity AT the disc (m/s) at RPS_REF
U_WAKE = 6.0             # fully developed slipstream at RPS_REF; ~2x inflow
TAU_A = 0.13             # s over which the flow eases from U_DISC to U_WAKE
CONTRACT = 0.78          # slipstream contracts to this fraction of its birth radius
TAU_C = 0.16             # s of that contraction
SWIRL_F = 0.22           # self-induced rotation of the helix, as a fraction of OMEGA
TAU_S = 0.40             # s over which the swirl decays away
W_CORE0 = 0.010          # vortex core radius at shedding (m)
# ── Core growth: BURST, not sqrt ────────────────────────────────────────────
# Lamb-Oseen's w = sqrt(w0^2 + 2 nu_t tau) was the first thing tried and it is
# the wrong law for this picture, for a reason worth writing down: a sqrt
# spends most of its growth in the FIRST instants, so any nu_t large enough to
# fog the far wake has already fogged the near one and there is no rope left to
# look at. A tip vortex does not do that. It holds a tight, coherent core while
# it is stable and then BURSTS when the helical instability catches up with it,
# which is a delayed growth -- and that is both what the photographs show and
# what makes the near/far split of this demo exist at all.
W_BURST = cli_arg("--burst", 0.20, float)   # core radius added by end of life (m)
W_BURST_P = 2.4                             # its delay: flat, then a knee
TURB = cli_arg("--turb", 0.30, float)    # m of curl displacement at end of life
TURB_P = 1.9             # its growth exponent in age -- tight at the tip, wild downstream
TURB_FREQ = 0.55         # label-space wavelength of the meander
# ── WHAT A TIP PARCEL LOOKS LIKE, WHICH IN WATER IS ALMOST NOTHING ──────────
# This is the single biggest departure from the air version and it took some
# nerve. In air the tip population was the STAR: a bright cyan-white filament
# from the moment it left the blade. In water there is no tracer, and a
# non-cavitating tip vortex is a pressure minimum in clear water -- it is not
# dark, it is INVISIBLE. Real footage of a lightly loaded screw shows the hub
# and a faint bubbly haze and nothing else at all.
#
# TIP_HINT is what is left: 9% of the old radiance, enough that the helix is a
# suggestion at the default helm rather than an absence (the geometry is still
# the demo's whole thesis, and a completely empty frame proves nothing about
# it), and far too little to compete with the bubble column. Everything above
# that comes from the vapour, which is gated.
#
# BOTH ARE RELATIVE TO THE RE-ANCHORED BRIGHT (see OMG_OP), so 1.0 is the
# radiance the air version's tip rope had at ITS default. A full vapour core is
# a little brighter than that and a clear one is a tenth of it, which is the
# whole dynamic range this demo works in. The first cut had VAP_GAIN at 2.3 --
# set before the re-anchor and therefore 5x over -- and the 300 rpm frame came
# out as SAND: each parcel bright enough to be an individual visible dot, which
# no amount of grain growth fixes because the problem is contrast per sample.
TIP_HINT = cli_arg("--tip-hint", 0.09, float)
VAP_GAIN = cli_arg("--vapour", 0.85, float)  # radiance of a fully vapour-filled core
# The vapour rope's grain does NOT balloon with AGE the way the air wake's did:
# growth in age was the diffusion story and vapour does not diffuse. What it
# does do is grow with the EXCESS, in step with the core it is sampling --
# VAP_R0 beside CAV_R -- and that turned out to be the fix for the sand. A
# cavitating core is a centimetres-thick TUBE; sampling it with the same 2.6 mm
# grain the clear-water hint uses leaves a fat rope drawn as a scatter of
# individual bright dots, which is exactly how the 300 rpm frame first came
# out. Widen the grain with the tube and neighbouring samples OVERLAP, which is
# the only thing that ever makes a particle wake smooth.
VAP_R0 = 2.2             # x grain radius at full vapour, beside CAV_R's 2.6
VAP_GROW = 5.0           # x more, over the rope's life
# ── The entrained SHEET, WHICH IS NOW THE BUBBLY WAKE ───────────────────────
# It began as a supporting structure -- the vortex sheet rolling up behind the
# blade, there to shear against the tip helix and start the braiding -- and it
# turns out to be the thing that decides whether the slipstream has an INSIDE.
# The tip filaments are ropes on the tube's surface and nothing else fills the
# volume they bound, so at SHEET 0.30 with a 0.42 radiance penalty the near
# wake was a wire cage with black between the loops however many parcels were
# thrown at it. Raised here, and the penalty relaxed, because the sheet is the
# only population that lives BETWEEN the ropes.
#
# UNDERWATER IT KEEPS ITS JOB AND CHANGES ITS NAME. The machinery is untouched;
# what it draws is the entrained air and the turbulent microbubble mist a
# working screw drags into its own slipstream, which is the one part of a
# propeller wake that is visible whether or not anything is cavitating. It is
# WHITE rather than blue-grey (bubbles are broadband scatterers), it scales
# with the loading as the sheet always did, and it takes a BOOST when the tips
# are cavitating, because collapsing vapour is itself a bubble source. That
# boost is the reason the whole column brightens on the same slider push that
# lights the ropes, instead of the ropes appearing over a dead wake.
SHEET = cli_arg("--sheet", 0.46, float)  # fraction of slots born across the SPAN
# 0.62 -> 0.40, and the reason is the DEMO'S BEAT rather than the physics. At
# the default helm nothing is cavitating, so the bubble column is the entire
# picture, and at the air version's level it filled the frame with white and
# said "look at all this" -- which is the opposite of what shot A has to say.
# Clear water first, then the switch. What it loses at the default it takes
# back through BUB_CAV the moment the tips start boiling.
SHEET_GAIN = cli_arg("--sheet-gain", 0.40, float)   # its radiance vs a tip rope
BUB_CAV = cli_arg("--bubble-cav", 1.35, float)   # x radiance at full cavitation
# ── The HUB VORTEX, taken out of the sheet's own budget ─────────────────────
# A small population born on the AXIS, aft of the cap, and it is a separate
# population rather than a special case of the sheet for one reason: it reads
# the cavitation gate at HUB_CAV, so it lights first. HUB_X0 puts its birth
# station behind the dome (which ends at x = 0.435) so the rope emerges from
# the cap rather than out of the middle of the hub, and its slots skip the
# inflow leg entirely -- there is nothing upstream of a hub vortex.
HUB_SHARE = cli_arg("--hub-share", 0.05, float)  # fraction of slots, from SHEET's
HUB_X0 = 0.46            # m aft of the disc: just clear of the cap dome
HUB_R = 0.055            # m, the core's own radius -- fatter than a tip's
HUB_W = 2.2              # x the tip's vapour core width at the same excess
HUB_GAIN = 1.7           # x a tip rope's radiance: a fatter core holds more vapour
# The sheet's inner station. It used to start at 0.26 R -- just outside the old
# slim hub -- which left the tube's own axis empty; a side view then showed a
# hollow core wherever the near and far walls did not overlap. A real prop does
# put a wake behind its hub (the hub vortex is a genuine structure), so the
# inner stations now reach in to the hub's own radius and the column is solid.
SHEET_R0 = cli_arg("--sheet-r0", 0.155, float)      # x R_TIP
# ── The INFLOW: a prop drinks from a tube WIDER than its own disc ────────────
# Every parcel used to be BORN at the disc, which made the air in front of the
# propeller inert -- and a propeller's most legible single fact is that it is
# pulling. So each slot's closed form gains a leg BEFORE the disc: the period
# is T_IN + LIFETIME, phase phi < T_IN is upstream, phi >= T_IN is the wake it
# already had with age tau = phi - T_IN. The particle count does not change;
# the same N is re-budgeted, which is why BRIGHT below gains a factor.
#
# THE SHAPE OF THE LEG is actuator-disc flavoured and walked BACKWARD from the
# crossing point the wake leg already defines (a blade tip for a tip slot, a
# spanwise station for a sheet slot), so continuity at the disc is exact by
# construction rather than by tuning:
#
#   d(u)   upstream distance, u = 1 at birth and 0 at the disc. Its slope at
#          u = 0 is pinned to the wake's own disc velocity, so there is NO
#          axial-speed pop where the two legs meet -- the one discontinuity
#          this could not afford. Far upstream the slope falls to IN_F of it,
#          which is the slow drift the photographs show.
#   r(d)   mass conservation through the on-axis induced-velocity law
#          u(d)/u_disc = 1 - d/sqrt(d^2 + Rv^2): the tube's radius is
#          r0 / sqrt(that), which flares past 2x by the mouth. Because it is a
#          function of the DISTANCE and not of the phase, a spun-down prop has
#          a short AND narrow funnel and nothing teleports up the ramp.
#   phi    a per-slot RANDOM azimuth, locked onto the crossing azimuth only in
#          the last few centimetres. See below: this is the whole causality.
#
# ── THE CAUSALITY, WHICH THE FIRST CUT GOT BACKWARDS ────────────────────────
# The first working version put the parcel upstream at its CROSSING azimuth --
# the physically defensible choice, since a parcel really does hold its azimuth
# while it drifts in. It looked wrong, and it looked wrong in a way worth
# writing down, because the geometry was right and the STORY was inverted.
#
# The crossing azimuth is theta(tc) + blade offset. Parcels arriving at
# different times therefore sit at different azimuths, and the set of them
# currently upstream traces a spiral -- the same helix the wake continues,
# drawn on an expanding cone. So the picture said: a vortex already exists in
# front of the propeller, drifts in, and passes through. That is exactly
# backwards. Nothing is spinning ahead of a propeller. The blade is what makes
# the vortex, at the instant it cuts the air.
#
# So the inflow leg is PLAIN WATER carrying motes and it takes none of the
# wake's identity. Its azimuth is a per-slot hash, uniform over the circle, which
# leaves an axisymmetric haze with no helix, no spoke and no phase relationship
# to the blades at all; it converges onto the crossing azimuth only over the
# last IN_LOCK-worth of the leg, a few centimetres, so position is still exactly
# continuous at the disc. Its grain is the FAR WAKE's -- big, soft, dim and
# neutral -- and its core is a loose 10 cm blob rather than a 1 cm rope.
#
# Everything that says "vortex" therefore switches on AT the disc: the tight
# W_CORE0 core, the swirl, the blade phase lock, and (above inception) the
# vapour. The step in identity across the disc plane is not a seam to be tuned
# away -- it IS the propeller doing its job, and it is the one thing in the
# frame that says the blade caused this rather than merely stood in it.
# Position stays continuous; IDENTITY deliberately does not.
#
# T_IN GREW WITH THE WATER CALIBRATION for the same reason LIFETIME did, and
# by the same arithmetic: the funnel's reach is a SPEED times this time, and
# the speed fell to a third. 0.28 s at 1.29 m/s reaches 27 cm, which is inside
# the hub; 0.80 s puts the mouth back at 0.77 m, where the air version had it.
# It also keeps T_IN / PERIOD at 25%, so IN_SHARE, WAKE_SHARE and every budget
# derived from them are untouched.
T_IN = cli_arg("--t-in", 0.80, float)   # s spent upstream; ~25% of the period
# IN_SHARE is the answer to the first thing that went wrong here. Giving EVERY
# slot an inflow leg costs the wake 25% of its parcels, and the far wake is
# exactly the place in this picture that cannot afford them: it is already at
# the count where core_weight = 0 is what keeps it smoke instead of sand, and
# the before/after crop showed it turn grainy at once. The funnel does not need
# that many. It is a faint, smooth, low-contrast object spread over far less of
# the frame than four metres of slipstream, so HALF the slots run the full
# inflow+wake loop and half are still born at the disc. The wake keeps ~88% of
# its parcels for a funnel drawn with a quarter of a million.
IN_SHARE = cli_arg("--in-share", 0.50, float)
IN_K = 1.6               # the acceleration knee: 1 = uniform, higher = later rush
IN_F = 0.45              # far-upstream drift, as a fraction of the disc speed
IN_RV = 0.70             # m, the induced-velocity law's radius -- sets the flare
# The flare multiplier, spread across the rake's radial stations below: a stream
# is a COLUMN and not a cone shell, so the inner stations have to fill the
# middle of it while the outer one puts the mouth past 2x the tip radius.
IN_FLARE_HI = 1.50
# How sharply the random upstream azimuth converges onto the crossing azimuth.
# (1 - uu)^26 is ~0 over 90% of the leg and snaps home in the last 3 cm, which
# is where a real prop imparts rotation: at the blade, not before it.
IN_LOCK_P = 26.0
# ── THE RAKE, AND WHY THE HAZE ALONE WOULD NOT DO ──────────────────────────
# The first attempt at plain flow was a smooth axisymmetric haze -- a per-slot
# random azimuth and nothing else -- and it failed in the opposite direction
# from the helix: bright enough to see, it was a shapeless glow around the
# propeller that said nothing about direction; dim enough not to glow, it was
# not there at all. A clear fluid has no texture of its own. A stream has to be
# MADE visible, and in a wind tunnel that is a smoke rake; underwater it is
# already done for you, because silt is not uniformly distributed -- it comes
# in strands and clouds, and a diver's light picks out a finite set of them.
#
# So the upstream azimuth and the flare are quantised onto IN_LINES_A x
# IN_LINES_R streamlines, jittered per line so the grid does not read as a
# grid. The lines are STATIC IN THE WORLD -- they do not rotate, they carry no
# blade phase, they are the same lines frame to frame -- which is what drawn-in
# sediment looks like and is the exact opposite of the helix that made the
# first cut read backwards. What moves along them is the water.
IN_LINES_A = 30          # streamlines around the axis
IN_LINES_R = 5           # and across the tube's radius
IN_CORE = 0.030          # m, the loose blob a parcel of undisturbed water is
IN_TURB = 0.05           # m of curl wander at the mouth -- open water is not a lathe
IN_TURB_P = 1.5
PERIOD = T_IN + LIFETIME
# ── THE PITCH-HISTORY RING ──────────────────────────────────────────────────
# The closed form survives an interactive pitch on one condition: a parcel shed
# at tb was shed under the pitch that was set AT tb, not the one set now. So the
# state f(seed, slot, t) gains one more argument -- a host-side ring of the past
# PERIOD's worth of loadings, indexed by FRAME (floor(tb / DT)) and never by a
# wall clock, so a seek and a walk still land on the same wake. One period is
# all a slot can remember; the margin is for the epsilon in the floor.
HIST_N = int(math.ceil(PERIOD / DT)) + 8
# Fraction of slots in the WAKE at any instant, which is what BRIGHT and SIGMA
# below have to pay back.
WAKE_SHARE = (1.0 - IN_SHARE) + IN_SHARE * LIFETIME / PERIOD
# ── Sprite size and the flux it carries ─────────────────────────────────────
# The sprite is a GRAIN of the core, not the core: tying its radius to wc (the
# obvious first move) grows its area ~400x over the life, and since a sprite
# contributes radiance x area, the far wake then out-blazes the very filaments
# it diffused from -- the picture inverts, which is what the first pass did.
# The grain grows instead, and its radiance is divided by the area it gained,
# so a parcel's total flux is roughly conserved and the far wake dims because
# it is SPREAD rather than because it was told to.
#
# GROWTH IS A KNEE, NOT A RAMP -- the second thing this had to learn. The first
# pass grew the grain LINEARLY in age (1 + 3.2 af), which is the one law that
# cannot serve both halves of this picture: small enough to keep the near ropes
# tight left the aged grains at ~1.5 px, far too small to touch their
# neighbours, and the far wake came out as SAND -- discrete specks reading as
# static rather than smoke. Growing them enough to fuse then fattened the young
# ones and dissolved the ropes. Putting the grain on the same af^p knee the
# burst core already uses (W_BURST_P) separates the two populations: below
# af~0.4 the grain is still essentially SPRITE_R0 and the rope is a rope, above
# it the grain balloons ~13x and neighbouring sprites OVERLAP into a continuum.
# Overlap is the whole point -- a smooth far wake is not a smoother sprite, it
# is enough sprites per pixel that no individual one is visible.
SPRITE_R0 = cli_arg("--grain", 0.0026, float)   # grain radius at shedding (m)
SPRITE_GROW = cli_arg("--grow", 13.0, float)    # x radius by end of life
SPRITE_GROW_P = 2.2                             # its delay -- the same knee as the burst
# The flux split pays for that area. Radius x13 is area x170, so a grain that
# kept its radiance would bloom the far wake to white; FLUX_P near 1 divides the
# radiance by very nearly the area gained, which holds the far wake at the
# per-pixel brightness it had while spreading its energy over 170x the film.
# 0.80 is a deliberate half-step back from conserving: it trades a little
# smoothness for a far wake that is still THERE.
FLUX_P = cli_arg("--flux", 0.80, float)         # 1 = flux-conserving, >1 = dims faster
# The inflow leg does not use that knee at all: it is not a rope that later
# diffuses, it is smoke from the first instant, so its grain is simply the FAR
# wake's -- big, soft, overlapping -- for the whole leg and right up to the disc.
# That is what carries the causal beat. The parcel is a fat dim blob on the
# upstream side of the disc plane and a 1 cm bright filament on the downstream
# side, and the reader's eye puts the cause in between, on the blade.
#
# IN_GAIN stays flat-ish rather than ramping up into the disc, which the first
# cut did and which is precisely how the funnel took on the wake's identity: a
# parcel one centimetre ahead of the blade was drawn at the shed vortex's own
# brightness, so the rope appeared to already exist in front of the propeller.
#
# DIMMED FOR THE MEDIUM: 0.90 -> 0.42. In air the funnel was the same smoke the
# wake was made of and could sit at nearly the wake's own brightness; here it
# is suspended silt lit only by what the downwelling sun has left after three
# metres of water, and the wake it feeds is a bubble column. Left at 0.90 the
# motes out-glowed the slipstream and the frame read as a fog machine again.
IN_GROW = cli_arg("--in-grow", 5.0, float)      # x grain radius, the whole leg
IN_GAIN = cli_arg("--in-gain", 0.30, float)     # pre-flux radiance of the motes
IN_GAIN_FAR = 0.30       # x that, at the mouth: the far end is much fainter
# The last speckle source was the sprite's t^9 CORE spike: the falloff plants
# a 1-2 px hard dot at every sprite's centre regardless of drawn radius, and
# `softness` deliberately does not reach it (it is what keeps a few-pixel ember
# from reading as a blob). It is a knob now -- BillboardRepr.core_weight, 0.85
# being the old hardcoded constant -- and this demo turns it OFF below, which
# is what lets the default read smooth where it used to take --n 5000000 to
# average the dots away. 1.7M is where the 25 fps floor puts the default on a
# 4070 at 1280x800 -- 44.2 ms at 2M, 40.4 at 1.7M, 38.2 flat.

# ── The look ────────────────────────────────────────────────────────────────
# The density volume is LATCHED at these bounds and never refitted: a box that
# tracks its own matter re-phases the lattice every frame and the volume swims.
# Elongated along the wake and tight across it.
#
# THE BOX GREW UPSTREAM and only upstream. It used to start at x = -0.35, a
# hand's width in front of a disc nothing had ever been in front of; the funnel
# reaches U_DISC * T_IN / IN_K_DEN = 0.68 m, so x = -0.85 covers it with margin.
# It did NOT grow sideways to the mouth's ~2.0 m, and that is a decision rather
# than an oversight: the volume is a fixed BOX_RES^3 lattice however big the box
# is, so widening it to the mouth would coarsen the lattice by 1.7x across the
# slipstream -- which is the one axis whose detail the self-shadowing lives on --
# to compute transmittance for parcels that have nothing between them and either
# the eye or the sun and would have come back as 1.0 anyway. The mouth sits
# outside the box and is unshadowed, which is what unshadowed water looks like.
#
# AND IT WAS RE-DERIVED ONCE FOR THE WATER CALIBRATION, downstream this time.
# Reach at the default helm is u x LIFETIME = 2.00 x 2.4 = 4.80 m, and the
# funnel now stands 0.77 m ahead of the disc, so the box spans x = -0.85 to
# +4.85: centre 2.00, half 2.85. Done ONCE and latched, because the point of a
# latched box is that it does not move; and because it moved at all, the
# BOX_VOX_RATIO below has to be re-read, which is exactly the trap it exists
# for. The cross-section did not change, so the lattice is coarser along the
# wake by 6% and unchanged across it -- the axis the self-shadowing lives on.
BOX_CENTER = (2.00, 0.0, 0.0)
BOX_HALF = (2.85, 1.20, 1.20)
BOX_RES = cli_arg("--vol-res", 160, int)
BOX_HALF_REF = (2.45, 1.20, 1.20)   # the pre-inflow box SIGMA was tuned against
# sigma_t one particle contributes, and it pays TWO rents. WAKE_SHARE is the
# one BRIGHT pays: fewer parcels downstream is less optical depth downstream.
# The box-volume ratio is subtler and is the trap in resizing a latched box at
# all -- the scatter deposits sigma per PARCEL with no division by cell volume,
# so a bigger voxel simply catches more parcels, and tau along a ray comes out
# proportional to the voxel volume while the 8-step march's ds cancels. Grow the
# box 10% and the whole wake gets 10% thicker for no reason anyone would guess
# from the diff. Divide it back out here and the box is free to move.
BOX_VOX_RATIO = (BOX_HALF[0] * BOX_HALF[1] * BOX_HALF[2]) \
    / (BOX_HALF_REF[0] * BOX_HALF_REF[1] * BOX_HALF_REF[2])
# 0.32 -> 0.50, and the extra depth is the second half of the body fix. On its
# own more sigma only darkened (the medium had no albedo to scatter with); with
# VOL_ALBEDO raised it is what makes the column SHADOW ITSELF -- a thin medium
# lets the sun through everywhere and comes out flat and milky, and the sunward
# flank only separates from the belly once there is enough tau across the tube
# for the far side to be in the near side's shade. Past ~0.7 it veils the ropes,
# which is the nebula's failure and the other wall of this corridor.
#
# AND IT IS NOW DIVIDED BY N, which it never was and should always have been.
# The scatter deposits sigma per PARCEL, so the column's optical depth was
# proportional to the particle count: --n 9000000 was a wake 4.5x thicker, not
# a wake more finely sampled. With a black medium that only ever showed up as a
# slightly murkier far wake and nobody caught it; the moment the medium carries
# light it is unmissable, and 9M came out as milk with the ropes gone. Optical
# depth is a property of the AIR, not of how finely one chooses to sample it.
SIGMA = cli_arg("--sigma", 0.42, float) * (2_000_000 / N) \
    / (WAKE_SHARE * BOX_VOX_RATIO)
EXPOSURE = cli_arg("--exposure", 1.0, float)
EXTINCTION = 0.0 if FLAT else cli_arg("--extinction", 1.0, float)
SHADOW = 0.0 if FLAT else cli_arg("--shadow", 0.72, float)
# 1/N, and NOT the 1/sqrt(N) the other field demos use. Sprite accumulation is
# additive and linear, so N parcels each at 1/N radiance put exactly the same
# total light on the film as 2M parcels at 1x -- the pixel is then an average
# over more samples and nothing else changes. The sqrt law was inherited and it
# meant --n 9000000 arrived 2.1x brighter, which reads as "thicker" and is
# precisely the confusion this demo has to stop making: N is not a body knob.
# Both laws are anchored at 2M, so the default frame is untouched by the change.
#
# The 1/WAKE_SHARE is the inflow leg's rent: the same N now covers the funnel as
# well as the wake, so only WAKE_SHARE of the slots are downstream at any
# instant and the wake's linear density -- and with it its brightness -- would
# otherwise drop by exactly that fraction. This puts the wake back where the
# pre-inflow baseline had it, which is what makes the far/near crops comparable.
# It does NOT undo the lost parcels, only the lost brightness; that is what
# IN_SHARE is for.
#
# AND IT IS RE-ANCHORED AT THE OPERATING POINT, which is pure arithmetic and
# was the single biggest thing the water calibration broke. omg (in the kernel)
# holds the wake's flux per METRE invariant under the throttle, and it is
# normalised at RPS_REF -- so moving the demo's operating point to RPS_REF / 3
# divided every sprite's radiance by 2.8. SIGMA did NOT move with it, because
# sigma is deposited per PARCEL and parcels per metre is reach / count, which
# the longer LIFETIME left within 1% of where it was. So the volume kept
# glowing at full strength while the sprites went out: the first water render
# was MILK WITH NO ROPES, and that is exactly the arithmetic that made it.
#
# OMG_OP is the value omg takes at the DEFAULT helm. Dividing by it puts the
# sprites back beside the volume they are supposed to be drawn on. It is a
# CONSTANT and deliberately not a function of --rps: the throttle's own
# invariance is the thing being re-anchored, not the thing being overridden.
RPS_OP = 2.0             # the helm the look is anchored at
OMG_OP = OM_G_FLOOR + (1.0 - OM_G_FLOOR) * (RPS_OP / RPS_REF)
BRIGHT = cli_arg("--bright", 0.038, float) * (2_000_000 / N) \
    / (WAKE_SHARE * OMG_OP)
# ── THE N CONTRACT: N BUYS SMOOTHNESS, NEVER BODY ───────────────────────────
# Worth stating outright, because the natural response to a thin-looking wake
# is to raise --n, and raising --n cannot fix it. Read the two laws above:
#
#   BRIGHT ~ 1/N         so total emitted radiance is INVARIANT in N. Raise the
#                        count and the same light is diced into more grains:
#                        each pixel is the average of more samples, which is
#                        smoothness and is the only thing it is.
#   SIGMA  ~ 1/N         so the column's optical depth is invariant too. Both
#                        of these were wrong until the body pass: BRIGHT went
#                        as 1/sqrt(N) and SIGMA did not scale at all, so 9M
#                        arrived 2.1x brighter through 4.5x the optical depth
#                        and the honest answer to "why does more N not help"
#                        was that more N was changing the wrong things.
#
# So the two dials are separate and neither substitutes for the other:
#
#   N              = grain smoothness. How finely the same light is diced.
#   VOL_ALBEDO,
#   SIGMA, SHEET   = optical BODY. How much lit medium is actually there.
#
# 9M looks like 2M with less speckle because that is precisely what it is.

TWO_PI = 2.0 * math.pi


# ── The kernel ──────────────────────────────────────────────────────────────
# One launch, no state: out_pos is the exported POSITION allocation (xyz + w =
# the per-particle world radius, WSemantic.Radius) and out_col the exported
# ATTRIBUTE allocation (rgb = linear HDR radiance). Both written in place,
# device to device.
#
# THETA IS NO LONGER A CLOSED FORM, and that is the one thing the throttle
# could not have without breaking. theta_of(t) = OMEGA t^2 / 2 T_RAMP was
# exact while omega was a compile-time constant; the moment omega is a lever,
# any azimuth DERIVED from the current omega is retroactive -- drag the speed
# slider and every filament in the four metres of existing wake is re-placed at
# the azimuth it would have had if the shaft had ALWAYS been at the new speed,
# so the whole helix twists bodily and detaches from the blade tips it was
# welded to. The credibility detail dies in one frame and never comes back.
#
# So theta is INTEGRATED ON THE HOST, once per frame, and stored in the ring
# beside the loading and the rate. The kernel never integrates anything and the
# visible prop mesh reads the same theta_now the ring was stamped with, so
# blades and wake stay welded by construction rather than by agreement between
# two formulae.
@wp.kernel
def shed(out_pos: wp.array(dtype=wp.vec4),
         out_col: wp.array(dtype=wp.vec4),
         hist: wp.array(dtype=wp.vec3),
         t: float, n: int, blades: int,
         turb: float, sheet: float, hub_share: float,
         burst: float, sprite_r0: float, flux_p: float, bright: float,
         t_in: float, in_share: float, hist_n: int,
         k_cav: float, cav_margin: float,
         tip_hint: float, vap_gain: float, bub_cav: float):
    i = wp.tid()
    s = wp.rand_init(90210, i)
    # The inflow leg's own stream, drawn from a SEPARATE seed so that adding
    # this leg did not renumber a single one of the wake's random draws -- the
    # before/after crops are then comparing the same noise realisation.
    s2 = wp.rand_init(1337, i)

    # ── WHICH POPULATION, decided before anything else ──────────────────────
    # Three now: the TIP filaments, the entrained SHEET that fills between
    # them, and the HUB rope on the axis. The hub's share comes OUT of the
    # sheet's -- the tip share (1 - sheet) is untouched -- so the helices are
    # drawn with exactly the parcels they were tuned with. The draw moved to
    # the top of the kernel because tin below has to know about it, and it
    # consumes the same single randf(s) in the same position it always did, so
    # not one of the wake's random numbers is renumbered by the move.
    q = wp.randf(s)
    is_hub = q < hub_share
    is_sheet = q < sheet
    if is_hub:
        is_sheet = False

    # ── The slot's own clock ────────────────────────────────────────────────
    # Uniform phase offsets over one PERIOD give a uniform shedding RATE, so
    # the filament has even density along its whole length rather than beads.
    # The period is the inflow leg plus the wake, and the phase splits the
    # slot's life at the disc: phi < t_in is the funnel walking toward the
    # crossing, phi >= t_in is the wake walking away from it. tb is the
    # CROSSING TIME either way -- in the past for the wake, in the future for
    # the funnel -- which is what makes the two legs agree on the blade.
    #
    # A slot that did not win the IN_SHARE draw simply has t_in = 0: its period
    # collapses back to LIFETIME and it is born at the disc exactly as before.
    # A HUB slot always has t_in = 0 -- there is nothing upstream of a hub
    # vortex to draw. The randf is taken into a local FIRST so the branch
    # cannot short-circuit past it and desynchronise the s2 stream.
    rin = wp.randf(s2)
    tin = t_in
    if is_hub or rin >= in_share:
        tin = 0.0
    u = (float(i) + 0.5) / float(n)
    period = tin + LIFETIME
    ph = t / period + u
    phi_t = (ph - wp.floor(ph)) * period
    inflow = phi_t < tin
    tau = 0.0                                   # age, past the disc
    w_in = 0.0                                  # time still to run before it
    tb = t
    if inflow:
        w_in = tin - phi_t
        tb = t + w_in
    else:
        # tin, NOT t_in. This was a genuine bug and it had been eating a sixth
        # of the parcels since the inflow leg landed: a born-at-disc slot has
        # tin = 0, so subtracting the SHARED T_IN gave every such parcel
        # younger than T_IN a NEGATIVE age -- and wp.pow(negative, 2.4) in the
        # core and grain laws is NaN, which the rasteriser silently discards.
        # The parcels lost were precisely the youngest ones, at the disc, which
        # is where the vapour rope is born and the one place this slice cannot
        # afford to be missing anything.
        tau = phi_t - tin
        tb = t - tau
    af = tau / LIFETIME                         # age fraction, [0, 1)
    if tb < 0.0:
        # Before the propeller started there is no wake. Park the slot on the
        # w < 0 dead sentinel every consumer already tests.
        out_pos[i] = wp.vec4(0.0, 0.0, 0.0, -1.0)
        out_col[i] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        return

    # ── THE HELM AT THIS PARCEL'S OWN CROSSING ──────────────────────────────
    # hist[k] = (loading, omega, theta) as they stood on frame k. The ring is
    # read at floor(tb / DT), which for the WAKE leg is a frame in the past --
    # that is the whole causal beat: move a lever and the near wake changes at
    # once while a metre of old wake is still spiralling away under the pitch
    # and the speed that made it, the boundary between them convecting
    # downstream until it ages out.
    #
    # The INFLOW leg has tb in the FUTURE and there is no future to read, so
    # the clamp to t is not a fudge but the physics: the flow AHEAD of a disc
    # is set by the disc's pressure field, which is established at the speed of
    # sound and not convected. Move a lever and the funnel collapses NOW while
    # the wake still remembers. Those two rates differing on screen is exactly
    # what a controllable-pitch propeller does.
    tbh = wp.min(tb, t)
    kh = int(wp.floor(tbh / DT + 1.0e-3))
    if kh < 0:
        kh = 0
    h = hist[kh % hist_n]
    load = h[0]
    om = h[1]
    # ── AZIMUTH: INTERPOLATE INSIDE THE FRAME, EXTRAPOLATE PAST IT ──────────
    # The ring is stamped once per frame and tb is continuous, so reading
    # h[2] raw would quantise every filament's birth azimuth to 16.7 ms of
    # shaft rotation -- 36 degrees at 360 rpm. The helix would be drawn as a
    # staircase of 36-degree steps instead of a curve. Walking the residual out
    # at that frame's own rate is exact for constant omega and smooth for any
    # omega, and it costs one multiply.
    #
    #   WAKE leg:   tb <= t, kh is the frame at or before it, the residual is
    #               under one frame. This is INTERPOLATION.
    #   INFLOW leg: tb is in the future, kh is clamped to now, and the same
    #               expression becomes theta_now + omega_now * (tb - t).
    #
    # THE EXTRAPOLATION IS SAFE HERE AND NOWHERE ELSE, for a reason that is a
    # decision made three revisions ago rather than a happy accident: the
    # upstream rake carries NO BLADE PHASE. Its azimuths are a fixed set of
    # world-static streamlines and only the last (1 - uu)^IN_LOCK_P of the leg
    # -- a few centimetres -- blends toward the crossing azimuth at all. So the
    # extrapolated theta is invisible over 97% of the funnel and, where it does
    # show, the error is bounded by omega drift over at most T_IN = 0.28 s and
    # is hidden inside a blend that is itself going to zero. Extrapolate the
    # WAKE's azimuth by the same amount and it would be a lie about a filament
    # welded to a blade tip, which is the one thing this demo cannot get wrong.
    th_b = h[2] + om * (tb - float(kh) * DT)
    # ── THE TWO COUPLINGS ───────────────────────────────────────────────────
    # omf is the speed factor and it replaces, exactly, the spin-up ramp
    # fraction the kernel used to compute for itself: at RPS_REF it is 1 and
    # every formula below is the one that was tuned.
    #
    # Static-thrust momentum theory is the anchor. At a fixed pitch a screw has
    # a fixed thrust coefficient, so T ~ omega^2 and the induced velocity
    # v_i ~ sqrt(T) ~ omega -- LINEAR in shaft speed, and the sqrt is spent on
    # the loading instead. The consequence is worth stating because it is not
    # what one expects: the helix's advance PER REVOLUTION is u / n, and with
    # u ~ omega that is INDEPENDENT OF RPM. Change the speed and the loop
    # spacing in metres does not move; what moves is how far down the tube
    # 2.4 s of wake reaches, and how fast the loops are laid down.
    omf = om / OMEGA_REF
    # Momentum theory again, on the other lever: thrust goes as the loading,
    # induced velocity as its square root.
    usc = wp.sqrt(U_FLOOR + (1.0 - U_FLOOR) * load) / USC_DESIGN
    # Radiance follows the circulation, which is linear in omega like v_i is.
    # Since parcels per metre goes as 1 / u ~ 1 / omega, this holds the wake's
    # flux per METRE invariant under the throttle and lets the throttle change
    # the wake's LENGTH, which is the honest reading.
    omg = OM_G_FLOOR + (1.0 - OM_G_FLOOR) * omf
    # ── THE CAVITATION NUMBER, AT THIS PARCEL'S OWN CROSSING ────────────────
    # Momentum-theory level, like everything else in this file. The tip core's
    # suction goes as the circulation SQUARED and Gamma ~ L omega R for a
    # lifting section, so ca = K (L omega R)^2 -- quadratic in BOTH levers,
    # which is why a helm that feels gentle on the wake is violent on the
    # bubbles. It is compared against a specific energy (m2/s2) so rho never
    # enters the kernel, and it is read from the RING, so a parcel carries the
    # cavitation state of the moment it was shed for the rest of its life. The
    # throttle-step shots therefore show a lit stub of new wake against a dark
    # slug of old, which is a thing a ship's engineer has actually seen.
    vt = om * R_TIP                        # tip speed, m/s
    caq = k_cav * load * load * vt * vt / cav_margin

    blade = i % blades

    # ── Birth on the blade ──────────────────────────────────────────────────
    # The tip population is born in the last 2% of the span (that IS the tip
    # vortex); the sheet population is born across the whole span and rolls up
    # behind the blade as the entrained vortex sheet; the hub population is
    # born on the AXIS, a few centimetres off it, and its axial station is
    # pushed aft of the cap dome further down.
    r0 = R_TIP * (0.985 + 0.02 * wp.randf(s))
    if is_sheet:
        r0 = R_TIP * (SHEET_R0 + (0.98 - SHEET_R0) * wp.randf(s))
    if is_hub:
        r0 = HUB_R * (0.25 + 0.75 * wp.randf(s))
    span = r0 / R_TIP

    # ── The crossing ────────────────────────────────────────────────────────
    # Both legs are anchored here: the point (0, r0, phi_c) on the disc, at the
    # time tb. The wake leg leaves it, the inflow leg arrives at it, and every
    # continuity claim below is just "this term is zero at the crossing".
    # Axial: eased from the disc velocity to the developed slipstream. The
    # inner span pushes less air, so it lags -- which is what shears the sheet
    # against the tip helix and starts the braiding.
    ax = 0.55 + 0.45 * span
    u_w = U_WAKE * omf * ax * usc
    u_d = U_DISC * omf * ax * usc
    phi_c = th_b + float(blade) * TWO_PI / float(blades)

    x = float(0.0)
    r = r0
    phi = phi_c
    wc = W_CORE0
    grow = float(1.0)
    gain = float(1.0)
    col = wp.vec3(0.58, 0.80, 1.00)

    if inflow:
        # ── Upstream: the converging streamtube ─────────────────────────────
        uu = w_in / tin                       # 1 at birth, 0 at the disc
        dnm = (1.0 - IN_F) * IN_K + IN_F
        # d_max falls out of pinning the slope at the disc to u_d: no axial
        # speed pop, and a funnel that GROWS with the spin-up ramp for free.
        d_max = u_d * tin / dnm
        d = d_max * ((1.0 - IN_F) * (1.0 - wp.pow(1.0 - uu, IN_K)) + IN_F * uu)
        x = -d
        # Mass conservation through the on-axis induced velocity. Keyed on the
        # DISTANCE, so a short funnel is also a narrow one.
        g = 1.0 - d / wp.sqrt(d * d + IN_RV * IN_RV)
        flare = 1.0 / wp.sqrt(wp.max(g, 0.02)) - 1.0
        # ── THE RAKE: NO SWIRL, NO BLADE ────────────────────────────────────
        # Which of the IN_LINES_A x IN_LINES_R streamlines this parcel is on.
        # Both indices come off the inflow's own stream, and the LINE's own
        # jitter comes off a stream seeded by the line index -- so every parcel
        # on a line agrees on where the line is, which is what makes it a line.
        la = wp.floor(wp.randf(s2) * float(IN_LINES_A))
        lr = wp.floor(wp.randf(s2) * float(IN_LINES_R))
        sl = wp.rand_init(7717, int(la) * 101 + int(lr))
        r = r0 * (1.0 + flare * IN_FLARE_HI
                  * (lr + 0.5 + 0.45 * (wp.randf(sl) - 0.5)) / float(IN_LINES_R))
        # The line's azimuth is fixed in the WORLD -- no omega, no blade offset,
        # nothing that turns. It converges onto the crossing azimuth only in the
        # last (1 - uu)^IN_LOCK_P of the leg, a few centimetres, which keeps
        # position exactly continuous at the disc while putting every trace of
        # rotation where the blade actually imparts it.
        alpha = (la + 0.5 + 0.5 * (wp.randf(sl) - 0.5)) * TWO_PI / float(IN_LINES_A)
        dphi = wp.atan2(wp.sin(alpha - phi_c), wp.cos(alpha - phi_c))
        phi = phi_c + dphi * (1.0 - wp.pow(1.0 - uu, IN_LOCK_P))
        # A loose blob, not a rope: the tight core is the vortex's, and the
        # vortex does not exist yet.
        wc = IN_CORE * (0.42 + 0.58 * uu)
        grow = IN_GROW
        # A feathered or barely-turning prop draws almost nothing, so the
        # funnel does not merely get short and narrow (which d_max and the
        # distance-keyed flare give for free above) -- it goes faint too.
        gain = IN_GAIN * (1.0 - (1.0 - IN_GAIN_FAR) * uu) * omg \
            * (IN_G_FLOOR + (1.0 - IN_G_FLOOR) * load) / IN_GAIN_DESIGN
        # Suspended sediment lit by what is left of the sun three metres down:
        # a pale green-grey, warmer than the vapour and much dimmer than it.
        col = wp.vec3(0.50, 0.56, 0.52)
    else:
        # ── Downstream: the wake ────────────────────────────────────────────
        x = u_w * tau - (u_w - u_d) * TAU_A * (1.0 - wp.exp(-tau / TAU_A))
        # Radial: contraction toward CONTRACT * r0.
        r = r0 * (CONTRACT + (1.0 - CONTRACT) * wp.exp(-tau / TAU_C))
        # Azimuth: the blade's own phase at shedding, plus a self-induced
        # rotation of the helical system that DECAYS with age (integral of a
        # decaying rate). The inner sheet sits deeper in the swirl and turns
        # faster, which is the roll-up.
        # The swirl is the filament's own circulation acting on the helical
        # system, so it scales with the loading LINEARLY (not as its root):
        # feather the blade and the helix stops winding, which is the single
        # most legible thing the pitch slider does to the near wake. Against
        # the throttle it is a fixed FRACTION of the shaft rate, which is what
        # SWIRL_F always meant -- so the wake's angular twist per metre, like
        # its advance per revolution, is a property of the pitch alone.
        sw = SWIRL_F * om * (1.6 - 0.6 * span) * load / SWIRL_DESIGN
        phi = phi_c + sw * TAU_S * (1.0 - wp.exp(-tau / TAU_S))
        # ── DOES THIS PARCEL BOIL? ──────────────────────────────────────────
        # One comparison, evaluated on the helm as it stood when the blade cut
        # this parcel. The hub reads the same number with a 12% head start,
        # which is the whole reason its rope lights first.
        cq = caq
        if is_hub:
            cq = caq * HUB_CAV
        exc = wp.max(cq - 1.0, 0.0)          # the EXCESS over the margin
        # INTENSITY SATURATES and LENGTH DOES NOT, and the split is the point.
        # Past a modest excess a core is simply full of vapour and cannot get
        # any whiter; what a harder-pressed screw buys is a rope that survives
        # FURTHER before the pressure recovers and it condenses.
        cavi = 1.0 - wp.exp(-exc / CAV_KNEE)
        # ── COLLAPSE, NOT DIFFUSION ─────────────────────────────────────────
        # The air version's far wake DIFFUSED: burst core, ballooning grain,
        # rope dissolving into smoke. Vapour does not do that. It condenses the
        # moment the core pressure recovers, and it does so over centimetres --
        # so the visible rope has an END, at an age set by the excess, with a
        # short smoothstep in front of it so the tail reads as a collapse and
        # not as a fade. This is the single most recognisable thing about
        # cavitation footage after the whiteness itself.
        frac = wp.min(1.0, CAV_L0 + CAV_LK * exc)
        cl = wp.clamp((frac - af) / (CAV_COLLAPSE * frac + 1.0e-4), 0.0, 1.0)
        vap = cavi * cl * cl * (3.0 - 2.0 * cl)
        # The loading/speed radiance factor both populations share.
        omgl = omg * (G_FLOOR + (1.0 - G_FLOOR) * load) / GAIN_DESIGN
        # ── SELECTS, NOT BRANCHES, AND THE POWS HOISTED OUT OF BOTH ─────────
        # The obvious way to write the two populations is `if is_sheet: ...
        # else: ...`, and it cost 7.5 ms/frame at 2M. Slots are assigned to
        # populations by a hash, so a 32-thread warp is essentially always
        # mixed and executes BOTH sides of that branch -- which means both
        # sides' wp.pow calls, and pow is the expensive instruction in this
        # kernel. Divergence does not save the work; it only hides it.
        #
        # So the two age powers are computed ONCE, above the split, and the
        # per-population values are chosen with wp.where. Same arithmetic on
        # paper, one pow each instead of two, and it is the whole difference
        # between 48.6 and 41 ms. Measured, not assumed: this is the single
        # largest line item in this slice's frame-time budget.
        pa_b = wp.pow(af, W_BURST_P)
        pa_g = wp.pow(af, SPRITE_GROW_P)
        pa_m = wp.pow(af, 0.55)
        # ── The bubbly wake ────────────────────────────────────────────────
        # Unchanged machinery, reframed: entrained air and turbulent
        # microbubble mist. Broadband white, because that is what a bubble
        # scatters, and BOOSTED when the tips are boiling because collapsing
        # vapour is itself a bubble source -- which is what makes the whole
        # column brighten on the same push that lights the ropes, instead of
        # ropes appearing over a dead wake.
        wc_b = W_CORE0 + burst * pa_b
        grow_b = 1.0 + SPRITE_GROW * pa_g
        gain_b = (1.0 - 0.20 * af) * omgl * (1.0 + bub_cav * cavi)
        col_b = wp.vec3(0.80, 0.87, 0.94) * (1.0 - pa_m) \
            + wp.vec3(0.50, 0.58, 0.64) * pa_m
        # ── A TIP OR HUB FILAMENT: vapour, or almost nothing ────────────────
        # Below inception this is a pressure minimum in clear water and it is
        # INVISIBLE, not dark. TIP_HINT leaves a suggestion of the helix and no
        # more. Above inception the core fills, widens with the excess and goes
        # white; the burst-and-diffuse law is faded out by the same factor,
        # because a vapour tube does not diffuse.
        ww = wp.where(is_hub, HUB_W, 1.0)
        hg = wp.where(is_hub, HUB_GAIN, 1.0)
        wc_f = W_CORE0 * (1.0 + CAV_R * ww * vap) + burst * pa_b * (1.0 - vap)
        grow_f = (1.0 + VAP_R0 * vap) \
            * (1.0 + (VAP_GROW * vap + SPRITE_GROW * (1.0 - vap)) * pa_g)
        gain_f = (tip_hint + vap_gain * vap) * omgl * hg
        col_f = wp.vec3(0.44, 0.58, 0.70) * (1.0 - vap) \
            + wp.vec3(0.88, 0.94, 1.00) * vap
        wc = wp.where(is_sheet, wc_b, wc_f)
        grow = wp.where(is_sheet, grow_b, grow_f)
        gain = wp.where(is_sheet, gain_b, gain_f)
        col = wp.where(is_sheet, col_b, col_f)
        # The hub rope is shed from the CAP, not from the disc plane, so its
        # whole leg is pushed aft far enough to clear the dome.
        x = x + wp.where(is_hub, HUB_X0, 0.0)

    p = wp.vec3(x, r * wp.cos(phi), r * wp.sin(phi))
    # Tip vs sheet is a WAKE distinction -- which structure the blade rolled the
    # parcel into -- and means nothing to water that has not reached the blade.
    if is_sheet and not inflow:
        wc = wc * 2.6                     # the sheet is diffuse from the start
    p = p + wp.vec3(wp.randn(s), wp.randn(s), wp.randn(s)) * (0.55 * wc)

    # ── Instability: curl noise at the LABEL, growing with age ──────────────
    # Sampled at (shed time, birth azimuth) rather than at the position, so a
    # bump on the filament CONVECTS with it. Position-keyed noise would leave a
    # standing disturbance in the world that the wake slides through, and the
    # wake would shimmer instead of braid. Neighbouring shed times get
    # neighbouring labels, so the meander is smooth ALONG a filament while the
    # blades wander independently -- which is what makes them collide.
    # UPSTREAM the same label drives a much smaller wander that grows with
    # distance from the disc: open water arriving at a propeller is not
    # laminar, and a few centimetres of coherent drift keeps the column from
    # reading as a rendered cone. Both terms vanish at the crossing.
    #
    # IT STILL APPLIES TO A CAVITATING CORE, and it should: the vapour rope is
    # a vortex before it is a rope, and the helical instability wiggles it for
    # as long as it lives. What changed is only how long that is.
    lab = wp.vec4(tb * 3.1, wp.cos(phi_c) * TURB_FREQ, wp.sin(phi_c) * TURB_FREQ,
                  span * 1.3)
    amp = turb * wp.pow(af, TURB_P)
    if inflow:
        amp = IN_TURB * wp.pow(w_in / tin, IN_TURB_P)
    if is_sheet and not inflow:
        amp = amp * 1.7
    if is_hub:
        # The hub rope is the STRAIGHT one -- it is held on the axis by its own
        # symmetry and only starts to whip once it is well downstream.
        amp = amp * 0.45
    n1 = wp.rand_init(11)
    n2 = wp.rand_init(29)
    p = p + wp.curlnoise(n1, lab) * amp
    p = p + wp.curlnoise(n2, lab * 2.7) * (amp * 0.35)

    # ── Colour, in kernel ───────────────────────────────────────────────────
    # Three palettes and one flux split. VAPOUR is white with a cold cast (a
    # vapour core is dense enough to be broadband, and the ambient it sits in
    # is blue). BUBBLES are white too but ageing toward the murk's own colour,
    # so the far wake's shape is carried by the SUN term (volume_shadow) rather
    # than by its own emission. SEDIMENT upstream is a pale green-grey mote.
    # Only the flux split is common to all of them.
    c = col
    gain = gain / wp.pow(grow, flux_p)
    if is_sheet and not inflow:
        gain = gain * SHEET_GAIN
    c = c * (bright * gain)

    out_pos[i] = wp.vec4(p[0], p[1], p[2], sprite_r0 * grow)
    out_col[i] = wp.vec4(c[0], c[1], c[2], 1.0)


wp.init()
device = wp.get_preferred_device()
print(f"prop vortex: {N:,} particles on {device}"
      f"{'  [--flat: volumetrics OFF]' if FLAT else ''}")
INTEROP = INTEROP and device.is_cuda

# ── Scene ───────────────────────────────────────────────────────────────────
canvas = tp.Canvas("threepp x warp - propeller cavitation", width=W, height=H,
                   vsync=False, headless=HEADLESS)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
# PINNED: the billboards composite after the post stack and so bypass auto
# exposure entirely; a drifting exposure would move the sprites and the
# background against each other and make the --flat A/B meaningless.
renderer.tone_mapping_exposure = EXPOSURE

# The helm. HEADLESS never builds one, which is what keeps --shot and --bench
# free of UI pixels and reproducible: with no ui, advance() takes its pitch
# from the scripted schedule instead of from a slider nobody is dragging.
ui = tp.ImguiContext(canvas, renderer) if (tp.HAS_IMGUI and not HEADLESS) else None

scene = tp.Scene()
# ── THE STAGE IS UNDERWATER, AND IT IS THE ENGINE'S OWN ─────────────────────
# No new rendering was written for this. Three calls, all of them already load-
# bearing in warp_sailboat.py and warp_hull_sculpt.py:
#
#   background            the deep blue-green a light loses itself in
#   fog_water_surface_y   where air stops. DEPTH, so the world agrees with the
#                         number the cavitation margin is computed from: the
#                         shaft is on y = 0 and the surface is 3 m up.
#   underwater_murk       sigma_t and the in-scatter tint below that line, so
#                         distance falls into murk instead of into black
#
# --flat does NOT turn the murk off, deliberately: the A/B this demo is graded
# on is the FIELD's own volume transport, and zeroing the stage as well would
# be comparing two different scenes rather than one scene with and without the
# two marches.
#
# THE BACKGROUND HAS TO STAY NEARLY BLACK, which is not what the first cut
# assumed. A wake made of ADDITIVE sprites needs something dark to be added to;
# put a mid-teal behind it and the volume's extinction turns the whole
# slipstream into a SILHOUETTE cut out of the backdrop, which is the one read
# this demo cannot have (--vol-albedo 0 shows it perfectly). So the background
# is the colour of water at the edge of a diver's light, the murk supplies the
# in-scatter that lifts DISTANCE out of it, and the wake is still drawn on
# something close to black.
#
# THE TINT *IS* THE BACKGROUND at any distance the murk has saturated over, and
# that is the trap in this pair of knobs: scene.background can be set to
# anything at all and the far field still converges on MURK_COLOR. The first
# pass set the tint by eye as "the colour of the water" and got a mid-teal
# backdrop that no amount of darkening the background could touch.
MURK = cli_arg("--murk", 0.12, float)          # sigma_t, 1/m
MURK_COLOR = (0.004, 0.017, 0.021)
scene.background = tp.Color(0.0018, 0.0075, 0.0092)
if MURK > 0.0:
    renderer.set_fog_water_surface_y(DEPTH)
    renderer.set_underwater_murk(MURK, tp.Color(*MURK_COLOR))

# ── Two framings ────────────────────────────────────────────────────────────
# "side" is the wake's own shot: the disc at the left edge and four metres of
# helix running off to the right. It is the wrong lens for the INFLOW, which
# lives in the half-metre in front of the disc and is three metres wide -- from
# the side that reads as a faint flare crowding the left margin.
# "mouth" is the shot the inflow needs, and it is deliberately almost IN the
# disc plane. Two things have to be legible at once and this is the only angle
# that gives both. The first is scale: edge-on, the blade sweeps the disc's
# diameter as a vertical line, and the stream's mouth is more than twice it in
# the same image, with nothing to convert between them. The second is CAUSE:
# the disc plane cuts the frame in half, plain converging air on the left of it
# and tight bright helices on the right, so the blade is visibly the thing that
# turned one into the other. Three-quarter angles flatter the propeller and
# lose both. A crop centred on the blade is the acceptance test.
VIEW = cli_arg("--view", "side", str)
CAMS = {"side":  ((1.10, 0.56, 3.00), (1.42, -0.02, 0.0)),
        "mouth": ((0.45, 0.60, 5.60), (0.95, -0.05, 0.0)),
        # "prop" is the geometry's own shot and nothing else's: close enough
        # that a 2 cm bolt head is several pixels, three-quarters on so the
        # flange discs are ellipses rather than edge-on lines, and aimed at the
        # hub so the blade roots and their bolt circles fill the frame. It is
        # deliberately NOT a framing the wake reads well at.
        "prop":  ((1.25, 0.62, 2.15), (0.06, 0.0, 0.0))}
CAM_P, CAM_T = CAMS[VIEW if VIEW in CAMS else "side"]
camera = tp.PerspectiveCamera(46, canvas.aspect(), 0.02, 200)
camera.position.set(cli_arg("--cam-x", CAM_P[0], float),
                    cli_arg("--cam-y", CAM_P[1], float),
                    cli_arg("--cam-z", CAM_P[2], float))
camera.look_at(*CAM_T)

# The KEY, and the thing volume_shadow marches toward. Placed across the wake
# and slightly BEHIND it: a backlit slipstream is the arrangement that makes
# self-shadowing legible -- the sunward flank of every loop flares through the
# forward HG lobe while the near flank sits in the wake's own shadow.
#
# UNDERWATER IT IS DOWNWELLING, and that is not a mood choice. Three metres of
# water is a low-pass filter on daylight: the red is gone, the intensity is
# down, and every ray that reaches the propeller has come from ABOVE. Steepened
# from (-0.9, 1.5) to nearly overhead, blue-shifted and dimmed 3.2 -> 2.3, and
# the payoff is the bubble column: a bright top and a dark belly, which is what
# every photograph of a working screw looks like and what the flat-lit air
# version never had.
sun = tp.DirectionalLight(0xB6D2FF, 2.3)
sun.position.set(-0.55, 3.10, -1.55)
scene.add(sun)

# The RAKE -- the second half of "the prop is not a hole". It exists for the
# propeller alone and is aimed to graze the blade faces from the camera's own
# side, low and off to the right, so the twist catches it as a moving edge down
# the span rather than as flat frontal fill.
#
# IT MUST STAY DIMMER THAN THE SUN. The billboards do not sum the light rig:
# they take the single BRIGHTEST DirectionalLight in the scene as their sun and
# march T_sun toward it (VulkanCoreUploads.cpp). Let this one win on max
# channel and the whole wake's self-shadowing silently flips to a front-lit
# read, which is the one arrangement that makes a helix illegible. 1.05 vs 2.3
# is the margin (it was 1.5 vs 3.2 in air; the ratio is what matters and it is
# unchanged); anything that raises it has to check the far crop.
#
# AND UNDERWATER IT IS THE ONLY WARM THING IN THE FRAME, which is the second
# job it picked up with the medium. Three metres of water has taken the red out
# of the sun, and a bronze casting lit by nothing but blue is a grey casting;
# the yard's work light on the hull (or a diver's lamp, take your pick) is the
# one source that still has a spectrum, and it is aimed at the propeller and
# nothing else. This is why the blades still read as bronze in a blue scene.
rake = tp.DirectionalLight(0xFFE2BE, 1.05)
rake.position.set(2.4, 0.95, 3.1)
scene.add(rake)
# The ambient IS the water. Underwater there is no black: every direction the
# bronze can look returns scattered blue-green, and with the key dimmed and
# steepened this is now most of what keeps the casting from going to silhouette
# -- a metal has no diffuse term, so the ambient reaching it is doing real work
# on a metalness 0.62 surface. Warmed slightly toward green and raised 1.0 ->
# 1.5 to pay for the dimmer sun; checked against --view prop, not against the
# wake shot, because the wake does not care and the bronze does.
scene.add(tp.AmbientLight(0x628A94, 1.3))


# ── The propeller: a MARINE CONTROLLABLE-PITCH prop, from primitives ────────
# The reference is a five-blade bronze CPP off a ship's shaft, and three things
# in that photograph are what make it read as one rather than as "a fan":
#
#   the PLANFORM -- wide rounded trapezoids, chord ~0.5 m on a 0.9 m radius, so
#       the blades nearly touch each other and visually swallow the hub. An
#       aircraft prop's narrow paddle is the wrong silhouette entirely.
#   the FLANGE -- every blade sits on a circular palm with a bolt circle. That
#       is the pitch bearing, and it is the single feature that says the blade
#       CAN turn. Without it a pitch slider is just a mesh spinning for no
#       visible reason; with it, the bolts turn with the blade against a fixed
#       seat and the articulation is legible in one frame.
#   the HUB -- bulbous, because the crank ring and the servo live inside it,
#       capped by a bolted ring flange and a dome. R_HUB went 0.14 -> 0.27.
#
# All of it is first-party geometry: stock primitives plus the same loft.
C_MAX = 0.50            # m, the widest chord -- ~0.75 expanded area ratio at B=5
SKEW_DEG = 20.0         # tip skew; moderate, as on the reference
RAKE_M = 0.085          # m of aft rake at the tip
# The rounded-trapezoid outline, as control points in (span fraction, chord /
# C_MAX). Interpolated rather than fitted to a polynomial because the SHAPE is
# the deliverable here and a cubic cannot hold a broad tip and a narrow root at
# once. The last point is what rounds the tip off; the first is the root boss.
PF_F = (0.00, 0.06, 0.15, 0.30, 0.48, 0.65, 0.80, 0.90, 0.96, 1.00)
PF_C = (0.34, 0.52, 0.72, 0.89, 0.98, 1.00, 0.95, 0.86, 0.74, 0.36)
# Thickness / chord down the span: a fat structural root fairing into a thin
# outer blade. At the root this is near 0.6, which is what turns the section
# from an aerofoil into the circular boss that enters the palm.
PF_T = (0.62, 0.44, 0.30, 0.21, 0.16, 0.13, 0.105, 0.088, 0.075, 0.070)
BOLTS = 10              # per blade flange
BOLT_CIRCLE = 0.128     # m
FLANGE_R = 0.165        # m, the palm disc


def blade_geometry(stations=18, around=14):
    """One wide, skewed, twisted marine blade, lofted from cambered sections.

    Span runs along +Y from the palm to R_TIP, chord along Z, and each section
    is rotated about the span axis by its own pitch angle.

    THE TWIST IS A CONSTANT-PITCH HELIX, not a hand-tuned ramp: beta(r) =
    atan(P / 2 pi r) with the pitch P chosen so beta(0.7 R) == BETA_DESIGN.
    That is what a propeller blade is -- a slice of a screw thread -- and it
    matters here beyond looks, because BETA_DESIGN is now a live variable. The
    loft is authored AT the design angle, so the pivot group below is at zero
    rotation there and the slider adds (beta - BETA_DESIGN) about the same
    spanwise axis the sections were already rotated about. Neutral is neutral
    by construction rather than by a magic offset.
    """
    import numpy as np
    sp = np.linspace(R_PALM, R_TIP, stations)
    f = (sp - R_PALM) / (R_TIP - R_PALM)                # 0 at root, 1 at tip
    chord = C_MAX * np.interp(f, PF_F, PF_C)
    thick = chord * np.interp(f, PF_F, PF_T)
    p_screw = TWO_PI * 0.7 * R_TIP * math.tan(math.radians(BETA_DESIGN))
    pitch = np.arctan(p_screw / (TWO_PI * sp))          # 46 deg root, 16 deg tip
    # Skew is NEGATIVE in azimuth: the tip TRAILS the root, which is the whole
    # point of skew (each station enters the wake field a moment after the one
    # inboard of it, so the blade never takes a load impulse across its span).
    # theta increases with time here, so trailing is -phi.
    skew = -np.radians(SKEW_DEG * f ** 1.9)
    rake = RAKE_M * f ** 2.0                            # aft, into the wake

    a = np.linspace(0.0, 2.0 * np.pi, around, endpoint=False)
    # Section in (chordwise, thickness), cambered a touch so the two faces are
    # not mirror images and the light breaks across the blade.
    cw = 0.5 * np.cos(a)
    th = 0.5 * np.sin(a) + 0.13 * (1.0 - np.cos(a) ** 2)
    v = np.empty((stations, around, 3), np.float32)
    for k in range(stations):
        z = cw * chord[k]
        y = th * thick[k]
        c, s = math.cos(pitch[k]), math.sin(pitch[k])
        axl = y * c - z * s + rake[k]                   # axial (out of the disc)
        chd = y * s + z * c                             # in-plane chord
        # Skew is applied as a ROTATION of the whole station about the prop
        # axis, not as a Z translation: a 20 deg lean at 0.9 m is 0.3 m of arc
        # and the difference between an arc and a chord is visible at that
        # size. The spindle axis stays local +Y, which is what the pivot needs.
        ck, sk = math.cos(skew[k]), math.sin(skew[k])
        v[k, :, 0] = axl
        v[k, :, 1] = sp[k] * ck - chd * sk              # span
        v[k, :, 2] = sp[k] * sk + chd * ck

    # ── UVs, because a loft has none ────────────────────────────────────────
    # compute_vertex_normals() fills in normals; nothing fills in uv, and a
    # rust map on a missing attribute is a solid smear of whatever texel 0 is.
    # v runs root -> tip. u runs CHORDWISE as a triangle wave over the section
    # index rather than a sawtooth: the ring closes with j2 = (j+1) % around,
    # so a 0..1 sawtooth would put one quad per section carrying the entire
    # texture backwards across the seam -- a visible stripe down the blade. The
    # triangle wave is 0 at the seam and 1 at the far side, continuous at both,
    # and mirrors the two faces, which for a rust map with no direction in it
    # costs exactly nothing.
    uv = np.empty((stations, around, 2), np.float32)
    tri = 1.0 - np.abs(2.0 * a / (2.0 * np.pi) - 1.0)
    uv[:, :, 0] = tri[None, :]
    uv[:, :, 1] = f[:, None]
    idx = []
    for k in range(stations - 1):
        for j in range(around):
            j2 = (j + 1) % around
            a0 = k * around + j
            b0 = k * around + j2
            a1 = (k + 1) * around + j
            b1 = (k + 1) * around + j2
            # WINDING. The first pass wound these the other way round and the
            # blades were INSIDE-OUT: compute_vertex_normals() then hands every
            # blade vertex a normal pointing into the solid, N.L is negative on
            # the face the camera can see, and the blade shades black no matter
            # what light you put on it. That, not the material, is why this
            # demo's prop was a silhouette -- the hub is a stock primitive
            # with correct windings and it was always lit. Along
            # the ring, increasing j sweeps from the chord's +Z edge toward +X,
            # so (a0, b1, a1) is the order whose cross product points OUT.
            idx += [a0, b1, a1, a0, b0, b1]
    # Fan-cap both ends so the blade is a closed solid.
    for k, flip in ((0, False), (stations - 1, True)):
        base = k * around
        for j in range(1, around - 1):
            tri = [base, base + j, base + j + 1]
            idx += tri[::-1] if flip else tri
    g = tp.BufferGeometry()
    g.set_attribute("position", v.reshape(-1, 3))
    g.set_attribute("uv", uv.reshape(-1, 2))
    g.set_index(np.asarray(idx, np.uint32))
    g.compute_vertex_normals()
    return g


# ── The prop has to be an OBJECT, not a hole ────────────────────────────────
# A bare grey MeshStandardMaterial gave this demo a black silhouette, and for a
# reason that is the wake's own doing: the key is deliberately BEHIND the
# slipstream (see the light below) and there is no environment map, so the blade
# faces the camera sees have nothing at all to return.
#
# BRONZE IS THE HARD CASE OF THAT, and the metalness knob is where it bites. A
# metal has no diffuse term, so on a full metalness=1 surface the AmbientLight
# is refused outright and the only light left is two directional speculars over
# a black background -- a mirror in an empty room, which is black. metalness is
# a knob here for exactly that reason; MET_DEFAULT is the highest value at
# which the prop still reads as an object at the demo's own framing rather than
# as a rim-lit outline. The albedo carries the warm gun-metal bronze, and the
# rust maps stay on as WEAR ONLY -- roughness + normal, no albedo map, because a
# rust albedo over a bronze base is just rust.
RUST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "assets", "rust_coarse_01")
RUST_REPEAT = 2.0     # tiles per uv unit; ~0.35 m of blade per tile
MET = cli_arg("--metalness", 0.62, float)
BRONZE = tp.Color(0.66, 0.42, 0.17)     # linear, warm gold


def rust_map(suffix, srgb):
    path = os.path.join(RUST_DIR, f"rust_coarse_01_{suffix}_1k.jpg")
    if not os.path.exists(path):
        return None
    t = tp.TextureLoader().load(
        path, tp.ColorSpace.SRGB if srgb else tp.ColorSpace.NoColorSpace)
    t.wrap_s = tp.TextureWrapping.Repeat
    t.wrap_t = tp.TextureWrapping.Repeat
    t.repeat = tp.Vector2(RUST_REPEAT, RUST_REPEAT)
    return t


metal = tp.MeshStandardMaterial()
metal.color = BRONZE
metal.metalness = MET
nrm = rust_map("nor_gl", False)
if nrm is not None:
    metal.normal_map = nrm
    metal.roughness_map = rust_map("rough", False)
    # normal_scale well under the old 1.5: the reference is a POLISHED casting
    # with a used surface, not a corroded one, and the map's job is to break the
    # rake into a hundred small highlights down a 0.5 m chord rather than to
    # pit it. roughness MULTIPLIES its map, so 0.75 lands the smoother patches
    # near 0.3 -- a bronze that still returns a broad specular to a grazing key.
    metal.normal_scale = tp.Vector2(0.55, 0.55)
    metal.roughness = 0.75
else:
    print(f"  (no wear maps in {RUST_DIR}; the bronze stays clean)")
    metal.roughness = 0.34
# The machined parts -- flanges, bolt heads, the hub cap -- get their own
# material with no wear map at all. They are the surfaces a yard actually
# machines and re-faces, so they are cleaner and shinier than the cast blade,
# and the roughness step between the two is most of what separates the bolt
# circle from the flange it sits on at the demo's framing distance.
polish = tp.MeshStandardMaterial()
polish.color = tp.Color(0.72, 0.48, 0.21)
polish.metalness = min(MET + 0.12, 1.0)
polish.roughness = 0.24
# aoMap is deliberately NOT set and its 1k JPG deliberately not shipped: the
# Vulkan deferred G-buffer carries albedo / rough-metal / normal texture slots
# and no occlusion term (gbuffer.frag), so it would be 640 kB in the repo for
# a guaranteed no-op.

# ── Assembly ────────────────────────────────────────────────────────────────
# prop
#  |- hub sphere, shaft stub, cap flange + bolts, dome
#  `- arm[b]                 rotate_x(b * 2pi / B) -- the kernel's own offset
#      |- seat ring          FIXED: the hub-side collar the palm turns inside
#      `- pivot[b]           rotation.y = -(beta - BETA_DESIGN)  <-- THE PITCH
#          |- palm disc      and its bolt circle, which turn WITH the blade
#          `- blade
#
# The pivot's axis is local +Y, which after the arm's rotate_x is that blade's
# own spanwise (radial) direction -- the spindle. Putting the bolt circle
# INSIDE the pivot and the seat ring outside it is the whole legibility trick:
# there is then a stationary ring and a rotating one in contact, so a pitch
# change is visible as ten bolts walking round a collar and not merely as a
# blade that happens to look different.
prop = tp.Group()
hub = tp.Mesh(tp.SphereGeometry(R_HUB, 40, 26), metal)
prop.add(hub)
# The shaft, coming forward out of the picture: without it the hub floats.
shaft = tp.Mesh(tp.CylinderGeometry(0.115, 0.115, 0.34, 24), polish)
shaft.rotate_z(math.pi / 2)                  # the cylinder's own axis is Y
shaft.position.x = -0.28
prop.add(shaft)
# The cap: a bolted ring flange, then the dome. Aft (+X), downstream, which on
# this axis convention is the side the wake leaves from.
cap_ring = tp.Mesh(tp.CylinderGeometry(0.198, 0.208, 0.048, 40), polish)
cap_ring.rotate_z(math.pi / 2)
cap_ring.position.x = 0.222
prop.add(cap_ring)
dome = tp.Mesh(tp.SphereGeometry(0.192, 32, 18, 0.0, TWO_PI, 0.0, math.pi / 2),
               polish)
dome.rotate_z(-math.pi / 2)                  # the +Y pole becomes +X
dome.position.x = 0.243
prop.add(dome)

BOLT_G = tp.CylinderGeometry(0.0195, 0.0215, 0.028, 8)   # a hex-ish head
for k in range(12):                          # the cap's own bolt circle
    ang = (k + 0.5) * TWO_PI / 12
    bolt = tp.Mesh(BOLT_G, polish)
    bolt.rotate_z(math.pi / 2)
    bolt.position.set(0.240, 0.163 * math.cos(ang), 0.163 * math.sin(ang))
    prop.add(bolt)

bg = blade_geometry()
SEAT_G = tp.CylinderGeometry(0.186, 0.192, 0.030, 40)
PALM_G = tp.CylinderGeometry(FLANGE_R, FLANGE_R + 0.018, 0.052, 40)
pivots = []
for b in range(BLADES):
    arm = tp.Group()
    arm.rotate_x(b * TWO_PI / BLADES)        # same offset the kernel uses
    seat = tp.Mesh(SEAT_G, polish)           # fixed to the hub
    seat.position.y = R_PALM - 0.036
    arm.add(seat)
    pivot = tp.Group()
    palm = tp.Mesh(PALM_G, polish)
    palm.position.y = R_PALM - 0.004
    pivot.add(palm)
    for k in range(BOLTS):
        ang = (k + 0.5) * TWO_PI / BOLTS
        bolt = tp.Mesh(BOLT_G, polish)
        bolt.position.set(BOLT_CIRCLE * math.sin(ang), R_PALM + 0.030,
                          BOLT_CIRCLE * math.cos(ang))
        pivot.add(bolt)
    pivot.add(tp.Mesh(bg, metal))
    arm.add(pivot)
    prop.add(arm)
    pivots.append(pivot)
scene.add(prop)


def set_pitch(beta_deg):
    """Turn every blade in its flange. The loft is authored at BETA_DESIGN, so
    this is a DELTA about the spindle -- zero rotation is the designed blade.
    The sign is the loft's: a section's chord goes to -X as its pitch angle
    grows, and a +Y rotation takes it to +X, so more pitch is a NEGATIVE
    rotation about the spindle."""
    d = -math.radians(beta_deg - BETA_DESIGN)
    for pv in pivots:
        pv.rotation.y = d

# ── The field ───────────────────────────────────────────────────────────────
cfg = tp.ParticleField.Config()
cfg.capacity = N
cfg.ownership = tp.ParticleField.Ownership.Interop if INTEROP \
    else tp.ParticleField.Ownership.HostRing
cfg.w_semantic = tp.ParticleField.WSemantic.Radius
cfg.attributes = True              # per-particle colour, straight out of the kernel
field = tp.ParticleField.create(cfg)
field.frustum_culled = False
scene.add(field)

# The DENSITY half: the same particles, in one latched world box that only the
# LIGHT TRANSPORT reads. center/half_extent are per-frame writable and are
# deliberately never written.
field.set_density_repr(tp.Vector3(*BOX_CENTER), tp.Vector3(*BOX_HALF), SIGMA, BOX_RES)
# ── THE MEDIUM'S OWN LIGHT, WHICH IS WHERE THE BODY COMES FROM ──────────────
# This was 0.016 -- effectively black -- and inherited straight from the nebula,
# where a lit haze was the enemy. It was the wrong constant for a slipstream and
# it is the single reason this demo read as wireframe: sigma_s = sigma_t x
# albedo, so at 0.016 the volume ABSORBS and in-scatters nothing. Every particle
# added to it subtracted light. That is why 9M did not help -- see the N
# CONTRACT above; the medium had no way to turn density into brightness.
#
# Raised, the same march that computes T_cam and T_sun also fills the tube: the
# sunward flank of the column glows through the phase lobe and the near flank
# sits in its own shadow, and the ropes are then drawn ON something instead of
# on black. The knob is the whole walk between wireframe and milk and it is
# tuned by eye against exactly one failure -- the nebula's veil, where the
# in-scatter rises far enough to wash the filaments out. VOL_ALBEDO is the last
# value at which the near ropes still read at full contrast.
#
# --flat ZEROES IT, which it did not have to do and which keeps the A/B honest:
# the acceptance test is "the same frame with the volume's contribution at 0",
# and now that the medium carries light as well as shadow, leaving it lit under
# --flat would hand the flat render half the win it is supposed to be missing.
#
# AND IT WENT UP AGAIN FOR THE WATER, 0.12 -> 0.30, which is a change of KIND
# and not of taste. In air the medium was smoke: a dark, absorbing, weakly
# scattering thing, and the corridor above (wireframe below, milk above) was
# narrow because smoke has no business being bright. What fills a propeller's
# slipstream underwater is BUBBLES, and a bubble is very nearly a pure
# scatterer -- an air/water interface absorbs essentially nothing. A high
# single-scattering albedo is what that IS. The upper wall is still there, and
# it is still the nebula's veil, but the whole corridor has moved with the
# medium, and the tint below is very slightly blue rather than neutral.
VOL_ALBEDO = 0.0 if FLAT else cli_arg("--vol-albedo", 0.22, float)
# Forward-scattering, and NOT the isotropic 0 it was. A slipstream backlit by a
# key placed deliberately behind it is exactly the arrangement an HG lobe is for:
# g > 0 puts the in-scattered light on the sun side and leaves the camera side
# dark, which is the asymmetry that makes a cylinder read as a cylinder. It also
# matches the billboards' own lit_phase_g so the two halves of the render agree
# about which way the light is going.
VOL_G = cli_arg("--vol-g", 0.55, float)
field.density_repr.albedo = tp.Color(VOL_ALBEDO * 0.92, VOL_ALBEDO * 0.96,
                                     VOL_ALBEDO * 1.00)
field.density_repr.anisotropy = VOL_G

# The IMAGE half.
field.set_billboard_repr(tp.Color(1, 1, 1), tp.Color(1, 1, 1), 1.0, 1.0)
bb = field.billboard_repr
# The falloff is skirt + a t^9 core; softness slides the skirt exponent 4 -> 1.2
# (particlefield_billboard.frag). High is what a SMOKE grain wants: the tight
# core is a spark's shape and, drawn at the aged grain's 10-odd pixels, it is
# exactly the hard speck the far wake must stop reading as. This is a per-FIELD
# knob and not per-particle, which sounds like it should soften the near ropes
# too -- it does not, because a young grain is ~1.5 px across and a radial
# profile has nowhere to act inside a pixel and a half. The knee gives softness
# its age selectivity for free.
bb.softness = 0.92
# The core dot OFF entirely: a smoke parcel has no spark at its centre. The
# ropes survive this -- a young grain is ~1.5 px and its whole profile is one
# bright sample either way -- but the wake stops being 2M pinpricks.
bb.core_weight = 0.0
bb.bright_jitter = 0.0        # the sim authors the colour; do not hash over it
bb.fade_power = 0.0           # no age exists on an interop field
bb.size_taper = 0.0
bb.stretch_seconds = 0.0      # a hard no-op on an Interop field -- see the docstring
bb.lit_phase_g = 0.58         # forward: the backlit flank flares
bb.volume_extinction = EXTINCTION
bb.volume_shadow = SHADOW
bb.volume_ambient = 0.26
bb.volume_sun_gain = 0.85

sim_time = 0.0
frame_no = 0
imported_pos = imported_col = None
host_pos = host_col = None

# ── The helm, and its history ───────────────────────────────────────────────
# beta_now / rpm_now are what the levers ARE this frame -- the sliders, or the
# scripted schedules when there is no UI. hist_np[k] = (loading, omega, theta)
# as they stood on frame k, and it is the only channel through which the past
# reaches the kernel. Pre-filled with the starting loading at a STOPPED shaft
# (omega 0, theta 0) so that every index a slot can reach is defined before
# frame 1 -- there is no separate "before the prop started" case any more, only
# the first entries of a schedule that happens to begin at rest.
import numpy as _np
beta_now = PITCH
rpm_now = RPS * 60.0
omega_now = 0.0
theta_now = 0.0
hist_np = _np.zeros((HIST_N, 3), _np.float32)
hist_np[:, 0] = loading(PITCH)
hist_wp = wp.array(hist_np, dtype=wp.vec3, device=device)


def launch(out_pos, out_col):
    wp.launch(shed, dim=N, device=device,
              inputs=[out_pos, out_col, hist_wp, sim_time, N, BLADES,
                      TURB, SHEET, HUB_SHARE,
                      W_BURST, SPRITE_R0, FLUX_P, BRIGHT,
                      T_IN, IN_SHARE, HIST_N,
                      K_CAV, CAV_MARGIN, TIP_HINT, VAP_GAIN, BUB_CAV])


def advance():
    """The clock, the phase lock and both levers. The sim time is the FRAME
    INDEX times DT and never a wall clock, and the helm history is stamped at
    the SAME frame index, so a seek and a walk see the same lever positions in
    the same order.

    THETA IS INTEGRATED HERE AND ONLY HERE. The trapezoid rule, not Euler: with
    omega linear in t (which is exactly what the spin-up ramp is) the trapezoid
    is EXACT, so the ramp reproduces the closed form this function replaced to
    the last bit and the shots stay comparable across the change. Euler would
    have slipped a fifth of a radian by the top of the ramp -- invisible as
    motion, fatal as a regression diff.

    theta is wrapped into [0, 2pi) before it is stored. Only cos and sin ever
    see it, so wrapping costs nothing, and it keeps a float32 ring from losing
    a degree of resolution after an hour of a window being left open.

    The blades are set here and the ring entry is written here, in one place,
    on purpose: what the reader sees the propeller doing this frame and what
    the wake will remember about this frame are the same numbers."""
    global sim_time, frame_no, beta_now, rpm_now, omega_now, theta_now
    frame_no += 1
    sim_time = frame_no * DT
    if ui is None:
        beta_now = scheduled_beta(sim_time)
        rpm_now = scheduled_rps(sim_time) * 60.0
        om = TWO_PI * rpm_now / 60.0
    else:
        # The slider commands a speed; the spin-up ramp still owns the first
        # T_RAMP seconds of it, so the wake grows into existence rather than
        # appearing whole on frame one.
        om = TWO_PI * (rpm_now / 60.0) * ramp_frac(sim_time)
    theta_now = (theta_now + 0.5 * (omega_now + om) * DT) % TWO_PI
    omega_now = om
    prop.rotation.x = theta_now
    set_pitch(beta_now)
    hist_np[frame_no % HIST_N] = (loading(beta_now), omega_now, theta_now)
    hist_wp.assign(hist_np)      # 3 x 76 floats: ~900 B/frame, below measurable


def device_copy():
    """The renderer's per-frame callback: evaluate the wake straight into the
    two exported allocations, then synchronize. It runs INSIDE render(),
    pre-record, and the host ordering it provides is the ONLY thing sequencing
    these writes against the frame that reads them."""
    launch(imported_pos.array, imported_col.array)
    wp.synchronize_device(device)


def step_frame_host():
    advance()
    launch(host_pos, host_col)
    wp.synchronize_device(device)
    field.submit(host_pos.numpy(), DT)
    field.set_attributes(host_col.numpy())


def step_frame_interop():
    advance()          # device_copy, inside render(), reads sim_time


# ── ARM THE ZERO-COPY PATH ──────────────────────────────────────────────────
# enable_particle_field_interop returns None until after the FIRST render(): the
# field's device state and the renderer's field pass are both created on the
# frame the field is first seen.
field.set_live_count(0)
renderer.render(scene, camera)
step_frame = step_frame_host
if INTEROP:
    from threepp.cuda_interop import VkInteropArray
    try:
        h = renderer.enable_particle_field_interop(field, device_copy)
        if h is None:
            raise RuntimeError("this device cannot export memory "
                               f"(host_fallback={field.host_fallback})")
        if len(h) < 4:
            raise RuntimeError("the renderer exported no attribute buffer -- "
                               "Config.attributes was not honoured")
        imported_pos = VkInteropArray(h[0], h[1], wp.vec4, N, device)
        imported_col = VkInteropArray(h[2], h[3], wp.vec4, N, device)
        step_frame = step_frame_interop
        print(f"       zero-copy: {2 * 16 * N / 1e6:.0f} MB exported "
              f"(positions + attributes), 0 B/frame across the bus")
    except Exception as e:                                # noqa: BLE001
        INTEROP = False
        print(f"       no zero-copy export ({e}); host ring")

if not INTEROP:
    host_pos = wp.zeros(N, dtype=wp.vec4, device="cpu", pinned=True)
    host_col = wp.zeros(N, dtype=wp.vec4, device="cpu", pinned=True)

field.set_live_count(N)

# The funnel's own numbers, evaluated on the host with the kernel's formulae so
# the banner states the claim the demo exists to make rather than a guess: how
# far ahead of the disc the outermost streamline starts, and how much wider than
# the tip radius it is when it does.
# The mouth is quoted at the COMMANDED operating point and not at the
# reference one, because both levers move it: the funnel's length is pinned to
# the disc velocity, which now carries the throttle and the pitch.
OPF = (RPS / RPS_REF) * math.sqrt(
    U_FLOOR + (1.0 - U_FLOOR) * loading(PITCH)) / USC_DESIGN
IN_D_MAX = U_DISC * OPF * T_IN / ((1.0 - IN_F) * IN_K + IN_F)
IN_R_MOUTH = R_TIP * (1.0 + IN_FLARE_HI * (IN_LINES_R - 0.5) / IN_LINES_R * (
    1.0 / math.sqrt(1.0 - IN_D_MAX / math.hypot(IN_D_MAX, IN_RV)) - 1.0))


def cav_ratio(beta_deg, rps):
    """ca / margin for a tip filament shed at this helm. > 1 is cavitating.
    The kernel's own expression, on the host, so the banner and the panel
    state the number the wake was actually drawn with."""
    return K_CAV * loading(beta_deg) ** 2 \
        * (TWO_PI * rps * R_TIP) ** 2 / CAV_MARGIN


print(f"       prop:   {BLADES} blades, R {R_TIP:g} m, wake {LIFETIME:g} s\n"
      f"       shaft:  {RPS * 60:g} rpm (ref {RPS_REF * 60:g}, "
      f"ramp {T_RAMP:g} s)"
      + (f" -> {RPS_TO * 60:g} rpm at t={RPS_AT:g} s" if RPS_AT < 1e8 else "")
      + f", tip {TWO_PI * RPS * R_TIP:.1f} m/s"
      + f", slipstream {U_WAKE * OPF:.2f} m/s, "
      f"advance {(U_WAKE * OPF / RPS if RPS > 0 else 0):.2f} m/rev\n"
      f"       pitch:  beta {PITCH:g} deg (design {BETA_DESIGN:g}), "
      f"L {loading(PITCH):.2f}"
      + (f" -> {PITCH_TO:g} deg (L {loading(PITCH_TO):.2f}) at t={PITCH_AT:g} s"
         if PITCH_AT < 1e8 else "")
      + f", helm history {HIST_N} frames = {HIST_N * DT:.2f} s\n"
      f"       water:  rho {RHO:g}, {DEPTH:g} m down, margin {CAV_MARGIN:.1f} "
      f"m2/s2; tips at {cav_ratio(PITCH, RPS):.2f} x, "
      f"hub at {cav_ratio(PITCH, RPS) * HUB_CAV:.2f} x "
      + ("-> CAVITATING" if cav_ratio(PITCH, RPS) > 1.0
         else ("-> hub rope only" if cav_ratio(PITCH, RPS) * HUB_CAV > 1.0
               else "-> clear")) + "\n"
      f"       inflow: {IN_SHARE:.0%} of slots, {T_IN:g} s upstream "
      f"({T_IN / PERIOD:.0%} of the period), reaching {IN_D_MAX:.2f} m ahead of "
      f"the disc at {IN_R_MOUTH:.2f} m = {IN_R_MOUTH / R_TIP:.1f} x the tip radius\n"
      f"       volume: {BOX_RES}^3 over {2 * BOX_HALF[0]:.1f} x "
      f"{2 * BOX_HALF[1]:.1f} x {2 * BOX_HALF[2]:.1f} m, sigma/particle {SIGMA:g}\n"
      f"       knobs:  extinction {EXTINCTION:g}, shadow {SHADOW:g}, "
      f"grain {SPRITE_R0:g} m, flux {FLUX_P:g}, bright {BRIGHT:.5f}")


def run_to(seconds):
    frames = int(round(seconds * FPS))
    t0 = time.perf_counter()
    for f in range(frames):
        step_frame()
        renderer.render(scene, camera)
        if f % 60 == 0:
            print(f"  t={sim_time:5.2f}", flush=True)
    return frames, time.perf_counter() - t0


if BENCH:
    # Warm to a FULLY DEVELOPED wake, not merely a warm pipeline. The old 2.5 s
    # was longer than the air version's whole 0.85 s lifetime; at 2.4 s it is
    # shorter than the spin-up plus one lifetime, so the frame being timed would
    # be a half-grown wake covering half the film -- which is a cheaper frame
    # than the one anybody looks at, and a bench that flatters itself.
    run_to(T_RAMP + LIFETIME)
    n = 240
    t0 = time.perf_counter()
    for _ in range(n):
        step_frame()
        renderer.render(scene, camera)
    dt = (time.perf_counter() - t0) / n
    print(f"bench {N:,} particles [{'flat' if FLAT else 'volumetric'}]: "
          f"{1e3 * dt:.2f} ms/frame ({1.0 / dt:.0f} fps)")
elif SHOT:
    frames, wall = run_to(SHOT_TIME)
    out = cli_arg("--out", "warp_prop_vortex_flat.png" if FLAT
                  else "warp_prop_vortex.png", str)
    renderer.save_frame(scene, camera, out)
    print(f"simulated {SHOT_TIME:.1f} s ({frames} frames) in {wall:.1f}s, wrote {out}")
else:
    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target.set(*CAM_T)

    def draw_ui():
        """The helm. Two levers, and everything else is a readout of what they
        have already done -- there is nothing here to configure, only the pitch
        and the throttle and the consequences to watch."""
        global beta_now, rpm_now
        tp.imgui.set_next_window_pos(12, 12)
        tp.imgui.set_next_window_size(434, 0)
        tp.imgui.begin("Helm")
        _, beta_now = tp.imgui.slider_float("blade pitch (deg)", beta_now,
                                            BETA_MIN, BETA_MAX)
        _, rpm_now = tp.imgui.slider_float("shaft speed (rpm)", rpm_now,
                                           RPM_MIN, RPM_MAX)
        ld = loading(beta_now)
        # Read the ACTUAL shaft state back out of the integrator rather than
        # recomputing it from the sliders: during the spin-up the two differ,
        # and the number that means anything is the one the wake was given.
        omf = omega_now / OMEGA_REF
        rpm = omega_now * 60.0 / TWO_PI
        usc = math.sqrt(U_FLOOR + (1.0 - U_FLOOR) * ld) / USC_DESIGN
        uw = U_WAKE * omf * usc
        tp.imgui.text(f"beta   {beta_now:5.1f} deg at 0.7R   "
                      f"(design {BETA_DESIGN:.0f})   thrust {ld:4.2f}")
        tp.imgui.text(f"shaft  {rpm:5.0f} rpm   (ref {RPS_REF * 60:.0f})   "
                      f"tip {omega_now * R_TIP:5.1f} m/s")
        tp.imgui.text(f"wake   {uw:5.2f} m/s of water   "
                      f"advance {(uw / (rpm / 60.0) if rpm > 1.0 else 0.0):5.2f} "
                      f"m/rev   reach {uw * LIFETIME:4.1f} m")
        tp.imgui.separator()
        # ── THE CAVITATION INDICATOR ────────────────────────────────────────
        # The one readout this demo exists for. The bar is the tip core's
        # suction as a fraction of the pressure margin, and the line under it
        # is the answer: how far short, or how far over and by how much. imgui
        # here has no coloured text and no progress bar, so the bar is ASCII --
        # which costs nothing and is legible in a 1:1 crop, which is where it
        # actually gets read.
        cr = cav_ratio(beta_now, max(rpm, 0.0) / 60.0)
        fill = int(round(min(cr, 1.0) * 24))
        tp.imgui.text(f"cav    [{'#' * fill}{'-' * (24 - fill)}] "
                      f"{cr:4.2f} x margin")
        tp.imgui.text(f"       {DEPTH:.1f} m down, margin {CAV_MARGIN:.0f} m2/s2"
                      f"   ({RHO:.0f} kg/m3)")
        if cr > 1.0:
            tp.imgui.text(f"CAVITATING   tips +{100.0 * (cr - 1.0):.0f}%   "
                          f"hub +{100.0 * (cr * HUB_CAV - 1.0):.0f}%")
        elif cr * HUB_CAV > 1.0:
            tp.imgui.text(f"HUB ROPE ONLY   hub +{100.0 * (cr * HUB_CAV - 1.0):.0f}%"
                          f"   tips {100.0 * (1.0 - cr):.0f}% short")
        else:
            tp.imgui.text(f"clear water   tips {100.0 * (1.0 - cr):.0f}% short, "
                          f"hub {100.0 * (1.0 - cr * HUB_CAV):.0f}% short")
        tp.imgui.text(f"{tp.imgui.get_framerate():5.0f} fps   "
                      f"{N / 1e6:.1f} M parcels   t={sim_time:6.2f} s")
        tp.imgui.separator()
        # The one thing to say about the lag, because it is the feature and it
        # looks like a bug for the seconds it takes to convect away.
        tp.imgui.text("water carries no tracer. what you see is VAPOUR, and")
        tp.imgui.text("it only exists above the bar. push either lever.")
        tp.imgui.text("both levers move NOW; the wake answers with history.")
        tp.imgui.text(f"{LIFETIME:.1f} s of old slipstream keeps the pitch, the")
        tp.imgui.text(f"speed AND the cavitation it was shed under ({HIST_N}-frame")
        tp.imgui.text("ring), and the boundary convects away. the inflow does")
        tp.imgui.text("not lag: a pressure field is not convected.")
        tp.imgui.text("advance/rev is a property of PITCH alone -- the")
        tp.imgui.text("throttle changes how far the wake reaches, not its")
        tp.imgui.text("loop spacing. that is what a screw does.")
        tp.imgui.end()

    def animate():
        step_frame()
        if ui is not None:
            controls.enabled = not ui.want_capture_mouse
        controls.update()
        renderer.render(scene, camera)
        if ui is not None:
            ui.render(draw_ui)               # overlay: after render(), same frame

    canvas.animate(animate)
