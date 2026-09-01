"""A twin-screw stern CAVITATING underwater -- NVIDIA Warp sim, Vulkan rendering.

warp_nebula_vk.py proved the thesis on a cloud: particles carry the IMAGE, the
field's own density volume carries the LIGHT TRANSPORT. A nebula is the easy
case for it, because a cloud has no shape you can be wrong about. This is the
hard case, and the one the feature was built for: a five-bladed controllable-
pitch propeller on a ship's shaft three metres down, shedding tip vortices into
the classic interleaved helices -- and, when it is pushed hard enough, filling
their cores with VAPOUR and boiling the backs of its own blades.

THE SHIP IS A TWIN-SCREW WORKBOAT -- a tug or a coastal supply vessel, 1.8 m
controllable-pitch screws 2.34 m apart under eight metres of counter -- and that
class is what every rpm below has to be read against: 120 rpm is a working helm
on a 1.8 m screw, where the same number on a 6 m container-ship wheel would be
flank speed and on a 0.3 m outboard would be idle.

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

THERE IS A HULL OVER IT NOW, AND THE HULL IS NOT SET DRESSING. That was the
trap this revision had to avoid: a stern lowered over perfectly STEADY
cavitation quietly advertises the model's limits, because real cavitation next
to a hull BREATHES once per revolution. The hull drags a boundary layer, the
disc turns inside it, the blade sweeping through the slower water at twelve
o'clock loads up, and the vapour blooms at the top of the disc every rev and
corkscrews away downstream as a bright band. That breathing is the headline of
this revision, it costs one scalar in the helm ring, and the second screw and
the stern above it are the stage it is played on.

    w(theta) = w_mean + w_peak lobe(theta)   the circumferential wake fraction
    V_a      = V_ship (1 - w_mean)           the SPEED OF ADVANCE, which is
                                             what J has always meant behind a
                                             hull and is what the panel says
    lambda   = K_T(J_local) / K_T(J)         the local loading, linearised, and
                                             it multiplies the cavitation
                                             criterion and NOTHING else

    THE BOLLARD CONTROL IS DERIVED AND NOT AUTHORED. lambda's amplitude is
    proportional to J -- see wake_lobe_k -- so at V_ship 0 there is no
    breathing at all, because a boundary-layer shadow needs way through the
    water. Push the ship-speed slider and the ropes start pulsing at the top
    of the disc. There is no `if` anywhere in that sentence.

TWO SCREWS, AND THEY ARE A MIRRORED PAIR. Outward-turning, as nearly every
twin-screw ship is built: seen from astern the starboard screw runs clockwise
and the port screw anticlockwise, so both tips sweep OUTBOARD over the top.
The screw this demo always had turns its top toward +Z, which makes it the
LEFT-HANDED PORT one -- that is a deduction from the axis convention, not a
choice -- and the starboard screw is its reflection in z: mirrored loft with
the winding flipped, reversed blade order, reversed pitch rotation, and in the
kernel one sign on the last line. Half a blade pitch of phase between the
shafts so they do not strobe together. `--view stern` is where the two helices
are legible winding OPPOSITE ways in one frame, and `--view prop` / `--view
stbd` are the same screw twice and have to be mirror images of each other.

    pip install warp-lang
    python warp_prop_vortex.py                   # window; PUSH ANY SLIDER
    python warp_prop_vortex.py --view back --shot 5.5 --rps 3.2
                                                 # THE SHEET, on the backs of
                                                 # the blades. 192 rpm -- just
                                                 # over its own line, which is
                                                 # where it READS: past ~250
                                                 # the slipstream is a wall of
                                                 # white and the blades are a
                                                 # silhouette cut out of it
    python warp_prop_vortex.py --view back --shot 5.5 --rps 4.9 --vapour 0 --sheet-gain 0 --in-gain 0
                                                 # the sheet ALONE: the three
                                                 # wake populations turned off
                                                 # and nothing else touched
    python warp_prop_vortex.py --view back --shot 5.5 --rps 4.9 --speed 5
                                                 # AND IT BREATHES: the top of
                                                 # the disc carries a sheet and
                                                 # the bottom carries none
    python warp_prop_vortex.py --view back --shot 5.5 --rps 4.9 --speed 5 --wake-peak 0
                                                 # the same helm, lobe OFF
    python warp_prop_vortex.py --shot 5.5 --rps 4.2 --speed 5
                                                 # THE BREATHING: 0.90 x mean,
                                                 # 1.21 x at twelve o'clock
    python warp_prop_vortex.py --shot 5.5 --rps 4.2 --speed 5 --wake-peak 0
                                                 # the same helm, lobe OFF
    python warp_prop_vortex.py --shot 5.5 --rps 2.9    # the bollard control
    python warp_prop_vortex.py --view stern      # both wakes, opposite senses
    python warp_prop_vortex.py --openwater       # the B-series check, then exit
    python warp_prop_vortex.py --shot 5.5        # 120 rpm: JUST BELOW inception
    python warp_prop_vortex.py --shot 5.5 --rps 3    # 180 rpm: ropes that END
    python warp_prop_vortex.py --shot 5.5 --rps 5    # 300 rpm: the whole wake boils
    python warp_prop_vortex.py --shot 5.5 --pitch 30 # same gate, other lever
    python warp_prop_vortex.py --shot 5.5 --speed 4  # the ship has way on
    python warp_prop_vortex.py --shot 5.5 --rps 4 --speed 6   # J 0.83, eta_0 0.59
    python warp_prop_vortex.py --shot 5.5 --flat # the SAME frame, knobs at 0
    python warp_prop_vortex.py --depth 1.2       # shallower: it cavitates sooner
    python warp_prop_vortex.py --no-ocean        # the flat analytic ceiling, as before
    python warp_prop_vortex.py --view mouth      # across the disc: water IN, ropes OUT
    python warp_prop_vortex.py --view prop       # the bronze, close
    python warp_prop_vortex.py --pitch 8         # P/D 0.31: a third of the thrust
    python warp_prop_vortex.py --shot 5.0 --rps 5 --rps-to 2 --rps-at 4.6
                                                 # the throttle's step, 0.4 s later
    python warp_prop_vortex.py --n 5000000       # more particles: a grainless far wake
    python warp_prop_vortex.py --bench           # frame time, knobs on vs --flat
    python warp_prop_vortex.py --film            # THE FILM: 55 s, 1080p, 5M
    python warp_prop_vortex.py --film --probe 100    # what it will cost first
    python warp_prop_vortex.py --film --takes feather # re-cut ONE beat in place

    NOTE the 5.5 s: the wake is 1.7 s deep and the shaft takes 2 s to spin up,
    so anything before ~3.8 s is still showing the ramp in its far half.

THE NUMBERS ARE OUTPUTS NOW, and that is what this revision is. Thrust used to
be an authored smoothstep on the blade angle, torque did not exist at all, and
the clip ran at the bollard where efficiency is zero by definition -- so the
one question a marine engineer asks of a propeller demo ("can you pull out Kt,
Kq and eta_0?") had no honest answer. It does now, because the blade geometry
this demo drew to LOOK like a ship's screw turned out to be one: five blades at
an expanded area ratio of 0.75 is a Wageningen B5-75, the most measured
propeller family there is.

    K_T, K_Q   the Oosterveld & van Oossanen (1975) regression polynomials,
               39 and 47 terms in (J, P/D, EAR, Z), hard-coded first-party.
               `--openwater` prints the check they had to pass against the
               published B5-75 curves before anything was wired to them.
    J          V_a / n D, from the NEW ship-speed slider. At the bollard it is
               zero and so is eta_0, which is honest and is still the default.
    T, Q       K_T rho n^2 D^4 and K_Q rho n^2 D^5. 17.8 kN and 4.25 kNm at
               the default helm, for 53 kW at the shaft.
    v_i        the momentum inversion WITH advance, which is the ONLY channel
               from those numbers to the water:
                   v_i = (-V_a + sqrt(V_a^2 + 2 T / (rho A))) / 2
               u_disc = V_a + v_i, u_wake = V_a + 2 v_i. Nothing downstream is
               tuned separately any more -- wake speed, funnel length, helix
               advance and cavitation all hang off T.

The panel draws the open-water diagram live, with the operating point moving
on it as the sliders move, and says "outside B-series validity" whenever a
lever has pushed J or P/D past the box the regression was fitted over.

--film STILL RUNS AND ITS CHOREOGRAPHY IS NOW OUT OF DATE, which is stated
rather than quietly tolerated: the track was composed for one screw turning on
the axis, and beat 3's push-in key is 0.89 m off the port wake's axis against a
0.90 m tip radius -- the camera flies through the slipstream. The clearance
check measures both shafts now and --film prints the warning every run. The
beats, the helm schedule and the keyframe times are untouched; re-cutting the
track is a film-v2 slice and taking the look budget for it here would have come
out of the breathing this revision exists for.

--film IS THE DEMO ARGUING ITS OWN CASE, and the argument it makes is that
inception is a threshold rather than a look. Six beats and no cuts: a still
prop, a spin-up to 120 rpm in CLEAR water so the pull is felt before anything
is seen, a crawl to 138 rpm where the hub rope lights ALONE (the hub crosses at
127.8 rpm and the tips not until 141.2, so there is a real window and the
camera holds in it), a throttle step down that shortens the ropes and sends the
old wake off frame, a pitch feather that kills the wake while the shaft keeps
turning, and a slam back to 30 deg for the re-bloom. The camera is a Hermite
track and the helm is a keyframe table, both read at the film's own frame
index -- so the film is a pure function of frame number and --takes re-renders
one beat into exactly the picture the full run made. See THE FILM below.

CAVITATION IS A THRESHOLD AND THE DEMO IS TUNED TO SIT UNDER IT. At the default
helm -- 120 rpm, beta 22 deg, 3 m down -- the blade is loaded to 83% of what it
takes to start boiling and NOTHING happens: the water is clear, the wake is a
faint bubbly column, and the helices are barely a hint. Push a slider and the
ropes switch on. That is the beat, and it is the beat a ship's engineer lives
with: cavitation is not a setting, it is an operating point you cross.

WHAT DECIDES IT IS BURRILL'S, not a coefficient chosen to make the beat work.
Both of the standard axes are computed live from quantities this file already
has, and both of them move when the shaft does:

    V_R^2   = V_a^2 + (0.7 pi n D)^2         resultant speed at 0.7 R
    sigma   = (p_atm + rho g DEPTH - p_v) / (0.5 rho V_R^2)      at 0.7 R
    tau_c   = T / (0.5 rho A_P V_R^2)        thrust loading, A_P from EAR by
                                             Taylor's 1.067 - 0.229 P/D
    limit   = 0.30 sigma^0.57                Burrill & Emerson's 5% back-
                                             cavitation line, as fitted
    e       = max(0, tau_c / (0.610 limit) - 1)   the EXCESS, per parcel

The one authored number left is that 0.610: Burrill's lines are percentage BACK
COVERAGE limits and inception is below all of them, by an amount that depends
on section shape and nuclei content and that no chart of this kind carries. So
inception is a stated fraction of the 5% line and that fraction is set, as the
old coefficient was, to put the default helm just under.

AND THE SHAPE OF IT CHANGED THE WHOLE DEMO. tau_c does not depend on shaft
speed at all at the bollard -- T and V_R^2 both go as n^2 -- and sigma goes as
1/n^2, so the criterion goes as n^1.14 where the authored one went as n^2.
Cavitation arrives more gently and never gets as violent: the hub rope lights
at 127.8 rpm and the tips at 141.2 (they were 124.6 and 131.8), and 300 rpm is
2.36x inception where the old model claimed 5.19x. That is a 13 rpm hub-only
window instead of 7 -- a better beat, and one that was derived rather than
tuned.

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

AND THE CRITERION HAS FOUND ITS OWN PHENOMENON. Burrill's diagram is a BACK
CAVITATION chart: every line on it is a stated percentage of the blade's
SUCTION SIDE covered by a vapour sheet. This file has been gating tip vortices
with it -- the one structure it does not describe -- since the criterion
landed, and the sheet it was always about was not drawn at all. It is now, on
the backs of the blades, and the ladder that falls out of the three gates is
the thing a ship's engineer recognises before any of the numbers:

    hub    x 1.12 (HUB_CAV)        127.8 rpm   the flow behind the cap is
                                               slowest, so the axial rope is
                                               first and it is last to go
    tips   x 1                     141.2 rpm   the helices light
    sheet  / 1.20 (SHEET_CAV_INC)  167.2 rpm   the blade's own back boils

    --view back --rps 2.05 / 2.25 / 2.55 / 3.2 walks all four rungs.

WHAT A SHEET PARCEL IS, AND WHY IT COST NO NEW MACHINERY. It is the one
population in this file that is not shed. Every other parcel is water that
crossed the disc at some time tb and has been convecting since; a sheet cavity
is a feature OF A SURFACE, and the surface is turning. So it has no age, no
history and nothing to remember: its whole state is f(seed, slot, t) at the
CURRENT t, placed in BLADE-LOCAL coordinates off the same planform, twist,
skew, rake and pivot angle the casting is lofted from, and rotated by the same
theta the mesh is drawn at. It cannot lag, because its position contains
exactly one clock. It is on the UPSTREAM face -- the blade advances toward +X,
so the face that pushes is the +X one and the BACK is the other -- and the
casting occludes it for free, because the billboards are depth-tested against
the scene. That is why the demo's own aft framings do not show it and why
`--view back` exists.

    THE EXTENT LAW IS AUTHORED AND IS THE ONLY PART THAT IS. How far the sheet
    reaches for a given excess -- SHEET_CAV_KNEE, SHEET_CAV_R0, SHEET_CAV_C --
    is a curve chosen to look right. A real extent comes from the section's own
    pressure distribution, which is the BEMT rung and is out of scope here.
    What is NOT authored is WHEN it appears (Burrill, as above) or HOW IT
    BREATHES: the sheet reads the hull's loading lobe at the blade's CURRENT
    azimuth rather than at a birth azimuth, because it rides the blade, so it
    grows through twelve o'clock and shrinks through six once per revolution.
    At 294 rpm with 5 m/s of way on the extent runs 0.00 at the bottom of the
    sweep to 0.78 at the top -- the vapour exists only over the top of the disc
    -- and at the bollard lamk is zero and every blade carries the same sheet.
    No `if` anywhere in either sentence.

THE HUB VORTEX is the single most recognisable feature of real CPP footage, and
it cavitates FIRST: the flow behind the cap is slower, the pressure lower, and
the head start is worth about 12% here (HUB_CAV). So at the default the hub
sits at 93% of its inception line against the tips' 83%, and a nudge on either
slider lights the axial rope 13 rpm before the helices. A small share of slots
is taken out of the sheet's budget and born on the axis, aft of the cap.

IT IS A CONTROLLABLE-PITCH PROPELLER, and that is the demo's second thesis.
The first (below) is that particles carry the image and a volume carries the
light. The second is that a CLOSED-FORM wake can still be INTERACTIVE and still
be honest about causality. Drag the slider: five bronze blades turn in their
flanges at once, and the wake does NOT change at once, because the wake is made
of parcels that were shed under whatever pitch was set when each of them
crossed the disc. The boundary between the old slipstream and the new one
convects away downstream at the slipstream's own speed and the old wake
persists until it ages out, one and three quarter seconds later. That lag is
the whole point -- and the boundary is also a boundary in CAVITATION, so a
throttle step lights the new wake and leaves the old one dark.

    beta   the blade angle at 0.7 R, 0..35 deg, BETA_DESIGN = 22 the neutral.
           It is a PITCH RATIO in the polynomials' own units: a blade is a
           slice of a screw thread, so P/D = 0.7 pi tan(beta), and 22 deg is
           P/D 0.889. The slider's ends (P/D 0.19 and 1.54) are outside the
           B-series validity box and the panel says so.
    omega  the shaft rate, 60..450 rpm, RPS_REF = 6 rev/s the reference.
           Tip speed 11 m/s at the default 120 rpm, 34 at 360 and 42 at the
           slider's top -- see RPM_MAX for why the top is 450 and not 600.
    V_ship the ship's speed through the water, 0..6 m/s, 0 by default. The
           model's own input is the speed of ADVANCE, V_a = V_ship (1 - w_mean)
           -- there is a hull in front of the disc now -- and it is what makes
           J = V_a / n D real, and with it eta_0.
    hist   HIST_N vec4 (omega, theta, v_i, V_a), one per frame, one full slot
           period deep, beside a vec4 (K_T/K_T_design, Burrill ratio, the wake
           lobe's amplitude, spare) at the
           same index. Written on the host at frame_no % HIST_N, read in the
           kernel at the parcel's OWN crossing frame. The split is measured
           rather than tidy: both vec2 entries are per-FRAME quantities, and
           evaluating the Burrill pow per parcel cost 1.6 ms of a 42 ms frame.

    K_T(J, P/D)  the polynomials, and the ONLY channel from either lever to
                  the flow. Thrust falls out of it, the induced velocity falls
                  out of thrust, and wake speed, funnel length, funnel WIDTH
                  and the cavitation criterion all fall out of that. The ANCHOR
                  is no longer a number somebody liked: the default helm makes
                  17.8 kN, which is a 3.69 m/s slipstream.
    swirl ~ omega K_T   circulation, linear in both
    tau_c ~ K_T         thrust loading, and INDEPENDENT of shaft speed

WHAT THE THROTTLE DOES NOT DO IS THE INTERESTING PART. Advance per revolution
is u / n, and at the bollard u ~ n, so it is INDEPENDENT OF RPM: change the
shaft speed and the helix's loop spacing in metres does not move at all. What
moves is how far 1.7 s of wake reaches (halve the rpm and the slipstream is
half as long), how fast the loops are laid down, and -- only in water --
whether they are VISIBLE at all. Loop SPACING is a property of the PITCH alone
at the bollard; put a ship speed on and it stops being, because J moves and
K_T with it. That is what a screw is: it advances its geometric pitch per turn
whatever speed you turn it
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

THE STERN IS FIRST-PARTY GEOMETRY, lofted from cross sections by the blade's
own technique at ten times the size, and there is ONE THING IT CANNOT DO: the
sprites' T_sun march samples the particle density volume and nothing else, so
the hull does not shadow the wake. Geometry-aware sprite shadowing is a
renderer feature and is out of scope here. What IS in scope is refusing to put
the hull where the missing shadow can be read, so the key was raked down to
the Snell floor and swung onto the QUARTER: every path from the slipstream to
the sun now runs up and AFT, over open water, past a ship that ends at
x = 1.06. The funnel and the first half metre of wake ARE under the counter
and are lit as though they were not; they are silt motes and the dimmest thing
in the frame. Rudders are out of scope for the same family of reason -- they
would sit IN the slipstream, and this closed form has no interaction to give
them.

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
from warp_common import cli_arg, find_ffmpeg

# 2M -> 1.7M, and it is the 25 fps floor that moved it, not taste. Fixing the
# negative-age bug (see the kernel) put a sixth of the parcels BACK into the
# frame that the rasteriser had been silently discarding as NaN, which is 5.6
# ms of work that was always supposed to be there; the underwater medium is
# another 3. Both are honest, neither is optional, and the count is the dial
# that pays for them. Nothing in the look moves with it: BRIGHT and SIGMA are
# both anchored at 2M and both go as 1/N, so total radiance and total optical
# depth are invariant and the only thing 1.7M buys less of is smoothness. See
# THE N CONTRACT below.
BENCH = "--bench" in sys.argv
SHOT = "--shot" in sys.argv
SHOT_TIME = cli_arg("--shot", 4.0, float)
FLAT = "--flat" in sys.argv
INTEROP = "--no-interop" not in sys.argv
# --film renders OFFLINE, so it takes the offline defaults: 5M parcels and
# 1920x1080. Nothing in the LOOK moves with either -- BRIGHT and SIGMA both go
# as 1/N (see THE N CONTRACT), so 5M is the same picture as 1.7M with a third
# of the speckle, which is the one thing a 55 s film cannot fake.
FILM = "--film" in sys.argv
N = cli_arg("--n", 5_000_000 if FILM else 1_700_000, int)
W, H = (cli_arg("--width", 1920 if FILM else 1280, int),
        cli_arg("--height", 1080 if FILM else 800, int))
HEADLESS = SHOT or BENCH or FILM
FPS = 60
DT = 1.0 / FPS

# ── The propeller ───────────────────────────────────────────────────────────
# Axis is +X: the disc lies in YZ at x = 0 and the wake blows toward +X. A
# point at (0, r, 0) rotated about X by theta lands at (0, r cos, r sin), which
# is exactly the kernel's (y, z) -- so the mesh's rotation.x and the kernel's
# azimuth are the same number with no sign to get wrong.
TWO_PI = 2.0 * math.pi
BLADES = cli_arg("--blades", 5, int)
R_TIP = 0.90                        # m
R_HUB = 0.27       # the bulbous hub: a 5-blade CPP carries the whole pitch
                   # mechanism inside it, so hub/diameter is ~0.30, not 0.15
R_PALM = 0.245     # radius at which a blade root meets its own flange
# ── THE BLADE'S OWN SECTION MATH, AND WHY IT LIVES UP HERE NOW ──────────────
# These six numbers used to sit beside blade_geometry, eighteen hundred lines
# down, because the loft was the only thing that had ever needed them. The
# SHEET CAVITY needs them too -- it is drawn ON the blade's suction surface, so
# it is placed by the same planform, the same thickness law, the same twist and
# the same skew the casting was lofted from -- and a second copy of a blade's
# shape is the one duplication this file cannot afford: a sheet drawn on a
# planform that has drifted from the mesh's would float off the blade, which is
# the exact failure the whole slice is graded against.
#
# So they moved to the geometry block and there is exactly one of each. The
# loft reads them (blade_geometry), the kernel reads them (through PLAN_TAB,
# resampled once onto a uniform span grid so a parcel's lookup is an index and
# not a search), and neither can drift from the other.
C_MAX = 0.50            # m, the widest chord -- ~0.75 expanded area ratio at B=5
SKEW_DEG = 20.0         # tip skew; moderate, as on the reference
RAKE_M = 0.085          # m of aft rake at the tip
SKEW_RAD = math.radians(SKEW_DEG)
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
PLAN_N = 33             # the uniform span grid the kernel indexes into
# ── R2: TWO SCREWS, AND THE HANDEDNESS THAT MAKES THEM A PAIR ───────────────
# A twin-screw ship does not carry two of the same propeller. It carries a
# MIRRORED PAIR, turning in opposite senses, and the standard arrangement is
# OUTWARD-TURNING: seen from astern the starboard screw runs clockwise and the
# port screw anticlockwise, so both blade tips are moving OUTBOARD as they pass
# top dead centre. (Outward turning keeps the blade tips out of the hull's own
# boundary layer at the top of the disc and is what nearly every twin-screw
# ship is built with.)
#
# WHICH ONE IS THE ONE THIS DEMO ALREADY HAD is a question with an answer, not
# a choice. The wake blows toward +X, so +X is ASTERN and the ship's bow is
# -X; with +Y up that makes +Z the PORT side. theta increases with time and a
# point at (0, r, 0) rotates to (0, r cos, r sin), so the existing screw's top
# moves toward +Z -- outboard, if and only if it sits on the port side. The
# screw that shipped is therefore the LEFT-HANDED, PORT screw, and the new one
# is its mirror image: right-handed, starboard, at -Z, turning -theta.
#
# Every consequence follows from ONE operation, applied at the very end of the
# kernel and to the whole assembly in the scene: reflect z. The mesh mirrors
# (with the winding flipped -- see blade_geometry, and see f2010a73 for what
# forgetting that looks like), the blade order reverses, the pitch rotation
# reverses, and in the kernel the helix, the swirl and the blade phase all
# reverse together because they are all just z of a point that was reflected.
# Nothing in the closed form has a handedness of its own to get wrong.
SHAFT_Z = cli_arg("--shaft-z", 1.30 * R_TIP, float)   # m, half the spacing
# A phase offset on the starboard shaft so the two sets of blades do not cross
# top dead centre together. Half a blade pitch is the maximum possible
# separation for identical shaft speeds, and it is what stops the pair reading
# as one ten-bladed strobe when both wakes are in frame.
PHASE_OFF = math.pi / BLADES        # rad, starboard blades vs port
# ── THE HAPPY ACCIDENT: THIS IS VERY NEARLY A WAGENINGEN B5-75 ──────────────
# The blade planform below was drawn to LOOK like a ship's CPP -- wide rounded
# trapezoids, C_MAX 0.50 m on a 0.90 m radius -- and the expanded area that
# falls out of it is ~0.75 of the disc. Five blades at EAR 0.75 IS a B5-75, the
# most thoroughly measured propeller family there is, so the Wageningen
# open-water regression applies to this prop directly and every integral
# quantity below is a LOOKUP rather than an invention. D is the one number the
# polynomials need that the demo did not already have a name for.
D_PROP = 2.0 * R_TIP                # m, diameter
D_PROP2 = D_PROP * D_PROP
D_PROP4 = D_PROP2 * D_PROP2
A_DISC = math.pi * R_TIP * R_TIP    # m2, the actuator disc
EAR = 0.75                          # expanded area ratio, A_E / A_0
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
# ── ONE NUMBER, FOUR CONSUMERS ──────────────────────────────────────────────
# The shaft centreline is y = 0, so "the surface" and "the submergence" are the
# same quantity seen from the two ends, and there is now enough hanging off it
# that repeating it would be a bug waiting to happen. WATERLINE is stated once,
# HERE, beside the head it is the head OF, and everything downstream reads it:
#
#   CAV_MARGIN                     static head in the Burrill margin (above)
#   renderer.set_fog_water_surface_y   where the murk's air stops
#   Ocean.position.y               the FFT surface itself, so the waves are on
#                                  the same plane the murk refracts the sun at
#                                  and the caustic proxy walks up to
#   HULL_TOP / the stern's stations   the ship floats on it
#
# --depth therefore moves all four together: --depth 1.2 is a shallow-draught
# boat whose screws cavitate at the default helm AND whose surface is 1.2 m over
# them, close enough that the Snell window fills the frame.
WATERLINE = DEPTH
# ── THE CRITERION IS BURRILL'S, NOT AN AUTHORED COEFFICIENT ─────────────────
# This used to be `ca = K_CAV (L omega R)^2` against the margin, with K_CAV
# picked so the default helm landed at 0.83. The shape of that was a guess: it
# made the criterion quadratic in shaft speed, because the suction went as
# omega^2 and the margin did not move. That is not how a propeller cavitates.
#
# The standard pair a naval architect actually plots is Burrill's, and BOTH of
# its axes move with the shaft:
#
#   sigma_0.7R = (p_atm + rho g h - p_v) / (0.5 rho V_R^2)   the cavitation no.
#   V_R^2      = V_a^2 + (0.7 pi n D)^2      resultant water speed at 0.7 R
#   tau_c      = T / (0.5 rho A_P V_R^2)     the thrust-loading coefficient
#
# Burrill & Emerson's diagram then draws limit lines of tau_c against sigma for
# a given percentage of BACK CAVITATION, and the line quoted everywhere as the
# upper limit for merchant-ship propellers -- 5% back cavitation -- is fitted
# by tau_c = 0.30 sigma_0.7R^0.57. That is BURRILL_5PCT below, and the panel
# quotes it as-is because it is the number an engineer recognises.
#
# INCEPTION IS NOT THAT LINE, and the demo needs inception. 5% back coverage is
# a developed sheet; the first bubbles appear well below even Burrill's 2.5%
# line, and where exactly is a function of section shape, roughness and nuclei
# content that no chart of this kind carries. So the inception line is a stated
# FRACTION of the 5% line, and that fraction is the one authored number left in
# the whole cavitation model -- chosen, as K_CAV was, so the default helm
# (120 rpm, beta 22, 3 m down) sits at 0.83 of it. Just under. Everything the
# demo is for happens in the last 17%.
#
# WHAT CHANGED WITH THE SHAPE is the demo's whole dynamic range, and it is
# worth stating because it moved every look constant below. tau_c is
# INDEPENDENT of shaft speed at the bollard (T and V_R^2 both go as n^2), and
# sigma goes as 1/n^2, so the ratio goes as n^1.14 -- not n^2. Cavitation
# arrives more gently and never gets as violent: 300 rpm is 2.36x the inception
# line where the old model said 5.19x. The excess is therefore ~3x smaller at
# every interesting helm, and CAV_KNEE and CAV_LK are divided and multiplied by
# that 3 to put the vapour back where it was. See each of them.
BURRILL_5PCT = 0.30     # tau_c = 0.30 sigma^0.57, Burrill's 5% back-cavitation
BURRILL_P = 0.57        # line (Burrill & Emerson 1963), as commonly fitted
BURRILL_INC = cli_arg("--inception", 0.610, float)   # x that line = inception
# The projected blade area, from the expanded area, by Taylor's approximation
# A_P / A_D = 1.067 - 0.229 P/D with the developed area taken as the expanded
# one. It rides the DESIGN pitch and not the parcel's: the correction swings
# 14% over the usable P/D range, which would cost the history ring a fifth slot
# it does not have, and the panel quotes this same design-pitch A_P so the
# number on screen is the number the wake was drawn with.
# The hub vortex's head start. Behind the cap the axial flow is slower and the
# static pressure lower, so the hub core reaches vapour pressure BEFORE the tips
# do -- which is why the axial rope is the first thing to appear in real CPP
# footage and the last thing to go. 1.12 puts the hub at 0.93 of the inception
# line at the default against the tips' 0.83, and under Burrill that is now a
# 13 rpm window (128 -> 141) rather than the old model's 7.
HUB_CAV = 1.12
# ── SHEET CAVITATION, AND THE IRONY OF WHAT WAS ALREADY GATING THE ROPES ────
# Burrill's diagram is a BACK-CAVITATION chart. Every line on it is a stated
# percentage of the blade's SUCTION SIDE covered by a vapour sheet, and this
# file has been using it as a proxy for the tip vortices -- the one structure
# it does not describe -- since the criterion landed. So the sheet is not a new
# model bolted on: it is the phenomenon the criterion was always about, drawn
# at last, and the tip ropes keep it as an acknowledged proxy.
#
# THE LADDER IS THE POINT. A propeller pushed off its design point does not
# light up all at once, and the order it lights up in is the thing a ship's
# engineer recognises: the hub rope first (the flow behind the cap is slowest),
# then the tip vortices, and only then does the blade's own back start to boil.
# So the sheet's inception line sits ABOVE the tips':
#
#   hub    caq x 1.12  > 1     127.8 rpm at the default pitch, 3 m down
#   tips   caq         > 1     141.2
#   sheet  caq / 1.20  > 1     167.2      <- this line
#
# 1.20 sits in the middle of the 1.15-1.25 the plan allows, and what sets it is
# that the three crossings have to be far enough apart to be a LADDER at the
# throttle's own resolution: 128 / 141 / 167 rpm is two windows of 13 and 26
# rpm, both of which a slider can be parked in.
SHEET_CAV_INC = cli_arg("--sheet-inception", 1.20, float)   # x the tips' line
# 0.45 -> 0.15: the excess this saturates against is ~3x smaller under Burrill
# (see above), so the knee divides by 3 to leave the vapour's brightness at
# 180 and 300 rpm within a percent of where it was.
CAV_KNEE = 0.15      # excess at which the vapour is 63% of its full brightness
CAV_R = 2.6          # x core radius at full vapour: a fat rope, not a wire
# How long the vapour SURVIVES, as a fraction of LIFETIME, against the excess.
# See COLLAPSE in the docstring: this is the length knob and it is deliberately
# not the intensity knob, and it is the LEVER'S OWN RANGE that sets the slope.
# 0.50 saturated at e ~ 1.7 -- 230 rpm -- which put every interesting helm
# setting at "the rope runs off the frame" and made the collapse invisible at
# anything you would actually want to look at. 0.30 saturated at e ~ 2.9 and
# spread the useful part over the 130-240 rpm band, where a rope that ENDS is
# the whole point -- and 0.90 is that same band under Burrill, whose excess is
# 3x smaller. The check is that the two documented shots come out where they
# were: --rps 3 gives 0.43 of a life (it was 0.40) and --rps 5 saturates at
# 1.00 (it did too). The slope moved so the picture would not.
CAV_L0, CAV_LK = 0.14, 0.90
CAV_COLLAPSE = 0.30  # the last 30% of that life is the condensation ramp
# ── CONTROLLABLE SHAFT SPEED ────────────────────────────────────────────────
# RPS_REF is the ANCHOR and not the default: every flow coupling below is
# normalised by it, so a prop turning at RPS_REF with beta at BETA_DESIGN
# evaluates to exactly the numbers this demo had before either lever existed.
# RPS is merely where the throttle starts.
RPS_REF = 6.0                       # rev/s, the tuning reference
RPS_OP = 2.0                        # rev/s: the helm the LOOK is anchored at.
                                    # Defined here rather than beside OMG_OP
                                    # because the B-series anchor below needs
                                    # it, and there must be exactly one of it.
# 2 rev/s = 120 rpm is where the DEMO starts, and it is an honest ship speed:
# a large slow-speed diesel turns its screw at 60-120 rpm and this is the top
# of that. It is also, by construction, the operating point that sits just
# under inception -- see K_CAV.
RPS = cli_arg("--rps", 2.0, float)  # revolutions per second commanded
OMEGA = 2.0 * math.pi * RPS
OMEGA_REF = 2.0 * math.pi * RPS_REF
# THE SLIDER'S RANGE, AND ITS TOP IS A PHYSICAL LIMIT RATHER THAN A ROUND
# NUMBER. Tip speed is pi D n, so on this 1.8 m screw 600 rpm is 57 m/s at the
# tips -- past anything a marine propeller is turned at, and far enough past it
# that every number the panel prints up there is an extrapolation of the
# B-series regression into water no ship has ever put a blade through. 450 rpm
# is 42 m/s, which is still hard driving for a workboat and is the top of the
# band the model can be held to.
RPM_MIN, RPM_MAX = 60.0, 450.0      # the slider's range
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
# The screw the loft is a slice of: P = 2 pi (0.7 R) tan(BETA_DESIGN), so
# beta(r) = atan(P / 2 pi r) is the twist law and it is exact at 0.7 R by
# construction. ONE statement of it, because the mesh and the sheet cavity are
# both placed by it and a blade whose sheet sits at a different twist is a
# sheet floating off the blade.
P_SCREW = TWO_PI * 0.7 * R_TIP * math.tan(math.radians(BETA_DESIGN))
BETA_MIN, BETA_MAX = 0.0, 35.0      # the slider's range
# THE PITCH SLIDER IS NOW P/D, in the only units the polynomials speak. A
# propeller blade is a slice of a screw thread, so a blade angle at 0.7 R and a
# pitch ratio are the same statement twice: P = 2 pi (0.7 R) tan(beta), and
# dividing by D = 2R gives P/D = 0.7 pi tan(beta). BETA_DESIGN 22 deg is
# P/D 0.889, comfortably inside the B-series box; the slider's own ends are
# not, which is what R5's validity label is for.
BETA_0 = 2.0    # deg: the no-lift angle. It used to be the foot of an authored
                # smoothstep; it is now the P/D at which the extrapolation
                # below sends K_T to zero, which is the same claim in the same
                # place -- a blade at its no-lift angle makes no thrust.
# Floors, and they earn their keep. U_FLOOR keeps a feathered prop's wake
# CREEPING rather than stacking every parcel of every slot on the disc plane
# in one bright annulus, which is what a hard zero does -- and it is now a
# fraction of TIP SPEED rather than of an authored loading, because the honest
# chain really does return u_wake = 0 at zero pitch and something has to be
# left over. G_FLOOR is the matching brightness floor -- and it has to be well
# BELOW the speed floor, because parcels per unit length goes as 1/u.
U_FLOOR = 0.02                      # x tip speed: the creep of a dead screw
G_FLOOR = 0.05
IN_G_FLOOR = 0.06
PITCH = cli_arg("--pitch", BETA_DESIGN, float)      # headless: the pitch held
PITCH_TO = cli_arg("--pitch-to", PITCH, float)      # ... and the step it takes
PITCH_AT = cli_arg("--pitch-at", 1.0e9, float)      # ... at this sim time


def pd_of(beta_deg):
    """P/D from the blade angle at 0.7 R. The blade IS a screw thread."""
    return 0.7 * math.pi * math.tan(math.radians(max(beta_deg, 0.0)))


PD_DESIGN = pd_of(BETA_DESIGN)      # 0.8885
# ── R5: THE VALIDITY BOX, AND WHAT HAPPENS AT ITS EDGES ─────────────────────
# The Wageningen regression was fitted over J in [0, 1.5] and P/D in [0.5, 1.4]
# and it is a POLYNOMIAL: outside that box it does not degrade, it diverges.
# Both of this demo's own sliders leave it -- beta 5 deg is P/D 0.19 and beta
# 35 deg is P/D 1.54 -- so the inputs are clamped to the box edge and the panel
# says so, in words, for as long as anything is clamped. Never silently.
PD_LO, PD_HI = 0.5, 1.4
J_LO, J_HI = 0.0, 1.5
# BELOW the box there is one thing the polynomial cannot say and the physics
# can: a blade at its no-lift angle makes no thrust at all, and the clamped
# edge value (K_T 0.21 at P/D 0.5) is not that. So under the edge K_T is faded
# LINEARLY to zero at P/D(BETA_0) -- the file's own no-lift angle -- which
# keeps the feather beat alive and is stated as an extrapolation wherever it is
# used. K_Q is NOT faded: it is left at the edge value, an overestimate that
# keeps a feathered shaft absorbing power rather than none, which is the
# conservative direction for a torque readout.
PD_ZERO = pd_of(BETA_0)             # 0.0768


def scheduled_beta(t):
    """Headless pitch: hold PITCH, step to PITCH_TO at PITCH_AT. A hard step
    and not a servo ramp, deliberately -- the acceptance evidence is the
    BOUNDARY between old and new wake convecting downstream, and a ramp smears
    the one edge the shots exist to show."""
    return PITCH_TO if t >= PITCH_AT else PITCH

# ── THE THIRD LEVER: SHIP SPEED, WHICH IS WHAT MAKES J REAL ─────────────────
# Everything above this line ran at the BOLLARD -- a screw turning in water
# that is not going anywhere -- and at the bollard the advance ratio J is zero,
# which means the open-water diagram is being read at one end of its own x
# axis and the efficiency is zero by definition. That is a perfectly honest
# operating point (a tug on a tow, a ship on trials at a wall) and it stays the
# DEFAULT, because it is the framing every shot and the whole film were
# composed at. But a propeller demo that cannot move the ship cannot answer
# "what is eta_0", so:
#
#   V_a   the speed of advance: the water's own speed into the disc, which on
#         a real hull is the ship's speed reduced by the wake fraction. Here
#         there is no hull, so it is simply the freestream.
#   J     V_a / (n D), the advance ratio, and the x axis of every open-water
#         diagram ever drawn.
#
# It advects BOTH legs of the closed form: the wake leaves at V_a + 2 v_i
# instead of 2 v_i, and the funnel's far-upstream drift becomes the freestream
# instead of a fraction of the disc speed. The funnel's MOUTH follows from
# continuity, A_inf = A_disc u_disc / V_a -- which is infinite at the bollard,
# exactly the flare the shipped demo already had, and closes down to something
# barely wider than the disc once the ship is moving.
VA_MIN, VA_MAX = 0.0, 6.0
V_SHIP = cli_arg("--speed", 0.0, float)     # m/s through the water; 0 = bollard

# ── R1: THE HULL'S WAKE, AND THE ONCE-PER-REV IT CAUSES ─────────────────────
# There is a hull over these propellers now, and the single most consequential
# thing a hull does to a screw is not to hide it: it drags a BOUNDARY LAYER,
# and the disc turns inside it. The water arriving at the top of the disc --
# nearest the hull -- has been slowed most; the water at the bottom is nearly
# freestream. So the blade sweeps through a circumferentially varying advance
# once every revolution, loads up at twelve o'clock, and a screw that is near
# its threshold BREATHES: vapour blooms at the top of the disc, is shed there,
# and corkscrews away downstream as a bright band on an otherwise even rope.
# That is what cavitation next to a hull looks like in every photograph of it,
# and a demo that put a hull over perfectly steady cavitation would be
# advertising its own limits.
#
#   w(theta) = w_mean + w_peak * lobe(theta)      the wake fraction
#   lobe     = ((1 + cos theta) / 2)^p - <that>   ZERO MEAN, peaked at TDC
#   V_a      = V_ship (1 - w_mean)                the SPEED OF ADVANCE
#
# The zero mean is the whole discipline of it. w_mean is then the honest mean
# wake fraction and V_a is the honest speed of advance, so the B-series
# evaluation -- J, K_T, K_Q, T, Q, eta_0, the diagram and every readout hung
# off them -- is done on the MEAN and stays exactly as citable as it was. The
# peak term is a redistribution around that mean and is spent in ONE place: a
# per-parcel loading factor lambda(theta_birth) on the cavitation criterion.
# It never touches the momentum chain. If it did, the thrust readout would
# stop matching the diagram the panel draws it on.
#
# W_MEAN 0.15 is a twin-screw stern's own number and not a single-screw one:
# the shafts here run out well off the centreline, in water the hull has
# disturbed far less than the water directly behind a single screw's deadwood
# (a single-screw ship is 0.25-0.35, a twin with open shafts 0.10-0.20).
# W_PEAK 0.30 puts the top of the disc at w 0.37 and the bottom at 0.07, which
# is the kind of range a circumferential wake survey of a fine twin stern
# actually draws.
W_MEAN = cli_arg("--wake-frac", 0.15, float)    # mean wake fraction, hull
W_PEAK = cli_arg("--wake-peak", 0.30, float)    # its peak-to-mean swing at TDC
W_LOBE_P = 4.0          # how tightly the deficit is packed around top-dead-centre
# The mean of ((1 + cos)/2)^p over the circle, which is what makes the peak
# term integrate to zero. Closed form for even p; the quadrature is exact to
# float and costs one import-time loop, which is cheaper than being wrong.
W_LOBE_MEAN = sum((0.5 * (1.0 + math.cos(TWO_PI * k / 2048))) ** W_LOBE_P
                  for k in range(2048)) / 2048.0        # 0.2734 at p = 4
# The speed of advance: the ship's speed reduced by the mean wake fraction.
# THIS is what J is built on, and it is the standard definition of advance
# behind a hull. The panel says so, in words, beside the number.
V_A = V_SHIP * (1.0 - W_MEAN)

# ── THE WAGENINGEN B-SERIES OPEN-WATER POLYNOMIALS ──────────────────────────
# Oosterveld, M.W.C. and van Oossanen, P., "Further Computer-Analyzed Data of
# the Wageningen B-Screw Series", International Shipbuilding Progress, Vol. 22,
# No. 251, 1975, pp. 251-262. K_T is 39 terms and K_Q is 47, each of the form
#
#     C * J^s * (P/D)^t * (A_E/A_0)^u * Z^v
#
# fitted to the Wageningen towing-tank measurements at Rn = 2e6. The
# coefficients are a page of public-domain numbers, so they are hard-coded
# here as first-party tables: no asset, no dependency, no licence.
#
# TRANSCRIPTION IS THE WHOLE RISK. A wrong digit in a 39-term polynomial does
# not crash and does not look wrong; it quietly shifts a curve. So the tables
# are checked against the published B5-75 open-water diagram before anything
# reads them -- run `--openwater` and the demo prints the check and exits.
# The gate the numbers had to pass: K_T(J=0) at P/D 0.889 in 0.40-0.50,
# eta_0 peaking at 0.60-0.65, and the zero-thrust J landing where the chart
# puts it for every P/D on the sheet. See the table --openwater prints.
#
# EAR and Z ARE CONSTANTS HERE, so the (A_E/A_0)^u Z^v part of every term is
# folded into the coefficient once at import and the per-evaluation work is
# 86 multiply-adds over J and P/D alone.
_KT_TERMS = (
    (+0.00880496, 0, 0, 0, 0), (-0.204554, 1, 0, 0, 0),
    (+0.166351, 0, 1, 0, 0), (+0.158114, 0, 2, 0, 0),
    (-0.147581, 2, 0, 1, 0), (-0.481497, 1, 1, 1, 0),
    (+0.415437, 0, 2, 1, 0), (+0.0144043, 0, 0, 0, 1),
    (-0.0530054, 2, 0, 0, 1), (+0.0143481, 0, 1, 0, 1),
    (+0.0606826, 1, 1, 0, 1), (-0.0125894, 0, 0, 1, 1),
    (+0.0109689, 1, 0, 1, 1), (-0.133698, 0, 3, 0, 0),
    (+0.00638407, 0, 6, 0, 0), (-0.00132718, 2, 6, 0, 0),
    (+0.168496, 3, 0, 1, 0), (-0.0507214, 0, 0, 2, 0),
    (+0.0854559, 2, 0, 2, 0), (-0.0504475, 3, 0, 2, 0),
    (+0.010465, 1, 6, 2, 0), (-0.00648272, 2, 6, 2, 0),
    (-0.00841728, 0, 3, 0, 1), (+0.0168424, 1, 3, 0, 1),
    (-0.00102296, 3, 3, 0, 1), (-0.0317791, 0, 3, 1, 1),
    (+0.018604, 1, 0, 2, 1), (-0.00410798, 0, 2, 2, 1),
    (-0.000606848, 0, 0, 0, 2), (-0.0049819, 1, 0, 0, 2),
    (+0.0025983, 2, 0, 0, 2), (-0.000560528, 3, 0, 0, 2),
    (-0.00163652, 1, 2, 0, 2), (-0.000328787, 1, 6, 0, 2),
    (+0.000116502, 2, 6, 0, 2), (+0.000690904, 0, 0, 1, 2),
    (+0.00421749, 0, 3, 1, 2), (+0.0000565229, 3, 6, 1, 2),
    (-0.00146564, 0, 3, 2, 2),
)
_KQ_TERMS = (
    (+0.00379368, 0, 0, 0, 0), (+0.00886523, 2, 0, 0, 0),
    (-0.032241, 1, 1, 0, 0), (+0.00344778, 0, 2, 0, 0),
    (-0.0408811, 0, 1, 1, 0), (-0.108009, 1, 1, 1, 0),
    (-0.0885381, 2, 1, 1, 0), (+0.188561, 0, 2, 1, 0),
    (-0.00370871, 1, 0, 0, 1), (+0.00513696, 0, 1, 0, 1),
    (+0.0209449, 1, 1, 0, 1), (+0.00474319, 2, 1, 0, 1),
    (-0.00723408, 2, 0, 1, 1), (+0.00438388, 1, 1, 1, 1),
    (-0.0269403, 0, 2, 1, 1), (+0.0558082, 3, 0, 1, 0),
    (+0.0161886, 0, 3, 1, 0), (+0.00318086, 1, 3, 1, 0),
    (+0.015896, 0, 0, 2, 0), (+0.0471729, 1, 0, 2, 0),
    (+0.0196283, 3, 0, 2, 0), (-0.0502782, 0, 1, 2, 0),
    (-0.030055, 3, 1, 2, 0), (+0.0417122, 2, 2, 2, 0),
    (-0.0397722, 0, 3, 2, 0), (-0.00350024, 0, 6, 2, 0),
    (-0.0106854, 3, 0, 0, 1), (+0.00110903, 3, 3, 0, 1),
    (-0.000313912, 0, 6, 0, 1), (+0.0035985, 3, 0, 1, 1),
    (-0.00142121, 0, 6, 1, 1), (-0.00383637, 1, 0, 2, 1),
    (+0.0126803, 0, 2, 2, 1), (-0.00318278, 2, 3, 2, 1),
    (+0.00334268, 0, 6, 2, 1), (-0.00183491, 1, 1, 0, 2),
    (+0.000112451, 3, 2, 0, 2), (-0.0000297228, 3, 6, 0, 2),
    (+0.000269551, 1, 0, 1, 2), (+0.00083265, 2, 0, 1, 2),
    (+0.00155334, 0, 2, 1, 2), (+0.000302683, 0, 6, 1, 2),
    (-0.0001843, 0, 0, 2, 2), (-0.000425399, 0, 3, 2, 2),
    (+0.0000869243, 3, 3, 2, 2), (-0.0004659, 0, 6, 2, 2),
    (+0.0000554194, 1, 6, 2, 2),
)


def _fold(terms):
    """(C, s, t, u, v) -> (C * EAR^u * Z^v, s, t), EAR and Z being fixed."""
    return tuple((c * EAR ** u * float(BLADES) ** v, s, t)
                 for c, s, t, u, v in terms)


_KT = _fold(_KT_TERMS)
_KQ = _fold(_KQ_TERMS)


def _poly(folded, j, pd):
    tot = 0.0
    for c, s, t in folded:
        tot += c * j ** s * pd ** t
    return tot


def open_water(j, pd):
    """K_T, K_Q at this advance ratio and pitch ratio, plus whether either
    input had to be clamped to the validity box. The clamp is R5: the
    polynomials are a REGRESSION and outside their box they diverge, so the
    caller is told and the panel says so on screen."""
    jc = min(max(j, J_LO), J_HI)
    pdc = min(max(pd, PD_LO), PD_HI)
    kt, kq = _poly(_KT, jc, pdc), _poly(_KQ, jc, pdc)
    if pd < PD_LO:
        # Below the box: fade the thrust to zero at the no-lift pitch. See
        # PD_ZERO. K_Q keeps the edge value.
        kt *= max((pd - PD_ZERO) / (PD_LO - PD_ZERO), 0.0)
    return kt, kq, (jc != j or pdc != pd)


class PropState:
    """Everything the model says about one helm setting, in one object.

    THE CHAIN RUNS ONE WAY AND ONLY ONE WAY: the polynomials give K_T and K_Q,
    those give thrust and torque, and thrust gives the induced velocity through
    the actuator disc's own momentum balance. Nothing downstream is tuned
    independently any more -- the wake's speed, the funnel's length, the
    helix's advance per revolution and the cavitation criterion are all
    functions of T, and T is a function of the diagram.

        T   = K_T rho n^2 D^4          Q  = K_Q rho n^2 D^5
        P_D = 2 pi n Q                 eta_0 = J K_T / (2 pi K_Q)
        v_i = (-V_a + sqrt(V_a^2 + 2 T / (rho A))) / 2
        u_disc = V_a + v_i             u_wake = V_a + 2 v_i

    The induced velocity is the momentum inversion WITH advance, which reduces
    to the familiar static sqrt(T / 2 rho A) at V_a = 0 and is the reason the
    wake speeds up rather than doubling when the ship starts moving.

    v_i CAN COME OUT NEGATIVE and that is not a bug: past the zero-thrust J the
    screw is braking, and a braking disc leaves the water behind it SLOWER than
    the freestream. The wake then convects at V_a + 2 v_i, which the kernel
    floors so parcels still leave the disc rather than piling up on it."""

    __slots__ = ("n", "va", "beta", "pd", "j", "kt", "kq", "thrust", "torque",
                 "power", "eta", "vi", "u_disc", "u_wake", "extrapolated")

    def __init__(self, rpm, beta_deg, va):
        self.n = max(rpm, 0.0) / 60.0
        self.va = max(va, 0.0)
        self.beta = beta_deg
        self.pd = pd_of(beta_deg)
        self.j = self.va / (self.n * D_PROP) if self.n > 1.0e-4 else 0.0
        self.kt, self.kq, self.extrapolated = open_water(self.j, self.pd)
        rn2d4 = RHO * self.n * self.n * D_PROP4
        self.thrust = self.kt * rn2d4
        self.torque = self.kq * rn2d4 * D_PROP
        self.power = TWO_PI * self.n * self.torque
        self.eta = (self.j * self.kt / (TWO_PI * self.kq)
                    if self.kq > 1.0e-6 and self.kt > 0.0 else 0.0)
        rad = self.va * self.va + 2.0 * self.thrust / (RHO * A_DISC)
        self.vi = 0.5 * (-self.va + math.sqrt(max(rad, 0.0)))
        self.u_disc = self.va + self.vi
        self.u_wake = self.va + 2.0 * self.vi

    def creep(self):
        """The kernel's own speed floor, mirrored so the banner and the panel
        quote the number the wake was actually drawn with."""
        umin = U_FLOOR * TWO_PI * self.n * R_TIP + 0.10 * self.va
        return max(self.u_disc, umin), max(self.u_wake, umin)


# The anchor: everything the LOOK is calibrated against is quoted here, at the
# demo's own default helm and at the bollard. K_T_DESIGN normalises the
# kernel's loading term to exactly 1 at the design pitch, which is what keeps
# G_FLOOR and IN_G_FLOOR meaning what they meant before the polynomials landed.
OP = PropState(RPS_OP * 60.0, BETA_DESIGN, 0.0)
KT_DESIGN = OP.kt                   # 0.4133
U_DISC_OP, U_WAKE_OP = OP.u_disc, OP.u_wake     # 1.847, 3.693 m/s


def projected_area(pd):
    """Taylor's approximation A_P / A_D = 1.067 - 0.229 P/D, with the developed
    area taken as the expanded one. It is evaluated at the LIVE pitch because
    the criterion it feeds is computed once per frame on the host -- per parcel
    it would have cost a ring slot nobody had, and the design-pitch value is
    14% out at the ends of the slider."""
    return EAR * A_DISC * (1.067 - 0.229 * min(max(pd, PD_LO), PD_HI))


def burrill(st):
    """(sigma_0.7R, tau_c, tau_c at the 5% line, ratio to the INCEPTION line).

    The ratio is the number the wake is gated on and the number the panel draws
    its bar from: > 1 is cavitating. Both of Burrill's axes are computed from
    quantities this file already has -- the pressure margin the water block
    defines, the resultant speed at 0.7 R, and the thrust the polynomials gave
    -- so "inception at X rpm at 3 m depth" is now a sentence derived from
    named quantities rather than a coefficient chosen to make it true."""
    om = TWO_PI * st.n
    vr2 = st.va * st.va + (0.7 * om * R_TIP) ** 2
    if vr2 < 1.0e-6:
        return 0.0, 0.0, 0.0, 0.0
    sigma = 2.0 * CAV_MARGIN / vr2
    tau_c = st.thrust / (0.5 * RHO * projected_area(st.pd) * vr2)
    line5 = BURRILL_5PCT * sigma ** BURRILL_P
    return sigma, tau_c, line5, max(tau_c, 0.0) / (BURRILL_INC * line5)


def wake_lobe_k(st):
    """The breathing's one number: lambda(theta) = 1 + k * lobe(theta).

    A blade section at birth azimuth theta sees J_local = J (1 - w(theta)) /
    (1 - w_mean), and what that does to its loading is dK_T/dJ times the
    difference. Linearised about the operating point, the whole modulation
    collapses to a single per-frame scalar:

        k = -(dK_T/dJ) * J * w_peak / ((1 - w_mean) K_T)

    and lambda multiplies the thrust loading in Burrill's tau_c, which is what
    the vapour is gated on. THE BOLLARD CONTROL IS BUILT IN AND NOT AUTHORED:
    k is proportional to J, so at V_ship = 0 it is exactly zero and the ropes
    do not breathe at all. A boundary-layer shadow needs way through the water,
    and this is that sentence in arithmetic rather than in an `if`.

    dK_T/dJ is central-differenced on the same polynomial the wake is drawn
    from -- 78 more multiply-adds once per FRAME, against the 1.7 million
    per-parcel evaluations the helm ring exists to avoid.

    THE CLAMP IS REAL: past the zero-thrust J, K_T goes through zero and the
    ratio diverges. A screw that close to windmilling is not cavitating at all
    (tau_c has gone to zero with it), so the clamp is never the operating
    regime -- it is there so a slider dragged to the end of its travel cannot
    put a NaN in the ring."""
    if st.j <= 1.0e-6 or st.kt <= 1.0e-4:
        return 0.0
    jc = min(max(st.j, J_LO), J_HI)
    pdc = min(max(st.pd, PD_LO), PD_HI)
    hi, lo = min(jc + 0.01, J_HI), max(jc - 0.01, J_LO)
    dkt = (_poly(_KT, hi, pdc) - _poly(_KT, lo, pdc)) / max(hi - lo, 1.0e-6)
    return min(max(-dkt * st.j * W_PEAK / ((1.0 - W_MEAN) * st.kt), 0.0), 1.5)


def inception_rpm(beta_deg, va, boost=1.0):
    """The shaft speed at which the criterion above crosses 1, by bisection.
    A derived sentence, not a tuned constant: this is what R4 bought.

    `boost` is which of the three structures is being asked about, and it is
    the SAME factor the kernel multiplies the criterion by: HUB_CAV for the
    axial rope, 1 for the tip filaments, 1 / SHEET_CAV_INC for the blade
    sheet. There is no second gate anywhere -- three numbers on one line."""
    lo, hi = 1.0, 4000.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if burrill(PropState(mid, beta_deg, va))[3] * boost < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if "--openwater" in sys.argv:
    # ── THE GATE THE COEFFICIENTS HAD TO PASS ───────────────────────────────
    # Printed from the same tables the wake reads, so this is a check OF the
    # demo and not a check beside it. Compare against any published B5-75
    # open-water diagram.
    print(f"Wageningen B{BLADES}-{EAR * 100:.0f} open water, "
          f"{len(_KT)} K_T terms / {len(_KQ)} K_Q terms\n"
          f"D {D_PROP:g} m, EAR {EAR:g}, Z {BLADES}\n")
    for pd in (0.6, 0.8, PD_DESIGN, 1.0, 1.2, 1.4):
        best_e, best_j, jz = 0.0, 0.0, 0.0
        for i in range(1, 1601):
            jj = i * 0.001
            k_t, k_q, _ = open_water(jj, pd)
            if k_q > 1e-6 and k_t > 0.0:
                e = jj * k_t / (TWO_PI * k_q)
                if e > best_e:
                    best_e, best_j = e, jj
            if jz == 0.0 and k_t <= 0.0:
                jz = jj
        k_t0 = open_water(0.0, pd)[0]
        print(f"  P/D {pd:5.3f}   K_T(J=0) {k_t0:6.4f}   "
              f"peak eta_0 {best_e:6.4f} at J {best_j:5.3f}   "
              f"K_T = 0 at J {jz:5.3f}")
    print(f"\n  at the default helm ({RPS_OP * 60:g} rpm, beta {BETA_DESIGN:g}, "
          f"V_a 0, {DEPTH:g} m):")
    sg, tc, l5, rr = burrill(OP)
    print(f"    J {OP.j:.3f}  P/D {OP.pd:.4f}  K_T {OP.kt:.4f}  "
          f"10 K_Q {10 * OP.kq:.4f}  eta_0 {OP.eta:.4f}\n"
          f"    T {OP.thrust / 1e3:.2f} kN  Q {OP.torque / 1e3:.2f} kNm  "
          f"P_D {OP.power / 1e3:.1f} kW\n"
          f"    v_i {OP.vi:.3f}  u_disc {OP.u_disc:.3f}  "
          f"u_wake {OP.u_wake:.3f} m/s\n"
          f"    sigma_0.7R {sg:.3f}  tau_c {tc:.4f}  "
          f"Burrill 5% line {l5:.4f}  -> {rr:.3f} x inception\n"
          f"    INCEPTION: hub {inception_rpm(BETA_DESIGN, 0.0, HUB_CAV):.1f} rpm, "
          f"tips {inception_rpm(BETA_DESIGN, 0.0, 1.0):.1f} rpm")
    sys.exit(0)

# ── The wake, AND THE VELOCITIES THAT BECAME HONEST ─────────────────────────
# It used to be anchored at RPS_REF by two authored constants -- U_DISC 3.86
# and U_WAKE 6.0 -- which made 1.29 and 2.00 m/s at the default helm, "the
# slipstream a large screw actually throws at the bollard" as judged by eye
# over four revisions. The eye was low by a factor of 1.85, and the whole of
# this block is what putting that right cost.
# U_DISC 3.86 and U_WAKE 6.0 at the reference are GONE. They were authored, and
# what replaces them is the momentum inversion on the polynomials' own thrust:
# at the default helm B5-75 makes 17.8 kN, which is v_i = 1.85 m/s, a disc
# velocity of 1.85 and a developed slipstream of 3.69. The tuned numbers were
# 1.29 and 2.00 -- so the water is moving 1.85x faster than four revisions of
# eyeballing had it, and nothing about the picture may move with it.
#
# THE FIX IS ONE FACTOR APPLIED EVERYWHERE, and it is arithmetic rather than
# taste. Every constant in this block is a LENGTH divided by a SPEED:
#
#   LIFETIME  = REACH / u_wake          how far the wake runs before it ages out
#   T_IN      = IN_REACH * dnm / u_disc how far ahead of the disc the funnel starts
#   TAU_A/C/S = (an ease length) / u_wake
#
# So the lengths are what get authored -- they are what the camera sees and what
# the latched box was cut for -- and the times fall out of them at whatever
# speed the model says. LIFETIME lands at 1.30 s where it was 2.4, T_IN at 0.56
# where it was 0.80, and the wake's SHAPE, its grain, its burst, its turbulence
# and its collapse are all keyed on age FRACTIONS and so are untouched to the
# last pixel. What changed is that the water now moves at a speed a naval
# architect would recognise, and the box did not have to move at all: reach is
# 4.80 m and the funnel 0.77 m, exactly the numbers the box was latched around,
# so BOX_VOX_RATIO below re-derives to the same value it had. That was the
# point of authoring the lengths.
#
# THE ONE VISIBLE CONSEQUENCE is the helix's advance per revolution, which is
# u_wake / n and therefore rose with u_wake: 1.85 m/rev against the tuned 1.00.
# The loops are twice as far apart and there are half as many of them in the
# same four metres. That is not a regression -- it is the geometric pitch
# (P = 1.60 m) plus the slipstream's own acceleration past it, which is what a
# screw at the bollard actually lays down, and the tuned 1.00 m was simply half
# of the truth.
# ── AND THE FIRST RETUNE WAS WRONG, WHICH IS THE INTERESTING PART ──────────
# Setting LIFETIME = REACH / u_wake and stopping there gave a wake of exactly
# the right length that read as a WIRE CAGE: the helix's advance per revolution
# is u / n, so at 3.69 m/s it is 1.85 m and the frame held 2.6 turns of it
# where the tuned version held 4.8. Same reach, half the loops, and a column
# you can see straight through -- which is the failure the SHEET block below
# was written about.
#
# The fix was not to put the speed back. It was that u_wake is the velocity
# INFINITELY FAR DOWNSTREAM, and the shipped TAU_A reached it in 26 cm. A real
# slipstream takes a couple of diameters to accelerate and contract, and this
# demo draws 2.7 diameters in total -- so nearly the whole visible wake is
# still developing, and it convects at something much closer to the DISC
# velocity than to the far-field one. Authoring the development LENGTH instead
# of a time constant is what states that:
#
#   the loops leave the blade at u_disc / n = 0.92 m apart and open out to
#   1.7 m by the end of the frame -- a helix that STRETCHES downstream, which
#   is what an accelerating slipstream is and what the constant-pitch tuned
#   version could not draw at all.
#
# LIFETIME then has to be SOLVED rather than divided, because the reach is the
# integral of a velocity that is not constant. It comes out at 1.70 s.
REACH = 4.80             # m of slipstream the framing was composed around
IN_REACH = 0.774         # m the funnel stands ahead of the disc, ditto
EASE_L = 2.0 * D_PROP    # m: the slipstream's acceleration length, ~2 diameters
CONTRACT_L = 1.0 * D_PROP    # m: and its contraction length, ~1
SWIRL_L = 0.80           # m over which the swirl decays away
TAU_A = EASE_L / U_WAKE_OP          # 0.97 s
TAU_C = CONTRACT_L / U_WAKE_OP      # 0.49 s
TAU_S = SWIRL_L / U_WAKE_OP         # 0.22 s


def wake_reach(u_d, u_w, life):
    """How far a parcel gets in `life` seconds under the kernel's own axial
    law -- the same expression, on the host, so the banner does not have to
    guess. x = u_w tau - (u_w - u_d) TAU_A (1 - exp(-tau / TAU_A))."""
    return u_w * life - (u_w - u_d) * TAU_A * (1.0 - math.exp(-life / TAU_A))


def _solve_lifetime():
    lo, hi = 0.05, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if wake_reach(U_DISC_OP, U_WAKE_OP, mid) < REACH:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


LIFETIME = _solve_lifetime()        # 1.70 s; also the slot recycle period
# CONTRACT IS DERIVED PER PARCEL NOW and is not a constant at all -- see the
# kernel. Mass conservation across the disc says r_wake / r_0 = sqrt(u_disc /
# u_wake), which at the bollard is exactly 1/sqrt(2) = 0.707 against the 0.78
# that was tuned by eye, and which INVERTS into an expansion once the ship is
# moving fast enough for the screw to be braking. It costs one sqrt.
# SWIRL_F is a RATE and so scales the other way: to keep the same angular twist
# per METRE against a slipstream that now averages REACH / LIFETIME = 2.82 m/s
# where the authored one was a flat 2.00, the rate rises by that ratio.
# 0.22 -> 0.31, and the wake winds as tightly per metre of tube as it did.
SWIRL_F = 0.31           # self-induced rotation of the helix, as a fraction of OMEGA
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
# TIP_HINT shipped at 0.09 -- a visibility crutch so the helix stayed a
# suggestion below inception -- and a propeller specialist on the receiving
# end read it, correctly, as smoke coming off the blades. Real sub-inception
# footage shows clean water and nothing else, so the default is now 0.0 and
# the sediment inflow alone carries the "it is pulling" story. The flag
# remains for anyone who wants the geometry legible while tuning.
#
# BOTH ARE RELATIVE TO THE RE-ANCHORED BRIGHT (see OMG_OP), so 1.0 is the
# radiance the air version's tip rope had at ITS default. A full vapour core is
# a little brighter than that and a clear one is a tenth of it, which is the
# whole dynamic range this demo works in. The first cut had VAP_GAIN at 2.3 --
# set before the re-anchor and therefore 5x over -- and the 300 rpm frame came
# out as SAND: each parcel bright enough to be an individual visible dot, which
# no amount of grain growth fixes because the problem is contrast per sample.
TIP_HINT = cli_arg("--tip-hint", 0.0, float)
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
# ── THE BLADE SHEET, ALSO OUT OF THE SHEET'S BUDGET ─────────────────────────
# The one population in this file that is NOT shed. Every other parcel is a
# piece of water that crossed the disc at some time tb and has been convecting
# ever since; a sheet cavity is a feature OF THE BLADE, welded to a surface
# that is turning, and it has no age, no history and nothing to remember. Its
# whole state is f(seed, slot, t) with t the CURRENT time -- which is what the
# closed form was built to make cheap, and is why R1 costs no new machinery:
# the blade's pose at t is already in the ring (theta is host-integrated), the
# blade's SHAPE is already in the planform tables, and the criterion is already
# on the helm ring. The parcel is placed in blade-local coordinates and rotated
# by the same azimuth the mesh is drawn at. No lag is possible, because there
# is no second clock for it to lag against.
#
# THE SUCTION SIDE IS THE UPSTREAM FACE and getting it wrong is the one error
# this slice's audience spots without reading anything else. The blade advances
# toward +X (that is what makes the wake blow that way), so the face that does
# the pushing -- the PRESSURE side, the "face" -- is the +X one, and the BACK,
# where the pressure drops and the water boils, is the -X one. In the loft's
# own section coordinates that is negative thickness, which is the sign on
# th_s below and the only place the choice is made.
#
# ── AND THE CASTING OCCLUDES IT, WHICH IS FREE ─────────────────────────────
# The billboards are depth-TESTED against the scene's own depth (reverse-Z,
# GREATER_OR_EQUAL, depth write off -- VulkanCorePipelines.cpp), so a sheet on
# the far face of a blade is rejected by the bronze in front of it exactly as a
# mesh would be. Nothing here has to know where the camera is: the sheet is put
# on the surface it belongs to, offset a few millimetres clear of the casting,
# and which blades show it is the rasteriser's business. That is also why the
# demo's own aft framings do NOT show it and --view back exists.
SHEET_CAV = cli_arg("--sheet-cav", 0.030, float)  # fraction of slots, from SHEET's
# How fast the sheet grows past its own inception line. It is the ONE authored
# curve in the sheet and it is labelled as such in the docstring: a real extent
# comes from the section pressure distribution, which is the BEMT rung and is
# out of scope. What is NOT authored is when it starts (Burrill) or how it
# breathes (the wake shadow, below).
SHEET_CAV_KNEE = 0.22    # excess at which the sheet is 63% of its full extent
SHEET_CAV_R0 = 0.55      # x R_TIP: how far inboard a FULLY developed sheet reaches
SHEET_CAV_C = 0.55       # chord fraction it covers at the outer radii, developed
# The chordwise reach grows OUTBOARD -- (0.25 + 0.75 rr) of the above across
# the sheet's own radial span -- which is the shape every photograph of a
# partial sheet shows: a wedge, widest near the tip, tapering to nothing at its
# inboard edge. The two numbers are also what the radial sampling is weighted
# by, so parcels per square metre of blade is constant across it.
SHEET_CAV_CR = 0.25      # the inboard end of that wedge, as a fraction of the tip's
SHEET_CAV_T = 0.026      # m, the cavity's own thickness where it is fattest
SHEET_CAV_OFF = 0.005    # m of standoff: the vapour is ON the casting, not IN it
SHEET_CAV_GAIN = 1.05    # x, against a fully vapour-filled tip core's VAP_GAIN
# The grain is DELIBERATELY FATTER than the wake's. A sheet has to read as a
# CONNECTED surface, and the VAP_GAIN sand lesson two blocks up applies with
# the sign it always had: the fix for a scatter of dots is never more contrast
# per sample, it is neighbouring samples that OVERLAP. 2.1x the wake's grain on
# a sheet sampled at ~4.5 mm centres is a coverage of ~4, and the flux split
# divides the radiance back out so the total light is what it was.
SHEET_CAV_GROW = 3.4     # x SPRITE_R0
# The unsteadiness, in LABEL space -- on (radial station, blade, t) rather than
# on the world position, for exactly the reason the wake's turbulence is: a
# world-keyed wobble is a standing disturbance the blade sweeps through, which
# reads as a shimmer in the ROOM rather than on the sheet. This one rides the
# blade, which is what a cavity's closure line actually does.
SHEET_CAV_SHIM = 0.30    # +- of the chordwise extent
SHEET_CAV_RATE = 5.5     # label-space Hz of the closure line's own flutter
# The lobe is a property of the DISC's rim, and the blade's outer half sits at
# a skewed azimuth rather than at its spindle's. This is that skew at 0.7 R,
# taken as one constant: the lobe is 60 deg wide at half height and the skew
# swings 13 deg over the useful span, so resolving it per parcel would be
# arithmetic spent below the resolution of the thing it corrects.
SHEET_CAV_SKEWREF = -SKEW_RAD * 0.7 ** 1.9
# ── THE SLOT BUDGET, AND WHOSE PARCELS THESE ARE ────────────────────────────
# The sheet takes the TAIL of the buffer -- a contiguous range, because unlike
# every other population its slot index carries no meaning (see sheet_cav) --
# and the wake kernel launches over what is left. That would quietly cost every
# other population SHEET_CAV of its parcels, including the tip filaments, and
# R5 says the sheet's share comes out of the BODY populations and not out of
# the helices. So the two shares the wake kernel is given are re-normalised on
# the shorter launch: the tip count and the hub count come out bit for bit what
# they were, and the entrained bubbly sheet -- the one population that is a
# volume rather than a line, and the only one that can absorb it invisibly --
# pays the whole bill. Measured: 1.5% off the mean lit luminance of the
# sub-inception frame, against v10's own, which is under the noise the change
# of noise realisation puts there anyway.
N_CAV = int(round(N * max(SHEET_CAV, 0.0)))
N_MAIN = max(N - N_CAV, 1)
SHEET_EFF = 1.0 - (1.0 - SHEET) * N / N_MAIN        # tips: unchanged count
HUB_EFF = HUB_SHARE * N / N_MAIN                    # hub rope: unchanged count
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
# T_IN IS DERIVED FROM THE FUNNEL'S REACH, not authored, for the same reason
# LIFETIME is -- see the wake block. IN_REACH is the length the framing and the
# latched box were cut around (0.77 m ahead of the disc); the time it takes is
# that length over whatever disc velocity the polynomials give, which at the
# default helm is 1.85 m/s and puts T_IN at 0.56 s where the authored velocities
# had it at 0.80. It moves T_IN / PERIOD from 25% to 30%, which WAKE_SHARE picks
# up and BRIGHT and SIGMA divide back out, so no budget below has to be told.
T_IN = 0.0               # set below, once IN_K and IN_F exist to make dnm
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
# IN_F is the far-upstream drift as a fraction of the disc speed, and it is now
# a FLOOR rather than the value: with a ship speed there is a real freestream
# out there and the drift is that, so the kernel takes max(V_a / u_disc, IN_F).
# At the bollard the max is IN_F and the leg is bit-for-bit what it was.
IN_F = 0.45
IN_RV = 0.70             # m, the induced-velocity law's radius -- sets the flare
# The funnel's own time constant, from its authored length. dnm is what the
# kernel's d(u) profile integrates to over the leg, so IN_REACH / (u_disc/dnm)
# is exactly the time the outermost parcel spends getting there.
T_IN = cli_arg("--t-in", IN_REACH * ((1.0 - IN_F) * IN_K + IN_F) / U_DISC_OP,
               float)    # 0.56 s upstream; ~30% of the period
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
# average the dots away. 1.7M is where the 25 fps floor USED to put the
# default on a 4070 at 1280x800 -- 44.2 ms at 2M, 40.4 at 1.7M, 38.2 flat.
#
# ── AND THE TWIN STERN MISSED THAT FLOOR, WHICH IS SAID HERE RATHER THAN
# ── QUIETLY ABSORBED. 46.2 ms volumetric and 39.9 flat at the same count and
# the same size (three interleaved rounds on a quiet 4070, spread 0.2% and
# 1.7%), against 40.4 and 38.2 before. That is 22 fps where the documented
# floor is 25, and the whole of the 5.9 ms is the VOLUME rather than the ship:
# the same wide box at the old 160^3 lattice benches 44.0, so ~2.3 ms is the
# box that had to cover two wakes and ~3.6 ms is the resolution that had to go
# up with it to hold the voxel edge -- the self-shadowing this demo is graded
# on lives on that edge (see BOX_HALF). The hull, the second screw and the
# breathing together are inside the noise: flat moved 1.7 ms for all of them.
# --vol-res 160 buys the floor back at the price of a 36 mm lateral voxel, and
# --n 1400000 buys it back without touching the lattice; neither is the
# default, because the default is the picture.

# ── The look ────────────────────────────────────────────────────────────────
# The density volume is LATCHED at these bounds and never refitted: a box that
# tracks its own matter re-phases the lattice every frame and the volume swims.
# Elongated along the wake and tight across it.
#
# THE BOX GREW UPSTREAM and only upstream. It used to start at x = -0.35, a
# hand's width in front of a disc nothing had ever been in front of; the funnel
# reaches u_disc * T_IN / dnm = IN_REACH = 0.77 m, so x = -0.85 covers it.
# It did NOT grow sideways to the mouth's ~2.0 m, and that is a decision rather
# than an oversight: the volume is a fixed BOX_RES^3 lattice however big the box
# is, so widening it to the mouth would coarsen the lattice by 1.7x across the
# slipstream -- which is the one axis whose detail the self-shadowing lives on --
# to compute transmittance for parcels that have nothing between them and either
# the eye or the sun and would have come back as 1.0 anyway. The mouth sits
# outside the box and is unshadowed, which is what unshadowed water looks like.
#
# AND IT WAS RE-DERIVED ONCE FOR THE WATER CALIBRATION, downstream this time.
# Reach at the default helm is 4.80 m and the funnel stands 0.77 m ahead of the
# disc, so the box spans x = -0.85 to +4.85: centre 2.00, half 2.85. Done ONCE
# and latched, because the point of a latched box is that it does not move.
#
# THE B-SERIES REWRITE DID NOT MOVE IT AGAIN, and that was the point of
# authoring REACH and IN_REACH as LENGTHS rather than deriving them from a
# lifetime. The velocities nearly doubled and every time constant in the wake
# block halved to absorb it, so the two numbers this box was cut around --
# 4.80 and 0.77 -- came out bit for bit the same and BOX_VOX_RATIO below
# re-derives to exactly the value it had. Re-read it anyway when either length
# moves; that is the trap it exists for.
#
# ── AND THEN THE SECOND SCREW WIDENED IT, WHICH IS THE TRAP IT WARNED ABOUT ─
# Two wakes at z = +-SHAFT_Z need a box that covers both: 1.17 + 1.20 = 2.37
# each side, which is 1.98x the width the lattice was cut for. Left at
# BOX_RES 160 that would coarsen the lattice ACROSS the slipstream by exactly
# that factor -- and lateral resolution across the tube is the one axis the
# self-shadowing lives on (the v7 lesson: coarsen it and the sunward flank
# stops separating from the belly). So the resolution goes up with the box and
# the rule is stated as a rule: the voxel EDGE stays within 1.2x of what it
# was, on every axis.
#
#   was  5.70 x 2.40 x 2.40 m over 160^3   = 35.6 x 15.0 x 15.0 mm
#   now  5.70 x 2.40 x 4.74 m over 264^3   = 21.6 x  9.1 x 18.0 mm
#
# The lateral edge grows 1.20x -- exactly the budget -- and the other two
# SHRINK by 1.65x, which is free detail along the wake and across the tube and
# is paid for in memory rather than in milliseconds (the marches are a fixed
# 8 steps whatever the lattice is; only the scatter's write locality moves).
# The banner prints the bill. ONE ParticleField, not two: a second field would
# double the pass, the volume and the plumbing for two wakes that have to
# share a lattice anyway if either is ever to shadow the other.
BOX_CENTER = (2.00, 0.0, 0.0)
BOX_HALF = (2.85, 1.20, SHAFT_Z + 1.20)
BOX_RES = cli_arg("--vol-res", 264, int)
BOX_RES_REF = 160                   # the lattice SIGMA was tuned against
BOX_HALF_REF = (2.45, 1.20, 1.20)   # the pre-inflow box SIGMA was tuned against
# sigma_t one particle contributes, and it pays TWO rents. WAKE_SHARE is the
# one BRIGHT pays: fewer parcels downstream is less optical depth downstream.
# The box-volume ratio is subtler and is the trap in resizing a latched box at
# all -- the scatter deposits sigma per PARCEL with no division by cell volume,
# so a bigger voxel simply catches more parcels, and tau along a ray comes out
# proportional to the voxel volume while the 8-step march's ds cancels. Grow the
# box 10% and the whole wake gets 10% thicker for no reason anyone would guess
# from the diff. Divide it back out here and the box is free to move.
#
# AND IT IS A VOXEL RATIO, NOT A BOX RATIO, which only became visible the
# moment BOX_RES moved with the box. The quantity that matters is the volume
# of ONE CELL -- that is what decides how many parcels a cell catches -- so the
# resolution belongs in it cubed. Widening the box 1.98x laterally while
# raising the lattice 1.65x on every axis leaves each cell 0.51x its old
# volume, so each parcel now has to deposit ~2x the sigma it did. Read the
# comment above and then read this line: the box ratio alone would have got
# the sign of the correction right and the size of it wrong by 4.4x.
BOX_VOX_RATIO = (BOX_HALF[0] * BOX_HALF[1] * BOX_HALF[2]
                 / (BOX_HALF_REF[0] * BOX_HALF_REF[1] * BOX_HALF_REF[2])) \
    * (float(BOX_RES_REF) / float(BOX_RES)) ** 3
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
# ── R5: TWO WAKES OUT OF ONE BUDGET, AND WHAT IT COSTS PER WAKE ─────────────
# The slots split 50/50 between the screws, so each wake is drawn with half the
# parcels per metre it had -- and parcels per metre is exactly what both the
# radiance and the optical depth are made of. Doubling N would put it back and
# is the wrong answer: N is the SMOOTHNESS dial (see THE N CONTRACT) and this
# is a BODY problem, so it is paid for the way every body problem in this file
# is paid for, by moving the per-parcel quantities under the same flux
# discipline that already holds BRIGHT and SIGMA invariant in N.
#
# 1.4 AND NOT 2.0, and the 0.7 that leaves per wake is deliberate rather than
# timid. Two slipstreams are in frame now and a good deal of the time one is
# behind the other, so a per-wake restoration to 1.0 puts more light and more
# tau on the film than the single-screw frame ever had. 1.4 was judged at the
# v8/v9 mid-wake crop: the near wake reads at the density it had, and the pair
# does not add up to milk. It is one number for both because they are the same
# claim -- if a wake is half as dense it is half as bright AND half as deep.
TWIN_BOOST = cli_arg("--twin-boost", 1.4, float)
SIGMA = cli_arg("--sigma", 0.42, float) * (2_000_000 / N) * TWIN_BOOST \
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
OMG_OP = OM_G_FLOOR + (1.0 - OM_G_FLOOR) * (RPS_OP / RPS_REF)
BRIGHT = cli_arg("--bright", 0.038, float) * (2_000_000 / N) * TWIN_BOOST \
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
         hist: wp.array(dtype=wp.vec4),
         helm: wp.array(dtype=wp.vec4),
         t: float, n: int, blades: int,
         turb: float, sheet: float, hub_share: float,
         burst: float, sprite_r0: float, flux_p: float, bright: float,
         t_in: float, in_share: float, hist_n: int,
         tip_hint: float, vap_gain: float, bub_cav: float):
    i = wp.tid()
    # ── WHICH SHAFT, BY PARITY ──────────────────────────────────────────────
    # An even slot is the PORT screw, an odd one the STARBOARD, so the split is
    # exactly 50/50 with no draw and no state, and every population -- tips,
    # sheet, hub -- splits with it. sgn is the reflection: the whole closed form
    # below is evaluated in ONE screw's frame and the last line of the kernel
    # mirrors z and slides the result out to its own shaft. Counter-rotation,
    # the opposite helix handedness and the opposite swirl are not three things
    # to implement; they are one reflection of a point.
    sgn = 1.0 - 2.0 * float(i & 1)              # +1 port (+Z), -1 starboard
    s = wp.rand_init(90210, i)
    # The inflow leg's own stream, drawn from a SEPARATE seed so that adding
    # this leg did not renumber a single one of the wake's random draws -- the
    # before/after crops are then comparing the same noise realisation.
    s2 = wp.rand_init(1337, i)

    # ── WHICH POPULATION, decided before anything else ──────────────────────
    # Three here: the TIP filaments, the entrained SHEET that fills between
    # them, and the HUB rope on the axis. The FOURTH -- the blade-attached
    # cavity -- is not in this kernel at all; see sheet_cav below for why it
    # had to be its own launch. The hub's share comes OUT of the sheet's -- the
    # tip share (1 - sheet) is untouched -- so the helices are drawn with
    # exactly the parcels they were tuned with. The draw moved to the top of
    # the kernel because tin below has to know about it, and it consumes the
    # same single randf(s) in the same position it always did, so not one of
    # the wake's random numbers is renumbered by the move.
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
    # hist[k] = (omega, theta, v_i, V_a) as they stood on frame k. The ring is
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
    # ── THE RING WIDENED, AND THEN IT SPLIT IN TWO ─────────────────────────
    # It used to carry (loading, omega, theta): an authored loading scalar and
    # the shaft state. The loading is gone -- there is no such number any more,
    # only a thrust -- and what a parcel needs at birth came to six quantities,
    # so the history is two arrays indexed by the SAME frame:
    #
    #   hist[k]  vec4 (omega, theta, v_i, V_a)   the shaft and the water
    #   helm[k]  vec2 (K_T / K_T_design, Burrill ratio)   what they MEAN
    #
    # THE SPLIT IS A MEASUREMENT, not a tidiness preference. Both entries of
    # helm are functions of the helm alone -- they do not vary over the disc,
    # over the span or with age -- and computing them per parcel meant running
    # sigma_0.7R, tau_c and a wp.pow(sigma, 0.57) 1.7 million times a frame for
    # one number per frame. That pow measured 1.6 ms, which is 4% of the frame
    # and the difference between 24 and 25 fps at the shipped count; pow is the
    # expensive instruction in this kernel and this file has paid for it once
    # already (see SELECTS, NOT BRANCHES). On the host it is free, and it
    # brought a bonus with it: A_P can use the parcel's OWN pitch through
    # Taylor's correction instead of the design pitch, so the 14% approximation
    # that a fifth ring slot would have been needed to remove is simply gone.
    #
    # What is STILL derived in the kernel is what momentum theory can recover
    # exactly from the four state slots and nothing else:
    #
    #   u_disc = V_a + v_i        u_wake = V_a + 2 v_i     (definitions)
    #
    # Both rings are read at the same index, so a parcel's loading, its
    # cavitation and its velocities are all the ones that stood at ITS OWN
    # crossing frame. That is the causal beat and it is not negotiable.
    h = hist[kh % hist_n]
    hl = helm[kh % hist_n]
    om = h[0]
    vi = h[2]
    va = h[3]
    ld = hl[0]                                  # K_T / K_T_design at birth
    caq = hl[1]                                 # Burrill ratio at birth
    lamk = hl[2]                                # the wake lobe's own amplitude
    # The two legs' own speeds, before the per-span lag below.
    u_d0 = va + vi
    u_w0 = va + 2.0 * vi
    # ── AZIMUTH: INTERPOLATE INSIDE THE FRAME, EXTRAPOLATE PAST IT ──────────
    # The ring is stamped once per frame and tb is continuous, so reading
    # h[1] raw would quantise every filament's birth azimuth to 16.7 ms of
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
    th_b = h[1] + om * (tb - float(kh) * DT)
    # ── THE RADIANCE COUPLING ───────────────────────────────────────────────
    # Radiance follows the circulation, which is linear in omega like v_i is.
    # Since parcels per metre goes as 1 / u ~ 1 / omega, this holds the wake's
    # flux per METRE invariant under the throttle and lets the throttle change
    # the wake's LENGTH, which is the honest reading. It stays a function of
    # omega alone -- the LOADING half of the same coupling is ld, above.
    omg = OM_G_FLOOR + (1.0 - OM_G_FLOOR) * om / OMEGA_REF
    # caq -- the Burrill ratio at this parcel's own crossing -- came off the
    # helm ring above. It is what the throttle-step shots are made of: a parcel
    # carries the cavitation state of the moment it was shed for the rest of
    # its life, so a lit stub of new wake sits against a dark slug of old,
    # which is a thing a ship's engineer has actually seen.

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
    #
    # THE FLOOR IS ON THE KINEMATICS ONLY. u_w0 really does go to zero at zero
    # pitch (and negative past the zero-thrust J, where the screw is braking
    # and leaves the water slower than it found it), and a wake at zero speed
    # stacks every parcel of every slot in one bright annulus on the disc
    # plane. So the two speeds are floored at a small fraction of TIP speed --
    # after thr, ld and caq have already been taken from the unfloored values,
    # which is what keeps the thrust readout and the vapour honest while the
    # picture still shows a creep.
    ax = 0.55 + 0.45 * span
    # The floor has a term for each way water can be moving past the disc: a
    # fraction of TIP speed, and a fraction of the FREESTREAM. Either alone
    # goes to zero in a case the other covers -- a feathered screw turning in
    # still water, and a stopped screw in a moving one -- and the second term
    # is exactly zero at the bollard, so the default and the film are
    # bit-for-bit what they were.
    umin = U_FLOOR * om * R_TIP + 0.10 * va
    u_w = wp.max(u_w0, umin) * ax
    u_d = wp.max(u_d0, umin) * ax
    # The blade phase, plus half a blade pitch on the starboard shaft so the
    # two screws do not cross top dead centre together. It goes in HERE, in the
    # crossing azimuth, which is the one place both legs and the mesh agree on.
    phi_c = th_b + float(blade) * TWO_PI / float(blades) \
        + wp.where(sgn < 0.0, PHASE_OFF, 0.0)

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
        # ── THE FREESTREAM ENTERS HERE, IN TWO PLACES ───────────────────────
        # fcont is the ratio the flow far upstream bears to the flow at the
        # disc, V_a / u_disc, and it is the whole of what the ship's speed does
        # to this leg. At the BOLLARD it is zero and both expressions below
        # collapse, term for term, to the ones that shipped.
        #
        #   drift  the far-upstream axial speed. Without a freestream it was an
        #          authored IN_F of the disc speed; with one it is the
        #          freestream itself, so IN_F becomes a floor. A funnel in
        #          moving water is LONGER, because the water was already
        #          travelling when it started.
        #   flare  continuity. A_inf / A_disc = u_disc / V_a, so the tube's
        #          radius saturates at sqrt(u_disc / V_a) rather than growing
        #          without bound -- which is why a prop at the bollard drinks
        #          from a mouth twice its own diameter and a prop on a ship
        #          barely flares at all. The induced-velocity law still sets
        #          the SHAPE of the approach to it.
        fcont = wp.clamp(va / wp.max(u_d, 1.0e-4), 0.0, 1.0)
        fdr = wp.max(fcont, IN_F)
        dnm = (1.0 - fdr) * IN_K + fdr
        # d_max falls out of pinning the slope at the disc to u_d: no axial
        # speed pop, and a funnel that GROWS with the spin-up ramp for free.
        d_max = u_d * tin / dnm
        d = d_max * ((1.0 - fdr) * (1.0 - wp.pow(1.0 - uu, IN_K)) + fdr * uu)
        x = -d
        # Mass conservation through the on-axis induced velocity, floored at
        # the freestream. Keyed on the DISTANCE, so a short funnel is also a
        # narrow one.
        gd = 1.0 - d / wp.sqrt(d * d + IN_RV * IN_RV)
        g = fcont + (1.0 - fcont) * gd
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
            * (IN_G_FLOOR + (1.0 - IN_G_FLOOR) * ld)
        # Suspended sediment lit by what is left of the sun three metres down:
        # a pale green-grey, warmer than the vapour and much dimmer than it.
        col = wp.vec3(0.50, 0.56, 0.52)
    else:
        # ── Downstream: the wake ────────────────────────────────────────────
        x = u_w * tau - (u_w - u_d) * TAU_A * (1.0 - wp.exp(-tau / TAU_A))
        # Radial: contraction toward sqrt(u_disc / u_wake) * r0, which is mass
        # conservation across the disc and nothing else -- the constant 0.78
        # that used to sit here was an eyeball of the same quantity. At the
        # bollard it evaluates to 1/sqrt(2) = 0.707; past the zero-thrust J,
        # where the screw brakes and u_wake falls below u_disc, it goes above 1
        # and the slipstream EXPANDS, which is the correct picture and one the
        # authored constant could not draw. The clamp keeps a floored u_w0 from
        # turning the tube inside out.
        contract = wp.sqrt(wp.clamp(u_d0 / wp.max(u_w0, 1.0e-4), 0.25, 1.4))
        r = r0 * (contract + (1.0 - contract) * wp.exp(-tau / TAU_C))
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
        sw = SWIRL_F * om * (1.6 - 0.6 * span) * ld
        phi = phi_c + sw * TAU_S * (1.0 - wp.exp(-tau / TAU_S))
        # ── DOES THIS PARCEL BOIL? ──────────────────────────────────────────
        # One comparison, evaluated on the helm as it stood when the blade cut
        # this parcel. The hub reads the same number with a 12% head start,
        # which is the whole reason its rope lights first.
        # ── AND THE HULL'S SHADOW BREATHES IT ───────────────────────────────
        # lambda is the LOCAL loading at this parcel's own birth azimuth,
        # relative to the mean the whole momentum chain is anchored on: the
        # blade at twelve o'clock is inside the hull's boundary layer, sees a
        # lower advance and is working harder, and the blade at six o'clock is
        # in nearly clean water and is not. lamk carries the entire
        # linearisation (see wake_lobe_k) and it is proportional to J, so the
        # bollard needs no special case -- lamk is zero there and the rope is
        # as steady as it was before there was a hull.
        #
        # IT MULTIPLIES THE CRITERION AND NOTHING ELSE. Not u_w, not u_d, not
        # ld: the thrust, the torque, the slipstream speed and the helix
        # spacing all belong to the MEAN and have to keep matching the diagram
        # the panel draws them on. What breathes is the vapour -- intensity,
        # core width and rope length, all of which hang off exc below.
        lobe = wp.pow(0.5 * (1.0 + wp.cos(phi_c)), W_LOBE_P) - W_LOBE_MEAN
        cq = caq * wp.max(1.0 + lamk * lobe, 0.0)
        if is_hub:
            # The hub rope is born ON the axis, where there is no azimuth to
            # be modulated at -- the circumferential deficit is a property of
            # the disc's rim, not of its centre. So the axial rope stays STEADY
            # while the helices pulse, which is both the physics and, as a
            # bonus, the control that makes the pulsing legible.
            cq = caq * HUB_CAV
        exc = wp.max(cq - 1.0, 0.0)          # the EXCESS over the margin
        # (the excess is now over the Burrill INCEPTION LINE rather than over
        # a pressure margin, which is why CAV_KNEE and CAV_LK moved with it.)
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
        omgl = omg * (G_FLOOR + (1.0 - G_FLOOR) * ld)
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

    # ── THE REFLECTION, AND IT IS THE LAST THING THAT HAPPENS ───────────────
    # Everything above ran in ONE screw's frame -- the port one, which is the
    # screw this demo has always had. A starboard parcel is that parcel
    # reflected in the ship's centreline plane, and reflecting z is the whole
    # of it: the helix winds the other way, the swirl turns the other way, the
    # blade phase runs the other way, and the mesh does the same reflection so
    # the filaments stay welded to the tips that made them.
    #
    # It is done AFTER the turbulence on purpose. The curl noise is sampled at
    # the Lagrangian label, so the two screws' wobbles are already independent
    # realisations; reflecting the sum rather than the position keeps the
    # displacement attached to the filament it belongs to.
    out_pos[i] = wp.vec4(p[0], p[1], sgn * (p[2] + SHAFT_Z), sprite_r0 * grow)
    out_col[i] = wp.vec4(c[0], c[1], c[2], 1.0)


# ── R1: THE SHEET ON THE BACK OF THE BLADE, AND WHY IT IS A SECOND LAUNCH ───
# It was written as a fourth population inside shed() first, which is where it
# belongs on paper: one kernel, one closed form, one pass. It cost ELEVEN
# MILLISECONDS of a 45 ms frame -- and it cost them with `--sheet-cav 0`, which
# is the measurement that settles what they were. Not the work: there was none.
# REGISTERS. The blade-local branch declares three dozen live locals (a
# planform lerp, a section solve, a twist, a skew and a pivot rotation), the
# compiler allocates the union of every branch's registers, occupancy falls for
# the whole launch, and 1.7 million parcels pay for arithmetic 3% of them will
# execute. Divergence hides work; register pressure does not even hide.
#
# So the population gets its own kernel over its own CONTIGUOUS slot range, and
# the split is free in a way it is free for nothing else in this file: a sheet
# parcel has no phase, no lifetime and no shedding rate, so which slot it sits
# in means nothing at all. Every other population is hash-assigned precisely
# because the slot index IS its clock. This one has no clock.
#
# 45.5 -> 45.8 ms after the move, against 56.4 before it. See THE BENCH.
@wp.kernel
def sheet_cav(out_pos: wp.array(dtype=wp.vec4),
              out_col: wp.array(dtype=wp.vec4),
              hist: wp.array(dtype=wp.vec4),
              helm: wp.array(dtype=wp.vec4),
              plan: wp.array(dtype=wp.vec2),
              t: float, base: int, blades: int,
              sprite_r0: float, flux_p: float, bright: float,
              hist_n: int, sheet_inc: float, dpitch: float):
    i = wp.tid()
    j = base + i                                # the slot in the shared buffers
    s = wp.rand_init(4271, j)
    # Screw and blade by INDEX and not by hash, for the same reason the range
    # is contiguous: with nothing riding on the slot number, i & 1 and
    # (i / 2) % B give an exactly balanced ten-way split rather than a
    # binomially wobbly one, and a sheet is a surface whose sampling density is
    # the whole of its look.
    sgn = 1.0 - 2.0 * float(i & 1)              # +1 port (+Z), -1 starboard
    blade = (i // 2) % blades

    # ── THE HELM, AT NOW ────────────────────────────────────────────────────
    # A sheet cavity is not shed and does not convect: it is a feature OF a
    # surface, and the surface is turning. So there is no crossing time in the
    # past to look up -- the index is simply this frame's, which advance()
    # stamped a few microseconds ago in the same breath as setting
    # prop.rotation.x to the theta it carries. That is the whole of the
    # anti-lag argument. Every other parcel is welded to a blade by AGREEMENT
    # between two clocks and the docstring's CREDIBILITY DETAIL is about
    # getting that agreement wrong by a frame; this population cannot lag,
    # because its position contains exactly one clock.
    kh = int(wp.floor(t / DT + 1.0e-3))
    if kh < 0:
        kh = 0
    h = hist[kh % hist_n]
    hl = helm[kh % hist_n]
    om = h[0]
    ld = hl[0]                                  # K_T / K_T_design
    caq = hl[1]                                 # the Burrill ratio
    lamk = hl[2]                                # the wake lobe's amplitude
    omg = OM_G_FLOOR + (1.0 - OM_G_FLOOR) * om / OMEGA_REF
    # The same azimuth the mesh is drawn at, built the same way phi_c is in
    # shed() -- sub-frame residual walked out at the frame's own rate, blade
    # offset, and half a blade pitch on the starboard shaft.
    phi_c = h[1] + om * (t - float(kh) * DT) \
        + float(blade) * TWO_PI / float(blades) \
        + wp.where(sgn < 0.0, PHASE_OFF, 0.0)

    # ── THE GATE, AND IT BREATHES AT THE BLADE'S OWN AZIMUTH ────────────────
    # The wake parcels evaluate the hull's loading lobe at their BIRTH azimuth,
    # because that is where the blade was when it cut them and they have been
    # convecting ever since. The sheet does not convect: it rides the blade, so
    # it evaluates the same lobe at the azimuth the blade has NOW -- and the
    # consequence is the headline of this slice. The sheet grows as the blade
    # sweeps through the hull's boundary layer at twelve o'clock and shrinks
    # again through the bottom of the sweep, once per revolution.
    #
    # lamk is the ring's own linearisation and it is PROPORTIONAL TO J, so the
    # bollard control needs no `if` here either: with no way on there is no
    # boundary layer, lamk is zero, and the sheet is as steady as the ropes
    # were before there was a hull over them.
    lobe = wp.pow(0.5 * (1.0 + wp.cos(phi_c + SHEET_CAV_SKEWREF)),
                  W_LOBE_P) - W_LOBE_MEAN
    cq = caq * wp.max(1.0 + lamk * lobe, 0.0)
    # THE LADDER: this line sits sheet_inc ABOVE the tip vortices'. Below it
    # there is no sheet AT ALL -- not a dim one -- for the same reason TIP_HINT
    # is 0.0: clean water is clean water (619dd5e1).
    exs = wp.max(cq / sheet_inc - 1.0, 0.0)
    ext = 1.0 - wp.exp(-exs / SHEET_CAV_KNEE)
    if ext <= 1.0e-4:
        out_pos[j] = wp.vec4(0.0, 0.0, 0.0, -1.0)
        out_col[j] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        return
    # ── THE EXTENT GROWS IN AREA, NOT IN DENSITY ────────────────────────────
    # The obvious way to draw a growing sheet is to spread a fixed parcel count
    # over a growing patch, and it is wrong twice over: the newborn sheet comes
    # out as an over-bright smear and the developed one thins into sand. So the
    # patch's own AREA fraction is the survival probability and everything else
    # -- radiance, grain, spacing -- is held constant. Parcels per square metre
    # of blade is then invariant, what the excess buys is exactly what it
    # should buy (more blade covered), and the density volume stays honest: a
    # sliver of a sheet deposits a sliver of optical depth rather than the
    # whole population's worth of it into a tenth of the voxels.
    ur = wp.randf(s)
    uc = wp.randf(s)
    ut = wp.randf(s)
    ua = wp.randf(s)
    spf = 0.25 + 0.75 * ext                     # radial reach, 1 = developed
    cam = 0.25 + 0.75 * ext                     # chordwise reach, ditto
    if ua > spf * cam:
        out_pos[j] = wp.vec4(0.0, 0.0, 0.0, -1.0)
        out_col[j] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        return
    # Radial station, sampled with a density proportional to the local
    # chordwise reach so the WEDGE below is filled evenly rather than bunched
    # at its narrow end. Closed-form inversion for a linear reach: one sqrt,
    # against a rejection loop that has no bound.
    cr0 = SHEET_CAV_CR
    cr1 = 1.0 - SHEET_CAV_CR
    rr = (-cr0 + wp.sqrt(cr0 * cr0
                         + 2.0 * cr1 * (cr0 + 0.5 * cr1) * ur)) / cr1
    r_in = R_TIP * (1.0 - (1.0 - SHEET_CAV_R0) * spf)
    rs = r_in + (0.995 * R_TIP - r_in) * rr
    fs = wp.clamp((rs - R_PALM) / (R_TIP - R_PALM), 0.0, 1.0)
    # The blade's own planform and thickness law, off the same PF_ tables the
    # loft is built from -- resampled onto a uniform span grid at import so the
    # lookup is an index and a lerp. See PLAN_TAB.
    fi = fs * float(PLAN_N - 1)
    i0 = int(fi)
    if i0 > PLAN_N - 2:
        i0 = PLAN_N - 2
    fr = fi - float(i0)
    pa = plan[i0]
    pb = plan[i0 + 1]
    chord = C_MAX * (pa[0] + (pb[0] - pa[0]) * fr)
    thick = chord * (pa[1] + (pb[1] - pa[1]) * fr)
    # ── THE CLOSURE LINE FLUTTERS, IN LABEL SPACE ───────────────────────────
    # Sampled on (radial station, which blade, time) and NOT on the world
    # position, for the reason the wake's curl noise is: a world-keyed
    # disturbance stands still while the blade sweeps through it, which reads
    # as a shimmer in the ROOM. This one is welded to the blade and is the only
    # thing in the sheet that moves independently of the casting.
    sh = wp.noise(wp.rand_init(6131), wp.vec4(
        rr * 3.4, float(blade) * 5.0 - sgn * 11.0, t * SHEET_CAV_RATE, 0.0))
    cext = wp.max(SHEET_CAV_C * cam * (cr0 + cr1 * rr)
                  * (1.0 + SHEET_CAV_SHIM * sh), 0.02)
    cf = 0.008 + (cext - 0.008) * uc        # chord fraction, 0 = leading edge
    # ── THE SECTION, AND THE SIGN THAT PUTS IT ON THE BACK ──────────────────
    # The loft's own section is cw = 0.5 cos a chordwise and th = 0.5 sin a +
    # 0.13 sin^2 a in thickness. Solving the first for the chord fraction and
    # taking the a in (pi, 2pi) branch is the SUCTION side -- negative
    # thickness, which is the -X, upstream-facing face, because the blade
    # advances toward +X (that is what makes the wake blow that way) and the
    # face that does the pushing is therefore the +X one. That single minus
    # sign is the whole of "which side of the blade", and it is the first thing
    # a propeller man checks.
    eh = wp.sqrt(wp.max(cf * (1.0 - cf), 0.0))
    # A partial cavity is thin at the leading edge and thickest toward its
    # closure. Offsetting along the section's own normal is, for a pure
    # thickness displacement, simply a subtraction from yt.
    u01 = cf / cext
    tprof = wp.sqrt(u01) * (1.0 - 0.35 * u01)
    off = SHEET_CAV_OFF + SHEET_CAV_T * (0.35 + 0.65 * ext) * tprof * ut
    yt = (-eh + 0.52 * eh * eh) * thick - off
    zc = chord * (0.5 - cf)
    # The twist law, the rake and the skew: the loft's, line for line.
    bet = wp.atan2(P_SCREW, TWO_PI * rs)
    cb = wp.cos(bet)
    sb = wp.sin(bet)
    axl = yt * cb - zc * sb + RAKE_M * fs * fs
    chd = yt * sb + zc * cb
    skw = -SKEW_RAD * wp.pow(fs, 1.9)
    ck = wp.cos(skw)
    sk = wp.sin(skw)
    py = rs * ck - chd * sk
    pz = rs * sk + chd * ck
    # ── AND THE PITCH PIVOT, BECAUSE THE BLADE TURNS IN ITS FLANGE ──────────
    # set_pitch rotates each blade about its own spindle (local +Y) by
    # -(beta - BETA_DESIGN) and hands the angle back; the sheet is ON that
    # blade, so it takes the same rotation about the same axis, in the local
    # frame and BEFORE the azimuth, exactly as the scene graph does.
    cd = wp.cos(dpitch)
    sd = wp.sin(dpitch)
    pxr = axl * cd + pz * sd
    pzr = -axl * sd + pz * cd
    # The azimuth. Everything above is in the PORT screw's own frame and the
    # last line mirrors z, which is the same one operation shed() ends with --
    # and it is what makes the starboard sheet a reflection rather than a
    # second piece of code (M_z R_y(d) is R_y(-d) M_z, which is exactly what
    # build_screw does to the casting).
    cp = wp.cos(phi_c)
    sp = wp.sin(phi_c)
    p = wp.vec3(pxr, py * cp - pzr * sp, py * sp + pzr * cp)
    # A little body across the cavity, so the sheet is a layer and not a decal.
    p = p + wp.vec3(wp.randn(s), wp.randn(s), wp.randn(s)) * 0.0018

    # WHITE, because a vapour cavity is a dense broadband scatterer and it is
    # the whitest thing in the frame; brighter than a tip core because it is a
    # sheet seen face-on rather than a tube seen edge-on. The grain is
    # DELIBERATELY fatter than the wake's -- the VAP_GAIN sand lesson with the
    # sign it always had, that the cure for a scatter of dots is neighbouring
    # samples which OVERLAP and never more contrast per sample -- and the flux
    # split divides the radiance back out so the total light is unchanged.
    grow = SHEET_CAV_GROW
    vapi = 1.0 - wp.exp(-exs / CAV_KNEE)
    gain = SHEET_CAV_GAIN * omg * (G_FLOOR + (1.0 - G_FLOOR) * ld) \
        * (0.55 + 0.45 * vapi) * (1.0 + 0.22 * sh)
    c = wp.vec3(0.90, 0.95, 1.00) * (bright * gain / wp.pow(grow, flux_p))
    out_pos[j] = wp.vec4(p[0], p[1], sgn * (p[2] + SHAFT_Z), sprite_r0 * grow)
    out_col[j] = wp.vec4(c[0], c[1], c[2], 1.0)


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
    renderer.set_fog_water_surface_y(WATERLINE)
    renderer.set_underwater_murk(MURK, tp.Color(*MURK_COLOR))

# ── AND NOW THERE IS A REAL SURFACE ON THAT PLANE ───────────────────────────
# set_fog_water_surface_y states WHERE the air stops; until this revision
# nothing was drawn there, so the ceiling of this scene was an infinite flat
# analytic plane. Everything the murk does with the sun was already correct
# against it -- the sun is refracted at the interface (kMurkEtaAirWater), the
# shadow rays follow the REFRACTED direction, and the shaft march carries a
# CAUSTIC PROXY sampled from an FFT ocean's fine cascade at the surface point
# each step's light passed through. That last one was simply inert here,
# because it arms itself on `oceanFineTileSize > 0` and there was no Ocean in
# the scene. Adding one is the whole of the change: the god-rays stop being
# smooth cones and braid, with no renderer knob touched and none invented.
#
# THE SEA STATE IS SHORT-FETCH CHOP, NOT SWELL, and that is a caustics choice
# rather than a weather one. The proxy reads the FINE cascade -- the metre-scale
# ripple -- so what braids the shafts is the small stuff; a long ocean swell
# would heave the whole surface without changing the pattern the light is cut
# into. fetch = 25 km at 6.5 m/s is a coastal sea: steep centimetre-to-metre
# waves, a surface that moves visibly through the Snell window, and no drama.
#
# THE RESOLUTION IS DELIBERATE, and the number that matters is not the FFT's.
# `resolution` is the MESH grid, and it was a 1024-vertex sheet that turned out
# to be the whole of an unrelated demo's 5 fps mystery. Here the murk saturates
# at 6/sigma = 50 m, so a 400 m sheet is already twice as much water as can ever
# be seen and 256 columns puts a vertex every 1.6 m -- coarse in the far field
# that murk has eaten anyway, and irrelevant near the camera because the
# cascades DISPLACE below mesh resolution and the chit perturbs the normal
# below that. tile_size_2 is PINNED at 7 m rather than left to auto (which
# would derive 3.7 m from a 400 m extent): that is the band the murk's caustic
# constants were shaped against in the fjord, and letting the mesh extent
# silently retune the light pattern is not a coupling this demo wants.
#
# WHAT IT DOES NOT BUY, MEASURED RATHER THAN HOPED: a Snell window. Looking UP
# from under a real surface you see the whole sky packed into a 48.6 deg cone,
# and that is the postcard shot of any underwater scene -- but only if there is
# a sky. This scene has none: scene.background is deliberately near-black (see
# THE BACKGROUND HAS TO STAY NEARLY BLACK), there is no environment map, and the
# daylight is three DirectionalLights standing in for one. So the surface from
# below refracts black and reflects dark water, and an up-looking framing comes
# back an almost empty frame. Giving it a real sky would mean an environment
# map, and the env is what feeds BOTH the murk's in-scatter and the ambient the
# bronze is read against -- retuning the entire look for one shot. Not done, and
# the same reason is why the ceiling in the aft framings is DARKER than it was:
# the analytic plane's stand-in gradient (applyMurkSky) was quietly flattering a
# sky that is not there, and the real surface is honest about it.
#
# --no-ocean is the A/B and the regression: it is the frame this demo drew
# before the surface existed, and it must still be that frame exactly.
OCEAN = "--no-ocean" not in sys.argv
ocean = None
if OCEAN and MURK > 0.0:
    ocean = tp.Ocean(size=400.0, resolution=256, fft_size=512,
                     wind_speed=6.5, wind_theta=0.35, fetch=25.0e3,
                     choppiness=0.45, tile_size_1=64.0, tile_size_2=7.0)
    ocean.position.y = WATERLINE
    scene.add(ocean)

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
        # ── AND IT MOVED OUT FOR THE SECOND SCREW, BY EXACTLY ITS SPACING ───
        # "side" used to sit 3.00 m off a wake on the axis. There are two wakes
        # now, 2.34 m apart, and the shot that reads is the one down the PORT
        # slipstream with the starboard one running away behind it -- so the
        # eye moved out by the shaft offset and the aim moved half of it. The
        # near wake is still 3.0 m from the lens, which is what keeps a 1:1
        # mid-wake crop comparable with every crop taken before the pair
        # existed; that was the point of moving it this way rather than simply
        # pulling back.
        # AND IT DROPPED BELOW THE SHAFT LINE FOR THE HULL. A stern is read
        # from UNDERNEATH -- the counter overhead, the screws hanging out of
        # it, the slipstreams running aft under open water -- and a camera
        # above the shaft looks down at the ship's FLANK instead, which is a
        # wall of paint with no shape in it. The eye is 0.45 m under the shaft
        # and the aim 0.3 m over it, so the counter closes the top of the frame
        # and the wake owns the middle of it.
CAMS = {"side":  ((1.02, -0.45, 3.00 + SHAFT_Z), (1.44, 0.26, 0.5 * SHAFT_Z)),
        "mouth": ((0.45, 0.60, 7.40), (0.95, -0.05, 0.0)),
        # "prop" is the geometry's own shot and nothing else's: close enough
        # that a 2 cm bolt head is several pixels, three-quarters on so the
        # flange discs are ellipses rather than edge-on lines, and aimed at the
        # hub so the blade roots and their bolt circles fill the frame. It is
        # deliberately NOT a framing the wake reads well at.
        "prop":  ((1.25, 0.62, 2.15 + SHAFT_Z), (0.06, 0.0, SHAFT_Z)),
        # "stbd" is "prop" REFLECTED, and it exists as a view because it is the
        # acceptance test for R2: the starboard screw is supposed to be the
        # port screw's mirror image, so these two frames have to be mirror
        # images of each other -- same bronze, same lighting sense reversed,
        # same blades leaning the other way. A mirrored loft whose winding was
        # not flipped shades BLACK, and putting the two crops side by side is
        # the fastest way there is to see that it did not.
        "stbd":  ((1.25, 0.62, -2.15 - SHAFT_Z), (0.06, 0.0, -SHAFT_Z)),
        # "stern" is the pair's own shot and the one frame the counter-rotation
        # is legible in: high on the port quarter, looking forward and down the
        # two slipstreams at once, so both helices show their handedness in the
        # same image and wind visibly OPPOSITE ways.
        "stern": ((6.60, 2.85, 5.60), (0.80, -0.05, 0.0)),
        # ── "back" IS THE SHEET'S OWN SHOT, AND IT HAS TO BE FROM AHEAD ─────
        # The suction side of a propeller blade is its UPSTREAM face, and every
        # other framing in this table sits AFT of the disc plane and therefore
        # looks at the pressure side. The billboards are depth-tested against
        # the casting, so from those views the sheet is correctly hidden BEHIND
        # the blades and there is nothing to grade. This one stands forward of
        # the disc and outboard of the port shaft, below the counter and clear
        # of the bossing, and it is the composition the reference photograph
        # has: the backs of the blades filling the frame, the vapour sheet at
        # their leading edges and outer radii, and a tip rope leaving the tip
        # into the slipstream running away behind.
        # ── AND IT CANNOT BE HEAD-ON, WHICH IS THE STERN BEING RIGHT ───────
        # The obvious framing is straight up the shaft from ahead, and the ship
        # refuses it: the port BOSSING is a cone flaring from 0.165 m at
        # x = -3.08 to 0.72 m at x = -0.73, so a near-axial eye is looking at
        # the inside of the shaft fairing and the blades are behind it. That is
        # what a bossing IS -- the hull's plating faired around the shaft --
        # and it is exactly why a cavitation observer on a real ship works
        # through a window rather than from in front of the disc. So the eye
        # goes to a 50-degree forward quarter, from BELOW (a stern is read from
        # underneath, as "side" already found), where the sight line clears the
        # cone's widest station by 0.16 m and the backs of all five blades are
        # open. The wake runs away to the right of the frame, which is the
        # reference photograph's own layout.
        # It is also CLOSE -- 2.5 m, where every other framing in this table
        # stands off at 3 or more. The slipstream is directly behind the disc
        # from any forward angle and it is the brightest thing in the frame, so
        # the shot has to be tight enough to put the blades against water
        # rather than against four metres of boiling wake. The sheet is a
        # centimetres-thick feature on a half-metre chord; it is graded at 1:1
        # and it needs the pixels.
        "back":  ((-1.54, -0.77, 1.69 + SHAFT_Z), (0.05, 0.00, SHAFT_Z))}
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
#
# ── AND THEN THE HULL MOVED IT ONTO THE QUARTER ─────────────────────────────
# The sprites' T_sun march samples the density volume alone, so the stern above
# these screws does not shadow their wakes. That is a renderer limit and it is
# out of scope to fix; what is in scope is refusing to put the hull where the
# missing shadow is READABLE. So the key was raked down and swung aft, and the
# geometry of it is the whole of the staging:
#
#   from any point in the slipstream, the path to this sun runs UP AND AFT --
#   +X, away from a hull that ends at x = 0.62 -- so it crosses open water for
#   its entire length and there is nothing there to have cast a shadow.
#
# It was (-0.55, 3.10, -1.55): 62 deg up, off the BOW. Every ray from the wake
# to that sun went forward and up, straight through eight metres of ship.
#
# 43 DEGREES IS A FLOOR AND NOT A PREFERENCE. Refraction packs the entire sky
# into Snell's 48.6 deg cone, so underwater there is no such thing as a sun
# lower than 41.4 deg above the horizontal however low the real one is. This
# sits just inside that, which is as raking as downwelling light is allowed to
# be -- and it keeps the column's bright top and dark belly, because 0.69 of
# this direction is still straight up.
#
# WHAT IS STILL DISHONEST, stated rather than hidden: the funnel and the first
# half-metre of slipstream ARE under the counter and are lit as though they
# were not. They are suspended silt and the dimmest thing in the frame, and
# everything the eye actually reads -- the ropes, the column, the collapse --
# is aft of the transom in genuinely open water.
sun = tp.DirectionalLight(0xB6D2FF, 2.3)
sun.position.set(2.05, 2.55, -1.75)
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
# ── AND THE HULL NEEDED ONE MORE, WHICH THE WATER WAS ALREADY OWED ──────────
# The key is downwelling and the counter faces DOWN, so with two lights the
# whole underside of the ship went to black and the screws hung out of a void:
# the one thing R4 exists to show -- propellers under a hull -- was the thing
# that could not be seen. What lights the underside of a hull in every piece of
# diver's footage is the water itself: the column below and the bottom beyond
# it scatter the downwelling light back UP, and a hull's belly is lit by that
# and by nothing else. It is a fill, it is cold and green like the murk it
# comes out of, and it is a fifth of the key -- well under it, because a
# DirectionalLight brighter than the sun would steal the billboards' own sun
# (they take the brightest one in the scene) and flip the whole wake to a
# front-lit read.
upwell = tp.DirectionalLight(0x7FB6A8, 0.80)
upwell.position.set(-1.20, -2.40, 1.35)
scene.add(upwell)


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
# C_MAX, SKEW_DEG, RAKE_M and the three PF_ tables are the blade's SECTION MATH
# and they now live in the geometry block at the top of the file, because the
# cavitation sheet is drawn on this same surface and there must be exactly one
# statement of the shape. See THE BLADE'S OWN SECTION MATH up there.
BOLTS = 10              # per blade flange
BOLT_CIRCLE = 0.128     # m
FLANGE_R = 0.165        # m, the palm disc


def blade_geometry(stations=18, around=14, mirror=False):
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

    mirror=True IS THE STARBOARD SCREW, and it is a reflection in z -- the same
    operation the kernel applies to every parcel. A reflection reverses
    ORIENTATION, so every triangle has to be wound back the other way or
    compute_vertex_normals() hands the blade a normal pointing into the solid
    and the casting shades black on the face the camera can see. That is not a
    hypothetical: it is exactly what this loft did the first time it was
    written (see WINDING below), and mirroring is the one operation guaranteed
    to reproduce it. The uv is left alone -- u is a triangle wave about the
    seam and already mirrors the two faces, and a wear map has no direction in
    it to get wrong.
    """
    import numpy as np
    sp = np.linspace(R_PALM, R_TIP, stations)
    f = (sp - R_PALM) / (R_TIP - R_PALM)                # 0 at root, 1 at tip
    chord = C_MAX * np.interp(f, PF_F, PF_C)
    thick = chord * np.interp(f, PF_F, PF_T)
    pitch = np.arctan(P_SCREW / (TWO_PI * sp))          # 46 deg root, 16 deg tip
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
    if mirror:
        v[:, :, 2] = -v[:, :, 2]
        idx = [idx[k + o] for k in range(0, len(idx), 3) for o in (2, 1, 0)]
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
HUB_G = tp.SphereGeometry(R_HUB, 40, 26)
SHAFT_G = tp.CylinderGeometry(0.115, 0.115, 2.90, 24)
CAPR_G = tp.CylinderGeometry(0.198, 0.208, 0.048, 40)
DOME_G = tp.SphereGeometry(0.192, 32, 18, 0.0, TWO_PI, 0.0, math.pi / 2)
BOLT_G = tp.CylinderGeometry(0.0195, 0.0215, 0.028, 8)   # a hex-ish head
SEAT_G = tp.CylinderGeometry(0.186, 0.192, 0.030, 40)
PALM_G = tp.CylinderGeometry(FLANGE_R, FLANGE_R + 0.018, 0.052, 40)
BLADE_G = (blade_geometry(mirror=False), blade_geometry(mirror=True))


def build_screw(mirror):
    """One complete screw and its pitch pivots.

    THE STARBOARD ONE IS THE PORT ONE REFLECTED IN z, and it is built by
    reflecting the three things that have a handedness rather than by writing a
    second assembly: the blade loft (mirror=True, winding flipped), the blade
    ORDER around the hub (the arm rotation reverses), and the spindle rotation
    the pitch slider drives (see set_pitch). Everything else in here is a body
    of revolution about the shaft and is its own mirror image."""
    g = tp.Group()
    g.add(tp.Mesh(HUB_G, metal))
    # The shaft, running forward to the bracket and on into the hull: without
    # something at the far end of it the hub floats, which was survivable when
    # there was nothing else in frame and is not now that there is a stern
    # over it.
    shaft = tp.Mesh(SHAFT_G, polish)
    shaft.rotate_z(math.pi / 2)                  # the cylinder's own axis is Y
    shaft.position.x = -1.56
    g.add(shaft)
    # The cap: a bolted ring flange, then the dome. Aft (+X), downstream, which
    # on this axis convention is the side the wake leaves from.
    cap_ring = tp.Mesh(CAPR_G, polish)
    cap_ring.rotate_z(math.pi / 2)
    cap_ring.position.x = 0.222
    g.add(cap_ring)
    dome = tp.Mesh(DOME_G, polish)
    dome.rotate_z(-math.pi / 2)                  # the +Y pole becomes +X
    dome.position.x = 0.243
    g.add(dome)

    for k in range(12):                          # the cap's own bolt circle
        ang = (k + 0.5) * TWO_PI / 12
        bolt = tp.Mesh(BOLT_G, polish)
        bolt.rotate_z(math.pi / 2)
        bolt.position.set(0.240, 0.163 * math.cos(ang), 0.163 * math.sin(ang))
        g.add(bolt)

    pvs = []
    for b in range(BLADES):
        arm = tp.Group()
        # The same offset the kernel uses, with the same reflection: a mirrored
        # screw's blades run round the hub the other way.
        arm.rotate_x((-1.0 if mirror else 1.0) * b * TWO_PI / BLADES)
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
        pivot.add(tp.Mesh(BLADE_G[1 if mirror else 0], metal))
        arm.add(pivot)
        g.add(arm)
        pvs.append(pivot)
    return g, pvs


# PORT is +Z and is the screw this demo has always had; STARBOARD is -Z and is
# its mirror image. See R2 up beside SHAFT_Z for why that assignment is forced
# rather than chosen.
prop, pivots = build_screw(False)
prop.position.z = SHAFT_Z
scene.add(prop)
prop_s, pivots_s = build_screw(True)
prop_s.position.z = -SHAFT_Z
scene.add(prop_s)


def set_pitch(beta_deg):
    """Turn every blade in its flange, on both shafts. The loft is authored at
    BETA_DESIGN, so this is a DELTA about the spindle -- zero rotation is the
    designed blade. The sign is the loft's: a section's chord goes to -X as its
    pitch angle grows, and a +Y rotation takes it to +X, so more pitch is a
    NEGATIVE rotation about the spindle -- and a POSITIVE one on the mirrored
    screw, because a reflection reverses the sense of every rotation about an
    axis lying in its plane. Get this one backwards and the two screws feather
    in opposite directions, which the film's beat 5 would show immediately."""
    d = -math.radians(beta_deg - BETA_DESIGN)
    for pv in pivots:
        pv.rotation.y = d
    for pv in pivots_s:
        pv.rotation.y = -d
    # Handed back so the sheet-cavitation kernel can turn its parcels in the
    # flange by the SAME number the casting turned by. The kernel works in the
    # port screw's frame and mirrors z on its last line, and M_z R_y(d) is
    # R_y(-d) M_z -- so the port angle is the only one it needs.
    return d

# ── R4: THE STERN THE SCREWS HANG UNDER ─────────────────────────────────────
# A hull is what makes a twin screw a SHIP and it is what R1's wake deficit is
# a consequence of, so it has to be here and it has to be first-party. It is
# the blade loft's own technique at ten times the size: a rake of cross
# sections along x, quads between neighbours, both ends capped, one closed
# solid. Eight metres of run, a counter that rises over the discs, a centreline
# skeg, bossings that fair the shafts into the underbody, and a transom just
# aft of the disc plane so the slipstream runs into open water at once.
#
# WHAT IT IS NOT is as important. No rudders: they would sit IN the slipstream
# and this closed form has no interaction to give them, so a rudder would be
# the one piece of geometry in the frame that is visibly lying (out of scope,
# stated in the plan). No propeller-hull interaction beyond R1's own wake
# fraction. No plating detail, no weld seams, no name on the transom: the
# silhouette is the deliverable at this framing distance and detail on a shape
# that is not right is what makes a model read as amateur.
#
# ── THE ONE DISHONESTY, AND WHY IT IS STAGED RATHER THAN CODED ──────────────
# The sprites' T_sun march samples the particle density volume and NOTHING
# else, so this hull does not shadow the wake. Geometry-aware sprite shadowing
# is a renderer feature and is out of scope here. What is IN scope is not
# putting the hull where the missing shadow can be read: the sun is raked down
# to the Snell floor and swung onto the QUARTER, so its path from the wake goes
# up and AFT over open water and never crosses the hull at all -- see the sun
# below. What remains under the counter is the funnel and the first half metre
# of slipstream, which are silt motes and the dimmest part of the frame.
HULL_X0, HULL_X1 = -8.60, 1.06      # the run this section covers, m
# WATERLINE is DEPTH, stated once up beside CAV_MARGIN -- see ONE NUMBER, FOUR
# CONSUMERS. The stern floats on the same plane the Ocean is built on and the
# murk refracts the sun at; a second `WATERLINE = DEPTH` here is exactly the
# kind of duplicate that survives one --depth change and not the next.
HULL_TOP = WATERLINE + 0.30         # the deck edge, and it is never in frame
# (x, y of the underside on the centreline, half beam, section fullness).
# FULLNESS is the exponent in z = z_keel + (B - z_keel) sin(pi u / 2)^p and it
# is what makes this read as a stern rather than as a hull: p well UNDER 1 is a
# full, flat-bottomed section that carries its beam right down to the keel,
# which is exactly what a counter over a pair of propellers is. The first cut
# used p > 1 -- a fine V, which is what a bow looks like -- and the result put
# the hull's surface 1.5 m above the blade tips at the shaft's own z, so the
# screws hung in open water under a distant roof and the stern read as scenery
# rather than as the thing they are bolted to.
#
# THE TIP CLEARANCE IS THE NUMBER TO CHECK and the banner prints it: the
# underside directly over each disc lands ~0.45 m over a 0.90 m tip, which is
# 0.25 D and is what a real twin-screw arrangement is drawn with.
#
# THE BEAM AT THE TRANSOM IS THE SECOND THING THE FIRST CUT GOT WRONG. It
# narrowed to a 2.1 m transom over a 2.34 m shaft spacing, so the shafts stuck
# out past the sides of the ship and the stern read as a wedge somebody had
# sharpened. A transom stern carries most of its beam right to the after
# perpendicular -- that is what makes it a transom -- so the run aft loses
# height much faster than it loses width, and the counter over the screws is a
# BROAD flat roof rather than a taper.
HULL_STATIONS = (
    (-8.60, -1.38, 2.94, 0.34),
    (-6.40, -1.24, 2.92, 0.34),
    (-4.60, -1.02, 2.86, 0.33),
    (-3.30, -0.72, 2.76, 0.32),
    (-2.35, -0.36, 2.64, 0.31),
    (-1.55, +0.06, 2.50, 0.30),
    (-0.85, +0.56, 2.36, 0.29),
    (-0.25, +1.10, 2.22, 0.28),
    (+0.25, +1.46, 2.08, 0.27),
    (+0.62, +1.86, 1.88, 0.27),
    # ── AND THEN IT FAIRS AWAY UPWARD, WHICH IS THE THIRD FIX ───────────────
    # Ending the run at x +0.62 left a flat transom plate 2.5 m wide and 1.5 m
    # tall in the middle of the side shot, and at the demo's own framing that
    # is what it read as: a shard. A COUNTER does not end in a plate under
    # water -- its buttocks sweep up and aft and the hull leaves the water
    # somewhere above the frame -- so the last two stations lift the underside
    # clear through the surface and pull the beam in with it. What is left in
    # frame is one long curve, which is a silhouette a stern can be recognised
    # from, and the cap that closes the solid is now a small quad above the
    # waterline that nothing will ever see.
    (+0.90, +2.48, 1.46, 0.28),
    (+1.06, +3.12, 0.88, 0.30),
)
HULL_KEEL_W = 0.06      # half-width of the flat at the keel: no degenerate ring
HULL_RING = 20           # points up ONE side of a section


def loft(rings, xs):
    """A closed solid from a rake of same-length (y, z) rings at stations xs.

    The blade's loft at ship scale, and with BOTH of its traps handled by
    construction rather than by getting them right:

    WINDING. Which cross product points outward is exactly what went wrong at
    f2010a73 and exactly what the mirrored blade above had to be careful about
    all over again. Here each piece is oriented against a criterion it cannot
    argue with -- the shell's first quad must face AWAY from its own ring
    centre, and each cap must face along +-x -- and the whole piece is reversed
    if it does not. A test beats a convention.

    CAP VERTICES. Fan-capping the END RINGS in place is the obvious way to
    close the solid and it is what the first cut did.
    compute_vertex_normals() then averages the transom's own +x normal into the
    shell's sideways one at every vertex they share, so the hard corner a
    transom IS gets smoothed away -- and eight metres of ship came out looking
    like an inflated balloon with propellers under it. The caps get their own
    copies of the rings, so the edge survives. Same class of mistake as the
    winding one, and just as invisible in the diff.
    """
    import numpy as np
    k, m = len(rings), len(rings[0])
    v = np.empty((k, m, 3), np.float32)
    for i, (x, ring) in enumerate(zip(xs, rings)):
        v[i, :, 0] = x
        v[i, :, 1] = [p[0] for p in ring]
        v[i, :, 2] = [p[1] for p in ring]

    def oriented(tris, verts, want):
        """tris wound so the FIRST one's normal points the way `want` does."""
        n = np.cross(verts[tris[1]] - verts[tris[0]],
                     verts[tris[2]] - verts[tris[0]])
        return tris if float(np.dot(n, want)) > 0.0 else \
            [tris[q + o] for q in range(0, len(tris), 3) for o in (2, 1, 0)]

    shell = []
    for i in range(k - 1):
        for j in range(m):
            j2 = (j + 1) % m
            a0, b0 = i * m + j, i * m + j2
            a1, b1 = (i + 1) * m + j, (i + 1) * m + j2
            shell += [a0, b1, a1, a0, b0, b1]
    flat = v.reshape(-1, 3)
    ctr0 = np.array([0.0, v[0, :, 1].mean(), v[0, :, 2].mean()], np.float32)
    shell = oriented(shell, flat, flat[shell[0]] - ctr0 - np.array(
        [flat[shell[0]][0], 0.0, 0.0], np.float32))
    caps = np.concatenate([v[0], v[-1]], 0)
    mid = np.array([[xs[0], v[0, :, 1].mean(), v[0, :, 2].mean()],
                    [xs[-1], v[-1, :, 1].mean(), v[-1, :, 2].mean()]],
                   np.float32)
    v = np.concatenate([flat, caps, mid], 0)
    idx = list(shell)
    for c, want in ((0, np.array([-1.0, 0.0, 0.0], np.float32)),
                    (1, np.array([1.0, 0.0, 0.0], np.float32))):
        b, cv = k * m + c * m, k * m + 2 * m + c
        fan = []
        for j in range(m):
            fan += [cv, b + j, b + (j + 1) % m]
        idx += oriented(fan, v, want)
    # uv: v runs fore to aft along the stations, u around the girth as a
    # triangle wave so the two sides mirror and the seam carries no stripe --
    # the same trick, and for the same reason, as the blade's. The caps take
    # the uv of the ring they were copied from.
    uv = np.empty((k, m, 2), np.float32)
    gir = np.abs(2.0 * np.arange(m, dtype=np.float32) / m - 1.0)
    uv[:, :, 0] = gir[None, :] * 3.0
    uv[:, :, 1] = np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None] * 6.0
    uv = np.concatenate([uv.reshape(-1, 2), uv[0], uv[-1],
                         np.zeros((2, 2), np.float32)], 0)
    g = tp.BufferGeometry()
    g.set_attribute("position", v)
    g.set_attribute("uv", uv)
    g.set_index(np.asarray(idx, np.uint32))
    g.compute_vertex_normals()
    return g


def hull_ring(y_bot, half_beam, p):
    """One station, as a closed ring of (y, z) from the keel round to the keel.

    Up the port side to the deck edge, straight across the deck, down the
    starboard side. The keel carries a small flat so the ring does not close on
    a degenerate edge, which is what would put a NaN in the vertex normals of
    the two triangles either side of it."""
    out = []
    for sign in (1.0, -1.0):
        rng = range(HULL_RING) if sign > 0 else range(HULL_RING - 1, -1, -1)
        for j in rng:
            u = j / float(HULL_RING - 1)
            z = HULL_KEEL_W + (half_beam - HULL_KEEL_W) \
                * math.sin(0.5 * math.pi * u) ** p
            out.append((y_bot + (HULL_TOP - y_bot) * u, sign * z))
    return out


def hull_y(x, z):
    """The underside's height at (x, z), by the same expression the loft uses.
    The banner quotes the tip clearance out of this rather than out of a
    number somebody measured off a screenshot."""
    xs = [s[0] for s in HULL_STATIONS]
    i = max(0, min(len(xs) - 2, next((k for k in range(len(xs) - 1)
                                      if x < xs[k + 1]), len(xs) - 2)))
    f = (x - xs[i]) / (xs[i + 1] - xs[i])
    y0 = HULL_STATIONS[i][1] + f * (HULL_STATIONS[i + 1][1] - HULL_STATIONS[i][1])
    bm = HULL_STATIONS[i][2] + f * (HULL_STATIONS[i + 1][2] - HULL_STATIONS[i][2])
    pp = HULL_STATIONS[i][3] + f * (HULL_STATIONS[i + 1][3] - HULL_STATIONS[i][3])
    q = min(max((abs(z) - HULL_KEEL_W) / max(bm - HULL_KEEL_W, 1e-6), 0.0), 1.0)
    u = 2.0 / math.pi * math.asin(min(q ** (1.0 / pp), 1.0))
    return y0 + (HULL_TOP - y0) * u


# ── The paint ───────────────────────────────────────────────────────────────
# ANTIFOULING RED, and it is a working choice rather than a decorative one: it
# is what the underwater half of nearly every ship is painted, it is the one
# colour that says "this is below the waterline" without a caption, and a matte
# oxide red hides modelling sins that a bare grey would broadcast. The whole of
# this hull is below the waterline as far as the camera is concerned -- the
# surface is 3 m over the shaft and the widest framing in the film tops out
# well under it -- so there is no boot top and no topside colour here, and
# authoring them would be authoring pixels nothing can see.
#
# The same wear maps the bronze uses, at a coarser tile: roughness and normal
# only, never albedo. Antifouling weathers and fouls, it does not rust through.
antifoul = tp.MeshStandardMaterial()
#
# AND IT IS A DARK RED, WHICH IS THE POINT RATHER THAN A COMPROMISE. Red is
# the first thing three metres of water takes out of daylight, and the key
# here is already blue-shifted for exactly that reason -- so a bright oxide
# albedo under it comes back as a cartoon slab of poster paint, which is what
# the first render of this hull was. Dropped to 0.10: what reaches the camera
# is a dark warm brown that reads as red only where the rake grazes it, which
# is what antifouling actually looks like at this depth.
antifoul.color = tp.Color(0.118, 0.029, 0.023)      # linear oxide red
antifoul.metalness = 0.0
antifoul.roughness = 0.90
_hn = rust_map("nor_gl", False)
if _hn is not None:
    antifoul.normal_map = _hn
    antifoul.roughness_map = rust_map("rough", False)
    antifoul.normal_scale = tp.Vector2(0.40, 0.40)

hull = tp.Group()
hull.add(tp.Mesh(loft([hull_ring(s[1], s[2], s[3]) for s in HULL_STATIONS],
                      [s[0] for s in HULL_STATIONS]), antifoul))
# The SKEG: the centreline fin between the shafts. It is the single feature
# that stops the stern reading as a slab -- it gives the underbody a keel line
# to run aft along, and it is what a twin-screw ship carries there instead of
# the single-screw deadwood. Same loft, five stations, a tapered plate.
#
# ITS THICKNESS IS SET BY THE WAKES and not by taste: the two slipstream tubes
# reach in to |z| = SHAFT_Z - R_TIP = 0.27 m at the disc, so a skeg wider than
# that would have the slipstream growing out of the inside of it.
SKEG_T = 0.22            # half-thickness at mid-depth, m
SKEG = ((-4.20, -1.28), (-2.80, -1.06), (-1.60, -0.74),
        (-0.55, -0.28), (0.10, +0.36))       # (x, the bottom of the fin)


def skeg_ring(x, y_low, n=9):
    """A lens section from the fin's bottom up into the hull's underbody. The
    same small flat at both ends as the hull's keel, and for the same reason:
    a ring that closes on a point closes on a degenerate triangle."""
    top = hull_y(x, 0.0) + 0.30
    out = []
    for sign in (1.0, -1.0):
        rng = range(n) if sign > 0 else range(n - 1, -1, -1)
        for j in rng:
            u = j / float(n - 1)
            out.append((y_low + (top - y_low) * u,
                        sign * (0.035 + SKEG_T * math.sin(math.pi * u))))
    return out


hull.add(tp.Mesh(loft([skeg_ring(x, y) for x, y in SKEG],
                      [s[0] for s in SKEG]), antifoul))
# The BOSSINGS: each shaft has to come OUT of something. A shaft hanging in
# open water with nothing holding it is the single most amateur thing a stern
# can do, and it is what the first render of this showed. A bossing -- the
# hull's own plating faired out around the shaft, which is what a twin-screw
# ship with shallow shaft angles carries instead of A-brackets -- is a cone,
# and a cone is the cheapest possible fix that is also the right answer.
BOSS_G = tp.CylinderGeometry(0.165, 0.72, 2.35, 26)
for sd in (1.0, -1.0):
    boss = tp.Mesh(BOSS_G, antifoul)
    boss.rotate_z(math.pi / 2)
    boss.position.set(-1.90, -0.06, sd * SHAFT_Z)
    hull.add(boss)
scene.add(hull)

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
# beta_now / rpm_now / va_now are what the three levers ARE this frame -- the
# sliders, or the scripted schedules when there is no UI. hist_np[k] =
# (omega, theta, v_i, V_a) as they stood on frame k, and it is the only channel
# through which the past reaches the kernel.
#
# THE RING WIDENED FROM vec3 TO vec4 AND THE AUDIT IS THE INTERESTING PART.
# The old three slots were (loading, omega, theta) and the loading no longer
# exists -- there is a THRUST now, and thrust needs the speed of advance to be
# defined at all. Every consumer of hist[] in the kernel had to be walked:
# the azimuth moved from h[2] to h[1], the loading term became K_T / K_T_design
# derived from v_i and V_a, and the cavitation criterion, which used to read
# the loading directly, is now Burrill's on the thrust the same two slots give.
# Nothing that a parcel reads at birth is passed as a kernel parameter any
# more except the constants, which is the property the causal beat rests on.
#
# Pre-filled at a STOPPED shaft (omega 0, theta 0, v_i 0) carrying the starting
# ship speed, so that every index a slot can reach is defined before frame 1.
import numpy as _np
beta_now = PITCH
rpm_now = RPS * 60.0
# The SLIDER is the ship's speed through the water; the model's input is the
# speed of ADVANCE, which is that reduced by the mean wake fraction. Keeping
# both names is not redundancy -- it is the one place the hull enters the
# momentum chain, and mixing them up would quietly move every J on the panel.
vs_now = V_SHIP
va_now = V_A
omega_now = 0.0
theta_now = 0.0
state_now = PropState(0.0, PITCH, V_A)   # the model at rest; advance() restamps
hist_np = _np.zeros((HIST_N, 4), _np.float32)
hist_np[:, 3] = V_A
hist_wp = wp.array(hist_np, dtype=wp.vec4, device=device)
# The second ring: the two DERIVED scalars, so the kernel does not recompute
# a per-frame quantity per parcel. See the kernel's own note on the split.
# It is a vec4 now, not a vec2, and the third slot is the hull's: the wake
# lobe's amplitude at this frame's own operating point (wake_lobe_k). It rides
# the ring for the same reason everything else here does -- a parcel shed
# before the ship had way on must not start breathing because the ship has way
# on NOW. The fourth slot is spare.
helm_np = _np.zeros((HIST_N, 4), _np.float32)
helm_np[:, 0] = 1.0                 # a stopped shaft still has a pitch
helm_wp = wp.array(helm_np, dtype=wp.vec4, device=device)
# ── THE PLANFORM, AS THE KERNEL WANTS IT ────────────────────────────────────
# The same PF_C / PF_T control points the loft interpolates, resampled ONCE
# onto a uniform span grid so a parcel's lookup is an index and a lerp rather
# than a ten-step search through an irregular table. 33 stations over a span
# the loft itself draws with 18, so the sheet's chord and thickness agree with
# the casting's to better than the grain it is drawn with. It is a table, not
# a second model: change PF_C and both move.
PLAN_TAB = wp.array(
    _np.stack([_np.interp(_np.linspace(0.0, 1.0, PLAN_N), PF_F, PF_C),
               _np.interp(_np.linspace(0.0, 1.0, PLAN_N), PF_F, PF_T)],
              axis=1).astype(_np.float32),
    dtype=wp.vec2, device=device)
# The blade's pivot angle THIS FRAME, in radians, and it is the same number
# set_pitch hands the scene graph -- see advance(). The sheet is on the blade,
# so it turns in the flange with it.
dpitch_now = 0.0


def launch(out_pos, out_col):
    wp.launch(shed, dim=N_MAIN, device=device,
              inputs=[out_pos, out_col, hist_wp, helm_wp, sim_time,
                      N_MAIN, BLADES,
                      TURB, SHEET_EFF, HUB_EFF,
                      W_BURST, SPRITE_R0, FLUX_P, BRIGHT,
                      T_IN, IN_SHARE, HIST_N,
                      TIP_HINT, VAP_GAIN, BUB_CAV])
    # The blade sheet, over the tail of the same two buffers. A second launch
    # rather than a fourth branch, and the eleven milliseconds that bought are
    # written up beside the kernel.
    if N_CAV > 0:
        wp.launch(sheet_cav, dim=N_CAV, device=device,
                  inputs=[out_pos, out_col, hist_wp, helm_wp, PLAN_TAB,
                          sim_time, N_MAIN, BLADES,
                          SPRITE_R0, FLUX_P, BRIGHT, HIST_N,
                          SHEET_CAV_INC, dpitch_now])


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
    global sim_time, frame_no, beta_now, rpm_now, omega_now, theta_now, state_now
    global dpitch_now
    frame_no += 1
    sim_time = frame_no * DT
    # AND THE SEA IS ON THE SAME CLOCK, which the Ocean made non-optional. Every
    # renderer-side animated field -- the FFT deform above all -- reads a WALL
    # clock unless it is pinned, and this demo's whole contract is that a frame
    # is a pure function of its index: --shot 5.5 has to be the frame --film
    # renders at 5.5 s, and the wake was always careful about that while there
    # was nothing else moving. Add a surface and leave it unpinned and the sea
    # is at whatever phase 20 s of offline rendering put it at, so two runs of
    # the same command disagree and --takes cannot re-cut a beat in place.
    # Pinned here, one line, once per frame, before render().
    renderer.sim_time = sim_time
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
    # The starboard screw is the port one reflected, and a reflection negates
    # the rotation about an axis lying IN its plane. The phase offset goes in
    # before the negation so it matches the kernel's own phi_c, which adds it
    # to the crossing azimuth in the un-reflected frame.
    prop_s.rotation.x = -(theta_now + PHASE_OFF)
    # The spindle angle set_pitch just gave every blade, taken FROM it rather
    # than recomputed beside it, for the same reason theta is integrated in one
    # place: what the reader sees the blade doing and what the sheet is drawn
    # on have to be the same number and not two agreeing ones.
    dpitch_now = set_pitch(beta_now)
    # THE MODEL IS EVALUATED ON THE SHAFT SPEED THE SHAFT ACTUALLY HAS, not on
    # the one the slider commands: during the spin-up ramp the two differ by up
    # to a factor of the ramp fraction, and J, K_T and the thrust that follows
    # them all belong to the turning shaft. The panel reads the same object.
    state_now = PropState(omega_now * 60.0 / TWO_PI, beta_now, va_now)
    hist_np[frame_no % HIST_N] = (omega_now, theta_now,
                                  state_now.vi, state_now.va)
    helm_np[frame_no % HIST_N] = (
        min(max(state_now.kt / KT_DESIGN, 0.0), 4.0), burrill(state_now)[3],
        wake_lobe_k(state_now), 0.0)
    hist_wp.assign(hist_np)      # 6 x 144 floats: ~3 kB/frame, below measurable
    helm_wp.assign(helm_np)


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
ST0 = PropState(RPS * 60.0, PITCH, V_A)          # the commanded operating point
UD0, UW0 = ST0.creep()
IN_D_MAX = UD0 * T_IN / ((1.0 - max(min(V_A / max(UD0, 1e-4), 1.0), IN_F))
                         * IN_K + max(min(V_A / max(UD0, 1e-4), 1.0), IN_F))
_gd = 1.0 - IN_D_MAX / math.hypot(IN_D_MAX, IN_RV)
_fc = min(max(V_A / max(UD0, 1e-4), 0.0), 1.0)
IN_R_MOUTH = R_TIP * (1.0 + IN_FLARE_HI * (IN_LINES_R - 0.5) / IN_LINES_R * (
    1.0 / math.sqrt(max(_fc + (1.0 - _fc) * _gd, 0.02)) - 1.0))


def cav_ratio(beta_deg, rps, va=None):
    """The Burrill ratio for a tip filament shed at this helm. > 1 is
    cavitating. The kernel's own expression, on the host, so the banner and the
    panel state the number the wake was actually drawn with."""
    return burrill(PropState(rps * 60.0, beta_deg,
                             V_A if va is None else va))[3]


def sheet_extent(cav_ratio_now, lam=1.0):
    """The blade sheet's extent fraction at this Burrill ratio, 0 = none.

    The kernel's own two lines, on the host, so the panel and the banner quote
    the number the blades were actually drawn with. `lam` is the local loading
    factor -- 1 for the disc-mean answer, 1 + k (1 - lobe_mean) at twelve
    o'clock -- which is what makes the panel's pulse figure a measurement of
    the same expression rather than a second one."""
    exs = max(cav_ratio_now * lam / SHEET_CAV_INC - 1.0, 0.0)
    return 1.0 - math.exp(-exs / SHEET_CAV_KNEE)


def sheet_pulse(st):
    """(extent at six o'clock, at the disc mean, at twelve o'clock).

    The once-per-rev breathing as a triple, straight out of sheet_extent at the
    lobe's own two extremes. At the bollard lamk is zero and all three are the
    same number, which is the control being derived rather than authored."""
    lk = wake_lobe_k(st)
    cr = burrill(st)[3]
    return (sheet_extent(cr, 1.0 - lk * W_LOBE_MEAN),
            sheet_extent(cr, 1.0),
            sheet_extent(cr, 1.0 + lk * (1.0 - W_LOBE_MEAN)))


_SG, _TC, _L5, _CR = burrill(ST0)
_SHX = sheet_pulse(ST0)
print(f"       prop:   B{BLADES}-{EAR * 100:.0f}, D {D_PROP:g} m, "
      f"EAR {EAR:g}, wake {LIFETIME:.2f} s over "
      f"{wake_reach(UD0, UW0, LIFETIME):.2f} m\n"
      f"       shaft:  {RPS * 60:g} rpm (ref {RPS_REF * 60:g}, "
      f"ramp {T_RAMP:g} s)"
      + (f" -> {RPS_TO * 60:g} rpm at t={RPS_AT:g} s" if RPS_AT < 1e8 else "")
      + f", tip {TWO_PI * RPS * R_TIP:.1f} m/s"
      + f", slipstream {UW0:.2f} m/s, "
      f"advance {(UW0 / RPS if RPS > 0 else 0):.2f} m/rev\n"
      f"       pitch:  beta {PITCH:g} deg (design {BETA_DESIGN:g}), "
      f"P/D {ST0.pd:.3f}"
      + (f" -> {PITCH_TO:g} deg (P/D {pd_of(PITCH_TO):.3f}) at t={PITCH_AT:g} s"
         if PITCH_AT < 1e8 else "")
      + f", helm history {HIST_N} frames = {HIST_N * DT:.2f} s\n"
      f"       open water: J {ST0.j:.3f}, K_T {ST0.kt:.4f}, "
      f"10 K_Q {10 * ST0.kq:.4f}, eta_0 {ST0.eta:.3f}"
      + ("   [OUTSIDE B-SERIES VALIDITY -- EXTRAPOLATED]"
         if ST0.extrapolated else "") + "\n"
      f"               T {ST0.thrust / 1e3:.2f} kN, Q {ST0.torque / 1e3:.2f} kNm, "
      f"P_D {ST0.power / 1e3:.1f} kW, V_a {V_A:g} m/s, "
      f"v_i {ST0.vi:.2f} m/s\n"
      f"       water:  rho {RHO:g}, {DEPTH:g} m down, "
      f"sigma_0.7R {_SG:.2f}, tau_c {_TC:.3f} "
      f"(Burrill 5% line {_L5:.3f}); tips at {_CR:.2f} x, "
      f"hub at {_CR * HUB_CAV:.2f} x "
      + ("-> CAVITATING" if _CR > 1.0
         else ("-> hub rope only" if _CR * HUB_CAV > 1.0
               else "-> clear")) + "\n"
      f"               inception at this pitch: hub "
      f"{inception_rpm(PITCH, V_A, HUB_CAV):.0f} rpm, "
      f"tips {inception_rpm(PITCH, V_A, 1.0):.0f} rpm, "
      f"sheet {inception_rpm(PITCH, V_A, 1.0 / SHEET_CAV_INC):.0f} rpm\n"
      f"       sheet:  back cavitation at {SHEET_CAV_INC:.2f} x the tips' line, "
      f"{_CR / SHEET_CAV_INC:.2f} x here -> extent {_SHX[1]:.2f}"
      + (f", breathing {_SHX[0]:.2f} at 6 o'clock -> {_SHX[2]:.2f} at 12"
         if _SHX[2] - _SHX[0] > 1.0e-3 else " (steady: no way on)") + "\n"
      f"       inflow: {IN_SHARE:.0%} of slots, {T_IN:g} s upstream "
      f"({T_IN / PERIOD:.0%} of the period), reaching {IN_D_MAX:.2f} m ahead of "
      f"the disc at {IN_R_MOUTH:.2f} m = {IN_R_MOUTH / R_TIP:.1f} x the tip radius\n"
      f"       stern:  twin screws at z {SHAFT_Z:+.2f} / {-SHAFT_Z:+.2f} m "
      f"(port left-handed, starboard right, tops outboard), "
      f"phase {math.degrees(PHASE_OFF):.0f} deg apart\n"
      f"       hull:   stern section {HULL_X1 - HULL_X0:.1f} m, transom at "
      f"x {HULL_X1:+.2f}, counter {hull_y(0.0, SHAFT_Z):.2f} m over the shaft "
      f"= {hull_y(0.0, SHAFT_Z) - R_TIP:.2f} m tip clearance "
      f"({(hull_y(0.0, SHAFT_Z) - R_TIP) / D_PROP:.2f} D)\n"
      f"       wake:   w_mean {W_MEAN:.2f}, w_peak {W_PEAK:.2f} at TDC "
      f"(w {W_MEAN - W_PEAK * W_LOBE_MEAN:.2f} at the bottom, "
      f"{W_MEAN + W_PEAK * (1.0 - W_LOBE_MEAN):.2f} at the top); "
      f"V_ship {V_SHIP:g} -> V_a {V_A:.2f} m/s\n"
      f"               once-per-rev loading swing "
      f"{100.0 * wake_lobe_k(ST0) * (1.0 - W_LOBE_MEAN):+.0f}% / "
      f"{-100.0 * wake_lobe_k(ST0) * W_LOBE_MEAN:+.0f}% "
      + ("(BOLLARD: no way on, no boundary layer, no breathing)"
         if wake_lobe_k(ST0) <= 0.0 else "on the cavitation criterion") + "\n"
      f"       volume: {BOX_RES}^3 over {2 * BOX_HALF[0]:.1f} x "
      f"{2 * BOX_HALF[1]:.1f} x {2 * BOX_HALF[2]:.1f} m "
      f"({1e3 * 2 * BOX_HALF[0] / BOX_RES:.1f} x "
      f"{1e3 * 2 * BOX_HALF[1] / BOX_RES:.1f} x "
      f"{1e3 * 2 * BOX_HALF[2] / BOX_RES:.1f} mm voxels, "
      f"{6 * BOX_RES ** 3 / 1e6:.0f} MB r32ui+r16f), "
      f"sigma/particle {SIGMA:g}\n"
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
    # was longer than the air version's whole 0.85 s lifetime; at 1.7 s it is
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
elif FILM:
    # ── THE FILM ────────────────────────────────────────────────────────────
    # 55 s at 60 fps, and the physics is the story: inception is the third act
    # and everything before it exists to make the audience feel the threshold
    # before they see it crossed. Six beats, ONE continuous simulation and ONE
    # continuous camera move -- there is not a single cut in it, and that is
    # not restraint for its own sake. The wake carries 1.7 s of history, so a
    # cut throws away the one thing this demo is about; the lag between a lever
    # and the water is only legible if the camera stays on the water while it
    # happens.
    #
    #   1 OPEN       0.0 -  7.0   10 rpm, beta 22. Bronze in deep water, motes
    #                             drifting, the funnel collapsed onto the disc.
    #   2 SPIN-UP    7.0 - 15.0   -> 120 rpm. The funnel establishes and the
    #                             bubble column grows. CLEAN: tips at 0.83 of
    #                             the margin, hub at 0.93. No ropes.
    #   3 INCEPTION 15.0 - 26.0   120 -> 138 rpm and HOLD. The hub crosses at
    #                             127.8 rpm and the tips not until 141.2, so
    #                             138 is a 4 s window with the axial rope lit
    #                             and the helices still clear -- the thing real
    #                             CPP footage shows and the reason the beat is
    #                             shaped this way instead of as one ramp. Then
    #                             138 -> 300 and the tips bloom.
    #   4 COLLAPSE  26.0 - 34.0   300 -> 180 rpm, a HARD step. The rope length
    #                             gate falls from 1.00 to 0.40 of a life, the
    #                             fast slug already downstream keeps its own
    #                             speed, and the rarefaction between the two
    #                             convects off frame.
    #   5 FEATHER   34.0 - 41.5   beta 22 -> 5 at 180 rpm. The other lever,
    #                             and the causality lesson in one shot: the
    #                             funnel dies in a frame, the wake takes 1.7 s
    #                             to age out, and the propeller never stops.
    #   6 FINALE    41.5 - 55.0   beta 5 -> 30. Full re-bloom (2.08 x inception)
    #                             and a slow pull back to the widest framing
    #                             the murk allows: funnel, prop, slipstream.
    #
    # THE FILM IS A PURE FUNCTION OF THE FRAME INDEX. The helm is a keyframe
    # table read at film time, the camera is a cubic Hermite track read at film
    # time, and the sim clock is frame_no * DT exactly as it is everywhere else
    # -- so --takes re-renders one beat into the same picture the full run made.
    import glob
    import subprocess

    from PIL import Image

    # CWD-relative, as --shot's own output is and as the other films are: an
    # 11-minute render should not land in the source tree because that is where
    # the script happens to live.
    FILM_DIR = cli_arg("--film-dir", "prop_vortex_film", str)
    CRF = cli_arg("--crf", 17, int)
    PROBE = cli_arg("--probe", 0, int)
    ONLY = [s.strip() for s in cli_arg("--takes", "", str).split(",") if s.strip()]
    # The pre-roll is not padding. The wake is LIFETIME deep and the shaft has
    # a T_RAMP spin-up, so frame 0 of the film has to be preceded by at least
    # both of them or the opening shot is a wake still growing into existence
    # under a ramp the schedule does not know about. It is rendered (the
    # temporal history needs it too) and thrown away.
    PREROLL = int(round((T_RAMP + LIFETIME + 0.2) * FPS))
    LEAD = 24        # discarded frames before a captured beat in a --takes run

    # ── The helm schedule ───────────────────────────────────────────────────
    # (film time, value, how to approach it). "ease" is a smoothstep, "hold" a
    # constant, "step" an instantaneous jump -- written as a second key at the
    # SAME time, which is what makes a step a step and not a one-frame ramp.
    RPM_KEYS = [(0.0, 10.0),
                (7.0, 10.0, "hold"),        # 1 OPEN: barely turning
                (10.0, 120.0, "ease"),      # 2 SPIN-UP
                (15.0, 120.0, "hold"),
                # 138, NOT 130. Burrill moved both inception speeds -- the hub
                # crosses at 127.8 rpm now and the tips at 141.2, where the
                # authored gate had them at 124.6 and 131.8 -- so the hub-only
                # window is 13 rpm wide instead of 7 and this is the rpm that
                # sits in it exactly as 130 sat in the old one: hub +9.1% over
                # its inception line, tips 2.6% short of theirs. The beat is
                # unchanged; the number under it is now derived.
                (18.5, 138.0, "ease"),      # 3 INCEPTION: into the hub's window
                (21.0, 138.0, "hold"),      #   HUB ROPE ONLY lives here
                (24.5, 300.0, "ease"),      #   and the tips bloom
                (26.0, 300.0, "hold"),
                (26.0, 180.0, "step"),      # 4 COLLAPSE: the throttle's step
                (55.0, 180.0, "hold")]
    BETA_KEYS = [(0.0, BETA_DESIGN),
                 (34.0, BETA_DESIGN, "hold"),
                 (34.0, 5.0, "step"),       # 5 FEATHER
                 (41.5, 5.0, "hold"),
                 (41.5, 30.0, "step"),      # 6 FINALE
                 (55.0, 30.0, "hold")]
    BEATS = [("open", 7.0), ("spinup", 15.0), ("inception", 26.0),
             ("collapse", 34.0), ("feather", 41.5), ("finale", 55.0)]
    FILM_SECS = BEATS[-1][1]

    def keyed(keys, t):
        prev = keys[0]
        for k in keys[1:]:
            if t < k[0]:
                shape = k[2] if len(k) > 2 else "ease"
                if shape == "hold" or k[0] <= prev[0]:
                    return prev[1]
                u = (t - prev[0]) / (k[0] - prev[0])
                return prev[1] + (k[1] - prev[1]) * (
                    u * u * (3.0 - 2.0 * u) if shape == "ease" else u)
            prev = k
        return prev[1]

    # advance() reads these two names out of the module globals every frame, so
    # rebinding them here is the whole of installing the film's schedule -- the
    # helm ring, the blade angles and the banner all follow with no other edit.
    FILM_T0 = PREROLL * DT

    def scheduled_rps(t):                                       # noqa: F811
        return keyed(RPM_KEYS, max(t - FILM_T0, 0.0)) / 60.0

    def scheduled_beta(t):                                      # noqa: F811
        return keyed(BETA_KEYS, max(t - FILM_T0, 0.0))

    # ── The camera track ────────────────────────────────────────────────────
    # (film time, eye, look-at, stop). A cubic Hermite through the keys with
    # finite-difference tangents taken over the ACTUAL times, so the spline is
    # C1 and the camera never restarts at a key -- a smoothstep between every
    # pair would ease in and out fourteen times over 55 s and read as a series
    # of nudges. `stop` zeroes the tangent, which is how the first and last
    # frames sit still.
    CAM_KEYS = [
        (0.0, (1.90, 0.95, 2.95), (0.30, 0.00, 0.00), True),
        (7.0, (1.05, 0.60, 2.55), (0.25, -0.02, 0.10), False),
        # 2: out to the mouth, where the funnel is legible and the disc is not
        (11.0, (0.30, 0.75, 4.60), (0.85, -0.05, 0.00), False),
        (15.0, (0.60, 0.65, 3.55), (1.05, -0.04, 0.00), False),
        # 3: the push in. Inception happens ON SCREEN and this is why.
        (18.0, (1.35, 0.42, 1.95), (0.55, -0.02, 0.00), False),
        (21.0, (1.30, 0.45, 2.20), (0.75, -0.02, 0.00), False),
        (26.0, (1.55, 0.80, 3.90), (1.65, -0.03, 0.00), False),
        # 4: wide enough that the rarefaction has somewhere to travel
        (30.0, (2.20, 0.95, 6.00), (2.60, -0.04, 0.00), False),
        (34.0, (1.95, 0.85, 5.30), (2.20, -0.04, 0.00), False),
        # 5: back in, because the point of the beat is a prop that is TURNING
        (38.0, (1.25, 0.58, 3.30), (0.95, -0.02, 0.00), False),
        (41.5, (1.10, 0.52, 3.00), (0.80, -0.02, 0.00), False),
        # 6: hold for the re-bloom, then the pull back
        (45.0, (1.20, 0.62, 3.40), (1.15, -0.02, 0.00), False),
        (50.0, (1.70, 1.05, 5.10), (2.00, -0.05, 0.00), False),
        (55.0, (2.00, 1.40, 6.90), (2.40, -0.07, 0.00), True),
    ]

    def _tangents(idx):
        ts = [k[0] for k in CAM_KEYS]
        out = []
        for i, k in enumerate(CAM_KEYS):
            if k[3] or i == 0 or i == len(CAM_KEYS) - 1:
                out.append((0.0, 0.0, 0.0))
            else:
                dt = ts[i + 1] - ts[i - 1]
                out.append(tuple((CAM_KEYS[i + 1][idx][c] - CAM_KEYS[i - 1][idx][c])
                                 / dt for c in range(3)))
        return out

    TAN_E, TAN_T = _tangents(1), _tangents(2)

    def cam_at(t):
        t = min(max(t, CAM_KEYS[0][0]), CAM_KEYS[-1][0])
        i = 0
        while i < len(CAM_KEYS) - 2 and t >= CAM_KEYS[i + 1][0]:
            i += 1
        h = CAM_KEYS[i + 1][0] - CAM_KEYS[i][0]
        u = (t - CAM_KEYS[i][0]) / h
        u2, u3 = u * u, u * u * u
        h00, h10 = 2 * u3 - 3 * u2 + 1, u3 - 2 * u2 + u
        h01, h11 = -2 * u3 + 3 * u2, u3 - u2
        out = []
        for idx, tan in ((1, TAN_E), (2, TAN_T)):
            a, b = CAM_KEYS[i][idx], CAM_KEYS[i + 1][idx]
            out.append(tuple(h00 * a[c] + h10 * h * tan[i][c]
                             + h01 * b[c] + h11 * h * tan[i + 1][c]
                             for c in range(3)))
        return out

    # ── The banner, and the clearance check that goes with it ───────────────
    # A keyframed track can be right at every key and still put the eye inside
    # the slipstream between two of them, so the track is SAMPLED before a
    # frame is rendered and the closest approach to the prop and to the wake
    # axis is stated. R_TIP + the turbulence is ~1.2 m; anything under that is
    # a camera about to fly through its own subject.
    # ── AND THE CHECK IS AGAINST BOTH SHAFTS NOW, WHICH IS THE WHOLE POINT ──
    # It used to measure the eye against ONE axis at z = 0 and one hub at the
    # origin. There are two of each and neither is where those were, so the
    # check that was supposed to catch a camera flying through its own subject
    # was measuring the clear water BETWEEN the screws and reporting it as
    # clearance. Measuring against both is what turns the composition failure
    # below from an opinion into a number.
    def _clear(t):
        e = cam_at(t)[0]
        return (min(math.dist(e, (0.0, 0.0, sd * SHAFT_Z)) for sd in (1, -1)),
                min(math.hypot(e[1], e[2] - sd * SHAFT_Z) for sd in (1, -1)))
    d_prop = min(_clear(t * FILM_SECS / 600.0)[0] for t in range(601))
    d_axis = min(_clear(t * FILM_SECS / 600.0)[1] for t in range(601))
    print(f"\n       FILM  {FILM_SECS:.1f} s at {FPS} fps = "
          f"{int(round(FILM_SECS * FPS))} frames, {W}x{H}, {N:,} parcels\n"
          f"       track closest approach: {d_prop:.2f} m to the nearer hub, "
          f"{d_axis:.2f} m off the nearer wake axis (tips at {R_TIP:.2f} m)")
    # ── R6: THE FILM IS VERIFIED, NOT RE-CHOREOGRAPHED ──────────────────────
    # The camera track and the six beats were composed for ONE screw turning on
    # the axis, and the acceptance test for this revision was to render the
    # beat keyframes once and look. Beat 3 does not survive: its push-in key
    # sits at (1.35, 0.42, 1.95), which was 0.58 m off a wake on the axis and
    # is 0.89 m off the PORT wake's axis now -- inside the tip radius. The
    # camera flies through the port slipstream and the shot that was supposed
    # to show inception happening on screen shows the inside of a propeller
    # instead. Re-directing it is a film-v2 slice and is deliberately NOT done
    # here: spending the look budget on a re-cut would have come out of the
    # breathing, which is what this revision is for. So the film says so, every
    # time it runs, with the number that proves it.
    if d_axis < R_TIP:
        print(f"\n       *** the film choreography predates the twin stern ***\n"
              f"       the track comes within {d_axis:.2f} m of a wake axis "
              f"(tips at {R_TIP:.2f} m), so at least one beat has the camera\n"
              f"       INSIDE a slipstream. beat 3's push-in is the one that "
              f"breaks. re-cutting the track is a film-v2 slice;\n"
              f"       the beats and the helm schedule are unchanged here.\n")
    for bi, (name, end) in enumerate(BEATS):
        start = 0.0 if bi == 0 else BEATS[bi - 1][1]
        mid = 0.5 * (start + end)
        for tag, tt in (("in", start + 0.05), ("mid", mid), ("out", end - 0.05)):
            bd, rp = keyed(BETA_KEYS, tt), keyed(RPM_KEYS, tt)
            cr = cav_ratio(bd, rp / 60.0)
            # Three rungs now, and the table says which one each beat is on.
            # The film's schedule was written when there were two, and beat 6's
            # 300 rpm is the only one that reaches the sheet -- which is worth
            # knowing before watching it and is the kind of thing a re-cut (see
            # the choreography warning above) would be composed around.
            state = "SHEET" if cr > SHEET_CAV_INC else (
                "CAVITATING" if cr > 1.0 else (
                    "hub rope only" if cr * HUB_CAV > 1.0 else "clear"))
            print(f"       {name:>9} {tag:>3} t={tt:5.1f}  {rp:5.1f} rpm  "
                  f"beta {bd:4.1f}  tips {cr:5.2f}x  hub {cr * HUB_CAV:5.2f}x  "
                  f"sheet {cr / SHEET_CAV_INC:5.2f}x  {state}")

    # ── The frames worth reading at 1:1 ─────────────────────────────────────
    # One per beat plus the two the beat structure is actually graded on: the
    # hub-only window, and the frame just after the throttle step.
    KEYSHOTS = [("b1_open", 3.0), ("b2_funnel", 9.5), ("b2_clean", 14.0),
                ("b3a_hub_only", 20.0), ("b3b_tips", 22.5),
                ("b3c_ropes_run", 25.8), ("b4_front", 27.2),
                ("b4_short", 32.0), ("b5_feather", 39.5),
                ("b6a_bloom", 44.5), ("b6_wide", 54.8)]
    shot_at = {int(round(t * FPS)): n for n, t in KEYSHOTS}

    os.makedirs(FILM_DIR, exist_ok=True)
    FFMPEG = find_ffmpeg()
    if FFMPEG is None:
        print("       no ffmpeg on PATH and no imageio-ffmpeg: frames only")
    log = open(os.path.join(FILM_DIR, "ffmpeg.log"), "w")

    # The film's own camera, because the demo's is aimed by --view/--cam-* and
    # the film owns its framing outright. 16:9 at the film's real aspect.
    camera.fov = cli_arg("--fov", 46.0, float)
    camera.aspect = W / float(H)
    camera.update_projection_matrix()

    def place_camera(tf):
        eye, tgt = cam_at(tf)
        camera.position.set(*eye)
        camera.look_at(*tgt)

    # PRE-ROLL: settle the wake, the spin-up and the temporal history at the
    # opening helm and the opening framing, then throw it all away.
    place_camera(0.0)
    for _ in range(PREROLL):
        step_frame()
        renderer.render(scene, camera)

    # The frame size is what the FRAMEBUFFER says it is and not what the canvas
    # was asked for -- Canvas::size() is a request. rawvideo -s has to agree
    # with the bytes actually being piped or the encode shears.
    px = renderer.read_pixels()
    FH, FW = px.shape[0], px.shape[1]
    if (FW, FH) != (W, H):
        print(f"       framebuffer is {FW}x{FH}, not the requested {W}x{H}")

    def encoder(path):
        cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{FW}x{FH}",
               "-r", str(FPS), "-i", "-", "-an",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(CRF),
               "-preset", "slow", "-movflags", "+faststart"]
        if FW % 2 or FH % 2:
            cmd += ["-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
        return subprocess.Popen(cmd + [path], stdin=subprocess.PIPE,
                                stdout=log, stderr=log)

    # ── The run ─────────────────────────────────────────────────────────────
    # One pass over the film's frames. A beat that is not being captured still
    # SIMULATES every frame (advance() is what keeps theta and the helm ring
    # honest, and it is nearly free) and renders only the last LEAD of them, so
    # that a --takes re-render of one beat starts with the same temporal
    # history the full run gave it.
    total = int(round(FILM_SECS * FPS))
    caps = [b[0] for b in BEATS] if not ONLY else ONLY
    t_run = time.perf_counter()
    rendered = 0
    f = 0
    for bi, (name, end) in enumerate(BEATS):
        n_end = min(int(round(end * FPS)), total)
        capture = name in caps
        nxt = BEATS[bi + 1][0] if bi + 1 < len(BEATS) else None
        enc = None
        seg = os.path.join(FILM_DIR, f"seg_{bi:02d}_{name}.mp4")
        if capture and FFMPEG is not None and not PROBE:
            enc = encoder(seg)
        while f < n_end:
            tf = f * DT
            warm = (not capture) and nxt in caps and f >= n_end - LEAD
            step_frame()
            if capture or warm:
                place_camera(tf)
                renderer.render(scene, camera)
                rendered += 1
            if capture:
                px = renderer.read_pixels()
                if enc is not None:
                    enc.stdin.write(_np.ascontiguousarray(px).tobytes())
                if f in shot_at:
                    Image.fromarray(px).save(
                        os.path.join(FILM_DIR, f"{shot_at[f]}.png"))
            f += 1
            if PROBE and rendered >= PROBE:
                break
            if f % 120 == 0:
                el = time.perf_counter() - t_run
                print(f"  {name:>9}  t={tf:5.1f}  {f}/{total}  "
                      f"{1e3 * el / max(rendered, 1):.0f} ms/rendered frame",
                      flush=True)
        if enc is not None:
            enc.stdin.close()
            enc.wait()
        if PROBE and rendered >= PROBE:
            break
    wall = time.perf_counter() - t_run

    if PROBE:
        ms = 1e3 * wall / max(rendered, 1)
        print(f"\nprobe: {rendered} frames in {wall:.1f} s = {ms:.0f} ms/frame\n"
              f"       the full {total}-frame film is ~{total * ms / 60e3:.1f} min")
    elif FFMPEG is None:
        # The beat keyframes are still on disk, which is the part that cannot be
        # recovered without another 10 minutes of GPU. State the command that
        # would have finished the job so it can be run by hand.
        print(f"\nno ffmpeg on PATH and no imageio-ffmpeg: {len(KEYSHOTS)} beat "
              f"keyframes in {FILM_DIR}, no mp4.\n"
              "install it (`pip install imageio-ffmpeg`) and re-run --film, or "
              "encode a PNG sequence by hand with:\n"
              f'  ffmpeg -y -framerate {FPS} -i "%05d.png" -c:v libx264 '
              f"-crf {CRF} -preset slow -pix_fmt yuv420p -movflags +faststart "
              f'"{os.path.join(FILM_DIR, "warp_prop_vortex.mp4")}"')
    else:
        # Assembly: the concat demuxer over every segment in beat order, stream
        # copy, so a --takes re-render of one beat drops straight back into the
        # film without touching a pixel of the other five.
        segs = sorted(glob.glob(os.path.join(FILM_DIR, "seg_*.mp4")))
        lst = os.path.join(FILM_DIR, "segments.txt")
        with open(lst, "w") as fh:
            for s in segs:
                fh.write("file '%s'\n" % s.replace("\\", "/"))
        mp4 = os.path.join(FILM_DIR, cli_arg("--out", "warp_prop_vortex.mp4", str))
        subprocess.run([FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", mp4], check=True)
        print(f"\nfilm: {rendered} frames in {wall / 60.0:.1f} min "
              f"({1e3 * wall / max(rendered, 1):.0f} ms/frame)\n"
              f"      {mp4}\n"
              f"      {len(KEYSHOTS)} beat keyframes beside it")
    log.close()
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

    # ── THE OPEN-WATER DIAGRAM, LIVE ────────────────────────────────────────
    # K_T, 10 K_Q and eta_0 against J at the CURRENT P/D, with the operating
    # point on it. This is the picture every propeller is sold with and the one
    # a marine engineer recognises before reading a single label -- and it is
    # not decoration here, because the curve under the moving dot is literally
    # the function the wake is being drawn from.
    #
    # 10 K_Q RATHER THAN K_Q, always, for the reason the convention exists: at
    # this EAR and Z the torque coefficient is a twentieth of the thrust one
    # and would be a flat line on the floor of the axes.
    #
    # THE CURVES ARE CACHED ON P/D. 41 samples x 86 polynomial terms is a
    # millisecond of Python, which is nothing on a slider drag and everything
    # at 25 fps when the pitch has not moved -- and it has not moved on all but
    # a handful of frames.
    OW_N = 41
    _ow_cache = {}

    def open_water_curve(pd):
        key = round(pd, 3)
        hit = _ow_cache.get(key)
        if hit is None:
            if len(_ow_cache) > 64:
                _ow_cache.clear()
            js, kts, kqs, ets = [], [], [], []
            for i in range(OW_N):
                jj = J_HI * i / (OW_N - 1)
                k_t, k_q, _ = open_water(jj, pd)
                js.append(jj)
                kts.append(k_t)
                kqs.append(10.0 * k_q)
                ets.append(jj * k_t / (TWO_PI * k_q)
                           if k_q > 1e-6 and k_t > 0.0 else 0.0)
            hit = (js, kts, kqs, ets)
            _ow_cache[key] = hit
        return hit

    # Explicit, because Dummy(0, h) reserves a ZERO-width item and the rect
    # comes back as a stub -- there is no content-region binding to ask the
    # panel how wide it is, and hard-coding to the window width is honest
    # enough for a panel that sets its own.
    OW_W, OW_H = 424.0, 132.0
    OW_YMAX = 1.0            # K_T, 10 K_Q and eta_0 all live in 0..1 here
    C_KT = (0.62, 0.80, 1.00, 1.0)      # the vapour's own blue
    C_KQ = (1.00, 0.78, 0.42, 1.0)      # the rake's warm
    C_ET = (0.55, 0.95, 0.62, 1.0)      # efficiency, and nothing else is green
    C_AX = (0.55, 0.60, 0.66, 0.85)
    C_GR = (0.35, 0.40, 0.46, 0.45)
    C_OP = (1.00, 1.00, 1.00, 1.0)

    def draw_open_water(st):
        """The diagram, drawn straight onto the panel's own draw list."""
        tp.imgui.dummy(OW_W, OW_H)
        x0, y0, x1, y1 = tp.imgui.item_rect()
        x1 = max(x1, x0 + 120.0)              # a collapsed panel still has axes
        w, h = x1 - x0, y1 - y0

        def px(j, v):
            return (x0 + w * min(max(j, 0.0), J_HI) / J_HI,
                    y1 - h * min(max(v, 0.0), OW_YMAX) / OW_YMAX)

        tp.imgui.draw_rect(x0, y0, x1, y1, (0.06, 0.09, 0.11, 0.85), 1.0, True)
        for gv in (0.2, 0.4, 0.6, 0.8):
            gy = y1 - h * gv / OW_YMAX
            tp.imgui.draw_line(x0, gy, x1, gy, C_GR, 1.0)
        for gj in (0.5, 1.0):
            gx = x0 + w * gj / J_HI
            tp.imgui.draw_line(gx, y0, gx, y1, C_GR, 1.0)
        tp.imgui.draw_rect(x0, y0, x1, y1, C_AX, 1.0, False)

        js, kts, kqs, ets = open_water_curve(st.pd)
        for vals, col in ((kts, C_KT), (kqs, C_KQ), (ets, C_ET)):
            tp.imgui.draw_polyline([px(j, v) for j, v in zip(js, vals)],
                                   col, 1.8)
        # THE OPERATING POINT, and it is drawn at the CLAMPED J and P/D -- the
        # same place the wake is being evaluated -- so a slider pushed outside
        # the validity box parks the dot on the box's edge rather than sliding
        # off into a region the curve does not cover. The panel says why.
        jc = min(max(st.j, J_LO), J_HI)
        opx = x0 + w * jc / J_HI
        tp.imgui.draw_line(opx, y0, opx, y1, (1.0, 1.0, 1.0, 0.35), 1.0)
        for v, col in ((st.kt, C_KT), (10.0 * st.kq, C_KQ), (st.eta, C_ET)):
            cx, cy = px(jc, v)
            tp.imgui.draw_circle(cx, cy, 3.5, col, 1.0, True)
            tp.imgui.draw_circle(cx, cy, 5.0, C_OP, 1.2, False)
        tp.imgui.draw_text(x0 + 5.0, y0 + 3.0,
                           f"B{BLADES}-{EAR * 100:.0f}  P/D {st.pd:.3f}", C_AX)
        tp.imgui.draw_text(x0 + 5.0, y0 + 17.0, "K_T", C_KT)
        tp.imgui.draw_text(x0 + 38.0, y0 + 17.0, "10 K_Q", C_KQ)
        tp.imgui.draw_text(x0 + 96.0, y0 + 17.0, "eta_0", C_ET)
        tp.imgui.draw_text(x1 - 46.0, y1 - 15.0, f"J {J_HI:g}", C_AX)

    def draw_ui():
        """The helm. Three levers, and everything else is a readout of what
        they have already done -- there is nothing here to configure, only the
        pitch, the throttle, the ship's speed and the consequences to watch."""
        global beta_now, rpm_now, vs_now, va_now
        tp.imgui.set_next_window_pos(12, 12)
        tp.imgui.set_next_window_size(452, 0)
        tp.imgui.begin("Helm")
        _, beta_now = tp.imgui.slider_float("blade pitch (deg)", beta_now,
                                            BETA_MIN, BETA_MAX)
        _, rpm_now = tp.imgui.slider_float("shaft speed (rpm)", rpm_now,
                                           RPM_MIN, RPM_MAX)
        # THE SLIDER IS THE SHIP and the model's input is the ADVANCE. There is
        # a hull now, so the two are no longer the same number and the panel
        # has to say which one it is quoting.
        _, vs_now = tp.imgui.slider_float("ship speed (m/s)", vs_now,
                                          VA_MIN, VA_MAX)
        va_now = vs_now * (1.0 - W_MEAN)
        # Read the ACTUAL shaft state back out of the integrator rather than
        # recomputing it from the sliders: during the spin-up the two differ,
        # and the number that means anything is the one the wake was given.
        # state_now IS that object -- advance() built it from omega_now.
        st = state_now
        rpm = omega_now * 60.0 / TWO_PI
        _, uw = st.creep()
        draw_open_water(st)
        tp.imgui.text(f"J {st.j:5.3f}   K_T {st.kt:6.4f}   "
                      f"10 K_Q {10 * st.kq:6.4f}   eta_0 {st.eta:5.3f}")
        # ── R1: SAY WHICH SPEED THE DIAGRAM IS BEING READ AT ────────────────
        # J is the advance ratio and advance is not the ship's speed once there
        # is a hull in front of the disc. Every number above this line belongs
        # to V_a = V_ship (1 - w_mean); the once-per-rev swing below belongs to
        # the peak term, and it is spent on the vapour alone.
        lk = wake_lobe_k(st)
        tp.imgui.text(f"J = V_a / nD,  V_a {st.va:4.2f} = V_ship {vs_now:4.2f} "
                      f"x (1 - w_mean {W_MEAN:4.2f})")
        tp.imgui.text(f"wake shadow  w {W_MEAN - W_PEAK * W_LOBE_MEAN:4.2f} at "
                      f"6 o'clock -> {W_MEAN + W_PEAK * (1.0 - W_LOBE_MEAN):4.2f} "
                      f"at 12   loading "
                      + (f"{100.0 * lk * (1.0 - W_LOBE_MEAN):+.0f}% at TDC: "
                         "the ropes BREATHE" if lk > 0.005 else
                         "steady (no way on: no boundary layer)"))
        tp.imgui.text(f"T {st.thrust / 1e3:7.2f} kN   Q {st.torque / 1e3:6.2f} kNm"
                      f"   P_D {st.power / 1e3:7.1f} kW")
        # ── R5: SAY IT, DO NOT HIDE IT ──────────────────────────────────────
        # The polynomials are a regression over J in [0, 1.5] and P/D in
        # [0.5, 1.4]. Both sliders leave that box at their ends, and a panel
        # that kept quoting four decimals there would be lying with precision.
        if st.extrapolated:
            tp.imgui.text("*** outside B-series validity -- EXTRAPOLATED ***")
            tp.imgui.text(f"    P/D {st.pd:5.3f} (box {PD_LO:g}-{PD_HI:g})   "
                          f"J {st.j:5.3f} (box {J_LO:g}-{J_HI:g})")
        else:
            tp.imgui.text(f"beta {beta_now:5.1f} deg at 0.7R -> P/D {st.pd:5.3f}"
                          f"   (design {BETA_DESIGN:.0f} deg / {PD_DESIGN:.3f})")
        tp.imgui.text(f"shaft  {rpm:5.0f} rpm   tip {omega_now * R_TIP:5.1f} m/s"
                      f"   v_i {st.vi:5.2f} m/s")
        ud, _ = st.creep()
        tp.imgui.text(f"wake   {uw:5.2f} m/s   advance "
                      f"{(ud / (rpm / 60.0) if rpm > 1.0 else 0.0):4.2f}-"
                      f"{(uw / (rpm / 60.0) if rpm > 1.0 else 0.0):4.2f} m/rev"
                      f"   reach {wake_reach(ud, uw, LIFETIME):4.1f} m")
        tp.imgui.separator()
        # ── THE CAVITATION INDICATOR, NOW BURRILL'S ─────────────────────────
        # The one readout this demo exists for, and it finally quotes the pair
        # a naval architect would ask for: the cavitation number at 0.7 R and
        # the thrust-loading coefficient, against Burrill's own 5% line. The
        # bar is how far the loading has come toward INCEPTION, which sits at
        # BURRILL_INC of that line. imgui here has no coloured text and no
        # progress bar, so the bar is ASCII -- which costs nothing and is
        # legible in a 1:1 crop, which is where it actually gets read.
        sg, tc, l5, cr = burrill(st)
        fill = int(round(min(cr, 1.0) * 24))
        tp.imgui.text(f"cav    [{'#' * fill}{'-' * (24 - fill)}] "
                      f"{cr:4.2f} x inception")
        tp.imgui.text(f"       sigma_0.7R {sg:6.2f}   tau_c {tc:5.3f}   "
                      f"Burrill 5% {l5:5.3f}")
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
        # ── R4: THE SHEET'S OWN LINE ────────────────────────────────────────
        # State, extent and -- when the hull's shadow is driving anything --
        # the once-per-rev pulse, as the two ends of the same expression rather
        # than as a percentage somebody wrote down. At the bollard the two ends
        # coincide and the line says so.
        lo6, mid, hi12 = sheet_pulse(st)
        sfill = int(round(min(cr / SHEET_CAV_INC, 1.0) * 24))
        tp.imgui.text(f"sheet  [{'#' * sfill}{'-' * (24 - sfill)}] "
                      f"{cr / SHEET_CAV_INC:4.2f} x back-cav inception")
        if mid > 0.0:
            tp.imgui.text(f"       SHEET ON THE BACKS   extent {mid:4.2f} "
                          f"of the blade")
            tp.imgui.text(f"       once per rev: {lo6:4.2f} at 6 o'clock -> "
                          f"{hi12:4.2f} at 12" if hi12 - lo6 > 1e-3 else
                          "       steady (no way on: no boundary layer)")
        else:
            tp.imgui.text(f"       no back cavitation   "
                          f"{100.0 * (1.0 - cr / SHEET_CAV_INC):.0f}% short "
                          f"({SHEET_CAV_INC:.2f} x the tips')")
        tp.imgui.text(f"       inception here: hub "
                      f"{inception_rpm(beta_now, va_now, HUB_CAV):.0f}, tips "
                      f"{inception_rpm(beta_now, va_now, 1.0):.0f}, sheet "
                      f"{inception_rpm(beta_now, va_now, 1.0 / SHEET_CAV_INC):.0f}"
                      f" rpm")
        tp.imgui.text(f"{tp.imgui.get_framerate():5.0f} fps   "
                      f"{N / 1e6:.1f} M parcels   t={sim_time:6.2f} s")
        tp.imgui.separator()
        # The one thing to say about the lag, because it is the feature and it
        # looks like a bug for the seconds it takes to convect away.
        tp.imgui.text("K_T, K_Q and eta_0 are the Wageningen B5-75 regression")
        tp.imgui.text("(Oosterveld & van Oossanen 1975), and the wake is what")
        tp.imgui.text("the thrust they give does to the water. NOTHING below")
        tp.imgui.text("the diagram is tuned separately from it.")
        tp.imgui.text("water carries no tracer. what you see is VAPOUR, and")
        tp.imgui.text("it only exists above the bar. push any lever.")
        tp.imgui.text(f"{LIFETIME:.1f} s of old slipstream keeps the pitch, the")
        tp.imgui.text(f"speed AND the cavitation it was shed under ({HIST_N}-frame")
        tp.imgui.text("ring), and the boundary convects away. the inflow does")
        tp.imgui.text("not lag: a pressure field is not convected.")
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
