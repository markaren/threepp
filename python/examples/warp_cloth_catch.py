"""Four arms catch a cannonball in a cloth, tracking it with two cameras on tripods.

Aim the cannon, fire, and a rig of four anchors holding a sheet works out where
the ball will be and gets there in time. The rig is never told the ball's true
position: it sees only two 320x240 pinhole frame cameras standing in the scene,
and everything else is inferred from what changed between their frames.

Three things make this work, and each of them is a measurement, not a guess:

  * SYNCHRONIZED STEREO, not one camera guessing range. Both sensors are
    secondary views on the same renderer, so one render() produces both of them
    from ONE scene build at the SAME simulated instant -- which two render()
    calls can never give, and which is the whole reason the two bearings may be
    intersected at all. Placed 87 degrees apart as seen from the catch zone,
    they turn two bearings into a 3-D point by least-squares midpoint, and the
    residual gap between the two rays is a free quality signal: the fit throws
    away any instant whose rays miss each other by more than 12 cm.

    That kills the old range-from-apparent-size crutch, which carried a
    systematic -0.43 m bias (the RMS radius of a blob is only PROPORTIONAL to
    the ball's apparent radius, and the constant depends on how the ball happens
    to look). |g| is still fixed at 9.81 in the arc fit, but it is now a
    consistency constraint on six well-observed unknowns rather than the only
    thing pinning the scale.

  * FRAME DIFFERENCING, per camera, on pixels only. Grayscale, |I_t - I_t-1|,
    threshold, then seed on the DENSEST 4-pixel cell rather than the centroid:
    the sheet and the arms are a far bigger source of changed pixels than the
    ball, but they change as thin outlines while the ball changes as a solid
    disc, so pixels-per-cell separates them cleanly. Once the arc is fitted the
    search window is re-anchored on the PREDICTED pixel, not on the last blob:
    a window that follows its own measurement walks onto the cloth and never
    comes back, and that alone was worth 0.96 m on the az -12 shot. Measured:
    1.5 and 1.9 px median bearing error, 90th percentiles under 3.5 px, and
    3-D observations 0.039 m from truth against the old sensor's 0.088 m.

    The last 80 ms are dropped on purpose. There the ball and the sheet occupy
    the same pixels in both cameras and no difference detector can say which of
    two overlapping movers it is centred on; by then the plan is 150 frames old
    and the rig is committed, so nothing is lost.

  * The rig arrives MATCHING THE BALL'S HORIZONTAL VELOCITY, and TILTED. The
    bowl in a slack sheet is only ~0.3 m deep, worth about 3 J/kg, while a ball
    crossing it at 3 m/s carries 4.5 J/kg -- park the rig at the landing spot
    and the ball skips straight out the far side. So the approach is a cubic
    Hermite (position AND velocity at arrival), re-solved every frame from the
    rig's current state as the track sharpens, and over the last 0.45 s of it
    the square pitches 8 degrees with its DOWNRANGE corners high, about an axis
    perpendicular to the fitted horizontal velocity. The scoop is a small
    effect and it is reported as one: on the default shot it takes the ball's
    first contact from 0.063 m off the sheet centre to 0.052.

  * AND THEN IT THROWS THE BALL BACK, into a basket 1.35 m away. The stroke is
    planned without looking at the ball at all: the launch point is the rig's
    OWN centre, which it knows from its four tool frames, and v0 comes out of
    (B - p - g T^2 / 2) / T solved for the cheapest T the arms can deliver.
    Then the sheet is pulled taut, dipped 15 cm along -v0, driven along +v0 to
    the commanded speed and braked. Nothing touches the ball: it rides the same
    contact and friction that caught it, and leaves when the impulse stops.

    That plan is only true if the ball is AT the rig's centre, so the rig puts
    it there first. The catch ends wherever the ball was -- 0.98 m from home on
    the default shot, 0.68 m at az +8 -- and a stroke launched from there threw
    something that was not on it: the az +8 ball left at 0.17 of the commanded
    speed, 145 degrees off plan, and landed 1.93 m from the bin. So after the
    absorb the rig CARRIES THE BALL HOME, walking back at 1.6 m/s while the
    sheet re-tensions under it. The bowl's low point is the rig centre, so the
    ball rolls to the middle by construction rather than by being looked at:
    0.98 m of catch offset becomes 0.12 m of residual, and the throw leaves
    within 8 degrees of plan on every shot the arms can reach.

    A trampoline is not a hand, and the measured transfer says so. The ball is
    NOT free at the top of the stroke -- scoring it there read 2.60 m/s while
    the ball was actually still in the cloth and left at 8.7 m/s once the
    long-range attachment snapped taut -- so the release is measured at
    separation, after 80 ms of daylight. Two numbers are calibrated on that
    measurement and nothing else: the transfer ratio (1.02-1.11 x the anchors'
    own speed, so the stroke is commanded at speed/0.85) and the launch POINT,
    which is where the ball sat and not where the stroke ends, because the sheet
    slides under the ball for most of the stroke. Assuming the ball rides
    forward with the rig put the planned launch point 0.285 m from the truth and
    that is most of a basket at 1.35 m. The verdict is truth, and only the
    verdict: DELIVERED if the ball comes to rest in the bin, and the miss
    distance either way.

  * THE ARMS ARE THE ENVELOPE, and they are now allowed to say so. --arm-reach
    was a flag with 2.0 m typed into it while four Franka arms on 0.5 m
    pedestals ran out at about 1.1 m downrange and 0.16 m sideways, so every
    off-axis catch was executed with one corner up to a metre behind where it
    was commanded -- the sheet stopped being a sheet, and the ball left at
    whatever the wreckage gave it. The tool reach is now MEASURED at startup by
    walking a target away from one arm's base until it stops converging (1.08 m
    for an FR3 to fr3_hand_tcp), the travel envelope is solved from it per
    direction, and the rig is clamped to that envelope instead of to a flag.
    Standing the pedestals at 0.70 m and tucking them 0.34 m out took the
    default shot's worst tracking error from 0.186 m to 0.019 m. Shots outside
    the envelope are named at startup and reported as reach-capped rather than
    attempted: az -12 el 58 at 7.0 m/s crosses 2.06 m from home against a 1.11 m
    boundary. That shot was thought to be a regression; it is not. Re-running
    the pre-sensor build on it misses by the same margin with the same 1.01 m of
    arm error, so it has never been caught, and it now says so on the first line
    of the run instead of on the last.

The range around all this is procedural and static: a painted concrete pad with
a hazard ring on the catch zone and a lane out of the cannon, a backstop, crates
and drums, four lamp poles and a pennant line. The markings live on their OWN
mesh and their own texture -- 10.8 x 8.05 m at 5.3 mm per texel against the
ground's 26 mm -- because a 26 mm step is invisible in gravel noise and a
staircase in the edge of a painted line, and because every edge on the pad is a
smoothstep over a signed distance rather than a boolean mask. Resolution alone
just makes smaller stairs. The bin's placement is arithmetic too: it stands
uprange, which is the only corridor that is 1.2-1.5 m from home, 0.35 m clear of
every pedestal and outside every arm's swept envelope, and clear of everywhere
the sheet goes -- which is all of +x. check_props() prints those four clearances
and the angle the film camera looks into the bin at, because the first bin was
standing inside an arm and rendered perfectly well doing it.

Firing is a flash, a puff of 60 smoke particles and 6 cm of recoil, and all
three are things the SENSORS SEE.
The flash was the expensive one: a point light throws a pool of changing
brightness metres wide across the floor, the detector locked onto it, and the
first crossing call came out 0.99 m wrong. Two things fix that, and neither is
a special case for the ball. The flash is over in five sensor ticks, before the
tracker's first accepted observation. And a disc around the cannon's own
projected position is excluded from the difference for the first 0.35 s: the rig
knows where its own gun is, the disc never moves and never follows the target,
and at 0.75 m the ball is out of it 15 ms after leaving the barrel. Lighting the
range for dusk also cost the detector two thirds of its observations at the old
threshold of 16 luma, so the threshold is 9 now: same bearing medians, 248
observations instead of 59.

The sensors are bolted to tripods you can see, and the view camera is now free:
orbit it during flight, fire from anywhere, it changes nothing the tracker sees.
Secondary views also run no overlay pass, so the aim arc, the reticles and ImGui
are structurally incapable of leaking into a sensor frame.

In the window the cannon carries its aim visibly and draws the arc the shot will
actually take, with a ring where it crosses the catch plane. That preview is
ground truth and is the PLAYER's aid -- it is hidden the instant the shot leaves,
so it can never be confused with, or leak into, what the tracker has to work out
for itself. Watching the tracker's cyan reticle converge onto that amber ring is
the whole perception story in one picture.

    python warp_cloth_catch.py                      # window; A/D aim, W/S elevate,
                                                    # Q/E power, SPACE fire, R reset
    python warp_cloth_catch.py --frames 900 --autofire   # bounded window run, for testing
    python warp_cloth_catch.py --tune               # headless, numbers only
    python warp_cloth_catch.py --clip catch.mp4     # headless mp4
    python warp_cloth_catch.py --clip c.mp4 --split # with both sensor feeds inset
    python warp_cloth_catch.py --az -12 --el 58 --speed 7.0
    python warp_cloth_catch.py --oracle             # ground truth instead of the sensors,
                                                    # to separate tracking error from control
    python warp_cloth_catch.py --tilt 0             # flat sheet, to A/B the scoop
    python warp_cloth_catch.py --no-toss            # catch only, no throw
    python warp_cloth_catch.py --no-effects         # no flash, smoke or recoil, so the
                                                    # sensor pollution can be A/B'd

Prints the predicted intercept against where the ball actually crossed, and the
throw's commanded speed against what the ball actually left with, so perception,
catch and throw can each be judged separately.
"""
import math
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import cli_arg, find_ffmpeg, parse_size, standard_material

# ---- flags -------------------------------------------------------------------
TUNE = "--tune" in sys.argv
CLIP = cli_arg("--clip", "", str)
SEQ = cli_arg("--seq", "", str)
SPLIT = "--split" in sys.argv          # show both sensor feeds inset in the frame
# Film pass: full render scale, a bigger frame, a proper key/fill/rim set, the
# dead air before the shot trimmed, and the catch itself in slow motion. The
# sheet still has to settle for the tracker to bootstrap -- it just does not
# have to be watched doing it.
FILM = "--film" in sys.argv
ORACLE = "--oracle" in sys.argv        # bypass the sensors; use true ball state
TRACE = "--trace" in sys.argv
# Bounded interactive run: drives the real window loop -- ImGui, HUD, orbit
# controls and all -- for N frames and exits, so the interactive path can be
# exercised rather than assumed. --autofire shoots as soon as it reads READY.
FRAMES = int(cli_arg("--frames", 0, float))
AUTOFIRE = "--autofire" in sys.argv
HEADLESS = bool(TUNE or CLIP or SEQ) and not FRAMES

# ---- the cannon --------------------------------------------------------------
CANNON = np.array([float(x) for x in cli_arg("--cannon", "-2.40,0.55,0", str).split(",")])
AZ = cli_arg("--az", 0.0, float)             # degrees; 0 fires along +x
EL = cli_arg("--el", 62.0, float)            # degrees above horizontal
MUZZLE = cli_arg("--speed", 6.5, float)      # m/s
aim = {"az": AZ, "el": EL, "speed": MUZZLE}
# The sheet must be STILL before the ball flies. The detector reports change, so
# a ringing sheet fills both frames with changed pixels and the tracker locks
# onto cloth instead of ball -- measured centre 300 px from the true one. Once it
# has settled the ball is the only mover and the same tracker is sub-pixel.
FIRE_AT = cli_arg("--fire-at", 1.2, float)   # headless: seconds of settle first

# ---- the rig -----------------------------------------------------------------
HOME = np.array([float(x) for x in cli_arg("--home", "0.50,0", str).split(",")])  # x, z
ARM_SPEED = cli_arg("--arm-speed", 4.5, float)   # m/s cap on the rig centre
ARM_REACH = cli_arg("--arm-reach", 2.0, float)   # m from home the rig may travel
ABSORB = cli_arg("--absorb", 0.35, float)        # s to bleed the ball's motion off
GIVE = cli_arg("--give", 0.30, float)            # m the sheet may drop while absorbing
RECOVER = cli_arg("--recover", 0.55, float)      # s to lift back to the ready height
PRESENT = cli_arg("--present", 0.75, float)      # how taut to pull on recovery, in
                                                 # units of the slack (1 = dead flat)
# The catch ends wherever the ball was, which on an off-axis shot is up to a
# metre from home -- and a throw planned from the rig's own centre is then
# throwing something that is NOT at that centre. So the rig walks the ball home
# while it re-tensions, slowly enough to keep it in the bowl, and only plans the
# stroke once it is back at the pose it knows. Peak carry speed, not duration:
# the carry stretches to whatever the distance needs at this speed.
CARRY_SPEED = cli_arg("--carry-speed", 1.60, float)   # m/s while cradling
SETTLE_DWELL = cli_arg("--settle-dwell", 0.35, float) # s of stillness before the throw
# The sheet does not have to stay flat. Tilting the square so the DOWNRANGE side
# rides high turns it into a scoop: the ball's horizontal energy runs uphill into
# the pocket instead of skating out over the far edge, which is the exact way the
# marginal catches were being lost (ball at rest 0.766 m out against a 0.75 m
# half-width). The axis is horizontal and perpendicular to the incoming
# horizontal velocity AS THE TRACKER FITTED IT -- no truth is read.
TILT_DEG = cli_arg("--tilt", 8.0, float)        # degrees of scoop at arrival
TILT_LEAD = cli_arg("--tilt-lead", 0.45, float)  # s before arrival the tilt ramps in
TILT_EASE = cli_arg("--tilt-ease", 0.30, float)  # s to flatten again during recovery

# ---- the throw ----------------------------------------------------------------
# The basket stands UPRANGE of the rig, back toward the cannon, and every part
# of that is forced rather than chosen. Three things want the same floor:
#
#   * the catch. Every shot in the aim envelope crosses DOWNRANGE of home, from
#     x +0.56 out to x +2.30, and the sheet sweeps 0.75 m further still -- so a
#     bin anywhere on the +x side is something the cloth flies through. The
#     first placement, x +0.85 z -1.35, was inside the swept envelope of the
#     rear-left arm as well (0.56 m from its pedestal axis against a 0.42 m
#     bin), which is what check_props() below now refuses.
#   * the four pedestals, which occupy a 1.30 x 2.30 m rectangle around home.
#     Only two corridors leave 1.2-1.5 m of throw AND 0.35 m of clearance: past
#     +x, which the catch owns, and back past -x, which is empty because the
#     ball is 2.5 m up when it passes over.
#   * the film camera, which stands off the +x +z shoulder. Upranging the bin
#     puts the throw broadside to it and out from behind the sheet.
#
# The bin is also WIDER AND SHALLOWER than it was (0.42 x 0.28 against 0.40 x
# 0.42): a camera 4.6 m away and 3.3 m up looks into it at 29 degrees, and a bin
# needs the sight line to clear its own near rim -- 23 degrees at this aspect,
# 39 at the old one, which is why the ball in the bin used to be a rumour.
BASKET = np.array([float(x) for x in cli_arg("--basket", "-0.85,0.0", str).split(",")])
BASKET_R = cli_arg("--basket-r", 0.42, float)     # inner radius of the bin
BASKET_RIM = cli_arg("--basket-rim", 0.90, float)
BASKET_DEPTH = cli_arg("--basket-depth", 0.28, float)
BASKET_FLOOR = BASKET_RIM - BASKET_DEPTH
TOSS = "--no-toss" not in sys.argv
# The stroke is bounded by the same arm that does the catching: ARM_SPEED is a
# hard cap on the rig centre, so anything the solve asks for above this is a
# throw the rig cannot make and is reported as such rather than faked.
TOSS_VMAX = cli_arg("--toss-vmax", 0.93 * ARM_SPEED, float)
TOSS_DIP = cli_arg("--toss-dip", 0.15, float)     # m of windup along -v0
TOSS_WINDUP = cli_arg("--toss-windup", 0.26, float)
TOSS_STROKE = cli_arg("--toss-stroke", 0.17, float)   # s of acceleration
TOSS_BRAKE = cli_arg("--toss-brake", 0.10, float)     # s to stop after release
TOSS_SPREAD = cli_arg("--toss-spread", 0.92, float)   # taut, but not dead flat:
                                                      # a flat sheet has no bowl
                                                      # to hold the ball during
                                                      # the windup
# What the ball leaves with, as a multiple of the rig's own commanded speed --
# a calibration of the rig's actuator, measured open loop over the stroke rather
# than read off the ball in flight. It is not a constant, because a trampoline
# is not a hand: measured 1.05x at a 2.9 m/s stroke, 1.14x at 3.7 m/s and 1.54x
# at 4.15 m/s, since the long-range attachment snaps taut at the top of a hard
# stroke and slings the ball out faster than the anchors ever move. 0.95 is the
# working point that puts the default shot in the bin; the first throw the rig
# ever made assumed 0.80, was commanded at 4.15 and went 6.6 m past the basket.
TOSS_GAIN = cli_arg("--toss-gain", 0.85, float)
# The flattest throw the planner may choose. Left free, the solve takes the
# cheapest T it can find and asks for 2.70 m/s, which is below the speed at
# which the sheet slings at all: measured, the ball then left with 0.80 of the
# rig's own speed and fell 0.72 m short. Above 0.60 s the plan lands in the
# range the gain above was calibrated over, and it also looks like a throw
# rather than a shove.
TOSS_T_MIN = cli_arg("--toss-t-min", 0.60, float)
# How far the ball rises off its rest point before it is free of the sheet,
# measured at separation rather than assumed from the stroke.
TOSS_RISE = cli_arg("--toss-rise", 0.03, float)

# ---- the sheet ---------------------------------------------------------------
N = int(cli_arg("--res", 48, float))
CLOTH = cli_arg("--cloth", 1.50, float)
SPAN = cli_arg("--span", 1.30, float)
CATCH_Y = cli_arg("--catch-y", 1.10, float)
CLOTH_KG = cli_arg("--cloth-kg", 0.45, float)

BALL_R = cli_arg("--ball-r", 0.10, float)
BALL_KG = cli_arg("--ball-kg", 0.40, float)
MU = cli_arg("--mu", 0.55, float)            # cloth-on-ball Coulomb friction

# ---- rates -------------------------------------------------------------------
# The sensor rate IS the loop rate: the detector samples once per render, so a
# fast tracker means a fast loop. The substep rate is held constant across it,
# because that -- not the frame rate -- is what the contact needs.
SENSOR_HZ = cli_arg("--sensor-hz", 240.0, float)
SUBSTEP_HZ = cli_arg("--substep-hz", 2880.0, float)
ITERS = int(cli_arg("--iters", 32, float))
SUBSTEPS = max(1, int(round(SUBSTEP_HZ / SENSOR_HZ)))
DT = 1.0 / (SENSOR_HZ * SUBSTEPS)
CLIP_FPS = 60.0
# Long enough for the whole beat: settle, flight, absorb, recover, the window
# that decides CAUGHT against MISSED, and then the throw -- windup, stroke,
# ~0.6 s of second flight and the ball coming to rest in the basket. At 3.0 s
# the run stopped mid-lift and reported "still lifting", which is not a verdict;
# the extra seconds cost 0.06 ms/tick of the average, measured.
# The carry home added roughly a second: an off-axis catch ends up to a metre
# out and walks back at a cradling speed.
SECONDS = cli_arg("--seconds", 4.2 if "--no-toss" in sys.argv else 8.0, float)
DAMPING = cli_arg("--damping", 0.010, float)

# Each sensor is a secondary view rendered at exactly this size, so unlike the
# old DVS there is no resampling between render and detector: one focal length
# describes both axes and the pixels are square by construction. 320x240 is a
# deliberate choice, not a leftover -- see SENSORS below for the measured trade
# against 480x360, and note that picture-in-picture is 1:1 only, so the sensor
# size IS the inset size in the window.
SENSOR_W, SENSOR_H = parse_size(cli_arg("--sensor", "320x240", str))
SENSOR_FOV = cli_arg("--sensor-fov", 45.0, float)
# 4:3, to match the sensors. It is also the better aspect for a social clip.
VIEW_W, VIEW_H = parse_size(cli_arg("--size", "1280x960" if FILM else "960x720", str))
# Frame differencing threshold, in 0-255 luma. The ball moves ~0.5 px per tick
# at 240 Hz, so what clears this is not the disc but the disc's TEXTURE sliding
# across the pixels under it -- which means the threshold is really a statement
# about local contrast, and local contrast is set by the lighting.
#
# It used to be 16, against a near-black floor and a single hard sun. Lighting
# the range for dusk (a warm sun, a cool fill, four lamp poles) lifted the mid
# tones and cost the detector two thirds of its observations at 16: 59 tracked
# frames against 146. At 9 the same scene gives 248, with the bearing medians
# and the stereo gap unchanged or better, so this is a threshold that was tuned
# for a different scene rather than a floor imposed by noise.
DIFF_THRESHOLD = cli_arg("--diff-threshold", 9.0, float)
# Render is ~65% of the loop, so this is the lever that matters. The worry was
# that it would cost accuracy -- the detector samples the post-TAA frame, so a
# lower scale softens the edges it fires on -- but measured across 0.5..1.0 the
# bearing stays at 2.4-2.6 px and every scale still catches. Interleaved A/B
# (the run-to-run spread is ~15%, so singles are not worth quoting): 25.8 ms at
# 1.0 against 18.7 ms at 0.6.
RENDER_SCALE = cli_arg("--render-scale", 1.0 if FILM else 0.8, float)
PROFILE = "--profile" in sys.argv
# The GI machinery is built for many-light and emissive-geometry scenes. This
# one has two lights and no emitters, so it was paying for convergence it does
# not need: measured, probe GI + deferred AO + ReSTIR DI cost 1.6 ms/tick of an
# 11.2 ms render here and change the look of a matte floor and four white arms
# very little. So they are OFF by default and opt-back-in per feature; the film
# pass keeps them, because there the look is the point.
GI = "--gi" in sys.argv or FILM
AO = "--ao" in sys.argv or FILM
RESTIR = "--restir" in sys.argv or FILM
NO_DENOISE = "--no-denoise" in sys.argv
prof = {}
MIN_PIXELS = int(cli_arg("--min-pixels", 24, float))

V = (N + 1) * (N + 1)
REST = CLOTH / N
M_PARTICLE = CLOTH_KG / V
GRAVITY = wp.vec3(0.0, -9.81, 0.0)
G_NP = np.array([0.0, -9.81, 0.0])

# ---- warp: the sheet ----------------------------------------------------------


@wp.func
def grid_index(ix: int, iy: int, nx: int) -> int:
    return iy * (nx + 1) + ix


@wp.func
def spring(p: wp.vec3, pos: wp.array(dtype=wp.vec3), im: wp.array(dtype=float),
           w_self: float, ix: int, iy: int, nx: int, ny: int,
           rest: float, stiffness: float) -> wp.vec3:
    if ix < 0 or ix > nx or iy < 0 or iy > ny:
        return wp.vec3(0.0, 0.0, 0.0)
    j = grid_index(ix, iy, nx)
    w_nb = im[j]
    denom = w_self + w_nb
    if denom < 1.0e-12:
        return wp.vec3(0.0, 0.0, 0.0)
    d = pos[j] - p
    l = wp.length(d)
    if l < 1.0e-9:
        return wp.vec3(0.0, 0.0, 0.0)
    return d * (stiffness * (w_self / denom) * (l - rest) / l)


@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec3), prev: wp.array(dtype=wp.vec3),
              pred: wp.array(dtype=wp.vec3), im: wp.array(dtype=float),
              dt: float, damping: float):
    i = wp.tid()
    p = pos[i]
    prev_p = prev[i]
    prev[i] = p
    if im[i] == 0.0:
        pred[i] = p
        return
    vel = (p - prev_p) * (1.0 - damping)
    pred[i] = p + vel + GRAVITY * dt * dt


@wp.kernel
def set_anchors(pred: wp.array(dtype=wp.vec3), idx: wp.array(dtype=int),
                target: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    pred[idx[k]] = target[k]


@wp.kernel
def solve(p_in: wp.array(dtype=wp.vec3), p_out: wp.array(dtype=wp.vec3),
          im: wp.array(dtype=float), nx: int, ny: int, rest: float,
          ball: wp.array(dtype=wp.vec3), ball_prev: wp.array(dtype=wp.vec3),
          ball_r: float, mu: float,
          prev_pos: wp.array(dtype=wp.vec3),
          anchors: wp.array(dtype=wp.vec3), lra: wp.array(dtype=float),
          impulse: wp.array(dtype=wp.vec3), m_particle: float, relax: float):
    i = wp.tid()
    p = p_in[i]
    w = im[i]
    if w == 0.0:
        p_out[i] = p
        return
    ix = i % (nx + 1)
    iy = i // (nx + 1)
    rd = rest * wp.sqrt(2.0)
    c = wp.vec3(0.0, 0.0, 0.0)
    c += spring(p, p_in, im, w, ix - 1, iy, nx, ny, rest, 1.0)
    c += spring(p, p_in, im, w, ix + 1, iy, nx, ny, rest, 1.0)
    c += spring(p, p_in, im, w, ix, iy - 1, nx, ny, rest, 1.0)
    c += spring(p, p_in, im, w, ix, iy + 1, nx, ny, rest, 1.0)
    c += spring(p, p_in, im, w, ix - 1, iy - 1, nx, ny, rd, 0.85)
    c += spring(p, p_in, im, w, ix + 1, iy - 1, nx, ny, rd, 0.85)
    c += spring(p, p_in, im, w, ix - 1, iy + 1, nx, ny, rd, 0.85)
    c += spring(p, p_in, im, w, ix + 1, iy + 1, nx, ny, rd, 0.85)
    c += spring(p, p_in, im, w, ix - 2, iy, nx, ny, 2.0 * rest, 0.35)
    c += spring(p, p_in, im, w, ix + 2, iy, nx, ny, 2.0 * rest, 0.35)
    c += spring(p, p_in, im, w, ix, iy - 2, nx, ny, 2.0 * rest, 0.35)
    c += spring(p, p_in, im, w, ix, iy + 2, nx, ny, 2.0 * rest, 0.35)
    p = p + c * relax

    # Long-range attachment: no particle may be further from an anchor than the
    # sheet's own material distance to it. Jacobi propagates tension one edge per
    # iteration, far too slow to stop a 49-wide grid stretching like a trampoline
    # under an impact; this says it in one step. Unilateral, so it adds no energy.
    for k in range(4):
        ak = anchors[k]
        dk = p - ak
        lk = wp.length(dk)
        lmax = lra[i * 4 + k]
        if lk > lmax:
            p = ak + dk * (lmax / lk)

    # Ball contact. The pushout delta is the only thing the ball did to this
    # particle, so m * delta summed over the sheet IS the momentum transfer --
    # the spring corrections above cancel pairwise and cannot leak into it.
    bc = ball[0]
    d = p - bc
    l = wp.length(d)
    if l < ball_r:
        nrm_c = d / wp.max(l, 1.0e-6)
        push = nrm_c * ball_r - d
        p = p + push

        # Coulomb friction, position-based. Without it the pushout is purely
        # radial, so nothing can slow the ball ALONG the sheet -- it skates
        # across and out the far side no matter where the rig is. The sheet
        # holds a ball with friction, not with geometry.
        # Tangential slip is this particle's motion relative to the ball's over
        # the substep, projected out of the normal, and it is limited by the
        # penetration depth exactly as Macklin's PBD friction is.
        slip = (p - prev_pos[i]) - (bc - ball_prev[0])
        slip = slip - nrm_c * wp.dot(slip, nrm_c)
        lt = wp.length(slip)
        fric = wp.vec3(0.0, 0.0, 0.0)
        if lt > 1.0e-9:
            fric = slip * (-wp.min(1.0, mu * (ball_r - l) / lt))
            p = p + fric
        wp.atomic_add(impulse, 0, (push + fric) * m_particle)

    p_out[i] = wp.vec3(p[0], wp.max(p[1], 0.005), p[2])


@wp.kernel
def ball_predict(bp: wp.array(dtype=wp.vec3), bv: wp.array(dtype=wp.vec3),
                 bpred: wp.array(dtype=wp.vec3), h: float):
    bpred[0] = bp[0] + bv[0] * h + GRAVITY * h * h


@wp.kernel
def ball_finish(bp: wp.array(dtype=wp.vec3), bv: wp.array(dtype=wp.vec3),
                impulse: wp.array(dtype=wp.vec3), h: float, inv_bm: float,
                radius: float,
                bk_x: float, bk_z: float, bk_floor: float, bk_rim: float,
                bk_r: float):
    v = bv[0] + GRAVITY * h - impulse[0] * (inv_bm / h)
    p = bp[0] + v * h
    if p[1] < radius:
        p = wp.vec3(p[0], radius, p[2])
        v = wp.vec3(v[0] * 0.6, wp.abs(v[1]) * 0.3, v[2] * 0.6)

    # The basket is a real collider, not a scoring region: an open cylinder
    # (wall, floor, rim ring) the ball can enter, rattle in, or bounce off the
    # lip of. A throw that clips the rim has to be allowed to fail visibly,
    # which a "was it inside the radius" test would quietly hide.
    dx = p[0] - bk_x
    dz = p[2] - bk_z
    d = wp.sqrt(dx * dx + dz * dz)
    ux = float(1.0)
    uz = float(0.0)
    if d > 1.0e-6:
        ux = dx / d
        uz = dz / d
    if p[1] < bk_rim:
        if d < bk_r:                              # inside: wall and floor
            if d > bk_r - radius:
                p = wp.vec3(bk_x + ux * (bk_r - radius), p[1], bk_z + uz * (bk_r - radius))
                vr = v[0] * ux + v[2] * uz
                if vr > 0.0:
                    v = wp.vec3(v[0] - 1.4 * vr * ux, v[1], v[2] - 1.4 * vr * uz)
            if p[1] < bk_floor + radius:
                p = wp.vec3(p[0], bk_floor + radius, p[2])
                v = wp.vec3(v[0] * 0.55, wp.abs(v[1]) * 0.22, v[2] * 0.55)
        elif d < bk_r + radius:                   # outside the wall
            p = wp.vec3(bk_x + ux * (bk_r + radius), p[1], bk_z + uz * (bk_r + radius))
            vr = v[0] * ux + v[2] * uz
            if vr < 0.0:
                v = wp.vec3(v[0] - 1.4 * vr * ux, v[1], v[2] - 1.4 * vr * uz)
    else:
        # The lip itself, as a circle: this is what makes a rim-out look like
        # a rim-out instead of a ball passing through a hole.
        cx = bk_x + ux * bk_r
        cz = bk_z + uz * bk_r
        ex = p[0] - cx
        ey = p[1] - bk_rim
        ez = p[2] - cz
        el = wp.sqrt(ex * ex + ey * ey + ez * ez)
        if el < radius and el > 1.0e-6:
            nx = ex / el
            ny = ey / el
            nz = ez / el
            p = wp.vec3(cx + nx * radius, bk_rim + ny * radius, cz + nz * radius)
            vn = v[0] * nx + v[1] * ny + v[2] * nz
            if vn < 0.0:
                v = wp.vec3(v[0] - 1.5 * vn * nx, v[1] - 1.5 * vn * ny, v[2] - 1.5 * vn * nz)
    bv[0] = v
    bp[0] = p


@wp.kernel
def compute_normals(pos: wp.array(dtype=wp.vec3), nrm: wp.array(dtype=wp.vec3),
                    nx: int, ny: int):
    i = wp.tid()
    ix = i % (nx + 1)
    iy = i // (nx + 1)
    xm = pos[grid_index(wp.max(ix - 1, 0), iy, nx)]
    xp = pos[grid_index(wp.min(ix + 1, nx), iy, nx)]
    zm = pos[grid_index(ix, wp.max(iy - 1, 0), nx)]
    zp = pos[grid_index(ix, wp.min(iy + 1, ny), nx)]
    n = wp.cross(zp - zm, xp - xm)
    nrm[i] = n / wp.max(wp.length(n), 1.0e-9)


# ---- sheet state --------------------------------------------------------------

wp.init()
device = wp.get_preferred_device()

xs = np.linspace(-SPAN / 2, SPAN / 2, N + 1, dtype=np.float32)
gx, gz = np.meshgrid(xs, xs)
r2 = (gx / (SPAN / 2)) ** 2 + (gz / (SPAN / 2)) ** 2
gy = CATCH_Y - 0.5 * (CLOTH - SPAN) * np.clip(1.0 - r2, 0.0, 1.0)
p0 = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)

CORNERS = [(0, 0), (N, 0), (0, N), (N, N)]
anchor_idx_np = np.array([iy * (N + 1) + ix for ix, iy in CORNERS], dtype=np.int32)
anchor_local = np.array([[float(xs[ix]), 0.0, float(xs[iy])] for ix, iy in CORNERS],
                        dtype=np.float32)
p0[anchor_idx_np] = anchor_local + np.array([0.0, CATCH_Y, 0.0], np.float32)
p0[:, 0] += HOME[0]
p0[:, 2] += HOME[1]

fx = np.linspace(-CLOTH / 2, CLOTH / 2, N + 1, dtype=np.float32)
mx, mz = np.meshgrid(fx, fx)
flat = np.stack([mx, np.zeros_like(mx), mz], axis=-1).reshape(-1, 3)
lra_np = np.linalg.norm(flat[:, None, :] - flat[anchor_idx_np][None, :, :],
                        axis=2).astype(np.float32).reshape(-1)

im_np = np.full(V, 1.0, dtype=np.float32)
im_np[anchor_idx_np] = 0.0

pos = wp.array(p0, dtype=wp.vec3, device=device)
prev = wp.array(p0, dtype=wp.vec3, device=device)
pred = wp.zeros(V, dtype=wp.vec3, device=device)
scratch = wp.zeros(V, dtype=wp.vec3, device=device)
nrm = wp.zeros(V, dtype=wp.vec3, device=device)
im = wp.array(im_np, dtype=float, device=device)
lra = wp.array(lra_np, dtype=float, device=device)
anchor_idx = wp.array(anchor_idx_np, dtype=int, device=device)
anchor_tgt = wp.array(p0[anchor_idx_np].copy(), dtype=wp.vec3, device=device)

bp = wp.array(np.array([CANNON], np.float32), dtype=wp.vec3, device=device)
bv = wp.zeros(1, dtype=wp.vec3, device=device)
bpred = wp.zeros(1, dtype=wp.vec3, device=device)
impulse = wp.zeros(1, dtype=wp.vec3, device=device)

solve_graph = None
if device.is_cuda and ITERS % 2 == 0:
    try:
        with wp.ScopedCapture(device) as _cap:
            _a, _b = pred, scratch
            for _ in range(ITERS):
                wp.launch(solve, dim=V, device=device,
                          inputs=[_a, _b, im, N, N, REST, bpred, bp, BALL_R + 0.012, MU,
                                  prev, anchor_tgt, lra, impulse, M_PARTICLE, 0.35])
                _a, _b = _b, _a
        solve_graph = _cap.graph
    except Exception as exc:                       # noqa: BLE001 - capture is optional
        print(f"  (no graph capture: {exc})")


# ---- the rig ------------------------------------------------------------------

class Rig:
    """Four anchors on a square that translates. Position and velocity of the
    centre are the state; the corners ride along rigidly."""

    def __init__(self):
        self.p = np.array([HOME[0], CATCH_Y, HOME[1]])
        self.v = np.zeros(3)
        self.travel = 0.0
        self.peak_speed = 0.0
        self.starved = 0          # frames the speed cap bit
        self.reach_capped = 0     # frames the arms' own envelope bit
        self.p_pending = self.p.copy()
        self.spread = 0.0
        self.tilt = 0.0                       # rad; + raises the downrange side
        self.tilt_dir = np.array([1.0, 0.0, 0.0])   # horizontal, unit, downrange
        self.peak_tilt = 0.0

    def targets(self):
        # Spreading the anchors re-tensions the sheet. Left slack it simply
        # crumples over the ball and hides it, which is the same lesson the
        # throw taught in reverse: a slack sheet has no shape to speak of.
        scale = 1.0 + self.spread * (CLOTH - SPAN) / SPAN
        c = anchor_local * scale
        if self.tilt != 0.0:
            # Per-corner height offsets on the UNSCALED square, so re-tensioning
            # for the throw cannot also deepen the scoop. Corners downrange of
            # the centre go up, corners uprange go down, and the centre -- which
            # is what the whole approach was planned to -- stays on CATCH_Y.
            s = (anchor_local[:, 0] * self.tilt_dir[0]
                 + anchor_local[:, 2] * self.tilt_dir[2])
            c = c.copy()
            c[:, 1] += math.tan(self.tilt) * s
        return (c + self.p).astype(np.float32)

    def set_tilt(self, angle, direction=None):
        if direction is not None:
            d = np.array([direction[0], 0.0, direction[2]], float)
            n = float(np.linalg.norm(d))
            if n > 1e-6:
                self.tilt_dir = d / n
        self.tilt = float(angle)
        self.peak_tilt = max(self.peak_tilt, abs(self.tilt))

    def goto(self, dt, p_cmd, v_cmd):
        """Move toward the commanded pose, capped at ARM_SPEED."""
        step = p_cmd - self.p
        d = float(np.linalg.norm(step))
        if d > ARM_SPEED * dt:
            step *= ARM_SPEED * dt / d
            self.starved += 1
        self.p_pending = self.p + step
        self.v = step / dt if dt > 0 else v_cmd
        self._commit()

    def _commit(self):
        home3 = np.array([HOME[0], CATCH_Y, HOME[1]])
        nxt = self.p_pending
        off = nxt - home3
        d = math.hypot(off[0], off[2])
        # The arms simply run out of arm, and they do it much sooner sideways
        # than downrange -- reach_limit() below is that envelope, and it is a
        # tenth of what --arm-reach used to allow across the z axis. Letting the
        # plan drag the rig past it does not fail gracefully: one corner ends up
        # a metre behind the other three, the sheet stops being a sheet, and the
        # ball leaves at whatever speed the wreckage gives it. Stopping at the
        # boundary loses the same catches and keeps the sheet.
        lim = ARM_REACH if d < 1e-6 else min(ARM_REACH, reach_limit(off[0] / d,
                                                                   off[2] / d))
        if d > lim:
            nxt = home3 + np.array([off[0] * lim / d, off[1], off[2] * lim / d])
            self.v[0] = self.v[2] = 0.0
            self.reach_capped += 1
        self.travel += float(np.linalg.norm(nxt - self.p))
        self.peak_speed = max(self.peak_speed, float(np.linalg.norm(self.v)))
        self.p = nxt

    def step(self, dt, accel):
        self.v += accel * dt
        sp = float(np.linalg.norm(self.v))
        if sp > ARM_SPEED:
            self.v *= ARM_SPEED / sp
        self.p_pending = self.p + self.v * dt
        self._commit()


rig = Rig()


# ---- the sensors --------------------------------------------------------------
# Two pinhole frame cameras on tripods, standing in the scene where they can be
# seen (and where they can see each other). Both are SECONDARY VIEWS on the
# renderer, so one render() produces both of them from one scene build in one
# submission -- the same simulated instant, twice. That is the property the
# whole estimator rests on: bearings taken a tick apart cannot be intersected,
# and two render() calls could never hand back the same instant twice.
#
# Placement is measured rather than eyeballed. The pair subtends 87 degrees at
# the catch zone -- a midpoint triangulation's depth error scales as 1/sin of
# that angle, so a narrow pair would be no better than the single camera it
# replaced -- and both see the whole muzzle-to-apex-to-catch arc with margin.
# Both facts are asserted at startup rather than trusted.
# Aimed at the middle of the WORKING VOLUME (the whole muzzle-to-crossing box
# over the aim range), not at the catch zone: a sensor framed on the catch loses
# the launch, and the first 150 ms of track is what lets the rig commit early.
# The 6 m standoff is set by the widest shot the cannon can take -- at az -12,
# el 58, 7 m/s the ball crosses at x +2.3, and from 4.5 m out that fell off the
# edge of the east frame. The coverage check below is what found that.
SENSOR_TARGET = np.array([0.10, 1.90, 0.0])
SENSOR_POSES = [("east", np.array([4.34, 2.40, 4.24])),
                ("west", np.array([-4.14, 2.40, 4.24]))]
# Two rays that miss each other by more than this are not looking at the same
# object, whatever each camera thinks it found. It costs nothing to compute --
# the midpoint solve produces it -- and it is the one quality signal a single
# camera can never have.
STEREO_GAP = cli_arg("--stereo-gap", 0.12, float)
# How far an observation may sit from the fitted arc before it is disbelieved.
# The ball moves 15 mm per tick, so this is 20 ticks of slack: loose enough that
# a real manoeuvre would survive it, tight enough that a lock on the cloth
# cannot.
OBS_GATE = cli_arg("--obs-gate", 0.35, float)
# The cannon's own exclusion disc, in metres of world radius around the TRUNNION
# (not the muzzle): centred there it covers the carriage, the wheels and the
# flash, and because the ball starts 0.62 m out along the barrel it leaves the
# disc 58 ms after firing at 6.5 m/s -- so masking the gun costs almost nothing
# of the early track. Held for 0.30 s, which is past the recoil and past the
# point where the smoke is still bright enough to out-vote a ball.
MUZZLE_MASK_R = cli_arg("--mask-r", 0.75, float)
MUZZLE_MASK_S = cli_arg("--mask-s", 0.35, float)
# The flash is a different problem from the gun. A PointLight puts a pool of
# changing brightness on the ground several metres wide, and no disc that covers
# it can avoid also covering the ball -- the ball flies out THROUGH the pool.
# Measured: with an 80 ms flash the detector spent its first 20 frames locked on
# the lit floor 1.2 m below the shot and the first crossing call came out 0.99 m
# wrong. So the flash is short instead: it is over in five sensor ticks, which
# is before the ball has cleared the barrel and before the tracker's first
# accepted observation (21 ms, measured), and for exactly those ticks the mask
# widens to cover the whole lit pool. Nothing is lost, because there was nothing
# to see yet.
FLASH_MASK_R = cli_arg("--flash-mask-r", 3.0, float)


def basis_of(eye, target):
    """Forward / right / up of a camera at `eye` looking at `target`, built the
    same way three.js lookAt builds them (world up, right = fwd x up_world), so
    this projection model IS the one the view renders with."""
    fwd = np.asarray(target, float) - np.asarray(eye, float)
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    return fwd, right, np.cross(right, fwd)


class Sensor:
    """One frame camera: a pose, a pinhole model, and a frame-difference blob
    detector that reports where the ball is IN PIXELS. It never reports range;
    that is the pair's job, and it is why the range bias is gone."""

    def __init__(self, name, eye, target):
        self.name = name
        self.eye = np.asarray(eye, float)
        self.fwd, self.right, self.up = basis_of(eye, target)
        self.w, self.h = SENSOR_W, SENSOR_H
        # ONE focal length: the view renders at exactly (w, h) with this
        # vertical fov and aspect w/h, so the pixels are square by construction.
        # The old DVS needed two because its detector resolution and its render
        # resolution disagreed, and using one for both biased every range by 14%.
        self.f = (self.h / 2.0) / math.tan(math.radians(SENSOR_FOV) / 2.0)
        self.camera = tp.PerspectiveCamera(SENSOR_FOV, self.w / self.h, 0.1, 100)
        self.camera.position.set(*[float(v) for v in self.eye])
        self.camera.look_at(*[float(v) for v in np.asarray(target, float)])
        self.handle = 0
        self.prev = None          # last frame's luma, for the difference
        self.gate = None          # (u, v, half-width) to search inside
        self.mask = None          # (u, v, r) to ignore entirely: the cannon
        self.last_px = None       # (u, v, r_rms, n) of the last accepted blob
        self.rejected = 0
        # The cell size and the starting search radius are in PIXELS, so they
        # have to follow the sensor resolution -- otherwise --sensor silently
        # changes what counts as a cluster.
        self.cell = max(3, int(round(8 * self.h / 480.0)))
        self.r0 = 40.0 * self.h / 480.0

    def project(self, p):
        d = np.asarray(p, float) - self.eye
        z = float(np.dot(d, self.fwd))
        if z < 0.05:
            return None
        return (self.w / 2.0 + self.f * float(np.dot(d, self.right)) / z,
                self.h / 2.0 - self.f * float(np.dot(d, self.up)) / z)

    def ray(self, u, v):
        d = (self.fwd + self.right * ((u - self.w / 2.0) / self.f)
             - self.up * ((v - self.h / 2.0) / self.f))
        return d / np.linalg.norm(d)

    def changed(self, rgb):
        """Pixels whose luma moved since this camera's previous frame, as (n, 2)
        (x, y). Nothing here knows anything a real camera would not: no ids, no
        segmentation, no depth -- eight bits per channel and a subtraction."""
        a = rgb.astype(np.int32)
        luma = (a[:, :, 0] * 77 + a[:, :, 1] * 150 + a[:, :, 2] * 29) >> 8
        prev, self.prev = self.prev, luma
        if prev is None or prev.shape != luma.shape:
            return None
        ys, xs = np.nonzero(np.abs(luma - prev) > DIFF_THRESHOLD)
        return np.stack([xs, ys], axis=1).astype(np.float64)

    def blob(self, xy):
        """Densest-cell seed, then shrink onto whatever is actually there.

        Seed on the DENSEST patch, not the centroid. The sheet and four arms are
        a far bigger source of changed pixels than the ball -- a moving rig
        outlines the whole cloth -- so a centroid over the raw difference lands
        on cloth. But the cloth changes as a thin sparse OUTLINE while the ball
        changes as a solid disc, so pixels-per-cell separates them cleanly.

        The first lock is unguarded (before the rig moves the ball is the only
        thing changing); after that the search is gated to a window around the
        prediction, which is what keeps the sheet's own motion out of the track.
        """
        if xy is None or xy.shape[0] < MIN_PIXELS:
            return None
        if self.mask is not None:
            # The rig knows where its own cannon is. Flash, smoke and a recoiling
            # carriage are all change, and for the first fraction of a second
            # they are far more change than the ball -- so a disc around the
            # gun's own projected position is excluded. This is a prior about the
            # ROBOT, not about the ball: it is the same disc whether a shot has
            # been fired or not, it never moves with the target, and the ball is
            # outside it within ~60 ms of leaving the barrel.
            mu, mv, mr = self.mask
            xy = xy[(xy[:, 0] - mu) ** 2 + (xy[:, 1] - mv) ** 2 > mr * mr]
            if xy.shape[0] < MIN_PIXELS:
                return None
        if self.gate is not None:
            gu, gv, gr = self.gate
            keep = (np.abs(xy[:, 0] - gu) < gr) & (np.abs(xy[:, 1] - gv) < gr)
            if keep.sum() >= MIN_PIXELS:
                xy = xy[keep]
        ncx = self.w // self.cell + 1
        key = ((xy[:, 1] // self.cell).astype(np.int64) * ncx
               + (xy[:, 0] // self.cell).astype(np.int64))
        seed = int(np.bincount(key).argmax())
        c = np.array([(seed % ncx + 0.5) * self.cell, (seed // ncx + 0.5) * self.cell])

        # Collect around the seed and let the radius settle: start wide enough
        # for a close ball, then shrink onto whatever is actually there.
        radius, sel = self.r0, xy
        for _ in range(3):
            d = np.linalg.norm(xy - c, axis=1)
            sel = xy[d < radius]
            if sel.shape[0] < MIN_PIXELS:
                self.rejected += 1
                self.last_px = None
                return None
            c = sel.mean(axis=0)
            rr = float(np.sqrt(((sel - c) ** 2).sum(axis=1).mean()))
            radius = min(max(2.5 * rr, 0.2 * self.r0), 1.5 * self.r0)
        c = sel.mean(axis=0)
        r_rms = float(np.sqrt(((sel - c) ** 2).sum(axis=1).mean()))
        if r_rms < 0.5:
            self.rejected += 1
            self.last_px = None
            return None
        self.last_px = (c[0], c[1], r_rms, sel.shape[0])
        # Tight gate: the sheet is a large, bright, fast mover, and once the
        # ball nears it a loose window swallows cloth pixels and the centre
        # walks off the ball onto the cloth.
        self.gate = (c[0], c[1], max(4.0 * r_rms, 0.75 * self.r0))
        return c


class StereoTracker:
    """Ball state from two synchronized frame cameras.

    Per tick each sensor reports a bearing and the pair is intersected by
    least-squares midpoint into one 3-D observation; the arc is fitted to those.
    |g| is still fixed at 9.81 in the fit, but it is now a CONSISTENCY
    constraint on six well-observed unknowns rather than the only thing pinning
    the scale, which is what it had to be with one camera and no range.
    """

    def __init__(self, sensors):
        self.sensors = sensors
        self.obs = []                # (t, x, y, z) in world space
        self.fit = None              # (p0, v0, t0) of the ballistic fit
        self.rejected = 0
        self.gap = []                # ray-to-ray miss: the triangulation's own error bar
        self.recent = []

    def triangulate(self, uv):
        """Closest approach of the two bearing rays. The midpoint is the
        observation; the gap between them is how much to believe it."""
        s0, s1 = self.sensors
        d0, d1 = s0.ray(*uv[0]), s1.ray(*uv[1])
        w0 = s0.eye - s1.eye
        b = float(np.dot(d0, d1))
        denom = 1.0 - b * b
        if denom < 1e-6:                      # parallel rays carry no depth
            return None, float('inf')
        dd, ee = float(np.dot(d0, w0)), float(np.dot(d1, w0))
        s, t = (b * ee - dd) / denom, (ee - b * dd) / denom
        if s < 0.1 or t < 0.1:                # behind a camera: not a sighting
            return None, float('inf')
        p0, p1 = s0.eye + d0 * s, s1.eye + d1 * t
        return 0.5 * (p0 + p1), float(np.linalg.norm(p0 - p1))

    def predict_at(self, t):
        if self.fit is None:
            return None
        p0, v0, t0 = self.fit
        dt = t - t0
        return p0 + v0 * dt + 0.5 * G_NP * dt * dt

    def observe(self, frames, t_now):
        # The difference is taken between the frame just rendered and the one
        # before it, so what it measures is motion ACROSS that interval: the
        # blob is the union of two discs and its centroid belongs half a tick
        # back. Stamping it there rather than at "now" removes a bias worth
        # 6 mm at 3 m/s, and costs one subtraction.
        t_obs = t_now - 0.5 / SENSOR_HZ
        pred = self.predict_at(t_obs)
        uv = []
        for s, rgb in zip(self.sensors, frames):
            if pred is not None:
                # Re-anchor each search window on the PREDICTED position rather
                # than on the last blob. A gate that follows its own measurement
                # walks: one bad lock onto the cloth and it never comes back --
                # measured on the az -12 shot, where the west camera's window
                # slid onto the sheet and put the final call 0.96 m out. A gate
                # anchored on the fitted arc cannot walk, because the arc is
                # fitted to 200 observations and one frame cannot move it.
                puv = s.project(pred)
                if puv is not None:
                    r = 4.0 * s.last_px[2] if s.last_px else 0.75 * s.r0
                    s.gate = (puv[0], puv[1], max(r, 0.5 * s.r0))
            c = s.blob(s.changed(rgb))
            if c is None:
                return None
            uv.append((c[0], c[1]))
        p, gap = self.triangulate(uv)
        if p is None or gap > STEREO_GAP:
            self.rejected += 1
            return None
        # Innovation gate. Both cameras can agree on the wrong thing -- they see
        # the same cloth -- so the ray gap alone is not enough. This is the one
        # place the estimator is allowed to disbelieve its own sensors, and it
        # is bounded: the arc has to have converged first, and OBS_GATE is loose
        # enough (0.35 m) that real ball motion never trips it.
        if pred is not None and float(np.linalg.norm(p - pred)) > OBS_GATE:
            self.rejected += 1
            return None
        self.gap.append(gap)
        self.obs.append((t_obs, p[0], p[1], p[2]))
        return p

    def solve_fit(self):
        """Least squares p(t) = p0 + v0 t + g t^2 / 2 over the observations,
        with one Huber-style reweighting pass so the occasional collapsed-cluster
        frame cannot drag the arc."""
        if len(self.obs) < 24:
            return None
        a = np.array(self.obs)
        if a[-1, 0] - a[0, 0] < 0.12:
            return None
        t = a[:, 0] - a[0, 0]
        y = a[:, 1:] - 0.5 * G_NP[None, :] * (t ** 2)[:, None]
        A = np.stack([np.ones_like(t), t], axis=1)
        w = np.ones_like(t)
        for _ in range(2):
            Aw = A * w[:, None]
            sol, *_ = np.linalg.lstsq(Aw, y * w[:, None], rcond=None)
            res = np.linalg.norm(A @ sol - y, axis=1)
            s = max(float(np.median(res)), 1e-3)
            w = 1.0 / (1.0 + (res / (3.0 * s)) ** 2)
        self.fit = (sol[0], sol[1], a[0, 0])
        return self.fit

    def predict_crossing(self, y_plane):
        """Where and when the fitted arc next crosses y_plane going down."""
        if self.fit is None:
            return None
        p0, v0, t0 = self.fit
        a, b, c = 0.5 * G_NP[1], v0[1], p0[1] - y_plane
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        r = math.sqrt(disc)
        ts = sorted(((-b + r) / (2 * a), (-b - r) / (2 * a)))
        t_hit = ts[-1]                       # the descending root
        if t_hit <= 0:
            return None
        p = p0 + v0 * t_hit + 0.5 * G_NP * t_hit * t_hit
        v = v0 + G_NP * t_hit
        if not np.all(np.isfinite(p)) or np.max(np.abs(p)) > 12.0:
            return None                       # outside any arena: not a prediction
        # Publish only once the answer stops moving. A ballistic fit over a short
        # baseline is confident and wrong -- the first call here came out 13 m
        # away -- and the rig will happily chase it. Agreement across consecutive
        # frames is the cheapest honest confidence signal available.
        self.recent.append(p)
        if len(self.recent) > 5:
            self.recent.pop(0)
        if len(self.recent) < 5:
            return None
        spread = float(np.max(np.linalg.norm(np.array(self.recent) -
                                             np.mean(self.recent, axis=0), axis=1)))
        if spread > 0.15:
            return None
        return t0 + t_hit, p, v


# ---- scene --------------------------------------------------------------------

if not tp.HAS_VULKAN:
    print("This demo needs the Vulkan backend (secondary views live there).")
    sys.exit(0)

canvas = tp.Canvas("threepp - cloth catch", width=VIEW_W, height=VIEW_H,
                   headless=HEADLESS, vsync=False)
# Outside the window loop, render() drives `flush_frames` full GPU frames per
# call. Three of them exist to make a readback off the MAILBOX SWAPCHAIN
# deterministic -- and --tune never touches the swapchain: the sensors are
# secondary views read from their own images. So the perception run pays for one
# frame instead of three, which is worth 7.5 ms/tick of a 15.9 ms tick. The clip
# path does read the swapchain and keeps all three.
renderer = tp.VulkanRenderer(canvas, 1 if TUNE else 3)

scene = tp.Scene()
# Dusk, not void. The range is lit by a low warm sun and four lamp poles, and a
# near-black sky made every silhouette read as a cut-out; this is the tone the
# analytic lights sit against.
scene.background = 0x1b2634

# The SHOT camera, and nothing else. It no longer feeds any sensor, so it may
# be orbited freely, mid-flight and all -- which is the point of moving the
# measurement onto its own cameras.
# High enough to see INTO the bowl -- from a low angle the sheet's near lip
# hides the very thing the shot is about -- while still framing the whole arc.
# The film pass sits closer: the arms are the subject and the wide framing left
# most of the frame as empty floor. Still wide enough that the cannon and the
# whole arc stay in shot.
# Framed on the RIG, not on the whole arena. The cannon sits at the edge and the
# ball flies in from off-frame, which reads better than watching it be launched --
# and it stops most of the frame being empty floor between the two.
# The film pose also has to LOOK INTO THE BIN, which is a stronger constraint
# than framing anything: a bin occludes its own floor with its own near rim
# unless the sight line drops faster than depth/radius, and check_props() below
# turns that into a printed margin instead of a thing noticed in an mp4.
EYE = np.array([2.00, 3.70, 4.55]) if FILM else np.array([0.60, 3.15, 5.60])
TGT = np.array([0.35, 1.30, 0.10]) if FILM else np.array([-0.30, 1.05, 0.0])
camera = tp.PerspectiveCamera(45, VIEW_W / VIEW_H, 0.1, 100)
camera.position.set(*EYE)
camera.look_at(*TGT)

sensors = [Sensor(name, eye, SENSOR_TARGET) for name, eye in SENSOR_POSES]

scene.add(tp.HemisphereLight(0xbfd2ea, 0x1a1e24, 0.50))
sun = tp.DirectionalLight(0xffd9b0, 3.0)            # low and warm: late afternoon
sun.position.set(3.5, 5.2, 4.0)
sun.cast_shadow = True
# The default ortho frustum is a metre wide, so the backstop was half in shadow
# and half in full sun with a hard vertical seam across it. 11 m covers the pad,
# the wall and the props.
sun.set_shadow_frustum(-8.0, 8.0, 8.0, -8.0)
scene.add(sun)
# White arms on a dark floor lose their silhouette under a single lamp, and with
# GI off there is nothing to fill the shadow side back in. This one costs
# 0.05 ms/tick, measured, so it is no longer a film-only luxury.
_fill = tp.DirectionalLight(0x9fb6d8, 0.9)          # cool fill from the shadow side
_fill.position.set(-5.0, 2.5, 2.0)
scene.add(_fill)
if FILM:
    _rim = tp.DirectionalLight(0xffd7a8, 1.7)       # warm rim, from behind
    _rim.position.set(-1.5, 3.0, -6.0)
    scene.add(_rim)

# ---- the range ----------------------------------------------------------------
# A place rather than a void, and procedural rather than downloaded: one baked
# ground texture, a backstop, some crates, four lamp poles and the basket. All of
# it is STATIC, which is the only reason it is affordable -- nothing here moves,
# so nothing here costs a per-frame update, and (measured) it also puts nothing
# into the frame difference the detector reads.

GROUND_M = 40.0                       # metres the ground plane spans
GROUND_PX = 1536                      # 26 mm per texel, enough for painted lines


def _cyl_between(a, b, radius, mat):
    """A cylinder spanning a -> b. The geometry's axis is +y, and a look_at is
    degenerate for a near-vertical leg, so the orientation goes in as Euler
    angles: XYZ order applies Ry*Rz to (0,1,0), giving
    (-sin t cos p, cos t, sin t sin p) for tilt t from vertical, azimuth p."""
    d = np.asarray(b, float) - np.asarray(a, float)
    length = float(np.linalg.norm(d))
    if length < 1e-6:
        return None
    d /= length
    tilt = math.acos(max(-1.0, min(1.0, d[1])))
    sin_t = math.sqrt(max(1.0 - d[1] * d[1], 1e-12))
    mesh = tp.Mesh(tp.CylinderGeometry(radius, radius, length, 10), mat)
    mesh.rotation.set(0.0, math.atan2(d[2] / sin_t, -d[0] / sin_t), tilt)
    mid = 0.5 * (np.asarray(a, float) + np.asarray(b, float))
    mesh.position.set(*[float(v) for v in mid])
    mesh.cast_shadow = True
    return mesh


def _bilinear_up(a, s):
    """Bilinear upsample of a square array to s x s, in numpy alone."""
    c = a.shape[0]
    idx = (np.arange(s) + 0.5) * c / s - 0.5
    i0 = np.floor(idx).astype(np.int64)
    f = (idx - i0).astype(np.float32)
    lo, hi = np.clip(i0, 0, c - 1), np.clip(i0 + 1, 0, c - 1)
    b = a[lo, :] * (1.0 - f)[:, None] + a[hi, :] * f[:, None]
    return b[:, lo] * (1.0 - f)[None, :] + b[:, hi] * f[None, :]


def _range_texture(s=GROUND_PX, w=GROUND_M):
    """The asphalt the whole range stands on: 40 m of noise and nothing else.

    Nothing PAINTED lives here any more. 1536 texels over 40 m is 26 mm each,
    and a 26 mm step is perfectly invisible in gravel noise and perfectly
    visible in the edge of a painted line -- the hazard ring came out of this
    texture as a staircase you could count. The markings moved to the pad below,
    which covers a twentieth of the area at more than the same resolution."""
    rng = np.random.default_rng(7)
    xs = ((np.arange(s) + 0.5) / s - 0.5) * w
    X, Z = np.meshgrid(xs, -xs)
    n = (0.55 * _bilinear_up(rng.random((24, 24)).astype(np.float32), s)
         + 0.30 * _bilinear_up(rng.random((96, 96)).astype(np.float32), s)
         + 0.15 * rng.random((s, s)).astype(np.float32))
    img = np.empty((s, s, 3), np.float32)
    img[:] = np.array([0.128, 0.140, 0.156]) * (0.72 + 0.56 * n)[:, :, None]
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


# The poured pad, and everything painted on it: its own mesh, its own texture,
# sized to the marked area instead of to the arena. 10.8 x 8.05 m at 5.3 mm per
# texel -- five times the density of the ground, on a twentieth of the area, for
# a fifth of the memory. It is opaque and it replaces the ground underneath
# rather than being an alpha decal over it, which means no transparency, no
# overlay pass, no sort order, and no risk of the markings vanishing from the
# sensor views (secondary views run no overlay pass at all).
PAD_X0, PAD_X1 = -5.20, 5.60
PAD_Z0, PAD_Z1 = -3.05, 5.00
PAD_PX = int(cli_arg("--pad-px", 2048, float))


def _pad_texture(w=PAD_PX):
    """Concrete, expansion joints, a hazard ring around the catch zone, a firing
    lane out of the cannon and a circle under the basket -- painted in WORLD
    coordinates, so the ring really is centred on the rig and the lane really
    does point down the barrel.

    Every edge is a SMOOTHSTEP over a signed distance in metres rather than a
    boolean mask, so a line half a texel wide still reads as a line and the
    chevrons stop being a staircase. That is what actually fixed the blocky
    ring: resolution alone just makes smaller stairs."""
    h = max(2, int(round(w * (PAD_Z1 - PAD_Z0) / (PAD_X1 - PAD_X0))))
    rng = np.random.default_rng(11)
    X, Z = np.meshgrid(PAD_X0 + (np.arange(w) + 0.5) * (PAD_X1 - PAD_X0) / w,
                       PAD_Z1 - (np.arange(h) + 0.5) * (PAD_Z1 - PAD_Z0) / h)
    px = (PAD_X1 - PAD_X0) / w                    # metres per texel
    n = (0.55 * _bilinear_up(rng.random((24, 24)).astype(np.float32), max(w, h))[:h, :w]
         + 0.30 * _bilinear_up(rng.random((96, 96)).astype(np.float32), max(w, h))[:h, :w]
         + 0.15 * rng.random((h, w)).astype(np.float32))
    img = np.array([0.208, 0.212, 0.213]) * (0.80 + 0.40 * n)[:, :, None]

    def edge(sdf):
        """1 inside, 0 outside, one texel of ramp across the boundary."""
        return np.clip(0.5 - sdf / (1.4 * px), 0.0, 1.0)

    def paint(alpha, rgb, strength=1.0):
        a = (strength * (0.55 + 0.45 * n) * alpha)[:, :, None]
        img[:] = img * (1.0 - a) + np.array(rgb) * a

    # Expansion joints, so the pad reads as poured slabs and not as a decal.
    joint = np.minimum(np.abs(((X - 0.2 + 1.35) % 2.7) - 1.35),
                       np.abs(((Z + 1.35) % 2.7) - 1.35))
    img *= (1.0 - 0.38 * edge(joint - 0.016))[:, :, None]

    # Firing lane: two solid edges and a dashed centre, from behind the cannon
    # to the near edge of the hazard ring.
    lane = np.maximum(np.maximum(CANNON[0] - 1.1 - X, X - (HOME[0] - 1.9)),
                      np.abs(Z) - 0.62)
    paint(edge(np.maximum(lane, np.abs(np.abs(Z) - 0.60) - 0.045)),
          (0.80, 0.80, 0.78), 0.85)
    dash = np.abs(((X + 40.0) % 0.72) - 0.21) - 0.21
    paint(edge(np.maximum(np.maximum(lane, np.abs(Z) - 0.045), dash)),
          (0.80, 0.80, 0.78), 0.85)
    # Hazard ring: a yellow annulus around the catch zone, chevroned in black.
    rr = np.hypot(X - HOME[0], Z - HOME[1])
    ring = np.abs(rr - 1.65) - 0.13
    paint(edge(ring), (0.86, 0.66, 0.10), 0.9)
    # The chevron edge is measured along the ring, not in radians, or the stripes
    # would blur at the inner radius and stay hard at the outer one.
    theta = np.arctan2(Z - HOME[1], X - HOME[0])
    chev = (np.abs(((theta + math.pi) % 0.32) - 0.08) - 0.08) * np.maximum(rr, 1e-3)
    paint(edge(np.maximum(ring, chev)), (0.06, 0.06, 0.07), 0.9)
    # And a plain circle under the basket, so the target is marked on the floor
    # as well as standing on a post.
    br = np.hypot(X - BASKET[0], Z - BASKET[1])
    paint(edge(np.abs(br - 0.56) - 0.04), (0.75, 0.76, 0.78), 0.8)
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


_ground_mat = standard_material(0xffffff, 0.95)
_ground_mat.map = tp.data_texture(_range_texture(), True)
ground = tp.Mesh(tp.PlaneGeometry(GROUND_M, GROUND_M), _ground_mat)
ground.rotate_x(-math.pi / 2)
ground.receive_shadow = True
scene.add(ground)

_pad_mat = standard_material(0xffffff, 0.95)
_pad_mat.map = tp.data_texture(_pad_texture(), True)
pad = tp.Mesh(tp.PlaneGeometry(PAD_X1 - PAD_X0, PAD_Z1 - PAD_Z0), _pad_mat)
pad.rotate_x(-math.pi / 2)
pad.position.set(0.5 * (PAD_X0 + PAD_X1), 0.004, 0.5 * (PAD_Z0 + PAD_Z1))
pad.receive_shadow = True
scene.add(pad)


def _box(size, pos, mat, ry=0.0):
    m = tp.Mesh(tp.BoxGeometry(*size), mat)
    m.position.set(*[float(v) for v in pos])
    if ry:
        m.rotate_y(ry)
    m.cast_shadow = True
    m.receive_shadow = True
    scene.add(m)
    return m


# Backstop: what a range puts behind the thing it is throwing at. It also gives
# the sensors a flat, still background behind the catch zone instead of 40 m of
# empty floor, which the frame difference is measurably happier with.
_panel = standard_material(0x353b44, 0.85, 0.05)
_frame_mat = standard_material(0x1a1d22, 0.7, 0.35)
for _i in range(7):
    _cx = -6.3 + _i * 2.1
    # The panels do not RECEIVE shadow: the sun's ortho shadow camera has to
    # cover the whole pad, and at that texel density the wall self-shadowed in
    # one hard-edged patch that read as a white sheet hung on it. Nothing casts
    # onto the wall from the front anyway, so this costs nothing real.
    _box((2.0, 2.45, 0.09), (_cx, 1.225, -3.20), _panel).receive_shadow = False
    _box((0.14, 2.75, 0.16), (_cx - 1.05, 1.375, -3.24), _frame_mat).cast_shadow = False
_box((14.9, 0.16, 0.26), (0.0, 2.52, -3.22), _frame_mat).cast_shadow = False
_box((14.9, 0.22, 0.34), (0.0, 0.11, -3.22), _frame_mat).cast_shadow = False

# Crates and drums, kept off to the sides and behind: nothing may stand between
# a sensor and the arc, and the coverage check does not test occlusion.
_crate = standard_material(0x6a5433, 0.9)
_crate2 = standard_material(0x4e5a48, 0.9)
_drum = standard_material(0x9c3d2a, 0.6, 0.25)
for _p, _s, _m, _r in (((-4.55, 0.35, -2.35), (0.70, 0.70, 0.70), _crate, 0.22),
                       ((-4.30, 0.95, -2.30), (0.52, 0.52, 0.52), _crate2, -0.35),
                       ((-5.20, 0.28, -1.55), (0.56, 0.56, 0.56), _crate2, 0.51),
                       ((4.70, 0.35, -2.20), (0.70, 0.70, 0.70), _crate, -0.18),
                       ((4.95, 0.95, -2.24), (0.50, 0.50, 0.50), _crate, 0.40),
                       ((5.55, 0.30, 1.10), (0.60, 0.60, 0.60), _crate2, 0.12)):
    _box(_s, _p, _m, _r)
for _x, _z in ((-5.75, -2.55), (-5.75, -1.90), (5.90, -2.50)):
    _d = tp.Mesh(tp.CylinderGeometry(0.29, 0.29, 0.88, 16), _drum)
    _d.position.set(_x, 0.44, _z)
    _d.cast_shadow = True
    scene.add(_d)

# Lamp poles. With GI off an emissive surface lights nothing, so each pole
# carries its own small PointLight -- four cheap analytic lights that put a warm
# pool on the pad and give the dusk sky something to be dusk against.
_pole_mat = standard_material(0x2a2f36, 0.6, 0.5)
_lamp_mat = standard_material(0x201d18, 0.5, 0.2,
                             emissive=0xffc98a, emissive_intensity=1.2)
for _x, _z in ((-5.0, 3.6), (5.2, 3.6), (-5.0, -2.9), (5.2, -2.9)):
    _p = tp.Mesh(tp.CylinderGeometry(0.055, 0.075, 3.30, 12), _pole_mat)
    _p.position.set(_x, 1.65, _z)
    _p.cast_shadow = True
    scene.add(_p)
    _head = tp.Mesh(tp.CylinderGeometry(0.20, 0.12, 0.16, 14), _lamp_mat)
    _head.position.set(_x, 3.32, _z)
    scene.add(_head)
    _lp = tp.PointLight(0xffc98a, 4.0, 12.0, 2.0)
    _lp.position.set(_x, 3.24, _z)
    scene.add(_lp)

# A pennant line between the two poles at the BACK, sagging as a cable does.
# Not the camera-side pair, which is where it started: those poles stand 0.6 m
# in front of the shot camera and both sensors, so a 20 cm flag covered a third
# of every frame and read as a sheet hung on the backstop.
_cable_mat = standard_material(0x14171b, 0.8, 0.2)
_FLAGS = (0xd8532c, 0xe6b13a, 0xdfe3e8)
_a, _b = np.array([-5.0, 3.24, -2.9]), np.array([5.2, 3.24, -2.9])
_prev_pt = None
for _k in range(13):
    _u = _k / 12.0
    _pt = _a + (_b - _a) * _u + np.array([0.0, -0.85 * 4.0 * _u * (1.0 - _u), 0.0])
    if _prev_pt is not None:
        _seg = _cyl_between(_prev_pt, _pt, 0.010, _cable_mat)
        if _seg is not None:
            _seg.cast_shadow = False
            scene.add(_seg)
        _f = tp.Mesh(tp.PlaneGeometry(0.20, 0.26),
                     standard_material(_FLAGS[_k % 3], 0.9, 0.0, side=tp.Side.Double))
        _f.position.set(*[float(v) for v in 0.5 * (_prev_pt + _pt) - np.array([0, 0.15, 0])])
        _f.rotate_y(0.30 * math.sin(_k * 2.1))
        scene.add(_f)
    _prev_pt = _pt

# The basket: a bin on a post, and the target of the throw. Its collider is in
# ball_finish -- an open cylinder with a rim ring -- so a throw that catches the
# lip rattles out instead of scoring.
_bin_mat = standard_material(0x8d9299, 0.55, 0.55, side=tp.Side.Double)
_bin = tp.Mesh(tp.CylinderGeometry(BASKET_R, BASKET_R * 0.82,
                                   BASKET_RIM - BASKET_FLOOR, 28, 1, True), _bin_mat)
_bin.position.set(float(BASKET[0]), 0.5 * (BASKET_RIM + BASKET_FLOOR), float(BASKET[1]))
_bin.cast_shadow = True
scene.add(_bin)
_bin_floor = tp.Mesh(tp.CylinderGeometry(BASKET_R * 0.82, BASKET_R * 0.82, 0.03, 28),
                     standard_material(0x2c3138, 0.8))
_bin_floor.position.set(float(BASKET[0]), BASKET_FLOOR, float(BASKET[1]))
scene.add(_bin_floor)
_rim_ring = tp.Mesh(tp.TorusGeometry(BASKET_R, 0.028, 8, 30),
                    standard_material(0xe4b23a, 0.4, 0.7))
_rim_ring.rotate_x(math.pi / 2)
_rim_ring.position.set(float(BASKET[0]), BASKET_RIM, float(BASKET[1]))
_rim_ring.cast_shadow = True
scene.add(_rim_ring)
_post = tp.Mesh(tp.CylinderGeometry(0.055, 0.07, BASKET_FLOOR, 14), _pole_mat)
_post.position.set(float(BASKET[0]), 0.5 * BASKET_FLOOR, float(BASKET[1]))
_post.cast_shadow = True
scene.add(_post)
_foot = tp.Mesh(tp.CylinderGeometry(0.30, 0.34, 0.05, 18), _frame_mat)
_foot.position.set(float(BASKET[0]), 0.025, float(BASKET[1]))
scene.add(_foot)

geometry = tp.PlaneGeometry(CLOTH, CLOTH, N, N)
cloth_mesh = tp.Mesh(geometry, standard_material(0xd4542e, 0.88, side=tp.Side.Double))
cloth_mesh.cast_shadow = True
scene.add(cloth_mesh)

def _ball_texture(w=768, h=384):
    u = (np.arange(w) + 0.5) / w
    v = (np.arange(h) + 0.5) / h
    U, V = np.meshgrid(u, v)
    img = np.empty((h, w, 3), np.uint8)
    img[:] = (243, 246, 252)
    img[((np.floor(U * 12) + np.floor(V * 6)) % 2).astype(bool)] = (22, 26, 34)
    img[np.abs(V - 0.5) < 0.030] = (0, 208, 255)      # equator
    img[np.abs(U - 0.5) < 0.010] = (255, 96, 30)      # meridian
    return img


ball_mat = standard_material(0xffffff, 0.34)
ball_mat.map = tp.data_texture(_ball_texture(), True)
ball_mesh = tp.Mesh(tp.SphereGeometry(BALL_R, 48, 32), ball_mat)
ball_mesh.cast_shadow = True
scene.add(ball_mesh)

anchor_meshes = []
for _ in range(4):
    m = tp.Mesh(tp.SphereGeometry(0.045, 18, 14), standard_material(0x1b1f25, 0.45, 0.4))
    scene.add(m)
    anchor_meshes.append(m)


# ---- the sensor rigs ----------------------------------------------------------
# Each sensor gets a tripod and a camera body at its actual pose, pointing along
# its actual look direction. They are in shot and they can see each other, which
# is what a two-camera capture rig looks like; more usefully, the measurement
# geometry is now something you can point at instead of a constant in a comment.

def build_sensor_rig(sensor):
    metal = standard_material(0x2b3038, 0.55, 0.45)
    body_mat = standard_material(0x14171c, 0.5, 0.35)
    apex = sensor.eye + np.array([0.0, -0.12, 0.0])
    for k in range(3):                       # three legs, splayed 120 degrees
        psi = math.radians(90.0 + 120.0 * k + 25.0)
        foot = np.array([apex[0] + 0.40 * math.cos(psi), 0.0,
                         apex[2] + 0.40 * math.sin(psi)])
        leg = _cyl_between(apex, foot, 0.018, metal)
        if leg is not None:
            scene.add(leg)
    column = _cyl_between(apex, sensor.eye + np.array([0.0, -0.02, 0.0]), 0.028, metal)
    if column is not None:
        scene.add(column)
    # The body is a Group at the eye pointed with look_at, which is safe here
    # (the look direction is nowhere near vertical) and puts -z along fwd, the
    # same convention the camera itself uses.
    head = tp.Group()
    head.position.set(*[float(v) for v in sensor.eye])
    head.look_at(*[float(v) for v in SENSOR_TARGET])
    scene.add(head)
    shell = tp.Mesh(tp.BoxGeometry(0.17, 0.13, 0.22), body_mat)
    shell.position.set(0.0, 0.0, 0.05)
    shell.cast_shadow = True
    head.add(shell)
    lens = tp.Mesh(tp.CylinderGeometry(0.048, 0.055, 0.11, 18), metal)
    lens.rotate_x(-math.pi / 2)              # +y -> -z, which is forward
    lens.position.set(0.0, 0.0, -0.11)
    lens.cast_shadow = True
    head.add(lens)
    glass = tp.Mesh(tp.CylinderGeometry(0.040, 0.040, 0.012, 18),
                    standard_material(0x0a2b3a, 0.15, 0.95))
    glass.rotate_x(-math.pi / 2)
    glass.position.set(0.0, 0.0, -0.168)
    head.add(glass)


for _s in sensors:
    build_sensor_rig(_s)

# ---- the cannon ---------------------------------------------------------------
# A barrel that visibly carries the aim, so a shot can be lined up before it is
# taken rather than discovered afterwards. CANNON is the trunnion; the ball
# leaves the MUZZLE, so the visual and the ballistics agree.
BARREL_L = 0.62
BARREL_R = 0.075

cannon = tp.Group()
cannon.position.set(*[float(x) for x in CANNON])
scene.add(cannon)

# The trunnion sits at CANNON (0.55 m up, which is where the shot leaves from),
# so the carriage has to reach the floor FROM there: wheel bottom at local
# -0.55 puts it exactly on y = 0 rather than hovering a quarter of a metre up.
_AXLE_Y, _WHEEL_R = -0.27, 0.28
for _z in (-0.145, 0.145):
    _cheek = tp.Mesh(tp.BoxGeometry(0.34, 0.34, 0.035),
                     standard_material(0x343b45, 0.6, 0.45))
    _cheek.position.set(-0.02, -0.15, _z)
    _cheek.cast_shadow = True
    cannon.add(_cheek)
for _z in (-0.19, 0.19):
    _w = tp.Mesh(tp.CylinderGeometry(_WHEEL_R, _WHEEL_R, 0.05, 24),
                 standard_material(0x15181d, 0.55, 0.3))
    _w.rotate_x(math.pi / 2)
    _w.position.set(0.0, _AXLE_Y, _z)
    _w.cast_shadow = True
    cannon.add(_w)
    _hub = tp.Mesh(tp.CylinderGeometry(0.055, 0.055, 0.075, 14),
                   standard_material(0x6b7480, 0.4, 0.8))
    _hub.rotate_x(math.pi / 2)
    _hub.position.set(0.0, _AXLE_Y, _z)
    cannon.add(_hub)
_axle = tp.Mesh(tp.CylinderGeometry(0.022, 0.022, 0.40, 12),
                standard_material(0x22262d, 0.6, 0.5))
_axle.rotate_x(math.pi / 2)
_axle.position.set(0.0, _AXLE_Y, 0.0)
cannon.add(_axle)
# Trail spar down to the ground behind, so it stands on three points like a gun
# rather than on an invisible plinth.
_trail = tp.Mesh(tp.BoxGeometry(0.82, 0.065, 0.11),
                 standard_material(0x343b45, 0.65, 0.4))
_trail.position.set(-0.42, -0.40, 0.0)
_trail.rotate_z(math.radians(-24.0))
_trail.cast_shadow = True
cannon.add(_trail)

barrel_pivot = tp.Group()
cannon.add(barrel_pivot)
_barrel = tp.Mesh(tp.CylinderGeometry(BARREL_R * 0.80, BARREL_R, BARREL_L, 24),
                  standard_material(0x4a515b, 0.30, 0.85))
_barrel.position.set(0.0, BARREL_L / 2, 0.0)
_barrel.cast_shadow = True
barrel_pivot.add(_barrel)
_band = tp.Mesh(tp.CylinderGeometry(BARREL_R * 1.02, BARREL_R * 1.02, 0.05, 24),
                standard_material(0x6b7480, 0.35, 0.9))
_band.position.set(0.0, BARREL_L - 0.035, 0.0)
barrel_pivot.add(_band)
_breech = tp.Mesh(tp.SphereGeometry(BARREL_R * 1.15, 20, 14),
                  standard_material(0x4a515b, 0.30, 0.85))
barrel_pivot.add(_breech)


def aim_dir():
    el, az = math.radians(aim["el"]), math.radians(aim["az"])
    return np.array([math.cos(el) * math.cos(az), math.sin(el),
                     math.cos(el) * math.sin(az)])


def muzzle():
    return CANNON + aim_dir() * BARREL_L


def point_barrel(recoil=0.0):
    barrel_pivot.rotation.set(0.0, -math.radians(aim["az"]),
                              math.radians(aim["el"] + RECOIL_DEG * recoil) - math.pi / 2)


# ---- what firing looks like ---------------------------------------------------
# A flash, a puff and a recoil, and all three of them are things the sensors can
# see. That is the point: the detector is frame differencing, so the cannon
# announcing itself is a real pollution source and has to be handled rather than
# rendered somewhere the sensors are not looking.
RECOIL_M = cli_arg("--recoil", 0.06, float)       # m the carriage rides back
RECOIL_DEG = cli_arg("--recoil-deg", 4.0, float)  # degrees the barrel rocks up
RECOIL_TAU = 0.048                                # s to the peak; back down by ~0.20
FLASH_S = cli_arg("--flash", 0.030, float)        # s the point light lasts: five
                                                  # sensor ticks, one clip frame
EFFECTS = "--no-effects" not in sys.argv

# The flash is a child of the barrel pivot, so it rides the aim and the recoil
# without any of its own maths. Emissive rather than a light source, because
# with GI and ReSTIR off (the default path) emissive geometry lights nothing --
# the PointLight below is what actually reaches the carriage and the smoke.
flash_mat = standard_material(0x100a04, 0.4, 0.0,
                              emissive=0xffcf8c, emissive_intensity=0.0)
flash = tp.Mesh(tp.CylinderGeometry(0.235, 0.030, 0.34, 18), flash_mat)
flash.position.set(0.0, BARREL_L + 0.14, 0.0)
flash.visible = False
barrel_pivot.add(flash)
flash_light = tp.PointLight(0xffb066, 0.0, 4.0, 2.0)
scene.add(flash_light)

# ONE ParticleField, capacity 256, host-owned: the puff is 60 particles whose
# whole life is 12 lines of numpy, so the GPU never has to know it is a
# simulation. HostRing with stable slots means slot i is the same particle every
# frame, which is what lets the radius grow with age.
SMOKE_N = 60
SMOKE_CAP = 256
_smoke_cfg = tp.ParticleField.Config()
_smoke_cfg.capacity = SMOKE_CAP
_smoke_cfg.ownership = tp.ParticleField.Ownership.HostRing
_smoke_cfg.w_semantic = tp.ParticleField.WSemantic.Radius
_smoke_cfg.uniform_radius = 0.08
_smoke_cfg.host_stable_slots = True
smoke = tp.ParticleField.create(_smoke_cfg)
smoke.set_billboard_repr(tp.Color(0x8e959e), tp.Color(0x30353c), 1.0, 1.0)
_sb = smoke.billboard_repr
_sb.alpha_over = True            # smoke OCCLUDES; additive smoke is a firework
_sb.opacity = 0.22
_sb.lit = True                   # lit by the scene's own sun, so the puff sits
_sb.lit_phase_g = 0.30           # in the same light as the gun that made it
_sb.lit_ambient = 0.16
_sb.softness = 1.0
_sb.fade_power = 1.7
_sb.bright_jitter = 0.30
_sb.glow = 0.0
smoke.set_live_count(0)
scene.add(smoke)

_smoke_p = np.zeros((SMOKE_CAP, 3))
_smoke_v = np.zeros((SMOKE_CAP, 3))
_smoke_age = np.full(SMOKE_CAP, 1e9)
_smoke_life = np.ones(SMOKE_CAP)
_smoke_r0 = np.full(SMOKE_CAP, 0.05)
_smoke_buf = np.zeros((SMOKE_CAP, 4), np.float32)
_smoke_rng = np.random.default_rng(11)
_smoke_live = [False]
_SMOKE_BUOY = np.array([0.0, 0.62, 0.0])


def puff(at, direction):
    """Sixty particles out of the bore, slow and spreading."""
    n = SMOKE_N
    _smoke_p[:n] = at + 0.05 * _smoke_rng.normal(0.0, 1.0, (n, 3))
    _smoke_v[:n] = (direction * _smoke_rng.uniform(0.35, 1.30, n)[:, None]
                    + 0.30 * _smoke_rng.normal(0.0, 1.0, (n, 3)))
    _smoke_age[:n] = 0.0
    _smoke_life[:n] = _smoke_rng.uniform(0.9, 1.8, n)
    _smoke_r0[:n] = _smoke_rng.uniform(0.045, 0.085, n)
    _smoke_live[0] = True


def step_smoke(dt):
    """Buoyant, heavily damped, growing with age. Deliberately slow and soft:
    a fast bright puff is a second mover in both sensor frames and the densest
    -cell seed would happily lock onto it."""
    if not _smoke_live[0]:
        return
    _smoke_age[:] += dt
    alive = _smoke_age < _smoke_life
    if not alive.any():
        _smoke_live[0] = False
        smoke.set_live_count(0)
        return
    _smoke_v[:] += (_SMOKE_BUOY - 2.3 * _smoke_v) * dt
    _smoke_p[:] += _smoke_v * dt
    f = np.clip(_smoke_age / np.maximum(_smoke_life, 1e-3), 0.0, 1.0)
    _smoke_buf[:, :3] = _smoke_p
    _smoke_buf[:, 3] = np.where(alive, _smoke_r0 * (1.0 + 2.4 * f), -1.0)
    smoke.submit(_smoke_buf, dt)


def muzzle_effects(t):
    """Flash, light and recoil as functions of seconds since the shot. `t` is
    None outside a shot, which parks everything."""
    if t is None or not EFFECTS:
        flash.visible = False
        flash_light.intensity = 0.0
        point_barrel()
        cannon.position.set(*[float(x) for x in CANNON])
        return
    # Impulse response of a recoil spring: out fast, back with no overshoot.
    s = (t / RECOIL_TAU) * math.exp(1.0 - t / RECOIL_TAU) if t >= 0.0 else 0.0
    point_barrel(s)
    back = aim_dir()
    back = np.array([back[0], 0.0, back[2]])
    back /= max(float(np.linalg.norm(back)), 1e-9)
    cannon.position.set(*[float(x) for x in CANNON - back * (RECOIL_M * s)])
    flash.visible = 0.0 <= t < FLASH_S
    if flash.visible:
        flash_mat.emissive_intensity = 38.0 * (1.0 - t / FLASH_S)
    flash_light.intensity = 70.0 * math.exp(-t / (FLASH_S / 2.2)) if flash.visible else 0.0
    if flash_light.intensity > 0.0:
        flash_light.position.set(*[float(x) for x in muzzle() + aim_dir() * 0.12])


point_barrel()

# ---- the aim preview ----------------------------------------------------------
# Ground truth, and deliberately so: this is the PLAYER's aid, not the robot's.
# It is hidden the moment the shot is fired, so nothing it draws can be confused
# with -- or leak into -- what the tracker has to work out for itself. Watching
# the cyan reticle converge onto this amber one IS the perception story.
ARC_N = 72
_arc_geo = tp.BufferGeometry()
_arc_geo.set_attribute("position", np.zeros((ARC_N, 3), np.float32))
_arc_mat = tp.LineBasicMaterial()
_arc_mat.color = 0xffae3a
_arc_mat.transparent = True
_arc_mat.opacity = 0.55
aim_arc = tp.Line(_arc_geo, _arc_mat)
scene.add(aim_arc)

_hit_mat = tp.MeshBasicMaterial()
_hit_mat.color = 0xffae3a
_hit_mat.transparent = True
_hit_mat.opacity = 0.6
_hit_mat.side = tp.Side.Double
aim_hit = tp.Mesh(tp.RingGeometry(0.13, 0.17, 40), _hit_mat)
aim_hit.rotate_x(-math.pi / 2)
scene.add(aim_hit)


def true_crossing(p0, v0, y_plane):
    """Where an unobstructed shot would cross y_plane on the way down."""
    a, b, c = 0.5 * G_NP[1], v0[1], p0[1] - y_plane
    disc = b * b - 4 * a * c
    if disc < 0:
        return None, None
    r = math.sqrt(disc)
    t = max((-b + r) / (2 * a), (-b - r) / (2 * a))
    return (t, p0 + v0 * t + 0.5 * G_NP * t * t) if t > 0 else (None, None)


def update_aim_preview(visible):
    aim_arc.visible = visible
    aim_hit.visible = visible
    if not visible:
        return
    p0, v0 = muzzle(), aim["speed"] * aim_dir()
    t_hit, p_hit = true_crossing(p0, v0, CATCH_Y)
    span = t_hit if t_hit else 1.4
    ts = np.linspace(0.0, span, ARC_N)[:, None]
    pts = p0[None, :] + v0[None, :] * ts + 0.5 * G_NP[None, :] * ts * ts
    _arc_geo.update_attribute("position", pts.astype(np.float32))
    if p_hit is not None:
        aim_hit.visible = True
        aim_hit.position.set(float(p_hit[0]), CATCH_Y + 0.004, float(p_hit[2]))
    else:
        aim_hit.visible = False


if RENDER_SCALE < 0.999:
    renderer.render_scale = RENDER_SCALE
renderer.probe_gi = GI
renderer.deferred_ao = AO
if NO_DENOISE:
    renderer.denoise = False
# ReSTIR earns its keep with many lights and emissive geometry. This scene has
# two analytic lights and no emitters, which is exactly the case the legacy
# per-light NEE loops handle more cheaply -- and at 1 spp with two lights the
# two paths agree, so unlike the GI toggles this one is not a look/perf trade.
# (Needs a threepp built after the restir_di binding was added.)
if hasattr(renderer, "restir_di"):
    renderer.restir_di = RESTIR


# ---- the arms -----------------------------------------------------------------
# Four 5-DOF arms, defined as a URDF STRING and parsed by the engine's own loader
# rather than fetched: no external asset, but real joints with real limits and
# real speed caps. They are not set dressing -- the cloth corner follows each
# arm's ACHIEVED tool pose, so if an arm cannot get there, the sheet does not go
# there either, and the catch fails for a reason you can point at.
#
# Mounted on pedestals at plate height, which is the whole trick. Floor-mounted
# they would need ~1.25 m of reach just to climb to the catch plane before
# covering any of the corner's ~1.25 m sweep; measured, this arm's practical
# envelope is ~0.85 m horizontal (well under its 1.14 m nominal -- joint limits
# and the base link eat the rest). On a pedestal it only has to cover the sweep,
# and base-to-corner distance stays between 0.5 and 0.95 m throughout.
ARMS = "--no-arms" not in sys.argv
ARM_KIND = cli_arg("--arm", "fr3", str)        # fr3 | iiwa | proc
# Pedestal height and stand-off are the rig's REACH ENVELOPE, not decoration.
# At 0.50 m and 0.50 m out the arms spent 0.60 m of their metre going straight
# up and another 0.50 m going sideways, which left 0.16 m of travel across the z
# axis -- so the az +8 shot, whose crossing is only 0.42 m off the axis, already
# had an arm 0.19 m behind its corner. Standing them taller and tucking them in
# costs nothing and triples the sideways envelope (0.16 -> 0.44 m).
PEDESTAL_Y = cli_arg("--pedestal-y", 0.70, float)
ARM_OUT = cli_arg("--arm-out", 0.34, float)    # outward from the sheet, in z
ARM_LEAD = cli_arg("--arm-lead", 0.48, float)  # toward the intercept, in x
JOINT_SPEED = cli_arg("--joint-speed", 5.0, float)   # rad/s cap per joint
# How far an arm can put its TOOL frame from its own base. Nothing at run time
# is bounded by this -- the arms are bounded by their own IK, and the report
# prints what they actually achieved -- but check_props() needs a number to keep
# scenery out of the arms, and 0.917 m is already known to be reachable because
# that is the distance from a pedestal to its corner at home and the settle
# converges there to 0.0000 m.
ARM_TOOL_REACH = cli_arg("--arm-tool-reach", 1.00, float)


def reach_limit(ux, uz):
    """How far the rig centre may travel along a horizontal unit direction
    before an arm runs out of arm.

    --arm-reach was a flag with 2.0 typed into it and it was fiction. A pedestal
    sits at its corner plus (ARM_LEAD, -(CATCH_Y - PEDESTAL_Y), +/-ARM_OUT), so
    the corner OFFSETS cancel and the envelope depends only on the direction
    travelled and the z sign of the pedestal. Solving |(r*u - offset)| =
    ARM_TOOL_REACH for r and taking the worse of the two z signs gives the real
    boundary: 1.33 m straight downrange, 0.44 m straight across. That asymmetry
    is why every off-axis shot in this demo had an arm a fifth of a metre behind
    its corner and nobody could see why.
    """
    h = CATCH_Y - PEDESTAL_Y
    best = ARM_REACH
    for sz in (1.0, -1.0):
        bx, bz = -ARM_LEAD, -sz * ARM_OUT
        b = 2.0 * (ux * bx + uz * bz)
        c = bx * bx + bz * bz + h * h - ARM_TOOL_REACH ** 2
        disc = b * b - 4.0 * c
        if disc <= 0.0:
            return 0.0
        best = min(best, max(0.5 * (-b + math.sqrt(disc)), 0.0))
    return best

_L = dict(base=0.16, upper=0.50, fore=0.44, wrist=0.12)


def _link(name, radius, length, rgba, origin_z):
    return f"""
  <link name="{name}">
    <visual>
      <origin xyz="0 0 {origin_z}" rpy="0 0 0"/>
      <geometry><cylinder radius="{radius}" length="{length}"/></geometry>
      <material name="{name}_m"><color rgba="{rgba}"/></material>
    </visual>
    <inertial><mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>"""


def arm_urdf():
    L = _L
    return f"""<?xml version="1.0"?>
<robot name="catcher">
  {_link("base", 0.085, L['base'], "0.13 0.15 0.18 1", L['base'] / 2)}
  {_link("shoulder", 0.070, 0.14, "0.78 0.80 0.85 1", 0.0)}
  {_link("upper", 0.055, L['upper'], "0.78 0.80 0.85 1", L['upper'] / 2)}
  {_link("fore", 0.045, L['fore'], "0.26 0.30 0.36 1", L['fore'] / 2)}
  {_link("wrist", 0.034, L['wrist'], "0.78 0.80 0.85 1", L['wrist'] / 2)}
  {_link("tool", 0.026, 0.06, "0.95 0.42 0.14 1", 0.03)}
  <joint name="pan" type="revolute">
    <parent link="base"/><child link="shoulder"/>
    <origin xyz="0 0 {L['base']}" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="200" velocity="4.0"/>
  </joint>
  <joint name="lift" type="revolute">
    <parent link="shoulder"/><child link="upper"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" effort="200" velocity="4.0"/>
  </joint>
  <joint name="elbow" type="revolute">
    <parent link="upper"/><child link="fore"/>
    <origin xyz="0 0 {L['upper']}" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.6" upper="2.6" effort="200" velocity="5.0"/>
  </joint>
  <joint name="wrist1" type="revolute">
    <parent link="fore"/><child link="wrist"/>
    <origin xyz="0 0 {L['fore']}" rpy="0 0 0"/><axis xyz="0 1 0"/>
    <limit lower="-2.6" upper="2.6" effort="120" velocity="6.0"/>
  </joint>
  <joint name="wrist2" type="revolute">
    <parent link="wrist"/><child link="tool"/>
    <origin xyz="0 0 {L['wrist']}" rpy="0 0 0"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="120" velocity="6.0"/>
  </joint>
</robot>"""


# threepp_data ships a Franka FR3 and a KUKA iiwa. Both measure ~0.90 m of
# practical reach against a corner sweep of ~1.25 m, so the pedestal is doing the
# work: at 0.50 m high and 0.35-0.50 m outboard, both track the corner through
# the whole sweep to under a millimetre. Falls back to the procedural arm when
# the data checkout is not present, so the demo still runs from a bare clone.
_ARM_URDFS = {"fr3": (os.path.join("urdf", "franka", "fr3.urdf"), "fr3_hand_tcp"),
              "iiwa": (os.path.join("urdf", "lbr_iiwa_14_r820.urdf"), None)}
# Franka's ready pose, so the redundant joints settle somewhere an arm would
# actually sit rather than wherever the solver happens to leave them.
_REST = {"fr3": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
         "iiwa": [0.0, 0.6, 0.0, -1.5, 0.0, 1.0, 0.0],
         "proc": [0.0, -0.95, 1.75, -0.80, 0.0]}


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


def _tool_pos(m):
    """Translation of a Matrix4. tool_transform reports in the robot's parent
    frame, which -- measured -- INCLUDES the Robot object's own placement, so
    these come out directly in world coordinates."""
    a = m.to_numpy().reshape(4, 4)
    return a[:3, 3] if abs(a[3, 3] - 1.0) < 1e-6 else a[3, :3]


class Arm:
    def __init__(self, base_xyz, kind):
        rel = _ARM_URDFS.get(kind)
        path = os.path.join(data_dir(), rel[0]) if (rel and data_dir()) else ""
        if path and os.path.isfile(path):
            self.robot = tp.URDFLoader().load(path)
            if rel[1]:
                self.robot.set_end_effector(rel[1])
            self.kind = kind
        else:
            self.robot = tp.URDFLoader().parse(os.getcwd(), arm_urdf())
            self.kind = "proc"
        # The URDF loader gives every collision shape a white wireframe material
        # and adds it alongside the visual mesh, so each link wears a triangulated
        # shell of itself. show_colliders() hides the collider GROUP; the meshes
        # get hidden too so it holds regardless of how visibility is inherited.
        self.robot.show_colliders(False)
        self.robot.position.set(*[float(v) for v in base_xyz])
        self.robot.rotate_x(-math.pi / 2)   # URDF is Z-up; the scene is Y-up
        self.robot.update_matrix()
        opts = tp.IkOptions()
        opts.max_joint_speed = JOINT_SPEED
        opts.position_tolerance = 0.002
        # A 3-DOF position task on a 5-DOF arm leaves two joints free, and left
        # to itself the solver folds them into the shortest pose -- which sits
        # bolt upright against the pedestal and reads as a post, not an arm.
        # Biasing the null space toward shoulder-out / elbow-bent costs nothing
        # in reach and is the difference between four posts and four arms.
        rest = list(_REST.get(self.kind, []))
        if rest:
            rest = (rest + [0.0] * self.robot.num_dof)[:self.robot.num_dof]
            opts.rest_pose = rest
            opts.rest_pose_gain = 0.06
        self.solver = tp.IkSolver(self.robot, opts)
        self.q = list(opts.rest_pose) if opts.rest_pose else [0.0] * self.robot.num_dof
        self.err = 0.0
        scene.add(self.robot)
        ped = tp.Mesh(tp.CylinderGeometry(0.11, 0.13, base_xyz[1], 20),
                      standard_material(0x1b1f25, 0.7, 0.35))
        ped.position.set(float(base_xyz[0]), float(base_xyz[1]) / 2, float(base_xyz[2]))
        ped.cast_shadow = True
        scene.add(ped)

    def settle(self, target):
        """Converge onto a corner with the speed cap OFF, once, before the run.

        The arms are born at q = 0 (straight up) while the sheet corners are
        elsewhere; with a 5 rad/s cap they spend the first second of every run
        catching up, and that transient IS the worst tracking error. A real rig
        starts already holding the cloth."""
        t = tp.Vector3(float(target[0]), float(target[1]), float(target[2]))
        for _ in range(300):
            self.q, _r = self.solver.solve(self.q, t, 0.0)
        self.robot.set_joint_values(self.q)
        return _tool_pos(self.solver.tool_transform(self.q))

    def track(self, target, dt):
        """Drive toward `target` and return where the tool ACTUALLY got to.

        ONE solve per tick. The max_joint_speed cap is applied per CALL, so
        looping the solver N times over the same dt quietly multiplies the cap by
        N -- measured, six calls at a 5 rad/s cap let joints run at 30 rad/s, and
        that is what made the arms look janky. solve() already iterates
        internally (IkOptions.max_iterations, default 100); the outer loop was
        buying nothing but a broken speed limit.
        """
        t = tp.Vector3(float(target[0]), float(target[1]), float(target[2]))
        self.q, _r = self.solver.solve(self.q, t, dt)
        self.robot.set_joint_values(self.q)
        got = _tool_pos(self.solver.tool_transform(self.q))
        self.err = float(np.linalg.norm(got - np.asarray(target, float)))
        return got


arms = []
if ARMS:
    for _k, _c in enumerate(anchor_local):
        arms.append(Arm((HOME[0] + _c[0] + ARM_LEAD,
                         PEDESTAL_Y,
                         HOME[1] + _c[2] + math.copysign(ARM_OUT, _c[2])), ARM_KIND))
    _home = np.array(p0[anchor_idx_np], np.float64)
    _res = np.array([a.settle(_home[k]) for k, a in enumerate(arms)])
    if "--arm-tool-reach" not in " ".join(sys.argv):
        # MEASURE the reach instead of asserting it. A datasheet number would be
        # the wrong number anyway (the tool frame is past the flange, and the
        # null-space bias costs a little of it), and everything downstream --
        # the travel envelope the rig is clamped to, and whether the basket is
        # standing inside an arm -- is only as honest as this.
        _a0 = arms[0]
        _base = np.array([HOME[0] + anchor_local[0][0] + ARM_LEAD, PEDESTAL_Y,
                          HOME[1] + anchor_local[0][2] - ARM_OUT])
        _u = (_home[0] - _base) / np.linalg.norm(_home[0] - _base)
        _q, _best = list(_a0.q), 0.0
        for _r in np.arange(0.70, 1.45, 0.02):
            _t = _base + _u * _r
            _tv = tp.Vector3(*[float(v) for v in _t])
            for _ in range(90):
                _a0.q, _ = _a0.solver.solve(_a0.q, _tv, 0.0)
            if np.linalg.norm(_tool_pos(_a0.solver.tool_transform(_a0.q)) - _t) > 0.005:
                break
            _best = float(_r)
        _a0.q = _q
        _a0.settle(_home[0])
        ARM_TOOL_REACH = _best
    _kind = arms[0].kind
    print(f"arms: 4 x {_kind} ({arms[0].robot.num_dof} dof, tool "
          f"{arms[0].robot.end_effector_link}), pedestals at {PEDESTAL_Y:.2f} m, "
          f"settled onto the corners to {max(a.err for a in arms):.4f} m, tool "
          f"reach measured at {ARM_TOOL_REACH:.2f} m"
          + ("   [threepp_data not found - procedural fallback]"
             if _kind == "proc" and ARM_KIND != "proc" else ""))


# ---- attaching the sensors ----------------------------------------------------
# add_view shares the primary's render pass and pipelines, so it returns 0 until
# a first render() has happened. One throwaway frame, then the handles, then an
# assert -- a silently-zero handle would show up much later as a tracker that
# never sees anything.
renderer.render(scene, camera)
for _s in sensors:
    _s.handle = renderer.add_view(_s.camera, SENSOR_W, SENSOR_H)
    if _s.handle == 0:
        print("could not create a secondary view for the sensors")
        sys.exit(1)


def shot_landmarks(az, el, speed):
    """Muzzle, apex and catch-plane crossing of one shot."""
    e, a = math.radians(el), math.radians(az)
    d = np.array([math.cos(e) * math.cos(a), math.sin(e), math.cos(e) * math.sin(a)])
    p_m, v_m = CANNON + d * BARREL_L, speed * d
    t_apex = max(v_m[1] / 9.81, 0.0)
    _t, p_hit = true_crossing(p_m, v_m, CATCH_Y)
    return {"muzzle": p_m,
            "apex": p_m + v_m * t_apex + 0.5 * G_NP * t_apex ** 2,
            "crossing": p_hit if p_hit is not None else p_m}


def check_coverage():
    """Both sensors must see the whole SHOT, not just the catch. Projecting the
    muzzle, the apex and the crossing into each and demanding a margin turns
    'the framing looks fine' into something that fails loudly when a pose, an
    fov or a default is changed into uselessness -- which is exactly how the
    4.5 m standoff these poses started at was found to lose the widest shot.

    Swept over the corners of the aim envelope, not just the aim this run
    happens to use, so a pose is judged against every shot the cannon can take.
    """
    # The envelope is the set of shots the RIG can reach, not everything the
    # cannon can throw: at az -14, el 56, 7.5 m/s the ball crosses 2.6 m from
    # home against an ARM_REACH of 2.0, so framing for it would cost resolution
    # on every shot that can actually be caught. These corners all land inside.
    pts = {"rig": np.array([HOME[0], CATCH_Y, HOME[1]])}
    for az in (-12.0, 0.0, 12.0):
        for el in (58.0, 66.0):
            for speed in (6.0, 7.0):
                for k, p in shot_landmarks(az, el, speed).items():
                    pts[f"{k} az{az:+.0f} el{el:.0f} v{speed:.1f}"] = p
    for k, p in shot_landmarks(aim["az"], aim["el"], aim["speed"]).items():
        pts[f"{k} (this shot)"] = p
    worst = 1e9
    for s in sensors:
        for name, p in pts.items():
            uv = s.project(p)
            if uv is None:
                raise SystemExit(f"sensor {s.name}: {name} is behind the camera")
            m = min(uv[0], s.w - uv[0], uv[1], s.h - uv[1])
            worst = min(worst, m)
            if m < 8.0:
                raise SystemExit(f"sensor {s.name}: {name} projects to "
                                 f"({uv[0]:.0f},{uv[1]:.0f}), outside a "
                                 f"{s.w}x{s.h} frame")
    # Depth error from a midpoint triangulation scales as 1/sin(subtended
    # angle), so a narrow pair would be no better than the one camera this
    # replaced. 40 degrees is the floor; the chosen poses give 87.
    catch = np.array([HOME[0], CATCH_Y, HOME[1]])
    a = sensors[0].eye - catch
    b = sensors[1].eye - catch
    ang = math.degrees(math.acos(float(np.dot(a, b))
                                 / (np.linalg.norm(a) * np.linalg.norm(b))))
    if ang < 40.0:
        raise SystemExit(f"sensor pair subtends only {ang:.0f} deg at the catch zone")
    return ang, worst


SUBTENDED, COVER_MARGIN = check_coverage()
print(f"sensors: 2 x {SENSOR_W}x{SENSOR_H} at {SENSOR_FOV:.0f} deg, "
      f"{SUBTENDED:.0f} deg apart at the catch zone, whole aim envelope inside "
      f"both frames with {COVER_MARGIN:.0f} px to spare")


def check_props():
    """The basket has to stand somewhere no arm, no sheet and no camera wants.

    This is the same idea as check_coverage: a placement that used to be checked
    by looking at a frame is checked by arithmetic instead, and it fails loudly.
    The first basket really was inside an arm -- 0.56 m from the rear-left
    pedestal axis with a 0.42 m bin on a 0.34 m foot -- and nothing in the demo
    said so, because a bin and an arm interpenetrating renders perfectly well.

    Four tests:
      * throwing reach: 1.2-1.5 m from rig home.
      * pedestals: the bin's foot clear of every pedestal axis by CLEARANCE, and
        the bin outside every arm's swept envelope (the horizontal circle an
        arm can put its tool on at CATCH_Y, which is what the sqrt below is).
      * the catch: the bin clear of everywhere the SHEET goes, which is every
        crossing in the aim envelope plus the cloth's own half-diagonal.
      * the film camera: the bin inside the frame, and the sight line into it
        steeper than the bin's own near rim.
    """
    CLEARANCE = 0.35
    b = np.array([BASKET[0], BASKET[1]])
    d_home = float(np.linalg.norm(b - HOME))
    if not 1.15 < d_home < 1.55:
        raise SystemExit(f"basket is {d_home:.2f} m from rig home, outside the "
                         f"1.2-1.5 m the stroke can throw")
    # An arm mounted at PEDESTAL_Y can put its tool anywhere within this radius
    # of its own axis at the catch height; beyond it the arm simply is not.
    sweep = math.sqrt(max(ARM_TOOL_REACH ** 2 - (CATCH_Y - PEDESTAL_Y) ** 2, 0.0))
    worst_ped, worst_sweep = 1e9, 1e9
    for c in anchor_local:
        p = np.array([HOME[0] + c[0] + ARM_LEAD,
                      HOME[1] + c[2] + math.copysign(ARM_OUT, c[2])])
        dd = float(np.linalg.norm(b - p))
        worst_ped = min(worst_ped, dd - 0.34 - 0.13)     # bin foot vs pedestal
        worst_sweep = min(worst_sweep, dd - BASKET_R - sweep)
    if worst_ped < CLEARANCE or worst_sweep < 0.0:
        raise SystemExit(f"basket is inside an arm: {worst_ped:.2f} m of foot "
                         f"clearance to the nearest pedestal (want {CLEARANCE:.2f}), "
                         f"{worst_sweep:.2f} m outside its swept envelope")
    # Everywhere the sheet gets to: home, and every crossing in the aim
    # envelope. The sheet is an axis-aligned square of half-width CLOTH/2 when
    # fully spread, so this is a box distance, not a radius -- a radius would
    # have condemned this placement over a corner that points the other way.
    centres = [np.array([HOME[0], HOME[1]])]
    for az in (-12.0, 0.0, 12.0):
        for el in (58.0, 66.0):
            for speed in (6.0, 7.0):
                c = shot_landmarks(az, el, speed)["crossing"]
                centres.append(np.array([c[0], c[2]]))
    worst_sheet = min(math.hypot(max(abs(b[0] - c[0]) - 0.5 * CLOTH, 0.0),
                                 max(abs(b[1] - c[1]) - 0.5 * CLOTH, 0.0))
                      - BASKET_R for c in centres)
    if worst_sheet < 0.10:
        raise SystemExit(f"the sheet sweeps through the basket "
                         f"({-worst_sheet:.2f} m of overlap)")
    # Looking into the bin: the sight line must drop faster across the bin's
    # radius than the rim stands above the resting ball.
    ball_y = BASKET_FLOOR + BALL_R
    need = math.degrees(math.atan2(BASKET_RIM - ball_y, BASKET_R))
    got = math.degrees(math.atan2(EYE[1] - ball_y,
                                  math.hypot(EYE[0] - BASKET[0], EYE[2] - BASKET[1])))
    print(f"props:   basket {d_home:.2f} m from home, {worst_ped:.2f} m of foot "
          f"clearance to the nearest pedestal ({worst_sweep:.2f} m outside its "
          f"reach), {worst_sheet:.2f} m clear of the swept sheet; camera looks "
          f"into the bin at {got:.0f} deg against the {need:.0f} deg its own rim "
          f"needs")
    if FILM and got < need + 3.0:
        print(f"         (the ball in the bin will be hidden by the near rim: "
              f"{got:.0f} deg of sight line against {need:.0f} needed)")


def check_reach():
    """Print which shots in the aim envelope the ARMS can actually be taken to.

    The catch was being planned against --arm-reach and executed against four
    Franka arms, and those are not the same number. Saying so at startup is the
    difference between "the wide shot is flaky" and "the wide shot crosses
    2.06 m from home and the arms stop at 1.10"."""
    rows, out = [], 0
    for az in (-12.0, 0.0, 12.0):
        for el in (58.0, 66.0):
            for speed in (6.0, 7.0):
                c = shot_landmarks(az, el, speed)["crossing"]
                off = np.array([c[0] - HOME[0], c[2] - HOME[1]])
                d = float(np.linalg.norm(off))
                lim = reach_limit(*(off / max(d, 1e-9)))
                if d > lim:
                    out += 1
                    rows.append(f"az{az:+.0f} el{el:.0f} v{speed:.1f} crosses "
                                f"{d:.2f} m out, arms stop at {lim:.2f}")
    print(f"reach:   arms take the rig {reach_limit(1.0, 0.0):.2f} m downrange, "
          f"{reach_limit(0.0, 1.0):.2f} m across; {12 - out} of 12 envelope shots "
          f"land inside it")
    for r in rows[:4]:
        print(f"         OUT OF REACH: {r}")
    if rows[4:]:
        print(f"         ... and {len(rows) - 4} more")


check_props()
check_reach()


# ---- state --------------------------------------------------------------------

anchor_from = np.array(p0[anchor_idx_np], np.float64)
anchor_to = anchor_from.copy()

sim_time = 0.0
state = "idle"                # idle -> flight -> catch -> settled / missed
t_fire = None
t_contact = None
v_contact = np.zeros(3)
ball_vy_at_contact = 0.0
t_recover = None
dip_depth = 0.0
carry_vec = np.zeros(3)       # xz walk back to home during recovery
carry_d = 0.0                 # how far the catch ended from home
carry_T = RECOVER             # s the carry takes at CARRY_SPEED
toss_offset = None            # (carry distance, ball-off-centre at the plan)
t_windup = t_launch = t_toss = None
toss_dir = np.array([0.0, 1.0, 0.0])
toss_cmd = 0.0                # m/s the rig is commanded to reach
toss_plan = None              # (T, v0, |v0|, reachable) of the solved throw
toss_release = None           # truth at separation, for the report only
toss_rig_v = np.zeros(3)      # the rig's own peak velocity during the stroke
toss_free = None              # first tick with no contact, pending confirmation
toss_launch_p = None          # where the plan expected the ball to leave from
toss_miss = None


def held(p, v):
    """Is the ball still in the sheet? Distance from the rig centre against the
    sheet's own half-width, at rest, and clear of the floor -- not an absolute
    height, which just measures how far the rig happened to dip."""
    return (math.hypot(p[0] - rig.p[0], p[2] - rig.p[2]) < 0.5 * CLOTH
            and p[1] > 2.5 * BALL_R
            and float(np.linalg.norm(v)) < 0.6)
truth_cross = None            # where the ball really crossed the catch plane
plan = None                   # (t_hit, p_hit, v_hit) from the tracker
first_plan = None             # the earliest usable prediction, for the report
approach = None               # (t0, p0, v0) the approach cubic departs from
tracker = StereoTracker(sensors)
frames_tracked = 0
_prev_ball_y = None
obs_err = []
arm_worst = [0.0]
arm_peak = [0.0, 0.0, 'idle']
bearing_err = [[], []]      # per sensor, in pixels
contact_miss = [0.0]        # how far off centre the ball first touched the sheet


def fire():
    global state, t_fire, tracker, frames_tracked, plan, first_plan
    el, az = math.radians(aim["el"]), math.radians(aim["az"])
    v0 = aim["speed"] * np.array([math.cos(el) * math.cos(az), math.sin(el),
                                  math.cos(el) * math.sin(az)])
    bp.assign(np.array([muzzle()], np.float32))
    bv.assign(np.array([v0], np.float32))
    for s in sensors:
        # The poses never change; only the lock and the reference frame do. The
        # detector runs during flight only, so `prev` has to be dropped here or
        # the first difference of this shot would be against the last shot.
        s.gate, s.last_px, s.prev = None, None, None
    tracker = StereoTracker(sensors)
    frames_tracked, plan, first_plan = 0, None, None
    globals()['approach'] = None
    rig.set_tilt(0.0)
    t_fire, state = sim_time, "flight"
    if EFFECTS:
        puff(muzzle() + aim_dir() * 0.06, aim_dir())


def in_basket(p):
    """Is the ball in the bin? Inside the inner radius and between the bin floor
    and a little above the rim -- the same cylinder the collider uses."""
    return (math.hypot(p[0] - BASKET[0], p[2] - BASKET[1]) < BASKET_R
            and BASKET_FLOOR - 0.02 < p[1] < BASKET_RIM + 0.12)


def plan_toss(p_rest):
    """Solve the throw WITHOUT looking at the ball.

    The launch point is the rig's own centre, which is proprioception rather
    than vision: the rig knows where its four tool frames are, the sheet's
    middle is the average of them, and the ball demonstrably settles there
    (measured 0.19 m off centre on the default shot, and that residual offset is
    an honest error source in the throw, not something to be looked up).

    v0 = (B - p - g T^2 / 2) / T is solved for every T at or above TOSS_T_MIN
    and the CHEAPEST one is taken, because the rig's speed cap is the binding
    constraint at one end -- and the sheet is the binding constraint at the
    other, which is why the search does not simply start at zero. The release
    point moves with the stroke, so this runs three times, feeding its own
    answer back in.
    """
    target = np.array([BASKET[0], BASKET_RIM + 0.05, BASKET[1]])
    launch_p = np.asarray(p_rest, float).copy()
    T, v0, sp = 0.7, np.zeros(3), 0.0
    for _ in range(3):
        best = None
        for T_ in np.arange(TOSS_T_MIN, 1.20, 0.01):
            vv = (target - launch_p - 0.5 * G_NP * T_ * T_) / T_
            s_ = float(np.linalg.norm(vv))
            if best is None or s_ < best[1]:
                best = (T_, s_, vv)
        T, sp, v0 = best
        # Where the ball is when it leaves. The first model said it rides
        # forward with the rig -- cmd * stroke * 2/pi, less the windup dip --
        # and the measurement says it does not: the sheet slides UNDER the ball
        # for most of the stroke and the ball only takes off at the end, so it
        # leaves within 0.09 m of where it sat and 0.03 m above it. That model
        # put the planned launch point 0.285 m from where the ball actually
        # left, and 0.285 m of a 1.35 m throw is most of a basket.
        launch_p = np.asarray(p_rest, float) + np.array([0.0, TOSS_RISE, 0.0])
    globals()['toss_launch_p'] = launch_p
    return T, v0, sp, sp / max(TOSS_GAIN, 1e-3) <= ARM_SPEED + 1e-6


def hermite(p0, v0, p1, v1, T, t):
    """Position and velocity at time t along the cubic that leaves (p0,v0) and
    arrives at (p1,v1) after T.

    The rig RIDES this curve rather than being pushed along it by the
    acceleration at s=0. Applying only a(0) of a plan that is re-solved every
    frame is a lagging controller -- the rig never actually follows the curve it
    just planned, and here it topped out at 1.35 m/s against the 3.05 m/s it
    needed to arrive matched. Riding the curve makes arrival exact by
    construction; the arm's speed limit is then enforced against the curve, so
    an impossible shot shows up as the rig falling behind and missing rather
    than as a controller that quietly cannot track."""
    T = max(T, 1e-3)
    s_ = min(max(t / T, 0.0), 1.0)
    s2, s3 = s_ * s_, s_ * s_ * s_
    h00, h10 = 2 * s3 - 3 * s2 + 1, s3 - 2 * s2 + s_
    h01, h11 = -2 * s3 + 3 * s2, s3 - s2
    g00, g10 = 6 * s2 - 6 * s_, 3 * s2 - 4 * s_ + 1
    g01, g11 = -6 * s2 + 6 * s_, 3 * s2 - 2 * s_
    pos_ = h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1
    vel_ = (g00 * p0 + g10 * T * v0 + g01 * p1 + g11 * T * v1) / T
    return pos_, vel_


def substep(alpha=1.0):
    """One physics substep. `alpha` walks the anchors from where they were at the
    start of this tick to where the arms got them, so a 240 Hz IK update does not
    arrive as a step change the cloth has to absorb."""
    anchor_tgt.assign((anchor_from + (anchor_to - anchor_from) * alpha).astype(np.float32))
    impulse.zero_()
    wp.launch(ball_predict, dim=1, device=device, inputs=[bp, bv, bpred, DT])
    wp.launch(integrate, dim=V, device=device, inputs=[pos, prev, pred, im, DT, DAMPING])
    wp.launch(set_anchors, dim=4, device=device, inputs=[pred, anchor_idx, anchor_tgt])
    if solve_graph is not None:
        wp.capture_launch(solve_graph)
        wp.copy(pos, pred)
    else:
        a, b = pred, scratch
        for _ in range(ITERS):
            wp.launch(solve, dim=V, device=device,
                      inputs=[a, b, im, N, N, REST, bpred, bp, BALL_R + 0.012, MU,
                              prev, anchor_tgt, lra, impulse, M_PARTICLE, 0.35])
            a, b = b, a
        wp.copy(pos, a)
    if state != "idle":
        wp.launch(ball_finish, dim=1, device=device,
                  inputs=[bp, bv, impulse, DT, 1.0 / BALL_KG, BALL_R,
                          float(BASKET[0]), float(BASKET[1]), BASKET_FLOOR,
                          BASKET_RIM, BASKET_R])


def step_frame():
    """Sim, render, read the sensor, re-plan, move the rig. One sensor tick."""
    global sim_time, state, t_contact, v_contact, ball_vy_at_contact
    global t_recover, dip_depth, carry_vec, carry_d, carry_T
    global t_windup, t_launch, t_toss, toss_dir, toss_cmd, toss_plan
    global toss_release, toss_miss, toss_offset, toss_rig_v, toss_free
    global plan, first_plan, frames_tracked, truth_cross, _prev_ball_y

    import time as _t
    _mark = _t.perf_counter()

    def _lap(name):
        nonlocal _mark
        if PROFILE:
            now = _t.perf_counter()
            prof[name] = prof.get(name, 0.0) + (now - _mark) * 1000.0
            _mark = now

    muzzle_effects(None if t_fire is None or state == "idle" else sim_time - t_fire)
    update_aim_preview(state in ("idle", "settled", "missed", "delivered", "adrift"))

    dt = 1.0 / SENSOR_HZ
    if EFFECTS:
        step_smoke(dt)
    global anchor_from, anchor_to
    anchor_from = anchor_to.copy()
    cmd = rig.targets()
    if arms:
        anchor_to = np.array([a.track(cmd[k], dt) for k, a in enumerate(arms)])
    else:
        anchor_to = cmd.astype(np.float64)
    for _i in range(SUBSTEPS):
        substep((_i + 1) / SUBSTEPS)
        sim_time += DT
    _lap("sim")

    p_true = bp.numpy()[0].astype(np.float64)
    v_true = bv.numpy()[0].astype(np.float64)
    contact = float(np.linalg.norm(impulse.numpy()[0])) > 0.0
    if arms:
        e = max(a.err for a in arms)
        arm_worst.append(e)
        if e > arm_peak[0]:
            arm_peak[:] = [e, sim_time, state]
    _lap("readback")

    # Ground truth of the crossing, recorded once, purely so the prediction can
    # be scored afterwards. Nothing in the control path reads it.
    if state == "flight" and _prev_ball_y is not None:
        if _prev_ball_y > CATCH_Y >= p_true[1] and truth_cross is None:
            truth_cross = (sim_time, p_true.copy(), v_true.copy())
    _prev_ball_y = p_true[1]

    # --- meshes ---------------------------------------------------------------
    wp.launch(compute_normals, dim=V, device=device, inputs=[pos, nrm, N, N])
    geometry.update_attribute("position", pos.numpy())
    geometry.update_attribute("normal", nrm.numpy())
    ball_mesh.visible = state != "idle"      # loaded in the barrel until fired
    ball_mesh.position.set(*[float(x) for x in p_true])
    for k, m in enumerate(anchor_meshes):
        m.position.set(float(anchor_to[k][0]), float(anchor_to[k][1]), float(anchor_to[k][2]))
        m.visible = not arms          # the tool link IS the gripper once arms exist
    _lap("mesh upload")

    # --- render, then read both sensors --------------------------------------
    # One render() produces the primary AND both views from one scene build, so
    # the two frames read back below are the SAME simulated instant. That is
    # what makes intersecting their bearings legitimate.
    renderer.render(scene, camera)
    _lap("render")
    frames = [renderer.read_view_rgb_pixels(s.handle) for s in sensors]
    _lap("sensor read")

    # --- perceive and plan ----------------------------------------------------
    # Stop measuring just before arrival. In the last ~80 ms the ball and the
    # sheet occupy the same pixels in both cameras, and a difference detector
    # cannot tell which of two overlapping movers it is centred on -- measured,
    # the tail frames put the bearing 80-95 px out at the 90th percentile and
    # dragged the fitted arc 0.36 m low. Nothing is lost by dropping them: by
    # then the plan is 200 observations old and the rig is already committed.
    # The cannon's exclusion disc, projected per sensor. Its world radius is
    # fixed, so the pixel radius is just f * R / depth -- and both are constants
    # of the rig, not of the shot.
    for s in sensors:
        s.mask = None
        if EFFECTS and state == "flight" and sim_time - t_fire < MUZZLE_MASK_S:
            uv = s.project(CANNON)
            if uv is not None:
                depth = max(float(np.dot(CANNON - s.eye, s.fwd)), 0.05)
                # Wide while the flash is lighting the floor, tight afterwards
                # for the recoil and the birth of the smoke.
                r = (FLASH_MASK_R if sim_time - t_fire < FLASH_S + 2.0 / SENSOR_HZ
                     else MUZZLE_MASK_R)
                s.mask = (uv[0], uv[1], s.f * r / depth)

    blind = plan is not None and plan[0] - sim_time < 0.08
    if state == "flight" and not blind:
        if ORACLE:
            tracker.obs.append((sim_time, p_true[0], p_true[1], p_true[2]))
            frames_tracked += 1
        else:
            est = tracker.observe(frames, sim_time)
            if est is not None:
                frames_tracked += 1
                obs_err.append(est - p_true)
                # Score each camera's BEARING separately against where the ball
                # truly projects. A stereo estimate can be wrong two ways --
                # a bad bearing, or a good pair intersected badly -- and the
                # ray gap already reports the second.
                for k, s in enumerate(sensors):
                    lp, tuv = s.last_px, s.project(p_true)
                    if lp is not None and tuv is not None:
                        bearing_err[k].append(math.hypot(lp[0] - tuv[0], lp[1] - tuv[1]))
        fitted = tracker.solve_fit()
        if fitted is not None:
            hit = tracker.predict_crossing(CATCH_Y)
            if hit is not None:
                plan = hit
                if first_plan is None:
                    first_plan = (sim_time, hit)

    _lap("track + fit")

    # --- control --------------------------------------------------------------
    if state == "flight" and plan is not None:
        global approach
        t_hit, p_hit, v_hit = plan
        if approach is None:
            approach = (sim_time, rig.p.copy(), rig.v.copy())
        t0, p_start, v_start = approach
        target_p = np.array([p_hit[0], CATCH_Y, p_hit[2]])
        target_v = np.array([v_hit[0], 0.0, v_hit[2]])
        p_cmd, v_cmd = hermite(p_start, v_start, target_p, target_v,
                               t_hit - t0, sim_time - t0)
        rig.goto(dt, p_cmd, v_cmd)
        # Ramp the scoop in over the last TILT_LEAD seconds of the approach,
        # along the FITTED horizontal velocity -- the same arc the rig is
        # chasing, not the ball's true motion. Early in the track the fit is
        # still moving, and a sheet that pitches about while the arms are also
        # sprinting is the worst of both; by the time the tilt matters the
        # direction has been stable for a hundred observations.
        u_t = (t_hit - sim_time) / max(TILT_LEAD, 1e-3)
        rig.set_tilt(math.radians(TILT_DEG) * min(max(1.0 - u_t, 0.0), 1.0),
                     np.array([v_hit[0], 0.0, v_hit[2]]))
    elif state == "catch":
        u = min((sim_time - t_contact) / ABSORB, 1.0)
        # Ride with the ball, then bleed off. Zero relative velocity at first
        # contact is what stops it skipping straight back out of the sheet.
        v_des = v_contact * (1.0 - u)
        v_des[1] = -dip_depth * (math.pi / ABSORB) * 0.5 * math.sin(math.pi * u)
        rig.step(dt, (v_des - rig.v) / dt)
    elif state == "recover":
        # Cradle, lift back to the ready height, AND CARRY THE BALL HOME.
        #
        # The lift is what a person does, and without it the sheet stays parked
        # at the bottom of its give with the ball sitting in a pit. The carry is
        # what makes the throw possible at all: the catch ends wherever the ball
        # was, and on the az +8 shot that was 0.83 m from home with the ball
        # near the edge of the sheet, so a stroke planned from the rig's own
        # centre was accelerating something that was not on it -- the ball left
        # at 0.17 of the commanded speed, 145 degrees off plan. Walking back to
        # home at a cradling speed re-centres the ball BY CONSTRUCTION rather
        # than by looking at it: the bowl's low point is the rig centre, so the
        # ball rolls to the middle while the sheet re-tensions under it. The
        # carry lasts as long as the distance needs at CARRY_SPEED; the lift and
        # the re-tension still take RECOVER.
        u = min((sim_time - t_recover) / RECOVER, 1.0)
        uc = min((sim_time - t_recover) / carry_T, 1.0)
        v_des = carry_vec * ((math.pi / (2.0 * carry_T)) * math.sin(math.pi * uc))
        v_des[1] = dip_depth * (math.pi / RECOVER) * 0.5 * math.sin(math.pi * u)
        rig.spread = PRESENT * 0.5 * (1.0 - math.cos(math.pi * u))
        # Flat again before the carry gets going: a tilted sheet being walked
        # sideways tips the ball out of the low corner.
        rig.set_tilt(math.radians(TILT_DEG)
                     * max(0.0, 1.0 - (sim_time - t_recover) / max(TILT_EASE, 1e-3)))
        rig.step(dt, (v_des - rig.v) / dt)
    elif state == "windup":
        # Pull the sheet taut and drop the rig back along -v0. The sine profile
        # integrates to exactly TOSS_DIP and ends at zero velocity, so the
        # stroke starts from rest rather than from whatever the dip left behind.
        u = min((sim_time - t_windup) / TOSS_WINDUP, 1.0)
        rig.spread = PRESENT + (TOSS_SPREAD - PRESENT) * 0.5 * (1.0 - math.cos(math.pi * u))
        v_des = -toss_dir * TOSS_DIP * (math.pi / TOSS_WINDUP) * 0.5 * math.sin(math.pi * u)
        rig.step(dt, (v_des - rig.v) / dt)
    elif state == "launch":
        # Accelerate along v0 and arrive at the commanded speed exactly at the
        # end of the stroke. Nothing is done to the ball: it is carried by the
        # sheet and leaves when the contact impulse stops, which is the same
        # contact model that caught it.
        u = min((sim_time - t_launch) / TOSS_STROKE, 1.0)
        rig.step(dt, (toss_dir * toss_cmd * math.sin(0.5 * math.pi * u) - rig.v) / dt)
    elif state == "toss":
        u = min((sim_time - t_toss) / TOSS_BRAKE, 1.0)
        if u < 1.0:
            rig.step(dt, (toss_dir * toss_cmd * (1.0 - u) - rig.v) / dt)
        else:
            # Get the sheet out from under the ball. The rig ends the stroke
            # high, moving at nearly the ball's own speed, and if it just stops
            # there the ball falls straight back into it -- measured, the first
            # throws that reached the right speed still ended with the ball at
            # rest in the sheet at y=1.31. Withdrawing to the ready pose is the
            # follow-through, and it is what makes the throw a throw.
            home3 = np.array([HOME[0], CATCH_Y, HOME[1]])
            v_des = (home3 - rig.p) * 3.0
            sp_ = float(np.linalg.norm(v_des))
            if sp_ > 2.2:
                v_des *= 2.2 / sp_
            rig.spread = max(rig.spread - 2.0 * dt, 0.0)
            rig.step(dt, (v_des - rig.v) / dt)
    else:
        rig.set_tilt(0.0)
        rig.step(dt, -rig.v / dt)

    if state == "flight" and contact:
        state = "catch"
        t_contact = sim_time
        v_contact = np.array([rig.v[0], 0.0, rig.v[2]])
        contact_miss[0] = math.hypot(p_true[0] - rig.p[0], p_true[2] - rig.p[2])
        ball_vy_at_contact = float(v_true[1])
        # Give the ball room proportional to how hard it arrived, up to the
        # stroke the arms actually have.
        dip_depth = min(abs(ball_vy_at_contact) * ABSORB / math.pi, GIVE)
    elif state == "catch" and sim_time - t_contact > ABSORB:
        state, t_recover = "recover", sim_time
        # Everything the carry needs, fixed once at the moment it starts. The
        # rig's own pose is proprioception; the ball is not consulted.
        carry_vec = np.array([HOME[0] - rig.p[0], 0.0, HOME[1] - rig.p[2]])
        carry_d = float(np.linalg.norm(carry_vec))
        carry_T = max(RECOVER, carry_d * math.pi / (2.0 * max(CARRY_SPEED, 1e-3)))
    elif state == "recover" and sim_time - t_recover > carry_T + SETTLE_DWELL:
        # One instant is a coin flip on a damped oscillation. Measured on the
        # az +8 shot: the ball was 0.11 m off the sheet centre and never left
        # it, but happened to be swinging at 0.19 m/s when the clock said now,
        # and got reported MISSED -- at 4.2 s it was sitting still in exactly
        # the same place. So the verdict waits for the motion to die, and the
        # deadline (not the sample) is what makes a real miss a miss: a ball on
        # the floor or out the side fails the height and distance tests
        # permanently, so waiting cannot rescue one.
        if held(p_true, v_true):
            if TOSS:
                # Plan from the rig's own centre and commit. Everything after
                # this is open loop against a plan made from proprioception --
                # the tracker is not consulted, and neither is the ball.
                T, v0, sp, ok = plan_toss(rig.p)
                toss_plan = (T, v0, sp, ok)
                toss_dir = v0 / max(sp, 1e-9)
                toss_cmd = min(sp / max(TOSS_GAIN, 1e-3), ARM_SPEED)
                # Truth, for the report alone: how far the ball actually is from
                # the point the throw was just planned from. That residual is
                # the throw's own error source, and the carry exists to shrink
                # it -- so it has to be printed, not assumed.
                toss_offset = (carry_d,
                               math.hypot(p_true[0] - rig.p[0], p_true[2] - rig.p[2]))
                state, t_windup = "windup", sim_time
            else:
                state = "settled"
        elif sim_time - t_recover > carry_T + SETTLE_DWELL + 0.60:
            state = "missed"
    elif state == "windup" and sim_time - t_windup > TOSS_WINDUP:
        state, t_launch = "launch", sim_time
    elif state == "launch" and sim_time - t_launch > TOSS_STROKE:
        state, t_toss = "toss", sim_time
    elif state == "toss" and sim_time - t_toss > 0.45:
        settled_ = float(np.linalg.norm(v_true)) < 0.35
        if settled_ or sim_time - t_toss > 2.4:
            toss_miss = math.hypot(p_true[0] - BASKET[0], p_true[2] - BASKET[1])
            state = "delivered" if in_basket(p_true) else "adrift"

    # The ball LEAVES when the contact impulse stops, and that is not the top of
    # the stroke -- which is what the throw was being scored against, and it was
    # scoring the wrong instant. Once the ball rides in the middle of the bowl
    # (which is what the carry home is for) the sheet keeps hold of it through
    # the brake, the long-range attachment snaps taut on the way past, and the
    # ball leaves LATER and much faster: measured 2.60 m/s at the top of a 3.34
    # m/s stroke, and 8.7 m/s vertical by the time it was actually free. So the
    # release is recorded at separation. Truth, and for the report only.
    if state in ("launch", "toss") and toss_release is None and t_launch is not None:
        if contact:
            # Contact flickers on and off through a stroke, so a single free
            # tick is not a release; only 80 ms of daylight is.
            toss_free = None
            if float(np.linalg.norm(rig.v)) > float(np.linalg.norm(toss_rig_v)):
                toss_rig_v = rig.v.copy()
        elif toss_free is None:
            toss_free = (sim_time, p_true.copy(), v_true.copy())
        elif sim_time - toss_free[0] > 0.08:
            toss_release = toss_free + (toss_rig_v.copy(),)

    _lap("control")
    if TRACE and state != "idle":
        pe = "  -" if plan is None else f"{plan[1][0]:+.2f}"
        oe = obs_err[-1] if obs_err else np.zeros(3)
        gap = tracker.gap[-1] if tracker.gap else float('nan')
        cams = ""
        for s in sensors:
            lp, tuv = s.last_px, s.project(p_true)
            cams += (f" | {s.name} ({lp[0]:5.1f},{lp[1]:5.1f}) n={lp[3]:4d} "
                     f"true ({tuv[0]:5.1f},{tuv[1]:5.1f})"
                     if lp and tuv else f" | {s.name} -")
        print(f"t={sim_time:6.3f} {state:8s} obs={len(tracker.obs):4d} "
              f"pred_x={pe} rig_x={rig.p[0]:+.2f} ball=({p_true[0]:+.2f},{p_true[1]:.2f}) "
              f"obs_err=({oe[0]:+.3f},{oe[1]:+.3f},{oe[2]:+.3f}) gap={gap:.3f}" + cams)
    return p_true, v_true


def report(p_true, v_true):
    src = ("GROUND TRUTH (--oracle)" if ORACLE else
           f"2 x {SENSOR_W}x{SENSOR_H} pinhole, {SUBTENDED:.0f} deg apart")
    print(f"\n  sensor      {src} at {SENSOR_HZ:.0f} Hz")
    print(f"  sheet       {V} particles, {SUBSTEPS} substeps x {ITERS} iters per tick")
    print(f"  shot        az {aim['az']:+.1f} deg, el {aim['el']:.1f} deg, "
          f"{aim['speed']:.2f} m/s")
    for k, s in enumerate(sensors):
        be = np.array(bearing_err[k])
        if be.size:
            print(f"  bearing {s.name:<4s} median {np.median(be):.2f} px, "
                  f"90th {np.percentile(be, 90):.2f} px"
                  f"   (= {np.median(be) / s.f * 1000:.2f} mrad)")
    if tracker.gap:
        g = np.array(tracker.gap)
        print(f"  stereo gap  median {np.median(g):.3f} m, 90th {np.percentile(g, 90):.3f} m "
              f"(rays rejected above {STEREO_GAP:.2f})")
    if obs_err:
        e = np.array(obs_err)
        print(f"  obs error   mean ({e[:,0].mean():+.3f},{e[:,1].mean():+.3f},"
              f"{e[:,2].mean():+.3f}) m, |err| median {np.median(np.linalg.norm(e,axis=1)):.3f} m")
    print(f"  tracked     {frames_tracked} frames, {len(tracker.obs)} observations, "
          f"{tracker.rejected} pairs rejected, "
          + " / ".join(f"{s.rejected} {s.name}" for s in sensors) + " blobs rejected")
    if truth_cross is None:
        # The sheet was in the way ABOVE the plane, so there is no unobstructed
        # crossing to score the prediction against. Still worth saying what the
        # rig did -- a catch that happened early is not a failure to report.
        print("  truth       no clean crossing: the sheet met the ball above "
              f"y={CATCH_Y:.2f}")
    else:
        tt, tp_, tv = truth_cross
        print(f"  truth       crossed y={CATCH_Y:.2f} at x{tp_[0]:+.3f} z{tp_[2]:+.3f}, "
              f"t={tt:.3f} s, horizontal {math.hypot(tv[0], tv[2]):.2f} m/s")
        if first_plan is not None:
            t_at, (th, ph, vh) = first_plan
            print(f"  first call  at t={t_at:.3f} s ({1000 * (t_at - t_fire):.0f} ms after "
                  f"firing): x{ph[0]:+.3f} z{ph[2]:+.3f}, err "
                  f"{np.linalg.norm(ph - tp_):.3f} m")
        if plan is not None:
            th, ph, vh = plan
            print(f"  final call  x{ph[0]:+.3f} z{ph[2]:+.3f}, "
                  f"err {np.linalg.norm(ph - tp_):.3f} m, timing err "
                  f"{1000 * (th - tt):+.0f} ms")
    if arms:
        print(f"  arms        4 x {arms[0].kind}, worst IK error this "
              f"run {max(arm_worst):.4f} m at t={arm_peak[1]:.2f} s ({arm_peak[2]})"
              + ("  (an arm could not hold its corner)" if max(arm_worst) > 0.10 else
                 "  (lag at peak speed, within the joint limits)" if max(arm_worst) > 0.01 else ""))
    print(f"  rig         travelled {rig.travel:.2f} m, peak {rig.peak_speed:.2f} m/s "
          f"(cap {ARM_SPEED:.1f})"
          + (f", SPEED-CAPPED on {rig.starved} frames" if rig.starved else "")
          + (f", HELD AT THE ARMS' REACH BOUNDARY on {rig.reach_capped} frames"
             if rig.reach_capped else ""))
    if t_contact is not None:
        print(f"  contact     t={t_contact:.3f} s, ball {contact_miss[0]:.3f} m from the "
              f"sheet centre when it first touched")
    if rig.peak_tilt > 0.0:
        print(f"  scoop       sheet tilted to {math.degrees(rig.peak_tilt):.0f} deg "
              f"on arrival (corners +/- {0.5 * SPAN * math.tan(rig.peak_tilt):.3f} m)")
    if toss_offset is not None:
        print(f"  carry home  catch ended {toss_offset[0]:.3f} m from home, carried "
              f"back in {carry_T:.2f} s at <= {CARRY_SPEED:.2f} m/s; ball then "
              f"{toss_offset[1]:.3f} m off the point the throw was planned from")
    if toss_plan is not None:
        T, v0, sp, ok = toss_plan
        print(f"  throw plan  basket at x{BASKET[0]:+.2f} z{BASKET[1]:+.2f}, "
              f"{np.linalg.norm(np.array([BASKET[0] - HOME[0], BASKET[1] - HOME[1]])):.2f} m "
              f"from rig home; T={T:.2f} s needs {sp:.2f} m/s, "
              f"commanded {toss_cmd:.2f} (cap {ARM_SPEED:.1f})"
              + ("" if ok else "   [BEYOND THE STROKE - clamped]"))
    if toss_release is not None and toss_plan is not None:
        tr, _pr, vr, rv = toss_release
        if toss_launch_p is not None:
            print(f"  launch pt   planned x{toss_launch_p[0]:+.2f} y{toss_launch_p[1]:.2f} "
                  f"z{toss_launch_p[2]:+.2f}, ball actually left from "
                  f"x{_pr[0]:+.2f} y{_pr[1]:.2f} z{_pr[2]:+.2f} "
                  f"({np.linalg.norm(_pr - toss_launch_p):.3f} m away)")
        _, v0, sp, _ = toss_plan
        vb, vg = float(np.linalg.norm(vr)), float(np.linalg.norm(rv))
        cosang = float(np.dot(vr, v0)) / max(vb * sp, 1e-9)
        off = math.degrees(math.acos(max(-1.0, min(1.0, cosang))))
        print(f"  release     t={tr:.3f} s (separation), rig {vg:.2f} m/s, ball {vb:.2f} m/s "
              f"({vb / max(vg, 1e-6):.2f} x the rig, gain assumed {TOSS_GAIN:.2f}), "
              f"wanted {sp:.2f} m/s at {off:.0f} deg off the planned direction")
    resting = float(np.linalg.norm(v_true))
    verdict = {"settled": "CAUGHT", "missed": "MISSED", "catch": "still absorbing",
               "recover": "caught, still lifting", "windup": "caught, winding up",
               "launch": "caught, mid-throw", "toss": "thrown, still in the air",
               "delivered": "CAUGHT and DELIVERED",
               "adrift": "CAUGHT, throw missed the basket",
               "flight": "NEVER CONTACTED", "idle": "never fired"}.get(state, state)
    dx = math.hypot(p_true[0] - rig.p[0], p_true[2] - rig.p[2])
    if state in ("delivered", "adrift", "toss") or toss_miss is not None:
        d = (toss_miss if toss_miss is not None
             else math.hypot(p_true[0] - BASKET[0], p_true[2] - BASKET[1]))
        print(f"  RESULT      {verdict} -- ball {d:.3f} m from the basket axis "
              f"(rim radius {BASKET_R:.2f}), y={p_true[1]:.2f}, {resting:.2f} m/s")
    else:
        print(f"  RESULT      {verdict} -- ball {dx:.3f} m off the sheet centre "
              f"(half-width {0.5 * CLOTH:.2f}), y={p_true[1]:.2f}, {resting:.2f} m/s")


# ---- run ----------------------------------------------------------------------

TOTAL = int(SECONDS * SENSOR_HZ)


def show_sensor_pip(on):
    """Picture-in-picture of both sensor feeds, bottom corners. Returns the
    framebuffer size the rects were laid out for, so the caller can notice when
    it changes.

    set_view_display_rect is a single image copy inside the frame's own command
    buffer -- already resolved, already on the device, already in the
    swapchain's format -- so this costs no readback, no upload and no second
    submission. It is 1:1 ONLY: the rect must be exactly the size the view was
    added at, which is why the sensor resolution and the inset size are the same
    number. A mismatch draws nothing rather than a filtered rescale, so on a
    resize only x and y may move.

    The size comes from renderer.size(), which reports the FRAMEBUFFER, not the
    size the canvas was asked for: Canvas.size() is a request, and a window
    manager or a display scale is free to hand back something else. Laying the
    insets out against the requested size put them in the wrong corner on the
    first frame, before anyone had resized anything.
    """
    if not on:
        for s in sensors:
            renderer.hide_view(s.handle)
        return None
    w, h = renderer.size()
    m = 18
    if w >= 2 * SENSOR_W + 3 * m:
        rects = [(m, h - SENSOR_H - m), (w - SENSOR_W - m, h - SENSOR_H - m)]
    else:
        # Too narrow for both along the bottom, so stack them up the left edge
        # rather than let the second one slide off the right.
        rects = [(m, h - 2 * SENSOR_H - 2 * m), (m, h - SENSOR_H - m)]
    for s, (x, y) in zip(sensors, rects):
        renderer.set_view_display_rect(s.handle, max(0, x), max(0, y),
                                       SENSOR_W, SENSOR_H)
    return w, h


if TUNE:
    import time
    print(f"cloth catch: {V} particles on {device}, {SENSOR_HZ:.0f} Hz sensor tick, "
          f"render scale {RENDER_SCALE:.2f}")
    pt = vt = None
    for i in range(TOTAL):
        if state == "idle" and sim_time >= FIRE_AT:
            fire()
        if i == 60:                        # warm up before timing
            t0 = time.perf_counter()
        pt, vt = step_frame()
    loop_ms = 1000.0 * (time.perf_counter() - t0) / (TOTAL - 60)
    report(pt, vt)
    print(f"  loop        {loop_ms:.2f} ms/tick ({1000.0 / loop_ms:.0f} Hz achievable, "
          f"{SENSOR_HZ:.0f} Hz asked)")
    if PROFILE:
        n = TOTAL
        for k, v in sorted(prof.items(), key=lambda kv: -kv[1]):
            print(f"    {k:<12} {v / n:6.2f} ms/tick  ({100 * v / (loop_ms * n):4.1f}%)")
    sys.exit(0)

if (CLIP or SEQ) and not FRAMES:
    import shutil
    import tempfile
    from PIL import Image
    outdir = SEQ or os.path.join(tempfile.mkdtemp(prefix="clothcatch_"), "f")
    os.makedirs(os.path.dirname(outdir) or ".", exist_ok=True)
    every = max(1, int(round(SENSOR_HZ / CLIP_FPS)))
    written = 0
    pt = vt = None
    show_sensor_pip(SPLIT)
    for i in range(TOTAL):
        if state == "idle" and sim_time >= FIRE_AT:
            fire()
        pt, vt = step_frame()
        # Real time on the approach, then every tick through the catch -- the sim
        # runs at SENSOR_HZ and the clip plays at CLIP_FPS, so keeping every tick
        # is free slow motion at exactly that ratio. And nothing before the shot
        # is worth watching: the settle is a requirement, not a beat.
        # Slow motion for the two beats that are actually about the rig: the
        # catch, and the throw that follows it. Everything between them (the
        # lift, the windup) plays at speed, so the clip does not sag.
        slowmo = FILM and (state in ("catch", "recover", "launch")
                           or (plan is not None and 0 < plan[0] - sim_time < 0.22)
                           or (state == "toss" and sim_time - t_toss < 0.30))
        if FILM and sim_time < FIRE_AT - 0.35:
            keep = False
        else:
            keep = slowmo or (i % every == 0)
        if keep:
            Image.fromarray(renderer.read_pixels()).save(f"{outdir}_{written:04d}.png")
            written += 1
    report(pt, vt)
    print(f"  wrote {written} frames at {outdir}_*.png")
    if CLIP:
        ff = find_ffmpeg()
        if ff is None:
            print("  no ffmpeg on PATH; frames kept")
        else:
            subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", f"{CLIP_FPS:.0f}",
                            "-i", f"{outdir}_%04d.png", "-an", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                            "-movflags", "+faststart", CLIP], check=True)
            print(f"  wrote {CLIP}")
            if not SEQ:
                shutil.rmtree(os.path.dirname(outdir), ignore_errors=True)
    sys.exit(0)

# --- interactive -----------------------------------------------------------------
# The one thing a headless run never exercises: the sheet must be QUIET before a
# shot. The detector reports change, so a sheet still ringing fills the frame
# with changed pixels and the tracker locks onto cloth rather than ball. The old
# reset() slammed the cloth back to its domed initial state and fired on the same
# frame, which is why interactive shots tracked to the wrong place while headless
# ones (which settle for FIRE_AT seconds first) were fine. Nothing here resets the
# cloth any more: it is left hanging where it is, and firing is gated on measured
# quiescence instead of on a hopeful delay.
#
# The VIEW camera used to be gated the same way, because it WAS the sensor. It is
# not any more: the sensors are two fixed cameras of their own, so the orbit
# camera may be flown around mid-flight and nothing measured changes. That gate
# and its "tracking degraded" warning are gone.
#
# Secondary views also run no overlay pass at all, so the aim arc, the amber and
# cyan rings and ImGui are STRUCTURALLY incapable of reaching a sensor frame --
# they are unlit/transparent and land in a pass the views never run. The debug
# picture-in-picture cannot perturb the measurement it is showing.
controls = tp.OrbitControls(camera, canvas)
controls.enable_damping = True
controls.target.set(*[float(x) for x in TGT])
ui = tp.ImguiContext(canvas, renderer)

QUIET_SHEET = cli_arg("--quiet-sheet", 0.08, float)   # m/s, max particle speed

_prev_pos = None
sheet_speed = 1e9
last_result = ""
pip_size = show_sensor_pip(True)


# Where the tracker currently thinks the ball will cross the catch plane. A flat
# reticle lying IN that plane, not a sphere -- a sphere out in the scene just
# reads as a second ball to catch.
_ring_mat = tp.MeshBasicMaterial()
_ring_mat.color = 0x2ad4ff
_ring_mat.transparent = True
_ring_mat.opacity = 0.75
_ring_mat.side = tp.Side.Double
aim_mark = tp.Mesh(tp.RingGeometry(0.17, 0.21, 40), _ring_mat)
aim_mark.rotate_x(-math.pi / 2)
aim_mark.visible = False
scene.add(aim_mark)


# --- input ----------------------------------------------------------------------
_held = {}


def pressed(k):
    now = canvas.is_key_down(k)
    fired = now and not _held.get(k, False)
    _held[k] = now
    return fired


def ready():
    # A quiet SHEET, and nothing about the camera: the sensors do not move, so
    # where the view camera happens to be is no longer a precondition for a shot.
    return (state in ("idle", "settled", "missed", "delivered", "adrift")
            and sheet_speed < QUIET_SHEET)


def arm():
    """The cloth is deliberately NOT reset -- it is already hanging quiet, and
    resetting it is what broke this path."""
    global last_result, toss_plan, toss_release, toss_free, toss_miss
    global toss_offset, toss_launch_p, toss_rig_v, truth_cross, t_contact
    bp.assign(np.array([muzzle()], np.float32))
    bv.zero_()
    rig.__init__()
    last_result = ""
    toss_plan = toss_release = toss_free = toss_miss = None
    toss_offset = toss_launch_p = truth_cross = t_contact = None
    toss_rig_v = np.zeros(3)
    fire()


def draw_ui():
    tp.imgui.set_next_window_pos(14, 14)
    tp.imgui.set_next_window_size(420, 0)
    tp.imgui.begin("cloth catch")
    tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps   2 x {SENSOR_W}x{SENSOR_H} "
                  f"sensors at {SENSOR_HZ:.0f} Hz")
    tp.imgui.separator()
    tp.imgui.text(f"azimuth    {aim['az']:+6.1f} deg      A / D")
    tp.imgui.text(f"elevation  {aim['el']:6.1f} deg      W / S")
    tp.imgui.text(f"muzzle     {aim['speed']:6.2f} m/s      Q / E")
    tp.imgui.separator()
    if state == "flight":
        n = len(tracker.obs)
        if plan is None:
            tp.imgui.text(f"TRACKING   {n} obs, no confident fit yet")
        else:
            tp.imgui.text(f"TRACKING   {n} obs")
            tp.imgui.text(f"intercept  x {plan[1][0]:+.2f}  z {plan[1][2]:+.2f}"
                          f"   in {max(plan[0] - sim_time, 0.0) * 1000:.0f} ms")
        if tracker.gap:
            tp.imgui.text(f"stereo gap {tracker.gap[-1] * 1000:.0f} mm")
    elif ready():
        tp.imgui.text("READY - SPACE to fire")
    else:
        why = []
        if sheet_speed >= QUIET_SHEET:
            why.append(f"sheet settling ({sheet_speed:.2f} m/s)")
        if state in ("catch", "recover"):
            why.append("catching")
        tp.imgui.text("WAIT - " + ", ".join(why or ["..."]))
        tp.imgui.text("frame differencing needs one moving thing, not two")
    if last_result:
        tp.imgui.separator()
        tp.imgui.text(last_result)
    tp.imgui.separator()
    tp.imgui.text("mouse orbits freely, mid-flight too - R resets")
    tp.imgui.end()


def animate():
    global _prev_pos, sheet_speed, last_result, state, pip_size

    # The insets are placed in WINDOW PIXELS, so a resize leaves them at the old
    # coordinates: wrong corner, and off the edge (where they are clipped) if the
    # window shrank. The rect cannot be rescaled -- it is 1:1 or it draws nothing
    # -- so the answer is to re-issue it with new x and y, and only when the
    # framebuffer actually changed size. The camera has to be told as well, or
    # the frame stretches.
    now_size = renderer.size()
    if now_size != pip_size:
        pip_size = show_sensor_pip(True)
        camera.aspect = now_size[0] / max(now_size[1], 1)
        camera.update_projection_matrix()

    if canvas.is_key_down("A"):
        aim["az"] -= 0.6
    if canvas.is_key_down("D"):
        aim["az"] += 0.6
    if canvas.is_key_down("W"):
        aim["el"] = min(aim["el"] + 0.4, 85.0)
    if canvas.is_key_down("S"):
        aim["el"] = max(aim["el"] - 0.4, 10.0)
    if canvas.is_key_down("Q"):
        aim["speed"] = max(aim["speed"] - 0.03, 2.0)
    if canvas.is_key_down("E"):
        aim["speed"] = min(aim["speed"] + 0.03, 12.0)


    if (pressed("SPACE") or (AUTOFIRE and state in ("idle",))) and ready():
        arm()
    if pressed("R"):
        bp.assign(np.array([muzzle()], np.float32))
        bv.zero_()
        rig.__init__()
        state, last_result = "idle", ""

    controls.update()
    aim_mark.visible = plan is not None and state == "flight"
    if aim_mark.visible:
        aim_mark.position.set(float(plan[1][0]), CATCH_Y, float(plan[1][2]))

    _DONE = ("settled", "missed", "delivered", "adrift")
    was = state
    p_true, v_true = step_frame()      # the render both sensors are sampled from
    if state in _DONE and was not in _DONE:
        if state in ("delivered", "adrift"):
            d = math.hypot(p_true[0] - BASKET[0], p_true[2] - BASKET[1])
            last_result = ("CAUGHT + DELIVERED" if state == "delivered"
                           else "CAUGHT, throw missed") + f" - {d:.2f} m from the basket"
        else:
            dx = math.hypot(p_true[0] - rig.p[0], p_true[2] - rig.p[2])
            last_result = ("CAUGHT" if state == "settled" else "MISSED") + \
                          f" - ball {dx:.2f} m off the sheet centre"

    now = pos.numpy()
    if _prev_pos is not None and _prev_pos.shape == now.shape:
        sheet_speed = float(np.abs(now - _prev_pos).max()) * SENSOR_HZ
    _prev_pos = now.copy()
    # No debug texture to build: the two sensor feeds are composited by the GPU
    # inside the frame's own command buffer (show_sensor_pip), so the panel that
    # used to cost a readback, an overlay draw and a texture upload per frame
    # now costs nothing per frame at all.
    ui.render(draw_ui)


print("cloth catch:  A/D aim   W/S elevation   Q/E power   SPACE fire   R reset"
      "   mouse orbits")

if FRAMES:
    from PIL import Image
    n = 0
    while n < FRAMES and canvas.animate_once(animate):
        n += 1
        if SEQ and n % 20 == 0:
            Image.fromarray(renderer.read_pixels()).save(f"{SEQ}_{n:05d}.png")
    print(f"  ran {n} interactive frames, state={state}, "
          f"sheet {sheet_speed:.3f} m/s, {len(tracker.obs)} observations")
    print(f"  {last_result or 'no shot completed'}")
else:
    canvas.animate(animate)
