"""Differentiable hull sculpting: gradient descent designs a hull, live in the water.

The optimisation variable IS the rendered mesh. A Warp tape runs backward
through an analytic naval-architecture objective -- drag, righting arms,
displacement, fairness -- straight onto the 2562 vertices of an icosphere, Adam
takes a step, and the new positions are written into the Vulkan renderer's own
vertex buffers with `enable_vertex_interop`. Nothing crosses the bus. What you
watch is not a playback of a solved shape: it is the descent, at the rate the
GPU can do it, floating at whatever waterline the objective currently believes
in.

The renderer is the VIEWPORT, not part of the graph. There is no pixel loss and
no differentiable rasteriser here; the gradients come from the integrals below
and the picture is a consequence.

ACT 1 -- SCULPT.
  A blunt ellipsoid blob, 7.2 m x 4.8 m deep x 4.0 m beam, uniform density, half
  submerged. It is deeper than it is wide, which makes it genuinely unstable
  (GM < 0), and its bow is a wall. Loss:

    L = w_drag L_drag + w_stab L_stab + w_disp L_disp + w_reg L_reg + w_box L_box

  * SOFT WATERLINE. Every buoyancy quantity is a face-integral weighted by a
    smooth submerged indicator w(p) = sigmoid(-(n.p - d) / eps) evaluated at the
    face centroid, with `n` the body-frame direction of world-up and `d` the
    waterplane offset. eps is about a face width, so a face crossing the surface
    contributes partially and the derivative of the waterline position with
    respect to the vertices is not a step function. Volume and centroid come
    from the divergence-theorem tet sum taken about a point ON the waterplane
    (q = p - d n): the cone over the waterplane disc is then degenerate and the
    weighted sum is the submerged volume, not the submerged volume plus a wedge.
  * L_drag. Newton pressure-drag proxy along +x: sum over front-facing faces of
    max(0, n.v)^2 * area, weighted by the upright submerged indicator, so it is
    a HULL that is being faired and not an aircraft. It is a proxy and the demo
    says so -- no CFD, no panel method, no wave-making term.
  * L_stab. Righting arm at heel. Gravity is rotated, not the mesh: for each
    heel angle the waterplane normal becomes n(t) = (0, cos t, -sin t) and the
    horizontal transverse direction becomes b(t) = (0, sin t, cos t). The draft
    d(t) is re-solved (outside the tape) so the hull still displaces its own
    weight, then GZ(t) = (B(t) - G) . b(t), with B the soft-weighted centre of
    buoyancy and G the centroid of the whole solid at uniform density. Sampled
    at +/-15, +/-30 and +/-45 degrees -- both signs, or the optimiser finds a
    hull that is only stable to port. L_stab = -mean GZ.
  * L_disp. The hull must keep its VOLUME: (V_total / V0 - 1)^2. See "As built"
    in plans/hull-sculpt-diffsim.md for why the constraint sits on the total
    volume rather than on the submerged one.
  * L_reg. Uniform Laplacian plus edge-length preservation. This is the only
    thing standing between a per-vertex optimiser and a wrinkled mess.
  * L_box. Soft half-extent box. Without it -Sum GZ has no minimum: the answer
    to "be stable" is an infinitely wide raft, and the answer to "have no drag"
    is an infinitely long needle.

  The raw per-vertex gradient is high-frequency by nature. It gets a few Jacobi
  smoothing sweeps over the 1-ring before Adam sees it -- a cheap stand-in for
  the (I + lambda L)^-1 preconditioner that inverse-geometry work uses, and the
  difference between a shape that morphs and a shape that fizzes.

  DYNAMICS IN THE LOOP. Halfway through the descent (--dyn-after, default 0.5)
  L_dyn joins the loss: a T-step forward rollout of the very body act 2 runs,
  from the same 25 degree kick, taped end to end. Two kernels per step -- the
  soft buoyancy integral at the current (heave, roll), then a semi-implicit
  Euler update -- and the loss is the mean squared roll over the LAST 60% of an
  8 second horizon, normalised by the kick. The horizon has to be longer than a
  roll period or the tail scores the swing rather than the settling. No
  draft solve appears anywhere in the rollout: heave IS the draft, which is what
  keeps it differentiable end to end. The optimiser stops being told only "have
  a big righting arm" and starts also being told "come back and STAY back",
  which is the quantity act 2 then measures. L_stab does not switch off: it
  stays, plus a one-sided FLOOR under the righting arm, because a quiet tail on
  its own can be bought with marginal stability.

  SPEED, NOT DRAG. The rollout also carries a surge rate u, driven by a FIXED
  thrust against the very drag accumulator L_drag is built from, and in the
  dynamic phase L_speed = -mean(u over the tail) / u_ref REPLACES w_drag
  L_drag. It is the same physics said as achieved motion instead of as an
  integral, and it is deliberately a reweighting and not new information: run
  side by side with `--w-speed 0` the two land on the same hull (S 1.181 vs
  1.182 m^2, GZ(30) +0.288 vs +0.287, same bounding box to the centimetre).
  What it buys is a unit anyone can feel. The thrust is chosen once, from the
  BLOB, so the blob makes exactly u_ref and every metre per second the hull has
  over it is shape. The mass in that integrator is the DESIGN displacement,
  passed in detached: left differentiable, the term promptly buys speed by
  delivering a smaller ship.

  SYMMETRY AND CARGO. The port-starboard mirror pairing is exact on an
  icosphere, so symmetry is a PROJECTION applied after every Adam step
  (x <- 0.5 (x + mirror x_pair)) rather than a loss to be tuned against the
  others -- it halves the effective DOF, and the rendered hull is symmetric to
  the eye and not to three decimals. And the hull carries a payload: a point
  mass at 15% of the hull's own mass fixed at body y = +1.4 m, a wheelhouse and
  not ballast. It raises G, so stability has to be EARNED against a top-heavy
  loading condition, and the displacement target rises with the mass.

ACT 2 -- VERIFY BY SIMULATION (key V, or --act2).
  The same shapes, run forward as a dynamics model with no tape at all: two
  bodies side by side in the FFT ocean, the original blob and the sculpted hull,
  each a 3-DOF rigid body (heave + roll + pitch), weight at G and linear
  damping. Both get the same 25 degree kick. The blob goes over. The hull comes
  back.

  The forcing is a PRESSURE INTEGRAL, not a floating plane. Every face is taken
  to world space, reads the water height above its own centroid from a 7x5 grid
  of `ocean.sample_height` probes spanning the body's footprint, and pushes
  inward with p = rho g d. That is the whole model, and it is the difference
  between a hull and a cork: when the wavelength is near the hull's length the
  bow is in a crest while the stern is in a trough, the two ends are forced in
  antiphase, and the pitch moment largely cancels -- the hull BRIDGES the wave,
  which is exactly what a 9.6 m boat does in 8 m chop and exactly what no
  single fitted plane can express. In calm water the same integral is Archimedes
  to the last digit (the free surface is the p = 0 lid), and that identity is
  asserted at startup along with the sign of the trim moment.

ACT 3 -- THE RISING SEA, AND THE RACE (key T, or --act3).
  Act 2 asks who rights from a kick. Act 3 asks who survives weather, and who
  gets anywhere. Both bodies open the throttle to the SAME fixed thrust and
  motor into it, surging against their own shape factors, so the drag integral
  act 1 spent two thousand steps minimising finally cashes out as metres per
  second and metres made good. The blob tops out near 1.5 m/s, the hull near
  3.0; when the blob goes over, her engine dies with her and she drifts astern.
  The camera follows the pack, then the leader. No kick:
  both bodies just ride the FFT ocean while the WIND ramps from 8 to 20 m/s over
  90 seconds and the spectrum regenerates around it -- height, length and
  steepness together, at a short 12 km fetch that starts the waves at about the
  boat's length and ends at twice it. That crossing is the whole trial: the
  peak wave period rises through the hull's 3.65 s pitch period on the way up,
  so she is asked the resonance question and then asked to keep going.
  Hs is measured off the water, never commanded.
  `DisplacedMesh::sampleHeight` returns the field times
  waveScale, so the sea the bodies feel IS the sea you can see -- one ramp, not
  a visual one and a physical one. Past 85 degrees of roll a body is marked
  capsized at the significant wave height it reached, and is then left to float
  inverted: the integrals find the inverted equilibrium on their own, which is
  the honest picture. Hs is 4 sigma of the centre probe over a trailing 20 s
  window. The scoreboard is the headline: Hs_hull / Hs_blob.

    python warp_hull_sculpt.py               # window: watch it sculpt, then V, T
    python warp_hull_sculpt.py --act2        # optimise headless-fast, then act 2
    python warp_hull_sculpt.py --act3        # optimise headless-fast, then act 3
    python warp_hull_sculpt.py --classic     # exactly the pre-2026-08-27 demo
    python warp_hull_sculpt.py --k 4         # 4 optimiser steps per frame
    python warp_hull_sculpt.py --tune        # no window at all: numbers only
    python warp_hull_sculpt.py --frames 900  # window, then quit on its own
    python warp_hull_sculpt.py --selftest    # headless acceptance run, exit code
    python warp_hull_sculpt.py --film        # the 45 s film, headless, to mp4
    python warp_hull_sculpt.py --film --film-probe   # candidate framings, stills
    python warp_hull_sculpt.py --motion-csv m.txt   # per-frame trial motion
    python warp_hull_sculpt.py --w-drag 7    # every weight has a flag
    python warp_hull_sculpt.py --no-dyn      # static L_stab all the way, as before
    python warp_hull_sculpt.py --no-sym      # let it go lopsided
    python warp_hull_sculpt.py --no-cargo    # empty ship
    python warp_hull_sculpt.py --no-interop  # the update_attribute fallback
    python warp_hull_sculpt.py --cpu         # CPU Warp; slow, same answer

Keys: SPACE pause/resume   R reset to the blob   1 / 2 bias drag / stability
      3 toggle the cargo   G ghost of the initial blob   V run act 2
      T run act 3 (the rising sea)   Esc quit

Needs a Vulkan build (-DTHREEPP_WITH_VULKAN=ON). Warp falls back to CPU without
CUDA, and the demo falls back to `geometry.update_attribute` whenever
`enable_vertex_interop` hands back a null handle -- same as every other warp
example here.
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
from warp_common import sky_env
from warp_common import (Encoder, accum_normals, cli_arg, icosphere, normalize_vec3,
                         parse_size, resize_handler, signed_volume,
                         smooth_vec3_csr, standard_material, unique_edges,
                         vertex_adjacency)

# ---- flags -------------------------------------------------------------------
SELFTEST = "--selftest" in sys.argv
TUNE = "--tune" in sys.argv          # optimise only, no renderer, print numbers
ACT2_FIRST = "--act2" in sys.argv    # sculpt at full speed, then go straight to act 2
ACT3_FIRST = "--act3" in sys.argv    # ... or straight to the rising sea
CLASSIC = "--classic" in sys.argv    # the demo exactly as it was before 2026-08-27
NO_INTEROP = "--no-interop" in sys.argv
FORCE_CPU = "--cpu" in sys.argv
STEPS = int(cli_arg("--steps", 1200 if SELFTEST else 2000, float))
K_STEPS = int(cli_arg("--k", 2, float))          # optimiser steps per rendered frame
LR = cli_arg("--lr", 0.006, float)               # Adam step, metres
SHOT_DIR = cli_arg("--shots", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "doc", "screenshots"), str)
FILM = "--film" in sys.argv          # offline deterministic render of the story
WIN_W, WIN_H = parse_size(cli_arg(
    "--size", "640x360" if SELFTEST else ("1920x1080" if FILM else "1280x720"),
    str))
FILM_FPS = int(cli_arg("--film-fps", 30, float))      # output frame rate
FILM_DIR = cli_arg("--film-dir", "", str)             # optional: also keep PNGs
FILM_OUT = cli_arg("--film-out", "", str)             # default: doc/screenshots
# Candidate framings as stills, so a shot can be judged before 45 s of it is
# rendered. Writes doc/screenshots/film_probe/*.png and exits.
FILM_PROBE = "--film-probe" in sys.argv
# The water as a MEDIUM (the sailboat demo's idiom): the waterline plane the
# renderer splits air from water at, and the column's extinction + in-scatter
# colour. On for the WHOLE film -- the engine clips the murk to the
# below-surface leg of a ray, so an above-water shot only gains a sea with a
# colour of its own instead of a dark mirror, and a submerged lens gets Snell's
# window for nothing. Density is set low: the story underwater is the blob
# hanging ELEVEN METRES behind the hull, and 0.17 (the sailboat's) buries it.
MURK_SIGMA = cli_arg("--murk", 0.09, float)
MURK_COLOR = (0.020, 0.105, 0.150)
FLOOR_Y_FILM = -70.0    # a seabed at -16 m reads as a shoal under her keel
SPACING = 11.0          # the two lanes act 2 and act 3 are run in, z = +-5.5
ACT2_SECONDS = cli_arg("--act2-seconds", 24.0, float)
ACT3_SECONDS = cli_arg("--act3-seconds", 90.0, float)
# The ramp is in WIND, not in wave_scale. wave_scale multiplies height and
# leaves wavelength alone, so ramping it invents seas that cannot exist; wind is
# the physical knob and the spectrum regenerates around it, height and length
# together. Measured off this ocean by spatial FFT along the wind, fetch 12 km:
# wind 8 -> Hs 0.26 m, peak lambda 11 m; wind 20 -> Hs 0.61 m, peak lambda 21 m,
# with the energy-weighted mean running 16 -> 30 m. So the sea is NOT shorter
# than the boat -- it starts near her length and ends at twice it, and the
# slope she has to answer over her own 9.4 m is only 0.5 to 1.8 deg rms. What
# makes this sea dangerous to her is not steepness, it is that the peak wave
# period crosses her pitch period (3.65 s) on the way up.
ACT3_WIND0 = cli_arg("--act3-wind0", 8.0, float)
ACT3_WIND1 = cli_arg("--act3-wind1", 20.0, float)
FRAME_BUDGET = int(cli_arg("--frames", 0, float))   # window: quit after N frames

# The dynamic-loss phase: a taped rollout of act 2's own body.
NO_DYN = CLASSIC or "--no-dyn" in sys.argv
DYN_STEPS = int(cli_arg("--dyn-steps", 96, float))   # T, rollout steps on the tape
# The horizon has to CONTAIN a roll period, or the tail is scoring where in the
# first swing the body happens to be rather than whether it settled. With added
# mass the period is ~6 s, so 96 x 1/12 = 8.0 s. The rollout dt answers to the
# roll period, not to a frame rate: 77 steps per period is plenty for
# semi-implicit Euler, and it costs nothing over a shorter, blinder horizon.
DYN_DT = cli_arg("--dyn-dt", 1.0 / 12.0, float)      # 96 * 1/12 = 8.0 s of horizon
DYN_AFTER = cli_arg("--dyn-after", 0.5, float)       # fraction of STEPS before it bites
DYN_TAIL = cli_arg("--dyn-tail", 0.6, float)         # tail fraction the loss averages
DYN_KICK = cli_arg("--dyn-kick", 25.0, float)        # the same kick act 2 gives

# Design realism: mirror symmetry as a projection, and a raised-CoG payload.
NO_SYM = CLASSIC or "--no-sym" in sys.argv
CARGO = dict(frac=0.0 if (CLASSIC or "--no-cargo" in sys.argv)
             else cli_arg("--cargo-frac", 0.15, float),
             y=cli_arg("--cargo-y", 1.4, float))

# The film renders offline: nobody watches the window, so there is no window.
# A hidden canvas plus SUPPRESS_PRESENT is the same offline idiom the selftest
# runs, and it takes the swapchain present off the per-frame bill.
HEADLESS = SELFTEST or FILM

if not TUNE and (not tp.HAS_VULKAN or not tp.vulkan_available()):
    print("This example needs the Vulkan backend (configure with "
          "-DTHREEPP_WITH_VULKAN=ON) and a working Vulkan loader.")
    sys.exit(0)

if HEADLESS:
    # Presenting to the hidden window is pure cost offline.
    os.environ.setdefault("THREEPP_VULKAN_SUPPRESS_PRESENT", "1")


# --------------------------------------------------------------------------- #
#  The body. +x forward, +y up, +z to port; the waterplane in the rest pose is
#  y = d. Y-up rather than the plan's Z-up so the body frame and the scene frame
#  are the same one and nothing has to be re-based on the way to the renderer.
# --------------------------------------------------------------------------- #
SUBDIV = int(cli_arg("--subdiv", 4, float))   # 4 = 2562 verts / 5120 faces,
                                 # 5 = 10242 / 20480. Everything downstream --
                                 # eps, the face width, the 1-ring adjacency,
                                 # every per-vertex and per-edge normaliser --
                                 # is derived from the mesh, so this is the one
                                 # number that has to change.
BLOB = (3.6, 2.4, 2.0)           # semi-axes: 7.2 m long, 4.8 m deep, 4.0 m beam
                                 # deeper than wide, so GM < 0: it WILL roll over
BOX_HALF = (5.2, 2.6, 2.2)       # the soft envelope the shape may not leave
DENSITY = 0.5                    # of water: it floats at half its own volume
RHO_W = 1025.0
GRAV = 9.81

# Heel angles the righting arm is sampled at. Both signs, or the optimiser is
# free to build something that is only stable to one side.
HEELS_DEG = (0.0, 15.0, -15.0, 30.0, -30.0, 45.0, -45.0)
NHEEL = 7                        # len(HEELS_DEG), as a kernel-visible constant
GZ30 = 3                         # index of +30 deg in HEELS_DEG (the reported one)

# Loss weights. Live: 1 and 2 bias the first two toward drag / stability.
W = dict(drag=cli_arg("--w-drag", 4.50, float),
         stab=cli_arg("--w-stab", 1.60, float),
         disp=cli_arg("--w-disp", 250.0, float),
         lap=cli_arg("--w-lap", 200.0, float),
         edge=cli_arg("--w-edge", 4.0, float),
         box=cli_arg("--w-box", 600.0, float),
         dyn=cli_arg("--w-dyn", 3.0, float),
         # L_speed REPLACES w_drag in the dynamic phase -- the same physics
         # said as achieved motion instead of as an integral. Quoted so the two
         # pull equally hard at the operating point: d L_drag / dS is
         # w_drag / A_ref = 4.5 / 17.34 = 0.260 per m^2, and the ROLLOUT's
         # tail-mean surge moves 0.0502 m/s per m^2 at S = 1.19, so
         # 0.260 x u_ref / 0.0502 = 7.8.
         # Two ways to get this badly wrong, both of which happened: quoting it
         # off the TERMINAL speed (an 8 s horizon from rest reaches 1.10 m/s of
         # a 2.95 m/s terminal, so the analytic sensitivity is 25x too large),
         # and guessing A_ref. At w_speed 0.10 the hull FATTENED from 1.21 to
         # 3.73 m^2 the moment the drag integral handed over.
         speed=cli_arg("--w-speed", 7.8, float))
# What fraction of w_stab survives into the dynamic phase. NOT zero: the
# calm-water roll energy is happy with marginal stability -- a hull that barely
# rights, slowly, scores about as well as one that means it -- so the static
# righting arm stays on as a floor under the rollout.
W_STAB_DYN = cli_arg("--w-stab-dyn", 1.0, float)
# ... and a hard floor under the righting arm, active in the dynamic phase.
# Quoted at 30 deg and tapered by sin(heel) below that, capped past it: "at
# least this much lever arm once she is properly over".
GZ_FLOOR = cli_arg("--gz-floor", 0.24, float)
W_GZ_FLOOR = cli_arg("--w-gz-floor", 600.0, float)
CAPSIZE_DEG = 85.0               # past this a body is called over, and left there

# Added mass. A hull accelerating in water drags a slab of water with it, and
# without it these bodies bob like corks: standard deep-body heuristics, heave
# ~0.8 of the displaced mass, roll ~0.25 and pitch ~0.6 of the dry inertia.
ADD_HEAVE = cli_arg("--add-heave", 0.80, float)
ADD_ROLL = cli_arg("--add-roll", 0.25, float)
ADD_PITCH = cli_arg("--add-pitch", 0.60, float)
# The wave heightfield the trial bodies are pressed by, sampled on a grid around
# each body's footprint: 1.4 hull lengths by 1.6 beams, so a body still finds
# water under it at any heel it reaches. Odd counts put a probe on the centreline.
GRID_NX = int(cli_arg("--grid-nx", 7, float))
GRID_NZ = int(cli_arg("--grid-nz", 5, float))
MOTION_CSV = cli_arg("--motion-csv", "", str)   # per-frame trial motion, for tuning

# FORWARD MOTION. `hull_drag` accumulates the u-independent SHAPE FACTOR
# S [m^2]: Newton pressure drag at speed u is D(u) = rho u^2 S. Under a fixed
# thrust T a body tops out at sqrt(T / (rho S)), so the whole drag objective
# finally has a unit anyone can feel -- metres per second -- and the sculpted
# hull's S ~ 1.19 against the blob's 4.59 predicts it will race at about twice
# the speed. Thrust is picked ONCE, from the blob, so that the blob makes
# U_REF and the hull's advantage is entirely its shape.
U_REF = cli_arg("--u-ref", 1.5, float)          # the blob's design speed, m/s
THRUST = dict(N=cli_arg("--thrust", -1.0, float))    # -1 = auto; 0 = no engine
ADD_SURGE = cli_arg("--add-surge", 0.20, float)      # added mass, surge

# RESISTANCE. R(u) = R_form + R_friction + R_wave, one law for the taped
# rollout and for the trial race both.
#
#   R_form  = rho u|u| S_form          the Newton pressure proxy, as before
#   R_fric  = 0.5 rho u|u| Cf S_wet    ITTC-57, Cf = 0.075 / (log10 Re - 2)^2
#   R_wave  = Michell thin-ship, below
#
# The first two are shape-hungry only in the dullest way -- one wants a small
# frontal projection, the other a small wetted area, and BOTH are minimised by
# the same thing the round-6 A/B showed: with a single u^2 law, "go fast" and
# "have low drag" are literally the same objective, which is why round 6 did
# not change the hull. Wave-making is the term that is not a rescaling of the
# others: it depends on where the volume sits ALONG the hull, it peaks near
# Fr 0.3, and the hull cruises at Fr 0.31.
NU_WATER = 1.19e-6               # kinematic viscosity of seawater, m^2/s
NTHETA = int(cli_arg("--wave-theta", 32, float))     # Michell quadrature nodes
# Michell's constant is exact but this is a THIN-SHIP integral applied to a
# beamy body through a centreplane collapse, so the absolute level is a model
# choice, not a measurement. Calibrated at startup instead: --w-wave -1 picks
# the scale that puts the BLOB's wave share at 40-50% of total resistance at
# its own hump speed, which is what a full-bodied hull actually shows at
# Fr 0.3. `--w-wave 0` removes wave-making entirely and is the round-6 A/B.
W_WAVE = dict(v=cli_arg("--w-wave", -1.0, float))
WAVE_SHARE = cli_arg("--wave-share", 0.45, float)    # the calibration target
UREF_EVERY = int(cli_arg("--uref-every", 10, float)) # cruise-speed refresh cadence
WAVE_ANCHORS = "--wave-anchors" in sys.argv          # run the checks and stop


def cf_ittc(u, lwl):
    """ITTC-57 model-ship correlation line. About 0.0026 at Re 2e7."""
    re = max(abs(u) * lwl / NU_WATER, 1.0e5)
    return 0.075 / (math.log10(re) - 2.0) ** 2


def resist_visc(u, s_form, s_wet, lwl):
    """(R_form, R_fric) in newtons -- everything that IS quadratic in u."""
    return (RHO_W * u * u * s_form,
            0.5 * RHO_W * u * u * cf_ittc(u, lwl) * s_wet)


def terminal_speed(s_form, s_wet, lwl, wave_at):
    """Solve R(u) = T. `wave_at(u)` returns R_wave there, so the hump is honoured.

    Bisection rather than the round-6 closed form: with wave-making in the law
    R(u) is no longer proportional to u^2 and there is no square root to take.
    """
    t = max(THRUST["N"], 0.0)
    if t <= 0.0:
        return 0.0

    def excess(u):
        rf, rr = resist_visc(u, s_form, s_wet, lwl)
        return rf + rr + wave_at(u) - t

    lo, hi = 0.05, 12.0
    if excess(lo) > 0.0:
        return lo
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def calibrate_wave(sculpt):
    """Set the wave scale from the BLOB, at the BLOB's own hump speed.

    Michell's constant is exact for a thin ship and this body is anything but,
    so the absolute level is not something to be trusted out of the algebra --
    it is fixed against what a full-bodied hull is known to do, which is to
    carry 40-50% of its resistance as wave-making around Fr 0.3. Everything
    that matters for shaping (where the hump sits, how it responds to length
    and to fineness) comes out of the integral, not out of this number.
    """
    if W_WAVE["v"] >= 0.0:
        return W_WAVE["v"]
    W_WAVE["v"] = 1.0
    lwl = max(float(sculpt.extents()[0]), 1.0)
    u_h = 0.30 * math.sqrt(GRAV * lwl)
    raw = sculpt.wave_resistance(u_h)
    s = float(sculpt.drag.numpy()[0])
    a = float(sculpt.area.numpy()[0])
    visc = sum(resist_visc(u_h, s, a, lwl))
    W_WAVE["v"] = (WAVE_SHARE / (1.0 - WAVE_SHARE)) * visc / max(raw, 1.0e-12)
    return W_WAVE["v"]


def set_thrust(sculpt):
    """T so the BLOB tops out at U_REF under the FULL resistance law."""
    if THRUST["N"] < 0.0:
        lwl = max(float(sculpt.extents()[0]), 1.0)
        s = float(sculpt.drag.numpy()[0])
        a = float(sculpt.area.numpy()[0])
        r = sum(resist_visc(U_REF, s, a, lwl)) + sculpt.wave_resistance(U_REF)
        THRUST["N"] = round(r, -2)
    return THRUST["N"]


def wave_anchors(sculpt):
    """Two checks on R_wave, run BEFORE it is allowed anywhere near the loss.

    (a) Swept over speed, R_wave must show a hump in the Froude range where
        wave-making is known to peak for a body like this, and must die away
        at low Fr. A monotone curve would mean the phase factor is not doing
        anything and the term is just another area penalty.
    (b) A 2:1 lengthened body of the SAME volume must make clearly less wave
        at the same speed. That is the one physical statement the whole round
        rests on: if it fails, the term cannot lengthen a hull and there is no
        point wiring it in.
    """
    keep = sculpt.x.numpy().copy()
    lwl = max(float(sculpt.extents()[0]), 1.0)
    frs = np.round(np.arange(0.10, 1.21, 0.05), 2)
    us = [float(f) * math.sqrt(GRAV * lwl) for f in frs]
    rw = np.array([sculpt.wave_resistance(u) for u in us])
    ib = int(np.argmin(np.abs(frs - 0.30)))       # the operating Froude number
    ig = int(np.argmax(rw))
    # Three things have to be true, and none of them is "the peak is where I
    # expected". Low Fr must be dead; the curve must have a FINITE peak and
    # come down past it (an area penalty in disguise would rise for ever, and
    # it is the phase factor that brings it back down); and the operating point
    # must sit somewhere the term actually has teeth.
    ok_a = (rw[0] < 0.05 * rw[ib] and 0 < ig < len(rw) - 1
            and rw[-1] < 0.8 * rw[ig])
    print("  wave anchor (a) R_wave over Fr (kN): "
          + "  ".join(f"{f:.2f}:{r / 1000.0:.1f}"
                      for f, r in zip(frs[::2], rw[::2])))
    print(f"      Fr 0.10 is {100.0 * rw[0] / max(rw[ib], 1e-9):.2f}% of the "
          f"Fr 0.30 value; last hump at Fr {frs[ig]:.2f} "
          f"({rw[ig] / 1000.0:.0f} kN) falling to "
          f"{100.0 * rw[-1] / max(rw[ig], 1e-9):.0f}% by Fr {frs[-1]:.2f} "
          f"-- {'PASS' if ok_a else 'FAIL'}")
    # Worth naming: the last hump lands high because this body is STUBBY
    # (L/B 1.8, where a ship is 6-8), and a stubby body's hump is late. The
    # operating point is therefore on the steep rising FLANK rather than on the
    # crest, which is if anything the more shape-hungry place to be -- on the
    # crest dR/du is zero, on the flank it is at its largest.

    u_h = us[ib]
    rw_h = rw[ib]
    st = keep.copy()
    st[:, 0] *= 2.0
    st[:, 1] /= math.sqrt(2.0)
    st[:, 2] /= math.sqrt(2.0)
    sculpt.x.assign(st.astype(np.float32))
    sculpt.solve_drafts(wide=True)
    r_long = sculpt.wave_resistance(u_h)
    sculpt.x.assign(keep)
    sculpt.solve_drafts(wide=True)
    sculpt.forward()
    ok_b = r_long < 0.7 * rw_h
    print(f"  wave anchor (b) 2:1 lengthened, equal volume: "
          f"{r_long / 1000.0:.2f} kN vs {rw_h / 1000.0:.2f} kN at Fr "
          f"{frs[ib]:.2f} ({100.0 * r_long / max(rw_h, 1e-9):.0f}%) "
          f"-- {'PASS' if ok_b else 'FAIL'}")
    return ok_a and ok_b
# Paint. Topsides above the design waterline, antifouling below -- a fixed boot
# stripe, exactly as it is on a real hull, and the only thing needed to make a
# turtled body unmistakable: it shows its bottom.
TOPSIDE_HULL = 0xdde4ec
TOPSIDE_BLOB = 0xdd8a6a
ANTIFOUL = 0x3a1512
GRAD_SMOOTH = int(cli_arg("--grad-smooth", 3, float))   # 1-ring Jacobi sweeps
GRAD_ALPHA = cli_arg("--grad-alpha", 0.65, float)
# ... and MORE of them once the rollout and the wave integral are in the loss.
# The static objective's gradient is a smooth field; backpropagating eight
# seconds of rigid-body dynamics and a Michell integral through the same
# vertices is not. It arrives with real high-frequency content, and the place
# it shows is the sheer: the deck edge is exactly where the hydrodynamic
# gradients stop dead (a face just above the waterline gets no wave and no
# drag force at all, its neighbour a face-width below gets the lot), so the
# unsmoothed part of the gradient piles up along one line and the edge goes
# spiky. Round 2's notes predicted this knob; round 7 made it necessary.
GRAD_SMOOTH_DYN = int(cli_arg("--grad-smooth-dyn", 4, float))
# The LAPLACIAN is turned up with it, and the edge-length term is NOT. They
# pull on different things and only one of them is about spikes: Laplacian
# fairness resists a vertex standing proud of its neighbours, edge-length
# preservation resists the mesh being STRETCHED -- which is exactly what the
# wave term is buying when it lengthens the hull. Raising both (2.5x each) did
# fix the fairness numbers and cost the entire round-7 result: 11.20 m and
# 2.69 m/s became 9.75 m and 2.08 m/s. Raise the one that fights spikes.
LAP_DYN = cli_arg("--lap-dyn", 2.0, float)
EPS_FRAC = 0.22                  # waterline softening, as a fraction of a face width


# --------------------------------------------------------------------------- #
#  Kernels
# --------------------------------------------------------------------------- #
@wp.func
def _submerged(h: float, eps: float) -> float:
    """sigmoid(-h / eps), clamped so exp() cannot overflow at any draft."""
    return 1.0 / (1.0 + wp.exp(wp.clamp(h / eps, -40.0, 40.0)))


@wp.kernel
def hull_buoyancy(x: wp.array(dtype=wp.vec3),
                  tris: wp.array(dtype=int),
                  plane_n: wp.array(dtype=wp.vec3),
                  plane_d: wp.array(dtype=float),
                  eps: float,
                  vsub: wp.array(dtype=float),
                  bmom: wp.array(dtype=wp.vec3)):
    """Soft-weighted submerged volume and first moment, per heel plane.

    The tet apex is `d n`, a point ON the waterplane, which is what makes the
    weighted divergence sum the submerged volume: the cone the waterline curve
    subtends at an apex in its own plane is flat.
    """
    a, f = wp.tid()
    n = plane_n[a]
    o = n * plane_d[a]
    qa = x[tris[f * 3 + 0]] - o
    qb = x[tris[f * 3 + 1]] - o
    qc = x[tris[f * 3 + 2]] - o
    vol = wp.dot(qa, wp.cross(qb, qc)) / 6.0
    w = _submerged(wp.dot(qa + qb + qc, n) / 3.0, eps)
    wp.atomic_add(vsub, a, w * vol)
    wp.atomic_add(bmom, a, (qa + qb + qc) * (w * vol * 0.25))


@wp.kernel
def hull_solid(x: wp.array(dtype=wp.vec3),
               tris: wp.array(dtype=int),
               vtot: wp.array(dtype=float),
               gmom: wp.array(dtype=wp.vec3)):
    """Enclosed volume and its centroid at uniform density -- exact, any origin."""
    f = wp.tid()
    pa = x[tris[f * 3 + 0]]
    pb = x[tris[f * 3 + 1]]
    pc = x[tris[f * 3 + 2]]
    vol = wp.dot(pa, wp.cross(pb, pc)) / 6.0
    wp.atomic_add(vtot, 0, vol)
    wp.atomic_add(gmom, 0, (pa + pb + pc) * (vol * 0.25))


@wp.kernel
def hull_drag(x: wp.array(dtype=wp.vec3),
              tris: wp.array(dtype=int),
              plane_n: wp.array(dtype=wp.vec3),
              plane_d: wp.array(dtype=float),
              eps: float,
              fwd: wp.vec3,
              drag: wp.array(dtype=float)):
    """Newton pressure-drag proxy over the WETTED, front-facing faces.

    max(0, n.v)^2 * area with n the unit outward normal; written against the
    unnormalised cross product so there is one length() instead of three.
    """
    f = wp.tid()
    pa = x[tris[f * 3 + 0]]
    pb = x[tris[f * 3 + 1]]
    pc = x[tris[f * 3 + 2]]
    cr = wp.cross(pb - pa, pc - pa)                      # 2 * area * outward normal
    facing = wp.max(wp.dot(cr, fwd), 0.0)
    w = _submerged(wp.dot(pa + pb + pc, plane_n[0]) / 3.0 - plane_d[0], eps)
    drag_f = facing * facing / (2.0 * (wp.length(cr) + 1.0e-9))
    wp.atomic_add(drag, 0, w * drag_f)


@wp.kernel
def hull_area(x: wp.array(dtype=wp.vec3),
              tris: wp.array(dtype=int),
              plane_d: wp.array(dtype=float),
              eps: float,
              area: wp.array(dtype=float)):
    """Soft-weighted WETTED area -- the sibling of hull_drag, same waterline.

    Friction wants this small, which is the only term in the whole objective
    that argues against spreading the same volume into a thin plate.
    """
    f = wp.tid()
    pa = x[tris[f * 3 + 0]]
    pb = x[tris[f * 3 + 1]]
    pc = x[tris[f * 3 + 2]]
    w = _submerged((pa[1] + pb[1] + pc[1]) / 3.0 - plane_d[0], eps)
    wp.atomic_add(area, 0, w * 0.5 * wp.length(wp.cross(pb - pa, pc - pa)))


@wp.kernel
def wave_amp(x: wp.array(dtype=wp.vec3),
             tris: wp.array(dtype=int),
             plane_d: wp.array(dtype=float),
             eps: float,
             k0: float,
             sec: wp.array(dtype=float),
             amp: wp.array2d(dtype=float)):
    """Michell thin-ship amplitude P(theta), real and imaginary, per theta node.

    P(theta) = sum over the wetted faces of (n_x A) exp(k0 sec^2(t) y_c)
               exp(i k0 sec(t) x_c)

    with y_c the centroid's depth BELOW the free surface (negative) and x_c its
    station. This is the CENTREPLANE form: the hull's transverse coordinate is
    collapsed away and the source strength is n_x A, which is exactly the thin-
    ship df/dx dx dz written on a closed triangulation. Summed over a closed
    surface the DC term vanishes -- a body displaces no net volume -- and it is
    the phase and depth factors that break the degeneracy. That is the whole
    physics: two hulls of identical volume and identical frontal area radiate
    completely different wave systems depending on WHERE along the length the
    volume sits, which is the shape information the Newton proxy does not have.

    Thin-ship applied to a beamy body is a real approximation and the demo says
    so. It gets the Fr-dependence and the fineness-dependence right, which is
    what has to be right for it to shape a hull; the absolute level is set by
    calibration (see W_WAVE).
    """
    j, f = wp.tid()
    pa = x[tris[f * 3 + 0]]
    pb = x[tris[f * 3 + 1]]
    pc = x[tris[f * 3 + 2]]
    c = (pa + pb + pc) / 3.0
    h = c[1] - plane_d[0]                      # signed height above the surface
    w = _submerged(h, eps)
    lam = sec[j]
    # n_x A, from the unnormalised cross product: 0.5 * cr[0].
    src = 0.5 * wp.cross(pb - pa, pc - pa)[0] * w
    # exp(-k0 sec^2 |h|), NOT exp(k0 sec^2 min(h, 0)). The clamped form gives
    # every face ABOVE the waterline a depth factor of exactly 1, so the source
    # sheet there thins only as fast as the submerged sigmoid (over eps) while
    # the sheet below thins as 1/(k0 sec^2). At low speed k0 is large, the
    # below-water strip collapses, and what is left is a spurious surface layer
    # that does not vanish at all: R_wave came out at 152 kN at Fr 0.10 against
    # 25 kN at the hump, i.e. rising as the speed fell. Decaying at the wave's
    # own rate on BOTH sides makes the sheet vanish the way it must.
    # sqrt(h^2 + eps^2), not |h|. The absolute value has a derivative
    # DISCONTINUITY exactly at the waterline, so every face straddling the
    # surface hands the tape a kink -- a ring of high-frequency gradient right
    # where the sheer is, which is precisely where the spikes appeared. This is
    # |h| to well within a face width away from the surface and smooth across
    # it, which is the same bargain the submerged sigmoid already makes.
    dep = wp.exp(wp.clamp(-k0 * lam * lam * wp.sqrt(h * h + eps * eps),
                          -60.0, 0.0))
    ph = k0 * lam * c[0]
    wp.atomic_add(amp, j, 0, src * dep * wp.cos(ph))
    wp.atomic_add(amp, j, 1, src * dep * wp.sin(ph))


@wp.kernel
def wave_resist(amp: wp.array2d(dtype=float),
                sec3dt: wp.array(dtype=float),
                coef: float,
                out: wp.array(dtype=float)):
    """Havelock: R_wave = coef * sum |P(theta)|^2 sec^3(theta) dtheta.

    sec^3 diverges at theta -> pi/2, and the integral converges only because
    the depth factor exp(k0 sec^2 y) kills |P| faster. In float that product is
    an underflow times an overflow, so the exponent is clamped in `wave_amp`
    and the last node contributes an exact zero rather than a NaN.
    """
    j = wp.tid()
    p2 = amp[j, 0] * amp[j, 0] + amp[j, 1] * amp[j, 1]
    wp.atomic_add(out, 0, coef * p2 * sec3dt[j])


@wp.kernel
def hull_fairness(x: wp.array(dtype=wp.vec3),
                  offsets: wp.array(dtype=wp.int32),
                  indices: wp.array(dtype=wp.int32),
                  reg: wp.array(dtype=float)):
    """Uniform Laplacian: how far each vertex is off its own 1-ring's average."""
    i = wp.tid()
    s = offsets[i]
    e = offsets[i + 1]
    acc = wp.vec3(0.0, 0.0, 0.0)
    for k in range(s, e):
        acc += x[indices[k]]
    d = x[i] - acc / float(wp.max(e - s, 1))
    wp.atomic_add(reg, 0, wp.dot(d, d))


@wp.kernel
def hull_edges(x: wp.array(dtype=wp.vec3),
               e0: wp.array(dtype=wp.int32),
               e1: wp.array(dtype=wp.int32),
               rest: wp.array(dtype=float),
               reg: wp.array(dtype=float)):
    """Relative edge-length preservation -- the anti-sliver half of the fairing."""
    e = wp.tid()
    d = x[e1[e]] - x[e0[e]]
    r = rest[e]
    s = wp.length(d) / r - 1.0
    wp.atomic_add(reg, 1, s * s)


@wp.kernel
def hull_box(x: wp.array(dtype=wp.vec3),
             half: wp.vec3,
             boxp: wp.array(dtype=float)):
    """One-sided quadratic penalty outside the half-extent box."""
    i = wp.tid()
    p = x[i]
    ex = wp.max(wp.abs(p[0]) - half[0], 0.0)
    ey = wp.max(wp.abs(p[1]) - half[1], 0.0)
    ez = wp.max(wp.abs(p[2]) - half[2], 0.0)
    wp.atomic_add(boxp, 0, ex * ex + ey * ey + ez * ez)


@wp.kernel
def hull_loss(vsub: wp.array(dtype=float),
              bmom: wp.array(dtype=wp.vec3),
              vtot: wp.array(dtype=float),
              gmom: wp.array(dtype=wp.vec3),
              drag: wp.array(dtype=float),
              reg: wp.array(dtype=float),
              boxp: wp.array(dtype=float),
              heel_b: wp.array(dtype=wp.vec3),
              wts: wp.array(dtype=float),
              scal: wp.array(dtype=float),
              gz_min: wp.array(dtype=float),
              cargo_f: float,
              cargo_p: wp.vec3,
              loss: wp.array(dtype=float)):
    """Everything above, combined into one scalar. Launched at dim=1.

    `scal` carries the non-differentiable normalisers so the weights stay O(1)
    across mesh resolutions and hull scales:
    (V0, area ref, length ref, 1/N verts, 1/N edges, heel count).

    The cargo is a point mass at `cargo_p` worth `cargo_f` of the hull's own
    mass. Both masses are proportional to the hull's density-times-volume, so
    the mass cancels out of the combined centre and G_eff is a pure blend.
    """
    g = (gmom[0] / vtot[0] + cargo_p * cargo_f) / (1.0 + cargo_f)
    stab = float(0.0)
    short = float(0.0)
    for a in range(1, NHEEL):
        b = bmom[a] / vsub[a]
        gz = wp.dot(b - g, heel_b[a])
        stab = stab - gz
        # A one-sided FLOOR, not another term to be traded away. -sum GZ is
        # linear, so where the rollout's gradient is large the optimiser is
        # happy to sell righting arm for a quieter tail at any exchange rate.
        # This is zero above the floor and quadratic below it, so it cannot be
        # bought off: it is the difference between a hull that rights and a
        # hull that merely stops rolling eventually.
        d = wp.max(gz_min[a] - gz, 0.0)
        short = short + d * d
    dv = vtot[0] / scal[0] - 1.0
    loss[0] = (wts[0] * drag[0] / scal[1]
               + wts[1] * stab / (scal[2] * scal[5])
               + wts[6] * short / (scal[2] * scal[2] * scal[5])
               + wts[2] * dv * dv
               + wts[3] * reg[0] * scal[3] / (scal[2] * scal[2])
               + wts[4] * reg[1] * scal[4]
               + wts[5] * boxp[0] * scal[3] / (scal[2] * scal[2]))


@wp.kernel
def hull_report(vsub: wp.array(dtype=float),
                bmom: wp.array(dtype=wp.vec3),
                vtot: wp.array(dtype=float),
                gmom: wp.array(dtype=wp.vec3),
                drag: wp.array(dtype=float),
                reg: wp.array(dtype=float),
                boxp: wp.array(dtype=float),
                heel_b: wp.array(dtype=wp.vec3),
                cargo_f: float,
                cargo_p: wp.vec3,
                out: wp.array(dtype=float)):
    """The same quantities as numbers for the HUD. Outside the tape, on purpose."""
    g = (gmom[0] / vtot[0] + cargo_p * cargo_f) / (1.0 + cargo_f)
    out[0] = vtot[0]
    out[1] = drag[0]
    out[2] = reg[0]
    out[3] = reg[1]
    out[4] = boxp[0]
    for a in range(0, NHEEL):
        b = bmom[a] / vsub[a]
        out[5 + a] = wp.dot(b - g, heel_b[a])
        out[5 + NHEEL + a] = vsub[a]


# --------------------------------------------------------------------------- #
#  The rollout: act 2's physics, restated as kernels, ON the tape.
#
#  Three things make this differentiable where the static objective needed a
#  detached draft solve: heave is a state variable rather than a solved-for
#  constant, the waterplane is the same smooth sigmoid, and nothing anywhere
#  branches on the sign of anything. Every array is (T+1)-long and every step
#  writes its own slot -- reusing one accumulator would be correct forward and
#  silently wrong backward.
# --------------------------------------------------------------------------- #
@wp.kernel
def dyn_init(rphi: wp.array(dtype=float), kick: float):
    rphi[0] = kick


@wp.kernel
def roll_step_buoyancy(x: wp.array(dtype=wp.vec3),
                       tris: wp.array(dtype=int),
                       rphi: wp.array(dtype=float),
                       ry: wp.array(dtype=float),
                       t: int,
                       d0: float,
                       eps: float,
                       rvsub: wp.array(dtype=float),
                       rbmom: wp.array(dtype=wp.vec3)):
    """hull_buoyancy at step t's pose. Gravity is rotated, not the mesh.

    The waterplane offset is the REST draft minus the heave: the mesh still
    lives in its sculpting frame, so d0 is where the water was when the body was
    at rest and lifting the body by y lowers the plane through it by the same y.
    """
    f = wp.tid()
    phi = rphi[t]
    n = wp.vec3(0.0, wp.cos(phi), -wp.sin(phi))
    o = n * (d0 - ry[t])
    qa = x[tris[f * 3 + 0]] - o
    qb = x[tris[f * 3 + 1]] - o
    qc = x[tris[f * 3 + 2]] - o
    vol = wp.dot(qa, wp.cross(qb, qc)) / 6.0
    w = _submerged(wp.dot(qa + qb + qc, n) / 3.0, eps)
    wp.atomic_add(rvsub, t, w * vol)
    wp.atomic_add(rbmom, t, (qa + qb + qc) * (w * vol * 0.25))


@wp.kernel
def roll_step_integrate(rvsub: wp.array(dtype=float),
                        rbmom: wp.array(dtype=wp.vec3),
                        gmom: wp.array(dtype=wp.vec3),
                        vtot: wp.array(dtype=float),
                        rphi: wp.array(dtype=float),
                        rom: wp.array(dtype=float),
                        ry: wp.array(dtype=float),
                        rvy: wp.array(dtype=float),
                        t: int,
                        dt: float,
                        cargo_f: float,
                        cargo_p: wp.vec3,
                        inertia: float,
                        add_heave: float,
                        c_heave: float,
                        c_roll: float):
    """Semi-implicit Euler, dim=1 -- Act2.step's force and torque, verbatim."""
    vsub = wp.max(rvsub[t], 1.0e-6)
    b = rbmom[t] / vsub
    g = (gmom[0] / vtot[0] + cargo_p * cargo_f) / (1.0 + cargo_f)
    mass = RHO_W * DENSITY * vtot[0] * (1.0 + cargo_f)
    phi = rphi[t]
    bv = wp.vec3(0.0, wp.sin(phi), wp.cos(phi))
    tq = -RHO_W * GRAV * vsub * wp.dot(b - g, bv) - c_roll * rom[t]
    fz = RHO_W * GRAV * vsub - mass * GRAV - c_heave * rvy[t]
    # Weight is the real mass; the acceleration divides by mass PLUS the water
    # that comes along with it. `inertia` arrives already carrying its own
    # added term (see Sculptor._dyn_constants).
    om1 = rom[t] + dt * tq / inertia
    vy1 = rvy[t] + dt * fz / (mass * (1.0 + add_heave))
    rom[t + 1] = om1
    rvy[t + 1] = vy1
    rphi[t + 1] = phi + dt * om1
    ry[t + 1] = ry[t] + dt * vy1


@wp.kernel
def surge_step(drag: wp.array(dtype=float),
               swet: wp.array(dtype=float),
               rwave: wp.array(dtype=float),
               ru: wp.array(dtype=float),
               t: int,
               dt: float,
               thrust: float,
               mass: float,
               add_surge: float,
               lwl: float,
               u_ref: float):
    """Surge under fixed thrust against the FULL resistance law.

    All three terms come from accumulators the forward pass already built and
    that are still on the tape, so the gradient of achieved speed reaches the
    vertices through one resistance model and not through three copies of it.

    R_wave enters as its value at the running cruise speed times (u/u_ref)^2.
    That is an approximation and an honest one to name: real wave-making is
    emphatically NOT quadratic in u -- it is humped, and the hump is the whole
    reason the term is here. What the rollout needs is the right GRADIENT with
    respect to shape AT CRUISE, and R_wave(u_ref) carries that exactly; the
    quadratic only shapes how the body accelerates up to cruise, where wave
    drag is small anyway. `u_ref` is a detached running estimate of terminal
    speed, refreshed between optimiser steps, so it tracks the hull as it
    lengthens without ever putting a second gradient path into the loss.

    The mass is the DESIGN displacement, passed in detached, and that is not a
    shortcut. Read off `vtot` it would be differentiable, and the speed term
    promptly discovers that a lighter ship accelerates harder: with the mass on
    the tape the displacement error went from 0.28% to 1.34% as the hull quietly
    shrank. Displacement is a requirement here, the way it is on a real order,
    so speed may be bought with SHAPE and not by delivering a smaller ship.
    """
    u = ru[t]
    au = wp.abs(u)
    re = wp.max(au * lwl / NU_WATER, 1.0e5)
    cd = wp.log(re) / 2.302585093 - 2.0
    r_form = RHO_W * u * au * drag[0]
    r_fric = 0.5 * RHO_W * u * au * (0.075 / (cd * cd)) * swet[0]
    r_wave = rwave[0] * u * au / (u_ref * u_ref)
    ru[t + 1] = u + dt * (thrust - r_form - r_fric - r_wave) \
        / (mass * (1.0 + add_surge))


@wp.kernel
def surge_speed(ru: wp.array(dtype=float),
                t0: int,
                inv_norm: float,
                out: wp.array(dtype=float)):
    """-mean(u) over the TAIL, in units of the design speed. Same tail as L_dyn."""
    k = wp.tid()
    wp.atomic_add(out, 0, -ru[t0 + k] * inv_norm)


@wp.kernel
def rigid_drag(x: wp.array2d(dtype=wp.vec3),
               tris: wp.array(dtype=int),
               eps: float,
               fwd: wp.vec3,
               drag: wp.array(dtype=float),
               area: wp.array(dtype=float)):
    """hull_drag and hull_area per trial body, in one launch. The verts arrive
    draft-shifted, so y = 0 is the waterline and no plane is needed at all."""
    b, f = wp.tid()
    pa = x[b, tris[f * 3 + 0]]
    pb = x[b, tris[f * 3 + 1]]
    pc = x[b, tris[f * 3 + 2]]
    cr = wp.cross(pb - pa, pc - pa)
    facing = wp.max(wp.dot(cr, fwd), 0.0)
    w = _submerged((pa[1] + pb[1] + pc[1]) / 3.0, eps)
    wp.atomic_add(drag, b, w * facing * facing / (2.0 * (wp.length(cr) + 1.0e-9)))
    wp.atomic_add(area, b, w * 0.5 * wp.length(cr))


@wp.kernel
def rigid_wave_amp(x: wp.array2d(dtype=wp.vec3),
                   tris: wp.array(dtype=int),
                   eps: float,
                   k0: float,
                   sec: wp.array(dtype=float),
                   amp: wp.array3d(dtype=float)):
    """wave_amp per trial body, waterline at y = 0. What is optimised races."""
    b, j, f = wp.tid()
    pa = x[b, tris[f * 3 + 0]]
    pb = x[b, tris[f * 3 + 1]]
    pc = x[b, tris[f * 3 + 2]]
    c = (pa + pb + pc) / 3.0
    w = _submerged(c[1], eps)
    lam = sec[j]
    src = 0.5 * wp.cross(pb - pa, pc - pa)[0] * w
    dep = wp.exp(wp.clamp(-k0 * lam * lam * wp.sqrt(c[1] * c[1] + eps * eps),
                          -60.0, 0.0))
    ph = k0 * lam * c[0]
    wp.atomic_add(amp, b, j, 0, src * dep * wp.cos(ph))
    wp.atomic_add(amp, b, j, 1, src * dep * wp.sin(ph))


@wp.kernel
def roll_energy(rphi: wp.array(dtype=float),
                t0: int,
                inv_norm: float,
                out: wp.array(dtype=float)):
    """Mean squared roll over the TAIL of the horizon, in units of the kick.

    The tail is the whole point: the head of the trajectory is the kick itself
    and carries no signal, while a hull that rights fast but then rings is not
    what anyone wants to be aboard.
    """
    k = wp.tid()
    p = rphi[t0 + k]
    wp.atomic_add(out, 0, p * p * inv_norm)


@wp.kernel
def add_scaled(src: wp.array(dtype=float), w: float, dst: wp.array(dtype=float)):
    wp.atomic_add(dst, 0, w * src[0])


@wp.kernel
def symmetrize(x_in: wp.array(dtype=wp.vec3),
               pair: wp.array(dtype=wp.int32),
               x_out: wp.array(dtype=wp.vec3)):
    """Average every vertex with its mirrored partner. Two buffers, never in place."""
    i = wp.tid()
    p = x_in[i]
    q = x_in[pair[i]]
    x_out[i] = wp.vec3(0.5 * (p[0] + q[0]), 0.5 * (p[1] + q[1]),
                       0.5 * (p[2] - q[2]))


@wp.kernel
def probe_volume(x: wp.array(dtype=wp.vec3),
                 tris: wp.array(dtype=int),
                 plane_n: wp.array(dtype=wp.vec3),
                 cand: wp.array2d(dtype=float),
                 eps: float,
                 out: wp.array2d(dtype=float)):
    """V_sub for a whole grid of candidate drafts at once -- the draft solve.

    One launch instead of a bisection's worth of launch-and-sync round trips:
    the interval is refined on the host between rounds, and there are only
    three rounds.
    """
    a, c, f = wp.tid()
    n = plane_n[a]
    o = n * cand[a, c]
    qa = x[tris[f * 3 + 0]] - o
    qb = x[tris[f * 3 + 1]] - o
    qc = x[tris[f * 3 + 2]] - o
    w = _submerged(wp.dot(qa + qb + qc, n) / 3.0, eps)
    wp.atomic_add(out, a, c, w * wp.dot(qa, wp.cross(qb, qc)) / 6.0)


@wp.kernel
def rigid_buoyancy(x: wp.array2d(dtype=wp.vec3),
                   tris: wp.array(dtype=int),
                   plane_n: wp.array(dtype=wp.vec3),
                   plane_d: wp.array(dtype=float),
                   eps: float,
                   vsub: wp.array(dtype=float),
                   bmom: wp.array(dtype=wp.vec3)):
    """hull_buoyancy for act 2: one row of `x` per body, one plane per body."""
    b, f = wp.tid()
    n = plane_n[b]
    o = n * plane_d[b]
    qa = x[b, tris[f * 3 + 0]] - o
    qb = x[b, tris[f * 3 + 1]] - o
    qc = x[b, tris[f * 3 + 2]] - o
    vol = wp.dot(qa, wp.cross(qb, qc)) / 6.0
    w = _submerged(wp.dot(qa + qb + qc, n) / 3.0, eps)
    wp.atomic_add(vsub, b, w * vol)
    wp.atomic_add(bmom, b, (qa + qb + qc) * (w * vol * 0.25))


@wp.kernel
def wave_pressure(x: wp.array2d(dtype=wp.vec3),
                  tris: wp.array(dtype=int),
                  rot: wp.array(dtype=wp.mat33),
                  org: wp.array(dtype=wp.vec3),
                  gpos: wp.array(dtype=wp.vec3),
                  grid: wp.array3d(dtype=float),
                  gorg: wp.array(dtype=wp.vec2),
                  gstep: wp.array(dtype=wp.vec2),
                  nx: int,
                  nz: int,
                  rho_g: float,
                  eps: float,
                  force: wp.array(dtype=wp.vec3),
                  mom: wp.array(dtype=wp.vec3)):
    """Hydrostatic pressure over the wetted surface, against a SAMPLED wave field.

    Every face goes to WORLD space, reads the water height above ITS OWN
    centroid by bilinear interpolation of the probe grid, and pushes inward with
    p = rho g d. Bow and stern therefore see different water, which is the whole
    point -- see the module docstring's act 2.

    The free surface is the p = 0 lid, so over the closed triangulation the sum
    is exactly buoyancy: in calm water F.y = rho g V_sub by the divergence
    theorem, asserted at startup rather than assumed. `d * sigmoid(d / eps)` is
    the soft rectifier -- max(0, d) with the same face-width eps the sculpting
    integrals use, so a face straddling the surface is neither wet nor dry.
    """
    b, f = wp.tid()
    r = rot[b]
    o = org[b]
    pa = r * x[b, tris[f * 3 + 0]] + o
    pb = r * x[b, tris[f * 3 + 1]] + o
    pc = r * x[b, tris[f * 3 + 2]] + o
    an = wp.cross(pb - pa, pc - pa) * 0.5                 # area * outward normal
    c = (pa + pb + pc) / 3.0
    gg = gorg[b]
    st = gstep[b]
    fx = wp.clamp((c[0] - gg[0]) / st[0], 0.0, float(nx - 1) - 1.0e-4)
    fz = wp.clamp((c[2] - gg[1]) / st[1], 0.0, float(nz - 1) - 1.0e-4)
    i0 = int(fx)
    j0 = int(fz)
    i1 = wp.min(i0 + 1, nx - 1)
    j1 = wp.min(j0 + 1, nz - 1)
    tx = fx - float(i0)
    tz = fz - float(j0)
    h = ((1.0 - tx) * ((1.0 - tz) * grid[b, i0, j0] + tz * grid[b, i0, j1])
         + tx * ((1.0 - tz) * grid[b, i1, j0] + tz * grid[b, i1, j1]))
    d = h - c[1]                                          # depth below the surface
    p = rho_g * d * _submerged(-d, eps)
    fv = an * (-p)
    wp.atomic_add(force, b, fv)
    wp.atomic_add(mom, b, wp.cross(c - gpos[b], fv))


@wp.kernel
def emit_pose(x: wp.array(dtype=wp.vec3),
              rot: wp.mat33,
              off: wp.vec3,
              out: wp.array(dtype=wp.vec3)):
    """Body -> mesh-local positions: the rigid pose the renderer draws."""
    i = wp.tid()
    out[i] = rot * x[i] + off


@wp.kernel
def emit_pose_row(x: wp.array2d(dtype=wp.vec3),
                  row: int,
                  rot: wp.mat33,
                  off: wp.vec3,
                  out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[i] = rot * x[row, i] + off


# --------------------------------------------------------------------------- #
#  Geometry helpers (numpy, once)
# --------------------------------------------------------------------------- #
def heel_frames(deg):
    """(waterplane normal, signed transverse axis) in body coordinates, per angle.

    Gravity is rotated, not the mesh. If the body is heeled by t about +x, then
    world up in body coordinates is n = (0, cos t, -sin t) and the horizontal
    direction B is displaced along is (0, sin t, cos t). The two are orthogonal,
    which is why the draft offset drops out of GZ entirely.

    THE SIGN IS LOAD-BEARING. The moment about +x is -F (B - G).(0, sin t, cos t),
    so the RIGHTING arm -- positive when the hull comes back -- is that dot
    product times sgn(t). Without the sgn, GZ(-t) = -GZ(+t) on any symmetric
    hull and a loss summed over +/- pairs cancels to zero: the first cut of this
    demo sculpted a LOPSIDED hull, because leaning to one side was the only way
    left to make the sum move.
    """
    n = np.zeros((len(deg), 3), np.float32)
    b = np.zeros((len(deg), 3), np.float32)
    for i, d in enumerate(deg):
        t = math.radians(d)
        sgn = -1.0 if d < 0.0 else 1.0
        n[i] = (0.0, math.cos(t), -math.sin(t))
        b[i] = (0.0, sgn * math.sin(t), sgn * math.cos(t))
    return n, b


def rot_x(phi):
    c, s = math.cos(phi), math.sin(phi)
    return wp.mat33(1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c)


def rot_np(phi, theta):
    """Body -> world for (roll about +x, pitch about +z), as a numpy 3x3.

    +x forward, +z to port, so the transverse axis pitch turns about is +z and
    positive theta lifts the bow. Roll is applied outside pitch; at the angles
    act 3 reaches the two no longer commute, and this is the order the pose and
    the plane normal are both built with, which is the part that matters.
    """
    cp, sp = math.cos(phi), math.sin(phi)
    ct, st = math.cos(theta), math.sin(theta)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]])
    rz = np.array([[ct, -st, 0.0], [st, ct, 0.0], [0.0, 0.0, 1.0]])
    return rx @ rz


def rot_mat(phi, theta):
    return wp.mat33(*rot_np(phi, theta).astype(np.float32).reshape(-1))


def mirror_pairing(rest):
    """For each vertex, the vertex its z-mirror image lands on, plus the residual.

    An icosphere is exactly z-symmetric -- the base icosahedron is, and midpoint
    subdivision preserves it -- so this pairing is a permutation to float
    rounding, and symmetry can be a projection instead of yet another weight to
    balance. Chunked so the O(N^2) distance matrix never exists all at once.
    """
    mir = rest.astype(np.float64).copy()
    mir[:, 2] *= -1.0
    ref = rest.astype(np.float64)
    pair = np.empty(len(rest), np.int32)
    worst = 0.0
    for s in range(0, len(rest), 256):
        blk = mir[s:s + 256]
        d2 = ((blk[:, None, :] - ref[None, :, :]) ** 2).sum(-1)
        j = d2.argmin(1)
        pair[s:s + 256] = j
        worst = max(worst, float(np.sqrt(d2[np.arange(len(j)), j]).max()))
    return pair, worst


# --------------------------------------------------------------------------- #
#  The optimiser
# --------------------------------------------------------------------------- #
class Sculptor:
    """The whole differentiable half: state, one tape step, and the draft solve."""

    def __init__(self, device):
        self.device = device
        v, f = icosphere(SUBDIV)
        self.rest = (v * np.array(BLOB, np.float32)).astype(np.float32)
        self.faces = f
        self.n_verts = len(self.rest)
        self.n_faces = len(f)

        # Mirror pairing off the REST pose, once. Symmetrising the rest first
        # makes the pairing exact rather than nearly exact, so the projection
        # below is idempotent and the residual it reports means something.
        self.pair_np, self.sym_res0 = mirror_pairing(self.rest)
        mir = self.rest.copy()
        mir[:, 2] *= -1.0
        self.rest = (0.5 * (self.rest + mir[self.pair_np])).astype(np.float32)

        self.v_hull0 = signed_volume(self.rest, f)          # enclosed volume, m^3
        self.cargo_f = float(CARGO["frac"])
        self.cargo_p = (0.0, float(CARGO["y"]), 0.0)
        # It has to float at its LOADED weight, so the payload moves the
        # waterline as well as the centre of gravity.
        self.v_disp = DENSITY * self.v_hull0 * (1.0 + self.cargo_f)
        self.mass = RHO_W * self.v_disp
        self.lref = self.v_hull0 ** (1.0 / 3.0)
        self.aref = self.v_hull0 ** (2.0 / 3.0)
        # A face width sets how sharp the waterline may be: sharper than the
        # mesh can resolve is just a discontinuity with extra steps.
        self.face_w = math.sqrt(2.0 * 4.0 * math.pi * (self.lref * 0.5) ** 2
                                / max(self.n_faces, 1))
        self.eps = EPS_FRAC * max(self.face_w, 1e-4) * 4.0

        edges = unique_edges(f)
        offs, idx = vertex_adjacency(f, self.n_verts)
        rest_len = np.linalg.norm(self.rest[edges[:, 1]] - self.rest[edges[:, 0]],
                                  axis=1).astype(np.float32)

        d = device
        self.x = wp.array(self.rest, dtype=wp.vec3, device=d, requires_grad=True)
        self.tris = wp.array(f.reshape(-1), dtype=int, device=d)
        self.e0 = wp.array(edges[:, 0], dtype=wp.int32, device=d)
        self.e1 = wp.array(edges[:, 1], dtype=wp.int32, device=d)
        self.rest_len = wp.array(rest_len, dtype=float, device=d)
        self.offsets = wp.array(offs, dtype=wp.int32, device=d)
        self.indices = wp.array(idx, dtype=wp.int32, device=d)
        self.n_edges = len(edges)

        hn, hb = heel_frames(HEELS_DEG)
        self.plane_n = wp.array(hn, dtype=wp.vec3, device=d)
        self.heel_b = wp.array(hb, dtype=wp.vec3, device=d)
        self.plane_d = wp.zeros(NHEEL, dtype=float, device=d)

        mk = lambda n, t: wp.zeros(n, dtype=t, device=d, requires_grad=True)
        self.vsub = mk(NHEEL, float)
        self.bmom = mk(NHEEL, wp.vec3)
        self.vtot = mk(1, float)
        self.gmom = mk(1, wp.vec3)
        self.drag = mk(1, float)
        self.area = mk(1, float)
        self.wamp = mk((NTHETA, 2), float)
        self.rwave = mk(1, float)
        self.rwave_probe = wp.zeros(1, dtype=float, device=d)
        # Michell quadrature: midpoint nodes in the OPEN interval (0, pi/2), so
        # neither the theta = 0 end nor the sec^3 singularity is ever evaluated.
        th = (np.arange(NTHETA) + 0.5) * (0.5 * math.pi / NTHETA)
        sec_h = 1.0 / np.cos(th)
        self.sec = wp.array(sec_h.astype(np.float32), dtype=float, device=d)
        self.sec3dt = wp.array(
            (sec_h ** 3 * (0.5 * math.pi / NTHETA)).astype(np.float32),
            dtype=float, device=d)
        # Detached running cruise speed, and the Lwl that goes with it.
        self.u_run = U_REF
        self.lwl = 2.0 * BLOB[0]
        self.reg = mk(2, float)
        self.boxp = mk(1, float)
        self.loss = mk(1, float)
        self.gsm = [wp.zeros(self.n_verts, dtype=wp.vec3, device=d) for _ in range(2)]

        self.pair = wp.array(self.pair_np, dtype=wp.int32, device=d)
        self.xsym = wp.zeros(self.n_verts, dtype=wp.vec3, device=d)

        # The rollout. (T+1) slots everywhere: the backward pass reads every
        # step's own accumulator, so nothing here may be recycled.
        self.T = max(DYN_STEPS, 2)
        self.dyn_t0 = min(int((1.0 - DYN_TAIL) * self.T), self.T)
        self.kick = math.radians(DYN_KICK)
        self.dyn_inv_norm = 1.0 / (self.kick ** 2 * (self.T + 1 - self.dyn_t0))
        self.rphi = mk(self.T + 1, float)
        self.rom = mk(self.T + 1, float)
        self.ry = mk(self.T + 1, float)
        self.rvy = mk(self.T + 1, float)
        self.rvsub = mk(self.T + 1, float)
        self.rbmom = mk(self.T + 1, wp.vec3)
        self.dynloss = mk(1, float)
        # Surge rides the same horizon and is scored over the same tail.
        self.ru = mk(self.T + 1, float)
        self.spdloss = mk(1, float)
        self.spd_inv_norm = 1.0 / (U_REF * (self.T + 1 - self.dyn_t0))
        self._dyn_pushed = None

        self.wts = wp.zeros(7, dtype=float, device=d)
        gzf = np.array([GZ_FLOOR * min(abs(math.sin(math.radians(t)))
                                       / math.sin(math.radians(30.0)), 1.0)
                        for t in HEELS_DEG], np.float32)
        self.gz_min = wp.array(gzf, dtype=float, device=d)
        self.gz_floor_host = gzf
        self.scal = wp.array(np.array([
            self.v_hull0, self.aref, self.lref, 1.0 / self.n_verts,
            1.0 / self.n_edges, float(NHEEL - 1)], np.float32),
            dtype=float, device=d)
        self.report = wp.zeros(5 + 2 * NHEEL, dtype=float, device=d)

        self.cand = wp.zeros((NHEEL, 9), dtype=float, device=d)
        self.probe = wp.zeros((NHEEL, 9), dtype=float, device=d)
        self._d_host = np.zeros(NHEEL, np.float64)

        self.opt = None
        self.step_count = 0
        self._dyn_constants()
        self.push_weights()
        self.reset()

    def _dyn_constants(self):
        """Inertia and damping for the rollout -- detached, and quoted once.

        The rollout's body is the design at rest, so its radius of gyration and
        its damping are held at the blob's, not re-derived from the current
        extents every step: a mass property that chases the shape would put a
        second, undocumented gradient path into the loss.
        """
        beam0 = BLOB[2]
        k = 0.42 * 2.0 * beam0
        m0 = RHO_W * DENSITY * self.v_hull0 * (1.0 + self.cargo_f)
        self.dyn_inertia = m0 * k * k * (1.0 + ADD_ROLL)
        ma = m0 * (1.0 + ADD_HEAVE)
        self.dyn_c_heave = 0.55 * 2.0 * math.sqrt(ma * RHO_W * GRAV * beam0 * beam0)
        self.dyn_c_roll = 0.16 * 2.0 * math.sqrt(self.dyn_inertia * m0 * GRAV * 0.6)

    def set_cargo(self, frac):
        """Re-load the ship live: G moves, and so does the waterline."""
        self.cargo_f = float(frac)
        CARGO["frac"] = self.cargo_f
        self.v_disp = DENSITY * self.v_hull0 * (1.0 + self.cargo_f)
        self.mass = RHO_W * self.v_disp
        self._dyn_constants()
        self.solve_drafts(wide=True)

    # -- weights ------------------------------------------------------------
    def in_dyn(self):
        """True once the schedule has handed the stability term to the rollout."""
        return not NO_DYN and self.step_count >= int(DYN_AFTER * STEPS)

    @staticmethod
    def stab_scale(dyn):
        return W_STAB_DYN if dyn else 1.0

    @staticmethod
    def drag_scale(dyn):
        """The bare drag integral is OFF in the dynamic phase -- L_speed is the
        same information said as motion, and charging for both is double
        counting. The static phase keeps it: it is far cheaper shaping.

        `--w-speed 0` hands the dynamic phase back to the drag integral, which
        is the pre-round-6 formulation and therefore the A/B.
        """
        return 1.0 if (not dyn or W["speed"] <= 0.0) else 0.0

    @staticmethod
    def lap_scale(dyn):
        """Laplacian fairness is turned UP for the dynamic phase. See LAP_DYN."""
        return LAP_DYN if dyn else 1.0

    def push_weights(self, dyn=False):
        # The static righting-arm term is turned DOWN in the dynamic phase, not
        # off. The two are answers to the same question and double counting
        # them at full weight would be arbitrary -- but switching L_stab off
        # entirely let the rollout settle for marginal stability (GZ(30)
        # collapsed from +0.31 to +0.08 m), because a hull that barely rights,
        # slowly, still has a quiet tail. L_stab stays as the floor.
        self.wts.assign(np.array([W["drag"] * self.drag_scale(dyn),
                                  W["stab"] * self.stab_scale(dyn),
                                  W["disp"],
                                  W["lap"] * self.lap_scale(dyn),
                                  W["edge"], W["box"],
                                  W_GZ_FLOOR if dyn else 0.0], np.float32))
        self._dyn_pushed = dyn

    def reset(self):
        from warp.optim import Adam
        self.x.assign(self.rest)
        self.opt = Adam([self.x], lr=LR)
        self.step_count = 0
        self.solve_drafts(wide=True)

    # -- the draft solve, outside the tape ----------------------------------
    def solve_drafts(self, wide=False):
        """Find d(theta) with V_sub(d) = V_disp, per heel plane.

        A detached constant per optimiser step: the standard implicit-function
        shortcut. V_sub is monotonic in d, so this is an interval refinement on
        a 9-wide probe grid -- three launches, three syncs, not thirty.
        """
        c = self.cand.shape[1]
        if wide:
            lo = np.full(NHEEL, -1.6 * max(BOX_HALF), np.float64)
            hi = np.full(NHEEL, +1.6 * max(BOX_HALF), np.float64)
            rounds = 4
        else:
            lo = self._d_host - 0.25 * self.lref
            hi = self._d_host + 0.25 * self.lref
            rounds = 2
        u = np.linspace(0.0, 1.0, c)
        for r in range(rounds):
            grid = lo[:, None] + (hi - lo)[:, None] * u[None, :]
            self.cand.assign(grid.astype(np.float32))
            self.probe.zero_()
            wp.launch(probe_volume, dim=(NHEEL, c, self.n_faces), device=self.device,
                      inputs=[self.x, self.tris, self.plane_n, self.cand, self.eps,
                              self.probe])
            v = self.probe.numpy().astype(np.float64)
            miss = False
            for a in range(NHEEL):
                k = int(np.searchsorted(v[a], self.v_disp))
                if k <= 0 or k >= c:
                    miss = True
                    k = min(max(k, 1), c - 1)
                lo[a], hi[a] = grid[a, k - 1], grid[a, k]
            if miss and not wide:
                return self.solve_drafts(wide=True)
        self._d_host = 0.5 * (lo + hi)
        self.plane_d.assign(self._d_host.astype(np.float32))

    # -- the rollout --------------------------------------------------------
    def rollout(self):
        """T steps of heave+roll from the kick, recorded wherever it is called.

        Inside `with tape:` this is the differentiable L_dyn; outside it, it is
        the same trajectory as a plain forward measurement. One code path, so
        the number the demo reports is the number the gradient came from.
        """
        dev = self.device
        for a in (self.rphi, self.rom, self.ry, self.rvy, self.rvsub,
                  self.rbmom, self.dynloss, self.ru, self.spdloss):
            a.zero_()
        d0 = float(self._d_host[0])
        wp.launch(dyn_init, dim=1, device=dev, inputs=[self.rphi, self.kick])
        for t in range(self.T):
            wp.launch(roll_step_buoyancy, dim=self.n_faces, device=dev,
                      inputs=[self.x, self.tris, self.rphi, self.ry, t, d0,
                              self.eps, self.rvsub, self.rbmom])
            wp.launch(roll_step_integrate, dim=1, device=dev,
                      inputs=[self.rvsub, self.rbmom, self.gmom, self.vtot,
                              self.rphi, self.rom, self.ry, self.rvy, t, DYN_DT,
                              self.cargo_f, wp.vec3(*self.cargo_point()),
                              self.dyn_inertia, ADD_HEAVE, self.dyn_c_heave,
                              self.dyn_c_roll])
            # Surge is uncoupled from heave and roll here on purpose: this is a
            # design objective, not a manoeuvring model, and a coupled one would
            # put the added-resistance-in-waves question on the tape without
            # any of the physics that answers it.
            wp.launch(surge_step, dim=1, device=dev,
                      inputs=[self.drag, self.area, self.rwave, self.ru, t,
                              DYN_DT, THRUST["N"], self.mass, ADD_SURGE,
                              self.lwl, self.u_run])
        wp.launch(roll_energy, dim=self.T + 1 - self.dyn_t0, device=dev,
                  inputs=[self.rphi, self.dyn_t0, self.dyn_inv_norm, self.dynloss])
        wp.launch(surge_speed, dim=self.T + 1 - self.dyn_t0, device=dev,
                  inputs=[self.ru, self.dyn_t0, self.spd_inv_norm, self.spdloss])

    def _wave_coef(self, u):
        """Havelock's prefactor, times the calibration scale. 4 rho g^2 / (pi U^2)."""
        return (W_WAVE["v"] * 4.0 * RHO_W * GRAV * GRAV
                / (math.pi * max(u * u, 1.0e-4)))

    def wave_resistance(self, u):
        """R_wave at speed u, forward only.

        Clobbers `wamp`, so it must never be called between a taped forward and
        its backward. It is for the anchors, the calibration and the printed
        breakdown, none of which are on a tape.
        """
        if W_WAVE["v"] <= 0.0:
            return 0.0
        self.wamp.zero_()
        self.rwave_probe.zero_()
        wp.launch(wave_amp, dim=(NTHETA, self.n_faces), device=self.device,
                  inputs=[self.x, self.tris, self.plane_d, self.eps,
                          GRAV / max(u * u, 1.0e-4), self.sec, self.wamp])
        wp.launch(wave_resist, dim=NTHETA, device=self.device,
                  inputs=[self.wamp, self.sec3dt, self._wave_coef(u),
                          self.rwave_probe])
        return float(self.rwave_probe.numpy()[0])

    def wave_curve(self, n=9, umax=5.0):
        """R_wave sampled over speed, as a cheap interpolant.

        The terminal-speed solve needs R_wave at forty trial speeds and each
        one is a pair of launches; nine samples and a linear interpolation is
        the same answer for a fortieth of the syncs.
        """
        us = np.linspace(0.25, umax, n)
        rs = np.array([self.wave_resistance(float(u)) for u in us])
        return us, rs

    def breakdown(self, u=None):
        """(R_form, R_fric, R_wave, u) in newtons at speed u -- default cruise."""
        u = self.u_run if u is None else u
        s = float(self.drag.numpy()[0])
        a = float(self.area.numpy()[0])
        return (RHO_W * u * u * s,
                0.5 * RHO_W * u * u * cf_ittc(u, self.lwl) * a,
                self.wave_resistance(u), u)

    def cruise_speed(self):
        """Terminal speed under the full law, solved properly (not the u^2 form)."""
        self.lwl = max(float(self.extents()[0]), 1.0)
        us, rs = self.wave_curve()
        return terminal_speed(float(self.drag.numpy()[0]),
                              float(self.area.numpy()[0]), self.lwl,
                              lambda u: float(np.interp(u, us, rs, left=0.0,
                                                        right=rs[-1])))

    def refresh_cruise(self):
        """Re-solve the detached cruise speed, and the Lwl it is quoted at.

        Between optimiser steps, never inside one. R_wave is held at its value
        for the CURRENT k0 while this iterates, which is why it is a fixed
        point rather than a formula: the hull lengthens, the terminal speed
        rises, and the wavenumber the Michell integral is evaluated at has to
        follow it or the term would be shaping the hull for a speed it no
        longer makes. Damped by half, because a bare fixed point on a hull that
        is being lengthened under it oscillates.
        """
        self.lwl = max(float(self.extents()[0]), 1.0)
        s = float(self.drag.numpy()[0])
        a = float(self.area.numpy()[0])
        rw = float(self.rwave.numpy()[0])
        t = max(THRUST["N"], 0.0)
        if t <= 0.0:
            return self.u_run
        u = self.u_run
        for _ in range(4):
            k = (RHO_W * s + 0.5 * RHO_W * cf_ittc(u, self.lwl) * a
                 + rw / max(self.u_run * self.u_run, 1.0e-4))
            u = math.sqrt(t / max(k, 1.0e-9))
        self.u_run = 0.5 * (self.u_run + u)
        return self.u_run

    def roll_trace(self):
        """The rollout as numbers: (tail energy, peak |roll| deg, final |roll| deg)."""
        self.forward()
        self.rollout()
        p = np.degrees(self.rphi.numpy().astype(np.float64))
        return (float(self.dynloss.numpy()[0]), float(np.abs(p).max()),
                float(abs(p[-1])))

    # -- one Adam step ------------------------------------------------------
    def forward(self, dyn=False):
        """Zero the accumulators, run every integral, combine into loss[0]."""
        for a in (self.vsub, self.bmom, self.vtot, self.gmom, self.drag,
                  self.reg, self.boxp, self.loss, self.area, self.wamp,
                  self.rwave):
            a.zero_()
        dev = self.device
        wp.launch(hull_buoyancy, dim=(NHEEL, self.n_faces), device=dev,
                  inputs=[self.x, self.tris, self.plane_n, self.plane_d, self.eps,
                          self.vsub, self.bmom])
        wp.launch(hull_solid, dim=self.n_faces, device=dev,
                  inputs=[self.x, self.tris, self.vtot, self.gmom])
        wp.launch(hull_drag, dim=self.n_faces, device=dev,
                  inputs=[self.x, self.tris, self.plane_n, self.plane_d, self.eps,
                          wp.vec3(1.0, 0.0, 0.0), self.drag])
        wp.launch(hull_area, dim=self.n_faces, device=dev,
                  inputs=[self.x, self.tris, self.plane_d, self.eps, self.area])
        if W_WAVE["v"] > 0.0:
            k0 = GRAV / max(self.u_run * self.u_run, 1.0e-4)
            wp.launch(wave_amp, dim=(NTHETA, self.n_faces), device=dev,
                      inputs=[self.x, self.tris, self.plane_d, self.eps, k0,
                              self.sec, self.wamp])
            wp.launch(wave_resist, dim=NTHETA, device=dev,
                      inputs=[self.wamp, self.sec3dt, self._wave_coef(self.u_run),
                              self.rwave])
        wp.launch(hull_fairness, dim=self.n_verts, device=dev,
                  inputs=[self.x, self.offsets, self.indices, self.reg])
        wp.launch(hull_edges, dim=self.n_edges, device=dev,
                  inputs=[self.x, self.e0, self.e1, self.rest_len, self.reg])
        wp.launch(hull_box, dim=self.n_verts, device=dev,
                  inputs=[self.x, wp.vec3(*BOX_HALF), self.boxp])
        wp.launch(hull_loss, dim=1, device=dev,
                  inputs=[self.vsub, self.bmom, self.vtot, self.gmom, self.drag,
                          self.reg, self.boxp, self.heel_b, self.wts, self.scal,
                          self.gz_min, self.cargo_f,
                          wp.vec3(*self.cargo_point()), self.loss])
        if dyn:
            # L_dyn is added to the SAME scalar, so one backward pass carries
            # both the geometry terms and four seconds of rigid-body dynamics.
            self.rollout()
            wp.launch(add_scaled, dim=1, device=dev,
                      inputs=[self.dynloss, W["dyn"], self.loss])
            wp.launch(add_scaled, dim=1, device=dev,
                      inputs=[self.spdloss, W["speed"], self.loss])

    def step(self):
        dyn = self.in_dyn()
        if dyn != self._dyn_pushed:
            self.push_weights(dyn)
        self.solve_drafts(wide=(self.step_count == 0))
        tape = wp.Tape()
        with tape:
            self.forward(dyn)
        tape.backward(loss=self.loss)
        # Precondition: a raw per-vertex gradient is dominated by the mesh's own
        # highest frequency, and Adam's per-element normalisation keeps it there.
        g = self.x.grad
        ns = GRAD_SMOOTH_DYN if dyn else GRAD_SMOOTH
        for i in range(ns):
            src = g if i == 0 else self.gsm[(i - 1) % 2]
            wp.launch(smooth_vec3_csr, dim=self.n_verts, device=self.device,
                      inputs=[src, self.offsets, self.indices, GRAD_ALPHA,
                              self.gsm[i % 2]])
        self.opt.step([self.gsm[(ns - 1) % 2] if ns else g])
        if not NO_SYM:
            # Projection, not penalty: after the step, never during it.
            wp.launch(symmetrize, dim=self.n_verts, device=self.device,
                      inputs=[self.x, self.pair, self.xsym])
            wp.copy(self.x, self.xsym)
        tape.zero()
        self.step_count += 1
        # Detached, and cheap enough at this cadence to be worth doing honestly.
        if dyn and self.step_count % UREF_EVERY == 0:
            self.refresh_cruise()

    # -- readout ------------------------------------------------------------
    def measure(self):
        """Forward-only evaluation at the solved drafts; returns a dict."""
        dyn = self.in_dyn()
        self.solve_drafts(wide=True)
        self.forward(dyn)
        wp.launch(hull_report, dim=1, device=self.device,
                  inputs=[self.vsub, self.bmom, self.vtot, self.gmom, self.drag,
                          self.reg, self.boxp, self.heel_b, self.cargo_f,
                          wp.vec3(*self.cargo_point()), self.report])
        r = self.report.numpy().astype(np.float64)
        gz = r[5:5 + NHEEL]
        parts = dict(
            drag=W["drag"] * self.drag_scale(dyn) * r[1] / self.aref,
            stab=(-W["stab"] * self.stab_scale(dyn) * gz[1:].sum()
                  / (self.lref * (NHEEL - 1))),
            disp=W["disp"] * (r[0] / self.v_hull0 - 1.0) ** 2,
            lap=(W["lap"] * self.lap_scale(dyn) * r[2]
                 / (self.n_verts * self.lref ** 2)),
            edge=W["edge"] * r[3] / self.n_edges,
            box=W["box"] * r[4] / (self.n_verts * self.lref ** 2))
        e_dyn = float(self.dynloss.numpy()[0]) if dyn else float("nan")
        if dyn:
            parts["dyn"] = W["dyn"] * e_dyn
            parts["speed"] = W["speed"] * float(self.spdloss.numpy()[0])
            short = np.maximum(self.gz_floor_host[1:] - gz[1:], 0.0)
            parts["gzfloor"] = (W_GZ_FLOOR * float((short ** 2).sum())
                                / (self.lref ** 2 * (NHEEL - 1)))
        return dict(parts=parts, loss=float(self.loss.numpy()[0]),
                    v_tot=r[0], drag=r[1], lap=r[2], edge=r[3], box=r[4],
                    gz=gz, gz30=0.5 * (gz[GZ30] + gz[GZ30 + 1]),
                    v_sub=r[5 + NHEEL:5 + 2 * NHEEL],
                    v_err=abs(r[0] - self.v_hull0) / self.v_hull0,
                    dyn=e_dyn, s_wet=float(self.area.numpy()[0]),
                    u_term=self.cruise_speed(), lwl=self.lwl,
                    r_parts=self.breakdown(),
                    phase="dynamic" if dyn else "static",
                    asym=self.asymmetry(),
                    draft=self._d_host.copy(), bbox=self.extents())

    def cargo_point(self):
        """Where the payload sits, in the CURRENT sculpting frame.

        Anchored to the waterline, not to the mesh origin. The draft solve moves
        the hull's origin around underneath the design as the shape changes, so
        a fixed body-frame y would drift from "on the deck" to "at the waterline"
        mid-descent -- and act 2, whose vertices arrive already draft-shifted,
        would then be verifying a differently loaded ship than the one the loss
        designed. Detached: it is a constant within the step.
        """
        return (0.0, float(CARGO["y"] + self._d_host[0]), 0.0)

    def asymmetry(self):
        """Largest distance between a vertex and its mirrored partner, in metres."""
        p = self.x.numpy().astype(np.float64)
        q = p[self.pair_np].copy()
        q[:, 2] *= -1.0
        return float(np.linalg.norm(p - q, axis=1).max())

    def extents(self):
        p = self.x.numpy()
        return p.max(0) - p.min(0)

    def finite(self):
        return bool(np.isfinite(self.x.numpy()).all())

    def positions(self):
        return self.x.numpy()


# --------------------------------------------------------------------------- #
#  Act 2: forward rigid-body buoyancy, no tape
# --------------------------------------------------------------------------- #
class Floater:
    """One 3-DOF (heave + roll + pitch) body, integrated against the soft integrals."""

    __slots__ = ("y", "vy", "phi", "om", "theta", "omt", "g_body", "mass",
                 "m_acc", "inertia", "inertia_p", "c_heave", "c_roll", "c_pitch",
                 "peak", "peak_t", "name", "cap_t", "cap_hs", "beam",
                 "u", "px", "s_drag", "thrust", "u_acc", "u_n",
                 "s_wet", "lwl", "wave_r")

    def __init__(self, name, mass, inertia, inertia_p, g_body, beam):
        self.name = name
        self.mass = mass
        # Weight is the real mass; every ACCELERATION divides by the mass plus
        # the water the body has to shove aside with it. Both bodies get the
        # same factors, so the comparison stays a comparison of shapes.
        self.m_acc = mass * (1.0 + ADD_HEAVE)
        self.inertia = inertia * (1.0 + ADD_ROLL)
        self.inertia_p = inertia_p * (1.0 + ADD_PITCH)
        self.g_body = np.asarray(g_body, np.float64)
        self.y = 0.0
        self.vy = 0.0
        self.phi = 0.0
        self.om = 0.0
        self.theta = 0.0
        self.omt = 0.0
        self.peak = 0.0
        self.peak_t = 0.0
        self.cap_t = None                 # when it went over, and at what sea
        self.cap_hs = None
        self.beam = beam
        # Surge: speed, world x, the body's own shape factor, and the engine.
        # `thrust` is a state, not a constant -- it goes to zero for good when
        # the body turtles, because an inverted boat is not going anywhere.
        self.u = 0.0
        self.px = 0.0
        self.s_drag = 1.0
        self.s_wet = 1.0
        self.lwl = 2.0 * BLOB[0]
        self.wave_r = None            # R_wave sampled over speed, set at build
        self.thrust = 0.0
        self.u_acc = 0.0                  # for the mean speed over a run
        self.u_n = 0
        self.set_damping(0.0, 0.0, 0.0)   # the nominal floors, until calibrated

    def mean_u(self):
        return self.u_acc / max(self.u_n, 1)

    def set_damping(self, k_heave, k_roll, k_pitch):
        """Linear damping, as a fraction of critical for the stiffness this body
        ACTUALLY has -- measured off the pressure integral, not guessed.

        The fractions have not moved: roll on a real hull is 5-15% of critical,
        heave heavier, pitch heaviest at 30-50% because a hull pitching radiates
        waves off its whole length. What has moved is the stiffness they are
        quoted against. That used to be a stand-in, `m g x 0.6` for every body
        and both angles, and for a long hull's PITCH it is an order of magnitude
        low -- measured here, 1850 kNm/rad against a stand-in of 250 -- so a
        "40% of critical" that was really 16% left the boat ringing at the one
        period this sea has most of its energy at. The round blob happened to be
        quoted correctly (314 against 245), which is precisely why the ball
        looked rock stable while the boat did not.

        The old expressions survive as FLOORS, which is what an unstable body
        needs: negative roll stiffness has no critical damping to be a fraction
        of, and the blob then damps in roll exactly as it always did.
        """
        self.c_heave = 0.55 * 2.0 * math.sqrt(
            max(k_heave, RHO_W * GRAV * self.beam * self.beam) * self.m_acc)
        self.c_roll = 0.16 * 2.0 * math.sqrt(
            max(k_roll, self.mass * GRAV * 0.6) * self.inertia)
        self.c_pitch = 0.40 * 2.0 * math.sqrt(
            max(k_pitch, self.mass * GRAV * 0.6) * self.inertia_p)

    def capsized(self):
        return self.cap_t is not None


class Act2:
    """Two bodies, one sea. Act 2 kicks them; act 3 raises the sea on them."""

    def __init__(self, sculpt, verts_blob, verts_hull, ocean, spacing):
        d = sculpt.device
        self.s = sculpt
        self.device = d
        self.ocean = ocean
        self.spacing = spacing
        stack = np.stack([verts_blob, verts_hull]).astype(np.float32)
        self.x = wp.array(stack, dtype=wp.vec3, device=d)
        self.plane_n = wp.zeros(2, dtype=wp.vec3, device=d)
        self.plane_d = wp.zeros(2, dtype=float, device=d)
        self.vsub = wp.zeros(2, dtype=float, device=d)
        self.bmom = wp.zeros(2, dtype=wp.vec3, device=d)
        # The pressure integral's state: pose per body, and the wave heightfield
        # it is pressed by. The grid is world-space and re-sampled once a frame.
        self.rot = wp.zeros(2, dtype=wp.mat33, device=d)
        self.org = wp.zeros(2, dtype=wp.vec3, device=d)
        self.gpos = wp.zeros(2, dtype=wp.vec3, device=d)
        self.force = wp.zeros(2, dtype=wp.vec3, device=d)
        self.mom = wp.zeros(2, dtype=wp.vec3, device=d)
        self.wgrid = wp.zeros((2, GRID_NX, GRID_NZ), dtype=float, device=d)
        self.gorg = wp.zeros(2, dtype=wp.vec2, device=d)
        self.gstep = wp.zeros(2, dtype=wp.vec2, device=d)
        self.bodies = []
        frac = float(CARGO["frac"])
        cargo_p = np.array([0.0, float(CARGO["y"]), 0.0])
        self.g_solid = []
        self.half_len = []
        self.half_beam = []
        for i, v in enumerate((verts_blob, verts_hull)):
            vol = signed_volume(v, sculpt.faces)
            gs = _centroid(v, sculpt.faces)
            self.g_solid.append(gs)
            g = (gs + frac * cargo_p) / (1.0 + frac)
            mass = RHO_W * DENSITY * vol * (1.0 + frac)
            beam = float(np.abs(v[:, 2]).max())
            length = float(np.abs(v[:, 0]).max()) * 2.0
            self.half_len.append(0.5 * length)
            self.half_beam.append(beam)
            k = 0.42 * beam * 2.0
            kp = 0.30 * length
            self.bodies.append(Floater(("blob", "hull")[i], mass, mass * k * k,
                                       mass * kp * kp, g, beam))
        # Each body races on its OWN shape factor, taken from its own vertices
        # by the same integral the loss minimised. The blob's is the number the
        # thrust was set from, so the blob makes U_REF by construction and
        # everything the hull has over it is hull.
        sd = wp.zeros(2, dtype=float, device=d)
        sa = wp.zeros(2, dtype=float, device=d)
        wp.launch(rigid_drag, dim=(2, sculpt.n_faces), device=d,
                  inputs=[self.x, sculpt.tris, sculpt.eps,
                          wp.vec3(1.0, 0.0, 0.0), sd, sa])
        sdn, san = sd.numpy().astype(np.float64), sa.numpy().astype(np.float64)
        for i, b in enumerate(self.bodies):
            b.s_drag = float(sdn[i])
            b.s_wet = float(san[i])
            v = (verts_blob, verts_hull)[i]
            b.lwl = float(v[:, 0].max() - v[:, 0].min())
        self.wave_u, tab = self._wave_tables()
        for i, b in enumerate(self.bodies):
            b.wave_r = tab[i]
        self.t = 0.0
        self.gap_cap = None         # lead over the blob when the blob went over
        self.log = [[], []]
        # Act 3 state: the ramp, and the trailing window Hs is read off.
        self.ramp = None
        self.wind0 = wind_speed(ocean)
        self.hist = []
        self.hs = 0.0
        self.pitch_marks = {}
        self.trim_acc = [0.0, 0.0]
        self.trim_n = 0
        self.hs_max = 0.0
        # A trailing window on a ramping sea reports the sea of half a window
        # ago; 15 s is the compromise between that lag and having enough
        # samples for a sigma to mean anything.
        self.hs_window = 15.0

    def _upright_vb(self):
        """Submerged volume and centre of buoyancy, upright and level.

        The trial vertices arrive draft-shifted, so the upright waterplane is
        y = 0 and this is one launch with a flat plane. The pressure integral
        below is checked against the rho g V this produces.
        """
        self.plane_n.assign(np.tile(np.float32([0.0, 1.0, 0.0]), (2, 1)))
        self.plane_d.assign(np.zeros(2, np.float32))
        self.vsub.zero_()
        self.bmom.zero_()
        wp.launch(rigid_buoyancy, dim=(2, self.s.n_faces), device=self.device,
                  inputs=[self.x, self.s.tris, self.plane_n, self.plane_d,
                          self.s.eps, self.vsub, self.bmom])
        v = self.vsub.numpy().astype(np.float64)
        m = self.bmom.numpy().astype(np.float64)
        return v, [m[i] / max(v[i], 1e-6) for i in range(2)]

    def trim_cargo(self):
        """Shift the payload fore and aft until the ship floats LEVEL.

        A sculpted hull is not fore-aft symmetric, so upright B_x is not G_x and
        it would sit bow-down for ever -- which is what "constant negative
        pitch" was. Any naval architect's answer is to load her properly rather
        than to redesign her: solve x_cargo for G_x = B_x. This is trial-side
        reality only. The loss never sees it, and pitch is not a degree of
        freedom in the taped rollout anyway.
        """
        frac = float(CARGO["frac"])
        self.cargo_x = [0.0, 0.0]
        if frac <= 0.0:
            return self.cargo_x
        _, bxs = self._upright_vb()
        for i, b in enumerate(self.bodies):
            gs = self.g_solid[i]
            # The payload is only 15% of the mass, so shifting total G by 0.23 m
            # takes 1.8 m of cargo travel -- 0.35 of the half-length clamped
            # before it reached level. 0.6 still keeps it inside the hull.
            lim = 0.60 * self.half_len[i]
            x = float(np.clip(((1.0 + frac) * bxs[i][0] - gs[0]) / frac,
                              -lim, lim))
            self.cargo_x[i] = x
            b.g_body = (gs + frac * np.array([x, float(CARGO["y"]), 0.0])) \
                / (1.0 + frac)
        return self.cargo_x

    def _wave_tables(self, n=13, umax=6.0):
        """R_wave over speed for each racer, precomputed once at build_trial.

        The same Michell integral the loss was minimising, evaluated on each
        body's own vertices: what is optimised is what races. Sampling it and
        interpolating costs nothing per frame and is far more honest than the
        quadratic stand-in the taped rollout has to live with, because the
        trial can afford to see the whole hump.
        """
        d = self.device
        amp = wp.zeros((2, NTHETA, 2), dtype=float, device=d)
        sec3 = self.s.sec3dt.numpy().astype(np.float64)
        us = np.linspace(0.25, umax, n)
        tab = np.zeros((2, n))
        for j, u in enumerate(us):
            amp.zero_()
            wp.launch(rigid_wave_amp, dim=(2, NTHETA, self.s.n_faces), device=d,
                      inputs=[self.x, self.s.tris, self.s.eps,
                              GRAV / max(u * u, 1.0e-4), self.s.sec, amp])
            a = amp.numpy().astype(np.float64)
            coef = (W_WAVE["v"] * 4.0 * RHO_W * GRAV * GRAV
                    / (math.pi * max(u * u, 1.0e-4)))
            for b in range(2):
                tab[b, j] = coef * float(
                    ((a[b, :, 0] ** 2 + a[b, :, 1] ** 2) * sec3).sum())
        return us, tab

    def wave_at(self, b, u):
        """R_wave for body `b` at speed u, off its precomputed curve."""
        if b.wave_r is None:
            return 0.0
        return float(np.interp(abs(u), self.wave_u, b.wave_r,
                               left=0.0, right=b.wave_r[-1]))

    def _grid_axes(self):
        """Origin and spacing of each body's probe grid.

        Re-centred every frame on the body's CURRENT x, because in act 3 the
        bodies are under power and go somewhere: `sample_height` takes world
        coordinates, so following a racer is just moving the grid origin. The
        span is generous enough that a body still finds grid under it when it is
        over on its side, and the lookup clamps at the edge rather than falling
        off it. Sway is still not a degree of freedom, so z is fixed.
        """
        org = np.zeros((2, 2), np.float32)
        stp = np.zeros((2, 2), np.float32)
        for i in range(2):
            ex = 1.4 * 2.0 * self.half_len[i]
            ez = 1.6 * 2.0 * self.half_beam[i]
            org[i] = (self.bodies[i].px - 0.5 * ex,
                      (i - 0.5) * self.spacing - 0.5 * ez)
            stp[i] = (ex / (GRID_NX - 1), ez / (GRID_NZ - 1))
        self.gorg.assign(org)
        self.gstep.assign(stp)
        return org, stp

    def _sample_grid(self):
        """Read the wave field under both bodies -- 2 x GRID_NX x GRID_NZ probes.

        Once a FRAME, not once a substep: `sample_height` reads a CPU mirror of
        a field that only advances when the renderer does, so a second read
        inside the same frame returns the same numbers at twice the price.
        """
        org, stp = self._grid_axes()
        g = np.zeros((2, GRID_NX, GRID_NZ), np.float32)
        for i in range(2):
            x0, z0 = float(org[i][0]), float(org[i][1])
            dx, dz = float(stp[i][0]), float(stp[i][1])
            for a in range(GRID_NX):
                for c in range(GRID_NZ):
                    g[i, a, c] = self._height(x0 + a * dx, z0 + c * dz)
        self.wgrid.assign(g)

    def _calm_grid(self):
        """Flat water at y = 0, for the two startup assertions."""
        self.wgrid.zero_()
        self._grid_axes()

    def _forces(self, poses):
        """Pressure force and moment on both bodies at the given (roll, pitch, heave).

        Returns world-frame F and M about each body's own G, in newtons and
        newton-metres. One launch for both bodies.
        """
        rot = np.zeros((2, 3, 3), np.float32)
        org = np.zeros((2, 3), np.float32)
        gp = np.zeros((2, 3), np.float32)
        for i, (phi, theta, y) in enumerate(poses):
            r = rot_np(phi, theta)
            o = np.array([self.bodies[i].px, y, (i - 0.5) * self.spacing])
            rot[i] = r
            org[i] = o
            gp[i] = r @ self.bodies[i].g_body + o
        self.rot.assign(rot)
        self.org.assign(org)
        self.gpos.assign(gp)
        self.force.zero_()
        self.mom.zero_()
        wp.launch(wave_pressure, dim=(2, self.s.n_faces), device=self.device,
                  inputs=[self.x, self.s.tris, self.rot, self.org, self.gpos,
                          self.wgrid, self.gorg, self.gstep, GRID_NX, GRID_NZ,
                          RHO_W * GRAV, self.s.eps, self.force, self.mom])
        return (self.force.numpy().astype(np.float64),
                self.mom.numpy().astype(np.float64))

    def calibrate_damping(self):
        """Measure each body's real restoring stiffness, and damp it against THAT.

        Four launches of the pressure integral in calm water: lift the body
        10 cm for the heave stiffness, heel it 5 degrees for the roll stiffness,
        trim it 5 degrees for the pitch stiffness. Cheap, exact for this model,
        and the thing the old plane model was never asked for -- which is how a
        stand-in stiffness ended up in the damping in the first place. Must run
        AFTER the cargo trim, since every angular stiffness is taken about G.
        Returns (k_heave, k_roll, k_pitch) per body, in N/m and Nm/rad.
        """
        self._calm_grid()
        dy, da = 0.10, math.radians(5.0)
        f0, m0 = self._forces([(0.0, 0.0, 0.0)] * 2)
        fh, _ = self._forces([(0.0, 0.0, dy)] * 2)
        _, mr = self._forces([(da, 0.0, 0.0)] * 2)
        _, mp = self._forces([(0.0, da, 0.0)] * 2)
        out = []
        for i, b in enumerate(self.bodies):
            k = ((f0[i][1] - fh[i][1]) / dy,
                 -(mr[i][0] - m0[i][0]) / da,
                 -(mp[i][2] - m0[i][2]) / da)
            b.set_damping(*k)
            out.append(k)
        return out

    def check_buoyancy(self):
        """Calm water, upright: the pressure integral MUST be rho g V_sub.

        Pressure over the closed wetted surface, with the free surface as the
        p = 0 lid, IS Archimedes -- the divergence theorem says so. If this
        drifts, the model has quietly stopped being hydrostatics, and every
        number act 2 and act 3 report afterwards is decoration. The reference
        V_sub is the tet sum against a flat plane, a completely different code
        path. Returns the relative error per body.
        """
        v, _ = self._upright_vb()
        self._calm_grid()
        f, _ = self._forces([(0.0, 0.0, 0.0)] * 2)
        out = []
        for i, b in enumerate(self.bodies):
            ref = RHO_W * GRAV * float(v[i])
            err = f[i][1] / ref - 1.0
            out.append(err)
            assert abs(err) < 0.01, (
                f"{b.name}: the calm-water pressure integral gives "
                f"{f[i][1] / 1.0e3:.1f} kN against rho g V = {ref / 1.0e3:.1f} kN, "
                f"{100.0 * err:+.2f}% off. The wetted surface is not closing.")
        return out

    def check_pitch_sign(self):
        """Bow up must push the bow DOWN. Asserted, not assumed.

        Roll's sign was verified against the expression it replaced; pitch never
        had a predecessor to check against, and a sign error there is exactly
        the kind of thing that looks like "the boat has insane pitch" rather
        than like a crash. One launch of the real integrator, calm water, +10
        deg trim: bow up moves the buoyancy centre aft of G, so M.z < 0.
        """
        self._calm_grid()
        _, m = self._forces([(0.0, math.radians(10.0), 0.0)] * 2)
        out = []
        for i, b in enumerate(self.bodies):
            out.append(m[i][2] / 1000.0)
            assert m[i][2] < 0.0, (
                f"pitch moment is NOT restoring for the {b.name}: bow up 10 deg "
                f"gives Mz = {m[i][2] / 1000.0:+.1f} kNm, expected negative. "
                f"Check the rot_np composition order.")
        return out

    def kick(self, deg=25.0):
        for b in self.bodies:
            b.phi = math.radians(deg)
            b.om = 0.0
            b.vy = 0.0
            b.y = 0.0
            b.theta = 0.0
            b.omt = 0.0
            b.peak = abs(b.phi)
            b.cap_t = None
            b.cap_hs = None
            b.u = 0.0
            b.px = 0.0
            b.thrust = 0.0            # act 2 is a pure kick test: engines off
            b.u_acc = 0.0
            b.u_n = 0
        self.t = 0.0
        self.gap_cap = None
        self.log = [[], []]

    def begin_act3(self, seconds, lo=ACT3_WIND0, hi=ACT3_WIND1):
        """No kick: upright, calm, under power, and then the sea comes up.

        Act 2 asks who rights. Act 3 asks who gets there -- both bodies open the
        throttle to the SAME fixed thrust and motor into a rising sea, so the
        scoreboard is a race and a survival trial at once.
        """
        self.kick(0.0)
        for b in self.bodies:
            b.thrust = float(THRUST["N"])
        self.ramp = (float(lo), float(hi), float(seconds))
        self.hist = []
        self.hs = 0.0
        self.pitch_marks = {}
        self.trim_acc = [0.0, 0.0]
        self.trim_n = 0
        self.hs_max = 0.0
        set_wind_speed(self.ocean, lo)

    def ramp_done(self):
        return self.ramp is not None and self.t >= self.ramp[2]

    def _height(self, x, z):
        """Water height at one world point. The whole ocean interface the trial uses.

        Without a renderer -- the CPU-only paths -- it is two travelling waves
        instead, spatial and not just temporal, or a heightfield model would
        have nothing to bridge.
        """
        if self.ocean is None:
            return (0.55 * math.sin(0.62 * self.t - 0.55 * x)
                    + 0.22 * math.sin(1.70 * self.t + 1.1 - 0.35 * z))
        return float(self.ocean.sample_height(x, z))

    def _sea_state(self, dt):
        """Ramp the real ocean, then read Hs back off the water it produced.

        The bodies feel the sea that is on screen -- one field, sampled. Hs is
        4 sigma over a trailing window, the standard definition, MEASURED
        rather than commanded: the wind is what is ramped, and the sea it
        raises is whatever the spectrum decides it should be.
        """
        if self.ramp is None:
            return
        lo, hi, secs = self.ramp
        lam = min(self.t / max(secs, 1e-6), 1.0)
        want = lo + (hi - lo) * lam
        # Only when it has actually moved: every change regenerates the spectra.
        if abs(want - wind_speed(self.ocean)) > 0.05:
            set_wind_speed(self.ocean, want)
        # Probed BETWEEN the racers rather than at the origin: they sail away
        # from where they started, and Hs has to be the sea they are in.
        self.hist.append(self._height(
            0.5 * (self.bodies[0].px + self.bodies[1].px), 0.0))
        n = max(int(self.hs_window / max(dt, 1e-6)), 8)
        if len(self.hist) > n:
            del self.hist[:len(self.hist) - n]
        if len(self.hist) >= 8:
            self.hs = 4.0 * float(np.std(np.asarray(self.hist)))
            # A survivor is credited with the WORST sea it has come through, not
            # with whatever the sea happens to be doing when you look: Hs
            # wanders, and "survived to 0.25 m" after riding out 0.6 m is not a
            # report, it is an accident of sampling time.
            self.hs_max = max(self.hs_max, self.hs)
        # Mean trim over the calm opening: this is the "does she float level"
        # number, and it is separate from how much she pitches in a seaway.
        if self.hs < 0.5:
            self.trim_n += 1
            for i, b in enumerate(self.bodies):
                self.trim_acc[i] += b.theta
        # Peak pitch reached by the time the sea got this big -- the number the
        # "insane pitch" complaint is really about.
        for thr in (1.0, 2.5):
            if self.hs >= thr and thr not in self.pitch_marks:
                self.pitch_marks[thr] = [math.degrees(b.peak_t)
                                         for b in self.bodies]

    def step(self, dt, substeps=2):
        self._sea_state(dt)
        self._sample_grid()
        h = dt / substeps
        for _ in range(substeps):
            f, m = self._forces([(b.phi, b.theta, b.y) for b in self.bodies])
            for i, b in enumerate(self.bodies):
                # Heave takes F.y straight; the pressure integral already IS
                # rho g V_sub in calm water, so the equilibrium draft and the
                # heave stiffness are the ones the old model had.
                fy = f[i][1] - b.mass * GRAV - b.c_heave * b.vy
                b.vy += h * fy / b.m_acc
                b.y += h * b.vy
                # Generalised torques, not raw components. Roll is the outer
                # rotation so its axis is world +x; pitch turns about the body's
                # own +z, which roll has already tilted -- and getting that
                # right is what keeps a body that is over on its beam ends, or
                # inverted, from being trimmed by a moment of the wrong sign.
                pz = np.array([0.0, -math.sin(b.phi), math.cos(b.phi)])
                b.om += h * (m[i][0] - b.c_roll * b.om) / b.inertia
                b.phi += h * b.om
                b.omt += h * (float(m[i] @ pz) - b.c_pitch * b.omt) / b.inertia_p
                b.theta += h * b.omt
                # Surge against the FULL resistance law -- form, friction and
                # wave-making, each on this body's own integrals. The race is
                # run on the model the shape was optimised against.
                rf, rr = resist_visc(b.u, b.s_drag, b.s_wet, b.lwl)
                sgn = 1.0 if b.u >= 0.0 else -1.0
                fu = b.thrust - sgn * (rf + rr + self.wave_at(b, b.u))
                b.u += h * fu / (b.mass * (1.0 + ADD_SURGE))
                b.px += h * b.u
                b.u_acc += b.u
                b.u_n += 1
                b.peak = max(b.peak, abs(b.phi))
                b.peak_t = max(b.peak_t, abs(b.theta))
                if b.cap_t is None and abs(b.phi) > math.radians(CAPSIZE_DEG):
                    # Marked, and then left alone: the integrals find the
                    # inverted equilibrium by themselves, which is the picture
                    # worth showing. The engine dies with her -- she keeps the
                    # way she had on and drifts astern of the survivor.
                    b.cap_t = self.t
                    b.cap_hs = self.hs
                    b.thrust = 0.0
                    if b.name == "blob":
                        self.gap_cap = self.bodies[1].px - self.bodies[0].px
            self.t += h
        for i, b in enumerate(self.bodies):
            self.log[i].append(math.degrees(b.phi))
        if MOTION_CSV:
            with open(MOTION_CSV, "a") as fh:
                fh.write(f"{self.t:.4f} {self.hs:.4f} " + " ".join(
                    f"{math.degrees(b.phi):.4f} {math.degrees(b.theta):.4f} "
                    f"{b.y:.4f} {b.u:.4f} {b.px:.3f}"
                    for b in self.bodies) + "\n")

    def focus(self):
        """Where the camera should be looking: the pack, or the leader once
        someone is out of it -- biased HALF the way back toward the wreck, out
        to 14 m, so the story stays in one frame. Half rather than a third
        because the two lanes are 11 m apart in z: a camera parked on the
        leader puts the wreck in the near corner rather than astern of her.
        """
        alive = [b for b in self.bodies if not b.capsized()]
        if not alive:
            return sum(b.px for b in self.bodies) / len(self.bodies), 0.0
        lead = sum(b.px for b in alive) / len(alive)
        lost = [b.px for b in self.bodies if b.capsized()]
        if lost:
            lead -= min(0.50 * (lead - min(lost)), 14.0)
        return lead, 0.0

    def scoreboard(self):
        out = []
        for b in self.bodies:
            if b.capsized():
                out.append(f"{b.name:<5} capsized at Hs = {b.cap_hs:.2f} m "
                           f"(t = {b.cap_t:.0f} s)   {b.px:6.1f} m made good")
            else:
                out.append(f"{b.name:<5} survived to Hs = {self.hs_max:.2f} m"
                           f"   {b.px:6.1f} m made good")
        return out

    def race_report(self):
        """Speed made good, which is what the drag integral was ever about."""
        out = []
        if THRUST["N"] <= 0.0:
            return out
        for b in self.bodies:
            ut = terminal_speed(b.s_drag, b.s_wet, b.lwl,
                                lambda u: self.wave_at(b, u))
            rf, rr = resist_visc(ut, b.s_drag, b.s_wet, b.lwl)
            rw = self.wave_at(b, ut)
            tot = max(rf + rr + rw, 1e-9)
            out.append(f"{b.name:<5} {b.u:5.2f} m/s ({1.944 * b.u:4.1f} kn) now,"
                       f" {b.mean_u():5.2f} m/s mean, {b.px:6.1f} m in "
                       f"{self.t:.0f} s   -> {ut:.2f} m/s flat water "
                       f"(form {100.0 * rf / tot:.0f}% fric "
                       f"{100.0 * rr / tot:.0f}% wave {100.0 * rw / tot:.0f}%)")
        bl, hl = self.bodies
        r = hl.mean_u() / max(bl.mean_u(), 1e-6)
        line = (f"hull averaged {hl.mean_u():.2f} m/s against the blob's "
                f"{bl.mean_u():.2f} ({r:.2f}x) at equal thrust")
        if self.gap_cap is not None:
            line += f", {self.gap_cap:.0f} m ahead when the blob went over"
        out.append(line)
        return out

    def pitch_report(self):
        out = []
        if self.trim_n:
            mt = [math.degrees(a / self.trim_n) for a in self.trim_acc]
            out.append(f"mean trim in the calm start (Hs < 0.5 m): blob "
                       f"{mt[0]:+5.2f} deg, hull {mt[1]:+5.2f} deg")
        for thr in sorted(self.pitch_marks):
            bl, hl = self.pitch_marks[thr]
            out.append(f"peak |pitch| by Hs {thr:.1f} m: blob {bl:5.1f} deg, "
                       f"hull {hl:5.1f} deg")
        return out

    def pose(self, i):
        b = self.bodies[i]
        return rot_mat(b.phi, b.theta), wp.vec3(0.0, float(b.y), 0.0)


def _linear_rgb(hexcol):
    """sRGB hex -> linear float3, which is the space a vertex colour lands in."""
    c = np.array([(hexcol >> 16) & 255, (hexcol >> 8) & 255, hexcol & 255],
                 np.float32) / 255.0
    return (c ** 2.2).astype(np.float32)


def hull_paint(verts, waterline, topside, raise_frac=0.0):
    """Two-tone per-vertex colour, split at the design waterline.

    Paint, not physics: the line is struck once, from the shape and draft at
    the time, and then it rides with the hull. That is what makes a capsize
    legible -- an inverted body shows its bottom -- and it is also just what
    boats look like.

    `raise_frac` lifts the line toward the crown by that fraction of the
    freeboard. A ROUND body split at its equator shows a half-dark,
    half-light ball whichever way up it settles, so whether the dark half
    faces the camera is luck -- and the settled roll sign genuinely varies
    run to run. Carrying the antifouling most of the way up leaves a small
    topside cap whose absence reads from every azimuth once the body
    turtles. The shaped hull keeps the true waterline split (0.0).
    """
    line = waterline
    if raise_frac > 0.0:
        line = waterline + raise_frac * (float(verts[:, 1].max()) - waterline)
    c = np.empty((len(verts), 3), np.float32)
    above = verts[:, 1] > line
    c[above] = _linear_rgb(topside)
    c[~above] = _linear_rgb(ANTIFOUL)
    return c


def wind_speed(ocean):
    """The one knob act 3 turns. It lives on `params`, not on the Ocean itself."""
    return float(ocean.params.wind_speed) if ocean is not None else 0.0


def set_wind_speed(ocean, v):
    """Live wind change: the renderer regenerates the spectra next frame and the
    sea morphs into the new state -- height, length and steepness together."""
    if ocean is not None:
        ocean.set_wind(float(v), float(ocean.params.wind_theta))


def _centroid(v, faces):
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    vol = np.einsum("ij,ij->i", a, np.cross(b, c)) / 6.0
    cen = (a + b + c) * 0.25
    return (cen * vol[:, None]).sum(0) / vol.sum()


# --------------------------------------------------------------------------- #
#  A mesh the GPU writes directly
# --------------------------------------------------------------------------- #
class LiveMesh:
    """A fixed-topology Mesh whose vertex buffers Warp fills, zero-copy if it can.

    Same shape as every other warp demo's interop block: try the export, and if
    anything at all refuses, fall back to `update_attribute` and say so once.
    """

    def __init__(self, verts, faces, material, device, colors=None):
        self.device = device
        self.n = len(verts)
        g = tp.BufferGeometry()
        g.set_attribute("position", np.ascontiguousarray(verts, np.float32))
        nrm = np.zeros_like(verts, np.float32)
        nrm[:, 1] = 1.0
        g.set_attribute("normal", nrm)
        if colors is not None:
            # HERE, before the first render and therefore before arming: adding
            # an attribute later rebuilds the renderer's record for this mesh,
            # and doing that under a live interop registration is how you get a
            # producer writing into a buffer nothing reads any more. Colours are
            # static; only position and normal are ever producer-written.
            g.set_attribute("color", np.ascontiguousarray(colors, np.float32))
        g.set_index(np.ascontiguousarray(faces.reshape(-1), np.uint32))
        g.compute_vertex_normals()
        self.geo = g
        self.mesh = tp.Mesh(g, material)
        self.mesh.cast_shadow = True
        self.mesh.receive_shadow = True
        self.pos = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.nrm = wp.zeros(self.n, dtype=wp.vec3, device=device)
        self.vk = None
        self.route = "host copy"

    def arm(self, renderer):
        if NO_INTEROP or not self.device.is_cuda:
            return
        if not hasattr(renderer, "enable_vertex_interop"):
            return
        try:
            from threepp.cuda_interop import VkInteropArray
            h = renderer.enable_vertex_interop(self.mesh, self._on_frame)
            if h is None:
                raise RuntimeError("no external-memory export for this mesh yet")
            self.vk = [VkInteropArray(hh[0], hh[1], wp.vec3, self.n, self.device)
                       for hh in h]
            self.route = "zero-copy CUDA -> Vulkan"
        except Exception as e:                      # noqa: BLE001 - any refusal
            print(f"  note: vertex interop did not arm ({e}) -- host copy route")
            try:
                renderer.disable_vertex_interop(self.mesh)
            except Exception:
                pass
            self.vk = None

    def _on_frame(self):
        # Post-fence, pre-record, inside render(). The synchronize is the whole
        # contract: nothing else orders this write against the frame reading it.
        if self.vk is not None:
            wp.copy(self.vk[0].array, self.pos)
            wp.copy(self.vk[1].array, self.nrm)
            wp.synchronize_device(self.device)

    def publish(self, tris, n_faces):
        """Recompute normals for whatever is in `pos`, then hand it over."""
        self.nrm.zero_()
        wp.launch(accum_normals, dim=n_faces, device=self.device,
                  inputs=[self.pos, tris, self.nrm])
        wp.launch(normalize_vec3, dim=self.n, device=self.device, inputs=[self.nrm])
        if self.vk is None:
            wp.synchronize_device(self.device)
            self.geo.update_attribute("position", self.pos.numpy())
            self.geo.update_attribute("normal", self.nrm.numpy())

    def repaint(self, colors):
        """Re-strike the boot stripe. UPDATE, never add -- see __init__."""
        self.geo.update_attribute("color", np.ascontiguousarray(colors, np.float32))

    def release(self, renderer):
        if self.vk is None:
            return
        for a in self.vk:
            a.close()
        self.vk = None
        try:
            renderer.disable_vertex_interop(self.mesh)
        except Exception:
            pass


# --------------------------------------------------------------------------- #
#  Reporting
# --------------------------------------------------------------------------- #
def print_metrics(tag, m):
    print(f"  {tag:<8} loss {m['loss']:9.5f}   S {m['drag']:7.4f} / A "
          f"{m['s_wet']:6.2f} m^2 -> {m['u_term']:.2f} m/s   "
          f"GZ(30) {m['gz30']:+7.4f} m   V {m['v_tot']:8.3f} m^3 "
          f"(err {100.0 * m['v_err']:.2f}%)   draft {m['draft'][0]:+.3f} m   "
          f"LxBxD {m['bbox'][0]:.2f} x {m['bbox'][2]:.2f} x {m['bbox'][1]:.2f} m")


def print_resistance(tag, m):
    """The three terms at cruise, in newtons and as shares. The headline table."""
    rf, rr, rw, u = m["r_parts"]
    tot = max(rf + rr + rw, 1e-9)
    fr = u / math.sqrt(GRAV * max(m["lwl"], 1e-6))
    print(f"  {tag:<8} at {u:.2f} m/s (Fr {fr:.3f}): form {rf:7.0f} N "
          f"({100.0 * rf / tot:4.1f}%)  friction {rr:6.0f} N "
          f"({100.0 * rr / tot:4.1f}%)  wave {rw:7.0f} N "
          f"({100.0 * rw / tot:4.1f}%)   total {tot:7.0f} N")


def print_parts(m):
    print("           terms     " + "   ".join(f"{k} {v:.5f}"
                                               for k, v in m["parts"].items()))


def act3_verdict(act):
    """The headline: how much more weather the sculpted hull took."""
    blob, hull = act.bodies
    hb = blob.cap_hs if blob.capsized() else act.hs_max
    hh = hull.cap_hs if hull.capsized() else act.hs_max
    tag = "" if hull.capsized() else " (still upright at ramp end)"
    if hb <= 1e-6:
        return f"blob Hs {hb:.2f} m, hull Hs {hh:.2f} m{tag}"
    return (f"Hs_hull / Hs_blob = {hh / hb:.2f}x  "
            f"({hh:.2f} m vs {hb:.2f} m){tag}")


def gz_curve(m):
    return "   ".join(f"{d:+.0f}: {g:+.3f}"
                      for d, g in zip(HEELS_DEG, m["gz"]))


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    wp.init()
    device = wp.get_device("cpu") if FORCE_CPU else wp.get_preferred_device()
    print(f"warp device: {device}")

    sculpt = Sculptor(device)
    print(f"hull mesh: {sculpt.n_verts} vertices, {sculpt.n_faces} faces, "
          f"{sculpt.n_edges} edges")
    print(f"blob: {2 * BLOB[0]:.1f} x {2 * BLOB[2]:.1f} x {2 * BLOB[1]:.1f} m "
          f"(L x B x D), V {sculpt.v_hull0:.2f} m^3, displaces "
          f"{sculpt.v_disp:.2f} m^3 = {sculpt.mass / 1000.0:.1f} t, "
          f"waterline eps {sculpt.eps:.3f} m")

    dyn_txt = "off" if NO_DYN else (
        f"after step {int(DYN_AFTER * STEPS)}, T={sculpt.T} x {DYN_DT:.4f} s "
        f"= {sculpt.T * DYN_DT:.1f} s horizon, w {W['dyn']:.2f}")
    sym_txt = "off" if NO_SYM else f"on (rest residual {sculpt.sym_res0:.1e} m)"
    print(f"config: {'CLASSIC' if CLASSIC else 'advanced'}   dyn {dyn_txt}")
    print(f"        symmetry {sym_txt}   cargo {100.0 * sculpt.cargo_f:.0f}% "
          f"@ y={sculpt.cargo_p[1]:+.2f} m")

    # Resistance model: calibrate the wave scale off the blob, then size the
    # engine so the blob still makes U_REF under the FULL law, then prove the
    # wave term is doing what it claims before anything is optimised against it.
    sculpt.forward()
    sculpt.lwl = max(float(sculpt.extents()[0]), 1.0)
    calibrate_wave(sculpt)
    set_thrust(sculpt)
    print(f"        wave scale {W_WAVE['v']:.4g} (blob {100.0 * WAVE_SHARE:.0f}% "
          f"wave at its own hump), thrust {THRUST['N']:.0f} N so the blob still "
          f"makes {U_REF:.1f} m/s under form + friction + wave")
    ok = wave_anchors(sculpt)
    if not ok:
        print("  WAVE ANCHORS FAILED -- the Michell term is not fit to shape a "
              "hull; not proceeding.")
        return 1
    if WAVE_ANCHORS:
        return 0
    sculpt.refresh_cruise()

    m0 = sculpt.measure()
    print("initial:")
    print_metrics("blob", m0)
    print_resistance("blob", m0)
    print(f"           GZ curve  {gz_curve(m0)}")

    if TUNE:
        return run_tune(sculpt, m0)
    return run_scene(sculpt, m0, device)


def run_tune(sculpt, m0):
    """No renderer at all: the weight-tuning loop."""
    t0 = time.perf_counter()
    for i in range(STEPS):
        sculpt.step()
        if (i + 1) % 200 == 0:
            m = sculpt.measure()
            print(f"  step {i + 1:5d}", end="")
            print_metrics("", m)
            if not sculpt.finite():
                print("  NON-FINITE vertices")
                return 1
    dt = time.perf_counter() - t0
    m1 = sculpt.measure()
    print("final:")
    print_metrics("hull", m1)
    print_resistance("hull", m1)
    print(f"           GZ curve  {gz_curve(m1)}")
    print_parts(m1)
    print(f"  {STEPS} steps in {dt:.2f} s ({1000.0 * dt / STEPS:.2f} ms/step)")
    e1, peak1, fin1 = sculpt.roll_trace()
    print(f"  rollout   roll energy {e1:.4f} (tail, kick^2 units)   "
          f"peak |roll| {peak1:.1f} deg   final |roll| {fin1:.2f} deg   "
          f"[{'dyn phase ON' if not NO_DYN else 'static only'}]")
    print(f"  symmetry  max |x - mirror x_pair| {m1['asym']:.2e} m"
          f"   cargo {100.0 * sculpt.cargo_f:.0f}% @ y={sculpt.cargo_p[1]:+.2f}")
    print(f"  GZ(30) {m0['gz30']:+.4f} -> {m1['gz30']:+.4f} m   "
          f"drag {m0['drag']:.4f} -> {m1['drag']:.4f} m^2 "
          f"({100.0 * (m1['drag'] / m0['drag'] - 1.0):+.1f}%)   "
          f"V err {100.0 * m1['v_err']:.2f}%")
    return 0


SUN_DIR = np.array([0.52, 0.75, 0.41], np.float32)
SUN_DIR /= np.linalg.norm(SUN_DIR)


# The procedural sky lives in warp_common.sky_env now (the prop vortex is its
# third caller); the maths is the one that was here.


def build_scene(canvas):
    renderer = tp.VulkanRenderer(canvas)
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    renderer.tone_mapping_exposure = 0.80
    renderer.shadow_map_enabled = True

    scene = tp.Scene()
    env = sky_env(SUN_DIR)
    scene.background = env
    scene.environment = env
    sun = tp.DirectionalLight(0xfff3dc, 2.6)
    sun.position.set(*(SUN_DIR * 40.0))
    sun.cast_shadow = True
    scene.add(sun)

    # A YOUNG sea, not an ocean swell. fetch 30 km puts tens-of-metres
    # wavelengths under a 9.6 m boat, which is a bathtub duck in a swell: it
    # just follows the surface. A short fetch is the JONSWAP knob for exactly
    # this -- steeper, shorter chop on the scale of the hull, so the waves are
    # something she has to work against rather than ride.
    ocean = tp.Ocean(size=600.0, wind_speed=ACT3_WIND0, choppiness=0.5,
                     fetch=12e3)
    scene.add(ocean)

    floor = tp.Mesh(tp.PlaneGeometry(600.0, 600.0), standard_material(0x04101a))
    floor.rotate_x(-math.pi / 2)
    floor.position.y = -16.0
    floor.name = "seafloor"          # the film sinks it out of the murk's reach
    scene.add(floor)

    camera = tp.PerspectiveCamera(42, canvas.aspect(), 0.1, 2000)
    camera.position.set(10.5, 3.4, 10.5)
    camera.look_at(0.0, 0.2, 0.0)
    return renderer, scene, camera, ocean


def run_scene(sculpt, m0, device):
    canvas = tp.Canvas("threepp - differentiable hull sculpting",
                       width=WIN_W, height=WIN_H, vsync=False, headless=HEADLESS)
    renderer, scene, camera, ocean = build_scene(canvas)

    # The hue lives in the vertex colours, so the materials are white and just
    # multiply through. The ghost keeps its own tint until it becomes a body.
    hull_mat = standard_material(0xffffff, roughness=0.35, metalness=0.08)
    hull_mat.vertex_colors = True
    ghost_mat = standard_material(0x64d0ff, roughness=0.5, metalness=0.0)
    ghost_mat.transparent = True
    ghost_mat.opacity = 0.22
    # Set HERE, not when the ghost becomes a body: flipping vertex_colors on a
    # material that has already been drawn leaves it on a shader variant
    # compiled without the vertex-colour path, and the blob then renders its
    # topsides whichever way up it is floating.
    ghost_mat.vertex_colors = True

    d_rest = float(m0["draft"][0])
    live = LiveMesh(sculpt.rest, sculpt.faces, hull_mat, device,
                    hull_paint(sculpt.rest, d_rest, TOPSIDE_HULL))
    live.mesh.position.set(0.0, 0.0, 0.0)
    scene.add(live.mesh)

    # The ghost of the initial blob doubles as act 2's second body. It has to be
    # VISIBLE for the arming render -- a mesh the renderer has never drawn has no
    # BLAS record, and so nothing to export.
    blob_mesh = LiveMesh(sculpt.rest, sculpt.faces, ghost_mat, device,
                         hull_paint(sculpt.rest, d_rest, TOPSIDE_BLOB,
                                    raise_frac=0.55))
    blob_mesh.mesh.position.set(0.0, 0.0, 0.0)
    scene.add(blob_mesh.mesh)

    act2 = None
    ui = (tp.ImguiContext(canvas, renderer)
          if (tp.HAS_IMGUI and not HEADLESS and not FILM) else None)

    state = dict(mode="sculpt", paused=False, ghost=False, step_ms=0.0,
                 render_ms=0.0, m=m0, m_final=None, frames=0, shot3=0)

    def publish_sculpt():
        wp.launch(emit_pose, dim=sculpt.n_verts, device=device,
                  inputs=[sculpt.x, rot_x(0.0),
                          wp.vec3(0.0, -float(sculpt._d_host[0]), 0.0), live.pos])
        live.publish(sculpt.tris, sculpt.n_faces)

    def publish_ghost():
        blob_mesh.pos.assign(
            (sculpt.rest - np.array([0.0, m0["draft"][0], 0.0], np.float32)
             ).astype(np.float32))
        blob_mesh.publish(sculpt.tris, sculpt.n_faces)

    def publish_act2():
        for i, mesh in enumerate((blob_mesh, live)):
            rot, off = act2.pose(i)
            wp.launch(emit_pose_row, dim=sculpt.n_verts, device=device,
                      inputs=[act2.x, i, rot, off, mesh.pos])
            # Surge moves the NODE, not the vertices: the vertex buffer stays
            # centred on the body so its bounds and its shadow stay honest as
            # she runs kilometres down the course.
            mesh.mesh.position.x = act2.bodies[i].px
            mesh.publish(sculpt.tris, sculpt.n_faces)

    # The mesh's renderer record only exists once it has been drawn.
    publish_sculpt()
    publish_ghost()
    renderer.render(scene, camera)
    live.arm(renderer)
    blob_mesh.arm(renderer)
    blob_mesh.mesh.visible = False
    print(f"vertex route: {live.route} / {blob_mesh.route}")

    def build_trial():
        nonlocal act2
        if act2 is not None:
            return
        state["m_final"] = sculpt.measure()
        blob = (sculpt.rest - np.array([0.0, m0["draft"][0], 0.0], np.float32))
        hull = (sculpt.positions()
                - np.array([0.0, state["m_final"]["draft"][0], 0.0], np.float32))
        act2 = Act2(sculpt, blob, hull, ocean, SPACING)
        for i, x in enumerate(act2.trim_cargo()):
            if CARGO["frac"] > 0.0:
                where = "fwd" if x >= 0.0 else "aft"
                print(f"  {act2.bodies[i].name:<5} cargo trimmed "
                      f"{x:+.2f} m {where} to float level")
        for i, k in enumerate(act2.calibrate_damping()):
            b = act2.bodies[i]
            print(f"  {b.name:<5} stiffness heave {k[0] / 1e3:7.0f} kN/m   roll "
                  f"{k[1] / 1e3:+8.0f} kNm/rad   pitch {k[2] / 1e3:8.0f} kNm/rad"
                  f"   T_pitch {2.0 * math.pi * math.sqrt(b.inertia_p / max(k[2], 1.0)):.2f} s")
        eb = act2.check_buoyancy()
        print(f"  calm-water check: pressure integral vs rho g V within "
              f"{100.0 * eb[0]:+.2f}% / {100.0 * eb[1]:+.2f}% (blob / hull)")
        mz = act2.check_pitch_sign()
        print(f"  pitch sign check (bow up 10 deg): Mz = {mz[0]:+.0f} / "
              f"{mz[1]:+.0f} kNm (blob / hull), both restoring")
        print(f"  wave probe grid: {GRID_NX} x {GRID_NZ} per body, "
              f"{2.0 * 1.4 * act2.half_len[1] / (GRID_NX - 1):.2f} m along the "
              f"hull")
        # Re-strike both boot stripes for the poses that are about to be sailed:
        # these vertex arrays are already draft-shifted, so the waterline is
        # y = 0 and the paint lands where each body actually floats.
        blob_mesh.repaint(hull_paint(blob, 0.0, TOPSIDE_BLOB, raise_frac=0.55))
        live.repaint(hull_paint(hull, 0.0, TOPSIDE_HULL))
        # The ghost stops being a ghost: same Material object, opaque now, so
        # nothing structural changes under the live interop registration.
        ghost_mat.transparent = False
        ghost_mat.opacity = 1.0
        ghost_mat.color = 0xffffff
        blob_mesh.mesh.visible = True
        blob_mesh.mesh.position.set(0.0, 0.0, -0.5 * SPACING)
        live.mesh.position.set(0.0, 0.0, +0.5 * SPACING)

    def enter_act2():
        build_trial()
        act2.kick(25.0)
        camera.position.set(19.0, 5.6, 0.0)
        camera.look_at(0.0, 0.4, 0.0)
        state["mode"] = "act2"
        print("\nact 2: both bodies kicked to 25 deg roll")

    def enter_act3():
        build_trial()
        act2.begin_act3(ACT3_SECONDS)
        # A quarter view, deliberately: roll is about the fore-aft axis, so a
        # bow-on camera is looking straight DOWN the axis a capsize turns about
        # and a turtled body still shows you its topsides. From the quarter, a
        # rolled body shows its bottom.
        camera.position.set(26.0, 8.5, -21.0)
        camera.look_at(0.0, 0.3, 0.0)
        state["mode"] = "act3"
        state["shot3"] = 0
        print(f"  bodies at z = {blob_mesh.mesh.position.z:+.1f} m (blob) and "
              f"{live.mesh.position.z:+.1f} m (hull)")
        print(f"\nact 3: rising sea, wind {ACT3_WIND0:.0f} -> {ACT3_WIND1:.0f} "
              f"m/s over {ACT3_SECONDS:.0f} s, capsize past {CAPSIZE_DEG:.0f} deg")

    def step_sim(dt):
        t0 = time.perf_counter()
        if state["mode"] == "sculpt":
            if not state["paused"] and sculpt.step_count < STEPS:
                for _ in range(K_STEPS):
                    sculpt.step()
                if sculpt.step_count % 10 == 0:
                    state["m"] = sculpt.measure()
                    # Re-strike the stripe as the shape moves under it. Paint
                    # applied once to the REST blob would ride the vertices to
                    # wherever the descent takes them, which is not a waterline.
                    live.repaint(hull_paint(sculpt.positions(),
                                            float(sculpt._d_host[0]),
                                            TOPSIDE_HULL))
            publish_sculpt()
        else:
            act2.step(dt)
            publish_act2()
        state["step_ms"] = 1000.0 * (time.perf_counter() - t0)

    def draw_ui():
        tp.imgui.set_next_window_pos(10, 10)
        tp.imgui.set_next_window_size(390, 0)
        tp.imgui.begin("differentiable hull")
        m = state["m"]
        if state["mode"] == "sculpt":
            tp.imgui.text(f"{m['phase']} {sculpt.step_count} / {STEPS}"
                          f"{'  [paused]' if state['paused'] else ''}")
            if m["phase"] == "dynamic":
                tp.imgui.text(f"L_dyn       {m['dyn']:.4f}  "
                              f"(rollout {sculpt.T * DYN_DT:.1f} s)")
            tp.imgui.text(f"loss        {m['loss']:.5f}")
            tp.imgui.text(f"drag        {m['drag']:.4f} m^2")
            tp.imgui.text(f"GZ(30 deg)  {m['gz30']:+.4f} m")
            tp.imgui.text(f"V           {m['v_tot']:.2f} m^3 "
                          f"({100.0 * m['v_err']:+.2f}%)")
            tp.imgui.text(f"draft       {m['draft'][0]:+.3f} m")
            tp.imgui.text(f"cargo       {100.0 * sculpt.cargo_f:.0f}% @ "
                          f"y={sculpt.cargo_p[1]:+.2f}   (key 3)")
            tp.imgui.separator()
            tp.imgui.text(f"w_drag {W['drag']:.2f}   w_stab {W['stab']:.2f}"
                          "   (keys 1 / 2)")
        else:
            for i, b in enumerate(act2.bodies):
                tp.imgui.text(f"{b.name:<5} roll {math.degrees(b.phi):+7.1f}"
                              f"   pitch {math.degrees(b.theta):+6.1f}"
                              f"   peak {math.degrees(b.peak):6.1f} /"
                              f" {math.degrees(b.peak_t):5.1f}")
            if state["mode"] == "act3":
                tp.imgui.separator()
                for b in act2.bodies:
                    tp.imgui.text(f"{b.name:<5} {b.u:5.2f} m/s "
                                  f"({1.944 * b.u:4.1f} kn)   {b.px:7.1f} m"
                                  f"{'   ENGINE OUT' if b.capsized() else ''}")
                tp.imgui.text(f"Hs {act2.hs:.2f} m   wind "
                              f"{wind_speed(ocean):.1f} m/s")
                for line in act2.scoreboard():
                    tp.imgui.text(line)
            tp.imgui.text(f"t = {act2.t:.1f} s")
        tp.imgui.separator()
        tp.imgui.text(f"sim {state['step_ms']:.1f} ms   "
                      f"render {state['render_ms']:.1f} ms   "
                      f"{tp.imgui.get_framerate():.0f} fps")
        tp.imgui.text(f"route: {live.route}")
        tp.imgui.end()

    keys = {}

    def pressed(k):
        now = canvas.is_key_down(k)
        hit = now and not keys.get(k, False)
        keys[k] = now
        return hit

    def handle_keys():
        if pressed("SPACE"):
            state["paused"] = not state["paused"]
        if pressed("R") and state["mode"] == "sculpt":
            sculpt.reset()
            state["m"] = sculpt.measure()
        if pressed("G"):
            state["ghost"] = not state["ghost"]
            if state["mode"] == "sculpt":
                blob_mesh.mesh.visible = state["ghost"]
        if pressed("V") and state["mode"] == "sculpt":
            enter_act2()
        if pressed("T") and state["mode"] != "act3":
            enter_act3()
        if pressed("3") and state["mode"] == "sculpt":
            sculpt.set_cargo(0.0 if sculpt.cargo_f > 0.0
                             else cli_arg("--cargo-frac", 0.15, float))
            state["m"] = sculpt.measure()
        if pressed("1"):
            W["drag"] = min(W["drag"] * 1.35, 40.0)
            W["stab"] = max(W["stab"] / 1.35, 0.05)
            sculpt.push_weights()
        if pressed("2"):
            W["stab"] = min(W["stab"] * 1.35, 40.0)
            W["drag"] = max(W["drag"] / 1.35, 0.05)
            sculpt.push_weights()

    if SELFTEST:
        rc = run_selftest(sculpt, m0, renderer, scene, camera,
                          publish_sculpt, enter_act2, enter_act3, publish_act2,
                          lambda: act2)
        live.release(renderer)
        blob_mesh.release(renderer)
        return rc

    if FILM:
        rc = run_film(sculpt, m0, renderer, scene, camera, ocean, live,
                      blob_mesh, publish_sculpt, publish_act2, build_trial,
                      lambda: act2)
        live.release(renderer)
        blob_mesh.release(renderer)
        return rc

    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.max_distance = 220.0

    if ACT2_FIRST or ACT3_FIRST:
        print(f"optimising {STEPS} steps...")
        t0 = time.perf_counter()
        t_draw = 0.0
        paused = False

        def descent_frame():
            # The fast-path descent is a scene the user inhabits, not a
            # cutscene: orbit + HUD + SPACE pause, exactly like act 1's own
            # descent. Only the mode keys (V/T/R/1/2) stay out -- half a
            # descent is not a hull yet, and the trial entry that follows
            # this loop wants the FINISHED shape.
            nonlocal paused
            if pressed("SPACE"):
                paused = not paused
            if ui is not None:
                controls.enabled = not ui.want_capture_mouse
            controls.update()
            ocean.warp_toward(controls.target.x, controls.target.z, 0.3)
            renderer.render(scene, camera)
            if ui is not None:
                ui.render(draw_ui)

        i = 0
        alive = True
        while i < STEPS and alive:
            if paused:
                time.sleep(0.005)
            else:
                sculpt.step()
                i += 1
                if i % 200 == 0:
                    state["m"] = sculpt.measure()  # keep the HUD numbers live
                    print(f"  step {i:5d} / {STEPS}  "
                          f"{time.perf_counter() - t0:5.1f} s", flush=True)
            # Keep the window alive and showing the descent. Time-based rather
            # than every-N-steps: a static step is 2.4 ms and a dynamic one is
            # 27, so a fixed stride would be 40 fps in the first half and 4 in
            # the second. ~25 fps either way, for 2-3 ms of render.
            now = time.perf_counter()
            if paused or now - t_draw > 0.040:
                t_draw = now
                if not paused:
                    publish_sculpt()
                    live.repaint(hull_paint(sculpt.positions(),
                                            float(sculpt._d_host[0]), TOPSIDE_HULL))
                alive = canvas.animate_once(descent_frame)
        print(f"  {time.perf_counter() - t0:.1f} s")
        # Window closed mid-descent: skip the trial entry and fall through --
        # the main loop's animate_once returns False immediately and the
        # finally below runs the interop release, same as any other close.
        if alive:
            state["m"] = sculpt.measure()
            publish_sculpt()
            renderer.render(scene, camera)
            enter_act3() if ACT3_FIRST else enter_act2()
    if state["mode"] == "act3":
        # The orbit TARGET, not look_at, is what survives the first update().
        # Keep it near the waterline: raising it tilts the camera down, which
        # both foreshortens the 11 m gap between the bodies (they read as
        # overlapping) and pushes the sky out of frame. Act 2's framing is the
        # one that works; act 3 just stands further back.
        controls.target.set(0.0, 0.3, 0.0)
        # A quarter view, deliberately: roll is about the fore-aft axis, so a
        # bow-on camera is looking straight DOWN the axis a capsize turns about
        # and a turtled body still shows you its topsides. From the quarter, a
        # rolled body shows its bottom.
        camera.position.set(26.0, 8.5, -21.0)
    canvas.on_window_resize(resize_handler(camera, renderer))
    clock = tp.Clock()

    def animate():
        dt = min(clock.get_delta(), 0.1)
        handle_keys()
        if ui is not None:
            controls.enabled = not ui.want_capture_mouse
        step_sim(max(dt, 1.0 / 240.0))
        # Follow the race. Only the orbit TARGET is moved, never the camera
        # itself, so OrbitControls carries the eye along with it and a viewer
        # who is dragging the mouse keeps the angle they chose. Without this
        # the racers are out of frame in twenty seconds and off the warped
        # ocean patch soon after.
        if state["mode"] == "act3" and act2 is not None:
            tx, tz = act2.focus()
            k = min(3.0 * dt, 1.0)
            controls.target.x += k * (tx - controls.target.x)
            controls.target.z += k * (tz - controls.target.z)
        controls.update()
        ocean.warp_toward(controls.target.x, controls.target.z, 0.3)
        t0 = time.perf_counter()
        renderer.render(scene, camera)
        state["render_ms"] = 1000.0 * (time.perf_counter() - t0)
        if ui is not None:
            ui.render(draw_ui)
        state["frames"] += 1
        # One frame of act 3, taken 1.5 s after the first body goes over --
        # the moment the scoreboard is about, rather than a random frame.
        if state["mode"] == "act3" and state["shot3"] >= 0:
            cap = [b.cap_t for b in act2.bodies if b.capsized()]
            if cap:
                state["shot3"] = 1
                # Wait for the RACE to have happened, not just the capsize: the
                # blob goes over inside ten seconds, while both are still
                # accelerating, and a frame taken there shows two boats abreast.
                gap = act2.bodies[1].px - act2.bodies[0].px
                if act2.t > min(cap) + 4.0 and (gap > 12.0
                                                or act2.t > min(cap) + 30.0):
                    os.makedirs(SHOT_DIR, exist_ok=True)
                    png = os.path.join(SHOT_DIR, "hull_sculpt_act3.png")
                    renderer.save_frame(scene, camera, png)
                    print(f"  wrote {png} (t = {act2.t:.1f} s, "
                          f"Hs {act2.hs:.2f} m)")
                    state["shot3"] = -1

    try:
        # animate_once rather than animate(): a frame budget makes the window
        # path runnable in a smoke test without a human to close it.
        n = 0
        while canvas.animate_once(animate):
            n += 1
            if FRAME_BUDGET and n >= FRAME_BUDGET:
                break
    finally:
        m1 = sculpt.measure()
        print("\nfinal:")
        print_metrics("hull", m1)
        print(f"           GZ curve  {gz_curve(m1)}")
        print(f"  GZ(30) {m0['gz30']:+.4f} -> {m1['gz30']:+.4f} m   "
              f"drag {m0['drag']:.4f} -> {m1['drag']:.4f} m^2 "
              f"({100.0 * (m1['drag'] / m0['drag'] - 1.0):+.1f}%)")
        e1, peak1, fin1 = sculpt.roll_trace()
        print(f"  rollout roll energy {e1:.4f}   peak |roll| {peak1:.1f} deg   "
              f"symmetry {m1['asym']:.2e} m")
        if act2 is not None and state["mode"] == "act3":
            print("  act 3 scoreboard:")
            for line in (act2.scoreboard() + act2.race_report()
                         + act2.pitch_report()):
                print(f"    {line}")
            print(f"    {act3_verdict(act2)}")
        live.release(renderer)
        blob_mesh.release(renderer)
    return 0


# --------------------------------------------------------------------------- #
#  --film: the whole story, offline and deterministic
#
#  Nothing here reads a clock. `renderer.sim_time` is driven explicitly at a
#  fixed 1/60 s, two sim steps to every saved frame, so a re-render is identical
#  frame for frame -- which is the only way a film of a physics demo can be cut,
#  re-cut and trusted. No imgui: the picture carries it, and the numbers get an
#  end card.
#
#  45.5 s at 30 fps, headless, streamed frame by frame into x264 with nothing
#  written to disk but the mp4:
#
#     4.0 s  the blob alone in a calm sea
#    16.5 s  the descent -- 2000 optimiser steps, and the last six seconds of
#            it are BELOW the surface, because the submerged body is the thing
#            being designed and the first cut of this film never showed it
#     8.0 s  the 25-degree kick: the blob goes over, the hull rights
#     9.0 s  the race into a rising sea, chase camera on Act2.focus()
#     5.0 s  the same race from two metres under it, the hull slicing past
#     3.0 s  the end card
# --------------------------------------------------------------------------- #
def _ease(t):
    """Smoothstep. Camera moves start and stop, they do not snap."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def _lerp3(a, b, t):
    return tuple(x + (y - x) * t for x, y in zip(a, b))


def _end_card(lines, size):
    """The closing numbers, as an RGB array. PIL rather than ffmpeg drawtext: no
    font-path quoting to lose an afternoon to, and the layout is visible here."""
    from PIL import Image, ImageDraw, ImageFont
    w, h = size
    im = Image.new("RGB", (w, h), (6, 8, 11))
    d = ImageDraw.Draw(im)
    big = small = None
    for name in ("consola.ttf", "arial.ttf", "DejaVuSansMono.ttf"):
        try:
            big = ImageFont.truetype(name, int(h * 0.052))
            small = ImageFont.truetype(name, int(h * 0.034))
            break
        except OSError:
            continue
    if big is None:
        big = small = ImageFont.load_default()
    y = int(h * 0.30)
    for i, ln in enumerate(lines):
        f = big if i == 0 else small
        tw = d.textbbox((0, 0), ln, font=f)[2]
        d.text(((w - tw) * 0.5, y), ln, font=f,
               fill=(238, 242, 247) if i == 0 else (150, 168, 186))
        y += int(h * (0.085 if i == 0 else 0.055))
    return np.asarray(im, dtype=np.uint8)


# ---- the water as a MEDIUM, for the film's two submerged shots ------------- #
#  What the optimiser actually designs is the SUBMERGED body, and the first cut
#  of the film never showed it. `set_fog_water_surface_y` tells the renderer
#  where air stops, `set_underwater_murk` gives the column its extinction and
#  in-scatter colour, and the rest is free: below the line the sky folds into
#  Snell's window and the hull reads as a silhouette against it.
def _film_water(renderer, scene):
    renderer.set_fog_water_surface_y(0.0)
    renderer.set_underwater_murk(MURK_SIGMA, tp.Color(*MURK_COLOR))
    for ch in scene.children:
        if ch.name == "seafloor":
            # A transform, not a rebuild. At -16 m the floor is a shoal two
            # boat-lengths under the keel; it has no business in a shot of a
            # hull at sea and the murk cannot hide it at this density.
            ch.position.y = FLOOR_Y_FILM


def _descent_cam(u):
    """The whole descent as one camera, `u` in [0, 1].

    Two movements welded at U1, not two shots: the dolly stands off as she
    lengthens, then swings from broadside round to the bow quarter and SINKS,
    and the surface crossing is a thing the shot does rather than a cut. Below
    the line the lens looks slightly up, so the underbody the optimiser has
    been carving all this time prints against Snell's window.
    """
    U1 = 0.60
    if u < U1:
        e = _ease(u / U1)
        a, r, y = 52.0 + 38.0 * e, 15.0 + 8.0 * e, 5.0 - 1.6 * e
        look = (0.0, 0.2, 0.0)
    else:
        # Stand OFF as she sinks, do not close in: at r = 10 the 11 m hull is
        # cropped by the frame and the shot stops being a picture of a shape.
        # There is one right distance and it is narrow. At 10 m the 11 m hull
        # is cropped and the shot stops being a picture of a shape; at 19 m the
        # murk has eaten a third of its contrast and she reads as a smudge.
        # ~14 m holds her whole length in frame with the water still thin
        # enough to see her by.
        v = _ease((u - U1) / (1.0 - U1))
        a, r, y = 90.0 - 42.0 * v, 23.0 - 9.5 * v, 3.4 - 6.6 * v
        look = (0.0, 0.2 - 0.9 * v, 0.0)
    ar = math.radians(a)
    return (r * math.cos(ar), y, r * math.sin(ar)), look


def _uw_pass_cam(v, hull_x, lane_z):
    """The race's submerged pass. One continuous move: the lens hangs on the
    far side of the hull's lane, two and a half metres down, and pans as she
    comes up from astern and slices past -- with the blob, turtled, eleven
    metres beyond her and most of a stop darker for it.

    `hull_x` is where she was when the shot started: she makes 2.7 m/s and runs
    13 m through it, so the lens is parked eight metres up the course and
    TWELVE off her lane. Astern-to-ahead, closest approach 12.5 m -- half that
    and she is a wall of hull with no shape to read."""
    eye = (hull_x + 8.0 + 0.8 * v, -2.1 - 0.9 * v, lane_z + 12.5 - 1.2 * v)
    return eye, (hull_x, -0.5, lane_z)


def _film_probe(renderer, scene, camera, ocean, sculpt, publish_sculpt,
                publish_act2, build_trial, get_act2, blob_mesh, live):
    """Candidate framings as stills. Cheap: the descent runs with no renders at
    all, and the race is stepped forward without drawing anything either."""
    out = os.path.join(SHOT_DIR, "film_probe")
    os.makedirs(out, exist_ok=True)
    dt = 1.0 / 60.0
    t = [0.0]

    def shoot(name, eye, look):
        camera.position.set(*eye)
        camera.look_at(*look)
        ocean.warp_toward(look[0], look[2], 1.0)
        for _ in range(4):                      # let TAA and the ocean settle
            t[0] += dt
            renderer.sim_time = t[0]
            renderer.render(scene, camera)
        p = os.path.join(out, f"{name}.png")
        from PIL import Image
        Image.fromarray(renderer.read_pixels()).save(p)
        print(f"  probe: {p}")

    _film_water(renderer, scene)
    set_wind_speed(ocean, 6.0)
    blob_mesh.mesh.visible = False
    publish_sculpt()
    eye, look = _descent_cam(0.0)
    shoot("a_open_abovewater", eye, look)

    print(f"  probe: {STEPS} optimiser steps, no renders")
    while sculpt.step_count < STEPS:
        sculpt.step()
    publish_sculpt()
    live.repaint(hull_paint(sculpt.positions(), float(sculpt._d_host[0]),
                            TOPSIDE_HULL))
    for u in (0.80, 1.0):
        eye, look = _descent_cam(u)
        shoot(f"b_descent_uw_u{int(u * 100):03d}", eye, look)

    build_trial()
    act2 = get_act2()
    act2.begin_act3(45.0)
    for _ in range(9 * 30):                     # 9 s of race, undrawn
        for _ in range(2):
            act2.step(dt, substeps=1)
        t[0] += 2 * dt
    publish_act2()
    hx = act2.bodies[1].px
    lane = 0.5 * SPACING
    print(f"  probe: race at t = {act2.t:.1f} s, hull x = {hx:.1f} m, "
          f"blob {'CAPSIZED' if act2.bodies[0].capsized() else 'upright'}")
    for v in (0.0, 0.6):
        eye, look = _uw_pass_cam(v, hx, lane)
        shoot(f"c_race_submerged_v{int(v * 100):03d}", eye, look)
    # The half-in candidate: a lens ON the surface ahead of the pack. The split
    # lives inside the few centimetres the near plane spans, so the eye rides
    # the local wave height rather than mean sea level.
    for trim in (0.0, 0.04):
        wl = float(ocean.sample_height(hx + 16.0, lane))
        eye = (hx + 16.0, wl + trim, lane)
        shoot(f"d_race_waterline_t{int(trim * 100):02d}", eye,
              (hx, 0.1, lane))
    renderer.sim_time = None
    return 0


def run_film(sculpt, m0, renderer, scene, camera, ocean, live, blob_mesh,
             publish_sculpt, publish_act2, build_trial, get_act2):
    import tempfile

    if FILM_PROBE:
        return _film_probe(renderer, scene, camera, ocean, sculpt,
                           publish_sculpt, publish_act2, build_trial, get_act2,
                           blob_mesh, live)

    dt = 1.0 / 60.0
    sub = 2                                   # sim steps per encoded frame
    out = FILM_OUT or os.path.join(SHOT_DIR, "hull_sculpt.mp4")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    log = os.path.join(tempfile.gettempdir(), "hull_sculpt_ffmpeg.log")
    enc = Encoder(out, WIN_W, WIN_H, FILM_FPS, crf=18, preset="medium",
                  faststart=True, log=log)
    print(f"\nfilm: {WIN_W}x{WIN_H} at {FILM_FPS} fps, streaming -> {out}")

    _film_water(renderer, scene)
    state = dict(n=0, t=0.0)
    checked = [False]

    def frame(look):
        """Advance the sea by one output frame, render it, pipe it."""
        for _ in range(sub):
            state["t"] += dt
        renderer.sim_time = state["t"]
        ocean.warp_toward(look[0], look[2], 0.35)
        renderer.render(scene, camera)
        px = renderer.read_pixels()
        if not checked[0]:
            checked[0] = True
            # A hidden canvas is still a window on Windows and a window can be
            # clamped to the desktop. The readback, not the request, is what
            # ends up in the file -- so verify it once, loudly.
            print(f"  readback {px.shape[1]}x{px.shape[0]} "
                  f"(renderer {renderer.size()}, asked {WIN_W}x{WIN_H})")
            if (px.shape[1], px.shape[0]) != (WIN_W, WIN_H):
                raise RuntimeError(
                    f"framebuffer clamped to {px.shape[1]}x{px.shape[0]}")
        enc.send(px)
        if FILM_DIR:
            from PIL import Image
            Image.fromarray(px).save(
                os.path.join(FILM_DIR, f"f{state['n']:05d}.png"))
        state["n"] += 1

    def shot(seconds):
        return int(round(seconds * FILM_FPS))

    if FILM_DIR:
        os.makedirs(FILM_DIR, exist_ok=True)
    t_wall = time.perf_counter()

    # -- 1. the blob, alone, in a calm sea (4.0 s) --------------------------
    set_wind_speed(ocean, 6.0)
    blob_mesh.mesh.visible = False
    publish_sculpt()
    n1 = shot(4.0)
    print(f"  shot 1  open, {n1} frames")
    for i in range(n1):
        a = math.radians(28.0 + 26.0 * _ease(i / max(n1 - 1, 1)))
        look = (0.0, 0.3, 0.0)
        camera.position.set(15.5 * math.cos(a), 4.4, 15.5 * math.sin(a))
        camera.look_at(*look)
        frame(look)

    # -- 2. the descent, above and then below the line (16.5 s) -------------
    #  The hero shot, and it stays the hero shot at 16.5 s instead of 28: the
    #  optimiser steps are spread over whatever length the cut is, so shortening
    #  it makes the timelapse faster, not shorter. All 2000 steps are here.
    n2 = shot(16.5)
    print(f"  shot 2  the descent, {n2} frames, {STEPS} optimiser steps")
    for i in range(n2):
        want = int(round(STEPS * (i + 1) / n2))
        while sculpt.step_count < want:
            sculpt.step()
        publish_sculpt()
        if i % 6 == 0:
            live.repaint(hull_paint(sculpt.positions(),
                                    float(sculpt._d_host[0]), TOPSIDE_HULL))
        eye, look = _descent_cam(i / max(n2 - 1, 1))
        camera.position.set(*eye)
        camera.look_at(*look)
        frame(look)

    # -- 3. the kick (8.0 s) ------------------------------------------------
    build_trial()
    act2 = get_act2()
    act2.kick(25.0)
    n3 = shot(8.0)
    print(f"  shot 3  the kick, {n3} frames")
    for i in range(n3):
        for _ in range(sub):
            act2.step(dt, substeps=1)
        publish_act2()
        look = (0.0, 0.4, 0.0)
        camera.position.set(20.0, 5.6, 1.5)
        camera.look_at(*look)
        frame(look)

    # -- 4. the race (9.0 s of chase) ---------------------------------------
    #  The ramp is passed 30 s rather than the trial's 90: the film is half as
    #  long as it was, so the sea has to come up in the time the picture has.
    #  Over the 14 s of race the wind runs 8 -> 13.6 m/s, past where the blob
    #  went over in the 90 s cut, so the capsize still happens ON CAMERA. A
    #  film argument, not a physics one -- `--act3` and the selftest are
    #  untouched and still run the 90 s trial.
    FILM_RAMP = 30.0
    act2.begin_act3(FILM_RAMP)
    n4 = shot(9.0)
    print(f"  shot 4  the race, {n4} frames, ramp {FILM_RAMP:.0f} s")
    for i in range(n4):
        for _ in range(sub):
            act2.step(dt, substeps=1)
        publish_act2()
        e = _ease(max(i / max(n4 - 1, 1) - 0.60, 0.0) / 0.40)
        fx, _ = act2.focus()
        look = (fx, 0.3, 0.0)
        camera.position.set(*_lerp3((fx + 26.0, 8.5, -21.0),
                                    (fx + 38.0, 13.5, -30.0), e))
        camera.look_at(*look)
        frame(look)

    # -- 5. the race, from under it (5.0 s) ---------------------------------
    n5 = shot(5.0)
    lane_z = 0.5 * SPACING
    hx0 = act2.bodies[1].px
    print(f"  shot 5  the submerged pass, {n5} frames, hull at x = {hx0:.1f} m,"
          f" blob {'capsized' if act2.bodies[0].capsized() else 'upright'}")
    for i in range(n5):
        for _ in range(sub):
            act2.step(dt, substeps=1)
        publish_act2()
        v = i / max(n5 - 1, 1)
        eye, look = _uw_pass_cam(v, hx0, lane_z)
        camera.position.set(*eye)
        camera.look_at(act2.bodies[1].px, look[1], look[2])
        frame((act2.bodies[1].px, 0.0, lane_z))

    # -- 6. the end card (3.0 s) --------------------------------------------
    b, h = act2.bodies
    ub = terminal_speed(b.s_drag, b.s_wet, b.lwl, lambda u: act2.wave_at(b, u))
    uh = terminal_speed(h.s_drag, h.s_wet, h.lwl, lambda u: act2.wave_at(h, u))
    m1 = sculpt.measure()
    print_metrics("hull", m1)
    print_resistance("hull", m1)
    card = _end_card([
        "Designed by gradient descent.",
        f"same displacement, same thrust:  {ub:.2f} -> {uh:.2f} m/s",
        f"GZ(30 deg):  {m0['gz30']:+.2f} -> {m1['gz30']:+.2f} m",
        f"length:  {m0['bbox'][0]:.1f} -> {m1['bbox'][0]:.1f} m",
        f"{STEPS} Adam steps through form, friction and wave resistance",
    ], (WIN_W, WIN_H))
    n6 = shot(3.0)
    print(f"  shot 6  end card, {n6} frames")
    for _ in range(n6):
        enc.send(card)
        state["n"] += 1

    renderer.sim_time = None
    total = state["n"]
    wall = time.perf_counter() - t_wall
    print(f"  {total} frames in {wall:.0f} s "
          f"({total / max(wall, 1e-9):.1f} frames/s), "
          f"{total / FILM_FPS:.1f} s of film")
    print("  closing the pipe...")
    rc = enc.close()
    if rc != 0:
        print(f"  ffmpeg failed ({rc}); log at {log}")
        return 1
    mb = os.path.getsize(out) / 1e6
    print(f"  wrote {out} -- {total / FILM_FPS:.1f} s, {mb:.1f} MB, "
          f"{time.perf_counter() - t_wall:.0f} s wall in total")
    return 0


# --------------------------------------------------------------------------- #
#  --selftest: the acceptance run
# --------------------------------------------------------------------------- #
def run_selftest(sculpt, m0, renderer, scene, camera,
                 publish_sculpt, enter_act2, enter_act3, publish_act2, get_act2):
    fails = []
    os.makedirs(SHOT_DIR, exist_ok=True)
    before = os.path.join(SHOT_DIR, "hull_sculpt_before.png")
    after = os.path.join(SHOT_DIR, "hull_sculpt_after.png")

    for _ in range(6):                       # let the ocean spin up before the shot
        renderer.render(scene, camera)
    renderer.save_frame(scene, camera, before)
    print(f"wrote {before}")

    # -- 1. the descent -----------------------------------------------------
    print(f"\n[1] {STEPS} optimiser steps")
    losses = []
    t_opt = t_pub = t_ren = 0.0
    frames = max(STEPS // K_STEPS, 1)
    for fr in range(frames):
        t0 = time.perf_counter()
        for _ in range(K_STEPS):
            sculpt.step()
        t_opt += time.perf_counter() - t0
        t0 = time.perf_counter()
        publish_sculpt()
        t_pub += time.perf_counter() - t0
        t0 = time.perf_counter()
        renderer.render(scene, camera)
        t_ren += time.perf_counter() - t0
        i = sculpt.step_count - 1
        if (i + 1) % 100 < K_STEPS:
            if not sculpt.finite():
                fails.append(f"non-finite vertices at step {i + 1}")
                break
            m = sculpt.measure()
            losses.append((m["phase"], m["loss"], m["dyn"]))
            if not np.isfinite(m["loss"]):
                fails.append(f"non-finite loss at step {i + 1}")
                break
            dyn_txt = "" if not np.isfinite(m["dyn"]) else f"  E_roll {m['dyn']:.4f}"
            print(f"    step {i + 1:5d} {m['phase'][:3]}  loss {m['loss']:9.5f}  "
                  f"GZ(30) {m['gz30']:+.4f}  drag {m['drag']:8.4f}  "
                  f"V err {100.0 * m['v_err']:.2f}%{dyn_txt}")
    n = frames
    print(f"    per rendered FRAME at k={K_STEPS}: optimiser "
          f"{1000.0 * t_opt / n:.2f} ms ({1000.0 * t_opt / (n * K_STEPS):.2f} "
          f"ms/step) | publish {1000.0 * t_pub / n:.2f} ms | render "
          f"{1000.0 * t_ren / n:.2f} ms = {1000.0 * (t_opt + t_pub + t_ren) / n:.2f}"
          f" ms/frame ({n / max(t_opt + t_pub + t_ren, 1e-9):.0f} fps)")
    # Per PHASE, and each phase judged on what it is actually minimising. The
    # switch hands the objective over mid-descent, so the total loss steps UP
    # there -- Adam's moments are still the static term's -- and comes back down
    # over the following few hundred steps. What has to fall monotonically in
    # the dynamic phase is the rollout energy: that is the term that was turned
    # on, and the one act 2 then goes and measures.
    seq = [v for p, v, _ in losses if p == "static"]
    if len(seq) >= 2 and not (seq[-1] < seq[0]):
        fails.append(f"static loss did not decrease ({seq[0]:.5f} -> {seq[-1]:.5f})")
    seq = [d for p, _, d in losses if p == "dynamic" and np.isfinite(d)]
    if len(seq) >= 2 and not (seq[-1] < seq[0]):
        fails.append(f"rollout energy did not decrease in the dynamic phase "
                     f"({seq[0]:.5f} -> {seq[-1]:.5f})")

    m1 = sculpt.measure()
    print("\n[2] quantitative improvement")
    print_metrics("blob", m0)
    print_metrics("hull", m1)
    print_resistance("blob", m0)
    print_resistance("hull", m1)
    print(f"    GZ curve blob  {gz_curve(m0)}")
    print(f"    GZ curve hull  {gz_curve(m1)}")
    # The plan's "> 2x the blob's GZ(30)" is vacuous for a blob whose GZ(30) is
    # NEGATIVE (an unstable blob is the premise of act 2), so the test is the
    # sign flip plus an absolute margin. See the plan's "As built".
    margin = 0.04 * sculpt.lref
    if not (m1["gz30"] > 0.0):
        fails.append(f"final GZ(30) not positive ({m1['gz30']:+.4f} m)")
    if not (m1["gz30"] > m0["gz30"] + margin):
        fails.append(f"GZ(30) gain below margin: {m0['gz30']:+.4f} -> "
                     f"{m1['gz30']:+.4f} (needs +{margin:.3f})")
    if not (m1["drag"] < m0["drag"]):
        fails.append(f"drag not reduced ({m0['drag']:.4f} -> {m1['drag']:.4f})")
    if not (m1["v_err"] < 0.02):
        fails.append(f"displacement off by {100.0 * m1['v_err']:.2f}% (> 2%)")
    # The projection is exact or it is not on: this is a bug check, not a
    # quality metric, so the tolerance is tight on purpose.
    print(f"    symmetry residual {m1['asym']:.2e} m   "
          f"cargo {100.0 * sculpt.cargo_f:.0f}% @ y={sculpt.cargo_p[1]:+.2f}")
    if not NO_SYM and not (m1["asym"] < 1.0e-3):
        fails.append(f"port-starboard residual {m1['asym']:.2e} m (> 1e-3)")
    e1, peak1, fin1 = sculpt.roll_trace()
    print(f"    rollout roll energy {e1:.4f}   peak |roll| {peak1:.1f} deg   "
          f"final |roll| {fin1:.2f} deg")

    renderer.save_frame(scene, camera, after)
    print(f"    wrote {after}")

    # -- 3. act 2 -----------------------------------------------------------
    print("\n[3] act 2: 25 deg kick, blob vs hull")
    enter_act2()
    act2 = get_act2()
    frames = int(ACT2_SECONDS * 60.0)
    shot_at = int(6.0 * 60.0)
    act2_png = os.path.join(SHOT_DIR, "hull_sculpt_act2.png")
    for k in range(frames):
        act2.step(1.0 / 60.0)
        publish_act2()
        renderer.render(scene, camera)
        if k == shot_at:
            renderer.save_frame(scene, camera, act2_png)
            print(f"    wrote {act2_png} (t = {act2.t:.1f} s)")
    for b in act2.bodies:
        print(f"    {b.name:<5} peak |roll| {math.degrees(b.peak):6.1f} deg   "
              f"final roll {math.degrees(b.phi):+7.1f} deg")
    blob, hull = act2.bodies
    if not (math.degrees(blob.peak) > 90.0):
        fails.append(f"blob did not roll over (peak "
                     f"{math.degrees(blob.peak):.1f} deg)")
    if not (abs(math.degrees(blob.phi)) > 60.0):
        fails.append(f"blob recovered (final {math.degrees(blob.phi):+.1f} deg)")
    if not (abs(math.degrees(hull.phi)) < 15.0):
        fails.append(f"hull did not settle (final "
                     f"{math.degrees(hull.phi):+.1f} deg)")
    if not (math.degrees(hull.peak) < 40.0):
        fails.append(f"hull peak roll {math.degrees(hull.peak):.1f} deg (> 40)")
    if not sculpt.finite():
        fails.append("non-finite vertices after act 2")

    # -- 4. act 3 -----------------------------------------------------------
    secs = min(ACT3_SECONDS, 60.0)
    print(f"\n[4] act 3: rising sea over {secs:.0f} s, sim_time driven")
    enter_act3()
    act3 = get_act2()
    act3.ramp = (act3.ramp[0], act3.ramp[1], secs)
    dt = 1.0 / 30.0
    act3_png = os.path.join(SHOT_DIR, "hull_sculpt_act3.png")
    shot_done = False
    t = 0.0
    for k in range(int(secs / dt)):
        act3.step(dt)
        publish_act2()
        # No OrbitControls here, so the follow is on the camera itself.
        fx, _ = act3.focus()
        camera.position.x = 26.0 + fx
        camera.look_at(fx, 0.3, 0.0)
        # Pinned time, fixed dt: the sea the bodies are scored against replays
        # bit for bit, which is the only way this assertion means anything.
        t += dt
        renderer.sim_time = t
        renderer.render(scene, camera)
        # Same rule as the windowed shot: wait for the RACE, not just the
        # capsize, or the frame is two boats abreast still accelerating.
        if not shot_done and any(b.capsized() for b in act3.bodies) \
                and act3.t > (act3.bodies[0].cap_t or 0.0) + 4.0 \
                and (act3.bodies[1].px - act3.bodies[0].px) > 12.0:
            renderer.save_frame(scene, camera, act3_png)
            print(f"    wrote {act3_png} (t = {act3.t:.1f} s, "
                  f"Hs {act3.hs:.2f} m)")
            shot_done = True
    renderer.sim_time = None
    for line in (act3.scoreboard() + act3.race_report()
                 + act3.pitch_report()):
        print(f"    {line}")
    print(f"    {act3_verdict(act3)}")
    b3, h3 = act3.bodies
    if not b3.capsized():
        fails.append(f"blob survived act 3 (peak {math.degrees(b3.peak):.1f} deg)")
    else:
        hs_hull = h3.cap_hs if h3.capsized() else act3.hs
        if not (hs_hull > b3.cap_hs):
            fails.append(f"hull did not outlast the blob (hull Hs {hs_hull:.2f} m "
                         f"vs blob {b3.cap_hs:.2f} m)")
    # The race. The shape factors predict sqrt(S_blob / S_hull) ~ 2x; the sea,
    # the acceleration from rest and the blob's dying engine all pull the
    # ACHIEVED ratio around, so the gate is well below the prediction.
    if THRUST["N"] > 0.0 and not (h3.mean_u() > 1.3 * b3.mean_u()):
        fails.append(f"hull did not outrun the blob (mean {h3.mean_u():.2f} vs "
                     f"{b3.mean_u():.2f} m/s, needs 1.3x)")

    print()
    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        return 1
    print("selftest PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
