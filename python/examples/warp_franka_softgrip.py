"""Franka FR3 with soft Fin Ray fingers picking a soft fish -- Warp sim, Vulkan render.

The arm is the real FR3 URDF driven by threepp's own damped-least-squares IK
(tp.IkSolver) along joint-space quintic trajectories. The two fingers are NOT
the rigid stock ones: each stock finger carries a GPU soft body -- an XPBD
two-skin truss, front skin and back skin joined by stiff ribs and SOFT
diagonals, which is what makes it a Fin Ray: press the front skin and the tip
curls toward the object. The fish is a real FEM solid: a voxel-carved
conforming tet cage running STABLE NEO-HOOKEAN XPBD (Macklin/Muller/Chentanez
2021) -- two constraints per tet, a deviatoric (shape) and a hydrostatic
(volume) one, with per-constraint Lagrange multipliers. That formulation is
inversion-safe, which is exactly why the fish stops "dissolving": the old
hand-rolled distance + single-volume-row PBD had no real elastic energy and
resisted inversion weakly, so under a tail hang it stretched to 4x and the head
tore off the flank. The drawn surface is skinned off the cage -- either the
procedural herring or, with --fish-model, the Khronos Barramundi glTF.

Fingers and fish live in ONE solver over ONE hash grid with Coulomb friction,
so the grasp is friction plus form closure. There is no kinematic attach and no
glue anywhere: if the fingers do not hold it, it falls.

The close drives against a MEASURED backstop with a force proxy over the top of
it: the carriages come inward until the summed pad contact force crosses a
threshold or the pad face reaches the flank, then hold.

Each finger is drawn the way it is built -- two thin silicone skins with the
cross ribs between them, sides open -- so you can watch the ribs shear as the
front skin is pushed back. That shear IS the Fin Ray effect.

The fish is graded by length into one of two crates, and the wrist camera is a
live secondary view (picture-in-picture, and the POV camera in the film).

HONEST SCOPE: the arm's poses are SCRIPTED in this version -- a fixed waypoint
list solved by IK. The wrist camera is a real render, not a decoration, but
nothing in the loop is driven by it yet: there is no segmentation, no grasp
synthesis, no closed-loop servoing. What is not scripted is the physics -- the
grasp is friction and form closure between two GPU soft bodies, and if the
fingers do not hold the fish it falls.

    pip install warp-lang
    python warp_franka_softgrip.py                # window
    python warp_franka_softgrip.py --shot 3.2     # headless PNG at sim time 3.2 s
    python warp_franka_softgrip.py --frames 600   # timed phase breakdown
    python warp_franka_softgrip.py --film out.mp4 # the cut: 6 shots + end card
    python warp_franka_softgrip.py --film --preview   # 640x360 at 30 fps, no PIP
    python warp_franka_softgrip.py --grip-force 8 # close threshold, newtons
    python warp_franka_softgrip.py --fish-len 0.3 # graded sizes: 0.24 / 0.27 / 0.30
    python warp_franka_softgrip.py --fish-model fish.glb  # skin a glTF fish (Barramundi)
    python warp_franka_softgrip.py --fish-model scan.usdz --fish-len 0.63 --fish-mass 3.74
                                                  # an FHF photogrammetry cod scan at its
                                                  # MEASURED length and weight (the FR3 hand
                                                  # cannot span it -- see --fish-mass below)
    python warp_franka_softgrip.py --flesh-visc 0.01  # strain-rate damping, s (UNSTABLE here)
    python warp_franka_softgrip.py --young 2.5e5  # fish flesh Young's modulus, Pa
    python warp_franka_softgrip.py --poisson 0.45 # near-incompressible tet FEM
    python warp_franka_softgrip.py --grade-mm 265 # length that picks the far crate
    python warp_franka_softgrip.py --cam macro    # wide | macro (follows the TCP) | top
    python warp_franka_softgrip.py --pov          # render from the wrist camera
    python warp_franka_softgrip.py --dof          # depth of field (see the note below)
    python warp_franka_softgrip.py --no-slowmo    # no 0.15x window around CLOSE -> LIFT
    (interactive: R replays the scenario from t = 0; F cycles the FEM debug view
     -- translucent skin + tet-cage wireframe -> wireframe only -> normal)
    python warp_franka_softgrip.py --no-pip       # no wrist-camera picture-in-picture
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import (accum_normals, cli_arg, icosphere,
                         parse_size, scatter_soup, signed_volume,
                         standard_material)

# --- command line ---------------------------------------------------------------

SHOT = "--shot" in sys.argv
SHOT_TIME = cli_arg("--shot", 3.2, float)
FRAMES = cli_arg("--frames", 0, int)
BENCH = FRAMES > 0
WIDTH, HEIGHT = parse_size(cli_arg("--size", "1280x720", str))
# Newtons, but normalised: the proxy below is divided by the iteration count and
# by the contacting-node count, so it no longer moves when the solver settings do.
F_GRASP = cli_arg("--grip-force", 30.0, float)   # N, both pads: firm enough to seat around the back/belly ridges (palm stops the ride-up; a 9 N pinch lets the almond section ROLL and wedge diagonally)
HEADLESS = SHOT or BENCH
CAM = cli_arg("--cam", "drop" if "--drop" in sys.argv else "wide", str)   # wide | macro | top | drop
# Render skin: default is the self-contained procedural herring so the committed
# example runs on any machine; pass --fish-model to skin a glTF fish (the
# Khronos Barramundi) off the same tet cage. The mesh is NOT vendored.
FISH_MODEL = cli_arg("--fish-model", "", str)
# Environment: a real HDRI (Poly Haven empty_warehouse_01) turns the stainless
# from flat paint into a mirror of an actual room and lights the whole cell by
# IBL. The committed example falls back to the tiny procedural studio env when
# the .hdr is absent, so it still runs anywhere; the .hdr is NOT vendored.
_ENV_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "assets", "empty_warehouse_01_2k.hdr")
ENV_HDRI = cli_arg("--env-hdri", _ENV_DEFAULT if os.path.exists(_ENV_DEFAULT) else "", str)
SLOWMO = "--no-slowmo" not in sys.argv
PIP = "--no-pip" not in sys.argv
DOF = "--dof" in sys.argv                    # see the note at the renderer setup
POV = "--pov" in sys.argv                    # render from the wrist camera
FILM = "--film" in sys.argv
FILM_OUT = cli_arg("--film", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "warp_franka_softgrip.mp4"), str)
PREVIEW = "--preview" in sys.argv
# FISH DROP scenario: no grasp. The fish comes off a raised ledge head-first at
# belt speed, bends over the edge under its own weight, falls to the table and
# settles -- the material test bench for the soft fish (is it soft? does it sag,
# tumble, land and lie like a fish?) with nothing else in the loop.
DROP = "--drop" in sys.argv
FEM_VIEW = "--fem-view" in sys.argv     # start with the tet cage visible
BELT_V = cli_arg("--belt-v", 0.15, float)          # the ledge top is a belt: m/s along +x
DROP_OVERHANG = cli_arg("--drop-overhang", 0.04, float)   # m of fish past the edge at t=0
if FILM:
    HEADLESS = True
    if PREVIEW:
        WIDTH, HEIGHT = 640, 360

# --- solver tunables ------------------------------------------------------------

FPS = 60.0
# SMALL STEPS (Macklin et al. 2019): many substeps, few iterations. A substep's
# constraint error shrinks with dt^2 while iterations only converge linearly, so
# at a fixed launch budget 24 x 4 beats the old 6 x 32 by a wide margin -- and it
# is what lets a plain Jacobi solve (below) reach the stiffness the truss needs
# without any gain cap. DT is derived from the substep count so ONE frame of
# physics IS one frame of task time (the old 6 x 1/240 s ran the physics 1.5x
# faster than the robot).
SUBSTEPS = cli_arg("--substeps", 24, int)
ITERATIONS = cli_arg("--iters", 4, int)
DT = 1.0 / (FPS * SUBSTEPS)
# Velocity damping as RATES (1/s), converted to per-substep factors below so the
# substep count can change without retuning. The truss no longer needs the old
# 0.22-per-substep sledgehammer: that was absorbing the over-relaxed Jacobi ring,
# which the mass-split solve does not have. What is left is material damping.
DAMP_RATE = cli_arg("--damp", 10.0, float)          # fish flesh
DAMP_RATE_FIN = cli_arg("--damp-fin", 20.0, float)  # silicone finger
GRAVITY = wp.vec3(0.0, -9.81, 0.0)
# Jacobi over-relaxation for the structural pass (1 = the plain mass-split step,
# which is unconditionally stable; >1 trades margin for convergence).
OMEGA = cli_arg("--omega", 1.0, float)
# The contact pass (pads vs skin, skin vs floor, both mass-split over the cage
# nodes they share) is light and under-converges in one sweep: friction can only
# cancel the relative slide it gets to see, so sweep it a few times per
# structural iteration.
CONTACT_SWEEPS = cli_arg("--contact-sweeps", 3, int)

# Fin Ray split: the skins are structural, the diagonals are the compliance.
# PHASE 9: the finger read as floppy foil -- it deflected too much and did not
# hold a plate shape. The plate is the two skins + the ribs holding them parallel
# + the bending chords resisting curvature; the COMPLIANCE that makes it a Fin
# Ray is the cell diagonals. So the plate members are stiffened (skins, ribs,
# bending) and the gain headroom raised to let them converge, while STIFF_DIAG is
# left soft so the tip still curls in and wraps the fish.
# The truss is XPBD now: every row has a REAL spring constant k = weight * K_FIN
# (N/m) and its own Lagrange multiplier, so the weights below are relative
# stiffnesses of an elastic truss that converges to a physical equilibrium --
# not Jacobi gains that only meant something at one iteration count. The Fin
# Ray curl is the soft diagonals' compliance, and it survives convergence.
K_FIN = cli_arg("--fin-k", 1500.0, float)       # N/m for a weight-1.0 member
STIFF_SKIN = 1.15
STIFF_RIB = 0.85
STIFF_DIAG = 0.35        # LEFT SOFT: this is the Fin Ray curl compliance
# BENDING. A distance-only sheet has no bending stiffness at all: every fold is
# free, so under the clamp the skins buckled into crumpled amber foil (phase 4's
# poster shows the lower half of each finger reading as a crushed paper bag) and
# the buckles folded triangles back on themselves, which is where the wrist-POV
# "holes" came from. A real Fin Ray skin is a stiff plastic strip -- it bends in
# a smooth arc. Skip-a-node chords give the sheet exactly that: they resist
# CURVATURE without adding any in-plane stretch resistance, so the truss's shear
# compliance -- the Fin Ray effect itself -- is untouched. Same trick as the
# fish's spine chords (phase 3), applied in both directions of both skins.
STIFF_BEND = 1.0
STIFF_BEND2 = 0.3    # skip-TWO chords along the columns (PHASE 9, ON now): a
                      # skip-one chord resists a single-cell kink; the skip-two
                      # chord resists a longer, gentler bow, and the two together
                      # make the skin hold a stiff-plate arc instead of a floppy
                      # foil. The gain cap + extra iterations absorb the new rows.
# The cell's two diagonals. Softening them to 0.6 to pay for the bending rows'
# Jacobi budget was TRIED and is wrong: the in-plane shear rows are load path
# from the pinned root to the pad face, and at 0.6 the hand ploughed the fish
# across the tray instead of lifting it (grip 1.65 N, slip 188 mm/s). The budget
# is bought with FIN_GAIN_CAP below instead. Left as a named knob, not a magic 1.0.
STIFF_SHEAR = 1.0
# Jacobi gain cap for the TRUSS nodes. The fish is normalised to 1 (see the
# solver); the truss deliberately is not, because normalising it to 1 costs the
# clamp (3.1 N -> 0.9 N, measured in phase 3). But a truss node already ran a
# gain of ~2.8, and simply adding the bending rows on top took it to ~3.7 and
# the skins came back SPIKY -- a per-row checkerboard, the classic over-relaxed
# Jacobi signature. So: cap, do not normalise. Rows below the cap behave exactly
# as they did before this phase; the surplus the new rows would have added is
# what gets divided out.
# (FIN_GAIN_CAP is gone: the mass-split XPBD solve has no per-node gain to cap.)
# Root fan. Only the two root rows are pinned to the carriage, and a Jacobi
# solve moves a rigid translation down an 11-row chain appallingly slowly: the
# pad face lagged the carriage so badly that a hand provably squeezing the fish
# (178 fish particles in contact, 13 N) still could not lift it -- the load
# never reached the pinned rows. A chord from the root row straight to every
# node ties the pad face to the carriage in ONE hop. It costs some of the
# finger's gross floppiness; the local conformity and the Fin Ray shear (soft
# diagonals) survive, because those change the chord lengths hardly at all.
STIFF_FAN = 0.5
# The fish's body is a FILLED XPBD tet body, not a pressurised shell. A hollow
# shell squeezed between two converging paddles is a tube of toothpaste: phase 2
# measured the fish ejected along its own axis at 660 mm/s, centroid x 0.455 ->
# 0.60 m, while the pads shut on nothing. There is no interior to hold a shell
# apart, so the only thing resisting the squeeze is one global volume row and
# some chords, and the cheapest way for those to be satisfied is to send the
# material out of the hand. A filled body cannot be extruded that way: every
# tet in the pinched section carries its own incompressibility row.
# The fish flesh is a REAL FEM material now, not a bag of distance + volume rows.
# Stable Neo-Hookean (Macklin, Muller, Chentanez 2021), solved as two XPBD
# constraints per tet -- a deviatoric one (shape, stiffness mu) and a hydrostatic
# one (volume, stiffness lambda) -- with per-constraint Lagrange multipliers
# accumulated across each substep. This is the honest upgrade AND the dissolve
# fix: the hydrostatic constraint is C = det(F) - gamma, so it pushes an
# INVERTED tet (det(F) < 0) back toward a positive volume instead of letting it
# collapse, and the deviatoric constraint is sqrt(tr(F^T F)), which is always
# well defined. The hand-rolled distance edges + single volume row + spine
# chords had no elastic energy and resisted inversion weakly, so under a tail
# hang the body stretched unboundedly and self-intersected. They are ALL gone.
# Fish flesh is soft, but a WHOLE fish (skin + bone) held by two pads under its
# own weight and swung across the cell needs enough modulus to keep its shape
# instead of drawing out like taffy. At 2e4 Pa the body stretched to 4x and the
# head tore off the flank (max edge stretch 3.8); 2.5e5 Pa keeps the p99 edge
# stretch near ~1.3 and still droops like a fresh fish.
YOUNG = cli_arg("--young", 2.5e5, float)      # Pa
POISSON = cli_arg("--poisson", 0.45, float)   # near-incompressible
MU_LAME = YOUNG / (2.0 * (1.0 + POISSON))
LAM_LAME = YOUNG * POISSON / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON))
# Strain-rate (Kelvin-Voigt) damping on the DEVIATORIC corotational row, in
# SECONDS -- ported verbatim from warp_fish_collide.py's tet_corot, where it was
# measured against a slow soft-vs-soft collision. gamma = tau / DT and the solved
# row becomes C + tau*Cdot = 0, so the flesh resists the RATE of shape change,
# not just the shape: a fast squeeze is met with a viscous back-pressure instead
# of an elastic spike. Rigid motion is untouched by construction (g0 = -(g1+g2+g3)),
# so free fall, the carry swing and the drop glide never see this term.
# 0 = EXACTLY the old purely-elastic solver, byte for byte.
# The volumetric row gets NONE by default: the collide rig measured it worse at
# high Poisson, where it is already the stiff row and rate-freezing it makes the
# body rigid. Left as a knob, not a hidden zero.
#
# MEASURED HERE, 2026-08-27, AND IT DOES NOT WORK IN THIS RIG -- default 0 is
# not a placeholder, it is the verdict. In the collide rig (free bodies meeting
# each other and a floor) tau = 10..40 ms is safe and 40 ms is best. Under the
# PADS it detonates: the frame the pad face meets the flank, cage volume goes
# 99 % -> 400-800 %, detF -> -1e3, and the force proxy trips at 150-370 N after
# 3.4 mm of close travel. That is true at tau = 1 ms as well as 40 (so it is not
# a gamma-magnitude tuning problem), and at ITERATIONS = 1, 2 and 4 (so it is
# not the multiplier being re-projected against a stale prev either -- fewer
# iterations is WORSE: 1798 % cage volume at ITERATIONS = 1). The SAME tau = 40
# is stable in this file's own --drop bench, which has gravity, the floor and
# the ledge but no pad: cage volume 97.5 %, detF 0.37..1.20. So the instability
# is specific to the pad contact, whose corrections are mass-split back onto the
# cage nodes THROUGH the skinning weights -- the rate term then reads the
# contact solve's own within-substep corrections as strain rate and pumps them.
# Fixing that means damping a rate measured before the contact sweeps, not
# after; it is not a knob change. Until then: leave this at 0.
FLESH_VISC = cli_arg("--flesh-visc", 0.0, float)          # s, deviatoric row
FLESH_VISC_VOL = cli_arg("--flesh-visc-vol", 0.0, float)  # s, volumetric row
TET_RES = cli_arg("--tet-res", 40, int)   # voxel cells along the fish's length
                      # (22 was visibly blocky; finer also hugs the skin tighter
                      # since the carve dilation is measured in cells)
CARVE = 0.25          # cells of dilation, so the cage measures the fish

# Contact radius, and it is a CLEARANCE budget as much as a collision one: the
# pads' effective surface stands its radius proud of the particles, so a fat
# radius means the open hand is already pressing the fish as it descends and
# ploughs it out of the way before the close begins.
# Radius is PER PARTICLE now, because the two bodies are discretised at
# completely different pitches: the pad nodes sit 6 mm apart and want a small
# radius for clearance, while the fish's tet cage is a blocky lattice whose
# nodes sit a cell apart and needs ~0.45 cells to read as a smooth flank rather
# than a bag of marbles (and to stop a pad node slipping between four cage
# nodes). Contact distance is the SUM of the two radii.
# PHASE 11 (corotational fish): the contact radii are CLEARANCE now, and they
# have to be honest about an 80 mm hand. A full-size 240 mm Barramundi body is
# ~60 mm wide; at FINGER_OPEN the pad faces stand 2*(40+6-4-PART_R) mm apart,
# so with PART_R 5 + CAGE_R 4.9 the OPEN hand already overlapped the fish by
# ~3 mm a side on the way down (119 N of 'grip' during DESCEND, fish ploughed
# 7 cm along the tray). The half-volume fish of phases 6-10 hid that. 3 + 3.3
# leaves ~2.7 mm of clearance per side at full open.
PART_R = cli_arg("--pad-r", 0.003, float)     # pad / finger nodes
# PHASE 10: the contact radius WAS the visible gap. contact_r = PART_R + CAGE_R
# stops the pad particle that far short of a cage node, and the pad face is drawn
# AT the particle, so a fat contact_r left the amber pad ~1 cm proud of the drawn
# fish even while the physics reported a firm bite. LEVER 2 (shrink contact_r) was
# tried at 0.10..0.25 cells / TET_RES 22..30 and CONSISTENTLY broke the RELEASE: a
# wide contact band actively pushes the fish off the pad as the jaws open, while a
# thin band leaves only tacky friction, which drags the fish back up on RETREAT
# (touch stayed 66..262 at DONE vs phase-9's clean 0). Clean ejection needs
# contact_r ~9.9 mm. So the physics is left EXACTLY at phase 9 (grip, release, FEM
# all unchanged) and the visible gap is closed render-only by RENDER_PAD_INSET.
# PHASE 11: the cage nodes no longer collide with anything (see the surface
# contact below); CAGE_R only sizes the cage's own safety floor and a couple of
# diagnostics.
CAGE_R_CELLS = 0.30
# Surface contact query margin: a pad node looks for the fish skin this far
# beyond its own radius when the substep's contacts are latched.
CONTACT_MARGIN = 0.004
# How deep inside the fish a finger/palm node can still FIND the skin. With the
# old 7 mm search radius, a node squeezed deeper than that lost its contact
# silently and was then free INSIDE the fish -- the "fish fuses with the bent
# finger" the user caught in the FEM view. Recovery pushes are capped per sweep
# so a deeply embedded node heals over a few substeps instead of detonating.
CONTACT_DEEP = 0.030
CONTACT_HEAL = 0.008
SURF_R = 0.0005         # the skin's own thickness against the tray / crate

# 0.7: wet-slimy fish on silicone. 1.10 was tuned when friction was the only
# thing holding a half-volume fish; at full contact it made the open hand a
# trap -- the wedged fish would hang between the OPEN pads by stiction alone
# and ride the retreat. Carry margin at 0.7 is still ~2*0.7*N / 2.1 N >> 2.
MU_PAD = cli_arg("--mu-pad", 0.70, float)
MU_STEEL = cli_arg("--mu-steel", 0.25, float)   # fish on the stainless tray / table / belt
MU_CRATE = 0.35

# --- cell geometry --------------------------------------------------------------

TABLE_Y = 0.0
TRAY_C = (0.46, 0.10)         # tray centre (x, z)
TRAY_HX, TRAY_HZ = 0.22, 0.16
TRAY_Y = 0.018                # tray floor height
CRATE_C = (0.36, -0.45)
CRATE2_C = (0.36, 0.45)
CRATE_HX, CRATE_HZ = 0.19, 0.13
CRATE_FLOOR = 0.03
CRATE_RIM = 0.20
TRAY_CX, TRAY_CZ = float(TRAY_C[0]), float(TRAY_C[1])
# The drop ledge: a raised slab in the tray's footprint whose +x edge the fish
# comes off. Only solid when --drop (kernel constant).
LEDGE_ON = 1 if DROP else 0
LEDGE_Y = 0.22
LEDGE_X0, LEDGE_X1 = 0.10, 0.46
LEDGE_CZ, LEDGE_HZ = 0.10, 0.16
CRATE_CX, CRATE_CZ = float(CRATE_C[0]), float(CRATE_C[1])
CRATE2_CZ = float(CRATE2_C[1])

# A herring, not a salmon. The stock FR3 hand opens to 8 cm, so what limits the
# size is the WIDTH, never the length: 45 mm is about all the carriages can bite
# into and still have travel left. Length is a free parameter (--fish-len), and
# the three graded sizes are 0.24 / 0.27 / 0.30 m.
FISH_LEN = cli_arg("--fish-len", 0.24, float)
FISH_H = 0.22 * FISH_LEN
FISH_W = min(0.045, 0.17 * FISH_LEN)
FISH_SUBDIV = 3
FISH_P = (TRAY_C[0], TRAY_Y + FISH_H * 0.5, TRAY_C[1])
# Grasp station: the body's CENTRE OF MASS (computed off the tet cage further
# down, since the cage is the mass). Phase 2b gripped 40 % back from the nose --
# the thick shoulder -- and the fish promptly pivoted to hang vertically by that
# shoulder with only a third of its length between the pads, then walked out of
# the hand between CARRY and LOWER. Held at the COM it hangs balanced and the
# head and tail merely droop.
GRADE_MM = cli_arg("--grade-mm", 265.0, float)   # >= this goes in the LARGE crate

# Fin Ray finger, in the TCP frame: length along the tool axis (+z), width
# across the grasp (x), the two skins separated in y.
# Particle spacing has to be FINER than the contact diameter or the pad is a
# sieve: the fish bulges through the gaps between pad particles and the grip
# leaks. 14 x 9 over 50 x 75 mm puts the nodes ~6 mm apart against a 7 mm
# contact diameter.
# The paddle is WIDE along the fish and only as tall as the fish is: a narrow
# tall pad pinches a 50 mm patch of a 220 mm body, the fish pivots about that
# patch and hangs out of the hand by one end (measured: it did exactly that).
# 90 mm of flank between the pads is a cradle, and the head and tail still
# droop off the ends.
FIN_ROWS, FIN_COLS = 11, 19
FIN_Z0, FIN_LEN = -0.032, 0.060   # root at the carriage, tip ~28 mm past the TCP
FIN_WIDTH = 0.110                 # across the grasp, i.e. along the fish
FIN_SKIN_IN = 0.004               # front skin, inboard of the carriage origin
# The tip hooks INWARD. Two flat plates pinching a floppy shell is not a grasp:
# the fish squirts out along its own length and the pads never stop it. Curving
# the last third of the front skin toward the other finger puts a lip under the
# fish, which is form closure -- the fish has to deform its way OUT to escape,
# not merely slide.
# PHASE 11: OFF by default. A 10 mm hook put the tip's contact surface 24 mm
# from the tool axis, so on a 55 mm wide (real-size) fish the OPEN hand's tips
# landed on the back and raked down the flanks, shoving the fish sideways before
# the close. The hook was a crutch for the phase-2 shell fish that squirted out
# of flat pads; a stiff corotational fish with Coulomb friction is held by the
# pads themselves, and real Fin Ray fingers are straight when open -- they curl
# under load, which the soft diagonals still do. --fin-curl 0.010 brings it back.
# 3 mm for the SIDE-LYING pick: at 10 mm the tip gap was 48.6 mm against a
# 60 mm fish -- the tips ploughed the flanks on descent (20 N before the close
# began) and caged the fish on OPEN so it rode the hand up ("stuck in the
# gripper"). 3 mm keeps a lip against axial squirt-out without closing the exit.
# 3 mm: enough of a lip that the almond cross-section cannot squirt axially
# while the close settles (at 0 the fish squirted 46 mm and jack-knifed into
# the wrist at 75 N), small enough that the tips clear the flanks on descent
# and the fish can leave the open hand (10 mm caged it).
FIN_CURL = cli_arg("--fin-curl", 0.003, float)
FIN_SKIN_OUT = 0.010              # back skin, outboard
# Heavy relative to the fish ON PURPOSE. The contact correction is split by
# inverse mass, so a light pad simply gets shoved aside by the fish instead of
# clamping it -- the mass ratio IS the clamping stiffness at a contact.
FIN_MASS = 0.8
RIB_EVERY = 2                     # draw a cross rib every Nth row (render only)
# PHASE 10, and the whole gap fix now lives here. The pad particle stops contact_r
# (~9.9 mm) short of a cage node and the pad face is drawn AT the particle, so the
# amber face reads ~contact_r minus the indent (~2 mm) = ~8 mm proud of the drawn
# fish. Shift ONLY the drawn FRONT-skin vertices inward (toward the other finger)
# by this much when building the finger render soup each frame -- applied to BOTH
# the skin soup and the rib soup's front edge so they stay joined. This is a pure
# render move: the physics particles, the truss and the contact nodes do not move,
# so the grasp, the release and the FEM are byte-for-byte the phase-9 ones. Front
# and back skins sit ~14 mm apart, so 8 mm inward leaves a ~6 mm drawn pad -- thin
# but still a pad, not a collapsed sheet (verified by looking). The back skin and
# the ribs' back edge are NOT touched.
# PHASE 11: the pad touches the DRAWN fish skin at PART_R from the particle, so
# the drawn front skin is inset by exactly that -- what you see is what touches.
RENDER_PAD_INSET = PART_R         # m, drawn front skin only (0 disables)

URDF = "C:/dev/threepp/cmake-build-relwithdebinfo/_deps/threepp_data-src/urdf/franka/fr3.urdf"
if not os.path.exists(URDF):
    URDF = "C:/dev/threepp-data/urdf/franka/fr3.urdf"


# --- fit the cage + pads to the REAL fish body (phase 7 defect fix) --------------
# The procedural FISH_H/FISH_W above are ~half a real fish: the Khronos Barramundi
# at FISH_LEN=0.24 m is ~92 mm tall and ~60 mm wide once scaled, but the cage was
# built to the procedural 53 x 41 mm. So the pads closed on a small invisible core
# while the visible fish ballooned around it and dangled off one corner -- the
# "empty space between the fish and the gripper" the user saw. Fix: load the model
# UP FRONT, measure its BODY extents (thin fins excluded via a robust percentile
# over the central-length vertices), and resize the cage, the spawn height and the
# pad paddle to the real flesh before anything downstream is built. The procedural
# path is untouched.
def load_fish_model(path, length):
    """Load a fish mesh; return (verts, faces, uv, material) with the verts in
    the procedural template frame: length along +X, dorsal +Y, lateral +Z,
    centred at the origin, scaled so the long axis is `length` metres.

    EVERY mesh in the file is welded, not just the first: the Khronos Barramundi
    is one mesh, but an FHF photogrammetry .usdz scan is a scene graph and taking
    found[0] silently loaded a fragment (or a stray helper) instead of the fish.
    Missing indices / UVs are synthesised rather than crashing, since a scanned
    surface is often non-indexed and need not carry a UV set.

    ORIENTATION is measured, not assumed. The old path hard-coded a +90 deg Y
    rotation because the Barramundi's length runs along +Z; a scan's axes are
    whatever the photogrammetry rig happened to use. Sorting the bbox extents
    puts the longest axis on +X, the middle (dorsal-ventral) on +Y and the
    thinnest (lateral) on +Z -- exactly the template frame -- and the permutation
    is corrected to a proper rotation so the fish is never mirrored and the
    surface winding, which the contact normals ride on, survives. On the
    Barramundi this reproduces the old Ry90 up to a lateral sign.

    The scan need not be CLOSED: nothing here needs an inside test. The cage is
    carved from the per-slice body profile (body_profile) and the contact surface
    is a wp.Mesh of the body triangles, both of which are winding-agnostic."""
    root = tp.ModelLoader().load(path)
    if root is None:
        sys.exit(f"could not load {path}")
    root.update_matrix_world(True)
    found = []
    root.traverse(lambda o: found.append(o) if isinstance(o, tp.Mesh) else None)
    verts, faces, uvs, material, base = [], [], [], None, 0
    for mesh in found:
        g = mesh.geometry
        if g is None:
            continue
        p = g.get_attribute("position")
        if p is None or not len(p):
            continue
        P = np.asarray(p, np.float64).reshape(-1, 3)
        M = np.asarray(mesh.matrix_world.to_numpy(), np.float64).reshape(4, 4)
        verts.append((M[:3, :3] @ P.T).T + M[:3, 3])     # bake the node transform
        idx = g.get_index()
        idx = (np.arange(len(P), dtype=np.int64) if idx is None
               else np.asarray(idx, np.int64).reshape(-1))
        faces.append(idx.reshape(-1, 3) + base)
        uv = g.get_attribute("uv")
        uvs.append(np.zeros((len(P), 2), np.float32) if uv is None
                   else np.asarray(uv, np.float32).reshape(-1, 2))
        base += len(P)
        if material is None:
            material = mesh.material
    if not verts:
        sys.exit(f"no geometry in {path}")
    Pw = np.concatenate(verts)
    faces = np.concatenate(faces).astype(np.int32)
    uv = np.concatenate(uvs)
    ext = Pw.max(0) - Pw.min(0)
    order = np.argsort(ext)                              # [thinnest, middle, longest]
    perm = [order[2], order[1], order[0]]                # -> [X long, Y dorsal, Z lateral]
    Pr = Pw[:, perm]
    if np.linalg.det(np.eye(3)[perm]) < 0:               # keep it a ROTATION
        Pr = Pr * np.array([1.0, 1.0, -1.0])
    Pr -= 0.5 * (Pr.max(0) + Pr.min(0))                  # centre at origin
    scale = length / (Pr[:, 0].max() - Pr[:, 0].min())
    print(f"  fish-model: {len(found)} mesh node(s) welded to {len(Pw)} verts / "
          f"{len(faces)} tris, raw bbox {ext[perm][0]:.3f} x {ext[perm][1]:.3f} x "
          f"{ext[perm][2]:.3f} -> scaled x{scale:.4f} to {length * 1000:.0f} mm")
    return (Pr * scale).astype(np.float32), faces, uv, material


def body_profile(rv, nb=28, keep=90.0):
    """Per-slice body half-extents (yhw, zhw) along the fish's long axis (+x),
    with the thin fins REMOVED so the cage is carved to the FLESH, not the fins.

    Slice x into `nb` bins. The two vertical fins (dorsal, anal) and the caudal
    fan all lie in the median plane (z ~ 0) and are TALL but paper-THIN in z: if
    the half-height were taken over all verts they would set a 100 mm cage on a
    60 mm fish (measured). So the half-height is taken only over FLANK verts --
    those a few mm off the median plane -- which are the body's own sides and
    exclude every median-plane fin. The half-width is a high percentile of |z|
    with the odd pectoral spike trimmed by a median gate. Both profiles are then
    linearly interpolated across empty slices and lightly smoothed. Returns
    (xc, yhw, zhw)."""
    x = rv[:, 0].astype(np.float64)
    ay = np.abs(rv[:, 1].astype(np.float64))
    az = np.abs(rv[:, 2].astype(np.float64))
    edges = np.linspace(x.min(), x.max(), nb + 1)
    xc = 0.5 * (edges[:-1] + edges[1:])
    bidx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, nb - 1)

    def fill_smooth(v):
        idx = np.arange(nb)
        good = ~np.isnan(v)
        if not good.any():
            return np.zeros(nb)
        v = np.interp(idx, idx[good], v[good])
        return np.convolve(np.pad(v, 1, mode="edge"), np.array([0.25, 0.5, 0.25]), mode="valid")

    # Width: 90th percentile of |z| per slice. NO median gate: on slices where
    # median-plane fin verts outnumber the (sparsely tessellated) mid-body flank
    # the gate threw away the BODY and kept the fins -- measured, zhw collapsed
    # to 0.9-3.4 mm at x = +4..+21 mm and the cage necked to a one-cell sliver
    # (the hinge the fish folded over in the drop). p90 is already robust to a
    # few pectoral outliers; single-slice spikes die in a cross-slice median-of-3.
    zhw = np.full(nb, np.nan)
    for i in range(nb):
        m = bidx == i
        if int(m.sum()) >= 4:
            zhw[i] = np.percentile(az[m], keep)
    good = ~np.isnan(zhw)
    med3 = zhw.copy()
    gi = np.nonzero(good)[0]
    for j, i in enumerate(gi):
        nbrs = zhw[gi[max(j - 1, 0):j + 2]]
        med3[i] = np.median(nbrs)
    zhw = np.maximum(fill_smooth(med3), 1.0e-4)

    # Height over FLANK verts only (|z| off the median plane), so the dorsal /
    # anal / caudal fins -- which live at z ~ 0 -- never set the cage height.
    yhw = np.full(nb, np.nan)
    for i in range(nb):
        zf = max(0.004, 0.30 * zhw[i])
        m = (bidx == i) & (az >= zf)
        if int(m.sum()) >= 4:
            yhw[i] = np.percentile(ay[m], keep)
    yhw = np.maximum(fill_smooth(yhw), 1.0e-4)
    return xc, yhw, zhw


FISH_MODEL_CACHE = None
PROF_XC = PROF_YHW = PROF_ZHW = None
if FISH_MODEL and os.path.exists(FISH_MODEL):
    FISH_MODEL_CACHE = load_fish_model(FISH_MODEL, FISH_LEN)
    _rv = FISH_MODEL_CACHE[0]
    # PHASE 9: the cage is carved to the real per-slice body profile, so the
    # collision surface MATCHES the render surface -- what you see is what you
    # touch. The old path measured only two numbers (a symmetric ellipsoid) and
    # capped the height for stability; now both axes follow the flesh.
    PROF_XC, PROF_YHW, PROF_ZHW = body_profile(_rv, nb=28)
    _old_h, _old_w = FISH_H, FISH_W
    # Full body height (not capped to a stable aspect any more): the conforming
    # belly holds the fish upright better than the old ellipsoid egg, and the
    # user wants the touch cage to match the visible body in Y as well as Z.
    FISH_W = 2.0 * float(PROF_ZHW.max())
    FISH_H = 2.0 * float(PROF_YHW.max())
    FISH_P = (TRAY_C[0], TRAY_Y + FISH_H * 0.5, TRAY_C[1])
    print(f"  fish-model body profile: conforming cage {FISH_H * 1000:.0f} x {FISH_W * 1000:.0f} mm "
          f"over {len(PROF_XC)} slices "
          f"(procedural was {_old_h * 1000:.0f} x {_old_w * 1000:.0f} mm)")


# --- [B] robot ------------------------------------------------------------------

robot = tp.URDFLoader().load(URDF)
robot.rotation.x = -math.pi / 2          # URDF is Z-up, the world is Y-up
robot.position.set(0.0, 0.0, 0.0)
robot.update_matrix_world(True)

# The white wireframe junk in the raw render is the URDF's COLLISION meshes, not
# a material problem: Vulkan draws the DAE's own MeshPhongMaterial perfectly
# well, and RobotCell (the C++ demo that looks right) never touches materials --
# it just hides the colliders. Phase 1 and 2 swapped all 54 materials for flat
# grey Standards on that misdiagnosis, which is why the arm read grey and dead.
robot.show_colliders(False)

Q_HOME = [0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.785, 0.04, 0.04]
FINGER_OPEN, FINGER_SHUT = 0.04, 0.0
# A hard stop on the close. The stock FR3 hand opens to 8 cm and the fish is
# 4.6 cm across, so the whole useful travel is about a centimetre; without a
# floor under the force control a soft fish just gets squeezed out sideways
# rather than pushing back, and the carriages run to zero.
FINGER_MIN = 0.021

ik_opts = tp.IkOptions()
ik_opts.task = tp.IkTask.Pose
ik_opts.max_iterations = 60
ik_opts.orientation_weight = 0.6
ik_opts.rest_pose = list(Q_HOME)
ik_opts.rest_pose_gain = 0.02
ik = tp.IkSolver(robot, ik_opts)

left_link = robot.get_object_by_name("fr3_leftfinger")
right_link = robot.get_object_by_name("fr3_rightfinger")


def tool_pose(x, y, z, yaw=0.0):
    """TCP target: tool +z (the approach axis) pointing straight down, spun by
    `yaw` about it. At the FR3's zero pose the TCP +z already points along
    world -y once the robot is laid Y-up, so 'down' is the identity here."""
    q = tp.Quaternion().set_from_axis_angle(tp.Vector3(1, 0, 0), math.pi / 2)
    if yaw:
        qy = tp.Quaternion().set_from_axis_angle(tp.Vector3(0, 1, 0), yaw)
        # world-frame yaw premultiplies
        m = tp.Matrix4().compose(tp.Vector3(0, 0, 0), qy, tp.Vector3(1, 1, 1))
        m.multiply(tp.Matrix4().compose(tp.Vector3(0, 0, 0), q, tp.Vector3(1, 1, 1)))
        m.set_position(x, y, z)
        return m
    return tp.Matrix4().compose(tp.Vector3(x, y, z), q, tp.Vector3(1, 1, 1))


def ik_to(target, q_seed):
    """Loop the incremental solve to convergence; returns (q, IkResult)."""
    q = list(q_seed)
    res = None
    for _ in range(12):
        q, res = ik.solve(q, target)
        if res.converged:
            break
    return q, res


def set_q(q):
    robot.set_joint_values(list(q))
    robot.update_matrix_world(True)


def link_mat(link):
    return np.array(link.matrix_world.to_numpy(), dtype=np.float32).reshape(4, 4)


# --- [C] soft bodies: build the particle system ---------------------------------

# Reference pose for laying out the fingers: hand down, fully open.
q_ref = list(Q_HOME)
q_ref[7] = q_ref[8] = FINGER_OPEN
set_q(q_ref)
M_TCP_REF = np.array(ik.tool_transform(q_ref).to_numpy(), dtype=np.float32).reshape(4, 4)
CARRIAGE_REF = [link_mat(left_link), link_mat(right_link)]

# Which side of the TCP each carriage sits on, in TCP coordinates.
_inv_tcp = np.linalg.inv(M_TCP_REF)
CARRIAGE_TCP = [(_inv_tcp @ np.append(C[:3, 3], 1.0))[:3] for C in CARRIAGE_REF]
SIDE = [1.0 if c[1] >= 0 else -1.0 for c in CARRIAGE_TCP]


def fin_ray_particles(k):
    """The finger's particles in the TCP frame at the reference pose, as
    (rows, cols, 2 skins, 3). Front skin faces the other finger."""
    s = SIDE[k]
    y_c = CARRIAGE_TCP[k][1]
    pts = np.zeros((FIN_ROWS, FIN_COLS, 2, 3), dtype=np.float32)
    for i in range(FIN_ROWS):
        z = FIN_Z0 + FIN_LEN * i / (FIN_ROWS - 1)
        for j in range(FIN_COLS):
            x = CARRIAGE_TCP[k][0] + FIN_WIDTH * (j / (FIN_COLS - 1) - 0.5)
            hook = FIN_CURL * max(0.0, z / 0.030) ** 2 if z > 0.0 else 0.0
            pts[i, j, 0] = (x, y_c - s * (FIN_SKIN_IN + hook), z)   # front (inboard)
            pts[i, j, 1] = (x, y_c + s * FIN_SKIN_OUT, z)  # back (outboard)
    return pts


def finger_faces(base):
    """Render triangles for one finger, as TWO soups: (skins, ribs).

    Phase 3 drew the truss as a closed box, and a closed box in a dark material
    is exactly what it looked like -- a black block with nothing to say it was
    soft. A Fin Ray finger is two thin skins joined by cross ribs, and the
    ribs SHEARING as the front skin is pushed back is the whole visual idea, so
    the sides stay open and the ribs are drawn: a strut plate from the front
    skin to the back skin, spanning the finger's width, every RIB_EVERY rows.
    They are separate soups because they need separate smooth normals (a rib's
    normal is perpendicular to the skin it lands on) and separate materials.
    """
    def pid(i, j, skin):
        return base + skin * FIN_ROWS * FIN_COLS + i * FIN_COLS + j

    skins, ribs = [], []

    def quad(f, a, b, c, d):
        f.append((a, b, c))
        f.append((a, c, d))

    for i in range(FIN_ROWS - 1):
        for j in range(FIN_COLS - 1):
            quad(skins, pid(i, j, 0), pid(i, j + 1, 0), pid(i + 1, j + 1, 0), pid(i + 1, j, 0))
            quad(skins, pid(i, j, 1), pid(i + 1, j, 1), pid(i + 1, j + 1, 1), pid(i, j + 1, 1))
    rows = list(range(0, FIN_ROWS, RIB_EVERY))
    if rows[-1] != FIN_ROWS - 1:
        rows.append(FIN_ROWS - 1)          # the tip rib closes the two skins
    for i in rows:
        for j in range(FIN_COLS - 1):
            quad(ribs, pid(i, j, 0), pid(i, j, 1), pid(i, j + 1, 1), pid(i, j + 1, 0))
    return np.array(skins, dtype=np.int32), np.array(ribs, dtype=np.int32)


def finger_pairs(base):
    """Two-skin truss: stiff skins, stiff ribs, SOFT diagonals. The diagonals
    are what let the finger shear, and shear along the length IS the Fin Ray
    curl -- a front skin pushed back drags the tip inward."""
    def pid(i, j, skin):
        return base + skin * FIN_ROWS * FIN_COLS + i * FIN_COLS + j

    p = []
    for skin in (0, 1):
        for i in range(FIN_ROWS):
            for j in range(FIN_COLS):
                if i + 1 < FIN_ROWS:
                    p.append((pid(i, j, skin), pid(i + 1, j, skin), STIFF_SKIN))
                if j + 1 < FIN_COLS:
                    p.append((pid(i, j, skin), pid(i, j + 1, skin), STIFF_SKIN))
                if i + 1 < FIN_ROWS and j + 1 < FIN_COLS:
                    # The shear pair across every cell. Already at full skin
                    # stiffness, so the sheet cannot shear IN PLANE -- what it
                    # was missing is the out-of-plane term below.
                    p.append((pid(i, j, skin), pid(i + 1, j + 1, skin), STIFF_SHEAR))
                    p.append((pid(i, j + 1, skin), pid(i + 1, j, skin), STIFF_SHEAR))
                # Bending: skip-a-node chords, root->tip (the bending direction)
                # and across the finger. A chord over two cells is only violated
                # by CURVATURE -- slide the middle node sideways along the sheet
                # and the chord does not change length -- so this buys smooth
                # arcs instead of buckles and costs the truss no shear freedom.
                if i + 2 < FIN_ROWS:
                    p.append((pid(i, j, skin), pid(i + 2, j, skin), STIFF_BEND))
                if j + 2 < FIN_COLS:
                    p.append((pid(i, j, skin), pid(i, j + 2, skin), STIFF_BEND))
                if STIFF_BEND2 > 0.0 and i + 3 < FIN_ROWS:
                    p.append((pid(i, j, skin), pid(i + 3, j, skin), STIFF_BEND2))
    for skin in (0, 1):
        for i in range(3, FIN_ROWS):
            for j in range(FIN_COLS):
                p.append((pid(1, j, skin), pid(i, j, skin), STIFF_FAN))
                if skin == 0:
                    p.append((pid(1, j, 1), pid(i, j, 0), STIFF_FAN))
    for i in range(FIN_ROWS):
        for j in range(FIN_COLS):
            p.append((pid(i, j, 0), pid(i, j, 1), STIFF_RIB))
            if i + 1 < FIN_ROWS:
                p.append((pid(i, j, 0), pid(i + 1, j, 1), STIFF_DIAG))
                p.append((pid(i, j, 1), pid(i + 1, j, 0), STIFF_DIAG))
    return p


FIN_N = FIN_ROWS * FIN_COLS * 2
positions, pairs, body_of, mu_of, invm_of, pad_of, pinned_of = [], [], [], [], [], [], []
fin_local = []      # per finger: (N, 3) in the carriage frame
fin_faces, rib_faces = [], []

for k in (0, 1):
    base = len(positions)
    pts_tcp = fin_ray_particles(k).transpose(2, 0, 1, 3).reshape(-1, 3)   # skin-major
    world = (M_TCP_REF[:3, :3] @ pts_tcp.T).T + M_TCP_REF[:3, 3]
    Cm = CARRIAGE_REF[k]
    local = (np.linalg.inv(Cm)[:3, :3] @ world.T).T + np.linalg.inv(Cm)[:3, 3]
    fin_local.append(local.astype(np.float32))
    _sk, _rb = finger_faces(base)
    fin_faces.append(_sk)
    rib_faces.append(_rb)
    pairs += finger_pairs(base)
    m = FIN_MASS / FIN_N
    for n in range(FIN_N):
        skin, i = n // (FIN_ROWS * FIN_COLS), (n % (FIN_ROWS * FIN_COLS)) // FIN_COLS
        root = i <= 1                       # the root rows ride the carriage
        positions.append(world[n])
        body_of.append(k)
        mu_of.append(MU_PAD)
        invm_of.append(0.0 if root else 1.0 / m)
        pad_of.append(1 if skin == 0 else 0)
        pinned_of.append(1 if root else 0)

# THE PALM. The real FR3 hand has a solid body between the finger roots; the
# sim had an open funnel there, and the almond cross-section of a side-lying
# fish gets pumped UP the funnel by every squeeze/acceleration until it wedges
# in the "wrist" and rides the retreat ("the fish gets stuck inside the
# gripper"). A pinned plate of contact particles across the root plane gives
# the squeezed fish a ceiling to seat against -- two pads + palm, like the real
# grasp. Pinned to the TOOL frame (owner 2), zero inverse mass: the fish side
# takes the whole contact correction.
PALM_NX, PALM_NY = 19, 13
_palm_local = []
for _pi in range(PALM_NX):
    for _pj in range(PALM_NY):
        _palm_local.append((FIN_WIDTH * (_pi / (PALM_NX - 1) - 0.5),
                            2.0 * abs(CARRIAGE_TCP[0][1]) * (_pj / (PALM_NY - 1) - 0.5),
                            FIN_Z0 - PART_R))
_palm_local = np.array(_palm_local, dtype=np.float32)
PALM_BASE = len(positions)
_palm_world = (M_TCP_REF[:3, :3] @ _palm_local.T).T + M_TCP_REF[:3, 3]
for _pw in _palm_world:
    positions.append(_pw.astype(np.float32))
    body_of.append(3)
    mu_of.append(MU_PAD)
    invm_of.append(0.0)
    pad_of.append(0)
    pinned_of.append(1)
PALM_N = len(_palm_local)

FIN_TOTAL = len(positions)

# --- the fish: render surface + filled tet cage ---------------------------------

# The drawn surface is skinned off the cage. By default it is the tapered
# ellipsoid (self-contained, runs anywhere); with --fish-model it is a glTF fish
# (the Khronos Barramundi) bound to the SAME body-sized tet cage, so the FEM and
# the grasp are identical -- only the skin changes. load_fish_model() is defined
# up top, since --fish-model resizes the cage + pads to the real body before the
# fingers and cage are built.
unit, fish_faces_t = icosphere(FISH_SUBDIV)


def fish_taper(ux):
    return 1.0 - 0.45 * (np.clip(ux, -1.0, 1.0) * 0.5 + 0.5)


taper = fish_taper(unit[:, 0]).astype(np.float32)
fish_tmpl = np.stack([unit[:, 0] * (FISH_LEN / 2),
                      unit[:, 1] * (FISH_H / 2) * taper,
                      unit[:, 2] * (FISH_W / 2) * taper], axis=-1).astype(np.float32)
FISH_NV, FISH_NF = len(fish_tmpl), len(fish_faces_t)
# Water, near enough: 1000 kg/m^3 on the template's own volume. 0.24 m of herring
# comes out at ~0.21 kg, which is what a 0.24 m herring weighs.
FISH_MASS = 1000.0 * abs(signed_volume(fish_tmpl, fish_faces_t))
# ...unless the fish was WEIGHED. The FHF scan database ships a measured
# weight_kg beside every scan (CF2504MR0096: 0.63 m cod, 3.74 kg), and a
# density-times-a-fitted-ellipsoid estimate is not that number. --fish-mass sets
# the cage's total mass directly, so the grasp is tested against the real load.
_FISH_MASS_CLI = cli_arg("--fish-mass", 0.0, float)      # kg, 0 = density estimate
if _FISH_MASS_CLI > 0.0:
    FISH_MASS = _FISH_MASS_CLI

# The two 5-tet splits of a cube, alternated by cell parity so neighbouring
# cells agree on the diagonal of every shared face (a conforming lattice).
_EVEN = np.array([[0, 1, 2, 4], [1, 2, 3, 7], [1, 4, 5, 7], [2, 4, 6, 7], [1, 2, 4, 7]])
_ODD = np.array([[0, 1, 3, 5], [0, 2, 3, 6], [0, 4, 5, 6], [3, 5, 6, 7], [0, 3, 5, 6]])
_TET_E = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def fish_depth(p):
    """Approximate signed distance to the tapered ellipsoid: the normalised
    radius, scaled back to metres by the smallest local semi-axis (which
    under-estimates the distance, so a carve against it never eats the body)."""
    ux = p[:, 0] / (FISH_LEN * 0.5)
    t = fish_taper(ux)
    ry, rz = FISH_H * 0.5 * t, FISH_W * 0.5 * t
    r = np.sqrt(ux ** 2 + (p[:, 1] / ry) ** 2 + (p[:, 2] / rz) ** 2)
    return (r - 1.0) * np.minimum(ry, rz)


def fish_cage(h):
    """Voxel-carve a conforming tet cage out of the fish. Cells are kept when
    their centre is within CARVE cells OUTSIDE the surface, not strictly inside:
    carving at the zero level set leaves the cage half a cell short on every
    face, which at this pitch is the whole snout and tail."""
    lo = np.array([-FISH_LEN, -FISH_H, -FISH_W], np.float64) * 0.5 - 1.5 * h
    dims = np.maximum(np.ceil((-2.0 * lo) / h).astype(int), 1)
    gi = np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"), -1).reshape(-1, 3)
    solid = (fish_depth(lo + (gi + 0.5) * h) < CARVE * h).reshape(dims)
    ci, cj, ck = np.nonzero(solid)
    used = np.zeros(tuple(dims + 1), bool)
    for c in range(8):
        used[ci + (c & 1), cj + ((c >> 1) & 1), ck + ((c >> 2) & 1)] = True
    cid = -np.ones(tuple(dims + 1), np.int64)
    ui, uj, uk = np.nonzero(used)
    cid[ui, uj, uk] = np.arange(len(ui))
    cv = (lo + np.stack([ui, uj, uk], 1) * h).astype(np.float32)
    corners = np.stack([cid[ci + (c & 1), cj + ((c >> 1) & 1), ck + ((c >> 2) & 1)]
                        for c in range(8)], 1)
    par = (ci + cj + ck) % 2
    tets = np.concatenate([corners[par == 0][:, _EVEN].reshape(-1, 4),
                           corners[par == 1][:, _ODD].reshape(-1, 4)])
    a, b, c_, d = (cv[tets[:, i]] for i in range(4))
    neg = np.einsum("ij,ij->i", np.cross(b - a, c_ - a), d - a) < 0.0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return cv, tets.astype(np.int32), lo, dims, solid, cid


def fish_cage_profile(xc, yhw, zhw, h):
    """PHASE 9: voxel-carve a CONFORMING tet cage from the measured body profile.

    Keep a cell when its centre is inside the per-slice half-extents:
    |y| <= yhw(x) and |z| <= zhw(x), interpolating the profile at the cell's x.
    This is the same conforming-cube -> 5-tet parity split (neighbouring cells
    agree on shared-face diagonals) and the same negative-volume flip as
    fish_cage; only the solid test changes from the symmetric ellipsoid SDF to
    the real body envelope, so the collision cage now has the fish's deep
    shoulder and slim tail instead of an ellipsoid."""
    xlo, xhi = float(xc.min()), float(xc.max())
    ymax, zmax = float(yhw.max()), float(zhw.max())
    lo = np.array([xlo - 1.5 * h, -ymax - 1.5 * h, -zmax - 1.5 * h], np.float64)
    hi = np.array([xhi + 1.5 * h, ymax + 1.5 * h, zmax + 1.5 * h], np.float64)
    dims = np.maximum(np.ceil((hi - lo) / h).astype(int), 1)
    gi = np.stack(np.meshgrid(*[np.arange(d) for d in dims], indexing="ij"), -1).reshape(-1, 3)
    cc = lo + (gi + 0.5) * h
    yh = np.interp(cc[:, 0], xc, yhw)
    zh = np.interp(cc[:, 0], xc, zhw)
    # Dilate by ~0.35 cell so the cage reaches the surface rather than sitting
    # half a cell short on every face (fish_cage's CARVE lesson).
    dl = 0.35 * h
    solid = ((np.abs(cc[:, 1]) <= yh + dl) & (np.abs(cc[:, 2]) <= zh + dl)
             & (cc[:, 0] >= xlo - h) & (cc[:, 0] <= xhi + h)).reshape(dims)
    ci, cj, ck = np.nonzero(solid)
    used = np.zeros(tuple(dims + 1), bool)
    for c in range(8):
        used[ci + (c & 1), cj + ((c >> 1) & 1), ck + ((c >> 2) & 1)] = True
    cid = -np.ones(tuple(dims + 1), np.int64)
    ui, uj, uk = np.nonzero(used)
    cid[ui, uj, uk] = np.arange(len(ui))
    cv = (lo + np.stack([ui, uj, uk], 1) * h).astype(np.float32)
    corners = np.stack([cid[ci + (c & 1), cj + ((c >> 1) & 1), ck + ((c >> 2) & 1)]
                        for c in range(8)], 1)
    par = (ci + cj + ck) % 2
    tets = np.concatenate([corners[par == 0][:, _EVEN].reshape(-1, 4),
                           corners[par == 1][:, _ODD].reshape(-1, 4)])
    a, b, c_, d = (cv[tets[:, i]] for i in range(4))
    neg = np.einsum("ij,ij->i", np.cross(b - a, c_ - a), d - a) < 0.0
    tets[neg] = tets[neg][:, [0, 2, 1, 3]]
    return cv, tets.astype(np.int32), lo, dims, solid, cid


def bind_lattice(surf, lo, h, dims, solid, cid):
    """Bind every render vertex to the 8 corners of ITS cell, trilinearly.

    The cage is a regular lattice, so there is no search to do. Weights are
    UNCLAMPED, so a vertex outside the cage -- and most of the skin is, since the
    carve only keeps cells near the surface -- reproduces its rest position
    exactly by extrapolation instead of being rounded off onto the cage."""
    idx = np.clip(np.floor((surf - lo) / h).astype(int), 0, dims - 1)
    bad = ~solid[idx[:, 0], idx[:, 1], idx[:, 2]]
    if bad.any():                                # snout / tail tips the carve missed
        sc = np.stack(np.nonzero(solid), 1)
        idx[bad] = sc[((idx[bad][:, None, :] - sc[None]) ** 2).sum(-1).argmin(1)]
    f = (surf - (lo + idx * h)) / h
    ids = np.zeros((len(surf), 8), np.int32)
    w = np.zeros((len(surf), 8), np.float32)
    for c in range(8):
        dx, dy, dz = c & 1, (c >> 1) & 1, (c >> 2) & 1
        ids[:, c] = cid[idx[:, 0] + dx, idx[:, 1] + dy, idx[:, 2] + dz]
        w[:, c] = ((f[:, 0] if dx else 1.0 - f[:, 0]) * (f[:, 1] if dy else 1.0 - f[:, 1])
                   * (f[:, 2] if dz else 1.0 - f[:, 2]))
    return ids, w


# The template's OWN max extent, which is what "undeformed" means. FISH_H is the
# height at taper 1, and the taper is 1 only at the nose where the ellipsoid has
# no height at all, so measuring droop against FISH_H reads a healthy fish as 80 %.
FISH_H_REST = float(fish_tmpl[:, 1].max() - fish_tmpl[:, 1].min())

CAGE_H = FISH_LEN / TET_RES
if FISH_MODEL_CACHE is not None:
    cage_v, cage_t, CAGE_LO, CAGE_DIMS, CAGE_SOLID, CAGE_ID = fish_cage_profile(
        PROF_XC, PROF_YHW, PROF_ZHW, CAGE_H)
else:
    cage_v, cage_t, CAGE_LO, CAGE_DIMS, CAGE_SOLID, CAGE_ID = fish_cage(CAGE_H)
CAGE_N, N_TETS = len(cage_v), len(cage_t)
CAGE_R = CAGE_R_CELLS * CAGE_H
# PHASE 9 grasp station. For the conforming cage, grip the SHOULDER: the deepest
# body station (max cross-section yhw*zhw), which for a real fish is forward of
# centre. With a faithful cage that station has a deep cross-section, so the pads
# catch many nodes and the clamp is strong -- AND it is roughly the centre of
# mass of a head-heavy fish, so the body hangs roughly level instead of by the
# tail. The procedural ellipsoid keeps its cage-centroid station.
if FISH_MODEL_CACHE is not None:
    _cross = PROF_YHW * PROF_ZHW
    FISH_COM_X = float(PROF_XC[int(np.argmax(_cross))])
    FISH_HALF_W = float(np.interp(FISH_COM_X, PROF_XC, PROF_ZHW))
else:
    # The cage nodes are a uniform lattice through a body of uniform density, so
    # their mean IS the centre of mass. It sits behind the mid-point because the
    # taper puts the meat in the front half.
    FISH_COM_X = float(cage_v[:, 0].mean())
    _near = np.abs(fish_tmpl[:, 0] - FISH_COM_X) < 0.012
    FISH_HALF_W = float(np.abs(fish_tmpl[_near, 2]).max())
GRASP_X = FISH_P[0] + FISH_COM_X

# The render surface: procedural ellipsoid, or the glTF fish bound to the same
# cage. bind_lattice's weights are unclamped, so vertices outside the body cage
# (the barramundi's fins) extrapolate from their nearest cell and deform with
# the flesh -- no separate fin rig needed for the glTF skin.
gltf_mat = None
if FISH_MODEL_CACHE is not None:
    render_v, render_faces, render_uv, gltf_mat = FISH_MODEL_CACHE   # loaded up top
    render_normals0 = None
else:
    if FISH_MODEL:
        print(f"  note: --fish-model {FISH_MODEL} not found; using the procedural fish")
    render_v = fish_tmpl
    render_faces = fish_faces_t
    render_uv = np.stack([unit[:, 0] * 0.5 + 0.5, unit[:, 1] * 0.5 + 0.5], -1).astype(np.float32)
    render_normals0 = unit
USE_GLTF = gltf_mat is not None    # the glTF fish brings its own fins + eye
FISH_NV, FISH_NF = len(render_v), len(render_faces)
skin_ids, skin_w = bind_lattice(render_v.astype(np.float64), CAGE_LO, CAGE_H,
                                CAGE_DIMS, CAGE_SOLID, CAGE_ID)
_bind_err = np.linalg.norm((cage_v[skin_ids] * skin_w[:, :, None]).sum(1) - render_v, axis=1)

# PHASE 11: the fish LIES ON ITS SIDE. A soft dead fish on a tray does not stand
# on its belly; it lies on a flank, so the hand closes across its back and
# belly. The cage is built and bound in the fish's own frame (dorsal +y), then
# rolled 90 deg about its length axis here -- the skin follows through the
# weights, nothing downstream cares which way is up.
FISH_ROLL = "--upright" not in sys.argv
if FISH_ROLL:
    cage_v = np.stack([cage_v[:, 0], -cage_v[:, 2], cage_v[:, 1]], 1).astype(np.float32)
    _rv_rolled = np.stack([render_v[:, 0], -render_v[:, 2], render_v[:, 1]], 1)
else:
    _rv_rolled = render_v.astype(np.float64)
# Rest the cage's lowest node 2 mm over the tray; the skin settles onto it.
FISH_P = (TRAY_C[0], TRAY_Y - float(cage_v[:, 1].min()) + 0.002, TRAY_C[1])
if DROP:
    # nose DROP_OVERHANG past the ledge edge, lying on its side on the ledge
    FISH_P = (LEDGE_X1 + DROP_OVERHANG - float(cage_v[:, 0].max()),
              LEDGE_Y - float(cage_v[:, 1].min()) + 0.002, LEDGE_CZ)
# Which render triangles are BODY (for contact): all three vertices inside, or
# within a third of a cell of, the cage. The fins extrapolate far outside it
# and stay render-only.
body_vert = np.ones(len(render_v), dtype=bool)
if FISH_MODEL_CACHE is not None:
    # BODY = inside the measured per-slice body profile (+3 mm): that is the
    # fin-free flank/back/belly envelope the pads are meant to meet. A weights-
    # based test let the dorsal and anal fin sheets through, and a pad node next
    # to a thin sheet reads as 9 mm "inside" it.
    _yh = np.interp(render_v[:, 0], PROF_XC, PROF_YHW) + 0.003
    _zh = np.interp(render_v[:, 0], PROF_XC, PROF_ZHW) + 0.003
    body_vert[:] = (np.abs(render_v[:, 1]) <= _yh) & (np.abs(render_v[:, 2]) <= _zh)
body_tri_mask = body_vert[render_faces].all(1)
body_tris = render_faces[body_tri_mask].astype(np.int32)
body_vids = np.nonzero(body_vert)[0].astype(np.int32)
print(f"  surface contact: {len(body_tris)} body triangles / {len(render_faces)} "
      f"({len(body_vids)} body vertices of {len(render_v)}); fish {'on its side' if FISH_ROLL else 'upright'}")

# Grasp station (see the phase-9 block above). For the glTF fish it is the
# deepest body cross-section (the shoulder); for the procedural fish it is the
# cage centroid. `--grasp-shift` (metres, +x) nudges it manually.
GRASP_SHIFT = cli_arg("--grasp-shift", 0.0, float)
GRASP_X = FISH_P[0] + FISH_COM_X + GRASP_SHIFT
# Half-extent of the BODY skin along the closing axis (world z) at the grasp
# station -- measured on the body vertices of the (rolled) render mesh, so the
# backstop is about the surface the pads actually meet, fins excluded.
_near = body_vert & (np.abs(_rv_rolled[:, 0] - (FISH_COM_X + GRASP_SHIFT)) < 0.014)
if _near.sum() >= 3:
    FISH_HALF_W = float(np.abs(_rv_rolled[_near, 2]).max())

# NEO-HOOKEAN FEM. No distance edges, no volume row, no spine chords -- the tet
# cage is a real elastic solid. Per tet we precompute the rest inverse shape
# matrix Dm_inv (Dm columns = [x1-x0, x2-x0, x3-x0]) and the rest volume V; the
# deformation gradient F = Ds @ Dm_inv is recomputed each iteration on the
# device. The XPBD compliance of each constraint is alpha = 1/(k*V), and the
# per-substep alpha~ = alpha/dt^2 is precomputed here. There are no CSR rows for
# the fish now -- it goes through `solve` only for contacts.
FISH_BASE = FIN_TOTAL
_e = np.unique(np.sort(np.concatenate([cage_t[:, [a, b]] for a, b in _TET_E]), 1), axis=0)
_tv = cage_v[cage_t].astype(np.float64)                       # (N_TETS, 4, 3)
_Dm = np.stack([_tv[:, 1] - _tv[:, 0], _tv[:, 2] - _tv[:, 0],
                _tv[:, 3] - _tv[:, 0]], axis=2)                # columns
_detDm = np.linalg.det(_Dm)
tet_rest_vol = np.maximum(np.abs(_detDm) / 6.0, 1.0e-12).astype(np.float32)
tet_Dm_inv = np.linalg.inv(_Dm).astype(np.float32)            # (N_TETS, 3, 3)
_dt2 = DT * DT
tet_alpha_d = (1.0 / (MU_LAME * tet_rest_vol.astype(np.float64) * _dt2)).astype(np.float32)
tet_alpha_h = (1.0 / (LAM_LAME * tet_rest_vol.astype(np.float64) * _dt2)).astype(np.float32)
_YOUNG0, _KFIN0 = YOUNG, K_FIN          # bases for the live imgui rescale
# Relaxation time -> XPBD damping coefficient. gamma is the SAME number for every
# tet regardless of volume (the damping is stiffness-proportional, Rayleigh), so
# these are plain scalars baked into the CUDA graph at capture.
GAMMA_D = float(FLESH_VISC / DT)
GAMMA_H = float(FLESH_VISC_VOL / DT)
# Rest edge lengths, for the max-stretch diagnostic only (not a constraint).
_rest_edge_len = np.linalg.norm(cage_v[_e[:, 0]] - cage_v[_e[:, 1]], axis=1).astype(np.float32)

cage_p0 = cage_v + np.array(FISH_P, dtype=np.float32)
mfish = FISH_MASS / CAGE_N
for n in range(CAGE_N):
    positions.append(cage_p0[n])
    body_of.append(2)
    mu_of.append(MU_PAD)
    invm_of.append(1.0 / mfish)
    pad_of.append(0)
    pinned_of.append(0)

NP = len(positions)
rad_np = np.full(NP, PART_R, dtype=np.float32)
rad_np[FISH_BASE:] = -2.0 * CAGE_H      # cage nodes: a safety floor two cells DOWN (the
                                        # dilated cage sits up to a cell outside the skin);
                                        # the skin carries the real floor contact + friction
p0 = np.array(positions, dtype=np.float32)
# Truss rows as a flat pair list (one XPBD constraint each, solved per ROW and
# scattered -- the per-particle CSR gather is gone with the Jacobi gains), plus
# the mass-splitting count: how many structural rows (truss pairs or tets) touch
# each particle. Tonge et al. 2012: give every constraint a copy of the particle
# carrying 1/n of its mass (inverse mass n*w in the denominator), move the real
# particle by the plain w-weighted correction and SUM over its rows. The sum is
# then an average of per-row corrections -- unconditionally stable, no gain cap,
# and the XPBD multiplier stays consistent with what the particle actually moved.
_pairs_np = np.array([(a, b) for a, b, _ in pairs], dtype=np.int32)
_pairs_w = np.array([wgt for _, _, wgt in pairs], dtype=np.float64)
pair_rest_np = np.linalg.norm(p0[_pairs_np[:, 0]] - p0[_pairs_np[:, 1]], axis=1).astype(np.float32)
pair_alpha_np = (1.0 / (_pairs_w * K_FIN * DT * DT)).astype(np.float32)   # alpha~ = 1/(k dt^2)
nsplit_np = np.zeros(NP, dtype=np.float32)
np.add.at(nsplit_np, _pairs_np[:, 0], 1.0)
np.add.at(nsplit_np, _pairs_np[:, 1], 1.0)
np.add.at(nsplit_np, FISH_BASE + cage_t.reshape(-1), 1.0)
nsplit_np = np.maximum(nsplit_np, 1.0)
N_PAIRS = len(_pairs_np)

wp.init()
device = wp.get_preferred_device()
print(f"franka softgrip: {NP} particles ({FIN_TOTAL} finger + {CAGE_N} fish cage), "
      f"{N_PAIRS} truss rows + {N_TETS} tets, {SUBSTEPS} x {ITERATIONS} mass-split XPBD, on {device}")
print(f"  fish {FISH_LEN * 1000:.0f} x {FISH_H * 1000:.0f} x {FISH_W * 1000:.0f} mm, "
      f"{FISH_MASS * 1000:.0f} g: cage {CAGE_N} verts / {N_TETS} tets / {len(_e)} edges at "
      f"{CAGE_H * 1000:.1f} mm pitch, contact r {CAGE_R * 1000:.1f} mm, "
      f"{FISH_NV} skinned render verts (bind err {_bind_err.mean() * 1e6:.1f} um)")
print(f"  flesh E {YOUNG:.3g} Pa nu {POISSON:.2f}, strain-rate damping "
      f"{FLESH_VISC * 1000:.1f} ms dev"
      f"{f' / {FLESH_VISC_VOL * 1000:.1f} ms vol' if FLESH_VISC_VOL > 0.0 else ''}"
      f"{'  (elastic, = the old solver)' if FLESH_VISC <= 0.0 and FLESH_VISC_VOL <= 0.0 else ''}")

MASS_PAD = FIN_MASS / FIN_N
MASS_FISH = mfish


# --- warp kernels ---------------------------------------------------------------

@wp.kernel
def pin_follow(local: wp.array(dtype=wp.vec3),
               mats: wp.array(dtype=wp.mat44),
               owner: wp.array(dtype=int),
               pinned: wp.array(dtype=int),
               pos: wp.array(dtype=wp.vec3),
               prev: wp.array(dtype=wp.vec3)):
    # The root rows are the carriage: their world position is the finger's own
    # local layout carried by the live carriage transform. prev follows too, so
    # a moving carriage adds no spurious Verlet velocity to the pinned row --
    # the drag on the rest of the finger comes through the truss, as it should.
    i = wp.tid()
    if pinned[i] == 0:
        return
    m = mats[owner[i]]
    p = local[i]
    w = wp.vec3(m[0, 0] * p[0] + m[0, 1] * p[1] + m[0, 2] * p[2] + m[0, 3],
                m[1, 0] * p[0] + m[1, 1] * p[1] + m[1, 2] * p[2] + m[1, 3],
                m[2, 0] * p[0] + m[2, 1] * p[1] + m[2, 2] * p[2] + m[2, 3])
    d = w - pos[i]
    pos[i] = w
    prev[i] = prev[i] + d


@wp.kernel
def integrate_damped(x: wp.array(dtype=wp.vec3),
                     prev: wp.array(dtype=wp.vec3),
                     pred: wp.array(dtype=wp.vec3),
                     dt: float,
                     damp: wp.array(dtype=float),
                     gravity: wp.vec3):
    # Verlet predict with PER-PARTICLE damping. The truss nodes get several
    # times the fish's damping: a node in the finger sits on ~30 constraint rows
    # (skin, rib, two diagonals, root fan) and a Jacobi sweep over that many rows
    # overshoots, so the pads ring visibly at the truss's own frequency. Damping
    # the finger alone kills the ring without making the fish look like putty.
    i = wp.tid()
    p = x[i]
    v = (p - prev[i]) * (1.0 - damp[i])
    prev[i] = p
    pred[i] = p + v + gravity * dt * dt


@wp.kernel
def lambda_reset(lam: wp.array(dtype=float)):
    # XPBD multipliers are accumulated across one substep and reset before the
    # next -- this is what makes the compliance dt-independent (a real elastic
    # modulus) rather than an iteration-count-dependent PBD stiffness.
    t = wp.tid()
    lam[t] = 0.0


@wp.kernel
def pairs_xpbd(pos: wp.array(dtype=wp.vec3),
               pi: wp.array(dtype=int),
               pj: wp.array(dtype=int),
               rest: wp.array(dtype=float),
               alpha: wp.array(dtype=float),
               lam: wp.array(dtype=float),
               invm: wp.array(dtype=float),
               nsplit: wp.array(dtype=float),
               dpos: wp.array(dtype=wp.vec3)):
    # One XPBD distance row, mass-split Jacobi: the denominator sees each end as
    # n times lighter (its mass is shared by n rows), the move is the plain
    # w-weighted one and is SUMMED into dpos with every other row on that node.
    # C = |xj - xi| - rest, grad_i = -n, grad_j = +n.
    c = wp.tid()
    i = pi[c]
    j = pj[c]
    d = pos[j] - pos[i]
    l = wp.length(d)
    if l < 1.0e-9:
        return
    n = d / l
    wi = invm[i]
    wj = invm[j]
    a = alpha[c]
    den = wi * nsplit[i] + wj * nsplit[j] + a
    dl = (-(l - rest[c]) - a * lam[c]) / den
    lam[c] = lam[c] + dl
    if wi > 0.0:
        wp.atomic_add(dpos, i, n * (-wi * dl))
    if wj > 0.0:
        wp.atomic_add(dpos, j, n * (wj * dl))


@wp.kernel
def apply_delta(pos: wp.array(dtype=wp.vec3),
                dpos: wp.array(dtype=wp.vec3),
                omega: float):
    # The structural pass lands here: truss rows + FEM tets have scattered their
    # mass-split corrections, the node takes the sum (no division -- the split
    # already made it an average) and the buffer self-clears for the next sweep.
    i = wp.tid()
    pos[i] = pos[i] + dpos[i] * omega
    dpos[i] = wp.vec3(0.0, 0.0, 0.0)


@wp.kernel
def tet_corot(pos: wp.array(dtype=wp.vec3),
              prev: wp.array(dtype=wp.vec3),
              tets: wp.array2d(dtype=int),
              dm_inv: wp.array(dtype=wp.mat33),
              alpha_d: wp.array(dtype=float),
              alpha_h: wp.array(dtype=float),
              invm: wp.array(dtype=float),
              nsplit: wp.array(dtype=float),
              base: int,
              gamma_d: float,
              gamma_h: float,
              lam_d: wp.array(dtype=float),
              lam_h: wp.array(dtype=float),
              dpos: wp.array(dtype=wp.vec3)):
    # COROTATIONAL FEM as two XPBD rows per tet, both ZERO AT REST:
    #   deviatoric  C_D = |F - R|_F   (R = closest rotation to F), stiffness mu
    #   volumetric  C_H = det(F) - 1,                              stiffness lambda
    # i.e. the energy mu/2 |F - R|^2 + lambda/2 (J - 1)^2 -- linear corotational
    # elasticity with the right shear and bulk moduli, and the volume row kept
    # separate so the flesh stays near-incompressible.
    #
    # WHY NOT the stable Neo-Hookean constraint pair (phase 6)? Its two rows are
    # NONZERO at rest (C_D = sqrt 3, C_H = -mu/lambda) and only balance once each
    # multiplier has climbed to lambda* = -C/alpha~ -- from zero, ~den/alpha~
    # iterations, about 10 for the deviatoric row and ~250 for the hydrostatic
    # one at this mesh. Reset every substep and truncated at a few iterations,
    # that climb is permanently biased toward the deviatoric row's HARD target,
    # which is F = 0: measured, the fish sat at 34-58 % of its rest volume with
    # nothing touching it, and carrying the multipliers across substeps instead
    # re-applies the (huge, mutually cancelling) pre-stress as an explicit step
    # and blows up. Rows that are zero at rest carry only the actual load in
    # their multipliers, so a truncated solve is merely a little soft, never
    # biased, and the fixed point is F = I.
    #
    # F = Ds @ Dm_inv, Ds columns = [x1-x0, x2-x0, x3-x0]; for any scalar phi(F)
    # the gradient wrt x_k (k=1..3) is column k of (dphi/dF) @ Dm_inv^T and wrt
    # x0 minus their sum. dC_D/dF = (F - R)/C_D (the dR term vanishes), and
    # dC_H/dF = cof(F). R comes from the SVD with the sign of the smallest
    # singular direction flipped when det < 0, so an inverted tet is pulled back
    # through its flat state toward a proper rotation (inversion-safe).
    #
    # STRAIN-RATE DAMPING (--flesh-visc), Macklin et al.'s damped XPBD update:
    #   dlam = (-C - alpha~*lam - gamma*dot(gradC, x - x_prev))
    #          / ((1 + gamma) * sum_k s_k |gradC_k|^2 + alpha~)
    # `prev` is the position at the START of this substep (integrate_damped
    # writes it), so dot(gradC, x - prev) is C's change over the substep. With
    # gamma = tau/DT the row solves C + tau*Cdot = 0: Kelvin-Voigt flesh with
    # relaxation time tau. gamma_d = gamma_h = 0 is the old elastic solver.
    t = wp.tid()
    la, lb, lc, ld = tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3]
    ia, ib, ic, id_ = base + la, base + lb, base + lc, base + ld
    x0, x1, x2, x3 = pos[ia], pos[ib], pos[ic], pos[id_]
    u0 = x0 - prev[ia]        # this substep's displacement, per node: the
    u1 = x1 - prev[ib]        # strain RATE reference for the damping term
    u2 = x2 - prev[ic]
    u3 = x3 - prev[id_]
    w0, w1, w2, w3 = invm[ia], invm[ib], invm[ic], invm[id_]
    s0 = w0 * nsplit[ia]
    s1 = w1 * nsplit[ib]
    s2 = w2 * nsplit[ic]
    s3 = w3 * nsplit[id_]
    e1 = x1 - x0
    e2 = x2 - x0
    e3 = x3 - x0
    di = dm_inv[t]
    # deformation gradient columns
    f0 = e1 * di[0, 0] + e2 * di[1, 0] + e3 * di[2, 0]
    f1 = e1 * di[0, 1] + e2 * di[1, 1] + e3 * di[2, 1]
    f2 = e1 * di[0, 2] + e2 * di[1, 2] + e3 * di[2, 2]
    F = wp.mat33(f0[0], f1[0], f2[0],
                 f0[1], f1[1], f2[1],
                 f0[2], f1[2], f2[2])

    # closest rotation R = U V^T (det +1)
    U = wp.mat33()
    sig = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sig, V)
    R = U * wp.transpose(V)
    if wp.determinant(R) < 0.0:
        # flip the singular direction with the smallest sigma
        m = int(0)
        if wp.abs(sig[1]) < wp.abs(sig[m]):
            m = 1
        if wp.abs(sig[2]) < wp.abs(sig[m]):
            m = 2
        um = wp.vec3(U[0, m], U[1, m], U[2, m])
        vm = wp.vec3(V[0, m], V[1, m], V[2, m])
        R = R - wp.outer(um, vm) * 2.0

    da0 = wp.vec3(0.0, 0.0, 0.0)
    da1 = wp.vec3(0.0, 0.0, 0.0)
    da2 = wp.vec3(0.0, 0.0, 0.0)
    da3 = wp.vec3(0.0, 0.0, 0.0)

    # deviatoric row: C = |F - R|, stiffness mu (alpha_d)
    D = F - R
    cd = wp.sqrt(D[0, 0] * D[0, 0] + D[0, 1] * D[0, 1] + D[0, 2] * D[0, 2]
                 + D[1, 0] * D[1, 0] + D[1, 1] * D[1, 1] + D[1, 2] * D[1, 2]
                 + D[2, 0] * D[2, 0] + D[2, 1] * D[2, 1] + D[2, 2] * D[2, 2])
    ad = alpha_d[t]
    if cd > 1.0e-9:
        inv_cd = 1.0 / cd
        d0 = wp.vec3(D[0, 0], D[1, 0], D[2, 0])     # columns of D
        d1 = wp.vec3(D[0, 1], D[1, 1], D[2, 1])
        d2 = wp.vec3(D[0, 2], D[1, 2], D[2, 2])
        g1 = (d0 * di[0, 0] + d1 * di[0, 1] + d2 * di[0, 2]) * inv_cd
        g2 = (d0 * di[1, 0] + d1 * di[1, 1] + d2 * di[1, 2]) * inv_cd
        g3 = (d0 * di[2, 0] + d1 * di[2, 1] + d2 * di[2, 2]) * inv_cd
        g0 = -(g1 + g2 + g3)
        gd = (s0 * wp.dot(g0, g0) + s1 * wp.dot(g1, g1)
              + s2 * wp.dot(g2, g2) + s3 * wp.dot(g3, g3))
        rate = (wp.dot(g0, u0) + wp.dot(g1, u1)
                + wp.dot(g2, u2) + wp.dot(g3, u3))
        den = (1.0 + gamma_d) * gd + ad
        dl = (-cd - ad * lam_d[t] - gamma_d * rate) / den
        lam_d[t] = lam_d[t] + dl
        da0 = g0 * (w0 * dl)
        da1 = g1 * (w1 * dl)
        da2 = g2 * (w2 * dl)
        da3 = g3 * (w3 * dl)
    else:
        # at rest the row carries no load; the multiplier relaxes with it
        lam_d[t] = 0.0

    # volume row: C = det(F) - 1, stiffness lambda (alpha_h)
    c0 = wp.cross(f1, f2)
    c1 = wp.cross(f2, f0)
    c2 = wp.cross(f0, f1)
    detf = wp.dot(f0, c0)
    h1 = c0 * di[0, 0] + c1 * di[0, 1] + c2 * di[0, 2]
    h2 = c0 * di[1, 0] + c1 * di[1, 1] + c2 * di[1, 2]
    h3 = c0 * di[2, 0] + c1 * di[2, 1] + c2 * di[2, 2]
    h0 = -(h1 + h2 + h3)
    ch = detf - 1.0
    ah = alpha_h[t]
    gh = (s0 * wp.dot(h0, h0) + s1 * wp.dot(h1, h1)
          + s2 * wp.dot(h2, h2) + s3 * wp.dot(h3, h3))
    rateh = (wp.dot(h0, u0) + wp.dot(h1, u1)
             + wp.dot(h2, u2) + wp.dot(h3, u3))
    denh = (1.0 + gamma_h) * gh + ah
    dlh = (-ch - ah * lam_h[t] - gamma_h * rateh) / denh
    lam_h[t] = lam_h[t] + dlh
    da0 = da0 + h0 * (w0 * dlh)
    da1 = da1 + h1 * (w1 * dlh)
    da2 = da2 + h2 * (w2 * dlh)
    da3 = da3 + h3 * (w3 * dlh)

    wp.atomic_add(dpos, ia, da0)
    wp.atomic_add(dpos, ib, da1)
    wp.atomic_add(dpos, ic, da2)
    wp.atomic_add(dpos, id_, da3)


@wp.kernel
def skin_lattice(pos: wp.array(dtype=wp.vec3),
                 base: int,
                 ids: wp.array2d(dtype=int),
                 w: wp.array2d(dtype=float),
                 out: wp.array(dtype=wp.vec3)):
    """The render surface rides the cage: trilinear over its cell's 8 corners."""
    i = wp.tid()
    p = wp.vec3(0.0, 0.0, 0.0)
    for k in range(8):
        p += pos[base + ids[i, k]] * w[i, k]
    out[i] = p


@wp.func
def cell_floor(p: wp.vec3):
    """(floor height, friction) of the analytic cell under point p: table, the
    raised tray, or a crate's inner floor."""
    floor = TABLE_Y
    m = MU_STEEL
    if wp.abs(p[0] - TRAY_CX) < TRAY_HX and wp.abs(p[2] - TRAY_CZ) < TRAY_HZ:
        floor = TRAY_Y
    if (LEDGE_ON == 1 and p[0] > LEDGE_X0 and p[0] < LEDGE_X1 and wp.abs(p[2] - LEDGE_CZ) < LEDGE_HZ
            and (LEDGE_Y - p[1]) <= (LEDGE_X1 - p[0])):
        floor = LEDGE_Y          # on top; the +x end face is handled by the caller
    cz = CRATE_CZ
    if p[2] > 0.0:
        cz = CRATE2_CZ
    if wp.abs(p[0] - CRATE_CX) < CRATE_HX and wp.abs(p[2] - cz) < CRATE_HZ and p[1] < CRATE_RIM:
        floor = CRATE_FLOOR
        m = MU_CRATE
    return floor, m


@wp.func
def skin_point(pos: wp.array(dtype=wp.vec3), base: int,
               ids: wp.array2d(dtype=int), w: wp.array2d(dtype=float), v: int):
    q = wp.vec3(0.0, 0.0, 0.0)
    for k in range(8):
        q += pos[base + ids[v, k]] * w[v, k]
    return q


@wp.kernel
def surface_contacts_build(pos: wp.array(dtype=wp.vec3),
                           mesh: wp.uint64,
                           invm: wp.array(dtype=float),
                           body: wp.array(dtype=int),
                           tris: wp.array(dtype=int),
                           ids: wp.array2d(dtype=int),
                           margin: float,
                           mc_face: wp.array(dtype=int),
                           mc_uv: wp.array(dtype=wp.vec2),
                           mc_n: wp.array(dtype=wp.vec3),
                           mc_d0: wp.array(dtype=float),
                           mc_acc: wp.array(dtype=float),
                           csplit: wp.array(dtype=int)):
    # PHASE 11: the pads meet the fish's DRAWN SKIN -- a wp.Mesh of the skinned
    # body triangles, refit every substep -- not the tet-cage nodes as spheres.
    # One closest-point query per finger node latches the contact for the
    # substep: face + barycentrics (so the iterations can re-evaluate the skin
    # point from the live cage through the binding weights), the outward normal
    # and the penetration at prediction time (the friction budget's floor). The
    # cage nodes under the contact are counted for mass splitting.
    i = wp.tid()
    had = mc_face[i] >= 0          # was in contact last substep
    mc_face[i] = -1
    mc_acc[i] = 0.0
    if body[i] == 2:
        return
    if invm[i] == 0.0 and body[i] != 3:
        return                     # pinned finger roots do not collide; the palm does
    p = pos[i]
    q = wp.mesh_query_point_sign_normal(mesh, p, PART_R + CONTACT_DEEP)
    if not q.result:
        return
    cp = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
    d = p - cp
    dist = wp.length(d)
    if dist > PART_R + margin:
        # Beyond the shallow band, trust the "inside" verdict only for a node
        # that was ALREADY in contact last substep: real embedding is always
        # entered through the shallow band, while the body mesh's open fin
        # boundaries make the sign unreliable at range (phantom "embedded"
        # palm nodes hovering over the dorsal cut inflated the fish to 164 %).
        if q.sign > 0.0 or not had:
            return
    n = wp.mesh_eval_face_normal(mesh, q.face)
    if dist > 1.0e-6:
        n = d * (q.sign / dist)
    mc_face[i] = q.face
    mc_uv[i] = wp.vec2(q.u, q.v)
    mc_n[i] = n
    mc_d0[i] = PART_R - q.sign * dist
    va = tris[q.face * 3 + 0]
    vb = tris[q.face * 3 + 1]
    vc = tris[q.face * 3 + 2]
    for k in range(8):
        wp.atomic_add(csplit, ids[va, k], 1)
        wp.atomic_add(csplit, ids[vb, k], 1)
        wp.atomic_add(csplit, ids[vc, k], 1)


@wp.kernel
def surface_floor_count(pos: wp.array(dtype=wp.vec3),
                        base: int,
                        vids: wp.array(dtype=int),
                        ids: wp.array2d(dtype=int),
                        w: wp.array2d(dtype=float),
                        csplit: wp.array(dtype=int)):
    # body skin vertices near the cell floor count toward their cage nodes' split
    t = wp.tid()
    v = vids[t]
    q = skin_point(pos, base, ids, w, v)
    floor, m = cell_floor(q)
    if q[1] < floor + SURF_R + 0.003:
        for k in range(8):
            wp.atomic_add(csplit, ids[v, k], 1)


@wp.kernel
def surface_contacts_solve(pos: wp.array(dtype=wp.vec3),
                           prev: wp.array(dtype=wp.vec3),
                           base: int,
                           tris: wp.array(dtype=int),
                           ids: wp.array2d(dtype=int),
                           w: wp.array2d(dtype=float),
                           invm: wp.array(dtype=float),
                           mu: wp.array(dtype=float),
                           pad: wp.array(dtype=int),
                           body: wp.array(dtype=int),
                           csplit: wp.array(dtype=int),
                           mc_face: wp.array(dtype=int),
                           mc_uv: wp.array(dtype=wp.vec2),
                           mc_n: wp.array(dtype=wp.vec3),
                           mc_d0: wp.array(dtype=float),
                           mc_acc: wp.array(dtype=float),
                           dpos: wp.array(dtype=wp.vec3),
                           pad_force: wp.array(dtype=float),
                           pad_hits: wp.array(dtype=int)):
    # Pad sphere vs the live skin point of its latched contact. The skin point
    # is a weighted sum of up to 24 cage nodes (3 vertices x 8 corners), so the
    # XPBD row is C = dot(p - q, n) - PART_R >= 0 with gradient +n on the pad and
    # -W_k n on cage node k; the correction is mass-split over the pad's one
    # contact and each cage node's csplit. Coulomb friction on the same row:
    # cancel the tangential relative motion of pad vs skin point this substep,
    # up to mu times the normal correction the contact has carried (latched
    # penetration + what the iterations applied).
    i = wp.tid()
    f = mc_face[i]
    if f < 0:
        return
    p = pos[i]
    wi = invm[i]                   # 0 for the palm: the fish takes it all
    n = mc_n[i]
    uv = mc_uv[i]
    bu = uv[0]
    bv = uv[1]
    bw = 1.0 - bu - bv
    va = tris[f * 3 + 0]
    vb = tris[f * 3 + 1]
    vc = tris[f * 3 + 2]
    # live skin point, its start-of-substep position, and the split denominator
    q = wp.vec3(0.0, 0.0, 0.0)
    q0 = wp.vec3(0.0, 0.0, 0.0)
    den = wi
    for k in range(8):
        ka = base + ids[va, k]
        kb = base + ids[vb, k]
        kc = base + ids[vc, k]
        wa = bu * w[va, k]
        wb = bv * w[vb, k]
        wc = bw * w[vc, k]
        q += pos[ka] * wa + pos[kb] * wb + pos[kc] * wc
        q0 += prev[ka] * wa + prev[kb] * wb + prev[kc] * wc
        den += (invm[ka] * wa * wa * float(wp.max(csplit[ka - base], 1))
                + invm[kb] * wb * wb * float(wp.max(csplit[kb - base], 1))
                + invm[kc] * wc * wc * float(wp.max(csplit[kc - base], 1)))
    pen = wp.min(PART_R - wp.dot(p - q, n), CONTACT_HEAL)
    if pen <= 0.0:
        return
    if den <= 0.0:
        return
    dl = pen / den
    mc_acc[i] = mc_acc[i] + pen
    corr = n * dl
    # friction
    rel = (p - prev[i]) - (q - q0)
    t = rel - n * wp.dot(rel, n)
    tl = wp.length(t)
    if tl > 1.0e-9:
        budget = mu[i] * (wp.max(mc_d0[i], 0.0) + mc_acc[i])
        lim = wp.min(tl, budget)
        corr -= t * (lim / tl / den)
    wp.atomic_add(dpos, i, corr * wi)
    for k in range(8):
        ka = base + ids[va, k]
        kb = base + ids[vb, k]
        kc = base + ids[vc, k]
        wp.atomic_add(dpos, ka, corr * (-invm[ka] * bu * w[va, k]))
        wp.atomic_add(dpos, kb, corr * (-invm[kb] * bv * w[vb, k]))
        wp.atomic_add(dpos, kc, corr * (-invm[kc] * bw * w[vc, k]))
    if pad[i] == 1:
        wp.atomic_add(pad_force, body[i], dl)      # normal impulse, kg*m
        wp.atomic_add(pad_hits, 0, 1)


@wp.kernel
def surface_floor_solve(pos: wp.array(dtype=wp.vec3),
                        prev: wp.array(dtype=wp.vec3),
                        base: int,
                        vids: wp.array(dtype=int),
                        ids: wp.array2d(dtype=int),
                        w: wp.array2d(dtype=float),
                        invm: wp.array(dtype=float),
                        csplit: wp.array(dtype=int),
                        dpos: wp.array(dtype=wp.vec3)):
    # The body SKIN rests on the tray / table / crate floor (and stays inside
    # the crate walls) -- not the cage nodes, which sit a few mm outside the skin
    # and used to leave the drawn fish floating. Same skinned-point XPBD row as
    # the pad contact, gradient w_k on the 8 corners, Coulomb friction against
    # the surface with the per-substep normal correction as the budget.
    t = wp.tid()
    v = vids[t]
    q = skin_point(pos, base, ids, w, v)
    q0 = skin_point(prev, base, ids, w, v)
    floor, m = cell_floor(q)
    cz = CRATE_CZ
    if q[2] > 0.0:
        cz = CRATE2_CZ
    in_crate = (wp.abs(q[0] - CRATE_CX) < CRATE_HX and wp.abs(q[2] - cz) < CRATE_HZ
                and q[1] < CRATE_RIM)
    corr = wp.vec3(0.0, 0.0, 0.0)
    den = float(0.0)
    for k in range(8):
        kk = base + ids[v, k]
        den += invm[kk] * w[v, k] * w[v, k] * float(wp.max(csplit[ids[v, k]], 1))
    if den <= 0.0:
        return
    pen = floor + SURF_R - q[1]
    on_belt = LEDGE_ON == 1 and floor == LEDGE_Y
    if pen > 0.0:
        corr += wp.vec3(0.0, pen, 0.0)
        rel = q - q0
        if on_belt:
            rel = rel - wp.vec3(BELT_V * DT, 0.0, 0.0)   # the ledge top is a belt moving at BELT_V
        tt = wp.vec3(rel[0], 0.0, rel[2])
        tl = wp.length(tt)
        if tl > 1.0e-9:
            lim = wp.min(tl, m * pen * 2.0)
            corr -= tt * (lim / tl)
    if (LEDGE_ON == 1 and q[0] > LEDGE_X0 and q[0] < LEDGE_X1 and wp.abs(q[2] - LEDGE_CZ) < LEDGE_HZ
            and q[1] < LEDGE_Y and (LEDGE_X1 - q[0]) < (LEDGE_Y - q[1])):
        corr += wp.vec3(LEDGE_X1 + SURF_R - q[0], 0.0, 0.0)        # the ledge's end face
    if in_crate:
        # keep inside the walls
        x = wp.clamp(q[0], CRATE_CX - CRATE_HX + SURF_R, CRATE_CX + CRATE_HX - SURF_R)
        z = wp.clamp(q[2], cz - CRATE_HZ + SURF_R, cz + CRATE_HZ - SURF_R)
        corr += wp.vec3(x - q[0], 0.0, z - q[2])
    if wp.length(corr) <= 0.0:
        return
    dl = corr / den
    for k in range(8):
        kk = base + ids[v, k]
        wp.atomic_add(dpos, kk, dl * (invm[kk] * w[v, k]))


@wp.kernel
def statics(pos: wp.array(dtype=wp.vec3),
            prev: wp.array(dtype=wp.vec3),
            invm: wp.array(dtype=float),
            rad: wp.array(dtype=float),
            body: wp.array(dtype=int)):
    # Analytic cell: table plane, the raised tray floor, and the crate as an
    # open box (inner floor plus four walls). Tangential motion at a contact is
    # damped by the surface's friction, which is what stops a fish sliding off
    # a tray it was dropped onto.
    i = wp.tid()
    if invm[i] == 0.0:
        return
    p = pos[i]
    floor = TABLE_Y
    m = MU_STEEL
    on_tray = wp.abs(p[0] - TRAY_CX) < TRAY_HX and wp.abs(p[2] - TRAY_CZ) < TRAY_HZ
    if on_tray:
        floor = TRAY_Y
    if LEDGE_ON == 1 and p[0] > LEDGE_X0 and p[0] < LEDGE_X1 and wp.abs(p[2] - LEDGE_CZ) < LEDGE_HZ:
        if (LEDGE_Y - p[1]) <= (LEDGE_X1 - p[0]):
            floor = LEDGE_Y
        elif p[1] < LEDGE_Y:
            p = wp.vec3(LEDGE_X1, p[1], p[2])              # safety: out through the end face
    # Two crates, mirrored in z; pick the near one by sign and test that.
    cz = CRATE_CZ
    if p[2] > 0.0:
        cz = CRATE2_CZ
    in_crate = wp.abs(p[0] - CRATE_CX) < CRATE_HX and wp.abs(p[2] - cz) < CRATE_HZ
    if in_crate and p[1] < CRATE_RIM:
        floor = CRATE_FLOOR
        m = MU_CRATE
        # keep inside the walls
        x = wp.clamp(p[0], CRATE_CX - CRATE_HX + PART_R, CRATE_CX + CRATE_HX - PART_R)
        z = wp.clamp(p[2], cz - CRATE_HZ + PART_R, cz + CRATE_HZ - PART_R)
        p = wp.vec3(x, p[1], z)
    if p[1] < floor + rad[i]:
        p = wp.vec3(p[0], floor + rad[i], p[2])
        if body[i] != 2:                      # the fish's friction lives on its skin
            t = p - prev[i]
            p = p - wp.vec3(t[0], 0.0, t[2]) * wp.min(m, 1.0)
    pos[i] = p


@wp.kernel
def fish_centroid(pos: wp.array(dtype=wp.vec3), base: int, n: int,
                  out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    wp.atomic_add(out, 0, pos[base + i] / float(n))


# --- device state ---------------------------------------------------------------

x = wp.array(p0, dtype=wp.vec3, device=device)
_p0_prev = p0.copy()
if DROP:
    _p0_prev[FISH_BASE:, 0] -= BELT_V * DT          # already riding the belt
prev = wp.array(_p0_prev, dtype=wp.vec3, device=device)
pred = wp.zeros(NP, dtype=wp.vec3, device=device)
scratch = wp.zeros(NP, dtype=wp.vec3, device=device)
dpos = wp.zeros(NP, dtype=wp.vec3, device=device)      # structural-pass scatter buffer
nsplit_d = wp.array(nsplit_np, dtype=float, device=device)
pair_i_d = wp.array(np.ascontiguousarray(_pairs_np[:, 0]), dtype=int, device=device)
pair_j_d = wp.array(np.ascontiguousarray(_pairs_np[:, 1]), dtype=int, device=device)
pair_rest_d = wp.array(pair_rest_np, dtype=float, device=device)
pair_alpha_d = wp.array(pair_alpha_np, dtype=float, device=device)
pair_lam = wp.zeros(N_PAIRS, dtype=float, device=device)
nrm = wp.zeros(NP, dtype=wp.vec3, device=device)
centroid = wp.zeros(1, dtype=wp.vec3, device=device)
pad_force = wp.zeros(2, dtype=float, device=device)
pad_hits = wp.zeros(1, dtype=int, device=device)
tets_d = wp.array(cage_t, dtype=int, device=device)
tet_dm_inv_d = wp.array(tet_Dm_inv, dtype=wp.mat33, device=device)
tet_alpha_d_d = wp.array(tet_alpha_d, dtype=float, device=device)
tet_alpha_h_d = wp.array(tet_alpha_h, dtype=float, device=device)
tet_lam_d = wp.zeros(N_TETS, dtype=float, device=device)   # deviatoric multiplier
tet_lam_h = wp.zeros(N_TETS, dtype=float, device=device)   # hydrostatic multiplier
skin_ids_d = wp.array(skin_ids, dtype=int, device=device)
skin_w_d = wp.array(skin_w, dtype=float, device=device)

_damp_np = np.full(NP, 1.0 - math.exp(-DAMP_RATE * DT), dtype=np.float32)
_damp_np[:FIN_TOTAL] = 1.0 - math.exp(-DAMP_RATE_FIN * DT)
damp_arr = wp.array(_damp_np, dtype=float, device=device)
invm = wp.array(np.array(invm_of, dtype=np.float32), dtype=float, device=device)
mu_arr = wp.array(np.array(mu_of, dtype=np.float32), dtype=float, device=device)
rad_arr = wp.array(rad_np, dtype=float, device=device)
body_arr = wp.array(np.array(body_of, dtype=np.int32), dtype=int, device=device)
pad_arr = wp.array(np.array(pad_of, dtype=np.int32), dtype=int, device=device)
pinned_arr = wp.array(np.array(pinned_of, dtype=np.int32), dtype=int, device=device)

# The pin table: every finger particle's layout in ITS carriage frame, plus
# which carriage owns it. Fish rows are inert (pinned = 0).
local_np = np.zeros((NP, 3), dtype=np.float32)
owner_np = np.zeros(NP, dtype=np.int32)
for k in (0, 1):
    b = k * FIN_N
    local_np[b:b + FIN_N] = fin_local[k]
    owner_np[b:b + FIN_N] = k
local_np[PALM_BASE:PALM_BASE + PALM_N] = _palm_local
owner_np[PALM_BASE:PALM_BASE + PALM_N] = 2          # the tool frame
local_arr = wp.array(local_np, dtype=wp.vec3, device=device)
owner_arr = wp.array(owner_np, dtype=int, device=device)
mats = wp.zeros(3, dtype=wp.mat44, device=device)

# The fish's render surface is its own little vertex array now (skinned off the
# cage), so its normals are accumulated over its own triangles, not over the
# particle set.
fish_tris_np = render_faces.reshape(-1).astype(np.int32)
fish_tris_d = wp.array(fish_tris_np, dtype=int, device=device)
fish_v = wp.zeros(FISH_NV, dtype=wp.vec3, device=device)
fish_vn = wp.zeros(FISH_NV, dtype=wp.vec3, device=device)

@wp.kernel
def scatter_soup_safe(pos: wp.array(dtype=wp.vec3),
                      nrm: wp.array(dtype=wp.vec3),
                      tris: wp.array(dtype=int),
                      out_pos: wp.array(dtype=wp.vec3),
                      out_nrm: wp.array(dtype=wp.vec3)):
    """`scatter_soup`, but a vertex whose smooth normal has CANCELLED falls back
    to its own face's normal instead of normalising numerical noise.

    A sheet with no bending stiffness folds, and where it folds the area-weighted
    face normals meeting at a node point in opposite directions and sum to ~0.
    The shared kernel then normalises that near-zero vector, i.e. amplifies float
    noise into a random normal -- which is what the wrist POV showed as dark torn
    patches across the skins. The bending chords above stop most of the folding;
    this stops the shading from exploding on whatever fold is left.
    """
    k = wp.tid()
    i = tris[k]
    out_pos[k] = pos[i]
    b0 = k - (k % 3)
    a = pos[tris[b0]]
    b = pos[tris[b0 + 1]]
    c = pos[tris[b0 + 2]]
    fn = wp.cross(b - a, c - a)
    n = nrm[i]
    if wp.length(n) < 0.3 * wp.length(fn):
        n = fn
    out_nrm[k] = n / wp.max(wp.length(n), 1.0e-9)


fin_tris_np = np.concatenate([fin_faces[0].reshape(-1), fin_faces[1].reshape(-1)]).astype(np.int32)
fin_tris_d = wp.array(fin_tris_np, dtype=int, device=device)
fin_corners, fish_corners = len(fin_tris_np), len(fish_tris_np)
N_FIN_FACES = fin_corners // 3
N_FISH_FACES = fish_corners // 3
fin_pos = wp.zeros(fin_corners, dtype=wp.vec3, device=device)
fin_nrm = wp.zeros(fin_corners, dtype=wp.vec3, device=device)

# PHASE 10 render-only pad inset. Per drawn corner: the finger index (0/1) if the
# corner belongs to that finger's FRONT skin, else -1. The front skin is the first
# FIN_ROWS*FIN_COLS particles of each finger's FIN_N block. offset_front_skin then
# shifts only those corners inward (toward the other finger) after the scatter, so
# the amber contact face meets the fish without moving any physics particle.
_fin_corner_pid = fin_tris_np
_fin_corner_front = np.where(
    (_fin_corner_pid % FIN_N) < (FIN_ROWS * FIN_COLS),
    _fin_corner_pid // FIN_N, -1).astype(np.int32)
fin_front_of_d = wp.array(_fin_corner_front, dtype=int, device=device)


@wp.kernel
def offset_front_skin(finger_of: wp.array(dtype=int),
                      dir0: wp.vec3, dir1: wp.vec3, amt: float,
                      out_pos: wp.array(dtype=wp.vec3)):
    c = wp.tid()
    f = finger_of[c]
    if f == 0:
        out_pos[c] = out_pos[c] + dir0 * amt
    elif f == 1:
        out_pos[c] = out_pos[c] + dir1 * amt

# The ribs are their own soup with their own vertex normals: a rib plate stands
# perpendicular to the skins, so averaging its face normals into the skin nodes
# would band the skins and averaging the skins' into the ribs would make the
# ribs shade like the skin and vanish.
rib_tris_np = np.concatenate([rib_faces[0].reshape(-1), rib_faces[1].reshape(-1)]).astype(np.int32)
rib_tris_d = wp.array(rib_tris_np, dtype=int, device=device)
rib_corners = len(rib_tris_np)
N_RIB_FACES = rib_corners // 3
rib_vn = wp.zeros(NP, dtype=wp.vec3, device=device)
rib_pos = wp.zeros(rib_corners, dtype=wp.vec3, device=device)
rib_nrm = wp.zeros(rib_corners, dtype=wp.vec3, device=device)
# The rib soup's FRONT edge (the corners on front-skin particles) is inset by the
# same RENDER_PAD_INSET so the ribs stay welded to the shifted front skin; the
# rib's back edge (back-skin particles) is left where it is. Same front-index rule.
_rib_corner_front = np.where(
    (rib_tris_np % FIN_N) < (FIN_ROWS * FIN_COLS),
    rib_tris_np // FIN_N, -1).astype(np.int32)
rib_front_of_d = wp.array(_rib_corner_front, dtype=int, device=device)
fish_pos = wp.zeros(fish_corners, dtype=wp.vec3, device=device)
fish_nrm = wp.zeros(fish_corners, dtype=wp.vec3, device=device)

# Surface contact state: the body triangles as a wp.Mesh over the skinned
# vertex buffer (refit each substep), one latched contact slot per finger node,
# and the per-cage-node split counter.
body_tris_d = wp.array(body_tris.reshape(-1), dtype=int, device=device)
body_vids_d = wp.array(body_vids, dtype=int, device=device)
N_BODY_V = len(body_vids)
# Skin the rest shape into fish_v BEFORE building the BVH: a tree built on an
# all-zero buffer is degenerate (every box at the origin) and refit never fixes
# the topology, so every query walked every triangle (2 ms a substep).
wp.launch(skin_lattice, dim=FISH_NV, device=device,
          inputs=[x, FISH_BASE, skin_ids_d, skin_w_d, fish_v])
fish_cmesh = wp.Mesh(points=fish_v, indices=body_tris_d)
mc_face = wp.full(FIN_TOTAL, -1, dtype=int, device=device)
mc_uv = wp.zeros(FIN_TOTAL, dtype=wp.vec2, device=device)
mc_n = wp.zeros(FIN_TOTAL, dtype=wp.vec3, device=device)
mc_d0 = wp.zeros(FIN_TOTAL, dtype=float, device=device)
mc_acc = wp.zeros(FIN_TOTAL, dtype=float, device=device)
csplit = wp.zeros(CAGE_N, dtype=int, device=device)

def substep_body():
    """One substep. Device work ONLY -- no host reads, no branching on sim
    state -- so the whole thing captures into a CUDA graph."""
    wp.launch(pin_follow, dim=NP, device=device,
              inputs=[local_arr, mats, owner_arr, pinned_arr, x, prev])
    wp.launch(integrate_damped, dim=NP, device=device,
              inputs=[x, prev, pred, DT, damp_arr, GRAVITY])
    wp.launch(pin_follow, dim=NP, device=device,
              inputs=[local_arr, mats, owner_arr, pinned_arr, pred, prev])
    # Latch this substep's surface contacts: skin the body off the predicted
    # cage, refit the mesh, one closest-point query per finger node.
    wp.launch(skin_lattice, dim=FISH_NV, device=device,
              inputs=[pred, FISH_BASE, skin_ids_d, skin_w_d, fish_v])
    fish_cmesh.refit()
    csplit.zero_()
    wp.launch(surface_contacts_build, dim=FIN_TOTAL, device=device,
              inputs=[pred, fish_cmesh.id, invm, body_arr, body_tris_d, skin_ids_d, CONTACT_MARGIN,
                      mc_face, mc_uv, mc_n, mc_d0, mc_acc, csplit])
    wp.launch(surface_floor_count, dim=N_BODY_V, device=device,
              inputs=[pred, FISH_BASE, body_vids_d, skin_ids_d, skin_w_d, csplit])
    wp.launch(lambda_reset, dim=N_TETS, device=device, inputs=[tet_lam_d])
    wp.launch(lambda_reset, dim=N_TETS, device=device, inputs=[tet_lam_h])
    wp.launch(lambda_reset, dim=N_PAIRS, device=device, inputs=[pair_lam])
    pa = pred
    for _ in range(ITERATIONS):
        # Structural pass: every truss row and every tet scatters its mass-split
        # XPBD correction, the nodes take the sum.
        wp.launch(pairs_xpbd, dim=N_PAIRS, device=device,
                  inputs=[pa, pair_i_d, pair_j_d, pair_rest_d, pair_alpha_d, pair_lam,
                          invm, nsplit_d, dpos])
        wp.launch(tet_corot, dim=N_TETS, device=device,
                  inputs=[pa, prev, tets_d, tet_dm_inv_d, tet_alpha_d_d, tet_alpha_h_d,
                          invm, nsplit_d, FISH_BASE, GAMMA_D, GAMMA_H,
                          tet_lam_d, tet_lam_h, dpos])
        wp.launch(apply_delta, dim=NP, device=device, inputs=[pa, dpos, OMEGA])
        # Contact pass: pads vs the skin, the skin vs the cell floor -- both
        # scatter through the binding weights -- so the substep ends with the
        # bodies separated and the friction budget reflecting the real squeeze.
        for _c in range(CONTACT_SWEEPS):
            wp.launch(surface_contacts_solve, dim=FIN_TOTAL, device=device,
                      inputs=[pa, prev, FISH_BASE, body_tris_d, skin_ids_d, skin_w_d, invm, mu_arr,
                              pad_arr, body_arr, csplit, mc_face, mc_uv, mc_n, mc_d0, mc_acc,
                              dpos, pad_force, pad_hits])
            wp.launch(surface_floor_solve, dim=N_BODY_V, device=device,
                      inputs=[pa, prev, FISH_BASE, body_vids_d, skin_ids_d, skin_w_d, invm, csplit, dpos])
            wp.launch(apply_delta, dim=NP, device=device, inputs=[pa, dpos, 1.0])
    wp.launch(statics, dim=NP, device=device, inputs=[pa, prev, invm, rad_arr, body_arr])
    wp.copy(x, pa)


substep_graph = None


def capture_substep():
    global substep_graph
    if "--no-graph" in sys.argv or device.is_cpu:
        return
    try:
        wp.load_module(device=device)      # no JIT inside the capture
        with wp.ScopedCapture(device) as cap:
            substep_body()
        substep_graph = cap.graph
        print("  substep captured as a CUDA graph")
    except Exception as e:                 # noqa: BLE001
        substep_graph = None
        print(f"  note: CUDA graph capture failed ({e}); launching kernel by kernel")


# --- [D] task: waypoints, trajectories, force-controlled close ------------------

YAW = 0.0
# Graded by length: the small crate sits at -z, the large one at +z.
CRATE_NAME = "LARGE" if FISH_LEN * 1000.0 >= GRADE_MM else "SMALL"
TARGET_C = CRATE2_C if CRATE_NAME == "LARGE" else CRATE_C
TARGET_CZ = float(TARGET_C[1])
WP_HOME = (0.40, 0.42, 0.05)
WP_PRE = (GRASP_X, 0.26, FISH_P[2])
# TCP at the fish's mid-height, so the 90 x 60 mm paddles cover the flank instead
# of scraping the top of it. The pad tip reaches ~28 mm below the TCP, i.e. just
# under the belly, which is where the inward hook does its work.
# PHASE 11: the TCP goes to the CAGE's mid-height (the body the pads actually
# meet), not the procedural template's. The Barramundi cage is ~65 mm tall, the
# template 48: aiming at 24 mm drove the hand 11 mm into the fish's back.
CAGE_H_REST = float(cage_v[:, 1].max() - cage_v[:, 1].min())
WP_GRASP = (GRASP_X, TRAY_Y + CAGE_R + 0.5 * CAGE_H_REST, FISH_P[2])
WP_LIFT = (GRASP_X, 0.30, FISH_P[2])
WP_CARRY = (TARGET_C[0], 0.32, TARGET_CZ)
WP_LOWER = (TARGET_C[0], 0.17, TARGET_CZ)
# Retreat up AND sideways: a straight-up retreat rose through the still-falling
# fish and the tips re-snagged it out of the crate.
WP_OUT = (TARGET_C[0], 0.36, TARGET_CZ + (0.14 if TARGET_CZ < 0 else -0.14))

# (name, waypoint or None to hold, seconds)
# CARRY and LOWER are 30 % longer than phase 2b. The quintic's peak acceleration
# scales as 1/T^2, and the swing across to the crate was throwing the hanging
# fish out of the pads on the way in and out of the blend.
PLAN = [("DROP", None, 600.0)] if DROP else [
        ("SETTLE", None, 0.4),
        ("PREGRASP", WP_PRE, 1.1),
        ("DESCEND", WP_GRASP, 1.0),
        ("CLOSE", None, 2.3),
        ("LIFT", WP_LIFT, 2.6),
        ("CARRY", WP_CARRY, 2.1),
        ("LOWER", WP_LOWER, 1.5),
        ("OPEN", None, 1.5),
        ("RETREAT", WP_OUT, 1.0),
        ("DONE", None, 1.2)]

q_cur = list(Q_HOME)
q_cur[7] = q_cur[8] = FINGER_OPEN
q_seg_start = list(q_cur)
q_seg_end = list(q_cur)
seg_i, seg_t = 0, 0.0
finger_cmd = FINGER_OPEN
grip_force = 0.0
grip_held = False
_close_diag_pending = False    # print the pad/fish gap the frame the close completes
sim_t = 0.0
tcp_prev = None
slip_peak = 0.0
slip_mean, slip_n = 0.0, 0
report = {}
# Halved from phase 2. The ejection that lost the fish was DYNAMIC -- solver
# energy injected by the converging paddles, not a slide -- and a slower close
# gives the volume rows time to answer each bite before the next one arrives.
CLOSE_V = 0.006          # m/s per carriage -- the corotational fish is STIFF: 1 mm = tens of N
SQUEEZE = 0.0003         # extra bite once the threshold trips
# The release used CLOSE_V too, which at 9 mm/s needs 1.8 s to give back the
# 16 mm the close took -- longer than the OPEN phase, so on half the runs the
# fish was still in the hand at DONE. Letting go is not a delicate operation.
OPEN_V = 0.060           # m/s per carriage
GRIP_CONFIRM = 2         # consecutive frames over F_GRASP before the close stops
                         # (5 frames of debounce = 0.75 mm of extra travel, and a
                         # full-contact corotational fish answers that with 70 N)
grip_hi_n = 0
# Position backstop, and it is the primary control -- the force proxy only
# short-circuits it. Measured, not assumed: the carriage frame sits Y_CARRIAGE
# off the tool axis at FINGER_OPEN and tracks the dof one-for-one, and the pad's
# flat working face is FIN_SKIN_IN + PART_R inboard of that. So the dof at which
# the pad face reaches a given half-width is arithmetic, and it has to be,
# because the dof is NOT the half-gap (the carriage origin is offset).
Y_CARRIAGE = abs(float(CARRIAGE_TCP[0][1]))
# PHASE 9: a deeper bite. The conforming cage's shoulder is deep, so the pads
# catch many nodes but the contact force spreads thin across them -- the close
# held on the backstop at ~0.9 N and lifted at ~1.2 N (below the ~2.5 N target)
# on the shallow 3 mm bite. Driving the pad face ~6 mm inside the flank firms the
# clamp without over-stretching the FEM (p99 stays well under 1.4).
# PHASE 11: BITE is measured from the fish's CONTACT surface (cage node + CAGE_R),
# not from the cage node line -- the old 7 mm "bite" plus the 4.9 mm cage radius
# was a 12 mm commanded squeeze per side, which a fish at its real stiffness
# answers with >100 N. The force trip is the primary stop now; this backstop is
# a 3 mm indentation for when the force never reads (e.g. a soft --young).
BITE = cli_arg("--bite", 0.003, float)   # 8 mm was tried: the wrap pressed the
                                         # fish into the palm pocket and it left
                                         # the cell stuck in the hand at 134 N
FINGER_STOP = max(FINGER_MIN, FINGER_OPEN - Y_CARRIAGE + FIN_SKIN_IN + PART_R
                  + FISH_HALF_W + CAGE_R - BITE)


def begin_segment(i):
    global q_seg_start, q_seg_end, seg_t
    name, wpt, _ = PLAN[i]
    q_seg_start = list(q_cur)
    if wpt is None:
        q_seg_end = list(q_cur)
    else:
        q_seg_end, res = ik_to(tool_pose(*wpt, YAW), q_cur)
        if not res.converged:
            print(f"  [{name}] IK left {res.position_error * 1000:.1f} mm / "
                  f"{math.degrees(res.orientation_error):.1f} deg on the table")
    seg_t = 0.0


begin_segment(0)


def advance_task(dt):
    """Joint-space quintic between IK-solved waypoints; the fingers are a
    separate, force-controlled drive."""
    global seg_i, seg_t, q_cur, finger_cmd, grip_held, _close_diag_pending
    name = PLAN[seg_i][0]
    dur = PLAN[seg_i][2]
    seg_t += dt
    s = min(seg_t / dur, 1.0)
    s = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))      # quintic ease
    q = [a + (b - a) * s for a, b in zip(q_seg_start, q_seg_end)]

    if name == "CLOSE":
        # DEBOUNCE the force short-circuit. The hooked tips clip the fish for a
        # frame or two long before the flanks are loaded, and a single-frame trip
        # stopped the close 1-2 mm early -- which is exactly what lost the heavy
        # fish in phase 2b, at a carriage where the pads still push more than
        # they hold. Only N consecutive frames over the threshold count.
        global grip_hi_n
        # ARMING: during the descent the pads brush the flanks of a fish that
        # nearly fills the open hand, and that brushing alone read ~20 N -- the
        # force stop tripped after 1.6 mm of travel and the "grasp" was the
        # brush. A real controller tares at close start: the force trip only
        # arms once the carriages have actually closed 3 mm.
        armed = finger_cmd <= FINGER_OPEN - 0.003
        grip_hi_n = grip_hi_n + 1 if (armed and grip_force >= F_GRASP) else 0
        if not grip_held and grip_hi_n < GRIP_CONFIRM and finger_cmd > FINGER_STOP:
            finger_cmd = max(FINGER_STOP, finger_cmd - CLOSE_V * dt)
        elif not grip_held:
            grip_held = True
            _close_diag_pending = True
            why = "force" if grip_hi_n >= GRIP_CONFIRM else "backstop"
            finger_cmd = max(FINGER_MIN, finger_cmd - SQUEEZE)
            report["grip_N"] = grip_force
            report["grip_open_mm"] = finger_cmd * 1000.0
            print(f"  CLOSE: held on {why} at {grip_force:.2f} N, "
                  f"carriage {finger_cmd * 1000:.1f} mm "
                  f"(backstop {FINGER_STOP * 1000:.1f})")
    elif name == "OPEN":
        finger_cmd = min(FINGER_OPEN, finger_cmd + OPEN_V * dt)
    q[7] = q[8] = finger_cmd
    q_cur = q
    if seg_t >= dur and seg_i + 1 < len(PLAN):
        seg_i += 1
        begin_segment(seg_i)
    return name


def _tool_mat():
    return np.array(ik.tool_transform(q_cur).to_numpy(), dtype=np.float32).reshape(4, 4)


_mats_cur = np.stack([link_mat(left_link), link_mat(right_link), _tool_mat()]).astype(np.float32)
_mats_prev = _mats_cur.copy()
_mats_host = _mats_cur.copy()


def _push_mats(a):
    """Upload the carriage frames a fraction `a` of the way from last frame's to
    this frame's. The pins were being handed the new carriage matrix ONCE PER
    FRAME while six substeps ran on it, so the root rows teleported on substep 0
    and the truss rang for the other five -- that is the oscillation, and it is
    kinematic, not a solver instability. The graph replays whatever is in `mats`,
    so re-uploading between capture_launch calls costs one 128-byte copy."""
    np.multiply(_mats_cur - _mats_prev, a, out=_mats_host)
    np.add(_mats_host, _mats_prev, out=_mats_host)
    for k in (0, 1, 2):                    # re-orthonormalise the lerped basis
        r = _mats_host[k][:3, :3]
        c0 = r[:, 0] / (np.linalg.norm(r[:, 0]) + 1e-12)
        c1 = r[:, 1] - c0 * np.dot(c0, r[:, 1])
        c1 /= np.linalg.norm(c1) + 1e-12
        r[:, 0], r[:, 1], r[:, 2] = c0, c1, np.cross(c0, c1)
    mats.assign(_mats_host)


_diag_tet = cage_t.astype(np.int64)
_diag_dm_inv = tet_Dm_inv.astype(np.float64)


def fem_diag(cage_pos):
    """(max, p99 tet edge stretch, max |det(F) - 1|) over the fish cage -- the
    numbers that say whether the material is behaving. p99 << 1.4 with the max a
    little higher (a few boundary edges) means it droops without dissolving; the
    det deviation says it is not inverting."""
    ratio = np.linalg.norm(cage_pos[_e[:, 0]] - cage_pos[_e[:, 1]], axis=1) / np.maximum(_rest_edge_len, 1e-9)
    tv = cage_pos[_diag_tet].astype(np.float64)
    ds = np.stack([tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0], tv[:, 3] - tv[:, 0]], axis=2)
    detf = np.linalg.det(ds @ _diag_dm_inv)
    return float(ratio.max()), float(np.percentile(ratio, 99.0)), float(np.abs(detf - 1.0).max())


def _pad_inward_dirs():
    """Unit vectors, per finger, pointing toward the OTHER finger (the grasp
    closing axis in world), taken from the two live carriage frames. Finger 0's
    front skin is inset along dirs[0], finger 1's along dirs[1]."""
    c0 = _mats_cur[0][:3, 3]
    c1 = _mats_cur[1][:3, 3]
    d = c1 - c0
    n = float(np.linalg.norm(d))
    if n < 1.0e-6:
        return [(0.0, 1.0, 0.0), (0.0, -1.0, 0.0)]
    d = (d / n).astype(np.float64)
    return [tuple(d), tuple(-d)]


def pad_gap_diag(tag):
    """Print, for BOTH fingers, the pad inner-face (front-skin) centroid and the
    minimum surface gap from that pad to the nearest fish tet-cage vertex, plus
    the fish cage bbox+centroid and the render-mesh bbox. The surface gap is the
    centre-to-centre distance less the two contact radii, so <~3 mm means the pad
    is genuinely biting the flank rather than closing on a small hidden core with
    the visible fish ballooning around it. This is the acceptance measurement for
    the gripper-fish gap defect."""
    xp = x.numpy()
    cage = xp[FISH_BASE:FISH_BASE + CAGE_N]
    cmin, cmax, cc = cage.min(0), cage.max(0), cage.mean(0)
    rv = fish_v.numpy()
    rmin, rmax = rv.min(0), rv.max(0)
    contact_r = PART_R
    print(f"  [gap:{tag:5s}] cage bbox ({cmin[0]:.3f},{cmin[1]:.3f},{cmin[2]:.3f})"
          f"..({cmax[0]:.3f},{cmax[1]:.3f},{cmax[2]:.3f}) c=({cc[0]:.3f},{cc[1]:.3f},{cc[2]:.3f})"
          f" render bbox ({rmin[0]:.3f},{rmin[1]:.3f},{rmin[2]:.3f})..({rmax[0]:.3f},{rmax[1]:.3f},{rmax[2]:.3f})")
    dirs = _pad_inward_dirs()          # world inward per finger (grasp closing axis)
    for k in (0, 1):
        b = k * FIN_N
        pad = xp[b:b + FIN_ROWS * FIN_COLS]          # front skin = the inner face
        pc = pad.mean(0)
        d = np.sqrt(((pad[:, None, :] - rv[None, :, :]) ** 2).sum(-1))
        gap = float(d.min()) - contact_r
        # VISUAL gap -- the user's actual complaint. Compare the DRAWN front-skin
        # face (particle + the render-only inward inset) to the DRAWN fish surface
        # (fish_v render vertices). For each front-skin particle take the nearest
        # render vertex; report the closest one, signed along the inward axis so a
        # negative value means the drawn pad has crossed into the drawn fish.
        inward = np.array(dirs[k], dtype=np.float32)
        drawn = pad + inward * RENDER_PAD_INSET
        dr = np.sqrt(((drawn[:, None, :] - rv[None, :, :]) ** 2).sum(-1))
        near_j = dr.argmin(1)                         # nearest render vtx per pad pt
        near_d = dr[np.arange(len(pad)), near_j]
        p = int(near_d.argmin())                      # closest pad point
        nn = rv[near_j[p]]
        vis_signed = float(np.dot(nn - drawn[p], inward))   # + = gap, - = overlap
        print(f"  [gap:{tag:5s}] finger {k} pad-face c=({pc[0]:+.3f},{pc[1]:+.3f},{pc[2]:+.3f})"
              f" min surface gap {gap * 1000:+5.1f} mm | VISUAL pad->render "
              f"{vis_signed * 1000:+5.1f} mm (min {float(near_d.min()) * 1000:4.1f} mm)")


# --- stability probe (--diag) ---------------------------------------------------
# The two complaints this probe measures: (1) a SHAKY gripper = the free finger
# nodes moving relative to their own carriage while the hand holds still or
# carries; (2) a CRUSHED fish = the cross-section between the pads collapsing
# beyond the commanded bite, and the cage losing volume. Both are printed every
# DIAG_EVERY frames and summarised at the end of the run.
DIAG = "--diag" in sys.argv
DIAG_EVERY = cli_arg("--diag", 15, int)
_diag_state = {"prev_local": None, "jit_acc": 0.0, "jit_n": 0, "jit_peak": 0.0,
               "thick_min": 9.9, "vol_min": 9.9, "det_min": 9.9, "det_max": -9.9,
               "hold_jit_acc": 0.0, "hold_jit_n": 0}
_diag_free = np.array(pinned_of[:FIN_TOTAL]) == 0
_diag_sec = np.abs(cage_v[:, 0] - (GRASP_X - FISH_P[0])) < 0.015      # cage nodes at the grasp
_diag_rest_thick = float(cage_v[_diag_sec, 2].max() - cage_v[_diag_sec, 2].min())
_diag_rest_vol = float(tet_rest_vol.sum())


def _finger_local(xp):
    """Free finger nodes in their own carriage frame (arm motion removed)."""
    out = np.zeros((FIN_TOTAL, 3), np.float64)
    for k in (0, 1):
        b = k * FIN_N
        Minv = np.linalg.inv(_mats_cur[k].astype(np.float64))
        w = xp[b:b + FIN_N].astype(np.float64)
        out[b:b + FIN_N] = w @ Minv[:3, :3].T + Minv[:3, 3]
    return out


def stab_diag(phase, frame_i):
    st = _diag_state
    xp = x.numpy()
    loc = _finger_local(xp)
    jit = 0.0
    if st["prev_local"] is not None:
        sp = np.linalg.norm((loc - st["prev_local"])[_diag_free] * FPS, axis=1)
        jit = float(np.sqrt((sp ** 2).mean()))
        st["jit_acc"] += jit; st["jit_n"] += 1
        st["jit_peak"] = max(st["jit_peak"], jit)
        if phase in ("LIFT", "CARRY", "LOWER"):
            st["hold_jit_acc"] += jit; st["hold_jit_n"] += 1
    st["prev_local"] = loc
    _d0 = mc_d0.numpy()
    _emb = int((_d0[mc_face.numpy() >= 0] > PART_R + 0.004).sum())
    st["emb_max"] = max(st.get("emb_max", 0), _emb)
    cage = xp[FISH_BASE:FISH_BASE + CAGE_N].astype(np.float64)
    ax = np.array(_pad_inward_dirs()[0], np.float64)           # live closing axis
    pr = cage[_diag_sec] @ ax
    thick = float(pr.max() - pr.min()) / max(_diag_rest_thick, 1e-9)
    tv = cage[_diag_tet]
    ds = np.stack([tv[:, 1] - tv[:, 0], tv[:, 2] - tv[:, 0], tv[:, 3] - tv[:, 0]], axis=2)
    detf = np.linalg.det(ds @ _diag_dm_inv)
    vol = float(np.abs(detf * tet_rest_vol).sum()) / _diag_rest_vol
    if phase in ("CLOSE", "LIFT", "CARRY", "LOWER"):
        st["thick_min"] = min(st["thick_min"], thick)
        st["vol_min"] = min(st["vol_min"], vol)
    st["det_min"] = min(st["det_min"], float(detf.min()))
    st["det_max"] = max(st["det_max"], float(detf.max()))
    if "--lift-diag" in sys.argv and phase in ("DESCEND", "CLOSE", "LIFT") and frame_i % 5 == 0:
        tcp = np.array(ik.tool_transform(q_cur).to_numpy(), dtype=np.float32).reshape(4, 4)
        tinv = np.linalg.inv(tcp.astype(np.float64))
        cc = cage.mean(0) @ tinv[:3, :3].T + tinv[:3, 3]
        d0 = mc_d0.numpy()
        fc = mc_face.numpy()
        act = d0[fc >= 0]
        pen_i = np.nonzero((fc >= 0) & (d0 > 0))[0]
        rows = ((pen_i % FIN_N) % (FIN_ROWS * FIN_COLS)) // FIN_COLS
        fing = pen_i // FIN_N
        pts = xp[pen_i].astype(np.float64) @ tinv[:3, :3].T + tinv[:3, 3] if len(pen_i) else np.zeros((0, 3))
        ext = (f" | pen nodes rows {np.bincount(rows, minlength=FIN_ROWS).tolist()} fingers {np.bincount(fing, minlength=2).tolist()}"
               f" TCP z {pts[:, 2].min() * 1000:.0f}..{pts[:, 2].max() * 1000:.0f}" if len(pen_i) else "")
        print(f"  [lift] f{frame_i} {phase} tcp y {tcp[1, 3] * 1000:6.1f} | fish c in TCP frame "
              f"({cc[0] * 1000:+6.1f},{cc[1] * 1000:+6.1f},{cc[2] * 1000:+6.1f}) mm | pad contacts {len(act):3d} "
              f"penetrating {int((act > 0).sum()):3d} max {act.max() * 1000 if len(act) else 0:.2f} mm | grip {grip_force:5.2f} N" + ext)
    if frame_i % DIAG_EVERY == 0:
        cm = cage.mean(0)
        print(f"  [diag] f{frame_i:4d} {phase:8s} fish c=({cm[0]:.3f},{cm[1]:.3f},{cm[2]:.3f}) | finger jitter {jit:7.2f} mm/s | grasp "
              f"section {thick * 100:5.1f} % of rest | cage vol {vol * 100:5.1f} % | "
              f"detF {detf.min():5.2f}..{detf.max():5.2f} | embedded {_emb:3d} | grip {grip_force:5.2f} N "
              f"/ {grip_nodes_last[0]:5.1f} nodes | carriage {finger_cmd * 1000:4.1f} mm")


def stab_summary():
    st = _diag_state
    if not DIAG or st["jit_n"] == 0:
        return
    print(f"stability: finger jitter RMS {st['jit_acc'] / st['jit_n']:.2f} mm/s "
          f"(hold phases {st['hold_jit_acc'] / max(st['hold_jit_n'], 1):.2f}, peak {st['jit_peak']:.2f}) | "
          f"grasp section min {st['thick_min'] * 100:.1f} % of rest | cage volume min "
          f"{st['vol_min'] * 100:.1f} % | detF {st['det_min']:.2f}..{st['det_max']:.2f} | "
          f"embedded peak {st.get('emb_max', 0)}")


grip_nodes_last = [0.0]


def step_frame():
    """One 60 fps frame: task -> robot -> pins -> substeps -> surfaces."""
    global grip_force, sim_t, tcp_prev, slip_peak, slip_mean, slip_n, _close_diag_pending
    t0 = time.perf_counter()
    phase = advance_task(1.0 / FPS)
    set_q(q_cur)
    _mats_prev[:] = _mats_cur
    _mats_cur[0] = link_mat(left_link)
    _mats_cur[1] = link_mat(right_link)
    _mats_cur[2] = _tool_mat()
    t1 = time.perf_counter()

    pad_force.zero_()
    pad_hits.zero_()
    for _s in range(SUBSTEPS):
        _push_mats((_s + 1.0) / SUBSTEPS)
        if substep_graph is not None:
            wp.capture_launch(substep_graph)
        else:
            substep_body()
    centroid.zero_()
    wp.launch(fish_centroid, dim=CAGE_N, device=device,
              inputs=[x, FISH_BASE, CAGE_N, centroid])
    nrm.zero_()
    wp.launch(accum_normals, dim=N_FIN_FACES, device=device, inputs=[x, fin_tris_d, nrm])
    wp.launch(scatter_soup_safe, dim=fin_corners, device=device,
              inputs=[x, nrm, fin_tris_d, fin_pos, fin_nrm])
    if RENDER_PAD_INSET > 0.0:
        _din = _pad_inward_dirs()
        _d0, _d1 = wp.vec3(*_din[0]), wp.vec3(*_din[1])
        wp.launch(offset_front_skin, dim=fin_corners, device=device,
                  inputs=[fin_front_of_d, _d0, _d1, RENDER_PAD_INSET, fin_pos])
    rib_vn.zero_()
    wp.launch(accum_normals, dim=N_RIB_FACES, device=device, inputs=[x, rib_tris_d, rib_vn])
    wp.launch(scatter_soup_safe, dim=rib_corners, device=device,
              inputs=[x, rib_vn, rib_tris_d, rib_pos, rib_nrm])
    if RENDER_PAD_INSET > 0.0:
        wp.launch(offset_front_skin, dim=rib_corners, device=device,
                  inputs=[rib_front_of_d, _d0, _d1, RENDER_PAD_INSET, rib_pos])
    wp.launch(skin_lattice, dim=FISH_NV, device=device,
              inputs=[x, FISH_BASE, skin_ids_d, skin_w_d, fish_v])
    fish_vn.zero_()
    wp.launch(accum_normals, dim=N_FISH_FACES, device=device,
              inputs=[fish_v, fish_tris_d, fish_vn])
    wp.launch(scatter_soup, dim=fish_corners, device=device,
              inputs=[fish_v, fish_vn, fish_tris_d, fish_pos, fish_nrm])
    wp.launch(skin_lattice, dim=FIN_NV, device=device,
              inputs=[x, FISH_BASE, fin_ids_d, fin_w_d, finv])
    finvn.zero_()
    wp.launch(accum_normals, dim=FIN_NF, device=device, inputs=[finv, fin_ftris_d, finvn])
    wp.launch(scatter_soup, dim=len(fin_ftris_np), device=device,
              inputs=[finv, finvn, fin_ftris_d, finf_pos, finf_nrm])
    wp.synchronize_device(device)
    t2 = time.perf_counter()

    # pad_force holds the contact impulse on the pads (kg*m, summed over the
    # frame's substeps and iterations); divided by dt^2 it is the normal force
    # in newtons over one substep, averaged over the frame's substeps. This is
    # a real force now, both pads together.
    hits = max(int(pad_hits.numpy()[0]), 1)
    grip_force = float(pad_force.numpy().sum()) / (DT * DT) / SUBSTEPS
    grip_nodes = hits / float(SUBSTEPS * ITERATIONS)
    c = centroid.numpy()[0]
    tcp = np.array(ik.tool_transform(q_cur).to_numpy(), dtype=np.float32).reshape(4, 4)[:3, 3]
    if tcp_prev is not None and phase in ("LIFT", "CARRY"):
        v_fish = (c - step_frame.c_prev) * FPS
        v_tcp = (tcp - tcp_prev) * FPS
        slip = float(np.linalg.norm(v_fish - v_tcp))
        slip_peak = max(slip_peak, slip)
        slip_mean += slip
        slip_n += 1
    step_frame.c_prev = c
    tcp_prev = tcp
    if phase != step_frame.phase_prev:
        step_frame.phase_prev = phase
        fv = fish_v.numpy()
        ncon = int((mc_face.numpy() >= 0).sum())
        stretch, p99, detdev = fem_diag(x.numpy()[FISH_BASE:])
        print(f"  {phase:9s} t={sim_t:5.2f}s grip {grip_force:6.2f} N over "
              f"{grip_nodes:5.1f} nodes | fish y {fv[:, 1].min():.3f}..{fv[:, 1].max():.3f} "
              f"(h {(fv[:, 1].max() - fv[:, 1].min()) * 1000:4.1f} of {FISH_H_REST * 1000:.0f} mm) "
              f"c=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) touch {ncon} "
              f"| carriage {finger_cmd * 1000:.1f} mm "
              f"| FEM stretch p99 {p99:.2f} max {stretch:.2f} max|detF-1| {detdev:.2f}")
        if phase in ("LIFT", "CARRY"):
            pad_gap_diag(phase)
    if _close_diag_pending:
        pad_gap_diag("CLOSE")
        _close_diag_pending = False
    grip_nodes_last[0] = grip_nodes
    if DIAG:
        stab_diag(phase, step_frame.frame_i)
    step_frame.frame_i += 1
    sim_t += 1.0 / FPS
    fin_np, fin_nn = fin_pos.numpy(), fin_nrm.numpy()
    fish_np, fish_nn = fish_pos.numpy(), fish_nrm.numpy()
    t3 = time.perf_counter()

    fin_geo.update_attribute("position", fin_np)
    fin_geo.update_attribute("normal", fin_nn)
    rib_geo.update_attribute("position", rib_pos.numpy())
    rib_geo.update_attribute("normal", rib_nrm.numpy())
    fish_geo.update_attribute("position", fish_np)
    fish_geo.update_attribute("normal", fish_nn)
    fins_geo.update_attribute("position", finf_pos.numpy())
    fins_geo.update_attribute("normal", finf_nrm.numpy())
    if _fem_state["mode"] > 0:
        _x_now = x.numpy()
        _cage_now = _x_now[FISH_BASE:FISH_BASE + CAGE_N]
        _w, _g, _edges = fem_wires[_fem_state["mode"] - 1]
        _g.update_attribute("position", np.ascontiguousarray(_cage_now[_edges.reshape(-1)]))
        fin_wire_geo.update_attribute("position", np.ascontiguousarray(_x_now[_fin_pairs.reshape(-1)]))
    for _ey, _ci in zip(eyes, EYE_CORNER):
        _p = fish_np[_ci]
        _ey.position.set(float(_p[0]), float(_p[1]), float(_p[2]))
    aim_camera(tcp)
    return t1 - t0, t2 - t1, t3 - t2, phase, c


step_frame.c_prev = np.array(FISH_P, dtype=np.float32)
step_frame.phase_prev = ""
step_frame.frame_i = 0


def in_crate(c):
    return (abs(c[0] - TARGET_C[0]) < CRATE_HX and abs(c[2] - TARGET_CZ) < CRATE_HZ
            and c[1] < CRATE_RIM)


# --- [A] cell + [F] present -----------------------------------------------------

canvas = tp.Canvas("threepp x warp - Franka FR3 soft gripper", width=WIDTH, height=HEIGHT,
                   antialiasing=4, headless=HEADLESS, vsync=False)
renderer = tp.VulkanRenderer(canvas)


def _try(fn, *a):
    try:
        fn(*a)
        return True
    except Exception:                                # noqa: BLE001
        return False


def _set(name, value):
    try:
        setattr(renderer, name, value)
        return True
    except Exception:                                # noqa: BLE001
        return False


# Clean, bright, slightly cool. Phase 1/2 rendered a dark back half; this is the
# "bright cell" half of the brief.
# NOTE: the scene lights are not in physical units, so a physical camera at
# f/2.8 1/60 ISO 200 renders it essentially black (measured). Auto exposure for
# every rig; MACRO keeps only the depth of field, driven by focus_distance.
_set("auto_exposure", True)
# A real HDRI is far brighter than the procedural env (its windows are ~physical
# daylight), so the auto-exposure needs room to stop down or the frame blooms.
_ae_lo = 0.03 if (ENV_HDRI and os.path.exists(ENV_HDRI)) else 0.15
_try(renderer.set_auto_exposure_range, _ae_lo, 3.0)
# Depth of field is OFF by default even on MACRO (`--dof` forces it on). The
# renderer's circle of confusion is resolved at a lower rate than the frame, so
# every depth discontinuity -- the hand against the table, the crate rims --
# comes back as a hard STAIRCASE of halo blocks. Looked at it: a clean frame
# beats buggy bokeh, and the macro rig is close enough that perspective does
# most of the separating anyway.
if CAM == "macro" and DOF:
    _set("depth_of_field", True)
_try(renderer.set_white_balance, 6900.0, 0.0)
_set("bloom_strength", 0.12)
_set("bloom_threshold", 1.6)

scene = tp.Scene()
scene.background = 0x39434f


def studio_env(h=128, w=256):
    """A tiny HDR equirect: a soft overhead skylight box, a neutral horizon and
    a darker floor. Without one, `metalness 0.9` has nothing to reflect and
    stainless renders as flat grey paint -- which is exactly how phase 3's
    table read, and why its metalness had to be dialled back to 0.55.

    Row 0 is the NADIR as the renderer maps it (same convention as the water
    balloon's sky), so elevation runs -pi/2 .. +pi/2 up the rows.
    """
    v = (np.arange(h, dtype=np.float32) + 0.5) / h
    el = (v - 0.5) * math.pi
    up = np.clip(np.sin(el), 0.0, 1.0)
    down = np.clip(-np.sin(el), 0.0, 1.0)
    sky = np.float32([2.5, 2.7, 3.0])          # cool skylight overhead
    hor = np.float32([0.85, 0.88, 0.92])       # neutral walls
    flr = np.float32([0.15, 0.15, 0.16])       # dark floor
    col = (hor[None, :] * (1.0 - up[:, None] - down[:, None] * 0.9)
           + sky[None, :] * (up[:, None] ** 1.4)
           + flr[None, :] * down[:, None] * 0.9)
    img = np.repeat(col[:, None, :], w, axis=1)
    # One warm practical off to the +x side, so the steel has a highlight to
    # travel across as the arm moves rather than a uniform grey wash.
    u = (np.arange(w, dtype=np.float32) + 0.5) / w
    az = u * 2.0 * math.pi
    cosang = (np.sin(el)[:, None] * math.sin(math.radians(28.0))
              + np.cos(el)[:, None] * math.cos(math.radians(28.0))
              * np.cos(az[None, :] - math.radians(20.0)))
    img += (np.exp(-np.maximum(1.0 - cosang, 0.0) * 26.0).astype(np.float32)[:, :, None]
            * np.float32([1.6, 1.35, 1.0]))
    return np.ascontiguousarray(np.dstack([img, np.ones((h, w, 1), np.float32)]), np.float32)


env_tex = None
env_is_hdri = False
try:
    if ENV_HDRI and os.path.exists(ENV_HDRI):
        env_tex = tp.RGBELoader().load(ENV_HDRI)
        env_is_hdri = True
        scene.environment = env_tex
        scene.background = env_tex               # show the real room behind the cell
        print(f"  environment: {os.path.basename(ENV_HDRI)} (HDRI, IBL + backdrop)")
    else:
        env_tex = tp.float_texture(studio_env())
        scene.environment = env_tex
except Exception as _e:                              # noqa: BLE001
    print(f"  note: environment unavailable ({_e})")
    try:
        env_tex = tp.float_texture(studio_env())
        scene.environment = env_tex
        env_is_hdri = False
    except Exception:                               # noqa: BLE001
        pass

camera = tp.PerspectiveCamera(42 if CAM != "macro" else 34, canvas.aspect(), 0.03, 40)
CAM_HOME = {"wide": ((1.30, 0.72, 0.92), (0.30, 0.20, 0.0)),
            "drop": ((0.72, 0.24, 0.52), (0.52, 0.11, 0.10)),   # side view of the ledge end + floor
            "top": ((0.46, 1.02, 0.30), (0.40, 0.05, 0.0)),
            "macro": ((0.80, 0.40, 0.34), (0.45, 0.10, 0.10))}
_cp, _ct = CAM_HOME.get(CAM, CAM_HOME["wide"])
camera.position.set(*_cp)
camera.look_at(*_ct)
_cam_target = np.array(_ct, np.float32)
# MACRO rides the tool: 45 cm off the TCP on a fixed bearing, aimed at a
# smoothed TCP so the frame does not snap when the arm changes direction, with
# the focus distance driven from the same number.
MACRO_OFF = np.array([0.34, 0.17, 0.20], np.float32)
MACRO_OFF = MACRO_OFF / np.linalg.norm(MACRO_OFF) * 0.45


def aim_camera(tcp):
    global _cam_target
    if CAM != "macro":
        return
    _cam_target += (tcp - _cam_target) * 0.06        # smoothed follow
    eye = _cam_target + MACRO_OFF
    camera.position.set(float(eye[0]), float(eye[1]), float(eye[2]))
    camera.look_at(float(_cam_target[0]), float(_cam_target[1]), float(_cam_target[2]))
    try:
        renderer.focus_distance = float(np.linalg.norm(eye - _cam_target))
    except Exception:                                # noqa: BLE001
        pass

# RobotCell's base (cool skylight + one clean white sun) plus a warm key from
# front-left. The phase-1/2 frames were dark in the back half; this is the
# "clean bright cell" the brief asks for. When a real HDRI is lighting the scene
# it already supplies the fill and the reflections, so the hemisphere/key are
# pulled right down and only the sun stays for a crisp contact shadow -- feeding
# them at full strength on top of the HDRI just washes the frame out.
_hemi_i = 0.20 if env_is_hdri else 0.9
_key_i = 0.35 if env_is_hdri else 1.5
scene.add(tp.HemisphereLight(0xcfe0ff, 0x2a2e33, _hemi_i))
sun = tp.DirectionalLight(0xeaf2ff, 2.0)
sun.position.set(-0.8, 3.0, 1.0)
sun.cast_shadow = True
scene.add(sun)
key = tp.DirectionalLight(0xffe0c0, _key_i)
key.position.set(1.8, 1.4, 1.6)
scene.add(key)

# Brushed stainless, now that there IS an environment for it to reflect: the
# arm and the crates show up in the table as soft vertical smears.
table = tp.Mesh(tp.BoxGeometry(2.2, 0.06, 1.6), standard_material(0xb9bdc2, 0.28, 0.90))
table.position.set(0.35, -0.03, 0.0)
table.receive_shadow = True
scene.add(table)

if DROP:
    ledge = tp.Mesh(tp.BoxGeometry(LEDGE_X1 - LEDGE_X0, LEDGE_Y, 2 * LEDGE_HZ),
                    standard_material(0xc4c9ce, 0.22, 0.92))
    ledge.position.set(0.5 * (LEDGE_X0 + LEDGE_X1), LEDGE_Y * 0.5, LEDGE_CZ)
    ledge.receive_shadow = True
    ledge.cast_shadow = True
    scene.add(ledge)
tray = tp.Mesh(tp.BoxGeometry(2 * TRAY_HX, TRAY_Y, 2 * TRAY_HZ),
               standard_material(0xc4c9ce, 0.22, 0.92))
tray.position.set(TRAY_C[0], TRAY_Y * 0.5, TRAY_C[1])
tray.receive_shadow = True
scene.add(tray)

crate_mat = standard_material(0x2f6fae, 0.55, 0.0)
for cc in () if DROP else (CRATE_C, CRATE2_C):     # no crates in the drop bench (they block the view)
    crate_floor = tp.Mesh(tp.BoxGeometry(2 * CRATE_HX, CRATE_FLOOR, 2 * CRATE_HZ), crate_mat)
    crate_floor.position.set(cc[0], CRATE_FLOOR * 0.5, cc[1])
    scene.add(crate_floor)
    for dx, dz, w, d in ((CRATE_HX, 0.0, 0.012, 2 * CRATE_HZ),
                         (-CRATE_HX, 0.0, 0.012, 2 * CRATE_HZ),
                         (0.0, CRATE_HZ, 2 * CRATE_HX, 0.012),
                         (0.0, -CRATE_HZ, 2 * CRATE_HX, 0.012)):
        wall = tp.Mesh(tp.BoxGeometry(w, CRATE_RIM, d), crate_mat)
        wall.position.set(cc[0] + dx, CRATE_RIM * 0.5, cc[1] + dz)
        wall.cast_shadow = True
        scene.add(wall)

# The painted backdrop/side wall exist only to give the PROCEDURAL env something
# to stand in front of. With a real HDRI they do the opposite -- they box the
# scene in and hide the warehouse the environment is showing -- so skip them and
# let the HDRI be the backdrop.
if not env_is_hdri:
    backdrop = tp.Mesh(tp.PlaneGeometry(8, 4), standard_material(0x50606e, 0.95))
    backdrop.position.set(-0.6, 1.2, -1.6)
    scene.add(backdrop)
    side_wall = tp.Mesh(tp.PlaneGeometry(6, 4), standard_material(0x50606e, 0.95))
    side_wall.position.set(-1.6, 1.2, 0.0)
    side_wall.rotation.y = math.pi / 2
    scene.add(side_wall)

scene.add(robot)

def silicone(color, rough, clearcoat, translucency):
    """Cast urethane / TPU: a light dielectric with a wet clearcoat and some
    light coming through the thin sections. Double-sided, because these are
    single-thickness sheets and the two skins cross wherever the truss shears
    (single-sided culling flickered holes through the pads in phase 3)."""
    m = tp.MeshPhysicalMaterial()
    m.color = color
    m.roughness = rough
    m.metalness = 0.0
    m.clearcoat = clearcoat
    m.clearcoat_roughness = 0.15
    m.side = tp.Side.Double
    for _k, _v in (("translucency", translucency), ("translucency_color", 0xffe8c8)):
        try:
            setattr(m, _k, _v)
        except Exception:                            # noqa: BLE001
            pass
    return m


# The skins: amber silicone, thin sheets, open at the sides. Amber rather than
# the pale grey-blue because the hand behind them is cream-white and the fish
# in front of them is silver -- a pale finger disappeared into both.
fin_geo = tp.BufferGeometry()
fin_geo.set_attribute("position", p0[fin_tris_np])
fin_geo.set_attribute("normal", np.tile(np.array([0, 1, 0], np.float32), (fin_corners, 1)))
# Translucency 0.55 -> 0.35: at 0.55 a double-sided sheet lit from behind bleeds
# the far side's shading through, and from the wrist camera -- which looks along
# the finger, straight into the open side of the truss -- that read as dark
# patches on the near skin.
fingers_mesh = tp.Mesh(fin_geo, silicone(0xe8c890, 0.38, 0.35, 0.35))
fingers_mesh.cast_shadow = True
fingers_mesh.frustum_culled = False
scene.add(fingers_mesh)

# The ribs: the same material a shade darker and less glossy, so the struts
# read against the skin they join instead of merging into it.
rib_geo = tp.BufferGeometry()
rib_geo.set_attribute("position", p0[rib_tris_np])
rib_geo.set_attribute("normal", np.tile(np.array([0, 0, 1], np.float32), (rib_corners, 1)))
ribs_mesh = tp.Mesh(rib_geo, silicone(0xc79a5c, 0.5, 0.15, 0.35))
ribs_mesh.frustum_culled = False
scene.add(ribs_mesh)

def skin_texture(w=384, h=192):
    """Procedural herring: silver-white belly -> blue-green-grey back, a darker
    lateral line, fine scale noise, and a wash of pink at the gills. Rows are v
    (0 = belly), columns are u along the body with u = 0 at the head."""
    stops = np.array([0.00, 0.30, 0.46, 0.56, 0.60, 0.78, 1.00])
    cols = np.array([[248, 250, 252], [236, 241, 246], [186, 200, 208],
                     [120, 146, 152], [84, 108, 112], [66, 104, 100], [40, 66, 70]],
                    dtype=np.float64)
    v = np.linspace(0.0, 1.0, h)
    img = np.stack([np.interp(v, stops, cols[:, c]) for c in range(3)], -1)
    img = np.repeat(img[:, None, :], w, axis=1)
    uu = np.linspace(0.0, 1.0, w)[None, :, None]
    vv = v[:, None, None]
    # scales: a fine diamond lattice, brighter on the flank than on the back
    sc = (np.cos(uu * 210.0) * np.cos(vv * 96.0)) * 14.0 * (1.0 - np.abs(vv - 0.42) * 1.3)
    rng = np.random.default_rng(11)
    img = img + sc + rng.normal(0.0, 3.0, (h, w, 1))
    # lateral line, and the gill blush just behind the head
    img *= (1.0 - 0.16 * np.exp(-((vv - 0.545) / 0.018) ** 2))
    gill = np.exp(-((uu - 0.155) / 0.055) ** 2) * np.clip(1.0 - vv * 1.5, 0.0, 1.0)
    img = img + gill * np.array([70.0, -14.0, 4.0])
    return tp.data_texture(np.clip(img, 0, 255).astype(np.uint8), srgb=True)


# Corner-soup geometry, updated every frame from the cage skinning. For the
# procedural fish the UVs are linear in the unit icosphere (u along the body, v
# belly->back, no atan2 seam); for the glTF skin they come straight off the
# model. Normals are recomputed each frame, so the initial ones are placeholders.
fish_uv = render_uv
fish_geo = tp.BufferGeometry()
fish_geo.set_attribute("position", (render_v + np.array(FISH_P, np.float32))[fish_tris_np])
_n0 = (render_normals0[fish_tris_np] if render_normals0 is not None
       else np.tile(np.array([0, 1, 0], np.float32), (len(fish_tris_np), 1)))
fish_geo.set_attribute("normal", _n0.astype(np.float32))
fish_geo.set_attribute("uv", fish_uv[fish_tris_np])
if gltf_mat is not None:
    # The Barramundi's own PBR material and maps (base color, normal, AO,
    # roughness, metalness), with a wet clearcoat added for the fresh-fish sheen.
    fish_mat = gltf_mat
    for _k, _v in (("clearcoat", 0.6), ("clearcoat_roughness", 0.12)):
        try:
            setattr(fish_mat, _k, _v)
        except Exception:                            # noqa: BLE001
            pass
else:
    fish_mat = tp.MeshPhysicalMaterial()
    fish_mat.color = 0xffffff
    fish_mat.map = skin_texture()
    fish_mat.roughness = 0.5
    fish_mat.metalness = 0.0
    fish_mat.clearcoat = 0.85          # the wet coat is what sells a fresh fish
    fish_mat.clearcoat_roughness = 0.1
fish_mesh = tp.Mesh(fish_geo, fish_mat)
fish_mesh.cast_shadow = True
fish_mesh.frustum_culled = False
scene.add(fish_mesh)

# FEM debug view: the tet cage's 2430 unique edges as line segments riding the
# live cage nodes, plus a translucent skin so you can see the flesh through it.
# --fem-view starts with it on; F cycles skin+wire -> wire only -> normal.
# Two edge sets. Mode 1 draws the cage's BOUNDARY QUAD GRID -- the axis-aligned
# edges whose endpoints touch a non-solid cell -- which reads as the voxel
# surface ("squares") without the interior/diagonal clutter; mode 2 draws every
# tet edge with the skin hidden. Both are x-ray (no depth test): a depth-tested
# wire is clipped by the opaque skin wherever the skin stands proud of the cage
# and the lattice reads as "missing squares" (runtime opacity cannot re-sort a
# material into Vulkan's transparent pass, so the skin never turns translucent).
_nid = np.stack(np.nonzero(CAGE_ID >= 0), 1)                     # node -> (i,j,k)
_solidp = np.zeros(tuple(np.array(CAGE_SOLID.shape) + 2), bool)
_solidp[1:-1, 1:-1, 1:-1] = CAGE_SOLID
_bnode = ~np.stack([_solidp[_nid[:, 0] + di, _nid[:, 1] + dj, _nid[:, 2] + dk]
                    for di in (0, 1) for dj in (0, 1) for dk in (0, 1)]).all(0)
_rest_e = np.linalg.norm(cage_v[_e[:, 0]] - cage_v[_e[:, 1]], axis=1)
_e_surf = _e[(_rest_e < 1.05 * CAGE_H) & _bnode[_e[:, 0]] & _bnode[_e[:, 1]]]
fem_mat = tp.LineBasicMaterial()
fem_mat.color = tp.Color(0x27e58a)
fem_mat.depth_test = False
fem_wires = []
for _edges in (_e_surf, _e):
    g = tp.BufferGeometry()
    g.set_attribute("position", cage_p0[_edges.reshape(-1)].astype(np.float32))
    w = tp.LineSegments(g, fem_mat)
    w.frustum_culled = False
    w.visible = False
    scene.add(w)
    fem_wires.append((w, g, _edges))
print(f"  fem view: {len(_e_surf)} surface-grid edges / {len(_e)} tet edges")
# The GRIPPER's truss in the same view, another colour: the structural members
# only (skins + ribs) -- the fan/bend/diagonal chords are load paths, not shape,
# and drawing all 7k rows is soup. Positions ride the live finger particles.
_fin_pairs = _pairs_np[np.isin(_pairs_w, (STIFF_SKIN, STIFF_RIB))]
fin_wire_mat = tp.LineBasicMaterial()
fin_wire_mat.color = tp.Color(0xffa63c)
fin_wire_mat.depth_test = False
fin_wire_geo = tp.BufferGeometry()
fin_wire_geo.set_attribute("position", p0[_fin_pairs.reshape(-1)].astype(np.float32))
fin_wire = tp.LineSegments(fin_wire_geo, fin_wire_mat)
fin_wire.frustum_culled = False
fin_wire.visible = False
scene.add(fin_wire)
print(f"  fem view: {len(_fin_pairs)} finger truss edges (skins + ribs)")
_fem_state = {"mode": 0, "opacity": float(getattr(fish_mat, "opacity", 1.0)),
              "transparent": bool(getattr(fish_mat, "transparent", False))}


def set_fem_view(mode):
    """0 = normal render, 1 = translucent skin + tet wireframe, 2 = wireframe only."""
    _fem_state["mode"] = mode % 3
    m = _fem_state["mode"]
    fem_wires[0][0].visible = m == 1
    fem_wires[1][0].visible = m == 2
    fin_wire.visible = m > 0
    fish_mesh.visible = m < 2
    fingers_mesh.visible = m < 2
    ribs_mesh.visible = m < 2
    fish_mat.transparent = _fem_state["transparent"] or m == 1
    fish_mat.opacity = 0.35 if m == 1 else _fem_state["opacity"]
    try:
        fish_mat.needs_update = True
    except AttributeError:
        pass                     # pybind exposes it read-only on some materials
    print(f"  fem view: {('off', 'skin + cage', 'cage only')[m]}")


if FEM_VIEW:
    set_fem_view(1)


def build_fins():
    """Tail, dorsal and two pectorals as thin triangle fans, in the fish's own
    template frame. They ride the SAME tet cage as the body: bind_lattice's
    weights are unclamped, so a vertex outside the cage extrapolates from its
    nearest cell, and the fins flutter with the flesh they are attached to."""
    L, H, W = FISH_LEN, FISH_H_REST, FISH_W
    verts, faces = [], []

    def strip(root, tip):
        b = len(verts)
        verts.extend(root)
        verts.extend(tip)
        n = len(root)
        for i in range(n - 1):
            faces.append((b + i, b + i + 1, b + n + i + 1))
            faces.append((b + i, b + n + i + 1, b + n + i))

    # caudal fan: root across the tail stalk, tip swept back into a fork
    ts = np.linspace(-1.0, 1.0, 7)
    strip([(0.44 * L, t * 0.055 * H, 0.0) for t in ts],
          [(0.50 * L + abs(t) * 0.085 * L, t * 0.36 * H, 0.0) for t in ts])
    # dorsal
    xs = np.linspace(-0.06 * L, 0.14 * L, 6)
    strip([(x, 0.47 * H, 0.0) for x in xs],
          [(x + 0.02 * L, 0.47 * H + 0.20 * H * math.sin(math.pi * i / 5.0), 0.0)
           for i, x in enumerate(xs)])
    # pectorals, one per side, swept back and down
    for sgn in (-1.0, 1.0):
        ys = np.linspace(-0.02 * H, 0.16 * H, 4)
        strip([(-0.20 * L, y, sgn * 0.42 * W) for y in ys],
              [(-0.10 * L, y - 0.16 * H, sgn * 0.95 * W) for y in ys])
    return np.array(verts, np.float32), np.array(faces, np.int32)


fin_v_tmpl, fin_f_tmpl = build_fins()
FIN_NV, FIN_NF = len(fin_v_tmpl), len(fin_f_tmpl)
fin_ids, fin_w = bind_lattice(fin_v_tmpl.astype(np.float64), CAGE_LO, CAGE_H,
                              CAGE_DIMS, CAGE_SOLID, CAGE_ID)
fin_ids_d = wp.array(fin_ids, dtype=int, device=device)
fin_w_d = wp.array(fin_w, dtype=float, device=device)
finv = wp.zeros(FIN_NV, dtype=wp.vec3, device=device)
finvn = wp.zeros(FIN_NV, dtype=wp.vec3, device=device)
fin_ftris_np = fin_f_tmpl.reshape(-1)
fin_ftris_d = wp.array(fin_ftris_np, dtype=int, device=device)
finf_pos = wp.zeros(len(fin_ftris_np), dtype=wp.vec3, device=device)
finf_nrm = wp.zeros(len(fin_ftris_np), dtype=wp.vec3, device=device)

fins_geo = tp.BufferGeometry()
fins_geo.set_attribute("position", (fin_v_tmpl + np.array(FISH_P, np.float32))[fin_ftris_np])
fins_geo.set_attribute("normal", np.tile(np.array([0, 0, 1], np.float32),
                                         (len(fin_ftris_np), 1)))
fins_mat = tp.MeshPhysicalMaterial()
fins_mat.color = 0xc8d6dc
fins_mat.roughness = 0.35
fins_mat.metalness = 0.0
fins_mat.clearcoat = 0.6
fins_mat.side = tp.Side.Double
for _k, _v in (("translucency", 0.6), ("translucency_color", 0xdfe8ee)):
    try:
        setattr(fins_mat, _k, _v)
    except Exception:                      # noqa: BLE001
        pass
fins_mesh = tp.Mesh(fins_geo, fins_mat)
fins_mesh.frustum_culled = False
if not USE_GLTF:
    scene.add(fins_mesh)

# The eye is a plain little sphere parked on the head each frame. Its anchor is
# a render vertex, so it deforms with the head instead of floating. (The glTF
# fish has its own eye baked into the mesh + textures, so it gets neither.)
_eye_tmpl = np.array([(-0.34 * FISH_LEN, 0.16 * FISH_H_REST, sg * FISH_W * 0.30)
                      for sg in (-1.0, 1.0)], np.float32)
_corner_of = np.zeros(FISH_NV, np.int64)
_corner_of[fish_tris_np] = np.arange(len(fish_tris_np))
EYE_CORNER = [int(_corner_of[int(np.argmin(((render_v - e) ** 2).sum(1)))]) for e in _eye_tmpl]
eyes = []
if not USE_GLTF:
    for _ in range(2):
        e = tp.Mesh(tp.SphereGeometry(0.0075, 12, 10), standard_material(0x14181c, 0.15, 0.0))
        e.frustum_culled = False
        scene.add(e)
        eyes.append(e)

# --- [E] wrist camera picture-in-picture ---------------------------------------

# A secondary view is a real render of the scene from the hand, blitted into the
# primary frame by setViewDisplayRect at 1:1 -- no extra pass over the main
# image. The handle is only valid after the first render, so it is created lazily.
wrist_cam = tp.PerspectiveCamera(54, 320.0 / 180.0, 0.02, 4.0)
wrist_cam.position.set(0.0, 0.0, 0.035)      # just ahead of the hand flange
wrist_cam.rotation.y = math.pi               # a camera looks down -z; the tool is +z
_hand = robot.get_object_by_name("fr3_hand")
if _hand is not None:
    _hand.add(wrist_cam)
_pip = {"h": 0}


def ensure_pip():
    if not PIP or _pip["h"]:
        return
    try:
        h = renderer.add_view(wrist_cam, 320, 180)
        if h:
            renderer.set_view_display_rect(h, WIDTH - 330, 10, 320, 180)
            _pip["h"] = h
    except Exception as e:                   # noqa: BLE001
        print(f"  note: wrist PIP unavailable ({e})")
        _pip["h"] = -1


capture_substep()


def summary(c):
    ok = in_crate(c)
    mean = slip_mean / max(slip_n, 1)
    print(f"pick: fish {FISH_LEN * 1000:.0f} mm -> {CRATE_NAME} crate | "
          f"grip {report.get('grip_N', grip_force):.2f} N at "
          f"{report.get('grip_open_mm', finger_cmd * 1000):.1f} mm carriage | "
          f"slip mean {mean * 1000:.1f} mm/s, peak {slip_peak * 1000:.1f} mm/s | "
          f"fish at ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) -> "
          f"{'SUCCESS: IN THE CRATE' if ok else 'FAILED: not in the crate'}")
    stab_summary()
    return ok


def reset_scenario():
    """R in the viewer: rewind the whole scenario -- particles, multipliers,
    task plan, grip state -- to t = 0 without reloading (JIT + assets ~10 s).
    The prev buffer carries the belt speed again, so a --drop replays
    identically (up to solver non-determinism)."""
    global sim_t, seg_i, seg_t, q_cur, finger_cmd, grip_force, grip_held, grip_hi_n
    global tcp_prev, slip_peak, slip_mean, slip_n, _close_diag_pending
    x.assign(p0)
    prev.assign(_p0_prev)
    dpos.zero_()
    tet_lam_d.zero_()
    tet_lam_h.zero_()
    pair_lam.zero_()
    mc_acc.zero_()
    mc_face.fill_(-1)
    pad_force.zero_()
    pad_hits.zero_()
    sim_t = 0.0
    q_cur = list(Q_HOME)
    q_cur[7] = q_cur[8] = FINGER_OPEN if not DROP else Q_HOME[7]
    finger_cmd = FINGER_OPEN
    grip_force = 0.0
    grip_held = False
    grip_hi_n = 0
    tcp_prev = None
    slip_peak = 0.0
    slip_mean, slip_n = 0.0, 0
    _close_diag_pending = False
    set_q(q_cur)
    _mats_cur[0] = link_mat(left_link)
    _mats_cur[1] = link_mat(right_link)
    _mats_cur[2] = _tool_mat()
    _mats_prev[:] = _mats_cur
    _push_mats(1.0)
    seg_i, seg_t = 0, 0.0
    begin_segment(0)
    step_frame.c_prev = np.array(FISH_P, dtype=np.float32)
    step_frame.phase_prev = ""
    step_frame.frame_i = 0
    _diag_state["prev_local"] = None
    print("  -- replay --")


def set_rig(name):
    """Cut to a camera rig. MACRO rides the TCP (aim_camera), the others are
    fixed, so the rig name has to be the module-level CAM the follower reads."""
    global CAM, _cam_target
    CAM = name if name in CAM_HOME else "wide"
    p, t = CAM_HOME[CAM]
    camera.position.set(*p)
    camera.look_at(*t)
    camera.fov = 34 if CAM == "macro" else 42
    camera.update_projection_matrix()
    _cam_target = np.array(t, np.float32)
    _set("depth_of_field", CAM == "macro" and DOF)


def run_film():
    """The cut. Shots are keyed to the TASK's own phases rather than to
    absolute times, so retuning a phase duration re-times the film instead of
    breaking it; `rate` is sim seconds per film second, which is where the
    slow motion lives (the sim's dt is baked into the CUDA graph and cannot be
    shrunk, so slowing down means repeating film frames on one sim step)."""
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    from warp_common import load_font

    fps = 30 if PREVIEW else 60
    steps_per_frame = FPS / fps
    warm = 20                                  # AE/TAA settle at every cut
    names = [p[0] for p in PLAN]
    ends = np.cumsum([p[2] for p in PLAN])

    def t_end(nm):
        return float(ends[names.index(nm)])

    # (rig, last phase of the shot, sim seconds per film second)
    shots = [("wide", "DESCEND", 0.50),        # the arm wakes and comes to look
             ("macro", "CLOSE", 0.25),         # THE SHOT: the wrap, in slow motion
             ("macro", "LIFT", 0.60),          # off the tray, and the droop
             ("pov", "CARRY", 0.45),           # what the wrist sees on the way over
             ("wide", "RETREAT", 0.55),        # place, let go, come away
             ("macro", "DONE", 0.40)]
    out = os.path.abspath(FILM_OUT)
    hero = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "warp_franka_softgrip.png")
    writer = imageio.get_writer(out, fps=fps, codec="libx264", quality=None,
                                macro_block_size=None,
                                ffmpeg_params=["-crf", "17", "-pix_fmt", "yuv420p",
                                               "-preset", "medium"])
    print(f"film: {len(shots)} shots at {WIDTH}x{HEIGHT} {fps} fps -> {out}")
    t_wall = time.perf_counter()
    acc, nf, last_c = 0.0, 0, None
    px = None
    for rig, upto, rate in shots:
        pov = rig == "pov"
        set_rig("macro" if pov else rig)
        cam = wrist_cam if pov else camera
        if pov and _pip["h"] > 0:
            _try(renderer.hide_view, _pip["h"])
        elif _pip["h"] > 0:
            _try(renderer.set_view_display_rect, _pip["h"], WIDTH - 330, 10, 320, 180)
        stop = t_end(upto)
        for _ in range(warm):
            renderer.render(scene, cam)
            ensure_pip()
        guard = int(60.0 / max(rate, 0.05) * 40)
        while sim_t < stop and guard > 0:
            guard -= 1
            acc += rate * steps_per_frame
            while acc >= 1.0:
                acc -= 1.0
                parts = step_frame()
                last_c = parts[4]
            _set("sim_time", nf / float(fps))
            renderer.render(scene, cam)
            ensure_pip()
            px = renderer.read_pixels()
            writer.append_data(px)
            nf += 1
        if upto == "CLOSE" and px is not None:
            Image.fromarray(px).save(hero)     # the poster: the wrap, fully closed
            print(f"  hero -> {hero}")
    # End card over the last frame, dimmed.
    if px is not None:
        card = (px.astype(np.float32) * 0.30).astype(np.uint8)
        im = Image.fromarray(card)
        d = ImageDraw.Draw(im)
        big, small = load_font(int(HEIGHT * 0.052)), load_font(int(HEIGHT * 0.032))
        lines = [("Franka FR3 + Warp soft Fin Ray gripper", big),
                 ("every finger a GPU soft body", small),
                 ("friction grasp, no glue", small),
                 ("threepp", small)]
        y = HEIGHT * 0.32
        for txt, fnt in lines:
            w = d.textlength(txt, font=fnt)
            d.text((WIDTH * 0.5 - w * 0.5, y), txt, font=fnt, fill=(238, 242, 247))
            y += fnt.size * 1.6
        a = np.asarray(im)
        for _ in range(int(2.0 * fps)):
            writer.append_data(a)
            nf += 1
    writer.close()
    wall = time.perf_counter() - t_wall
    print(f"film: {nf} frames = {nf / fps:.1f} s at {fps} fps, "
          f"{wall:.1f} s wall ({wall / max(nf, 1) * 1000:.1f} ms/frame) -> {out}")
    if last_c is not None:
        summary(last_c)


if "--fin-rest" in sys.argv:
    # PHASE 9 acceptance (a): with the hand OPEN and stationary in mid-air, the
    # fingertip velocity RMS must be ~0 -- the truss must not jitter or ring. Hold
    # the reference open pose, let the truss settle under gravity, then measure the
    # RMS speed of the free (non-pinned) finger nodes over N frames.
    nrest = cli_arg("--fin-rest", 150, int)
    set_q(q_ref)
    _mats_cur[0] = link_mat(left_link)
    _mats_cur[1] = link_mat(right_link)
    _mats_cur[2] = _tool_mat()
    _mats_prev[:] = _mats_cur

    def _tick():
        for _s in range(SUBSTEPS):
            _push_mats(1.0)
            if substep_graph is not None:
                wp.capture_launch(substep_graph)
            else:
                substep_body()

    for _ in range(90):                        # settle the truss, hand held open
        _tick()
    free = np.array(pinned_of[:FIN_TOTAL]) == 0
    prevpos = x.numpy()[:FIN_TOTAL].copy()
    acc, cnt = 0.0, 0
    for _ in range(nrest):
        _tick()
        cur = x.numpy()[:FIN_TOTAL]
        sp = np.linalg.norm((cur - prevpos)[free] * FPS, axis=1)
        acc += float((sp ** 2).mean())
        cnt += 1
        prevpos = cur.copy()
    print(f"fin-rest: fingertip velocity RMS at rest = {(acc / max(cnt, 1)) ** 0.5 * 1000:.3f} "
          f"mm/s over {nrest} frames (hand open, stationary)")
    sys.exit(0)
elif FILM:
    if PREVIEW:
        PIP = False
    run_film()
elif SHOT:
    frames = int(round(SHOT_TIME * FPS))
    last = None
    for _ in range(frames):
        last = step_frame()[4]
    for _ in range(40):                    # let AE / TAA settle
        renderer.render(scene, wrist_cam if POV else camera)
        ensure_pip()
    from PIL import Image
    out = cli_arg("--out", "warp_franka_softgrip.png", str)
    Image.fromarray(renderer.read_pixels()).save(out)
    print(f"t = {SHOT_TIME:.2f} s, phase {PLAN[seg_i][0]}, wrote {out}")
    summary(last)
elif BENCH:
    acc = [0.0, 0.0, 0.0, 0.0]
    last = None
    for n in range(FRAMES):
        parts = step_frame()
        last = parts[4]
        t = time.perf_counter()
        renderer.render(scene, wrist_cam if POV else camera)
        ensure_pip()
        r = time.perf_counter() - t
        if n >= 30:
            for i in range(3):
                acc[i] += parts[i]
            acc[3] += r
    n = max(FRAMES - 30, 1)
    ms = 1000.0 / n
    total = sum(acc) * ms
    print(f"bench: robot+ik {acc[0] * ms:.2f} | sim {acc[1] * ms:.2f} | "
          f"readback {acc[2] * ms:.2f} | render {acc[3] * ms:.2f} = "
          f"{total:.2f} ms/frame ({1000.0 / total:.0f} fps)")
    summary(last)
else:
    # Slow motion around the wrap: 1x -> 0.15x -> 1x. The sim advances on an
    # accumulator rather than by shrinking dt, because dt is baked into the
    # captured CUDA graph.
    slow = {"acc": 0.0}

    def rate():
        if not SLOWMO:
            return 1.0
        nm = PLAN[seg_i][0]
        if nm == "DROP":
            return 1.0        # the imgui speed slider owns the pace in --drop
        if nm == "CLOSE":
            return 0.15
        if nm == "LIFT":
            return min(1.0, 0.15 + seg_t / 1.2 * 0.85)
        return 1.0

    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target.set(*(CAM_HOME["drop"][1] if DROP else (0.44, 0.10, 0.02)))

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        renderer.set_size(w, h)

    canvas.on_window_resize(on_resize)
    state = {"done": False, "phase": "", "r_down": False}

    # --- imgui control panel: eyeball speed, debug views, live physics ---------
    ui = tp.ImguiContext(canvas, renderer) if tp.HAS_IMGUI else None
    UIS = {"speed": 0.35 if DROP else 1.0, "young": YOUNG, "fin_k": K_FIN,
           "mu": MU_PAD, "damp": DAMP_RATE, "damp_fin": DAMP_RATE_FIN,
           "grip": F_GRASP, "close_v": CLOSE_V * 1000.0, "bite": BITE * 1000.0}

    def _apply_young(e):
        # alpha ~ 1/stiffness: rescale the precomputed compliance arrays on the
        # device -- the captured graph reads the arrays, so this is live.
        mu_s = _YOUNG0 / max(e, 1.0)
        tet_alpha_d_d.assign((tet_alpha_d * mu_s).astype(np.float32))
        tet_alpha_h_d.assign((tet_alpha_h * mu_s).astype(np.float32))

    def _apply_fin_k(k):
        pair_alpha_d.assign((pair_alpha_np * (_KFIN0 / max(k, 1.0))).astype(np.float32))

    def _apply_mu(m):
        mu_arr.assign(np.full(NP, m, dtype=np.float32))

    def _apply_damp(fish_rate, fin_rate):
        d = np.full(NP, 1.0 - math.exp(-fish_rate * DT), dtype=np.float32)
        d[:FIN_TOTAL] = 1.0 - math.exp(-fin_rate * DT)
        damp_arr.assign(d)

    def draw_ui():
        global F_GRASP, CLOSE_V, BITE, FINGER_STOP
        tp.imgui.set_next_window_pos(12, 12)
        tp.imgui.set_next_window_size(330, 0)
        tp.imgui.begin("softgrip")
        tp.imgui.text(f"{state['phase'] or PLAN[seg_i][0]}  t={sim_t:5.2f}s  "
                      f"grip {grip_force:5.1f} N  {tp.imgui.get_framerate():.0f} fps")
        ch, UIS["speed"] = tp.imgui.slider_float("speed", UIS["speed"], 0.02, 1.0)
        if tp.imgui.button("replay (R)"):
            reset_scenario()
            state["done"] = False
            slow["acc"] = 0.0
        tp.imgui.same_line()
        ch, m = tp.imgui.combo("FEM view", _fem_state["mode"], ["off", "skin+cage", "cage only"])
        if ch:
            set_fem_view(m)
        if tp.imgui.collapsing_header("physics (live)"):
            ch, UIS["young"] = tp.imgui.slider_float("young E (Pa)", UIS["young"], 2.0e4, 6.0e5)
            if ch:
                _apply_young(UIS["young"])
            ch, UIS["fin_k"] = tp.imgui.slider_float("finger k (N/m)", UIS["fin_k"], 200.0, 6000.0)
            if ch:
                _apply_fin_k(UIS["fin_k"])
            ch, UIS["mu"] = tp.imgui.slider_float("pad friction", UIS["mu"], 0.1, 1.5)
            if ch:
                _apply_mu(UIS["mu"])
            ch1, UIS["damp"] = tp.imgui.slider_float("fish damping 1/s", UIS["damp"], 0.0, 40.0)
            ch2, UIS["damp_fin"] = tp.imgui.slider_float("finger damping 1/s", UIS["damp_fin"], 0.0, 80.0)
            if ch1 or ch2:
                _apply_damp(UIS["damp"], UIS["damp_fin"])
        if tp.imgui.collapsing_header("grasp"):
            ch, UIS["grip"] = tp.imgui.slider_float("grip force N", UIS["grip"], 2.0, 80.0)
            if ch:
                F_GRASP = UIS["grip"]
            ch, UIS["close_v"] = tp.imgui.slider_float("close mm/s", UIS["close_v"], 2.0, 15.0)
            if ch:
                CLOSE_V = UIS["close_v"] / 1000.0
            ch, UIS["bite"] = tp.imgui.slider_float("bite mm", UIS["bite"], 0.0, 10.0)
            if ch:
                BITE = UIS["bite"] / 1000.0
                FINGER_STOP = max(FINGER_MIN, FINGER_OPEN - Y_CARRIAGE + FIN_SKIN_IN + PART_R
                                  + FISH_HALF_W + CAGE_R - BITE)
        tp.imgui.end()

    def animate():
        typing_ui = bool(ui and ui.want_capture_keyboard)
        r = (not typing_ui) and canvas.is_key_down("R")
        if r and not state["r_down"]:
            reset_scenario()
            state["done"] = False
            slow["acc"] = 0.0
        state["r_down"] = r
        f = (not typing_ui) and canvas.is_key_down("F")
        if f and not state.get("f_down"):
            set_fem_view(_fem_state["mode"] + 1)
        state["f_down"] = f
        slow["acc"] += rate() * UIS["speed"]
        while slow["acc"] >= 1.0:
            slow["acc"] -= 1.0
            parts = step_frame()
            state["phase"] = parts[3]
            if parts[3] == "DONE" and not state["done"]:
                state["done"] = True
                summary(parts[4])
        # OrbitControls reacts to canvas mouse events directly, so gating
        # update() is not enough: flip its enabled flag while ImGui owns the
        # pointer, or dragging a slider also spins the camera.
        controls.enabled = not (ui and ui.want_capture_mouse)
        if CAM != "macro" and controls.enabled:
            controls.update()
        renderer.render(scene, wrist_cam if POV else camera)
        ensure_pip()
        if ui:
            ui.render(draw_ui)

    canvas.animate(animate)
