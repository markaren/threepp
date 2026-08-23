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
    python warp_franka_softgrip.py --young 2.5e5  # fish flesh Young's modulus, Pa
    python warp_franka_softgrip.py --poisson 0.45 # near-incompressible tet FEM
    python warp_franka_softgrip.py --grade-mm 265 # length that picks the far crate
    python warp_franka_softgrip.py --cam macro    # wide | macro (follows the TCP) | top
    python warp_franka_softgrip.py --pov          # render from the wrist camera
    python warp_franka_softgrip.py --dof          # depth of field (see the note below)
    python warp_franka_softgrip.py --no-slowmo    # no 0.15x window around CLOSE -> LIFT
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
from warp_common import (accum_normals, cli_arg, csr_from_pairs, icosphere,
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
F_GRASP = cli_arg("--grip-force", 6.0, float)
HEADLESS = SHOT or BENCH
CAM = cli_arg("--cam", "wide", str)          # wide | macro | top
# Render skin: default is the self-contained procedural herring so the committed
# example runs on any machine; pass --fish-model to skin a glTF fish (the
# Khronos Barramundi) off the same tet cage. The mesh is NOT vendored.
FISH_MODEL = cli_arg("--fish-model", "", str)
SLOWMO = "--no-slowmo" not in sys.argv
PIP = "--no-pip" not in sys.argv
DOF = "--dof" in sys.argv                    # see the note at the renderer setup
POV = "--pov" in sys.argv                    # render from the wrist camera
FILM = "--film" in sys.argv
FILM_OUT = cli_arg("--film", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "warp_franka_softgrip.mp4"), str)
PREVIEW = "--preview" in sys.argv
if FILM:
    HEADLESS = True
    if PREVIEW:
        WIDTH, HEIGHT = 640, 360

# --- solver tunables ------------------------------------------------------------

FPS = 60.0
DT = 1.0 / 240.0
SUBSTEPS = 6
ITERATIONS = 26
DAMPING = 0.04
DAMPING_FIN = 0.22       # the truss rings otherwise -- see integrate_damped
GRAVITY = wp.vec3(0.0, -9.81, 0.0)
RELAX = 0.5

# Fin Ray split: the skins are structural, the diagonals are the compliance.
STIFF_SKIN = 1.0
STIFF_RIB = 0.7
STIFF_DIAG = 0.35
# BENDING. A distance-only sheet has no bending stiffness at all: every fold is
# free, so under the clamp the skins buckled into crumpled amber foil (phase 4's
# poster shows the lower half of each finger reading as a crushed paper bag) and
# the buckles folded triangles back on themselves, which is where the wrist-POV
# "holes" came from. A real Fin Ray skin is a stiff plastic strip -- it bends in
# a smooth arc. Skip-a-node chords give the sheet exactly that: they resist
# CURVATURE without adding any in-plane stretch resistance, so the truss's shear
# compliance -- the Fin Ray effect itself -- is untouched. Same trick as the
# fish's spine chords (phase 3), applied in both directions of both skins.
STIFF_BEND = 0.9
STIFF_BEND2 = 0.0     # skip-TWO chords along the columns, if skip-one is not
                      # enough. Off: skip-one plus the gain cap below is enough,
                      # and every row costs budget.
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
FIN_GAIN_CAP = 5.5    # in units of 0.5 * sum(stiffness) at the node
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
NH_GAMMA = 1.0 + MU_LAME / LAM_LAME            # rest det(F) the hydro row targets
TET_RES = 22          # voxel cells along the fish's length
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
PART_R = 0.005        # pad / finger nodes
CAGE_R_CELLS = 0.45   # fish cage nodes, in cells of the voxel pitch
CONTACT_CAP = 10
CONTACT_MARGIN = 1.4

MU_PAD = 1.10         # tacky silicone pad on wet fish -- high, but elastomers do this
MU_STEEL = 0.25       # fish on the stainless tray / table
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
FIN_CURL = 0.010
FIN_SKIN_OUT = 0.010              # back skin, outboard
# Heavy relative to the fish ON PURPOSE. The contact correction is split by
# inverse mass, so a light pad simply gets shoved aside by the fish instead of
# clamping it -- the mass ratio IS the clamping stiffness at a contact.
FIN_MASS = 0.8
RIB_EVERY = 2                     # draw a cross rib every Nth row (render only)

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
    """Load a glTF fish mesh; return (verts, faces, uv, material) with the verts
    in the procedural template frame: length along +X, up +Y, centred at the
    origin, scaled so the long axis is `length` metres. The model's length runs
    along +Z, so a +90 deg rotation about Y (proper, winding preserved) maps it
    to +X. The mesh is NOT vendored -- this only runs when --fish-model is given."""
    root = tp.GLTFLoader().load(path).scene
    found = []

    def rec(o):
        if isinstance(o, tp.Mesh):
            found.append(o)
        for ch in o.children:
            rec(ch)

    rec(root)
    mesh = found[0]
    mesh.update_matrix_world(True)
    g = mesh.geometry
    P = np.asarray(g.get_attribute("position"), np.float64).reshape(-1, 3)
    uv = np.asarray(g.get_attribute("uv"), np.float32).reshape(-1, 2)
    faces = np.asarray(g.get_index(), np.int64).reshape(-1, 3).astype(np.int32)
    M = np.asarray(mesh.matrix_world.to_numpy(), np.float64).reshape(4, 4)
    Pw = (M[:3, :3] @ P.T).T + M[:3, 3]                  # bake the node transform
    Ry90 = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], np.float64)   # +Z -> +X
    Pr = (Ry90 @ Pw.T).T
    Pr -= 0.5 * (Pr.max(0) + Pr.min(0))                  # centre at origin
    scale = length / (Pr[:, 0].max() - Pr[:, 0].min())
    return (Pr * scale).astype(np.float32), faces, uv, mesh.material


FISH_MODEL_CACHE = None
if FISH_MODEL and os.path.exists(FISH_MODEL):
    FISH_MODEL_CACHE = load_fish_model(FISH_MODEL, FISH_LEN)
    _rv = FISH_MODEL_CACHE[0]
    _body = _rv[np.abs(_rv[:, 0]) < 0.40 * FISH_LEN]     # drop the caudal (tail fin)
    if len(_body) < 16:
        _body = _rv
    # Robust body half-extents: a high percentile of |y| / |z| over the central
    # body verts, so the thin dorsal/anal/pectoral fins do not inflate the cage.
    _bh = 2.0 * float(np.percentile(np.abs(_body[:, 1]), 92.0))
    _bw = 2.0 * float(np.percentile(np.abs(_body[:, 2]), 92.0))
    _old_h, _old_w = FISH_H, FISH_W
    # WIDTH is the grasp axis and the source of the user's gap: match it to the
    # real flank so both pads bite the flesh with no side gap.
    FISH_W = _bw
    # HEIGHT: the fish is spawned STANDING (flanks facing the pads) so the 48 mm
    # width fits the 80 mm hand. A tall-narrow ellipsoid standing on its belly is
    # an egg on its end -- unstable -- and a full-height (91 mm) cage tips onto its
    # flank during the settle and escapes the jaws (measured). So cap the cage's
    # cross-section aspect near the procedural fish's (which stayed upright): the
    # render still shows the full 91 mm body (its top/bottom extrapolate off the
    # cage and droop upright, exactly as before this fix), while the physics cage
    # stays stable AND fills the flank the pads actually close on.
    STAND_ASPECT = 1.15
    FISH_H = min(_bh, STAND_ASPECT * FISH_W)
    FISH_P = (TRAY_C[0], TRAY_Y + FISH_H * 0.5, TRAY_C[1])
    # The pad paddle (FIN_LEN along the tool axis, FIN_WIDTH along the fish) already
    # spans a 55 mm cage flank comfortably -- 60 mm tall, 110 mm along the body --
    # so it is left at its tuned size; growing it just buckled the thin skins.
    print(f"  fish-model body extents: real body {_bh * 1000:.0f} x {_bw * 1000:.0f} mm -> "
          f"cage {FISH_H * 1000:.0f} x {FISH_W * 1000:.0f} mm "
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
cage_v, cage_t, CAGE_LO, CAGE_DIMS, CAGE_SOLID, CAGE_ID = fish_cage(CAGE_H)
CAGE_N, N_TETS = len(cage_v), len(cage_t)
CAGE_R = CAGE_R_CELLS * CAGE_H
# The cage nodes are a uniform lattice through a body of uniform density, so
# their mean IS the centre of mass. It sits behind the mid-point because the
# taper puts the meat in the front half.
FISH_COM_X = float(cage_v[:, 0].mean())
GRASP_X = FISH_P[0] + FISH_COM_X
# The widest half-section within a centimetre of the grasp station: the position
# backstop for the close is quoted off this.
_near = np.abs(fish_tmpl[:, 0] - FISH_COM_X) < 0.012
FISH_HALF_W = float(np.abs(fish_tmpl[_near, 2]).max())

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
rad_np[FISH_BASE:] = CAGE_R
GRID_R = 2.0 * float(rad_np.max()) * CONTACT_MARGIN
p0 = np.array(positions, dtype=np.float32)
offsets_np, idx_np, rest_np, stiff_np, _ = csr_from_pairs(NP, pairs, p0)

wp.init()
device = wp.get_preferred_device()
print(f"franka softgrip: {NP} particles ({FIN_TOTAL} finger + {CAGE_N} fish cage), "
      f"{len(idx_np)} constraint rows, on {device}")
print(f"  fish {FISH_LEN * 1000:.0f} x {FISH_H * 1000:.0f} x {FISH_W * 1000:.0f} mm, "
      f"{FISH_MASS * 1000:.0f} g: cage {CAGE_N} verts / {N_TETS} tets / {len(_e)} edges at "
      f"{CAGE_H * 1000:.1f} mm pitch, contact r {CAGE_R * 1000:.1f} mm, "
      f"{FISH_NV} skinned render verts (bind err {_bind_err.mean() * 1e6:.1f} um)")

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
def build_contacts(pos: wp.array(dtype=wp.vec3),
                   grid: wp.uint64,
                   body: wp.array(dtype=int),
                   rad: wp.array(dtype=float),
                   cont_idx: wp.array(dtype=int),
                   cont_d: wp.array(dtype=float),
                   cont_n: wp.array(dtype=int)):
    # One hash-grid sweep per substep; the iterations replay the list. Same-body
    # pairs are skipped -- a tet body is held open by its own volume rows and a
    # truss by its own struts, and self-contact there is only noise.
    i = wp.tid()
    p = pos[i]
    b = body[i]
    n = int(0)
    q = wp.hash_grid_query(grid, p, GRID_R)
    for j in q:
        if body[j] != b and n < CONTACT_CAP:
            dist = wp.length(p - pos[j])
            cr = rad[i] + rad[j]
            if dist < cr * CONTACT_MARGIN:
                cont_idx[i * CONTACT_CAP + n] = j
                # The penetration LATCHED at collision time is the friction
                # budget. Sizing friction off the live residual overlap instead
                # is self-defeating: the normal constraint drives that residual
                # to zero, so a well-solved contact ends the substep with no
                # friction left and the fish slides out of a hand that is
                # provably squeezing it.
                cont_d[i * CONTACT_CAP + n] = cr - dist
                n += 1
    cont_n[i] = n


@wp.kernel
def solve(p_in: wp.array(dtype=wp.vec3),
          p_out: wp.array(dtype=wp.vec3),
          prev: wp.array(dtype=wp.vec3),
          offsets: wp.array(dtype=int),
          indices: wp.array(dtype=int),
          rests: wp.array(dtype=float),
          stiffs: wp.array(dtype=float),
          cont_idx: wp.array(dtype=int),
          cont_d: wp.array(dtype=float),
          cont_n: wp.array(dtype=int),
          invm: wp.array(dtype=float),
          mu: wp.array(dtype=float),
          rad: wp.array(dtype=float),
          pad: wp.array(dtype=int),
          body: wp.array(dtype=int),
          record: int,
          f_scale: float,
          pad_force: wp.array(dtype=float),
          pad_hits: wp.array(dtype=int)):
    i = wp.tid()
    p = p_in[i]
    wi = invm[i]
    if wi == 0.0:
        p_out[i] = p
        return
    c = wp.vec3(0.0, 0.0, 0.0)
    # Truss / tet distance constraints, Jacobi half-corrections, NORMALISED by
    # the node's own summed stiffness once that sum exceeds 1. A Jacobi sweep is
    # only stable while the total gain at a node is below unity, and a truss node
    # here sits on ~30 rows (skin, rib, two diagonals, root fan) while a cage
    # node sits on 14 tet edges plus its spine chords -- so the raw sum was 2-9x
    # over. That surplus is what rang the pads at the truss frequency and what
    # blew the fish to NaN the moment the edges were stiffened. Normalising
    # keeps the RELATIVE weighting (stiff skins, soft diagonals) exactly and
    # caps the per-iteration move at RELAX; 16 iterations do the converging.
    # Contacts are deliberately left OUT of the normalisation: a clamp has to be
    # able to move a node the whole penetration in one go.
    wsum = float(0.0)
    for k in range(offsets[i], offsets[i + 1]):
        d = p_in[indices[k]] - p
        l = wp.length(d)
        if l > 1.0e-9:
            c += d * (0.5 * stiffs[k] * (l - rests[k]) / l)
        wsum += stiffs[k]
    # ...and only on the FISH. The truss was never the thing that diverged, and
    # normalising it as well costs the clamp: measured, the same close went from
    # 3.1 N to 0.9 N and pushed the fish across the tray instead of lifting it.
    # The pads' ringing is handled where it actually comes from -- the once-per-
    # frame pin update (now interpolated) and DAMPING_FIN.
    if body[i] == 2:
        c = c * (1.0 / wp.max(1.0, 0.5 * wsum))
    else:
        # ...and on the TRUSS, a CAP rather than a normalisation: phase 5 added
        # the skins' bending chords, which took the node's gain past what
        # DAMPING_FIN could absorb and made the skins ring in a per-row spike
        # pattern. Dividing only the surplus keeps every pre-phase-5 row at full
        # strength, so the clamp survives (measured below 3 N either way).
        c = c * (FIN_GAIN_CAP / wp.max(FIN_GAIN_CAP, 0.5 * wsum))
    # particle-particle contact, normal split by inverse mass + Coulomb friction
    for k in range(cont_n[i]):
        j = cont_idx[i * CONTACT_CAP + k]
        pj = p_in[j]
        d = p - pj
        l = wp.length(d)
        cr = rad[i] + rad[j]
        if l < cr and l > 1.0e-9:
            wj = invm[j]
            share = wi / (wi + wj + 1.0e-12)
            n = d / l
            push = n * ((cr - l) * share)
            c += push
            # Static friction, positionally: cancel the tangential relative
            # motion of this substep, up to mu times the normal correction.
            rel = (p - prev[i]) - (pj - prev[j])
            t = rel - n * wp.dot(rel, n)
            tl = wp.length(t)
            if tl > 1.0e-9:
                budget = wp.max(cont_d[i * CONTACT_CAP + k], cr - l)
                lim = wp.min(tl, mu[i] * wp.max(budget, 0.0))
                c -= t * (lim / tl * share)
            if record == 1 and pad[i] == 1 and body[j] == 2:
                wp.atomic_add(pad_force, body[i], wp.length(push) * RELAX * f_scale)
                wp.atomic_add(pad_hits, 0, 1)
    p_out[i] = p + c * RELAX


@wp.kernel
def tet_lambda_reset(lam_d: wp.array(dtype=float), lam_h: wp.array(dtype=float)):
    # XPBD multipliers are accumulated across one substep and reset before the
    # next -- this is what makes the compliance dt-independent (a real elastic
    # modulus) rather than an iteration-count-dependent PBD stiffness.
    t = wp.tid()
    lam_d[t] = 0.0
    lam_h[t] = 0.0


@wp.kernel
def tet_neohookean(pos: wp.array(dtype=wp.vec3),
                   tets: wp.array2d(dtype=int),
                   dm_inv: wp.array(dtype=wp.mat33),
                   alpha_d: wp.array(dtype=float),
                   alpha_h: wp.array(dtype=float),
                   gamma: float,
                   invm: wp.array(dtype=float),
                   base: int,
                   lam_d: wp.array(dtype=float),
                   lam_h: wp.array(dtype=float),
                   dpos: wp.array(dtype=wp.vec3),
                   cnt: wp.array(dtype=int)):
    # Stable Neo-Hookean (Macklin/Muller/Chentanez 2021), two XPBD constraints
    # per tet. F = Ds @ Dm_inv, Ds columns = [x1-x0, x2-x0, x3-x0]. The gradient
    # of any scalar phi(F) wrt the inner verts is the columns of (dphi/dF)@Dm_inv^T
    # and wrt x0 is minus their sum. This is inversion-safe: the hydrostatic row
    # C = det(F) - gamma restores volume even from det(F) < 0, and the deviatoric
    # row C = sqrt(tr(F^T F)) is always well defined. gamma = 1 + mu/lambda is
    # chosen so the energy minimum is exactly F = I (undeformed rest).
    t = wp.tid()
    la, lb, lc, ld = tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3]
    ia, ib, ic, id_ = base + la, base + lb, base + lc, base + ld
    x0, x1, x2, x3 = pos[ia], pos[ib], pos[ic], pos[id_]
    w0, w1, w2, w3 = invm[ia], invm[ib], invm[ic], invm[id_]
    e1 = x1 - x0
    e2 = x2 - x0
    e3 = x3 - x0
    di = dm_inv[t]
    # deformation gradient columns
    f0 = e1 * di[0, 0] + e2 * di[1, 0] + e3 * di[2, 0]
    f1 = e1 * di[0, 1] + e2 * di[1, 1] + e3 * di[2, 1]
    f2 = e1 * di[0, 2] + e2 * di[1, 2] + e3 * di[2, 2]

    # deviatoric constraint: C = sqrt(tr(F^T F)), stiffness mu (alpha_d)
    s = wp.dot(f0, f0) + wp.dot(f1, f1) + wp.dot(f2, f2)
    cd = wp.sqrt(wp.max(s, 1.0e-12))
    inv_cd = 1.0 / cd
    # (F @ Dm_inv^T) columns give grads wrt x1,x2,x3
    g1 = (f0 * di[0, 0] + f1 * di[0, 1] + f2 * di[0, 2]) * inv_cd
    g2 = (f0 * di[1, 0] + f1 * di[1, 1] + f2 * di[1, 2]) * inv_cd
    g3 = (f0 * di[2, 0] + f1 * di[2, 1] + f2 * di[2, 2]) * inv_cd
    g0 = -(g1 + g2 + g3)
    ad = alpha_d[t]
    den = (w0 * wp.dot(g0, g0) + w1 * wp.dot(g1, g1)
           + w2 * wp.dot(g2, g2) + w3 * wp.dot(g3, g3) + ad)
    dl = (-cd - ad * lam_d[t]) / den
    lam_d[t] = lam_d[t] + dl
    da0 = g0 * (w0 * dl)
    da1 = g1 * (w1 * dl)
    da2 = g2 * (w2 * dl)
    da3 = g3 * (w3 * dl)

    # hydrostatic constraint: C = det(F) - gamma, stiffness lambda (alpha_h)
    c0 = wp.cross(f1, f2)
    c1 = wp.cross(f2, f0)
    c2 = wp.cross(f0, f1)
    detf = wp.dot(f0, c0)
    h1 = c0 * di[0, 0] + c1 * di[0, 1] + c2 * di[0, 2]
    h2 = c0 * di[1, 0] + c1 * di[1, 1] + c2 * di[1, 2]
    h3 = c0 * di[2, 0] + c1 * di[2, 1] + c2 * di[2, 2]
    h0 = -(h1 + h2 + h3)
    ch = detf - gamma
    ah = alpha_h[t]
    denh = (w0 * wp.dot(h0, h0) + w1 * wp.dot(h1, h1)
            + w2 * wp.dot(h2, h2) + w3 * wp.dot(h3, h3) + ah)
    dlh = (-ch - ah * lam_h[t]) / denh
    lam_h[t] = lam_h[t] + dlh
    da0 = da0 + h0 * (w0 * dlh)
    da1 = da1 + h1 * (w1 * dlh)
    da2 = da2 + h2 * (w2 * dlh)
    da3 = da3 + h3 * (w3 * dlh)

    wp.atomic_add(dpos, la, da0)
    wp.atomic_add(dpos, lb, da1)
    wp.atomic_add(dpos, lc, da2)
    wp.atomic_add(dpos, ld, da3)
    wp.atomic_add(cnt, la, 1)
    wp.atomic_add(cnt, lb, 1)
    wp.atomic_add(cnt, lc, 1)
    wp.atomic_add(cnt, ld, 1)


@wp.kernel
def tet_apply(pos: wp.array(dtype=wp.vec3),
              base: int,
              dpos: wp.array(dtype=wp.vec3),
              cnt: wp.array(dtype=int)):
    # Jacobi mass-split: average the XPBD corrections over the tets incident to
    # this node, then self-clear. The division by the incident-tet count IS the
    # under-relaxation that keeps the parallel-over-tets solve stable; adding the
    # distance rows' extra RELAX 0.5 on top double-damped it and the body could
    # not even resist gravity (rest det(F) deviation 0.22), so there is no RELAX
    # here.
    i = wp.tid()
    n = cnt[i]
    if n > 0:
        pos[base + i] = pos[base + i] + dpos[i] * (1.0 / float(n))
        dpos[i] = wp.vec3(0.0, 0.0, 0.0)
        cnt[i] = 0


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


@wp.kernel
def statics(pos: wp.array(dtype=wp.vec3),
            prev: wp.array(dtype=wp.vec3),
            invm: wp.array(dtype=float),
            rad: wp.array(dtype=float)):
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
        t = p - prev[i]
        p = p - wp.vec3(t[0], 0.0, t[2]) * m
    pos[i] = p


@wp.kernel
def fish_centroid(pos: wp.array(dtype=wp.vec3), base: int, n: int,
                  out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    wp.atomic_add(out, 0, pos[base + i] / float(n))


# --- device state ---------------------------------------------------------------

x = wp.array(p0, dtype=wp.vec3, device=device)
prev = wp.array(p0, dtype=wp.vec3, device=device)
pred = wp.zeros(NP, dtype=wp.vec3, device=device)
scratch = wp.zeros(NP, dtype=wp.vec3, device=device)
nrm = wp.zeros(NP, dtype=wp.vec3, device=device)
centroid = wp.zeros(1, dtype=wp.vec3, device=device)
pad_force = wp.zeros(2, dtype=float, device=device)
pad_hits = wp.zeros(1, dtype=int, device=device)
tet_dpos = wp.zeros(CAGE_N, dtype=wp.vec3, device=device)
tet_cnt = wp.zeros(CAGE_N, dtype=int, device=device)
tets_d = wp.array(cage_t, dtype=int, device=device)
tet_dm_inv_d = wp.array(tet_Dm_inv, dtype=wp.mat33, device=device)
tet_alpha_d_d = wp.array(tet_alpha_d, dtype=float, device=device)
tet_alpha_h_d = wp.array(tet_alpha_h, dtype=float, device=device)
tet_lam_d = wp.zeros(N_TETS, dtype=float, device=device)   # deviatoric multiplier
tet_lam_h = wp.zeros(N_TETS, dtype=float, device=device)   # hydrostatic multiplier
skin_ids_d = wp.array(skin_ids, dtype=int, device=device)
skin_w_d = wp.array(skin_w, dtype=float, device=device)

offsets = wp.array(offsets_np, dtype=int, device=device)
indices = wp.array(idx_np, dtype=int, device=device)
rests = wp.array(rest_np, dtype=float, device=device)
stiffs = wp.array(stiff_np, dtype=float, device=device)
_damp_np = np.full(NP, DAMPING, dtype=np.float32)
_damp_np[:FIN_TOTAL] = DAMPING_FIN
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
local_arr = wp.array(local_np, dtype=wp.vec3, device=device)
owner_arr = wp.array(owner_np, dtype=int, device=device)
mats = wp.zeros(2, dtype=wp.mat44, device=device)

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
fish_pos = wp.zeros(fish_corners, dtype=wp.vec3, device=device)
fish_nrm = wp.zeros(fish_corners, dtype=wp.vec3, device=device)

grid = wp.HashGrid(48, 48, 48, device)
cont_idx = wp.zeros(NP * CONTACT_CAP, dtype=int, device=device)
cont_d = wp.zeros(NP * CONTACT_CAP, dtype=float, device=device)
cont_n = wp.zeros(NP, dtype=int, device=device)

# PBD impulse -> newtons. The contact corrections summed over a substep's
# iterations ARE the total correction the contact had to make, and a correction
# dx applied to a particle of mass m over a substep is the force m dx / dt^2.
# Measured on the PAD particles, so the mass is the pad's; measuring the fish
# side instead gives the same number, because the split is mass-weighted.
F_SCALE = MASS_PAD / (DT * DT) / float(ITERATIONS)


def substep_body():
    """One substep. Device work ONLY -- no host reads, no branching on sim
    state -- so the whole thing captures into a CUDA graph."""
    wp.launch(pin_follow, dim=NP, device=device,
              inputs=[local_arr, mats, owner_arr, pinned_arr, x, prev])
    wp.launch(integrate_damped, dim=NP, device=device,
              inputs=[x, prev, pred, DT, damp_arr, GRAVITY])
    wp.launch(pin_follow, dim=NP, device=device,
              inputs=[local_arr, mats, owner_arr, pinned_arr, pred, prev])
    grid.build(points=pred, radius=GRID_R)
    wp.launch(build_contacts, dim=NP, device=device,
              inputs=[pred, grid.id, body_arr, rad_arr, cont_idx, cont_d, cont_n])
    wp.launch(tet_lambda_reset, dim=N_TETS, device=device, inputs=[tet_lam_d, tet_lam_h])
    pa, pb = pred, scratch
    for _ in range(ITERATIONS):
        wp.launch(solve, dim=NP, device=device,
                  inputs=[pa, pb, prev, offsets, indices, rests, stiffs, cont_idx, cont_d, cont_n,
                          invm, mu_arr, rad_arr, pad_arr, body_arr, 1, F_SCALE,
                          pad_force, pad_hits])
        pa, pb = pb, pa
        # The Neo-Hookean FEM rows for the fish, in the same Jacobi sweep as the
        # finger truss's distance rows. Two constraints per tet, XPBD multipliers
        # carried across the substep, accumulated per cage node then averaged.
        wp.launch(tet_neohookean, dim=N_TETS, device=device,
                  inputs=[pa, tets_d, tet_dm_inv_d, tet_alpha_d_d, tet_alpha_h_d,
                          NH_GAMMA, invm, FISH_BASE, tet_lam_d, tet_lam_h,
                          tet_dpos, tet_cnt])
        wp.launch(tet_apply, dim=CAGE_N, device=device,
                  inputs=[pa, FISH_BASE, tet_dpos, tet_cnt])
    wp.launch(statics, dim=NP, device=device, inputs=[pa, prev, invm, rad_arr])
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
WP_GRASP = (GRASP_X, TRAY_Y + 0.5 * FISH_H_REST, FISH_P[2])
WP_LIFT = (GRASP_X, 0.30, FISH_P[2])
WP_CARRY = (TARGET_C[0], 0.32, TARGET_CZ)
WP_LOWER = (TARGET_C[0], 0.17, TARGET_CZ)
WP_OUT = (TARGET_C[0], 0.36, TARGET_CZ)

# (name, waypoint or None to hold, seconds)
# CARRY and LOWER are 30 % longer than phase 2b. The quintic's peak acceleration
# scales as 1/T^2, and the swing across to the crate was throwing the hanging
# fish out of the pads on the way in and out of the blend.
PLAN = [("SETTLE", None, 0.4),
        ("PREGRASP", WP_PRE, 1.1),
        ("DESCEND", WP_GRASP, 1.0),
        ("CLOSE", None, 2.3),
        ("LIFT", WP_LIFT, 2.6),
        ("CARRY", WP_CARRY, 2.1),
        ("LOWER", WP_LOWER, 1.5),
        ("OPEN", None, 0.9),
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
CLOSE_V = 0.009          # m/s per carriage
SQUEEZE = 0.002          # extra bite once the threshold trips
# The release used CLOSE_V too, which at 9 mm/s needs 1.8 s to give back the
# 16 mm the close took -- longer than the OPEN phase, so on half the runs the
# fish was still in the hand at DONE. Letting go is not a delicate operation.
OPEN_V = 0.060           # m/s per carriage
GRIP_CONFIRM = 5         # consecutive frames over F_GRASP before the close stops
grip_hi_n = 0
# Position backstop, and it is the primary control -- the force proxy only
# short-circuits it. Measured, not assumed: the carriage frame sits Y_CARRIAGE
# off the tool axis at FINGER_OPEN and tracks the dof one-for-one, and the pad's
# flat working face is FIN_SKIN_IN + PART_R inboard of that. So the dof at which
# the pad face reaches a given half-width is arithmetic, and it has to be,
# because the dof is NOT the half-gap (the carriage origin is offset).
Y_CARRIAGE = abs(float(CARRIAGE_TCP[0][1]))
BITE = 0.003             # how far inside the flank the pad face is driven
FINGER_STOP = max(FINGER_MIN, FINGER_OPEN - Y_CARRIAGE + FIN_SKIN_IN + PART_R
                  + FISH_HALF_W - BITE)


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
        grip_hi_n = grip_hi_n + 1 if grip_force >= F_GRASP else 0
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


_mats_cur = np.stack([link_mat(left_link), link_mat(right_link)]).astype(np.float32)
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
    for k in (0, 1):                       # re-orthonormalise the lerped basis
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
    contact_r = PART_R + CAGE_R
    print(f"  [gap:{tag:5s}] cage bbox ({cmin[0]:.3f},{cmin[1]:.3f},{cmin[2]:.3f})"
          f"..({cmax[0]:.3f},{cmax[1]:.3f},{cmax[2]:.3f}) c=({cc[0]:.3f},{cc[1]:.3f},{cc[2]:.3f})"
          f" render bbox ({rmin[0]:.3f},{rmin[1]:.3f},{rmin[2]:.3f})..({rmax[0]:.3f},{rmax[1]:.3f},{rmax[2]:.3f})")
    for k in (0, 1):
        b = k * FIN_N
        pad = xp[b:b + FIN_ROWS * FIN_COLS]          # front skin = the inner face
        pc = pad.mean(0)
        d = np.sqrt(((pad[:, None, :] - cage[None, :, :]) ** 2).sum(-1))
        gap = float(d.min()) - contact_r
        print(f"  [gap:{tag:5s}] finger {k} pad-face c=({pc[0]:+.3f},{pc[1]:+.3f},{pc[2]:+.3f})"
              f" min surface gap {gap * 1000:+5.1f} mm")


def step_frame():
    """One 60 fps frame: task -> robot -> pins -> substeps -> surfaces."""
    global grip_force, sim_t, tcp_prev, slip_peak, slip_mean, slip_n, _close_diag_pending
    t0 = time.perf_counter()
    phase = advance_task(1.0 / FPS)
    set_q(q_cur)
    _mats_prev[:] = _mats_cur
    _mats_cur[0] = link_mat(left_link)
    _mats_cur[1] = link_mat(right_link)
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
    rib_vn.zero_()
    wp.launch(accum_normals, dim=N_RIB_FACES, device=device, inputs=[x, rib_tris_d, rib_vn])
    wp.launch(scatter_soup_safe, dim=rib_corners, device=device,
              inputs=[x, rib_vn, rib_tris_d, rib_pos, rib_nrm])
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

    # Normalised: the raw sum scales with ITERATIONS (folded into F_SCALE) and
    # with how many pad nodes happen to be touching, which made the phase-2
    # number unusable as a threshold. Per contacting node, it is comparable.
    hits = max(int(pad_hits.numpy()[0]), 1)
    grip_force = float(pad_force.numpy().sum()) / SUBSTEPS
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
        ncon = int((cont_n.numpy()[FISH_BASE:] > 0).sum())
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
    for _e, _ci in zip(eyes, EYE_CORNER):
        _p = fish_np[_ci]
        _e.position.set(float(_p[0]), float(_p[1]), float(_p[2]))
    aim_camera(tcp)
    return t1 - t0, t2 - t1, t3 - t2, phase, c


step_frame.c_prev = np.array(FISH_P, dtype=np.float32)
step_frame.phase_prev = ""


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
_try(renderer.set_auto_exposure_range, 0.15, 3.0)
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
try:
    env_tex = tp.float_texture(studio_env())
    scene.environment = env_tex
except Exception as _e:                              # noqa: BLE001
    print(f"  note: environment unavailable ({_e})")

camera = tp.PerspectiveCamera(42 if CAM != "macro" else 34, canvas.aspect(), 0.03, 40)
CAM_HOME = {"wide": ((1.30, 0.72, 0.92), (0.30, 0.20, 0.0)),
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
# "clean bright cell" the brief asks for.
scene.add(tp.HemisphereLight(0xcfe0ff, 0x2a2e33, 0.9))
sun = tp.DirectionalLight(0xeaf2ff, 2.0)
sun.position.set(-0.8, 3.0, 1.0)
sun.cast_shadow = True
scene.add(sun)
key = tp.DirectionalLight(0xffe0c0, 1.5)
key.position.set(1.8, 1.4, 1.6)
scene.add(key)

# Brushed stainless, now that there IS an environment for it to reflect: the
# arm and the crates show up in the table as soft vertical smears.
table = tp.Mesh(tp.BoxGeometry(2.2, 0.06, 1.6), standard_material(0xb9bdc2, 0.28, 0.90))
table.position.set(0.35, -0.03, 0.0)
table.receive_shadow = True
scene.add(table)

tray = tp.Mesh(tp.BoxGeometry(2 * TRAY_HX, TRAY_Y, 2 * TRAY_HZ),
               standard_material(0xc4c9ce, 0.22, 0.92))
tray.position.set(TRAY_C[0], TRAY_Y * 0.5, TRAY_C[1])
tray.receive_shadow = True
scene.add(tray)

crate_mat = standard_material(0x2f6fae, 0.55, 0.0)
for cc in (CRATE_C, CRATE2_C):
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
    return ok


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


if FILM:
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
        if nm == "CLOSE":
            return 0.15
        if nm == "LIFT":
            return min(1.0, 0.15 + seg_t / 1.2 * 0.85)
        return 1.0

    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target.set(0.44, 0.10, 0.02)

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        renderer.set_size(w, h)

    canvas.on_window_resize(on_resize)
    state = {"done": False, "phase": ""}

    def animate():
        slow["acc"] += rate()
        while slow["acc"] >= 1.0:
            slow["acc"] -= 1.0
            parts = step_frame()
            state["phase"] = parts[3]
            if parts[3] == "DONE" and not state["done"]:
                state["done"] = True
                summary(parts[4])
        if CAM != "macro":
            controls.update()
        renderer.render(scene, wrist_cam if POV else camera)
        ensure_pip()

    canvas.animate(animate)
