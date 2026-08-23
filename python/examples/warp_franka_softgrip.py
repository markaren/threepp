"""Franka FR3 with soft Fin Ray fingers picking a soft fish -- Warp sim, Vulkan render.

The arm is the real FR3 URDF driven by threepp's own damped-least-squares IK
(tp.IkSolver) along joint-space quintic trajectories. The two fingers are NOT
the rigid stock ones: each stock finger carries a GPU soft body -- an XPBD
two-skin truss, front skin and back skin joined by stiff ribs and SOFT
diagonals, which is what makes it a Fin Ray: press the front skin and the tip
curls toward the object. The fish is a FILLED XPBD tet body: a voxel-carved
conforming cage with a volume row per tet, with the drawn surface skinned off
it. A hollow shell was the thing that lost the grasp -- pinching one between
two converging paddles extrudes it out of the hand.

Fingers and fish live in ONE solver over ONE hash grid with Coulomb friction,
so the grasp is friction plus form closure. There is no kinematic attach and no
glue anywhere: if the fingers do not hold it, it falls.

The close is force-controlled -- the carriages drive inward until the summed
pad contact force crosses a threshold, then hold.

    pip install warp-lang
    python warp_franka_softgrip.py                # window
    python warp_franka_softgrip.py --shot 3.2     # headless PNG at sim time 3.2 s
    python warp_franka_softgrip.py --frames 600   # timed phase breakdown
    python warp_franka_softgrip.py --grip-force 8 # close threshold, newtons
    python warp_franka_softgrip.py --fish-len 0.3 # graded sizes: 0.24 / 0.27 / 0.30
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import (accum_normals, cli_arg, csr_from_pairs, icosphere, integrate,
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

# --- solver tunables ------------------------------------------------------------

FPS = 60.0
DT = 1.0 / 240.0
SUBSTEPS = 6
ITERATIONS = 16
DAMPING = 0.04
GRAVITY = wp.vec3(0.0, -9.81, 0.0)
RELAX = 0.5

# Fin Ray split: the skins are structural, the diagonals are the compliance.
STIFF_SKIN = 1.0
STIFF_RIB = 0.7
STIFF_DIAG = 0.35
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
STIFF_TET_EDGE = 0.6  # squishy flesh, in tension and compression
STIFF_TET_VOL = 1.0   # a fish is water: volume is the constraint that matters
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
CRATE_C = (0.36, -0.42)
CRATE_HX, CRATE_HZ = 0.16, 0.13
CRATE_FLOOR = 0.03
CRATE_RIM = 0.20
TRAY_CX, TRAY_CZ = float(TRAY_C[0]), float(TRAY_C[1])
CRATE_CX, CRATE_CZ = float(CRATE_C[0]), float(CRATE_C[1])

# A herring, not a salmon. The stock FR3 hand opens to 8 cm, so what limits the
# size is the WIDTH, never the length: 45 mm is about all the carriages can bite
# into and still have travel left. Length is a free parameter (--fish-len), and
# the three graded sizes are 0.24 / 0.27 / 0.30 m.
FISH_LEN = cli_arg("--fish-len", 0.24, float)
FISH_H = 0.22 * FISH_LEN
FISH_W = min(0.045, 0.17 * FISH_LEN)
FISH_SUBDIV = 3
FISH_P = (TRAY_C[0], TRAY_Y + FISH_H * 0.5, TRAY_C[1])
# Grasp station: 40 % of the length back from the nose, i.e. the thick shoulder
# just behind the head, not the mid-body waist.
GRASP_X = FISH_P[0] + (0.4 - 0.5) * FISH_LEN

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
FIN_ROWS, FIN_COLS = 11, 16
FIN_Z0, FIN_LEN = -0.032, 0.060   # root at the carriage, tip ~28 mm past the TCP
FIN_WIDTH = 0.090                 # across the grasp, i.e. along the fish
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

URDF = "C:/dev/threepp/cmake-build-relwithdebinfo/_deps/threepp_data-src/urdf/franka/fr3.urdf"
if not os.path.exists(URDF):
    URDF = "C:/dev/threepp-data/urdf/franka/fr3.urdf"


# --- [B] robot ------------------------------------------------------------------

robot = tp.URDFLoader().load(URDF)
robot.rotation.x = -math.pi / 2          # URDF is Z-up, the world is Y-up
robot.position.set(0.0, 0.0, 0.0)
robot.update_matrix_world(True)

# The URDF ships MeshPhongMaterial, which the Vulkan deferred path draws as
# white wireframe junk. Swap every Phong/Basic for a Standard of the same
# colour (material is read-only; set_material is the door).
_swapped = 0


def _restandardise(o):
    global _swapped
    if isinstance(o, tp.Mesh):
        m = getattr(o, "material", None)
        if m is not None and not isinstance(m, tp.MeshStandardMaterial):
            col = getattr(m, "color", None)
            o.set_material(standard_material(col if col is not None else 0xd8d8d8, 0.45, 0.0))
            _swapped += 1


robot.traverse(_restandardise)

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
    """Closed-box triangles over one finger's particle block, wound outward."""
    def pid(i, j, skin):
        return base + skin * FIN_ROWS * FIN_COLS + i * FIN_COLS + j

    f = []

    def quad(a, b, c, d):
        f.append((a, b, c))
        f.append((a, c, d))

    for i in range(FIN_ROWS - 1):
        for j in range(FIN_COLS - 1):
            quad(pid(i, j, 0), pid(i, j + 1, 0), pid(i + 1, j + 1, 0), pid(i + 1, j, 0))
            quad(pid(i, j, 1), pid(i + 1, j, 1), pid(i + 1, j + 1, 1), pid(i, j + 1, 1))
    for i in range(FIN_ROWS - 1):
        quad(pid(i, 0, 0), pid(i, 0, 1), pid(i + 1, 0, 1), pid(i + 1, 0, 0))
        quad(pid(i, FIN_COLS - 1, 0), pid(i + 1, FIN_COLS - 1, 0),
             pid(i + 1, FIN_COLS - 1, 1), pid(i, FIN_COLS - 1, 1))
    for j in range(FIN_COLS - 1):
        quad(pid(0, j, 0), pid(0, j + 1, 0), pid(0, j + 1, 1), pid(0, j, 1))
        quad(pid(FIN_ROWS - 1, j, 0), pid(FIN_ROWS - 1, j, 1),
             pid(FIN_ROWS - 1, j + 1, 1), pid(FIN_ROWS - 1, j + 1, 0))
    return np.array(f, dtype=np.int32)


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
                    p.append((pid(i, j, skin), pid(i + 1, j + 1, skin), STIFF_SKIN))
                    p.append((pid(i, j + 1, skin), pid(i + 1, j, skin), STIFF_SKIN))
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
fin_faces = []

for k in (0, 1):
    base = len(positions)
    pts_tcp = fin_ray_particles(k).transpose(2, 0, 1, 3).reshape(-1, 3)   # skin-major
    world = (M_TCP_REF[:3, :3] @ pts_tcp.T).T + M_TCP_REF[:3, 3]
    Cm = CARRIAGE_REF[k]
    local = (np.linalg.inv(Cm)[:3, :3] @ world.T).T + np.linalg.inv(Cm)[:3, 3]
    fin_local.append(local.astype(np.float32))
    fin_faces.append(finger_faces(base))
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

# The drawn surface is still the tapered ellipsoid; only what simulates it has
# changed. It is no longer a particle set at all -- it is SKINNED off the cage.
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
# The widest half-section within a centimetre of the grasp station: the position
# backstop for the close is quoted off this.
_near = np.abs(fish_tmpl[:, 0] - (GRASP_X - FISH_P[0])) < 0.010
FISH_HALF_W = float(np.abs(fish_tmpl[_near, 2]).max())

CAGE_H = FISH_LEN / TET_RES
cage_v, cage_t, CAGE_LO, CAGE_DIMS, CAGE_SOLID, CAGE_ID = fish_cage(CAGE_H)
CAGE_N, N_TETS = len(cage_v), len(cage_t)
CAGE_R = CAGE_R_CELLS * CAGE_H
skin_ids, skin_w = bind_lattice(fish_tmpl.astype(np.float64), CAGE_LO, CAGE_H,
                                CAGE_DIMS, CAGE_SOLID, CAGE_ID)
_bind_err = np.linalg.norm((cage_v[skin_ids] * skin_w[:, :, None]).sum(1) - fish_tmpl, axis=1)

# Every unique tet edge is a distance constraint; every tet is a volume row.
_e = np.unique(np.sort(np.concatenate([cage_t[:, [a, b]] for a, b in _TET_E]), 1), axis=0)
FISH_BASE = FIN_TOTAL
for a, b in _e:
    pairs.append((FISH_BASE + int(a), FISH_BASE + int(b), STIFF_TET_EDGE))
_t = cage_v[cage_t]
tet_v0 = (np.einsum("ni,ni->n", _t[:, 1] - _t[:, 0],
                    np.cross(_t[:, 2] - _t[:, 0], _t[:, 3] - _t[:, 0])) / 6.0).astype(np.float32)

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
      f"{FISH_NV} skinned render verts (bind err {_bind_err.mean() * 1e6:.1f} um), "
      f"{_swapped} robot materials restandardised")

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
    # truss / shell distance constraints (Jacobi half-corrections)
    for k in range(offsets[i], offsets[i + 1]):
        d = p_in[indices[k]] - p
        l = wp.length(d)
        if l > 1.0e-9:
            c += d * (0.5 * stiffs[k] * (l - rests[k]) / l)
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
def tet_volume(pos: wp.array(dtype=wp.vec3),
               tets: wp.array2d(dtype=int),
               v0: wp.array(dtype=float),
               invm: wp.array(dtype=float),
               base: int,
               dpos: wp.array(dtype=wp.vec3),
               cnt: wp.array(dtype=int)):
    # One incompressibility row PER TET, not one global row over the whole
    # surface. This is the whole point of the filled body: a squeeze in the
    # middle has to be answered locally, so the material cannot satisfy the
    # constraint by leaving the hand.
    t = wp.tid()
    la, lb, lc, ld = tets[t, 0], tets[t, 1], tets[t, 2], tets[t, 3]
    ia, ib, ic, id_ = base + la, base + lb, base + lc, base + ld
    a, b, c, d = pos[ia], pos[ib], pos[ic], pos[id_]
    ga = wp.cross(d - b, c - b) / 6.0
    gb = wp.cross(c - a, d - a) / 6.0
    gc = wp.cross(d - a, b - a) / 6.0
    gd = wp.cross(b - a, c - a) / 6.0
    wa, wb, wc, wd = invm[ia], invm[ib], invm[ic], invm[id_]
    den = (wa * wp.dot(ga, ga) + wb * wp.dot(gb, gb)
           + wc * wp.dot(gc, gc) + wd * wp.dot(gd, gd))
    if den < 1.0e-16:
        return
    v = wp.dot(b - a, wp.cross(c - a, d - a)) / 6.0
    lam = -(v - v0[t]) / den * STIFF_TET_VOL
    wp.atomic_add(dpos, la, ga * (lam * wa))
    wp.atomic_add(dpos, lb, gb * (lam * wb))
    wp.atomic_add(dpos, lc, gc * (lam * wc))
    wp.atomic_add(dpos, ld, gd * (lam * wd))
    wp.atomic_add(cnt, la, 1)
    wp.atomic_add(cnt, lb, 1)
    wp.atomic_add(cnt, lc, 1)
    wp.atomic_add(cnt, ld, 1)


@wp.kernel
def tet_apply(pos: wp.array(dtype=wp.vec3),
              base: int,
              dpos: wp.array(dtype=wp.vec3),
              cnt: wp.array(dtype=int)):
    # Jacobi average, same RELAX as the distance rows, then self-clear so the
    # pass costs two launches per iteration and no memsets.
    i = wp.tid()
    n = cnt[i]
    if n > 0:
        pos[base + i] = pos[base + i] + dpos[i] * (RELAX / float(n))
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
    in_crate = wp.abs(p[0] - CRATE_CX) < CRATE_HX and wp.abs(p[2] - CRATE_CZ) < CRATE_HZ
    if in_crate and p[1] < CRATE_RIM:
        floor = CRATE_FLOOR
        m = MU_CRATE
        # keep inside the walls
        x = wp.clamp(p[0], CRATE_CX - CRATE_HX + PART_R, CRATE_CX + CRATE_HX - PART_R)
        z = wp.clamp(p[2], CRATE_CZ - CRATE_HZ + PART_R, CRATE_CZ + CRATE_HZ - PART_R)
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
tet_v0 = wp.array(tet_v0, dtype=float, device=device)
skin_ids_d = wp.array(skin_ids, dtype=int, device=device)
skin_w_d = wp.array(skin_w, dtype=float, device=device)

offsets = wp.array(offsets_np, dtype=int, device=device)
indices = wp.array(idx_np, dtype=int, device=device)
rests = wp.array(rest_np, dtype=float, device=device)
stiffs = wp.array(stiff_np, dtype=float, device=device)
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
fish_tris_np = fish_faces_t.reshape(-1).astype(np.int32)
fish_tris_d = wp.array(fish_tris_np, dtype=int, device=device)
fish_v = wp.zeros(FISH_NV, dtype=wp.vec3, device=device)
fish_vn = wp.zeros(FISH_NV, dtype=wp.vec3, device=device)

fin_tris_np = np.concatenate([fin_faces[0].reshape(-1), fin_faces[1].reshape(-1)]).astype(np.int32)
fin_tris_d = wp.array(fin_tris_np, dtype=int, device=device)
fin_corners, fish_corners = len(fin_tris_np), len(fish_tris_np)
N_FIN_FACES = fin_corners // 3
N_FISH_FACES = fish_corners // 3
fin_pos = wp.zeros(fin_corners, dtype=wp.vec3, device=device)
fin_nrm = wp.zeros(fin_corners, dtype=wp.vec3, device=device)
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
    wp.launch(integrate, dim=NP, device=device, inputs=[x, prev, pred, DT, DAMPING, GRAVITY])
    wp.launch(pin_follow, dim=NP, device=device,
              inputs=[local_arr, mats, owner_arr, pinned_arr, pred, prev])
    grid.build(points=pred, radius=GRID_R)
    wp.launch(build_contacts, dim=NP, device=device,
              inputs=[pred, grid.id, body_arr, rad_arr, cont_idx, cont_d, cont_n])
    pa, pb = pred, scratch
    for _ in range(ITERATIONS):
        wp.launch(solve, dim=NP, device=device,
                  inputs=[pa, pb, prev, offsets, indices, rests, stiffs, cont_idx, cont_d, cont_n,
                          invm, mu_arr, rad_arr, pad_arr, body_arr, 1, F_SCALE,
                          pad_force, pad_hits])
        pa, pb = pb, pa
        # The tet volume rows, in the same Jacobi sweep as the distance rows.
        wp.launch(tet_volume, dim=N_TETS, device=device,
                  inputs=[pa, tets_d, tet_v0, invm, FISH_BASE, tet_dpos, tet_cnt])
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
WP_HOME = (0.40, 0.42, 0.05)
WP_PRE = (GRASP_X, 0.26, FISH_P[2])
# TCP at the fish's mid-height, so the 90 x 60 mm paddles cover the flank instead
# of scraping the top of it. The pad tip reaches ~28 mm below the TCP, i.e. just
# under the belly, which is where the inward hook does its work.
WP_GRASP = (GRASP_X, TRAY_Y + 0.5 * FISH_H_REST, FISH_P[2])
WP_LIFT = (GRASP_X, 0.30, FISH_P[2])
WP_CARRY = (CRATE_C[0], 0.32, CRATE_C[1])
WP_LOWER = (CRATE_C[0], 0.17, CRATE_C[1])
WP_OUT = (CRATE_C[0], 0.36, CRATE_C[1])

# (name, waypoint or None to hold, seconds)
PLAN = [("SETTLE", None, 0.4),
        ("PREGRASP", WP_PRE, 1.1),
        ("DESCEND", WP_GRASP, 1.0),
        ("CLOSE", None, 1.8),
        ("LIFT", WP_LIFT, 2.6),
        ("CARRY", WP_CARRY, 1.6),
        ("LOWER", WP_LOWER, 1.1),
        ("OPEN", None, 0.8),
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
    global seg_i, seg_t, q_cur, finger_cmd, grip_held
    name = PLAN[seg_i][0]
    dur = PLAN[seg_i][2]
    seg_t += dt
    s = min(seg_t / dur, 1.0)
    s = s * s * s * (10.0 + s * (-15.0 + 6.0 * s))      # quintic ease
    q = [a + (b - a) * s for a, b in zip(q_seg_start, q_seg_end)]

    if name == "CLOSE":
        if not grip_held and grip_force < F_GRASP and finger_cmd > FINGER_STOP:
            finger_cmd = max(FINGER_STOP, finger_cmd - CLOSE_V * dt)
        elif not grip_held:
            grip_held = True
            why = "force" if grip_force >= F_GRASP else "backstop"
            finger_cmd = max(FINGER_MIN, finger_cmd - SQUEEZE)
            report["grip_N"] = grip_force
            report["grip_open_mm"] = finger_cmd * 1000.0
            print(f"  CLOSE: held on {why} at {grip_force:.2f} N, "
                  f"carriage {finger_cmd * 1000:.1f} mm "
                  f"(backstop {FINGER_STOP * 1000:.1f})")
    elif name == "OPEN":
        finger_cmd = min(FINGER_OPEN, finger_cmd + CLOSE_V * dt)
    q[7] = q[8] = finger_cmd
    q_cur = q
    if seg_t >= dur and seg_i + 1 < len(PLAN):
        seg_i += 1
        begin_segment(seg_i)
    return name


_mats_host = np.zeros((2, 4, 4), dtype=np.float32)


def step_frame():
    """One 60 fps frame: task -> robot -> pins -> substeps -> surfaces."""
    global grip_force, sim_t, tcp_prev, slip_peak, slip_mean, slip_n
    t0 = time.perf_counter()
    phase = advance_task(1.0 / FPS)
    set_q(q_cur)
    _mats_host[0] = link_mat(left_link)
    _mats_host[1] = link_mat(right_link)
    mats.assign(_mats_host)
    t1 = time.perf_counter()

    pad_force.zero_()
    pad_hits.zero_()
    for _ in range(SUBSTEPS):
        if substep_graph is not None:
            wp.capture_launch(substep_graph)
        else:
            substep_body()
    centroid.zero_()
    wp.launch(fish_centroid, dim=CAGE_N, device=device,
              inputs=[x, FISH_BASE, CAGE_N, centroid])
    nrm.zero_()
    wp.launch(accum_normals, dim=N_FIN_FACES, device=device, inputs=[x, fin_tris_d, nrm])
    wp.launch(scatter_soup, dim=fin_corners, device=device,
              inputs=[x, nrm, fin_tris_d, fin_pos, fin_nrm])
    wp.launch(skin_lattice, dim=FISH_NV, device=device,
              inputs=[x, FISH_BASE, skin_ids_d, skin_w_d, fish_v])
    fish_vn.zero_()
    wp.launch(accum_normals, dim=N_FISH_FACES, device=device,
              inputs=[fish_v, fish_tris_d, fish_vn])
    wp.launch(scatter_soup, dim=fish_corners, device=device,
              inputs=[fish_v, fish_vn, fish_tris_d, fish_pos, fish_nrm])
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
        print(f"  {phase:9s} t={sim_t:5.2f}s grip {grip_force:6.2f} N over "
              f"{grip_nodes:5.1f} nodes | fish y {fv[:, 1].min():.3f}..{fv[:, 1].max():.3f} "
              f"(h {(fv[:, 1].max() - fv[:, 1].min()) * 1000:4.1f} of {FISH_H_REST * 1000:.0f} mm) "
              f"c=({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) touch {ncon} "
              f"| carriage {finger_cmd * 1000:.1f} mm")
    sim_t += 1.0 / FPS
    fin_np, fin_nn = fin_pos.numpy(), fin_nrm.numpy()
    fish_np, fish_nn = fish_pos.numpy(), fish_nrm.numpy()
    t3 = time.perf_counter()

    fin_geo.update_attribute("position", fin_np)
    fin_geo.update_attribute("normal", fin_nn)
    fish_geo.update_attribute("position", fish_np)
    fish_geo.update_attribute("normal", fish_nn)
    return t1 - t0, t2 - t1, t3 - t2, phase, c


step_frame.c_prev = np.array(FISH_P, dtype=np.float32)
step_frame.phase_prev = ""


def in_crate(c):
    return (abs(c[0] - CRATE_C[0]) < CRATE_HX and abs(c[2] - CRATE_C[1]) < CRATE_HZ
            and c[1] < CRATE_RIM)


# --- [A] cell + [F] present -----------------------------------------------------

canvas = tp.Canvas("threepp x warp - Franka FR3 soft gripper", width=WIDTH, height=HEIGHT,
                   antialiasing=4, headless=HEADLESS, vsync=False)
renderer = tp.VulkanRenderer(canvas)

scene = tp.Scene()
scene.background = 0x2b3138

camera = tp.PerspectiveCamera(42, canvas.aspect(), 0.05, 40)
camera.position.set(0.92, 0.44, 0.62)
camera.look_at(0.44, 0.10, 0.02)

scene.add(tp.HemisphereLight(0xdfeaff, 0x2a2f36, 0.55))
key = tp.DirectionalLight(0xfff2e0, 3.0)
key.position.set(1.6, 2.4, 1.2)
key.cast_shadow = True
scene.add(key)
fill = tp.DirectionalLight(0xbcd4ff, 1.1)
fill.position.set(-1.4, 1.6, -1.6)
scene.add(fill)

table = tp.Mesh(tp.BoxGeometry(2.2, 0.06, 1.6), standard_material(0x9aa3ac, 0.28, 0.9))
table.position.set(0.35, -0.03, 0.0)
table.receive_shadow = True
scene.add(table)

tray = tp.Mesh(tp.BoxGeometry(2 * TRAY_HX, TRAY_Y, 2 * TRAY_HZ),
               standard_material(0xb8c0c8, 0.22, 0.95))
tray.position.set(TRAY_C[0], TRAY_Y * 0.5, TRAY_C[1])
tray.receive_shadow = True
scene.add(tray)

crate_mat = standard_material(0x2f6fae, 0.55, 0.0)
crate_floor = tp.Mesh(tp.BoxGeometry(2 * CRATE_HX, CRATE_FLOOR, 2 * CRATE_HZ), crate_mat)
crate_floor.position.set(CRATE_C[0], CRATE_FLOOR * 0.5, CRATE_C[1])
scene.add(crate_floor)
for dx, dz, w, d in ((CRATE_HX, 0.0, 0.012, 2 * CRATE_HZ), (-CRATE_HX, 0.0, 0.012, 2 * CRATE_HZ),
                     (0.0, CRATE_HZ, 2 * CRATE_HX, 0.012), (0.0, -CRATE_HZ, 2 * CRATE_HX, 0.012)):
    wall = tp.Mesh(tp.BoxGeometry(w, CRATE_RIM, d), crate_mat)
    wall.position.set(CRATE_C[0] + dx, CRATE_RIM * 0.5, CRATE_C[1] + dz)
    wall.cast_shadow = True
    scene.add(wall)

backdrop = tp.Mesh(tp.PlaneGeometry(6, 3), standard_material(0x323a44, 0.95))
backdrop.position.set(-0.6, 1.0, -1.4)
scene.add(backdrop)

scene.add(robot)

fin_geo = tp.BufferGeometry()
fin_geo.set_attribute("position", p0[fin_tris_np])
fin_geo.set_attribute("normal", np.tile(np.array([0, 1, 0], np.float32), (fin_corners, 1)))
fingers_mesh = tp.Mesh(fin_geo, standard_material(0x2c3038, 0.55, 0.0))
fingers_mesh.cast_shadow = True
fingers_mesh.frustum_culled = False
scene.add(fingers_mesh)

fish_geo = tp.BufferGeometry()
fish_geo.set_attribute("position", (fish_tmpl + np.array(FISH_P, np.float32))[fish_tris_np])
fish_geo.set_attribute("normal", unit[fish_tris_np].astype(np.float32))
fish_mesh = tp.Mesh(fish_geo, standard_material(0xa9bcce, 0.38, 0.25))
fish_mesh.cast_shadow = True
fish_mesh.frustum_culled = False
scene.add(fish_mesh)

capture_substep()


def summary(c):
    ok = in_crate(c)
    mean = slip_mean / max(slip_n, 1)
    print(f"pick: grip {report.get('grip_N', grip_force):.2f} N at "
          f"{report.get('grip_open_mm', finger_cmd * 1000):.1f} mm carriage | "
          f"slip mean {mean * 1000:.1f} mm/s, peak {slip_peak * 1000:.1f} mm/s | "
          f"fish at ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) -> "
          f"{'IN THE CRATE' if ok else 'NOT in the crate'}")
    return ok


if SHOT:
    frames = int(round(SHOT_TIME * FPS))
    last = None
    for _ in range(frames):
        last = step_frame()[4]
    for _ in range(40):                    # let AE / TAA settle
        renderer.render(scene, camera)
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
        renderer.render(scene, camera)
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
        parts = step_frame()
        state["phase"] = parts[3]
        if parts[3] == "DONE" and not state["done"]:
            state["done"] = True
            summary(parts[4])
        controls.update()
        renderer.render(scene, camera)

    canvas.animate(animate)
