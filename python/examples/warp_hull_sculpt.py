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

ACT 2 -- VERIFY BY SIMULATION (key V, or --act2).
  The same integrals, run forward as a dynamics model with no tape at all: two
  bodies side by side in the FFT ocean, the original blob and the sculpted hull,
  each a 2-DOF rigid body (heave + roll) with buoyancy rho g V acting at B,
  weight at G, and linear damping. The local water plane is fitted from three
  `ocean.sample_height` probes, so a passing crest genuinely rolls them. Both
  get the same 25 degree kick. The blob goes over. The hull comes back.

    python warp_hull_sculpt.py               # window: watch it sculpt, then V
    python warp_hull_sculpt.py --act2        # optimise headless-fast, then act 2
    python warp_hull_sculpt.py --k 4         # 4 optimiser steps per frame
    python warp_hull_sculpt.py --tune        # no window at all: numbers only
    python warp_hull_sculpt.py --frames 900  # window, then quit on its own
    python warp_hull_sculpt.py --selftest    # headless acceptance run, exit code
    python warp_hull_sculpt.py --w-drag 7    # every weight has a flag
    python warp_hull_sculpt.py --no-interop  # the update_attribute fallback
    python warp_hull_sculpt.py --cpu         # CPU Warp; slow, same answer

Keys: SPACE pause/resume   R reset to the blob   1 / 2 bias drag / stability
      G ghost of the initial blob   V run act 2   Esc quit

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
from warp_common import (accum_normals, cli_arg, icosphere, normalize_vec3,
                         parse_size, resize_handler, signed_volume,
                         smooth_vec3_csr, standard_material, unique_edges,
                         vertex_adjacency)

# ---- flags -------------------------------------------------------------------
SELFTEST = "--selftest" in sys.argv
TUNE = "--tune" in sys.argv          # optimise only, no renderer, print numbers
ACT2_FIRST = "--act2" in sys.argv    # sculpt at full speed, then go straight to act 2
NO_INTEROP = "--no-interop" in sys.argv
FORCE_CPU = "--cpu" in sys.argv
STEPS = int(cli_arg("--steps", 1200 if SELFTEST else 2000, float))
K_STEPS = int(cli_arg("--k", 2, float))          # optimiser steps per rendered frame
LR = cli_arg("--lr", 0.006, float)               # Adam step, metres
SHOT_DIR = cli_arg("--shots", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "doc", "screenshots"), str)
WIN_W, WIN_H = parse_size(cli_arg("--size", "640x360" if SELFTEST else "1280x720", str))
ACT2_SECONDS = cli_arg("--act2-seconds", 24.0, float)
FRAME_BUDGET = int(cli_arg("--frames", 0, float))   # window: quit after N frames

HEADLESS = SELFTEST

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
SUBDIV = 4                       # icosphere(4) = 2562 vertices, 5120 faces
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
         box=cli_arg("--w-box", 600.0, float))
GRAD_SMOOTH = 3                  # Jacobi sweeps over the 1-ring, per step
GRAD_ALPHA = 0.65
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
              loss: wp.array(dtype=float)):
    """Everything above, combined into one scalar. Launched at dim=1.

    `scal` carries the non-differentiable normalisers so the weights stay O(1)
    across mesh resolutions and hull scales:
    (V0, area ref, length ref, 1/N verts, 1/N edges, heel count).
    """
    g = gmom[0] / vtot[0]
    stab = float(0.0)
    for a in range(1, NHEEL):
        b = bmom[a] / vsub[a]
        stab = stab - wp.dot(b - g, heel_b[a])
    dv = vtot[0] / scal[0] - 1.0
    loss[0] = (wts[0] * drag[0] / scal[1]
               + wts[1] * stab / (scal[2] * scal[5])
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
                out: wp.array(dtype=float)):
    """The same quantities as numbers for the HUD. Outside the tape, on purpose."""
    g = gmom[0] / vtot[0]
    out[0] = vtot[0]
    out[1] = drag[0]
    out[2] = reg[0]
    out[3] = reg[1]
    out[4] = boxp[0]
    for a in range(0, NHEEL):
        b = bmom[a] / vsub[a]
        out[5 + a] = wp.dot(b - g, heel_b[a])
        out[5 + NHEEL + a] = vsub[a]


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

        self.v_hull0 = signed_volume(self.rest, f)          # enclosed volume, m^3
        self.v_disp = DENSITY * self.v_hull0                # what it must displace
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
        self.reg = mk(2, float)
        self.boxp = mk(1, float)
        self.loss = mk(1, float)
        self.gsm = [wp.zeros(self.n_verts, dtype=wp.vec3, device=d) for _ in range(2)]

        self.wts = wp.zeros(6, dtype=float, device=d)
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
        self.push_weights()
        self.reset()

    # -- weights ------------------------------------------------------------
    def push_weights(self):
        self.wts.assign(np.array([W["drag"], W["stab"], W["disp"], W["lap"],
                                  W["edge"], W["box"]], np.float32))

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

    # -- one Adam step ------------------------------------------------------
    def forward(self, tape=None):
        """Zero the accumulators, run every integral, combine into loss[0]."""
        for a in (self.vsub, self.bmom, self.vtot, self.gmom, self.drag,
                  self.reg, self.boxp, self.loss):
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
        wp.launch(hull_fairness, dim=self.n_verts, device=dev,
                  inputs=[self.x, self.offsets, self.indices, self.reg])
        wp.launch(hull_edges, dim=self.n_edges, device=dev,
                  inputs=[self.x, self.e0, self.e1, self.rest_len, self.reg])
        wp.launch(hull_box, dim=self.n_verts, device=dev,
                  inputs=[self.x, wp.vec3(*BOX_HALF), self.boxp])
        wp.launch(hull_loss, dim=1, device=dev,
                  inputs=[self.vsub, self.bmom, self.vtot, self.gmom, self.drag,
                          self.reg, self.boxp, self.heel_b, self.wts, self.scal,
                          self.loss])

    def step(self):
        self.solve_drafts(wide=(self.step_count == 0))
        tape = wp.Tape()
        with tape:
            self.forward()
        tape.backward(loss=self.loss)
        # Precondition: a raw per-vertex gradient is dominated by the mesh's own
        # highest frequency, and Adam's per-element normalisation keeps it there.
        g = self.x.grad
        for i in range(GRAD_SMOOTH):
            src = g if i == 0 else self.gsm[(i - 1) % 2]
            wp.launch(smooth_vec3_csr, dim=self.n_verts, device=self.device,
                      inputs=[src, self.offsets, self.indices, GRAD_ALPHA,
                              self.gsm[i % 2]])
        self.opt.step([self.gsm[(GRAD_SMOOTH - 1) % 2] if GRAD_SMOOTH else g])
        tape.zero()
        self.step_count += 1

    # -- readout ------------------------------------------------------------
    def measure(self):
        """Forward-only evaluation at the solved drafts; returns a dict."""
        self.solve_drafts(wide=True)
        self.forward()
        wp.launch(hull_report, dim=1, device=self.device,
                  inputs=[self.vsub, self.bmom, self.vtot, self.gmom, self.drag,
                          self.reg, self.boxp, self.heel_b, self.report])
        r = self.report.numpy().astype(np.float64)
        gz = r[5:5 + NHEEL]
        parts = dict(
            drag=W["drag"] * r[1] / self.aref,
            stab=-W["stab"] * gz[1:].sum() / (self.lref * (NHEEL - 1)),
            disp=W["disp"] * (r[0] / self.v_hull0 - 1.0) ** 2,
            lap=W["lap"] * r[2] / (self.n_verts * self.lref ** 2),
            edge=W["edge"] * r[3] / self.n_edges,
            box=W["box"] * r[4] / (self.n_verts * self.lref ** 2))
        return dict(parts=parts, loss=float(self.loss.numpy()[0]),
                    v_tot=r[0], drag=r[1], lap=r[2], edge=r[3], box=r[4],
                    gz=gz, gz30=0.5 * (gz[GZ30] + gz[GZ30 + 1]),
                    v_sub=r[5 + NHEEL:5 + 2 * NHEEL],
                    v_err=abs(r[0] - self.v_hull0) / self.v_hull0,
                    draft=self._d_host.copy(), bbox=self.extents())

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
    """One 2-DOF (heave + roll) body, integrated against the soft integrals."""

    __slots__ = ("y", "vy", "phi", "om", "g_body", "mass", "inertia",
                 "c_heave", "c_roll", "peak", "name")

    def __init__(self, name, mass, inertia, g_body, beam):
        self.name = name
        self.mass = mass
        self.inertia = inertia
        self.g_body = np.asarray(g_body, np.float64)
        self.y = 0.0
        self.vy = 0.0
        self.phi = 0.0
        self.om = 0.0
        self.peak = 0.0
        # Linear damping, quoted as a fraction of critical for a nominal
        # stiffness -- a real hull's roll damping is 5-15% of critical and the
        # point here is that neither body is being held up by it.
        self.c_heave = 0.55 * 2.0 * math.sqrt(mass * RHO_W * GRAV * beam * beam)
        self.c_roll = 0.16 * 2.0 * math.sqrt(inertia * mass * GRAV * 0.6)


class Act2:
    """Two bodies, one sea, one kick each."""

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
        self.bodies = []
        for i, v in enumerate((verts_blob, verts_hull)):
            vol = signed_volume(v, sculpt.faces)
            g = _centroid(v, sculpt.faces)
            mass = RHO_W * DENSITY * vol
            beam = float(np.abs(v[:, 2]).max())
            k = 0.42 * beam * 2.0
            self.bodies.append(Floater(("blob", "hull")[i], mass, mass * k * k,
                                       g, beam))
        self.t = 0.0
        self.log = [[], []]

    def kick(self, deg=25.0):
        for b in self.bodies:
            b.phi = math.radians(deg)
            b.om = 0.0
            b.vy = 0.0
            b.y = 0.0
            b.peak = abs(b.phi)
        self.t = 0.0
        self.log = [[], []]

    def _water(self, x0, z0):
        """Local water plane: height plus a transverse slope from three probes."""
        if self.ocean is None:
            h = 0.55 * math.sin(0.62 * self.t) + 0.22 * math.sin(1.7 * self.t + 1.1)
            s = 0.055 * math.sin(0.62 * self.t + 0.9)
            return h, s
        dz = 2.5
        h0 = float(self.ocean.sample_height(x0, z0))
        hp = float(self.ocean.sample_height(x0, z0 + dz))
        hm = float(self.ocean.sample_height(x0, z0 - dz))
        return h0, (hp - hm) / (2.0 * dz)

    def step(self, dt, substeps=2):
        h = dt / substeps
        for _ in range(substeps):
            pn = np.zeros((2, 3), np.float32)
            pd = np.zeros(2, np.float32)
            slope = [0.0, 0.0]
            for i, b in enumerate(self.bodies):
                z0 = (i - 0.5) * self.spacing
                wh, ws = self._water(0.0, z0)
                slope[i] = ws
                inv = 1.0 / math.sqrt(1.0 + ws * ws)
                c, s = math.cos(b.phi), math.sin(b.phi)
                # R_x(phi)^T (0, 1, -ws), normalised.
                pn[i] = (0.0, (c - ws * s) * inv, (-s - ws * c) * inv)
                pd[i] = (wh - b.y) * inv
            self.plane_n.assign(pn)
            self.plane_d.assign(pd)
            self.vsub.zero_()
            self.bmom.zero_()
            wp.launch(rigid_buoyancy, dim=(2, self.s.n_faces), device=self.device,
                      inputs=[self.x, self.s.tris, self.plane_n, self.plane_d,
                              self.s.eps, self.vsub, self.bmom])
            v = self.vsub.numpy().astype(np.float64)
            m = self.bmom.numpy().astype(np.float64)
            for i, b in enumerate(self.bodies):
                vol = max(v[i], 1e-6)
                bc = m[i] / vol
                # GZ measured along the transverse horizontal of the CURRENT
                # heel -- the same expression the objective minimises.
                phi_w = b.phi + math.atan(slope[i])
                bvec = np.array([0.0, math.sin(phi_w), math.cos(phi_w)])
                gz = float(np.dot(bc - b.g_body, bvec))
                fz = RHO_W * GRAV * vol - b.mass * GRAV - b.c_heave * b.vy
                b.vy += h * fz / b.mass
                b.y += h * b.vy
                tq = -RHO_W * GRAV * vol * gz - b.c_roll * b.om
                b.om += h * tq / b.inertia
                b.phi += h * b.om
                b.peak = max(b.peak, abs(b.phi))
            self.t += h
        for i, b in enumerate(self.bodies):
            self.log[i].append(math.degrees(b.phi))

    def pose(self, i):
        b = self.bodies[i]
        return rot_x(b.phi), wp.vec3(0.0, float(b.y), 0.0)


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

    def __init__(self, verts, faces, material, device):
        self.device = device
        self.n = len(verts)
        g = tp.BufferGeometry()
        g.set_attribute("position", np.ascontiguousarray(verts, np.float32))
        nrm = np.zeros_like(verts, np.float32)
        nrm[:, 1] = 1.0
        g.set_attribute("normal", nrm)
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
    print(f"  {tag:<8} loss {m['loss']:9.5f}   drag {m['drag']:9.4f} m^2   "
          f"GZ(30) {m['gz30']:+7.4f} m   V {m['v_tot']:8.3f} m^3 "
          f"(err {100.0 * m['v_err']:.2f}%)   draft {m['draft'][0]:+.3f} m   "
          f"LxBxD {m['bbox'][0]:.2f} x {m['bbox'][2]:.2f} x {m['bbox'][1]:.2f} m")


def print_parts(m):
    print("           terms     " + "   ".join(f"{k} {v:.5f}"
                                               for k, v in m["parts"].items()))


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

    m0 = sculpt.measure()
    print("initial:")
    print_metrics("blob", m0)
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
    print(f"           GZ curve  {gz_curve(m1)}")
    print_parts(m1)
    print(f"  {STEPS} steps in {dt:.2f} s ({1000.0 * dt / STEPS:.2f} ms/step)")
    print(f"  GZ(30) {m0['gz30']:+.4f} -> {m1['gz30']:+.4f} m   "
          f"drag {m0['drag']:.4f} -> {m1['drag']:.4f} m^2 "
          f"({100.0 * (m1['drag'] / m0['drag'] - 1.0):+.1f}%)   "
          f"V err {100.0 * m1['v_err']:.2f}%")
    return 0


SUN_DIR = np.array([0.52, 0.75, 0.41], np.float32)
SUN_DIR /= np.linalg.norm(SUN_DIR)


def sky_env(w=512, h=256):
    """A procedural float equirect: background AND image-based light, no assets.

    Same trick vulkan_ocean.py uses to avoid shipping an .hdr, minus the RGBE
    round trip -- `float_texture` takes the linear array directly. The lower
    hemisphere fades to a haze rather than to black, or the ocean's grazing
    reflections drop out at the horizon.
    """
    elev = ((np.arange(h, dtype=np.float32) + 0.5) / h - 0.5) * math.pi
    az = ((np.arange(w, dtype=np.float32) + 0.5) / w - 0.5) * 2.0 * math.pi
    d = np.empty((h, w, 3), np.float32)
    d[..., 0] = np.cos(elev)[:, None] * np.cos(az)[None, :]
    d[..., 1] = np.sin(elev)[:, None]
    d[..., 2] = np.cos(elev)[:, None] * np.sin(az)[None, :]
    y = d[..., 1]
    up = np.clip(y, 0.0, 1.0)[..., None] ** 0.40
    down = np.clip(-y, 0.0, 1.0)[..., None] ** 0.60
    col = np.where(y[..., None] >= 0.0,
                   np.float32([0.60, 0.71, 0.90]) * (1.0 - up)
                   + np.float32([0.09, 0.23, 0.56]) * up,
                   np.float32([0.60, 0.71, 0.90]) * (1.0 - down)
                   + np.float32([0.15, 0.19, 0.25]) * down).astype(np.float32)
    col += (np.exp(-(y * y) / (2.0 * 0.0040))[..., None]
            * np.float32([0.36, 0.28, 0.20]))
    ang = np.arccos(np.clip(d @ SUN_DIR, -1.0, 1.0))
    col += ((np.exp(-(ang / math.radians(1.7)) ** 2) * 42.0
             + np.exp(-(ang / math.radians(12.0)) ** 2) * 2.6)[..., None]
            * np.float32([1.0, 0.96, 0.88]))
    out = np.ones((h, w, 4), np.float32)
    out[..., :3] = col
    return tp.float_texture(out)


def build_scene(canvas):
    renderer = tp.VulkanRenderer(canvas)
    renderer.tone_mapping = tp.ToneMapping.ACESFilmic
    renderer.tone_mapping_exposure = 0.80
    renderer.shadow_map_enabled = True

    scene = tp.Scene()
    env = sky_env()
    scene.background = env
    scene.environment = env
    sun = tp.DirectionalLight(0xfff3dc, 2.6)
    sun.position.set(*(SUN_DIR * 40.0))
    sun.cast_shadow = True
    scene.add(sun)

    ocean = tp.Ocean(size=600.0, wind_speed=7.0, choppiness=0.5, fetch=30e3)
    scene.add(ocean)

    floor = tp.Mesh(tp.PlaneGeometry(600.0, 600.0), standard_material(0x04101a))
    floor.rotate_x(-math.pi / 2)
    floor.position.y = -16.0
    scene.add(floor)

    camera = tp.PerspectiveCamera(42, canvas.aspect(), 0.1, 2000)
    camera.position.set(10.5, 3.4, 10.5)
    camera.look_at(0.0, 0.2, 0.0)
    return renderer, scene, camera, ocean


def run_scene(sculpt, m0, device):
    canvas = tp.Canvas("threepp - differentiable hull sculpting",
                       width=WIN_W, height=WIN_H, vsync=False, headless=HEADLESS)
    renderer, scene, camera, ocean = build_scene(canvas)

    hull_mat = standard_material(0xdde4ec, roughness=0.35, metalness=0.08)
    ghost_mat = standard_material(0x64d0ff, roughness=0.5, metalness=0.0)
    ghost_mat.transparent = True
    ghost_mat.opacity = 0.22

    live = LiveMesh(sculpt.rest, sculpt.faces, hull_mat, device)
    live.mesh.position.set(0.0, 0.0, 0.0)
    scene.add(live.mesh)

    # The ghost of the initial blob doubles as act 2's second body. It has to be
    # VISIBLE for the arming render -- a mesh the renderer has never drawn has no
    # BLAS record, and so nothing to export.
    blob_mesh = LiveMesh(sculpt.rest, sculpt.faces, ghost_mat, device)
    blob_mesh.mesh.position.set(0.0, 0.0, 0.0)
    scene.add(blob_mesh.mesh)

    SPACING = 11.0
    act2 = None
    ui = tp.ImguiContext(canvas, renderer) if (tp.HAS_IMGUI and not HEADLESS) else None

    state = dict(mode="sculpt", paused=False, ghost=False, step_ms=0.0,
                 render_ms=0.0, m=m0, m_final=None, frames=0)

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
            mesh.publish(sculpt.tris, sculpt.n_faces)

    # The mesh's renderer record only exists once it has been drawn.
    publish_sculpt()
    publish_ghost()
    renderer.render(scene, camera)
    live.arm(renderer)
    blob_mesh.arm(renderer)
    blob_mesh.mesh.visible = False
    print(f"vertex route: {live.route} / {blob_mesh.route}")

    def enter_act2():
        nonlocal act2
        state["m_final"] = sculpt.measure()
        blob = (sculpt.rest - np.array([0.0, m0["draft"][0], 0.0], np.float32))
        hull = (sculpt.positions()
                - np.array([0.0, state["m_final"]["draft"][0], 0.0], np.float32))
        act2 = Act2(sculpt, blob, hull, ocean, SPACING)
        act2.kick(25.0)
        # The ghost stops being a ghost: same Material object, opaque now, so
        # nothing structural changes under the live interop registration.
        ghost_mat.transparent = False
        ghost_mat.opacity = 1.0
        ghost_mat.color = 0xdd8a6a
        blob_mesh.mesh.visible = True
        blob_mesh.mesh.position.set(0.0, 0.0, -0.5 * SPACING)
        live.mesh.position.set(0.0, 0.0, +0.5 * SPACING)
        camera.position.set(19.0, 5.6, 0.0)
        camera.look_at(0.0, 0.4, 0.0)
        state["mode"] = "act2"
        print("\nact 2: both bodies kicked to 25 deg roll")

    def step_sim(dt):
        t0 = time.perf_counter()
        if state["mode"] == "sculpt":
            if not state["paused"] and sculpt.step_count < STEPS:
                for _ in range(K_STEPS):
                    sculpt.step()
                state["m"] = sculpt.measure() if sculpt.step_count % 10 == 0 \
                    else state["m"]
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
            tp.imgui.text(f"step {sculpt.step_count} / {STEPS}"
                          f"{'  [paused]' if state['paused'] else ''}")
            tp.imgui.text(f"loss        {m['loss']:.5f}")
            tp.imgui.text(f"drag        {m['drag']:.4f} m^2")
            tp.imgui.text(f"GZ(30 deg)  {m['gz30']:+.4f} m")
            tp.imgui.text(f"V           {m['v_tot']:.2f} m^3 "
                          f"({100.0 * m['v_err']:+.2f}%)")
            tp.imgui.text(f"draft       {m['draft'][0]:+.3f} m")
            tp.imgui.separator()
            tp.imgui.text(f"w_drag {W['drag']:.2f}   w_stab {W['stab']:.2f}"
                          "   (keys 1 / 2)")
        else:
            for i, b in enumerate(act2.bodies):
                tp.imgui.text(f"{b.name:<5} roll {math.degrees(b.phi):+7.1f} deg"
                              f"   peak {math.degrees(b.peak):6.1f}")
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
                          publish_sculpt, enter_act2, publish_act2,
                          lambda: act2)
        live.release(renderer)
        blob_mesh.release(renderer)
        return rc

    if ACT2_FIRST:
        print(f"optimising {STEPS} steps...")
        t0 = time.perf_counter()
        for _ in range(STEPS):
            sculpt.step()
        print(f"  {time.perf_counter() - t0:.1f} s")
        state["m"] = sculpt.measure()
        publish_sculpt()
        renderer.render(scene, camera)
        enter_act2()

    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.max_distance = 220.0
    canvas.on_window_resize(resize_handler(camera, renderer))
    clock = tp.Clock()

    def animate():
        dt = min(clock.get_delta(), 0.1)
        handle_keys()
        if ui is not None:
            controls.enabled = not ui.want_capture_mouse
        step_sim(max(dt, 1.0 / 240.0))
        controls.update()
        ocean.warp_toward(controls.target.x, controls.target.z, 0.3)
        t0 = time.perf_counter()
        renderer.render(scene, camera)
        state["render_ms"] = 1000.0 * (time.perf_counter() - t0)
        if ui is not None:
            ui.render(draw_ui)
        state["frames"] += 1

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
        live.release(renderer)
        blob_mesh.release(renderer)
    return 0


# --------------------------------------------------------------------------- #
#  --selftest: the acceptance run
# --------------------------------------------------------------------------- #
def run_selftest(sculpt, m0, renderer, scene, camera,
                 publish_sculpt, enter_act2, publish_act2, get_act2):
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
            losses.append(m["loss"])
            if not np.isfinite(m["loss"]):
                fails.append(f"non-finite loss at step {i + 1}")
                break
            print(f"    step {i + 1:5d}  loss {m['loss']:9.5f}  "
                  f"GZ(30) {m['gz30']:+.4f}  drag {m['drag']:8.4f}  "
                  f"V err {100.0 * m['v_err']:.2f}%")
    n = frames
    print(f"    per rendered FRAME at k={K_STEPS}: optimiser "
          f"{1000.0 * t_opt / n:.2f} ms ({1000.0 * t_opt / (n * K_STEPS):.2f} "
          f"ms/step) | publish {1000.0 * t_pub / n:.2f} ms | render "
          f"{1000.0 * t_ren / n:.2f} ms = {1000.0 * (t_opt + t_pub + t_ren) / n:.2f}"
          f" ms/frame ({n / max(t_opt + t_pub + t_ren, 1e-9):.0f} fps)")
    if losses and not (losses[-1] < losses[0]):
        fails.append(f"loss did not decrease ({losses[0]:.5f} -> {losses[-1]:.5f})")

    m1 = sculpt.measure()
    print("\n[2] quantitative improvement")
    print_metrics("blob", m0)
    print_metrics("hull", m1)
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

    print()
    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        return 1
    print("selftest PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
