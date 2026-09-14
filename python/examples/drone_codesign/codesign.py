"""Sea-drone co-design, step 3: the hull and the screw in ONE tape.

    python python/examples/drone_codesign/codesign.py --selftest
    python python/examples/drone_codesign/codesign.py --shots out/shots

WHAT IS NEW HERE. The pilot (`optimize.py`) optimised a propeller against a
hull that was three numbers in a dict. The hull sculpt (`../warp_hull_sculpt.py`)
optimised a hull against a thrust that was one number on the command line.
Neither knew the other existed. This file puts both on the same `wp.Tape`:

  * the hull's form-drag integral, its wetted area and its Michell wave
    resistance -- evaluated at the BRIEF's two speeds, not at the sculpt's own
    cruise -- become the R_T the screw's self-propulsion Newton solves against,
    so a fairer hull is a longer-ranged drone through the propeller, not
    through a proxy;
  * the upright waterline plane the sculpt already solves for gives the shaft
    its immersion h, so the cavitation number is the one this hull's draft
    actually provides;
  * a differentiable tip clearance between the disc's top point and the hull
    surface lets the screw ASK FOR ROOM: the hinge pushes on the hull's
    vertices near the afterbody, which is the one thing a fixed hull cannot do.

THE CLEARANCE GRADIENT. `wp.mesh_query_point` returns a face index, two
barycentrics and a sign, and none of those is differentiated -- Warp's mesh
queries are lookups, not functions of the points. So a SECOND kernel
recomputes the closest point from that face's three vertices,
`p = a u + b v + c (1-u-v)` (Warp's own barycentric convention, the one
`mesh_eval_position` uses -- getting it backwards silently returns a different
vertex), and the distance to it. Holding the face and the barycentrics fixed is
the standard envelope-theorem move: at the base point the recomputed distance
IS the clearance, and its gradient is the clearance's gradient.

PRECISION, AND WHY A GATE RUNS ON THE CPU. The hull's integrals are float32
atomics over 5120 faces. On CUDA the summation ORDER varies between launches,
so two nominally identical forwards differ by ~1e-6 relative -- which is
nothing for a gradient and fatal for a finite difference, where it is divided
by 2h. Gate 1 therefore freezes the hull and differences only the float64
propeller half; gate 2, which must difference the hull itself, runs on the CPU
device, where Warp's launches are a serial loop and the two forwards agree to
the bit. Both gates print the repeatability they measured.

Gates (`--selftest`, exit 0/2):

  1  FD on the four screw variables, hull frozen.
  2  FD on hull vertices, draft plane held -- the gate that says the coupling
     kernels (two more wave-resistance speeds, clearance, immersion) are right.
  3  frozen-hull regression: the frozen hull's two R_T numbers go into a brief
     the pilot's brute-force grid can read, and the coupled Adam has to land on
     that grid's optimum inside the pilot's own tolerances.
  4  co-design vs frozen hull from the same start, as a table.
  5  matplotlib 3-views, the npz and the design JSONs.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.dirname(_HERE)                 # python/examples
_PY = os.path.dirname(_EX)                   # python
for _p in (_HERE, _EX, _PY):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── our own flags, parsed BEFORE the sculpt sees sys.argv ──────────────────
def parse_args():
    ap = argparse.ArgumentParser(description="hull + screw in one tape")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--freeze-hull", action="store_true",
                    help="optimise the screw only; the hull is a constant")
    ap.add_argument("--steps", type=int, default=-1,
                    help="-1 = the sculpt's own default (2000)")
    ap.add_argument("--subdiv", type=int, default=4)
    ap.add_argument("--blob-scale", type=float, default=0.42)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--w-prop", type=float, default=-1.0,
                    help="-1 = match the hull loss at step 0")
    ap.add_argument("--shots", default="")
    ap.add_argument("--device", default=None)
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


ARGS = parse_args()

# ── THE BRIEF ──────────────────────────────────────────────────────────────
# A ~3 m survey drone. The speeds are LOW because the body is not: the sculpt's
# blob is a 3.1 t brick whose Newton pressure proxy puts a kilonewton of form
# drag on it at 1 m/s, and its Michell hump sits just past 2 m/s. Picking a
# 7 m/s sprint (the pilot's, against a placeholder hull with C_R 0.006) would
# put the whole run inside a cavitation penalty no diameter can pay off, which
# measures the hinge weight and nothing else.
#
# D_max is 0.70 m, not the pilot's 0.25: the point of this step is that the
# CLEARANCE, not a box bound, is what limits the diameter -- a heavily loaded
# slow hull wants the biggest, slowest-turning screw it can be given room for,
# and the room is the hull's to give.
BRIEF = dict(
    V_survey=1.0, V_sprint=1.8,
    w=0.15, t=0.12,
    E_batt=5.0,                  # kWh
    n_max=50.0,                  # rps at SPRINT
    D_max=0.70,                  # m
    depth_min=0.15, depth_max=1.40,   # a guard; the geometry sets h
    lam_cav=100.0, lam_n=100.0, lam_box=100.0,
    payload_frac=0.15,           # the sculpt's --cargo-frac
    cargo_y=0.50,                # m above the waterline
    gz_floor=0.10,               # m, absolute: a 3 m boat, not a 7 m one
    # The shaft lives in the hull frame, z = 0, single screw.
    x_lo=-1.30, x_hi=-0.60,      # aft, and NOT past the stern
    y_lo=-0.95, y_hi=-0.20,      # above the blob's keel (-1.008 m). y_hi is
                                 # high enough that a big disc up there breaks
                                 # the immersion hinge -- a guard that can
                                 # never fire is a guard nothing tests.
    clear_frac=0.20,             # tip clearance >= 0.20 D
    imm_margin=0.05,             # whole disc >= 0.05 m under the surface
    hull=dict(L=3.0, S_wet=8.5, k_form=0.25, C_R=0.0060),   # pilot fallback
)
STEP0 = dict(D=0.40, PD=0.80, x_s=-0.95, y_s=-0.625)

# ── hand the sculpt its own argv, then import it ───────────────────────────
_SCULPT_STEPS = ARGS.steps if ARGS.steps > 0 else 2000
sys.argv = [sys.argv[0], "--tune",
            "--subdiv", str(ARGS.subdiv),
            "--blob-scale", f"{ARGS.blob_scale:.6g}",
            "--cargo-frac", f"{BRIEF['payload_frac']:.6g}",
            "--cargo-y", f"{BRIEF['cargo_y']:.6g}",
            "--gz-floor", f"{BRIEF['gz_floor']:.6g}",
            "--u-ref", f"{BRIEF['V_survey']:.6g}",
            "--steps", str(_SCULPT_STEPS)]

import warp as wp            # noqa: E402
import warp.optim            # noqa: E402

import warp_hull_sculpt as hs    # noqa: E402
import optimize                  # noqa: E402
import prop                      # noqa: E402

F = wp.float64
STEPS = hs.STEPS
GATE3_STEPS = 4000       # see gate3_regression: a wide box and a curved ridge
GATE3_LR = 0.03


# ── the coupling kernels ───────────────────────────────────────────────────
# All of them are dim=1 (one design, one screw), so their float32 atomics have
# nothing to race with and their forward is bit-reproducible on any device.

@wp.kernel
def k_shaft(theta: wp.array(dtype=float), ptop: wp.array(dtype=wp.vec3)):
    """The disc's top point, (x_s, y_s + D/2, 0), as a DIFFERENTIABLE vec3.

    theta here is the float32 twin of the design vector: the hull half of the
    graph is float32 and the propeller half is float64, and rather than cast a
    parameter back and forth the two halves each read their own copy of the
    same four numbers. The host keeps them equal and adds the two gradients --
    which is just the chain rule for a variable that enters a loss twice.
    """
    ptop[0] = wp.vec3(theta[2], theta[3] + 0.5 * theta[0], 0.0)


@wp.kernel
def k_query(mesh: wp.uint64, p: wp.array(dtype=wp.vec3), max_d: float,
            face: wp.array(dtype=wp.int32), uv: wp.array(dtype=wp.vec2),
            sgn: wp.array(dtype=float)):
    """The lookup half of the clearance: WHICH face, and inside or outside.

    Never taped -- `mesh_query_point`'s outputs carry no adjoint. It is run
    before the tape opens, off the host's own copy of the shaft point.
    """
    q = wp.mesh_query_point(mesh, p[0], max_d)
    if q.result:
        face[0] = q.face
        uv[0] = wp.vec2(q.u, q.v)
        sgn[0] = q.sign
    else:
        face[0] = wp.int32(-1)
        uv[0] = wp.vec2(0.0, 0.0)
        sgn[0] = 1.0


@wp.kernel
def k_clear(x: wp.array(dtype=wp.vec3), tris: wp.array(dtype=int),
            face: wp.array(dtype=wp.int32), uv: wp.array(dtype=wp.vec2),
            sgn: wp.array(dtype=float), p: wp.array(dtype=wp.vec3),
            out: wp.array(dtype=float)):
    """... and the differentiable half: rebuild the closest point from the
    face's three VERTICES, so the hull can move to make room.

    `a u + b v + c (1-u-v)` is Warp's barycentric convention (mesh_eval_position).
    The sign comes from the query: inside the hull is negative clearance, which
    is what the hinge has to see when the disc is buried in the afterbody.
    """
    f = face[0]
    if f < 0:
        out[0] = 1.0e6
        return
    a = x[tris[f * 3 + 0]]
    b = x[tris[f * 3 + 1]]
    c = x[tris[f * 3 + 2]]
    u = uv[0][0]
    v = uv[0][1]
    cp = a * u + b * v + c * (1.0 - u - v)
    d = p[0] - cp
    s = 1.0
    if sgn[0] < 0.0:
        s = -1.0
    # +eps under the root, not |.|: at zero distance the derivative of a bare
    # length is undefined and the tape would hand back a NaN.
    out[0] = s * wp.sqrt(wp.dot(d, d) + 1.0e-12)


@wp.kernel
def k_resist(drag: wp.array(dtype=float), area: wp.array(dtype=float),
             rw_s: wp.array(dtype=float), rw_p: wp.array(dtype=float),
             c_form_s: F, c_fric_s: F, c_form_p: F, c_fric_p: F,
             rt: wp.array(dtype=F)):
    """R_T(V) = rho V^2 s_form + 0.5 rho V^2 C_F(V, Lwl) S_wet + R_wave(V),
    at the survey speed (slot 0) and the sprint speed (slot 1).

    The four coefficients are folded on the host -- C_F needs Lwl, which is a
    detached per-step constant (a bounding-box measurement, not a function the
    tape may differentiate), and a composite constant assembled inside a kernel
    is the overload trap prop.py's header warns about.
    """
    rt[0] = (wp.float64(drag[0]) * c_form_s + wp.float64(area[0]) * c_fric_s
             + wp.float64(rw_s[0]))
    rt[1] = (wp.float64(drag[0]) * c_form_p + wp.float64(area[0]) * c_fric_p
             + wp.float64(rw_p[0]))


@wp.kernel
def k_design(theta: wp.array(dtype=F), plane_n: wp.array(dtype=wp.vec3),
             plane_d: wp.array(dtype=float), design: wp.array2d(dtype=F)):
    """(D, P/D, x_s, y_s) -> the pilot's design row (D, P/D, h).

    h is the signed distance from the shaft point to the UPRIGHT waterline
    plane n.x = d, positive below the surface. The plane itself is detached --
    the sculpt solves the draft outside its own tape and this file keeps that
    bargain -- so the gradient that reaches y_s through h is d(h)/d(y_s) = -n_y
    and nothing else.
    """
    d = theta[0]
    design[0, 0] = d
    design[0, 1] = theta[1]
    nx = wp.float64(plane_n[0][0])
    ny = wp.float64(plane_n[0][1])
    design[0, 2] = wp.float64(plane_d[0]) - (nx * theta[2] + ny * theta[3])


@wp.kernel
def k_geo(theta: wp.array(dtype=F), clear: wp.array(dtype=float),
          design: wp.array2d(dtype=F), frac: F, marg: F, lam: F,
          out: wp.array(dtype=F)):
    """The two GEOMETRIC hinges, squared, at the pilot's lam_box.

    clearance c >= frac D   -- the tip must not be in the hull;
    h - D/2   >= marg       -- and the whole disc must be under water at trim.
    Both are in metres, so lam 100 means "a centimetre of violation costs 0.01
    of objective", which is the same currency the pilot's box hinges use.
    """
    d = theta[0]
    hc = wp.max(frac * d - wp.float64(clear[0]), wp.float64(0.0))
    hi = wp.max(marg - (design[0, 2] - wp.float64(0.5) * d), wp.float64(0.0))
    out[0] = lam * (hc * hc + hi * hi)


@wp.kernel
def k_total(hull: wp.array(dtype=float), prop_l: wp.array(dtype=F),
            geo: wp.array(dtype=F), w_prop: F, out: wp.array(dtype=F)):
    """L = L_hull + w_prop (L_prop + L_geom).

    The sculpt's own --w-drag term STAYS. It is a form-drag area integral, a
    cheap shape-hungry proxy that works from step one; the range term is the
    physical statement (watts at the shaft, kilometres on a battery) and it
    only has teeth once the propeller solve is converged and inside its box.
    They are not the same term and neither is redundant -- see the report.
    """
    out[0] = wp.float64(hull[0]) + w_prop * (prop_l[0] + geo[0])


# ── the sculpt, set up exactly as its own main() does ───────────────────────
def build_sculpt(device, subdiv=None, verbose=True):
    """A Sculptor plus the calibration its resistance law needs.

    `subdiv` overrides the module-level SUBDIV for this one construction. That
    global is read once, in `Sculptor.__init__`, and gate 2 needs a coarser
    mesh than gate 4 in the same process; the sculpt file itself is untouched.
    """
    keep = hs.SUBDIV
    if subdiv is not None:
        hs.SUBDIV = int(subdiv)
    try:
        s = hs.Sculptor(device)
    finally:
        hs.SUBDIV = keep
    # Re-calibrate per Sculptor: both globals latch on first use.
    hs.W_WAVE["v"] = -1.0
    hs.THRUST["N"] = -1.0
    s.forward()
    s.lwl = max(float(s.extents()[0]), 1.0)
    hs.calibrate_wave(s)
    hs.set_thrust(s)
    s.refresh_cruise()
    if verbose:
        print(f"  hull    {s.n_verts} verts / {s.n_faces} faces, "
              f"L {2.0 * hs.BLOB[0]:.2f} m, V {s.v_hull0:.3f} m3, "
              f"displaces {s.mass / 1000.0:.2f} t, waterline eps {s.eps:.3f} m")
        print(f"          wave scale {hs.W_WAVE['v']:.4g}, sculpt thrust "
              f"{hs.THRUST['N']:.0f} N, u_run {s.u_run:.3f} m/s, "
              f"Lwl {s.lwl:.3f} m")
    return s


# ── the coupled step ───────────────────────────────────────────────────────
class CoDesign:
    """Hull vertices and (D, P/D, x_s, y_s) descending one scalar together."""

    def __init__(self, sculpt, brief, device, w_prop=1.0, w_geo=1.0,
                 lr=0.03, steps=None):
        self.s = sculpt
        self.brief = brief
        self.dev = device
        self.w_prop = float(w_prop)
        self.w_geo = float(w_geo)       # gate 3 turns the geometric hinges off
        self.lr = float(lr)
        self.steps = STEPS if steps is None else int(steps)
        d = device

        self.buf = prop.Buffers(1, device=d, requires_grad=True)
        self.rt = wp.zeros(2, dtype=F, device=d, requires_grad=True)
        self.geo = wp.zeros(1, dtype=F, device=d, requires_grad=True)
        self.total = wp.zeros(1, dtype=F, device=d, requires_grad=True)
        # The design vector, twice: float64 for the screw, float32 for the hull.
        self.th64 = wp.zeros(4, dtype=F, device=d, requires_grad=True)
        self.th32 = wp.zeros(4, dtype=float, device=d, requires_grad=True)
        # Two more Michell accumulators: the sculpt's own pair is busy at
        # u_run, and `wave_resistance()` clobbers it.
        self.wamp_s = wp.zeros((hs.NTHETA, 2), dtype=float, device=d,
                               requires_grad=True)
        self.wamp_p = wp.zeros((hs.NTHETA, 2), dtype=float, device=d,
                               requires_grad=True)
        self.rw_s = wp.zeros(1, dtype=float, device=d, requires_grad=True)
        self.rw_p = wp.zeros(1, dtype=float, device=d, requires_grad=True)
        # Clearance.
        self.mesh = wp.Mesh(points=sculpt.x, indices=sculpt.tris)
        self.pq = wp.zeros(1, dtype=wp.vec3, device=d)          # host-set, detached
        self.ptop = wp.zeros(1, dtype=wp.vec3, device=d, requires_grad=True)
        self.qface = wp.zeros(1, dtype=wp.int32, device=d)
        self.quv = wp.zeros(1, dtype=wp.vec2, device=d)
        self.qsgn = wp.zeros(1, dtype=float, device=d)
        self.clear = wp.zeros(1, dtype=float, device=d, requires_grad=True)
        self.hull_const = wp.zeros(1, dtype=float, device=d, requires_grad=True)
        # A permanent zero in the hull slot, so the same k_total can be asked
        # for the COUPLING half on its own (gate 2's seed).
        self.zero32 = wp.zeros(1, dtype=float, device=d, requires_grad=True)
        self.coupl = wp.zeros(1, dtype=F, device=d, requires_grad=True)

        # Normalised screw variables, exactly the pilot's scheme.
        self.scale = np.array([brief["D_max"], 1.0,
                               max(abs(brief["x_lo"]), abs(brief["x_hi"])),
                               max(abs(brief["y_lo"]), abs(brief["y_hi"]))])
        self.lo = np.array([prop.D_MIN / self.scale[0], prop.PD_LO,
                            brief["x_lo"] / self.scale[2],
                            brief["y_lo"] / self.scale[3]])
        self.hi = np.array([1.0, prop.PD_HI,
                            brief["x_hi"] / self.scale[2],
                            brief["y_hi"] / self.scale[3]])
        self.u = np.array([STEP0["D"], STEP0["PD"], STEP0["x_s"],
                           STEP0["y_s"]]) / self.scale
        self.u_wp = wp.array(self.u.astype(np.float32), dtype=wp.float32,
                             device=d)
        self.g_wp = wp.zeros(4, dtype=wp.float32, device=d)
        self.adam = wp.optim.Adam([self.u_wp], lr=self.lr)
        self.it = 0
        self.frozen = False
        self._rt_host = None
        self._frozen_gz30 = float("nan")
        self._frozen_parts = {}
        self._frozen_lwl = float("nan")
        self.lwl_hold = None

    # -- helpers ------------------------------------------------------------
    def design(self):
        return np.clip(self.u, self.lo, self.hi) * self.scale

    def set_design(self, phys):
        self.u = np.asarray(phys, dtype=np.float64) / self.scale
        self.u_wp.assign(self.u.astype(np.float32))

    def _push_theta(self):
        p = self.design()
        self.th64.assign(np.ascontiguousarray(p, dtype=np.float64))
        self.th32.assign(np.ascontiguousarray(p, dtype=np.float32))
        return p

    def _query(self, phys):
        """The detached closest-face lookup, off the host's own shaft point."""
        self.pq.assign(np.array([[phys[2], phys[3] + 0.5 * phys[0], 0.0]],
                                np.float32))
        self.mesh.refit()
        wp.launch(k_query, dim=1, device=self.dev,
                  inputs=[self.mesh.id, self.pq, 1.0e6],
                  outputs=[self.qface, self.quv, self.qsgn])

    def hull_forward(self, dyn=False, taped=True):
        """The sculpt's own forward, plus the two extra Michell speeds and the
        resistance bridge. Called inside the tape when the hull is a variable,
        once outside it when the hull is frozen."""
        s, dev, b = self.s, self.dev, self.brief
        s.forward(dyn)
        for a in (self.wamp_s, self.wamp_p, self.rw_s, self.rw_p):
            a.zero_()
        if hs.W_WAVE["v"] > 0.0:
            for v, wamp, rw in ((b["V_survey"], self.wamp_s, self.rw_s),
                                (b["V_sprint"], self.wamp_p, self.rw_p)):
                k0 = hs.GRAV / max(v * v, 1.0e-4)
                wp.launch(hs.wave_amp, dim=(hs.NTHETA, s.n_faces), device=dev,
                          inputs=[s.x, s.tris, s.plane_d, s.eps, k0, s.sec,
                                  wamp])
                wp.launch(hs.wave_resist, dim=hs.NTHETA, device=dev,
                          inputs=[wamp, s.sec3dt, s._wave_coef(v), rw])
        # Lwl is a DETACHED host constant per step (a bounding-box reading, and
        # the tape is never asked to differentiate a max over vertices). A
        # finite difference therefore has to hold it too, or moving the bow
        # vertex moves C_F under the difference and the check is measuring a
        # path the adjoint was never meant to have. `lwl_hold` is what a gate
        # sets; the optimiser leaves it None and re-reads it every step.
        lwl = self.lwl_hold or max(float(s.extents()[0]), 1.0)
        vs, vp = b["V_survey"], b["V_sprint"]
        wp.launch(k_resist, dim=1, device=dev,
                  inputs=[s.drag, s.area, self.rw_s, self.rw_p,
                          hs.RHO_W * vs * vs,
                          0.5 * hs.RHO_W * vs * vs * hs.cf_ittc(vs, lwl),
                          hs.RHO_W * vp * vp,
                          0.5 * hs.RHO_W * vp * vp * hs.cf_ittc(vp, lwl)],
                  outputs=[self.rt])

    def screw_forward(self):
        """Everything downstream of the hull: shaft point, clearance, the
        pilot's chain against the ARRAY resistance, and the two hinges."""
        dev, b = self.dev, self.brief
        wp.launch(k_shaft, dim=1, device=dev, inputs=[self.th32],
                  outputs=[self.ptop])
        wp.launch(k_clear, dim=1, device=dev,
                  inputs=[self.s.x, self.s.tris, self.qface, self.quv,
                          self.qsgn, self.ptop],
                  outputs=[self.clear])
        wp.launch(k_design, dim=1, device=dev,
                  inputs=[self.th64, self.s.plane_n, self.s.plane_d],
                  outputs=[self.buf.design])
        prop._chain_body(self.buf, b, rt_arr=self.rt)
        self.geo.zero_()
        wp.launch(k_geo, dim=1, device=dev,
                  inputs=[self.th64, self.clear, self.buf.design,
                          b["clear_frac"], b["imm_margin"],
                          b["lam_box"] * self.w_geo],
                  outputs=[self.geo])
        wp.launch(k_total, dim=1, device=dev,
                  inputs=[self.hull_const if self.frozen else self.s.loss,
                          self.buf.loss, self.geo, self.w_prop],
                  outputs=[self.total])
        wp.launch(k_total, dim=1, device=dev,
                  inputs=[self.zero32, self.buf.loss, self.geo, self.w_prop],
                  outputs=[self.coupl])

    def freeze(self, dyn=False):
        """Run the hull once, keep its R_T and its loss as constants."""
        self.s.solve_drafts(wide=True)
        self.hull_forward(dyn=dyn)
        self._rt_host = self.rt.numpy().copy()
        self.hull_const.assign(self.s.loss.numpy().copy())
        self.frozen = True

    # -- one coupled step ---------------------------------------------------
    def step(self, dyn=None):
        s = self.s
        if dyn is None:
            dyn = s.in_dyn()
        if not self.frozen:
            if dyn != s._dyn_pushed:
                s.push_weights(dyn)
            s.solve_drafts(wide=(self.it == 0))
        phys = self._push_theta()
        self._query(phys)
        tape = wp.Tape()
        with tape:
            if not self.frozen:
                self.hull_forward(dyn=dyn)
            self.screw_forward()
        tape.backward(loss=self.total)

        if not self.frozen:
            # The sculpt's own twelve lines, copied rather than called: step()
            # owns the tape and we own this one.
            g = s.x.grad
            ns = hs.GRAD_SMOOTH_DYN if dyn else hs.GRAD_SMOOTH
            for i in range(ns):
                src = g if i == 0 else s.gsm[(i - 1) % 2]
                wp.launch(hs.smooth_vec3_csr, dim=s.n_verts, device=self.dev,
                          inputs=[src, s.offsets, s.indices, hs.GRAD_ALPHA,
                                  s.gsm[i % 2]])
            s.opt.step([s.gsm[(ns - 1) % 2] if ns else g])
            if not hs.NO_SYM:
                wp.launch(hs.symmetrize, dim=s.n_verts, device=self.dev,
                          inputs=[s.x, s.pair, s.xsym])
                wp.copy(s.x, s.xsym)
            s.step_count += 1

        g4 = (self.th64.grad.numpy().astype(np.float64)
              + self.th32.grad.numpy().astype(np.float64))
        self.g_wp.assign((g4 * self.scale).astype(np.float32))
        self.adam.lr = self.lr * (0.05 + 0.95 * 0.5 *
                                  (1.0 + math.cos(math.pi * self.it /
                                                  max(self.steps, 1))))
        self.adam.step([self.g_wp])
        self.u = np.clip(self.u_wp.numpy().astype(np.float64), self.lo, self.hi)
        self.u_wp.assign(self.u.astype(np.float32))
        tape.zero()
        self.it += 1
        if (not self.frozen and dyn
                and s.step_count % hs.UREF_EVERY == 0):
            s.refresh_cruise()
        return g4

    def evaluate(self, dyn=None):
        """Forward only, no tape -- the numbers a report prints."""
        s = self.s
        if dyn is None:
            dyn = s.in_dyn()
        phys = self._push_theta()
        self._query(phys)
        if not self.frozen:
            self.hull_forward(dyn=dyn)
        self.screw_forward()
        return phys

    # -- readout ------------------------------------------------------------
    def snapshot(self, tag, dyn=None, full=False):
        phys = self.evaluate(dyn=dyn)
        r = optimize.row_of(prop.read_rows(self.buf), 0)
        rt = self.rt.numpy().copy()
        out = dict(
            tag=tag, D=phys[0], PD=phys[1], x_s=phys[2], y_s=phys[3],
            clearance=float(self.clear.numpy()[0]),
            clear_req=self.brief["clear_frac"] * phys[0],
            draft=float(self.s._d_host[0]),
            depth=r["depth"], range_km=r["range_km"], P_survey=r["P_survey"],
            n_survey=r["n_survey"], n_sprint=r["n_sprint"],
            eta_survey=r["eta_survey"], J_survey=r["J_survey"],
            ratio_sprint=r["ratio_sprint"], ratio_survey=r["ratio_survey"],
            res=max(r["res_survey"], r["res_sprint"]),
            s_form=float(self.s.drag.numpy()[0]),
            s_wet=float(self.s.area.numpy()[0]),
            rt_survey=float(rt[0]), rt_sprint=float(rt[1]),
            rwave_survey=float(self.rw_s.numpy()[0]),
            rwave_sprint=float(self.rw_p.numpy()[0]),
            loss_prop=float(self.buf.loss.numpy()[0]),
            loss_geo=float(self.geo.numpy()[0]),
            loss_total=float(self.total.numpy()[0]),
            loss_hull=float(self.hull_const.numpy()[0] if self.frozen
                            else self.s.loss.numpy()[0]),
            row=r,
        )
        if full and not self.frozen:
            m = self.s.measure()
            out["gz30"] = float(m["gz30"])
            out["parts"] = {k: float(v) for k, v in m["parts"].items()}
            out["lwl"] = float(m["lwl"])
        elif full:
            out["gz30"] = self._frozen_gz30
            out["parts"] = self._frozen_parts
            out["lwl"] = self._frozen_lwl
        return out


def line(sn):
    return (f"D {sn['D']:.4f}  P/D {sn['PD']:.4f}  x_s {sn['x_s']:+.3f}  "
            f"y_s {sn['y_s']:+.3f}  c {sn['clearance']:+.4f}/"
            f"{sn['clear_req']:.3f}  h {sn['depth']:.3f}  "
            f"n {sn['n_survey'] * 60.0:6.1f} rpm  P_D {sn['P_survey']:8.1f} W  "
            f"range {sn['range_km']:7.3f} km  Bur(spr) {sn['ratio_sprint']:.3f}"
            f"  R_T {sn['rt_survey']:7.1f}/{sn['rt_sprint']:7.1f} N")


def hinges_ok(sn, brief, tol=1.01, slack=0.005):
    """`slack` is the penalty method's own residual, not a fudge: a squared
    hinge at lam is in equilibrium when 2 lam (violation) equals the objective's
    pull, so a binding constraint is always satisfied to about |g| / 2 lam and
    never exactly. The pilot allows Burrill 1.01 for the same reason; 5 mm on a
    ~0.11 m clearance requirement is the same statement in metres. The actual
    violation is printed, so the number is not hidden behind the word 'pass'.
    """
    bad = []
    if sn["clearance"] < sn["clear_req"] - slack:
        bad.append(f"clearance {sn['clearance']:.4f} < {sn['clear_req']:.4f} "
                   f"(short by {1000.0 * (sn['clear_req'] - sn['clearance']):.1f}"
                   f" mm, slack {1000.0 * slack:.0f} mm)")
    if sn["depth"] - 0.5 * sn["D"] < brief["imm_margin"] - slack:
        bad.append("disc not fully immersed")
    if sn["ratio_sprint"] > tol:
        bad.append(f"Burrill(sprint) {sn['ratio_sprint']:.4f} > {tol}")
    if sn["n_sprint"] > brief["n_max"] * tol:
        bad.append(f"n_sprint {sn['n_sprint']:.2f} > {brief['n_max']}")
    return bad


def binding(sn, brief, tol=2.0e-3):
    out = []
    if sn["D"] >= brief["D_max"] * (1.0 - tol):
        out.append("D = D_max")
    if sn["D"] <= prop.D_MIN * (1.0 + tol):
        out.append("D = D_min")
    if abs(sn["clearance"] - sn["clear_req"]) <= 5.0e-3:
        out.append("clearance = 0.2 D")
    if sn["ratio_sprint"] >= 1.0 - tol:
        out.append("Burrill(sprint) = 1")
    if sn["n_sprint"] >= brief["n_max"] * (1.0 - tol):
        out.append("n_sprint = n_max")
    if sn["PD"] >= prop.PD_HI * (1.0 - tol):
        out.append("P/D = P/D_hi")
    if sn["PD"] <= prop.PD_LO * (1.0 + tol):
        out.append("P/D = P/D_lo")
    for nm, k, b in (("x_s", "x_s", "x_lo"), ("x_s", "x_s", "x_hi"),
                     ("y_s", "y_s", "y_lo"), ("y_s", "y_s", "y_hi")):
        if abs(sn[k] - brief[b]) <= 2.0e-3:
            out.append(f"{nm} = {b}")
    return out or ["interior"]


# ── the runs ───────────────────────────────────────────────────────────────
def fd5(loss_at, h):
    """Five-point central difference, error O(h^4).

    A two-point rule would do if the objective were mild, and it is not: the
    cavitation hinge is a stiff quadratic wall and the step-0 design is well
    inside it, so the third derivative along a hull vertex is large and a
    two-point difference at the only h the float32 hull leaves room for (~0.02
    m, a sixth of a face) measures curvature, not slope -- measured: 1-40 %
    disagreement that shrank as h fell, which is the signature of truncation
    and not of a wrong adjoint. Richardson's four evaluations buy back the two
    orders and let h stay big enough to clear the noise floor.
    """
    return (8.0 * (loss_at(h) - loss_at(-h))
            - (loss_at(2.0 * h) - loss_at(-2.0 * h))) / (12.0 * h)


def pick_w_prop(cd, dyn=False):
    """w_prop so the two halves of the loss are the same size at step 0.

    The instruction is "within a factor ~3", and the honest reading of it is
    to make the ratio one and then say what that bought: if the propeller's
    step-0 loss is dominated by a hinge that later goes quiet, this weight is
    quoted off a number that does not survive, and the report has to say so.
    """
    sn = cd.snapshot("probe", dyn=dyn)
    lh = abs(sn["loss_hull"])
    lp = abs(sn["loss_prop"] + sn["loss_geo"])
    w = lh / max(lp, 1.0e-9)
    return w, sn


def run(cd, steps, label, dyn_schedule=True, verbose=True, every=None,
        snapshots=(0, 50, 200), out_dir=None, brief=None):
    """The coupled descent. Returns (step-0 snapshot, final snapshot, wall)."""
    saved = {}
    first = cd.snapshot(label + " step 0", dyn=False if not dyn_schedule else None,
                        full=True)
    if out_dir:
        saved[0] = first
    if verbose:
        print(f"    it {0:5d}  {line(first)}")
    every = every or max(steps // 10, 1)
    t0 = time.perf_counter()
    for it in range(steps):
        cd.step(dyn=None if dyn_schedule else False)
        if out_dir and (it + 1) in snapshots:
            saved[it + 1] = cd.snapshot(f"{label} it {it + 1}")
        if verbose and (it + 1) % every == 0:
            sn = cd.snapshot(f"{label} it {it + 1}")
            print(f"    it {it + 1:5d}  {line(sn)}")
    wall = time.perf_counter() - t0
    last = cd.snapshot(label + " final", full=True)
    if out_dir:
        saved["final"] = last
        write_designs(out_dir, saved, cd.brief if brief is None else brief)
    return first, last, wall


def write_designs(out_dir, saved, brief):
    os.makedirs(out_dir, exist_ok=True)
    for key, sn in saved.items():
        blob = optimize.design_json(sn["row"], brief,
                                    key if key != "final" else -1)
        blob.update(x_s=sn["x_s"], y_s=sn["y_s"], clearance=sn["clearance"],
                    clear_req=sn["clear_req"], draft=sn["draft"],
                    R_T_survey=sn["rt_survey"], R_T_sprint=sn["rt_sprint"],
                    S_wet=sn["s_wet"], s_form=sn["s_form"])
        name = ("design_final.json" if key == "final"
                else f"design_iter{int(key):04d}.json")
        with open(os.path.join(out_dir, name), "w", encoding="utf-8") as fh:
            json.dump(blob, fh, indent=2, sort_keys=True, default=float)


# ── gate 5: the pictures ───────────────────────────────────────────────────
def section_at(verts, faces, x0):
    """Polyline segments where the plane x = x0 cuts the triangulation."""
    segs = []
    for tri in faces:
        p = verts[tri]
        s = p[:, 0] - x0
        hit = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (s[a] > 0.0) != (s[b] > 0.0):
                t = s[a] / (s[a] - s[b])
                hit.append(p[a] + t * (p[b] - p[a]))
        if len(hit) == 2:
            segs.append((hit[0], hit[1]))
    return segs


def three_view(path, verts, faces, sn, brief, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    e = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.unique(np.sort(e, axis=1), axis=0)
    d, xs, ys = sn["D"], sn["x_s"], sn["y_s"]
    wl = sn["draft"]

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.2))
    for k, (ia, ib, nm) in enumerate(((0, 1, "side  (x, y)"),
                                      (0, 2, "top  (x, z)"))):
        seg = np.stack([verts[e[:, 0]][:, [ia, ib]],
                        verts[e[:, 1]][:, [ia, ib]]], axis=1)
        ax[k].add_collection(LineCollection(seg, colors="0.55", linewidths=0.25,
                                            alpha=0.5))
        ax[k].set_title(nm)
        ax[k].set_xlabel("x [m]")
        ax[k].set_ylabel("y [m]" if k == 0 else "z [m]")
    # side: waterline, shaft, disc edge-on
    ax[0].axhline(wl, color="#1f77b4", lw=1.2, ls="--")
    ax[0].plot([xs, xs], [ys - 0.5 * d, ys + 0.5 * d], color="#d62728", lw=3.0,
               solid_capstyle="butt")
    ax[0].plot([xs], [ys], "o", color="#d62728", ms=4)
    ax[0].plot([xs - 0.35, xs], [ys, ys], color="#d62728", lw=1.2)
    # top: the disc seen from above is the same chord
    ax[1].plot([xs, xs], [-0.5 * d, 0.5 * d], color="#d62728", lw=3.0)
    for k in (0, 1):
        ax[k].set_aspect("equal")
        ax[k].autoscale_view()

    segs = section_at(verts, faces, xs)
    if segs:
        arr = np.array(segs)[:, :, [2, 1]]
        ax[2].add_collection(LineCollection(arr, colors="0.25", linewidths=1.2))
    th = np.linspace(0.0, 2.0 * math.pi, 128)
    ax[2].plot(0.5 * d * np.cos(th), ys + 0.5 * d * np.sin(th),
               color="#d62728", lw=1.8)
    ax[2].plot([0.0], [ys], "o", color="#d62728", ms=4)
    # the clearance the hinge actually scores, drawn from the disc's top point
    ax[2].plot([0.0], [ys + 0.5 * d], "x", color="#d62728", ms=7)
    ax[2].axhline(wl, color="#1f77b4", lw=1.2, ls="--")
    ax[2].set_title(f"section at x_s = {xs:+.3f} m   (disc to scale)")
    ax[2].set_xlabel("z [m]")
    ax[2].set_ylabel("y [m]")
    ax[2].set_aspect("equal")
    ax[2].autoscale_view()

    fig.suptitle(
        f"{title}   D {d:.3f} m  P/D {sn['PD']:.3f}  clearance "
        f"{sn['clearance']:+.4f} m (needs {sn['clear_req']:.3f})  "
        f"h {sn['depth']:.3f} m  range {sn['range_km']:.3f} km  "
        f"R_T {sn['rt_survey']:.0f}/{sn['rt_sprint']:.0f} N  "
        f"S_wet {sn['s_wet']:.3f} m2", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ── gates ──────────────────────────────────────────────────────────────────
def gate1_fd(brief, device):
    """Central FD on (D, P/D, x_s, y_s) with the hull frozen.

    The hull runs ONCE, outside the tape; R_T and the hull's loss are then
    constants and every number the difference touches is float64 out of a
    dim=1 kernel. That is deliberate: the hull's own float32 atomics are
    order-dependent on CUDA, and at h ~ 1e-5 their jitter would swamp the
    signal. Gate 2 is where the hull itself gets differenced, and it runs
    where the sum is reproducible.
    """
    s = build_sculpt(device, subdiv=3)
    cd = CoDesign(s, brief, device, w_prop=1.0)
    cd.freeze()
    print(f"  frozen hull: R_T {cd._rt_host[0]:.2f} / {cd._rt_host[1]:.2f} N, "
          f"s_form {float(s.drag.numpy()[0]):.4f} m2, "
          f"S_wet {float(s.area.numpy()[0]):.4f} m2, "
          f"draft {s._d_host[0]:+.4f} m")

    base = np.array([STEP0["D"], STEP0["PD"], STEP0["x_s"], STEP0["y_s"]])
    designs = [base,
               np.array([0.62, 1.15, -1.22, -0.86]),
               np.array([0.22, 0.60, -0.70, -0.40]),
               np.array([0.50, 1.32, -1.05, -0.88]),
               # ... and one with the IMMERSION hinge active (disc top above
               # the waterline), which the other four leave at zero.
               np.array([0.68, 0.90, -1.10, -0.22])]
    # range_ref is the objective's own normaliser and is HELD, as it is during
    # an optimisation: a difference that moved it would score two objectives.
    cd.set_design(base)
    cd.evaluate()
    cd.buf.range_ref.assign(np.ascontiguousarray(
        prop.read_rows(cd.buf)["range_km"]))

    names = ("dL/dD", "dL/d(P/D)", "dL/dx_s", "dL/dy_s")
    print("    #      D     P/D     x_s     y_s   clear  coord        "
          "     finite-diff          adjoint      rel")
    ok, worst, worst_at = True, 0.0, ""
    for i, dz in enumerate(designs):
        cd.set_design(dz)
        cd.evaluate()
        # Hold the closest-FACE lookup for the whole row: with the face and the
        # barycentrics fixed the clearance is a smooth function of (p_top, x)
        # and it is exactly the function the tape differentiates.
        cd._query(dz)
        held = (cd.qface.numpy().copy(), cd.quv.numpy().copy(),
                cd.qsgn.numpy().copy())
        cl = float(cd.clear.numpy()[0])

        tape = wp.Tape()
        cd._push_theta()
        with tape:
            cd.screw_forward()
        tape.backward(loss=cd.total)
        adj = (cd.th64.grad.numpy().astype(np.float64)
               + cd.th32.grad.numpy().astype(np.float64))
        tape.zero()

        def loss_at(p):
            cd.u = np.asarray(p, dtype=np.float64) / cd.scale
            cd._push_theta()
            cd.qface.assign(held[0])
            cd.quv.assign(held[1])
            cd.qsgn.assign(held[2])
            cd.screw_forward()
            return float(cd.total.numpy()[0])

        for c in range(4):
            # 1e-4 relative: small enough that 2h stays well inside the
            # distance |p_top - p_closest| (that length has a cusp at zero, and
            # a stencil that straddles it measures the cusp), big enough that
            # a float64 chain has three digits of signal left.
            h = 1.0e-4 * max(abs(dz[c]), 1.0e-3)

            def shift(dh, _c=c, _p=dz):
                q = _p.copy()
                q[_c] += dh
                return loss_at(q)

            fd = fd5(shift, h)
            a = adj[c]
            absd = abs(a - fd)
            rel = absd / max(abs(fd), 1.0e-300)
            good = rel < 0.01 or absd < 1.0e-6
            ok = ok and good
            if abs(fd) > 1.0e-9 and rel > worst:
                worst, worst_at = rel, f"row {i} {names[c]}"
            print(f"  {i:3d} {dz[0]:6.3f} {dz[1]:7.3f} {dz[2]:7.3f} "
                  f"{dz[3]:7.3f} {cl:+7.3f}  {names[c]:<12s}"
                  f"{fd:+18.10e} {a:+16.10e} {rel:8.1e}  "
                  f"{'ok' if good else 'FAIL'}")
    print(f"  worst component {worst:.2e} relative"
          f"{' at ' + worst_at if worst_at else ''}"
          f"  (pass < 1e-2, or < 1e-6 absolute)")
    return ok


def gate2_fd_hull(brief, verbose=True):
    """Central FD on hull VERTICES, draft plane held, static phase.

    Runs on the CPU device on purpose: Warp's CPU launches are a serial loop,
    so the float32 atomics in `hull_drag`, `hull_area` and `wave_amp` sum in
    index order and two identical forwards agree to the bit. The gate prints
    that repeatability before it prints a single derivative -- without it the
    numbers below would be measuring summation order.
    """
    dev = "cpu"
    s = build_sculpt(dev, subdiv=3, verbose=False)
    cd = CoDesign(s, brief, dev, w_prop=1.0)
    print(f"  cpu hull: {s.n_verts} verts / {s.n_faces} faces")
    s.solve_drafts(wide=True)          # ONCE: the plane is held from here on
    base_d = s._d_host.copy()
    cd.lwl_hold = max(float(s.extents()[0]), 1.0)

    dz = np.array([STEP0["D"], STEP0["PD"], STEP0["x_s"], STEP0["y_s"]])
    cd.set_design(dz)
    cd.evaluate(dyn=False)
    cd.buf.range_ref.assign(np.ascontiguousarray(
        prop.read_rows(cd.buf)["range_km"]))
    cd._query(dz)
    held = (cd.qface.numpy().copy(), cd.quv.numpy().copy(),
            cd.qsgn.numpy().copy())

    def loss_now():
        """(the COUPLING half, the sculpt's own half).

        The gate scores the first. The coupling half is everything this step
        added -- R_T at the two brief speeds (form + friction + a second and a
        third Michell launch), the shaft depth off the draft plane, the tip
        clearance -- reached from the hull's vertices, which is exactly the
        claim "the coupling kernels are right". The sculpt's own hull loss is
        differenced too and reported, but it is not this file's code and not
        this gate's verdict.
        """
        cd.qface.assign(held[0])
        cd.quv.assign(held[1])
        cd.qsgn.assign(held[2])
        cd.hull_forward(dyn=False)
        cd.screw_forward()
        return float(cd.coupl.numpy()[0]), float(s.loss.numpy()[0])

    l1, l2 = loss_now(), loss_now()
    print(f"  forward repeatability: L_couple {l1[0]!r} vs {l2[0]!r}, "
          f"|dL| {abs(l1[0] - l2[0]):.3e}  (a CUDA pair differs by ~1e-6)")

    x0 = s.x.numpy().copy()
    cd._push_theta()
    tape = wp.Tape()
    with tape:
        cd.hull_forward(dyn=False)
        cd.screw_forward()
    tape.backward(loss=cd.coupl)
    gx = s.x.grad.numpy().astype(np.float64).copy()
    tape.zero()
    tape2 = wp.Tape()
    with tape2:
        cd.hull_forward(dyn=False)
        cd.screw_forward()
    tape2.backward(loss=s.loss)
    gh = s.x.grad.numpy().astype(np.float64).copy()
    tape2.zero()

    rng = np.random.default_rng(20260914)
    verts = rng.choice(s.n_verts, size=8, replace=False)
    # h is a sixth of a face width: small against the waterline softening eps
    # (0.11 m here) so the loss is smooth over it, large against the float32
    # noise floor so the difference is signal.
    h = 0.01
    print("    vert  comp        finite-diff          adjoint      rel"
          "     | sculpt's own L_hull: fd / adj")
    ok, worst, worst_h = True, 0.0, 0.0
    rows = []
    for vi in verts:
        for c in range(3):
            def shift(dh, _v=vi, _c=c, _k=0):
                xp = x0.copy()
                xp[_v, _c] += dh
                s.x.assign(xp)
                s.plane_d.assign(base_d.astype(np.float32))
                return loss_now()[_k]

            fd = fd5(shift, h)
            fdh = fd5(lambda dh, _v=vi, _c=c: shift(dh, _v, _c, 1), h)
            rows.append((vi, c, fd, gx[vi, c], fdh, gh[vi, c]))
    s.x.assign(x0)

    fda = np.array([r[2] for r in rows])
    adj = np.array([r[3] for r in rows])
    # The absolute floor is not 1e-6 here and pretending otherwise would be
    # tuning the gate to the answer. It is set by the float32 quantum of the
    # hull's own accumulators (R_wave at the sprint is a float32 near 1 kN;
    # one ulp is 6e-5 N) multiplied by the cavitation wall's stiffness,
    # 2 lam (ratio - 1) ~ 250 at this design -- about 1e-3 of loss per metre.
    # MEASURED, not assumed: softening lam_cav to 1 drops the same components'
    # disagreement to ~1e-5, which is the same floor scaled by the same 100.
    # So the floor is quoted as a fraction of the largest component present,
    # and the headline is the standard vector-norm gradient check.
    tol_abs = 1.0e-4 * float(np.abs(fda).max())
    for vi, c, fd, a, fdh, ah in rows:
        absd = abs(a - fd)
        rel = absd / max(abs(fd), 1.0e-300)
        good = rel < 0.01 or absd < tol_abs
        ok = ok and good
        if absd >= tol_abs:
            worst = max(worst, rel)
        if abs(fdh) > 1.0e-6:
            worst_h = max(worst_h, abs(ah - fdh) / abs(fdh))
        if verbose:
            print(f"  {vi:6d}  {'xyz'[c]}    {fd:+18.10e} {a:+16.10e} "
                  f"{rel if abs(fd) > 1.0e-12 else 0.0:8.1e}  "
                  f"{'ok' if good else 'FAIL'}"
                  f"  | {fdh:+11.4e} {ah:+11.4e}")
    nrm = (np.linalg.norm(adj - fda)
           / (np.linalg.norm(adj) + np.linalg.norm(fda)))
    good = nrm < 1.0e-3
    ok = ok and good
    print(f"  ||adj - fd|| / (||adj|| + ||fd||) = {nrm:.3e} over 8 vertices x "
          f"3 components, h {h} m  (pass < 1e-3)  {'ok' if good else 'FAIL'}")
    print(f"  worst component {worst:.2e} relative above the floor "
          f"{tol_abs:.2e} (1e-4 of the largest component)  (pass < 1e-2)")
    # NOT a verdict, and NOT this step's code: the sculpt's own hull_loss
    # adjoint disagrees with its own central difference by up to the number
    # below at these vertices. It is reported because the co-design descends
    # the SUM, and because nobody had differenced warp_hull_sculpt's static
    # loss before. The coupling column above is what this file is answerable
    # for; `Sculptor.forward()` and its kernels are out of scope here by the
    # brief's own rule.
    print(f"  note  the SCULPT's own L_hull adjoint vs its own FD, same "
          f"vertices: worst {worst_h:.2e} relative -- not this gate's verdict, "
          f"and not code this step may touch")
    return ok


def gate3_regression(brief, device):
    """The frozen hull's two R_T numbers, handed to the pilot's brute force."""
    s = build_sculpt(device, subdiv=3, verbose=False)
    cd = CoDesign(s, brief, device, w_prop=1.0, w_geo=0.0)
    cd.freeze()
    rt = cd._rt_host
    d0 = float(s._d_host[0])
    # depth maps to y_s through the frozen plane: h = d - y_s.
    b2 = json.loads(json.dumps({k: v for k, v in brief.items()
                                if k != "hull"}))
    b2["hull"] = dict(L=float(s.lwl), S_wet=float(s.area.numpy()[0]),
                      k_form=0.25, C_R=0.0,
                      R_T={brief["V_survey"]: float(rt[0]),
                           brief["V_sprint"]: float(rt[1])})
    b2["depth_min"] = d0 - brief["y_hi"]
    b2["depth_max"] = d0 - brief["y_lo"]
    print(f"  frozen hull R_T {rt[0]:.2f} N at {brief['V_survey']} m/s, "
          f"{rt[1]:.2f} N at {brief['V_sprint']} m/s; draft {d0:+.4f} m -> "
          f"depth box [{b2['depth_min']:.3f}, {b2['depth_max']:.3f}] m")

    best = optimize.grid_search(b2, device, nd=240, npd=240, nh=8)

    # The coupled run must score the SAME objective the grid scored: the
    # geometric hinges are off (w_geo 0) and range_ref is the grid's own.
    cd.brief = b2
    ref = optimize.step0_range(b2, device)
    cd.buf.range_ref.fill_(ref)
    st0 = optimize.step0_design(b2)[0]
    cd.set_design(np.array([st0[0], st0[1], STEP0["x_s"], d0 - st0[2]]))
    cd.lo[3] = brief["y_lo"] / cd.scale[3]
    cd.hi[3] = brief["y_hi"] / cd.scale[3]
    # The pilot's own remedy for its own pathology: the range optimum lies
    # along a curved ridge in (D, P/D) that Adam's per-coordinate scaling does
    # not see, and here the box is nearly three times wider in D than the
    # pilot's (0.15-0.70 against 0.15-0.25) with a start that violates the
    # cavitation wall by a factor of five. 1000 steps stalled at D 0.596
    # against the grid's 0.700 -- not a wrong gradient, a slow one; optimize.py
    # gives its own hard case max(steps, 1500) for the same reason.
    cd.steps = GATE3_STEPS
    cd.lr = GATE3_LR
    first, last, wall = run(cd, GATE3_STEPS, "gate3", dyn_schedule=False,
                            verbose=False)
    print(f"  coupled Adam, {GATE3_STEPS} steps at lr {GATE3_LR}, {wall:.2f} s"
          f"   (range_ref {ref:.4f} km, the grid's)")
    print(f"  step    0  {optimize.design_line(first['row'], b2)}")
    print(f"  step {GATE3_STEPS:4d}  {optimize.design_line(last['row'], b2)}")
    agree, lines = optimize.verdict(best, last["row"], b2)
    for ln in lines:
        print(ln)
    return agree


def gate4_compare(brief, device, steps, shots="", w_prop_cli=-1.0):
    """The co-design against the same hull held still."""
    out = {}
    # --- the frozen control -------------------------------------------------
    s_f = build_sculpt(device, subdiv=ARGS.subdiv)
    cd_f = CoDesign(s_f, brief, device, w_prop=1.0, lr=ARGS.lr, steps=steps)
    cd_f.freeze()
    # range_ref is the STEP-0 range, so the objective's range term starts at
    # exactly -1 and w_prop below is quoted against a known number.
    cd_f.evaluate()
    cd_f.buf.range_ref.assign(np.ascontiguousarray(
        prop.read_rows(cd_f.buf)["range_km"]))
    ref = float(cd_f.buf.range_ref.numpy()[0])
    w_probe, sn0 = pick_w_prop(cd_f)
    w_prop = w_probe if w_prop_cli < 0.0 else w_prop_cli
    print(f"  step-0 losses: hull {sn0['loss_hull']:.4f}   "
          f"prop {sn0['loss_prop']:.4f} + geom {sn0['loss_geo']:.4f} = "
          f"{sn0['loss_prop'] + sn0['loss_geo']:.4f}   ->  w_prop {w_prop:.5f}"
          f"   (w_prop x prop = {w_prop * (sn0['loss_prop'] + sn0['loss_geo']):.4f},"
          f" ratio {abs(w_prop * (sn0['loss_prop'] + sn0['loss_geo']) / max(abs(sn0['loss_hull']), 1e-9)):.2f})")
    cd_f.w_prop = w_prop
    # The frozen hull never moves, so its measured stability and Lwl are one
    # reading, taken here and reused by every snapshot.
    m = s_f.measure()
    cd_f._frozen_gz30 = float(m["gz30"])
    cd_f._frozen_parts = {k: float(v) for k, v in m["parts"].items()}
    cd_f._frozen_lwl = float(m["lwl"])
    print(f"\n  [frozen hull] {steps} steps")
    f0, f1, wall_f = run(cd_f, steps, "frozen", dyn_schedule=False)
    out["frozen0"], out["frozen1"], out["wall_f"] = f0, f1, wall_f

    # --- the co-design ------------------------------------------------------
    s_c = build_sculpt(device, subdiv=ARGS.subdiv, verbose=False)
    cd_c = CoDesign(s_c, brief, device, w_prop=w_prop, lr=ARGS.lr, steps=steps)
    cd_c.buf.range_ref.fill_(ref)      # the SAME normaliser as the control
    print(f"\n  [co-design] {steps} steps, hull + screw, dyn phase after step "
          f"{int(hs.DYN_AFTER * STEPS)}")
    shot_dir = shots or None
    c0, c1, wall_c = run(cd_c, steps, "codesign", dyn_schedule=True,
                         out_dir=shot_dir, brief=brief)
    out["co0"], out["co1"], out["wall_c"] = c0, c1, wall_c
    out["w_prop"] = w_prop
    out["sculpt"] = s_c
    out["cd"] = cd_c
    return out


def table(out, brief):
    f0, f1, c1 = out["frozen0"], out["frozen1"], out["co1"]
    rows = [
        ("range [km]", "range_km", "{:10.4f}"),
        ("P_D(survey) [W]", "P_survey", "{:10.1f}"),
        ("D [m]", "D", "{:10.4f}"),
        ("P/D", "PD", "{:10.4f}"),
        ("x_s [m]", "x_s", "{:10.4f}"),
        ("y_s [m]", "y_s", "{:10.4f}"),
        ("clearance [m]", "clearance", "{:10.4f}"),
        ("  required 0.2 D", "clear_req", "{:10.4f}"),
        ("draft d [m]", "draft", "{:10.4f}"),
        ("shaft depth h [m]", "depth", "{:10.4f}"),
        ("S_wet [m2]", "s_wet", "{:10.4f}"),
        ("s_form [m2]", "s_form", "{:10.4f}"),
        ("R_T(survey) [N]", "rt_survey", "{:10.2f}"),
        ("R_T(sprint) [N]", "rt_sprint", "{:10.2f}"),
        ("R_wave(survey) [N]", "rwave_survey", "{:10.2f}"),
        ("GZ(30) [m]", "gz30", "{:10.4f}"),
        ("Burrill(sprint)", "ratio_sprint", "{:10.4f}"),
        ("n_sprint [rps]", "n_sprint", "{:10.3f}"),
        ("eta_0(survey)", "eta_survey", "{:10.4f}"),
        ("L_hull", "loss_hull", "{:10.4f}"),
        ("L_prop", "loss_prop", "{:10.4f}"),
        ("L_geom", "loss_geo", "{:10.4f}"),
    ]
    print(f"  {'quantity':<22s}{'step 0':>12s}{'frozen final':>16s}"
          f"{'co-design final':>18s}")
    for nm, key, fmt in rows:
        def g(sn):
            v = sn.get(key)
            return fmt.format(v) if v is not None else "         -"
        print(f"  {nm:<22s}{g(f0):>12s}{g(f1):>16s}{g(c1):>18s}")
    parts = sorted(set(list(f0.get("parts", {}).keys())
                       + list(c1.get("parts", {}).keys())))
    for p in parts:
        print(f"  {'  L_hull.' + p:<22s}"
              f"{f0.get('parts', {}).get(p, float('nan')):>12.4f}"
              f"{f1.get('parts', {}).get(p, float('nan')):>16.4f}"
              f"{c1.get('parts', {}).get(p, float('nan')):>18.4f}")


def main():
    wp.init()
    device = ARGS.device or ("cuda:0" if wp.get_cuda_device_count() else "cpu")
    brief = json.loads(json.dumps(BRIEF))
    steps = ARGS.steps if ARGS.steps > 0 else STEPS

    print("=" * 100)
    print(f"sea-drone co-design, step 3: hull + screw in ONE tape -- device "
          f"{device}")
    print(f"  brief    V_survey {brief['V_survey']} m/s, V_sprint "
          f"{brief['V_sprint']} m/s, w {brief['w']}, t {brief['t']}, "
          f"E_batt {brief['E_batt']} kWh, n_max {brief['n_max']} rps")
    print(f"  screw    B{prop.BLADES}-{prop.EAR:.2f}, D [{prop.D_MIN}, "
          f"{brief['D_max']}] m, P/D [{prop.PD_LO}, {prop.PD_HI}], "
          f"x_s [{brief['x_lo']}, {brief['x_hi']}] m, y_s [{brief['y_lo']}, "
          f"{brief['y_hi']}] m")
    print(f"  geometry tip clearance >= {brief['clear_frac']} D, whole disc "
          f">= {brief['imm_margin']} m under the surface")
    print(f"  sculpt   --blob-scale {ARGS.blob_scale} --subdiv {ARGS.subdiv} "
          f"--cargo-frac {brief['payload_frac']} --cargo-y {brief['cargo_y']} "
          f"--gz-floor {brief['gz_floor']} --u-ref {brief['V_survey']}, "
          f"{steps} steps")
    print("=" * 100)

    if not ARGS.selftest and not ARGS.shots:
        ARGS.selftest = True

    ok = True
    if ARGS.selftest:
        print("\n[1/5] FD on the four screw variables, hull frozen")
        ok &= gate1_fd(brief, device)
        print("\n[2/5] FD on hull vertices, draft held, static phase (CPU)")
        ok &= gate2_fd_hull(brief)
        print("\n[3/5] frozen-hull regression to the pilot's brute force")
        ok &= gate3_regression(brief, device)

    print("\n[4/5] co-design vs frozen hull, same start")
    out = gate4_compare(brief, device, steps, shots=ARGS.shots,
                        w_prop_cli=ARGS.w_prop)
    print()
    table(out, brief)
    c1, f1, f0 = out["co1"], out["frozen1"], out["frozen0"]
    gain = c1["range_km"] >= f1["range_km"]
    bad = hinges_ok(c1, brief)
    print(f"\n  range  step 0 {f0['range_km']:.4f} km  ->  frozen "
          f"{f1['range_km']:.4f} km  ->  co-design {c1['range_km']:.4f} km "
          f"({100.0 * (c1['range_km'] / max(f1['range_km'], 1e-9) - 1.0):+.2f}% "
          f"on the frozen hull)")
    print(f"  binding at the end: co-design [{', '.join(binding(c1, brief))}]"
          f"   frozen [{', '.join(binding(f1, brief))}]")
    print(f"  clearance at the end: co-design {c1['clearance']:+.4f} m vs "
          f"{c1['clear_req']:.4f} required "
          f"({1000.0 * (c1['clearance'] - c1['clear_req']):+.1f} mm), "
          f"frozen {f1['clearance']:+.4f} vs {f1['clear_req']:.4f} "
          f"({1000.0 * (f1['clearance'] - f1['clear_req']):+.1f} mm)")
    print(f"  hinges at the end: "
          f"{'all satisfied' if not bad else 'VIOLATED -> ' + '; '.join(bad)}")
    print(f"  frozen hinges:     "
          f"{'all satisfied' if not hinges_ok(f1, brief) else 'violated -> ' + '; '.join(hinges_ok(f1, brief))}")
    print(f"  wall time: frozen {out['wall_f']:.1f} s, co-design "
          f"{out['wall_c']:.1f} s ({1000.0 * out['wall_c'] / max(steps, 1):.1f} "
          f"ms per coupled step)")
    ok &= bool(gain) and not bad

    if ARGS.shots:
        print("\n[5/5] shots")
        d = ARGS.shots
        os.makedirs(d, exist_ok=True)
        s = out["sculpt"]
        v1 = s.x.numpy().astype(np.float64)
        v0 = s.rest.astype(np.float64)
        np.savez(os.path.join(d, "hull.npz"), rest=v0, final=v1, faces=s.faces,
                 plane_n=s.plane_n.numpy(), plane_d=s.plane_d.numpy(),
                 draft=s._d_host)
        p0 = three_view(os.path.join(d, "hull_iter0000.png"), v0, s.faces,
                        out["co0"], brief, "step 0 (the blob)")
        p1 = three_view(os.path.join(d, "hull_final.png"), v1, s.faces,
                        out["co1"], brief, f"co-design final ({steps} steps)")
        print(f"  wrote {p0}")
        print(f"  wrote {p1}")
        print(f"  wrote {os.path.join(d, 'hull.npz')} and the design JSONs")

    print(f"\n{'PASS' if ok else 'FAIL'}: the hull and the screw are on one "
          f"tape and the coupled optimum "
          f"{'beats' if gain else 'does NOT beat'} the frozen hull's")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
