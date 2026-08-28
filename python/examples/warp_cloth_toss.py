"""Can a cloth throw a ball? -- the spike behind the four-robot toss demo.

No robots here on purpose. Four *kinematic* corner anchors hold a square sheet;
a ball drops in; the anchors run a scripted down-then-snap pump and the sheet
launches the ball. That is the one question the whole demo rests on, because
cloth is dissipative: a ball dropped on a slack sheet just stops. The energy for
a throw comes from the anchors yanking down and then snapping up faster than g
-- exactly how a blanket toss works -- not from any elasticity in the fabric.

The physics that matter here:

  * Cloth is Jacobi-projected distance constraints (structural / shear / bend)
    on the GPU, under-relaxed. Corners are infinite-mass, which is the *worst*
    case for stability -- a hard pin yanked at several m/s puts a shock through
    the sheet every substep. If this detonates, the fix is a stiff spring to the
    anchor instead of a pin, which is also what a real gripper is.

  * Ball <-> cloth is two-way. The solve pushes cloth particles out of the
    sphere and atomically accumulates `m_i * (pushout delta)`; that sum IS the
    momentum the ball handed the cloth, so the reaction on the ball is just
    -sum/h. Internal cloth constraints cancel pairwise in that sum by
    construction, so nothing else leaks in. Same shape as the sail->hull
    transfer in warp_sailboat.py and the shell contact in warp_water_balloon.py.

  * Aim comes from tilting the snap: the anchors on one side rise further, the
    sheet leaves under the ball at an angle, the ball goes sideways. That is the
    knob an optimiser would later turn; here it is a flag so the launch can be
    steered by hand first.

    python warp_cloth_toss.py                 # window, orbit with the mouse
    python warp_cloth_toss.py --tune          # headless, numbers only
    python warp_cloth_toss.py --shot out.png  # headless still at the end
    python warp_cloth_toss.py --rise 0.34 --snap 0.13 --tilt 0.35

Reported per run: launch speed, apex, landing point, and the peak edge strain --
the last one is the detonation detector. Strain much over ~0.1 means the sheet
is being stretched, not thrown with.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import warp as wp

import threepp as tp
from warp_common import cli_arg, ground_plane, standard_material, studio_lights

# ---- flags -------------------------------------------------------------------
TUNE = "--tune" in sys.argv          # no window, no renderer: just the numbers
SHOT = cli_arg("--shot", "", str)    # headless PNG at the end of the run
CLIP = cli_arg("--clip", "", str)    # headless mp4 of the whole run
SEQ = cli_arg("--seq", "", str)      # keep the numbered PNGs at this prefix

N = int(cli_arg("--res", 48, float))         # quads per side; (N+1)^2 particles
CLOTH = cli_arg("--cloth", 1.50, float)      # sheet edge length, m
SPAN = cli_arg("--span", 1.30, float)        # anchor spacing, m  (< CLOTH = slack)
SHEET_Y = cli_arg("--sheet-y", 1.10, float)  # anchor height at rest, m
CLOTH_KG = cli_arg("--cloth-kg", 0.45, float)

BALL_R = cli_arg("--ball-r", 0.10, float)
BALL_KG = cli_arg("--ball-kg", 0.40, float)
DROP = cli_arg("--drop", 0.90, float)        # release height above the sheet, m
VX0 = cli_arg("--vx", 0.0, float)            # sideways speed on entry, m/s

DIP = cli_arg("--dip", 0.15, float)          # how far the anchors load downward, m
RISE = cli_arg("--rise", 0.28, float)        # how far they snap up from the dip, m
T_DIP = cli_arg("--dip-t", 0.16, float)      # seconds to ease down into the catch
FIRE_FRAC = cli_arg("--fire-frac", 0.15, float)  # snap once the ball's descent has slowed
                                                 # to this fraction of its impact speed.
                                                 # Relative, because a soft catch settles
                                                 # asymptotically and never cleanly hits
                                                 # vy = 0 -- waiting for that stalled the
                                                 # snap by two full seconds.
T_SNAP = cli_arg("--snap", 0.15, float)      # seconds for the throwing stroke
T_RET = cli_arg("--return-t", 0.35, float)   # seconds to ease back to rest
TILT = cli_arg("--tilt", 0.0, float)         # -1..1: asymmetric rise -> sideways aim
TAUT = cli_arg("--taut", 1.0, float)         # outward pull during the snap, in units of
                                             # the available slack. 1.0 spreads the anchors
                                             # from SPAN to exactly CLOTH, i.e. dead taut.

# Converged defaults. These used to cost ~100 ms/frame, which is why they were
# not the default; with the solve loop captured into a CUDA graph they cost
# ~7.4 ms, so there is no longer a trade to make. At 8x16 this sim reports a
# 0.67 peak strain and a launch speed that swings 2x with the timestep -- an
# under-resolved contact, not physics.
SUBSTEPS = int(cli_arg("--substeps", 48, float))
ITERS = int(cli_arg("--iters", 32, float))     # even: the graph ping-pong needs it
DAMPING = cli_arg("--damping", 0.010, float)
SECONDS = cli_arg("--seconds", 4.5, float)
SETTLE = cli_arg("--settle", 0.5, float)   # seconds of bowl-forming before release
TRACE = "--trace" in sys.argv
BENCH = int(cli_arg("--bench", 0, float))  # frames of pure sim, no host readback
FPS = 60.0
DT = 1.0 / (FPS * SUBSTEPS)

V = (N + 1) * (N + 1)
REST = CLOTH / N
M_PARTICLE = CLOTH_KG / V
GRAVITY = wp.vec3(0.0, -9.81, 0.0)

# ---- warp: cloth --------------------------------------------------------------


@wp.func
def grid_index(ix: int, iy: int, nx: int) -> int:
    return iy * (nx + 1) + ix


@wp.func
def spring(p: wp.vec3, pos: wp.array(dtype=wp.vec3), im: wp.array(dtype=float),
           w_self: float, ix: int, iy: int, nx: int, ny: int,
           rest: float, stiffness: float) -> wp.vec3:
    # Outside the grid there is no neighbour and nothing to project against.
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
    # Mass-weighted split. Against a pinned neighbour w_nb is 0 and this particle
    # takes the whole correction -- which is exactly the corner, where the throw
    # enters the sheet, so getting that weight right is not cosmetic.
    return d * (stiffness * (w_self / denom) * (l - rest) / l)


@wp.kernel
def integrate(pos: wp.array(dtype=wp.vec3),
              prev: wp.array(dtype=wp.vec3),
              pred: wp.array(dtype=wp.vec3),
              im: wp.array(dtype=float), dt: float, damping: float):
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
def set_anchors(pred: wp.array(dtype=wp.vec3),
                idx: wp.array(dtype=int), target: wp.array(dtype=wp.vec3)):
    k = wp.tid()
    pred[idx[k]] = target[k]


@wp.kernel
def solve(p_in: wp.array(dtype=wp.vec3), p_out: wp.array(dtype=wp.vec3),
          im: wp.array(dtype=float), nx: int, ny: int, rest: float,
          ball: wp.array(dtype=wp.vec3), ball_r: float,
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
    # structural
    c += spring(p, p_in, im, w, ix - 1, iy, nx, ny, rest, 1.0)
    c += spring(p, p_in, im, w, ix + 1, iy, nx, ny, rest, 1.0)
    c += spring(p, p_in, im, w, ix, iy - 1, nx, ny, rest, 1.0)
    c += spring(p, p_in, im, w, ix, iy + 1, nx, ny, rest, 1.0)
    # shear
    c += spring(p, p_in, im, w, ix - 1, iy - 1, nx, ny, rd, 0.85)
    c += spring(p, p_in, im, w, ix + 1, iy - 1, nx, ny, rd, 0.85)
    c += spring(p, p_in, im, w, ix - 1, iy + 1, nx, ny, rd, 0.85)
    c += spring(p, p_in, im, w, ix + 1, iy + 1, nx, ny, rd, 0.85)
    # bending
    c += spring(p, p_in, im, w, ix - 2, iy, nx, ny, 2.0 * rest, 0.35)
    c += spring(p, p_in, im, w, ix + 2, iy, nx, ny, 2.0 * rest, 0.35)
    c += spring(p, p_in, im, w, ix, iy - 2, nx, ny, 2.0 * rest, 0.35)
    c += spring(p, p_in, im, w, ix, iy + 2, nx, ny, 2.0 * rest, 0.35)
    p = p + c * relax

    # Long-range attachment. Jacobi propagates tension one edge per iteration,
    # so on a 49-wide grid the load under the ball never reaches the corners
    # inside a substep and the sheet stretches like a trampoline. This says the
    # thing the springs cannot say fast enough: no particle may be further from
    # an anchor than the sheet's own material distance to it. Unilateral -- it
    # only ever removes stretch -- so it adds no energy.
    for k in range(4):
        ak = anchors[k]
        dk = p - ak
        lk = wp.length(dk)
        lmax = lra[i * 4 + k]
        if lk > lmax:
            p = ak + dk * (lmax / lk)

    # Ball contact. The pushout delta is the ONLY thing the ball did to this
    # particle, so accumulating m * delta gives the momentum transfer exactly;
    # the spring corrections above are internal and cancel pairwise.
    bc = ball[0]
    d = p - bc
    l = wp.length(d)
    if l < ball_r:
        push = d * (ball_r / wp.max(l, 1.0e-6)) - d
        p = p + push
        wp.atomic_add(impulse, 0, push * m_particle)

    p_out[i] = wp.vec3(p[0], wp.max(p[1], 0.005), p[2])


@wp.kernel
def ball_predict(bp: wp.array(dtype=wp.vec3), bv: wp.array(dtype=wp.vec3),
                 bpred: wp.array(dtype=wp.vec3), h: float):
    bpred[0] = bp[0] + bv[0] * h + GRAVITY * h * h


@wp.kernel
def ball_finish(bp: wp.array(dtype=wp.vec3), bv: wp.array(dtype=wp.vec3),
                impulse: wp.array(dtype=wp.vec3), h: float, inv_bm: float,
                radius: float):
    # Third law: the cloth gained impulse/h of momentum, so the ball loses it.
    v = bv[0] + GRAVITY * h - impulse[0] * (inv_bm / h)
    p = bp[0] + v * h
    if p[1] < radius:
        p = wp.vec3(p[0], radius, p[2])
        v = wp.vec3(v[0] * 0.7, wp.abs(v[1]) * 0.35, v[2] * 0.7)
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


@wp.kernel
def strain_field(pos: wp.array(dtype=wp.vec3), out: wp.array(dtype=float),
                 nx: int, ny: int, rest: float):
    i = wp.tid()
    ix = i % (nx + 1)
    iy = i // (nx + 1)
    e = float(0.0)
    if ix < nx:
        l = wp.length(pos[grid_index(ix + 1, iy, nx)] - pos[i])
        e = wp.max(e, wp.max(l - rest, 0.0) / rest)
    if iy < ny:
        l = wp.length(pos[grid_index(ix, iy + 1, nx)] - pos[i])
        e = wp.max(e, wp.max(l - rest, 0.0) / rest)
    out[i] = e


# ---- setup --------------------------------------------------------------------

wp.init()
device = wp.get_preferred_device()

# Particle grid, laid out flat in XZ so PlaneGeometry's row-major vertex order
# (ix fastest) maps straight onto it.
# Laid out at the ANCHOR span, not the cloth size: every edge starts uniformly
# compressed by SPAN/CLOTH and buckles into the bowl, so there is no corner
# discontinuity on frame 1. A shallow dome picks the buckling direction.
xs = np.linspace(-SPAN / 2, SPAN / 2, N + 1, dtype=np.float32)
zs = np.linspace(-SPAN / 2, SPAN / 2, N + 1, dtype=np.float32)
gx, gz = np.meshgrid(xs, zs)
r2 = (gx / (SPAN / 2)) ** 2 + (gz / (SPAN / 2)) ** 2
gy = SHEET_Y - 0.5 * (CLOTH - SPAN) * np.clip(1.0 - r2, 0.0, 1.0)
p0 = np.stack([gx, gy, gz], axis=-1).reshape(-1, 3).astype(np.float32)

# The four corners are pulled in to SPAN, which is what puts the slack in.
CORNERS = [(0, 0), (N, 0), (0, N), (N, N)]
anchor_idx_np = np.array([iy * (N + 1) + ix for ix, iy in CORNERS], dtype=np.int32)
anchor_rest = np.array([[float(xs[ix]), SHEET_Y, float(zs[iy])] for ix, iy in CORNERS],
                       dtype=np.float32)
p0[anchor_idx_np] = anchor_rest

# Straight-line distance on the flat, full-size sheet. The true geodesic over
# grid edges is slightly longer, so this under-estimates -- erring toward a
# marginally tight sheet, which is the safe direction.
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
anchor_idx = wp.array(anchor_idx_np, dtype=int, device=device)
anchor_tgt = wp.array(anchor_rest.copy(), dtype=wp.vec3, device=device)

bp = wp.array(np.array([[0.0, SHEET_Y + DROP, 0.0]], np.float32), dtype=wp.vec3, device=device)
bv = wp.array(np.array([[VX0, 0.0, 0.0]], np.float32), dtype=wp.vec3, device=device)
bpred = wp.zeros(1, dtype=wp.vec3, device=device)
impulse = wp.zeros(1, dtype=wp.vec3, device=device)
strain = wp.zeros(V, dtype=float, device=device)
lra = wp.array(lra_np, dtype=float, device=device)

# TILT raises the +x pair further than the -x pair during the snap, so the sheet
# is inclined at the moment the ball leaves it.
tilt_gain = np.array([1.0 + TILT * math.copysign(1.0, xs[ix]) for ix, _ in CORNERS],
                     dtype=np.float32)

# CUDA graph for the inner solve loop. At ~45 us of Python-side launch overhead
# per kernel -- against ~2 us of actual GPU work for 2401 particles -- the loop is
# almost entirely dispatch cost, and a graph collapses ITERS launches into one.
# Requires an even ITERS so the ping-pong ends back on `pred`, and nothing
# host-touching inside; the anchor upload is per-substep and stays outside.
solve_graph = None
if device.is_cuda and ITERS % 2 == 0:
    try:
        with wp.ScopedCapture(device) as _cap:
            _a, _b = pred, scratch
            for _ in range(ITERS):
                wp.launch(solve, dim=V, device=device,
                          inputs=[_a, _b, im, N, N, REST, bpred, BALL_R + 0.012,
                                  anchor_tgt, lra, impulse, M_PARTICLE, 0.35])
                _a, _b = _b, _a
        solve_graph = _cap.graph
    except Exception as exc:                       # noqa: BLE001 - capture is optional
        print(f"  (no graph capture: {exc})")

sim_time = 0.0
t_contact = None          # first substep the cloth really carries the ball
t_snap = None             # the bottom of the catch: when the ball stops falling
v_impact = [0.0]          # the ball's vy the moment the sheet took its weight
state = "fall"            # fall -> caught -> flying -> landed
launch_v = None
apex = 0.0
landing = None
peak_strain = 0.0
peak_anchor_v = 0.0
peak_at = [0, 0, 0.0]     # ix, iy, t of the worst stretch
_prev_anchor = anchor_rest[0].copy()
anchor_sign = np.sign(anchor_rest[:, [0, 2]])


def spread_at(t):
    """Outward pull, 0 at rest and 1 when the anchors are spread to CLOTH.

    A slack sheet cannot throw: lifting the anchors just takes up slack while
    the ball sits in the bag, which is why the launch died out entirely as the
    solver got accurate. The arms have to extend *outward* as they rise, so the
    bowl flattens into a taut surface that can carry the anchors' velocity into
    the ball. This is the half of a blanket toss that is easy to miss.
    """
    if t_contact is None or t_snap is None:
        return 0.0
    sigma = t - t_snap
    if sigma < T_SNAP:
        return 0.5 * (1.0 - math.cos(math.pi * sigma / T_SNAP))
    sigma -= T_SNAP
    if sigma < T_RET:
        return 0.5 * (1.0 + math.cos(math.pi * sigma / T_RET))
    return 0.0


def _dip_at(tau):
    """Depth of the loading stroke tau seconds after contact (eases, then holds)."""
    return -DIP * 0.5 * (1.0 - math.cos(math.pi * min(tau / T_DIP, 1.0)))


def anchor_y(t, k):
    """Vertical offset of anchor k at absolute time t.

    Two events drive this, not a stopwatch: the catch starts the dip, and the
    snap only fires once the ball has stopped falling. Firing on first touch
    instead -- which is what this did originally -- puts the up-stroke into the
    middle of the rebound, so the ball bounces off a sheet that is still moving
    and the launch becomes a coin-toss on contact timing rather than physics.
    """
    if t_contact is None:
        return 0.0
    if t_snap is None:
        return _dip_at(t - t_contact)
    bottom = _dip_at(t_snap - t_contact)
    top = RISE * float(tilt_gain[k])
    sigma = t - t_snap
    if sigma < T_SNAP:
        return bottom + (top - bottom) * 0.5 * (1.0 - math.cos(math.pi * sigma / T_SNAP))
    sigma -= T_SNAP
    if sigma < T_RET:
        return top * 0.5 * (1.0 + math.cos(math.pi * sigma / T_RET))
    return 0.0


def substep():
    global sim_time, t_contact, t_snap, state, launch_v, peak_anchor_v, _prev_anchor

    half = SPAN / 2.0 + 0.5 * (CLOTH - SPAN) * TAUT * spread_at(sim_time)
    tgt = anchor_rest.copy()
    for k in range(4):
        tgt[k, 0] = anchor_sign[k, 0] * half
        tgt[k, 2] = anchor_sign[k, 1] * half
        tgt[k, 1] = SHEET_Y + anchor_y(sim_time, k)
    anchor_tgt.assign(tgt)
    peak_anchor_v = max(peak_anchor_v,
                        float(np.linalg.norm(tgt[0] - _prev_anchor)) / DT)
    _prev_anchor = tgt[0].copy()

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
                      inputs=[a, b, im, N, N, REST, bpred, BALL_R + 0.012,
                              anchor_tgt, lra, impulse, M_PARTICLE, 0.35])
            a, b = b, a
        wp.copy(pos, a)
    if sim_time >= SETTLE:
        wp.launch(ball_finish, dim=1, device=device,
                  inputs=[bp, bv, impulse, DT, 1.0 / BALL_KG, BALL_R])
    sim_time += DT

    # Contact = the sheet carrying a meaningful fraction of the ball's weight.
    if t_contact is None:
        f_up = -float(impulse.numpy()[0][1]) / (DT * DT)
        if f_up > 0.25 * BALL_KG * 9.81:
            t_contact = sim_time
            v_impact[0] = float(bv.numpy()[0][1])
            state = "caught"
    elif t_snap is None and float(bv.numpy()[0][1]) >= -FIRE_FRAC * abs(v_impact[0]):
        t_snap = sim_time
    elif state == "caught" and t_snap is not None:
        v = bv.numpy()[0]
        if float(np.linalg.norm(impulse.numpy()[0])) == 0.0 and v[1] > 0.0:
            state, launch_v = "flying", v.copy()


def step_frame():
    global peak_strain, state, launch_v, apex, landing
    for _ in range(SUBSTEPS):
        substep()
    wp.launch(strain_field, dim=V, device=device, inputs=[pos, strain, N, N, REST])
    sf = strain.numpy()
    j = int(np.argmax(sf))
    if float(sf[j]) > peak_strain:
        peak_strain = float(sf[j])
        peak_at[:] = [j % (N + 1), j // (N + 1), sim_time]
    wp.launch(compute_normals, dim=V, device=device, inputs=[pos, nrm, N, N])

    p = bp.numpy()[0]
    v = bv.numpy()[0]
    if TRACE:
        tau = -1.0 if t_contact is None else sim_time - t_contact
        print(f"t={sim_time:6.3f} tau={tau:+6.3f} anchor_y={float(anchor_tgt.numpy()[0][1]):.3f} "
              f"ball_y={p[1]:.3f} vy={v[1]:+6.2f} state={state} strain={peak_strain:.3f}")
    if state == "flying":
        apex = max(apex, float(p[1]))
        if v[1] < 0.0 and p[1] <= SHEET_Y and landing is None:
            landing = p.copy()
            state = "landed"


def report():
    print(f"\n  cloth      {V} particles, {CLOTH:.2f} m over a {SPAN:.2f} m span "
          f"({100 * (CLOTH - SPAN) / SPAN:.0f}% slack), {CLOTH_KG:.2f} kg")
    print(f"  ball       r={BALL_R:.3f} m  m={BALL_KG:.2f} kg  dropped {DROP:.2f} m")
    print(f"  pump       dip {DIP:.3f} m, rise {RISE:.3f} m / {T_SNAP:.2f} s, "
          f"spread {100 * (CLOTH - SPAN) * TAUT:.0f} cm, tilt {TILT:+.2f}")
    peak_a = (DIP + RISE) * math.pi ** 2 / (2.0 * T_SNAP ** 2)
    print(f"  anchors    peak {peak_anchor_v:.2f} m/s, "
          f"peak accel {peak_a:.0f} m/s^2 ({peak_a / 9.81:.1f} g)")
    if t_contact is None:
        print("  RESULT     never caught the ball")
    elif launch_v is None:
        print(f"  RESULT     caught at t={t_contact:.3f} s but NEVER LAUNCHED")
    else:
        speed = float(np.linalg.norm(launch_v))
        print(f"  caught     t={t_contact:.3f} s at {abs(v_impact[0]):.2f} m/s, "
              f"cradled {1000 * (t_snap - t_contact):.0f} ms before the snap")
        print(f"  launch     {speed:.2f} m/s  "
              f"(vx {launch_v[0]:+.2f}, vy {launch_v[1]:+.2f}, vz {launch_v[2]:+.2f})")
        print(f"  apex       {apex:.2f} m  ({apex - SHEET_Y:+.2f} m above the sheet)")
        if landing is not None:
            print(f"  landing    x {landing[0]:+.2f}  z {landing[2]:+.2f} "
                  f"(range {math.hypot(float(landing[0]), float(landing[2])):.2f} m)")
    verdict = "OK" if peak_strain < 0.10 else ("STRETCHY" if peak_strain < 0.5 else "DETONATED")
    print(f"  max strain {peak_strain:.3f}  -> {verdict}   "
          f"at grid ({peak_at[0]},{peak_at[1]}) of {N}, t={peak_at[2]:.3f} s")


# ---- run ----------------------------------------------------------------------

TOTAL = int(SECONDS * FPS)

if BENCH:
    # The sim alone: kernel launches only, no per-substep .numpy() (each of those
    # is a device sync, and the trigger logic does one or two PER SUBSTEP -- at 48
    # substeps that is ~50 stalls a frame purely for diagnostics).
    import time
    def raw_frame():
        for _ in range(SUBSTEPS):
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
                              inputs=[a, b, im, N, N, REST, bpred, BALL_R + 0.012,
                                      anchor_tgt, lra, impulse, M_PARTICLE, 0.35])
                    a, b = b, a
                wp.copy(pos, a)
            wp.launch(ball_finish, dim=1, device=device,
                      inputs=[bp, bv, impulse, DT, 1.0 / BALL_KG, BALL_R])
        wp.launch(compute_normals, dim=V, device=device, inputs=[pos, nrm, N, N])
    for _ in range(20):
        raw_frame()
    wp.synchronize()
    t0 = time.perf_counter()
    for _ in range(BENCH):
        raw_frame()
    wp.synchronize()
    sim_ms = 1000.0 * (time.perf_counter() - t0) / BENCH
    # And what one host readback per frame costs on top (the renderer needs two).
    t0 = time.perf_counter()
    for _ in range(BENCH):
        raw_frame()
        pos.numpy(); nrm.numpy()
    wp.synchronize()
    rb_ms = 1000.0 * (time.perf_counter() - t0) / BENCH
    print(f"  {V} particles, {SUBSTEPS} substeps x {ITERS} iters "
          f"= {SUBSTEPS * (ITERS + 3) + 1} launches/frame")
    print(f"  sim only          {sim_ms:6.2f} ms/frame  ({1000 / sim_ms:5.0f} fps)")
    print(f"  + geometry upload {rb_ms:6.2f} ms/frame  ({1000 / rb_ms:5.0f} fps)")
    sys.exit(0)

if TUNE:
    print(f"cloth toss: {V} particles on {device}, "
          f"{SUBSTEPS * ITERS} solves/frame, {TOTAL} frames")
    for _ in range(TOTAL):
        step_frame()
    report()
    sys.exit(0)

HEADLESS = bool(SHOT or CLIP or SEQ)
canvas = tp.Canvas("threepp x warp - cloth toss", width=1280, height=800,
                   antialiasing=4, headless=HEADLESS)
renderer = tp.GLRenderer(canvas)
renderer.shadow_map_enabled = True

scene = tp.Scene()
scene.background = 0x1b2026
camera = tp.PerspectiveCamera(50, canvas.aspect(), 0.1, 100)
camera.position.set(2.9, 1.9, 3.4)
camera.look_at(0, 1.3, 0)
studio_lights(scene)
ground_plane(scene)

geometry = tp.PlaneGeometry(CLOTH, CLOTH, N, N)
cloth_mesh = tp.Mesh(geometry, standard_material(0xd4542e, 0.88, side=tp.Side.Double))
cloth_mesh.cast_shadow = True
scene.add(cloth_mesh)

ball_mesh = tp.Mesh(tp.SphereGeometry(BALL_R, 32, 24), standard_material(0x8fa6c4, 0.35, 0.45))
ball_mesh.cast_shadow = True
scene.add(ball_mesh)

# Stand-ins for the end-effectors the real demo will bolt on.
anchor_meshes = []
for _ in range(4):
    m = tp.Mesh(tp.SphereGeometry(0.035, 16, 12), standard_material(0x22262b, 0.5, 0.3))
    scene.add(m)
    anchor_meshes.append(m)


def refresh():
    step_frame()
    geometry.update_attribute("position", pos.numpy())
    geometry.update_attribute("normal", nrm.numpy())
    p = bp.numpy()[0]
    ball_mesh.position.set(float(p[0]), float(p[1]), float(p[2]))
    a = anchor_tgt.numpy()
    for k, m in enumerate(anchor_meshes):
        m.position.set(float(a[k][0]), float(a[k][1]), float(a[k][2]))


if CLIP or SEQ:
    import shutil
    import subprocess
    import tempfile
    from warp_common import find_ffmpeg
    outdir = SEQ or os.path.join(tempfile.mkdtemp(prefix="clothtoss_"), "f")
    os.makedirs(os.path.dirname(outdir) or ".", exist_ok=True)
    for i in range(TOTAL):
        refresh()
        renderer.render(scene, camera)
        renderer.save_frame(f"{outdir}_{i:04d}.png")
    report()
    print(f"  wrote {TOTAL} frames at {outdir}_*.png")
    if CLIP:
        ff = find_ffmpeg()
        if ff is None:
            print("  no ffmpeg on PATH; frames kept")
        else:
            subprocess.run([ff, "-y", "-loglevel", "error", "-framerate", "60",
                            "-i", f"{outdir}_%04d.png", "-an", "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium",
                            "-movflags", "+faststart", CLIP], check=True)
            print(f"  wrote {CLIP}")
            if not SEQ:
                shutil.rmtree(os.path.dirname(outdir), ignore_errors=True)
elif SHOT:
    for _ in range(TOTAL):
        refresh()
    renderer.render(scene, camera)
    renderer.save_frame(SHOT)
    report()
    print(f"  wrote {SHOT}")
else:
    print("cloth toss: close the window to stop")
    frames = [0]

    def animate():
        refresh()
        frames[0] += 1
        if frames[0] == TOTAL:
            report()
        renderer.render(scene, camera)
        controls.update()

    controls = tp.OrbitControls(camera, canvas)
    controls.enable_damping = True
    controls.target.set(0, 1.3, 0)
    canvas.animate(animate)
