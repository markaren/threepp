"""TP-1 on a tilt bed: a lunar-rover-class vehicle drives up loose sand in GPU grains.

A four-wheeled, VIPER-class rover of our own design (rover.glb in this folder,
its numbers in rover_spec.json; the .glb is not in git, build it once with
build_rover_blender.py) drives up a 9 x 3 m sand bed that tilts like
NASA's tilt-bed rig. The sand under each wheel is MLS-MPM grains
(threepp.granular_mpm.GranularPatches: four moving windows over a heightfield
far field, one CUDA graph a frame); the wheels are carried by the grains, dig
ruts, and slip more as the bed tilts up. No PhysX: the rover's rigid-body
dynamics are here, in numpy.

    PYTHONPATH=python python python/examples/rover/warp_rover_demo.py             # interactive
    ... warp_rover_demo.py --slope 15                                        # start tilted
    ... warp_rover_demo.py --shot                                            # headless stills
    ... warp_rover_demo.py --record 14                                       # headless mp4
    ... warp_rover_demo.py --bench                                           # ms/frame

    W / S    commanded wheel speed up / down       SPACE  stop (omega = 0)
    A / D    skid-steer                            UP / DOWN  tilt the bed
    C        chase <-> lab camera                  R      reset the rover and the sand
    F        save a frame

Outputs go under --out (default out/demo beside this script).

The coupling, per 1/60 s frame (the same architecture as warp_mudsnow_drive.py's
gravel lane, minus PhysX):

  1. Each wheel's load W_i = its leg spring force + the wheel's own weight goes to
     the grains as the FREE_Y load, with the hub's in-plane pose/velocity and spin.
  2. The grains return the hub height (the wheel sinks until they carry W), the
     soil force on the wheel (drawbar pull along the heading, lateral force) and
     the axle moment T.
  3. The legs' spring-dampers (hub height vs chassis) and the in-plane soil forces
     at the hubs act on the chassis, a rigid 6-DOF body, integrated in substeps.
     The in-plane soil force is linearised about the frame's commanded hub
     velocity with a damping C (implicit): a light body on a stiff slip curve
     integrated explicitly at 60 Hz rings; C moves the transient, not the
     equilibrium.
  4. Wheel spin: J w' = tau_motor - T, a PI speed controller with a torque limit
     (also implicit in the P gain and in the soil's dT/dw).

The tilt: gravity in the BED frame is rotated by theta, the same vector for the
grains and the rover. The whole bed + rover + sand group sits under a parent
rotated by theta, so the world stays level and the bed looks tilted.

TODO (v1 simplifications): the hub x stays at nominal (the swing arm's
wheelbase change is drawn, not simulated); a leg past full extension does not
lift its wheel off the sand; the wheel is the grouser ENVELOPE (a cylinder at
r + h_g with mu = tan(phi)), the lugs are drawn but not simulated.
"""
import json
import math
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))   # python/ (threepp)
sys.path.insert(0, os.path.dirname(_HERE))                    # examples/ (warp_common)

import numpy as np
import warp as wp

import threepp as tp
from threepp import granular_mpm as gm
from warp_common import Encoder, cli_arg, grain_detail_texture

# =============================================================================
# PHYSICS -- every material / wheel / rover number the sim uses. Phase-A tuning
# drops its values in HERE; nothing below hard-codes a physical constant.
# =============================================================================
PHYSICS = dict(
    # --- soil: loose GRC-1-like sand, Drucker-Prager + a BOUNDED compaction cap
    #     (phase-A tuning, python/examples/rover/warp_rover_slope.py "bcap") ---
    h=0.030,                # MPM cell, m
    E=1.0e6,                # Young's modulus, Pa
    nu=0.3,
    rho=1650.0,             # kg/m^3
    phi_deg=38.0,           # friction angle
    cohesion=0.0,
    cap_p0=1.0e3,           # compaction cap: p_c = p0 exp(-e_vp / lambda) <= p_max
    cap_lambda=0.025,
    cap_pmax=5.0e3,         # the bound (sets the bearing pressure); 0 = unbounded
    ppc_spacing=0.5,        # particle spacing / h (8 per cell)
    cfl=0.5,
    patch=(0.6, 0.4),       # moving-window interior (x, z), m
    depth=0.30,             # sand depth in the bin, m (0.20: a digging wheel reached the floor)
    # --- wheel in the grains: the grouser ENVELOPE (effective radius) ---
    r_eff=0.275,            # rim 0.25 + grouser 0.025
    half_width=0.10,
    mu_wheel=math.tan(math.radians(38.0)),   # soil-on-soil: tan(phi) = 0.781
    jh_k=3.0,               # slip-mobilised friction build-up (0 = plain Coulomb)
    wheel_dof_mass=25.0,    # the grains' vertical wheel DOF inertia, kg
    wheel_dof_damp=2500.0,  # ...and its damper, N s/m
    # --- soil back-end ---
    #   "lane":    ONE GranularMPM box over the rover's lane, wheels = LUGGED (the
    #              real 24 grousers, inflated to one cell thick, Coulomb rim and
    #              lug faces). The lugs scoop: a stalled wheel DIGS (dig test:
    #              225 mm in 12 s at 100 % slip, vs 38 mm and stop for the
    #              envelope). ~10x the cost of the patches.
    #   "patches": GranularPatches (4 moving windows), the grouser ENVELOPE
    #              cylinder: real time, but a stalled wheel only polishes its rut.
    soil="patches",
    lugs=1,                 # patches: GranularPatches.set_lugs (the grousers in the grains)
    rim=0.25,               # lugged wheel: set_wheels' radius is the RIM; lugs stand lug_h past it
    lug_h=0.025,
    lug_mu=0.5,             # rim + lug-face Coulomb friction (DP/N 0.14/0.35/0.59 @ s=0.1/0.3/0.6)
    lug_n=8,                # PHYSICS grousers: 8 x 30 mm conserves the real 24 x 10 mm lug volume
    lug_t=0.030,            #   (24 inflated to 30 mm dug ~3x too fast); all 24 are DRAWN
    lane_x=(-4.5, 2.0),     # the lane's grains (bed frame), m
    lane_z=(-0.85, 0.85),
    # --- rover (rover_spec.json supplies geometry, masses and CoM) ---
    leg_static_defl=0.04,   # static spring deflection, m (sets k)
    leg_zeta=0.7,           # leg damping ratio
    soil_c=1500.0,          # implicit in-plane soil damping per wheel, N s/m
    wheel_J=0.6,            # wheel + reflected drive inertia, kg m^2
    motor_kp=40.0,          # PI speed loop, N m s/rad
    motor_ki=150.0,         # N m/rad
    motor_tau_max=70.0,     # torque limit per wheel, N m (above the soil's ~mu W r)
    soil_ct=40.0,           # implicit dT/domega of the soil, N m s
    chassis_substeps=4,
    g=9.81,
)
P_ = PHYSICS
if "--P" in sys.argv:
    for _kv in cli_arg("--P", "", str).split(","):
        _k, _v = _kv.split("=")
        assert _k in PHYSICS, _k
        if isinstance(PHYSICS[_k], str):
            PHYSICS[_k] = _v
        elif not isinstance(PHYSICS[_k], tuple):
            PHYSICS[_k] = type(PHYSICS[_k])(float(_v))

OUT = cli_arg("--out", os.path.join(_HERE, "out", "demo"), str)
os.makedirs(OUT, exist_ok=True)
SHOT = "--shot" in sys.argv
def _record_arg():
    if "--record" not in sys.argv:
        return 0.0
    i = sys.argv.index("--record")
    try:
        return float(sys.argv[i + 1])
    except (IndexError, ValueError):
        return 1.0e9            # the whole story


RECORD = _record_arg()
BENCH = "--bench" in sys.argv
DRY = "--dry" in sys.argv          # the story's physics only: no window, no frames
HEADLESS = SHOT or RECORD > 0 or BENCH or DRY or "--preview" in sys.argv
SLOPE0 = cli_arg("--slope", 0.0, float)
TH_FROM = cli_arg("--tilt-from", 10.0, float)      # the scripted tilt starts here
if RECORD > 0 or (SHOT and "--tilt" in sys.argv):
    SLOPE0 = TH_FROM
W_, H_ = 1280, 720
if "--size" in sys.argv:
    W_, H_ = (int(v) for v in cli_arg("--size", "1280x720", str).split("x"))
DT = 1.0 / 60.0
MC_SIGN = 1.0

SPEC = json.load(open(os.path.join(_HERE, "rover_spec.json")))
WN = ("FL", "FR", "RL", "RR")
HUBS = np.array([SPEC["hubs"][k] for k in WN], float)            # chassis frame
PIVOTS = np.array([SPEC["legs"]["pivots"][k] for k in WN], float)
ARM_L = SPEC["legs"]["arm_length"]
ARM_B = math.asin(SPEC["legs"]["nominal_extension"] / ARM_L)     # arm below horizontal at 0
FRONT = np.array([1.0, 1.0, -1.0, -1.0])
M_TOTAL = SPEC["mass"]["total"]
M_WHEEL = SPEC["mass"]["per_wheel"]
M_SPRUNG = M_TOTAL - 4.0 * M_WHEEL
COM_S = np.array(SPEC["mass"]["com"], float) * M_TOTAL / M_SPRUNG  # wheels sit at y = 0
_bx, _by, _bz = 1.52, 0.86, 1.38                                   # body envelope
I_BODY = M_SPRUNG / 12.0 * np.array([_by ** 2 + _bz ** 2, _bx ** 2 + _bz ** 2, _bx ** 2 + _by ** 2])
K_LEG = M_SPRUNG * P_["g"] / 4.0 / P_["leg_static_defl"]
C_LEG = 2.0 * P_["leg_zeta"] * math.sqrt(K_LEG * M_SPRUNG / 4.0)
STROKE = 0.18                                                      # +- about nominal

# The bed: bed-local x along the slope (uphill +x), y the bed normal, z across.
BED_X, BED_Z = (-4.5, 4.5), (-1.5, 1.5)     # 9 m of sand: the climb never runs out
DEPTH = P_["depth"]
HINGE = np.array([-4.75, 0.62, 0.0])        # world position of the tilt hinge
HINGE_LOCAL = np.array([-4.75, -0.10, 0.0])  # the hinge in the bed frame
X_START = -3.3
SAFE_X = 0.95               # the chassis origin never gets closer than this to a bin end wall


def gravity_bed(theta_deg):
    t = math.radians(theta_deg)
    return np.array([-math.sin(t), -math.cos(t), 0.0]) * P_["g"]


# --- small math -----------------------------------------------------------------

def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def qmat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def qaxis(axis, ang):
    s = math.sin(0.5 * ang)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(0.5 * ang)])


def q_from_to(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if d < -0.9999:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = np.array([c[0], c[1], c[2], 1.0 + d])
    return q / np.linalg.norm(q)


def nlerp(q0, q1, a):
    if float(np.dot(q0, q1)) < 0.0:
        q1 = -q1
    q = q0 + (q1 - q0) * a
    return q / max(float(np.linalg.norm(q)), 1e-12)


def skew(r):
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def leg_angle(delta, front):
    """Swing-arm angle (about local +Z) that puts the hub `delta` above nominal."""
    q = np.clip((delta - SPEC["legs"]["nominal_extension"]) / ARM_L, -1.0, 1.0)
    a = math.asin(q) + ARM_B
    return a if front > 0 else -a


# =============================================================================
# the sand: GranularPatches + its look
# =============================================================================

@wp.kernel
def k_tops(x: wp.array(dtype=wp.vec3), alive: wp.array(dtype=int),
           hox: float, hoz: float, hdx: float, pd: float, r: float,
           top: wp.array2d(dtype=float)):
    """The live grains' surface with the SAME estimator the core's write-back
    uses (granular_mpm `retire`): every live grain splats its top y + pd/2 into
    the heightfield cells within `r` = P.splat_r of it, max. A cell then has the
    same height the moment it leaves the patch as it had inside it."""
    p = wp.tid()
    if alive[p] == 0:
        return
    xp = x[p]
    t = xp[1] + 0.5 * pd
    i0 = int(wp.floor((xp[0] - r - hox) / hdx))
    j0 = int(wp.floor((xp[2] - r - hoz) / hdx))
    n = int(2.0 * r / hdx) + 2
    for a in range(n):
        for b in range(n):
            ci = i0 + a
            cj = j0 + b
            if ci >= 0 and cj >= 0 and ci < top.shape[0] and cj < top.shape[1]:
                dx = hox + (float(ci) + 0.5) * hdx - xp[0]
                dz = hoz + (float(cj) + 0.5) * hdx - xp[2]
                if dx * dx + dz * dz <= r * r:
                    wp.atomic_max(top, ci, cj, t)


@wp.kernel
def k_pitfill(top: wp.array2d(dtype=float), thr: float, out: wp.array2d(dtype=float)):
    """The write-back's pit repair (fill_dips), approximately: a live cell more
    than `thr` below its valid neighbours' mean takes that mean."""
    i, j = wp.tid()
    t = top[i, j]
    out[i, j] = t
    if t <= 0.0:
        return
    acc = float(0.0)
    n = float(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            if di != 0 or dj != 0:
                ii = i + di
                jj = j + dj
                if ii >= 0 and jj >= 0 and ii < top.shape[0] and jj < top.shape[1]:
                    v = top[ii, jj]
                    if v > 0.0:
                        acc += v
                        n += 1.0
    if n > 0.0:
        m = acc / n
        if t < m - thr:
            out[i, j] = m


@wp.kernel
def k_surface(H: wp.array2d(dtype=float), hox: float, hoz: float, hdx: float,
              top: wp.array2d(dtype=float), pc: wp.array2d(dtype=int), marg: int, h: float,
              ix: int, iz: int, ramp: float, settle: float, touched: wp.array2d(dtype=int),
              pos: wp.array(dtype=wp.vec3)):
    """The bed's ONE surface. Inside a live patch: the grains' own top (k_tops,
    the write-back's estimator), unblended -- blending toward the far field there
    pulls a rut up to the STALE pre-patch height at the trailing edge. Outside:
    the far-field heightfield, blended toward the nearest live edge value over
    `ramp`, so the leading edge and the sides meet the far field without a step.

    The patches' grains settle a few mm under their own weight (the compaction
    cap) and the far field -- a heightfield -- does not, so sand no patch has
    touched yet is DRAWN `settle` lower (measured at start-up, Sand.measure_settle)."""
    i, j = wp.tid()
    x = hox + (float(i) + 0.5) * hdx
    z = hoz + (float(j) + 0.5) * hdx
    yf = H[i, j] - settle * float(1 - touched[i, j])
    y = yf
    wx = float(ix) * h
    wz = float(iz) * h
    best = float(0.0)
    for k in range(pc.shape[0]):
        lx = float(pc[k, 0] + marg) * h
        lz = float(pc[k, 1] + marg) * h
        ex = x - lx
        ez = z - lz
        if ex >= 0.0 and ex < wx and ez >= 0.0 and ez < wz:
            touched[i, j] = 1
            t = top[i, j]
            if t > 0.0:
                y = t
                best = 2.0
        elif best < 2.0:
            ddx = wp.max(wp.max(-ex, ex - wx), 0.0)
            ddz = wp.max(wp.max(-ez, ez - wz), 0.0)
            d = wp.sqrt(ddx * ddx + ddz * ddz)
            if d < ramp:
                xi = wp.clamp(x, lx + 0.5 * hdx, lx + wx - 0.5 * hdx)
                zi = wp.clamp(z, lz + 0.5 * hdx, lz + wz - 0.5 * hdx)
                ii = wp.clamp(int(wp.floor((xi - hox) / hdx)), 0, top.shape[0] - 1)
                jj = wp.clamp(int(wp.floor((zi - hoz) / hdx)), 0, top.shape[1] - 1)
                t = top[ii, jj]
                if t > 0.0:
                    a = 1.0 - d / ramp
                    a = a * a * (3.0 - 2.0 * a)
                    if a > best:
                        best = a
                        y = yf * (1.0 - a) + t * a
    pos[i * H.shape[1] + j] = wp.vec3(x, y, z)


@wp.kernel
def k_normals(pos: wp.array(dtype=wp.vec3), nx: int, nz: int, nrm: wp.array(dtype=wp.vec3)):
    i, j = wp.tid()
    i0 = wp.max(i - 1, 0)
    i1 = wp.min(i + 1, nx - 1)
    j0 = wp.max(j - 1, 0)
    j1 = wp.min(j + 1, nz - 1)
    dx = pos[i1 * nz + j] - pos[i0 * nz + j]
    dz = pos[i * nz + j1] - pos[i * nz + j0]
    n = wp.cross(dz, dx)
    nrm[i * nz + j] = wp.normalize(n)


EDGE_RAMP = 0.085         # m, far field -> live edge blend (~3 cells)
PRINT_PHI = [np.zeros(4)]  # the drawn wheels' spin angles, set by present()
VTOP = 2.0
DETAIL_REPEAT = cli_arg("--detail-repeat", 12.0, float)
DETAIL_SCALE = cli_arg("--detail-scale", 0.55, float)


class Sand:
    """Four MPM patches (one per wheel) over the bed's heightfield, and the look."""

    def __init__(self, theta_deg, device="cuda:0"):
        self.dev = device
        self.theta = None
        self.mat = gm.Material(E=P_["E"], nu=P_["nu"], rho=P_["rho"], phi_deg=P_["phi_deg"],
                               cohesion=P_["cohesion"], cap_p0=P_["cap_p0"],
                               cap_lambda=P_["cap_lambda"], cap_pmax=P_["cap_pmax"])
        self._make(theta_deg)
        sim = self.sim
        xs, zs = self._display_coords()
        hnx, hnz = len(xs), len(zs)
        self.hnx, self.hnz = hnx, hnz
        self.top = wp.zeros((hnx, hnz), dtype=float, device=device)
        self.touched = wp.zeros((hnx, hnz), dtype=int, device=device)
        self.top2 = wp.zeros((hnx, hnz), dtype=float, device=device)
        self.settle = 0.0                  # measured in measure_settle()
        self.pos = wp.zeros(hnx * hnz, dtype=wp.vec3, device=device)
        self.nrm = wp.zeros(hnx * hnz, dtype=wp.vec3, device=device)
        # far-field mesh
        g = tp.BufferGeometry()
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        p0 = np.stack([X, np.full_like(X, DEPTH), Z], -1).reshape(-1, 3).astype(np.float32)
        g.set_attribute("position", p0)
        g.set_attribute("normal", np.tile(np.float32([0, 1, 0]), (p0.shape[0], 1)))
        ii, jj = np.meshgrid(np.arange(hnx - 1), np.arange(hnz - 1), indexing="ij")
        a = (ii * hnz + jj).ravel()
        b = a + 1
        c = a + hnz
        d = c + 1
        idx = np.stack([a, b, c, b, d, c], -1).ravel().astype(np.uint32)
        g.set_index(idx)
        self.hf_geom = g
        m = tp.MeshPhysicalMaterial()
        m.color = 0x8f8574
        m.roughness = 0.95
        m.specular_intensity = 0.15
        self.material = m
        self.hf_mesh = tp.Mesh(g, m)
        self.hf_mesh.receive_shadow = True
        self.hf_mesh.cast_shadow = True
        self.hf_mesh.frustum_culled = False
        if hasattr(m, "detail_normal_map"):
            grain = tp.data_texture(grain_detail_texture(256, 144), srgb=False)
            m.detail_normal_map = grain
            m.detail_repeat = DETAIL_REPEAT
            m.detail_normal_scale = DETAIL_SCALE
        self.out = np.zeros((4, 16), np.float32)
        self.ms = 0.0
        self._describe()

    def _display_coords(self):
        return self.sim.hf_coords()

    def grain_arrays(self):
        return self.sim.x, self.sim.v, self.sim.alive, self.sim.n_slots

    def hubs_inside(self):
        """Every hub over its own patch's interior (the numerical guard)."""
        sim, o = self.sim, self.out
        lo = (o[:, gm.OUT_BLOCK] + sim.marg) * sim.h
        hx, hz = o[:, 8], o[:, 10]
        ok = (hx >= lo[:, 0]) & (hx <= lo[:, 0] + sim.ix * sim.h) & (hz >= lo[:, 1]) & (hz <= lo[:, 1] + sim.iz * sim.h)
        return bool(ok.all()) or not o.any()

    def _describe(self):
        sim = self.sim
        if P_["lugs"]:
            print(f"  wheels: LUGGED, {P_['lug_n']} grousers x {1000 * max(P_['lug_t'], P_['h']):.0f} mm, "
                  f"{1000 * P_['lug_h']:.0f} mm tall on a {1000 * P_['rim']:.0f} mm rim, mu {P_['lug_mu']}",
                  flush=True)
        print(f"  sand: 4 MLS-MPM patches, {sim.n_slots:,} slots, h = {P_['h'] * 1000:.0f} mm, "
              f"{P_['patch'][0]:.2f} x {P_['patch'][1]:.2f} x {DEPTH:.2f} m each; "
              f"heightfield {self.hnx}x{self.hnz} @ {sim.hf_dx * 1000:.0f} mm", flush=True)

    def _make(self, theta_deg):
        self.sim = gm.GranularPatches(
            self.mat, P_["h"], 4, patch_size=P_["patch"], depth=DEPTH,
            hf_origin=(BED_X[0], BED_Z[0]), hf_size=(BED_X[1] - BED_X[0], BED_Z[1] - BED_Z[0]),
            floor_y=0.0, device=self.dev, ppc_spacing=P_["ppc_spacing"], cfl=P_["cfl"],
            jh_k=P_["jh_k"], gravity=tuple(gravity_bed(theta_deg)))
        if P_["lugs"]:
            self.sim.set_lugs(P_["lug_n"], P_["lug_h"], max(P_["lug_t"], P_["h"]))
        self.theta = theta_deg
        self.g_moving = False          # gravity changed since the last frame
        self.g_stale = False           # captured graphs hold an old gravity
        self.nsub = self.sim.nsub_for(2.0)
        self.nsub += self.nsub & 1

    def set_slope(self, theta_deg):
        """New gravity, continuous (no quantisation), every frame the tilt moves.
        The grid kernel reads gravity from a device array (set_gravity), so the
        captured frame graph stays valid through a moving tilt. Older cores baked
        P.grav into the graph by value: there the frame runs eagerly while tilting
        and the stale graphs are re-captured once it rests."""
        if self.theta is not None and abs(theta_deg - self.theta) < 1e-9:
            return
        g = gravity_bed(theta_deg)
        self.theta = theta_deg
        if hasattr(self.sim, "set_gravity"):
            self.sim.set_gravity(g)
            return
        self.sim.gravity = g
        self.sim.P.grav = wp.vec3(*g)
        self.g_moving = True

    def step(self, pos, vel, omega, axis, load, hover=False, set_y=False):
        t0 = time.perf_counter()
        sim = self.sim
        lugged = bool(P_["lugs"])
        sim.set_wheels(pos, vel, omega, axis, load, P_["rim"] if lugged else P_["r_eff"],
                       P_["half_width"], mu=P_["lug_mu"] if lugged else P_["mu_wheel"],
                       mass=P_["wheel_dof_mass"], damp=P_["wheel_dof_damp"],
                       free_y=not hover, set_y=set_y or hover)
        if self.g_moving:
            self.out = sim.step_frame(self.nsub, graph=False)
            self.g_moving, self.g_stale = False, True
        else:
            if self.g_stale:
                sim._graphs.clear()        # re-capture with the resting gravity
                self.g_stale = False
            self.out = sim.step_frame(self.nsub)
        self.ms = (time.perf_counter() - t0) * 1000.0
        return self.out

    def _prints(self):
        """Stamp and apply the grouser prints (display only; PRINT_AMP 0 = off)."""
        if PRINT_AMP <= 0.0 or not self.out.any():
            return
        dev = self.dev
        if not hasattr(self, "stamp"):
            self.stamp = wp.zeros((self.hnx, self.hnz), dtype=float, device=dev)
            self.p_hub = wp.zeros(4, dtype=wp.vec3, device=dev)
            self.p_phi = wp.zeros(4, dtype=float, device=dev)
        self.p_hub.assign(np.asarray(self.out[:, 8:11], np.float32))
        self.p_phi.assign(np.asarray(PRINT_PHI[0], np.float32))
        wp.launch(k_prints, dim=(self.hnx, self.hnz), device=dev,
                  inputs=[self.pos, self.hnz, self.stamp, self.p_hub, self.p_phi, P_["r_eff"],
                          P_["half_width"], PRINT_AMP, math.radians(2.6), math.radians(15.0)])
        wp.launch(k_apply_stamp, dim=(self.hnx, self.hnz), device=dev,
                  inputs=[self.pos, self.hnz, self.stamp])

    def _live_tops(self):
        sim, dev = self.sim, self.dev
        self.top.fill_(-1.0)
        wp.launch(k_tops, dim=sim.n_slots, device=dev,
                  inputs=[sim.x, sim.alive, sim.hf_origin[0], sim.hf_origin[1], sim.hf_dx,
                          sim.pd, float(sim.P.splat_r), self.top])
        wp.launch(k_pitfill, dim=(self.hnx, self.hnz), device=dev,
                  inputs=[self.top, float(sim.P.dip_thr), self.top2])

    def measure_settle(self):
        """Mean (far field - live top) over the patches' interiors, taken once the
        grains have settled under their own weight with the wheels still hovering."""
        self._live_tops()
        t, Hh = self.top2.numpy(), self.sim.hf.numpy()
        xs, zs = self.sim.hf_coords()
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        sel = np.zeros(t.shape, bool)
        for k in range(4):
            xlo, xhi, zlo, zhi = self.sim.block_interior(k)
            sel |= (X > xlo + 0.03) & (X < xhi - 0.03) & (Z > zlo + 0.03) & (Z < zhi - 0.03)
        sel &= t > 0.0
        if sel.sum() > 20:
            self.settle = float(np.mean(Hh[sel] - t[sel]))
        print(f"  sand: virgin-bed settle under self-weight {1000 * self.settle:+.1f} mm "
              f"(drawn offset of untouched sand; {int(sel.sum())} cells)", flush=True)

    def publish(self):
        """The bed's surface mesh (host route: ~80k vertices, one copy)."""
        sim, dev = self.sim, self.dev
        self._live_tops()
        wp.launch(k_surface, dim=(self.hnx, self.hnz), device=dev,
                  inputs=[sim.hf, sim.hf_origin[0], sim.hf_origin[1], sim.hf_dx, self.top2, sim.pc,
                          sim.marg, sim.h, sim.ix, sim.iz, EDGE_RAMP, self.settle, self.touched,
                          self.pos])
        self._prints()
        wp.launch(k_normals, dim=(self.hnx, self.hnz), device=dev,
                  inputs=[self.pos, self.hnx, self.hnz, self.nrm])
        self.hf_geom.update_attribute("position", self.pos.numpy())
        self.hf_geom.update_attribute("normal", self.nrm.numpy())


@wp.kernel
def k_surface_lane(top: wp.array2d(dtype=float), ox: float, oz: float, dx: float,
                   lx0: float, lx1: float, lz0: float, lz1: float, far: float, ramp: float,
                   pos: wp.array(dtype=wp.vec3)):
    """The bed's surface over the lane: the grains' splat tops inside, the static
    far field outside, blended toward the lane's edge value over `ramp`."""
    i, j = wp.tid()
    x = ox + (float(i) + 0.5) * dx
    z = oz + (float(j) + 0.5) * dx
    y = far
    xi = wp.clamp(x, lx0 + 0.5 * dx, lx1 - 0.5 * dx)
    zi = wp.clamp(z, lz0 + 0.5 * dx, lz1 - 0.5 * dx)
    ddx = x - xi
    ddz = z - zi
    d = wp.sqrt(ddx * ddx + ddz * ddz)
    ii = wp.clamp(int(wp.floor((xi - ox) / dx)), 0, top.shape[0] - 1)
    jj = wp.clamp(int(wp.floor((zi - oz) / dx)), 0, top.shape[1] - 1)
    t = top[ii, jj]
    if t > 0.0 and d < ramp:
        a = 1.0 - d / ramp
        a = a * a * (3.0 - 2.0 * a)
        y = far * (1.0 - a) + t * a
    pos[i * top.shape[1] + j] = wp.vec3(x, y, z)


@wp.func
def speed_color(s: float) -> wp.vec3:
    """A turbo-like ramp, s in [0, 1]: blue, cyan, green, yellow, red."""
    s = wp.clamp(s, 0.0, 1.0)
    c0 = wp.vec3(0.08, 0.10, 0.45)
    c1 = wp.vec3(0.05, 0.55, 0.95)
    c2 = wp.vec3(0.20, 0.90, 0.35)
    c3 = wp.vec3(1.00, 0.80, 0.08)
    c4 = wp.vec3(0.95, 0.12, 0.06)
    u = s * 4.0
    if u < 1.0:
        return c0 + (c1 - c0) * u
    if u < 2.0:
        return c1 + (c2 - c1) * (u - 1.0)
    if u < 3.0:
        return c2 + (c3 - c2) * (u - 2.0)
    return c3 + (c4 - c3) * (u - 3.0)


@wp.kernel
def k_gather(x: wp.array(dtype=wp.vec3), v: wp.array(dtype=wp.vec3), alive: wp.array(dtype=int),
             mode: int, zc: float,
             half: float, vmin: float, vscale: float, cnt: wp.array(dtype=int),
             outp: wp.array(dtype=wp.vec3), outc: wp.array(dtype=wp.vec3), nmax: int):
    """mode 0: the grains that MOVE (|v| > vmin), sand-coloured; mode 1: every
    grain in the slab |z - zc| < half, coloured by speed (the X-ray)."""
    p = wp.tid()
    if alive[p] == 0:
        return
    xp = x[p]
    sp = wp.length(v[p])
    c = wp.vec3(0.0)
    if mode == 0:
        if sp < vmin:
            return
        c = wp.vec3(0.62, 0.55, 0.43)
    else:
        if wp.abs(xp[2] - zc) > half:
            return
        c = speed_color(sp / vscale)
    k = wp.atomic_add(cnt, 0, 1)
    if k < nmax:
        outp[k] = xp
        outc[k] = c


GRAIN_MAX = 120_000
XRAY_VSCALE = 0.15          # m/s at the red end of the X-ray ramp


class GrainPoints:
    """Real MPM grains drawn as points: moving grains, or the X-ray slab."""

    def __init__(self, dev="cuda:0"):
        self.dev = dev
        self.cnt = wp.zeros(1, dtype=int, device=dev)
        self.p = wp.zeros(GRAIN_MAX, dtype=wp.vec3, device=dev)
        self.c = wp.zeros(GRAIN_MAX, dtype=wp.vec3, device=dev)
        g = tp.BufferGeometry()
        g.set_attribute("position", np.zeros((GRAIN_MAX, 3), np.float32))
        g.set_attribute("color", np.ones((GRAIN_MAX, 3), np.float32))
        g.set_draw_range(0, 0)
        m = tp.PointsMaterial()
        m.size = 0.010
        m.size_attenuation = True
        m.vertex_colors = True
        self.geom, self.mat = g, m
        self.points = tp.Points(g, m)
        self.points.frustum_culled = False
        self.n = 0

    def update(self, sand_, mode, zc=0.0, half=0.03, vmin=0.05, size=0.010):
        self.cnt.zero_()
        x, v, alive, n = sand_.grain_arrays()
        wp.launch(k_gather, dim=n, device=self.dev,
                  inputs=[x, v, alive, int(mode), float(zc), float(half), float(vmin), XRAY_VSCALE,
                          self.cnt, self.p, self.c, GRAIN_MAX])
        n = min(int(self.cnt.numpy()[0]), GRAIN_MAX)
        self.n = n
        if n > 0:
            self.geom.update_attribute("position", self.p.numpy())
            self.geom.update_attribute("color", self.c.numpy())
        self.geom.set_draw_range(0, n)
        self.mat.size = size


@wp.kernel
def k_blur_tops(src: wp.array2d(dtype=float), dst: wp.array2d(dtype=float)):
    """Display only: 1-2-1 binomial smoothing of the valid splat tops."""
    i, j = wp.tid()
    t = src[i, j]
    dst[i, j] = t
    if t <= 0.0:
        return
    acc = float(0.0)
    wsum = float(0.0)
    for di in range(-1, 2):
        for dj in range(-1, 2):
            ii = wp.clamp(i + di, 0, src.shape[0] - 1)
            jj = wp.clamp(j + dj, 0, src.shape[1] - 1)
            v = src[ii, jj]
            if v > 0.0:
                w = float((2 - wp.abs(di)) * (2 - wp.abs(dj)))
                acc += w * v
                wsum += w
    dst[i, j] = acc / wsum


@wp.kernel
def k_prints(pos: wp.array(dtype=wp.vec3), nz: int, stamp: wp.array2d(dtype=float),
             hub: wp.array(dtype=wp.vec3), phi: wp.array(dtype=float), r_tip: float, hw: float,
             amp: float, half_ang: float, pitch: float):
    """Grouser prints (DISPLAY ONLY): wherever a wheel's grouser tips reach the
    sand, the cell takes the transverse lug profile of the DRAWN wheel at its real
    spin angle (24 lugs at 15 deg, lug k at k*15 - phi about +z, rolling +x =
    phi increasing). The last lug to leave a cell leaves its print. The physics
    has 8 fatter lugs (volume-conserving); the prints follow the 24 drawn ones."""
    i, j = wp.tid()
    q = pos[i * nz + j]
    for k in range(4):
        c = hub[k]
        dxw = q[0] - c[0]
        if wp.abs(q[2] - c[2]) < hw and wp.abs(dxw) < r_tip:
            below = wp.sqrt(r_tip * r_tip - dxw * dxw)
            if q[1] >= c[1] - below - 0.004:
                beta = wp.atan2(-below, dxw)
                u = (beta + phi[k]) / pitch
                dl = (u - wp.round(u)) * pitch
                stamp[i, j] = -amp * wp.exp(-(dl * dl) / (half_ang * half_ang))


@wp.kernel
def k_apply_stamp(pos: wp.array(dtype=wp.vec3), nz: int, stamp: wp.array2d(dtype=float)):
    i, j = wp.tid()
    q = pos[i * nz + j]
    pos[i * nz + j] = wp.vec3(q[0], q[1] + stamp[i, j], q[2])


PRINT_AMP = cli_arg("--print-amp", 0.007, float)     # m, display only


class LaneSand(Sand):
    """One GranularMPM box over the rover's lane; four LUGGED wheels in it."""

    DX = 0.015

    def _make(self, theta_deg):
        h = P_["h"]
        lx, lz = P_["lane_x"], P_["lane_z"]
        self.nsub = int(math.ceil((1.0 / 60.0) / self.mat.cfl_dt(h, P_["cfl"], vmax=3.0)))
        self.nsub += self.nsub & 1
        self.sim = gm.GranularMPM(self.mat, h, lo=(lx[0], 0.0, lz[0]), hi=(lx[1], 0.0, lz[1]),
                                  fill_height=DEPTH, dt=(1.0 / 60.0) / self.nsub, device=self.dev,
                                  ppc_spacing=P_["ppc_spacing"], headroom=0.35, max_bodies=4,
                                  gravity=tuple(gravity_bed(theta_deg)))
        r, hw, hg = P_["r_eff"] - 0.025, P_["half_width"], 0.025
        for i in range(4):
            self.sim.add_body(gm.LUGGED, (r, hw, hg), (HUBS[i, 0] + X_START, DEPTH + 1.0, HUBS[i, 2]),
                              axis=(0, 0, 1), mu=P_["lug_mu"], mode=gm.KINEMATIC,
                              lug=(max(P_["lug_t"], h), float(P_["lug_n"]), 0.0), mass=P_["wheel_dof_mass"],
                              damp=P_["wheel_dof_damp"])
        self.ones = wp.array(np.ones(self.sim.n, np.int32), dtype=int, device=self.dev)
        self.theta = theta_deg
        self.g_moving = self.g_stale = False
        self.settle = 0.0
        self.out = np.zeros((4, 16), np.float32)

    def grain_arrays(self):
        return self.sim.x, self.sim.v, self.ones, self.sim.n

    def hubs_inside(self):
        return True

    def _display_coords(self):
        n_x = int(round((BED_X[1] - BED_X[0]) / self.DX))
        n_z = int(round((BED_Z[1] - BED_Z[0]) / self.DX))
        return (BED_X[0] + (np.arange(n_x) + 0.5) * self.DX, BED_Z[0] + (np.arange(n_z) + 0.5) * self.DX)

    def _describe(self):
        print(f"  sand: ONE GranularMPM lane {P_['lane_x']} x {P_['lane_z']} m, {self.sim.n:,} grains, "
              f"h = {P_['h'] * 1000:.0f} mm, {self.nsub} substeps; 4 LUGGED wheels ({P_['lug_n']} grousers, "
              f"t -> {1000 * max(P_['lug_t'], P_['h']):.0f} mm, mu {P_['lug_mu']}); display "
              f"{self.hnx}x{self.hnz} @ {self.DX * 1000:.0f} mm", flush=True)

    def set_slope(self, theta_deg):
        if self.theta is not None and abs(theta_deg - self.theta) < 1e-9:
            return
        self.theta = theta_deg
        self.sim.set_gravity(gravity_bed(theta_deg))

    def step(self, pos, vel, omega, axis, load, hover=False, set_y=False):
        t0 = time.perf_counter()
        sim = self.sim
        cur = sim.bpos.numpy()
        curv = sim.bvel.numpy()
        ld = np.broadcast_to(np.asarray(load, float), (4,))
        for i in range(4):
            sim._set(sim.baxis, i, tuple(float(v) for v in axis[i]))
            if hover:
                sim.set_body(i, pos=tuple(float(v) for v in pos[i]), vel=(0.0, 0.0, 0.0),
                             omega=(0.0, 0.0, 0.0), mode=gm.KINEMATIC, load=0.0)
                continue
            y = float(pos[i][1]) if set_y else float(cur[i][1])
            vy = 0.0 if set_y else float(curv[i][1])
            sim.set_body(i, pos=(float(pos[i][0]), y, float(pos[i][2])),
                         vel=(float(vel[i][0]), vy, float(vel[i][2])),
                         omega=tuple(float(v) for v in omega[i]), mode=gm.FREE_Y, load=float(ld[i]))
        sim.step(self.nsub)
        f, tq = sim.take_mean_force()
        bp, bv = sim.bpos.numpy(), sim.bvel.numpy()
        out = np.zeros((4, 16), np.float32)
        out[:, gm.OUT_F] = f[:4]
        out[:, gm.OUT_T] = tq[:4]
        out[:, gm.OUT_T_AXLE] = np.einsum("ij,ij->i", tq[:4], np.asarray(axis, float))
        out[:, gm.OUT_POS] = bp[:4]
        out[:, gm.OUT_VY] = bv[:4, 1]
        self.out = out
        self.ms = (time.perf_counter() - t0) * 1000.0
        return out

    def _live_tops(self):
        sim, dev = self.sim, self.dev
        self.top.fill_(-1.0)
        wp.launch(k_tops, dim=sim.n, device=dev,
                  inputs=[sim.x, self.ones, BED_X[0], BED_Z[0], self.DX, sim.pd, float(sim.pd), self.top])
        wp.launch(k_pitfill, dim=(self.hnx, self.hnz), device=dev,
                  inputs=[self.top, 0.006, self.top2])
        wp.launch(k_blur_tops, dim=(self.hnx, self.hnz), device=dev, inputs=[self.top2, self.top])
        wp.copy(self.top2, self.top)

    def measure_settle(self):
        self._live_tops()
        t = self.top2.numpy()
        xs, zs = self._display_coords()
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        lx, lz = P_["lane_x"], P_["lane_z"]
        sel = (X > lx[0] + 0.1) & (X < lx[1] - 0.1) & (Z > lz[0] + 0.1) & (Z < lz[1] - 0.1) & (t > 0)
        if sel.sum() > 20:
            self.settle = float(DEPTH - np.mean(t[sel]))
        print(f"  sand: the lane's grains settled {1000 * self.settle:+.1f} mm under self-weight "
              f"(the static far field is drawn that much lower)", flush=True)

    def publish(self):
        dev = self.dev
        self._live_tops()
        lx, lz = P_["lane_x"], P_["lane_z"]
        wp.launch(k_surface_lane, dim=(self.hnx, self.hnz), device=dev,
                  inputs=[self.top2, BED_X[0], BED_Z[0], self.DX, lx[0], lx[1], lz[0], lz[1],
                          DEPTH - self.settle, EDGE_RAMP, self.pos])
        self._prints()
        wp.launch(k_normals, dim=(self.hnx, self.hnz), device=dev,
                  inputs=[self.pos, self.hnx, self.hnz, self.nrm])
        self.hf_geom.update_attribute("position", self.pos.numpy())
        self.hf_geom.update_attribute("normal", self.nrm.numpy())


# =============================================================================
# the rover: a rigid chassis on four spring-damper legs, wheels in the grains
# =============================================================================

class Rover:
    def __init__(self):
        self.reset(X_START)

    def reset(self, x0, z0=0.0, yaw=0.0):
        # state at the SPRUNG CoM, bed frame
        self.q = qaxis((0.0, 1.0, 0.0), yaw)
        origin = np.array([x0, DEPTH + P_["r_eff"] + 0.03, z0])
        self.com = origin + qmat(self.q) @ COM_S
        self.v = np.zeros(3)
        self.w = np.zeros(3)
        self.hub_y = np.full(4, DEPTH + P_["r_eff"] + 0.03)
        self.hub_vy = np.zeros(4)
        self.spin = np.zeros(4)            # forward spin rate, rad/s (+ = rolling +x)
        self.phi = np.zeros(4)             # accumulated spin angle
        self.integ = np.zeros(4)           # PI integrators
        self.tau = np.zeros(4)
        self.T = np.zeros(4)               # soil axle moment (+ resists forward)
        self.F = np.zeros((4, 3))          # soil force on each wheel, bed frame
        self.Fs = np.full(4, M_SPRUNG * P_["g"] / 4.0)   # leg spring forces
        self.W = np.zeros(4)
        self.delta = np.zeros(4)
        self.t = 0.0
        self.phase = "hover"               # hover -> settle -> free
        self.phase_t = 0.0
        self.power = True                  # False: motors unpowered, wheels free-wheel
        self.hit_wall = False

    # geometry helpers
    def origin(self):
        return self.com - qmat(self.q) @ COM_S

    def attach(self):
        """Nominal hub points (the leg attachment) in the bed frame, and their velocity."""
        R = qmat(self.q)
        o = self.origin()
        A = o + HUBS @ R.T
        r = A - self.com
        vA = self.v + np.cross(self.w, r)
        return A, vA, R

    def heading(self):
        R = qmat(self.q)
        f = R[:, 0].copy()
        f[1] = 0.0
        return f / max(np.linalg.norm(f), 1e-9)

    def forward_speed(self):
        return float(np.dot(self.v, self.heading()))

    def spring_forces(self, A, vA, R):
        u = R[:, 1]
        delta = (self.hub_y - A[:, 1]) * u[1]
        ddot = (self.hub_vy - vA[:, 1]) * u[1]
        F = K_LEG * (P_["leg_static_defl"] + delta) + C_LEG * ddot
        # bump stop at the end of the stroke
        over = np.maximum(delta - STROKE, 0.0)
        F += 40.0 * K_LEG * over
        return np.maximum(F, 0.0), delta

    def step(self, sand, w_cmd, g_bed):
        """One 1/60 s frame of the coupled rover + grains."""
        A, vA, R = self.attach()
        axis = R[:, 2].copy()
        axis[1] = 0.0
        axis /= max(np.linalg.norm(axis), 1e-9)
        cos_n = -g_bed[1] / P_["g"]
        Fs, _ = self.spring_forces(A, vA, R)
        pos = A.copy()
        vel = vA.copy()
        vel[:, 1] = 0.0
        omega = -self.spin[:, None] * axis[None, :]
        axes = np.tile(axis, (4, 1))
        W = Fs * R[1, 1] + M_WHEEL * P_["g"] * cos_n
        if self.phase == "hover":
            pos[:, 1] = DEPTH + P_["r_eff"] + 0.25
            out = sand.step(pos, vel * 0, omega * 0, axes, 0.0, hover=True)
            self.phase_t += DT
            if self.phase_t > 0.5:
                sand.measure_settle()
                self.phase, self.phase_t = "settle", 0.0
                self.hub_y[:] = DEPTH + P_["r_eff"] + sand.sim.eps + 0.002
                self.hub_vy[:] = 0.0
                self._set_origin_y(self.hub_y.mean())
            self.W = np.zeros(4)
            return out
        set_y = False
        if self.phase == "settle":
            ramp = min(1.0, self.phase_t / 0.3)
            W = W * ramp
            set_y = self.phase_t == 0.0
            if set_y:
                pos[:, 1] = self.hub_y
            vel[:] = 0.0
            omega[:] = 0.0
        self.W = W
        out = sand.step(pos, vel, omega, axes, W, set_y=set_y)
        self.hub_y = out[:, gm.OUT_POS][:, 1].astype(float)
        self.hub_vy = out[:, gm.OUT_VY].astype(float)
        self.F = out[:, gm.OUT_F].astype(float)
        self.T = out[:, gm.OUT_T_AXLE].astype(float)
        if self.phase == "settle":
            # the chassis rides on the settling hubs, held level; the legs at nominal
            self.phase_t += DT
            self._set_origin_y(self.hub_y.mean())
            self.v[:] = 0.0
            self.w[:] = 0.0
            if self.phase_t > 0.45:
                self.phase = "free"
            self.delta = np.zeros(4)
            return out
        self._wheels(w_cmd)
        self._chassis(vA, g_bed, axis)
        self._walls()
        self.phi += self.spin * DT
        self.t += DT
        return out

    def _walls(self):
        """Safety stop: the chassis origin stays SAFE_X from the bin's end walls
        (and off the side walls); running into one kills that velocity component.
        The rover can never leave the bed, whatever the physics does."""
        o = self.origin()
        lo, hi = BED_X[0] + SAFE_X, BED_X[1] - SAFE_X
        if o[0] < lo:
            self.com[0] += lo - o[0]
            self.v[0] = max(self.v[0], 0.0)
            self.hit_wall = True
        elif o[0] > hi:
            self.com[0] += hi - o[0]
            self.v[0] = min(self.v[0], 0.0)
            self.hit_wall = True
        zl, zh = BED_Z[0] + 0.75, BED_Z[1] - 0.75
        if o[2] < zl:
            self.com[2] += zl - o[2]
            self.v[2] = max(self.v[2], 0.0)
        elif o[2] > zh:
            self.com[2] += zh - o[2]
            self.v[2] = min(self.v[2], 0.0)

    def _set_origin_y(self, y):
        o = self.origin()
        self.com[1] += y - o[1]

    def _wheels(self, w_cmd):
        """PI speed loop with a torque limit; implicit in Kp and the soil's dT/dw."""
        J, kp, ki, tm = P_["wheel_J"], P_["motor_kp"], P_["motor_ki"], P_["motor_tau_max"]
        ct = P_["soil_ct"]
        w_sent = self.spin.copy()
        if not self.power:
            # unpowered: no torque, no brake; the wheel turns as the soil turns it
            for i in range(4):
                self.spin[i] = (J * self.spin[i] - DT * self.T[i] + DT * ct * w_sent[i]) / (J + DT * ct)
                self.tau[i] = 0.0
            return
        for i in range(4):
            e = w_cmd[i] - self.spin[i]
            self.integ[i] = float(np.clip(self.integ[i] + ki * e * DT, -tm, tm))
            w_new = (J * self.spin[i] + DT * (kp * w_cmd[i] + self.integ[i] - self.T[i])
                     + DT * ct * w_sent[i]) / (J + DT * (kp + ct))
            tau = kp * (w_cmd[i] - w_new) + self.integ[i]
            if abs(tau) > tm:
                tau = math.copysign(tm, tau)
                w_new = (J * self.spin[i] + DT * (tau - self.T[i]) + DT * ct * w_sent[i]) / (J + DT * ct)
            self.tau[i] = tau
            self.spin[i] = w_new

    def _chassis(self, vA_cmd, g_bed, axis):
        n = int(P_["chassis_substeps"])
        h = DT / n
        C = np.diag([P_["soil_c"], 0.0, P_["soil_c"]])
        Fin = self.F.copy()
        Fin[:, 1] = 0.0
        Mw_in = M_WHEEL * g_bed.copy()
        Mw_in[1] = 0.0
        for _ in range(n):
            A, vA, R = self.attach()
            Fs, delta = self.spring_forces(A, vA, R)
            u = R[:, 1]
            force = M_SPRUNG * g_bed.copy()
            torque = np.zeros(3)
            Mgen = np.zeros((6, 6))
            rhs_c = np.zeros(6)
            for i in range(4):
                hub = np.array([A[i, 0], self.hub_y[i], A[i, 2]])
                r_att = A[i] - self.com
                r_hub = hub - self.com
                fs = Fs[i] * u
                force += fs
                torque += np.cross(r_att, fs)
                fe = Fin[i] + Mw_in
                force += fe
                torque += np.cross(r_hub, fe)
                torque += self.tau[i] * axis        # drive reaction on the chassis
                Jm = np.hstack([np.eye(3), -skew(r_hub)])
                Mgen += Jm.T @ C @ Jm
                rhs_c += Jm.T @ C @ vA_cmd[i]
            self.delta = delta
            Iw = R @ np.diag(I_BODY) @ R.T
            M = np.zeros((6, 6))
            M[:3, :3] = M_SPRUNG * np.eye(3)
            M[3:, 3:] = Iw
            x = np.concatenate([self.v, self.w])
            f = np.concatenate([force, torque - np.cross(self.w, Iw @ self.w)])
            x1 = np.linalg.solve(M + h * Mgen, M @ x + h * (f + rhs_c))
            self.v, self.w = x1[:3], x1[3:]
            self.com = self.com + self.v * h
            wn = np.linalg.norm(self.w)
            if wn > 1e-12:
                self.q = qmul(qaxis(self.w / wn, wn * h), self.q)
                self.q /= np.linalg.norm(self.q)

    def sinkage(self):
        return DEPTH - (self.hub_y - P_["r_eff"])

    def slip(self):
        w = float(np.mean(self.spin))
        if abs(w) < 1e-3:
            return 0.0
        return 1.0 - self.forward_speed() / (w * P_["r_eff"])

    def snapshot(self):
        A, vA, R = self.attach()
        d = (self.hub_y - A[:, 1]) * R[1, 1] if self.phase != "hover" else np.zeros(4)
        if self.phase == "hover":
            d = np.full(4, 0.0)
        return (self.origin().copy(), self.q.copy(), np.clip(d, -STROKE, STROKE), self.phi.copy(),
                sand.theta)


# =============================================================================
# scene
# =============================================================================

canvas = tp.Canvas("threepp x warp - TP-1 on a tilt bed", width=W_, height=H_,
                   antialiasing=4, vsync=False, headless=HEADLESS)
renderer = tp.VulkanRenderer(canvas, 1 if BENCH else 3)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = cli_arg("--exposure", 1.0, float)

scene = tp.Scene()
BG = 0x06070a
scene.background = BG

# Low raking key from the side (+z, a little uphill) so rut walls throw shadows.
KEY_EL = math.radians(cli_arg("--key-el", 13.0, float))
KEY_AZ = math.radians(cli_arg("--key-az", 12.0, float))  # from +z toward -x (behind)
# The key comes from the rover's RIGHT (+z), the side every camera looks from
# (the lab camera sees the rover's right flank with uphill to the right), and
# rakes ACROSS the ruts (they run along x): the rut walls facing the lens are lit,
# their shadows fall away from it, and the rover's own faces toward us are lit.
key_dir = np.array([-math.sin(KEY_AZ) * math.cos(KEY_EL), math.sin(KEY_EL),
                    math.cos(KEY_AZ) * math.cos(KEY_EL)])
key = tp.DirectionalLight(0xfff0dc, cli_arg("--key", 4.2, float))
key.position.set(*(key_dir * 30.0))
key.cast_shadow = True
key.set_shadow_frustum(-10.0, 10.0, 10.0, -10.0)
key.set_shadow_bias(-0.0005)
scene.add(key)
scene.add(tp.HemisphereLight(0x9aa6b8, 0x141210, cli_arg("--fill", 0.35, float)))
# a faint cool counter-light from the other side, no shadows: the rover's shadow side
rim = tp.DirectionalLight(0xb8c8e0, cli_arg("--rim", 0.6, float))
rim.position.set(8.0, 7.0, -20.0)
scene.add(rim)

# The lab floor.
def std(color, rough=0.6, metal=0.0):
    m = tp.MeshStandardMaterial()
    m.color = color
    m.roughness = rough
    m.metalness = metal
    return m


def box(size, pos, mat, parent):
    b = tp.Mesh(tp.BoxGeometry(*size), mat)
    b.position.set(*pos)
    b.cast_shadow = True
    b.receive_shadow = True
    parent.add(b)
    return b


floor = tp.Mesh(tp.PlaneGeometry(60.0, 60.0), std(0x121314, 1.0))
floor.rotation.x = -math.pi / 2
floor.receive_shadow = True
scene.add(floor)

# tilt = the hinge; bed = the bed frame (the sim's coordinates) under it
tilt = tp.Group()
tilt.position.set(*HINGE)
scene.add(tilt)
bed = tp.Group()
bed.position.set(*(-HINGE_LOCAL))
tilt.add(bed)

steel = std(0x7a7e84, 0.5, 0.3)
dark_steel = std(0x33363a, 0.6, 0.25)
yellow = std(0xc8a020, 0.5, 0.2)
x0, x1 = BED_X
z0, z1 = BED_Z
LX, LZ = x1 - x0, z1 - z0
WALL_H = DEPTH + 0.06           # low: the raking key's wall shadow stays short
T_W = 0.03
# bin: floor plate + four walls
box((LX + 2 * T_W, 0.03, LZ + 2 * T_W), (0.0, -0.015, 0.0), steel, bed)
box((LX + 2 * T_W, WALL_H, T_W), (0.0, WALL_H / 2, z0 - T_W / 2), steel, bed)
NEAR_WALL = [box((LX + 2 * T_W, WALL_H, T_W), (0.0, WALL_H / 2, z1 + T_W / 2), steel, bed)]  # cut away in the X-ray
box((T_W, WALL_H, LZ), (x0 - T_W / 2, WALL_H / 2, 0.0), steel, bed)
box((T_W, WALL_H, LZ), (x1 + T_W / 2, WALL_H / 2, 0.0), steel, bed)
# top rails (yellow edge so the tilt reads)
for zz in (z0 - T_W / 2, z1 + T_W / 2):
    _r = box((LX + 2 * T_W + 0.02, 0.025, 0.05), (0.0, WALL_H + 0.0125, zz), yellow, bed)
    if zz > 0:
        NEAR_WALL.append(_r)
# the frame under the bed: two I-beam-ish rails and cross members
for zz in (-1.1, 1.1):
    box((LX + 0.5, 0.16, 0.08), (-0.1, -0.11, zz), dark_steel, bed)
for xx in np.linspace(x0 + 0.2, x1 - 0.2, 5):
    box((0.08, 0.10, LZ - 0.2), (xx, -0.08, 0.0), dark_steel, bed)
# hinge knuckles on the bed
for zz in (-1.1, 1.1):
    k_ = tp.Mesh(tp.CylinderGeometry(0.07, 0.07, 0.16, 20), steel)
    k_.rotation.x = math.pi / 2
    k_.position.set(*HINGE_LOCAL)
    k_.position.z = zz
    k_.cast_shadow = True
    bed.add(k_)
# static pedestals under the hinge (world)
for zz in (-1.1, 1.1):
    box((0.24, HINGE[1] - 0.05, 0.22), (HINGE[0], (HINGE[1] - 0.05) / 2, zz), dark_steel, scene)
# rest posts at the uphill end (the bed sits on them at 0 deg)
REST_X = 4.0
for zz in (-1.1, 1.1):
    box((0.2, HINGE[1] - 0.09, 0.2), (REST_X, (HINGE[1] - 0.09) / 2, zz), dark_steel, scene)
# lifting rams: floor anchor -> bed underside, re-aimed every frame
RAM_FLOOR = [np.array([1.2, 0.0, zz]) for zz in (-0.75, 0.75)]
RAM_BED = [np.array([2.6, -0.19, zz]) for zz in (-0.75, 0.75)]
rams = []
for _ in range(2):
    barrel = tp.Mesh(tp.CylinderGeometry(0.07, 0.07, 1.0, 20), yellow)
    rod = tp.Mesh(tp.CylinderGeometry(0.04, 0.04, 1.0, 16), steel)
    for m_ in (barrel, rod):
        m_.cast_shadow = True
        scene.add(m_)
    rams.append((barrel, rod))
# a few lab props for scale: a light mast and a cabinet, off the bed
box((0.6, 1.9, 0.5), (-5.2, 0.95, -2.6), std(0x2a2c30, 0.7, 0.3), scene)
box((0.08, 3.2, 0.08), (5.0, 1.6, -2.8), dark_steel, scene)

sand = LaneSand(SLOPE0) if P_["soil"] == "lane" else Sand(SLOPE0)
bed.add(sand.hf_mesh)
grains = GrainPoints()
if grains is not None:
    bed.add(grains.points)
LOOK = dict(grains=True, xray=False, xray_wheel=3)   # what present() draws

# --- the rover model ---
gl = tp.GLTFLoader().load(os.path.join(_HERE, "rover.glb"))
rover_root = gl.scene
rover_root.traverse(lambda o: (setattr(o, "cast_shadow", True), setattr(o, "receive_shadow", True))
                    if isinstance(o, tp.Mesh) else None)
bed.add(rover_root)
node = {n: rover_root.get_object_by_name(n) for n in
        ["chassis"] + [f"leg_{k}" for k in WN] + [f"wheel_{k}" for k in WN]
        + [f"actuator_{k}_barrel" for k in WN] + [f"actuator_{k}_rod" for k in WN]}
ACT_A = np.array([SPEC["actuators"]["chassis_mount"][k] for k in WN], float)
ACT_B = np.array([SPEC["actuators"]["arm_mount_leg_frame"][k] for k in WN], float)
Y = np.array([0.0, 1.0, 0.0])

rover = Rover()


def pose_rover(origin, q, delta, phi):
    ch = node["chassis"]
    ch.position.set(*origin)
    ch.quaternion.set(*q)
    for i, k in enumerate(WN):
        a = leg_angle(float(delta[i]), FRONT[i])
        node[f"leg_{k}"].quaternion.set(*qaxis((0, 0, 1), a))
        node[f"wheel_{k}"].quaternion.set(*qaxis((0, 0, 1), -float(phi[i]) - a))
        Rl = qmat(qaxis((0, 0, 1), a))
        B = PIVOTS[i] + Rl @ ACT_B[i]
        node[f"actuator_{k}_barrel"].quaternion.set(*q_from_to(Y, B - ACT_A[i]))
        node[f"actuator_{k}_rod"].quaternion.set(*q_from_to(Y, Rl.T @ (ACT_A[i] - B)))


def bed_to_world(p, theta_deg):
    t = math.radians(theta_deg)
    c, s = math.cos(t), math.sin(t)
    lp = np.asarray(p) - HINGE_LOCAL
    return HINGE + np.array([c * lp[0] - s * lp[1], s * lp[0] + c * lp[1], lp[2]])


def pose_rig(theta_deg):
    tilt.quaternion.set(*qaxis((0, 0, 1), math.radians(theta_deg)))
    for (barrel, rod), fa, bb in zip(rams, RAM_FLOOR, RAM_BED):
        top = bed_to_world(bb, theta_deg)
        d = top - fa
        L = float(np.linalg.norm(d))
        q = q_from_to(Y, d)
        lb = min(2.2, 0.55 * L + 0.2)
        barrel.scale.set(1.0, lb, 1.0)
        barrel.position.set(*(fa + d / L * (0.5 * lb)))
        barrel.quaternion.set(*q)
        lr = max(L - lb + 0.15, 0.1)
        rod.scale.set(1.0, lr, 1.0)
        rod.position.set(*(top - d / L * (0.5 * lr)))
        rod.quaternion.set(*q)


# --- cameras ---
camera = tp.PerspectiveCamera(50, W_ / H_, 0.05, 200)
cam_pos = None
cam_tgt = None
view = 0            # 0 chase, 1 lab


CAM_FOV = {0: 50, 1: 50, 2: 50, 3: 42, 4: 40, 5: 55, 6: 34, 7: 55, 8: 40}
fly_u = [0.0]


def camera_goal(theta_deg, snap, mode):
    o, q = snap[0], snap[1]
    R = qmat(q)
    if mode == 3:        # the rut behind the rear-right wheel, low and close
        tgt = o + R @ np.array([-1.15, -0.27, 0.61])
        eye = o + R @ np.array([-2.15, 0.30, 1.05])
        return bed_to_world(eye, theta_deg), bed_to_world(tgt, theta_deg)
    if mode == 4:        # the front-right wheel digging in, side-on and low
        tgt = o + R @ np.array([0.55, -0.22, 0.61])
        eye = o + R @ np.array([1.25, 0.05, 1.75])
        return bed_to_world(eye, theta_deg), bed_to_world(tgt, theta_deg)
    if mode == 5:        # downhill, low, looking UP the bed at the rover coming toward it
        eye = np.array([BED_X[0] + 0.20, DEPTH + 0.24, 1.38])
        tgt = o + np.array([-0.3, 0.0, 0.2])
        return bed_to_world(eye, theta_deg), bed_to_world(tgt, theta_deg)
    if mode == 6:        # X-ray: side-on through the rear-right wheel's plane
        hub = o + R @ HUBS[3]
        eye = np.array([hub[0] - 0.10, hub[1] - 0.02, hub[2] + 1.45])
        tgt = np.array([hub[0] - 0.10, hub[1] - 0.22, hub[2]])
        return bed_to_world(eye, theta_deg), bed_to_world(tgt, theta_deg)
    if mode == 8:        # the rear-right wheel, close and low, from behind its flank
        tgt = o + R @ np.array([-0.62, -0.20, 0.61])
        eye = o + R @ np.array([-1.45, 0.12, 1.28])
        return bed_to_world(eye, theta_deg), bed_to_world(tgt, theta_deg)
    if mode == 7:        # the fly-over: slowly up the bed, over the pits and furrows
        u = fly_u[0]
        eye = np.array([-4.0 + 5.0 * u, DEPTH + 2.1, 1.3])
        tgt = np.array([-3.0 + 5.0 * u, DEPTH, 0.0])
        return bed_to_world(eye, theta_deg), bed_to_world(tgt, theta_deg)
    c_w = bed_to_world(o + np.array([0.0, 0.4, 0.0]), theta_deg)
    if mode == 0:        # chase: behind, left, above -- in the bed's direction
        R = qmat(q)
        f = R[:, 0].copy()
        f[1] = 0.0
        f /= np.linalg.norm(f)
        side = np.array([-f[2], 0.0, f[0]])        # the rover's right (+z when f = +x)
        back_bed = o - 3.4 * f + 1.9 * side + np.array([0.0, 1.55, 0.0])
        eye = bed_to_world(back_bed, theta_deg)
        eye[1] = max(eye[1], c_w[1] + 0.9)
        return eye, c_w + np.array([0.0, -0.1, 0.0])
    if mode == 1:        # lab: side-on, level, from -z
        mid = bed_to_world(np.array([0.2, DEPTH, 0.0]), theta_deg)
        tgt = 0.55 * mid + 0.45 * c_w
        return np.array([tgt[0], 1.6 + 0.45 * tgt[1], 11.5]), tgt
    # 2: low 3/4 front: the wheels, the ruts behind them
    R = qmat(q)
    f = R[:, 0].copy()
    f[1] = 0.0
    f /= np.linalg.norm(f)
    side = np.array([-f[2], 0.0, f[0]])
    eye_bed = o + 2.4 * f + 1.9 * side + np.array([0.0, 0.75, 0.0])
    return bed_to_world(eye_bed, theta_deg), bed_to_world(o - 0.5 * f + np.array([0, 0.05, 0]), theta_deg)


def update_camera(theta_deg, snap, dt, mode, snapcam=False):
    global cam_pos, cam_tgt
    eye, tgt = camera_goal(theta_deg, snap, mode)
    if cam_pos is None or snapcam:
        cam_pos, cam_tgt = eye, tgt
    else:
        k = 1.0 - math.exp(-dt * 3.0)
        cam_pos = cam_pos + (eye - cam_pos) * k
        cam_tgt = cam_tgt + (tgt - cam_tgt) * k
    camera.position.set(*cam_pos)
    camera.look_at(tp.Vector3(*cam_tgt))
    fov = CAM_FOV.get(mode, 50)
    if abs(camera.fov - fov) > 1e-3:
        camera.fov = fov
        camera.update_projection_matrix()


# =============================================================================
# the loop
# =============================================================================

state = dict(theta=SLOPE0, target=SLOPE0, rate=0.0, w_cmd=0.0, steer=0.0, frames=0)
TILT_OMEGA = 1.6          # 1/s, the bed's critically damped follower
TILT_RATE_MAX = 1.0       # deg/s


def tilt_follow(dt):
    """Critically damped approach of theta to the target, rate-limited: never snaps."""
    e = state["target"] - state["theta"]
    acc = TILT_OMEGA * TILT_OMEGA * e - 2.0 * TILT_OMEGA * state["rate"]
    state["rate"] = float(np.clip(state["rate"] + acc * dt, -TILT_RATE_MAX, TILT_RATE_MAX))
    state["theta"] += state["rate"] * dt
snap_prev = snap_cur = None
hist = []   # per-frame log (t, theta, v, w, slip, sink..., W...)


def sim_frame(w_base, steer):
    """One 60 Hz frame: slope -> gravity, commands -> rover + grains."""
    global snap_prev, snap_cur
    sand.set_slope(state["theta"])
    g = gravity_bed(sand.theta)
    w_cmd = np.array([w_base - steer, w_base + steer, w_base - steer, w_base + steer])
    # stop at the top of the bed
    x_end = P_["lane_x"][1] if P_["soil"] == "lane" else BED_X[1]
    if rover.origin()[0] + HUBS[0, 0] > x_end - 0.45:
        w_cmd = np.minimum(w_cmd, 0.0)
    rover.step(sand, w_cmd, g)
    snap_prev, snap_cur = snap_cur, rover.snapshot()
    if snap_prev is None:
        snap_prev = snap_cur
    state["frames"] += 1
    hist.append([rover.t, sand.theta, rover.forward_speed(), float(np.mean(rover.spin)), rover.slip(),
                 *rover.sinkage(), *rover.W, *rover.F[:, 0], *rover.T, sand.ms, rover.origin()[0]])


def blended(a):
    p = snap_prev[0] + (snap_cur[0] - snap_prev[0]) * a
    q = nlerp(snap_prev[1], snap_cur[1], a)
    d = snap_prev[2] + (snap_cur[2] - snap_prev[2]) * a
    ph = snap_prev[3] + (snap_cur[3] - snap_prev[3]) * a
    th = snap_prev[4] + (snap_cur[4] - snap_prev[4]) * a
    return p, q, d, ph, th


def present(a, dt, mode, snapcam=False):
    s = blended(a)
    PRINT_PHI[0] = np.asarray(s[3], float)
    pose_rover(*s[:4])
    pose_rig(s[4])
    update_camera(s[4], s, dt, mode, snapcam)
    sand.hf_mesh.visible = not LOOK["xray"]
    for _m in NEAR_WALL:
        _m.visible = not LOOK["xray"]
    if not LOOK["xray"]:
        sand.publish()
    if grains is not None:
        if LOOK["xray"]:
            k = LOOK["xray_wheel"]
            grains.update(sand, 1, zc=float(sand.out[k, 10]), half=0.03, size=0.011)
        elif LOOK["grains"]:
            grains.update(sand, 0, vmin=0.05, size=0.008)
        else:
            grains.geom.set_draw_range(0, 0)


# first frames
sim_frame(0.0, 0.0)
present(1.0, DT, view, True)
renderer.render(scene, camera)


def script_theta(t, t0, t1, th_max):
    """The scripted tilt: smootherstep in the angle, so the tilt RATE rises and
    falls smoothly (zero rate and zero acceleration at both ends)."""
    u = min(max((t - t0) / (t1 - t0), 0.0), 1.0)
    return TH_FROM + (th_max - TH_FROM) * u * u * u * (u * (6.0 * u - 15.0) + 10.0)


def save_png(path, img):
    from PIL import Image
    Image.fromarray(img).save(path)
    print(f"  wrote {path}", flush=True)


def still(path, mode):
    global cam_pos
    for _ in range(3):
        present(1.0, DT, mode, snapcam=True)
        renderer.render(scene, camera)
    save_png(path, caption(renderer.read_pixels(), legend=LOOK["xray"]))


_font = None


NASA_SLOPE = [(0.3, 2.5), (5.6, 5.1), (10.0, 10.4), (15.3, 35.5), (20.0, 73.8), (25.2, 83.8),
              (30.1, 89.7)]     # NASA MGRU3 tilt bed, 2022 campaign, as digitised in Hu et al. 2024


def _fonts():
    global _font
    from PIL import ImageFont
    if _font is None:
        try:
            _font = (ImageFont.truetype("consola.ttf", 22), ImageFont.truetype("consola.ttf", 15),
                     ImageFont.truetype("consolab.ttf", 30))
        except OSError:
            f = ImageFont.load_default()
            _font = (f, f, f)
    return _font


def inset(d, x0, y0, w, h):
    """Slip vs slope: NASA's tilt-bed points and our live trace."""
    f1 = _fonts()[1]
    d.rectangle([x0, y0, x0 + w, y0 + h], fill=(0, 0, 0, 150))
    px0, py0, pw, ph = x0 + 44, y0 + 26, w - 58, h - 76
    smax, tmax = 150.0, 36.0

    def P(th, sl):
        return (px0 + pw * th / tmax, py0 + ph * (1.0 - min(max(sl, 0.0), smax) / smax))
    d.line([px0, py0, px0, py0 + ph, px0 + pw, py0 + ph], fill=(200, 200, 200, 255), width=1)
    for sl in (0, 50, 100, 150):
        yy = P(0, sl)[1]
        d.text((px0 - 34, yy - 8), f"{sl:3d}", fill=(200, 200, 200, 255), font=f1)
        if sl == 100:
            d.line([px0, yy, px0 + pw, yy], fill=(120, 120, 120, 200), width=1)
    for th in (0, 10, 20, 30):
        xx = P(th, 0)[0]
        d.text((xx - 8, py0 + ph + 3), f"{th}", fill=(200, 200, 200, 255), font=f1)
    d.text((px0 + pw - 70, py0 + ph + 18), "slope deg", fill=(200, 200, 200, 255), font=f1)
    d.text((x0 + 6, y0 + 4), "slip %", fill=(200, 200, 200, 255), font=f1)
    pts = [P(th, sl) for th, sl in NASA_SLOPE]
    for (xx, yy) in pts:
        d.ellipse([xx - 4, yy - 4, xx + 4, yy + 4], fill=(255, 150, 40, 255))
    tr = story.trace
    if len(tr) > 2:
        step = max(1, len(tr) // 300)
        line = [P(th, 100.0 * sl) for th, sl in tr[::step]]
        d.line(line, fill=(120, 220, 255, 255), width=2)
        xx, yy = line[-1]
        d.ellipse([xx - 4, yy - 4, xx + 4, yy + 4], fill=(120, 220, 255, 255))
    d.text((x0 + 6, y0 + h - 20), "\u25cf NASA tilt-bed data (Hu et al. 2024)", fill=(255, 150, 40, 255), font=f1)
    d.text((x0 + w - 104, y0 + 4), "\u2014 TP-1 sim", fill=(120, 220, 255, 255), font=f1)


def caption(img, labels=(), legend=False):
    """HUD burned into a recorded frame (no ImGui headless)."""
    from PIL import Image, ImageDraw
    f0, f1, f2 = _fonts()
    r = rover
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    sk = 1000 * r.sinkage()
    sl = r.slip() if abs(np.mean(r.spin)) > 0.1 else float("nan")
    lines = [f"TP-1  tilt bed  {sand.theta:4.1f} deg",
             f"speed {100 * r.forward_speed():+6.1f} cm/s   omega {np.mean(r.spin):4.2f} rad/s",
             f"slip  {100 * sl:6.0f} %" if np.isfinite(sl) else "slip      -",
             f"sinkage  front {0.5 * (sk[0] + sk[1]):3.0f} mm   rear {0.5 * (sk[2] + sk[3]):3.0f} mm",
             f"drive {'ON' if r.power else 'UNPOWERED (free-wheeling)'}"]
    d.rectangle([14, 14, 560, 20 + 28 * len(lines)], fill=(0, 0, 0, 120))
    for k, t in enumerate(lines):
        d.text((24, 20 + 28 * k), t, fill=(235, 228, 210, 255), font=f0)
    y = 26 + 28 * len(lines)
    for lab in labels:
        col = (255, 90, 60, 255) if "LOST" in lab else (255, 230, 120, 255)
        d.rectangle([14, y, 24 + 18 * len(lab), y + 40], fill=(0, 0, 0, 150))
        d.text((24, y + 4), lab, fill=col, font=f2)
        y += 46
    H_, W__ = im.size[1], im.size[0]
    inset(d, W__ - 380, H_ - 250, 366, 236)
    if legend:
        x0, y0 = 24, H_ - 118
        d.rectangle([14, y0 - 30, 470, H_ - 14], fill=(0, 0, 0, 150))
        d.text((x0, y0 - 26), "X-RAY: grains in a 6 cm slab through the rear wheel", fill=(235, 228, 210, 255), font=f1)
        from_c = [(20, 26, 115), (13, 140, 242), (51, 230, 89), (255, 204, 20), (242, 31, 15)]
        n = 300
        for i in range(n):
            u = i / (n - 1) * 4.0
            k = min(int(u), 3)
            f = u - k
            c = tuple(int(from_c[k][j] + (from_c[k + 1][j] - from_c[k][j]) * f) for j in range(3))
            d.line([x0 + i, y0 + 4, x0 + i, y0 + 28], fill=c + (255,))
        d.text((x0, y0 + 34), "0", fill=(235, 228, 210, 255), font=f1)
        d.text((x0 + n - 90, y0 + 34), f"{XRAY_VSCALE:.2f}+ m/s", fill=(235, 228, 210, 255), font=f1)
        d.text((x0 + 110, y0 + 34), "grain speed", fill=(235, 228, 210, 255), font=f1)
    return np.asarray(im)


def write_log(name):
    import csv
    path = os.path.join(OUT, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "theta", "v", "w", "slip"] + [f"sink_{k}" for k in WN] + [f"W_{k}" for k in WN]
                   + [f"Fx_{k}" for k in WN] + [f"T_{k}" for k in WN] + ["sand_ms", "x"])
        w.writerows(hist)
    print(f"  wrote {path}")


W_DRIVE = cli_arg("--omega", 0.8, float)


# --- the story ----------------------------------------------------------------
# The bed angle and the motor command follow the story; the rover does what the
# grains let it. Nothing about its motion is scripted: the tilt stops RISING when
# the rover is seen sliding (an operator's reaction), and drive power is cut only
# if it still holds at the maximum tilt (a labelled event).
TILT_KNOTS = [(4.0, 0.0, 0.0), (29.0, 28.0, 0.4)]   # flat drive-off, then ONE slow eased ramp
CREEP_RATE = 0.4          # deg/s past 28 (the ramp's own end rate: no corner)
SLIDE_GOAL = 1.0          # m of backward slide, or...
CREEP_HOLD_MAX = cli_arg("--hold", 6.0, float)    # ...s held at the top, whichever first
W_BACK = cli_arg("--w-back", -0.3, float)          # the operator's back-out wheel speed, rad/s
BACKOUT_MAX = 14.0        # s of backing out without a skid before the bed comes down anyway
LOWER_S, LOWER_FAST_S = 9.0, 3.5                   # the operator lowers the bed over this long
                          # (at 36 deg the default coupling held ~17 s, then blew up numerically)
LAPSE = 8                 # sim frames per film frame in the time-lapse (8x at 30 fps from 60 Hz = 4x... see below)
TH_CAP = 36.0             # stay below phi = 38: the far field is a static heightfield
T_DRIVE_ON = 1.0


def hermite(k, t):
    if t <= k[0][0]:
        return k[0][1], 0.0
    for (t0, y0, m0), (t1, y1, m1) in zip(k[:-1], k[1:]):
        if t <= t1:
            h = t1 - t0
            u = (t - t0) / h
            y = ((2 * u ** 3 - 3 * u ** 2 + 1) * y0 + (u ** 3 - 2 * u ** 2 + u) * h * m0
                 + (-2 * u ** 3 + 3 * u ** 2) * y1 + (u ** 3 - u ** 2) * h * m1)
            dy = ((6 * u ** 2 - 6 * u) * y0 + (3 * u ** 2 - 4 * u + 1) * h * m0
                  + (-6 * u ** 2 + 6 * u) * y1 + (3 * u ** 2 - 2 * u) * h * m1) / h
            return y, dy
    return k[-1][1], k[-1][2]


def smoother(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (6.0 * u - 15.0) + 10.0)


class Story:
    """flat drive-off -> eased tilt 0-28 (the climb) -> creep 0.4 deg/s to 36 and hold
    (the dig-in and, if the grains let go, the slide: physics only) -> ease back ->
    fly-over. No event is forced on the rover."""

    def __init__(self):
        self.t = 0.0
        self.phase = "flat"
        self.t0 = 0.0
        self.theta = 0.0
        self.rate = 0.0
        self.hold = 0.0
        self.th_hold = 0.0
        self.trace = []           # (theta, slip) for the inset
        self.x_max = -1e9
        self.slid = 0.0           # m back from the furthest point reached
        self.stall = None         # (t, theta, x) where it stopped making progress
        self.slide_v = []
        self.release = None       # (t, theta): the slide became clear (>= 5 cm back)
        self.power_lost = None    # kept for the caption; never set (no forced events)
        self.x_start = None
        self.blowup = None
        self.neg = 0.0
        self.skid = None          # (t, theta, x) where the skid started
        self.skid_log = []        # (t, theta, v, omega, slip, x) through the skid
        self.slid_skid = 0.0
        self.still = 0.0
        self.runaway = False
        self.w_now = 0.0
        self.skid_over = None

    def trace_add(self, th, sl):
        """The inset's trace: slip EMA'd over ~0.5 s (a frame's v is noisy)."""
        self.sl_ema = sl if not self.trace else self.sl_ema + (sl - self.sl_ema) * (DT / 0.5)
        self.trace.append((th, self.sl_ema))

    def _go(self, ph):
        self.phase, self.t0 = ph, self.t
        print(f"  [story] t={self.t:5.1f} -> {ph}  theta={self.theta:.1f} x={rover.origin()[0]:+.2f}",
              flush=True)

    def step(self):
        """Advance one sim frame; returns the wheel speed command."""
        t, tp_ = self.t, self.t - self.t0
        v = rover.forward_speed()
        x = rover.origin()[0]
        if self.x_start is None:
            self.x_start = x
        w = W_DRIVE if t >= T_DRIVE_ON else 0.0
        if rover.phase == "free":
            if x > self.x_max:
                self.x_max = x
                self.t_xmax = t
            self.slid = self.x_max - x
            if self.stall is None and t > 8.0 and t - self.t_xmax > 3.0:
                self.stall = (self.t_xmax, story_theta_at(self.t_xmax), self.x_max)
                print(f"  [story] STALL: no progress since t={self.t_xmax:.1f} s at "
                      f"{self.stall[1]:.1f} deg, x={self.x_max:+.2f} ({self.x_max - self.x_start:.2f} m climbed)",
                      flush=True)
            if self.release is None and self.slid > 0.05 and self.phase == "creep":
                self.release = (t, self.theta)
                print(f"  [story] SLIDING BACK (5 cm) at t={t:.1f} s, {self.theta:.1f} deg", flush=True)
        if (rover.phase == "free" and self.phase not in ("end", "blowup")
                and (abs(v) > 1.5 or np.max(np.abs(rover.spin)) > 20.0 or np.min(rover.sinkage()) < -0.08
                     or not sand.hubs_inside())):
            # a divergence guard, not a story beat: the coupling has left the physical regime
            print(f"  [story] NUMERICAL BLOW-UP at t={t:.1f} s: v={v:+.2f} m/s, spin "
                  f"{np.round(rover.spin, 2)}, sinkage {np.round(1000 * rover.sinkage())} mm -- stopping", flush=True)
            self.blowup = (t, self.theta)
            self._go("end")
        if self.phase == "flat":
            if t >= TILT_KNOTS[0][0]:
                self._go("tilt")
        elif self.phase == "tilt":
            self.theta, self.rate = hermite(TILT_KNOTS, t)
            if t >= TILT_KNOTS[-1][0]:
                self._go("creep")
        elif self.phase == "creep":
            if self.theta < TH_CAP:
                self.rate = CREEP_RATE
                self.theta = min(TH_CAP, self.theta + CREEP_RATE * DT)
            else:
                self.rate = 0.0
                self.hold += DT
            if self.release is not None:
                self.slide_v.append(v)
            if self.slid >= SLIDE_GOAL or self.hold >= CREEP_HOLD_MAX or rover.hit_wall:
                self.th_hold = self.theta
                self.w_now = w
                self._go("backout")
        elif self.phase == "backout":
            # OPERATOR: back out. The speed command eases from forward to slow reverse;
            # the torque limit is unchanged. The bed holds its angle.
            w = W_DRIVE + (W_BACK - W_DRIVE) * smoother(tp_ / 2.0)
            self.neg = self.neg + DT if v < -0.05 else 0.0
            if self.neg >= 0.3:
                self.skid = (t, self.theta, x)
                self.th_lower0, self.t_lower0, self.lower_s = self.theta, t, LOWER_S
                self._go("skid")
            elif tp_ >= BACKOUT_MAX:
                self._go("ease")
        elif self.phase == "skid":
            # the skid is physics; the operator lowers the bed as soon as it starts
            w = W_BACK
            if abs(v) > 0.8 and self.lower_s > LOWER_FAST_S:
                # a runaway: lower faster (restart the ease from here, shorter)
                print(f"  [story] skid at {v:+.2f} m/s: lowering the bed faster", flush=True)
                self.th_lower0, self.t_lower0, self.lower_s = self.theta, t, LOWER_FAST_S
                self.runaway = True
            u = (t - self.t_lower0) / self.lower_s
            self.theta = self.th_lower0 * (1.0 - smoother(u))
            if self.skid_over is None:
                self.skid_log.append((t, self.theta, v, float(np.mean(rover.spin)), rover.slip(), x))
                self.slid_skid = self.skid[2] - x
                if tp_ > 1.0 and v > -0.1:
                    self.skid_over = t        # the skid has died: the operator stops the wheels
                    print(f"  [story] skid over at t={t:.1f} s, {self.theta:.1f} deg, "
                          f"{self.slid_skid:.2f} m", flush=True)
            else:
                w = 0.0
            if u >= 1.0 or tp_ > 20.0 or rover.hit_wall:
                self.th_hold = self.theta
                self._go("ease")
        elif self.phase == "ease":
            w = 0.0
            dur = 6.0 if self.th_hold > 5.0 else 2.0
            self.theta = self.th_hold * (1.0 - smoother(tp_ / dur))
            if tp_ >= dur + 0.5:
                self._go("fly")
        elif self.phase == "fly":
            w = 0.0
            self.theta = 0.0
            if tp_ >= 7.0:
                self._go("end")
        if self.phase in ("tilt", "creep") and rover.phase == "free" and np.mean(rover.spin) > 0.05:
            self.trace_add(self.theta, rover.slip())
        state["theta"] = self.theta
        self.t += DT
        return w


def story_theta_at(t):
    return hermite(TILT_KNOTS, t)[0] if t <= TILT_KNOTS[-1][0] else min(
        TH_CAP, TILT_KNOTS[-1][1] + CREEP_RATE * (t - TILT_KNOTS[-1][0]))


story = Story()


def story_step():
    w = story.step()
    sim_frame(w, 0.0)


if DRY:
    T_END = cli_arg("--dry-end", 60.0, float)
    n = int(T_END * 60)
    t0 = time.perf_counter()
    for i in range(n):
        if story.phase == "end":
            break
        story_step()
        if i % 30 == 0:
            r = rover
            print(f"  t={i * DT:5.1f} th={sand.theta:4.1f} x={r.origin()[0]:+.3f} v={r.forward_speed():+.3f} "
                  f"w={np.mean(r.spin):.2f} slip={100 * r.slip():5.0f}% sink={np.round(1000 * r.sinkage()).astype(int)} "
                  f"W={np.round(r.W).astype(int)} Fx={np.round(r.F[:, 0]).astype(int)} "
                  f"T={np.round(r.T).astype(int)} tau={np.round(r.tau).astype(int)}", flush=True)
    print(f"  dry: {i} frames in {time.perf_counter() - t0:.1f} s ({1000 * (time.perf_counter() - t0) / max(i, 1):.1f} ms/frame); "
          f"skid {story.skid} {story.slid_skid:.2f} m runaway {story.runaway} blowup {story.blowup}; "
          f"stall {story.stall}; release {story.release}; slid {story.slid:.2f} m; climbed "
          f"{story.x_max - story.x_start:.2f} m; slide v mean {np.mean(story.slide_v) if story.slide_v else 0:+.3f} m/s",
          flush=True)
    write_log(f"dry_log{cli_arg('--tag', '', str)}.csv")
    sys.exit(0)
TH_MAX = cli_arg("--tilt-to", 25.0, float)
T_TILT0 = cli_arg("--tilt-start", 3.0, float)
T_TILT1 = cli_arg("--tilt-end", 12.0, float)

if SHOT:
    secs = cli_arg("--shot-time", 6.0, float)
    n = int(secs * 60)
    t0 = time.perf_counter()
    for i in range(n):
        t = i * DT
        if "--tilt" in sys.argv:
            state["theta"] = script_theta(t, T_TILT0, T_TILT1, TH_MAX)
        sim_frame(W_DRIVE if t > 1.0 else 0.0, 0.0)
    print(f"  {n} frames in {time.perf_counter() - t0:.1f} s; sand {sand.ms:.1f} ms/frame (last), "
          f"nsub {sand.nsub}", flush=True)
    tag = cli_arg("--tag", "", str)
    if "--debug-surface" in sys.argv:
        sand.publish()
        top, Hh = sand.top2.numpy(), sand.sim.hf.numpy()
        pos = sand.pos.numpy().reshape(sand.hnx, sand.hnz, 3)
        sim = sand.sim
        for k in range(4):
            xlo, xhi, zlo, zhi = sim.block_interior(k)
            j = int((0.5 * (zlo + zhi) + 0.13 - sim.hf_origin[1]) / sim.hf_dx)   # beside the rut
            i1 = int((xhi - sim.hf_origin[0]) / sim.hf_dx)
            i0 = int((xlo - sim.hf_origin[0]) / sim.hf_dx)
            print(f"patch {k}: x {xlo:.3f}..{xhi:.3f}  i {i0}..{i1}  j {j}")
            for i in list(range(i0 - 3, i0 + 4)) + list(range(i1 - 3, i1 + 4)):
                print(f"   i={i:4d} top={1000 * top[i, j]:7.1f} H={1000 * Hh[i, j]:7.1f} surf={1000 * pos[i, j, 1]:7.1f}")
            jj = np.arange(int((zlo - sim.hf_origin[1]) / sim.hf_dx) + 2, int((zhi - sim.hf_origin[1]) / sim.hf_dx) - 2)
            ii = int((0.5 * (xlo + xhi) - sim.hf_origin[0]) / sim.hf_dx)
            print("   across z (mm):", np.round(1000 * pos[ii, jj, 1]).astype(int).tolist())
    if "--story-shots" in sys.argv:
        still(os.path.join(OUT, f"still_downhill{tag}.png"), 5)
        still(os.path.join(OUT, f"still_rear{tag}.png"), 8)
        LOOK["xray"] = True
        still(os.path.join(OUT, f"still_xray{tag}.png"), 6)
        LOOK["xray"] = False
        fly_u[0] = 0.4
        still(os.path.join(OUT, f"still_fly{tag}.png"), 7)
    else:
        still(os.path.join(OUT, f"still_chase{tag}.png"), 0)
        still(os.path.join(OUT, f"still_lab{tag}.png"), 1)
        still(os.path.join(OUT, f"still_front{tag}.png"), 2)
    write_log(f"shot_log{tag}.csv")
    r = rover
    print(f"  t={r.t:.2f}s x={r.origin()[0]:.2f} v={r.forward_speed():.3f} m/s w={np.mean(r.spin):.3f} "
          f"slip={100 * r.slip():.0f}% sink(mm)={np.round(1000 * r.sinkage(), 0)} W={np.round(r.W, 0)} "
          f"delta(mm)={np.round(1000 * r.delta, 0)}", flush=True)
    sys.exit(0)

PREVIEW = cli_arg("--preview", 0.0, float)
if PREVIEW > 0:
    # the owner's look at the lugged lane: on a tilted bed from the start, ramping up
    fps = 30
    path = os.path.join(OUT, "preview_lugged.mp4")
    enc = Encoder(path, W_, H_, fps, crf=18, preset="medium", faststart=True)
    fdir = os.path.join(OUT, "frames_preview")
    os.makedirs(fdir, exist_ok=True)
    th0, th1 = SLOPE0, cli_arg("--preview-to", 33.0, float)
    n = int(PREVIEW * 60)
    t0 = time.perf_counter()
    cam_mode = [-1]
    ms_sim = []
    neg = 0.0
    for i in range(n):
        t = i * DT
        state["theta"] = th0 + (th1 - th0) * smoother((t - 2.0) / (PREVIEW - 3.0))
        ts = time.perf_counter()
        sim_frame(W_DRIVE if t >= 1.0 else 0.0, 0.0)
        ms_sim.append(1000.0 * (time.perf_counter() - ts))
        v = rover.forward_speed()
        # a release counts only once it is driving (not the set-down transient)
        neg = neg + DT if (v < -0.02 and rover.phase == "free" and t > 3.0) else 0.0
        if neg >= 0.5 and story.release is None:
            story.release = (t, sand.theta)
            print(f"  RELEASE at t={t:.1f} theta={sand.theta:.1f}", flush=True)
        if rover.phase == "free" and np.mean(rover.spin) > 0.05:
            story.trace_add(sand.theta, rover.slip())
        if i % 2:
            continue
        mode = 0 if t < PREVIEW * 0.5 else 8
        cut = mode != cam_mode[0]
        cam_mode[0] = mode
        present(1.0, 2 * DT, mode, snapcam=cut)
        renderer.render(scene, camera)
        labels = [f"offline preview, {np.mean(ms_sim[-120:]):.0f} ms/frame sim",
                  f"LUGGED wheels: {P_['lug_n']} x {1000 * max(P_['lug_t'], P_['h']):.0f} mm grousers in the grains (24 drawn)"]
        if story.release is not None:
            labels.append(f"released at {story.release[1]:.1f} deg: sliding back")
        img = caption(renderer.read_pixels(), labels)
        enc.send(img)
        if enc.n % 30 == 0:
            save_png(os.path.join(fdir, f"p{enc.n:04d}.png"), img)
        if i % 60 == 0:
            r = rover
            print(f"  t={t:5.1f} theta={sand.theta:4.1f} x={r.origin()[0]:+.2f} v={r.forward_speed():+.3f} "
                  f"slip={100 * r.slip():5.0f}% sink={np.round(1000 * r.sinkage()).astype(int)} "
                  f"W={np.round(r.W).astype(int)} sim {np.mean(ms_sim[-60:]):.0f} ms", flush=True)
    enc.close()
    print(f"  wrote {path} ({enc.n} frames, {time.perf_counter() - t0:.0f} s); release {story.release}", flush=True)
    write_log("preview_log.csv")
    sys.exit(0)

if RECORD > 0:
    fps = 30
    tag = cli_arg("--tag", "", str)
    path = os.path.join(OUT, f"rover_story{tag}.mp4")
    if os.path.exists(path) and "--overwrite" not in sys.argv:
        sys.exit(f"{path} exists (a reported cut is never overwritten): pick a new --tag")
    part = path[:-4] + ".partial.mp4"      # renamed on completion: no half-written file under the final name
    enc = Encoder(part, W_, H_, fps, crf=18, preset="medium", faststart=True)
    fdir = os.path.join(OUT, f"frames{tag}")
    t_sim_ms = []
    os.makedirs(fdir, exist_ok=True)
    t0 = time.perf_counter()
    cam_mode = [-1]
    i = 0
    skip_acc = 0
    xray_t0 = None
    while story.phase != "end" and i * DT < RECORD:
        ts_ = time.perf_counter()
        story_step()
        t_sim_ms.append(1000.0 * (time.perf_counter() - ts_))
        i += 1
        ph, tp_ = story.phase, story.t - story.t0
        labels = []
        every = 2                           # real time: 60 Hz sim -> 30 fps film
        LOOK["xray"] = False
        if ph == "flat":
            mode = 2 if story.t < 2.0 else 3
        elif ph == "tilt":
            mode = 1 if story.t < 13.0 else (0 if story.t < 22.0 else 1)
        elif ph == "creep":
            if tp_ < 14.0:
                mode, every = 8, LAPSE
            elif tp_ < 20.0:
                mode, every = 6, 4
                LOOK["xray"] = True
            else:
                mode, every = (5, LAPSE) if tp_ < 48.0 else (0, LAPSE)
            labels.append(f"{every // 2}x time-lapse")
            if story.slid > 0.03:
                labels.append(f"sliding back: {100 * story.slid:.0f} cm, wheels still driving forward")
            elif story.hold > 0.0:
                labels.append("embedded: the lugs are anchored in their own pits")
        elif ph == "backout":
            mode, every = 8, 4
            labels.append(f"OPERATOR: back out (wheels {W_BACK:+.1f} rad/s)")
            labels.append("2x time-lapse")
        elif ph == "skid":
            if story.skid_over is None:
                labels.append("skidding downhill -- operator lowers the bed")
                labels.append("0.5x slow motion")
                every = 1                   # slow motion through the skid itself
                if 1.2 <= tp_ < 2.7:
                    mode = 6
                    LOOK["xray"] = True
                else:
                    mode = 5
            else:
                labels.append("skid over -- the bed comes down")
                mode = 1
            if story.runaway:
                labels.append("fast skid: bed lowered faster")
        elif ph == "ease":
            mode = 1
        else:
            mode = 7
            fly_u[0] = smoother(tp_ / 7.0)
        skip_acc += 1
        if skip_acc < every:
            continue
        skip_acc = 0
        cut = mode != cam_mode[0]
        cam_mode[0] = mode
        present(1.0, every * DT, mode, snapcam=cut)
        renderer.render(scene, camera)
        img = caption(renderer.read_pixels(), labels, legend=LOOK["xray"])
        enc.send(img)
        if enc.n % 15 == 0:
            save_png(os.path.join(fdir, f"f{enc.n:04d}_{ph}_m{mode}.png"), img)
        if i % 60 == 0:
            r = rover
            print(f"  t={story.t:5.1f} {ph:6s} theta={sand.theta:4.1f} x={r.origin()[0]:+.2f} "
                  f"v={r.forward_speed():+.3f} slip={100 * r.slip():5.0f}% sink={np.round(1000 * r.sinkage()).astype(int)} "
                  f"sand {sand.ms:.0f} ms", flush=True)
    LOOK["xray"] = False
    enc.close()
    os.replace(part, path)
    print(f"  wrote {path} ({enc.n} frames, {time.perf_counter() - t0:.0f} s); release {story.release} "
          f"power_lost {story.power_lost}; slide v mean {np.mean(story.slide_v) if story.slide_v else 0:+.3f} "
          f"min {np.min(story.slide_v) if story.slide_v else 0:+.3f} m/s", flush=True)
    if story.skid_log:
        a_ = np.array(story.skid_log)
        print(f"  skid: start {story.skid}; {a_[-1, 0] - a_[0, 0]:.1f} s, {story.slid_skid:.2f} m; v min {a_[:, 2].min():+.3f} "
              f"mean {a_[:, 2].mean():+.3f} m/s; omega mean {a_[:, 3].mean():+.2f}; slip median {np.median(a_[:, 4]):+.2f}; "
              f"runaway {story.runaway}; blowup {story.blowup}", flush=True)
        np.savetxt(os.path.join(OUT, f"skid_log{tag}.csv"), a_, delimiter=",",
                   header="t,theta,v,omega,slip,x", comments="")
    print(f"  story: stall {story.stall}; release {story.release}; slid {story.slid:.2f} m; "
          f"climbed {story.x_max - story.x_start:.2f} m; sim {np.median(t_sim_ms):.1f} ms/frame median "
          f"(mean {np.mean(t_sim_ms):.1f}) while recording", flush=True)
    write_log(f"story_log{tag}.csv")
    sys.exit(0)

if BENCH:
    nf = cli_arg("--frames", 600, int)
    sim_ms, pub_ms, ren_ms = [], [], []
    for i in range(nf):
        t = i * DT
        a = time.perf_counter()
        sim_frame(W_DRIVE if t > 1.0 else 0.0, 0.0)
        b = time.perf_counter()
        present(1.0, DT, 0)
        c = time.perf_counter()
        renderer.render(scene, camera)
        d = time.perf_counter()
        if i >= 120:
            sim_ms.append((b - a) * 1000)
            pub_ms.append((c - b) * 1000)
            ren_ms.append((d - c) * 1000)
    tot = np.array(sim_ms) + np.array(pub_ms) + np.array(ren_ms)
    print(f"bench {W_}x{H_}: sim {np.mean(sim_ms):.2f} ms (sand {sand.ms:.2f}, nsub {sand.nsub}), "
          f"publish {np.mean(pub_ms):.2f} ms, render {np.mean(ren_ms):.2f} ms -> "
          f"{np.mean(tot):.2f} ms/frame = {1000 / np.mean(tot):.1f} fps", flush=True)
    sys.exit(0)

# --- interactive ---------------------------------------------------------------

ui = tp.ImguiContext(canvas, renderer)
edge = {}
wall_prev = None
sim_debt = 0.0
fps_ema = 60.0
shot_no = 0


def pressed(k):
    down = canvas.is_key_down(k)
    was = edge.get(k, False)
    edge[k] = down
    return down and not was


def draw_ui():
    r = rover
    tp.imgui.set_next_window_pos(8, 8)
    tp.imgui.set_next_window_size(360, 0)
    tp.imgui.begin("TP-1 tilt bed")
    tp.imgui.text(f"slope {sand.theta:5.1f} deg (target {state['target']:4.1f})   fps {fps_ema:5.1f}")
    tp.imgui.text(f"speed {r.forward_speed() * 100:6.1f} cm/s  omega_cmd {state['w_cmd']:4.2f} rad/s")
    tp.imgui.text(f"slip  {100 * r.slip():6.1f} %   (1 - v / (omega r_eff))")
    tp.imgui.text(f"sand  {sand.ms:5.1f} ms/frame  ({sand.nsub} substeps)")
    tp.imgui.separator()
    tp.imgui.text("wheel  sink mm  load N  omega  tau Nm")
    sk = r.sinkage()
    for i, k in enumerate(WN):
        tp.imgui.text(f"  {k}   {1000 * sk[i]:6.1f}  {r.W[i]:6.0f}  {r.spin[i]:5.2f}  {r.tau[i]:6.1f}")
    tp.imgui.separator()
    tp.imgui.text("W/S speed  A/D steer  SPACE stop  UP/DOWN tilt")
    tp.imgui.text("C camera  V x-ray  T tilt target  R reset  F frame")
    tp.imgui.end()


AUTO_FRAMES = cli_arg("--frames", 0, int)     # windowed timing: autopilot N frames, then exit
_ft = dict(n=0, t0=0.0, sim=0.0, steps=0)


def frame():
    global wall_prev, sim_debt, fps_ema, view, shot_no
    t0 = time.perf_counter()
    wall = 0.0 if wall_prev is None else t0 - wall_prev
    wall_prev = t0
    dt = min(wall, 0.1)
    if AUTO_FRAMES > 0:
        _ft["n"] += 1
        state["w_cmd"] = W_DRIVE if rover.phase == "free" else 0.0
        if _ft["n"] == 120:
            _ft["t0"], _ft["sim"], _ft["steps"] = t0, 0.0, 0
        if _ft["n"] == 120 + AUTO_FRAMES:
            ms = (t0 - _ft["t0"]) * 1000.0 / AUTO_FRAMES
            print(f"windowed {W_}x{H_}: {AUTO_FRAMES} frames, {ms:.2f} ms/frame = {1000.0 / ms:.1f} fps; "
                  f"sim {_ft['sim'] / max(_ft['steps'], 1):.2f} ms/step (sand {sand.ms:.2f}), "
                  f"{_ft['steps'] / AUTO_FRAMES:.2f} steps/frame; rover x {rover.origin()[0]:+.2f}", flush=True)
            canvas.close()
            return
    if canvas.is_key_down("W"):
        state["w_cmd"] = min(1.5, state["w_cmd"] + 0.6 * dt)
    if canvas.is_key_down("S"):
        state["w_cmd"] = max(-1.0, state["w_cmd"] - 0.6 * dt)
    if canvas.is_key_down("SPACE"):
        state["w_cmd"] = 0.0
    steer = (0.35 if canvas.is_key_down("D") else 0.0) - (0.35 if canvas.is_key_down("A") else 0.0)
    if canvas.is_key_down("UP"):
        state["target"] = min(30.0, state["target"] + 3.0 * dt)
    if canvas.is_key_down("DOWN"):
        state["target"] = max(0.0, state["target"] - 3.0 * dt)
    if pressed("C"):
        view = 1 - view
    if pressed("V") and grains is not None:
        LOOK["xray"] = not LOOK["xray"]           # the X-ray cutaway through the rear-right wheel
    if pressed("T"):
        # the tilt target steps through 0/10/20/25/30/35; the bed follows, damped, <= 1 deg/s
        steps = [0.0, 10.0, 20.0, 25.0, 30.0, 35.0]
        nxt = [a for a in steps if a > state["target"] + 0.5]
        state["target"] = nxt[0] if nxt else 0.0
    if pressed("R"):
        sand._make(state["theta"])
        sand.touched.zero_()
        if hasattr(sand, "stamp"):
            sand.stamp.zero_()
        rover.reset(X_START)
    if pressed("F"):
        renderer.save_frame(scene, camera, os.path.join(OUT, f"frame_{shot_no:03d}.png"))
        shot_no += 1
    # D steers right: right wheels slower
    steer_w = -steer
    sim_debt = min(sim_debt + wall, 3.0 * DT)
    while sim_debt >= DT:
        sim_debt -= DT
        tilt_follow(DT)
        ts = time.perf_counter()
        sim_frame(state["w_cmd"], steer_w)
        _ft["sim"] += (time.perf_counter() - ts) * 1000.0
        _ft["steps"] += 1
    present(sim_debt / DT, dt, 6 if LOOK["xray"] else view)
    renderer.render(scene, camera)
    ui.render(draw_ui)
    fps_ema += (1.0 / max(time.perf_counter() - t0, 1e-4) - fps_ema) * 0.05


canvas.animate(frame)
