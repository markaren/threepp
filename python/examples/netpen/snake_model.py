"""An underwater snake robot on the Mamba swimming model, simulated with PhysX from Python.

Pure code: numpy + threepp's PhysX bindings only (no renderer, no Warp, no scene). The scene
(snake_netpen.py) and the validation sweeps (snake_sweep.py) import this module; on its own:

    python snake_model.py --selftest

SOURCE. E. Kelasidi, P. Liljeback, K. Y. Pettersen, J. T. Gravdahl (2015), "Experimental
investigation of efficient locomotion of underwater snake robots for lateral undulation and
eel-like motion patterns", Robotics and Biomimetics 2:8. Open access:
https://pmc.ncbi.nlm.nih.gov/articles/PMC4679098/

What the paper says (quoted from the PMC text; equation numbers are the paper's):

  Basic notations: "The underwater snake robot consists of n rigid links of equal length 2l
  interconnected by n - 1 joints. The links are assumed to have the same mass m and moment of
  inertia J = 1/3 m l^2." "... the joint angle of joint i in 1..n-1 is given by
  phi_i = theta_i - theta_(i-1)."

  Fluid forces on link i (per link, link frame: x along the link, y normal to it):
    (7)  [f_Dx^I ; f_Dy^I]   = -[c_t C_th, -c_n S_th ; c_t S_th, c_n C_th] [V_rx ; V_ry]
    (8)  [f_Dx^II; f_Dy^II]  = -[c_t C_th, -c_n S_th ; c_t S_th, c_n C_th] sgn([V_rx; V_ry]) [V_rx^2 ; V_ry^2]
    (9)  [V_rx ; V_ry]       =  [C_th, S_th ; -S_th, C_th] [X_dot - V_x ; Y_dot - V_y]
    (6)  added mass: f_A = -[mu_n S^2, -mu_n S C ; -mu_n S C, mu_n C^2] [X_ddot ; Y_ddot]
                          - [-mu_n S C, -mu_n S^2 ; mu_n C^2, mu_n S C] V_a theta_dot
         "the added mass parameter in the x direction is considered equal to zero (mu_t = 0),
          because the added mass of a slender body in the longitudinal direction can be
          neglected compared to the body mass"
    (10) fluid torques: tau = -Lambda_1 theta_ddot - Lambda_2 theta_dot - Lambda_3 theta_dot |theta_dot|,
         Lambda_k = lambda_k I_n.
         "The coefficients c_t, c_n, lambda_2, lambda_3 represent the drag forces parameters due to
          the pressure difference between the two sides of the body, and the parameters mu_n,
          lambda_1 represent the added mass of the fluid carried by the moving body."

  Coefficients (simulation section): "c_t = 0.2639, c_n = 4.2, mu_n = 0.3957,
  lambda_1 = 2.2988e-7, lambda_2 = 4.3103e-4, for the complex model and c_t = 0.45, c_n = 5,
  mu_n = 0.4, lambda~_1 = 0.5, lambda~_2 = 20, lambda~_3 = 0.01 for the control-oriented model."

  Gait (22): phi_i*(t) = alpha g(i,n) sin(omega t + (i-1) delta) + phi_0; "lateral undulation and
  eel-like motion are achieved by choosing g(i,n) = 1 and g(i,n) = (n-i)/(n+1)".
  Joint controller (23): u_i = phi_ddot_i* + k_d (phi_dot_i* - phi_dot_i) + k_p (phi_i* - phi_i),
  "k_p = 20, k_d = 5".
  Power (28): P_avg = 1/T int_0^T sum_i u_i(t) phi_dot_i(t) dt, "calculated considering the
  absolute value of the theoretical joint power".
  Mamba: "18 identical joint modules mounted horizontally and vertically in an alternating
  fashion ... the angles for the joints with vertical rotating axis were set to zero degrees. In
  this case, the kinematics of the snake robot corresponds to a planar snake robot with links of
  length 2l = 0.18 m and mass m ~ 0.8 kg." "slightly positive buoyancy". Mamba-matched
  simulation: "n = 9 links, each one having length 2l = 0.18 m and mass m = 0.8 kg", elliptic
  section 2a = 2 x 0.055 m, 2b = 2 x 0.05 m, rho = 1000 kg/m^3, C_f = 0.03, C_D = 1, C_A = 1, C_M = 1.

Parameter table used here (SnakeParams defaults):
    n 9 links (0.8 kg each, i.e. the paper's m per EFFECTIVE link = one horizontal + one vertical
    module), 2l 0.18 m, capsule radius 0.0525 m (the mean of the 0.055 / 0.050 semi-axes; the
    drag does not see the radius, only the coefficients), joint PD kp 20 N m/rad, kd 5 N m s/rad,
    torque cap 10 N m, joint limits +-60 deg, 240 Hz.

Conventions and choices, each with its reason:
  * Numbering. Link 1 (index 0 here) is the TAIL, link n the HEAD; joint i sits between link i and
    link i+1. Reason: with (i-1) delta the phase grows toward the head, so the body wave travels
    head -> tail and the robot swims toward link n; and eel-like g(i,n) = (n-i)/(n+1) is then
    largest at the tail joint (0.8) and smallest at the head joint (0.1), which is the paper's
    "increasing amplitude from head to tail". Joint angle phi_i = yaw(link i+1) - yaw(link i)
    about world +Y (PhysX's joint position), i.e. headward minus tailward, as in the paper.
  * 3D. The paper's planar model is applied in each link's frame: tangential coefficients along
    the link axis, the normal coefficients (c_n, mu_n) on BOTH normal axes (lateral and vertical),
    the fluid torque (10) about the link's vertical axis. Relative velocity is to params.current.
  * Added mass (6), generalised to 3D: f_A = sum_{e in normals} -mu_n e [e . a - (omega x e) . V_c].
    In 2D this is exactly (6) (the V_a theta_dot term is mu_n n d/dt(n . V_c)). The link
    acceleration a is the backward difference of the link's own velocity over the LAST substep
    (a lagged explicit scheme). Stability proof: with M the articulated mass matrix and A the
    projected added-mass matrix, M (v' - v)/dt = F - A (v - v_prev)/dt has the characteristic
    roots z = 1 and z = -eig(M^-1 A); since A <= (mu_n / m) sum J_i^T m J_i <= (mu_n / m) M,
    |z| <= mu_n / m = 0.49 (complex) / 0.50 (control) < 1. The selftest checks it in the time
    domain (a constant lateral push gives a = F / (m + mu_n) with the Nyquist ringing gone).
  * The control-oriented set. Its lambda~_1..3 belong to the AVERAGED model's heading equation
    (19f: v_theta_dot = -lambda~_1/(1+lambda~_3) v_theta + ...), not to per-link torques, so they
    have no place in (10). The 'control' set therefore uses its own c_t, c_n, mu_n with the
    complex set's lambda_1, lambda_2 (a deviation, stated; lambda_3 is not given for the complex
    model and is 0 in both).
  * The paper's complex-model coefficients were stated for its generic study (links l = 0.14 m,
    m = 0.6597 kg); for the Mamba-matched run it computed them from C_f/C_D/C_A/C_M "by using
    equations derived in [2]" without printing the values. We keep the stated numbers (see the
    sweep report for what that implies).
  * Controller: the paper's (23), realised with the PhysX drive (position target phi*, velocity
    target phi_dot*, stiffness kp, damping kd, force cap) plus the feedforward phi_ddot* applied
    as an equal-and-opposite torque pair about the joint axis. feedforward=False gives the plain
    PD u = kp (phi* - phi) - kd phi_dot. Logged u = the law evaluated at the start of each substep
    (PD part clipped at the cap) + feedforward; P_abs = sum |u phi_dot| (the paper's), P_net = sum u phi_dot.
  * PhysX damps every articulation link by 0.05 /s (linear and angular); Python cannot change it,
    so each substep adds +0.05 m v and +0.05 I omega back (selftest: coast-down keeps >= 99 %).
  * Buoyancy: gravity 0 (neutral) + a righting couple per link from buoyancy m g acting 5 mm above
    the centre of mass, so the chain stays upright.

THRUSTERS (optional; params.thrusters, empty by default so the validation sweep is the paper's pure
undulation). NTNU's later design, the underwater swimming manipulator behind Eelume, carries
"side mounted thrusters that provide longitudinal thrust" and tunnel thruster modules
(Sverdrup-Thygeson, Kelasidi, Pettersen, Gravdahl 2016, IFAC-PapersOnLine 49-23). thruster_layout()
is that arrangement on this 9-link body: a pair of side-mounted ducted thrusters on the second module
behind the head (longitudinal), a tunnel thruster behind the head and one ahead of the tail (lateral).
Each is a force along its axis at its mounting point (so a lateral one also yaws the body), spun up
with a first-order lag; electrical power from actuator-disc momentum theory, P = T (V + v_i) / eta,
v_i = -V/2 + sqrt(V^2/4 + T / (2 rho A)), V the inflow along the axis. The thrust ratings and eta are
assumptions of this model (stated in thruster_layout), not Eelume data. The fluid model does not see
the thrusters' wash.
"""
import argparse
import math
import os
import sys
from dataclasses import dataclass, field

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.dirname(_HERE)                 # python/examples
_PY = os.path.dirname(_EX)                   # python
for _p in (_HERE, _EX, _PY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import threepp as tp

# The paper's coefficient sets. 'control' keeps its lambda~ only for the record (see docstring).
COEFFS = {
    "complex": dict(c_t=0.2639, c_n=4.2, mu_n=0.3957, lam1=2.2988e-7, lam2=4.3103e-4, lam3=0.0),
    "control": dict(c_t=0.45, c_n=5.0, mu_n=0.4, lam1=2.2988e-7, lam2=4.3103e-4, lam3=0.0,
                    lam_tilde=(0.5, 20.0, 0.01)),
}
PATTERNS = ("lateral", "eel")
PHYSX_LINK_DAMPING = 0.05


@dataclass
class SnakeParams:
    n: int = 9                           # links
    link_length: float = 0.18            # 2l, joint to joint (m)
    radius: float = 0.0525               # capsule radius (m); visual + collider only
    mass: float = 0.8                    # per link (kg)
    coeffs: str = "complex"              # 'complex' | 'control'
    kp: float = 20.0                     # N m / rad
    kd: float = 5.0                      # N m s / rad
    torque_cap: float = 10.0             # N m (drive_limits_are_forces=True)
    joint_limit: float = math.radians(60.0)
    dt: float = 1.0 / 240.0
    current: tuple = (0.0, 0.0, 0.0)     # water velocity, world frame (m/s)
    feedforward: bool = True             # the paper's (23): velocity target + phi_ddot* torque
    fluid: bool = True                   # drag + added mass + fluid torque
    added_mass: bool = True
    damping_comp: bool = True            # add PhysX's 0.05 /s link damping back
    righting: bool = True
    cb_offset: float = 0.005             # centre of buoyancy above the centre of mass (m)
    g: float = 9.81
    ramp_time: float = 2.0               # alpha / omega / delta / pattern changes settle to 99 % in this
    phi0_tau: float = 0.5                # phi0 (steering) first-order-like settling (s)
    thrusters: list = field(default_factory=list)   # [Thruster]; thruster_layout() for the USM set
    rho: float = 1025.0                  # sea water (kg/m^3), thruster power only
    thrust_eta: float = 0.5              # propeller x motor efficiency (assumed)
    extra: dict = field(default_factory=dict)

    def coef(self):
        return COEFFS[self.coeffs]


@dataclass
class Thruster:
    """One thruster: on link `link` (0 = tail), at `pos` and thrusting along `axis`, both in the link's
    (along toward the head, lateral to starboard, up) axes. Thrust in [t_min, t_max] N."""
    name: str
    link: int
    pos: tuple
    axis: tuple
    t_max: float
    t_min: float
    duct_d: float                        # m, the disc area for the momentum-theory power
    tau: float = 0.15                    # s, spin-up


SIDE_THRUSTER_LINK, BOW_TUNNEL_LINK, STERN_TUNNEL_LINK = 6, 7, 1
SIDE_THRUSTER_Y = 0.079                  # duct axis off the body axis: the 0.049 shell + a 0.03 duct


def thruster_layout():
    """The swimming-manipulator arrangement on the Mamba body (ratings assumed: small 4 N ducted
    thrusters on the flanks, 2.5 N tunnel thrusters; a 0.5 m/s cruise needs ~1.8 N in total)."""
    return [Thruster("side_port", SIDE_THRUSTER_LINK, (0.0, -SIDE_THRUSTER_Y, 0.0), (1.0, 0.0, 0.0), 4.0, -2.5, 0.05),
            Thruster("side_stbd", SIDE_THRUSTER_LINK, (0.0, SIDE_THRUSTER_Y, 0.0), (1.0, 0.0, 0.0), 4.0, -2.5, 0.05),
            Thruster("tunnel_bow", BOW_TUNNEL_LINK, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2.5, -2.5, 0.045),
            Thruster("tunnel_stern", STERN_TUNNEL_LINK, (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), 2.5, -2.5, 0.045)]


def quat_to_R(q):
    """(n,4) xyzw -> (n,3,3) body->world."""
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _yaw_dir(h):
    """Unit direction of heading h (rad) about world +Y: h = 0 is +X, positive turns +X toward -Z."""
    return np.array([math.cos(h), 0.0, -math.sin(h)])


def _v3(a):
    return tp.Vector3(float(a[0]), float(a[1]), float(a[2]))


def _axis_angle(u, a):
    """Rotation matrix about unit axis u by angle a (Rodrigues)."""
    K = np.array([[0.0, -u[2], u[1]], [u[2], 0.0, -u[0]], [-u[1], u[0], 0.0]])
    return np.eye(3) + math.sin(a) * K + (1.0 - math.cos(a)) * (K @ K)


class _Smooth:
    """Critically damped second-order follower: x -> target with 99 % settled at `settle` s."""

    def __init__(self, x, settle):
        self.x = np.array(x, dtype=float)
        self.xd = np.zeros_like(self.x)
        self.xdd = np.zeros_like(self.x)
        self.target = self.x.copy()
        self.w = 6.64 / max(settle, 1e-6)

    def jump(self, x):
        self.x = np.array(x, dtype=float); self.target = self.x.copy()
        self.xd[:] = 0.0; self.xdd[:] = 0.0

    def step(self, dt):
        w = self.w
        self.xdd = w * w * (self.target - self.x) - 2.0 * w * self.xd
        self.xd = self.xd + self.xdd * dt
        self.x = self.x + self.xd * dt


class Snake:
    """A planar-jointed (yaw) underwater snake robot as one PhysX articulation.

    Link index 0 is the tail, n-1 the head; joint k (0-based) joins link k to link k+1.
    Heading h: direction (cos h, 0, -sin h) (rotation about world +Y). phi0 > 0 turns toward +h.
    """

    def __init__(self, world, params=None, origin=(0.0, 0.0, 0.0), heading=0.0):
        self.world = world
        self.p = p = params if params is not None else SnakeParams()
        c = p.coef()
        self.c_t, self.c_n, self.mu_n = c["c_t"], c["c_n"], c["mu_n"]
        self.lam1, self.lam2, self.lam3 = c["lam1"], c["lam2"], c["lam3"]
        n, L, r = p.n, p.link_length, p.radius
        self.n = n
        self.current = np.array(p.current, dtype=float)
        d = _yaw_dir(heading)
        up = np.array([0.0, 1.0, 0.0])
        o = np.array(origin, dtype=float)
        h_cyl = max(L - 2.0 * r, 1e-4)
        vol = math.pi * r * r * h_cyl + 4.0 / 3.0 * math.pi * r ** 3
        density = p.mass / vol
        self.art = world.create_articulation(fixed_base=False, solver_position_iterations=8,
                                             disable_self_collision=True, drive_limits_are_forces=True)
        self.meshes, self.links = [], []
        centers = [o + d * (k - (n - 1) / 2.0) * L for k in range(n)]
        for k in range(n):
            m = tp.Mesh(tp.CapsuleGeometry(r, h_cyl), tp.MeshStandardMaterial())
            m.rotation.y = float(heading)            # Euler XYZ: R = Ry(h) Rz(-pi/2): capsule Y -> d
            m.rotation.z = -math.pi / 2.0
            m.position.set(*[float(v) for v in centers[k]])
            if k == 0:
                lk = self.art.add_link(m, parent=None, density=density)
            else:
                anchor = centers[k] - d * L / 2.0
                lk = self.art.add_link(m, parent=self.links[-1], density=density, axis=(0.0, 1.0, 0.0),
                                       anchor=tuple(float(v) for v in anchor), lower=-p.joint_limit,
                                       upper=p.joint_limit, stiffness=p.kp, damping=p.kd,
                                       max_force=p.torque_cap, joint_type="revolute")
            self.meshes.append(m); self.links.append(lk)
        self.art.finalize()
        self.mass = np.array([lk.mass for lk in self.links])
        # Body-frame axes from the build pose (every link has the same build orientation).
        _, q0 = self._read_poses()
        R0 = quat_to_R(q0)[0]
        self.t_loc = R0.T @ d                     # along the link, toward the head
        self.up_loc = R0.T @ up                   # the yaw-joint axis / vertical normal
        self.lat_loc = R0.T @ np.cross(d, up)     # lateral normal
        self.anc_loc = np.array([0.5 * L * self.t_loc])[0]   # joint to the child, in the PARENT frame
        # Capsule inertia in the body frame (PhysX computes it from the same shape and density).
        m_cyl = density * math.pi * r * r * h_cyl
        m_s = density * 4.0 / 3.0 * math.pi * r ** 3
        I_ax = m_cyl * r * r / 2.0 + m_s * 0.4 * r * r
        I_pe = m_cyl * (h_cyl ** 2 / 12.0 + r * r / 4.0) + m_s * (0.4 * r * r + h_cyl ** 2 / 4.0 + 3.0 * h_cyl * r / 8.0)
        self.I_loc = I_pe * np.eye(3) + (I_ax - I_pe) * np.outer(self.t_loc, self.t_loc)
        # Gait state.
        nj = n - 1
        self._pattern = "lateral"
        self._amp = _Smooth(np.zeros(nj), p.ramp_time)
        self._omega = _Smooth([0.0], p.ramp_time)
        self._delta = _Smooth([0.0], p.ramp_time)
        self._phi0 = _Smooth([0.0], p.phi0_tau * 6.64 / 4.74)   # ~95 % in ~phi0_tau*... (smooth steering)
        self._shape = _Smooth(np.zeros(nj), 0.4)                   # per-joint bend (follow-the-leader)
        self._psi = 0.0
        self.gait_on = False
        self.phi_ref = np.zeros(nj); self.phid_ref = np.zeros(nj); self.phidd_ref = np.zeros(nj)
        # Bookkeeping.
        self.time = 0.0
        self.energy_abs = 0.0
        self.energy_net = 0.0
        self.power_abs = 0.0
        self.power_net = 0.0
        self.peak_torque = 0.0
        self.u = np.zeros(nj)
        self.extra_force = None                   # optional (n,3) world forces (tests / disturbances)
        self.extra_torque = None                  # optional (n,3) world torques (contact at a point)
        self.contact = None                       # optional f(pos, R, v, w) -> (F, T) (n,3) each, every substep
        # Thrusters: link-local mounting point and axis in the body basis.
        th = list(p.thrusters)
        self.thrusters = th
        B = np.stack([self.t_loc, self.lat_loc, self.up_loc], 1)   # (along, lateral, up) -> link-local
        self.thr_link = np.array([t.link for t in th], dtype=int)
        self.thr_pos = np.array([B @ np.asarray(t.pos, float) for t in th]).reshape(-1, 3)
        self.thr_axis = np.array([B @ np.asarray(t.axis, float) for t in th]).reshape(-1, 3)
        self.thr_cmd = np.zeros(len(th)); self.thr = np.zeros(len(th))
        self.power_thr = 0.0; self.energy_thr = 0.0
        self._v_prev = None
        self._thd_prev = None
        self.last_v = np.zeros((n, 3)); self.last_w = np.zeros((n, 3))
        self.last_fluid = np.zeros((n, 3)); self.last_added = np.zeros((n, 3))
        self._handle = world.on_pre_substep(self._substep)

    # ------------------------------------------------------------------ state
    def _read_poses(self):
        pos = np.empty((self.n, 3)); q = np.empty((self.n, 4))
        for k, lk in enumerate(self.links):
            a = lk.position; b = lk.quaternion
            pos[k] = (a.x, a.y, a.z); q[k] = (b.x, b.y, b.z, b.w)
        return pos, q

    def link_poses(self):
        """(positions (n,3), quaternions (n,4) xyzw), world frame, index 0 = tail, n-1 = head."""
        return self._read_poses()

    def head_pose(self):
        """(position (3,), quaternion (4,) xyzw, forward unit (3,)) of the head link."""
        a = self.links[-1].position; b = self.links[-1].quaternion
        q = np.array([b.x, b.y, b.z, b.w])
        return np.array([a.x, a.y, a.z]), q, quat_to_R(q[None])[0] @ self.t_loc

    def com(self):
        pos, _ = self._read_poses()
        return (self.mass[:, None] * pos).sum(0) / self.mass.sum()

    def link_yaws(self):
        _, q = self._read_poses()
        t = quat_to_R(q) @ self.t_loc
        return np.arctan2(-t[:, 2], t[:, 0])

    def mean_heading(self):
        """theta_bar = mean link yaw (rad, about +Y, 0 = +X), unwrapped along the chain."""
        return float(np.mean(np.unwrap(self.link_yaws())))

    def joint_angles(self):
        return np.asarray(self.art.joint_positions(), dtype=float)

    def velocities(self, pos=None, q=None):
        """Exact link CoM velocities (n,3) and angular velocities (n,3), world frame, from the root
        velocity and the joint rates along the chain."""
        if pos is None:
            pos, q = self._read_poses()
        R = quat_to_R(q)
        rv = np.asarray(self.art.root_velocity(), dtype=float)
        qd = np.asarray(self.art.joint_velocities(), dtype=float)
        return self._chain(pos, R, rv, qd)

    def _chain(self, pos, R, rv, qd):
        a = R[:-1] @ self.up_loc                        # joint k axis (world), from the parent
        rj = pos[:-1] + R[:-1] @ self.anc_loc           # joint k anchor (world)
        w = np.empty((self.n, 3)); v = np.empty((self.n, 3))
        w[0] = rv[3:]; v[0] = rv[:3]
        w[1:] = rv[3:] + np.cumsum(qd[:, None] * a, axis=0)
        dv = np.cross(w[:-1], rj - pos[:-1]) + np.cross(w[1:], pos[1:] - rj)
        v[1:] = rv[:3] + np.cumsum(dv, axis=0)
        return v, w

    def fk(self, root_pos, root_R, q):
        """Forward kinematics: link centres (n,3) and rotations (n,3,3) from the root pose and the
        joint angles (joint k rotates link k+1 about the shared body axis up_loc by +q_k)."""
        R = np.empty((self.n, 3, 3)); c = np.empty((self.n, 3))
        R[0] = root_R; c[0] = root_pos
        for k in range(1, self.n):
            R[k] = R[k - 1] @ _axis_angle(self.up_loc, q[k - 1])
            c[k] = c[k - 1] + R[k - 1] @ self.anc_loc + R[k] @ self.anc_loc
        return c, R

    # ------------------------------------------------------------------ gait
    def set_gait(self, pattern="lateral", alpha=0.0, omega=0.0, delta=0.0, phi0=0.0, ramp=True):
        """phi_k*(t) = alpha g(k) sin(psi + k delta) + phi0 (radians, rad/s; k = 0 at the tail joint).
        Changes settle smoothly over params.ramp_time (phi0 over params.phi0_tau) unless ramp=False."""
        if pattern not in PATTERNS:
            raise ValueError(f"pattern must be one of {PATTERNS}")
        n = self.n
        i = np.arange(1, n)                               # the paper's joint index 1..n-1
        g = np.ones(n - 1) if pattern == "lateral" else (n - i) / (n + 1.0)
        self._pattern = pattern
        self._amp.target = float(alpha) * g
        self._omega.target = np.array([float(omega)])
        self._delta.target = np.array([float(delta)])
        self._phi0.target = np.array([float(phi0)])
        if not ramp:
            self._amp.jump(self._amp.target); self._omega.jump(self._omega.target)
            self._delta.jump(self._delta.target); self._phi0.jump(self._phi0.target)
        self.gait_on = True

    def set_phi0(self, phi0):
        """Steering offset only (radians), smoothed by params.phi0_tau."""
        self._phi0.target = np.array([float(phi0)])
        self.gait_on = True

    def set_shape(self, phi):
        """Per-joint bend targets (rad, joint k = 0 at the tail), added to the gait; settles in 0.4 s."""
        self._shape.target = np.asarray(phi, float).copy()
        self.gait_on = True

    def set_current(self, v):
        self.current = np.array(v, dtype=float)

    def set_thrust(self, cmd):
        """Thrust commands (N), one per params.thrusters entry, clipped to each rating."""
        th = self.thrusters
        self.thr_cmd = np.clip(np.asarray(cmd, float), [t.t_min for t in th], [t.t_max for t in th])

    def thrust_alloc(self, surge, yaw_moment, sway=0.0):
        """thruster_layout() commands for a surge force (N, headward), a yaw moment (N m about +Y,
        + turns toward +heading) and a sway force (N, to starboard), with the body straight. The two
        tunnel thrusters sit 6 links apart about the chain's middle: a lateral force F at x ahead of
        the middle yaws by -x F (+Y up, starboard lateral)."""
        L = self.p.link_length
        mid = (self.n - 1) / 2.0
        xb, xs = (BOW_TUNNEL_LINK - mid) * L, (STERN_TUNNEL_LINK - mid) * L
        # sway = Fb + Fs, yaw = -(xb Fb + xs Fs)
        A = np.array([[1.0, 1.0], [-xb, -xs]])
        fb, fs = np.linalg.solve(A, [sway, yaw_moment])
        return np.array([0.5 * surge, 0.5 * surge, fb, fs])

    def gait_state(self):
        return dict(pattern=self._pattern, amp=self._amp.x.copy(), omega=float(self._omega.x[0]),
                    delta=float(self._delta.x[0]), phi0=float(self._phi0.x[0]))

    def _gait_step(self, dt):
        for s in (self._amp, self._omega, self._delta, self._phi0, self._shape):
            s.step(dt)
        self._psi += float(self._omega.x[0]) * dt
        k = np.arange(self.n - 1)
        A, Ad, Add = self._amp.x, self._amp.xd, self._amp.xdd
        om, omd = float(self._omega.x[0]), float(self._omega.xd[0])
        de, ded, dedd = float(self._delta.x[0]), float(self._delta.xd[0]), float(self._delta.xdd[0])
        arg = self._psi + k * de
        s, c = np.sin(arg), np.cos(arg)
        argd = om + k * ded
        argdd = omd + k * dedd
        lim = 0.98 * self.p.joint_limit
        sh = self._shape
        self.phi_ref = np.clip(A * s + self._phi0.x[0] + sh.x, -lim, lim)
        self.phid_ref = Ad * s + A * c * argd + self._phi0.xd[0] + sh.xd
        self.phidd_ref = Add * s + 2.0 * Ad * c * argd - A * s * argd ** 2 + A * c * argdd + self._phi0.xdd[0] + sh.xdd

    # ------------------------------------------------------------------ the substep hook
    def _substep(self, dt):
        p = self.p
        pos, q = self._read_poses()
        R = quat_to_R(q)
        rv = np.asarray(self.art.root_velocity(), dtype=float)
        qd = np.asarray(self.art.joint_velocities(), dtype=float)
        qq = np.asarray(self.art.joint_positions(), dtype=float)
        v, w = self._chain(pos, R, rv, qd)
        F = np.zeros((self.n, 3)); T = np.zeros((self.n, 3))
        t = R @ self.t_loc; e1 = R @ self.lat_loc; e2 = R @ self.up_loc
        if p.fluid:
            vr = v - self.current
            vt = np.einsum("ij,ij->i", vr, t); v1 = np.einsum("ij,ij->i", vr, e1); v2 = np.einsum("ij,ij->i", vr, e2)
            ft = -(self.c_t * vt + self.c_t * np.abs(vt) * vt)          # (7) + (8), tangential
            f1 = -(self.c_n * v1 + self.c_n * np.abs(v1) * v1)          # normal, lateral
            f2 = -(self.c_n * v2 + self.c_n * np.abs(v2) * v2)          # normal, vertical
            Fd = ft[:, None] * t + f1[:, None] * e1 + f2[:, None] * e2
            F += Fd
            self.last_fluid = Fd
            thd = np.einsum("ij,ij->i", w, e2)                          # yaw rate about the link's vertical axis
            thdd = np.zeros(self.n) if self._thd_prev is None else (thd - self._thd_prev) / dt
            tau = -self.lam1 * thdd - self.lam2 * thd - self.lam3 * thd * np.abs(thd)   # (10)
            T += tau[:, None] * e2
            self._thd_prev = thd
            if p.added_mass and self._v_prev is not None:
                acc = (v - self._v_prev) / dt                          # lagged: the last substep's
                fa = np.zeros((self.n, 3))
                for e in (e1, e2):                                      # (6) on both normal axes
                    ea = np.einsum("ij,ij->i", e, acc)
                    edV = np.einsum("ij,j->i", np.cross(w, e), self.current)
                    fa += (-self.mu_n * (ea - edV))[:, None] * e
                F += fa
                self.last_added = fa
        self._v_prev = v
        if p.damping_comp:
            F += PHYSX_LINK_DAMPING * self.mass[:, None] * v
            Iw = np.einsum("nij,jk,nlk,nl->ni", R, self.I_loc, R, w)     # R I R^T w
            T += PHYSX_LINK_DAMPING * Iw
        if p.righting:
            B = self.mass * p.g                                          # neutral: rho V g = m g
            arm = p.cb_offset * e2
            T += np.cross(arm, B[:, None] * np.array([0.0, 1.0, 0.0]))
        if self.extra_force is not None:
            F += self.extra_force
        if self.extra_torque is not None:
            T += self.extra_torque
        if self.contact is not None:
            cF, cT = self.contact(pos, R, v, w)
            F += cF; T += cT
        if len(self.thrusters):
            self._thrust_step(dt, R, v, w, F, T)
        # Gait: drive targets + the paper's feedforward.
        nj = self.n - 1
        if self.gait_on:
            self._gait_step(dt)
        self.art.set_drive_targets(self.phi_ref.astype(np.float32))
        ff = self.phidd_ref if p.feedforward else np.zeros(nj)
        if p.feedforward:
            for k in range(nj):
                self.links[k + 1].set_drive_velocity(float(self.phid_ref[k]))
            a = R[:-1] @ self.up_loc
            T[1:] += ff[:, None] * a
            T[:-1] -= ff[:, None] * a
            u_pd = p.kp * (self.phi_ref - qq) + p.kd * (self.phid_ref - qd)
        else:
            u_pd = p.kp * (self.phi_ref - qq) - p.kd * qd
        u = np.clip(u_pd, -p.torque_cap, p.torque_cap) + ff
        self.u = u
        pw = u * qd
        self.power_abs = float(np.abs(pw).sum()); self.power_net = float(pw.sum())
        self.energy_abs += self.power_abs * dt; self.energy_net += self.power_net * dt
        self.peak_torque = max(self.peak_torque, float(np.abs(u).max()))
        for k, lk in enumerate(self.links):
            lk.add_force(_v3(F[k])); lk.add_torque(_v3(T[k]))
        self.last_v, self.last_w = v, w
        self.time += dt

    def _thrust_step(self, dt, R, v, w, F, T):
        """Spin-up lag, the force at each mounting point (force + moment on its link), and the
        momentum-theory power."""
        p = self.p
        tau = np.array([t.tau for t in self.thrusters])
        self.thr += (self.thr_cmd - self.thr) * (1.0 - np.exp(-dt / tau))
        P = 0.0
        for j, t in enumerate(self.thrusters):
            k = self.thr_link[j]
            r = R[k] @ self.thr_pos[j]
            a = R[k] @ self.thr_axis[j]
            f = self.thr[j] * a
            F[k] += f
            T[k] += np.cross(r, f)
            Tm = abs(self.thr[j])
            if Tm > 1e-9:
                vin = float((v[k] + np.cross(w[k], r) - self.current) @ a) * math.copysign(1.0, self.thr[j])
                V = max(vin, 0.0)
                A = math.pi * 0.25 * t.duct_d ** 2
                vi = -0.5 * V + math.sqrt(0.25 * V * V + Tm / (2.0 * p.rho * A))
                P += Tm * (V + vi) / p.thrust_eta
        self.power_thr = P
        self.energy_thr += P * dt

    def remove(self):
        self.world.remove_substep_callback(self._handle)


def make_world(dt=1.0 / 240.0, max_substeps=64):
    return tp.PhysxWorld(gravity=tp.Vector3(0, 0, 0), fixed_timestep=dt, max_substeps=max_substeps)


def roll_pitch(snake):
    """Max tilt (deg) of any link's vertical body axis from world +Y."""
    _, q = snake.link_poses()
    e2 = quat_to_R(q) @ snake.up_loc
    return float(np.degrees(np.arccos(np.clip(e2[:, 1], -1, 1))).max())


# ---------------------------------------------------------------------- selftest
def selftest():
    import gc
    ok = True

    def report(name, passed, msg):
        nonlocal ok
        ok &= bool(passed)
        print(f"{'PASS' if passed else 'FAIL'} {name}: {msg}", flush=True)

    dt = 1.0 / 240.0
    world = make_world(dt)
    D = 30.0                                                 # spacing between the test snakes (m)
    # A: the reference lateral gait (tests a, c). B/C: coast-down with/without damping compensation.
    # D: the 60 s stress gait. E: added mass under a constant lateral push (no drag).
    A = Snake(world, SnakeParams(), origin=(0, 0, 0), heading=0.0)
    B = Snake(world, SnakeParams(fluid=False), origin=(D, 0, 0), heading=0.0)
    C = Snake(world, SnakeParams(fluid=False, damping_comp=False), origin=(2 * D, 0, 0), heading=0.0)
    Dn = Snake(world, SnakeParams(), origin=(0, 0, D), heading=0.0)
    E = Snake(world, SnakeParams(), origin=(D, 0, D), heading=0.0)
    E.c_t = E.c_n = 0.0
    E.lam1 = E.lam2 = E.lam3 = 0.0
    A.set_gait("lateral", math.radians(30), math.radians(120), math.radians(30))
    Dn.set_gait("lateral", math.radians(40), math.radians(150), math.radians(30))
    v0 = 0.2
    for S in (B, C):
        for k, lk in enumerate(S.links):
            lk.add_impulse(_v3(S.mass[k] * v0 * _yaw_dir(0.0)))
    Fpush = 0.5
    E.extra_force = np.tile([0.0, 0.0, Fpush], (E.n, 1))    # lateral (+Z), every link
    # F: the side thrusters, 1 N each, body straight. G: the tunnel couple, 0.8 N m, no surge.
    Fs = Snake(world, SnakeParams(thrusters=thruster_layout()), origin=(2 * D, 0, D), heading=0.0)
    Gs = Snake(world, SnakeParams(thrusters=thruster_layout()), origin=(0, 0, 2 * D), heading=0.0)
    Fs.set_thrust([1.0, 1.0, 0.0, 0.0])
    M_G = 0.8
    Gs.set_thrust(Gs.thrust_alloc(0.0, M_G, 0.0))
    yawG = []
    com0 = A.com(); y0 = com0[1]
    fd_err = []; fk_err = []; step_err = []; root_err = []; vmax_y = 0.0; tilt_max = 0.0
    speeds_B = {}; speeds_C = {}
    E_rec = []
    nsteps = int(round(60.0 / dt))
    prev_pos = None
    for s in range(nsteps):
        t_now = s * dt
        # (a) velocity kinematics on A, 5..6 s in (gait fully on): (i) FK from PhysX's root pose and
        # joint angles reproduces PhysX's link poses (geometry and joint sign); (ii) the chain
        # velocity equals the central finite difference of that exact map along PhysX's reported
        # root velocity and joint rates; (iii) for the record, the FD of PhysX's own positions over
        # one substep, whose floor is set by the TGS integrator (seen on the ROOT link too).
        if 5.0 <= t_now < 6.0 and s % 24 in (0, 1):
            pos, q = A.link_poses()
            if s % 24 == 0:
                R = quat_to_R(q)
                rv = np.asarray(A.art.root_velocity(), float); qd = np.asarray(A.art.joint_velocities(), float)
                qq = A.joint_angles()
                cf, Rf = A.fk(pos[0], R[0], qq)
                fk_err.append(float(max(np.abs(cf - pos).max(), np.abs(Rf - R).max() * A.p.link_length)))
                v_ch, _ = A._chain(pos, R, rv, qd)
                h = 1e-6
                def moved(sign):
                    ang = np.linalg.norm(rv[3:])
                    Rr = (_axis_angle(rv[3:] / ang, sign * h * ang) if ang > 0 else np.eye(3)) @ R[0]
                    return A.fk(pos[0] + sign * h * rv[:3], Rr, qq + sign * h * qd)[0]
                fd = (moved(1) - moved(-1)) / (2 * h)
                mag = np.linalg.norm(fd, axis=1); mv = mag > 0.02
                fd_err.append(float((np.linalg.norm(fd - v_ch, axis=1)[mv] / mag[mv]).max()))
                prev_pos = (pos, rv)
            elif prev_pos is not None:
                v1, _ = A.velocities(pos, q)
                fdp = (pos - prev_pos[0]) / dt
                mag = np.linalg.norm(fdp, axis=1)
                step_err.append(float((np.linalg.norm(fdp - v1, axis=1) / mag).max()))
                rv1 = np.asarray(A.art.root_velocity(), float)
                root_err.append(float(np.linalg.norm(fdp[0] - rv1[:3]) / mag[0]))
                prev_pos = None
        world.step(dt)
        t_now = (s + 1) * dt
        if t_now <= 20.0 + 1e-9:
            c = A.com(); vmax_y = max(vmax_y, abs(c[1] - y0)); tilt_max = max(tilt_max, roll_pitch(A))
        if abs(t_now - 0.5) < 0.5 * dt:
            speeds_B[0.5] = float(np.linalg.norm(B.velocities()[0].mean(0)))
            speeds_C[0.5] = float(np.linalg.norm(C.velocities()[0].mean(0)))
        if abs(t_now - 10.5) < 0.5 * dt:
            speeds_B[10.5] = float(np.linalg.norm(B.velocities()[0].mean(0)))
            speeds_C[10.5] = float(np.linalg.norm(C.velocities()[0].mean(0)))
        if abs(t_now - 20.0) < 0.5 * dt:
            comA20 = A.com(); headA = A.head_pose()[0]
        if t_now <= 3.0 and (s + 1) % 4 == 0:
            E_rec.append((t_now, E.velocities()[0].mean(0)[2]))
        if abs(t_now - 20.0) < 0.5 * dt:
            vF = Fs.velocities()[0].mean(0)
        if 15.0 <= t_now <= 20.0 and (s + 1) % 24 == 0:
            yawG.append((t_now, Gs.mean_heading()))
    # (a)
    report("a chain-velocity", len(fd_err) > 0 and max(fd_err) < 1e-3 and max(fk_err) < 1e-4,
           f"max rel err {max(fd_err):.2e} vs central FD of the link kinematics ({len(fd_err)} samples, links > 2 cm/s); "
           f"FK vs PhysX poses {max(fk_err):.1e} m; PhysX one-substep FD vs chain {max(step_err):.2e} "
           f"(TGS integrator floor: root link alone {max(root_err):.2e})")
    # (b)
    kb = speeds_B[10.5] / speeds_B[0.5]; kc = speeds_C[10.5] / speeds_C[0.5]
    report("b coast-down", kb >= 0.99,
           f"compensated keeps {kb * 100:.2f} % after 10 s (uncompensated {kc * 100:.2f} %, exp(-0.5) = {math.exp(-0.5) * 100:.1f} %)")
    # (c)
    disp = comA20 - com0
    fwd = float(disp @ _yaw_dir(0.0))
    sp = fwd / 20.0
    bl = sp / (A.n * A.p.link_length)
    head_first = fwd > 0 and float((headA - comA20) @ disp) > 0
    report("c head-first", head_first and vmax_y < 0.02 and tilt_max < 5.0,
           f"lateral a30 w120 d30: {sp:.4f} m/s = {bl:.4f} bl/s over 0-20 s (head leads: {head_first}); "
           f"vertical drift {vmax_y * 100:.3f} cm; max tilt {tilt_max:.3f} deg; peak torque {A.peak_torque:.2f} N m")
    # (d)
    posD, _ = Dn.link_poses()
    finite = bool(np.isfinite(posD).all())
    spread = float(np.linalg.norm(posD - posD.mean(0), axis=1).max())
    report("d stress 60 s", finite and spread < Dn.n * Dn.p.link_length and Dn.peak_torque <= Dn.p.torque_cap + 5,
           f"a40 w150: finite {finite}, body radius {spread:.3f} m, drift {np.linalg.norm(Dn.com() - np.array([0, 0, D])):.2f} m, "
           f"peak torque {Dn.peak_torque:.2f} N m, max tilt {roll_pitch(Dn):.2f} deg")
    # (e) added mass: a straight chain pushed laterally, no drag: a = F / (m + mu_n), no ringing.
    tE = np.array([r[0] for r in E_rec]); vE = np.array([r[1] for r in E_rec])
    sel = (tE > 0.5) & (tE < 3.0)
    slope = float(np.polyfit(tE[sel], vE[sel], 1)[0])
    pred = Fpush / (E.p.mass + E.mu_n)
    resid = float(np.std(vE[sel] - np.polyval(np.polyfit(tE[sel], vE[sel], 1), tE[sel])))
    report("e added mass", abs(slope / pred - 1) < 0.01 and resid < 1e-4,
           f"lateral accel {slope:.5f} m/s^2 vs F/(m+mu_n) {pred:.5f} ({(slope / pred - 1) * 100:+.3f} %), residual {resid:.1e} m/s; "
           f"root bound |z| <= mu_n/m = {E.mu_n / E.p.mass:.3f}")
    # (f) thrust: straight body, terminal speed where 2 N = the tangential drag sum n c_t (v + v^2)
    n_, ct = Fs.n, Fs.c_t
    v_pred = (-1.0 + math.sqrt(1.0 + 4.0 * 2.0 / (n_ * ct))) / 2.0
    vfwd = float(vF @ _yaw_dir(0.0))
    report("f thrust", abs(vfwd / v_pred - 1) < 0.02 and abs(vF[2]) < 0.01,
           f"2 x 1 N side thrust: {vfwd:.4f} m/s vs drag balance {v_pred:.4f} ({(vfwd / v_pred - 1) * 100:+.2f} %), "
           f"sideways {vF[2]:+.4f} m/s; thruster power {Fs.power_thr:.2f} W")
    # (g) the tunnel couple: steady yaw rate where M = sum c_n (x^2 r + |x|^3 r|r|) + n lam2 r
    tg = np.array([r[0] for r in yawG]); yg = np.array([r[1] for r in yawG])
    r_meas = float(np.polyfit(tg, yg, 1)[0])
    xk = (np.arange(Gs.n) - (Gs.n - 1) / 2.0) * Gs.p.link_length
    a2, a1 = Gs.c_n * np.sum(np.abs(xk) ** 3), Gs.c_n * np.sum(xk ** 2) + Gs.n * Gs.lam2
    r_pred = (-a1 + math.sqrt(a1 * a1 + 4.0 * a2 * M_G)) / (2.0 * a2)
    drift = float(np.linalg.norm(Gs.com() - np.array([0.0, 0.0, 2 * D])))
    report("g tunnel couple", abs(r_meas / r_pred - 1) < 0.03 and drift < 0.05,
           f"{M_G} N m: yaw rate {math.degrees(r_meas):+.3f} deg/s vs rotational drag {math.degrees(r_pred):+.3f} "
           f"({(r_meas / r_pred - 1) * 100:+.2f} %, + = toward +heading), centre drift {drift * 100:.2f} cm")
    for S in (A, B, C, Dn, E, Fs, Gs):
        S.remove()
    del A, B, C, Dn, E, Fs, Gs, world
    gc.collect()
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(0 if selftest() else 1)
    ap.print_help()
