"""What does inverse kinematics actually solve?  A ~105 s explainer on a Franka FR3.

    python ik_fr3.py                               # the film (1080p60), lesson_out/ik_fr3.mp4
    python ik_fr3.py --stills 5,15,40              # individual frames, into lesson_out/
    python ik_fr3.py --sheet                       # contact sheet, one frame per 3 s
    python ik_fr3.py --preview --out preview.mp4   # 960x540 @ 30 fps

Everything the robot does is computed by threepp's own IK solver
(`tp.IkSolver`, damped least squares), and the equation shown on screen is the
update that solver applies. The Jacobian arrows are measured from the robot's
forward kinematics by finite differences, not drawn by hand.

Built on `lesson.py`: pass 1 (choreography) runs the solver front to back and
records every frame's joint vector; pass 2 renders any frame from that record.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from lesson import (Arrow3D, Film, Hud, Keys, Marker3D, Ring3D, shrink, write_png, Stage, Timeline, Tube3D, clamp01,  # noqa: E402
                    data_file, ease_out, ease_out_back, envelope, remap, smooth, smoother, standard, tp, xray)

FPS = 60
DOF = 7
JSCALE = 0.5
FINGERS = 0.035

# joint colours, base -> wrist
JC = [0x5b8cff, 0x36c5f0, 0x2ec4b6, 0x8bd346, 0xf4d35e, 0xf78c6b, 0xef476f]
C_TARGET = 0xffb347
C_TOOL = 0x4cc9f0
C_ERR = 0xff5d73
C_OK = 0x5ee27a
C_TEXT = 0xf2f5fa
C_DIM = 0x9fb2cc

Q_READY = [0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4]

# ── the script ────────────────────────────────────────────────────────────────
TL = Timeline()
TL.add("open", 0.0, 7.0)
TL.add("fk", 7.0, 23.0)
TL.add("ask", 23.0, 33.0)
TL.add("jac", 33.0, 48.0)
TL.add("iter", 48.0, 64.0)
TL.add("null", 64.0, 77.0)
TL.add("track", 77.0, 91.0)
TL.add("reach", 91.0, 100.0)
TL.add("outro", 100.0, 108.0)

# (start, end, text)
CAPTIONS = [
    (7.4, 13.2, "Forward kinematics: set the seven joint angles, and the hand's pose follows."),
    (13.6, 22.6, "Chain one transform per joint, from the base out to the hand. One set of angles gives exactly one pose."),
    (23.6, 28.4, "Inverse kinematics asks the reverse question: the hand must be here. Which angles put it there?"),
    (28.8, 32.8, "For seven joints there is no neat formula to invert. So we search, one small step at a time."),
    (33.6, 40.2, "Start with a local question: if one joint turns a little, which way does the hand move?"),
    (40.6, 47.6, "Stack those directions, one column per joint, and you have the Jacobian J. It turns small joint changes into hand motion."),
    (48.4, 55.2, "Solve for the joint step that best closes the gap e, take it, then look again."),
    (55.6, 63.6, "No single step is perfect, but the error shrinks fast."),
    (64.4, 70.4, "Seven joints, six numbers to match: one degree of freedom is left over."),
    (70.8, 76.6, "So the hand can hold still while the elbow swings. Solvers use this spare motion to avoid limits and obstacles."),
    (77.4, 83.4, "Solve once per frame, and the hand follows a moving target."),
    (83.8, 90.6, None),   # filled with the measured solve time
    (91.4, 99.6, "Ask for the impossible, and the damping keeps the arm calm: it gets as close as it can instead of thrashing."),
]


# ── kinematics helpers ────────────────────────────────────────────────────────
def full(q7):
    return list(map(float, q7[:DOF])) + [FINGERS, FINGERS]


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def tool_down(p, yaw=0.0):
    """4x4 pose: tool at p, pointing straight down, turned by yaw."""
    M = np.eye(4)
    M[:3, :3] = rot_z(yaw) @ np.diag([1.0, -1.0, -1.0])
    M[:3, 3] = p
    return M


def mat4(M):
    m = tp.Matrix4()
    m.set(*[float(v) for v in np.asarray(M, float).reshape(-1)])
    return m


def rotvec(R):
    """Rotation matrix -> rotation vector (axis * angle)."""
    c = clamp01((np.trace(R) - 1.0) / 2.0 * 0.5 + 0.5) * 2.0 - 1.0
    ang = math.acos(max(-1.0, min(1.0, c)))
    if ang < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = math.sin(ang)
    if abs(s) < 1e-6:
        return np.zeros(3)
    return v * (ang / (2.0 * s))


class Kin:
    def __init__(self, robot):
        self.robot = robot
        self.ranges = [(r.min, r.max) for r in robot.get_joint_ranges()][:DOF]

    def fk(self, q7):
        return self.robot.compute_end_effector_transform(full(q7)).to_numpy().astype(np.float64)

    def jacobian(self, q7, h=1e-4):
        """6x7: rows 0..2 = tool linear velocity per unit joint rate, 3..5 = angular."""
        M0 = self.fk(q7)
        J = np.zeros((6, DOF))
        for i in range(DOF):
            q = list(q7)
            q[i] += h
            M1 = self.fk(q)
            J[:3, i] = (M1[:3, 3] - M0[:3, 3]) / h
            J[3:, i] = rotvec(M1[:3, :3] @ M0[:3, :3].T) / h
        return J

    def pose_error(self, q7, T):
        M = self.fk(q7)
        return float(np.linalg.norm(T[:3, 3] - M[:3, 3])), float(np.linalg.norm(rotvec(T[:3, :3] @ M[:3, :3].T)))


# ── choreography ──────────────────────────────────────────────────────────────
class Record:
    """Per-frame state for the whole film, at FPS."""

    def __init__(self, n):
        self.n = n
        self.q = np.zeros((n, DOF))
        self.target = np.zeros((n, 4, 4))
        self.target[:] = np.eye(4)
        self.iter_step = np.full(n, -1, int)      # which DLS step is displayed
        self.iter_u = np.zeros(n)                 # 0..1 progress of the displayed step's motion
        self.task_pos_err = np.zeros(n)
        self.meta = {}


def choreograph(robot, verbose=True):
    kin = Kin(robot)
    n = int(round(TL.duration * FPS)) + 1
    rec = Record(n)
    T = lambda i: i / FPS  # noqa: E731
    idx = lambda t: int(round(t * FPS))  # noqa: E731

    # ---- open: a graceful move through a few poses (IK-driven, so it is honest)
    full_opts = tp.IkOptions()
    full_opts.task = tp.IkTask.Pose
    full_opts.max_iterations = 200
    full_opts.position_tolerance = 1e-4
    full_opts.orientation_tolerance = 1e-3
    full_opts.rest_pose = full(Q_READY)
    full_opts.rest_pose_gain = 0.05
    solver = tp.IkSolver(robot, full_opts)

    def solve_to(q7, Tm, opts=None, iters=1):
        s = solver if opts is None else tp.IkSolver(robot, opts)
        q = full(q7)
        res = None
        for _ in range(iters):
            q, res = s.solve(q, mat4(Tm))
        return list(q[:DOF]), res

    A_open = [tool_down([0.50, -0.12, 0.22], 0.5), tool_down([0.38, 0.10, 0.62], -0.4),
              tool_down([0.31, 0.0, 0.487], 0.0)]
    q_open = [solve_to(Q_READY, M, iters=3)[0] for M in A_open]
    keys_open = Keys([(0.0, Q_READY), (2.2, q_open[0]), (4.4, q_open[1]), (7.0, Q_READY)])
    for i in range(0, idx(TL.end("open"))):
        rec.q[i] = keys_open(T(i))

    # ---- fk: wiggle each joint in turn, starting at 13.8 s
    wig0, wig_dur = 13.8, 1.2
    amps = [0.55, 0.45, 0.55, 0.45, 0.7, 0.55, 0.9]
    for i in range(idx(TL.start("fk")), idx(TL.end("fk"))):
        t = T(i)
        q = np.array(Q_READY, float)
        for j in range(DOF):
            u = (t - (wig0 + j * wig_dur)) / wig_dur
            if 0.0 <= u <= 1.0:
                q[j] += amps[j] * math.sin(math.pi * 2 * smooth(u)) * math.sin(math.pi * u)
        rec.q[i] = q
    rec.meta["wig0"], rec.meta["wig_dur"] = wig0, wig_dur

    # ---- ask / jac: arm rests at READY, target appears
    T_A = tool_down([0.56, -0.10, 0.24], 0.55)
    for i in range(idx(TL.start("ask")), idx(TL.end("jac"))):
        rec.q[i] = Q_READY
        rec.target[i] = T_A
    rec.meta["T_A"] = T_A
    rec.meta["J_ready"] = kin.jacobian(Q_READY)

    # ---- iter: single DLS steps with threepp's solver
    step_opts = tp.IkOptions()
    step_opts.task = tp.IkTask.Pose
    step_opts.max_iterations = 1
    step_opts.max_position_step = 0.08
    step_opts.position_tolerance = 1e-4
    step_opts.orientation_tolerance = 1e-3
    step_opts.rest_pose = full(Q_READY)
    step_opts.rest_pose_gain = 0.05
    stepper = tp.IkSolver(robot, step_opts)
    qs = [list(Q_READY)]
    errs = [kin.pose_error(Q_READY, T_A)[0]]
    q = full(Q_READY)
    for k in range(80):
        q, res = stepper.solve(q, mat4(T_A))
        qs.append(list(q[:DOF]))
        pe, oe = kin.pose_error(q, T_A)
        errs.append(pe)
        if pe < 1e-4 and oe < 1e-3:
            break
    nsteps = len(qs) - 1
    rec.meta["iter_err"] = errs
    rec.meta["iter_q"] = qs
    if verbose:
        print(f"[iter] {nsteps} DLS steps, error {errs[0] * 100:.1f} cm -> {errs[-1] * 1000:.3f} mm")

    # step schedule: first steps slow, then accelerate
    t0 = TL.start("iter") + 0.8
    t_end = TL.end("iter") - 1.6
    durs = []
    for k in range(nsteps):
        durs.append(2.0 if k < 3 else 0.9 if k < 6 else 0.34)
    scale = min(1.0, (t_end - t0 - 0.0) / sum(durs))
    durs = [d * scale for d in durs]
    sched = []
    tt = t0
    for d in durs:
        sched.append((tt, tt + d))
        tt += d
    rec.meta["iter_sched"] = sched
    for i in range(idx(TL.start("iter")), idx(TL.end("iter"))):
        t = T(i)
        rec.target[i] = T_A
        k = -1
        for kk, (a, b) in enumerate(sched):
            if t >= a:
                k = kk
        if k < 0:
            rec.q[i] = qs[0]
            rec.iter_step[i] = -1
            continue
        a, b = sched[k]
        move = 0.62 * (b - a)
        u = smoother((t - a) / move)
        rec.q[i] = np.array(qs[k]) * (1 - u) + np.array(qs[k + 1]) * u
        rec.iter_step[i] = k
        rec.iter_u[i] = u
    q_sol = qs[-1]

    # ---- null: self-motion. Hand locked on T_A, elbow swings along the null space.
    lock_opts = tp.IkOptions()
    lock_opts.task = tp.IkTask.Pose
    lock_opts.max_iterations = 20
    lock_opts.position_tolerance = 2e-5
    lock_opts.orientation_tolerance = 2e-4
    locker = tp.IkSolver(robot, lock_opts)   # no rest pose: nothing pulls the elbow back
    q = np.array(q_sol, float)
    i0, i1 = idx(TL.start("null")), idx(TL.end("null"))
    ns_t0, ns_t1 = TL.start("null") + 1.0, TL.end("null") - 0.8
    max_null_err = 0.0
    AMP = 1.15                     # peak excursion along the (unit) null direction, rad
    s_prev = 0.0
    for i in range(i0, i1):
        t = T(i)
        u = remap(t, ns_t0, ns_t1)
        # out one way, back through, out the other way, home: zero speed at both ends
        s_now = AMP * math.sin(2 * math.pi * smoother(u))
        ds = s_now - s_prev
        s_prev = s_now
        J = kin.jacobian(q)
        W = np.diag([1, 1, 1, 0.3, 0.3, 0.3])
        _, _, Vt = np.linalg.svd(W @ J)
        nvec = Vt[-1]
        if "_nprev" in rec.meta and np.dot(nvec, rec.meta["_nprev"]) < 0:
            nvec = -nvec
        elif "_nprev" not in rec.meta and nvec[3] > 0:
            nvec = -nvec          # start by lifting the elbow (joint 4 folds further)
        rec.meta["_nprev"] = nvec
        q = q + nvec * ds
        for j in range(DOF):
            lo, hi = kin.ranges[j]
            q[j] = min(max(q[j], lo + 0.02), hi - 0.02)
        qq, res = locker.solve(full(q), mat4(T_A))
        q = np.array(qq[:DOF])
        rec.q[i] = q
        rec.target[i] = T_A
        max_null_err = max(max_null_err, kin.pose_error(q, T_A)[0])
    if verbose:
        print(f"[null] worst hand drift during self-motion {max_null_err * 1000:.3f} mm, "
              f"end q3 {q[2]:+.2f}")
    rec.meta["null_drift"] = max_null_err

    def t_of_u(target):
        # invert smoother(): the swing peaks where smoother(u) = 0.25 and 0.75
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if smoother(mid) < target:
                lo = mid
            else:
                hi = mid
        return ns_t0 + lo * (ns_t1 - ns_t0)
    rec.meta["null_peaks"] = [t_of_u(0.25), t_of_u(0.75)]

    # ---- track: target glides to the figure-eight, then runs it; one full solve per frame
    track_opts = tp.IkOptions()
    track_opts.task = tp.IkTask.Pose
    track_opts.max_iterations = 100
    track_opts.position_tolerance = 1e-4
    track_opts.orientation_tolerance = 1e-3
    track_opts.rest_pose = full(Q_READY)
    track_opts.rest_pose_gain = 0.05
    tracker = tp.IkSolver(robot, track_opts)
    c = np.array([0.46, -0.04, 0.38])

    def eight(s):
        # a figure-eight lying in the picture plane (x right, z up), a little depth in y
        return c + np.array([0.20 * math.sin(2 * math.pi * s), -0.05 * math.sin(4 * math.pi * s),
                             0.085 * math.sin(4 * math.pi * s)])

    tr_glide0, tr_run0, tr_run1 = TL.start("track"), TL.start("track") + 1.6, TL.end("track") - 0.2
    p_A = T_A[:3, 3]
    solve_ms = []
    track_err = []
    qf = full(q)
    for i in range(idx(TL.start("track")), idx(TL.end("track"))):
        t = T(i)
        if t < tr_run0:
            u = smoother(remap(t, tr_glide0, tr_run0))
            p = p_A * (1 - u) + eight(0.0) * u
            yaw = 0.55 * (1 - u)
        else:
            s = (t - tr_run0) / (tr_run1 - tr_run0)   # one loop at constant pace
            p = eight(s)
            yaw = 0.0
        Tm = tool_down(p, yaw)
        t_a = time.perf_counter()
        qf, res = tracker.solve(qf, mat4(Tm))
        solve_ms.append((time.perf_counter() - t_a) * 1000.0)
        rec.q[i] = qf[:DOF]
        rec.target[i] = Tm
        track_err.append(res.position_error)
    solve_ms = np.array(solve_ms)
    rec.meta["solve_ms_median"] = float(np.median(solve_ms))
    rec.meta["track_err_max"] = float(np.max(track_err))
    rec.meta["track_run"] = (tr_run0, tr_run1)
    rec.meta["eight"] = eight
    if verbose:
        print(f"[track] solve median {np.median(solve_ms):.3f} ms, p95 {np.percentile(solve_ms, 95):.3f} ms, "
              f"worst position error {max(track_err) * 1000:.2f} mm")

    # ---- reach: target leaves the workspace; position-only task, same damping
    reach_opts = tp.IkOptions()
    reach_opts.task = tp.IkTask.Position
    reach_opts.max_iterations = 60
    reach_opts.position_tolerance = 1e-4
    reacher = tp.IkSolver(robot, reach_opts)
    p_start = eight(1.0)
    p_far = np.array([1.06, -0.16, 0.46])
    gaps = []
    for i in range(idx(TL.start("reach")), idx(TL.end("reach"))):
        t = T(i)
        u = smoother(remap(t, TL.start("reach") + 0.3, TL.start("reach") + 5.2))
        p = p_start * (1 - u) + p_far * u
        Tm = tool_down(p, 0.0)
        qf, res = reacher.solve(qf, tp.Vector3(*[float(v) for v in p]))
        rec.q[i] = qf[:DOF]
        rec.target[i] = Tm
        rec.task_pos_err[i] = res.position_error
        gaps.append(res.position_error)
    rec.meta["reach_gap"] = float(gaps[-1])

    # ---- outro: the arm folds back home while the summary comes up
    q_end = np.array(qf[:DOF])
    for i in range(idx(TL.start("outro")), n):
        u = smoother(remap(T(i), TL.start("outro"), TL.start("outro") + 3.0))
        rec.q[i] = q_end * (1 - u) + np.array(Q_READY) * u
        rec.target[i] = rec.target[idx(TL.end("reach")) - 1]
    if verbose:
        print(f"[reach] final gap {gaps[-1] * 100:.1f} cm")
    return rec, kin


# ── the picture ───────────────────────────────────────────────────────────────
class Film3D:
    def __init__(self, st: Stage, robot):
        self.st = st
        self.robot = robot
        self.links = [robot.get_object_by_name(f"fr3_link{i}") for i in range(1, 8)]
        self.tcp = robot.get_object_by_name("fr3_hand_tcp")
        root = st.zup

        radii = [0.105, 0.105, 0.098, 0.098, 0.094, 0.086, 0.078]
        self.rings = [Ring3D(JC[i], radius=radii[i], tube=0.0065, parent=self.links[i]) for i in range(DOF)]
        self.axes = []
        for i in range(DOF):
            a = Arrow3D(JC[i], radius=0.0042, parent=self.links[i], emissive=1.0)
            a.set((0, 0, -0.16), (0, 0, 0.16), opacity=0.0)
            self.axes.append(a)
        # tool frame triad
        self.tool_triad = []
        for c, d in ((0xff5a5f, (1, 0, 0)), (0x5ee27a, (0, 1, 0)), (0x4f8dff, (0, 0, 1))):
            a = Arrow3D(c, radius=0.0045, parent=self.tcp, emissive=0.8)
            a.set((0, 0, 0), np.array(d, float) * 0.09, opacity=0.0)
            self.tool_triad.append(a)
        # kinematic chain skeleton
        self.skel = Tube3D(0xdfe8ff, radius=0.0036, parent=root, emissive=0.7, xray=True)
        self.joint_dots = []
        for i in range(DOF):
            m = tp.Mesh(tp.SphereGeometry(0.014, 24, 16), standard(JC[i], 0.3, emissive=JC[i], emissive_intensity=1.2))
            m.visible = False
            root.add(m)
            xray(m, order=11)
            self.joint_dots.append(m)
        self.target = Marker3D(C_TARGET, parent=root)
        self.jcols = [Arrow3D(JC[i], radius=0.0055, parent=root, emissive=0.9) for i in range(DOF)]
        for a in self.jcols:
            a.set_opacity(0.0)
        self.err = Arrow3D(C_ERR, radius=0.0062, parent=root, emissive=1.1)
        self.err.set_opacity(0.0)
        self.tool_trail = Tube3D(C_TOOL, radius=0.0032, parent=root, emissive=1.0)
        self.target_trail = Tube3D(C_TARGET, radius=0.0026, parent=root, emissive=1.1)
        self.elbow_trail = Tube3D(JC[3], radius=0.0045, parent=root, emissive=1.0, xray=True)
        self.ghosts = [make_ghost(st, 0x8fb4ff) for _ in range(2)]
        self.step_dots = []
        for k in range(12):
            m = tp.Mesh(tp.SphereGeometry(0.0085, 20, 12), standard(C_TOOL, 0.3, emissive=C_TOOL, emissive_intensity=1.4))
            m.visible = False
            root.add(m)
            self.step_dots.append(m)

        self._zinv = None

    def pose(self, q7):
        self.robot.set_joint_values(full(q7))
        self.st.scene.update_matrix_world()

    def zinv(self):
        if self._zinv is None:
            self.st.zup.update_matrix_world()
            self._zinv = np.linalg.inv(self.st.zup.matrix_world.to_numpy())
        return self._zinv

    def link_origins(self):
        """Joint origins (7) + tool, in the robot frame, for the CURRENT pose."""
        Zi = self.zinv()
        pts = []
        for o in self.links + [self.tcp]:
            M = Zi @ o.matrix_world.to_numpy()
            pts.append(M[:3, 3].copy())
        return np.array(pts)


def make_ghost(st, color):
    """A second FR3 with one translucent material: a pose remembered on screen."""
    g = tp.URDFLoader().load(data_file("urdf", "franka", "fr3.urdf"))
    g.show_colliders(False)
    mat = standard(color, roughness=0.4, emissive=color, emissive_intensity=0.35)
    mat.transparent = True
    mat.depth_write = False
    mat.opacity = 0.0

    def f(o):
        if type(o).__name__ == "Mesh" and o.visible:
            o.set_material(mat)
            o.cast_shadow = False
    g.traverse(f)
    g.visible = False
    st.zup.add(g)
    return g, mat


def link_positions_all(robot, st, rec, which):
    """Robot-frame origin of link `which` (1..7) or 'tcp' for every recorded frame."""
    obj = robot.get_object_by_name("fr3_hand_tcp" if which == "tcp" else f"fr3_link{which}")
    st.zup.update_matrix_world()
    Zi = np.linalg.inv(st.zup.matrix_world.to_numpy())
    out = np.zeros((rec.n, 3))
    for i in range(rec.n):
        robot.set_joint_values(full(rec.q[i]))
        robot.update_matrix_world()
        out[i] = (Zi @ obj.matrix_world.to_numpy())[:3, 3]
    return out


# camera: (time, azimuth deg, elevation deg, distance m, look point in the ROBOT frame, fov)
# azimuth is measured in the robot's XY plane from +x; at -90 the camera sits on -y and
# the robot's reach (+x) runs left to right across the picture.
CAM = [
    (0.0, -28, 13, 3.05, (-0.34, 0.05, 0.40), 30),
    (6.2, -46, 15, 2.85, (-0.26, 0.02, 0.42), 30),
    (8.0, -58, 16, 2.60, (0.16, 0.0, 0.42), 30),
    (22.5, -64, 17, 2.50, (0.18, 0.0, 0.42), 30),
    (25.0, -62, 16, 2.45, (0.26, 0.0, 0.38), 30),
    (33.0, -60, 17, 2.30, (0.30, -0.02, 0.38), 30),
    (47.5, -66, 17, 2.25, (0.32, -0.02, 0.38), 30),
    (63.5, -70, 16, 2.25, (0.32, -0.02, 0.38), 30),
    (66.5, -48, 34, 2.30, (0.28, 0.0, 0.40), 30),
    (76.5, -54, 32, 2.30, (0.28, 0.0, 0.40), 30),
    (79.0, -72, 14, 2.40, (0.34, -0.04, 0.40), 30),
    (90.5, -78, 14, 2.45, (0.34, -0.04, 0.40), 30),
    (93.5, -80, 12, 2.95, (0.52, -0.06, 0.38), 30),
    (99.5, -84, 12, 3.00, (0.52, -0.06, 0.38), 30),
    (108.0, -60, 16, 3.30, (0.30, 0.0, 0.40), 30),
]


def camera_at(t):
    az, el, dist, fov = (float(v) for v in Keys([(k[0], [k[1], k[2], k[3], k[5]]) for k in CAM])(t))
    look = Keys([(k[0], k[4]) for k in CAM])(t)
    az += 1.2 * math.sin(0.21 * t)          # slow breathing drift: held shots never freeze
    el += 0.6 * math.sin(0.17 * t + 1.0)
    a, e = math.radians(az), math.radians(el)
    eye_r = look + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
    zw = lambda p: np.array([p[0], p[2], -p[1]])  # noqa: E731  robot Z-up -> world Y-up
    return zw(eye_r), zw(look), fov


class Painter:
    """Everything drawn per frame."""

    def __init__(self, st, film3d, rec, kin, ov: Hud):
        self.st, self.f3, self.rec, self.kin, self.ov = st, film3d, rec, kin, ov
        self.tcp_path = link_positions_all(film3d.robot, st, rec, "tcp")
        self.elbow_path = link_positions_all(film3d.robot, st, rec, 4)
        self.ranges = kin.ranges
        ms = rec.meta["solve_ms_median"]
        caps = []
        for a, b, txt in CAPTIONS:
            if txt is None:
                txt = (f"Each solve takes about {ms:.2f} ms on this PC. A 60 fps frame lasts 16.7 ms."
                       if ms < 1.0 else f"Each solve takes about {ms:.1f} ms here, comfortably inside one frame.")
            caps.append((a, b, txt))
        self.captions = caps

    # 3D --------------------------------------------------------------------
    def set3d(self, t, i):
        rec, f3 = self.rec, self.f3
        f3.pose(rec.q[i])
        eye, look, fov = camera_at(t)
        self.st.look(eye, look, fov)

        # rings + joint axes: appear one by one in fk, then pulse on wiggle / Jacobian build
        fk_in = [envelope(t, 8.0 + 0.32 * j, TL.end("fk") - 0.3, 0.5, 0.6) for j in range(DOF)]
        wig0, wd = rec.meta["wig0"], rec.meta["wig_dur"]
        for j in range(DOF):
            wig = envelope(t, wig0 + j * wd - 0.15, wig0 + (j + 1) * wd + 0.1, 0.25, 0.3)
            jb = self.jac_build(t, j)
            o = max(fk_in[j] * (0.55 + 0.45 * wig), 0.9 * jb)
            f3.rings[j].set_opacity(o)
            f3.rings[j].mat.emissive_intensity = 0.8 + 1.6 * max(wig, jb)
            f3.axes[j].set_opacity(max(fk_in[j] * wig, 0.9 * jb) * 0.95)

        # chain skeleton
        sk = envelope(t, 9.6, TL.end("fk") - 0.4, 0.8, 0.8)
        pts = f3.link_origins()
        if sk > 0.003:
            f3.skel.set_points(pts, opacity=0.85 * sk)
            for j, m in enumerate(f3.joint_dots):
                m.visible = True
                m.material.opacity = sk
                m.material.transparent = sk < 0.999
                m.position.set(*pts[j])
        else:
            f3.skel.mesh.visible = False
            for m in f3.joint_dots:
                m.visible = False

        # tool triad
        tt = max(envelope(t, 8.4, TL.end("jac") - 1.0, 0.6, 0.8), envelope(t, TL.start("null") + 0.2, TL.end("null") - 0.3, 0.6, 0.6))
        for a in f3.tool_triad:
            a.set_opacity(tt)

        # target
        tv = max(envelope(t, TL.start("ask") + 0.8, TL.end("reach") + 0.8, 0.6, 0.9), 0.0)
        pop = ease_out_back(remap(t, TL.start("ask") + 0.8, TL.start("ask") + 1.6)) if t < TL.start("ask") + 2 else 1.0
        f3.target.place(rec.target[i])
        f3.target.group.scale.set(max(pop, 1e-3), max(pop, 1e-3), max(pop, 1e-3))
        f3.target.set_opacity(tv, pulse=0.5 + 0.5 * math.sin(4.0 * t))
        # the target's triad matters while orientation is part of the task
        tri = 1.0 - smooth(remap(t, TL.start("reach"), TL.start("reach") + 0.8))
        for a in f3.target.triad:
            a.set_opacity(tv * tri)

        # Jacobian columns at the hand
        tool = self.tcp_path[i]
        J = rec.meta["J_ready"]
        colv = envelope(t, TL.start("jac"), TL.end("jac") + 0.2, 0.3, 0.8)
        for j in range(DOF):
            b = self.jac_build(t, j, persist=True) * colv
            v = J[:3, j] * JSCALE        # hand motion for JSCALE rad at joint j
            grow = ease_out(self.jac_grow(t, j))
            f3.jcols[j].set(tool, tool + v * max(grow, 1e-3), opacity=b)

        # error vector e: jac (end) + iter
        e_op = envelope(t, 44.0, TL.end("iter") - 0.8, 0.6, 0.8)
        tgt = rec.target[i][:3, 3]
        if e_op > 0.003 and np.linalg.norm(tgt - tool) > 0.004:
            f3.err.set_color(C_ERR)
            f3.err.set(tool, tgt - (tgt - tool) / max(np.linalg.norm(tgt - tool), 1e-9) * 0.028, opacity=e_op)
        else:
            f3.err.set_opacity(0.0)
        # reach: the unclosable gap in red
        if TL.active("reach", t):
            g = envelope(t, TL.start("reach") + 3.0, TL.end("reach") + 0.8, 0.6, 0.8)
            if np.linalg.norm(tgt - tool) > 0.03:
                f3.err.set(tool, tgt - (tgt - tool) / np.linalg.norm(tgt - tool) * 0.03, opacity=g)

        # iteration path: dots at each step's hand position + faint path
        k = rec.iter_step[i]
        iv = envelope(t, TL.start("iter"), TL.end("iter") + 0.4, 0.3, 0.8)
        if iv > 0.003 and k >= 0:
            qs = rec.meta["iter_q"]
            hp = [self.kin.fk(qs[m])[:3, 3] for m in range(min(k + 1, len(qs)))]
            hp.append(tool)
            f3.tool_trail.set_points(np.array(hp), opacity=0.7 * iv)
            for m, dot in enumerate(f3.step_dots):
                if m < min(k + 1, len(hp) - 1):
                    dot.visible = True
                    dot.position.set(*hp[m])
                    dot.material.opacity = iv
                    dot.material.transparent = iv < 0.999
                else:
                    dot.visible = False
        else:
            for dot in f3.step_dots:
                dot.visible = False
            f3.tool_trail.mesh.visible = False

        # null-space: elbow trail
        nv = envelope(t, TL.start("null") + 0.8, TL.end("null") + 0.2, 0.4, 0.8)
        if nv > 0.003:
            a = int(round((TL.start("null") + 0.8) * FPS))
            pts_e = self.elbow_path[a:i + 1]
            if len(pts_e) > 2:
                f3.elbow_trail.set_points(pts_e[::2] if len(pts_e) > 4 else pts_e, opacity=nv)
            else:
                f3.elbow_trail.mesh.visible = False
        else:
            f3.elbow_trail.mesh.visible = False

        # ghost arms at the two extremes of the swing
        for (g, gm), t_pk in zip(f3.ghosts, rec.meta["null_peaks"]):
            go = envelope(t, t_pk, TL.end("null") - 0.2, 0.5, 0.7) * 0.2
            if go > 0.003:
                g.visible = True
                gm.opacity = go
                g.set_joint_values(full(rec.q[int(round(t_pk * FPS))]))
            else:
                g.visible = False

        # tracking: target + hand trails (last ~2.5 s)
        trv = envelope(t, TL.start("track") + 1.6, TL.end("track") + 0.6, 0.4, 0.8)
        if trv > 0.003:
            a = max(int(round((TL.start("track") + 1.6) * FPS)), i - int(2.6 * FPS))
            tp_pts = self.rec.target[a:i + 1, :3, 3]
            if len(tp_pts) > 2:
                f3.target_trail.set_points(tp_pts[::2], opacity=0.9 * trv)
            hp = self.tcp_path[a:i + 1]
            if len(hp) > 2 and not (iv > 0.003 and k >= 0):
                f3.tool_trail.set_points(hp[::2], opacity=0.9 * trv)
        else:
            f3.target_trail.mesh.visible = False
            if not (iv > 0.003 and k >= 0):
                f3.tool_trail.mesh.visible = False
        return tool, tgt

    def jac_grow(self, t, j):
        s = 34.1 + 0.8 * j
        return remap(t, s, s + 0.55)

    def jac_build(self, t, j, persist=False):
        """Highlight envelope for column j during the Jacobian build-up."""
        s = 34.1 + 0.8 * j
        if persist:
            return envelope(t, s, TL.end("jac") + 0.2, 0.3, 0.8)
        return envelope(t, s - 0.1, s + 0.9, 0.2, 0.4) if t < 40.2 else 0.0

    # 2D ----------------------------------------------------------------------
    def draw2d(self, t, i, tool, tgt):
        ov, st, rec = self.ov, self.st, self.rec

        # title
        a = envelope(t, 0.9, 6.4, 0.9, 0.8)
        if a > 0:
            y = 360 + 14 * (1 - ease_out(remap(t, 0.9, 2.2)))
            ov.text(128, y - 58, "A THREEPP LESSON", size=22, color=C_TARGET, alpha=a, kind="semibold", tracking=4)
            ov.text(122, y, "Inverse Kinematics", size=96, color=C_TEXT, alpha=a, kind="semibold")
            ov.text(128, y + 118, "How a robot arm finds its joint angles", size=38, color=C_DIM,
                    alpha=a * smooth(remap(t, 1.6, 2.6)), kind="light")

        # captions
        for (a0, b0, txt) in self.captions:
            al = envelope(t, a0, b0, 0.45, 0.4)
            if al > 0:
                ov.caption(txt, al)

        # equation card (top left)
        self.equations(t)
        # joint panel (right)
        jp = max(envelope(t, 8.2, TL.end("jac") - 0.2, 0.7, 0.6),
                 envelope(t, TL.start("null") + 0.4, TL.end("reach") + 0.4, 0.7, 0.8))
        if jp > 0:
            self.joint_panel(t, i, jp)

        # callouts
        tool_px = st.project_zup(tool)
        tgt_px = st.project_zup(tgt)
        c1 = envelope(t, 8.6, 13.4, 0.5, 0.4)
        if c1 > 0:
            ov.callout(tool_px[:2], (tool_px[0] + 150, tool_px[1] - 110), "hand pose  x", C_TOOL, c1,
                       grow=ease_out(remap(t, 8.6, 9.3)))
        c2 = envelope(t, TL.start("ask") + 1.2, TL.start("ask") + 7.6, 0.5, 0.5)
        if c2 > 0:
            ov.callout(tgt_px[:2], (tgt_px[0] + 150, tgt_px[1] + 90), "target", C_TARGET, c2,
                       grow=ease_out(remap(t, TL.start("ask") + 1.2, TL.start("ask") + 1.9)))
        # J labels at arrow tips
        J = rec.meta["J_ready"]
        colv = envelope(t, TL.start("jac"), TL.end("jac") - 0.6, 0.3, 0.6)
        placed = []
        if colv > 0:
            for j in range(DOF):
                b = self.jac_build(t, j, persist=True) * colv * smooth(self.jac_grow(t, j))
                if b > 0.01:
                    tip = st.project_zup(tool + J[:3, j] * JSCALE)[:2]
                    d = tip - tool_px[:2]
                    L = float(np.linalg.norm(d))
                    if L > 28:
                        lp = tip + d / L * 24
                        # push a label further out along its arrow until it clears the others
                        for _ in range(8):
                            if all(np.linalg.norm(lp - o) > 34 for o in placed):
                                break
                            lp = lp + d / L * 16
                        placed.append(lp)
                        ov.text(lp[0], lp[1], f"J{j + 1}", size=22, color=JC[j], alpha=b, kind="semibold",
                                anchor="mm")
        # e label
        e_op = envelope(t, 44.4, TL.end("iter") - 0.8, 0.6, 0.8)
        d = np.linalg.norm(tgt - tool)
        if e_op > 0 and d > 0.02:
            mid = st.project_zup((tool + tgt) / 2)
            ov.math(mid[0] + 18, mid[1] - 16, r"$\mathbf{e}$", size=40, color=C_ERR, alpha=e_op, anchor="lm")

        # iteration HUD
        self.iter_hud(t, i)
        # tracking HUD
        self.track_hud(t, i)
        # reach HUD
        g = envelope(t, TL.start("reach") + 3.4, TL.end("reach") + 0.6, 0.6, 0.6)
        if g > 0:
            gap = rec.task_pos_err[i]
            ov.text(tgt_px[0], tgt_px[1] + 64, f"{gap * 100:.0f} cm out of reach", size=28, color=C_ERR,
                    alpha=g, kind="semibold", anchor="mm")

        # outro
        self.outro(t)
        # fades to/from black
        return None

    def equations(self, t):
        ov = self.ov
        x, y = 70, 64
        eqs = [
            # (t_in, t_out, tex, note)
            (8.6, 22.8, r"$\mathbf{x} = f(\mathbf{q})$", "forward kinematics"),
            (14.2, 22.8, r"$T_{\mathrm{hand}} = T_1(q_1)\,T_2(q_2)\,\cdots\,T_7(q_7)$", None),
            (24.2, 32.9, r"$\mathbf{q} = f^{-1}(\mathbf{x})\ \ ?$", "inverse kinematics"),
            (41.0, 64.0, r"$\dot{\mathbf{x}} = J(\mathbf{q})\,\dot{\mathbf{q}}$", "the Jacobian"),
            (49.0, 64.0, r"$\Delta\mathbf{q} = J^{T}\left(JJ^{T} + \lambda^{2} I\right)^{-1}\mathbf{e}$",
             "damped least squares"),
            (65.4, 76.8, r"$\dot{\mathbf{q}} = \left(I - J^{+}J\right)\mathbf{z}$", "motion in the null space"),
        ]
        rows = [e for e in eqs if e[0] - 0.1 <= t <= e[1] + 0.6]
        if not rows:
            return
        sizes = [ov.math_size(tex, 40) for _, _, tex, _ in rows]
        w = max(max(s[0] for s in sizes) + 60, 470)
        h = sum(s[1] + (30 if note else 8) for s, (_, _, _, note) in zip(sizes, rows)) + 40
        pa = max(envelope(t, a, b, 0.6, 0.6) for a, b, _, _ in rows)
        ov.panel(x, y, w, h, radius=16, alpha=0.66 * pa, outline=0x8aa0c0, outline_alpha=0.16)
        yy = y + 22
        for (a, b, tex, note), (sw, sh) in zip(rows, sizes):
            al = envelope(t, a, b, 0.6, 0.6)
            if note:
                ov.text(x + 30, yy, note.upper(), size=16, color=C_DIM, alpha=al, kind="semibold", tracking=2.2)
                yy += 24
            ov.math(x + 30, yy + sh / 2, tex, size=40, color=C_TEXT, alpha=al, anchor="lm")
            yy += sh + 12

    def joint_panel(self, t, i, alpha):
        ov = self.ov
        q = self.rec.q[i]
        x, y, w = 1488, 64, 368
        rh = 44
        ov.panel(x, y, w, 64 + rh * DOF, radius=16, alpha=0.66 * alpha, outline=0x8aa0c0, outline_alpha=0.16)
        ov.text(x + 24, y + 22, "JOINT ANGLES", size=16, color=C_DIM, alpha=alpha, kind="semibold", tracking=2.2)
        ov.math(x + w - 24, y + 31, r"$\mathbf{q}$", size=30, color=C_DIM, alpha=alpha, anchor="rm")
        wig0, wd = self.rec.meta["wig0"], self.rec.meta["wig_dur"]
        for j in range(DOF):
            yy = y + 62 + j * rh
            lo, hi = self.ranges[j]
            frac = (q[j] - lo) / (hi - lo)
            hl = max(envelope(t, wig0 + j * wd - 0.15, wig0 + (j + 1) * wd + 0.1, 0.25, 0.3),
                     self.jac_build(t, j))
            lab_a = alpha * (0.75 + 0.25 * hl)
            ov.text(x + 24, yy + 12, f"q{j + 1}", size=22, color=JC[j], alpha=lab_a, kind="semibold", anchor="lm")
            bx, bw = x + 72, 196
            ov.bar(bx, yy + 7, bw, 10, 1.0, 0x2a3342, alpha=alpha)
            # zero mark
            zf = (0.0 - lo) / (hi - lo)
            if 0 < zf < 1:
                ov.line([(bx + bw * zf, yy + 3), (bx + bw * zf, yy + 21)], 0x55627a, 1.2, alpha)
            cx = bx + bw * clamp01(frac)
            ov.circle(cx, yy + 12, 8.5 + 2.5 * hl, fill=JC[j], alpha=alpha)
            ov.text(x + w - 24, yy + 12, f"{math.degrees(q[j]):+5.0f}\u00b0", size=21,
                    color=C_TEXT, alpha=alpha * (0.8 + 0.2 * hl), kind="numeric", anchor="rm")

    def iter_hud(self, t, i):
        ov, rec = self.ov, self.rec
        a = envelope(t, TL.start("iter") + 0.3, TL.end("iter") - 0.2, 0.6, 0.6)
        if a <= 0:
            return
        errs = rec.meta["iter_err"]
        k = rec.iter_step[i]
        u = rec.iter_u[i]
        shown = errs[:k + 1] if k >= 0 else errs[:1]
        if k >= 0 and u > 0.0:
            # the live point slides along the segment as the arm moves
            e_live = math.exp(math.log(max(errs[k], 1e-6)) * (1 - u) + math.log(max(errs[k + 1], 1e-6)) * u)
            shown = list(errs[:k + 1]) + [e_live]
        x, y, w, h = 1488, 440, 368, 300
        fmt = lambda v: (f"{v * 100:.0f} cm" if v >= 0.01 else f"{v * 1000:.0f} mm" if v >= 0.001 else f"{v * 1e6:.0f} \u00b5m")  # noqa: E731
        ov.logplot(x, y, w, h, shown, alpha=a, color=C_ERR, ymin=3e-5, ymax=1.0, xmax=len(errs),
                   title="HAND ERROR  |e|", threshold=1e-4, unit_fmt=fmt)
        cur = shown[-1]
        step = (k + 1) if k >= 0 else 0
        ov.text(x + 22, y + h + 36, f"step {step:2d}", size=30, color=C_TEXT, alpha=a, kind="numeric", anchor="ls")
        val = f"{cur * 100:.1f} cm" if cur >= 0.01 else f"{cur * 1000:.2f} mm"
        ov.text(x + w - 22, y + h + 36, val, size=30, color=C_ERR if cur > 1e-4 else C_OK, alpha=a,
                kind="numeric", anchor="rs")
        if k == len(errs) - 2 and u >= 0.999:
            done = smooth(remap(t, rec.meta["iter_sched"][-1][0] + 0.25, rec.meta["iter_sched"][-1][0] + 0.8))
            ov.text(x + w / 2, y + h + 78, f"converged in {len(errs) - 1} steps", size=24, color=C_OK,
                    alpha=a * done, kind="semibold", anchor="ms")

    def track_hud(self, t, i):
        ov, rec = self.ov, self.rec
        a = envelope(t, TL.start("track") + 1.0, TL.end("track") - 0.3, 0.6, 0.6)
        if a <= 0:
            return
        x, y = 1488, 440
        ov.panel(x, y, 368, 132, alpha=0.66 * a, outline=0x8aa0c0, outline_alpha=0.16)
        ov.text(x + 24, y + 22, "ONE SOLVE PER FRAME", size=16, color=C_DIM, alpha=a, kind="semibold", tracking=2.2)
        ms = rec.meta["solve_ms_median"]
        ov.text(x + 24, y + 96, f"{ms:.2f} ms", size=46, color=C_TEXT, alpha=a, kind="numeric", anchor="ls")
        ov.text(x + 344, y + 96, "median", size=20, color=C_DIM, alpha=a, anchor="rs")

    def outro(self, t):
        ov = self.ov
        a = envelope(t, TL.start("outro") + 0.2, TL.end("outro") + 1, 0.8, 0.1)
        if a <= 0:
            return
        W, H = ov.W, ov.H
        # darken the picture behind the summary
        ov.panel(-20, -20, W + 40, H + 40, radius=0, fill=0x070a10, alpha=0.8 * a)
        x = 250
        ov.text(x, 250, "IN SHORT", size=22, color=C_TARGET, alpha=a, kind="semibold", tracking=4)
        rows = [
            (r"$\mathbf{q}\ \rightarrow\ \mathbf{x}$", "Forward kinematics: one set of angles, one pose."),
            (r"$\mathbf{x}\ \rightarrow\ \mathbf{q}$", "Inverse kinematics: search for the angles, step by step with J."),
            (r"$7 - 6 = 1$", "A seven-joint arm has one spare motion to spend."),
        ]
        for k, (tex, txt) in enumerate(rows):
            al = a * smooth(remap(t, TL.start("outro") + 0.6 + 0.5 * k, TL.start("outro") + 1.3 + 0.5 * k))
            yy = 350 + k * 110
            ov.math(x, yy, tex, size=46, color=C_TEXT, alpha=al, anchor="lm")
            ov.text(x + 260, yy, txt, size=38, color=C_TEXT, alpha=al, kind="regular", anchor="lm")
        al = a * smooth(remap(t, TL.start("outro") + 2.4, TL.start("outro") + 3.2))
        ov.text(x, 780, "Rendered and solved with threepp", size=28, color=C_DIM, alpha=al, kind="semibold")
        ov.text(x, 824, "github.com/markaren/threepp", size=26, color=C_TOOL, alpha=al, kind="regular")


def fade_amount(t):
    """Fade from black at the start, to black at the end."""
    return 1.0 - min(smooth(remap(t, 0.0, 0.8)), 1.0 - smooth(remap(t, TL.duration - 0.9, TL.duration)))


# ── main ──────────────────────────────────────────────────────────────────────
def build(width, height):
    st = Stage(width, height, renderer="gl")
    robot = tp.URDFLoader().load(data_file("urdf", "franka", "fr3.urdf"))
    robot.show_colliders(False)
    robot.traverse(lambda o: setattr(o, "cast_shadow", True))
    st.zup.add(robot)
    robot.set_end_effector("fr3_hand_tcp")
    return st, robot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="the film (default: <outdir>/ik_fr3.mp4)")
    ap.add_argument("--outdir", default="lesson_out", help="stills, sheets and the default film")
    ap.add_argument("--stills", default=None, help="comma-separated times (s)")
    ap.add_argument("--sheet", action="store_true", help="contact sheet, one frame every --every seconds")
    ap.add_argument("--every", type=float, default=3.0)
    ap.add_argument("--preview", action="store_true", help="960x540 @ 30 fps")
    ap.add_argument("--from", dest="t_from", type=float, default=0.0)
    ap.add_argument("--to", dest="t_to", type=float, default=None)
    ap.add_argument("--fps", type=int, default=None)
    args = ap.parse_args()

    W, H = (960, 540) if args.preview else (1920, 1080)
    fps = args.fps or (30 if args.preview else FPS)
    st, robot = build(W, H)
    t0 = time.time()
    rec, kin = choreograph(robot)
    print(f"[choreo] {rec.n} frames in {time.time() - t0:.1f}s")
    f3 = Film3D(st, robot)
    # authored in 1920x1080 units whatever the render size; equations come from the cache
    ov = Hud(1920, 1080, math_cache=os.path.join(_HERE, "ik_fr3.math.json"))
    painter = Painter(st, f3, rec, kin, ov)

    def render(t):
        i = min(int(round(t * FPS)), rec.n - 1)
        tool, tgt = painter.set3d(t, i)     # poses the robot AND the camera for this frame
        ov.begin()
        painter.draw2d(t, i, tool, tgt)     # callouts project with that camera
        ov.fade(fade_amount(t))
        ov.end()
        return st.frame(t, hud=ov)

    os.makedirs(args.outdir, exist_ok=True)
    if args.stills:
        for tok in args.stills.split(","):
            t = float(tok)
            p = os.path.join(args.outdir, f"ik_still_{t:06.2f}.png")
            write_png(p, render(t))
            print("saved", p)
        return
    if args.sheet:
        f = max(1, W // 480)
        thumbs = [shrink(render(float(t)), f) for t in np.arange(0.5, TL.duration, args.every)]
        cols = 6
        th, tw = thumbs[0].shape[:2]
        rows = (len(thumbs) + cols - 1) // cols
        sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
        for k, im in enumerate(thumbs):
            y, x = (k // cols) * th, (k % cols) * tw
            sheet[y:y + th, x:x + tw] = im
        p = os.path.join(args.outdir, "ik_sheet.png")
        write_png(p, sheet)
        print("saved", p, f"({len(thumbs)} frames, every {args.every} s from 0.5 s, row-major)")
        return

    t_to = args.t_to if args.t_to is not None else TL.duration
    out = args.out or os.path.join(args.outdir, "ik_fr3.mp4")
    film = Film(out, W, H, fps=fps, crf=16 if not args.preview else 22,
                preset="slow" if not args.preview else "veryfast")
    nfr = int(round((t_to - args.t_from) * fps))
    t_start = time.time()
    for f in range(nfr):
        t = args.t_from + f / fps
        film.write(render(t))
        if f % (fps * 5) == 0:
            el = time.time() - t_start
            print(f"[film] {t:6.1f}s  frame {f}/{nfr}  {el / max(f, 1) * 1000:.0f} ms/frame", flush=True)
    path = film.close()
    print(f"[film] wrote {path} ({nfr} frames, {time.time() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
