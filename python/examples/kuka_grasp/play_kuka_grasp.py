"""play_kuka_grasp.py — watch a trained KUKA grasp policy reach, grip, and lift a cube.

CPU physics (no CUDA needed for interactive play), following the same pattern as spot_deploy.py:
single PhysxWorld, meshes auto-update after world.step(), obs built from CPU articulation state.
The --eval flag still uses the GPU batch env for batch statistics (needs CUDA).

    python play_kuka_grasp.py                               # interactive, latest checkpoint
    python play_kuka_grasp.py --model kuka_grasp_best.pt
    python play_kuka_grasp.py --shot out.png                # headless: step one episode, save PNG
    python play_kuka_grasp.py --eval 256                    # GPU batch eval (needs CUDA)
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PYTHON_DIR = os.path.dirname(os.path.dirname(_HERE))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

import threepp as tp

import kuka_grasp_contract as C
import kuka_grasp_robot as R
from kuka_grasp_env import FIRM_LAT, FIRM_VERT, FIRM_OPEN, LOST_MARGIN

# Verified via kuka_grasp_robot.py --inspect: link order in arm_meshes matches GPU link_pose indices.
FLANGE_LINK = 7   # link_7 (flange)
FINGER_L_LINK = 8
FINGER_R_LINK = 9


def _quat_to_R(q):
    """[qx, qy, qz, qw] → 3x3 rotation matrix."""
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),   2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),   1 - 2*(x*x+y*y)]], np.float32)


class KukaGraspCpuSim:
    """Single-env CPU PhysX sim for KUKA grasp play. No GPU, no batch.

    arm_meshes auto-update after world.step() via the rigid-body bind registered in addLink(),
    so arm_meshes[i].position/.quaternion gives the PhysX link body frame pose — identical to
    what the GPU batch reads in link_pose[:, i, :].  No manual sync needed."""

    def __init__(self, spawn_half=(0.0, 0.0)):
        self.world = tp.PhysxWorld(
            gravity=tp.Vector3(0, 0, -9.81),
            fixed_timestep=C.PHYS_DT,
            max_substeps=C.SUBSTEPS,
        )
        self.spawn_half = np.asarray(spawn_half, np.float32)
        self._rng = np.random.default_rng()

        urdf = R.generate_combined_urdf()
        self.ground, self.tables = R.add_ground_and_tables(self.world, 1, spacing=0.0)
        cube_mat = R.make_cube_material(self.world)

        self.arm, self.arm_meshes, _ = R.build_arm(
            self.world, 0, spacing=0.0, urdf_path=urdf, render_visuals=True)
        self.cube_art, self.cube_mesh = R.build_cube(
            self.world, 0, spacing=0.0, material=cube_mat, render_visuals=True)

        # perm[add_order_i] = DOF-cache slot — needed by set_joint_positions (cache order)
        self._perm = np.array(self.arm.dof_order(), dtype=np.int64)

    def reset(self):
        noise = self._rng.uniform(-0.05, 0.05, 7).astype(np.float32)
        q9_add = np.concatenate([C.DEFAULT_Q + noise, [C.GRIP_OPEN, C.GRIP_OPEN]]).astype(np.float32)
        q9_dof = np.zeros(C.N_DOF, np.float32)
        q9_dof[self._perm] = q9_add
        self.arm.set_joint_positions(q9_dof)
        self.arm.set_drive_targets(q9_add)
        for _ in range(30):
            self.world.step(C.PHYS_DT)

        rx = float(self._rng.uniform(-self.spawn_half[0], self.spawn_half[0]))
        ry = float(self._rng.uniform(-self.spawn_half[1], self.spawn_half[1]))
        yaw = float(self._rng.uniform(-math.pi, math.pi))
        self.cube_art.reset(
            tp.Vector3(C.TABLE_CX + rx, C.TABLE_CY + ry, C.CUBE_REST_Z),
            tp.Quaternion(0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)),
        )
        for _ in range(20):
            self.world.step(C.PHYS_DT)

    def set_action(self, a):
        a = np.clip(a, -1.0, 1.0)
        arm_tgt = np.clip(C.DEFAULT_Q + C.ARM_ACTION_SCALE * a[0:7],
                          C.ARM_LIMITS[:, 0], C.ARM_LIMITS[:, 1])
        grip = C.GRIP_OPEN + np.clip(a[7], 0.0, 1.0) * (C.GRIP_CLOSE - C.GRIP_OPEN)   # a7<=0 open, a7=+1 close
        self.arm.set_drive_targets(
            np.concatenate([arm_tgt, [grip, grip]]).astype(np.float32))

    def step(self):
        self.world.step(C.DT)   # fixed_timestep=PHYS_DT, max_substeps=SUBSTEPS → 5 substeps

    def tip_state(self):
        """Grasp point = MIDPOINT of the two finger links + tool approach axis (matches GPU tip_state).
        Uses the actual finger meshes — NOT flange + TIP_Z, which sat ~0.13 m below the real fingers."""
        ml = self.arm_meshes[FINGER_L_LINK]
        mr = self.arm_meshes[FINGER_R_LINK]
        tip = 0.5 * (np.array([ml.position.x, ml.position.y, ml.position.z], np.float32)
                     + np.array([mr.position.x, mr.position.y, mr.position.z], np.float32))
        mf = self.arm_meshes[FLANGE_LINK]
        q = np.array([mf.quaternion.x, mf.quaternion.y, mf.quaternion.z, mf.quaternion.w], np.float32)
        approach = _quat_to_R(q)[:, 2]
        return tip, approach

    def cube_pos(self):
        rs = self.cube_art.root_state()    # [px, py, pz, qx, qy, qz, qw]
        return rs[0:3].copy()

    def cube_state(self):
        rs = self.cube_art.root_state()
        rv = self.cube_art.root_velocity() # [vx, vy, vz, wx, wy, wz]
        return rs[0:3].copy(), rs[3:7].copy(), rv[0:3].copy()

    def joint_state(self):
        return self.arm.joint_positions(), self.arm.joint_velocities()


def _build_obs(sim, last_action):
    """Reconstruct the 41-d obs from CPU sim state. Matches kuka_grasp_env._obs() exactly."""
    jp, jv = sim.joint_state()
    tip, approach = sim.tip_state()
    cube_pos, cube_quat, cube_linvel = sim.cube_state()

    opening = 0.5 * (float(jp[7]) + float(jp[8]))
    finger_vel = 0.5 * (float(jv[7]) + float(jv[8]))
    d = cube_pos - tip
    lateral = float(np.linalg.norm(d[0:2]))
    vert = float(abs(d[2]))
    firm = lateral < FIRM_LAT and vert < FIRM_VERT and opening < FIRM_OPEN

    tip_rel = tip - np.array([C.TABLE_CX, C.TABLE_CY, C.TABLE_TOP_Z], np.float32)
    cube_up = _quat_to_R(cube_quat)[:, 2]
    lift_frac = float(np.clip((cube_pos[2] - C.CUBE_REST_Z) / C.LIFT_SUCCESS_DZ, 0.0, 1.5))

    return np.concatenate([
        (jp[0:7] - C.DEFAULT_Q).astype(np.float32),
        (jv[0:7] * C.QVEL_SCALE).astype(np.float32),
        np.array([opening, finger_vel * C.QVEL_SCALE], np.float32),
        tip_rel.astype(np.float32),
        (cube_pos - tip).astype(np.float32),
        approach.astype(np.float32),                 # 3 tool approach axis
        cube_up.astype(np.float32),
        cube_linvel.astype(np.float32),
        last_action.astype(np.float32),
        np.array([float(firm), lift_frac], np.float32),
    ]), firm


def _is_done(cube_pos, step):
    off_x = abs(float(cube_pos[0]) - C.TABLE_CX) > C.TABLE_HALF[0] + LOST_MARGIN
    off_y = abs(float(cube_pos[1]) - C.TABLE_CY) > C.TABLE_HALF[1] + LOST_MARGIN
    fell = float(cube_pos[2]) < C.TABLE_TOP_Z - C.DROP_FAIL_DZ
    return fell or off_x or off_y or step >= C.MAX_STEPS


def _run_eval(model_path, args, play_iter):
    """GPU batch eval — uses the training env directly (needs CUDA)."""
    from kuka_grasp_env import KukaGraspEnv
    from threepp.rl import load_policy
    ac, norm, _ = load_policy(model_path, device="cuda")
    env = KukaGraspEnv(num_envs=args.eval, device="cuda", render_visuals=False)
    env.set_iter(play_iter)
    env.deploy = True            # deploy conditions: arm always starts at the DEFAULT_Q home (df=0)
    obs = env.reset()
    R_log, G_log, S_log = [], [], []
    for t in range(380):
        with torch.no_grad():
            a = ac.act_mean(norm.norm(obs) if norm is not None else obs)
        obs, _, _, _, _ = env.step(a)
        if t >= 80:
            R_log.append(env.last_reach)
            G_log.append(env.last_grasp_rate)
            S_log.append(env.last_success_rate)
    print(f"[eval] N={args.eval}  iter~{play_iter}  (no weld, no pre-pose — true deploy)")
    print(f"[eval] reach={np.mean(R_log):.3f} m   grasp={np.mean(G_log):.2f}   success={np.mean(S_log):.2f}")


def _print_reset_diag(sim):
    """Frame-0 diagnostic: does the CPU play sim reproduce the GPU --inspect pose at DEFAULT_Q?
    GPU --inspect reports tip=(0.524, 0.0, 0.541), down=0.998 (tool straight DOWN). If CPU play
    disagrees here, the bug is in play (control / mesh index / CPU-vs-GPU sim), NOT in training."""
    jp, jv = sim.joint_state()
    tip, approach = sim.tip_state()
    cpos, _, _ = sim.cube_state()
    print(f"[diag] reset jp[0:7] = {np.round(jp[0:7], 3).tolist()}")
    print(f"[diag]     DEFAULT_Q = {np.round(C.DEFAULT_Q, 3).tolist()}  (should match within reset noise)")
    print(f"[diag] reset tip = {np.round(tip, 3).tolist()}  (GPU expects [0.524, 0.0, 0.541])")
    print(f"[diag] reset approach = {np.round(approach, 3).tolist()}  down = {float(-approach[2]):.3f}  (GPU expects 0.998)")
    print(f"[diag] reset cube = {np.round(cpos, 3).tolist()}")
    print(f"[diag] arm_meshes z (link 0..{len(sim.arm_meshes)-1}) = "
          f"{[round(float(m.position.z), 3) for m in sim.arm_meshes]}  (FLANGE_LINK={FLANGE_LINK})")
    # actual gripper OPEN gap (fingers are at GRIP_OPEN after reset) vs the cube width
    fl = sim.arm_meshes[FINGER_L_LINK].position
    fr = sim.arm_meshes[FINGER_R_LINK].position
    centre_gap = float(np.linalg.norm(np.array([fl.x - fr.x, fl.y - fr.y, fl.z - fr.z])))
    inner_gap = centre_gap - C.FINGER_BOX[0]   # subtract one finger thickness → free space between pads
    print(f"[diag] OPEN finger centre-gap = {centre_gap:.3f}  inner (pad-to-pad) = {inner_gap:.3f}  "
          f"cube = {C.CUBE:.3f}  -> {'CLEARS' if inner_gap > C.CUBE else 'TOO NARROW!'} "
          f"(margin {inner_gap - C.CUBE:+.3f})")


def _run_gripcheck():
    """Sweep the gripper command and report, for each, the actual finger JOINT positions and the
    measured pad gap. Tells us whether the fingers track the command (control) or are stuck (geometry).
    The cube (0.07 m) needs an inner pad gap > 0.07 to be straddled."""
    sim = KukaGraspCpuSim(spawn_half=(0.0, 0.0))
    sim.reset()
    arm_q = np.asarray(C.DEFAULT_Q, np.float32)

    def gap():
        fl = sim.arm_meshes[FINGER_L_LINK].position
        fr = sim.arm_meshes[FINGER_R_LINK].position
        return float(np.linalg.norm([fl.x - fr.x, fl.y - fr.y, fl.z - fr.z]))

    print(f"[gripcheck] cube={C.CUBE:.3f}  FINGER_MOUNT_X={C.FINGER_MOUNT_X}  GRIP_OPEN={C.GRIP_OPEN} "
          f"FINGER_UPPER={C.FINGER_UPPER}  finger_thick={C.FINGER_BOX[0]}")
    print(f"[gripcheck] expected inner gap if slide works = 2*(MOUNT_X+grip) - thick")
    for g in (C.GRIP_CLOSE, 0.0, C.GRIP_OPEN, C.FINGER_UPPER, 0.06, 0.09):
        sim.arm.set_drive_targets(np.concatenate([arm_q, [g, g]]).astype(np.float32))
        for _ in range(120):
            sim.world.step(C.PHYS_DT)
        jp, _ = sim.joint_state()
        cg = gap()
        inner = cg - C.FINGER_BOX[0]
        print(f"[gripcheck] cmd={g:+.3f}  ->  jp[7,8]=({float(jp[7]):+.4f},{float(jp[8]):+.4f})  "
              f"centre-gap={cg:.3f}  inner={inner:.3f}  {'CLEARS cube' if inner > C.CUBE else 'too narrow'}")


def _run_scripted(args):
    """Drive the arm DOWN to the verified grasp posture, close the gripper, and lift — IGNORING the
    policy entirely. This isolates kinematics/control/physics from the policy: if the arm visibly
    descends to the cube and lifts it here, then "hovering at a fixed distance" in normal play is a
    POLICY problem (it isn't commanding the descent), not wrong kinematics."""
    pose_path = os.path.join(_HERE, "kuka_grasp_pose.npy")
    if not os.path.exists(pose_path):
        print("[scripted] no kuka_grasp_pose.npy found — run training once; its init writes the "
              "verified descended grasp posture this test drives to.")
        return
    pose = np.load(pose_path).astype(np.float32)
    grasp_q = pose[0:7]
    grasp_tip_xy = pose[7:9]          # finger-midpoint xy of the descended posture
    default_q = np.asarray(C.DEFAULT_Q, np.float32)
    print(f"[scripted] DEFAULT_Q (home)  ->  grasp_q finger-midpoint≈{np.round(pose[7:10], 3).tolist()}")

    sim = KukaGraspCpuSim(spawn_half=(0.0, 0.0))
    sim.reset()
    # Centre the cube directly under the descended posture's fingers (the gate does this too). Without
    # it the cube sits at TABLE_CX while grasp_q's fingers land at ~x=0.52, so a finger clips the cube
    # on the way down and the grasp looks "too narrow" when it is really just off-centre.
    sim.cube_art.reset(tp.Vector3(float(grasp_tip_xy[0]), float(grasp_tip_xy[1]), C.CUBE_REST_Z),
                       tp.Quaternion(0.0, 0.0, 0.0, 1.0))
    for _ in range(20):
        sim.world.step(C.PHYS_DT)
    _print_reset_diag(sim)

    scene = tp.Scene()
    scene.background = tp.Color(0x10151c) if hasattr(tp, "Color") else 0x10151c
    scene.add(tp.HemisphereLight(0xbfd8ff, 0x223044, 1.0))
    sun = tp.DirectionalLight(0xffffff, 2.0)
    sun.position.set(2.5, -2.0, 4.0)
    scene.add(sun)
    scene.add(sim.ground)
    scene.add(sim.tables[0])
    for m in sim.arm_meshes:
        scene.add(m)
    scene.add(sim.cube_mesh)

    canvas = tp.Canvas("threepp – KUKA grasp SCRIPTED (CPU)", width=1100, height=720, antialiasing=4)
    rend = tp.GLRenderer(canvas)
    rend.shadow_map_enabled = True
    camera = tp.PerspectiveCamera(50, canvas.aspect(), 0.02, 50.0)
    camera.up.set(0, 0, 1)
    camera.position.set(1.25, -1.05, 0.95)
    camera.look_at(0.45, 0.0, 0.45)
    controls = tp.OrbitControls(camera, canvas) if hasattr(tp, "OrbitControls") else None
    if controls is not None:
        controls.target.set(0.45, 0.0, 0.45)

    # scripted phases (control steps @ 50 Hz): descend → settle → close → lift → hold, then loop
    DESCEND, SETTLE, CLOSE, LIFT, HOLD = 60, 20, 40, 80, 40
    total = DESCEND + SETTLE + CLOSE + LIFT + HOLD
    state = {"k": 0}

    def scripted_step():
        i = state["k"] % total
        if i < DESCEND:
            f = i / DESCEND
            q, grip, ph = default_q + (grasp_q - default_q) * f, C.GRIP_OPEN, "descend"
        elif i < DESCEND + SETTLE:
            q, grip, ph = grasp_q, C.GRIP_OPEN, "settle"
        elif i < DESCEND + SETTLE + CLOSE:
            q, grip, ph = grasp_q, C.GRIP_CLOSE, "close"
        elif i < DESCEND + SETTLE + CLOSE + LIFT:
            f = (i - (DESCEND + SETTLE + CLOSE)) / LIFT
            q, grip, ph = grasp_q + (default_q - grasp_q) * f, C.GRIP_CLOSE, "lift"
        else:
            q, grip, ph = default_q, C.GRIP_CLOSE, "hold"
        sim.arm.set_drive_targets(np.concatenate([q, [grip, grip]]).astype(np.float32))
        sim.step()
        if i % 15 == 0:
            tip, _ = sim.tip_state()
            print(f"[scripted] {ph:7s} i={i:3d}  tip_z={float(tip[2]):.3f}  cube_z={float(sim.cube_pos()[2]):.3f}")
        if i == total - 1:
            print("[scripted] ---- loop restart (reset) ----")
            sim.reset()
        state["k"] += 1

    wall0, sim_time = [None], [0.0]

    def frame():
        now = time.perf_counter()
        if wall0[0] is None:
            wall0[0] = now
        elapsed = (now - wall0[0]) * max(args.speed, 1e-6)
        guard = 0
        while sim_time[0] + C.DT <= elapsed and guard < 4:
            scripted_step()
            sim_time[0] += C.DT
            guard += 1
        if controls is not None:
            controls.update()
        rend.render(scene, camera)

    print("[scripted] driving the arm down to the cube, closing, lifting — watch the arm AND the "
          "tip_z/cube_z prints (tip_z should fall ~0.54→0.46; cube_z should rise during 'lift')")
    canvas.animate(frame)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(_HERE, "kuka_grasp_latest.pt"))
    ap.add_argument("--shot", metavar="PNG", help="headless: step one episode and save a PNG")
    ap.add_argument("--eval", type=int, metavar="N", help="GPU batch eval over N envs (needs CUDA)")
    ap.add_argument("--spawn", type=float, metavar="F",
                    help="cube spawn breadth 0..1 (default: match the checkpoint's curriculum stage)")
    ap.add_argument("--diag", action="store_true",
                    help="print the frame-0 reset state (tip/down/joints) and exit — no policy needed")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback speed multiplier (1.0 = real-time 50 Hz control; 0.5 = half speed)")
    ap.add_argument("--scripted", action="store_true",
                    help="ignore the policy: drive the arm DOWN to the verified grasp posture, close, "
                         "and lift — proves CPU kinematics/control descend (vs. the policy not commanding it)")
    ap.add_argument("--gripcheck", action="store_true",
                    help="sweep the gripper open/closed and print commanded vs actual finger joint pos "
                         "and the resulting pad gap vs the cube — diagnoses why the gripper is too narrow")
    args = ap.parse_args()

    if args.diag:
        sim = KukaGraspCpuSim(spawn_half=(0.0, 0.0))
        sim.reset()
        _print_reset_diag(sim)
        return

    if args.scripted:
        _run_scripted(args)
        return

    if args.gripcheck:
        _run_gripcheck()
        return

    # fall back to any checkpoint that exists
    for alt in ("kuka_grasp_latest.pt", "kuka_grasp_best.pt", "kuka_grasp.pt"):
        if os.path.exists(args.model):
            break
        args.model = os.path.join(_HERE, alt)

    if args.eval:
        from threepp.rl import load_policy
        _, _, meta = load_policy(args.model, device="cpu")
        _run_eval(args.model, args, int(meta.get("iter", 400)))
        return

    from threepp.rl import load_policy
    print(f"[play] policy: {args.model}")
    ac, norm, meta = load_policy(args.model, device="cpu")
    play_iter = int(meta.get("iter", 400))

    t = np.clip(float(args.spawn) if args.spawn is not None else play_iter / 1200.0, 0.0, 1.0)
    spawn_half = (0.02 + t * (C.TABLE_HALF[0] - 0.06), 0.02 + t * (C.TABLE_HALF[1] - 0.06))
    print(f"[play] spawn={spawn_half[0]:.2f}x{spawn_half[1]:.2f}  iter~{play_iter}  CPU physics (no weld)")

    sim = KukaGraspCpuSim(spawn_half=spawn_half)
    sim.reset()
    _print_reset_diag(sim)

    # ---- scene ----------------------------------------------------------------
    scene = tp.Scene()
    scene.background = tp.Color(0x10151c) if hasattr(tp, "Color") else 0x10151c
    scene.add(tp.HemisphereLight(0xbfd8ff, 0x223044, 1.0))
    sun = tp.DirectionalLight(0xffffff, 2.0)
    sun.position.set(2.5, -2.0, 4.0)
    scene.add(sun)
    scene.add(sim.ground)
    scene.add(sim.tables[0])
    for m in sim.arm_meshes:
        scene.add(m)
    scene.add(sim.cube_mesh)

    headless = bool(args.shot)
    canvas = tp.Canvas("threepp – KUKA grasp (CPU)", width=1100, height=720,
                       antialiasing=4, headless=headless)
    rend = tp.GLRenderer(canvas)
    rend.shadow_map_enabled = True

    camera = tp.PerspectiveCamera(50, canvas.aspect(), 0.02, 50.0)
    camera.up.set(0, 0, 1)
    camera.position.set(1.25, -1.05, 0.95)
    camera.look_at(0.45, 0.0, 0.45)

    last_action = np.zeros(C.ACT_DIM, np.float32)
    step_count = [0]
    reset_held = [False]

    def do_reset():
        sim.reset()
        last_action[:] = 0.0
        step_count[0] = 0

    @torch.no_grad()
    def policy_step():
        obs, _ = _build_obs(sim, last_action)
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        if norm is not None:
            obs_t = norm.norm(obs_t)
        a = ac.act_mean(obs_t).squeeze(0).numpy()
        sim.set_action(a)
        sim.step()
        last_action[:] = np.clip(a, -1.0, 1.0)
        step_count[0] += 1
        if _is_done(sim.cube_pos(), step_count[0]):
            do_reset()

    if headless:
        for _ in range(C.MAX_STEPS):
            policy_step()
        rend.render(scene, camera)
        rend.save_frame(args.shot)
        print(f"[play] saved {args.shot}  cube_z={float(sim.cube_pos()[2]):.3f}")
        return

    controls = tp.OrbitControls(camera, canvas) if hasattr(tp, "OrbitControls") else None
    if controls is not None:
        controls.target.set(0.45, 0.0, 0.45)

    def on_resize(w, h):
        camera.aspect = w / max(h, 1)
        camera.update_projection_matrix()
        rend.set_size(w, h)
    canvas.on_window_resize(on_resize)

    print(f"[play] keys:  R = reset   drag = orbit   |   speed {args.speed:g}x (50 Hz control)")

    # Pace the control loop to WALL-CLOCK time so it plays at real-time (1x), independent of the render
    # FPS. Previously policy_step() ran once per rendered frame, so at 100+ FPS the sim advanced many
    # 50 Hz control ticks per real second — hence "super fast". We now advance the sim only as much real
    # time has elapsed (× speed), and render every frame regardless.
    wall0 = [None]
    sim_time = [0.0]

    def frame():
        if hasattr(canvas, "is_key_down") and canvas.is_key_down("R"):
            if not reset_held[0]:
                do_reset()
                wall0[0] = time.perf_counter()
                sim_time[0] = 0.0
                reset_held[0] = True
        else:
            reset_held[0] = False

        now = time.perf_counter()
        if wall0[0] is None:
            wall0[0] = now
        elapsed = (now - wall0[0]) * max(args.speed, 1e-6)
        guard = 0
        while sim_time[0] + C.DT <= elapsed and guard < 4:   # cap catch-up so a stall can't spiral
            policy_step()
            sim_time[0] += C.DT
            guard += 1

        if controls is not None:
            controls.update()
        rend.render(scene, camera)

    canvas.animate(frame)


if __name__ == "__main__":
    main()
