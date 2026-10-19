"""kuka_grasp_env.py — the KUKA grasp RL environment, a threepp.rl.VecTask.

The policy commands the 7 arm joints and the gripper (8 actions) to reach a cube on the table,
close the parallel-jaw gripper on it and lift it, held by finger friction alone (there is no
weld or kinematic attach anywhere).

The world is KukaGraspSim: a GpuSim arm batch plus a second batch for the free-base cubes, so
the env passes build_robot=None to VecTask and attaches the sim itself; everything else (the
step/reset choreography, terminal-obs capture, env_state() re-initialisation, per-term reward
logging, CUDA-graph replay of the per-step torch region) is VecTask's.

Two modes:

    scaffold  the June env: a shaped reach/centre/descend/close/lift reward and a reset-height
              curriculum driven by set_iter(it), so early episodes start with the open fingers
              straddling a centred cube and later ones start anywhere up to the DEFAULT_Q home.
              Failure = the cube leaves the table.
    pool      the recipe of Zhang et al. (CoRL 2026): every reset draws a cell of the
              collision-checked reset pool (kuka_grasp_pool.py; Reaching, Stable Grasp and Near-
              Goal starts) from a threepp.rl.sgs.SuccessGuidedSampler (any mode) or uniformly,
              and one shared sparse reward
                  r = 100 success - 1 failure - 1e-5 |a|^2 - 1e-4 |a - a_prev|^2 - 1e-4 |qdot|^2
              (each squared norm clipped at 1e4). Success and failure end the episode. Failure =
              the cube leaves the table or an arm joint exceeds twice its URDF velocity limit.
              Actions are deltas of the PD targets (contract: "delta"): the arm's at up to the
              URDF velocity limits, the gripper's at GRIP_RATE. Zero action holds the command,
              so a Stable Grasp or Near-Goal start stays held under it; under the June absolute
              mapping zero action opens the gripper, and the drive (stiffness 1200, max force
              300) exceeds twice the velocity limit on almost every large command (the
              scaffold's own zero-action return to DEFAULT_Q: A4 at 5.4x).

Success, for every mode: the cube centre LIFT_SUCCESS_DZ above its rest height and within
SUCCESS_NEAR of the finger midpoint for SUCCESS_HOLD_STEPS consecutive control steps.

    python kuka_grasp_env.py            # selftest (the gate)
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import kuka_grasp_contract as C  # noqa: E402
import kuka_grasp_robot as R  # noqa: E402
from kuka_grasp_kin import KukaKinematics, down_frame  # noqa: E402
from kuka_grasp_sim import KukaGraspSim  # noqa: E402
from threepp.rl import RobotState, VecTask  # noqa: E402

# --------------------------------------------------------------------------- #
#  Scaffold reward weights + grasp thresholds (the June env's)
# --------------------------------------------------------------------------- #
W_REACH = 1.0
K_REACH = 6.0          # exp(-K_REACH * ||tip - cube||)
W_DOWN = 1.0           # tool pointing straight down
K_DOWN = 3.0           # exp(-K_DOWN * (1 - down)); down = -approach_z, 1 = straight down
W_CENTER = 0.6
K_CENTER = 22.0        # exp(-K_CENTER * lateral_xy)
W_VERT = 1.5           # tip descending to the cube's height, gated on being over it
K_VERT = 10.0
W_CLOSE = 0.6          # closing the fingers while in the grasp zone
W_GRASP = 1.5          # per-step bonus for a firm grasp
W_LIFT = 24.0          # per metre of lift, gated on a firm grasp
W_SUCCESS = 6.0        # per step held above the lift height
W_ACT_RATE = 0.004
W_ACT_MAG = 0.001
W_LIMIT = 0.08
W_FALL = 3.0           # one-time penalty when the cube leaves the table

# A firm grasp = tightly centred and the fingers closed on the cube (an observation flag in
# every mode; the scaffold's lift and grasp rewards are gated on it).
FIRM_LAT = 0.018
FIRM_VERT = 0.045
FIRM_OPEN = 0.018
OVER_LAT = 0.045       # the scaffold's close reward fires only in the grasp zone
OVER_VERT = 0.045
LOST_MARGIN = C.LOST_MARGIN

ARM_RESET_NOISE = 0.05  # rad jitter on the scaffold's reset posture

# the pool mode's shared reward (Zhang et al.)
R_SUCCESS = 100.0
R_FAILURE = 1.0
R_ACT = 1e-5
R_RATE = 1e-4
R_QVEL = 1e-4
R_CLIP = 1e4


class KukaGraspEnv(VecTask):
    control_hz = C.CONTROL_HZ
    episode_s = C.EPISODE_S
    act_dim = C.ACT_DIM
    control = "drive"
    substeps = C.SUBSTEPS
    clip_actions = 1.0

    def __init__(self, num_envs=512, device="cuda", seed=0, *, mode="scaffold", pool=None,
                 sampler=None, cells=None, render_visuals=False, graph=False):
        """
        mode      "scaffold" (the June env) or "pool"
        pool      pool mode: kuka_grasp_pool.load_pool(...) tensors
        sampler   pool mode: a SuccessGuidedSampler over the pool's cells (any of its modes);
                  None draws cells uniformly
        cells     pool mode: a fixed [M] cell list, env i always resets into cells[i % M]
                  (the evaluation and the viewer); the sampler is then neither drawn nor fed
        """
        if mode not in ("scaffold", "pool"):
            raise ValueError(f"unknown mode {mode!r}")
        if mode == "pool" and pool is None:
            raise ValueError("pool mode needs a pool (kuka_grasp_pool.load_pool)")
        super().__init__(num_envs, None, device=device, seed=seed, graph=graph)
        self.mode = mode
        self.sim = KukaGraspSim(num_envs, device=device, render_visuals=render_visuals)
        self.state = RobotState(self.sim, self.dt)
        dev = self.device
        self.default_q = torch.tensor(C.DEFAULT_Q, device=dev)
        self.arm_lo = torch.tensor(C.ARM_LIMITS[:, 0], device=dev)
        self.arm_hi = torch.tensor(C.ARM_LIMITS[:, 1], device=dev)
        self.arm_scale = torch.tensor(C.ARM_ACTION_SCALE, device=dev)
        self.vel_fail = torch.tensor(C.ARM_VEL_LIMITS * C.FAIL_VEL_FACTOR, device=dev)
        self.env_x = self.sim.env_x
        self.kin = KukaKinematics(R.find_kuka_urdf(), device=dev)

        # per-env state the base re-initialises on every reset
        self.last_action = self.env_state((C.ACT_DIM,))
        self.prev_action = self.env_state((C.ACT_DIM,))
        self.hold = self.env_state(())                       # consecutive lifted-and-held steps
        self.ep_success = self.env_state((), init=False, dtype=torch.bool)
        self.target = self.env_state((C.N_DOF,))             # pool: the PD target the deltas move

        # per-step masks, written in place inside the (graph-capturable) hot region
        self.success = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        self.fail_cube = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        self.fail_vel = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        # device-side metric accumulators, read (one host sync each) only in stats_line()
        self._m = {k: torch.zeros((), device=dev) for k in ("reach", "firm", "steps")}
        self._ep = {k: torch.zeros((), device=dev) for k in ("n", "succ", "fail_cube", "fail_vel")}
        # first-episode outcome per env (-1 = not finished yet): what the evaluation reads
        self.first_outcome = torch.full((num_envs,), -1, dtype=torch.long, device=dev)
        self.last_metrics = {}

        # the scaffold's curriculum state (set_iter) and its descended grasp posture
        self.spawn_half = torch.tensor([0.0, 0.0], device=dev)
        self.start_high = 0.0
        self.cur_iter = 0
        if mode == "scaffold":
            self._init_grasp_posture()
            self.set_iter(0)

        # the pool mode's cells
        self.pool, self.sampler = pool, sampler
        self.fixed_cells = None if cells is None else torch.as_tensor(cells, dtype=torch.long, device=dev)
        self.cell = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self.delta = mode == "pool"
        # whether a velocity violation ends a pool episode; a tensor because terminated() runs
        # inside the CUDA graph (the evaluation turns it off for scaffold policies)
        self.vel_terminal = torch.ones((), dtype=torch.bool, device=dev)
        self.arm_step = torch.tensor(C.ARM_VEL_LIMITS, device=dev) * self.dt
        self.grip_step = C.GRIP_RATE * self.dt
        self._fam = torch.zeros(3, 2, device=dev)                # per family: episodes, successes

    # ---- the scaffold ---------------------------------------------------------------
    def _init_grasp_posture(self):
        """The descended posture of the reset-height curriculum: open fingers straddling a cube
        resting at the table centre, tool down. Solved by IK (the June env ran a random
        posture search in the simulator and cached it), then settled under the PD drive with
        gravity in every env at once and measured, because the arm sags below its target.
        The grasp point sits at GRASP_TABLE_Z; the June target, the cube centre, put the
        finger tips 7.5 mm inside the table."""
        dev = self.device
        target = torch.tensor([[C.TABLE_CX, C.TABLE_CY, C.GRASP_TABLE_Z]], device=dev)
        _, R0 = self.kin.grasp_frame(self.default_q.view(1, 7))
        yaw0 = torch.atan2(R0[:, 1, 0], R0[:, 0, 0])
        q, pe, re = self.kin.ik(target, down_frame(yaw0), self.default_q.view(1, 7))
        if pe.item() > 2e-3 or re.item() > 2e-2:
            raise RuntimeError(f"kuka_grasp: no IK posture at the cube (err {pe.item():.4f} m, "
                               f"{re.item():.3f} rad)")
        q9 = torch.cat([q[0], torch.tensor([C.GRIP_OPEN, C.GRIP_OPEN], device=dev)])
        q9k = q9.view(1, 9).expand(self.K, 9).contiguous()
        self.sim.set_joint_state(torch.arange(self.K, device=dev), q9k, torch.zeros_like(q9k))
        self.sim.settle_arm(q9, 150)
        tip = self.sim.tip_state()[0][0] - torch.stack([self.env_x[0], torch.zeros((), device=dev),
                                                       torch.zeros((), device=dev)])
        if (tip[0:2] - target[0, 0:2]).norm().item() > 0.03 or abs(tip[2].item() - target[0, 2].item()) > 0.04:
            raise RuntimeError(f"kuka_grasp: the settled grasp posture misses the cube (tip {tip.tolist()})")
        self.grasp_q = q[0]
        self.grasp_cube_xy = tip[0:2].clone()

    def set_iter(self, it):
        """The scaffold's schedule: the cube spawn region grows from the table centre to the
        whole table over 1200 iterations, and the reset height spreads from at-the-cube
        (start_high 0) to anywhere up to the DEFAULT_Q home (1) over 500."""
        t = min(max(it / 1200.0, 0.0), 1.0)
        sx = 0.02 + t * (C.TABLE_HALF[0] - 0.06)
        sy = 0.02 + t * (C.TABLE_HALF[1] - 0.06)
        self.spawn_half = torch.tensor([sx, sy], device=self.device)
        self.start_high = float(min(it / 500.0, 1.0))
        self.cur_iter = it

    def _reset_scaffold(self, idx):
        n = idx.numel()
        dev, g = self.device, self.g
        u = torch.rand(n, generator=g, device=dev) * self.start_high
        df = (1.0 - u).unsqueeze(1)                 # 1 = at the cube, 0 = the DEFAULT_Q home
        q9 = torch.zeros(n, C.N_DOF, device=dev)
        q9[:, 0:7] = (self.default_q + df * (self.grasp_q - self.default_q)
                      + ARM_RESET_NOISE * (torch.rand(n, 7, generator=g, device=dev) * 2 - 1))
        q9[:, 7] = C.GRIP_OPEN
        q9[:, 8] = C.GRIP_OPEN
        self.sim.set_joint_state(idx, q9, torch.zeros_like(q9))
        # the cube under the descended fingers; its lateral spread grows as the start rises
        spread = 1.0 - df.squeeze(1)
        rx = (torch.rand(n, generator=g, device=dev) * 2 - 1) * self.spawn_half[0] * spread
        ry = (torch.rand(n, generator=g, device=dev) * 2 - 1) * self.spawn_half[1] * spread
        yaw = (torch.rand(n, generator=g, device=dev) * 2 - 1) * np.pi * spread
        pose = torch.zeros(n, 7, device=dev)
        pose[:, 2] = torch.sin(yaw / 2)
        pose[:, 3] = torch.cos(yaw / 2)
        pose[:, 4] = self.env_x[idx] + self.grasp_cube_xy[0] + rx
        pose[:, 5] = self.grasp_cube_xy[1] + ry
        pose[:, 6] = C.CUBE_REST_Z + C.CUBE_PLACE_DZ
        self.sim.set_cube_state(idx, pose)

    # ---- the pool ---------------------------------------------------------------------
    def _reset_pool(self, idx):
        n = idx.numel()
        p = self.pool
        if self.fixed_cells is not None:
            cells = self.fixed_cells[idx % self.fixed_cells.numel()]
        elif self.sampler is not None:
            cells = self.sampler.sample(n)
        else:
            cells = torch.randint(0, p["n"], (n,), device=self.device, generator=self.g)
        self.cell[idx] = cells
        q9 = p["q9"][cells]
        self.sim.set_joint_state(idx, q9, torch.zeros_like(q9))
        self.target[idx] = p["hold"][cells]        # the command the cell was verified under
        pose = p["cube"][cells].clone()
        pose[:, 4] += self.env_x[idx]
        pose[:, 6] += C.CUBE_PLACE_DZ * p["on_table"][cells].float()
        self.sim.set_cube_state(idx, pose)

    # ---- VecTask hooks ----------------------------------------------------------------
    def on_reset(self, idx):
        if self.mode == "pool":
            self._reset_pool(idx)
        else:
            self._reset_scaffold(idx)

    def reset(self):
        self.first_outcome.fill_(-1)
        return super().reset()

    def act(self, a):
        """The PD targets [K, 9] for the clamped action. delta (pool): move the held targets by
        a * rate * dt. absolute (scaffold): arm targets about DEFAULT_Q and one mirrored gripper
        command, a7 <= 0 open, a7 = +1 fully closed."""
        self.prev_action.copy_(self.last_action)
        self.last_action.copy_(a)
        if self.delta:
            arm = torch.clamp(self.target[:, 0:7] + a[:, 0:7] * self.arm_step, self.arm_lo, self.arm_hi)
            grip = (self.target[:, 7] - a[:, 7] * self.grip_step).clamp(C.GRIP_CLOSE, C.GRIP_OPEN)
            self.target[:, 0:7] = arm
            self.target[:, 7] = grip
            self.target[:, 8] = grip
            return self.target.clone()
        arm = torch.clamp(self.default_q + self.arm_scale * a[:, 0:7], self.arm_lo, self.arm_hi)
        grip = C.GRIP_OPEN + a[:, 7].clamp(0.0, 1.0) * (C.GRIP_CLOSE - C.GRIP_OPEN)
        return torch.cat([arm, grip.unsqueeze(1), grip.unsqueeze(1)], dim=1)

    def _grasp_state(self):
        tip, approach = self.sim.tip_state()
        cube = self.sim.cube_position()
        d = cube - tip
        lateral = torch.linalg.norm(d[:, 0:2], dim=1)
        vert = d[:, 2].abs()
        dist = torch.linalg.norm(d, dim=1)
        opening = self.sim.finger_opening()
        firm = (lateral < FIRM_LAT) & (vert < FIRM_VERT) & (opening < FIRM_OPEN)
        return tip, cube, lateral, vert, dist, opening, firm, approach

    def on_step(self, s):
        """Advance the hold counter: lifted and between the fingers, consecutively."""
        tip, cube = self.sim.tip_state()[0], self.sim.cube_position()
        held = ((cube[:, 2] > C.CUBE_REST_Z + C.LIFT_SUCCESS_DZ)
                & ((cube - tip).norm(dim=1) < C.SUCCESS_NEAR))
        self.hold.copy_(torch.where(held, self.hold + 1.0, torch.zeros_like(self.hold)))
        self.success.copy_(self.hold >= C.SUCCESS_HOLD_STEPS)
        self.ep_success.logical_or_(self.success)

    def terminated(self, s):
        cube = self.sim.cube_position()
        off_x = (cube[:, 0] - (self.env_x + C.TABLE_CX)).abs() > (C.TABLE_HALF[0] + LOST_MARGIN)
        off_y = (cube[:, 1] - C.TABLE_CY).abs() > (C.TABLE_HALF[1] + LOST_MARGIN)
        self.fail_cube.copy_((cube[:, 2] < (C.TABLE_TOP_Z - C.DROP_FAIL_DZ)) | off_x | off_y)
        self.fail_vel.copy_((s.joint_vel[:, 0:7].abs() > self.vel_fail).any(dim=1))
        if self.mode == "pool":
            return self.success | self.fail_cube | (self.fail_vel & self.vel_terminal)
        return self.fail_cube.clone()

    def observe(self, s):
        tip, cube, lateral, vert, dist, opening, firm, approach = self._grasp_state()
        tip_rel = torch.stack([tip[:, 0] - self.env_x - C.TABLE_CX, tip[:, 1] - C.TABLE_CY,
                               tip[:, 2] - C.TABLE_TOP_Z], dim=1)
        lift_frac = ((cube[:, 2] - C.CUBE_REST_Z) / C.LIFT_SUCCESS_DZ).clamp(0.0, 1.5)
        finger_vel = 0.5 * (s.joint_vel[:, 7] + s.joint_vel[:, 8])
        return torch.cat([
            s.joint_pos[:, 0:7] - self.default_q,                      # 7 arm qpos_rel
            s.joint_vel[:, 0:7] * C.QVEL_SCALE,                        # 7 arm qvel
            torch.stack([opening, finger_vel * C.QVEL_SCALE], dim=1),  # 2 gripper
            tip_rel,                                                   # 3 tip, table-relative
            cube - tip,                                                # 3 tip -> cube
            approach,                                                  # 3 tool approach axis
            self.sim.cube_up(),                                        # 3 cube up-axis
            self.sim.cube_linvel,                                      # 3 cube linvel
            self.last_action,                                          # 8 last action
            torch.stack([firm.float(), lift_frac], dim=1),             # 2 grasp flag, lift frac
        ], dim=1)

    def reward_terms(self, s, a):
        tip, cube, lateral, vert, dist, opening, firm, approach = self._grasp_state()
        self._m["reach"] += dist.sum()
        self._m["firm"] += firm.float().sum()
        self._m["steps"] += float(self.K)
        if self.mode == "pool":
            failure = (self.fail_cube | self.fail_vel) & ~self.success
            return {
                "success": R_SUCCESS * self.success.float(),
                "failure": -R_FAILURE * failure.float(),
                "act": -R_ACT * (a ** 2).sum(dim=1).clamp(max=R_CLIP),
                "rate": -R_RATE * ((a - self.prev_action) ** 2).sum(dim=1).clamp(max=R_CLIP),
                "qvel": -R_QVEL * (s.joint_vel ** 2).sum(dim=1).clamp(max=R_CLIP),
            }
        over_cube = (lateral < OVER_LAT) & (vert < OVER_VERT)
        close_amt = ((C.GRIP_OPEN - opening) / (C.GRIP_OPEN - C.GRIP_CLOSE)).clamp(0.0, 1.0)
        down = -approach[:, 2]
        lift = (cube[:, 2] - C.CUBE_REST_Z).clamp(min=0.0)
        above = firm & (cube[:, 2] > C.CUBE_REST_Z + C.LIFT_SUCCESS_DZ)
        q = s.joint_pos[:, 0:7]
        over = (q - 0.95 * self.arm_hi).clamp(min=0.0) + (0.95 * self.arm_lo - q).clamp(min=0.0)
        tip_above = (tip[:, 2] - cube[:, 2]).clamp(min=0.0)
        over_xy = torch.exp(-12.0 * lateral)
        return {
            "reach": W_REACH * torch.exp(-K_REACH * dist),
            "down": W_DOWN * torch.exp(-K_DOWN * (1.0 - down)),
            "centre": W_CENTER * torch.exp(-K_CENTER * lateral),
            "descend": W_VERT * torch.exp(-K_VERT * tip_above) * over_xy,
            "close": W_CLOSE * close_amt * over_cube.float(),
            "grasp": W_GRASP * firm.float(),
            "lift": W_LIFT * lift * firm.float(),
            "above": W_SUCCESS * above.float(),
            "rate": -W_ACT_RATE * ((a - self.prev_action) ** 2).sum(dim=1),
            "mag": -W_ACT_MAG * (a ** 2).sum(dim=1),
            "limit": -W_LIMIT * (over ** 2).sum(dim=1),
            "fall": -W_FALL * s.terminated.float(),
        }

    def on_done(self, idx):
        succ = self.ep_success[idx]
        self._ep["n"] += float(idx.numel())
        self._ep["succ"] += succ.float().sum()
        self._ep["fail_cube"] += self.fail_cube[idx].float().sum()
        self._ep["fail_vel"] += self.fail_vel[idx].float().sum()
        first = self.first_outcome[idx]
        self.first_outcome[idx] = torch.where(first < 0, succ.long(), first)
        if self.mode == "pool":
            fam = self.pool["family"][self.cell[idx]]
            self._fam[:, 0].index_add_(0, fam, torch.ones_like(succ, dtype=torch.float32))
            self._fam[:, 1].index_add_(0, fam, succ.float())
            if self.sampler is not None and self.fixed_cells is None:
                self.sampler.record(self.cell[idx], succ)

    # ---- logging ------------------------------------------------------------------------
    def metrics(self, reset=True):
        """Episode success / failure rates and mean tip-cube distance / firm-grasp share since
        the last call, as host floats (one sync per value)."""
        n, st = self._ep["n"].item(), self._m["steps"].item()
        out = {"episodes": int(n),
               "success": self._ep["succ"].item() / n if n else float("nan"),
               "fail_cube": self._ep["fail_cube"].item() / n if n else float("nan"),
               "fail_vel": self._ep["fail_vel"].item() / n if n else float("nan"),
               "reach": self._m["reach"].item() / st if st else float("nan"),
               "firm": self._m["firm"].item() / st if st else float("nan")}
        if self.mode == "pool":
            fam = self._fam.cpu()
            for f, name in enumerate(("reaching", "stable_grasp", "near_goal")):
                out[f"success_{name}"] = float(fam[f, 1] / fam[f, 0]) if fam[f, 0] > 0 else float("nan")
        if reset:
            for v in (*self._m.values(), *self._ep.values()):
                v.zero_()
            self._fam.zero_()
        return out

    def stats_line(self):
        base = super().stats_line()
        m = self.last_metrics = self.metrics()
        line = (f"{base}\n        succ {m['success']:.3f}  fail_cube {m['fail_cube']:.3f}  "
                f"fail_vel {m['fail_vel']:.3f}  reach {m['reach']:.3f}  firm {m['firm']:.3f}  "
                f"eps {m['episodes']}")
        if self.mode == "pool":
            line += (f"  | succ R {m['success_reaching']:.3f}  SG {m['success_stable_grasp']:.3f}  "
                     f"NG {m['success_near_goal']:.3f}")
        return line

    def config(self):
        return {**super().config(), "task": "kuka_grasp", "mode": self.mode,
                "obs_dim": C.OBS_DIM, "act_dim": C.ACT_DIM, "substeps": C.SUBSTEPS,
                "phys_dt": C.PHYS_DT, "default_q": C.DEFAULT_Q.tolist(),
                "arm_action_scale": C.ARM_ACTION_SCALE.tolist(),
                "grip_open": C.GRIP_OPEN, "grip_close": C.GRIP_CLOSE,
                "action": "delta" if self.delta else "absolute", "grip_rate": C.GRIP_RATE,
                "arm_vel_limits": C.ARM_VEL_LIMITS.tolist(), "gravity_comp": self.sim.gravity_comp}


# --------------------------------------------------------------------------- #
#  selftest: shapes / no NaN / cube rests / graph replay matches eager
# --------------------------------------------------------------------------- #
def _selftest(K=64, mode="pool"):
    """One mode per process (PhysX allows one foundation per process):
    python kuka_grasp_env.py 64 pool  /  python kuka_grasp_env.py 64 scaffold"""
    import time
    torch.manual_seed(0)
    pool = None
    if mode == "pool":
        from kuka_grasp_pool import NEAR_GOAL, REACHING, STABLE_GRASP, load_pool
        pool = load_pool()
    t0 = time.perf_counter()
    env = KukaGraspEnv(num_envs=K, mode=mode, pool=pool, graph=True)
    print(f"[selftest] {mode}: built K={K} in {time.perf_counter() - t0:.1f} s")
    obs = env.reset()
    assert obs.shape == (K, C.OBS_DIM), obs.shape
    assert torch.isfinite(obs).all(), "non-finite obs after reset"
    print(f"[selftest] obs {tuple(obs.shape)} OBS_DIM={C.OBS_DIM} ACT_DIM={C.ACT_DIM}  OK")

    zero = torch.zeros(K, C.ACT_DIM, device=env.device)
    if mode == "pool":
        # zero action holds every command: Reaching cubes stay put, held cubes stay held
        fam = pool["family"]
        pick = [torch.nonzero(fam == f).squeeze(1)[: K // 3 + 1] for f in (REACHING, STABLE_GRASP, NEAR_GOAL)]
        cells = torch.cat(pick)[:K]
        env.fixed_cells = cells
        env.reset()
        c0 = env.sim.cube_position().clone()
        for _ in range(10):
            env.step(zero)
        f = fam[cells]
        tip, cube = env.sim.tip_state()[0], env.sim.cube_position()
        moved = (cube - c0).norm(dim=1)
        near = (cube - tip).norm(dim=1)
        ok_r = (moved[f == REACHING] < 5e-3).all().item()
        ok_h = (near[f != REACHING] < 0.02).all().item()
        ok_ng = (cube[f == NEAR_GOAL, 2] > C.CUBE_REST_Z + 0.005).all().item()
        print(f"[selftest] zero action, 10 steps: Reaching cubes moved max {moved[f == REACHING].max() * 1e3:.1f} mm, "
              f"held cubes to finger midpoint max {near[f != REACHING].max() * 1e3:.1f} mm  "
              f"{'OK' if ok_r and ok_h and ok_ng else 'FAIL'}")
        assert ok_r and ok_h and ok_ng
        env.fixed_cells = None
        env.reset()
    else:
        for _ in range(10):        # the cube must rest on the table
            env.step(zero)
        cube_z = env.sim.cube_position()[:, 2]
        assert (cube_z > C.TABLE_TOP_Z - 0.02).all(), f"cube fell through the table: {cube_z}"
        print(f"[selftest] cube on the table after 10 zero-action steps  z={cube_z.mean():.3f} "
              f"(rest {C.CUBE_REST_Z:.3f})  OK")

    worst = env.verify_graph(steps=16)
    print(f"[selftest] graph replay vs eager observe: worst {worst:.2e}  OK")

    t0 = time.perf_counter()
    n = 2 * env.max_steps
    for k in range(n):
        a = torch.rand(K, C.ACT_DIM, device=env.device) * 2 - 1
        obs, rew, done, term, tmo = env.step(a)
        assert obs.shape == (K, C.OBS_DIM) and rew.shape == (K,) and done.shape == (K,)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all(), f"non-finite at step {k}"
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    m = env.metrics(reset=False)
    print(f"[selftest] {n} random steps OK  {n * K / dt:.0f} env-steps/s  episodes {m['episodes']}  "
          f"success {m['success']:.2f}  fail_cube {m['fail_cube']:.2f}  fail_vel {m['fail_vel']:.2f}")
    print(f"[selftest] per-term: {env.stats_line()}")
    print("[selftest] (the friction-grasp physics gate is kuka_grasp_friction.py)")
    print("[selftest] PASS")


if __name__ == "__main__":
    _selftest(int(sys.argv[1]) if len(sys.argv) > 1 else 64, sys.argv[2] if len(sys.argv) > 2 else "pool")
