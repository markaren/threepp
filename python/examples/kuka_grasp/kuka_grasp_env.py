"""kuka_grasp_env.py — the KUKA grasp RL environment (threepp.rl contract).

The policy commands all 7 arm joints + the gripper (8 actions) to reach a cube on the table, close
the parallel-jaw gripper, and lift it. Reward is a sum of shaped terms (reach → centre → grasp →
lift → success). A friction grasp is the goal; an annealed grasp-lock scaffold bootstraps the
reach→close→lift sequence early in training and is removed (assist_frac → 0) over the curriculum.

Contract (see python/threepp/rl/README.md):
    reset() -> obs [K, OBS_DIM]
    step(action [K, ACT_DIM]) -> (obs, reward, done, terminal_obs, is_timeout)
`step` auto-resets finished envs; `is_timeout` marks time-limit truncations (bootstrap V(term_obs))
vs. true terminals (cube knocked off the table → bootstrap 0).
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import kuka_grasp_contract as C  # noqa: E402
from kuka_grasp_sim import KukaGraspSim  # noqa: E402

# --------------------------------------------------------------------------- #
#  Reward weights + grasp thresholds (tunable; kept here, not scattered)
# --------------------------------------------------------------------------- #
W_REACH = 1.0
K_REACH = 6.0          # exp(-K_REACH * ||tip - cube||)
W_CENTER = 0.5
K_CENTER = 18.0        # exp(-K_CENTER * lateral_xy) — centre the tip over the cube
W_GRASP = 1.0          # per-step bonus while the gripper is closed around the cube
W_LIFT = 22.0          # per metre of lift, gated on actually holding the cube
W_SUCCESS = 5.0        # per-step bonus while holding the cube above the success height
W_ACT_RATE = 0.004
W_ACT_MAG = 0.001
W_LIMIT = 0.08
W_FALL = 3.0           # one-time penalty when the cube is knocked off the table

GRASP_LAT = 0.035      # lateral tip↔cube distance for "between the fingers"
GRASP_VERT = 0.055     # vertical tip↔cube distance for "fingers around the cube"
GRASP_OPEN_MAX = 0.024 # finger opening below this == closed on the cube
HELD_RADIUS = 0.08     # cube within this of the tip counts as held (gates the lift reward)
LOST_MARGIN = 0.10     # cube this far beyond the table edge == knocked off → terminal fail

ARM_RESET_NOISE = 0.05  # rad jitter on the arm reset posture


class KukaGraspEnv:
    def __init__(self, num_envs=2048, device="cuda", seed=0, spacing=2.5, render_visuals=False):
        self.sim = KukaGraspSim(num_envs, device=device, spacing=spacing, render_visuals=render_visuals)
        self.K = num_envs
        self.device = self.sim.device
        self.max_steps = C.MAX_STEPS
        self.g = torch.Generator(device=self.device).manual_seed(seed)

        dev = self.device
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self.last_action = torch.zeros(num_envs, C.ACT_DIM, device=dev)
        self.default_q = torch.tensor(C.DEFAULT_Q, device=dev)
        self.arm_lo = torch.tensor(C.ARM_LIMITS[:, 0], device=dev)
        self.arm_hi = torch.tensor(C.ARM_LIMITS[:, 1], device=dev)
        self.arm_scale = torch.tensor(C.ARM_ACTION_SCALE, device=dev)
        self.env_x = self.sim.env_x

        # curriculum state (set via set_iter)
        self.spawn_half = torch.tensor([0.0, 0.0], device=dev)   # cube spawn region half-extent (x,y)
        self.assist_frac = 1.0                                    # fraction of envs with grasp-lock assist
        self.assisted = torch.zeros(num_envs, dtype=torch.bool, device=dev)

        # metrics (read in on_log)
        self.last_reach = 0.0
        self.last_grasp_rate = 0.0
        self.last_lift = 0.0
        self.last_success_rate = 0.0

        self.set_iter(0)

    # ---- curriculum ---------------------------------------------------------
    def set_iter(self, it):
        """Curriculum schedule. Early: cube near the table centre, arm resets pre-posed, full
        grasp-lock assist. Late: cube anywhere on the table, no assist (pure friction grasp)."""
        t = min(max(it / 1500.0, 0.0), 1.0)        # 0→1 over the first 1500 iters
        sx = 0.02 + t * (C.TABLE_HALF[0] - 0.06)
        sy = 0.02 + t * (C.TABLE_HALF[1] - 0.06)
        self.spawn_half = torch.tensor([sx, sy], device=self.device)
        self.assist_frac = float(max(0.0, 1.0 - it / 1000.0))   # full assist → 0 over 1000 iters
        self.cur_iter = it

    # ---- observation --------------------------------------------------------
    def _grasp_state(self):
        """Return (lateral_xy, vert, dist, grasped, held) tensors describing the tip↔cube relation."""
        tip, _ = self.sim.tip_state()
        cube = self.sim.cube_position()
        d = cube - tip
        lateral = torch.linalg.norm(d[:, 0:2], dim=1)
        vert = d[:, 2].abs()
        dist = torch.linalg.norm(d, dim=1)
        opening = self.sim.finger_opening()
        grasped = (lateral < GRASP_LAT) & (vert < GRASP_VERT) & (opening < GRASP_OPEN_MAX)
        held = dist < HELD_RADIUS
        return tip, cube, lateral, vert, dist, opening, grasped, held

    def _obs(self):
        s = self.sim
        tip, cube, lateral, vert, dist, opening, grasped, held = self._grasp_state()
        # tip relative to this env's table centre (strip the per-env X offset)
        tip_rel = tip.clone()
        tip_rel[:, 0] -= self.env_x + C.TABLE_CX
        tip_rel[:, 1] -= C.TABLE_CY
        tip_rel[:, 2] -= C.TABLE_TOP_Z
        tip_to_cube = cube - tip                       # env_x cancels
        lift_frac = ((cube[:, 2] - C.CUBE_REST_Z) / C.LIFT_SUCCESS_DZ).clamp(0.0, 1.5)
        flags = torch.stack([grasped.float(), lift_frac], dim=1)
        finger_vel = 0.5 * (s.arm_jv[:, 7] + s.arm_jv[:, 8])
        obs = torch.cat([
            s.arm_jp[:, 0:7] - self.default_q,         # 7 arm qpos_rel
            s.arm_jv[:, 0:7] * C.QVEL_SCALE,           # 7 arm qvel
            torch.stack([opening, finger_vel * C.QVEL_SCALE], dim=1),  # 2 gripper
            tip_rel,                                   # 3 tip (table-relative)
            tip_to_cube,                               # 3 tip→cube
            s.cube_up(),                               # 3 cube up-axis
            s.cube_linvel,                             # 3 cube linvel
            self.last_action,                          # 8 last action
            flags,                                     # 2 grasp flag + lift frac
        ], dim=1)
        return obs

    # ---- reset --------------------------------------------------------------
    def _reset_idx(self, idx):
        n = idx.numel()
        if n == 0:
            return
        dev = self.device
        # arm → default posture + small jitter, gripper open
        q9 = torch.zeros(n, C.N_DOF, device=dev)
        q9[:, 0:7] = self.default_q + ARM_RESET_NOISE * (torch.rand(n, 7, generator=self.g, device=dev) * 2 - 1)
        q9[:, 7] = C.GRIP_OPEN
        q9[:, 8] = C.GRIP_OPEN
        self.sim.set_arm_joint_state(idx, q9, torch.zeros(n, C.N_DOF, device=dev))

        # cube → random spot on the table within the curriculum region, resting, random yaw
        rx = (torch.rand(n, generator=self.g, device=dev) * 2 - 1) * self.spawn_half[0]
        ry = (torch.rand(n, generator=self.g, device=dev) * 2 - 1) * self.spawn_half[1]
        pos = torch.stack([self.env_x[idx] + C.TABLE_CX + rx, C.TABLE_CY + ry,
                           torch.full((n,), C.CUBE_REST_Z, device=dev)], dim=1)
        yaw = (torch.rand(n, generator=self.g, device=dev) * 2 - 1) * np.pi
        quat = torch.zeros(n, 4, device=dev)
        quat[:, 2] = torch.sin(yaw / 2)   # qz
        quat[:, 3] = torch.cos(yaw / 2)   # qw
        pose = torch.cat([quat, pos], dim=1)
        self.sim.set_cube_state(idx, pose)

        self.steps[idx] = 0
        self.last_action[idx] = 0.0
        # which of the reset envs get grasp-lock assist this episode
        self.assisted[idx] = torch.rand(n, generator=self.g, device=dev) < self.assist_frac

    def reset(self):
        self._reset_idx(torch.arange(self.K, device=self.device))
        self.sim.read()
        return self._obs()

    # ---- step ---------------------------------------------------------------
    @torch.no_grad()
    def step(self, action):
        a = action.clamp(-1.0, 1.0)
        # arm targets (add-order joints 0..6) + mirrored gripper (7,8)
        arm_tgt = torch.clamp(self.default_q + self.arm_scale * a[:, 0:7], self.arm_lo, self.arm_hi)
        grip = C.GRIP_OPEN + (a[:, 7] * 0.5 + 0.5) * (C.GRIP_CLOSE - C.GRIP_OPEN)   # a=-1 open, a=+1 close
        targets = torch.cat([arm_tgt, grip.unsqueeze(1), grip.unsqueeze(1)], dim=1)
        self.sim.set_arm_targets(targets)
        self.sim.substep(C.SUBSTEPS)

        tip, cube, lateral, vert, dist, opening, grasped, held = self._grasp_state()

        # --- grasp-lock scaffold: weld the cube to the tip when an assisted env brings the
        #     CLOSING gripper onto the cube. Trigger on the COMMANDED close + proximity (not the
        #     achieved finger gap) so the weld fires before a hard squeeze can eject the cube. ---
        near = (lateral < GRASP_LAT) & (vert < GRASP_VERT)
        weld = self.assisted & near & (a[:, 7] > 0.0)
        if weld.any():
            widx = torch.nonzero(weld, as_tuple=False).squeeze(-1)
            self.sim.set_cube_state(widx, self.sim.make_pose(tip[widx], device=self.device))
            cube = cube.clone()
            cube[widx] = tip[widx]
            dist = torch.linalg.norm(cube - tip, dim=1)
            held = dist < HELD_RADIUS

        self.steps += 1
        lift = (cube[:, 2] - C.CUBE_REST_Z).clamp(min=0.0)
        success = held & (cube[:, 2] > C.CUBE_REST_Z + C.LIFT_SUCCESS_DZ)
        # terminal fail: cube knocked below the table OR batted off the table footprint. The lateral
        # check is essential — without it a cube swept sideways never resets and its distance inflates
        # the reach metric (and starves the reward) for the rest of the episode.
        off_x = (cube[:, 0] - (self.env_x + C.TABLE_CX)).abs() > (C.TABLE_HALF[0] + LOST_MARGIN)
        off_y = (cube[:, 1] - C.TABLE_CY).abs() > (C.TABLE_HALF[1] + LOST_MARGIN)
        fell = (cube[:, 2] < (C.TABLE_TOP_Z - C.DROP_FAIL_DZ)) | off_x | off_y

        # joint-limit proximity (arm only)
        q = self.sim.arm_jp[:, 0:7]
        over = (q - 0.95 * self.arm_hi).clamp(min=0.0) + (0.95 * self.arm_lo - q).clamp(min=0.0)

        rew = (
            W_REACH * torch.exp(-K_REACH * dist)
            + W_CENTER * torch.exp(-K_CENTER * lateral)
            + W_GRASP * grasped.float()
            + W_LIFT * lift * held.float()
            + W_SUCCESS * success.float()
            - W_ACT_RATE * ((a - self.last_action) ** 2).sum(dim=1)
            - W_ACT_MAG * (a ** 2).sum(dim=1)
            - W_LIMIT * (over ** 2).sum(dim=1)
            - W_FALL * fell.float()
        )

        self.last_action = a.clone()

        timeout = self.steps >= self.max_steps
        done = timeout | fell
        is_timeout = timeout & ~fell           # fell is a true terminal (bootstrap 0)

        # metrics
        self.last_reach = float(dist.mean())
        self.last_grasp_rate = float(grasped.float().mean())
        self.last_lift = float((lift * held.float()).mean())
        self.last_success_rate = float(success.float().mean())

        term_obs = self._obs()                 # snapshot BEFORE auto-reset
        d = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if d.numel() > 0:
            self._reset_idx(d)
            self.sim.read()
        obs = self._obs()
        return obs, rew, done, term_obs, is_timeout


# --------------------------------------------------------------------------- #
#  --selftest : shapes / no-NaN / cube rests / grasp detection
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    env = KukaGraspEnv(num_envs=8, render_visuals=False)
    obs = env.reset()
    assert obs.shape == (8, C.OBS_DIM), obs.shape
    assert torch.isfinite(obs).all(), "non-finite obs after reset"
    print(f"[selftest] obs {tuple(obs.shape)} OBS_DIM={C.OBS_DIM} ACT_DIM={C.ACT_DIM}  OK")

    # cube must rest on the table shortly after reset (not fall through / fly off)
    env.reset()
    for _ in range(10):
        env.step(torch.zeros(8, C.ACT_DIM, device=env.device))   # hold the default posture
    cube_z = env.sim.cube_position()[:, 2]
    assert (cube_z > C.TABLE_TOP_Z - 0.02).all(), f"cube fell through table: {cube_z}"
    print(f"[selftest] cube rests on table after reset  z={cube_z.mean():.3f} "
          f"(rest {C.CUBE_REST_Z:.3f})  OK")

    # a few random steps — check shapes, finiteness, reward sanity
    for k in range(20):
        a = torch.rand(8, C.ACT_DIM, device=env.device) * 2 - 1
        obs, rew, done, term, tmo = env.step(a)
        assert obs.shape == (8, C.OBS_DIM)
        assert rew.shape == (8,) and done.shape == (8,)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all(), f"non-finite at step {k}"
    print(f"[selftest] 20 random steps OK  reach={env.last_reach:.3f} "
          f"grasp_rate={env.last_grasp_rate:.3f}  reward[0]={float(rew[0]):.3f}")
    print("[selftest] (the friction-grasp physics gate is kuka_grasp_friction.py)")
    print("[selftest] PASS")


if __name__ == "__main__":
    _selftest()
