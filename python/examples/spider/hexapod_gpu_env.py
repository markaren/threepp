"""GPU-vectorized residual-RL-on-CPG for the hexapod (owned stack: threepp.rl, no SB3).

K hexapods run in ONE PhysX direct-GPU scene via threepp.rl.GpuSim. The CPG tripod gait
(hexapod.py) is computed for all K at once in torch; the policy adds small per-joint *residual*
target corrections on top, and is rewarded for tracking a commanded (forward, turn) velocity
while staying upright. Obs/reward/reset and the PPO update all live on the GPU — no CPU readback,
no stable_baselines3.

This module is the SINGLE SOURCE OF TRUTH (CONFIG + make_obs), shared by the trainer and the
deploy viewer so they can never drift. Control = position PD drive targets (the hexapod's legs
are stiff position drives), so the action is a residual on the CPG's target angles.

  observation (34): 12 joint pos, 12 joint vel, chassis up_y, forward (x,z),
                    body velocity (x,z), yaw rate, gait phase (cos,sin), command (fwd,turn)
  action (12):      residual coxa/femur target per leg, scaled by RESIDUAL_SCALE
"""
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from hexapod import Hexapod
from threepp.rl import GpuSim

# ---- single source of truth: config + observation ----------------------------
CONTROL_HZ = 30
DT = 1.0 / CONTROL_HZ
EPISODE_S = 8.0
COMMAND_HOLD_S = 2.5
RESIDUAL_SCALE = 0.28    # action [-1,1] -> joint-target residual (rad)
MAX_SPEED = 0.7          # m/s at |forward cmd| = 1 (target the policy is pushed toward)
MAX_YAW = 1.2            # rad/s at |turn cmd| = 1
SPACING = 2.5            # metres between robots (no inter-robot collision)
START_Y = 0.40           # chassis spawn height
SETTLE_STEPS = 6         # steps to settle into a stand after a reset
OBS_DIM = 34
ACT_DIM = 12
_PI = math.pi

# Persisted into the policy checkpoint so deploy reconstructs + asserts the contract.
CONFIG = {"control_hz": CONTROL_HZ, "dt": DT, "episode_s": EPISODE_S,
          "residual_scale": RESIDUAL_SCALE, "max_speed": MAX_SPEED, "max_yaw": MAX_YAW,
          "start_y": START_Y, "command_hold_s": COMMAND_HOLD_S}


def make_obs(jp, jv, up, fx, fz, vx, vz, yawrate, psi, cmd):
    """The 34-d observation, defined ONCE. jp/jv: [N,12]; up/fx/fz/vx/vz/yawrate/psi: [N];
    cmd: [N,2]. Works batched (training) and for a single robot ([1,...]) at deploy."""
    tail = torch.stack([up, fx, fz, vx, vz, yawrate,
                        torch.cos(psi), torch.sin(psi), cmd[:, 0], cmd[:, 1]], dim=-1)   # [N,10]
    return torch.cat([jp, jv, tail], dim=-1)                                            # [N,34]


def quat_to_frame(q):
    """q: [...,4] = (qx,qy,qz,qw). Returns up_y, forward_x, forward_z (unit), yaw — matching
    hexapod.py's conventions. forward = q*(1,0,0), up = q*(0,1,0)."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    up = 1.0 - 2.0 * (qx * qx + qz * qz)
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fz = 2.0 * (qx * qz - qw * qy)
    fl = torch.sqrt(fx * fx + fz * fz).clamp_min(1e-6)
    yaw = torch.atan2(2.0 * (qw * qy + qx * qz), 1.0 - 2.0 * (qy * qy + qz * qz))
    return up, fx / fl, fz / fl, yaw


class HexapodGpuEnv:
    def __init__(self, num_envs=2048, device="cuda", seed=0):
        self.K, self.dt = num_envs, DT
        self.max_steps = int(EPISODE_S * CONTROL_HZ)
        self.command_hold = int(COMMAND_HOLD_S * CONTROL_HZ)
        self.sim = GpuSim(num_envs,
                          lambda world, i: Hexapod(world, position=(i * SPACING, START_Y, 0.0)),
                          spacing=SPACING, device=device, read_root=True,
                          build_world=lambda world: self._add_ground(world, num_envs))
        dev = self.sim.device
        self.g = torch.Generator(device=dev).manual_seed(seed)

        # CPG constants — read from the built robot so this follows the hexapod factory.
        legs = self.sim.robots[0].legs
        self.gait_w = self.sim.robots[0].gait_freq * 2.0 * _PI
        self.coxa_amp = self.sim.robots[0].coxa_amp
        self.lift_amp = self.sim.robots[0].lift_amp
        self.coxa_sign = torch.tensor([l["coxa_sign"] for l in legs], device=dev)        # [6]
        self.femur_sign = torch.tensor([l["femur_sign"] for l in legs], device=dev)      # [6]
        self.side_mult = torch.tensor([1.0 if l["side"] < 0 else -1.0 for l in legs], device=dev)
        self.parity = torch.tensor([j % 2 for j in range(len(legs))], dtype=torch.float32, device=dev)

        # upright base pose per env (chassis at its spawn, identity orientation) for resets.
        # PhysX root-pose layout is [qx,qy,qz,qw, px,py,pz] (quat first) — build via make_root_pose.
        pos = torch.zeros(self.K, 3, device=dev)
        pos[:, 0] = torch.arange(self.K, device=dev) * SPACING
        pos[:, 1] = START_Y
        self.base_pose = GpuSim.make_root_pose(pos, quat=(0.0, 0.0, 0.0, 1.0), device=dev)

        z = lambda *s: torch.zeros(*s, device=dev)
        self.steps = torch.zeros(self.K, dtype=torch.long, device=dev)
        self.psi = z(self.K)
        self.cmd = z(self.K, 2)
        self.prev_px, self.prev_pz, self.prev_yaw = z(self.K), z(self.K), z(self.K)
        self.ema_v = z(self.K, 2)            # EMA body velocity (world x,z)
        self.ema_w = z(self.K)               # EMA yaw rate
        self.up = z(self.K); self.fx = z(self.K); self.fz = z(self.K)

    @staticmethod
    def _add_ground(world, num_envs):
        ground = tp.Mesh(tp.BoxGeometry(SPACING * num_envs + 40, 1, 60), tp.MeshStandardMaterial())
        ground.position.set(SPACING * num_envs * 0.5, -0.5, 0.0)
        world.add_static(ground)

    # --- helpers ---------------------------------------------------------------
    def _sample_command(self, n):
        f = torch.rand(n, generator=self.g, device=self.sim.device) * 0.7 + 0.3
        t = torch.rand(n, generator=self.g, device=self.sim.device) * 1.2 - 0.6
        t = torch.where(torch.rand(n, generator=self.g, device=self.sim.device) < 0.5, t,
                        torch.zeros_like(t))
        return torch.stack([f, t], dim=-1)

    def _cpg_targets(self, action):
        """Advance the gait and return [K,12] add-order drive targets (coxa,femur per leg)
        plus the policy residual — the exact torch mirror of Hexapod.gait_targets()."""
        self.psi = self.psi + self.gait_w * self.dt
        phase = self.psi[:, None] + self.parity[None, :] * _PI                            # [K,6]
        drive = (self.cmd[:, 0:1] - self.cmd[:, 1:2] * self.side_mult[None, :]).clamp(-1.0, 1.0)
        coxa = self.coxa_amp * drive * self.coxa_sign[None, :] * torch.cos(phase)         # [K,6]
        femur = self.femur_sign[None, :] * self.lift_amp * torch.relu(-torch.sin(phase))  # [K,6]
        t = torch.empty(self.K, 12, device=self.sim.device)
        t[:, 0::2] = coxa
        t[:, 1::2] = femur
        return t + action * RESIDUAL_SCALE

    def _orientation(self):
        self.up, self.fx, self.fz, yaw = quat_to_frame(self.sim.root_quat)
        return yaw

    def _update_velocity(self, yaw):
        px, pz = self.sim.root_position[:, 0], self.sim.root_position[:, 2]
        self.ema_v[:, 0] = 0.8 * self.ema_v[:, 0] + 0.2 * (px - self.prev_px) / self.dt
        self.ema_v[:, 1] = 0.8 * self.ema_v[:, 1] + 0.2 * (pz - self.prev_pz) / self.dt
        dy = torch.remainder(yaw - self.prev_yaw + _PI, 2.0 * _PI) - _PI                 # unwrap
        self.ema_w = 0.8 * self.ema_w + 0.2 * dy / self.dt
        self.prev_px, self.prev_pz, self.prev_yaw = px.clone(), pz.clone(), yaw.clone()

    def _obs(self):
        return make_obs(self.sim.joint_pos, self.sim.joint_vel, self.up, self.fx, self.fz,
                        self.ema_v[:, 0], self.ema_v[:, 1], self.ema_w, self.psi, self.cmd)

    def _reset_idx(self, idx):
        n = idx.numel()
        if n == 0:
            return
        zj = torch.zeros(n, self.sim.dof, device=self.sim.device)
        self.sim.set_joint_state(idx, zj, zj)
        self.sim.set_root_state(idx, self.base_pose[idx])
        self.steps[idx] = 0
        self.psi[idx] = 0.0
        self.cmd[idx] = self._sample_command(n)
        # re-baseline the finite-diff velocity to the spawn so it starts at ~0 (px,pz = root_pose 4,6)
        self.prev_px[idx] = self.base_pose[idx, 4]
        self.prev_pz[idx] = self.base_pose[idx, 6]
        self.prev_yaw[idx] = 0.0
        self.ema_v[idx] = 0.0
        self.ema_w[idx] = 0.0

    # --- API -------------------------------------------------------------------
    def reset(self):
        self._reset_idx(torch.arange(self.K, device=self.sim.device))
        zero_act = torch.zeros(self.K, ACT_DIM, device=self.sim.device)
        for _ in range(SETTLE_STEPS):                       # settle the batch into a stand
            self.sim.apply_drive_target(self._cpg_targets(zero_act))
            self.sim.step(self.dt)
        self.psi[:] = 0.0
        # baseline finite-diff state to the settled pose
        self.prev_px = self.sim.root_position[:, 0].clone()
        self.prev_pz = self.sim.root_position[:, 2].clone()
        self.prev_yaw = self._orientation()
        self.ema_v[:] = 0.0
        self.ema_w[:] = 0.0
        return self._obs()

    @torch.no_grad()
    def step(self, actions):
        """Returns (next_obs, reward, done, terminal_obs, timeout). done = fell (up<0, a true
        terminal) OR timeout (step limit, a truncation). `timeout` flags which dones are
        truncations so the trainer bootstraps V(terminal) there and zeroes it on a real fall."""
        a = actions.clamp(-1.0, 1.0)
        self.sim.apply_drive_target(self._cpg_targets(a))
        self.sim.step(self.dt)
        self.steps += 1
        yaw = self._orientation()
        self._update_velocity(yaw)

        # resample the command periodically so the policy tracks changing goals
        roll = (self.steps % self.command_hold == 0)
        ridx = torch.nonzero(roll, as_tuple=False).squeeze(-1)
        if ridx.numel() > 0:
            self.cmd[ridx] = self._sample_command(ridx.numel())

        v_fwd = self.ema_v[:, 0] * self.fx + self.ema_v[:, 1] * self.fz
        v_lat = self.ema_v[:, 0] * self.fz - self.ema_v[:, 1] * self.fx
        tgt_v = self.cmd[:, 0] * MAX_SPEED
        tgt_w = self.cmd[:, 1] * MAX_YAW
        rew = (2.2 * torch.exp(-3.0 * (v_fwd - tgt_v) ** 2)
               + 0.8 * torch.exp(-2.0 * (self.ema_w - tgt_w) ** 2)
               + 0.6 * self.up.clamp_min(0.0)
               + 0.15
               - 0.5 * v_lat.abs()
               - 0.04 * a.pow(2).mean(dim=1))
        fell = self.up < 0.0
        rew = rew - 5.0 * fell.float()

        timeout = (self.steps >= self.max_steps) & ~fell
        done = fell | (self.steps >= self.max_steps)
        term_obs = self._obs()
        d = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if d.numel() > 0:
            self._reset_idx(d)
            self.sim.read()
            self._orientation()          # refresh up/fx/fz for the reset rows (no velocity update)
            obs = self._obs()
        else:
            obs = term_obs
        return obs, rew, done, term_obs, timeout


if __name__ == "__main__":
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    env = HexapodGpuEnv(num_envs=128)
    obs = env.reset()
    print("obs", tuple(obs.shape), "finite", bool(torch.isfinite(obs).all()))
    env.cmd[:] = torch.tensor([1.0, 0.0], device=env.sim.device)   # all walk forward
    for _ in range(120):
        obs, rew, done, term, to = env.step(torch.zeros(env.K, ACT_DIM, device=env.sim.device))
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    up = env.up.mean().item()
    vfwd = (env.ema_v[:, 0] * env.fx + env.ema_v[:, 1] * env.fz).mean().item()
    print(f"open-loop CPG (cmd=1,0): mean up_y={up:.2f}, mean forward speed={vfwd:.2f} m/s")
    print("HEXAPOD GPU ENV SELFTEST: PASS")
