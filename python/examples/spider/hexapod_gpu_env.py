"""GPU-vectorized residual-RL-on-CPG for the hexapod (owned stack: threepp.rl, no SB3).

K hexapods run in ONE PhysX direct-GPU scene. The choreography — CUDA context, steps/timeout
bookkeeping, terminal-obs capture, partial resets, EMA-velocity re-baselining — lives in
threepp.rl (GpuSim + VecTask); this file is ONLY the task. The CPG tripod gait (hexapod.py) is
computed for all K at once in torch; the policy adds small per-joint *residual* target
corrections on top, and is rewarded for tracking a commanded (forward, turn) velocity while
staying upright.

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
from threepp.rl import GpuSim, VecTask
from threepp.rl import quat_to_frame  # noqa: F401  (re-export: play.py imports it from here)

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


class HexapodGpuEnv(VecTask):
    control_hz = CONTROL_HZ
    episode_s = EPISODE_S
    act_dim = ACT_DIM
    control = "drive"                 # stiff position PD drives -> targets, not forces
    settle_steps = SETTLE_STEPS

    def __init__(self, num_envs=2048, device="cuda", seed=0):
        super().__init__(num_envs,
                         lambda world, i: Hexapod(world, position=(i * SPACING, START_Y, 0.0)),
                         spacing=SPACING, device=device, seed=seed, read_root=True,
                         build_world=lambda world: self._add_ground(world, num_envs))
        dev = self.device
        self.command_hold = int(COMMAND_HOLD_S * CONTROL_HZ)

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

        # per-episode state: registered so the base auto-zeroes them on every reset
        self.psi = self.env_state(())        # gait phase clock
        self.cmd = self.env_state((2,))      # commanded (forward, turn), sampled in on_reset

    @staticmethod
    def _add_ground(world, num_envs):
        ground = tp.Mesh(tp.BoxGeometry(SPACING * num_envs + 40, 1, 60), tp.MeshStandardMaterial())
        ground.position.set(SPACING * num_envs * 0.5, -0.5, 0.0)
        world.add_static(ground)

    def _sample_command(self, n):
        f = torch.rand(n, generator=self.g, device=self.device) * 0.7 + 0.3
        t = torch.rand(n, generator=self.g, device=self.device) * 1.2 - 0.6
        t = torch.where(torch.rand(n, generator=self.g, device=self.device) < 0.5, t,
                        torch.zeros_like(t))
        return torch.stack([f, t], dim=-1)

    # ---- the task ---------------------------------------------------------------
    def on_reset(self, idx):
        n = idx.numel()
        zj = torch.zeros(n, self.sim.dof, device=self.device)
        self.sim.set_joint_state(idx, zj, zj)
        self.sim.set_root_state(idx, self.base_pose[idx])
        self.cmd[idx] = self._sample_command(n)
        # psi and the EMA-velocity baselines are re-zeroed by the base (env_state + rebaseline)

    def on_settled(self):
        self.psi.zero_()   # the settle steps advanced the gait clock; restart it for the episode

    def act(self, action):
        """Advance the gait and return [K,12] add-order drive targets (coxa,femur per leg)
        plus the policy residual — the exact torch mirror of Hexapod.gait_targets()."""
        self.psi += self.gait_w * self.dt
        phase = self.psi[:, None] + self.parity[None, :] * _PI                            # [K,6]
        drive = (self.cmd[:, 0:1] - self.cmd[:, 1:2] * self.side_mult[None, :]).clamp(-1.0, 1.0)
        coxa = self.coxa_amp * drive * self.coxa_sign[None, :] * torch.cos(phase)         # [K,6]
        femur = self.femur_sign[None, :] * self.lift_amp * torch.relu(-torch.sin(phase))  # [K,6]
        t = torch.empty(self.K, 12, device=self.device)
        t[:, 0::2] = coxa
        t[:, 1::2] = femur
        return t + action * RESIDUAL_SCALE

    def on_step(self, s):
        # resample the command periodically so the policy tracks changing goals
        roll = (self.steps % self.command_hold == 0)
        ridx = torch.nonzero(roll, as_tuple=False).squeeze(-1)
        if ridx.numel() > 0:
            self.cmd[ridx] = self._sample_command(ridx.numel())

    def observe(self, s):
        return make_obs(s.joint_pos, s.joint_vel, s.up, s.fwd_x, s.fwd_z,
                        s.v_x, s.v_z, s.yaw_rate, self.psi, self.cmd)

    def terminated(self, s):
        return s.up < 0.0                    # fell over: a true terminal (bootstraps V=0)

    def reward_terms(self, s, a):
        tgt_v = self.cmd[:, 0] * MAX_SPEED
        tgt_w = self.cmd[:, 1] * MAX_YAW
        return {
            "track_v": 2.2 * torch.exp(-3.0 * (s.v_fwd - tgt_v) ** 2),
            "track_w": 0.8 * torch.exp(-2.0 * (s.yaw_rate - tgt_w) ** 2),
            "upright": 0.6 * s.up.clamp_min(0.0),
            "alive":   torch.full((self.K,), 0.15, device=self.device),
            "lateral": -0.5 * s.v_lat.abs(),
            "effort":  -0.04 * a.pow(2).mean(dim=1),
            "fell":    -5.0 * s.terminated.float(),
        }

    def config(self):
        return {**super().config(), **CONFIG}


if __name__ == "__main__":
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    env = HexapodGpuEnv(num_envs=128)
    obs = env.reset()
    print("obs", tuple(obs.shape), "finite", bool(torch.isfinite(obs).all()))
    env.cmd[:] = torch.tensor([1.0, 0.0], device=env.device)   # all walk forward
    for _ in range(120):
        obs, rew, done, term, to = env.step(torch.zeros(env.K, ACT_DIM, device=env.device))
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    up = env.state.up.mean().item()
    vfwd = env.state.v_fwd.mean().item()
    print(f"open-loop CPG (cmd=1,0): mean up_y={up:.2f}, mean forward speed={vfwd:.2f} m/s")
    assert up > 0.9, "open-loop CPG should (mostly) stand upright"
    assert vfwd > 0.2, "open-loop CPG should walk forward"
    print("HEXAPOD GPU ENV SELFTEST: PASS")
