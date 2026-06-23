"""Spot locomotion trained DIRECTLY on the CPU articulation — NO GpuSim->CPU sim-to-sim gap.

K Spots in ONE CPU PhysxWorld (tgs_pcm=True, the exact deploy contact model); whatever walks here
IS the deploy gait (train == deploy). ~5k env-steps/s vs GpuSim's ~60k, but: (1) no transfer loss,
and (2) the CPU TGS solver doesn't allow the coarse-GPU hop/fold exploits, so a clean walk should
emerge from a LEAN reward (velocity tracking + upright + height-band + smoothness — no foot-contact
machinery, which needs per-link velocity reads the CPU articulation doesn't expose).

Obs is identical to the GpuSim env (48-d) so the same policy net (and a GpuSim-trained warm start)
drops in. Physics steps on CPU; the policy net runs on GPU (obs->cuda, actions->cpu each step).
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, _HERE)

import threepp as tp
from spot_deploy import build_spot, default_q, add_to_isaac
from spot_walk_env import (WALK_GAINS, WALK_SPAWN_Z, FOOT_FRIC_MIN, FOOT_FRIC_MAX, GROUND_FRIC,
                           ACTION_SCALE, MAX_SPEED, MAX_LAT, MAX_YAW, P_STAND, H_LOW, H_HIGH,
                           W_CROUCH, W_HOP, OBS_NOISE, EPISODE_S, COMMAND_HOLD_S, CONTROL_HZ, DT,
                           SPACING, OBS_DIM, ACT_DIM, TARGET_H)

STAND_Q = default_q[add_to_isaac].astype(np.float32)
GRAV = np.array([0.0, 0.0, -1.0], np.float32)
CPU_SPAWN_Z = 0.50          # spawn a touch above the ~0.47 stand so feet start ABOVE ground (no penetration->tilt on teleport)
# the CPU TGS solver's rigid stand sits ~0.50-0.53 (taller than GpuSim's ~0.47); free-band the height reward there
CPU_H_LOW, CPU_H_HIGH = 0.46, 0.58

CONFIG = {"control_hz": CONTROL_HZ, "dt": DT, "action_scale": ACTION_SCALE, "max_speed": MAX_SPEED,
          "target_h": TARGET_H, "h_low": H_LOW, "h_high": H_HIGH, "spawn_z": WALK_SPAWN_Z,
          "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "backend": "cpu"}


def _quat_rot_inv(q, v):
    """World-frame v [K,3] -> body frame, given body->world quat q [K,4] (xyzw). Isaac's formula,
    identical to the GpuSim env's, so obs match exactly."""
    qw = q[:, 3:4]; qv = q[:, :3]
    return v * (2.0 * qw ** 2 - 1.0) - np.cross(qv, v) * qw * 2.0 + qv * (qv * v).sum(1, keepdims=True) * 2.0


class SpotCpuEnv:
    def __init__(self, num_envs=256, device="cuda", seed=0, forward_only=True):
        if not tp.HAS_PHYSX:
            raise RuntimeError("need a PhysX-enabled threepp build")
        self.K = num_envs
        self.device = torch.device(device)
        self.forward_only = forward_only
        self.max_steps = int(EPISODE_S * CONTROL_HZ)
        self.command_hold = int(COMMAND_HOLD_S * CONTROL_HZ)
        self.world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=DT / 6,
                                   max_substeps=10, tgs_pcm=True)
        gmat = self.world.create_material(GROUND_FRIC, GROUND_FRIC * 0.9, 0.0,
                                          friction_combine="min", restitution_combine="min")
        ground = tp.Mesh(tp.BoxGeometry(40, SPACING * num_envs + 10, 1.0), tp.MeshStandardMaterial())
        ground.position.set(15.0, SPACING * (num_envs - 1) * 0.5, -0.5)
        self.world.add_static(ground, material=gmat)
        rng = np.random.default_rng(seed)
        fr = rng.uniform(FOOT_FRIC_MIN, FOOT_FRIC_MAX, num_envs).astype(np.float32)   # per-env foot friction DR
        self.lane_y = (np.arange(num_envs) * SPACING).astype(np.float32)
        self.arts = []
        for i in range(num_envs):
            fmat = self.world.create_material(float(fr[i]), float(fr[i] * 0.9), 0.0,
                                              friction_combine="min", restitution_combine="min")
            art, _ = build_spot(self.world, base_xy=(0.0, float(self.lane_y[i])),
                                gains=WALK_GAINS, foot_material=fmat)
            self.arts.append(art)

        self._rp = np.zeros((num_envs, 7), np.float32); self._rv = np.zeros((num_envs, 6), np.float32)
        self._jp = np.zeros((num_envs, 12), np.float32); self._jv = np.zeros((num_envs, 12), np.float32)
        self.steps = np.zeros(num_envs, np.int64)
        self.cmd = np.zeros((num_envs, 3), np.float32)
        self.last_act = np.zeros((num_envs, 12), np.float32)
        self.g = np.random.default_rng(seed + 1)
        self.last_speed = 0.0; self.last_up = 0.0; self.last_fell = 0.0; self.last_height = 0.0
        self.reset()

    def _sample_cmd(self, n):
        vx = self.g.uniform(0.2, MAX_SPEED, n).astype(np.float32)
        stand = self.g.random(n) < P_STAND
        vx = np.where(stand, 0.0, vx).astype(np.float32)
        if self.forward_only:
            z = np.zeros(n, np.float32)
            return np.stack([vx, z, z], 1)
        vy = np.where(stand, 0.0, self.g.uniform(-MAX_LAT, MAX_LAT, n)).astype(np.float32)
        wz = np.where(stand, 0.0, self.g.uniform(-MAX_YAW, MAX_YAW, n)).astype(np.float32)
        return np.stack([vx, vy, wz], 1)

    def _read(self):
        for i, art in enumerate(self.arts):
            self._rp[i] = art.root_state(); self._rv[i] = art.root_velocity()
            self._jp[i] = art.joint_positions(); self._jv[i] = art.joint_velocities()

    def _reset_one(self, i):
        self.arts[i].reset(tp.Vector3(0.0, float(self.lane_y[i]), CPU_SPAWN_Z))
        self.arts[i].set_joint_positions(STAND_Q)        # place at the stand pose (reset() zeros to straight legs)
        self.arts[i].set_drive_targets(STAND_Q)
        self.steps[i] = 0; self.last_act[i] = 0.0; self.cmd[i] = self._sample_cmd(1)[0]

    def reset(self):
        for i in range(self.K):
            self._reset_one(i)
        for _ in range(40):                              # settle to a clean still stand before learning
            for art in self.arts:
                art.set_drive_targets(STAND_Q)
            self.world.step(0.02)
        self._read()
        return self._obs()

    def _obs(self):
        q = self._rp[:, 3:7]
        lin_b = _quat_rot_inv(q, self._rv[:, 0:3]); ang_b = _quat_rot_inv(q, self._rv[:, 3:6])
        proj_g = _quat_rot_inv(q, np.tile(GRAV, (self.K, 1)))
        obs = np.concatenate([lin_b, ang_b, proj_g, self.cmd, self._jp - STAND_Q, self._jv, self.last_act], 1)
        obs = obs.astype(np.float32) + np.random.randn(self.K, OBS_DIM).astype(np.float32) * OBS_NOISE
        return torch.from_numpy(obs).to(self.device)

    @torch.no_grad()
    def step(self, action):
        a = action.clamp(-1.0, 1.0).cpu().numpy().astype(np.float32)
        arate = a - self.last_act; self.last_act = a
        targets = STAND_Q + ACTION_SCALE * a
        for i, art in enumerate(self.arts):
            art.set_drive_targets(targets[i])
        self.world.step(0.02)
        self._read()
        self.steps += 1

        q = self._rp[:, 3:7]
        lin_b = _quat_rot_inv(q, self._rv[:, 0:3]); ang_b = _quat_rot_inv(q, self._rv[:, 3:6])
        proj_g = _quat_rot_inv(q, np.tile(GRAV, (self.K, 1)))
        zz = self._rp[:, 2]; up = -proj_g[:, 2]

        roll = (self.steps % self.command_hold == 0)
        if roll.any():
            idx = np.nonzero(roll)[0]; self.cmd[idx] = self._sample_cmd(len(idx))

        r_lin = 1.5 * np.exp(-((lin_b[:, 0] - self.cmd[:, 0]) ** 2 + (lin_b[:, 1] - self.cmd[:, 1]) ** 2) / 0.25)
        r_ang = 0.75 * np.exp(-((ang_b[:, 2] - self.cmd[:, 2]) ** 2) / 0.25)
        rew = (r_lin + r_ang + 0.3                                  # track + alive
               - W_CROUCH * np.maximum(CPU_H_LOW - zz, 0.0) ** 2    # height band (anti-crouch), CPU stand ~0.50-0.53
               - W_HOP * np.maximum(zz - CPU_H_HIGH, 0.0) ** 2      #            (anti-hop); free between
               - 1.5 * lin_b[:, 2] ** 2                             # don't bounce vertically
               - 0.05 * (ang_b[:, 0] ** 2 + ang_b[:, 1] ** 2)       # damp roll/pitch rate
               - 2.5 * (proj_g[:, 0] ** 2 + proj_g[:, 1] ** 2)      # stay upright
               - 0.0003 * (self._jv ** 2).sum(1)                    # joint velocity (energy)
               - 0.02 * (arate ** 2).sum(1)).astype(np.float32)     # action rate (smooth)
        fell = (up < 0.4) | (zz < 0.30)
        rew = rew - 2.0 * fell.astype(np.float32)

        timeout = (self.steps >= self.max_steps) & ~fell
        done = fell | (self.steps >= self.max_steps)
        term_obs = self._obs()
        d = np.nonzero(done)[0]
        if len(d):
            for i in d:
                self._reset_one(int(i))
            self._read()
            obs = self._obs()
        else:
            obs = term_obs

        self.last_speed = float(lin_b[:, 0].mean()); self.last_up = float(up.mean())
        self.last_fell = float(fell.mean()); self.last_height = float(zz.mean())
        return (obs,
                torch.from_numpy(rew).to(self.device),
                torch.from_numpy(done).to(self.device),
                term_obs,
                torch.from_numpy(timeout).to(self.device))


if __name__ == "__main__":
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    import time
    env = SpotCpuEnv(num_envs=int(os.environ.get("K", "64")))
    obs = env.reset()
    print("obs", tuple(obs.shape), "finite", bool(torch.isfinite(obs).all()))
    t0 = time.time(); N = 60
    for _ in range(N):
        obs, rew, done, term, to = env.step(torch.zeros(env.K, ACT_DIM, device=env.device))
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    sps = N * env.K / (time.time() - t0)
    print(f"zero-action: fwd={env.last_speed:+.2f} up={env.last_up:.2f} base_z={env.last_height:.3f} "
          f"fell={env.last_fell:.3f} | {sps:.0f} env-steps/s")
    print("SPOT-CPU ENV SELFTEST: PASS")
