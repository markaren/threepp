"""Vectorized hexapod env: K robots in ONE PhysX scene, stepped together (Stage 3).

Following the pyrl-branch pattern — one PxScene, many instances, a single
simulate()/fetchResults() per control step — all K hexapods advance in one
world.step(). The CPG gait is computed for all K at once in numpy, and per-robot
joint I/O goes through the batched articulation accessors (set_drive_targets,
joint_positions/velocities), so the Python bridge costs a handful of calls per
robot instead of ~40. Implements SB3's VecEnv directly (no multiprocessing).

  python train_vec.py --envs 64 --steps 4000000
"""
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from hexapod import Hexapod
from hexapod_env import (RESIDUAL_SCALE, START, assemble_obs, read_state, reward_terms, sample_command)

SPACING = 2.5  # metres between adjacent robots — far enough to never collide


class HexapodVecEnv(VecEnv):
    def __init__(self, num_envs=64, control_hz=30, episode_s=8.0, command_hold_s=2.5,
                 perturb=True, push_prob=0.015, push_mag=(3.0, 11.0), num_threads=8, seed=0):
        self.control_dt = 1.0 / control_hz
        self.max_steps = int(episode_s * control_hz)
        self.command_hold = int(command_hold_s * control_hz)
        self.perturb = perturb
        self.push_prob = push_prob
        self.push_mag = push_mag
        self.rng = np.random.default_rng(seed)

        self.world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0), num_threads=num_threads)
        ground = tp.Mesh(tp.BoxGeometry(SPACING * num_envs + 40, 1, 60), tp.MeshStandardMaterial())
        ground.position.set(SPACING * num_envs * 0.5, -0.5, 0.0)
        self.world.add_static(ground)
        self.bases = [(i * SPACING, START[1], 0.0) for i in range(num_envs)]
        self.spiders = [Hexapod(self.world, position=b) for b in self.bases]

        super().__init__(num_envs,
                         spaces.Box(-np.inf, np.inf, (34,), np.float32),
                         spaces.Box(-1.0, 1.0, (12,), np.float32))

        K = num_envs
        self.command = np.zeros((K, 2), np.float32)
        self.steps = np.zeros(K, np.int32)
        self.prev = np.zeros((K, 2), np.float32)
        self.prev_yaw = np.zeros(K, np.float32)
        self.vel = np.zeros((K, 2), np.float32)
        self.yawrate = np.zeros(K, np.float32)
        self.psi = np.zeros(K, np.float32)
        self.ep_ret = np.zeros(K, np.float64)
        self.ep_len = np.zeros(K, np.int32)
        self._actions = np.zeros((K, 12), np.float32)

        # CPG constants, identical across robots -> drive the gait vectorised.
        legs = self.spiders[0].legs
        self.coxa_sign = np.array([l["coxa_sign"] for l in legs], np.float32)
        self.femur_sign = np.array([l["femur_sign"] for l in legs], np.float32)
        self.side_mult = np.array([1.0 if l["side"] < 0 else -1.0 for l in legs], np.float32)
        self.parity = np.array([j % 2 for j in range(len(legs))], np.float32)
        self.coxa_amp = self.spiders[0].coxa_amp
        self.lift_amp = self.spiders[0].lift_amp
        self.gait_freq = self.spiders[0].gait_freq

    # --- helpers ---------------------------------------------------------
    def _sample_command(self):
        return sample_command(self.rng)

    def _drive(self, command, actions):
        """Advance the gait for all K robots (numpy) and push their joint targets."""
        self.psi += self.gait_freq * 2.0 * math.pi * self.control_dt
        fwd, turn = command[:, 0:1], command[:, 1:2]
        phase = self.psi[:, None] + self.parity[None, :] * math.pi               # [K, 6]
        drive = np.clip(fwd - turn * self.side_mult[None, :], -1.0, 1.0)          # [K, 6]
        coxa = self.coxa_amp * drive * self.coxa_sign[None, :] * np.cos(phase)
        femur = self.femur_sign[None, :] * self.lift_amp * np.maximum(0.0, -np.sin(phase))
        t = np.empty((self.num_envs, 12), np.float32)
        t[:, 0::2] = coxa
        t[:, 1::2] = femur
        t += actions * RESIDUAL_SCALE
        for i in range(self.num_envs):
            self.spiders[i].art.set_drive_targets(t[i])

    def _ingest(self, i, st):
        yaw, px, pz = st[5], st[6], st[7]
        v = np.array([(px - self.prev[i][0]) / self.control_dt,
                      (pz - self.prev[i][1]) / self.control_dt], np.float32)
        self.vel[i] = 0.8 * self.vel[i] + 0.2 * v
        dy = (yaw - self.prev_yaw[i] + math.pi) % (2 * math.pi) - math.pi
        self.yawrate[i] = 0.8 * self.yawrate[i] + 0.2 * (dy / self.control_dt)
        self.prev[i] = (px, pz)
        self.prev_yaw[i] = yaw

    def _obs_from(self, i, st):
        jp, jv, up, fx, fz, _, _, _ = st
        return assemble_obs(jp, jv, up, fx, fz, self.vel[i], self.yawrate[i], self.psi[i], self.command[i])

    def _teleport_reset(self, i):
        self.spiders[i].reset(self.bases[i])
        self.command[i] = self._sample_command()
        self.steps[i] = 0
        self.ep_ret[i] = 0.0
        self.ep_len[i] = 0
        self.psi[i] = 0.0
        self.vel[i] = 0.0
        self.yawrate[i] = 0.0
        st = read_state(self.spiders[i])
        self.prev[i] = (st[6], st[7])
        self.prev_yaw[i] = st[5]
        return st

    # --- VecEnv API ------------------------------------------------------
    def reset(self):
        K = self.num_envs
        for i in range(K):
            self.spiders[i].reset(self.bases[i])
            self.command[i] = self._sample_command()
            self.steps[i] = 0
            self.ep_ret[i] = 0.0
            self.ep_len[i] = 0
        self.psi[:] = 0.0
        zc, za = np.zeros((K, 2), np.float32), np.zeros((K, 12), np.float32)
        for _ in range(6):  # settle the whole batch into a stand
            self._drive(zc, za)
            self.world.step(self.control_dt)
        obs = np.empty((K, 34), np.float32)
        for i in range(K):
            st = read_state(self.spiders[i])
            self.vel[i] = 0.0
            self.yawrate[i] = 0.0
            self.prev[i] = (st[6], st[7])
            self.prev_yaw[i] = st[5]
            obs[i] = self._obs_from(i, st)
        return obs

    def step_async(self, actions):
        self._actions = np.clip(np.asarray(actions, np.float32), -1.0, 1.0)

    def step_wait(self):
        K = self.num_envs
        self._drive(self.command, self._actions)
        if self.perturb:
            for i in np.nonzero(self.rng.random(K) < self.push_prob)[0]:
                mag = float(self.rng.uniform(*self.push_mag))
                ang = float(self.rng.uniform(0.0, 2 * math.pi))
                self.spiders[i].chassis.add_impulse(tp.Vector3(mag * math.cos(ang), 0.0, mag * math.sin(ang)))
        self.world.step(self.control_dt)   # one batched step for ALL K robots

        self.steps += 1
        rews = np.zeros(K, np.float32)
        dones = np.zeros(K, bool)
        infos = [{} for _ in range(K)]
        obs = np.empty((K, 34), np.float32)
        for i in range(K):
            st = read_state(self.spiders[i])
            self._ingest(i, st)
            if self.steps[i] % self.command_hold == 0:
                self.command[i] = self._sample_command()
            _, _, up, fx, fz, _, _, _ = st
            r, fell = reward_terms(up, fx, fz, self.vel[i], self.yawrate[i], self.command[i], self._actions[i])
            if fell:
                r -= 5.0
            done = bool(fell or self.steps[i] >= self.max_steps)
            rews[i] = r
            dones[i] = done
            self.ep_ret[i] += r
            self.ep_len[i] += 1
            if done:
                infos[i]["episode"] = {"r": float(self.ep_ret[i]), "l": int(self.ep_len[i])}
                infos[i]["terminal_observation"] = self._obs_from(i, st)
                obs[i] = self._obs_from(i, self._teleport_reset(i))
            else:
                obs[i] = self._obs_from(i, st)
        return obs, rews, dones, infos

    def close(self):
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name, None) for _ in self._idx(indices)]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [None for _ in self._idx(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False for _ in self._idx(indices)]

    def _idx(self, indices):
        if indices is None:
            return range(self.num_envs)
        if isinstance(indices, int):
            return [indices]
        return indices
