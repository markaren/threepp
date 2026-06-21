"""From-scratch hexapod locomotion (no CPG gait): the policy drives the 12 joint
targets DIRECTLY. With no fixed cadence it can, in principle, find a faster gait
than the residual-on-CPG policy — at the cost of a much longer, finickier train.

Tricks that make from-scratch legged locomotion actually converge:
  - a free-running CLOCK in the observation (cos/sin) — a rhythm to latch onto;
  - a JERK penalty (|action - prev_action|) — discourages jittery flailing;
  - reward forward velocity + an alive bonus, terminate on a fall.

Vectorized single-scene like hexapod_vec_env.py. Wrap in VecNormalize for training.
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
from hexapod_env import MAX_YAW, START, read_state, sample_command

SPACING = 2.5
CLOCK_HZ = 1.6                     # rhythm hint frequency
JOINT_SCALE = np.tile([0.7, 1.1], 6).astype(np.float32)  # coxa, femur per leg (rad)


class HexapodScratchVecEnv(VecEnv):
    def __init__(self, num_envs=64, control_hz=30, episode_s=12.0, command_hold_s=3.0,
                 num_threads=8, seed=0):
        self.control_dt = 1.0 / control_hz
        self.max_steps = int(episode_s * control_hz)
        self.command_hold = int(command_hold_s * control_hz)
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
        self.steps = np.zeros(K, np.int32)
        self.clock = np.zeros(K, np.float32)
        self.prev = np.zeros((K, 2), np.float32)
        self.prev_yaw = np.zeros(K, np.float32)
        self.vel = np.zeros((K, 2), np.float32)
        self.yawrate = np.zeros(K, np.float32)
        self.prev_act = np.zeros((K, 12), np.float32)
        self._actions = np.zeros((K, 12), np.float32)
        self.command = np.zeros((K, 2), np.float32)   # (forward, turn) — now randomized
        self.ep_ret = np.zeros(K, np.float64)
        self.ep_len = np.zeros(K, np.int32)

    # --- helpers ---------------------------------------------------------
    def _obs(self, i, st):
        jp, jv, up, fx, fz, _, _, _ = st
        tail = np.array([up, fx, fz, self.vel[i][0], self.vel[i][1], self.yawrate[i],
                         math.cos(self.clock[i]), math.sin(self.clock[i]),
                         self.command[i][0], self.command[i][1]], np.float32)
        return np.concatenate([jp, jv * 0.1, tail]).astype(np.float32)

    def _ingest(self, i, st):
        yaw, px, pz = st[5], st[6], st[7]
        v = np.array([(px - self.prev[i][0]) / self.control_dt,
                      (pz - self.prev[i][1]) / self.control_dt], np.float32)
        self.vel[i] = 0.8 * self.vel[i] + 0.2 * v
        dy = (yaw - self.prev_yaw[i] + math.pi) % (2 * math.pi) - math.pi
        self.yawrate[i] = 0.8 * self.yawrate[i] + 0.2 * (dy / self.control_dt)
        self.prev[i] = (px, pz)
        self.prev_yaw[i] = yaw

    def _reset_one(self, i):
        self.spiders[i].reset(self.bases[i])
        self.command[i] = sample_command(self.rng)
        self.steps[i] = 0
        self.ep_ret[i] = 0.0
        self.ep_len[i] = 0
        self.clock[i] = float(self.rng.uniform(0, 2 * math.pi))  # random phase
        self.prev_act[i] = 0.0
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
            self._reset_one(i)
        for _ in range(4):  # brief settle holding the stand pose
            for i in range(K):
                self.spiders[i].art.set_drive_targets(np.zeros(12, np.float32))
            self.world.step(self.control_dt)
        return np.stack([self._obs(i, read_state(self.spiders[i])) for i in range(K)])

    def step_async(self, actions):
        self._actions = np.clip(np.asarray(actions, np.float32), -1.0, 1.0)

    def step_wait(self):
        K = self.num_envs
        self.clock += CLOCK_HZ * 2.0 * math.pi * self.control_dt
        for i in range(K):
            self.spiders[i].art.set_drive_targets(self._actions[i] * JOINT_SCALE)
        self.world.step(self.control_dt)

        self.steps += 1
        rews = np.zeros(K, np.float32)
        dones = np.zeros(K, bool)
        infos = [{} for _ in range(K)]
        obs = np.empty((K, 34), np.float32)
        for i in range(K):
            st = read_state(self.spiders[i])
            self._ingest(i, st)
            if self.steps[i] % self.command_hold == 0:
                self.command[i] = sample_command(self.rng)   # track a changing command
            _, _, up, fx, fz, _, _, _ = st
            v_fwd = self.vel[i][0] * fx + self.vel[i][1] * fz
            v_lat = self.vel[i][0] * fz - self.vel[i][1] * fx
            cmd_f = float(self.command[i][0])
            tgt_w = float(self.command[i][1]) * MAX_YAW
            jerk = float(np.mean(np.square(self._actions[i] - self.prev_act[i])))
            r = (1.5 * min(max(v_fwd, -0.3), 1.0) * cmd_f     # go fast in the commanded forward amount
                 + 0.8 * math.exp(-2.0 * (self.yawrate[i] - tgt_w) ** 2)  # track turn (0 -> hold heading)
                 + 0.5 * max(0.0, up)                         # upright
                 + 0.2                                        # alive
                 - 0.3 * abs(v_lat)                           # straight: no body-frame sideways drift
                 - 0.01 * float(np.mean(np.square(self._actions[i])))  # energy
                 - 0.06 * jerk)                               # smoothness
            fell = up < 0.3
            if fell:
                r -= 2.0
            done = bool(fell or self.steps[i] >= self.max_steps)
            rews[i] = r
            dones[i] = done
            self.ep_ret[i] += r
            self.ep_len[i] += 1
            if done:
                infos[i]["episode"] = {"r": float(self.ep_ret[i]), "l": int(self.ep_len[i])}
                infos[i]["terminal_observation"] = self._obs(i, st)
                obs[i] = self._obs(i, self._reset_one(i))
            else:
                obs[i] = self._obs(i, st)
        self.prev_act = self._actions.copy()
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
