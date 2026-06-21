"""Gymnasium environment for residual RL on the hexapod (Stage 2).

The open-loop CPG gait (hexapod.py) already walks; the policy outputs small
*residual* joint-target corrections on top of it and is rewarded for tracking a
commanded (forward, turn) velocity while staying upright. Learning corrections
to a working gait converges far faster than learning to walk from scratch.

  observation (34): 12 joint pos, 12 joint vel, chassis up_y, forward (x,z),
                    body velocity (x,z), yaw rate, gait phase (cos,sin), command (fwd,turn)
  action (12):      residual coxa/femur target per leg, scaled by residual_scale
  reward:           track commanded speed + yaw rate + upright + alive - effort - drift
"""
import math
import os
import sys

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # allow `import hexapod_env` to fail gracefully without gym
    gym = None

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from hexapod import Hexapod

MAX_SPEED = 0.7         # m/s at |forward command| = 1 (target the policy is pushed to reach)
MAX_YAW = 1.2           # rad/s at |turn command| = 1
RESIDUAL_SCALE = 0.28   # action [-1,1] -> joint-target residual (rad); shared with play.py
START = (0.0, 0.40, 0.0)


def sample_command(rng):
    """A (forward, turn) command biased toward fast straight-line walking — mostly
    forward, often perfectly straight, sometimes turning."""
    f = float(rng.uniform(0.3, 1.0))
    t = float(rng.uniform(-0.6, 0.6)) if rng.random() < 0.5 else 0.0
    return np.array([f, t], np.float32)


def read_state(spider):
    """Read a robot's state in 3 batched pybind calls (vs ~26 per-property ones):
    joint positions/velocities + the root pose. Returns
    (jpos[12], jvel[12], up_y, fwd_x, fwd_z, yaw, pos_x, pos_z) with forward unit."""
    jp = spider.art.joint_positions()
    jv = spider.art.joint_velocities()
    px, _, pz, qx, qy, qz, qw = spider.art.root_state()
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)            # forward = q * (1,0,0)
    fz = 2.0 * (qx * qz - qw * qy)
    fl = math.hypot(fx, fz) or 1.0
    up = 1.0 - 2.0 * (qx * qx + qz * qz)            # up.y = (q * (0,1,0)).y
    yaw = math.atan2(2.0 * (qw * qy + qx * qz), 1.0 - 2.0 * (qy * qy + qz * qz))
    return jp, jv, up, fx / fl, fz / fl, yaw, float(px), float(pz)


def assemble_obs(jp, jv, up, fx, fz, vel_xz, yawrate, psi, command):
    tail = np.array([up, fx, fz, vel_xz[0], vel_xz[1], yawrate,
                     math.cos(psi), math.sin(psi), command[0], command[1]], np.float32)
    return np.concatenate([jp, jv, tail]).astype(np.float32)


def reward_terms(up, fx, fz, vel_xz, yawrate, command, action):
    v_fwd = vel_xz[0] * fx + vel_xz[1] * fz
    v_lat = vel_xz[0] * fz - vel_xz[1] * fx
    tgt_v = float(command[0]) * MAX_SPEED
    tgt_w = float(command[1]) * MAX_YAW
    # Track the commanded forward speed (MAX_SPEED is set high, so this pulls toward
    # a faster walk than the gait's default) and the commanded yaw rate.
    r = (2.2 * math.exp(-3.0 * (v_fwd - tgt_v) ** 2)
         + 0.8 * math.exp(-2.0 * (yawrate - tgt_w) ** 2)   # track turn
         + 0.6 * max(0.0, up)                              # stay/get upright -> push recovery
         + 0.15                                            # alive (reward not terminating)
         - 0.5 * abs(v_lat)                                # straight line: punish sideways drift
         - 0.04 * float(np.mean(np.square(action))))       # effort
    # Only terminate once truly toppled (past ~horizontal), so the policy has room to
    # catch a stumble and recover rather than ending the episode at the first lean.
    return r, up < 0.0


def make_observation(spider, vel_xz, yawrate, command):
    """The 34-d observation, shared by the env and the play script so they match."""
    jp, jv, up, fx, fz, _, _, _ = read_state(spider)
    return assemble_obs(jp, jv, up, fx, fz, vel_xz, yawrate, spider.psi, command)


def compute_reward(spider, vel_xz, yawrate, command, action):
    """Reward + fell flag (single-env path; the vec env calls reward_terms directly)."""
    _, _, up, fx, fz, _, _, _ = read_state(spider)
    return reward_terms(up, fx, fz, vel_xz, yawrate, command, action)


_Base = gym.Env if gym is not None else object


class HexapodEnv(_Base):
    metadata = {"render_modes": []}

    def __init__(self, control_hz=30, episode_s=8.0, residual_scale=RESIDUAL_SCALE,
                 command_hold_s=2.5, perturb=True, push_prob=0.015, push_mag=(3.0, 11.0)):
        assert gym is not None, "gymnasium is required for HexapodEnv (pip install gymnasium)"
        super().__init__()
        self.control_dt = 1.0 / control_hz
        self.max_steps = int(episode_s * control_hz)
        self.command_hold = int(command_hold_s * control_hz)
        self.residual_scale = residual_scale
        self.perturb = perturb
        self.push_prob = push_prob
        self.push_mag = push_mag

        self.world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0))
        ground = tp.Mesh(tp.BoxGeometry(200, 1, 200), tp.MeshStandardMaterial())
        ground.position.y = -0.5
        self.world.add_static(ground)
        self.spider = Hexapod(self.world, position=START)

        self.observation_space = spaces.Box(-np.inf, np.inf, (34,), np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, (12,), np.float32)

        self._steps = 0
        self.command = np.zeros(2, np.float32)
        self._prev = (0.0, 0.0)
        self._prev_yaw = 0.0
        self._vel = np.zeros(2, np.float32)
        self._yawrate = 0.0

    # --- helpers ---------------------------------------------------------
    def _sample_command(self):
        return sample_command(self.np_random)

    def _obs(self):
        return make_observation(self.spider, self._vel, self._yawrate, self.command)

    def _measure(self):
        p = self.spider.position
        v = np.array([(p.x - self._prev[0]) / self.control_dt,
                      (p.z - self._prev[1]) / self.control_dt], np.float32)
        self._vel = 0.8 * self._vel + 0.2 * v  # EMA: damp gait-induced bob
        yaw = self.spider.yaw
        dy = yaw - self._prev_yaw
        dy = (dy + math.pi) % (2 * math.pi) - math.pi  # unwrap
        self._yawrate = 0.8 * self._yawrate + 0.2 * (dy / self.control_dt)
        self._prev = (p.x, p.z)
        self._prev_yaw = yaw

    # --- gym API ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.spider.reset(START)
        self.spider.set_command(0.0, 0.0)
        for _ in range(8):  # settle into a stand
            self.spider.update(self.control_dt, None)
            self.world.step(self.control_dt)
        self.command = self._sample_command()
        self._steps = 0
        p = self.spider.position
        self._prev = (p.x, p.z)
        self._prev_yaw = self.spider.yaw
        self._vel[:] = 0.0
        self._yawrate = 0.0
        return self._obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        residuals = (action * self.residual_scale).tolist()
        self.spider.set_command(float(self.command[0]), float(self.command[1]))
        self.spider.update(self.control_dt, residuals)
        # Random horizontal shove — the open-loop gait can't correct for it, so this
        # is where the policy earns its keep (push recovery / heading hold).
        if self.perturb and self.np_random.random() < self.push_prob:
            mag = float(self.np_random.uniform(*self.push_mag))
            ang = float(self.np_random.uniform(0.0, 2 * math.pi))
            self.spider.chassis.add_impulse(tp.Vector3(mag * math.cos(ang), 0.0, mag * math.sin(ang)))
        self.world.step(self.control_dt)
        self._measure()
        self._steps += 1
        if self._steps % self.command_hold == 0:
            self.command = self._sample_command()  # track changing commands

        r, fell = compute_reward(self.spider, self._vel, self._yawrate, self.command, action)
        terminated = bool(fell)
        truncated = self._steps >= self.max_steps
        if fell:
            r -= 5.0
        return self._obs(), float(r), terminated, truncated, {}
