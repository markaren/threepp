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

MAX_SPEED = 0.5         # m/s at |forward command| = 1
MAX_YAW = 1.2           # rad/s at |turn command| = 1
RESIDUAL_SCALE = 0.28   # action [-1,1] -> joint-target residual (rad); shared with play.py
START = (0.0, 0.40, 0.0)


def make_observation(spider, vel_xz, yawrate, command):
    """The 34-d observation, shared by the env and the play script so they match."""
    pos, vel = spider.joint_states()
    f = spider.forward
    fl = math.hypot(f.x, f.z) or 1.0
    return np.array(
        pos + vel + [
            spider.up_y,
            f.x / fl, f.z / fl,
            float(vel_xz[0]), float(vel_xz[1]),
            float(yawrate),
            math.cos(spider.psi), math.sin(spider.psi),
            float(command[0]), float(command[1]),
        ], dtype=np.float32)


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
        f = float(self.np_random.uniform(-0.4, 1.0))
        t = float(self.np_random.uniform(-1.0, 1.0))
        return np.array([f, t], np.float32)

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

        # reward
        f = self.spider.forward
        fl = math.hypot(f.x, f.z) or 1.0
        fx, fz = f.x / fl, f.z / fl
        v_fwd = self._vel[0] * fx + self._vel[1] * fz
        v_lat = self._vel[0] * fz - self._vel[1] * fx
        tgt_v = float(self.command[0]) * MAX_SPEED
        tgt_w = float(self.command[1]) * MAX_YAW
        up = self.spider.up_y
        r = (2.0 * math.exp(-4.0 * (v_fwd - tgt_v) ** 2)
             + 1.0 * math.exp(-2.0 * (self._yawrate - tgt_w) ** 2)
             + 0.3 * max(0.0, up)
             + 0.1                                   # alive bonus
             - 0.03 * float(np.mean(action ** 2))    # effort
             - 0.2 * abs(v_lat))                     # sideways drift
        fell = up < 0.4
        terminated = bool(fell)
        truncated = self._steps >= self.max_steps
        if fell:
            r -= 5.0
        return self._obs(), float(r), terminated, truncated, {}
