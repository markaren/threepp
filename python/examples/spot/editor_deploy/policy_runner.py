"""Run a policy bundle with numpy alone — no torch, no threepp.rl.

This is the deployment half of `export_bundle.py`: it reads spec.json and assembles exactly
the observation the spec declares, from whatever simulator an Adapter wraps. Two adapters ship
(see ab_check.py and spot_editor_script.py) — one for a threepp PhysxWorld articulation, one for
the editor's `articulation_from_object` handle — and neither knows anything about Spot.

FRAME-AGNOSTIC ON PURPOSE. A policy is trained in some world (Spot's is Z-up) and deployed in
another (the editor's PhysX world is Y-up). Nothing here assumes either: every term is built
from the root's rotation matrix and the world's own gravity direction, so the numbers the policy
sees are identical as long as the robot is oriented the same way WITH RESPECT TO GRAVITY. That
is the whole trick — the policy never learns "up is z", it learns "up is where gravity isn't".
"""
import json
import os

import numpy as np


def _elu(x):
    return np.where(x > 0.0, x, np.exp(np.minimum(x, 0.0)) - 1.0)


class PolicyBundle:
    """spec.json + policy.npz: the network, its normalizer, and the contract it was trained on."""

    def __init__(self, folder):
        self.folder = folder
        with open(os.path.join(folder, "spec.json"), encoding="utf-8") as f:
            self.spec = json.load(f)
        z = np.load(os.path.join(folder, self.spec["policy"]["file"]))
        n = len(self.spec["policy"]["layers"])
        self.W = [z[f"W{i}"] for i in range(n)]
        self.b = [z[f"b{i}"] for i in range(n)]
        nrm = self.spec.get("normalizer")
        if nrm is not None:
            self.mean, self.var = z["norm_mean"], z["norm_var"]
            self.eps, self.clip = float(nrm["eps"]), float(nrm["clip"])
        else:
            self.mean = self.var = None
        act = self.spec["policy"].get("activation", "elu")
        if act != "elu":
            raise ValueError(f"bundle activation '{act}' not implemented (elu only)")
        self.joints = list(self.spec["joints"])
        self.default_q = np.asarray(self.spec["default_q"], np.float32)
        self.action_scale = float(self.spec["action"]["scale"])
        self.obs_dim = int(self.spec["obs_dim"])
        self.act_dim = int(self.spec["act_dim"])
        self.dt = float(self.spec["control"]["dt"])

    def act(self, obs):
        """Deterministic action (the Gaussian's mean) for a single [obs_dim] observation."""
        x = np.asarray(obs, np.float32)
        if self.mean is not None:
            x = np.clip((x - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            x = x @ W.T + b
            if i < len(self.W) - 1:
                x = _elu(x)
        return x.astype(np.float32)

    def joint_targets(self, action):
        """Decode an action into joint targets, in the bundle's own joint order."""
        kind = self.spec["action"]["kind"]
        if kind != "joint_position_offset":
            raise ValueError(f"action kind '{kind}' not implemented")
        return (self.default_q + self.action_scale * np.asarray(action, np.float32)).astype(np.float32)


class Adapter:
    """What the runner needs from a simulator. Subclass per host; all world-frame, all SI.

    Required: joint_names, joint_positions, joint_velocities, root_position, root_rotation_matrix
    (body->world), root_linear_velocity, root_angular_velocity, gravity_direction (unit, points
    DOWN), set_drive_targets(values in joint_names order). Optional: height_at(point) for the
    terrain scan — the default is a flat ground plane through the world origin.
    """

    def height_at(self, point):
        """Ground height (along the up axis) at a world point. Flat ground unless overridden."""
        return 0.0


class PolicyController:
    """Assembles the spec's observation from an Adapter, ticks the policy, writes joint targets.

    Call `tick(dt, cmd)` every frame: it accumulates time and runs the policy at the bundle's
    control rate, holding the previous targets in between (a policy trained at 50 Hz must not be
    run at the renderer's frame rate — that is a different controller).
    """

    def __init__(self, bundle, adapter, *, warn=print):
        self.bundle, self.adapter = bundle, adapter
        self.last_action = np.zeros(bundle.act_dim, np.float32)
        self.phi = 0.0
        self.acc = 0.0
        self.ticks = 0
        # Joint mapping BY NAME, never by position: the simulator's DOF order is its own
        # business (the editor collapses fixed joints, so it rarely matches the policy's).
        sim = list(adapter.joint_names)
        missing = [j for j in bundle.joints if j not in sim]
        if missing:
            raise RuntimeError(f"the articulation has no joint named {missing} - it reports {sim}")
        self.p2s = np.array([sim.index(j) for j in bundle.joints], int)   # policy index -> sim index
        if len(sim) != len(bundle.joints):
            warn(f"[policy] articulation has {len(sim)} DOFs, policy drives {len(bundle.joints)}"
                 f" - the rest are left alone")
        self._terms = bundle.spec["obs"]
        self._scan = next((t for t in self._terms if t["term"] == "height_scan"), None)
        if self._scan is not None:
            self._gx = np.asarray(self._scan["grid_x"], np.float64)
            self._gy = np.asarray(self._scan["grid_y"], np.float64)

    # ---------------------------------------------------------------- frames
    def _basis(self):
        """(up, forward, left) unit world vectors: up against gravity, forward = body +x in the
        ground plane. Reduces to the training convention exactly when the world is Z-up."""
        a = self.adapter
        R = np.asarray(a.root_rotation_matrix, float)
        g = np.asarray(a.gravity_direction, float)
        up = -g / (np.linalg.norm(g) or 1.0)
        fwd = R[:, 0] - up * float(np.dot(R[:, 0], up))     # body +x, flattened into the plane
        n = np.linalg.norm(fwd)
        # Nose-up/down past vertical: fall back to body +z so the heading stays defined.
        if n < 1e-6:
            fwd = R[:, 2] - up * float(np.dot(R[:, 2], up))
            n = np.linalg.norm(fwd) or 1.0
        fwd = fwd / n
        return up, fwd, np.cross(up, fwd)

    def observe(self, cmd):
        a = self.adapter
        R = np.asarray(a.root_rotation_matrix, float)
        Rt = R.T
        g = np.asarray(a.gravity_direction, float)
        g = g / (np.linalg.norm(g) or 1.0)
        pos = np.asarray(a.root_position, float)
        jp = np.asarray(a.joint_positions, float)[self.p2s]
        jv = np.asarray(a.joint_velocities, float)[self.p2s]

        out = []
        for t in self._terms:
            k = t["term"]
            if k == "base_lin_vel_body":
                out.append(Rt @ np.asarray(a.root_linear_velocity, float))
            elif k == "base_ang_vel_body":
                out.append(Rt @ np.asarray(a.root_angular_velocity, float))
            elif k == "projected_gravity":
                out.append(Rt @ g)
            elif k == "command":
                out.append(np.asarray(cmd, float)[:t["dim"]])
            elif k == "joint_pos_rel_default":
                out.append(jp - self.bundle.default_q)
            elif k == "joint_vel":
                out.append(jv)
            elif k == "last_action":
                out.append(self.last_action)
            elif k == "gait_clock":
                out.append([np.sin(2 * np.pi * self.phi), np.cos(2 * np.pi * self.phi)])
            elif k == "base_height_above_ground":
                up, _, _ = self._basis()
                out.append([float(np.dot(pos, up)) - a.height_at(pos)])
            elif k == "height_scan":
                up, fwd, left = self._basis()
                h0 = a.height_at(pos)
                lo, hi = t.get("clip") or (-np.inf, np.inf)
                pts = pos[None, :] + self._gx[:, None] * fwd[None, :] + self._gy[:, None] * left[None, :]
                out.append(np.clip([a.height_at(p) - h0 for p in pts], lo, hi))
            else:
                raise ValueError(f"obs term '{k}' not implemented by this runner")
        obs = np.concatenate([np.asarray(o, np.float32).ravel() for o in out]).astype(np.float32)
        if obs.size != self.bundle.obs_dim:
            raise RuntimeError(f"assembled {obs.size} obs values, spec says {self.bundle.obs_dim}")
        return obs

    # ---------------------------------------------------------------- driving
    def hold_default(self):
        """Write the default pose — the settle before the policy takes over (clock stays at 0)."""
        self._write(self.bundle.default_q)

    def _write(self, targets_policy_order):
        full = np.asarray(self.adapter.joint_positions, np.float32).copy()
        full[self.p2s] = targets_policy_order
        self.adapter.set_drive_targets([float(v) for v in full])

    def step(self, cmd):
        """One control tick at the bundle's rate: observe -> act -> write targets."""
        obs = self.observe(cmd)
        a = self.bundle.act(obs)
        self.last_action = a
        self._write(self.bundle.joint_targets(a))
        self.ticks += 1
        # Advance the clock AFTER acting, so the next observation's phase matches training.
        self.phi = (self.phi + self.bundle.dt / self._period()) % 1.0
        return a

    def _period(self):
        t = next((t for t in self._terms if t["term"] == "gait_clock"), None)
        return float(t["period"]) if t else 1.0

    def tick(self, dt, cmd):
        """Frame-rate-independent driving: runs `step` as many times as `dt` has earned."""
        self.acc += float(dt)
        n = 0
        while self.acc >= self.bundle.dt:
            self.acc -= self.bundle.dt
            self.step(cmd)
            n += 1
            if n >= 4:                 # a long stall must not turn into a burst of control
                self.acc = 0.0
                break
        return n
