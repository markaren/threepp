"""Spot stair-climbing on GpuSim — foundation: the frozen Isaac walker, batched on the GPU.

Step 1 (this file, validated by the selftest): run K Spots in ONE PhysX direct-GPU scene
(threepp.rl.GpuSim) driven by the frozen Isaac walker (TorchScript MLP 48->12) executed
*batched on CUDA* — the 48-d Isaac observation is assembled as torch tensor ops on the GpuSim
GPU state (Isaac's quat_rotate_inverse for body-frame base velocity + projected gravity, the
isaac<->add joint permutation as a tensor index). No CPU per-robot loop. If K Spots walk
forward on flat ground here, the GPU substrate is proven and a learned residual + stair terrain
(step 2+) plugs straight in.

Z-up world (gravity 0,0,-9.81) to match the Isaac contract.
"""
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from threepp.rl import GpuSim
from spot_deploy import (build_spot, fetch_assets, default_q, add_to_isaac, isaac_to_add,
                         ACTION_SCALE, Z0)

CONTROL_HZ = 50            # Isaac Spot runs the policy at 50 Hz (dt = 0.02 s)
DT = 1.0 / CONTROL_HZ
SUBSTEPS = 4              # GPU physics substeps per control tick (Isaac used 10x0.002; 4 is the lever)
SPACING = 3.0            # metres between Spots (lateral, along +Y) so they don't inter-collide
SETTLE = 80              # ticks holding the default stance to fold from zero-config into a stand


class SpotGpu:
    """build_robot factory for GpuSim: one Spot at a per-env lateral offset. Exposes `.art`.
    `gains` overrides the PD gains (the from-scratch gait uses a stiffer set than the Isaac default).
    `foot_friction` (if given) creates a per-env grippy restitution-0 foot material (eMIN so the foot's
    friction governs the contact) — exposes `.foot_mat` for per-env friction domain randomization."""
    def __init__(self, world, i, spacing=SPACING, gains=None, foot_friction=None):
        self.foot_mat = None
        if foot_friction is not None:
            self.foot_mat = world.create_material(static_friction=float(foot_friction),
                                                  dynamic_friction=float(foot_friction) * 0.9,
                                                  restitution=0.0, friction_combine="min",
                                                  restitution_combine="min")
        self.art, _ = build_spot(world, assets=None, base_xy=(0.0, i * spacing),
                                 gains=gains, foot_material=self.foot_mat)


def quat_rotate_inverse(q, v):
    """Rotate world-frame vectors v [N,3] into the body frame given body->world quat q [N,4]
    (qx,qy,qz,qw). Isaac Lab's exact formula (the policy was trained with it)."""
    qw = q[:, 3]
    qvec = q[:, :3]
    a = v * (2.0 * qw ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(qvec, v, dim=-1) * qw.unsqueeze(-1) * 2.0
    c = qvec * (qvec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


class SpotWalker:
    """The frozen Isaac walker, batched on CUDA over all K Spots in a GpuSim."""
    def __init__(self, sim, assets):
        self.sim = sim
        dev = sim.device
        self.policy = torch.jit.load(os.path.join(assets, "spot_policy.pt"), map_location=dev).eval()
        self.default_q = torch.from_numpy(default_q).to(dev)                       # [12] isaac order
        self.default_q_add = self.default_q[torch.from_numpy(add_to_isaac).to(dev)]  # [12] add order
        self.i2a = torch.from_numpy(isaac_to_add.astype(np.int64)).to(dev)         # add -> isaac index
        self.a2i = torch.from_numpy(add_to_isaac.astype(np.int64)).to(dev)         # isaac -> add index
        self.grav = torch.tensor([0.0, 0.0, -1.0], device=dev)
        self.last_action = torch.zeros(sim.K, 12, device=dev)

    def obs(self, cmd):
        s = self.sim
        q = s.root_quat
        lin_b = quat_rotate_inverse(q, s.root_linvel)
        ang_b = quat_rotate_inverse(q, s.root_angvel)
        proj_g = quat_rotate_inverse(q, self.grav.expand(s.K, 3))
        jp_isaac = s.joint_pos[:, self.i2a]
        jv_isaac = s.joint_vel[:, self.i2a]
        qpos = jp_isaac - self.default_q
        return torch.cat([lin_b, ang_b, proj_g, cmd, qpos, jv_isaac, self.last_action], dim=1)  # [K,48]

    @torch.no_grad()
    def targets(self, cmd):
        a = self.policy(self.obs(cmd))                       # [K,12] isaac-order action
        self.last_action = a
        targets_isaac = self.default_q + ACTION_SCALE * a    # [K,12] isaac
        return targets_isaac[:, self.a2i]                    # [K,12] add-order drive targets


def _flat_ground(world, k, spacing, material=None):
    g = tp.Mesh(tp.BoxGeometry(60, spacing * k + 20, 1.0), tp.MeshStandardMaterial())
    g.position.set(20.0, spacing * (k - 1) * 0.5, -0.5)      # top at z=0, spans the lane grid + forward travel
    world.add_static(g, material=material)                   # grippy restitution-0 ground when a material is given


STAIR_X0, STAIR_RUN, STAIR_N = 1.6, 0.32, 10     # stairs start 1.6 m ahead; default tread depth; step count


def _add_stairs(world, k, spacing, rises, runs=None, x0=STAIR_X0, n=STAIR_N):
    """Per-lane static-box staircase ahead of each Spot. `rises[i]`/`runs[i]` = step height/tread
    for lane i (graded riser across lanes = a built-in difficulty sweep; randomized tread so the
    policy doesn't overfit one geometry). Each step is a solid box from the ground up to its tread."""
    for i in range(k):
        r = float(rises[i]); run = float(runs[i]) if runs is not None else STAIR_RUN
        for s in range(n):
            h = (s + 1) * r
            box = tp.Mesh(tp.BoxGeometry(run, spacing * 0.92, h), tp.MeshStandardMaterial())
            box.position.set(x0 + s * run + run * 0.5, i * spacing, h * 0.5)
            world.add_static(box)


def up_z(q):
    """World-z component of the body up-axis (cos of base tilt). q=[N,4] (qx,qy,qz,qw)."""
    return 1.0 - 2.0 * (q[:, 0] ** 2 + q[:, 1] ** 2)


def make_sim(k, seed=0):
    sim = GpuSim(k, lambda world, i: SpotGpu(world, i, SPACING),
                 gravity=(0.0, 0.0, -9.81), spacing=SPACING, read_root=True,
                 build_world=lambda world: _flat_ground(world, k, SPACING))
    walker = SpotWalker(sim, fetch_assets())
    return sim, walker


def settle(sim, walker):
    sim.read()
    for _ in range(SETTLE):
        sim.apply_drive_target(walker.default_q_add.expand(sim.K, -1))
        for _ in range(SUBSTEPS):
            sim.step(DT / SUBSTEPS)
    walker.last_action.zero_()


# =============================================================================
#  SpotStairEnv — residual RL on the frozen walker to climb stairs (GPU-native)
# =============================================================================
RISE_MIN, RISE_MAX = 0.02, 0.13      # per-lane step height (graded across envs = difficulty sweep)
RUN_MIN, RUN_MAX = 0.28, 0.36        # per-lane tread depth (randomized so it doesn't overfit)
FWD_CMD = 0.6                        # fixed forward velocity command fed to the walker
RESIDUAL_SCALE = 0.25               # action [-1,1] -> joint-target residual (rad) on the walker (0.35 destabilized the gait)
SPAWN_Z = 0.42                      # reset base height = natural default-pose stand (feet on ground, no drop)
STAND_Z = 0.50                      # climb reference (base height while walking on flat) — climb = z - STAND_Z
EPISODE_S = 16.0                    # long enough to climb a full flight (climbing is slow)
PROBE_DX = (-0.35, -0.15, 0.05, 0.2, 0.35, 0.5, 0.7, 0.9, 1.1)   # forward height-scan profile (m ahead of base)
OBS_DIM = 3 + 3 + 3 + 12 + 12 + 12 + 1 + len(PROBE_DX)   # = 55
ACT_DIM = 12

CONFIG = {"control_hz": CONTROL_HZ, "dt": DT, "substeps": SUBSTEPS, "spacing": SPACING,
          "rise_min": RISE_MIN, "rise_max": RISE_MAX, "run_min": RUN_MIN, "run_max": RUN_MAX,
          "fwd_cmd": FWD_CMD, "residual_scale": RESIDUAL_SCALE, "stand_z": STAND_Z,
          "episode_s": EPISODE_S, "stair_x0": STAIR_X0, "stair_n": STAIR_N,
          "probe_dx": list(PROBE_DX), "obs_dim": OBS_DIM, "act_dim": ACT_DIM}


def terrain_height(x, rise, run):
    """Stair-profile height at world-x for a lane of step `rise` and tread `run`. Vectorized
    (x/rise/run broadcast — pass [K] or [K,P])."""
    steps = torch.clamp(torch.floor((x - STAIR_X0) / run) + 1.0, 0.0, float(STAIR_N))
    return steps * rise


class SpotStairEnv:
    def __init__(self, num_envs=2048, device="cuda", seed=0):
        self.K, self.dt = num_envs, DT
        self.max_steps = int(EPISODE_S * CONTROL_HZ)
        rises = np.linspace(RISE_MIN, RISE_MAX, num_envs).astype(np.float32)
        runs = np.random.default_rng(seed).uniform(RUN_MIN, RUN_MAX, num_envs).astype(np.float32)
        self.sim = GpuSim(num_envs, lambda world, i: SpotGpu(world, i, SPACING),
                          gravity=(0.0, 0.0, -9.81), spacing=SPACING, device=device, read_root=True,
                          build_world=lambda world: (_flat_ground(world, num_envs, SPACING),
                                                      _add_stairs(world, num_envs, SPACING, rises, runs)))
        dev = self.sim.device
        self.walker = SpotWalker(self.sim, fetch_assets())
        self.rise = torch.from_numpy(rises).to(dev)                       # [K] per-lane step height
        self.run = torch.from_numpy(runs).to(dev)                         # [K] per-lane tread depth
        self.probe = torch.tensor(PROBE_DX, device=dev)                   # [P]
        self.cmd = torch.tensor([FWD_CMD, 0.0, 0.0], device=dev).expand(self.K, 3).contiguous()

        lane_y = torch.arange(self.K, device=dev, dtype=torch.float32) * SPACING
        pos = torch.zeros(self.K, 3, device=dev); pos[:, 1] = lane_y; pos[:, 2] = SPAWN_Z
        self.base_pose = GpuSim.make_root_pose(pos, quat=(0.0, 0.0, 0.0, 1.0), device=dev)
        self.stand_q = self.walker.default_q_add.expand(self.K, -1).contiguous()   # standing joints
        self.lane_y = lane_y                                              # each Spot's lane centre (for heading hold)
        self.fwd_v = torch.full((self.K,), FWD_CMD, device=dev)           # forward command into the stairs

        z = lambda *s: torch.zeros(*s, device=dev)
        self.steps = torch.zeros(self.K, dtype=torch.long, device=dev)
        self.up = z(self.K); self.prev_x = z(self.K); self.prev_z = z(self.K)
        self.last_act = z(self.K, ACT_DIM)
        self.ep_max_climb = z(self.K)        # highest step reached this episode (clean metric)
        self.ep_max_x = z(self.K)            # furthest forward x reached (forward-progress ratchet)
        self.last_climb = 0.0; self.last_fell = 0.0

    def _refresh(self):
        self.up = up_z(self.sim.root_quat)

    def _obs(self):
        s = self.sim
        q = s.root_quat
        lin_b = quat_rotate_inverse(q, s.root_linvel)
        ang_b = quat_rotate_inverse(q, s.root_angvel)
        proj_g = quat_rotate_inverse(q, self.walker.grav.expand(self.K, 3))
        qpos = s.joint_pos - self.stand_q
        x, zz = s.root_position[:, 0], s.root_position[:, 2]
        h_here = terrain_height(x, self.rise, self.run)
        base_above = (zz - h_here).unsqueeze(-1)                          # base height over local ground
        ahead = (terrain_height(x[:, None] + self.probe[None, :], self.rise[:, None], self.run[:, None])
                 - h_here[:, None])                                       # height-scan: terrain rise ahead
        return torch.cat([lin_b, ang_b, proj_g, qpos, s.joint_vel, self.last_act,
                          base_above, ahead], dim=-1)                     # [K, OBS_DIM]

    def _reset_idx(self, idx):
        n = idx.numel()
        if n == 0:
            return
        self.sim.set_root_state(idx, self.base_pose[idx])
        self.sim.set_joint_state(idx, self.stand_q[idx], torch.zeros(n, self.sim.dof, device=self.sim.device))
        self.steps[idx] = 0
        self.walker.last_action[idx] = 0.0
        self.last_act[idx] = 0.0
        self.ep_max_climb[idx] = 0.0
        self.ep_max_x[idx] = self.base_pose[idx, 4]      # spawn x (forward-progress ratchet baseline)
        self.prev_x[idx] = self.base_pose[idx, 4]
        self.prev_z[idx] = SPAWN_Z

    def reset(self):
        self._reset_idx(torch.arange(self.K, device=self.sim.device))
        self.sim.read()
        for _ in range(20):                                              # settle with DEFAULT targets (clean stand) — NOT the walker
            self.sim.apply_drive_target(self.stand_q)
            for _ in range(SUBSTEPS):
                self.sim.step(DT / SUBSTEPS)
        self.walker.last_action.zero_()
        self.prev_x = self.sim.root_position[:, 0].clone()
        self.prev_z = self.sim.root_position[:, 2].clone()
        self._refresh()
        return self._obs()

    @torch.no_grad()
    def step(self, action):
        a = action.clamp(-1.0, 1.0)
        prev_a = self.last_act
        # heading + lane hold: the Isaac walker only zeros yaw RATE, so a fixed forward command lets
        # Spot drift off the side and miss the stairs. Steer it straight at its staircase.
        q0 = self.sim.root_quat
        yaw = torch.atan2(2.0 * (q0[:, 0] * q0[:, 1] + q0[:, 2] * q0[:, 3]),
                          1.0 - 2.0 * (q0[:, 1] ** 2 + q0[:, 2] ** 2))
        y_err = self.sim.root_position[:, 1] - self.lane_y
        cmd = torch.stack([self.fwd_v,
                           torch.clamp(-1.0 * y_err, -0.4, 0.4),       # strafe back to lane centre
                           torch.clamp(-1.5 * yaw, -1.0, 1.0)], dim=1)  # turn back to face the stairs
        targets = self.walker.targets(cmd) + a * RESIDUAL_SCALE          # walker gait + learned residual
        self.sim.apply_drive_target(targets)
        for _ in range(SUBSTEPS):
            self.sim.step(DT / SUBSTEPS)
        self.steps += 1
        self._refresh()
        q = self.sim.root_quat
        x, zz = self.sim.root_position[:, 0], self.sim.root_position[:, 2]
        dx, dz = x - self.prev_x, zz - self.prev_z
        self.prev_x, self.prev_z = x.clone(), zz.clone()

        roll = 2.0 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])             # lateral tilt R[2,1] (keep level)
        ang_b = quat_rotate_inverse(q, self.sim.root_angvel)           # body-frame angular velocity
        lin_b = quat_rotate_inverse(q, self.sim.root_linvel)           # body-frame linear velocity
        th = terrain_height(x, self.rise, self.run)                    # stair height under the BASE
        base_above = zz - th                                           # base height over the local step
        climb = th                                                     # whole-body climb progress: which step the BASE is over
        arate = a - prev_a                                             #   (uncheatable by tilting; only rises when the rear legs come up too)
        self.last_act = a

        # looser tilt (climbing pitches a lot) + terrain-relative collapse (not a hard world-z threshold)
        fell = (self.up < 0.35) | (base_above < 0.18)
        # ONLY reward = base reaching a NEW (higher) step. No up-gate — climbing REQUIRES pitch, so gating
        # on uprightness fought the task. Standing/holding pays ~0; uprightness/level are penalties.
        new_high = (climb - self.ep_max_climb).clamp_min(0.0)
        new_x = (x - self.ep_max_x).clamp_min(0.0)     # furthest-forward ratchet
        rew = (30.0 * new_high                         # base advances onto a new (higher) step
               + 5.0 * new_x                           # forward-progress ratchet — standing still pays NOTHING (kills "stop before the stairs")
               + 0.02                                  # tiny alive (anti-suicide only)
               - 4.0 * roll.pow(2)                      # stay level — tipping sideways off the step
               - 0.3 * (ang_b[:, 0].pow(2) + ang_b[:, 2].pow(2))   # damp roll/yaw rate (balance; pitch ok)
               - 0.4 * lin_b[:, 1].abs()               # no lateral drift off the edge
               - 3.0 * torch.relu(0.35 - base_above)   # anti-scrape: don't drag the body up the steps
               - 0.02 * a.pow(2).mean(dim=1)           # residual energy (don't wreck the walker)
               - 0.05 * arate.pow(2).mean(dim=1)       # action rate (smooth, no jerky targets)
               - 5.0 * fell.float())

        self.ep_max_climb = torch.maximum(self.ep_max_climb, climb)     # peak step reached (clean metric, no tilt gate)
        self.ep_max_x = torch.maximum(self.ep_max_x, x)                 # furthest forward reached
        timeout = (self.steps >= self.max_steps) & ~fell
        done = fell | (self.steps >= self.max_steps)
        term_obs = self._obs()
        d = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if d.numel() > 0:
            self.last_climb = self.ep_max_climb[d].mean().item()        # mean peak climb of episodes ending now
            self._reset_idx(d)
            self.sim.read()
            self._refresh()
            obs = self._obs()
        else:
            obs = term_obs
        self.last_fell = fell.float().mean().item()
        return obs, rew, done, term_obs, timeout


def eval_stairs(k, policy_path):
    """Run the TRAINED residual on the graded staircase and report peak climb per riser height —
    compare against baseline_stairs (the blind walker) to see how much the residual added."""
    from threepp.rl import load_policy
    env = SpotStairEnv(num_envs=k)
    ac, _, _ = load_policy(policy_path, device="cuda")
    obs = env.reset()
    dev = env.sim.device
    peak = torch.zeros(k, device=dev)
    for _ in range(env.max_steps):           # one full episode (climbing is slow)
        with torch.no_grad():
            a = ac.act_mean(obs)
        obs, rew, done, term, to = env.step(a)
        climb = terrain_height(env.sim.root_position[:, 0], env.rise, env.run)   # step height under the base
        peak = torch.maximum(peak, climb)
    rises = env.rise.cpu().numpy(); pk = peak.cpu().numpy()
    print("  rise(m)  peak-climb(m)  ~steps climbed   (TRAINED residual)")
    for lo in np.arange(RISE_MIN, RISE_MAX - 1e-6, 0.02):
        m = (rises >= lo - 1e-4) & (rises < lo + 0.02 - 1e-4)
        if m.any():
            pc = pk[m].mean()
            print(f"   {lo:.02f}      {pc:.02f}          ~{pc / lo:.1f}")


def baseline_stairs(k):
    """Graded staircase (each lane a different step rise) walked by the BLIND frozen walker —
    finds how tall a step it climbs before stalling/falling (sets the curriculum start)."""
    rises = np.linspace(0.04, 0.20, k).astype(np.float32)
    sim = GpuSim(k, lambda world, i: SpotGpu(world, i, SPACING),
                 gravity=(0.0, 0.0, -9.81), spacing=SPACING, read_root=True,
                 build_world=lambda world: (_flat_ground(world, k, SPACING),
                                            _add_stairs(world, k, SPACING, rises)))
    walker = SpotWalker(sim, fetch_assets())
    settle(sim, walker)
    z0 = sim.root_position[:, 2].clone()
    cmd = torch.tensor([1.0, 0.0, 0.0], device=sim.device).expand(k, 3).contiguous()
    for _ in range(320):                       # ~6.4 s to walk into + up the stairs
        sim.apply_drive_target(walker.targets(cmd))
        for _ in range(SUBSTEPS):
            sim.step(DT / SUBSTEPS)
    climb = (sim.root_position[:, 2] - z0).cpu().numpy()
    xf = sim.root_position[:, 0].cpu().numpy()
    fell = (up_z(sim.root_quat) < 0.5).cpu().numpy()
    print("  rise(m)  climb(m)  reached_x  fell   (blind Isaac walker, no residual)")
    for lo in np.arange(0.04, 0.20, 0.02):
        m = (rises >= lo - 1e-4) & (rises < lo + 0.02 - 1e-4)
        if m.any():
            print(f"   {lo:.02f}    {climb[m].mean():+.2f}     {xf[m].mean():5.2f}    {fell[m].mean():.2f}")
    print(f"(stairs start at x={STAIR_X0} m; each step run={STAIR_RUN} m, {STAIR_N} steps)")


if __name__ == "__main__":
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    K = int(os.environ.get("K", "64"))
    if os.environ.get("ENVTEST"):
        env = SpotStairEnv(num_envs=K)
        obs = env.reset()
        print(f"obs {tuple(obs.shape)} (OBS_DIM={OBS_DIM}) finite={bool(torch.isfinite(obs).all())}")
        for _ in range(220):
            obs, rew, done, term, to = env.step(torch.zeros(K, ACT_DIM, device=env.sim.device))
            assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
        print(f"zero-residual (frozen walker only): mean climb={env.last_climb:.3f} m  "
              f"fell/step={env.last_fell:.3f}  mean rew={rew.mean().item():+.3f}")
        print("SPOT-STAIR ENV SELFTEST: PASS")
        sys.exit(0)
    if os.environ.get("EVAL"):
        eval_stairs(K, os.environ["EVAL"]); sys.exit(0)
    if os.environ.get("STAIRS"):
        baseline_stairs(K); sys.exit(0)
    sim, walker = make_sim(K)
    settle(sim, walker)
    x0 = sim.root_position[:, 0].clone()
    z_settle = sim.root_position[:, 2].mean().item()
    up_settle = up_z(sim.root_quat).mean().item()
    print(f"after settle: mean base z={z_settle:.3f}  mean up_z={up_settle:.3f}")

    cmd = torch.tensor([1.0, 0.0, 0.0], device=sim.device).expand(K, 3).contiguous()   # walk forward 1 m/s
    import time
    t0 = time.time()
    STEPS = 150
    for _ in range(STEPS):
        sim.apply_drive_target(walker.targets(cmd))
        for _ in range(SUBSTEPS):
            sim.step(DT / SUBSTEPS)
    dx = (sim.root_position[:, 0] - x0).mean().item()
    up = up_z(sim.root_quat).mean().item()
    z = sim.root_position[:, 2].mean().item()
    fell = (up_z(sim.root_quat) < 0.5).float().mean().item()
    sps = STEPS * K / (time.time() - t0)
    print(f"after {STEPS} ticks ({STEPS * DT:.1f}s) walking fwd: mean dx={dx:+.2f} m  "
          f"speed={dx / (STEPS * DT):+.2f} m/s  up_z={up:.3f}  base z={z:.3f}  fell={fell:.2f}")
    print(f"throughput ~{sps:.0f} env-steps/s (K={K})")
    ok = dx > 0.5 and up > 0.6 and fell < 0.3
    print("SPOT-ON-GPUSIM WALKER SELFTEST:", "PASS" if ok else "SUSPICIOUS")
