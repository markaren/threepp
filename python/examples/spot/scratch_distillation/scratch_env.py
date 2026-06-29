"""SpotScratchEnv — flat-ground from-scratch teacher-guided phase-RL environment.

Edit-down of SpotTerrainEnv (spot_terrain_env.py). Differences vs. the terrain env:
  - Flat ground ONLY (no tent geometry, no 2-D scan, no base_above in obs).
  - GpuSim(read_root=True, read_links=True) for foot kinematics.
  - OBS_DIM=50: 48 Isaac proprio + 2 clock dims.  Teacher reads raw _last_obs[:,:48].
  - Phase clock self.phi [K], randomised on reset, advanced after each substep.
  - Per-env foot-friction DR at build time (world.create_material per env).
  - Stiffer PD gains (~90) for a rigid from-scratch stance.
  - Iteration-driven schedules for W_IMIT/W_TICK and the command envelope.
  - Siekmann tick reward (optional, gated by tick_enabled flag and cmd_gate).
  - Richer logging metrics: last_track, last_fell, last_imit_div, last_tick,
    last_drift, last_duty (per leg mean contact per step).

obs layout  (OBS_DIM=50, named constants from scratch_clock):
  [0:3]   lin_b        body-frame linear velocity
  [3:6]   ang_b        body-frame angular velocity
  [6:9]   proj_g       gravity direction in body frame
  [9:12]  cmd          [vx, vy, wz]
  [12:24] qpos_rel     joint positions minus default_q (Isaac order)
  [24:36] qvel         joint velocities (Isaac order)
  [36:48] last_action  (Isaac order)
  [48:50] clock        [sin(2π·phi), cos(2π·phi)]   <- CLOCK0=48

action (12): Isaac-order joint targets = default_q + ACTION_SCALE * a  (unclamped).
HIDDEN = (512, 256, 128) — matches the Isaac actor for the teacher query.
"""
import math
import os
import sys

import numpy as np
import torch

_HERE      = os.path.dirname(os.path.abspath(__file__))   # scratch_distillation/
_SPOT_DIR  = os.path.dirname(_HERE)                        # examples/spot/
_EXAMPLES  = os.path.dirname(_SPOT_DIR)                   # examples/
_PYROOT    = os.path.dirname(_EXAMPLES)                   # python/
sys.path.insert(0, _PYROOT)    # threepp / threepp.rl
sys.path.insert(0, _SPOT_DIR)  # spot_deploy / spot_terrain_env

import threepp as tp
from threepp.rl import GpuSim
from spot_deploy import (build_spot, fetch_assets, default_q, add_to_isaac, isaac_to_add,
                         ACTION_SCALE, GAINS)
from spot_terrain_env import (quat_rotate_inverse, up_z, heading_cossin, _flat_ground,
                               CONTROL_HZ, DT, SUBSTEPS, SPACING, SPAWN_Z)
from scratch_clock import (OBS_DIM, CLOCK0, ACT_DIM, N_PROPRIO, GAIT_PERIOD,
                            advance, clock_obs, leg_phase, desired_stance, reset_phi, DUTY)
from foot_contact import foot_world, contact_soft, FEET, H_C

# --------------------------------------------------------------------------- #
#  Constants
# --------------------------------------------------------------------------- #
HIDDEN      = (512, 256, 128)
EPISODE_S   = 20.0
SIG         = 0.25          # exp-kernel half-width (same as IsaacLab / terrain env)
STAND_PROB  = 0.12          # applied only at full-envelope iter >= 800

# Stiffer PD gains for rigid from-scratch stance:
#   spot_deploy.GAINS uses stiffness 60 — calibrated for the warm-started gait that runs
#   lightly loaded. The from-scratch student starts from zero and needs a stiff "skeleton"
#   to stand upright at all before the imit signal teaches movement. 60 sags ~12-34 deg.
#   Raising to 90 gives a much stiffer rest pose without changing the action contract.
STIFF_GAINS = {
    "hx": (90.0, 1.5, GAINS["hx"][2]),
    "hy": (90.0, 1.5, GAINS["hy"][2]),
    "kn": (90.0, 1.5, GAINS["kn"][2]),
}

CONFIG = {
    "control_hz": CONTROL_HZ, "dt": DT, "substeps": SUBSTEPS, "spacing": SPACING,
    "episode_s": EPISODE_S, "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "hidden": list(HIDDEN),
    "gait_period": GAIT_PERIOD, "duty": DUTY, "sig": SIG,
    "stiff_gains": {k: list(v) for k, v in STIFF_GAINS.items()},
}


# --------------------------------------------------------------------------- #
#  Per-env robot factory (new; NOT SpotGpu — must pass material + gains)
# --------------------------------------------------------------------------- #
class _SpotScratch:
    """build_robot factory for GpuSim. Draws friction from a pre-seeded mu array."""
    def __init__(self, world, i, mu_arr):
        mu = float(mu_arr[i])
        mat = world.create_material(mu, mu, 0.0)   # static_friction, dynamic_friction, restitution
        self.art, _ = build_spot(world, assets=None, base_xy=(0.0, i * SPACING),
                                 gains=STIFF_GAINS, foot_material=mat)


# --------------------------------------------------------------------------- #
#  SpotScratchEnv
# --------------------------------------------------------------------------- #
class SpotScratchEnv:
    """Flat-ground from-scratch phase-RL environment for Spot.

    Observation: OBS_DIM=50 (48 Isaac proprio + 2 clock dims).
    Teacher reads raw _last_obs[:, :48] — NEVER the clock block.

    Schedules are iteration-driven. Call env.set_iter(it) each PPO iteration.

    Args:
        num_envs    : number of parallel environments
        device      : torch device ('cuda')
        seed        : RNG seed (deterministic friction DR)
        tick_enabled: if False W_TICK stays 0 always (obs-clock-only baseline)
    """
    def __init__(self, num_envs=2048, device="cuda", seed=0, tick_enabled=True):
        self.K = num_envs
        self.dt = DT
        self.tick_enabled = tick_enabled
        self.max_steps = int(EPISODE_S * CONTROL_HZ)

        # Pre-generate per-env friction values deterministically (build-time DR).
        # Range U(0.6, 1.2): gentle enough that the low end doesn't destabilize early
        # standing, but provides meaningful inter-env variety from iter 0.
        rng = np.random.default_rng(seed)
        mu_arr = rng.uniform(0.6, 1.2, num_envs).astype(np.float32)

        # Build sim: flat ground + per-env Spot with friction DR + stiffer gains.
        self.sim = GpuSim(
            num_envs,
            lambda world, i: _SpotScratch(world, i, mu_arr),
            gravity=(0.0, 0.0, -9.81),
            spacing=SPACING,
            device=device,
            read_root=True,
            read_links=True,
            build_world=lambda world: _flat_ground(world, num_envs, SPACING),
        )
        dev = self.sim.device

        self.default_q   = torch.from_numpy(default_q).to(dev)                  # [12] isaac order
        self.i2a         = torch.from_numpy(isaac_to_add.astype(np.int64)).to(dev)
        self.a2i         = torch.from_numpy(add_to_isaac.astype(np.int64)).to(dev)
        self.stand_q_add = self.default_q[self.a2i].expand(num_envs, -1).contiguous()
        self.grav        = torch.tensor([0.0, 0.0, -1.0], device=dev)

        # Lane spawn positions (faces +x, aligned with lane offset).
        lane_y = torch.arange(num_envs, device=dev, dtype=torch.float32) * SPACING
        pos = torch.zeros(num_envs, 3, device=dev)
        pos[:, 1] = lane_y
        pos[:, 2] = SPAWN_Z
        self.base_pose = GpuSim.make_root_pose(pos, quat=(0.0, 0.0, 0.0, 1.0), device=dev)

        z = lambda *s: torch.zeros(*s, device=dev)
        self.steps    = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self.last_act = z(num_envs, ACT_DIM)
        self._last_obs = z(num_envs, OBS_DIM)   # raw obs; teacher reads [:, :48]
        self.cmd      = z(num_envs, 3)
        self.cmd_timer = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self.phi      = z(num_envs)              # phase clock ∈ [0,1)

        # Load the teacher (Isaac flat walker) — reward oracle only; weights never enter student.
        self.imit_policy = torch.jit.load(
            os.path.join(fetch_assets(), "spot_policy.pt"), map_location=dev
        ).eval()

        # Logging metrics (updated each step like terrain env).
        self.last_track    = 0.0
        self.last_fell     = 0.0
        self.last_imit_div = 0.0
        self.last_tick     = 0.0
        self.last_drift    = 0.0
        self.last_duty     = [0.0, 0.0, 0.0, 0.0]  # per-leg mean contact

        # Iteration-driven schedules — initialise to iter=0 settings so the standalone
        # selftest is sane (forward-only command, W_IMIT=0.5, W_TICK=0).
        self.iter   = 0
        self.W_IMIT = 0.5
        self.W_TICK = 0.0
        self._vx_lo = 0.0;  self._vx_hi = 0.6
        self._vy_hi = 0.0;  self._wz_hi = 0.0
        self._use_stand = False

        self._resample_cmd(torch.arange(num_envs, device=dev))

    # --------------------------------------------------------------------- #
    #  Iteration-driven schedule (trainer calls set_iter each PPO iteration)
    # --------------------------------------------------------------------- #
    def set_iter(self, it: int):
        """Update all iteration-driven schedules. Call once per PPO iteration."""
        self.iter = it

        # W_IMIT(it): 0.5 for it<600; cosine 0.5->0 over [600,2000]; 0 after.
        if it < 600:
            self.W_IMIT = 0.5
        elif it < 2000:
            self.W_IMIT = 0.25 * (1.0 + math.cos(math.pi * (it - 600) / 1400))
        else:
            self.W_IMIT = 0.0

        # W_TICK(it): 0 for it<200; linear 0->0.8 over [200,800]; 0.8 after.
        # Ignored entirely when tick_enabled=False (obs-clock-only baseline).
        if not self.tick_enabled:
            self.W_TICK = 0.0
        elif it < 200:
            self.W_TICK = 0.0
        elif it < 800:
            self.W_TICK = 0.8 * (it - 200) / 600.0
        else:
            self.W_TICK = 0.8

        # Command envelope:
        if it < 200:
            # Forward-only; defeats stand-and-collect basin early on.
            self._vx_lo, self._vx_hi = 0.0,  0.6
            self._vy_hi, self._wz_hi = 0.0,  0.0
            self._use_stand = False
        elif it < 400:
            self._vx_lo, self._vx_hi = -0.5, 1.0
            self._vy_hi, self._wz_hi = 0.0,  0.0
            self._use_stand = False
        elif it < 800:
            self._vx_lo, self._vx_hi = -0.8, 1.2
            self._vy_hi, self._wz_hi = 0.4,  0.8
            self._use_stand = False
        else:
            self._vx_lo, self._vx_hi = -1.0, 1.5
            self._vy_hi, self._wz_hi = 0.8,  1.2
            self._use_stand = True

    # --------------------------------------------------------------------- #
    #  Command resampling (reads the current envelope)
    # --------------------------------------------------------------------- #
    def _resample_cmd(self, idx):
        n = idx.numel()
        if n == 0:
            return
        dev = self.sim.device
        vx = torch.empty(n, device=dev).uniform_(self._vx_lo, self._vx_hi)
        vy = torch.empty(n, device=dev).uniform_(-self._vy_hi, self._vy_hi) if self._vy_hi > 0 \
             else torch.zeros(n, device=dev)
        wz = torch.empty(n, device=dev).uniform_(-self._wz_hi, self._wz_hi) if self._wz_hi > 0 \
             else torch.zeros(n, device=dev)
        cmd = torch.stack([vx, vy, wz], dim=1)
        if self._use_stand:
            cmd[torch.rand(n, device=dev) < STAND_PROB] = 0.0
        self.cmd[idx] = cmd
        # In-episode resamples: every 120-320 steps.
        self.cmd_timer[idx] = torch.randint(120, 321, (n,), device=dev)

    # --------------------------------------------------------------------- #
    #  Observation assembly
    # --------------------------------------------------------------------- #
    def _obs(self):
        s   = self.sim
        q   = s.root_quat                                         # [K,4]
        lin_b   = quat_rotate_inverse(q, s.root_linvel)          # [K,3]
        ang_b   = quat_rotate_inverse(q, s.root_angvel)          # [K,3]
        proj_g  = quat_rotate_inverse(q, self.grav.expand(self.K, 3))
        qpos    = s.joint_pos[:, self.i2a] - self.default_q      # [K,12] Isaac order
        jv_isaac = s.joint_vel[:, self.i2a]                      # [K,12]
        clk = clock_obs(self.phi)                                 # [K,2]
        obs = torch.cat([lin_b, ang_b, proj_g, self.cmd, qpos, jv_isaac, self.last_act, clk], dim=1)
        # Store raw obs; teacher reads [:, :48] (no clock).
        self._last_obs = obs
        return obs

    # --------------------------------------------------------------------- #
    #  Reset helpers
    # --------------------------------------------------------------------- #
    def _reset_idx(self, idx):
        n = idx.numel()
        if n == 0:
            return
        dev = self.sim.device
        pose = self.base_pose[idx].clone()
        self.sim.set_root_state(idx, pose)
        self.sim.set_joint_state(idx, self.stand_q_add[idx], torch.zeros(n, self.sim.dof, device=dev))
        self.steps[idx]    = 0
        self.last_act[idx] = 0.0
        self.phi[idx]      = reset_phi(n, dev)   # randomise phase (decorrelate batch)
        self._resample_cmd(idx)

    def reset(self):
        self._reset_idx(torch.arange(self.K, device=self.sim.device))
        self.sim.read()
        for _ in range(20):
            self.sim.apply_drive_target(self.stand_q_add)
            self.sim.substep(DT / SUBSTEPS, SUBSTEPS)
        self.last_act.zero_()
        return self._obs()

    # --------------------------------------------------------------------- #
    #  Step
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def step(self, action):
        a      = action
        prev_a = self.last_act
        targets_isaac = self.default_q + ACTION_SCALE * a
        self.sim.apply_drive_target(targets_isaac[:, self.a2i])
        self.sim.substep(DT / SUBSTEPS, SUBSTEPS)

        # Advance clock AFTER the physics substep so phi aligns with the NEXT obs.
        self.phi = advance(self.phi)

        self.steps += 1
        self.last_act = a
        self.cmd_timer -= 1
        self._resample_cmd(torch.nonzero(self.cmd_timer <= 0, as_tuple=False).squeeze(-1))

        # ----- state readout ----- #
        q      = self.sim.root_quat
        lin_b  = quat_rotate_inverse(q, self.sim.root_linvel)     # [K,3]
        ang_b  = quat_rotate_inverse(q, self.sim.root_angvel)     # [K,3]
        base_z = self.sim.root_position[:, 2]                     # [K]

        # Roll proxy (same formula as terrain env):  2(q1*q2 + q0*q3)
        roll  = 2.0 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        # Pitch proxy: 2(q0*q1 - q2*q3)  (Isaac convention)
        pitch = 2.0 * (q[:, 0] * q[:, 1] - q[:, 2] * q[:, 3])

        # base_above used only for the fell check (NOT in obs).
        base_above = base_z   # flat ground -> terrain height = 0

        fell = (up_z(q) < 0.5) | (base_z < 0.25)

        # ----- tracking reward ----- #
        e_lin     = (self.cmd[:, 0] - lin_b[:, 0]).pow(2) + (self.cmd[:, 1] - lin_b[:, 1]).pow(2)
        e_ang     = (self.cmd[:, 2] - ang_b[:, 2]).pow(2)
        track_lin = torch.exp(-e_lin / SIG)
        track_ang = torch.exp(-e_ang / SIG)

        # ----- imitation reward (penalty, decays to 0) ----- #
        it       = self.iter
        ramp     = min(it / 50.0, 1.0)
        a_t      = self.imit_policy(self._last_obs[:, :48])     # raw obs[:48] — no clock
        imit_div = (a - a_t).pow(2).mean(dim=1).clamp(0.0, 9.0)
        imit     = (~fell).float() * ramp * imit_div

        # ----- tick reward (Siekmann contact-schedule, optional) ----- #
        if self.tick_enabled:
            tip_pos, tip_vel = foot_world(self.sim)
            contact  = contact_soft(tip_pos[..., 2])                     # [K,4]
            v_xy     = tip_vel[..., :2].norm(dim=-1).div(0.5).clamp(0, 1)  # [K,4]
            s_stance = desired_stance(self.phi)                           # [K,4]
            s_swing  = 1.0 - s_stance
            siek     = (s_swing * contact + s_stance * v_xy).sum(dim=1)  # [K]
            cmd_norm = self.cmd.norm(dim=1)
            cmd_gate = torch.sigmoid((cmd_norm - 0.15) / 0.05)
            tick     = cmd_gate * siek
        else:
            tip_pos, tip_vel = foot_world(self.sim)   # still compute for metrics
            contact = contact_soft(tip_pos[..., 2])
            siek    = torch.zeros(self.K, device=self.sim.device)
            tick    = siek

        # ----- full reward ----- #
        arate = a - prev_a
        rew = (3.0  * track_lin
               + 1.5  * track_ang
               + 0.05                                                          # alive
               - 1.0  * roll.pow(2)                                            # lateral level
               - 0.5  * pitch.pow(2)                                           # longitudinal level
               - 0.3  * lin_b[:, 2].pow(2)                                    # vz damp
               - 0.1  * (ang_b[:, 0].pow(2) + ang_b[:, 1].pow(2))            # roll/pitch rate damp
               - 0.005 * arate.pow(2).mean(dim=1)                             # action rate
               - 0.0002 * a.pow(2).mean(dim=1)                                # effort
               - 2.0  * (base_z - 0.42).pow(2)                               # height anchor
               - self.W_IMIT * imit                                            # imitation (decays)
               - self.W_TICK * tick                                            # tick (optional)
               - 5.0  * fell.float())                                          # fell penalty

        # ----- episode termination ----- #
        timeout = (self.steps >= self.max_steps) & ~fell
        done    = fell | (self.steps >= self.max_steps)
        term_obs = self._obs()
        d = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if d.numel() > 0:
            self._reset_idx(d)
            self.sim.read()
            obs = self._obs()
        else:
            obs = term_obs

        # ----- logging metrics ----- #
        self.last_track    = (track_lin + track_ang).mean().item()
        self.last_fell     = fell.float().mean().item()
        self.last_imit_div = imit_div.mean().item()
        self.last_tick     = siek.mean().item()
        # Lateral drift: envs where |cmd_vy|<0.05 AND |cmd_wz|<0.05
        straight = (self.cmd[:, 1].abs() < 0.05) & (self.cmd[:, 2].abs() < 0.05)
        self.last_drift = lin_b[straight, 1].abs().mean().item() if straight.any() else 0.0
        # Per-leg duty (mean contact over this step)
        self.last_duty = contact.mean(dim=0).tolist()

        return obs, rew, done, term_obs, timeout

    # --------------------------------------------------------------------- #
    #  Tracking evaluation (teacher/trainer calls this)
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def measure_tracking(self, act_fn, cmd, steps=160, warm=60):
        """Hold a FIXED command and report mean tracking error
        ||lin_b_xy - cmd_xy|| + |ang_b_z - wz| over the last (steps-warm) ticks.
        act_fn(obs) -> [K,12] action (already norm-aware if normalize_obs=True)."""
        dev = self.sim.device
        c   = torch.tensor(cmd, device=dev, dtype=torch.float32).expand(self.K, 3).contiguous()
        obs = self.reset()
        errs = []
        for t in range(steps):
            self.cmd        = c
            self.cmd_timer.fill_(10 ** 9)
            obs, _, _, _, _ = self.step(act_fn(obs))
            if t >= warm:
                q     = self.sim.root_quat
                lin_b = quat_rotate_inverse(q, self.sim.root_linvel)
                ang_b = quat_rotate_inverse(q, self.sim.root_angvel)
                e = (lin_b[:, :2] - c[:, :2]).norm(dim=1).mean() + (ang_b[:, 2] - c[:, 2]).abs().mean()
                errs.append(e.item())
        return sum(errs) / max(1, len(errs))


# --------------------------------------------------------------------------- #
#  Selftest (K=64, 200 zero-action steps, print stand-time metrics)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)

    K = int(os.environ.get("K", "64"))
    print(f"Building SpotScratchEnv (K={K}, tick_enabled=True)...")
    env = SpotScratchEnv(num_envs=K, tick_enabled=True)
    print("Resetting...")
    obs = env.reset()

    # --- assertion 1: obs shape and finiteness ---
    assert obs.shape == (K, OBS_DIM), f"Expected obs.shape=({K},{OBS_DIM}), got {tuple(obs.shape)}"
    assert torch.isfinite(obs).all(), "obs has non-finite values after reset!"
    print(f"obs shape: {tuple(obs.shape)}  finite: {bool(torch.isfinite(obs).all())}")

    # --- 200 zero-action steps ---
    dev = env.sim.device
    zero_a = torch.zeros(K, ACT_DIM, device=dev)
    for step_i in range(200):
        obs, rew, done, term, to = env.step(zero_a)
        assert torch.isfinite(obs).all(), f"obs non-finite at step {step_i}"
        assert torch.isfinite(rew).all(), f"rew non-finite at step {step_i}"

    print(f"200 zero-action steps: all finite")
    print(f"  track     = {env.last_track:.3f}")
    print(f"  fell/step = {env.last_fell:.3f}")
    print(f"  imit_div  = {env.last_imit_div:.3f}")
    print(f"  drift     = {env.last_drift:.3f}")
    print(f"  duty(fl,fr,hl,hr) = {[round(d, 3) for d in env.last_duty]}")
    print(f"  rew_mean  = {rew.mean().item():+.3f}")
    print(f"  W_IMIT    = {env.W_IMIT:.3f}  W_TICK = {env.W_TICK:.3f}")
    print("SPOT-SCRATCH ENV SELFTEST: PASS")
