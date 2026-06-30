"""spotv2 — DISCRETE STAIRS with an ADAPTIVE per-env difficulty curriculum.

Stairs are the hard case (a flat walker stalled at ~0.17 m because it can't lift its feet over a
tall riser). Two things make this attempt different: (1) warm-start from scratch_flat_best.pt — it has the
clock-augmented flat gait (OBS_DIM=50, normalize_obs=True, stiff gains=90); (2) an ADAPTIVE terrain-level
curriculum so each env only faces a riser it has earned.

Curriculum design (works within GpuSim's static terrain — terrain is built once, only the robot moves):
each STAIR lane is a strip of constant-difficulty BANDS along +x; band j is a flat approach + an up/down
"tent" at riser RISERS[j] + a run-out (so every band starts and ends at ground level). Each env carries a
`level` that selects its spawn band. After an episode the level PROMOTES if the robot cleared its tent
(forward distance > CLEAR_DIST) and DEMOTES if it fell — so envs start at the smallest riser and ramp only
as the policy improves. FLAT_FRAC of lanes are kept flat (no tents) for full-steering replay.

New obs layout: [0:48] proprio | [48:50] clock | [50:51] base_above | [51:96] scan (45)  =  OBS_DIM=96.
First 50 dims byte-identical to scratch_flat → warm-start copies cols [0:50], terrain cols zero-init.
Anchor = scratch_flat_best.pt (50-d, norm-aware); stiff gains (90) on both training and deploy robots.
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spot"))
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))   # scratch_clock / scratch_env

import threepp as tp
from threepp.rl import GpuSim, load_policy
from spot_deploy import build_spot, default_q, add_to_isaac, isaac_to_add, ACTION_SCALE
from spot_terrain_env import (quat_rotate_inverse, up_z, heading_cossin, _flat_ground,
                              scan_offsets, scan_xy, N_SCAN,
                              CONTROL_HZ, DT, SUBSTEPS, SPACING, SPAWN_Z, PROBE_DX, ACT_DIM,
                              HIDDEN, HALF_W, VX_LO, VX_HI, VY_HI, WZ_HI, STAND_PROB,
                              FWD_DRIVE_FRAC, CMD_MIN, CMD_MAX, SIG)
from scratch_clock import CLOCK0, CLOCK_DIM, GAIT_PERIOD, advance, clock_obs, reset_phi
from scratch_env import STIFF_GAINS

OBS_DIM = 48 + CLOCK_DIM + 1 + N_SCAN   # = 96: [proprio(48)|clock(2)|base_above(1)|scan(45)]

# --------------------------------------------------------------------------- #
#  Stair bands (the difficulty ladder along +x) + curriculum
# --------------------------------------------------------------------------- #
RISERS = (0.04, 0.07, 0.10, 0.13, 0.16, 0.20)   # per-LEVEL riser height (the difficulty axis)
N_LEVELS = len(RISERS)
N_UP = 3                       # steps up (= steps down) per tent
STEP_RUN = 0.30                # tread depth (m)
FLAT_APPROACH = 1.6            # flat run before each tent (room to spawn + line up)
LAND = 0.8                     # landing at the tent peak
RUNOUT = 0.9                   # flat run-out after each tent (back to ground level)
TENT_LEN = 2.0 * N_UP * STEP_RUN + LAND          # ascend + landing + descend
BAND_LEN = FLAT_APPROACH + TENT_LEN + RUNOUT     # one difficulty band's x-extent
SPAWN_OFF = 0.7               # spawn this far into the band's flat approach
# promote/demote are measured as forward distance FROM SPAWN (spawn is SPAWN_OFF into the approach):
CLEAR_DIST = (FLAT_APPROACH - SPAWN_OFF) + TENT_LEN   # cleared the whole tent -> promote
REACH_DIST = (FLAT_APPROACH - SPAWN_OFF) * 0.6        # never even reached the tent -> demote
STRIP_LEN = N_LEVELS * BAND_LEN

FLAT_FRAC = 0.30               # fraction of FLAT lanes (no tents) -> full-steering replay
W_IMIT = 0.25                  # scan-gated imitation anchor (hold the teacher's gait on flat patches;
                               # 0.2 let BACKWARD steering regress ~2x, so anchor harder — cf. heightfield)
STEPS_EPISODE_S = 16.0         # time to approach + climb a tent at the commanded speed
HALF_W_STEPS = SPACING * 0.5   # on-lane gate = FULL lane half-width (no flat strip to walk around)
HALF_W_BOX = SPACING           # tent box width = full lane -> tents tile with no gap between lanes
FOOT_DX = (0.30, 0.30, -0.30, -0.30)   # stance foot offsets for spawn clearance
FOOT_DY = (0.17, -0.17, 0.17, -0.17)

CONFIG = {"control_hz": CONTROL_HZ, "dt": DT, "substeps": SUBSTEPS, "spacing": SPACING,
          "terrain": "steps", "risers": list(RISERS), "n_up": N_UP, "step_run": STEP_RUN,
          "band_len": BAND_LEN, "episode_s": STEPS_EPISODE_S, "probe_dx": list(PROBE_DX),
          "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "hidden": list(HIDDEN),
          "vx": [VX_LO, VX_HI], "vy_hi": VY_HI, "wz_hi": WZ_HI, "stand_prob": STAND_PROB,
          "sig": SIG, "w_imit": W_IMIT,
          "stiff_gains": {k: list(v) for k, v in STIFF_GAINS.items()},
          "gait_period": GAIT_PERIOD}


def _tent(t, riser, run=STEP_RUN, n=N_UP, land=LAND):
    """Up/down tent height at LOCAL x `t` (t<0 = before the tent -> 0; ascend n -> landing -> descend n
    -> 0). Vectorized; t and riser broadcast. riser may be a per-element tensor."""
    up_end = n * run
    land_end = up_end + land
    tent_end = land_end + n * run
    asc = torch.clamp(torch.floor(t / run) + 1.0, 0.0, float(n))
    desc = torch.clamp(torch.floor((t - land_end) / run) + 1.0, 0.0, float(n))
    steps = torch.where(t < 0.0, torch.zeros_like(t),
            torch.where(t < up_end, asc,
            torch.where(t < land_end, torch.full_like(t, float(n)),
            torch.where(t < tent_end, float(n) - desc, torch.zeros_like(t)))))
    return steps * riser


def _add_steps(world, k, spacing, is_stairs, risers_np, w=HALF_W_BOX):
    """Per STAIR lane: the band ladder — for each level j a tent at riser RISERS[j], placed at band j's
    x-offset (flat approach in front, run-out behind). Solid boxes from the ground up to each tread top."""
    for i in range(k):
        if not bool(is_stairs[i]):
            continue
        y = i * spacing
        for j in range(N_LEVELS):
            r = float(risers_np[j])
            x0 = j * BAND_LEN + FLAT_APPROACH                      # tent starts after the flat approach
            for s in range(N_UP):                                  # ascend: tread s top at (s+1)*r
                h = (s + 1) * r
                b = tp.Mesh(tp.BoxGeometry(STEP_RUN, w, h), tp.MeshStandardMaterial())
                b.position.set(x0 + s * STEP_RUN + STEP_RUN * 0.5, y, h * 0.5)
                world.add_static(b)
            up_end = x0 + N_UP * STEP_RUN
            top = N_UP * r
            lb = tp.Mesh(tp.BoxGeometry(LAND, w, top), tp.MeshStandardMaterial())   # landing
            lb.position.set(up_end + LAND * 0.5, y, top * 0.5)
            world.add_static(lb)
            land_end = up_end + LAND
            for s in range(N_UP - 1):                              # descend: tread s top at (n-1-s)*r
                h = (N_UP - 1 - s) * r
                b = tp.Mesh(tp.BoxGeometry(STEP_RUN, w, h), tp.MeshStandardMaterial())
                b.position.set(land_end + s * STEP_RUN + STEP_RUN * 0.5, y, h * 0.5)
                world.add_static(b)


class SpotStepsEnv:
    def __init__(self, num_envs=1024, device="cuda", seed=0, flat_only=False):
        self.K, self.dt = num_envs, DT
        self.max_steps = int(STEPS_EPISODE_S * CONTROL_HZ)
        rng = np.random.default_rng(seed)
        is_stairs_np = np.ones(num_envs, bool) if not flat_only else np.zeros(num_envs, bool)
        if not flat_only:
            is_stairs_np[rng.random(num_envs) < FLAT_FRAC] = False     # FLAT lanes -> steering replay
        risers_np = np.array(RISERS, np.float32)
        # Stiff gains (90) = same plant the base gait scratch_flat_best.pt was trained on.
        class _SpotStepsRobot:
            def __init__(self_, world, i):
                self_.art, _ = build_spot(world, assets=None, base_xy=(0.0, i * SPACING),
                                          gains=STIFF_GAINS)
        self.sim = GpuSim(num_envs, lambda world, i: _SpotStepsRobot(world, i),
                          gravity=(0.0, 0.0, -9.81), spacing=SPACING, device=device, read_root=True,
                          build_world=lambda world: (_flat_ground(world, num_envs, SPACING),
                                                      _add_steps(world, num_envs, SPACING, is_stairs_np, risers_np)))
        dev = self.sim.device
        self.default_q = torch.from_numpy(default_q).to(dev)
        self.i2a = torch.from_numpy(isaac_to_add.astype(np.int64)).to(dev)
        self.a2i = torch.from_numpy(add_to_isaac.astype(np.int64)).to(dev)
        self.stand_q_add = self.default_q[self.a2i].expand(num_envs, -1).contiguous()
        self.grav = torch.tensor([0.0, 0.0, -1.0], device=dev)
        self.risers = torch.from_numpy(risers_np).to(dev)                 # [N_LEVELS]
        self.is_stairs = torch.from_numpy(is_stairs_np).to(dev)           # [K] bool
        self.gx, self.gy = scan_offsets(dev)                              # [N_SCAN] heading-relative grid offsets
        # Anchor = the clock-aware base gait (50-d, normalize_obs=True); frozen throughout.
        _scratch = os.path.join(_HERE, "scratch_distillation", "scratch_flat_best.pt")
        self.anchor_ac, self.anchor_norm, _ = load_policy(_scratch, device=dev)
        self.anchor_ac.eval()
        self.lane_y = torch.arange(num_envs, device=dev, dtype=torch.float32) * SPACING
        pos = torch.zeros(num_envs, 3, device=dev); pos[:, 1] = self.lane_y; pos[:, 2] = SPAWN_Z
        self.base_pose = GpuSim.make_root_pose(pos, quat=(0.0, 0.0, 0.0, 1.0), device=dev)
        z = lambda *s: torch.zeros(*s, device=dev)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self.level = torch.zeros(num_envs, dtype=torch.long, device=dev)   # CURRICULUM: per-env difficulty band
        self.last_act = z(num_envs, ACT_DIM); self._last_obs = z(num_envs, OBS_DIM)
        self.phi = z(num_envs)                                             # phase clock ∈ [0,1)
        self.up = z(num_envs); self.cmd = z(num_envs, 3)
        self.cmd_timer = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self.ep_start_x = z(num_envs); self.ep_max_climb = z(num_envs)
        self._resample_cmd(torch.arange(num_envs, device=dev))
        self.last_track = 0.0; self.last_flat_track = 0.0; self.last_level = 0.0; self.last_fell = 0.0
        self.last_clear = 0.0

    def _terrain_h(self, x, y):
        """Tent height at (x,y): pick the band from x, evaluate that band's tent, gate to stair lanes
        + lane width. x,y [K] or [K,P]."""
        lane = self.lane_y if x.dim() == 1 else self.lane_y[:, None]
        stairs = self.is_stairs.float() if x.dim() == 1 else self.is_stairs.float()[:, None]
        band = torch.clamp(torch.floor(x / BAND_LEN).long(), 0, N_LEVELS - 1)
        riser = self.risers[band]                                          # per-(x) riser of its band
        t = (x - band.to(x.dtype) * BAND_LEN) - FLAT_APPROACH              # local x within the band's tent
        h = _tent(t, riser)
        on = ((torch.abs(y - lane) < HALF_W_STEPS) & (x >= 0.0) & (x < STRIP_LEN)).float()
        return h * on * stairs

    def _resample_cmd(self, idx):
        n = idx.numel()
        if n == 0:
            return
        dev = self.sim.device
        vx = torch.empty(n, device=dev).uniform_(VX_LO, VX_HI)
        vy = torch.empty(n, device=dev).uniform_(-VY_HI, VY_HI)
        wz = torch.empty(n, device=dev).uniform_(-WZ_HI, WZ_HI)
        drive = (torch.rand(n, device=dev) < FWD_DRIVE_FRAC) & self.is_stairs[idx]   # stair lanes drive at the tent
        vx = torch.where(drive, torch.empty(n, device=dev).uniform_(0.4, VX_HI), vx)
        vy = torch.where(drive, vy * 0.2, vy)
        wz = torch.where(drive, wz * 0.2, wz)
        cmd = torch.stack([vx, vy, wz], dim=1)
        cmd[torch.rand(n, device=dev) < STAND_PROB] = 0.0
        self.cmd[idx] = cmd
        self.cmd_timer[idx] = torch.randint(CMD_MIN, CMD_MAX + 1, (n,), device=dev)

    def _obs(self):
        s = self.sim
        q = s.root_quat
        lin_b = quat_rotate_inverse(q, s.root_linvel)
        ang_b = quat_rotate_inverse(q, s.root_angvel)
        proj_g = quat_rotate_inverse(q, self.grav.expand(self.K, 3))
        qpos = s.joint_pos[:, self.i2a] - self.default_q
        jv_isaac = s.joint_vel[:, self.i2a]
        x, y, zz = s.root_position[:, 0], s.root_position[:, 1], s.root_position[:, 2]
        cyaw, syaw = heading_cossin(q)
        h_here = self._terrain_h(x, y)
        px, py = scan_xy(x, y, cyaw, syaw, self.gx, self.gy)
        ahead = (self._terrain_h(px, py) - h_here[:, None]).clamp(-1.0, 1.0)
        base_above = (zz - h_here).unsqueeze(-1)
        clk = clock_obs(self.phi)                                          # [K,2] clock after last substep
        # Layout: [proprio(48)|clock(2)|base_above(1)|scan(45)] = 96-d
        # First 50 (proprio+clock) byte-identical to scratch_flat -> anchor reads obs[:,:50]
        obs = torch.cat([lin_b, ang_b, proj_g, self.cmd, qpos, jv_isaac, self.last_act,
                         clk, base_above, ahead], dim=1)
        self._last_obs = obs
        return obs

    def _spawn_x(self, idx):
        """Spawn x for the subset: stair lanes at their level's band approach; flat lanes anywhere flat."""
        dev = self.sim.device
        sx = self.level[idx].to(torch.float32) * BAND_LEN + SPAWN_OFF
        flat = ~self.is_stairs[idx]
        sx = torch.where(flat, torch.rand(idx.numel(), device=dev) * 3.0, sx)
        return sx

    def _foot_clear_z(self, idx, sx):
        """Highest terrain under the stance footprint at the spawn -> drop-settle (no spawn penetration)."""
        dev = self.sim.device
        fdx = torch.tensor(FOOT_DX, device=dev); fdy = torch.tensor(FOOT_DY, device=dev)
        fx = sx[:, None] + fdx[None, :]                                   # [n,4]
        fy = self.lane_y[idx][:, None] + fdy[None, :]                     # [n,4] world-y
        # evaluate terrain at the foot points for these specific lanes (per-row band + riser + gate)
        lane = self.lane_y[idx][:, None]
        stairs = self.is_stairs[idx].float()[:, None]
        band = torch.clamp(torch.floor(fx / BAND_LEN).long(), 0, N_LEVELS - 1)
        riser = self.risers[band]
        t = (fx - band.to(fx.dtype) * BAND_LEN) - FLAT_APPROACH
        h = _tent(t, riser)
        on = ((torch.abs(fy - lane) < HALF_W_STEPS) & (fx >= 0.0) & (fx < STRIP_LEN)).float()
        return (h * on * stairs).max(dim=1).values

    def _reset_idx(self, idx):
        n = idx.numel()
        if n == 0:
            return
        dev = self.sim.device
        pose = self.base_pose[idx].clone()
        sx = self._spawn_x(idx)
        sz = self._foot_clear_z(idx, sx) + SPAWN_Z + 0.03                 # drop-settle clearance
        pose[:, 4] = sx; pose[:, 6] = sz
        self.sim.set_root_state(idx, pose)
        self.sim.set_joint_state(idx, self.stand_q_add[idx], torch.zeros(n, self.sim.dof, device=dev))
        self.steps[idx] = 0; self.last_act[idx] = 0.0
        self.phi[idx] = reset_phi(n, dev)    # randomise phase (decorrelate batch; don't advance during settle)
        self.ep_start_x[idx] = sx; self.ep_max_climb[idx] = 0.0
        self._resample_cmd(idx)
        # stair lanes: force a forward command at spawn so the robot ATTEMPTS the tent (level evaluation)
        st = idx[self.is_stairs[idx]]
        if st.numel() > 0:
            self.cmd[st, 0] = torch.empty(st.numel(), device=dev).uniform_(0.5, VX_HI)
            self.cmd[st, 1:] = 0.0

    def reset(self):
        self._reset_idx(torch.arange(self.K, device=self.sim.device))
        self.sim.read()
        for _ in range(20):
            self.sim.apply_drive_target(self.stand_q_add)
            self.sim.substep(DT / SUBSTEPS, SUBSTEPS)        # advance n substeps, read once
        self.last_act.zero_()
        self.up = up_z(self.sim.root_quat)
        return self._obs()

    @torch.no_grad()
    def step(self, action):
        a = action
        prev_a = self.last_act
        targets_isaac = self.default_q + ACTION_SCALE * a
        self.sim.apply_drive_target(targets_isaac[:, self.a2i])
        self.sim.substep(DT / SUBSTEPS, SUBSTEPS)                            # advance n substeps, read once
        self.phi = advance(self.phi)                                          # clock after physics, before next obs
        self.steps += 1
        self.last_act = a
        self.cmd_timer -= 1
        self._resample_cmd(torch.nonzero(self.cmd_timer <= 0, as_tuple=False).squeeze(-1))

        q = self.sim.root_quat
        self.up = up_z(q)
        x, y, zz = self.sim.root_position[:, 0], self.sim.root_position[:, 1], self.sim.root_position[:, 2]
        roll = 2.0 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
        ang_b = quat_rotate_inverse(q, self.sim.root_angvel)
        lin_b = quat_rotate_inverse(q, self.sim.root_linvel)
        h_here = self._terrain_h(x, y)
        base_above = zz - h_here
        cyaw, syaw = heading_cossin(q)
        px, py = scan_xy(x, y, cyaw, syaw, self.gx, self.gy)
        ahead = self._terrain_h(px, py) - h_here[:, None]
        change = ahead.abs().max(dim=1).values
        w_imit = (1.0 - change / 0.10).clamp(0.0, 1.0)
        # Anchor: the 50-d clock base gait with its frozen RunningNorm (obs[:,:50] = proprio+clock).
        anchor_a = self.anchor_ac.act_mean(self.anchor_norm.norm(self._last_obs[:, :50]))
        imit = w_imit * (a - anchor_a).pow(2).mean(dim=1)
        arate = a - prev_a
        fell = (self.up < 0.35) | (base_above < 0.18)
        self.ep_max_climb = torch.maximum(self.ep_max_climb, (x - self.ep_start_x).clamp_min(0.0))

        e_lin = (self.cmd[:, 0] - lin_b[:, 0]).pow(2) + (self.cmd[:, 1] - lin_b[:, 1]).pow(2)
        e_ang = (self.cmd[:, 2] - ang_b[:, 2]).pow(2)
        track_lin = torch.exp(-e_lin / SIG)
        track_ang = torch.exp(-e_ang / SIG)
        rew = (3.0 * track_lin
               + 1.5 * track_ang
               + 0.05
               - 1.0 * roll.pow(2)                       # ROLL only — climbing legitimately PITCHES
               - 0.1 * lin_b[:, 2].pow(2)
               - 0.05 * (ang_b[:, 0].pow(2) + ang_b[:, 2].pow(2))
               - 3.0 * torch.relu(0.30 - base_above)     # anti-scrape over the steps
               - 0.001 * arate.pow(2).mean(dim=1)
               - W_IMIT * imit
               - 5.0 * fell.float())

        timeout = (self.steps >= self.max_steps) & ~fell
        done = fell | (self.steps >= self.max_steps)
        term_obs = self._obs()
        d = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if d.numel() > 0:
            # CURRICULUM update for STAIR envs ending now: promote on clearing the tent, demote on a fall.
            st = d[self.is_stairs[d]]
            if st.numel() > 0:
                # DISTANCE-based (not fall-gated): clearing the tent promotes even if it falls later;
                # only a robot that never reached the tent (fell early / stalled) demotes.
                prog = self.ep_max_climb[st]
                self.level[st] = torch.clamp(self.level[st]
                                             + (prog > CLEAR_DIST).long()
                                             - (prog < REACH_DIST).long(), 0, N_LEVELS - 1)
                self.last_clear = (prog > CLEAR_DIST).float().mean().item()
            self._reset_idx(d)
            self.sim.read()
            self.up = up_z(self.sim.root_quat)
            obs = self._obs()
        else:
            obs = term_obs
        self.last_fell = fell.float().mean().item()
        self.last_track = (track_lin + track_ang).mean().item()
        flat = ~self.is_stairs
        self.last_flat_track = ((track_lin + track_ang)[flat]).mean().item() if bool(flat.any()) else float("nan")
        self.last_level = self.level[self.is_stairs].float().mean().item() if bool(self.is_stairs.any()) else 0.0
        return obs, rew, done, term_obs, timeout

    @torch.no_grad()
    def measure_tracking(self, act_fn, cmd, steps=160, warm=60):
        dev = self.sim.device
        c = torch.tensor(cmd, device=dev, dtype=torch.float32).expand(self.K, 3).contiguous()
        obs = self.reset()
        errs = []
        for t in range(steps):
            self.cmd = c
            self.cmd_timer.fill_(10 ** 9)
            obs, _, _, _, _ = self.step(act_fn(obs))
            if t >= warm:
                q = self.sim.root_quat
                lin_b = quat_rotate_inverse(q, self.sim.root_linvel)
                ang_b = quat_rotate_inverse(q, self.sim.root_angvel)
                e = (lin_b[:, :2] - c[:, :2]).norm(dim=1).mean() + (ang_b[:, 2] - c[:, 2]).abs().mean()
                errs.append(e.item())
        return sum(errs) / max(1, len(errs))


if __name__ == "__main__":
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    K = int(os.environ.get("K", "64"))
    env = SpotStepsEnv(num_envs=K)
    obs = env.reset()
    assert obs.shape == (K, 96), f"expected obs (K,96), got {tuple(obs.shape)}"
    print(f"obs {tuple(obs.shape)} (OBS_DIM={OBS_DIM}=96) finite={bool(torch.isfinite(obs).all())}  "
          f"levels={N_LEVELS} risers={RISERS}")
    for _ in range(200):
        obs, rew, done, term, to = env.step(torch.zeros(K, ACT_DIM, device=env.sim.device))
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    print(f"zero-action (stand): track={env.last_track:.3f}  flat_track={env.last_flat_track:.3f}  "
          f"level={env.last_level:.2f}  fell/step={env.last_fell:.3f}  rew={rew.mean().item():+.3f}")
    print("SPOTV2-STEPS ENV SELFTEST: PASS")
