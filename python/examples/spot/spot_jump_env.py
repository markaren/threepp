"""A jump on command for Spot: from standing, jump (in place or forward by a commanded distance), land upright and stand.

Built on SpotRecoveryEnv's plant and bookkeeping (stiff gains, real torque limits 45/45/115 N·m, self-collision, the
joint-target rate limit qd_max and the measured-speed penalty w_qd_excess), so a jump policy runs on the robot the
recovery policy runs on, and continues it whole: the observation is the same 96-d walking layout.

The jump signal rides in the channels the walking policy reads its command and clock from:
  - cmd  [9:12]  = (dx, 0, 0) during the jump window, else zero. dx is the forward jump distance (m) along the heading.
  - clock [48:50] = (1, p) during the window, p = ticks since the trigger / WINDOW_TICKS (0 on the first window obs, just
    under 1 on the last), else the (0, 0) stand sentinel.
The window is a fixed WINDOW_S from the trigger, so a deploy needs only a timer (no contact sensing): press, count ticks.

Episodes (EPISODE_S): a standing start near the stance (random yaw). A share JUMP_FRAC of episodes gets a trigger at a
random time in TRIGGER_S; dx is 0 in DX_ZERO_FRAC of them, else uniform in (0, dx_max]. The rest never trigger and
must stand. No curriculum.

Airborne: all four lower-leg capsules clear the ground by AIR_Z (from the GPU link poses; flat ground at z = 0). A take-off
also needs the base rising faster than TAKEOFF_VZ (probe 2026-09-14: folding the legs at the rate limit lifts every foot
2-7 cm while the base drops). A flight counts once it lasts MIN_FLIGHT_TICKS; a shorter one re-arms the jump while the
window is open.

Rewards, on top of the recovery env's upright term and effort penalties:
  - stand terms (height, pose, still) are off during the window, so crouching and leaving the ground cost nothing there.
  - rise: W_RISE x the increase of the running max base height since the trigger, until landing (a bob cannot farm it).
  - flight: W_FLIGHT per airborne tick of the commanded flight.
  - land: W_LAND on touching down upright (up_z > LAND_UP) after a counted flight; dist: W_DIST x exp(-(err / 0.15)^2)
    at that landing, err = achieved minus commanded forward distance (take-off to touch-down along the trigger heading).
  - miss: -W_MISS when the window closes without a counted flight.
  - uncommanded: -W_UNCMD per airborne tick that is not the commanded flight (no trigger, before it, after landing).
  - fall: a terminal (up_z < FALL_UP or base below FALL_Z, after the settle ticks) costing W_FALL.
Standing still in the stance pays 3.0 per tick (the recovery env's terms), so a fall forfeits the rest of the episode;
the jump bonuses are sized so that a clean jump (~150) outweighs the risk of trying.

Metrics per episode (jump_stats): jumped (a counted flight in the window), landed upright, success (jumped, landed upright,
not fallen, and standing still at the end: the recovery env's stand criterion held for its last 1 s), fell, flight time,
apex base rise, foot clearance (the lowest foot's highest point during the flight), forward distance and its error, and
uncommanded airborne ticks; by command bucket (no jump / dx = 0 / dx <= 0.2 / dx > 0.2). Joint speeds over all live ticks
(the recovery env's histogram) and over window ticks only (qd_window), since the jump's speeds hide in a standing average.

Wave 2 knobs (2026-09-14), off by default so the defaults are the pilot. The pilot (spot_jump1, 4 x 1500 iterations, arms
at 10 and 20 rad/s) never left the ground: 0 flights in ~25k training episodes per run, the rise term 0.004-0.05 per step
(standing taller), every window missed, no falls. Exploration from the standing policy never finds a coordinated crouch
and push, and the height max has no slope past full leg extension without one.
  - w_vz W: W x the increase of the running max of the base's upward speed (capped at VZ_CAP) since the trigger, while the
    jump is pending or in flight. It telescopes to W x the peak take-off speed, a slope all the way from standing up to a
    jump (apex = vz^2 / 2g), and a bob cannot farm it.
  - crouch_frac F: a share F of the jump episodes spawns crouched (the stance plus u x CROUCH_Q, u uniform in [0, 1], every
    depth in between) with the trigger CROUCH_TRIGGER_S after the spawn: the stroke a push needs is given, not searched
    for. Counted in their own bucket (crouch), outside the standing-start jump totals.
  - w_imit W: W x exp(-mean((q - q_ref)^2) / IMIT_SIGMA2) per window tick, q_ref the stance plus REF_KEYS interpolated at
    the tick since the trigger (crouch, hold, push, tuck, stand; the push is the open-loop probe hop that flew 0.38 s and
    stayed up). A 150-iteration local check with w_vz 40 and crouch_frac 0.5 alone (K=1024) moved the peak upward speed
    from 0.17 to 0.22 m/s on standing starts and 0.24 to 0.29 on crouched ones: a slope, but a take-off needs ~2 m/s.
    The reference puts exploration on a jump; flight, landing and distance still say what a good one is.
  - metrics, always on: peak upward base speed and height rise over every jump episode, not only those that flew.
"""
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from spot_recovery_env import (SpotRecoveryEnv, quat_axis_angle, quat_mul, quat_rot, lowest_point, OBS_DIM, ACT_DIM,
                               CONTROL_HZ, DT, N_SCAN, STAND_Z, SUCCESS_HOLD, QD_BINS, QD_W, QD_OVER, JOINT_GROUPS,
                               STIFF_GAINS)
from spot_terrain_env import quat_rotate_inverse
from spot_deploy import ACTION_SCALE

EPISODE_S = 4.0
SETTLE_TICKS = 5                       # rewards and fall checks start after these
JUMP_FRAC = 0.85
TRIGGER_S = (0.4, 1.4)
WINDOW_S = 1.2
WINDOW_TICKS = int(round(WINDOW_S * CONTROL_HZ))
DX_MAX = 0.4
DX_ZERO_FRAC = 0.3
AIR_Z = 0.02                           # every lower-leg capsule this far above the ground = airborne
MIN_FLIGHT_TICKS = 5                   # 0.1 s
LAND_UP = 0.85
FALL_UP, FALL_Z = 0.3, 0.11           # base box half height 0.095: below 0.11 the body is on the ground (a deep crouch is
                                       # legal: the open-loop probe crouched to 0.188 m)
TAKEOFF_VZ = 0.2                       # a take-off needs the base rising: a fast crouch lifts all four feet while the body drops
SEG_HALF, SHIN_R = 0.15, 0.028         # lower-leg capsule half length and radius (build_spot)
W_RISE, W_FLIGHT, W_LAND, W_DIST, W_MISS, W_UNCMD, W_FALL = 150.0, 3.0, 50.0, 30.0, 30.0, 3.0, 10.0
DIST_SIGMA = 0.15
VZ_CAP = 3.0                           # m/s: w_vz stops paying above this (0.46 m of apex)
CROUCH_Q = (0.0, 0.3, -0.6)            # hx, hy, kn offsets from the stance at full crouch depth (the probes' crouch)
CROUCH_TRIGGER_S = (0.12, 0.3)
# w_imit's reference: joint offsets from the stance (hx, hy, kn, the same on every leg) at window ticks, linear in between
REF_KEYS = ((0, (0.0, 0.0, 0.0)), (12, (0.0, 0.3, -0.6)), (16, (0.0, 0.3, -0.6)), (24, (0.0, -0.3, 0.6)),
            (40, (0.0, 0.1, -0.3)), (60, (0.0, 0.0, 0.0)))
IMIT_SIGMA2 = 0.1
BUCKETS = ("nojump", "dx0", "dx_short", "dx_long", "crouch")   # dx_short: 0 < dx <= 0.2, dx_long: dx > 0.2; crouch: w2
# per-episode counters: episodes, jumped, landed upright, success, fell, flight s, rise, clearance, dx achieved,
# |dx err|, uncommanded air ticks, peak upward base speed and rise over every episode
NSTAT = 13
CONFIG = {"task": "jump", "control_hz": CONTROL_HZ, "dt": DT, "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
          "episode_s": EPISODE_S, "settle_ticks": SETTLE_TICKS, "jump_frac": JUMP_FRAC, "trigger_s": list(TRIGGER_S),
          "window_s": WINDOW_S, "window_ticks": WINDOW_TICKS, "dx_zero_frac": DX_ZERO_FRAC, "air_z": AIR_Z, "vz_cap": VZ_CAP, "crouch_q": list(CROUCH_Q),
          "crouch_trigger_s": list(CROUCH_TRIGGER_S),
          "ref_keys": [[k, list(v)] for k, v in REF_KEYS], "imit_sigma2": IMIT_SIGMA2,
          "min_flight_ticks": MIN_FLIGHT_TICKS, "takeoff_vz": TAKEOFF_VZ, "land_up": LAND_UP, "fall": {"up": FALL_UP, "z": FALL_Z},
          "jump_weights": {"rise": W_RISE, "flight": W_FLIGHT, "land": W_LAND, "dist": W_DIST, "dist_sigma": DIST_SIGMA,
                           "miss": W_MISS, "uncmd": W_UNCMD, "fall": W_FALL},
          "signal": "cmd[9:12] = (dx, 0, 0) and clock[48:50] = (1, ticks since trigger / window_ticks) in the window",
          "stiff_gains": {k: list(v) for k, v in STIFF_GAINS.items()}, "stand_mode": True}
PEND, FLY, LANDED, MISSED = 0, 1, 2, 3


class SpotJumpEnv(SpotRecoveryEnv):
    episode_s = EPISODE_S

    def __init__(self, num_envs=2048, device="cuda", seed=0, self_collision=True, action_scale=0.5, qd_max=10.0,
                 w_qd_excess=0.0, jump_frac=JUMP_FRAC, dx_max=DX_MAX, trigger_s=TRIGGER_S, count_episodes=None,
                 freeze_level=True, w_vz=0.0, crouch_frac=0.0, w_imit=0.0):
        """jump_frac: share of episodes with a trigger; trigger_s (lo, hi) seconds, or a number for a fixed trigger
        (evaluation). The plant and the target rate limit are the recovery env's (target clip on). w_vz, crouch_frac:
        w_imit: the wave 2 knobs (module docstring)."""
        super().__init__(num_envs=num_envs, device=device, seed=seed, self_collision=self_collision, stand_frac=1.0,
                         init_level=0, freeze_level=True, count_episodes=count_episodes, limp_ticks=0,
                         action_scale=action_scale, target_clip=True, qd_max=qd_max, w_qd_excess=w_qd_excess)
        self.jump_frac, self.dx_max = float(jump_frac), float(dx_max)
        self.w_vz, self.crouch_frac, self.w_imit = float(w_vz), float(crouch_frac), float(w_imit)
        ref = np.zeros((WINDOW_TICKS, 3), np.float32)
        vals = np.array([v for _, v in REF_KEYS], np.float32)
        for g in range(3):
            ref[:, g] = np.interp(np.arange(WINDOW_TICKS), [k for k, _ in REF_KEYS], vals[:, g])
        self._ref = torch.from_numpy(np.tile(ref, (1, 4))).to(self.device)   # [window ticks, 12], add order per leg
        self.trigger_s = (float(trigger_s), float(trigger_s)) if np.ndim(trigger_s) == 0 else tuple(map(float, trigger_s))
        K, dev = num_envs, self.device
        self.trig = torch.full((K,), -1, dtype=torch.long, device=dev)     # trigger tick (-1: none this episode)
        self.dx_cmd = torch.zeros(K, device=dev)
        self.bucket = torch.zeros(K, dtype=torch.long, device=dev)
        self.crouched = torch.zeros(K, dtype=torch.bool, device=dev)
        self.jstate = self.env_state((), init=PEND, dtype=torch.long)
        self.air_run = self.env_state((), init=0, dtype=torch.long)         # airborne ticks of the current flight
        self.flight_ticks = self.env_state((), init=0, dtype=torch.long)    # the counted flight's length
        self.jumped = self.env_state((), init=False, dtype=torch.bool)
        self.landed_up = self.env_state((), init=False, dtype=torch.bool)
        self.zmax = self.env_state(())
        self.vzmax = self.env_state(())
        self.z_trig = self.env_state(())
        self.xy_trig = self.env_state((2,))
        self.fwd = self.env_state((2,))
        self.xy_takeoff = self.env_state((2,))
        self.clear_max = self.env_state(())
        self.dx_ach = self.env_state(())
        self.uncmd = self.env_state((), init=0, dtype=torch.long)
        self.fell = self.env_state((), init=False, dtype=torch.bool)
        self._rise = torch.zeros(K, device=dev)
        self._vz_gain = torch.zeros(K, device=dev)
        self._fly_tick = torch.zeros(K, dtype=torch.bool, device=dev)
        self._uncmd_tick = torch.zeros(K, dtype=torch.bool, device=dev)
        self._land_bonus = torch.zeros(K, device=dev)
        self._miss = torch.zeros(K, dtype=torch.bool, device=dev)
        self._in_win = torch.zeros(K, dtype=torch.bool, device=dev)
        self._foot_z = torch.zeros(K, 4, device=dev)
        self._jstats = torch.zeros(len(BUCKETS), NSTAT, dtype=torch.float64, device=dev)
        self._jmax = torch.zeros(len(BUCKETS), 2, dtype=torch.float64, device=dev)          # max rise, max clearance
        self._qdw_hist = torch.zeros(3, QD_BINS, dtype=torch.float64, device=dev)
        self._qdw_peak = torch.zeros(3, device=dev)

    # ---- resets -----------------------------------------------------------------------------------------------
    def on_reset(self, idx):
        n, dev = idx.numel(), self.device
        ang = self._rand(n) * math.radians(3)
        a = self._rand(n) * 2 * math.pi
        tilt = quat_axis_angle(torch.stack([torch.cos(a), torch.sin(a), torch.zeros_like(a)], dim=1), ang)
        yaw = quat_axis_angle(torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(n, 3), self._rand(n) * 2 * math.pi)
        quat = quat_mul(yaw, tilt)
        joints = (self.stance + 0.03 * self._randn(n, 12)).clamp(self.q_lo, self.q_hi)
        jump = self._rand(n) < self.jump_frac
        lo, hi = self.trigger_s
        t = lo + self._rand(n) * (hi - lo)
        zero = self._rand(n) < DX_ZERO_FRAC
        dx = torch.where(zero, 0.0, self._rand(n) * self.dx_max)
        trig = torch.where(jump, (t * CONTROL_HZ).round().long(), -1)
        bucket = torch.where(~jump, 0, torch.where(zero, 1, torch.where(dx <= 0.2, 2, 3)))
        crouch = torch.zeros(n, dtype=torch.bool, device=dev)
        if self.crouch_frac > 0:                # drawn only when on, so the pilot's random stream is untouched
            crouch = jump & (self._rand(n) < self.crouch_frac)
            u = self._rand(n)
            off = torch.tensor(CROUCH_Q, device=dev).repeat(4)             # add order: per leg hx, hy, kn
            joints = torch.where(crouch[:, None], (joints + u[:, None] * off).clamp(self.q_lo, self.q_hi), joints)
            clo, chi = CROUCH_TRIGGER_S
            tc = clo + self._rand(n) * (chi - clo)
            trig = torch.where(crouch, (tc * CONTROL_HZ).round().long(), trig)
            bucket = torch.where(crouch, len(BUCKETS) - 1, bucket)
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :4] = quat
        pose[:, 5] = self.lane_y[idx]
        pose[:, 6] = 0.005 - lowest_point(quat, joints)
        self.sim.set_root_state(idx, pose)
        self.sim.set_joint_state(idx, joints, torch.zeros(n, self.sim.dof, device=dev))
        self._blowup[idx] = False
        # the rate limiter starts from the spawn joints (there are no limp ticks to hand it the landed ones)
        self.last_act[idx] = (joints[:, self.i2a] - self.default_q) / self.action_scale
        self._tgt[idx] = joints
        self.trig[idx] = trig
        self.dx_cmd[idx] = torch.where(jump, dx, 0.0)
        self.bucket[idx] = bucket
        self.crouched[idx] = crouch

    # ---- step -------------------------------------------------------------------------------------------------
    def _window(self):
        t = self.steps - self.trig
        return (self.trig >= 0) & (t >= 0) & (t < WINDOW_TICKS), t

    def on_step(self, s):
        super().on_step(s)
        lp = self.sim.link_pose
        ll = lp[:, 9:13]                                                    # lower legs, capsule axis +Y
        x, y, z, w = ll[..., 0], ll[..., 1], ll[..., 2], ll[..., 3]
        ax_z = 2 * (y * z + w * x)
        foot_z = ll[..., 6] - SEG_HALF * ax_z.abs() - SHIN_R
        self._foot_z.copy_(foot_z)
        air = (foot_z > AIR_Z).all(dim=1)
        in_win, t = self._window()
        self._in_win.copy_(in_win)
        bz = self.sim.root_position[:, 2]
        bxy = self.sim.root_position[:, :2]
        st = self.jstate
        # at the trigger: reference height, position and heading
        first = in_win & (t == 0)
        f = quat_rot(self.sim.root_quat, torch.tensor([1.0, 0.0, 0.0], device=self.device).expand(self.K, 1, 3))[:, 0, :2]
        f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.z_trig.copy_(torch.where(first, bz, self.z_trig))
        self.xy_trig.copy_(torch.where(first[:, None], bxy, self.xy_trig))
        self.fwd.copy_(torch.where(first[:, None], f, self.fwd))
        # rise: the running max base height since the trigger, while the jump is pending or in flight
        rising = (self.trig >= 0) & (t >= 0) & ((st == PEND) & in_win | (st == FLY))
        new_max = torch.where(first, bz, torch.maximum(self.zmax, bz))
        self._rise.copy_(torch.where(rising & ~first, new_max - self.zmax, 0.0))
        self.zmax.copy_(torch.where(rising, new_max, self.zmax))
        # peak upward base speed since the trigger: w_vz pays its increase; a metric always
        vz = torch.nan_to_num(self.sim.root_linvel[:, 2], nan=0.0).clamp(0.0, VZ_CAP)
        old_v = torch.where(first, 0.0, self.vzmax)
        new_v = torch.maximum(old_v, vz)
        self._vz_gain.copy_(torch.where(rising, new_v - old_v, 0.0))
        self.vzmax.copy_(torch.where(rising, new_v, self.vzmax))
        # flight state machine
        takeoff = (st == PEND) & in_win & air & (self.sim.root_linvel[:, 2] > TAKEOFF_VZ)
        self.xy_takeoff.copy_(torch.where(takeoff[:, None], bxy, self.xy_takeoff))
        flying = (st == FLY) | takeoff
        self.air_run.copy_(torch.where(flying & air, self.air_run + 1, 0))
        self._fly_tick.copy_(flying & air)
        self.clear_max.copy_(torch.where(flying & air, torch.maximum(self.clear_max, foot_z.amin(dim=1)), self.clear_max))
        touch = (st == FLY) & ~air
        counted = touch & (self.flight_ticks >= MIN_FLIGHT_TICKS)
        short = touch & ~counted
        up_ok = self.up > LAND_UP
        dx = ((bxy - self.xy_takeoff) * self.fwd).sum(dim=1)
        self.dx_ach.copy_(torch.where(counted, dx, self.dx_ach))
        self.jumped.copy_(self.jumped | counted)
        self.landed_up.copy_(self.landed_up | (counted & up_ok))
        err = dx - self.dx_cmd
        self._land_bonus.copy_(torch.where(counted & up_ok,
                                           W_LAND + W_DIST * torch.exp(-(err / DIST_SIGMA) ** 2), 0.0))
        # a flight shorter than MIN_FLIGHT_TICKS re-arms the jump while the window is open (else the window is missed)
        self.flight_ticks.copy_(torch.where(flying & air, self.flight_ticks + 1, torch.where(short, 0, self.flight_ticks)))
        nst = torch.where(takeoff, FLY, st)
        nst = torch.where(counted, LANDED, nst)
        nst = torch.where(short, torch.where(in_win, PEND, MISSED), nst)
        closing = ((self.trig >= 0) & (t == WINDOW_TICKS) & (nst == PEND)) | (short & ~in_win)
        self._miss.copy_(closing)
        nst = torch.where(closing, MISSED, nst)
        self.jstate.copy_(nst)
        live = self.steps > SETTLE_TICKS
        self._uncmd_tick.copy_(air & ~(flying & air) & live)
        self.uncmd += self._uncmd_tick.long()
        # joint speed over window ticks only (per joint group), a metric
        live_w = in_win
        if self.count_episodes:
            live_w = live_w & (self.ep_index < self.count_episodes)
        jva = torch.nan_to_num(self.sim.joint_vel.abs(), nan=0.0, posinf=1e3).view(-1, 4, 3)
        qbins = torch.clamp((jva / QD_W).long(), 0, QD_BINS - 1).permute(2, 0, 1).reshape(3, -1)
        self._qdw_hist.scatter_add_(1, qbins, live_w.double()[:, None].expand(-1, 4).reshape(1, -1).expand(3, -1).contiguous())
        self._qdw_peak.copy_(torch.maximum(self._qdw_peak, (jva.amax(dim=1) * live_w[:, None]).amax(dim=0)))

    def terminated(self, s):
        live = self.steps > SETTLE_TICKS
        fall = live & ((self.up < FALL_UP) | (self.sim.root_position[:, 2] < FALL_Z))
        self.fell.copy_(self.fell | fall)
        return self._blowup | fall

    def reward_terms(self, s, a):
        live = (self.steps > SETTLE_TICKS).float()
        base = super().reward_terms(s, a)                                   # upright, height, pose, still, effort, impact
        stand = (~self._in_win).float()
        for k in ("height", "pose", "still"):
            base[k] = base[k] * stand
        terms = {
            "rise": W_RISE * self._rise,
            "flight": W_FLIGHT * self._fly_tick.float(),
            "land": self._land_bonus,
            "miss": -W_MISS * self._miss.float(),
            "uncmd": -W_UNCMD * self._uncmd_tick.float(),
            "fall": -W_FALL * (s.terminated & ~self._blowup).float(),
        }
        if self.w_vz > 0:
            terms["vz"] = self.w_vz * self._vz_gain
        if self.w_imit > 0:
            in_win, t = self._window()
            k = t.clamp(0, WINDOW_TICKS - 1)
            err = (self.sim.joint_pos - self.stance - self._ref[k]).pow(2).mean(dim=1)
            terms["imit"] = self.w_imit * in_win.float() * torch.exp(-err / IMIT_SIGMA2)
        base.update({k: v * live for k, v in terms.items()})
        return base

    def observe(self, s):
        obs = super().observe(s)
        in_win, t = self._window()
        w = in_win.float()
        obs[:, 9] = self.dx_cmd * w
        obs[:, 48] = w
        obs[:, 49] = (t.float() / WINDOW_TICKS).clamp(0.0, 1.0) * w
        self._last_obs.copy_(obs)
        return obs

    # ---- episode ends -----------------------------------------------------------------------------------------
    def on_done(self, idx):
        j = self.jumped[idx]
        jf = j.double()
        stood = self.succ_run[idx] >= SUCCESS_HOLD
        fell = self.fell[idx] | self._blowup[idx]
        rise = (self.zmax[idx] - self.z_trig[idx]).clamp_min(0.0)
        m = torch.stack([torch.ones_like(jf), jf, self.landed_up[idx].double(),
                         (self.landed_up[idx] & stood & ~fell).double(), fell.double(),
                         jf * self.flight_ticks[idx].double() * DT, jf * rise.double(), jf * self.clear_max[idx].double(),
                         jf * self.dx_ach[idx].double(), jf * (self.dx_ach[idx] - self.dx_cmd[idx]).abs().double(),
                         self.uncmd[idx].double(), self.vzmax[idx].double(), rise.double()], dim=1)
        # a no-jump episode succeeds by standing at the end without falling or leaving the ground
        nj = self.trig[idx] < 0
        m[:, 3] = torch.where(nj, (stood & ~fell & (self.uncmd[idx] == 0)).double(), m[:, 3])
        if self.count_episodes:
            keep = (self.ep_index[idx] < self.count_episodes).double()
            m = m * keep[:, None]
        else:
            keep = torch.ones_like(jf)
        b = self.bucket[idx]
        self._jstats.index_add_(0, b, m)
        mx = torch.stack([jf * rise.double() * keep, jf * self.clear_max[idx].double() * keep], dim=1)
        self._jmax.index_reduce_(0, b, mx, "amax")
        self.ep_index[idx] += 1

    def jump_stats(self, reset=True):
        s, mx = self._jstats.cpu().numpy(), self._jmax.cpu().numpy()
        div = lambda a, b: (float(a) / float(b)) if b else None
        out = {"by_bucket": {}}
        for i, name in enumerate(BUCKETS):
            e, jn = s[i, 0], s[i, 1]
            if not e:
                continue
            out["by_bucket"][name] = {"episodes": int(e), "jumped": div(jn, e), "landed_up": div(s[i, 2], e),
                                      "success": div(s[i, 3], e), "fell": div(s[i, 4], e),
                                      "flight_s": div(s[i, 5], jn), "rise_m": div(s[i, 6], jn),
                                      "clearance_m": div(s[i, 7], jn), "dx_m": div(s[i, 8], jn),
                                      "dx_err_m": div(s[i, 9], jn), "uncmd_air_ticks_per_ep": div(s[i, 10], e),
                                      "rise_max_m": float(mx[i, 0]), "clearance_max_m": float(mx[i, 1]),
                                      "vz_max": div(s[i, 11], e), "rise_all_m": div(s[i, 12], e)}
        tj = s[1:4].sum(axis=0)                                            # standing starts
        out["jump"] = {"episodes": int(tj[0]), "jumped": div(tj[1], tj[0]), "landed_up": div(tj[2], tj[0]),
                       "success": div(tj[3], tj[0]), "fell": div(tj[4], tj[0]), "flight_s": div(tj[5], tj[1]),
                       "rise_m": div(tj[6], tj[1]), "clearance_m": div(tj[7], tj[1]), "dx_err_m": div(tj[9], tj[1]),
                       "uncmd_air_ticks_per_ep": div(tj[10], tj[0]), "vz_max": div(tj[11], tj[0]),
                       "rise_all_m": div(tj[12], tj[0])}
        rec = self.recovery_stats(reset=reset)
        for k in ("qd", "cmd_qd_max_hx_hy_kn", "tau_p95_hx_hy_kn", "qd_excess", "spawn"):
            if k in rec:
                out[k] = rec[k]
        hq, pk = self._qdw_hist.cpu().numpy(), self._qdw_peak.cpu().numpy()
        qd = {}
        for g, name in enumerate(JOINT_GROUPS):
            c = np.cumsum(hq[g]); tot = c[-1]
            pct = [float((np.searchsorted(c, p * tot) + 1) * QD_W) if tot else None for p in (0.95, 0.99)]
            qd[name] = {"samples": int(tot), "p95": pct[0], "p99": pct[1], "max": float(pk[g]) if tot else None,
                        **{f"over{int(v)}": (float(hq[g, int(round(v / QD_W)):].sum() / tot) if tot else None)
                           for v in QD_OVER}}
        out["qd_window"] = qd
        if reset:
            self._jstats.zero_(); self._jmax.zero_(); self._qdw_hist.zero_(); self._qdw_peak.zero_()
        return out

    def knob_config(self):
        return {**super().knob_config(), "jump_frac": self.jump_frac, "dx_max": self.dx_max,
                "trigger_s": list(self.trigger_s), "w_vz": self.w_vz, "crouch_frac": self.crouch_frac,
                "w_imit": self.w_imit}

    def config(self):
        return {**super().config(), **CONFIG, "self_collision": self.self_collision, **self.knob_config()}


def jump_mirror_obs(obs):
    """spot_steps_symmetry's mirror, except the clock block: the walking gait's clock mirrors as a half-period shift
    (negated), but the jump signal (1, p) is the same on both sides of the robot."""
    from spot_steps_symmetry import mirror_obs
    o = mirror_obs(obs)
    o[..., 48:50] = obs[..., 48:50]
    return o


def make_jump_aux_loss(coef):
    import _common
    from spot_symmetry import mirror_act
    return _common.make_aux_loss(coef, jump_mirror_obs, mirror_act)


if __name__ == "__main__":
    # smoke + open-loop feasibility: stand, then at the trigger crouch and extend at the rate limit
    K = int(os.environ.get("K", "64"))
    QD = float(os.environ.get("QD", "10"))
    env = SpotJumpEnv(num_envs=K, qd_max=QD, jump_frac=1.0, trigger_s=0.5, count_episodes=1)
    obs = env.reset()
    assert obs.shape == (K, OBS_DIM)
    dev = env.device
    crouch = torch.zeros(ACT_DIM, device=dev)
    ext = torch.zeros(ACT_DIM, device=dev)
    crouch_d = torch.tensor([0.0, 0.35, -0.7], device=dev)      # hx, hy, kn offsets (rad) from the stance
    ext_d = torch.tensor([0.0, -0.25, 0.8], device=dev)
    # Isaac order: all hx, all hy, all kn
    for g in range(3):
        crouch[4 * g:4 * g + 4] = crouch_d[g] / env.action_scale
        ext[4 * g:4 * g + 4] = ext_d[g] / env.action_scale
    fz0 = None
    for i in range(int(EPISODE_S * CONTROL_HZ) + 2):
        t = i - 25
        a = torch.zeros(K, ACT_DIM, device=dev)
        if 0 <= t < 20:
            a[:] = crouch
        elif 20 <= t < 40:
            a[:] = ext
        obs, rew, done, _, _ = env.step(a)
        if i == 10:
            fz0 = env._foot_z.mean().item()
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    print(f"foot bottom z standing (mean over feet and envs): {fz0:.4f} m (AIR_Z {AIR_Z})")
    print(env.stats_line())
    st = env.jump_stats()
    print({k: st[k] for k in ("jump", "qd_window", "tau_p95_hx_hy_kn")})
    print("SPOT JUMP ENV SELFTEST: PASS")
