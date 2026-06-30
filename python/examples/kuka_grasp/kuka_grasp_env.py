"""kuka_grasp_env.py — the KUKA grasp RL environment (threepp.rl contract).

The policy commands all 7 arm joints + the gripper (8 actions) to reach a cube on the table, close
the parallel-jaw gripper, and lift it. Reward is a sum of shaped terms (reach → down → centre →
descend → grasp → lift → success). A real FRICTION grasp is the goal — there is NO grasp-lock weld.
The bootstrap is a graduated reset-height curriculum (start_high 0→1): early episodes start with the arm
already straddling the cube (learn the grasp), then the start height spreads up toward the DEFAULT_Q home
so the descent is trained at every depth.

Contract (see python/threepp/rl/README.md):
    reset() -> obs [K, OBS_DIM]
    step(action [K, ACT_DIM]) -> (obs, reward, done, terminal_obs, is_timeout)
`step` auto-resets finished envs; `is_timeout` marks time-limit truncations (bootstrap V(term_obs))
vs. true terminals (cube knocked off the table → bootstrap 0).
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import kuka_grasp_contract as C  # noqa: E402
from kuka_grasp_sim import KukaGraspSim  # noqa: E402

# --------------------------------------------------------------------------- #
#  Reward weights + grasp thresholds (tunable; kept here, not scattered)
# --------------------------------------------------------------------------- #
W_REACH = 1.0
K_REACH = 6.0          # exp(-K_REACH * ||tip - cube||)
W_DOWN = 1.0           # reward the TOOL pointing straight down (DESIGN's "align tool-axis down" term)
K_DOWN = 3.0           # exp(-K_DOWN * (1 - down)); down = -approach_z ∈ [-1,1], 1 = straight down
W_CENTER = 0.6
K_CENTER = 22.0        # exp(-K_CENTER * lateral_xy) — centre the tip over the cube (sharp)
W_VERT = 1.5           # reward tip descending to cube Z level (the missing descent gradient)
K_VERT = 10.0          # exp(-K_VERT * max(0, tip_z - cube_z))
W_CLOSE = 0.6          # dense: reward closing the fingers while positioned over the cube
W_GRASP = 1.5          # per-step bonus for a real FIRM grasp
W_LIFT = 24.0          # per metre of lift, GATED ON A FIRM GRASP (not a loose "held")
W_SUCCESS = 6.0        # per-step bonus while holding the cube above the success height
W_ACT_RATE = 0.004
W_ACT_MAG = 0.001
W_LIMIT = 0.08
W_FALL = 3.0           # one-time penalty when the cube is knocked off the table

# A FIRM grasp = tightly centred AND fingers closed on the cube. The lift + success rewards require
# THIS (not a loose proximity), so the only way to earn the lift reward is a real, centred squeeze
# that holds the cube by friction. (There is no weld to fake it — see step().)
FIRM_LAT = 0.018       # lateral tip↔cube must be tight (was 0.035 — too loose, off-centre grips slip)
FIRM_VERT = 0.045      # fingers vertically around the cube
FIRM_OPEN = 0.018      # fingers genuinely closed on the cube
OVER_LAT = 0.045       # loosely above the cube — gates the dense "close" reward
OVER_VERT = 0.045      # close reward fires ONLY in the grasp zone (fingers around the cube). Looser values
                       # reward closing while still high up — and once shut the fingers (gap 0.044 < cube
                       # 0.07) can't descend, so the arm grips the air above the cube.
LOST_MARGIN = 0.10     # cube this far beyond the table edge == knocked off → terminal fail

ARM_RESET_NOISE = 0.05  # rad jitter on the arm reset posture


class KukaGraspEnv:
    def __init__(self, num_envs=2048, device="cuda", seed=0, spacing=2.5, render_visuals=False):
        self.sim = KukaGraspSim(num_envs, device=device, spacing=spacing, render_visuals=render_visuals)
        self.K = num_envs
        self.device = self.sim.device
        self.max_steps = C.MAX_STEPS
        self.g = torch.Generator(device=self.device).manual_seed(seed)

        dev = self.device
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self.last_action = torch.zeros(num_envs, C.ACT_DIM, device=dev)
        self.default_q = torch.tensor(C.DEFAULT_Q, device=dev)
        self.arm_lo = torch.tensor(C.ARM_LIMITS[:, 0], device=dev)
        self.arm_hi = torch.tensor(C.ARM_LIMITS[:, 1], device=dev)
        self.arm_scale = torch.tensor(C.ARM_ACTION_SCALE, device=dev)
        self.env_x = self.sim.env_x

        # curriculum state (set via set_iter)
        self.spawn_half = torch.tensor([0.0, 0.0], device=dev)   # cube spawn region half-extent (x,y)
        self.start_high = 0.0      # graduated reset: 0 → all envs start AT the cube; 1 → up to the home pose
        self.deploy = False        # eval/deploy: force the true home start (df=0)

        # descended grasp posture (tip down at the cube) for the reset-posture curriculum — the
        # scaffold that lets the policy ever EXPERIENCE the grasp+lift states (see _init_grasp_posture).
        self._init_grasp_posture()

        # metrics (read in on_log)
        self.last_reach = 0.0
        self.last_grasp_rate = 0.0
        self.last_lift = 0.0
        self.last_success_rate = 0.0

        self.set_iter(0)

    def _init_grasp_posture(self):
        """The descended grasp posture (arm posed so the open fingers straddle a centred cube) used by
        the reset-posture curriculum. Found once via the same posture search the friction gate uses
        (kuka_grasp_friction.py), then CACHED to disk so later runs are instant + deterministic.

        Without this scaffold the policy resets only ever start at DEFAULT_Q (hovering ~9 cm above the
        cube); the coordinated descent to the cube is too hard to discover by exploration, so grasp/lift
        states are never visited and never learned. Starting a (annealed) fraction of episodes already
        in the straddle teaches close→grasp→lift directly and propagates value back to teach the reach.

        CRITICAL: the posture must genuinely DESCEND the tip to the cube. The search is stochastic and a
        single seed can return a near-hover (high tip) that ranks well only because the PD arm hadn't
        settled when it was scored. So we settle FULLY, measure the achieved tip, and accept only a
        posture whose tip is actually at the cube (xy + z within tolerance) — retrying seeds, and failing
        LOUDLY rather than silently training on a no-op scaffold."""
        cache = os.path.join(_HERE, "kuka_grasp_pose.npy")
        target = np.array([C.TABLE_CX, C.TABLE_CY, C.CUBE_REST_Z], np.float32)
        XY_TOL, Z_TOL = 0.030, 0.040            # tip must straddle the 0.07 m cube (half=0.035)
        if os.path.exists(cache):
            data = np.load(cache).astype(np.float32)
            q7, tip = data[0:7], data[7:10]
        else:
            best = None                          # (err, q7, tip) — keep the best across seeds for the error msg
            q7 = tip = None
            for seed in range(12):
                cand = self.sim.search_posture(tuple(target), n=200, settle=80, seed=seed, grip=C.GRIP_OPEN)
                if cand is None:
                    continue
                self.sim.settle_arm(list(cand) + [C.GRIP_OPEN, C.GRIP_OPEN], 150)  # settle FULLY before measuring
                t = self.sim.tip_state()[0][0].detach().cpu().numpy().astype(np.float32)
                xy_err = float(np.linalg.norm(t[0:2] - target[0:2]))
                z_err = float(abs(t[2] - target[2]))
                err = xy_err + z_err
                print(f"[kuka_grasp] grasp-posture seed {seed}: tip={np.round(t,3).tolist()} "
                      f"xy_err={xy_err:.3f} z_err={z_err:.3f}")
                if best is None or err < best[0]:
                    best = (err, np.asarray(cand, np.float32), t)
                if xy_err < XY_TOL and z_err < Z_TOL:
                    q7, tip = np.asarray(cand, np.float32), t
                    break
            if q7 is None:
                be = best[2] if best else None
                raise RuntimeError(
                    "kuka_grasp: could not find a posture that DESCENDS the tip to the cube "
                    f"(target {target.tolist()}, best achieved tip {None if be is None else be.tolist()}). "
                    "The reset-posture scaffold would be a no-op — fix the geometry/reach before training.")
            np.save(cache, np.concatenate([q7, tip]))
        print(f"[kuka_grasp] grasp posture ACCEPTED: q7={np.round(q7,3).tolist()} tip(fingers)={np.round(tip,3).tolist()} "
              f"(must be ~[{C.TABLE_CX}, {C.TABLE_CY}, {C.CUBE_REST_Z:.2f}] — the FINGER midpoint at the cube)")
        self.grasp_q = torch.tensor(q7, device=self.device, dtype=torch.float32)        # [7] arm posture
        self.grasp_cube_xy = torch.tensor(tip[0:2], device=self.device, dtype=torch.float32)  # [2] tip xy (env-local)

    # ---- curriculum ---------------------------------------------------------
    def set_iter(self, it):
        """Curriculum schedule. Early: cube near the table centre, episodes start AT the cube (learn the
        grasp). Later: the arm's start height spreads up toward the DEFAULT_Q home, so the DESCENT is
        practised at every depth, and the cube spreads across the table."""
        t = min(max(it / 1200.0, 0.0), 1.0)        # cube spawn region grows 0→full over 1200 iters
        sx = 0.02 + t * (C.TABLE_HALF[0] - 0.06)
        sy = 0.02 + t * (C.TABLE_HALF[1] - 0.06)
        self.spawn_half = torch.tensor([sx, sy], device=self.device)
        # graduated reset height: 0 → every env starts AT the cube (grasp_q); grows to 1 → start anywhere
        # between the cube and the home pose. Trains the last-mile descent at all depths (not one plunge),
        # while keeping grasp-zone starts alive — the scaffold replacing the removed grasp-lock weld.
        self.start_high = float(min(it / 500.0, 1.0))
        self.cur_iter = it

    # ---- observation --------------------------------------------------------
    def _grasp_state(self):
        """Return (tip, cube, lateral_xy, vert, dist, opening, firm, approach) describing the tip↔cube
        relation. `firm` = a real grasp: tightly centred + fingers closed on the cube. `approach` is the
        tool approach axis (world); -approach_z is "down-ness" (1 = pointing straight down)."""
        tip, approach = self.sim.tip_state()
        cube = self.sim.cube_position()
        d = cube - tip
        lateral = torch.linalg.norm(d[:, 0:2], dim=1)
        vert = d[:, 2].abs()
        dist = torch.linalg.norm(d, dim=1)
        opening = self.sim.finger_opening()
        firm = (lateral < FIRM_LAT) & (vert < FIRM_VERT) & (opening < FIRM_OPEN)
        return tip, cube, lateral, vert, dist, opening, firm, approach

    def _obs(self):
        s = self.sim
        tip, cube, lateral, vert, dist, opening, firm, approach = self._grasp_state()
        # tip relative to this env's table centre (strip the per-env X offset)
        tip_rel = tip.clone()
        tip_rel[:, 0] -= self.env_x + C.TABLE_CX
        tip_rel[:, 1] -= C.TABLE_CY
        tip_rel[:, 2] -= C.TABLE_TOP_Z
        tip_to_cube = cube - tip                       # env_x cancels
        lift_frac = ((cube[:, 2] - C.CUBE_REST_Z) / C.LIFT_SUCCESS_DZ).clamp(0.0, 1.5)
        flags = torch.stack([firm.float(), lift_frac], dim=1)
        finger_vel = 0.5 * (s.arm_jv[:, 7] + s.arm_jv[:, 8])
        obs = torch.cat([
            s.arm_jp[:, 0:7] - self.default_q,         # 7 arm qpos_rel
            s.arm_jv[:, 0:7] * C.QVEL_SCALE,           # 7 arm qvel
            torch.stack([opening, finger_vel * C.QVEL_SCALE], dim=1),  # 2 gripper
            tip_rel,                                   # 3 tip (table-relative)
            tip_to_cube,                               # 3 tip→cube
            approach,                                  # 3 tool approach axis (so the policy SEES orientation)
            s.cube_up(),                               # 3 cube up-axis
            s.cube_linvel,                             # 3 cube linvel
            self.last_action,                          # 8 last action
            flags,                                     # 2 grasp flag + lift frac
        ], dim=1)
        return obs

    # ---- reset --------------------------------------------------------------
    def _reset_idx(self, idx):
        n = idx.numel()
        if n == 0:
            return
        dev = self.device
        g = self.g

        # GRADUATED reset height. df=1 starts the arm AT the cube (grasp_q, open fingers straddling);
        # df=0 at the DEFAULT_Q home ~22 cm above. The start height spreads up to home over the curriculum
        # (start_high), so the last-mile descent is trained at EVERY depth — not only as one full plunge —
        # and grasp-zone starts never vanish. Deploy/eval forces df=0 (the true home start).
        if self.deploy:
            df = torch.zeros(n, 1, device=dev)
        else:
            u = torch.rand(n, generator=g, device=dev) * self.start_high
            df = (1.0 - u).unsqueeze(1)                 # [n,1] in [1-start_high, 1]

        q9 = torch.zeros(n, C.N_DOF, device=dev)
        q9[:, 0:7] = (self.default_q + df * (self.grasp_q - self.default_q)
                      + ARM_RESET_NOISE * (torch.rand(n, 7, generator=g, device=dev) * 2 - 1))
        q9[:, 7] = C.GRIP_OPEN
        q9[:, 8] = C.GRIP_OPEN
        self.sim.set_arm_joint_state(idx, q9, torch.zeros(n, C.N_DOF, device=dev))

        # cube under the descended fingers (grasp_cube_xy); lateral spread GROWS as the arm starts higher
        # (df→0), so a start-at-cube env gets a clean straddle while a home-start env must also centre on a
        # cube placed anywhere in the (curriculum-widening) spawn region.
        spread = (1.0 - df.squeeze(1))
        rx = (torch.rand(n, generator=g, device=dev) * 2 - 1) * self.spawn_half[0] * spread
        ry = (torch.rand(n, generator=g, device=dev) * 2 - 1) * self.spawn_half[1] * spread
        cx = self.env_x[idx] + self.grasp_cube_xy[0] + rx
        cy = self.grasp_cube_xy[1] + ry
        yaw = (torch.rand(n, generator=g, device=dev) * 2 - 1) * np.pi * spread
        pos = torch.stack([cx, cy, torch.full((n,), C.CUBE_REST_Z, device=dev)], dim=1)
        quat = torch.zeros(n, 4, device=dev)
        quat[:, 2] = torch.sin(yaw / 2)   # qz
        quat[:, 3] = torch.cos(yaw / 2)   # qw
        pose = torch.cat([quat, pos], dim=1)
        self.sim.set_cube_state(idx, pose)

        self.steps[idx] = 0
        self.last_action[idx] = 0.0

    def reset(self):
        self._reset_idx(torch.arange(self.K, device=self.device))
        self.sim.read()
        return self._obs()

    # ---- step ---------------------------------------------------------------
    @torch.no_grad()
    def step(self, action):
        a = action.clamp(-1.0, 1.0)
        # arm targets (add-order joints 0..6) + mirrored gripper (7,8). Gripper DEFAULTS OPEN: a7<=0 keeps
        # the fingers fully open (clears the cube during the descent); only a7>0 closes. This makes the
        # action-penalty-neutral pose open, so the arm doesn't half-shut the gripper and jam on the cube.
        arm_tgt = torch.clamp(self.default_q + self.arm_scale * a[:, 0:7], self.arm_lo, self.arm_hi)
        grip = C.GRIP_OPEN + a[:, 7].clamp(0.0, 1.0) * (C.GRIP_CLOSE - C.GRIP_OPEN)   # a7<=0 open, a7=+1 close
        targets = torch.cat([arm_tgt, grip.unsqueeze(1), grip.unsqueeze(1)], dim=1)
        self.sim.set_arm_targets(targets)
        self.sim.substep(C.SUBSTEPS)

        tip, cube, lateral, vert, dist, opening, firm, approach = self._grasp_state()
        over_cube = (lateral < OVER_LAT) & (vert < OVER_VERT)
        close_amt = ((C.GRIP_OPEN - opening) / (C.GRIP_OPEN - C.GRIP_CLOSE)).clamp(0.0, 1.0)
        down = -approach[:, 2]        # tool down-ness: 1 = straight down, 0 = horizontal, -1 = up

        # NO grasp-lock weld. The earlier latch (teleport the cube to the tip once firm, hold it there)
        # let the policy fake the task: brief firm → latch → raise the arm → the weld carried the cube up
        # and paid the lift reward WITHOUT a real grip. Result: succ ~0.6 but real grasp ~0.1, and in
        # deploy (no weld) the arm just lifted away from the cube. The pre-pose curriculum already gives
        # the bootstrap the weld was meant to provide (episodes that START straddling the box) — but here
        # lift only pays off if the fingers actually hold the cube by FRICTION, which is the real skill.
        firm_gate = firm                 # lift/success require a genuine firm friction grasp, full stop

        self.steps += 1
        lift = (cube[:, 2] - C.CUBE_REST_Z).clamp(min=0.0)
        success = firm_gate & (cube[:, 2] > C.CUBE_REST_Z + C.LIFT_SUCCESS_DZ)
        # terminal fail: cube knocked below the table OR batted off the table footprint. The lateral
        # check is essential — without it a cube swept sideways never resets and its distance inflates
        # the reach metric (and starves the reward) for the rest of the episode.
        off_x = (cube[:, 0] - (self.env_x + C.TABLE_CX)).abs() > (C.TABLE_HALF[0] + LOST_MARGIN)
        off_y = (cube[:, 1] - C.TABLE_CY).abs() > (C.TABLE_HALF[1] + LOST_MARGIN)
        fell = (cube[:, 2] < (C.TABLE_TOP_Z - C.DROP_FAIL_DZ)) | off_x | off_y

        # joint-limit proximity (arm only)
        q = self.sim.arm_jp[:, 0:7]
        over = (q - 0.95 * self.arm_hi).clamp(min=0.0) + (0.95 * self.arm_lo - q).clamp(min=0.0)

        tip_above = (tip[:, 2] - cube[:, 2]).clamp(min=0.0)   # 0 when tip at or below cube Z level
        # GATE the descent reward on being laterally OVER the cube. Without this gate, vert is z-only:
        # the policy banks the full descent bonus by dropping to the cube's height ANYWHERE laterally,
        # and (1.5) it outweighs the centring term (0.6) — so it "descends to the side". over_xy → 0
        # off-centre, so descending only pays off above the cube (reach/centre pull it over first).
        over_xy = torch.exp(-12.0 * lateral)

        rew = (
            W_REACH * torch.exp(-K_REACH * dist)
            + W_DOWN * torch.exp(-K_DOWN * (1.0 - down))    # keep the TOOL pointing straight down
            + W_CENTER * torch.exp(-K_CENTER * lateral)
            + W_VERT * torch.exp(-K_VERT * tip_above) * over_xy   # descend, but only while OVER the cube
            + W_CLOSE * close_amt * over_cube.float()       # dense: close the fingers when over the cube
            + W_GRASP * firm.float()                        # firm-grasp bonus (REAL grasp only)
            + W_LIFT * lift * firm_gate.float()             # lift reward requires a genuine firm grasp
            + W_SUCCESS * success.float()
            - W_ACT_RATE * ((a - self.last_action) ** 2).sum(dim=1)
            - W_ACT_MAG * (a ** 2).sum(dim=1)
            - W_LIMIT * (over ** 2).sum(dim=1)
            - W_FALL * fell.float()
        )

        self.last_action = a.clone()

        timeout = self.steps >= self.max_steps
        done = timeout | fell
        is_timeout = timeout & ~fell           # fell is a true terminal (bootstrap 0)

        # metrics (grasp_rate / lift / success now all reflect a REAL firm grasp)
        self.last_reach = float(dist.mean())
        self.last_grasp_rate = float(firm.float().mean())          # REAL firm grasps
        self.last_lift = float((lift * firm_gate.float()).mean())
        self.last_success_rate = float(success.float().mean())     # REAL friction lifts (no weld) — honest now

        term_obs = self._obs()                 # snapshot BEFORE auto-reset
        d = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if d.numel() > 0:
            self._reset_idx(d)
            self.sim.read()
        obs = self._obs()
        return obs, rew, done, term_obs, is_timeout


# --------------------------------------------------------------------------- #
#  --selftest : shapes / no-NaN / cube rests / grasp detection
# --------------------------------------------------------------------------- #
def _selftest():
    torch.manual_seed(0)
    env = KukaGraspEnv(num_envs=8, render_visuals=False)
    obs = env.reset()
    assert obs.shape == (8, C.OBS_DIM), obs.shape
    assert torch.isfinite(obs).all(), "non-finite obs after reset"
    print(f"[selftest] obs {tuple(obs.shape)} OBS_DIM={C.OBS_DIM} ACT_DIM={C.ACT_DIM}  OK")

    # cube must rest on the table shortly after reset (not fall through / fly off)
    env.reset()
    for _ in range(10):
        env.step(torch.zeros(8, C.ACT_DIM, device=env.device))   # hold the default posture
    cube_z = env.sim.cube_position()[:, 2]
    assert (cube_z > C.TABLE_TOP_Z - 0.02).all(), f"cube fell through table: {cube_z}"
    print(f"[selftest] cube rests on table after reset  z={cube_z.mean():.3f} "
          f"(rest {C.CUBE_REST_Z:.3f})  OK")

    # a few random steps — check shapes, finiteness, reward sanity
    for k in range(20):
        a = torch.rand(8, C.ACT_DIM, device=env.device) * 2 - 1
        obs, rew, done, term, tmo = env.step(a)
        assert obs.shape == (8, C.OBS_DIM)
        assert rew.shape == (8,) and done.shape == (8,)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all(), f"non-finite at step {k}"
    print(f"[selftest] 20 random steps OK  reach={env.last_reach:.3f} "
          f"grasp_rate={env.last_grasp_rate:.3f}  reward[0]={float(rew[0]):.3f}")
    print("[selftest] (the friction-grasp physics gate is kuka_grasp_friction.py)")
    print("[selftest] PASS")


if __name__ == "__main__":
    _selftest()
