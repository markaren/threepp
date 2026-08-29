"""VecTask — the env-authoring base for GPU-vectorized RL tasks.

An env written on raw GpuSim spends most of its lines on choreography that is identical in
every task and easy to get subtly wrong: the steps/timeout bookkeeping, capturing terminal_obs
BEFORE the partial reset overwrites it, re-baselining every piece of per-env state (EMA
velocities, previous positions, phase clocks) on reset, and refreshing derived state for the
reset rows. VecTask owns all of that; a task defines only what is actually task-specific:

    class CartPoleEnv(VecTask):
        control_hz = 60
        episode_s = 10.0
        act_dim = 1
        control = "force"                    # or "drive" (PD position targets)

        def on_reset(self, idx): ...         # set sim state (+ own buffers) for envs `idx`
        def act(self, a): ...                # action [-1,1] -> [K, dof] force/target tensor
        def observe(self, s): ...            # s = the task's RobotState
        def reward_terms(self, s, a): ...    # dict of NAMED reward terms, each [K]
        def terminated(self, s): ...         # true-failure terminals (default: none)

The reward is a dict of named terms on purpose: the base sums them for the trainer AND keeps
per-term per-step episode means, surfaced through stats_line() into the PPO log line — so when
a policy converges to the wrong optimum you can see WHICH term dominates without adding prints.

Per-env state that must survive a step but reset with the episode (phase clocks, commands,
previous-action buffers) is registered with env_state(); the base masks it back to its init
value on every reset — full or partial — so a task cannot forget a re-baseline.

step()/reset() speak exactly the threepp.rl.PPO env protocol:
    reset() -> obs [K, obs_dim]
    step(a) -> (obs, reward, done, terminal_obs, is_timeout)
"""
import math

import torch

from .sim import GpuSim


def quat_to_frame(q):
    """q: [...,4] = (qx,qy,qz,qw). Returns (up_y, forward_x, forward_z, yaw): forward is the
    body +X axis projected to the ground plane (unit), up_y is the body +Y axis' world Y.
    Torch, batched, and usable on a single CPU quaternion too — deploy viewers reuse it
    verbatim so train and deploy measure orientation identically."""
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    up = 1.0 - 2.0 * (qx * qx + qz * qz)
    fx = 1.0 - 2.0 * (qy * qy + qz * qz)
    fz = 2.0 * (qx * qz - qw * qy)
    fl = torch.sqrt(fx * fx + fz * fz).clamp_min(1e-6)
    yaw = torch.atan2(2.0 * (qw * qy + qx * qz), 1.0 - 2.0 * (qy * qy + qz * qz))
    return up, fx / fl, fz / fl, yaw


class RobotState:
    """Derived robot state, refreshed once per control step from the GpuSim buffers.

    Always: joint_pos / joint_vel — GpuSim's live add-order views.
    With read_root: up, fwd_x, fwd_z, yaw (body frame), v_x / v_z (EMA world-frame velocity
    from finite differences), yaw_rate (EMA, wrap-safe), and the v_fwd / v_lat body-frame
    projections every locomotion reward wants. `terminated` is set by the VecTask loop right
    before reward_terms so a reward can reference the failure mask (e.g. a fall penalty).

    All tensors are persistent buffers updated in place, so references a task stashes stay
    live across steps — the same identity contract GpuSim gives for joint_pos/joint_vel.
    """

    def __init__(self, sim, dt, ema=0.2):
        self.sim, self.dt, self.ema = sim, dt, ema
        self.joint_pos, self.joint_vel = sim.joint_pos, sim.joint_vel
        self.terminated = torch.zeros(sim.K, dtype=torch.bool, device=sim.device)
        self.has_root = sim.read_root
        if self.has_root:
            z = lambda: torch.zeros(sim.K, device=sim.device)
            self.up, self.fwd_x, self.fwd_z, self.yaw = z(), z(), z(), z()
            self.v_x, self.v_z, self.yaw_rate = z(), z(), z()
            self._prev_px, self._prev_pz, self._prev_yaw = z(), z(), z()

    @property
    def root_pos(self):
        return self.sim.root_position

    @property
    def root_quat(self):
        return self.sim.root_quat

    @property
    def v_fwd(self):
        """Body-frame forward velocity (m/s), from the EMA world velocity."""
        return self.v_x * self.fwd_x + self.v_z * self.fwd_z

    @property
    def v_lat(self):
        """Body-frame lateral velocity (m/s) — the sideways slip locomotion rewards penalize."""
        return self.v_x * self.fwd_z - self.v_z * self.fwd_x

    def _orient(self):
        up, fx, fz, yaw = quat_to_frame(self.sim.root_quat)
        self.up.copy_(up); self.fwd_x.copy_(fx); self.fwd_z.copy_(fz)
        return yaw

    def update(self):
        """Refresh orientation and the finite-difference EMA velocities after a sim step."""
        if not self.has_root:
            return
        yaw = self._orient()
        px, pz = self.sim.root_position[:, 0], self.sim.root_position[:, 2]
        a = self.ema
        self.v_x.copy_((1.0 - a) * self.v_x + a * (px - self._prev_px) / self.dt)
        self.v_z.copy_((1.0 - a) * self.v_z + a * (pz - self._prev_pz) / self.dt)
        dy = torch.remainder(yaw - self._prev_yaw + math.pi, 2.0 * math.pi) - math.pi
        self.yaw_rate.copy_((1.0 - a) * self.yaw_rate + a * dy / self.dt)
        self._prev_px.copy_(px); self._prev_pz.copy_(pz); self._prev_yaw.copy_(yaw)
        self.yaw.copy_(yaw)

    def rebaseline(self, idx=None):
        """After a reset: refresh orientation and re-anchor the finite-difference state of the
        reset envs (idx=None -> all) to the CURRENT pose, so their first post-reset velocity
        sample is ~0 instead of a spawn-teleport spike."""
        if not self.has_root:
            return
        yaw = self._orient()
        self.yaw.copy_(yaw)
        px, pz = self.sim.root_position[:, 0], self.sim.root_position[:, 2]
        if idx is None:
            self._prev_px.copy_(px); self._prev_pz.copy_(pz); self._prev_yaw.copy_(yaw)
            self.v_x.zero_(); self.v_z.zero_(); self.yaw_rate.zero_()
        else:
            self._prev_px[idx] = px[idx]; self._prev_pz[idx] = pz[idx]
            self._prev_yaw[idx] = yaw[idx]
            self.v_x[idx] = 0.0; self.v_z[idx] = 0.0; self.yaw_rate[idx] = 0.0


class _NullState:
    """RobotState stand-in for sim-less tasks (build_robot=None): just the terminated mask."""

    def __init__(self, K, device):
        self.terminated = torch.zeros(K, dtype=torch.bool, device=device)

    def update(self):
        pass

    def rebaseline(self, idx=None):
        pass


class VecTask:
    """Base class owning the vectorized-env choreography. Subclasses set the class-level
    contract (control_hz, episode_s, act_dim, control mode) and implement the hooks; the base
    provides the PPO-protocol reset()/step(), auto-reset of registered env_state() buffers,
    RobotState maintenance, and per-term reward logging.

    Tasks whose world is not a GpuSim (e.g. a rendered pixel env with hand-rolled dynamics)
    pass build_robot=None and override simulate() instead of act(); they keep everything else
    (timeout bookkeeping, terminal-obs capture, auto-reset of env_state() buffers, reward-term
    logging).
    """

    # ---- the task contract (override as class attributes) ------------------------
    control_hz = 60          # policy control rate; dt = 1 / control_hz
    episode_s = 10.0         # time-limit truncation
    act_dim = None           # REQUIRED: action dimension
    control = "force"        # "force" -> apply_force, "drive" -> apply_drive_target (PD)
    substeps = 1             # physics substeps per control step (substep(): one read at the end)
    settle_steps = 0         # full-reset settle steps (zero action) before the first obs
    clip_actions = 1.0       # actions clamped to [-clip, clip]; None/0 = no clamp
    vel_ema = 0.2            # EMA blend for the finite-difference body velocity (read_root)

    def __init__(self, num_envs, build_robot, *, gravity=(0, -9.81, 0), spacing=3.0,
                 device="cuda", seed=0, read_root=False, read_links=False, build_world=None,
                 graph=False):
        if self.act_dim is None:
            raise TypeError(f"{type(self).__name__} must define act_dim")
        self.K = num_envs
        self.dt = 1.0 / self.control_hz
        self.max_steps = int(self.episode_s * self.control_hz)
        if build_robot is None:                                  # sim-less task: override simulate()
            self.sim = None
            self.device = torch.device(device)
            self.state = _NullState(num_envs, self.device)
        else:
            self.sim = GpuSim(num_envs, build_robot, gravity=gravity, spacing=spacing,
                              device=device, read_root=read_root, read_links=read_links,
                              build_world=build_world)
            self.device = self.sim.device
            self.state = RobotState(self.sim, self.dt, ema=self.vel_ema)
        self.g = torch.Generator(device=self.device).manual_seed(seed)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._env_state = []
        # per-term reward stats: device-side accumulators, host-synced only in stats_line()
        self._ep_term = None
        self._done_sum = None
        self._done_steps = torch.zeros((), dtype=torch.long, device=self.device)
        # CUDA-graph replay of the per-step torch region — see _capture()
        self.graph = bool(graph)
        self._cg_hot = self._cg_obs = None
        self._cg_act = None
        self._cg_out = self._cg_obs_out = None
        self.settling = False

    # ---- hooks a task implements --------------------------------------------------
    def on_reset(self, idx):
        """Set the sim state for envs `idx` (set_joint_state / set_root_state) and sample any
        task state (e.g. commands). Registered env_state() buffers are already re-initialized
        when this runs, so only overwrite the ones that need a non-init value."""
        raise NotImplementedError

    def act(self, a):
        """Map the clamped policy action [K, act_dim] to the [K, dof] tensor the control mode
        applies: generalized forces for control='force', PD position targets for 'drive'.
        Called once per control step (and with a zero action during reset settling), so gait
        clocks and action filters may advance in here."""
        raise NotImplementedError

    def observe(self, s):
        """Build the observation [K, obs_dim] from the RobotState (and task buffers)."""
        raise NotImplementedError

    def reward_terms(self, s, a):
        """Return the reward as a dict of NAMED terms, each [K]. The base sums them for the
        trainer and logs the per-term per-step episode means through stats_line() — name the
        terms after what they pay for, that's what you'll read when training goes sideways.
        `s.terminated` already holds this step's failure mask (usable for a fall penalty)."""
        raise NotImplementedError

    def terminated(self, s):
        """True-failure terminals (fell, crashed, out of bounds), [K] bool. These bootstrap
        V=0; time-limit truncation is handled by the base and bootstraps V(terminal_obs).
        Default: no failure terminals (timeout-only task)."""
        return torch.zeros(self.K, dtype=torch.bool, device=self.device)

    def simulate(self, a):
        """Advance the world one control step for the (already clamped) action [K, act_dim].
        Default: map the action through act() and advance the GpuSim (substeps>1 -> substep(),
        one state read at the end). Sim-less tasks (build_robot=None) override this with their
        own dynamics instead of act()."""
        self._apply(self.act(a))
        self.on_pre_substep()
        if self.substeps > 1:
            self.sim.substep(self.dt / self.substeps, self.substeps)
        else:
            self.sim.step(self.dt)

    def on_pre_substep(self):
        """Runs after the action is written and before the physics advances — the one place an
        external force belongs, since PhysX clears link forces on every step."""
        pass

    def on_step(self, s):
        """Optional per-step task logic (e.g. periodic command resampling). Runs after the
        state refresh, before termination/reward — changes made here are seen by both."""

    def on_done(self, idx):
        """Optional hook over the envs ending THIS step (terminal or timeout), called before
        their sim state and env_state() buffers are reset — the place for difficulty-curriculum
        updates and end-of-episode metrics that need the pre-reset values."""

    def on_settled(self):
        """Optional hook after the full-reset settle loop (e.g. zero a gait clock that the
        settle steps advanced)."""

    def config(self):
        """The train/deploy contract persisted into the policy checkpoint (PPO picks this up
        when constructed without meta). Extend with every constant a deploy viewer must
        reproduce — force scales, obs scales, drive parameters."""
        return {"control_hz": self.control_hz, "dt": self.dt, "episode_s": self.episode_s}

    # ---- infrastructure ------------------------------------------------------------
    def env_state(self, shape=(), init=0.0, dtype=torch.float32):
        """Allocate a [K, *shape] per-env state buffer that the base automatically resets to
        `init` for every env that resets, full or partial — phase clocks, commands, previous
        actions. Registering is what makes forgetting a re-baseline impossible."""
        buf = torch.full((self.K, *shape), init, dtype=dtype, device=self.device)
        self._env_state.append((buf, init))
        return buf

    def _apply(self, target):
        if self.control == "drive":
            self.sim.apply_drive_target(target)
        else:
            self.sim.apply_force(target)

    def _reset_envs(self, idx):
        for buf, init in self._env_state:
            buf[idx] = init
        self.steps[idx] = 0
        self.on_reset(idx)

    def _accumulate(self, terms, d):
        if self._ep_term is None:
            self._ep_term = {k: torch.zeros(self.K, device=self.device) for k in terms}
            self._done_sum = {k: torch.zeros((), device=self.device) for k in terms}
        for k, v in terms.items():
            self._ep_term[k] += v
        if d.numel() > 0:
            self._done_steps += self.steps[d].sum()
            for k in terms:
                self._done_sum[k] += self._ep_term[k][d].sum()
                self._ep_term[k][d] = 0.0

    def stats_line(self):
        """Per-term per-step reward means over the episodes completed since the last call —
        the PPO log line appends this. Empty until an episode has finished. Costs one host
        sync per term, only at log time."""
        if self._done_sum is None or self._done_steps.item() == 0:
            return ""
        n = float(self._done_steps.item())
        parts = [f"{k} {v.item() / n:+.3f}" for k, v in self._done_sum.items()]
        self._done_steps.zero_()
        for v in self._done_sum.values():
            v.zero_()
        return "  ".join(parts)

    # ---- the PPO env protocol --------------------------------------------------------
    def _reset_all(self):
        self._reset_envs(torch.arange(self.K, device=self.device))
        if self.settle_steps:
            zero = torch.zeros(self.K, self.act_dim, device=self.device)
            self.settling = True          # tasks skip perturbations while the spawn settles
            for _ in range(self.settle_steps):
                self.simulate(zero)
            self.settling = False
            self.on_settled()
        elif self.sim is not None:
            self.sim.read()   # refresh joint_pos/joint_vel with the freshly written state
        self.state.rebaseline()
        return self.observe(self.state)

    def reset(self):
        obs = self._reset_all()
        if self.graph and self._cg_hot is None:
            self._capture()
            obs = self._reset_all()      # wipe what the capture warm-up did to the env state
        return obs

    @torch.no_grad()
    def _hot(self, a):
        """The step's pure-torch region: everything between the physics and the auto-reset.

        Kept as one function because it is also the CUDA-graph capture unit. Nothing in here may
        read a device value back to the host or produce a data-dependent shape, or the graph will
        freeze whatever the capture happened to see — see `graph` on the constructor.
        """
        self.state.update()
        self.on_step(self.state)
        term = self.terminated(self.state)
        self.state.terminated.copy_(term)
        timeout = (self.steps >= self.max_steps) & ~term
        done = term | timeout
        terms = self.reward_terms(self.state, a)
        rew = torch.stack(list(terms.values())).sum(0)
        # terminal_obs, read BEFORE any reset overwrites the sim state
        return rew, done, timeout, self.observe(self.state), terms

    def _capture(self):
        """Record `_hot` and the post-reset `observe` as two CUDA graphs.

        Worth doing because the region is launch-bound, not compute-bound: a few hundred small
        torch kernels whose dispatch costs an order of magnitude more than their execution, and a
        graph replays the whole sequence with one launch. Measured on SpotStepsEnv at K=2048, the
        region goes 9.4 ms -> 0.9 ms and the whole step drops about a third.

        The seeded generator has to be registered or capture refuses the first RNG op in it, and
        the warm-up on a side stream is PyTorch's requirement, not ours.

        The warm-up genuinely RUNS `_hot`, side effects and all, so this is called from reset() and
        the reset is then redone — otherwise a graphed run starts a few phase-clock ticks ahead of
        an eager one and the two trajectories are not comparable. That only cleans up state the task
        registered with env_state(); anything a task keeps outside it will carry the warm-up's marks.
        """
        self._cg_act = torch.zeros(self.K, self.act_dim, device=self.device)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                self._hot(self._cg_act)
                self.observe(self.state)
        torch.cuda.current_stream().wait_stream(side)

        self._cg_hot = torch.cuda.CUDAGraph()
        self._cg_hot.register_generator_state(self.g)
        with torch.cuda.graph(self._cg_hot):
            self._cg_out = self._hot(self._cg_act)
        self._cg_obs = torch.cuda.CUDAGraph()
        self._cg_obs.register_generator_state(self.g)
        with torch.cuda.graph(self._cg_obs):
            self._cg_obs_out = self.observe(self.state)

    def verify_graph(self, steps=32, tol=1e-5, actions=None):
        """Guard on the capture: after each replayed step, recompute the observation eagerly from
        the state the step left behind, and require the two to agree.

        A graph bakes in the control flow it saw. If an env resamples with a data-dependent shape,
        or rebinds a tensor in Python where the graph captured an address, replay keeps reading the
        tensor that existed at capture time — and the observations go stale in a way no shape or
        finiteness check will catch. `observe` is deterministic given the state, so recomputing it
        is a real check and not a tautology. Comparing whole trajectories would not be: both arms
        draw from the same RNG stream and would diverge for honest reasons.

        What it CANNOT catch: Python-side state inside `_hot` that simply stops updating, because
        replay never runs the Python and the eager recomputation then reads the same frozen value.
        `self.k += 0.1` in on_step is invisible to this check and to every other one. The rule that
        actually keeps a task safe is that everything in the hot region is tensor ops on buffers
        that are written through, never rebound.

        Raises on the first divergence; returns the worst difference seen.
        """
        if not self.graph:
            raise RuntimeError("verify_graph: this env was not built with graph=True")
        a = actions if actions is not None else torch.zeros(self.K, self.act_dim, device=self.device)
        worst = 0.0
        for i in range(steps):
            obs = self.step(a)[0].clone()
            d = (obs - self.observe(self.state)).abs().max().item()
            worst = max(worst, d)
            if d > tol:
                raise AssertionError(
                    f"verify_graph: the replayed observation is {d:.3e} away from recomputing it at "
                    f"step {i} — the capture froze something this env varies per step")
        return worst

    def step(self, actions):
        a = actions.clamp(-self.clip_actions, self.clip_actions) if self.clip_actions else actions
        self.simulate(a)
        self.steps += 1
        if self.graph:
            if self._cg_hot is None:
                raise RuntimeError("graph=True: call reset() before step(), the capture happens there")
            self._cg_act.copy_(a)
            self._cg_hot.replay()
            rew, done, timeout, term_obs, terms = self._cg_out
        else:
            rew, done, timeout, term_obs, terms = self._hot(a)

        d = torch.nonzero(done, as_tuple=False).squeeze(-1)
        self._accumulate(terms, d)   # needs steps[d] pre-zero for the episode lengths
        if d.numel() > 0:
            self.on_done(d)
            self._reset_envs(d)
            if self.sim is not None:
                self.sim.read()
            self.state.rebaseline(d)
            # the reset wrote through the same state buffers the graph reads, so a replay sees it
            if self.graph:
                self._cg_obs.replay()
                obs = self._cg_obs_out
            else:
                obs = self.observe(self.state)
        else:
            obs = term_obs
        return obs, rew, done, term_obs, timeout
