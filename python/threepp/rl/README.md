# threepp.rl — writing environments and training

A small, **owned** RL stack for GPU-resident vectorized environments: a compact PPO
(`ppo.py`), a direct-GPU PhysX batch (`sim.py`, `GpuSim`), an env-authoring base that owns the
step/reset choreography (`task.py`, `VecTask`), and the building blocks (`ActorCritic`,
`RunningNorm`, `compute_gae`). No `rl_games` / `rsl_rl` / Gym dependency — the whole thing is a
few hundred readable lines you can audit and bend.

```python
from threepp.rl import GpuSim, VecTask, PPO, load_policy, save_policy, ActorCritic, RunningNorm
```

For a **robot task**, subclass [`VecTask`](#robot-tasks-vectask--recommended) and write only the
observation, the named reward terms, and the reset — the base handles everything below. The raw
protocol is documented first because it *is* the contract (and non-physics envs implement it
directly), but you rarely write it by hand anymore.

This is the **GPU-vectorized** RL idiom (think Isaac Lab / rsl_rl / Brax), **not** the classic
single-env Gym idiom. If you've only seen Gym tutorials, read [How this compares](#how-this-compares)
first — a couple of conventions here are deliberately different.

---

## The environment contract

An environment is a **plain Python class** — no registration, no spaces, no wrappers. It owns
`K` parallel environments and talks to PPO in GPU tensors. It must provide exactly two methods:

```python
env.reset() -> obs                                  # [K, obs_dim]  (or [K, C, H, W] for images)
env.step(action) -> (obs, reward, done, terminal_obs, is_timeout)
#   action       [K, act_dim]   float
#   obs          [K, obs_dim]   the observation for the NEXT step (already auto-reset, see below)
#   reward       [K]            float
#   done         [K]            bool   — episode ended this step (terminal OR truncation)
#   terminal_obs [K, obs_dim]   the observation BEFORE any reset (for value bootstrapping)
#   is_timeout   [K]            bool   — which `done`s are time-limit truncations
```

Everything is a GPU tensor on the same device. PPO infers `K`, `obs_dim`, and the device from the
first `reset()`; you pass `act_dim` to the `PPO(...)` constructor. There is **no `info` dict** —
expose metrics as plain attributes (see [Metrics](#metrics--logging)).

### Auto-reset, `terminal_obs`, `is_timeout` — the part that's different

PPO never calls `reset()` mid-rollout. Instead **`step()` auto-resets** any environment that just
finished: for those lanes the returned `obs` is already the *fresh* post-reset observation, while
`terminal_obs` carries the observation from *before* the reset. This lets the trainer bootstrap the
value target at the boundary without a stale-obs bug.

`is_timeout` splits the two reasons an episode ends, because they need **different value targets**:

| ended because… | `done` | `is_timeout` | value target at the boundary |
|---|---|---|---|
| true terminal (e.g. robot fell) | `True` | `False` | `0` — the future really is worthless |
| time-limit truncation (ran out of steps) | `True` | `True` | `V(terminal_obs)` — the episode would have continued |

Getting this wrong silently biases the value function. If your task has **only** a time limit (no
failure terminal), just return `is_timeout = done`.

> Gym mapping: `done` ≈ `terminated or truncated`, `is_timeout` ≈ `truncated`, and `terminal_obs`
> is what Gym hides in `info["final_observation"]`. We return it explicitly so the bootstrap is
> obvious in `compute_gae`.

---

## A minimal environment

A complete, physics-free env that shows the whole contract — `K` dots steering toward a goal:

```python
import torch
from threepp.rl import PPO

class ReachEnv:
    def __init__(self, num_envs=4096, device="cuda", max_steps=100):
        self.K, self.device, self.max_steps = num_envs, device, max_steps
        z = lambda *s: torch.zeros(*s, device=device)
        self.pos, self.goal = z(num_envs, 2), z(num_envs, 2)
        self.steps = torch.zeros(num_envs, dtype=torch.long, device=device)

    def _obs(self):
        return torch.cat([self.pos, self.goal - self.pos], dim=1)        # [K, 4]

    def _reset_idx(self, idx):
        self.pos[idx] = 0.0
        self.goal[idx] = torch.randn(idx.numel(), 2, device=self.device)
        self.steps[idx] = 0

    def reset(self):
        self._reset_idx(torch.arange(self.K, device=self.device))
        return self._obs()

    @torch.no_grad()
    def step(self, action):                                             # action [K, 2]
        self.pos += 0.1 * action.clamp(-1, 1)
        self.steps += 1
        dist = (self.goal - self.pos).norm(dim=1)
        reward = -dist
        reached  = dist < 0.1                                           # true terminal (success)
        timeout  = self.steps >= self.max_steps                        # truncation
        done     = reached | timeout
        is_timeout = timeout & ~reached
        term_obs = self._obs()                                         # BEFORE reset
        d = done.nonzero(as_tuple=False).squeeze(-1)
        if d.numel():
            self._reset_idx(d)                                          # auto-reset finished lanes
        return self._obs(), reward, done, term_obs, is_timeout

PPO(ReachEnv(), act_dim=2, hidden=(64, 64)).learn(200)
```

The shape is always the same: assemble the obs in `_obs()`, advance the world in `step()`,
compute a `reward` as a sum of terms, decide `done`/`is_timeout`, snapshot `terminal_obs`, then
auto-reset the finished lanes. The Spot examples (`python/examples/spot/`) are this exact pattern
with a `GpuSim` PhysX world instead of two tensors — see below.

---

## Robot tasks: `VecTask` — recommended

For physics tasks, the choreography above (steps/timeout bookkeeping, `terminal_obs` capture,
partial resets, re-baselining EMA velocities and phase clocks) is identical in every env and easy
to get subtly wrong. `VecTask` owns it; a task is only its task logic:

```python
from threepp.rl import VecTask

class WalkEnv(VecTask):
    control_hz = 30                  # dt = 1/control_hz
    episode_s = 8.0                  # time-limit truncation
    act_dim = 12
    control = "drive"                # "drive" = PD position targets, "force" = generalized forces
    settle_steps = 6                 # zero-action settle after a full reset (0 = off)

    def __init__(self, num_envs, device="cuda"):
        super().__init__(num_envs, build_robot=lambda world, i: MyRobot(world, i),
                         read_root=True, device=device,
                         build_world=lambda w: add_ground(w))
        self.cmd = self.env_state((2,))          # registered: auto-reset on every episode end

    def on_reset(self, idx):                     # set sim state (+ own buffers) for envs `idx`
        self.sim.set_joint_state(idx, ...); self.sim.set_root_state(idx, ...)
        self.cmd[idx] = sample_commands(idx.numel())

    def act(self, a):                            # clamped action [-1,1] -> [K, dof] targets
        return self.gait_targets() + a * RESIDUAL_SCALE

    def observe(self, s):                        # s = RobotState (derived, refreshed per step)
        return torch.cat([s.joint_pos, s.joint_vel, s.up[:, None], self.cmd], dim=-1)

    def reward_terms(self, s, a):                # NAMED terms, each [K] — summed by the base
        return {"track_v": 2.2 * torch.exp(-3.0 * (s.v_fwd - self.cmd[:, 0]) ** 2),
                "upright": 0.6 * s.up.clamp_min(0.0),
                "effort":  -0.04 * a.pow(2).mean(-1),
                "fell":    -5.0 * s.terminated.float()}

    def terminated(self, s):                     # true-failure terminals (default: none)
        return s.up < 0.0
```

What the base gives you:

- **`env_state(shape, init=...)`** — allocate per-env episode state (phase clocks, commands,
  previous actions). The base masks it back to `init` on every reset, full or partial, so
  forgetting a re-baseline is impossible.
- **`RobotState` (`s`)** — derived state computed once per step: `joint_pos`/`joint_vel`
  (add-order), and with `read_root=True`: `up`, `fwd_x`/`fwd_z`, `yaw`, EMA world velocity
  `v_x`/`v_z`, wrap-safe `yaw_rate`, and the body-frame `v_fwd`/`v_lat` every locomotion reward
  wants. The finite-difference baselines re-anchor automatically on reset (no spawn-teleport
  velocity spikes). `s.terminated` holds this step's failure mask, so a fall penalty is one line.
- **Named reward terms** — the base sums the dict for the trainer *and* accumulates per-term
  per-step episode means. PPO's log line shows them (`stats_line()`), so when a policy converges
  to the wrong optimum you can see *which* term dominates without adding prints:

  ```
  it   40 | ep_ret   -590.7 | ep_len   600 |   38.9k steps/s |  33.7s
          up +0.130  up_bonus +0.413  energy -1.076  rail -0.205  effort -0.004  stabilize -0.265
  ```
- **`config()`** — the train/deploy contract. Extend it with every constant a deploy viewer must
  reproduce (force scales, obs scales); a `PPO` constructed without `meta` persists it into the
  checkpoint automatically.
- Optional hooks: `on_step(s)` (periodic command resampling), `on_settled()` (e.g. zero a gait
  clock the settle steps advanced), `on_done(idx)` (end-of-episode work that needs the pre-reset
  buffer values — difficulty-curriculum promote/demote, episode metrics).
- **`substeps`** — physics substeps per control step; the base uses `sim.substep()` (one state
  read per control tick) when it is > 1.
- **Sim-less tasks** — a task whose world is not a GpuSim (e.g. a rendered pixel env with
  hand-rolled torch dynamics) passes `build_robot=None` and overrides `simulate(a)` instead of
  `act()`; everything else (timeout bookkeeping, terminal-obs capture, `env_state()` auto-reset,
  reward-term logging) still applies. See `examples/turret/turret_env.py`.

Worked examples: `examples/cartpole/cartpole_env.py` (force control, timeout-only),
`examples/spider/hexapod_gpu_env.py` (PD drive, free base, CPG clock, command resampling),
`examples/spot/spot_terrain_env.py` / `spot_heightfield_env.py` / `spot_steps_env.py` (substeps,
terrain scan, imitation anchor, adaptive curriculum via `on_done`),
`examples/spot/scratch_distillation/scratch_env.py` (read_links foot contacts, iteration-driven
schedules), and `examples/turret/turret_env.py` (sim-less pixel task).

### Physics-based envs (`GpuSim`)

For robots, build the world on `GpuSim`, which runs `K` articulations in one direct-GPU PhysX scene:

```python
self.sim = GpuSim(num_envs, build_robot=lambda world, i: MyRobot(world, i),
                  gravity=(0, 0, -9.81), spacing=3.0, device=device,
                  read_root=True, read_links=False, build_world=lambda w: add_ground(w))
```

It exposes the batched state as tensors (`root_position`, `root_quat`, `root_linvel`,
`joint_pos`, `joint_vel`, …) and the controls (`apply_drive_target`, `substep`, `set_root_state`,
`set_joint_state`, `make_root_pose`). Your `_obs()` is then just torch ops over those tensors. See
`sim.py` and `examples/spot/spot_terrain_env.py` for a full worked example.

### Terrain height without a formula (`raycast.py`)

A legged env asks "how high is the ground here?" every control step — for the height scan in the
observation, for the drop-settle spawn, for the fell-over test. Answering that in torch means the
terrain has to be something you can write in closed form: a tent, a bilinear grid, a plane.
`threepp.rl.raycast` answers it instead by casting a ray down into a Warp BVH built over the
world's static triangles, so authored geometry, an imported mesh, or a scene loaded out of the
editor all work — and the answer comes from the surface the feet actually collide with.

```python
from threepp.rl.raycast import CollectedWorld, TerrainRays

collector = []
def build(world):
    world = CollectedWorld(world)        # forwards every call, keeps each Mesh it is handed
    collector.append(world)
    add_my_terrain(world)                # unchanged builder code

super().__init__(K, build_robot, build_world=build, ...)
self.rays = TerrainRays.from_objects(collector[0].meshes, device=self.device, up="z")

h = self.rays.heights(px, py)            # [K, N] ground height, same shape as its inputs
```

It is not a cost you absorb — it is *one* kernel where the formula was thirty small torch ops, so
it comes out ahead. Measured on the stairs env at K=2048 (RTX 4070, 616 044 triangles, the 45-cell
scan, interleaved A/B):

| | raycast | analytic formula |
|---|---|---|
| the scan | **0.102 ms** (906 M rays/s) | 0.841 ms |
| the whole `env.step` | **38.3 ms** | 42.0 ms |

Warp shares torch's CUDA primary context — the same one `GpuSim` hands PhysX — so the query reads
the sim's state tensors and writes its answer with no copy and no context switch. Warp is imported
lazily on the first `TerrainRays`, so `import threepp.rl` still works without it.

`examples/spot/spot_scan_ab.py` is the acceptance test: it runs both arms through the env's own
`_terrain_h` and reports agreement and cost. Where a formula is exact — the stairs env inside a
lane — the two match to 0.000000 m over 436 730 probes.

`heights()` returns a **fresh** tensor. It used to hand back a view of one reused buffer, which
aliased the moment a caller held two results at once — `h_here = heights(x, y)` followed by
`heights(px, py)` left `h_here` pointing at the second call's numbers, and every env doing exactly
that read garbage from its second control step onward. Pass `out=` to reuse a buffer only where
that cannot happen.

### The scan a robot could actually have seen (`perception.py`)

A height scan read straight off the terrain is privileged information. `PerceivedScan` limits it to
what a body-mounted depth camera could have observed: each control step it casts one ray from the
camera to every query point (`TerrainRays.visible` — frustum, range, and a real occlusion test
against the same BVH), marks what it saw into a per-env world-anchored map, and answers from that
map. Occluded and never-observed cells read as flat ground; cells the camera swept earlier are
remembered.

```python
from threepp.rl.perception import PerceivedScan

scan = PerceivedScan(rays, K, origin_b=env.lane_y, bounds=(-1.0, 31.6, 1.5))
ahead, h_here = scan.read(sim.root_position, sim.root_quat, px, py)   # the deploy-side pair
scan.forget(reset_idx)                                               # a teleport drops the map
```

The map stores a *seen* bit rather than a fused height — equivalent for static terrain, and cheap
enough to hold at K=2048 — and `noise` re-adds the sensor error as a fixed per-cell bias, which is
what a converged elevation map's error looks like. Use it for the **observation only**: reward,
termination and spawn placement should keep reading ground truth, or you are handicapping the
training signal rather than the policy. `examples/spot/spot_occlusion.py` measures what the camera
can and cannot see, and whether it changes what the policy does.

### Replaying the step from a CUDA graph (`graph=True`)

A GpuSim step is not compute-bound, it is **launch**-bound. `SpotStepsEnv` at K=2048 dispatches
**584 torch ops per control step** — each a few microseconds of GPU work behind ~20 µs of CPU
dispatch — and the cost of that is *flat in K*: 12 ms at K=1024 and the same 12 ms at K=2048,
against 13.6 ms of actual physics. It is not host syncs either; there are only five in a step, and
removing four of them is worth 1.6%.

So `VecTask(..., graph=True)` records the whole torch region between the physics and the auto-reset
as a CUDA graph and replays it with one launch:

| K=2048, RTX 4070 | eager | graph |
|---|---|---|
| the torch region alone | 9.40 ms | **0.87 ms** |
| `env.step` | 22.54 ms | **13.20 ms** (1.71×) |
| training incl. PPO | 41.8k steps/s | **52.6k steps/s** (1.26×) |

Capture happens inside `reset()`, and the reset is then redone, because the warm-up genuinely runs
the region — so a graphed run and an eager one start from the same state. **That only cleans up
buffers registered with `env_state()`**; state a task keeps outside it will carry the warm-up's
marks.

Writing a task that can be captured is one rule: **the hot region is tensor ops on buffers that are
written through, never rebound, and never shaped by data**. In practice:

- `self.buf.copy_(x)` / `buf.add_()`, not `self.buf = x` — the graph recorded an address.
- masks and `torch.where`, not `torch.nonzero` — a data-dependent shape is baked at capture.
- no `.item()` in the step path; keep per-step stats on device and read them when a trainer logs.
- RNG from the default generator is fine and advances properly across replays; a seeded
  `torch.Generator` needs registering, which the capture does for `self.g`.

`env.verify_graph()` is the guard: it steps, then recomputes `observe` from the state the step left
behind and requires them to match. That catches a stale address or a frozen shape. It **cannot**
catch Python-side state that simply stops updating (`self.k += 0.1` in `on_step`), because replay
never runs the Python and the eager recomputation then reads the same frozen value — which is why
the rule above, not the check, is what keeps a task correct.

---

## Training

```python
ppo = PPO(env, act_dim,
          hidden=(512, 256, 128),     # actor/critic MLP hidden sizes (ELU between)
          lr=3e-4, horizon=32,        # rollout length per update
          normalize_obs=True,         # running mean/var on observations (RunningNorm)
          normalize_returns=True,     # critic predicts normalized returns (scale-robust)
          entropy=0.0, log_std_init=-0.5,
          target_kl=0.02,             # early-stop an update epoch on KL blow-up
          anneal_lr=True,
          aux_loss=None,              # optional extra loss term, see below
          meta={"obs_dim_note": "..."})   # arbitrary JSON-able dict saved into the checkpoint
ac, norm, meta = ppo.learn(iterations=1500, log_every=20, on_log=my_log_callback)
ppo.save("policy.pt")
```

Defaults are sensible (GAE `γ=0.99 λ=0.95`, clip `0.2`, 5 epochs × 4 minibatches). What it gives
you over a from-scratch loop: **running obs + return normalization** (so the value head works whether
returns are ~1 or ~1000), linear LR anneal, and target-KL early stopping.

- **`normalize_obs`** — keep it `True` for state obs unless you are warm-starting a network trained
  on raw obs. **Image obs** (`[K, C, H, W]` uint8) are normalized in-net (`/255`) and skip the
  RunningNorm automatically — pass `image_shape` is inferred from `reset()`.
- **`aux_loss(ac, obs_minibatch) -> scalar`** — added to the PPO loss each minibatch. General hook for
  symmetry augmentation, a BC/KL anchor to a reference policy, etc. It's called on the *already-
  normalized* obs.
- **`on_log(msg)`** — called every `log_every` iterations with the formatted progress line. Use it to
  drive schedules (`env.set_iter(it)`), checkpoint, and read your env metrics.

---

## Evaluating & deploying a checkpoint

`save`/`load_policy` round-trip the actor-critic, the obs normalizer, and the `meta` dict:

```python
ac, norm, meta = load_policy("policy.pt", device="cpu")   # norm is None if normalize_obs was False
```

> **Gotcha (the #1 thing people miss):** `ac.act_mean(obs)` does **NOT** normalize. During `learn`
> PPO normalizes obs *before* the network; at eval/deploy you must do it yourself:
>
> ```python
> a = ac.act_mean(norm.norm(obs)) if norm is not None else ac.act_mean(obs)
> ```
>
> Feeding raw obs to a `normalize_obs=True` policy produces garbage actions with no error. `act_mean`
> is the deterministic mean action (deployment); `act` returns `(action, logprob, value)` with
> Gaussian exploration noise (training/rollout).

To **warm-start / fine-tune** from a checkpoint, load it into a fresh `PPO`'s `ac`/`norm`
(`ppo.ac.load_state_dict(...)`, `ppo.norm.load(src_norm.state())`); if the obs dimension changed,
copy the overlapping input-layer columns and zero-init the new ones (see
`examples/spot/train_spot_stairs.py:warmstart_scratch_to_terrain` for the expand-the-norm pattern).

---

## Metrics & logging

`VecTask` envs get per-term reward means in the PPO log line for free (see above). Beyond that,
there is no per-step `info` dict. The convention is: stash scalars as **plain attributes** on the env
inside `step()`, and read them in `on_log`:

```python
# in env.step(): self.last_track = (track_lin + track_ang).mean().item()
def my_log_callback(msg):
    print(f"{msg} | track {env.last_track:.3f} | fell {env.last_fell:.3f}")
    ppo.save("policy_latest.pt")
```

**Curriculum** is likewise a side channel: give the env a `set_iter(it)` method that updates its
schedules (command envelope, reward weights, difficulty), and call it from `on_log`. PPO itself knows
nothing about curricula.

---

## How this compares

This stack is the **GPU-vectorized, auto-reset, tensors-in/tensors-out** family — closest to
**Isaac Lab / rsl_rl / Brax**. It is intentionally *not* Gym/Gymnasium:

| | here | Gym / Gymnasium |
|---|---|---|
| envs | `K` parallel, GPU tensors, one class | one env (or a `VectorEnv` wrapper), numpy/CPU |
| `step` returns | `(obs, reward, done, terminal_obs, is_timeout)` | `(obs, reward, terminated, truncated, info)` |
| reset | auto-reset inside `step` | you call `reset()` (classic) / auto in `VectorEnv` |
| spaces | none — just `obs_dim`/`act_dim` ints | `observation_space` / `action_space` |
| metrics | env attributes read in `on_log` | `info` dict per step |
| obs/reward | a literal `torch.cat` / summed terms in the env | often config-driven managers (Isaac Lab) |

**What this buys you:** the env is legible top-to-bottom (the obs is a `cat` you can point at, the
reward is a commented sum), training is one line, and the PPO internals are auditable.

**What it costs:** these envs are **not** drop-in for SB3 / CleanRL / RLlib / Gym tooling. To use the
ecosystem you'd wrap one env lane in a Gym adapter (declare spaces, split `done` into
`terminated`/`truncated`, surface `terminal_obs` via `info["final_observation"]`, move metrics into
`info`). The mental model transfers cleanly to Isaac Lab / rsl_rl; it will feel foreign coming from
plain Gym.
