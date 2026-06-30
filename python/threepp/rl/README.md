# threepp.rl — writing environments and training

A small, **owned** RL stack for GPU-resident vectorized environments: a compact PPO
(`ppo.py`), a direct-GPU PhysX batch (`sim.py`, `GpuSim`), and the building blocks
(`ActorCritic`, `RunningNorm`, `compute_gae`). No `rl_games` / `rsl_rl` / Gym dependency —
the whole thing is a few hundred readable lines you can audit and bend.

```python
from threepp.rl import GpuSim, PPO, load_policy, save_policy, ActorCritic, RunningNorm
```

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

There is no per-step `info` dict. The convention is: stash scalars as **plain attributes** on the env
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
