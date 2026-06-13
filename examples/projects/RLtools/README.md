# RLtools × threepp — live SAC pendulum field

A single [Soft-Actor-Critic](https://arxiv.org/abs/1801.01290) agent from
[**RLtools**](https://rl.tools) (a dependency-free, header-only C++ deep-RL
library) is trained *from scratch, live*, on the classic Pendulum swing-up task.
Training runs on a background thread while the threepp scene renders a whole
**field of pendulums**, all driven by that one **current** policy from different
starting angles. Over the first ~minute you watch the field go from random
flailing to a synchronized swing-up-and-balance — a vivid way to see one
controller generalize across initial conditions.

It's a compact illustration of threepp as a real-time visualization front-end for
robotics / control / RL work: the physics and the learned controller are plain
C++, and the rendering, HUD and UI are threepp.

## What you see

- **A field of pendulums** (5×3) — each a rotating arm + bob, all driven every
  step by the *same* policy from its own random start. The bob tints
  **blue ↔ red** with the sign/magnitude of the applied torque.
- **Goal rings** turn green on each pendulum that's balanced upright — so you can
  count, at a glance, how many the policy has solved.
- **HUD** (top-left): training step / progress, phase (WARMUP → LEARNING →
  TRAINED), and field aggregates (how many upright, mean angle, mean score).
- **Control panel** (ImGui, right): training throughput + ETA, a live learning
  curve (the field's mean score per run climbing as it learns), and buttons to
  pause, restart from a fresh random policy, skip to the next run, plus a
  view-speed slider (speeds the view, not training).

## A second demo: parallel-rollout swarm (`rltools_swarm`)

`rltools_pendulum` (above) trains on one internal environment and shows a *preview*
field. `rltools_swarm` is the honest, scaled-up version: a **single** SAC learner is
fed by a **field of 64 environments stepped in parallel** — the "many envs → one
learner" pattern that makes GPU RL (Isaac-Gym/Brax) worthwhile, here at toy scale and
fully cross-platform. **The pendulums you see ARE the training data.** A background
thread steps all 64 envs, feeds their transitions to the learner, and runs SAC
gradient steps; the render thread only visualizes the published states. Each env's
links are drawn as instanced cylinders with a sphere at every distal joint (a
pendulum's bob; an acrobot's elbow + tip), and that sphere turns **green the moment
the env is solved** (`Env::upright`) — so the whole field becomes a live, at-a-glance
"how many has the policy cracked" meter (and it honestly shows acrobot *flashing*
through the top without holding).

This is Stage 1 (CPU) of a staged plan; the environment dynamics live in a plain
[`pendulum_env.hpp`](pendulum_env.hpp) and the learner behind
[`RLSwarmTrainer`](RLSwarmTrainer.hpp) only does inference + replay-buffer insertion +
gradient updates — that boundary is the seam where a **Vulkan-compute rollout** (Stage 2,
reusing the RF-DETR `VkInfer` GEMM kernels) later plugs in with no learner change.

### Pluggable tasks — swap the pendulum in one line

The swarm is **environment-generic**. The task is chosen by a single line in `swarm.cpp`:

```cpp
using Env = rldemo::PendulumEnv;   // or rldemo::AcrobotEnv
```

An `Env` is a small plain-C++ struct ([`env_pendulum.hpp`](env_pendulum.hpp),
[`env_acrobot.hpp`](env_acrobot.hpp)) that supplies its dims, `observe/step/reward/
sampleInitial`, and its renderable links (`rod(...)`). The learner is templated on the
env's dims (`RLSwarmTrainer<OBS, ACT>`), and the viz renders any env's links via
instanced cylinders — so nothing else changes. Verified: the **same** architecture
trains **pendulum** (obs 3, 1 link — solves 100%) and **acrobot** (obs 6, 2 links,
underactuated — reaches the goal in 100% of episodes), just by switching that typedef.
Adding a task whose dims aren't a native RLtools env needs one extra mapping in
`RLSwarmTrainer.cpp`.

### Write the dynamics once — it runs on CPU *and* GPU

The catch with running a rollout on the GPU is normally that the env's dynamics have to
be written twice — once in C++ (for the learner) and once in GLSL (for the shader) — and
kept bit-identical by hand, which is a real barrier for anyone who doesn't know GLSL. The
swarm avoids that: each task's dynamics live **once**, in a single
[`envdyn/<task>.envdyn`](envdyn/) file written in a small subset that is valid as **both
C++ and GLSL**. The CPU adapter (`env_*.hpp`) and the GPU rollout shader
(`swarm_rollout.comp`) both `#include` that one file, so they can't drift.

It's the standard "shared shader source" technique ([`env_dyn_compat.hpp`](envdyn/env_dyn_compat.hpp)
documents the dialect): by-value struct returns, `const` dims/constants (namespaced in
C++ so two envs coexist in one TU), `f`-suffixed literals, and two tiny macros
(`ENVDYN_FN`, `ENVDYN_OUT_ARR`) for the only qualifiers that differ. A cross-check over
100k random transitions confirms the single source is numerically identical to the
original hand-written dynamics (step/obs exact; reward within one ULP). So **adding a
task is: write one `.envdyn` (plain C++-looking math) + a thin adapter — and it runs on
the CPU learner and compiles to the GPU kernel automatically.** No GLSL authoring.

## Build

RLtools is opt-in (it is fetched only when you ask for it):

```bash
cmake -S . -B build -DTHREEPP_WITH_RLTOOLS=ON
cmake --build build --target rltools_pendulum   # single-policy preview field
cmake --build build --target rltools_swarm      # parallel-rollout swarm (64 envs -> 1 learner, CPU)
# Stage 2 (also needs the Vulkan backend):
cmake -S . -B build -DTHREEPP_WITH_RLTOOLS=ON -DTHREEPP_WITH_VULKAN=ON
cmake --build build --target vulkan_rltools_swarm  # GPU rollout (Vulkan compute) -> 1 CPU learner
```

## Stage 2 — GPU rollout (Vulkan compute)

`vulkan_rltools_swarm` (`examples/vulkan/`) moves the **rollout onto the GPU**,
cross-vendor, while the learner stays on the CPU. A single fused compute shader
([`swarm_rollout.comp`](../../vulkan/rl_swarm/shaders/swarm_rollout.comp)) runs the
*entire* rollout for every environment in **one dispatch** — one GPU thread per env:
observe → actor MLP forward (3→64→64→2) → `sample_and_squash` → step dynamics →
emit a transition + reset. The transitions are read back and fed to the **same,
unchanged `RLSwarmTrainer`** — only the rollout moved to the GPU. It reuses the
RF-DETR `VkInfer` compute harness; the actor weights are snapshotted from RLtools
(`RLSwarmTrainer::extractWeights`) and uploaded each tick.

The GPU forward pass is validated **bit-exact** against the CPU at startup
(`max |diff| ≈ 4e-8`), and a deterministic-policy learning curve is printed to the
console so the pipeline is verifiable without watching the window. This is the
"thousands of envs → one learner" (Isaac-Gym/Brax) pattern, at toy scale and on any
GPU vendor — the lever NVIDIA's CUDA-only stack can't claim. Honest scope: the CPU
learner is the bottleneck (the GPU collects far faster than one SAC stream trains),
and stochastic sampling uses CPU-generated noise uploaded per tick; full on-GPU
training (backprop) is the unbuilt Stage 3.

**The GPU rollout is pluggable too — not capped to one task.** Only the env's
*dynamics + observation* and its dims are task-specific; the rest of the shader (actor
MLP forward, `sample_and_squash`, transition emit/reset) is generic. So
`swarm_rollout.comp` `#include`s the selected task's single-source
[`envdyn/*.envdyn`](envdyn/) — the **same file the CPU learner compiles** (see "Write the
dynamics once" above) — and is built to **one SPIR-V variant per task**; the host selects
the matching variant + dims from the same `using Env = …` line as the CPU swarm. Verified
end-to-end on the GPU for **both** tasks: pendulum (parity `3.7e-8`, solves) and
**acrobot** (parity `3.4e-8`, 6-dim obs + 2-link RK4, det score climbs `−399 → −86` —
swings up), each by flipping that one typedef and rebuilding. The viz renders any task's
links via instanced cylinders.

`-DTHREEPP_WITH_RLTOOLS=ON` fetches the (pinned) header-only library via
`FetchContent` and exposes it as the `rltools` interface target. To build against
a local checkout instead of cloning, add
`-DFETCHCONTENT_SOURCE_DIR_RL_TOOLS=<path-to-rl-tools>`.

## Performance / acceleration

The demo builds RLtools' **dependency-free generic CPU backend** on purpose — it
runs anywhere with zero extra dependencies. The trade-off is training throughput:
RLtools' headline numbers (pendulum trains in seconds) come from a BLAS backend
(Accelerate / OpenBLAS / MKL) **and** the `-O3 -ffast-math -march=native` flags
its CMake applies only for GNU/Clang. A plain MSVC (`cl.exe`) build gets neither,
so full training (5000 steps) takes ~40 s here instead of seconds — fine for a
"watch it learn" loop, but leaving a lot on the table.

**Build type matters a lot on MSVC.** RLtools' speed depends entirely on the
compiler inlining its many tiny templated matrix kernels. RelWithDebInfo defaults
to `/Ob1` (limited inlining), which leaves them un-inlined and makes training
**~10× slower** (measured: 85 ms/step vs 7.9 ms/step). This example's CMakeLists
forces `/Ob2` (full inlining) in optimized configs, so RelWithDebInfo and Release
both train fast. A **Debug** build (`/Od`) has no optimization at all and stays
slow — it warns at configure time; use RelWithDebInfo or Release.

To speed up the (isolated, compute-bound) trainer translation unit, configure with
`-DTHREEPP_RLTOOLS_FAST=ON`. It applies the best flags for the detected compiler:

| Compiler | Flags applied | Note |
|---|---|---|
| **clang-cl** (LLVM, MSVC driver) | `/clang:-O3 -ffast-math -march=native` | biggest win on Windows; no new deps |
| clang / gcc (GNU driver) | `-O3 -ffast-math -march=native` | |
| `cl.exe` (plain MSVC) | *(none — prints guidance)* | MSVC doesn't vectorize the kernels |

On Windows the real lever is the **LLVM frontend**: install the *"C++ Clang tools
for Windows"* Visual Studio component (or LLVM from llvm.org), then configure from
a `vcvars64` prompt with:

```bash
cmake -S . -B build-clang -G Ninja ^
  -DCMAKE_CXX_COMPILER=clang-cl ^
  -DTHREEPP_WITH_RLTOOLS=ON -DTHREEPP_RLTOOLS_FAST=ON
```

`THREEPP_RLTOOLS_FAST` bakes the build machine's instruction set (`-march=native`)
into the binary, so it is **OFF by default**. For the published "seconds" figures,
add a BLAS backend (MKL/OpenBLAS) on top — RLtools' own autodetect supports MKL on
Windows.

## Architecture

- [`RLPendulumTrainer.hpp`](RLPendulumTrainer.hpp) / [`.cpp`](RLPendulumTrainer.cpp)
  — a plain-C++ facade. **All** of RLtools' template machinery is confined to the
  one `.cpp`; the header exposes only `float`/`long`. The render thread reads the
  policy through a periodically-refreshed snapshot, so the heavy SAC training step
  stays lock-free and never stalls rendering.
- [`main.cpp`](main.cpp) — the threepp scene, the (identical) pendulum dynamics,
  the HUD and the ImGui panel.

## Credits

- RLtools — Bhatt et al., *"RLtools: A Fast, Portable Deep Reinforcement Learning
  Library for Continuous Control"* ([arXiv:2306.03530](https://arxiv.org/abs/2306.03530)),
  <https://rl.tools>
- Pendulum-v1 dynamics follow the Gymnasium/RLtools formulation.
