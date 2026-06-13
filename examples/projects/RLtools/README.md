# RLtools × threepp — live SAC pendulum swing-up

A [Soft Actor-Critic](https://arxiv.org/abs/1801.01290) agent from
[**RLtools**](https://rl.tools) (a dependency-free, header-only C++ deep-RL
library) is trained *from scratch, live*, on the classic Pendulum swing-up task.
Training runs on a background thread while the threepp scene continuously renders
the **current** policy controlling a pendulum that obeys the very same dynamics
the agent trains on. Over the first ~minute you watch it go from random flailing
to a crisp swing-up-and-balance.

It's a compact illustration of threepp as a real-time visualization front-end for
robotics / control / RL work: the physics and the learned controller are plain
C++, and the rendering, HUD and UI are threepp.

## What you see

- **The pendulum** — a rotating arm + bob driven each step by the policy's torque.
  The bob tints **blue ↔ red** with the sign/magnitude of the applied torque, and
  a yellow arrow shows the tangential push.
- **The goal ring** turns green when the pole is balanced upright.
- **A trail** traces the swing-up arc.
- **HUD** (top-left): training step / progress, phase (WARMUP → LEARNING →
  CONVERGED), and live angle / angular-velocity / torque / episode-return.
- **Control panel** (ImGui, right): training throughput, a live learning curve
  (episode return climbing as it learns), and buttons to pause, restart from a
  fresh random policy, skip to the next episode, or fast-forward the sim.

## Build

RLtools is opt-in (it is fetched only when you ask for it):

```bash
cmake -S . -B build -DTHREEPP_WITH_RLTOOLS=ON
cmake --build build --target rltools_pendulum
```

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
