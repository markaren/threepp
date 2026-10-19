# Success-guided sampling on a pendulum pool

A toy-scale test of Success-Guided Sampling (Zhang et al., "A Balanced Data Diet: Addressing the
Exploration Bottleneck in Mega-Scale RL for Robot Control", CoRL 2026, arXiv 2610.12465). The
method changes only which configuration an environment resets into: each configuration keeps a
window of its recent episode outcomes, and resets are drawn from a Beta-shaped kernel over those
success estimates that peaks at a target rate and floors at the always-solved and never-solved
ends. The sampler lives in `threepp/rl/sgs.py` and plugs into any `VecTask` through `on_reset`
(draw a cell) and `on_done` (record the outcome). PPO is untouched.

## The task

`pendulum_env.py` is a sim-less `VecTask`: a torque-limited pendulum with gym's constants, 20 Hz
control, 8 s episodes. The configuration pool is 24 start-angle bins x 5 start-velocity bins x 5
torque limits (1.0, 1.5, 2.0, 3.0, 5.0) = 600 cells. The reward is sparse: one point per step
while upright, a small effort penalty. Success is holding upright through the last 1.5 s. The
fixed evaluation runs the deterministic policy on every cell and reports the mean over the uniform
pool, the paper's metric.

## Running it

    python train_sgs_pendulum.py --sampler sgs --envs 1024 --iters 600 --lr 1e-3 --graph --out sgs.pt
    python sweep.py --envs 256 1024 4096 --seeds 0 1 2 --iters 600 -- --lr 1e-3 --graph
    python play_sgs_pendulum.py --policy sgs.pt

`--graph` captures the per-step torch region as a CUDA graph; with the trainer's single minibatch
an iteration costs about 0.17 s at 1024 envs on an RTX 4060 laptop, so a run is two minutes.
`sweep.py` runs the matrix and prints the table; `play_sgs_pendulum.py` renders a 5 x 4 grid of
pendulums (columns torque limit, rows start state) driven by one policy in the GL viewer.

## Result, 2026-10-19

Sparse reward, lr 1e-3, 600 iterations, three seeds, final evaluation over the uniform pool:

| envs | uniform | SGS | torque 1.0 cells, uniform | torque 1.0 cells, SGS |
|-----:|--------:|----:|--------------------------:|----------------------:|
|  256 | 0.68 +- 0.08 | 0.70 +- 0.01 | 0.05 | 0.06 |
| 1024 | 0.75 +- 0.02 | 0.78 +- 0.02 | 0.17 | 0.29 |
| 4096 | 0.79 +- 0.05 | 0.84 +- 0.02 | 0.31 | 0.47 |

The gap grows with the number of environments and sits in the cells the policy can only just
solve. At 256 environments it is within seed noise. Uniform sampling solves most of this pool
within the budget, so the toy is not in the paper's regime of a sparse success region in a very
large pool; it checks the mechanism, not the magnitude.

Two findings from setting it up: a shaping term proportional to cos(theta) teaches the policy to
spin at the speed cap (mean return 0.52 per step), so the optional `--dense` shaping uses gym's
quadratic cost instead; and the sparse reward only takes off once the exploration noise is small
(log_std_init -1.0) and the pool has a genuinely easy end.
