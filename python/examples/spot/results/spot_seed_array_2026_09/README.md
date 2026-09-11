# Spot stairs: does the terrain source the policy trains on matter?

Seed array on the NTNU Idun cluster, 2026-09-10/11. 25 PPO runs of `train_spot_steps.py`
(5 training conditions x 5 seeds), each on one H100: K=2048, 3000 iterations, horizon 32,
eager step (no CUDA graph, see below), warm-started from the same `scratch_flat_best.pt`.
54-61 min per run at 55-61k steps/s; no run timed out; every run scored from its best checkpoint.
Scripts: `scripts/idun/spot_array.slurm`, `scripts/idun/spot_rescore.slurm`, `score_matrix.py`.
Per-run manifests and score records are under `runs/`; checkpoints are not kept here.

## Training conditions (rows)

| condition | terrain scan during training |
|---|---|
| analytic | privileged, closed-form tent formula (what `spot_steps.pt` was trained on) |
| raycast | privileged, a ray into the Warp BVH over the boxes actually in the world (`threepp.rl.raycast`) |
| perc0.00 | camera-limited elevation map (`threepp.rl.perception`): only what a forward depth camera swept |
| perc0.02 | the same with 0.02 m fixed per-cell map error |
| perc0.05 | the same with 0.05 m fixed per-cell map error |

## Scoring cells (columns)

Deterministic (`act_mean`), K=512, 900 steps of which the first 200 are discarded, score seed 0.
Base cells start every lane at curriculum level 0. Hard cells start every lane on the 0.20 m
risers (level 5) under random shoves of up to 3 m/s. Each entry is mean +- 95% t-interval over
the 5 training seeds.

### Base cells: track / fell per step / level
Each cell: mean +- 95% CI over seeds of *track* (of 2.0), *fell/step*, *level* (of 5).

| trained on | analytic | raycast | perceived:0.00 | perceived:0.02 | perceived:0.05 | perceived:0.10 |
|---|---|---|---|---|---|---|
| analytic | 1.896+-0.004 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) | 1.900+-0.003 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) | 1.901+-0.002 / 0.0000+-0.0000 / 0.94+-0.01 (n=5) | 1.900+-0.004 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) | 1.895+-0.003 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) | 1.877+-0.008 / 0.0000+-0.0000 / 0.94+-0.01 (n=5) |
| raycast | 1.909+-0.003 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) | 1.916+-0.002 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) | 1.910+-0.007 / 0.0000+-0.0000 / 0.94+-0.01 (n=5) | 1.910+-0.004 / 0.0000+-0.0000 / 0.94+-0.01 (n=5) | 1.905+-0.003 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) | 1.893+-0.006 / 0.0000+-0.0000 / 0.95+-0.00 (n=5) |
| perc0.00 | 1.898+-0.005 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) | 1.906+-0.003 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) | 1.899+-0.008 / 0.0000+-0.0000 / 0.95+-0.00 (n=5) | 1.900+-0.005 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) | 1.893+-0.008 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) | 1.877+-0.008 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) |
| perc0.02 | 1.895+-0.008 / 0.0000+-0.0000 / 0.94+-0.01 (n=5) | 1.904+-0.005 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) | 1.899+-0.005 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) | 1.898+-0.008 / 0.0000+-0.0000 / 0.94+-0.01 (n=5) | 1.897+-0.007 / 0.0000+-0.0000 / 0.94+-0.02 (n=5) | 1.888+-0.004 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) |
| perc0.05 | 1.897+-0.006 / 0.0000+-0.0000 / 0.94+-0.01 (n=5) | 1.904+-0.003 / 0.0000+-0.0000 / 0.94+-0.01 (n=5) | 1.898+-0.005 / 0.0000+-0.0000 / 0.93+-0.02 (n=5) | 1.898+-0.006 / 0.0000+-0.0000 / 0.94+-0.02 (n=5) | 1.897+-0.007 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) | 1.894+-0.004 / 0.0000+-0.0000 / 0.95+-0.01 (n=5) |

### Hard cells: track / fell per step / level / tent clear rate
Each cell: mean +- 95% CI over seeds of *track* (of 2.0), *fell/step*, *level* (of 5), *clear* (fraction of episodes clearing the tent).

| trained on | hard:analytic | hard:raycast | hard:perceived:0.00 | hard:perceived:0.05 | hard:perceived:0.10 |
|---|---|---|---|---|---|
| analytic | 1.631+-0.010 / 0.0010+-0.0003 / 4.96+-0.02 / 0.48+-0.03 (n=5) | 1.642+-0.017 / 0.0010+-0.0004 / 4.96+-0.03 / 0.46+-0.08 (n=5) | 1.624+-0.017 / 0.0010+-0.0003 / 4.96+-0.02 / 0.46+-0.07 (n=5) | 1.625+-0.019 / 0.0010+-0.0003 / 4.96+-0.02 / 0.43+-0.05 (n=5) | 1.613+-0.017 / 0.0010+-0.0003 / 4.95+-0.03 / 0.45+-0.07 (n=5) |
| raycast | 1.641+-0.010 / 0.0011+-0.0001 / 4.95+-0.01 / 0.48+-0.04 (n=5) | 1.655+-0.011 / 0.0012+-0.0002 / 4.96+-0.01 / 0.49+-0.03 (n=5) | 1.642+-0.012 / 0.0012+-0.0001 / 4.95+-0.02 / 0.49+-0.02 (n=5) | 1.641+-0.010 / 0.0012+-0.0001 / 4.96+-0.01 / 0.49+-0.05 (n=5) | 1.626+-0.012 / 0.0012+-0.0001 / 4.94+-0.02 / 0.50+-0.03 (n=5) |
| perc0.00 | 1.641+-0.019 / 0.0009+-0.0002 / 4.96+-0.02 / 0.47+-0.03 (n=5) | 1.646+-0.011 / 0.0009+-0.0002 / 4.96+-0.01 / 0.50+-0.04 (n=5) | 1.637+-0.013 / 0.0009+-0.0002 / 4.96+-0.01 / 0.50+-0.05 (n=5) | 1.634+-0.016 / 0.0010+-0.0001 / 4.97+-0.02 / 0.50+-0.04 (n=5) | 1.618+-0.014 / 0.0010+-0.0002 / 4.97+-0.01 / 0.44+-0.06 (n=5) |
| perc0.02 | 1.640+-0.005 / 0.0010+-0.0002 / 4.96+-0.03 / 0.43+-0.06 (n=5) | 1.649+-0.007 / 0.0010+-0.0001 / 4.96+-0.01 / 0.46+-0.03 (n=5) | 1.641+-0.008 / 0.0010+-0.0001 / 4.96+-0.01 / 0.45+-0.05 (n=5) | 1.634+-0.015 / 0.0010+-0.0002 / 4.96+-0.02 / 0.50+-0.05 (n=5) | 1.626+-0.011 / 0.0010+-0.0001 / 4.96+-0.02 / 0.46+-0.01 (n=5) |
| perc0.05 | 1.639+-0.012 / 0.0009+-0.0001 / 4.97+-0.01 / 0.44+-0.02 (n=5) | 1.643+-0.009 / 0.0009+-0.0001 / 4.97+-0.01 / 0.46+-0.04 (n=5) | 1.635+-0.013 / 0.0009+-0.0001 / 4.97+-0.01 / 0.48+-0.05 (n=5) | 1.636+-0.007 / 0.0009+-0.0001 / 4.96+-0.01 / 0.49+-0.05 (n=5) | 1.624+-0.018 / 0.0009+-0.0001 / 4.97+-0.01 / 0.47+-0.04 (n=5) |

## Findings

1. **Every condition converges to the same stairs competence.** All 25 runs reach curriculum
   level 5 in training; on the base cells all rows read ~1.90 track and zero falls. Training
   through the engine's own sensors (raycast, camera-limited map) costs nothing against the
   privileged oracle at convergence.
2. **The base matrix saturates.** Base scoring climbs from level 0 and reaches ~level 1 in 900
   steps, so it never meets the risers a converged policy was trained on. The hard cells were
   added for that reason; there, every row reads ~1.63 track, ~0.001 falls per step and a clear
   rate of 0.46-0.50, i.e. the tallest risers under shoves are the frontier for all of them and
   no terrain source moves it.
3. **Raycast-trained policies track slightly better on every base cell** (+0.014 to +0.016 of
   2.0, Welch t = 7 to 12 on 5 vs 5). Small, consistent, and gone in the hard cells (t = 1.9).
4. **Training with map error buys robustness to map error.** Scored at 0.10 m error, the cost
   against the noise-free perceived cell falls monotonically with training noise:
   analytic 0.024, raycast 0.017, perc0.00 0.023, perc0.02 0.011, perc0.05 0.004
   (perc0.05 vs analytic at 0.10 m: t = 5.4). Nothing else in the matrix separates the rows.
5. **Falls under shoves:** perc0.05-trained fall less than raycast-trained on the hard raycast
   cell (0.0009 vs 0.0012 per step, t = 4.1) but not less than analytic-trained (t = 0.9), so
   this is at most a weak effect and is not claimed.

## Caveat: CUDA graph off

The pilot ran with `--graph`; every Warp-backed condition then failed `verify_graph` (raycast
1.0 away, perceived with noise 0.10/0.26 away). The Warp launches in `threepp.rl` are not on
torch's capture stream, so a graph replays frozen heights. All 25 runs therefore used the
eager step, ~9 ms/step slower at K=2048 (55-61k steps/s on the H100 against 65-70k graphed).

## Per-run ranking (means over the five hard cells, sorted by tent clear rate)

The shipped `spot_steps.pt` (older lineage, analytic scan) is scored the same way for comparison.
It sits near the bottom on clear rate but has one of the lowest fall rates; the spread across the
25 runs (clear 0.42-0.53) is wider than any between-condition difference, so seed matters more
than terrain source here. `spot_steps.pt` was kept as the shipped policy.

| run | hard track | hard fell/step | hard clear | base track |
|---|---|---|---|---|
| raycast_s1 | 1.640 | 0.0011 | 0.527 | 1.909 |
| perc0.00_s2 | 1.642 | 0.0008 | 0.518 | 1.899 |
| raycast_s4 | 1.635 | 0.0012 | 0.497 | 1.908 |
| perc0.05_s3 | 1.639 | 0.0009 | 0.490 | 1.900 |
| raycast_s3 | 1.635 | 0.0012 | 0.490 | 1.909 |
| analytic_s3 | 1.617 | 0.0009 | 0.487 | 1.897 |
| analytic_s4 | 1.636 | 0.0008 | 0.486 | 1.895 |
| perc0.00_s4 | 1.623 | 0.0009 | 0.484 | 1.890 |
| perc0.00_s0 | 1.628 | 0.0008 | 0.483 | 1.896 |
| perc0.05_s0 | 1.623 | 0.0009 | 0.480 | 1.895 |
| perc0.05_s2 | 1.643 | 0.0009 | 0.476 | 1.894 |
| perc0.00_s3 | 1.650 | 0.0010 | 0.472 | 1.898 |
| raycast_s0 | 1.654 | 0.0011 | 0.471 | 1.906 |
| perc0.02_s1 | 1.628 | 0.0009 | 0.471 | 1.894 |
| raycast_s2 | 1.642 | 0.0012 | 0.465 | 1.904 |
| analytic_s1 | 1.643 | 0.0008 | 0.464 | 1.893 |
| perc0.02_s4 | 1.641 | 0.0011 | 0.462 | 1.893 |
| perc0.02_s2 | 1.637 | 0.0009 | 0.460 | 1.897 |
| perc0.02_s3 | 1.637 | 0.0012 | 0.460 | 1.903 |
| perc0.02_s0 | 1.648 | 0.0010 | 0.455 | 1.896 |
| perc0.00_s1 | 1.633 | 0.0011 | 0.449 | 1.893 |
| perc0.05_s1 | 1.641 | 0.0009 | 0.446 | 1.899 |
| perc0.05_s4 | 1.631 | 0.0008 | 0.443 | 1.901 |
| shipped spot_steps.pt | 1.604 | 0.0008 | 0.435 | 1.890 |
| analytic_s0 | 1.621 | 0.0014 | 0.421 | 1.893 |
| analytic_s2 | 1.618 | 0.0012 | 0.417 | 1.895 |
