# The tendon hand: CPU demo, GPU training, Idun array

Twenty joints, twenty-five routed cables, no joint motor anywhere. `tendon_hand.py` is the
reference: the cables are `threepp::TendonCable`, which applies the true frictionless-pulley
force at every via point, so the moment arms are not authored anywhere — they emerge from
where the pulleys sit, and `--selftest` measures them.

`TendonCable` puts its forces in through `addForce`/`addTorque`, which PhysX rejects under the
direct-GPU API, so the reference hand cannot be vectorized. `threepp/rl/cable.py` is the same
law as a batched torch kernel over [K envs, C cables], written back through the batch's own
`write_link_force` / `write_link_torque`. `tendon_hand_gpu_check.py` is what says the two are
the same cable.

## The reference hand (CPU)

    python tendon_hand.py --selftest        # measure the mechanics, no window
    python tendon_hand.py --grasp ball      # close on an object and pull until it slips
    python tendon_hand.py --view            # window: live grasp with every cable drawn

## The gates

    python tendon_hand_gpu_check.py                 # A1 static + A2 dynamic parity, 8 arc points
    python tendon_hand_gpu_check.py --arc 4         # the same at 4 points per wrap arc
    python tendon_hand_env.py --sanity --envs 1024 --steps 600     # B1

A1 compares routed lengths over 30 poses spanning the joint limits and the index moment arms
by finite difference; A2 pulls one cable at 20 N for 0.25 s from rest and compares joint
angles, for a via-only cable, a thumb cable and the wrapped extensor. B1 checks the env for
NaN, prints the eager step rate, shows that a slack hand does NOT hold the object (so the task
is not trivial), and measures thumb-to-index opposition under a scripted 35 N fist.

## Training

    python train_tendon_hand.py --envs 4096 --iters 3000 --seed 0
    python train_tendon_hand.py --envs 1024 --max-minutes 12        # a bounded local smoke

`--max-minutes` stops cleanly at a wall-clock cap and still saves. Training is EAGER on
purpose: the per-step torch region calls into PhysX four times through the cable evaluation, so
a CUDA graph would replay the cable math on frozen link poses.

## Deploy, and the sim-to-sim check

    python play_tendon_hand.py tendon_hand_hold.pt --object sphere --seconds 6
    python play_tendon_hand.py tendon_hand_hold.pt --view

Runs the trained policy on the CPU hand with its REAL `TendonCable` cables — the same 25
tensions, the reference implementation of the law they go through. It is the deploy path and
the end-to-end check on the port in one: after the 12-minute local smoke, the policy held all
three shapes for the full 6 s against the full 7.97 N pull, 6.7 to 11.9 mm from the palm
target, having only ever seen the torch cable.

## Idun

Once, on the login node:

    mkdir -p /cluster/work/$USER/logs /cluster/work/$USER/hand_runs

Pilot first — two jobs, 100 iterations, about ten minutes:

    sbatch --array=0-1 --export=ALL,ITERS=100 scripts/idun/hand_array.slurm

Then the array, one H100 per seed, capped at the QOS's 20 GPUs:

    sbatch --array=0-7%20 scripts/idun/hand_array.slurm

Outputs land in `/cluster/work/$USER/hand_runs/s<SEED>/` as `tendon_hand_hold.pt`,
`train.log` and `manifest.json`. The numbers to read:

    grep -H 'hold ' /cluster/work/$USER/hand_runs/s*/train.log | tail -n 40

`hold` is the fraction of episodes that reached the full 6 s, and `pull` is the mean pull
magnitude those episodes were carrying when they ended — a policy that is learning to grasp
raises the first while the second climbs on its own.

Knobs, all through `--export=ALL,NAME=value`: `ENVS` (4096), `ITERS` (3000), `RESERVE_MIN` (5),
`EXTRA` for extra trainer flags.
