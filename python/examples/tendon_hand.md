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

## The task (`tendon_hand_env.py`)

An object drops into the palm-up hand. Between 1.0 and 1.8 s gravity starts to turn about an
axis in the palm plane, through 120 to 180 degrees over 1.0 to 1.8 s, so the palm ends up facing
down and the object hangs from the fingers; it stays there 1.5 to 2.6 s and turns back. From
1.5 s a pull in a direction resampled every 0.5 s ramps to 8 N by 5.0 s and stays there to the
end at 8 s. The policy commands 25 cable tensions and is rewarded for keeping the object near the
palm; every timing is drawn per env at reset.

The hand never moves. It is a fixed-base articulation and the K hands share one PhysX scene, so
there is no per-env wrist and no per-env scene gravity. Scene gravity is zero and gravity is a
per-env force m*g at every link's centre of mass — the hand's 20 moving links through the same
`write_link_force` the cables use, the object through its own batch, masses from the new
`ArticulationLink.mass`. Measured against scene gravity with a slack hand over 2 s, the two agree
per shape to the noise level in episode count, speed and rest offset. On the deploy side the same
schedule drives `world.set_gravity`.

Objects are randomized per env at build and keep their parameters for the env's life (a collider
cannot be resized or re-massed after `add_link`): sphere, capsule, box, bar or cylinder, dimensions
from `OBJ_DIMS`, density log-uniform 300–3000 kg/m^3 (about 5 to 400 g), object-material friction
0.3–1.2 under 'min' combine. Shape and dimensions are observed (84-d observation); mass and
friction are not.

## The gates

    python tendon_hand_gpu_check.py                 # A1 static + A2 dynamic parity, 8 arc points
    python tendon_hand_gpu_check.py --arc 4         # the same at 4 points per wrap arc
    python tendon_hand_env.py --sanity --envs 1024 --steps 600     # B1

A1 compares routed lengths over 30 poses spanning the joint limits and the index moment arms
by finite difference; A2 pulls one cable at 20 N for 0.25 s from rest and compares joint
angles, for a via-only cable, a thumb cable and the wrapped extensor. B1 checks the env for
NaN, prints the eager step rate, shows that a slack hand does NOT hold the object (so the task
is not trivial), measures thumb-to-index opposition under a scripted 35 N fist, and runs a fixed
15 N grip, closed 0.6 s after the drop, through the roll with no pull.

B1 as measured (RTX 4060, K=1024): link map residual 0.0000 mm; 20 moving links, 110.3 g in
total; objects 5.3–401.0 g; slack hand with pull and roll on: 0.0 % hold over 12308 episodes,
no NaN, 2.9k env-steps/s (the previous env: 3.2k); fist: closest mean tip separation 17.0 mm,
opposes. A slack palm loses rolling shapes within a second (spheres 2.7 episodes per env in 2 s,
boxes and bars 0.6), so most episodes end at the catch, as they did before the roll existed. Of
the static grips still holding when the roll begins, about nine in ten keep the object through
the turn and the turn back: the turn is survivable and the catch is the hard part.

## Training

    python train_tendon_hand.py --envs 4096 --iters 3000 --seed 0
    python train_tendon_hand.py --envs 1024 --max-minutes 12        # a bounded local smoke

`--max-minutes` stops cleanly at a wall-clock cap and still saves. Training is EAGER on
purpose: the per-step torch region calls into PhysX four times through the cable evaluation, so
a CUDA graph would replay the cable math on frozen link poses.

The log line carries `hold` (episodes that reached 8 s), `inv` (episodes still holding when the
hand had turned fully over) and `pull` (mean pull at episode end). The 12-minute local smoke,
K=1024 on the 4060, made 74 iterations at 3.3k env-steps/s and ended at hold 60.4 %, inv 82.8 %,
pull 6.49 N, still climbing at the cap — that is the case for the Idun run.

## Deploy, and the sim-to-sim check

    python play_tendon_hand.py tendon_hand_hold.pt --object sphere --density 1000
    python play_tendon_hand.py tendon_hand_hold.pt --view
    python play_tendon_hand.py tendon_hand_hold.pt --shots out/               # stills
    python play_tendon_hand.py tendon_hand_hold.pt --film out/film_sphere/    # 60 fps sequence

Runs the trained policy on the CPU hand with its REAL `TendonCable` cables — the same 25
tensions, the reference implementation of the law they go through — with PhysX's own gravity
turned by `world.set_gravity` on the checkpoint's film schedule (pronation about the finger axis,
180 degrees from 1.5 s over 1.5 s, held 2.5 s). `--roll-angle`, `--roll-axis x,y,z` and
`--no-roll` change that; `--density` and `--friction` fix the object, which is otherwise drawn
from the training ranges with `--seed`. The log prints the turn angle beside the offset and the
pull.

The smoke checkpoint, evaluated with the mean action: on the GPU env over 5118 episodes with
random schedules, hold 27.2 %, still held when turned over 58.5 % (the training log's 60 % /
83 % are with exploration noise on); on the CPU hand with its real cables and the film schedule,
5 of 20 runs held to 8 s and every one of the 20 carried its object through the half turn, the
losses coming while hanging under the ramping pull. The two rates agree, so the port holds;
the policy is what the Idun run is for.

`--shots` and `--film` render headless, out of the SAME run that prints the hold numbers —
not a replay. The roll is rendered by turning the camera, the lights and the HUD about the palm
by the roll angle: the hand is fixed and gravity turns, and the picture of that under a fixed
camera is a hand that turned under fixed gravity. `--film` writes `frame_0001.png` … at 1280x720
and a `contact.png` contact sheet (3x2: settled, mid-turn, hanging, hanging at full pull,
mid-turn back, the end). There is no ffmpeg in this tree; encode the sequence yourself:

    ffmpeg -framerate 60 -i out/film_sphere/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 tendon_hand.mp4

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

`hold` is the fraction of episodes that reached the full 8 s, `inv` the fraction still holding
when the hand had turned fully over, and `pull` the mean pull those episodes were carrying when
they ended. The pilot's `steps/s` sizes the array: 3000 iterations at K=4096 and horizon 32 are
393M env-steps, and the job's walltime is 3 h with `--max-minutes` saving whatever is trained if
that is not enough; raise `--time` or lower `ITERS` from the pilot's rate.

Knobs, all through `--export=ALL,NAME=value`: `ENVS` (4096), `ITERS` (3000), `RESERVE_MIN` (5),
`EXTRA` for extra trainer flags.
