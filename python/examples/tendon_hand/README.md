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

The Idun array (2026-09-17, eight seeds, K=4096, 3000 iterations each, one H100 per seed, all
finished) ended at hold 78.3–86.0 %, inv 89.5–93.6 %, pull 7.16–7.46 N across the seeds, with
exploration noise on. The curve of seed 7 is typical: 59 % at iteration 100, 73 % at 400, flat
near 72 % to 1600, then a slow climb to 82 % by 3000, still rising about one point per 300
iterations at the end. The per-seed numbers are in the Idun section below.

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

The smoke checkpoint, evaluated with the mean action: on the GPU env, hold 27.2 % and still held
when turned over 58.5 % over 5118 episodes with random schedules, 20.6 % and 49.1 % over 5627
with the film schedule on every env (the training log's 60 % / 83 % are with exploration noise
on, which adds tension through the action clamp). On the CPU hand with its real cables and the
film schedule, 20 runs over five shapes and four seeds with drop orientation randomized as in
training: 5 held to 8 s, 18 were still holding when turned over, 12 were lost while hanging
under the ramping pull and 3 during a turn. The hold rate matches the GPU's; the CPU loses fewer
at the catch than the GPU env does in its first second, which is not explained. The policy is
what the Idun run is for.

The Idun checkpoints on the same CPU tally (film schedule, five shapes, drop seeds 0–3, 20 runs
per checkpoint, about 8.5 s per run):

    for s in 0 1 2 3 4 5 6 7; do for o in sphere capsule box bar cylinder; do for k in 0 1 2 3; do
        python play_tendon_hand.py hand_runs/s$s/tendon_hand_hold.pt --object $o --seed $k | tail -1
    done; done; done

Seed 2 holds 18 of 20 to the end and 18 of 20 at the full turn (offset under 120 mm at 3.0 s);
seeds 3, 5 and 6 hold 17; seed 1 is the weakest at 13 (the table is in the Idun section). Over
all 160 runs 127 held: sphere 23/32, capsule 27/32, box 26/32, bar 26/32, cylinder 25/32. Of the
33 losses, not one was in the turn back or palm-up under the full 8 N pull, the phases the smoke
checkpoint lost 12 of 20 to. Eight are the same run: the sphere with drop seed 3 (radius 16 mm,
9.4 g) is never caught by any seed, it is 69 mm off the palm at 0.5 s before the policy has
closed, a catch failure of that draw and not a hold failure. Twenty are lost in the first turn,
between 1.5 and 3.0 s, most of them objects at friction 0.31–0.43 (the capsule, cylinder and box
with drop seed 0 go in three or four seeds each) and the two heaviest, the 145 g box and the 175 g
bar; four are lost while hanging, at 4.5–5.0 s. Seed 2's own two losses are that sphere and the
22 g cylinder at friction 0.31. Seed 2 with the 34.6 g sphere of drop seed 0 holds with a worst
offset of 31 mm, mean tension 10 N at rest rising to about 20 N through the turn and the pull.

In a still or a film the ropes are coloured by their commanded tension, slate when slack
through amber and red to warm white at 40 N, with the ramp as a legend bar bottom-left and the
object's name, mass and friction top-left (both corners no finger reaches: text sprites are not
depth-tested against the hand). `--pull-max N` carries the training ramp on past the trained
8 N up to N, `--roll-hang S` keeps the hand turned over for S seconds, and `--until-drop S` ends
the take S seconds after the object leaves the hand and prints the pull it let go at. That is a
measurement of margin, not of skill: the policy never saw more than 8 N. Seed 2 at drop seed 0
with a 14 s hang: the sphere lets go at 26.4 N, the capsule at 17.2 N, the bar at the 40 N cap
palm-up after 19.5 s, and the box never, holding the 40 N cap for the whole 20 s; the cylinder is
the known first-turn loss at 2.3 N.

    python play_tendon_hand.py hand_runs/s2/tendon_hand_hold.pt --object sphere --seed 0 \
        --pull-max 40 --roll-hang 14 --seconds 20 --until-drop 1.0 --film out/fail_sphere/

`--shots` and `--film` render headless, out of the SAME run that prints the hold numbers —
not a replay. The roll is rendered by turning the camera, the lights and the HUD about the palm
by the roll angle: the hand is fixed and gravity turns, and the picture of that under a fixed
camera is a hand that turned under fixed gravity. `--film` writes `frame_0001.png` … at 1280x720
and a `contact.png` contact sheet (3x2: settled, mid-turn, hanging, hanging at full pull,
mid-turn back, the end). There is no ffmpeg in this tree; encode the sequence yourself (the
`imageio-ffmpeg` pip package ships one, `python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"`):

    ffmpeg -framerate 60 -i out/film_sphere/frame_%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 tendon_hand.mp4

## Idun

Once, on the login node:

    mkdir -p /cluster/work/$USER/logs /cluster/work/$USER/hand_runs

Pilot first — two jobs, 100 iterations, about 17 minutes on an H100:

    sbatch --array=0-1 --export=ALL,ITERS=100 scripts/idun/hand_array.slurm

Then the array, one H100 per seed, capped at the QOS's 20 GPUs:

    sbatch --array=0-7%20 scripts/idun/hand_array.slurm

Outputs land in `/cluster/work/$USER/hand_runs/s<SEED>/` as `tendon_hand_hold.pt`,
`train.log` and `manifest.json`. The numbers to read:

    grep -H 'hold ' /cluster/work/$USER/hand_runs/s*/train.log | tail -n 40

`hold` is the fraction of episodes that reached the full 8 s, `inv` the fraction still holding
when the hand had turned fully over, and `pull` the mean pull those episodes were carrying when
they ended. The pilot measured 13.2k env-steps/s on an H100 at K=4096 and horizon 32, 9.9 s per
iteration, 101 iterations in 16.5 min. 3000 iterations are about 8.3 h, so the job's walltime is
10 h; `--max-minutes` saves whatever is trained if a node is slower.

The eight-seed array of 2026-09-17, every seed the full 3000 iterations. `log` columns are the
last training log line (exploration noise on); `CPU` columns are the film-schedule tally on the
reference hand described under Deploy, 20 runs per checkpoint:

| seed | log hold | log inv | log pull | CPU held to 8 s | CPU held at full turn |
|------|---------:|--------:|---------:|----------------:|----------------------:|
| s0   |   79.6 % |  91.4 % |   7.26 N |           15/20 |                 16/20 |
| s1   |   78.3 % |  89.8 % |   7.16 N |           13/20 |                 14/20 |
| s2   |   86.0 % |  93.6 % |   7.46 N |           18/20 |                 18/20 |
| s3   |   80.8 % |  92.0 % |   7.26 N |           17/20 |                 17/20 |
| s4   |   82.0 % |  90.2 % |   7.23 N |           16/20 |                 16/20 |
| s5   |   84.4 % |  91.5 % |   7.32 N |           17/20 |                 17/20 |
| s6   |   81.7 % |  89.5 % |   7.20 N |           17/20 |                 18/20 |
| s7   |   82.4 % |  92.0 % |   7.32 N |           14/20 |                 15/20 |

The training log ranks the seeds only loosely: s5 is second in the log and tied third on the CPU,
s7 is mid-table in the log and second to last on the CPU. Pick the film seed by the CPU tally.
The checkpoints are 736 KB each; fetch the whole tree from your own machine, not from the login
node:

    scp -r <user>@idun-login1.hpc.ntnu.no:/cluster/work/<user>/hand_runs .

Knobs, all through `--export=ALL,NAME=value`: `ENVS` (4096), `ITERS` (3000), `RESERVE_MIN` (5),
`EXTRA` for extra trainer flags.
