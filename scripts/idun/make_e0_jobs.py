"""Write the E0-lite jobs.tsv that scripts/idun/spot_suite.slurm reads. CPU only, nothing is scored.

    python scripts/idun/make_e0_jobs.py --out jobs.tsv                 # Idun paths (the defaults)
    python scripts/idun/make_e0_jobs.py --out jobs.tsv --per-task 5    # + the sbatch lines to run it

One line per item: label <TAB> checkpoint <TAB> score seed <TAB> extras, where extras is a comma list
of legacy (the hard:analytic reconciliation cell), steer (held-out flat steering), repeat (a same-seed
re-run for the determinism check), or 'none'.

Items (125): the 25 array runs x {best, latest} at S_sel = 1 and S_test = 2; the controls at 1 and 2;
the shipped checkpoint and push2 at noise seeds 3-6; one exact repeat of the shipped checkpoint at
seed 1. The first lines are the pilot (task 0): shipped s1, its repeat, push2 s1, analytic_s0 best s1
with the legacy cell, shipped s2. spot_steps_push.pt is deliberately absent: it shoved the flat lanes
and regressed steering (c090e3e0), so it is no reference for anything here.
"""
import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(_HERE)), "python", "examples", "spot"))
from score_matrix import CONDITIONS   # noqa: E402  the array's condition list, one source of truth

S_SEL, S_TEST = 1, 2
NOISE_SEEDS = (3, 4, 5, 6)
SEEDS_PER_CONDITION = 5
CONTROLS = ("spot_steps", "spot_steps_latest", "spot_steps_push2", "spot_steps_push2_latest",
            "spot_steps_gen", "spot_steps_chain", "spot_hf_chain", "spot_steps_v2")
STEER_CONTROLS = ("spot_steps", "spot_steps_push2")
NOISE_CONTROLS = ("spot_steps", "spot_steps_push2")


def items(runs, ctrl):
    tags = [f"{c}_s{s}" for s in range(SEEDS_PER_CONDITION) for c, _ in CONDITIONS]
    arr = lambda tag, w: (f"{tag}:{w}", f"{runs}/{tag}/spot_steps{'' if w == 'best' else '_latest'}.pt")
    ctl = lambda name: (f"ctrl:{name}", f"{ctrl}/{name}.pt")
    out = []

    def add(pair, seed, extras=()):
        out.append((pair[0], pair[1], seed, ",".join(extras) or "none"))

    # pilot first: the determinism pair, the positive control, the reconciliation cell, a test seed
    add(ctl("spot_steps"), S_SEL, ("steer",))
    add(ctl("spot_steps"), S_SEL, ("repeat",))
    add(ctl("spot_steps_push2"), S_SEL, ("steer",))
    add(arr("analytic_s0", "best"), S_SEL, ("legacy", "steer"))
    add(ctl("spot_steps"), S_TEST)
    done = {(o[0], o[2], "repeat" in o[3]) for o in out}
    for tag in tags:
        for w in ("best", "latest"):
            for seed in (S_SEL, S_TEST):
                pair = arr(tag, w)
                if (pair[0], seed, False) in done:
                    continue
                steer = seed == S_SEL and tag.startswith("analytic_")
                add(pair, seed, ("steer",) if steer else ())
    for name in CONTROLS:
        for seed in (S_SEL, S_TEST):
            if (f"ctrl:{name}", seed, False) in done:
                continue
            add(ctl(name), seed, ("steer",) if (seed == S_SEL and name in STEER_CONTROLS) else ())
    for name in NOISE_CONTROLS:
        for seed in NOISE_SEEDS:
            add(ctl(name), seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="/cluster/work/laht/spot_runs", help="the array's run root")
    ap.add_argument("--ctrl", default="/cluster/work/laht/spot_ctrl", help="where the controls were scp'd")
    ap.add_argument("--out", default="jobs.tsv")
    # 7: one E0 item measured 65 s on a 4070 (K=2048, build 5.4 s + 1600 steps at 37 ms), so a task
    # of 7 (plus the odd legacy/steer extra) sits around 10 min of the hour, and 125 items fit in 18
    # tasks — a single wave under the 20-GPU QOS.
    ap.add_argument("--per-task", dest="per_task", type=int, default=7)
    args = ap.parse_args()
    lines = items(args.runs.rstrip("/"), args.ctrl.rstrip("/"))
    with open(args.out, "w", newline="\n") as f:
        for label, ckpt, seed, extras in lines:
            f.write(f"{label}\t{ckpt}\t{seed}\t{extras}\n")
    n_tasks = -(-len(lines) // args.per_task)
    print(f"{len(lines)} items -> {args.out}; PER_TASK={args.per_task} -> {n_tasks} tasks "
          f"(pilot = task 0, lines 1-{args.per_task})")
    print(f"  sbatch --array=0 --export=ALL,PER_TASK={args.per_task} scripts/idun/spot_suite.slurm")
    print(f"  sbatch --array=1-{n_tasks - 1}%20 --export=ALL,PER_TASK={args.per_task} scripts/idun/spot_suite.slurm")


if __name__ == "__main__":
    main()
