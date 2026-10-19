"""Run the uniform-vs-SGS matrix and tabulate it.

    python sweep.py --envs 256 1024 4096 --seeds 0 1 2 --iters 300 --dir runs
    python sweep.py --dir runs --table            # just the table from existing logs

One log file per (sampler, envs, seed); the table reports the FINAL evaluation (mean and
spread over seeds) overall and per torque limit, plus the iteration at which the evaluation
first passed half of its final value as a crude speed measure.
"""
import argparse
import glob
import json
import os
import statistics
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def run(args):
    os.makedirs(args.dir, exist_ok=True)
    for envs in args.envs:
        for seed in args.seeds:
            for sampler in args.samplers:
                log = os.path.join(args.dir, f"{sampler}_K{envs}_s{seed}.jsonl")
                if os.path.exists(log) and not args.redo:
                    continue
                if os.path.exists(log):
                    os.remove(log)
                cmd = [sys.executable, os.path.join(_HERE, "train_sgs_pendulum.py"), "--sampler", sampler,
                       "--envs", str(envs), "--iters", str(args.iters), "--seed", str(seed),
                       "--eval-every", str(args.eval_every), "--log", log,
                       "--out", log[:-len(".jsonl")] + ".pt", *args.extra]
                print(">", " ".join(cmd[1:]), flush=True)
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def table(args):
    rows = {}
    for f in sorted(glob.glob(os.path.join(args.dir, "*.jsonl"))):
        recs = [json.loads(l) for l in open(f) if l.strip()]
        if not recs:
            continue
        last = recs[-1]
        half = next((r["it"] for r in recs if r["eval"] >= 0.5 * last["eval"]), None) if last["eval"] > 0 else None
        rows.setdefault((last["envs"], last["sampler"]), []).append((last, half))
    if not rows:
        print("no logs in", args.dir); return
    taus = None
    print(f"{'envs':>6} {'sampler':>8} {'n':>2} {'final eval':>14}  {'per torque limit (hard -> easy)':<34} {'it to half':>10}")
    for (envs, sampler), rs in sorted(rows.items()):
        ev = [r["eval"] for r, _ in rs]
        bt = [r["eval_by_tau"] for r, _ in rs]
        halves = [h for _, h in rs if h is not None]
        m = statistics.mean(ev); sd = statistics.stdev(ev) if len(ev) > 1 else 0.0
        per = " ".join(f"{statistics.mean(c):.2f}" for c in zip(*bt))
        hh = f"{statistics.mean(halves):.0f}" if halves else "-"
        print(f"{envs:>6} {sampler:>8} {len(rs):>2} {m:>8.3f} ±{sd:.3f}  {per:<34} {hh:>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(_HERE, "runs"))
    ap.add_argument("--envs", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--samplers", nargs="+", default=["uniform", "sgs"])
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--table", action="store_true", help="only print the table")
    ap.add_argument("extra", nargs="*", help="extra args for the trainer, after --")
    args = ap.parse_args()
    if not args.table:
        run(args)
    table(args)


if __name__ == "__main__":
    main()
