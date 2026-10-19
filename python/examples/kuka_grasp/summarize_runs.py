"""summarize_runs.py — the results table from runs/<arm>_s<seed>/{train,eval}.jsonl.

    python summarize_runs.py                    # every run under runs/
    python summarize_runs.py runs/sgs_s0 ...

Per run: the build time and mean training throughput from train.jsonl, and from eval.jsonl the
final and best evaluated success on the held-out Reaching starts (overall, near-home, aligned),
and the first evaluated iterations that reached half of the final success and 0.9.
"""
import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _lines(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(x) for x in f if x.strip()]


def summarize(run):
    tr = _lines(os.path.join(run, "train.jsonl"))
    ev = sorted(_lines(os.path.join(run, "eval.jsonl")), key=lambda r: r["it"])
    start = next((r for r in tr if r.get("event") == "start"), {})
    logs = [r for r in tr if r.get("event") == "log"]
    out = {"run": os.path.basename(os.path.normpath(run)),
           "sampler": start.get("args", {}).get("sampler"),
           "envs": start.get("args", {}).get("envs"),
           "build_s": start.get("build_s"),
           "steps_per_s": logs[-1]["steps_per_s"] if logs else None,
           "iters": logs[-1]["it"] if logs else 0}
    if ev:
        fin = ev[-1]
        best = max(ev, key=lambda r: r["success"])
        half = next((r["it"] for r in ev if fin["success"] > 0 and r["success"] >= 0.5 * fin["success"]), None)
        to90 = next((r["it"] for r in ev if r["success"] >= 0.9), None)
        out.update({"eval_it": fin["it"], "success": fin["success"], "home": fin["success_home"],
                    "aligned": fin["success_aligned"], "vel_violation": fin.get("vel_violation"),
                    "best": best["success"], "best_it": best["it"], "half_it": half, "it90": to90})
    return out


def table(rows):
    head = ("| arm | envs | build | steps/s | iters | Reaching success (final) | near home | aligned "
            "| best (iter) | iter to half of final | iter to 0.9 |")
    out = [head, "|" + "---|" * 11]
    f = lambda v, fmt: "-" if v is None else format(v, fmt)
    for r in rows:
        out.append(f"| {r['run']} | {r['envs']} | {f(r['build_s'], '.1f')} s | {f(r['steps_per_s'], ',.0f')} "
                   f"| {r['iters']} | {f(r.get('success'), '.3f')} | {f(r.get('home'), '.3f')} "
                   f"| {f(r.get('aligned'), '.3f')} | {f(r.get('best'), '.3f')} ({r.get('best_it', '-')}) "
                   f"| {r.get('half_it') if r.get('half_it') is not None else '-'} "
                   f"| {r.get('it90') if r.get('it90') is not None else '-'} |")
    return "\n".join(out)


if __name__ == "__main__":
    runs = sys.argv[1:] or sorted(d for d in glob.glob(os.path.join(_HERE, "runs", "*")) if os.path.isdir(d))
    print(table([summarize(r) for r in runs]))
