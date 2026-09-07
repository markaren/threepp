"""Run the fresh-process replay protocol of the crane lift on the FROZEN scene, one GPU job at a
time, resumable: the open-loop control, N fresh processes at seed 0, and seeds 1..9.

    python run_protocol.py <out_folder> --assets <dir> [--processes 10] [--seeds 9] [--op 65]
                           [--size 1280x720] [--allow-dirty] [--dry-run] [-- <extra crane_lift.py args>]

Outputs follow analyze.py's names so `python analyze.py <out_folder>` reads them directly:
    op_s0_noas.json/.npz            the control (anti-swing off), seed 0
    op_s0_a.json ... op_s0_j.json   N fresh processes at seed 0 (replay: every row must agree)
    op_s0_film.json/.npz            with --film: one more seed-0 process that also writes film_s0.mp4 and
                                    hero_s0_*.png (kept separate so the N above share one configuration)
    op_s1.json ... op_s9.json       one process per seed (the fan-out of the trajectory)
plus protocol_meta.json (git head, dirty state, GPU, driver, the exact command line per run,
wall seconds) and one .log per run with the script's last lines.

The scene must be frozen: the runner refuses to start while python/examples/crane_lift.py has
uncommitted changes, because the paper names the commit the manifests came from. Pass
--allow-dirty only for a rehearsal whose manifests will be thrown away.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(EXAMPLES))
SCRIPT = os.path.join(EXAMPLES, "crane_lift.py")


def sh(cmd, cwd=ROOT):
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<{e}>"


def done(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path) as fh:
            m = json.load(fh)
        return "rows" in m and os.path.exists(path[:-5] + ".npz")
    except Exception:  # noqa: BLE001
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--assets", default=os.environ.get("CRANE_ASSETS_DIR", ""))
    ap.add_argument("--processes", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=9)
    ap.add_argument("--op", type=float, default=65.0)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-control", action="store_true")
    ap.add_argument("--film", action="store_true", help="also write films and hero PNGs: from the control, and from an EXTRA seed-0 process op_s0_film so the N replay processes stay identical in configuration")
    ap.add_argument("extra", nargs="*", help="arguments passed through to crane_lift.py (after --)")
    a = ap.parse_args()
    if not a.assets:
        sys.exit("--assets (or CRANE_ASSETS_DIR) is required")

    head = sh(["git", "rev-parse", "--short=12", "HEAD"])
    dirty = sh(["git", "status", "--porcelain", "--", "python/examples/crane_lift.py"])
    if dirty and not a.allow_dirty:
        sys.exit(f"crane_lift.py has uncommitted changes ({dirty.strip()}); commit the frozen scene first, "
                 f"or pass --allow-dirty for a rehearsal")
    gpu = sh(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    os.makedirs(a.out, exist_ok=True)
    base = [sys.executable, SCRIPT, "--op", str(a.op), "--size", a.size, "--assets", a.assets] + a.extra

    jobs = []
    film = lambda name, tag: (["--op-film", os.path.join(a.out, f"film_{tag}.mp4"), "--op-png", os.path.join(a.out, f"hero_{tag}")] if a.film else [])
    if not a.no_control:
        jobs.append(("op_s0_noas", ["--seed", "0", "--no-antiswing"] + film("op_s0_noas", "off")))
    for k in range(a.processes):
        jobs.append((f"op_s0_{'abcdefghijklmnopqrstuvwxyz'[k]}", ["--seed", "0"]))
    if a.film:
        jobs.append(("op_s0_film", ["--seed", "0"] + film("op_s0_film", "s0")))     # the eleventh process: the film run, compared like the others
    for s in range(1, a.seeds + 1):
        jobs.append((f"op_s{s}", ["--seed", str(s)]))

    meta_path = os.path.join(a.out, "protocol_meta.json")
    meta = {"git_head": head, "crane_lift_dirty": bool(dirty), "gpu": gpu, "platform": sys.platform,
            "python": sys.version.split()[0], "base_command": base, "runs": {}}
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            meta["runs"] = json.load(fh).get("runs", {})

    print(f"protocol: {len(jobs)} runs of {a.op:.0f} s into {a.out}; git {head}{' DIRTY' if dirty else ''}; {gpu}")
    for name, args in jobs:
        out_json = os.path.join(a.out, name + ".json")
        cmd = base + ["--op-out", out_json] + args
        if done(out_json):
            print(f"  {name:12s} exists, skipped")
            continue
        print(f"  {name:12s} " + ("(dry run) " if a.dry_run else "") + " ".join(cmd[2:]))
        if a.dry_run:
            continue
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        wall = time.perf_counter() - t0
        with open(os.path.join(a.out, name + ".log"), "w") as fh:
            fh.write(p.stdout[-20000:])
            if p.stderr:
                fh.write("\n--- stderr ---\n" + p.stderr[-20000:])
        ok = p.returncode == 0 and done(out_json)
        meta["runs"][name] = {"command": cmd, "wall_seconds": round(wall, 1), "returncode": p.returncode, "ok": ok,
                              "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(meta_path, "w") as fh:
            json.dump(meta, fh, indent=1)
        print(f"  {name:12s} {'ok' if ok else 'FAILED'} in {wall / 60:.1f} min")
        if not ok:
            print(p.stdout[-1500:])
            sys.exit(f"run {name} failed; fix and rerun (finished runs are skipped)")
    if not a.dry_run:
        print("\nanalyze:")
        subprocess.run([sys.executable, os.path.join(HERE, "analyze.py"), a.out], cwd=ROOT)


if __name__ == "__main__":
    main()
