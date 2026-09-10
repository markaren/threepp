"""Fresh-process runs of the crane lift, short, one variant at a time, for the open replay case.

    python run_diag.py <out> --assets <dir> --variants base:6,nolod:4,reset:4,settle:4 [--op 4] [--size 1280x720]
"""
import argparse, json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # the threepp tree
SCRIPT = os.path.join(ROOT, "python", "examples", "crane_lift.py")
VARIANTS = {
    "base": [],
    "nolod": ["--no-auto-lod"],
    "reset": ["--reset-after-warmup"],
    "settle": ["--settle", "300"],
    "settle_reset": ["--settle", "300", "--reset-after-warmup"],
    "wh": ["--warmup-hash"],
    "wh_nolod": ["--warmup-hash", "--no-auto-lod"],
    "wh_reset": ["--warmup-hash", "--reset-after-warmup"],
    "wh_notip": ["--warmup-hash", "--no-tip"],
    "wh_ff1": ["--warmup-hash", "--flush-first"],
    "wh_notip_ff1": ["--warmup-hash", "--no-tip", "--flush-first"],
    "wh_norestir": ["--warmup-hash", "--no-restir"],
    "wh_nodenoise": ["--warmup-hash", "--no-denoise"],
    "wh_noao": ["--warmup-hash", "--no-ao"],
    "wh_noprobegi": ["--warmup-hash", "--no-probe-gi"],
    "wh_hardsun": ["--warmup-hash", "--hard-sun"],
    "wh_msaa1": ["--warmup-hash", "--msaa1"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--assets", required=True)
    ap.add_argument("--variants", default="base:6,nolod:4,reset:4,settle:4")
    ap.add_argument("--op", default="4")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--extra", default="")
    ap.add_argument("--dump", action="store_true", help="also write frame dumps <out>/<name>_*.npy")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    meta_path = os.path.join(a.out, "diag_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"runs": {}}
    for spec in a.variants.split(","):
        var, n = spec.split(":")
        for k in range(int(n)):
            name = f"{var}_{k}"
            out_json = os.path.join(a.out, name + ".json")
            if os.path.exists(out_json) and os.path.getsize(out_json) > 0:
                print(f"  {name:14s} exists, skipped", flush=True)
                continue
            cmd = ([sys.executable, SCRIPT, "--op", a.op, "--size", a.size, "--assets", a.assets,
                    "--seed", "0", "--op-out", out_json] + VARIANTS[var] + (a.extra.split() if a.extra else [])
                   + (["--dump-frames", os.path.join(a.out, name)] if a.dump else []))
            t0 = time.perf_counter()
            p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            wall = time.perf_counter() - t0
            with open(os.path.join(a.out, name + ".log"), "w") as fh:
                fh.write(p.stdout[-20000:])
                if p.stderr:
                    fh.write("\n--- stderr ---\n" + p.stderr[-20000:])
            ok = p.returncode == 0 and os.path.exists(out_json)
            meta["runs"][name] = {"variant": var, "cmd": cmd, "wall_s": round(wall, 1), "rc": p.returncode,
                                  "ok": ok, "finished": time.strftime("%Y-%m-%d %H:%M:%S")}
            json.dump(meta, open(meta_path, "w"), indent=1)
            print(f"  {name:14s} {'ok' if ok else 'FAILED'} {wall:.0f} s", flush=True)
            if not ok:
                print(p.stdout[-1500:])
                print(p.stderr[-3000:])
                sys.exit(1)


if __name__ == "__main__":
    main()
