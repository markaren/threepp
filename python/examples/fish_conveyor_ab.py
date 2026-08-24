"""Run the fish conveyor on both engines and report which one you would ship.

Three arms, one belt:

    newton-voxel        Newton SolverVBD on the voxel cage the Newton example carves
    newton-physx-tets   Newton SolverVBD on the tet mesh PhysX cooked -- the arm
                        that removes the mesh as a variable
    physx               PhysX 5 deformable volumes, at PhysX's own defaults

Every arm is a fresh SUBPROCESS. That is not tidiness: PhysX allows one
PxFoundation per process, PhysX and Warp would otherwise fight over the CUDA
context, and one GL context per process is the only arrangement the renderer is
happy with. It also means a run that dies takes nothing else with it.

    python fish_conveyor_ab.py --fish scans/
    python fish_conveyor_ab.py --fish scans/ --counts 16,60 --seeds 1
    python fish_conveyor_ab.py --fish scans/ --report ab.md

TWO PROTOCOLS, because "which is faster" and "which is right" are different
questions and the honest answer to the first depends on the second.

  EQUAL BUDGET  each engine at its shipped defaults, and the question is which
                configurations fit 16.7 ms/frame. Compare quality only among the
                ones that do -- a config that misses realtime is not a candidate,
                however well behaved it is.
  ACCEPTANCE    at 60 fish: no inverted tets, worst tet edge stretch under 2.0x,
                worst cross-fish penetration under 10 mm, every fish still on the
                belt, and conveying at least 0.8 of belt speed. Pass or fail, per
                engine. This is the bar a fish-handling simulator has to clear
                before its frame rate is worth discussing.

The measured window includes skinning and a headless render, because a solver
that is quick and then needs a host round trip to be drawn has not saved anyone
anything.
"""
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fish_conveyor_common import cli_arg  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FISH = cli_arg("--fish", "", str)
COUNTS = [int(c) for c in cli_arg("--counts", "16,40,60,100", str).split(",")]
SEEDS = [int(s) for s in cli_arg("--seeds", "7,11,13", str).split(",")]
ENGINES = cli_arg("--engines", "newton-voxel,newton-physx-tets,physx", str).split(",")
SECONDS = cli_arg("--seconds", 8.0, float)
VARIANTS = cli_arg("--variants", 4, int)
SHOT_COUNT = cli_arg("--shot-count", 60, int)   # the count the paired PNGs come from
REPORT = cli_arg("--report", os.path.join(HERE, "fish_conveyor_ab_report.md"), str)
SHOT_DIR = cli_arg("--shot-dir", HERE, str)
TIMEOUT = cli_arg("--timeout", 900.0, float)    # per run, seconds

# The realtime budget, and the bar every arm is held to at SHOT_COUNT fish.
BUDGET_MS = 1000.0 / 60.0
BAR = {
    "inverted": ("no inverted tets", lambda r: r["inverted"] == 0),
    "stretch": ("worst tet edge stretch < 2.0x", lambda r: r["stretch"] < 2.0),
    "penetration": ("worst cross-fish penetration < 10 mm", lambda r: r["pen_worst_mm"] < 10.0),
    "on belt": ("every fish still on the belt", lambda r: r["onbelt"] == r["count"]),
    "conveying": ("conveying >= 0.8 x belt speed",
                  lambda r: r["convey"] >= 0.8 * r["belt"]),
}


# Arm registry: name -> (script, extra flags). The original three are unchanged;
# the rest exist to answer "are we sure the settings are correct" -- each one
# undoes exactly one of the tuning decisions the shipped Newton config made, or
# tries a regime the first A/B never visited.
ARMS = {
    "newton-voxel": ("newton_fish_conveyor.py", ["--tets", "voxel"]),
    "newton-physx-tets": ("newton_fish_conveyor.py", ["--tets", "physx"]),
    "physx": ("physx_fish_conveyor.py", []),
    # VBD with Newton's own contact pipeline settings: detect twice per substep,
    # stock 32/64 contact buffers. Same wall-clock class, none of the speed cuts.
    "newton-stock": ("newton_fish_conveyor.py",
                     ["--tets", "voxel", "--detect-interval", "0",
                      "--contact-buf", "32,64"]),
    # VBD in its natural regime: many small steps, few iterations -- the corner
    # the substep sweeps never visited (8x2 = the same 16 solver passes as 4x4).
    "newton-s8i2": ("newton_fish_conveyor.py",
                    ["--tets", "voxel", "--substeps", "8", "--iterations", "2"]),
    "newton-s8i2-stock": ("newton_fish_conveyor.py",
                          ["--tets", "voxel", "--substeps", "8", "--iterations", "2",
                           "--detect-interval", "0", "--contact-buf", "32,64"]),
    # XPBD at its best-known configuration (which --solver xpbd now implies):
    # 16 small steps x 2 iterations, relaxation 0.5 (the stock 0.9 is what made
    # stiff E diverge -- Jacobi overshoot, not a solver limit), E=1e6, and a
    # COARSE res-8 cage, which is the actual stiffness mechanism: XPBD moves a
    # correction one tet-ring per iteration, so a short chain is a stiff body.
    # The fat res-8 contact spheres are also what grips the rollers (conveying
    # 0.87x belt against 0.71x at res 14).
    "newton-xpbd": ("newton_fish_conveyor.py",
                    ["--tets", "voxel", "--solver", "xpbd"]),
    # The physx-tets arm with the contact radius swept instead of the
    # median-shortest-edge heuristic the first A/B invented.
    "newton-ptets-r15": ("newton_fish_conveyor.py",
                         ["--tets", "physx", "--self-radius", "0.15",
                          "--self-margin", "0.24"]),
    "newton-ptets-r35": ("newton_fish_conveyor.py",
                         ["--tets", "physx", "--self-radius", "0.35",
                          "--self-margin", "0.56"]),
    # The hand-rolled solver from the soft-gripper demo: corotational XPBD
    # (zero-at-rest rows, mass-split Jacobi) on the res-8 cage, analytic roller
    # contact with a positional Coulomb stick. Its defaults ARE its shipped
    # configuration -- no extra flags here by design.
    "warp-corot": ("warp_fish_conveyor.py", []),
}


def command(engine, count, seed, shot=None):
    """One arm's command line. Everything shared is passed explicitly.

    --scale-jitter 0 on BOTH sides. Newton scales each soft mesh at spawn for
    free and PhysX's cook cache cannot, so leaving Newton's 0.07 default on
    would compare a jittered catch against a uniform one.
    """
    if engine not in ARMS:
        sys.exit(f"unknown engine {engine!r} (know: {', '.join(ARMS)})")
    script, extra = ARMS[engine]
    common = ["--fish", FISH, "--count", str(count), "--seed", str(seed),
              "--variants", str(VARIANTS), "--scale-jitter", "0",
              "--ab", "--ab-seconds", str(SECONDS)]
    if shot:
        common += ["--shot", shot, "--shot-time", str(SECONDS)]
    return [sys.executable, os.path.join(HERE, script)] + common + extra


def run_one(engine, count, seed, shot=None):
    """Spawn, wait, and pull the one JSON line back out. None if it failed.

    The exit code is deliberately not the pass criterion: both examples tear
    down a GL context and a CUDA context on the way out and have been seen to
    exit non-zero after printing a complete, valid result. The JSON line is the
    contract; a missing one is the failure.
    """
    cmd = command(engine, count, seed, shot)
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"  {engine:<18} {count:>4} seed {seed} | TIMEOUT after {TIMEOUT:.0f} s", flush=True)
        return None
    out = p.stdout or ""
    line = next((l for l in out.splitlines() if l.startswith("AB_JSON ")), None)
    if line is None:
        tail = (p.stderr or out)[-1200:]
        print(f"  {engine:<18} {count:>4} seed {seed} | FAILED (exit {p.returncode})\n{tail}",
              flush=True)
        return None
    r = json.loads(line[len("AB_JSON "):])
    r["engine"] = engine                       # the arm's name, not the solver's
    r["wall_s"] = time.perf_counter() - t0
    r["exit"] = p.returncode
    print(f"  {engine:<18} {count:>4} seed {seed} | {r['ms_frame']:7.2f} ms "
          f"{r['fps']:6.1f} fps | inv {r['inverted']:>4} | stretch {r['stretch']:5.2f} | "
          f"pen {r['pen_worst_mm']:5.1f} mm | {r['onbelt']}/{r['count']} on belt | "
          f"{r['wall_s']:.0f} s wall", flush=True)
    return r


def crop(path, out, size=800):
    """An 800 px 1:1 centre crop, so a claim about the fish can be checked.

    A downscaled 1600x900 frame is evidence of nothing: every artefact worth
    arguing about is a few pixels wide. Silently skipped when Pillow is absent --
    the full frames are still written.
    """
    try:
        from PIL import Image
    except ImportError:
        return None
    im = Image.open(path)
    w, h = im.size
    box = (max(0, (w - size) // 2), max(0, (h - size) // 2))
    im.crop((box[0], box[1], box[0] + size, box[1] + size)).save(out)
    return out


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def aggregate(rows):
    """One row per (engine, count): the median over seeds, worst case on quality.

    Speed is a median because it is a sample of one machine's throughput. Quality
    is a WORST over seeds, because a solver that inverts a tet on one layout in
    three has a failure mode, not a good average.
    """
    keys = sorted({(r["engine"], r["count"]) for r in rows},
                  key=lambda k: (ENGINES.index(k[0]) if k[0] in ENGINES else 99, k[1]))
    out = []
    for engine, count in keys:
        g = [r for r in rows if r["engine"] == engine and r["count"] == count]
        out.append({
            "engine": engine, "count": count, "seeds": len(g),
            "ms_frame": median([r["ms_frame"] for r in g]),
            "fps": median([r["fps"] for r in g]),
            "tets": g[0]["tets"], "cage_verts": g[0]["cage_verts"],
            "inverted": max(r["inverted"] for r in g),
            "vol_lo": min(r["vol_lo"] for r in g), "vol_hi": max(r["vol_hi"] for r in g),
            "stretch": max(r["stretch"] for r in g),
            "droop": min(r["droop"] for r in g),
            "pen_pairs": max(r["pen_pairs"] for r in g),
            "pen_worst_mm": max(r["pen_worst_mm"] for r in g),
            "convey": median([r["convey"] for r in g]),
            "belt": g[0]["belt"],
            "onbelt": min(r["onbelt"] for r in g),
            "vram": max(r["vram"] for r in g),
            "readback_ms": median([r["readback_ms"] for r in g]) if "readback_ms" in g[0] else None,
        })
    return out


def write_report(path, agg, rows, shots, cut):
    L = []
    L.append("# Newton VBD vs PhysX 5 deformable volumes, on the same fish conveyor\n")
    L.append(f"_{time.strftime('%Y-%m-%d %H:%M')} - {len(rows)} runs, "
             f"{SECONDS:.0f} s of settled conveying each, headless, one subprocess per run._\n")
    L.append("Every arm shares the scans, the auto-orientation, the decimated render "
             "surface, the barycentric bind, the belt, the roller pitch, the rails, the "
             "spawn layout and seed, the GPU skinning path, the headless renderer and the "
             "quality metrics. The solver -- and in one arm the tetrahedral mesh -- is what "
             "differs.\n")

    L.append("## Results\n")
    L.append("Speed is the median over seeds; every quality number is the WORST over seeds, "
             "because a solver that inverts a tet on one layout in three has a failure mode "
             "rather than a good average. Penetration is measured between cage vertices of "
             "different fish at a fixed 30 mm thickness, so it compares across cages.\n")
    L.append("| arm | fish | ms/frame | fps | realtime | tets | inverted | volume | stretch | "
             "droop | pen pairs | worst pen | convey | on belt | VRAM |")
    L.append("|---|---:|---:|---:|:--:|---:|---:|---|---:|---:|---:|---:|---:|:--:|---:|")
    for a in agg:
        rt = "yes" if a["ms_frame"] <= BUDGET_MS else "no"
        L.append(f"| {a['engine']} | {a['count']} | {a['ms_frame']:.1f} | {a['fps']:.0f} | {rt} | "
                 f"{a['tets']} | {a['inverted']} | {a['vol_lo']:.2f}-{a['vol_hi']:.2f} | "
                 f"{a['stretch']:.2f} | {a['droop']:.2f} | {a['pen_pairs']} | "
                 f"{a['pen_worst_mm']:.1f} mm | {a['convey']:+.3f} of {a['belt']:.2f} | "
                 f"{a['onbelt']}/{a['count']} | {a['vram']:.0f} MB |")
    L.append("")

    L.append("## Protocol 1 - equal budget (16.7 ms/frame at 60 Hz)\n")
    for engine in ENGINES:
        fits = [a for a in agg if a["engine"] == engine and a["ms_frame"] <= BUDGET_MS]
        misses = [a for a in agg if a["engine"] == engine and a["ms_frame"] > BUDGET_MS]
        if not fits and not misses:
            continue
        if fits:
            best = max(fits, key=lambda a: a["count"])
            L.append(f"- **{engine}** holds 60 Hz to **{best['count']} fish** "
                     f"({best['ms_frame']:.1f} ms, {best['tets']} tets); at that count "
                     f"{best['inverted']} inverted tets, {best['stretch']:.2f}x worst stretch, "
                     f"{best['pen_worst_mm']:.1f} mm worst penetration.")
        else:
            L.append(f"- **{engine}** does not reach 60 Hz at any tested count "
                     f"(cheapest {min(a['ms_frame'] for a in misses):.1f} ms).")
        if misses:
            L.append("  - misses the budget at: "
                     + ", ".join(f"{a['count']} fish ({a['ms_frame']:.1f} ms)" for a in misses))
    L.append("")

    L.append(f"## Protocol 2 - acceptance bar at {SHOT_COUNT} fish\n")
    L.append("| arm | " + " | ".join(name for name, _ in BAR.values()) + " | verdict |")
    L.append("|---|" + "|".join([":--:"] * len(BAR)) + "|:--:|")
    for engine in ENGINES:
        a = next((x for x in agg if x["engine"] == engine and x["count"] == SHOT_COUNT), None)
        if a is None:
            L.append(f"| {engine} | " + " | ".join(["-"] * len(BAR)) + " | not run |")
            continue
        cells, ok = [], True
        for _, test in BAR.values():
            passed = bool(test(a))
            ok = ok and passed
            cells.append("PASS" if passed else "**FAIL**")
        L.append(f"| {engine} | " + " | ".join(cells) +
                 f" | {'**PASS**' if ok else '**FAIL**'} |")
    L.append("")
    for engine in ENGINES:
        a = next((x for x in agg if x["engine"] == engine and x["count"] == SHOT_COUNT), None)
        if a is None:
            continue
        fails = [name for name, (_, test) in BAR.items() if not test(a)]
        if fails:
            L.append(f"- **{engine}** fails on {', '.join(fails)}: inverted {a['inverted']}, "
                     f"stretch {a['stretch']:.2f}x, worst penetration {a['pen_worst_mm']:.1f} mm, "
                     f"{a['onbelt']}/{a['count']} on belt, conveying {a['convey']:+.3f} m/s.")
        else:
            L.append(f"- **{engine}** clears the bar at {a['ms_frame']:.1f} ms/frame "
                     f"({a['fps']:.0f} fps).")
    L.append("")

    if shots:
        L.append(f"## Frames ({SHOT_COUNT} fish, same camera, same sim time)\n")
        for engine, paths in shots.items():
            L.append(f"- **{engine}**: `{paths['full']}`" +
                     (f", 1:1 crop `{paths['crop']}`" if paths.get("crop") else
                      " (no 1:1 crop -- Pillow is not installed)"))
        L.append("")

    L.append("## Where this is still not apples-to-apples\n")
    L.append("- **Different tetrahedral meshes in two of the three arms.** Newton carves a "
             "voxel cage; PhysX cooks a conforming collision mesh plus a voxelised "
             "simulation mesh. Volume ratio and edge stretch are then the same measurement "
             "on different meshes. That is exactly what `newton-physx-tets` removes -- read "
             "it against `physx` for a mesh-controlled comparison and against "
             "`newton-voxel` for the cost of the mesh alone.")
    L.append("- **PhysX simulates two meshes per fish, Newton one.** The tet counts in the "
             "table are the collision meshes. PhysX additionally integrates a voxel "
             "simulation mesh and skins the collision mesh to it; that work is in its "
             "milliseconds but not in its tet count.")
    L.append("- **Different substep counts, on purpose.** Each engine runs at its own "
             "shipped rate: Newton 4 substeps x 4 VBD iterations, PhysX 1 substep x 20 "
             "solver iterations. Equalising substeps would favour whichever solver the "
             "chosen number happened to suit. Protocol 1 is the honest form of the "
             "comparison: fixed wall-clock budget, whatever each engine spends it on.")
    L.append("- **Skinning is zero-copy for Newton and a readback for PhysX.** Newton's cage "
             "vertices are already Warp arrays; PhysX's live in GPU memory it owns, with no "
             "bridge into a foreign CUDA context, so each frame copies them device->host and "
             "back up. It is a real cost of using PhysX from this pipeline, not a "
             "measurement artefact, and it is reported separately as `readback_ms` in the "
             "raw JSON so it can be subtracted if the question is only about the solver.")
    L.append("- **Rollers are cylinders in Newton and capsules in PhysX** (PhysX has no "
             "cylinder primitive). Over the belt the two surfaces are identical; the "
             "capsule's hemispherical caps sit outside rails the fish cannot pass. The "
             "kinematic drive differs though: PhysX needs a fresh setKinematicTarget per "
             "substep, ~300 bound calls, which is inside its frame time.")
    L.append("- **No per-fish scale jitter, on either side.** PhysX's cook cache keys on the "
             "source geometry and applies rotation and translation only, so a per-fish scale "
             "would mean one cook per fish. Both arms run with it off; yaw and position "
             "jitter are unchanged.")
    L.append("- **The same seed does not give the same pile twice.** Both solvers accumulate "
             "contacts with atomics, so the summation order varies run to run, and a heap of "
             "sixty soft bodies over eight seconds amplifies a last-bit difference into a "
             "visibly different arrangement. Measured: newton-voxel at 16 fish, seed 7, "
             "reported 3.03x worst stretch on one run and 2.18x on a repeat of the identical "
             "command, at 7.26 and 7.27 ms. Milliseconds are stable; WORST-case quality "
             "numbers are a sample, which is what the multi-seed worst-case aggregation is "
             "for. Do not read a 10% difference in a stretch column as a difference between "
             "engines.")
    L.append("- **One machine, one driver, one session.** Every number is an RTX 4070 on the "
             "day. The ordering should travel; the absolute milliseconds will not.")
    if cut:
        L.append(f"- **Matrix reduced.** {cut}")
    L.append("")

    L.append("## Raw\n")
    L.append("```json")
    for r in rows:
        L.append(json.dumps(r))
    L.append("```")

    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(f"\nwrote {path}", flush=True)


def main():
    if not FISH:
        sys.exit("--fish PATH is required (a scan, a directory of scans, or a list)")
    print(f"matrix: {len(ENGINES)} engines x {len(COUNTS)} counts x {len(SEEDS)} seeds = "
          f"{len(ENGINES) * len(COUNTS) * len(SEEDS)} runs, {SECONDS:.0f} s each\n", flush=True)

    rows, shots = [], {}
    t0 = time.perf_counter()
    for count in COUNTS:
        for seed in SEEDS:
            for engine in ENGINES:
                # One paired frame per engine, from the first seed at SHOT_COUNT
                # fish: same camera, same sim time, same catch.
                shot = None
                if count == SHOT_COUNT and seed == SEEDS[0]:
                    shot = os.path.join(SHOT_DIR, f"fish_conveyor_ab_{engine}.png")
                r = run_one(engine, count, seed, shot)
                if r is None:
                    continue
                rows.append(r)
                if shot and os.path.isfile(shot):
                    c = crop(shot, os.path.join(SHOT_DIR, f"fish_conveyor_ab_{engine}_crop.png"))
                    shots[engine] = {"full": shot, "crop": c}
    print(f"\n{len(rows)} runs in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    if not rows:
        sys.exit("every run failed -- nothing to report")

    cut = ""
    if len(SEEDS) < 3 or len(COUNTS) < 4:
        cut = (f"seeds {SEEDS} and counts {COUNTS} rather than the full "
               f"3 seeds x (16, 40, 60, 100), for time. Quality columns are a worst-case "
               f"over {len(SEEDS)} layout(s), not 3.")
    write_report(REPORT, aggregate(rows), rows, shots, cut)


if __name__ == "__main__":
    main()
