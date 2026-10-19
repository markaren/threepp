"""Kilometre flight over a GeoScene region pack: the streaming yardstick.

A scripted drone flight, headless at 1080p (auto-LOD on, ray query on where the GPU has it), around a closed
loop over a geodata region pack, the same loop flown `--laps` times. GeoScene streams its content around the
camera (CellStreamer cells for the forest and the parked cars, the tile quadtree, the grass scatter): cells
are built, removed, and re-attached from its cache on the next lap. Every frame records the wall time of
GeoScene.update (the content builds) and of render(), the renderer's CPU/GPU timings, tlas_stats
(structural_rebuilds, slot_edits, history_resets), the streamers' stats and the device memory. It writes
<out>/<tag>.json (a summary and every frame) and stills at fixed sim times, and prints where the frames
above twice the median come from.

Not collected by pytest. Run it with the package on sys.path:

    python python/tests/bench_geo_flight.py --pack <geodata>/aalesund --tag slots_on
    THREEPP_VULKAN_NO_SLOTS=1 python python/tests/bench_geo_flight.py --pack <geodata>/aalesund --tag slots_off

The pack: `python scripts/geodata/fetch_norway_terrain.py` (see geodata/README.md); --pack defaults to the
repository's geodata/aalesund. The clock is pinned (sim_time = frame / 60) with replayable_lod_schedule off,
the way a game runs; --replayable-lod turns the replay schedule back on.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "python"))
import threepp as tp  # noqa: E402

FPS = 60.0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pack", default=os.environ.get("THREEPP_GEO_PACK", os.path.join(REPO, "geodata", "aalesund")))
    ap.add_argument("--tag", default="flight")
    ap.add_argument("--out", default=os.path.join(os.getcwd(), "geo_flight_out"))
    ap.add_argument("--seconds", type=float, default=300.0, help="sim seconds flown")
    ap.add_argument("--laps", type=int, default=2, help="the loop is flown this many times in --seconds")
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--loop", default="200,600,2600,520",
                    help="cx,cz,a,b: the ellipse flown (world metres), centre and semi-axes along x and z")
    ap.add_argument("--altitude", type=float, default=90.0, help="metres above the highest ground near the path")
    ap.add_argument("--warm", type=int, default=120, help="frames held at the start before the clock runs")
    ap.add_argument("--stills", default="20,75,95,170,225,245", help="sim seconds to save a still at")
    ap.add_argument("--replayable-lod", action="store_true", help="keep the pinned clock's replayable LOD schedule")
    ap.add_argument("--urban-forest-cap", type=int, default=80000, help="GeoScene urban_forest_cap")
    ap.add_argument("--stream-budget", type=int, default=2, help="GeoScene stream_budget (cell builds per update)")
    ap.add_argument("--tile-stage", type=int, default=0,
                    help="GeoScene tile_stage_frames: terrain tiles attached hidden, swapped in this many updates later")
    ap.add_argument("--prefetch", type=float, default=0.0,
                    help="GeoScene stream_prefetch: cells built and attached hidden this far ahead of their ring")
    return ap.parse_args(argv)


class Loop:
    """An ellipse at a height that clears the ground ahead and behind, parametrised by sim time."""

    def __init__(self, geo, cx, cz, a, b, period, altitude, n=2048):
        self.cx, self.cz, self.a, self.b, self.period = cx, cz, a, b, period
        ground = []
        for i in range(n):
            u = 2.0 * math.pi * i / n
            x, z = cx + a * math.cos(u), cz + b * math.sin(u)
            ground.append(max(geo.height_at(x, z), geo.sea_level))
        self.over_land = sum(1 for i in range(n) if ground[i] > geo.sea_level + 0.5) / n
        win = max(1, n // 48)                       # the highest ground within ~1/48 of the loop
        high = [max(ground[(i + j) % n] for j in range(-win, win + 1)) for i in range(n)]
        sm = max(1, n // 96)                        # then a moving average, so the climb is smooth
        self.alt = [sum(high[(i + j) % n] for j in range(-sm, sm + 1)) / (2 * sm + 1) + altitude for i in range(n)]
        self.n = n

    def at(self, t):
        f = (t / self.period) % 1.0
        u = 2.0 * math.pi * f
        x, z = self.cx + self.a * math.cos(u), self.cz + self.b * math.sin(u)
        k = f * self.n
        i0 = int(k) % self.n
        w = k - int(k)
        y = self.alt[i0] * (1.0 - w) + self.alt[(i0 + 1) % self.n] * w
        return x, y, z

    def look(self, t):
        x0, y0, z0 = self.at(t)
        x1, _, z1 = self.at(t + 0.5)
        dx, dz = x1 - x0, z1 - z0
        d = math.hypot(dx, dz) or 1.0
        ahead = 400.0                                # look 400 m ahead, ~13 degrees down
        return x0 + dx / d * ahead, y0 - 90.0, z0 + dz / d * ahead


def save_still(renderer, path):
    px = renderer.read_pixels()
    try:
        from PIL import Image
        Image.fromarray(px).save(path)
    except ImportError:
        import numpy as np
        np.save(os.path.splitext(path)[0] + ".npy", px)


def pct(xs, q):
    s = sorted(xs)
    return s[int(q * (len(s) - 1))] if s else float("nan")


def main(argv=None):
    args = parse_args(argv)
    if not os.path.isfile(os.path.join(args.pack, "region.json")):
        print(f"no region pack at {args.pack} (see geodata/README.md)", file=sys.stderr)
        return 2
    os.makedirs(args.out, exist_ok=True)
    w, h = (int(v) for v in args.size.lower().split("x"))
    cx, cz, a, b = (float(v) for v in args.loop.split(","))
    frames = int(round(args.seconds * FPS))
    period = args.seconds / max(1, args.laps)
    still_frames = {int(round(float(s) * FPS)): float(s) for s in args.stills.split(",") if s.strip()}

    canvas = tp.Canvas("geo-flight", width=w, height=h, headless=True, vsync=False)
    renderer = tp.VulkanRenderer(canvas, flush_frames=1)
    renderer.replayable_lod_schedule = bool(args.replayable_lod)
    t0 = time.perf_counter()
    geo = tp.GeoScene(args.pack, urban_forest_cap=args.urban_forest_cap, stream_budget=args.stream_budget,
                      stream_prefetch=args.prefetch, tile_stage_frames=args.tile_stage)
    load_s = time.perf_counter() - t0
    scene = tp.Scene()
    scene.background = 0x9fb8d0
    scene.add(geo)
    scene.add(tp.HemisphereLight(0xcfe0ff, 0x4a4030, 0.6))
    sun = tp.DirectionalLight(0xfff2e0, 3.0)
    sun.position.set(-0.45, 0.75, 0.35)
    scene.add(sun)
    cam = tp.PerspectiveCamera(60.0, w / h, 0.5, 6000.0)
    loop = Loop(geo, cx, cz, a, b, period, args.altitude)
    print(f"geo-flight {args.tag}: pack {os.path.basename(os.path.normpath(args.pack))} loaded in {load_s:.1f} s, "
          f"loop {2 * math.pi * math.sqrt((a * a + b * b) / 2) / 1000:.1f} km, {loop.over_land * 100:.0f} % over land, "
          f"{period:.0f} s a lap, NO_SLOTS={os.environ.get('THREEPP_VULKAN_NO_SLOTS', '')}, "
          f"replayable_lod={renderer.replayable_lod_schedule}, prefetch={args.prefetch:g} m, tile_stage={args.tile_stage}", flush=True)

    def place(t):
        x, y, z = loop.at(t)
        lx, ly, lz = loop.look(t)
        cam.position.set(x, y, z)
        cam.look_at(lx, ly, lz)

    # The start: the first update builds the whole ring, the first render admits it all; then the scene
    # settles at the start position before the clock runs.
    place(0.0)
    a0 = time.perf_counter()
    geo.update(cam.position)
    renderer.sim_time = 0.0
    renderer.render(scene, cam)
    first_ms = (time.perf_counter() - a0) * 1e3
    for _ in range(args.warm):
        geo.update(cam.position)
        renderer.render(scene, cam)

    rows = []
    ts0 = dict(renderer.tlas_stats)
    for k in range(frames):
        t = k / FPS
        place(t)
        a1 = time.perf_counter()
        geo.update(cam.position)
        a2 = time.perf_counter()
        renderer.sim_time = t
        renderer.render(scene, cam)
        a3 = time.perf_counter()
        ts = renderer.tlas_stats
        ft = renderer.frame_timings or {}
        ss = geo.stream_stats
        mem = renderer.gpu_memory
        rows.append([
            round(t, 4), round((a3 - a1) * 1e3, 3), round((a2 - a1) * 1e3, 3), round((a3 - a2) * 1e3, 3),
            round(float(ft.get("cpu_ensure_scene_ms", 0.0)), 3), round(float(ft.get("cpu_record_ms", 0.0)), 3),
            round(float(ft.get("cpu_frame_ms", 0.0)), 3), round(float(ft.get("gpu_total_ms", 0.0)), 3),
            int(ts["structural_rebuilds"]), int(ts["slot_edits"]), int(ts["history_resets"]),
            ss["forest"]["builds"], ss["forest"]["adds"], ss["forest"]["removes"], ss["forest"]["pending"],
            ss["cars"]["builds"], ss["cars"]["adds"], ss["cars"]["removes"],
            ss["forest"]["active"], ss["forest"]["cached"],
            int(mem["allocated_bytes"]), int(mem["reserved_bytes"]),
            1 if k in still_frames else 0,
            int(ts["admissions_deferred"]), ss["forest"]["shows"] + ss["cars"]["shows"],
        ])
        if k in still_frames:
            save_still(renderer, os.path.join(args.out, f"{args.tag}_t{still_frames[k]:05.1f}.png"))
    cols = ["t", "whole", "update", "render", "ensure", "record", "cpu_frame", "gpu", "sr", "se", "hr",
            "f_builds", "f_adds", "f_removes", "f_pending", "c_builds", "c_adds", "c_removes",
            "f_active", "f_cached", "mem_alloc", "mem_reserved", "still", "deferred", "shows"]
    summary = summarise(rows, cols, ts0, frames, period)
    summary.update({"tag": args.tag, "pack": os.path.basename(os.path.normpath(args.pack)),
                    "no_slots": os.environ.get("THREEPP_VULKAN_NO_SLOTS", ""),
                    "replayable_lod": bool(args.replayable_lod), "prefetch": args.prefetch, "tile_stage": args.tile_stage, "size": [w, h],
                    "load_s": round(load_s, 1),
                    "first_frame_ms": round(first_ms, 1), "loop_over_land": round(loop.over_land, 3),
                    "ray_tracing": bool(getattr(renderer, "ray_tracing", True))})
    with open(os.path.join(args.out, f"{args.tag}.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "cols": cols, "rows": rows}, f)
    print("geo-flight SUMMARY " + json.dumps(summary), flush=True)
    renderer.sim_time = None
    return 0


def summarise(rows, cols, ts0, frames, period):
    c = {n: i for i, n in enumerate(cols)}
    # A still's read_pixels drains the device: that frame and the next are not flight frames.
    skip = set()
    for i, r in enumerate(rows):
        if r[c["still"]]:
            skip.update((i, i + 1))
    idx = [i for i in range(len(rows)) if i not in skip]
    whole = [rows[i][c["whole"]] for i in idx]
    med = statistics.median(whole)
    meds = {n: statistics.median(rows[i][c[n]] for i in idx) for n in ("update", "ensure", "record", "cpu_frame")}
    meds["wait"] = statistics.median(rows[i][c["render"]] - rows[i][c["cpu_frame"]] for i in idx)

    def delta(i, n):
        return rows[i][c[n]] - (rows[i - 1][c[n]] if i else 0)

    slow = [i for i in idx if rows[i][c["whole"]] > 2.0 * med]
    by_cause = {}
    for i in slow:
        r = rows[i]
        # what the frame spent over the median, by part: the content build (GeoScene.update), the
        # renderer's scene build (ensureSceneBuilt: admission, slot edits, rebuilds), its recording,
        # and the rest of render() (waiting for the GPU, presenting)
        wait = r[c["render"]] - r[c["cpu_frame"]]
        parts = {"content_build": r[c["update"]] - meds["update"],
                 "scene_build": r[c["ensure"]] - meds["ensure"],
                 "record": r[c["record"]] - meds["record"],
                 "gpu_wait": wait - meds["wait"]}
        cause = max(parts, key=parts.get)
        ev = []
        if delta(i, "sr") > 0:
            ev.append("rebuild")
        if delta(i, "se") > 0:
            ev.append("slot_edit")
        if r[c["f_builds"]] or r[c["c_builds"]]:
            ev.append("cell_build")
        if r[c["f_adds"]] or r[c["c_adds"]]:
            ev.append("cell_add")
        if r[c["f_removes"]] or r[c["c_removes"]]:
            ev.append("cell_remove")
        if r[c["shows"]]:
            ev.append("cell_show")
        key = cause + ("[" + ",".join(ev) + "]" if ev else "")
        by_cause[key] = by_cause.get(key, 0) + 1

    def lap_stats(lo, hi):
        sel = [i for i in idx if lo <= rows[i][c["t"]] < hi]
        wv = [rows[i][c["whole"]] for i in sel]
        if not wv:
            return {}
        m = statistics.median(wv)
        return {"frames": len(sel), "median": round(m, 2), "p99": round(pct(wv, 0.99), 2),
                "max": round(max(wv), 1), "over_2x": sum(1 for v in wv if v > 2 * m),
                "over_16_7": sum(1 for v in wv if v > 16.7),
                "mem_alloc_mb_end": round(rows[sel[-1]][c["mem_alloc"]] / 2**20, 1),
                "mem_reserved_mb_end": round(rows[sel[-1]][c["mem_reserved"]] / 2**20, 1)}

    ev_frames = [i for i in idx if delta(i, "se") > 0 or delta(i, "sr") > 0]
    ev_whole = [rows[i][c["whole"]] for i in ev_frames]
    quiet = [rows[i][c["whole"]] for i in idx if i not in set(ev_frames)
             and not (rows[i][c["f_builds"]] or rows[i][c["c_builds"]])]
    last = rows[-1]
    laps = {f"lap{j + 1}": lap_stats(j * period, (j + 1) * period) for j in range(int(round(frames / FPS / period)))}
    return {
        "frames": len(idx), "median": round(med, 2), "p99": round(pct(whole, 0.99), 2), "max": round(max(whole), 1),
        "over_2x_median": len(slow), "over_16_7": sum(1 for v in whole if v > 16.7),
        "part_medians": {n: round(v, 3) for n, v in meds.items()},
        "gpu_median": round(statistics.median(rows[i][c["gpu"]] for i in idx), 2),
        "structural_rebuilds": last[c["sr"]] - int(ts0["structural_rebuilds"]),
        "slot_edits": last[c["se"]] - int(ts0["slot_edits"]),
        "history_resets": last[c["hr"]] - int(ts0["history_resets"]),
        "event_frames": len(ev_frames),
        "event_frame_ms": ({"median": round(statistics.median(ev_whole), 2), "p90": round(pct(ev_whole, 0.9), 2),
                            "max": round(max(ev_whole), 1)} if ev_whole else {}),
        "quiet_median": round(statistics.median(quiet), 2) if quiet else None,
        "cell_builds": sum(r[c["f_builds"]] + r[c["c_builds"]] for r in rows),
        "cell_adds": sum(r[c["f_adds"]] + r[c["c_adds"]] for r in rows),
        "cell_removes": sum(r[c["f_removes"]] + r[c["c_removes"]] for r in rows),
        "forest_cells_cached_end": last[c["f_cached"]],
        "admissions_deferred": last[c["deferred"]] - rows[0][c["deferred"]],
        "cell_shows": sum(r[c["shows"]] for r in rows),
        "slow_frames_by_cause": dict(sorted(by_cause.items(), key=lambda kv: -kv[1])),
        "laps": laps,
        "mem_alloc_mb": {"start": round(rows[0][c["mem_alloc"]] / 2**20, 1),
                         "max": round(max(r[c["mem_alloc"]] for r in rows) / 2**20, 1),
                         "end": round(last[c["mem_alloc"]] / 2**20, 1)},
        "worst": [(round(rows[i][c["t"]], 2), rows[i][c["whole"]], rows[i][c["update"]], rows[i][c["ensure"]])
                  for i in sorted(idx, key=lambda i: -rows[i][c["whole"]])[:10]],
    }


if __name__ == "__main__":
    sys.exit(main())
