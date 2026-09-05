"""E2: same-instant multi-view. Two measurements, one script.

1. The temporal-coherence figure. Three cameras watch a fast mover. With
   secondary views, every camera samples ONE simulated instant: the mover is
   in one place. A sequential-render simulator renders camera 1, steps the
   world, renders camera 2, steps, renders camera 3: the mover is in three
   places. Both are rendered here and composed into one triptych PNG.

2. The scaling curve. Frame time for N in {1, 2, 4, 8} views produced from
   one scene build per frame, against the sequential baseline of N render()
   calls per frame with the primary camera swapped between them. Interleaved
   A/B, auto-exposure pinned, upscalers off, warm-up frames dropped, medians
   over repeated blocks, GPU timings from frame_timings() next to wall time.

    python multiview_bench.py --out e2/           # both
    python multiview_bench.py --out e2/ --figure  # figure only
    python multiview_bench.py --out e2/ --bench   # curve only

Views are persistent (add_view allocates a full deferred chain), so the bench
adds the views once per N and never per frame.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import numpy as np

W, H = 1280, 720
DT = 1.0 / 60.0
MOVER_SPEED = 20.0  # m/s: a third of a metre between two 60 Hz frames


def build(tp, renderer, aspect):
    scene = tp.Scene()
    scene.background = tp.Color(0x304050)
    sun = tp.DirectionalLight(tp.Color(0xFFFFFF), 3.0)
    sun.position.set(20, 30, 15)
    scene.add(sun)

    def mesh(geom, color):
        mat = tp.MeshStandardMaterial()
        mat.color = tp.Color(color)
        m = tp.Mesh(geom, mat)
        scene.add(m)
        return m

    ground = mesh(tp.BoxGeometry(40, 0.5, 40), 0x556B45)
    ground.position.y = -0.25
    for i in range(7):
        p = mesh(tp.BoxGeometry(0.6, 3.0, 0.6), 0x808890)
        p.position.set(-6 + 2 * i, 1.5, -4)
    # The fast mover: a 0.6 m ball crossing the stage at MOVER_SPEED.
    mover = mesh(tp.SphereGeometry(0.3, 24, 12), 0xE04030)

    def place(t):
        mover.position.set(-6.0 + MOVER_SPEED * (t - 0.2), 1.2, 0.0)

    cams = []
    for i, (x, z) in enumerate(((-4.0, 6.0), (0.0, 7.0), (4.0, 6.0))):
        c = tp.PerspectiveCamera(55, aspect, 0.1, 100)
        c.position.set(x, 2.5, z)
        c.look_at(0, 1, 0)
        cams.append(c)
    return scene, cams, place


def pin(renderer):
    renderer.auto_exposure = False
    for name in ("fsr", "dlss"):
        if hasattr(renderer, name):
            try:
                setattr(renderer, name, False)
            except Exception:
                pass


def figure(tp, out):
    from PIL import Image, ImageDraw
    canvas = tp.Canvas("e2 figure", W, H, vsync=False, headless=True)
    renderer = tp.VulkanRenderer(canvas)
    pin(renderer)
    scene, cams, place = build(tp, renderer, W / H)
    t0 = 0.5  # mover mid-stage
    # Warm the temporal pipeline on the primary and settle the views.
    for f in range(30):
        renderer.sim_time = f * DT
        place(t0 - (30 - f) * DT)
        renderer.render(scene, cams[1])
    handles = [renderer.add_view(c, W, H) for c in cams]
    assert all(h > 0 for h in handles), handles
    for f in range(20):
        renderer.sim_time = (30 + f) * DT
        place(t0)
        renderer.render(scene, cams[1])
    # Same instant: one render, three views.
    renderer.sim_time = 51 * DT
    place(t0)
    renderer.render(scene, cams[1])
    same = [renderer.read_view_rgb_pixels(h) for h in handles]
    for h in handles:
        renderer.remove_view(h)
    # Sequential: the world keeps stepping while each camera renders.
    seq = []
    t = 52 * DT
    for i, c in enumerate(cams):
        renderer.sim_time = t
        place(t0 + i * DT)
        for _ in range(3):  # let the primary's temporal resolve settle on this camera
            renderer.render(scene, c)
        seq.append(renderer.read_pixels())
        t += DT
    rows = [same, seq]
    labels = ["same instant (secondary views, one scene build)",
              "sequential renders (the world steps between cameras)"]
    tile = Image.new("RGB", (3 * W, 2 * H), (0, 0, 0))
    draw = ImageDraw.Draw(tile)
    for r, imgs in enumerate(rows):
        for c, im in enumerate(imgs):
            a = np.asarray(im)
            if a.ndim == 3 and a.shape[2] == 4:
                a = a[:, :, :3]
            tile.paste(Image.fromarray(np.ascontiguousarray(a[:, :, :3])), (c * W, r * H))
        draw.text((8, r * H + 8), labels[r], fill=(255, 255, 255))
    path = os.path.join(out, "e2_fast_mover.png")
    tile.save(path)
    # The measurable version of the picture: mover centroid per camera.
    def centroid(a):
        a = np.asarray(a)[:, :, :3].astype(np.int32)
        red = (a[:, :, 0] > 150) & (a[:, :, 1] < 90) & (a[:, :, 2] < 90)
        ys, xs = np.nonzero(red)
        return (float(xs.mean()), float(ys.mean()), int(red.sum())) if xs.size else None
    stats = {"same_instant": [centroid(a) for a in same], "sequential": [centroid(a) for a in seq]}
    json.dump(stats, open(os.path.join(out, "e2_fast_mover.json"), "w"), indent=1)
    print("figure:", path)
    print("centroids:", json.dumps(stats))


def bench(tp, out, frames, reps):
    canvas = tp.Canvas("e2 bench", W, H, vsync=False, headless=True)
    renderer = tp.VulkanRenderer(canvas)
    pin(renderer)
    scene, cams, place = build(tp, renderer, W / H)
    # More cameras for N = 8: reuse the three poses around a ring.
    allcams = list(cams)
    while len(allcams) < 8:
        i = len(allcams)
        c = tp.PerspectiveCamera(55, W / H, 0.1, 100)
        ang = 2 * math.pi * i / 8
        c.position.set(7 * math.sin(ang), 2.5, 7 * math.cos(ang))
        c.look_at(0, 1, 0)
        allcams.append(c)
    frame = [0]

    def one_frame(cam):
        renderer.sim_time = frame[0] * DT
        place((frame[0] % 60) / 60.0)
        renderer.render(scene, cam)
        frame[0] += 1

    def gpu_ms():
        try:
            ft = renderer.frame_timings
            if callable(ft):
                ft = ft()
        except Exception:
            return None
        for k in ("gpu_total_ms", "gpu_total", "gpuTotalMs", "gpu_ms"):
            if k in ft:
                return float(ft[k])
        return None

    def block(mode, n):
        # mode 'same': one render per frame, n-1 secondary views live.
        # mode 'seq':  n renders per frame of the SAME primary camera: the
        #              ideal sequential simulator, N x the single-view cost,
        #              with no temporal-history penalty for swapping cameras.
        # mode 'swap': n renders per frame, primary camera swapped each time,
        #              which is what one renderer context rendering N cameras
        #              in turn actually costs here (history reset per swap).
        walls, gpus = [], []
        cams_for = {"same": [allcams[0]], "seq": [allcams[0]] * n, "swap": allcams[:n]}[mode]
        for _ in range(10):  # warm-up
            for c in cams_for:
                one_frame(c)
        for _ in range(frames):
            t0 = time.perf_counter()
            g = 0.0
            gok = True
            for c in cams_for:
                one_frame(c)
                m = gpu_ms()
                gok = gok and m is not None
                g += m or 0.0
            walls.append((time.perf_counter() - t0) * 1000.0)
            gpus.append(g if gok else None)
        return statistics.median(walls), (statistics.median([x for x in gpus if x is not None]) if any(x is not None for x in gpus) else None)

    results = []
    for _ in range(5):
        one_frame(allcams[0])
    for n in (1, 2, 4, 8):
        acc = {m: {"w": [], "g": []} for m in ("same", "seq", "swap")}
        for r in range(reps):
            for m in ("same", "seq", "swap"):  # interleaved A/B/C
                # The secondary views exist ONLY during the same-instant block:
                # a view left attached renders on every sequential call too, and
                # the baseline would then be timing N views N times over.
                handles = [renderer.add_view(c, W, H) for c in allcams[1:n]] if m == "same" else []
                assert all(h > 0 for h in handles), handles
                w, g = block(m, n)
                for h in handles:
                    renderer.remove_view(h)
                acc[m]["w"].append(w)
                acc[m]["g"].append(g)

        def med(xs):
            xs = [x for x in xs if x is not None]
            return statistics.median(xs) if xs else None

        row = {"views": n}
        for m, key in (("same", "same_instant"), ("seq", "sequential_ideal"), ("swap", "sequential_swap")):
            row[key + "_wall_ms"] = med(acc[m]["w"])
            row[key + "_gpu_ms"] = med(acc[m]["g"])
        row["wall_ratio_vs_ideal"] = row["sequential_ideal_wall_ms"] / row["same_instant_wall_ms"]
        if row["same_instant_gpu_ms"] and row["sequential_ideal_gpu_ms"]:
            row["gpu_ratio_vs_ideal"] = row["sequential_ideal_gpu_ms"] / row["same_instant_gpu_ms"]
        results.append(row)
        print(json.dumps(row))
    meta = {"size": [W, H], "frames": frames, "reps": reps}
    try:
        import subprocess
        meta["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,clocks.sm", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    json.dump({"meta": meta, "rows": results}, open(os.path.join(out, "e2_scaling.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="e2")
    ap.add_argument("--figure", action="store_true")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--frames", type=int, default=150)
    ap.add_argument("--reps", type=int, default=3)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    import threepp as tp
    both = not (a.figure or a.bench)
    if a.figure or both:
        figure(tp, a.out)
    if a.bench or both:
        bench(tp, a.out, a.frames, a.reps)


if __name__ == "__main__":
    main()
