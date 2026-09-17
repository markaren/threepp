"""Phase 1 timing: eye view render, receptors, OpticLobe step and the whole loop rate.

    py -3.14 python/examples/flyeye/timing.py [--rounds 5 --frames 60]

Scene with real content: a textured ground plane, eight lit textured meshes that spin,
a hemisphere light and a sun. A small primary canvas (--primary px square), and one or
two eye views (403 or 256 px, 90 deg FOV) yawed +-45 deg on a vehicle that drives
forward. Blocks are interleaved: every round runs every configuration once, each block
adds its views, warms up, times --frames loop iterations and removes the views. Medians
over all timed iterations of all rounds; torch.cuda.synchronize around every GPU stage.

Per iteration:
    render   renderer.render() + the frame fence (FrameTensors.sync)
    recept   EyeView.receptors() for every eye (BGRA -> luma -> box_eye), synchronized
    step     OpticLobe.step on (721,) for one eye, on (2, 721) batched for two
    loop     render + recept + step, and its rate in Hz
The per-eye view cost is (render with views - render without) / number of views at the
same flush setting.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore", message=".*[Ss]parse.*")

import threepp as tp  # noqa: E402

from flyeye.eye import EyeView, configure_sensor_renderer  # noqa: E402
from flyeye.lattice import HexLattice  # noqa: E402
from flyeye.optic_lobe import OpticLobe  # noqa: E402


def checker_png(path, n=256, cells=8):
    from PIL import Image

    y, x = np.mgrid[0:n, 0:n] * cells // n
    base = ((x + y) % 2).astype(np.float32)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.08, (n, n)).astype(np.float32)
    g = np.clip(0.25 + 0.5 * base + noise, 0, 1)
    rgb = np.stack([g * 0.9, g, g * 0.8], -1)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)
    return path


def build_scene(tex_path):
    scene = tp.Scene()
    scene.background = tp.Color(0x8FA8C0)
    tex = tp.TextureLoader().load(str(tex_path), tp.ColorSpace.SRGB)
    tex.wrap_s = tp.TextureWrapping.Repeat
    tex.wrap_t = tp.TextureWrapping.Repeat
    tex.repeat.set(20, 20)

    ground_mat = tp.MeshStandardMaterial()
    ground_mat.map = tex
    ground_mat.roughness = 0.9
    ground = tp.Mesh(tp.PlaneGeometry(200, 200), ground_mat)
    ground.rotate_x(-math.pi / 2)
    scene.add(ground)

    tex2 = tp.TextureLoader().load(str(tex_path), tp.ColorSpace.SRGB)
    geoms = [tp.BoxGeometry(1.5, 1.5, 1.5), tp.SphereGeometry(0.9, 32, 16), tp.TorusKnotGeometry(0.6, 0.2),
             tp.CylinderGeometry(0.6, 0.6, 2.0, 24)]
    movers = []
    for i in range(8):
        m = tp.MeshStandardMaterial()
        m.map = tex2
        m.color = tp.Color([0xFF8866, 0x88DD88, 0x8899FF, 0xEEDD77][i % 4])
        m.roughness = 0.5
        mesh = tp.Mesh(geoms[i % 4], m)
        ang = i * 2 * math.pi / 8
        mesh.position.set(6 * math.sin(ang), 1.2, -8 - 6 * math.cos(ang))
        scene.add(mesh)
        movers.append(mesh)

    scene.add(tp.HemisphereLight(tp.Color(0xFFFFFF), tp.Color(0x404048), 1.0))
    sun = tp.DirectionalLight(tp.Color(0xFFFFFF), 3.0)
    sun.position.set(20, 30, 10)
    scene.add(sun)
    return scene, movers


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--warm", type=int, default=15)
    ap.add_argument("--primary", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path(r"C:\dev\_flyeye\timing.json"))
    ap.add_argument("--snapshot", type=Path, default=None,
                    help="write the primary and both 403 px eye images of one frame to this directory and exit")
    args = ap.parse_args(argv)

    canvas =tp.Canvas("flyeye timing", width=args.primary, height=args.primary, headless=True, vsync=False)
    r = tp.VulkanRenderer(canvas, flush_frames=1)
    settings = configure_sensor_renderer(r)
    tex = checker_png(Path(tempfile.gettempdir()) / "flyeye_timing_checker.png")
    scene, movers = build_scene(tex)
    main_cam = tp.PerspectiveCamera(60, 1.0, 0.1, 500)
    main_cam.position.set(0, 6, 8)
    main_cam.look_at(0, 1, -8)
    vehicle = tp.Group()
    vehicle.position.set(0, 1.5, 4)
    scene.add(vehicle)
    eye_cams = []
    for yaw in (math.radians(45), math.radians(-45)):
        c = tp.PerspectiveCamera(90, 1.0, 0.05, 500)
        c.rotation.y = yaw
        vehicle.add(c)
        eye_cams.append(c)

    lobe = OpticLobe(device="cuda")
    lat = HexLattice()
    lat.centers = lat.centers.cuda()
    dt = 0.01
    lobe.fade_in(torch.full((721,), 0.5, device="cuda"), dt)
    frame = [0]

    def advance():
        frame[0] += 1
        t = frame[0] * dt
        r.sim_time = t
        vehicle.position.z = 4 - 2.0 * (t % 6.0)
        for i, m in enumerate(movers):
            m.rotation.y = 0.7 * t * (1 + i % 3)

    for _ in range(30):  # pipelines
        advance()
        r.render(scene, main_cam)
    # interop on the primary too, so the frame fence waits for the GPU frame in every
    # configuration, including the one without eyes
    from threepp.torch_frames import FrameTensors

    primary = FrameTensors(r, 0, ("color",))

    if args.snapshot is not None:
        from PIL import Image

        args.snapshot.mkdir(parents=True, exist_ok=True)
        eyes = [EyeView(r, c, size=403, lattice=lat) for c in eye_cams]
        advance()
        r.render(scene, main_cam)
        for e in eyes:
            e.arm()
        for _ in range(20):
            advance()
            r.render(scene, main_cam)
        for i, e in enumerate(eyes):
            x = e.receptors()
            Image.fromarray(e.color[..., [2, 1, 0]].cpu().numpy()).save(args.snapshot / f"eye{i}.png")
            print(f"eye{i}: receptors mean {x.mean().item():.3f} std {x.std().item():.3f}")
        Image.fromarray(primary.color[..., [2, 1, 0]].cpu().numpy()).save(args.snapshot / "primary.png")
        for e in eyes:
            e.close()
        return

    configs = [(f, n, s) for f in (1, 3) for (n, s) in ((0, 0), (1, 403), (2, 403), (1, 256), (2, 256))]
    samples = {c: dict(render=[], recept=[], step=[], loop=[]) for c in configs}
    # standalone step: single-eye path and a batch of 2, interleaved
    step_alone = {1: [], 2: []}
    x2 = torch.full((2, 721), 0.5, device="cuda")
    for rnd in range(args.rounds):
        order = configs if rnd % 2 == 0 else configs[::-1]
        for cfg in order:
            flush, n_eyes, size = cfg
            r.set_flush_frames(flush)
            eyes = [EyeView(r, eye_cams[i], size=size, lattice=lat) for i in range(n_eyes)]
            advance()
            r.render(scene, main_cam)
            for e in eyes:
                e.arm()
            if n_eyes:
                lobe.fade_in(torch.full((n_eyes, 721), 0.5, device="cuda") if n_eyes > 1
                             else torch.full((721,), 0.5, device="cuda"), dt)
            s = samples[cfg]
            for k in range(args.warm + args.frames):
                advance()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                r.render(scene, main_cam)
                primary.sync()
                t1 = time.perf_counter()
                xs = [e.receptors() for e in eyes]
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                if n_eyes == 1:
                    lobe.step(xs[0], dt)
                elif n_eyes == 2:
                    lobe.step(torch.stack(xs), dt)
                torch.cuda.synchronize()
                t3 = time.perf_counter()
                if k >= args.warm:
                    s["render"].append((t1 - t0) * 1e3)
                    s["recept"].append((t2 - t1) * 1e3)
                    s["step"].append((t3 - t2) * 1e3)
                    s["loop"].append((t3 - t0) * 1e3)
            for e in eyes:
                e.close()
            if primary.stale:
                raise RuntimeError("primary interop went stale after removing a view")
        # standalone step timing
        for B in ((1, 2) if rnd % 2 == 0 else (2, 1)):
            lobe.reset(None if B == 1 else 2)
            for k in range(20 + args.frames):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                lobe.step(x2[0] if B == 1 else x2, dt)
                torch.cuda.synchronize()
                if k >= 20:
                    step_alone[B].append((time.perf_counter() - t0) * 1e3)
        print(f"round {rnd + 1}/{args.rounds} done", flush=True)

    rows = []
    base = {f: med(samples[(f, 0, 0)]["render"]) for f in (1, 3)}
    for cfg in configs:
        flush, n_eyes, size = cfg
        s = samples[cfg]
        row = dict(flush=flush, eyes=n_eyes, size=size, render_ms=med(s["render"]),
                   render_p90=float(np.percentile(s["render"], 90)),
                   recept_ms=med(s["recept"]), step_ms=med(s["step"]), loop_ms=med(s["loop"]),
                   loop_hz=1e3 / med(s["loop"]), n=len(s["loop"]))
        row["view_ms_per_eye"] = (row["render_ms"] - base[flush]) / n_eyes if n_eyes else 0.0
        row["recept_ms_per_eye"] = row["recept_ms"] / n_eyes if n_eyes else 0.0
        rows.append(row)
        print(f"flush {flush} eyes {n_eyes} size {size:3d}: render {row['render_ms']:6.2f} (p90 {row['render_p90']:6.2f})"
              f"  view/eye {row['view_ms_per_eye']:5.2f}  recept/eye {row['recept_ms_per_eye']:5.2f}"
              f"  step {row['step_ms']:5.2f}  loop {row['loop_ms']:6.2f} ms = {row['loop_hz']:6.1f} Hz  (N={row['n']})")
    alone = {B: med(v) for B, v in step_alone.items()}
    print(f"OpticLobe.step alone: 1 eye {alone[1]:.3f} ms, 2 eyes batched {alone[2]:.3f} ms")
    out = dict(gpu=torch.cuda.get_device_name(), primary_px=args.primary, rounds=args.rounds, frames=args.frames,
               rows=rows, step_alone_ms=alone, renderer=settings and {k: str(v) for k, v in settings.items()})
    args.out.write_text(json.dumps(out, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
