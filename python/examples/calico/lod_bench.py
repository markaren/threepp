"""A/B the splat LOD policies from the calico spawn: one level for the whole
cloud, against one level per SSOG tree node.

WHY IT EXISTS. From spawn_v2 the whole-cloud rule submits 4.81 M splats looking
back into the brushy end of the canyon and 1.07 M looking forward, at 71 ms and
15 ms, for the same rule and the same camera: the rule sees ONE footprint for a
25.6 M-splat cloud and cannot tell a dense near view from a sparse one, and the
9 chunks of the level it picks are too coarse for the frustum to help. Per-node
selection gives every leaf of the asset's tree its own level.

WHAT IT PRINTS, per direction and per path: submitted splats, how many submit
ranges that took, the per-node level histogram, and wall milliseconds split into
the selection call and the render call. Headless, Vulkan, no window.

    python python/examples/calico/lod_bench.py [--asset DIR] [--spp 8] [--shots]

--shots additionally writes shots/lodnode_old.png and shots/lodnode_new.png:
the same +x frame at native resolution under each path, for a 1:1 look at rock
within a few metres of the camera.
"""

import argparse
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import threepp as tp

HERE = os.path.dirname(os.path.abspath(__file__))

# The five framings the complaint was measured in. The cloud is flipped Y-down
# to Y-up (rotation.x = pi) exactly as the demo flips it, so these are world
# directions in that frame: +x is "back into the brush", -x is the walk.
DIRECTIONS = [
    ("+x", (1.0, 0.0, 0.0)),
    ("-x", (-1.0, 0.0, 0.0)),
    ("+z", (0.0, 0.0, 1.0)),
    ("-z", (0.0, 0.0, -1.0)),
    ("up-back", (0.7, 0.7, 0.0)),
]


def histogram(levels, n_levels):
    """Per-node level counts, plus how many nodes the frustum culled."""
    h = [0] * n_levels
    culled = 0
    for l in levels:
        if l < 0:
            culled += 1
        else:
            h[l] += 1
    return h, culled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="C:/dev/splats/calico_tanks")
    ap.add_argument("--width", type=int, default=1200)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--render-scale", type=float, default=0.5)
    ap.add_argument("--spp", type=float, default=8.0, help="target splats per screen pixel")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--shots", action="store_true", help="also write the +x A/B stills")
    args = ap.parse_args()

    shots = json.load(open(os.path.join(HERE, "shots.json")))
    sp = shots["spawn_v2"]["position"]
    eye = (sp[0], sp[1] + 1.5, sp[2])

    t0 = time.perf_counter()
    cloud = tp.SplatCloud.from_sog_lod(args.asset)
    cloud.rotation.x = math.pi
    print("loaded %.1f s: %d splats, %d levels %s, %d tree nodes"
          % (time.perf_counter() - t0, cloud.splat_count, len(cloud.lod_levels),
             [l["count"] for l in cloud.lod_levels], cloud.lod_node_count), flush=True)

    n_levels = len(cloud.lod_levels)
    scene = tp.Scene()
    scene.add(cloud)
    cam = tp.PerspectiveCamera(60, args.width / args.height, 0.1, 500)
    canvas = tp.Canvas("lod_bench", width=args.width, height=args.height,
                       headless=True, vsync=False)
    rend = tp.VulkanRenderer(canvas)
    rend.render_scale = args.render_scale
    vph = int(args.height * args.render_scale)

    def run(name, d, per_node):
        cam.position.set(*eye)
        cam.look_at(eye[0] + d[0], eye[1] + d[1], eye[2] + d[2])
        for _ in range(args.warmup):
            tp.select_lod(cloud, cam, vph, target_splats_per_pixel=args.spp, per_node=per_node)
            rend.render(scene, cam)
        t_sel = t_ren = 0.0
        for _ in range(args.frames):
            a = time.perf_counter()
            lvl = tp.select_lod(cloud, cam, vph, target_splats_per_pixel=args.spp,
                                per_node=per_node)
            b = time.perf_counter()
            rend.render(scene, cam)
            c = time.perf_counter()
            t_sel += b - a
            t_ren += c - b
        ranges = cloud.submit_ranges
        sub = sum(c for _, c in ranges) if ranges else cloud.splat_count
        h, culled = histogram(cloud.lod_node_levels, n_levels) if per_node else ([], 0)
        print("%-7s %-9s lvl=%d sub=%6.3f M ranges=%3d sel=%5.2f ms render=%6.1f ms  %s"
              % (name, "per-node" if per_node else "whole", lvl, sub / 1e6, len(ranges),
                 t_sel / args.frames * 1000.0, t_ren / args.frames * 1000.0,
                 ("hist=%s culled=%d" % (h, culled)) if per_node else ""), flush=True)
        return sub, len(ranges), t_ren / args.frames * 1000.0

    print("\n%dx%d render_scale %.2f (viewport height %d px), target %.1f splats/px, "
          "%d warm-up + %d timed frames\n" % (args.width, args.height, args.render_scale,
                                              vph, args.spp, args.warmup, args.frames))
    table = []
    for name, d in DIRECTIONS:
        row = [name]
        for per_node in (False, True):
            row.append(run(name, d, per_node))
        table.append(row)
        print()

    print("%-8s | %-26s | %-26s | speedup" % ("dir", "whole cloud", "per node"))
    print("-" * 78)
    for name, old, new in table:
        print("%-8s | %6.3f M  %3d r  %6.1f ms | %6.3f M  %3d r  %6.1f ms | %5.2fx"
              % (name, old[0] / 1e6, old[1], old[2], new[0] / 1e6, new[1], new[2],
                 old[2] / max(new[2], 1e-6)))

    if args.shots:
        # Native resolution, so the crop is 1:1 pixels of the renderer's own
        # output rather than an upscale. Both paths get the same treatment and
        # the same viewport height, so the only difference is the policy.
        import numpy as np
        from PIL import Image

        out = os.path.join(HERE, "shots")
        os.makedirs(out, exist_ok=True)
        rend.render_scale = 1.0
        d = DIRECTIONS[0][1]
        cam.position.set(*eye)
        cam.look_at(eye[0] + d[0], eye[1] + d[1], eye[2] + d[2])
        for tag, per_node, spp in (("old", False, args.spp),
                                   ("new", True, args.spp),
                                   ("new_spp4", True, 4.0)):
            for _ in range(8):
                lvl = tp.select_lod(cloud, cam, args.height,
                                    target_splats_per_pixel=spp, per_node=per_node)
                rend.render(scene, cam)
            px = rend.read_pixels()
            sub = sum(c for _, c in cloud.submit_ranges)
            h, culled = histogram(cloud.lod_node_levels, n_levels) if per_node else ([], 0)
            Image.fromarray(np.asarray(px)).save(os.path.join(out, "lodnode_%s.png" % tag))
            print("wrote shots/lodnode_%s.png  spp=%g lvl=%d sub=%.3f M ranges=%d hist=%s"
                  % (tag, spp, lvl, sub / 1e6, len(cloud.submit_ranges), h), flush=True)


if __name__ == "__main__":
    main()
