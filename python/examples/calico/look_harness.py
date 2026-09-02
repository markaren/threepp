"""Iterate on the LOOK without the physics: the scan, a standing Spot, two poses.

`spot_calico.py` bakes a 311k-triangle surface, probes the floor with dropped
balls and runs a policy before it can show you a frame. None of that changes how
the light falls, and all of it costs a minute per iteration, so this harness
loads the scan exactly the way spot_calico does (`SplatCloud.from_sog_lod`,
`rotation.x = pi + pi/2`, the same `Frame` translation, Z-up world), stands Spot
on a flat collider at the known probed floor of spine waypoint 4, and renders the
trackside and lookback poses. It shares no state with spot_calico -- it IMPORTS
its Frame so the two cannot drift -- and it writes nothing but PNGs.

    set PYTHONPATH=C:/dev/threepp/python
    py -3.14 python/examples/calico/look_harness.py --look before --tag before
    py -3.14 python/examples/calico/look_harness.py --look after  --tag after
    py -3.14 python/examples/calico/look_harness.py --look after --sky env --tag envsky
    py -3.14 python/examples/calico/look_harness.py --look after --no-shadows --tag noshadow
    py -3.14 python/examples/calico/look_harness.py --look after --cam lookback \
        --no-robot --exposures 0.75,0.9,1.05

Every run writes shots/wp4_<tag>_<cam>.png at --size (default 1600x1000).
`--exposures` instead renders ONE scene at each value in turn -- same splats,
same LOD, same sun, only the tone-mapping constant differs -- writes
shots/wp4b_exp<v>_<cam>.png plus a side-by-side shots/wp4b_exposure_ab.png, and
prints the slab-mean / p99 table that chose apply_look's pinned default.
"""
import argparse, json, math, os, sys, time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPOT = os.path.join(os.path.dirname(_HERE), "spot")
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))       # repo/python
sys.path.insert(0, _HERE)
sys.path.insert(0, _SPOT)
sys.path.insert(0, os.path.join(_SPOT, "scratch_distillation"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import threepp as tp
import calico_look
from spot_calico import Frame, DEFAULT_ASSET, SHOTS_JSON     # the frame, not a copy of it
from spot_deploy import build_spot, fetch_assets, default_q, add_to_isaac, Z0
from scratch_env import STIFF_GAINS

SHOTS_DIR = os.path.join(_HERE, "shots")

# spine waypoint 4 is spot_calico's default spawn; the judge's WP2 report has its
# PROBED floor at -0.10 (the ball-drop number, not the baked-vertex one).
SPAWN_WP = 4
FLOOR_Z = -0.10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default=DEFAULT_ASSET)
    ap.add_argument("--look", choices=("before", "after"), default="after")
    ap.add_argument("--sky", choices=("proc", "env", "none"), default="proc")
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--cam", default="both",
                    help="trackside | lookback | both")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--size", default="1600x1000")
    ap.add_argument("--spp", type=float, default=8.0)
    ap.add_argument("--exposure", type=float, default=None,
                    help="override calico_look.apply_look's pinned default")
    ap.add_argument("--exposures", default=None,
                    help="comma list, e.g. 0.75,0.9,1.05: render the SAME scene "
                         "at each, print the slab/p99 table and write "
                         "shots/wp4b_exposure_ab.png")
    ap.add_argument("--crop", default=None,
                    help="also write shots/<name>.png, the 1:1 shadow crop of "
                         "the trackside frame (SHADOW_CROP below)")
    ap.add_argument("--no-robot", action="store_true",
                    help="skip the physics, Spot and the contact shadows "
                         "(the exposure A/B does not need them)")
    ap.add_argument("--render-scale", type=float, default=1.0)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()
    tag = args.tag or args.look
    os.makedirs(SHOTS_DIR, exist_ok=True)

    with open(SHOTS_JSON) as f:
        shots = json.load(f)
    F = Frame(shots)
    spine = np.array([F(w) for w in shots["spine"]["waypoints"]], np.float64)
    SPAWN = spine[SPAWN_WP].copy()
    fwd = spine[SPAWN_WP + 1] - spine[SPAWN_WP]
    yaw = math.atan2(fwd[1], fwd[0])

    w, h = (int(v) for v in args.size.split("x"))
    canvas = tp.Canvas("threepp - calico look harness", width=w, height=h,
                       antialiasing=4, headless=not args.interactive, vsync=False)
    rend = tp.VulkanRenderer(canvas)
    rend.render_scale = float(args.render_scale)

    scene = tp.Scene()
    look = None
    if args.look == "before":
        # spot_calico.py as WP2 left it: a flat blue background, a generic sun
        # from the +x/-y/+z octant and a strong hemisphere fill.
        scene.background = tp.Background(0x9fb6cc)
        scene.add(tp.HemisphereLight(0xd8e6ff, 0x6b5a44, 0.55))
        sun = tp.DirectionalLight(0xfff4e0, 2.6)
        sun.position.set(12, -14, 22)
        sun.cast_shadow = True
        sun.set_shadow_frustum(-14, 14, 14, -14)
        sun.set_shadow_bias(-0.0005)
        scene.add(sun)
        rend.shadow_map_enabled = True
        rend.auto_exposure = False
        rend.tone_mapping = tp.ToneMapping.ACESFilmic
        rend.tone_mapping_exposure = 1.1

    print(f"[splat] loading {args.asset} ...")
    t0 = time.perf_counter()
    cloud = tp.SplatCloud.from_sog_lod(args.asset)
    cloud.rotation.x = math.pi + math.pi / 2
    cloud.position.set(float(F.t[0]), float(F.t[1]), float(F.t[2]))
    scene.add(cloud)
    print(f"[splat] {time.perf_counter() - t0:.2f}s, levels "
          f"{[l['count'] for l in cloud.lod_levels]}")

    if args.look == "after":
        kw = {} if args.exposure is None else {"exposure": args.exposure}
        look = calico_look.apply_look(scene, rend, sky=args.sky, asset=args.asset,
                                      cloud_position=(F.t[0], F.t[1], F.t[2]), **kw)
        look.follow((SPAWN[0], SPAWN[1], FLOOR_Z))

    camera = tp.PerspectiveCamera(75, w / max(h, 1), 0.05, 4000)
    camera.up.set(0, 0, 1)

    art = None
    rs = (SPAWN[0], SPAWN[1], FLOOR_Z + 0.55)
    if not args.no_robot:
        art, rs = _stand_spot(scene, SPAWN, yaw)

    shadows = None
    if art is not None and args.look == "after" and not args.no_shadows:
        shadows = calico_look.ContactShadows(scene, art, floor_z=FLOOR_Z)
        shadows.update()
        print("[look] contact shadows: feet at " +
              " ".join(f"({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})"
                       for p in shadows.foot_positions()))

    # ── the two poses ─────────────────────────────────────────────────────────
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    ts_eye = np.array([SPAWN[0] + right[0] * 1.2, SPAWN[1] + right[1] * 1.2,
                       FLOOR_Z + 0.6])
    cams = {
        "trackside": (ts_eye, np.array(rs[:3], float) + [0.0, 0.0, 0.15]),
        "lookback": (np.array(F(shots["cameras"]["lookback"]["pos"])),
                     np.array(F(shots["cameras"]["lookback"]["look"]))),
    }
    want = list(cams) if args.cam == "both" else [args.cam]

    def draw(n=4):
        for _ in range(n):
            tp.select_lod(cloud, camera, h, target_splats_per_pixel=args.spp)
            cloud.update(camera)
            if look is not None and look.env_cloud is not None:
                look.env_cloud.update(camera)
            rend.render(scene, camera)

    def save(path):
        try:
            rend.save_frame(path)
        except TypeError:
            rend.save_frame(scene, camera, path)

    if args.interactive:
        pos, tgt = cams[want[0]]
        camera.position.set(*pos)
        controls = tp.OrbitControls(camera, canvas)
        controls.target.set(*tgt)
        controls.update()

        def frame():
            tp.select_lod(cloud, camera, canvas.size()[1], target_splats_per_pixel=args.spp)
            cloud.update(camera)
            if look is not None and look.env_cloud is not None:
                look.env_cloud.update(camera)
            controls.update()
            rend.render(scene, camera)
        canvas.animate(frame)
        return

    for name in want:
        pos, tgt = cams[name]
        camera.position.set(*pos)
        camera.look_at(*tgt)
        camera.update_projection_matrix()
        if args.exposures:
            evs = [float(v) for v in args.exposures.split(",")]
            paths = []
            for ev in evs:
                rend.tone_mapping_exposure = ev
                draw()
                out = os.path.join(SHOTS_DIR, f"wp4b_exp{ev:g}_{name}.png")
                save(out)
                paths.append((ev, out))
                print(f"saved {out}")
            _exposure_report(paths, name)
        else:
            t0 = time.perf_counter()
            draw()
            out = os.path.join(SHOTS_DIR, f"wp4_{tag}_{name}.png")
            save(out)
            print(f"saved {out}   ({(time.perf_counter() - t0) / 4 * 1000:.1f} ms/frame, "
                  f"4 frames, GPU shared)")
            if args.crop and name == "trackside":
                _crop(out, os.path.join(SHOTS_DIR, f"{args.crop}.png"))
    canvas.close()


# The rect shots/wp4_shadow_crop_*.png were cut with, recovered by matching that
# crop back into shots/wp4_after_trackside.png (exact, zero pixel difference):
# 740x380 at (380, 620) of the 1600x1000 trackside frame -- the front feet and
# the rock they stand on. Every shadow crop has to use it or the A/B is not one.
SHADOW_CROP = (380, 620, 740, 380)


def _crop(src, dst, rect=SHADOW_CROP):
    from PIL import Image
    x, y, w, h = rect
    Image.open(src).convert("RGB").crop((x, y, x + w, y + h)).save(dst)
    print(f"saved {dst}   (1:1 crop {w}x{h} at ({x}, {y}))")


# ── the exposure A/B ──────────────────────────────────────────────────────────
# The slab is the sunlit rock the robot stands on, and in the lookback frame it
# fills the bottom-centre of the picture. Measure it there and nowhere else: a
# whole-frame mean is dominated by the bright brush on the right, which is the
# asset's own pallor and not something exposure should be asked to fix.
SLAB_RECT = (0.375, 0.75, 0.625, 1.0)          # x0, y0, x1, y1 as fractions


def _luma(a):
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722


def _stats(path):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    h, w = a.shape[:2]
    x0, y0, x1, y1 = SLAB_RECT
    slab = a[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    return float(_luma(slab).mean()), float(np.percentile(_luma(a), 99.0))


def _exposure_report(paths, cam):
    from PIL import Image
    print(f"\n  exposure   slab mean   p99      (slab = centre-bottom quarter, "
          f"{cam})")
    rows = []
    for ev, p in paths:
        m, p99 = _stats(p)
        ok = "  <-- 165-185 and p99<250" if (165 <= m <= 185 and p99 < 250) else ""
        print(f"  {ev:>7.2f}   {m:9.1f}   {p99:6.1f}{ok}")
        rows.append((ev, p, m, p99))
    ims = [Image.open(p).convert("RGB") for _, p, _, _ in rows]
    tw = sum(i.width for i in ims) + 8 * (len(ims) - 1)
    strip = Image.new("RGB", (tw, ims[0].height), (24, 24, 24))
    x = 0
    for i in ims:
        strip.paste(i, (x, 0))
        x += i.width + 8
    out = os.path.join(SHOTS_DIR, "wp4b_exposure_ab.png")
    strip.save(out)
    print(f"  strip -> {out}\n")


def _stand_spot(scene, SPAWN, yaw):
    """A STATIC Spot: no bake, no policy, a flat collider at the probed floor."""
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.002,
                          max_substeps=20)
    g = tp.BufferGeometry()
    q = [(-8, -8), (8, -8), (8, 8), (-8, -8), (8, 8), (-8, 8)]
    g.set_attribute("position", np.array(
        [[SPAWN[0] + a, SPAWN[1] + b, FLOOR_Z] for a, b in q], np.float32))
    world.add_static_trimesh(tp.Mesh(g, tp.MeshStandardMaterial()))

    art, meshes = build_spot(world, fetch_assets(), gains=STIFF_GAINS)
    for m in meshes:
        m.cast_shadow = True
        m.receive_shadow = True
        scene.add(m)
    hs = math.sin(yaw / 2.0)
    art.reset(tp.Vector3(float(SPAWN[0]), float(SPAWN[1]), Z0 + FLOOR_Z + 0.03),
              tp.Quaternion(0.0, 0.0, hs, math.cos(yaw / 2.0)))
    for _ in range(200):                                   # settle onto the plane
        art.set_drive_targets(default_q[add_to_isaac].astype(np.float32))
        world.step(0.02)
    rs = art.root_state()
    print(f"[spot] standing at ({rs[0]:+.3f}, {rs[1]:+.3f}, {rs[2]:.3f})")
    return art, rs


if __name__ == "__main__":
    main()
