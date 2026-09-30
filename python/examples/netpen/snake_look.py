"""A look at the snake's gaits: lateral undulation, then eel-like, in plain murky water.

Headless Vulkan, 1280x720, 30 fps, piped to x264 (demo_common.Encoder). The physics is
snake_model.Snake at 240 Hz; the camera follows the centre of mass from a 3/4 view. Outputs
look.mp4 and look_strip.png (6 frames) to out/sweep beside this script. Called by snake_sweep.py --look.

    python snake_look.py [--seconds 10] [--out DIR]
"""
import argparse
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_EX = os.path.dirname(_HERE)
_PY = os.path.dirname(_EX)
for _p in (_HERE, _EX, _PY):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import threepp as tp

import snake_model as sm
from demo_common import Encoder

W, H, FPS = 1280, 720, 30
MURK = (0.05, 0.16, 0.17)                    # murky blue-green, linear-ish


def grid_texture(px=2048, cells=64):
    img = np.empty((px, px, 3), np.uint8)
    img[:] = (10, 14, 14)
    step = px // cells
    for k in range(0, px, step):
        img[k:k + 2, :] = (34, 44, 43)
        img[:, k:k + 2] = (34, 44, 43)
    for k in range(0, px, step * 4):
        img[k:k + 3, :] = (52, 66, 63)
        img[:, k:k + 3] = (52, 66, 63)
    return tp.data_texture(img, srgb=True)


def label(rgb, text, sub):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(rgb)
    d = ImageDraw.Draw(im)
    try:
        f1 = ImageFont.truetype("arial.ttf", 30); f2 = ImageFont.truetype("arial.ttf", 20)
    except Exception:                        # noqa: BLE001
        f1 = f2 = ImageFont.load_default()
    d.text((32, 26), text, font=f1, fill=(245, 240, 225))
    d.text((32, 66), sub, font=f2, fill=(190, 205, 200))
    return np.asarray(im)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--switch", type=float, default=4.5, help="film time of the switch to eel-like")
    ap.add_argument("--preroll", type=float, default=4.0, help="lateral swimming before the first frame")
    ap.add_argument("--out", default=os.path.join(_HERE, "out", "sweep"))
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    dt = 1.0 / 240.0
    sub = int(round(1.0 / (FPS * dt)))       # 8 substeps per frame
    world = sm.make_world(dt)
    snake = sm.Snake(world, sm.SnakeParams(), origin=(0.0, 0.0, 0.0), heading=0.0)
    a, w, d = math.radians(30), math.radians(120), math.radians(30)
    snake.set_gait("lateral", a, w, d)

    canvas = tp.Canvas("snake look", width=W, height=H, vsync=False, headless=True)
    renderer = tp.VulkanRenderer(canvas)
    renderer.tone_mapping = tp.ToneMapping.AgX
    renderer.tone_mapping_exposure = 1.0
    renderer.auto_exposure = False
    scene = tp.Scene()
    scene.background = tp.Color(*MURK)
    scene.set_fog_exp2(tp.Color(*MURK), 0.09)
    scene.add(tp.HemisphereLight(0x9fd8d0, 0x10201c, 0.9))
    sun = tp.DirectionalLight(0xfff2da, 2.6)
    sun.position.set(3.0, 10.0, 4.0)
    scene.add(sun)
    floor = tp.Mesh(tp.PlaneGeometry(64, 64), tp.MeshStandardMaterial())
    floor.material.map = grid_texture()
    floor.material.roughness = 0.95
    floor.rotation.x = -math.pi / 2
    floor.position.set(8.0, -0.45, 0.0)
    scene.add(floor)
    for k, m in enumerate(snake.meshes):
        m.material.color = tp.Color(1.0, 0.58, 0.0) if k < snake.n - 1 else tp.Color(1.0, 0.75, 0.2)
        m.material.roughness = 0.45
        m.material.metalness = 0.0
        scene.add(m)
    camera = tp.PerspectiveCamera(38, W / H, 0.05, 200)

    t_sim = 0.0
    for _ in range(int(round(args.preroll / dt))):
        world.step(dt); t_sim += dt
    cam_c = snake.com()
    cam_h = snake.mean_heading()
    enc = Encoder(os.path.join(args.out, "look.mp4"), W, H, FPS, crf=18)
    nfr = int(round(args.seconds * FPS))
    keep_at = set(int(round(x)) for x in np.linspace(8, nfr - 8, 6))
    strip = []
    switched = False
    for fr in range(nfr):
        tf = fr / FPS
        if not switched and tf >= args.switch:
            snake.set_gait("eel", a, w, d)
            switched = True
        for _ in range(sub):
            world.step(dt); t_sim += dt
        c = snake.com(); h = snake.mean_heading()
        k = 1.0 - math.exp(-(1.0 / FPS) / 0.6)          # a smoothed follow: no undulation judder
        cam_c = cam_c + k * (c - cam_c)
        cam_h = cam_h + k * (h - cam_h)
        fwd = np.array([math.cos(cam_h), 0.0, -math.sin(cam_h)])
        lat = np.cross(fwd, [0.0, 1.0, 0.0])
        eye = cam_c - 1.7 * fwd + 1.6 * lat + np.array([0.0, 1.25, 0.0])
        camera.position.set(*[float(v) for v in eye])
        tgt = cam_c + 0.35 * fwd
        camera.look_at(tp.Vector3(*[float(v) for v in tgt]))
        renderer.sim_time = t_sim
        renderer.render(scene, camera)
        px = np.asarray(renderer.read_pixels())
        if px.ndim == 3 and px.shape[2] == 4:
            px = px[..., :3]
        if px.shape[0] != H:
            px = px.reshape(H, W, -1)[..., :3]
        gs = snake.gait_state()
        if tf < args.switch:
            txt, sub_ = "lateral undulation", "g(i,n) = 1: constant amplitude, head to tail"
        else:
            txt, sub_ = "eel-like motion", "g(i,n) = (n-i)/(n+1): amplitude grows toward the tail"
        info = (f"{sub_}   |   alpha 30 deg, omega 120 deg/s, delta 30 deg   |   "
                f"t = {t_sim:5.1f} s")
        frame = label(np.ascontiguousarray(px, dtype=np.uint8), txt, info)
        enc.send(frame)
        if fr in keep_at:
            strip.append(frame)
    rc = enc.close()
    from PIL import Image
    tiles = [Image.fromarray(f).resize((640, 360), Image.LANCZOS) for f in strip]
    sheet = Image.new("RGB", (640 * 3, 360 * 2))
    for i, tl in enumerate(tiles):
        sheet.paste(tl, ((i % 3) * 640, (i // 3) * 360))
    sheet.save(os.path.join(args.out, "look_strip.png"))
    print(f"look.mp4: {nfr} frames, encoder rc {rc}; look_strip.png: {len(tiles)} frames; "
          f"com travelled {np.linalg.norm(snake.com() - np.zeros(3)):.2f} m in {t_sim:.1f} s")
    snake.remove()


if __name__ == "__main__":
    main()
