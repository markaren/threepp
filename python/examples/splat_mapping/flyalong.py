"""A fly-along film: the trained map beside the truth, along a camera path the site runner recorded.

The site runner writes RUN/flyalong/ (truth_NNNNN.jpg at half size and poses.json: the camera's matrix_world per
frame, its vertical FOV, near and far, the frame rate). This renders the .ply from the same poses in threepp's
splat pass (node turned upright, no tone curve), --frames frames a pose, and writes truth | map side by side, with
a label, as an mp4 (ffmpeg from imageio-ffmpeg or FFMPEG).

    py -3.14 python/examples/splat_mapping/flyalong.py --dir RUN/flyalong --splats RUN/seed0/splats.ply --out film.mp4
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # python/ (threepp)
import threepp as tp  # noqa: E402

import dataset as ds  # noqa: E402
from manifest import write_manifest  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--splats", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frames", type=int, default=4, help="frames a pose is held (the path is continuous)")
ap.add_argument("--label", default="truth (deferred camera output)  |  splat map (threepp splat pass)")
args = ap.parse_args()

with open(os.path.join(args.dir, "poses.json"), encoding="utf-8") as f:
    P = json.load(f)
W, H = P["w"], P["h"]
canvas = tp.Canvas("splat-flyalong", width=W, height=H, headless=True, vsync=False)
renderer = tp.VulkanRenderer(canvas)
renderer.dlss = False
renderer.fsr = False
renderer.render_scale = 1.0
renderer.auto_exposure = False
renderer.tone_mapping = tp.ToneMapping.NoToneMapping
renderer.tone_mapping_exposure = 1.0
renderer.bloom_intensity = 0.0
renderer.sharpen_strength = 0.0
camera = tp.PerspectiveCamera(P["vfov"], W / H, P["near"], P["far"])
cloud = tp.SplatCloud(tp.SplatLoader.load_ply(args.splats))
cloud.rotation.x = math.pi
scene = tp.Scene()
scene.background = tp.Color(0.0, 0.0, 0.0)
scene.add(cloud)


def find_ffmpeg():
    if os.environ.get("FFMPEG"):
        return os.environ["FFMPEG"]
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return shutil.which("ffmpeg")


ff = find_ffmpeg()
if not ff:
    raise SystemExit("no ffmpeg: pip install imageio-ffmpeg, or set FFMPEG")
w2, h2 = W // 2, H // 2
proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * w2}x{h2 + 20}",
                         "-framerate", str(P["fps"]), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                         "-preset", "medium", args.out + ".tmp.mp4"], stdin=subprocess.PIPE)
t0 = time.perf_counter()
for i, fr in enumerate(P["frames"]):
    pos, tgt, up = ds.pose_of(fr)
    ds.place(camera, pos, tgt, up)
    for _ in range(args.frames):
        renderer.render(scene, camera)
    img = Image.fromarray(renderer.read_pixels()).resize((w2, h2), Image.LANCZOS)
    truth = Image.open(os.path.join(args.dir, f"truth_{i:05d}.jpg")).convert("RGB")
    panel = Image.new("RGB", (2 * w2, h2 + 20), (0, 0, 0))
    panel.paste(truth, (0, 20))
    panel.paste(img, (w2, 20))
    ImageDraw.Draw(panel).text((6, 4), f"{args.label}    sim t {fr['t']:6.2f} s", fill=(255, 255, 255))
    proc.stdin.write(np.asarray(panel).tobytes())
proc.stdin.close()
proc.wait()
os.replace(args.out + ".tmp.mp4", args.out)
write_manifest(os.path.dirname(os.path.abspath(args.out)), name="flyalong",
               extra={"frames": len(P["frames"]), "fps": P["fps"], "seconds": round(time.perf_counter() - t0, 1)})
print(f"wrote {args.out}: {len(P['frames'])} frames at {P['fps']} fps in {time.perf_counter() - t0:.0f} s")
