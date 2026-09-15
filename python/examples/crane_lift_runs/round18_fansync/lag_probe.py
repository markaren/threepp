"""Does a scan_lidar issued BEFORE a frame's render see that frame's scene build, or the previous one?

A box is moved a known STEP along +z every frame (a 0.5 m step, so no noise or float question
can blur the answer). Every frame the same downward beam table is scanned twice, once before
`render()` and once after it. For each scan the hit distance is compared with the CPU's exact
box distance at THIS frame and at the PREVIOUS frame.

    python lag_probe.py            (headless; needs a ray-tracing GPU)

Prints, per frame: pre-render scan distance, post-render scan distance, CPU distance now and
one frame ago, and a verdict. Exit code 0 when every pre-render scan matched the PREVIOUS frame
and every post-render scan matched the CURRENT frame (the one-frame lag of crane_lift.py's
round-12 fan), 2 when both matched the current frame (no lag), 1 otherwise.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))       # <threepp>/python
import threepp as tp  # noqa: E402

STEP = 0.5          # m per frame, along +z
Z0 = -8.0           # first frame's box centre z
HALF = 0.5          # box half-size
N = 12              # frames
TOL = 1e-3          # m

canvas = tp.Canvas("lag_probe", 320, 240, vsync=False, headless=True)
renderer = tp.VulkanRenderer(canvas)
scene = tp.Scene()
scene.background = tp.Color(0x304050)
sun = tp.DirectionalLight(tp.Color(0xFFFFFF), 3.0)
sun.position.set(20, 30, 15)
scene.add(sun)
camera = tp.PerspectiveCamera(60, 320 / 240, 0.1, 100)
camera.position.set(0, 3, 9)
camera.look_at(0, 1, 0)

mat = tp.MeshStandardMaterial()
mat.color = tp.Color(0xC8783C)
box = tp.Mesh(tp.BoxGeometry(2 * HALF, 2 * HALF, 2 * HALF), mat)
scene.add(box)
renderer.set_instance_id(box, 4040)
ground_mat = tp.MeshStandardMaterial()
ground_mat.color = tp.Color(0x556B45)
ground = tp.Mesh(tp.BoxGeometry(60, 0.5, 60), ground_mat)
ground.position.y = -0.25
scene.add(ground)
renderer.set_instance_id(ground, 1)

# One beam straight down from 10 m above the box's TRACK, at z = 0: the box passes under it.
# The beam hits the box top (distance 10 - 2*HALF... = 9.5 m) only when |z_box| < HALF, so use a
# beam that ALWAYS hits the box instead: a horizontal beam along +z from far behind the box.
origin = np.array([[0.0, 0.5, -30.0]], np.float32)
direction = np.array([[0.0, 0.0, 1.0]], np.float32)
params = tp.LidarParams()
params.max_range = 120.0
params.detector_threshold = 0.0


def cpu_distance(z_box):
    """Exact distance from the beam origin to the box's -z face."""
    return float((z_box - HALF) - origin[0, 2])


def scan():
    r = renderer.scan_lidar(origin, direction, params)
    hit = r["return_no"] > 0
    if not hit.any():
        return None, None
    return float(r["distance"][0]), int(r["instance_id"][0])


renderer.set_flush_frames(1)
box.position.set(0.0, 0.5, Z0)
box.update_matrix_world(True)
renderer.sim_time = 0.0
renderer.render(scene, camera)           # frame 0: the build the first pre-render scan would see
z_prev = Z0

rows = []
verdicts = []
for f in range(1, N + 1):
    z_now = Z0 + f * STEP
    box.position.set(0.0, 0.5, z_now)
    box.update_matrix_world(True)
    renderer.sim_time = f / 60.0
    pre, pre_id = scan()                 # BEFORE this frame's render: crane_lift.py round-12 ordering
    renderer.render(scene, camera)
    post, post_id = scan()               # AFTER it
    d_now, d_prev = cpu_distance(z_now), cpu_distance(z_prev)

    def which(d):
        if d is None:
            return "miss"
        if abs(d - d_now) < TOL:
            return "now"
        if abs(d - d_prev) < TOL:
            return "prev"
        return f"neither ({d:.4f})"

    v_pre, v_post = which(pre), which(post)
    verdicts.append((v_pre, v_post))
    rows.append((f, pre, post, d_now, d_prev))
    print(f"frame {f:2d}: pre-render scan {pre if pre is None else f'{pre:.4f}':>8} (id {pre_id}) = {v_pre:5s} | "
          f"post-render scan {post if post is None else f'{post:.4f}':>8} (id {post_id}) = {v_post:5s} | "
          f"CPU now {d_now:.4f}, prev {d_prev:.4f}")
    z_prev = z_now

lag = all(v == ("prev", "now") for v in verdicts)
sync = all(v == ("now", "now") for v in verdicts)
print()
if lag:
    print("VERDICT: every pre-render scan returned the PREVIOUS frame's build; every post-render scan the current one. "
          "A scan issued before render() is one frame behind.")
    sys.exit(0)
if sync:
    print("VERDICT: both scans returned the current frame's build; no lag.")
    sys.exit(2)
print("VERDICT: mixed:", verdicts)
sys.exit(1)
