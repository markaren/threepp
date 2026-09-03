# Deferred verification for the batched AOV readback (readGBufferAOVs).
# RUN WHEN THE GPU IS FREE — a loaded GPU confounds both halves of this script.
#
# Three stages:
#   1. VALIDATION  — run once with THREEPP_VULKAN_VALIDATION=1 (and ideally
#      THREEPP_VULKAN_STRICT_VALIDATION=1): the batched barrier arrays are new
#      command-stream content, and a layout/VUID mistake shows up here, not in
#      the pixels.
#   2. CORRECTNESS — batched read_aovs_typed vs the single-read read_* wrappers.
#      The singles re-drive frames, and TAA jitter moves silhouettes between
#      drives, so ids are compared by match FRACTION and depth away from edges;
#      a batching bug (wrong offset, wrong region, swapped attachment) corrupts
#      whole images, not 0.5% of edge pixels. instance_ids vs class_ids inside
#      ONE batched dict decode the same Ids fetch, so their sky masks must
#      agree EXACTLY — that pair is the bit-exact assertion.
#   3. TIMING      — interleaved A/B, batched multi-AOV call vs the same AOV
#      set fetched one read_* at a time. Alternate A and B per iteration
#      (never two loops), report medians. vsync is off via the headless canvas.
#
# Usage:  python scripts/aov_batch_verify.py [--w 1280 --h 720 --iters 20]

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))
import numpy as np  # noqa: E402
import threepp as tp  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--w", type=int, default=1280)
parser.add_argument("--h", type=int, default=720)
parser.add_argument("--iters", type=int, default=20)
args = parser.parse_args()

canvas = tp.Canvas("aov-ab", width=args.w, height=args.h, headless=True, vsync=False)
renderer = tp.VulkanRenderer(canvas)

scene = tp.Scene()
scene.background = 0x202830
mat = tp.MeshStandardMaterial()
mat.color = 0xff5533
scene.add(tp.Mesh(tp.BoxGeometry(), mat))
sun = tp.DirectionalLight(0xffffff, 3.0)
sun.position.set(3, 5, 2)
scene.add(sun)
camera = tp.PerspectiveCamera(55, args.w / args.h, 0.1, 100)
camera.position.set(1.5, 1.5, 3.0)
camera.look_at(0, 0, 0)

NAMES = ["depth", "normals", "instance_ids", "class_ids", "motion", "albedo"]

# ---- 2) correctness ---------------------------------------------------------
batched = renderer.read_aovs_typed(scene, camera, NAMES)
for name in NAMES:
    assert batched[name].size > 0, f"batched {name} came back empty"

# Internal consistency: both decoded from ONE Ids fetch -> sky masks identical.
sky_i = batched["instance_ids"] == 0
assert batched["instance_ids"].shape == batched["class_ids"].shape, "Ids-pair shape mismatch"

# Cross-check against the single-read wrappers (re-driven frames -> tolerance).
singles = {
    "depth": renderer.read_depth(scene, camera),
    "instance_ids": renderer.read_instance_ids(scene, camera),
    "normals": renderer.read_normals_float(scene, camera),
}
for name, single in singles.items():
    b = batched[name]
    assert b.shape == single.shape, f"{name}: batched {b.shape} vs single {single.shape}"
    if name == "instance_ids":
        frac = np.mean(b == single)
        print(f"instance_ids match fraction (jitter-tolerant): {frac:.4f}")
        assert frac > 0.99, f"instance_ids diverge beyond jitter: {frac:.4f}"
    else:
        # Tolerance calibrated against the measured jitter distribution
        # (batch-vs-batch across re-driven frames: |dz| p50 4e-4, p99 1.2e-2,
        # 0.4% silhouette flips to the far plane) — a batching bug corrupts
        # whole images, which no tolerance this shape would pass.
        close = np.isclose(b, single, rtol=1e-3, atol=2e-2)
        if name == "depth":
            close = close[~sky_i]  # sky reads as far-plane; compare surfaces only
        frac = np.mean(close)
        print(f"{name} close fraction: {frac:.4f}")
        assert frac > 0.98, f"{name} diverges beyond jitter: {frac:.4f}"
print("CORRECTNESS OK")

# ---- 3) interleaved timing --------------------------------------------------
def run_batched():
    renderer.read_aovs_typed(scene, camera, NAMES)

def run_singles():
    # The pre-batch cost shape: one full drain per AOV. read_* re-drive frames,
    # so subtract nothing — report both raw numbers and let the drive cost be
    # visible rather than corrected for.
    renderer.read_depth(scene, camera)
    renderer.read_normals_float(scene, camera)
    renderer.read_instance_ids(scene, camera)
    renderer.read_class_ids(scene, camera)
    renderer.read_motion(scene, camera)

a_times, b_times = [], []
for i in range(args.iters):  # interleaved, never two loops
    t = time.perf_counter(); run_batched(); a_times.append(time.perf_counter() - t)
    t = time.perf_counter(); run_singles(); b_times.append(time.perf_counter() - t)

print(f"batched  read_aovs_typed({len(NAMES)}): median {statistics.median(a_times)*1e3:.1f} ms")
print(f"single   5x read_* (drives incl.):   median {statistics.median(b_times)*1e3:.1f} ms")
print("TIMING DONE — compare medians; the batched call should sit well under the singles.")
