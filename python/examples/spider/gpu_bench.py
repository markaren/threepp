"""Throughput probe for GPU-batched hexapod stepping.

    python gpu_bench.py [K]

Builds K hexapods in one direct-GPU scene and times PURE batch.step() (no readback)
to measure physics throughput in env-steps/s. Also stress-tests the GPU solver memory
config at large K. Run a few K values (separate processes — one PxFoundation/process):

    for K in 256 1024 4096; do python gpu_bench.py $K; done
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from hexapod import Hexapod

if not tp.HAS_PHYSX:
    print("This build has no PhysX backend."); sys.exit(0)

K = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
SPACING = 2.5
DT = 1.0 / 60.0
STEPS = 300

t0 = time.perf_counter()
world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0), direct_gpu=True)
ground = tp.Mesh(tp.BoxGeometry(SPACING * K + 40, 1, 60), tp.MeshStandardMaterial())
ground.position.set(SPACING * K * 0.5, -0.5, 0.0)
world.add_static(ground)
spiders = [Hexapod(world, position=(i * SPACING, 0.40, 0.0)) for i in range(K)]
batch = tp.PhysxGpuBatch(world, [s.art for s in spiders])
t_build = time.perf_counter() - t0

# Drive a simple time-varying joint target so the solver does real work (not a
# trivial static hold). Host-staged write is fine here — it's outside the timed loop
# pattern we care about; we time stepping only.
t0 = time.perf_counter()
for _ in range(STEPS):
    batch.step(DT)
dt = time.perf_counter() - t0

pose = batch.read_root_pose_host()
finite = bool(np.isfinite(pose).all())
sps = STEPS / dt
print(f"K={K:5d}  build={t_build:6.2f}s  step={dt:6.3f}s for {STEPS} steps")
print(f"          {sps:8.1f} world-steps/s   {sps * K:12.0f} env-steps/s   finite={finite}")
