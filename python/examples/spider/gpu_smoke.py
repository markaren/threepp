"""M1 smoke test for the native PhysX direct-GPU path (no torch yet).

Spawns K hexapods in ONE direct-GPU scene, steps them on the GPU, and reads the
state back through PhysxGpuBatch's host-staged debug readers. Proves the whole
native pipeline end-to-end:
  - the direct-GPU scene + CUDA context come up,
  - getArticulationData returns finite, evolving, PER-ROBOT-DISTINCT state,
  - stepping integrates dynamics on the GPU.

    python gpu_smoke.py

Needs a CUDA GPU + the PhysX GPU build (this is exactly what the RL path rides on).
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from hexapod import Hexapod

if not tp.HAS_PHYSX:
    print("This build has no PhysX backend."); sys.exit(0)

K = 64
SPACING = 2.5
DT = 1.0 / 60.0

print(f"building {K} hexapods in one direct-GPU scene ...")
world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0), direct_gpu=True)
ground = tp.Mesh(tp.BoxGeometry(SPACING * K + 40, 1, 60), tp.MeshStandardMaterial())
ground.position.set(SPACING * K * 0.5, -0.5, 0.0)
world.add_static(ground)
spiders = [Hexapod(world, position=(i * SPACING, 0.40, 0.0)) for i in range(K)]

batch = tp.PhysxGpuBatch(world, [s.art for s in spiders])
print(f"  batch.count={batch.count}  max_dofs={batch.max_dofs}")
assert batch.count == K
assert batch.max_dofs == 12  # 6 legs x (coxa, femur)

# Root pose layout: [qx, qy, qz, qw, px, py, pz]
pose0 = batch.read_root_pose_host()
assert pose0.shape == (K, 7), pose0.shape
y0 = pose0[:, 5].copy()
px0 = pose0[:, 4].copy()
print(f"  initial chassis y: {y0.min():.3f}..{y0.max():.3f}")

# Robots must be at DISTINCT x positions (~ i*SPACING) — proves the GPU index
# buffer maps each block to a different articulation (not all reading robot 0).
expected_x = np.arange(K) * SPACING
x_err = np.abs(px0 - expected_x).max()
print(f"  max |px - i*spacing| = {x_err:.3f}  (distinct-robot check)")
assert x_err < 0.05, "articulations not distinctly indexed!"

print("stepping 90 frames on the GPU ...")
for _ in range(90):
    batch.step(DT)

pose1 = batch.read_root_pose_host()
jp = batch.read_joint_pos_host()
jv = batch.read_joint_vel_host()
lin = batch.read_root_linvel_host()
y1 = pose1[:, 5]

print(f"  after  chassis y: {y1.min():.3f}..{y1.max():.3f}")
print(f"  joint_pos shape {jp.shape}  joint_vel shape {jv.shape}  linvel shape {lin.shape}")
print(f"  |dy| mean over robots: {np.abs(y1 - y0).mean():.4f}  (dynamics evolved)")

ok = True
ok &= bool(np.isfinite(pose1).all() and np.isfinite(jp).all() and np.isfinite(jv).all())
ok &= jp.shape == (K, 12) and jv.shape == (K, 12) and lin.shape == (K, 3)
ok &= bool((y1 > -1.0).all() and (y1 < 1.0).all())      # settled, didn't explode
ok &= float(np.abs(pose1 - pose0).mean()) > 1e-4         # state actually changed

# x should still be ~ i*spacing (robots don't teleport into each other)
ok &= float(np.abs(pose1[:, 4] - expected_x).max()) < 1.0

print("\nM1 SMOKE:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
