"""Find grasp posture with table/cube placed along arm's natural reach axis (world Z).

The KUKA URDF zero pose extends the arm along world Z. The table should be
placed at world Z ≈ 0.8-1.3m (in front of / below the arm).
The arm base is at world origin. The arm natural reach (fully extended) is ~1.3m
along world Z.

We place:
- Table top at Z = 0.8m (horizontal platform in the XY plane at Z=0.8)
- Cube on table at (0, 0, 0.8 + cube_half)
- Arm base at (0, 0, 0) — Z is its reach direction

This means gravity (-Y) pulls the cube down, but the table supports it.
The table is a flat slab: width in X, height in Y, with surface normal pointing +Z.
Wait, no — if the table is horizontal (XZ plane), then it has a Y-normal pointing up,
and the cube falls onto it and rests at the correct Y position.

Actually the cleanest fix: the KUKA arm extends along world Z, so:
  - The arm's "reach" is world Z.
  - Gravity is world -Y.
  - The table is a XZ-plane slab, siting on the ground plane.
  - The cube rests ON the table (Y position stable) but the arm approaches from
    a NEGATIVE Z direction (gripper points in +Z toward the cube).

This means the cube should be at:
  - Y = TABLE_TOP + CUBE_HALF (resting on table)
  - Z = 0.8..1.2 m (in front of arm, within reach)
  - X = 0 (centred)

And the gripper approaches from Z < cube_Z (arm from behind) toward +Z.

Let me find postures where the gripper is at (0, cube_Y, 0.8..1.0) with
approach axis +Z.
"""
import sys, os, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import threepp as tp
import torch
import numpy as np

GRIPPER_URDF = r'C:\dev\threepp\build\_deps\threepp_data-src\urdf\kuka_iiwa_gripper.urdf'
DT = 0.002

torch.zeros(1, device='cuda')
torch.cuda.synchronize()
drv = ctypes.CDLL('nvcuda.dll')
drv.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
drv.cuCtxGetCurrent.restype = ctypes.c_int
ctx_ptr = ctypes.c_void_p()
drv.cuCtxGetCurrent(ctypes.byref(ctx_ptr))
ctx = int(ctx_ptr.value)

TABLE_H = 0.0       # arm base Y = 0, table base Y = 0
TABLE_THICK = 0.05
TABLE_TOP = TABLE_H + TABLE_THICK
CUBE_EDGE = 0.07
CUBE_HALF = CUBE_EDGE / 2
# Cube rests on table: Y = TABLE_TOP + CUBE_HALF = 0.085 m
CUBE_REST_Y = TABLE_TOP + CUBE_HALF
# Cube in front of arm at Z = 1.0
CUBE_Z = 1.0
CUBE_POS = np.array([0.0, CUBE_REST_Y, CUBE_Z])
print(f"Cube target: {CUBE_POS}")

world = tp.PhysxWorld(
    gravity=tp.Vector3(0, -9.81, 0),
    fixed_timestep=DT, max_substeps=1,
    direct_gpu=True, cuda_context=ctx,
)

art, meshes, joint_names = world.load_articulation(
    GRIPPER_URDF, fixed_base=True,
    base_position=(0.0, 0.0, 0.0),
    default_density=1200.0,
    stiffness=500.0, damping=50.0,
    max_force=500.0, render_visuals=False,
)
dof = art.dof_order().shape[0]

arm_batch = tp.PhysxGpuBatch(world, [art])
dev = torch.device('cuda')
max_dofs = arm_batch.max_dofs
perm = torch.from_numpy(art.dof_order().astype(np.int64)).to(dev)
gpu_tgt = torch.zeros(1, max_dofs, device=dev)
arm_lp = torch.zeros(1, arm_batch.max_links * 7, device=dev)

def set_q(q7):
    canon = torch.zeros(1, dof, device=dev)
    for i in range(7): canon[0,i] = q7[i]
    canon[0,7] = 0.020; canon[0,8] = 0.020
    gpu_tgt.zero_()
    gpu_tgt[0, perm] = canon[0]
    arm_batch.write_joint_target_pos(gpu_tgt)

def settle_and_read(q7, n=300):
    for _ in range(n):
        set_q(q7)
        arm_batch.step(DT)
    arm_batch.read_link_pose(arm_lp)
    lp = arm_lp[0].cpu().numpy()
    def get_pos(idx): off = idx*7; return lp[off+4:off+7]
    return {'link7': get_pos(7), 'fl': get_pos(8), 'fr': get_pos(9)}

# Zero pose
r0 = settle_and_read([0,0,0,0,0,0,0], 300)
mid0 = (r0['fl'] + r0['fr'])/2
print(f"Zero pose: link7={r0['link7']}, tip_mid={mid0}")
print(f"  dist to cube: {np.linalg.norm(mid0 - CUBE_POS):.4f}")

# Grid search
best_dist = 1e9; best_q = None; best_r = None
# Focusing on a2 (main shoulder), a4 (elbow) which affect Z position most
# a2 about Y at z=0.36 -> rotates arm in XZ plane
# a4 about -Y at z=0.42 above -> elbow
for a2 in np.linspace(-2.0, 2.0, 9):
    for a4 in np.linspace(-2.0, 2.0, 9):
        for a6 in np.linspace(-2.0, 2.0, 5):
            q7 = [0, float(a2), 0, float(a4), 0, float(a6), 0]
            r = settle_and_read(q7, 200)
            mid = (r['fl'] + r['fr']) / 2
            dist = np.linalg.norm(mid - CUBE_POS)
            if dist < best_dist:
                best_dist = dist
                best_q = q7
                best_r = r

print(f"\nBest: dist={best_dist:.4f}")
print(f"  q7={[f'{x:.3f}' for x in best_q]}")
if best_r:
    mid = (best_r['fl'] + best_r['fr'])/2
    print(f"  tip midpoint: {mid}")
    print(f"  link_7: {best_r['link7']}")
    gapx = abs(best_r['fl'][0] - best_r['fr'][0])
    gapy = abs(best_r['fl'][1] - best_r['fr'][1])
    gapz = abs(best_r['fl'][2] - best_r['fr'][2])
    print(f"  finger gaps: X={gapx:.4f}  Y={gapy:.4f}  Z={gapz:.4f}")

# Also try with a1 rotation to see if we can reach better
print("\n\nTrying with a1 rotation to reach cube at (0, 0.085, 1.0)...")
if best_dist > 0.1:
    for a1 in np.linspace(-0.5, 0.5, 5):
        q7 = [float(a1)] + best_q[1:]
        r = settle_and_read(q7, 200)
        mid = (r['fl'] + r['fr']) / 2
        dist = np.linalg.norm(mid - CUBE_POS)
        print(f"  a1={a1:.2f}: dist={dist:.4f}, tip={mid}")
