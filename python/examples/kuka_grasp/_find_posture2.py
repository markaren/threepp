"""Find posture for KUKA mounted so its Z-up URDF maps to threepp Y-up world.

The KUKA URDF is Z-up. In threepp Y-up world, we rotate the arm base by -90 deg
about X so URDF-Z -> world-Y. Then the arm reaches upward (world Y) and we can
find a posture where the gripper tip is above the table cube.

The Python load_articulation doesn't expose base_rotation directly.
But we can use the world's Y-up and figure out postures that have the arm
extending in the world Y direction with the table at Y≈0.8m.

Actually: let's try placing the arm at a lower Y origin and with a different
default pose. The KUKA zero pose with all joints at 0 extends along world-Z.
With joint_a2 at +pi/2 (about Y axis), the arm link chain bends toward world-Y.

Let me systematically search postures that put the tip near (0, 0.95, 0.2).
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

# Place arm base 1.5m above ground (mounted on a platform) so arm can reach down
# to a table at Y=0.8m
ARM_BASE_Y = 1.5
CUBE_POS = np.array([0.0, 0.845, 0.2])

world = tp.PhysxWorld(
    gravity=tp.Vector3(0, -9.81, 0),
    fixed_timestep=DT, max_substeps=1,
    direct_gpu=True, cuda_context=ctx,
)

# No rotation option in Python binding, so we rely on the URDF zero pose
# being along world-Z (arm sticks up in Z). Let's instead mount the arm
# elevated high and look for postures that can reach the cube below.
#
# arm base at (0, 1.5, 0) -- high enough to reach down to Y=0.845
art, meshes, joint_names = world.load_articulation(
    GRIPPER_URDF, fixed_base=True,
    base_position=(0.0, ARM_BASE_Y, 0.0),
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

def settle_and_read(q7, n=400):
    for _ in range(n):
        set_q(q7)
        arm_batch.step(DT)
    arm_batch.read_link_pose(arm_lp)
    lp = arm_lp[0].cpu().numpy()
    def get_pos(idx): off = idx*7; return lp[off+4:off+7]
    return {'link7': get_pos(7), 'fl': get_pos(8), 'fr': get_pos(9)}

# Try postures with arm base at Y=1.5
# The arm extends along URDF-Z = world-Z. joint_a2 bends about Y axis (link_1 to link_2).
# After a2=-pi/2 the arm bends "down" in the URDF Z frame.
# Actually let's just print what the zero-pose and various joint combos give us.

print(f"Arm base at Y={ARM_BASE_Y}, cube at {CUBE_POS}")
print(f"{'q7':40s}  {'dist':8s}  {'tip(fl mid)':30s}  {'link7':30s}")

import itertools

best_dist = 1e9
best_q = None

# Zero pose first
r0 = settle_and_read([0,0,0,0,0,0,0], 300)
mid0 = (r0['fl'] + r0['fr'])/2
print(f"zero                                        {np.linalg.norm(mid0-CUBE_POS):8.4f}  "
      f"({mid0[0]:+.3f},{mid0[1]:+.3f},{mid0[2]:+.3f})  "
      f"({r0['link7'][0]:+.3f},{r0['link7'][1]:+.3f},{r0['link7'][2]:+.3f})")

# Grid search: a2 from -2.0 to 2.0, a4 from -2.0 to 2.0, a6 from -2.0 to 2.0
# (these are the main arm-stretch joints)
import itertools as it
best_dist = 1e9
best_q = None
results = []
for a2 in np.linspace(-1.8, 1.8, 7):
    for a4 in np.linspace(-1.8, 1.8, 7):
        for a6 in np.linspace(-1.8, 1.8, 7):
            q7 = [0, float(a2), 0, float(a4), 0, float(a6), 0]
            r = settle_and_read(q7, 200)
            mid = (r['fl'] + r['fr']) / 2
            dist = np.linalg.norm(mid - CUBE_POS)
            # Only keep reasonable (arm not self-colliding badly, dist < 0.5)
            if dist < best_dist:
                best_dist = dist
                best_q = q7
                best_r = r
            if dist < 0.3:
                results.append((dist, q7, mid, r['link7']))

print(f"\nBest: dist={best_dist:.4f}, q7={[f'{x:.2f}' for x in best_q]}")
if best_r:
    mid = (best_r['fl'] + best_r['fr'])/2
    print(f"  tip midpoint: {mid}")
    print(f"  link_7: {best_r['link7']}")
    print(f"  finger_left: {best_r['fl']}")
    print(f"  finger_right: {best_r['fr']}")
    gap = abs(best_r['fl'][0] - best_r['fr'][0])
    gapy = abs(best_r['fl'][1] - best_r['fr'][1])
    gapz = abs(best_r['fl'][2] - best_r['fr'][2])
    print(f"  finger gaps: X={gap:.4f}  Y={gapy:.4f}  Z={gapz:.4f}")

print(f"\nAll within 0.3 m of cube:")
results.sort(key=lambda x: x[0])
for dist, q7, mid, l7 in results[:20]:
    print(f"  {dist:.4f}  q={[f'{x:.2f}' for x in q7]}  tip={mid}  l7={l7}")
