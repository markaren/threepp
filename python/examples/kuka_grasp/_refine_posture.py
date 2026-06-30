"""Refined posture search near best found: [0, 0.5, 0, 1.5, 0, 2.0, 0]"""
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

# World setup: arm at origin, table extending in XZ plane, cube in front
TABLE_H = 0.0
TABLE_THICK = 0.05
TABLE_TOP = TABLE_H + TABLE_THICK
CUBE_EDGE = 0.07
CUBE_HALF = CUBE_EDGE / 2
CUBE_REST_Y = TABLE_TOP + CUBE_HALF   # = 0.085 m (world Y)
CUBE_Z = 0.9   # cube is 0.9m in front of arm base

CUBE_POS = np.array([0.0, CUBE_REST_Y, CUBE_Z])
print(f"Cube: {CUBE_POS}")

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

# Fine grid around [0, 0.5, 0, 1.5, 0, 2.0, 0] but targeting (0, 0.085, 0.9)
best_dist = 1e9; best_q = None; best_r = None
print(f"\nFine grid search near [0, 0.5, 0, 1.5, 0, 2.0, 0]...")
for a2 in np.linspace(0.0, 1.5, 7):
    for a4 in np.linspace(0.5, 2.0, 7):
        for a6 in np.linspace(1.0, 2.0, 5):
            q7 = [0, float(a2), 0, float(a4), 0, float(a6), 0]
            r = settle_and_read(q7, 200)
            mid = (r['fl'] + r['fr']) / 2
            dist = np.linalg.norm(mid - CUBE_POS)
            if dist < best_dist:
                best_dist = dist; best_q = q7; best_r = r

print(f"Best: dist={best_dist:.4f}")
print(f"  q7={[f'{x:.3f}' for x in best_q]}")
mid = (best_r['fl'] + best_r['fr'])/2
print(f"  tip midpoint: ({mid[0]:+.4f}, {mid[1]:+.4f}, {mid[2]:+.4f})")
print(f"  link_7: {best_r['link7']}")
gapx = abs(best_r['fl'][0] - best_r['fr'][0])
gapy = abs(best_r['fl'][1] - best_r['fr'][1])
gapz = abs(best_r['fl'][2] - best_r['fr'][2])
print(f"  finger gaps: X={gapx:.4f}  Y={gapy:.4f}  Z={gapz:.4f}")
print(f"  finger_left Y={best_r['fl'][1]:.4f}  finger_right Y={best_r['fr'][1]:.4f}")

# Ultra-fine around best
print(f"\nUltra-fine around best...")
q_best = best_q
for da2 in np.linspace(-0.3, 0.3, 7):
    for da4 in np.linspace(-0.3, 0.3, 7):
        for da6 in np.linspace(-0.3, 0.3, 7):
            q7 = [0, q_best[1]+da2, 0, q_best[3]+da4, 0, q_best[5]+da6, 0]
            # clip to limits
            lims = [(-2.97,2.97),(0,0),(-2.09,2.09),(0,0),(-2.09,2.09),(0,0),(-2.09,2.09)]
            # just clip a2,a4,a6
            q7[1] = np.clip(q7[1], -2.09, 2.09)
            q7[3] = np.clip(q7[3], -2.09, 2.09)
            q7[5] = np.clip(q7[5], -2.09, 2.09)
            r = settle_and_read(q7, 150)
            mid = (r['fl'] + r['fr']) / 2
            dist = np.linalg.norm(mid - CUBE_POS)
            if dist < best_dist:
                best_dist = dist; best_q = q7; best_r = r

print(f"Best after ultra-fine: dist={best_dist:.4f}")
print(f"  q7={[f'{x:.4f}' for x in best_q]}")
mid = (best_r['fl'] + best_r['fr'])/2
print(f"  tip midpoint: ({mid[0]:+.4f}, {mid[1]:+.4f}, {mid[2]:+.4f})")
gapx = abs(best_r['fl'][0] - best_r['fr'][0])
print(f"  finger X gap: {gapx:.4f}")
print(f"  link_7 pos: {best_r['link7']}")

# Also try various cube Z positions
print(f"\n\nSweeping cube Z target with best q7={best_q}:")
for cube_z in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
    target = np.array([0, CUBE_REST_Y, cube_z])
    dist = np.linalg.norm(mid - target)
    print(f"  cube_Z={cube_z}: dist_to_mid={dist:.4f}")
