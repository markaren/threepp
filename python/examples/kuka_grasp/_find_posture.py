"""Find a pre-grasp posture for the KUKA arm.

Runs the arm at various joint configs and reports link_7 + finger positions
to find a posture where the gripper straddles the cube at (0, 0.845, 0.2).

The KUKA arm base is at world (0,0,0).
The cube is at ~(0, 0.845, 0.2).
We want the gripper tip (fingers) to be at ~(0, 0.85, 0.2) with fingers
straddling in X and approaching from above (link_7 slightly above cube).
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
print(f'DOF={dof}, joints={joint_names}')

arm_batch = tp.PhysxGpuBatch(world, [art])
dev = torch.device('cuda')
max_dofs = arm_batch.max_dofs
perm = torch.from_numpy(art.dof_order().astype(np.int64)).to(dev)
gpu_tgt = torch.zeros(1, max_dofs, device=dev)
arm_lp = torch.zeros(1, arm_batch.max_links * 7, device=dev)

CUBE_POS = np.array([0.0, 0.845, 0.2])

def set_q(q7):
    """Set arm joints to q7 (7 values), fingers open."""
    canon = torch.zeros(1, dof, device=dev)
    for i in range(7): canon[0,i] = q7[i]
    canon[0,7] = 0.020   # finger open
    canon[0,8] = 0.020
    gpu_tgt.zero_()
    gpu_tgt[0, perm] = canon[0]
    arm_batch.write_joint_target_pos(gpu_tgt)

def settle_and_read(q7, n=400):
    """Settle arm to q7 and read link poses."""
    for _ in range(n):
        set_q(q7)
        arm_batch.step(DT)
    arm_batch.read_link_pose(arm_lp)
    lp = arm_lp[0].cpu().numpy()
    # Each link: 7 floats [qx,qy,qz,qw,px,py,pz]
    def get_pos(idx):
        off = idx * 7
        return lp[off+4:off+7]  # px,py,pz
    return {
        'link7': get_pos(7),
        'finger_left': get_pos(8),
        'finger_right': get_pos(9),
    }

def score(q7):
    """Score = distance from gripper midpoint to cube."""
    r = settle_and_read(q7, 300)
    mid = (r['finger_left'] + r['finger_right']) / 2
    d = np.linalg.norm(mid - CUBE_POS)
    return d, r

# Try candidate postures
# KUKA in its "zero" pose extends straight up. We need to fold it.
# joint_a2 (Y-axis from link_1, at z=0.36) is the main shoulder pitch.
# joint_a4 (-Y-axis from link_3, at z=0.42 above link_1) is the elbow.
# joint_a6 (Y-axis from link_5, at z=0.4 above link_3) is the wrist.
#
# For a downward-pointing posture reaching forward:
#   a2 = +pi/2 (shoulder pitch forward = fold arm toward +Z of base)
#   a4 = -pi/2 (elbow up)
#   a6 = ... (wrist pitch)
# But the KUKA is typically mounted at floor level with Z-up.
# In threepp the world Y is up. The URDF was originally designed for Z-up ROS.
# The URDF loader places the arm with "Z-up" (arm extends in the URDF's Z direction).
# In threepp Y-up world, the URDF Z becomes world Z (not Y).
# So the arm extends in world +Z by default (joint_a1 is about world Z -> world Y in threepp??)
#
# Actually let's just try many postures and see where link_7 ends up.

candidates = [
    # Name, [a1, a2, a3, a4, a5, a6, a7]
    ("zero", [0, 0, 0, 0, 0, 0, 0]),
    ("c1",   [0, 1.0, 0, -1.5, 0, 0, 0]),
    ("c2",   [0, 1.5, 0, -1.0, 0, 0.5, 0]),
    ("c3",   [0, 1.5, 0, -1.5, 0, 0.5, 0]),
    ("c4",   [0, 1.5, 0, -1.0, 0, 1.0, 0]),
    ("c5",   [0, 1.5, 0, -1.5, 0, 1.0, 0]),
    ("c6",   [0, 1.5, 0, -1.0, 0, 1.5, 0]),
    ("c7",   [0, 1.5, 0, -1.5, 0, 1.5, 0]),
    ("c8",   [0, 1.5, 0, -1.0, 0, 2.0, 0]),
    ("c9",   [0, 1.0, 0, -1.0, 0, 1.5, 0]),
    ("c10",  [0, 1.0, 0, -1.5, 0, 1.5, 0]),
    ("c11",  [0, 1.2, 0, -1.0, 0, 1.0, 0]),
    ("c12",  [0, 1.2, 0, -1.2, 0, 1.0, 0]),
    ("c13",  [0, 1.0, 0, -1.2, 0, 1.2, 0]),
    ("c14",  [0, 0.8, 0, -0.8, 0, 1.0, 0]),
    ("c15",  [0, 1.0, 0, -1.0, 0, 1.0, 0]),
]

print(f"\nCube target position: {CUBE_POS}")
print(f"{'Name':8s}  {'dist':8s}  {'link7':30s}  {'finger_left':30s}  {'gap_X':8s}")
best_dist = 1e9
best_name = None
best_q = None

for name, q7 in candidates:
    r = settle_and_read(q7, 300)
    mid = (r['finger_left'] + r['finger_right']) / 2
    dist = np.linalg.norm(mid - CUBE_POS)
    gap_x = abs(r['finger_left'][0] - r['finger_right'][0])
    l7 = r['link7']
    fl = r['finger_left']
    print(f"{name:8s}  {dist:8.4f}  "
          f"({l7[0]:+.3f},{l7[1]:+.3f},{l7[2]:+.3f})  "
          f"({fl[0]:+.3f},{fl[1]:+.3f},{fl[2]:+.3f})  "
          f"{gap_x:.4f}")
    if dist < best_dist:
        best_dist = dist
        best_name = name
        best_q = q7

print(f"\nBest candidate: {best_name} (dist={best_dist:.4f}), q7={best_q}")
print("\nZero-pose link_7 position (to understand URDF frame):")
r0 = settle_and_read([0,0,0,0,0,0,0], 300)
print(f"  link_7: {r0['link7']}")
print(f"  finger_left: {r0['finger_left']}")
print(f"  finger_right: {r0['finger_right']}")
