"""Single-trial friction gate runner — run with: python _run_trial.py"""
import sys, os, ctypes, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
import threepp as tp
import torch
import numpy as np

GRIPPER_URDF = r'C:\dev\threepp\build\_deps\threepp_data-src\urdf\kuka_iiwa_gripper.urdf'
DT = 0.002
TABLE_H = 0.75
TABLE_THICK = 0.05
CUBE_EDGE = 0.07
CUBE_HALF = CUBE_EDGE / 2
CUBE_REST_Y = TABLE_H + TABLE_THICK + CUBE_HALF   # 0.845
LIFT_THRESHOLD = 0.10
HOLD_TICKS = 20
READ_EVERY = 10
FINGER_OPEN   =  0.020
FINGER_CLOSED = -0.005
PRE_GRASP_Q = [0.0, -0.40, 0.0, 1.50, 0.0, -0.80, 0.0]
LIFT_Q = [0.0, -0.85, 0.0, 1.50, 0.0, -0.80, 0.0]

torch.zeros(1, device='cuda')
torch.cuda.synchronize()
drv = ctypes.CDLL('nvcuda.dll')
drv.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
drv.cuCtxGetCurrent.restype = ctypes.c_int
ctx_ptr = ctypes.c_void_p()
drv.cuCtxGetCurrent(ctypes.byref(ctx_ptr))
ctx = int(ctx_ptr.value)
print(f'CUDA ctx: 0x{ctx:x}')

def box_mesh(w, h, d, c=0xaaaaaa):
    m = tp.Mesh(tp.BoxGeometry(w, h, d), tp.MeshStandardMaterial())
    m.material.color = c
    return m

friction = 2.0
cube_mass_kg = 0.10
max_force = 60.0
arm_stiffness = 200.0
arm_damping = 20.0
print(f'Trial: friction={friction}, cube_mass={cube_mass_kg}kg, max_force={max_force}N')

world = tp.PhysxWorld(
    gravity=tp.Vector3(0, -9.81, 0),
    fixed_timestep=DT, max_substeps=1,
    direct_gpu=True, cuda_context=ctx,
)

gnd = box_mesh(60, 0.1, 60, 0x888888)
gnd.position.set(0, -0.05, 0)
world.add_static(gnd)
table_m = box_mesh(0.8, TABLE_THICK, 0.8, 0xc8a060)
table_m.position.set(0.0, TABLE_H + TABLE_THICK/2, 0.2)
world.add_static(table_m)

art, meshes, joint_names = world.load_articulation(
    GRIPPER_URDF, fixed_base=True,
    base_position=(0.0, 0.0, 0.0),
    default_density=1200.0,
    stiffness=arm_stiffness, damping=arm_damping,
    max_force=max_force, render_visuals=False,
)
dof = art.dof_order().shape[0]
print(f'DOF={dof}, joints={joint_names}')

gripper_mat = world.create_material(
    static_friction=friction, dynamic_friction=friction,
    restitution=0.0, friction_combine='max', restitution_combine='min',
)

cube_volume = CUBE_EDGE**3
cube_density = cube_mass_kg / cube_volume
cube_m = box_mesh(CUBE_EDGE, CUBE_EDGE, CUBE_EDGE, 0xe04444)
cube_m.position.set(0.0, CUBE_REST_Y + 0.02, 0.2)
cube_art = world.create_articulation(fixed_base=False, solver_position_iterations=12)
cube_art.add_link(cube_m, density=cube_density, material=gripper_mat)
cube_art.finalize()

arm_batch  = tp.PhysxGpuBatch(world, [art])
cube_batch = tp.PhysxGpuBatch(world, [cube_art])

dev = torch.device('cuda')
max_dofs = arm_batch.max_dofs
perm = torch.from_numpy(art.dof_order().astype(np.int64)).to(dev)
cube_pose = torch.zeros(1, 7, device=dev)
gpu_tgt = torch.zeros(1, max_dofs, device=dev)
cube_gpu_idx = torch.from_numpy(cube_batch.gpu_indices().astype(np.int32)).to(dev)
arm_lp = torch.zeros(1, arm_batch.max_links * 7, device=dev)

def set_targets(q_arm, fl, fr):
    canon = torch.zeros(1, dof, device=dev)
    for i in range(7): canon[0,i] = q_arm[i]
    canon[0,7] = fl
    canon[0,8] = fr
    gpu_tgt.zero_()
    gpu_tgt[0, perm] = canon[0]
    arm_batch.write_joint_target_pos(gpu_tgt)

def read_cube_y():
    cube_batch.read_root_pose(cube_pose)
    return cube_pose[0,5].item()

print('Phase (a): settling 500 substeps...')
for _ in range(500):
    set_targets(PRE_GRASP_Q, FINGER_OPEN, FINGER_OPEN)
    arm_batch.step(DT)

arm_batch.read_link_pose(arm_lp)
link7_offset = 7 * 7
link7_pos = arm_lp[0, link7_offset+4:link7_offset+7].cpu().numpy()
print(f'link_7 world pos: x={link7_pos[0]:.3f}, y={link7_pos[1]:.3f}, z={link7_pos[2]:.3f}')
fl_offset = 8 * 7
fl_pos = arm_lp[0, fl_offset+4:fl_offset+7].cpu().numpy()
print(f'finger_left world pos: x={fl_pos[0]:.3f}, y={fl_pos[1]:.3f}, z={fl_pos[2]:.3f}')
fr_offset = 9 * 7
fr_pos = arm_lp[0, fr_offset+4:fr_offset+7].cpu().numpy()
print(f'finger_right world pos: x={fr_pos[0]:.3f}, y={fr_pos[1]:.3f}, z={fr_pos[2]:.3f}')
print(f'finger gap X: {abs(fl_pos[0]-fr_pos[0]):.4f} m')
print(f'finger Z (tip): {fl_pos[2]:.4f} m  cube Z: 0.2000 m')

rest_pose = torch.zeros(1, 7, device=dev)
rest_pose[0,3] = 1.0; rest_pose[0,4] = 0.0; rest_pose[0,5] = CUBE_REST_Y; rest_pose[0,6] = 0.2
cube_batch.write_subset_root_pose(rest_pose, cube_gpu_idx)
cube_batch.write_subset_root_linvel(torch.zeros(1,3,device=dev), cube_gpu_idx)
cube_batch.write_subset_root_angvel(torch.zeros(1,3,device=dev), cube_gpu_idx)

for _ in range(150):
    set_targets(PRE_GRASP_Q, FINGER_OPEN, FINGER_OPEN)
    arm_batch.step(DT)

y_open = read_cube_y()
print(f'Cube Y at open straddle: {y_open:.4f} m (expected ~{CUBE_REST_Y:.3f})')

# CHECK: is the arm near the cube?
arm_batch.read_link_pose(arm_lp)
fl_pos2 = arm_lp[0, fl_offset+4:fl_offset+7].cpu().numpy()
fr_pos2 = arm_lp[0, fr_offset+4:fr_offset+7].cpu().numpy()
print(f'finger Y (after settle): left={fl_pos2[1]:.3f}, right={fr_pos2[1]:.3f}, cube_Y={y_open:.3f}')
print(f'finger Z: left={fl_pos2[2]:.3f}, right={fr_pos2[2]:.3f}')

print('Phase (b): closing fingers 200 substeps...')
for _ in range(200):
    set_targets(PRE_GRASP_Q, FINGER_CLOSED, FINGER_CLOSED)
    arm_batch.step(DT)

y_closed = read_cube_y()
arm_batch.read_link_pose(arm_lp)
fl_pos_c = arm_lp[0, fl_offset+4:fl_offset+7].cpu().numpy()
fr_pos_c = arm_lp[0, fr_offset+4:fr_offset+7].cpu().numpy()
print(f'Cube Y after close: {y_closed:.4f} m')
print(f'finger gap after close: {abs(fl_pos_c[0]-fr_pos_c[0]):.4f} m (should be ~0.038 m)')

print('Phase (c)+(d): lift 500 + hold 500 substeps...')
cube_y_hist = []
for step in range(1000):
    set_targets(LIFT_Q, FINGER_CLOSED, FINGER_CLOSED)
    arm_batch.step(DT)
    if step % READ_EVERY == 0:
        cube_y_hist.append(read_cube_y())

y_arr = np.array(cube_y_hist)
max_lift = (y_arr - CUBE_REST_Y).max()
hold_vals = y_arr[50:]
above = hold_vals - CUBE_REST_Y >= LIFT_THRESHOLD
consec = 0; max_consec = 0
for v in above:
    if v: consec += 1; max_consec = max(max_consec, consec)
    else: consec = 0

print(f'\nCube Y history (every {READ_EVERY} substeps):')
for i, y in enumerate(cube_y_hist):
    ph = 'LIFT' if i < 50 else 'HOLD'
    lift = y - CUBE_REST_Y
    flag = '<ABOVE>' if lift >= LIFT_THRESHOLD else ''
    print(f'  {ph} step {i*READ_EVERY:4d}: Y={y:.4f}  lift={lift:+.4f} m  {flag}')

passed = max_lift >= LIFT_THRESHOLD and max_consec >= HOLD_TICKS
print(f'\nmax_lift={max_lift:.4f} m  (threshold={LIFT_THRESHOLD})')
print(f'max_consec_above={max_consec}/{HOLD_TICKS}')
print(f'GATE RESULT: {"PASS" if passed else "FAIL"}')
