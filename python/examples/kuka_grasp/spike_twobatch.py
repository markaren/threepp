"""Two-batch coexistence spike for the KUKA grasp experiment.

Proves that TWO PhysxGpuBatch objects can coexist over ONE PhysxWorld scene:
  - arm_batch  : K fixed-base arms (KUKA URDF loaded via world.load_articulation)
  - cube_batch : K free-base single-link cube articulations

Assertions verified:
  1. Cube starts above the table, falls under gravity, comes to rest on table top
     (root Y stabilizes approx TABLE_TOP + CUBE/2 = 0.810 m).
  2. Writing arm drive targets visibly moves the arm joints.
  3. Both batches are readable every step with no CUDA errors.
  4. write_subset_root_pose teleports a cube (root-reset test).

Also determines whether world.load_articulation works for the KUKA URDF (7 DOF?).

Run from the python/ dir or anywhere -- the script self-inserts the right sys.path.

Usage:
    python python/examples/kuka_grasp/spike_twobatch.py
"""
import ctypes
import os
import sys

import numpy as np
import torch

# ---- path setup (same pattern as cartpole_env / spot) -----------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PYTHON_DIR)

import threepp as tp

# ---- constants --------------------------------------------------------------
K = 16                  # number of envs in each batch
DT = 0.002              # physics substep (s) -- 0.002 x 10 = 0.02 s per control step
SUBSTEPS = 10           # substeps per control step
SETTLE_STEPS = 300      # number of substeps to settle the scene before assertions
ASSERT_STEPS = 200      # number of substeps over which we check cube settling
DRIVE_STEPS = 100       # substeps over which we check arm joint movement

TABLE_H = 0.75          # table top height (m)
TABLE_THICK = 0.05
CUBE_HALF = 0.035       # half-edge of 0.07m cube (m)
TABLE_TOP = TABLE_H + TABLE_THICK / 2   # = 0.775 m
CUBE_REST_Y = TABLE_TOP + CUBE_HALF     # expected resting Y of cube CoM approx 0.810 m

_CANDIDATES = [
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))),
                 "build", "_deps", "threepp_data-src", "urdf", "lbr_iiwa_14_r820.urdf"),
    r"C:\dev\threepp\build\_deps\threepp_data-src\urdf\lbr_iiwa_14_r820.urdf",
    r"C:\dev\threepp\cmake-build-relwithdebinfo\_deps\threepp_data-src\urdf\lbr_iiwa_14_r820.urdf",
]
KUKA_URDF = next((p for p in _CANDIDATES if os.path.exists(p)), None)


# ---- CUDA context (mirror sim.py) -------------------------------------------
def _torch_cuda_context():
    """Read torch's current CUDA primary context so PhysX can adopt it."""
    try:
        drv = ctypes.CDLL("nvcuda.dll")
    except OSError as e:
        raise RuntimeError("cannot load nvcuda.dll") from e
    drv.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    drv.cuCtxGetCurrent.restype = ctypes.c_int
    ctx = ctypes.c_void_p()
    err = drv.cuCtxGetCurrent(ctypes.byref(ctx))
    if err != 0 or not ctx.value:
        raise RuntimeError(
            f"could not read torch's CUDA context (err={err}, ctx={ctx.value})")
    return int(ctx.value)


# ---- helpers ----------------------------------------------------------------
def _box_mesh(w, h, d, color=0xaaaaaa):
    m = tp.Mesh(tp.BoxGeometry(w, h, d), tp.MeshStandardMaterial())
    m.material.color = color
    return m


def _build_arm(world, env_idx, spacing=2.5):
    """Build one arm articulation for env env_idx.

    If the KUKA URDF is available, load it via world.load_articulation (7 DOF revolute).
    Otherwise fall back to a simple 3-DOF programmatic arm so the two-batch mechanics
    can still be proven.
    """
    ox = env_idx * spacing

    if KUKA_URDF:
        # ---- KUKA URDF route ------------------------------------------------
        # load_articulation returns (Articulation, list[Mesh], list[str]) and calls
        # finalize() internally.
        art, meshes, joint_names = world.load_articulation(
            KUKA_URDF,
            fixed_base=True,
            base_position=(ox, 0.0, 0.0),   # X spacing per env
            default_density=1200.0,
            stiffness=200.0,
            damping=20.0,
            max_force=300.0,
            render_visuals=False,            # skip visuals for headless spike
        )
        return art, meshes, joint_names

    else:
        # ---- 3-DOF fallback arm (no URDF) -----------------------------------
        print("  [warn] KUKA URDF not found -- using 3-DOF fallback arm")
        art = world.create_articulation(fixed_base=True, solver_position_iterations=8,
                                        disable_self_collision=True)
        # root link (base, pinned)
        base_m = _box_mesh(0.15, 0.15, 0.15, 0x333333)
        base_m.position.set(ox, 0.15, 0.0)
        base = art.add_link(base_m, density=2000.0)

        # link 1 -- revolute about Z at base top
        l1_m = _box_mesh(0.08, 0.3, 0.08, 0x2a6fae)
        l1_m.position.set(ox, 0.45, 0.0)
        l1 = art.add_link(l1_m, parent=base, density=1000.0,
                          axis=(0, 0, 1), anchor=(ox, 0.3, 0.0),
                          lower=-1.5, upper=1.5,
                          stiffness=200.0, damping=20.0, max_force=200.0)
        # link 2 -- revolute about X
        l2_m = _box_mesh(0.06, 0.3, 0.06, 0xf2a93b)
        l2_m.position.set(ox, 0.75, 0.0)
        l2 = art.add_link(l2_m, parent=l1, density=800.0,
                          axis=(1, 0, 0), anchor=(ox, 0.6, 0.0),
                          lower=-1.2, upper=1.2,
                          stiffness=150.0, damping=15.0, max_force=150.0)
        # link 3 -- revolute about Z
        l3_m = _box_mesh(0.05, 0.25, 0.05, 0x29b06a)
        l3_m.position.set(ox, 1.0, 0.0)
        art.add_link(l3_m, parent=l2, density=600.0,
                     axis=(0, 0, 1), anchor=(ox, 0.875, 0.0),
                     lower=-1.5, upper=1.5,
                     stiffness=100.0, damping=10.0, max_force=100.0)
        art.finalize()
        return art, [base_m, l1_m, l2_m, l3_m], ["j1", "j2", "j3"]


def _build_cube(world, env_idx, spacing=2.5):
    """Build one free-base single-link cube articulation.

    free-base single-link -> 0 DOF (root pose only), readable via read_root_pose.
    The cube starts above the table and should fall under gravity and rest on it.
    """
    ox = env_idx * spacing
    cube_start_y = TABLE_TOP + CUBE_HALF + 0.5   # 0.5 m above the table

    art = world.create_articulation(fixed_base=False, solver_position_iterations=8,
                                    disable_self_collision=False)
    cube_m = _box_mesh(CUBE_HALF * 2, CUBE_HALF * 2, CUBE_HALF * 2, 0xe04444)
    cube_m.position.set(ox, cube_start_y, 0.0)
    art.add_link(cube_m, density=300.0)   # light mass, no joints
    art.finalize()
    return art, [cube_m]


# ---- main spike -------------------------------------------------------------
def run_spike():
    print("=" * 60)
    print("Two-batch coexistence spike")
    print("=" * 60)

    # 1. sanity: PhysX available
    assert tp.HAS_PHYSX, "Need a PhysX-enabled threepp build (tp.HAS_PHYSX is False)"
    assert torch.cuda.is_available(), "Need CUDA"
    print(f"[OK] threepp.HAS_PHYSX = True,  CUDA = {torch.cuda.get_device_name(0)}")

    # 2. Warm torch CUDA so it owns the primary context
    torch.zeros(1, device="cuda")
    torch.randn(64, 64, device="cuda").sum().item()
    torch.cuda.synchronize()
    ctx = _torch_cuda_context()
    print(f"[OK] torch CUDA context: 0x{ctx:x}")

    # 3. Build the PhysxWorld (direct_gpu=True, shared context)
    world = tp.PhysxWorld(
        gravity=tp.Vector3(0, -9.81, 0),
        fixed_timestep=DT,
        max_substeps=1,          # step() advances exactly DT each call
        direct_gpu=True,
        cuda_context=ctx,
    )

    # 4. Static ground + table (added BEFORE articulations)
    ground_m = _box_mesh(60.0, 0.1, 60.0, 0x888888)
    ground_m.position.set(0, -0.05, 0)
    world.add_static(ground_m)

    SPACING = 2.5
    for i in range(K):
        table_m = _box_mesh(0.8, TABLE_THICK, 0.8, 0xc8a060)
        table_m.position.set(i * SPACING, TABLE_H + TABLE_THICK / 2, 0.2)
        world.add_static(table_m)

    # 5. Build ALL arm articulations
    print(f"\n[build] Loading {K} arm articulations...")
    if KUKA_URDF:
        print(f"  KUKA URDF: {KUKA_URDF}")
    else:
        print("  KUKA URDF not found -- using 3-DOF fallback")

    arm_arts = []
    arm_joint_names = None
    for i in range(K):
        art, meshes, joint_names = _build_arm(world, i, SPACING)
        arm_arts.append(art)
        if arm_joint_names is None:
            arm_joint_names = joint_names
    arm_dof = arm_arts[0].dof_order().shape[0]
    print(f"  arm DOF: {arm_dof}  joint_names: {arm_joint_names}")

    # 6. Build ALL cube articulations (0 DOF -- root pose only)
    print(f"\n[build] Building {K} cube articulations...")
    cube_arts = []
    for i in range(K):
        art, meshes = _build_cube(world, i, SPACING)
        cube_arts.append(art)
    cube_dof = cube_arts[0].dof_order().shape[0]
    print(f"  cube DOF: {cube_dof}  (expected 0)")

    # 7. Construct TWO batches -- arm first, then cube
    #    DESIGN note: PhysxGpuBatch ctor runs one warmup simulate() to assign GPU indices.
    #    Both batches share the same world, so the warmup step from the first batch already
    #    steps the cubes too -- that is fine, both must be fully finalized before EITHER batch
    #    is constructed (which they are at this point).
    print("\n[batch] Constructing arm_batch...")
    arm_batch = tp.PhysxGpuBatch(world, arm_arts)
    print(f"  arm_batch: count={arm_batch.count}, max_dofs={arm_batch.max_dofs}, "
          f"max_links={arm_batch.max_links}")

    print("[batch] Constructing cube_batch...")
    cube_batch = tp.PhysxGpuBatch(world, cube_arts)
    print(f"  cube_batch: count={cube_batch.count}, max_dofs={cube_batch.max_dofs}, "
          f"max_links={cube_batch.max_links}")

    # NOTE: cube_batch.max_dofs will equal arm max_dofs (7) even though cubes have 0 DOF --
    # max_dofs is the scene-global maxDofs, not the per-batch per-art DOF count.
    # The cube DOF count (0) comes from cube_arts[0].dof_order().shape[0].

    # 8. Allocate GPU tensors
    dev = torch.device("cuda")
    arm_jp  = torch.zeros(K, arm_batch.max_dofs, device=dev)
    arm_jv  = torch.zeros(K, arm_batch.max_dofs, device=dev)
    # cube root pose [qx,qy,qz,qw,px,py,pz]
    cube_pose = torch.zeros(K, 7, device=dev)

    # DOF remapping (arm)
    arm_perm = torch.from_numpy(arm_arts[0].dof_order().astype(np.int64)).to(dev)

    # --- gather GPU indices for cube batch (needed for subset reset) ----------
    cube_gpu_idx = torch.from_numpy(cube_batch.gpu_indices().astype(np.int32)).to(dev)

    # ---- assertion accumulators -----------------------------------------
    passes = []
    fails = []

    def check(name, cond, detail=""):
        if cond:
            passes.append(name)
            print(f"  [PASS] {name}" + (f"  {detail}" if detail else ""))
        else:
            fails.append(name)
            print(f"  [FAIL] {name}" + (f"  {detail}" if detail else ""))

    # =========================================================================
    # TEST 1 -- Both batches readable every step (no CUDA errors)
    # =========================================================================
    print(f"\n[test1] Reading both batches for {SETTLE_STEPS} settle substeps...")
    read_errors = 0
    for step in range(SETTLE_STEPS):
        arm_batch.step(DT)       # steps the whole scene (arms + cubes)
        try:
            arm_batch.read_joint_pos(arm_jp)
            arm_batch.read_joint_vel(arm_jv)
            cube_batch.read_root_pose(cube_pose)
        except Exception as e:
            read_errors += 1
            print(f"  ERROR at step {step}: {e}")
    check("both_batches_readable_every_step",
          read_errors == 0, f"({read_errors} errors in {SETTLE_STEPS} steps)")

    # =========================================================================
    # TEST 2 -- Cube falls under gravity and rests on the table
    # The cube may have already settled during the test1 settle steps.
    # Strategy: teleport ALL cubes to DROP_Y, then watch them fall and rest.
    # =========================================================================
    print(f"\n[test2] Checking cube gravity/rest -- teleport up then let fall...")
    DROP_Y = TABLE_TOP + CUBE_HALF + 0.8   # well above the table
    all_idx = cube_gpu_idx.contiguous()
    drop_poses = torch.zeros(K, 7, device=dev)
    drop_poses[:, 3] = 1.0  # qw=1 (identity quaternion): layout [qx,qy,qz,qw,px,py,pz]
    for i in range(K):
        drop_poses[i, 4] = i * SPACING   # px
        drop_poses[i, 5] = DROP_Y         # py
        drop_poses[i, 6] = 0.2            # pz (table centre)
    drop_linvel = torch.zeros(K, 3, device=dev)
    drop_angvel = torch.zeros(K, 3, device=dev)
    torch.cuda.synchronize()
    cube_batch.write_subset_root_pose(drop_poses, all_idx)
    cube_batch.write_subset_root_linvel(drop_linvel, all_idx)
    cube_batch.write_subset_root_angvel(drop_angvel, all_idx)

    cube_batch.read_root_pose(cube_pose)
    y_initial = cube_pose[:, 5].mean().item()
    print(f"  y_initial after teleport: {y_initial:.3f} m  (expected ~{DROP_Y:.3f})")

    y_history = []
    for step in range(ASSERT_STEPS):
        arm_batch.step(DT)
        cube_batch.read_root_pose(cube_pose)
        y_history.append(cube_pose[:, 5].mean().item())

    y_final = y_history[-1]
    y_settled = abs(y_final - CUBE_REST_Y) < 0.08
    y_fell = y_final < (y_initial - 0.1)   # dropped by at least 0.1 m
    check("cube_fell_under_gravity", y_fell,
          f"y_initial={y_initial:.3f} y_final={y_final:.3f}")
    check("cube_rested_on_table", y_settled,
          f"y_final={y_final:.3f}  expected~{CUBE_REST_Y:.3f} +-0.08 m")

    # =========================================================================
    # TEST 3 -- Writing arm drive targets visibly moves joints
    # =========================================================================
    print(f"\n[test3] Commanding arm joints and checking movement over {DRIVE_STEPS} substeps...")
    arm_batch.read_joint_pos(arm_jp)
    jp_before = arm_jp[:, arm_perm].clone()   # add-order

    target_val = 0.3   # radians
    arm_tgt_canon = torch.zeros(K, arm_dof, device=dev)
    arm_tgt_canon[:, 0] = target_val   # move first add-order joint

    gpu_tgt = torch.zeros(K, arm_batch.max_dofs, device=dev)
    gpu_tgt[:, arm_perm] = arm_tgt_canon
    torch.cuda.synchronize()
    arm_batch.write_joint_target_pos(gpu_tgt)

    for step in range(DRIVE_STEPS):
        arm_batch.step(DT)

    arm_batch.read_joint_pos(arm_jp)
    jp_after = arm_jp[:, arm_perm].clone()
    delta = (jp_after[:, 0] - jp_before[:, 0]).abs().mean().item()
    check("arm_joints_moved_on_drive_target", delta > 0.01,
          f"mean |delta_joint0|={delta:.4f} rad  (expected >0.01)")

    # =========================================================================
    # TEST 4 -- subset root reset teleports a cube
    # =========================================================================
    print("\n[test4] Testing write_subset_root_pose (cube teleport)...")
    TELEPORT_Y = 3.0   # well above current cube position
    sub_idx = cube_gpu_idx[0:1].contiguous()
    teleport_pose = torch.zeros(1, 7, device=dev)
    teleport_pose[0, 3] = 1.0   # qw = 1 (identity quaternion)
    teleport_pose[0, 5] = TELEPORT_Y   # Y position index 5

    torch.cuda.synchronize()
    cube_batch.write_subset_root_pose(teleport_pose, sub_idx)

    cube_batch.read_root_pose(cube_pose)
    y_after_teleport = cube_pose[0, 5].item()
    check("subset_root_reset_teleports_cube", abs(y_after_teleport - TELEPORT_Y) < 0.05,
          f"env0 Y after teleport: {y_after_teleport:.3f}  expected~{TELEPORT_Y:.1f}")

    # =========================================================================
    # ARM ROUTE REPORT
    # =========================================================================
    print("\n[arm_route]")
    if KUKA_URDF:
        print(f"  URDF loaded: {KUKA_URDF}")
        print(f"  DOF: {arm_dof}   joint_names: {arm_joint_names}")
        urdf_7dof = arm_dof == 7
        check("kuka_urdf_gives_7dof", urdf_7dof, f"arm_dof={arm_dof}")
    else:
        print("  KUKA URDF not found -- arm route: programmatic 3-DOF fallback only")
        print("  (re-run after confirming build/_deps/threepp_data-src/urdf/ path)")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 60)
    print(f"RESULTS: {len(passes)} PASS, {len(fails)} FAIL")
    for p in passes:
        print(f"  PASS  {p}")
    for f in fails:
        print(f"  FAIL  {f}")
    print("=" * 60)
    if not fails:
        print("TWO-BATCH SPIKE: PASS")
    else:
        print("TWO-BATCH SPIKE: FAIL -- see above")
    print()


if __name__ == "__main__":
    run_spike()
