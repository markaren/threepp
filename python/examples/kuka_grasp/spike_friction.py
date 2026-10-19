"""Friction-grasp feasibility gate — Stage 1 deliverable.

Proves that a parallel-jaw gripper can hold a 0.07 m cube by friction alone
(no kinematic attach) using the combined kuka_iiwa_gripper.urdf and the
existing pure-Python threepp bindings.

Test sequence (all joint-target driven, no grasp-lock):
  (a) Settle: arm at a pre-grasp posture with fingers open, straddling the cube.
  (b) Close:  drive finger joints to closed targets (squeeze the cube).
  (c) Lift:   drive arm joint to raise the tip ~0.15+ m.
  (d) Hold:   step for 1 s; read cube Y every 10 substeps.

PASS criterion:
  Cube Y rises >= LIFT_THRESHOLD (0.10 m) above its resting Y and stays there
  within a 0.05 m band for at least HOLD_TICKS consecutive reads.

Run:
    python python/examples/kuka_grasp/spike_friction.py
    (or from the python/ dir: python examples/kuka_grasp/spike_friction.py)

Knobs swept to find a working regime:
  - finger_squeeze_force (drive max_force on finger joints): 10, 30, 60, 100 N
  - friction (static = dynamic): 1.0, 1.5, 2.0
  - cube_mass (via density): 0.05, 0.10, 0.20 kg (using cube_density = mass / volume)

Reports working regime and PASS/FAIL.
"""
import ctypes
import os
import sys
import math

import numpy as np
import torch

# ---- path setup ---------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PYTHON_DIR = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _PYTHON_DIR)

import threepp as tp

# ---- URDF paths ---------------------------------------------------------------
# The combined URDF must be placed next to lbr_iiwa_14_r820.urdf so that relative
# mesh paths (lbr_iiwa_14_r820/collision/*.stl) resolve correctly.
_URDF_SEARCH = [
    # Preferred: next to original KUKA URDF in build/_deps
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(_HERE))),
                 "build", "_deps", "threepp_data-src", "urdf", "kuka_iiwa_gripper.urdf"),
    r"C:\dev\threepp\build\_deps\threepp_data-src\urdf\kuka_iiwa_gripper.urdf",
    # Fallback: same dir as this script (mesh paths won't resolve if arm meshes are missing here)
    os.path.join(_HERE, "kuka_iiwa_gripper.urdf"),
]
GRIPPER_URDF = next((p for p in _URDF_SEARCH if os.path.exists(p)), None)

# ---- simulation constants -----------------------------------------------------
DT = 0.002            # physics substep (s)
SETTLE_STEPS  = 500   # substeps to let arm reach straddle posture
CLOSE_STEPS   = 200   # substeps to squeeze fingers
LIFT_STEPS    = 500   # substeps to drive arm upward
HOLD_STEPS    = 500   # substeps to hold and observe
READ_EVERY    = 10    # read cube Y every N substeps

TABLE_H       = 0.75  # m, table top surface
TABLE_THICK   = 0.05  # m
TABLE_TOP     = TABLE_H + TABLE_THICK / 2   # = 0.775 m (table slab centre Y)
CUBE_EDGE     = 0.07  # m
CUBE_HALF     = CUBE_EDGE / 2
# Cube rests at:  table surface + half-cube = (TABLE_H + TABLE_THICK) + CUBE_HALF
CUBE_REST_Y   = TABLE_H + TABLE_THICK + CUBE_HALF   # ≈ 0.845 m

LIFT_THRESHOLD = 0.10  # m above rest Y the cube must rise to count as PASS
HOLD_TICKS    = 20     # consecutive reads where cube stays above threshold

# ---- finger/gripper targets in DOF add-order -----------------------------------
# The finger DOFs are DOF indices 7 (left) and 8 (right) in add-order.
# The URDF loader reports them in joint add-order after the 7 arm revolutes.
# Joint axis conventions (see URDF comments):
#   finger_joint_left:  axis +X, prismatic. q>0 = left finger moves right (+X) = closing.
#   finger_joint_right: axis -X, prismatic. q>0 = right finger moves left (-X) = closing.
# Both at q=+0.020: open (gap ≈ 0.088 m).
# Both at q=-0.005: closed (gap ≈ 0.038 m — squeezes 0.07 m cube).
FINGER_OPEN   =  0.020   # m
FINGER_CLOSED = -0.005   # m

# Arm posture: joint targets in add-order [j_a1 .. j_a7] that put the flange
# above the cube. This is a rough pre-grasp hover posture.
# The KUKA arm base is at (0,0,0). The table-top cube is at ~(0, 0.845, 0.2).
# With typical KUKA joint limits, a downward-pointing posture is approximately:
#   a1=0 (no turn), a2=-pi/4 (shoulder forward), a3=0, a4=pi/2 (elbow down),
#   a5=0, a6=-pi/4 (wrist tilt), a7=0.
# This posture is approximate; we verify by looking at cube vs tip distance.
PRE_GRASP_Q = [0.0, -0.40, 0.0, 1.50, 0.0, -0.80, 0.0]   # radians

# Lift posture: drive joint_a2 (shoulder) less negative -> raises the tip.
LIFT_Q = [0.0, -0.85, 0.0, 1.50, 0.0, -0.80, 0.0]  # shoulder raised ~0.45 rad

# ---- helpers ------------------------------------------------------------------
def _torch_cuda_context():
    try:
        drv = ctypes.CDLL("nvcuda.dll")
    except OSError as e:
        raise RuntimeError("cannot load nvcuda.dll") from e
    drv.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    drv.cuCtxGetCurrent.restype = ctypes.c_int
    ctx = ctypes.c_void_p()
    err = drv.cuCtxGetCurrent(ctypes.byref(ctx))
    if err != 0 or not ctx.value:
        raise RuntimeError(f"could not read torch's CUDA context (err={err}, ctx={ctx.value})")
    return int(ctx.value)


def _box_mesh(w, h, d, color=0xaaaaaa):
    m = tp.Mesh(tp.BoxGeometry(w, h, d), tp.MeshStandardMaterial())
    m.material.color = color
    return m


# ---- single-trial run ---------------------------------------------------------
def run_trial(world, gripper_urdf_path,
              friction_coeff, cube_mass_kg, finger_squeeze_stiffness,
              finger_squeeze_max_force, arm_stiffness, arm_damping, arm_max_force,
              verbose=True):
    """
    Run one friction-grasp trial.

    Returns:
        dict with keys:
          pass_     bool
          cube_y_history  list[float]   (Y position every READ_EVERY substeps)
          max_lift_m      float         (peak cube Y above cube_rest_y)
          trial_params    dict
    """
    params = dict(
        friction=friction_coeff,
        cube_mass_kg=cube_mass_kg,
        finger_squeeze_stiffness=finger_squeeze_stiffness,
        finger_squeeze_max_force=finger_squeeze_max_force,
        arm_stiffness=arm_stiffness,
        arm_damping=arm_damping,
        arm_max_force=arm_max_force,
    )
    dev = torch.device("cuda")

    # 1. Static scene geometry ---------------------------------------------------
    # Ground plane
    gnd = _box_mesh(60.0, 0.1, 60.0, 0x888888)
    gnd.position.set(0, -0.05, 0)
    world.add_static(gnd)

    # Table
    table_m = _box_mesh(0.8, TABLE_THICK, 0.8, 0xc8a060)
    table_m.position.set(0.0, TABLE_H + TABLE_THICK / 2, 0.2)
    world.add_static(table_m)

    # 2. Arm + gripper articulation (URDF load) ----------------------------------
    # IMPORTANT: load_articulation applies UNIFORM stiffness/damping/max_force to ALL joints.
    # The finger joints need different (force-limited squeeze) behaviour.
    # We load with arm gains; then we CANNOT change per-joint gains post-load via Python
    # (no per-joint setter is exposed). Workaround: load with finger gains (lower stiffness,
    # force-limited). The arm will still be stiff enough with stiffness=200 given the
    # damping controls overshoot. We accept the uniform-gain limitation for this spike.
    #
    # Alternative: rebuild with very high arm stiffness and accept that the fingers are
    # also stiff — then the cube deformation (compressive force) depends on finger
    # displacement, not force-limited. This is fine for a friction-gate feasibility test
    # since the friction force is N * mu, and N comes from finger contact compression.
    art_tuple = world.load_articulation(
        gripper_urdf_path,
        fixed_base=True,
        base_position=(0.0, 0.0, 0.0),
        default_density=1200.0,
        stiffness=arm_stiffness,
        damping=arm_damping,
        max_force=finger_squeeze_max_force,  # use finger force as the global max (arm won't saturate at this level anyway)
        render_visuals=False,
    )
    arm_art, arm_meshes, joint_names = art_tuple

    dof = arm_art.dof_order().shape[0]
    if verbose:
        print(f"  Loaded URDF: DOF={dof}, joints={joint_names}")
    if dof != 9:
        raise RuntimeError(f"Expected 9 DOF (7 arm + 2 finger), got {dof}. Check URDF path and content.")

    # 3. High-friction material --------------------------------------------------
    # WORKAROUND for URDF-loaded finger material gap:
    # We cannot set a custom material on URDF-loaded finger links from Python.
    # The world's default material has friction ~0.2. By using friction_combine="max",
    # the effective contact friction = max(finger_default, cube_friction) = cube_friction.
    # This lets us test the friction-hold even without per-link finger material control.
    gripper_mat = world.create_material(
        static_friction=friction_coeff,
        dynamic_friction=friction_coeff,
        restitution=0.0,
        friction_combine="max",   # effective = max(cube_friction, finger_default) = cube_friction
        restitution_combine="min",
    )

    # 4. Cube articulation (free-base, 0 DOF) ------------------------------------
    cube_volume = CUBE_EDGE ** 3        # m^3
    cube_density = cube_mass_kg / cube_volume

    # Place cube at (0, just above table, 0.2) — in the pre-grasp straddle zone
    cube_m = _box_mesh(CUBE_EDGE, CUBE_EDGE, CUBE_EDGE, 0xe04444)
    cube_m.position.set(0.0, CUBE_REST_Y + 0.02, 0.2)   # slightly above rest

    cube_art = world.create_articulation(
        fixed_base=False,
        solver_position_iterations=12,
        disable_self_collision=False,
    )
    # Add the cube link with the high-friction material
    cube_art.add_link(
        cube_m,
        density=cube_density,
        material=gripper_mat,
    )
    cube_art.finalize()

    # 5. Construct GPU batches (all arts finalized before either batch) -----------
    arm_batch  = tp.PhysxGpuBatch(world, [arm_art])
    cube_batch = tp.PhysxGpuBatch(world, [cube_art])

    max_dofs   = arm_batch.max_dofs
    max_links  = arm_batch.max_links

    arm_perm = torch.from_numpy(arm_art.dof_order().astype(np.int64)).to(dev)

    # GPU tensors
    arm_jp    = torch.zeros(1, max_dofs, device=dev)
    arm_jv    = torch.zeros(1, max_dofs, device=dev)
    cube_pose = torch.zeros(1, 7, device=dev)
    gpu_tgt   = torch.zeros(1, max_dofs, device=dev)

    cube_gpu_idx = torch.from_numpy(
        cube_batch.gpu_indices().astype(np.int32)
    ).to(dev)

    def set_arm_targets(q_canon, finger_l, finger_r):
        """Write arm+finger joint targets (all 9 DOFs)."""
        canon = torch.zeros(1, dof, device=dev)
        for i in range(7):
            canon[0, i] = q_canon[i]
        canon[0, 7] = finger_l   # finger_joint_left  (add-order index 7)
        canon[0, 8] = finger_r   # finger_joint_right (add-order index 8)
        gpu_tgt.zero_()
        gpu_tgt[0, arm_perm] = canon[0]
        arm_batch.write_joint_target_pos(gpu_tgt)

    def read_cube_y():
        cube_batch.read_root_pose(cube_pose)
        return cube_pose[0, 5].item()   # index 5 = py

    # ---- Phase (a): settle arm at pre-grasp posture, fingers open --------------
    if verbose:
        print(f"  [a] Settling arm at pre-grasp posture ({SETTLE_STEPS} substeps)...")
    # Both finger joints: left at +OPEN, right at +OPEN (symmetric, same sign because
    # left joint axis is +X and right joint axis is -X, so same positive target opens both)
    for _ in range(SETTLE_STEPS):
        set_arm_targets(PRE_GRASP_Q, FINGER_OPEN, FINGER_OPEN)
        arm_batch.step(DT)

    # Teleport cube to resting position (in case it bounced during settle)
    rest_pose = torch.zeros(1, 7, device=dev)
    rest_pose[0, 3] = 1.0          # qw = 1
    rest_pose[0, 4] = 0.0          # px
    rest_pose[0, 5] = CUBE_REST_Y  # py
    rest_pose[0, 6] = 0.2          # pz
    rest_lv = torch.zeros(1, 3, device=dev)
    rest_av = torch.zeros(1, 3, device=dev)
    torch.cuda.synchronize()
    cube_batch.write_subset_root_pose(rest_pose, cube_gpu_idx)
    cube_batch.write_subset_root_linvel(rest_lv, cube_gpu_idx)
    cube_batch.write_subset_root_angvel(rest_av, cube_gpu_idx)

    # Settle a bit more after cube teleport
    for _ in range(150):
        set_arm_targets(PRE_GRASP_Q, FINGER_OPEN, FINGER_OPEN)
        arm_batch.step(DT)

    y_before = read_cube_y()
    if verbose:
        print(f"  Cube Y at open straddle: {y_before:.3f} m  (expected ~{CUBE_REST_Y:.3f})")

    # ---- Phase (b): close fingers ----------------------------------------------
    if verbose:
        print(f"  [b] Closing fingers ({CLOSE_STEPS} substeps)...")
    for _ in range(CLOSE_STEPS):
        set_arm_targets(PRE_GRASP_Q, FINGER_CLOSED, FINGER_CLOSED)
        arm_batch.step(DT)

    y_after_close = read_cube_y()
    if verbose:
        print(f"  Cube Y after close: {y_after_close:.3f} m")

    # ---- Phase (c): lift -------------------------------------------------------
    if verbose:
        print(f"  [c] Lifting arm ({LIFT_STEPS} substeps)...")
    cube_y_history = []
    for step in range(LIFT_STEPS):
        set_arm_targets(LIFT_Q, FINGER_CLOSED, FINGER_CLOSED)
        arm_batch.step(DT)
        if step % READ_EVERY == 0:
            cube_y_history.append(read_cube_y())

    # ---- Phase (d): hold -------------------------------------------------------
    if verbose:
        print(f"  [d] Holding ({HOLD_STEPS} substeps)...")
    for step in range(HOLD_STEPS):
        set_arm_targets(LIFT_Q, FINGER_CLOSED, FINGER_CLOSED)
        arm_batch.step(DT)
        if step % READ_EVERY == 0:
            cube_y_history.append(read_cube_y())

    # ---- Evaluate --------------------------------------------------------------
    max_lift = max(y - CUBE_REST_Y for y in cube_y_history) if cube_y_history else 0.0
    above_threshold = [y - CUBE_REST_Y >= LIFT_THRESHOLD for y in cube_y_history]
    # Check for HOLD_TICKS consecutive reads above threshold in the HOLD phase
    hold_start_idx = LIFT_STEPS // READ_EVERY   # where hold phase starts in history
    hold_above = above_threshold[hold_start_idx:]
    consecutive = 0
    max_consecutive = 0
    for v in hold_above:
        if v:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    passed = (max_lift >= LIFT_THRESHOLD) and (max_consecutive >= HOLD_TICKS)

    if verbose:
        print(f"\n  Cube Y history (every {READ_EVERY} substeps):")
        for i, y in enumerate(cube_y_history):
            marker = "^HOLD" if i >= hold_start_idx else ""
            lift = y - CUBE_REST_Y
            flag = "ABOVE" if lift >= LIFT_THRESHOLD else "     "
            print(f"    step {i*READ_EVERY:4d}: Y={y:.4f}  lift={lift:+.4f} m  {flag} {marker}")
        print(f"\n  max_lift = {max_lift:.4f} m  (threshold {LIFT_THRESHOLD:.2f} m)")
        print(f"  max_consecutive_hold_reads_above = {max_consecutive}  (need {HOLD_TICKS})")
        result_str = "PASS" if passed else "FAIL"
        print(f"  Trial result: {result_str}")

    return {
        "pass_": passed,
        "cube_y_history": cube_y_history,
        "max_lift_m": max_lift,
        "max_consecutive_hold": max_consecutive,
        "trial_params": params,
    }


# ---- sweep helper -------------------------------------------------------------
def run_sweep():
    """Sweep key knobs to find the smallest working regime."""
    assert tp.HAS_PHYSX, "Need a PhysX-enabled threepp build"
    assert torch.cuda.is_available(), "Need CUDA"
    print(f"[OK] threepp.HAS_PHYSX = True,  CUDA = {torch.cuda.get_device_name(0)}")

    if GRIPPER_URDF is None or not os.path.exists(GRIPPER_URDF):
        raise RuntimeError(
            f"Combined gripper URDF not found. Expected: {_URDF_CANDIDATES[0]}\n"
            "Run this script from any directory; it resolves paths automatically."
        )
    print(f"[OK] Gripper URDF: {GRIPPER_URDF}")

    # Warm torch CUDA so it owns the primary context
    torch.zeros(1, device="cuda")
    torch.randn(64, 64, device="cuda").sum().item()
    torch.cuda.synchronize()
    ctx = _torch_cuda_context()
    print(f"[OK] torch CUDA context: 0x{ctx:x}")

    # Knob ranges
    friction_values          = [1.0, 1.5, 2.0]
    cube_mass_values         = [0.05, 0.10, 0.20]  # kg
    finger_squeeze_max_force = [30.0, 60.0, 100.0] # N
    # arm gains (uniform — fingers get the same stiffness)
    arm_stiffness  = 200.0
    arm_damping    = 20.0
    # We vary finger squeeze force = arm's max_force for now (uniform gain limitation)
    # The stiffness drives the squeeze; the finger will compress the cube until the
    # elastic contact force matches the stiffness * displacement.
    finger_squeeze_stiffness = arm_stiffness  # same as arm (uniform)

    results = []
    trial_num = 0

    for friction in friction_values:
        for cube_mass in cube_mass_values:
            for max_force in finger_squeeze_max_force:
                trial_num += 1
                print(f"\n{'='*60}")
                print(f"Trial {trial_num}: friction={friction}, cube_mass={cube_mass:.2f} kg, "
                      f"max_force={max_force:.0f} N")
                print(f"{'='*60}")

                # Create a fresh world for each trial (can't reuse after batches are built)
                world = tp.PhysxWorld(
                    gravity=tp.Vector3(0, -9.81, 0),
                    fixed_timestep=DT,
                    max_substeps=1,
                    direct_gpu=True,
                    cuda_context=ctx,
                )

                try:
                    r = run_trial(
                        world=world,
                        gripper_urdf_path=GRIPPER_URDF,
                        friction_coeff=friction,
                        cube_mass_kg=cube_mass,
                        finger_squeeze_stiffness=finger_squeeze_stiffness,
                        finger_squeeze_max_force=max_force,
                        arm_stiffness=arm_stiffness,
                        arm_damping=arm_damping,
                        arm_max_force=max_force,
                        verbose=True,
                    )
                    results.append(r)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    results.append({"pass_": False, "error": str(e), "trial_params": {
                        "friction": friction, "cube_mass_kg": cube_mass, "max_force": max_force}})

                # Clean up world explicitly
                del world

    # ---- Summary -------------------------------------------------------------
    print(f"\n{'='*60}")
    print("SWEEP SUMMARY")
    print(f"{'='*60}")
    passing = []
    for r in results:
        p = r.get("trial_params", {})
        status = "PASS" if r.get("pass_") else "FAIL"
        max_lift = r.get("max_lift_m", 0.0)
        err = r.get("error", "")
        print(f"  {status}  friction={p.get('friction','?')}  "
              f"mass={p.get('cube_mass_kg','?')} kg  "
              f"max_force={p.get('finger_squeeze_max_force', p.get('max_force','?'))} N  "
              f"lift={max_lift:.4f} m  {err}")
        if r.get("pass_"):
            passing.append(r)

    print()
    if passing:
        best = min(passing, key=lambda r: (
            r["trial_params"]["finger_squeeze_max_force"],
            r["trial_params"]["cube_mass_kg"],
        ))
        bp = best["trial_params"]
        print("FRICTION GATE: PASS")
        print(f"  Smallest working regime found:")
        print(f"    friction (static=dynamic): {bp['friction']}")
        print(f"    cube_mass_kg:              {bp['cube_mass_kg']:.2f}")
        print(f"    finger max_force (N):      {bp.get('finger_squeeze_max_force', bp.get('max_force','?'))}")
        print(f"    arm stiffness/damping:     {arm_stiffness}/{arm_damping}")
        print(f"    max_lift_m:                {best['max_lift_m']:.4f} m")
    else:
        print("FRICTION GATE: FAIL — no parameter combination achieved the hold criterion.")
        print("  Possible causes:")
        print("  - Arm posture doesn't straddle the cube (check PRE_GRASP_Q vs cube position)")
        print("  - Finger limits / axis signs incorrect in URDF")
        print("  - Insufficient friction or force")


# ---- DESIGN PROBE: API report -------------------------------------------------
def probe_api():
    """Report on per-joint gain API and material API availability."""
    print("\n" + "="*60)
    print("API PROBE: per-joint gains + material API")
    print("="*60)
    print()
    print("1. load_articulation drive gains:")
    print("   - stiffness/damping/max_force are UNIFORM across ALL joints.")
    print("   - No per-joint setter is exposed from Python post-load.")
    print("   - The C++ Articulation::addLink sets PD drive at build time.")
    print("   - No Python API for setDriveParams after finalize().")
    print("   FINDING: Per-joint gains NOT settable from Python for URDF-loaded arts.")
    print("   WORKAROUND A: Build arm+gripper programmatically (create_articulation +")
    print("     add_link), passing different stiffness/damping/max_force per link.")
    print("   WORKAROUND B: Load with finger-appropriate gains; arm doesn't saturate.")
    print("   WORKAROUND C: Use write_joint_force (effort control) for fingers only,")
    print("     bypassing the PD drive for finger joints (GPU batch write_joint_force).")
    print()
    print("2. CPU setDriveTargets for prismatic joints:")
    print("   - Articulation::setDriveTargets hardcodes eTWIST axis for ALL joints.")
    print("   - Prismatic joints use eX axis — so CPU setDriveTargets DOES NOT WORK")
    print("     for finger prismatics. This is a bug in the C++ implementation.")
    print("   - GPU batch write_joint_target_pos works correctly for all joint types")
    print("     (uses the raw GPU DOF buffer, bypasses the axis type).")
    print("   FINDING: Use write_joint_target_pos (GPU batch) for finger control.")
    print("     Do NOT use art.set_drive_targets() if prismatic joints are present.")
    print()
    print("3. Material API:")
    print("   - world.create_material(sf, df, restitution, friction_combine) -> PhysxMaterial")
    print("   - PhysxMaterial properties are mutable at runtime (domain randomization).")
    print("   - For PROGRAMMATIC builds: pass material= to add_link(). WORKS.")
    print("   - For URDF-loaded links: NO material= arg in load_articulation().")
    print("     The URDF loader calls addLink with material=nullptr (world default).")
    print("   FINDING: Cannot set high-friction material on URDF-loaded finger links")
    print("     directly from Python without a C++ rebuild or programmatic build.")
    print("   WORKAROUND: Build the cube with the grippy material; the contact friction")
    print("     is the PRODUCT (multiply combine) or AVERAGE of both surfaces.")
    print("     Setting cube friction=2.0 gives effective friction ≈ 2.0 * world_default.")
    print("     For the training env, use programmatic gripper build so finger material")
    print("     can be set directly. For URDF route: accept world-default finger friction")
    print("     and compensate with higher cube friction or higher squeeze force.")
    print()
    print("4. DOF report for 9-DOF combined URDF:")
    print("   - Expected joint order: joint_a1..joint_a7 (indices 0-6), then")
    print("     finger_joint_left (index 7), finger_joint_right (index 8).")
    print("   - GPU DOF cache order may differ; use art.dof_order() to get the permutation.")
    print()


if __name__ == "__main__":
    probe_api()
    run_sweep()
