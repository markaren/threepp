# Stage 1 Spike Findings — KUKA Grasp Two-Batch Architecture

**Date:** 2026-06-30  
**Script:** `python/examples/kuka_grasp/spike_twobatch.py`  
**Verdict:** TWO-BATCH SPIKE: **PASS** (6/6 assertions)

---

## Headline: Two batches coexist over one scene — confirmed

Two `PhysxGpuBatch` objects over one `PhysxWorld(direct_gpu=True)` work correctly.
The arm batch and cube batch independently read/write their respective articulations
every step with no CUDA errors, no cross-contamination, and no ordering constraints
beyond "finalize all arts before constructing any batch."

---

## Real output (verbatim)

```
============================================================
Two-batch coexistence spike
============================================================
[OK] threepp.HAS_PHYSX = True,  CUDA = NVIDIA GeForce RTX 4060 Laptop GPU
[OK] torch CUDA context: 0x287f9624440

[build] Loading 16 arm articulations...
  KUKA URDF: C:\dev\threepp\build\_deps\threepp_data-src\urdf\lbr_iiwa_14_r820.urdf
  arm DOF: 7  joint_names: ['joint_a1', 'joint_a2', 'joint_a3', 'joint_a4', 'joint_a5', 'joint_a6', 'joint_a7']

[build] Building 16 cube articulations...
  cube DOF: 0  (expected 0)

[batch] Constructing arm_batch...
  arm_batch: count=16, max_dofs=7, max_links=8
[batch] Constructing cube_batch...
  cube_batch: count=16, max_dofs=7, max_links=8

[test1] Reading both batches for 300 settle substeps...
  [PASS] both_batches_readable_every_step  (0 errors in 300 steps)

[test2] Checking cube gravity/rest -- teleport up then let fall...
  y_initial after teleport: 1.610 m  (expected ~1.610)
  [PASS] cube_fell_under_gravity  y_initial=1.610 y_final=0.838
  [PASS] cube_rested_on_table  y_final=0.838  expected~0.810 +-0.08 m

[test3] Commanding arm joints and checking movement over 100 substeps...
  [PASS] arm_joints_moved_on_drive_target  mean |delta_joint0|=0.0730 rad  (expected >0.01)

[test4] Testing write_subset_root_pose (cube teleport)...
  [PASS] subset_root_reset_teleports_cube  env0 Y after teleport: 3.000  expected~3.0

[arm_route]
  URDF loaded: C:\dev\threepp\build\_deps\threepp_data-src\urdf\lbr_iiwa_14_r820.urdf
  DOF: 7   joint_names: ['joint_a1', 'joint_a2', 'joint_a3', 'joint_a4', 'joint_a5', 'joint_a6', 'joint_a7']
  [PASS] kuka_urdf_gives_7dof  arm_dof=7

============================================================
RESULTS: 6 PASS, 0 FAIL
  PASS  both_batches_readable_every_step
  PASS  cube_fell_under_gravity
  PASS  cube_rested_on_table
  PASS  arm_joints_moved_on_drive_target
  PASS  subset_root_reset_teleports_cube
  PASS  kuka_urdf_gives_7dof
============================================================
TWO-BATCH SPIKE: PASS
```

---

## Findings by topic

### 1. Two-batch coexistence

**Works.** Key mechanics confirmed:

- Build and `finalize()` ALL articulations (arms + cubes) **before** constructing
  either `PhysxGpuBatch`. The batch constructor runs one warmup `simulateRaw()` to
  assign GPU indices; if any art is not yet finalized it won't have a GPU index.
- Construct `arm_batch` first, then `cube_batch`. Order does not matter logically
  (both call the same `simulateRaw`), but constructing arm first is cleaner.
- Step the scene via **`arm_batch.step(dt)`** only — this calls `world.simulateRaw(dt)`
  which advances the entire scene (all arts). The cube batch is **never stepped**,
  only read.
- `cube_batch.read_root_pose(tensor)` works every step with no errors.
- `arm_batch.read_joint_pos / read_joint_vel` work every step with no errors.

### 2. Homogeneity surprise: cube_batch.max_dofs == 7, not 0

The `max_dofs` reported by a batch is the **scene-global** `maxDofs` from
`PxDirectGPUAPI::getArticulationGPUAPIMaxCounts()`, not the per-batch DOF count.
So `cube_batch.max_dofs` returns 7 (same as the arms), even though each cube has 0
DOF. This is expected PhysX behaviour.

**Impact:** When sizing read/write buffers for the cube batch, use `cube_batch.max_dofs`
as the stride (even though the cubes have no joints). The `read_root_pose` tensor is
`[K, 7]` regardless — that is unaffected.

The actual per-art DOF count for cubes (0) is obtained via
`cube_arts[0].dof_order().shape[0]`, not from the batch's `max_dofs`.

### 3. Arm-build route: URDF load — confirmed working

**Route: `world.load_articulation(path, fixed_base=True, ...)` — CONFIRMED.**

- Python binding exists at `tp.PhysxWorld.load_articulation(...)`.
- Returns `(Articulation, list[Mesh], list[str])` — the articulation is already
  finalized; no extra `finalize()` call needed.
- The KUKA URDF at `build/_deps/threepp_data-src/urdf/lbr_iiwa_14_r820.urdf`
  loads correctly and produces **7 DOF** with joints named:
  `['joint_a1', 'joint_a2', 'joint_a3', 'joint_a4', 'joint_a5', 'joint_a6', 'joint_a7']`
  in add-order (= drive-target order).
- The loaded articulation is immediately usable in a `PhysxGpuBatch` with no issues.
- There is also a higher-level wrapper at `python/threepp/urdf.py`:
  `from threepp.urdf import load_articulation` which returns a `UrdfArticulation`
  with `.articulation`, `.meshes`, `.joint_names`, `.num_dof`, `.set_targets()`.

**The URDF has no `<inertial>` tags**, so all link masses are approximated as
`default_density * shape_volume`. For training this is fine; the arm will be
stiff-PD-driven anyway.

**The collision geometry in the URDF is `<mesh>` (STL files)** — the loader falls
back to the **bounding box** of each STL as the collision shape. This gives rough
box colliders, which is sufficient for RL training but not a digital twin.

### 4. Gripper approach: append links after URDF load — NOT POSSIBLE

`load_articulation` calls `art->finalize()` internally before returning. The Python
`Articulation` object returned is already finalized — `add_link()` cannot be called
on it.

**Therefore the gripper must be added via a combined URDF** (DESIGN.md preferred
route) or a two-articulation approach. The recommended path is:

**Option A (preferred): Combined URDF** — create `kuka_iiwa_gripper.urdf` extending
`lbr_iiwa_14_r820.urdf` with two prismatic finger links on `link_7`. Load the
combined URDF in one call. Gives 7 arm DOF + 2 finger DOF = 9 DOF total per arm art,
all in one batch.

**Option B (fallback): Load KUKA URDF, add gripper programmatically** — this is NOT
possible via `load_articulation` (finalized). Instead, build the entire arm+gripper
manually with `create_articulation` + `add_link` (same approach as `build_spot`),
using the KUKA kinematic parameters from the URDF. More work but full control.

The 6-DOF-appendage cube fallback (DESIGN.md fallback) is NOT needed — two-batch
works cleanly.

### 5. Cube physics

- Free-base single-link articulation with `fixed_base=False` and one `add_link(mesh)`
  (no joints) gives a floating rigid body. DOF count = 0.
- Cube falls correctly under gravity and rests on the table (Y final = 0.838 m vs
  expected 0.810 m — within 0.03 m, the slight overshoot is penetration settling).
- `read_root_pose` returns `[K, 7]` in PhysX layout `[qx, qy, qz, qw, px, py, pz]`.
  The **Y position is index 5** (positions start at index 4).
- `write_subset_root_pose` teleports a cube immediately (readback shows exact target Y).

### 6. Scene stepping

Only one `step()` call per substep is needed — `arm_batch.step(dt)` calls
`world.simulateRaw(dt)` which advances all articulations in the scene (arms + cubes).
Do NOT call `cube_batch.step(dt)` — it would double-step the physics. The cube batch
is read-only during the training loop.

### 7. DOF permutation (add-order vs GPU-cache-order)

Same as `sim.py`: `art.dof_order()` returns the permutation mapping add-order indices
to PhysX GPU cache slot indices. Use it to reorder write targets before writing:
```python
perm = torch.from_numpy(arm_arts[0].dof_order().astype(np.int64)).to(dev)
gpu_tgt = torch.zeros(K, arm_batch.max_dofs, device=dev)
gpu_tgt[:, perm] = canon_tgt   # canon_tgt is in add-order
arm_batch.write_joint_target_pos(gpu_tgt)
```

### 8. Verified API (exact method signatures)

```python
# World construction
world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0),
                      fixed_timestep=0.002, max_substeps=1,
                      direct_gpu=True, cuda_context=int_ctx)

# URDF arm load (returns finalized art)
art, meshes, joint_names = world.load_articulation(
    path_str, fixed_base=True, base_position=(ox, 0.0, 0.0),
    default_density=1200.0, stiffness=200.0, damping=20.0,
    max_force=300.0, render_visuals=False)

# Free-base single-link cube
art = world.create_articulation(fixed_base=False, ...)
art.add_link(mesh, density=300.0)   # no parent, no axis, no limits
art.finalize()

# Batch construction (all arts finalized before either batch)
arm_batch = tp.PhysxGpuBatch(world, [art1, art2, ...])   # arm arts
cube_batch = tp.PhysxGpuBatch(world, [cart1, cube2, ...]) # cube arts

# Batch properties
arm_batch.count        # K
arm_batch.max_dofs     # scene-global maxDofs (7 for KUKA scene, even for cube batch)
arm_batch.max_links    # scene-global maxLinks (8 for KUKA)

# Stepping (one call steps the whole scene)
arm_batch.step(dt)

# Reading (zero-copy CUDA tensor path)
arm_batch.read_joint_pos(tensor_Kxmax_dofs)
arm_batch.read_joint_vel(tensor_Kxmax_dofs)
cube_batch.read_root_pose(tensor_Kx7)   # [qx,qy,qz,qw,px,py,pz]
cube_batch.read_root_linvel(tensor_Kx3)
cube_batch.read_root_angvel(tensor_Kx3)

# Writing arm targets
arm_batch.write_joint_target_pos(tensor_Kxmax_dofs)

# Subset reset (done envs only)
cube_batch.write_subset_root_pose(pose_nbx7, gpu_idx_int32_nb)
cube_batch.write_subset_root_linvel(linvel_nbx3, gpu_idx_int32_nb)
cube_batch.write_subset_root_angvel(angvel_nbx3, gpu_idx_int32_nb)

# GPU indices (for subset reset index buffers)
idx = torch.from_numpy(cube_batch.gpu_indices().astype(np.int32)).to(dev)
```

---

## Deviations from DESIGN.md

| Item | DESIGN.md assumption | Reality |
|------|----------------------|---------|
| Arm build route | Combined URDF (preferred) or append-links fallback | `load_articulation` works; append-links NOT possible (art is finalized on return). Combined URDF is the path. |
| cube_batch.max_dofs | Not specified | Equals the scene-global maxDofs (7), not 0. Use `art.dof_order().shape[0]` for actual DOF count. |
| Two-batch viability | Preferred; fallback to 6-DOF appendage | TWO-BATCH CONFIRMED. Fallback not needed. |
| KUKA URDF location | "search for it" | `build/_deps/threepp_data-src/urdf/lbr_iiwa_14_r820.urdf` |
| KUKA collision | Not specified | STL meshes -> bounding box. Sufficient for RL. |

---

## Next step (Stage 2)

- Create `kuka_iiwa_gripper.urdf` (KUKA + 2 prismatic finger links on `link_7`), giving
  9-DOF arm batch (7 arm + 2 finger). Verify it loads via `load_articulation`.
- Implement `kuka_grasp_contract.py` with the single-source-of-truth constants.
- Extend `GpuSim` (or write a standalone `KukaGraspSim`) with `prop_batch` for cubes.
- Write the env, reward, and selftests.

---

## Friction-grasp gate — PASS (2026-06-30)

`kuka_grasp_friction.py` scripts a top-down grasp with NO kinematic attach: straddle → close → lift.
Result: **cube rose 0.210 m with the gripper, lateral tip-cube 0.002 m → held by friction.**

Working regime (now the contract constants):
- Combined URDF `kuka_iiwa_gripper.urdf`: KUKA + 2 prismatic fingers on link_7 → **9 DOF**,
  joint order `[a1..a7, finger_left_joint, finger_right_joint]`, link_pose order
  `[base_link, link_1..link_7(=idx7), finger_left(8), finger_right(9)]`, max_links=10.
- Uniform PD `stiffness=1200, damping=60, max_force=300` — squeezes the prismatic fingers (~34 N at
  the commanded overshoot) while capping arm torque. No per-joint gains needed.
- Cube material `create_material(2.0, 2.0, 0.0, friction_combine="max")` — wins the contact friction
  against the fingers' default material, so the grip holds without setting a finger material.
- `DEFAULT_Q = [0, 0.195, 0, -1.073, 0, 1.699, 0]` → tip (0.524, 0, 0.541), tool straight down.
- Gripper geometry: open gap 0.095, closed gap 0.041 (squeezes the 0.07 m cube).

**Decisions locked:** two-batch (no 6-DOF appendage); combined URDF (no append-after-load);
high uniform stiffness + max-combine cube friction (no per-joint gain API needed); pure Python.
