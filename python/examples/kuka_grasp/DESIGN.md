# KUKA Grasp RL Experiment — Design

Train the **KUKA LBR iiwa 14 R820** (the arm in `examples/projects/RobotCell`) fitted with a
**parallel-jaw gripper** to **reach, grasp, and lift a cube by friction**, using the in-repo
`threepp.rl` GPU-batched PPO stack. This replaces the RobotCell demo's *suction cheat* (a
proximity-based kinematic attach that teleports the crate to follow the tool — see
`tryGrab`/`onPreSubstep` in `examples/projects/RobotCell/main.cpp`) with a **learned physical grasp**.

**Locked decisions (from the user):**
- **Control:** full joint-space — the policy commands all 7 arm joints + the gripper (PD joint-target
  deltas, Spot-from-scratch style). No IK assist.
- **Grasp physics:** real **friction grasp**, *scaffolded* — an assisted grasp-lock bootstraps
  reach→close→lift early in the curriculum, then **anneals to zero** so the final policy holds by
  finger friction.
- **Deliverable:** Python **training + a Python play/viewer**. No C++ deploy-back this round (a clean
  follow-on: export `.tpnn` via the `SpotPolicy.hpp` path and wire into RobotCell).

Lives in `python/examples/kuka_grasp/`.

---

## Constraints discovered (must respect)

1. **`PhysxGpuBatch` is homogeneous** — *"one batch == one robot type"*: every articulation in a batch
   must share a DOF count (`include/threepp/extras/physx/PhysxGpuBatch.hpp:47-55`). There is **no
   separate rigid-body channel** — only articulation joint/root/link state is GPU-readable. **No GPU
   contact forces** exist.
2. **Therefore the cube is its own free body:** arm+gripper = one batch; the manipulable cube = a
   **second batch of free-base single-link articulations** (`create_articulation(fixed_base=False)`,
   one link, 0 joint-DOF, read via `root_pose`). The shared scene is stepped once
   (`world.simulateRaw`/`batch.step` advances *all* arts); the cube batch is **read, never stepped
   itself**. *Fallback* (if two batches won't coexist over one scene): model the cube as a **6-DOF
   free appendage** (3 prismatic + 3 revolute, passive) inside the arm articulation, read via
   `link_pose`. Two-batch is preferred (true free body, proper quaternion, no gimbal).
3. **No C++ rebuild needed.** Every binding required already exists and is used in-repo:
   `world.create_articulation(fixed_base=False)` and `art.add_link(...)` (`spot_deploy.py`),
   `tp.PhysxGpuBatch`, `read_root_pose`, `write_subset_root_pose`, `read_link_pose`,
   `write_joint_target_pos` (`python/threepp/rl/sim.py`). **Pure Python.**
4. **Grasp success is kinematic.** Detect a successful grasp by the cube rising with the gripper
   (cube height above the table while close to the tip / between the fingers). Sufficient for reward
   and terminal signals; no contact forces required.
5. **Build order:** create AND finalize **all** arts (arms + cubes) **before** constructing either
   `PhysxGpuBatch` — GPU indices are assigned on the first `simulate()`.

---

## Asset: arm + parallel-jaw gripper

- **Route (preferred): combined URDF** `kuka_iiwa_gripper.urdf` = `lbr_iiwa_14_r820.urdf` + a gripper
  subtree on `link_7` (flange): two **prismatic** finger links (boxes) sliding along the flange
  opening axis, **mirrored** by one action. The URDF loader parses prismatic + branching chains and
  approximates collision with Box/Capsule, which is exactly what fingers need.
- **Fallback routes** (decided by the Stage-1 spike, based on what the Python binding exposes):
  (a) load `lbr_iiwa_14_r820.urdf` into an articulation and **append** the two finger links via
  `add_link` before `finalize()`; (b) attach the two prismatic fingers directly to the flange link
  (no separate palm) — minimal gripper, 2 prismatic DOFs.
- **Gripper sizing:** cube edge `0.07 m`. Finger stroke so the **open** gap ≈ `0.09–0.10 m` (cube
  clears) and **closed** squeezes the `0.07 m` cube. **High finger-pad friction** material + enough
  finger drive `max_force` to generate a real friction hold.
- **Cube:** free-base single-link box (`0.07 m`), grippy friction material, light mass.
- **World:** table slab + ground as static colliders in `build_world` (shared geometry, built first).

---

## Contract — `kuka_grasp_contract.py` (single source of truth, imported everywhere)

- `CONTROL_HZ ≈ 50`, `DT = 0.02`; physics substep `0.002 × 10`.
- `ACTION_SCALE` (arm joint-delta), gripper stroke mapping `[closed, open]`.
- `default_q` (arm posture), gripper open/closed targets, PD gains (stiff enough not to sag —
  cf. Spot's stiffness-90 lesson).
- `OBS_DIM`, `ACT_DIM = 8` (7 arm deltas + 1 gripper), geometry constants (table, cube, gripper,
  lift threshold).

---

## Observation (~40 d; finalize exact layout in the contract; assemble with `torch.cat`)

`arm qpos_rel (7)` · `arm qvel·scale (7)` · `gripper opening + vel (2)` · `tip pos, base-relative (3)`
· `tip→cube vector (3)` · `cube up-axis / quat (3–4)` · `cube linvel (3)` · `last_action (8)` ·
`grasp/lift flags (1–2)`. Tip pose + cube pose come from `read_links` / the cube batch `root_pose`.

## Action (8)

`target_arm = default_q + ACTION_SCALE · a[:7]`. `a[7] → finger target ∈ [closed, open]`, both fingers
mirrored. Apply via `apply_drive_target`.

## Reward (sum of shaped, commented terms — the crux)

```
+ reach     exp(-k·‖tip − cube‖)
+ align     tool-axis down + tip above cube + fingers straddling the cube
+ pregrasp  fingers open while approaching, small standoff
+ grasp     tip near cube AND fingers closing with the cube between them
+ lift      cube height above table while near the tip   (capped)  ← the success driver
+ success   bonus when cube is above the lift threshold, held N ticks
- effort    action magnitude + action-rate (smoothness)
- limits    joint-limit proximity
- drop      cube falls / knocked off the table
```
Weights ramp via the curriculum.

## Scaffold — annealed grasp-lock

When the grasp condition holds (fingers closed around the cube, aligned, cube within finger span),
optionally hold the cube to the tip (kinematic `root`-target track, like the suction attach) with
strength/probability `assist ∈ [0,1]`. `assist` anneals **1 → 0** over the curriculum. The final
policy must hold the cube by **friction** alone.

## Curriculum — `set_iter(it)`

- **cube spawn region:** a fixed reachable spot → the full table.
- **arm reset posture:** pre-grasp hover over the cube → neutral home (must learn the full reach).
- **assist:** `1 → 0`.
- **reward weights:** lift/success ramp in.

## Reset / termination (follow the `threepp.rl` contract exactly)

- **reset:** arm to the curriculum posture (`set_joint_state`); cube random on the table within the
  current region (`set_prop_root_state`), resting, small random yaw.
- **done:** fixed-horizon **timeout** (`is_timeout = done`) **plus** a terminal **fail** when the cube
  is knocked off the table / unreachable (`is_timeout = False`, bootstrap V = 0). Snapshot
  `terminal_obs` **before** the auto-reset.

---

## `GpuSim` extension (additive, backward-compatible) — `python/threepp/rl/sim.py`

Add optional `build_props` (+ `n_props`, default 1) → build free-base prop arts and a **second
`PhysxGpuBatch`**; expose `prop_root_pose / prop_root_linvel / prop_root_angvel` (`[K, n_props, 7|3]`)
refreshed in `read()`, and `set_prop_root_state(idx, pose, linvel?, angvel?)`. **Default `None` →
behaviour identical to today.** Regression-check `cartpole` + `spot` smoke after the change.

---

## Training — `train_kuka_grasp.py`

PPO, `num_envs 4096` (start 2048), `hidden (512,256,128)`, `normalize_obs=True`, `horizon 32`,
`lr 3e-4`, `target_kl 0.02`, `anneal_lr`. `on_log` → `env.set_iter(it)` + checkpoint best (by
`success_rate`). Metrics as plain env attrs: `last_reach`, `last_grasp_rate`, `last_lift_rate`,
`last_success_rate`. Budget ~1500–3000 iters.

## Play — `play_kuka_grasp.py`

Single env in a normal `PhysxWorld`, GL render, `load_policy` (**norm-aware** `act_mean(norm.norm(o))`),
closed-loop grasp of a spawned cube; respawn on success. Shows the fingers + cube.

## Selftests

- contract dims + (if used) mirror symmetry.
- env smoke: shapes, no NaN, cube rests on the table, grasp detection fires when expected.
- training smoke: reach distance ↓ over ~100 iters.

---

## Stages (go/no-go; Opus reviews each before the next)

1. **Spike + assets** — prove two-batch coexistence (arm tracks drive targets; cube falls then rests
   on the table; both batches readable each step). Build the arm+gripper articulation, the cube prop,
   and the table. Resolve the arm-build route. **Output a findings report** (two-batch OK vs. fall
   back to the 6-DOF appendage; chosen asset route).
2. **Contract + env + `GpuSim` extension + selftests.**
3. **Training + reward/curriculum tuning** until reach→grasp→lift emerges (the long pole).
4. **Play/viewer + docs.**
