# KUKA Grasp — learning a physical parallel-jaw grasp

The KUKA LBR iiwa 14 R820 (the arm from `examples/projects/RobotCell`) fitted with a parallel-jaw
gripper learns, via PPO, to **reach a cube on a table, close the gripper, and lift it — held by real
finger friction**. This replaces RobotCell's *suction cheat* (a proximity-based kinematic attach that
teleports the crate) with a learned, physically-grasped pick.

Built on the in-repo `threepp.rl` GPU-batched PPO stack — same machinery as the Spot / cartpole
examples. **Pure Python**, no C++ rebuild: the cube is a free-base articulation read alongside the
arm in a second `PhysxGpuBatch` (see `SPIKE_FINDINGS.md`).

## Files

| File | Role |
|------|------|
| `kuka_grasp_contract.py` | Single source of truth: obs/action layout, geometry, control rates, PD gains, `DEFAULT_Q`. |
| `kuka_grasp_robot.py` | Generates the combined `kuka_iiwa_gripper.urdf` (KUKA + 2 prismatic fingers on link_7); builds arm / cube / table. `--inspect` verifies DOF + tunes posture. |
| `kuka_grasp_sim.py` | `KukaGraspSim` — two-batch GPU sim (arm batch + free-base cube batch over one Z-up scene). |
| `kuka_grasp_env.py` | `KukaGraspEnv` — the RL env (reach→grasp→lift reward, annealed grasp-lock scaffold, curriculum). `python kuka_grasp_env.py` runs the selftest. |
| `kuka_grasp_friction.py` | The friction-grasp physics gate: scripts straddle→close→lift, proves the cube is held by friction. |
| `train_kuka_grasp.py` | PPO trainer + curriculum + best-checkpoint. |
| `play_kuka_grasp.py` | Watch a trained policy grasp (GL, assist off). |

## Quickstart

```bash
cd python/examples/kuka_grasp

python kuka_grasp_robot.py --inspect     # verify the 9-DOF arm+gripper loads; check geometry
python kuka_grasp_friction.py            # physics gate: cube held by friction (PASS/FAIL)
python kuka_grasp_env.py                 # env selftest (obs shapes, cube rests, no NaN)

python train_kuka_grasp.py --envs 256 --iters 60 --smoke   # quick smoke (learning starts)
python train_kuka_grasp.py --envs 2048 --iters 2000        # full training run

python play_kuka_grasp.py --model kuka_grasp_best.pt        # watch it grasp
python play_kuka_grasp.py --shot grasp.png                  # headless render to PNG
```

## How it works

- **Control:** the policy outputs 8 actions — 7 arm joint-target offsets about `DEFAULT_Q` (clamped to
  the KUKA limits) + 1 gripper command (both prismatic fingers, mirrored). Joints are PD-driven.
- **Grasp = friction.** Fingers squeeze the cube (high uniform PD stiffness); a high-friction,
  `max`-combine cube material makes the contact grippy enough to hold a ~0.1 kg cube.
- **Reward** (sum of shaped terms): reach the cube → centre the tip over it → close around it → lift
  it (gated on actually holding it) → hold above the success height. Plus smoothness / joint-limit
  penalties and a knock-off-the-table terminal.
- **Scaffold + curriculum:** early on, a fraction of envs get a grasp-lock assist (the cube welds to
  the tip once the closed gripper straddles it) to bootstrap reach→close→lift; this anneals to 0 so
  the final policy grasps by friction. The cube spawn region grows from the table centre to the whole
  table over the first ~1500 iterations.

## Observation (38) / Action (8)

`OBS = arm qpos_rel(7) · arm qvel(7) · gripper open+vel(2) · tip (table-rel, 3) · tip→cube(3) ·
cube up(3) · cube linvel(3) · last_action(8) · [grasp flag, lift frac](2)`.
`ACT = arm joint offsets(7) · gripper(1)`. World is **Z-up** (matches the KUKA URDF's native frame).
