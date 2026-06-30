"""kuka_grasp_contract.py — the single source of truth for the KUKA grasp experiment.

Every other module (robot builder, sim, env, trainer, player, selftests) imports its constants
from HERE so the obs/action layout, geometry, control rates, and PD gains can never drift between
training and deployment. Mirrors the house style of `scratch_clock.py` (Spot) and
`cartpole_env.py`'s top-of-file constants block.

World convention: **Z-up** (gravity (0,0,-9.81)), matching the KUKA URDF's native frame —
`load_articulation` takes no base rotation, so we keep the world Z-up like `spot_deploy.py`.
The KUKA base sits at the origin; +X is "front" (toward the table); the flange (link_7) local +Z
is the tool **approach axis** (points down toward the table when the arm is posed to reach).
"""
import numpy as np

# --------------------------------------------------------------------------- #
#  Control / physics rates
# --------------------------------------------------------------------------- #
CONTROL_HZ = 50
DT = 1.0 / CONTROL_HZ          # 0.02 s control tick (policy acts at this rate)
PHYS_DT = 0.004                # physics substep
SUBSTEPS = int(round(DT / PHYS_DT))   # 5 substeps per control tick

EPISODE_S = 3.0                # short episode (keep horizons tight) — a grasp is a few seconds
MAX_STEPS = int(EPISODE_S * CONTROL_HZ)   # 150 control steps

# --------------------------------------------------------------------------- #
#  Arm (KUKA LBR iiwa 14 R820) — 7 revolute DOF, + 2 prismatic fingers = 9 DOF
# --------------------------------------------------------------------------- #
ARM_DOF = 7
GRIP_DOF = 2                   # two prismatic fingers (mirrored, driven by ONE action)
N_DOF = ARM_DOF + GRIP_DOF     # 9

# add-order joint names of the combined URDF (arm first, then fingers); verified at load time
ARM_JOINTS = ["joint_a1", "joint_a2", "joint_a3", "joint_a4", "joint_a5", "joint_a6", "joint_a7"]
FINGER_JOINTS = ["finger_left_joint", "finger_right_joint"]

# KUKA joint limits (rad), from the URDF — used to clamp targets
ARM_LIMITS = np.array([
    [-2.9668, 2.9668],
    [-2.0942, 2.0942],
    [-2.9668, 2.9668],
    [-2.0942, 2.0942],
    [-2.9668, 2.9668],
    [-2.0942, 2.0942],
    [-3.0541, 3.0541],
], np.float32)

# Default arm posture: flange hovering above the table centre with the gripper pointing down.
# TUNED empirically (see kuka_grasp_robot.py --inspect). Joint order = ARM_JOINTS (add order).
DEFAULT_Q = np.array([0.0, 0.195, 0.0, -1.073, 0.0, 1.699, 0.0], np.float32)
# ^ tuned via `kuka_grasp_robot.py --inspect`: tip at (0.524, 0.0, 0.541), tool straight down,
#   ~0.05 m above the cube top, all joints clear of limits.

# Action: 7 arm joint-target offsets (about DEFAULT_Q) + 1 gripper. ACT_DIM = 8.
ACT_DIM = ARM_DOF + 1          # 8
ARM_ACTION_SCALE = np.array([1.4, 1.0, 1.4, 1.2, 1.6, 1.4, 1.6], np.float32)  # rad span of a[i]∈[-1,1]

# PD drive (uniform across all 9 joints — load_articulation applies one gain set).
# High stiffness so the prismatic fingers squeeze hard (N/m) AND the arm tracks tightly (N·m/rad);
# max_force caps both the arm torque and the finger squeeze force.
PD_STIFFNESS = 1200.0
PD_DAMPING = 60.0
PD_MAX_FORCE = 300.0

# --------------------------------------------------------------------------- #
#  Gripper geometry (combined-URDF finger params, link_7 frame, metres)
# --------------------------------------------------------------------------- #
FINGER_BOX = (0.012, 0.030, 0.085)   # x(thickness) y(depth) z(length, along approach axis)
FINGER_MOUNT_X = 0.028               # prismatic joint origin: half-gap at joint pos 0
FINGER_MOUNT_Z = 0.165               # prismatic joint origin: z below the flange (tool0 is at 0.126)
FINGER_LOWER = -0.020                # prismatic limit: closed (fingers squeeze inward)
FINGER_UPPER = 0.035                 # prismatic limit: open (gap clears the cube)
GRIP_OPEN = 0.035                    # finger target → open  (gap ≈ 0.10 m)
GRIP_CLOSE = -0.015                  # finger target → squeeze a 0.07 m cube
TIP_Z = 0.200                        # grasp point: link_7 + (0,0,TIP_Z) in link_7 frame (between finger tips)

# --------------------------------------------------------------------------- #
#  Scene geometry (Z-up, metres)
# --------------------------------------------------------------------------- #
TABLE_TOP_Z = 0.42             # table surface height
TABLE_CX = 0.50                # table centre (front of the robot)
TABLE_CY = 0.00
TABLE_HALF = (0.22, 0.30)      # table slab half-extent in (x, y)
TABLE_THICK = 0.04

CUBE = 0.07                    # cube edge
CUBE_HALF = CUBE / 2
CUBE_REST_Z = TABLE_TOP_Z + CUBE_HALF   # cube CoM at rest on the table
CUBE_DENSITY = 300.0           # → ~0.10 kg
CUBE_FRICTION = 2.0            # high static/dynamic friction, "max" combine → grippy contact

LIFT_SUCCESS_DZ = 0.12         # cube lifted this far above the table top → grasp success
DROP_FAIL_DZ = 0.10            # cube knocked this far below the table top → terminal fail (off the table)

# --------------------------------------------------------------------------- #
#  Observation layout (assembled in kuka_grasp_env._obs via torch.cat)
# --------------------------------------------------------------------------- #
#   arm qpos_rel (7)  · arm qvel·scale (7)  · gripper opening+vel (2)
#   · tip pos, table-relative (3)  · tip→cube vector (3)  · cube up-axis (3)
#   · cube linvel (3)  · last_action (8)  · grasp/lift flags (2)
OBS_DIM = 7 + 7 + 2 + 3 + 3 + 3 + 3 + ACT_DIM + 2   # = 38
QVEL_SCALE = 0.1

assert OBS_DIM == 38, OBS_DIM
