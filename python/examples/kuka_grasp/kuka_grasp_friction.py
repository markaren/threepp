"""kuka_grasp_friction.py — the friction-grasp feasibility gate.

Scripts a top-down parallel-jaw grasp (no kinematic attach, no grasp-lock): straddle the cube with
the open gripper, CLOSE the fingers, then LIFT the arm — and check the cube rises with the gripper,
held by friction alone. This proves the physics supports a real friction grasp before we build the
RL env around it.

    python kuka_grasp_friction.py

PASS if the cube lifts with the gripper (cube z rises ~0.15 m) and stays between the fingers.
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import kuka_grasp_contract as C  # noqa: E402
from kuka_grasp_sim import KukaGraspSim  # noqa: E402


def main():
    np.set_printoptions(precision=3, suppress=True)
    sim = KukaGraspSim(num_envs=1, render_visuals=False)
    dev = sim.device

    cube_center = (C.TABLE_CX, C.TABLE_CY, C.CUBE_REST_Z)

    def reset_cube():
        pose = sim.make_pose(torch.tensor([cube_center], device=dev), device=dev)
        sim.set_cube_state(torch.tensor([0], device=dev), pose)

    def cube_xyz():
        return sim.cube_position()[0].cpu().numpy()

    def tip_xyz():
        return sim.tip_state()[0][0].cpu().numpy()

    print("[friction] searching grasp + lift postures...")
    q_grasp = sim.search_posture((C.TABLE_CX, C.TABLE_CY, C.CUBE_REST_Z - 0.005), grip=C.GRIP_OPEN)
    q_lift = sim.search_posture((C.TABLE_CX, C.TABLE_CY, C.CUBE_REST_Z + 0.22), seed=1, grip=C.GRIP_CLOSE)
    assert q_grasp and q_lift, "posture search failed"
    print(f"  q_grasp={[round(v,3) for v in q_grasp]}")
    print(f"  q_lift ={[round(v,3) for v in q_lift]}")

    # 1. hover above the cube, gripper OPEN, cube reset to centre
    reset_cube()
    sim.settle_arm(list(C.DEFAULT_Q) + [C.GRIP_OPEN, C.GRIP_OPEN], 120)
    reset_cube()
    sim.settle_arm(list(C.DEFAULT_Q) + [C.GRIP_OPEN, C.GRIP_OPEN], 60)
    print(f"\n[1] hover     tip={tip_xyz()}  cube={cube_xyz()}")

    # 2. descend so the open fingers straddle the cube, then CENTRE the cube under the actual
    #    tip (x,y) at table height — the posture search is imprecise, and this gate tests the
    #    friction hold, not IK precision (the policy will learn the centring itself).
    sim.settle_arm(q_grasp + [C.GRIP_OPEN, C.GRIP_OPEN], 150)
    tip = tip_xyz()
    centred = sim.make_pose(torch.tensor([[tip[0], tip[1], C.CUBE_REST_Z]], device=dev), device=dev)
    sim.set_cube_state(torch.tensor([0], device=dev), centred)
    sim.settle_arm(q_grasp + [C.GRIP_OPEN, C.GRIP_OPEN], 20)
    print(f"[2] straddle  tip={tip_xyz()}  cube={cube_xyz()}  finger_open={sim.finger_opening()[0]:.3f}")

    # 3. CLOSE the fingers on the cube (arm holds the grasp posture)
    sim.settle_arm(q_grasp + [C.GRIP_CLOSE, C.GRIP_CLOSE], 120)
    print(f"[3] close     tip={tip_xyz()}  cube={cube_xyz()}  finger_open={sim.finger_opening()[0]:.3f}")
    cube_before_lift = cube_xyz()

    # 4. LIFT — raise the arm, fingers stay closed; the cube must come up by friction
    zs = []
    for k in range(8):
        sim.settle_arm(q_lift + [C.GRIP_CLOSE, C.GRIP_CLOSE], 40)
        zs.append(float(cube_xyz()[2]))
    cube_after = cube_xyz()
    tip_after = tip_xyz()
    print(f"[4] lift      tip={tip_after}  cube={cube_after}")
    print(f"    cube z during lift: {[round(z,3) for z in zs]}")

    # ---- verdict ----
    rose = cube_after[2] - cube_before_lift[2]
    lateral = float(np.linalg.norm(cube_after[:2] - tip_after[:2]))
    held = rose > 0.10 and lateral < 0.05
    print("\n" + "=" * 56)
    print(f"cube rose {rose:.3f} m,  lateral tip-cube {lateral:.3f} m")
    print(f"FRICTION GRASP GATE: {'PASS' if held else 'FAIL'}")
    print("=" * 56)
    return 0 if held else 1


if __name__ == "__main__":
    sys.exit(main())
