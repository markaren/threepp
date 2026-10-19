"""kuka_grasp_friction.py — the friction-grasp feasibility gate.

Scripts a top-down parallel-jaw grasp (no kinematic attach): open fingers straddle the cube,
close, then lift the arm by IK waypoints and hold — the cube must rise with the gripper, held
by friction alone. Runs K envs at once with the cube yawed and offset differently in each, so
the gate is a hold rate rather than one lucky case.

    python kuka_grasp_friction.py [K]

PASS if in at least 90 % of the envs the cube rose with the fingers (cube rise over 90 % of the
grasp point's, which is at least 0.08 m) and its centre ends within 0.02 m of the finger
midpoint. The arm sags under gravity on its PD drive, so the grasp point does not reach the
commanded 0.15 m; the gate measures the hold, not the tracking.
"""
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import kuka_grasp_contract as C  # noqa: E402
import kuka_grasp_robot as R  # noqa: E402
from kuka_grasp_kin import KukaKinematics, down_frame  # noqa: E402
from kuka_grasp_sim import KukaGraspSim  # noqa: E402

LIFT_DZ = 0.15


def grasp_yaw(cube_yaw, ref_yaw):
    """The jaw yaw square to the cube nearest to ref_yaw (the cube has fourfold symmetry)."""
    k = torch.round((ref_yaw - cube_yaw) / (np.pi / 2))
    return cube_yaw + k * (np.pi / 2)


def run(K=64, seed=0, verbose=True):
    sim = KukaGraspSim(K)
    dev = sim.device
    kin = KukaKinematics(R.find_kuka_urdf(), device=dev)
    g = torch.Generator(device=dev).manual_seed(seed)
    idx = torch.arange(K, device=dev)

    # cube placements: yaw in [-pi, pi), offset up to 8 cm from the table centre
    cyaw = (torch.rand(K, generator=g, device=dev) * 2 - 1) * np.pi
    cxy = (torch.rand(K, 2, generator=g, device=dev) * 2 - 1) * 0.08
    cxy[:, 0] += C.TABLE_CX
    cxy[:, 1] += C.TABLE_CY
    dq = torch.tensor(C.DEFAULT_Q, device=dev).expand(K, 7).contiguous()
    _, R0 = kin.grasp_frame(dq)
    yaw = grasp_yaw(cyaw, torch.atan2(R0[:, 1, 0], R0[:, 0, 0]))

    def solve(z):
        p = torch.stack([cxy[:, 0], cxy[:, 1], torch.full((K,), z, device=dev)], 1)
        q, pe, re = kin.ik(p, down_frame(yaw), dq)
        assert pe.max().item() < 3e-3 and re.max().item() < 3e-2, (pe.max().item(), re.max().item())
        return q

    q_grasp = solve(C.GRASP_TABLE_Z)
    q_lift = solve(C.GRASP_TABLE_Z + LIFT_DZ)
    opened = torch.full((K, 2), C.GRIP_OPEN, device=dev)
    closed = torch.full((K, 2), C.GRIP_CLOSE, device=dev)

    # 1. arm at the grasp posture, fingers open, cube placed between them
    q9 = torch.cat([q_grasp, opened], 1)
    sim.set_joint_state(idx, q9, torch.zeros_like(q9))
    pose = torch.zeros(K, 7, device=dev)
    pose[:, 2], pose[:, 3] = torch.sin(cyaw / 2), torch.cos(cyaw / 2)
    pose[:, 4] = sim.env_x + cxy[:, 0]
    pose[:, 5] = cxy[:, 1]
    pose[:, 6] = C.CUBE_REST_Z + C.CUBE_PLACE_DZ
    sim.set_cube_state(idx, pose)
    sim.settle_arm(q9, 50)
    c0 = sim.cube_position().clone()
    t0 = sim.tip_state()[0].clone()
    moved = (c0 - pose[:, 4:7]).norm(dim=1)

    # 2. close; 3. lift through eight waypoints; 4. hold
    sim.settle_arm(torch.cat([q_grasp, closed], 1), 100)
    for k in range(1, 9):
        f = k / 8.0
        sim.settle_arm(torch.cat([q_grasp + f * (q_lift - q_grasp), closed], 1), 25)
    sim.settle_arm(torch.cat([q_lift, closed], 1), 125)        # 0.5 s hold

    tip = sim.tip_state()[0]
    cube = sim.cube_position()
    rose = cube[:, 2] - c0[:, 2]
    tip_rose = tip[:, 2] - t0[:, 2]
    off = (cube - tip).norm(dim=1)
    held = (tip_rose > 0.08) & (rose > 0.9 * tip_rose) & (off < 0.02)
    rate = held.float().mean().item()
    if verbose:
        print(f"[friction] K={K}  placement disturbance (straddle) max {moved.max().item() * 1e3:.1f} mm")
        print(f"[friction] grasp point rose: mean {tip_rose.mean().item():.3f} m (commanded {LIFT_DZ})  "
              f"cube rose: mean {rose.mean().item():.3f} m  min {rose.min().item():.3f} m")
        print(f"[friction] cube to finger midpoint after the hold: mean {off.mean().item() * 1e3:.1f} mm  "
              f"max {off.max().item() * 1e3:.1f} mm")
    return rate


def main():
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    t0 = time.perf_counter()
    rate = run(K)
    ok = rate >= 0.9
    print("=" * 60)
    print(f"held {rate * 100:.1f} % of {K} envs ({time.perf_counter() - t0:.1f} s, solver iterations "
          f"arm {C.ARM_SOLVER_ITERS} / cube {C.CUBE_SOLVER_ITERS})")
    print(f"FRICTION GRASP GATE: {'PASS' if ok else 'FAIL'}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
