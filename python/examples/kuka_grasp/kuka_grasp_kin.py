"""kuka_grasp_kin.py — batched forward and inverse kinematics of the KUKA + gripper in torch.

The reset pool needs thousands of arm postures that put the grasp point (the finger-link
midpoint) at a given place with the tool pointing down and the jaws square to the cube. A
random posture search in the simulator (what the June scaffold did) costs minutes per posture
at large K; the chain is seven revolute joints with axis-aligned origins, so a closed-form FK
and a damped-least-squares IK over a [N, 7] batch solve all of them at once.

The joint origins and axes are read from the stock URDF. The frames match PhysX's link poses
on the direct-GPU path; `python kuka_grasp_kin.py` checks that against a one-env sim.
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import kuka_grasp_contract as C  # noqa: E402

GRASP_OFFSET = (0.0, 0.0, C.FINGER_MOUNT_Z)   # finger midpoint in the link_7 frame (symmetric fingers)


def _chain(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    offs, axes = [], []
    for name in C.ARM_JOINTS:
        j = joints[name]
        o = j.find("origin")
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
        assert max(abs(v) for v in rpy) == 0.0, f"{name}: non-zero rpy is not supported"
        offs.append(xyz)
        axes.append([float(v) for v in j.find("axis").get("xyz").split()])
    return np.array(offs, np.float32), np.array(axes, np.float32)


def _axis_angle(axis, ang):
    """Rotation matrices [N,3,3] about a fixed unit axis [3] by angles [N] (Rodrigues)."""
    k = axis.view(1, 3).expand(ang.shape[0], 3)
    K = torch.zeros(ang.shape[0], 3, 3, device=ang.device)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = torch.sin(ang).view(-1, 1, 1), torch.cos(ang).view(-1, 1, 1)
    eye = torch.eye(3, device=ang.device).expand_as(K)
    return eye + s * K + (1.0 - c) * (K @ K)


class KukaKinematics:
    def __init__(self, urdf_path, device="cuda"):
        self.device = torch.device(device)
        offs, axes = _chain(urdf_path)
        self.offs = torch.tensor(offs, device=self.device)
        self.axes = torch.tensor(axes, device=self.device)
        self.grasp = torch.tensor(GRASP_OFFSET, device=self.device)
        self.lo = torch.tensor(C.ARM_LIMITS[:, 0], device=self.device)
        self.hi = torch.tensor(C.ARM_LIMITS[:, 1], device=self.device)

    def fk(self, q, base=None):
        """q [N,7] -> (R [N,7,3,3], p [N,7,3]) world frames of link_1..link_7, base at `base`
        ([N,3], default the origin)."""
        n = q.shape[0]
        R = torch.eye(3, device=self.device).expand(n, 3, 3)
        p = torch.zeros(n, 3, device=self.device) if base is None else base.clone()
        Rs, ps = [], []
        for i in range(7):
            p = p + (R @ self.offs[i].view(3, 1)).squeeze(-1)
            R = R @ _axis_angle(self.axes[i], q[:, i])
            Rs.append(R)
            ps.append(p)
        return torch.stack(Rs, 1), torch.stack(ps, 1)

    def grasp_frame(self, q, base=None):
        """The grasp point (finger midpoint) [N,3] and the link_7 rotation [N,3,3]: column 2 is
        the approach axis, column 0 the axis the left finger opens along."""
        R, p = self.fk(q, base)
        R7 = R[:, 6]
        return p[:, 6] + (R7 @ self.grasp.view(3, 1)).squeeze(-1), R7

    def jacobian(self, q, base=None):
        """Geometric Jacobian of the grasp point, [N,6,7] (linear rows first)."""
        R, p = self.fk(q, base)
        tip = p[:, 6] + (R[:, 6] @ self.grasp.view(3, 1)).squeeze(-1)
        J = torch.zeros(q.shape[0], 6, 7, device=self.device)
        for i in range(7):
            a = (R[:, i] @ self.axes[i].view(3, 1)).squeeze(-1)    # joint i's axis in the world
            J[:, 0:3, i] = torch.linalg.cross(a, tip - p[:, i], dim=-1)
            J[:, 3:6, i] = a
        return J, tip, R[:, 6]

    def ik(self, target_p, target_R, q0, q_rest=None, iters=200, damping=0.05, null_gain=0.1):
        """Damped least squares for the grasp point at target_p [N,3] with link_7 rotation
        target_R [N,3,3], from q0 [N,7], clamped to the joint limits every iteration. The
        seventh degree of freedom is pulled toward q_rest (default q0) in the null space.
        Returns (q, position error [N] m, orientation error [N] rad)."""
        q = q0.clone()
        q_rest = q0 if q_rest is None else q_rest
        eye7 = torch.eye(7, device=self.device)
        eye6 = torch.eye(6, device=self.device)
        for _ in range(iters):
            J, tip, R7 = self.jacobian(q)
            e_p = target_p - tip
            # orientation error as the rotation vector of target_R R7^T (small-angle via the
            # cross products of corresponding axes)
            e_w = 0.5 * sum(torch.linalg.cross(R7[:, :, k], target_R[:, :, k], dim=-1) for k in range(3))
            e = torch.cat([e_p, e_w], 1).unsqueeze(-1)
            Jt = J.transpose(1, 2)
            JJt = J @ Jt + (damping ** 2) * eye6
            Jpinv = Jt @ torch.linalg.solve(JJt, eye6.expand_as(JJt))
            dq = (Jpinv @ e).squeeze(-1)
            null = eye7 - Jpinv @ J
            dq = dq + null_gain * (null @ (q_rest - q).unsqueeze(-1)).squeeze(-1)
            q = torch.max(torch.min(q + dq, self.hi), self.lo)
        tip, R7 = self.grasp_frame(q)
        pos_err = (target_p - tip).norm(dim=-1)
        rot_err = torch.acos(((R7 * target_R).sum(dim=(1, 2)) * 0.5 - 0.5).clamp(-1.0, 1.0))
        return q, pos_err, rot_err


def down_frame(yaw):
    """Link_7 rotations [N,3,3] with the approach axis straight down and the left finger's
    opening axis at `yaw` [N] in the table plane."""
    n = yaw.shape[0]
    x = torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], -1)
    z = torch.tensor([0.0, 0.0, -1.0], device=yaw.device).expand(n, 3)
    y = torch.linalg.cross(z, x, dim=-1)
    return torch.stack([x, y, z], -1)


def _check():
    """FK vs PhysX at a few random postures, one env. PhysX places each link's pose at its
    collision-box centre, not at the URDF link frame, so the arm links' positions differ by
    design; the finger boxes are centred on their link origins, so the grasp point (their
    midpoint) and link_7's rotation are what is compared."""
    import kuka_grasp_robot as R
    from kuka_grasp_sim import FLANGE_LINK, KukaGraspSim
    from threepp.rl import quat_rotate
    sim = KukaGraspSim(1)
    kin = KukaKinematics(R.find_kuka_urdf(), device=sim.device)
    g = torch.Generator(device="cpu").manual_seed(0)
    worst_t = worst_r = 0.0
    for k in range(6):
        q7 = torch.tensor(C.DEFAULT_Q) if k == 0 else (torch.rand(7, generator=g) * 2 - 1) * 1.2
        q9 = torch.cat([q7, torch.tensor([C.GRIP_OPEN, C.GRIP_OPEN])]).to(sim.device).view(1, 9)
        sim.set_joint_state(torch.tensor([0], device=sim.device), q9, torch.zeros_like(q9))
        sim.apply_drive_target(q9)
        sim.substep(C.PHYS_DT * 1e-3, 1)          # a near-zero step refreshes the link poses
        tip, R7 = kin.grasp_frame(q7.to(sim.device).view(1, 7))
        q = sim.link_pose[:, FLANGE_LINK, 0:4]
        axes = torch.eye(3, device=sim.device)
        dr = max((quat_rotate(q, axes[i].view(1, 3))[0] - R7[0, :, i]).norm().item() for i in range(3))
        dt = (tip[0] - sim.tip_state()[0][0]).norm().item()
        worst_t, worst_r = max(worst_t, dt), max(worst_r, dr)
        print(f"posture {k}: grasp point {dt * 1e3:.2f} mm  link_7 axes {dr:.2e}  "
              f"tip {np.round(tip[0].cpu().numpy(), 3).tolist()}")
    ok = worst_t < 2e-3 and worst_r < 2e-3
    print(f"FK vs PhysX: worst grasp point {worst_t * 1e3:.2f} mm, worst link_7 axis {worst_r:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_check())
