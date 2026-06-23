"""Shared GpuSim Spot infrastructure — used by both the from-scratch walk (spot_walk_env) and the
frozen-walker stairs example (spot_stairs_env).

K Spots run in ONE PhysX direct-GPU scene (threepp.rl.GpuSim); state/obs/reward are torch tensor ops
on the GPU, no per-robot CPU loop. Z-up world (gravity 0,0,-9.81) to match the Spot/Isaac contract.
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from spot_deploy import build_spot

CONTROL_HZ = 50            # Spot runs the policy at 50 Hz (dt = 0.02 s)
DT = 1.0 / CONTROL_HZ
SUBSTEPS = 6              # GPU physics substeps per control tick (finer than the stairs env's 4)
SPACING = 3.0            # metres between Spots (lateral, along +Y) so they don't inter-collide


class SpotGpu:
    """build_robot factory for GpuSim: one Spot at a per-env lateral offset. Exposes `.art`.
    `gains` overrides the PD gains (the from-scratch gait uses a stiffer set than the Isaac default).
    `foot_friction` (if given) creates a per-env grippy restitution-0 foot material (eMIN so the foot's
    friction governs the contact) — exposes `.foot_mat` for per-env friction domain randomization."""
    def __init__(self, world, i, spacing=SPACING, gains=None, foot_friction=None):
        self.foot_mat = None
        if foot_friction is not None:
            self.foot_mat = world.create_material(static_friction=float(foot_friction),
                                                  dynamic_friction=float(foot_friction) * 0.9,
                                                  restitution=0.0, friction_combine="min",
                                                  restitution_combine="min")
        self.art, _ = build_spot(world, assets=None, base_xy=(0.0, i * spacing),
                                 gains=gains, foot_material=self.foot_mat)


def quat_rotate_inverse(q, v):
    """Rotate world-frame vectors v [N,3] into the body frame given body->world quat q [N,4]
    (qx,qy,qz,qw). Isaac Lab's exact formula (the policy was trained with it)."""
    qw = q[:, 3]
    qvec = q[:, :3]
    a = v * (2.0 * qw ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(qvec, v, dim=-1) * qw.unsqueeze(-1) * 2.0
    c = qvec * (qvec * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


def flat_ground(world, k, spacing, material=None):
    """A single large static floor box spanning the K-lane grid + forward travel; top at z=0.
    Pass a `material` (from world.create_material) for a grippy restitution-0 ground."""
    g = tp.Mesh(tp.BoxGeometry(60, spacing * k + 20, 1.0), tp.MeshStandardMaterial())
    g.position.set(20.0, spacing * (k - 1) * 0.5, -0.5)
    world.add_static(g, material=material)
