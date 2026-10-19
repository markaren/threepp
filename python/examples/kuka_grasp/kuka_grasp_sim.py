"""kuka_grasp_sim.py — the two-batch GPU sim for the KUKA grasp task, on threepp.rl.GpuSim.

GpuSim owns the world, the CUDA-context handover and the arm batch (K fixed-base KUKA+gripper
articulations, 9 DOF each, add-order state). The K free-base cubes are a second PhysxGpuBatch
over the same Z-up world, built in the same build_robot call so every articulation is
finalized before either batch exists. Only the arm batch is stepped: its step is
world.simulateRaw and advances the whole scene, cubes included.

State, refreshed in place by read() (one read per buffer, so tensor identities stay stable):
    joint_pos, joint_vel   [K, 9]     arm + finger joints, add-order (GpuSim's)
    link_pose              [K, L, 7]  arm link world poses [qx,qy,qz,qw, px,py,pz]
    cube_pose              [K, 7]     cube root pose, same layout
    cube_linvel            [K, 3]     cube root linear velocity (world)
Link order on this path: base_link 0, link_1..link_7 1..7, finger_left 8, finger_right 9.

The arm is gravity-compensated, as a real KUKA controller is: before every substep each arm
link gets an upward m*g at its centre of mass (write_link_force; PhysX clears link forces after
every step). Without it the 64 kg arm on its PD drive (stiffness 1200, max force 300) sags
58 mm on average and up to 103 mm below an IK posture with the tool down over the table, and
a grasp posture presses the fingers into the table hard enough that friction stops them
closing. The cube keeps the scene's gravity.
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PYTHON_DIR = os.path.dirname(os.path.dirname(_HERE))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

import threepp as tp  # noqa: E402
from threepp.rl import GpuSim, quat_rotate  # noqa: E402

import kuka_grasp_contract as C  # noqa: E402
import kuka_grasp_robot as R  # noqa: E402

FLANGE_LINK = 7        # link_pose index of link_7 (the flange)
FINGER_L_LINK = 8
FINGER_R_LINK = 9


class _Cell:
    """What build_robot returns for env i: GpuSim wants `.art`; the rest is for the viewer."""

    def __init__(self, art, meshes, cube_art, cube_mesh):
        self.art, self.meshes, self.cube_art, self.cube_mesh = art, meshes, cube_art, cube_mesh


class KukaGraspSim(GpuSim):
    def __init__(self, num_envs, device="cuda", spacing=C.SPACING, render_visuals=False,
                 gravity_comp=True):
        self.spacing = spacing
        self.gravity_comp = gravity_comp
        self._urdf = R.generate_combined_urdf()
        self._render = render_visuals
        self._k = num_envs
        self.joint_names = None
        super().__init__(num_envs, self._build, gravity=(0.0, 0.0, -9.81), spacing=spacing,
                         device=device, build_world=self._build_world)
        assert self._perm.shape[0] == C.N_DOF, f"arm DOF {self._perm.shape[0]} != {C.N_DOF}"
        dev = self.device
        self.arm_meshes = [r.meshes for r in self.robots]
        self.cube_meshes = [r.cube_mesh for r in self.robots]
        self.cube_batch = tp.PhysxGpuBatch(self.world, [r.cube_art for r in self.robots])
        self._cube_gpu_idx = torch.from_numpy(self.cube_batch.gpu_indices().astype(np.int32)).to(dev)
        self.max_links = self.batch.max_links
        self._lp_flat = torch.zeros(self.K, self.max_links * 7, device=dev)
        self.link_pose = self._lp_flat.view(self.K, self.max_links, 7)
        self.cube_pose = torch.zeros(self.K, 7, device=dev)
        self.cube_linvel = torch.zeros(self.K, 3, device=dev)
        self._cube_angvel = torch.zeros(self.K, 3, device=dev)
        self.env_x = torch.arange(self.K, device=dev, dtype=torch.float32) * spacing
        self._ez = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(self.K, 3).contiguous()
        # links list order == GPU slot order for this chain (base, link_1..7, the two fingers)
        mass = torch.tensor([float(lk.mass) for lk in self.robots[0].art.links], device=dev)
        assert mass.numel() == self.max_links, (mass.numel(), self.max_links)
        self._gcomp = torch.zeros(self.K, self.max_links, 3, device=dev)
        self._gcomp[:, :, 2] = mass * 9.81
        torch.cuda.synchronize()     # _gcomp is never written again, so later writes need no sync
        self.read()

    # ---- construction ---------------------------------------------------------------
    def _build_world(self, world):
        self.ground, self.tables = R.add_ground_and_tables(world, self._k, self.spacing)
        self.cube_material = R.make_cube_material(world)

    def _build(self, world, i):
        art, meshes, joints = R.build_arm(world, i, self.spacing, self._urdf,
                                          render_visuals=self._render)
        if self.joint_names is None:
            self.joint_names = joints
        cube_art, cube_mesh = R.build_cube(world, i, self.spacing, self.cube_material)
        return _Cell(art, meshes, cube_art, cube_mesh)

    # ---- stepping ---------------------------------------------------------------------
    def step(self, dt, read=True):
        if self.gravity_comp:
            self.batch.write_link_force(self._gcomp)
        self.batch.step(dt)
        if read:
            self.read()

    def substep(self, dt, n):
        for _ in range(n):
            if self.gravity_comp:
                self.batch.write_link_force(self._gcomp)
            self.batch.step(dt)
        self.read()

    # ---- state ----------------------------------------------------------------------
    def read(self):
        """Joint state (GpuSim's add-order buffers), arm link poses and the cube root. Link
        velocities and the cube's angular velocity are not read: nothing uses them."""
        self.batch.read_joint_pos(self._jp_gpu)
        self.batch.read_joint_vel(self._jv_gpu)
        self.joint_pos.copy_(self._jp_gpu[:, self._perm])
        self.joint_vel.copy_(self._jv_gpu[:, self._perm])
        if hasattr(self, "cube_batch"):          # GpuSim's constructor has no cube batch yet
            self.batch.read_link_pose(self._lp_flat)
            self.cube_batch.read_root_pose(self.cube_pose)
            self.cube_batch.read_root_linvel(self.cube_linvel)

    def set_cube_state(self, idx, pose, linvel=None, angvel=None):
        """Teleport the cubes of envs `idx`: pose [m,7] = [qx,qy,qz,qw, px,py,pz] (world)."""
        sub = self._cube_gpu_idx[idx].contiguous()
        m = pose.shape[0]
        lv = (linvel if linvel is not None else torch.zeros(m, 3, device=self.device)).contiguous()
        av = (angvel if angvel is not None else torch.zeros(m, 3, device=self.device)).contiguous()
        pose = pose.contiguous()
        torch.cuda.synchronize()
        self.cube_batch.write_subset_root_pose(pose, sub)
        self.cube_batch.write_subset_root_linvel(lv, sub)
        self.cube_batch.write_subset_root_angvel(av, sub)

    # ---- derived state --------------------------------------------------------------
    def tip_state(self):
        """Grasp point = midpoint of the two finger links; approach = link_7's +Z in the world."""
        tip = 0.5 * (self.link_pose[:, FINGER_L_LINK, 4:7] + self.link_pose[:, FINGER_R_LINK, 4:7])
        approach = quat_rotate(self.link_pose[:, FLANGE_LINK, 0:4], self._ez)
        return tip, approach

    def cube_position(self):
        return self.cube_pose[:, 4:7]

    def cube_up(self):
        return quat_rotate(self.cube_pose[:, 0:4], self._ez)

    def finger_opening(self):
        """Mean finger joint position (add-order 7, 8); larger = more open."""
        return 0.5 * (self.joint_pos[:, 7] + self.joint_pos[:, 8])

    def settle_arm(self, q9, n):
        """Drive every env to the 9-DOF target q9 ([9] or [K, 9]) for n substeps, no policy."""
        if not torch.is_tensor(q9):
            q9 = torch.tensor(q9, device=self.device, dtype=torch.float32)
        if q9.dim() == 1:
            q9 = q9.unsqueeze(0).expand(self.K, -1).contiguous()
        self.apply_drive_target(q9)
        self.substep(C.PHYS_DT, n)
