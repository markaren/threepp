"""kuka_grasp_sim.py — the two-batch GPU sim for the KUKA grasp task.

One `PhysxWorld(direct_gpu=True)` (Z-up) hosts K fixed-base KUKA+gripper articulations (the
`arm_batch`, 9 DOF each) and K free-base cube articulations (the `cube_batch`, read via root pose).
The scene is stepped once per substep via the arm batch (`world.simulateRaw` advances everything);
the cube batch is read-only. This is the architecture proven in SPIKE_FINDINGS.md.

It plays the role `GpuSim` plays for the locomotion examples, but kept standalone here so the grasp
experiment never has to touch the shared `threepp.rl.sim` (no regression risk to Spot/cartpole) and
can manage the second (cube) batch directly.

State exposed as stable CUDA tensors, refreshed in-place by `read()`:
    arm_jp, arm_jv      [K, 9]   arm+finger joint pos/vel, add-order
    link_pose           [K, L, 7]  every arm link's world pose [qx,qy,qz,qw, px,py,pz]
    cube_pose           [K, 7]   cube root pose [qx,qy,qz,qw, px,py,pz]
    cube_linvel/angvel  [K, 3]   cube root velocities (world frame)
Helpers: tip_state() (grasp-point pos + approach axis), cube_pos / cube_up.
"""
import ctypes
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

import kuka_grasp_contract as C  # noqa: E402
import kuka_grasp_robot as R  # noqa: E402

FLANGE_LINK = 7        # link_pose index of link_7 (the flange): base_link(0), link_1..7 (1..7)
FINGER_L_LINK = 8
FINGER_R_LINK = 9


def _torch_cuda_context():
    drv = ctypes.CDLL("nvcuda.dll")
    drv.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    drv.cuCtxGetCurrent.restype = ctypes.c_int
    ctx = ctypes.c_void_p()
    if drv.cuCtxGetCurrent(ctypes.byref(ctx)) != 0 or not ctx.value:
        raise RuntimeError("KukaGraspSim: could not read torch's CUDA context")
    return int(ctx.value)


def quat_rotate(q, v):
    """Rotate vectors v [.,3] by quaternions q [.,4]=(x,y,z,w). Batched."""
    xyz = q[..., 0:3]
    w = q[..., 3:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


class KukaGraspSim:
    def __init__(self, num_envs, device="cuda", spacing=2.5, render_visuals=False):
        if not tp.HAS_PHYSX:
            raise RuntimeError("KukaGraspSim needs a PhysX-enabled threepp build")
        self.K = num_envs
        self.device = torch.device(device)
        self.spacing = spacing

        # torch owns the CUDA primary context, then PhysX adopts it
        torch.zeros(1, device=self.device)
        torch.randn(64, 64, device=self.device).sum().item()
        torch.cuda.synchronize()
        ctx = _torch_cuda_context()
        self.world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=C.PHYS_DT,
                                   max_substeps=1, direct_gpu=True, cuda_context=ctx)

        urdf = R.generate_combined_urdf()
        # shared static geometry FIRST (ground + per-env tables)
        self.ground, self.tables = R.add_ground_and_tables(self.world, num_envs, spacing)
        self.cube_material = R.make_cube_material(self.world)

        # build + finalize ALL arts (arms then cubes) BEFORE constructing either batch
        self.arm_arts, self.arm_meshes, self.cube_arts, self.cube_meshes = [], [], [], []
        self.joint_names = None
        for i in range(num_envs):
            art, meshes, joints = R.build_arm(self.world, i, spacing, urdf, render_visuals=render_visuals)
            self.arm_arts.append(art)
            self.arm_meshes.append(meshes)
            if self.joint_names is None:
                self.joint_names = joints
        for i in range(num_envs):
            art, mesh = R.build_cube(self.world, i, spacing, self.cube_material, render_visuals=render_visuals)
            self.cube_arts.append(art)
            self.cube_meshes.append(mesh)

        self.arm_batch = tp.PhysxGpuBatch(self.world, self.arm_arts)
        self.cube_batch = tp.PhysxGpuBatch(self.world, self.cube_arts)
        self.max_dofs = self.arm_batch.max_dofs          # scene-global (== 9 here)
        self.max_links = self.arm_batch.max_links        # 10
        self.arm_dof = self.arm_arts[0].dof_order().shape[0]
        assert self.arm_dof == C.N_DOF, f"arm DOF {self.arm_dof} != {C.N_DOF}"

        dev = self.device
        self._perm = torch.from_numpy(self.arm_arts[0].dof_order().astype(np.int64)).to(dev)
        self._arm_gpu_idx = torch.from_numpy(self.arm_batch.gpu_indices().astype(np.int32)).to(dev)
        self._cube_gpu_idx = torch.from_numpy(self.cube_batch.gpu_indices().astype(np.int32)).to(dev)

        # GPU scratch (cache-order) + public state (add-order), refreshed in place by read()
        self._jp_gpu = torch.zeros(self.K, self.max_dofs, device=dev)
        self._jv_gpu = torch.zeros(self.K, self.max_dofs, device=dev)
        self._tgt_gpu = torch.zeros(self.K, self.max_dofs, device=dev)
        self.arm_jp = torch.zeros(self.K, C.N_DOF, device=dev)
        self.arm_jv = torch.zeros(self.K, C.N_DOF, device=dev)
        self._lp_flat = torch.zeros(self.K, self.max_links * 7, device=dev)
        self.link_pose = self._lp_flat.view(self.K, self.max_links, 7)
        self.cube_pose = torch.zeros(self.K, 7, device=dev)
        self.cube_linvel = torch.zeros(self.K, 3, device=dev)
        self.cube_angvel = torch.zeros(self.K, 3, device=dev)

        # per-env world X offset (to convert world poses to env-local)
        self.env_x = (torch.arange(self.K, device=dev, dtype=torch.float32) * spacing)
        self.read()

    # ---- stepping / reading -------------------------------------------------
    def read(self):
        self.arm_batch.read_joint_pos(self._jp_gpu)
        self.arm_batch.read_joint_vel(self._jv_gpu)
        self.arm_jp.copy_(self._jp_gpu[:, self._perm])
        self.arm_jv.copy_(self._jv_gpu[:, self._perm])
        self.arm_batch.read_link_pose(self._lp_flat)
        self.cube_batch.read_root_pose(self.cube_pose)
        self.cube_batch.read_root_linvel(self.cube_linvel)
        self.cube_batch.read_root_angvel(self.cube_angvel)

    def substep(self, n=C.SUBSTEPS, dt=C.PHYS_DT):
        for _ in range(n):
            self.arm_batch.step(dt)          # steps the whole scene (arms + cubes)
        self.read()

    # ---- control ------------------------------------------------------------
    def set_arm_targets(self, targets):
        """PD drive targets, [K, 9] add-order (arm 0..6, fingers 7,8)."""
        self._tgt_gpu[:, self._perm] = targets
        torch.cuda.synchronize()
        self.arm_batch.write_joint_target_pos(self._tgt_gpu)

    # ---- resets -------------------------------------------------------------
    def set_arm_joint_state(self, idx, pos, vel=None):
        """Reset arm+finger joints of envs `idx` to pos/vel ([m, 9] add-order)."""
        sub = self._arm_gpu_idx[idx].contiguous()
        gpu_pos = torch.zeros(pos.shape[0], self.max_dofs, device=self.device)
        gpu_pos[:, self._perm] = pos
        gpu_vel = torch.zeros(pos.shape[0], self.max_dofs, device=self.device)
        if vel is not None:
            gpu_vel[:, self._perm] = vel
        torch.cuda.synchronize()
        self.arm_batch.write_subset_joint_pos(gpu_pos, sub)
        self.arm_batch.write_subset_joint_vel(gpu_vel, sub)

    def set_cube_state(self, idx, pose, linvel=None, angvel=None):
        """Reset cube root of envs `idx`: pose [m,7]=[qx,qy,qz,qw,px,py,pz]."""
        sub = self._cube_gpu_idx[idx].contiguous()
        m = pose.shape[0]
        lv = (linvel if linvel is not None else torch.zeros(m, 3, device=self.device)).contiguous()
        av = (angvel if angvel is not None else torch.zeros(m, 3, device=self.device)).contiguous()
        torch.cuda.synchronize()
        self.cube_batch.write_subset_root_pose(pose.contiguous(), sub)
        self.cube_batch.write_subset_root_linvel(lv, sub)
        self.cube_batch.write_subset_root_angvel(av, sub)

    @staticmethod
    def make_pose(position, quat=(0.0, 0.0, 0.0, 1.0), device="cuda"):
        n = position.shape[0]
        out = torch.zeros(n, 7, device=device)
        out[:, 0:4] = torch.tensor(quat, device=device)
        out[:, 4:7] = position
        return out

    # ---- derived state ------------------------------------------------------
    def tip_state(self):
        """Grasp point = MIDPOINT of the two finger links (the real point where the cube is held) +
        the tool approach axis from the flange.

        Previously this was flange_pos + R(flange_quat)@(0,0,TIP_Z), but TIP_Z=0.20 placed that point
        ~0.13 m BELOW the actual fingers (flange link origin is well above the finger pads). So every
        reward/firm test that used `tip` thought the gripper was at the cube while the fingers hovered a
        gripper-length above it — the grasp could never physically close. Using the finger midpoint makes
        the grasp point coincide with the fingers, so reach/vert/firm are honest and the cube is reachable."""
        fl = self.link_pose[:, FINGER_L_LINK, 4:7]
        fr = self.link_pose[:, FINGER_R_LINK, 4:7]
        tip = 0.5 * (fl + fr)
        approach = quat_rotate(self.link_pose[:, FLANGE_LINK, 0:4],
                               torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.K, 3))
        return tip, approach

    def cube_position(self):
        return self.cube_pose[:, 4:7]

    def cube_up(self):
        return quat_rotate(self.cube_pose[:, 0:4],
                           torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.K, 3))

    def finger_opening(self):
        """Mean finger joint position (add-order indices 7,8); larger = more open."""
        return 0.5 * (self.arm_jp[:, 7] + self.arm_jp[:, 8])

    def settle_arm(self, q9, n):
        """Drive all envs to the 9-DOF target q9 (tensor or list) and step n substeps (no policy)."""
        if not torch.is_tensor(q9):
            q9 = torch.tensor(q9, device=self.device, dtype=torch.float32)
        if q9.dim() == 1:
            q9 = q9.unsqueeze(0).expand(self.K, -1).contiguous()
        self.set_arm_targets(q9)
        for _ in range(n):
            self.arm_batch.step(C.PHYS_DT)
        self.read()

    def search_posture(self, target_xyz, n=240, seed=0, settle=55, require_down=0.95, grip=None):
        """Sequential random search over (a2,a4,a6) for an arm posture whose tip reaches
        target_xyz with the tool pointing down. Returns the best 7-DOF arm q (a1=a3=a5=a7=0).
        Drives env 0 (all envs get the same target). For tests/scripted demos, not training."""
        if grip is None:
            grip = C.GRIP_OPEN
        rng = np.random.default_rng(seed)
        target = torch.tensor(target_xyz, device=self.device, dtype=torch.float32)
        best = None
        for _ in range(n):
            q7 = [0.0, float(rng.uniform(0.0, 1.0)), 0.0, float(rng.uniform(-1.6, -0.4)),
                  0.0, float(rng.uniform(1.0, 2.0)), 0.0]
            self.settle_arm(q7 + [grip, grip], settle)
            tip, approach = self.tip_state()
            down = float(-approach[0, 2])
            if down < require_down:
                continue
            err = float((tip[0] - target).norm()) + (1.0 - down)
            if best is None or err < best[0]:
                best = (err, q7)
        return best[1] if best else None
