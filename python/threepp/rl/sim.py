"""GpuSim — the easy way to run K articulations on the GPU.

Building a GPU-vectorized robot sim by hand means juggling: making torch own the CUDA
context so PhysX can adopt it, the direct-GPU batch, the GPU-vs-add DOF-order remap,
pre-allocated device buffers, and the torch<->PhysX sync. None of that is your problem.

    from threepp.rl import GpuSim
    sim = GpuSim(4096, lambda world, i: CartPole(world, x0=i * 3.0))
    sim.apply_force(force)          # [K, dof] generalized forces, in add-order
    sim.step(dt)                    # advance physics, refresh state
    pos, vel = sim.joint_pos, sim.joint_vel   # [K, dof] torch cuda tensors, add-order

`build_robot(world, i)` returns anything with an `.art` (a finalized threepp Articulation),
e.g. a CartPole/Hexapod factory. All state tensors are in canonical add-order — joint i is
the i-th link you added — for ANY robot, so your obs/reward indices never depend on PhysX's
internal cache ordering.
"""
import ctypes

import numpy as np
import torch

import threepp as tp


def _torch_cuda_context():
    """The CUDA primary context torch is using, as an int — so PhysX can adopt the SAME
    context (a separate PhysX context makes torch's cuBLAS fail). torch must have touched
    CUDA first (it has, by the time we call this)."""
    try:
        drv = ctypes.CDLL("nvcuda.dll")
    except OSError:
        return 0
    ctx = ctypes.c_void_p()
    return int(ctx.value or 0) if drv.cuCtxGetCurrent(ctypes.byref(ctx)) == 0 else 0


class GpuSim:
    def __init__(self, num_envs, build_robot, gravity=(0, -9.81, 0), spacing=3.0,
                 device="cuda", read_root=False, build_world=None):
        if not tp.HAS_PHYSX:
            raise RuntimeError("GpuSim needs a PhysX-enabled threepp build (threepp.HAS_PHYSX is False)")
        self.K = num_envs
        self.device = torch.device(device)
        self.read_root = read_root   # also read the floating base each step (free-base robots)

        # make torch create + own the device primary context, then hand it to PhysX
        torch.zeros(1, device=self.device)
        torch.randn(64, 64, device=self.device).sum().item()   # warm cuBLAS in this context
        torch.cuda.synchronize()
        self.world = tp.PhysxWorld(gravity=tp.Vector3(*gravity), direct_gpu=True,
                                   cuda_context=_torch_cuda_context())
        # add shared environment geometry (ground, obstacles) BEFORE the robots/batch so it's
        # part of the GPU scene from the first step — free-base robots need a ground to stand on.
        if build_world is not None:
            build_world(self.world)
        self.robots = [build_robot(self.world, i) for i in range(num_envs)]
        self.batch = tp.PhysxGpuBatch(self.world, [r.art for r in self.robots])
        self.dof = self.batch.max_dofs

        # add-order <-> GPU-cache-order permutation (perm[i] = GPU slot of add-order joint i),
        # applied only here so everything the caller sees is stable add-order.
        self._perm = torch.from_numpy(self.robots[0].art.dof_order().astype(np.int64)).to(self.device)
        self._gpu_idx = torch.from_numpy(self.batch.gpu_indices().astype(np.int32)).to(self.device)
        self._jp_gpu = torch.zeros(self.K, self.dof, device=self.device)
        self._jv_gpu = torch.zeros(self.K, self.dof, device=self.device)
        self._force = torch.zeros(self.K, self.dof, device=self.device)
        self.joint_pos = torch.zeros(self.K, self.dof, device=self.device)   # add-order, after read()
        self.joint_vel = torch.zeros(self.K, self.dof, device=self.device)
        if self.read_root:
            # floating-base state (root link), refreshed by read(). root_pose is PhysX's
            # PxTransform layout: [qx,qy,qz,qw, px,py,pz] (QUATERNION FIRST, then position) —
            # use the root_quat / root_position accessors, don't index it by hand. linear/angular
            # velocity are world-frame. Root is single per robot -> no DOF remap.
            self._rp_gpu = torch.zeros(self.K, 7, device=self.device)
            self._rlv_gpu = torch.zeros(self.K, 3, device=self.device)
            self._rav_gpu = torch.zeros(self.K, 3, device=self.device)
            self.root_pose = torch.zeros(self.K, 7, device=self.device)
            self.root_linvel = torch.zeros(self.K, 3, device=self.device)
            self.root_angvel = torch.zeros(self.K, 3, device=self.device)

    @property
    def root_quat(self):
        """Root orientation (qx,qy,qz,qw), [K,4] — the first 4 of root_pose. Needs read_root."""
        return self.root_pose[:, 0:4]

    @property
    def root_position(self):
        """Root world position (x,y,z), [K,3] — the last 3 of root_pose. Needs read_root."""
        return self.root_pose[:, 4:7]

    @staticmethod
    def make_root_pose(position, quat=(0.0, 0.0, 0.0, 1.0), device="cuda"):
        """Build an [n,7] root pose in PhysX layout [qx,qy,qz,qw, px,py,pz] from positions
        ([n,3] tensor) and a single quaternion (qx,qy,qz,qw). For set_root_state."""
        n = position.shape[0]
        out = torch.zeros(n, 7, device=device)
        out[:, 0:4] = torch.tensor(quat, device=device)
        out[:, 4:7] = position
        return out

    def _to_gpu(self, canon):
        out = torch.zeros(canon.shape[0], self.dof, device=self.device)
        out[:, self._perm] = canon
        return out

    def read(self):
        """Refresh joint_pos / joint_vel (add-order) — and, if read_root, the floating base —
        from the GPU without stepping."""
        self.batch.read_joint_pos(self._jp_gpu)
        self.batch.read_joint_vel(self._jv_gpu)
        self.joint_pos = self._jp_gpu[:, self._perm].contiguous()
        self.joint_vel = self._jv_gpu[:, self._perm].contiguous()
        if self.read_root:
            self.batch.read_root_pose(self._rp_gpu)
            self.batch.read_root_linvel(self._rlv_gpu)
            self.batch.read_root_angvel(self._rav_gpu)
            self.root_pose, self.root_linvel, self.root_angvel = self._rp_gpu, self._rlv_gpu, self._rav_gpu

    def step(self, dt):
        """Advance every robot one physics step and refresh joint_pos / joint_vel."""
        self.batch.step(dt)
        self.read()

    def apply_force(self, force):
        """Apply per-DOF generalized forces/torques, [K, dof] add-order (re-apply each step)."""
        self._force[:, self._perm] = force
        torch.cuda.synchronize()
        self.batch.write_joint_force(self._force)

    def apply_drive_target(self, target):
        """Set per-DOF PD drive position targets, [K, dof] add-order."""
        torch.cuda.synchronize()
        self.batch.write_joint_target_pos(self._to_gpu(target))

    def set_joint_state(self, idx, pos, vel):
        """Reset the joints of envs `idx` to `pos`/`vel` ([m, dof] add-order) — for episode resets."""
        sub = self._gpu_idx[idx].contiguous()
        torch.cuda.synchronize()
        self.batch.write_subset_joint_pos(self._to_gpu(pos), sub)
        self.batch.write_subset_joint_vel(self._to_gpu(vel), sub)

    def set_root_state(self, idx, pose, linvel=None, angvel=None):
        """Reset the floating base of envs `idx`: pose [m,7] in PhysX layout [qx,qy,qz,qw, px,py,pz]
        (build it with make_root_pose), optional linear/angular velocity [m,3] (default zero) —
        for teleporting a free-base robot on reset."""
        sub = self._gpu_idx[idx].contiguous()
        m = pose.shape[0]
        torch.cuda.synchronize()
        self.batch.write_subset_root_pose(pose.contiguous(), sub)
        self.batch.write_subset_root_linvel((linvel if linvel is not None
                                             else torch.zeros(m, 3, device=self.device)).contiguous(), sub)
        self.batch.write_subset_root_angvel((angvel if angvel is not None
                                             else torch.zeros(m, 3, device=self.device)).contiguous(), sub)
