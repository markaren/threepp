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
    CUDA first (it has, by the time we call this).

    Raises rather than returning 0 on failure: PhysxWorld treats cuda_context=0 as "make my
    own context", which silently produces the exact split-context failure this exists to
    prevent — so a failed lookup must be loud, not a silent fallback."""
    try:
        drv = ctypes.CDLL("nvcuda.dll")
    except OSError as e:
        raise RuntimeError("GpuSim: cannot load nvcuda.dll to read torch's CUDA context") from e
    drv.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    drv.cuCtxGetCurrent.restype = ctypes.c_int
    ctx = ctypes.c_void_p()
    err = drv.cuCtxGetCurrent(ctypes.byref(ctx))
    if err != 0 or not ctx.value:
        raise RuntimeError(
            f"GpuSim: could not read torch's current CUDA context (cuCtxGetCurrent err={err}, "
            f"ctx={ctx.value}); PhysX would otherwise create a separate context and break "
            "torch's cuBLAS. Ensure torch has run a CUDA op before constructing GpuSim.")
    return int(ctx.value)


class GpuSim:
    def __init__(self, num_envs, build_robot, gravity=(0, -9.81, 0), spacing=3.0,
                 device="cuda", read_root=False, read_links=False, build_world=None):
        if not tp.HAS_PHYSX:
            raise RuntimeError("GpuSim needs a PhysX-enabled threepp build (threepp.HAS_PHYSX is False)")
        self.K = num_envs
        self.device = torch.device(device)
        self.read_root = read_root   # also read the floating base each step (free-base robots)
        self.read_links = read_links  # also read every link's world pose + velocity (foot kinematics)

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
        # add-order public state, refreshed in-place by read() so the tensor identity is
        # stable across steps (obs = sim.joint_pos stays valid after the next step). Sized by
        # the permutation, not self.dof, so the copy_ in read() can't shape-mismatch if the
        # scene reports a larger maxDofs than this robot's DOF count.
        self.joint_pos = torch.zeros(self.K, self._perm.shape[0], device=self.device)
        self.joint_vel = torch.zeros(self.K, self._perm.shape[0], device=self.device)
        if self.read_root:
            # floating-base state (root link), refreshed in-place by read() so these tensors
            # keep a stable identity across steps. root_pose is PhysX's PxTransform layout:
            # [qx,qy,qz,qw, px,py,pz] (QUATERNION FIRST, then position) — use the root_quat /
            # root_position accessors, don't index it by hand. linear/angular velocity are
            # world-frame. Root is single per robot -> no DOF remap, so read straight in.
            self.root_pose = torch.zeros(self.K, 7, device=self.device)
            self.root_linvel = torch.zeros(self.K, 3, device=self.device)
            self.root_angvel = torch.zeros(self.K, 3, device=self.device)
        if self.read_links:
            # every link's world pose + velocity, refreshed by read(). link 0 = root, then links
            # in add_link order (NO DOF remap — these are per-LINK, not per-DOF). link_pose is PhysX
            # layout [qx,qy,qz,qw, px,py,pz]. Use for foot kinematics (clearance/slip rewards):
            # a foot tip = link_position + rotate(link_quat, local_tip_offset). The reads write the
            # flat buffers in place, so the [K, max_links, ...] views below stay live across read().
            self.max_links = self.batch.max_links
            self._lp_gpu = torch.zeros(self.K, self.max_links * 7, device=self.device)
            self._llv_gpu = torch.zeros(self.K, self.max_links * 3, device=self.device)
            self._lav_gpu = torch.zeros(self.K, self.max_links * 3, device=self.device)
            self.link_pose = self._lp_gpu.view(self.K, self.max_links, 7)
            self.link_linvel = self._llv_gpu.view(self.K, self.max_links, 3)
            self.link_angvel = self._lav_gpu.view(self.K, self.max_links, 3)

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
        # copy_ into the persistent buffers (don't rebind) so callers who stashed a reference
        # to joint_pos/joint_vel see this step's data instead of a stale tensor.
        self.joint_pos.copy_(self._jp_gpu[:, self._perm])
        self.joint_vel.copy_(self._jv_gpu[:, self._perm])
        if self.read_root:
            self.batch.read_root_pose(self.root_pose)        # in-place -> identity stable
            self.batch.read_root_linvel(self.root_linvel)
            self.batch.read_root_angvel(self.root_angvel)
        if self.read_links:
            self.batch.read_link_pose(self._lp_gpu)      # in-place -> the [K, max_links, ...] views update
            self.batch.read_link_linvel(self._llv_gpu)
            self.batch.read_link_angvel(self._lav_gpu)

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
        # Build the GPU buffer FIRST, then sync, then write: PhysX consumes the pointer on its
        # own CUDA stream with no start-event, so the scatter must be drained before the write
        # is enqueued. Syncing before _to_gpu would leave the scatter racing PhysX's read.
        gpu = self._to_gpu(target)
        torch.cuda.synchronize()
        self.batch.write_joint_target_pos(gpu)

    def set_joint_state(self, idx, pos, vel):
        """Reset the joints of envs `idx` to `pos`/`vel` ([m, dof] add-order) — for episode resets."""
        # Prepare every device buffer PhysX will read (subset indices + remapped pos/vel) before
        # the sync, so none of them is still being written on torch's stream when PhysX reads it.
        sub = self._gpu_idx[idx].contiguous()
        gpu_pos, gpu_vel = self._to_gpu(pos), self._to_gpu(vel)
        torch.cuda.synchronize()
        self.batch.write_subset_joint_pos(gpu_pos, sub)
        self.batch.write_subset_joint_vel(gpu_vel, sub)

    def set_root_state(self, idx, pose, linvel=None, angvel=None):
        """Reset the floating base of envs `idx`: pose [m,7] in PhysX layout [qx,qy,qz,qw, px,py,pz]
        (build it with make_root_pose), optional linear/angular velocity [m,3] (default zero) —
        for teleporting a free-base robot on reset."""
        sub = self._gpu_idx[idx].contiguous()
        m = pose.shape[0]
        # Materialize all the contiguous device buffers PhysX will read before the sync, so the
        # sync actually covers them (PhysX reads on its own stream with no start-event).
        gpu_pose = pose.contiguous()
        gpu_linvel = (linvel if linvel is not None else torch.zeros(m, 3, device=self.device)).contiguous()
        gpu_angvel = (angvel if angvel is not None else torch.zeros(m, 3, device=self.device)).contiguous()
        torch.cuda.synchronize()
        self.batch.write_subset_root_pose(gpu_pose, sub)
        self.batch.write_subset_root_linvel(gpu_linvel, sub)
        self.batch.write_subset_root_angvel(gpu_angvel, sub)
