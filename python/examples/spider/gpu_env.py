"""GPU-resident vectorized hexapod locomotion env (the payoff of the direct-GPU path).

K hexapods live in ONE PhysX direct-GPU scene. Every step:
  - actions (torch cuda) -> joint PD targets  (PhysxGpuBatch.write_joint_target_pos)
  - batch.step()                              (all K advance on the GPU)
  - state (joints, root pose/vel) -> torch cuda tensors (zero-copy reads)
  - obs / reward / done / per-env reset all computed in torch on the GPU.

Nothing crosses to the CPU in the hot loop, so this runs ~100x the CPU vec env and a
walking gait trains in minutes. The policy directly controls the 12 joints (no CPG); a
free-running clock in the obs gives it a rhythm to latch onto.

NOTE: torch must touch CUDA before the PhysxWorld is created so PhysX adopts the device
primary context (then torch .data_ptr()s are valid in PhysX's context). __init__ does this.
"""
import ctypes
import math
import os
import sys

import numpy as np
import torch


def current_cuda_context():
    """torch's current CUDA (primary) context handle as an int, via the driver API.
    PhysX adopts it so the two share one context (else cuBLAS errors — see gpu_env)."""
    try:
        drv = ctypes.CDLL("nvcuda.dll")
    except OSError:
        return 0
    ctx = ctypes.c_void_p()
    if drv.cuCtxGetCurrent(ctypes.byref(ctx)) != 0:
        return 0
    return int(ctx.value or 0)

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)

import threepp as tp
from hexapod import Hexapod

OBS_DIM = 34
ACT_DIM = 12
START_Y = 0.40
# Max commanded yaw rate. The hexapod morphology physically tops out around ~0.3 rad/s
# (the hand-tuned CPG gait manages only ~0.14 rad/s), so the earlier 1.2 put the yaw
# target (0.96 rad/s) far in the tracking bell's flat tail -> no gradient to turn, policy
# gave up. 0.5 -> cmd +/-0.8 targets +/-0.4 rad/s, reachable-with-effort (clean gradient).
MAX_YAW = 0.5
CLOCK_HZ = 1.6

# Command-tracking reward constants — designed via judge-panel + adversarial verify
# (legged_gym style). Tracking bells peak AT the commanded speed/yaw so exceeding the
# command is penalised -> genuine speed & turn control (the monotonic v_fwd*cmd_f term
# the first run used just rewarded "always sprint"). MAX_SPEED=1.9 sits below the ~2.2
# m/s ceiling so the top command's bell is reachable from both sides (no sprint bias).
MAX_SPEED = 1.0   # lowered 1.9->1.0: a slow, deliberate gait is steerable; the fast ~1.9
                  # m/s run is a committed tripod that falls when perturbed to turn, so RL
                  # suppressed turning. Trade top speed for the controllability we actually want.
# SIG_YAW must be TIGHT vs the yaw target (0.4 rad/s at cmd 0.8): a wide bell lets the
# policy collect most of the turn reward WITHOUT turning (not turning saves forward
# speed), so it never learns to turn. 0.15 (~target/2.7, the legged_gym ratio) makes
# not-turning cost ~exp(-3.6)=0.03 -> a real gradient to turn. SIG_V stays loose (speed
# already controls well; raw v_fwd bobs).
SIG_V, SIG_YAW = 0.30, 0.15
W_VEL, W_YAW = 1.6, 1.1   # W_YAW bumped (0.9->1.1): turning costs forward speed, so the
                          # turn bell needs more weight to be worth the v_fwd it sacrifices
W_ALIVE, W_UP = 0.10, 0.20
W_LAT, W_ENERGY, W_JERK = 0.30, 0.004, 0.04
FALL_PEN, UP_GATE_FLOOR = 2.0, 0.30
# Per-joint action scale (coxa, femur) tiled over 6 legs -> target angle (rad).
JOINT_SCALE = torch.tensor([0.7, 1.1] * 6, dtype=torch.float32)


class HexapodGpuVecEnv:
    def __init__(self, num_envs=2048, control_hz=30, episode_s=12.0, command_hold_s=3.0,
                 spacing=2.5, device="cuda", seed=0):
        self.K = num_envs
        self.dt = 1.0 / control_hz
        self.max_steps = int(episode_s * control_hz)
        self.command_hold = int(command_hold_s * control_hz)
        self.device = torch.device(device)
        self.g = torch.Generator(device=self.device).manual_seed(seed)

        # Make torch create + make-current its device primary context, then hand that
        # same context to PhysX so the two share ONE CUDA context. Without this PhysX
        # makes its own context and torch's cuBLAS GEMMs fail (CUBLAS_STATUS_INTERNAL_ERROR).
        torch.zeros(1, device=self.device)
        # warm cuBLAS in this context too (binds its handle here, pre-PhysX)
        _w = torch.randn(64, 64, device=self.device)
        (_w @ _w).sum().item()
        torch.cuda.synchronize()
        ctx = current_cuda_context()

        self.world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0), direct_gpu=True, cuda_context=ctx)
        ground = tp.Mesh(tp.BoxGeometry(spacing * self.K + 40, 1, 80), tp.MeshStandardMaterial())
        ground.position.set(spacing * self.K * 0.5, -0.5, 0.0)
        self.world.add_static(ground)
        self.bases_x = torch.arange(self.K, device=self.device, dtype=torch.float32) * spacing
        self.spiders = [Hexapod(self.world, position=(i * spacing, START_Y, 0.0)) for i in range(self.K)]
        self.batch = tp.PhysxGpuBatch(self.world, [s.art for s in self.spiders])

        K, dev = self.K, self.device
        self.joint_scale = JOINT_SCALE.to(dev)
        # GPU index buffer for subset resets (int32 bit-pattern == PxU32 for small ids).
        self.gpu_idx = torch.from_numpy(self.batch.gpu_indices().astype(np.int32)).to(dev)

        # Persistent state tensors (filled by zero-copy reads each step).
        self.jp = torch.zeros(K, 12, device=dev)
        self.jv = torch.zeros(K, 12, device=dev)
        self.pose = torch.zeros(K, 7, device=dev)     # qx,qy,qz,qw, px,py,pz
        self.linvel = torch.zeros(K, 3, device=dev)
        self.angvel = torch.zeros(K, 3, device=dev)

        self.clock = torch.zeros(K, device=dev)
        self.command = torch.zeros(K, 2, device=dev)  # (forward in [0.3,1], turn in [-1,1])
        self.prev_act = torch.zeros(K, 12, device=dev)
        self.steps = torch.zeros(K, dtype=torch.long, device=dev)

        # Home pose per env: identity quat, base x, START_Y, z=0.
        self.home_pose = torch.zeros(K, 7, device=dev)
        self.home_pose[:, 3] = 1.0
        self.home_pose[:, 4] = self.bases_x
        self.home_pose[:, 5] = START_Y

    # --- helpers ---------------------------------------------------------
    def _read_state(self):
        self.batch.read_joint_pos(self.jp.data_ptr())
        self.batch.read_joint_vel(self.jv.data_ptr())
        self.batch.read_root_pose(self.pose.data_ptr())
        self.batch.read_root_linvel(self.linvel.data_ptr())
        self.batch.read_root_angvel(self.angvel.data_ptr())

    def _sample_command(self, n):
        rand = lambda: torch.rand(n, device=self.device, generator=self.g)
        f = 0.3 + 0.7 * rand()
        # Turning is a hard-to-discover asymmetric behaviour; with the old 50% straight
        # mass the policy could satisfice by nailing the straight commands and never learn
        # to turn. Make turning dominant (only ~25% straight) so it MUST learn to turn.
        t = torch.where(rand() < 0.25, torch.zeros(n, device=self.device), rand() * 1.6 - 0.8)
        # ~18% explicit pivot draws (slow forward + hard turn) so the policy is trained
        # on the slow-tight-turn regime that uniform (f,t) sampling under-covers — exactly
        # where the eval probes hardest (pivot rows).
        pivot = rand() < 0.18
        sign = (rand() < 0.5).float() * 2.0 - 1.0
        f = torch.where(pivot, 0.3 + 0.2 * rand(), f)
        t = torch.where(pivot, sign * (0.6 + 0.2 * rand()), t)
        return torch.stack([f, t], dim=1)

    def _kinematics(self):
        qx, qy, qz, qw = self.pose[:, 0], self.pose[:, 1], self.pose[:, 2], self.pose[:, 3]
        up = 1.0 - 2.0 * (qx * qx + qz * qz)
        fx = 1.0 - 2.0 * (qy * qy + qz * qz)
        fz = 2.0 * (qx * qz - qw * qy)
        fl = torch.sqrt(fx * fx + fz * fz).clamp_min(1e-6)
        fx, fz = fx / fl, fz / fl
        vx, vz = self.linvel[:, 0], self.linvel[:, 2]
        v_fwd = vx * fx + vz * fz
        v_lat = vx * fz - vz * fx
        yawrate = self.angvel[:, 1]
        return up, fx, fz, v_fwd, v_lat, yawrate

    def _obs(self, up, fx, fz, v_fwd, v_lat, yawrate):
        tail = torch.stack([up, fx, fz, v_fwd, v_lat, yawrate,
                            torch.cos(self.clock), torch.sin(self.clock),
                            self.command[:, 0], self.command[:, 1]], dim=1)
        return torch.cat([self.jp, self.jv * 0.1, tail], dim=1)

    def _reset_idx(self, idx):
        """Reset the envs in `idx` (long tensor) in-place on the GPU."""
        n = idx.numel()
        if n == 0:
            return
        sub = self.gpu_idx[idx].contiguous()
        zeros_dof = torch.zeros(n, 12, device=self.device)
        zeros3 = torch.zeros(n, 3, device=self.device)
        pose = self.home_pose[idx].contiguous()
        # Materialize the input buffers on torch's stream before PhysX reads them on
        # its own stream (else PhysX races ahead and reads uninitialized memory).
        torch.cuda.synchronize()
        self.batch.write_subset_joint_pos(zeros_dof.data_ptr(), sub.data_ptr(), n)
        self.batch.write_subset_joint_vel(zeros_dof.data_ptr(), sub.data_ptr(), n)
        self.batch.write_subset_root_pose(pose.data_ptr(), sub.data_ptr(), n)
        self.batch.write_subset_root_linvel(zeros3.data_ptr(), sub.data_ptr(), n)
        self.batch.write_subset_root_angvel(zeros3.data_ptr(), sub.data_ptr(), n)
        self.clock[idx] = torch.rand(n, device=self.device, generator=self.g) * (2 * math.pi)
        self.command[idx] = self._sample_command(n)
        self.prev_act[idx] = 0.0
        self.steps[idx] = 0

    # --- API -------------------------------------------------------------
    def reset(self):
        all_idx = torch.arange(self.K, device=self.device)
        self._reset_idx(all_idx)
        # A few settle steps holding the stand pose so robots start grounded.
        hold = torch.zeros(self.K, 12, device=self.device)
        torch.cuda.synchronize()
        for _ in range(4):
            self.batch.write_joint_target_pos(hold.data_ptr())
            self.batch.step(self.dt)
        self._read_state()
        up, fx, fz, v_fwd, v_lat, yawrate = self._kinematics()
        return self._obs(up, fx, fz, v_fwd, v_lat, yawrate)

    @torch.no_grad()
    def step(self, actions):
        actions = actions.clamp(-1.0, 1.0)
        targets = (actions * self.joint_scale).contiguous()
        torch.cuda.synchronize()  # ensure `targets` is produced before PhysX reads it
        self.batch.write_joint_target_pos(targets.data_ptr())
        self.batch.step(self.dt)
        self._read_state()

        self.clock += CLOCK_HZ * 2.0 * math.pi * self.dt
        self.steps += 1
        up, fx, fz, v_fwd, v_lat, yawrate = self._kinematics()

        cmd_f, cmd_t = self.command[:, 0], self.command[:, 1]
        tgt_v = cmd_f * MAX_SPEED                                  # commanded forward speed (0.57..1.90 m/s)
        tgt_w = cmd_t * MAX_YAW                                    # commanded yaw rate (0 -> hold heading)
        # Dominant command-tracking: exp bells peaked AT the command, so going faster or
        # turning more than commanded is penalised -> the only way to score is to obey.
        r_vel = W_VEL * torch.exp(-((v_fwd - tgt_v) ** 2) / (2.0 * SIG_V ** 2))
        r_yaw = W_YAW * torch.exp(-((yawrate - tgt_w) ** 2) / (2.0 * SIG_YAW ** 2))
        # Upright gate: tracking only pays while the chassis is level (gate=1 at up=1,
        # 0 at the fall threshold) so a tipping robot can't farm the tracking bells.
        up_gate = (up.clamp(UP_GATE_FLOOR, 1.0) - UP_GATE_FLOOR) / (1.0 - UP_GATE_FLOOR)
        energy = (actions ** 2).mean(dim=1)
        jerk = ((actions - self.prev_act) ** 2).mean(dim=1)
        rew = (up_gate * (r_vel + r_yaw)
               + W_ALIVE                                           # tiny unconditional alive floor (bootstrap)
               - W_UP * (1.0 - up.clamp(0.0, 1.0))                 # smooth uprightness shaping (recovery/transfer)
               - W_LAT * v_lat.abs()                               # no sideways skating; keep v_fwd honest
               - W_ENERGY * energy - W_JERK * jerk)
        fell = up < 0.3
        rew = rew - FALL_PEN * fell.float()

        # Resample command mid-episode for envs that hit the hold interval (still alive).
        hold_due = (self.steps % self.command_hold == 0)
        resample = torch.nonzero(hold_due & ~fell, as_tuple=False).squeeze(-1)
        if resample.numel() > 0:
            self.command[resample] = self._sample_command(resample.numel())

        timeout = self.steps >= self.max_steps
        done = fell | timeout
        self.prev_act = actions

        obs = self._obs(up, fx, fz, v_fwd, v_lat, yawrate)
        done_idx = torch.nonzero(done, as_tuple=False).squeeze(-1)
        if done_idx.numel() > 0:
            self._reset_idx(done_idx)
            # Overwrite obs rows for reset envs with their fresh stand-pose obs
            # (jp=jv=0, up=1, fwd=(1,0), vel=0; new clock phase + command).
            r = done_idx
            obs[r] = 0.0
            obs[r, 24] = 1.0  # up
            obs[r, 25] = 1.0  # fx
            obs[r, 30] = torch.cos(self.clock[r])
            obs[r, 31] = torch.sin(self.clock[r])
            obs[r, 32] = self.command[r, 0]
            obs[r, 33] = self.command[r, 1]
        return obs, rew, done


# Quick standalone check: random policy, verify shapes/finiteness + that resets fire.
if __name__ == "__main__":
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    env = HexapodGpuVecEnv(num_envs=256)
    obs = env.reset()
    print("obs", tuple(obs.shape), "finite", bool(torch.isfinite(obs).all()))
    tot_done = 0
    for i in range(200):
        a = torch.rand(env.K, ACT_DIM, device=env.device) * 2 - 1
        obs, rew, done = env.step(a)
        tot_done += int(done.sum())
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    print(f"200 steps ok; obs finite, reward range [{rew.min():.2f},{rew.max():.2f}]; "
          f"resets fired={tot_done}")
    print("GPU ENV SELFTEST: PASS")
