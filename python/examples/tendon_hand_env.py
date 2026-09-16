"""GPU-vectorized palm-up hold for the tendon hand: 25 tensions, one object, and a pull.

THE TASK IS THE SIMPLEST THING THAT READS AS "IT GRASPS". An object of one of three shapes
drops into the palm; from 1.5 s a force pulls it out in a direction that changes every half
second and grows to 8 N; the policy commands 25 cable tensions and the reward is keeping the
object near the palm. Nothing about the task is clever, because the mechanism is the point:
the tensions do not become joint torques through a table, they become forces at via points and
the moment arms emerge from where the pulleys sit, exactly as in the CPU hand.

WHAT MAKES THIS ENV UNUSUAL, and the whole reason threepp.rl.cable exists: there is no joint
actuator anywhere. `control` is never used, `act()` is never called, and `simulate()` is
overridden instead, because the actuation is 25 routed cables that have to be RE-RESOLVED
EVERY PHYSICS SUBSTEP -- a cable evaluated once per control step would apply a 60 Hz-stale
direction through four 240 Hz substeps, which is precisely the mistake TendonCable's
pre-substep callback exists to avoid on the CPU path.

PALM UP WITHOUT TOUCHING THE HAND. The hand is authored with its pad along -Y (FINGER_PAD),
and it is a fixed-base articulation, so there is no orientation to set: gravity is turned
around instead, to 9.81 * (-FINGER_PAD) = +Y, and objects then fall onto the volar face.
Every direction in this file is written against `G_HAT` rather than against Y, so "the object
fell out of the hand" means "it travelled with gravity past the palm" and stays true if the
convention ever moves.

TWO BATCHES, ONE SCENE. A PhysxGpuBatch holds one robot type, so the K hands are one batch and
the K single-link free-base objects are a second one over the same world. Only the HAND batch
is stepped: `step` is `world.simulateRaw`, so it advances the whole scene, and stepping both
would advance the world twice per substep.
"""
import argparse
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))
sys.path.insert(0, _HERE)

import threepp as tp
from tendon_hand import DENSITY, FINGER_PAD, Hand, skin
from threepp.rl import BatchedCables, CableRouting, VecTask, link_index_map, quat_rotate

# ---- single source of truth: the train/deploy contract ---------------------------
CONTROL_HZ = 60
SUBSTEPS = 4                  # physics at 240 Hz, the CPU demo's DT
EPISODE_S = 6.0
T_MAX = 40.0                  # action [-1,1] -> cable tension (N)
TAU_FILTER = 0.15             # first-order tension filter (s); a step in tension is an impulse
PULL_START_S = 1.5
PULL_MAX = 8.0
PULL_RESAMPLE_S = 0.5
NEAR_SIGMA = 0.03
DROP_DIST = 0.12
DROP_FALL = 0.05
SPACING = 0.5
OBJ_DENSITY = 600.0
ARC_POINTS = 8                # measured in tendon_hand_gpu_check.py gate A1a

G_HAT = np.array([-v for v in FINGER_PAD], dtype=float)     # gravity direction, palm up
GRAVITY = tuple(9.81 * G_HAT)

# Where a held object sits, in the hand's base frame. MEASURED, not guessed: `--sanity` drops
# an object into a fully slack hand and prints the offset it comes to rest at, and this is the
# first guess plus that offset. It landed on x = 0.0720, which is where tendon_hand.py's own
# grasp test places its object after measuring where the CLOSED hand forms a cavity -- the two
# were arrived at independently and agree to the millimetre.
PALM_TARGET = (0.0720, -0.0352, -0.0016)

# Object types: (name, code). Sizes are drawn once per env at build and never re-rolled, so an
# env is always the same object and only its pose is randomized on reset.
OBJ_TYPES = ("sphere", "capsule", "box")
OBS_DIM = 20 + 20 + 25 + 3 + 3 + 3 + 3 + 3
ACT_DIM = 25

CONFIG = {"control_hz": CONTROL_HZ, "substeps": SUBSTEPS, "episode_s": EPISODE_S,
          "t_max": T_MAX, "tau_filter": TAU_FILTER, "pull_start_s": PULL_START_S,
          "pull_max": PULL_MAX, "pull_resample_s": PULL_RESAMPLE_S,
          "near_sigma": NEAR_SIGMA, "drop_dist": DROP_DIST, "drop_fall": DROP_FALL,
          "palm_target": PALM_TARGET, "gravity": GRAVITY, "obj_density": OBJ_DENSITY,
          "arc_points": ARC_POINTS, "obs_dim": OBS_DIM, "act_dim": ACT_DIM,
          "vel_scale": 0.1, "obj_types": OBJ_TYPES}


def object_mesh(kind, a, b):
    """The three shapes, each as the analytic collider Articulation.add_link infers."""
    if kind == "sphere":
        return tp.Mesh(tp.SphereGeometry(a, 20, 14), skin(0x3388CC, 0.5, 0.9))
    if kind == "capsule":
        return tp.Mesh(tp.CapsuleGeometry(a, max(1e-3, b - 2 * a)), skin(0x33AA66, 0.5, 0.9))
    return tp.Mesh(tp.BoxGeometry(a, a, a), skin(0xCC8833, 0.5, 0.9))


class TendonHandEnv(VecTask):
    control_hz = CONTROL_HZ
    substeps = SUBSTEPS
    episode_s = EPISODE_S
    act_dim = ACT_DIM
    clip_actions = 1.0

    def __init__(self, num_envs=1024, device="cuda", seed=0, arc_points=ARC_POINTS):
        self._rest, self._obj_arts, self._obj_spec = {}, [], []
        self._rng = np.random.default_rng(1234 + seed)
        self._arc = arc_points
        super().__init__(num_envs, self._build, gravity=GRAVITY, device=device, seed=seed,
                         read_links=True)
        K, dev = self.K, self.device

        # ---- the cables -------------------------------------------------------------
        # Match the link order at the REST pose, not at whatever the batch's warm-up step
        # left behind: gravity is on here and nothing holds the distal phalanges, so one
        # warm-up frame already drops a fingertip 2.3 mm and the rest-position match has to
        # be made against a hand that has been put back where it was built.
        zero = torch.zeros(K, self.sim.dof, device=dev)
        self.sim.set_joint_state(torch.arange(K, device=dev), zero, zero)
        self.sim.batch.step(1e-5)
        self.sim.read()
        idx_map, self.map_resid = link_index_map(self._rest, self.sim.link_pose[0].cpu().numpy())
        self.idx_map = idx_map
        self.cables = BatchedCables(CableRouting.from_hand(self.sim.robots[0]), idx_map,
                                    self.sim.max_links, device=dev, arc_points=self._arc)
        self.cable_names = list(self.cables.names)
        self.dof_names = list(self.sim.robots[0].dof_names)
        self._lp = torch.zeros(K, self.sim.max_links * 7, device=dev)
        self.link_pose = self._lp.view(K, self.sim.max_links, 7)

        # ---- the objects, as a second batch over the same world ---------------------
        self.obj_batch = tp.PhysxGpuBatch(self.sim.world, self._obj_arts)
        self._obj_gpu = torch.from_numpy(
            self.obj_batch.gpu_indices().astype(np.int32)).to(dev)
        self.obj_pose = torch.zeros(K, 7, device=dev)
        self.obj_linvel = torch.zeros(K, 3, device=dev)
        self.obj_angvel = torch.zeros(K, 3, device=dev)
        # max_links is the SCENE's, not this batch's: the object articulations have one link
        # each but the buffer PhysX wants is still [K, scene max_links, 3]. Only row 0 is ever
        # non-zero, which is the object's only link.
        self._obj_force = torch.zeros(K, self.obj_batch.max_links, 3, device=dev)
        spec = np.array([[OBJ_TYPES.index(k), a, b] for k, a, b in self._obj_spec])
        self.obj_desc = torch.as_tensor(
            np.stack([spec[:, 1] * 20.0, spec[:, 2] * 20.0, spec[:, 0] / 2.0], 1),
            dtype=torch.float32, device=dev)
        self.obj_radius = torch.as_tensor(spec[:, 1], dtype=torch.float32, device=dev)

        base = torch.zeros(K, 3, device=dev)
        base[:, 2] = torch.arange(K, device=dev, dtype=torch.float32) * SPACING
        self.target = base + torch.tensor(PALM_TARGET, device=dev)
        self.g_hat = torch.tensor(G_HAT, dtype=torch.float32, device=dev)

        # ---- per-env state the base re-initializes on every reset --------------------
        self.tension = self.env_state((ACT_DIM,))       # the FILTERED command, in newtons
        self.pull_dir = self.env_state((3,))
        self.alpha = self.dt / (TAU_FILTER + self.dt)
        self.hold_hits = self.hold_n = 0
        self.pull_survived = 0.0
        self.read_objects()

    # ---- construction ---------------------------------------------------------------
    def _build(self, world, i):
        if not hasattr(self, "_pad"):
            # rubber on plastic; 'min' so the softer pair member governs, as in the CPU demo
            self._pad = world.create_material(1.2, 1.1, 0.0, friction_combine="min")
        hand = Hand(world, base=(0.0, 0.0, i * SPACING), material=self._pad).finalize()
        hand.route(build=False)              # record the routing; a CPU cable would be inert here
        if i == 0:
            self._rest.update({n: (m.position.x, m.position.y, m.position.z)
                               for n, m in zip(hand.links, hand.meshes)})

        kind = OBJ_TYPES[i % len(OBJ_TYPES)]
        if kind == "sphere":
            a, b = self._rng.uniform(0.018, 0.025), 0.0
        elif kind == "capsule":
            a, b = self._rng.uniform(0.015, 0.020), self._rng.uniform(0.060, 0.080)
        else:
            a, b = self._rng.uniform(0.030, 0.045), 0.0
        self._obj_spec.append((kind, a, b))
        mesh = object_mesh(kind, a, b)
        p = np.array(PALM_TARGET) + np.array([0.0, 0.0, i * SPACING]) - G_HAT * 0.045
        mesh.position.set(*p)
        art = world.create_articulation(fixed_base=False)
        art.add_link(mesh, density=OBJ_DENSITY, material=self._pad)
        art.finalize()
        self._obj_arts.append(art)
        return hand

    # ---- state ----------------------------------------------------------------------
    def read_objects(self):
        self.obj_batch.read_root_pose(self.obj_pose)
        self.obj_batch.read_root_linvel(self.obj_linvel)
        self.obj_batch.read_root_angvel(self.obj_angvel)

    @property
    def obj_pos(self):
        return self.obj_pose[:, 4:7]

    def offset(self):
        """Object position relative to the palm target, in the hand's (axis-aligned) frame."""
        return self.obj_pos - self.target

    # ---- the control step -----------------------------------------------------------
    def simulate(self, a):
        """Filter the action into tensions, then run 4 substeps, re-resolving every cable at
        each one. No joint force is ever applied: the hand has no drives except the collateral
        ligament springs, which are build-time drive stiffness at target 0 and carry over."""
        cmd = T_MAX * 0.5 * (a + 1.0)
        self.tension.mul_(1.0 - self.alpha).add_(cmd, alpha=self.alpha)
        pull = self.pull_force()
        self._obj_force[:, 0] = pull
        dt = self.dt / self.substeps
        for _ in range(self.substeps):
            self.sim.batch.read_link_pose(self._lp)
            f, tq, _ = self.cables.apply(self.link_pose, self.tension)
            torch.cuda.synchronize()          # PhysX consumes the pointer on its own stream
            self.sim.batch.write_link_force(f.contiguous())
            self.sim.batch.write_link_torque(tq.contiguous())
            self.obj_batch.write_link_force(self._obj_force)
            self.sim.batch.step(dt)           # == world.simulateRaw: advances the whole scene
        self.sim.read()
        self.read_objects()

    def pull_force(self):
        """The disturbance the grasp has to survive: a force in a direction resampled every
        0.5 s, ramping linearly from nothing at 1.5 s to PULL_MAX at the end of the episode."""
        t = self.steps.float() * self.dt
        k = ((t - PULL_START_S) / (EPISODE_S - PULL_START_S)).clamp(0.0, 1.0)
        if self.settling:
            k = torch.zeros_like(k)
        return self.pull_dir * (PULL_MAX * k).unsqueeze(-1)

    def on_step(self, s):
        every = max(1, int(round(PULL_RESAMPLE_S * self.control_hz)))
        due = (self.steps % every) == 0
        if bool(due.any()):
            d = torch.randn(self.K, 3, device=self.device, generator=self.g)
            d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            self.pull_dir[due] = d[due]

    # ---- the task -------------------------------------------------------------------
    def on_reset(self, idx):
        n = idx.numel()
        z = torch.zeros(n, self.sim.dof, device=self.device)
        self.sim.set_joint_state(idx, z, z)

        r = lambda *s: torch.rand(*s, device=self.device, generator=self.g)
        up = (0.03 + 0.03 * r(n)).unsqueeze(-1)                 # 3-6 cm above the palm
        jitter = torch.zeros(n, 3, device=self.device)
        jitter[:, 0] = (r(n) * 2 - 1) * 0.015
        jitter[:, 2] = (r(n) * 2 - 1) * 0.015
        pos = self.target[idx] + jitter - self.g_hat * up
        q = torch.randn(n, 4, device=self.device, generator=self.g)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        pose = torch.cat([q, pos], dim=-1).contiguous()
        zero = torch.zeros(n, 3, device=self.device)
        sub = self._obj_gpu[idx].contiguous()
        torch.cuda.synchronize()
        self.obj_batch.write_subset_root_pose(pose, sub)
        self.obj_batch.write_subset_root_linvel(zero, sub)
        self.obj_batch.write_subset_root_angvel(zero, sub)
        self.read_objects()

    def observe(self, s):
        d = self.offset()
        return torch.cat([s.joint_pos, s.joint_vel * 0.1, self.tension / T_MAX,
                          d, self.obj_linvel, self.obj_angvel * 0.1,
                          self.obj_desc, self.pull_force() / 10.0], dim=-1)

    def terminated(self, s):
        d = self.offset()
        fallen = (d * self.g_hat).sum(-1) > DROP_FALL      # travelled WITH gravity past the palm
        return (d.norm(dim=-1) > DROP_DIST) | fallen

    def reward_terms(self, s, a):
        d = self.offset().norm(dim=-1)
        return {
            "near": torch.exp(-(d / NEAR_SIGMA) ** 2),
            "still": -0.05 * self.obj_linvel.norm(dim=-1),
            "effort": -0.002 * ((self.tension / T_MAX) ** 2).mean(-1),
            "drop": -2.0 * s.terminated.float(),
        }

    def on_done(self, idx):
        """Hold rate and the pull actually survived -- the two numbers that say whether the
        policy is learning to grasp rather than to sit still while nothing pulls."""
        held = (~self.state.terminated[idx]).float()
        self.hold_hits += float(held.sum().item())
        self.hold_n += int(idx.numel())
        t = self.steps[idx].float() * self.dt
        k = ((t - PULL_START_S) / (EPISODE_S - PULL_START_S)).clamp(0.0, 1.0)
        self.pull_survived += float((k * PULL_MAX).sum().item())

    def hold_stats(self, reset=True):
        if self.hold_n == 0:
            return None
        out = (self.hold_hits / self.hold_n, self.pull_survived / self.hold_n, self.hold_n)
        if reset:
            self.hold_hits = self.pull_survived = 0.0
            self.hold_n = 0
        return out

    def stats_line(self):
        base = super().stats_line()
        h = self.hold_stats()
        if h is None:
            return base
        return f"{base}  hold {h[0]*100:5.1f}%  pull {h[1]:4.2f}N  n {h[2]}"

    def config(self):
        return {**super().config(), **CONFIG,
                "cable_names": list(self.cable_names), "dof_names": list(self.dof_names)}


# ================================================================================
# B1: the sanity gate. Printed numbers, no test framework.
# ================================================================================

def sanity(K=1024, steps=600, device="cuda"):
    import time
    env = TendonHandEnv(num_envs=K, device=device)
    print(f"env: K={K}  obs {OBS_DIM}  act {ACT_DIM}  {env.cables.C} cables x "
          f"{env.cables.N} slots  link map residual {env.map_resid*1000:.4f} mm")
    obs = env.reset()
    assert obs.shape == (K, OBS_DIM), f"obs is {tuple(obs.shape)}, expected {(K, OBS_DIM)}"

    # ---- where does a SLACK hand let the object rest? ------------------------------
    # This is the measurement PALM_TARGET is set from, and the check that gravity really is
    # palm-up: with every cable at zero the object must come to rest ON the palm.
    zero = torch.full((K, ACT_DIM), -1.0, device=env.device)      # tension 0
    for _ in range(90):                                           # 1.5 s, before the pull
        env.step(zero)
    d = env.offset()
    rest = env.obj_pos - env.target
    print(f"\n[B1a] slack hand, 1.5 s, no pull: object rest offset from PALM_TARGET "
          f"(m, hand frame)")
    print(f"      mean ({rest[:,0].mean():+.4f}, {rest[:,1].mean():+.4f}, "
          f"{rest[:,2].mean():+.4f})   |d| mean {d.norm(dim=-1).mean():.4f} "
          f"max {d.norm(dim=-1).max():.4f}")
    v = float(env.obj_linvel.norm(dim=-1).mean().item())
    print(f"      mean speed {v:.4f} m/s -- "
          f"{'resting on the palm' if v < 0.05 else 'STILL MOVING, not settled'}")

    # ---- slack, with the pull: the task must not be trivial --------------------------
    env.reset()
    t0 = time.perf_counter()
    nan = False
    for i in range(steps):
        obs, rew, done, term, to = env.step(zero)
        nan |= not bool(torch.isfinite(obs).all() and torch.isfinite(rew).all())
    torch.cuda.synchronize()
    rate = K * steps / (time.perf_counter() - t0)
    h = env.hold_stats()
    print(f"\n[B1b] {steps} steps at K={K}, every cable slack, pulls on:")
    print(f"      NaN anywhere: {nan}")
    print(f"      eager rate {rate:,.0f} env-steps/s")
    print(f"      hold rate {h[0]*100:.1f} % over {h[2]} episodes  (mean pull "
          f"{h[1]:.2f} N) -- a slack hand must NOT hold, or the task is trivial")
    print(f"      per-term: {env.stats_line()}")

    # ---- scripted fist: does the thumb oppose? ---------------------------------------
    flex = ["index_fdp", "index_fds", "middle_fdp", "middle_fds", "ring_fdp", "ring_fds",
            "little_fdp", "little_fds", "thumb_fpl", "thumb_fpb", "thumb_add"]
    a = torch.full((K, ACT_DIM), -1.0, device=env.device)
    for c in flex:
        a[:, env.cable_names.index(c)] = 2.0 * 35.0 / T_MAX - 1.0      # 35 N
    env.reset()
    ti, ii = env.idx_map["thumb_ip"], env.idx_map["index_dip"]
    # The TIPS, not the link origins: a phalanx capsule runs along its own local +Y from its
    # centre, so the tip is half its total length out along that axis. Opposition measured
    # between origins would read 20-odd mm of phalanx as separation that is not there.
    tip_t = torch.tensor([0.0, 0.5 * (0.025 + 0.007), 0.0], device=env.device)
    tip_i = torch.tensor([0.0, 0.5 * (0.017 + 0.007), 0.0], device=env.device)
    print(f"\n[B1c] scripted fist, 35 N on every flexor: thumb tip to index tip (mm)")
    best = 1e9
    for i in range(1, 121):                     # 2 s
        env.step(a)
        if i % 15 == 0:
            env.sim.batch.read_link_pose(env._lp)
            pt = (env.link_pose[:, ti, 4:7]
                  + quat_rotate(env.link_pose[:, ti, 0:4], tip_t.expand(env.K, 3)))
            pi = (env.link_pose[:, ii, 4:7]
                  + quat_rotate(env.link_pose[:, ii, 0:4], tip_i.expand(env.K, 3)))
            sep = (pt - pi).norm(dim=-1)
            best = min(best, float(sep.mean().item()) * 1000)
            print(f"      t = {i/60:.2f} s   mean {sep.mean()*1000:6.2f}   "
                  f"min {sep.min()*1000:6.2f}   max {sep.max()*1000:6.2f}")
    print(f"      closest mean separation {best:.2f} mm   "
          f"{'OPPOSES (< 30 mm)' if best < 30.0 else 'DOES NOT OPPOSE (>= 30 mm)'}")
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true", help="gate B1: sanity, opposition")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=600)
    a = ap.parse_args()
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need a PhysX build + CUDA"); sys.exit(0)
    sanity(K=a.envs, steps=a.steps)


if __name__ == "__main__":
    main()
