"""GPU-vectorized hold for the tendon hand: 25 tensions, a randomized object, a roll-over, a pull.

THE EPISODE. An object drops into the palm-up hand. Between 1.0 and 1.8 s gravity starts to
turn about an axis in the palm plane, through 120 to 180 degrees over 1.0 to 1.8 s, so the palm
ends up facing down and the object hangs from the fingers; it stays there for 1.5 to 2.6 s and
turns back. From 1.5 s a pull in a direction resampled every 0.5 s ramps to 8 N by 5.0 s and
stays there to the end at 8 s. The policy commands 25 cable tensions and the reward is keeping
the object near the palm. Every timing above is drawn per env at reset.

WHY GRAVITY TURNS AND THE HAND DOES NOT. The hand is a fixed-base articulation authored with its
pad along -Y (FINGER_PAD), and the K hands share ONE PhysX scene: there is no per-env wrist to
rotate and no per-env scene gravity to set. Scene gravity is therefore zero, and gravity is
applied per env as an explicit force m*g at every link's centre of mass -- the hand's 20 moving
links through the same write_link_force the cables go through, the object through its own batch.
PhysX's own gravity is that same force through the centre of mass, so the plant is the CPU hand
under world.set_gravity, which is what play_tendon_hand.py runs the trained policy against. The
link masses are ArticulationLink.mass, the value PhysX computed from collider and density.
CONFIG["gravity"] stays the palm-up baseline 9.81 * (-FINGER_PAD) = +Y that the deploy side
builds its world with. Every direction in this file is written against G_HAT or the current
per-env gravity rather than against Y, so "the object fell out of the hand" means "it travelled
with gravity past the palm" in whichever way the hand is turned.

OBJECT RANDOMIZATION is per env and fixed for the env's life, because a collider cannot be resized
or re-massed after add_link: five shapes round-robin (sphere, capsule, box, bar, cylinder),
dimensions drawn from OBJ_DIMS, density log-uniform over OBJ_DENSITY (about 7 g to 270 g) and an
object material with friction drawn from OBJ_FRICTION under 'min' combine, so the pair friction
is the object's. Shape and dimensions are observed; mass and friction are not. Only the pose is
re-randomized on reset.

WHAT MAKES THIS ENV UNUSUAL, and the whole reason threepp.rl.cable exists: there is no joint
actuator anywhere. `control` is never used, `act()` is never called, and `simulate()` is
overridden instead, because the actuation is 25 routed cables that have to be RE-RESOLVED
EVERY PHYSICS SUBSTEP -- a cable evaluated once per control step would apply a 60 Hz-stale
direction through four 240 Hz substeps, which is precisely the mistake TendonCable's
pre-substep callback exists to avoid on the CPU path.

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
from tendon_hand import FINGER_BONE, FINGER_PAD, Hand, skin
from threepp.rl import BatchedCables, CableRouting, VecTask, link_index_map, quat_rotate

# ---- single source of truth: the train/deploy contract ---------------------------
CONTROL_HZ = 60
SUBSTEPS = 4                  # physics at 240 Hz, the CPU demo's DT
EPISODE_S = 8.0
T_MAX = 40.0                  # action [-1,1] -> cable tension (N)
TAU_FILTER = 0.15             # first-order tension filter (s); a step in tension is an impulse
PULL_START_S = 1.5
PULL_FULL_S = 5.0             # the pull reaches PULL_MAX here and stays there
PULL_MAX = 8.0
PULL_RESAMPLE_S = 0.5
NEAR_SIGMA = 0.03
DROP_DIST = 0.12
DROP_FALL = 0.05
SPACING = 0.5
ARC_POINTS = 8                # measured in tendon_hand_gpu_check.py gate A1a
G = 9.81

# The roll. Per-env draws, each uniform over its range; t0 + 2*dur + hang never exceeds
# EPISODE_S. ROLL_FILM is the one schedule play_tendon_hand.py runs by default: a pronation
# about the finger axis through a full half turn.
ROLL_T0 = (1.0, 1.8)          # s, when gravity starts to turn
ROLL_DUR = (1.0, 1.8)         # s, for the turn (and again for the turn back)
ROLL_HANG = (1.5, 2.6)        # s, held turned over
ROLL_ANG_DEG = (120.0, 180.0)
ROLL_FILM = {"t0": 1.5, "dur": 1.5, "hang": 2.5, "ang_deg": 180.0,
             "axis": tuple(float(v) for v in FINGER_BONE)}

# The objects. Dimensions per type in metres: sphere (r), capsule (r, total length), box (side),
# bar (x, y, z), cylinder (r, height). Density and friction are drawn per env too.
OBJ_TYPES = ("sphere", "capsule", "box", "bar", "cylinder")
OBJ_DIMS = {
    "sphere":   ((0.015, 0.030),),
    "capsule":  ((0.012, 0.022), (0.050, 0.090)),
    "box":      ((0.025, 0.050),),
    "bar":      ((0.020, 0.040), (0.020, 0.040), (0.050, 0.090)),
    "cylinder": ((0.012, 0.025), (0.040, 0.090)),
}
OBJ_DENSITY = (300.0, 3000.0)     # kg/m^3, log-uniform
OBJ_FRICTION = (0.3, 1.2)         # the object material; the pad is 1.2/1.1, combine 'min'

G_HAT = np.array([-v for v in FINGER_PAD], dtype=float)     # baseline gravity direction, palm up
# Plain Python floats, not numpy scalars: CONFIG is persisted into the policy checkpoint and
# load_policy reads it with weights_only=True, which refuses numpy._core.multiarray.scalar.
# A tuple() over a numpy array keeps the numpy element type, so the checkpoint loads in
# training and then cannot be opened on the deploy side at all.
GRAVITY = tuple(float(v) for v in G * G_HAT)

# Where a held object sits, in the hand's base frame. MEASURED, not guessed: `--sanity` drops
# an object into a fully slack hand and prints the offset it comes to rest at, and this is the
# first guess plus that offset. It landed on x = 0.0720, which is where tendon_hand.py's own
# grasp test places its object after measuring where the CLOSED hand forms a cavity -- the two
# were arrived at independently and agree to the millimetre.
PALM_TARGET = (0.0720, -0.0352, -0.0016)

DESC_DIM = 4                  # three dimensions * 20 and the type code
OBS_DIM = 20 + 20 + 25 + 3 + 3 + 3 + DESC_DIM + 3 + 3
ACT_DIM = 25

CONFIG = {"control_hz": CONTROL_HZ, "substeps": SUBSTEPS, "episode_s": EPISODE_S,
          "t_max": T_MAX, "tau_filter": TAU_FILTER, "pull_start_s": PULL_START_S,
          "pull_full_s": PULL_FULL_S, "pull_max": PULL_MAX, "pull_resample_s": PULL_RESAMPLE_S,
          "near_sigma": NEAR_SIGMA, "drop_dist": DROP_DIST, "drop_fall": DROP_FALL,
          "palm_target": PALM_TARGET, "gravity": GRAVITY,
          "roll": {"t0": ROLL_T0, "dur": ROLL_DUR, "hang": ROLL_HANG, "ang_deg": ROLL_ANG_DEG},
          "roll_film": ROLL_FILM,
          "obj_types": OBJ_TYPES, "obj_dims": OBJ_DIMS, "obj_density": OBJ_DENSITY,
          "obj_friction": OBJ_FRICTION,
          "arc_points": ARC_POINTS, "obs_dim": OBS_DIM, "act_dim": ACT_DIM, "vel_scale": 0.1}


# ---- shared with play_tendon_hand.py: the object draw and the roll --------------------
def sample_object(kind, rng, density=None, friction=None):
    """One object's fixed parameters: dims padded to three (m), density (kg/m^3), friction.
    `density` / `friction` override the draw (the deploy side sets them from the command line)."""
    dims = [float(rng.uniform(lo, hi)) for lo, hi in OBJ_DIMS[kind]]
    dims += [0.0] * (3 - len(dims))
    lo, hi = OBJ_DENSITY
    rho = float(math.exp(rng.uniform(math.log(lo), math.log(hi)))) if density is None else float(density)
    mu = float(rng.uniform(*OBJ_FRICTION)) if friction is None else float(friction)
    return tuple(dims), rho, mu


def object_mesh(kind, dims):
    """The five shapes. Sphere, capsule and box are the analytic colliders Articulation.add_link
    infers; the cylinder is cooked to a convex hull of its vertices by the same call."""
    a, b, c = dims
    if kind == "sphere":
        return tp.Mesh(tp.SphereGeometry(a, 20, 14), skin(0x3388CC, 0.5, 0.9))
    if kind == "capsule":
        return tp.Mesh(tp.CapsuleGeometry(a, max(1e-3, b - 2 * a)), skin(0x33AA66, 0.5, 0.9))
    if kind == "box":
        return tp.Mesh(tp.BoxGeometry(a, a, a), skin(0xCC8833, 0.5, 0.9))
    if kind == "bar":
        return tp.Mesh(tp.BoxGeometry(a, b, c), skin(0x8855CC, 0.5, 0.9))
    if kind == "cylinder":
        return tp.Mesh(tp.CylinderGeometry(a, a, b, 24), skin(0xCC4455, 0.5, 0.9))
    raise ValueError(f"unknown object type {kind!r}")


def object_desc(kind, dims):
    """The observed part of an object: its dimensions (x20) and its type code."""
    return [d * 20.0 for d in dims] + [OBJ_TYPES.index(kind) / (len(OBJ_TYPES) - 1)]


def roll_theta(t, t0, dur, hang, ang):
    """Roll angle (rad) at time t for the schedule (t0, dur, hang, ang): zero before t0, a
    smoothstep up to `ang` over `dur`, held for `hang`, a smoothstep back down over `dur`, zero
    after. Torch, broadcasting over per-env tensors; the deploy side calls it with 0-d ones."""
    def s(x):
        x = x.clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)
    return ang * (s((t - t0) / dur) - s((t - (t0 + dur + hang)) / dur))


def gravity_at(theta, axis, g0):
    """g0 [3] rotated by theta [K] about the unit axis [K,3] (Rodrigues) -> [K,3]."""
    c, s = torch.cos(theta).unsqueeze(-1), torch.sin(theta).unsqueeze(-1)
    g0 = g0.expand_as(axis)
    return (g0 * c + torch.linalg.cross(axis, g0, dim=-1) * s
            + axis * (axis * g0).sum(-1, keepdim=True) * (1.0 - c))


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
        # Scene gravity ZERO: gravity is a per-env force, see the module docstring.
        super().__init__(num_envs, self._build, gravity=(0.0, 0.0, 0.0), device=device,
                         seed=seed, read_links=True)
        K, dev = self.K, self.device

        # ---- the cables -------------------------------------------------------------
        # Match the link order at the REST pose, not at whatever the batch's warm-up step
        # left behind: the rest-position match has to be made against a hand that has been
        # put back where it was built.
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

        # The hand's link masses by GPU link slot, for the per-env gravity force. The root is
        # the fixed palm; a force on it does nothing, so its slot stays zero.
        mass = torch.zeros(self.sim.max_links, device=dev)
        for name, lk in self.sim.robots[0].links.items():
            if not lk.is_root:
                mass[idx_map[name]] = float(lk.mass)
        self.link_mass = mass

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
        self.obj_mass = torch.as_tensor([float(a.link(0).mass) for a in self._obj_arts],
                                        dtype=torch.float32, device=dev)
        self.obj_desc = torch.as_tensor([object_desc(k, d) for k, d, _, _ in self._obj_spec],
                                        dtype=torch.float32, device=dev)
        self.obj_radius = torch.as_tensor([d[0] for _, d, _, _ in self._obj_spec],
                                          dtype=torch.float32, device=dev)

        base = torch.zeros(K, 3, device=dev)
        base[:, 2] = torch.arange(K, device=dev, dtype=torch.float32) * SPACING
        self.target = base + torch.tensor(PALM_TARGET, device=dev)
        self.g_hat = torch.tensor(G_HAT, dtype=torch.float32, device=dev)
        self.g0 = self.g_hat * G                                   # baseline, palm up
        # An orthonormal basis of the palm plane, for the roll axis draw.
        e1 = np.array(FINGER_BONE, dtype=float)
        e1 -= G_HAT * float(np.dot(e1, G_HAT))
        e1 /= np.linalg.norm(e1)
        self._e1 = torch.tensor(e1, dtype=torch.float32, device=dev)
        self._e2 = torch.tensor(np.cross(G_HAT, e1), dtype=torch.float32, device=dev)

        # ---- per-env state the base re-initializes on every reset --------------------
        self.tension = self.env_state((ACT_DIM,))       # the FILTERED command, in newtons
        self.pull_dir = self.env_state((3,))
        self.roll_t0 = self.env_state(init=ROLL_T0[0])
        self.roll_dur = self.env_state(init=ROLL_DUR[0])
        self.roll_hang = self.env_state(init=ROLL_HANG[0])
        self.roll_ang = self.env_state()
        self.roll_axis = self.env_state((3,))
        self.g_vec = self.env_state((3,))                # the CURRENT gravity (m/s^2)
        self.alpha = self.dt / (TAU_FILTER + self.dt)
        self.pull_scale = 1.0                            # the sanity gate switches these off
        self.roll_scale = 1.0
        self.hold_hits = self.hold_n = 0
        self.inv_hits = self.pre_hits = self.pull_survived = 0.0
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
        dims, rho, mu = sample_object(kind, self._rng)
        self._obj_spec.append((kind, dims, rho, mu))
        mat = world.create_material(mu, 0.9 * mu, 0.0, friction_combine="min")
        mesh = object_mesh(kind, dims)
        p = np.array(PALM_TARGET) + np.array([0.0, 0.0, i * SPACING]) - G_HAT * 0.045
        mesh.position.set(*p)
        art = world.create_articulation(fixed_base=False)
        art.add_link(mesh, density=rho, material=mat)
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

    def update_gravity(self):
        t = self.steps.float() * self.dt
        th = roll_theta(t, self.roll_t0, self.roll_dur, self.roll_hang, self.roll_ang)
        self.g_vec.copy_(gravity_at(th * self.roll_scale, self.roll_axis, self.g0))

    # ---- the control step -----------------------------------------------------------
    def simulate(self, a):
        """Filter the action into tensions, then run 4 substeps, re-resolving every cable at
        each one and adding the per-env gravity force to every link. No joint force is ever
        applied: the hand has no drives except the collateral ligament springs, which are
        build-time drive stiffness at target 0 and carry over."""
        cmd = T_MAX * 0.5 * (a + 1.0)
        self.tension.mul_(1.0 - self.alpha).add_(cmd, alpha=self.alpha)
        self.update_gravity()
        pull = self.pull_force()
        self._obj_force[:, 0] = pull + self.obj_mass.unsqueeze(-1) * self.g_vec
        g_link = self.link_mass.view(1, -1, 1) * self.g_vec.unsqueeze(1)     # [K, L, 3]
        dt = self.dt / self.substeps
        for _ in range(self.substeps):
            self.sim.batch.read_link_pose(self._lp)
            f, tq, _ = self.cables.apply(self.link_pose, self.tension)
            f = f + g_link
            torch.cuda.synchronize()          # PhysX consumes the pointer on its own stream
            self.sim.batch.write_link_force(f.contiguous())
            self.sim.batch.write_link_torque(tq.contiguous())
            self.obj_batch.write_link_force(self._obj_force)
            self.sim.batch.step(dt)           # == world.simulateRaw: advances the whole scene
        self.sim.read()
        self.read_objects()

    def pull_fraction(self, t):
        return ((t - PULL_START_S) / (PULL_FULL_S - PULL_START_S)).clamp(0.0, 1.0)

    def pull_force(self):
        """The disturbance the grasp has to survive: a force in a direction resampled every
        0.5 s, ramping linearly from nothing at 1.5 s to PULL_MAX at 5.0 s, held to the end."""
        k = self.pull_fraction(self.steps.float() * self.dt)
        if self.settling:
            k = torch.zeros_like(k)
        return self.pull_dir * (PULL_MAX * self.pull_scale * k).unsqueeze(-1)

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
        u = lambda lo, hi: lo + (hi - lo) * r(n)
        self.roll_t0[idx] = u(*ROLL_T0)
        self.roll_dur[idx] = u(*ROLL_DUR)
        self.roll_hang[idx] = u(*ROLL_HANG)
        self.roll_ang[idx] = u(*ROLL_ANG_DEG) * (math.pi / 180.0)
        phi = (2.0 * math.pi * r(n)).unsqueeze(-1)
        self.roll_axis[idx] = torch.cos(phi) * self._e1 + torch.sin(phi) * self._e2
        self.g_vec[idx] = self.g0                                 # t = 0: palm up

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
                          self.obj_desc, self.pull_force() / 10.0, self.g_vec / G], dim=-1)

    def terminated(self, s):
        d = self.offset()
        # travelled WITH the current gravity past the palm, whichever way the hand is turned
        fallen = (d * self.g_vec).sum(-1) / G > DROP_FALL
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
        """Hold rate, the fraction that was still holding when the roll had turned the hand
        fully over, and the pull reached -- the numbers that say whether the policy is
        learning to grasp rather than to sit still while nothing pulls."""
        held = (~self.state.terminated[idx]).float()
        self.hold_hits += float(held.sum().item())
        self.hold_n += int(idx.numel())
        t = self.steps[idx].float() * self.dt
        self.pull_survived += float((self.pull_fraction(t) * PULL_MAX).sum().item())
        inv = t >= self.roll_t0[idx] + self.roll_dur[idx]
        self.inv_hits += float(inv.float().sum().item())
        self.pre_hits += float((t < self.roll_t0[idx]).float().sum().item())

    def hold_stats(self, reset=True):
        """(held to the end, still held when turned over, mean pull at the end, episodes,
        ended before the roll started) as fractions of the episodes since the last call."""
        if self.hold_n == 0:
            return None
        out = (self.hold_hits / self.hold_n, self.inv_hits / self.hold_n,
               self.pull_survived / self.hold_n, self.hold_n, self.pre_hits / self.hold_n)
        if reset:
            self.hold_hits = self.inv_hits = self.pre_hits = self.pull_survived = 0.0
            self.hold_n = 0
        return out

    def stats_line(self):
        base = super().stats_line()
        h = self.hold_stats()
        if h is None:
            return base
        return (f"{base}  hold {h[0]*100:5.1f}%  inv {h[1]*100:5.1f}%  "
                f"pull {h[2]:4.2f}N  n {h[3]}")

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
    m = env.link_mass[env.link_mass > 0]
    print(f"     hand links {int((env.link_mass > 0).sum())} moving, {m.sum()*1000:.1f} g, "
          f"{m.min()*1000:.2f}-{m.max()*1000:.2f} g each;  objects "
          f"{env.obj_mass.min()*1000:.1f}-{env.obj_mass.max()*1000:.1f} g")
    obs = env.reset()
    assert obs.shape == (K, OBS_DIM), f"obs is {tuple(obs.shape)}, expected {(K, OBS_DIM)}"

    # ---- where does a SLACK hand let the object rest? ------------------------------
    # This is the measurement PALM_TARGET is set from, and the check that the per-env gravity
    # force really is palm-up: with every cable at zero the object must come to rest ON the
    # palm. 0.9 s, which is before the earliest roll starts.
    zero = torch.full((K, ACT_DIM), -1.0, device=env.device)      # tension 0
    for _ in range(54):
        env.step(zero)
    d = env.offset()
    rest = env.obj_pos - env.target
    print(f"\n[B1a] slack hand, 0.9 s, no pull, no roll yet: object rest offset from "
          f"PALM_TARGET (m, hand frame)")
    print(f"      mean ({rest[:,0].mean():+.4f}, {rest[:,1].mean():+.4f}, "
          f"{rest[:,2].mean():+.4f})   |d| mean {d.norm(dim=-1).mean():.4f} "
          f"max {d.norm(dim=-1).max():.4f}")
    v = float(env.obj_linvel.norm(dim=-1).mean().item())
    print(f"      mean speed {v:.4f} m/s -- "
          f"{'resting on the palm' if v < 0.05 else 'STILL MOVING, not settled'}")

    # ---- slack, with the pull and the roll: the task must not be trivial --------------
    env.reset()
    t0 = time.perf_counter()
    nan = False
    for i in range(steps):
        obs, rew, done, term, to = env.step(zero)
        nan |= not bool(torch.isfinite(obs).all() and torch.isfinite(rew).all())
    torch.cuda.synchronize()
    rate = K * steps / (time.perf_counter() - t0)
    h = env.hold_stats()
    print(f"\n[B1b] {steps} steps at K={K}, every cable slack, pull and roll on:")
    print(f"      NaN anywhere: {nan}")
    print(f"      eager rate {rate:,.0f} env-steps/s")
    print(f"      hold rate {h[0]*100:.1f} % over {h[3]} episodes, {h[1]*100:.1f} % still "
          f"held when turned over (mean pull {h[2]:.2f} N) -- a slack hand must NOT hold, "
          f"or the task is trivial")
    print(f"      per-term: {env.stats_line()}")

    # ---- scripted fist: does the thumb oppose? ---------------------------------------
    flex = ["index_fdp", "index_fds", "middle_fdp", "middle_fds", "ring_fdp", "ring_fds",
            "little_fdp", "little_fds", "thumb_fpl", "thumb_fpb", "thumb_add"]
    a = torch.full((K, ACT_DIM), -1.0, device=env.device)
    for c in flex:
        a[:, env.cable_names.index(c)] = 2.0 * 35.0 / T_MAX - 1.0      # 35 N
    env.pull_scale, env.roll_scale = 0.0, 0.0
    env.reset()
    ti, ii = env.idx_map["thumb_ip"], env.idx_map["index_dip"]
    # The TIPS, not the link origins: a phalanx capsule runs along its own local +Y from its
    # centre, so the tip is half its total length out along that axis. Opposition measured
    # between origins would read 20-odd mm of phalanx as separation that is not there.
    tip_t = torch.tensor([0.0, 0.5 * (0.025 + 0.007), 0.0], device=env.device)
    tip_i = torch.tensor([0.0, 0.5 * (0.017 + 0.007), 0.0], device=env.device)
    print(f"\n[B1c] scripted fist, 35 N on every flexor, no pull, no roll: thumb tip to "
          f"index tip (mm)")
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

    # ---- a static grip through the roll, no pull: is the turn-over survivable at all? --
    # Not the 35 N fist: closed from the start it slams shut on the landing object and loses
    # 97 % of them before the roll begins (measured), which says nothing about the roll. A 15 N
    # grip closed 0.6 s after the drop holds whatever is still on the palm by then, and what
    # that grip keeps through the turn and the turn back is the number wanted here.
    env.pull_scale, env.roll_scale = 0.0, 1.0
    grip = torch.full((K, ACT_DIM), -1.0, device=env.device)
    for c in flex:
        grip[:, env.cable_names.index(c)] = 2.0 * 15.0 / T_MAX - 1.0      # 15 N
    env.reset()
    for _ in range(env.max_steps + 1):
        closed = (env.steps.float() * env.dt > 0.6).unsqueeze(-1)
        env.step(torch.where(closed, grip, zero))
    h = env.hold_stats()
    holding = max(1e-9, 1.0 - h[4])
    print(f"\n[B1d] a fixed 15 N grip closed 0.6 s after the drop, roll on, no pull, one full "
          f"episode at K={K}:")
    print(f"      lost before the roll started {h[4]*100:.1f} %   still held when turned "
          f"over {h[1]*100:.1f} %   held to the end {h[0]*100:.1f} %   over {h[3]} episodes")
    print(f"      of the grips still holding when the roll began: {h[1]/holding*100:.0f} % "
          f"held through the turn, {h[0]/holding*100:.0f} % to the end -- a static grip that "
          f"keeps SOME of them says the roll is survivable; one that keeps all of them says "
          f"the roll is not the hard part")
    env.pull_scale = 1.0
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true", help="gate B1: sanity, opposition, roll")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=600)
    a = ap.parse_args()
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need a PhysX build + CUDA"); sys.exit(0)
    sanity(K=a.envs, steps=a.steps)


if __name__ == "__main__":
    main()
