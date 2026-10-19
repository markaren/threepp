"""kuka_grasp_pool.py — the reset pool: collision-checked start configurations in three families.

The recipe of Zhang et al., "A Balanced Data Diet" (CoRL 2026), for manipulation: episodes
start from a pre-built pool balanced across start families, and each configuration is a cell
of the success-guided sampler (threepp.rl.sgs). Here:

    Reaching     cube resting anywhere in the table region, fingers open; the arm either near
                 the DEFAULT_Q home (sub 0) or IK-aligned above the cube, jaws square to it,
                 4-15 cm above the grasp height (sub 1). No grasp.
    StableGrasp  cube on the table between closed fingers, squeezed.
    NearGoal     cube held between closed fingers in the air, lifted 1 cm up to the lift
                 target (LIFT_SUCCESS_DZ) above its rest height.

Every candidate is placed in the simulator and settled under a holding command; what is
recorded is the settled state (the PD arm sags under gravity, so IK postures are not states).
The arm is gravity-compensated (kuka_grasp_sim.py), so the settled
states sit within a millimetre of the IK postures. A verification pass then resets every recorded configuration from whatever state the env was
in before (as training resets do), settles it again under the same holding command and drops
it if a Reaching cube moved, or a held cube is no longer between the fingers. A separate
Reaching set with another seed is the fixed evaluation.

    python kuka_grasp_pool.py              # build kuka_grasp_pool.npz next to this file
    python kuka_grasp_pool.py --check      # print the pool's composition
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import kuka_grasp_contract as C  # noqa: E402

FAMILIES = ("reaching", "stable_grasp", "near_goal")
REACHING, STABLE_GRASP, NEAR_GOAL = 0, 1, 2
POOL_PATH = os.path.join(_HERE, "kuka_grasp_pool.npz")

# cube region on the table (the scaffold's final spawn region), env-local
REGION_X = (C.TABLE_CX - 0.16, C.TABLE_CX + 0.16)
REGION_Y = (C.TABLE_CY - 0.24, C.TABLE_CY + 0.24)
HOME_NOISE = 0.10             # rad, uniform per joint, Reaching sub 0
ALIGN_DZ = (0.04, 0.15)       # grasp point above GRASP_TABLE_Z, Reaching sub 1
LIFT_CMD_DZ = (0.01, C.LIFT_SUCCESS_DZ)   # IK lift of the grasp point for NearGoal
SETTLE = 10                   # control steps of holding command per placement / verification


def load_pool(path=POOL_PATH, device="cuda", split="train"):
    """The pool as device tensors: q9 [N,9] joint state, hold [N,9] the holding command it was
    verified under, cube [N,7] env-local cube pose [qx,qy,qz,qw, px,py,pz], family [N],
    sub [N] (Reaching: 0 home, 1 aligned; else -1), on_table [N] bool. split "eval" is the
    held-out Reaching set."""
    z = np.load(path)
    p = "" if split == "train" else "eval_"
    out = {k: torch.as_tensor(z[p + k], device=device)
           for k in ("q9", "hold", "cube", "family", "sub", "on_table")}
    out["family"] = out["family"].long()
    out["sub"] = out["sub"].long()
    out["on_table"] = out["on_table"].bool()
    out["n"] = int(out["q9"].shape[0])
    return out


# --------------------------------------------------------------------------- #
#  The builder
# --------------------------------------------------------------------------- #
class _Builder:
    def __init__(self, K, seed):
        import kuka_grasp_robot as R
        from kuka_grasp_kin import KukaKinematics
        from kuka_grasp_sim import KukaGraspSim
        self.sim = KukaGraspSim(K)
        self.K, self.dev = K, self.sim.device
        self.kin = KukaKinematics(R.find_kuka_urdf(), device=self.dev)
        self.g = torch.Generator(device=self.dev).manual_seed(seed)
        self.idx = torch.arange(K, device=self.dev)
        self.dq = torch.tensor(C.DEFAULT_Q, device=self.dev)
        self.ik_rejects = 0

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.g, device=self.dev)

    def cubes(self):
        """K cube poses on the table, env-local, [K,7], and their yaws."""
        K = self.K
        yaw = (self.rand(K) * 2 - 1) * np.pi
        pose = torch.zeros(K, 7, device=self.dev)
        pose[:, 2], pose[:, 3] = torch.sin(yaw / 2), torch.cos(yaw / 2)
        pose[:, 4] = REGION_X[0] + self.rand(K) * (REGION_X[1] - REGION_X[0])
        pose[:, 5] = REGION_Y[0] + self.rand(K) * (REGION_Y[1] - REGION_Y[0])
        pose[:, 6] = C.CUBE_REST_Z
        return pose, yaw

    def ik_over(self, pose, yaw, z):
        """IK postures with the grasp point over each cube at height z [K] and the jaws square to
        it (the square yaw nearest the A1-rotated home jaw). Returns q [K,7] and a valid mask."""
        from kuka_grasp_kin import down_frame
        x, y = pose[:, 4], pose[:, 5]
        q0 = self.dq.expand(self.K, 7).clone()
        q0[:, 0] = torch.atan2(y, x)
        _, R0 = self.kin.grasp_frame(q0)
        ref = torch.atan2(R0[:, 1, 0], R0[:, 0, 0])
        jaw = yaw + torch.round((ref - yaw) / (np.pi / 2)) * (np.pi / 2)
        q, pe, re = self.kin.ik(torch.stack([x, y, z], 1), down_frame(jaw), q0)
        ok = (pe < 3e-3) & (re < 3e-2)
        self.ik_rejects += int((~ok).sum().item())
        return q, ok

    def world_pose(self, local):
        w = local.clone()
        w[:, 4] += self.sim.env_x
        return w

    def place(self, q9, cube_local, hold, on_table, steps=SETTLE):
        """Teleport every env to (q9, cube) with zero velocity and settle under `hold`."""
        sim = self.sim
        sim.set_joint_state(self.idx, q9, torch.zeros_like(q9))
        w = self.world_pose(cube_local)
        w[:, 6] += C.CUBE_PLACE_DZ * on_table.float()
        sim.set_cube_state(self.idx, w)
        sim.apply_drive_target(hold)
        sim.substep(C.PHYS_DT, steps * C.SUBSTEPS)

    def drive(self, hold, steps):
        self.sim.apply_drive_target(hold)
        self.sim.substep(C.PHYS_DT, steps * C.SUBSTEPS)

    def state(self):
        """Settled (q9, env-local cube pose)."""
        q9 = self.sim.joint_pos.clone()
        cube = self.sim.cube_pose.clone()
        cube[:, 4] -= self.sim.env_x
        return q9, cube

    def held(self):
        tip = self.sim.tip_state()[0]
        cube = self.sim.cube_position()
        return ((cube - tip).norm(dim=1) < 0.02) & (self.sim.finger_opening() < 0.018)

    def upright(self):
        return self.sim.cube_up()[:, 2] > 0.995

    # ---- families -------------------------------------------------------------------
    def reaching(self, sub):
        K, dev = self.K, self.dev
        pose, yaw = self.cubes()
        opened = torch.full((K, 2), C.GRIP_OPEN, device=dev)
        if sub == 0:
            q = self.dq + (self.rand(K, 7) * 2 - 1) * HOME_NOISE
            ok = torch.ones(K, dtype=torch.bool, device=dev)
        else:
            z = C.GRASP_TABLE_Z + ALIGN_DZ[0] + self.rand(K) * (ALIGN_DZ[1] - ALIGN_DZ[0])
            q, ok = self.ik_over(pose, yaw, z)
        hold = torch.cat([q, opened], 1)
        self.place(hold, pose, hold, torch.ones(K, dtype=torch.bool, device=dev))
        q9, cube = self.state()
        moved = (cube[:, 4:7] - pose[:, 4:7]).norm(dim=1)
        ok &= (moved < 4e-3) & self.upright()
        return q9, hold, cube, ok

    def stable_grasp(self, keep=False):
        K, dev = self.K, self.dev
        pose, yaw = self.cubes()
        q, ok = self.ik_over(pose, yaw, torch.full((K,), C.GRASP_TABLE_Z, device=dev))
        opened = torch.full((K, 2), C.GRIP_OPEN, device=dev)
        closed = torch.full((K, 2), C.GRIP_CLOSE, device=dev)
        self.place(torch.cat([q, opened], 1), pose, torch.cat([q, opened], 1),
                   torch.ones(K, dtype=torch.bool, device=dev))
        moved = (self.sim.cube_position() - self.world_pose(pose)[:, 4:7]).norm(dim=1)
        hold = torch.cat([q, closed], 1)
        self.drive(hold, 15)
        q9, cube = self.state()
        ok &= (moved < 4e-3) & self.held() & ((cube[:, 6] - C.CUBE_REST_Z).abs() < 5e-3)
        if keep:
            return q9, hold, cube, ok, (pose, yaw, q)
        return q9, hold, cube, ok

    def near_goal(self):
        K, dev = self.K, self.dev
        _, hold0, _, ok, (pose, yaw, q_grasp) = self.stable_grasp(keep=True)
        dz = LIFT_CMD_DZ[0] + self.rand(K) * (LIFT_CMD_DZ[1] - LIFT_CMD_DZ[0])
        q_lift, ok2 = self.ik_over(pose, yaw, C.GRASP_TABLE_Z + dz)
        closed = torch.full((K, 2), C.GRIP_CLOSE, device=dev)
        for k in range(1, 9):
            f = k / 8.0
            self.drive(torch.cat([q_grasp + f * (q_lift - q_grasp), closed], 1), 5)
        hold = torch.cat([q_lift, closed], 1)
        self.drive(hold, 10)
        q9, cube = self.state()
        rise = cube[:, 6] - C.CUBE_REST_Z
        ok &= ok2 & self.held() & (rise > 0.01) & (rise <= C.LIFT_SUCCESS_DZ)
        return q9, hold, cube, ok

    def collect(self, fn, n, *a):
        out, got, tries = [], 0, 0
        while got < n:
            q9, hold, cube, ok = fn(*a)
            out.append((q9[ok], hold[ok], cube[ok]))
            got += int(ok.sum().item())
            tries += self.K
        q9, hold, cube = (torch.cat(t)[:n] for t in zip(*out))
        return q9, hold, cube, tries

    # ---- verification -----------------------------------------------------------------
    def verify(self, q9, hold, cube, family):
        """Reset each configuration from whatever the env held before and settle it under its
        holding command; True where it is still what it claims to be."""
        n = q9.shape[0]
        keep = torch.zeros(n, dtype=torch.bool, device=self.dev)
        perm = torch.randperm(n, generator=self.g, device=self.dev)
        for s in range(0, n, self.K):
            j = perm[s:s + self.K]
            m = j.numel()
            pad = torch.cat([j, j[:1].expand(self.K - m)]) if m < self.K else j
            on_table = family[pad] != NEAR_GOAL
            self.place(q9[pad], cube[pad], hold[pad], on_table)
            c = self.sim.cube_pose.clone()
            c[:, 4] -= self.sim.env_x
            moved = (c[:, 4:7] - cube[pad, 4:7]).norm(dim=1)
            f = family[pad]
            ok = torch.where(f == REACHING, (moved < 5e-3) & self.upright(), self.held())
            ok &= torch.where(f == NEAR_GOAL, c[:, 6] > C.CUBE_REST_Z + 0.005, torch.ones_like(ok))
            keep[j] = ok[:m]
        return keep


def build(n_reach=2048, n_grasp=1024, n_near=1024, n_eval=256, K=1024, seed=0, path=POOL_PATH):
    t0 = time.perf_counter()
    b = _Builder(K, seed)
    parts, report = [], {}
    for name, fam, sub, fn, n in (("reaching_home", REACHING, 0, b.reaching, n_reach // 2),
                                  ("reaching_aligned", REACHING, 1, b.reaching, n_reach - n_reach // 2),
                                  ("stable_grasp", STABLE_GRASP, -1, b.stable_grasp, n_grasp),
                                  ("near_goal", NEAR_GOAL, -1, b.near_goal, n_near)):
        args = (sub,) if fam == REACHING else ()
        q9, hold, cube, tries = b.collect(fn, n, *args)
        parts.append((q9, hold, cube, fam, sub))
        report[name] = {"kept": n, "tried": tries}
        print(f"[pool] {name:17s} {n} of {tries} candidates accepted")

    def stack(ps):
        q9 = torch.cat([p[0] for p in ps]); hold = torch.cat([p[1] for p in ps])
        cube = torch.cat([p[2] for p in ps])
        fam = torch.cat([torch.full((p[0].shape[0],), p[3], device=b.dev) for p in ps]).long()
        sub = torch.cat([torch.full((p[0].shape[0],), p[4], device=b.dev) for p in ps]).long()
        return q9, hold, cube, fam, sub

    q9, hold, cube, fam, sub = stack(parts)
    keep = b.verify(q9, hold, cube, fam)
    for f, name in enumerate(FAMILIES):
        m = fam == f
        report[name + "_verified"] = {"kept": int((keep & m).sum().item()), "of": int(m.sum().item())}
        print(f"[pool] verify {name:13s} {int((keep & m).sum().item())} of {int(m.sum().item())} kept")
    q9, hold, cube, fam, sub = q9[keep], hold[keep], cube[keep], fam[keep], sub[keep]

    # the held-out evaluation set: Reaching only, another seed
    b.g.manual_seed(seed + 7919)
    ev = []
    for s in (0, 1):
        e_q9, e_hold, e_cube, _ = b.collect(b.reaching, n_eval // 2 if s == 0 else n_eval - n_eval // 2, s)
        ev.append((e_q9, e_hold, e_cube, REACHING, s))
    e_q9, e_hold, e_cube, e_fam, e_sub = stack(ev)
    e_keep = b.verify(e_q9, e_hold, e_cube, e_fam)
    print(f"[pool] eval reaching set {int(e_keep.sum().item())} of {e_keep.numel()} verified")
    e_q9, e_hold, e_cube, e_fam, e_sub = (t[e_keep] for t in (e_q9, e_hold, e_cube, e_fam, e_sub))

    np_ = lambda t: t.detach().cpu().numpy()
    np.savez(path, q9=np_(q9), hold=np_(hold), cube=np_(cube), family=np_(fam).astype(np.int8),
             sub=np_(sub).astype(np.int8), on_table=np_(fam != NEAR_GOAL),
             eval_q9=np_(e_q9), eval_hold=np_(e_hold), eval_cube=np_(e_cube),
             eval_family=np_(e_fam).astype(np.int8), eval_sub=np_(e_sub).astype(np.int8),
             eval_on_table=np.ones(e_q9.shape[0], bool), seed=seed, ik_rejects=b.ik_rejects)
    el = time.perf_counter() - t0
    print(f"[pool] {q9.shape[0]} cells + {e_q9.shape[0]} eval cells -> {path}  ({el:.1f} s, "
          f"{b.ik_rejects} IK rejects)")
    return report


def check(path=POOL_PATH):
    p = load_pool(path, "cpu")
    e = load_pool(path, "cpu", "eval")
    for f, name in enumerate(FAMILIES):
        m = p["family"] == f
        rise = (p["cube"][m, 6] - C.CUBE_REST_Z)
        print(f"{name:13s} {int(m.sum()):5d} cells  cube rise {rise.min():.3f}..{rise.max():.3f} m")
    print(f"eval reaching {e['n']} cells (home {int((e['sub'] == 0).sum())}, aligned {int((e['sub'] == 1).sum())})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=1024)
    ap.add_argument("--out", default=POOL_PATH)
    args = ap.parse_args()
    if args.check:
        check(args.out)
    else:
        build(K=args.K, seed=args.seed, path=args.out)
        check(args.out)
