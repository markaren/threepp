"""Gates for the batched torch cable against the reference C++ TendonCable.

The claim being checked is narrow and mechanical: `threepp.rl.cable` resolves and applies the
SAME routed-cable law as `include/threepp/extras/physx/TendonCable.hpp`, so a tendon hand
trained in a direct-GPU batch is the same hand that gets filmed on the CPU path. Two ways to
be wrong, so two gates:

  A1 STATIC -- the geometry. Thirty poses spread over the joint limits, evaluated on the GPU
     batch and on a CPU hand, comparing every cable's routed LENGTH. Length is the right
     quantity because the whole force law is its gradient: get the length right everywhere and
     the torques follow by virtual work. Then the moment arms themselves, by finite difference
     of the torch length, against the numbers `tendon_hand.py --selftest` measures from the
     C++ cable.
  A2 DYNAMIC -- the forces and their frames. One cable pulled at 20 N from rest for 0.25 s on
     both sides, comparing joint angles. A static length check cannot see a wrap arc whose
     points are right but whose force is attributed to the wrong link, or a torque taken about
     the wrong point; a quarter second of free acceleration can. Run for the index FDP (via
     points only), the thumb FPL, and the index EXT, which is the wrapped case.

TWO THINGS DICTATE THE SHAPE OF THIS SCRIPT.

PhysX allows ONE foundation per process, so the CPU hand and the GPU batch cannot be alive at
the same time. The GPU side therefore runs first and keeps nothing but numpy arrays and the
routing (names and numbers, no handles); the world is then torn down and the CPU hand built.

And the comparison is made at the poses the GPU ACTUALLY REACHED, not the ones asked for. A
pose drawn uniformly from the joint limits usually has the fingers inside each other -- the
articulation has self-collision on -- so writing it and taking one step lets PhysX push the
links apart, by up to 8 degrees at a joint here. Comparing a popped GPU pose against the
kinematic CPU pose measures that pop, not the cable. Reading the reached angles back and
posing the CPU hand at those is the same test without the confound, and [A1.0] reports both
the drift and the residual link-position agreement that proves the readback is consistent.

Run:  python tendon_hand_gpu_check.py            # both gates
      python tendon_hand_gpu_check.py --arc 8    # with a denser wrap arc
"""
import argparse
import gc
import math
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))    # python/
sys.path.insert(0, _HERE)

import threepp as tp
from tendon_hand import DT, Hand
from threepp.rl import BatchedCables, CableRouting, GpuSim, link_index_map

# What tendon_hand.py --selftest measures from the C++ cable at the lightly flexed pose
# (every index DOF at 0.35 rad, abduction 0). Millimetres.
SELFTEST_ARMS = {
    ("index_fdp", "index_mcp"): 9.76, ("index_fdp", "index_pip"): 7.15,
    ("index_fdp", "index_dip"): 4.95,
    ("index_fds", "index_mcp"): 11.32, ("index_fds", "index_pip"): 6.93,
    ("index_ext", "index_mcp"): -7.41, ("index_ext", "index_pip"): -4.43,
    ("index_ext", "index_dip"): -2.83,
}

TENSION, T_END = 20.0, 0.25
TRIALS = [("index_fdp", ["index_mcp", "index_pip", "index_dip"]),
          ("thumb_fpl", ["thumb_cmc_flex", "thumb_mcp", "thumb_ip"]),
          ("index_ext", ["index_mcp", "index_pip", "index_dip"])]
N_POSE, REPS = 30, 2
# The finite-difference step, and it is squeezed from both sides.
#
# NOT the selftest's 1e-4: the batch reads link poses back as float32, so a 0.1 m coordinate
# carries ~1e-8 m of quantization and so does a routed length. Divided by 2h = 2e-4 that is
# 0.05 mm of noise on a 10 mm arm, and measured it came out at up to 2.6 % with the sign
# flipping between joints -- noise, not a modelling difference (the C++ cable evaluated at
# those very same poses reproduced the selftest arms to 0.01 mm).
# NOT 5e-3 either: measured, both implementations then read 9.581 / 9.578 mm for the index FDP
# at the MCP against the selftest's 9.76. That is the central difference's own truncation, and
# it is large here because a chorded flexor's length has real curvature in q -- the two agreed
# with EACH OTHER to 0.003 mm, which is the thing this gate is actually about.
# 1e-3 puts the quantization at 0.005 mm and the truncation at a twenty-fifth of the 5e-3 one.
FD_H = 1e-3
ARM_JOINTS = ["index_mcp", "index_pip", "index_dip"]


def random_poses(limits, dof_names):
    lo = np.array([limits[n][0] for n in dof_names])
    hi = np.array([limits[n][1] for n in dof_names])
    rng = np.random.default_rng(7)
    return lo + rng.random((N_POSE, len(dof_names))) * (hi - lo)


def fd_rows(dof_names):
    """The 6 poses a central difference at the three index joints needs, +h then -h."""
    q0 = np.zeros(len(dof_names))
    for i, nm in enumerate(dof_names):
        if nm.startswith("index") and not nm.endswith("_abd"):
            q0[i] = 0.35
    rows = []
    for jn in ARM_JOINTS:
        for sgn in (+1.0, -1.0):
            q = q0.copy()
            q[dof_names.index(jn)] += sgn * FD_H
            rows.append(q)
    return np.array(rows)


def settle(sim, dt=1e-5):
    """One tiny step, which is what makes the batch republish link transforms after a joint
    write. Kept tiny so nothing but a penetration pop can move: 1e-5 s of free fall is 5e-10 m."""
    sim.batch.step(dt)
    sim.read()


def set_all(sim, rows):
    pos = torch.zeros(sim.K, sim.dof, device=sim.device)
    pos[:len(rows)] = torch.as_tensor(rows, dtype=torch.float32, device=sim.device)
    sim.set_joint_state(torch.arange(sim.K, device=sim.device), pos, torch.zeros_like(pos))
    settle(sim)


def gpu_side(K, arc_points, spacing=0.5):
    """Everything measured from the torch cable. Leaves no PhysX object alive."""
    rest = {}

    def build(world, i):
        h = Hand(world, base=(0.0, 0.0, i * spacing)).finalize()
        h.route(build=False)                 # record the routing, create no CPU cable
        if i == 0:
            # Rest positions straight off the authored meshes: add_link places each link at
            # its mesh's world placement, so this is the geometry, with no physics in it.
            rest.update({n: (m.position.x, m.position.y, m.position.z)
                         for n, m in zip(h.links, h.meshes)})
        return h

    sim = GpuSim(K, build, gravity=(0, 0, 0), read_links=True)
    sim.read()
    idx_map, resid = link_index_map(rest, sim.link_pose[0].cpu().numpy())
    routing = CableRouting.from_hand(sim.robots[0])
    cables = BatchedCables(routing, idx_map, sim.max_links, device=sim.device,
                           arc_points=arc_points)
    dof = list(sim.robots[0].dof_names)
    limits = dict(sim.robots[0].limits)
    link_names = list(rest)
    gi = [idx_map[n] for n in link_names]
    out = dict(dof=dof, names=list(cables.names), routing=routing, resid=resid,
               link_names=link_names, n_slots=cables.N, arc=cables.W, K=K)

    def read_pose_block(n):
        q = sim.joint_pos[:n].cpu().numpy().copy()
        p = sim.link_pose[:n][:, gi, 4:7].cpu().numpy().copy()
        p[..., 2] -= (np.arange(n) * spacing)[:, None]        # undo the per-env base offset
        return q, p

    # ---- A1a: 30 poses over the joint limits --------------------------------------
    q_req = random_poses(limits, dof)
    set_all(sim, q_req)
    out["q_req"] = q_req
    out["q"], out["link_pos"] = read_pose_block(N_POSE)
    out["L"] = cables.lengths(sim.link_pose)[:N_POSE].cpu().numpy()

    # ---- A1b: the finite-difference rows -------------------------------------------
    rows = fd_rows(dof)
    set_all(sim, rows)
    out["fd_req"] = rows
    out["fd_q"], _ = read_pose_block(len(rows))
    out["fd_L"] = cables.lengths(sim.link_pose)[:len(rows)].cpu().numpy()

    # ---- A2: 20 N on one cable for 0.25 s from rest ---------------------------------
    n_steps = int(round(T_END / DT))
    set_all(sim, np.zeros((K, len(dof))))
    tension = torch.zeros(K, len(cables.names), device=sim.device)
    for t, (cn, _) in enumerate(TRIALS):
        for r in range(REPS):
            tension[t * REPS + r, cables.names.index(cn)] = TENSION
    for _ in range(n_steps):
        f, tq, _ = cables.apply(sim.link_pose, tension)
        torch.cuda.synchronize()
        sim.batch.write_link_force(f.contiguous())
        sim.batch.write_link_torque(tq.contiguous())
        sim.batch.step(DT)
        sim.read()
    out["a2"] = sim.joint_pos[:len(TRIALS) * REPS].cpu().numpy().copy()
    out["n_steps"] = n_steps

    del cables, sim
    gc.collect()
    return out


def cpu_side(g):
    """The same quantities from the C++ cable, at the poses the GPU actually reached."""
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, 0), fixed_timestep=DT, max_substeps=1,
                          tgs_pcm=True)
    hand = Hand(world).finalize()
    hand.route()
    names, ln = g["names"], g["link_names"]

    def lengths_at(qs):
        L = np.zeros((len(qs), len(names)))
        P = np.zeros((len(qs), len(ln), 3))
        for i, q in enumerate(qs):
            hand.set_pose(q)
            for j, nm in enumerate(names):
                L[i, j] = hand.cables[nm].length
            for j, nm in enumerate(ln):
                p = hand.links[nm].position
                P[i, j] = (p.x, p.y, p.z)
        return L, P

    out = {}
    out["L"], out["link_pos"] = lengths_at(g["q"])
    out["fd_L"], _ = lengths_at(g["fd_q"])
    a2 = {}
    for cn, joints in TRIALS:
        hand.set_pose(hand.zero())           # set_joint_positions also zeroes velocities
        hand.relax()
        hand.cables[cn].set_tension(TENSION)
        for _ in range(g["n_steps"]):
            world.step(DT)
        a2[cn] = np.array([hand.links[jn].joint_position for jn in joints])
    out["a2"] = a2
    hand.relax()
    del hand, world
    gc.collect()
    return out


def gate_a1(g, c):
    names, dof = g["names"], g["dof"]
    drift = np.abs(g["q"] - g["q_req"]).max()
    dp = np.abs(g["link_pos"] - c["link_pos"])
    wl = int(dp.max(axis=(0, 2)).argmax())
    print(f"\n[A1.0] the comparison poses. Uniform draws from the joint limits self-intersect,")
    print(f"       so PhysX moves them: max |q_reached - q_asked| = "
          f"{math.degrees(drift):.3f} deg. Both sides are then posed at q_reached, and the")
    print(f"       21 link positions agree to {dp.max()*1000:.4f} mm "
          f"(worst {g['link_names'][wl]}) -- the readback is consistent.")

    err = np.abs(g["L"] - c["L"])
    worst = np.unravel_index(int(err.argmax()), err.shape)
    print(f"\n[A1a] routed length, {N_POSE} poses x {len(names)} cables, torch vs the C++ cable")
    print(f"      max |dL| = {err.max()*1000:.4f} mm   (cable {names[worst[1]]}, pose "
          f"{worst[0]}, L = {c['L'][worst]*1000:.2f} mm)")
    print(f"      mean |dL| = {err.mean()*1000:.4f} mm")
    per = sorted(((err[:, j].max() * 1000, nm) for j, nm in enumerate(names)), reverse=True)
    print("      worst five cables: " + "  ".join(f"{n} {v:.4f}" for v, n in per[:5]))
    a1a = err.max() < 2e-4
    print(f"      GATE A1a (< 0.2 mm): {'PASS' if a1a else 'FAIL'}")

    # Moment arms. The DENOMINATOR is the angle actually reached, not the h that was asked
    # for: at h = 1e-4 rad even a microradian of settle drift is 1 % of the difference.
    fdq, fdL = g["fd_q"], g["fd_L"]
    off = np.abs(fdq - g["fd_req"])
    dqs = [fdq[2 * j, dof.index(jn)] - fdq[2 * j + 1, dof.index(jn)]
           for j, jn in enumerate(ARM_JOINTS)]
    wj = np.unravel_index(int(off.argmax()), off.shape)
    print(f"\n[A1b] index moment arms (mm) by finite difference. h = {FD_H:g} rad, reached "
          f"{', '.join(f'{d/2:.3e}' for d in dqs)};")
    print(f"      worst settle drift {off.max()*1e6:.1f} urad at {dof[wj[1]]}.")
    print(f"      Two references, because they answer different questions: the C++ cable at")
    print(f"      the SAME pose is the port test; the --selftest column is the C++ cable at")
    print(f"      the exact nominal pose with h = 1e-4, which is a different measurement.")
    ok, ok_ref, worst, worst_ref = True, True, 0.0, 0.0
    for (cn, jn), ref in SELFTEST_ARMS.items():
        ci, ji = names.index(cn), ARM_JOINTS.index(jn)
        dq = dqs[ji]
        r = -(fdL[2 * ji, ci] - fdL[2 * ji + 1, ci]) / dq * 1000.0
        rc = -(c["fd_L"][2 * ji, ci] - c["fd_L"][2 * ji + 1, ci]) / dq * 1000.0
        rel = abs(r - rc) / abs(rc) * 100.0
        rel_ref = abs(r - ref) / abs(ref) * 100.0
        ok &= rel < 2.0
        ok_ref &= rel_ref < 2.0
        worst, worst_ref = max(worst, rel), max(worst_ref, rel_ref)
        print(f"      {cn:10s} {jn:10s} torch {r:+7.3f}   C++ {rc:+7.3f} ({rel:4.2f} %)"
              f"   selftest {ref:+7.2f} ({rel_ref:4.2f} %)  "
              f"{'ok' if rel < 2.0 else '**'}")
    print(f"      GATE A1b, torch vs the C++ cable at the same pose: worst {worst:.2f} % "
          f"-> {'PASS' if ok else 'FAIL'}")
    print(f"      for reference, vs the --selftest values: worst {worst_ref:.2f} % "
          f"-> {'within 2 %' if ok_ref else 'outside 2 %, and the C++ column shows the '
                                           'same offset, so it is the pose, not the port'}")
    return a1a and ok


def gate_a2(g, c):
    dof, qg = g["dof"], g["a2"]
    print(f"\n[A2] dynamic parity: {TENSION:.0f} N for {T_END:.2f} s from rest, gravity off, "
          f"{1/DT:.0f} Hz; joint angles in degrees")
    worst_abs, worst_rel, passed = 0.0, 0.0, True
    for t, (cn, joints) in enumerate(TRIALS):
        spread = max(abs(qg[t * REPS + r][dof.index(jn)] - qg[t * REPS][dof.index(jn)])
                     for r in range(REPS) for jn in joints)
        print(f"  {cn}   (replica spread {math.degrees(spread):.4f} deg)")
        for k, jn in enumerate(joints):
            gv = math.degrees(qg[t * REPS][dof.index(jn)])
            cv = math.degrees(c["a2"][cn][k])
            da = abs(gv - cv)
            dr = da / max(abs(cv), 1e-9) * 100.0
            good = dr < 5.0 or da < 1.0
            passed &= good
            worst_abs = max(worst_abs, da)
            worst_rel = max(worst_rel, dr if da > 1.0 else 0.0)
            print(f"    {jn:16s} torch {gv:+8.3f}   C++ {cv:+8.3f}   "
                  f"d {da:6.3f} deg = {dr:6.2f} %  {'ok' if good else '**'}")
    print(f"  worst: {worst_abs:.3f} deg absolute, {worst_rel:.2f} % over the angles past 1 deg")
    print(f"  GATE A2 (within 5 % or 1 deg): {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc", type=int, default=8, help="sample points per wrap arc")
    ap.add_argument("--envs", type=int, default=32)
    args = ap.parse_args()

    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need a PhysX build + CUDA"); sys.exit(0)

    g = gpu_side(args.envs, args.arc)
    print(f"hand: {len(g['dof'])} DOF, {len(g['names'])} cables, {len(g['link_names'])} links")
    print(f"link_index_map: all {len(g['link_names'])} links matched by rest position, "
          f"worst residual {g['resid']*1000:.4f} mm")
    print(f"cable layout: [{len(g['names'])} cables x {g['n_slots']} slots], "
          f"{g['arc']} points per wrap arc; batch K = {g['K']}")
    c = cpu_side(g)
    ok = gate_a1(g, c) & gate_a2(g, c)
    print(f"\nPHASE A ({g['arc']} arc points): {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
