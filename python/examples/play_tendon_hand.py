"""Run a trained hold policy on the CPU hand, with its REAL cables.

This is the deploy path and the sim-to-sim check in one. Training runs on the batched torch
cable because PhysX rejects addForce under the direct-GPU API; the film runs on the C++
`TendonCable`, which is the reference implementation of the same law. Nothing about the policy
changes between the two -- the same 25 tensions go in -- so if the port is right the posture
here is the posture the policy was trained into, and if it is not, this is where it shows.

The observation is rebuilt from the CPU state in the same order `TendonHandEnv.observe` builds
it, and every constant it needs (T_MAX, the filter time constant, the palm target, the pull
schedule) is read from the checkpoint's meta rather than re-declared, so the two cannot drift.

Run:  python play_tendon_hand.py tendon_hand_hold.pt --view
      python play_tendon_hand.py tendon_hand_hold.pt --object sphere --seconds 6
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
from tendon_hand import DT, Forearm, Hand, RopeView, VIEWS, _scene
from tendon_hand_env import OBJ_TYPES, object_mesh
from threepp.rl import load_policy


def build(meta, kind, size_a, size_b, seed):
    g = np.array(meta["gravity"], dtype=float)
    world = tp.PhysxWorld(gravity=tp.Vector3(*g), fixed_timestep=DT, max_substeps=1,
                          tgs_pcm=True)
    pad = world.create_material(1.2, 1.1, 0.0, friction_combine="min")
    hand = Hand(world, material=pad).finalize()
    hand.route()                                    # the REAL cables, this time
    rng = np.random.default_rng(seed)

    target = np.array(meta["palm_target"], dtype=float)
    g_hat = g / np.linalg.norm(g)
    mesh = object_mesh(kind, size_a, size_b)
    jitter = np.array([(rng.random() * 2 - 1) * 0.015, 0.0, (rng.random() * 2 - 1) * 0.015])
    mesh.position.set(*(target + jitter - g_hat * (0.03 + 0.03 * rng.random())))
    art = world.create_articulation(fixed_base=False)
    art.add_link(mesh, density=meta["obj_density"], material=pad)
    art.finalize()
    return world, hand, art, mesh, target, g_hat


def observe(meta, hand, art, mesh, target, tension, pull):
    """The same 80 numbers, in the same order, that TendonHandEnv.observe stacks."""
    q = hand.pose()
    qd = np.array([hand.links[n].joint_velocity for n in hand.dof_names])
    rv = art.root_velocity()
    pos = np.array([mesh.position.x, mesh.position.y, mesh.position.z])
    desc = meta["_desc"]
    return np.concatenate([q, qd * 0.1, tension / meta["t_max"], pos - target,
                           rv[0:3], rv[3:6] * 0.1, desc, pull / 10.0]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--object", default="sphere", choices=list(OBJ_TYPES))
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--view", action="store_true", help="window, with every cable drawn")
    ap.add_argument("--size", default="1280x800")
    a = ap.parse_args()

    ac, norm, meta = load_policy(a.checkpoint, "cpu")
    hz, dt = meta["control_hz"], 1.0 / meta["control_hz"]
    sub = meta["substeps"]
    alpha = dt / (meta["tau_filter"] + dt)
    rng = np.random.default_rng(a.seed)
    if a.object == "sphere":
        sa, sb = rng.uniform(0.018, 0.025), 0.0
    elif a.object == "capsule":
        sa, sb = rng.uniform(0.015, 0.020), rng.uniform(0.060, 0.080)
    else:
        sa, sb = rng.uniform(0.030, 0.045), 0.0
    meta["_desc"] = np.array([sa * 20.0, sb * 20.0, OBJ_TYPES.index(a.object) / 2.0])

    world, hand, art, mesh, target, g_hat = build(meta, a.object, sa, sb, a.seed)
    # The cable order the 25 actions are in. The checkpoint carries it; the hand's own routing
    # is the same list by construction, so it is both the fallback for an older checkpoint and
    # a check that the two have not drifted.
    names = list(hand.routing)
    if "cable_names" in meta and list(meta["cable_names"]) != names:
        raise RuntimeError("the checkpoint's cable order is not this hand's:\n  "
                           f"checkpoint {list(meta['cable_names'])}\n  hand {names}")
    tension = np.zeros(len(names))
    pull = np.zeros(3)
    pull_dir = np.zeros(3)
    body = art.link(0)
    n_steps = int(round(a.seconds * hz))
    every = max(1, int(round(meta["pull_resample_s"] * hz)))

    canvas = renderer = scene = cam = view = arm = None
    if a.view:
        w, h = (int(v) for v in a.size.split("x"))
        canvas, renderer, scene, cam = _scene(w, h, False)
        for m in hand.meshes:
            scene.add(m)
        scene.add(mesh)
        arm, view = Forearm(scene, hand), RopeView(scene, hand)
        pos, tgt = VIEWS["3q"]
        cam.position.set(*pos)
        cam.look_at(*tgt)
        controls = tp.OrbitControls(cam, canvas)
        controls.target = tp.Vector3(*tgt)
        controls.enable_damping = True

    state = {"i": 0}

    def control():
        i = state["i"]
        nonlocal pull_dir, pull, tension
        if i % every == 0:
            d = rng.normal(size=3)
            pull_dir = d / max(np.linalg.norm(d), 1e-6)
        t = i * dt
        k = min(1.0, max(0.0, (t - meta["pull_start_s"])
                         / (meta["episode_s"] - meta["pull_start_s"])))
        pull = pull_dir * (meta["pull_max"] * k)

        obs = torch.from_numpy(observe(meta, hand, art, mesh, target, tension, pull))[None]
        if norm is not None:
            obs = norm.norm(obs)
        act = ac.act_mean(obs)[0].clamp(-1.0, 1.0).numpy()
        cmd = meta["t_max"] * 0.5 * (act + 1.0)
        tension = (1.0 - alpha) * tension + alpha * cmd
        for n, T in zip(names, tension):
            hand.cables[n].set_tension(float(T))
        for _ in range(sub):
            body.add_force(tp.Vector3(*pull))        # re-applied: PhysX clears it every step
            world.step(dt / sub)
        state["i"] = i + 1
        return np.linalg.norm(np.array([mesh.position.x, mesh.position.y,
                                        mesh.position.z]) - target)

    if a.view:
        def loop():
            d = control()
            view.update(arm.update())
            controls.update()
            renderer.render(scene, cam)
            if state["i"] % 30 == 0:
                print(f"  t {state['i']*dt:4.2f} s   |d| {d*1000:6.1f} mm   "
                      f"pull {np.linalg.norm(pull):4.2f} N   "
                      f"mean tension {tension.mean():5.2f} N")
        canvas.animate(loop)
        return

    print(f"policy {a.checkpoint} on the CPU hand with its real cables: {a.object} "
          f"r={sa*1000:.1f} mm, {a.seconds:.1f} s at {hz} Hz")
    worst = 0.0
    for i in range(n_steps):
        d = control()
        worst = max(worst, d)
        if (i + 1) % 30 == 0:
            print(f"  t {(i+1)*dt:4.2f} s   |d| {d*1000:6.1f} mm   "
                  f"pull {np.linalg.norm(pull):4.2f} N   "
                  f"mean tension {tension.mean():5.2f} N")
    held = worst < meta["drop_dist"]
    print(f"  worst offset {worst*1000:.1f} mm against a {meta['drop_dist']*1000:.0f} mm "
          f"limit -> {'HELD' if held else 'DROPPED'}")


if __name__ == "__main__":
    main()
