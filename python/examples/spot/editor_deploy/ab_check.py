"""Verify the exported bundle against the torch checkpoint, on the plant the policy was TRAINED on.

Two phases, both on flat ground:

  PARITY  drive the sim with the torch policy and, at every tick, also assemble the observation
          and action through the numpy bundle from the SAME state. Identical inputs, so any
          difference is a bug in the export or in the runner's obs assembly — not in the physics.
  WALK    reset, hand the sim to the numpy controller alone, and measure: did it stay upright and
          go where it was told? Parity on one tick does not prove a closed loop stays closed.

    python ab_check.py                    # 120 parity ticks + a 400-tick walk
    python ab_check.py --parity 300 --walk 600

Needs a PhysX-enabled threepp and torch. The native module lives in the main checkout's
python/threepp (a build artifact, shared rather than copied); THREEPP_DLL_DIR overrides.
"""
import argparse
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPOT = os.path.dirname(_HERE)
_PYROOT = os.path.dirname(os.path.dirname(_SPOT))
sys.path.insert(0, _PYROOT)
sys.path.insert(0, _SPOT)
sys.path.insert(0, os.path.join(_SPOT, "scratch_distillation"))
sys.path.insert(0, _HERE)

# PhysX's DLLs sit beside the built .pyd in the main checkout.
_dll = os.environ.get("THREEPP_DLL_DIR", r"C:\dev\threepp\python\threepp")
if os.path.isdir(_dll):
    os.add_dll_directory(_dll)

import torch                                                              # noqa: E402
import threepp as tp                                                      # noqa: E402
from threepp.rl import load_policy                                        # noqa: E402
from spot_deploy import (build_spot, fetch_assets, default_q, add_to_isaac,  # noqa: E402
                         isaac_to_add, ACTION_SCALE, Z0, STIFF_GAINS, LEGS, _quat_to_R)
from policy_runner import PolicyBundle, PolicyController, Adapter         # noqa: E402

GRAV_Z_UP = np.array([0.0, 0.0, -1.0])


class DeployAdapter(Adapter):
    """The training plant: a threepp PhysxWorld articulation built by spot_deploy.build_spot.

    Its DOF order is build_spot's add order (per leg, hx/hy/kn) — deliberately NOT the policy's
    order, so the runner's name mapping is exercised rather than accidentally bypassed.
    """

    JOINTS = [f"{L}.{j}" for L in LEGS for j in ("hx", "hy", "kn")]

    def __init__(self, art):
        self.art = art

    @property
    def joint_names(self):
        return self.JOINTS

    @property
    def joint_positions(self):
        return np.asarray(self.art.joint_positions(), float)

    @property
    def joint_velocities(self):
        return np.asarray(self.art.joint_velocities(), float)

    @property
    def root_position(self):
        return np.asarray(self.art.root_state()[0:3], float)

    @property
    def root_rotation_matrix(self):
        return _quat_to_R(self.art.root_state()[3:7])

    @property
    def root_linear_velocity(self):
        return np.asarray(self.art.root_velocity()[0:3], float)

    @property
    def root_angular_velocity(self):
        return np.asarray(self.art.root_velocity()[3:6], float)

    @property
    def gravity_direction(self):
        return GRAV_Z_UP

    def set_drive_targets(self, values):
        self.art.set_drive_targets(np.asarray(values, np.float32))


def reference_obs(art, last_act, cmd, phi):
    """VERBATIM from play_spot_steps.v2_obs, specialised to flat ground (ahead = 0, h_here = 0).
    This is the shipped player's own observation code — the thing the runner must reproduce."""
    rs, rv = art.root_state(), art.root_velocity()
    R = _quat_to_R(rs[3:7])
    Rt = R.T
    lin_b, ang_b, proj_g = Rt @ rv[0:3], Rt @ rv[3:6], Rt @ GRAV_Z_UP
    jp_isaac = art.joint_positions()[isaac_to_add]
    jv_isaac = art.joint_velocities()[isaac_to_add]
    qpos = jp_isaac - default_q
    z = float(rs[2])
    clk = [math.sin(2 * math.pi * phi), math.cos(2 * math.pi * phi)]
    ahead = np.zeros(45, np.float32)
    return np.concatenate([lin_b, ang_b, proj_g, cmd, qpos, jv_isaac, last_act,
                           clk, [z - 0.0], ahead]).astype(np.float32)


def heading_hold(art, lock):
    """The player's yaw-rate P controller: the policy regulates yaw RATE, so idle bias spirals."""
    R = _quat_to_R(art.root_state()[3:7])
    yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    if lock[0] is None:
        lock[0] = yaw
    err = (yaw - lock[0] + math.pi) % (2 * math.pi) - math.pi
    return float(np.clip(-2.0 * err, -1.0, 1.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=os.path.join(_HERE, "bundle_spot_steps"))
    ap.add_argument("--model", default=os.path.join(_SPOT, "spot_steps.pt"))
    ap.add_argument("--parity", type=int, default=120)
    ap.add_argument("--walk", type=int, default=400)
    ap.add_argument("--vx", type=float, default=0.8)
    args = ap.parse_args()
    assert tp.HAS_PHYSX, "needs a PhysX-enabled threepp build"

    ac, norm, meta = load_policy(args.model, device="cpu")
    bundle = PolicyBundle(args.bundle)
    print(f"[bundle] {os.path.basename(args.bundle)}  obs={bundle.obs_dim} act={bundle.act_dim} "
          f"dt={bundle.dt}  joints={len(bundle.joints)}")

    # tgs_pcm/0.005 = the GpuSim contact model the policy was trained against
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=0.005,
                          max_substeps=8, tgs_pcm=True)
    ground = tp.Mesh(tp.BoxGeometry(200, 200, 1.0), tp.MeshStandardMaterial())
    ground.position.set(0, 0, -0.5)
    world.add_static(ground)
    art, _ = build_spot(world, None, gains=STIFF_GAINS)

    adapter = DeployAdapter(art)
    ctrl = PolicyController(bundle, adapter)
    # The runner derived its joint mapping from names alone; spot_deploy computed the same
    # permutation by hand. They must agree, or one of the two contracts is wrong.
    assert np.array_equal(ctrl.p2s, isaac_to_add), f"mapping mismatch {ctrl.p2s} vs {isaac_to_add}"
    print(f"[map] policy->sim = {list(ctrl.p2s)}  (== spot_deploy.isaac_to_add)")

    for _ in range(150):                      # stand up, exactly as spot_deploy.hold does
        ctrl.hold_default()
        world.step(0.02)

    # ------------------------------------------------------------------ PARITY
    last_act = np.zeros(12, np.float32)
    phi = 0.0
    lock = [None]
    d_obs = d_act = 0.0
    for _ in range(args.parity):
        wz = heading_hold(art, lock)
        cmd = np.array([args.vx, 0.0, wz], np.float32)

        ref_o = reference_obs(art, last_act, cmd, phi)
        ctrl.last_action, ctrl.phi = last_act, phi          # same state, both paths
        my_o = ctrl.observe(cmd)
        d_obs = max(d_obs, float(np.abs(ref_o - my_o).max()))

        with torch.no_grad():
            o_t = torch.from_numpy(ref_o)[None]
            if norm is not None:
                o_t = norm.norm(o_t)
            ref_a = ac.act_mean(o_t)[0].numpy()
        my_a = bundle.act(my_o)
        d_act = max(d_act, float(np.abs(ref_a - my_a).max()))

        art.set_drive_targets((default_q + ACTION_SCALE * ref_a)[add_to_isaac].astype(np.float32))
        world.step(0.02)
        last_act = ref_a
        phi = (phi + 0.02 / 0.5) % 1.0

    print(f"[parity] {args.parity} ticks   max |obs diff| = {d_obs:.3e}   "
          f"max |action diff| = {d_act:.3e}")
    obs_ok, act_ok = d_obs < 1e-5, d_act < 1e-4
    print(f"         obs {'OK' if obs_ok else 'FAIL'}   action {'OK' if act_ok else 'FAIL'}")

    # ------------------------------------------------------------------- WALK
    art.reset(tp.Vector3(0, 0, Z0))
    ctrl.last_action = np.zeros(12, np.float32)
    ctrl.phi = 0.0
    for _ in range(150):
        ctrl.hold_default()
        world.step(0.02)
    p0 = adapter.root_position.copy()
    lock = [None]
    zmin = 9e9
    for _ in range(args.walk):
        cmd = np.array([args.vx, 0.0, heading_hold(art, lock)], np.float32)
        ctrl.step(cmd)                     # numpy bundle alone drives the plant now
        world.step(0.02)
        zmin = min(zmin, float(adapter.root_position[2]))
    p1 = adapter.root_position
    R = adapter.root_rotation_matrix
    dist = float(p1[0] - p0[0])
    secs = args.walk * 0.02
    upright = float(R[2, 2])
    print(f"[walk]   {args.walk} ticks ({secs:.1f}s) at vx={args.vx}: "
          f"travelled {dist:+.2f} m  ({dist / secs:+.2f} m/s)")
    print(f"         base=({p1[0]:+.2f},{p1[1]:+.2f},{p1[2]:.2f})  min height {zmin:.2f}  "
          f"upright(cos tilt)={upright:.3f}")
    walk_ok = upright > 0.8 and zmin > 0.35 and dist > 0.5 * args.vx * secs
    print(f"         {'OK' if walk_ok else 'FAIL'} (upright, never collapsed, tracked the command)")

    ok = obs_ok and act_ok and walk_ok
    print(f"\nBUNDLE A/B: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
