"""Left-right symmetry mirror for the spotv2 58-d obs / 12-d action (Isaac joint order).

The learned gait drifts right (~0.05 m/s lateral bias at zero strafe command) and strafe tracking
regressed — both are a left/right asymmetry the symmetric reward didn't iron out. The fix is symmetry
AUGMENTATION: penalize the policy for not being equivariant under a left-right mirror, so it must learn
a symmetric gait. This module defines the mirror and the symmetry loss; the trainer passes the loss to
PPO via its `aux_loss` hook.

Mirror = reflection across the body x-z plane (y -> -y):
  lin_b [vx,vy,vz]   -> [vx,-vy,vz]        (true vector)
  ang_b [wx,wy,wz]   -> [-wx,wy,-wz]       (pseudovector: components perpendicular to the mirror negate)
  proj_g [gx,gy,gz]  -> [gx,-gy,gz]        (true vector)
  cmd [vx,vy,wz]     -> [vx,-vy,-wz]       (lateral vel + yaw rate negate)
  joints (qpos/qvel/last_act/action, Isaac order): swap L<->R legs, negate hip-x (abduction)
  base_above         -> unchanged
  scan (1-D forward) -> unchanged          (probes along heading, on the centerline -> symmetric)
"""
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spot"))

# Isaac joint order: [fl_hx,fr_hx,hl_hx,hr_hx, fl_hy,fr_hy,hl_hy,hr_hy, fl_kn,fr_kn,hl_kn,hr_kn].
# Mirror swaps fl<->fr and hl<->hr within each (hx,hy,kn) group; the hip-x (abduction) joints negate.
MIRROR_PERM = [1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10]
MIRROR_SIGN = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

_CACHE = {}


def _consts(device):
    key = str(device)
    if key not in _CACHE:
        _CACHE[key] = {
            "perm": torch.tensor(MIRROR_PERM, dtype=torch.long, device=device),
            "jsign": torch.tensor(MIRROR_SIGN, dtype=torch.float32, device=device),
            "lin": torch.tensor([1.0, -1.0, 1.0], device=device),     # true vector: -y
            "ang": torch.tensor([-1.0, 1.0, -1.0], device=device),    # pseudovector: -x,-z
            "cmd": torch.tensor([1.0, -1.0, -1.0], device=device),    # -vy, -wz
        }
    return _CACHE[key]


def mirror_joints(q, c):
    return q[..., c["perm"]] * c["jsign"]


def mirror_obs(obs):
    """Mirror the 58-d observation left<->right. obs [..., 58]."""
    c = _consts(obs.device)
    o = obs.clone()
    o[..., 0:3] = obs[..., 0:3] * c["lin"]      # lin_b
    o[..., 3:6] = obs[..., 3:6] * c["ang"]      # ang_b
    o[..., 6:9] = obs[..., 6:9] * c["lin"]      # proj_g
    o[..., 9:12] = obs[..., 9:12] * c["cmd"]    # cmd
    o[..., 12:24] = mirror_joints(obs[..., 12:24], c)   # qpos (Isaac)
    o[..., 24:36] = mirror_joints(obs[..., 24:36], c)   # qvel
    o[..., 36:48] = mirror_joints(obs[..., 36:48], c)   # last_act
    # [48:49] base_above and [49:58] forward scan are unchanged
    return o


def mirror_act(a):
    """Mirror a 12-d Isaac-order action left<->right. a [..., 12]."""
    return mirror_joints(a, _consts(a.device))


def symmetry_loss(ac, obs):
    """Equivariance penalty: actor(mirror(obs)) should equal mirror(actor(obs)). Mean-squared over the
    action. Drives the policy toward a symmetric gait (kills the rightward drift + strafe asymmetry)."""
    feat = ac._feat(obs)
    mean = ac.actor(feat)
    mean_m = ac.actor(ac._feat(mirror_obs(obs)))
    return (mean_m - mirror_act(mean)).pow(2).mean()


def make_aux_loss(coef):
    """Build an aux_loss(ac, obs) -> scalar for PPO. coef weights the symmetry penalty."""
    return lambda ac, obs: coef * symmetry_loss(ac, obs)


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # 1) involution: mirroring twice returns the original
    x = torch.randn(64, 58, device=dev)
    err_obs = (mirror_obs(mirror_obs(x)) - x).abs().max().item()
    a = torch.randn(64, 12, device=dev)
    err_act = (mirror_act(mirror_act(a)) - a).abs().max().item()
    print(f"involution: obs err {err_obs:.2e}  act err {err_act:.2e}  (expect ~0)")
    assert err_obs < 1e-5 and err_act < 1e-5, "mirror is not an involution"

    # 2) teacher equivariance on REAL obs: the symmetric Isaac walker should be ~equivariant under a
    #    CORRECT mirror -> teacher(mirror(obs)) ~= mirror(teacher(obs)). A large error = a mirror bug.
    if torch.cuda.is_available():
        import threepp as tp
        from spot_deploy import fetch_assets
        from spot_heightfield_env import SpotHeightfieldEnv
        if tp.HAS_PHYSX:
            env = SpotHeightfieldEnv(num_envs=64, device="cuda")
            obs = env.reset()
            for _ in range(40):                                  # let it move so obs isn't a trivial stand
                obs, _, _, _, _ = env.step(env.imit_policy(env._last_obs[:, :48]))
            teacher = env.imit_policy
            with torch.no_grad():
                a0 = teacher(obs[:, :48])                        # teacher action on obs
                a1 = teacher(mirror_obs(obs)[:, :48])            # teacher action on the mirrored obs
                equiv_err = (a1 - mirror_act(a0)).abs().mean().item()
                base = a0.abs().mean().item()
            print(f"teacher equivariance: mean|t(mirror(o)) - mirror(t(o))| = {equiv_err:.4f}  "
                  f"(action scale {base:.3f}; small relative to scale = mirror correct + teacher ~symmetric)")
    print("SPOT-SYMMETRY SELFTEST: PASS")
