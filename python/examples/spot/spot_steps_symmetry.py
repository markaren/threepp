"""Left-right symmetry mirror for the spotv2-steps 96-d obs / 12-d action (Isaac joint order).

New obs layout: [0:48] proprio | [48:50] clock | [50:51] base_above | [51:96] scan (45).
Mirror combines scratch_symmetry.py (proprio + clock) and spot_symmetry.py (scan), adapted
for the new slot positions:

    [0:3]   lin_b  × [1,−1,1]
    [3:6]   ang_b  × [−1,1,−1]
    [6:9]   proj_g × [1,−1,1]
    [9:12]  cmd    × [1,−1,−1]          (-vy, -wz)
    [12:24] qpos        -> mirror_joints
    [24:36] qvel        -> mirror_joints
    [36:48] last_act    -> mirror_joints
    [48:50] clock  × [−1,−1]            (half-period shift: sin(φ+0.5)=-sin(φ), cos(φ+0.5)=-cos(φ))
    [50:51] base_above  unchanged
    [51:96] scan        -> [..., SCAN_MIRROR_PERM]   (lateral column swap within each forward row)
"""
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spot"))
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))   # scratch_clock

from spot_terrain_env import N_SCAN, SCAN_MIRROR_PERM       # 2-D scan contract
from spot_symmetry import MIRROR_PERM, MIRROR_SIGN, mirror_joints, _consts, mirror_act  # noqa: F401
from scratch_clock import CLOCK0, CLOCK_DIM                 # clock slot constants

OBS_DIM = 96   # [proprio(48)|clock(2)|base_above(1)|scan(45)]
_SCAN0 = 51    # scan block starts at index 51 (= 48 proprio + 2 clock + 1 base_above)

_STEPS_CACHE = {}


def _steps_consts(device):
    """Cache all steps-mirror tensors. Extends spot_symmetry._consts with clock sign + scan."""
    key = str(device)
    if key not in _STEPS_CACHE:
        base = _consts(device)   # perm, jsign, lin, ang, cmd from spot_symmetry
        _STEPS_CACHE[key] = {
            **base,
            "clock_sign": torch.tensor([-1.0, -1.0], dtype=torch.float32, device=device),
            "scan": torch.tensor(SCAN_MIRROR_PERM, dtype=torch.long, device=device),
        }
    return _STEPS_CACHE[key]


def mirror_obs(obs: torch.Tensor) -> torch.Tensor:
    """Mirror the 96-d steps obs left<->right. obs: [..., 96]."""
    c = _steps_consts(obs.device)
    o = obs.clone()
    o[..., 0:3]  = obs[..., 0:3]  * c["lin"]                          # lin_b
    o[..., 3:6]  = obs[..., 3:6]  * c["ang"]                          # ang_b
    o[..., 6:9]  = obs[..., 6:9]  * c["lin"]                          # proj_g
    o[..., 9:12] = obs[..., 9:12] * c["cmd"]                          # cmd
    o[..., 12:24] = mirror_joints(obs[..., 12:24], c)                 # qpos
    o[..., 24:36] = mirror_joints(obs[..., 24:36], c)                 # qvel
    o[..., 36:48] = mirror_joints(obs[..., 36:48], c)                 # last_act
    # Clock [48:50]: negate both dims (half-period shift; see module docstring)
    o[..., CLOCK0:CLOCK0 + CLOCK_DIM] = (obs[..., CLOCK0:CLOCK0 + CLOCK_DIM]
                                          * c["clock_sign"])
    # [50:51] base_above: unchanged (scalar height, no lateral component)
    # [51:96] scan: lateral column swap within each forward row
    o[..., _SCAN0:_SCAN0 + N_SCAN] = obs[..., _SCAN0:_SCAN0 + N_SCAN][..., c["scan"]]
    return o


# mirror_act is re-exported from spot_symmetry (same 12-d Isaac joint mirror)


def symmetry_loss(ac, obs: torch.Tensor) -> torch.Tensor:
    """Equivariance penalty: actor(mirror(obs)) should equal mirror(actor(obs))."""
    feat   = ac._feat(obs)
    mean   = ac.actor(feat)
    mean_m = ac.actor(ac._feat(mirror_obs(obs)))
    return (mean_m - mirror_act(mean)).pow(2).mean()


def make_aux_loss(coef: float):
    """Build an aux_loss(ac, obs) -> scalar for PPO. obs is already normalized by PPO."""
    return lambda ac, obs: coef * symmetry_loss(ac, obs)


# --------------------------------------------------------------------------- #
#  Selftest (no PhysX needed)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import math
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running spot_steps_symmetry selftest on device={device}  OBS_DIM={OBS_DIM}")

    B = 64

    # ---- (a) Involution: mirror(mirror(x)) ≈ x ----
    x_obs = torch.randn(B, OBS_DIM, device=device)
    err_obs = (mirror_obs(mirror_obs(x_obs)) - x_obs).abs().max().item()
    a_rand  = torch.randn(B, 12, device=device)
    err_act = (mirror_act(mirror_act(a_rand)) - a_rand).abs().max().item()
    print(f"\n(a) Involution:")
    print(f"    mirror_obs(mirror_obs(x)) max err = {err_obs:.2e}  (expect < 1e-5)")
    print(f"    mirror_act(mirror_act(a)) max err = {err_act:.2e}  (expect < 1e-5)")
    assert err_obs < 1e-5, f"mirror_obs not an involution: err={err_obs}"
    assert err_act < 1e-5, f"mirror_act not an involution: err={err_act}"
    print("    -> PASS")

    # ---- (b) Clock-sign test: mirror maps [sin,cos] -> [-sin,-cos] ----
    print(f"\n(b) Clock-sign test:")
    phi    = torch.rand(B, device=device)
    phi_05 = (phi + 0.5) % 1.0
    angle     = 2.0 * math.pi * phi
    angle_05  = 2.0 * math.pi * phi_05
    c_phi     = torch.stack([angle.sin(),    angle.cos()],    dim=-1)   # [B,2]
    c_phi05   = torch.stack([angle_05.sin(), angle_05.cos()], dim=-1)   # [B,2] = [-sin,-cos]
    # Build two obs identical except for the clock block
    proprio = torch.randn(B, 48, device=device)
    base_ab = torch.randn(B, 1,  device=device)
    scan    = torch.randn(B, N_SCAN, device=device)
    obs_phi  = torch.cat([proprio, c_phi,   base_ab, scan], dim=-1)   # [B, 96]
    obs_ph05 = torch.cat([proprio, c_phi05, base_ab, scan], dim=-1)

    mirrored_clock = mirror_obs(obs_phi)[..., CLOCK0:CLOCK0 + CLOCK_DIM]
    target_clock   = obs_ph05[..., CLOCK0:CLOCK0 + CLOCK_DIM]
    err_clock = (mirrored_clock - target_clock).abs().max().item()
    print(f"    mirror_obs(obs_phi)[clock] ~= obs_{{phi+0.5}}[clock]: max err = {err_clock:.2e}")
    assert err_clock < 1e-5, f"clock mirror wrong: err={err_clock}"

    # Verify sign: c_mirror = -c_orig
    c_orig   = obs_phi[..., CLOCK0:CLOCK0 + CLOCK_DIM]
    c_mirror = mirror_obs(obs_phi)[..., CLOCK0:CLOCK0 + CLOCK_DIM]
    sign_err = (c_mirror + c_orig).abs().max().item()
    print(f"    mirror maps [sin,cos] -> [-sin,-cos]: max|c_mirror + c_orig| = {sign_err:.2e}")
    assert sign_err < 1e-5, f"clock sign wrong: err={sign_err}"
    print("    -> PASS")

    # ---- (c) base_above unchanged ----
    print(f"\n(c) base_above unchanged:")
    m = mirror_obs(x_obs)
    err_ba = (m[..., 50:51] - x_obs[..., 50:51]).abs().max().item()
    print(f"    max|mirrored_base_above - orig| = {err_ba:.2e}  (expect 0)")
    assert err_ba < 1e-7, f"base_above should be unchanged: err={err_ba}"
    print("    -> PASS")

    print("\nSPOT-STEPS-SYMMETRY SELFTEST: PASS")
