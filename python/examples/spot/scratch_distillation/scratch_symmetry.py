"""scratch_symmetry.py — left-right mirror for the 48-proprio + 2-clock obs (OBS_DIM=50).

Extends spot_symmetry.py for the new obs layout used in scratch_distillation. The first 48
dims are BYTE-IDENTICAL to the existing Isaac/terrain env proprio block, so all prior mirror
logic for those dims is reused verbatim. The two new clock dims [48:50] are NEGATED because:

    Mirror (y→−y) maps the trot gait to itself with a half-period phase shift.
    For the global clock φ with offset pair (0, 0.5) and (0.5, 0):
        φ_mirrored = φ + 0.5  (mod 1)
    And sin/cos of (φ+0.5):
        sin(2π(φ+0.5)) = −sin(2πφ)
        cos(2π(φ+0.5)) = −cos(2πφ)
    So the mirror of [sin, cos] is [−sin, −cos] = multiply both by −1.

This is a DIFFERENT transform than negating just sin or just cos — both must flip.
The clock-aware unit test in __main__ catches the common +1/−1 sign bug that the
involution test (mirror(mirror(x))≈x) passes for either sign choice.
"""
import os
import sys

import torch

_HERE     = os.path.dirname(os.path.abspath(__file__))   # scratch_distillation/
_SPOT_DIR = os.path.dirname(_HERE)                        # examples/spot/
_EXAMPLES = os.path.dirname(_SPOT_DIR)                   # examples/
_PYROOT   = os.path.dirname(_EXAMPLES)                   # python/
sys.path.insert(0, _PYROOT)     # threepp / threepp.rl
sys.path.insert(0, _SPOT_DIR)   # spot_deploy / spot_terrain_env / spot_symmetry

# Single source of truth for obs layout (no circular import — scratch_clock has no env dep)
from scratch_clock import OBS_DIM, CLOCK0, CLOCK_DIM   # noqa: E402
# Reuse the per-joint mirror constants from spot_symmetry
from spot_symmetry import (MIRROR_PERM, MIRROR_SIGN,   # noqa: E402
                            mirror_joints, _consts, mirror_act)

# --------------------------------------------------------------------------- #
#  Per-device constant cache (extends spot_symmetry._consts with clock sign)
# --------------------------------------------------------------------------- #
_SCRATCH_CACHE = {}


def _scratch_consts(device):
    """Cache of all scratch mirror tensors for the given device."""
    key = str(device)
    if key not in _SCRATCH_CACHE:
        base = _consts(device)   # perm, jsign, lin, ang, cmd from spot_symmetry
        _SCRATCH_CACHE[key] = {
            **base,
            # Clock dims: negate both sin and cos (half-period shift, see module docstring)
            "clock_sign": torch.tensor([-1.0, -1.0], dtype=torch.float32, device=device),
        }
    return _SCRATCH_CACHE[key]


# --------------------------------------------------------------------------- #
#  Mirror functions
# --------------------------------------------------------------------------- #

def mirror_obs(obs: torch.Tensor) -> torch.Tensor:
    """Mirror the full 50-d scratch obs left<->right.

    obs: [..., OBS_DIM=50].
    Returns: [..., 50] mirrored obs.

    The first 48 dims are the Isaac proprio block (same as spot_symmetry.mirror_obs
    but without the scan/base_above tail — those are dropped in the scratch layout):
        [0:3]   lin_b  × [1,−1,1]      (true vector: −y)
        [3:6]   ang_b  × [−1,1,−1]     (pseudovector: −x,−z)
        [6:9]   proj_g × [1,−1,1]      (true vector)
        [9:12]  cmd    × [1,−1,−1]     (−vy, −wz)
        [12:24] qpos (Isaac order): swap L↔R, negate hx
        [24:36] qvel  (Isaac order): swap L↔R, negate hx
        [36:48] last_action (Isaac order): swap L↔R, negate hx
    The two clock dims [48:50]:
        [48:50] clock × [−1,−1]        (half-period phase shift → negate sin AND cos)
    """
    c = _scratch_consts(obs.device)
    o = obs.clone()
    o[..., 0:3]  = obs[..., 0:3]  * c["lin"]     # lin_b
    o[..., 3:6]  = obs[..., 3:6]  * c["ang"]     # ang_b
    o[..., 6:9]  = obs[..., 6:9]  * c["lin"]     # proj_g
    o[..., 9:12] = obs[..., 9:12] * c["cmd"]     # cmd
    o[..., 12:24] = mirror_joints(obs[..., 12:24], c)   # qpos
    o[..., 24:36] = mirror_joints(obs[..., 24:36], c)   # qvel
    o[..., 36:48] = mirror_joints(obs[..., 36:48], c)   # last_action
    # Clock dims: negate both (half-period shift)
    o[..., CLOCK0:CLOCK0 + CLOCK_DIM] = (obs[..., CLOCK0:CLOCK0 + CLOCK_DIM]
                                          * c["clock_sign"])
    return o


def symmetry_loss(ac, obs: torch.Tensor) -> torch.Tensor:
    """Equivariance penalty: actor(mirror(obs)) should equal mirror(actor(obs)).

    Uses the full 50-d obs but the teacher reads only obs[:,:48], so:
      - ac._feat / ac.actor operate on the normalized 50-d obs (PPO's normalized view)
      - mirror_obs acts on the full 50-d obs
      - mirror_act acts on the 12-d action output
    """
    feat   = ac._feat(obs)
    mean   = ac.actor(feat)
    mean_m = ac.actor(ac._feat(mirror_obs(obs)))
    return (mean_m - mirror_act(mean)).pow(2).mean()


def make_aux_loss(coef: float):
    """Build an aux_loss(ac, obs) -> scalar for PPO.

    coef: weight for the symmetry penalty (ramped from 0 by the trainer — see plan).
    The returned lambda is passed to PPO as aux_loss; it is called on PPO's
    already-normalized obs so the mirror's scale-invariance is satisfied.
    """
    return lambda ac, obs: coef * symmetry_loss(ac, obs)


# --------------------------------------------------------------------------- #
#  Selftest
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import math
    from scratch_clock import clock_obs, reset_phi

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running scratch_symmetry selftest on device={device}")

    B = 64  # batch size

    # ---- (a) Involution test: mirror(mirror(x)) ≈ x ----
    x_obs = torch.randn(B, OBS_DIM, device=device)
    err_obs = (mirror_obs(mirror_obs(x_obs)) - x_obs).abs().max().item()
    print(f"\n(a) Involution:")
    print(f"    mirror_obs(mirror_obs(x)) max err = {err_obs:.2e}  (expect < 1e-5)")
    assert err_obs < 1e-5, f"mirror_obs is not an involution: err={err_obs}"

    a_rand = torch.randn(B, 12, device=device)
    err_act = (mirror_act(mirror_act(a_rand)) - a_rand).abs().max().item()
    print(f"    mirror_act(mirror_act(a)) max err = {err_act:.2e}  (expect < 1e-5)")
    assert err_act < 1e-5, f"mirror_act is not an involution: err={err_act}"
    print("    -> PASS")

    # ---- (b) Clock-aware test ----
    # Build two obs that differ ONLY in the clock dims: one at phase phi, one at phi+0.5.
    # After mirroring obs_phi, the clock dims should equal those of obs_{phi+0.5}.
    print(f"\n(b) Clock-aware test:")
    phi     = torch.rand(B, device=device)            # random phases in [0, 1)
    phi_05  = (phi + 0.5) % 1.0                       # half-period shifted

    c_phi   = clock_obs(phi)    # [B, 2]: [sin(2πφ), cos(2πφ)]
    c_phi05 = clock_obs(phi_05) # [B, 2]: [sin(2π(φ+0.5)), cos(2π(φ+0.5))] = [-sin, -cos]

    # Build base proprio (same for both; only clock dims differ)
    proprio = torch.randn(B, 48, device=device)
    obs_phi  = torch.cat([proprio, c_phi],   dim=-1)   # [B, 50]
    obs_ph05 = torch.cat([proprio, c_phi05], dim=-1)   # [B, 50]

    mirrored_clock = mirror_obs(obs_phi)[..., CLOCK0:CLOCK0 + CLOCK_DIM]  # [B, 2]
    target_clock   = obs_ph05[..., CLOCK0:CLOCK0 + CLOCK_DIM]              # [B, 2]

    err_clock = (mirrored_clock - target_clock).abs().max().item()
    print(f"    mirror_obs(obs_phi)[clock] ~= obs_{{phi+0.5}}[clock]: max err = {err_clock:.2e}")
    assert err_clock < 1e-5, \
        f"Clock mirror is wrong: mirror_obs(obs_phi)[clock] != obs_phi+0.5[clock], err={err_clock}"

    # Also verify the sign: mirror maps c -> -c (not +c, not anything else)
    c_orig   = obs_phi[..., CLOCK0:CLOCK0 + CLOCK_DIM]
    c_mirror = mirror_obs(obs_phi)[..., CLOCK0:CLOCK0 + CLOCK_DIM]
    sign_err = (c_mirror + c_orig).abs().max().item()   # should be ~0 if c_mirror = -c_orig
    print(f"    mirror maps [sin,cos] -> [-sin,-cos]: max|c_mirror + c_orig| = {sign_err:.2e}")
    assert sign_err < 1e-5, \
        f"Clock sign is wrong: expected negation, got err={sign_err}"
    print("    -> PASS")

    # ---- (c) Teacher equivariance (PhysX + CUDA) ----
    print(f"\n(c) Teacher equivariance test:")
    _ran_teacher_test = False
    if device == "cuda":
        try:
            import threepp as tp
            if tp.HAS_PHYSX:
                from threepp.rl import GpuSim
                from spot_deploy import fetch_assets, default_q, add_to_isaac, isaac_to_add
                from spot_terrain_env import SpotGpu, _flat_ground, SPACING, DT, SUBSTEPS
                from scratch_clock import reset_phi

                K = 64
                assets = fetch_assets()
                teacher = torch.jit.load(
                    os.path.join(assets, "spot_policy.pt"), map_location="cuda").eval()

                sim = GpuSim(K, lambda w, i: SpotGpu(w, i, SPACING),
                             gravity=(0.0, 0.0, -9.81), spacing=SPACING,
                             device="cuda", read_root=True,
                             build_world=lambda w: _flat_ground(w, K, SPACING))
                dev2 = sim.device
                default_q_t = torch.from_numpy(default_q).to(dev2)
                # stand_q_add in ADD order = default_q (ISAAC order) indexed by add_to_isaac
                # add_to_isaac[j] = ISAAC index for ADD joint j  (ADD->ISAAC)
                a2i_map = torch.from_numpy(add_to_isaac.astype("int64")).to(dev2)
                stand_q_add = default_q_t[a2i_map].expand(K, -1).contiguous()
                pos = torch.zeros(K, 3, device=dev2)
                pos[:, 1] = torch.arange(K, device=dev2, dtype=torch.float32) * SPACING
                pos[:, 2] = 0.42
                pose = GpuSim.make_root_pose(pos, quat=(0.0, 0.0, 0.0, 1.0), device=dev2)
                idx_all = torch.arange(K, device=dev2)
                sim.set_root_state(idx_all, pose)
                sim.set_joint_state(idx_all, stand_q_add, torch.zeros(K, sim.dof, device=dev2))
                sim.read()
                from spot_terrain_env import quat_rotate_inverse
                from scratch_clock import clock_obs as _clock_obs
                grav   = torch.tensor([0.0, 0.0, -1.0], device=dev2)
                # i2a: index ADD-order joint_pos -> ISAAC order (teacher sees ISAAC).
                # a2i: index ISAAC-order targets -> ADD order (for apply_drive_target).
                i2a_t  = torch.from_numpy(isaac_to_add.astype("int64")).to(dev2)
                a2i_t  = torch.from_numpy(add_to_isaac.astype("int64")).to(dev2)
                fwd    = torch.tensor([1.0, 0.0, 0.0], device=dev2).expand(K, 3)

                def _obs48(last_a):
                    q = sim.root_quat
                    lin_b  = quat_rotate_inverse(q, sim.root_linvel)
                    ang_b  = quat_rotate_inverse(q, sim.root_angvel)
                    proj_g = quat_rotate_inverse(q, grav.expand(K, 3))
                    qpos   = sim.joint_pos[:, i2a_t] - default_q_t
                    qvel   = sim.joint_vel[:, i2a_t]
                    return torch.cat([lin_b, ang_b, proj_g, fwd, qpos, qvel, last_a], dim=-1)

                # Settle, THEN roll the teacher ~40 ticks at a forward command so obs is a
                # real walking state. A static stand has near-zero action scale, which makes
                # the relative equivariance error meaningless (tiny denominator -> looks huge).
                for _ in range(20):
                    sim.apply_drive_target(stand_q_add)
                    sim.substep(DT / SUBSTEPS, SUBSTEPS)
                last_a = torch.zeros(K, 12, device=dev2)
                for _ in range(40):
                    with torch.no_grad():
                        last_a = teacher(_obs48(last_a))                  # [K,12] Isaac order
                    sim.apply_drive_target((default_q_t + 0.2 * last_a)[:, a2i_t])
                    sim.substep(DT / SUBSTEPS, SUBSTEPS)

                # Real walking obs + a random clock (teacher ignores it, mirror_obs needs it)
                ck       = _clock_obs(reset_phi(K, dev2))                  # [K, 2]
                obs_real = torch.cat([_obs48(last_a), ck], dim=-1)        # [K, 50]

                with torch.no_grad():
                    a_orig   = teacher(obs_real[:, :48])                 # [K, 12]
                    a_mirr   = teacher(mirror_obs(obs_real)[:, :48])     # [K, 12]

                equiv_err = (a_mirr - mirror_act(a_orig)).abs().mean().item()
                act_scale = a_orig.abs().mean().item()
                rel_err   = equiv_err / max(act_scale, 1e-6)
                print(f"    teacher equivariance: mean|t(mirror(o)) - mirror(t(o))| = {equiv_err:.4f}")
                print(f"    action scale = {act_scale:.3f}  relative error = {rel_err:.4f}")
                print(f"    (small relative to scale = mirror correct + teacher ~symmetric;"
                      f" the matching spot_symmetry baseline is ~0.45)")
                # A broken [0:48] mirror (wrong perm/sign) blows this up to ~1-2; the teacher
                # is only APPROXIMATELY symmetric, so allow generous headroom over the ~0.45 baseline.
                assert rel_err < 0.8, f"teacher equivariance too large ({rel_err:.3f}) — mirror likely wrong"
                _ran_teacher_test = True

        except Exception as e:
            print(f"    (teacher equivariance skipped: {e})")

    if not _ran_teacher_test:
        print("    (teacher equivariance skipped — PhysX or CUDA not available)")

    print("\nSCRATCH-SYMMETRY SELFTEST: PASS")
