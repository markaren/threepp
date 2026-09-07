"""scratch_clock.py — trot phase clock + observation/action contract constants.

Single source of truth for the obs/action layout used by all scratch_distillation modules.
The policy observes a single global clock phi ∈ [0,1); the clock is advanced AFTER each
physics substep so phi aligns with the observation the policy next sees.

Trot diagonal offsets (fl,fr,hl,hr) = (0, 0.5, 0.5, 0):
  - fl and hr swing together (diagonal pair A)
  - fr and hl swing together (diagonal pair B), half a period offset from A

These offsets live ONLY in the reward (desired_stance). The obs carries the single global phi
encoded as [sin(2π·phi), cos(2π·phi)] so the policy can self-phase-lock to it.

GAIT_PERIOD is a calibrated default (0.5 s); scratch_calibrate.py will refine it via an FFT
of the teacher's foot-z trace under a forward command.
"""
import math
import os
import sys

import torch

_HERE     = os.path.dirname(os.path.abspath(__file__))   # scratch_distillation/
_SPOT_DIR = os.path.dirname(_HERE)                        # examples/spot/
_EXAMPLES = os.path.dirname(_SPOT_DIR)                   # examples/
_PYROOT   = os.path.dirname(_EXAMPLES)                   # python/
sys.path.insert(0, _PYROOT)     # threepp / threepp.rl
sys.path.insert(0, _SPOT_DIR)   # spot_deploy / spot_terrain_env / spot_symmetry

# --------------------------------------------------------------------------- #
#  Obs / action contract (single source of truth — imported by scratch_symmetry
#  and scratch_env to avoid circular imports)
# --------------------------------------------------------------------------- #
N_PROPRIO = 48      # byte-identical to the Isaac 48-d block; teacher reads obs[:, :48]
CLOCK_DIM = 2       # [sin(2π·phi), cos(2π·phi)]
CLOCK0    = 48      # index where the clock dims start in the full obs
OBS_DIM   = 50      # N_PROPRIO + CLOCK_DIM
ACT_DIM   = 12      # 12 joint targets (Isaac order)

# --------------------------------------------------------------------------- #
#  Clock parameters
# --------------------------------------------------------------------------- #
GAIT_PERIOD = 0.5   # seconds; calibrated default — scratch_calibrate.py will refine this

# Trot diagonal offsets for legs (fl, fr, hl, hr):
#   diagonal pair A: fl (offset 0) and hr (offset 0) swing together
#   diagonal pair B: fr (offset 0.5) and hl (offset 0.5) swing together, a half-period late
LEG_OFFSETS = (0.0, 0.5, 0.5, 0.0)   # (fl, fr, hl, hr)

DUTY  = 0.55   # stance fraction (slightly > 0.5 — no all-feet-off instant while learning)
KAPPA = 12.0   # sharpness of the sigmoid stance window (higher = boxier, lower = softer)

# Pull DT from spot_terrain_env to stay consistent with the rest of the chain
from spot_terrain_env import DT as DT  # noqa: E402  (import after sys.path setup)


# --------------------------------------------------------------------------- #
#  Clock functions
# --------------------------------------------------------------------------- #

def advance(phi: torch.Tensor, dt: float = DT) -> torch.Tensor:
    """Advance the phase by one control tick.

    phi: [K] in [0, 1). Returns the new phi [K] in [0, 1).
    Call AFTER the physics substep so the returned phi aligns with the next obs.
    """
    return (phi + dt / GAIT_PERIOD) % 1.0


def clock_obs(phi: torch.Tensor) -> torch.Tensor:
    """Encode global clock as [sin(2π·phi), cos(2π·phi)].

    phi: [K] in [0, 1). Returns [K, 2].
    The sin/cos encoding is smooth at phi=1→0 (no discontinuity), and the two dims
    together uniquely identify the phase in [0,1) — unlike sin alone, which is
    ambiguous for phi vs 0.5-phi.
    """
    angle = 2.0 * math.pi * phi
    return torch.stack([angle.sin(), angle.cos()], dim=-1)


def leg_phase(phi: torch.Tensor) -> torch.Tensor:
    """Per-leg phase from the global clock and the trot offsets.

    phi: [K]. Returns [K, 4] (fl, fr, hl, hr), each in [0, 1).
    """
    offsets = phi.new_tensor(LEG_OFFSETS)           # [4]
    return (phi[:, None] + offsets[None, :]) % 1.0  # [K, 4]


def desired_stance(phi: torch.Tensor) -> torch.Tensor:
    """Smooth stance membership in [0, 1] for each leg, wrap-safe.

    phi: [K]. Returns [K, 4].

    Formula (wrap-safe smooth box over [0, DUTY)):
      For each leg's phase phi_i (including the trot offset):
        d  = phi_i if phi_i <= 0.5 else phi_i - 1.0   (wrapped distance to 0 on the circle)
        d2 = DUTY/2 - |d|                               (positive inside the stance window)
        stance = sigmoid(KAPPA * (d + DUTY/2)) * sigmoid(KAPPA * (DUTY/2 - d))
               = sigmoid(KAPPA * (phi_i))   *   sigmoid(KAPPA * (DUTY - phi_i))
      but evaluated on the wrapped distance so the window is centred at 0 (= start of
      stance) and wraps cleanly at phi=1→0.

    Implementation: wrap phi_i into [-0.5, 0.5) (shortest arc to 0), then the two edges
    of the stance box are at -DUTY/2 and +DUTY/2. A stance pixel has both edges positive.
    """
    phi_i = leg_phase(phi)                  # [K, 4] in [0, 1)
    # Map to [-0.5, 0.5): positive on "near 0 side", negative on "near 0.5 side"
    # This centres the stance window at phi=0 and handles the wrap automatically.
    d = phi_i - (phi_i > 0.5).float()      # phi_i > 0.5 wraps to phi_i - 1 ∈ [-0.5, 0)
    half = DUTY / 2.0
    # Two sigmoid gates: left edge at -half, right edge at +half
    s_left  = torch.sigmoid(KAPPA * (d + half))   # 1 when d > -half
    s_right = torch.sigmoid(KAPPA * (half - d))   # 1 when d <  half
    return s_left * s_right                        # [K, 4] in (0, 1)


def reset_phi(n: int, device) -> torch.Tensor:
    """Initialise n independent clocks uniformly in [0, 1).

    Random reset decorrelates episodes in the batch so the policy can't infer its
    episode age from the phase, and prevents all environments from synchronising to
    the same gait phase at reset.
    """
    return torch.rand(n, device=device)


# --------------------------------------------------------------------------- #
#  CPU selftest (no PhysX needed)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import math

    device = "cpu"
    K = 32

    # ---- 1. clock_obs is continuous across the wrap ----
    phi_lo = torch.full((K,), 0.001)
    phi_hi = torch.full((K,), 0.999)
    c_lo = clock_obs(phi_lo)
    c_hi = clock_obs(phi_hi)
    # The wrap from 0.999 to 0.001 is a step of 0.002 in phase (= 4π·0.001 in angle).
    # sin/cos must be nearly equal (< 0.03 tolerance for delta_phi ≈ 0.002).
    gap = (c_lo - c_hi).abs().max().item()
    print(f"[1] clock_obs wrap continuity: max|c(0.001) - c(0.999)| = {gap:.4f}  (expect < 0.03)")
    assert gap < 0.03, f"discontinuity at wrap: {gap}"

    # ---- 2. desired_stance has diagonal trot structure ----
    # At phi=0: fl and hr (offsets 0,0) should be in stance; fr and hl (offset 0.5) in swing.
    phi_zero = torch.zeros(1)
    s = desired_stance(phi_zero)   # [1, 4]
    fl, fr, hl, hr = s[0, 0].item(), s[0, 1].item(), s[0, 2].item(), s[0, 3].item()
    print(f"[2] stance at phi=0: fl={fl:.3f} fr={fr:.3f} hl={hl:.3f} hr={hr:.3f}")
    print(f"    expect: fl~1, fr~0, hl~0, hr~1 (trot diagonal A)")
    assert fl > 0.85 and hr > 0.85, "fl and hr should be in stance at phi=0"
    assert fr < 0.15 and hl < 0.15, "fr and hl should be in swing at phi=0"

    # At phi=0.5: diagonal pair B in stance; diagonal pair A in swing
    phi_half = torch.full((1,), 0.5)
    s2 = desired_stance(phi_half)
    fl2, fr2, hl2, hr2 = s2[0, 0].item(), s2[0, 1].item(), s2[0, 2].item(), s2[0, 3].item()
    print(f"    stance at phi=0.5: fl={fl2:.3f} fr={fr2:.3f} hl={hl2:.3f} hr={hr2:.3f}")
    print(f"    expect: fl~0, fr~1, hl~1, hr~0 (trot diagonal B)")
    assert fr2 > 0.85 and hl2 > 0.85, "fr and hl should be in stance at phi=0.5"
    assert fl2 < 0.15 and hr2 < 0.15, "fl and hr should be in swing at phi=0.5"

    # ---- 3. advance wraps correctly ----
    phi_near1 = torch.full((4,), 0.99)
    phi_adv = advance(phi_near1)
    assert (phi_adv >= 0).all() and (phi_adv < 1).all(), "advanced phi out of [0,1)"
    print(f"[3] advance: phi=0.99 -> {phi_adv[0].item():.4f}  (wraps mod 1, expect ~{(0.99 + DT/GAIT_PERIOD) % 1.0:.4f})")

    # ---- 4. reset_phi is uniform in [0,1) ----
    phis = reset_phi(10000, device)
    assert (phis >= 0).all() and (phis < 1).all(), "reset_phi out of [0,1)"
    mean = phis.mean().item()
    print(f"[4] reset_phi(10000): mean={mean:.4f}  (expect ~0.50)")
    assert abs(mean - 0.5) < 0.02, f"reset_phi mean too far from 0.5: {mean}"

    # ---- 5. Print a small stance table ----
    print("\nStance schedule (phi -> fl, fr, hl, hr):")
    for v in [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]:
        p = torch.full((1,), v)
        sc = desired_stance(p)[0]
        print(f"  phi={v:.3f}: fl={sc[0]:.3f} fr={sc[1]:.3f} hl={sc[2]:.3f} hr={sc[3]:.3f}")

    print("\nSCRATCH-CLOCK SELFTEST: PASS")
