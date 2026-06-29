"""scratch_calibrate.py — gait-period calibration + end-to-end contact validation.

This is the decisive in-motion check of the whole contact pipeline.

Steps:
1. Build SpotScratchEnv(K=64) and load the frozen teacher.
2. Roll the teacher closed-loop at a fixed forward command cmd=[1,0,0] for ~700 ticks.
   Record contact_soft(foot_world(sim)[0][...,2]) -> a [T,K,4] trace.
3. Analyse (mean over envs, drop first ~150 warmup ticks):
   - Per-leg FFT of contact trace -> dominant frequency -> stride period in seconds.
   - Median recommended GAIT_PERIOD.
   - Cross-correlate diagonal pairs: confirm {fl(0),hr(3)} ~in-phase,
     {fr(1),hl(2)} ~in-phase, A vs B ~0.5 cycle apart.
   - Duty factor (mean contact fraction per leg).
4. Print summary + trot verdict.
5. If GAIT_PERIOD differs materially from 0.5 s, print the update line for scratch_clock.py.

Contact pipeline proof: contact_soft must clearly oscillate 0<->1 per foot during the
teacher roll. This gates R_tick inclusion.
"""
import os
import sys

import numpy as np
import torch

_HERE      = os.path.dirname(os.path.abspath(__file__))
_SPOT_DIR  = os.path.dirname(_HERE)
_EXAMPLES  = os.path.dirname(_SPOT_DIR)
_PYROOT    = os.path.dirname(_EXAMPLES)
sys.path.insert(0, _PYROOT)
sys.path.insert(0, _SPOT_DIR)

import threepp as tp
from spot_terrain_env import DT
from scratch_clock import GAIT_PERIOD, OBS_DIM
from foot_contact import foot_world, contact_soft
from scratch_env import SpotScratchEnv, ACT_DIM


def cross_corr_offset(a, b):
    """Estimate fractional-cycle phase offset of series b relative to a in [0, 1).

    Uses the FFT cross-correlation at the fundamental frequency of a.
    Returns phase_offset in cycles (0 = in-phase, 0.5 = antiphase).
    """
    n = len(a)
    fa = np.fft.rfft(a - a.mean())
    fb = np.fft.rfft(b - b.mean())
    # Dominant frequency of a
    dom_idx = int(np.argmax(np.abs(fa[1:])) + 1)
    # Phase of a and b at that frequency
    phase_a = np.angle(fa[dom_idx])
    phase_b = np.angle(fb[dom_idx])
    offset = (phase_b - phase_a) / (2.0 * np.pi)   # cycles
    offset = offset % 1.0
    # Wrap to [-0.5, 0.5) for signed offset
    if offset > 0.5:
        offset -= 1.0
    return offset


def fft_period(trace, dt):
    """Estimate the dominant stride period (seconds) from a 1-D contact trace.

    trace: [T] float array (contact signal, 0-1).
    dt   : seconds per sample.
    Returns (period_s, peak_freq_hz, peak_power).
    """
    n = len(trace)
    sig = trace - trace.mean()
    ft = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, d=dt)
    # Skip DC (index 0)
    power = np.abs(ft[1:]) ** 2
    dom = int(np.argmax(power))
    freq = freqs[dom + 1]
    period = 1.0 / freq if freq > 0 else float("inf")
    return period, freq, power[dom]


def main():
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)

    K = 64
    TICKS  = 700     # total teacher rollout ticks
    WARMUP = 150     # discard first N ticks (settlement transient)

    print(f"Building SpotScratchEnv (K={K})...")
    env = SpotScratchEnv(num_envs=K, tick_enabled=False)
    teacher = env.imit_policy
    dev     = env.sim.device

    # --- Step 2: roll frozen teacher at fixed forward command ---
    print(f"Resetting + rolling teacher at cmd=[1,0,0] for {TICKS} ticks...")
    obs    = env.reset()
    FWD_CMD = torch.tensor([[1.0, 0.0, 0.0]], device=dev, dtype=torch.float32).expand(K, 3).contiguous()

    # contact trace: [T, K, 4]
    contact_trace = []
    tip_z_trace   = []

    for t in range(TICKS):
        # Freeze command
        env.cmd        = FWD_CMD
        env.cmd_timer.fill_(10 ** 9)

        # Teacher acts on raw obs[:, :48]
        a   = teacher(obs[:, :48])
        obs, _, _, _, _ = env.step(a)

        # Record contact
        tip_pos, _ = foot_world(env.sim)         # [K, 4, 3]
        tip_z_now  = tip_pos[..., 2]             # [K, 4]
        contact_now = contact_soft(tip_z_now)    # [K, 4]
        contact_trace.append(contact_now.cpu().float())
        tip_z_trace.append(tip_z_now.cpu().float())

    contact_trace = torch.stack(contact_trace, dim=0)   # [T, K, 4]
    tip_z_trace   = torch.stack(tip_z_trace, dim=0)     # [T, K, 4]
    print(f"Rollout complete. contact trace shape: {tuple(contact_trace.shape)}")

    # --- Step 3: analyse ---
    # Use mean over envs, drop warmup.
    contact_mean = contact_trace[WARMUP:].mean(dim=1).numpy()   # [T-WARMUP, 4]
    tip_z_mean   = tip_z_trace[WARMUP:].mean(dim=1).numpy()     # [T-WARMUP, 4]
    T_anal = contact_mean.shape[0]

    leg_names = ["fl", "fr", "hl", "hr"]
    periods, freqs, duties = [], [], []
    print(f"\n--- Per-leg FFT analysis (after {WARMUP}-tick warmup, T={T_anal} ticks) ---")
    for leg in range(4):
        sig = contact_mean[:, leg]
        period_s, freq_hz, power = fft_period(sig, DT)
        duty  = float(sig.mean())
        periods.append(period_s)
        freqs.append(freq_hz)
        duties.append(duty)
        min_c = float(sig.min())
        max_c = float(sig.max())
        print(f"  {leg_names[leg]}: period={period_s:.3f}s  freq={freq_hz:.2f}Hz  "
              f"duty={duty:.2f}  range=[{min_c:.2f},{max_c:.2f}]")

    # Contact clearly oscillates?
    contact_range = [float((contact_trace[WARMUP:, :, leg].max() - contact_trace[WARMUP:, :, leg].min()).item())
                     for leg in range(4)]
    print(f"\n  Contact ranges (max-min per leg, over all envs+time): "
          f"{[round(r, 3) for r in contact_range]}")
    contact_oscillates = all(r > 0.3 for r in contact_range)
    print(f"  Contact clearly oscillates (range>0.3 each leg): {contact_oscillates}")

    # Tip-z range confirms physical movement.
    tipz_range = [float(tip_z_trace[WARMUP:, :, leg].max() - tip_z_trace[WARMUP:, :, leg].min())
                  for leg in range(4)]
    print(f"  Tip-z ranges [m]: {[round(r, 3) for r in tipz_range]}")

    # Recommended GAIT_PERIOD: median over legs.
    valid_periods = [p for p in periods if 0.1 < p < 2.0]
    if valid_periods:
        rec_period = float(np.median(valid_periods))
    else:
        rec_period = GAIT_PERIOD   # fallback
    print(f"\n  Recommended GAIT_PERIOD: {rec_period:.4f} s  "
          f"(current default: {GAIT_PERIOD:.4f} s)")

    # --- Cross-correlate diagonal pairs ---
    print("\n--- Diagonal phase analysis ---")
    # Diagonal pair A: fl(0) + hr(3) — should be in-phase (~0 offset).
    # Diagonal pair B: fr(1) + hl(2) — should be in-phase (~0 offset).
    # A vs B: should be ~0.5 cycle apart.
    fl_sig = contact_mean[:, 0]
    fr_sig = contact_mean[:, 1]
    hl_sig = contact_mean[:, 2]
    hr_sig = contact_mean[:, 3]

    off_fl_hr = cross_corr_offset(fl_sig, hr_sig)   # fl vs hr  (same diagonal -> ~0)
    off_fr_hl = cross_corr_offset(fr_sig, hl_sig)   # fr vs hl  (same diagonal -> ~0)
    off_A_B   = cross_corr_offset(fl_sig, fr_sig)   # fl vs fr  (opposite diagonals -> ~0.5)

    print(f"  fl vs hr phase offset: {off_fl_hr:+.3f} cycles  (expect ~0)")
    print(f"  fr vs hl phase offset: {off_fr_hl:+.3f} cycles  (expect ~0)")
    print(f"  fl vs fr phase offset: {off_A_B:+.3f} cycles   (expect ~±0.5)")

    diag_A_in_phase = abs(off_fl_hr) < 0.20
    diag_B_in_phase = abs(off_fr_hl) < 0.20
    diag_split_half = 0.35 < abs(off_A_B) < 0.65

    print(f"\n  Pair A (fl+hr) in-phase:  {diag_A_in_phase} (|offset|={abs(off_fl_hr):.3f})")
    print(f"  Pair B (fr+hl) in-phase:  {diag_B_in_phase} (|offset|={abs(off_fr_hl):.3f})")
    print(f"  A vs B split ~0.5 cycle:  {diag_split_half} (|offset|={abs(off_A_B):.3f})")

    # --- Duty ---
    mean_duty = float(np.mean(duties))
    duty_ok   = 0.45 < mean_duty < 0.65
    print(f"\n  Mean duty factor: {mean_duty:.3f}  (expect [0.45, 0.65]): {duty_ok}")
    print(f"  Per-leg duties: fl={duties[0]:.3f}  fr={duties[1]:.3f}  "
          f"hl={duties[2]:.3f}  hr={duties[3]:.3f}")

    # --- Verdict ---
    is_clean_trot = (diag_A_in_phase and diag_B_in_phase and diag_split_half
                     and duty_ok and contact_oscillates)

    print("\n" + "=" * 60)
    print("CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"  Recommended GAIT_PERIOD : {rec_period:.4f} s")
    print(f"  Diagonal phase split    : {abs(off_A_B):.3f} cycles (pair A vs B)")
    print(f"  Mean duty factor        : {mean_duty:.3f}")
    print(f"  Contact oscillates 0<->1: {contact_oscillates}")
    print(f"  Diagonal A (fl+hr) in-phase : {diag_A_in_phase}")
    print(f"  Diagonal B (fr+hl) in-phase : {diag_B_in_phase}")
    if is_clean_trot:
        print("\n  VERDICT: TEACHER IS A CLEAN DIAGONAL TROT")
        print("  -> R_tick is appropriate; proceed with tick_enabled=True run.")
    else:
        print("\n  VERDICT: TEACHER GAIT IS NOT A CLEAN TROT")
        print("  -> recommend obs-clock-only baseline, drop R_tick.")
    print("=" * 60)

    # --- GAIT_PERIOD update recommendation ---
    if abs(rec_period - GAIT_PERIOD) > 0.05:
        print(f"\n  NOTE: recommended GAIT_PERIOD ({rec_period:.4f} s) differs from "
              f"the current default ({GAIT_PERIOD:.4f} s) by "
              f"{abs(rec_period - GAIT_PERIOD)*1000:.0f} ms.")
        print(f"  To update scratch_clock.py, change line:")
        print(f"      GAIT_PERIOD = {GAIT_PERIOD}   # seconds; calibrated default ...")
        print(f"  to:")
        print(f"      GAIT_PERIOD = {rec_period:.4f}   # calibrated by scratch_calibrate.py")
        print(f"  (Do NOT auto-edit; review the calibration trace first.)")
    else:
        print(f"\n  GAIT_PERIOD {GAIT_PERIOD:.4f} s is within 50 ms of the measured "
              f"{rec_period:.4f} s — no update needed.")

    return {
        "rec_period": rec_period,
        "diag_split": abs(off_A_B),
        "duty": mean_duty,
        "is_clean_trot": is_clean_trot,
        "contact_oscillates": contact_oscillates,
    }


if __name__ == "__main__":
    result = main()
