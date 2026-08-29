"""Does the policy have more than one way of walking?

"It always walks the same way" is a claim you can measure. A gait is a frequency, a stride size and
a set of inter-leg phase offsets, so roll the policy at a range of commanded speeds on flat ground
and read those three off the joint traces:

  * cadence      dominant frequency of each knee angle (FFT) -> strides per second
  * stride       peak-to-peak knee excursion -> how far each step reaches
  * phase        offset between the two diagonal pairs -> trot (~0.5) vs pace/bound

A policy with one gait answers with one cadence at every speed and simply lengthens its stride. A
policy with a repertoire changes cadence too. That distinction is the whole question, and it is not
visible by eye in a viewer.

    python spot_gait_probe.py                          # the shipped checkpoint
    python spot_gait_probe.py --model spot_steps_v2.pt --model spot_steps.pt   # compare two

Flat ground only (`flat_only=True`): terrain would confound cadence with obstacle timing.
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))

import threepp as tp
from threepp.rl import load_policy
from spot_steps_env import SpotStepsEnv
from spot_terrain_env import DT, quat_rotate_inverse
from scratch_clock import GAIT_PERIOD
from scratch_calibrate import cross_corr_offset

# Octave errors are the whole difficulty here. A knee trace is rich in harmonics, so the dominant
# FFT bin is often 2f, and a doubled reading looks exactly like "the policy changed gait" when
# nothing changed at all. Two threshold-based attempts at folding it down both missed cases, so use
# AUTOCORRELATION, which finds the fundamental period directly: the first prominent peak is the
# stride, and its harmonics show up later, not earlier. The FFT peak is kept only as a coarse guard
# rail on the search range.
BAND_LO, BAND_HI = 0.4, 6.0          # plausible stride fundamentals (Hz)
ACF_MIN = 0.25                       # a lag must correlate at least this well to count as a period


def cadence(sig, dt):
    """Stride frequency (Hz) of a joint trace, via the first prominent autocorrelation peak."""
    x = np.asarray(sig, float)
    x = x - x.mean()
    if np.allclose(x, 0.0):
        return 0.0
    n = len(x)
    acf = np.correlate(x, x, mode="full")[n - 1:]
    acf /= acf[0]
    lo = max(1, int(round(1.0 / (BAND_HI * dt))))      # shortest lag we would believe
    hi = min(n - 2, int(round(1.0 / (BAND_LO * dt))))  # longest
    # Two failure directions, so two rules. A signal repeats perfectly only at its true period, so
    # acf(P) > acf(P/2) even when the second harmonic carries most of the energy — that rejects the
    # OCTAVE, which a "first peak over a threshold" rule walks straight into. But acf(2P) can tie
    # with acf(P), so among near-ties take the SHORTEST lag, which rejects the sub-octave. Getting
    # either wrong reads as "the policy changed gait" when nothing changed.
    if hi <= lo:
        return 0.0
    seg = acf[lo:hi]
    peaks = [lo + i for i in range(1, len(seg) - 1) if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1]]
    if not peaks:
        best = None
    else:
        top = max(acf[q] for q in peaks)
        best = min((q for q in peaks if acf[q] >= 0.95 * top), default=None)
        if best is not None and acf[best] < ACF_MIN:
            best = None
    if best is None:                                    # nothing periodic enough; fall back to the FFT
        power = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(n, d=dt)
        band = (freqs >= BAND_LO) & (freqs <= BAND_HI)
        if not band.any():
            return 0.0
        idx = np.where(band)[0]
        return float(freqs[idx[np.argmax(power[idx])]])
    best = max(lo + 1, min(best, hi - 2))
    # parabolic refinement around the integer lag, so the answer is not quantised to the sample rate
    y0, y1, y2 = acf[best - 1], acf[best], acf[best + 1]
    denom = y0 - 2 * y1 + y2
    shift = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    return 1.0 / ((best + shift) * dt)


KNEE = [3 * i + 2 for i in range(4)]          # add-order: [fl,fr,hl,hr] x (hx,hy,kn)
LEGS = ["fl", "fr", "hl", "hr"]
CMDS = [(0.25, 0.0, 0.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0), (1.5, 0.0, 0.0),
        (-0.5, 0.0, 0.0), (0.0, 0.0, 1.0)]


@torch.no_grad()
def probe(env, act, cmd, steps=400, warm=120, hold_period=None):
    """Roll one command and return (cadence Hz, stride rad, diag phase, achieved v_fwd)."""
    c = torch.tensor(cmd, device=env.device, dtype=torch.float32).expand(env.K, 3).contiguous()
    obs = env.reset()
    trace, vfwd = [], []
    for t in range(steps):
        env.cmd.copy_(c)
        env.cmd_timer.fill_(10 ** 9)              # never resample: hold this command
        if hold_period is not None:
            env.period.fill_(hold_period)         # reset() re-inits period; pin it every tick
        obs, _, _, _, _ = env.step(act(obs))
        if t >= warm:
            trace.append(env.sim.joint_pos[:, KNEE].mean(0).clone())
            lin_b = quat_rotate_inverse(env.sim.root_quat, env.sim.root_linvel)
            vfwd.append(lin_b[:, 0].mean().item())
    sig = torch.stack(trace).cpu().numpy()        # [T, 4]
    hz, amp = [], []
    for leg in range(4):
        hz.append(cadence(sig[:, leg], DT))
        amp.append(float(sig[:, leg].max() - sig[:, leg].min()))
    # diagonal pairs: {fl, hr} should swing together, {fr, hl} a half cycle later
    diag = cross_corr_offset(sig[:, 0], sig[:, 1])
    return float(np.median(hz)), float(np.median(amp)), float(diag), float(np.mean(vfwd))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=None, help="checkpoint(s); repeatable")
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--periods", default="0.35,0.50,0.65",
                    help="gait periods (s) to drive the clock at, comma separated. Cadence jitter "
                         "makes cadence a controllable INPUT rather than something the policy picks "
                         "from speed, so following a commanded clock is the capability to test — "
                         "and it is a different claim from 'chooses a gait to suit the speed'.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return 0
    models = args.model or ["spot_steps.pt"]

    torch.manual_seed(args.seed)
    env = SpotStepsEnv(num_envs=args.envs, seed=args.seed, flat_only=True, graph=True)
    for m in models:
        path = m if os.path.isabs(m) else os.path.join(_HERE, m)
        if not os.path.exists(path):
            print(f"\n{m}: not found"); continue
        ac, norm, meta = load_policy(path, device=env.device)
        ac.eval()
        act = (lambda o: ac.act_mean(norm.norm(o))) if norm is not None else ac.act_mean
        print(f"\n=== {os.path.basename(path)} "
              f"(cadence_jitter={meta.get('cadence_jitter', 'n/a')}, "
              f"imit_anneal={meta.get('imit_anneal', 'n/a')}) ===")
        print(f"{'command':>18} {'v_fwd':>7} {'cadence':>9} {'stride':>8} {'diag phase':>11}")
        rows = []
        for cmd in CMDS:
            hz, amp, diag, v = probe(env, act, cmd, steps=args.steps)
            rows.append((cmd, hz, amp, diag, v))
            print(f"  [{cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f}] {v:7.2f} {hz:8.2f}Hz "
                  f"{amp:8.3f} {diag:11.2f}")
        # Does it FOLLOW a commanded cadence? Drive the clock at several periods at one speed.
        periods = [float(v) for v in args.periods.split(",") if v.strip()]
        print(f"  {'period':>10} {'asked':>8} {'achieved':>10} {'v_fwd':>7} {'stride':>8}")
        for per in periods:
            env.period.fill_(per)
            hz, amp, _, v = probe(env, act, (1.0, 0.0, 0.0), steps=args.steps, hold_period=per)
            print(f"  {per:9.2f}s {1.0/per:7.2f}Hz {hz:9.2f}Hz {v:7.2f} {amp:8.3f}")
        env.period.fill_(GAIT_PERIOD)

        fwd = [r for r in rows if r[0][1] == 0.0 and r[0][2] == 0.0 and r[0][0] > 0]
        hzs = [r[1] for r in fwd]
        amps = [r[2] for r in fwd]
        span = (max(hzs) - min(hzs)) / max(np.mean(hzs), 1e-6) * 100
        aspan = (max(amps) - min(amps)) / max(np.mean(amps), 1e-6) * 100
        print(f"  over the forward commands: cadence varies {span:.1f}%, stride varies {aspan:.1f}%")
        print("  " + ("one gait, longer steps — cadence is pinned"
                      if span < 10.0 else "cadence adapts with speed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
