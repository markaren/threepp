"""train_scratch.py — From-scratch teacher-guided phase-RL trainer for Spot flat-terrain.

NO warm-start: the actor is trained from random initialisation. The Isaac teacher is used
ONLY as a reward oracle (imitation penalty); its weights never enter the student network.

Usage:
    # Full training run (from-scratch, obs-clock-only baseline):
    python train_scratch.py --envs 4096 --iters 4000

    # Obs-clock-only baseline (no Siekmann tick reward):
    python train_scratch.py --envs 4096 --iters 4000 --no-tick

    # Steering regression eval (norm-aware — normalises obs before feeding the student):
    python train_scratch.py --eval scratch_flat.pt

    # Score/periodicity snapshot (norm-aware):
    python train_scratch.py --score scratch_flat.pt

Design notes (vs train_spot_terrain.py):
  - normalize_obs=True:  the student sees RUNNING-NORM obs, so all eval paths MUST call
    norm.norm(o) before ac.act_mean(o).  See eval_flat_scratch / score_checkpoint below.
  - normalize_returns=True: value head works regardless of absolute reward scale.
  - Symmetry aux coef is a MUTABLE cell updated from on_log so PPO sees the live ramp.
  - set_iter() and the sym coef update happen every log_every=10 iters — ~10-iter
    schedule-update granularity; schedules span hundreds of iters so the lag is negligible.
  - CSV log: one row per log_every block; flush each write (no Tensorboard dependency).
  - Checkpoint gate: best track subject to fell <= fell_max AND drift <= drift_max.
"""
import argparse
import csv
import os
import sys

import torch

_HERE     = os.path.dirname(os.path.abspath(__file__))
_SPOT_DIR = os.path.dirname(_HERE)          # examples/spot/
_EXAMPLES = os.path.dirname(_SPOT_DIR)      # examples/
_PYROOT   = os.path.dirname(_EXAMPLES)      # python/
sys.path.insert(0, _PYROOT)                 # threepp / threepp.rl
sys.path.insert(0, _SPOT_DIR)               # spot_deploy / spot_terrain_env
sys.path.insert(0, _HERE)                   # scratch_env / scratch_symmetry / scratch_clock

import threepp as tp
from scratch_env import SpotScratchEnv, OBS_DIM, ACT_DIM, HIDDEN, CONFIG
import scratch_symmetry
from threepp.rl import PPO, load_policy


# --------------------------------------------------------------------------- #
#  Norm-aware eval: flat-steering regression vs the Isaac teacher
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_flat_scratch(policy_path, k=512, device="cuda"):
    """Held-out steering regression — NORM-AWARE student arm.

    Loads the checkpoint, wraps the student as ac.act_mean(norm.norm(o)),
    and compares against the teacher (raw obs[:,:48]).

    PASS threshold: worst ratio <= 1.15 (relaxed from terrain's 1.10 — no
    weights were copied so absolute steering starts weaker).

    Prints absolute student error per command so teacher-inherited weaknesses
    remain visible even when all ratios pass.
    """
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return
    ac, norm, meta = _load(policy_path, device)
    assert norm is not None, (
        "expected a RunningNorm (normalize_obs=True) but checkpoint has norm=None; "
        "this policy was trained without obs normalisation — feeding raw obs to the "
        "student would produce garbage actions."
    )
    env = SpotScratchEnv(num_envs=k, device=device, tick_enabled=False)
    teacher = env.imit_policy

    # Student arm: NORMALISE before forwarding (critical — net sees RunningNorm obs)
    pol = lambda o: ac.act_mean(norm.norm(o))
    # Teacher arm: raw obs[:,:48] only (stateless; no clock block)
    tea = lambda o: teacher(o[:, :48])

    grid = [
        (1.0,  0.0,  0.0),
        (0.0,  0.0,  0.0),
        (-0.5, 0.0,  0.0),
        (0.0,  0.5,  0.0),
        (0.0, -0.5,  0.0),
        (0.0,  0.0,  1.0),
        (0.0,  0.0, -1.0),
        (1.0,  0.0,  0.5),
    ]
    print(f"flat-steering regression ({os.path.basename(policy_path)} vs Isaac teacher, K={k}):")
    print("   cmd[vx,vy,wz]      student_err  teacher_err   ratio")
    worst = 0.0
    for cmd in grid:
        ep = env.measure_tracking(pol, cmd)
        et = env.measure_tracking(tea, cmd)
        ratio = ep / max(et, 1e-6)
        worst = max(worst, ratio)
        flag = "" if ratio <= 1.15 else "  <- REGRESSED"
        print(f"   [{cmd[0]:+.1f},{cmd[1]:+.1f},{cmd[2]:+.1f}]"
              f"     {ep:8.3f}    {et:8.3f}    {ratio:5.2f}{flag}")
    print(f"worst ratio {worst:.2f}  ->  "
          f"{'PASS (steering ok vs teacher)' if worst <= 1.15 else 'FAIL (steering degraded)'}")


# --------------------------------------------------------------------------- #
#  Norm-aware score / periodicity snapshot
# --------------------------------------------------------------------------- #
@torch.no_grad()
def score_checkpoint(policy_path, k=512, steps=900, warm=200, device="cuda"):
    """Deterministic snapshot: norm-aware rollout, accumulate metrics.

    Prints mean track, fell/step, drift, and per-leg duty (periodicity sanity).
    All eval without tick_enabled so it measures the pure steering/stability.
    """
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return
    ac, norm, meta = _load(policy_path, device)
    assert norm is not None, (
        "expected a RunningNorm; checkpoint was saved with normalize_obs=False which "
        "is incompatible with this trainer's norm-aware eval."
    )
    env = SpotScratchEnv(num_envs=k, device=device, tick_enabled=False)
    pol = lambda o: ac.act_mean(norm.norm(o))

    obs = env.reset()
    track_acc, fell_acc, drift_acc = [], [], []
    duty_acc = [[], [], [], []]
    for t in range(steps):
        obs, _, _, _, _ = env.step(pol(obs))
        if t >= warm:
            track_acc.append(env.last_track)
            fell_acc.append(env.last_fell)
            drift_acc.append(env.last_drift)
            for leg in range(4):
                duty_acc[leg].append(env.last_duty[leg])

    n = max(1, len(track_acc))
    mean_track = sum(track_acc) / n
    mean_fell  = sum(fell_acc)  / n
    mean_drift = sum(drift_acc) / n
    mean_duty  = [sum(duty_acc[l]) / n for l in range(4)]

    print(f"score_checkpoint  ({os.path.basename(policy_path)}, K={k}, "
          f"steps={steps}, warm={warm}):")
    print(f"  mean track   = {mean_track:.4f}")
    print(f"  fell/step    = {mean_fell:.4f}  "
          f"{'OK' if mean_fell <= 0.01 else 'HIGH'}")
    print(f"  lateral drift= {mean_drift:.4f}  "
          f"{'OK' if mean_drift <= 0.03 else 'HIGH'}")
    print(f"  duty (fl,fr,hl,hr) = "
          f"{[round(d, 3) for d in mean_duty]}  "
          f"(periodicity sanity: trot ~0.5 each)")


# --------------------------------------------------------------------------- #
#  Internal: schedule helpers
# --------------------------------------------------------------------------- #
def _sym_coef(it: int, plateau: float) -> float:
    """sym_coef(it): 0 for it<400; linear 0->plateau over [400,1000]; hold after.

    Ramped late so the RunningNorm has converged (>400 iters) before the
    equivariance loss operates on normalised obs, and to avoid fighting the
    pre-gait random policy during early imit-dominant phase.
    """
    if it < 400:
        return 0.0
    if it < 1000:
        return plateau * (it - 400) / 600.0
    return plateau


def _parse_iter(msg: str, fallback: int) -> int:
    """Parse the PPO log msg 'it NNNN | ...' -> integer iter.

    Falls back to `fallback` on any parse error (so a format change doesn't
    crash the training loop).
    """
    try:
        return int(msg.split()[1])
    except Exception:
        return fallback


def _load(path: str, device: str):
    """Thin wrapper around load_policy that works even though load_policy
    returns (ac, norm, meta) already — here just to keep the eval fns clean."""
    return load_policy(path, device)


# --------------------------------------------------------------------------- #
#  main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="From-scratch phase-RL trainer for Spot flat-terrain (no warm-start).")
    ap.add_argument("--envs",        type=int,   default=4096,
                    help="Number of parallel envs (default 4096; use 2048 if VRAM-bound)")
    ap.add_argument("--iters",       type=int,   default=4000)
    ap.add_argument("--horizon",     type=int,   default=32)
    ap.add_argument("--lr",          type=float, default=3e-4)
    ap.add_argument("--entropy",     type=float, default=3e-3,
                    help="Entropy bonus — floor prevents stand-and-collect collapse")
    ap.add_argument("--log_std_init",type=float, default=-0.5,
                    help="Initial log_std — more exploration than warm-start (-1.5)")
    ap.add_argument("--sym_coef",    type=float, default=0.5,
                    help="Symmetry aux-loss plateau weight (ramped in over [400,1000])")
    ap.add_argument("--no-tick",     dest="no_tick", action="store_true",
                    help="Disable Siekmann tick reward (obs-clock-only baseline)")
    ap.add_argument("--out",         default=os.path.join(_HERE, "scratch_flat.pt"),
                    help="Output checkpoint path (best-under-gate checkpoint)")
    ap.add_argument("--eval",        default="",
                    help="Path to a .pt checkpoint -> run norm-aware flat-steering eval and exit")
    ap.add_argument("--score",       default="",
                    help="Path to a .pt checkpoint -> run norm-aware score snapshot and exit")
    ap.add_argument("--fell_max",    type=float, default=0.01,
                    help="Gate: checkpoint only saved if fell/step <= this")
    ap.add_argument("--drift_max",   type=float, default=0.03,
                    help="Gate: checkpoint only saved if lateral drift <= this")
    ap.add_argument("--seed",        type=int,   default=0)
    ap.add_argument("--resume",      type=str,   default="",
                    help="Path to a scratch checkpoint to continue from (empty = fresh run)")
    ap.add_argument("--iter_offset", type=int,   default=-1,
                    help="Global curriculum offset when resuming; -1 = read from checkpoint meta iters_done")
    ap.add_argument("--total_iters", type=int,   default=-1,
                    help="Total iters for resume LR anneal; -1 = iter_offset + args.iters")
    args = ap.parse_args()

    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)

    # ---------------------------------------------------------------------- #
    #  Short-circuit modes
    # ---------------------------------------------------------------------- #
    if args.score:
        score_checkpoint(args.score)
        return
    if args.eval:
        eval_flat_scratch(args.eval)
        return

    # ---------------------------------------------------------------------- #
    #  Build env + PPO
    # ---------------------------------------------------------------------- #
    tick_enabled = not args.no_tick
    env = SpotScratchEnv(
        num_envs=args.envs,
        device="cuda",
        seed=args.seed,
        tick_enabled=tick_enabled,
    )

    # Mutable symmetry-coef cell: the aux_loss closure reads the LIVE value so the
    # trainer can ramp it from on_log without rebuilding PPO.  This mirrors how
    # make_aux_loss(coef) works but lets us update coef dynamically.
    coef_cell = [0.0]

    def sym_loss(ac, obs):
        return coef_cell[0] * scratch_symmetry.symmetry_loss(ac, obs)

    # ---------------------------------------------------------------------- #
    #  Resume flag: manual LR anneal when resuming so we don't restart LR high
    # ---------------------------------------------------------------------- #
    manual_lr = bool(args.resume)

    ppo = PPO(
        env,
        ACT_DIM,
        hidden=HIDDEN,
        lr=args.lr,
        horizon=args.horizon,
        log_std_init=args.log_std_init,
        entropy=args.entropy,
        normalize_obs=True,
        normalize_returns=True,
        anneal_lr=not manual_lr,
        target_kl=0.02,
        meta=CONFIG,
        aux_loss=sym_loss,
    )

    # ---------------------------------------------------------------------- #
    #  Resume loading (restore weights + RunningNorm from a prior checkpoint)
    # ---------------------------------------------------------------------- #
    OFFSET = 0
    if args.resume:
        rac, rnorm, rmeta = load_policy(args.resume, device="cuda")
        assert rmeta.get("obs_dim") == OBS_DIM and rmeta.get("act_dim") == ACT_DIM, \
            f"arch mismatch: checkpoint obs_dim={rmeta.get('obs_dim')} act_dim={rmeta.get('act_dim')} " \
            f"vs expected obs_dim={OBS_DIM} act_dim={ACT_DIM}"
        ppo.ac.load_state_dict(rac.state_dict())          # restore actor-critic weights (incl. log_std)
        assert ppo.norm is not None and rnorm is not None, \
            "resume requires an obs RunningNorm on both sides (normalize_obs=True)"
        ppo.norm.load(rnorm.state())                       # CRITICAL: restore running mean/var
        OFFSET = args.iter_offset if args.iter_offset >= 0 else int(rmeta.get("iters_done", 0))
        print(f"[resume] loaded {os.path.basename(args.resume)}  iter_offset={OFFSET}")

    TOTAL = args.total_iters if args.total_iters >= 0 else OFFSET + args.iters

    print(f"FROM-SCRATCH trainer: {'RESUMING from ' + args.resume if args.resume else 'random init, NO warm-start'}.")
    print(f"  envs={args.envs}  iters={args.iters}  horizon={args.horizon}")
    print(f"  lr={args.lr}  entropy={args.entropy}  log_std_init={args.log_std_init}")
    print(f"  sym_coef_plateau={args.sym_coef}  tick_enabled={tick_enabled}")
    print(f"  fell_max={args.fell_max}  drift_max={args.drift_max}")
    print(f"  out={args.out}")
    if args.resume:
        print(f"  OFFSET={OFFSET}  TOTAL={TOTAL}  manual_lr=True")

    # ---------------------------------------------------------------------- #
    #  CSV log setup
    # ---------------------------------------------------------------------- #
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    csv_header = [
        "it", "ep_ret", "ep_len",
        "track", "imit_div", "tick", "drift", "fell",
        "duty_fl", "duty_fr", "duty_hl", "duty_hr",
        "W_IMIT", "W_TICK", "sym_coef",
    ]
    # Append mode when resuming (extend existing CSV history); write mode + header for fresh runs.
    _csv_resume = args.resume and os.path.exists(csv_path)
    csv_file = open(csv_path, "a" if _csv_resume else "w", newline="")
    csv_writer = csv.writer(csv_file)
    if not _csv_resume:
        csv_writer.writerow(csv_header)
    csv_file.flush()

    # ---------------------------------------------------------------------- #
    #  Checkpoint state
    # ---------------------------------------------------------------------- #
    latest = os.path.splitext(args.out)[0] + "_latest.pt"
    best_track = [-1e9]
    # Fallback iter counter for _parse_iter (increments by log_every each call)
    _log_counter = [0]

    # ---------------------------------------------------------------------- #
    #  on_log callback
    #
    #  Called by ppo.learn every log_every=10 iters.  Responsible for:
    #    1. Parsing the current iter from the PPO log msg.
    #    2. Driving env.set_iter(it) and coef_cell[0] = sym_coef(it).
    #       NOTE: there is ~10-iter lag between the rollout and this update
    #       (schedules are only updated at each log boundary), which is
    #       negligible for schedules that span hundreds of iters.
    #    3. Saving _latest.pt every call.
    #    4. Best-under-gate checkpointing: save args.out when
    #       track > best AND fell <= fell_max AND drift <= drift_max.
    #    5. Printing the per-log diagnostic line.
    #    6. Appending one row to the CSV.
    # ---------------------------------------------------------------------- #
    def log(msg):
        _log_counter[0] += 10  # each on_log = log_every=10 iters
        it = _parse_iter(msg, fallback=_log_counter[0])
        g = it + OFFSET  # global iteration (continuous across resumes)

        # --- manual LR anneal for resume runs (fresh runs use PPO's built-in anneal) ---
        if manual_lr:
            frac = max(0.0, 1.0 - (g - 1) / max(1, TOTAL))
            for grp in ppo.opt.param_groups:
                grp["lr"] = args.lr * frac

        # --- update schedules using GLOBAL iter ---
        env.set_iter(g)
        coef_cell[0] = _sym_coef(g, args.sym_coef)

        # --- parse ep_ret / ep_len from msg defensively ---
        try:
            parts = msg.split("|")
            ep_ret = float(parts[1].split()[-1])
            ep_len = float(parts[2].split()[-1])
        except Exception:
            ep_ret = float("nan")
            ep_len = float("nan")

        # --- env metrics ---
        trk   = env.last_track
        idiv  = env.last_imit_div
        tick  = env.last_tick
        drift = env.last_drift
        fell  = env.last_fell
        duty  = env.last_duty
        w_im  = env.W_IMIT
        w_tk  = env.W_TICK
        sym_c = coef_cell[0]

        # --- record progress in meta so a future resume knows where to pick up ---
        ppo.meta["iters_done"] = g

        # --- always save latest ---
        ppo.save(latest)

        # --- best-under-gate checkpoint ---
        gate_ok = (fell <= args.fell_max) and (drift <= args.drift_max)
        mark = ""
        if trk > best_track[0] and gate_ok:
            best_track[0] = trk
            ppo.save(args.out)
            mark = "  <- best"

        # --- diagnostic print (global g shown as "it") ---
        print(
            f"it {g:4d} | track {trk:.3f} | imit_div {idiv:.3f} | tick {tick:.3f} | "
            f"drift {drift:.3f} | fell {fell:.4f} | "
            f"W_IMIT {w_im:.3f} | sym_c {sym_c:.3f}{mark}"
        )
        # Also print the full PPO msg (ep_ret, steps/s, etc.)
        print(f"       {msg}")

        # --- CSV row (use global g as the "it" column so history is continuous) ---
        csv_writer.writerow([
            g, f"{ep_ret:.3f}", f"{ep_len:.1f}",
            f"{trk:.4f}", f"{idiv:.4f}", f"{tick:.4f}", f"{drift:.4f}", f"{fell:.5f}",
            f"{duty[0]:.4f}", f"{duty[1]:.4f}", f"{duty[2]:.4f}", f"{duty[3]:.4f}",
            f"{w_im:.4f}", f"{w_tk:.4f}", f"{sym_c:.4f}",
        ])
        csv_file.flush()

    # ---------------------------------------------------------------------- #
    #  Train
    # ---------------------------------------------------------------------- #
    ppo.learn(args.iters, log_every=10, on_log=log)

    # Final save of latest checkpoint
    ppo.save(latest)
    csv_file.close()

    print(f"\nTraining complete.")
    print(f"  Best checkpoint (gate: fell<={args.fell_max}, drift<={args.drift_max}): {args.out}")
    print(f"  Latest checkpoint: {latest}")
    print(f"  CSV log: {csv_path}")
    print(f"  Best track: {best_track[0]:.4f}")
    print(f"\nNext step:")
    print(f"  python {os.path.basename(__file__)} --eval {args.out}")
    print(f"  python {os.path.basename(__file__)} --score {args.out}")


if __name__ == "__main__":
    main()
