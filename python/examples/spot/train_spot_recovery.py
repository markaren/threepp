"""Train Spot's fall-recovery policy (spot_recovery_env.py), or evaluate one from fallen starts.

    python train_spot_recovery.py --iters 1000 --warmstart spot_steps.pt --out runs/recA/spot_recovery.pt     # arm A
    python train_spot_recovery.py --iters 1000 --warmstart "" --out runs/recB/spot_recovery.pt                # arm B, fresh
    python train_spot_recovery.py --eval runs/recA/spot_recovery.pt --eval-envs 2048 --eval-json eval.json

Arm A continues the shipped 96-d walking policy whole (actor, critic, log_std, obs norm) through the same continue path
as train_spot_steps.py; arm B starts from a fresh network. Both run the walking plant (stiff gains, real torque limits)
with self-collision, and write every choice into the checkpoint meta. --eval scores one episode per env from fallen
starts at the top level (40% back, 40% side, 20% any orientation), with no standing starts and the curriculum frozen,
then the same from L1 and L2 starts (by_level in the JSON), so curriculum progress is readable.

Wave 2 flags (2026-09-13), all off by default = the pilot: --action-scale, --log-std-min, --w-prog / --up-slope,
--hard-frac, --fallen-effort-free, --w-down (spot_recovery_env.py's docstring says what each does). --action-scale S with
a warm start also rescales the continued actor's output layer and the obs norm's last_act entries by 0.2 / S
(--no-warm-rescale to skip), so the continued policy commands the same joint targets on its first tick: the shipped
policy's last_act has std up to 2.3 action units, which unscaled at S = 1 would be 2.3 rad target swings.

Wave 3 flags (2026-09-14), off by default = wave 2: --qd-max V rate-limits the drive targets to V rad/s and --w-qd-excess W
penalizes measured joint speed above V; joint speed metrics are always logged and in the eval JSON. --eval-qd-max V
evaluates a checkpoint under a cap it did not train with (0 = none; default: its own). A warm start from a wave 2 checkpoint
at its own --action-scale needs no rescale (0.5 / 0.5 = 1), and the log says so.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))

import threepp as tp
from threepp.rl import PPO, load_policy
from spot_recovery_env import (SpotRecoveryEnv, CONFIG, OBS_DIM, ACT_DIM, N_LEVELS, KIND_NAMES, EPISODE_S, CONTROL_HZ,
                               OBS_LAST_ACT)
from spot_deploy import ACTION_SCALE
from spot_terrain_env import HIDDEN
from spot_steps_symmetry import make_aux_loss

PPO_GAMMA = 0.99                         # PPO's default gamma, which the progress shaping must use


def qd_line(st):
    """Joint speed per group after the limp ticks (|qd| p95 / p99 / max, share over 10 / 15 / 20 rad/s) and the fastest
    commanded target, from recovery_stats."""
    parts = [f"{g} p95 {r['p95']:.2f} p99 {r['p99']:.2f} max {r['max']:.1f} >10/15/20 {100 * r['over10']:.2f}/"
             f"{100 * r['over15']:.2f}/{100 * r['over20']:.2f}%" for g, r in st["qd"].items() if r["samples"]]
    cm = st["cmd_qd_max_hx_hy_kn"]
    return (f"joint speed rad/s: {' | '.join(parts) or 'no live ticks'} | commanded target max hx/hy/kn "
            f"{cm[0]:.2f}/{cm[1]:.2f}/{cm[2]:.2f}")


@torch.no_grad()
def evaluate(ckpt, k=2048, seed=0, self_collision=None, json_path="", levels=(1, 2), qd_max=None):
    """One episode per env from fallen starts at level N_LEVELS-1, deterministic act_mean -> dict (and JSON). Then one
    episode per env from each of `levels` on the same env (after the L3 pass, so the L3 numbers are the pilot eval's
    exactly), under by_level. The env takes the checkpoint's action scale and target clip (a pilot meta: 0.2, none), and
    its target rate limit unless qd_max says otherwise (0: none)."""
    ac, norm, meta = load_policy(ckpt, device="cuda")
    sc = bool(meta.get("self_collision", True)) if self_collision is None else bool(self_collision)
    scale, clip = float(meta.get("action_scale", ACTION_SCALE)), bool(meta.get("target_clip", False))
    qd = meta.get("qd_max") if qd_max is None else (float(qd_max) or None)
    torch.manual_seed(seed)
    t0 = time.perf_counter()
    env = SpotRecoveryEnv(num_envs=k, seed=seed, self_collision=sc, stand_frac=0.0, init_level=N_LEVELS - 1,
                          freeze_level=True, count_episodes=1, action_scale=scale, target_clip=clip, qd_max=qd)
    build_s = time.perf_counter() - t0
    pol = (lambda o: ac.act_mean(norm.norm(o))) if norm is not None else ac.act_mean
    steps = int(EPISODE_S * CONTROL_HZ) + 5

    def run(level):
        if level != N_LEVELS - 1:
            env.level.fill_(level); env.ep_index.zero_()
        obs = env.reset()
        env.recovery_stats(reset=True)
        t1 = time.perf_counter()
        for _ in range(steps):
            obs, _, _, _, _ = env.step(pol(obs))
        torch.cuda.synchronize()
        return env.recovery_stats(reset=False), time.perf_counter() - t1

    st, steps_s = run(N_LEVELS - 1)
    res = {"checkpoint": os.path.abspath(ckpt), "envs": k, "seed": seed, "self_collision": sc, "level": N_LEVELS - 1,
           "action_scale": scale, "target_clip": clip, "qd_max": qd, "build_s": build_s, "steps": steps, "steps_s": steps_s,
           "meta": {kk: meta.get(kk) for kk in ("warmstart", "arm", "iters", "seed", "self_collision", "lr", "action_scale",
                                                "target_clip", "log_std_min", "w_prog", "up_slope", "hard_frac",
                                                "fallen_effort_free", "w_down", "warm_rescale", "qd_max", "w_qd_excess")},
           **st}
    by_level = {f"L{N_LEVELS - 1}": st}
    for L in levels:
        by_level[f"L{L}"], _ = run(L)
    res["by_level"] = {name: {"fallen": s["total"]["fallen"], "by_kind": s["by_level_kind"],
                              "tau_p95_hx_hy_kn": s["tau_p95_hx_hy_kn"], "qd": s["qd"],
                              "cmd_qd_max_hx_hy_kn": s["cmd_qd_max_hx_hy_kn"]} for name, s in sorted(by_level.items())}

    def fmt(x, nd=3):
        return "n/a" if x is None else f"{x:.{nd}f}"

    tot = st["total"]["fallen"]
    print(f"[eval] {os.path.basename(ckpt)}  K={k}  action scale {scale}{' clipped' if clip else ''}"
          f"{f', targets <= {qd:g} rad/s' if qd else ', no target rate limit'}  fallen episodes "
          f"{tot['episodes']}: success by 3 s {tot['success_3s']:.3f}  by 6 s {tot['success_6s']:.3f}  recovery "
          f"{tot['recovery_s'] or float('nan'):.2f} s  gave up {fmt(tot['gave_up'])}  blow-ups {tot['blowups']}"
          f"  self-contact ticks/ep {tot['contact_ticks_per_ep']:.1f}  p95 tau hx/hy/kn {st['tau_p95_hx_hy_kn']}")
    for name, r in st["by_level_kind"].items():
        print(f"        {name:10s} eps {r['episodes']:5d}  3 s {r['success_3s']:.3f}  6 s {r['success_6s']:.3f}  "
              f"recovery {r['recovery_s'] if r['recovery_s'] is None else round(r['recovery_s'], 2)} s  gave up {r['gave_up']:.3f}")
    for name, r in res["by_level"].items():
        f = r["fallen"]
        print(f"        {name} starts: eps {f['episodes']:5d}  3 s {fmt(f['success_3s'])}  6 s {fmt(f['success_6s'])}  "
              f"gave up {fmt(f['gave_up'])}  recovery {fmt(f['recovery_s'], 2)} s\n          {qd_line(r)}")
    if json_path:
        os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(res, f, indent=1, default=float)
        print(f"[eval] -> {json_path}")
    return res


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 warm-started, 3e-4 fresh")
    ap.add_argument("--warmstart", default=os.path.join(_HERE, "spot_steps.pt"),
                    help="a 96-d checkpoint to continue whole (arm A: the shipped walking policy); '' = fresh network (arm B)")
    ap.add_argument("--log-std-init", dest="log_std_init", type=float, default=-1.0,
                    help="fresh network's initial action log-std")
    ap.add_argument("--log-std-floor", dest="log_std_floor", type=float, default=-1.0,
                    help="a warm start's log-std is raised to at least this (the walking policy explores at ~-1.5, too "
                         "narrow to find a roll); None-like values below -10 leave it alone")
    ap.add_argument("--self-collision", dest="self_collision", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--stand-frac", dest="stand_frac", type=float, default=CONFIG["stand_frac"])
    ap.add_argument("--init-level", dest="init_level", type=int, default=0)
    ap.add_argument("--sym-coef", dest="sym_coef", type=float, default=1.0)
    ap.add_argument("--entropy", type=float, default=0.0)
    ap.add_argument("--target-kl", dest="target_kl", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="", help="label recorded in meta (A / B)")
    ap.add_argument("--out", default=os.path.join(_HERE, "spot_recovery.pt"))
    ap.add_argument("--metrics", default="")
    ap.add_argument("--snapshot-every", dest="snapshot_every", type=int, default=0)
    ap.add_argument("--eval", default="", help="evaluate this checkpoint from fallen starts instead of training")
    ap.add_argument("--eval-envs", dest="eval_envs", type=int, default=2048)
    ap.add_argument("--eval-json", dest="eval_json", default="")
    ap.add_argument("--log-every", dest="log_every", type=int, default=20)
    # ---- wave 2 (2026-09-13): every default is the pilot ----
    ap.add_argument("--action-scale", dest="action_scale", type=float, default=None,
                    help="targets = default_q + S * action, clipped to the joint limits (default: the pilot's 0.2, unclipped)")
    ap.add_argument("--warm-rescale", dest="warm_rescale", action=argparse.BooleanOptionalAction, default=True,
                    help="with --action-scale and a warm start, scale the continued actor output and the norm's last_act "
                         "by 0.2 / S so the first tick commands the walking policy's targets")
    ap.add_argument("--log-std-min", dest="log_std_min", type=float, default=None,
                    help="clamp log_std >= this after every PPO update, for the whole run (pilot: the floor held only at "
                         "the start; log_std ended at -1.50 warm / -2.2 fresh)")
    ap.add_argument("--w-prog", dest="w_prog", type=float, default=0.0,
                    help="potential-based progress shaping W * (gamma phi' - phi), phi = (1 - g_z) / 2")
    ap.add_argument("--up-slope", dest="up_slope", type=float, default=0.0,
                    help="upright term W_UP ((1 - C) phi^2 + C phi): slope W_UP C on the back (pilot: phi^2, slope 0)")
    ap.add_argument("--hard-frac", dest="hard_frac", type=float, default=0.0,
                    help="share of fallen resets drawn at L3 whatever the env's level (no promote / demote on them)")
    ap.add_argument("--fallen-effort-free", dest="fallen_effort_free", action="store_true",
                    help="torque / action-rate / joint-limit penalties x clamp((up_z - 0.3) / 0.6, 0, 1)")
    ap.add_argument("--w-down", dest="w_down", type=float, default=0.0, help="-C per live step while up_z < 0.5")
    # ---- wave 3 (2026-09-14): every default is wave 2 ----
    ap.add_argument("--qd-max", dest="qd_max", type=float, default=None,
                    help="rate-limit the drive targets: at most V rad/s per joint, V * control dt per tick (default: none)")
    ap.add_argument("--w-qd-excess", dest="w_qd_excess", type=float, default=0.0,
                    help="-W mean((|qd| - qd_max)_+^2) on the measured joint speeds (needs --qd-max)")
    ap.add_argument("--eval-qd-max", dest="eval_qd_max", type=float, default=None,
                    help="with --eval: evaluate under this target rate limit (0 = none; default: the checkpoint's own)")
    args = ap.parse_args()
    if args.w_qd_excess > 0 and not args.qd_max:
        ap.error("--w-qd-excess penalizes joint speed above --qd-max: set --qd-max too")
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    if args.eval:
        evaluate(args.eval, k=args.eval_envs, seed=args.seed, json_path=args.eval_json, qd_max=args.eval_qd_max)
        return

    torch.manual_seed(args.seed)
    warm = bool(args.warmstart)
    lr = args.lr if args.lr is not None else (1e-4 if warm else 3e-4)
    t0 = time.perf_counter()
    scale = ACTION_SCALE if args.action_scale is None else float(args.action_scale)
    env = SpotRecoveryEnv(num_envs=args.envs, seed=args.seed, self_collision=args.self_collision,
                          stand_frac=args.stand_frac, init_level=args.init_level, action_scale=scale,
                          target_clip=args.action_scale is not None, w_prog=args.w_prog, prog_gamma=PPO_GAMMA,
                          up_slope=args.up_slope, hard_frac=args.hard_frac, fallen_effort_free=args.fallen_effort_free,
                          w_down=args.w_down, qd_max=args.qd_max, w_qd_excess=args.w_qd_excess)
    print(f"recovery env: K={args.envs}, self-collision {'on' if args.self_collision else 'off'}, built in "
          f"{time.perf_counter() - t0:.1f} s | {env.knob_config()}")
    aux = make_aux_loss(args.sym_coef) if args.sym_coef > 0 else None
    meta = {**CONFIG, **env.knob_config(), "log_std_min": args.log_std_min, "warm_rescale": None,
            "self_collision": args.self_collision, "warmstart": os.path.basename(args.warmstart) if warm else "",
            "warmstart_path": os.path.abspath(args.warmstart) if warm else "", "arm": args.arm, "iters": args.iters,
            "envs": args.envs, "seed": args.seed, "lr": lr, "horizon": args.horizon, "log_std_init": args.log_std_init,
            "log_std_floor": args.log_std_floor if warm else None, "stand_frac": args.stand_frac,
            "init_level": args.init_level, "sym_coef": args.sym_coef, "entropy": args.entropy, "target_kl": args.target_kl,
            "epochs": args.epochs, "minibatches": args.minibatches, "clip": args.clip, "select": "final",
            "drive_limits_are_forces": True, "height_source": "flat", "stand_mode": True}
    ppo = PPO(env, ACT_DIM, hidden=HIDDEN, lr=lr, horizon=args.horizon, log_std_init=args.log_std_init,
              entropy=args.entropy, clip=args.clip, epochs=args.epochs, minibatches=args.minibatches,
              target_kl=args.target_kl if args.target_kl > 0 else None, normalize_obs=True, meta=meta, aux_loss=aux)
    assert ppo.gamma == env.prog_gamma, "the progress shaping must discount with the PPO gamma"
    if warm:
        src, src_norm, src_meta = load_policy(args.warmstart, device="cuda")
        if src_meta.get("obs_dim") != OBS_DIM:
            ap.error(f"--warmstart must be a {OBS_DIM}-d checkpoint, {args.warmstart} is {src_meta.get('obs_dim')}-d")
        ppo.ac.load_state_dict(src.state_dict())
        if ppo.norm is not None and src_norm is not None:
            ppo.norm.load(src_norm.state())
        before = float(ppo.ac.log_std.mean())
        if args.log_std_floor > -10:
            ppo.ac.log_std.data.clamp_(min=args.log_std_floor)
        ppo.meta["warmstart_log_std"] = [before, float(ppo.ac.log_std.mean())]
        print(f"continued from {os.path.basename(args.warmstart)} (96-d: actor+critic+log_std+norm), log_std mean "
              f"{before:.2f} -> {float(ppo.ac.log_std.mean()):.2f}")
        f = float(src_meta.get("action_scale", ACTION_SCALE)) / scale
        if args.warm_rescale and f != 1.0:
            # same targets on the first tick: the actor's output and the last_act it reads both shrink by f, and the
            # norm's last_act mean / var follow, so the normalized obs the network sees are unchanged
            last = ppo.ac.actor[-1]
            last.weight.data.mul_(f); last.bias.data.mul_(f)
            if ppo.norm is not None:
                ppo.norm.mean[OBS_LAST_ACT] *= f
                ppo.norm.var[OBS_LAST_ACT] *= f * f
            ppo.meta["warm_rescale"] = f
            print(f"warm rescale: actor output and norm last_act x {f:g} (action scale {src_meta.get('action_scale', ACTION_SCALE)}"
                  f" -> {scale})")
        else:
            print(f"warm start's action scale {src_meta.get('action_scale', ACTION_SCALE)} -> this run's {scale}: "
                  f"{'the same, no rescale' if f == 1.0 else 'NOT rescaled (--no-warm-rescale)'}")
        ppo.meta["warmstart_action_scale"] = float(src_meta.get("action_scale", ACTION_SCALE))
    else:
        print(f"fresh network, log_std {args.log_std_init}")

    ls_seen = [float("inf")]                # smallest per-dim log_std any rollout acted with (the clamp's own check)

    def clamp_log_std():
        """--log-std-min, trainer side: after every update (on_iter) and before the first rollout. PPO's Adam keeps its
        moments, so an update can still step below the bound, and the next clamp takes it back before any rollout."""
        if args.log_std_min is not None:
            ppo.ac.log_std.data.clamp_(min=args.log_std_min)
            ls_seen[0] = min(ls_seen[0], float(ppo.ac.log_std.min()))

    clamp_log_std()
    if args.log_std_min is not None:
        print(f"log_std clamped >= {args.log_std_min} for the whole run: mean now {float(ppo.ac.log_std.mean()):.2f}")

    stem = os.path.splitext(args.out)[0]
    latest = stem + "_latest.pt"
    metrics_path = args.metrics or os.path.join(os.path.dirname(os.path.abspath(args.out)), "metrics.jsonl")
    for d in {os.path.dirname(os.path.abspath(args.out)), os.path.dirname(os.path.abspath(metrics_path))}:
        os.makedirs(d, exist_ok=True)

    def metric(rec):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    metric({"type": "start", "out": os.path.abspath(args.out), "argv": sys.argv[1:], "meta": ppo.meta})
    env.recovery_stats(reset=True)

    def log(msg):
        clamp_log_std()                     # learn() logs before on_iter: clamp first so _latest and the line are post-clamp
        ppo.save(latest)
        st = env.recovery_stats(reset=True)
        fa, sd = st["total"].get("fallen", {}), st["total"].get("stand", {})
        f3 = fa.get("success_3s"); f6 = fa.get("success_6s")
        print(f"{msg}\n        fallen eps {fa.get('episodes', 0)}: success 3 s {f3 if f3 is None else round(f3, 3)}  6 s "
              f"{f6 if f6 is None else round(f6, 3)}  recovery {fa.get('recovery_s') and round(fa['recovery_s'], 2)} s | stand "
              f"eps {sd.get('episodes', 0)} success {sd.get('success_6s') and round(sd['success_6s'], 3)} | levels "
              f"{st['levels']} | p95 tau {st['tau_p95_hx_hy_kn']} | contact ticks/ep {fa.get('contact_ticks_per_ep') and round(fa['contact_ticks_per_ep'], 1)}"
              f" | blow-ups {fa.get('blowups', 0) + sd.get('blowups', 0)} | spawn {st['spawn']}")
        g3 = {k: (v if v is None else round(v, 3)) for k, v in st["gave_up_L3"].items()}
        extra = {"log_std_now": [float(ppo.ac.log_std.min()), float(ppo.ac.log_std.mean())]}
        if args.log_std_min is not None:
            extra["log_std_min_acted"] = ls_seen[0]
        line = (f"        gave up: fallen {fa.get('gave_up') and round(fa['gave_up'], 3)}  L3 {g3} | log_std now min/mean "
                f"{extra['log_std_now'][0]:.2f}/{extra['log_std_now'][1]:.2f} (update diag {ppo.diagnostics().get('log_std', float('nan')):.2f})")
        if "hard" in st:
            line += f" | hard {st['hard']}"
        if "shaping" in st:
            line += f" | shaping {st['shaping']}"
        print(line)
        print(f"        {qd_line(st)}" + (f" | qd_excess {st['qd_excess']}" if "qd_excess" in st else ""))
        if args.qd_max:
            # the limiter checked from the targets it wrote (float32 round-off is ~1e-5 rad/s)
            worst = max(st["cmd_qd_max_hx_hy_kn"])
            assert worst <= args.qd_max + 1e-3, f"commanded target speed {worst:.4f} rad/s > --qd-max {args.qd_max}"
        metric({"type": "log", **ppo.last_log, **ppo.diagnostics(), **extra, **st})

    def on_iter(it):
        clamp_log_std()                     # before the checkpoint below and the next rollout
        if args.snapshot_every > 0 and it % args.snapshot_every == 0:
            ppo.save(f"{stem}_it{it:05d}.pt")

    ppo.learn(args.iters, log_every=args.log_every, on_log=log, on_iter=on_iter)
    ppo.save(latest)
    ppo.save(args.out)
    print(f"saved -> {args.out} = {latest} (final weights) | metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
