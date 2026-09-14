"""Train Spot's jump policy (spot_jump_env.py), or evaluate one.

    python train_spot_jump.py --iters 1500 --qd-max 10 --out runs/jumpA/spot_jump.pt     # continues spot_recovery.pt
    python train_spot_jump.py --eval runs/jumpA/spot_jump.pt --eval-json eval.json

Continues a 96-d stand-mode checkpoint whole (default: the shipped fall-recovery policy, which already stands and gets
up on this plant), at that checkpoint's action scale; a warm start at another scale is refused rather than rescaled.
The target rate limit defaults to the warm start's own (the shipped recovery: 10 rad/s). The symmetry loss uses a mirror
that leaves the jump signal in the clock channels alone (spot_jump_env.jump_mirror_obs).

--eval runs one episode per env with a trigger at --eval-trigger s (dx: 30% zero, else uniform up to dx_max), then one
episode per env with no trigger, deterministic, and writes both to the JSON (jump / nojump).
"""
import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))

import threepp as tp
from threepp.rl import PPO, load_policy
from spot_jump_env import SpotJumpEnv, CONFIG, OBS_DIM, ACT_DIM, EPISODE_S, CONTROL_HZ, DX_MAX, make_jump_aux_loss
from spot_terrain_env import HIDDEN


def fmt(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def summary(st):
    j = st["jump"]
    lines = [f"jump eps {j['episodes']}: jumped {fmt(j['jumped'])}  landed up {fmt(j['landed_up'])}  success "
             f"{fmt(j['success'])}  fell {fmt(j['fell'])}  flight {fmt(j['flight_s'], 2)} s  rise {fmt(j['rise_m'])} m  "
             f"clearance {fmt(j['clearance_m'])} m  |dx err| {fmt(j['dx_err_m'])} m  uncmd air/ep "
             f"{fmt(j['uncmd_air_ticks_per_ep'], 2)}"]
    for name, b in st["by_bucket"].items():
        lines.append(f"  {name:9s} eps {b['episodes']:5d}  jumped {fmt(b['jumped'])}  success {fmt(b['success'])}  fell "
                     f"{fmt(b['fell'])}  flight {fmt(b['flight_s'], 2)}  clear {fmt(b['clearance_m'])} (max "
                     f"{b['clearance_max_m']:.3f})  dx {fmt(b['dx_m'])} err {fmt(b['dx_err_m'])}  uncmd/ep "
                     f"{fmt(b['uncmd_air_ticks_per_ep'], 2)}")
    qw = " | ".join(f"{g} p95 {fmt(r['p95'], 1)} p99 {fmt(r['p99'], 1)} max {fmt(r['max'], 1)}"
                    for g, r in st["qd_window"].items() if r["samples"])
    qa = " | ".join(f"{g} p99 {fmt(r['p99'], 1)} max {fmt(r['max'], 1)}" for g, r in st["qd"].items() if r["samples"])
    lines.append(f"  joint speed rad/s, window: {qw or 'none'} ; all live: {qa} ; commanded max "
                 f"{[round(v, 2) for v in st['cmd_qd_max_hx_hy_kn']]} ; p95 tau {st['tau_p95_hx_hy_kn']}")
    return "\n        ".join(lines)


@torch.no_grad()
def evaluate(ckpt, k=2048, seed=0, trigger=1.0, json_path=""):
    ac, norm, meta = load_policy(ckpt, device="cuda")
    env = SpotJumpEnv(num_envs=k, seed=seed, self_collision=bool(meta.get("self_collision", True)),
                      action_scale=float(meta["action_scale"]), qd_max=meta.get("qd_max"),
                      dx_max=float(meta.get("dx_max", DX_MAX)), jump_frac=1.0, trigger_s=trigger, count_episodes=1)
    pol = (lambda o: ac.act_mean(norm.norm(o))) if norm is not None else ac.act_mean
    steps = int(EPISODE_S * CONTROL_HZ) + 5
    res = {"checkpoint": os.path.abspath(ckpt), "envs": k, "seed": seed, "trigger_s": trigger,
           "action_scale": env.action_scale, "qd_max": env.qd_max,
           "meta": {kk: meta.get(kk) for kk in ("warmstart", "arm", "iters", "seed", "qd_max", "w_qd_excess", "lr",
                                                "log_std_min", "jump_frac", "dx_max")}}
    for name, frac in (("jump", 1.0), ("nojump", 0.0)):
        env.jump_frac = frac
        env.ep_index.zero_()
        obs = env.reset()
        env.jump_stats(reset=True)
        t0 = time.perf_counter()
        for _ in range(steps):
            obs, _, _, _, _ = env.step(pol(obs))
        torch.cuda.synchronize()
        st = env.jump_stats(reset=False)
        st["steps_s"] = time.perf_counter() - t0
        res[name] = st
        print(f"[eval {name}] {os.path.basename(ckpt)} K={k} trigger {trigger} s, targets <= {env.qd_max} rad/s\n        "
              + summary(st) if name == "jump" else
              f"[eval nojump] success (stood, never airborne) {fmt(st['by_bucket']['nojump']['success'])}  fell "
              f"{fmt(st['by_bucket']['nojump']['fell'])}  uncmd air/ep {fmt(st['by_bucket']['nojump']['uncmd_air_ticks_per_ep'], 2)}")
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
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default 1e-4 warm-started, 3e-4 fresh")
    ap.add_argument("--warmstart", default=os.path.join(_HERE, "spot_recovery.pt"),
                    help="a 96-d stand-mode checkpoint to continue whole; '' = fresh network")
    ap.add_argument("--action-scale", dest="action_scale", type=float, default=None,
                    help="default: the warm start's (fresh: 0.5)")
    ap.add_argument("--qd-max", dest="qd_max", type=float, default=None,
                    help="drive target rate limit, rad/s (default: the warm start's, else 10)")
    ap.add_argument("--w-qd-excess", dest="w_qd_excess", type=float, default=0.02)
    ap.add_argument("--log-std-init", dest="log_std_init", type=float, default=-1.0)
    ap.add_argument("--log-std-min", dest="log_std_min", type=float, default=-0.8,
                    help="clamp log_std >= this after every update (exploration floor, as recovery waves 2-3)")
    ap.add_argument("--jump-frac", dest="jump_frac", type=float, default=CONFIG["jump_frac"])
    ap.add_argument("--dx-max", dest="dx_max", type=float, default=DX_MAX)
    ap.add_argument("--self-collision", dest="self_collision", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--sym-coef", dest="sym_coef", type=float, default=1.0)
    ap.add_argument("--entropy", type=float, default=0.0)
    ap.add_argument("--target-kl", dest="target_kl", type=float, default=0.02)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--arm", default="")
    ap.add_argument("--out", default=os.path.join(_HERE, "spot_jump.pt"))
    ap.add_argument("--metrics", default="")
    ap.add_argument("--snapshot-every", dest="snapshot_every", type=int, default=0)
    ap.add_argument("--log-every", dest="log_every", type=int, default=20)
    ap.add_argument("--eval", default="")
    ap.add_argument("--eval-envs", dest="eval_envs", type=int, default=2048)
    ap.add_argument("--eval-json", dest="eval_json", default="")
    ap.add_argument("--eval-trigger", dest="eval_trigger", type=float, default=1.0)
    args = ap.parse_args()
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    if args.eval:
        evaluate(args.eval, k=args.eval_envs, seed=args.seed, trigger=args.eval_trigger, json_path=args.eval_json)
        return

    torch.manual_seed(args.seed)
    warm = bool(args.warmstart)
    src_meta = load_policy(args.warmstart, device="cuda")[2] if warm else {}
    if warm and src_meta.get("obs_dim") != OBS_DIM:
        ap.error(f"--warmstart must be a {OBS_DIM}-d checkpoint, {args.warmstart} is {src_meta.get('obs_dim')}-d")
    warm_scale = float(src_meta.get("action_scale", 0.2)) if warm else None
    scale = args.action_scale if args.action_scale is not None else (warm_scale if warm else 0.5)
    if warm and abs(scale - warm_scale) > 1e-9:
        ap.error(f"the warm start trained at action scale {warm_scale}, --action-scale {scale}: not rescaled here")
    qd = args.qd_max if args.qd_max is not None else (src_meta.get("qd_max") or 10.0)
    lr = args.lr if args.lr is not None else (1e-4 if warm else 3e-4)
    t0 = time.perf_counter()
    env = SpotJumpEnv(num_envs=args.envs, seed=args.seed, self_collision=args.self_collision, action_scale=scale,
                      qd_max=qd, w_qd_excess=args.w_qd_excess, jump_frac=args.jump_frac, dx_max=args.dx_max)
    print(f"jump env: K={args.envs}, built in {time.perf_counter() - t0:.1f} s | {env.knob_config()}")
    meta = {**env.config(), "log_std_min": args.log_std_min, "warmstart": os.path.basename(args.warmstart) if warm else "",
            "warmstart_path": os.path.abspath(args.warmstart) if warm else "", "arm": args.arm, "iters": args.iters,
            "envs": args.envs, "seed": args.seed, "lr": lr, "horizon": args.horizon, "sym_coef": args.sym_coef,
            "entropy": args.entropy, "target_kl": args.target_kl, "epochs": args.epochs, "minibatches": args.minibatches,
            "clip": args.clip, "select": "final", "drive_limits_are_forces": True, "height_source": "flat",
            "hidden": list(HIDDEN)}
    ppo = PPO(env, ACT_DIM, hidden=HIDDEN, lr=lr, horizon=args.horizon, log_std_init=args.log_std_init,
              entropy=args.entropy, clip=args.clip, epochs=args.epochs, minibatches=args.minibatches,
              target_kl=args.target_kl if args.target_kl > 0 else None, normalize_obs=True, meta=meta,
              aux_loss=make_jump_aux_loss(args.sym_coef) if args.sym_coef > 0 else None)
    if warm:
        src, src_norm, _ = load_policy(args.warmstart, device="cuda")
        ppo.ac.load_state_dict(src.state_dict())
        if ppo.norm is not None and src_norm is not None:
            ppo.norm.load(src_norm.state())
        print(f"continued from {os.path.basename(args.warmstart)} (actor+critic+log_std+norm), action scale {scale}, "
              f"targets <= {qd} rad/s, log_std mean {float(ppo.ac.log_std.mean()):.2f}")
    else:
        print(f"fresh network, log_std {args.log_std_init}, action scale {scale}, targets <= {qd} rad/s")

    def clamp_log_std():
        if args.log_std_min is not None:
            ppo.ac.log_std.data.clamp_(min=args.log_std_min)

    clamp_log_std()
    stem = os.path.splitext(args.out)[0]
    latest = stem + "_latest.pt"
    metrics_path = args.metrics or os.path.join(os.path.dirname(os.path.abspath(args.out)), "metrics.jsonl")
    for d in {os.path.dirname(os.path.abspath(args.out)), os.path.dirname(os.path.abspath(metrics_path))}:
        os.makedirs(d, exist_ok=True)

    def metric(rec):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec, default=float) + "\n")

    metric({"type": "start", "out": os.path.abspath(args.out), "argv": sys.argv[1:], "meta": ppo.meta})
    env.jump_stats(reset=True)

    def log(msg):
        clamp_log_std()
        ppo.save(latest)
        st = env.jump_stats(reset=True)
        print(f"{msg}\n        {summary(st)}\n        log_std min/mean {float(ppo.ac.log_std.min()):.2f}/"
              f"{float(ppo.ac.log_std.mean()):.2f}")
        worst = max(st["cmd_qd_max_hx_hy_kn"])
        assert worst <= qd + 1e-3, f"commanded target speed {worst:.4f} rad/s > qd_max {qd}"
        metric({"type": "log", **ppo.last_log, **ppo.diagnostics(), **st})

    def on_iter(it):
        clamp_log_std()
        if args.snapshot_every > 0 and it % args.snapshot_every == 0:
            ppo.save(f"{stem}_it{it:05d}.pt")

    ppo.learn(args.iters, log_every=args.log_every, on_log=log, on_iter=on_iter)
    ppo.save(latest)
    ppo.save(args.out)
    print(f"saved -> {args.out} | metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
