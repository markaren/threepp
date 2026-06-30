"""train_kuka_grasp.py — train the KUKA to reach, grasp, and lift a cube (threepp.rl PPO).

    python train_kuka_grasp.py --envs 2048 --iters 2000        # full run
    python train_kuka_grasp.py --envs 256 --iters 60 --smoke   # quick smoke test

Drives the env curriculum (cube spawn region grows; graduated reset height start_high 0→1 so the arm
starts from "at the cube" → "anywhere up to the DEFAULT_Q home") from the PPO log callback, and
checkpoints the best policy by success rate. Watch a checkpoint with
`python play_kuka_grasp.py --model kuka_grasp_best.pt`.
"""
import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_PYTHON_DIR = os.path.dirname(os.path.dirname(_HERE))
if _PYTHON_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_DIR)

import kuka_grasp_contract as C  # noqa: E402
from kuka_grasp_env import KukaGraspEnv  # noqa: E402


def _iter_of(msg):
    # PPO log line is "it  120 | ep_ret ..." — pull the iteration out to drive the curriculum
    try:
        return int(msg.split("|")[0].split()[1])
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=os.path.join(_HERE, "kuka_grasp.pt"))
    ap.add_argument("--smoke", action="store_true", help="tiny run; just checks learning happens")
    args = ap.parse_args()

    from threepp.rl import PPO

    env = KukaGraspEnv(num_envs=args.envs, device=args.device)
    ppo = PPO(env, C.ACT_DIM, hidden=(512, 256, 128), lr=args.lr, horizon=args.horizon,
              normalize_obs=True, normalize_returns=True, target_kl=0.02, log_std_init=-0.5,
              meta={"task": "kuka_grasp", "dt": C.DT, "control_hz": C.CONTROL_HZ,
                    "arm_action_scale": C.ARM_ACTION_SCALE.tolist(), "default_q": C.DEFAULT_Q.tolist()})

    best_path = os.path.splitext(args.out)[0] + "_best.pt"
    latest_path = os.path.splitext(args.out)[0] + "_latest.pt"
    best = [-1.0]

    def on_log(msg):
        it = _iter_of(msg)
        env.set_iter(it)
        ppo.meta["iter"] = it    # record the curriculum stage so play can match the spawn region
        print(f"{msg} | reach {env.last_reach:.3f} grasp {env.last_grasp_rate:.2f} "
              f"lift {env.last_lift:.3f} succ {env.last_success_rate:.2f} "
              f"spawn {env.spawn_half[0]:.2f}x{env.spawn_half[1]:.2f} startH {env.start_high:.2f}")
        ppo.save(latest_path)
        if env.last_success_rate >= best[0]:
            best[0] = env.last_success_rate
            ppo.save(best_path)

    print(f"[train] {args.envs} envs x {args.iters} iters  obs={C.OBS_DIM} act={C.ACT_DIM}")
    ppo.learn(args.iters, log_every=10, on_log=on_log)
    ppo.save(args.out)
    print(f"[train] saved {args.out}  (best succ={best[0]:.3f} -> {best_path})")

    if args.smoke:
        print(f"[smoke] final reach={env.last_reach:.3f} grasp_rate={env.last_grasp_rate:.2f} "
              f"success={env.last_success_rate:.2f}")


if __name__ == "__main__":
    main()
