"""spotv2 — train the velocity-tracking policy to climb DISCRETE STAIRS with an adaptive curriculum.

    python train_spot_steps.py --iters 1500                       # warm-starts from scratch_flat_best.pt (default)
    python train_spot_steps.py --score spot_steps.pt              # deterministic track/flat/fell + curriculum level
    python train_spot_steps.py --eval  spot_steps.pt              # flat-steering regression vs the base gait

    # the mixed-terrain course (spot_course.py), continued from a 96-d raycast checkpoint:
    python train_spot_steps.py --course --warmstart idun_runs/raycast_s4/spot_steps_latest.pt --iters 300 \
        --shove-ramp 0.5,2.0,600 --foot-mu-dr 0.3,1.0,8 --stand-mode --w-place 10 --select final
    # wave 2 adds abrupt command switches and messy resets (and trains on the 'hard' ladder by default):
    python train_spot_steps.py --course --warmstart ... --cmd-switch 3,8,0.3 --reset-noise 0.08,0.06,0.3 --w-place 30

The curriculum (per-env level, promote on clearing the tent / demote only when the robot never reached
it; a fall does not demote) lives in SpotStepsEnv.
The warm-start transfers the 50-d clock base gait (scratch_flat_best.pt, normalize_obs=True, stiff gains)
into the 96-d AC: input cols [0:50] copied (proprio+clock), terrain cols [50:96] zero-init; RunningNorm
expanded with matching stats (terrain dims left at fresh default mean=0/var=1 so they adapt freely).
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), "spot"))
sys.path.insert(0, os.path.join(_HERE, "scratch_distillation"))

import threepp as tp
from spot_terrain_env import quat_rotate_inverse
from spot_steps_env import (ACT_DIM, CONFIG, HIDDEN, N_LEVELS, OBS_DIM, RISERS, W_IMIT, W_STAND, SPAWN_HOLD,
                            PLACE_SAT, PLACE_PERCH, SpotStepsEnv)
import spot_course as sc
from spot_steps_symmetry import make_aux_loss
from threepp.rl import PPO, load_policy
from _common import (warmstart_scratch_to_terrain, sanity_walk, stochastic_flat_baseline,
                     eval_flat_steering)


@torch.no_grad()
def score_checkpoint(policy_path, k=512, device="cuda", steps=900, warm=200, height_source=None,
                     perceive=None, seed=0, perceive_noise=None, start_level=None, push_vel=0.0):
    """Deterministic track/flat/fell + the curriculum LEVEL it climbs to (how tall a riser it handles).
    The env starts every stair env at level 0 and promotes as the policy clears tents.

    The height source comes from the checkpoint's own meta, so a policy trained on the raycast scan
    is scored against the raycast scan — scoring it against the other one measures the mismatch."""
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); return None
    # The command sampler draws from torch's DEFAULT generator, which nothing here was seeding —
    # so two scoring runs of one checkpoint saw different velocity commands and landed up to 0.012
    # apart, which is wider than any difference this script is usually asked to resolve.
    torch.manual_seed(seed)
    ac, norm, meta = load_policy(policy_path, device=device)
    trained_on = meta.get("height_source", "analytic")
    src = height_source or trained_on
    saw = bool(meta.get("perceive", False))
    see = saw if perceive is None else bool(perceive)
    if see:
        src = "raycast"
    print(f"scoring against the {src} height source, scan {'PERCEIVED' if see else 'privileged'}"
          + ("" if (src == trained_on and see == saw) else
             f"  (MISMATCH: trained on {trained_on}, "
             f"{'perceived' if saw else 'privileged'} scan)"))
    # Score-time map error: the checkpoint's own unless overridden (the robustness sweep scores one
    # policy at several noise levels it was never trained at).
    noise = float(meta.get("perceive_noise", 0.0)) if perceive_noise is None else float(perceive_noise)
    env = SpotStepsEnv(num_envs=k, device=device, height_source=None if see else src,
                       perceive=see, perceive_noise=noise, seed=seed, push_vel=push_vel)
    if start_level is not None:
        # The default scoring starts every lane at level 0 and climbs about one level in 900 steps,
        # which never reaches the risers a converged policy was trained on: every checkpoint reads
        # ~1.9 track / 0 falls. Pinning the START level puts the whole window on the hard stairs;
        # demotion on a fall still applies, so `level` and `clear` become the discriminating numbers.
        env.level.fill_(int(start_level))
        print(f"start level {int(start_level)} (~{RISERS[int(start_level)]:.02f} m risers)"
              + (f", shoves up to {push_vel} m/s" if push_vel > 0 else ""))
    pol = (lambda o: ac.act_mean(norm.norm(o))) if norm is not None else ac.act_mean
    obs = env.reset()
    trk, flt, fl, clr = [], [], [], []
    for t in range(steps):
        if t == warm:
            # Start the episode counters where the averages below start, so the two describe the
            # same window: an episode that ended inside the warm-up would otherwise be counted
            # here and averaged nowhere, and this cell is the legacy reconciliation number.
            env.reset_stats()
        obs, _, _, _, _ = env.step(pol(obs))
        if t >= warm:
            trk.append(env.last_track); flt.append(env.last_flat_track); fl.append(env.last_fell)
            clr.append(env.last_clear)
    m = lambda a: sum(a) / max(1, len(a))
    lvl = env.last_level
    riser = RISERS[min(int(round(lvl)), N_LEVELS - 1)]
    print(f"[score] {os.path.basename(policy_path)}  (deterministic, K={k}, {steps - warm} steps)")
    print(f"        track {m(trk):.3f}/2.0   flat {m(flt):.3f}/2.0   fell/step {m(fl):.4f}   "
          f"curriculum level {lvl:.2f}/{N_LEVELS - 1}  (~{riser:.02f} m risers)   clear {m(clr):.3f}")
    # The counters are per episode, not per step: 'clear' above is the legacy sample-and-hold of the
    # last done batch and is not an episode rate (kept for the committed tables).
    counters = env.episode_stats()
    for r in counters["rows"]:
        n = r["episodes"]
        if n == 0:
            continue
        # The shoves here are the legacy 'random' ones: at push_prob they land about every
        # 1/push_prob steps, which is INSIDE the 2 s attribution window, so every fall on a shoved
        # lane counts as one after a push and the ratio is not a causal rate. Only the one-shove-
        # per-episode mode (score_e0) earns the unqualified name.
        print(f"        {r['lane_type']:5s} episodes {n}: success {r['success'] / n:.3f}  "
              f"cleared {r['cleared'] / n:.3f}  falls {r['terminations'] / n:.3f} "
              f"(tilt {r['term_tilt']}, low {r['term_low']}; after clear {r['fell_after_clear']})"
              + (f"  falls/push{'' if env.push_mode == 'once_on_tent' else ' (uncorrected)'} "
                 f"{r['falls_after_push'] / r['pushes']:.4f} of {r['pushes']}"
                 if r["pushes"] else ""))
    score_checkpoint.last = {"track": m(trk), "flat": m(flt), "fell": m(fl), "level": float(lvl),
                             "clear": m(clr), "start_level": start_level, "push_vel": push_vel,
                             "score_source": src, "score_perceive": bool(see), "score_noise": noise,
                             "score_envs": k, "score_seed": seed,
                             "trained_source": trained_on, "trained_perceive": saw,
                             "trained_noise": float(meta.get("perceive_noise", 0.0)),
                             "counters": counters}
    return m(trk), m(flt), m(fl), lvl


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=2048)        # ~52k stair boxes -> builds fine; 2x throughput vs 1024
    ap.add_argument("--iters", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gate", type=float, default=0.90)
    ap.add_argument("--out", default=os.path.join(_HERE, "spot_steps.pt"))
    ap.add_argument("--warmstart", default=os.path.join(_HERE, "scratch_distillation", "scratch_flat_best.pt"),
                    help="what to start the fine-tune from. A 50-d clock base gait is expanded into "
                         "96-d (default = scratch_flat_best.pt); a checkpoint that is ALREADY 96-d — "
                         "spot_hf.pt, say — is continued whole, which is how you chain "
                         "heightfield -> stairs.")
    ap.add_argument("--fell_max", type=float, default=0.006)
    ap.add_argument("--sym_coef", type=float, default=1.0)   # left-right symmetry augmentation (kills the veer)
    ap.add_argument("--push-vel", dest="push_vel", type=float, default=0.0,
                    help="random shoves, as the velocity change (m/s) each delivers to the base. "
                         "Nothing has ever perturbed this robot in training, so recovery was never "
                         "selected for. Measured on spot_steps.pt: falls per 1000 pushes go 0.5 at "
                         "1.2 m/s, 3 at 2.0, 29 at 3.0, 92 at 4.0 — so ~3.0 is where it starts to "
                         "matter and is the sensible thing to train against.")
    ap.add_argument("--push-prob", dest="push_prob", type=float, default=0.02,
                    help="per-env per-step probability of a shove")
    ap.add_argument("--cadence-jitter", dest="cadence_jitter", type=float, default=0.0,
                    help="sample each episode's gait period from GAIT_PERIOD * (1 +- this). 0 = the "
                         "single fixed cadence, under which the policy can vary stride length and "
                         "nothing else — one gait, whatever the terrain does.")
    ap.add_argument("--imit-anneal", dest="imit_anneal", type=int, default=-1,
                    help="iteration at which the imitation anchor starts decaying to 0 by the end "
                         "of training. The anchor protects steering, so the decay HOLDS on any log "
                         "where flat tracking is under the gate. -1 = never decay (the default, and "
                         "what pins the gait to the teacher for all 1500 iterations).")
    ap.add_argument("--graph", action="store_true",
                    help="replay the step's torch region from a CUDA graph (VecTask graph=True). "
                         "The region is launch-bound — a few hundred small kernels whose dispatch "
                         "costs far more than their execution — so this is ~1.7x on the step. "
                         "Verified against a fresh eager observation before training starts.")
    ap.add_argument("--perceive", action="store_true",
                    help="train on the scan a body-mounted depth camera could actually have seen "
                         "(threepp.rl.perception): occluded and never-observed cells read flat, and "
                         "the map remembers what the camera swept. Implies --height-source raycast. "
                         "Reward, termination and the spawn keep reading ground truth.")
    ap.add_argument("--perceive-noise", dest="perceive_noise", type=float, default=0.0,
                    help="elevation-map error (m) as a fixed per-cell bias, with --perceive")
    ap.add_argument("--height-source", dest="height_source", choices=("analytic", "raycast"),
                    default=None,
                    help="where the terrain height (h_here + the 45-cell scan) comes from. "
                         "analytic = the closed-form tent formula; raycast = a ray into a Warp BVH "
                         "over the boxes actually added to the world (threepp.rl.raycast). The two "
                         "agree exactly inside a lane; the raycast also sees the neighbour's tent. "
                         "It is recorded in the checkpoint meta, so --score/--eval follow it. Default: analytic, "
                         "raycast with --course (which requires it).")
    ap.add_argument("--eval", default="")
    ap.add_argument("--score-envs", dest="score_envs", type=int, default=512,
                    help="envs for --score/--eval. 512 is the historical default; the run-to-run "
                         "spread at that size is about 0.012 of tracking, so raise it before "
                         "trusting a small difference between two checkpoints.")
    ap.add_argument("--score", default="")
    ap.add_argument("--score-perceive", dest="score_perceive", choices=("on", "off"), default="",
                    help="score with the camera-limited scan on or off, whatever the checkpoint was "
                         "trained with — how much does the policy lose when it only sees what a "
                         "forward depth camera could have seen?")
    ap.add_argument("--score-source", dest="score_source", choices=("analytic", "raycast"), default="",
                    help="score against this height source instead of the one the checkpoint was "
                         "trained on — the deliberate mismatch experiment (how much does the "
                         "difference between the two oracles cost a trained policy?)")
    ap.add_argument("--score-noise", dest="score_noise", type=float, default=None,
                    help="score with this elevation-map error (m) instead of the checkpoint's own; "
                         "implies --score-perceive on")
    ap.add_argument("--score-level", dest="score_level", type=int, default=None,
                    help=f"start every lane at this curriculum level (0..{N_LEVELS - 1}) instead of "
                         "climbing from 0 — the only way a 900-step score reaches the tall risers")
    ap.add_argument("--score-push", dest="score_push", type=float, default=0.0,
                    help="random shoves (m/s) during scoring, as --push-vel does in training")
    ap.add_argument("--score-json", dest="score_json", default="",
                    help="append the score as one JSON line to this file (the seed-array matrix)")
    ap.add_argument("--seed", type=int, default=0,
                    help="torch + env seed for training and scoring. Nothing in a run is seeded "
                         "otherwise, so two runs of one config differ by the command sampler alone.")
    ap.add_argument("--eval-json", dest="eval_json", default="",
                    help="with --eval: also write the steering result (worst ratio + per-command "
                         "errors) to this JSON file")
    ap.add_argument("--select", choices=("trainstat", "final"), default="trainstat",
                    help="what --out holds at the end. trainstat = the in-run best by level + "
                         "0.01*track among gated logs (the historical rule; at a clamped level it is "
                         "a max over single-step stochastic readings); final = the last weights.")
    ap.add_argument("--snapshot-every", dest="snapshot_every", type=int, default=0,
                    help="also save <out>_itNNNNN.pt every N iterations (0 = off), for scoring a "
                         "learning curve afterwards")
    ap.add_argument("--init-level", dest="init_level", type=int, default=None,
                    help=f"start every stair lane at this curriculum level (0..{N_LEVELS - 1}) "
                         "instead of 0 — skips the ~125-iteration re-climb of a continued run")
    ap.add_argument("--entropy", type=float, default=0.0, help="PPO entropy coefficient")
    ap.add_argument("--target-kl", dest="target_kl", type=float, default=0.02,
                    help="PPO early-stop KL per epoch (<= 0 disables the stop)")
    ap.add_argument("--epochs", type=int, default=5, help="PPO epochs per iteration")
    ap.add_argument("--minibatches", type=int, default=4, help="PPO minibatches per epoch")
    ap.add_argument("--clip", type=float, default=0.2, help="PPO ratio (and value) clip")
    ap.add_argument("--metrics", default="",
                    help="append one JSON line per log (PPO diagnostics, curriculum, counters) here "
                         "(default: metrics.jsonl next to --out)")
    # ---- the mixed-terrain course and its training ingredients (2026-09-13). Every one is off by default. ----------
    ap.add_argument("--course", action="store_true",
                    help="train on spot_course's lanes: flat / stairs / hills / cross-tilted bands / rough ground, six "
                         "levels per family, per-family promotion, IK-stance spawns held through a settle. Implies "
                         "--height-source raycast, honest termination and real torque limits unless overridden.")
    ap.add_argument("--course-shares", dest="course_shares", default="",
                    help="lane shares, e.g. 'flat=0.15,stairs=0.35,hills=0.20,cross=0.15,rough=0.15' (the default)")
    ap.add_argument("--rough-backend", dest="rough_backend", choices=sc.ROUGH_BACKENDS, default="trimesh",
                    help="rough-ground collider: one cooked trimesh per lane (the BVH is that very mesh) or a height "
                         "field, whose BVH copy measured a median 8.5 mm off the PhysX surface (spot_course.py), so "
                         "the scan and the honest termination would not see what the feet touch")
    ap.add_argument("--spawn-hold", dest="spawn_hold", type=int, default=SPAWN_HOLD,
                    help="control ticks a hills / cross / rough spawn holds its IK joints at every episode start")
    ap.add_argument("--termination", choices=("legacy", "honest"), default=None,
                    help="legacy (up < 0.35 or base_above < 0.18) or honest (spot_feet: tilt held 0.2 s, or a base / "
                         "upper-leg sample on the terrain; the knee end does not terminate). Default: honest with "
                         "--course, legacy otherwise")
    ap.add_argument("--drive-limits-are-forces", dest="drive_limits_are_forces",
                    action=argparse.BooleanOptionalAction, default=None,
                    help="joint effort caps as real torques (45/45/115 N·m). Default: on with --course, off otherwise "
                         "(the impulse caps every earlier checkpoint trained on)")
    ap.add_argument("--shove-ramp", dest="shove_ramp", default="",
                    help="'start,end,iters': interval shoves (push_mode 'interval') on every non-flat lane, dv ~ U(dv_min, "
                         "push_max) with push_max ramped linearly from start to end m/s over the first `iters` "
                         "iterations (E1: 0.5,2.0,600). Empty = off. Not with --push-vel.")
    ap.add_argument("--shove-interval", dest="shove_interval", default="3,8",
                    help="'lo,hi' seconds between interval shoves, drawn per shove")
    ap.add_argument("--shove-dv-min", dest="shove_dv_min", type=float, default=0.3)
    ap.add_argument("--foot-mu-dr", dest="foot_mu_dr", default="",
                    help="'lo,hi,n': each env's feet get one of n friction buckets spaced over [lo, hi] (restitution 0, "
                         "'min' combine) and the ground, tents and course boxes an explicit --terrain-mu 'min' material "
                         "(E1: 0.3,1.0,8). Rough-ground colliders keep the default material (0.5 'average'): the "
                         "bindings take no material for a trimesh, so a foot there reads min(mu, 0.5). Empty = off.")
    ap.add_argument("--terrain-mu", dest="terrain_mu", type=float, default=1.0)
    ap.add_argument("--stand-mode", dest="stand_mode", action="store_true",
                    help="a zero command freezes the gait clock and shows the policy a (0,0) clock, zeroes the "
                         "imitation weight and charges --w-stand * mean(joint_vel^2). Deploy must send the same sentinel.")
    ap.add_argument("--w-stand", dest="w_stand", type=float, default=W_STAND)
    # ---- wave 2 (2026-09-13): what the pilot policy was not ready for in spot_slam: resets and a key jumping 0 -> full ---
    ap.add_argument("--course-ladder", dest="course_ladder", choices=tuple(sc.LADDERS), default="hard",
                    help="hills / cross / rough levels and the course stair risers (spot_course.LADDERS). hard (default): "
                         "rough 0.03-0.25 m, cross 5-25 deg, risers 0.05-0.23 m, hills 4-25 deg; pilot: the 2026-09-13 "
                         "pilot's ladder (rough 0.02-0.17, cross 4-20, the default risers)")
    ap.add_argument("--cmd-switch", dest="cmd_switch", default="",
                    help="'lo,hi,p_standgo': resample every env's command every U(lo, hi) s; with probability p_standgo "
                         "the switch is abrupt (stand -> vx U(1.0, 1.5), or walking -> exactly 0), otherwise the usual "
                         "sampler. Suggested 3,8,0.3. Empty = off (the CMD_MIN..CMD_MAX step timer)")
    ap.add_argument("--reset-noise", dest="reset_noise", default="",
                    help="'q_std,drop_max,v_max': joint preset + N(0, q_std) rad, spawn height + U(0, drop_max) m, base "
                         "velocity U(-v_max, v_max) m/s per horizontal axis and yaw rate U(-0.5, 0.5) rad/s on every reset. "
                         "Suggested 0.08,0.06,0.3. Empty = off")
    ap.add_argument("--reset-nohold-frac", dest="reset_nohold_frac", type=float, default=0.3,
                    help="with --reset-noise: the share of flat and stair resets that get no joint hold at all")
    ap.add_argument("--w-place", dest="w_place", type=float, default=0.0,
                    help=f"foot-placement reward on stair treads: per touchdown min(d_edge, {PLACE_SAT}) / {PLACE_SAT}, "
                         f"-{PLACE_PERCH} if perched on a nosing. Needs honest termination (the touchdowns). 0 = off")
    # ---- walk + jump in one policy (2026-09-14): every default is the walking trainer as before ----
    ap.add_argument("--jump", default="", metavar="LO,HI",
                    help="a jump trigger every U(LO, HI) s on the flat lanes, the 98-d observation (the jump flag and phase "
                         "appended) and the jump rewards (spot_steps_env JUMP_*); a 96-d warm start is widened with zero "
                         "weights on the new inputs. Needs the honest termination (--course). '' = off")
    ap.add_argument("--w-jump-imit", dest="w_jump_imit", type=float, default=2.0, help="the jump reference's weight")
    ap.add_argument("--qd-max", dest="qd_max", type=float, default=None,
                    help="rate-limit the drive targets to V rad/s (spot_recovery_env's limit; default none)")
    ap.add_argument("--self-collision", dest="self_collision", action="store_true",
                    help="legs collide with the body and each other (the recovery and jump plant)")
    ap.add_argument("--eval-jump-steps", dest="eval_jump_steps", type=int, default=0,
                    help="with --jump, after training: this many deterministic steps on the training env; jump and fall "
                         "counters -> <out stem>_jumpeval.json")
    args = ap.parse_args()
    if not tp.HAS_PHYSX or not torch.cuda.is_available():
        print("need PhysX + CUDA"); sys.exit(0)
    if args.score:
        see = None if not args.score_perceive else args.score_perceive == "on"
        if args.score_noise is not None:
            see = True
        score_checkpoint(args.score, k=args.score_envs,
                         height_source=args.score_source or None,
                         perceive=see, seed=args.seed, perceive_noise=args.score_noise,
                         start_level=args.score_level, push_vel=args.score_push)
        if args.score_json:
            import json
            rec = {"checkpoint": os.path.basename(args.score), **score_checkpoint.last}
            with open(args.score_json, "a") as f:
                f.write(json.dumps(rec) + "\n")
        return
    if args.eval:
        res = eval_flat_steering(SpotStepsEnv, args.eval, k=args.score_envs,
                                 height_source=True, seed=0)
        if args.eval_json and res is not None:
            import json
            with open(args.eval_json, "w") as f:
                json.dump({"checkpoint": os.path.abspath(args.eval), **res}, f, indent=1)
            print(f"steering -> {args.eval_json}")
        return

    torch.manual_seed(args.seed)
    if args.course and (args.perceive or args.graph):
        ap.error("--course needs the eager raycast: no --perceive, no --graph")
    if args.shove_ramp and args.push_vel > 0:
        ap.error("--shove-ramp and --push-vel are two shove modes; pick one")
    termination = args.termination or ("honest" if args.course else "legacy")
    dlf = args.drive_limits_are_forces if args.drive_limits_are_forces is not None else bool(args.course)
    if args.w_place > 0 and termination != "honest":
        ap.error("--w-place reads the honest termination's touchdowns: add --termination honest")
    kw = {}
    shove = None
    if args.shove_ramp:
        s0, s1, s_it = (float(v) for v in args.shove_ramp.split(","))
        lo, hi = (float(v) for v in args.shove_interval.split(","))
        shove = (s0, s1, int(s_it))
        kw.update(push_mode="interval", push_interval=(lo, hi), push_dv_min=args.shove_dv_min, push_max=s0)
    foot_mu_np = None
    if args.foot_mu_dr:
        m0, m1, mn = args.foot_mu_dr.split(",")
        buckets = np.linspace(float(m0), float(m1), int(mn))
        # its own seeded stream: which env gets which bucket is a function of --seed alone
        foot_mu_np = buckets[np.random.default_rng(args.seed + 101).integers(0, len(buckets), args.envs)]
        kw.update(foot_mu=foot_mu_np, foot_combine="min",
                  terrain_material=(args.terrain_mu, args.terrain_mu, 0.0, "min", "min"))
    if args.course:
        kw.update(course=True, course_shares=args.course_shares or None, rough_backend=args.rough_backend,
                  spawn_hold=args.spawn_hold)
    if args.course:
        kw.update(course_ladder=args.course_ladder)
    cmd_switch = tuple(float(v) for v in args.cmd_switch.split(",")) if args.cmd_switch else None
    reset_noise = tuple(float(v) for v in args.reset_noise.split(",")) if args.reset_noise else None
    if cmd_switch is not None:
        kw.update(cmd_switch=cmd_switch)
    if reset_noise is not None:
        kw.update(reset_noise=reset_noise, reset_nohold_frac=args.reset_nohold_frac)
    if args.stand_mode:
        kw.update(stand_mode=True, w_stand=args.w_stand)
    if args.w_place > 0:
        kw.update(w_place=args.w_place)
    if termination != "legacy":
        kw.update(termination=termination)
    if dlf:
        kw.update(drive_limits_are_forces=True)
    jump = tuple(float(v) for v in args.jump.split(",")) if args.jump else None
    if jump is not None:
        if termination != "honest":
            ap.error("--jump reads the instrument's foot clearance: it needs the honest termination (--course)")
        kw.update(jump=jump, w_jump_imit=args.w_jump_imit)
    if args.qd_max:
        kw.update(qd_max=args.qd_max)
    if args.self_collision:
        kw.update(self_collision=True)
    height_source = args.height_source or ("raycast" if args.course else "analytic")
    env = SpotStepsEnv(num_envs=args.envs, device="cuda", seed=args.seed, perceive=args.perceive,
                       perceive_noise=args.perceive_noise, graph=args.graph,
                       cadence_jitter=args.cadence_jitter,
                       push_vel=args.push_vel, push_prob=args.push_prob,
                       height_source=None if args.perceive else height_source, **kw)
    if args.course:
        fam = env._cc.cpu().numpy()
        print("course lanes: " + ", ".join(f"{n} {int((fam == c).sum())}" for c, n in enumerate(sc.FAMILIES))
              + f" | rough backend {args.rough_backend} ({env._course.n_rough_tris} tris), ground x {env.ground_extent}")
    if shove is not None:
        print(f"interval shoves: every U({args.shove_interval}) s on the non-flat lanes, dv U({args.shove_dv_min}, "
              f"push_max), push_max {shove[0]} -> {shove[1]} m/s over iterations 0-{shove[2]}")
    if foot_mu_np is not None:
        print(f"foot friction DR: {args.foot_mu_dr} buckets 'min', terrain {args.terrain_mu} 'min' "
              f"(rough colliders on the default material)")
    if args.course:
        c = env._course
        print(f"course ladder {args.course_ladder}: stair risers {list(env.riser_list)}, rough {list(c.rough_amps)} m, "
              f"cross {list(c.cross_degs)} deg (pivot {c.cross_pivot:.2f} m), hills {list(c.hill_degs)} deg")
    if cmd_switch is not None:
        print(f"command switches every U({cmd_switch[0]:g}, {cmd_switch[1]:g}) s, abrupt stand<->go with p {cmd_switch[2]:g}")
    if reset_noise is not None:
        print(f"reset noise: q N(0, {reset_noise[0]:g}) rad, drop U(0, {reset_noise[1]:g}) m, v U(+-{reset_noise[2]:g}) m/s, "
              f"no hold on {args.reset_nohold_frac:g} of flat/stair resets")
    if jump is not None or args.qd_max or args.self_collision:
        print(f"jump triggers every U({args.jump}) s on the flat lanes, obs {env.obs_dim}-d, reference weight "
              f"{args.w_jump_imit}" if jump is not None else "no jump triggers",
              f"| joint targets <= {args.qd_max} rad/s" if args.qd_max else "| no joint target limit",
              f"| self-collision {'on' if args.self_collision else 'off'}")
    print(f"termination {termination}, drive limits {'torques' if dlf else 'impulses'}"
          + (", stand mode" if args.stand_mode else "") + (f", w_place {args.w_place}" if args.w_place > 0 else ""))
    if env.rays is not None:
        print(f"terrain height by raycast: {env.rays}")
    if env.percept is not None:
        print(f"camera-limited scan: {env.percept}")
    if args.push_vel > 0:
        print(f"random shoves: up to {args.push_vel} m/s "
              f"({args.push_vel * 24.0:.0f} N*s) at p={args.push_prob} per env per step, stair lanes only")
    if args.init_level is not None:
        if not 0 <= args.init_level < N_LEVELS:
            ap.error(f"--init-level must lie in 0..{N_LEVELS - 1}")
        # Before PPO(...): its constructor runs the first reset(), which spawns each lane at env.level.
        env.level.fill_(args.init_level)
        print(f"curriculum starts at level {args.init_level} (~{RISERS[args.init_level]:.02f} m risers)")
    aux = make_aux_loss(args.sym_coef) if args.sym_coef > 0 else None
    # height_source rides in the meta so --score and --eval rebuild the env the way it was trained.
    ppo = PPO(env, ACT_DIM, hidden=HIDDEN, lr=args.lr, horizon=args.horizon, log_std_init=-1.5,
              entropy=args.entropy, clip=args.clip, epochs=args.epochs, minibatches=args.minibatches,
              target_kl=args.target_kl if args.target_kl > 0 else None, normalize_obs=True,
              meta={**CONFIG, "height_source": env.height_source, "perceive": args.perceive,
                    "perceive_noise": args.perceive_noise,
                    # Lineage: which gait this fine-tune started from. spot_steps.pt records none,
                    # so whether it came off the old heightfield -> stairs chain or straight off the
                    # scratch gait is not recoverable from the file — only guessable from its date.
                    "warmstart": os.path.basename(args.warmstart) if args.warmstart else "",
                    "iters": args.iters, "envs": args.envs, "seed": args.seed,
                    "cadence_jitter": args.cadence_jitter,
                    "imit_anneal": args.imit_anneal,
                    "push_vel": args.push_vel, "push_prob": args.push_prob,
                    # Which lanes the shoves hit. Not recorded before 2026-09-12, which is why
                    # spot_steps_push.pt (flat lanes shoved) and _push2.pt (stair lanes only) carry
                    # identical meta.
                    "push_lanes": "stairs", "push_mode": "random",
                    "select": args.select, "snapshot_every": args.snapshot_every,
                    "init_level": args.init_level, "lr": args.lr, "horizon": args.horizon,
                    "entropy": args.entropy, "target_kl": args.target_kl, "epochs": args.epochs,
                    "minibatches": args.minibatches, "clip": args.clip, "sym_coef": args.sym_coef,
                    "graph": args.graph, "gate": args.gate, "fell_max": args.fell_max,
                    # the course and its ingredients (2026-09-13); the deploy side reads stand_mode for the sentinel
                    "course": bool(args.course), "course_shares": sc.parse_shares(args.course_shares or None)
                    if args.course else None, "rough_backend": args.rough_backend if args.course else None,
                    "spawn_hold": args.spawn_hold if args.course else None,
                    "termination": termination, "drive_limits_are_forces": dlf,
                    "shove_ramp": args.shove_ramp, "shove_interval": args.shove_interval if shove else "",
                    "shove_dv_min": args.shove_dv_min if shove else None,
                    "foot_mu_dr": args.foot_mu_dr, "terrain_mu": args.terrain_mu if args.foot_mu_dr else None,
                    "stand_mode": bool(args.stand_mode), "w_stand": args.w_stand if args.stand_mode else None,
                    "w_place": args.w_place, "place_sat": PLACE_SAT, "place_perch": PLACE_PERCH,
                    "course_ladder": args.course_ladder if args.course else None,
                    "risers": list(env.riser_list),
                    "course_values": ({"rough": list(env._course.rough_amps), "cross": list(env._course.cross_degs),
                                       "hills": list(env._course.hill_degs)} if args.course else None),
                    "cmd_switch": args.cmd_switch, "reset_noise": args.reset_noise,
                    "reset_nohold_frac": args.reset_nohold_frac if reset_noise is not None else None,
                    **env.jump_config()},
              aux_loss=aux)
    if shove is not None:
        ppo.meta["push_lanes"], ppo.meta["push_mode"] = "non-flat", "interval"
    if aux is not None:
        print(f"symmetry augmentation ON (coef {args.sym_coef})")
    if args.warmstart and os.path.exists(args.warmstart):
        src, src_norm, src_meta = load_policy(args.warmstart, device="cuda")
        if src_meta.get("obs_dim") == OBS_DIM and env.obs_dim > OBS_DIM:
            # a 96-d walking policy into the 98-d jump layout: its weights on the first 96 inputs, zeros on the new ones, so
            # the first rollout walks exactly as the warm start did; the norm's new entries start at mean 0, var 1
            own, n_wide = ppo.ac.state_dict(), 0
            for name, v in src.state_dict().items():
                if own[name].shape == v.shape:
                    own[name].copy_(v)
                else:
                    own[name].zero_(); own[name][:, :v.shape[1]].copy_(v); n_wide += 1
            ppo.ac.load_state_dict(own)
            if ppo.norm is not None and src_norm is not None:
                st_ = src_norm.state()
                ppo.norm.mean[:OBS_DIM] = st_["mean"]; ppo.norm.var[:OBS_DIM] = st_["var"]
                ppo.norm.mean[OBS_DIM:] = 0.0; ppo.norm.var[OBS_DIM:] = 1.0
                ppo.norm.count, ppo.norm.clip = float(st_["count"]), float(st_["clip"])
            print(f"continued from {os.path.basename(args.warmstart)} widened {OBS_DIM} -> {env.obs_dim} inputs "
                  f"({n_wide} first layers, zero weights on the jump channels; actor+critic+log_std+norm)")
        elif src_meta.get("obs_dim") == env.obs_dim:
            # Already a terrain policy, not the 50-d base gait: continue it whole. This is what a
            # heightfield -> stairs chain needs, and until now it could not be run at all — the
            # expander below asserts on anything that is not 50-d, so handing it spot_hf.pt
            # stopped the trainer dead.
            ppo.ac.load_state_dict(src.state_dict())
            if ppo.norm is not None and src_norm is not None:
                ppo.norm.load(src_norm.state())
            print(f"continued from {os.path.basename(args.warmstart)} "
                  f"({env.obs_dim}-d terrain policy: full actor+critic+log_std+norm)")
        else:
            warmstart_scratch_to_terrain(ppo.ac, ppo.norm,
                                         args.warmstart, n_keep=50, device="cuda")
    else:
        print(f"(warmstart path not found: {args.warmstart} — starting from scratch)")
    if args.graph:
        env.reset()
        print(f"CUDA-graph step ON — replay verified to "
              f"{env.verify_graph(steps=32):.1e} against a recomputed observation")
    sanity_walk(env, ppo.ac, ppo.norm)
    flat0 = stochastic_flat_baseline(env, ppo.ac, ppo.norm)
    gate = args.gate * flat0
    print(f"flat-steering gate = {gate:.3f}  (= {args.gate:.2f} x warm-start STOCHASTIC flat tracking {flat0:.3f})")

    import json
    stem = os.path.splitext(args.out)[0]
    latest = stem + "_latest.pt"
    best = [-1e9]
    seen = [0]
    LOG_EVERY = 20
    metrics_path = args.metrics or os.path.join(os.path.dirname(os.path.abspath(args.out)), "metrics.jsonl")
    # the checkpoints land here too, so make the directory before the first write rather than at the
    # first save, 20 iterations in
    for d in {os.path.dirname(os.path.abspath(args.out)), os.path.dirname(os.path.abspath(metrics_path))}:
        os.makedirs(d, exist_ok=True)

    def metric(rec):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    metric({"type": "start", "out": os.path.abspath(args.out), "argv": sys.argv[1:], "meta": ppo.meta,
            "flat_gate": gate})
    env.reset_stats(episodes=False)      # the counters below cover one log window each

    def log(msg):
        trk, ftrk, lvl = env.last_track, env.last_flat_track, env.last_level
        ppo.save(latest)
        ok = ftrk >= gate and env.last_fell <= args.fell_max
        # STAIRS: the objective is climb HEIGHT (curriculum level), not track — track DROPS with
        # difficulty, so best-by-track would pick the easy early checkpoint. Select by level (track ties).
        score = lvl + 0.01 * trk
        seen[0] += LOG_EVERY
        anneal = ""
        if args.imit_anneal >= 0 and seen[0] >= args.imit_anneal:
            # Let go of the teacher, but only while steering is still healthy: the anchor exists
            # because steering regressed without it, so a decay that ignores that just reintroduces
            # the bug it was added to fix.
            if ok:
                frac = (seen[0] - args.imit_anneal) / max(args.iters - args.imit_anneal, 1)
                w = W_IMIT * max(0.0, 1.0 - frac)
                env.set_imit_weight(w)
                anneal = f" | imit {w:.3f}"
            else:
                anneal = " | imit HELD"
        mark = ""
        if args.select == "trainstat" and score > best[0] and ok:
            best[0] = score
            ppo.save(args.out)
            mark = "  <- saved best"
        print(f"{msg} | track {trk:.3f} | flat {ftrk:.3f}{'' if ok else ' LOW!'} | "
              f"level {lvl:.2f}/{N_LEVELS - 1} | clear {env.last_clear:.2f} | "
              f"fell {env.last_fell:.3f}{anneal}{mark}")
        rows = env.episode_stats()["rows"]
        if env.block_names is None:
            counters = {r["lane_type"]: {k: v for k, v in r.items() if k not in ("by_episode", "instrument")}
                        for r in rows}
        else:
            counters = {r["block_name"]: {k: v for k, v in r.items() if k not in ("by_episode", "instrument")}
                        for r in rows}
        course = {}
        if args.course:
            import spot_feet as sf
            course = {"course_levels": env.course_levels(), "place_raw": env.last_place}
            if shove is not None:
                course["push_max"] = float(env._push_max.item())
            for r in rows:
                if r.get("block_name") == "stairs" and "instrument" in r:
                    pl = sf.placement_summary(r["instrument"])
                    course["placement_stairs"] = {d: {k: pl[d][k] for k in ("touchdowns", "median_d_edge",
                                                                            "p_edge_lt05", "perched")}
                                                  for d in ("asc", "desc")}
            lv = course["course_levels"]
            print("        course " + " | ".join(
                f"{n} lvl {v['level']:.2f} (+{v['promoted']}/-{v['demoted']} of {v['episodes']})" for n, v in lv.items())
                  + f" | place raw {env.last_place:+.4f}/stair-step"
                  + (f" | push_max {course['push_max']:.2f}" if "push_max" in course else ""))
            bad = {r["block_name"]: (r["spawn"]["unreachable"], r["spawn"].get("hold_bad"), r["spawn"].get("hold_checked"))
                   for r in rows if "spawn" in r}
            if bad:
                course["spawn"] = {r["block_name"]: r["spawn"] for r in rows if "spawn" in r}
                print("        spawn IK-bad / hold-bad of hold-checked: "
                      + " | ".join(f"{n} {a}/{b} of {c}" for n, (a, b, c) in bad.items()))
        if cmd_switch is not None:
            course["switches"] = env.switch_counts()
            print("        switches (stand->go / go->stand of all): " + " | ".join(
                f"{n} {v['stand_to_go']}/{v['go_to_stand']} of {v['switches']}" for n, v in course["switches"].items()))
        if reset_noise is not None:
            course["spawn_families"] = sf_ = env.spawn_families()
            print("        spawn falls <=1 s held / unheld (hold-bad of checked): " + " | ".join(
                f"{n} {v['early_falls_held']}/{v['held']} {v['early_falls_unheld']}/{v['resets'] - v['held']} "
                f"({v['hold_bad']}/{v['hold_checked']})" for n, v in sf_.items()))
        if jump is not None:
            course["jump"] = js = env.jump_stats(reset=True)
            _f = lambda v, n=3: "n/a" if v is None else f"{v:.{n}f}"
            print("        jump " + " | ".join(
                f"{n} {v['windows']}: jumped {_f(v['jumped'])} landed {_f(v['landed_up'])} fell {_f(v['fell'])} flight "
                f"{_f(v['flight_s'], 2)} s clear {_f(v['clearance_m'])} vz {_f(v['vz_max'], 2)}"
                for n, v in js.items() if isinstance(v, dict) and n != "all")
                  + f" | uncommanded flights/env-min {_f(js['uncommanded_flights_per_env_min'], 3)}")
        env.reset_stats(episodes=False)
        metric({"type": "log", **ppo.last_log, **ppo.diagnostics(), "track": trk, "flat": ftrk,
                "level": lvl, "clear_legacy": env.last_clear, "fell": env.last_fell, "gate_ok": ok,
                "imit_w": float(env._imit_w.item()), "saved_best": bool(mark), "counters": counters, **course})

    def snapshot(it):
        if it % args.snapshot_every == 0:
            ppo.save(f"{stem}_it{it:05d}.pt")

    def on_iter(it):
        if shove is not None:
            # the shove ramp, applied for the NEXT iteration's rollouts: linear from start to end over iterations 0..n
            s0, s1, n = shove
            env.set_push_max(s0 + (s1 - s0) * min(1.0, it / max(n, 1)))
        if args.snapshot_every > 0:
            snapshot(it)

    ppo.learn(args.iters, log_every=LOG_EVERY, on_log=log,
              on_iter=on_iter if (args.snapshot_every > 0 or shove is not None) else None)
    ppo.save(latest)
    if jump is not None and args.eval_jump_steps > 0:
        # the final weights, deterministic, on the training env as it ends (course levels, shove size, friction)
        pol = (lambda o: ppo.ac.act_mean(ppo.norm.norm(o))) if ppo.norm is not None else ppo.ac.act_mean
        env.reset_stats(episodes=True); env.jump_stats(reset=True)
        obs = env.reset()
        fl, ftk = [], []
        for _ in range(args.eval_jump_steps):
            obs, _, _, _, _ = env.step(pol(obs))
            fl.append(env.last_fell); ftk.append(env.last_flat_track)
        ev = {"checkpoint": os.path.abspath(latest), "steps": args.eval_jump_steps, "envs": args.envs,
              "jump": env.jump_stats(reset=True), "fell_per_step": sum(fl) / len(fl), "flat_track": sum(ftk) / len(ftk),
              "rows": [{k: v for k, v in r.items() if k not in ("by_episode", "instrument")}
                       for r in env.episode_stats()["rows"]]}
        with open(stem + "_jumpeval.json", "w") as f:
            json.dump(ev, f, indent=1, default=str)
        a_ = ev["jump"]["all"]
        print(f"[jump eval] {args.eval_jump_steps} deterministic steps: windows {a_['windows']} jumped {a_['jumped']} landed "
              f"{a_['landed_up']} fell {a_['fell']} | fell/step {ev['fell_per_step']:.4f} flat track {ev['flat_track']:.3f}"
              f" -> {stem}_jumpeval.json")
    if args.select == "final":
        ppo.save(args.out)
        print(f"saved -> {args.out} = {latest} (final weights, --select final) | metrics -> {metrics_path}")
    else:
        print(f"saved -> {args.out} (best level-score {best[0]:.3f}, steering gate {gate:.3f}) + {latest} (final)"
              f" | metrics -> {metrics_path}")
    print(f"next: python {os.path.basename(__file__)} --score {args.out}")


if __name__ == "__main__":
    main()
