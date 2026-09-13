#!/usr/bin/env bash
# Spot FALL RECOVERY wave 2 on IDUN (python/examples/spot/spot_recovery_env.py, train_spot_recovery.py): arm x seed, one
# big GPU each. Train, then evaluate the final weights from 2048 fallen starts at L3 (40% back, 40% side, 20% any
# orientation) and again from L1 and L2 starts to eval.json, and write a manifest.
#
#   mkdir -p /cluster/work/$USER/logs                                                          # ONCE
#   sbatch --array=0-3 --export=ALL,ITERS=3000,RUNS_NAME=spot_recovery2 scripts/idun/spot_recovery.slurm   # A s0, B s0, A s1, B s1
#   sbatch --array=0-1 --export=ALL,ITERS=20,RUNS_NAME=spot_recovery2_check scripts/idun/spot_recovery.slurm
#
# Index -> (arm, seed): arm = ID % 2, seed = ID / 2, so 0 = A s0, 1 = B s0, 2 = A s1, 3 = B s1.
#   A  --action-scale 0.5
#   B  --action-scale 1.0
# Both continue the shipped walking policy (python/examples/spot/spot_steps.pt, 96-d) whole, with its actor output and
# the norm's last_act rescaled by 0.2 / S so the first tick commands the walking policy's targets. Both also get
#   --log-std-min (A -0.8, B -1.5) --hard-frac 0.35 --w-prog $W_PROG --up-slope $UP_SLOPE --fallen-effort-free --w-down $W_DOWN
# (spot_recovery_env.py's docstring and train_spot_recovery.py --help say what each does). The floors give both arms the
# same 0.22 rad of joint-target noise, so the arms differ in reach alone (see LOG_STD_MIN below: -0.8 at S 1.0 measured
# 0% success even from standing).
# Weights, measured 2026-09-13 on the laptop (K=256, no training): with unit weights a rolling tick carries |gamma phi' -
# phi| 0.022 mean / 0.047 p90, so W_PROG 5 pays ~0.11 per rolling tick (effort penalties in the 20-iteration smoke:
# torque -0.010, impact -0.015, arate / limit -0.001), and the (1 - gamma) W drain on a standing tick is 0.05 against
# +2.67 standing reward. W_DOWN 0.03 per down tick is 1/4 of that rolling pay, and -8 over a 5.5 s episode spent lying. Both run the walking plant (stiff gains, real torque limits
# 45/45/115 N·m) with self-collision, flat ground, 6 s episodes, the fall curriculum L0 (tilted <= 25 deg) -> L3 (back /
# side / anyhow), 18% standing starts.
# The pilot (spot_recovery1: arm A warm, arm B fresh, scale 0.2, none of these flags, 1000 iterations) is this script
# at bdb52b34.
#
# Walltime: 3000 iterations x 2048 envs x 32 steps = 197M steps. An H100 ran the pilot at 82.7k steps/s (40 min for
# this); an A100 at half that is 80 min, inside the 110 min left before the 10 min eval reserve. A task that still runs
# out is stopped by the timeout below and evaluates the _latest checkpoint (saved every log).
# A task refuses a run directory that already holds a manifest.json (OVERWRITE=1 to allow).
#
#SBATCH --account=share-ie-iir
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
# H100 or H200, 6 cores, 32 GB: the loop uses ~3 cores, and an h100-only 12-core request left 3 of 4
# array tasks pending on (Resources) on 2026-09-13 (free GPUs sit on CPU-starved nodes). No A100: it ran this loop at
# about half the H100 speed, and an array waits for its slowest task.
#SBATCH --constraint=h100|h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=spotrecov
#SBATCH --output=/cluster/work/laht/logs/spotrecov-%A_%a.out
#SBATCH --error=/cluster/work/laht/logs/spotrecov-%A_%a.out
set -euo pipefail

WORK=${WORK:-/cluster/work/$USER}
SRC=${SRC:-$WORK/threepp}
RUNS_NAME=${RUNS_NAME:-spot_recovery2}
RUNS=${RUNS:-$WORK/$RUNS_NAME}
WARM=${WARM:-$SRC/python/examples/spot/spot_steps.pt}
ENVS=${ENVS:-2048}
ITERS=${ITERS:-3000}
EVAL_ENVS=${EVAL_ENVS:-2048}
SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-250}
EXTRA=${EXTRA:-}
EVAL_RESERVE_MIN=${EVAL_RESERVE_MIN:-10}
SCALE_A=${SCALE_A:-0.5}
SCALE_B=${SCALE_B:-1.0}
LOG_STD_MIN=${LOG_STD_MIN:--0.8}
# Per-arm override. log_std is in ACTION units, so the same floor is S x wider in joint targets: laptop rollouts on
# 2026-09-13 (K=256, the rescaled walking policy + that noise, the wave 2 mix, no training) stood 63% / recovered 37% at
# S 0.5 / -0.8 (0.22 rad) but 0% / 0% at S 1.0 / -0.8 (0.45 rad, nothing promoted), and 100% / 63% at a 0.09 rad
# floor (S 1.0 / -2.4, S 0.5 / -1.7, the pilot's S 0.2 / -1.5 alike).
LOG_STD_MIN_A=${LOG_STD_MIN_A:-$LOG_STD_MIN}
LOG_STD_MIN_B=${LOG_STD_MIN_B:--1.5}   # 0.22 rad of target noise at S 1.0, the same as arm A's -0.8 at S 0.5
HARD_FRAC=${HARD_FRAC:-0.35}
W_PROG=${W_PROG:-5.0}
UP_SLOPE=${UP_SLOPE:-0.5}
W_DOWN=${W_DOWN:-0.03}

. "$WORK/threepp-env.sh"

ID=${SLURM_ARRAY_TASK_ID:-0}
SEED=$(( ID / 2 ))
if [ $(( ID % 2 )) -eq 0 ]; then ARM=A; SCALE=$SCALE_A; LSM=$LOG_STD_MIN_A; else ARM=B; SCALE=$SCALE_B; LSM=$LOG_STD_MIN_B; fi
TAG="recov${ARM}_s${SEED}"
RUN="$RUNS/$TAG"
if [ -e "$RUN/manifest.json" ] && [ "${OVERWRITE:-0}" != 1 ]; then
    echo "!! $RUN already holds a finished run (manifest.json): pick another RUNS_NAME, or OVERWRITE=1"; exit 1
fi
mkdir -p "$RUN" "$WORK/logs"
LS=${LOCALSCRATCH:-/localscratch/$SLURM_JOB_ID}
[ -d "$LS" ] || LS=/localscratch/$USER/${SLURM_JOB_ID:-$$}
mkdir -p "$LS/tmp"
export TMPDIR="$LS/tmp"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
trap 'rm -rf "$LS" 2>/dev/null || true' EXIT

cd "$SRC/python/examples/spot"
[ -s "$WARM" ] || { echo "!! no warm-start checkpoint at $WARM"; exit 1; }
WARM_SHA=$(sha256sum "$WARM" | cut -d' ' -f1)
WAVE_ARGS=(--action-scale "$SCALE" --log-std-min "$LSM" --hard-frac "$HARD_FRAC" --w-prog "$W_PROG"
           --up-slope "$UP_SLOPE" --fallen-effort-free --w-down "$W_DOWN")
GIT=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
DIRTY=$(git -C "$SRC" status --porcelain 2>/dev/null | wc -l)
echo "== $TAG  arm=$ARM seed=$SEED envs=$ENVS iters=$ITERS  ${WAVE_ARGS[*]}  git=$GIT (+$DIRTY dirty)  node=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# ---- train ---------------------------------------------------------------------------------------
LIMIT_S=$(( (${SLURM_JOB_END_TIME:-0} > 0 ? SLURM_JOB_END_TIME - $(date +%s) : 2*3600) - EVAL_RESERVE_MIN*60 ))
[ "$LIMIT_S" -gt 600 ] || LIMIT_S=600
T0=$(date +%s)
set +e
# shellcheck disable=SC2086
timeout -s TERM "${LIMIT_S}s" python -u train_spot_recovery.py --envs "$ENVS" --iters "$ITERS" --seed "$SEED" \
    --arm "$ARM" --warmstart "$WARM" "${WAVE_ARGS[@]}" --out "$RUN/spot_recovery.pt" --metrics "$RUN/metrics.jsonl" \
    --snapshot-every "$SNAPSHOT_EVERY" $EXTRA 2>&1 | tee "$RUN/train.log"
TRC=${PIPESTATUS[0]}
set -e
T1=$(date +%s)
TIMED_OUT=$([ "$TRC" -eq 124 ] && echo True || echo False)
CKPT="$RUN/spot_recovery.pt"
[ -s "$CKPT" ] || CKPT="$RUN/spot_recovery_latest.pt"       # a timed-out run saved _latest every 20 iterations
[ -s "$CKPT" ] || { echo "!! no checkpoint produced"; exit 1; }
STEPS_S=$(grep -oE '[0-9.]+k? ?steps/s' "$RUN/train.log" | tail -1 || true)

# ---- evaluate from fallen starts: L3, then L1 and L2 --------------------------------------------------
rm -f "$RUN/eval.json"
set +e
python -u train_spot_recovery.py --eval "$CKPT" --eval-envs "$EVAL_ENVS" --eval-json "$RUN/eval.json" 2>&1 | tee "$RUN/eval.log"
ERC=${PIPESTATUS[0]}
set -e
T2=$(date +%s)

python - "$RUN/manifest.json" "$RUN/eval.json" <<EOF
import json, os, sys
ev = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else {}
fa = ev.get("total", {}).get("fallen", {})
bl = ev.get("by_level", {})
json.dump({
    "tag": "$TAG", "arm": "$ARM", "seed": $SEED, "warmstart": "$WARM", "warmstart_sha256": "$WARM_SHA",
    "wave_args": "${WAVE_ARGS[*]}", "action_scale": $SCALE,
    "envs": $ENVS, "iters": $ITERS, "extra": "$EXTRA", "runs": "$RUNS", "checkpoint": os.path.basename("$CKPT"),
    "train_exit": $TRC, "timed_out": $TIMED_OUT, "train_seconds": $((T1 - T0)), "throughput": "$STEPS_S",
    "eval_envs": $EVAL_ENVS, "eval_exit": $ERC, "eval_seconds": $((T2 - T1)),
    "eval_success_3s": fa.get("success_3s"), "eval_success_6s": fa.get("success_6s"),
    "eval_recovery_s": fa.get("recovery_s"), "eval_blowups": fa.get("blowups"), "eval_gave_up": fa.get("gave_up"),
    "eval_gave_up_L3": ev.get("gave_up_L3"),
    "eval_success_6s_by_level": {k: v["fallen"].get("success_6s") for k, v in bl.items()},
    "eval_gave_up_by_level": {k: v["fallen"].get("gave_up") for k, v in bl.items()},
    "eval_tau_p95_hx_hy_kn": ev.get("tau_p95_hx_hy_kn"), "eval_contact_ticks_per_ep": fa.get("contact_ticks_per_ep"),
    "git": "$GIT", "dirty_files": $DIRTY, "node": "$(hostname)",
    "slurm_job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}",
}, open(sys.argv[1], "w"), indent=1)
EOF
echo "== done $TAG: train $((T1 - T0)) s, eval $((T2 - T1)) s -> $RUN/eval.json"
[ "$TRC" -eq 0 ] || [ "$TRC" -eq 124 ] || exit "$TRC"
exit "$ERC"
