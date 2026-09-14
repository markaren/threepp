#!/usr/bin/env bash
# Spot JUMP wave 2 on IDUN (python/examples/spot/spot_jump_env.py, train_spot_jump.py): four variants, one big GPU each.
# Train, then evaluate the final weights: one jump per env from standing (trigger at 1 s, dx 30% zero else up to 0.4 m),
# one standing episode per env with no trigger, and one jump per env from a crouched start, to eval.json + a manifest.
#
#   mkdir -p /cluster/work/$USER/logs                                                          # ONCE
#   sbatch --array=0-3 --export=ALL,ITERS=3000,RUNS_NAME=spot_jump2 scripts/idun/spot_jump.slurm
#   sbatch --array=0-1 --export=ALL,ITERS=20,RUNS_NAME=spot_jump2_check scripts/idun/spot_jump.slurm
#
# Index -> variant, seed 0 each (the question is which signal gets Spot off the ground; seeds come after):
#   0 = A  targets <= 10 rad/s, lr 1e-4, --w-imit $W_IMIT
#   1 = B  targets <= 20 rad/s, lr 1e-4, --w-imit $W_IMIT
#   2 = C  targets <= 10 rad/s, lr 3e-4, --w-imit $W_IMIT
#   3 = D  targets <= 10 rad/s, lr 3e-4, --w-imit $W_IMIT_D (a stronger pull onto the reference)
# All continue the shipped fall-recovery policy (python/examples/spot/spot_recovery.pt, action scale 0.5) whole, on its
# plant: stiff gains, real torque limits 45/45/115 N·m, self-collision, flat ground; --w-qd-excess 0.02, --log-std-min -0.8,
# and wave 2's --w-vz $W_VZ (pay the peak upward base speed since the trigger), --crouch-frac $CROUCH_FRAC (that share of
# jump episodes starts crouched) and --w-imit (track a crouch-push-tuck joint reference during the jump window).
# Why: the pilot (spot_jump1, this script at 354fdcfc: 10 / 20 rad/s x 2 seeds, 1500 iterations, no shaping) never left
# the ground: 0 flights, every window missed, no falls. Local 150-iteration checks (K=1024) with --w-vz and --crouch-frac,
# then with --w-imit too, moved the peak upward speed only from 0.17 to 0.20-0.22 m/s (a take-off needs ~2), with PPO
# updates the size recovery wave 3's were (approx KL 0.004-0.006, 5 epochs run): the checks were short (the learning rate
# anneals to zero by the last iteration), hence 3000 iterations and a faster-learning pair.
#
# Walltime: 3000 iterations x 2048 envs x 32 steps = 197M steps; the pilot ran 85-94k steps/s on H100s, about 37 min.
# A task that runs out is stopped by the timeout below and evaluates the _latest checkpoint.
# A task refuses a run directory that already holds a manifest.json (OVERWRITE=1 to allow).
#
#SBATCH --account=share-ie-iir
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100|h200
#SBATCH --exclude=idun-01-01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=spotjump
#SBATCH --output=/cluster/work/laht/logs/spotjump-%A_%a.out
#SBATCH --error=/cluster/work/laht/logs/spotjump-%A_%a.out
set -euo pipefail

WORK=${WORK:-/cluster/work/$USER}
SRC=${SRC:-$WORK/threepp}
RUNS_NAME=${RUNS_NAME:-spot_jump2}
RUNS=${RUNS:-$WORK/$RUNS_NAME}
WARM=${WARM:-$SRC/python/examples/spot/spot_recovery.pt}
ENVS=${ENVS:-2048}
ITERS=${ITERS:-3000}
EVAL_ENVS=${EVAL_ENVS:-2048}
SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-250}
EXTRA=${EXTRA:-}
EVAL_RESERVE_MIN=${EVAL_RESERVE_MIN:-8}
W_QD_EXCESS=${W_QD_EXCESS:-0.02}
LOG_STD_MIN=${LOG_STD_MIN:--0.8}
W_VZ=${W_VZ:-40}
CROUCH_FRAC=${CROUCH_FRAC:-0.25}
W_IMIT=${W_IMIT:-2.0}
W_IMIT_D=${W_IMIT_D:-5.0}
SEED=${SEED:-0}

ID=${SLURM_ARRAY_TASK_ID:-0}
case "$ID" in
    0) ARM=A; QD=10; LR=1e-4; WI=$W_IMIT ;;
    1) ARM=B; QD=20; LR=1e-4; WI=$W_IMIT ;;
    2) ARM=C; QD=10; LR=3e-4; WI=$W_IMIT ;;
    3) ARM=D; QD=10; LR=3e-4; WI=$W_IMIT_D ;;
    *) echo "!! no variant for array index $ID"; exit 1 ;;
esac
WAVE_ARGS=(--qd-max "$QD" --lr "$LR" --w-qd-excess "$W_QD_EXCESS" --log-std-min "$LOG_STD_MIN" --w-vz "$W_VZ"
           --crouch-frac "$CROUCH_FRAC" --w-imit "$WI")
TAG="jump${ARM}_s${SEED}"
if [ "${DRY:-0}" = 1 ]; then echo "$TAG ${WAVE_ARGS[*]}"; exit 0; fi

. "$WORK/threepp-env.sh"

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
GIT=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
DIRTY=$(git -C "$SRC" status --porcelain 2>/dev/null | wc -l)
echo "== $TAG  arm=$ARM seed=$SEED envs=$ENVS iters=$ITERS  warm=$WARM (${WARM_SHA:0:8})  ${WAVE_ARGS[*]}  git=$GIT (+$DIRTY dirty)  node=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# ---- train ---------------------------------------------------------------------------------------
LIMIT_S=$(( (${SLURM_JOB_END_TIME:-0} > 0 ? SLURM_JOB_END_TIME - $(date +%s) : 90*60) - EVAL_RESERVE_MIN*60 ))
[ "$LIMIT_S" -gt 600 ] || LIMIT_S=600
T0=$(date +%s)
set +e
# shellcheck disable=SC2086
timeout -s TERM "${LIMIT_S}s" python -u train_spot_jump.py --envs "$ENVS" --iters "$ITERS" --seed "$SEED" \
    --arm "$ARM" --warmstart "$WARM" "${WAVE_ARGS[@]}" --out "$RUN/spot_jump.pt" --metrics "$RUN/metrics.jsonl" \
    --snapshot-every "$SNAPSHOT_EVERY" $EXTRA 2>&1 | tee "$RUN/train.log"
TRC=${PIPESTATUS[0]}
set -e
T1=$(date +%s)
TIMED_OUT=$([ "$TRC" -eq 124 ] && echo True || echo False)
CKPT="$RUN/spot_jump.pt"
[ -s "$CKPT" ] || CKPT="$RUN/spot_jump_latest.pt"
[ -s "$CKPT" ] || { echo "!! no checkpoint produced"; exit 1; }
STEPS_S=$(grep -oE '[0-9.]+k? ?steps/s' "$RUN/train.log" | tail -1 || true)

# ---- evaluate: one jump per env, one standing episode per env, one jump per env from a crouch ----------
rm -f "$RUN/eval.json"
set +e
python -u train_spot_jump.py --eval "$CKPT" --eval-envs "$EVAL_ENVS" --eval-json "$RUN/eval.json" 2>&1 | tee "$RUN/eval.log"
ERC=${PIPESTATUS[0]}
set -e
T2=$(date +%s)

python - "$RUN/manifest.json" "$RUN/eval.json" <<EOF
import json, os, sys
ev = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else {}
j = ev.get("jump", {})
nj = ev.get("nojump", {}).get("by_bucket", {}).get("nojump", {})
json.dump({
    "tag": "$TAG", "arm": "$ARM", "seed": $SEED, "warmstart": "$WARM", "warmstart_sha256": "$WARM_SHA",
    "wave_args": "${WAVE_ARGS[*]}", "qd_max": $QD, "lr": $LR, "w_qd_excess": $W_QD_EXCESS, "w_vz": $W_VZ,
    "crouch_frac": $CROUCH_FRAC, "w_imit": $WI,
    "envs": $ENVS, "iters": $ITERS, "extra": "$EXTRA", "runs": "$RUNS", "checkpoint": os.path.basename("$CKPT"),
    "train_exit": $TRC, "timed_out": $TIMED_OUT, "train_seconds": $((T1 - T0)), "throughput": "$STEPS_S",
    "eval_envs": $EVAL_ENVS, "eval_exit": $ERC, "eval_seconds": $((T2 - T1)),
    "eval_jump": j.get("jump"), "eval_jump_by_bucket": j.get("by_bucket"), "eval_qd_window": j.get("qd_window"),
    "eval_qd": j.get("qd"), "eval_cmd_qd_max_hx_hy_kn": j.get("cmd_qd_max_hx_hy_kn"),
    "eval_tau_p95_hx_hy_kn": j.get("tau_p95_hx_hy_kn"), "eval_nojump": nj,
    "eval_crouch": ev.get("crouch", {}).get("by_bucket", {}).get("crouch"),
    "git": "$GIT", "dirty_files": $DIRTY, "node": "$(hostname)",
    "slurm_job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}",
}, open(sys.argv[1], "w"), indent=1)
EOF
echo "== done $TAG: train $((T1 - T0)) s, eval $((T2 - T1)) s -> $RUN/eval.json"
[ "$TRC" -eq 0 ] || [ "$TRC" -eq 124 ] || exit "$TRC"
exit "$ERC"
