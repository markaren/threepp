#!/usr/bin/env bash
# Spot WALK + JUMP in one policy on IDUN (train_spot_steps.py --jump, spot_steps_env JUMP_*): four variants, one GPU each.
# Continue the shipped walking policy (python/examples/spot/spot_steps.pt, courseB_s2) on the course with jump triggers on
# the flat lanes, widened 96 -> 98 inputs with zero weights on the jump channels; then 1500 deterministic steps on the
# training env (jump + fall counters, <run>/spot_steps_jumpeval.json) and the flat steering eval against the base gait.
#
#   mkdir -p /cluster/work/$USER/logs                                                          # ONCE
#   sbatch --array=0-3 --export=ALL,ITERS=3000,RUNS_NAME=spot_unified1 scripts/idun/spot_unified.slurm
#   sbatch --array=0-1 --export=ALL,ITERS=20,RUNS_NAME=spot_unified1_check scripts/idun/spot_unified.slurm
#
# Why: the jump works as its own policy (spot_jump2: 99-100% jumped and landed) but Lars wants it part of the overall
# movement, not a separate policy handed control. Index -> variant, seed 0 each:
#   0 = A  the walking plant as shipped (no joint target limit)
#   1 = B  joint targets <= 10 rad/s (the recovery and jump plants' limit)
#   2 = C  B + the jump reference at weight 5 (spot_jump2's best standing-start variant)
#   3 = D  B + lr 3e-4 (spot_jump2's fastest learner)
# All: course wave 2's flags (courseB: --w-place 30), flat lanes raised to 40% of the course (the jump lanes, and more
# steering replay), self-collision on (the recovery plant), a jump trigger every U(2, 5) s after each 1.2 s window.
#
# Walltime: course wave 2 ran 40k steps/s on H100s (2000 iterations in ~3350 s); 3000 iterations = 197M steps = ~82 min,
# plus the jump bookkeeping. Training gets the walltime minus EVAL_RESERVE_MIN; the jump eval and steering keep that slot.
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
#SBATCH --time=02:00:00
#SBATCH --job-name=spotunify
#SBATCH --output=/cluster/work/laht/logs/spotunify-%A_%a.out
#SBATCH --error=/cluster/work/laht/logs/spotunify-%A_%a.out
set -euo pipefail

WORK=${WORK:-/cluster/work/$USER}
SRC=${SRC:-$WORK/threepp}
RUNS_NAME=${RUNS_NAME:-spot_unified1}
RUNS=${RUNS:-$WORK/$RUNS_NAME}
PARENT=${PARENT:-$SRC/python/examples/spot/spot_steps.pt}
ENVS=${ENVS:-2048}
ITERS=${ITERS:-3000}
STEER_ENVS=${STEER_ENVS:-512}
EVAL_JUMP_STEPS=${EVAL_JUMP_STEPS:-1500}
COURSE_FLAGS=${COURSE_FLAGS:---course --course-shares flat=0.40,stairs=0.25,hills=0.15,cross=0.10,rough=0.10 --shove-ramp 0.5,2.0,600 --foot-mu-dr 0.3,1.0,8 --stand-mode --cmd-switch 3,8,0.3 --reset-noise 0.08,0.06,0.3 --w-place 30}
JUMP_FLAGS=${JUMP_FLAGS:---jump 2,5 --self-collision}
EXTRA=${EXTRA:-}
EVAL_RESERVE_MIN=${EVAL_RESERVE_MIN:-15}
SEED=${SEED:-0}

ID=${SLURM_ARRAY_TASK_ID:-0}
case "$ID" in
    0) ARM=A; ARM_FLAGS="" ;;
    1) ARM=B; ARM_FLAGS="--qd-max 10" ;;
    2) ARM=C; ARM_FLAGS="--qd-max 10 --w-jump-imit 5" ;;
    3) ARM=D; ARM_FLAGS="--qd-max 10 --lr 3e-4" ;;
    *) echo "!! no variant for array index $ID"; exit 1 ;;
esac
TAG="unified${ARM}_s${SEED}"
if [ "${DRY:-0}" = 1 ]; then echo "$TAG $COURSE_FLAGS $JUMP_FLAGS $ARM_FLAGS $EXTRA"; exit 0; fi

. "$WORK/threepp-env.sh"

RUN="$RUNS/$TAG"
if [ -e "$RUN/manifest.json" ] && [ "${OVERWRITE:-0}" != 1 ]; then
    echo "!! $RUN already holds a finished run (manifest.json): pick another RUNS_NAME, or OVERWRITE=1"; exit 1
fi
mkdir -p "$RUN" "$WORK/logs"
LS=${LOCALSCRATCH:-/localscratch/$SLURM_JOB_ID}
[ -d "$LS" ] || LS=/localscratch/$USER/${SLURM_JOB_ID:-$$}
mkdir -p "$LS/warp" "$LS/tmp"
export WARP_CACHE_PATH="$LS/warp" TMPDIR="$LS/tmp"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
trap 'rm -rf "$LS" 2>/dev/null || true' EXIT

[ -s "$PARENT" ] || { echo "!! no parent checkpoint at $PARENT"; exit 1; }
cd "$SRC/python/examples/spot"
GIT=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
DIRTY=$(git -C "$SRC" status --porcelain 2>/dev/null | wc -l)
PARENT_SHA=$(sha256sum "$PARENT" | cut -d' ' -f1)
echo "== $TAG  seed=$SEED envs=$ENVS iters=$ITERS  flags='$COURSE_FLAGS $JUMP_FLAGS $ARM_FLAGS $EXTRA'  git=$GIT (+$DIRTY dirty)  node=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# ---- train (+ the jump eval on the training env) ------------------------------------------------------
LIMIT_S=$(( (${SLURM_JOB_END_TIME:-0} > 0 ? SLURM_JOB_END_TIME - $(date +%s) : 2*3600) - EVAL_RESERVE_MIN*60 ))
[ "$LIMIT_S" -gt 600 ] || LIMIT_S=600
T0=$(date +%s)
set +e
# shellcheck disable=SC2086
timeout -s TERM "${LIMIT_S}s" python -u train_spot_steps.py --envs "$ENVS" --iters "$ITERS" --seed "$SEED" \
    --warmstart "$PARENT" --out "$RUN/spot_steps.pt" --select final --snapshot-every 250 \
    --metrics "$RUN/metrics.jsonl" --eval-jump-steps "$EVAL_JUMP_STEPS" $COURSE_FLAGS $JUMP_FLAGS $ARM_FLAGS $EXTRA \
    2>&1 | tee "$RUN/train.log"
TRC=${PIPESTATUS[0]}
set -e
T1=$(date +%s)
TIMED_OUT=$([ "$TRC" -eq 124 ] && echo True || echo False)
CKPT="$RUN/spot_steps.pt"
[ -s "$CKPT" ] || CKPT="$RUN/spot_steps_latest.pt"
[ -s "$CKPT" ] || { echo "!! no checkpoint produced"; exit 1; }
STEPS_S=$(grep -oE '[0-9.]+k? ?steps/s' "$RUN/train.log" | tail -1 || true)
LAST_LOG=$(grep -E '\| track ' "$RUN/train.log" | tail -1 | tr -d '"' || true)

# ---- flat steering against the base gait (the checkpoint's plant, stand-mode and jump-channel contract) ------
rm -f "$RUN/steer.json"
set +e
python -u train_spot_steps.py --eval "$CKPT" --score-envs "$STEER_ENVS" --eval-json "$RUN/steer.json" 2>&1 | tee "$RUN/steer.log"
STEER_RC=${PIPESTATUS[0]}
set -e
T2=$(date +%s)

python - "$RUN/manifest.json" "$RUN/steer.json" "$RUN/spot_steps_jumpeval.json" <<EOF
import json, sys, os
steer = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else {}
je = json.load(open(sys.argv[3])) if os.path.exists(sys.argv[3]) else {}
json.dump({
    "tag": "$TAG", "arm": "$ARM", "seed": $SEED, "parent": "$PARENT", "parent_sha256": "$PARENT_SHA",
    "trainer_flags": "$COURSE_FLAGS $JUMP_FLAGS $ARM_FLAGS $EXTRA".strip(), "envs": $ENVS, "iters": $ITERS, "runs": "$RUNS",
    "checkpoint": os.path.basename("$CKPT"), "train_exit": $TRC, "timed_out": $TIMED_OUT, "train_seconds": $((T1 - T0)),
    "throughput": "$STEPS_S", "last_train_log": """$LAST_LOG""",
    "jump_eval": je.get("jump"), "jump_eval_fell_per_step": je.get("fell_per_step"), "jump_eval_flat_track": je.get("flat_track"),
    "steer_envs": $STEER_ENVS, "steer_seconds": $((T2 - T1)), "steer_exit": $STEER_RC,
    "steer_worst_ratio": steer.get("worst"), "steer_pass": steer.get("pass"),
    "steer_worst_moving": max((c["ratio"] for c in steer.get("per_cmd", []) if any(c["cmd"])), default=None),
    "git": "$GIT", "dirty_files": $DIRTY, "node": "$(hostname)",
    "slurm_job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}",
}, open(sys.argv[1], "w"), indent=1)
EOF
echo "== done $TAG: train $((T1 - T0)) s, steer $((T2 - T1)) s"
[ "$TRC" -eq 0 ] || [ "$TRC" -eq 124 ] || exit "$TRC"
exit "$STEER_RC"
