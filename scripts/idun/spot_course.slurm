#!/usr/bin/env bash
# Spot mixed-terrain COURSE pilot on IDUN (python/examples/spot/spot_course.py): 2 arms x 2 seeds, each job one H100.
# Continue the raycast parent on the course for ITERS iterations (eager raycast, K=2048), then score the final weights
# with score_e0.py --suite showpiece in the same task (the showpiece world, then the course cells' own world).
#
#   mkdir -p /cluster/work/$USER/logs /cluster/work/$USER/spot_course          # ONCE
#   sbatch --array=0-3 scripts/idun/spot_course.slurm                            # A s0, B s0, A s1, B s1
#   sbatch --array=0-1 --export=ALL,ITERS=20 scripts/idun/spot_course.slurm      # a 20-iteration check of both arms
#
# Index -> (arm, seed): arm = ID % 2, seed = ID / 2.
#   A  the course with every training ingredient: --course (honest termination, real torque limits, IK spawns held),
#      interval shoves on the non-flat lanes with push_max ramped 0.5 -> 2.0 m/s over iterations 0-600 (so a 300-
#      iteration pilot ends at 1.25), foot friction buckets U(0.3, 1.0) 'min' on an explicit 1.0 'min' terrain, stand mode
#   B  A + the foot-placement reward on stair treads (--w-place W_PLACE)
#
# Afterwards (CPU):
#   python python/examples/spot/score_e0.py --aggregate /cluster/work/$USER/spot_course
#
# Prerequisites: those of spot_array.slurm ($WORK/threepp-env.sh, the in-tree threepp_py with GPU PhysX, torch, and
# scratch_flat_best.pt, which the imitation anchor loads), plus the parent checkpoint at $PARENT (idun_runs/ is
# git-ignored: the seed array wrote it to $WORK/spot_runs/raycast_s4/).
#
#SBATCH --account=share-ie-iir
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=spotcourse
#SBATCH --output=/cluster/work/laht/logs/spotcourse-%A_%a.out
#SBATCH --error=/cluster/work/laht/logs/spotcourse-%A_%a.out
set -euo pipefail

WORK=${WORK:-/cluster/work/$USER}
SRC=${SRC:-$WORK/threepp}
RUNS=${RUNS:-$WORK/spot_course}
PARENT=${PARENT:-$WORK/spot_runs/raycast_s4/spot_steps_latest.pt}
ENVS=${ENVS:-2048}
ITERS=${ITERS:-300}
SCORE_ENVS=${SCORE_ENVS:-2048}
SCORE_SEED=${SCORE_SEED:-1}
# 10: at the raycast_s4 parent's placement the term is 6.3% of |reward| per stair-lane step (course smoke, K=2048,
# 2026-09-13: mean |placement| 0.0264 against |reward| 4.20), 8% if every tread touchdown were perfect
W_PLACE=${W_PLACE:-10}
COURSE_FLAGS=${COURSE_FLAGS:---course --shove-ramp 0.5,2.0,600 --foot-mu-dr 0.3,1.0,8 --stand-mode}
EXTRA=${EXTRA:-}
# Training gets the walltime minus this; scoring always keeps its slot (showpiece + course worlds at K=2048).
SCORE_RESERVE_MIN=${SCORE_RESERVE_MIN:-30}

. "$WORK/threepp-env.sh"

ID=${SLURM_ARRAY_TASK_ID:-0}
SEED=$(( ID / 2 ))
if [ $(( ID % 2 )) -eq 0 ]; then ARM=A; ARM_FLAGS=""; else ARM=B; ARM_FLAGS="--w-place $W_PLACE"; fi
TAG="course${ARM}_s${SEED}"
RUN="$RUNS/$TAG"
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
echo "== $TAG  arm=$ARM seed=$SEED envs=$ENVS iters=$ITERS  flags='$COURSE_FLAGS $ARM_FLAGS $EXTRA'  git=$GIT (+$DIRTY dirty)  node=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# ---- train ---------------------------------------------------------------------------------------
LIMIT_S=$(( (${SLURM_JOB_END_TIME:-0} > 0 ? SLURM_JOB_END_TIME - $(date +%s) : 2*3600) - SCORE_RESERVE_MIN*60 ))
[ "$LIMIT_S" -gt 600 ] || LIMIT_S=600
T0=$(date +%s)
set +e
# NO --graph: the course needs the eager raycast (Warp launches do not replay from a CUDA graph, spot_array.slurm).
# --select final: the pilot compares final weights; --out then holds them (= _latest).
# shellcheck disable=SC2086
timeout -s TERM "${LIMIT_S}s" python -u train_spot_steps.py --envs "$ENVS" --iters "$ITERS" --seed "$SEED" \
    --warmstart "$PARENT" --out "$RUN/spot_steps.pt" --select final --snapshot-every 100 \
    --metrics "$RUN/metrics.jsonl" $COURSE_FLAGS $ARM_FLAGS $EXTRA 2>&1 | tee "$RUN/train.log"
TRC=${PIPESTATUS[0]}
set -e
T1=$(date +%s)
TIMED_OUT=$([ "$TRC" -eq 124 ] && echo True || echo False)
echo "== train exit $TRC after $((T1 - T0)) s (timed_out=$TIMED_OUT)"

CKPT="$RUN/spot_steps.pt"
[ -s "$CKPT" ] || CKPT="$RUN/spot_steps_latest.pt"          # a timed-out run saved _latest every 20 iterations
if [ ! -s "$CKPT" ]; then
    echo "!! no checkpoint produced, nothing to score"; exit 1
fi
STEPS_S=$(grep -oE '[0-9.]+k? ?steps/s' "$RUN/train.log" | tail -1 || true)
LAST_LOG=$(grep -E '\| track ' "$RUN/train.log" | tail -1 | tr -d '"' || true)

# ---- score ---------------------------------------------------------------------------------------
# label <cond>_s<N>:latest, the shape score_e0's aggregate reads as an array arm (the final weights are _latest)
rm -f "$RUN/showpiece_s${SCORE_SEED}.jsonl" "$RUN/showpiece_s${SCORE_SEED}_course.jsonl"
set +e
python -u score_e0.py "$CKPT" --label "${TAG}:latest" --suite showpiece --seed "$SCORE_SEED" --envs "$SCORE_ENVS" \
    --json "$RUN/showpiece_s${SCORE_SEED}.jsonl" 2>&1 | tee "$RUN/score.log"
SRC_RC=${PIPESTATUS[0]}
set -e
T2=$(date +%s)

# ---- manifest ------------------------------------------------------------------------------------
python - "$RUN/manifest.json" <<EOF
import json, sys, os
json.dump({
    "tag": "$TAG", "arm": "$ARM", "seed": $SEED, "parent": "$PARENT", "parent_sha256": "$PARENT_SHA",
    "trainer_flags": "$COURSE_FLAGS $ARM_FLAGS $EXTRA".strip(), "envs": $ENVS, "iters": $ITERS,
    "score_envs": $SCORE_ENVS, "score_seed": $SCORE_SEED, "score_suite": "showpiece (+ course world)",
    "checkpoint": os.path.basename("$CKPT"), "train_exit": $TRC, "timed_out": $TIMED_OUT,
    "train_seconds": $((T1 - T0)), "score_seconds": $((T2 - T1)), "score_exit": $SRC_RC,
    "throughput": "$STEPS_S", "last_train_log": """$LAST_LOG""",
    "git": "$GIT", "dirty_files": $DIRTY, "node": "$(hostname)",
    "slurm_job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}",
}, open(sys.argv[1], "w"), indent=1)
EOF
echo "== done $TAG: train $((T1 - T0)) s, score $((T2 - T1)) s, scores in $RUN/showpiece_s${SCORE_SEED}*.jsonl"
exit "$SRC_RC"
