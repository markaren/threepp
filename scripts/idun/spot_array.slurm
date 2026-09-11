#!/usr/bin/env bash
# Spot perception-robustness seed array on IDUN: 5 training conditions x N seeds, each job one
# H100, PPO on the stairs curriculum, then the 6-cell scoring matrix. No rendering anywhere.
#
#   mkdir -p /cluster/work/$USER/logs /cluster/work/$USER/spot_runs        # ONCE
#   sbatch --array=0-24 scripts/idun/spot_array.slurm                        # 5 conditions x 5 seeds
#   sbatch --array=0-24%20 ...                                               # cap at the QOS's 20 GPUs
#   sbatch --array=0-4 --export=ALL,ITERS=100 scripts/idun/spot_array.slurm  # 5-job pilot, ~10 min
#
# Index -> (condition, seed):  condition = ID % 5, seed = ID / 5.  The conditions are the single
# list in python/examples/spot/score_matrix.py (CONDITIONS): analytic, raycast, perceived at 0 /
# 0.02 / 0.05 m map error. Every run warm-starts from the same scratch_flat_best.pt.
#
# Afterwards:
#   python python/examples/spot/score_matrix.py --aggregate /cluster/work/$USER/spot_runs --md table.md
#
# Why an array and not one big run: a single H100 only buys ~1.4x the 4070 on this loop (PhysX
# host-sync bound, see the step profile), while job arrays start immediately on the QOS.
#
# Prerequisites (from build_container.sh): $WORK/threepp-env.sh, the in-tree threepp_py with GPU
# PhysX, torch cu126 in --user site, and scratch_flat_best.pt copied to
# $SRC/python/examples/spot/scratch_distillation/ (git-ignored, comes from the laptop).
#
#SBATCH --account=share-ie-iir
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --job-name=spotarr
#SBATCH --output=/cluster/work/laht/logs/spotarr-%A_%a.out
#SBATCH --error=/cluster/work/laht/logs/spotarr-%A_%a.out
set -euo pipefail

WORK=${WORK:-/cluster/work/$USER}
SRC=${SRC:-$WORK/threepp}
RUNS=${RUNS:-$WORK/spot_runs}
ENVS=${ENVS:-2048}
ITERS=${ITERS:-3000}
SCORE_ENVS=${SCORE_ENVS:-512}
EXTRA=${EXTRA:-}                       # e.g. EXTRA="--push-vel 3.0"
# Training gets the walltime minus this; scoring always keeps its slot. The trainer saves the
# best and the latest checkpoint every 20 iterations, so a timed-out run still scores.
SCORE_RESERVE_MIN=${SCORE_RESERVE_MIN:-25}

# Runtime FIRST: threepp-env.sh loads the Python module that owns `python` below; the container
# is a build-time tool only and the module imports on the host Python.
. "$WORK/threepp-env.sh"

ID=${SLURM_ARRAY_TASK_ID:-0}
NCOND=$(python "$SRC/python/examples/spot/score_matrix.py" --n-conditions)
COND=$(( ID % NCOND ))
SEED=$(( ID / NCOND ))
IFS=$'\t' read -r CNAME CFLAGS < <(python "$SRC/python/examples/spot/score_matrix.py" --condition "$COND")
TAG="${CNAME}_s${SEED}"
RUN="$RUNS/$TAG"
mkdir -p "$RUN" "$WORK/logs"
# Warp's kernel cache under a shared home is a contention hazard across an array -> node-local.
LS=${LOCALSCRATCH:-/localscratch/$SLURM_JOB_ID}
[ -d "$LS" ] || LS=/localscratch/$USER/${SLURM_JOB_ID:-$$}
mkdir -p "$LS/warp" "$LS/tmp"
export WARP_CACHE_PATH="$LS/warp" TMPDIR="$LS/tmp"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
trap 'rm -rf "$LS" 2>/dev/null || true' EXIT

cd "$SRC/python/examples/spot"
GIT=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
DIRTY=$(git -C "$SRC" status --porcelain 2>/dev/null | wc -l)
echo "== $TAG  condition=$CNAME flags='$CFLAGS' seed=$SEED  envs=$ENVS iters=$ITERS  git=$GIT (+$DIRTY dirty)  node=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# ---- train ---------------------------------------------------------------------------------------
LIMIT_S=$(( (${SLURM_JOB_END_TIME:-0} > 0 ? SLURM_JOB_END_TIME - $(date +%s) : 3*3600) - SCORE_RESERVE_MIN*60 ))
[ "$LIMIT_S" -gt 600 ] || LIMIT_S=600
T0=$(date +%s)
set +e
# shellcheck disable=SC2086
# NO --graph: pilot 2026-09-10 on the H100 failed verify_graph for every Warp-backed condition
# (raycast 1.0 away, perceived+noise 0.10/0.26 away — the Warp launches are not on torch's
# capture stream, so replay reads frozen heights), and the noise-free perceived pass is weak
# (the check stands still on flat ground). Eager is the verified path for all five; it costs
# ~9 ms/step at K=2048, i.e. ~50k steps/s on the H100 against 65-70k graphed. EXTRA="--graph"
# puts it back once threepp.rl launches Warp on the torch stream.
timeout -s TERM "${LIMIT_S}s" python -u train_spot_steps.py --envs "$ENVS" --iters "$ITERS" \
    --seed "$SEED" --out "$RUN/spot_steps.pt" $CFLAGS $EXTRA 2>&1 | tee "$RUN/train.log"
TRC=${PIPESTATUS[0]}
set -e
T1=$(date +%s)
TIMED_OUT=$([ "$TRC" -eq 124 ] && echo True || echo False)   # a Python literal: it lands in the manifest
echo "== train exit $TRC after $((T1 - T0)) s (timed_out=$TIMED_OUT)"

# The best checkpoint is only written once the steering gate passes; fall back to the latest.
CKPT="$RUN/spot_steps.pt"
[ -s "$CKPT" ] || CKPT="$RUN/spot_steps_latest.pt"
if [ ! -s "$CKPT" ]; then
    echo "!! no checkpoint produced, nothing to score"; exit 1
fi
STEPS_S=$(grep -oE '[0-9.]+k? ?steps/s' "$RUN/train.log" | tail -1 || true)
LAST_LOG=$(grep -E '\| track ' "$RUN/train.log" | tail -1 || true)

# ---- score ---------------------------------------------------------------------------------------
rm -f "$RUN/scores.jsonl"
set +e
python -u score_matrix.py "$CKPT" --json "$RUN/scores.jsonl" --envs "$SCORE_ENVS" --seed 0 2>&1 | tee "$RUN/score.log"
SRC_RC=${PIPESTATUS[0]}
set -e
T2=$(date +%s)

# ---- manifest ------------------------------------------------------------------------------------
python - "$RUN/manifest.json" <<EOF
import json, sys, os
json.dump({
    "tag": "$TAG", "condition": "$CNAME", "trainer_flags": "$CFLAGS $EXTRA".strip(), "seed": $SEED,
    "envs": $ENVS, "iters": $ITERS, "score_envs": $SCORE_ENVS,
    "checkpoint": os.path.basename("$CKPT"), "train_exit": $TRC, "timed_out": $TIMED_OUT,
    "train_seconds": $((T1 - T0)), "score_seconds": $((T2 - T1)), "score_exit": $SRC_RC,
    "throughput": "$STEPS_S", "last_train_log": """$LAST_LOG""",
    "git": "$GIT", "dirty_files": $DIRTY, "node": "$(hostname)",
    "slurm_job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}",
}, open(sys.argv[1], "w"), indent=1)
EOF
echo "== done $TAG: train $((T1 - T0)) s, score $((T2 - T1)) s, scores in $RUN/scores.jsonl"
exit "$SRC_RC"
