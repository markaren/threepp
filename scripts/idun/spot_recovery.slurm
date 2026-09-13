#!/usr/bin/env bash
# Spot FALL RECOVERY pilot on IDUN (python/examples/spot/spot_recovery_env.py, train_spot_recovery.py): arm x seed, one
# H100 each. Train, then evaluate the final weights from 2048 fallen starts (40% back, 40% side, 20% any orientation) to
# eval.json, and write a manifest.
#
#   mkdir -p /cluster/work/$USER/logs                                                          # ONCE
#   sbatch --array=0-3 --export=ALL,ITERS=1000,RUNS_NAME=spot_recovery1 scripts/idun/spot_recovery.slurm   # A s0, B s0, A s1, B s1
#   sbatch --array=0-1 --export=ALL,ITERS=20,RUNS_NAME=spot_recovery_check scripts/idun/spot_recovery.slurm
#
# Index -> (arm, seed): arm = ID % 2, seed = ID / 2.
#   A  continue the shipped walking policy (python/examples/spot/spot_steps.pt, 96-d) whole, log-std raised to >= -1.0
#   B  a fresh network
# Both: the walking plant (stiff gains, real torque limits 45/45/115 N·m) with self-collision, flat ground, 6 s episodes,
# a fall curriculum L0 (tilted <= 25 deg) -> L3 (back / side / anyhow), 18% standing starts.
# A task refuses a run directory that already holds a manifest.json (OVERWRITE=1 to allow).
#
#SBATCH --account=share-ie-iir
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
# Any of the three big GPUs, 6 cores, 32 GB: the loop uses ~3 cores, and an h100-only 12-core request left 3 of 4
# array tasks pending on (Resources) behind one running task on 2026-09-13 (free GPUs sit on CPU-starved nodes).
#SBATCH --constraint=h100|a100|h200
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
RUNS_NAME=${RUNS_NAME:-spot_recovery}
RUNS=${RUNS:-$WORK/$RUNS_NAME}
WARM=${WARM:-$SRC/python/examples/spot/spot_steps.pt}
ENVS=${ENVS:-2048}
ITERS=${ITERS:-1000}
EVAL_ENVS=${EVAL_ENVS:-2048}
SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-250}
EXTRA=${EXTRA:-}
EVAL_RESERVE_MIN=${EVAL_RESERVE_MIN:-10}

. "$WORK/threepp-env.sh"

ID=${SLURM_ARRAY_TASK_ID:-0}
SEED=$(( ID / 2 ))
if [ $(( ID % 2 )) -eq 0 ]; then ARM=A; else ARM=B; fi
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
if [ "$ARM" = A ]; then
    [ -s "$WARM" ] || { echo "!! no warm-start checkpoint at $WARM"; exit 1; }
    WARM_ARGS=(--warmstart "$WARM"); WARM_SHA=$(sha256sum "$WARM" | cut -d' ' -f1)
else
    WARM_ARGS=(--warmstart ""); WARM_SHA=""
fi
GIT=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
DIRTY=$(git -C "$SRC" status --porcelain 2>/dev/null | wc -l)
echo "== $TAG  arm=$ARM seed=$SEED envs=$ENVS iters=$ITERS  git=$GIT (+$DIRTY dirty)  node=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# ---- train ---------------------------------------------------------------------------------------
LIMIT_S=$(( (${SLURM_JOB_END_TIME:-0} > 0 ? SLURM_JOB_END_TIME - $(date +%s) : 2*3600) - EVAL_RESERVE_MIN*60 ))
[ "$LIMIT_S" -gt 600 ] || LIMIT_S=600
T0=$(date +%s)
set +e
# shellcheck disable=SC2086
timeout -s TERM "${LIMIT_S}s" python -u train_spot_recovery.py --envs "$ENVS" --iters "$ITERS" --seed "$SEED" \
    --arm "$ARM" "${WARM_ARGS[@]}" --out "$RUN/spot_recovery.pt" --metrics "$RUN/metrics.jsonl" \
    --snapshot-every "$SNAPSHOT_EVERY" $EXTRA 2>&1 | tee "$RUN/train.log"
TRC=${PIPESTATUS[0]}
set -e
T1=$(date +%s)
TIMED_OUT=$([ "$TRC" -eq 124 ] && echo True || echo False)
CKPT="$RUN/spot_recovery.pt"
[ -s "$CKPT" ] || CKPT="$RUN/spot_recovery_latest.pt"       # a timed-out run saved _latest every 20 iterations
[ -s "$CKPT" ] || { echo "!! no checkpoint produced"; exit 1; }
STEPS_S=$(grep -oE '[0-9.]+k? ?steps/s' "$RUN/train.log" | tail -1 || true)

# ---- evaluate from fallen starts ---------------------------------------------------------------------
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
json.dump({
    "tag": "$TAG", "arm": "$ARM", "seed": $SEED, "warmstart": "${WARM_ARGS[1]}", "warmstart_sha256": "$WARM_SHA",
    "envs": $ENVS, "iters": $ITERS, "extra": "$EXTRA", "runs": "$RUNS", "checkpoint": os.path.basename("$CKPT"),
    "train_exit": $TRC, "timed_out": $TIMED_OUT, "train_seconds": $((T1 - T0)), "throughput": "$STEPS_S",
    "eval_envs": $EVAL_ENVS, "eval_exit": $ERC, "eval_seconds": $((T2 - T1)),
    "eval_success_3s": fa.get("success_3s"), "eval_success_6s": fa.get("success_6s"),
    "eval_recovery_s": fa.get("recovery_s"), "eval_blowups": fa.get("blowups"),
    "eval_tau_p95_hx_hy_kn": ev.get("tau_p95_hx_hy_kn"), "eval_contact_ticks_per_ep": fa.get("contact_ticks_per_ep"),
    "git": "$GIT", "dirty_files": $DIRTY, "node": "$(hostname)",
    "slurm_job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}",
}, open(sys.argv[1], "w"), indent=1)
EOF
echo "== done $TAG: train $((T1 - T0)) s, eval $((T2 - T1)) s -> $RUN/eval.json"
[ "$TRC" -eq 0 ] || [ "$TRC" -eq 124 ] || exit "$TRC"
exit "$ERC"
