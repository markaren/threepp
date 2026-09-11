#!/usr/bin/env bash
# Re-score every finished run of the Spot seed array on the HARD cell set (tallest risers from the
# start + 3 m/s shoves, see score_matrix.py HARD_CELLS). Same index -> (condition, seed) mapping as
# spot_array.slurm, so the two arrays line up; ~3 min per task on one GPU.
#
#   sbatch --array=0-24 scripts/idun/spot_rescore.slurm
#   sbatch --array=0-24 --export=ALL,CELLS=all scripts/idun/spot_rescore.slurm   # base + hard
#
# Afterwards:
#   python python/examples/spot/score_matrix.py --aggregate /cluster/work/$USER/spot_runs --cells hard
#
#SBATCH --account=share-ie-iir
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --job-name=spotscore
#SBATCH --output=/cluster/work/laht/logs/spotscore-%A_%a.out
#SBATCH --error=/cluster/work/laht/logs/spotscore-%A_%a.out
set -euo pipefail

WORK=${WORK:-/cluster/work/$USER}
SRC=${SRC:-$WORK/threepp}
RUNS=${RUNS:-$WORK/spot_runs}
CELLS=${CELLS:-hard}
SCORE_ENVS=${SCORE_ENVS:-512}

. "$WORK/threepp-env.sh"

ID=${SLURM_ARRAY_TASK_ID:-0}
NCOND=$(python "$SRC/python/examples/spot/score_matrix.py" --n-conditions)
IFS=$'\t' read -r CNAME _ < <(python "$SRC/python/examples/spot/score_matrix.py" --condition "$(( ID % NCOND ))")
TAG="${CNAME}_s$(( ID / NCOND ))"
RUN="$RUNS/$TAG"

LS=${LOCALSCRATCH:-/localscratch/$SLURM_JOB_ID}
[ -d "$LS" ] || LS=/localscratch/$USER/${SLURM_JOB_ID:-$$}
mkdir -p "$LS/warp" "$LS/tmp"
export WARP_CACHE_PATH="$LS/warp" TMPDIR="$LS/tmp"
trap 'rm -rf "$LS" 2>/dev/null || true' EXIT

CKPT="$RUN/spot_steps.pt"
[ -s "$CKPT" ] || CKPT="$RUN/spot_steps_latest.pt"
if [ ! -s "$CKPT" ]; then
    echo "!! $TAG: no checkpoint under $RUN (run not finished?)"; exit 1
fi
[ -f "$RUN/manifest.json" ] || echo "!! $TAG: no manifest.json yet - the training job is still running; scoring its latest checkpoint anyway"

cd "$SRC/python/examples/spot"
echo "== $TAG  cells=$CELLS  ckpt=$(basename "$CKPT")  node=$(hostname)"
JS="$RUN/scores_${CELLS}.jsonl"; [ "$CELLS" = base ] && JS="$RUN/scores.jsonl"
rm -f "$JS"
python -u score_matrix.py "$CKPT" --cells "$CELLS" --json "$JS" --envs "$SCORE_ENVS" --seed 0 2>&1 | tee "$RUN/score_${CELLS}.log"
exit "${PIPESTATUS[0]}"
