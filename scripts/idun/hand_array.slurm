#!/usr/bin/env bash
# Tendon-hand palm-up hold seed array on IDUN: N seeds, each job one H100, PPO on the
# GPU-vectorized routed-cable hand. No rendering anywhere.
#
#   mkdir -p /cluster/work/$USER/logs /cluster/work/$USER/hand_runs            # ONCE
#   sbatch --array=0-7 scripts/idun/hand_array.slurm                           # 8 seeds
#   sbatch --array=0-7%20 ...                                                  # cap at the QOS's 20 GPUs
#   sbatch --array=0-1 --export=ALL,ITERS=100 scripts/idun/hand_array.slurm    # 2-job pilot, ~10 min
#
# Index -> seed, one to one. Every run trains from scratch: there is no warm start for this
# task, because the cable law makes the plant different from anything already on disk.
#
# Afterwards, the numbers are in the per-run manifest and train.log:
#   grep -H 'hold ' /cluster/work/$USER/hand_runs/s*/train.log | tail -n 40
#   python -c "import json,glob;[print(json.load(open(f))['tag'], json.load(open(f))['last_train_log']) for f in sorted(glob.glob('/cluster/work/'+__import__('os').environ['USER']+'/hand_runs/s*/manifest.json'))]"
#
# Why an array and not one big run: the same reason as spot_array.slurm -- a single H100 only
# buys a modest multiple of the 4070 on this loop (it is PhysX host-sync bound, and the cable
# evaluation adds four more launches per control step), while job arrays start immediately.
#
# NO --graph, and it is not an option here. The per-step torch region calls into PhysX four
# times through the cable evaluation, and a CUDA graph captures torch kernels only: replay
# would re-run the cable math on frozen link poses. train_tendon_hand.py builds the env eager.
#
# Prerequisites (from build_container.sh): $WORK/threepp-env.sh, the in-tree threepp_py with
# GPU PhysX, torch cu126 in --user site.
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
#SBATCH --job-name=handarr
#SBATCH --output=/cluster/work/laht/logs/handarr-%A_%a.out
#SBATCH --error=/cluster/work/laht/logs/handarr-%A_%a.out
set -euo pipefail

WORK=${WORK:-/cluster/work/$USER}
SRC=${SRC:-$WORK/threepp}
RUNS=${RUNS:-$WORK/hand_runs}
ENVS=${ENVS:-4096}
ITERS=${ITERS:-3000}
EXTRA=${EXTRA:-}                       # e.g. EXTRA="--horizon 48"
# Training gets the walltime minus this. The trainer's own --max-minutes stops it cleanly and
# saves, so a job that runs long still leaves a checkpoint rather than being killed mid-update.
RESERVE_MIN=${RESERVE_MIN:-5}

# Runtime FIRST: threepp-env.sh loads the Python module that owns `python` below; the container
# is a build-time tool only and the module imports on the host Python.
. "$WORK/threepp-env.sh"

SEED=${SLURM_ARRAY_TASK_ID:-0}
TAG="s${SEED}"
RUN="$RUNS/$TAG"
mkdir -p "$RUN" "$WORK/logs"
LS=${LOCALSCRATCH:-/localscratch/$SLURM_JOB_ID}
[ -d "$LS" ] || LS=/localscratch/$USER/${SLURM_JOB_ID:-$$}
mkdir -p "$LS/tmp"
export TMPDIR="$LS/tmp"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
trap 'rm -rf "$LS" 2>/dev/null || true' EXIT

cd "$SRC/python/examples"
GIT=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
DIRTY=$(git -C "$SRC" status --porcelain 2>/dev/null | wc -l)
echo "== $TAG  seed=$SEED  envs=$ENVS iters=$ITERS  git=$GIT (+$DIRTY dirty)  node=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

# ---- train ---------------------------------------------------------------------------------------
LIMIT_MIN=$(( ( (${SLURM_JOB_END_TIME:-0} > 0 ? SLURM_JOB_END_TIME - $(date +%s) : 3*3600) / 60 ) - RESERVE_MIN ))
[ "$LIMIT_MIN" -gt 10 ] || LIMIT_MIN=10
T0=$(date +%s)
set +e
# shellcheck disable=SC2086
python -u train_tendon_hand.py --envs "$ENVS" --iters "$ITERS" --seed "$SEED" \
    --max-minutes "$LIMIT_MIN" --out "$RUN/tendon_hand_hold.pt" $EXTRA 2>&1 | tee "$RUN/train.log"
TRC=${PIPESTATUS[0]}
set -e
T1=$(date +%s)

CKPT="$RUN/tendon_hand_hold.pt"
if [ ! -s "$CKPT" ]; then
    echo "!! no checkpoint produced"; exit 1
fi
STEPS_S=$(grep -oE '[0-9,.]+k? ?steps/s' "$RUN/train.log" | tail -1 || true)
LAST_LOG=$(grep -E 'hold ' "$RUN/train.log" | tail -1 || true)

# ---- manifest ------------------------------------------------------------------------------------
python - "$RUN/manifest.json" <<EOF
import json, sys, os
json.dump({
    "tag": "$TAG", "seed": $SEED, "envs": $ENVS, "iters": $ITERS,
    "trainer_flags": "$EXTRA".strip(), "limit_minutes": $LIMIT_MIN,
    "checkpoint": os.path.basename("$CKPT"), "train_exit": $TRC,
    "train_seconds": $((T1 - T0)), "throughput": "$STEPS_S",
    "last_train_log": """$LAST_LOG""",
    "git": "$GIT", "dirty_files": $DIRTY, "node": "$(hostname)",
    "slurm_job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}",
}, open(sys.argv[1], "w"), indent=1)
EOF
echo "== done $TAG: train $((T1 - T0)) s, checkpoint $CKPT"
exit "$TRC"
