#!/usr/bin/env bash
# Spot mixed-terrain COURSE pilot on IDUN (python/examples/spot/spot_course.py): 2 arms x 2 seeds, each job one H100.
# Continue the raycast parent on the course for ITERS iterations (eager raycast, K=2048), then score the final weights
# with score_e0.py --suite showpiece in the same task (the showpiece world, then the course cells' own world).
#
#   mkdir -p /cluster/work/$USER/logs                                                        # ONCE
#   sbatch --array=0-5 --export=ALL,ITERS=2000,RUNS_NAME=spot_course2 scripts/idun/spot_course.slurm   # wave 2
#   sbatch --array=0-1 --export=ALL,ITERS=20,RUNS_NAME=spot_course2_check scripts/idun/spot_course.slurm  # a quick check
# The 2026-09-13 pilot was --array=0-3 with ITERS=300, W_PLACE=10 and COURSE_FLAGS without the wave-2 flags, into
# $WORK/spot_course. A task refuses to write into a run directory that already has a manifest.json (OVERWRITE=1 to
# allow), so a new wave cannot land on an old one's results by accident.
#
# Index -> (arm, seed): arm = ID % 2, seed = ID / 2.
#   A  the course with every training ingredient: --course (honest termination, real torque limits, IK spawns held, the
#      'hard' ladder), interval shoves on the non-flat lanes with push_max ramped 0.5 -> 2.0 m/s over iterations 0-600,
#      foot friction buckets U(0.3, 1.0) 'min' on an explicit 1.0 'min' terrain, stand mode, command switches every
#      U(3, 8) s with abrupt stand<->go at p 0.3, and messy resets (joints N(0, 0.08), drop U(0, 0.06) m, base velocity
#      U(+-0.3) m/s, 30% of flat and stair resets unheld)
#   B  A + the foot-placement reward on stair treads (--w-place W_PLACE)
#
# Walltime: training gets --time minus SCORE_RESERVE_MIN. 2000 iterations at K=2048 x horizon 32 are 131 M steps: 51 min
# at the pilot's 43k steps/s. The wave-2 options cost ~23% on the 4070 (course smoke, K=2048, 2026-09-13: 28.2k -> 21.8k
# env steps/s), i.e. ~33k and ~66 min on the H100, and the 85 min the trainer is given still fits them at 26k. Scoring
# (the showpiece world and the course world at K=2048) and the steering eval (K=512) share the 35 min reserve.
# The steering eval runs a stand-mode checkpoint with the (0,0) clock at its zero command and the teacher without it, so
# the (0,0,0) row compares a stand with the teacher's trot in place: the manifest also carries the worst ratio over the
# moving commands alone (steer_worst_moving).
#
# Afterwards (CPU):
#   python python/examples/spot/score_e0.py --aggregate /cluster/work/$USER/spot_course2
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
RUNS_NAME=${RUNS_NAME:-spot_course}    # the wave's directory under $WORK (RUNS overrides the whole path)
RUNS=${RUNS:-$WORK/$RUNS_NAME}
PARENT=${PARENT:-$WORK/spot_runs/raycast_s4/spot_steps_latest.pt}
ENVS=${ENVS:-2048}
ITERS=${ITERS:-300}
SCORE_ENVS=${SCORE_ENVS:-2048}
SCORE_SEED=${SCORE_SEED:-1}
STEER_ENVS=${STEER_ENVS:-512}
# 30 (wave 2): the pilot's 10 barely moved placement. At the raycast_s4 parent, stair lanes, K=2048 (2026-09-13): 6.3% of
# |reward| per step at 10 (mean |placement| 0.0264 against |reward| 4.20), 17% at 30 on the hard ladder with command
# switches and messy resets (|30 * placement| 0.84 against |reward| 4.94).
W_PLACE=${W_PLACE:-30}
COURSE_FLAGS=${COURSE_FLAGS:---course --shove-ramp 0.5,2.0,600 --foot-mu-dr 0.3,1.0,8 --stand-mode --cmd-switch 3,8,0.3 --reset-noise 0.08,0.06,0.3}
EXTRA=${EXTRA:-}
# Training gets the walltime minus this; scoring (showpiece + course worlds at K=2048) and steering keep their slot.
SCORE_RESERVE_MIN=${SCORE_RESERVE_MIN:-35}

. "$WORK/threepp-env.sh"

ID=${SLURM_ARRAY_TASK_ID:-0}
SEED=$(( ID / 2 ))
if [ $(( ID % 2 )) -eq 0 ]; then ARM=A; ARM_FLAGS=""; else ARM=B; ARM_FLAGS="--w-place $W_PLACE"; fi
TAG="course${ARM}_s${SEED}"
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

# ---- flat steering against the base gait (the checkpoint's own plant and stand-mode contract, _common.py) --------------
rm -f "$RUN/steer.json"
set +e
python -u train_spot_steps.py --eval "$CKPT" --score-envs "$STEER_ENVS" --eval-json "$RUN/steer.json" 2>&1 | tee "$RUN/steer.log"
STEER_RC=${PIPESTATUS[0]}
set -e
T3=$(date +%s)

# ---- manifest ------------------------------------------------------------------------------------
python - "$RUN/manifest.json" "$RUN/steer.json" <<EOF
import json, sys, os
steer = json.load(open(sys.argv[2])) if os.path.exists(sys.argv[2]) else {}
json.dump({
    "tag": "$TAG", "arm": "$ARM", "seed": $SEED, "parent": "$PARENT", "parent_sha256": "$PARENT_SHA",
    "trainer_flags": "$COURSE_FLAGS $ARM_FLAGS $EXTRA".strip(), "envs": $ENVS, "iters": $ITERS, "runs": "$RUNS",
    "score_envs": $SCORE_ENVS, "score_seed": $SCORE_SEED, "score_suite": "showpiece (+ course world)",
    "checkpoint": os.path.basename("$CKPT"), "train_exit": $TRC, "timed_out": $TIMED_OUT,
    "train_seconds": $((T1 - T0)), "score_seconds": $((T2 - T1)), "score_exit": $SRC_RC,
    "steer_envs": $STEER_ENVS, "steer_seconds": $((T3 - T2)), "steer_exit": $STEER_RC,
    "steer_worst_ratio": steer.get("worst"), "steer_pass": steer.get("pass"),
    "steer_worst_moving": max((c["ratio"] for c in steer.get("per_cmd", []) if any(c["cmd"])), default=None),
    "steer_stand_ratio": next((c["ratio"] for c in steer.get("per_cmd", []) if not any(c["cmd"])), None),
    "steer_stand_mode": steer.get("stand_mode"), "steer_drive_limits_are_forces": steer.get("drive_limits_are_forces"),
    "throughput": "$STEPS_S", "last_train_log": """$LAST_LOG""",
    "git": "$GIT", "dirty_files": $DIRTY, "node": "$(hostname)",
    "slurm_job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}",
}, open(sys.argv[1], "w"), indent=1)
EOF
echo "== done $TAG: train $((T1 - T0)) s, score $((T2 - T1)) s, steer $((T3 - T2)) s, scores in $RUN/showpiece_s${SCORE_SEED}*.jsonl"
[ "$SRC_RC" -eq 0 ] || exit "$SRC_RC"
exit "$STEER_RC"
