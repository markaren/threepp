#!/usr/bin/env bash
# Spot FALL RECOVERY wave 3 on IDUN (python/examples/spot/spot_recovery_env.py, train_spot_recovery.py): arm x seed, one
# big GPU each. Train, then evaluate the final weights from 2048 fallen starts at L3 (40% back, 40% side, 20% any
# orientation) and again from L1 and L2 starts to eval.json, with joint speed metrics, and write a manifest.
#
#   mkdir -p /cluster/work/$USER/logs                                                          # ONCE
#   sbatch --array=0-3 --export=ALL,ITERS=2000,RUNS_NAME=spot_recovery3 scripts/idun/spot_recovery.slurm   # A s0, B s0, A s1, B s1
#   sbatch --array=0-1 --export=ALL,ITERS=20,RUNS_NAME=spot_recovery3_check scripts/idun/spot_recovery.slurm
#
# Index -> (arm, seed): arm = ID % 2, seed = ID / 2, so 0 = A s0, 1 = B s0, 2 = A s1, 3 = B s1.
#   A  --qd-max 10   (a drive target moves at most 10 rad/s)
#   B  --qd-max 20
# Wave 3 makes the motion physically plausible. Wave 2 (spot_recovery2) got up from 99.5-100% of the hardest falls, but
# watched in spot_slam "the speed of motion it does does not seem physically possible": after the limp settle recovA_s0
# and recovB_s0 reach |qd| p99 14-27 and max 34-72 rad/s (256 L3 starts, 2026-09-14), where only PhysX's default max
# joint velocity (100 rad/s) caps a joint.
# Both arms continue wave 2's recovA_s0 whole (WARM) at its own action scale 0.5; the task refuses a WARM whose meta says
# another scale, so the trainer never applies a 0.2 -> S rescale on top. Common flags, wave 2's plus the speed penalty:
#   --action-scale 0.5 --log-std-min -0.8 --hard-frac 0.35 --w-prog 5 --up-slope 0.5 --fallen-effort-free --w-down 0.03
#   --w-qd-excess $W_QD_EXCESS
# W_QD_EXCESS: see its line below. Plant as before: stiff gains, real torque limits 45/45/115 N·m, self-collision, flat
# ground, 6 s episodes, the fall curriculum L0 (tilted <= 25 deg) -> L3 (back / side / anyhow), 18% standing starts.
# Wave 2 is this script at 75003d18 (plus the GPU and node lines since).
#
# Walltime: 2000 iterations x 2048 envs x 32 steps = 131M steps. Wave 2 ran 57.5k steps/s on an H100 (3000 iterations in
# 3421 s), so about 38 min here, well inside the 110 min before the 10 min eval reserve. A task that still runs out is
# stopped by the timeout below and evaluates the _latest checkpoint (saved every log).
# A task refuses a run directory that already holds a manifest.json (OVERWRITE=1 to allow).
#
#SBATCH --account=share-ie-iir
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
# H100 or H200, 6 cores, 32 GB: the loop uses ~3 cores, and an h100-only 12-core request left 3 of 4
# array tasks pending on (Resources) on 2026-09-13 (free GPUs sit on CPU-starved nodes). No A100: it ran this loop at
# about half the H100 speed, and an array waits for its slowest task.
#SBATCH --constraint=h100|h200
# idun-01-01 cancelled one task (by uid 0) and hung another before the trainer printed a line, 2026-09-13/14.
#SBATCH --exclude=idun-01-01
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
RUNS_NAME=${RUNS_NAME:-spot_recovery3}
RUNS=${RUNS:-$WORK/$RUNS_NAME}
WARM=${WARM:-$WORK/spot_recovery2/recovA_s0/spot_recovery.pt}   # wave 2 arm A seed 0: /cluster/work/laht/spot_recovery2/...
ENVS=${ENVS:-2048}
ITERS=${ITERS:-2000}
EVAL_ENVS=${EVAL_ENVS:-2048}
SNAPSHOT_EVERY=${SNAPSHOT_EVERY:-250}
EXTRA=${EXTRA:-}
EVAL_RESERVE_MIN=${EVAL_RESERVE_MIN:-10}
SCALE=${SCALE:-0.5}
QD_MAX_A=${QD_MAX_A:-10}
QD_MAX_B=${QD_MAX_B:-20}
LOG_STD_MIN=${LOG_STD_MIN:--0.8}
HARD_FRAC=${HARD_FRAC:-0.35}
W_PROG=${W_PROG:-5.0}
UP_SLOPE=${UP_SLOPE:-0.5}
W_DOWN=${W_DOWN:-0.03}
# W_QD_EXCESS 0.02, measured 2026-09-14 on the laptop (K=256, recovA_s0 + the -0.8 exploration noise in this wave's training
# mix with the cap on, no training): a unit weight costs 0.61 per live tick at qd-max 10 (1.58 on ticks spent down, 0.06
# upright; nonzero on 31% of ticks, p99 9, max 81; the target cap alone still leaves |qd| p99 11-17 and max 30-41 rad/s from
# whips and bounces) and 0.094 at 20 (0.81 down). At 0.02 a down tick pays 0.03 at qd-max 10, the size of w_down and under
# a third of the ~0.11 progress pay of a rolling tick, so trying stays worth more than lying still; upright pays 0.001.
W_QD_EXCESS=${W_QD_EXCESS:-0.02}

. "$WORK/threepp-env.sh"

ID=${SLURM_ARRAY_TASK_ID:-0}
SEED=$(( ID / 2 ))
if [ $(( ID % 2 )) -eq 0 ]; then ARM=A; QD=$QD_MAX_A; else ARM=B; QD=$QD_MAX_B; fi
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
# The continue path rescales the actor whenever the checkpoint's action scale differs from --action-scale. recovA_s0 trained
# at 0.5, so its meta must say 0.5 here (a walking or pilot checkpoint at 0.2 would get the x 0.4 rescale again).
WARM_SCALE=$(python -c "import sys, torch; print(float(torch.load(sys.argv[1], map_location='cpu', weights_only=True)['meta'].get('action_scale', 0.2)))" "$WARM")
if [ "$WARM_SCALE" != "$(python -c "import sys; print(float(sys.argv[1]))" "$SCALE")" ] && [ "${ALLOW_RESCALE:-0}" != 1 ]; then
    echo "!! $WARM trained at action scale $WARM_SCALE, this wave runs $SCALE: the trainer would rescale it (ALLOW_RESCALE=1)"; exit 1
fi
WAVE_ARGS=(--action-scale "$SCALE" --qd-max "$QD" --w-qd-excess "$W_QD_EXCESS" --log-std-min "$LOG_STD_MIN"
           --hard-frac "$HARD_FRAC" --w-prog "$W_PROG" --up-slope "$UP_SLOPE" --fallen-effort-free --w-down "$W_DOWN")
GIT=$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)
DIRTY=$(git -C "$SRC" status --porcelain 2>/dev/null | wc -l)
echo "== $TAG  arm=$ARM seed=$SEED envs=$ENVS iters=$ITERS  warm=$WARM (scale $WARM_SCALE)  ${WAVE_ARGS[*]}  git=$GIT (+$DIRTY dirty)  node=$(hostname)"
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
RESCALE_LINE=$(grep -m1 -E "warm (start's action scale|rescale)" "$RUN/train.log" | tr -d '"' || true)

# ---- evaluate from fallen starts: L3, then L1 and L2 (under the checkpoint's own qd_max) --------------------
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
    "warmstart_action_scale": $WARM_SCALE, "warm_rescale_log": "$RESCALE_LINE",
    "wave_args": "${WAVE_ARGS[*]}", "action_scale": $SCALE, "qd_max": $QD, "w_qd_excess": $W_QD_EXCESS,
    "envs": $ENVS, "iters": $ITERS, "extra": "$EXTRA", "runs": "$RUNS", "checkpoint": os.path.basename("$CKPT"),
    "train_exit": $TRC, "timed_out": $TIMED_OUT, "train_seconds": $((T1 - T0)), "throughput": "$STEPS_S",
    "eval_envs": $EVAL_ENVS, "eval_exit": $ERC, "eval_seconds": $((T2 - T1)),
    "eval_success_3s": fa.get("success_3s"), "eval_success_6s": fa.get("success_6s"),
    "eval_recovery_s": fa.get("recovery_s"), "eval_blowups": fa.get("blowups"), "eval_gave_up": fa.get("gave_up"),
    "eval_gave_up_L3": ev.get("gave_up_L3"),
    "eval_success_6s_by_level": {k: v["fallen"].get("success_6s") for k, v in bl.items()},
    "eval_gave_up_by_level": {k: v["fallen"].get("gave_up") for k, v in bl.items()},
    "eval_qd_max": ev.get("qd_max"), "eval_qd_L3": ev.get("qd"), "eval_cmd_qd_max_hx_hy_kn": ev.get("cmd_qd_max_hx_hy_kn"),
    "eval_qd_by_level": {k: v.get("qd") for k, v in bl.items()},
    "eval_tau_p95_hx_hy_kn": ev.get("tau_p95_hx_hy_kn"), "eval_contact_ticks_per_ep": fa.get("contact_ticks_per_ep"),
    "git": "$GIT", "dirty_files": $DIRTY, "node": "$(hostname)",
    "slurm_job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}",
}, open(sys.argv[1], "w"), indent=1)
EOF
echo "== done $TAG: train $((T1 - T0)) s, eval $((T2 - T1)) s -> $RUN/eval.json"
[ "$TRC" -eq 0 ] || [ "$TRC" -eq 124 ] || exit "$TRC"
exit "$ERC"
