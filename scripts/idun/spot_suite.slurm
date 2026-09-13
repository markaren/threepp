#!/usr/bin/env bash
# E0-lite on IDUN: score checkpoints on the episode-counted cells of score_e0.py, zero training.
# Each array task runs PER_TASK consecutive lines of a jobs.tsv (label, checkpoint, score seed,
# extras; written by scripts/idun/make_e0_jobs.py) and ALWAYS runs every line: one failed item costs
# that item, and the task's exit code says whether any failed. Per item: score_e0.py, plus the legacy
# hard:analytic cell (score_matrix.py --cells legacy1) when extras has 'legacy' and the held-out
# flat-steering eval (train_spot_steps.py --eval --eval-json) when it has 'steer'.
#
#   python scripts/idun/make_e0_jobs.py --out $WORK/spot_e0/jobs.tsv                  # ONCE
#   sbatch --array=0 --export=ALL,PER_TASK=7 scripts/idun/spot_suite.slurm            # pilot
#   sbatch --array=1-17%20 --export=ALL,PER_TASK=7 scripts/idun/spot_suite.slurm      # the rest
# make_e0_jobs.py prints both lines with the PER_TASK it sized the list for — paste those rather
# than these, since a task reads its lines by PER_TASK and a mismatch silently skips or repeats some.
#
# Afterwards:
#   python python/examples/spot/score_e0.py --aggregate $WORK/spot_e0
#
# Provenance per task in $OUT/_tasks/task<ID>/: manifest.json (git sha, sha256 of `git diff HEAD` and
# of every untracked file under the scored paths, sha256 of the threepp .so, node, per-item exit code
# and seconds), the diff itself and `git status --porcelain`.
#
#SBATCH --account=share-ie-iir
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:15:00
#SBATCH --job-name=spotsuite
#SBATCH --output=/cluster/work/laht/logs/spotsuite-%A_%a.out
#SBATCH --error=/cluster/work/laht/logs/spotsuite-%A_%a.out
set -euo pipefail

WORK=${WORK:-/cluster/work/$USER}
SRC=${SRC:-$WORK/threepp}
OUT=${OUT:-$WORK/spot_e0}
LIST=${LIST:-$OUT/jobs.tsv}
PER_TASK=${PER_TASK:-7}   # ~65 s per E0 item on a 4070 at K=2048, so 7 items is ~8 min. The PILOT is
                          # the long task: it also carries 4 steering evals and the legacy cell, so
                          # budget ~15 min for it and ~10 for the rest (hence --time 01:15:00).
ENVS=${ENVS:-2048}
LEGACY_ENVS=${LEGACY_ENVS:-512}
# Extra score_e0.py flags for every item, e.g. the showpiece suite into its own root:
#   sbatch --array=0 --export=ALL,OUT=$WORK/spot_showpiece,PER_TASK=2,SCORE_ARGS="--suite showpiece" ...
SCORE_ARGS=${SCORE_ARGS:-}

. "$WORK/threepp-env.sh"

ID=${SLURM_ARRAY_TASK_ID:-0}
LS=${LOCALSCRATCH:-/localscratch/$SLURM_JOB_ID}
[ -d "$LS" ] || LS=/localscratch/$USER/${SLURM_JOB_ID:-$$}
mkdir -p "$LS/warp" "$LS/tmp"
export WARP_CACHE_PATH="$LS/warp" TMPDIR="$LS/tmp"
export OMP_NUM_THREADS=4
trap 'rm -rf "$LS" 2>/dev/null || true' EXIT

[ -s "$LIST" ] || { echo "!! no job list at $LIST (run make_e0_jobs.py)"; exit 1; }
TD="$OUT/_tasks/task$ID"
mkdir -p "$TD" "$WORK/logs"
ITEMS="$TD/items.tsv"
: > "$ITEMS"

# ---- provenance ----------------------------------------------------------------------------------
GIT=$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo unknown)
git -C "$SRC" diff HEAD > "$TD/git.diff" 2>/dev/null || true
git -C "$SRC" status --porcelain > "$TD/git_status.txt" 2>/dev/null || true
DIFF_SHA=$(sha256sum "$TD/git.diff" | cut -d' ' -f1)
# new files are not in `git diff`: hash every untracked file under the paths that get scored
( cd "$SRC" && git ls-files --others --exclude-standard -- python/examples/spot/*.py python/threepp/rl scripts/idun \
    2>/dev/null | xargs -r sha256sum ) > "$TD/untracked.sha256" || true
SO=$(ls "$SRC"/python/threepp/threepp*.so 2>/dev/null | head -1 || true)
SO_SHA=$([ -n "$SO" ] && sha256sum "$SO" | cut -d' ' -f1 || echo missing)
echo "== spotsuite task $ID  lines $((ID * PER_TASK + 1))-$(((ID + 1) * PER_TASK)) of $LIST  git=${GIT:0:8}  node=$(hostname)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

cd "$SRC/python/examples/spot"
FAIL=0

run_item() {   # run_item <label> <seed> <name> <logfile> <cmd...>: never aborts the task
    local label="$1" seed="$2" name="$3" log="$4"; shift 4
    local t0 rc
    t0=$(date +%s)
    set +e
    "$@" < /dev/null > "$log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$seed" "$name" "$rc" "$(( $(date +%s) - t0 ))" >> "$ITEMS"
    echo "   $name  exit $rc  $(( $(date +%s) - t0 )) s  ($log)"
    [ "$rc" -eq 0 ] || FAIL=1
}

while IFS=$'\t' read -r LABEL CKPT SEED EXTRAS; do
    [ -n "${LABEL:-}" ] || continue
    SAFE=$(printf '%s' "$LABEL" | tr ':/' '__')
    D="$OUT/$SAFE"
    mkdir -p "$D"
    REP=""; RFLAG=""
    case ",$EXTRAS," in *,repeat,*) REP="_rep"; RFLAG="--repeat";; esac
    echo "-- $LABEL  seed $SEED  extras $EXTRAS  ($CKPT)"
    if [ ! -s "$CKPT" ]; then
        printf '%s\t%s\t%s\t%s\t%s\n' "$LABEL" "$SEED" "missing_checkpoint" 127 0 >> "$ITEMS"
        echo "   !! no checkpoint at $CKPT"; FAIL=1; continue
    fi
    JS="$D/e0_s${SEED}${REP}.jsonl"
    rm -f "$JS"
    run_item "$LABEL" "$SEED" "e0_s${SEED}${REP}" "$D/e0_s${SEED}${REP}.log" \
        python -u score_e0.py "$CKPT" --label "$LABEL" --seed "$SEED" --envs "$ENVS" --json "$JS" $RFLAG $SCORE_ARGS
    case ",$EXTRAS," in *,legacy,*)
        rm -f "$D/legacy1.jsonl"
        run_item "$LABEL" 0 legacy1 "$D/legacy1.log" \
            python -u score_matrix.py "$CKPT" --cells legacy1 --envs "$LEGACY_ENVS" --seed 0 --json "$D/legacy1.jsonl";;
    esac
    case ",$EXTRAS," in *,steer,*)
        run_item "$LABEL" 0 steer "$D/steer.log" \
            python -u train_spot_steps.py --eval "$CKPT" --score-envs "$LEGACY_ENVS" --eval-json "$D/steer.json";;
    esac
done < <(sed -n "$((ID * PER_TASK + 1)),$(((ID + 1) * PER_TASK))p" "$LIST")

# ---- manifest ------------------------------------------------------------------------------------
python - "$TD/manifest.json" "$ITEMS" <<EOF
import json, sys
items = []
for line in open(sys.argv[2]):
    label, seed, name, rc, sec = line.rstrip("\n").split("\t")
    items.append({"label": label, "seed": int(seed), "item": name, "exit": int(rc), "seconds": int(sec)})
json.dump({
    "task": $ID, "per_task": $PER_TASK, "list": "$LIST", "envs": $ENVS, "score_args": "$SCORE_ARGS",
    "git": "$GIT", "git_diff_sha256": "$DIFF_SHA", "git_diff": "$TD/git.diff",
    "untracked_sha256": open("$TD/untracked.sha256").read().splitlines(),
    "threepp_so": "$SO", "threepp_so_sha256": "$SO_SHA", "node": "$(hostname)",
    "slurm_job": "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}",
    "items": items, "failed": sum(1 for i in items if i["exit"] != 0),
}, open(sys.argv[1], "w"), indent=1)
EOF
echo "== task $ID done: $(wc -l < "$ITEMS") items, failed=$FAIL, manifest $TD/manifest.json"
exit "$FAIL"
