#!/usr/bin/env bash
# Render a threepp+warp demo headless on one IDUN GPU and write an mp4.
#
#   sbatch scripts/idun/render_warp_demo.slurm                       # warp_fluid, 8 s
#   sbatch --export=ALL,DEMO=warp_cloth,SECONDS_=6 render_warp_demo.slurm
#
# Route: OpenGL through EGL on the device platform. No X server, no GLFW.
# threepp's Vulkan backend cannot run here (no VK_KHR_ray_query on the H100
# driver), so this is the only way this cluster makes pixels.
#
# ONCE, before the first sbatch (Slurm opens --output before the script runs,
# so it will not create the directory for you):
#   mkdir -p /cluster/work/$USER/logs /cluster/work/$USER/renders
#
# --constraint=h100 is what was measured. Drop it to widen the pool (any
# NVIDIA GPU with the EGL half of the driver works); check with
#   sinfo -p GPUQ -o '%20N %10c %10m %25f %30G'
#
#SBATCH --account=share-ie-iir
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --job-name=warpvid
#SBATCH --output=/cluster/work/laht/logs/warpvid-%j.out
#SBATCH --error=/cluster/work/laht/logs/warpvid-%j.out

set -euo pipefail

WORK=${WORK:-/cluster/work/$USER}
SRC=${SRC:-$WORK/threepp}
DEMO=${DEMO:-warp_fluid}
SECONDS_=${SECONDS_:-8}          # not SECONDS: bash owns that name
EXTRA=${EXTRA:-}                 # e.g. EXTRA="--scale 1.5"
OUTDIR=${OUTDIR:-$WORK/renders}
TAG=${TAG:-${SLURM_ARRAY_TASK_ID:-$SLURM_JOB_ID}}
FINAL="$OUTDIR/${DEMO}_${TAG}.mp4"

# ---------------------------------------------------------------- node-local --
# /localscratch is node-local NVMe and is WIPED when the job ends, so everything
# transient lives here and only the mp4 is copied out. warp_fluid now pipes
# frames straight to x264 rather than writing a PNG each, so the small-file
# storm this was written for is gone -- but warp's kernel cache still belongs
# on node-local disk (a shared ~/.cache/warp under an array is a contention and
# corruption hazard), and a demo that still writes intermediates gets a fast
# TMPDIR for free.
LS=${LOCALSCRATCH:-/localscratch/$SLURM_JOB_ID}
[ -d "$LS" ] || LS=/localscratch/$USER/$SLURM_JOB_ID
RUN="$LS/warpvid.$$"
mkdir -p "$RUN/tmp" "$RUN/out" "$OUTDIR" "$WORK/logs"
export TMPDIR="$RUN/tmp"

# The trap is the point: on a --time overrun Slurm sends SIGTERM before the
# node is reclaimed, so a partial mp4 still reaches $WORK.
rescue() {
    rc=$?
    for f in "$RUN"/out/*.mp4; do
        [ -e "$f" ] || continue
        cp -f "$f" "$OUTDIR/${DEMO}_${TAG}.mp4" 2>/dev/null || true
        echo "RESCUED $f -> $OUTDIR/${DEMO}_${TAG}.mp4 ($(stat -c%s "$f") bytes)"
    done
    rm -rf "$RUN" 2>/dev/null || true
    exit $rc
}
trap rescue EXIT TERM INT

# -------------------------------------------------------------- environment --
# ORDER MATTERS. FFmpeg is loaded FIRST so that threepp-env.sh's explicit
# `module load Python/3.10.8-GCCcore-12.2.0` runs last and wins any Python that
# the FFmpeg dependency chain drags in, and so that threepp-env.sh's
# LD_LIBRARY_PATH prepend (vcpkg PhysX) ends up AHEAD of the FFmpeg tree.
# Both modules are GCCcore-12.2.0, so they share one libstdc++ and one binutils.
module purge
module load FFmpeg/5.1.2-GCCcore-12.2.0
# shellcheck disable=SC1090
source "$WORK/threepp-env.sh"

module list 2>&1
echo "python : $(command -v python)  $(python -V 2>&1)"
echo "ffmpeg : $(command -v ffmpeg)"
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# No display of any kind. GLFW would take the Null platform and throw; EGL does
# not look at these, but an inherited stale DISPLAY makes the failure confusing.
unset DISPLAY WAYLAND_DISPLAY XDG_SESSION_TYPE

# Warp's kernel cache: per-job on node-local disk. A shared ~/.cache/warp under
# a job array is a lock-contention and corruption hazard; see the note at the
# bottom about pre-warming a read-only shared cache instead.
export WARP_CACHE_PATH="$RUN/warp_cache"
export PYTHONUNBUFFERED=1

echo "=== node ==="
hostname; nvidia-smi -L; echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "localscratch: $RUN  ($(df -h "$LS" | tail -1))"

# ------------------------------------------------------------- preflight #1 --
# Every one of these fails in seconds; without them the same faults surface
# only AFTER the render, as an exception or a 0-byte file.
echo "=== preflight: encoder ==="
ffmpeg -hide_banner -encoders 2>/dev/null | grep -q ' libx264 ' \
    || { echo "FAIL: this ffmpeg has no libx264; encode_png_sequence would fail after the render"; exit 3; }
command -v ffprobe >/dev/null || { echo "FAIL: no ffprobe"; exit 3; }

echo "=== preflight: hardware GL through EGL ==="
python "$SRC/python/examples/idun_egl_smoke.py" --width 256 --height 256 \
    || { echo "FAIL: no hardware EGL context on this node"; exit 4; }

# ------------------------------------------------------------- preflight #2 --
# The one that actually decides this job: warp's CUDA device and the EGL
# context's device must be the SAME GPU, or wp.RegisteredGLBuffer cannot
# register the renderer's VBO -- and warp_fluid's GL route has no host fallback.
echo "=== preflight: CUDA <-> GL interop ==="
python - <<'PY' || { echo "FAIL: CUDA/GL interop unavailable (device mismatch?)"; exit 5; }
import os, sys
sys.path.insert(0, os.path.join(os.environ["THREEPP_SRC"], "python"))
import threepp as tp, warp as wp
ctx = tp.EglContext(width=256, height=256, require_hardware=True)
print("GL_RENDERER      :", ctx.renderer)
print("egl device index :", ctx.device_index, " egl->cuda index:", ctx.cuda_device_index)
r = tp.GLRenderer(ctx, 256, 256)
s = tp.Scene(); c = tp.PerspectiveCamera(50, 1.0, 0.1, 10); c.position.z = 3
g = tp.BoxGeometry(1, 1, 1)
m = tp.Mesh(g, tp.MeshBasicMaterial()); s.add(m)
r.render(s, c)                      # the VBO does not exist until the first draw
wp.init()
dev = wp.get_preferred_device()
print("warp device      :", dev, "| CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
bid = r.gl_buffer_id(g, "position")
assert bid, "gl_buffer_id returned None"
b = wp.RegisteredGLBuffer(int(bid), dev, wp.RegisteredGLBuffer.WRITE_DISCARD)
a = b.map(dtype=wp.vec3, shape=(8,)); b.unmap()
print("interop OK: registered GL buffer", bid, "on", dev)
PY

# ------------------------------------------------------------------ render --
# warp_fluid writes its mp4 into the CWD, so cd decides where it lands.
cd "$RUN/out"
echo "=== render: $DEMO --video $SECONDS_ $EXTRA ==="
t0=$SECONDS
# --egl is the headless entry point: build tp.EglContext(W,H) + tp.GLRenderer(ctx,W,H)
# instead of tp.Canvas(). The EGL pbuffer IS framebuffer 0, so the drawable must
# be >= the render size or readback returns undefined bytes.
# shellcheck disable=SC2086
python "$SRC/python/examples/${DEMO}.py" --egl --video "$SECONDS_" $EXTRA
echo "render+encode wall: $((SECONDS - t0)) s"

# ------------------------------------------------------------------ verify --
MP4=$(ls -1 "$RUN"/out/*.mp4 2>/dev/null | head -1 || true)
[ -n "$MP4" ] || { echo "FAIL: the run produced no mp4"; exit 6; }

BYTES=$(stat -c%s "$MP4")
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$MP4")
NF=$(ffprobe -v error -select_streams v:0 -count_frames \
             -show_entries stream=nb_read_frames -of csv=p=0 "$MP4")
WH=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height \
             -of csv=p=0:s=x "$MP4")
echo "mp4: $MP4  $BYTES bytes  ${DUR}s  ${NF} frames  ${WH}"

EXPECT_F=$(python -c "print(int(round($SECONDS_*60)))")
MIN_BYTES=$(python -c "print(int($SECONDS_*80000))")   # ~80 KB/s floor; a all-black
                                                       # CRF18 clip encodes far under this
python - "$BYTES" "$DUR" "$NF" "$EXPECT_F" "$MIN_BYTES" <<'PY' || exit 7
import sys
b, d, nf, ef, mb = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
ok = True
def chk(c, msg):
    global ok
    print(("ok   " if c else "FAIL ") + msg); ok = ok and c
chk(b >= mb,              f"size {b} >= {mb} bytes (a black or frozen render falls under this)")
chk(abs(nf - ef) <= 2,    f"frames {nf} ~= {ef}")
chk(d > 0.9 * ef / 60.0,  f"duration {d:.2f}s ~= {ef/60.0:.2f}s")
sys.exit(0 if ok else 1)
PY

# A render that came up all black still encodes to a valid, correctly sized
# mp4 if the scene has a lit background, so ask ffmpeg directly.
BLACK=$(ffmpeg -v error -i "$MP4" -vf "blackdetect=d=0.4:pix_th=0.06" -f null - 2>&1 \
        | grep -c black_start || true)
echo "blackdetect segments: $BLACK"
[ "$BLACK" -eq 0 ] || { echo "FAIL: the video contains black stretches -- the render is not drawing"; exit 8; }

cp -f "$MP4" "$FINAL"
sync
echo "OK: $FINAL  ($(stat -c%s "$FINAL") bytes)"

# -------------------------------------------------------------------- array --
# To render many variants later (QOS 'high' caps you at 20 GPUs; short jobs
# backfill sooner, so keep --time honest rather than padded):
#
#   #SBATCH --array=0-19%10
#   #SBATCH --time=00:30:00
#   VARIANTS=(--scale\ 1 --scale\ 1.5 --scale\ 2 ...)
#   EXTRA=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
#   TAG=$SLURM_ARRAY_TASK_ID
#
# and pre-warm one shared warp kernel cache first so the 20 tasks do not each
# pay the JIT, and do not race one writable cache:
#
#   WARP_CACHE_PATH=$WORK/warp_cache  (one warm-up job writes it, then chmod -R a-w)
