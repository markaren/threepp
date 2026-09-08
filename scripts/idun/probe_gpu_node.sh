#!/usr/bin/env bash
# Reconnaissance for running threepp + warp on an IDUN GPU node.
#
# Run it INSIDE an interactive GPU allocation, e.g.
#   srun --account=<acct> --partition=GPUQ --gres=gpu:1 --constraint=h100 \
#        --time=00:20:00 --pty bash
#   bash scripts/idun/probe_gpu_node.sh 2>&1 | tee probe_$(hostname).txt
#
# It answers, in order, the four questions that decide the whole plan:
#   1. what GPU and driver the job got
#   2. whether the host driver install carries a Vulkan ICD at all
#      (HPC "compute-only" installs often omit libGLX_nvidia + nvidia_icd.json;
#      without them Vulkan cannot see the GPU, and Apptainer --nv cannot bind
#      what the host does not have)
#   3. whether the Slurm node can reach the internet (pip, FetchContent)
#   4. what the module system offers for the build (Python 3.10, CUDA, CMake, GCC)
#
# Nothing here installs or writes anything outside the current directory.

set -u
section() { printf '\n=== %s ===\n' "$*"; }

section "host / job"
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<none: not inside a job>}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-?}  GRES=${SLURM_JOB_GRES:-?}"
cat /etc/os-release 2>/dev/null | grep -E '^(PRETTY_NAME|VERSION_ID)='
uname -r
nproc; free -g | head -2

section "GPU + driver"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,pstate --format=csv 2>&1
nvidia-smi -L 2>&1
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
ls -la /dev/nvidia* 2>/dev/null

section "Vulkan ICD on the HOST (the decisive check)"
for f in /usr/share/vulkan/icd.d/nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json \
         /usr/share/vulkan/icd.d/nvidia_icd.x86_64.json; do
  [ -e "$f" ] && { echo "FOUND $f"; cat "$f"; } || echo "absent $f"
done
echo "-- driver libraries that carry the Vulkan/graphics half --"
for lib in libGLX_nvidia.so.0 libnvidia-glcore.so libnvidia-vulkan-producer.so \
           libvulkan.so.1 libnvidia-egl-gbm.so libEGL_nvidia.so.0; do
  found=$(ldconfig -p 2>/dev/null | grep -m1 "$lib" | awk '{print $NF}')
  if [ -n "$found" ]; then echo "FOUND   $lib -> $found"; else
    alt=$(ls /usr/lib64/$lib* /usr/lib/x86_64-linux-gnu/$lib* 2>/dev/null | head -1)
    [ -n "$alt" ] && echo "FOUND   $lib -> $alt (not in ldconfig)" || echo "absent  $lib"
  fi
done
command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary 2>&1 | head -60 \
  || echo "(vulkaninfo not installed on host; a container with vulkan-tools can run it under --nv)"

section "Apptainer"
command -v apptainer && apptainer --version || echo "apptainer not on PATH"
echo "-- what --nv would bind (from the apptainer nvliblist) --"
for c in /etc/apptainer/nvliblist.conf /etc/singularity/nvliblist.conf; do
  [ -e "$c" ] && { echo "$c:"; grep -E 'GLX_nvidia|nvidia_icd|glcore|vulkan' "$c"; }
done
ls -ld /localscratch 2>/dev/null || echo "no /localscratch on this node"
df -h /localscratch /cluster/work/"$USER" 2>/dev/null

section "internet from this node"
for u in https://pypi.org/simple/warp-lang/ https://github.com https://packages.lunarg.com; do
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 8 "$u" 2>/dev/null)
  echo "$u -> HTTP ${code:-none}"
done

section "modules of interest"
module avail Python 2>&1 | grep -E 'Python/3\.1[0-4]' | head
module avail CUDA 2>&1 | grep -E 'CUDA/1[2-9]' | head
module avail CMake 2>&1 | grep -E 'CMake/3\.(2[4-9]|[3-9][0-9])' | head
module avail GCC 2>&1 | grep -E '^ *GCC/1[2-9]' | head
module avail Vulkan Mesa glslang 2>&1 | grep -iE 'vulkan|mesa|glslang' | head

section "python 3.10 + pip reach"
module load Python/3.10.8-GCCcore-12.2.0 2>/dev/null && python -V && \
  python -m pip --version && \
  python -m pip download --no-deps -d /tmp/probe_pip_$$ warp-lang 2>&1 | tail -2 && \
  rm -rf /tmp/probe_pip_$$
echo
echo "probe done"
