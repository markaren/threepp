#!/usr/bin/env bash
# Build threepp's Python module with GPU PhysX on IDUN, inside the same
# manylinux_2_28 container the release wheels are built in.
#
# Run on a LOGIN node (that is where IDUN says to compile, and where
# /localscratch + --fakeroot are available for the container build):
#
#   bash build_container.sh
#
# Phases, each skipped when its product already exists (delete the product to
# redo that phase):
#   1. clone threepp ($BRANCH) into $SRC
#   2. sandbox from quay.io/pypa/manylinux_2_28_x86_64 + the X11/GL headers
#      GLFW needs + cmake/ninja in its CPython 3.10, packed into $SIF
#   3. in the container: vcpkg (pinned $VCPKG_TAG) installs the physx feature
#      (and vulkan if WITH_VULKAN=ON) into $VCPKG_INSTALLED on the host
#   4. in the container: cmake configures and builds threepp_py IN-TREE, so
#      the module lands at $SRC/python/threepp/threepp.cpython-310-*.so
#   5. writes $WORK/threepp-env.sh for the runtime and import-tests the
#      module with the HOST Python 3.10 module
#
# Why in-tree and not a wheel: the examples put the repo's python/ directory
# first on sys.path and load the package there, so a module built in the tree
# is what they find. The container's CPython 3.10 and the module-system
# Python/3.10.8 share the cp310 ABI; manylinux_2_28 targets glibc 2.28 and
# links its newer libstdc++ pieces statically, so the module loads on Rocky 9.
#
# Runtime contract (threepp-env.sh): PhysX dlopens libPhysXGpu_64.so by bare
# name and the vcpkg port installs it under x64-linux/tools/, not lib/, so
# LD_LIBRARY_PATH must name that directory. Vulkan is OFF by default: no IDUN
# GPU exposes VK_KHR_ray_query, so the renderer cannot run there anyway, and
# leaving it out skips the glslang build.
#
# Things most likely to need a second try, in order:
#   - apptainer build of the sandbox: needs outbound HTTPS to quay.io (login
#     nodes have it) and the fakeroot mapping the docs describe.
#   - vcpkg bootstrap: downloads a vcpkg binary; PhysX then compiles from
#     source (several minutes on 16 cores, longer on a busy login node).
#   - the host import test: a "GLIBCXX_x.y.z not found" here means the
#     toolset trick did not hold; report the message.
set -euo pipefail

WORK=${WORK:-/cluster/work/$USER}
SRC=${SRC:-$WORK/threepp}
BRANCH=${BRANCH:-dev}
SIF=${SIF:-$WORK/threepp-build.sif}
SCRATCH=${SCRATCH:-/localscratch/$USER}
IMAGE=${IMAGE:-docker://quay.io/pypa/manylinux_2_28_x86_64}
VCPKG_TAG=${VCPKG_TAG:-2026.01.16}
VCPKG_ROOT_HOST=${VCPKG_ROOT_HOST:-$WORK/vcpkg}
VCPKG_INSTALLED=${VCPKG_INSTALLED:-$WORK/vcpkg_installed}
WITH_VULKAN=${WITH_VULKAN:-OFF}
BUILD_DIR=${BUILD_DIR:-$SRC/build/idun$([ "${PYTAG:-310}" = 310 ] || echo "-cp${PYTAG}")}
# Interpreter: PYTAG picks the container's CPython and must match the host
# module the runtime loads. 310 pairs with Python/3.10.8-GCCcore-12.2.0 (the
# laptop's CUDA-torch interpreter, so checkpoints move both ways); IDUN's
# newest module is 3.13.5, e.g.
#   PYTAG=313 HOST_PY_MODULE=Python/3.13.5-GCCcore-14.3.0 bash build_container.sh
# CMake and Ninja always come from the cp310 tools install in the SIF.
PYTAG=${PYTAG:-310}
HOST_PY_MODULE=${HOST_PY_MODULE:-Python/3.10.8-GCCcore-12.2.0}
CONTAINER_PY=/opt/python/cp${PYTAG}-cp${PYTAG}/bin/python
TOOLS_PY=/opt/python/cp310-cp310/bin/python
_n=$(nproc); JOBS=${JOBS:-$(( _n < 16 ? _n : 16 ))}
# IDUN users have no /etc/subuid range, so plain --fakeroot falls back to
# running the HOST's faked daemon inside the container -- and that binary
# wants Rocky 9's glibc 2.34 while manylinux_2_28 carries 2.28 ("GLIBC_2.33
# not found", measured 2026-09-08). --ignore-fakeroot-command keeps the
# root-mapped user namespace and skips the host binary: uid 0 inside, the
# sandbox files owned by you outside, which is all yum and pip need here.
FAKEROOT=${FAKEROOT:---fakeroot --ignore-fakeroot-command}

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

mkdir -p "$WORK"

# ── 1. source ─────────────────────────────────────────────────────────────────
say "1/5 source: $SRC ($BRANCH)"
if [ -d "$SRC/.git" ]; then
    # Report the commit AND say whether it is current. This phase used to only
    # print, so a rerun after pushing new work happily rebuilt the OLD tree and
    # announced success — the most expensive kind of quiet. UPDATE=1 fast-
    # forwards; it is off by default because the tree may carry local patches
    # (an scp'd diff is how work reaches this machine before it is pushed).
    echo "present: $(git -C "$SRC" log -1 --format='%h %s' 2>/dev/null)"
    if [ -n "$(git -C "$SRC" status --porcelain 2>/dev/null)" ]; then
        echo "NOTE: the checkout has local modifications; leaving them alone."
    elif [ "${UPDATE:-0}" = "1" ]; then
        git -C "$SRC" fetch origin "$BRANCH" && \
            git -C "$SRC" merge --ff-only "origin/$BRANCH" && \
            echo "updated to: $(git -C "$SRC" log -1 --format='%h %s')"
    else
        behind=$(git -C "$SRC" rev-list --count "HEAD..origin/$BRANCH" 2>/dev/null || echo 0)
        [ "${behind:-0}" != "0" ] && echo "NOTE: $behind commit(s) behind origin/$BRANCH - rerun with UPDATE=1 to fast-forward."
    fi
else
    git clone -b "$BRANCH" https://github.com/markaren/threepp.git "$SRC"
fi

# ── 2. container ──────────────────────────────────────────────────────────────
say "2/5 container: $SIF"
if [ -f "$SIF" ]; then
    echo "present"
else
    mkdir -p "$SCRATCH"
    pushd "$SCRATCH" >/dev/null
    if [ ! -d manylinux ]; then
        apptainer build --sandbox manylinux "$IMAGE"
    fi
    # The CI wheel job's yum line (GLFW's X11/Wayland/GL headers, vcpkg's zip
    # tools), plus a CMake/Ninja/stubgen in the container's CPython 3.10.
    # shellcheck disable=SC2086  # FAKEROOT is deliberately word-split
    apptainer exec $FAKEROOT --writable --no-mount hostfs --pwd / manylinux bash -c "
        set -e
        yum install -y libX11-devel libXrandr-devel libXinerama-devel libXcursor-devel \
                       libXi-devel mesa-libGL-devel libxcb-devel wayland-devel \
                       pkgconfig zip unzip tar
        $TOOLS_PY -m pip install --no-cache-dir 'cmake>=3.24' ninja pybind11-stubgen==2.5.5
        yum clean all"
    apptainer build $FAKEROOT "$SIF" manylinux
    # the OCI layers carry files without owner write permission; rm needs it
    chmod -R u+rwX manylinux 2>/dev/null || true
    rm -rf manylinux
    popd >/dev/null
fi

# ── 3 + 4. vcpkg deps and the build, inside the container ─────────────────────
# One inner script for both phases; it runs against the host checkout and the
# host vcpkg tree through the bind mount, so every product persists outside
# the container.
INNER="$WORK/threepp-build-inner.sh"
cat > "$INNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PATH=/opt/python/cp310-cp310/bin:\$PATH
export VCPKG_ROOT="$VCPKG_ROOT_HOST"
export VCPKG_DEFAULT_BINARY_CACHE="$WORK/vcpkg-cache"
mkdir -p "\$VCPKG_DEFAULT_BINARY_CACHE"

echo "== 3/5 vcpkg: $VCPKG_INSTALLED"
if [ ! -x "\$VCPKG_ROOT/vcpkg" ]; then
    git clone --depth 1 --branch "$VCPKG_TAG" https://github.com/microsoft/vcpkg.git "\$VCPKG_ROOT"
    "\$VCPKG_ROOT/bootstrap-vcpkg.sh" -disableMetrics
fi
features="--x-feature=physx"
[ "$WITH_VULKAN" = "ON" ] && features="\$features --x-feature=vulkan"
cd "$SRC"
"\$VCPKG_ROOT/vcpkg" install --triplet x64-linux \$features --x-install-root="$VCPKG_INSTALLED"
ls "$VCPKG_INSTALLED/x64-linux/tools/libPhysXGpu_64.so" >/dev/null \
    || { echo "libPhysXGpu_64.so missing from the vcpkg install (GPU dynamics would be unavailable)"; exit 1; }

echo "== 4/5 cmake: $BUILD_DIR"
vk_args=""
if [ "$WITH_VULKAN" = "ON" ]; then
    vk_args="-DTHREEPP_WITH_VULKAN=ON -DGLSLANG_VALIDATOR=$VCPKG_INSTALLED/x64-linux/tools/glslang/glslangValidator"
fi
cmake -S "$SRC" -B "$BUILD_DIR" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DTHREEPP_WITH_PYTHON:BOOL=ON -DPYBIND11_FINDPYTHON=ON \
    -DPython_EXECUTABLE=$CONTAINER_PY \
    -DTHREEPP_BUILD_EXAMPLES=OFF -DTHREEPP_BUILD_TESTS=OFF -DTHREEPP_WITH_AUDIO=OFF \
    -DCMAKE_PREFIX_PATH="$VCPKG_INSTALLED/x64-linux" \
    -Dunofficial-omniverse-physx-sdk_DIR="$VCPKG_INSTALLED/x64-linux/share/unofficial-omniverse-physx-sdk" \
    \$vk_args
cmake --build "$BUILD_DIR" --target threepp_py -j"$JOBS"
ls -la "$SRC"/python/threepp/threepp.cpython-${PYTAG}-*.so
EOF
chmod +x "$INNER"

say "3/5 + 4/5 inside the container (log: $WORK/threepp-build.log)"
apptainer exec --bind "$WORK" "$SIF" bash "$INNER" 2>&1 | tee "$WORK/threepp-build.log"

# ── 5. runtime environment + host import test ─────────────────────────────────
say "5/5 runtime: $WORK/threepp-env.sh"
cat > "$WORK/threepp-env.sh" <<EOF
# source this before running threepp on IDUN (login or compute node)
if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
        [ -f "\$f" ] && . "\$f" && break
    done
fi
module load $HOST_PY_MODULE
# PhysX finds libPhysXGpu_64.so by bare name: vcpkg puts it in tools/, not lib/
export LD_LIBRARY_PATH="$VCPKG_INSTALLED/x64-linux/tools:$VCPKG_INSTALLED/x64-linux/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
# the in-tree module, ahead of any pip-installed threepp in ~/.local
export PYTHONPATH="$SRC/python\${PYTHONPATH:+:\$PYTHONPATH}"
export THREEPP_SRC="$SRC"
EOF

# shellcheck disable=SC1090
. "$WORK/threepp-env.sh"
python - <<'EOF'
import threepp as tp
print("threepp:", tp.__file__)
print("HAS_PHYSX", tp.HAS_PHYSX, "| HAS_VULKAN", tp.HAS_VULKAN, "| HAS_EGL", tp.HAS_EGL)
print("egl_available:", tp.egl_available(), "(False on a login node is correct: no GPU driver there)")
assert tp.HAS_PHYSX, "PhysX did not make it into the module"
EOF

cat <<EOF

build done. Next, on a GPU node:

  srun --account=<acct> --partition=GPUQ --gres=gpu:1 --constraint=h100 \\
       --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash
  source $WORK/threepp-env.sh
  python -c "import threepp as tp; w = tp.PhysxWorld(gpu_dynamics=True); print('GPU PhysX ok')"
  python -m pip install --user numpy torch --index-url https://download.pytorch.org/whl/cu126
  cd $SRC/python/examples/spot && python train_spot_steps.py --envs 2048 --iters 100 --graph

The trainer warm-starts from spot/scratch_distillation/scratch_flat_best.pt,
which is git-ignored: copy it from the laptop to that path under
$SRC/python/examples/ first (or pass --warmstart <file>), or train the base
gait on the cluster with scratch_distillation/train_scratch.py. The steps/s
figure on each log line is the number to compare with the 40k measured on an
RTX 4070.
EOF
