#!/usr/bin/env bash
# Does threepp's Vulkan backend render on an IDUN GPU that has no VK_KHR_ray_query?
#
# The H100 driver exposes no ray query, and since 2026-10-11 the backend runs
# such a device in its no-RT mode instead of refusing it. This script installs
# the PyPI wheel into a directory of its own, takes one GPU through srun,
# renders a small lit scene headless, and prints what the renderer reports.
#
# Run it on a LOGIN node (it calls srun itself):
#   bash scripts/idun/vulkan_nort_check.sh
# or from a machine without a checkout on the cluster, in one ssh session:
#   ssh <user>@idun-login1.hpc.ntnu.no "tr -d '\r' | bash -s" < scripts/idun/vulkan_nort_check.sh
#
# Knobs (environment):
#   CONSTRAINT  Slurm feature, default h100; empty takes any GPUQ GPU
#   ACCOUNT     Slurm account, default share-ie-iir
#   THREEPP_PIP pip requirement, default threepp (the newest wheel)
#   THREEPP_ENV an environment file from build_container.sh; the check then
#               runs that build's in-tree module, not a wheel
#   EMIT_PNG=1  print the picture base64 between NORT_PNG markers, for a caller
#               that only has this session's stdout
#
# Writes only under /cluster/work/$USER/nort_check. Exit 0 = a picture that is
# neither flat nor black came off the GPU.

set -euo pipefail

WORK=${WORK:-/cluster/work/$USER}
DIR=$WORK/nort_check
ACCOUNT=${ACCOUNT:-share-ie-iir}
CONSTRAINT=${CONSTRAINT-h100}
THREEPP_PIP=${THREEPP_PIP:-threepp}
THREEPP_ENV=${THREEPP_ENV:-}
PYMOD=${PYMOD:-Python/3.10.8-GCCcore-12.2.0}

mkdir -p "$DIR"

# ------------------------------------------------------------ the render ---
cat > "$DIR/nort_render.py" <<'PY'
"""Render a small lit scene headless on the Vulkan backend and say what came out.

Needs only the threepp wheel and numpy. THREEPP_VULKAN_NO_RT=1 forces the no-RT
mode on a device that has ray query, which is how this is checked off-cluster.
"""
import math
import os
import struct
import sys
import time
import zlib

import numpy as np
import threepp as tp

W, H, FRAMES = 640, 360, 90
out = sys.argv[1] if len(sys.argv) > 1 else "nort.png"


def write_png(path, rgb):
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


print("threepp", getattr(tp, "__version__", "?"), "| HAS_VULKAN", getattr(tp, "HAS_VULKAN", False))
print("module:", tp.__file__)
print("THREEPP_VULKAN_NO_RT =", os.environ.get("THREEPP_VULKAN_NO_RT", "<unset>"))
if not getattr(tp, "HAS_VULKAN", False):
    print("RESULT: FAIL this wheel has no Vulkan backend")
    sys.exit(2)

def mat(color, roughness, metalness=0.0):
    m = tp.MeshStandardMaterial()
    m.color = tp.Color(color)
    m.roughness = roughness
    m.metalness = metalness
    return m


scene = tp.Scene()
scene.background = tp.Color(0.55, 0.70, 0.90)

ground = tp.Mesh(tp.PlaneGeometry(30, 30), mat(0x808a80, 0.9))
ground.rotation.x = -math.pi / 2
ground.receive_shadow = True
scene.add(ground)

colors = [0xd04030, 0x30a050, 0x3060d0, 0xd0b030, 0xa040c0]
for i, c in enumerate(colors):
    x = (i - 2) * 1.6
    if i % 2 == 0:
        m = tp.Mesh(tp.BoxGeometry(1, 1 + 0.4 * i, 1), mat(c, 0.5))
        m.position.set(x, 0.5 + 0.2 * i, 0)
    else:
        m = tp.Mesh(tp.SphereGeometry(0.6, 48, 24), mat(c, 0.25, 0.6))
        m.position.set(x, 0.6, 0.6)
    m.cast_shadow = True
    m.receive_shadow = True
    scene.add(m)

sun = tp.DirectionalLight(0xffffff, 3.0)
sun.position.set(4, 6, 3)
sun.cast_shadow = True
scene.add(sun)
scene.add(tp.HemisphereLight(0xbfd4ff, 0x40382c, 0.8))

camera = tp.PerspectiveCamera(45, W / H, 0.1, 100)
camera.position.set(0, 3.2, 8.5)
camera.look_at(0, 0.7, 0)

t0 = time.perf_counter()
canvas = tp.Canvas("nort", width=W, height=H, headless=True, vsync=False)
renderer = tp.VulkanRenderer(canvas)
t_init = time.perf_counter() - t0
rt = bool(renderer.ray_tracing)
print("renderer up in %.1f s | ray_tracing = %s" % (t_init, rt))

t0 = time.perf_counter()
renderer.render(scene, camera)
t_first = time.perf_counter() - t0
t0 = time.perf_counter()
for _ in range(FRAMES - 1):
    renderer.render(scene, camera)
t_rest = time.perf_counter() - t0
print("first frame %.1f s (pipeline compile) | then %.2f ms/frame over %d frames"
      % (t_first, 1000.0 * t_rest / (FRAMES - 1), FRAMES - 1))

try:
    ft = dict(renderer.frame_timings)
    print("frame_timings:", ", ".join("%s=%.3g" % (k, v) for k, v in sorted(ft.items())
                                      if isinstance(v, (int, float))))
except Exception as e:  # timings are a report, not the test
    print("frame_timings unavailable:", e)

img = np.ascontiguousarray(np.asarray(renderer.read_pixels())[..., :3])
if img.dtype != np.uint8:
    img = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
write_png(out, img)

lum = img.astype(np.float32).mean(axis=2)
mean = img.reshape(-1, 3).mean(axis=0)
black = 100.0 * float((lum < 2.0).mean())
# Under a lit object the ground is darker than beside it only if the sun's
# shadow (a screen-space depth march without ray query) is drawn.
print("picture %dx%d -> %s" % (img.shape[1], img.shape[0], out))
print("mean rgb %.1f %.1f %.1f | min %d max %d | black %.1f%% | distinct colours %d"
      % (mean[0], mean[1], mean[2], img.min(), img.max(), black,
         len(np.unique(img.reshape(-1, 3), axis=0))))

ok = img.max() != img.min() and black < 50.0 and lum.std() > 8.0
print("RESULT: %s ray_tracing=%s" % ("OK" if ok else "FAIL flat or black picture", rt))
sys.exit(0 if ok else 1)
PY

# ------------------------------------------------------- the loader probe ---
cat > "$DIR/nort_vkprobe.py" <<'PY'
"""What the Vulkan loader reports: instance extensions, devices, ray query."""
import ctypes
import glob
import os
import struct


class Ext(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char * 256), ("spec", ctypes.c_uint32)]


class CI(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_uint32), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("app", ctypes.c_void_p), ("nl", ctypes.c_uint32), ("l", ctypes.c_void_p),
                ("ne", ctypes.c_uint32), ("e", ctypes.c_void_p)]


def run(path):
    print("loader:", path)
    try:
        lib = ctypes.CDLL(path)
    except OSError as e:
        print("  cannot load:", e)
        return
    u32p = ctypes.POINTER(ctypes.c_uint32)
    lib.vkEnumerateInstanceExtensionProperties.argtypes = [ctypes.c_char_p, u32p, ctypes.c_void_p]
    lib.vkCreateInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    lib.vkEnumeratePhysicalDevices.argtypes = [ctypes.c_void_p, u32p, ctypes.c_void_p]
    lib.vkGetPhysicalDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.vkEnumerateDeviceExtensionProperties.argtypes = [ctypes.c_void_p, ctypes.c_char_p, u32p, ctypes.c_void_p]
    n = ctypes.c_uint32(0)
    lib.vkEnumerateInstanceExtensionProperties(None, ctypes.byref(n), None)
    arr = (Ext * max(n.value, 1))()
    lib.vkEnumerateInstanceExtensionProperties(None, ctypes.byref(n), arr)
    inst = sorted(arr[i].name.decode() for i in range(n.value))
    print("  instance extensions (%d): %s" % (len(inst), " ".join(inst)))
    print("  VK_EXT_headless_surface:", "yes" if "VK_EXT_headless_surface" in inst else "NO")
    ci = CI()
    ci.sType = 1
    h = ctypes.c_void_p()
    r = lib.vkCreateInstance(ctypes.byref(ci), None, ctypes.byref(h))
    if r != 0:
        print("  vkCreateInstance failed: VkResult", r)
        return
    n = ctypes.c_uint32(0)
    lib.vkEnumeratePhysicalDevices(h, ctypes.byref(n), None)
    devs = (ctypes.c_void_p * max(n.value, 1))()
    lib.vkEnumeratePhysicalDevices(h, ctypes.byref(n), devs)
    print("  physical devices:", n.value)
    for i in range(n.value):
        buf = ctypes.create_string_buffer(8192)
        lib.vkGetPhysicalDeviceProperties(devs[i], buf)
        api, drv, vendor, _d, dtype = struct.unpack_from("<IIIII", buf, 0)
        name = buf.raw[20:276].split(b"\0", 1)[0].decode(errors="replace")
        m = ctypes.c_uint32(0)
        lib.vkEnumerateDeviceExtensionProperties(devs[i], None, ctypes.byref(m), None)
        ext = (Ext * max(m.value, 1))()
        lib.vkEnumerateDeviceExtensionProperties(devs[i], None, ctypes.byref(m), ext)
        de = {ext[j].name.decode() for j in range(m.value)}
        print("  [%d] %s vendor 0x%04x type %d | %d device extensions | %s" % (
            i, name, vendor, dtype, m.value,
            " ".join("%s=%s" % (e[7:], "yes" if e in de else "NO") for e in (
                "VK_KHR_ray_query", "VK_KHR_acceleration_structure", "VK_KHR_ray_tracing_pipeline",
                "VK_KHR_swapchain", "VK_KHR_external_memory_fd"))))


here = os.path.dirname(os.path.abspath(__file__))
for p in ["libvulkan.so.1"] + sorted(glob.glob(os.path.join(here, "site", "threepp.libs", "libvulkan*"))):
    run(p)
PY

# --------------------------------------------------------- the GPU step ---
cat > "$DIR/nort_job.sh" <<'JOB'
#!/usr/bin/env bash
set -uo pipefail
DIR=$1; PYMOD=$2; ENVF=${3:-}
module purge >/dev/null 2>&1
module load "$PYMOD" >/dev/null 2>&1
if [ -n "$ENVF" ]; then
  # A source build: its environment file names the in-tree module and the
  # vcpkg libraries (the Vulkan loader among them); numpy comes from site_np.
  # shellcheck disable=SC1090
  . "$ENVF"
  export PYTHONPATH="$PYTHONPATH:$DIR/site_np"
else
  export PYTHONPATH="$DIR/site"
fi
export PYTHONUNBUFFERED=1
unset DISPLAY WAYLAND_DISPLAY XDG_SESSION_TYPE
echo "=== node ==="
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "=== Vulkan ICD manifests and driver libraries ==="
ICDS=""
for d in /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
  [ -d "$d" ] || { echo "absent $d"; continue; }
  for f in "$d"/*.json; do
    [ -e "$f" ] || continue
    echo "-- $f"; tr -d '\n' < "$f" | head -c 400; echo
    case "$f" in *nvidia*) ICDS="$ICDS $f" ;; esac
  done
done
for lib in libvulkan.so.1 libGLX_nvidia.so.0 libEGL_nvidia.so.0 libnvidia-glcore.so libnvidia-gpucomp.so libnvidia-glvkspirv.so libX11.so.6 libxcb.so.1; do
  p=$(ldconfig -p 2>/dev/null | grep -m1 -F "$lib" | awk '{print $NF}')
  printf '%-26s %s\n' "$lib" "${p:-ABSENT}"
done
echo "bundled in the wheel: $(ls "$DIR"/site/threepp.libs 2>/dev/null | grep -i vulkan | tr '\n' ' ')"
echo "VK_ICD_FILENAMES=${VK_ICD_FILENAMES:-<unset>} VK_DRIVER_FILES=${VK_DRIVER_FILES:-<unset>} XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-<unset>}"

cd "$DIR"
probe() {   # probe <label>: what the loader reports under the current environment
  echo "--- loader probe: $1 ---"
  VK_LOADER_DEBUG=error,warn python nort_vkprobe.py 2>&1 | tail -40
}
render() {  # render <label>
  echo "=== render: $1 ==="
  python nort_render.py "$DIR/nort.png"
  rc=$?
  echo "render exit code: $rc"
  return $rc
}

probe "environment as given"
render "environment as given" && exit 0

# The loader found no ICD with a headless surface. Name each NVIDIA manifest
# explicitly, which is what a loader that does not search that directory needs.
for icd in $ICDS; do
  export VK_ICD_FILENAMES="$icd" VK_DRIVER_FILES="$icd"
  probe "VK_ICD_FILENAMES=$icd"
  render "VK_ICD_FILENAMES=$icd" && { echo "WORKED WITH: VK_ICD_FILENAMES=$icd"; exit 0; }
done
exit 1
JOB
chmod +x "$DIR/nort_job.sh"

# ------------------------------------------------------------ the wheel ---
echo "=== wheel (login node $(hostname)) ==="
module purge >/dev/null 2>&1 || true
module load "$PYMOD"
if [ -n "$THREEPP_ENV" ]; then
  echo "source build: $THREEPP_ENV"
  python -m pip install --quiet --upgrade --target "$DIR/site_np" numpy 2>&1 | grep -v 'notice' | tail -5 || true
else
  python -m pip install --quiet --upgrade --target "$DIR/site" "$THREEPP_PIP" numpy 2>&1 | grep -v 'notice' | tail -5 || true
  PYTHONPATH="$DIR/site" python - <<'PY'
import importlib.metadata as m
print("installed: threepp", m.version("threepp"), "| numpy", m.version("numpy"))
PY
fi

# ----------------------------------------------------------------- srun ---
echo "=== srun (GPUQ, constraint '${CONSTRAINT:-<any>}'); waits in the queue ==="
rm -f "$DIR/nort.png"
set +e
# shellcheck disable=SC2086
srun --account="$ACCOUNT" --partition=GPUQ --gres=gpu:1 ${CONSTRAINT:+--constraint=$CONSTRAINT} \
     --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:15:00 --job-name=nort \
     --input=none "$DIR/nort_job.sh" "$DIR" "$PYMOD" "$THREEPP_ENV"
rc=$?
set -e

if [ -s "$DIR/nort.png" ]; then
  echo "picture: $DIR/nort.png ($(stat -c%s "$DIR/nort.png") bytes)"
  if [ "${EMIT_PNG:-0}" = "1" ]; then
    echo "-----BEGIN NORT_PNG-----"
    base64 -w 76 "$DIR/nort.png"
    echo "-----END NORT_PNG-----"
  fi
fi
echo "vulkan no-RT check done, exit $rc"
exit $rc
