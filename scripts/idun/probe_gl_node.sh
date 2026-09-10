#!/usr/bin/env bash
# Can an IDUN GPU node do HARDWARE OpenGL, and by which route?
#
# Companion to probe_gpu_node.sh (which settled Vulkan: the NVIDIA ICD is
# there and enumerates the H100, but the driver exposes no VK_KHR_ray_query,
# so threepp's Vulkan backend cannot run). That leaves OpenGL -- a GL 3.3 core
# renderer with no ray tracing anywhere in it -- as the only way this cluster
# renders pixels. The question this script answers is whether GL on a compute
# node reaches the GPU, or only the CPU.
#
# Run it INSIDE an interactive GPU allocation, e.g.
#   srun --account=<acct> --partition=GPUQ --gres=gpu:1 --constraint=h100 \
#        --time=00:30:00 --pty bash
#   bash scripts/idun/probe_gl_node.sh 2>&1 | tee gl_probe_$(hostname).txt
#
# It tests seven candidate routes and prints ONE VERDICT LINE PER ROUTE at the
# end, each labelled HARDWARE, SOFTWARE or NO:
#   A  EGL device platform      EGL_EXT_platform_device, no X server at all
#   B  EGL surfaceless (Mesa)   EGL_PLATFORM_SURFACELESS_MESA
#   C  EGL surfaceless + zink   Mesa's GL-on-Vulkan over the NVIDIA ICD
#   D  Xvfb + GLX               the "OpenGL with the X server" idea
#   E  Xvfb + GLX + zink        same display, GL routed through Vulkan
#   F  VirtualGL                its EGL back end needs no 3D X server
#   G  real Xorg + fake monitor the only route to NVIDIA server-side GLX
#   H  OSMesa / llvmpipe        software, the floor
#
# READ-ONLY. It installs nothing, loads no kernel module, and touches nothing
# outside the current directory except one private Xvfb on an unused display
# (its socket lives in /tmp/.X11-unix like any X server) which is killed on
# exit. Every probe is a query; nothing is configured.
#
# Needs python3 for the ctypes probes. If none is on PATH it tries the module
# system (Python/3.10.8-GCCcore-12.2.0). Everything else degrades gracefully.

set -u

section() { printf '\n=== %s ===\n' "$*"; }
note()    { printf '    %s\n' "$*"; }

# ---------------------------------------------------------------- scratch ---
TMPD=""
for cand in "$PWD/.gl_probe.$$" "${TMPDIR:-/tmp}/gl_probe.$$"; do
  if mkdir -p "$cand" 2>/dev/null; then TMPD="$cand"; break; fi
done
if [ -z "$TMPD" ]; then echo "cannot create a scratch directory; aborting"; exit 2; fi
XVFB_PID=""
cleanup() {
  [ -n "$XVFB_PID" ] && kill "$XVFB_PID" 2>/dev/null
  rm -rf "$TMPD" 2>/dev/null
  return 0
}
trap cleanup EXIT INT TERM

TO=""
command -v timeout >/dev/null 2>&1 && TO="timeout 90"

PY_OUT=""
run_py() {                       # run_py <cmd...>  -> echoes output, sets PY_OUT
  PY_OUT="$("$@" 2>&1)"
  printf '%s\n' "$PY_OUT"
}
getres() {                       # getres <tag> <key> -> value from RESULT:tag:key=
  printf '%s\n' "$PY_OUT" | sed -n "s/^RESULT:$1:$2=//p" | head -1
}
classify() {                     # classify "<GL_RENDERER>" -> HARDWARE|SOFTWARE|UNKNOWN|NONE
  case "$1" in
    "")                                   echo NONE ;;
    *NVIDIA*|*Tesla*|*Quadro*|*GeForce*)  echo HARDWARE ;;
    *llvmpipe*|*softpipe*|*swrast*|*SWR*|*"Software Rasterizer"*) echo SOFTWARE ;;
    *)                                    echo UNKNOWN ;;
  esac
}
have() { command -v "$1" >/dev/null 2>&1; }
findlib() {                      # findlib <soname> -> path or empty
  local p
  p=$(ldconfig -p 2>/dev/null | grep -m1 -F "$1" | awk '{print $NF}')
  [ -z "$p" ] && p=$(ls /usr/lib64/"$1"* /usr/lib/x86_64-linux-gnu/"$1"* 2>/dev/null | head -1)
  printf '%s' "${p:-}"
}

# ------------------------------------------------------------------ python ---
PY=""
for c in python3 python; do have "$c" && { PY=$(command -v "$c"); break; }; done
if [ -z "$PY" ]; then
  for lm in /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh /etc/profile.d/modules.sh; do
    [ -r "$lm" ] && . "$lm" 2>/dev/null
  done
  if type module >/dev/null 2>&1; then
    module load Python/3.10.8-GCCcore-12.2.0 >/dev/null 2>&1 || \
    module load Python >/dev/null 2>&1 || true
    have python3 && PY=$(command -v python3)
  fi
fi

section "host / job"
hostname
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-<none: not inside a job>}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-?}  GRES=${SLURM_JOB_GRES:-?}"
grep -E '^(PRETTY_NAME|VERSION_ID)=' /etc/os-release 2>/dev/null
uname -r
echo "cores=$(nproc 2>/dev/null || echo '?')  (sets the ceiling if we end up on llvmpipe)"
free -g 2>/dev/null | head -2
echo "user=$(id -un) uid=$(id -u)  (root would be uid 0; a real Xorg needs it)"
echo "DISPLAY=${DISPLAY:-<unset>}  WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-<unset>}"
echo "python3: ${PY:-<none - the ctypes probes below will be skipped>}"
[ -n "$PY" ] && "$PY" -V 2>&1

section "GPU, driver, MIG"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,compute_mode,display_active,display_mode \
           --format=csv 2>&1 | head -12
nvidia-smi -L 2>&1 | head -12
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
MIG_ON=""
if nvidia-smi -q 2>/dev/null | grep -q 'MIG Mode' ; then
  nvidia-smi -q 2>/dev/null | grep -A2 -m2 'MIG Mode'
  nvidia-smi -q 2>/dev/null | grep -A2 -m1 'MIG Mode' | grep -qE 'Current *: *Enabled' && MIG_ON=yes
fi
[ -n "$MIG_ON" ] && note "MIG appears ENABLED. MIG instances do not support OpenGL or Vulkan;" \
                 && note "every graphics route below is expected to fail while it is on."
echo "-- device nodes --"
ls -l /dev/nvidia* 2>/dev/null | head -12 || echo "no /dev/nvidia*"
ls -l /dev/dri 2>/dev/null || echo "no /dev/dri (no DRM render node; Mesa's surfaceless/zink path usually needs one)"
lsmod 2>/dev/null | grep -E '^nvidia' | head

section "glvnd + driver GL libraries"
# The front (dispatch) half and the NVIDIA vendor half are separate packages.
# A compute-only driver install ships the Vulkan ICD but not these.
for lib in libGLdispatch.so.0 libOpenGL.so.0 libGLX.so.0 libEGL.so.1 libGL.so.1 \
           libGLX_nvidia.so.0 libEGL_nvidia.so.0 libnvidia-glcore.so libnvidia-eglcore.so \
           libnvidia-glsi.so libnvidia-tls.so libnvidia-glvkspirv.so \
           libGLX_mesa.so.0 libEGL_mesa.so.0 libglapi.so.0 libOSMesa.so.8 libOSMesa.so.6 \
           libX11.so.6 libvulkan.so.1; do
  p=$(findlib "$lib")
  if [ -n "$p" ]; then printf 'FOUND   %-26s -> %s\n' "$lib" "$p"
  else                 printf 'absent  %s\n' "$lib"; fi
done
HAVE_LIBEGL=$(findlib libEGL.so.1)
HAVE_EGL_NV=$(findlib libEGL_nvidia.so.0)
HAVE_GLX_NV=$(findlib libGLX_nvidia.so.0)
HAVE_LIBGL=$(findlib libGL.so.1)
HAVE_OSMESA=$(findlib libOSMesa.so)
HAVE_X11=$(findlib libX11.so.6)

section "vendor manifests (what glvnd and the Vulkan loader actually read)"
for d in /usr/share/glvnd/egl_vendor.d /etc/glvnd/egl_vendor.d \
         /usr/share/vulkan/icd.d /etc/vulkan/icd.d \
         /usr/share/egl/egl_external_platform.d; do
  if [ -d "$d" ]; then echo "-- $d"; ls -l "$d" 2>&1 | sed 's/^/   /'
  else echo "absent $d"; fi
done
EGL_JSON=""
for f in /usr/share/glvnd/egl_vendor.d/10_nvidia.json /etc/glvnd/egl_vendor.d/10_nvidia.json; do
  [ -e "$f" ] && { EGL_JSON="$f"; echo "-- $f"; cat "$f" | sed 's/^/   /'; }
done
MESA_EGL_JSON=""
for f in /usr/share/glvnd/egl_vendor.d/50_mesa.json /etc/glvnd/egl_vendor.d/50_mesa.json; do
  [ -e "$f" ] && MESA_EGL_JSON="$f"
done
echo "nvidia EGL vendor json: ${EGL_JSON:-<absent>}"
echo "mesa   EGL vendor json: ${MESA_EGL_JSON:-<absent>}"
echo "__EGL_VENDOR_LIBRARY_FILENAMES=${__EGL_VENDOR_LIBRARY_FILENAMES:-<unset>}"
echo "__EGL_VENDOR_LIBRARY_DIRS=${__EGL_VENDOR_LIBRARY_DIRS:-<unset>}"
echo "__GLX_VENDOR_LIBRARY_NAME=${__GLX_VENDOR_LIBRARY_NAME:-<unset>}"

section "Mesa: gallium drivers and zink"
DRIDIR=""
for d in /usr/lib64/dri /usr/lib/x86_64-linux-gnu/dri /usr/lib/dri ${LIBGL_DRIVERS_PATH:-}; do
  [ -d "$d" ] && { DRIDIR="$d"; echo "-- $d"; ls "$d" 2>&1 | sed 's/^/   /'; }
done
[ -z "$DRIDIR" ] && echo "no DRI driver directory found (no Mesa gallium drivers on this node)"
HAVE_ZINK=""
HAVE_SWRAST=""
for d in /usr/lib64/dri /usr/lib/x86_64-linux-gnu/dri /usr/lib/dri; do
  [ -e "$d/zink_dri.so" ] && HAVE_ZINK="$d/zink_dri.so"
  ls "$d"/swrast_dri.so "$d"/kms_swrast_dri.so >/dev/null 2>&1 && HAVE_SWRAST="$d"
done
echo "zink_dri.so : ${HAVE_ZINK:-<absent>}   (GL-on-Vulkan; the only hardware route that needs no NVIDIA GL driver)"
echo "swrast      : ${HAVE_SWRAST:-<absent>} (llvmpipe software rasteriser)"
echo "LIBGL_DRIVERS_PATH=${LIBGL_DRIVERS_PATH:-<unset>}"
have rpm && rpm -q mesa-dri-drivers mesa-libGL mesa-libEGL mesa-libOSMesa libglvnd \
                   libglvnd-glx libglvnd-egl libglvnd-opengl 2>&1 | sed 's/^/   /'

section "X server, GL tools, VirtualGL"
for b in Xvfb xvfb-run Xorg X xauth xdpyinfo glxinfo glxgears eglinfo es2gears \
         vglrun vglconnect vglserver_config vncserver apptainer singularity; do
  printf '%-18s %s\n' "$b" "$(command -v "$b" 2>/dev/null || echo MISSING)"
done
echo "-- X server modules that would give hardware GLX under a real Xorg --"
for f in /usr/lib64/xorg/modules/extensions/libglxserver_nvidia.so \
         /usr/lib64/xorg/modules/drivers/nvidia_drv.so; do
  ls -l "$f"* 2>/dev/null || echo "absent $f"
done
ls /usr/lib64/xorg/modules/extensions/ 2>/dev/null | sed 's/^/   /' | head
[ -e /etc/X11/Xwrapper.config ] && { echo "-- /etc/X11/Xwrapper.config"; cat /etc/X11/Xwrapper.config | sed 's/^/   /'; }
HAVE_XVFB=$(command -v Xvfb 2>/dev/null || true)
HAVE_XORG=$(command -v Xorg 2>/dev/null || command -v X 2>/dev/null || true)
HAVE_VGL=$(command -v vglrun 2>/dev/null || true)
HAVE_GLXSERVER_NV=$(ls /usr/lib64/xorg/modules/extensions/libglxserver_nvidia.so* 2>/dev/null | head -1)

section "Apptainer: what --nv would bind"
have apptainer && apptainer --version
have singularity && singularity --version
for c in /etc/apptainer/nvliblist.conf /etc/singularity/nvliblist.conf; do
  if [ -e "$c" ]; then
    echo "-- $c (GL/EGL/manifest entries)"
    grep -iE 'GLX|EGL|glcore|eglcore|glsi|GLdispatch|OpenGL|libGL|glvnd|icd\.d|egl_vendor|json|OSMesa' "$c" \
      | sed 's/^/   /'
    echo "   ($(grep -cvE '^\s*(#|$)' "$c") active entries total)"
  else echo "absent $c"; fi
done
have nvidia-container-cli && echo "nvidia-container-cli present (--nvccli mode available; needs NVIDIA_DRIVER_CAPABILITIES=graphics)"

section "module system: Mesa / VirtualGL / X11 / OSMesa"
if type module >/dev/null 2>&1; then
  module avail 2>&1 | grep -iE 'mesa|osmesa|virtualgl|turbovnc|glu|glew|glfw|xvfb|xorg|libglvnd|vulkan' | head -30 \
    || echo "   (nothing matching)"
else
  echo "no 'module' function in this shell"
fi

# =========================================================================== #
#  PROBE A -- EGL device platform. No X server, no display, no root.          #
# =========================================================================== #
cat > "$TMPD/egl_probe.py" <<'PY'
#!/usr/bin/env python3
"""Does this node give a hardware OpenGL context with no X server?

modes:
  device       EGL_EXT_platform_device        NVIDIA's documented headless path
  surfaceless  EGL_PLATFORM_SURFACELESS_MESA  Mesa's (the zink route uses it)
  default      eglGetDisplay(EGL_DEFAULT_DISPLAY)

Prints human text plus RESULT:<mode>:<key>=<value> lines for the caller.
Exit 0 = a GL context came up, 1 = no context, 2 = no usable EGL at all.
"""
import ctypes
import sys

EGL_VENDOR = 0x3053
EGL_VERSION_STR = 0x3054
EGL_EXTENSIONS = 0x3055
EGL_CLIENT_APIS = 0x308D
EGL_PLATFORM_DEVICE_EXT = 0x313F
EGL_PLATFORM_SURFACELESS_MESA = 0x31DD
EGL_OPENGL_API = 0x30A2
EGL_SURFACE_TYPE = 0x3033
EGL_PBUFFER_BIT = 0x0001
EGL_RENDERABLE_TYPE = 0x3040
EGL_OPENGL_BIT = 0x0008
EGL_BLUE_SIZE = 0x3022
EGL_GREEN_SIZE = 0x3023
EGL_RED_SIZE = 0x3024
EGL_NONE = 0x3038
EGL_HEIGHT = 0x3056
EGL_WIDTH = 0x3057
EGL_CONTEXT_MAJOR_VERSION = 0x3098
EGL_CONTEXT_MINOR_VERSION = 0x30FB
EGL_CONTEXT_OPENGL_PROFILE_MASK = 0x30FD
EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT = 0x00000001
EGL_DRM_DEVICE_FILE_EXT = 0x3233
EGL_DRM_RENDER_NODE_FILE_EXT = 0x3377
EGL_CUDA_DEVICE_NV = 0x323A
GL_VENDOR, GL_RENDERER, GL_VERSION, GL_SLV = 0x1F00, 0x1F01, 0x1F02, 0x8B8C

mode = sys.argv[1] if len(sys.argv) > 1 else "device"


def out(key, val):
    print("RESULT:%s:%s=%s" % (mode, key, val))


def cdll(*names):
    for n in names:
        try:
            return ctypes.CDLL(n), n
        except OSError:
            continue
    return None, None


def s(b):
    return b.decode(errors="replace") if b else ""


egl, eglname = cdll("libEGL.so.1", "libEGL.so")
if egl is None:
    print("no libEGL.so.1 / libEGL.so on this node")
    out("lib", "none")
    sys.exit(2)
print("libEGL: %s" % eglname)

egl.eglQueryString.restype = ctypes.c_char_p
egl.eglQueryString.argtypes = [ctypes.c_void_p, ctypes.c_int]
egl.eglGetProcAddress.restype = ctypes.c_void_p
egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
egl.eglGetError.restype = ctypes.c_int
egl.eglGetDisplay.restype = ctypes.c_void_p
egl.eglGetDisplay.argtypes = [ctypes.c_void_p]
egl.eglInitialize.restype = ctypes.c_uint
egl.eglInitialize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
egl.eglBindAPI.restype = ctypes.c_uint
egl.eglBindAPI.argtypes = [ctypes.c_uint]
egl.eglChooseConfig.restype = ctypes.c_uint
egl.eglChooseConfig.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                                ctypes.POINTER(ctypes.c_int)]
egl.eglCreateContext.restype = ctypes.c_void_p
egl.eglCreateContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                 ctypes.POINTER(ctypes.c_int)]
egl.eglCreatePbufferSurface.restype = ctypes.c_void_p
egl.eglCreatePbufferSurface.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
egl.eglMakeCurrent.restype = ctypes.c_uint
egl.eglMakeCurrent.argtypes = [ctypes.c_void_p] * 4
egl.eglTerminate.argtypes = [ctypes.c_void_p]

client_exts = s(egl.eglQueryString(None, EGL_EXTENSIONS))
print("EGL client extensions: %s" % (client_exts or "<none: not a glvnd/EGL 1.5 loader>"))
out("client_ext_count", len(client_exts.split()))


def proc(name, restype, *argtypes):
    a = egl.eglGetProcAddress(name.encode())
    return ctypes.CFUNCTYPE(restype, *argtypes)(a) if a else None


def gl_strings(tag):
    """A context is current: report who is actually going to draw."""
    gl, glname = cdll("libOpenGL.so.0", "libGL.so.1", "libGL.so")
    getstr = None
    if gl is not None:
        try:
            gl.glGetString.restype = ctypes.c_char_p
            gl.glGetString.argtypes = [ctypes.c_uint]
            getstr = gl.glGetString
        except AttributeError:
            getstr = None
    if getstr is None:
        a = egl.eglGetProcAddress(b"glGetString")
        if a:
            getstr = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_uint)(a)
            glname = "eglGetProcAddress"
    if getstr is None:
        print("    no glGetString reachable (no libOpenGL.so.0 / libGL.so.1)")
        return ""
    vend, rend = s(getstr(GL_VENDOR)), s(getstr(GL_RENDERER))
    ver, slv = s(getstr(GL_VERSION)), s(getstr(GL_SLV))
    print("    GL via %s" % glname)
    print("    GL_VENDOR   : %s" % vend)
    print("    GL_RENDERER : %s" % rend)
    print("    GL_VERSION  : %s" % ver)
    print("    GL_SL       : %s" % slv)
    out("renderer", rend)
    out("gl_vendor", vend)
    out("gl_version", ver)
    return rend


def make_context(dpy, tag):
    """eglBindAPI(OPENGL) + 3.3 core, pbuffer if possible else surfaceless."""
    if not egl.eglBindAPI(EGL_OPENGL_API):
        print("    eglBindAPI(EGL_OPENGL_API) FAILED (0x%x) -- desktop GL not offered here"
              % egl.eglGetError())
        out("bindapi", "no")
        return ""
    out("bindapi", "yes")
    cattrs = (ctypes.c_int * 11)(EGL_SURFACE_TYPE, EGL_PBUFFER_BIT,
                                 EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                                 EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                                 EGL_NONE)
    cfg = ctypes.c_void_p()
    n = ctypes.c_int(0)
    ok = egl.eglChooseConfig(dpy, cattrs, ctypes.byref(cfg), 1, ctypes.byref(n))
    if not ok or n.value == 0:
        # surfaceless-only stacks advertise no pbuffer configs; retry without it
        cattrs2 = (ctypes.c_int * 9)(EGL_RENDERABLE_TYPE, EGL_OPENGL_BIT,
                                     EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8,
                                     EGL_NONE)
        ok = egl.eglChooseConfig(dpy, cattrs2, ctypes.byref(cfg), 1, ctypes.byref(n))
    print("    eglChooseConfig(OPENGL_BIT) -> %d config(s)" % n.value)
    out("configs", n.value)
    if not ok or n.value == 0:
        print("    no EGL config renders desktop OpenGL here (0x%x)" % egl.eglGetError())
        return ""
    ctxattrs = (ctypes.c_int * 7)(EGL_CONTEXT_MAJOR_VERSION, 3,
                                  EGL_CONTEXT_MINOR_VERSION, 3,
                                  EGL_CONTEXT_OPENGL_PROFILE_MASK,
                                  EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
                                  EGL_NONE)
    ctx = egl.eglCreateContext(dpy, cfg, None, ctxattrs)
    if not ctx:
        print("    eglCreateContext(3.3 core) FAILED (0x%x); retrying without a version hint"
              % egl.eglGetError())
        ctx = egl.eglCreateContext(dpy, cfg, None, None)
    if not ctx:
        print("    eglCreateContext FAILED (0x%x)" % egl.eglGetError())
        out("context", "no")
        return ""
    out("context", "yes")
    pattrs = (ctypes.c_int * 5)(EGL_WIDTH, 64, EGL_HEIGHT, 64, EGL_NONE)
    surf = egl.eglCreatePbufferSurface(dpy, cfg, pattrs)
    if surf:
        cur = egl.eglMakeCurrent(dpy, surf, surf, ctx)
        kind = "pbuffer 64x64"
    else:
        cur = egl.eglMakeCurrent(dpy, None, None, ctx)
        kind = "surfaceless (EGL_NO_SURFACE)"
    print("    eglMakeCurrent %s -> %s" % (kind, "OK" if cur else
                                           "FAILED (0x%x)" % egl.eglGetError()))
    out("current", "yes" if cur else "no")
    if not cur:
        return ""
    rend = gl_strings(tag)
    egl.eglMakeCurrent(dpy, None, None, None)
    return rend


def describe(dpy, tag):
    maj, mnr = ctypes.c_int(0), ctypes.c_int(0)
    if not egl.eglInitialize(dpy, ctypes.byref(maj), ctypes.byref(mnr)):
        print("    eglInitialize FAILED (0x%x)" % egl.eglGetError())
        out("init", "no")
        return ""
    out("init", "yes")
    print("    EGL %d.%d  vendor=%s  version=%s" % (
        maj.value, mnr.value,
        s(egl.eglQueryString(dpy, EGL_VENDOR)), s(egl.eglQueryString(dpy, EGL_VERSION_STR))))
    print("    client APIs: %s" % s(egl.eglQueryString(dpy, EGL_CLIENT_APIS)))
    exts = s(egl.eglQueryString(dpy, EGL_EXTENSIONS))
    print("    display extensions: %s" % (exts[:400] + ("..." if len(exts) > 400 else "")))
    rend = make_context(dpy, tag)
    egl.eglTerminate(dpy)
    return rend


rc = 1
if mode == "device":
    qd = proc("eglQueryDevicesEXT", ctypes.c_uint, ctypes.c_int,
              ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int))
    gpd = proc("eglGetPlatformDisplayEXT", ctypes.c_void_p, ctypes.c_uint,
               ctypes.c_void_p, ctypes.c_void_p)
    qds = proc("eglQueryDeviceStringEXT", ctypes.c_char_p, ctypes.c_void_p, ctypes.c_int)
    qda = proc("eglQueryDeviceAttribEXT", ctypes.c_uint, ctypes.c_void_p, ctypes.c_int,
               ctypes.POINTER(ctypes.c_ssize_t))
    print("eglQueryDevicesEXT      : %s" % ("present" if qd else "ABSENT"))
    print("eglGetPlatformDisplayEXT: %s" % ("present" if gpd else "ABSENT"))
    out("platform_device", "yes" if (qd and gpd) else "no")
    if not (qd and gpd):
        print("EGL_EXT_platform_device is not available -> the X-free hardware route is closed")
        sys.exit(1)
    n = ctypes.c_int(0)
    devs = (ctypes.c_void_p * 32)()
    qd(32, devs, ctypes.byref(n))
    print("EGL devices: %d" % n.value)
    out("devices", n.value)
    for i in range(n.value):
        d = devs[i]
        dexts = s(qds(d, EGL_EXTENSIONS)) if qds else ""
        drm = s(qds(d, EGL_DRM_DEVICE_FILE_EXT)) if qds else ""
        node = s(qds(d, EGL_DRM_RENDER_NODE_FILE_EXT)) if qds else ""
        vend = s(qds(d, EGL_VENDOR)) if qds else ""
        cuda = ctypes.c_ssize_t(-1)
        if qda:
            qda(d, EGL_CUDA_DEVICE_NV, ctypes.byref(cuda))
        print("\n[device %d] vendor=%s drm=%s node=%s cuda_index=%s"
              % (i, vend or "?", drm or "-", node or "-",
                 cuda.value if cuda.value >= 0 else "-"))
        print("    device extensions: %s" % (dexts or "<none>"))
        dpy = gpd(EGL_PLATFORM_DEVICE_EXT, d, None)
        if not dpy:
            print("    eglGetPlatformDisplayEXT FAILED (0x%x)" % egl.eglGetError())
            continue
        r = describe(ctypes.c_void_p(dpy), "dev%d" % i)
        if r:
            rc = 0
elif mode == "surfaceless":
    gpd = proc("eglGetPlatformDisplayEXT", ctypes.c_void_p, ctypes.c_uint,
               ctypes.c_void_p, ctypes.c_void_p)
    have = "EGL_MESA_platform_surfaceless" in client_exts
    print("EGL_MESA_platform_surfaceless: %s" % ("advertised" if have else "NOT advertised"))
    out("advertised", "yes" if have else "no")
    if not gpd:
        print("no eglGetPlatformDisplayEXT")
        sys.exit(1)
    dpy = gpd(EGL_PLATFORM_SURFACELESS_MESA, None, None)
    if not dpy:
        print("eglGetPlatformDisplayEXT(SURFACELESS_MESA) FAILED (0x%x)" % egl.eglGetError())
        sys.exit(1)
    if describe(ctypes.c_void_p(dpy), "surfaceless"):
        rc = 0
else:
    dpy = egl.eglGetDisplay(None)   # EGL_DEFAULT_DISPLAY
    print("eglGetDisplay(EGL_DEFAULT_DISPLAY) -> %s" % ("ok" if dpy else "NULL"))
    if not dpy:
        sys.exit(1)
    if describe(ctypes.c_void_p(dpy), "default"):
        rc = 0

sys.exit(rc)
PY

# =========================================================================== #
#  GLX probe -- used against whatever DISPLAY we have, and against Xvfb.      #
# =========================================================================== #
cat > "$TMPD/glx_probe.py" <<'PY'
#!/usr/bin/env python3
"""Against $DISPLAY: which GLX implementation serves it, and what renders?

The GLX *server* vendor string is the whole Xvfb question. Xvfb carries
X.Org's in-tree GLX backed by Mesa (server vendor "SGI"); NVIDIA's server-side
GLX is a separate X module (libglxserver_nvidia.so) that only a real Xorg
running the NVIDIA driver loads, and it reports "NVIDIA Corporation".

Prints RESULT:glx:<key>=<value>. Exit 0 = a GL context came up.
"""
import ctypes
import os
import sys

GLX_VENDOR, GLX_VERSION, GLX_EXTENSIONS = 1, 2, 3
GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT = 0x8010, 0x00000004
GLX_RENDER_TYPE, GLX_RGBA_BIT, GLX_RGBA_TYPE = 0x8011, 0x00000001, 0x8014
GLX_RED_SIZE, GLX_GREEN_SIZE, GLX_BLUE_SIZE = 8, 9, 10
GLX_PBUFFER_HEIGHT, GLX_PBUFFER_WIDTH = 0x8040, 0x8041
GL_VENDOR, GL_RENDERER, GL_VERSION, GL_SLV = 0x1F00, 0x1F01, 0x1F02, 0x8B8C


def out(k, v):
    print("RESULT:glx:%s=%s" % (k, v))


def cdll(*names):
    for n in names:
        try:
            return ctypes.CDLL(n), n
        except OSError:
            continue
    return None, None


def s(b):
    return b.decode(errors="replace") if b else ""


dpyname = os.environ.get("DISPLAY", "")
print("DISPLAY=%s" % (dpyname or "<unset>"))
if not dpyname:
    out("display", "unset")
    sys.exit(1)

x11, x11name = cdll("libX11.so.6", "libX11.so")
if x11 is None:
    print("no libX11.so.6 -- no X client library on this node")
    out("libx11", "none")
    sys.exit(2)
gl, glname = cdll("libGL.so.1", "libGLX.so.0", "libGL.so")
if gl is None:
    print("no libGL.so.1 / libGLX.so.0 -- nothing can speak GLX here")
    out("libgl", "none")
    sys.exit(2)
print("libX11: %s   libGL: %s" % (x11name, glname))

x11.XOpenDisplay.restype = ctypes.c_void_p
x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
x11.XDefaultScreen.restype = ctypes.c_int
x11.XDefaultScreen.argtypes = [ctypes.c_void_p]
dpy = x11.XOpenDisplay(dpyname.encode())
if not dpy:
    print("XOpenDisplay(%s) FAILED" % dpyname)
    out("open", "no")
    sys.exit(1)
out("open", "yes")
scr = x11.XDefaultScreen(dpy)

gl.glXQueryExtension.restype = ctypes.c_int
gl.glXQueryExtension.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
                                 ctypes.POINTER(ctypes.c_int)]
e1, e2 = ctypes.c_int(0), ctypes.c_int(0)
hasglx = gl.glXQueryExtension(dpy, ctypes.byref(e1), ctypes.byref(e2))
print("server speaks GLX: %s" % bool(hasglx))
out("glx", "yes" if hasglx else "no")
if not hasglx:
    sys.exit(1)

gl.glXQueryServerString.restype = ctypes.c_char_p
gl.glXQueryServerString.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
gl.glXGetClientString.restype = ctypes.c_char_p
gl.glXGetClientString.argtypes = [ctypes.c_void_p, ctypes.c_int]
sv = s(gl.glXQueryServerString(dpy, scr, GLX_VENDOR))
sver = s(gl.glXQueryServerString(dpy, scr, GLX_VERSION))
cv = s(gl.glXGetClientString(dpy, GLX_VENDOR))
print("GLX server vendor : %s   (version %s)" % (sv, sver))
print("GLX client vendor : %s" % cv)
out("server_vendor", sv)
out("client_vendor", cv)
sexts = s(gl.glXQueryServerString(dpy, scr, GLX_EXTENSIONS))
print("GLX server extensions: %s" % (sexts[:300] + ("..." if len(sexts) > 300 else "")))
out("nv_glx", "yes" if "NVIDIA" in sv else "no")

gl.glXChooseFBConfig.restype = ctypes.POINTER(ctypes.c_void_p)
gl.glXChooseFBConfig.argtypes = [ctypes.c_void_p, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
gl.glXCreatePbuffer.restype = ctypes.c_ulong
gl.glXCreatePbuffer.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
gl.glXCreateNewContext.restype = ctypes.c_void_p
gl.glXCreateNewContext.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                   ctypes.c_void_p, ctypes.c_int]
gl.glXMakeContextCurrent.restype = ctypes.c_int
gl.glXMakeContextCurrent.argtypes = [ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
                                     ctypes.c_void_p]
gl.glXIsDirect.restype = ctypes.c_int
gl.glXIsDirect.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

attrs = (ctypes.c_int * 11)(GLX_DRAWABLE_TYPE, GLX_PBUFFER_BIT,
                            GLX_RENDER_TYPE, GLX_RGBA_BIT,
                            GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, 0)
n = ctypes.c_int(0)
cfgs = gl.glXChooseFBConfig(dpy, scr, attrs, ctypes.byref(n))
print("glXChooseFBConfig(pbuffer) -> %d config(s)" % n.value)
out("fbconfigs", n.value)
if not cfgs or n.value == 0:
    sys.exit(1)
cfg = cfgs[0]
pattrs = (ctypes.c_int * 5)(GLX_PBUFFER_WIDTH, 64, GLX_PBUFFER_HEIGHT, 64, 0)
pbuf = gl.glXCreatePbuffer(dpy, cfg, pattrs)
ctx = gl.glXCreateNewContext(dpy, cfg, GLX_RGBA_TYPE, None, 1)
if not ctx:
    print("glXCreateNewContext FAILED")
    out("context", "no")
    sys.exit(1)
out("context", "yes")
direct = gl.glXIsDirect(dpy, ctx)
print("direct rendering: %s" % bool(direct))
out("direct", "yes" if direct else "no")
if not gl.glXMakeContextCurrent(dpy, pbuf, pbuf, ctx):
    print("glXMakeContextCurrent FAILED")
    out("current", "no")
    sys.exit(1)
out("current", "yes")
gl.glGetString.restype = ctypes.c_char_p
gl.glGetString.argtypes = [ctypes.c_uint]
vend, rend = s(gl.glGetString(GL_VENDOR)), s(gl.glGetString(GL_RENDERER))
ver, slv = s(gl.glGetString(GL_VERSION)), s(gl.glGetString(GL_SLV))
print("GL_VENDOR   : %s" % vend)
print("GL_RENDERER : %s" % rend)
print("GL_VERSION  : %s" % ver)
print("GL_SL       : %s" % slv)
out("renderer", rend)
out("gl_vendor", vend)
out("gl_version", ver)
# glad's exact precondition: it dlopens libGL.so.1 and takes glXGetProcAddressARB
print("glXGetProcAddressARB in libGL: %s" % hasattr(gl, "glXGetProcAddressARB"))
out("glxgetprocaddressarb", "yes" if hasattr(gl, "glXGetProcAddressARB") else "no")
sys.exit(0)
PY

EGLDEV_DEVICES=""; EGLDEV_RENDERER=""; EGLDEV_PLATFORM=""
SFL_RENDERER=""; ZINK_SFL_RENDERER=""
XVFB_RENDERER=""; XVFB_SERVER_VENDOR=""; XVFB_ZINK_RENDERER=""
NATIVE_RENDERER=""; NATIVE_SERVER_VENDOR=""

if [ -n "$PY" ]; then

  section "PROBE A -- EGL device platform (no X server, no root)"
  note "NVIDIA's documented headless GL path since driver 355. If this reports an"
  note "NVIDIA GL_RENDERER, the cluster can render on the GPU with no display at all."
  run_py $TO "$PY" "$TMPD/egl_probe.py" device
  EGLDEV_PLATFORM=$(getres device platform_device)
  EGLDEV_DEVICES=$(getres device devices)
  EGLDEV_RENDERER=$(getres device renderer)

  section "PROBE B -- EGL surfaceless (Mesa platform)"
  run_py $TO "$PY" "$TMPD/egl_probe.py" surfaceless
  SFL_RENDERER=$(getres surfaceless renderer)

  section "PROBE C -- EGL surfaceless forced to zink (GL on the Vulkan ICD)"
  note "zink needs none of the ray-tracing extensions the H100 driver lacks."
  if [ -n "$HAVE_ZINK" ]; then
    if [ -n "$MESA_EGL_JSON" ]; then
      run_py $TO env MESA_LOADER_DRIVER_OVERRIDE=zink GALLIUM_DRIVER=zink \
        __EGL_VENDOR_LIBRARY_FILENAMES="$MESA_EGL_JSON" \
        "$PY" "$TMPD/egl_probe.py" surfaceless
    else
      run_py $TO env MESA_LOADER_DRIVER_OVERRIDE=zink GALLIUM_DRIVER=zink \
        "$PY" "$TMPD/egl_probe.py" surfaceless
    fi
    ZINK_SFL_RENDERER=$(getres surfaceless renderer)
  else
    echo "zink_dri.so is not installed on this node -- skipped."
    echo "(it lives inside the mesa-dri-drivers package; a container can supply it)"
  fi

  section "PROBE D -- EGL default display"
  run_py $TO "$PY" "$TMPD/egl_probe.py" default

  if [ -n "${DISPLAY:-}" ]; then
    section "PROBE D2 -- GLX against the DISPLAY we were given"
    run_py $TO "$PY" "$TMPD/glx_probe.py"
    NATIVE_RENDERER=$(getres glx renderer)
    NATIVE_SERVER_VENDOR=$(getres glx server_vendor)
  fi

  # ------------------------------------------------------------- Xvfb + GLX --
  section "PROBE E -- Xvfb + GLX (the \"OpenGL with the X server\" idea)"
  if [ -z "$HAVE_XVFB" ]; then
    echo "Xvfb is not installed on this node -- skipped."
    echo "(it is a package, not a privilege: a container can carry it)"
  elif [ -z "$HAVE_X11" ]; then
    echo "no libX11.so.6 client library -- an X server would have no client here."
  else
    DPY=""
    for n in $(seq 90 130); do
      if [ ! -e "/tmp/.X11-unix/X$n" ]; then DPY=":$n"; break; fi
    done
    if [ -z "$DPY" ]; then
      echo "no free display number in :90-:130 -- skipped"
    else
      echo "starting a private Xvfb on $DPY (killed when this script exits)"
      Xvfb "$DPY" -screen 0 1280x1024x24 -nolisten tcp >"$TMPD/xvfb.log" 2>&1 &
      XVFB_PID=$!
      up=""
      for i in $(seq 1 20); do
        if [ -e "/tmp/.X11-unix/X${DPY#:}" ]; then up=yes; break; fi
        kill -0 "$XVFB_PID" 2>/dev/null || break
        sleep 0.5
      done
      if [ -z "$up" ]; then
        echo "Xvfb did not come up:"; sed 's/^/   /' "$TMPD/xvfb.log"
        XVFB_PID=""
      else
        run_py $TO env DISPLAY="$DPY" "$PY" "$TMPD/glx_probe.py"
        XVFB_RENDERER=$(getres glx renderer)
        XVFB_SERVER_VENDOR=$(getres glx server_vendor)
        if have glxinfo; then
          echo "-- glxinfo -B for corroboration --"
          DISPLAY="$DPY" $TO glxinfo -B 2>&1 | head -20 | sed 's/^/   /'
        fi

        section "PROBE F -- Xvfb + GLX forced to zink"
        if [ -n "$HAVE_ZINK" ]; then
          run_py $TO env DISPLAY="$DPY" MESA_LOADER_DRIVER_OVERRIDE=zink \
            GALLIUM_DRIVER=zink LIBGL_KOPPER_DRI2=1 __GLX_VENDOR_LIBRARY_NAME=mesa \
            "$PY" "$TMPD/glx_probe.py"
          XVFB_ZINK_RENDERER=$(getres glx renderer)
        else
          echo "zink_dri.so absent -- skipped"
        fi

        section "PROBE G -- VirtualGL EGL back end under the same Xvfb"
        if [ -n "$HAVE_VGL" ]; then
          for d in egl0 /dev/dri/card0 /dev/dri/renderD128; do
            echo "-- vglrun -d $d"
            DISPLAY="$DPY" VGL_DISPLAY="$d" $TO vglrun -d "$d" \
              "$PY" "$TMPD/glx_probe.py" 2>&1 | sed -n '1,24p' | sed 's/^/   /'
          done
        else
          echo "vglrun not installed -- skipped"
        fi

        section "PROBE H -- software floor (LIBGL_ALWAYS_SOFTWARE=1)"
        run_py $TO env DISPLAY="$DPY" LIBGL_ALWAYS_SOFTWARE=1 \
          "$PY" "$TMPD/glx_probe.py"
        SOFT_RENDERER=$(getres glx renderer)
      fi
    fi
  fi
else
  section "ctypes probes SKIPPED"
  echo "no python3 found and the module system did not provide one."
  echo "Load a Python module and re-run; every decisive answer below needs it."
fi
SOFT_RENDERER=${SOFT_RENDERER:-}

# ------------------------------------------------------- optional container --
if [ -n "${THREEPP_SIF:-}" ] && [ -e "${THREEPP_SIF:-}" ] && have apptainer; then
  section "same questions INSIDE ${THREEPP_SIF} under --nv"
  binds=""
  for d in /usr/share/glvnd/egl_vendor.d /usr/share/vulkan/icd.d /etc/vulkan/icd.d; do
    [ -d "$d" ] && binds="$binds --bind $d"
  done
  # shellcheck disable=SC2086
  apptainer exec --nv $binds --bind "$TMPD" "$THREEPP_SIF" bash -lc '
    echo "-- LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
    ls /.singularity.d/libs 2>/dev/null | head -30
    ls /usr/share/glvnd/egl_vendor.d /usr/share/vulkan/icd.d 2>&1
    ls /usr/lib64/dri 2>/dev/null
    for p in /opt/python/cp310-cp310/bin/python3 python3; do command -v $p >/dev/null && PY=$p && break; done
    echo "container python: ${PY:-none}"
    [ -n "${PY:-}" ] && $PY '"$TMPD"'/egl_probe.py device' 2>&1 | sed 's/^/   /'
else
  section "container check"
  echo "set THREEPP_SIF=/path/to/threepp-build.sif to repeat probe A inside the"
  echo "container under 'apptainer exec --nv' (that is where the module runs)."
fi

# =========================================================================== #
#                                  VERDICTS                                   #
# =========================================================================== #
section "VERDICT -- one line per route"

v() { printf '  %-30s %s\n' "$1" "$2"; }

[ -n "$MIG_ON" ] && echo "  (MIG IS ENABLED on this GPU. MIG instances support neither OpenGL nor Vulkan," \
                 && echo "   so treat every NO below as 'not while MIG is on' and re-probe on a whole GPU.)"

# A. EGL device platform
if [ -z "$PY" ]; then                      v "A EGL device platform"  "UNKNOWN  -- no python3 to run the probe"
elif [ -z "$HAVE_LIBEGL" ]; then           v "A EGL device platform"  "NO       -- no libEGL.so.1 on this node"
elif [ "$EGLDEV_PLATFORM" != "yes" ]; then v "A EGL device platform"  "NO       -- EGL_EXT_platform_device absent"
elif [ -n "$EGLDEV_RENDERER" ]; then       v "A EGL device platform"  "$(classify "$EGLDEV_RENDERER") -- GL_RENDERER=$EGLDEV_RENDERER"
elif [ "${EGLDEV_DEVICES:-0}" != "0" ] && [ -n "${EGLDEV_DEVICES:-}" ]; then
                                           v "A EGL device platform"  "PARTIAL  -- $EGLDEV_DEVICES EGL device(s) but no GL context (see PROBE A)"
else                                       v "A EGL device platform"  "NO       -- eglQueryDevicesEXT enumerated nothing"
fi

# B. EGL surfaceless
if [ -z "$PY" ];                     then v "B EGL surfaceless (Mesa)" "UNKNOWN  -- no python3"
elif [ -n "$SFL_RENDERER" ];         then v "B EGL surfaceless (Mesa)" "$(classify "$SFL_RENDERER") -- GL_RENDERER=$SFL_RENDERER"
else                                      v "B EGL surfaceless (Mesa)" "NO       -- no context (see PROBE B)"
fi

# C. zink over the working Vulkan ICD, no X
if [ -z "$HAVE_ZINK" ];              then v "C EGL surfaceless + zink"  "NO       -- zink_dri.so not installed (add mesa-dri-drivers to the SIF)"
elif [ -n "$ZINK_SFL_RENDERER" ];    then v "C EGL surfaceless + zink"  "$(classify "$ZINK_SFL_RENDERER") -- GL_RENDERER=$ZINK_SFL_RENDERER"
else                                      v "C EGL surfaceless + zink"  "NO       -- zink present but no context (see PROBE C)"
fi

# D. Xvfb + GLX
if [ -z "$HAVE_XVFB" ];              then v "D Xvfb + GLX"              "NO       -- Xvfb not installed here (a container can carry it)"
elif [ -n "$XVFB_RENDERER" ];        then v "D Xvfb + GLX"              "$(classify "$XVFB_RENDERER") -- server GLX vendor='$XVFB_SERVER_VENDOR', GL_RENDERER=$XVFB_RENDERER"
else                                      v "D Xvfb + GLX"              "NO       -- no GLX context under Xvfb (see PROBE E)"
fi

# E. Xvfb + zink
if [ -z "$HAVE_ZINK" ];              then v "E Xvfb + GLX + zink"       "NO       -- zink_dri.so not installed"
elif [ -n "$XVFB_ZINK_RENDERER" ];   then v "E Xvfb + GLX + zink"       "$(classify "$XVFB_ZINK_RENDERER") -- GL_RENDERER=$XVFB_ZINK_RENDERER"
else                                      v "E Xvfb + GLX + zink"       "NO       -- no context (see PROBE F)"
fi

# F. VirtualGL
if [ -z "$HAVE_VGL" ];               then v "F VirtualGL"               "NO       -- vglrun not installed and not a module"
else                                      v "F VirtualGL"               "PRESENT  -- see PROBE G; its EGL back end is route A wearing a hat"
fi

# G. real Xorg + fake monitor
if [ "$(id -u)" = "0" ] && [ -n "$HAVE_GLXSERVER_NV" ]; then
                                          v "G real Xorg + fake monitor" "POSSIBLE -- root AND libglxserver_nvidia.so present"
elif [ -n "$HAVE_GLXSERVER_NV" ];    then v "G real Xorg + fake monitor" "NO       -- libglxserver_nvidia.so is here but you are not root; admin action"
elif [ -n "$HAVE_XORG" ];            then v "G real Xorg + fake monitor" "NO       -- Xorg exists but the NVIDIA GLX server module does not"
else                                      v "G real Xorg + fake monitor" "NO       -- no Xorg, no NVIDIA GLX server module, no root"
fi

# H. software floor
if [ -n "$SOFT_RENDERER" ];          then v "H OSMesa / llvmpipe"       "SOFTWARE -- GL_RENDERER=$SOFT_RENDERER"
elif [ -n "$HAVE_OSMESA" ];          then v "H OSMesa / llvmpipe"       "SOFTWARE -- libOSMesa present at $HAVE_OSMESA"
elif [ -n "$HAVE_SWRAST" ];          then v "H OSMesa / llvmpipe"       "SOFTWARE -- swrast_dri.so in $HAVE_SWRAST"
else                                      v "H OSMesa / llvmpipe"       "NO       -- no software rasteriser installed either"
fi

section "VERDICT -- what threepp can do on this node"
if [ -n "$XVFB_RENDERER" ]; then
  echo "  TODAY, unpatched: a GL Canvas works under the Xvfb display above and renders"
  echo "                    with '$XVFB_RENDERER'."
  echo "                    Set DISPLAY and run; no code change, no rebuild."
else
  echo "  TODAY, unpatched: NOTHING. threepp's GL canvas needs a DISPLAY (Canvas.cpp:183-189"
  echo "                    only picks the GLFW Null platform, whose GL context is OSMesa,"
  echo "                    and Canvas.cpp:325 sets GLFW_OPENGL_FORWARD_COMPAT which OSMesa"
  echo "                    rejects outright). It throws either way."
fi
if [ -n "$EGLDEV_RENDERER" ] || [ -n "$ZINK_SFL_RENDERER" ]; then
  echo "  WITH the EGL work: a threepp-owned EGL context (~200 lines beside Canvas, plus a"
  echo "                    gladLoadGLLoader(eglGetProcAddress) overload -- glad currently"
  echo "                    resolves through glXGetProcAddressARB on libGL.so.1) would reach"
  echo "                    ${EGLDEV_RENDERER:-$ZINK_SFL_RENDERER}."
fi
echo
echo "  Reminder: the GL renderer is GL 3.3 core with zero GL 4.x calls, so any of the"
echo "  above is functionally sufficient. The only question these verdicts answer is"
echo "  whether the pixels are drawn by the GPU or by the CPU."

echo
echo "gl probe done"
