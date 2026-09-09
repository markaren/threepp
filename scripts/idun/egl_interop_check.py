#!/usr/bin/env python3
"""Does CUDA<->OpenGL interop work against an EGL context on a datacenter GPU?

    python egl_interop_check.py

That one question decides whether the warp demos can render on a compute node
at full speed or have to push every vertex through host memory. The demos write
marching-cubes triangles straight into the renderer's own vertex buffers
(GLRenderer.gl_buffer_id + wp.RegisteredGLBuffer), which needs CUDA to be able
to import a GL buffer -- and every published example of that does it against a
GLX context on a workstation card, not an EGL context on an H100.

Checks, in order, each fatal to the next:
  1. an EGL context on the GPU (not llvmpipe)
  2. warp sees the same GPU, through the same CUDA driver
  3. gl_buffer_id returns a buffer -- it is None until the first render
  4. wp.RegisteredGLBuffer accepts it
  5. map/unmap round-trips, and a kernel's writes are actually visible

Exit 0 = zero-copy is available. Exit 1 = it is not, and the demos need the
host-memory upload path. Exit 2 = something more basic is wrong.
"""
import os
import sys

sys.path.insert(0, "/cluster/work/laht/threepp/python")
for extra in (os.environ.get("THREEPP_SRC"),):
    if extra:
        sys.path.insert(0, os.path.join(extra, "python"))

import threepp as tp

# Imported at module scope, and the kernel defined there too: warp compiles a
# kernel against the module it is declared in, and a kernel defined inside a
# function is the kind of thing that works on one warp version and not the next.
# The flag keeps a missing warp a clean message rather than an import traceback.
try:
    import warp as wp

    _HAVE_WARP = True
except ImportError:
    _HAVE_WARP = False

if _HAVE_WARP:

    @wp.kernel
    def fill_vec3(out: wp.array(dtype=wp.vec3)):
        i = wp.tid()
        out[i] = wp.vec3(float(i), 2.0 * float(i), 3.0 * float(i))


W = H = 256


def main() -> int:
    if not tp.HAS_EGL or not tp.egl_available():
        print("FAIL: no EGL here (HAS_EGL=%s, egl_available=%s)"
              % (tp.HAS_EGL, tp.egl_available()))
        return 2

    # 1. a hardware context. --allow-software exists only to dry-run the rest of
    #    this script on a machine without a GPU: interop will then fail at step 4,
    #    which is the correct answer there and still exercises the code path.
    allow_software = "--allow-software" in sys.argv
    try:
        ctx = tp.EglContext(width=W, height=H, require_hardware=not allow_software)
    except RuntimeError as e:
        print("FAIL: %s" % e)
        return 2
    print("context  : %s | %s | cuda device %d"
          % (ctx.renderer, ctx.platform, ctx.cuda_device_index))

    # 2. warp on the same GPU. Warp shares torch's / the driver's primary
    #    context, so this is normally the same device -- but on a multi-GPU node
    #    EGL and CUDA are different enumerations, so say what we got.
    if not _HAVE_WARP:
        print("FAIL: warp is not installed here. On a login node:")
        print("      python -m pip install --user warp-lang")
        return 2
    wp.init()
    device = wp.get_device("cuda:0")
    print("warp     : %s (%s), is_primary=%s"
          % (wp.__version__, device, getattr(device, "is_primary", "?")))
    if ctx.cuda_device_index not in (-1, 0):
        print("NOTE: the EGL context is on CUDA device %d but warp is on cuda:0. "
              "On a single-GPU allocation these are the same card; on a shared "
              "node they need not be." % ctx.cuda_device_index)

    # 3. a geometry with position + normal, rendered once so the GL buffers exist
    renderer = tp.GLRenderer(ctx, W, H)
    scene = tp.Scene()
    camera = tp.PerspectiveCamera(45, 1.0, 0.1, 100)
    camera.position.set(0, 0, 4)
    camera.look_at(0, 0, 0)
    scene.add(tp.AmbientLight(0xffffff, 1.0))

    geometry = tp.SphereGeometry(1.0, 16, 12)
    mesh = tp.Mesh(geometry, tp.MeshStandardMaterial())
    scene.add(mesh)

    renderer.render(scene, camera)  # gl_buffer_id is None before this

    if not hasattr(renderer, "gl_buffer_id"):
        print("FAIL: this build has no GLRenderer.gl_buffer_id")
        return 1
    pid = renderer.gl_buffer_id(geometry, "position")
    nid = renderer.gl_buffer_id(geometry, "normal")
    print("buffers  : position=%s normal=%s" % (pid, nid))
    if pid is None or nid is None:
        print("FAIL: gl_buffer_id returned None after a render")
        return 1

    # 4 + 5. register, map, write from a kernel, read back
    try:
        reg = wp.RegisteredGLBuffer(int(pid), device, wp.RegisteredGLBuffer.WRITE_DISCARD)
    except Exception as e:
        print("FAIL: wp.RegisteredGLBuffer refused the buffer: %s: %s"
              % (type(e).__name__, e))
        print("      -> zero-copy is unavailable; the demos need a host-memory upload path.")
        return 1

    # get_attribute returns an (N, item_size) float32 copy, so N is shape[0].
    pos = geometry.get_attribute("position")
    if pos is None:
        print("FAIL: the geometry has no position attribute")
        return 2
    n = int(pos.shape[0])
    print("vertices : %d" % n)

    try:
        arr = reg.map(dtype=wp.vec3, shape=(n,))
        wp.launch(fill_vec3, dim=n, device=device, inputs=[arr])
        wp.synchronize_device(device)
        got = arr.numpy()[:4].copy()
        reg.unmap()
    except Exception as e:
        print("FAIL: map/launch/unmap raised %s: %s" % (type(e).__name__, e))
        print("      -> zero-copy is unavailable; the demos need a host-memory upload path.")
        return 1

    # ORDERING RULE, and it is not optional: cuGraphicsUnregisterResource needs
    # the GL context current, but Python tears objects down in whatever order it
    # likes at exit. Leave this to the garbage collector and EglContext is
    # destroyed first, so warp's destructor reports
    #   CUDA error 219: invalid OpenGL or DirectX context
    # Harmless at exit, and pure noise in a job log. Drop the registration here,
    # while the context is still alive and current.
    del reg

    expected = [[float(i), 2.0 * i, 3.0 * i] for i in range(4)]
    ok = all(abs(got[i][j] - expected[i][j]) < 1e-4 for i in range(4) for j in range(3))
    print("readback : %s" % got.tolist())
    if not ok:
        print("FAIL: the kernel's writes did not land in the GL buffer (expected %s)" % expected)
        return 1

    print("\nPASS: CUDA<->GL interop works against an EGL context on %s." % ctx.renderer)
    print("      The warp demos can render zero-copy on this node.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
