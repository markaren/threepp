"""Prove hardware OpenGL on a display-less GPU node, in one allocation.

    python idun_egl_smoke.py [--width 256] [--height 256] [--save frame.png]

Renders one lit sphere through an EGL context -- no X server, no Wayland, no
window of any kind -- and then checks that the GPU, and not a software
rasteriser, is what drew it.

Written for NTNU's Idun cluster, where threepp's Vulkan renderer cannot run at
all: the driver exposes the KHR ray tracing extensions but not VK_KHR_ray_query,
which every Vulkan shading pass traces through. OpenGL through EGL is the way a
compute node renders, and scripts/idun/probe_gl_node.sh measured
GL_RENDERER = "NVIDIA H100 80GB HBM3/PCIe/SSE2" there with no display server.

Exit status is 0 only if every check passes, so it works as a job-script gate.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

SOFTWARE_NAMES = ("llvmpipe", "lavapipe", "softpipe", "swrast", "Mesa Offscreen")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--device", type=int, default=-1,
                    help="EGL device index; -1 (default) picks the first hardware one")
    ap.add_argument("--allow-software", action="store_true",
                    help="do not require a GPU -- for checking the pipeline on a laptop")
    ap.add_argument("--save", metavar="PNG", help="also write the frame to this path")
    args = ap.parse_args()

    W, H = args.width, args.height

    if not tp.HAS_EGL:
        print("FAIL: this threepp was built without the EGL path (not Linux?)")
        return 2
    if not tp.egl_available():
        print("FAIL: no usable EGL here -- libEGL did not load, or it reports no device.")
        print("      On a login node this is expected: there is no GPU driver there.")
        return 2

    print(f"EGL devices: {tp.EglContext.device_count()}")

    # require_hardware=True is the point of the exercise: a context that comes
    # up on llvmpipe renders correctly, succeeds, and wastes the allocation.
    # Caught rather than raised, so a job script gets a verdict and an exit
    # code instead of a traceback.
    try:
        ctx = tp.EglContext(width=W, height=H, device=args.device,
                            require_hardware=not args.allow_software)
    except RuntimeError as e:
        print(f"FAIL: {e}")
        if not args.allow_software:
            print("      Re-run with --allow-software to render on the CPU and check the "
                  "rest of the pipeline.")
        return 2

    print(f"platform     : {ctx.platform} (egl device {ctx.device_index}, "
          f"cuda device {ctx.cuda_device_index})")
    print(f"EGL          : {ctx.egl_vendor} | {ctx.egl_version}")
    print(f"GL_VENDOR    : {ctx.vendor}")
    print(f"GL_RENDERER  : {ctx.renderer}")
    print(f"GL_VERSION   : {ctx.version}")
    print(f"drawable     : {ctx.drawable_width}x{ctx.drawable_height}")
    print(f"hardware     : {ctx.hardware_accelerated}")

    renderer = tp.GLRenderer(ctx, W, H)
    renderer.set_clear_color(0x101820)

    scene = tp.Scene()
    scene.add(tp.HemisphereLight(0xffffff, 0x404048, 1.0))
    sun = tp.DirectionalLight(0xffffff, 2.5)
    sun.position.set(5, 8, 5)
    scene.add(sun)

    material = tp.MeshStandardMaterial()
    material.color = 0xff8800
    material.roughness = 0.35
    scene.add(tp.Mesh(tp.SphereGeometry(1.0, 48, 32), material))

    camera = tp.PerspectiveCamera(45, W / H, 0.1, 100)
    camera.position.set(0, 0, 4)
    camera.look_at(0, 0, 0)

    renderer.render(scene, camera)
    img = renderer.read_pixels()

    if args.save:
        renderer.save_frame(args.save)
        print(f"wrote {args.save}")

    # --- the checks, each killing a specific failure -------------------------
    failures = []

    if img.shape != (H, W, 3):
        failures.append(f"read_pixels returned {img.shape}, expected {(H, W, 3)}")

    # A 1x1 pbuffer under a larger renderer returns undefined bytes with no GL
    # error. The binding guards against it, so this is belt and braces.
    if ctx.drawable_width < W or ctx.drawable_height < H:
        failures.append(f"drawable {ctx.drawable_width}x{ctx.drawable_height} is smaller "
                        f"than the renderer {W}x{H}: readback would be undefined")

    if not args.allow_software:
        if not ctx.hardware_accelerated:
            failures.append(f"GL_RENDERER {ctx.renderer!r} is a software rasteriser")
        if any(name in ctx.renderer for name in SOFTWARE_NAMES):
            failures.append(f"GL_RENDERER {ctx.renderer!r} names a software rasteriser")
        if ctx.platform != "device":
            failures.append(f"platform is {ctx.platform!r}, expected 'device' "
                            "(surfaceless still reaches the GPU, but the device platform "
                            "is what binds a NAMED GPU on a multi-GPU node)")

    # The frame: an orange sphere on a near-black ground. Clear-colour-only
    # would mean the context came up and drew nothing at all.
    lit = (img.mean(axis=2) > 40)
    lit_pct = 100.0 * float(lit.sum()) / float(W * H)
    r_over_b = float(img[:, :, 0].sum()) / max(float(img[:, :, 2].sum()), 1.0)
    print(f"pixels       : max={int(img.max())} lit={lit_pct:.1f}% r/b={r_over_b:.2f}")

    if int(img.max()) < 80:
        failures.append("the frame is essentially black: nothing was drawn")
    if not (5.0 < lit_pct < 80.0):
        failures.append(f"lit fraction {lit_pct:.1f}% is not a sphere on a dark ground")
    if r_over_b < 1.5:
        failures.append(f"red/blue ratio {r_over_b:.2f}: the orange material did not shade")

    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1

    print(f"\nPASS: threepp rendered on {ctx.renderer} with no display server.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
