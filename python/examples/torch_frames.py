"""Zero-copy frames out: the Vulkan renderer's images as live torch tensors.

The reverse of the vertex-interop demos. The renderer exports its colour and
G-buffer images as CUDA-importable buffers and fills them with copies recorded
in the frame's own command buffer; torch wraps those buffers once and reads
every subsequent frame straight off the GPU. No staging buffer, no host memcpy,
no vkDeviceWaitIdle — one frame fence and the pixels are there.

What it does per frame: a Sobel edge detector over the depth tensor, entirely in
torch on the device, plus a coverage statistic from the instance-id tensor. The
edge map is written out as a PNG every N frames so there is something to look
at, and both paths are timed:

    python torch_frames.py                 # live demo + per-frame ms
    python torch_frames.py --bench         # torch path vs the host readback

Needs an NVIDIA GPU, a CUDA build of torch, and a Vulkan build of threepp. Says
so and exits cleanly when any of those is missing — the fallback everyone has is
renderer.read_aovs_typed(...), which the benchmark measures side by side.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

W, H = 640, 480
FRAMES = 120
PNG_EVERY = 40

if not tp.HAS_VULKAN:
    print("This threepp build has no Vulkan backend "
          "(configure with -DTHREEPP_WITH_VULKAN=ON).")
    sys.exit(0)

try:
    import torch
except ImportError:
    print("torch is not installed. Zero-copy frames out is a CUDA path; install "
          "a CUDA build of torch, or use renderer.read_aovs_typed() for the "
          "host readback that works everywhere.")
    sys.exit(0)

if not torch.cuda.is_available():
    print(f"torch {torch.__version__} has no CUDA device. Zero-copy frames out "
          "is NVIDIA-only; the host fallback is renderer.read_aovs_typed().")
    sys.exit(0)

from threepp.torch_frames import FrameInteropUnavailable, FrameTensors  # noqa: E402


def build_scene():
    scene = tp.Scene()
    scene.background = 0x101820

    specs = [(0xff5555, 0.45, tp.BoxGeometry(1.1, 1.1, 1.1)),
             (0x55ff66, 0.30, tp.SphereGeometry(0.7, 32, 16)),
             (0x5599ff, 0.60, tp.TorusKnotGeometry(0.45, 0.16))]
    movers = []
    for i, (color, rough, geo) in enumerate(specs):
        mat = tp.MeshStandardMaterial()
        mat.color = color
        mat.roughness = rough
        mat.metalness = 0.1
        mesh = tp.Mesh(geo, mat)
        mesh.position.x = (i - 1) * 2.0
        scene.add(mesh)
        movers.append(mesh)

    ground_mat = tp.MeshStandardMaterial()
    ground_mat.color = 0x555560
    ground = tp.Mesh(tp.PlaneGeometry(40, 40), ground_mat)
    ground.position.y = -1.0
    ground.rotate_x(-3.14159 / 2)
    scene.add(ground)

    scene.add(tp.HemisphereLight(0xffffff, 0x404048, 1.0))
    sun = tp.DirectionalLight(0xffffff, 3.0)
    sun.position.set(4, 7, 5)
    scene.add(sun)

    cam = tp.PerspectiveCamera(50, W / H, 0.1, 100)
    cam.position.set(0, 2.5, 6.5)
    cam.look_at(0, 0, 0)
    return scene, cam, movers


# ── The per-frame torch work ────────────────────────────────────────────────
# A 3x3 Sobel over the depth tensor. depth is REVERSED-Z NDC (1 = near, 0 = far
# — see enable_frame_interop's layout table), which is exactly what an edge
# filter wants: it is a monotone function of distance, and the gradient is what
# gets thresholded. Nothing here leaves the GPU.
_SOBEL_X = torch.tensor([[1., 0., -1.], [2., 0., -2.], [1., 0., -1.]],
                        device="cuda").view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(2, 3).contiguous()


def depth_edges(depth):
    d = depth.unsqueeze(0).unsqueeze(0)
    gx = torch.nn.functional.conv2d(d, _SOBEL_X, padding=1)
    gy = torch.nn.functional.conv2d(d, _SOBEL_Y, padding=1)
    return torch.sqrt(gx * gx + gy * gy).squeeze()


def save_png(edges, path):
    try:
        from PIL import Image
    except ImportError:
        return False
    e = edges / max(float(edges.max()), 1e-6)
    img = (e.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    Image.fromarray(img, mode="L").save(path)
    return True


def animate(movers, i):
    for k, m in enumerate(movers):
        m.rotation.y = 0.05 * i * (1 + k)
        m.position.y = 0.35 * (1 + k) * __import__("math").sin(0.06 * i + k)


def demo():
    canvas = tp.Canvas("torch frames", width=W, height=H, headless=True, vsync=False)
    renderer = tp.VulkanRenderer(canvas)
    scene, cam, movers = build_scene()
    renderer.render(scene, cam)          # the exports are sized from a real frame

    here = os.path.dirname(os.path.abspath(__file__))
    try:
        frames = FrameTensors(renderer, 0, ("color", "depth", "ids"))
    except FrameInteropUnavailable as e:
        print(f"frame interop unavailable: {e}")
        return
    print(f"armed: {frames}  (bgra={frames.bgra})")

    with frames:
        render_ms = consume_ms = 0.0
        for i in range(FRAMES):
            animate(movers, i)
            t0 = time.perf_counter()
            renderer.render(scene, cam)
            frames.sync()                # ONE frame fence, not device idle
            t1 = time.perf_counter()

            edges = depth_edges(frames.depth)
            covered = float((frames.ids[..., 1] != 0).float().mean())
            # A stat that forces the result back to the host, so the timing
            # below is honest about the whole round trip rather than the launch.
            strength = float(edges.mean())
            torch.cuda.synchronize()
            t2 = time.perf_counter()

            render_ms += (t1 - t0) * 1e3
            consume_ms += (t2 - t1) * 1e3
            if (i + 1) % PNG_EVERY == 0:
                out = os.path.join(here, f"torch_frames_edges_{i + 1:04d}.png")
                wrote = save_png(edges, out)
                print(f"frame {i + 1:4d}  geometry {covered * 100:5.1f}% of pixels  "
                      f"edge {strength:.4f}  render+sync {render_ms / (i + 1):6.2f} ms  "
                      f"torch {consume_ms / (i + 1):5.2f} ms"
                      + (f"  -> {os.path.basename(out)}" if wrote else "  (no Pillow)"))
        print(f"\n{FRAMES} frames: render+sync {render_ms / FRAMES:.2f} ms/frame, "
              f"torch consume {consume_ms / FRAMES:.2f} ms/frame")


# ── Benchmark: the same three channels, both ways ───────────────────────────
def bench(n=60):
    canvas = tp.Canvas("torch frames bench", width=W, height=H, headless=True, vsync=False)
    renderer = tp.VulkanRenderer(canvas)
    scene, cam, movers = build_scene()
    renderer.render(scene, cam)

    # Baseline: render() alone, so the readback columns below are attributable.
    for i in range(10):
        animate(movers, i)
        renderer.render(scene, cam)
    t0 = time.perf_counter()
    for i in range(n):
        animate(movers, i)
        renderer.render(scene, cam)
    base_ms = (time.perf_counter() - t0) * 1e3 / n

    # Host path: the same three channels, read back to numpy every frame.
    # Colour comes from the scene capture (the same post-TAA / pre-overlay
    # picture the Color channel exports), depth and ids from the raw AOV reads
    # — each of which drains the device once, which is most of what this costs.
    renderer.scene_capture = True
    renderer.render(scene, cam)
    t0 = time.perf_counter()
    for i in range(n):
        animate(movers, i)
        renderer.render(scene, cam)
        color = renderer.read_scene_pixels()
        d = renderer.read_gbuffer_aov_raw("depth", 0)
        ids = renderer.read_gbuffer_aov_raw("ids", 0)
        assert color.size and d is not None and ids is not None
    host_ms = (time.perf_counter() - t0) * 1e3 / n
    renderer.scene_capture = False

    # Torch path: same frames, same three channels, zero copies.
    try:
        frames = FrameTensors(renderer, 0, ("color", "depth", "ids"))
    except FrameInteropUnavailable as e:
        print(f"frame interop unavailable: {e}")
        return
    with frames:
        for i in range(10):
            animate(movers, i)
            renderer.render(scene, cam)
            frames.sync()
        t0 = time.perf_counter()
        for i in range(n):
            animate(movers, i)
            renderer.render(scene, cam)
            frames.sync()
            # Touch all three, the way a consumer would, and make the device
            # actually finish before the clock is read.
            _ = frames.color[..., :3]
            _ = frames.depth
            _ = frames.ids
            torch.cuda.synchronize()
        torch_ms = (time.perf_counter() - t0) * 1e3 / n

    print(f"\n{W}x{H}, {n} frames, channels = color + depth + ids")
    print(f"  render() alone            {base_ms:7.2f} ms/frame")
    print(f"  + host readback           {host_ms:7.2f} ms/frame   "
          f"(+{host_ms - base_ms:.2f})")
    print(f"  + zero-copy torch tensors {torch_ms:7.2f} ms/frame   "
          f"(+{torch_ms - base_ms:.2f})")
    if torch_ms < host_ms:
        print(f"  torch path is {host_ms / torch_ms:.2f}x faster end to end")
    # The per-frame copy is a handful of device-to-device blits, so its cost
    # routinely lands inside the frame-time noise. Say that rather than
    # printing a speedup ratio computed from two numbers that differ by less
    # than the measurement can resolve.
    delta = torch_ms - base_ms
    if delta > 0.25:
        print(f"  the interop copies cost {delta:.2f} ms/frame, "
              f"{(host_ms - base_ms) / delta:.1f}x less than the host readback")
    else:
        print("  the interop copies cost less than the frame-time noise floor "
              f"({delta:+.2f} ms/frame measured)")


if __name__ == "__main__":
    if "--bench" in sys.argv:
        bench()
    else:
        demo()
