"""Fancy water in Python: the Vulkan FFT ocean, no boat required.

threepp's `Ocean` is the 3-cascade Phillips/FFT-displaced water surface from the
C++ showcase, exposed as a one-liner. Add it to a Scene and render with the
deferred Vulkan renderer — the renderer recognises the displaced mesh and runs
the FFT / displacement / foam pipeline every frame, with transmission and
whitecaps. There is no hero object: the adaptive-density warp here just follows
the orbit target (a world coordinate), the same field the showcase points at a
boat.

    python vulkan_ocean.py

Drag to orbit, scroll to zoom. Needs a Vulkan build (-DTHREEPP_WITH_VULKAN=ON)
and a display.
"""
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import threepp as tp

if not tp.HAS_VULKAN:
    print("This threepp build has no Vulkan backend "
          "(configure with -DTHREEPP_WITH_VULKAN=ON).")
    sys.exit(0)


# --------------------------------------------------------------------------- #
#  Procedural HDR sky (numpy -> Radiance .hdr -> RGBELoader) — self-contained,
#  so the example needs no downloaded environment map. (Same approach as
#  pbr_showcase.py.) The whole sphere is lit: the lower hemisphere fades to a
#  muted haze rather than a dark ground, so the ocean's horizon and its grazing
#  reflections never drop to black.
# --------------------------------------------------------------------------- #
SUN_DIR = np.array([0.45, 0.45, 0.80])
SUN_DIR = SUN_DIR / np.linalg.norm(SUN_DIR)


def _encode_rgbe(rgb):
    rgb = np.maximum(np.asarray(rgb, np.float64), 0.0)
    m = rgb.max(axis=2)
    mask = m >= 1e-32
    safe = np.where(mask, m, 1.0)
    mant, exp = np.frexp(safe)
    scale = np.where(mask, mant * 256.0 / safe, 0.0)
    out = np.zeros(rgb.shape[:2] + (4,), np.uint8)
    for c in range(3):
        out[..., c] = np.clip(rgb[..., c] * scale, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(mask, np.clip(exp + 128, 0, 255), 0).astype(np.uint8)
    return out


def make_sky_hdr(path, W=2048, H=1024):
    j = np.arange(H).reshape(H, 1)
    i = np.arange(W).reshape(1, W)
    theta = (j / H) * math.pi
    phi = (i / W) * 2 * math.pi - math.pi
    y = np.broadcast_to(np.cos(theta), (H, W))  # +1 zenith .. -1 nadir
    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)
    z = sin_t * np.sin(phi)

    horizon = np.array([0.55, 0.62, 0.78])
    zenith = np.array([0.05, 0.16, 0.42])
    haze = np.array([0.30, 0.34, 0.40])  # muted lower hemisphere, NOT black

    # Both hemispheres meet at the horizon colour, so there's no dark band where
    # the ocean meets the sky: above fades horizon -> zenith, below fades
    # horizon -> haze.
    up = np.clip(y, 0.0, 1.0)[..., None] ** 0.35
    down = np.clip(-y, 0.0, 1.0)[..., None] ** 0.6
    sky = np.where(y[..., None] >= 0.0,
                   horizon * (1.0 - up) + zenith * up,
                   horizon * (1.0 - down) + haze * down)

    # warm glow band straddling the horizon line (symmetric in y)
    glow = np.exp(-(y * y) / (2 * 0.0045))[..., None]
    sky = sky + glow * np.array([1.0, 0.7, 0.45]) * 0.5

    # sun: smooth gaussian core + wide halo (no hard disc -> no stair-stepping)
    d = np.stack([x, y, z], axis=-1)
    ang = np.arccos(np.clip((d * SUN_DIR).sum(-1), -1.0, 1.0))
    core = np.exp(-(ang / math.radians(1.5)) ** 2)
    halo = np.exp(-(ang / math.radians(11.0)) ** 2)
    sky = sky + (core * 60.0 + halo * 4.5)[..., None] * np.array([1.0, 0.95, 0.85])

    rgbe = _encode_rgbe(sky)
    if rgbe[0, 0, 0] == 2 and rgbe[0, 0, 1] == 2 and rgbe[0, 0, 2] < 128:
        rgbe[0, 0, 0] = 3
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (H, W))
        f.write(rgbe.tobytes())
    return path


# --------------------------------------------------------------------------- #
#  Scene
# --------------------------------------------------------------------------- #
canvas = tp.Canvas("threepp — Vulkan Ocean", width=1280, height=720, vsync=False)
renderer = tp.VulkanRenderer(canvas)
renderer.tone_mapping = tp.ToneMapping.ACESFilmic
renderer.tone_mapping_exposure = 0.7

# Optional in-window control panel (skipped on a build without imgui).
ui = tp.ImguiContext(canvas, renderer) if tp.HAS_IMGUI else None

hdr_path = os.path.join(tempfile.gettempdir(), "threepp_ocean_sky.hdr")
env = tp.RGBELoader().load(make_sky_hdr(hdr_path))

scene = tp.Scene()
scene.background = env   # sky shows wherever rays miss the water
scene.environment = env  # image-based lighting on the water

SIZE = 1000.0

# Dark sand floor under the water so refraction reads.
floor_mat = tp.MeshStandardMaterial()
floor_mat.color = 0x050505
floor_mat.roughness = 1.0
floor = tp.Mesh(tp.PlaneGeometry(SIZE, SIZE), floor_mat)
floor.rotate_x(-math.pi / 2)
floor.position.y = -5.0
scene.add(floor)

# The whole "fancy water" in one line.
ocean = tp.Ocean(size=SIZE, wind_speed=10.0, choppiness=0.55)
scene.add(ocean)

sun = tp.DirectionalLight(0xfff2d8, 2.0)
sun.position.set(2, 1, 2)
scene.add(sun)

camera = tp.PerspectiveCamera(50, canvas.aspect(), 0.1, 2000)
camera.position.set(0, 12, -40)
controls = tp.OrbitControls(camera, canvas)
controls.target = tp.Vector3(0, 0, 0)
controls.enable_damping = True
controls.max_distance = 400.0


def on_resize(w, h):
    camera.aspect = w / max(h, 1)
    camera.update_projection_matrix()
    renderer.set_size(w, h)


canvas.on_window_resize(on_resize)


# Live wave knobs. waveScale and choppiness are applied per-frame in the
# displace shader, so they update instantly. Wind speed/direction seed the
# Phillips spectrum once at build time, so they are NOT live-tunable (same as
# the C++ showcase) — left out on purpose.
wave = {"scale": ocean.params.wave_scale, "choppiness": ocean.params.choppiness}


def draw_ui():
    tp.imgui.set_next_window_pos(10, 10)
    tp.imgui.set_next_window_size(300, 0)
    tp.imgui.begin("Ocean")
    changed, wave["scale"] = tp.imgui.slider_float("wave scale", wave["scale"], 0.0, 3.0)
    if changed:
        ocean.params.wave_scale = wave["scale"]
    changed, wave["choppiness"] = tp.imgui.slider_float("choppiness", wave["choppiness"], 0.0, 1.0)
    if changed:
        ocean.params.choppiness = wave["choppiness"]
    tp.imgui.separator()
    tp.imgui.text("wind is fixed at build time (spectrum seed)")
    tp.imgui.text("drag = orbit, scroll = zoom")
    tp.imgui.text(f"{tp.imgui.get_framerate():.0f} fps")
    tp.imgui.end()


def animate():
    # Don't orbit while interacting with the panel.
    if ui is not None:
        controls.enabled = not ui.want_capture_mouse
    controls.update()
    # Pack vertex density toward where the camera is looking. The focus is just
    # a world coordinate — the same warp the showcase points at a boat.
    ocean.warp_toward(controls.target.x, controls.target.z, 0.3)
    renderer.render(scene, camera)
    if ui is not None:
        ui.render(draw_ui)


canvas.animate(animate)
