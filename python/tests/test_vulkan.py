"""Vulkan deferred renderer + G-buffer AOV readback.

Skips entirely on a GL-only build (threepp.HAS_VULKAN is False). On a Vulkan
build these require a Vulkan-capable GPU.
"""
import numpy as np
import pytest

import threepp as tp

pytestmark = pytest.mark.skipif(not tp.HAS_VULKAN, reason="built without the Vulkan backend")

W, H = 128, 96


@pytest.fixture(scope="module")
def vk_renderer():
    canvas = tp.Canvas("vk-test", width=W, height=H, headless=True, vsync=False)
    return tp.VulkanRenderer(canvas)


def make_scene():
    scene = tp.Scene()
    scene.background = 0x202830
    mat = tp.MeshStandardMaterial()
    mat.color = 0xff5533
    scene.add(tp.Mesh(tp.BoxGeometry(), mat))
    scene.add(tp.HemisphereLight(0xffffff, 0x404040, 1.0))
    sun = tp.DirectionalLight(0xffffff, 3.0)
    sun.position.set(3, 5, 2)
    scene.add(sun)
    cam = tp.PerspectiveCamera(55, W / H, 0.1, 100)
    cam.position.set(1.5, 1.5, 3.0)
    cam.look_at(0, 0, 0)
    return scene, cam


def test_rgb_render(vk_renderer):
    scene, cam = make_scene()
    img = vk_renderer.render_aov(scene, cam, "rgb")
    assert img.shape == (H, W, 3) and str(img.dtype) == "uint8"
    assert int(img.max()) > int(img.min()), "shaded render is flat"


def test_aov_shapes(vk_renderer):
    scene, cam = make_scene()
    out = vk_renderer.render_aovs(scene, cam, ["rgb", "normals", "segmentation", "albedo"])
    assert set(out) == {"rgb", "normals", "segmentation", "albedo"}
    for arr in out.values():
        assert arr.shape == (H, W, 3) and str(arr.dtype) == "uint8"


def test_aovs_are_distinct(vk_renderer):
    scene, cam = make_scene()
    rgb = vk_renderer.render_aov(scene, cam, "rgb")
    normals = vk_renderer.render_aov(scene, cam, "normals")
    seg = vk_renderer.render_aov(scene, cam, "segmentation")
    assert not np.array_equal(rgb, normals)
    assert not np.array_equal(rgb, seg)
    assert not np.array_equal(normals, seg)


def test_segmentation_has_distinct_regions(vk_renderer):
    scene, cam = make_scene()
    seg = vk_renderer.render_aov(scene, cam, "segmentation")
    colors = {tuple(c) for c in seg.reshape(-1, 3).tolist()}
    # at least background (black) + the box's hashed id colour
    assert len(colors) >= 2


def test_convenience_aov_accessors(vk_renderer):
    scene, cam = make_scene()
    assert vk_renderer.read_normals(scene, cam).shape == (H, W, 3)
    assert vk_renderer.read_segmentation(scene, cam).shape == (H, W, 3)
    assert vk_renderer.read_albedo(scene, cam).shape == (H, W, 3)


def test_unknown_aov_raises(vk_renderer):
    scene, cam = make_scene()
    with pytest.raises(ValueError):
        vk_renderer.render_aov(scene, cam, "not_an_aov")


def test_depth_is_metric(vk_renderer):
    # A fronto-parallel wall filling the frame: view-space depth equals the
    # camera distance, so read_depth must return that distance.
    scene = tp.Scene()
    scene.add(tp.AmbientLight(0xffffff, 1.0))
    scene.add(tp.Mesh(tp.PlaneGeometry(60, 60), tp.MeshStandardMaterial()))
    cam = tp.PerspectiveCamera(50, W / H, 0.1, 100)
    cy, cx = H // 2, W // 2
    for dist in (3.0, 7.0, 15.0):
        cam.position.set(0, 0, dist)
        cam.look_at(0, 0, 0)
        depth = vk_renderer.read_depth(scene, cam)
        assert depth.shape == (H, W) and str(depth.dtype) == "float32"
        assert depth[cy, cx] == pytest.approx(dist, abs=0.05)


@pytest.mark.skipif(not tp.HAS_IMGUI, reason="built without imgui")
def test_imgui_over_vulkan():
    # The ImGui overlay records into the Vulkan deferred frame after the scene.
    # (Kept here, not in test_imgui.py, so a GL and a Vulkan ImGui context are
    # never alive at the same time — two live ImGui contexts crash.)
    canvas = tp.Canvas("imgui-vk", width=320, height=240, headless=True, vsync=False)
    renderer = tp.VulkanRenderer(canvas)
    ui = tp.ImguiContext(canvas, renderer)  # Vulkan backend

    scene = tp.Scene()
    scene.add(tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial()))
    scene.add(tp.AmbientLight(0xffffff, 1.0))
    cam = tp.PerspectiveCamera(60, 320 / 240, 0.1, 100)
    cam.position.set(1.5, 1.5, 3)
    cam.look_at(0, 0, 0)

    def draw():
        tp.imgui.set_next_window_pos(8, 8)
        tp.imgui.set_next_window_size(180, 120)
        tp.imgui.begin("Vulkan UI")
        tp.imgui.text("hello")
        tp.imgui.slider_float("v", 0.5, 0.0, 1.0)
        tp.imgui.end()

    for _ in range(3):  # flush MAILBOX
        canvas.animate_once(lambda: (renderer.render(scene, cam), ui.render(draw)))
    img = renderer.read_pixels()
    panel = img[8:128, 8:188]
    assert int(panel.max()) - int(panel.min()) > 40, "Vulkan UI panel did not draw"


@pytest.mark.skipif(not tp.HAS_IMGUI, reason="built without imgui")
def test_imgui_vulkan_canvas_needs_renderer():
    canvas = tp.Canvas("imgui-vk2", width=64, height=64, headless=True, vsync=False)
    tp.VulkanRenderer(canvas)
    # On a Vulkan canvas the renderer must be passed (GL backend can't be used).
    with pytest.raises((ValueError, RuntimeError)):
        tp.ImguiContext(canvas)


def test_depth_occlusion(vk_renderer):
    # A box in front of the wall must read nearer at the centre than the wall.
    scene = tp.Scene()
    scene.add(tp.AmbientLight(0xffffff, 1.0))
    scene.add(tp.Mesh(tp.PlaneGeometry(60, 60), tp.MeshStandardMaterial()))  # z=0
    box = tp.Mesh(tp.BoxGeometry(2, 2, 2), tp.MeshStandardMaterial())
    box.position.set(0, 0, 3)  # front face at z=4
    scene.add(box)
    cam = tp.PerspectiveCamera(50, W / H, 0.1, 100)
    cam.position.set(0, 0, 8)  # 8 from wall, 4 from box front
    cam.look_at(0, 0, 0)
    depth = vk_renderer.read_depth(scene, cam)
    assert depth[H // 2, W // 2] == pytest.approx(4.0, abs=0.1)   # box front
    assert depth[5, 5] == pytest.approx(8.0, abs=0.1)             # wall behind
