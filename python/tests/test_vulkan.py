"""Vulkan deferred renderer + G-buffer AOV readback.

Skips entirely on a GL-only build (threepp.HAS_VULKAN is False). On a Vulkan
build these require a Vulkan-capable GPU.
"""
import math

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
    # A renderer is required on every backend; omitting it must fail (pybind
    # raises TypeError for the missing arg, our own guard raises ValueError).
    with pytest.raises((TypeError, ValueError, RuntimeError)):
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


def _ocean_scene():
    scene = tp.Scene()
    scene.add(tp.AmbientLight(0xffffff, 1.0))
    sun = tp.DirectionalLight(0xffffff, 2.0)
    sun.position.set(2, 1, 2)
    scene.add(sun)
    floor_mat = tp.MeshStandardMaterial()
    floor_mat.color = 0x050505
    floor = tp.Mesh(tp.PlaneGeometry(200, 200), floor_mat)
    floor.rotate_x(-math.pi / 2)
    floor.position.y = -5.0
    scene.add(floor)
    # Small grid + FFT so the per-frame displace/BLAS stays cheap in the test.
    ocean = tp.Ocean(size=200.0, resolution=128, fft_size=256, wind_speed=10.0)
    scene.add(ocean)
    cam = tp.PerspectiveCamera(55, W / H, 0.1, 400)
    cam.position.set(0, 8, 30)
    cam.look_at(0, 0, 0)
    return scene, cam, ocean


def test_ocean_is_displaced_mesh_with_knobs():
    ocean = tp.Ocean(size=500.0)
    assert isinstance(ocean, tp.DisplacedMesh)
    assert isinstance(ocean, tp.Mesh)
    # inherited Object3D API works across threepp's virtual base
    ocean.position.set(1, 2, 3)
    assert ocean.position.x == 1
    # params + warp are mutable sub-objects
    assert ocean.params.tile_size_0 == pytest.approx(500.0)
    ocean.params.wind_speed = 6.5
    assert ocean.params.wind_speed == pytest.approx(6.5)
    ocean.warp_toward(10.0, -4.0, 0.2)
    assert ocean.warp.half_range > 0 and ocean.warp.center_x == pytest.approx(10.0)
    # foam API is callable
    ocean.add_foam_disturbance(0.0, 0.0, 2.0, 1.0)
    ocean.clear_foam_disturbances()


def test_ocean_renders_and_displaces(vk_renderer):
    scene, cam, ocean = _ocean_scene()
    for _ in range(8):  # let the wave field evolve + the BLAS displace
        ocean.warp_toward(0.0, 0.0, 0.3)
        vk_renderer.render(scene, cam)
    img = vk_renderer.read_pixels()
    assert img.shape == (H, W, 3) and str(img.dtype) == "uint8"
    assert int(img.max()) > int(img.min()), "ocean render is flat"
    # The CPU height readback is filled by the Vulkan render -> waves vary across
    # the tile (validates the FFT/displace pipeline ran end to end).
    coords = np.linspace(-80, 80, 6)
    heights = np.array([ocean.sample_height(float(x), float(z)) for x in coords for z in coords])
    assert np.all(np.isfinite(heights))
    assert np.ptp(heights) > 1e-3, "wave height field looks flat"


def test_depthsensor_pathtraced(vk_renderer):
    # The backend-neutral tp.DepthSensor.scan must work on Vulkan (path-traced
    # through the renderer's TLAS) and reconstruct world-space heights — the same
    # call signature as on GL. Z-up scene: ground top at z=0, a 0.30 m cube on it.
    scene = tp.Scene()
    scene.add(tp.HemisphereLight(0xffffff, 0x404040, 1.0))
    ground = tp.Mesh(tp.BoxGeometry(20, 20, 1.0), tp.MeshStandardMaterial())
    ground.position.set(0, 0, -0.5)                       # top at z=0
    scene.add(ground)
    box = tp.Mesh(tp.BoxGeometry(0.6, 0.6, 0.3), tp.MeshStandardMaterial())
    box.position.set(0, 0, 0.15)                          # top at z=0.30
    scene.add(box)
    cam = tp.PerspectiveCamera(55, W / H, 0.05, 100)
    cam.up.set(0, 0, 1)
    cam.position.set(3, 3, 3)
    cam.look_at(0, 0, 0)
    vk_renderer.render(scene, cam)                        # build the TLAS (required before scan)

    sensor = tp.DepthSensor(fov_y=70, width=96, height=96, near=0.05, far=8.0)
    sensor.range_noise = 0.0                              # exact reconstruction for the assert
    sensor.position.set(0, 0, 3.0)                        # identity rot -> looks straight down (Z-up)
    pts = sensor.scan(vk_renderer, scene)

    assert pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] > 1000
    r = np.hypot(pts[:, 0], pts[:, 1])
    z = pts[:, 2]
    on_box = r < 0.25                                     # squarely over the cube top
    on_ground = (r > 1.0) & (r < 4.0)                     # well off the cube
    assert on_box.sum() > 5 and z[on_box].mean() == pytest.approx(0.30, abs=0.02)
    assert on_ground.sum() > 50 and z[on_ground].mean() == pytest.approx(0.0, abs=0.02)
