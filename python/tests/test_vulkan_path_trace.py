"""The path-trace feature of the Vulkan renderer (path_trace_begin / path_trace_read /
path_trace) and the capture module's path_trace helper.

A furnace (a grey sphere under a white environment comes back at its albedo), the
session API, and the capture helper's encoding. Skipped without the Vulkan backend or
without ray tracing (THREEPP_VULKAN_NO_RT=1)."""

import numpy as np
import pytest

import threepp as tp

pytestmark = pytest.mark.skipif(not tp.HAS_VULKAN, reason="built without the Vulkan backend")

W, H = 320, 240


@pytest.fixture(scope="module")
def vk_canvas():
    return tp.Canvas("vk-path-trace", width=W, height=H, headless=True, vsync=False)


@pytest.fixture(scope="module")
def vk_renderer(vk_canvas):
    r = tp.VulkanRenderer(vk_canvas)
    r.auto_lod = False
    r.auto_exposure = False
    r.tone_mapping = tp.ToneMapping.NoToneMapping
    r.tone_mapping_exposure = 1.0
    return r


def constant_env(r, g, b, w=64, h=32):
    data = np.ones((h, w, 3), dtype=np.float32)
    data[..., 0] = r
    data[..., 1] = g
    data[..., 2] = b
    tex = tp.float_texture(data)
    tex.mapping = tp.Mapping.EquirectangularReflection
    return tex


def furnace_scene():
    scene = tp.Scene()
    env = constant_env(1.0, 1.0, 1.0)
    scene.environment = env
    scene.background = env
    mat = tp.MeshStandardMaterial()
    mat.color = tp.Color(0.5, 0.5, 0.5)
    mat.roughness = 1.0
    mat.metalness = 0.0
    scene.add(tp.Mesh(tp.SphereGeometry(1.0, 48, 32), mat))
    cam = tp.PerspectiveCamera(50, W / H, 0.1, 100)
    cam.position.set(0, 0, 4)
    cam.look_at(0, 0, 0)
    return scene, cam


def skip_without_ray_tracing(renderer):
    if not renderer.ray_tracing:
        pytest.skip("needs ray tracing (ray query + acceleration structures)")


def test_path_trace_begin_refuses_without_ray_tracing(vk_renderer):
    if vk_renderer.ray_tracing:
        pytest.skip("this device traces rays")
    with pytest.raises(RuntimeError):
        vk_renderer.path_trace_begin()


def test_furnace(vk_renderer):
    skip_without_ray_tracing(vk_renderer)
    scene, cam = furnace_scene()
    vk_renderer.render(scene, cam)
    img = vk_renderer.path_trace(scene, cam, samples=64, samples_per_frame=8)
    assert img.shape == (H, W, 3) and img.dtype == np.float32
    sky = img[:24, :24].mean(axis=(0, 1))
    assert np.allclose(sky, 1.0, atol=1e-3), sky
    centre = img[H // 2 - 12:H // 2 + 12, W // 2 - 12:W // 2 + 12]
    lum = centre @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    assert 0.46 < lum.mean() < 0.60, lum.mean()
    assert np.abs(lum - lum.mean()).mean() < 0.05
    assert not vk_renderer.path_trace_active


def test_session_counts_samples(vk_renderer):
    skip_without_ray_tracing(vk_renderer)
    scene, cam = furnace_scene()
    vk_renderer.path_trace_begin(samples_per_frame=2)
    assert vk_renderer.path_trace_active
    assert vk_renderer.path_trace_samples == 0
    # The Python render() draws flush_frames (default 3) frames per call, each
    # adding samples_per_frame: count per call, not per frame.
    vk_renderer.render(scene, cam)
    per_call = vk_renderer.path_trace_samples
    assert per_call > 0 and per_call % 2 == 0
    for _ in range(2):
        vk_renderer.render(scene, cam)
    assert vk_renderer.path_trace_samples == 3 * per_call
    img = vk_renderer.path_trace_read()
    assert img.shape == (H, W, 3)
    vk_renderer.path_trace_stop()
    assert not vk_renderer.path_trace_active
    vk_renderer.render(scene, cam)
    assert vk_renderer.path_trace_samples == 3 * per_call


def test_capture_path_trace(vk_renderer):
    skip_without_ray_tracing(vk_renderer)
    from threepp import capture

    scene, cam = furnace_scene()
    frame = capture.path_trace(scene, cam, samples=16, samples_per_frame=8, backend=vk_renderer)
    assert frame.image.shape == (H, W, 3) and frame.image.dtype == np.uint8
    assert frame.hdr.shape == (H, W, 3)
    assert frame.frames >= 1
    # sRGB of linear 1.0 is 255; of ~0.5 is ~188
    assert int(frame.image[4, 4, 0]) == 255
    assert 170 < int(frame.image[H // 2, W // 2, 0]) < 205


def test_path_trace_own_extent(vk_renderer):
    """A picture of its own extent, larger than the canvas: traced band by band, counted in whole passes."""
    skip_without_ray_tracing(vk_renderer)
    from threepp import capture

    scene, cam = furnace_scene()
    frame = capture.path_trace(scene, cam, samples=16, samples_per_frame=8, backend=vk_renderer,
                               trace_size=(2 * W, 2 * H))
    assert frame.image.shape == (2 * H, 2 * W, 3)
    assert frame.hdr.shape == (2 * H, 2 * W, 3)
    # The sphere at its albedo (0.5 + the 4% lobe), as at the canvas's extent.
    assert 0.46 < float(frame.hdr[H, W].mean()) < 0.62
