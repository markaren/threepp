"""ParticleField (weather fields) + the float/HDR texture that carries a sky.

The field class exists on every build, but only the Vulkan backend draws it, so
the render assertions skip on a GL-only build (or without a Vulkan GPU) the same
way test_vulkan.py does.
"""
import numpy as np
import pytest

import threepp as tp

W, H = 256, 160


def rain_field(capacity=4096):
    cfg = tp.ParticleField.Config()
    cfg.capacity = capacity
    cfg.ownership = tp.ParticleField.Ownership.Renderer
    cfg.w_semantic = tp.ParticleField.WSemantic.Radius
    cfg.uniform_radius = 0.013
    field = tp.ParticleField.create(cfg)

    e = tp.ParticleField.EmitterParams()
    e.spawn_center = tp.Vector3(0, 6, 0)
    e.spawn_half_extent = tp.Vector3(8, 0.3, 8)
    e.velocity = tp.Vector3(0, -9, 0)
    e.speed_spread = 0.35
    e.lifetime = 1.9
    e.size = 0.013
    field.set_emitter(e)
    field.set_emitter_time(1.0, 1.0 / 60.0)
    field.set_billboard_repr(0xb8c9e6, 0x99abc7, 0.07, 0.30)
    field.billboard_repr.stretch_seconds = 0.024
    return field


def test_renderer_field_emitter_and_live_count():
    """A Renderer field is live at capacity, and its half of the API is the
    emitter half — submit() on it raises rather than silently doing nothing."""
    field = rain_field()
    assert field.capacity == 4096 and field.live_count == 4096
    assert field.emitter.velocity.y == pytest.approx(-9.0)
    with pytest.raises(ValueError):
        field.submit(np.zeros((4, 4), dtype=np.float32))


def test_float_texture_roundtrip():
    sky = np.full((16, 32, 3), 0.25, dtype=np.float32)
    tex = tp.float_texture(sky)
    tex.update_float(np.full((16, 32, 4), 0.4, dtype=np.float32))
    assert tex.color_space == tp.ColorSpace.Linear


@pytest.mark.skipif(not tp.HAS_VULKAN, reason="built without the Vulkan backend")
def test_headless_render_with_rain_under_a_float_sky():
    canvas = tp.Canvas("particles-test", width=W, height=H, headless=True, vsync=False)
    renderer = tp.VulkanRenderer(canvas)
    renderer.starfield = 0.0

    scene = tp.Scene()
    scene.environment = tp.float_texture(np.full((32, 64, 3), 0.3, dtype=np.float32))
    scene.add(rain_field())
    cam = tp.PerspectiveCamera(55, W / H, 0.1, 100)
    cam.position.set(0, 1.5, 6)
    cam.look_at(0, 1.5, 0)

    renderer.render(scene, cam)
    img = renderer.read_pixels()
    # Canvas size is a REQUEST — the swapchain may hand back a wider surface
    # (Windows enforces a minimum window width), so the height and the channel
    # count are what is actually promised here.
    assert img.ndim == 3 and img.shape[0] == H and img.shape[2] == 3
    assert str(img.dtype) == "uint8"
