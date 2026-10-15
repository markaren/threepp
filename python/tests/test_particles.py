"""ParticleField (weather fields) + the float/HDR texture that carries a sky.

The field class exists on every build, but only the Vulkan backend draws it, so
the render assertions skip on a GL-only build (or without a Vulkan GPU) the same
way test_vulkan.py does.
"""
import math

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


def test_billboard_soft_distance_defaults_off_and_round_trips():
    """BillboardRepr.soft_distance: 0 (off) until asked for, then metres."""
    field = rain_field()
    assert field.billboard_repr.soft_distance == 0.0
    field.billboard_repr.soft_distance = 0.8
    assert field.billboard_repr.soft_distance == pytest.approx(0.8)


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


@pytest.mark.skipif(not tp.HAS_VULKAN, reason="built without the Vulkan backend")
def test_headless_soft_billboard_fades_into_the_ground():
    """One big puff centred on a ground plane: with soft_distance off the depth
    test cuts it along a line, with it on the rows just above that line lose
    most of the sprite."""
    canvas = tp.Canvas("particles-soft", width=W, height=H, headless=True, vsync=False)
    renderer = tp.VulkanRenderer(canvas)
    renderer.starfield = 0.0
    renderer.auto_exposure = False

    scene = tp.Scene()
    sun = tp.DirectionalLight(tp.Color(1, 1, 1), 2.0)
    sun.position.set(3, 8, 5)
    scene.add(sun)
    ground_mat = tp.MeshStandardMaterial()
    ground_mat.color = tp.Color(0.06, 0.06, 0.06)
    ground = tp.Mesh(tp.PlaneGeometry(400, 400), ground_mat)
    ground.rotate_x(-math.pi / 2)
    scene.add(ground)

    cfg = tp.ParticleField.Config()
    cfg.capacity = 1
    cfg.ownership = tp.ParticleField.Ownership.HostRing
    cfg.w_semantic = tp.ParticleField.WSemantic.Radius
    field = tp.ParticleField.create(cfg)
    field.frustum_culled = False
    field.submit(np.array([[0.0, 0.0, 0.0, 1.2]], dtype=np.float32))
    field.set_billboard_repr(tp.Color(1, 0.9, 0.75), tp.Color(1, 0.9, 0.75), 1.0, 1.0)
    b = field.billboard_repr
    b.alpha_over = True
    b.opacity = 0.8
    b.bright_jitter = 0.0
    scene.add(field)

    cam = tp.PerspectiveCamera(50, W / H, 0.1, 300)
    cam.position.set(0, 1.0, 6)
    cam.look_at(0, 0.5, 0)

    def frame(n=10):
        for _ in range(n):
            renderer.render(scene, cam)
        return renderer.read_pixels().astype(np.float32).mean(axis=2)

    b.enabled = False
    none = frame()
    b.enabled = True
    off = frame()
    # Long enough that the rows next to the line (each ~4 cm of sprite at this
    # frame size) are well inside the ramp.
    b.soft_distance = 4.0
    on = frame()

    # The line is where the OFF sprite's contribution, down the centre column,
    # drops away; just above it the ON frame keeps little of the sprite.
    w = off.shape[1]
    col = slice(w // 2 - 2, w // 2 + 3)
    d_off = np.abs(off[:, col] - none[:, col]).mean(axis=1)
    d_on = np.abs(on[:, col] - none[:, col]).mean(axis=1)
    top = int(np.argmax(d_off > 0.5 * d_off.max()))
    line = top + int(np.argmax(d_off[top:] < 0.25 * d_off.max()))
    assert line > top + 5, "the OFF frame cuts the sprite along a line"
    assert d_off[line - 3:line - 1].mean() > 0.5 * d_off.max()
    assert d_on[line - 3:line - 1].mean() < 0.3 * d_off[line - 3:line - 1].mean()
