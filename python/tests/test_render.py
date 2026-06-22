import os

import threepp as tp


def _camera():
    cam = tp.PerspectiveCamera(60, 200 / 150, 0.1, 100)
    cam.position.set(0, 1.5, 4)
    cam.look_at(0, 0, 0)
    return cam


def test_read_pixels_shape_and_dtype(renderer, lit_scene):
    lit_scene.add(tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial()))
    renderer.render(lit_scene, _camera())
    img = renderer.read_pixels()
    assert img.shape == (150, 200, 3)
    assert str(img.dtype) == "uint8"


def test_render_draws_something(renderer, lit_scene):
    mat = tp.MeshStandardMaterial()
    mat.color = 0x44aa88
    lit_scene.add(tp.Mesh(tp.SphereGeometry(1.0), mat))
    renderer.render(lit_scene, _camera())
    img = renderer.read_pixels()
    assert int(img.max()) > int(img.min()), "rendered image is uniformly flat"


def test_clear_color_affects_background(renderer):
    scene = tp.Scene()
    renderer.set_clear_color(0x000000)
    renderer.render(scene, _camera())
    dark = int(renderer.read_pixels().max())
    renderer.set_clear_color(0xffffff)
    renderer.render(scene, _camera())
    light = int(renderer.read_pixels().min())
    renderer.set_clear_color(0x202830)  # restore session default
    assert light > dark


def test_save_frame(renderer, lit_scene, tmp_path):
    lit_scene.add(tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial()))
    renderer.render(lit_scene, _camera())
    out = tmp_path / "frame.png"
    renderer.save_frame(str(out))
    assert out.exists() and out.stat().st_size > 0


def test_tone_mapping_and_ibl(renderer, hdr_env):
    # ToneMapping operator + RGBELoader + scene.environment in one render pass.
    prev = renderer.tone_mapping
    try:
        renderer.tone_mapping = tp.ToneMapping.ACESFilmic
        scene = tp.Scene()
        scene.environment = tp.RGBELoader().load(hdr_env)  # image-based lighting
        mat = tp.MeshStandardMaterial()
        mat.metalness, mat.roughness = 1.0, 0.1
        scene.add(tp.Mesh(tp.SphereGeometry(1.0), mat))
        renderer.render(scene, _camera())
        img = renderer.read_pixels()
        assert int(img.max()) > int(img.min())  # the metal sphere is lit by the env
    finally:
        renderer.tone_mapping = prev


def test_read_pixels_flip_orientation(renderer, lit_scene):
    # flip=True (default) and flip=False should differ for a non-symmetric scene
    mesh = tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial())
    mesh.position.y = 1.0  # push content up so top/bottom differ
    lit_scene.add(mesh)
    renderer.render(lit_scene, _camera())
    top_down = renderer.read_pixels(flip=True)
    bottom_up = renderer.read_pixels(flip=False)
    assert top_down.shape == bottom_up.shape
    assert not (top_down == bottom_up).all()


def test_on_window_resize_registers():
    # Registering a resize callback must not raise. (No event fires for a headless
    # canvas, but this pins the binding the interactive demos rely on.)
    canvas = tp.Canvas("resize-reg", width=64, height=64, headless=True)
    canvas.on_window_resize(lambda w, h: None)


def test_resize_handler_operations(renderer):
    # The work an on_window_resize handler does: resize the renderer + fix the
    # camera aspect. Restores the session renderer's size afterwards.
    try:
        renderer.set_size(320, 240)
        assert renderer.size() == (320, 240)
        cam = tp.PerspectiveCamera(60, 1.0, 0.1, 100)
        cam.aspect = 320 / 240
        cam.update_projection_matrix()
        assert abs(cam.aspect - 320 / 240) < 1e-5
    finally:
        renderer.set_size(200, 150)  # restore the shared fixture's size
