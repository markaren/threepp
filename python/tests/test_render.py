
import numpy as np

import threepp as tp


def _camera():
    cam = tp.PerspectiveCamera(60, 200 / 150, 0.1, 100)
    cam.position.set(0, 1.5, 4)
    cam.look_at(0, 0, 0)
    return cam


def test_render_draws_something(renderer, lit_scene):
    # Against the same render without the sphere: the clear colour's channels
    # differ, so a spread across the image holds on an empty frame.
    camera = _camera()
    renderer.render(lit_scene, camera)
    empty = renderer.read_pixels().astype(int)

    mat = tp.MeshStandardMaterial()
    mat.color = 0x44aa88
    lit_scene.add(tp.Mesh(tp.SphereGeometry(1.0), mat))
    renderer.render(lit_scene, camera)
    img = renderer.read_pixels().astype(int)

    changed = np.abs(img - empty).max(axis=2) > 30
    assert changed[65:85, 90:110].all(), "no sphere in the middle of the frame"
    assert not changed[:30, :30].any(), "the sphere reached the corner"
    r, g, b = img[65:85, 90:110].mean(axis=(0, 1))
    assert g > r + 30 and b > r + 30, "the sphere lost its green-blue albedo"


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
    env = tp.RGBELoader().load(hdr_env)  # image-based lighting

    def centre_mean(environment, tone_mapping):
        renderer.tone_mapping = tone_mapping
        scene = tp.Scene()
        if environment is not None:
            scene.environment = environment
        mat = tp.MeshStandardMaterial()
        mat.metalness, mat.roughness = 1.0, 0.1
        scene.add(tp.Mesh(tp.SphereGeometry(1.0), mat))
        renderer.render(scene, _camera())
        return float(renderer.read_pixels()[60:90, 85:115].mean())

    prev = renderer.tone_mapping
    try:
        aces = centre_mean(env, tp.ToneMapping.ACESFilmic)
        unlit = centre_mean(None, tp.ToneMapping.ACESFilmic)
        linear = centre_mean(env, tp.ToneMapping.NoToneMapping)
    finally:
        renderer.tone_mapping = prev
    assert aces > unlit + 100, "only the environment can light a metal sphere here"
    assert abs(aces - linear) > 3, "the tone mapping operator changed nothing"


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


def test_canvas_fullscreen_is_a_runtime_property():
    # Run-time borderless fullscreen. A headless canvas records the request and
    # keeps its size (there is no window to cover a monitor with).
    canvas = tp.Canvas("fullscreen", width=64, height=48, headless=True)
    assert canvas.fullscreen is False
    canvas.fullscreen = True
    assert canvas.fullscreen is True
    assert canvas.size() == (64, 48)
    canvas.fullscreen = False
    assert canvas.fullscreen is False


def test_a_resized_headless_renderer_reads_back_at_the_new_size(renderer):
    """The render-to-array path: GLRenderer.set_size is what changes the shape
    read_pixels comes back with, and the frame is really drawn at that size."""
    scene = tp.Scene()
    scene.add(tp.Mesh(tp.BoxGeometry(), tp.MeshBasicMaterial()))
    renderer.set_clear_color(0x000000)

    try:
        for w, h in ((320, 240), (128, 96)):
            renderer.set_size(w, h)
            cam = tp.PerspectiveCamera(60, w / h, 0.1, 100)
            cam.position.z = 5
            renderer.render(scene, cam)

            pixels = renderer.read_pixels()
            assert pixels.shape == (h, w, 3)
            # The cube is actually drawn at every size, not just allocated.
            assert (pixels.sum(axis=2) > 10).sum() > 0
    finally:
        renderer.set_size(200, 150)  # restore the shared fixture's size
        renderer.set_clear_color(0x202830)


def test_layers_hide_an_object_from_a_render(renderer, lit_scene):
    """Debug content parked on channel 1 must not reach the frame.

    Lives here rather than beside the other Layers tests because a render needs
    the session's GL context, and this file is where that context is first used
    (see the note on the `renderer` fixture: one context per process).
    """
    blocker = tp.Mesh(tp.BoxGeometry(100, 100, 1), tp.MeshBasicMaterial())
    blocker.position.z = 1
    lit_scene.add(blocker)

    camera = tp.PerspectiveCamera(60, 200 / 150, 0.1, 100)
    camera.position.z = 5

    renderer.render(lit_scene, camera)
    with_blocker = renderer.read_pixels().copy()

    blocker.layers.set(1)
    renderer.render(lit_scene, camera)
    without = renderer.read_pixels().copy()

    assert not np.array_equal(with_blocker, without)
