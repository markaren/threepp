"""Dear ImGui UI binding (GL backend). Skips on a build without imgui."""
import gc

import threepp as tp
import pytest

pytestmark = pytest.mark.skipif(not tp.HAS_IMGUI, reason="built without imgui")

W, H = 320, 240


@pytest.fixture(scope="module")
def gl():
    canvas = tp.Canvas("imgui-test", width=W, height=H, headless=True)
    renderer = tp.GLRenderer(canvas)
    renderer.set_clear_color(0x202830)
    ui = tp.ImguiContext(canvas, renderer)  # after the renderer
    return canvas, renderer, ui


def test_imgui_panel_draws(gl):
    _, renderer, ui = gl
    scene = tp.Scene()
    cam = tp.PerspectiveCamera(60, W / H, 0.1, 100)
    cam.position.z = 5

    captured = {}

    def draw():
        tp.imgui.set_next_window_pos(8, 8)
        tp.imgui.set_next_window_size(180, 130)
        tp.imgui.begin("Panel")
        tp.imgui.text("hello")
        captured["slider"] = tp.imgui.slider_float("v", 0.5, 0.0, 1.0)
        captured["check"] = tp.imgui.checkbox("on", True)
        captured["button"] = tp.imgui.button("go")
        tp.imgui.end()

    renderer.render(scene, cam)
    ui.render(draw)
    img = renderer.read_pixels()

    # widget return shapes
    assert isinstance(captured["slider"], tuple) and len(captured["slider"]) == 2
    assert captured["check"][1] is True
    assert captured["button"] is False  # not clicked

    # the panel region should contain non-background pixels (it drew)
    panel = img[8:138, 8:188]
    assert int(panel.max()) - int(panel.min()) > 40


def test_material_after_disposed_program_and_imgui_frame_draws(gl):
    """A material built after another was disposed and an ImGui frame was drawn still draws.

    Disposing a material deletes its GL program, but the program is current, so it only
    lingers. ImGui's backend then binds its own program and does not restore a deleted
    one, which frees the name; the next program the renderer links is handed that name.
    The renderer's program cache has to have let go of it, or the new material is taken
    for already current and its mesh is drawn with ImGui's program: nothing on screen.
    """
    _, renderer, ui = gl
    cam = tp.PerspectiveCamera(60, W / H, 0.1, 100)
    cam.position.z = 3

    first = tp.Scene()
    lit = tp.MeshStandardMaterial()
    lit.color = 0x00ff00
    first.add(tp.Mesh(tp.BoxGeometry(), lit))
    first.add(tp.HemisphereLight(0xffffff, 0x404040, 1.0))
    renderer.render(first, cam)
    del first, lit
    gc.collect()  # the material dies here, and its program with it

    def draw():
        tp.imgui.begin("P")
        tp.imgui.text("x")
        tp.imgui.end()

    renderer.render(tp.Scene(), cam)
    ui.render(draw)

    second = tp.Scene()
    flat = tp.MeshBasicMaterial()
    flat.color = 0xff0000
    second.add(tp.Mesh(tp.BoxGeometry(2, 2, 2), flat))
    renderer.render(second, cam)

    assert tuple(int(v) for v in renderer.read_pixels()[H // 2, W // 2]) == (255, 0, 0)

# NOTE: the Vulkan ImGui tests live in test_vulkan.py — a GL and a Vulkan ImGui
# context must not be alive at the same time (two live ImGui contexts crash), and
# this module's GL `gl` fixture is torn down before test_vulkan.py runs.
