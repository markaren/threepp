"""Dear ImGui UI binding (GL backend). Skips on a build without imgui."""
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


def test_want_capture_flags(gl):
    _, _, ui = gl
    # no UI is hovered in a headless one-shot, but the properties resolve
    assert ui.want_capture_mouse in (True, False)
    assert ui.want_capture_keyboard in (True, False)

# NOTE: the Vulkan ImGui tests live in test_vulkan.py — a GL and a Vulkan ImGui
# context must not be alive at the same time (two live ImGui contexts crash), and
# this module's GL `gl` fixture is torn down before test_vulkan.py runs.
