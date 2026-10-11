"""Canvas input callbacks and cursor position.

A headless canvas is a hidden GLFW window, so no real key or mouse event can be
synthesised here: these tests cover registration, removal and the cursor cache's
initial state. Dispatch itself is the library's PeripheralsEventSource path.
"""
import threepp as tp


def test_register_and_remove_input_callbacks():
    canvas = tp.Canvas("input", width=64, height=48, headless=True)

    key_id = canvas.on_key_pressed(lambda key, mods: None)
    mouse_id = canvas.on_mouse_move(lambda x, y: None)
    assert key_id > 0 and mouse_id > 0 and key_id != mouse_id

    canvas.remove_key_listener(key_id)
    canvas.remove_mouse_listener(mouse_id)
    canvas.remove_key_listener(123456)  # unknown ids are ignored
    canvas.remove_mouse_listener(123456)

    pos = canvas.mouse_position()
    assert isinstance(pos, tuple) and len(pos) == 2
    assert pos == (0.0, 0.0) and all(isinstance(v, float) for v in pos)

    assert canvas.is_key_down("W") is False
