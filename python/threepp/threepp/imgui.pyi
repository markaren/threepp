"""
Dear ImGui immediate-mode widgets (call inside ImguiContext.render's draw callback).
"""
from __future__ import annotations
import collections.abc
import typing
__all__: list[str] = ['WINDOW_ALWAYS_AUTO_RESIZE', 'WINDOW_NO_BACKGROUND', 'WINDOW_NO_BRING_TO_FRONT_ON_FOCUS', 'WINDOW_NO_COLLAPSE', 'WINDOW_NO_DECORATION', 'WINDOW_NO_FOCUS_ON_APPEARING', 'WINDOW_NO_INPUTS', 'WINDOW_NO_MOVE', 'WINDOW_NO_RESIZE', 'WINDOW_NO_SAVED_SETTINGS', 'WINDOW_NO_SCROLLBAR', 'WINDOW_NO_TITLE_BAR', 'WINDOW_OVERLAY', 'begin', 'bullet_text', 'button', 'calc_text_size', 'checkbox', 'collapsing_header', 'color_edit3', 'combo', 'drag_float', 'draw_circle', 'draw_line', 'draw_polyline', 'draw_rect', 'draw_text', 'dummy', 'end', 'get_framerate', 'input_float', 'item_rect', 'plot_lines', 'same_line', 'separator', 'set_next_window_bg_alpha', 'set_next_window_pos', 'set_next_window_size', 'set_window_font_scale', 'show_demo_window', 'slider_float', 'slider_int', 'spacing', 'text', 'tree_node', 'tree_pop']
def begin(name: str, flags: typing.SupportsInt | typing.SupportsIndex = 0) -> bool:
    """
    Start a window; returns False if collapsed. Pair with end(). flags: an OR of the WINDOW_* constants (WINDOW_OVERLAY is a full-screen HUD layer: no decoration, inputs, background, focus or saved settings).
    """
def bullet_text(text: str) -> None:
    ...
def button(label: str) -> bool:
    """
    Returns True on the frame the button is clicked.
    """
def calc_text_size(text: str) -> tuple[float, float]:
    """
    (width, height) of a text at the current window's font and scale.
    """
def checkbox(label: str, value: bool) -> tuple[bool, bool]:
    ...
def collapsing_header(label: str) -> bool:
    ...
def color_edit3(label: str, rgb: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(3)"]) -> tuple[bool, tuple[float, float, float]]:
    ...
def combo(label: str, current: typing.SupportsInt | typing.SupportsIndex, items: collections.abc.Sequence[str]) -> tuple[bool, int]:
    ...
def drag_float(label: str, value: typing.SupportsFloat | typing.SupportsIndex, speed: typing.SupportsFloat | typing.SupportsIndex = 1.0, min: typing.SupportsFloat | typing.SupportsIndex = 0.0, max: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> tuple[bool, float]:
    ...
def draw_circle(x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, radius: typing.SupportsFloat | typing.SupportsIndex, rgba: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(4)"], thickness: typing.SupportsFloat | typing.SupportsIndex = 1.0, filled: bool = False) -> None:
    ...
def draw_line(x0: typing.SupportsFloat | typing.SupportsIndex, y0: typing.SupportsFloat | typing.SupportsIndex, x1: typing.SupportsFloat | typing.SupportsIndex, y1: typing.SupportsFloat | typing.SupportsIndex, rgba: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(4)"], thickness: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
    ...
def draw_polyline(points: collections.abc.Sequence[typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(2)"]], rgba: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(4)"], thickness: typing.SupportsFloat | typing.SupportsIndex = 1.0) -> None:
    """
    One open polyline through screen-space (x, y) points.
    """
def draw_rect(x0: typing.SupportsFloat | typing.SupportsIndex, y0: typing.SupportsFloat | typing.SupportsIndex, x1: typing.SupportsFloat | typing.SupportsIndex, y1: typing.SupportsFloat | typing.SupportsIndex, rgba: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(4)"], thickness: typing.SupportsFloat | typing.SupportsIndex = 1.0, filled: bool = False) -> None:
    ...
def draw_text(x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, text: str, rgba: typing.Annotated[collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], "FixedSize(4)"]) -> None:
    ...
def dummy(width: typing.SupportsFloat | typing.SupportsIndex, height: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """
    Reserve an empty rect in the layout; pair with item_rect().
    """
def end() -> None:
    ...
def get_framerate() -> float:
    ...
def input_float(label: str, value: typing.SupportsFloat | typing.SupportsIndex) -> tuple[bool, float]:
    ...
def item_rect() -> tuple[float, float, float, float]:
    """
    (x0, y0, x1, y1) of the LAST item, in screen coordinates.
    """
def plot_lines(label: str, values: collections.abc.Sequence[typing.SupportsFloat | typing.SupportsIndex], overlay: str = '', scale_min: typing.SupportsFloat | typing.SupportsIndex = 3.4028234663852886e+38, scale_max: typing.SupportsFloat | typing.SupportsIndex = 3.4028234663852886e+38, width: typing.SupportsFloat | typing.SupportsIndex = 0.0, height: typing.SupportsFloat | typing.SupportsIndex = 0.0) -> None:
    """
    A sparkline of `values`. scale_min/max default to the data's own range.
    """
def same_line() -> None:
    ...
def separator() -> None:
    ...
def set_next_window_bg_alpha(alpha: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """
    The next window's background opacity (0 transparent .. 1).
    """
def set_next_window_pos(x: typing.SupportsFloat | typing.SupportsIndex, y: typing.SupportsFloat | typing.SupportsIndex, always: bool = False) -> None:
    """
    Place the next window; once (its first use) unless always=True (a HUD that follows the screen).
    """
def set_next_window_size(width: typing.SupportsFloat | typing.SupportsIndex, height: typing.SupportsFloat | typing.SupportsIndex, always: bool = False) -> None:
    """
    Size the next window; once (its first use) unless always=True.
    """
def set_window_font_scale(scale: typing.SupportsFloat | typing.SupportsIndex) -> None:
    """
    Scale the current window's font (call between begin and end).
    """
def show_demo_window() -> None:
    """
    Show the built-in ImGui demo window (a gallery of every widget).
    """
def slider_float(label: str, value: typing.SupportsFloat | typing.SupportsIndex, min: typing.SupportsFloat | typing.SupportsIndex, max: typing.SupportsFloat | typing.SupportsIndex) -> tuple[bool, float]:
    ...
def slider_int(label: str, value: typing.SupportsInt | typing.SupportsIndex, min: typing.SupportsInt | typing.SupportsIndex, max: typing.SupportsInt | typing.SupportsIndex) -> tuple[bool, int]:
    ...
def spacing() -> None:
    ...
def text(text: str) -> None:
    ...
def tree_node(label: str) -> bool:
    ...
def tree_pop() -> None:
    ...
WINDOW_ALWAYS_AUTO_RESIZE: int = 64
WINDOW_NO_BACKGROUND: int = 128
WINDOW_NO_BRING_TO_FRONT_ON_FOCUS: int = 8192
WINDOW_NO_COLLAPSE: int = 32
WINDOW_NO_DECORATION: int = 43
WINDOW_NO_FOCUS_ON_APPEARING: int = 4096
WINDOW_NO_INPUTS: int = 197120
WINDOW_NO_MOVE: int = 4
WINDOW_NO_RESIZE: int = 2
WINDOW_NO_SAVED_SETTINGS: int = 256
WINDOW_NO_SCROLLBAR: int = 8
WINDOW_NO_TITLE_BAR: int = 1
WINDOW_OVERLAY: int = 209839
