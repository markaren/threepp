"""Fonts, 2D/3D text, billboard labels, and SVG -> meshes.

FontLoader.default_font() is embedded, so none of this needs a font asset.
"""
import numpy as np
import threepp as tp


def test_default_font():
    font = tp.FontLoader().default_font()
    assert isinstance(font, tp.Font)


def test_text2d_is_a_mesh():
    font = tp.FontLoader().default_font()
    t = tp.Text2D(font, "Hi", size=1.0)
    assert isinstance(t, tp.Mesh)           # flat text is a renderable mesh
    assert t.geometry is not None
    t.set_text("Hello")                     # dynamic re-text
    t.set_color(0xff8800)
    t.position.set(1, 2, 0)                 # inherits the Object3D/Mesh API
    assert t.position.x == 1


def test_text3d_extruded():
    font = tp.FontLoader().default_font()
    t = tp.Text3D(font, "3D", size=1.0, height=0.3)
    assert isinstance(t, tp.Mesh)
    assert t.geometry is not None


def test_text_sprite_label():
    font = tp.FontLoader().default_font()
    s = tp.TextSprite(font)
    assert isinstance(s, tp.Sprite)         # a billboard that faces the camera
    s.set_text("robot-1")
    s.set_color(0x00ffcc)
    s.set_world_scale(0.5)
    s.set_horizontal_alignment(tp.HorizontalAlignment.Center)
    s.set_vertical_alignment(tp.VerticalAlignment.Above)
    assert s.get_text() == "robot-1"


def test_svg_parse_to_group():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
           '<path d="M0,0 L10,0 L5,10 Z" fill="#ff3300"/></svg>')
    group = tp.SVGLoader().parse(svg)
    assert isinstance(group, tp.Group)
    meshes = [0]
    group.traverse(lambda o: meshes.__setitem__(0, meshes[0] + (1 if isinstance(o, tp.Mesh) else 0)))
    assert meshes[0] >= 1                    # at least one filled shape became a mesh


def _covers(geometry, x, y):
    """Whether a triangle of a flat geometry contains the point (x, y)."""
    pos = geometry.get_attribute("position")[:, :2].astype(np.float64)
    a, b, c = (pos[i] for i in np.asarray(geometry.get_index()).reshape(-1, 3).T)

    def side(u, v):
        return (v[:, 0] - u[:, 0]) * (y - u[:, 1]) - (v[:, 1] - u[:, 1]) * (x - u[:, 0])

    d = np.stack([side(a, b), side(b, c), side(c, a)])
    return bool(np.any(np.all(d >= 0, axis=0) | np.all(d <= 0, axis=0)))


def _rect_geometry(attrs):
    svg = f'<svg xmlns="http://www.w3.org/2000/svg"><rect {attrs} fill="#fff"/></svg>'
    return tp.SVGLoader().parse(svg, curve_segments=48).children[0].geometry


def test_svg_rounded_rect_corners():
    # a pill: rx is half the width, so each end is a semicircle about (256, 102) / (256, 282)
    pill = _rect_geometry('x="250" y="96" width="12" height="192" rx="6" ry="6"')
    pos = pill.get_attribute("position")
    assert np.allclose(pos.min(axis=0)[:2], (250, 96), atol=1e-3)   # the rect's own box,
    assert np.allclose(pos.max(axis=0)[:2], (262, 288), atol=1e-3)  # not one grown or shrunk
    assert _covers(pill, 256, 192)
    assert _covers(pill, 252, 98)            # 5.7 from the arc's centre: inside it
    # in the corner's 6x6 square but 7.8 from the centre. A corner used to be spread over
    # twice its radius, which collapsed at rx = w/4 and left this pill with square ends.
    assert not _covers(pill, 250.5, 96.5)
    assert not _covers(pill, 261.5, 287.5)

    # a radius is clamped to half its side, so an oversized one gives the same pill
    clamped = _rect_geometry('x="250" y="96" width="12" height="192" rx="50" ry="6"')
    assert np.allclose(clamped.get_attribute("position"), pos, atol=1e-4)

    # rx alone also sets ry; the r=8 arc crosses the corner's diagonal at 2.34
    card = _rect_geometry('x="0" y="0" width="100" height="60" rx="8"')
    assert _covers(card, 2.6, 2.6)
    assert not _covers(card, 2.15, 2.15)     # the old over-round corner crossed at 2.0
    assert not _covers(card, 97.85, 57.85)


def test_font_metrics_for_layout():
    font = tp.FontLoader().default_font()
    a, b = font.advance("a", 20), font.advance("b", 20)
    assert a > 0 and b > 0
    assert abs(font.advance("ab", 20) - (a + b)) < 1e-4     # advances add up, no kerning
    assert abs(font.advance("ab", 40) - 2 * (a + b)) < 1e-3  # and scale with size
    assert font.advance("ab" + chr(10) + "a", 20) == font.advance("ab", 20)  # widest line
    assert font.ascender(20) > 0 > font.descender(20)


def test_text_is_utf8():
    font = tp.FontLoader().default_font()
    micro = chr(0xB5) + "m"      # two characters, three UTF-8 bytes
    # it used to be measured (and drawn) as three '?' glyphs
    assert font.advance(micro, 20) != font.advance("??m", 20)
