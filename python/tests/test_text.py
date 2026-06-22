"""Fonts, 2D/3D text, billboard labels, and SVG -> meshes.

FontLoader.default_font() is embedded, so none of this needs a font asset.
"""
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
