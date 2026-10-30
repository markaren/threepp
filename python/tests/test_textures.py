import threepp as tp


def test_texture_loader(checker_texture):
    tex = tp.TextureLoader().load(checker_texture, tp.ColorSpace.SRGB)
    assert tex is not None
    assert tex.color_space == tp.ColorSpace.SRGB


def test_texture_default_load(checker_texture):
    tex = tp.TextureLoader().load(checker_texture)
    assert tex is not None
    assert tex.color_space == tp.ColorSpace.NoColorSpace  # default tag


def test_texture_fields():
    tex = tp.Texture()
    tex.name = "t"
    tex.wrap_s = tp.TextureWrapping.Repeat
    tex.wrap_t = tp.TextureWrapping.MirroredRepeat
    tex.repeat.set(2, 2)
    assert tex.wrap_s == tp.TextureWrapping.Repeat
    assert tex.repeat.x == 2


def test_textured_render_shows_both_checker_colours(renderer, lit_scene, checker_texture):
    tex = tp.TextureLoader().load(checker_texture, tp.ColorSpace.SRGB)
    mat = tp.MeshStandardMaterial()
    mat.map = tex
    lit_scene.add(tp.Mesh(tp.BoxGeometry(1.6, 1.6, 1.6), mat))

    cam = tp.PerspectiveCamera(55, 200 / 150, 0.1, 100)
    cam.position.set(2.2, 1.8, 3.0)
    cam.look_at(0, 0, 0)

    renderer.render(lit_scene, cam)
    img = renderer.read_pixels().astype(int)
    assert img.shape == (150, 200, 3)
    # White light keeps the hues apart, and the clear colour is neither: a lit box
    # without its map has no pixel on either side of these.
    red, blue = img[..., 0], img[..., 2]
    assert int((red > blue + 60).sum()) > 500, "no orange squares on the box"
    assert int((blue > red + 60).sum()) > 500, "no blue squares on the box"
