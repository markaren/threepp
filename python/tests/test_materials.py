import pytest

import threepp as tp

MATERIAL_CTORS = [
    tp.MeshStandardMaterial, tp.MeshPhongMaterial, tp.MeshLambertMaterial,
    tp.MeshBasicMaterial, tp.MeshNormalMaterial, tp.PointsMaterial,
    tp.LineBasicMaterial, tp.SpriteMaterial, tp.ShadowMaterial,
]

LONG = "material_name_long_enough_to_force_heap_allocation_xxxxxx"


@pytest.mark.parametrize("ctor", MATERIAL_CTORS, ids=lambda c: c.__name__)
def test_material_constructs(ctor):
    assert isinstance(ctor(), tp.Material)


@pytest.mark.parametrize("ctor", MATERIAL_CTORS, ids=lambda c: c.__name__)
def test_material_base_fields_virtual_base(ctor):
    # Material is a virtual base — exercise the std::string `name` write that
    # used to corrupt memory, repeated to surface intermittent corruption.
    for _ in range(40):
        m = ctor()
        m.name = LONG
        assert m.name == LONG
        m.opacity = 0.5
        assert abs(m.opacity - 0.5) < 1e-6
        m.transparent = True
        assert m.transparent is True
        m.side = tp.Side.Double
        assert m.side == tp.Side.Double


def test_color_hex_implicit_conversion():
    m = tp.MeshStandardMaterial()
    m.color = 0x00FF00  # int implicitly converts to Color
    assert m.color == tp.Color(0x00FF00)
    m.color = tp.Color(1.0, 0.0, 0.0)
    assert m.color.r == 1.0


def test_standard_material_pbr_fields():
    m = tp.MeshStandardMaterial()
    m.roughness = 0.25
    m.metalness = 0.8
    m.emissive = 0x111111
    m.emissive_intensity = 2.0
    m.flat_shading = True
    assert abs(m.roughness - 0.25) < 1e-6
    assert abs(m.metalness - 0.8) < 1e-6
    assert m.flat_shading is True


def test_phong_specular():
    m = tp.MeshPhongMaterial()
    m.shininess = 50
    m.specular = 0x222222
    assert abs(m.shininess - 50) < 1e-6


def test_points_and_line_fields():
    p = tp.PointsMaterial()
    p.size = 4.0
    p.size_attenuation = False
    assert p.size == 4.0
    line = tp.LineBasicMaterial()
    line.linewidth = 3.0
    assert line.linewidth == 3.0


def test_texture_slots_default_none():
    m = tp.MeshStandardMaterial()
    assert m.map is None
    assert m.normal_map is None
    assert m.roughness_map is None


def test_texture_slot_assignable():
    m = tp.MeshStandardMaterial()
    tex = tp.Texture()
    m.map = tex
    assert m.map is not None
