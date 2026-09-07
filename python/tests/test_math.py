import math

import numpy as np
import pytest

import threepp as tp


def test_matrix4_to_numpy_row_major():
    m = tp.Matrix4()
    m.make_translation(1, 2, 3)
    a = m.to_numpy()
    assert a.shape == (4, 4)
    # standard math layout: translation in the last column, [0,0,0,1] bottom row
    assert np.allclose(a[:3, 3], [1, 2, 3])
    assert np.allclose(a[3, :], [0, 0, 0, 1])


def test_vector3_basics():
    v = tp.Vector3(1, 2, 3)
    assert (v.x, v.y, v.z) == (1, 2, 3)
    assert tp.Vector3(0, 3, 4).length() == 5
    assert abs(tp.Vector3(0, 3, 4).length_sq() - 25) < 1e-6


def test_vector3_default_is_zero():
    v = tp.Vector3()
    assert (v.x, v.y, v.z) == (0, 0, 0)


def test_vector3_operators():
    assert tp.Vector3(1, 0, 0) + tp.Vector3(0, 1, 0) == tp.Vector3(1, 1, 0)
    assert tp.Vector3(2, 2, 2) - tp.Vector3(1, 1, 1) == tp.Vector3(1, 1, 1)
    assert tp.Vector3(1, 2, 3) * 2 == tp.Vector3(2, 4, 6)
    assert tp.Vector3(1, 0, 0).dot(tp.Vector3(0, 1, 0)) == 0


def test_vector3_mutation_and_methods():
    v = tp.Vector3(3, 0, 0)
    v.normalize()
    assert abs(v.length() - 1) < 1e-6
    v.set(1, 2, 3)
    assert (v.x, v.y, v.z) == (1, 2, 3)


def test_vector2_and_vector4():
    assert tp.Vector2(3, 4).length() == 5
    w = tp.Vector4(1, 2, 3, 4)
    assert (w.x, w.y, w.z, w.w) == (1, 2, 3, 4)
    assert tp.Vector4().w == 1  # w defaults to 1


def test_color_constructors():
    assert tp.Color(0xFF8000).r == 1.0
    assert tp.Color(0xFF8000).b == 0.0
    c = tp.Color(0.5, 0.25, 0.125)
    assert (c.r, c.g, c.b) == (0.5, 0.25, 0.125)
    assert tp.Color("#00ff00").get_hex() == 0x00FF00
    assert tp.Color("blue").get_hex() == 0x0000FF  # a name threepp knows


def test_color_hex_roundtrip():
    for hexval in (0x000000, 0x123456, 0xFFFFFF, 0xFF8800):
        assert tp.Color(hexval).get_hex() == hexval


def test_quaternion_floatview_properties():
    q = tp.Quaternion()
    assert (q.x, q.y, q.z, q.w) == (0, 0, 0, 1)
    q.set_from_axis_angle(tp.Vector3(0, 1, 0), math.pi / 2)
    assert abs(q.y) > 0.6  # ~sin(pi/4)
    q.x = 0.5
    assert q.x == 0.5  # float_view assignment round-trips


def test_euler_floatview_and_order():
    e = tp.Euler(0.1, 0.2, 0.3)
    assert abs(e.x - 0.1) < 1e-6 and abs(e.z - 0.3) < 1e-6
    e.y = 1.0
    assert abs(e.y - 1.0) < 1e-6
    assert e.order == tp.RotationOrder.XYZ


def test_matrix4():
    m = tp.Matrix4()
    assert len(m.elements()) == 16
    assert m.elements()[0] == 1 and m.elements()[5] == 1  # identity diagonal


def test_box3():
    b = tp.Box3(tp.Vector3(-1, -1, -1), tp.Vector3(1, 1, 1))
    assert not b.is_empty()
    assert b.contains_point(tp.Vector3(0, 0, 0))
    assert not b.contains_point(tp.Vector3(2, 0, 0))
    assert b.get_size().x == 2


def test_repr():
    assert "Vector3" in repr(tp.Vector3(1, 2, 3))
    assert "Color" in repr(tp.Color(0xff0000))
