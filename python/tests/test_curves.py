"""Curves, 2-D shapes, and the geometries built from them."""

import numpy as np
import pytest

import threepp as tp


def test_line_curve3_samples_the_segment():
    curve = tp.LineCurve3(tp.Vector3(0, 0, 0), tp.Vector3(10, 0, 0))
    assert isinstance(curve, tp.Curve3)
    assert curve.get_length() == pytest.approx(10.0)
    assert curve.get_point(0.5).x == pytest.approx(5.0)
    assert curve.get_point_at(0.25).x == pytest.approx(2.5)
    assert curve.get_tangent(0.5).x == pytest.approx(1.0)
    assert len(curve.get_points(4)) == 5


def test_catmullrom_is_a_curve3():
    curve = tp.CatmullRomCurve3([tp.Vector3(0, 0, 0), tp.Vector3(1, 2, 0), tp.Vector3(2, 0, 0)])
    assert isinstance(curve, tp.Curve3)
    assert curve.get_length() > 2.0
    assert len(curve.get_spaced_points(8)) == 9


def test_line_curve_2d():
    curve = tp.LineCurve(tp.Vector2(0, 0), tp.Vector2(3, 4))
    assert isinstance(curve, tp.Curve2)
    assert curve.get_length() == pytest.approx(5.0, rel=1e-4)
    mid = curve.get_point_at(0.5)
    assert isinstance(mid, tp.Vector2)
    assert (mid.x, mid.y) == pytest.approx((1.5, 2.0))


def test_spline_curve_2d_passes_through_its_points():
    curve = tp.SplineCurve([tp.Vector2(0, 0), tp.Vector2(5, 5), tp.Vector2(10, 0)])
    assert isinstance(curve, tp.Curve2)
    assert curve.get_point(0.5).y == pytest.approx(5.0, abs=1e-4)
    assert curve.get_length() > 10.0
    assert len(curve.get_lengths()) == curve.arc_length_divisions + 1


def test_path_walks_a_polyline():
    path = tp.Path()
    path.move_to(0, 0)
    path.line_to(10, 0)
    path.line_to(10, 10)
    assert path.get_length() == pytest.approx(20.0)
    assert path.get_point_at(0.5).x == pytest.approx(10.0)


def test_shape_from_points():
    shape = tp.Shape([tp.Vector2(0, 0), tp.Vector2(10, 0), tp.Vector2(10, 10), tp.Vector2(0, 10)])
    assert isinstance(shape, tp.Path)
    assert shape.holes == []
    outline, holes = shape.extract_points(4)
    assert len(outline) >= 4 and holes == []


def test_tube_geometry_from_a_line():
    curve = tp.LineCurve3(tp.Vector3(0, 0, 0), tp.Vector3(0, 0, 10))
    tube = tp.TubeGeometry(curve, tubular_segments=8, radius=2.0, radial_segments=6)
    assert isinstance(tube, tp.BufferGeometry)
    assert tube.radius == 2.0

    pos = tube.get_attribute("position")
    # A tube of radius 2 about the z axis.
    radii = np.hypot(pos[:, 0], pos[:, 1])
    assert radii.min() == pytest.approx(2.0, abs=1e-4)
    assert radii.max() == pytest.approx(2.0, abs=1e-4)
    assert pos[:, 2].min() == pytest.approx(0.0, abs=1e-4)
    assert pos[:, 2].max() == pytest.approx(10.0, abs=1e-4)


def test_tube_geometry_from_a_catmullrom_curve():
    curve = tp.CatmullRomCurve3([tp.Vector3(0, 0, 0), tp.Vector3(5, 3, 0), tp.Vector3(10, 0, 0)])
    tube = tp.TubeGeometry(curve, tubular_segments=12, radius=0.5)
    assert tube.get_index() is not None
    assert len(tube.get_attribute("position")) > 0


def test_tube_geometry_rejects_a_missing_path():
    with pytest.raises(ValueError):
        tp.TubeGeometry(None)


def test_extrude_geometry_makes_a_prism():
    shape = tp.Shape([tp.Vector2(0, 0), tp.Vector2(10, 0), tp.Vector2(10, 10), tp.Vector2(0, 10)])
    prism = tp.ExtrudeGeometry(shape, depth=4.0, steps=1, bevel_enabled=False)

    pos = prism.get_attribute("position")
    assert pos[:, 0].min() == pytest.approx(0.0) and pos[:, 0].max() == pytest.approx(10.0)
    assert pos[:, 2].min() == pytest.approx(0.0) and pos[:, 2].max() == pytest.approx(4.0)
    # ExtrudeGeometry emits a soup.
    assert prism.get_index() is None


def test_extrude_geometry_accepts_a_list_of_shapes():
    a = tp.Shape([tp.Vector2(0, 0), tp.Vector2(1, 0), tp.Vector2(1, 1)])
    b = tp.Shape([tp.Vector2(5, 5), tp.Vector2(6, 5), tp.Vector2(6, 6)])
    one = tp.ExtrudeGeometry(a, bevel_enabled=False)
    two = tp.ExtrudeGeometry([a, b], bevel_enabled=False)
    assert len(two.get_attribute("position")) == 2 * len(one.get_attribute("position"))


def test_shape_geometry_is_flat():
    shape = tp.Shape([tp.Vector2(0, 0), tp.Vector2(10, 0), tp.Vector2(10, 10)])
    flat = tp.ShapeGeometry(shape)
    assert np.ptp(flat.get_attribute("position")[:, 2]) == pytest.approx(0.0)


def test_apply_matrix4_swaps_axes():
    """The colnav move: a volume authored in xy gets its time axis onto y."""
    prism = tp.ExtrudeGeometry(
        tp.Shape([tp.Vector2(0, 0), tp.Vector2(10, 0), tp.Vector2(10, 10), tp.Vector2(0, 10)]),
        depth=3.0, bevel_enabled=False)
    before = prism.get_attribute("position")
    assert before[:, 2].max() == pytest.approx(3.0)

    swap = tp.Matrix4()
    swap.set(1, 0, 0, 0,
             0, 0, 1, 0,
             0, 1, 0, 0,
             0, 0, 0, 1)
    prism.apply_matrix4(swap)

    after = prism.get_attribute("position")
    assert after[:, 1].max() == pytest.approx(3.0)
    assert after[:, 2].max() == pytest.approx(10.0)


def test_set_index_on_an_extruded_soup():
    prism = tp.ExtrudeGeometry(
        tp.Shape([tp.Vector2(0, 0), tp.Vector2(1, 0), tp.Vector2(1, 1)]), bevel_enabled=False)
    n = len(prism.get_attribute("position"))
    prism.set_index(np.arange(n, dtype=np.uint32))
    assert len(prism.get_index()) == n


def test_to_non_indexed_expands_the_index():
    box = tp.BoxGeometry(1, 1, 1)
    assert len(box.get_attribute("position")) == 24
    soup = box.to_non_indexed()
    assert soup.get_index() is None
    assert len(soup.get_attribute("position")) == 36


def test_matrix4_set_and_multiply():
    a = tp.Matrix4().make_translation(1, 2, 3)
    b = tp.Matrix4().make_scale(2, 2, 2)
    m = tp.Matrix4().copy(a).multiply(b)
    v = tp.Vector3(1, 1, 1).apply_matrix4(m)
    assert (v.x, v.y, v.z) == pytest.approx((3.0, 4.0, 5.0))

    identity = tp.Matrix4()
    identity.set(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
    assert np.allclose(identity.to_numpy(), np.eye(4))


def test_ray_basics():
    ray = tp.Ray(tp.Vector3(0, 0, 0), tp.Vector3(0, 0, 1))
    assert ray.at(5).z == pytest.approx(5.0)
    assert ray.distance_to_point(tp.Vector3(3, 0, 4)) == pytest.approx(3.0)
    assert ray.intersects_box(tp.Box3(tp.Vector3(-1, -1, 4), tp.Vector3(1, 1, 6)))


def test_layers_gate_visibility_per_channel():
    mesh = tp.Mesh(tp.BoxGeometry(), tp.MeshBasicMaterial())
    camera = tp.PerspectiveCamera()

    # Everything starts on channel 0.
    assert mesh.layers.test(camera.layers)

    mesh.layers.set(1)
    assert not mesh.layers.test(camera.layers)
    camera.layers.enable(1)
    assert mesh.layers.test(camera.layers)

    camera.layers.disable(1)
    assert not mesh.layers.test(camera.layers)
    assert mesh.layers.is_enabled(1) and not mesh.layers.is_enabled(0)

