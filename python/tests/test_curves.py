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


# --- array forms of the point API ----------------------------------------------

def _rows(points):
    return np.array([[p.x, p.y] if isinstance(p, tp.Vector2) else [p.x, p.y, p.z] for p in points],
                    dtype=np.float32)


def test_catmullrom_from_array_matches_the_list_constructor():
    rows = np.array([[0, 0, 0], [1, 2, 0], [2, 0, 1], [3, 1, 1]], dtype=np.float32)
    from_list = tp.CatmullRomCurve3([tp.Vector3(*r) for r in rows])
    from_array = tp.CatmullRomCurve3.from_array(rows)
    assert isinstance(from_array, tp.Curve3)
    assert len(from_array.points) == 4
    assert from_array.get_length() == pytest.approx(from_list.get_length())
    u = np.linspace(0.0, 1.0, 9)
    np.testing.assert_allclose(from_array.get_points_at(u),
                               _rows([from_list.get_point_at(float(f)) for f in u]), atol=1e-5)


def test_catmullrom_from_array_forwards_the_options():
    rows = np.array([[0, 0, 0], [1, 2, 0], [2, 0, 1]], dtype=np.float64)  # float64 is cast
    curve = tp.CatmullRomCurve3.from_array(rows, closed=True,
                                           curve_type=tp.CatmullRomCurve3.CurveType.chordal, tension=0.3)
    assert curve.closed
    assert curve.curve_type == tp.CatmullRomCurve3.CurveType.chordal
    assert curve.tension == pytest.approx(0.3)


def test_spline_curve_from_array_matches_the_list_constructor():
    rows = np.array([[0, 0], [5, 5], [10, 0]], dtype=np.float32)
    from_list = tp.SplineCurve([tp.Vector2(*r) for r in rows])
    from_array = tp.SplineCurve.from_array(rows)
    assert isinstance(from_array, tp.Curve2)
    assert from_array.get_length() == pytest.approx(from_list.get_length())
    assert from_array.get_point(0.5).y == pytest.approx(5.0, abs=1e-4)


def test_from_array_rejects_the_wrong_shape():
    with pytest.raises(ValueError):
        tp.CatmullRomCurve3.from_array(np.zeros((4, 2), dtype=np.float32))
    with pytest.raises(ValueError):
        tp.SplineCurve.from_array(np.zeros((4, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        tp.SplineCurve.from_array(np.zeros(6, dtype=np.float32))


def test_array_sampling_matches_the_list_forms():
    curve = tp.CatmullRomCurve3.from_array(np.array([[0, 0, 0], [5, 3, 0], [10, 0, 2]], dtype=np.float32))

    pts = curve.get_points_array(8)
    assert pts.shape == (9, 3) and pts.dtype == np.float32
    np.testing.assert_allclose(pts, _rows(curve.get_points(8)), atol=1e-6)

    spaced = curve.get_spaced_points_array(8)
    assert spaced.shape == (9, 3)
    np.testing.assert_allclose(spaced, _rows(curve.get_spaced_points(8)), atol=1e-6)

    # get_spaced_points is get_point_at on an even grid, so the vectorised form agrees.
    u = np.linspace(0.0, 1.0, 9)
    np.testing.assert_allclose(curve.get_points_at(u), spaced, atol=1e-6)

    tangents = curve.get_tangents_at(u)
    assert tangents.shape == (9, 3)
    np.testing.assert_allclose(tangents, _rows([curve.get_tangent_at(float(f)) for f in u]), atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=1), 1.0, atol=1e-5)


def test_array_sampling_on_a_2d_curve():
    line = tp.LineCurve(tp.Vector2(0, 0), tp.Vector2(3, 4))
    pts = line.get_points_at(np.array([0.0, 0.5, 1.0]))
    assert pts.shape == (3, 2)
    np.testing.assert_allclose(pts, [[0, 0], [1.5, 2], [3, 4]], atol=1e-5)
    assert line.get_points_array(2).shape == (3, 2)
    assert line.get_tangents_at(np.array([0.5])).shape == (1, 2)
    with pytest.raises(ValueError):
        line.get_points_at(np.zeros((2, 2)))


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

