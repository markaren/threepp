"""tp.BVH — mesh-vs-mesh overlap, intersection points and distance.

The scenario throughout is the one the BVH exists for: a predicted track lifted
into a space-time volume, where x/z are horizontal position and y is time, so
the y of an intersection point is the predicted time of conflict.
"""
import numpy as np
import pytest

import threepp as tp


def tube(p0, p1, radius=5.0, leaf=4):
    """A BVH over the tube swept from p0 to p1."""
    geometry = tp.TubeGeometry(tp.LineCurve3(tp.Vector3(*p0), tp.Vector3(*p1)),
                               tubular_segments=16, radius=radius, radial_segments=8)
    bvh = tp.BVH(max_triangles_per_node=leaf)
    bvh.build_arrays(geometry.get_attribute("position"), geometry.get_index())
    return bvh


OFFSET = 12.0 / np.sqrt(2.0)  # perpendicular to the (100, 60, 100) sweep


@pytest.fixture(scope="module")
def volumes():
    return {
        "own": tube((0, 0, 0), (100, 60, 100)),
        # Crosses own path: the centrelines meet exactly at t = 30.
        "target": tube((100, 0, 0), (0, 60, 100)),
        # Far away in z.
        "clear": tube((0, 0, 500), (100, 60, 600)),
        # Runs alongside `own` 12 m off, perpendicular to the sweep: the node
        # boxes overlap nearly everywhere, the surfaces stay 2 m apart.
        "near": tube((OFFSET, 0, -OFFSET), (100 + OFFSET, 60, 100 - OFFSET)),
    }


def test_crossing_volumes_intersect(volumes):
    assert tp.BVH.intersects(volumes["own"], volumes["target"])


def test_intersection_points_carry_the_conflict_time(volumes):
    points = tp.BVH.intersect(volumes["own"], volumes["target"])
    assert points.ndim == 2 and points.shape[1] == 3
    assert points.dtype == np.float32
    assert len(points) > 0
    # The tracks cross in the middle of the 0..60 s span.
    assert 20 < points[:, 1].mean() < 40


def test_intersect_pairs_reports_which_triangles(volumes):
    idx_a, idx_b, points = tp.BVH.intersect_pairs(volumes["own"], volumes["target"])
    assert len(idx_a) == len(idx_b) == len(points)
    assert idx_a.min() >= 0 and idx_b.min() >= 0
    assert idx_a.max() < volumes["own"].triangle_count
    assert idx_b.max() < volumes["target"].triangle_count


def test_a_distant_volume_is_neither_hit_nor_near(volumes):
    assert not tp.BVH.intersects(volumes["own"], volumes["clear"])
    assert tp.BVH.intersect(volumes["own"], volumes["clear"]).shape == (0, 3)
    assert tp.BVH.distance(volumes["own"], volumes["clear"]) > 300


def test_a_near_miss_whose_boxes_overlap_is_not_a_hit(volumes):
    """The case an AABB-only leaf test gets wrong."""
    assert not tp.BVH.intersects(volumes["own"], volumes["near"])
    assert 0 < tp.BVH.distance(volumes["own"], volumes["near"]) < 3


def test_distance_is_zero_when_the_surfaces_meet(volumes):
    assert tp.BVH.distance(volumes["own"], volumes["target"]) == pytest.approx(0.0, abs=1e-5)
    assert tp.BVH.distance(volumes["own"], volumes["own"]) == pytest.approx(0.0, abs=1e-5)


def test_an_empty_bvh_is_infinitely_far(volumes):
    empty = tp.BVH()
    assert empty.triangle_count == 0
    assert not tp.BVH.intersects(volumes["own"], empty)
    assert np.isinf(tp.BVH.distance(volumes["own"], empty))


def test_a_matrix_moves_the_query_volume(volumes):
    shift = tp.Matrix4().make_translation(0, 0, 500)
    assert tp.BVH.intersects(volumes["own"], volumes["clear"], shift, tp.Matrix4())


def test_batch_queries_match_the_pair_queries(volumes):
    candidates = [volumes["own"]]
    targets = [volumes["target"], volumes["clear"], volumes["near"]]

    hits = tp.BVH.intersects_many(candidates, targets)
    assert hits.shape == (1, 3)
    assert hits.tolist() == [[True, False, False]]

    distances = tp.BVH.distance_many(candidates, targets)
    assert distances.shape == (1, 3)
    assert distances[0, 0] == pytest.approx(0.0, abs=1e-5)
    assert distances[0, 1] > 300
    assert 0 < distances[0, 2] < 3

    points = tp.BVH.intersect_many(candidates, targets)
    assert len(points) == 1 and len(points[0]) == 3
    assert len(points[0][0]) > 0
    assert points[0][1].shape == (0, 3)
    assert points[0][2].shape == (0, 3)


def test_batch_with_an_empty_side_returns_an_empty_grid(volumes):
    assert tp.BVH.intersects_many([], [volumes["own"]]).shape == (0, 1)
    assert tp.BVH.distance_many([volumes["own"]], []).shape == (1, 0)


def test_build_from_a_geometry_matches_build_from_arrays():
    geometry = tp.BoxGeometry(2, 2, 2)

    from_geometry = tp.BVH(max_triangles_per_node=1)
    from_geometry.build(geometry)

    from_arrays = tp.BVH(max_triangles_per_node=1)
    from_arrays.build_arrays(geometry.get_attribute("position"), geometry.get_index())

    assert from_geometry.triangle_count == from_arrays.triangle_count == 12
    assert tp.BVH.intersects(from_geometry, from_arrays)
    assert from_geometry.bounding_box().min().x == pytest.approx(-1.0)


def test_build_from_a_soup_needs_no_index():
    soup = tp.BoxGeometry(2, 2, 2).to_non_indexed()
    bvh = tp.BVH()
    bvh.build_arrays(soup.get_attribute("position"))
    assert bvh.triangle_count == 12


def test_separated_boxes_are_exactly_their_gap_apart():
    a, b = tp.BVH(), tp.BVH()
    a.build(tp.BoxGeometry(2, 2, 2))
    b.build(tp.BoxGeometry(2, 2, 2).translate(5, 0, 0))
    assert not tp.BVH.intersects(a, b)
    assert tp.BVH.distance(a, b) == pytest.approx(3.0, abs=1e-4)


def test_malformed_arrays_raise():
    bvh = tp.BVH()
    with pytest.raises(Exception):
        bvh.build_arrays(np.zeros((4, 2), dtype=np.float32))
    with pytest.raises(Exception):
        bvh.build_arrays(np.zeros((3, 3), dtype=np.float32), np.array([0, 1], dtype=np.uint32))
    with pytest.raises(Exception):
        bvh.build_arrays(np.zeros((3, 3), dtype=np.float32), np.array([0, 1, 99], dtype=np.uint32))


def test_raycast_hits_the_near_face():
    bvh = tp.BVH()
    bvh.build(tp.BoxGeometry(2, 2, 2))

    ray = tp.Ray(tp.Vector3(0, 0, 10), tp.Vector3(0, 0, -1))
    hit = bvh.raycast(ray)
    assert hit is not None
    assert hit.distance == pytest.approx(9.0, abs=1e-4)
    assert hit.point.z == pytest.approx(1.0, abs=1e-4)
    assert hit.normal.z == pytest.approx(1.0, abs=1e-4)
    assert bvh.raycast_any(ray, 20.0)

    assert bvh.raycast(tp.Ray(tp.Vector3(0, 0, 10), tp.Vector3(0, 1, 0))) is None


def test_collect_boxes_describes_the_tree():
    bvh = tp.BVH(max_triangles_per_node=1)
    bvh.build(tp.BoxGeometry(2, 2, 2))

    boxes = bvh.collect_boxes()
    leaves = bvh.collect_boxes(leaves_only=True)
    assert len(boxes) > len(leaves) > 0
    assert all(isinstance(b, tp.Box3) for b in boxes)


def test_queries_run_without_holding_the_gil(volumes):
    """Twenty candidates against three targets, evaluated from a thread pool."""
    from concurrent.futures import ThreadPoolExecutor

    targets = [volumes["target"], volumes["clear"], volumes["near"]]
    candidates = [volumes["own"]] * 20

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda c: [tp.BVH.intersects(c, t) for t in targets], candidates))

    assert all(r == [True, False, False] for r in results)
