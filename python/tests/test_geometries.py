import pytest

import threepp as tp

GEOMETRY_CTORS = [
    tp.BoxGeometry, tp.SphereGeometry, tp.PlaneGeometry, tp.CylinderGeometry,
    tp.ConeGeometry, tp.CapsuleGeometry, tp.TorusGeometry, tp.TorusKnotGeometry,
    tp.CircleGeometry, tp.RingGeometry, tp.IcosahedronGeometry, tp.OctahedronGeometry,
]


@pytest.mark.parametrize("ctor", GEOMETRY_CTORS, ids=lambda c: c.__name__)
def test_geometry_constructs_as_buffergeometry(ctor):
    geo = ctor()
    assert isinstance(geo, tp.BufferGeometry)


def test_default_dimensions():
    assert tp.BoxGeometry().width == 1
    assert tp.BoxGeometry().height == 1
    assert tp.BoxGeometry().depth == 1
    assert tp.SphereGeometry().radius == 1
    assert tp.CapsuleGeometry().radius == 0.5  # not 1


def test_explicit_dimensions():
    box = tp.BoxGeometry(2, 3, 4)
    assert (box.width, box.height, box.depth) == (2, 3, 4)
    assert tp.SphereGeometry(2.5).radius == 2.5


def test_keyword_args():
    box = tp.BoxGeometry(width=2, depth=5)
    assert box.width == 2 and box.depth == 5


def test_geometry_transforms():
    geo = tp.BoxGeometry()
    geo.translate(1, 0, 0)
    geo.scale(2, 2, 2)
    geo.compute_vertex_normals()
    geo.compute_bounding_box()  # should not raise


def test_buffergeometry_from_points():
    geo = tp.BufferGeometry()
    geo.set_from_points([tp.Vector3(0, 0, 0), tp.Vector3(1, 1, 1), tp.Vector3(2, 0, 0)])
    # usable as Line geometry
    line = tp.Line(geo, tp.LineBasicMaterial())
    assert line.geometry is not None
