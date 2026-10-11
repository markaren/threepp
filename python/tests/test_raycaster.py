"""Raycaster bindings: hits, misses, camera rays, recursion and instance ids."""
import pytest

import threepp as tp


def _box():
    mesh = tp.Mesh(tp.BoxGeometry(), tp.MeshBasicMaterial())
    mesh.update_matrix_world()
    return mesh


def test_hit_and_miss():
    mesh = _box()
    hits = tp.Raycaster(tp.Vector3(0, 0, 5), tp.Vector3(0, 0, -1)).intersect_object(mesh)
    assert len(hits) >= 1
    first = hits[0]
    assert first.distance == pytest.approx(4.5)
    assert first.point.z == pytest.approx(0.5)
    assert first.object is mesh

    miss = tp.Raycaster(tp.Vector3(5, 0, 5), tp.Vector3(0, 0, -1)).intersect_object(mesh)
    assert miss == []


def test_set_from_camera():
    mesh = _box()
    camera = tp.PerspectiveCamera(60, 1.0, 0.1, 100)
    camera.position.set(0, 0, 5)
    camera.update_matrix_world()

    raycaster = tp.Raycaster()
    raycaster.set_from_camera((0, 0), camera)
    hits = raycaster.intersect_object(mesh)
    assert hits and hits[0].distance == pytest.approx(4.5, abs=1e-3)


def test_recursive_and_instance_id():
    group = tp.Group()
    child = _box()
    group.add(child)
    group.update_matrix_world()
    raycaster = tp.Raycaster(tp.Vector3(0, 0, 5), tp.Vector3(0, 0, -1))
    assert raycaster.intersect_objects([group], recursive=True)[0].object is child
    assert raycaster.intersect_objects([group], recursive=False) == []

    inst = tp.InstancedMesh(tp.BoxGeometry(), tp.MeshBasicMaterial(), 3)
    for i in range(3):
        inst.set_matrix_at(i, tp.Matrix4().make_translation(2.0 * i, 0, 0))
    inst.update_matrix_world()
    ray = tp.Raycaster(tp.Vector3(4, 0, 5), tp.Vector3(0, 0, -1))
    assert ray.intersect_object(inst)[0].instance_id == 2
