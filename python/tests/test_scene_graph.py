"""Scene-graph behaviour, including the pybind11 virtual-base regression.

Mesh/Points/Line derive from Object3D virtually; writing inherited members (the
std::string `name` especially) used to corrupt the heap. These tests pin that
down — they crash the interpreter if the concrete-binding workaround regresses.
"""
import pytest

import threepp as tp

LONG = "a_name_long_enough_to_force_a_heap_allocation_xxxxxxxxxxxx"

VIRTUAL_OBJECTS = {
    "Mesh": lambda: tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial()),
    "Points": lambda: tp.Points(),
    "Line": lambda: tp.Line(tp.BufferGeometry(), tp.LineBasicMaterial()),
    "LineSegments": lambda: tp.LineSegments(tp.BufferGeometry(), tp.LineBasicMaterial()),
    "InstancedMesh": lambda: tp.InstancedMesh(tp.BoxGeometry(), tp.MeshStandardMaterial(), 4),
}


def test_mutate_in_place():
    mesh = tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial())
    mesh.position.x = 0.5
    assert mesh.position.x == 0.5  # live reference, not a copy
    mesh.rotation.y = 1.0
    assert abs(mesh.rotation.y - 1.0) < 1e-6
    mesh.position = tp.Vector3(1, 2, 3)
    assert mesh.position.z == 3


@pytest.mark.parametrize("name", list(VIRTUAL_OBJECTS))
def test_virtual_base_field_writes(name):
    make = VIRTUAL_OBJECTS[name]
    for _ in range(50):  # repeat: heap corruption is intermittent
        o = make()
        o.name = LONG
        assert o.name == LONG
        o.position.x = 1.5
        assert o.position.x == 1.5
        o.visible = False
        assert o.visible is False


@pytest.mark.parametrize("name", list(VIRTUAL_OBJECTS))
def test_virtual_base_methods(name):
    o = VIRTUAL_OBJECTS[name]()
    o.rotate_x(0.5)
    o.translate_z(2.0)
    o.look_at(0, 0, 0)
    wp = o.get_world_position()
    assert isinstance(wp, tp.Vector3)


@pytest.mark.parametrize("name", list(VIRTUAL_OBJECTS))
def test_world_pose_getters_virtual_leaves(name):
    # get_world_quaternion / get_world_scale / matrix_world / local<->world are
    # added to BOTH binding sites; this pins the virtual-base (Mesh/Points/Line)
    # path that shadows the bind_core.cpp base bindings.
    o = VIRTUAL_OBJECTS[name]()
    o.position.set(1, 2, 3)
    o.update_matrix_world(True)
    wp = o.get_world_position()
    assert (round(wp.x, 4), round(wp.y, 4), round(wp.z, 4)) == (1, 2, 3)
    assert isinstance(o.get_world_quaternion(), tp.Quaternion)
    ws = o.get_world_scale()
    assert (round(ws.x, 4), round(ws.y, 4), round(ws.z, 4)) == (1, 1, 1)
    # matrix_world translation is the world position (column-major elements 12..14)
    e = o.matrix_world.elements()
    assert (round(e[12], 4), round(e[13], 4), round(e[14], 4)) == (1, 2, 3)
    # local <-> world round-trip through the object's frame
    loc = o.world_to_local(tp.Vector3(1, 2, 3))
    assert abs(loc.x) < 1e-4 and abs(loc.y) < 1e-4 and abs(loc.z) < 1e-4
    wl = o.local_to_world(tp.Vector3(0, 0, 0))
    assert (round(wl.x, 4), round(wl.y, 4), round(wl.z, 4)) == (1, 2, 3)


def test_world_pose_getters_base_object3d():
    # the non-virtual base path (Object3D), bound directly in bind_core.cpp
    o = tp.Object3D()
    o.position.set(4, 5, 6)
    o.update_matrix_world(True)
    wp = o.get_world_position()
    assert (round(wp.x, 4), round(wp.y, 4), round(wp.z, 4)) == (4, 5, 6)
    assert isinstance(o.get_world_quaternion(), tp.Quaternion)
    loc = o.world_to_local(tp.Vector3(4, 5, 6))
    assert abs(loc.x) < 1e-4 and abs(loc.y) < 1e-4 and abs(loc.z) < 1e-4


def test_add_variadic_and_children():
    scene = tp.Scene()
    a = tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial())
    b = tp.Group()
    scene.add(a, b)
    assert len(scene.children) == 2
    types = {type(c).__name__ for c in scene.children}
    assert types == {"Mesh", "Group"}  # children downcast to concrete types


def test_traverse_downcasts_and_reads_fields():
    scene = tp.Scene()
    group = tp.Group()
    group.name = "grp"
    ball = tp.Mesh(tp.SphereGeometry(), tp.MeshStandardMaterial())
    ball.name = "ball"
    group.add(ball)
    scene.add(group)

    seen = []
    scene.traverse(lambda o: seen.append((type(o).__name__, o.name)))
    assert ("Mesh", "ball") in seen
    assert ("Group", "grp") in seen


def test_get_object_by_name():
    scene = tp.Scene()
    mesh = tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial())
    mesh.name = "target"
    scene.add(mesh)
    found = scene.get_object_by_name("target")
    assert found is not None and found.name == "target"
    assert scene.get_object_by_name("missing") is None


def test_remove():
    scene = tp.Scene()
    mesh = tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial())
    scene.add(mesh)
    assert len(scene.children) == 1
    scene.remove(mesh)
    assert len(scene.children) == 0


def test_mesh_material_downcasts_to_concrete():
    mesh = tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial())
    assert type(mesh.material).__name__ == "MeshStandardMaterial"
    mesh.material.roughness = 0.3  # concrete attribute reachable
    assert abs(mesh.material.roughness - 0.3) < 1e-6


def test_groups_nest():
    root = tp.Group()
    child = tp.Group()
    leaf = tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial())
    child.add(leaf)
    root.add(child)
    count = [0]
    root.traverse(lambda o: count.__setitem__(0, count[0] + 1))
    assert count[0] == 3  # root, child, leaf
