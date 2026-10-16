"""Object lifetimes across the bindings: a node fetched from a tree keeps only itself alive.

Object3D.parent, get_object_by_name and a light's get_target were bound with
py::keep_alive<0, 1>: the node handed back kept `self` alive through pybind11's patient
table, which the garbage collector cannot see. A lookup that returned `self` made it
immortal, `x.parent` tied x to a long-lived parent for good, and a clone's root and its
nodes, fetched from each other, kept each other alive in a cycle: the dune buggy's cloned
opponent cars never died (about 80 MB of GPU memory and 250 MB of RAM per race of five).
"""
import gc
import weakref

import threepp as tp


def _car():
    """A small car-like rig: a root, a body mesh, four named wheels under the body."""
    root = tp.Group()
    root.name = "car"
    body = tp.Mesh(tp.BoxGeometry(2.0, 0.5, 4.0), tp.MeshStandardMaterial())
    body.name = "body"
    root.add(body)
    for i in range(4):
        wheel = tp.Mesh(tp.BoxGeometry(0.4, 0.8, 0.8), tp.MeshStandardMaterial())
        wheel.name = f"wheel_{i}"
        wheel.position.set(-1.0 if i % 2 else 1.0, 0.0, -1.5 if i < 2 else 1.5)
        body.add(wheel)
    return root


def _alive(refs):
    gc.collect()
    return [name for name, r in refs.items() if r() is not None]


def test_a_clone_and_its_looked_up_nodes_die():
    template = _car()
    for _ in range(3):  # three opponents, the way a race spawns them
        clone = template.clone()
        refs = {"clone": weakref.ref(clone)}
        # The rig's lookups: wheels by name from the root, the body back up through .parent.
        for i in range(4):
            wheel = clone.get_object_by_name(f"wheel_{i}")
            assert wheel is not None and wheel.name == f"wheel_{i}"
            refs[f"wheel_{i}"] = weakref.ref(wheel)
            body = wheel.parent
            assert body is not None and body.name == "body"
            refs[f"body via wheel_{i}"] = weakref.ref(body)
            assert body.parent is clone  # the root's own proxy, not a second one
        del clone, wheel, body
        assert _alive(refs) == []
    # The template is still whole and still usable.
    assert template.get_object_by_name("wheel_3").parent.name == "body"


def test_a_lookup_that_returns_self_lets_it_die():
    obj = tp.Group()
    obj.name = "me"
    assert obj.get_object_by_name("me") is obj
    ref = weakref.ref(obj)
    del obj
    assert _alive({"obj": ref}) == []


def test_reading_parent_does_not_tie_a_child_to_its_parent():
    scene = tp.Scene()
    child = tp.Mesh(tp.BoxGeometry(), tp.MeshBasicMaterial())
    scene.add(child)
    assert child.parent is scene
    scene.remove(child)
    ref = weakref.ref(child)
    del child
    assert _alive({"child": ref}) == []  # while the scene lives on
    assert scene is not None


def test_a_light_target_lookup_keeps_only_what_it_must():
    # A target the scene owns: looking it up does not tie the light to it.
    scene = tp.Scene()
    target = tp.Object3D()
    scene.add(target)
    light = tp.SpotLight(0xffffff, 1.0)
    light.set_target(target)
    assert light.get_target() is target
    ref = weakref.ref(light)
    del light
    assert _alive({"light": ref}) == []

    # The built-in default target lives inside its light: its proxy keeps the light alive,
    # and the light goes with the proxy.
    light = tp.DirectionalLight(0xffffff, 1.0)
    default_target = light.get_target()
    ref = weakref.ref(light)
    del light
    assert _alive({"light": ref}) == ["light"]
    default_target.position.set(1.0, 2.0, 3.0)  # the target is still there to write to
    assert default_target.position.y == 2.0
    del default_target
    assert _alive({"light": ref}) == []
