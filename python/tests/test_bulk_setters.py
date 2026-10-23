"""The bulk setters: TransformBatch.set and PhysxVehicle.write_commands store exactly
what the per-object / per-vehicle calls they replace store."""
import gc

import numpy as np
import pytest
import threepp as tp


def _objects():
    geo, mat = tp.BoxGeometry(1, 1, 1), tp.MeshBasicMaterial()
    return [tp.Object3D(), tp.Group(), tp.Mesh(geo, mat), tp.PerspectiveCamera(), tp.Mesh(geo, mat), tp.Group()]


def _pose(o):
    return (o.position.x, o.position.y, o.position.z, o.quaternion.x, o.quaternion.y, o.quaternion.z,
            o.quaternion.w, o.rotation.x, o.rotation.y, o.rotation.z, o.scale.x, o.scale.y, o.scale.z)


def test_transform_batch_matches_the_single_calls():
    rng = np.random.default_rng(7)
    a, b = _objects(), _objects()
    batch = tp.TransformBatch(a)
    assert len(batch) == len(a)
    for _ in range(3):
        pq = rng.normal(size=(len(a), 7)) * 3.0       # values that do not fit float32 exactly
        pq[:, 3:] /= np.linalg.norm(pq[:, 3:], axis=1, keepdims=True)
        idx = np.array([4, 1])
        scales = rng.uniform(0.2, 2.0, size=(2, 3))
        batch.set(pq, idx, scales)
        for o, r in zip(b, pq.tolist()):
            o.position.set(*r[:3])
            o.quaternion.set(*r[3:])
        for i, s in zip(idx.tolist(), scales.tolist()):
            b[i].scale.set(*s)
        assert [_pose(o) for o in a] == [_pose(o) for o in b]
    # the Euler follows the quaternion (its change callback), as with quaternion.set
    assert a[0].rotation.x != 0.0 or a[0].rotation.y != 0.0


def test_transform_batch_without_scales_keeps_them():
    a = _objects()
    for o in a:
        o.scale.set(2, 3, 4)
    tp.TransformBatch(a).set(np.zeros((len(a), 7)) + [1, 2, 3, 0, 0, 0, 1])
    assert all((o.scale.x, o.scale.y, o.scale.z) == (2, 3, 4) for o in a)
    assert all((o.position.x, o.position.y, o.position.z) == (1, 2, 3) for o in a)


def test_transform_batch_holds_its_objects():
    batch = tp.TransformBatch(_objects())
    gc.collect()
    batch.set(np.ones((len(batch), 7)))


def test_transform_batch_rejects_bad_shapes():
    batch = tp.TransformBatch(_objects())
    n = len(batch)
    with pytest.raises(ValueError):
        batch.set(np.zeros((n - 1, 7)))
    with pytest.raises(ValueError):
        batch.set(np.zeros((n, 6)))
    with pytest.raises(ValueError):
        batch.set(np.zeros((n, 7)), np.array([0]))
    with pytest.raises(IndexError):
        batch.set(np.zeros((n, 7)), np.array([n]), np.ones((1, 3)))
    with pytest.raises(ValueError):
        tp.TransformBatch([tp.Group(), None])


physx = pytest.mark.skipif(not getattr(tp, "HAS_PHYSX", False), reason="no PhysX in this build")


def _cmds(step, n):
    """A command row per car for this step: steer, throttle, brake, gear, then (on, torque,
    dtorque_domega, omega_ref) per wheel, changing every step and switching wheels on and off."""
    c = np.zeros((n, 20))
    for k in range(n):
        t = 0.1 * step + k
        c[k, :4] = [0.8 * np.sin(t), 0.5 + 0.5 * np.cos(t), max(0.0, np.sin(3 * t)) ** 2, (2, 1, 0)[(step // 40) % 3]]
        for i in range(4):
            on = (step + i + k) % 3 != 0
            c[k, 4 + 4 * i:8 + 4 * i] = [on, -40.0 * np.sin(t + i), -2.5, 3.0 * np.cos(t)]
    return c


def _drive(bulk):
    """Two cars on a slab for 150 steps, commanded per call or in one write; every step's states."""
    world = tp.PhysxWorld()
    slab = tp.Mesh(tp.BoxGeometry(80, 1, 80), tp.MeshStandardMaterial())
    slab.position.set(0, -0.5, 0)
    world.add_static(slab)
    cars = [tp.PhysxVehicle(world, position=tp.Vector3(x, 0.6, 0)) for x in (-4.0, 4.0)]
    Gear = tp.PhysxVehicle.Gear
    states, gears = [], []
    for step in range(150):
        c = _cmds(step, len(cars))
        if bulk:
            tp.PhysxVehicle.write_commands(cars, c)
        else:
            for v, r in zip(cars, c.tolist()):
                v.set_steer(r[0])
                v.set_throttle(r[1])
                for i in range(4):
                    on, tq, damp, ref = r[4 + 4 * i:8 + 4 * i]
                    if on:
                        v.set_wheel_external_torque(i, tq, damp, ref)
                    else:
                        v.clear_wheel_external_torque(i)
                v.set_brake(r[2])
                v.gear = Gear(int(r[3]))
        gears.append([v.gear for v in cars])
        world.step(1 / 60)
        states.append(tp.PhysxVehicle.read_states(cars))
    del cars, world
    gc.collect()
    return np.array(states), gears


@physx
def test_write_commands_matches_the_single_setters():
    single, single_gears = _drive(False)
    bulk, bulk_gears = _drive(True)
    assert bulk_gears == single_gears
    assert np.array_equal(bulk, single)
    assert np.abs(bulk[-1, :, [0, 2]] - bulk[0, :, [0, 2]]).max() > 0.1, "the cars were driven"
    assert np.ptp(bulk[:, :, 7 + 11::15]) > 1.0, "the wheels spun up and down"


@physx
def test_write_commands_rejects_bad_input():
    world = tp.PhysxWorld()
    cars = [tp.PhysxVehicle(world, position=tp.Vector3(0, 0.6, 0))]
    with pytest.raises(ValueError):
        tp.PhysxVehicle.write_commands(cars, np.zeros((1, 19)))
    with pytest.raises(ValueError):
        tp.PhysxVehicle.write_commands(cars, np.zeros((2, 20)))
    bad = np.zeros((1, 20))
    bad[0, 3] = 3
    with pytest.raises(ValueError):
        tp.PhysxVehicle.write_commands(cars, bad)
    del cars, world
