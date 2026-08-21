"""PhysX rigid-body world. Skips on a build without the omniverse-physx-sdk.

Physics is pure CPU here — no canvas or renderer needed, so these run headless
anywhere the module was built with PhysX.
"""
import gc

import numpy as np
import pytest
import threepp as tp

pytestmark = pytest.mark.skipif(not tp.HAS_PHYSX, reason="built without the PhysX backend")


def box(size=1.0):
    m = tp.Mesh(tp.BoxGeometry(size, size, size), tp.MeshStandardMaterial())
    return m


def test_box_falls_under_gravity():
    world = tp.PhysxWorld()
    b = box()
    b.position.set(0, 10, 0)
    body = world.add(b, density=100)
    assert body.is_dynamic
    for _ in range(60):  # ~1 s
        world.step(1 / 60)
    assert b.position.y < 9.0, "box did not fall"


def test_static_floor_stops_fall():
    world = tp.PhysxWorld()
    floor = tp.Mesh(tp.BoxGeometry(20, 1, 20), tp.MeshStandardMaterial())
    floor.position.set(0, -0.5, 0)  # top face at y=0
    world.add_static(floor)
    b = box()
    b.position.set(0, 5, 0)
    world.add(b, density=100)
    for _ in range(240):  # 4 s — plenty to settle
        world.step(1 / 60)
    assert b.position.y == pytest.approx(0.5, abs=0.15), "box should rest on the floor (centre ~0.5)"


def test_no_gravity_floats():
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, 0))
    b = box()
    b.position.set(0, 5, 0)
    world.add(b, density=10)
    for _ in range(60):
        world.step(1 / 60)
    assert b.position.y == pytest.approx(5.0, abs=0.05), "no gravity → should not move"


def test_impulse_moves_body():
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, 0))
    b = box()
    b.position.set(0, 0, 0)
    body = world.add(b, density=10)
    body.add_impulse(tp.Vector3(40, 0, 0))
    for _ in range(30):
        world.step(1 / 60)
    assert b.position.x > 1.0, "impulse should push the body +x"
    assert body.linear_velocity.x > 0


def test_kinematic_follows_target_and_ignores_gravity():
    world = tp.PhysxWorld()
    b = box()
    b.position.set(0, 5, 0)
    body = world.add(b, density=10)
    body.set_kinematic(True)
    body.set_kinematic_target(tp.Vector3(3, 5, 0))
    for _ in range(30):
        world.step(1 / 60)
    assert b.position.x == pytest.approx(3.0, abs=0.1)
    assert b.position.y == pytest.approx(5.0, abs=0.05), "kinematic body must not fall"


def test_static_body_dynamic_op_raises():
    world = tp.PhysxWorld()
    floor = tp.Mesh(tp.BoxGeometry(10, 1, 10), tp.MeshStandardMaterial())
    b = world.add_static(floor)
    assert not b.is_dynamic
    with pytest.raises(RuntimeError):
        b.set_linear_velocity(tp.Vector3(1, 0, 0))


def test_substep_callback_fires():
    world = tp.PhysxWorld()
    b = box()
    b.position.set(0, 5, 0)
    world.add(b, density=10)
    calls = {"n": 0}
    world.on_pre_substep(lambda dt: calls.__setitem__("n", calls["n"] + 1))
    for _ in range(10):
        world.step(1 / 60)
    assert calls["n"] == 10, "pre-substep callback should fire once per fixed substep"


def test_instanced_bodies():
    world = tp.PhysxWorld()
    im = tp.InstancedMesh(tp.BoxGeometry(1, 1, 1), tp.MeshStandardMaterial(), 8)
    for i in range(8):
        mtx = tp.Matrix4()  # identity
        mtx.set_position(i * 2.0, 5.0 + i, 0.0)
        im.set_matrix_at(i, mtx)
    im.instance_matrix_needs_update()
    bodies = world.add_instanced(im, density=50)
    assert len(bodies) == 8
    y_before = bodies[0].position.y
    for _ in range(60):
        world.step(1 / 60)
    assert bodies[0].position.y < y_before, "instances should have fallen"


# --- Soft bodies (deformable volumes). GPU-only: PhysX cooks and solves these on
# CUDA, so the world needs gpu_dynamics and the box gets skipped on a CPU-only
# machine rather than failing there.

@pytest.fixture
def gpu_world():
    """One GPU world, torn down before the next test asks for one.

    PhysX allows exactly one PxFoundation per process, so a world that outlives
    its test takes the whole rest of the file down with it -- which is what a
    plain local variable does the moment a test fails and pytest keeps the
    frame alive for the traceback.
    """
    try:
        world = tp.PhysxWorld(gpu_dynamics=True)
    except Exception as e:                       # no CUDA device / no GPU PhysX build
        pytest.skip(f"no GPU PhysX: {e}")
    yield world
    del world
    gc.collect()


def test_soft_body_tet_mesh_shapes(gpu_world):
    world = gpu_world
    m = box(0.5)
    m.position.set(0, 2, 0)
    sb = world.add_soft_body(m, voxel_resolution=6, solver_iterations=15)
    verts, tets = sb.tet_mesh()
    assert verts.ndim == 2 and verts.shape[1] == 3
    assert tets.ndim == 2 and tets.shape[1] == 4
    assert len(verts) == sb.num_vertices and len(tets) == sb.num_tets
    assert tets.max() < len(verts) and tets.min() >= 0, "tet indices must address the rest vertices"
    assert sb.sim_positions().shape == verts.shape


def test_soft_body_falls_and_deforms(gpu_world):
    world = gpu_world
    floor = tp.Mesh(tp.BoxGeometry(10, 1, 10), tp.MeshStandardMaterial())
    floor.position.set(0, -0.5, 0)
    world.add_static(floor)
    m = box(0.5)
    m.position.set(0, 2, 0)
    sb = world.add_soft_body(m, voxel_resolution=6, solver_iterations=15)
    p0 = sb.sim_positions()
    for _ in range(60):
        world.step(1 / 60)
    p1 = sb.sim_positions()
    assert p1[:, 1].mean() < p0[:, 1].mean() - 0.5, "soft body did not fall"
    # A rigid drop would move every vertex by the same amount; a deformable one
    # does not, which is the only thing that distinguishes it here.
    assert float(np.ptp((p1 - p0)[:, 1])) > 1e-4, "every vertex moved identically — that is a rigid body"


def test_remove_soft_body_invalidates_handle(gpu_world):
    world = gpu_world
    m = box(0.5)
    sb = world.add_soft_body(m, voxel_resolution=6, solver_iterations=10)
    world.remove_soft_body(sb)
    with pytest.raises(RuntimeError):
        sb.sim_positions()
