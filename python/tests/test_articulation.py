"""PhysX reduced-coordinate articulations (robots): links, revolute joints, motors.

Skips on a build without the omniverse-physx-sdk. Pure CPU, headless.
"""
import threepp as tp
import pytest

pytestmark = pytest.mark.skipif(not tp.HAS_PHYSX, reason="built without the PhysX backend")


def box(sx, sy, sz, pos):
    m = tp.Mesh(tp.BoxGeometry(sx, sy, sz), tp.MeshStandardMaterial())
    m.position.set(*pos)
    return m


def test_two_link_arm_reaches_drive_targets():
    # Fixed-base 2-link arm in zero gravity: the PD drives should bring each joint
    # to its commanded angle (steady-state == target when there's no external load).
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, 0))
    art = world.create_articulation(fixed_base=True)
    base = art.add_link(box(0.2, 0.2, 0.2, (0, 2, 0)))
    link1 = art.add_link(box(1.0, 0.2, 0.2, (0.5, 2, 0)), parent=base,
                         axis=(0, 0, 1), anchor=(0.0, 2, 0),
                         lower=-1.6, upper=1.6, stiffness=2e4, damping=2e3, max_force=1e7)
    link2 = art.add_link(box(1.0, 0.2, 0.2, (1.5, 2, 0)), parent=link1,
                         axis=(0, 0, 1), anchor=(1.0, 2, 0),
                         lower=-1.6, upper=1.6, stiffness=2e4, damping=2e3, max_force=1e7)
    art.finalize()

    link1.set_drive_target(0.5)
    link2.set_drive_target(-0.3)
    for _ in range(300):  # 5 s — overdamped drive settles well within this
        world.step(1 / 60)

    assert link1.joint_position == pytest.approx(0.5, abs=0.05)
    assert link2.joint_position == pytest.approx(-0.3, abs=0.05)


def test_free_joint_swings_under_gravity():
    # A horizontal arm on a free (undriven) hinge must swing down under gravity.
    world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0))
    art = world.create_articulation(fixed_base=True)
    base = art.add_link(box(0.2, 0.2, 0.2, (0, 3, 0)))
    arm = art.add_link(box(1.0, 0.2, 0.2, (0.5, 3, 0)), parent=base,
                       axis=(0, 0, 1), anchor=(0.0, 3, 0))  # no stiffness → free
    art.finalize()

    # A frictionless free joint is an undamped pendulum: it swings far, then back through
    # ~its start. Assert on the PEAK swing over the trajectory, not the final angle (which
    # oscillates and can land near the start at any given sample).
    start = arm.joint_position
    max_swing = 0.0
    for _ in range(120):
        world.step(1 / 60)
        max_swing = max(max_swing, abs(arm.joint_position - start))
    assert max_swing > 0.5, "free joint should swing far under gravity"


def test_drive_holds_against_gravity():
    # A strong PD drive holding target=0 keeps the arm horizontal despite gravity
    # (only a tiny steady-state sag).
    world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0))
    art = world.create_articulation(fixed_base=True)
    base = art.add_link(box(0.2, 0.2, 0.2, (0, 3, 0)))
    arm = art.add_link(box(1.0, 0.2, 0.2, (0.5, 3, 0)), parent=base,
                       axis=(0, 0, 1), anchor=(0.0, 3, 0),
                       lower=-2, upper=2, stiffness=1e5, damping=1e4, max_force=1e7)
    art.finalize()
    arm.set_drive_target(0.0)
    for _ in range(180):
        world.step(1 / 60)
    assert abs(arm.joint_position) < 0.1, "drive should hold the arm near horizontal"


def test_root_link_has_no_joint():
    world = tp.PhysxWorld()
    art = world.create_articulation(fixed_base=True)
    base = art.add_link(box(0.2, 0.2, 0.2, (0, 1, 0)))
    art.finalize()
    assert base.is_root
    with pytest.raises(RuntimeError):
        _ = base.joint_position


def test_free_floating_base_falls():
    # A non-fixed articulation should fall as a whole under gravity.
    world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0))
    art = world.create_articulation(fixed_base=False)
    base = art.add_link(box(0.4, 0.2, 0.4, (0, 5, 0)), density=200)
    art.add_link(box(0.6, 0.15, 0.15, (0.5, 5, 0)), parent=base,
                 axis=(0, 0, 1), anchor=(0.2, 5, 0), stiffness=500, damping=50)
    art.finalize()
    y0 = base.position.y
    for _ in range(60):
        world.step(1 / 60)
    assert base.position.y < y0 - 1.0, "free-floating articulation should fall"
