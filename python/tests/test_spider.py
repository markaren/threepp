"""The procedural hexapod (examples/spider/hexapod.py) — stands, walks, turns.

Validates the articulation-based robot end to end, headless. Skips without PhysX.
"""
import os
import sys

import pytest

import threepp as tp

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "examples", "spider"))

pytestmark = pytest.mark.skipif(not tp.HAS_PHYSX, reason="built without the PhysX backend")


def _world_with_ground():
    world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0))
    ground = tp.Mesh(tp.BoxGeometry(50, 1, 50), tp.MeshStandardMaterial())
    ground.position.y = -0.5
    world.add_static(ground)
    return world


def _settle(spider, world, steps):
    for _ in range(steps):
        spider.update(1 / 60)
        world.step(1 / 60)


def test_hexapod_stands():
    from hexapod import Hexapod
    world = _world_with_ground()
    spider = Hexapod(world, position=(0, 0.40, 0))
    _settle(spider, world, 120)  # 2 s
    assert spider.up_y > 0.9, "hexapod should stay upright"
    assert 0.28 < spider.position.y < 0.45, "chassis should hold a standing height"


def test_hexapod_walks_forward():
    from hexapod import Hexapod
    world = _world_with_ground()
    spider = Hexapod(world, position=(0, 0.40, 0))
    _settle(spider, world, 90)
    x0, z0 = spider.position.x, spider.position.z
    spider.set_command(1.0, 0.0)
    _settle(spider, world, 180)  # 3 s
    dx = spider.position.x - x0
    dz = abs(spider.position.z - z0)
    assert dx > 1.0, f"should walk forward (+X), got dx={dx:.2f}"
    assert dz < 0.5, f"should walk roughly straight, drift dz={dz:.2f}"
    assert spider.up_y > 0.9, "should stay upright while walking"


def test_hexapod_turns_in_place():
    import math

    from hexapod import Hexapod
    world = _world_with_ground()
    spider = Hexapod(world, position=(0, 0.40, 0))
    _settle(spider, world, 90)
    yaw0 = spider.yaw
    spider.set_command(0.0, 1.0)
    _settle(spider, world, 240)  # 4 s
    dyaw = math.degrees(spider.yaw - yaw0)
    assert abs(dyaw) > 15.0, f"turn command should rotate the body, got {dyaw:.1f} deg"
    assert spider.up_y > 0.9, "should stay upright while turning"
