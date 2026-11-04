"""PhysxVehicle(wheel_shapes=True): each wheel a collision cylinder against other vehicles only.

Pure CPU, headless. One world at a time (PhysX allows one foundation per process): every helper
builds its world, runs it and returns plain numbers.
"""
import math

import pytest
import threepp as tp

pytestmark = pytest.mark.skipif(not tp.HAS_PHYSX, reason="built without the PhysX backend")

DT = 1 / 60
# The dune-buggy's sandrail: the wheels (track 1.80, 0.15 half width) stand 0.2 m out of the
# 1.70 m chassis box on each side, so side by side the wheels meet before the bodies do.
BUGGY = dict(chassis_width=1.70, chassis_height=0.30, chassis_length=3.2, chassis_mass=850.0,
             wheelbase=2.64, track_width=1.80, wheel_radius=0.40, wheel_half_width=0.15, wheel_mass=28.0,
             suspension_travel=0.55, suspension_attachment_y=0.205, suspension_stiffness=10690.0,
             suspension_damping=1266.0)


def flat_world():
    world = tp.PhysxWorld()
    slab = tp.Mesh(tp.BoxGeometry(200, 1, 200), tp.MeshStandardMaterial())
    slab.position.set(0, -0.5, 0)
    world.add_static(slab)
    return world


def drive_alone(**kw):
    """One car accelerating through a turn and braking: its trajectory, as exact floats."""
    world = flat_world()
    car = tp.PhysxVehicle(world, position=tp.Vector3(0, 0.6, 0), **BUGGY, **kw)
    assert car.wheel_shapes == kw.get("wheel_shapes", False)
    out = []
    for i in range(420):
        car.set_throttle(0.8 if i < 300 else 0.0)
        car.set_steer(0.4 if 120 <= i < 240 else 0.0)
        car.set_brake(1.0 if i >= 300 else 0.0)
        world.step(DT)
        if i % 30 == 29:
            p, q = car.position, car.quaternion
            out.append((p.x, p.y, p.z, q.x, q.y, q.z, q.w, car.forward_speed))
    return out


def test_a_car_alone_runs_bit_identical_with_wheel_shapes():
    off, on = drive_alone(), drive_alone(wheel_shapes=True)
    assert off[-1][2] > 20.0, "it drove somewhere"
    assert on == off, "a wheel shape must not touch the ground, the car's own body or its mass"


def side_by_side(**kw):
    """Two cars 3 m apart driving forward, steering into each other, and then kept steering into
    each other while they run alongside: the closest centre-to-centre distance across, the highest
    any chassis rose over its ride height, and the fastest any chassis rose."""
    world = flat_world()
    cars = [tp.PhysxVehicle(world, position=tp.Vector3(x, 0.6, 0), **BUGGY, **kw) for x in (0.0, 3.0)]
    for _ in range(60):  # settle on the springs
        world.step(DT)
    ride = [c.position.y for c in cars]
    closest, rise, climb = 9.0, 0.0, 0.0
    prev = list(ride)
    for i in range(360):
        for c, s in zip(cars, (1.0, -1.0)):
            c.set_throttle(0.5)
            c.set_steer(s * (0.25 if i < 90 else 0.12))
        world.step(DT)
        a, b = cars[0].position, cars[1].position
        if abs(a.z - b.z) < 1.0:  # alongside
            closest = min(closest, math.hypot(b.x - a.x, b.y - a.y))
        for k, c in enumerate(cars):
            y = c.position.y
            rise = max(rise, y - ride[k])
            climb = max(climb, (y - prev[k]) / DT)
            prev[k] = y
    return closest, rise, climb


def test_wheels_hit_the_other_cars_wheels_and_nobody_is_launched():
    # Wheel to wheel, side by side the centres stay a track plus a tyre's width apart
    # (1.80 + 0.30 = 2.10 m); a wheel against the other chassis' side holds at 0.9 + 0.15 + 0.85 = 1.90.
    closest_on, rise_on, climb_on = side_by_side(wheel_shapes=True)
    assert closest_on > 1.85, f"the wheels hold the cars apart ({closest_on:.3f} m)"
    assert rise_on < 0.15 and climb_on < 1.0, f"nobody is thrown up (rose {rise_on:.3f} m at {climb_on:.2f} m/s)"


def test_without_wheel_shapes_the_wheels_pass_through_each_other():
    closest_off, rise_off, _ = side_by_side()
    # only the 1.70 m chassis boxes meet: the inner wheels are inside each other
    assert closest_off < 1.80, f"today's behaviour: bodies touch, wheels interleave ({closest_off:.3f} m)"
    assert rise_off < 0.15
