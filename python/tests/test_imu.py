"""IMU sensor — physics-truth + noise-model tests.

Runs headless on the PhysX rigid-body world (no canvas / renderer). Skips on a
build without the omniverse-physx-sdk. All physics assertions use zero-noise
IMUs so the readings are the deterministic physics truth.

Each world uses fixed_timestep == DT and is stepped with DT, so exactly one
physics substep (and therefore one IMU sample) runs per step().
"""
import math

import numpy as np
import pytest

import threepp as tp

pytestmark = pytest.mark.skipif(not tp.HAS_PHYSX, reason="built without the PhysX backend")

DT = 1 / 240  # physics substep == step dt == one IMU sample per step


def _perfect():
    """An all-zero NoiseModel — a noiseless, unbiased sensor."""
    return tp.NoiseModel(seed=0)


def _perfect_imu(node, rate_hz=0.0):
    imu = tp.Imu(node, rate_hz=rate_hz)
    imu.gyro_noise = _perfect()
    imu.accel_noise = _perfect()
    return imu


def _box(size=1.0):
    return tp.Mesh(tp.BoxGeometry(size, size, size), tp.MeshStandardMaterial())


def _mean(samples, attr):
    vs = [getattr(s, attr) for s in samples]
    n = len(vs)
    return tp.Vector3(sum(v.x for v in vs) / n, sum(v.y for v in vs) / n, sum(v.z for v in vs) / n)


def test_resting_body_reads_gravity_reaction():
    """A body at rest on the ground: accel ~ (0, +9.81, 0) (specific force opposes
    gravity), gyro ~ 0."""
    world = tp.PhysxWorld(fixed_timestep=DT)
    floor = tp.Mesh(tp.BoxGeometry(20, 1, 20), tp.MeshStandardMaterial())
    floor.position.set(0, -0.5, 0)  # top face at y=0
    world.add_static(floor)
    box = _box()
    box.position.set(0, 0.5, 0)  # resting on the floor
    world.add(box, density=100)
    imu = _perfect_imu(box)
    world.register_sensor(imu)

    for _ in range(240):  # 1 s
        world.step(DT)

    samples = imu.drain()
    assert len(samples) == 240
    accel = _mean(samples[-60:], "linear_acceleration")  # last 0.25 s (settled)
    gyro = _mean(samples[-60:], "angular_velocity")
    assert accel.x == pytest.approx(0.0, abs=0.05)
    assert accel.y == pytest.approx(9.81, abs=0.1)
    assert accel.z == pytest.approx(0.0, abs=0.05)
    assert gyro.x == pytest.approx(0.0, abs=1e-3)
    assert gyro.y == pytest.approx(0.0, abs=1e-3)
    assert gyro.z == pytest.approx(0.0, abs=1e-3)


def test_free_fall_reads_near_zero():
    """A free-falling body: specific force ~ 0 (weightless)."""
    world = tp.PhysxWorld(fixed_timestep=DT)
    box = _box()
    box.position.set(0, 50, 0)
    world.add(box, density=100)
    imu = _perfect_imu(box)
    world.register_sensor(imu)

    for _ in range(120):
        world.step(DT)

    samples = imu.drain()
    # Skip the first few (finite-difference warm-up); the rest are weightless.
    for s in samples[5:]:
        a = s.linear_acceleration
        mag = math.sqrt(a.x * a.x + a.y * a.y + a.z * a.z)
        assert mag < 0.5, f"free-fall |accel| should be ~0, got {mag}"


def test_constant_spin_reads_gyro():
    """A body spinning at omega about an axis reads gyro ~ omega on that axis."""
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, 0), fixed_timestep=DT)
    box = _box()
    body = world.add(box, density=100)
    body.set_angular_damping(0.0)  # no decay, so gyro holds omega exactly
    omega = 2.0
    body.set_angular_velocity(tp.Vector3(0, omega, 0))
    imu = _perfect_imu(box)
    world.register_sensor(imu)

    for _ in range(120):
        world.step(DT)

    gyro = _mean(imu.drain()[5:], "angular_velocity")
    assert gyro.x == pytest.approx(0.0, abs=1e-3)
    assert gyro.y == pytest.approx(omega, abs=1e-2)
    assert gyro.z == pytest.approx(0.0, abs=1e-3)


def test_lever_arm_centripetal():
    """A sensor offset r from the CoM on a body spinning at omega reads the
    centripetal specific force ~ omega^2 * r pointing toward the rotation axis."""
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, 0), fixed_timestep=DT)  # isolate lever arm
    box = _box()
    body = world.add(box, density=100)
    body.set_angular_damping(0.0)
    omega = 3.0
    body.set_angular_velocity(tp.Vector3(0, omega, 0))
    # Sensor node offset +2 m along the body x axis.
    r = 2.0
    node = tp.Group()
    node.position.set(r, 0, 0)
    box.add(node)
    imu = _perfect_imu(node)
    world.register_sensor(imu)

    for _ in range(120):
        world.step(DT)

    accel = _mean(imu.drain()[10:], "linear_acceleration")
    expected = omega * omega * r  # 18.0
    horizontal = math.sqrt(accel.x * accel.x + accel.z * accel.z)
    assert horizontal == pytest.approx(expected, rel=0.02)
    # Centripetal: points from the sensor back toward the axis, i.e. -x in the
    # (still-aligned) sensor frame.
    assert accel.x == pytest.approx(-expected, rel=0.02)
    assert abs(accel.z) < 0.5


def test_zero_noise_is_deterministic_physics():
    """Two identical zero-noise runs produce bit-identical samples (pure physics)."""

    def run():
        world = tp.PhysxWorld(fixed_timestep=DT)
        box = _box()
        box.position.set(0, 5, 0)
        world.add(box, density=100)
        imu = _perfect_imu(box)
        world.register_sensor(imu)
        for _ in range(40):
            world.step(DT)
        return imu.drain_array()

    a = run()
    b = run()
    assert a.shape == (40, 7)
    assert np.array_equal(a, b)


def test_fixed_seed_noise_is_reproducible():
    """Same seed + same physics -> identical noisy samples; different seed differs."""

    def run(seed):
        world = tp.PhysxWorld(fixed_timestep=DT)
        box = _box()
        box.position.set(0, 5, 0)
        world.add(box, density=100)
        imu = tp.Imu(box)
        imu.gyro_noise = tp.NoiseModel(white_noise_density=tp.Vector3(0.02, 0.02, 0.02), seed=seed)
        imu.accel_noise = tp.NoiseModel(white_noise_density=tp.Vector3(0.2, 0.2, 0.2), seed=seed + 1)
        world.register_sensor(imu)
        for _ in range(40):
            world.step(DT)
        return imu.drain_array()

    a = run(1234)
    b = run(1234)
    c = run(9999)
    assert np.array_equal(a, b), "same seed must reproduce"
    assert not np.array_equal(a, c), "different seed must differ"
    # Noise actually perturbs the reading away from clean physics.
    assert not np.allclose(a[:, 1:], 0.0)


def test_latest_and_drain_semantics():
    """latest() survives drain(); drain() empties the buffer."""
    world = tp.PhysxWorld(fixed_timestep=DT)
    box = _box()
    box.position.set(0, 5, 0)
    world.add(box, density=100)
    imu = _perfect_imu(box)
    world.register_sensor(imu)
    for _ in range(10):
        world.step(DT)

    assert imu.available == 10
    last = imu.latest()
    assert last is not None
    samples = imu.drain()
    assert len(samples) == 10
    assert imu.available == 0
    # latest() persists after drain; matches the final drained sample.
    still = imu.latest()
    assert still is not None
    assert still.t == pytest.approx(samples[-1].t)


def test_rate_gating_subsamples():
    """rate_hz below the physics rate emits fewer samples (rate-gated)."""
    world = tp.PhysxWorld(fixed_timestep=DT)
    box = _box()
    box.position.set(0, 5, 0)
    world.add(box, density=100)
    imu = _perfect_imu(box, rate_hz=60.0)  # 1/4 of the 240 Hz physics rate
    world.register_sensor(imu)
    for _ in range(240):  # 1 s
        world.step(DT)

    n = imu.available
    assert 55 <= n <= 65, f"expected ~60 samples at 60 Hz over 1 s, got {n}"


def test_register_without_body_raises():
    """Attaching to a node with no managed rigid body raises at registration."""
    world = tp.PhysxWorld(fixed_timestep=DT)
    orphan = tp.Group()  # never added to the world
    imu = _perfect_imu(orphan)
    with pytest.raises((ValueError, RuntimeError)):
        world.register_sensor(imu)
