"""IMU sensor demo — a box dropped onto the floor with an IMU attached off-CoM.

    python imu_demo.py

Headless (no window / renderer): PhysX physics is pure CPU. We drop a box onto a
static floor, sample an Imu bolted slightly off the centre of mass, and print the
physics-truth checks a real IMU must satisfy:

  * free fall   -> specific force ~ 0            (weightless)
  * at rest     -> specific force ~ (0, +9.81, 0) (opposes gravity, threepp is Y-up)
  * gyro        -> ~ 0 for this non-spinning drop

The IMU rides the scene graph: attach it to a node, register it with the world,
and it is sampled once per physics substep (or at rate_hz). Read the buffer with
latest() / drain() / drain_array() — no polling loop needed. See the "Sensors"
section of the README for the units / frames / noise model.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import threepp as tp

if not tp.HAS_PHYSX:
    print("This build has no PhysX backend. Configure the module with the vcpkg "
          "toolchain (the omniverse-physx-sdk) to enable it.")
    sys.exit(0)

DT = 1 / 240  # one physics substep == one IMU sample per step()


def perfect_noise():
    """All-zero NoiseModel -> a noiseless, unbiased sensor (clean truth numbers)."""
    return tp.NoiseModel(seed=0)


def mean(samples, attr):
    vs = [getattr(s, attr) for s in samples]
    n = max(len(vs), 1)
    return (sum(v.x for v in vs) / n, sum(v.y for v in vs) / n, sum(v.z for v in vs) / n)


def fmt(v):
    return f"({v[0]:+7.3f}, {v[1]:+7.3f}, {v[2]:+7.3f})"


world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0), fixed_timestep=DT)

# Static floor (top face at y=0).
floor = tp.Mesh(tp.BoxGeometry(40, 1, 40), tp.MeshStandardMaterial())
floor.position.set(0, -0.5, 0)
world.add_static(floor)

# A box dropped from a height. Bottom starts at y=5.5, lands after ~1.06 s.
box = tp.Mesh(tp.BoxGeometry(1, 1, 1), tp.MeshStandardMaterial())
box.position.set(0, 6, 0)
world.add(box, density=200)

# Mount the IMU on a node offset from the CoM (exercises the lever-arm path;
# harmless for a non-spinning drop where omega = alpha = 0).
mount = tp.Group()
mount.position.set(0.3, 0.2, 0.1)
box.add(mount)

imu = tp.Imu(mount)                 # rate_hz=0 -> sample every substep
imu.gyro_noise = perfect_noise()    # clean truth for the table; MEMS-class by default
imu.accel_noise = perfect_noise()
world.register_sensor(imu)

DURATION = 3.0
steps = int(DURATION / DT)
print(f"Dropping a box from y=6 onto the floor; IMU mounted at "
      f"{fmt((0.3, 0.2, 0.1))} off the CoM.", flush=True)
print(f"Stepping {steps} substeps at {1/DT:.0f} Hz ({DURATION:.0f} s of sim)...", flush=True)

for _ in range(steps):
    world.step(DT)

samples = imu.drain()
n = len(samples)
rate = n / world.sim_time if world.sim_time > 0 else 0.0

# Classify by sim time: free fall before impact (~1.06 s), rest after settling.
free_fall = [s for s in samples if s.t < 0.9]
at_rest = [s for s in samples if s.t > 2.5]

ff_accel = mean(free_fall, "linear_acceleration")
rest_accel = mean(at_rest, "linear_acceleration")
rest_gyro = mean(at_rest, "angular_velocity")
ff_mag = math.sqrt(sum(c * c for c in ff_accel))

print("", flush=True)
print("  IMU summary (specific force in m/s^2, sensor frame; angular velocity rad/s)", flush=True)
print("  " + "-" * 66, flush=True)
print(f"  {'phase':<14}{'accel (x, y, z)':<28}{'|accel|':>10}", flush=True)
print("  " + "-" * 66, flush=True)
print(f"  {'free fall':<14}{fmt(ff_accel):<28}{ff_mag:>10.3f}   (expect ~0)", flush=True)
rest_mag = math.sqrt(sum(c * c for c in rest_accel))
print(f"  {'at rest':<14}{fmt(rest_accel):<28}{rest_mag:>10.3f}   (expect ~9.81)", flush=True)
print("  " + "-" * 66, flush=True)
print(f"  rest gyro:        {fmt(rest_gyro)}   (expect ~0)", flush=True)
print(f"  samples drained:  {n}", flush=True)
print(f"  effective rate:   {rate:.1f} Hz over {world.sim_time:.2f} s of sim", flush=True)

# A quick assertion so the demo doubles as a smoke check.
assert ff_mag < 0.5, "free fall should be weightless"
assert abs(rest_accel[1] - 9.81) < 0.2, "rest should read +g on Y"
print("", flush=True)
print("OK - free fall weightless, rest reads gravity reaction.", flush=True)
