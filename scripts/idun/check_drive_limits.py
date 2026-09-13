"""Does this build honour drive_limits_are_forces? CPU PhysX only, no GPU, a few seconds.

A 6 kg, 1 m arm held horizontal by a position drive (the S0b probe's arm): bisect the smallest max_force
that still holds it where an unlimited drive does. With the flag ON that threshold is a torque, ~50 N*m and
the same at every substep dt. With it OFF max_force is a per-substep impulse, so the threshold scales with
dt (4x from 0.005 to 0.020 s). Exit 0 on PASS, 1 on FAIL, 2 when the module predates the flag.

    python scripts/idun/check_drive_limits.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python"))
import threepp as tp  # noqa: E402


def settle_angle(max_force, dt, dlf, steps=400):
    world = tp.PhysxWorld(gravity=tp.Vector3(0, 0, -9.81), fixed_timestep=dt, max_substeps=64, tgs_pcm=True)
    art = world.create_articulation(fixed_base=True, solver_position_iterations=12,
                                    disable_self_collision=True, drive_limits_are_forces=dlf)
    bm = tp.Mesh(tp.BoxGeometry(0.1, 0.1, 0.1), tp.MeshStandardMaterial())
    bm.position.set(0.0, 0.0, 1.0)
    base = art.add_link(bm, parent=None, density=1000.0)
    L, R, mass = 1.0, 0.06, 6.0
    am = tp.Mesh(tp.CapsuleGeometry(R, L), tp.MeshStandardMaterial())
    am.position.set(0.5, 0.0, 1.0)
    am.rotation.z = np.pi / 2
    vol = np.pi * R * R * L + 4.0 / 3.0 * np.pi * R ** 3
    art.add_link(am, parent=base, density=mass / vol, axis=(0, 1, 0), anchor=(0.0, 0.0, 1.0),
                 lower=-2.0, upper=2.0, stiffness=90.0, damping=1.5, max_force=max_force, drive_target=0.0)
    art.finalize()
    if bool(art.drive_limits_are_forces) != dlf:
        raise RuntimeError("the flag did not reach the articulation")
    tgt = np.zeros(1, np.float32)
    for _ in range(steps):
        art.set_drive_targets(tgt)
        world.step(0.02)
    return float(art.joint_positions()[0])


def threshold(dt, dlf):
    ref = settle_angle(1e6, dt, dlf)
    lo, hi = 1e-4, 1e3
    for _ in range(22):
        mid = np.sqrt(lo * hi)
        if abs(np.degrees(settle_angle(mid, dt, dlf)) - np.degrees(ref)) < 1e-3:
            hi = mid
        else:
            lo = mid
    return hi


def main():
    try:
        settle_angle(1e6, 0.02, True, steps=1)
    except TypeError as e:
        print(f"OLD MODULE: create_articulation has no drive_limits_are_forces ({e})")
        return 2
    print(f"threepp: {tp.__file__}")
    th = {(dlf, dt): threshold(dt, dlf) for dlf in (True, False) for dt in (0.020, 0.005)}
    for dlf in (True, False):
        a, b = th[(dlf, 0.020)], th[(dlf, 0.005)]
        print(f"flag {'ON ' if dlf else 'OFF'}  threshold max_force  dt 0.020: {a:8.3f}   dt 0.005: {b:8.3f}   ratio {a / b:5.2f}")
    on_ratio = th[(True, 0.020)] / th[(True, 0.005)]
    off_ratio = th[(False, 0.020)] / th[(False, 0.005)]
    ok = abs(on_ratio - 1.0) < 0.1 and 40.0 < th[(True, 0.005)] < 60.0 and 3.5 < off_ratio < 4.5
    print("PASS: flag ON max_force is a torque, flag OFF is still the per-substep impulse" if ok else
          "FAIL: expected ON ratio ~1 at ~50 N*m and OFF ratio ~4")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
