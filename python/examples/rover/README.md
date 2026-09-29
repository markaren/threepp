# TP-1 rover on granular sand

A four-wheeled, VIPER-class rover of our own design, driven over MLS-MPM grains
(`threepp.granular_mpm`).

| File | What it is |
| --- | --- |
| `warp_rover_demo.py` | TP-1 climbs a tilting sand bed until the grains stop it (interactive, `--shot`, `--record`). |
| `warp_rover_slope.py` | One wheel's slip sweep against NASA's single-wheel data, and the four-wheel real-time cost. |
| `rover_spec.json` | The rover's geometry, masses and CoM. Both scripts read it. |
| `build_rover_blender.py` | Builds `rover.glb` from the spec in headless Blender. |

`rover.glb` is not in git. Build it once:

    blender --background --factory-startup --python build_rover_blender.py -- --spec rover_spec.json --out rover.glb

Then, from the repo root:

    PYTHONPATH=python python python/examples/rover/warp_rover_demo.py
