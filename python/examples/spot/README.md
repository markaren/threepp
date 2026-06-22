# Spot — deploying an Isaac Lab policy in threepp

[`spot_deploy.py`](spot_deploy.py) takes a **Boston Dynamics Spot URDF** and a
**locomotion policy trained in Isaac Lab**, and makes Spot walk in threepp's
PhysX — a sim-to-sim transfer with **no retraining**. Drive it with the **arrow
keys / numpad**; the camera follows.

| | | |
| --- | --- | --- |
| **forward** UP / NUM 8 | **back** DOWN / NUM 2 | **turn L** N / NUM 7 |
| **strafe L** LEFT / NUM 4 | **strafe R** RIGHT / NUM 6 | **turn R** M / NUM 9 |

![Spot walking under the Isaac policy](spot.png)

```sh
python spot_deploy.py                  # interactive — assets download on first run
python spot_deploy.py --shot out.png
```

On first run the assets download once to `~/.cache/threepp/spot` (re-runs reuse
them); pass `--assets <folder>` to use your own copy instead. They come from
Isaac Sim + the Spot SDK:

| file | source |
| --- | --- |
| `spot_policy.pt` | Isaac Lab **Spot velocity** policy (TorchScript) — *the only file the demo needs* |
| `spot_env.yaml` | the Isaac env config (the obs/action contract; baked into the script) |
| `model.urdf` + `link_models/` | Boston Dynamics Spot SDK URDF (cached for the URDF importer) |

Needs a **PhysX-enabled** threepp build (`tp.HAS_PHYSX`) and **torch**.

## How it works

The whole trick is reproducing Isaac Lab's observation/action contract exactly,
then letting PhysX do the rest:

- **Observation (48-d, this order):** base linear velocity, base angular velocity,
  projected gravity (all body-frame), the velocity command `[vx, vy, wz]`,
  joint positions relative to the default pose, joint velocities, and the last
  action — in Isaac's joint order (`hx`×4, `hy`×4, `kn`×4).
- **Action:** `target_q = default_pose + 0.2 · action`, applied as PD position
  targets (stiffness 60, damping 1.5; effort 45 hips / 115 knees).
- **Physics:** Spot is a reduced-coordinate `Articulation` stepped at 0.002 s with
  decimation 10 (50 Hz policy), in a **Z-up** world to match the URDF.

## Sim-to-sim caveats (why it transfers anyway)

threepp and Isaac Lab **both run PhysX 5**, which is what makes this work despite
the approximations:

- The URDF carries **no inertials or collision**, so link masses are approximated
  and each link gets a **Box/Capsule** collider for the physics (the articulation
  API takes primitive shapes). Those primitives are hidden — each link's **visual
  mesh** (`link_models/*.obj`) is parented under its collider, so Spot *renders* as
  the real robot while the capsules drive the simulation.
- The knee's **remotized** actuator is treated as a plain PD.

With the obs/action contract exact, the policy still produces a clean forward trot
at ~0.9 m/s for a 1.0 m/s command. Tune `MASS` / `GAINS` / `Z0` at the top of the
script for other robots or policies.
