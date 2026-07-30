# Running a trained policy inside the threepp editor

A policy trained by `threepp.rl` outside the editor, deployed as an editor behaviour script.
Verified with the Spot terrain policy (`spot_steps.pt`): **8.03 m in 10.04 s at a commanded
0.8 m/s, upright throughout** — inside the editor's play session, driven by a Python script.

Nothing in the runner is Spot-specific. The robot-specific parts are a *bundle* (exported from
training) and a *URDF that describes the trained plant*; everything else is shared.

```
export_bundle.py      trained checkpoint  ->  bundle/ (policy.npz + spec.json)
make_spot_urdf.py     the plant that only existed in code  ->  a URDF a host can import
make_scene.py         bundle + urdf  ->  an editor scene (.json)
policy_runner.py      spec.json  ->  observation, action, joint targets   [numpy only]
spot_editor_script.py the editor adapter + the behaviour script
ab_check.py           proves the bundle reproduces torch, on the training plant
```

## The four steps

```bash
python export_bundle.py         # -> bundle_spot_steps/
python ab_check.py              # PARITY + WALK must both pass before trusting the bundle
python make_spot_urdf.py        # -> ~/.cache/threepp/spot/spot_physics.urdf
python make_scene.py            # -> spot_policy.json
threepp_editor spot_policy.json # press Play (or: --play --frames=700)
```

## Why a bundle and not just the checkpoint

The editor embeds CPython, and a `.pt` needs torch. So the export writes weights as `.npz` and a
forward pass in numpy — the A/B check confirms this is not an approximation:

```
[parity] 120 ticks   max |obs diff| = 0.000e+00   max |action diff| = 4.768e-07
[walk]   400 ticks (8.0s) at vx=0.8: travelled +6.42 m  (+0.80 m/s)  upright 1.000
```

The observation is **bit-identical** to the shipped player's, and the action differs only by
float32 rounding.

## Why a spec and not just weights

A policy is meaningless without the exact observation it was trained on. `spec.json` declares that
observation as an ordered list of *terms* (`base_lin_vel_body`, `projected_gravity`, `gait_clock`,
`height_scan`, …), plus the action decoding, the control rate, and the PD gains the plant needs.
The training env writes it, because the training env is the only thing that knows. The runner
interprets it. Adding a robot means exporting a spec, not editing the runner.

## What is host-specific: the adapter

`policy_runner.Adapter` is the whole porting surface — joint names, joint state, root pose and
velocity, the world's gravity direction, and `set_drive_targets`. Two exist: `DeployAdapter`
(`ab_check.py`, a `tp.PhysxWorld` articulation) and `EditorAdapter` (`spot_editor_script.py`, the
editor's `articulation_from_object` handle).

Two things the adapter makes go away:

* **Up axis.** The trainer's world is Z-up; the editor's PhysX world is Y-up. The runner never
  assumes either — it takes up from the world's own gravity and builds the heading frame in the
  plane perpendicular to it. So the numbers the policy sees are identical in both, and the only
  adjustment is a -90° X rotation on the robot node (`make_scene.py`).
* **Joint order.** Mapped **by name**, never by position. This is not theoretical: the training
  plant reports joints per-leg (`fl.hx, fl.hy, fl.kn, fr.hx, …`) and needs the permutation
  `[0,3,6,9,1,4,7,10,2,5,8,11]`, while the editor's URDF articulation happens to report them
  type-grouped — the policy's own order — so the map is the identity. Same policy, same bundle,
  two different orders, no special case.

## The plant is the hard part, not the policy

Spot's trained plant existed **only as code**: `spot_deploy.build_spot` hand-builds box/capsule
colliders and per-link masses, because Boston Dynamics' URDF ships visuals and kinematics with
**no `<collision>` and no `<inertial>`**. A URDF-driven host cannot import that. `make_spot_urdf.py`
injects exactly those two things — from `spot_deploy`'s own constants — into the real URDF, leaving
its visuals, joint origins, axes and per-leg limits untouched. Result: 13 links, 28.0 kg, the
capsules the policy was trained against, and Spot still renders as Spot.

**Generalising:** if a policy's plant lives in training code, that code is the source of truth and
the URDF should be *generated from it*. A hand-written URDF that merely looks similar is the single
most likely reason a transferred policy fails.

## Fidelity gaps that did NOT stop it (and one that would)

The editor's play session is not the training rig, and the policy tolerated every difference:

| | training | editor | effect |
|---|---|---|---|
| physics step | 0.005 s, TGS+PCM | 1/60 s, PhysX default | none measured |
| per-joint effort | hx/hy 45, kn 115 | one uniform `maxforce` | none measured (set to 115) |
| world up | Z | Y | none (handled in the runner) |
| collider source | code | generated URDF | none (same shapes/masses) |

Worth naming as real editor gaps rather than glossing them:

1. **No script-visible scene raycast.** `height_at` is stubbed to a flat plane, so only the
   flat-ground policy runs. The 45-cell terrain scan needs a height query — one binding
   (`PxScene::raycast` is already there) unlocks the stairs and heightfield policies.
2. **`ArticulationConfig` gains are uniform.** Per-joint stiffness/damping/effort cannot be
   authored, so a URDF's own `<limit effort>` is ignored in favour of one number.
3. **The physics rate is not authorable.** `PhysicsPlaySession` is constructed with defaults
   (1/60, no TGS/PCM); a policy trained at 0.005 with TGS+PCM has no way to ask for its own
   contact model. It worked here; it will not always.

## Watch it

```bash
threepp_editor spot_policy.json          # then press Play
threepp_editor spot_policy.json --play --frames=700   # headless; writes spot_editor_trace.csv
```

The script exposes `vx`, `vy`, `wz` as inspector parameters, so the robot is steerable from the
inspector while it plays. `spot_editor_trace.csv` logs position, height, tilt and measured forward
speed per control tick; `spot_editor_trace_diag.txt` records the contract that was resolved.

## Notes

* The bundle is torch-free at run time; `export_bundle.py` and `ab_check.py` need torch.
* `ab_check.py` needs a PhysX-enabled `threepp` native module. It lives in the main checkout's
  `python/threepp` as a build artifact; `THREEPP_DLL_DIR` points the DLL loader at it.
* `make_spot_urdf.py --collision-axis` exists because threepp's URDF loader rotates **visual**
  cylinders into URDF's Z convention but not **collision** ones (they become Y-aligned threepp
  capsules). The default `threepp` matches today's loader; `urdf` is spec-correct for when that
  is fixed. Masses, radii and lengths are identical either way.
