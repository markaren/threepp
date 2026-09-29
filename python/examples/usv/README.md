# The Mariner USV

A model of Maritime Robotics' Mariner sea drone (MKII) for the USV demos, built the
way the TP-1 rover is (`../rover/`): one spec file, a headless-Blender generator, and a
`.glb` that threepp loads with `GLTFLoader`.

| file | what it is |
|---|---|
| `mariner_spec.json` | every number: brochure figures, the lines fitted to the vendor's renders and photo, the sensor tower, the waterjet, the mass budget, the materials |
| `build_mariner_blender.py` | builds the geometry (numpy), exports `mariner.glb` through Blender, and integrates the hydrostatics into `mariner_hydro.json` |
| `usv_ocean.py` | the demo: she floats on the FFT ocean on her own buoyancy tables and runs on a steerable jet |
| `mariner.glb`, `mariner_hydro.json` | generated, not committed |

## Build

From this folder, once (about 45 s):

```
"C:\Program Files\Blender Foundation\Blender 5.0\blender.exe" --background --factory-startup --python build_mariner_blender.py -- --spec mariner_spec.json --out mariner.glb
```

`python build_mariner_blender.py --hydro-only` rebuilds just `mariner_hydro.json` without
Blender. The build fails if the design draft leaves the brochure's 500 mm by more than
6 cm, if the trim passes 2 deg, or if any closed part is not closed and consistently wound.

## Frame and nodes

X forward, Y up, Z starboard, metres. The origin is on the centreline, on the canoe-body
baseline, halfway along the 5.98 m length overall; the keel shoe and gondola reach
y = -0.082. The root node `mariner` sits at identity.

- `hull`, `collar`, `deck_fittings`, `mast`: meshes.
- `jet` > `jet_steering` (turns about local +Y, +-27 deg; positive swings the exit to
  starboard and turns the boat to starboard) and `jet_reverse_bucket` (local +Z, 0..40 deg
  lowers it across the jet).
- `jet_thrust`, `bow_thruster`: empties at the thrust points, identity rotation.
- Sensor empties (look down local -Z, local +Y up): `camera_main`, `radar`, `gnss_fore`,
  `gnss_aft` (1.19 m baseline), `ais_vhf_antenna`, `lte_antenna_1`, `lte_antenna_2`, `imu`,
  `transducer` (gondola flat, looking down), `moonpool`; lights `nav_light_port`,
  `nav_light_stbd`, `allround_light`, `floodlight_port`, `floodlight_stbd`. Field of view,
  resolution and the like ride along as glTF extras.

## Hydrostatics (`mariner_hydro.json`)

Integrated from the buoyant meshes column by column (the winding number, so the collar
and the hull overlapping count once) and checked against the exact volume (0.09 %).

- `conditions`: lightship, departure (design) and full load, each solved for heave and
  trim: draft, trim, LCB, KB, waterplane area, GM_T, GM_L and the waterline outline.
  Departure (2168 kg) floats at 0.465 m maximum draft, 1.3 deg by the stern.
- `bonjean`: the starboard half-section's immersed area and centroid at 30 stations x 71
  water heights. A strip model sums both halves at the local water height on each side of
  each station; `usv_ocean.py` does exactly that, and in flat water it settles on the
  solver's pose (0.467 m, +1.31 deg).

## Demo

```
python usv_ocean.py                          # drive: W/S throttle, A/D steer, Q/E bow thruster,
                                             # R reverse bucket, Space stop, C mast camera, X hull
python usv_ocean.py --calm --drop            # flat water, start displaced, settle
python usv_ocean.py --record 36 --out usv.mp4
```

Hydrostatics come from the tables; the manoeuvring (resistance with a planing hump,
planing lift that grows with trim and moves aft as the bow rises and toward the low side
as she rolls, yaw and sway damping, banking) is a lumped sketch tuned so full throttle
makes the brochure's 24 kn with about 4 deg of running trim.

## What is measured and what is assumed

From the vendor: principal dimensions, dry weight, draft, hull material, propulsion type,
bow thruster, fuel, speed. Fitted by eye and by silhouette to the vendor's side and front
renders and top-down photo: every line and fitting (the side silhouette overlaps the
vendor render at 0.905 IoU). Assumed, and marked in the spec: sensor models and fields of
view, thrust figures, the mass budget, the moonpool position. No vendor marks: the livery
is the orange panel only (`livery.side_text` is empty).
