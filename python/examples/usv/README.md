# The Mariner USV

A model of Maritime Robotics' Mariner sea drone (MKII) for the USV demos, built the
way the TP-1 rover is (`../rover/`): one spec file, a headless-Blender generator, and a
`.glb` that threepp loads with `GLTFLoader`. The Otter X catamaran is built the same way;
see [The Otter X](#the-otter-x) at the end.

| file | what it is |
|---|---|
| `mariner_spec.json` | every number: brochure figures, the lines fitted to the vendor's renders and photo, the sensor tower, the waterjet, the mass budget, the materials |
| `build_mariner_blender.py` | builds the geometry (numpy), exports `mariner.glb` through Blender, and integrates the hydrostatics into `mariner_hydro.json` |
| `usv_ocean.py` | the demo: she floats on the FFT ocean on her own buoyancy tables and runs on a steerable jet (`--boat otterx`: the Otter X on her pods) |
| `mariner.glb`, `mariner_hydro.json` | generated, not committed |

## Build

From this folder, once (about 45 s; `blender` is your Blender executable):

```
blender --background --factory-startup --python build_mariner_blender.py -- --spec mariner_spec.json --out mariner.glb
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

## The Otter X

Maritime Robotics' Otter X, a 4.6 m electric catamaran on two rim-drive azimuth pods.

| file | what it is |
|---|---|
| `otterx_spec.json` | every number: brochure figures, the lines and fittings fitted to the vendor's side and front renders, the mass budget, the materials |
| `build_otterx_blender.py` | builds the geometry (numpy) with the Mariner generator's mesh, hydrostatics and export code, exports `otterx.glb` through Blender, and integrates `otterx_hydro.json` |
| `otterx.glb`, `otterx_hydro.json` | generated, not committed |

From this folder, once (about 90 s, most of it the hydrostatics):

```
blender --background --factory-startup --python build_otterx_blender.py -- --spec otterx_spec.json --out otterx.glb
```

`python build_otterx_blender.py --hydro-only` rebuilds just `otterx_hydro.json`. The build
fails if the design draft leaves the brochure's 400 mm by more than its +-100 mm, if the
trim passes 2 deg, or if any buoyant part is not closed and consistently wound.

Frame: X forward, Y up, Z starboard, metres. The origin is on the centreline, on the
pontoon keel line, halfway along the 4.6 m length overall; the skegs reach y = -0.22.
The root node `otterx` sits at identity.

- `pontoons` (both demihulls, skegs, the gondola), `body`, `deck_fittings`, `gantry`: meshes.
- `thruster_port`, `thruster_stbd`: the pods, turning about local +Y (positive turns the
  thrust line toward port and the boat to starboard). Each has a `thruster_<side>_rotor`
  that spins about local +X and a `thrust_<side>` empty at the duct centre, thrust along
  local +X.
- Sensor empties (look down local -Z, local +Y up): `camera_main`, `radar`, `gnss_port`,
  `gnss_stbd` (1.38 m athwartships baseline), `sat_compass`, `ais_vhf_antenna`,
  `mimo_antenna`, `lte_antenna`, `imu`, `transducer`, `mbes_mount`; lights
  `nav_light_port`, `nav_light_stbd`, `ram_light_top`, `ram_light_mid`, `ram_light_bottom`
  (red-white-red), `floodlight`.

`otterx_hydro.json` has the same layout as the Mariner's: lightship (950 kg), survey
(1030 kg, the design condition) and full load (1180 kg), and Bonjean tables at 23
stations x 78 water heights. At the survey load she floats at 0.365 m moulded draft,
0.06 deg by the head, GM_T 2.57 m.

From the vendor: principal dimensions, draft, dry weight (< 1000 kg), hull material,
propulsion type and power, battery count, the sensor and radio list. The brochure gives
L x W x H 4600 x 2213 x 1900 mm; the product page swaps width and height, and the front
render's gantry measures 2.17 m across. Fitted to the vendor's side and front renders:
the lines, the body blocks, the gantry and the fittings. The side render is a mild
perspective from above and astern; projected through that camera, the model's silhouette
overlaps the render at 0.957 IoU. Assumed, and marked in the spec: the mass budget and
battery placement, the loading conditions, thrust, sensor models, the transducer and
gondola. The vendor photos show a production boat carrying a customer's bow pole; the
model follows the renders.

### Otter X demo

```
python usv_ocean.py --boat otterx            # drive: W/S throttle (S past zero: astern),
                                             # A/D steer both pods, Q/E turn on the spot,
                                             # Space stop, C mast camera, X hull
python usv_ocean.py --boat otterx --calm --drop
python usv_ocean.py --boat otterx --record 36 --out otterx.mp4
```

The strip model and the rigid body are the Mariner's; the boat-specific half is its own
block in `usv_ocean.py`. Both pods steer to the helm's angle (+-35 deg) and push along
that line from the duct centres; Q/E moves 60 % of full thrust from one pod to the other.
Thrust falls from 900 N per pod at bollard to 60 % of that at the top speed, and astern
gives 60 %. The resistance of a displacement catamaran (linear and quadratic terms and a
mild wave-making hump at Froude number 0.45) is tuned so full throttle makes 8 kn in flat
water, an assumed figure until the vendor publishes one. Each pod loses thrust as the
aft sections on its side leave the water. In flat water she settles on the solver's pose
(0.582 m maximum draft against 0.583, -0.09 deg trim against -0.06). In the scripted run
she reaches 7.6 kn in the default sea, turns at about 20 deg/s on full helm, stops from
5 kn in 4 s on the pods astern, and turns on the spot at about 18 deg/s.

The ocean's hull footprint is a single monohull plan form, so the water in the tunnel
between the pontoons lies flat on her waterline plane.
