# The Mariner USV

A model of Maritime Robotics' Mariner sea drone (MKII) for the USV demos, built the
way the TP-1 rover is (`../rover/`): one spec file, a headless-Blender generator, and a
`.glb` that threepp loads with `GLTFLoader`. The Otter X and the Otter catamarans are built
the same way; see [The Otter X](#the-otter-x) and [The Otter](#the-otter) at the end.

| file | what it is |
|---|---|
| `mariner_spec.json` | every number: brochure figures, the lines fitted to the vendor's renders and photo, the sensor tower, the waterjet, the mass budget, the materials |
| `build_mariner_blender.py` | builds the geometry (numpy), exports `mariner.glb` through Blender, and integrates the hydrostatics into `mariner_hydro.json` |
| `../build_common.py` | what every generator shares: the mesh accumulator and its primitives, the hydrostatic integrator, the Blender export |
| `../rigs/usv_rig.py` | the boat for any scene: `StripHull` (her buoyancy tables and the rigid body), her drive (`Waterjet`, `AzimuthPods`, `FixedPods`: thrust, resistance, helm, actuator nodes, foam), `Wake` (her footprint and wake on the ocean) and `Wash` (her propulsors' races, her hull's lane and her waves on the water, in the ocean's wake field); see [In another scene](#in-another-scene) |
| `usv_ocean.py` | the demo on `usv_rig.py`: she floats on the FFT ocean on her own buoyancy tables and runs on a steerable jet (`--boat otterx`: the Otter X on her pods; `--boat otter`: the Otter) |
| `usv_radar_sonar.py` | her radar and her forward-looking sonar in Alesund harbour on the real terrain: see [Radar and sonar](#radar-and-sonar) |
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
  each station; `usv_rig.StripHull` does exactly that, and in flat water it settles on the
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

## Radar and sonar

```
python usv_radar_sonar.py                           # the loop on autopilot; W/S/A/D take the helm
python usv_radar_sonar.py --rain --array             # a squall over the basin, a 4 ft open array
python usv_radar_sonar.py --shot 12 --out rs.png     # headless: the frame, both displays, their labels,
                                                     # and the ground-truth report (below); it first
                                                     # lets the terrain stream in round the start (~30 s)
```

The Mariner runs a loop round the harbour basin south of Alesund town on the 14 m depth
contour, on the real terrain (`geodata/aalesund`: Kartverket heights, OpenStreetMap buildings,
quays and moored boats, synthetic bathymetry by distance to shore). She carries two sensors,
both C++ in the library (`include/threepp/helpers/MarineRadar.hpp`, `ImagingSonar.hpp`), both
fired through the renderer's ray tracer, and both doing the signal physics rather than drawing
a depth map:

- **Radar** on her mast (the spec's 20 in radome: 5.2 x 25 deg, 24 rpm; `--array` for a 4 ft
  open array at 1.8 deg). Every spoke is the radar equation over the rays of its slot: the
  NRL sea-clutter model (Gregers-Hansen & Mital 2012) at the mean grazing angle, spread over the
  rendered waves by their tilt; per-material backscatter (Lambert + a Kirchhoff facet glint,
  corner reflectors as a material); the two-ray propagation factor over the sea, so low targets
  fade with range; rain volume echoes and attenuation (Marshall-Palmer); the antenna's two-way
  azimuth pattern with its sidelobes; kTBF noise; gain, sea (STC) and rain (FTC) controls. Its
  rays go out level whatever she pitches and rolls (the vertical pattern tilts with her), densest
  just below the horizon, where the sea is, and just above it, where masts and reflectors taller
  than her antenna stand; surfaces inside the blind zone still hide what lies behind them. The
  scope is a PPI with afterglow and trails, head up or north up (N).
- **Forward-looking imaging sonar** on her gondola, the hull's deepest point (an Oculus M750d at
  750 kHz, tilted 24 deg down). The sonar equation with Francois-Garrison absorption and Mackenzie sound speed;
  Lambert + facet-glint seabed scattering; coherent beamforming through a tapered 512-beam
  array, so speckle, sidelobe arcs and curved range lines come out of the physics; the chirp's
  matched filter; volume reverberation; noise after the array gain; the sea surface as a
  mirror (image-source multipath, coherent only in calm water). The fold runs on a worker
  thread. Along the north shore she passes a lost Otter X lying on its side on the seabed,
  with rocks around it: watch for the echoes and their acoustic shadows.

Every radar range cell and every sonar (beam, bin) carries the stable instance id of what made
its echo: L colours both displays by it. That is the renderer's ground truth, not a detector.

**Checked against what is known.** `tests/extras/EchoSensors_test.cpp` pins the physics
without a GPU: the beam patterns' widths and sidelobes, Mackenzie's own check value for the
sound speed (1550.744 m/s), Francois-Garrison absorption against the independent Ainslie &
McColm fit, the NRL worked value, two-ray lobing, the surface-multipath ghost, determinism
across thread counts, and, end to end, the **calibration sphere**: a smooth sphere in front of
each sensor must return the sonar equation (TS = 20 log10(a / 2)) and the radar equation
(RCS = pi a^2) through the whole chain of rays, beam pattern, matched filter and noise. Both
land within half a dB.

**Checked against the scene.** `--shot` also scores the last radar revolution and the last
ping against truth that never comes from the sensors' rays (`rs_truth.py`): the terrain's
height field and the poses the demo gave its targets. For the radar: per spoke, the first land
echo against the first land on the height field along that bearing (allowing for a nearer
headland smeared in by the beam); every target's strongest echo against its true range and
bearing. For the sonar: per beam, the seabed's leading edge against where the vertical fan
first meets the height field; the wreck's and the rocks' strongest echoes against their true
range and bearing. The numbers go to `_truth.json`, and `_radar_bscan.png` /
`_sonar_bscan.png` draw the truth over the echoes. The tolerances are the sensors' resolution
and the targets' size, not tuned to pass. The shot first lets the terrain stream in round the
start: GeoScene begins with 2 km tiles and refines a level at a time, and on a coarse tile a
quay wall is a ramp that reaches the sea a few metres out. The radar draws that ramp, as it
should, and the check against the fine height field then reads it as land over water. The
survey's buildings, pier decks and moored boats carry their own ids (61000-61002) so the
report can tell a pier from the shore. A target that is not seen is checked for why: a radar
target with no line of sight over the height field, or a sonar target outside the vertical
fan, is reported as not expected. Her antenna is only 1.8 m above the water: inside the
harbour, quays and piers taller than that hide what lies behind them, and low hulls hide
behind the waves at range. A real radar on a boat this size does the same.

What is assumed: the scattering strengths are class values, not a calibration of any seabed,
hull or quay; the radar is flat-earth with no ducting and no target-to-target bounces; the
sonar's surface-bounce legs assume an unobstructed image path. Nothing here is compared with a
real recording yet.

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
| `build_otterx_blender.py` | builds the geometry (numpy) with `../build_common.py`'s mesh, hydrostatics and export code, exports `otterx.glb` through Blender, and integrates `otterx_hydro.json` |
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
gondola. The head is two blocks with a pole slot between them on the centreline, open
forward, upward and down through the tunnel roof (`body.head_opening`): the front render
shows it as the grey channel between the white panels, and the vendor photo shows a survey
pole standing in it. The model has the slot and no pole; `mbes_mount` is in it, and
`camera_main` stands on the bridge behind it.

### Otter X demo

```
python usv_ocean.py --boat otterx            # drive: W/S throttle (S past zero: astern),
                                             # A/D steer both pods, Q/E turn on the spot,
                                             # Space stop, C mast camera, X hull
python usv_ocean.py --boat otterx --calm --drop
python usv_ocean.py --boat otterx --record 36 --out otterx.mp4
```

The strip model and the rigid body are the Mariner's; the boat-specific half is her drive,
`usv_rig.AzimuthPods`. Both pods steer to the helm's angle (+-35 deg) and push along
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

## The Otter

Maritime Robotics' Otter, the 2 m electric catamaran: 2000 x 1080 x 1065 mm, 62 kg dry,
two fixed electric pods (she steers on differential thrust).

| file | what it is |
|---|---|
| `otter_spec.json` | every number: the product page's figures, the lines and fittings read off the vendor's side and front renders, the mass budget, the materials |
| `build_otter_blender.py` | builds the geometry with `../build_common.py`'s mesh, hydrostatics and export code and the Otter X's demihull lines, exports `otter.glb` through Blender, and integrates `otter_hydro.json` |
| `otter.glb`, `otter_hydro.json` | generated, not committed |

```
blender --background --factory-startup --python build_otter_blender.py -- --spec otter_spec.json --out otter.glb
```

`python build_otter_blender.py --hydro-only` rebuilds just `otter_hydro.json`. The build
fails if the budget does not add up to 62 kg, if the height or the length leave the
vendor's, if any condition trims more than 2 deg or puts the pontoon decks within 5 cm
of the water.

Frame as the others; the origin is at the lowest point of the pontoon keel line, the
thruster guards reach y = -0.178. The root node `otter` sits at identity.

- `pontoons`, `body` (pod, deck plate, orange side panels), `frame` (the tube frames, the
  thruster struts and guards, the cable loops), `deck_fittings`, `gantry`: meshes.
- `thruster_port`, `thruster_stbd`: the pods, fixed. Each has a `thruster_<side>_rotor`
  that spins about local +X and a `thrust_<side>` empty at the propeller.
- Empties: `camera_main` (under the gantry's top plate), `gnss_fore`, `gnss_aft` (1.6 m
  fore-and-aft baseline), `ais_vhf_antenna`, `lte_antenna`, `imu`, `transducer`,
  `nav_light_port`, `nav_light_stbd`.

Lightship 62 kg, survey (77 kg, the design condition) and full load (92 kg, the vendor's
30 kg payload). At the survey load she floats at 0.185 m on the pontoons (0.36 m to the
guards' feet), 0.9 deg by the stern, GM_T 1.56 m.

From the vendor: the three dimensions, dry weight, speed, endurance, payload, the battery
and thruster count. Read off the two renders (3.18 mm/px in the side one), to a few
centimetres: everything else. Assumed, and marked in the spec: the mass budget, thrust,
the fore frame's plan form, where the thruster struts pass the pontoons. No vendor marks.

### Otter demo

```
python usv_ocean.py --boat otter             # drive: W/S throttle (S past zero: astern),
                                             # A/D steer on the thrust difference,
                                             # Space stop, C mast camera, X hull
python usv_ocean.py --boat otter --calm --drop
python usv_ocean.py --boat otter --record 36 --out otter.mp4
```

Her drive, `usv_rig.FixedPods`, is the Otter X's at her size with fixed pods: the helm moves
60 % of full thrust from one pod to the other, so with the throttle at zero she turns on
the spot. 110 N per pod at bollard (assumed), 60 % of it at the top speed and astern; the
resistance is tuned so full throttle makes the vendor's 4.5 kn. The cameras stand at 0.45
of the distances the larger boats use. In flat water she settles on the solver's pose
(0.378 m to the guards' feet and +0.87 deg against 0.378 and +0.88). In the scripted run
she reaches 4.4 kn, trims about 3 deg by the stern at that speed (the pods push 0.3 m
below her centre of gravity), turns at about 22 deg/s under way and 38 deg/s on the spot,
and stops from 2.5 kn in under 2 s. She rides the default sea (7 m/s over 30 km) within
5 deg of roll.

## In another scene

`../rigs/usv_rig.py` imports no renderer and reads no scene's globals. A scene loads the `.glb`, makes
the hull on its ocean, and steps her after each render (`sample_height` reads the last rendered
field):

```python
from usv_rig import StripHull, Wake, Wash, drive_for

ocean.wake_field.resolution = 1024          # before the first render; .patches = one for each boat
ocean.wake_field.ripple_resolution = 512    # and the waves she makes (0, the default: none)
boat = tp.GLTFLoader().load("usv/otter.glb").scene
hull = StripHull("otter", ocean)             # spec_dir=... for a boat kept elsewhere
hull.seat(x, z, heading)
wake = Wake(hull, ocean)                    # ocean.vessel(i) where several boats share the sea
drive = drive_for(hull).bind(boat)
wash = Wash(hull, drive, ocean, 0)          # patch 0 of the wake field is hers
```

What she leaves on the water is the ocean's wake field (`ocean.wake_field`, `tp.DisplacedMesh.WakeField`):
patches of simulated surface that trail whoever is making a mark, world-anchored and without the
foam texture's repeat. Each texel carries foam, the bubbles under the surface, turbulence, the lane
where the short waves are flattened, and the water's own velocity; the field is carried by that
velocity and stirred by its turbulence, so a race is drawn out and torn by motion. The field knows
nothing of boats. A producer is a source (`ocean.add_wake_source`): a point, the water velocity it
imparts, a radius, and how much of each quantity it puts in. `Wash` makes the sources of a boat:
each propulsor's race from its thrust by momentum theory (`race()`; `Drive.thrusts(u)` names a
drive's propulsors), run to where it has widened to the surface, with air by how hard it is for the
water over it, and the lane her hull drags from each stern. Each frame: `ocean.clear_wake_sources()`
once, then every boat's `wash.update(dt)`. Anything else puts its own with `wash.put(...)`, as the
Nørvasundet scene does for a kayak's paddle blades (`Wash(hull, None, ...)` is a hull with no
propulsor).

Her waves are the field's ripples (`wake_field.ripple_resolution`). Each patch holds the spectrum of
a small linear sea, every wavenumber turning at its own deep-water rate, and a source's `push` (the
force in N it bears down on the water with) forces it. `Wash` lays her weight out as her hulls carry
it: the Bonjean strips at her design waterline, a stretch of hull's share of the buoyancy where its
centroid is and as wide as it is at the water, so a fine bow is a narrow load and a catamaran is
two. No wave is drawn. At rest she makes none; under way the waves that keep up with her are the
ones that grow, which is the Kelvin fan of her speed, bent where she turned; two boats' fans pass
through each other. The sea is periodic over its side, so her waves live in a window round her
(`Wash(..., ripples=side)`, 10 of her lengths by default) and die before they have crossed it. They
are slope on the water's normal, not height: they do not move a boat, and they meet no shore.

Under her own power, `hull.drive = drive`: the scene writes `drive.throttle_cmd` and
`drive.steer_cmd`, and each frame calls `drive.tick(dt)`, `hull.advance(dt, substeps)`,
`wake.update(dt)`, `hull.place(boat)`, `drive.pose()`. That is `usv_ocean.py`.

On a route, `hull.drive` stays `None` and a soft DP holds her to `hull.set_target(...)`; the
drive then only shows the motion: `drive.follow(u, yaw_rate, dt)` turns the nozzle or pods and
spins the rotors as the path implies.

What the strip model needs beyond the tables is the spec's `strip_model` block: `probe_z`, the
half-breadth where each side's water level is read, and `footprint`, the ocean's hull footprint.
A spec without one gets both from its design waterline.
