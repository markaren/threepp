# Harbour traffic: the sjark, the Norwegian buoys and MS Trollfjord

Boats and floating marks for the maritime demos and for synthetic perception data, built
the way the Mariner is (`../usv/`): one spec file holds every number, the geometry is
built in numpy, headless Blender only assembles and exports the `.glb`, and the
hydrostatics go to a JSON file. The builders import their primitives, the Newell closure
check and the hydrostatic integrator from `../usv/build_mariner_blender.py`.

| file | what it is |
|---|---|
| `sjark_spec.json` | a modern Norwegian GRP sjark, 10.99 m licence class, wheelhouse forward, rigged for gillnet |
| `build_sjark_blender.py` | builds `sjark.glb` and `sjark_hydro.json`; `--seeds` writes livery variants to `out/` |
| `buoys_spec.json` | the floating marks of the Norwegian system (IALA region A) and two kinds of fishing float |
| `build_buoys_blender.py` | builds `buoys.glb` and `buoys_hydro.json` |
| `trollfjord_spec.json` | MS Trollfjord, the 136 m coastal-express ship (2002), Hurtigruten livery without logo or wordmark |
| `build_trollfjord_blender.py` | builds `trollfjord.glb` and `trollfjord_hydro.json` |
| `harbour_scene.py`, `harbour_float.py`, `harbour_labels.py` | the Ålesund harbour scene, the buoy floater and the LaRS label exporter (below) |
| `*.glb`, `*_hydro.json`, `out/` | generated, not committed |

Numbers taken from a source name it in the spec; everything else is marked ASSUMED there.
The models carry no brand names, and the sjark's registration marks are fictional.

## Build

From this folder, with `blender` being your Blender executable (built with Blender 5.0):

```
blender --background --factory-startup --python build_sjark_blender.py -- --spec sjark_spec.json --out sjark.glb
blender --background --factory-startup --python build_sjark_blender.py -- --spec sjark_spec.json --seeds 1,2,3,4,5,6
blender --background --factory-startup --python build_buoys_blender.py -- --spec buoys_spec.json --out buoys.glb
blender --background --factory-startup --python build_trollfjord_blender.py -- --spec trollfjord_spec.json --out trollfjord.glb
```

`python build_sjark_blender.py --hydro-only` and `python build_buoys_blender.py --hydro-only`
rebuild only the hydrostatics, without Blender. A build fails if a closed part is not
closed and consistently wound (the sum of its Newell normals must vanish), if the sjark's
design draft leaves 1.08 m by more than 0.10 m, if her trim passes 2 deg or GM_T leaves
0.4 to 1.5 m, or if any mark floats unstable or more than 0.1 m off its design freeboard.

## The sjark

Frame and origin as the Mariner: X forward, Y up, Z starboard, metres, origin on the
centreline on the moulded baseline halfway along the length. `sjark_hydro.json` uses the
Mariner's `threepp.usv_hydro/1` schema (conditions plus Bonjean tables). At departure she
displaces 12.5 t and floats at 1.15 m maximum draft, 0.5 deg by the stern, GM_T 1.39 m.

- Meshes: `hull` (with the registration marks), `bulwark`, `wheelhouse`, `windows`,
  `mast`, `deck_fittings`, `net_hauler` (starboard).
- Moving parts: `derrick` (turns about +Y at its pivot), `rudder` (local +Y at the stock),
  `propeller` (spins about local +X), `radar` (spins about +Y).
- Empties: `prop_thrust`, `bow_thruster`, `bollard_fwd_port`, `bollard_fwd_stbd`,
  `bollard_aft_port`, `bollard_aft_stbd`, `tow_point`, `camera_mast` (looks down local -Z,
  +Y up), `nav_light_port`, `nav_light_stbd`, `masthead_light`, `stern_light`,
  `reg_mark_port`, `reg_mark_stbd`.
- Seeds vary the livery (hull, bulwark and wheelhouse colours, antifouling, the
  registration, whose letters take a colour that contrasts with the hull) and the gear
  (net hauler or line hauler). They share the default hydrostatics.

## MS Trollfjord

Same frame and hydrostatics schema as the sjark. LOA 135.75 m, beam 21.5 m, published
draught 5.1 m (Wikipedia) and 4.90 m (Deltamarin); twin azimuth thrusters and no rudders,
per the designer's "pram-type hull form ... azimuth thruster propulsion". At departure she
displaces 10,070 t at 5.05 m maximum draft, level trim, GM_T 1.96 m. Hull lines and the mass
budget are ASSUMED; the build fails if the draft leaves 5.1 m by more than 0.25 m, the trim
passes 1 deg or GM_T leaves 0.8 to 3.0 m.

- Meshes: `hull`, `superstructure`, `windows`, `funnel`, `mast`, `lifeboats`, `deck_fittings`.
- Moving parts: `radar_fwd`, `radar_aft` (spin about +Y), `rudder_port`, `rudder_stbd` (the
  azimuth units, steering about local +Y) with `propeller_port`, `propeller_stbd` (spin about
  local +X) and `prop_thrust_port`, `prop_thrust_stbd` as their children.
- Empties: `bow_thruster_1`, `bow_thruster_2`, `stern_thruster`, `bridge` (looks down local -Z,
  +Y up), navigation lights, `bollard_fwd_port`, `bollard_fwd_stbd`, `bollard_aft_port`,
  `bollard_aft_stbd`, `name_port`, `name_stbd`.

## The buoys

`buoys.glb` holds one root per mark, spaced 3 m apart along X for previewing; a demo
clones a root by name and places it. Each root's origin is on its axis at the lowest point
of the buoyant body.

- Spar marks (bøyestaker): `lateral_port`, `lateral_stbd`, `cardinal_n`, `cardinal_e`,
  `cardinal_s`, `cardinal_w`, `isolated_danger`, `safe_water`, `special`. Colours,
  topmarks and the Norwegian retroreflective bands follow Kystverket's guideline for
  navigation marks; the spar itself is sized from a 7.1 m, 225 mm plastic spar buoy
  datasheet (4.6 m draft, 2.5 m focal plane).
- Small marks: `lateral_port_can`, `lateral_stbd_cone`, `mooring_buoy`.
- Fishing floats: `blaase_a3`, `blaase_a5` (sized like the common A-series floats) and
  `garnblaase`, a gillnet end marker with pole, flag and radar reflector.
- Children: `<mark>_body`, `<mark>_mooring_eye` (chain attachment), `<mark>_top`, and where
  they exist `<mark>_topmark`, `<mark>_light` (lamp empty `<mark>_lamp`) and
  `<mark>_number` (lateral numbers, facing +-X; yaw the mark so they face the fairway).
- Retroreflective sheeting has its own materials (`retro_blue`, `retro_yellow`,
  `retro_red`, `retro_green`, `retro_white`), so a demo can brighten them under a light.

`buoys_hydro.json` (`threepp.buoy_hydro/1`) gives per mark the mass, the centre of gravity,
the design waterline including the pull of the mooring chain at the eye, GM, the heave and
roll periods and a draft table for bobbing. The spars are stable only with that chain
load, as real spar buoys are.

## The scene: the Mariner leaves Ålesund harbour

`harbour_scene.py` (plans/harbour-scene.md) sets the traffic in Ålesund's inner harbour on
the `geodata/aalesund` pack: five sjarks along the north quay below Keiser Wilhelms gate
(two lying starboard side to), MS Trollfjord alongside the Cruise Pier mole's outer face,
the marks at the basin mouth (a 30 m lateral gate, the special mark, a south cardinal), a
mooring buoy and two gillnet floats. Every boat floats on its own
`turbine/fleet_buoyancy.StripHull` and its mooring lines follow it each frame; the marks
float on `harbour_float.BuoyFloat`, whose wave excitation decays over the draft (a 4.6 m
spar sits nearly still in the short chop). The Mariner (`../usv/mariner.glb`) casts off
from the east end of the quay and runs out on her DP along a smooth spline (5 kn in the
basin, 8 kn past the mouth) and carries the ocean's hull exclusion, so she has the
wake (no foam: the ocean's foam texture spans one 80 m swell tile and repeats over the
sheet, so her foam would come back as a lattice of dashes over the basin). `S.registry`
lists every placed object as (node, LaRS class, instance name) for the label pass.

```
python harbour_scene.py --shot all --out DIR              # stills + the bob check
python harbour_scene.py --shot drone --light overcast      # phase A's light, for an A/B
python harbour_scene.py --film-test --out DIR              # 3 stills per cut + a contact sheet
python harbour_scene.py --film --out DIR                   # 30 fps frames, then an mp4 (imageio_ffmpeg)
```

The film's cuts sit in sim time; the simulation runs on (unsaved, at 5 fps) through the
gaps between them. Headless only. A file in `--out` is never overwritten.

The look is layered, and each layer has a switch so a before/after renders from one build:

- `--quay real|plain`: `harbour_quay.py` gives the quay and the mole concrete panels, an edge
  beam, a tidal zone at true heights (wet concrete, weed, barnacles), rust under the bollards
  and three quay ladders; `plain` is the bare slabs.
- `--weather AMOUNT` (default 1, 0 = clean): `harbour_weather.py` re-bakes the loaded boats and
  marks with waterline scum, rust streaks, scuffs, weed bands and gull droppings, in each
  asset's own frame, seeded per boat. The builders and the `.glb` files are not touched, and
  the Mariner stays clean.
- Trollfjord lies on ten lines to bollards on the mole, with floating fenders between.
- `--light bright|overcast` and `--sun AZ:EL` (degrees; default 250:30, a late-summer
  afternoon). The air's fog colour is its scattering albedo, so it is a blue, not the pale
  horizon colour.
- `--look-ab NAME --out DIR` renders twelve fixed frames from one continuous sim, and
  `--look-compose DIR` pairs two such folders (`DIR/A`, `DIR/B`) into labelled composites.

### Labels: a synthetic LaRS split

```
python harbour_scene.py --labels DIR --out DIR2            # 1280x720 by default
```

`--labels` runs the film's own simulation and writes what the Mariner's `camera_main` sees
as a split in the format of LaRS (Žust et al., ICCV 2023), so a LaRS model or the LaRS
evaluator reads it as it is. There are four sequences: `harbour_c0` (camera_main over the
chase cut, running west along the quay past the sjarks) and the film's three POV cuts.
`harbour_labels.py` holds the format, the id mapping and the checks. DIR gets:

- `images/<seq>_<frame>.jpg`, a keyframe every 13 frames, and `images_seq/`, each keyframe
  with the 9 frames before it, as LaRS gives them;
- `panoptic_masks/<name>.png`: the category id in R, the instance id in G * 256 + B;
- `semantic_masks/<name>.png`: 0 obstacle, 1 water, 2 sky, 255 ignore;
- `panoptic_annotations.json` (COCO panoptic: segments with bbox, area, iscrowd) and
  `image_list.txt`.

Categories are LaRS's: 1 static obstacle, 3 water, 5 sky, 11 boat/ship, 14 buoy, 17 float
(12, 13, 15, 16 and 19 are listed but not used here). The sjarks and Trollfjord are boats,
the marks and the mooring buoy are buoys and the gillnet floats are floats. The quay, the
mole, the mooring lines and the whole town are one static obstacle segment. The sea,
reflections included, is water. The Mariner's own bow at the bottom of her view is void,
(0, 0, 0) in the panoptic mask and 255 in the semantic mask, which LaRS ignores. Instance
ids stay the same from frame to frame, so they also work as track ids.

DIR2 gets the POV | mask film (`harbour_labels_film_vN.mp4`), overlay stills (RGB, mask,
50 % overlay), a contact sheet of every keyframe, `labels_stats.json` (per class: segments,
pixels, distinct instances and bbox sizes; per instance: its bboxes and areas) and
`labels_gates.json`. That file holds three checks. (1) Every object in view and not hidden
by another is in some keyframe's mask. (2) None of her own hull is labelled a boat: each
keyframe is rendered again with her hidden. (3) The mask and the picture come from the
same frame: a moving mark's mask centroid is compared with its colour in the image.
