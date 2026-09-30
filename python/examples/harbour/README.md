# Harbour traffic: the sjark and the Norwegian buoys

Boats and floating marks for the maritime demos and for synthetic perception data, built
the way the Mariner is (`../usv/`): one spec file holds every number, the geometry is
built in numpy, headless Blender only assembles and exports the `.glb`, and the
hydrostatics go to a JSON file. Both builders import their primitives, the Newell closure
check and the hydrostatic integrator from `../usv/build_mariner_blender.py`.

| file | what it is |
|---|---|
| `sjark_spec.json` | a modern Norwegian GRP sjark, 10.99 m licence class, wheelhouse forward, rigged for gillnet |
| `build_sjark_blender.py` | builds `sjark.glb` and `sjark_hydro.json`; `--seeds` writes livery variants to `out/` |
| `buoys_spec.json` | the floating marks of the Norwegian system (IALA region A) and two kinds of fishing float |
| `build_buoys_blender.py` | builds `buoys.glb` and `buoys_hydro.json` |
| `*.glb`, `*_hydro.json`, `out/` | generated, not committed |

Numbers taken from a source name it in the spec; everything else is marked ASSUMED there.
The models carry no brand names, and the sjark's registration marks are fictional.

## Build

From this folder, with `blender` being your Blender executable (built with Blender 5.0):

```
blender --background --factory-startup --python build_sjark_blender.py -- --spec sjark_spec.json --out sjark.glb
blender --background --factory-startup --python build_sjark_blender.py -- --spec sjark_spec.json --seeds 1,2,3,4,5,6
blender --background --factory-startup --python build_buoys_blender.py -- --spec buoys_spec.json --out buoys.glb
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
