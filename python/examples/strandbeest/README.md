# The Strandbeest

A wind-walking machine on Theo Jansen's leg linkage, simulated by PhysX. Twenty legs on one
crankshaft, a frame with sails, and nothing in the walk that is posed: the wind pushes on the
sails, the feet's grip on the sand turns the legs, and the legs turn the crank. Built the way the
boats and the aircraft are (`../usv/`, `../uav/`): a headless-Blender generator, a `.glb` that
threepp loads with `GLTFLoader`, and a rig any scene imports.

| file | what it is |
|---|---|
| `../rigs/strandbeest_rig.py` | the machine for any scene: the linkage in closed form (`solve`), `Machine` (the PhysX articulation), `Visual` (the model drawn link by link), `Sails` (cloth), `sail_thrust`; every dimension is here; no threepp import |
| `build_strandbeest_blender.py` | builds the parts (canes with lashings, crankshaft, frame, sails) from the rig's dimensions and exports `strandbeest.glb` |
| `strandbeest_walk.py` | the example: the machine walks downwind over sand with gentle swells; window, still or telemetry |
| `strandbeest.glb` | generated, not committed |

## Build

From this folder, once (about 10 s; `blender` is your Blender executable):

```
blender --background --factory-startup --python build_strandbeest_blender.py
```

Without it the example draws the same machine as plain tubes.

## Run

```
python strandbeest_walk.py                           # window: drag to orbit
python strandbeest_walk.py --wind 9                  # a stronger wind, m/s
python strandbeest_walk.py --telemetry --seconds 30  # no window: the run in numbers
python strandbeest_walk.py --shot 12 --out walk.png  # headless still at t = 12 s
python strandbeest_walk.py --gl                      # the GLRenderer where Vulkan is available
```

It draws with the VulkanRenderer where this threepp has one, else with the GLRenderer.

## How it is simulated

The linkage is a closed kinematic chain: every leg has three loops. The machine is one PhysX
articulation (reduced coordinates) holding the open tree exactly, and one ball joint per loop
closes it, a `tp.Joint` between two links of the same articulation:

```
frame (free root) -> crank                 the bearing, with viscous drag
crank -> j, k        frame -> b-d-e, c     hinges about the axle's direction
c -> g-h-i           b-d-e -> f
loop joints:  j to b-d-e at P1,  k to g-h-i at P2,  f to g-h-i at P4
```

That is 122 links and 60 loop joints. The two triangles of each leg (b-d-e and g-h-i) are rigid
links with convex-hull colliders; the foot is the lower corner of g-h-i. Self-collision is off.
The wind acts on the frame only, as drag on the sail area at the sails' centre (`sail_thrust`),
put on at every physics substep (a force on an articulation link lasts one substep). The sails
themselves are cloth that follows the frame and the same wind; they do not push the frame.

Measured 2026-11-10 with `--telemetry` (309 kg, 480 Hz, 32 position iterations, wind 7.6 m/s
with gusts): it walks at 0.4 to 2.0 m/s, 6 to 10 feet on the ground, the loop joints closed to
0.9 mm. At 6 m/s it rocks in place: the rigid feet have to lift the machine over the humps of
the gait. At about 19 m/s it is blown over.

What did not work: the same machine as 122 free rigid bodies tied by 181 joints (no
articulation) drifts apart from rounding noise beyond about six legs, in zero gravity too.
