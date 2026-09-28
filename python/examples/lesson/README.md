# lesson: explainer videos made with threepp

`lesson.py` is a small toolkit for YouTube-style teaching clips. Two lessons are
made with it so far:

| | |
|---|---|
| `ik_fr3.py` | Part 1, *What does inverse kinematics actually solve?* (108 s, Franka FR3) |
| `depth_map.py` | Part 2, *How a robot sees in 3D: from depth pixels to a map* (103 s) |

Everything in the picture is drawn by threepp. The robot, lights, shadows and 3D
annotations are one scene. Captions, equations, panels and plots are a second,
orthographic scene rendered on top. Python only choreographs and pipes finished
frames to ffmpeg.

```
python ik_fr3.py --out D:/dev/lesson_out/ik_fr3.mp4          # 1920x1080, 60 fps, about 5 min
python ik_fr3.py --preview --out preview.mp4                 # 960x540, 30 fps
python ik_fr3.py --stills 10.5,50,72 --outdir shots          # single frames
python ik_fr3.py --sheet --outdir shots                      # contact sheet, one frame per 3 s
python ik_fr3.py --from 48 --to 64 --out iter.mp4            # one beat
```

`depth_map.py` takes the same flags.

Needs threepp (GL renderer only, no Vulkan), numpy, threepp-data (for the FR3
URDF and the studio HDR), and `imageio_ffmpeg` or an `ffmpeg` on PATH. That is
all: text is measured and drawn by threepp, and PNGs are written with the
standard library. The equations are typeset by matplotlib once and cached in
`<lesson>.math.json`, so matplotlib is only needed when you add or edit one.

## Honest numbers

Nothing on screen is faked, and every number is measured during the run and
printed at the start.

**Part 1.** The arm is driven by threepp's own `IkSolver` (damped least squares),
and the update shown on screen is the one it applies. The Jacobian arrows are
finite differences of the robot's forward kinematics.

```
[iter] 12 DLS steps, error 36.7 cm -> 0.043 mm
[null] worst hand drift during self-motion 0.020 mm
[track] solve median 0.04 ms ...
[reach] final gap 13.2 cm
```

**Part 2.** A `tp.DepthSensor` (128 x 96, 3 mm range noise) rides on the hand.
`IkSolver` places the camera itself (`tool_offset`, `AxisAlign`), and every scan
is a real depth render of the scene. The pixel coordinates come from projecting
each hit back through the sensor's pinhole model. The map is a `tp.VoxelGrid`.
The surface is TSDF fusion of the ten depth images (numpy), extracted with
threepp's `marching_cubes` through `ScalarField.from_numpy`. The objects are
primitives so their exact signed distances can score the result:

```
[scan] 10 scans, first 12288 points of 12288 pixels
[map] 68788 points on the tray -> 4410 voxels (1 cm) -> 109912 triangles;
      cloud error 1.84 mm, surface error mean 0.47 mm, p95 1.50 mm
```

Two modelling choices matter for that number: space no view saw inside the tray's
footprint is treated as solid (it is what lies behind the surfaces), and the fused
field gets a light [1 2 1] blur before meshing.

## How a lesson is built

1. **Choreography pass.** Everything stateful (IK solves, scans, maps, trails,
   schedules) runs once, front to back, and records per-frame state.
2. **Frame pass.** `render(t)` depends only on that record and the clock, so any
   frame can be rendered on its own: a still, a sheet, one beat, or the film.

Pieces in `lesson.py`:

| | |
|---|---|
| `Timeline` | named beats; `p()` eased progress, `fade()` in/hold/out envelopes |
| `Keys` | keyframed vectors with smootherstep between keys (cameras, poses) |
| `Stage` | headless canvas + GL (or Vulkan) renderer + dark studio set; `project()` maps 3D to HUD pixels |
| `Arrow3D`, `Ring3D`, `Marker3D`, `Tube3D`, `xray()` | 3D annotations that fade and can ride on robot links |
| `Cloud`, `Segments` | point clouds (round sprites) and line fans, updated in place |
| `Hud` | immediate-mode 2D layer drawn by threepp: `Text2D` from the system TTF, `ShapeGeometry` panels, triangle-strip lines, images, colour bars, maths as matplotlib glyph outlines loaded through `SVGLoader` |
| `turbo`, `write_png`, `Film` | colour map, stdlib PNG writer, and frames to H.264 via an ffmpeg pipe (written to a temp name and renamed on success) |

The HUD is authored in a fixed 1920x1080 design space, whatever the render size.
