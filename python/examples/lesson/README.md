# lesson: explainer videos made with threepp

`lesson.py` is a small toolkit for YouTube-style teaching clips. Three lessons are
made with it so far:

| | |
|---|---|
| `ik_fr3.py` | Part 1, *What does inverse kinematics actually solve?* (108 s, Franka FR3) |
| `depth_map.py` | Part 2, *How a robot sees in 3D: from depth pixels to a map* (103 s) |
| `imu_tilt.py` | Part 3, *Which way is up? How a robot measures its own tilt* (108 s, Range Rover on a PhysX track) |

Everything in the picture is drawn by threepp. The robot, lights, shadows and 3D
annotations are one scene. Captions, equations, panels and plots are a second,
orthographic scene rendered on top. Python only choreographs and pipes finished
frames to ffmpeg.

```
python ik_fr3.py                                             # lesson_out/ik_fr3.mp4: 1920x1080, 60 fps, about 5 min
python ik_fr3.py --preview --out preview.mp4                 # 960x540, 30 fps
python ik_fr3.py --stills 10.5,50,72 --outdir shots          # single frames
python ik_fr3.py --sheet --outdir shots                      # contact sheet, one frame per 3 s
python ik_fr3.py --from 48 --to 64 --out iter.mp4            # one beat
python ik_fr3.py --srt                                       # captions as lesson_out/ik_fr3.srt, for YouTube
python ik_fr3.py --voice am_michael                          # another narrator (default af_heart)
python ik_fr3.py --no-voice                                  # a silent film
```

`depth_map.py` and `imu_tilt.py` take the same flags. Output goes to `lesson_out/` in the current
directory unless `--out` / `--outdir` say otherwise.

Needs threepp (GL renderer only, no Vulkan), numpy, threepp_data (for the FR3
URDF and the studio HDR), and `imageio_ffmpeg` or an `ffmpeg` on PATH. That is
all: text is measured and drawn by threepp, and PNGs are written with the
standard library. The equations are typeset by matplotlib once and cached in
`<lesson>.math.json`, so matplotlib is only needed when you add or edit one.
Part 3 also needs the PhysX backend (`tp.HAS_PHYSX`) and the Evoque glTF from
threepp_data, and a threepp module built after `PhysxVehicle.associate` was added.

**Narration.** The captions are read aloud by [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M),
an offline neural TTS model (82M parameters, Apache-2.0): `pip install kokoro soundfile`,
and the weights (about 330 MB) download on first use. Each caption is turned into speakable
text first (`lesson.spoken`: units after numbers and a few acronyms become words), and each
clip is cached in `<outdir>/voice_cache` by voice and text, so a re-render only synthesises
the lines whose measured numbers changed. Where a line is longer than its caption, the film
holds the picture until it has been said (`lesson.TimeMap`); the captions, the `.srt` and
`--stills` / `--from` times all follow the film's clock. The clips are placed on one track and
muxed into the mp4 as AAC. `--no-voice` renders without Kokoro.

Each threepp_data file is looked up in `THREEPP_DATA_DIR`, then a `threepp_data`
(or `threepp-data`) checkout next to the repo, then the copies CMake fetches into
`cmake-build-*/_deps/threepp_data-src`, and the first place that has that file wins.
A missing studio HDR is reported and the lesson renders without its
image-based lighting.

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

**Part 3.** A `tp.PhysxVehicle` (the Evoque tuning) drives a stadium track with
hills and a side slope, steered by pure pursuit. A `tp.Imu` rides the chassis
(`PhysxVehicle.associate`): ICM-42688-P white noise, plus a gyro turn-on bias of
(0.50, -0.40, 0.45) deg/s set in its noise model. The true tilt is the chassis pose.
Four estimates run on the recorded samples: the gyro integrated alone, the
accelerometer alone, a complementary filter (its blend swept, best one kept), and a
Kalman filter over up and gyro bias that also uses the wheel speeds to remove the
car's own acceleration (dv/dt forward plus w x v):

```
[parked] |f| 9.810 m/s^2, gyro [0.5 -0.4 0.45] deg/s (bias set [0.5 -0.4 0.45])
[accel] parked 0.08 deg; braking up to 28.7, cornering up to 15.7 deg
[comp]  best blend tau 20.83 s: 4.46 deg rms
[kalman] rms 0.41 deg, max 0.71; bias [0.57 -0.47 0.37] deg/s vs set [0.5 -0.4 0.45]
```

The heightfield collider is Z-up while the vehicle is Y-up, so the track is a
`add_static_trimesh` of the same terrain the film draws.

## How a lesson is built

1. **Choreography pass.** Everything stateful (IK solves, scans, maps, trails,
   schedules) runs once, front to back, and records per-frame state.
2. **Frame pass.** `render(t)` depends only on that record and the clock, so any
   frame can be rendered on its own: a still, a sheet, one beat, or the film.

A lesson script defines `setup(width, height)`, which builds the stage, runs the
choreography and returns `(render, captions)`, and ends with
`run(name, duration, setup)`, which provides the command line above. A lesson works on its
own clock; it can declare moments where the film stands still (`render.holds`, as Part 3
does to introduce each estimate on its own), and `run` adds holds for the narration.

Pieces in `lesson.py`:

| | |
|---|---|
| `run` | the shared command line: film, preview, one stretch, stills, contact sheet, `.srt` captions, narration |
| `Narration`, `spoken`, `TimeMap` | captions read aloud by Kokoro and cached; caption text in words; the film clock with holds |
| `Timeline` | named beats; `p()` eased progress, `fade()` in/hold/out envelopes |
| `Keys`, `OrbitCamera` | keyframed vectors with smootherstep between keys; a camera keyframed as azimuth, elevation, distance, look point and fov, with a slow drift, in a Z-up or Y-up frame, optionally following a moving target and its heading |
| `Stage` | headless canvas + GL (or Vulkan) renderer + dark studio set; `project()` maps 3D to HUD pixels; `follow()` moves the key light's shadow area with a moving subject |
| `Arrow3D`, `Ring3D`, `Marker3D`, `Tube3D`, `xray()`, `set_pose()`, `ghost()` | 3D annotations that fade and can ride on robot links; placing an object at a 4x4 pose; a translucent copy of any object |
| `Cloud`, `Segments` | point clouds (round sprites) and line fans, updated in place |
| `Hud` | immediate-mode 2D layer drawn by threepp: `Text2D` from the system TTF, `ShapeGeometry` panels, triangle-strip lines, images, colour bars, maths as matplotlib glyph outlines loaded through `SVGLoader` |
| `Hud.plot`, `readout` | line plots with several series and a legend on linear or log axes; a panel of live values |
| `Hud.title_card`, `captions`, `equation_card`, `summary` | the opening title, the lower-third captions, the top-left card of equations, and the closing "IN SHORT" card |
| `turbo`, `write_png`, `write_srt`, `Film` | colour map, stdlib PNG writer, SubRip subtitles, and frames to H.264 via an ffmpeg pipe (written to a temp name and renamed on success) |

The HUD is authored in a fixed 1920x1080 design space, whatever the render size.
