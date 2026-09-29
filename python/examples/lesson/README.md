# lesson: explainer videos made with threepp

`lesson.py` is a small toolkit for YouTube-style teaching clips. Seven lessons are
made with it so far:

| | |
|---|---|
| `threepp_intro.py` | Part 0, *What is threepp?* An introduction for new users, with the code in C++ (217 s) |
| `first_app.py` | Part 0b, *Your first real app:* hello.cpp grows into an app in eight steps (194 s) |
| `ik_fr3.py` | Part 1, *What does inverse kinematics actually solve?* (108 s, Franka FR3) |
| `depth_map.py` | Part 2, *How a robot sees in 3D: from depth pixels to a map* (103 s) |
| `imu_tilt.py` | Part 3, *Which way is up? How a robot measures its own tilt* (136 s, Range Rover on a PhysX track) |
| `rocket_pid.py` | Part 4, *Hold still: PID control, flown on a rocket* (150 s, a hopper in PhysX, exhaust in Warp) |
| `warp_threepp.py` | *Warp × threepp:* GPU simulation in Python, sharing its buffers with the renderer (about a minute; Vulkan, CUDA) |

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
python threepp_intro.py --remeasure                          # measure the narration again (lessons paced by it)
```

`threepp_intro.py`, `first_app.py`, `depth_map.py`, `imu_tilt.py`, `rocket_pid.py` and `warp_threepp.py` take the same flags. Output goes to `lesson_out/` in the current
directory unless `--out` / `--outdir` say otherwise.

Needs threepp (GL renderer only, no Vulkan), numpy, threepp_data (for the FR3
URDF and the studio HDR), and `imageio_ffmpeg` or an `ffmpeg` on PATH. That is
all: text is measured and drawn by threepp, and PNGs are written with the
standard library. The equations are typeset by matplotlib once and cached in
`<lesson>.math.json`, so matplotlib is only needed when you add or edit one.
Part 3 also needs the PhysX backend (`tp.HAS_PHYSX`) and the Evoque glTF from
threepp_data, and a threepp module built after `PhysxVehicle.associate` was added.
Part 0 needs the PhysX backend too, and the screenshots in `doc/screenshots`.
Part 4 needs the PhysX backend, a threepp module with `RigidBody.add_force_at_pos`, and
NVIDIA Warp on a CUDA GPU for the exhaust, smoke and dust.

**Narration.** The captions are read aloud by [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M),
an offline neural TTS model (82M parameters, Apache-2.0): `pip install kokoro soundfile`,
and the weights (about 330 MB) download on first use. Each caption is turned into speakable
text first (`lesson.spoken`: units after numbers and a few acronyms become words), and each
clip is cached in `<outdir>/voice_cache` by voice and text, so a re-render only synthesises
the lines whose measured numbers changed. Where a line is longer than its caption, the film
holds the picture until it has been said (`lesson.TimeMap`); the captions, the `.srt` and
`--stills` / `--from` times all follow the film's clock. The clips are placed on one track and
muxed into the mp4 as AAC. `--no-voice` renders without Kokoro. A caption given as
`(start, end, text, False)` is spoken (and written to the `.srt`) but not drawn: Part 4 uses one to
narrate its title card, so the film has sound from the first second.
Kokoro's own reading of a word can be wrong; `threepp.lesson.DEFAULT_WORDS` respells those for every
lesson, a lesson's own table (`WORDS`) adds its names, and a `[word](/phonemes/)` entry sets
the phonemes outright (PhysX is `/fˈɪzˌɛks/`). Parts 0, 0b, 3 and 4 and the Warp film go
further and time their scripts from the measured length of each spoken line (`lesson.Speech`,
read from `<lesson>.speech.json`), so their films never hold for the voice: Parts 0 and 0b and
the Warp film lay their whole timeline out from them (`Speech.layout`), Part 4 places each line
around its flight's events (`Speech.span`), and Part 3 gives its drive more time wherever a line
would overrun its caption (`Speech.stretch`), so the car keeps driving while the line is said.
The lengths never move by themselves: after editing a line, `--remeasure` says every line
again and rewrites the `.speech.json`, and until then the lesson reports the line and `run`
holds the picture for any overrun.

Each threepp_data file is looked up in `THREEPP_DATA_DIR`, then a `threepp_data`
(or `threepp-data`) checkout next to the repo, then the copies CMake fetches into
`cmake-build-*/_deps/threepp_data-src`, and the first place that has that file wins.
A missing studio HDR is reported and the lesson renders without its
image-based lighting.

## House rules

Every lesson film follows two rules:

- **No specific numbers**, in the narration or in the picture. Measured figures (times,
  counts, errors) are printed to the console and written up in this README, not put in
  the film.
- **The title card is narrated**, so the film has sound from its first second. A caption
  given as `(start, end, text, False)` is spoken without being drawn over the title.

`lesson.run` checks the captions against both rules and prints a `[house rules]` warning
for each break: a caption with a digit in it, or no caption in the first two seconds.
The picture is each lesson's own to check. The earlier parts were made before these rules.

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
[comp]  best blend tau 8.33 s: 4.48 deg rms
[kalman] rms 0.42 deg, max 0.75; bias [0.58 -0.41 0.36] deg/s vs set [0.5 -0.4 0.45]
```

The heightfield collider is Z-up while the vehicle is Y-up, so the track is a
`add_static_trimesh` of the same terrain the film draws.

**Part 0.** The code on screen is C++, for readers who will use the library from C++;
the film itself is made from Python like the others. `hello.cpp` is assembled step by
step (canvas and renderer, scene, camera, mesh as geometry plus material, light, render
loop), followed by its Python twin, shared ownership, the scene graph, materials and
lights, an FR3 from URDF driven by `IkSolver`, PhysX boxes, a `DepthSensor`, and a
`CMakeLists.txt` that fetches threepp. The Vulkan renderer appears only as screenshots
from `doc/screenshots`, labelled as such. What is on screen was checked:

* every C++ snippet compiles against the headers; the ownership beat's `use_count()`
  values (9 and 9 after the loop, 1 and 1 after `row->clear()`, 0 after
  `material.reset()`) are what that code prints when linked and run;
* the window after the render-loop step is the program's own output, not the film's
  studio: `HELLO_PY` runs in a separate process (headless, turned a fixed step per
  frame instead of by the clock). A capture build of `hello.cpp` made the same frames
  (3 of 44 million pixels differed by more than one level), so the window is
  labelled as what `hello.cpp` shows;
* the printed values under the IK, PhysX and depth-camera cards and the numbers in
  the captions come from the choreography pass. Wall-clock timings (solve, step and
  scan times) are left out on purpose: they describe the machine, not the library.

```
[ik] 4138 solves, worst position error 0.10 mm
[physx] 40 boxes, resting height 0.17 m
[urdf] fr3: 191 objects, 54 meshes
[depth] 156 scans of 160 x 120, median 17142 points
```

**Part 0b.** The program is `examples/lesson/app.cpp`: Part 0's hello.cpp, grown one step at a
time (OrbitControls, a glTF model, an HDR sky, an AnimationMixer, shadows, an InstancedMesh, a
Raycaster, a Dear ImGui panel). Each step is marked `#if STEP >= n`, so `-DLESSON_APP_STEP=n` builds
the program as it stood after step n. The film reads its code out of that file: a step's lines are
the ones that appear (or go) when the preprocessor runs with STEP = n instead of n - 1. The window
is the program's own output. Each step is its own build (`lesson_app_capture_0` .. `_8`, and
`_6_separate` with a Mesh per stone; not built by default), run while the film is made, headless on
a fixed 1/60 s clock, with the mouse replayed from a script through the canvas's own event path
(`examples/lesson/capture.hpp`). Its frames stream to the film through a pipe. The pointer over the
window marks where the script puts the mouse, and the draw calls are GLRenderer::info()'s counts
for that frame:

```
[app] 110 lines in the finished app
[draw calls] one Mesh per stone 961..962, one InstancedMesh 5..5
```

Part 0b needs those capture targets built (`cmake --build <dir> --target lesson_app_capture_0 ...`;
set `LESSON_APP_BIN` if they are not in a `cmake-build-*/bin` beside the repo).

**Part 4.** A 6 t, 12 m hopper is a PhysX convex body. Its engine pushes along the body
(`RigidBody.add_force_at_pos` at the nozzle) with up to 1.5 times its weight, no lower than
35 % once lit, and answers the controller a fifth of a second late (a first-order lag). The
altitude controller runs at 240 Hz in throttle units, u = Kp e + Ki ∫e dt - Kd v with Kp 0.10 /m,
Ki 0.03 /(m s) and Kd 0.15 s/m, built up on screen one term at a time; a gimbal PD keeps it upright
and leans it back over the pad. The film says what each term does, not these numbers. Five
flights are flown before the film is laid out: P alone, PD with I switched on part-way, PID with
and without anti-windup, and the anti-windup flight carrying on through a gust and a landing:

```
[P]  swings grow; hits the pad 10.7 s after ignition at 9.5 m/s
[PD] settles at 13.33 m: 6.67 m low (hover / Kp = 6.67 m)
[PID] holds 20 m within 4.2 mm
[windup] naive peak 26.72 m; with anti-windup 1.03 m over
[gust] 0.12 x weight for 1.5 s: tilt 5.8 deg, drift 2.57 m
[land] touchdown at 0.19 m/s
```

The PD sag is the textbook one to the centimetre: hovering takes 1/1.5 = 66.7 % throttle, and P
gives that only at an error of 0.667 / Kp. The exhaust, the ground cloud and the dust are
particles stepped by Warp kernels from the recorded engine state (a plume that turns along the
pad where it hits, dust kicked up while the engine is low) and drawn by threepp as soft points
in age bands. They are the only part of the picture that is not the simulation's state; they
are checkpointed so any frame can be rendered on its own.

**Warp × threepp.** The film's own program is `../warp_round_trip.py`. The film runs it in a
subprocess (`--stream`) that steps its simulation at 60 Hz on the film's clock and pipes each
frame in, and the code on screen is read from that file between its `# [name]` markers. In it:

- a curtain of 64 x 48 = 3,072 particles, 4 substeps of 1/240 s a frame with 24 Jacobi passes
  each, hung from a rod, blown by a gusty wind and pushed by a chrome ball;
- 8,192 sparks in the same wind, bouncing off the cloth (its nearest particle, through a Warp
  hash grid), the ball and the floor, from a source just out of shot;
- the cloth written into `enable_vertex_interop` buffers and the sparks into an
  `enable_particle_field_interop` field, by `wp.copy` and kernels inside `render()`;
- the finished colour image read in place through `enable_frame_interop` by an event-camera
  kernel: a pixel fires when its log brightness has moved 0.12 from its last event (ON red,
  OFF blue), and what is shown fades by 0.84 a frame.

The sparks' sprites are drawn after the ray-traced frame, and `ParticleField`'s traced
representation is not implemented yet, so each spark is also a 24-vertex emissive octahedron in a
second `enable_vertex_interop` mesh (196,608 vertices): that is what the chrome reflects.

The other shots are the examples' own films (`ensure_clips`, cached in `lesson_out/warp_clips`),
each sharing its simulation with the renderer rather than copying it through the CPU:

```
fluid  warp_fluid.py --vulkan                     336,864 PBF particles, marching-cubes surface     vertex interop
blast  warp_explosion.py                          5,869,024 gas particles                           particle-field interop
                                                  (the drums: vertex interop; the bricks: PhysX)
gummy  warp_gummy_rain.py --clean                 600 candies, 16,200 particles                    vertex interop
jelly  warp_jelly_wreck.py --no-sensors --clean   78 blocks, 9,750 particles, 85,644 constraints   vertex interop
```

`--clean` (new for this film) drops the jelly's and the gummies' burned-in captions. The film
opens on the pool (the fluid). The closing reel cuts on the narrator's words: `Narration.word_times`
returns Kokoro's own alignment of a line (cached beside its WAV), and the cuts land 0.06 s before
each word is heard. It ends on a wall of every shot, live, with the fire shrinking into its tile,
and the title over it as "drawn by threepp" is said.

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

The toolkit is the experimental `threepp.lesson` package, which ships in the threepp wheel
(`pip install "threepp[lesson]"`; see [python/README.md](../../README.md)). `lesson.py`
re-exports all of it and adds the house layer these films share, so a lesson imports
everything from `lesson`.

In `threepp.lesson`:

| | |
|---|---|
| `Timeline` | named beats; `p()` eased progress, `fade()` in/hold/out envelopes |
| `Keys`, `OrbitCamera` | keyframed vectors with smootherstep between keys; a camera keyframed as azimuth, elevation, distance, look point and fov, with a slow drift, in a Z-up or Y-up frame, optionally following a moving target and its heading |
| `TimeMap` | the film clock with holds, where the picture stands still |
| `Stage` | headless canvas + GL renderer (or one you pass in) + dark studio set, with an optional HDR for image-based light; `project()` maps 3D to HUD pixels; `follow()` moves the key light's shadow area with a moving subject |
| `Arrow3D`, `Ring3D`, `Marker3D`, `Tube3D`, `xray()`, `set_pose()`, `ghost()`, `quat_y_to()` | 3D annotations that fade and can ride on robot links; placing an object at a 4x4 pose; a translucent copy of any object |
| `Cloud`, `Segments` | point clouds (round sprites) and line fans, updated in place |
| `Hud` | immediate-mode 2D layer drawn by threepp: `Text2D` from the system TTF (or `fonts=`), `ShapeGeometry` panels, triangle-strip lines, images, colour bars, maths as matplotlib glyph outlines loaded through `SVGLoader` |
| `Hud.plot`, `readout`, `captions`, `equation_card` | line plots with several series and a legend on linear or log axes; a panel of live values; the lower-third captions; the top-left card of equations |
| `Hud.code`, `tokenize`, `KEYWORDS` | source lines with syntax colours, highlighted and typed in line by line; a lesson that wants more keywords passes its own set (`KEYWORDS["cpp"] \| {"for"}`) |
| `Narration`, `spoken`, `DEFAULT_WORDS` | captions read aloud by Kokoro and cached; caption text in words; the pronunciation table every lesson starts from |
| `Speech`, `overruns` | a lesson's lines and their measured lengths, and the captions laid out from them (below); the one rule for a line that needs more time than its caption |
| `turbo`, `write_png`, `write_srt`, `Film` | colour map, stdlib PNG writer, SubRip subtitles, and frames to H.264 via an ffmpeg pipe (written to a temp name and renamed on success) |

In `lesson.py`, the house layer:

| | |
|---|---|
| `Stage` | the package's Stage with the lessons' defaults: GL, and `env` a file name in threepp_data/textures/env (the studio HDR unless told otherwise) |
| `Hud.title_card`, `summary` | the opening title, and the closing "IN SHORT" card with the repo link |
| `card`, `code_card` | the house card, and a card of code that types itself in, with the program's output under it |
| `data_file` | a threepp_data file from `THREEPP_DATA_DIR`, a checkout next to the repo, or the build directories |
| `run` | the shared command line: film, preview, one stretch, stills, contact sheet, `.srt` captions, narration, `--remeasure`, `--gate` |
| `house_rule_warnings` | the house rules below, checked by `run` |

### Speech

A lesson paced by its narration keeps its lines in a `Speech`, with how long each takes to
say, read from `<lesson>.speech.json` next to the lesson: `{voice: {key: {"s": seconds,
"said": spoken text}}}`. Its captions are laid out from those lengths in one of three ways:
`layout(plan)` for a plan of beats (Parts 0, 0b and the Warp film), `span(k, start)` for one
line placed by hand (Part 4, around its flights' events), and `stretch(spans)` for captions
written at fixed times, giving the script more time wherever a line would run over (Part 3).
The lesson passes the `Speech` to `run(..., speech=SPEECH)`, which hands its pronunciation
table (`WORDS`) to the narrator.

The lengths never move by themselves. After editing a line, the lesson reports it as changed
since it was measured, and `run` holds the picture wherever it runs long. `--remeasure` says
every final line again (through the voice cache) and rewrites the `.speech.json`, and the next
render lays the film out from the new lengths. The layout is timed for one voice; `--voice`
with another prints a note and gets holds instead.

The HUD is authored in a fixed 1920x1080 design space, whatever the render size.
