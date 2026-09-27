# lesson: explainer videos made with threepp

`lesson.py` is a small toolkit for YouTube-style teaching clips, and `ik_fr3.py` is
the first lesson made with it: *What does inverse kinematics actually solve?*,
about 108 s on a Franka FR3.

Everything in the picture is drawn by threepp. The robot, lights, shadows and 3D
annotations are one scene. Captions, equations, panels and plots are a second,
orthographic scene rendered on top. Python only choreographs and pipes finished
frames to ffmpeg.

```
python ik_fr3.py --out D:/dev/lesson_out/ik_fr3.mp4          # 1920x1080, 60 fps, about 10 min
python ik_fr3.py --preview --out preview.mp4                 # 960x540, 30 fps, about 4 min
python ik_fr3.py --stills 10.5,50,72 --outdir shots          # single frames
python ik_fr3.py --sheet --outdir shots                      # contact sheet, one frame per 3 s
python ik_fr3.py --from 48 --to 64 --out iter.mp4            # one beat
```

Needs threepp (GL renderer only, no Vulkan), numpy, threepp-data (for the FR3
URDF and the studio HDR), and `imageio_ffmpeg` or an `ffmpeg` on PATH. That is
all: text is measured and drawn by threepp, and PNGs are written with the
standard library. The equations are typeset by matplotlib once and cached in
`ik_fr3.math.json`, so matplotlib is only needed when you add or edit one.

## Honest numbers

Nothing on screen is faked. The arm is driven by threepp's own `IkSolver` (damped
least squares), and the update shown on screen is the one it applies. The
Jacobian arrows are finite differences of the robot's forward kinematics. The
error plot, step count, null-space drift and solve time are measured during the
run and printed at the start:

```
[iter] 12 DLS steps, error 36.7 cm -> 0.043 mm
[null] worst hand drift during self-motion 0.020 mm
[track] solve median 0.04 ms ...
[reach] final gap 13.2 cm
```

## How a lesson is built

1. **Choreography pass.** Everything stateful (IK solves, trails, schedules) runs
   once, front to back, and records per-frame state. No rendering happens here.
2. **Frame pass.** `render(t)` depends only on that record and the clock, so any
   frame can be rendered on its own: a still, a sheet, one beat, or the film.

Pieces in `lesson.py`:

| | |
|---|---|
| `Timeline` | named beats; `p()` eased progress, `fade()` in/hold/out envelopes |
| `Keys` | keyframed vectors with smootherstep between keys (cameras, poses) |
| `Stage` | headless canvas + GL (or Vulkan) renderer + dark studio set; `project()` maps 3D to HUD pixels |
| `Arrow3D`, `Ring3D`, `Marker3D`, `Tube3D`, `xray()` | 3D annotations that fade and can ride on robot links |
| `Hud` | immediate-mode 2D layer drawn by threepp: `Text2D` from the system TTF, `ShapeGeometry` panels, triangle-strip lines, maths as matplotlib glyph outlines loaded through `SVGLoader` |
| `Film` | frames to H.264 via an ffmpeg pipe, written to a temp name and renamed on success |

The HUD is authored in a fixed 1920x1080 design space, whatever the render size.
