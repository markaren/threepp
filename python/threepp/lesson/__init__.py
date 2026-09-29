"""threepp.lesson: a small toolkit for making explainer videos with threepp. EXPERIMENTAL: its
API may change between releases.

Not imported by `import threepp`; use `import threepp.lesson` (or `from threepp.lesson import
Stage, Hud, Timeline, Film`). It needs only numpy. matplotlib typesets equations the first time
they are drawn (`Hud.math`, cached to JSON after that), imageio-ffmpeg supplies ffmpeg for
`Film`, and Kokoro reads captions aloud (`Narration`); each is imported only when used.

A lesson is rendered in two passes:

1. **Choreography.** Everything stateful (an IK solve, a simulation, a trail) runs once, front
   to back, and records per-frame state. No rendering.
2. **Frames.** `render(t)` is a pure function of the recorded state and the clock, so any
   frame can be rendered on its own: a still, a contact sheet, a low-res preview of one beat,
   or the final film.

The pieces, from bottom to top:

`timing`: `Timeline`, `Keys`, `OrbitCamera`, `TimeMap`
    Named beats with start/end times (`tl.p("fk", t)` is the eased 0..1 progress through a
    beat, `tl.fade("fk", t)` a fade-in/hold/fade-out envelope), keyframed vectors, a camera
    keyframed as azimuth, elevation, distance, look point and fov with a slow drift, and the
    film's clock with holds, where the picture stands still.

`scene`: `Stage`, `Arrow3D`, `Ring3D`, `Marker3D`, `Tube3D`, `Cloud`, `Segments`
    A headless canvas and GL renderer (or one you pass in) with a dark studio set: key and rim
    light, optional image-based light, a floor that dissolves into the background. `frame()`
    returns the picture as an (H, W, 3) uint8 array and `project()` maps a world point to Hud
    pixels, so 2D callouts can point at 3D things. The annotation kit: thick arrows, glowing
    rings around a joint axis, a target marker, a tube trail, point clouds and line fans; all
    fade with `.opacity` and can ride on any object. `set_pose` places an object at a 4x4 pose,
    `ghost()` makes a translucent copy.

`hud`: `Hud`, `tokenize`
    The 2D layer, also drawn by threepp: an orthographic scene rendered over the 3D one.
    Panels, text (Text2D from a system TTF, or `fonts=`), maths (matplotlib's mathtext outlines
    loaded through SVGLoader, no TeX needed), lines, arrows, bars, plots, readouts, callouts,
    captions, a card of equations, and source code with syntax colours (`Hud.code`).

`media`: `Film`, `FramePipe`, `write_png`, `write_srt`, `write_wav`
    Frames to H.264 through an ffmpeg pipe (written to a temporary name and renamed on
    success; `FramePipe` is the pipe under it, for your own ffmpeg arguments), PNG without an
    imaging library, SubRip captions, WAV.

`speech`: `Narration`, `spoken`, `Speech`, `overruns`
    Captions read aloud by Kokoro and cached by text; caption text in words a TTS reads well;
    a lesson's lines with their measured lengths, from which the captions are laid out.
"""
from . import hud, media, scene, speech, timing
from .hud import *  # noqa: F401,F403
from .media import *  # noqa: F401,F403
from .scene import *  # noqa: F401,F403
from .speech import *  # noqa: F401,F403
from .timing import *  # noqa: F401,F403

__all__ = timing.__all__ + scene.__all__ + hud.__all__ + media.__all__ + speech.__all__
