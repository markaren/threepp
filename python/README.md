# threepp for Python

[threepp](https://github.com/markaren/threepp) is a cross-platform C++ 3D
library with the high-level API of [three.js](https://threejs.org). This package
exposes that high-level API to Python through a single
[pybind11](https://github.com/pybind/pybind11) extension module, `threepp`.

You get the three.js scene graph — `Scene`, `Mesh`, geometries, materials,
cameras, lights, `OrbitControls`, an animation loop — plus **headless
render-to-numpy** for ML data generation, robotics cameras and machine vision.

```python
import threepp as tp

scene = tp.Scene()
scene.add(tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial()))
scene.add(tp.AmbientLight())

camera = tp.PerspectiveCamera(75, 1.0, 0.1, 100)
camera.position.z = 5

canvas = tp.Canvas("demo")
renderer = tp.GLRenderer(canvas)
canvas.animate(lambda: renderer.render(scene, camera))
```

## Install

> **Status: pre-1.0 / alpha.** This is a young package and the API is still
> moving — it **may change between releases**. Releases are dated (a `YYYY-MM-DD`
> tag builds a `YYYY.MM.DD` wheel); pin the exact release you tested and check
> the release notes before upgrading.

Prebuilt wheels (Windows / Linux, CPython 3.10–3.14) are on PyPI — no build
tools or system libraries needed:

```sh
pip install threepp
```

They ship the GL renderer, **CPU PhysX physics**
(`PhysxWorld`, articulations, proprioceptive sensors — `tp.HAS_PHYSX == True`)
**and the Vulkan deferred renderer** with its G-buffer AOVs
(`tp.HAS_VULKAN == True`). Vulkan needs a Vulkan-capable GPU driver at runtime —
on machines without one the wheel still imports and renders GL, and
`tp.vulkan_available()` tells you which world you're in. macOS has no prebuilt
wheel — `pip install` builds it from source there (GL-only; a C++ compiler and
CMake are all it needs).

The **scene editor** is its own package — author physics-ready scenes, robots
and sensors visually, then drive them from Python:

```sh
pip install threepp-editor      # or: pip install threepp[editor]  (Windows)
threepp-editor
```

(`threepp[full]` = `[rl,editor]`. The editor ships Windows wheels only; the
extra is a no-op elsewhere, and `pip install threepp-editor` states the
platform requirement honestly.)

Or build from source (needs a C++ compiler + CMake ≥ 3.21; pybind11 is fetched
automatically):

```sh
pip install "git+https://github.com/markaren/threepp"
```

Physics in the wheel is **CPU-only** — the GPU runtime (`PhysXGpu_64.dll`) is
238 MB and stays out. Everything `PhysxWorld` does by default works. The GPU
paths need the two GPU DLLs (`PhysXGpu_64.dll` + `PhysXDevice64.dll`) dropped
next to the installed module (they are loaded lazily by name, never linked):
that alone enables `PhysxWorld(gpu_dynamics=True)` and the direct-GPU RL batch
(verified). Soft bodies additionally need a Python API the wheel does not bind
yet — today they are an editor capability.

## Build

The module is built as part of threepp's CMake project, gated behind
`THREEPP_WITH_PYTHON`. pybind11 is fetched automatically; the host Python is
auto-detected (override with `-DPython_EXECUTABLE=...`).

```sh
cmake -S . -B build -DTHREEPP_WITH_PYTHON=ON \
      -DTHREEPP_BUILD_EXAMPLES=OFF -DTHREEPP_BUILD_TESTS=OFF
cmake --build build --target threepp_py
```

(On Windows, CMake defaults to the multi-config Visual Studio generator; the build
works under it, but `-G Ninja` gives a faster single-config build.)

The built module (`threepp.*.pyd` / `threepp.*.so`) is placed in this `python/`
directory, so scripts here can `import threepp`. The examples add this directory
to `sys.path` automatically, so they run from anywhere.

This is a GL-only build — it needs no Vulkan SDK, PhysX or CUDA, and works on any
laptop. Off-screen rendering uses a hidden GLFW window (a real GL context), so a
display is only required for the on-screen examples.

## Examples

| Script | What it shows |
| --- | --- |
| [`examples/hello_cube.py`](examples/hello_cube.py) | On-screen window: spinning, lit cubes you can orbit (`OrbitControls` + animation loop). Needs a display. |
| [`examples/pbr_showcase.py`](examples/pbr_showcase.py) | **Photoreal PBR playground** — metals + glowing gems under a *procedurally generated* HDR sky (no assets), ACES tone mapping, live ImGui controls. `--shot out.png` renders headless. |
| [`examples/text_overlay.py`](examples/text_overlay.py) | **Text & SVG overlay** — billboard `TextSprite` labels on 3D objects + a 2D HUD (`Text2D` + `SVGLoader` badge) via an `OrthographicCamera` overlay pass. |
| [`examples/headless_render.py`](examples/headless_render.py) | Off-screen render straight into a `(H, W, 3)` uint8 numpy array; saves a PNG. No window. |
| [`examples/textured_box.py`](examples/textured_box.py) | Load an image with `TextureLoader` and map it onto a mesh (headless). |
| [`examples/load_model.py`](examples/load_model.py) | `python load_model.py model.glb` — load a model with `ModelLoader`, auto-frame and render it. |
| [`examples/vulkan_aovs.py`](examples/vulkan_aovs.py) | Vulkan deferred render → G-buffer AOVs (normals / segmentation / albedo / depth) as numpy. Needs a Vulkan build. |
| [`examples/vulkan_ocean.py`](examples/vulkan_ocean.py) | The FFT-displaced **`Ocean`** — fancy water (waves, foam, transmission) in one line; orbit around it live. Needs a Vulkan build + display. |
| [`examples/ui_demo.py`](examples/ui_demo.py) | In-window Dear ImGui control panel (sliders/buttons) driving the scene live (GL). Needs a display. |
| [`examples/vulkan_ui.py`](examples/vulkan_ui.py) | The same ImGui control panel, over the **Vulkan** deferred renderer. Needs a Vulkan build + display. |
| [`examples/physics_demo.py`](examples/physics_demo.py) | A pile of boxes tumbling onto the floor — `PhysxWorld` rigid bodies driving the scene graph. Needs a PhysX build + display. |
| [`examples/imu_demo.py`](examples/imu_demo.py) | **Headless IMU** — a box dropped onto the floor with an `Imu` mounted off-CoM; prints the physics-truth table (free fall ~0, at rest ~+9.81). Needs a PhysX build; no display. |
| [`examples/cartpole/train_cartpole.py`](examples/cartpole/train_cartpole.py) / [`play_cartpole.py`](examples/cartpole/play_cartpole.py) | **GPU-vectorized RL end to end** — a `VecTask` cartpole swing-up trained with the owned `threepp.rl` PPO (committed checkpoint included; `play` renders the result). Needs a PhysX GPU build + torch. |
| [`examples/spot/`](examples/spot) | **Quadruped RL family** — Spot locomotion policies (steps, stairs, heightfield), sim-to-sim deploy into the editor, depth scanning; see its [README](examples/spot/README.md). |
| [`examples/smoke_test.py`](examples/smoke_test.py) | Assertion-based regression test of the whole surface; prints `ALL OK`. |

```sh
cd python
python examples/headless_render.py
```

## What's exposed

- **Math** (mutable value types): `Vector2/3/4`, `Color`, `Euler`, `Quaternion`,
  `Matrix3/4`, `Box3`. A hex int (`0xff0000`) or CSS/name string converts
  implicitly to a `Color`, so `material.color = 0xff0000` just works.
- **Scene graph**: `Object3D`, `Scene`, `Group`, `Mesh`, `InstancedMesh`,
  `Points`, `Line`, `LineSegments`, `Sprite`. `obj.position.x = 1` mutates in
  place, exactly like three.js; `scene.add(a, b, c)`, `traverse`,
  `get_object_by_name`, `children` all work.
- **Geometries**: `Box`, `Sphere`, `Plane`, `Cylinder`, `Cone`, `Capsule`,
  `Torus`, `TorusKnot`, `Circle`, `Ring`, `Icosahedron`, `Octahedron`, plus the
  swept/extruded pair — `TubeGeometry(curve, ...)` and
  `ExtrudeGeometry(shape, depth=...)` / `ShapeGeometry(shape)`.
- **Curves and 2-D shapes**: `CatmullRomCurve3`, `LineCurve3` (both `Curve3`),
  `LineCurve`, `SplineCurve`, `Path` (both `Curve2`) and `Shape` — the sampling
  API (`get_point`, `get_point_at`, `get_tangent`, `get_points`,
  `get_spaced_points`, `get_length`) is shared by all of them.
- **Materials**: `MeshStandard`, `MeshPhong`, `MeshLambert`, `MeshBasic`,
  `MeshNormal`, `Points`, `LineBasic`, `Sprite`, `Shadow` — concrete fields, the
  shared base fields (`opacity`, `transparent`, `side`, …), and texture-map slots
  (`map`, `normal_map`, `roughness_map`, `metalness_map`, `emissive_map`,
  `ao_map`, `alpha_map`, …).
- **Cameras**: `PerspectiveCamera`, `OrthographicCamera`.
- **Lights**: `Ambient`, `Directional`, `Point`, `Spot`, `Hemisphere`,
  `RectArea`.
- **Textures**: `Texture`, `TextureLoader` (`load(path, color_space=...)`),
  with `TextureWrapping` / `Filter` / `ColorSpace` enums.
- **Model loaders**: `ModelLoader` (`load(path)` → `Group`, dispatches by
  extension: `.obj` / `.gltf` / `.glb` / `.stl` / `.dae`), plus `OBJLoader`,
  `STLLoader`, and `GLTFLoader` (`load(path)` → `GLTFResult` with `.scene` and
  `.animations`). All first-party — no Assimp/FBX/USD needed.
- **Animation**: `AnimationMixer`, `AnimationClip`, `AnimationAction`, the
  `KeyframeTrack` family (`Vector`/`Quaternion`/`Number`/`Color`), and the `Loop`
  / `AnimationBlendMode` / `Interpolation` enums — three.js' animation system.
  Play clips loaded from glTF, or build them procedurally:

  ```python
  result = tp.GLTFLoader().load("model.glb")
  scene.add(result.scene)
  mixer = tp.AnimationMixer(result.scene)
  mixer.clip_action(result.animations[0]).set_loop(tp.Loop.REPEAT).play()

  clock = tp.Clock()
  canvas.animate(lambda: (mixer.update(clock.get_delta()),
                          renderer.render(scene, camera)))
  ```
- **Rendering**: `Canvas` (window / headless), `GLRenderer`
  (`render`, `set_clear_color`, `read_pixels` → numpy, `save_frame`, shadows,
  `tone_mapping` / `tone_mapping_exposure`), `OrbitControls`, `Clock`.
- **HDR image-based lighting**: `RGBELoader().load("env.hdr")` → an equirect
  `Texture`; assign it to `scene.environment` (IBL on standard/physical
  materials) or `scene.background`. `ToneMapping.ACESFilmic` (etc.) keeps HDR
  highlights from clipping.
- **Camera intrinsics / extrinsics**: `camera.projection_matrix`,
  `matrix_world_inverse`, `obj.matrix_world`, `get_world_quaternion`,
  `Matrix4.to_numpy()` (→ `(4, 4)`), and `Vector3.project(camera)` / `unproject`
  for 3D↔2D — the building blocks for 6-DoF pose ground truth and 2D annotation.
- **URDF robots**: `URDFLoader().load("robot.urdf")` → a `Robot` (an `Object3D`)
  with `set_joint_value(s)`, `get_joint_range(s)`, `num_dof`, and forward
  kinematics (`get_end_effector_transform()` / `compute_end_effector_transform`).
- **2D text & SVG**: `FontLoader().default_font()` (embedded — no font file),
  `Text2D` / `Text3D` (text meshes), `TextSprite` (billboard labels), and
  `SVGLoader().parse(...)` → a `Group` of filled meshes. Pair with an
  `OrthographicCamera` + `auto_clear=False` for a HUD.
- **In-window UI**: `ImguiContext` + the `threepp.imgui` submodule — Dear ImGui
  immediate-mode widgets (window/text/button/slider/checkbox/color/combo/…) for
  control panels. Works on **both** the GL and Vulkan renderers (`tp.HAS_IMGUI`).
- **3D Gaussian Splatting**: `SplatLoader.load_ply()` reads the `.ply` files
  3DGS optimisers emit (any SH degree, header-driven) and `SplatCloud` puts the
  result in the scene graph like any mesh — rendered by **both** the GL and
  Vulkan backends. `is_splat_ply()` discriminates splat files from mesh PLYs;
  `submit_ranges` exposes the chunk-LOD/culling mechanism (Vulkan).
- **Mesh collision queries**: `BVH` — `BVH.intersects(a, b)` (exact
  triangle-triangle, early-exit), `BVH.intersect(a, b)` → an `(N, 3)` array of
  points on both surfaces, `BVH.distance(a, b, max_distance=...)`, and `raycast`.
  Build from a `BufferGeometry` or from numpy (`build_arrays(positions, indices)`).
  Every query releases the GIL; `intersects_many`, `intersect_many` and
  `distance_many` evaluate a candidate x target grid in parallel in one call:

  ```python
  volumes = [tp.BVH() for _ in tracks]
  for bvh, track in zip(volumes, tracks):
      g = tp.TubeGeometry(tp.LineCurve3(*track), radius=5.0, radial_segments=8)
      bvh.build_arrays(g.get_attribute("position"), g.get_index())

  hits = tp.BVH.intersects_many(candidates, volumes)      # (C, T) bool
  when = tp.BVH.intersect(candidates[0], volumes[2])[:, 1]  # y of each contact point
  ```
- **Layers**: `obj.layers` / `camera.layers` (`set`, `enable`, `disable`,
  `test`) — the three.js visibility mask.
- **Rigid-body physics** (when built with PhysX, see below): `PhysxWorld` +
  `RigidBody` — add `Mesh`es as dynamic/static bodies (box/sphere/capsule, convex
  hull, or triangle mesh), `step(dt)`, and the bound meshes follow the simulation
  (`tp.HAS_PHYSX`). Reduced-coordinate `Articulation`s (URDF robots with joints,
  limits and drives) are bound too.
- **Proprioceptive sensors** (PhysX builds): `Imu` (gyroscope + accelerometer),
  `JointEncoder`, `ContactSensor` and `ForceTorqueSensor` — attached to scene
  nodes, driven from the physics step loop, with configurable seeded
  `NoiseModel`s. See *Proprioceptive sensors* below.
- **Vulkan deferred renderer + G-buffer AOVs** (when built with Vulkan, see
  below): `VulkanRenderer.render_aov(scene, camera, aov)` returns a deferred
  G-buffer attachment as `(H, W, 3)` uint8 — `'rgb'`, `'normals'`,
  `'segmentation'` (per-instance ids), `'albedo'`, `'motion'` — and the lossless
  typed family reads the native attachments: `read_depth` (**metric** `(H, W)`
  float32), `read_instance_ids` (**raw** uint32 ids), `read_normals_float`,
  `read_motion`, `read_aovs_typed`. This is the "labels for free" path for
  synthetic-data generation.
- **FFT ocean** (renders under the Vulkan backend): `tp.Ocean(size=1000.0)` is a
  ready-made 3-cascade Phillips/FFT-displaced water surface — waves, foam, and
  transmission. Add it to a scene and render. Tune `ocean.params` (wind / cascades
  / choppiness), pack vertex density toward any focus with `warp_toward(x, z)`, add
  `add_foam_disturbance(...)` splats, and read the wave height on the CPU with
  `sample_height(x, z)` (lazy opt-in: the first call enables the GPU->CPU
  mirror and may return 0 — values are live from the next rendered frame).
  The low-level `tp.DisplacedMesh` (own geometry + material)
  is exposed too. Not tied to any "hero" object — see
  [`examples/vulkan_ocean.py`](examples/vulkan_ocean.py).

Naming follows Python conventions (`snake_case` methods/properties), e.g.
`camera.update_projection_matrix()`, `renderer.set_clear_color(...)`.

## Tests

A pytest suite under [`tests/`](tests/) covers the whole surface — math,
scene graph, geometries, materials, cameras, lights, textures, loaders and
headless rendering — including regression tests that pin the pybind11
virtual-base workaround (they crash the interpreter if it ever regresses).

```sh
pip install pytest numpy pillow
cd python
pytest
```

[`examples/smoke_test.py`](examples/smoke_test.py) is a dependency-light
standalone alternative that prints `ALL OK`.

### Type stubs (IDE autocomplete)

Stubs ship alongside the module as a PEP 561 stub *package*, mirroring the
native module's own structure:

```
threepp/py.typed                 # PEP 561 marker (makes the types count once installed)
threepp/threepp/__init__.pyi     # the threepp.threepp extension module
threepp/threepp/imgui.pyi        # the threepp.threepp.imgui submodule
threepp/threepp/editor.pyi       # the threepp.threepp.editor submodule — HAND-MAINTAINED
```

This replaces the old flat `threepp/threepp.pyi`, which could not describe the
`imgui` submodule. The `threepp/threepp/` directory sits next to
`threepp.<abi>.pyd` / `.so` but does **not** shadow it at import time: it has no
`__init__.py`, so it is only a namespace-package candidate, and Python's import
machinery prefers the extension-module loader within the same directory.

Regenerate after changing the bindings:

```sh
pip install -r python/requirements-stubs.txt
cmake --build build --target threepp_stubs
```

Or, once the module is built into the package dir, standalone:

```sh
python python/scripts/gen_stubs.py
```

`pybind11-stubgen` is **pinned** in
[`requirements-stubs.txt`](requirements-stubs.txt) because its output layout and
formatting change between releases — an unpinned upgrade is what silently
reshaped the stubs from a flat file into this package. Bump the pin
deliberately, regenerate, and review the diff.

[`scripts/gen_stubs.py`](scripts/gen_stubs.py) wraps the generator so the result
is reproducible: it fixes the flags, repairs any keyword-named binding (see
below), **fails if the emitted stub does not parse**, and **fails if
regeneration dropped symbols the committed stubs declare**. The bare
`pybind11-stubgen` CLI exits 0 even when it writes a stub no type checker can
read, which is how the previous stubs went stale unnoticed.

#### Regenerate from a full-featured build

The stubs describe whatever the *built* module exposes, so regenerate from a
build with the same features as the last one — Vulkan, PhysX, FSR. A default or
GL-only rebuild drops `VulkanRenderer`, the PhysX world and every sensor
binding, and the result still parses, so nothing else would catch it.

`gen_stubs.py` guards this: it snapshots the committed stubs' symbols before
regenerating and **fails if any disappeared**, naming them. If a reduction is
genuinely intended, say so:

```sh
python python/scripts/gen_stubs.py --allow-removals
```

#### `editor.pyi` is hand-maintained

Every other stub here is generated; `editor.pyi` is written by hand, and
`gen_stubs.py` restores it verbatim after each run (`HAND_MAINTAINED`).

`threepp.editor` is served by two modules built from the same binding sources:
this wheel, and the editor app's embedded interpreter. Only the editor compiles
[`src/bind_editor_physics.cpp`](src/bind_editor_physics.cpp) — `RigidBody`,
`SoftBody`, `Articulation` and the three `*_from_object` lookups are handles onto
a live `PhysicsPlaySession`, which nothing outside a running editor has, so the
wheel does not bind them. The stub has to describe the **union** for a script
author's completion to be right, and a stub generated from the wheel alone would
delete the physics half (41 symbols).

So: change `bind_editor_physics.cpp`, and edit `editor.pyi` to match. The
restore happens *after* the produced-files check, so a `threepp.editor` that
disappeared from the module still fails the run rather than being papered over
by the committed copy.

The union stays in the source tree for the editor's Pylance integration, but it
does **not** ship in the wheel: the wheel carries `py.typed`, so an installed
stub is *certified* to a type checker, and certifying 40-odd names that raise
`AttributeError` on a pip install would be lying. The wheel instead installs
[`wheel-stubs/editor.pyi`](wheel-stubs/editor.pyi) — just `SplinePath` and
`spline_from_object`, the two names `bind_editor.cpp` actually provides there.
A name added to `bind_editor.cpp` belongs in **both** stubs; a name added to
the editor-only TUs belongs only in the union.

#### Keyword-named bindings

A bound name that is a Python keyword cannot be written in Python source *or* in
a stub, and one is enough to make the whole stub unparseable. Two such names
have been renamed at the binding site:

| Was | Now | Why |
| --- | --- | --- |
| `Blending.None` | `Blending.NoBlending` | `tp.Blending.None` is a SyntaxError; `NoBlending` is also the three.js name |
| `damp(..., lambda=...)` | `damp(..., lambda_=...)` | `lambda` is unusable as a keyword argument; PEP 8's trailing underscore |

No binding needs repair today. `gen_stubs.py` keeps a general repair pass as a
standing guard, since pybind11 will emit the same breakage from any future C++
identifier that collides with a Python keyword.

A few signatures show `...` instead of a real parameter type (e.g.
`Vector3.apply_matrix4`). That is pybind11 baking a signature before the
argument's C++ type was registered; the fix is binding declaration order, not
the stub generator.

## Vulkan deferred AOVs (synthetic data)

The deferred (RasterFirst) Vulkan renderer writes a full G-buffer every frame —
world normals, optical flow, per-instance segmentation ids, albedo, depth. The
binding exposes those attachments as numpy, so a scene authored in Python yields
ground-truth labels directly from the render, with no manual annotation:

```python
import threepp as tp
assert tp.HAS_VULKAN

canvas = tp.Canvas("aov", width=512, height=384, headless=True, vsync=False)
renderer = tp.VulkanRenderer(canvas)            # deferred / RasterFirst

# ... build scene + camera ...
out = renderer.render_aovs(scene, camera, ["rgb", "normals", "segmentation"])
rgb, normals, seg = out["rgb"], out["normals"], out["segmentation"]   # (H, W, 3) uint8
depth = renderer.read_depth(scene, camera)                           # (H, W) float32, metres
```

Enable the Vulkan backend at configure time (needs the Vulkan SDK + a
Vulkan-capable GPU). The GL renderer stays available in the same module:

```sh
cmake -S . -B build -DTHREEPP_WITH_PYTHON=ON -DTHREEPP_WITH_VULKAN=ON
cmake --build build --target threepp_py
```

`tp.HAS_VULKAN` reports whether the backend was compiled in.

**Two readback paths.** The 8-bit visualisation AOVs (`render_aov` /
`render_aovs`) come out as `(H, W, 3)` uint8 via the renderer's debug-resolve
pass: normals as `n*0.5+0.5`, segmentation as per-id hashed colours, albedo as
*linear* albedo (so it looks gamma-/hue-off as a viewed image, but the data is
the real linear base colour). The **lossless** path copies the native G-buffer
attachments straight to host memory: `read_depth` → **metric** `(H, W)` float32
(full-precision native D32 read, verified accurate against known distances),
`read_instance_ids` → **raw** `(H, W)` uint32 per-instance ids,
`read_normals_float`, `read_motion`, and `read_aovs_typed([...])` for several in
one render. Driving is via the deferred frame-model under the hood
(submit/present is deferred to the canvas frame-end callback, so each `render*`
repeats a few frames to make the MAILBOX readback deterministic — tune with
`set_flush_frames`).

**Headless / display-less machines (cloud GPUs).** A `headless=True` canvas
needs no display at all: the Vulkan renderer creates its surface via
`VK_EXT_headless_surface` (supported by NVIDIA's Linux driver and Mesa) instead
of a window surface, and on Linux with no `DISPLAY`/`WAYLAND_DISPLAY` the
canvas skips the window system entirely (GLFW Null platform). That is exactly
the Colab/EC2-style setup — compute-only NVIDIA kernel modules, no X server the
GPU can present to — where the wheel's synthetic-data path
(`render_aov` / `read_depth` / `read_rgb_pixels`) runs unmodified; presenting
to a headless surface is simply a no-op. Where the ICD lacks the extension
(e.g. NVIDIA's *Windows* driver), a headless canvas falls back to the previous
behaviour — a hidden window with a real surface — and logs that it did so.
Set `THREEPP_GLFW_PLATFORM=null` to force the window-system-free path on any
OS (useful for reproducing the cloud setup locally).

## In-window UI (Dear ImGui)

threepp integrates Dear ImGui; the binding exposes it as `ImguiContext` plus the
`threepp.imgui` submodule of immediate-mode widgets. Issue widgets each frame
inside a draw callback, and call `ui.render(draw)` in the animate loop after
`renderer.render(...)`:

```python
ui = tp.ImguiContext(canvas)        # create AFTER the GLRenderer

def draw():
    tp.imgui.begin("Controls")
    changed, rough = tp.imgui.slider_float("roughness", material.roughness, 0.0, 1.0)
    if changed: material.roughness = rough
    _, material.wireframe = tp.imgui.checkbox("wireframe", material.wireframe)
    if tp.imgui.button("reset"): camera.position.set(0, 2, 6)
    tp.imgui.end()

def animate():
    controls.enabled = not ui.want_capture_mouse   # don't orbit while over the panel
    controls.update()
    renderer.render(scene, camera)
    ui.render(draw)

canvas.animate(animate)
```

Value-returning widgets give back `(changed, new_value)` tuples; `button` returns
`True` on the clicked frame. `tp.imgui.show_demo_window()` opens the full ImGui
gallery. `tp.HAS_IMGUI` reports availability.

It works on **both** renderers — pass the renderer so the backend matches:
`ImguiContext(canvas, gl_renderer)` (or `ImguiContext(canvas)`) for GL, and
`ImguiContext(canvas, vulkan_renderer)` for the Vulkan overlay (recorded into the
deferred frame after the scene). Create the `ImguiContext` after the renderer. See
[examples/vulkan_ui.py](examples/vulkan_ui.py) for the Vulkan version.

One caveat: only **one** `ImguiContext` should be alive at a time (Dear ImGui has a
single global context) — don't keep a GL and a Vulkan one simultaneously.

## Rigid-body physics (PhysX)

threepp ships a scene-graph-integrated PhysX wrapper; the binding exposes it as
`PhysxWorld` (+ a `RigidBody` handle). Add meshes as bodies, `step(dt)` each frame,
and the bound meshes' transforms follow the simulation. It's pure CPU — **no canvas
or renderer required**, so it works headless:

```python
import threepp as tp
assert tp.HAS_PHYSX

world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0))

floor = tp.Mesh(tp.BoxGeometry(20, 1, 20), tp.MeshStandardMaterial())
floor.position.y = -0.5
world.add_static(floor)                 # static collider

box = tp.Mesh(tp.BoxGeometry(1, 1, 1), tp.MeshStandardMaterial())
box.position.set(0, 5, 0)
body = world.add(box, density=200)      # dynamic body, auto-bound to the mesh
body.add_impulse(tp.Vector3(2, 0, 0))

for _ in range(120):
    world.step(1 / 60)                   # box.position now follows the sim
```

Shapes are inferred from Box/Sphere/Capsule geometry; `add_dynamic_convex` (convex
hull), `add_static_trimesh` / `add_static_trimesh_tree` (exact triangles, e.g. an
imported glTF environment), and `add_instanced` (one body per `InstancedMesh`
instance) cover the rest. `RigidBody` exposes pose, linear/angular velocity,
`add_force` / `add_impulse`, damping, and a kinematic mode
(`set_kinematic` + `set_kinematic_target`). `on_pre_substep` / `on_post_substep`
hooks fire around each fixed substep.

Enable PhysX at configure time. It comes from vcpkg
(`unofficial-omniverse-physx-sdk`), so configure with the vcpkg toolchain:

```sh
cmake -S . -B build -DTHREEPP_WITH_PYTHON=ON \
      -DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake
cmake --build build --target threepp_py
```

(Or, to reuse an existing install without the toolchain, pass
`-Dunofficial-omniverse-physx-sdk_DIR=<vcpkg_installed>/x64-windows/share/unofficial-omniverse-physx-sdk`;
the build then stages the PhysX runtime DLLs next to the module.) `tp.HAS_PHYSX`
reports whether it was compiled in. Reduced-coordinate **articulations** (robots)
are bound too — `Articulation` / `ArticulationLink`, built from a URDF in one
call. Soft bodies and vehicles (which need the CUDA/GPU path) are not exposed
yet.

Combined with the Vulkan AOVs, this is the **dynamic** half of the synthetic-data
story: physics gives you moving scenes, the G-buffer gives you per-frame
segmentation / depth / optical-flow labels for free.

## Proprioceptive sensors (IMU)

On top of the PhysX world, threepp exposes a **proprioceptive sensor** suite for
robotics: `Imu` (gyroscope + accelerometer), `JointEncoder`, `ContactSensor` and
`ForceTorqueSensor`, each with seeded noise models. The walkthrough below uses
the `Imu`. A sensor **rides the scene graph**: you attach it to an
`Object3D`, and that node's world frame *is* the measurement frame. Register it
with the world and it is sampled from the physics step loop — one clean sample
per fixed substep, timestamped with the accumulated sim time — so the sampling
fidelity is right for lock-step co-simulation (e.g. a future ArduPilot SITL).

```python
import threepp as tp
assert tp.HAS_PHYSX

world = tp.PhysxWorld(gravity=tp.Vector3(0, -9.81, 0), fixed_timestep=1/240)
world.add_static(floor_mesh)                 # top face at y=0
body = world.add(box_mesh, density=200)      # dynamic body

mount = tp.Group()                           # IMU node, offset from the CoM
mount.position.set(0.3, 0.2, 0.1)
box_mesh.add(mount)

imu = tp.Imu(mount)                          # rate_hz=0 -> sample every substep
world.register_sensor(imu)                   # AFTER adding the body it rides

for _ in range(720):
    world.step(1/240)

for s in imu.drain():                        # oldest-first; empties the buffer
    print(s.t, s.angular_velocity, s.linear_acceleration)
data = imu.drain_array()                     # or a (N, 7) numpy [t, gx,gy,gz, ax,ay,az]
last = imu.latest()                          # most recent sample (survives drain)
```

**Contract.** Attach to a node; set `rate_hz` (0 = every physics substep, else
the sensor sub-samples). Read non-blocking: `latest()` is the most recent sample,
`drain(out)` / `drain_array()` move everything accumulated since the last drain
out of a bounded ring buffer (default 2048; oldest dropped on overflow).
`register_sensor` resolves the rigid body by walking up from the attachment node
to the nearest ancestor added to the world — it raises immediately if there is
none.

**IMU units & frames.** Both readings are expressed in the **sensor (node)
frame**:

| Field | Meaning | Units |
| --- | --- | --- |
| `angular_velocity` | the body's angular velocity | rad/s |
| `linear_acceleration` | **specific force** `f = R⁻¹·(a_point − g)` | m/s² |

Specific force is what a real accelerometer reads (proper acceleration minus
gravity, threepp is Y-up so `g = (0, −9.81, 0)`), so a **level body at rest reads
`(0, +9.81, 0)`** and a body in **free fall reads ~0**. When the node is offset
from the centre of mass, the lever-arm terms `α × r + ω × (ω × r)` are included
(`a_com` and `α` come from finite-differencing the PhysX velocities across
samples; the first sample after attach/`reset()` emits zero to avoid a start-up
spike).

**Noise.** `imu.gyro_noise` and `imu.accel_noise` are `NoiseModel`s with per-axis
continuous-time densities — `white_noise_density` [X/√Hz], `random_walk`
[X/(s·√Hz)] bias instability, and a constant `constant_bias` [X] — plus a `seed`
(deterministic given seed + call sequence). Defaults are consumer MEMS-class; set
every field to zero for a perfect sensor. Change the noise, then call
`imu.reset()` (which also re-arms the finite-difference after an episode reset).

See [`examples/imu_demo.py`](examples/imu_demo.py) for the full headless demo.

## Notes for maintainers

threepp uses **virtual inheritance** for some classes (`Mesh`/`Points`/`Line`
derive from `Object3D` virtually; concrete materials derive from `Material`
virtually). pybind11 mishandles pointer adjustment across a virtual base, which
corrupts memory. The bindings work around this by:

- never letting pybind up-cast a derived Python object to a `shared_ptr` of a
  virtual base — `as_object3d` / `as_material` cast to the concrete type and let
  the C++ compiler do the up-cast;
- binding the inherited `Object3D` / `Material` members on each concrete leaf
  with **concrete** member pointers (`&T::field`) and `T&` lambdas, instead of
  inheriting the (broken) base bindings.

See `src/bind_objects.cpp` (`bind_object3d_api`) and `src/bind_materials.cpp`
(`bind_material_base_fields`).
