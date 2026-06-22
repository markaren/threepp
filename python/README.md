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
> tag builds a `YYYY.MM.DD` wheel); pin a version (`threepp==2026.6.17`) and
> check the release notes before upgrading.

Prebuilt **GL-only** wheels (Windows / macOS / Linux, CPython 3.10–3.13) are
attached to each [GitHub release](https://github.com/markaren/threepp/releases) —
no build tools or system libraries needed. Download the wheel matching your OS +
Python from the Releases page, then:

```sh
pip install ./threepp-2026.6.17-cp312-cp312-win_amd64.whl
```

Or build from source (needs a C++ compiler + CMake ≥ 3.19; pybind11 is fetched
automatically):

```sh
pip install "git+https://github.com/markaren/threepp"
```

The wheel is **GL-only** by design — it needs no Vulkan SDK, PhysX, or CUDA and
runs on any machine. The Vulkan path-tracer / deferred AOVs and the PhysX physics
+ GPU-RL backends are opt-in **source** builds (see the backend sections below).

## Build

The module is built as part of threepp's CMake project, gated behind
`THREEPP_WITH_PYTHON`. pybind11 is fetched automatically; the host Python is
auto-detected (override with `-DPython_EXECUTABLE=...`).

```sh
cmake -S . -B build -DTHREEPP_WITH_PYTHON=ON \
      -DTHREEPP_BUILD_EXAMPLES=OFF -DTHREEPP_BUILD_TESTS=OFF
cmake --build build --target threepp_py
```

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
| [`examples/ui_demo.py`](examples/ui_demo.py) | In-window Dear ImGui control panel (sliders/buttons) driving the scene live (GL). Needs a display. |
| [`examples/vulkan_ui.py`](examples/vulkan_ui.py) | The same ImGui control panel, over the **Vulkan** deferred renderer. Needs a Vulkan build + display. |
| [`examples/physics_demo.py`](examples/physics_demo.py) | A pile of boxes tumbling onto the floor — `PhysxWorld` rigid bodies driving the scene graph. Needs a PhysX build + display. |
| [`examples/spider/spider_demo.py`](examples/spider/spider_demo.py) | **Drive a physics hexapod with WASD** — an articulated robot walking via a CPG tripod gait (no training). Needs a PhysX build + display. |
| [`examples/spider/train_vec.py`](examples/spider/train_vec.py) / [`play.py`](examples/spider/play.py) | Residual RL on the gait. `train_vec.py` runs **K robots in one PhysX scene** (one `world.step()` for all of them, ~7–35× faster than separate processes); `play.py` drives the trained policy with WASD. `train.py` is the slower `SubprocVecEnv` variant. Needs `gymnasium`, `stable-baselines3`. |
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
  `Torus`, `TorusKnot`, `Circle`, `Ring`, `Icosahedron`, `Octahedron`.
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
- **Rigid-body physics** (when built with PhysX, see below): `PhysxWorld` +
  `RigidBody` — add `Mesh`es as dynamic/static bodies (box/sphere/capsule, convex
  hull, or triangle mesh), `step(dt)`, and the bound meshes follow the simulation
  (`tp.HAS_PHYSX`).
- **Vulkan deferred renderer + G-buffer AOVs** (when built with Vulkan, see
  below): `VulkanRenderer.render_aov(scene, camera, aov)` returns a deferred
  G-buffer attachment as `(H, W, 3)` uint8 — `'rgb'`, `'normals'`,
  `'segmentation'` (per-instance ids), `'albedo'`, `'motion'` — and
  `read_depth(scene, camera)` returns **metric depth** as `(H, W)` float32
  (distance from the camera in scene units). This is the "labels for free" path
  for synthetic-data generation.

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

`threepp.pyi` ships alongside the module. Regenerate it after changing the
bindings with:

```sh
pip install pybind11-stubgen
cmake --build build --target threepp_stubs
```

## Vulkan deferred AOVs (synthetic data)

The deferred (RasterFirst) Vulkan renderer writes a full G-buffer every frame —
world normals, optical flow, per-instance segmentation ids, albedo, depth. The
binding exposes those attachments as numpy, so a scene authored in Python yields
ground-truth labels with no path tracer and no manual annotation:

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

**Current scope / honest limits.** The colour AOVs come out as `(H, W, 3)` uint8
via the renderer's debug-resolve pass: normals as `n*0.5+0.5`, segmentation as
per-id hashed colours, albedo as *linear* albedo (so it looks gamma-/hue-off as
a viewed image, but the data is the real linear base colour). **Depth is metric
float32** (`read_depth`) — the debug-resolve pass packs the reverse-Z depth into
24 bits, which the host decodes and linearizes with the camera near/far;
verified accurate to a fraction of a unit against known distances. Still not
exposed: **raw integer** instance ids (segmentation comes back as hashed colours,
not the underlying ids), which wants a device-buffer readback. Driving is via the
deferred frame-model under the hood (submit/present is deferred to the canvas
frame-end callback, so each `render*` repeats a few frames to make the MAILBOX
readback deterministic — tune with `set_flush_frames`).

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
reports whether it was compiled in. Soft bodies and vehicles (which need the
CUDA/GPU path) are not exposed yet — rigid bodies only.

Combined with the Vulkan AOVs, this is the **dynamic** half of the synthetic-data
story: physics gives you moving scenes, the G-buffer gives you per-frame
segmentation / depth / optical-flow labels for free.

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
