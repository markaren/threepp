# threepp scene editor

A desktop scene editor for threepp: build a scene, tweak it, save it as three.js
JSON, and press Play to hand it to a runtime (PhysX ships in the box).

```
cmake -S . -B build -DTHREEPP_BUILD_EDITOR=ON
cmake --build build --target threepp_editor
./build/bin/threepp_editor [scene.json]
```

`THREEPP_BUILD_EDITOR` defaults to ON when threepp is the top-level project and
GLFW is enabled, and OFF for anyone consuming threepp as a subproject.

Command line:

| flag | meaning |
| --- | --- |
| `scene.json` | open this document on start |
| `--vulkan` | use the Vulkan backend (OpenGL is the default and the supported path) |
| `--urdf=<file>` | selftest only: also exercise the URDF import and round trip |
| `--frames=N` | render N frames and exit — for smoke tests |

---

## Architecture

Two layers, deliberately separated:

**`threepp/extras/editor/*` — the reusable core.** Part of the threepp library,
with no ImGui, no window and no GL/Vulkan dependency. It is what a different
front end (a Qt tool, a Python binding, a batch script) would build on:

| type | responsibility |
| --- | --- |
| `Command`, `CommandStack` | undo/redo with drag coalescing |
| `SetTransformCommand`, `PropertyCommand<T>`, `AddObjectCommand`, `RemoveObjectCommand`, `ReparentCommand`, `SetMaterialMapCommand` | the concrete edits |
| `Selection` | the current object + change notifications |
| `SceneDocument` | the open Scene, its path, its dirty flag, save/open |
| `SceneSnapshot` | scene ⇄ JSON, with live textures preserved |
| `ObjectFactory` | primitives, lights, groups, cameras with unique names |
| `PlaySession`, `PlayController` | the play-mode state machine |
| `PhysicsConfig` | per-object rigid-body authoring, stored in `userData` |
| `PhysicsPlaySession` | the PhysX runtime (header-only, PhysX-gated) |
| `EditorSettings` | recent files and last-used directories |

**`apps/editor/*` — the application.** `EditorApp` owns the canvas, renderer,
editor camera, `OrbitControls`, `TransformControls`, raycaster and ImGui context,
and the panels are its member functions split across `panels/*.cpp`.

---

## Features

**Viewport.** Grid + origin axes, orbit camera, click to pick, bounding-box
highlight on the selection, full transform gizmo (`TransformControls`) with
local/world space and snapping. Orbiting is disabled while a gizmo handle is
being dragged.

Picking selects the top-level object of whatever was hit — clicking an imported
model selects the model, not one of its sub-meshes. Click again inside the same
subtree to drill down to the exact node.

**Markers.** Cameras and lights render nothing, so each one gets a billboarded
icon at a constant screen size, tinted with the accent colour while selected.
Every kind has its own shape — a camera body, a sun for directional, a bulb for
point, a beam for spot, a ringed core for ambient, a dome over ground for
hemisphere — so a scene reads without clicking anything.
Clicking an icon selects its owner, and it wins over geometry behind it — the
icons draw on top, so picking follows what you see rather than raw depth order.
The artwork is SVG parsed at startup by threepp's `SVGLoader` and embedded as
source in `ViewportMarkers.cpp`, so the editor does not depend on finding asset
files at runtime. Selecting a camera additionally shows its frustum
(`CameraHelper`), which tracks fov/near/far edits live. Objects that bound to
nothing get no selection box — an empty `Box3` would leave `BoxHelper` showing a
degenerate speck at the origin.

**Hierarchy.** The full `Object3D` tree, selection synced both ways with the
viewport, double-click to rename, drag-and-drop to reparent (undoable, world
transform preserved), and a right-click menu with Add ▸ / Duplicate / Delete /
Focus.

**Inspector.** Object flags, transform (rotation shown in degrees), material
(type, colour, emissive, roughness, metalness, opacity, transparency, side,
wireframe, flat shading, six texture slots with previews), read-only geometry
counts, light parameters, camera parameters, and physics. Every edit is
undoable, and a drag collapses into a single undo step.

**Bottom panel.** An asset browser (double-click a file to open/import/assign)
and a console showing loader and exporter warnings. Collapsible.

**Files.** New / Open / Save / Save As, Import Model, Set Environment (.hdr),
Recent Files. All dialogs are a first-party ImGui file browser — no native
dialog library. Drag-and-drop onto the window works too: `.json` opens a scene,
`.hdr` becomes the environment, model files are imported, image files become the
selected mesh's base-colour map.

**Play mode.** ▶ snapshots the scene and starts the runtimes, ⏸ suspends
stepping, ⏹ stops them and restores the snapshot. The inspector is read-only
while playing and a banner sits over the viewport.

---

## Keyboard shortcuts

| key | action |
| --- | --- |
| `W` / `E` / `R` | translate / rotate / scale gizmo |
| `Q` | toggle local / world space |
| `Shift` (hold) | snap while dragging |
| `F` | frame selection |
| `Del` | delete selection |
| `Esc` | deselect |
| `Ctrl+D` | duplicate |
| `Ctrl+Z` | undo |
| `Ctrl+Y`, `Ctrl+Shift+Z` | redo |
| `Ctrl+N` / `Ctrl+O` / `Ctrl+S` | new / open / save |

Shortcuts are suppressed whenever ImGui wants the keyboard (a text field has
focus, a modal is open), and picking is suppressed whenever ImGui wants the
mouse.

---

## File format

The editor's document format is **three.js "Object" JSON, metadata version
4.5** — exactly what `ObjectExporter` writes and `ObjectLoader` reads. There is
no editor-specific format and no sidecar file.

Two consequences worth knowing:

* Every uuid (object, geometry, material, texture) is preserved across a
  round-trip, which is what lets the editor re-resolve the selection after a
  reload or a play-mode restore.
* Output is deterministic: saving an unchanged scene twice produces
  byte-identical files, so documents diff cleanly in version control.

Saved documents embed texture images as base64 PNG data-URIs, so a scene file
stands on its own.

### Editor-only objects

The grid, origin axes, selection outline and transform gizmo have to be in the
scene graph to be rendered, but must never end up in a saved file.
`ObjectExporter` has no exclusion filter, and adding one would have meant
changing a format-level API for a tool-level concern. Instead the editor keeps
all of them under a single overlay `Group` registered with
`SceneDocument::addEditorOnly()`, and the document detaches that node for the
duration of every export (save *and* play snapshot) and re-attaches it
afterwards. The cost is one detach/attach per save; the benefit is that
`ObjectExporter` stays a pure serializer.

### Physics in `userData`

Per-object physics rides in `object.userData["physics"]`, so it serializes with
the scene for free. `userData` round-trips **scalars only** (bool / int / float /
double / string — see `ObjectExporter::writeUserData`), so a nested JSON object
is not available. The value is therefore one flat, deterministic string:

```
body=dynamic;shape=convex;mass=12.5;friction=0.8;restitution=0.35
```

`PhysicsConfig::encode()` / `decode()` own that format. Unknown keys are ignored
on read, so a document written by a newer editor still loads. A disabled body
removes the entry entirely rather than writing `enabled=false`, so turning
physics off leaves no trace in the file.

### Animations in `userData`

Animation clips themselves already round-trip through the three.js JSON
(`ObjectExporter`/`ObjectLoader` serialize `Object3D::animations`), and every
model loader attaches a file's clips to the imported root group. The editor
adds authoring on top, in `object.userData["animation"]`, same flat format as
physics:

```
autoplay=1;loop=1;speed=1.25;clip=Walk
```

`AnimationConfig` owns the format. The **Animation** inspector section appears
on any object that carries clips: pick the clip, toggle *Play on Start* and
*Loop*, set the speed, and *Preview* it right in the viewport — preview records
every node transform (and morph influences) up front and restores them on stop,
so it never dirties the document or the undo stack.

During Play, `AnimationPlaySession` (always registered) walks the scene and
plays the authored clip on every object with `autoplay` on — the default for
anything that has clips, so an imported character moves the first time you
press Play. The play snapshot restores all poses on Stop.

### Robots (URDF) in `userData`

Importing a `.urdf` / `.xacro` routes to `URDFLoader` instead of `ModelLoader`
and yields a `Robot` — an `Object3D` that additionally owns a joint table. The
inspector shows a **Joints** section with one slider per articulated DOF,
labelled with the joint's URDF name, clamped to its limits, in degrees for
revolute joints and metres for prismatic ones. Edits are undoable and coalesce
per drag; **Home** returns every joint to zero as a single undo entry.

None of that articulation can go into the three.js JSON, which knows only about
transforms — a saved robot would come back correctly posed but frozen. So the
document stores a reference instead:

```
userData["urdf"]        C:/models/lbr_iiwa_14_r820.urdf
userData["jointValues"] 0.3,0,-1.2,0,0,0,0
```

Joint values are always radians/metres regardless of what the slider displays.
On load — including the play snapshot restore — `EditorApp::rearticulateRobots`
re-imports each referenced URDF and transplants the live `Robot` over the frozen
placeholder, keeping its uuid, name and placement so selection and undo
rebinding still resolve, then reapplies the stored pose. The geometry is still
written to the document, so a scene whose URDF has moved away (or one opened by
another tool) renders exactly as saved — it just cannot be re-jointed, and says
so in the console.

### Async model import

`Import Model…` and file drops never block the UI: loads run one at a time on
a worker thread, with a spinner toast (bottom-left) showing the file and queue
depth. Success selects the new group and flashes a summary in the status bar;
failure opens a dialog with the loader's error. An import that finishes during
Play is parked until Stop, so it cannot be swallowed by the snapshot restore.
Passing a model path on the command line (`threepp_editor model.glb`) imports
it into a fresh template scene.

### One library fix this shook out

`Object3D::copy()` did not copy `userData`, so `clone()` silently dropped it —
a duplicated object lost its physics setup (and anything else an application had
attached). three.js copies `userData` in `Object3D.copy`, so this was a parity
bug rather than a design choice; it is fixed, with a test in
`tests/core/Object3d_test.cpp`.

---

## Extending Play mode with your own runtime

`PlaySession` is three methods:

```cpp
class MyRuntime : public threepp::editor::PlaySession {
public:
    void start(threepp::Scene& scene) override;  // build your state from the scene
    void update(float dt) override;              // one frame; not called while paused
    void stop() override;                        // release everything
    std::string name() const override { return "MyRuntime"; }
};
```

Register it before the first Play:

```cpp
playController.addSession(std::make_shared<MyRuntime>());
```

**Your session does not have to be reversible.** Before any session starts, the
editor captures the whole scene with `SceneSnapshot`; on Stop it parses that
snapshot back into a fresh `Scene` and hands it to the document. Move objects,
retarget materials, delete half the graph — none of it survives Stop.

Two details that make this cheap and lossless:

* The snapshot does **not** embed images. It keeps the live
  `std::shared_ptr<Texture>` objects keyed by uuid and re-binds them into the
  restored materials, so no texture is ever re-decoded or re-uploaded.
* Sessions start in registration order. If one throws from `start()`, the ones
  that already started are stopped again and the whole play attempt is
  abandoned — the editor never ends up half-playing.

The shipped `PhysicsPlaySession` reads `PhysicsConfig` from every object,
creates a static/dynamic/kinematic `PxRigidActor` with the requested shape, and
lets `PhysxWorld` write the poses back into the scene graph each step. It is
header-only and compiled only when the PhysX SDK is found (vcpkg feature
`physx`); without it the editor still authors and saves physics settings, and
Play simply runs with no physics session.

---

## Known limitations

* **Single selection.** Multi-select, box-select and group transforms are not
  implemented.
* **The viewport is the whole window.** Panels are drawn over a full-window
  render rather than into a framebuffer sized to the free space, so the camera
  aspect is the window aspect. Picking and gizmo input are suppressed wherever
  ImGui owns the mouse, so this is invisible in use, but a very wide inspector
  does shift the visual centre.
* **Texture previews are mosaics.** A `Texture`'s GPU handle belongs to the
  renderer and is not exposed, so previews are drawn as a 12×12 grid of filled
  rects sampled from the CPU-side image. It reads correctly as a thumbnail at
  inspector size and behaves identically on every backend, but it is not a
  full-resolution image.
* **Vulkan is best-effort.** `--vulkan` is wired through `createRenderer` and
  `ImguiContext` already supports the Vulkan backend, but OpenGL is the tested
  and supported path. The editor always names its backend explicitly —
  `createRenderer`'s "no preference" overload prints a console menu and blocks
  on `std::cin`, which a windowed application must never do.
* **Duplicate shares geometry.** `Object3D::clone()` shares both geometry and
  materials; the editor clones the materials afterwards so recolouring a copy
  does not recolour the original, but geometry stays shared. That is intentional
  (duplication stays cheap), and it means a future mesh-editing feature would
  need an explicit "make unique" step.
