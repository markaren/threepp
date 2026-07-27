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
`THREEPP_EDITOR_WITH_PYTHON` (also ON by default) builds the Python scripting
support and quietly turns itself off when no `Python3` with the
`Development.Embed` component is found.

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
| `ScriptConfig` | per-object Python script — a file path or inline source — plus its parameters, stored in `userData` |
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
(`CameraHelper`), which tracks fov/near/far edits live, and renders its view into
the camera dock beside the bottom panel. Objects that bound to
nothing get no selection box — an empty `Box3` would leave `BoxHelper` showing a
degenerate speck at the origin.

**Hierarchy.** The full `Object3D` tree, selection synced both ways with the
viewport, double-click to rename, drag-and-drop to reparent (undoable, world
transform preserved), and a right-click menu with Add ▸ / Duplicate / Delete /
Focus.

**Inspector.** Object flags, transform (rotation shown in degrees), material
(type, colour, emissive, roughness, metalness, opacity, transparency, side,
wireframe, flat shading, six texture slots with previews), read-only geometry
counts, light parameters, camera parameters, attached scripts and physics. Every
edit is undoable, and a drag collapses into a single undo step.

**Bottom panel.** A console showing loader and exporter warnings, and an asset
browser (double-click a file to open/import/assign). Console is the first tab —
it is where import results and errors land. Collapsible, and it runs to the left
edge of the window.

**Camera dock.** The band beside the bottom panel, under the inspector, renders
the selected camera live. It collapses with the bottom panel and paints itself
when nothing is selected, so that corner is never a sliver of viewport too small
to see into or click in.

**Panel sizes.** The hierarchy and inspector are resized by dragging their inner
edge, and the widths persist in the settings file. The hierarchy also scrolls
horizontally, because a deep tree indents past any fixed width and the panel
cannot be widened past the screen.

**Files.** New / Open / Save / Save As, Import Model, Set Environment (.hdr),
Recent Files. All dialogs are a first-party ImGui file browser — no native
dialog library. Drag-and-drop onto the window works too: `.json` opens a scene,
`.hdr` becomes the environment, model files are imported, `.py` attaches to the
selected object as a script, and image files become the selected mesh's
base-colour map.

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
inspector shows a **Robot** section with one slider per articulated DOF,
labelled with the joint's URDF name, clamped to its limits, in degrees for
revolute joints and metres for prismatic ones. Edits are undoable and coalesce
per drag; **Home** returns every joint to zero as a single undo entry.

A URDF also describes collision geometry, which `URDFLoader` builds as white
wireframe hulls sitting directly on top of the visual meshes — and leaves
visible. The editor hides them on import and offers **Show Colliders** to bring
them back. That choice is stored rather than treated as a transient view
setting, because Play rebuilds the robot from its URDF: a non-persisted toggle
would reset itself exactly when you wanted to watch the hulls. Hidden is the
default, so the key is only written when the toggle is on.

None of that articulation can go into the three.js JSON, which knows only about
transforms — a saved robot would come back correctly posed but frozen. So the
document stores a reference instead:

```
userData["urdf"]          C:/models/lbr_iiwa_14_r820.urdf
userData["jointValues"]   0.3,0,-1.2,0,0,0,0
userData["showColliders"] true      (omitted when hidden, which is the default)
```

Joint values are always radians/metres regardless of what the slider displays.
On load — including the play snapshot restore — `EditorApp::rearticulateRobots`
re-imports each referenced URDF and transplants the live `Robot` over the frozen
placeholder, keeping its uuid, name and placement so selection and undo
rebinding still resolve, then reapplies the stored pose. The geometry is still
written to the document, so a scene whose URDF has moved away (or one opened by
another tool) renders exactly as saved — it just cannot be re-jointed, and says
so in the console.

### Scripts in `userData`

Attach a script to any object and it becomes a behaviour, MonoBehaviour style.
On Play the editor instantiates the class and drives it; on Stop it calls
`stop()` and throws the instance away, and the usual snapshot restore puts the
scene back.

The code comes in one of two forms, never both: a `.py` **file** referenced by
path, or **inline source** stored in the scene itself and edited in the editor's
Script Editor window. A file is shared between objects and scenes and editable
in whatever tooling you already have; inline source travels with the scene, so a
`.json` handed to somebody else is complete.

**Opening a scene never runs anything.** Neither form executes on load, on
import, or on selection: scripts run when you press Play, and the inspector
imports a script's module only when it draws that object's parameters (a class
body has to execute for its attributes to exist). The Script Editor's Apply
compiles the source without executing it. A scene file is data, not a program.

```python
class Spinner:
    speed = 1.5          # exposed in the inspector, saved with the scene

    def start(self, obj):
        self.obj = obj

    def update(self, dt):
        self.obj.rotation.y += self.speed * dt

    def stop(self):
        pass
```

All three methods are optional; missing ones are skipped. In a file, the class is
the one whose name matches it (`spinner.py` → `Spinner`, case-insensitively), or
— failing that — the single class in the file that defines `update()`. Anything
else is reported rather than guessed at.

**Exposed parameters** are the class's plain `int` / `float` / `bool` / `str`
attributes: no leading underscore, not callable, no properties or descriptors.
They appear in the inspector as real widgets, undoable and coalescing per drag,
and their authored values are applied to the instance before `start()` runs. A
field the document has no value for shows the class attribute, which is exactly
what the script will see.

Like the URDF reference, this does not pack into one `key=value` string — a
Windows path contains the delimiters, and source text contains everything — so
each gets its own entry:

```
userData["script"]        C:/projects/scripts/spinner.py
userData["scriptSource"]  class Spinner:\n    speed = 1.5\n ...
userData["scriptFields"]  speed=1.5;clockwise=1;label=spin
```

`script` and `scriptSource` are mutually exclusive: assigning one clears the
other, and a document carrying both (hand-edited, or merged) is read as inline —
the scene's own copy of the code beats a path into someone else's disk — and
rewritten with one form the next time it is saved. `scriptFields` works
identically for both: the parameters belong to the class, not to where the class
came from.

Values are stored as text and typed on the way in, from the class attribute they
correspond to. That is what lets a build with no Python at all still round-trip a
script's parameters instead of silently dropping them.

**Editing the file and pressing Play again re-runs the new code.** The loader
reads and compiles the source itself rather than going through `importlib`,
whose `__pycache__` entry is validated against `(mtime, size)` — both of which
survive a same-second edit that happens not to change the file's length. That
one case is exactly the edit-run-edit loop this feature exists for, so there is
no bytecode cache to go stale.

#### The Script Editor

`New Inline Script` in the Script section writes a template into the object and
opens a floating **Script Editor** window on it — movable, resizable, closable,
and one instance: it edits the selected object's inline script, following the
selection while there is nothing to lose and **staying on the object it was
opened for whenever the buffer has unsaved edits**, because retargeting text
somebody is still typing would drop it without a word. `Edit…` reopens it, and
the title carries an asterisk while the buffer differs from the document.

* **Apply** (or `Ctrl+Enter`) commits the buffer as one undoable step, keyed per
  object so it lands in the undo stack next to every other inspector edit.
* Tabs become four spaces on the way in. The box takes real tabs — a Python
  editor that cannot indent is not one — and Python raises `TabError` on source
  that mixes them, so normalizing makes the failure impossible instead of
  diagnosing it later.
* The source is `compile()`d, never executed. A syntax error appears in red with
  its line number and is written to the console — but it does **not** block
  Apply: half-written code is a normal thing to want to save, and an editor that
  refuses to let go of what you typed is a worse editor.
* **Revert** reloads the committed text; **Clear** in the inspector removes the
  script and closes the window (one `Ctrl+Z` away, and the buffer is still there
  if you reopen the object).
* It is read-only while playing, like the inspector: the snapshot restore on Stop
  would throw the edit away.

Inline source is compiled **fresh on every Play** — it is inherently hot, with no
file, no cache and no stamp anywhere in the path. Each object's source gets its
own synthetic module named after its uuid, so two objects whose scripts both
define `class Behaviour` do not overwrite each other, and tracebacks are filed
under `<inline:Box>` — which object the code belongs to, since there is no file
name to point at. With no file stem to match, the class is the only one the
module defines, or failing that the single one defining `update()`; several
candidates is reported against the object, not guessed at.

Plain text, deliberately: no highlighting, no completion, no third-party editor
widget and no new dependency. Editing an external `.py` in this window is out of
scope — that file already has an editor.

**Errors are never fatal.** Every call into Python is wrapped; a traceback goes
to the console, the offending instance is disabled for the rest of the session,
and everything else keeps playing. A script that raises on every frame is
reported once, not sixty times a second. Syntax errors, a missing class, a file
that has moved away and an `update()` that throws are all just messages — the
last one is also shown in red in that object's Script section.

**The GIL, and threads.** The editor holds the GIL only inside a sweep and
acquires it once for the whole frame, not once per script. A script must
therefore only touch the scene from `start` / `update` / `stop`: work handed to a
`threading.Thread` runs without the editor's cooperation, and mutating an
`Object3D` from it races the renderer. Spawn threads for I/O if you must, and
apply what they produce from the next `update()`.

**Objects handed to a script are the concrete type** — a `Mesh` is a
`threepp.Mesh`, not a `threepp.Object3D`. That is not a nicety: `Mesh`, `Points`
and `Line` derive from `Object3D` *virtually*, and pybind11 mishandles every
access that crosses that base (assigning the inherited `name` through an
`Object3D`-typed handle corrupts the heap). They are also handed over as
`shared_ptr`, so a script that stashes `self.obj` past Stop holds a detached but
perfectly alive object rather than a dangling pointer.

The runtime lives in `apps/editor/scripting`, not in the library: `ScriptConfig`
is dependency-free like the other configs, but `ScriptPlaySession` needs CPython
and libthreepp stays free of it. The whole directory is compiled only when
`Python3` (component `Development.Embed`) and pybind11 are found, mirroring how
PhysX gates the physics session. Without it the editor still authors and saves
scripts, and the Script section says so.

The embedded interpreter is served the **same** binding translation units the
Python wheel is built from (`python/src/bind_*.cpp`), linked into the editor and
registered with `PYBIND11_EMBEDDED_MODULE`. Importing a wheel-built `threepp`
into the embedded interpreter instead would register a second, independent set
of C++ types, and the editor would be handing scripts objects that module has
never seen. Ten areas are linked — math, textures, core, geometries, materials,
objects, animation, cameras, lights, robot — and the renderer, loader, physics,
sensor and Vulkan areas are deliberately left out: a script has no business
creating a window, and the editor already owns the ones that exist.

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

Three sessions ship, and they start in this order: physics, then the animation
player, then scripts. Scripts run last on purpose — a script's transform edits
are the final word for the frame, after the simulation and the mixer have had
their say.

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
* **Scripting needs the Python it was built against.** The editor embeds
  CPython rather than shipping it: a binary built against 3.14 loads
  `python314.dll` and the standard library from that installation, and on a
  machine without it the executable will not start at all. Deploying the editor
  with scripting therefore means shipping the matching runtime beside it (or
  building the editor without Python, where it still authors and saves scripts
  and simply does not run them). The interpreter is also deliberately never
  finalized — `Py_Finalize` from pybind11 3.0.4 access-violates on CPython 3.14
  under Windows, measured with a bare `scoped_interpreter` and no bindings at
  all — so the process exits with Python still up. Nothing leaks that the OS
  does not reclaim.
* **One script per object, in one form.** `userData["script"]` is a single path
  and `userData["scriptSource"]` a single body of source, and an object carries
  one or the other. Several behaviours on one object means several child
  objects, or one script that composes them.
* **The Script Editor is a text box.** No syntax highlighting, no completion, no
  find-and-replace, and it edits inline source only — an external `.py` already
  has an editor. The one thing it adds over one is the compile check, and that
  is the deliberate scope of v1.
* **Duplicate shares geometry.** `Object3D::clone()` shares both geometry and
  materials; the editor clones the materials afterwards so recolouring a copy
  does not recolour the original, but geometry stays shared. That is intentional
  (duplication stays cheap), and it means a future mesh-editing feature would
  need an explicit "make unique" step.
