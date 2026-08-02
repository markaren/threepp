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
| `--play` | press Play as soon as the scene is open — with `--frames`, a whole play session without a hand on the mouse |
| `--example=<slug>` | open a [shipped example](#shipped-examples) instead of the template scene (`hover-arena`) |
| `--screenshot=<png>` | write PNGs and exit. With no document of its own it builds the spline-tube acceptance scenario and writes one file per view; **with a `scene.json` or `--example` it photographs that instead**, honouring the three flags below |
| `--seconds=N` | how long that pass plays before the first shot (default 3) |
| `--keys=W,A` | hold these keys for those seconds, then let go — a scene you are meant to *drive* cannot be reviewed standing still |
| `--hold-keys` | keep holding them **through** the shots instead of letting go and waiting for the scene to settle. What a manoeuvre looks like halfway through it — a drone mid-turn, with the chase camera coming round behind it — is a picture the settled pose cannot show |
| `--shot=px,py,pz@tx,ty,tz[:tag]` | camera position and target for that pass, repeatable; `tag` becomes a file-name suffix. None given keeps the camera the session already has — an authored [`editorView`](#opening-a-document-editorview--editorfollow), or wherever Follow Selection has chased to — and frames the whole document only when there is neither |
| `--bench=N` | time N frames after `--seconds` of warmup and print the result. **Turns vsync off**, because a present-capped frame time measures the display rather than the renderer; the status bar's fps cannot answer this question for the same reason (it is a smoothed average of frames the swapchain paced). Reports median, p95 and max CPU frame time in milliseconds, and on Vulkan the per-pass GPU medians from the backend's timestamp queries. Composes with `--example`/`--play`/`--keys`, so a demo can be timed while it is being flown. `THREEPP_BENCH_VSYNC=1` puts the cap back, for the one question the uncapped number cannot answer: whether the editor *as shipped* holds the refresh rate. `THREEPP_BENCH_DISABLE=a,b,…` strips pieces of the frame so a cost can be *attributed* rather than guessed at — `cloud`, `sensors`, `ui`, `overlay`, `play`, and the deferred pipeline's own `ao`, `probegi`, `restir`, `denoise`, `halfres` (render at half scale: a cost that halves is per-pixel work, one that does not is fixed) |

---

## Architecture

Two layers, deliberately separated:

**`threepp/extras/editor/*` — the reusable core.** Part of the threepp library,
with no ImGui, no window and no GL/Vulkan dependency. It is what a different
front end (a Qt tool, a Python binding, a batch script) would build on — and
what [`threepp_player`](player.md), the second front end that ships, is built
on:

| type | responsibility |
| --- | --- |
| `Command`, `CommandStack` | undo/redo with drag coalescing |
| `SetTransformCommand`, `PropertyCommand<T>`, `AddObjectCommand`, `RemoveObjectCommand`, `ReparentCommand`, `SetMaterialMapCommand` | the concrete edits |
| `Selection` | the current object + change notifications |
| `SceneDocument` | the open Scene, its path, its dirty flag, save/open |
| `SceneSnapshot` | scene ⇄ JSON, with live textures preserved |
| `ObjectFactory` | primitives, lights, groups, cameras with unique names |
| `PlaySession`, `PlayController` | the play-mode state machine |
| `PhysicsConfig` | per-object rigid-body and soft-body authoring, stored in `userData` |
| `RenderConfig` | the renderer settings a document is saved with, as a difference from the editor's defaults, stored in the scene root's `userData` |
| `PhysicsPlaySession` | the PhysX runtime (header-only, PhysX-gated) |
| `ScriptConfig` | per-object Python script — a file path or inline source — plus its parameters, stored in `userData` |
| `ScriptWorkspace` | the `.vscode/settings.json` that teaches Pylance about `threepp`, and the text normalization an external edit round-trips through |
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

**Follow Selection.** View ▸ Follow Selection (`Shift+F`) turns the viewport
into a chase camera: every frame the orbit target walks towards the selection's
world position and the camera is carried with it, keeping the offset you orbited
to, so orbiting, panning and zooming keep working while it follows. The approach
is exponential with a ~90 ms time constant rather than a hard lock — a rigid body
has a tremble, and a camera bolted to it inherits it.

The offset is kept in the subject's **heading frame**, which is what makes it a
chase rather than a fixed vantage that the subject flies out of: yaw the drone
with `A`/`D` and the camera comes round behind the new heading instead of
watching it leave sideways. What "heading" means is the object's forward axis
(`-Z`, threepp's own convention — the direction `lookAt` aims) projected onto the
ground plane, read off its world quaternion, so it works for anything that faces
where it goes. Specifically:

* **Yaw only.** Never pitch, never roll. A hover drone banks constantly to move,
  and a camera that inherited attitude would roll the horizon with every input.
  Camera up stays world up and is never touched.
* **Its own time constant**, 150 ms — slower than the position follow, because a
  heading error swings the whole frame. The number was measured on the shipped
  drone rather than guessed: hovering, its heading *wanders* slowly enough that
  no exponential filter takes that out, and at full stick it yaws at 200 °/s,
  where the steady lag is exactly `tau` × that. This one keeps the subject astern
  through a turn while still letting you see that it is turning.
* **Orbiting composes in the subject's frame.** Drag the camera over its left
  shoulder and it stays over its left shoulder through every turn. Zoom is still
  just a shorter offset.
* **A nose pointed at the sky keeps the last heading** — the ground-plane
  projection is noise there, and something tumbling must not whip the camera.
* **An orthographic view follows by translation alone.** An axis view *is* a
  direction of view; rotating it would mean it is no longer the view you asked
  for. Following a subject that never turns is therefore exactly what it always
  was, in either projection.

It works while **playing**, which is the point: a drone you fly with `W`/`A`/`S`/`D`
leaves the frame in about two seconds otherwise. `Shift+F` is answered even while
[a script owns the plain keys](#the-script-editor) — deciding to watch what you
are flying is a view command, not teleop. Deselecting *pauses* the chase (there
is nothing to chase, and the view stays where the chase left it); selecting
something again resumes it. It is session state, not a saved preference, and a
document can ask for it on open — see [`editorView`](#opening-a-document-editorview--editorfollow).

**Markers.** Cameras and lights render nothing, so each one gets a billboarded
icon at a constant screen size, tinted with the accent colour while selected.
Every kind has its own shape — a camera body, a sun for directional, a bulb for
point, a beam for spot, a ringed core for ambient, a dome over ground for
hemisphere, a speaker for a sound, an open hinge for a joint — so a scene reads
without clicking anything.
Clicking an icon selects its owner, and it wins over geometry behind it — the
icons draw on top, so picking follows what you see rather than raw depth order.
The artwork is SVG parsed at startup by threepp's `SVGLoader` and embedded as
source in `ViewportMarkers.cpp`, so the editor does not depend on finding asset
files at runtime. Selecting a camera additionally shows its frustum
(`CameraHelper`), which tracks fov/near/far edits live, and renders its view into
the camera dock beside the bottom panel. Objects that bound to
nothing get no selection box — an empty `Box3` would leave `BoxHelper` showing a
degenerate speck at the origin.

**Splines.** Add ▸ Spline drops a curve whose control points are ordinary scene
nodes — drag them with the gizmo, insert and delete them, and the sampled curve
redraws live. See [Splines in `userData`](#splines-in-userdata).

**Sound.** Add ▸ Sound places an emitter — a speaker marker, an audio file, and
a distance falloff drawn as min/max rings on the ground while it is selected.
**Audition** hears the file without pressing Play; Play spatializes it through
miniaudio and the sound follows whatever moves its object. See
[Sounds in `userData`](#sounds-in-userdata).

**Generators.** A scene can carry inline Python that BUILDS it: select `Scene`,
write a rule, press Regenerate, and what the script creates becomes ordinary
saved scene content. Editable in VS Code with completion, where a save re-runs it.
This is how you author at counts a mouse cannot — a few hundred instanced props
placed by rule, reproducible from a seed. See
[Generators in `userData`](#generators-in-userdata).

**Hierarchy.** The full `Object3D` tree, selection synced both ways with the
viewport, double-click to rename, drag-and-drop to reparent (undoable, world
transform preserved), and a right-click menu with Add ▸ / Duplicate / Delete /
Focus.

**Inspector.** Object flags, transform (rotation shown in degrees), material
(type, colour, emissive, roughness, metalness, opacity, transparency, side,
wireframe, flat shading, six texture slots with previews), read-only geometry
counts, light parameters, camera parameters, spline curve parameters, attached
scripts and physics. Every edit is undoable, and a drag collapses into a single
undo step.

**Bottom panel.** A console showing loader and exporter warnings, an asset
browser (double-click a file to open/import/assign), the live sensor readout,
and — while any are open — the Script Editor's scripts. Console is the first tab: it is
where import results and errors land. Collapsible, resizable by dragging its top
edge, and it runs to the left edge of the window.

**Camera dock.** The band beside the bottom panel, under the inspector, renders
the selected camera live. It collapses with the bottom panel and paints itself
when nothing is selected, so that corner is never a sliver of viewport too small
to see into or click in.

**Panel sizes.** The hierarchy and inspector are resized by dragging their inner
edge, the bottom panel by dragging its top edge, and all three sizes persist in
the settings file. Every one is clamped on read and on drag — the bottom panel
against the window itself, so a settings file written on a bigger monitor cannot
leave a small one with no viewport at all. The hierarchy also scrolls
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

The **transform gizmo is parked for the whole session**: Play detaches it, and
the toolbar's Select/Move/Rotate/Scale and Local/World buttons grey out with it.
Hiding it would not have been enough — attached, it rides a body the solver is
moving and keeps offering handles for an edit that Play would only refuse (every
document mutation goes through one gate; the greyed menu items are the visible
half of it). Stop puts the gizmo back on the selection (re-resolved by uuid
across the scene swap) with the mode, the space and the snap it had.

**The rest of the authoring layer goes with it.** During Play the viewport shows
the *scene*, not the editing furniture, so these are hidden for the session and
restored on Stop:

| hidden while playing | why |
| --- | --- |
| the selection bounding box, and the outline around a picked instance | it says what you are *editing* |
| the viewport marker icons — cameras, lights, spline points, sensors | same, and an icon you cannot see is not clickable either: picking through the icons is switched off with them |
| the selected camera's frustum helper | same |
| the transform gizmo | above |

And these deliberately **stay**:

| still drawn while playing | why |
| --- | --- |
| the sensor point cloud | it is what the scene is *doing* — play data, not furniture |
| the physics collider overlay | a debug view that only means anything while a world exists to debug |
| the grid and the origin axes | View-menu preferences somebody set on purpose |
| the play banner and the status bar | UI, not scene |

Picking itself still works, and the hierarchy still shows what is selected —
watching what the simulation does to an object is half of why anyone presses
Play. Selecting something *during* play is fine too: the outline it builds
arrives hidden and appears on Stop. The `--screenshot` pass over a document hides
the same layer through the same predicate, for the same reason.

---

## Shipped examples

**File ▸ Open Example** lists the scenes that ship *inside the binary*. They go
through the same unsaved-changes guard as Open and the same loader, and they
arrive as an **untitled** document — no path, not dirty — so the first Save asks
where to put it and nothing can write over an example in place. The JSON is
embedded as source (`apps/editor/ExampleScenes.cpp`) for the reason
`ViewportMarkers.cpp` embeds its SVG: the editor does not depend on finding
asset files at runtime.

### Hover Arena

A physics drone you fly through five glowing goal rings, in a generated arena of
pillars. It exists because every feature below is easy to *describe* and hard to
believe until something is using all of them at once:

**It opens ready to fly.** The document carries an
[`editorView`](#opening-a-document-editorview--editorfollow): 7 m behind the
drone and 2.4 m above it, looking down the course with the first gate behind the
drone, the rest of the rings receding and the beacon standing at the far end. It
also carries `editorFollow = Drone`, so the drone is selected and
[Follow Selection](#features) is on — and because Follow keeps the offset the
camera has when the document opens, that one authored vantage *is* the chase
camera. Press Play and the view goes with the drone — and because the offset is
kept in the drone's heading frame, `A`/`D` turn the *camera* too: it comes round
behind the new heading instead of watching the drone leave sideways.

| what you see | what it is |
| --- | --- |
| the drone holds a 2.2 m hover, hands off, with a slight wobble | [`fixed_update`](#the-physics-clock-fixed_update) applying `apply_force`/`apply_torque` through [`rigid_body_from_object`](#physics-from-a-script), on the physics clock so the gains mean the same thing on any machine |
| it knows how high it is | a short downward [`threepp.editor.raycast`](#raycasts-threeppeditorraycast) with `ignore=self` — without the `ignore` the ray starts inside the hull and finds it at range zero |
| **the wobble itself** | the attitude damping closes on the authored IMU's reading through [`imu_from_object`](#sensors-from-a-script) — noisy, seeded and rate-gated. It is not a controller that has never met a sensor, and the seed makes the same wobble happen on the next run |
| the coloured cloud sweeping the floor | a VLP-16 [LIDAR](#joint-sensors) authored on the drone, drawn by the Sensor Point Cloud overlay (on by default) |
| a ring turns amber and the beacon warms toward gold | each ring's hole is an invisible **[trigger volume](#trigger-volumes)**; its `on_trigger_enter` flashes the torus and calls the scoreboard's live instance through [`script_from_object`](#talking-to-other-scripts-threeppeditorscript_from_object). Take all five and every ring pulses |
| the hull flashes when you clip a pillar | [`on_collision_enter`](#collisions-on_collision_enter--on_collision_exit), which nobody had to tick a box for |
| the floor, walls, pillars and crates | a seeded **[generator](#generators-in-userdata)** on the scene root. Its committed output ships in the document — opening a scene never runs anything — and the rule travels with it, so Regenerate rebuilds the arena from `SEED` without moving the hand-placed course |

Controls, which the drone's script also prints to the console at Play:

| key | |
| --- | --- |
| `W` / `S` | forward / back |
| `A` / `D` | yaw left / right |
| `Q` / `E` | strafe left / right |
| `R` / `F` | climb / descend — the stick sets a *height*, and lets go holding it |

Remember that [a script reading the keyboard takes the plain keys](#the-script-editor)
off the editor for the rest of the session; `Ctrl+S` and friends still work, and
so does `Shift+F` — switching the chase camera on and off is the one view command
you want *while* flying.

**Without the PhysX build** there is no body to push, no ray to cast and no IMU
to read, and those names are *absent* from `threepp.editor` rather than present
and answering `None`. The scripts ask with `getattr` before reaching, so the
scene loads, renders and plays; the drone sits where it was put, and the console
says `Hover Arena needs the PhysX build to fly` once. **Without Python** nothing
runs at all — the scene is still a scene, the scripts are still saved, and the
Script section says which build would run them.

The document is regenerable end to end: `apps/editor/tools/HoverArenaAuthor.cpp`
builds it through the editor core, runs the generator exactly as **Regenerate**
does, and writes both the `.json` and the translation unit that embeds it.
`tests/extras/EditorExampleScene_test.cpp` loads that same embedded string and
flies it headlessly, and `--selftest` opens it through the menu's own code path.

---

## Keyboard shortcuts

| key | action |
| --- | --- |
| `W` / `E` / `R` | translate / rotate / scale gizmo |
| `Q` | toggle local / world space |
| `Shift` (hold) | snap while dragging |
| `F` | frame selection |
| `Shift+F` | follow selection — chase camera, works while playing |
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

### Opening a document: `editorView` / `editorFollow`

Two optional keys on the **scene root** say how a document wants to be looked at.
They are read by the open paths — File ▸ Open, File ▸ Open Example, a `.json`
dropped on the window, a scene named on the command line — and by nothing else:

```
userData["editorView"]    0,4.6,21@0,2.2,14        (camera position @ orbit target)
userData["editorFollow"]  Drone                     (an object name)
```

`editorView` is the same `px,py,pz@tx,ty,tz` the `--shot` command line speaks,
parsed by the same function (`EditorApp::parseViewSpec`), so a vantage found by
orbiting can be copied into a scene and back out again. It places the camera and
the orbit target in a perspective view; a malformed value is a console line and
is otherwise ignored. `editorFollow` names an object to select and
[follow](#features) on open — a name that is not in the scene is a console line
too.

Two rules make them safe rather than surprising:

* **They apply when a document is OPENED, and never again.** In particular they
  are not applied by the scene-replaced listener, which is also what Stop goes
  through: restoring the play snapshot must never teleport the camera away from
  wherever the user drove it while watching the simulation.
* **A document that says nothing gets nothing.** No `editorView` leaves the
  camera where it was (an example, having nothing to preserve, frames itself
  instead); no `editorFollow` switches following *off*, since following belongs
  to a subject and a new document is a new subject.

### The look of a document: `render`

A third key on the **scene root** carries what the **View ▸ Renderer Settings**
panel was set to when the document was saved — exposure and tone map, render
scale and upscaler, GI and denoiser, fog, bloom, clouds, depth of field, the
camera triplet. It is written on every save and read by the same open paths as
`editorView`, plus File ▸ New:

```
userData["render"]   renderscale=0.5;bloom=0.4;heightfog=1;fogdensity=0.05
```

`RenderConfig::encode()` / `decode()` own that format — the same flat
`key=value;key=value` string as `physics` below, for the same reason (`userData`
round-trips scalars only), and with the same rule that an unknown key is ignored
on read.

What is different from every other config in the editor is that the string is a
**difference from a baseline**, not a full state. The baseline is
`RenderConfig::capture()` of the renderer as the editor set it up, taken once in
`EditorApp`'s constructor and kept in `renderDefaults_`:

* On **save**, only the fields that differ from that baseline are written. A
  document nobody re-lit therefore carries no `render` key at all, and the four
  keys above are the whole of what one that was re-lit says.
* On **open**, the text is layered over the same baseline key by key. A document
  that says nothing about fog gets *the editor's* fog — not the ground mist the
  previously open document left on the renderer. That inheritance is the bug the
  baseline exists to prevent, and it is why File ▸ New applies it too.

Two things are deliberately **not** in here. The diagnostic controls (G-buffer
debug view, overlay layer) are about inspecting a frame, not authoring one. The
sensor-simulation controls (lens distortion, sensor noise) describe an
*instrument* pointed at the scene, and belong to the per-object sensor
authoring `SensorConfig` already carries.

On Vulkan the editor's baseline sets **render scale 0.8** — the deferred shade
is the frame's dominant cost and it scales with pixels, so an authoring viewport
takes 64% of them for a difference TAA reconstructs most of the way back. The
renderer's own default is still 1.0 for every example and test, and the Render
scale slider (or a saved `renderscale=`) overrides it per document.

### Physics in `userData`

Per-object physics rides in `object.userData["physics"]`, so it serializes with
the scene for free. `userData` round-trips **scalars only** (bool / int / float /
double / string — see `ObjectExporter::writeUserData`), so a nested JSON object
is not available. The value is therefore one flat, deterministic string:

```
body=dynamic;shape=convex;trigger=0;mass=12.5;friction=0.8;restitution=0.35;young=1000000;poisson=0.45;voxel=10;iterations=20;selfcollision=0;hulls=16;hullverts=64;voxels=100000
```

`PhysicsConfig::encode()` / `decode()` own that format. Unknown keys are ignored
on read, so a document written by a newer editor still loads — and a document
written by an *older* one loads too, since a missing key keeps its default. A
disabled body removes the entry entirely rather than writing `enabled=false`, so
turning physics off leaves no trace in the file. Every key is written whatever
the body type, so switching a body from `soft` to `dynamic` and back does not
quietly discard the settings the other type was using — `trigger=1` on a body
you flip to `soft` and back is still `trigger=1`, even though the checkbox is
hidden in between (see [Trigger volumes](#trigger-volumes)).

### Colliders

The **Shape** picker decides what a body actually collides with. The primitives
(**Box**, **Sphere**, **Capsule**) and **TriMesh** (a static/kinematic-only
triangle mesh) are exact and cheap; the interesting cases are the convex ones,
because PhysX dynamics can only use convex shapes and most real geometry is not
convex.

* **Auto** reads the geometry: a stock Box/Sphere/Capsule primitive gets its
  analytic shape, a static shaped mesh gets a triangle mesh, and a moving shaped
  mesh gets a single convex hull.
* **Convex** is one convex hull — the smallest envelope around the vertices. It
  roofs over any concavity, which is fine for a rock and wrong for a mug.
* **Convex Pieces** runs **V-HACD** convex decomposition: the mesh is split into
  several convex hulls that together approximate the concave shape, welded into
  one rigid body. This is what lets a mug hold water, a pipe stay hollow, a chair
  drop something between its legs. Three parameters ride in `userData`, always
  written:

  | Field | Meaning |
  | --- | --- |
  | **Max Hulls** (`hulls`) | The most pieces V-HACD produces (default 16). More hulls hug a concave shape more closely, at more narrowphase cost. |
  | **Verts / Hull** (`hullverts`) | Vertices per hull (default 64, PhysX's GPU-compatible ceiling). |
  | **Voxel Res** (`voxels`) | V-HACD's voxel grid resolution (default 100000). Higher resolves thin features but costs cook time. |

**Imported models collide as themselves now.** An imported model is a `Group`
with sub-meshes, and a `Group` has no geometry of its own — so a dynamic one
used to fall back to a **1 m unit box** at the origin (a chair colliding as a
cube). Authoring physics on the Group now gathers its sub-meshes into **one
compound convex actor**: *Auto* (moving) or *Convex* gives one hull per
sub-mesh, *Convex Pieces* decomposes each sub-mesh and divides the hull budget
across them (with a log line when it is split, so nothing is truncated
silently). Each sub-mesh's transform and scale bake into its shape's local pose,
so the compound matches where the parts actually are. A **static** Group is
unchanged: it still collides as its triangle-mesh subtree, which is exact.

**Cooking is cached.** V-HACD on a dense mesh is *seconds*, so a decomposition
is keyed on the geometry uuid **and** its parameters and reused for the rest of
the Play session — duplicates of one model decompose once, exactly like the
soft-body tet cache. The cook time is logged per geometry (so a Play that took a
moment says why), and a mesh over ~200k triangles logs a warning before it
starts rather than freezing silently. A build without V-HACD (no PhysX, or the
`v-hacd` vcpkg dependency absent) falls back to a single hull with one log line,
so *Convex Pieces* still simulates — just without the concavity.

The cooked hulls are drawn by the **Physics Debug** overlay (PhysX's own
collision-shape visualization), so the quality of a decomposition is visible in
the viewport, not just inferred from behaviour.

### Trigger volumes

Tick **Trigger** and the body stops colliding and starts *reporting*. Its shapes
are cooked with PhysX's `eTRIGGER_SHAPE` instead of `eSIMULATION_SHAPE` — the two
are mutually exclusive on one shape — so bodies pass straight through it and what
it produces instead is an overlap event: who came in, and who left. That is the
goal zone, the checkpoint, the kill plane, the "the player reached the door"
region. A script on either side hears about it through
[`on_trigger_enter` / `on_trigger_exit`](#trigger-volumes-on_trigger_enter--on_trigger_exit);
with no script anywhere it is simply a body that collides with nothing.

* **`trigger=0/1` in `userData`, written every time** like every other key, so
  ticking it, switching the body to *Soft* and switching back does not lose the
  tick.
* **Static, Dynamic and Kinematic.** A static volume is the common case and the
  cheap one; PhysX allows a trigger shape on any rigid actor, so a moving trigger
  (a zone carried by a lift, a proximity bubble on a vehicle) works too. A
  *dynamic* trigger still falls — it has mass and gravity, it just has nothing to
  land on.
* **Hidden for Soft.** A deformable volume's collider is the tetrahedral mesh
  PhysX cooks from the geometry, which is not a shape anything can raise the
  trigger flag on, so the checkbox disappears the same way the Shape picker does.
* **A trigger is not a collider, at all.** No contacts, so
  [`on_collision_enter`](#collisions-on_collision_enter--on_collision_exit) never
  fires for an overlap with one and a
  [`ContactSensor`](#sensors-from-a-script) never measures one. If you want an
  object to both block and report, use two objects: a collider and a trigger over
  it.
* **Scene queries still see it.** A trigger shape keeps `eSCENE_QUERY_SHAPE`, so
  [`threepp.editor.raycast`](#raycasts-threeppeditorraycast) hits it like
  anything else — which is what lets a script find a zone before entering it, and
  what a ground check has to know when casting through one.
* **The overlay draws it.** The **Physics Debug** overlay shows a trigger
  volume exactly as it shows a collider (measured: PhysX's collision-shape
  visualization emits the same lines either way), so a volume you cannot see
  bodies bouncing off is still visible in the viewport.

**Triangle meshes cannot be triggers**, and this is the one place the editor
substitutes something. PhysX refuses the flag outright — *"triangle mesh and
heightfield triggers are not supported"* — because a trigger asks whether a point
is *inside* it and a triangle soup is a surface with no inside. Rather than cook
a trigger that silently does not exist, the play session falls back to a **convex
hull** and says so in the console, once, naming the object. That covers:

| authored as | cooked as | why |
| --- | --- | --- |
| **TriMesh** on a mesh | one convex hull of that mesh | the flag would be refused |
| **Convex** on a *static* mesh | one convex hull of that mesh | a static Convex is normally cooked as an exact triangle mesh, which is free for something that never moves — and not available here |
| **Auto** on a static shaped mesh | one convex hull | *Auto* resolves to TriMesh for a static shaped mesh |
| **Auto**/**TriMesh** on a static `Group` | one hull per sub-mesh, welded into one compound | the subtree is cooked as triangle meshes |
| **Box** / **Sphere** / **Capsule** / **Convex Pieces** | unchanged | already volumes |

So a concave trigger volume wants **Convex Pieces**, whose hulls are volumes
already and need no substitution.

### Soft bodies

`body=soft` makes the object a **deformable volume**: PhysX cooks a tetrahedral
mesh out of the object's own geometry and rewrites its vertex positions every
step, so the mesh itself bends, squashes and jiggles rather than moving as a
rigid whole. The **Shape** picker disappears for a soft body — its collider is
always that cooked volume — and these parameters take its place:

| Field | Meaning |
| --- | --- |
| **Stiffness (Pa)** | Young's modulus. ~1e4 is jelly, 1e6 rubber, 1e8 near-rigid. Logarithmic drag, because the useful range spans four decades. |
| **Poisson Ratio** | 0 squashes freely, →0.5 preserves volume. |
| **Resolution** | Voxels along the longest axis of the simulation mesh. Higher = finer deformation and more solver work. |
| **Iterations** | Solver iterations per step. |
| **Self Collision** | Whether folds of the body collide with each other. |
| **Mass**, **Friction** | As for a rigid body. Restitution has no soft-body equivalent and is hidden. |

Two things follow from PhysX's implementation, and the editor deals with both:

* **Soft bodies run on CUDA only.** `PhysicsPlaySession` therefore switches its
  `PhysxWorld` to GPU dynamics as soon as the scene contains one — and only
  then, since a CUDA context is not free. On a machine without a CUDA device the
  world is rebuilt with the GPU off and the rigid bodies play on their own,
  with one line in the log; Play is not failed over it.
* **The mesh is the simulation output.** `addSoftBody` bakes the world matrix
  into the geometry and zeroes the object's local transform, then writes
  world-space vertices into it each step. Nothing special is needed to undo
  that: the play snapshot restores geometry along with everything else, so Stop
  hands back the rest pose and the authored transform.

Cooking is the expensive part of `start()`, so identical geometries share one
cooked mesh — keyed on the geometry uuid, and only for an unscaled object, since
the cached path re-applies rotation and translation but not scale.

A soft body needs a **closed, indexed triangle surface** to cook from. A mesh
with no index buffer is skipped with a log line rather than simulated into
nonsense; the stock Sphere, Box, Cylinder, Cone and Torus primitives are all
fine, rotated or not.

That last part is not free. The visual mesh is skinned from the cooked tet
mesh, and the tet mesh is built from a *remeshed* surface that does not quite
reach a sharp corner — the apex of a cone, or any corner of a rotated box. Two
things in `SoftBody` make that come out right, and both were originally wrong:
a vertex outside every tet binds to the nearest one by **extrapolation** (so it
sits exactly where it was authored, rather than snapped onto the hull and
visibly rounded off), and the index-for-index fast path is taken only when the
tet and visual positions genuinely **correspond** — equal vertex counts alone
are a coincidence waiting to scatter the mesh. `tests/extras/SoftBody_test.cpp`
pins the rest pose to under a millimetre for both.

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

### Sounds in `userData`

**Add ▸ Sound** creates a plain `Object3D` carrying `userData["sound"]` — a
sound has no geometry, so what shows it in the viewport is a speaker marker,
billboarded like the camera and light icons. Any *other* object can become one
too: load a file from the **Sound** inspector section and it gains the same two
entries.

Two entries, because the file needs its own key — a Windows path contains the
`=` and `;` the flat format uses as delimiters, the same wall `RobotConfig` and
`ScriptConfig` hit:

```
userData["sound"]      positional=1;autoplay=1;loop=1;volume=1;rate=1;minDistance=1;maxDistance=10000;rolloff=1;model=inverse
userData["soundFile"]  C:/audio/rain.wav
```

`SoundConfig` owns the format. The path is stored exactly as the file dialog
handed it over (absolute, forward slashes), like the script and URDF references;
a *relative* path in a hand-edited document resolves against the document's own
directory, so a scene and its audio can be moved together.

The **Sound** section carries the file row, *Positional* (off makes it ambience
at a constant level — music, room tone), *Play on Start*, *Loop*, *Volume*,
*Playback rate*, and, for a positional sound, the distance falloff: min/max
distance, rolloff and the distance-model curve (`none` / `inverse` / `linear` /
`exponential`, mirroring the Web Audio panner models). **Max distance is a hard
audibility bound under every model**: the play session eases the sound out over
a short band before max and silences it past it — miniaudio alone merely clamps
the falloff there, which left an `inverse` sound faintly audible across the
whole map. Under `none` that makes the sound a constant level within range and
silent outside (a zone ambience); min distance and rolloff shape nothing there
and are greyed. The falloff entries are hidden rather than dropped when
*Positional* is off, so switching it back on does not reset a tuned curve. **Remove sound** drops both entries as one undo
step.

Selecting a positional sound draws its **min and max distance rings** on the
ground — two line circles under the editor overlay, so they are never saved and
never picked. `maxDistance` defaults to miniaudio's 10000, which means "no
limit" rather than "ten kilometres", and a ring that big draws as a straight
line across the viewport and says nothing — so past 1000 m the ring is simply
left out. The number is still in the inspector.

**Audition** (edit mode only) plays the file without pressing Play, on the
editor's own listener. It is deliberately **flat**: the authored volume and
rate, no spatialization. It answers "is this the right file, at the right
level"; how it sounds from over there is what Play is for. The audition stops
on the button, on a selection change, on entering Play and on a scene replace.

During Play, `AudioPlaySession` walks the scene and builds one `PositionalAudio`
per authored node (parented to it with `addRef`, so the spatialization follows
whatever moves the object) or a plain non-spatial `Audio` when `positional=0`.
The listener rides the perspective viewport camera. Sounds with `autoplay` on
are rewound and started; on Stop every sound is unparented and destroyed before
the engine is, and the play snapshot takes the rest back.

**No device, no problem.** `AudioListener`'s constructor throws when
`ma_engine_init` fails — a headless CI machine, a driver holding the device in
exclusive mode — and each file load throws on its own. Both are caught: the
failure is logged to the console and the session becomes a no-op (or skips the
one bad file), so a scene whose audio moved still plays everything else. Same
discipline as the soft-body CUDA fallback.

The whole feature is gated on `-DTHREEPP_WITH_AUDIO=ON` (the default). A build
without it still *authors* sounds — the config is plain `userData` — and shows
"Built without audio - authoring only" where the audition button would be, so a
document does not lose its sounds by being opened on the wrong build.

miniaudio decodes `.wav`, `.mp3` and `.flac` with no third-party code, and those
are exactly the three the file dialog offers.

### Joints in `userData`

**Add ▸ Joint** creates a plain `Object3D` carrying `userData["joint"]`. A
joint has no geometry — what shows it in the viewport is a hinge marker, and
what *edits* it is the ordinary transform gizmo, because **the node's transform
is the joint frame**: anchor at its origin, hinge/slide axis along its local X
(the same convention the articulation builder uses). Its **parent chain is body
A** — the nearest ancestor with a rigid body governs it, the same walk a sensor
resolves its attachment with — and the **other body is referenced by name**;
empty means the world (a pendulum pivot, a door frame bolted to nothing).
Because each joint is its own child node, a chassis carries four wheel joints
as four children, and deleting a joint is deleting a node.

Two entries, because a scene-object name is user-typed and free to contain the
`=` and `;` the flat format uses as delimiters — the same wall the sound file
hit:

```
userData["joint"]      type=revolute;limited=1;lower=-0.5236;upper=0.5236;coney=0.785398;conez=0.785398;stiffness=0;damping=0;maxforce=1000000;target=0;velocity=0;breakforce=0;breaktorque=0;collide=0
userData["jointBody"]  Post
```

`JointConfig` owns the format. Angles are radians in the document and degrees
in the inspector, the same split the robot joint sliders make. Presence of the
entry is what makes the node a joint (writing it never omits defaults), and
every field rides along whatever the type is, so switching type and back does
not reset the ones the other type hides.

Five types: **Fixed** welds body A to body B; **Revolute** hinges about X, with
optional lower/upper limits and a PD drive (a door, a wheel, a pendulum);
**Prismatic** slides along X, limits and drive in metres (a piston, a drawer);
**Spherical** is a ball socket with an optional swing cone (cone Y/Z
half-angles; twist stays free); **Distance** is a tether that keeps the anchors
within min/max metres, its stiffness/damping pair acting as the tether's spring
instead of a motor. Drives are **force mode**, so stiffness/damping/max-force
mean the same as they do in `userData["articulation"]` — and each half gates
its own input: the drive force is `stiffness·(target−x) + damping·(velocity−v)`,
so **Target acts through stiffness and Velocity through damping**. The
inspector makes that structural: a **Driven** checkbox (a **Spring** one for
Distance) that seeds both to the articulation defaults (500/50) when ticked —
"driven" is not a stored flag, it *is* stiffness/damping > 0, exactly the
condition the play session builds a drive under. The seeding also matters
mechanically: the sliders are logarithmic, and a log drag anchored at zero
compresses so hard near the bottom it can never escape it. Every type can be
made **breakable** (a checkbox seeding break force/torque the same way; off =
unbreakable) and can opt back into **collision** between the two bodies it
joins — off by default, since bodies meeting at a joint overlap at the anchor
and contacts there fight the constraint.

The **Joint** inspector section leads with what actually connects ("Connects
Gate to Post"), resolved the way the play session will resolve it. Body B is a
combo over the bodies that *exist to be picked* — every enabled rigid body and
every simulated robot, ancestors excluded (they are body A's side) — because
the reference is a scene object: choosing one that exists is the feature,
typing one that resolves is the error path. Selecting a joint draws its **axis
helper** under the editor overlay: the anchor cross, the X axis with an
arrowhead (both ways for prismatic), the rotation ring for a hinge, two rings
for a ball — constant screen size, drawn through geometry, because the anchor
usually sits inside the body it hinges.

During Play, `PhysicsPlaySession` builds the authored joints **last**, after
every rigid body and articulation exists: body A from the parent walk, body B
by name — and either side may be an **articulation link** (a
`PxArticulationLink` is a rigid actor), which is how a prop attaches to a
robot's gripper. One `PxD6Joint` configured per type backs everything except
Distance, which is PhysX's own `PxDistanceJoint`. An unresolvable joint — a
missing name, a body with no rigid actor — is one console line and no joint,
never a refused Play. Stop releases the joints before the world, and the play
snapshot takes the scene back.

While playing, the **Physics Debug** overlay (View menu) draws the joints the
solver is actually enforcing, from PhysX's own render buffer: an RGB frame
triad at each constraint's live anchor and its limit geometry beside it, in
PhysX's own colours. That is the running counterpart of the authored axis
helper, which Play hides with the rest of the authoring layer — and a joint
that **broke** vanishes from the overlay, which is exactly the news.

The runtime type is `threepp::Joint` (`extras/physx/Joint.hpp`), usable
without the editor: two `PxRigidActor*` (either may be null, meaning the
world), one world-space frame, one `Params` struct — the same escape the
examples used to hand-roll per demo.

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

**Descendant userData survives the transplant.** The swap keeps the placeholder
root's identity and userData, but a sensor (or a physics entry) is often
authored on a *link*, not the root — and the fresh robot's link nodes are new
objects. So before the swap each placeholder descendant's userData is collected
by node name and re-applied to the same-named node of the rebuilt robot (first
match wins; a name the URDF no longer has is reported, not lost). That is what
makes "an encoder on the wrist link" a durable authoring choice rather than
something a document load quietly drops. The transplant is a free function,
`transplantRobot`, so this rule is the same headless and in the app.

#### Simulating a robot: `userData["articulation"]`

By default a saved robot is a kinematic prop — Play renders it and its joints
move only if a script drives them. Tick **Simulate** in the Robot section and it
becomes a PhysX **reduced-coordinate articulation** while playing: the joints
are real DOFs, gravity and contact act on the links, a PD drive holds the
authored pose, and — the point of the whole thing — the joint sensors below have
something to measure. Stop restores the pre-play snapshot, so nothing the
simulation did survives, exactly like the rigid bodies `PhysicsConfig` authors.

This is a separate entry from the URDF reference above because its fields are all
scalars, so unlike the path/vector pair it rides the same flat `key=value;`
format `PhysicsConfig` uses — one string in `userData["articulation"]`, every key
written every time, unknown keys ignored on read, and the entry removed entirely
when Simulate is off. Its *presence* is the "simulate this robot" signal.

```
fixedbase=1;stiffness=500;damping=50;maxforce=1000000;selfcollision=0;iterations=12;density=1000
```

| key | meaning |
| --- | --- |
| `fixedbase` | pin the base to the world (an arm) or let it float free (a quadruped, a drone) |
| `stiffness`, `damping` | the per-joint PD drive that holds the authored pose; 0 stiffness is a passive/force-controlled robot |
| `maxforce` | the drive's per-joint effort ceiling |
| `selfcollision` | whether links of the same robot collide (off by default — the primitive colliders overlap at the joints) |
| `iterations` | solver position iterations |
| `density` | fallback density for links whose URDF gives no `<inertial><mass>` |

The colliders PhysX simulates are an **approximation**: box/sphere/capsule for a
`<collision>` primitive, and a link's `<mesh>` collision by **one convex hull**
of that mesh — not the visual meshes, which still render exactly as imported. The
hull is a large improvement on the *bounding box* this used to produce (a slab
that swallowed every concavity between a robot's links); it is still a physical
twin, not a digital one, which is the right trade for grasping, balancing and
proprioception. A collision mesh with fewer than four vertices falls back to its
bounding box so the link still becomes a body.

`PhysicsPlaySession` builds the articulation from the same URDF the visual robot
came from, at the robot's world pose. The two share `URDFLoader`'s frame handling
(URDF is Z-up and neither side rotates it), so no Z-up→Y-up correction is applied
— a frame-consistency test pins the visual link and the simulated link to the
same place. Each step, the solved joint positions are mirrored back onto the
visual `Robot` (through a name map — the articulation's DOF add-order is not the
robot's joint order, so the *joint name* is the bridge); a floating base also
writes the solved root pose, so a walker actually travels. A link that carries
its own `PhysicsConfig` is **not** given a second rigid body — the articulation
link is already the body there — and a robot with a non-unit world scale is
skipped with one log line, because PhysX links cannot be scaled.

#### Joint sensors

An **Encoder** or a **Force/Torque** sensor reads one joint of a simulated
robot. Author it like any other sensor (Sensor section, on the robot or one of
its links), and pick the joint from the combo the section grows for these two
types — it lists the robot ancestor's articulated joints by name and writes the
choice as `joint=<name>` in `userData["sensor"]`. The section says why when
there is no robot ancestor, or when the robot has Simulate off.

The encoder's combo has one more entry: **All joints** (`joint=*`), which fans
one authored sensor out to one live encoder per articulated DOF at Play — the
joint-state sensor a whole robot reports itself with. It exists because an
object carries exactly one sensor entry, so without it instrumenting a 7-DOF
arm means seven encoders on seven link nodes. Each fanned-out encoder is its
own row in the Sensors panel and its own CSV, labelled with its joint name,
and each draws from its own seeded noise stream so one authored seed does not
correlate seven channels. The Force/Torque sensor deliberately does not take
it — a load cell is one piece of hardware bolted into one joint.

At Play the sensor session resolves that name against the played articulation and
builds a live `JointEncoder` (position/velocity, quantized by the encoder's
resolution) or `ForceTorqueSensor` (the six-axis wrench through the joint, in its
child frame). Both are registered with the physics world and sampled on its
fixed-substep loop, drained into the Sensor panel's traces and any CSV recording,
exactly like the IMU. Without the joint, or off a robot, the sensor is authored
and saved but reports why it is not measuring — the same "counted, not live"
contract every proprioceptive sensor has.

An **IMU** or a **contact sensor** authored on a robot link works too. The
visual link nodes are never bound to the world (the joint mirror drives them;
a world-pose write-back would fight the kinematic chain), so the play session
*associates* each link node with its articulation link — resolution only, no
pose writes — and the sensor's usual walk-up-the-ancestry lookup finds the
body. The IMU still measures in its authored node's frame, so an offset mount
under a link measures at the offset; a URDF's fixed `imu_link` resolves to the
link it was welded into, which is where the mount physically rides; and the
robot node itself resolves to the root link, so a base IMU needs no link names
at all.

One teardown detail worth knowing, because it is a class of bug this codebase
cares about: sessions stop in **reverse** registration order, physics last, so on
the normal Stop the sensor session unregisters from a world that still exists —
a Force/Torque sensor releases its `PxArticulationCache` cleanly through
`onUnregister`. The other path still exists for a teardown that skips the
controller (the editor closed mid-Play, destructors running in member order):
there the SDK is already gone, the cache's memory went with it, and the session
**abandons** the pointer rather than releasing it against a freed allocator; the
SDK teardown already reclaimed the buffer.

Everything a sensor measures is readable from a play script too — see
[Sensors from a script](#sensors-from-a-script), which is where a controller
closes its loop on the noisy reading instead of on the solver's truth.

### Scripts in `userData`

Attach a script to any object and it becomes a behaviour, MonoBehaviour style.
On Play the editor instantiates the class and drives it; on Stop it calls
`stop()` and throws the instance away, and the usual snapshot restore puts the
scene back.

The code comes in one of two forms, never both: a `.py` **file** referenced by
path, or **inline source** stored in the scene itself and edited in the editor's
Script Editor tab. A file is shared between objects and scenes and editable
in whatever tooling you already have; inline source travels with the scene, so a
`.json` handed to somebody else is complete.

**Opening a scene never runs anything.** Neither form executes on load, on
import, or on selection: scripts run when you press Play, and the inspector
imports a script's module only when it draws that object's parameters (a class
body has to execute for its attributes to exist). The Script Editor's Apply
compiles the source without executing it. A scene file is data, not a program.

```python
import threepp           # for the IDE; resolves against the embedded module at Play


class Spinner:
    speed = 1.5          # exposed in the inspector, saved with the scene

    def start(self, obj: threepp.Object3D):
        self.obj = obj

    def update(self, dt: float):
        self.obj.rotation.y += self.speed * dt

    def stop(self):
        pass
```

Every method is optional; missing ones are skipped. There are five more:
`fixed_update(self, dt)`, which runs on the *physics* clock rather than on the
frame (see [The physics clock](#the-physics-clock-fixed_update)),
`on_collision_enter(self, contact)` / `on_collision_exit(self, contact)`, which
run when the body governing the object starts and stops touching another (see
[Collisions](#collisions-on_collision_enter--on_collision_exit)), and
`on_trigger_enter(self, other)` / `on_trigger_exit(self, other)`, which run when
a body enters and leaves a [trigger volume](#trigger-volumes) — on the volume's
script and on the entering body's alike (see
[Trigger volumes](#trigger-volumes-on_trigger_enter--on_trigger_exit)). In a file, the
class is the one whose name matches it (`spinner.py` → `Spinner`,
case-insensitively), or — failing that — the single class in the file that
defines `update()`. Anything else is reported rather than guessed at.

**The scene** is `threepp.editor.scene()` — the ordinary `threepp.Scene`, so
`scene.get_object_by_name("Ground")`, `scene.children`, and so on. It answers
from `start()` onwards and for as long as the session runs, which means
`update()`, `fixed_update()` and the collision and trigger callbacks can all just
ask; there is no need to stash it on `self`. Outside a session it raises, because
outside a session there is no scene to answer with — a script's `__init__` runs
before its authored fields are even applied and is not a place to look at the
world from. The same lifetime rule as every handle applies to what you resolve
*through* it: use it during the session, never stash it across Play sessions.
That is also where a script reaches another script's *running instance*, rather
than its node: see [Talking to other
scripts](#talking-to-other-scripts-threeppeditorscript_from_object).

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

#### The physics clock: `fixed_update`

`update(dt)` is called once per **frame**, with the real delta between two
drawn images. That is right for anything visual and wrong for anything
physical: a force applied per frame is a force scaled by your frame rate, so
the same script settles at a different height on a different machine — and
gives a different answer on the same machine when a texture load stutters.

A script may therefore also define `fixed_update(self, dt)`. It runs from
*inside* the physics step, once per fixed substep, and its `dt` is the world's
timestep — the same number on every call, every run, every machine:

```python
import threepp


class Thrust:
    height = 3.0
    stiffness = 40.0

    def start(self, obj: threepp.Object3D):
        self.body = threepp.editor.rigid_body_from_object(obj)

    def fixed_update(self, dt: float):
        if self.body is None:
            return
        # Once per substep, at 1/60 s whatever the window is doing.
        error = self.height - self.body.position.y
        lift = self.stiffness * error - 5.0 * self.body.velocity.y
        self.body.apply_force(threepp.Vector3(0, lift * self.body.mass, 0))
```

* **Forces, impulses, drive targets and sensor reads belong here.** `update(dt)`
  stays the frame's final word on transforms — it runs after physics, the
  animation player and the sensors, and it is where you move a camera, a marker
  or anything else you are looking at.
* **It fires per substep, not per frame.** A frame longer than the timestep runs
  it several times; a frame shorter than one runs it not at all, and the
  leftover is carried over. `PhysxWorld` caps the catch-up at 4 substeps per
  frame, so a genuine hitch drops simulated time rather than spiralling.
* **It runs before that frame's `update()`**, because the physics session is
  updated first and `fixed_update` happens inside that call.
* **The measurements it reads are the ones the frame started with.** Sensors are
  sampled at the *end* of each substep, and the sensor session hands its retained
  copies to script handles once per frame (`drain()` empties a sensor's ring, so
  there is exactly one drainer — see [Sensors from a
  script](#sensors-from-a-script)). So `latest()` inside `fixed_update` is the
  newest measurement that existed when the frame began: one substep old at a
  matched frame rate, and the same value for both substeps of a double-length
  frame, with the pair arriving together on the next one. A controller is
  unbothered — acting on the last measurement is what a real one does, and the
  command written here lands in the solve that starts immediately after — but
  do not expect one new sample per call. `latest()` is `None` until the first
  substep of the session has been taken.
* **No physics world, no fixed clock.** Without the PhysX build, or with no
  physics session playing, `fixed_update` never runs and the console says so
  once when Play starts. Nothing here invents a clock of its own: a
  `fixed_update` that is not on the physics clock would be a lie about the one
  thing the name promises. (A scene with no physics *objects* is fine — the
  session builds a world regardless, and the clock is the world's, not any
  body's.)
* **Pause pauses it**, for free: paused means no session is updated, which means
  no step, which means no substeps.
* **Errors work as everywhere else.** The first raise is reported once with its
  traceback and disables that instance for the rest of the session — all of its
  methods, not just this one. Other scripts keep running.

#### Collisions: `on_collision_enter` / `on_collision_exit`

Two more optional methods, fired when the body governing the script's object
**starts** and **stops** touching another body:

```python
import threepp


class Impact:
    threshold = 2.0          # N*s; below this it is a nudge, not a hit

    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.material = obj.material
        self.rest = self.material.color.get_hex()

    def on_collision_enter(self, contact: threepp.editor.Collision):
        i = contact.impulse
        strength = (i.x * i.x + i.y * i.y + i.z * i.z) ** 0.5
        if strength >= self.threshold:
            print(self.obj.name, "hit", contact.other.name if contact.other else "?",
                  "at", strength)
        self.material.color.set_hex(0xff3020)

    def on_collision_exit(self, contact: threepp.editor.Collision):
        self.material.color.set_hex(self.rest)
```

`contact` is a `threepp.editor.Collision`, and it carries:

| field | meaning |
| --- | --- |
| `other` | the object on the far side, as its **concrete type** (`Mesh`, `Robot`, …) — or `None` when that body belongs to nothing the script can see |
| `point` | world-space position of the hardest-hit manifold point |
| `normal` | unit normal there, pointing **into this script's body** — the direction the other body is pushing it |
| `impulse` | total impulse over the manifold, in N·s, same orientation. Divide by the timestep to read it as a force |

* **`other` is the object the physics was authored on**, found by walking up the
  scene graph exactly as
  [`rigid_body_from_object`](#physics-from-a-script) does — so a contact against
  one cooked mesh of a spline's tube names the spline, not the mesh, and a
  contact against a simulated robot's link names the robot. It is a handle like
  any other: use it during the session, do not stash it across Play sessions.
* **Reporting turns itself on.** There is no box to tick. At Play the session
  finds the body governing every scripted object whose class defines either
  method and enables PhysX contact reporting on it. An object with the callbacks
  and **no body** gets one line in the console at Play and never fires — the
  same voice, and the same reason, as `fixed_update`'s missing clock.
* **Both sides are called.** If both objects in a pair carry scripts, each gets
  its own callback with its own `other`, and each `normal` points into its own
  body.
* **One touch is one pair of bodies.** A box landing on the ground touches
  through four manifold points, possibly through several shapes: that is *one*
  `on_collision_enter`. Settling on a surface does not re-fire it — only a
  genuine separation followed by a new touch does.
* **They are delivered from the frame sweep, just before `update(dt)`.** PhysX
  reports contacts from inside the solver; calling into Python there would let a
  script mutate a world the solver still owns. So the reports are copied into a
  queue and handed over where `update()` runs — after physics has stepped and
  mirrored, so a callback reads the settled world this frame will draw. Every
  edge this frame's substeps produced arrives in this frame.
* **Both edges always survive.** A touch that begins *and* ends between two
  deliveries — a fast bounce, or two substeps of one long frame — still produces
  `on_collision_enter` followed by `on_collision_exit`. The queue is a list, not
  a state flag; nothing collapses a whole touch into "no change".
* **Errors work as everywhere else.** The first raise is reported once with its
  traceback and disables that instance whole for the rest of the session.
* **A `ContactSensor` on the same body keeps working.** The opt-in is one bit per
  actor, shared: the sensor's noisy, rate-gated
  [readings](#sensors-from-a-script) and the script's edges come out of the same
  reports, and neither starves the other.

What is reported is what the physics world already generates: **dynamic against
dynamic, dynamic against static, and dynamic against kinematic**. PhysX does not
generate contacts for kinematic-against-static or kinematic-against-kinematic
pairs unless the scene asks for them, and this one does not — so a script on a
*kinematic* body hears about the dynamic bodies it pushes and about nothing else.

A **trigger volume** generates no contacts at all, so none of this fires for one
— that is a separate pair of callbacks, below.

Out of scope, deliberately: **`on_collision_stay`** (a resting contact stops
being re-reported the moment PhysX puts the pair to sleep, so a per-frame "still
touching" callback would lie), **per-contact-point lists** beyond the single
representative point above, **soft bodies** (a deformable volume has no rigid
actor to watch), and **watching a robot** — a simulated robot's links are the
articulation's, not the session's actor registry's, so a script on a robot
resolves no body of its own and gets the no-body line. (Being *hit* by one is
fine: a contact against a link names the robot, the same way a raycast does.)

#### Trigger volumes: `on_trigger_enter` / `on_trigger_exit`

The other pair, for the other kind of body. Tick **Trigger** on an object's
physics (see [Trigger volumes](#trigger-volumes)) and it stops colliding: things
pass through it and it reports who is inside. Both scripts hear about it — the
one on the **volume**, and the one on the **body that walked in**:

```python
def on_trigger_enter(self, other: threepp.Object3D): ...
def on_trigger_exit(self, other: threepp.Object3D): ...
```

A goal zone, a scoreboard, and a ball — three features composing (a trigger, a
script reaching another script's live instance, and the callbacks themselves):

`scoreboard.py`, on anything:

```python
import threepp


class Scoreboard:
    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.score = 0

    def scored(self, who: threepp.Object3D):
        self.score += 1
        print("goal by", who.name, "- score is now", self.score)
        self.obj.scale.y = 1.0 + self.score      # the bar grows
```

`goal_zone.py`, on the trigger volume:

```python
import threepp


class GoalZone:
    def start(self, obj: threepp.Object3D):
        # Resolve in start(), use later: every script instance exists by now,
        # but its own start() may not have run yet.
        self.board = threepp.editor.script_from_object(
            threepp.editor.scene().get_object_by_name("Scoreboard"))

    def on_trigger_enter(self, other: threepp.Object3D):
        if self.board is not None and other is not None:
            self.board.scored(other)
```

* **`other` is the object, not a `Collision`.** A trigger produces no manifold —
  no point, no normal, no impulse, because PhysX has none to give — and a struct
  full of zeroes would be a lie with fields on it. So the callback takes the far
  object directly, as its **concrete type** (`Mesh`, `Robot`, …), resolved the
  same way [`contact.other`](#collisions-on_collision_enter--on_collision_exit)
  is: the node the physics was authored on, so a body crossing one cooked mesh of
  a spline's tube names the spline, and a robot crossing it names the robot. It
  is `None` when that actor answers to nothing the scene can name.
* **Both sides are called, and only one of them is the volume.** The volume's
  script is told who came in; the entering body's script is told what it entered.
  The entering body is *not* itself a trigger and needs no tick of its own — a
  script that only defines these two methods, on an ordinary falling box, works.
* **Nothing turns it on.** Contact reporting is an opt-in bit per actor; trigger
  reporting is the shape *being* a trigger, which the document already says. So
  there is no box to tick beyond the one that made the volume. A script with
  these methods on a body that is not a trigger and never gets entered by one
  simply never fires — but a script with them on an object with **no physics body
  at all** gets one line in the console at Play, the same voice and the same
  reason as `fixed_update`'s missing clock.
* **One crossing is one pair of bodies.** A compound volume, or a body with
  several shapes, overlaps through several shape pairs: that is *one*
  `on_trigger_enter`, refcounted the same way a collision is, so a partial
  separation cannot report an exit the rest of the overlap is still holding.
* **Delivered from the frame sweep, just before `update(dt)`**, exactly like the
  collision callbacks and for exactly the same reason: PhysX reports from inside
  the solver, so the report is copied into a queue and handed over where
  `update()` runs. Contacts are delivered first, then triggers — an arbitrary but
  fixed order, so a script defining both sees the same sequence every run.
* **Both edges always survive.** A body that crosses a thin volume *and leaves*
  between two deliveries still produces `on_trigger_enter` followed by
  `on_trigger_exit`. The queue is a list, not a state flag.
* **There is no "still inside" event.** PhysX reports `TOUCH_FOUND` and
  `TOUCH_LOST` for a trigger and nothing between them — `eNOTIFY_TOUCH_PERSISTS`
  is explicitly unsupported for triggers. A body sitting inside a volume also
  falls **asleep**, and a sleeping pair reports nothing further. Neither is a
  problem *because* there is no persist event to lose: the enter has already been
  delivered and the exit still arrives when the body wakes and leaves. Track the
  two edges yourself if you need an "is inside" flag; do not try to derive one
  from a per-frame event that does not exist.
* **Errors work as everywhere else.** The first raise is reported once with its
  traceback and disables that instance whole for the rest of the session.

Interplay, stated rather than left to be discovered:

* **No collisions.** A trigger shape generates no contacts, so
  `on_collision_enter` / `on_collision_exit` never fire for an overlap with one,
  a `ContactSensor` never measures one, and bodies pass through. That last part
  *is* the feature.
* **Raycasts still hit it.** A trigger keeps its scene-query flag, so
  [`threepp.editor.raycast`](#raycasts-threeppeditorraycast) reports it like any
  other body — see the table there.
* **Triangle meshes are substituted.** PhysX has no triangle-mesh trigger, so a
  volume authored as one is cooked as a convex hull with one console line saying
  so. [The rule and the table](#trigger-volumes) are in the authoring section.
* **Trigger-against-trigger is nothing.** PhysX does not report overlaps between
  two trigger shapes, so two volumes crossing each other produce no callbacks on
  either.
* **Soft bodies have no trigger.** The checkbox is hidden for `body=soft`, and a
  soft body crossing a volume is not reported — a deformable volume is not a
  rigid actor and never enters the trigger pair.

#### Raycasts: `threepp.editor.raycast`

A collision tells you what already happened. A **raycast** asks a question about
the world *now* — what is under my feet, what is in front of me, what am I
aiming at — and answers synchronously, wherever you call it from:

```python
hit = threepp.editor.raycast(origin, direction, max_distance=100.0, ignore=self.obj)
if hit is not None:
    hit.object      # the object the physics was authored on, as its concrete type, or None
    hit.point       # threepp.Vector3, world space
    hit.normal      # threepp.Vector3, unit, pointing out of the surface hit
    hit.distance    # float, metres from origin along the ray
```

The whole ground check, which is what it is mostly for:

```python
import threepp


class Hopper:
    probe = 0.6              # metres: just past the bottom of a unit box

    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.body = threepp.editor.rigid_body_from_object(obj)

    def fixed_update(self, dt: float):
        if self.body is None:
            return
        down = threepp.Vector3(0, -1, 0)
        hit = threepp.editor.raycast(self.body.position, down, self.probe, ignore=self.obj)
        if hit is not None and self.body.velocity.y <= 0.0:
            self.body.apply_impulse(threepp.Vector3(0, 4.0 * self.body.mass, 0))
```

* **A miss is `None`. Not playing is a `RuntimeError`.** Those are two different
  answers and they are kept different: if "no physics world" also came back as
  `None`, a ground check written outside Play would look like it worked and
  silently report empty air forever. Same reasoning as
  [`encoder_from_object`](#sensors-from-a-script) raising for an unknown joint.
  A zero-length `direction` is a `ValueError`, as is a `max_distance` of zero or
  less.
* **`ignore=obj` excludes every actor governing that object** — the walk up the
  ancestry that finds its body, and then *all* of the actors recorded against
  the node it lands on, since a subtree collider or a compound is many actors
  under one authored node. **Pass your own object for any cast that starts
  inside your own collider**, which a cast from `body.position` always does: a
  ray that starts inside a box hits that box at distance zero, and a ground
  check without `ignore` is a ground check that has found itself.
* **`origin` and `direction` are `Vector3`s in world space.** The direction is
  normalised for you, so `Vector3(0, -1, 0)` and `Vector3(0, -37, 0)` are the
  same ray. `max_distance` is in metres and defaults to unbounded, so a miss
  means the ray genuinely left the world rather than ran out of budget.
* **The nearest hit, and only that one.** There is no all-hits form and no
  sweep; this is the primitive, not a query library.
* **`hit.object` is the object the physics was authored on**, as its concrete
  type (`Mesh`, `Group`, `Robot`, …) — the same answer `contact.other` gives, so
  a hit on the fourth cooked mesh of a spline's tube names the spline, and a hit
  on a robot's forearm names the robot. It is `None` only when the actor answers
  to nothing the scene can name; `hit.point`, `hit.normal` and `hit.distance`
  are still valid in that case.
* **Call it from anywhere a script runs**: `update`, `fixed_update`, the
  collision callbacks and the trigger callbacks. All of those run on the editor
  thread with the GIL held and *outside* `simulate()` — a substep is
  `pre-hook → simulate → fetchResults → post-hook`, so the pre-substep hook
  `fixed_update` runs from sits between one substep's results and the next
  solve, which is exactly where a scene query is legal. A cast from
  `fixed_update` reads the substep it is standing in, not last frame's pose.

What the query sees in this world, stated rather than assumed:

| body | seen? |
| --- | --- |
| static | yes |
| dynamic | yes |
| kinematic | yes — a kinematic actor is a `PxRigidDynamic`, and the query asks for static and dynamic both |
| an articulated robot's links | yes, and each one answers as the **robot** |
| [trigger volumes](#trigger-volumes) | **yes** — a trigger shape loses `eSIMULATION_SHAPE` but keeps `eSCENE_QUERY_SHAPE`, so a ray finds a volume that bodies pass straight through. Pass it to `ignore`, or check `hit.object`, if a ground check must not stop at one |
| soft bodies | **no** — a deformable volume's shape is created without PhysX's scene-query flag, so it is not in the query structures at all |

#### Talking to other scripts: `threepp.editor.script_from_object`

`scene.get_object_by_name("Door")` gives you the door's *node*.
`threepp.editor.script_from_object(...)` gives you the door's **live script
instance** — the actual Python object this Play session is driving:

`door.py`, on the door:

```python
import threepp


class Door:
    open = False

    def start(self, obj: threepp.Object3D):
        self.obj = obj

    def on_opened(self):
        self.obj.position.y += 1.0
```

`button.py`, on whatever opens it:

```python
import threepp


class Button:
    def start(self, obj: threepp.Object3D):
        # Resolve neighbours here, once: every instance exists by now.
        self.door = threepp.editor.script_from_object(
            threepp.editor.scene().get_object_by_name("Door"))

    def update(self, dt: float):
        if self.door is None or self.door.open:
            return
        if threepp.editor.is_key_down("SPACE"):
            self.door.on_opened()      # calling a method IS the signal
            self.door.open = True      # and so is setting an attribute
```

There is **no event bus and no message type**, deliberately. What comes back is
the instance, so its methods and its attributes are the whole API — this is
Unity's `GetComponent` without the component-type dance, since an object carries
exactly one script.

* **Every instance exists before any `start()` runs.** Play brings every script
  up first — compile, construct, apply the authored field values — and only then
  calls `start()` on all of them. So resolving a neighbour in `start()` works
  whichever way round the scene happens to be, and the neighbour's parameters are
  already the ones the inspector shows rather than the class defaults. What is
  *not* guaranteed is that its `start()` has run: if you need state a neighbour
  builds there, resolve in `start()` and use it from `update()`. That is the
  pattern above, and it is the one to copy.
* **`None` is a normal answer**, never an exception — a missing neighbour is
  something a script checks for, exactly as it is for
  `rigid_body_from_object`. You get `None` when nothing is playing, when the
  object carries no script, and when that script's instance is dead: a
  constructor that raised, or a `start` / `update` / `fixed_update` that raised
  later. A disabled script is disabled whole, so it stops answering lookups as
  well. That is about the *lookup*, though: a reference you are already holding
  onto a script that has since been disabled stays perfectly good Python. It is
  simply not being driven any more.
* **The exact object, never an ancestor.** `rigid_body_from_object` walks up the
  scene graph because a collider governs a whole subtree; a script governs
  nothing but the node it was authored on, so this asks for that node and no
  other. Pass the object the script is on.
* **Do not keep it across sessions.** Same rule as every other handle: resolve in
  `start()`, use it during the session, ask again after the next Play. A
  reference held past Stop is a harmless dead object — the session has dropped
  it and nothing calls it again — not a dangling pointer, but it is not the new
  session's instance either, and the new session will never see your writes to
  it.
* **Call it from anywhere a script runs** — `start`, `update`, `fixed_update`,
  the collision and trigger callbacks, and `stop`, where every instance is still
  alive. It
  reads the session's own instance list and touches neither the scene nor the
  physics world, so it needs no PhysX build: unlike the handles above it, this
  is there in every build that has scripts at all.

#### The Script Editor

`New Inline Script` in the Script section writes a template into the object and
opens it in the bottom panel's **Scripts** tab, bringing the panel up if it was
collapsed. `Edit…` opens an existing one the same way.

**As many at once as you open.** Inside the Scripts tab is a second bar with one
entry per open script, labelled with its object and carrying an asterisk while
its buffer differs from the document. Each keeps its own text, its own undo
position in the box and its own syntax error until you close it with its ×, so
writing two scripts that talk to each other is a click between tabs rather than
a round trip through the hierarchy. When more are open than fit, the ▾ button at
the left of the bar lists them all by name. Still **one script per object** —
that is `ScriptConfig`'s rule; this is one tab per object.

Selecting an object that already has a tab open **raises that tab**, and that is
all the selection does: on a change of selection, not on every frame the visible
tab and the selection disagree — otherwise opening a script for anything other
than what is selected would bounce straight back. Selecting an object with no
tab open opens nothing; `Edit…` is what opens one. The single editor this
replaced instead *retargeted itself* onto the selection, and could only ever do
it when the buffer was clean, because pointing a buffer somebody is typing into
at another object drops what they typed.

It is docked rather than floating because floating is what was wrong with it: a
text box big enough to write in covered the object the script was about. Beside
the console it takes room from nothing, and the bottom panel's top edge drags,
so "big enough" costs a drag rather than a window parked over the scene. The
panel's `v` collapses the lot; the Scripts tab itself is there only while at
least one script is.

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
  script and closes the tab (one `Ctrl+Z` away, and the buffer is still there
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
widget and no new dependency. Editing an external `.py` in this tab is out of
scope — that file already has an editor, and so, now, does inline source: see
below.

#### Edit in VS Code

`Edit in VS Code` in the Script section opens the script in a real editor, with
completion for the `threepp` API. It works on both forms, and what it does
differs because what they are differs.

**A file script is simply handed over.** VS Code opens on the folder, the file
is revealed, and nothing is watched: Play recompiles the `.py` from source every
time, so the file was already hot. Edit, save, press Play.

**Inline source has no file, so one is made.** The committed source is exported
to `<temp>/threepp-editor/scripts/<uuid8>_<name>.py`, VS Code opens on it, and
the editor polls that file about once a second from the frame loop. Every save
comes back in through the Script Editor's own **Apply** — the same tab and
line-ending normalization, the same compile check, the same undoable commit —
and the tab shows the source read-only with a banner and a **Stop external
edit** button for as long as the session lives.

Details that are not incidental:

* **Content decides, not the write time.** Two saves a millisecond apart can
  carry the same `last_write_time`; a watcher that misses one of those has
  silently lost work. The file is read and compared instead — and compared
  *normalized*, because otherwise every save from a Windows editor would look
  like a change to every line in the file.
* **Ten saves are one undo step.** The session holds a command-stack transaction
  open, so saves after the first merge into the same entry. The first still
  starts a fresh one: a transaction never merges into what preceded it.
* **The session survives a play/stop.** It holds the object's uuid, not a
  pointer, and re-resolves it on every poll — so the graph can be replaced
  underneath it and external editing simply carries on. Iterating on Play with
  VS Code open is the point of the whole feature.
* **A save that arrives while playing is parked.** Committing then would be
  swallowed by the snapshot restore on Stop, so nothing is consumed, the banner
  says the save applies on Stop, and the poll after Stop applies that same file
  to the restored object. (An async import that lands during Play is parked the
  same way.)
* It ends on **Stop external edit**, on the script being cleared, on the object
  leaving the scene, on the file being deleted, or when the editor closes — and
  the scratch file goes with it. One session at a time, the same rule the Script
  Editor window follows.
* Polling, not `ReadDirectoryChangesW`/`inotify`: a watcher means a thread, and a
  thread means a callback arriving mid-frame with the scene half-rebuilt. This
  lands in a frame, on the main thread, where every other editor mutation lands.
* Launching is `ShellExecute`/`CreateProcess`, never `std::system`: detached,
  non-blocking, and without the console flash a `.cmd` on the PATH would produce
  (`code` *is* a `.cmd`, so the real `Code.exe` beside it is preferred). With no
  `code` on the PATH at all, the file opens in whatever the OS uses for `.py`
  and the console says so — that editor knows nothing about the stubs.

##### The generated workspace

Both tiers first write `.vscode/settings.json` in the folder being opened, if —
and only if — there is not one there already. Your own settings are never
overwritten; delete the file and the editor writes a fresh one.

```jsonc
{
    "python.analysis.stubPath": "C:/dev/threepp/python/threepp",
    "python.analysis.diagnosticSeverityOverrides": {
        "reportMissingModuleSource": "none"
    },
    // "python.defaultInterpreterPath": "C:/Users/you/mambaforge/envs/robostack/python.exe",
    "files.eol": "\n",
    "editor.insertSpaces": true,
    "editor.tabSize": 4
}
```

`stubPath`, not `extraPaths`, and the difference is worth knowing. The stub
package is `python/threepp/threepp/__init__.pyi`, so a `stubPath` of
`python/threepp` resolves `import threepp` straight onto it — the **native**
module, which is what the editor's embedded interpreter actually serves. An
`extraPaths` of `python/` also resolves (both were checked with pyright against
the real tree), but it lands on the wheel's Python package first and would offer
`flush_material`, `threepp.rl` and friends, none of which exist inside the
editor. The cost of the stub route is `reportMissingModuleSource` — the stubs
have no `.py` beside them, because the module is compiled into the editor — and
the generated file turns that one diagnostic off.

The path is absolute and baked in at build time (`THREEPP_EDITOR_PYTHON_STUBS`,
from `PROJECT_SOURCE_DIR`); `THREEPP_PYTHON_STUBS` in the environment overrides
it, which is what an installed binary that has moved away from its source tree
needs. The commented-out `defaultInterpreterPath` is the other half: point it at
your own interpreter — a RoboStack environment, say — and VS Code will also
resolve `numpy`, `rclpy` and anything else that interpreter has. Scripts still
*run* in the editor's embedded interpreter; that line only changes what VS Code
understands.

Handles are yours to annotate. The editor passes the object's concrete type, and
saying so is what turns completion on for it:

```python
import threepp

class Spinner:

    speed = 1.5

    def start(self, obj: threepp.Mesh) -> None:   # or threepp.Robot, threepp.Light, ...
        self.obj = obj
        mat = obj.material
        assert isinstance(mat, threepp.MeshBasicMaterial)
        mat.color.set_hex(0xff8844)                # completes, and is checked

    def update(self, dt: float) -> None:
        self.obj.rotation.y += self.speed * dt
```

The `assert` is not ceremony. `.material` is typed `Material | None`, and
`Material` carries only what every material shares — `name`, `opacity`, `side`,
`blending`, `needs_update()`, and so on. `color` is not among them: a
`MeshNormalMaterial` has none, so a checker is right to reject it until you say
which material you mean. `isinstance` does that and opens up the rest of the
concrete surface. The same move works on an `Object3D` from
`get_object_by_name`, which is likewise typed as the base while the runtime
object is the concrete subclass.

The stubs are a *superset* of what the editor serves: they are generated from
the full wheel module, which also binds the renderers, loaders, physics and
sensors that the script host deliberately leaves out. `threepp.Mesh` completes
and works; `threepp.GLTFLoader` completes and raises `AttributeError` at Play.

`threepp/editor.pyi` is the exception, and the reason it is **hand-maintained**.
`threepp.editor` runs the superset the other way: `RigidBody`, `SoftBody`,
`Articulation` and the three `*_from_object` lookups come from
`python/src/bind_editor_physics.cpp`, which only *this* app compiles
(`apps/editor/CMakeLists.txt`, registered by `ScriptModule.cpp` behind
`THREEPP_EDITOR_WITH_PHYSX`). They are handles onto a live `PhysicsPlaySession`,
so the wheel has no use for them and does not bind them — and a stub generated
from the wheel would therefore delete them. `python/scripts/gen_stubs.py` lists
the file in `HAND_MAINTAINED` and restores it verbatim after every run. Change
`bind_editor_physics.cpp`, and edit `editor.pyi` by hand to match.

**Errors are never fatal.** Every call into Python is wrapped; a traceback goes
to the console, the offending instance is disabled for the rest of the session,
and everything else keeps playing. A script that raises on every frame is
reported once, not sixty times a second. Syntax errors, a missing class, a file
that has moved away and an `update()` that throws are all just messages — the
last one is also shown in red in that object's Script section.

**Input: `threepp.editor.is_key_down(key)`.** A play session is something you can *drive*.
Poll it from `update()` — `'W'`, `'UP'`, `'KP8'`, `'SPACE'`, the same key names
`Canvas.is_key_down` takes — and a script becomes a controller: teleop a robot, nudge a
body, cycle a mode.

```python
def update(self, dt):
    from threepp import editor
    vx = (1.0 if editor.is_key_down("UP") else 0.0) - (1.0 if editor.is_key_down("DOWN") else 0.0)
```

It answers **False while ImGui wants the keyboard** — real text entry, an open popup, the file
browser — on the same rule the editor's own shortcuts follow, so driving a robot can never eat
somebody's rename. It reads ImGui's key state rather than the canvas's: `preventKeyboardEvent`
gates on `WantCaptureKeyboard`, which stays true for as long as a panel keeps focus after a
click, so the canvas's held-key set would be stale exactly when someone is watching the
viewport. In a build or a pass with no window (no provider installed) it answers False rather
than raising, so a script that steers still *runs* headlessly, just uncommanded.

**A script that reads the keyboard takes it over.** The keys a controller reaches for are the
keys the editor already uses: `W`/`E`/`R` are the gizmo modes, `Q` is local/world, and the
numpad is the axis viewpoints. Both firing on one press means driving a robot forward also
retargets the gizmo and snaps the camera to a front view. So the first `is_key_down` call of a
play session hands the **plain keys** to the scripts for the rest of it — one console line says
so — and the shortcuts come back on Stop. Modified commands are untouched: `Ctrl+S`, `Ctrl+Z`
and the `Alt`+digit viewpoints keep working while playing, because a script polls unmodified
keys and silently losing save-while-playing would be its own surprise. `Shift+F`
([Follow Selection](#features)) is answered ahead of the hand-over for the same reason and one
more: the session where you most want to chase the thing you are flying is exactly this one.
A scene whose scripts never ask keeps every shortcut.

This is the only input a script gets, and it is a poll, not an event: there is no key-press
callback, no mouse, and the inspector is read-only while playing — so a parameter is something
you set *before* pressing Play, not a live control.

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
never seen. Eleven areas are linked — math, textures, core, geometries, curves,
materials, objects, animation, cameras, lights, robot — and the renderer,
loader, physics, sensor and Vulkan areas are deliberately left out: a script has
no business creating a window, and the editor already owns the ones that exist.

### Generators in `userData`

A **generator** is inline Python that AUTHORS content, as opposed to a script,
which is behaviour that runs during Play. It normally belongs to the **scene** — a
generated scene's rule is a property of that scene, and the root is where you
would look for it — so select `Scene` in the hierarchy and the inspector offers a
Generator section. Any `Group` can carry one too, for a scene that wants several
independently re-runnable ones (props, then clutter on top).

```
userData["generatorSource"]  count = 400\nfor i in range(count): ...
userData["generatorFields"]  count=400;seed=7
```

Source is newlines and arbitrary Python, so like a script's it gets its own plain
entry rather than riding in the flat `key=value` format `PhysicsConfig` uses.
Inline is the only form on purpose: the point is a document that carries the rule
that built it, and a path into another machine's disk gives that up.

Press **Regenerate** and what the script builds lands under ONE child tagged
`userData["generated"]`, replaced wholesale on every run:

```python
import math, random
import threepp
from threepp import editor

random.seed(7)                       # determinism is yours to declare

geometry = threepp.BoxGeometry(0.5, 0.5, 0.5)
material = threepp.MeshStandardMaterial()
material.color = threepp.Color(0x88aa55)

field = threepp.InstancedMesh(geometry, material, 400)
for i in range(400):
    angle = random.uniform(0, 2 * math.pi)
    r = 8.0 * math.sqrt(random.random())     # uniform over a disc
    m = threepp.Matrix4()
    m.make_rotation_y(random.uniform(0, 2 * math.pi))
    m.set_position(r * math.cos(angle), 0.25, r * math.sin(angle))
    field.set_matrix_at(i, m)
field.instance_matrix_needs_update()

editor.add(field)
```

**Generated content can carry configs.** `Object3D.set_user_data(key, value)` is
the write side of the `get_user_data` escape hatch, and it is what makes a
generated level *playable*: physics, a sensor, a script and an animation all ride
in `userData` as one flat `key=value;…` string, so the generator writes exactly
what the inspector would have.

```python
pillar.set_user_data("physics", "body=static;shape=box;friction=0.6;restitution=0.05")
```

Strings only, for the reason the reader gives — `userData` round-trips scalars
and the editor's own configs are all one string — and an empty value **removes**
the entry, which is how every config spells "off". Without this a generator could
build a hundred pillars and not one of them would collide with anything.

`editor.add(object, parent=None)` is the only authoring verb, and it returns what
you passed so you can nest into it. It appends to a node that is **not in the
document yet**; the editor attaches that node only once the script finishes. Two
consequences worth relying on: a script that raises — immediately, or after
adding half its objects — commits nothing at all, and calling `add()` outside a
generator run raises rather than quietly doing nothing. `editor.scene()` reads the
live scene, for placing content relative to what already exists; objects reached
that way are not this generator's output and survive its re-runs.

**Opening a scene never runs a generator**, exactly as for scripts. The output is
ordinary saved scene content, so loading a generated `.json` shows the result
without executing anything — and *that* is what makes such a file safe to open at
all. Regenerating is always something you asked for. It follows that a scene from
somebody else is data until you press the button, and pressing it runs their code
with your privileges; read it first, the same as any script.

The script is the source of truth for its output. Because a run replaces the
output wholesale, a material or a physics config hand-edited onto generated
content is gone at the next Regenerate — set it in the script instead. (The
spline's tube sync can preserve a uuid and a material because it knows the output
is one mesh; a generator's output is arbitrary.) **Clear** removes the script and
its output together, in one undo step, rather than leaving orphaned content
carrying the generated tag for a later generator to adopt.

**Edit in VS Code** exports the source to a scratch `.py`, opens it, and polls it
back once a second, committing each save through the undo stack — the same round
trip an inline behaviour script gets, including the `.vscode/settings.json` that
makes Pylance complete `import threepp`. A save also re-runs the generator, so
the loop is edit-save-look; a file that does not parse is synced but not run, so
the last good output stands.

Determinism is *your* declaration, not the editor's guarantee: seed your RNG (as
above) and a regenerate reproduces the scene exactly, which is what makes a
generated scene usable as reproducible training data. An unseeded script is a
different scene every run, and nothing stops you writing one.

Known gaps: `generatorFields` round-trips but is not yet injected into the run, so
there are no inspector inputs for parameters — a script's own module-level
`count = 400` would overwrite an injected global, and resolving that needs a
convention rather than plumbing. There is also no headless entry point yet, so
batch generation over many seeds is not a one-liner.

### Splines in `userData`

**Add ▸ Spline** creates a `Group` carrying `userData["spline"]`, with four
control-point children forming an arc. The rule is one sentence:

> **Every direct child of a spline is a control point, in child order —
> except the generated mesh, which is tagged.**

Nothing else marks them. A control point is a plain `Object3D`, its local
position is that point's position in the spline's space, and reordering,
renaming, deleting or dragging one is the ordinary operation on an ordinary
scene node. That is the entire point of the design: serialization, undo/redo,
the transform gizmo, the hierarchy, duplication and delete are the ones the
editor already has, and the spline feature adds none of them back.

The curve's own parameters ride in the same flat format as physics and
animation:

```
type=centripetal;closed=0;tension=0.5;samples=24;mesh=none;radius=0.25;radialSegments=8;width=4;uvLength=4
```

| key | values | meaning |
| --- | --- | --- |
| `type` | `centripetal`, `chordal`, `catmullrom` | three.js's parameterisations; centripetal avoids cusps |
| `closed` | `0`, `1` | join the last point back to the first |
| `tension` | `0`…`1` | `catmullrom` only — stored and inert for the other two, as in three.js |
| `samples` | `1`…`200` | curve samples per segment; drives the overlay *and* what the mesh is fitted from |
| `mesh` | `none`, `tube` | what the spline generates as real geometry
| `radius` | metres | `tube` — cross-section radius |
| `radialSegments` | `3`…`64` | `tube` — sides of the cross-section |

`SplineConfig::encode()` / `decode()` own that format, unknown keys are ignored
on read, and — unlike `PhysicsConfig` — `write()` never erases the entry: the
entry *is* the spline, so removing it would stop the object being one.

**Authoring.** Select the spline for type / tension / closed / samples and an
**Add Point** button, which appends past the last point along the last segment
so the curve visibly extends. Select a control point and the inspector shows its
index and **Insert Before** / **Insert After**, which put a new point midway to
its neighbour. Removing a point is `Del`. Every one of those is an undo entry.

**The curve you see is not in the document.** Control points draw nothing, so
the editor samples `CatmullRomCurve3` into one `Line` per spline under the
editor-only overlay — never saved, never picked, never in the camera preview.
It is rebuilt when a hash over the point count, their positions and the encoded
config changes, which is what makes dragging a point redraw the curve live.
Control points additionally get a marker icon through the same machinery
cameras and lights use, so they are clickable in the viewport at all.

**The mesh you generate is.** Set **Mesh** to Tube in the spline section and the
spline grows one more child: a real `Mesh`, with a `MeshStandardMaterial`,
shadows on, carrying `userData["splineDerived"] = "1"`. That tag is the only
thing separating it from a control point — every count and index in the editor
goes through `SplineConfig::controlPoints()` / `pointIndexOf()` rather than
`children`, so adding geometry never shifts a point index. It is a document node
in every other respect: it saves, it reloads, and a scene with a tube in it
renders and collides in a plain viewer with no editor present. Select it and the
ordinary Material and **Physics** sections apply — or put the physics on the
spline itself, which collides as the tube underneath it.

Tube sweeps threepp's `TubeGeometry` along the curve, at the radius and radial
segment count the config carries. A closed cross-section is also what a triangle
mesh collider handles well, which is why a tube needs nothing special to stand
on.

**Regeneration is derived state, not a command.** The undoable step is the
config edit; the sync pass then adds, rebuilds or removes the mesh to follow
whatever the config says — so undoing "Mesh = Tube" brings the mesh back on the
next frame's sync, and there is never a second entry on the undo stack fighting
the first. Rebuilds preserve the node: same object, same uuid, same material,
same `userData`, with only the geometry swapped (and the orphaned one disposed).
Anything you configure on the mesh survives dragging the control points around.
A loaded document already has its mesh, and the first sync adopts that one
rather than adding a second.

**From a script.** `threepp.editor.spline_from_object` turns an authored spline
into a `SplinePath` — no child filtering, no config parsing, and the authored
`closed`, curve type and `tension` are honored. It returns `None` when the
object is not a spline or has fewer than two control points, so the result of
`get_object_by_name` pipes straight in:

```python
import threepp


class FollowSpline:
    spline_name = "Spline"   # the spline Group to follow
    speed = 0.2              # laps per second

    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.u = 0.0
        self.path = threepp.editor.spline_from_object(
            threepp.editor.scene().get_object_by_name(self.spline_name))

    def update(self, dt: float):
        if self.path is None:
            return
        self.u = (self.u + self.speed * dt) % 1.0
        # get_point_at is arc-length parameterised, so the speed is constant
        # along the curve rather than per segment.
        self.obj.position.copy(self.path.get_point_at(self.u))
```

Attach it to any object, set `spline_name` to your spline in the inspector, and
press Play. `get_point_at(u)` and `get_tangent_at(u)` answer in **world space**
(the follower's `position` is world space too, as long as it sits under the
scene root); `get_length()` is the arc length in the spline's local space, so a
scaled spline scales distances with it.

The path's contract in one sentence: the local-space curve — control points,
`closed`, type, `tension` — is captured when the path is created (or
`refresh()`ed), and the spline's world transform is applied live on every
sample. A spline riding a moving platform therefore stays followable for free,
while moving control points during Play needs `path.refresh()` before the
change bends the path. Sampling runs off the curve's own arc-length table,
independent of the overlay's `samples` density. For anything beyond points and
tangents the captured local-space curve is exposed as `path.curve` (a plain
`threepp.CatmullRomCurve3`), and `Object3D.get_user_data(key)` remains the raw
escape hatch: it returns any string `userData` entry — every editor config
(`spline`, `physics`, script fields) is one flat `key=value;…` string.

### Conveyors in `userData`

**Add ▸ Conveyor** creates a `Group` carrying `userData["conveyor"]`, with
three waypoint children forming a straight run at working height. The authoring
model is the spline's, restated:

> **Every direct child of a conveyor is a path waypoint, in child order —
> except the generated parts group, which is tagged.**

A waypoint is a plain `Object3D`; drag it with the gizmo, insert, delete and
undo like any node. The belt's parameters ride in the usual flat format:

```
width=0.6;speed=0.6;reverse=0;smooth=1;separator=0;wallHeight=0.5;rollerRadius=0.05;cleatHeight=0.15;cleatSpacing=0.6;samples=12;frame=1
```

| key | values | meaning |
| --- | --- | --- |
| `width` | metres | belt width |
| `speed` | m/s | surface speed along travel; `0` = a static machine |
| `reverse` | `0`, `1` | flip the travel direction |
| `smooth` | `0`, `1` | Catmull-Rom through the waypoints, or the raw polyline |
| `separator` | `0`, `1` | a collision-only guide wall along the path instead of a belt |
| `wallHeight` | metres | separator only |
| `rollerRadius` | metres | roller segments — cylinder radius |
| `cleatHeight`, `cleatSpacing` | metres | cleat segments — flight bar size and pitch |
| `samples` | `2`…`64` | resample density per waypoint segment |
| `frame` | `0`, `1` | generate the support frame (rails, legs, end drums) |

Two things live **on the waypoint node itself**, under
`userData["conveyorWp"]` (`radius=…;seg=flat|rollers|cleats`), so they follow
the node through reorder, undo and serialization; the inspector's **Conveyor
Waypoint** section edits both:

- **Corner radius** — a non-zero radius rounds the corner at this waypoint
  with an exact circular fillet. The waypoint stays *on* the path; the arc's
  centre and tangent points are DERIVED, inserted tangent to both adjacent
  segments, and the radius **clamps itself** to what those segments allow —
  so a bend cannot kink, whatever gets dragged where. (Chained rounded
  corners split the straight they share rather than overlapping.) Under Play
  the whole bend is one rotating collider built from the same fillet, so the
  surface velocity is exactly tangential everywhere along it. A U-turn is two
  90° corners.
- **Segment surface** — the span leaving the waypoint is a flat belt by
  default; per segment you can choose a **roller bed** or **cleats** (flight
  bars standing across the belt that travel with it and catch cargo on an
  incline). Runs share boundary points, so a flat→rollers change meets
  gap-free. A roller bed is not décor: each roller is a **real driven
  collider** — a kinematic capsule spinning about its own axis at
  `speed / rollerRadius` — and a rollers span builds no belt box underneath,
  so cargo genuinely rides the rollers, gaps included.

**Walls belong to conveyors.** The **Add Wall** button in the Conveyor section
attaches a wall as a child of the conveyor — a Group carrying
`userData["conveyorWall"]` (`height=…`) whose own children are the wall's
points, so dragging a point, rotating the whole wall with the gizmo, deleting
and undoing are all the ordinary operations. The default is **one short
segment at the start of the belt**, on the outer edge — a piece, not a plan.
Building the walls you mean is incremental: **slide a segment along the belt**
with the gizmo (it stays on the edge — see below), **grow it point by point**
(select an end point, *Insert After*, drag the new point where the wall should
reach next, repeat), and press **Add Wall again for the next section** — each
new wall tiles in a gap after the furthest one on the same side, then starts
over on the other side once that edge is built out. Sections are the model:
a stretch **with no wall is just the gap between two wall pieces**, which is
exactly where a diverter feeds cargo off the belt. Drag any point **toward
the middle** to sweep that stretch inward into a diverter. The built wall
always
FOLLOWS the path between its points — each point reads as a station along the
belt plus a lateral offset, blended between points — so an edge guide hugs a
bend exactly rather than cutting the chord, and its base rides the deck
throughout, slopes included: author entirely in plan, nothing ever wedges
underneath. Wall colliders are deliberately **low-friction**, so the grippy
belt keeps pushing while cargo slides along a plow into its lane (a wall built
from belt material would hold cargo like a hand instead). The old
whole-conveyor `separator` flag remains for free-standing rails between
machines.

The viewport helps you author this: every waypoint gets a clickable diamond
marker (the spline points' machinery, with its own glyph), the path overlay
carries **chevrons pointing along the flow** (they flip with `reverse`), and
selecting a rounded corner draws its derived arc centre and the two tangent
spokes, so what the radius is doing to the path is visible while it is tuned.
The ball on the arc midpoint is a **radius handle**: drag it along the
corner's bisector to widen or tighten the bend directly in the viewport (a
sharp corner offers the ball just off the waypoint, so a bend can be dragged
into being). The whole drag lands as one undo entry, exactly like a gizmo
move, and dragging past what the segments allow just pins the bend at its
maximum.

**The look is generated, and it is first-party.** Every conveyor carries one
tagged child (`userData["conveyorDerived"]`), a Group holding the parts: the
belt ribbon with a procedural scrolling texture, roller cylinders, cleat bars,
the separator wall, and a support frame — side rails following the path, legs
down to the conveyor's local ground plane, an end drum (pulley) at each open
end. All of it is generated geometry; there are no imported models anywhere in
the feature. Each part is tagged with its role in `userData["conveyorRole"]`
(`belt` / `roller` / `cleat` / `wall` / `frame` / `drum`), which is how the
play session finds the moving parts — and how your own tooling can.

Unlike a spline's single tube, **regeneration is wholesale**: the part count
varies with the path, so the sync pass replaces the group's *content* whenever
a waypoint or the config changes (the group node itself keeps its uuid). Treat
the parts as output — a material you hand-edit on one belt mesh will not
survive the next regeneration.

**Play makes it convey.** The conveyor session builds kinematic colliders into
the physics session's world: straight runs as chains of drag boxes, each bend
as one body rotating about its arc centre, tiled with convex wedges that share
their radial faces. Every physics substep the colliders' kinematic targets are
advanced along travel and then teleported back — the surface never moves, but
everything resting on it inherits the belt's surface velocity. That trick works
for rigid bodies **and for soft bodies**: the wedges are cooked GPU-compatible,
so a `Body::Soft` object dropped on a belt is carried like anything else.
Cleat bars are the one exception — a teleported-back wall would un-do its push,
so the bars genuinely travel, wrapping end→start with a teleport and folding
flat at the pulleys. Alongside the physics, the session scrolls the belt
texture, spins the rollers and drums, and drives visible bars along each cleat
track, all from one speed scale so the picture never disagrees with the
simulation.

**A conveyor works without the editor.** Everything above is plain scene
content, so a saved document carries the whole machine. An external consumer —
a soft-body simulation, a headless data generator — rebuilds the physics in a
few lines against its own `PhysxWorld`:

```cpp
#include "threepp/extras/conveyor/ConveyorPhysics.hpp"
#include "threepp/extras/editor/ConveyorConfig.hpp"

std::vector<threepp::conveyor::ConveyorSpec> specs;
scene->updateMatrixWorld(true);
scene->traverse([&](threepp::Object3D& object) {
    const auto config = threepp::editor::ConveyorConfig::read(object);
    if (!config) return;
    auto spec = config->spec(object);            // local-space description
    for (auto& wp : spec.waypoints)              // → world space
        wp.pos.applyMatrix4(*object.matrixWorld);
    specs.push_back(std::move(spec));
});

threepp::conveyor::ConveyorPhysics belts(world, specs);  // colliders + substep hooks
belts.speedScale = 1.f;                                   // live speed control
// destroy `belts` before `world`; it unregisters and releases what it built
```

The generated meshes are already in the document, so a consumer that only
renders needs nothing regenerated; one that wants to re-skin or re-tessellate
can call `ConveyorConfig::syncDerived()` — the same function the editor's sync
pass uses, so the content is identical either way. `EditorConveyor_test`
holds this whole path green: author → save → load → rebuild → a box conveys.

### Physics from a script

`threepp.editor.rigid_body_from_object` hands a script the body PhysX is
actually simulating, so it can push and steer it instead of fighting the
simulation by writing transforms that the next step overwrites:

```python
import threepp


class Hover:
    height = 3.0
    stiffness = 40.0

    def start(self, obj: threepp.Object3D):
        self.obj = obj
        self.body = threepp.editor.rigid_body_from_object(obj)

    def update(self, dt: float):
        if self.body is None:
            return
        # A spring toward `height`, damped by the body's own velocity.
        error = self.height - self.body.position.y
        lift = self.stiffness * error - 5.0 * self.body.velocity.y
        self.body.apply_force(threepp.Vector3(0, lift * self.body.mass, 0))
```

Read `velocity`, `angular_velocity`, `mass`, `position`, `rotation`,
`sleeping`, `is_static` and `is_kinematic`; write `velocity`,
`angular_velocity` and `mass`; call `apply_force`, `apply_impulse`,
`apply_torque`, `apply_torque_impulse`, `wake_up`, and — on a body authored
Kinematic — `set_kinematic_target(position, rotation=None)`, which sweeps the
body so it pushes dynamics on the way rather than teleporting through them.
Forces are per-step, not settings: call them every `update` while the force
should act. Asking a **static** body for velocity, mass or forces raises rather
than answering a zero.

Scripts run *after* physics each frame, so a read sees the step that just
happened and a force lands on the next one. Which also means a force applied
from `update` is applied once per *frame*, at whatever rate the window happens
to be running — put it in
[`fixed_update`](#the-physics-clock-fixed_update) instead and it is applied once
per substep, at a constant dt.

The body is also what
[`on_collision_enter` / `on_collision_exit`](#collisions-on_collision_enter--on_collision_exit)
and
[`on_trigger_enter` / `on_trigger_exit`](#trigger-volumes-on_trigger_enter--on_trigger_exit)
watch: same lookup, same object, and no handle to ask for — the session resolves
it at Play for any script that defines any of them. To ask about a body that
is *not* yours — what is under your feet, what is in front of you — cast a ray:
[`threepp.editor.raycast`](#raycasts-threeppeditorraycast).

Soft bodies get `threepp.editor.soft_body_from_object`, which is deliberately
thinner — PhysX drives a deformable volume through per-vertex GPU buffers, so
there is no per-actor "push this" to expose. What it does answer is where the
thing *is*: `center`, `bounds_min`, `bounds_max`, `vertex_count`, and the
`recompute_normals` toggle. That matters more than it sounds, because a soft
body's object sits at the origin for the whole of Play — the mesh carries
world-space vertices — so `obj.position` is useless for following it and
`body.center` is the answer.

Robots being simulated (see [Robots](#robots-urdf-in-userdata)) get
`threepp.editor.articulation_from_object` — the joint-space face of the same
idea, and the seam a policy-playback script stands on. Read `joint_names`,
`num_dof`, `joint_positions` and `joint_velocities` (radians/metres, in the
articulation's own DOF order — fixed URDF joints are collapsed, so match by
name rather than assuming the inspector's slider order), plus the root link's
world `root_position` / `root_rotation` / `root_velocity` /
`root_angular_velocity`. Command it with `set_drive_targets(values)` (one per
DOF) or `set_drive_target(joint, value)` by name or index. Targets are PD
setpoints for the drive authored in the robot's Articulation section — the
joint is pulled there over the coming steps, not teleported, and with zero
authored stiffness the drive is off and targets are inert. Do **not** call
`robot.set_joint_value` from a script while the robot simulates: the mirror
writes the solved pose back every frame, so the simulation overwrites you —
the drive target is the steering wheel.

```python
import threepp


class Waver:
    amplitude = 0.6
    period = 4.0

    def start(self, obj: threepp.Robot):
        self.art = threepp.editor.articulation_from_object(obj)
        self.t = 0.0

    def update(self, dt: float):
        if self.art is None:
            return
        self.t += dt
        import math
        target = self.amplitude * math.sin(2 * math.pi * self.t / self.period)
        self.art.set_drive_targets([target] * self.art.num_dof)
```

The handle resolves from the robot *or any node inside it* — a script on a
gripper link gets the whole robot's joint table, which is what a controller
wants.

Three things separate these from `spline_from_object`:

* **They only exist during Play.** A spline is authoring data and reads back any
  time; a body is not created until the physics session builds it, so both
  functions return `None` outside Play (and for an object with no physics).
* **A handle belongs to the play session that made it.** Stop releases every
  actor, so a handle kept across a stop raises instead of reading freed memory —
  check `body.valid`, or just ask again in `start()`. A script's own `stop()`
  is *inside* the session: sessions stop in reverse registration order, physics
  last, so the body is still live there — parking a robot or logging a final
  pose from `stop()` is fine. What the check guards is a handle kept *beyond*
  the session, stashed somewhere the next Play can see.
* **They need the PhysX SDK.** Without it the names are absent from
  `threepp.editor` rather than present and always failing.

The lookup walks *up* the scene graph (like `PhysxWorld::findActor`), so a
script on a child of a physics object still finds the body governing it. It
resolves against the play session's own record of what it created, not
`PhysxWorld`'s binding list — a static body is never bound, since it has no
pose to write back, and would otherwise be invisible to a script.

### Spawning objects during Play

The play scene is an ordinary threepp graph, so a script builds and parents
objects into it with the ordinary calls — `scene.add(mesh)`, `remove`, `clear`.
There is no editor verb for this and none is needed. Stop restores the scene
from the snapshot it took at Play, so whatever a script spawned simply is not
there afterwards; nothing has to be cleaned up.

`threepp.editor.world()` is the other half — the `PhysxWorld` the session is
stepping, so a spawned mesh gets a body the same way a standalone threepp
program would give it one:

```python
import threepp

editor = threepp.editor


class Spawner:
    def start(self, obj):
        self.left = 5

    def update(self, dt):
        if self.left <= 0:
            return
        self.left -= 1
        crate = threepp.Mesh(threepp.BoxGeometry(0.6, 0.6, 0.6),
                             threepp.MeshStandardMaterial())
        crate.position.set(0.0, 6.0, 0.0)
        editor.scene().add(crate)     # it renders
        editor.world().add(crate)     # ...and it falls
```

Do not confuse this with [`editor.add()`](#generators-in-userdata), which is a
*generator* verb: that one writes into the **document**, as one undoable step,
and refuses during Play precisely because Play must not touch the document.
`scene().add()` writes into the **play scene**, which is transient by
construction. Different targets, both correct.

Two limits worth knowing before leaning on this:

* **The play sessions collect their objects once, at `start()`.** A spawned
  object gets no authored physics, no sensors, and no script of its own — even
  if you spawn it carrying `scriptSource` in its `userData`. `world().add()`
  works because it goes straight to the world rather than through the authored
  `PhysicsConfig` sweep. If a spawned thing needs behaviour, the spawner drives
  it.
* **`world().add()` hands back a raw `threepp.RigidBody`**, not the
  lifetime-checked `threepp.editor.RigidBody` that `rigid_body_from_object`
  returns. Within the session — `update()`, the callbacks, and `stop()`, which
  runs before the world goes down — it is fine. But it is *not* invalidated
  when the world dies, so one stashed beyond its session dereferences a
  released actor where the checked handle would raise. For anything held
  longer than the session, prefer the checked handle. This is the wheel's own
  contract, inherited: a `threepp.RigidBody` is valid while its world is alive.

`threepp.PhysxWorld` is visible in the editor but **cannot be constructed**
there — its constructor raises, and says to use `editor.world()` instead. The
editor plays exactly one world; a second would stand up a second PhysX
foundation beside it. Every other method on the class is the one the wheel
binds, because it is literally the same translation unit.

### Debug draw

Every controller script computes geometry nobody can see — the altimeter ray,
the contact normal, the drive target — and its only instrument used to be
`print()`. `threepp.editor.draw_*` puts those vectors in the viewport, over the
scene, for one frame:

| call | draws |
| --- | --- |
| `draw_line(a, b, color=None)` | a world-space segment |
| `draw_ray(origin, direction, length=1.0, color=None)` | `origin + direction*length` — pass `hit.distance` to see exactly the ray that hit |
| `draw_point(point, size=0.25, color=None)` | a small axis-aligned cross |
| `draw_box(center, size, color=None)` | the 12 edges of an AABB (`size` = full extents) |
| `draw_sphere(center, radius=1.0, color=None)` | three great circles |
| `draw_axes(object, size=1.0)` | the object's world frame, X red / Y green / Z blue |

`color` is a hex int (`0xff8800`) or a `threepp.Color`; default white.

```python
def update(self, dt):
    hit = editor.raycast(self.body.position, DOWN, self.probe, ignore=self.obj)
    if hit is not None:
        editor.draw_ray(self.body.position, DOWN, hit.distance, 0x25d6f0)
        editor.draw_point(hit.point, 0.3, 0xff9430)
    editor.draw_axes(self.obj)
```

The rules, all of one piece:

* **Immediate mode.** A draw lasts the frame it was made in; a line that should
  persist is redrawn every `update()` — which is exactly when a script is called
  anyway. Nothing to clean up, nothing to leak. (Pause keeps the last picture on
  screen rather than blanking it.)
* **No-op outside Play**, same reasoning as `is_key_down` answering `False`
  headless: a script that draws must still *run* in a headless pass, just
  unseen. Nothing raises.
* **Drawn on top.** Depth test off — the whole point is seeing the ray that
  ends inside a mesh. Everything decomposes to segments into one overlay
  `LineSegments`, so a thousand calls are still one draw.
* **Furniture, not scene.** Never saved, never picked, and hidden during every
  sensor scan — a lidar cannot range against your debug arrow. Screenshots
  taken with `--screenshot` *do* show it, which is the point: a review pass can
  see what the controller was thinking, not just where the body ended up.
* **Capped** (100 000 segments a frame), with one console line when a script
  blows past it — a per-body-per-substep loop gone wrong must not freeze the
  editor. Draw from `update()`, not `fixed_update()`, unless you want a
  segment per substep.

### Sensors from a script

The physics handles above read *ground truth*: `articulation_from_object` hands
back the exact joint angle the solver holds, to the last bit. A controller
written against that is a controller that has never met a sensor. The sensor
handles are the other half — the **noisy, seeded, rate-gated** numbers the
[authored sensors](#joint-sensors) actually produced, so a script can close its
loop on what the robot would really measure:

```python
import threepp


class ElbowHold:
    target = 0.6
    gain = 0.5

    def start(self, obj: threepp.Robot):
        self.art = threepp.editor.articulation_from_object(obj)
        self.enc = threepp.editor.encoder_from_object(obj, joint="elbow")

    def update(self, dt: float):
        if self.enc is None:
            return
        reading = self.enc.latest()
        if reading is None:
            return  # no measurement yet this Play
        # Closed on the MEASURED position - quantized to whole encoder ticks and
        # noise-corrupted - not on the joint angle the solver knows.
        command = reading.position + self.gain * (self.target - reading.position)
        self.art.set_drive_target(self.enc.joint, command)
```

Written in `update` this loop closes once per drawn frame, at whatever rate the
window happens to run. Rename the method to
[`fixed_update`](#the-physics-clock-fixed_update) — nothing else about the
script changes — and it closes once per physics substep instead, at a constant
dt, which is the clock the measurements themselves are stamped on and the one a
deployed controller runs on. What it reads is still the batch drained at the end
of the last frame; what changes is that the command goes out at a rate the frame
rate cannot move.

Four lookups, one per proprioceptive sensor type:

| function | returns | reading fields |
| --- | --- | --- |
| `imu_from_object(obj)` | `Imu` | `time`, `angular_velocity`, `acceleration` |
| `encoder_from_object(obj, joint=None)` | `Encoder` | `time`, `position`, `velocity` |
| `force_torque_from_object(obj)` | `ForceTorque` | `time`, `force`, `torque` |
| `contact_from_object(obj)` | `Contact` | `time`, `touching`, `force` |

The IMU's `acceleration` is **specific force**, so a level sensor at rest reads
`+9.81` on its up axis and one in free fall reads ~0. The contact sensor's
`touching` is the *latch*: it stays `True` while resting on something, including
after PhysX puts the pair to sleep and `force` goes quiet — which is the channel
a foot-down check wants, and the reason there are two.

Every handle reads the same two ways:

* `latest()` — the newest measurement, or `None` before the first one. Does not
  move the handle's read cursor, so polling it in a control loop is free.
* `read_new()` — every measurement since *this handle* last read, oldest first,
  and advances its cursor. A fresh handle starts empty: it reports what arrives
  from now on, never a backlog you did not ask for. Each handle carries its own
  cursor, so two scripts reading one sensor never steal samples from each other,
  and neither starves the Sensors panel's plots or the CSV recording. Falling
  more than 256 readings behind loses the oldest.

That last point is the design constraint the whole thing is built around.
`drain()` *empties* a sensor's ring, so the session is the only party allowed to
call it — a second drainer would leave the panel plotting half the data. Handles
therefore read the session's own retained copies, never a live sensor and never
PhysX state. Which is also what makes them safe at Stop: physics stops *first*,
so anything reaching into a sensor from a script's `stop()` would be reading
through a torn-down SDK.

**The encoder fans out.** An encoder authored for `All joints` becomes one live
encoder *per DOF* (see [Joint sensors](#joint-sensors)), so
`encoder_from_object(obj)` with no `joint` raises when more than one answers
rather than picking an arbitrary one — pass `joint="elbow"`, or take them all:

```python
for enc in threepp.editor.encoders_from_object(robot):
    print(enc.joint, enc.latest().position)
```

Naming a joint no encoder measures raises too, listing the ones that do: a typo
answering `None` is indistinguishable from "Play is not running", and that costs
an afternoon.

Otherwise the contract is the physics handles' contract, for the same reasons:

* **They only exist during Play.** An authored `userData["sensor"]` is not a
  sensor until the session builds one, so every lookup returns `None` outside
  Play, and for an object carrying no sensor of that kind.
* **A handle belongs to the play session that made it.** Stop drops every
  sensor, so a handle kept across a stop raises `RuntimeError` instead of
  reading freed memory — check `.valid`, or just ask again in `start()`.
* **They need the PhysX SDK.** Without it the names are absent from
  `threepp.editor` rather than present and always answering `None`.
* **The lookup walks *up* the scene graph**, so a script on a child of an
  instrumented link still finds the sensor measuring it. The nearest ancestor
  carrying one wins.

Vision sensors (depth, lidar) are deliberately **not** here: a scan is tens of
thousands of points, which wants a buffer protocol rather than a per-sample
handle. Read those from the overlay, the panel, or a CSV recording for now.

A vision sensor's **Near** and **Far** bound the *range* it can report — a blind
sphere of radius Near, out to a maximum range of Far — and mean exactly the same
thing on GL as on Vulkan, inclusive at both ends. They are not view-space clip
planes: an object off to the side is judged by how far away it is, not by its
depth along some axis, so a Near large enough to hide the machine the sensor
rides on has to clear that machine's *radius*. (The shipped Hover Arena uses
1.6 m for exactly this reason: the drone's rotor rings reach 1.46 m from the
lidar.) `SensorBackendParity_test` scans one scene on both backends and pins the
two to the same answer.

Sensors are rebuilt from the authored seed on every Play, so a script that
reacts to noise reacts to the *same* noise on the next run — a closed loop under
test is reproducible, which is the entire point of the seed.

A vision scan is **fired on one frame and delivered on a later one**. On a
raster backend the two are the same frame (the scan *is* six blocking
framebuffer reads); on Vulkan the beams are traced against the renderer's
acceleration structure and collected by a later frame's fence poll — never a
wait. That is not a refinement, it is the difference between a smooth frame and
a periodic stall: taking delivery means waiting on a GPU fence, and that fence
sits behind every frame already queued. Measured on an RTX 4070 with two frames
in flight, collecting a 1.2 ms VLP-16 trace immediately cost **28 ms**, and the
Hover Arena's 10 Hz lidar delivered that ten times a second — a 34 ms frame in
an otherwise 15 ms scene. So the cloud a panel or the overlay reads is one frame
old, at the pose the beams were fired from, and a sensor with a scan still owed
does not fire again. Which frame a scan lands on is a property of the machine;
what the seed guarantees — the same cloud, from the same pose, on every run — is
unchanged.

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
creates a static/dynamic/kinematic `PxRigidActor` with the requested shape (or a
soft body's deformable volume), and lets `PhysxWorld` write the poses back into
the scene graph each step. It is
header-only and compiled only when the PhysX SDK is found (vcpkg feature
`physx`); without it the editor still authors and saves physics settings, and
Play simply runs with no physics session. The sensor session is split along the
same line: `SensorPlaySession` is PhysX-free and runs the vision sensors (a
depth or lidar scan needs a renderer, not a physics world) in every build, while
`PhysxSensorPlaySession` — the subclass a PhysX build constructs instead — adds
the live IMU/contact/joint sensors. In a build without the SDK a body or joint
sensor is authored, counted, and carries a status naming the missing build.

Three sessions ship, and they start in this order: physics, then the animation
player, then scripts. Scripts run last on purpose — a script's transform edits
are the final word for the frame, after the simulation and the mixer have had
their say.

---

## Playing a document without the editor

[`threepp_player`](player.md) is the second front end over this same core: it
loads a `scene.json` and plays it with none of the editing machinery, for policy
validation and data generation.

```
threepp_player arena.json --headless --episodes=100 --seconds=20 --record=runs
```

It registers the same four sessions in the same order, runs each episode as a
full play/stop cycle (so episodes are independent), records what the sensors
measured, and exits nonzero if any script raised or the document would not play
— which is what makes a scene something CI can gate on. See
[doc/player.md](player.md).

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
  find-and-replace, and it edits inline source only. It does hold several
  scripts open at once, one tab each, because that costs nothing but a vector
  and the alternative — one buffer retargeting itself onto the selection — makes
  writing two scripts that talk to each other a chore. The one thing it adds
  over Notepad is the compile check; for anything more, `Edit in VS Code` hands
  the same source to an editor that has all of it.
* **External editing syncs one way.** The scratch file is the source of truth
  while a session is live: an undo in the editor moves the document out from
  under it without touching the file, and the next save puts the file's version
  back. Undo what you did in VS Code, or stop the session before undoing here.
  The stubs are also a superset of what a script can reach — they come from the
  full wheel module, while the script host binds ten areas of it — so a name
  that completes is not proof that it exists at Play.
* **Generators have no parameters yet.** `userData["generatorFields"]` round-trips,
  but the values are not injected into the run and the inspector shows no inputs
  for them, so changing a count means editing the line. This is a design gap, not
  missing plumbing: a script's own module-level `count = 400` would overwrite any
  value injected as a global, so making the field authoritative needs a convention
  (read-with-default, or rewriting the assignment) rather than a wire-up.
* **Generators are one-shot, and only from the UI.** A run appends nothing and
  replaces everything: re-running rebuilds the whole output, so there is no
  incremental or partial regeneration, and no *general* headless entry point —
  batch generation over many seeds is not a one-liner yet.
  (`apps/editor/tools/HoverArenaAuthor.cpp` is one for exactly one scene, and it
  is the shape a general one would take: set the sink, run the source, attach on
  success.) There is also no bake step,
  so keeping generated content while dropping its rule means clearing the script
  and accepting that Clear takes the output with it.
* **Splines are Catmull-Rom only.** One curve type, parameterised three ways —
  no Bézier handles, no per-point tangents or twist, and no closed-form
  reordering of control points beyond dragging them in the hierarchy. A spline
  is also not a path anything follows by itself: it is data, and a script (or
  your own `PlaySession`) is what walks it.
* **Generated geometry is derived, so removing it is destructive.** Setting
  **Mesh** back to None deletes the node; undoing that config edit re-derives a
  *new* mesh rather than restoring the old one, and the material, physics and
  name you had put on it are gone with the node. Undo covers the config edit,
  which is all it claims to. Nothing else loses that state — dragging points,
  changing the radius, reloading the document all rebuild through the same node.
  There is also one generated mesh per spline, and one material with it.
* **Duplicate shares geometry.** `Object3D::clone()` shares both geometry and
  materials; the editor clones the materials afterwards so recolouring a copy
  does not recolour the original, but geometry stays shared. That is intentional
  (duplication stays cheap), and it means a future mesh-editing feature would
  need an explicit "make unique" step.