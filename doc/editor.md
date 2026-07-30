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
| `PhysicsConfig` | per-object rigid-body and soft-body authoring, stored in `userData` |
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

**Splines.** Add ▸ Spline drops a curve whose control points are ordinary scene
nodes — drag them with the gizmo, insert and delete them, and the sampled curve
redraws live. See [Splines in `userData`](#splines-in-userdata).

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
body=dynamic;shape=convex;mass=12.5;friction=0.8;restitution=0.35;young=1000000;poisson=0.45;voxel=10;iterations=20;selfcollision=0;hulls=16;hullverts=64;voxels=100000
```

`PhysicsConfig::encode()` / `decode()` own that format. Unknown keys are ignored
on read, so a document written by a newer editor still loads — and a document
written by an *older* one loads too, since a missing key keeps its default. A
disabled body removes the entry entirely rather than writing `enabled=false`, so
turning physics off leaves no trace in the file. Every key is written whatever
the body type, so switching a body from `soft` to `dynamic` and back does not
quietly discard the settings the other type was using.

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

The cooked hulls are drawn by the **Physics Colliders** overlay (PhysX's own
collision-shape visualization), so the quality of a decomposition is visible in
the viewport, not just inferred from behaviour.

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
cares about: sessions stop physics-first, so by the time the sensor session stops
the PhysX SDK is already gone. A Force/Torque sensor holds a `PxArticulationCache`
whose memory went with it, so the session **abandons** that cache (drops the
pointer without releasing it) on the normal Stop rather than releasing it against
a freed allocator; the SDK teardown already reclaimed the buffer. On a mid-play
teardown, where the world is still alive, it unregisters cleanly instead.

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
Script Editor window. A file is shared between objects and scenes and editable
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

Every method is optional; missing ones are skipped. There are three more:
`fixed_update(self, dt)`, which runs on the *physics* clock rather than on the
frame (see [The physics clock](#the-physics-clock-fixed_update)), and
`on_collision_enter(self, contact)` / `on_collision_exit(self, contact)`, which
run when the body governing the object starts and stops touching another (see
[Collisions](#collisions-on_collision_enter--on_collision_exit)). In a file, the
class is the one whose name matches it (`spinner.py` → `Spinner`,
case-insensitively), or — failing that — the single class in the file that
defines `update()`. Anything else is reported rather than guessed at.

`start` may also ask for the scene — `def start(self, obj, scene):` — and the
editor passes it only when the signature does (`*args` counts as asking), so the
one-argument form keeps working untouched. The scene handle is the ordinary
`threepp.Scene`: `scene.get_object_by_name("Ground")`, `scene.children`, and so
on. The same lifetime rule as every handle applies — resolve neighbours in
`start`, use them during the session, never stash them across Play sessions.
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

Out of scope, deliberately: **trigger volumes** (an overlap that does not
collide), **`on_collision_stay`** (a resting contact stops being re-reported the
moment PhysX puts the pair to sleep, so a per-frame "still touching" callback
would lie), **per-contact-point lists** beyond the single representative point
above, **soft bodies** (a deformable volume has no rigid actor to watch), and
**watching a robot** — a simulated robot's links are the articulation's, not the
session's actor registry's, so a script on a robot resolves no body of its own
and gets the no-body line. (Being *hit* by one is fine: a contact against a link
names the robot, the same way a raycast does.)

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
* **Call it from anywhere a script runs**: `update`, `fixed_update`,
  `on_collision_enter` / `on_collision_exit`. All of those run on the editor
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
    def start(self, obj: threepp.Object3D, scene: threepp.Scene):
        # Resolve neighbours here, once: every instance exists by now.
        self.door = threepp.editor.script_from_object(scene.get_object_by_name("Door"))

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
  the collision callbacks, and `stop`, where every instance is still alive. It
  reads the session's own instance list and touches neither the scene nor the
  physics world, so it needs no PhysX build: unlike the handles above it, this
  is there in every build that has scripts at all.

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
and the window shows the source read-only with a banner and a **Stop external
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
keys and silently losing save-while-playing would be its own surprise. A scene whose scripts
never ask keeps every shortcut.

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

    def start(self, obj: threepp.Object3D, scene: threepp.Scene):
        self.obj = obj
        self.u = 0.0
        self.path = threepp.editor.spline_from_object(
            scene.get_object_by_name(self.spline_name))

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
watch: same lookup, same object, and no handle to ask for — the session resolves
it at Play for any script that defines either method. To ask about a body that
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
  check `body.valid`, or just ask again in `start()`. This is reachable in
  normal use: sessions stop in registration order, physics first, so a script's
  own `stop()` runs when the actors are already gone.
* **They need the PhysX SDK.** Without it the names are absent from
  `threepp.editor` rather than present and always failing.

The lookup walks *up* the scene graph (like `PhysxWorld::findActor`), so a
script on a child of a physics object still finds the body governing it. It
resolves against the play session's own record of what it created, not
`PhysxWorld`'s binding list — a static body is never bound, since it has no
pose to write back, and would otherwise be invisible to a script.

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

Sensors are rebuilt from the authored seed on every Play, so a script that
reacts to noise reacts to the *same* noise on the next run — a closed loop under
test is reproducible, which is the entire point of the seed.

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
  find-and-replace, and it edits inline source only. The one thing it adds over
  Notepad is the compile check — for anything more, `Edit in VS Code` hands the
  same source to an editor that has all of it.
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
  incremental or partial regeneration, and no headless entry point — batch
  generation over many seeds is not a one-liner yet. There is also no bake step,
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