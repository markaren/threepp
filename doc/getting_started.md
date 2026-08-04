# Getting started with threepp

This is a **concepts** guide. It explains the handful of ideas that everything in
`threepp` is built from, so that the [examples](../examples) folder reads as variations on a
theme rather than 200 unrelated programs.

If you already know [three.js](https://threejs.org/), skim [The five nouns](#the-five-nouns),
read [Ownership and lifetimes](#ownership-and-lifetimes) carefully (this is where C++ differs
most), then jump to [Beyond three.js](#beyond-threejs).

**Contents**

1. [What threepp is](#what-threepp-is)
2. [Your first program](#your-first-program)
3. [The five nouns](#the-five-nouns)
4. [The scene graph](#the-scene-graph)
5. [Ownership and lifetimes](#ownership-and-lifetimes)
6. [Meshes: geometry + material](#meshes-geometry--material)
7. [Textures and color space](#textures-and-color-space)
8. [Lights, shadows and environments](#lights-shadows-and-environments)
9. [Cameras and resizing](#cameras-and-resizing)
10. [The frame loop](#the-frame-loop)
11. [Input and controls](#input-and-controls)
12. [Loading models](#loading-models)
13. [Animation](#animation)
14. [Picking with the Raycaster](#picking-with-the-raycaster)
15. [Choosing a backend](#choosing-a-backend)
16. [Rendering off-screen and headless](#rendering-off-screen-and-headless)
17. [Beyond three.js](#beyond-threejs)
18. [Gotchas](#gotchas)
19. [Where to go next](#where-to-go-next)

---

## What threepp is

`threepp` is a C++20 3D library that ports the **high-level API of three.js** (roughly
[r129](https://github.com/mrdoob/three.js/tree/r129), with newer revisions mixed in where it
mattered) onto two native backends:

| | |
|---|---|
| **`GLRenderer`** | OpenGL 3.3 raster. The portable baseline — Windows, Linux, macOS, MinGW, and Emscripten/WebGL2. A mechanical port of three.js's WebGLRenderer, so its behaviour matches three.js closely. |
| **`VulkanRenderer`** | A deferred renderer: raster G-buffer with ray-traced shadows, ambient occlusion, GI and reflections, denoised, with TAA. Opt-in (`-DTHREEPP_WITH_VULKAN=ON`), evolves fast. |

The important structural fact: **the scene graph is backend-neutral.** You build a `Scene` of
`Object3D`s once, and hand it to either renderer. Nothing in your scene code knows which one it
got.

> **Scope.** `threepp` targets research, prototyping and education — not production
> hardening. APIs and (especially Vulkan) behaviour change. Pin a tag if you need
> reproducibility.

## Your first program

Consume `threepp` with CMake `FetchContent` — the recipe, along with Conan, xmake and the
build flags, lives in the [README](../README.md#consuming-threepp). This guide assumes you
have a target linking `threepp::threepp`.

The smallest complete program — a spinning cube:

```cpp
#include "threepp/threepp.hpp"

using namespace threepp;

int main() {

    Canvas canvas("Hello", {{"aa", 4}});
    GLRenderer renderer(canvas);

    Scene scene;
    scene.background = Color::aliceblue;

    PerspectiveCamera camera(75, canvas.aspect(), 0.1f, 100.f);
    camera.position.z = 5;

    OrbitControls controls(camera, canvas);

    scene.add(HemisphereLight::create());

    auto box = Mesh::create(
            BoxGeometry::create(),
            MeshPhongMaterial::create(MeshPhongMaterial::Params{}.color(Color::green)));
    scene.add(box);

    canvas.onWindowResize([&](WindowSize size) {
        camera.aspect = size.aspect();
        camera.updateProjectionMatrix();
        renderer.setSize(size);
    });

    Clock clock;
    canvas.animate([&] {
        box->rotation.y += 1.f * clock.getDelta();
        renderer.render(scene, camera);
    });
}
```

`"threepp/threepp.hpp"` is a convenience umbrella header pulling in the common types (canvas,
scene, cameras, geometries, materials, lights, controls, loaders, helpers, `GLRenderer`).
Anything outside that set — `AnimationMixer`, `Raycaster`, `VulkanRenderer`, `GLTFLoader`,
everything under `extras/` — you include explicitly.

## The five nouns

Almost every `threepp` program is these five things wired together:

```
    Canvas ──────── window, GL/Vulkan context, input events, the frame loop
      │
      ├── Renderer ── consumes (Scene, Camera) → pixels
      │
      ├── Scene ───── the root Object3D; holds background, environment, fog
      │     └── Object3D tree ── Mesh / Light / Group / Points / Line / …
      │
      └── Camera ──── an Object3D that also carries a projection matrix
```

- **`Canvas`** owns the OS window and the graphics context, dispatches keyboard/mouse events,
  and drives the frame loop. It is a `PeripheralsEventSource`.
- **`Renderer`** is an abstract interface (`renderers/Renderer.hpp`). `GLRenderer` and
  `VulkanRenderer` implement it. Everything a portable program needs — `render`, `setSize`,
  `setClearColor`, `shadowMap()`, `toneMapping`, `setRenderTarget`, `readRGBPixels` — lives on
  the base.
- **`Scene`** is just an `Object3D` with three extra fields: `background` (color, texture or
  cube texture), `environment` (an image-based lighting map), and `fog`.
- **`Camera`** is an `Object3D` too, which is why you position it with `camera.position` and
  aim it with `camera.lookAt(...)`. `PerspectiveCamera` and `OrthographicCamera` add the
  projection.
- **`Object3D`** is the base of everything that lives in the tree.

## The scene graph

`Object3D` gives every node a local transform, a parent, and children:

```cpp
Object3D::position    // Vector3, local
Object3D::rotation    // Euler, radians — kept in sync with quaternion
Object3D::quaternion  // Quaternion, local
Object3D::scale       // Vector3, local
```

`position`/`rotation`/`scale` are the ones you write. Each frame the renderer composes them
into `matrix` (local) and then multiplies down the tree into `matrixWorld` (world). You rarely
touch either directly.

### Coordinates, units and angles

`threepp` uses the three.js conventions: **Y is up** (`Object3D::defaultUp` is `(0, 1, 0)`),
the coordinate system is **right-handed**, and a camera looks down its **local −Z** axis. All
angles are **radians** — `math::degToRad(45)` converts. There is no enforced length unit, but
in practice everything (examples, physics, loaders) treats 1 unit = 1 meter.

Coming from robotics or CAD, where Z-up is common: loaders for formats that declare an
up-axis (Collada, USD) convert on import, and `URDFLoader` owns the frame handling for the
meshes a robot description references — your Z-up asset arrives Y-up in the scene.

Transforms compose down the tree, so grouping is how you build articulated things:

```cpp
auto arm = Group::create();
arm->position.set(0, 1, 0);

auto forearm = Group::create();
forearm->position.set(0, 0.5f, 0);   // relative to arm
arm->add(forearm);

scene.add(arm);
arm->rotation.z = math::degToRad(30); // forearm follows
```

Useful members you will reach for constantly:

| Member | Meaning |
|---|---|
| `visible` | Skip this node **and its subtree** when rendering. |
| `castShadow` / `receiveShadow` | Shadow map participation (off by default). |
| `frustumCulled` | Per-frame frustum test. On by default; turn off for objects whose bounds lie about their real extent. |
| `renderOrder` | Manual sort key; negative pushes behind everything (skyboxes, backdrops). |
| `layers` | Bitmask. An object renders only if it shares a layer with the camera; `Raycaster` honours it too. |
| `userData` | `unordered_map<string, std::any>` for your own per-node data. |
| `name` | Free-form label; the key for `getObjectByName`. |

And the traversal API:

```cpp
model->traverse([](Object3D& o) { /* every node, including model */ });
model->traverseVisible([](Object3D& o) { /* skips invisible subtrees */ });

// Type-filtered — the idiomatic way to touch every mesh of a loaded model:
model->traverseType<Mesh>([](Mesh& m) {
    m.castShadow = true;
    m.receiveShadow = true;
});

// Name lookup, optionally type-constrained:
auto* wheel = model->getObjectByName<Mesh>("wheel_fl");
```

For downcasts, `Object3D` carries `as<T>()`, `is<T>()`, and the null-safe
`materialAs<T>()` shorthand:

```cpp
if (auto* m = node.materialAs<MeshStandardMaterial>()) m->roughness = 0.4f;
```

### Matrix updates

Reading a world position **after** you changed a transform but **before** the renderer ran needs
an explicit update:

```cpp
obj->position.set(1, 2, 3);
obj->updateMatrixWorld();          // this node and descendants
Vector3 world;
obj->getWorldPosition(world);
```

`Scene::autoUpdate` (default `true`) is what makes the renderer call `updateMatrixWorld()` for
you each frame. Setting `matrixAutoUpdate = false` on a node means *you* own its `matrix` — the
escape hatch for nodes driven by an external source (physics, another node's `matrixWorld`).

## Ownership and lifetimes

This is the part with no three.js equivalent, and the part worth reading twice.

**Math types are values. Everything else is a `shared_ptr`.**

```cpp
Vector3 v{1, 2, 3};       // value — copy it, pass it, store it
Color c = Color::red;     // value
Matrix4 m;                // value

auto geo = BoxGeometry::create();          // shared_ptr<BoxGeometry>
auto mat = MeshStandardMaterial::create(); // shared_ptr<MeshStandardMaterial>
auto mesh = Mesh::create(geo, mat);        // shared_ptr<Mesh>
```

Every non-value type has a static `::create(...)` returning a `std::shared_ptr`. Geometries,
materials and textures **dispose their GPU resources automatically** when the last reference
dies — there is no manual cleanup step.

Because they are shared, sharing is cheap and intended:

```cpp
auto shared = MeshStandardMaterial::create();
for (int i = 0; i < 100; ++i) {
    auto m = Mesh::create(geo, shared);   // one geometry, one material, 100 meshes
    m->position.x = static_cast<float>(i);
    scene.add(m);
}
```

### Two ways to attach a child

```cpp
parent->add(child);      // shared_ptr overload — parent takes SHARED ownership
parent->addRef(*child);  // reference overload — parent takes NO ownership
```

`add()` is what you want almost always. `addRef()` exists for objects whose lifetime you
manage yourself (a stack object, a member of your own class) — and it makes **you** responsible
for outliving the parent.

Detaching mirrors that split:

```cpp
parent->remove(*child);              // drops the parent's reference — may destroy child
auto kept = child->removeFromParent(); // returns the owning ref; keep it to keep child alive
```

`removeFromParent()` returning `nullptr` means the parent never owned it (`addRef`) or there was
no parent. **Discarding the return value on an owned child destroys it** — that return value is
the only thing keeping it alive.

`children` is a `std::vector<Object3D*>` of raw pointers (owning references live in a private
parallel vector). Traversal therefore never touches a refcount — but a raw `Object3D*` you cache
is only valid as long as something owns the node.

### Stack or heap?

Both idioms appear in the examples and both are correct:

```cpp
Scene scene;                                  // stack — fine for the root
PerspectiveCamera camera(75, aspect, .1f, 100.f);
OrbitControls controls(camera, canvas);

auto scene = Scene::create();                 // heap — needed if it must outlive the frame
auto camera = PerspectiveCamera::create(...);
```

Root objects (scene, camera, controls) are usually stack locals in `main` — they live as long as
the program. Anything you add *into* the graph should be a `shared_ptr` via `::create`, since the
graph itself holds references.

## Meshes: geometry + material

A drawable is always **geometry (shape) + material (appearance)**, wrapped in an object type
that says *how* to draw the vertices:

| Object | Draws as |
|---|---|
| `Mesh` | Triangles |
| `Line`, `LineSegments`, `LineLoop` | Line primitives |
| `Points` | Point sprites |
| `Sprite` | A camera-facing quad |
| `InstancedMesh` | One geometry, N transforms, one draw call |
| `SkinnedMesh` | Triangles deformed by a `Skeleton` of `Bone`s |
| `Text`, `TextSprite` | Glyphs (from a `Font`, or as a screen-space sprite) |
| `LOD` | Picks a child by camera distance |

### Geometry

`BufferGeometry` is a bag of named vertex attributes (`position`, `normal`, `uv`, `color`, …)
plus an optional index buffer. The built-in generators in `geometries/` (`BoxGeometry`,
`SphereGeometry`, `PlaneGeometry`, `TubeGeometry`, `ExtrudeGeometry`, `TextGeometry`, …) are
just convenience subclasses that fill one in.

To build one by hand:

```cpp
auto geo = BufferGeometry::create();
geo->setAttribute("position", FloatBufferAttribute::create(positions, 3));
geo->setAttribute("normal",   FloatBufferAttribute::create(normals, 3));
geo->setIndex(std::move(indices));
geo->computeBoundingSphere();
```

To animate vertex data, mutate the attribute's array and mark it dirty
(`attribute->needsUpdate()`) — see [examples/geometries/dynamic.cpp](../examples/geometries/dynamic.cpp).

### Material

Materials are a family, ordered by cost and realism:

| Material | Lighting | Use for |
|---|---|---|
| `MeshBasicMaterial` | none (unlit) | Flat color, debug, UI, emissive-looking things |
| `MeshLambertMaterial` | per-vertex diffuse | Cheap matte surfaces |
| `MeshPhongMaterial` | per-pixel + specular | The cheap "shiny" classic |
| `MeshStandardMaterial` | physically based (metalness/roughness) | **The default choice.** Correct under environment maps. |
| `MeshPhysicalMaterial` | PBR + clearcoat, transmission, sheen | Glass, car paint, fabric |
| `MeshToonMaterial`, `MeshMatcapMaterial`, `MeshNormalMaterial`, `MeshDepthMaterial` | — | Stylised / diagnostic |
| `ShaderMaterial`, `RawShaderMaterial` | your GLSL | Custom effects |
| `LineBasicMaterial`, `PointsMaterial`, `SpriteMaterial`, `ShadowMaterial` | — | Non-triangle primitives |

Two ways to configure one. Prefer the typed `Params` builder — a typo is a compile error, the
fields autocomplete, and the setters chain in any order:

```cpp
auto mat = MeshStandardMaterial::create(
        MeshStandardMaterial::Params{}
                .color(0xff0000)
                .roughness(0.4f)
                .metalness(1.0f)
                .flatShading(true));
```

The stringly-typed map form still exists (it mirrors three.js and is what the Python bindings
use), but a typo there is only caught at runtime:

```cpp
auto mat = MeshStandardMaterial::create({{"color", Color::red}, {"roughness", 0.4f}});
```

Fields are plain public members afterwards, so you can also just assign:

```cpp
mat->color = Color::green;
mat->side = Side::Double;
mat->transparent = true;
mat->opacity = 0.5f;
```

Materials are composed from capability mixins (`MaterialWithColor`, `MaterialWithMap`,
`MaterialWithRoughness`, …). That is why `materialAs<MaterialWithColor>()` works across every
material that happens to have a color — handy for generic code.

Changing a *structural* property (adding a map, toggling `flatShading`, editing `defines`)
requires `mat->needsUpdate()` so the shader is recompiled. Changing a plain value (`color`,
`opacity`, `roughness`) does not.

### Instancing

When you need thousands of copies of one shape, `InstancedMesh` collapses them into a single
draw call:

```cpp
auto mesh = InstancedMesh::create(geo, mat, 10000);
Matrix4 m;
for (size_t i = 0; i < 10000; ++i) {
    m.setPosition(x(i), y(i), z(i));
    mesh->setMatrixAt(i, m);
    mesh->setColorAt(i, Color(0x00ff00));
}
mesh->instanceMatrix()->needsUpdate();
scene.add(mesh);
```

## Textures and color space

```cpp
TextureLoader loader;

auto albedo = loader.load("brick_color.png", ColorSpace::sRGB);  // color data
auto normal = loader.load("brick_normal.png");                   // NON-color data

auto mat = MeshStandardMaterial::create(
        MeshStandardMaterial::Params{}.map(albedo).normalMap(normal));
```

**The one rule that matters:** color maps (`map`, `emissiveMap`, and a cube/background texture)
are authored in sRGB and must be tagged `ColorSpace::sRGB` so the sampler decodes to linear.
Data maps (normal, roughness/metalness, AO, displacement) must **not** be — they are numbers,
not colors. Tagging these wrong is the most common cause of "my PBR material looks washed out /
too dark".

The default `load(path)` overload tags `NoColorSpace`, matching three.js.

Related loaders: `CubeTextureLoader` (six faces → skybox), `RGBELoader` (`.hdr` equirectangular
environments), `DDSLoader`, `ImageLoader`. `TextureLoader` caches by path by default.

## Lights, shadows and environments

```cpp
scene.add(AmbientLight::create(0x404040));                  // uniform fill, no direction
scene.add(HemisphereLight::create(0xffffff, 0x8d8d8d, 1.f)); // sky/ground gradient fill

auto sun = DirectionalLight::create(0xffffff, 3.f);         // parallel rays (sun)
sun->position.set(3, 10, 10);
sun->castShadow = true;
scene.add(sun);

scene.add(PointLight::create(0xffffff, 1.f));                // omni, distance falloff
scene.add(SpotLight::create(0xffffff, 1.f));                 // cone
scene.add(RectAreaLight::create(0xffffff, 1.f, 4.f, 2.f));   // soft panel (Standard/Physical only)
```

A `DirectionalLight`/`SpotLight` points at its **target** object, not at a rotation — move the
target to aim it (`light->setTarget(obj)`).

Shadows need three opt-ins, and forgetting one is the usual reason nothing appears:

```cpp
renderer.shadowMap().enabled = true;   // 1. renderer
light->castShadow = true;              // 2. the light
mesh->castShadow = true;               // 3. each caster
ground->receiveShadow = true;          //    and each receiver
```

### Environment maps

For `MeshStandardMaterial`/`MeshPhysicalMaterial`, an environment map is usually a bigger
visual win than adding lights — it supplies image-based ambient and specular:

```cpp
RGBELoader rgbe;
auto env = rgbe.load("puresky_2k.hdr");
scene.environment = env;   // lights everything
scene.background = env;    // and is visible behind it
```

Tone mapping turns the resulting HDR values into displayable ones:

```cpp
renderer.toneMapping = ToneMapping::ACESFilmic;
renderer.toneMappingExposure = 0.7f;
```

Fog is a scene property, applied by materials with `fog = true`:

```cpp
scene.fog = Fog(0x87ceeb, 20, 60);   // linear, near/far
scene.fog = FogExp2(0x87ceeb, 0.02f); // exponential-squared
```

## Cameras and resizing

```cpp
PerspectiveCamera camera(75 /*fov°*/, canvas.aspect(), 0.1f /*near*/, 100.f /*far*/);
OrthographicCamera ortho(left, right, top, bottom, near, far);
```

> Note the naming difference from three.js: the clip planes are **`nearPlane`** and
> **`farPlane`** (`near`/`far` are macros on some Windows headers).

> **`VulkanRenderer` and orthographic cameras.** An `OrthographicCamera` means one of
> two things to the deferred backend, and it cannot tell them apart on its own. By
> default a standalone `render(scene, orthoCam)` is the **2D/HUD** path: sprites, lines,
> points and meshes are drawn as flat unlit fills over the frame. If the ortho camera is
> a real 3D **view** — an editor's axis views, an isometric game camera — say so once:
> ```cpp
> renderer.setOrthographicSceneRendering(true);
> ```
> and the frame takes the same deferred path a perspective camera does (lights, shadows,
> GI, reflections, fog, tone mapping). Depth of field is skipped under it — a parallel
> projection has no lens. The HUD pattern (a perspective `render()`, then a second
> `render()` with an ortho camera over a HUD scene) is unaffected either way.
> `GLRenderer` never had the ambiguity and needs nothing.

Any change to a projection input (`fov`, `aspect`, `zoom`, `nearPlane`, `farPlane`) needs
`camera.updateProjectionMatrix()`. This is why every example has the same resize handler:

```cpp
canvas.onWindowResize([&](WindowSize size) {
    camera.aspect = size.aspect();
    camera.updateProjectionMatrix();
    renderer.setSize(size);
});
```

Pick `nearPlane` as large as you can tolerate. Depth precision is dominated by the near plane,
and a tiny near (`0.001`) on a large scene is the classic cause of z-fighting.

Since a camera is an `Object3D`, you can parent it — attach it to a vehicle and it inherits the
vehicle's motion for free.

## The frame loop

```cpp
Clock clock;
canvas.animate([&] {
    const float dt = clock.getDelta();   // seconds since last call
    // 1. update your state — always scale by dt, never by frame count
    // 2. renderer.render(scene, camera);
});
```

`Canvas::animate` runs until the window closes. Per iteration it calls your callback, then the
backend's frame-end hook (buffer swap / present), then polls events.

`renderer.render(scene, camera)` then does, in order: update world matrices (if
`scene.autoUpdate`), build the render list, frustum-cull, sort (opaque front-to-back,
transparent back-to-front, honouring `renderOrder`), render shadow maps, and draw.

For deferred work — anything you want to happen *later* rather than *now*, without threads:

```cpp
TaskManager tasks;
tasks.invokeLater([&] { doSomething(); }, 2.0 /* seconds */);

canvas.animate([&] {
    tasks.handleTasks();
    renderer.render(scene, camera);
});
```

If you need to own the loop yourself (embedding in another framework, or stepping frames from a
test), use the single-step form:

```cpp
while (canvas.animateOnce([&] { renderer.render(scene, camera); })) { /* returns false to quit */ }
```

One rule spans all of this: **the API is single-threaded by design.** Create, mutate and render
everything from the thread that owns the `Canvas` — in practice, the main thread. The one
sanctioned exception is `ModelLoader::loadAsync`, which builds the model on a worker thread and
hands it over when done. For your own background work, `TaskManager::invokeLater` is safe to
call from another thread and funnels the callback back into the loop.

## Input and controls

`Canvas` is a `PeripheralsEventSource`. Two styles, both valid:

```cpp
// Callback style — the canvas owns the lambda; nothing to keep alive.
canvas.onKeyPressed([&](KeyEvent evt) {
    if (evt.key == Key::SPACE) jump();
});

// Polling style — for continuous input like WASD.
canvas.animate([&] {
    if (canvas.isKeyDown(Key::W)) move(forward);
});

// Listener-object style — when you want to remove it later.
MouseMoveListener l([&](Vector2 pos) { mouse = pos; });
canvas.addMouseListener(l);   // NOTE: `l` must outlive the canvas usage
```

`canvas.onDrop(...)` gives you dropped file paths. `setIOCapture(...)` lets an overlay (ImGui)
swallow events before the scene sees them.

Built-in controls, all taking the event source as their last argument:

| Control | Constructor | Behaviour |
|---|---|---|
| `OrbitControls` | `(camera, eventSource)` | Orbit / pan / dolly around `controls.target`. The default. |
| `FlyControls` | `(object, eventSource)` | First-person WASD + mouse-look. Drives *any* `Object3D`, not just a camera. |
| `DragControls` | `(objects, camera, eventSource)` | Drag the given objects with the mouse. |
| `TransformControls` | `(camera, eventSource)` | An in-scene translate/rotate/scale gizmo. |

`OrbitControls` only needs `update()` in your loop if you enabled `enableDamping` or
`autoRotate` — otherwise it updates on input.

## Loading models

The generic front door dispatches on file extension (`.obj`, `.dae`, `.gltf`, `.glb`, `.stl`):

```cpp
ModelLoader loader;
auto model = loader.load("robot.glb");   // shared_ptr<Group>
scene.add(model);
```

Its async sibling returns immediately with an empty `AsyncGroup` whose children appear once the
worker thread finishes — the right choice for anything big enough to stall a frame:

```cpp
auto model = loader.loadAsync("city.glb");
scene.add(model);   // pops in when ready
```

Reach for a specific loader when you need format-specific results — most notably `GLTFLoader`,
which returns animation clips alongside the scene:

```cpp
GLTFLoader loader;
auto result = loader.load("Soldier.glb");
if (!result) return 1;

scene.add(result->scene);
auto& clips = result->animations;
```

Also available: `OBJLoader` + `MTLLoader`, `STLLoader`, `ColladaLoader`, `FBXLoader`
(`-DTHREEPP_WITH_FBX=ON`), `USDLoader` (`-DTHREEPP_WITH_USD=ON`), `SVGLoader`, `URDFLoader` (robot
descriptions → a `Robot` with joints), `FontLoader` (typeface.json / TTF), and `AssimpLoader` as
a catch-all if you link assimp yourself.

### URDF and xacro

`URDFLoader::load` runs a `.xacro` (or any document that declares the xacro namespace) through
a built-in xacro engine before parsing it, so a description that ships as macros needs no
external `xacro` command. The engine is also usable on its own:

```cpp
xacro::Processor processor;
processor.addPackagePath("ur_description", "/opt/ros/ur_description");
processor.setArgs({{"name", "ur"}, {"ur_type", "ur5e"}});

const auto result = processor.processFile("urdf/ur.urdf.xacro");
if (!result.ok) for (const auto& e : result.errors) std::cerr << e << '\n';
```

`URDFLoader` takes the same two settings (`setArgs`, `addPackagePath`); the package registry
also backs `package://` mesh URIs, ahead of the `package.xml` walk and `ROS_PACKAGE_PATH` /
`AMENT_PREFIX_PATH`.

Supported: `property` (including `scope="parent"`/`"global"` and `default=`), `arg` with
defaults and command-line overrides, `macro` with whitespace-separated params, `:=` defaults,
`:=^` / `:=^|default` inheritance and `*block` / `**block` + `insert_block`, `include`
(with cycle detection), `if` / `unless`, and the `${...}` / `$(...)` substitutions
`arg`, `find`, `env`, `optenv`, `dirname`, `eval` and the `$$` escape. Expressions are a
Python subset over None/bool/int/float/str/list/dict — arithmetic (`**`, `//`, `%`),
comparisons, `and`/`or`/`not`, `in`, ternaries, subscripting, list literals, `math.*` and the
usual builtins, plus `xacro.load_yaml`. Anything that cannot be evaluated is an error with the
offending file and text; nothing is silently dropped.

The bundled YAML reader covers block mappings and sequences, flow collections, comments and
typed scalars, and honours python xacro's `!degrees` / `!radians` tags. It does not do anchors,
aliases, block scalars, multi-document files or any other tag — those are reported rather than
guessed at. Also unsupported next to python xacro: namespaced includes (`<xacro:include ns=…>`),
`<xacro:element>` / `<xacro:attribute>`, property block bodies, and the parts of `$(eval)` that
need a real python interpreter (comprehensions, imports, attribute access on values).

`tests/loaders/Xacro_test.cpp` covers the engine on inline documents. `XacroUr_test.cpp`
expands the Universal Robots ROS 2 description end to end; that clone is not vendored, so
point `THREEPP_UR_DESCRIPTION` at one to run it — otherwise its cases skip.

## Animation

Three independent mechanisms, often combined:

**Keyframe animation** — clips drive node transforms and material properties through a mixer:

```cpp
AnimationMixer mixer(*model);
auto* action = mixer.clipAction(clips[0]);
action->play();

canvas.animate([&] {
    mixer.update(clock.getDelta());   // must be ticked every frame
    renderer.render(scene, camera);
});
```

Actions support `reset()`, `stop()`, looping, weights and `crossFadeTo(other, seconds)` for
blending between clips.

**Skinning** — a `SkinnedMesh` bound to a `Skeleton` of `Bone`s. Loaders build this for you;
`SkeletonHelper` visualises it.

**Morph targets** — blend shapes, driven by writing into `mesh->morphTargetInfluences()`.

## Picking with the Raycaster

```cpp
Raycaster raycaster;
raycaster.setFromCamera(mouseNdc, camera);   // NDC: x,y in [-1, 1]
auto hits = raycaster.intersectObjects(scene.children, true /*recursive*/);

if (!hits.empty()) {
    const auto& hit = hits.front();   // sorted nearest-first
    hit.object;      // Object3D*
    hit.point;       // world-space Vector3
    hit.distance;
    hit.face;        // optional Face3 (with normal)
    hit.uv;          // optional texture coordinate
    hit.instanceId;  // optional, for InstancedMesh
}
```

Converting mouse pixels to NDC is the step people get wrong — note the Y flip:

```cpp
const auto size = canvas.size();
mouse.x =  (pos.x / static_cast<float>(size.width()))  * 2 - 1;
mouse.y = -(pos.y / static_cast<float>(size.height())) * 2 + 1;
```

`Raycaster::layers` filters by layer, which is the cheap way to make helpers, gizmos and
backdrops un-pickable.

## Choosing a backend

Construct the renderer you want directly:

```cpp
GLRenderer renderer(canvas);       // OpenGL
VulkanRenderer renderer(canvas);   // Vulkan deferred (requires THREEPP_WITH_VULKAN)
```

Or let `createRenderer` decide — with no explicit API it prompts on stdin, which is how most
examples let you pick at launch:

```cpp
auto renderer = createRenderer(canvas);                       // interactive prompt
auto renderer = createRenderer(canvas, GraphicsAPI::OpenGL);  // explicit
```

`createRenderer` returns `std::unique_ptr<Renderer>` — the backend-neutral base. Write against
that and your program works on both. Reach for the concrete type only for backend-specific
features:

```cpp
VulkanRenderer renderer(canvas);
renderer.setDenoise(true);
renderer.setRestirDIEnabled(true);
renderer.setRenderScale(0.9f);   // trace below native res; TAA upsamples
renderer.setFireflyClamp(6.0f);
```

The `Canvas` picks its context from whichever renderer is constructed against it — you do not
configure the API on the canvas.

## Rendering off-screen and headless

Render into a texture instead of the window:

```cpp
RenderTarget target(512, 512, {});   // GLRenderTarget is a back-compat alias for this
renderer.setRenderTarget(&target);
renderer.render(scene, camera);
renderer.setRenderTarget(nullptr);
// target.texture is now sampleable by a material
```

Or run with no visible window at all — the basis of the Python bindings, the CI image tests, and
synthetic dataset generation:

```cpp
Canvas canvas("offscreen", {{"headless", true}, {"size", WindowSize{800, 600}}});
GLRenderer renderer(canvas);

renderer.render(scene, camera);
auto pixels = renderer.readRGBPixels();     // std::vector<unsigned char>, RGB
renderer.writeFramebuffer("frame.png");     // .png/.jpg/.bmp by extension
```

`CubeCamera` renders the six faces of a cube map in one call — for dynamic reflections.

## Beyond three.js

Everything above is the three.js surface. `threepp` adds a substantial layer on top, mostly
under `extras/` and `helpers/`:

**Simulation & robotics**
- `extras/physx/` — PhysX integration: `PhysxWorld` (fixed-timestep stepping), `Articulation`
  and `UrdfArticulation` (robots), `PhysxVehicle`, `PhysxSoftBody`, GPU batching for RL.
- `helpers/` sensors — `LidarSensor`, `PathTracedLidarSensor`, `DepthSensor`,
  `EventCameraSensor`: ray-traced range/depth/event data for perception work.
- `loaders/URDFLoader` + `objects/Robot` — robot descriptions with articulated joints.

**World building**
- `extras/terrain/` — `TerrainGenerator`, quadtree-LOD `TerrainTiles`, splat maps, real-world
  DEM/geodata (`GeoTerrain`, `GeoBuildings`).
- `extras/vegetation/` — `TreeGenerator`, `GrassField`, `GrassTiles`.
- `extras/road/` — `RoadNetwork`, `RoadGenerator`.
- `extras/pointcloud/` — `VoxelGrid`, `MarchingCubes`, ICP registration.

**Rendering extras** — `Ocean` (FFT-displaced water with foam), `Sky`, `Water`, `Reflector`,
`ParticleSystem`, `GrassMesh`, `DisplacedMesh`.

**Tooling**
- `extras/imgui/ImguiContext` — a few lines to get a Dear ImGui overlay on either backend, plus
  a prebuilt `RendererSettings` panel.
- **Python bindings** (`-DTHREEPP_WITH_PYTHON=ON`, or `pip install .`) — the same scene graph
  from Python, rendering into NumPy arrays.

**Curves and shapes** — `extras/core/` (`Curve`, `Path`, `Shape`, `Font`) and `extras/curves/`
(Catmull-Rom, Bézier, spline), feeding `ExtrudeGeometry`, `TubeGeometry`, `LatheGeometry`,
`TextGeometry`.

## Gotchas

| Symptom | Cause |
|---|---|
| Nothing renders | No light (and a lit material), or the camera is inside/behind the object, or the object is outside `[near, far]`. |
| Object vanished after a refactor | Its owning `shared_ptr` died. `remove()` and a discarded `removeFromParent()` both destroy an owned child. |
| Camera change has no effect | Missing `camera.updateProjectionMatrix()`. |
| Stretched image after resize | Missing `onWindowResize` handler (`aspect` + `setSize`). |
| No shadows | One of the three opt-ins missing: renderer, light, per-object cast/receive. |
| Washed-out or too-dark PBR | Wrong `ColorSpace` on a texture — color maps need `sRGB`, data maps must not have it. |
| New texture/`flatShading` ignored | Structural material change needs `mat->needsUpdate()`. |
| Edited vertex data ignored | Attribute needs `needsUpdate()`. |
| Z-fighting in a large scene | `nearPlane` far too small. |
| Transparent objects sort wrong | Inherent to sorted alpha; use `renderOrder`, or `depthWrite = false`, or `alphaTest`. |
| `getWorldPosition` returns stale values | Call `updateMatrixWorld()` after moving the node, before reading. |
| Animation frozen | `mixer.update(dt)` not called each frame. |
| Stack-object listener crashes | `addMouseListener`/`addKeyListener` are non-owning; the listener must stay alive while registered. Use the owning `onKeyPressed(lambda)` style to avoid the issue. |

## Where to go next

The [examples](../examples) folder is the de-facto documentation. The examples build from the
repository itself, not from a FetchContent consumer: clone `threepp`, configure with the
defaults (`THREEPP_BUILD_EXAMPLES` is ON for a top-level build), and the models/textures they need are fetched
automatically into a `threepp_data` checkout — see [How to build](../README.md#how-to-build).

A reading order:

| Start here | |
|---|---|
| [examples/demo.cpp](../examples/demo.cpp) | The canonical minimal app |
| [geometries/basic_geometries.cpp](../examples/geometries/basic_geometries.cpp) | The geometry catalogue |
| [lights/](../examples/lights) | One example per light type |
| [misc/raycast.cpp](../examples/misc/raycast.cpp) | Picking, end to end |
| [controls/](../examples/controls) | Fly, drag, transform gizmo |

| Then | |
|---|---|
| [loaders/gltf_loader.cpp](../examples/loaders/gltf_loader.cpp) | Loading real assets |
| [animation/gltf_animation.cpp](../examples/animation/gltf_animation.cpp) | Mixer, crossfade, skeletons |
| [textures/hdr_envmap.cpp](../examples/textures/hdr_envmap.cpp) | Image-based lighting |
| [objects/instancing.cpp](../examples/objects/instancing.cpp) | Thousands of objects, one draw call |
| [misc/transmission.cpp](../examples/misc/transmission.cpp) | Glass and `MeshPhysicalMaterial` |

| Deeper | |
|---|---|
| [vulkan/vulkan_ocean_minimal.cpp](../examples/vulkan/vulkan_ocean_minimal.cpp) | Deferred backend, in ~40 lines of scene code |
| [helpers/lidar.cpp](../examples/helpers/lidar.cpp), [helpers/depth_sensor.cpp](../examples/helpers/depth_sensor.cpp) | Simulated sensors |
| [extras/](../examples/extras) | Terrain, vegetation, curves |
| [projects/](../examples/projects) | Complete applications (Crane3R, Drive, FPS, MapView) |

Because the API mirrors three.js, the [three.js documentation](https://threejs.org/docs/) and
its [examples](https://threejs.org/examples/) are a usable reference for the shared surface —
translate `new THREE.Mesh(geo, mat)` to `Mesh::create(geo, mat)` and most of it carries over.
