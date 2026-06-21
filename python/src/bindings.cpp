// threepp — the high-level, three.js-style 3D API, exposed to Python.
//
//   import threepp as tp
//   scene  = tp.Scene()
//   camera = tp.PerspectiveCamera(75, 1.0, 0.1, 100)
//   camera.position.z = 5
//   scene.add(tp.Mesh(tp.BoxGeometry(), tp.MeshStandardMaterial()))
//   renderer = tp.GLRenderer(tp.Canvas("demo"))
//   renderer.render(scene, camera)
//
// The C++ object model is reflected as faithfully as is idiomatic in Python:
// math types are mutable value types, scene objects are reference types created
// through constructors that forward to threepp's `::create` factories and are
// kept alive by shared ownership.
#include "bindings.hpp"

namespace tp = threepp_py;

PYBIND11_MODULE(threepp, m) {
    m.doc() = "threepp — a cross-platform C++ 3D library with the high-level API "
              "of three.js, exposed to Python (scene graph, geometries, materials, "
              "cameras, lights, OpenGL renderer + headless render-to-numpy).";

    // Order matters: types must be registered before others reference them as
    // arguments/return values (math before everything; core Object3D before
    // objects/cameras/lights; geometries+materials before objects).
    tp::init_math(m);
    tp::init_textures(m);// before materials: their map slots reference Texture
    tp::init_core(m);
    tp::init_geometries(m);
    tp::init_materials(m);
    tp::init_objects(m);
    tp::init_cameras(m);
    tp::init_lights(m);
    tp::init_render(m);
    tp::init_loaders(m);// returns Group/Texture/BufferGeometry
    tp::init_vulkan(m); // optional deferred renderer + G-buffer AOVs
    tp::init_imgui(m);  // optional Dear ImGui UI (GL backend)
    tp::init_physx(m);  // optional PhysX rigid-body world
}
