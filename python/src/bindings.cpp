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

#ifdef _WIN32
#include <windows.h>
namespace {
    // PhysX loads its GPU module (PhysXGpu_64.dll, staged next to this .pyd by the
    // build) at runtime via a bare-name LoadLibrary, which searches python.exe's
    // directory + CWD + PATH — none of which is the extension's directory. Register
    // the module's own folder on the DLL search path at import time so GPU dynamics
    // work regardless of the process working directory.
    void register_module_dll_dir() {
        HMODULE self = nullptr;
        if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                        GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                reinterpret_cast<LPCWSTR>(&register_module_dll_dir), &self))
            return;
        wchar_t path[MAX_PATH];
        const DWORD n = GetModuleFileNameW(self, path, MAX_PATH);
        if (n == 0 || n >= MAX_PATH) return;
        for (DWORD i = n; i-- > 0;) {
            if (path[i] == L'\\' || path[i] == L'/') {
                path[i] = 0;
                break;
            }
        }
        SetDllDirectoryW(path);
    }
}// namespace
#endif

PYBIND11_MODULE(threepp, m) {
#ifdef _WIN32
    register_module_dll_dir();
#endif
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
    tp::init_ocean(m);   // DisplacedMesh + Ocean (subclasses of Mesh from init_objects)
    tp::init_animation(m);// before loaders: GLTFResult.animations is a list of AnimationClip
    tp::init_cameras(m);
    tp::init_lights(m);
    tp::init_helpers(m);// after lights: light helpers reference light types
    tp::init_audio(m);      // AudioListener + Audio + PositionalAudio
    tp::init_pointcloud(m); // VoxelGrid, ICP, MarchingCubes
    tp::init_render(m);
    tp::init_loaders(m);// returns Group/Texture/BufferGeometry + GLTFResult
    tp::init_robot(m);  // URDFLoader + Robot (needs Object3D from init_core)
    tp::init_text(m);   // fonts + text + SVG (needs Mesh/Sprite/Group + materials)
    tp::init_vulkan(m); // optional deferred renderer + G-buffer AOVs
    tp::init_imgui(m);  // optional Dear ImGui UI (GL backend)
    tp::init_physx(m);  // optional PhysX rigid-body world
}
