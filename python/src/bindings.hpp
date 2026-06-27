// Shared declarations for the threepp pybind11 module. The module is split
// across several translation units; each registers a slice of the API through
// one of the init_* helpers below, all called from PYBIND11_MODULE in
// bindings.cpp.
#ifndef THREEPP_PY_BINDINGS_HPP
#define THREEPP_PY_BINDINGS_HPP

#include <pybind11/pybind11.h>

#include <memory>

namespace threepp {
    class Material;
    class Object3D;
    class Renderer;
}

namespace threepp_py {

    namespace py = pybind11;

    // Returns `mat` wrapped as its concrete Python material type (so e.g.
    // mesh.material.roughness works), downcasting in C++ to dodge pybind11's
    // virtual-base limitation. Defined in bind_materials.cpp.
    py::object material_to_py(const std::shared_ptr<threepp::Material>& mat);

    // Convert a Python material (or None) to shared_ptr<Material> safely:
    // pybind11 corrupts the pointer when up-casting across the `virtual Material`
    // base, so cast to the concrete type and let C++ up-cast. Defined in
    // bind_materials.cpp; used by every object constructor that takes a material.
    std::shared_ptr<threepp::Material> as_material(const py::handle& h);

    // Same idea for objects: convert a Python Object3D-derived value to
    // shared_ptr<Object3D> across threepp's `virtual Object3D` base without
    // tripping pybind11's broken pointer adjustment. Defined in bind_core.cpp.
    std::shared_ptr<threepp::Object3D> as_object3d(const py::handle& h);

    void init_math(py::module_& m);
    void init_textures(py::module_& m);
    void init_core(py::module_& m);
    void init_geometries(py::module_& m);
    void init_materials(py::module_& m);
    void init_objects(py::module_& m);
    void init_ocean(py::module_& m);// DisplacedMesh + Ocean; no-op unless built with Vulkan
    void init_animation(py::module_& m);
    void init_cameras(py::module_& m);
    void init_lights(py::module_& m);
    void init_helpers(py::module_& m); // AxesHelper, GridHelper, ArrowHelper, box/camera/skeleton/light helpers
    void init_audio(py::module_& m);     // AudioListener, Audio, PositionalAudio; no-op unless THREEPP_WITH_AUDIO
    void init_pointcloud(py::module_& m); // VoxelGrid, ICP, MarchingCubes
    void init_vegetation(py::module_& m); // TreeGenerator, TreeTextures
    void init_render(py::module_& m);
    void init_loaders(py::module_& m);
    void init_robot(py::module_& m); // URDFLoader + Robot (articulated Object3D)
    void init_text(py::module_& m);  // fonts, Text2D/Text3D, TextSprite, SVGLoader
    void init_vulkan(py::module_& m);// no-op unless built with the Vulkan backend
    void init_imgui(py::module_& m); // no-op unless built with imgui
    void init_physx(py::module_& m); // no-op unless built with the omniverse-physx-sdk

    // If `h` is the Python VulkanRenderer facade, returns the underlying
    // threepp::Renderer* (for ImGui's Vulkan overlay), else nullptr. Defined in
    // bind_vulkan.cpp; returns nullptr in a GL-only build.
    threepp::Renderer* py_vulkan_native_renderer(const py::handle& h);

}// namespace threepp_py

#endif// THREEPP_PY_BINDINGS_HPP
