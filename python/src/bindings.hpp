// Shared declarations for the threepp pybind11 module. The module is split
// across several translation units; each registers a slice of the API through
// one of the init_* helpers below, all called from PYBIND11_MODULE in
// bindings.cpp.
#ifndef THREEPP_PY_BINDINGS_HPP
#define THREEPP_PY_BINDINGS_HPP

#include <pybind11/pybind11.h>

#include <memory>
#include <string>

namespace threepp {
    class Material;
    class Object3D;
    class Renderer;
}

namespace threepp_py {

    namespace py = pybind11;

    // ── GIL policy ─────────────────────────────────────────────────────────
    // Any def whose C++ body stalls the caller — PhysX step, a GPU frame, a
    // device-idle readback, a sensor scan — releases the GIL for the stall so
    // other Python threads (torch inference, dataset writers, ROS spinners)
    // keep running. Rules, in order of how expensive they are to violate:
    //  1. Never touch a py::* object (numpy arrays included) while released —
    //     build numpy AFTER the release scope closes.
    //  2. Callbacks that C++ may invoke while released must re-acquire:
    //     manual py::function wraps take py::gil_scoped_acquire (see
    //     bind_physx on_pre_substep); std::function params via
    //     pybind11/functional.h re-acquire automatically.
    //  3. gil_scoped_release does not nest. Release at the LEAF call only
    //     (see PyVulkanRenderer's *_released helpers), never in a wrapper
    //     around something that releases internally.
    //  4. Never hold a C++ mutex across a release if any other def takes that
    //     mutex with the GIL held — that is a lock-order deadlock (see the
    //     scan_begin/scan_collect restructure in bind_render.cpp).
    // Plain member-function defs with no py::* in the body can use
    // py::call_guard<py::gil_scoped_release>() instead (PhysxWorld.step).

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

    // The string userData entry for `key` as a Python str, or None when the key
    // is absent or the value is not a string. Read access for scripts to the
    // editor's flat `key=value;...` configs (spline, physics, script), which
    // are all stored as strings. Defined in bind_core.cpp; bound on Object3D
    // there and re-bound concretely on the virtual-base leaves in
    // bind_objects.cpp.
    py::object user_data_string(const threepp::Object3D& o, const std::string& key);

    // The write side of the same escape hatch, and the reason it exists: a
    // GENERATOR script builds real scene content, and content that cannot carry
    // `userData["physics"]` is content the play session will never simulate.
    // Strings only, for the reason the reader gives — and an empty value ERASES
    // the entry, which is how every editor config spells "off". Defined in
    // bind_core.cpp and re-bound concretely in bind_objects.cpp, exactly like
    // the reader.
    void set_user_data_string(threepp::Object3D& o, const std::string& key, const std::string& value);

    void init_math(py::module_& m);
    void init_textures(py::module_& m);
    void init_core(py::module_& m);
    void init_geometries(py::module_& m);
    // BVH + Ray queries (mesh-vs-mesh overlap / distance); needs BufferGeometry
    // from init_geometries and Ray/Box3/Matrix4 from init_math.
    void init_bvh(py::module_& m);
    void init_curves(py::module_& m);// Curve2/Curve3, CatmullRomCurve3, LineCurve(3), SplineCurve, Path, Shape
    void init_editor(py::module_& m);// threepp.editor — SplinePath sampling of authored splines
    // threepp.editor, physics half — RigidBody/SoftBody handles onto the live
    // play session. Defined only where the PhysX SDK was found, and must be
    // called AFTER init_editor, which creates the submodule it adds to.
    void init_editor_physics(py::module_& m);
    // threepp.editor, sensor half — read handles onto the proprioceptive sensors
    // the play session is running. Same gating and ordering as the physics half.
    void init_editor_sensors(py::module_& m);
    // threepp.editor, camera half — the read handle onto the play session's
    // colour cameras. Editor-only like the two above (it needs a play session),
    // and likewise must follow init_editor, but NOT gated on PhysX: an image is
    // a renderer product, so a build without the SDK still has this one.
    void init_editor_camera(py::module_& m);
    // threepp.editor, authoring half — add() for a generator script to build
    // document content. Editor-only for the same reason as the physics half (the
    // wheel has no document), and likewise must follow init_editor.
    void init_editor_authoring(py::module_& m);
    void init_materials(py::module_& m);
    void init_objects(py::module_& m);
    void init_ocean(py::module_& m);// DisplacedMesh + Ocean; no-op unless built with Vulkan
    void init_animation(py::module_& m);
    void init_cameras(py::module_& m);
    void init_lights(py::module_& m);
    void init_helpers(py::module_& m); // AxesHelper, GridHelper, ArrowHelper, box/camera/skeleton/light helpers
    void init_audio(py::module_& m);     // AudioListener, Audio, PositionalAudio; no-op unless THREEPP_WITH_AUDIO
    void init_pointcloud(py::module_& m); // VoxelGrid, ICP, MarchingCubes
    void init_terrain(py::module_& m);   // TerrainGenerator + TerrainParams
    void init_vegetation(py::module_& m); // TreeGenerator, TreeTextures
    void init_fauna(py::module_& m);      // Flock (extras/fauna ambient birds)
    // ParticleField (weather / granular fields; Vulkan-only at render time).
    // Needs Mesh from init_objects, Material from init_materials, BufferGeometry
    // from init_geometries and Texture from init_textures.
    void init_particles(py::module_& m);
    void init_splats(py::module_& m);    // SplatLoader + SplatCloud (3D Gaussian Splatting; GL + Vulkan)
    void init_render(py::module_& m);
    void init_loaders(py::module_& m);
    void init_robot(py::module_& m); // URDFLoader + Robot (articulated Object3D)
    void init_text(py::module_& m);  // fonts, Text2D/Text3D, TextSprite, SVGLoader
    void init_vulkan(py::module_& m);// no-op unless built with the Vulkan backend
    void init_imgui(py::module_& m); // no-op unless built with imgui
    void init_physx(py::module_& m); // no-op unless built with the omniverse-physx-sdk
    void init_sensor_base(py::module_& m);// Sensor + NoiseModel + RangeNoiseModel; always available
    void init_sensors(py::module_& m);// proprioceptive sensors (Imu); no-op without PhysX

    // If `h` is the Python VulkanRenderer facade, returns the underlying
    // threepp::Renderer* (for ImGui's Vulkan overlay), else nullptr. Defined in
    // bind_vulkan.cpp; returns nullptr in a GL-only build.
    threepp::Renderer* py_vulkan_native_renderer(const py::handle& h);

}// namespace threepp_py

#endif// THREEPP_PY_BINDINGS_HPP
