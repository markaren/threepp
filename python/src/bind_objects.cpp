// Renderable objects and the Scene container.
#include "bindings.hpp"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/typing.h>

#include <algorithm>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/materials/PointsMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/objects/GrassMesh.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/materials/ShaderMaterial.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Sky.hpp"
#include "threepp/objects/Bone.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Skeleton.hpp"
#include "threepp/objects/SkinnedMesh.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/scenes/Downwash.hpp"
#include "threepp/scenes/Fog.hpp"
#include "threepp/scenes/FogExp2.hpp"
#include "threepp/scenes/Scene.hpp"
#include "threepp/textures/Texture.hpp"

using namespace threepp;

namespace threepp_py {

    // What a `.material` property hands back, for the STUB'S benefit only.
    //
    // The object itself is still produced by material_to_py's
    // dynamic_pointer_cast chain, which is not an accident: concrete materials
    // derive from Material VIRTUALLY, so pybind11 must never be handed a
    // shared_ptr<Material> and asked to downcast it — it reinterprets the holder
    // and the virtual-base offset makes that pointer wrong. Casting the already
    // downcast shared_ptr<Concrete> is the safe form (see bind_materials.cpp).
    //
    // That returns py::object, which pybind11 annotates `typing.Any` — a type
    // checker then offers NO completion and checks nothing. Saying `Material`
    // gives back the fields every material shares (name, opacity, side, ...),
    // which the base carries for exactly this purpose, and `isinstance` narrows
    // to a concrete class for the rest (color, roughness, ...).
    using PyMaterial = py::typing::Optional<Material>;

    // Mesh/Points/Line derive from Object3D *virtually*. pybind11 mishandles
    // EVERY access that crosses that virtual base: inherited def_readwrite
    // members (writing std::string `name` corrupts the heap) and methods bound
    // via `&Object3D::method` or even a lambda taking `Object3D&` (the self
    // pointer is silently wrong). The only reliable access is through the
    // concrete type, so bind the whole Object3D surface on the virtual roots
    // with concrete (`T&` / `&T::field`) handlers. These shadow the broken
    // inherited bindings in the Python MRO. Non-virtual Object3D subclasses
    // (Scene/Group/cameras/lights/Sprite) use the base bindings directly, and
    // InstancedMesh/LineSegments inherit these via their non-virtual parent.
    template<class Cls>
    static void bind_object3d_api(Cls& c) {
        using T = typename Cls::type;
        c.def_readwrite("name", &T::name)
                .def_readwrite("position", &T::position)
                .def_readwrite("rotation", &T::rotation)
                .def_readwrite("quaternion", &T::quaternion)
                .def_readwrite("scale", &T::scale)
                .def_readwrite("up", &T::up)
                .def_readwrite("visible", &T::visible)
                .def_readwrite("cast_shadow", &T::castShadow)
                .def_readwrite("receive_shadow", &T::receiveShadow)
                .def_readwrite("frustum_culled", &T::frustumCulled)
                .def_readwrite("resident", &T::resident, kResidentDoc)
                .def_readwrite("auto_lod", &T::autoLod, kAutoLodDoc)
                .def_property("ray_shadow_distance", [](const T& o) { return o.rayShadowDistance(); },
                              [](T& o, float v) { o.setRayShadowDistance(v); }, kRayShadowDistanceDoc)
                .def_property("ray_shadow_from", [](const T& o) { return o.rayShadowFrom(); },
                              [](T& o, float v) { o.setRayShadowFrom(v); }, kRayShadowFromDoc)
                .def_property("ray_cast_shadow", [](const T& o) { return o.rayCastShadow(); },
                              [](T& o, std::optional<bool> v) { o.setRayCastShadow(v); }, kRayCastShadowDoc)
                .def_readwrite("render_order", &T::renderOrder)
                .def_readwrite("matrix_auto_update", &T::matrixAutoUpdate)
                .def_readwrite("layers", &T::layers)
                .def_property_readonly("id", [](const T& o) { return o.id; })
                .def_property_readonly("uuid", [](const T& o) { return o.uuid; })
                .def_property_readonly("parent", [](T& o) { return o.parent; }, py::return_value_policy::reference)
                .def_property_readonly("children", [](T& o) { return o.children; }, py::return_value_policy::reference)
                .def("add", [](T& self, const py::args& children) {
                    for (const auto& ch : children) self.add(as_object3d(ch));
                })
                .def("remove", [](T& self, const py::handle& child) { self.remove(*as_object3d(child)); }, py::arg("object"))
                .def("remove_from_parent", [](T& o) { o.removeFromParent(); })
                .def("clear", [](T& o) { o.clear(); })
                .def("rotate_x", [](T& o, float a) { o.rotateX(a); }, py::arg("angle"))
                .def("rotate_y", [](T& o, float a) { o.rotateY(a); }, py::arg("angle"))
                .def("rotate_z", [](T& o, float a) { o.rotateZ(a); }, py::arg("angle"))
                .def("rotate_on_axis", [](T& o, const Vector3& ax, float a) { o.rotateOnAxis(ax, a); }, py::arg("axis"), py::arg("angle"))
                .def("translate_x", [](T& o, float d) { o.translateX(d); }, py::arg("distance"))
                .def("translate_y", [](T& o, float d) { o.translateY(d); }, py::arg("distance"))
                .def("translate_z", [](T& o, float d) { o.translateZ(d); }, py::arg("distance"))
                .def("look_at", [](T& o, float x, float y, float z) { o.lookAt(x, y, z); }, py::arg("x"), py::arg("y"), py::arg("z"))
                .def("look_at", [](T& o, const Vector3& v) { o.lookAt(v); }, py::arg("vector"))
                .def("get_world_position", [](T& o) { Vector3 v; o.getWorldPosition(v); return v; })
                .def("get_world_direction", [](T& o) { Vector3 v; o.getWorldDirection(v); return v; })
                // World-space pose — mirrors the Object3D base bindings in bind_core.cpp
                // (must be kept in sync: this template shadows them for virtual-base
                // leaves Mesh/Points/Line). All refresh the world matrix first.
                .def("get_world_quaternion", [](T& o) { Quaternion q; o.getWorldQuaternion(q); return q; })
                .def("get_world_scale", [](T& o) { Vector3 v; o.getWorldScale(v); return v; })
                .def_property_readonly("matrix_world", [](T& o) { o.updateWorldMatrix(true, false); return *o.matrixWorld; })
                .def("local_to_world", [](T& o, Vector3 v) { o.localToWorld(v); return v; }, py::arg("vector"))
                .def("world_to_local", [](T& o, Vector3 v) { o.worldToLocal(v); return v; }, py::arg("vector"))
                .def("get_object_by_name", [](T& o, const std::string& n) { return node_to_py(o.getObjectByName(n), o); }, py::arg("name"))
                // The lambda takes T& so pybind extracts a correct self pointer;
                // the T& -> Object3D& upcast inside is plain C++ and safe.
                .def("get_user_data", [](const T& o, const std::string& key) { return user_data_string(o, key); }, py::arg("key"))
                .def("set_user_data", [](T& o, const std::string& key, const std::string& value) { set_user_data_string(o, key, value); },
                     py::arg("key"), py::arg("value"))
                .def("traverse", [](T& self, const std::function<void(py::object)>& cb) {
                    self.traverse([&cb](Object3D& o) { cb(py::cast(&o, py::return_value_policy::reference)); });
                }, py::arg("callback"))
                .def("update_matrix", [](T& o) { o.updateMatrix(); })
                .def("update_matrix_world", [](T& o, bool force) { o.updateMatrixWorld(force); }, py::arg("force") = false)
                .def("__repr__", [](const T& o) { return "<threepp." + o.type() + " name='" + o.name + "'>"; });
    }

    void init_objects(py::module_& m) {

        // ---- Mesh ------------------------------------------------------------
        auto mesh = py::class_<Mesh, Object3D, std::shared_ptr<Mesh>>(m, "Mesh");
        bind_object3d_api(mesh);
        mesh.def(py::init([](std::shared_ptr<BufferGeometry> g, const py::object& mat) {
                    return Mesh::create(std::move(g), as_material(mat));
                }),
                 py::arg("geometry") = std::shared_ptr<BufferGeometry>{}, py::arg("material") = py::none())
                .def_property_readonly("geometry", &Mesh::geometry)
                .def_property_readonly("material", [](Mesh& self) { return PyMaterial(material_to_py(self.material())); })
                .def("set_geometry", &Mesh::setGeometry, py::arg("geometry"))
                .def("set_material", [](Mesh& self, const py::object& mat) { self.setMaterial(as_material(mat)); }, py::arg("material"))
                .def("set_fade", &Mesh::setFade, py::arg("fade"), py::arg("complement") = false,
                     "Dithered LOD crossfade (Vulkan): draw this share of the mesh's pixels, 1 = all (the "
                     "default), 0 = none, through a per-frame screen-space dither the temporal resolve "
                     "blends. Crossfade two levels with fade f on the outgoing one and fade 1 - f, "
                     "complement=True, on the incoming one: together they cover every pixel exactly once. "
                     "Shadows and rays see one of the pair (the plain side while its fade >= 0.5). On an "
                     "InstancedMesh this applies to every instance until set_fade_at / set_fades.")
                .def_property_readonly("fade", &Mesh::fade, "The fade set_fade gave (1 = fully drawn).")
                .def_property_readonly("fade_complement", &Mesh::fadeComplement,
                                       "Whether set_fade picked the complementary dither pattern.");

        // ---- Sky ---------------------------------------------------------------
        // The Preetham daylight model (three.js's Sky). Its look is the six
        // uniforms of its ShaderMaterial, exposed here as properties since the
        // bindings carry no ShaderMaterial uniform map.
        {
            auto uniform = [](Sky& s, const char* name) -> Uniform& {
                return s.materialAs<ShaderMaterial>()->uniforms.at(name);
            };
            py::class_<Sky, Mesh, std::shared_ptr<Sky>>(
                    m, "Sky",
                    "The Preetham daylight model (three.js's Sky). On the GL renderer a dome the scene sits "
                    "in, shaded per pixel; on the Vulkan renderer the environment itself: the sky the camera "
                    "sees, the light the scene is lit by (and its sun, where the scene has no DirectionalLight), "
                    "baked from these properties each frame they change. A scene.environment outranks it.")
                    .def(py::init([]() { return Sky::create(); }))
                    .def_property(
                            "turbidity", [=](Sky& s) { return uniform(s, "turbidity").value<float>(); },
                            [=](Sky& s, float v) { uniform(s, "turbidity").value<float>() = v; },
                            "Haze: 2 is a clear sky, 10 a hazy one.")
                    .def_property(
                            "rayleigh", [=](Sky& s) { return uniform(s, "rayleigh").value<float>(); },
                            [=](Sky& s, float v) { uniform(s, "rayleigh").value<float>() = v; },
                            "Rayleigh scattering: the blue of the sky (1 = the model's).")
                    .def_property(
                            "mie_coefficient", [=](Sky& s) { return uniform(s, "mieCoefficient").value<float>(); },
                            [=](Sky& s, float v) { uniform(s, "mieCoefficient").value<float>() = v; },
                            "Mie scattering: the glow around the sun (0.005 = the model's).")
                    .def_property(
                            "mie_directional_g", [=](Sky& s) { return uniform(s, "mieDirectionalG").value<float>(); },
                            [=](Sky& s, float v) { uniform(s, "mieDirectionalG").value<float>() = v; },
                            "How tightly that glow hugs the sun, 0..1 (0.8 = the model's).")
                    .def_property(
                            "sun_position", [=](Sky& s) { return uniform(s, "sunPosition").value<Vector3>(); },
                            [=](Sky& s, const Vector3& v) { uniform(s, "sunPosition").value<Vector3>().copy(v); },
                            "Where the sun stands, as a direction from the origin (its length does not matter); "
                            "copy a DirectionalLight's position here to keep the two in step.");
        }

        // ---- Bone / Skeleton / SkinnedMesh -----------------------------------
        // Registering these lets pybind11's RTTI downcast hand back a glTF
        // character's meshes as SkinnedMesh (an isinstance of Mesh, so
        // .material/.geometry work) and its joints as Bone. Bone is a plain
        // non-virtual Object3D subclass (like Group): the base bindings apply.
        py::class_<Bone, Object3D, std::shared_ptr<Bone>>(m, "Bone")
                .def(py::init(&Bone::create));

        py::class_<Skeleton, std::shared_ptr<Skeleton>>(m, "Skeleton")
                .def(py::init([](const std::vector<std::shared_ptr<Bone>>& bones, const std::vector<Matrix4>& boneInverses) {
                         return Skeleton::create(bones, boneInverses);
                     }),
                     py::arg("bones"), py::arg("bone_inverses") = std::vector<Matrix4>{},
                     "A skeleton over `bones`. Without `bone_inverses` they are computed from the "
                     "bones' current world matrices (the bind pose).")
                // Copies: mutating the returned lists does not touch the skeleton.
                .def_property_readonly("bones", [](const Skeleton& s) { return s.bones; },
                                       "The bones, in skin-joint order (a copy of the list; the Bone objects are shared).")
                .def_property_readonly("bone_inverses", [](const Skeleton& s) { return s.boneInverses; },
                                       "Inverse bind matrices, one per bone (a copy).")
                .def_property_readonly("uuid", &Skeleton::uuid)
                .def("get_bone_by_name", [](Skeleton& s, const std::string& name) -> std::shared_ptr<Bone> {
                    for (const auto& b : s.bones) {
                        if (b && b->name == name) return b;
                    }
                    return nullptr;
                }, py::arg("name"), "The first bone named `name`, or None.")
                .def("calculate_inverses", &Skeleton::calculateInverses,
                     "Recompute bone_inverses from the bones' current world matrices.")
                .def("pose", &Skeleton::pose, "Move every bone back to its bind pose.")
                .def("update", &Skeleton::update, "Recompute the bone matrices the skinning reads.");

        py::enum_<SkinnedMesh::BindMode>(m, "SkinnedMeshBindMode")
                .value("ATTACHED", SkinnedMesh::BindMode::Attached)
                .value("DETACHED", SkinnedMesh::BindMode::Detached);

        // SkinnedMesh inherits Mesh's concrete Object3D API via its non-virtual
        // Mesh parent (same as InstancedMesh).
        auto skinned = py::class_<SkinnedMesh, Mesh, std::shared_ptr<SkinnedMesh>>(m, "SkinnedMesh");
        skinned.def(py::init([](std::shared_ptr<BufferGeometry> g, const py::object& mat) {
                        return SkinnedMesh::create(std::move(g), as_material(mat));
                    }),
                    py::arg("geometry"), py::arg("material"),
                    "A mesh deformed by a Skeleton. The geometry needs skinIndex and skinWeight "
                    "(4 per vertex); call bind() with a skeleton before rendering.")
                .def_property_readonly("skeleton", [](const SkinnedMesh& s) { return s.skeleton; },
                                       "The bound Skeleton, or None before bind().")
                .def_property_readonly("bind_matrix", [](const SkinnedMesh& s) { return s.bindMatrix; },
                                       "The mesh's world matrix at bind time (a copy; set it through bind()).")
                .def_property_readonly("bind_matrix_inverse", [](const SkinnedMesh& s) { return s.bindMatrixInverse; },
                                       "Inverse of bind_matrix (a copy).")
                .def_readwrite("bind_mode", &SkinnedMesh::bindMode,
                               "ATTACHED (default): the skin follows the mesh's own transform; "
                               "DETACHED: the bind matrix is held fixed.")
                // The C++ methods dereference the skeleton / skinWeight
                // attribute unchecked; raise instead of crashing the interpreter.
                .def("bind", [](SkinnedMesh& s, const std::shared_ptr<Skeleton>& skeleton, const std::optional<Matrix4>& bindMatrix) {
                    if (!skeleton) throw py::value_error("SkinnedMesh.bind: skeleton is None");
                    s.bind(skeleton, bindMatrix);
                }, py::arg("skeleton"), py::arg("bind_matrix") = py::none(),
                   "Bind `skeleton`. Without `bind_matrix` the mesh's current world matrix is used and "
                   "the skeleton's inverses are recomputed.")
                .def("pose", [](SkinnedMesh& s) {
                    if (!s.skeleton) throw py::value_error("SkinnedMesh.pose: no skeleton bound");
                    s.pose();
                }, "Move the skeleton back to its bind pose.")
                .def("normalize_skin_weights", [](SkinnedMesh& s) {
                    const auto g = s.geometry();
                    if (!g || !g->getAttribute<float>("skinWeight")) {
                        throw py::value_error("SkinnedMesh.normalize_skin_weights: the geometry has no float skinWeight attribute");
                    }
                    s.normalizeSkinWeights();
                }, "Scale each vertex's four skin weights to sum to 1.");

        // ---- Group -----------------------------------------------------------
        py::class_<Group, Object3D, std::shared_ptr<Group>>(m, "Group")
                .def(py::init(&Group::create));

        // ---- InstancedMesh ---------------------------------------------------
        // InstancedMesh inherits Mesh's concrete Object3D API (Mesh is a
        // non-virtual base of InstancedMesh, so that access is safe).
        auto inst = py::class_<InstancedMesh, Mesh, std::shared_ptr<InstancedMesh>>(m, "InstancedMesh");
        inst.def(py::init([](std::shared_ptr<BufferGeometry> g, const py::object& mat, size_t count) {
                    return InstancedMesh::create(std::move(g), as_material(mat), count);
                }),
                 py::arg("geometry"), py::arg("material"), py::arg("count"))
                .def_property_readonly("count", &InstancedMesh::count)
                .def("set_count", &InstancedMesh::setCount, py::arg("count"))
                .def("set_matrix_at", &InstancedMesh::setMatrixAt, py::arg("index"), py::arg("matrix"))
                .def("set_matrices", [](InstancedMesh& im, const py::array& matrices, size_t offset) {
                    const auto dt = matrices.dtype();
                    if (dt.kind() != 'f' || (dt.itemsize() != 4 && dt.itemsize() != 8)) {
                        throw py::value_error("set_matrices: matrices must be float32 or float64");
                    }
                    const bool flat = matrices.ndim() == 2 && matrices.shape(1) == 16;
                    const bool square = matrices.ndim() == 3 && matrices.shape(1) == 4 && matrices.shape(2) == 4;
                    if (!flat && !square) {
                        throw py::value_error("set_matrices: matrices must have shape (N, 16) or (N, 4, 4)");
                    }
                    const auto n = static_cast<size_t>(matrices.shape(0));
                    auto& dst = im.instanceMatrix()->array();
                    const size_t capacity = dst.size() / 16;
                    if (offset + n > capacity) {
                        throw py::value_error("set_matrices: offset + N = " + std::to_string(offset + n) +
                                              " exceeds the instance capacity " + std::to_string(capacity));
                    }
                    // One contiguous float32 view (a no-op for a C-contiguous float32 input).
                    const auto src = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(matrices);
                    const float* s = src.data();
                    float* d = dst.data() + offset * 16;
                    if (flat) {
                        std::copy_n(s, n * 16, d);
                    } else {
                        // M[row, col] row-major -> Matrix4.elements column-major.
                        for (size_t i = 0; i < n; ++i, s += 16, d += 16) {
                            for (int r = 0; r < 4; ++r) {
                                for (int c = 0; c < 4; ++c) d[c * 4 + r] = s[r * 4 + c];
                            }
                        }
                    }
                    im.instanceMatrix()->needsUpdate();
                }, py::arg("matrices"), py::arg("offset") = 0,
                   "Write N instance matrices in one call: instances offset .. offset+N-1. matrices is a "
                   "float32/float64 numpy array, shape (N, 16) in Matrix4.elements order (column-major, "
                   "what get_matrix_at(i).elements() holds) or (N, 4, 4) conventional row-major M[row, col] "
                   "(transposed on copy). Marks the instance matrix for re-upload, which set_matrix_at "
                   "does not. Raises ValueError on a bad shape/dtype or offset + N past the instance capacity.")
                .def("get_matrix_at", [](InstancedMesh& im, size_t i) {
                    Matrix4 mtx;
                    im.getMatrixAt(i, mtx);
                    return mtx;
                }, py::arg("index"))
                .def("set_color_at", &InstancedMesh::setColorAt, py::arg("index"), py::arg("color"))
                .def("get_color_at", [](InstancedMesh& im, size_t i) {
                    Color c;
                    im.getColorAt(i, c);
                    return c;
                }, py::arg("index"),
                   "The instance's colour. Raises before any set_color_at / set_colors (no instance colours).")
                .def("set_colors", [](InstancedMesh& im, const py::array& colors, size_t offset) {
                    const auto dt = colors.dtype();
                    if (dt.kind() != 'f' || (dt.itemsize() != 4 && dt.itemsize() != 8)) {
                        throw py::value_error("set_colors: colors must be float32 or float64");
                    }
                    if (colors.ndim() != 2 || colors.shape(1) != 3) {
                        throw py::value_error("set_colors: colors must have shape (N, 3)");
                    }
                    const auto n = static_cast<size_t>(colors.shape(0));
                    const size_t capacity = im.instanceMatrix()->array().size() / 16;
                    if (offset + n > capacity) {
                        throw py::value_error("set_colors: offset + N = " + std::to_string(offset + n) +
                                              " exceeds the instance capacity " + std::to_string(capacity));
                    }
                    // One contiguous float32 view (a no-op for a C-contiguous float32 input).
                    const auto src = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(colors);
                    im.setColors(offset, n, src.data());
                }, py::arg("colors"), py::arg("offset") = 0,
                   "Write N instance colours in one call: instances offset .. offset+N-1. colors is a "
                   "float32/float64 numpy array of shape (N, 3), the r, g, b each set_color_at's Color "
                   "holds. Marks the instance colours for upload once, which set_color_at does not. "
                   "Raises ValueError on a bad shape/dtype or offset + N past the instance capacity.")
                .def("instance_matrix_needs_update", [](InstancedMesh& im) { im.instanceMatrix()->needsUpdate(); })
                .def("instance_color_needs_update", [](InstancedMesh& im) {
                    if (auto* c = im.instanceColor()) c->needsUpdate();
                })
                .def("set_fade_at", &InstancedMesh::setFadeAt, py::arg("index"), py::arg("fade"),
                     py::arg("complement") = false,
                     "Mesh.set_fade for one instance. The first call gives every instance its own fade "
                     "(all fully drawn), which then replaces the mesh-wide set_fade. Marks the fades for "
                     "upload.")
                .def("set_fades", [](InstancedMesh& im, const py::array& fades, size_t offset, const py::object& complement) {
                    if (fades.dtype().kind() != 'f' && fades.dtype().kind() != 'i' && fades.dtype().kind() != 'u') {
                        throw py::value_error("set_fades: fades must be a numeric array");
                    }
                    const auto f = py::array_t<float, py::array::c_style | py::array::forcecast>::ensure(fades);
                    if (!f || f.ndim() != 1) throw py::value_error("set_fades: fades must have shape (N,)");
                    const auto n = static_cast<size_t>(f.shape(0));
                    if (offset + n > im.instanceMatrix()->array().size() / 16) {
                        throw py::value_error("set_fades: offset + N = " + std::to_string(offset + n) +
                                              " exceeds the instance capacity " +
                                              std::to_string(im.instanceMatrix()->array().size() / 16));
                    }
                    if (py::isinstance<py::bool_>(complement)) {
                        // One flag for the whole run: no per-instance array to build.
                        if (complement.cast<bool>()) {
                            const auto all = std::make_unique<bool[]>(n);
                            std::fill_n(all.get(), n, true);
                            im.setFades(offset, n, f.data(), all.get());
                        } else {
                            im.setFades(offset, n, f.data(), nullptr);
                        }
                        return;
                    }
                    const auto c = py::array_t<bool, py::array::c_style | py::array::forcecast>::ensure(complement);
                    if (!c || c.ndim() != 1 || static_cast<size_t>(c.shape(0)) != n) {
                        throw py::value_error("set_fades: complement must be a bool or a bool array of shape (N,)");
                    }
                    im.setFades(offset, n, f.data(), c.data());
                }, py::arg("fades"), py::arg("offset") = 0, py::arg("complement") = false,
                   "Fades for instances offset .. offset+N-1 in one call (Mesh.set_fade per instance). fades "
                   "is a numpy array of shape (N,), values in [0, 1]; complement a bool for the whole run or "
                   "a bool array of shape (N,). Marks the fades for upload. Raises ValueError on a bad "
                   "shape or offset + N past the instance capacity.")
                .def("get_fade_at", &InstancedMesh::getFadeAt, py::arg("index"),
                     "The instance's fade (the mesh-wide one before any set_fade_at / set_fades).")
                .def("get_fade_complement_at", &InstancedMesh::getFadeComplementAt, py::arg("index"));

        // ---- GrassMesh -------------------------------------------------------
        // Mesh subclass whose vertices are GPU wind-displaced by the Vulkan path
        // tracer's grass-wind compute pass (renders as a plain static Mesh on
        // GL). Inherits Mesh's concrete Object3D API via its non-virtual
        // Mesh parent (same as InstancedMesh). The geometry must be ONE merged
        // blade field carrying a per-vertex float "heightFrac" attribute (0 at a
        // blade's base, 1 at its tip) that drives the wind sway weighting.
        auto grass = py::class_<GrassMesh, Mesh, std::shared_ptr<GrassMesh>>(m, "GrassMesh");
        grass.def(py::init([](std::shared_ptr<BufferGeometry> g, const py::object& mat) {
                      return GrassMesh::create(std::move(g), as_material(mat));
                  }),
                  py::arg("geometry"), py::arg("material"))
                .def_property("wind_strength",
                              [](GrassMesh& g) { return g.params.windStrength; },
                              [](GrassMesh& g, float v) { g.params.windStrength = v; })
                .def_property("wind_dir",
                              [](GrassMesh& g) { return g.params.windDir; },
                              [](GrassMesh& g, const Vector2& v) { g.params.windDir = v; })
                .def_property("time",
                              [](GrassMesh& g) { return g.params.time; },
                              [](GrassMesh& g, float v) { g.params.time = v; },
                              "Animation clock (seconds); set per frame to advance the wind.")
                .def_property("max_anim_distance",
                              [](GrassMesh& g) { return g.params.maxAnimDistance; },
                              [](GrassMesh& g, float v) { g.params.maxAnimDistance = v; },
                              "Vulkan: the wind and its BLAS refit freeze once the camera is farther than this (m) "
                              "from the field's box; the blades hold their pose. 0 (default) = always animate. A "
                              "field a downwash jet reaches is never frozen.")
                .def_property("blade_height",
                              [](GrassMesh& g) { return g.params.bladeHeight; },
                              [](GrassMesh& g, float v) { g.params.bladeHeight = v; },
                              "The blades' height (local m) the rotor downwash bends them by (VulkanRenderer."
                              "downwash). 0 (default) = estimated from the geometry, the slope of y over heightFrac.");

        // ---- Downwash --------------------------------------------------------
        // The rotor downwash's sources (scenes/Downwash.hpp): jets and ring
        // sources, read by the Vulkan renderer every frame once set as
        // renderer.downwash. A Source handed out by add_jet / add_rings / [i]
        // is a live reference into the list: edit its fields per frame.
        auto downwash = py::class_<Downwash, std::shared_ptr<Downwash>>(m, "Downwash",
                "Rotor downwash and ring sources for the Vulkan renderer (renderer.downwash = dw). A JET is a rotor's "
                "wake: the centre of its rotor plane (world), a thrust (N) and an effective disc radius (m); its "
                "strength follows momentum theory, v_i = sqrt(T / (2 rho A)), relative to the reference rotor (an M350: "
                "90 N through a 1 m disc), and fades with the rotor's height over the struck surface (full under 2 m, "
                "none from 9 m). RINGS leave a point on the water one after another at a speed (a stone, a fish, a "
                "creature surfacing). Up to 8 sources. Readers: a MeshStandardMaterial with ripples > 0 (concentric "
                "rings on its normal, a rougher struck disc), every GrassMesh (blades lean out and are pressed down), "
                "Renderer-owned ParticleField emitters (a hollow blown under the jet). Host-owned particle fields "
                "take the same warp through displace().");
        py::enum_<Downwash::Kind>(downwash, "Kind")
                .value("Jet", Downwash::Kind::Jet)
                .value("Rings", Downwash::Kind::Rings);
        py::class_<Downwash::Source>(downwash, "Source",
                                     "One source; a live reference into its Downwash. Jet fields: center (rotor plane "
                                     "centre), thrust, strength, radius, lobes, lobe_phase, surface_y. Ring fields: "
                                     "center, start_time, speed, spacing, count, amplitude, start_radius, wavelength.")
                .def_readonly("kind", &Downwash::Source::kind)
                .def_readwrite("enabled", &Downwash::Source::enabled)
                .def_readwrite("center", &Downwash::Source::center,
                               "Jet: the centre of the rotor plane, world. Rings: their centre (y unused).")
                .def_readwrite("thrust", &Downwash::Source::thrust, "Jet: N, all the rotors it stands for.")
                .def_readwrite("strength", &Downwash::Source::strength,
                               "Jet: >= 0 sets the share directly (1 = the reference rotor) and thrust is unused; "
                               "negative (default) = from thrust.")
                .def_readwrite("radius", &Downwash::Source::radius, "Jet: effective disc radius, m.")
                .def_readwrite("lobes", &Downwash::Source::lobes, "Jet: the swell's lobes on the water; 0 = round.")
                .def_readwrite("lobe_phase", &Downwash::Source::lobePhase, "Jet: rad.")
                .def_readwrite("surface_y", &Downwash::Source::surfaceY,
                               "Jet: the struck surface's height for the particles (and displace()); NaN (default) "
                               "= each particle's own height. The water and the grass use their own.")
                .def_readwrite("start_time", &Downwash::Source::startTime,
                               "Rings: renderer time (sim_time) the first ring leaves.")
                .def_readwrite("speed", &Downwash::Source::speed, "Rings: m/s outward.")
                .def_readwrite("spacing", &Downwash::Source::spacing, "Rings: m between successive rings.")
                .def_readwrite("count", &Downwash::Source::count, "Rings: how many.")
                .def_readwrite("amplitude", &Downwash::Source::amplitude, "Rings: m, the first ring's height.")
                .def_readwrite("start_radius", &Downwash::Source::startRadius, "Rings: m, where a ring is born.")
                .def_readwrite("wavelength", &Downwash::Source::wavelength, "Rings: m, a ring's crest train.");
        downwash.def(py::init([] { return Downwash::create(); }))
                .def("add_jet", &Downwash::addJet, py::arg("center"), py::arg("thrust") = Downwash::kReferenceThrust,
                     py::arg("radius") = Downwash::kReferenceRadius, py::return_value_policy::reference_internal,
                     "A jet at `center` (the rotor plane's centre); returns its live Source.")
                .def("add_rings", &Downwash::addRings, py::arg("center"), py::arg("start_time"),
                     py::arg("speed") = 0.32f, py::arg("spacing") = 0.29f, py::arg("count") = 6,
                     py::arg("amplitude") = 0.004f, py::return_value_policy::reference_internal,
                     "Rings leaving `center` from `start_time` (renderer time); returns the live Source.")
                .def("__len__", &Downwash::size)
                .def("__getitem__", [](Downwash& d, std::size_t i) -> Downwash::Source& { return d.source(i); },
                     py::arg("index"), py::return_value_policy::reference_internal)
                .def("remove", &Downwash::remove, py::arg("index"),
                     "Removes source `index`; the ones after it move down a place (and so do their references).")
                .def("clear", &Downwash::clear)
                .def_readwrite("ripple_amplitude", &Downwash::rippleAmplitude, "m, the jet's rings at share 1.")
                .def_readwrite("ring_wavelength", &Downwash::ringWavelength, "m, the jet's shortest ring.")
                .def_readwrite("ring_wavelength_growth", &Downwash::ringWavelengthGrowth,
                               "m of ring wavelength per m of radius.")
                .def_readwrite("ring_frequency", &Downwash::ringFrequency,
                               "Wavelengths a second the jet's rings travel outward.")
                .def_readwrite("roughness_boost", &Downwash::roughnessBoost,
                               "Added to the water's roughness in the struck disc.")
                .def_readwrite("lean_max", &Downwash::leanMax, "rad, a blade's lean at the wall jet's peak.")
                .def_readwrite("press_max", &Downwash::pressMax, "Share of a blade's height pressed down under the jet.")
                .def_readwrite("flutter", &Downwash::flutter, "The blades' beat in the jet, a share of the lean.")
                .def_readwrite("hollow_radius", &Downwash::hollowRadius,
                               "m, the hollow the jet blows in a particle cloud at share 1.")
                .def_readwrite("rim_lift", &Downwash::rimLift, "The hollow's rim lifted, a share of its radius.")
                .def_static("induced_velocity", &Downwash::inducedVelocity, py::arg("thrust"), py::arg("radius"),
                            py::arg("air_density") = Downwash::kAirDensity,
                            "Momentum theory's induced velocity sqrt(T / (2 rho pi R^2)), m/s.")
                .def_static("jet_share", &Downwash::jetShare, py::arg("source"),
                            "A jet's share before the height fade (1 = the reference rotor), clamped to [0, 3].")
                .def("warp", &Downwash::warp, py::arg("point"), py::arg("jitter") = 1.f,
                     "The particle hollow on one world point, as the Renderer-owned emitters apply it. `jitter` in "
                     "[0, 1] is the particle's own share of the hollow's radius (0.45 + 0.55 jitter).")
                .def("displace",
                     [](const Downwash& d, py::array_t<float, py::array::c_style> pts) {
                         if (pts.ndim() != 2 || (pts.shape(1) != 3 && pts.shape(1) != 4))
                             throw std::runtime_error("displace: expected an (n, 3) or (n, 4) float32 array");
                         if (!pts.writeable()) throw std::runtime_error("displace: the array is read-only");
                         d.displace(pts.mutable_data(), static_cast<std::size_t>(pts.shape(0)),
                                    static_cast<std::size_t>(pts.shape(1)));
                     },
                     py::arg("points").noconvert(),
                     "The particle hollow over host-owned positions, in place: an (n, 3) or (n, 4) C-contiguous "
                     "float32 array of WORLD points (a HostRing field's buffer before submit). The renderer warps "
                     "only the particles its own emitters integrate; this is the same warp for the rest. Each "
                     "row's share of the hollow is a hash of its index: keep a particle in its row.");

        // ---- Points ----------------------------------------------------------
        auto points = py::class_<Points, Object3D, std::shared_ptr<Points>>(m, "Points");
        bind_object3d_api(points);
        points.def(py::init([](std::shared_ptr<BufferGeometry> g, const py::object& matObj) {
                    if (!g) g = BufferGeometry::create();
                    auto mat = as_material(matObj);
                    if (!mat) mat = PointsMaterial::create();
                    return Points::create(std::move(g), std::move(mat));
                }),
                   py::arg("geometry") = std::shared_ptr<BufferGeometry>{}, py::arg("material") = py::none())
                .def_property_readonly("geometry", &Points::geometry)
                .def_property_readonly("material", [](Points& p) { return PyMaterial(material_to_py(p.material())); });

        // ---- Line / LineSegments ---------------------------------------------
        auto line = py::class_<Line, Object3D, std::shared_ptr<Line>>(m, "Line");
        bind_object3d_api(line);
        line.def(py::init([](std::shared_ptr<BufferGeometry> g, const py::object& mat) {
                    return Line::create(std::move(g), as_material(mat));
                }),
                 py::arg("geometry") = std::shared_ptr<BufferGeometry>{}, py::arg("material") = py::none())
                .def_property_readonly("geometry", &Line::geometry)
                .def_property_readonly("material", [](Line& l) { return PyMaterial(material_to_py(l.material())); })
                .def("compute_line_distances", &Line::computeLineDistances);

        // LineSegments inherits Line's concrete Object3D API (non-virtual base).
        auto lineSeg = py::class_<LineSegments, Line, std::shared_ptr<LineSegments>>(m, "LineSegments");
        lineSeg.def(py::init([](std::shared_ptr<BufferGeometry> g, const py::object& mat) {
                    return LineSegments::create(std::move(g), as_material(mat));
                }),
                    py::arg("geometry") = std::shared_ptr<BufferGeometry>{}, py::arg("material") = py::none());

        // ---- Sprite ----------------------------------------------------------
        py::class_<Sprite, Object3D, std::shared_ptr<Sprite>>(m, "Sprite")
                .def(py::init([](std::shared_ptr<SpriteMaterial> mat) {
                    return Sprite::create(std::move(mat));
                }),
                     py::arg("material") = std::shared_ptr<SpriteMaterial>{})
                .def_readwrite("center", &Sprite::center)
                .def_readwrite("screen_space", &Sprite::screenSpace)
                .def_readwrite("screen_anchor", &Sprite::screenAnchor)
                .def_property_readonly("material", [](Sprite& s) { return PyMaterial(material_to_py(s.material())); });

        // ---- Background / Fog ------------------------------------------------
        py::class_<Background>(m, "Background")
                .def(py::init<int>(), py::arg("color"))
                .def(py::init<const Color&>(), py::arg("color"))
                .def(py::init<const std::shared_ptr<Texture>&>(), py::arg("texture"))
                .def("is_color", &Background::isColor)
                .def("is_texture", &Background::isTexture);
        py::implicitly_convertible<int, Background>();
        py::implicitly_convertible<Color, Background>();

        py::class_<Fog>(m, "Fog")
                .def(py::init<const Color&, float, float>(), py::arg("color"), py::arg("near") = 1.f, py::arg("far") = 1000.f)
                .def_readwrite("color", &Fog::color)
                .def_readwrite("near", &Fog::nearPlane)
                .def_readwrite("far", &Fog::farPlane);

        py::class_<FogExp2>(m, "FogExp2")
                .def(py::init<const Color&, float>(), py::arg("color"), py::arg("density") = 0.00025f,
                     "Exponential (Beer-Lambert) participating-media fog. density = σ_t (extinction per metre). "
                     "On Vulkan this drives full volumetric single-scattering; on GL it falls back to GL_EXP2.")
                .def_readwrite("color",   &FogExp2::color)
                .def_readwrite("density", &FogExp2::density);

        // ---- Scene -----------------------------------------------------------
        py::class_<Scene, Object3D, std::shared_ptr<Scene>>(m, "Scene")
                .def(py::init(&Scene::create))
                // Accepts a hex int / Color (solid color) or a Texture (e.g. an
                // equirect HDR from RGBELoader) for an image backdrop.
                .def_property("background",
                              [](Scene& s) { return s.background; },
                              [](Scene& s, const py::object& v) {
                                  if (py::isinstance<Texture>(v)) {
                                      s.background = Background(v.cast<std::shared_ptr<Texture>>());
                                  } else {
                                      s.background = v.cast<Background>();// int / Color / Background
                                  }
                              })
                // The image-based-lighting environment map (an equirect HDR Texture
                // from RGBELoader). Drives IBL on standard/physical materials; the GL
                // renderer PMREM-prefilters it. None to clear.
                .def_readwrite("environment", &Scene::environment)
                .def_readwrite("override_material", &Scene::overrideMaterial)
                .def_readwrite("auto_update", &Scene::autoUpdate)
                // three.js background/environment knobs (GL only; Vulkan ignores
                // them). The rotations are Euler values mutable in place, like
                // Object3D.rotation: scene.environment_rotation.y = 0.5 works.
                .def_readwrite("background_blurriness", &Scene::backgroundBlurriness,
                               "Blur of a texture background, 0 (sharp) .. 1 (fully diffuse).")
                .def_readwrite("background_intensity", &Scene::backgroundIntensity,
                               "Linear multiplier on a texture background (not a Color one).")
                .def_readwrite("background_rotation", &Scene::backgroundRotation)
                .def_readwrite("environment_intensity", &Scene::environmentIntensity,
                               "Multiplier on the image-based light of every material lit by `environment`.")
                .def_readwrite("environment_rotation", &Scene::environmentRotation)
                // Convenience: linear distance fog. (scene.fog is a std::variant
                // under the hood; this avoids exposing the variant to Python.)
                .def("set_fog", [](Scene& s, const Color& c, float near, float far) { s.fog = Fog(c, near, far); },
                     py::arg("color"), py::arg("near") = 1.f, py::arg("far") = 1000.f)
                .def("set_fog_exp2", [](Scene& s, const Color& c, float density) { s.fog = FogExp2(c, density); },
                     py::arg("color"), py::arg("density") = 0.02f,
                     "Exponential participating-media fog. On Vulkan: Beer-Lambert + volumetric scattering. "
                     "Call renderer.fog_anisotropy to tune the Henyey-Greenstein phase (0 = isotropic, "
                     "+0.9 = forward god-rays, -0.9 = back-scatter halo).")
                .def("clear_fog", [](Scene& s) { s.fog.reset(); });
    }

}// namespace threepp_py
