// Core scene-graph plumbing: Object3D (the base every renderable derives from),
// the BufferGeometry base, and Clock.
#include "bindings.hpp"

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "threepp/constants.hpp"
#include "threepp/core/BufferAttribute.hpp"
#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Clock.hpp"
#include "threepp/core/Layers.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"

#include <algorithm>
#include <any>
#include <cstdint>
#include <optional>

using namespace threepp;

namespace threepp_py {

    // pybind11 cannot up-cast a Python object to shared_ptr<Object3D> across
    // threepp's `virtual Object3D` base (Mesh/Points/Line and their subclasses)
    // — it assumes a constant base offset and corrupts the pointer. Work around
    // it by casting to the exact virtual-base-root shared_ptr (a non-virtual,
    // safe cast for any further-derived type) and letting the C++ compiler do
    // the virtual up-cast. Non-virtual Object3D subclasses (Group, Scene,
    // cameras, lights, Sprite) convert directly.
    std::shared_ptr<Object3D> as_object3d(const py::handle& h) {
        if (py::isinstance<Mesh>(h)) return h.cast<std::shared_ptr<Mesh>>();
        if (py::isinstance<Points>(h)) return h.cast<std::shared_ptr<Points>>();
        if (py::isinstance<Line>(h)) return h.cast<std::shared_ptr<Line>>();
        return h.cast<std::shared_ptr<Object3D>>();
    }

    // String entries only: userData holds std::any, and the editor's configs
    // (spline, physics, script) are all flat `key=value;...` strings — the one
    // shape a script has a documented reason to read. Non-string scalars return
    // None rather than a guessed conversion.
    py::object user_data_string(const Object3D& o, const std::string& key) {
        const auto it = o.userData.find(key);
        if (it == o.userData.end()) return py::none();
        if (it->second.type() == typeid(std::string)) {
            return py::str(std::any_cast<const std::string&>(it->second));
        }
        if (it->second.type() == typeid(const char*)) {
            return py::str(std::any_cast<const char*>(it->second));
        }
        return py::none();
    }

    // An empty value removes the entry rather than storing "", because that is
    // what every editor config means by off: PhysicsConfig::write erases when
    // the body is disabled, and a document with `physics=""` on it would be a
    // document carrying a config nothing can decode.
    void set_user_data_string(Object3D& o, const std::string& key, const std::string& value) {
        if (value.empty()) {
            o.userData.erase(key);
            return;
        }
        o.userData[key] = value;
    }

    void init_core(py::module_& m) {

        // ---- Layers ----------------------------------------------------------
        // 32-channel visibility mask: an object is drawn by a camera only if their
        // masks share a channel. Everything starts on channel 0.
        py::class_<Layers>(m, "Layers")
                .def(py::init<>())
                .def("set", &Layers::set, py::arg("channel"), "Membership of exactly this one channel.")
                .def("enable", &Layers::enable, py::arg("channel"))
                .def("enable_all", &Layers::enableAll)
                .def("toggle", &Layers::toggle, py::arg("channel"))
                .def("disable", &Layers::disable, py::arg("channel"))
                .def("disable_all", &Layers::disableAll)
                .def("test", &Layers::test, py::arg("layers"), "True if the two masks share a channel.")
                .def("is_enabled", &Layers::isEnabled, py::arg("channel"))
                .def("mask", &Layers::mask)
                .def("__repr__", [](const Layers& l) {
                    return "Layers(mask=" + std::to_string(l.mask()) + ")";
                });

        // ---- Object3D --------------------------------------------------------
        // Polymorphic base; registered with a shared_ptr holder so derived
        // objects (Mesh, Group, lights, cameras...) share one ownership model.
        // position/rotation/quaternion/scale are exposed with reference_internal
        // semantics (pybind's def_readwrite default for class members), so
        // `obj.position.x = 1` mutates the live member rather than a copy.
        py::class_<Object3D, std::shared_ptr<Object3D>>(m, "Object3D")
                .def(py::init(&Object3D::create))
                .def_property_readonly("id", [](const Object3D& o) { return o.id; })
                .def_property_readonly("uuid", [](const Object3D& o) { return o.uuid; })
                .def_readwrite("name", &Object3D::name)
                .def_readwrite("position", &Object3D::position)
                .def_readwrite("rotation", &Object3D::rotation)
                .def_readwrite("quaternion", &Object3D::quaternion)
                .def_readwrite("scale", &Object3D::scale)
                .def_readwrite("up", &Object3D::up)
                .def_readwrite("visible", &Object3D::visible)
                .def_readwrite("cast_shadow", &Object3D::castShadow)
                .def_readwrite("receive_shadow", &Object3D::receiveShadow)
                .def_readwrite("frustum_culled", &Object3D::frustumCulled)
                .def_readwrite("render_order", &Object3D::renderOrder)
                .def_readwrite("matrix_auto_update", &Object3D::matrixAutoUpdate)
                // Mutable in place: `obj.layers.set(1)` moves the object to channel 1.
                .def_readwrite("layers", &Object3D::layers)
                .def_property_readonly("parent", [](Object3D& o) { return o.parent; }, py::return_value_policy::reference)
                .def_property_readonly("children", [](Object3D& o) { return o.children; }, py::return_value_policy::reference)
                // add(child) / add(a, b, c) — takes shared ownership of each child.
                .def("add", [](Object3D& self, const py::args& children) {
                    for (const auto& c : children) self.add(as_object3d(c));
                })
                .def("remove", [](Object3D& self, const py::handle& child) { self.remove(*as_object3d(child)); }, py::arg("object"))
                // NB: Object3D methods are bound as lambdas taking `Object3D&`
                // rather than `&Object3D::method`. pybind11 mishandles direct
                // member/method pointers across threepp's virtual Object3D base
                // (crashes on Mesh/Points/Line); a lambda forces a correct
                // base-reference load.
                .def("remove_from_parent", [](Object3D& o) { o.removeFromParent(); })
                .def("clear", [](Object3D& o) { o.clear(); })
                .def("rotate_x", [](Object3D& o, float a) { o.rotateX(a); }, py::arg("angle"))
                .def("rotate_y", [](Object3D& o, float a) { o.rotateY(a); }, py::arg("angle"))
                .def("rotate_z", [](Object3D& o, float a) { o.rotateZ(a); }, py::arg("angle"))
                .def("rotate_on_axis", [](Object3D& o, const Vector3& axis, float a) { o.rotateOnAxis(axis, a); }, py::arg("axis"), py::arg("angle"))
                .def("translate_x", [](Object3D& o, float d) { o.translateX(d); }, py::arg("distance"))
                .def("translate_y", [](Object3D& o, float d) { o.translateY(d); }, py::arg("distance"))
                .def("translate_z", [](Object3D& o, float d) { o.translateZ(d); }, py::arg("distance"))
                .def("look_at", [](Object3D& o, float x, float y, float z) { o.lookAt(x, y, z); }, py::arg("x"), py::arg("y"), py::arg("z"))
                .def("look_at", [](Object3D& o, const Vector3& v) { o.lookAt(v); }, py::arg("vector"))
                .def("get_world_position", [](Object3D& o) {
                    Vector3 v;
                    o.getWorldPosition(v);
                    return v;
                })
                .def("get_world_direction", [](Object3D& o) {
                    Vector3 v;
                    o.getWorldDirection(v);
                    return v;
                })
                // World-space pose (all refresh the world matrix first, so they read
                // correctly without a manual update). get_world_quaternion + get_world_position
                // give 6-DoF ground truth; matrix_world is the full (camera-to-world)
                // extrinsics — use .to_numpy() for a (4,4). local<->world transform points.
                .def("get_world_quaternion", [](Object3D& o) {
                    Quaternion q;
                    o.getWorldQuaternion(q);
                    return q;
                })
                .def("get_world_scale", [](Object3D& o) {
                    Vector3 v;
                    o.getWorldScale(v);
                    return v;
                })
                .def_property_readonly("matrix_world", [](Object3D& o) {
                    o.updateWorldMatrix(true, false);
                    return *o.matrixWorld;
                })
                .def("local_to_world", [](Object3D& o, Vector3 v) { o.localToWorld(v); return v; }, py::arg("vector"))
                .def("world_to_local", [](Object3D& o, Vector3 v) { o.worldToLocal(v); return v; }, py::arg("vector"))
                .def("get_object_by_name", [](Object3D& o, const std::string& name) { return o.getObjectByName(name); }, py::arg("name"), py::return_value_policy::reference)
                .def("get_user_data", [](const Object3D& o, const std::string& key) { return user_data_string(o, key); }, py::arg("key"),
                     "String userData entry for `key`, or None when absent or not a string. "
                     "The editor's spline/physics/script configs live here as flat 'key=value;...' strings.")
                .def("set_user_data", [](Object3D& o, const std::string& key, const std::string& value) { set_user_data_string(o, key, value); },
                     py::arg("key"), py::arg("value"),
                     "Set the string userData entry for `key`; an empty value removes it. "
                     "This is how a GENERATOR script authors physics, a sensor or a script onto "
                     "the content it builds - the same flat 'key=value;...' strings the "
                     "inspector writes.")
                // Pass each visited object by reference (Object3D is non-copyable)
                // and let polymorphic_type_hook hand back the concrete subclass.
                .def("traverse", [](Object3D& self, const std::function<void(py::object)>& cb) {
                    self.traverse([&cb](Object3D& o) { cb(py::cast(&o, py::return_value_policy::reference)); });
                }, py::arg("callback"))
                .def("update_matrix", [](Object3D& o) { o.updateMatrix(); })
                .def("update_matrix_world", [](Object3D& o, bool force) { o.updateMatrixWorld(force); }, py::arg("force") = false)
                // clone() dispatches through the virtual createDefault(), so it
                // produces the concrete subclass (Mesh/Group/...) even though the
                // lambda takes Object3D&; pybind downcasts the returned
                // shared_ptr<Object3D> to the right Python type. recursive (default
                // True) deep-copies the child subtree. copy() routes the source
                // through as_object3d() to dodge the virtual-base pointer bug.
                .def("clone", [](Object3D& o, bool recursive) { return o.clone(recursive); }, py::arg("recursive") = true)
                .def("copy", [](Object3D& self, const py::handle& source, bool recursive) {
                    self.copy(*as_object3d(source), recursive);
                }, py::arg("source"), py::arg("recursive") = true)
                .def("__repr__", [](const Object3D& o) { return "<threepp." + o.type() + " name='" + o.name + "'>"; });

        // ---- BufferGeometry --------------------------------------------------
        py::class_<BufferGeometry, std::shared_ptr<BufferGeometry>>(m, "BufferGeometry")
                .def(py::init(&BufferGeometry::create))
                .def_readwrite("name", &BufferGeometry::name)
                .def("compute_vertex_normals", &BufferGeometry::computeVertexNormals)
                .def("compute_bounding_box", &BufferGeometry::computeBoundingBox)
                .def("compute_bounding_sphere", &BufferGeometry::computeBoundingSphere)
                .def("translate", &BufferGeometry::translate, py::arg("x"), py::arg("y"), py::arg("z"), py::return_value_policy::reference_internal)
                .def("scale", &BufferGeometry::scale, py::arg("x"), py::arg("y"), py::arg("z"), py::return_value_policy::reference_internal)
                .def("rotate_x", &BufferGeometry::rotateX, py::arg("angle"), py::return_value_policy::reference_internal)
                .def("rotate_y", &BufferGeometry::rotateY, py::arg("angle"), py::return_value_policy::reference_internal)
                .def("rotate_z", &BufferGeometry::rotateZ, py::arg("angle"), py::return_value_policy::reference_internal)
                .def("center", &BufferGeometry::center, py::return_value_policy::reference_internal)
                // Bakes a transform into the vertex positions (normals / tangents are rotated).
                .def("apply_matrix4", &BufferGeometry::applyMatrix4, py::arg("matrix"), py::return_value_policy::reference_internal)
                .def("to_non_indexed", &BufferGeometry::toNonIndexed,
                     "A new geometry with the index resolved into duplicated vertices (a triangle soup).")
                .def("set_from_points", [](BufferGeometry& g, const std::vector<Vector3>& pts) -> BufferGeometry& { return g.setFromPoints(pts); }, py::arg("points"), py::return_value_policy::reference_internal)
                // Set/replace a float vertex attribute (e.g. "position", "color", "normal") from an
                // (N, item_size) numpy array. The attribute is marked Dynamic. Setting "position" on a
                // non-indexed geometry also resets the draw range to all N rows. Allocates a new GPU
                // buffer — for per-frame updates of a fixed-capacity cloud prefer update_attribute
                // (in place, no buffer churn).
                .def("set_attribute", [](BufferGeometry& g, const std::string& name,
                                         py::array_t<float, py::array::c_style | py::array::forcecast> data) -> BufferGeometry& {
                    if (data.ndim() != 2) throw std::runtime_error("set_attribute: expected a 2-D (N, item_size) array");
                    const int n = static_cast<int>(data.shape(0)), item = static_cast<int>(data.shape(1));
                    const float* s = data.data();
                    auto attr = FloatBufferAttribute::create(std::vector<float>(s, s + static_cast<size_t>(n) * item), item);
                    attr->setUsage(DrawUsage::Dynamic);
                    g.setAttribute(name, std::move(attr));
                    // Reset the draw range only where the rows ARE the drawn elements: a
                    // non-indexed geometry's "position". An indexed draw counts INDICES and
                    // set_index owns that range; a secondary attribute (color, normal, uv)
                    // never defines how much geometry there is. The unconditional reset here
                    // used to truncate any indexed mesh that set a per-vertex colour after
                    // its index to a vertex-count's worth of indices — a torn fraction.
                    if (name == "position" && !g.getIndex()) g.setDrawRange(0, n);
                    return g;
                }, py::arg("name"), py::arg("data"), py::return_value_policy::reference_internal)
                // Overwrite the first N rows of an existing float attribute in place (no realloc, no GPU
                // buffer churn) and flag it for re-upload. N may be <= the allocated capacity; pair with
                // set_draw_range(0, N) to render exactly the rows you wrote. Raises if the attribute is
                // missing, the item_size differs, or N exceeds the allocated capacity.
                .def("update_attribute", [](BufferGeometry& g, const std::string& name,
                                            py::array_t<float, py::array::c_style | py::array::forcecast> data) -> BufferGeometry& {
                    auto* attr = g.getAttribute<float>(name);
                    if (!attr) throw std::runtime_error("update_attribute: no attribute '" + name + "' (call set_attribute first)");
                    if (data.ndim() != 2) throw std::runtime_error("update_attribute: expected a 2-D (N, item_size) array");
                    const int n = static_cast<int>(data.shape(0)), item = static_cast<int>(data.shape(1));
                    if (item != attr->itemSize()) throw std::runtime_error("update_attribute: item_size mismatch");
                    auto& arr = attr->array();
                    const size_t need = static_cast<size_t>(n) * item;
                    if (need > arr.size()) throw std::runtime_error("update_attribute: N exceeds allocated capacity (use set_attribute to grow)");
                    std::copy(data.data(), data.data() + need, arr.begin());
                    // Publish exactly what was written so both backends upload
                    // only the touched prefix — without this a geometry
                    // preallocated at capacity pays a full-capacity upload on
                    // every edit, which is the whole point of updating in place.
                    attr->updateRange.offset = 0;
                    attr->updateRange.count  = static_cast<int>(need);
                    attr->needsUpdate();
                    return g;
                }, py::arg("name"), py::arg("data"), py::return_value_policy::reference_internal)
                // Readback. set_attribute's counterpart, and the only way to reach geometry
                // Python did not author: a model that arrived through one of the LOADERS
                // (STL, OBJ, glTF/GLB, COLLADA, Assimp) is a BufferGeometry full of triangles
                // that Python could previously write to but never read. That gap is why a
                // consumer wanting a mesh's vertices — an SDF bake, a collider, a point
                // sampler, an export — had to re-parse the file itself in Python and
                // hand-support one format at a time.
                //
                // Returns a COPY, not a view: the backing std::vector is reallocated by
                // set_attribute and freed by dispose, so a view would dangle at a distance.
                // Readback happens once at load, where a copy costs nothing worth saving.
                .def("get_attribute", [](BufferGeometry& g, const std::string& name)
                             -> std::optional<py::array_t<float>> {
                    auto* attr = g.getAttribute<float>(name);
                    if (!attr) return std::nullopt;
                    const auto& a = attr->array();
                    const auto item = static_cast<py::ssize_t>(attr->itemSize());
                    if (item <= 0) return std::nullopt;
                    const auto n = static_cast<py::ssize_t>(a.size()) / item;
                    py::array_t<float> out({n, item});
                    std::copy_n(a.begin(), static_cast<size_t>(n * item), out.mutable_data());
                    return out;
                }, py::arg("name"),
                   "Read a float attribute back as an (N, item_size) float32 array, or None if "
                   "the geometry has no attribute of that name. Returns a copy.")
                .def("get_index", [](BufferGeometry& g)
                             -> std::optional<py::array_t<std::uint32_t>> {
                    const auto* idx = g.getIndex();
                    if (!idx) return std::nullopt;
                    const auto& a = idx->array();
                    py::array_t<std::uint32_t> out(static_cast<py::ssize_t>(a.size()));
                    std::copy(a.begin(), a.end(), out.mutable_data());
                    return out;
                }, "Read the index buffer back as a flat uint32 array, or None if the geometry "
                   "is non-indexed (a triangle soup, which is what an STL always is).")
                // get_index's counterpart. Without it, geometry authored in Python could only
                // ever be a triangle soup: every shared vertex duplicated once per face it
                // touches, which for a closed surface is ~6x the vertices, and no way to give
                // two faces the same normal sample. That is a real cost for a mesh whose
                // positions are rewritten every frame — a deforming soft body pays it on every
                // upload — and it is why a smooth-shaded, UV-mapped mesh could not be driven
                // from Python at all.
                //
                // Takes any integer array (flat, or (M, 3) triangles) and stores it as uint32.
                .def("set_index", [](BufferGeometry& g,
                                     py::array_t<std::uint32_t, py::array::c_style | py::array::forcecast> data)
                             -> BufferGeometry& {
                    const auto n = static_cast<size_t>(data.size());
                    if (n % 3 != 0) throw std::runtime_error("set_index: index count must be a multiple of 3");
                    const auto* p = data.data();
                    const auto nVerts = [&] {
                        size_t most = 0;
                        for (const auto& [_, attr]: g.getAttributes())
                            most = std::max(most, static_cast<size_t>(attr->count()));
                        return most;
                    }();
                    if (nVerts > 0) {
                        const auto bad = std::find_if(p, p + n, [&](std::uint32_t i) { return i >= nVerts; });
                        if (bad != p + n)
                            throw std::runtime_error("set_index: index " + std::to_string(*bad) +
                                                     " is out of range for " + std::to_string(nVerts) +
                                                     " vertices (set the attributes first)");
                    }
                    g.setIndex(std::vector<unsigned int>(p, p + n));
                    // An indexed draw counts INDICES, not vertices (GLRenderer.cpp clamps
                    // drawRange against index->count()), and set_attribute("position") on the
                    // not-yet-indexed geometry left drawRange at the vertex count — which for
                    // a closed mesh is roughly a sixth of the indices, so the geometry would
                    // draw a torn fraction of itself. Publishing the whole index buffer here
                    // is what makes "set the attributes, then set the index" mean what it
                    // looks like; set_attribute afterwards leaves an indexed range alone.
                    g.setDrawRange(0, static_cast<int>(n));
                    return g;
                }, py::arg("data"), py::return_value_policy::reference_internal,
                   "Give this geometry an index buffer, so its vertices can be shared between "
                   "faces. Accepts a flat or (M, 3) integer array. Validated against the "
                   "vertex count of the attributes already set, so set_attribute first. "
                   "Sets the draw range to the whole index buffer; once indexed, a later "
                   "set_attribute leaves the draw range alone.")
                .def("attribute_names", [](const BufferGeometry& g) {
                    std::vector<std::string> names;
                    names.reserve(g.getAttributes().size());
                    for (const auto& [name, _]: g.getAttributes()) names.push_back(name);
                    std::sort(names.begin(), names.end());
                    return names;
                }, "Sorted names of the attributes this geometry carries.")
                .def("set_draw_range", [](BufferGeometry& g, int start, int count) -> BufferGeometry& {
                    return g.setDrawRange(start, count);
                }, py::arg("start"), py::arg("count"), py::return_value_policy::reference_internal,
                   "Render only vertices [start, start+count). Use with a fixed-capacity attribute + update_attribute.")
                .def("dispose", &BufferGeometry::dispose);

        // ---- Clock -----------------------------------------------------------
        py::class_<Clock>(m, "Clock")
                .def(py::init<bool>(), py::arg("auto_start") = true)
                .def("start", &Clock::start)
                .def("stop", &Clock::stop)
                .def("get_elapsed_time", &Clock::getElapsedTime)
                .def("get_delta", &Clock::getDelta)
                .def_readwrite("elapsed_time", &Clock::elapsedTime)
                .def_readwrite("running", &Clock::running);
    }

}// namespace threepp_py
