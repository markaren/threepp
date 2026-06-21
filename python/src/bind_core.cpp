// Core scene-graph plumbing: Object3D (the base every renderable derives from),
// the BufferGeometry base, and Clock.
#include "bindings.hpp"

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/core/Clock.hpp"
#include "threepp/core/Object3D.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"

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

    void init_core(py::module_& m) {

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
                .def("get_object_by_name", [](Object3D& o, const std::string& name) { return o.getObjectByName(name); }, py::arg("name"), py::return_value_policy::reference)
                // Pass each visited object by reference (Object3D is non-copyable)
                // and let polymorphic_type_hook hand back the concrete subclass.
                .def("traverse", [](Object3D& self, const std::function<void(py::object)>& cb) {
                    self.traverse([&cb](Object3D& o) { cb(py::cast(&o, py::return_value_policy::reference)); });
                }, py::arg("callback"))
                .def("update_matrix", [](Object3D& o) { o.updateMatrix(); })
                .def("update_matrix_world", [](Object3D& o, bool force) { o.updateMatrixWorld(force); }, py::arg("force") = false)
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
                .def("set_from_points", [](BufferGeometry& g, const std::vector<Vector3>& pts) -> BufferGeometry& { return g.setFromPoints(pts); }, py::arg("points"), py::return_value_policy::reference_internal)
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
