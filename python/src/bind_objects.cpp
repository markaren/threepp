// Renderable objects and the Scene container.
#include "bindings.hpp"

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "threepp/core/BufferGeometry.hpp"
#include "threepp/materials/Material.hpp"
#include "threepp/materials/PointsMaterial.hpp"
#include "threepp/materials/SpriteMaterial.hpp"
#include "threepp/objects/Group.hpp"
#include "threepp/objects/InstancedMesh.hpp"
#include "threepp/objects/Line.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"
#include "threepp/objects/Points.hpp"
#include "threepp/objects/Sprite.hpp"
#include "threepp/scenes/Fog.hpp"
#include "threepp/scenes/Scene.hpp"

using namespace threepp;

namespace threepp_py {

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
                .def_readwrite("render_order", &T::renderOrder)
                .def_readwrite("matrix_auto_update", &T::matrixAutoUpdate)
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
                .def("get_object_by_name", [](T& o, const std::string& n) { return o.getObjectByName(n); }, py::arg("name"), py::return_value_policy::reference)
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
                .def_property_readonly("material", [](Mesh& self) { return material_to_py(self.material()); })
                .def("set_geometry", &Mesh::setGeometry, py::arg("geometry"))
                .def("set_material", [](Mesh& self, const py::object& mat) { self.setMaterial(as_material(mat)); }, py::arg("material"));

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
                .def("get_matrix_at", [](InstancedMesh& im, size_t i) {
                    Matrix4 mtx;
                    im.getMatrixAt(i, mtx);
                    return mtx;
                }, py::arg("index"))
                .def("set_color_at", &InstancedMesh::setColorAt, py::arg("index"), py::arg("color"))
                .def("instance_matrix_needs_update", [](InstancedMesh& im) { im.instanceMatrix()->needsUpdate(); })
                .def("instance_color_needs_update", [](InstancedMesh& im) {
                    if (auto* c = im.instanceColor()) c->needsUpdate();
                });

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
                .def_property_readonly("material", [](Points& p) { return material_to_py(p.material()); });

        // ---- Line / LineSegments ---------------------------------------------
        auto line = py::class_<Line, Object3D, std::shared_ptr<Line>>(m, "Line");
        bind_object3d_api(line);
        line.def(py::init([](std::shared_ptr<BufferGeometry> g, const py::object& mat) {
                    return Line::create(std::move(g), as_material(mat));
                }),
                 py::arg("geometry") = std::shared_ptr<BufferGeometry>{}, py::arg("material") = py::none())
                .def_property_readonly("geometry", &Line::geometry)
                .def_property_readonly("material", [](Line& l) { return material_to_py(l.material()); })
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
                .def_property_readonly("material", [](Sprite& s) { return material_to_py(s.material()); });

        // ---- Background / Fog ------------------------------------------------
        py::class_<Background>(m, "Background")
                .def(py::init<int>(), py::arg("color"))
                .def(py::init<const Color&>(), py::arg("color"))
                .def("is_color", &Background::isColor)
                .def("is_texture", &Background::isTexture);
        py::implicitly_convertible<int, Background>();
        py::implicitly_convertible<Color, Background>();

        py::class_<Fog>(m, "Fog")
                .def(py::init<const Color&, float, float>(), py::arg("color"), py::arg("near") = 1.f, py::arg("far") = 1000.f)
                .def_readwrite("color", &Fog::color)
                .def_readwrite("near", &Fog::nearPlane)
                .def_readwrite("far", &Fog::farPlane);

        // ---- Scene -----------------------------------------------------------
        py::class_<Scene, Object3D, std::shared_ptr<Scene>>(m, "Scene")
                .def(py::init(&Scene::create))
                .def_readwrite("background", &Scene::background)
                .def_readwrite("override_material", &Scene::overrideMaterial)
                .def_readwrite("auto_update", &Scene::autoUpdate)
                // Convenience: linear distance fog. (scene.fog is a std::variant
                // under the hood; this avoids exposing the variant to Python.)
                .def("set_fog", [](Scene& s, const Color& c, float near, float far) { s.fog = Fog(c, near, far); },
                     py::arg("color"), py::arg("near") = 1.f, py::arg("far") = 1000.f)
                .def("clear_fog", [](Scene& s) { s.fog.reset(); });
    }

}// namespace threepp_py
