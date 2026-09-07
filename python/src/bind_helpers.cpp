// Visual scene helpers: axes, grids, bounding-box visualisers, camera frustum,
// skeleton overlay, arrow, and per-light gizmos.
#include "bindings.hpp"

#include <pybind11/stl.h>

#include "threepp/cameras/Camera.hpp"
#include "threepp/helpers/ArrowHelper.hpp"
#include "threepp/helpers/AxesHelper.hpp"
#include "threepp/helpers/Box3Helper.hpp"
#include "threepp/helpers/BoxHelper.hpp"
#include "threepp/helpers/CameraHelper.hpp"
#include "threepp/helpers/DirectionalLightHelper.hpp"
#include "threepp/helpers/GridHelper.hpp"
#include "threepp/helpers/HemisphereLightHelper.hpp"
#include "threepp/helpers/PointLightHelper.hpp"
#include "threepp/helpers/PolarGridHelper.hpp"
#include "threepp/helpers/SkeletonHelper.hpp"
#include "threepp/helpers/SpotLightHelper.hpp"
#include "threepp/lights/DirectionalLight.hpp"
#include "threepp/lights/HemisphereLight.hpp"
#include "threepp/lights/PointLight.hpp"
#include "threepp/lights/SpotLight.hpp"
#include "threepp/math/Box3.hpp"
#include "threepp/math/Color.hpp"
#include "threepp/objects/LineSegments.hpp"
#include "threepp/objects/Mesh.hpp"

using namespace threepp;

namespace threepp_py {

    void init_helpers(py::module_& m) {

        // ---- AxesHelper ------------------------------------------------------
        py::class_<AxesHelper, LineSegments, std::shared_ptr<AxesHelper>>(m, "AxesHelper")
                .def(py::init([](float size) { return AxesHelper::create(size); }),
                     py::arg("size") = 1.f);

        // ---- GridHelper ------------------------------------------------------
        py::class_<GridHelper, LineSegments, std::shared_ptr<GridHelper>>(m, "GridHelper")
                .def(py::init([](unsigned int size, unsigned int divisions,
                                 const Color& color1, const Color& color2) {
                         return GridHelper::create(size, divisions, color1, color2);
                     }),
                     py::arg("size") = 10u, py::arg("divisions") = 10u,
                     py::arg("color1") = Color(0x444444), py::arg("color2") = Color(0x888888));

        // ---- PolarGridHelper -------------------------------------------------
        py::class_<PolarGridHelper, LineSegments, std::shared_ptr<PolarGridHelper>>(m, "PolarGridHelper")
                .def(py::init([](float radius, unsigned int sectors, unsigned int rings,
                                 unsigned int divisions, const Color& color1, const Color& color2) {
                         return PolarGridHelper::create(radius, sectors, rings, divisions, color1, color2);
                     }),
                     py::arg("radius") = 10.f, py::arg("sectors") = 16u,
                     py::arg("rings") = 8u, py::arg("divisions") = 64u,
                     py::arg("color1") = Color(0x444444), py::arg("color2") = Color(0x888888));

        // ---- Box3Helper ------------------------------------------------------
        // Box3Helper stores a const Box3& — keep the Python Box3 alive with keep_alive.
        py::class_<Box3Helper, LineSegments, std::shared_ptr<Box3Helper>>(m, "Box3Helper")
                .def(py::init([](const Box3& box, const Color& color) {
                         return Box3Helper::create(box, color);
                     }),
                     py::arg("box"), py::arg("color") = Color(0xffff00),
                     py::keep_alive<1, 2>());

        // ---- BoxHelper -------------------------------------------------------
        py::class_<BoxHelper, LineSegments, std::shared_ptr<BoxHelper>>(m, "BoxHelper")
                .def(py::init([](const py::handle& obj, const Color& color) {
                         return BoxHelper::create(*as_object3d(obj), color);
                     }),
                     py::arg("object"), py::arg("color") = Color(0xffff00),
                     py::keep_alive<1, 2>())
                .def("update", &BoxHelper::update)
                .def("set_from_object", [](BoxHelper& self, const py::handle& obj) -> BoxHelper& {
                         return self.setFromObject(*as_object3d(obj));
                     }, py::arg("object"), py::return_value_policy::reference);

        // ---- ArrowHelper -----------------------------------------------------
        py::class_<ArrowHelper, Object3D, std::shared_ptr<ArrowHelper>>(m, "ArrowHelper")
                .def(py::init([](const Vector3& dir, const Vector3& origin, float length,
                                 const Color& color, std::optional<float> headLength,
                                 std::optional<float> headWidth) {
                         return ArrowHelper::create(dir, origin, length, color, headLength, headWidth);
                     }),
                     py::arg("dir") = Vector3(0, 0, 1), py::arg("origin") = Vector3(0, 0, 0),
                     py::arg("length") = 1.f, py::arg("color") = Color(0xffff00),
                     py::arg("head_length") = py::none(), py::arg("head_width") = py::none())
                .def("set_direction", &ArrowHelper::setDirection, py::arg("dir"))
                .def("set_length", &ArrowHelper::setLength,
                     py::arg("length"), py::arg("head_length") = py::none(), py::arg("head_width") = py::none())
                .def("set_color", &ArrowHelper::setColor, py::arg("color"));

        // ---- CameraHelper ----------------------------------------------------
        py::class_<CameraHelper, LineSegments, std::shared_ptr<CameraHelper>>(m, "CameraHelper")
                .def(py::init([](Camera& camera) { return CameraHelper::create(camera); }),
                     py::arg("camera"), py::keep_alive<1, 2>())
                .def("update", &CameraHelper::update);

        // ---- SkeletonHelper --------------------------------------------------
        py::class_<SkeletonHelper, LineSegments, std::shared_ptr<SkeletonHelper>>(m, "SkeletonHelper")
                .def(py::init([](const py::handle& root) {
                         return SkeletonHelper::create(*as_object3d(root));
                     }),
                     py::arg("skeleton"), py::keep_alive<1, 2>());

        // ---- DirectionalLightHelper ------------------------------------------
        py::class_<DirectionalLightHelper, Object3D, std::shared_ptr<DirectionalLightHelper>>(m, "DirectionalLightHelper")
                .def(py::init([](DirectionalLight& light, float size, std::optional<Color> color) {
                         return DirectionalLightHelper::create(light, size, color);
                     }),
                     py::arg("light"), py::arg("size") = 1.f, py::arg("color") = py::none(),
                     py::keep_alive<1, 2>())
                .def("update", &DirectionalLightHelper::update);

        // ---- HemisphereLightHelper -------------------------------------------
        py::class_<HemisphereLightHelper, Object3D, std::shared_ptr<HemisphereLightHelper>>(m, "HemisphereLightHelper")
                .def(py::init([](HemisphereLight& light, float size, std::optional<Color> color) {
                         return HemisphereLightHelper::create(light, size, color);
                     }),
                     py::arg("light"), py::arg("size") = 1.f, py::arg("color") = py::none(),
                     py::keep_alive<1, 2>())
                .def_readwrite("color", &HemisphereLightHelper::color)
                .def("update", &HemisphereLightHelper::update);

        // ---- PointLightHelper ------------------------------------------------
        py::class_<PointLightHelper, Mesh, std::shared_ptr<PointLightHelper>>(m, "PointLightHelper")
                .def(py::init([](PointLight& light, float sphere_size, std::optional<Color> color) {
                         return PointLightHelper::create(light, sphere_size, color);
                     }),
                     py::arg("light"), py::arg("sphere_size") = 1.f, py::arg("color") = py::none(),
                     py::keep_alive<1, 2>())
                .def("update", &PointLightHelper::update);

        // ---- SpotLightHelper -------------------------------------------------
        py::class_<SpotLightHelper, Object3D, std::shared_ptr<SpotLightHelper>>(m, "SpotLightHelper")
                .def(py::init([](SpotLight& light, std::optional<Color> color) {
                         return SpotLightHelper::create(light, color);
                     }),
                     py::arg("light"), py::arg("color") = py::none(),
                     py::keep_alive<1, 2>())
                .def("update", &SpotLightHelper::update);
    }

}// namespace threepp_py
