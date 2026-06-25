// Lights. All derive from Light (color + intensity) which derives from Object3D
// (position comes from the core bindings). Factory `create(...)` takes an
// optional intensity in C++; here it is exposed as a plain float defaulting to 1.
#include "bindings.hpp"

#include "threepp/lights/lights.hpp"
#include "threepp/math/MathUtils.hpp"

using namespace threepp;

namespace threepp_py {

    void init_lights(py::module_& m) {

        // ---- Light (abstract base) ------------------------------------------
        py::class_<Light, Object3D, std::shared_ptr<Light>>(m, "Light")
                .def_readwrite("color", &Light::color)
                .def_readwrite("intensity", &Light::intensity);

        // ---- AmbientLight ----------------------------------------------------
        py::class_<AmbientLight, Light, std::shared_ptr<AmbientLight>>(m, "AmbientLight")
                .def(py::init([](const Color& color, float intensity) {
                    return AmbientLight::create(color, intensity);
                }),
                     py::arg("color") = Color(0xffffff), py::arg("intensity") = 1.f);

        // ---- DirectionalLight ------------------------------------------------
        py::class_<DirectionalLight, Light, std::shared_ptr<DirectionalLight>>(m, "DirectionalLight")
                .def(py::init([](const Color& color, float intensity) {
                    return DirectionalLight::create(color, intensity);
                }),
                     py::arg("color") = Color(0xffffff), py::arg("intensity") = 1.f)
                .def("set_target", [](DirectionalLight& l, Object3D& target) { l.setTarget(target); }, py::arg("target"))
                // Non-owning view of the aim target. target() may return the light's
                // internal defaultTarget (not shared_ptr-owned), so use the reference
                // policy (cf. Object3D.parent) — the returned target must not outlive
                // the light.
                .def("get_target", [](DirectionalLight& l) -> const Object3D& { return l.target(); }, py::return_value_policy::reference);

        // ---- PointLight ------------------------------------------------------
        py::class_<PointLight, Light, std::shared_ptr<PointLight>>(m, "PointLight")
                .def(py::init([](const Color& color, float intensity, float distance, float decay) {
                    return PointLight::create(color, intensity, distance, decay);
                }),
                     py::arg("color") = Color(0xffffff), py::arg("intensity") = 1.f,
                     py::arg("distance") = 0.f, py::arg("decay") = 1.f)
                .def_readwrite("distance", &PointLight::distance)
                .def_readwrite("decay", &PointLight::decay)
                .def("get_power", &PointLight::getPower)
                .def("set_power", &PointLight::setPower, py::arg("power"));

        // ---- SpotLight -------------------------------------------------------
        py::class_<SpotLight, Light, std::shared_ptr<SpotLight>>(m, "SpotLight")
                .def(py::init([](const Color& color, float intensity, float distance, float angle, float penumbra, float decay) {
                    return SpotLight::create(color, intensity, distance, angle, penumbra, decay);
                }),
                     py::arg("color") = Color(0xffffff), py::arg("intensity") = 1.f,
                     py::arg("distance") = 0.f, py::arg("angle") = math::PI / 3.f,
                     py::arg("penumbra") = 0.f, py::arg("decay") = 1.f)
                .def_readwrite("distance", &SpotLight::distance)
                .def_readwrite("angle", &SpotLight::angle)
                .def_readwrite("penumbra", &SpotLight::penumbra)
                .def_readwrite("decay", &SpotLight::decay)
                .def("set_target", [](SpotLight& l, Object3D& target) { l.setTarget(target); }, py::arg("target"))
                .def("get_target", [](SpotLight& l) -> const Object3D& { return l.target(); }, py::return_value_policy::reference);

        // ---- HemisphereLight -------------------------------------------------
        py::class_<HemisphereLight, Light, std::shared_ptr<HemisphereLight>>(m, "HemisphereLight")
                .def(py::init([](const Color& skyColor, const Color& groundColor, float intensity) {
                    return HemisphereLight::create(skyColor, groundColor, intensity);
                }),
                     py::arg("sky_color") = Color(0xffffff), py::arg("ground_color") = Color(0xffffff),
                     py::arg("intensity") = 1.f)
                .def_readwrite("ground_color", &HemisphereLight::groundColor);

        // ---- RectAreaLight ---------------------------------------------------
        py::class_<RectAreaLight, Light, std::shared_ptr<RectAreaLight>>(m, "RectAreaLight")
                .def(py::init([](const Color& color, float intensity, float width, float height) {
                    return RectAreaLight::create(color, intensity, width, height);
                }),
                     py::arg("color") = Color(0xffffff), py::arg("intensity") = 1.f,
                     py::arg("width") = 1.f, py::arg("height") = 1.f)
                .def_readonly("width", &RectAreaLight::width)
                .def_readonly("height", &RectAreaLight::height);
    }

}// namespace threepp_py
