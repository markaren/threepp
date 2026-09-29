// Lights. All derive from Light (color + intensity) which derives from Object3D
// (position comes from the core bindings). Factory `create(...)` takes an
// optional intensity in C++; here it is exposed as a plain float defaulting to 1.
#include "bindings.hpp"

#include "threepp/cameras/OrthographicCamera.hpp"
#include "threepp/lights/light_interfaces.hpp"
#include "threepp/lights/LightShadow.hpp"
#include "threepp/lights/lights.hpp"
#include "threepp/math/MathUtils.hpp"

using namespace threepp;

namespace threepp_py {

    void init_lights(py::module_& m) {

        // ---- Light (abstract base) ------------------------------------------
        py::class_<Light, Object3D, std::shared_ptr<Light>>(m, "Light")
                .def_readwrite("color", &Light::color)
                .def_readwrite("intensity", &Light::intensity);

        // ---- LightShadow -----------------------------------------------------
        // The shadow of a Directional/Point/SpotLight, reached as `light.shadow`
        // (owned by the light; no constructor). Only drawn when the renderer's
        // shadow_map_enabled is on and the light's cast_shadow is set.
        py::class_<LightShadow, std::shared_ptr<LightShadow>>(m, "LightShadow")
                .def_readwrite("bias", &LightShadow::bias)
                .def_readwrite("normal_bias", &LightShadow::normalBias)
                .def_readwrite("radius", &LightShadow::radius,
                               "Penumbra width in shadow-map texels (PCF: a soft Vogel-disk filter; VSM: the blur).")
                .def_readwrite("intensity", &LightShadow::intensity,
                               "How much light a fully shadowed receiver loses: 1 a hard shadow, 0 none.")
                .def_readwrite("map_size", &LightShadow::mapSize,
                               "Shadow-map resolution in texels (Vector2, default 2048 x 2048). Set before the first render.")
                .def_readwrite("auto_update", &LightShadow::autoUpdate)
                .def_readwrite("needs_update", &LightShadow::needsUpdate);

        // ---- AmbientLight ----------------------------------------------------
        py::class_<AmbientLight, Light, std::shared_ptr<AmbientLight>>(m, "AmbientLight")
                .def(py::init([](const Color& color, float intensity) {
                    return AmbientLight::create(color, intensity);
                }),
                     py::arg("color") = Color(0xffffff), py::arg("intensity") = 1.f);

        // ---- DirectionalLight ------------------------------------------------
        py::class_<DirectionalLight, Light, std::shared_ptr<DirectionalLight>>(m, "DirectionalLight")
                .def_property_readonly("shadow", [](DirectionalLight& l) { return l.shadow; },
                                       "The light's LightShadow (bias, radius, intensity, map_size).")
                .def(py::init([](const Color& color, float intensity) {
                    return DirectionalLight::create(color, intensity);
                }),
                     py::arg("color") = Color(0xffffff), py::arg("intensity") = 1.f)
                .def("set_target", [](DirectionalLight& l, Object3D& target) { l.setTarget(target); }, py::arg("target"))
                // Non-owning view of the aim target. target() may return the light's
                // internal defaultTarget (not shared_ptr-owned), so use the reference
                // policy (cf. Object3D.parent) — the returned target must not outlive
                // the light.
                .def("get_target", [](DirectionalLight& l) -> const Object3D& { return l.target(); }, py::return_value_policy::reference)
                // Shadow-camera ortho frustum. Call after constructing the light to
                // widen the shadow coverage area; default is ±1 (very tight).
                .def("set_shadow_frustum",
                     [](DirectionalLight& l, float left, float right, float top, float bottom) {
                         auto* cam = dynamic_cast<OrthographicCamera*>(l.shadow->camera.get());
                         if (!cam) return;
                         cam->left   = left;
                         cam->right  = right;
                         cam->top    = top;
                         cam->bottom = bottom;
                         cam->updateProjectionMatrix();
                     },
                     py::arg("left") = -10.f, py::arg("right") = 10.f,
                     py::arg("top") = 10.f, py::arg("bottom") = -10.f,
                     "Resize the directional-light shadow ortho frustum. "
                     "Call after creation; default ±1 clips almost everything.")
                .def("set_shadow_bias",
                     [](DirectionalLight& l, float bias) { l.shadow->bias = bias; },
                     py::arg("bias") = 0.f);

        // ---- PointLight ------------------------------------------------------
        py::class_<PointLight, Light, std::shared_ptr<PointLight>>(m, "PointLight")
                .def_property_readonly("shadow", [](PointLight& l) { return l.shadow; },
                                       "The light's LightShadow (bias, radius, intensity, map_size).")
                .def(py::init([](const Color& color, float intensity, float distance, float decay) {
                    return PointLight::create(color, intensity, distance, decay);
                }),
                     py::arg("color") = Color(0xffffff), py::arg("intensity") = 1.f,
                     py::arg("distance") = 0.f, py::arg("decay") = 1.f)
                .def_readwrite("distance", &PointLight::distance)
                .def_readwrite("decay", &PointLight::decay)
                .def_readwrite("radius", &PointLight::radius,
                               "Physical source radius (world units). Above 0 the Vulkan renderer "
                               "traces soft shadows with penumbras that widen with distance; 0 "
                               "(default) keeps hard shadows. GL ignores it.")
                .def("get_power", &PointLight::getPower)
                .def("set_power", &PointLight::setPower, py::arg("power"));

        // ---- SpotLight -------------------------------------------------------
        py::class_<SpotLight, Light, std::shared_ptr<SpotLight>>(m, "SpotLight")
                .def_property_readonly("shadow", [](SpotLight& l) { return l.shadow; },
                                       "The light's LightShadow (bias, radius, intensity, map_size).")
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
                .def_readwrite("radius", &SpotLight::radius,
                               "Physical source radius (world units). Above 0 the Vulkan renderer "
                               "traces soft shadows with penumbras that widen with distance; 0 "
                               "(default) keeps hard shadows. GL ignores it.")
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
