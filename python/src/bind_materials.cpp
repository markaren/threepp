// Materials. The Material base exposes the shared render-state fields; each
// concrete material adds its own scalar/color knobs. Texture-map slots
// (shared_ptr<Texture>) are intentionally omitted until the texture/loader
// layer is bound — every field here is a value type or enum.
#include "bindings.hpp"

#include "threepp/constants.hpp"
#include "threepp/materials/MeshDepthMaterial.hpp"// not aggregated by materials.hpp
#include "threepp/materials/materials.hpp"

using namespace threepp;

namespace threepp_py {

    py::object material_to_py(const std::shared_ptr<Material>& mat) {
        if (!mat) return py::none();
#define THREEPP_TRY_MAT(T)                              \
    if (auto p = std::dynamic_pointer_cast<T>(mat)) {   \
        return py::cast(p);                             \
    }
        THREEPP_TRY_MAT(MeshPhysicalMaterial)// before Standard — Physical IS-A Standard
        THREEPP_TRY_MAT(MeshStandardMaterial)
        THREEPP_TRY_MAT(MeshPhongMaterial)
        THREEPP_TRY_MAT(MeshBasicMaterial)
        THREEPP_TRY_MAT(MeshLambertMaterial)
        THREEPP_TRY_MAT(MeshNormalMaterial)
        THREEPP_TRY_MAT(MeshDepthMaterial)
        THREEPP_TRY_MAT(PointsMaterial)
        THREEPP_TRY_MAT(LineBasicMaterial)
        THREEPP_TRY_MAT(SpriteMaterial)
        THREEPP_TRY_MAT(ShadowMaterial)
#undef THREEPP_TRY_MAT
        return py::cast(mat);
    }

    std::shared_ptr<Material> as_material(const py::handle& h) {
        if (h.is_none()) return nullptr;
#define THREEPP_CAST_MAT(T)        \
    if (py::isinstance<T>(h)) {     \
        return h.cast<std::shared_ptr<T>>(); \
    }
        THREEPP_CAST_MAT(MeshPhysicalMaterial)// before Standard — Physical IS-A Standard
        THREEPP_CAST_MAT(MeshStandardMaterial)
        THREEPP_CAST_MAT(MeshPhongMaterial)
        THREEPP_CAST_MAT(MeshBasicMaterial)
        THREEPP_CAST_MAT(MeshLambertMaterial)
        THREEPP_CAST_MAT(MeshNormalMaterial)
        THREEPP_CAST_MAT(MeshDepthMaterial)
        THREEPP_CAST_MAT(PointsMaterial)
        THREEPP_CAST_MAT(LineBasicMaterial)
        THREEPP_CAST_MAT(SpriteMaterial)
        THREEPP_CAST_MAT(ShadowMaterial)
#undef THREEPP_CAST_MAT
        return h.cast<std::shared_ptr<Material>>();
    }

    // Concrete materials derive from Material *virtually*, so the same pybind11
    // virtual-base hazard as Object3D applies: accessing inherited base fields
    // through the Material base corrupts memory. Bind the shared base fields on
    // each concrete material via concrete member pointers (&T::field).
    template<class Cls>
    static void bind_material_base_fields(Cls& c) {
        using T = typename Cls::type;
        c.def_readwrite("name", &T::name)
                .def_readwrite("opacity", &T::opacity)
                .def_readwrite("transparent", &T::transparent)
                .def_readwrite("side", &T::side)
                .def_readwrite("vertex_colors", &T::vertexColors)
                .def_readwrite("depth_test", &T::depthTest)
                .def_readwrite("depth_write", &T::depthWrite)
                .def_readwrite("visible", &T::visible)
                .def_readwrite("fog", &T::fog)
                .def_readwrite("blending", &T::blending)
                .def_readwrite("alpha_test", &T::alphaTest)
                .def_readwrite("tone_mapped", &T::toneMapped)
                .def_readwrite("premultiplied_alpha", &T::premultipliedAlpha)
                // Bump the material version so backends that cache derived
                // per-material GPU state (Vulkan MaterialDesc SSBO) re-upload
                // after a runtime property edit. No-op visual cost on GL.
                .def("needs_update", [](T& mat) { mat.needsUpdate(); })
                .def("dispose", [](T& mat) { mat.dispose(); })
                .def("__repr__", [](const T& mat) { return "<threepp." + mat.type() + ">"; });
    }

    void init_materials(py::module_& m) {

        // ---- Enums -----------------------------------------------------------
        py::enum_<Side>(m, "Side")
                .value("Front", Side::Front)
                .value("Back", Side::Back)
                .value("Double", Side::Double);

        py::enum_<Blending>(m, "Blending")
                // "NoBlending" (the three.js name), not "None": a Python keyword
                // cannot be spelled as an attribute, so `tp.Blending.None` would
                // be a SyntaxError and unrepresentable in the type stubs.
                .value("NoBlending", Blending::None)
                .value("Normal", Blending::Normal)
                .value("Additive", Blending::Additive)
                .value("Subtractive", Blending::Subtractive)
                .value("Multiply", Blending::Multiply)
                .value("Custom", Blending::Custom);

        py::enum_<CombineOperation>(m, "CombineOperation")
                .value("Multiply", CombineOperation::Multiply)
                .value("Mix", CombineOperation::Mix)
                .value("Add", CombineOperation::Add);

        py::enum_<DepthPacking>(m, "DepthPacking")
                .value("Basic", DepthPacking::Basic)
                .value("RGBA", DepthPacking::RGBA);

        // ---- Material base ---------------------------------------------------
        // Abstract; never instantiated. Registered so concrete materials can
        // declare it as a pybind base (for isinstance and the as_material /
        // material_to_py shared_ptr<Material> bridge) — and carrying the shared
        // fields, so `Mesh.material` can be typed `Material | None` rather than a
        // union of every concrete class. A type checker then offers exactly the
        // fields EVERY material has, and `isinstance` narrows to the rest.
        //
        // These handlers take `py::object` and reach the fields through
        // as_material(), so the Derived -> Material step is a C++ shared_ptr
        // conversion, which resolves the VIRTUAL base correctly. Never
        // `[](Material& self)`: that asks pybind11 for the adjustment, which is
        // the corruption the note above bind_material_base_fields warns about.
        //
        // Runtime cost is nil: every concrete material binds these directly, and
        // those shadow these in the Python MRO. This is the base's *declaration*.
        auto material = py::class_<Material, std::shared_ptr<Material>>(m, "Material");
#define THREEPP_MAT_FIELD(pyname, field)                                                        \
    material.def_property(                                                                      \
            pyname,                                                                             \
            [](const py::object& self) { return as_material(self)->field; },                    \
            [](const py::object& self, decltype(Material::field) v) { as_material(self)->field = v; })
        THREEPP_MAT_FIELD("name", name);
        THREEPP_MAT_FIELD("opacity", opacity);
        THREEPP_MAT_FIELD("transparent", transparent);
        THREEPP_MAT_FIELD("side", side);
        THREEPP_MAT_FIELD("vertex_colors", vertexColors);
        THREEPP_MAT_FIELD("depth_test", depthTest);
        THREEPP_MAT_FIELD("depth_write", depthWrite);
        THREEPP_MAT_FIELD("visible", visible);
        THREEPP_MAT_FIELD("fog", fog);
        THREEPP_MAT_FIELD("blending", blending);
        THREEPP_MAT_FIELD("alpha_test", alphaTest);
        THREEPP_MAT_FIELD("tone_mapped", toneMapped);
        THREEPP_MAT_FIELD("premultiplied_alpha", premultipliedAlpha);
#undef THREEPP_MAT_FIELD
        material.def("needs_update", [](const py::object& self) { as_material(self)->needsUpdate(); })
                .def("dispose", [](const py::object& self) { as_material(self)->dispose(); })
                .def("__repr__", [](const py::object& self) { return "<threepp." + as_material(self)->type() + ">"; });

        // ---- MeshBasicMaterial ----------------------------------------------
        auto basic = py::class_<MeshBasicMaterial, Material, std::shared_ptr<MeshBasicMaterial>>(m, "MeshBasicMaterial");
        bind_material_base_fields(basic);
        basic.def(py::init([] { return MeshBasicMaterial::create(); }))
                .def_readwrite("color", &MeshBasicMaterial::color)
                .def_readwrite("wireframe", &MeshBasicMaterial::wireframe)
                .def_readwrite("wireframe_linewidth", &MeshBasicMaterial::wireframeLinewidth)
                .def_readwrite("reflectivity", &MeshBasicMaterial::reflectivity)
                .def_readwrite("refraction_ratio", &MeshBasicMaterial::refractionRatio)
                .def_readwrite("combine", &MeshBasicMaterial::combine)
                .def_readwrite("map", &MeshBasicMaterial::map)
                .def_readwrite("alpha_map", &MeshBasicMaterial::alphaMap)
                .def_readwrite("ao_map", &MeshBasicMaterial::aoMap)
                .def_readwrite("specular_map", &MeshBasicMaterial::specularMap)
                .def_readwrite("env_map", &MeshBasicMaterial::envMap);

        // ---- MeshStandardMaterial -------------------------------------------
        auto standard = py::class_<MeshStandardMaterial, Material, std::shared_ptr<MeshStandardMaterial>>(m, "MeshStandardMaterial");
        bind_material_base_fields(standard);
        standard.def(py::init([] { return MeshStandardMaterial::create(); }))
                .def_readwrite("color", &MeshStandardMaterial::color)
                .def_readwrite("roughness", &MeshStandardMaterial::roughness)
                .def_readwrite("metalness", &MeshStandardMaterial::metalness)
                .def_readwrite("emissive", &MeshStandardMaterial::emissive)
                .def_readwrite("emissive_intensity", &MeshStandardMaterial::emissiveIntensity)
                .def_readwrite("translucency", &MeshStandardMaterial::translucency)
                .def_readwrite("translucency_color", &MeshStandardMaterial::translucencyColor)
                .def_readwrite("flat_shading", &MeshStandardMaterial::flatShading)
                .def_readwrite("wireframe", &MeshStandardMaterial::wireframe)
                .def_readwrite("wireframe_linewidth", &MeshStandardMaterial::wireframeLinewidth)
                .def_readwrite("normal_scale", &MeshStandardMaterial::normalScale)
                .def_readwrite("env_map_intensity", &MeshStandardMaterial::envMapIntensity)
                .def_readwrite("map", &MeshStandardMaterial::map)
                .def_readwrite("normal_map", &MeshStandardMaterial::normalMap)
                .def_readwrite("roughness_map", &MeshStandardMaterial::roughnessMap)
                .def_readwrite("metalness_map", &MeshStandardMaterial::metalnessMap)
                .def_readwrite("emissive_map", &MeshStandardMaterial::emissiveMap)
                .def_readwrite("ao_map", &MeshStandardMaterial::aoMap)
                .def_readwrite("alpha_map", &MeshStandardMaterial::alphaMap)
                .def_readwrite("bump_map", &MeshStandardMaterial::bumpMap)
                .def_readwrite("displacement_map", &MeshStandardMaterial::displacementMap)
                .def_readwrite("env_map", &MeshStandardMaterial::envMap);

        // ---- MeshPhysicalMaterial -------------------------------------------
        // Extends Standard with the transmissive / clearcoat / attenuation set —
        // the water material (Ocean), glass, etc. Standard's fields come through
        // the pybind base; only the physical-layer scalars are added here.
        // (ior is a property: the C++ setter keeps reflectivity in sync.)
        auto physical = py::class_<MeshPhysicalMaterial, MeshStandardMaterial,
                                   std::shared_ptr<MeshPhysicalMaterial>>(m, "MeshPhysicalMaterial");
        physical.def(py::init([] { return MeshPhysicalMaterial::create(); }))
                .def_property("ior",
                              [](const MeshPhysicalMaterial& self) { return self.ior; },
                              &MeshPhysicalMaterial::setIor)
                .def_readwrite("transmission", &MeshPhysicalMaterial::transmission,
                               "0 = opaque, 1 = fully transmissive (water/glass).")
                .def_readwrite("thickness", &MeshPhysicalMaterial::thickness,
                               "Thin-shell in-medium proxy distance (m) for Beer-Lambert; also "
                               "scales the water body veil (attenuation_color^(2*thickness/attenuation_distance)).")
                .def_readwrite("thin_walled", &MeshPhysicalMaterial::thinWalled,
                               "Surface is a thin shell (ocean plane, lens), not a closed volume.")
                .def_readwrite("attenuation_color", &MeshPhysicalMaterial::attenuationColor,
                               "Beer-Lambert tint per attenuation_distance of travel — the water colour lever.")
                .def_readwrite("attenuation_distance", &MeshPhysicalMaterial::attenuationDistance,
                               "Distance (m) over which attenuation_color is applied once; smaller = murkier.")
                .def_readwrite("clearcoat", &MeshPhysicalMaterial::clearcoat)
                .def_readwrite("clearcoat_roughness", &MeshPhysicalMaterial::clearcoatRoughness)
                // KHR_materials_specular. Unlike clearcoat (which the Vulkan
                // deferred evaluates only for transmission materials), these
                // scale/tint dielectric F0 in the OPAQUE shading path on BOTH
                // backends — the knob for "less specular without more
                // roughness" (porous surfaces: soil, plaster, unfinished wood).
                .def_readwrite("specular_intensity", &MeshPhysicalMaterial::specularIntensity,
                               "Scales dielectric F0 linearly; 0 kills the specular lobe "
                               "(direct and environment) entirely. Default 1.")
                .def_readwrite("specular_color", &MeshPhysicalMaterial::specularColor,
                               "Tints dielectric F0; applied together with specular_intensity.")
                .def_readwrite("dispersion", &MeshPhysicalMaterial::dispersion)
                .def_readwrite("iridescence", &MeshPhysicalMaterial::iridescence)
                .def_readwrite("iridescence_ior", &MeshPhysicalMaterial::iridescenceIOR)
                .def_readwrite("iridescence_thickness_nm", &MeshPhysicalMaterial::iridescenceThicknessNm);

        // ---- MeshPhongMaterial ----------------------------------------------
        auto phong = py::class_<MeshPhongMaterial, Material, std::shared_ptr<MeshPhongMaterial>>(m, "MeshPhongMaterial");
        bind_material_base_fields(phong);
        phong.def(py::init([] { return MeshPhongMaterial::create(); }))
                .def_readwrite("color", &MeshPhongMaterial::color)
                .def_readwrite("specular", &MeshPhongMaterial::specular)
                .def_readwrite("shininess", &MeshPhongMaterial::shininess)
                .def_readwrite("emissive", &MeshPhongMaterial::emissive)
                .def_readwrite("emissive_intensity", &MeshPhongMaterial::emissiveIntensity)
                .def_readwrite("flat_shading", &MeshPhongMaterial::flatShading)
                .def_readwrite("wireframe", &MeshPhongMaterial::wireframe)
                .def_readwrite("reflectivity", &MeshPhongMaterial::reflectivity)
                .def_readwrite("combine", &MeshPhongMaterial::combine)
                .def_readwrite("map", &MeshPhongMaterial::map)
                .def_readwrite("normal_map", &MeshPhongMaterial::normalMap)
                .def_readwrite("specular_map", &MeshPhongMaterial::specularMap)
                .def_readwrite("emissive_map", &MeshPhongMaterial::emissiveMap)
                .def_readwrite("ao_map", &MeshPhongMaterial::aoMap)
                .def_readwrite("alpha_map", &MeshPhongMaterial::alphaMap)
                .def_readwrite("bump_map", &MeshPhongMaterial::bumpMap)
                .def_readwrite("env_map", &MeshPhongMaterial::envMap);

        // ---- MeshLambertMaterial --------------------------------------------
        auto lambert = py::class_<MeshLambertMaterial, Material, std::shared_ptr<MeshLambertMaterial>>(m, "MeshLambertMaterial");
        bind_material_base_fields(lambert);
        lambert.def(py::init([] { return MeshLambertMaterial::create(); }))
                .def_readwrite("color", &MeshLambertMaterial::color)
                .def_readwrite("emissive", &MeshLambertMaterial::emissive)
                .def_readwrite("emissive_intensity", &MeshLambertMaterial::emissiveIntensity)
                .def_readwrite("wireframe", &MeshLambertMaterial::wireframe)
                .def_readwrite("reflectivity", &MeshLambertMaterial::reflectivity)
                .def_readwrite("map", &MeshLambertMaterial::map)
                .def_readwrite("emissive_map", &MeshLambertMaterial::emissiveMap)
                .def_readwrite("ao_map", &MeshLambertMaterial::aoMap)
                .def_readwrite("alpha_map", &MeshLambertMaterial::alphaMap)
                .def_readwrite("env_map", &MeshLambertMaterial::envMap);

        // ---- MeshNormalMaterial ---------------------------------------------
        auto normal = py::class_<MeshNormalMaterial, Material, std::shared_ptr<MeshNormalMaterial>>(m, "MeshNormalMaterial");
        bind_material_base_fields(normal);
        normal.def(py::init([] { return MeshNormalMaterial::create(); }))
                .def_readwrite("flat_shading", &MeshNormalMaterial::flatShading)
                .def_readwrite("wireframe", &MeshNormalMaterial::wireframe)
                .def_readwrite("normal_scale", &MeshNormalMaterial::normalScale)
                .def_readwrite("normal_map", &MeshNormalMaterial::normalMap)
                .def_readwrite("bump_map", &MeshNormalMaterial::bumpMap)
                .def_readwrite("displacement_map", &MeshNormalMaterial::displacementMap);

        // ---- MeshDepthMaterial ----------------------------------------------
        // Renders fragment depth (optionally RGBA-packed). Assign via
        // scene.override_material to produce a depth pass on the GL path.
        auto depth = py::class_<MeshDepthMaterial, Material, std::shared_ptr<MeshDepthMaterial>>(m, "MeshDepthMaterial");
        bind_material_base_fields(depth);
        depth.def(py::init([] { return MeshDepthMaterial::create(); }))
                .def_readwrite("depth_packing", &MeshDepthMaterial::depthPacking)
                .def_readwrite("wireframe", &MeshDepthMaterial::wireframe)
                .def_readwrite("wireframe_linewidth", &MeshDepthMaterial::wireframeLinewidth)
                .def_readwrite("displacement_scale", &MeshDepthMaterial::displacementScale)
                .def_readwrite("displacement_bias", &MeshDepthMaterial::displacementBias)
                .def_readwrite("map", &MeshDepthMaterial::map)
                .def_readwrite("alpha_map", &MeshDepthMaterial::alphaMap)
                .def_readwrite("displacement_map", &MeshDepthMaterial::displacementMap);

        // ---- PointsMaterial --------------------------------------------------
        auto pointsMat = py::class_<PointsMaterial, Material, std::shared_ptr<PointsMaterial>>(m, "PointsMaterial");
        bind_material_base_fields(pointsMat);
        pointsMat.def(py::init([] { return PointsMaterial::create(); }))
                .def_readwrite("color", &PointsMaterial::color)
                .def_readwrite("size", &PointsMaterial::size)
                .def_readwrite("size_attenuation", &PointsMaterial::sizeAttenuation)
                .def_readwrite("map", &PointsMaterial::map)
                .def_readwrite("alpha_map", &PointsMaterial::alphaMap);

        // ---- LineBasicMaterial ----------------------------------------------
        auto lineMat = py::class_<LineBasicMaterial, Material, std::shared_ptr<LineBasicMaterial>>(m, "LineBasicMaterial");
        bind_material_base_fields(lineMat);
        lineMat.def(py::init([] { return LineBasicMaterial::create(); }))
                .def_readwrite("color", &LineBasicMaterial::color)
                .def_readwrite("linewidth", &LineBasicMaterial::linewidth);

        // ---- SpriteMaterial --------------------------------------------------
        auto spriteMat = py::class_<SpriteMaterial, Material, std::shared_ptr<SpriteMaterial>>(m, "SpriteMaterial");
        bind_material_base_fields(spriteMat);
        spriteMat.def(py::init([] { return SpriteMaterial::create(); }))
                .def_readwrite("color", &SpriteMaterial::color)
                .def_readwrite("rotation", &SpriteMaterial::rotation)
                .def_readwrite("size", &SpriteMaterial::size)
                .def_readwrite("size_attenuation", &SpriteMaterial::sizeAttenuation)
                .def_readwrite("map", &SpriteMaterial::map)
                .def_readwrite("alpha_map", &SpriteMaterial::alphaMap);

        // ---- ShadowMaterial --------------------------------------------------
        auto shadow = py::class_<ShadowMaterial, Material, std::shared_ptr<ShadowMaterial>>(m, "ShadowMaterial");
        bind_material_base_fields(shadow);
        shadow.def(py::init([] { return ShadowMaterial::create(); }))
                .def_readwrite("color", &ShadowMaterial::color);
    }

}// namespace threepp_py
