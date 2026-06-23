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
                .value("None", Blending::None)
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
        // Abstract; never instantiated. Registered only so concrete materials
        // can declare it as a pybind base (for isinstance and the as_material /
        // material_to_py shared_ptr<Material> bridge).
        py::class_<Material, std::shared_ptr<Material>>(m, "Material");

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
