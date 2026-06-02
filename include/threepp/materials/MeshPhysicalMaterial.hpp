
#ifndef THREEPP_MESHPHYSICALMATERIAL_HPP
#define THREEPP_MESHPHYSICALMATERIAL_HPP

#include "threepp/materials/MeshStandardMaterial.hpp"

#include <algorithm>

namespace threepp {

    class MeshPhysicalMaterial
        : public MeshStandardMaterial,
          public MaterialWithClearcoat,
          public MaterialWithTransmission,
          public MaterialWithThickness,
          public MaterialWithAttenuation,
          public MaterialWithSheen,
          public MaterialWithIridescence,
          public MaterialWithPbrSpecular,
          public MaterialWithReflectivity {

    public:
        // Typed, order-free, compiler-checked alternative to the stringly-typed
        // create({{"key", value}}) map. A typo or wrong type is a COMPILE error (not a runtime
        // "Unused material values" line or std::bad_variant_access), every field autocompletes,
        // and -- unlike C++20 designated initializers -- the fluent setters chain in ANY order,
        // matching three.js's order-free object literal:
        //
        //     auto m = MeshPhysicalMaterial::create(
        //         MeshPhysicalMaterial::Params{}
        //             .clearcoat(1.f)    // physical-specific field
        //             .roughness(0.4f)   // inherited Standard field
        //             .transmission(0.5f));
        //
        // MeshPhysicalMaterial inherits MeshStandardMaterial, so Params covers BOTH the Standard
        // fields and the Physical-specific fields. Only the fields you set are applied; the rest
        // keep their constructor defaults.
        class Params : public MaterialParams<Params> {
        public:
#define TPP_PARAM(type, field)  \
    Params& field(type v) {     \
        field##_ = std::move(v);\
        return *this;           \
    }
            // --- Standard (inherited) fields ---
            TPP_PARAM(Color, color)
            TPP_PARAM(float, roughness)
            TPP_PARAM(float, metalness)
            TPP_PARAM(std::shared_ptr<Texture>, map)
            TPP_PARAM(std::shared_ptr<Texture>, roughnessMap)
            TPP_PARAM(std::shared_ptr<Texture>, metalnessMap)
            TPP_PARAM(Color, emissive)
            TPP_PARAM(float, emissiveIntensity)
            TPP_PARAM(std::shared_ptr<Texture>, emissiveMap)
            TPP_PARAM(std::shared_ptr<Texture>, normalMap)
            TPP_PARAM(NormalMapType, normalMapType)
            TPP_PARAM(Vector2, normalScale)
            TPP_PARAM(std::shared_ptr<Texture>, bumpMap)
            TPP_PARAM(float, bumpScale)
            TPP_PARAM(std::shared_ptr<Texture>, aoMap)
            TPP_PARAM(float, aoMapIntensity)
            TPP_PARAM(std::shared_ptr<Texture>, displacementMap)
            TPP_PARAM(float, displacementScale)
            TPP_PARAM(float, displacementBias)
            TPP_PARAM(std::shared_ptr<Texture>, alphaMap)
            TPP_PARAM(std::shared_ptr<Texture>, lightMap)
            TPP_PARAM(float, lightMapIntensity)
            TPP_PARAM(std::shared_ptr<Texture>, envMap)
            TPP_PARAM(float, envMapIntensity)
            TPP_PARAM(float, refractionRatio)
            TPP_PARAM(bool, wireframe)
            TPP_PARAM(float, wireframeLinewidth)
            TPP_PARAM(bool, flatShading)
            TPP_PARAM(bool, vertexTangents)
            // --- Physical-specific fields ---
            TPP_PARAM(float, reflectivity)
            TPP_PARAM(float, ior)
            TPP_PARAM(float, clearcoat)
            TPP_PARAM(std::shared_ptr<Texture>, clearcoatMap)
            TPP_PARAM(float, clearcoatRoughness)
            TPP_PARAM(std::shared_ptr<Texture>, clearcoatRoughnessMap)
            TPP_PARAM(std::shared_ptr<Texture>, clearcoatNormalMap)
            TPP_PARAM(float, dispersion)
            TPP_PARAM(float, transmission)
            TPP_PARAM(std::shared_ptr<Texture>, transmissionMap)
            TPP_PARAM(float, thickness)
            TPP_PARAM(std::shared_ptr<Texture>, thicknessMap)
            TPP_PARAM(float, attenuationDistance)
            TPP_PARAM(Color, attenuationColor)
            TPP_PARAM(float, iridescence)
            TPP_PARAM(float, iridescenceIOR)
            TPP_PARAM(float, iridescenceThicknessNm)
#undef TPP_PARAM

        private:
            friend class MeshPhysicalMaterial;

            // --- Standard (inherited) fields ---
            std::optional<Color> color_;
            std::optional<float> roughness_;
            std::optional<float> metalness_;
            std::shared_ptr<Texture> map_;
            std::shared_ptr<Texture> roughnessMap_;
            std::shared_ptr<Texture> metalnessMap_;
            std::optional<Color> emissive_;
            std::optional<float> emissiveIntensity_;
            std::shared_ptr<Texture> emissiveMap_;
            std::shared_ptr<Texture> normalMap_;
            std::optional<NormalMapType> normalMapType_;
            std::optional<Vector2> normalScale_;
            std::shared_ptr<Texture> bumpMap_;
            std::optional<float> bumpScale_;
            std::shared_ptr<Texture> aoMap_;
            std::optional<float> aoMapIntensity_;
            std::shared_ptr<Texture> displacementMap_;
            std::optional<float> displacementScale_;
            std::optional<float> displacementBias_;
            std::shared_ptr<Texture> alphaMap_;
            std::shared_ptr<Texture> lightMap_;
            std::optional<float> lightMapIntensity_;
            std::shared_ptr<Texture> envMap_;
            std::optional<float> envMapIntensity_;
            std::optional<float> refractionRatio_;
            std::optional<bool> wireframe_;
            std::optional<float> wireframeLinewidth_;
            std::optional<bool> flatShading_;
            std::optional<bool> vertexTangents_;
            // --- Physical-specific fields ---
            std::optional<float> reflectivity_;
            std::optional<float> ior_;
            std::optional<float> clearcoat_;
            std::shared_ptr<Texture> clearcoatMap_;
            std::optional<float> clearcoatRoughness_;
            std::shared_ptr<Texture> clearcoatRoughnessMap_;
            std::shared_ptr<Texture> clearcoatNormalMap_;
            std::optional<float> dispersion_;
            std::optional<float> transmission_;
            std::shared_ptr<Texture> transmissionMap_;
            std::optional<float> thickness_;
            std::shared_ptr<Texture> thicknessMap_;
            std::optional<float> attenuationDistance_;
            std::optional<Color> attenuationColor_;
            std::optional<float> iridescence_;
            std::optional<float> iridescenceIOR_;
            std::optional<float> iridescenceThicknessNm_;
        };

        [[nodiscard]] std::string type() const override;

        void setIor(float value) {

            MaterialWithTransmission::ior = std::max(1.f, value);
            reflectivity = std::clamp(2.5f * (value - 1.f) / (value + 1.f), 0.f, 1.f);
        }

        static std::shared_ptr<MeshPhysicalMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

        static std::shared_ptr<MeshPhysicalMaterial> create(const Params& params);

    protected:
        MeshPhysicalMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& material) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_MESHPHYSICALMATERIAL_HPP
