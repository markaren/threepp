// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshStandardMaterial.js

#ifndef THREEPP_MESHSTANDARDMATERIAL_HPP
#define THREEPP_MESHSTANDARDMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

#include "threepp/math/Color.hpp"
#include "threepp/math/Vector2.hpp"

#include <optional>
#include <utility>

namespace threepp {

    class MeshStandardMaterial
        : public virtual Material,
          public MaterialWithMap,
          public MaterialWithColor,
          public MaterialWithRoughness,
          public MaterialWithMetalness,
          public MaterialWithNormalMap,
          public MaterialWithEmissive,
          public MaterialWithBumpMap,
          public MaterialWithAoMap,
          public MaterialWithAlphaMap,
          public MaterialWithEnvMap,
          public MaterialWithLightMap,
          public MaterialWithDisplacementMap,
          public MaterialWithWireframe,
          public MaterialWithFlatShading,
          public MaterialWithVertexTangents,
          public MaterialWithRefractionRatio,
          public MaterialWithMorphTargets,
          public MaterialWithDefines {

    public:
        // Typed, order-free, compiler-checked alternative to the stringly-typed
        // create({{"key", value}}) map. A typo or wrong type is a COMPILE error (not a runtime
        // "Unused material values" line or std::bad_variant_access), every field autocompletes,
        // and -- unlike C++20 designated initializers -- the fluent setters chain in ANY order,
        // matching three.js's order-free object literal:
        //
        //     auto m = MeshStandardMaterial::create(
        //         MeshStandardMaterial::Params{}
        //             .roughness(0.4f)   // order-free: roughness before color is fine
        //             .color(0xff0000)
        //             .flatShading(true));
        //
        // Only the fields you set are applied; the rest keep their constructor defaults.
        class Params : public MaterialParams<Params> {
        public:
#define TPP_PARAM(type, field)  \
    Params& field(type v) {     \
        field##_ = std::move(v);\
        return *this;           \
    }
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
#undef TPP_PARAM

        private:
            friend class MeshStandardMaterial;

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
        };

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<MeshStandardMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

        static std::shared_ptr<MeshStandardMaterial> create(const Params& params);

    protected:
        MeshStandardMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& material) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_MESHSTANDARDMATERIAL_HPP
