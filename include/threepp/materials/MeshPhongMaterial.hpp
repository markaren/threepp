// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshPhongMaterial.js

#ifndef THREEPP_MESHPHONGMATERIAL_HPP
#define THREEPP_MESHPHONGMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

#include "threepp/math/Color.hpp"
#include "threepp/math/Vector2.hpp"

#include <optional>
#include <utility>

namespace threepp {

    class MeshPhongMaterial
        : public virtual Material,
          public MaterialWithColor,
          public MaterialWithSpecular,
          public MaterialWithMap,
          public MaterialWithLightMap,
          public MaterialWithAoMap,
          public MaterialWithEmissive,
          public MaterialWithBumpMap,
          public MaterialWithNormalMap,
          public MaterialWithDisplacementMap,
          public MaterialWithSpecularMap,
          public MaterialWithAlphaMap,
          public MaterialWithEnvMap,
          public MaterialWithReflectivity,
          public MaterialWithWireframe,
          public MaterialWithCombine,
          public MaterialWithFlatShading,
          public MaterialWithMorphTargets {

    public:
        // Typed, order-free, compiler-checked alternative to the stringly-typed
        // create({{"key", value}}) map. A typo or wrong type is a COMPILE error (not a runtime
        // "Unused material values" line or std::bad_variant_access), every field autocompletes,
        // and -- unlike C++20 designated initializers -- the fluent setters chain in ANY order,
        // matching three.js's order-free object literal:
        //
        //     auto m = MeshPhongMaterial::create(
        //         MeshPhongMaterial::Params{}
        //             .shininess(30.f)   // order-free: shininess before color is fine
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
            TPP_PARAM(Color, emissive)
            TPP_PARAM(float, emissiveIntensity)
            TPP_PARAM(std::shared_ptr<Texture>, emissiveMap)
            TPP_PARAM(bool, wireframe)
            TPP_PARAM(float, wireframeLinewidth)
            TPP_PARAM(bool, flatShading)
            TPP_PARAM(std::shared_ptr<Texture>, map)
            TPP_PARAM(std::shared_ptr<Texture>, aoMap)
            TPP_PARAM(float, aoMapIntensity)
            TPP_PARAM(std::shared_ptr<Texture>, bumpMap)
            TPP_PARAM(float, bumpScale)
            TPP_PARAM(std::shared_ptr<Texture>, lightMap)
            TPP_PARAM(float, lightMapIntensity)
            TPP_PARAM(std::shared_ptr<Texture>, normalMap)
            TPP_PARAM(NormalMapType, normalMapType)
            TPP_PARAM(std::shared_ptr<Texture>, alphaMap)
            TPP_PARAM(std::shared_ptr<Texture>, specularMap)
            TPP_PARAM(Color, specular)
            TPP_PARAM(std::shared_ptr<Texture>, displacementMap)
            TPP_PARAM(float, displacementBias)
            TPP_PARAM(float, displacementScale)
            TPP_PARAM(float, shininess)
            TPP_PARAM(std::shared_ptr<Texture>, envMap)
            TPP_PARAM(CombineOperation, combine)
            TPP_PARAM(float, reflectivity)
            TPP_PARAM(float, refractionRatio)
            TPP_PARAM(Vector2, normalScale)
#undef TPP_PARAM

        private:
            friend class MeshPhongMaterial;

            std::optional<Color> color_;
            std::optional<Color> emissive_;
            std::optional<float> emissiveIntensity_;
            std::shared_ptr<Texture> emissiveMap_;
            std::optional<bool> wireframe_;
            std::optional<float> wireframeLinewidth_;
            std::optional<bool> flatShading_;
            std::shared_ptr<Texture> map_;
            std::shared_ptr<Texture> aoMap_;
            std::optional<float> aoMapIntensity_;
            std::shared_ptr<Texture> bumpMap_;
            std::optional<float> bumpScale_;
            std::shared_ptr<Texture> lightMap_;
            std::optional<float> lightMapIntensity_;
            std::shared_ptr<Texture> normalMap_;
            std::optional<NormalMapType> normalMapType_;
            std::shared_ptr<Texture> alphaMap_;
            std::shared_ptr<Texture> specularMap_;
            std::optional<Color> specular_;
            std::shared_ptr<Texture> displacementMap_;
            std::optional<float> displacementBias_;
            std::optional<float> displacementScale_;
            std::optional<float> shininess_;
            std::shared_ptr<Texture> envMap_;
            std::optional<CombineOperation> combine_;
            std::optional<float> reflectivity_;
            std::optional<float> refractionRatio_;
            std::optional<Vector2> normalScale_;
        };

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<MeshPhongMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

        static std::shared_ptr<MeshPhongMaterial> create(const Params& params);

    protected:
        explicit MeshPhongMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& material) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_MESHPHONGMATERIAL_HPP
