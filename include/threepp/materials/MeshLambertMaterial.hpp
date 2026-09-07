// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshLambertMaterial.js

#ifndef THREEPP_MESHLAMBERTMATERIAL_HPP
#define THREEPP_MESHLAMBERTMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

#include "threepp/math/Color.hpp"

#include <optional>
#include <utility>

namespace threepp {

    class MeshLambertMaterial
        : public virtual Material,
          public MaterialWithColor,
          public MaterialWithMap,
          public MaterialWithLightMap,
          public MaterialWithAoMap,
          public MaterialWithEmissive,
          public MaterialWithSpecularMap,
          public MaterialWithAlphaMap,
          public MaterialWithEnvMap,
          public MaterialWithReflectivity,
          public MaterialWithWireframe,
          public MaterialWithCombine,
          public MaterialWithMorphTargets {

    public:
        // Typed, order-free, compiler-checked alternative to the stringly-typed
        // create({{"key", value}}) map. A typo or wrong type is a COMPILE error (not a runtime
        // "Unused material values" line or std::bad_variant_access), every field autocompletes,
        // and -- unlike C++20 designated initializers -- the fluent setters chain in ANY order,
        // matching three.js's order-free object literal:
        //
        //     auto m = MeshLambertMaterial::create(
        //         MeshLambertMaterial::Params{}
        //             .wireframe(true)   // order-free: wireframe before color is fine
        //             .color(0xff0000));
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
            TPP_PARAM(std::shared_ptr<Texture>, map)
            TPP_PARAM(std::shared_ptr<Texture>, aoMap)
            TPP_PARAM(float, aoMapIntensity)
            TPP_PARAM(std::shared_ptr<Texture>, alphaMap)
            TPP_PARAM(std::shared_ptr<Texture>, specularMap)
            TPP_PARAM(std::shared_ptr<Texture>, lightMap)
            TPP_PARAM(float, lightMapIntensity)
            TPP_PARAM(bool, wireframe)
            TPP_PARAM(float, wireframeLinewidth)
            TPP_PARAM(std::shared_ptr<Texture>, envMap)
            TPP_PARAM(CombineOperation, combine)
            TPP_PARAM(float, reflectivity)
            TPP_PARAM(float, refractionRatio)
#undef TPP_PARAM

        private:
            friend class MeshLambertMaterial;

            std::optional<Color> color_;
            std::optional<Color> emissive_;
            std::shared_ptr<Texture> map_;
            std::shared_ptr<Texture> aoMap_;
            std::optional<float> aoMapIntensity_;
            std::shared_ptr<Texture> alphaMap_;
            std::shared_ptr<Texture> specularMap_;
            std::shared_ptr<Texture> lightMap_;
            std::optional<float> lightMapIntensity_;
            std::optional<bool> wireframe_;
            std::optional<float> wireframeLinewidth_;
            std::shared_ptr<Texture> envMap_;
            std::optional<CombineOperation> combine_;
            std::optional<float> reflectivity_;
            std::optional<float> refractionRatio_;
        };

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<MeshLambertMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

        static std::shared_ptr<MeshLambertMaterial> create(const Params& params);

    protected:
        MeshLambertMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& material) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_MESHLAMBERTMATERIAL_HPP
