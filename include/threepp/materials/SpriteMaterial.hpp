// https://github.com/mrdoob/three.js/blob/r129/src/materials/SpriteMaterial.js

#ifndef THREEPP_SPRITEMATERIAL_HPP
#define THREEPP_SPRITEMATERIAL_HPP

#include <memory>
#include <optional>
#include <utility>

#include "threepp/materials/materials.hpp"

#include "threepp/math/Color.hpp"

namespace threepp {

    class SpriteMaterial: public virtual Material,
                          public MaterialWithColor,
                          public MaterialWithRotation,
                          public MaterialWithMap,
                          public MaterialWithAlphaMap,
                          public MaterialWithSize {

    public:
        // Typed, order-free, compiler-checked alternative to the stringly-typed
        // create({{"key", value}}) map. A typo or wrong type is a COMPILE error (not a runtime
        // "Unused material values" line or std::bad_variant_access), every field autocompletes,
        // and -- unlike C++20 designated initializers -- the fluent setters chain in ANY order,
        // matching three.js's order-free object literal.
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
            TPP_PARAM(std::shared_ptr<Texture>, map)
            TPP_PARAM(std::shared_ptr<Texture>, alphaMap)
            TPP_PARAM(float, rotation)
            TPP_PARAM(bool, sizeAttenuation)
#undef TPP_PARAM

        private:
            friend class SpriteMaterial;

            std::optional<Color> color_;
            std::shared_ptr<Texture> map_;
            std::shared_ptr<Texture> alphaMap_;
            std::optional<float> rotation_;
            std::optional<bool> sizeAttenuation_;
        };

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<SpriteMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

        static std::shared_ptr<SpriteMaterial> create(const Params& params);

    protected:
        SpriteMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& material) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_SPRITEMATERIAL_HPP
