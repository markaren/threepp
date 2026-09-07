// https://github.com/mrdoob/three.js/blob/r129/src/materials/PointsMaterial.js

#ifndef THREEPP_POINTSMATERIAL_HPP
#define THREEPP_POINTSMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"

#include "threepp/math/Color.hpp"

#include <optional>
#include <utility>

namespace threepp {

    class PointsMaterial: public virtual Material,
                          public MaterialWithColor,
                          public MaterialWithMap,
                          public MaterialWithAlphaMap,
                          public MaterialWithSize,
                          public MaterialWithMorphTargets {

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
            TPP_PARAM(float, size)
            TPP_PARAM(bool, sizeAttenuation)
            TPP_PARAM(std::shared_ptr<Texture>, map)
            TPP_PARAM(std::shared_ptr<Texture>, alphaMap)
#undef TPP_PARAM

        private:
            friend class PointsMaterial;

            std::optional<Color> color_;
            std::optional<float> size_;
            std::optional<bool> sizeAttenuation_;
            std::shared_ptr<Texture> map_;
            std::shared_ptr<Texture> alphaMap_;
        };

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<PointsMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

        static std::shared_ptr<PointsMaterial> create(const Params& params);

    protected:
        PointsMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& m) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_POINTSMATERIAL_HPP
