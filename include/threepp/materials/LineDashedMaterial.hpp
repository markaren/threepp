// https://github.com/mrdoob/three.js/blob/r129/src/materials/LineDashedMaterial.js

#ifndef THREEPP_LINEDASHEDMATERIAL_HPP
#define THREEPP_LINEDASHEDMATERIAL_HPP

#include "threepp/materials/LineBasicMaterial.hpp"

#include "threepp/math/Color.hpp"

#include <optional>
#include <utility>

namespace threepp {

    class LineDashedMaterial: public LineBasicMaterial {

    public:
        float dashSize = 3;
        float gapSize = 1;
        float scale = 1;

        // Typed, order-free, compiler-checked alternative to the stringly-typed
        // create({{"key", value}}) map. A typo or wrong type is a COMPILE error (not a runtime
        // "Unused material values" line or std::bad_variant_access), every field autocompletes,
        // and -- unlike C++20 designated initializers -- the fluent setters chain in ANY order.
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
            TPP_PARAM(float, linewidth)
            TPP_PARAM(float, dashSize)
            TPP_PARAM(float, gapSize)
            TPP_PARAM(float, scale)
#undef TPP_PARAM

        private:
            friend class LineDashedMaterial;

            std::optional<Color> color_;
            std::optional<float> linewidth_;
            std::optional<float> dashSize_;
            std::optional<float> gapSize_;
            std::optional<float> scale_;
        };

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<LineDashedMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

        static std::shared_ptr<LineDashedMaterial> create(const Params& params);

    protected:
        LineDashedMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& material) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_LINEDASHEDMATERIAL_HPP
