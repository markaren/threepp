
#ifndef THREEPP_LINEBASICMATERIAL_HPP
#define THREEPP_LINEBASICMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"

#include "threepp/math/Color.hpp"

#include <optional>
#include <utility>

namespace threepp {

    class LineBasicMaterial: public virtual Material,
                             public MaterialWithColor,
                             public MaterialWithLineWidth {

    public:
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
#undef TPP_PARAM

        private:
            friend class LineBasicMaterial;

            std::optional<Color> color_;
            std::optional<float> linewidth_;
        };

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<LineBasicMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

        static std::shared_ptr<LineBasicMaterial> create(const Params& params);

    protected:
        LineBasicMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& material) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_LINEBASICMATERIAL_HPP
