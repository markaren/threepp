// https://github.com/mrdoob/three.js/blob/r129/src/materials/ShadowMaterial.js

#ifndef THREEPP_SHADOWMATERIAL_HPP
#define THREEPP_SHADOWMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

#include "threepp/math/Color.hpp"

#include <optional>
#include <utility>

namespace threepp {

    class ShadowMaterial: public virtual Material,
                          public MaterialWithColor {

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
#undef TPP_PARAM

        private:
            friend class ShadowMaterial;

            std::optional<Color> color_;
        };

        [[nodiscard]] std::string type() const override {

            return "ShadowMaterial";
        }

        static std::shared_ptr<ShadowMaterial> create() {

            return std::shared_ptr<ShadowMaterial>(new ShadowMaterial());
        }

        static std::shared_ptr<ShadowMaterial> create(const Params& p) {

            auto m = std::shared_ptr<ShadowMaterial>(new ShadowMaterial());
            p.applyBaseTo(*m);
#define TPP_SET(field) if (p.field##_) m->field = *p.field##_;
#define TPP_TEX(field) if (p.field##_) m->field = p.field##_;
            TPP_SET(color)
#undef TPP_SET
#undef TPP_TEX
            return m;
        }

    protected:
        ShadowMaterial(): MaterialWithColor(0x000000) {

            this->transparent = true;
        }

        std::shared_ptr<Material> createDefault() const override {

            return std::shared_ptr<ShadowMaterial>(new ShadowMaterial());
        }
    };

}// namespace threepp

#endif//THREEPP_SHADOWMATERIAL_HPP
