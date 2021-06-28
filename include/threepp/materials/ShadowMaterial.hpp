// https://github.com/mrdoob/three.js/blob/r129/src/materials/ShadowMaterial.js

#ifndef THREEPP_SHADOWMATERIAL_HPP
#define THREEPP_SHADOWMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "interfaces.hpp"

namespace threepp {

    class ShadowMaterial : public MaterialWithColor {

    public:
        Color &getColor() override {
            return color_;
        }

        [[nodiscard]] std::string type() const override {

            return "ShadowMaterial";
        }

        static std::shared_ptr<ShadowMaterial> create() {

            return std::shared_ptr<ShadowMaterial>(new ShadowMaterial());
        }

    protected:
        ShadowMaterial() {

            this->transparent = true;
        };

    private:
        Color color_ = Color(0x000000);
    };

}// namespace threepp

#endif//THREEPP_SHADOWMATERIAL_HPP
