// https://github.com/mrdoob/three.js/blob/r129/src/materials/ShadowMaterial.js

#ifndef THREEPP_SHADOWMATERIAL_HPP
#define THREEPP_SHADOWMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class ShadowMaterial: public virtual Material,
                          public MaterialWithColor {

    public:
        [[nodiscard]] std::string type() const override {

            return "ShadowMaterial";
        }

        static std::shared_ptr<ShadowMaterial> create() {

            return std::shared_ptr<ShadowMaterial>(new ShadowMaterial());
        }

    protected:
        ShadowMaterial(): MaterialWithColor(0x000000) {

            this->transparent = true;
        }
    };

}// namespace threepp

#endif//THREEPP_SHADOWMATERIAL_HPP
