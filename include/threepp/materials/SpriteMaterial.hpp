// https://github.com/mrdoob/three.js/blob/r129/src/materials/SpriteMaterial.js

#ifndef THREEPP_SPRITEMATERIAL_HPP
#define THREEPP_SPRITEMATERIAL_HPP

#include "threepp/materials/materials.hpp"

namespace threepp {

    class SpriteMaterial : public virtual Material,
                           public MaterialWithColor,
                           public MaterialWithRotation,
                           public MaterialWithMap,
                           public MaterialWithAlphaMap,
                           public MaterialWithSize {

    public:
        [[nodiscard]] std::string type() const override {

            return "SpriteMaterial";
        }

    private:
        SpriteMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithSize(0, true) {
            transparent = true;
        }
    };

}// namespace threepp

#endif//THREEPP_SPRITEMATERIAL_HPP