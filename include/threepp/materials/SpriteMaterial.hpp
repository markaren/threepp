// https://github.com/mrdoob/three.js/blob/r129/src/materials/SpriteMaterial.js

#ifndef THREEPP_SPRITEMATERIAL_HPP
#define THREEPP_SPRITEMATERIAL_HPP

#include <memory>

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

        static std::shared_ptr<SpriteMaterial> create() {

            return std::shared_ptr<SpriteMaterial>(new SpriteMaterial());
        }

    protected:
        SpriteMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithSize(0, true) {
            transparent = true;
        }
    };

}// namespace threepp

#endif//THREEPP_SPRITEMATERIAL_HPP
