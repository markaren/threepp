// https://github.com/mrdoob/three.js/blob/r129/src/materials/PointsMaterial.js

#ifndef THREEPP_POINTSMATERIAL_HPP
#define THREEPP_POINTSMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"

#include "threepp/textures/Texture.hpp"

namespace threepp {

    class PointsMaterial : public virtual Material,
                           public MaterialWithColor,
                           public MaterialWithMap,
                           public MaterialWithAlphaMap,
                           public MaterialWithSize {

    public:
        [[nodiscard]] std::string type() const override {

            return "PointsMaterial";
        }

        static std::shared_ptr<PointsMaterial> create() {

            return std::shared_ptr<PointsMaterial>(new PointsMaterial());
        }

    protected:
        PointsMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithSize(1, true) {}

    };

}// namespace threepp

#endif//THREEPP_POINTSMATERIAL_HPP
