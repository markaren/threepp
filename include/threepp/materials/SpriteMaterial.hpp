// https://github.com/mrdoob/three.js/blob/r129/src/materials/SpriteMaterial.js

#ifndef THREEPP_SPRITEMATERIAL_HPP
#define THREEPP_SPRITEMATERIAL_HPP

#include <memory>

#include "threepp/materials/materials.hpp"

namespace threepp {

    class SpriteMaterial: public virtual Material,
                          public MaterialWithColor,
                          public MaterialWithRotation,
                          public MaterialWithMap,
                          public MaterialWithAlphaMap,
                          public MaterialWithSize {

    public:
        [[nodiscard]] std::string type() const override {

            return "SpriteMaterial";
        }

        std::shared_ptr<Material> clone() const override {
            auto m = create();
            copyInto(m.get());

            m->color.copy(color);

            m->map = map;

            m->alphaMap = alphaMap;

            m->rotation = rotation;

            m->sizeAttenuation = sizeAttenuation;

            return m;
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
