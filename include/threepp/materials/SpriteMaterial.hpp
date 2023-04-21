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
        [[nodiscard]] std::string type() const override;

        std::shared_ptr<Material> clone() const override;

        static std::shared_ptr<SpriteMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

    protected:
        SpriteMaterial();

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_SPRITEMATERIAL_HPP
