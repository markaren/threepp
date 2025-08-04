// https://github.com/mrdoob/three.js/blob/r129/src/materials/PointsMaterial.js

#ifndef THREEPP_POINTSMATERIAL_HPP
#define THREEPP_POINTSMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"

namespace threepp {

    class PointsMaterial: public virtual Material,
                          public MaterialWithColor,
                          public MaterialWithMap,
                          public MaterialWithAlphaMap,
                          public MaterialWithSize,
                          public MaterialWithMorphTargets {

    public:
        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<PointsMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

    protected:
        PointsMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& m) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_POINTSMATERIAL_HPP
