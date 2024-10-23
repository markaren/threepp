// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshBasicMaterial.js

#ifndef THREEPP_MESHBASICMATERIAL_HPP
#define THREEPP_MESHBASICMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"

namespace threepp {

    class MeshBasicMaterial
        : public virtual Material,
          public MaterialWithColor,
          public MaterialWithMap,
          public MaterialWithLightMap,
          public MaterialWithAoMap,
          public MaterialWithSpecularMap,
          public MaterialWithAlphaMap,
          public MaterialWithEnvMap,
          public MaterialWithReflectivity,
          public MaterialWithWireframe,
          public MaterialWithCombine,
          public MaterialWithMorphTargets {

    public:
        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<MeshBasicMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

    protected:
        MeshBasicMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& m) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_MESHBASICMATERIAL_HPP
