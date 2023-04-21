// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshPhongMaterial.js

#ifndef THREEPP_MESHPHONGMATERIAL_HPP
#define THREEPP_MESHPHONGMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshPhongMaterial
        : public virtual Material,
          public MaterialWithColor,
          public MaterialWithSpecular,
          public MaterialWithMap,
          public MaterialWithLightMap,
          public MaterialWithAoMap,
          public MaterialWithEmissive,
          public MaterialWithBumpMap,
          public MaterialWithNormalMap,
          public MaterialWithDisplacementMap,
          public MaterialWithSpecularMap,
          public MaterialWithAlphaMap,
          public MaterialWithEnvMap,
          public MaterialWithReflectivity,
          public MaterialWithWireframe,
          public MaterialWithCombine,
          public MaterialWithFlatShading {

    public:
        [[nodiscard]] std::string type() const override;

        [[nodiscard]] std::shared_ptr<Material> clone() const override;

        static std::shared_ptr<MeshPhongMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

    protected:
        explicit MeshPhongMaterial();

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_MESHPHONGMATERIAL_HPP
