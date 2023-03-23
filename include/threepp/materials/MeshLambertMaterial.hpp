// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshLambertMaterial.js

#ifndef THREEPP_MESHLAMBERTMATERIAL_HPP
#define THREEPP_MESHLAMBERTMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshLambertMaterial
        : public virtual Material,
          public MaterialWithColor,
          public MaterialWithMap,
          public MaterialWithLightMap,
          public MaterialWithAoMap,
          public MaterialWithEmissive,
          public MaterialWithSpecularMap,
          public MaterialWithAlphaMap,
          public MaterialWithEnvMap,
          public MaterialWithReflectivity,
          public MaterialWithWireframe,
          public MaterialWithCombine {

    public:
        [[nodiscard]] std::string type() const override;

        [[nodiscard]] std::shared_ptr<Material> clone() const override;

        static std::shared_ptr<MeshLambertMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

    protected:
        MeshLambertMaterial();

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_MESHLAMBERTMATERIAL_HPP
