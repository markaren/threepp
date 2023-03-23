// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshStandardMaterial.js

#ifndef THREEPP_MESHSTANDARDMATERIAL_HPP
#define THREEPP_MESHSTANDARDMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshStandardMaterial: public virtual Material,
                                public MaterialWithMap,
                                public MaterialWithColor,
                                public MaterialWithRoughness,
                                public MaterialWithMetalness,
                                public MaterialWithNormalMap,
                                public MaterialWithEmissive,
                                public MaterialWithBumpMap,
                                public MaterialWithAoMap,
                                public MaterialWithAlphaMap,
                                public MaterialWithEnvMap,
                                public MaterialWithLightMap,
                                public MaterialWithDisplacementMap,
                                public MaterialWithWireframe,
                                public MaterialWithFlatShading,
                                public MaterialWithVertexTangents,
                                public MaterialWithReflectivityRatio,
                                public MaterialWithDefines {

    public:
        [[nodiscard]] std::string type() const override;

        [[nodiscard]] std::shared_ptr<Material> clone() const override;

        static std::shared_ptr<MeshStandardMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

    protected:
        MeshStandardMaterial();

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_MESHSTANDARDMATERIAL_HPP
