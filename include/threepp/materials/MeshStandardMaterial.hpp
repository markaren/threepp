// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshStandardMaterial.js

#ifndef THREEPP_MESHSTANDARDMATERIAL_HPP
#define THREEPP_MESHSTANDARDMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshStandardMaterial : public virtual Material,
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
                                 public MaterialWithDefines {

    public:
        [[nodiscard]] std::string type() const override {

            return "MeshStandardMaterial";
        }

        static std::shared_ptr<MeshStandardMaterial> create() {

            return std::shared_ptr<MeshStandardMaterial>(new MeshStandardMaterial());
        }

    protected:
        MeshStandardMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithWireframe(false, 1),
              MaterialWithRoughness(1),
              MaterialWithMetalness(0),
              MaterialWithLightMap(1),
              MaterialWithAoMap(1),
              MaterialWithEmissive(0x000000, 1),
              MaterialWithBumpMap(1),
              MaterialWithEnvMap(1.f),
              MaterialWithDisplacementMap(1, 0),
              MaterialWithNormalMap(TangentSpaceNormalMap, {1, 1}),
              MaterialWithFlatShading(false) {

            defines["STANDARD"] = "";
        }
    };

}// namespace threepp

#endif//THREEPP_MESHSTANDARDMATERIAL_HPP
