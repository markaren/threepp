// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshPhongMaterial.js

#ifndef THREEPP_MESHPHONGMATERIAL_HPP
#define THREEPP_MESHPHONGMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshPhongMaterial : public virtual Material,
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
                              public MaterialWithWireframe {

    public:
        int combine = MultiplyOperation;
        bool flatShading = false;

        [[nodiscard]] std::string type() const override {

            return "MeshPhongMaterial";
        }

        static std::shared_ptr<MeshPhongMaterial> create() {
            return std::shared_ptr<MeshPhongMaterial>(new MeshPhongMaterial());
        }

    protected:
        MeshPhongMaterial()
            : MaterialWithColor(0x000000),
              MaterialWithSpecular(0x111111, 30),
              MaterialWithLightMap(1),
              MaterialWithAoMap(1),
              MaterialWithEmissive(0x000000, 1),
              MaterialWithBumpMap(1),
              MaterialWithNormalMap(TangentSpaceNormalMap, {1, 1}),
              MaterialWithDisplacementMap(1, 0),
              MaterialWithReflectivity(1, 0.98f),
              MaterialWithWireframe(false, 1) {}
    };

}// namespace threepp

#endif//THREEPP_MESHPHONGMATERIAL_HPP
