// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshMatcapMaterial.js

#ifndef THREEPP_MESHMATCAPMATERIAL_HPP
#define THREEPP_MESHMATCAPMATERIAL_HPP

#include "Material.hpp"

namespace threepp {

    class MeshMatcapMaterial : public virtual Material,
                               public MaterialWithColor,
                               public MaterialWithMap,
                               public MaterialWithAlphaMap,
                               public MaterialWithBumpMap,
                               public MaterialWithDisplacementMap,
                               public MaterialWithNormalMap,
                               public MaterialWithMatCap {

        bool flatShading = false;

        [[nodiscard]] std::string type() const override {

            return "MeshMatcapMaterial";
        }

        static std::shared_ptr<MeshMatcapMaterial> create() {

            return std::shared_ptr<MeshMatcapMaterial>(new MeshMatcapMaterial());
        }

    protected:
        MeshMatcapMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithBumpMap(1),
              MaterialWithDisplacementMap(1, 0),
              MaterialWithNormalMap(TangentSpaceNormalMap, {1, 1}) {
        }
    };

}// namespace threepp

#endif//THREEPP_MESHMATCAPMATERIAL_HPP
