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
                               public MaterialWithMatCap,
                               public MaterialWithFlatShading,
                               public MaterialWithDefines {

        [[nodiscard]] std::string type() const override {

            return "MeshMatcapMaterial";
        }

        static std::shared_ptr<MeshMatcapMaterial> create() {

            return std::shared_ptr<MeshMatcapMaterial>(new MeshMatcapMaterial());
        }

    protected:
        MeshMatcapMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithFlatShading(false),
              MaterialWithBumpMap(1),
              MaterialWithDisplacementMap(1, 0),
              MaterialWithNormalMap(TangentSpaceNormalMap, {1, 1}) {

            this->defines["MATCAP"] = "";
        }
    };

}// namespace threepp

#endif//THREEPP_MESHMATCAPMATERIAL_HPP
