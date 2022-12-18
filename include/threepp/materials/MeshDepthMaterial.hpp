// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshDepthMaterial.js

#ifndef THREEPP_MESHDEPTHMATERIAL_HPP
#define THREEPP_MESHDEPTHMATERIAL_HPP

#include "Material.hpp"
#include "interfaces.hpp"

namespace threepp {

    class MeshDepthMaterial
        : public virtual Material,
          public MaterialWithMap,
          public MaterialWithAlphaMap,
          public MaterialWithDisplacementMap,
          public MaterialWithWireframe,
          public MaterialWithDepthPacking {

    public:
        [[nodiscard]] std::string type() const override {

            return "MeshDepthMaterial";
        }

        static std::shared_ptr<MeshDepthMaterial> create() {

            return std::shared_ptr<MeshDepthMaterial>(new MeshDepthMaterial());
        }

    protected:
        MeshDepthMaterial()
            : MaterialWithDepthPacking(BasicDepthPacking),
              MaterialWithDisplacementMap(1, 0),
              MaterialWithWireframe(false, 1) {

            this->fog = false;
        }
    };


}// namespace threepp

#endif//THREEPP_MESHDEPTHMATERIAL_HPP
