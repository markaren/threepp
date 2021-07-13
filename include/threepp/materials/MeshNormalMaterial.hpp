// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshNormalMaterial.js

#ifndef THREEPP_MESHNORMALMATERIAL_HPP
#define THREEPP_MESHNORMALMATERIAL_HPP

#include "Material.hpp"
#include "interfaces.hpp"

namespace threepp {

    class MeshNormalMaterial : public virtual Material,
                               public MaterialWithFlatShading {

    public:
        static std::shared_ptr<MeshNormalMaterial> create() {

            return std::shared_ptr<MeshNormalMaterial>(new MeshNormalMaterial());
        }

        [[nodiscard]] std::string type() const override {

            return "MeshNormalMaterial";
        }

    protected:
        MeshNormalMaterial(): MaterialWithFlatShading(false) {

            this->fog = false;
        }
    };

}// namespace threepp

#endif//THREEPP_MESHNORMALMATERIAL_HPP
