// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshLambertMaterial.js

#ifndef THREEPP_MESHLAMBERTMATERIAL_HPP
#define THREEPP_MESHLAMBERTMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshLambertMaterial : public virtual Material,
                                public MaterialWithColor,
                                public MaterialWithMap {

    public:
        [[nodiscard]] std::string type() const override {

            return "MeshLambertMaterial";
        }

        static std::shared_ptr<MeshLambertMaterial> create() {

            return std::shared_ptr<MeshLambertMaterial>(new MeshLambertMaterial());
        }

    protected:
        MeshLambertMaterial()
            : MaterialWithColor(0xffffff) {}
    };

}// namespace threepp

#endif//THREEPP_MESHLAMBERTMATERIAL_HPP
