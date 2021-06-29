// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshStandardMaterial.js

#ifndef THREEPP_MESHSTANDARDMATERIAL_HPP
#define THREEPP_MESHSTANDARDMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshStandardMaterial : public virtual Material,
                                 public MaterialWithColor {

    public:
        [[nodiscard]] std::string type() const override {

            return "MeshStandardMaterial";
        }

        static std::shared_ptr<MeshStandardMaterial> create() {
            return std::shared_ptr<MeshStandardMaterial>(new MeshStandardMaterial());
        }

    protected:
        MeshStandardMaterial() : MaterialWithColor(0xffffff) {}
    };

}// namespace threepp

#endif//THREEPP_MESHSTANDARDMATERIAL_HPP
