// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshToonMaterial.js

#ifndef THREEPP_MESHTOONMATERIAL_HPP
#define THREEPP_MESHTOONMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshToonMaterial : public virtual Material,
                             public MaterialWithColor {

    public:
        [[nodiscard]] std::string type() const override {

            return "MeshToonMaterial";
        }

        static std::shared_ptr<MeshToonMaterial> create() {
            return std::shared_ptr<MeshToonMaterial>(new MeshToonMaterial());
        }

    protected:
        MeshToonMaterial()
            : MaterialWithColor(0xffffff) {}
    };

}// namespace threepp

#endif//THREEPP_MESHTOONMATERIAL_HPP
