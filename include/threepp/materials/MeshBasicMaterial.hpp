// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshBasicMaterial.js

#ifndef THREEPP_MESHBASICMATERIAL_HPP
#define THREEPP_MESHBASICMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"

#include "threepp/math/Color.hpp"
#include "threepp/textures/Texture.hpp"

#include <optional>

namespace threepp {

    class MeshBasicMaterial
        : public virtual Material,
          public MaterialWithColor,
          public MaterialWithMap,
          public MaterialWithLightMap,
          public MaterialWithAoMap,
          public MaterialWithSpecularMap,
          public MaterialWithAlphaMap,
          public MaterialWithEnvMap,
          public MaterialWithReflectivity,
          public MaterialWithWireframe,
          public MaterialWithCombine {

    public:
        MeshBasicMaterial(const MeshBasicMaterial &m)
            : MaterialWithColor(m.color),
              MaterialWithAoMap(m.aoMapIntensity),
              MaterialWithLightMap(m.lightMapIntensity),
              MaterialWithCombine(m.combine),
              MaterialWithReflectivity(m.reflectivity, m.refractionRatio),
              MaterialWithWireframe(m.wireframe, m.wireframeLinewidth) {
        }

        [[nodiscard]] std::string type() const override {

            return "MeshBasicMaterial";
        }

        [[nodiscard]] std::shared_ptr<MeshBasicMaterial> clone() const {

            return std::make_shared<MeshBasicMaterial>(*this);
        }

        static std::shared_ptr<MeshBasicMaterial> create() {

            return std::shared_ptr<MeshBasicMaterial>(new MeshBasicMaterial());
        }

    protected:
        MeshBasicMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithAoMap(1),
              MaterialWithLightMap(1),
              MaterialWithCombine(MultiplyOperation),
              MaterialWithReflectivity(1, 0.98f),
              MaterialWithWireframe(false, 1) {}
    };

}// namespace threepp

#endif//THREEPP_MESHBASICMATERIAL_HPP
