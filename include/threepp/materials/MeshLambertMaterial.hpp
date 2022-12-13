// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshLambertMaterial.js

#ifndef THREEPP_MESHLAMBERTMATERIAL_HPP
#define THREEPP_MESHLAMBERTMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshLambertMaterial
        : public virtual Material,
          public MaterialWithColor,
          public MaterialWithMap,
          public MaterialWithLightMap,
          public MaterialWithAoMap,
          public MaterialWithEmissive,
          public MaterialWithSpecularMap,
          public MaterialWithAlphaMap,
          public MaterialWithEnvMap,
          public MaterialWithReflectivity,
          public MaterialWithWireframe,
          public MaterialWithCombine {

    public:
        MeshLambertMaterial(const MeshLambertMaterial &m)
            : MaterialWithColor(m.color),
              MaterialWithWireframe(m.wireframe, m.wireframeLinewidth),
              MaterialWithReflectivity(m.reflectivity, m.refractionRatio),
              MaterialWithLightMap(m.lightMapIntensity),
              MaterialWithEmissive(m.emissive, m.emissiveIntensity),
              MaterialWithAoMap(m.aoMapIntensity),
              MaterialWithCombine(m.combine) {}

        [[nodiscard]] std::string type() const override {

            return "MeshLambertMaterial";
        }

        [[nodiscard]] std::shared_ptr<MeshLambertMaterial> clone() const {

            return std::make_shared<MeshLambertMaterial>(*this);
        }

        static std::shared_ptr<MeshLambertMaterial> create() {

            return std::shared_ptr<MeshLambertMaterial>(new MeshLambertMaterial());
        }

    protected:
        MeshLambertMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithWireframe(false, 1),
              MaterialWithReflectivity(1, 0.98f),
              MaterialWithLightMap(1),
              MaterialWithEmissive(0x000000, 1),
              MaterialWithAoMap(1),
              MaterialWithCombine(MultiplyOperation) {}
    };

}// namespace threepp

#endif//THREEPP_MESHLAMBERTMATERIAL_HPP
