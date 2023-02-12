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
        [[nodiscard]] std::string type() const override {

            return "MeshLambertMaterial";
        }

        [[nodiscard]] std::shared_ptr<MeshLambertMaterial> clone() const {

            auto m = std::shared_ptr<MeshLambertMaterial>(new MeshLambertMaterial());
            copyInto(m.get());

            m->color.copy(color);

            m->map = map;

            m->lightMap = lightMap;
            m->lightMapIntensity = lightMapIntensity;

            m->aoMap = aoMap;
            m->aoMapIntensity = aoMapIntensity;

            m->emissive.copy(emissive);
            m->emissiveMap = emissiveMap;
            m->emissiveIntensity = emissiveIntensity;

            m->specularMap = specularMap;

            m->alphaMap = alphaMap;

            m->envMap = envMap;
            m->combine = combine;
            m->reflectivity = reflectivity;
            m->refractionRatio = refractionRatio;

            m->wireframe = wireframe;
            m->wireframeLinewidth = wireframeLinewidth;

            return m;
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
