// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshPhongMaterial.js

#ifndef THREEPP_MESHPHONGMATERIAL_HPP
#define THREEPP_MESHPHONGMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshPhongMaterial
        : public virtual Material,
          public MaterialWithColor,
          public MaterialWithSpecular,
          public MaterialWithMap,
          public MaterialWithLightMap,
          public MaterialWithAoMap,
          public MaterialWithEmissive,
          public MaterialWithBumpMap,
          public MaterialWithNormalMap,
          public MaterialWithDisplacementMap,
          public MaterialWithSpecularMap,
          public MaterialWithAlphaMap,
          public MaterialWithEnvMap,
          public MaterialWithReflectivity,
          public MaterialWithWireframe,
          public MaterialWithCombine,
          public MaterialWithFlatShading {

    public:
//        MeshPhongMaterial(const MeshPhongMaterial &m)
//            : MaterialWithColor(m.color),
//              MaterialWithCombine(m.combine),
//              MaterialWithFlatShading(m.flatShading),
//              MaterialWithSpecular(m.specular, m.shininess),
//              MaterialWithLightMap(m.lightMapIntensity),
//              MaterialWithAoMap(m.aoMapIntensity),
//              MaterialWithEmissive(m.emissive, m.emissiveIntensity),
//              MaterialWithBumpMap(m.bumpScale),
//              MaterialWithNormalMap(m.normalMapType, m.normalScale),
//              MaterialWithDisplacementMap(m.displacementScale, m.displacementBias),
//              MaterialWithReflectivity(m.reflectivity, m.refractionRatio),
//              MaterialWithWireframe(m.wireframe, m.wireframeLinewidth) {}

        [[nodiscard]] std::string type() const override {

            return "MeshPhongMaterial";
        }

        [[nodiscard]] std::shared_ptr<MeshPhongMaterial> clone() const {

            auto m = std::shared_ptr<MeshPhongMaterial>(new MeshPhongMaterial());

            copyInto(m.get());

            m->color.copy( color );
            m->specular.copy( specular );
            m->shininess = shininess;

            m->map = map;

            m->lightMap = lightMap;
            m->lightMapIntensity = lightMapIntensity;

            m->aoMap = aoMap;
            m->aoMapIntensity = aoMapIntensity;

            m->emissive.copy( emissive );
            m->emissiveMap = emissiveMap;
            m->emissiveIntensity = emissiveIntensity;

            m->bumpMap = bumpMap;
            m->bumpScale = bumpScale;

            m->normalMap = normalMap;
            m->normalMapType = normalMapType;
            m->normalScale.copy( normalScale );

            m->displacementMap = displacementMap;
            m->displacementScale = displacementScale;
            m->displacementBias = displacementBias;

            m->specularMap = specularMap;

            m->alphaMap = alphaMap;

            m->envMap = envMap;
            m->combine = combine;
            m->reflectivity = reflectivity;
            m->refractionRatio = refractionRatio;

            m->wireframe = wireframe;
            m->wireframeLinewidth = wireframeLinewidth;

            m->flatShading = flatShading;

            return m;

        }

        static std::shared_ptr<MeshPhongMaterial> create() {

            return std::shared_ptr<MeshPhongMaterial>(new MeshPhongMaterial());
        }

    protected:
        MeshPhongMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithCombine(MultiplyOperation),
              MaterialWithFlatShading(false),
              MaterialWithSpecular(0x111111, 30),
              MaterialWithLightMap(1),
              MaterialWithAoMap(1),
              MaterialWithEmissive(0x000000, 1),
              MaterialWithBumpMap(1),
              MaterialWithNormalMap(TangentSpaceNormalMap, {1, 1}),
              MaterialWithDisplacementMap(1, 0),
              MaterialWithReflectivity(1, 0.98f),
              MaterialWithWireframe(false, 1) {}
    };

}// namespace threepp

#endif//THREEPP_MESHPHONGMATERIAL_HPP
