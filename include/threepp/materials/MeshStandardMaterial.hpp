// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshStandardMaterial.js

#ifndef THREEPP_MESHSTANDARDMATERIAL_HPP
#define THREEPP_MESHSTANDARDMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshStandardMaterial: public virtual Material,
                                public MaterialWithMap,
                                public MaterialWithColor,
                                public MaterialWithRoughness,
                                public MaterialWithMetalness,
                                public MaterialWithNormalMap,
                                public MaterialWithEmissive,
                                public MaterialWithBumpMap,
                                public MaterialWithAoMap,
                                public MaterialWithAlphaMap,
                                public MaterialWithEnvMap,
                                public MaterialWithLightMap,
                                public MaterialWithDisplacementMap,
                                public MaterialWithWireframe,
                                public MaterialWithFlatShading,
                                public MaterialWithVertexTangents,
                                public MaterialWithReflectivityRatio,
                                public MaterialWithDefines {

    public:
        [[nodiscard]] std::string type() const override {

            return "MeshStandardMaterial";
        }

        [[nodiscard]] std::shared_ptr<Material> clone() const override {

            auto m = create();
            copyInto(m.get());

            m->defines["STANDARD"] = "";

            m->color.copy(color);
            m->roughness = roughness;
            m->metalness = metalness;

            m->map = map;

            m->lightMap = lightMap;
            m->lightMapIntensity = lightMapIntensity;

            m->aoMap = aoMap;
            m->aoMapIntensity = aoMapIntensity;

            m->emissive.copy(emissive);
            m->emissiveMap = emissiveMap;
            m->emissiveIntensity = emissiveIntensity;

            m->bumpMap = bumpMap;
            m->bumpScale = bumpScale;

            m->normalMap = normalMap;
            m->normalMapType = normalMapType;
            m->normalScale.copy(normalScale);

            m->displacementMap = displacementMap;
            m->displacementScale = displacementScale;
            m->displacementBias = displacementBias;

            m->roughnessMap = roughnessMap;

            m->metallnessMap = metallnessMap;

            m->alphaMap = alphaMap;

            m->envMap = envMap;
            m->envMapIntensity = envMapIntensity;

            m->refractionRatio = refractionRatio;

            m->wireframe = wireframe;
            m->wireframeLinewidth = wireframeLinewidth;

            m->flatShading = flatShading;

            m->vertexTangents = vertexTangents;

            return m;
        }

        static std::shared_ptr<MeshStandardMaterial> create() {

            return std::shared_ptr<MeshStandardMaterial>(new MeshStandardMaterial());
        }

    protected:
        MeshStandardMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithWireframe(false, 1),
              MaterialWithRoughness(1),
              MaterialWithMetalness(0),
              MaterialWithLightMap(1),
              MaterialWithAoMap(1),
              MaterialWithEmissive(0x000000, 1),
              MaterialWithBumpMap(1),
              MaterialWithEnvMap(1.f),
              MaterialWithDisplacementMap(1, 0),
              MaterialWithReflectivityRatio(0.98),
              MaterialWithNormalMap(TangentSpaceNormalMap, {1, 1}),
              MaterialWithVertexTangents(false),
              MaterialWithFlatShading(false) {

            defines["STANDARD"] = "";
        }
    };

}// namespace threepp

#endif//THREEPP_MESHSTANDARDMATERIAL_HPP
