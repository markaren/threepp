// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshToonMaterial.js

#ifndef THREEPP_MESHTOONMATERIAL_HPP
#define THREEPP_MESHTOONMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshToonMaterial: public virtual Material,
                            public MaterialWithColor,
                            public MaterialWithMap,
                            public MaterialWithAlphaMap,
                            public MaterialWithGradientMap,
                            public MaterialWithBumpMap,
                            public MaterialWithNormalMap,
                            public MaterialWithLightMap,
                            public MaterialWithAoMap,
                            public MaterialWithDisplacementMap,
                            public MaterialWithEmissive,
                            public MaterialWithWireframe,
                            public MaterialWithDefines {

    public:
        [[nodiscard]] std::string type() const override {

            return "MeshToonMaterial";
        }

        void copyInto(Material& material) const override {

            Material::copyInto(material);

            auto m = material.as<MeshToonMaterial>();

            m->color.copy(color);

            m->map = map;
            m->gradientMap = gradientMap;

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

            m->alphaMap = alphaMap;

            m->wireframe = wireframe;
            m->wireframeLinewidth = wireframeLinewidth;
        }

        static std::shared_ptr<MeshToonMaterial> create() {

            return std::shared_ptr<MeshToonMaterial>(new MeshToonMaterial());
        }

    protected:
        MeshToonMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithAoMap(1),
              MaterialWithBumpMap(1),
              MaterialWithLightMap(1),
              MaterialWithDisplacementMap(1, 0),
              MaterialWithEmissive(0x000000, 1),
              MaterialWithWireframe(false, 1),
              MaterialWithNormalMap(NormalMapType::TangentSpace, {1, 1}) {

            this->defines["TOON"] = "";
        }

        std::shared_ptr<Material> createDefault() const override {

            return std::shared_ptr<MeshToonMaterial>(new MeshToonMaterial());
        }
    };

}// namespace threepp

#endif//THREEPP_MESHTOONMATERIAL_HPP
