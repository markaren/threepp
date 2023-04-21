// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshMatcapMaterial.js

#ifndef THREEPP_MESHMATCAPMATERIAL_HPP
#define THREEPP_MESHMATCAPMATERIAL_HPP

#include "interfaces.hpp"
#include "Material.hpp"

namespace threepp {

    class MeshMatcapMaterial: public virtual Material,
                              public MaterialWithColor,
                              public MaterialWithMap,
                              public MaterialWithAlphaMap,
                              public MaterialWithBumpMap,
                              public MaterialWithDisplacementMap,
                              public MaterialWithNormalMap,
                              public MaterialWithMatCap,
                              public MaterialWithFlatShading,
                              public MaterialWithDefines {

    public:
        [[nodiscard]] std::string type() const override {

            return "MeshMatcapMaterial";
        }

        [[nodiscard]] std::shared_ptr<Material> clone() const override {
            auto m = create();
            copyInto(m.get());

            m->defines["MATCAP"] = "";

            m->color.copy(color);

            m->matcap = matcap;

            m->map = map;

            m->bumpMap = bumpMap;
            m->bumpScale = bumpScale;

            m->normalMap = normalMap;
            m->normalMapType = normalMapType;
            m->normalScale.copy(normalScale);

            m->displacementMap = displacementMap;
            m->displacementScale = displacementScale;
            m->displacementBias = displacementBias;

            m->alphaMap = alphaMap;

            m->flatShading = flatShading;

            return m;
        }

        static std::shared_ptr<MeshMatcapMaterial> create() {

            return std::shared_ptr<MeshMatcapMaterial>(new MeshMatcapMaterial());
        }

    protected:
        MeshMatcapMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithFlatShading(false),
              MaterialWithBumpMap(1),
              MaterialWithDisplacementMap(1, 0),
              MaterialWithNormalMap(TangentSpaceNormalMap, {1, 1}) {

            this->defines["MATCAP"] = "";
        }
    };

}// namespace threepp

#endif//THREEPP_MESHMATCAPMATERIAL_HPP
