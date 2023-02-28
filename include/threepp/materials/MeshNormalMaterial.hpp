// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshNormalMaterial.js

#ifndef THREEPP_MESHNORMALMATERIAL_HPP
#define THREEPP_MESHNORMALMATERIAL_HPP

#include "Material.hpp"
#include "interfaces.hpp"

namespace threepp {

    class MeshNormalMaterial: public virtual Material,
                              public MaterialWithBumpMap,
                              public MaterialWithNormalMap,
                              public MaterialWithDisplacementMap,
                              public MaterialWithWireframe,
                              public MaterialWithFlatShading {

    public:
        static std::shared_ptr<MeshNormalMaterial> create() {

            return std::shared_ptr<MeshNormalMaterial>(new MeshNormalMaterial());
        }

        std::shared_ptr<Material> clone() const override {
            auto m = create();
            copyInto(m.get());

            m->normalMap = normalMap;
            m->normalMapType = normalMapType;
            m->normalScale.copy(normalScale);

            m->displacementMap = displacementMap;
            m->displacementScale = displacementScale;
            m->displacementBias = displacementBias;

            m->wireframe = wireframe;
            m->wireframeLinewidth = wireframeLinewidth;

            m->flatShading = flatShading;

            return m;
        }


        [[nodiscard]] std::string type() const override {

            return "MeshNormalMaterial";
        }

    protected:
        MeshNormalMaterial(): MaterialWithFlatShading(false),
                              MaterialWithWireframe(false, 1),
                              MaterialWithDisplacementMap(1, 0),
                              MaterialWithNormalMap(TangentSpaceNormalMap, {1, 1}),
                              MaterialWithBumpMap(1) {

            this->fog = false;
        }
    };

}// namespace threepp

#endif//THREEPP_MESHNORMALMATERIAL_HPP
