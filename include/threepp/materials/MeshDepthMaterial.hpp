// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshDepthMaterial.js

#ifndef THREEPP_MESHDEPTHMATERIAL_HPP
#define THREEPP_MESHDEPTHMATERIAL_HPP

#include "Material.hpp"
#include "interfaces.hpp"

namespace threepp {

    class MeshDepthMaterial
        : public virtual Material,
          public MaterialWithMap,
          public MaterialWithAlphaMap,
          public MaterialWithDisplacementMap,
          public MaterialWithWireframe,
          public MaterialWithDepthPacking {

    public:
        [[nodiscard]] std::string type() const override {

            return "MeshDepthMaterial";
        }

        void copyInto(Material& material) const override {

            Material::copyInto(material);

            auto m = material.as<MeshDepthMaterial>();

            m->depthPacking = depthPacking;

            m->map = map;

            m->alphaMap = alphaMap;

            m->displacementMap = displacementMap;
            m->displacementScale = displacementScale;
            m->displacementBias = displacementBias;

            m->wireframe = false;
            m->wireframeLinewidth = 1;

            m->fog = false;
        }

        static std::shared_ptr<MeshDepthMaterial> create() {

            return std::shared_ptr<MeshDepthMaterial>(new MeshDepthMaterial());
        }

    protected:
        MeshDepthMaterial()
            : MaterialWithDepthPacking(DepthPacking::Basic),
              MaterialWithDisplacementMap(1, 0),
              MaterialWithWireframe(false, 1) {

            this->fog = false;
        }

        std::shared_ptr<Material> createDefault() const override {

            return std::shared_ptr<MeshDepthMaterial>(new MeshDepthMaterial());
        }
    };


}// namespace threepp

#endif//THREEPP_MESHDEPTHMATERIAL_HPP
