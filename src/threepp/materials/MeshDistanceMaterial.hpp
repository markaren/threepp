// https://github.com/mrdoob/three.js/blob/r129/src/materials/MeshDistanceMaterial.js

#ifndef THREEPP_MESHDISTANCEMATERIAL_HPP
#define THREEPP_MESHDISTANCEMATERIAL_HPP

#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshDistanceMaterial: public virtual Material,
                                public MaterialWithMap,
                                public MaterialWithAlphaMap,
                                public MaterialWithDisplacementMap {

    public:
        Vector3 referencePosition;
        float nearDistance = 1;
        float farDistance = 1000;

        [[nodiscard]] std::string type() const override {

            return "MeshDistanceMaterial";
        }

        static std::shared_ptr<MeshDistanceMaterial> create() {

            return std::shared_ptr<MeshDistanceMaterial>(new MeshDistanceMaterial());
        }

    protected:
        MeshDistanceMaterial(): MaterialWithDisplacementMap(1, 0) {

            this->fog = false;
        }

        std::shared_ptr<Material> createDefault() const override {

            return std::shared_ptr<MeshDistanceMaterial>(new MeshDistanceMaterial());
        }

        void copyInto(Material& material) const override {

            Material::copyInto(material);

            auto m = material.as<MeshDistanceMaterial>();

            m->referencePosition.copy(referencePosition);
            m->nearDistance = nearDistance;
            m->farDistance = farDistance;

            m->map = map;

            m->alphaMap = alphaMap;

            m->displacementMap = displacementMap;
            m->displacementScale = displacementScale;
            m->displacementBias = displacementBias;
        }
    };

}// namespace threepp

#endif//THREEPP_MESHDISTANCEMATERIAL_HPP
