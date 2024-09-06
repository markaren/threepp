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
        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<MeshNormalMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

    protected:
        MeshNormalMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& material) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_MESHNORMALMATERIAL_HPP
