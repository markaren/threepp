
#ifndef THREEPP_MESHPHYSICALMATERIAL_HPP
#define THREEPP_MESHPHYSICALMATERIAL_HPP

#include "threepp/materials/MeshStandardMaterial.hpp"

#include <algorithm>

namespace threepp {

    class MeshPhysicalMaterial
        : public MeshStandardMaterial,
          public MaterialWithClearcoat,
          public MaterialWithTransmission,
          public MaterialWithThickness,
          public MaterialWithAttenuation,
          public MaterialWithSheen,
          public MaterialWithReflectivity {

    public:
        [[nodiscard]] std::string type() const override;

        [[nodiscard]] float ior() const {

            return (1.f + 0.4f * reflectivity) / (1.f - 0.4f * reflectivity);
        }

        void setIor(float value) {

            reflectivity = std::clamp(2.5f * (value - 1.f) / (value + 1.f), 0.f, 1.f);
        }

        static std::shared_ptr<MeshPhysicalMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

    protected:
        MeshPhysicalMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& material) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_MESHPHYSICALMATERIAL_HPP
