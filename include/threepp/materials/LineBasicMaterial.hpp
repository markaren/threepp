
#ifndef THREEPP_LINEBASICMATERIAL_HPP
#define THREEPP_LINEBASICMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"

namespace threepp {

    class LineBasicMaterial: public virtual Material,
                             public MaterialWithColor,
                             public MaterialWithLineWidth {

    public:
        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<LineBasicMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

    protected:
        LineBasicMaterial();

        std::shared_ptr<Material> createDefault() const override;

        void copyInto(Material& material) const override;

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_LINEBASICMATERIAL_HPP
