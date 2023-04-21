
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

        std::shared_ptr<Material> clone() const override;

        static std::shared_ptr<LineBasicMaterial> create(const std::unordered_map<std::string, MaterialValue>& values = {});

    protected:
        LineBasicMaterial();

        bool setValue(const std::string& key, const MaterialValue& value) override;
    };

}// namespace threepp

#endif//THREEPP_LINEBASICMATERIAL_HPP
