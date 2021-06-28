//

#ifndef THREEPP_MESHSTANDARDMATERIAL_HPP
#define THREEPP_MESHSTANDARDMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "interfaces.hpp"

namespace threepp {

    class MeshStandardMaterial : public Material {

    public:
        [[nodiscard]] std::string type() const override {

            return "MeshStandardMaterial";
        }

        static std::shared_ptr<MeshStandardMaterial> create() {
            return std::shared_ptr<MeshStandardMaterial>(new MeshStandardMaterial());
        }

    protected:
        MeshStandardMaterial() = default;
    };

}// namespace threepp

#endif//THREEPP_MESHSTANDARDMATERIAL_HPP
