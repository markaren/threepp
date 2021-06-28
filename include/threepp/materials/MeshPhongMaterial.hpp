//

#ifndef THREEPP_MESHPHONGMATERIAL_HPP
#define THREEPP_MESHPHONGMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "interfaces.hpp"

namespace threepp {

    class MeshPhongMaterial: public MaterialWithColor {

    public:
        Color &getColor() override {

            return color_;
        }

        [[nodiscard]] std::string type() const override {

            return "MeshPhongMaterial";
        }

        static std::shared_ptr<MeshPhongMaterial> create() {
            return std::shared_ptr<MeshPhongMaterial>(new MeshPhongMaterial());
        }

    protected:
        MeshPhongMaterial() = default;

    private:
        Color color_ = Color(0x000000);
    };

}

#endif//THREEPP_MESHPHONGMATERIAL_HPP
