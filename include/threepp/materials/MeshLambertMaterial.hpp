//

#ifndef THREEPP_MESHLAMBERTMATERIAL_HPP
#define THREEPP_MESHLAMBERTMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshLambertMaterial : public MaterialWithColor {

    public:
        Color &getColor() override {

            return color_;
        }

        [[nodiscard]] std::string type() const override {

            return "MeshLambertMaterial";
        }

        static std::shared_ptr<MeshLambertMaterial> create() {

            return std::shared_ptr<MeshLambertMaterial>(new MeshLambertMaterial());
        }

    protected:
        MeshLambertMaterial() = default;

    private:
        Color color_ = Color(0xffffff);
    };

}// namespace threepp

#endif//THREEPP_MESHLAMBERTMATERIAL_HPP
