//

#ifndef THREEPP_MESHTOONMATERIAL_HPP
#define THREEPP_MESHTOONMATERIAL_HPP

#include "interfaces.hpp"
#include "threepp/materials/Material.hpp"

namespace threepp {

    class MeshToonMaterial : public MaterialWithColor {

    public:
        Color &getColor() override {

            return color_;
        }
        [[nodiscard]] std::string type() const override {

            return "MeshToonMaterial";
        }

        static std::shared_ptr<MeshToonMaterial> create() {
            return std::shared_ptr<MeshToonMaterial>(new MeshToonMaterial());
        }

    protected:
        MeshToonMaterial() = default;

    private:
        Color color_ = Color(0xffffff);
    };

}// namespace threepp

#endif//THREEPP_MESHTOONMATERIAL_HPP
