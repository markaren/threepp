
#ifndef THREEPP_LINEBASICMATERIAL_HPP
#define THREEPP_LINEBASICMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"

namespace threepp {

    class LineBasicMaterial: public MaterialWithColor {

    public:
        Color &getColor() override {
            return color_;
        }

        [[nodiscard]] float getLinewidth() const {
            return linewidth;
        }

        [[nodiscard]] const std::string &getLinecap() const {
            return linecap;
        }

        [[nodiscard]] const std::string &getLinejoin() const {
            return linejoin;
        }

        static std::shared_ptr<LineBasicMaterial> create() {
            return std::shared_ptr<LineBasicMaterial>(new LineBasicMaterial());
        }

    protected:
        LineBasicMaterial() = default;

    private:
        Color color_ = Color(0xffffff);

        float linewidth = 1;
        std::string linecap = "round";
        std::string linejoin = "round";

    };

}

#endif//THREEPP_LINEBASICMATERIAL_HPP
