
#ifndef THREEPP_LINEBASICMATERIAL_HPP
#define THREEPP_LINEBASICMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"

namespace threepp {

    class LineBasicMaterial : public virtual Material,
                              public MaterialWithColor,
                              public MaterialWithLineWidth {

    public:
        [[nodiscard]] std::string type() const override {

            return "LineBasicMaterial";
        }

        static std::shared_ptr<LineBasicMaterial> create() {

            return std::shared_ptr<LineBasicMaterial>(new LineBasicMaterial());
        }

    protected:
        LineBasicMaterial()
            : MaterialWithColor(0xffffff),
              MaterialWithLineWidth(1) {}
    };

}// namespace threepp

#endif//THREEPP_LINEBASICMATERIAL_HPP
