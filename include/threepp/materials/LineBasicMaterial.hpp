
#ifndef THREEPP_LINEBASICMATERIAL_HPP
#define THREEPP_LINEBASICMATERIAL_HPP

#include "threepp/materials/Material.hpp"
#include "threepp/materials/interfaces.hpp"

namespace threepp {

    class LineBasicMaterial: public virtual Material,
                             public MaterialWithColor,
                             public MaterialWithLineWidth {

    public:
        [[nodiscard]] std::string type() const override {

            return "LineBasicMaterial";
        }

        std::shared_ptr<Material> clone() const override {
            auto m = create();
            copyInto(m.get());

            m->color.copy(color);

            m->linewidth = linewidth;

            return m;
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
