// https://github.com/mrdoob/three.js/blob/r129/src/objects/LineLoop.js

#ifndef THREEPP_LINELOOP_HPP
#define THREEPP_LINELOOP_HPP

#include "Line"

namespace threepp {

    class LineLoop : public Line {

        std::string type() const override {

            return "LineLoop";
        }

        static std::shared_ptr<LineLoop> create(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material) {

            return std::shared_ptr<LineLoop>(new Line(std::move(geometry), std::move((material))));
        }

    protected:
        LineLoop(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material)
            : Line(std::move(geometry), std::move(material)) {}
    };

}// namespace threepp

#endif//THREEPP_LINELOOP_HPP
