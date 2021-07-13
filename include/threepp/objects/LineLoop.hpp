// https://github.com/mrdoob/three.js/blob/r129/src/objects/LineLoop.js

#ifndef THREEPP_LINELOOP_HPP
#define THREEPP_LINELOOP_HPP

#include "Line.hpp"

namespace threepp {

    class LineLoop : public Line {

    public:

        static std::shared_ptr<LineLoop> create(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material) {

            return std::shared_ptr<LineLoop>(new LineLoop(std::move(geometry), std::move((material))));
        }

    protected:
        LineLoop(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material)
            : Line(std::move(geometry), std::move(material)) {}
    };

}// namespace threepp

#endif//THREEPP_LINELOOP_HPP
