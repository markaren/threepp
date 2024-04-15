// https://github.com/mrdoob/three.js/blob/r129/src/objects/LineLoop.js

#ifndef THREEPP_LINELOOP_HPP
#define THREEPP_LINELOOP_HPP

#include "Line.hpp"

namespace threepp {

    class LineLoop: public Line {

    public:
        LineLoop(const std::shared_ptr<BufferGeometry>& geometry, const std::shared_ptr<Material>& material);

        [[nodiscard]] std::string type() const override;

        static std::shared_ptr<LineLoop> create(
                const std::shared_ptr<BufferGeometry>& geometry = nullptr,
                const std::shared_ptr<Material>& material = nullptr);
    };

}// namespace threepp

#endif//THREEPP_LINELOOP_HPP
