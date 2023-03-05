// https://github.com/mrdoob/three.js/blob/r129/src/objects/LineLoop.js

#ifndef THREEPP_LINELOOP_HPP
#define THREEPP_LINELOOP_HPP

#include "Line.hpp"

namespace threepp {

    class LineLoop: public Line {

    public:

        [[nodiscard]] std::string type() const override {

            return "LineLoop";
        }

        std::shared_ptr<Object3D> clone(bool recursive = true) override {
            auto clone = create();
            clone->copy(*this, recursive);

            return clone;
        }

        static std::shared_ptr<LineLoop> create(const std::shared_ptr<BufferGeometry>& geometry = nullptr, const std::shared_ptr<Material>& material = nullptr) {

            return std::shared_ptr<LineLoop>(new LineLoop(geometry, (material)));
        }

    protected:
        LineLoop(const std::shared_ptr<BufferGeometry>& geometry, const std::shared_ptr<Material>& material)
            : Line(geometry, material) {}
    };

}// namespace threepp

#endif//THREEPP_LINELOOP_HPP
