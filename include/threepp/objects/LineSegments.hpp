// https://github.com/mrdoob/three.js/blob/r129/src/objects/LineSegments.js

#ifndef THREEPP_LINESEGMENTS_HPP
#define THREEPP_LINESEGMENTS_HPP

#include "Line.hpp"

namespace threepp {

    class LineSegments: public Line {

    public:
        [[nodiscard]] std::string type() const override;

        void computeLineDistances() override;

        std::shared_ptr<Object3D> clone(bool recursive = true) override;

        static std::shared_ptr<LineSegments> create(const std::shared_ptr<BufferGeometry>& geometry = nullptr, const std::shared_ptr<Material>& material = nullptr);

    protected:
        LineSegments(const std::shared_ptr<BufferGeometry>& geometry, const std::shared_ptr<Material>& material);
    };

}// namespace threepp

#endif//THREEPP_LINESEGMENTS_HPP
