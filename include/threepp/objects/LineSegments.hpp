// https://github.com/mrdoob/three.js/blob/r129/src/objects/LineSegments.js

#ifndef THREEPP_LINESEGMENTS_HPP
#define THREEPP_LINESEGMENTS_HPP

#include "Line.hpp"

namespace threepp {

    class LineSegments: public Line {

    public:
        LineSegments(
                const std::shared_ptr<BufferGeometry>& geometry,
                const std::shared_ptr<Material>& material);

        [[nodiscard]] std::string type() const override;

        void computeLineDistances() override;

        static std::shared_ptr<LineSegments> create(
                const std::shared_ptr<BufferGeometry>& geometry = nullptr,
                const std::shared_ptr<Material>& material = nullptr);

    protected:
        std::shared_ptr<Object3D> createDefault() override;
    };

}// namespace threepp

#endif//THREEPP_LINESEGMENTS_HPP
