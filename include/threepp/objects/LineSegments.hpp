// https://github.com/mrdoob/three.js/blob/r129/src/objects/LineSegments.js

#ifndef THREEPP_LINESEGMENTS_HPP
#define THREEPP_LINESEGMENTS_HPP

#include "Line.hpp"

namespace threepp {

    class LineSegments : public Line {

    public:
        void computeLineDistances() override {

            // we assume non-indexed geometry

            if (geometry_->getIndex() == nullptr) {

                const auto positionAttribute = geometry_->getAttribute<float>("position");
                std::vector<float> lineDistances;

                for (int i = 0, l = positionAttribute->count(); i < l; i += 2) {

                    positionAttribute->setFromBufferAttribute(_start, i);
                    positionAttribute->setFromBufferAttribute(_end, i + 1);

                    lineDistances[i] = (i == 0) ? 0 : lineDistances[i - 1];
                    lineDistances[i + 1] = lineDistances[i] + _start.distanceTo(_end);
                }

                geometry_->setAttribute("lineDistance", FloatBufferAttribute::create(lineDistances, 1));

            } else {

                std::cerr << "THREE.LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry." << std::endl;
            }
        }

        static std::shared_ptr<LineSegments> create(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material) {

            return std::shared_ptr<LineSegments>(new LineSegments(std::move(geometry), std::move((material))));
        }

    protected:
        LineSegments(std::shared_ptr<BufferGeometry> geometry, std::shared_ptr<Material> material)
            : Line(std::move(geometry), std::move(material)) {}
    };

}// namespace threepp

#endif//THREEPP_LINESEGMENTS_HPP
