
#include "threepp/objects/LineSegments.hpp"

#include <iostream>

using namespace threepp;

void LineSegments::computeLineDistances() {

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
